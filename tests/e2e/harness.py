"""Process-level harness shared by the end-to-end tier (issue #466, epic #465).

Everything here exists so a test can start the *real* ``windbreak`` artifact,
observe it from outside, and be guaranteed not to leak a process when it fails.

Three rules shape this module, each one a direct response to how the suite got
into the state epic #465 describes:

* **Nothing is asserted from inside the process under test.** Callers get an
  exit code, a log file, a socket or a ledger row -- the same evidence an
  operator would have.
* **Teardown is not the test's responsibility.** :class:`ProcessLauncher` owns
  every process it starts and reaps them all, in reverse order, even when the
  test body raises, and even when one of them refuses to die -- a child that
  survives ``SIGKILL`` is reported after its siblings are reaped, never
  instead of them. Children are started in their own session so a whole
  process group can be signalled rather than just its leader.
* **Waiting is always bounded and always on a real condition.** See
  :func:`wait_until`. There is no unconditional sleep in this tier: a fixed
  sleep is either flaky or slow, and in a merge-gating job it is eventually
  both.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import IO, TYPE_CHECKING

import pytest

from windbreak.ledger.store import SqliteLedgerStore

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from windbreak.ledger.store import LedgerRecord

#: Seconds a ``SIGTERM``-ed child gets to exit before it is killed outright.
TERMINATE_GRACE_SECONDS = 10.0

#: Default ceiling on a one-shot CLI invocation.
DEFAULT_TIMEOUT_SECONDS = 120.0

#: Cadence at which :func:`wait_until` re-evaluates its predicate.
POLL_INTERVAL_SECONDS = 0.05

#: Ceiling on the container-runtime availability probe.
RUNTIME_PROBE_TIMEOUT_SECONDS = 20.0

#: Skip reason used when the docker CLI is not installed at all.
DOCKER_MISSING_REASON = "docker runtime unavailable: no `docker` CLI on PATH"

#: Skip reason used when the docker CLI exists but its daemon is unreachable.
DOCKER_DAEMON_REASON = (
    "docker runtime unavailable: `docker` CLI present but the daemon is "
    "unreachable (`docker info` failed)"
)

#: Skip reason used when `systemd-analyze` is not installed.
SYSTEMD_ANALYZE_MISSING_REASON = (
    "systemd runtime unavailable: no `systemd-analyze` on PATH"
)

#: Skip reason used when systemd is not the running init system.
SYSTEMD_NOT_RUNNING_REASON = (
    "systemd runtime unavailable: systemd is not running (/run/systemd/system absent)"
)

#: Marker file systemd creates when it is the running init system.
_SYSTEMD_RUNTIME_MARKER = Path("/run/systemd/system")

#: Environment variable that converts a missing-runtime SKIP into a FAILURE.
#:
#: The container tier is a REQUIRED check under branch protection, and a
#: required check that skips reports success -- the CI-level form of a corpus
#: scan asserting over zero hits. On a developer machine a skip is right: not
#: everyone has a docker daemon, and a tier that errored there would just be
#: deleted. In CI the runtime is provisioned deliberately, so a skip means the
#: provisioning broke and the gate silently stopped gating. Setting this to
#: ``1`` (the CI job does, in `.github/workflows/ci.yml`) makes that
#: impossible: every gate in the tier fails loudly instead of skipping.
#:
#: THAT LAST SENTENCE IS A UNIVERSAL, AND IT IS ONLY TRUE WHILE EVERY RUNTIME
#: PROBE IS ROUTED THROUGH :func:`require_runtime`. It was false when first
#: written: `tests/deploy/test_deployment_cli_contract.py` probed systemd and
#: called `pytest.skip` on the answer itself, so its four `systemd-analyze
#: verify` assertions over the shipped unit files would have vanished silently
#: on a runner that lost systemd -- while this comment promised they could
#: not. One opted-out call site is enough to make the promise a lie, so the
#: quantifier is enforced rather than asserted here:
#: `tests/e2e/test_tier_selection_contract.py` fails if any module in the tier
#: probes a runtime without importing this gate.
REQUIRE_RUNTIME_ENV_VAR = "WINDBREAK_E2E_REQUIRE_RUNTIME"

#: Value :data:`REQUIRE_RUNTIME_ENV_VAR` must hold to arm the fail-closed mode.
REQUIRE_RUNTIME_ENABLED_VALUE = "1"


@dataclass(frozen=True)
class RunRoot:
    """An isolated filesystem root for one end-to-end run.

    Bundles the paths the CLI's flags expect so a test names them once. Note
    that ``state_dir`` is deliberately separate from the ``run`` flags: the
    kill/ack/rearm commands take ``--state-dir`` while ``windbreak run`` does
    not, and conflating them has already produced wrong assumptions.

    Attributes:
        root: The run's top-level directory.
        ledger_path: Value for ``--ledger-path``.
        state_dir: Value for the safety commands' ``--state-dir``.
        report_dir: Value for ``--report-dir``.
        anchor_path: Value for ``anchor``/``verify``'s ``--anchor-path``.
        log_dir: Directory holding each spawned process's captured streams.
    """

    root: Path
    ledger_path: Path
    state_dir: Path
    report_dir: Path
    anchor_path: Path
    log_dir: Path

    @classmethod
    def create(cls, base: Path) -> RunRoot:
        """Create the directory skeleton for a run root under ``base``.

        Args:
            base: Directory to build the run root inside, typically ``tmp_path``.

        Returns:
            A :class:`RunRoot` whose directories already exist.
        """
        root = base / "run"
        state_dir = root / "state"
        report_dir = root / "reports"
        log_dir = root / "logs"
        for directory in (root, state_dir, report_dir, log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            ledger_path=root / "ledger.db",
            state_dir=state_dir,
            report_dir=report_dir,
            anchor_path=root / "anchors.txt",
            log_dir=log_dir,
        )


def windbreak_argv(*args: str) -> list[str]:
    """Build the argv that invokes the windbreak CLI as a child process.

    Uses ``python -m windbreak`` against the *current* interpreter so the tier
    exercises the installed package without depending on the console script
    being on ``PATH``. Issue #467 covers the console-script path separately,
    because that is a distinct claim about the built wheel.

    Args:
        *args: Command-line arguments to append after the module name.

    Returns:
        The full argv list, interpreter first.
    """
    return [sys.executable, "-m", "windbreak", *args]


def run_windbreak(
    *args: str,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the CLI to completion as a child process and capture its output.

    For processes that are expected to terminate on their own. Long-running
    ones belong to :class:`ProcessLauncher`, which guarantees their teardown.

    ``input_text`` defaults to ``None``, which closes the child's stdin rather
    than inheriting the test runner's. That matters: `windbreak rearm` reads a
    confirmation phrase with :func:`input`, and a child inheriting a terminal
    would block forever instead of failing.

    Args:
        *args: Arguments passed to the windbreak CLI.
        env: Complete environment for the child. ``None`` inherits this one.
        cwd: Working directory for the child. ``None`` inherits this one.
        timeout: Seconds to allow before the child is killed.
        input_text: Text fed to the child's stdin. ``None`` sends EOF at once.

    Returns:
        The completed process, with ``stdout``/``stderr`` decoded as text.

    Raises:
        subprocess.TimeoutExpired: If the child outlives ``timeout``.
    """
    return subprocess.run(
        windbreak_argv(*args),
        input="" if input_text is None else input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=None if cwd is None else str(cwd),
        env=None if env is None else dict(env),
        check=False,
    )


@dataclass
class SpawnedProcess:
    """A running windbreak child process and its captured streams.

    Streams go to files rather than pipes on purpose: a long-running child
    whose pipe buffer fills will block forever, and a harness that can
    deadlock the thing it is supposed to be observing is worse than no harness.

    Attributes:
        name: Short label, used for the log filenames.
        argv: The exact argv the child was started with.
        process: The underlying :class:`subprocess.Popen` handle.
        stdout_path: File the child's stdout is written to.
        stderr_path: File the child's stderr is written to.
        stdout_stream: Open handle backing ``stdout_path``.
        stderr_stream: Open handle backing ``stderr_path``.
    """

    name: str
    argv: tuple[str, ...]
    process: subprocess.Popen[bytes]
    stdout_path: Path
    stderr_path: Path
    stdout_stream: IO[bytes] = field(repr=False)
    stderr_stream: IO[bytes] = field(repr=False)

    @property
    def pid(self) -> int:
        """Return the child's process id.

        Returns:
            The operating-system process id.
        """
        return self.process.pid

    def is_running(self) -> bool:
        """Report whether the child is still running.

        Returns:
            ``True`` while the child has not exited.
        """
        return self.process.poll() is None

    def stdout_text(self) -> str:
        """Read everything written to the child's stdout so far.

        Returns:
            The captured stdout, decoded as UTF-8 with errors replaced.
        """
        return _read_text(self.stdout_path)

    def stderr_text(self) -> str:
        """Read everything written to the child's stderr so far.

        Returns:
            The captured stderr, decoded as UTF-8 with errors replaced.
        """
        return _read_text(self.stderr_path)

    def wait(self, *, timeout: float) -> int:
        """Block until the child exits.

        Args:
            timeout: Seconds to wait before giving up.

        Returns:
            The child's exit status.

        Raises:
            subprocess.TimeoutExpired: If the child outlives ``timeout``.
        """
        return self.process.wait(timeout=timeout)

    def close_streams(self) -> None:
        """Close both captured-stream handles, ignoring already-closed ones."""
        for stream in (self.stdout_stream, self.stderr_stream):
            if not stream.closed:
                stream.close()


class ProcessLauncher:
    """Starts windbreak child processes and guarantees they are reaped.

    The launcher -- not the test -- owns process lifetime. A test that fails
    halfway through, or that never reaches its own cleanup, still leaves no
    orphan, because the fixture wrapping this class reaps in a ``finally``.
    """

    def __init__(self, log_dir: Path) -> None:
        """Create a launcher that writes child logs into ``log_dir``.

        Args:
            log_dir: Existing directory to write captured streams into.
        """
        self._log_dir = log_dir
        self._spawned: list[SpawnedProcess] = []

    @property
    def spawned(self) -> tuple[SpawnedProcess, ...]:
        """Return every process started by this launcher, oldest first.

        Returns:
            The spawned processes, including ones that have already exited.
        """
        return tuple(self._spawned)

    def track(self, spawned: SpawnedProcess) -> SpawnedProcess:
        """Adopt an already-started process into this launcher's lifetime.

        For processes this launcher did not start itself -- a different
        executable, such as the console script installed into a clean
        virtualenv -- so they still get the same guaranteed teardown as
        everything else.

        Args:
            spawned: The already-started process to take ownership of.

        Returns:
            The same process, for convenient chaining.
        """
        self._spawned.append(spawned)
        return spawned

    def spawn(
        self,
        *args: str,
        name: str,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> SpawnedProcess:
        """Start the windbreak CLI as a tracked background child process.

        Args:
            *args: Arguments passed to the windbreak CLI.
            name: Short label used for this child's log filenames.
            env: Complete environment for the child. ``None`` inherits this one.
            cwd: Working directory for the child. ``None`` inherits this one.

        Returns:
            The started :class:`SpawnedProcess`.
        """
        stdout_path = self._log_dir / f"{name}.stdout.log"
        stderr_path = self._log_dir / f"{name}.stderr.log"
        stdout_stream = stdout_path.open("wb")
        stderr_stream = stderr_path.open("wb")
        argv = windbreak_argv(*args)
        process = subprocess.Popen(
            argv,
            stdout=stdout_stream,
            stderr=stderr_stream,
            cwd=None if cwd is None else str(cwd),
            env=None if env is None else dict(env),
            start_new_session=True,
        )
        spawned = SpawnedProcess(
            name=name,
            argv=tuple(argv),
            process=process,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_stream=stdout_stream,
            stderr_stream=stderr_stream,
        )
        self._spawned.append(spawned)
        return spawned

    def reap_all(self) -> None:
        """Stop and reap every tracked process, newest first.

        Reaping in reverse start order stops readers before the writers they
        depend on, which keeps shutdown logs legible.

        A process that survives ``SIGKILL`` within the grace period does not
        abandon its siblings. Every remaining process is still reaped, and the
        survivors are reported together at the end. The earlier version let the
        second :class:`subprocess.TimeoutExpired` propagate straight out of this
        loop, which left every not-yet-reaped process running -- turning one
        stuck child into a leak of all of them, and quietly falsifying this
        class's "guarantees" claim. Found in review of PR #477.

        Raises:
            AssertionError: If any process was still alive after ``SIGKILL``.
                Raised only once every other process has been reaped.
        """
        survivors: list[str] = []
        while self._spawned:
            spawned = self._spawned.pop()
            try:
                reap(spawned)
            except subprocess.TimeoutExpired:
                survivors.append(f"{spawned.name} (pid {spawned.pid})")
        if survivors:
            message = (
                "these processes were still alive after SIGKILL: "
                f"{', '.join(survivors)}. Every other tracked process was "
                "still reaped."
            )
            raise AssertionError(message)


def reap(spawned: SpawnedProcess, *, grace: float = TERMINATE_GRACE_SECONDS) -> None:
    """Terminate a spawned process if needed and release its stream handles.

    Args:
        spawned: The process to stop.
        grace: Seconds to allow between ``SIGTERM`` and ``SIGKILL``.
    """
    try:
        _stop(spawned.process, grace=grace)
    finally:
        spawned.close_streams()


def _stop(process: subprocess.Popen[bytes], *, grace: float) -> None:
    """Signal a process group to stop, escalating to ``SIGKILL`` if needed.

    Args:
        process: The child to stop.
        grace: Seconds to allow before escalating.
    """
    if process.poll() is not None:
        return
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait(timeout=grace)


def _signal_group(process: subprocess.Popen[bytes], signal_number: int) -> None:
    """Send a signal to a child's whole process group.

    Children are started with ``start_new_session=True``, so signalling the
    group reaches anything the child itself spawned. A vanished group is not
    an error: it is the outcome we wanted.

    Args:
        process: The child whose group should be signalled.
        signal_number: The signal to send.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal_number)
    except (ProcessLookupError, PermissionError):
        return


def pid_alive(pid: int) -> bool:
    """Report whether a process id is still live.

    Args:
        pid: The process id to probe.

    Returns:
        ``True`` if a process with this id exists.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_text(path: Path) -> str:
    """Read a captured stream file, tolerating partial UTF-8.

    Args:
        path: File to read.

    Returns:
        The file's contents, or the empty string if it does not exist yet.
    """
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float,
    description: str,
    interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    """Block until ``predicate`` holds, or fail with a readable message.

    The only sanctioned way to wait in this tier. The sleep inside is a poll
    cadence bounded by ``timeout``, not a guess at how long something takes.

    Args:
        predicate: Zero-argument condition, re-evaluated until true.
        timeout: Seconds to keep trying before failing.
        description: What is being waited for, quoted in the failure message.
        interval: Seconds between evaluations.

    Raises:
        AssertionError: If ``predicate`` never holds within ``timeout``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    if predicate():
        return
    message = f"timed out after {timeout:.1f}s waiting for {description}"
    raise AssertionError(message)


def free_port() -> int:
    """Reserve and release a loopback port, returning its number.

    Inherently racy against another binder, which is why callers should bind
    promptly. It is still far better than a hardcoded port, which collides
    deterministically under any parallel run.

    Returns:
        A port number that was free on 127.0.0.1 a moment ago.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_is_serving(port: int, *, timeout: float = 0.25) -> bool:
    """Report whether something accepts TCP connections on a loopback port.

    Args:
        port: The loopback port to probe.
        timeout: Seconds to allow for the connection attempt.

    Returns:
        ``True`` if the connection succeeded.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def require_runtime(reason: str | None) -> None:
    """Gate the calling test on a runtime probe, skipping or failing closed.

    The one place the tier decides what a missing runtime means, so the answer
    cannot differ between modules. See :data:`REQUIRE_RUNTIME_ENV_VAR` for why
    the answer is environment-dependent rather than fixed.

    Args:
        reason: ``None`` when the runtime is present, else a human-readable
            explanation of what is missing, as returned by
            :func:`docker_skip_reason` and friends.

    Raises:
        Failed: If the runtime is absent while :data:`REQUIRE_RUNTIME_ENV_VAR`
            is armed.
    """
    if reason is None:
        return
    if os.environ.get(REQUIRE_RUNTIME_ENV_VAR) == REQUIRE_RUNTIME_ENABLED_VALUE:
        message = (
            f"{reason}. {REQUIRE_RUNTIME_ENV_VAR}="
            f"{REQUIRE_RUNTIME_ENABLED_VALUE} is set, so this tier refuses to "
            "skip: it is a required check, and a required check that skips "
            "reports success while verifying nothing. Provision the runtime "
            f"or unset {REQUIRE_RUNTIME_ENV_VAR}."
        )
        pytest.fail(message)
    pytest.skip(reason)


@lru_cache(maxsize=1)
def docker_skip_reason() -> str | None:
    """Explain why the docker tier cannot run here, if it cannot.

    Probes the daemon rather than the CLI alone: a ``docker`` binary with no
    reachable daemon is exactly the shape of this environment, and treating it
    as "available" would turn every container assertion into an error.

    Returns:
        ``None`` when docker is usable, else a reason naming what is missing.
    """
    executable = shutil.which("docker")
    if executable is None:
        return DOCKER_MISSING_REASON
    try:
        completed = subprocess.run(
            [executable, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=RUNTIME_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return DOCKER_DAEMON_REASON
    if completed.returncode != 0:
        return DOCKER_DAEMON_REASON
    return None


@lru_cache(maxsize=1)
def systemd_skip_reason() -> str | None:
    """Explain why the systemd tier cannot run here, if it cannot.

    Returns:
        ``None`` when systemd is usable, else a reason naming what is missing.
    """
    if shutil.which("systemd-analyze") is None:
        return SYSTEMD_ANALYZE_MISSING_REASON
    if not _SYSTEMD_RUNTIME_MARKER.exists():
        return SYSTEMD_NOT_RUNNING_REASON
    return None


def read_ledger_records(ledger_path: Path) -> list[LedgerRecord]:
    """Read every row of a ledger written by another process.

    Args:
        ledger_path: Path to the ledger database.

    Returns:
        All records in chain order, or an empty list if nothing was written.
    """
    if not ledger_path.exists():
        return []
    store = SqliteLedgerStore(ledger_path)
    try:
        return store.read_all()
    finally:
        store.close()


def ledger_payloads(
    ledger_path: Path, event_type: str, *, component: str | None = None
) -> list[dict[str, object]]:
    """Read one event type's ``data`` payloads off a ledger another wrote.

    The store is opened on the live file rather than a copy, so rows still in
    the write-ahead log are read through SQLite itself; a byte-level read of
    ``ledger.db`` alone would miss them, and a cross-process reader would then
    silently see a stale file.

    Args:
        ledger_path: Path to the ledger database.
        event_type: The event type wanted.
        component: The component whose rows are wanted, or ``None`` for every
            component's.

    Returns:
        Each matching row's ``data`` payload, in chain order.
    """
    return [
        dict(json.loads(record.payload_json)["data"])
        for record in read_ledger_records(ledger_path)
        if record.event_type == event_type
        and (component is None or record.component == component)
    ]


def ledger_event_types(ledger_path: Path) -> list[str]:
    """Read just the event-type column of a ledger, in chain order.

    Args:
        ledger_path: Path to the ledger database.

    Returns:
        Each record's ``event_type``, in chain order.
    """
    return [record.event_type for record in read_ledger_records(ledger_path)]


def verify_ledger_chain(ledger_path: Path) -> None:
    """Verify a ledger's hash chain from outside the process that wrote it.

    Args:
        ledger_path: Path to the ledger database.

    Raises:
        AssertionError: If the ledger does not exist.
    """
    if not ledger_path.exists():
        message = f"no ledger to verify at {ledger_path}"
        raise AssertionError(message)
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
    finally:
        store.close()
