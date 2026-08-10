"""The end-to-end tier proves itself before anything depends on it (#466).

Two claims, both about the harness rather than about windbreak:

1. The tier can start the real CLI as a child process and read its output --
   the capability every sibling issue of epic #465 builds on.
2. :class:`~tests.e2e.harness.ProcessLauncher` reaps its children even when
   the test body raises. Without that, one failing end-to-end test leaves a
   heartbeat loop running on the machine, and the next run inherits it.

Claim 2 matters more than it looks. The reason this tier did not exist is that
process-level tests are easy to write badly; a harness that leaks processes on
failure gets disabled within a week, and the coverage goes with it.
"""

from __future__ import annotations

import subprocess
from typing import cast

import pytest

from tests.e2e.harness import (
    ProcessLauncher,
    RunRoot,
    SpawnedProcess,
    docker_skip_reason,
    pid_alive,
    run_windbreak,
    systemd_skip_reason,
    wait_until,
)

pytestmark = pytest.mark.e2e

#: Sentinel from the CLI's own top-level help, asserted verbatim.
_HELP_SENTINEL = "windbreak always-on forecast trader CLI."

#: A heartbeat interval long enough that the child is still running when we
#: assert on it, without the test ever waiting for a second beat.
_LONG_INTERVAL_SECONDS = "3600"


def test_cli_help_runs_as_a_child_process_and_exits_zero() -> None:
    """The real CLI runs as a subprocess, exits 0 and prints its own help.

    Trivial on purpose. It is the smallest claim that proves argv construction,
    interpreter resolution, output capture and exit-status propagation all work
    before a sibling issue trusts them.
    """
    completed = run_windbreak("--help", timeout=60.0)

    assert completed.returncode == 0, completed.stderr
    assert _HELP_SENTINEL in completed.stdout


def test_cli_rejects_an_unknown_flag_with_a_nonzero_exit() -> None:
    """A bogus flag fails the child process, so the harness can see failure.

    The counterpart to the test above: a harness that reports success for
    everything is not evidence. This pins that a real failure is visible as a
    non-zero exit status.
    """
    completed = run_windbreak("run", "--make-money", timeout=60.0)

    assert completed.returncode != 0
    assert "--make-money" in completed.stderr


def test_launcher_reaps_its_child_when_the_test_body_raises(
    run_root: RunRoot,
) -> None:
    """A failing test still leaves no orphaned windbreak process.

    Drives the launcher directly rather than through the ``launcher`` fixture
    so the failure can be raised *inside* the managed scope and the reaping
    asserted afterwards -- the fixture's own ``finally`` runs too late to be
    observed from within a test that it wraps.
    """
    process_launcher = ProcessLauncher(run_root.log_dir)
    boom = RuntimeError("deliberate failure inside the managed scope")

    try:
        spawned = process_launcher.spawn(
            "run",
            "--process",
            "pipeline",
            "--heartbeat-interval",
            _LONG_INTERVAL_SECONDS,
            "--ledger-path",
            str(run_root.ledger_path),
            name="orphan-probe",
        )
        wait_until(
            lambda: pid_alive(spawned.pid),
            timeout=30.0,
            description="the spawned windbreak process to be running",
        )
        raise boom
    except RuntimeError as raised:
        assert raised is boom
    finally:
        process_launcher.reap_all()

    assert not spawned.is_running()
    wait_until(
        lambda: not pid_alive(spawned.pid),
        timeout=30.0,
        description="the spawned windbreak process to be reaped",
    )


def test_launcher_captures_child_streams_to_files(run_root: RunRoot) -> None:
    """Child output lands in files, so a long-running child cannot deadlock.

    Piping a long-running process's stdout and never draining it blocks the
    child once the pipe buffer fills. Files make that failure mode impossible,
    and let a test wait on log content.
    """
    process_launcher = ProcessLauncher(run_root.log_dir)
    try:
        spawned = process_launcher.spawn("--help", name="help-capture")
        spawned.wait(timeout=60.0)
    finally:
        process_launcher.reap_all()

    assert spawned.stdout_path.exists()
    assert _HELP_SENTINEL in spawned.stdout_text()


def test_wait_until_fails_with_a_readable_message_on_timeout() -> None:
    """`wait_until` fails loudly rather than hanging or passing silently.

    The bounded-wait primitive is load-bearing for every sibling issue, so its
    failure path is pinned here rather than discovered during a red CI run.
    """
    with pytest.raises(AssertionError, match="a condition that never holds"):
        wait_until(
            lambda: False,
            timeout=0.2,
            description="a condition that never holds",
            interval=0.01,
        )


def test_runtime_probes_report_a_reason_or_none() -> None:
    """Runtime probes return either ``None`` or a reason naming what is missing.

    This is what makes a `container` test skip honestly. The assertion holds in
    both directions -- on a machine with docker the reason is ``None``, on one
    without it the string names the missing runtime -- so the test is
    meaningful wherever it runs.
    """
    for reason in (docker_skip_reason(), systemd_skip_reason()):
        if reason is None:
            continue
        assert "unavailable" in reason


def test_one_shot_runner_raises_on_a_child_that_overruns_its_timeout(
    run_root: RunRoot,
) -> None:
    """`run_windbreak` enforces its timeout instead of hanging the suite.

    A merge-gating tier that can hang is a tier that gets removed. The timeout
    is asserted as an exception, not as a wall-clock hope.
    """
    with pytest.raises(subprocess.TimeoutExpired):
        run_windbreak(
            "run",
            "--process",
            "pipeline",
            "--heartbeat-interval",
            _LONG_INTERVAL_SECONDS,
            "--ledger-path",
            str(run_root.ledger_path),
            timeout=2.0,
        )


class _UnkillableProcess:
    """A `Popen` stand-in that never dies, to exercise the reaping loop.

    A genuinely `SIGKILL`-proof child needs uninterruptible kernel I/O, which a
    test cannot reliably arrange. Faking the one behaviour that matters -- a
    `wait` that always times out -- tests the loop's resilience directly, which
    is the property under review.
    """

    #: Stands in for a real child's process id in failure messages.
    pid = -1

    def poll(self) -> None:
        """Report the process as still running.

        Returns:
            ``None``, meaning "has not exited".
        """
        return None

    def wait(self, timeout: float | None = None) -> int:
        """Never exit.

        Args:
            timeout: Ignored; present to match `Popen.wait`'s signature.

        Raises:
            subprocess.TimeoutExpired: Always.
        """
        raise subprocess.TimeoutExpired(cmd="unkillable", timeout=timeout or 0.0)


def _unkillable(run_root: RunRoot) -> SpawnedProcess:
    """Build a tracked process that cannot be reaped.

    Args:
        run_root: The run root whose log dir backs the fake streams.

    Returns:
        A :class:`SpawnedProcess` wrapping :class:`_UnkillableProcess`.
    """
    stdout_path = run_root.log_dir / "unkillable.stdout.log"
    stderr_path = run_root.log_dir / "unkillable.stderr.log"
    return SpawnedProcess(
        name="unkillable",
        argv=("fake",),
        process=cast("subprocess.Popen[bytes]", _UnkillableProcess()),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_stream=stdout_path.open("wb"),
        stderr_stream=stderr_path.open("wb"),
    )


def test_reap_all_still_reaps_siblings_when_one_survives_sigkill(
    run_root: RunRoot,
) -> None:
    """One stuck child does not strand the others, and is reported by name.

    Found in review of PR #477: the loop used to let the post-`SIGKILL`
    `TimeoutExpired` propagate, so a single unkillable process left every
    not-yet-reaped sibling running. That turned one stuck child into a leak of
    all of them while the module still claimed to "guarantee" reaping.

    ORDER IS THE WHOLE TEST. `reap_all` pops newest-first, so the unkillable
    process is tracked LAST and therefore fails on the FIRST iteration, leaving
    the real child to be reaped on the second -- and only if the loop genuinely
    continues past the failure.

    The reverse order looks equivalent and is not: the real child would be
    popped and fully reaped before the failure was ever raised, so the test
    would pass on a loop that merely happened to fail last. It was written that
    way, and a `break` in the exception handler -- the exact regression this
    guards -- survived it. Raised in review of PR #477.
    """
    process_launcher = ProcessLauncher(run_root.log_dir)
    real = process_launcher.spawn(
        "run",
        "--process",
        "pipeline",
        "--heartbeat-interval",
        _LONG_INTERVAL_SECONDS,
        "--ledger-path",
        str(run_root.ledger_path),
        name="sibling",
    )
    wait_until(
        lambda: pid_alive(real.pid),
        timeout=30.0,
        description="the sibling process to be running",
    )
    process_launcher.track(_unkillable(run_root))

    with pytest.raises(AssertionError, match="still alive after SIGKILL"):
        process_launcher.reap_all()

    assert not real.is_running()
    wait_until(
        lambda: not pid_alive(real.pid),
        timeout=30.0,
        description="the sibling process to be reaped despite the stuck one",
    )


#: Pid recorded by the `launcher`-fixture test below, asserted dead by the test
#: after it. Observing a fixture's teardown requires outliving the test that
#: requested it, and pytest runs a module's tests in declaration order.
_FIXTURE_SPAWNED_PID: list[int] = []


def test_launcher_fixture_spawns_a_tracked_process(
    launcher: ProcessLauncher, run_root: RunRoot
) -> None:
    """The `launcher` fixture starts a real, tracked child process.

    Closes a coverage gap raised in review: the fixture -- whose ``finally``
    is what the harness's raise-safety guarantee actually rests on -- was not
    requested by any test, so its teardown path never ran under test.

    Args:
        launcher: The fixture under test.
        run_root: The run root supplying the ledger path.
    """
    spawned = launcher.spawn(
        "run",
        "--process",
        "pipeline",
        "--heartbeat-interval",
        _LONG_INTERVAL_SECONDS,
        "--ledger-path",
        str(run_root.ledger_path),
        name="fixture-managed",
    )
    wait_until(
        lambda: pid_alive(spawned.pid),
        timeout=30.0,
        description="the fixture-managed process to be running",
    )
    _FIXTURE_SPAWNED_PID.append(spawned.pid)

    assert launcher.spawned == (spawned,)


def test_launcher_fixture_teardown_reaped_the_previous_test_s_process() -> None:
    """The `launcher` fixture's teardown actually reaped what it started.

    Reads the pid recorded by the test above. A fixture's teardown runs after
    its test returns, so the only way to observe it is from outside that test.
    """
    assert _FIXTURE_SPAWNED_PID, "the preceding test did not record a pid"

    wait_until(
        lambda: not pid_alive(_FIXTURE_SPAWNED_PID[0]),
        timeout=30.0,
        description="the fixture's teardown to have reaped its process",
    )
