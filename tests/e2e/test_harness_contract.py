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

import pytest

from tests.e2e.harness import (
    ProcessLauncher,
    RunRoot,
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
