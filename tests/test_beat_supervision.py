"""Tests for the heartbeat loop's beat supervision (issues #443 and #447).

Both issues are defects in the same seam -- `windbreak.main.run_loop` and the
per-beat hook it drives -- so they are pinned together here:

* **#443 (survival).** `run_loop` called `on_beat(seq)` bare, so any tick
  exception (a full ledger volume, a locked SQLite file, a transient venue
  error) unwound out of `main()` and killed the daemon: no alert, nothing
  ledgered, no next beat.
* **#447 (visibility).** The production `_on_beat` discarded the whole
  `TickOutcome`, so nothing read `kernel_halted`, and the heartbeat line logged
  the module constant `MODE_RESEARCH` rather than the kernel's own mode. A
  HALTed kernel emitted the same INFO line, byte for byte, as a healthy loop.

The two fixes interact, and the interaction is the point: catching the raise
without escalating it would have *deepened* #447 -- a loop that swallows every
tick failure while still logging `mode=RESEARCH heartbeat seq=N` is precisely
the undetectable-failure shape. So every test below asserts the failure or halt
becomes *louder*: a distinct heartbeat mode token, a CRITICAL log line on every
affected beat, and an alert dispatched exactly once per transition.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.alerts import (
    AlertDispatcher,
    AlertSeverity,
    AlertType,
    LoggingLedgerWriter,
)
from windbreak.alerts.factory import AlertSinkConfigError
from windbreak.config.schema import WindbreakConfig
from windbreak.ledger import SqliteLedgerStore
from windbreak.main import (
    MODE_RESEARCH,
    MODE_TICK_FAILED,
    MODE_UNKNOWN,
    BeatReport,
    BeatSupervisor,
    LedgerAlertWriter,
    _build_beat_supervisor,
    _build_dashboard_status_source,
    _build_paper_on_beat,
    main,
    run_loop,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.ledger import Event
    from windbreak.ledger.store import LedgerRecord


@dataclass
class _RecordingSink:
    """An `AlertSink` double recording every alert delivered through it.

    Attributes:
        name: The sink's identifier, as the `AlertSink` protocol requires.
        delivered: One `(type, severity, message)` triple per delivered alert,
            in delivery order.
    """

    name: str = "recording"
    delivered: list[tuple[AlertType, AlertSeverity, str]] = field(default_factory=list)

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Record one delivered alert.

        Args:
            alert_type: The dispatched alert type.
            severity: The alert's severity.
            message: The alert body.
        """
        self.delivered.append((alert_type, severity, message))


def _supervisor() -> tuple[BeatSupervisor, _RecordingSink]:
    """Build a supervisor whose dispatcher fans out to a recording sink.

    Returns:
        The supervisor and the sink recording everything it dispatches.
    """
    sink = _RecordingSink()
    return (
        BeatSupervisor(
            component="pipeline",
            dispatcher=AlertDispatcher(
                sinks=[sink], ledger_writer=LoggingLedgerWriter()
            ),
        ),
        sink,
    )


def _heartbeat_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Extract the heartbeat messages from a captured log.

    Args:
        caplog: The pytest log capture fixture.

    Returns:
        Every captured message containing ``heartbeat``, in order.
    """
    return [
        record.message for record in caplog.records if "heartbeat" in record.message
    ]


def _messages_at(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    """Extract the rendered messages logged at exactly one level.

    Args:
        caplog: The pytest log capture fixture.
        level: The `logging` level to select.

    Returns:
        Every matching record's rendered message, in order.
    """
    return [record.getMessage() for record in caplog.records if record.levelno == level]


def _raising_hook(seen: list[int]) -> Callable[[int], BeatReport | None]:
    """Build a beat hook that records its sequence then always raises.

    Args:
        seen: The list each invocation appends its sequence number to.

    Returns:
        A hook that raises `RuntimeError` on every beat.
    """

    def _hook(seq: int) -> BeatReport | None:
        """Record the beat then raise, as a full ledger volume would.

        Args:
            seq: The 1-based beat sequence number.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Always.
        """
        seen.append(seq)
        raise RuntimeError("database or disk is full")

    return _hook


# --- issue #443: a raising beat must not kill the loop ----------------------


def test_run_loop_survives_a_raising_beat_and_still_runs_every_later_beat() -> None:
    """A beat hook that raises every beat leaves the loop running to its budget.

    The premise of #443: before the fix this call propagated the first
    `RuntimeError` out of `run_loop`, so `seen` would have been `[1]` and the
    process would have died.
    """
    supervisor, _sink = _supervisor()
    seen: list[int] = []

    run_loop(0, max_beats=3, on_beat=_raising_hook(seen), supervisor=supervisor)

    assert seen == [1, 2, 3]


def test_run_loop_survives_a_raising_beat_with_no_supervisor_supplied() -> None:
    """Survival does not depend on the caller remembering to pass a supervisor.

    `supervisor` is optional, so a fix that only catches when one is supplied
    would leave every other `run_loop` caller exactly as fatal as before.
    """
    seen: list[int] = []

    run_loop(0, max_beats=2, on_beat=_raising_hook(seen))

    assert seen == [1, 2]


def test_run_loop_still_logs_its_shutdown_reason_after_a_raising_beat(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising beat does not rob the loop of its orderly shutdown line."""
    caplog.set_level(logging.INFO)
    supervisor, _sink = _supervisor()

    run_loop(0, max_beats=2, on_beat=_raising_hook([]), supervisor=supervisor)

    assert [
        record.message
        for record in caplog.records
        if record.message.startswith("shutdown reason=")
    ] == ["shutdown reason=max_beats"]


def test_a_run_of_failing_beats_dispatches_exactly_one_alert() -> None:
    """Three consecutive failing beats alert once -- on the transition only.

    Exactly-once is asserted with a counter across multiple beats, not `>= 1`
    on one beat: an alert re-dispatched every beat would be a pager storm, and
    the deduplication is what makes escalation usable on an always-on loop.
    """
    supervisor, sink = _supervisor()

    run_loop(0, max_beats=3, on_beat=_raising_hook([]), supervisor=supervisor)

    assert len(sink.delivered) == 1
    alert_type, severity, message = sink.delivered[0]
    assert alert_type is AlertType.HALT_KILL
    assert severity is AlertSeverity.CRITICAL
    assert message == "beat seq=1 failed: RuntimeError: database or disk is full"


def test_a_failure_that_heals_and_recurs_alerts_once_per_transition() -> None:
    """Beats 1 and 3 fail with beat 2 healthy: two transitions, two alerts.

    Deduplication must be per *transition*, not per process: a recurrence after
    a recovery is new information and must page again.
    """
    supervisor, sink = _supervisor()

    def _hook(seq: int) -> BeatReport | None:
        """Fail on the odd beats, succeed on the even one.

        Args:
            seq: The 1-based beat sequence number.

        Returns:
            `None` on an even beat.

        Raises:
            RuntimeError: On every odd beat.
        """
        if seq % 2 == 1:
            raise RuntimeError(f"failure on {seq}")
        return None

    run_loop(0, max_beats=3, on_beat=_hook, supervisor=supervisor)

    assert [message for _type, _severity, message in sink.delivered] == [
        "beat seq=1 failed: RuntimeError: failure on 1",
        "beat seq=3 failed: RuntimeError: failure on 3",
    ]


def test_a_failure_followed_by_a_halt_alerts_for_both_conditions() -> None:
    """A halt after a failure is a new condition, so it pages on its own.

    Deduplicating on "already escalated" rather than on *what* was escalated
    would let a HALT arrive silently behind an ongoing tick failure -- the two
    conditions have different causes and different operator responses.
    """
    supervisor, sink = _supervisor()

    def _hook(seq: int) -> BeatReport | None:
        """Fail beat 1, then report a halted kernel on beats 2 and 3.

        Args:
            seq: The 1-based beat sequence number.

        Returns:
            The halted report on beats after the first.

        Raises:
            RuntimeError: On beat 1.
        """
        if seq == 1:
            raise RuntimeError("boom")
        return BeatReport(mode="HALT", halted=True)

    run_loop(0, max_beats=3, on_beat=_hook, supervisor=supervisor)

    assert [message for _type, _severity, message in sink.delivered] == [
        "beat seq=1 failed: RuntimeError: boom",
        "risk kernel HALT at beat seq=2 (mode=HALT)",
    ]


def test_every_failing_beat_logs_its_own_critical_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The alert dedupes; the CRITICAL log line does not.

    This is what keeps the deduplication honest. If the *only* signal were the
    once-per-transition alert, beats 2 and 3 of an ongoing outage would emit
    nothing at all -- a quieter loop than before the fix.
    """
    caplog.set_level(logging.INFO)
    supervisor, _sink = _supervisor()

    run_loop(0, max_beats=3, on_beat=_raising_hook([]), supervisor=supervisor)

    assert _messages_at(caplog, logging.CRITICAL) == [
        "beat seq=1 failed: RuntimeError: database or disk is full",
        "beat seq=2 failed: RuntimeError: database or disk is full",
        "beat seq=3 failed: RuntimeError: database or disk is full",
    ]


def test_the_heartbeat_reports_tick_failed_never_research_on_a_failing_beat(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A swallowed failure changes the one line an operator watches.

    The whole hazard of catching the raise is a loop that keeps emitting
    `mode=RESEARCH heartbeat seq=N` while doing nothing. The heartbeat's mode
    slot must carry the failure instead.
    """
    caplog.set_level(logging.INFO)
    supervisor, _sink = _supervisor()

    run_loop(0, max_beats=2, on_beat=_raising_hook([]), supervisor=supervisor)

    assert _heartbeat_lines(caplog) == [
        "mode=TICK_FAILED heartbeat seq=1",
        "mode=TICK_FAILED heartbeat seq=2",
    ]
    assert MODE_TICK_FAILED == "TICK_FAILED"
    assert MODE_TICK_FAILED != MODE_RESEARCH


def test_a_recovered_beat_returns_the_heartbeat_to_its_reported_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failure token is not sticky: a healthy beat reports its own mode."""
    caplog.set_level(logging.INFO)
    supervisor, _sink = _supervisor()

    def _hook(seq: int) -> BeatReport | None:
        """Fail the first beat only, then report a healthy PAPER tick.

        Args:
            seq: The 1-based beat sequence number.

        Returns:
            A healthy PAPER report after the first beat.

        Raises:
            RuntimeError: On beat 1.
        """
        if seq == 1:
            raise RuntimeError("transient venue error")
        return BeatReport(mode="PAPER", halted=False)

    run_loop(0, max_beats=2, on_beat=_hook, supervisor=supervisor)

    assert _heartbeat_lines(caplog) == [
        "mode=TICK_FAILED heartbeat seq=1",
        "mode=PAPER heartbeat seq=2",
    ]


def test_a_hook_raising_a_base_exception_still_terminates_the_loop() -> None:
    """`KeyboardInterrupt` and `SystemExit` are not swallowed.

    Surviving a tick failure must not mean surviving an operator's Ctrl-C or an
    explicit `SystemExit`: those are shutdown requests, not tick failures.
    """
    supervisor, _sink = _supervisor()

    def _hook(seq: int) -> BeatReport | None:
        """Raise a `KeyboardInterrupt` on the first beat.

        Args:
            seq: The 1-based beat sequence number, unused.

        Returns:
            Never returns.

        Raises:
            KeyboardInterrupt: Always.
        """
        del seq
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_loop(0, max_beats=3, on_beat=_hook, supervisor=supervisor)


# --- issue #447: a HALTed kernel must not look like a healthy idle loop -----


def test_the_heartbeat_reports_the_beats_own_mode_not_the_research_constant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A PAPER tick's heartbeat says PAPER, which is what the ledger says.

    #447's reproduction: the log line said `mode=RESEARCH` while the
    `ModeHeartbeat` rows on the same run said `PAPER`.
    """
    caplog.set_level(logging.INFO)
    supervisor, sink = _supervisor()

    run_loop(
        0,
        max_beats=2,
        on_beat=lambda _seq: BeatReport(mode="PAPER", halted=False),
        supervisor=supervisor,
    )

    assert _heartbeat_lines(caplog) == [
        "mode=PAPER heartbeat seq=1",
        "mode=PAPER heartbeat seq=2",
    ]
    assert sink.delivered == []


def test_a_halted_kernel_reports_halt_in_the_heartbeat_and_alerts_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #447's acceptance criteria (a) and (b), on one halt transition.

    (a) the heartbeat line reports `HALT`, not `RESEARCH`; (b) exactly one
    alert is dispatched on the transition -- pinned with a counter across the
    three halted beats that follow, not with `>= 1` on the halting beat.
    """
    caplog.set_level(logging.INFO)
    supervisor, sink = _supervisor()

    def _hook(seq: int) -> BeatReport | None:
        """Report a healthy PAPER tick, then a halted kernel from beat 2 on.

        Args:
            seq: The 1-based beat sequence number.

        Returns:
            The beat's report.
        """
        if seq == 1:
            return BeatReport(mode="PAPER", halted=False)
        return BeatReport(mode="HALT", halted=True)

    run_loop(0, max_beats=4, on_beat=_hook, supervisor=supervisor)

    assert _heartbeat_lines(caplog) == [
        "mode=PAPER heartbeat seq=1",
        "mode=HALT heartbeat seq=2",
        "mode=HALT heartbeat seq=3",
        "mode=HALT heartbeat seq=4",
    ]
    assert len(sink.delivered) == 1
    alert_type, severity, message = sink.delivered[0]
    assert alert_type is AlertType.HALT_KILL
    assert severity is AlertSeverity.CRITICAL
    assert message == "risk kernel HALT at beat seq=2 (mode=HALT)"


def test_every_halted_beat_logs_its_own_critical_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ongoing halt keeps saying so at CRITICAL, beat after beat."""
    caplog.set_level(logging.INFO)
    supervisor, _sink = _supervisor()

    run_loop(
        0,
        max_beats=2,
        on_beat=lambda _seq: BeatReport(mode="HALT", halted=True),
        supervisor=supervisor,
    )

    assert _messages_at(caplog, logging.CRITICAL) == [
        "risk kernel HALT at beat seq=1 (mode=HALT)",
        "risk kernel HALT at beat seq=2 (mode=HALT)",
    ]


def test_a_halt_that_clears_and_returns_alerts_once_per_transition() -> None:
    """A cleared halt that recurs is new information and alerts again."""
    supervisor, sink = _supervisor()

    def _hook(seq: int) -> BeatReport | None:
        """Halt on beats 1 and 3, recover on beat 2.

        Args:
            seq: The 1-based beat sequence number.

        Returns:
            The beat's report.
        """
        if seq == 2:
            return BeatReport(mode="PAPER", halted=False)
        return BeatReport(mode="HALT", halted=True)

    run_loop(0, max_beats=3, on_beat=_hook, supervisor=supervisor)

    assert [message for _type, _severity, message in sink.delivered] == [
        "risk kernel HALT at beat seq=1 (mode=HALT)",
        "risk kernel HALT at beat seq=3 (mode=HALT)",
    ]


def test_a_research_budget_halt_warns_every_beat_without_paging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A research ceiling is visible on stdout but does not page CRITICAL.

    `research_halted` and `kernel_halted` are both `bool`, so only distinct
    *behaviour* can catch a swap: a budget ceiling is an expected, self-clearing
    daily event that must not raise the kernel-halt page, and the kernel's mode
    is unchanged by it.
    """
    caplog.set_level(logging.INFO)
    supervisor, sink = _supervisor()

    run_loop(
        0,
        max_beats=2,
        on_beat=lambda _seq: BeatReport(
            mode="PAPER", halted=False, research_halted=True
        ),
        supervisor=supervisor,
    )

    assert _messages_at(caplog, logging.WARNING) == [
        "research budget halted at beat seq=1 (mode=PAPER)",
        "research budget halted at beat seq=2 (mode=PAPER)",
    ]
    assert _messages_at(caplog, logging.CRITICAL) == []
    assert sink.delivered == []
    assert _heartbeat_lines(caplog) == [
        "mode=PAPER heartbeat seq=1",
        "mode=PAPER heartbeat seq=2",
    ]


def test_a_kernel_halt_is_not_downgraded_by_a_concurrent_research_halt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both flags set: the WARNING is emitted *and* the kernel halt still pages."""
    caplog.set_level(logging.INFO)
    supervisor, sink = _supervisor()

    run_loop(
        0,
        max_beats=1,
        on_beat=lambda _seq: BeatReport(mode="HALT", halted=True, research_halted=True),
        supervisor=supervisor,
    )

    assert _messages_at(caplog, logging.WARNING) == [
        "research budget halted at beat seq=1 (mode=HALT)"
    ]
    assert _messages_at(caplog, logging.CRITICAL) == [
        "risk kernel HALT at beat seq=1 (mode=HALT)"
    ]
    assert len(sink.delivered) == 1


def test_a_hook_reporting_nothing_leaves_the_research_heartbeat_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The snapshot hook reports no mode, so the loop stays RESEARCH and quiet.

    A research-only pass genuinely *is* RESEARCH, so the constant is truthful
    here; this pins that the #447 fix did not make every hook-less run shout.
    """
    caplog.set_level(logging.INFO)
    supervisor, sink = _supervisor()

    run_loop(0, max_beats=2, on_beat=lambda _seq: None, supervisor=supervisor)

    assert _heartbeat_lines(caplog) == [
        "mode=RESEARCH heartbeat seq=1",
        "mode=RESEARCH heartbeat seq=2",
    ]
    assert sink.delivered == []


# --- issue #447: the production PAPER hook must read the TickOutcome --------


@dataclass(frozen=True)
class _FakeMode:
    """A `Mode`-shaped double exposing only the `name` the heartbeat reads.

    Attributes:
        name: The mode token, e.g. ``PAPER`` or ``HALT``.
    """

    name: str


@dataclass(frozen=True)
class _FakeKernel:
    """A kernel double exposing only `mode`.

    Attributes:
        mode: The kernel's current mode.
    """

    mode: _FakeMode


@dataclass(frozen=True)
class _FakeDeps:
    """A `PaperTickDeps`-shaped double exposing only `kernel`.

    Attributes:
        kernel: The kernel whose real mode the heartbeat must report.
    """

    kernel: _FakeKernel


@dataclass(frozen=True)
class _FakeOutcome:
    """A `TickOutcome`-shaped double carrying the two halt flags.

    The two flags are deliberately independent so a hook that reads
    ``research_halted`` where it means ``kernel_halted`` -- a swap the fields'
    shared `bool` type cannot catch -- fails the test.

    Attributes:
        kernel_halted: Whether the Risk Kernel is in HALT after the tick.
        research_halted: Whether the tick's research hit a budget ceiling.
    """

    kernel_halted: bool
    research_halted: bool


def _paper_args(tmp_path: Path) -> argparse.Namespace:
    """Build a `run` argument namespace with all four PAPER flags supplied.

    Args:
        tmp_path: The per-test temporary directory the paths are rooted in.

    Returns:
        An `argparse.Namespace` carrying the PAPER flags.
    """
    return argparse.Namespace(
        paper_books_dir=tmp_path / "books",
        cassette_path=tmp_path / "cassette.json",
        ledger_path=tmp_path / "ledger.db",
        report_dir=tmp_path / "reports",
        paper_live_ticker=None,
    )


@pytest.mark.parametrize(
    ("mode_name", "kernel_halted", "research_halted"),
    [
        ("PAPER", False, False),
        ("HALT", True, False),
        ("PAPER", False, True),
        ("HALT", True, True),
    ],
)
def test_the_paper_hook_reports_the_kernel_mode_and_both_halt_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode_name: str,
    kernel_halted: bool,
    research_halted: bool,
) -> None:
    """The production hook reads `deps.kernel.mode` and the whole `TickOutcome`.

    #447's core finding was that this hook discarded the outcome: a repo-wide
    grep for `kernel_halted` returned zero hits in `main.py`. The four cases
    cover every combination of the two independent flags, so a hook that reads
    one where it means the other fails on the two mixed rows.

    Args:
        monkeypatch: Used to replace the scheduler seam the hook imports.
        tmp_path: The temporary directory the PAPER flags point into.
        mode_name: The kernel mode the double reports.
        kernel_halted: The tick outcome's kernel-halt flag.
        research_halted: The tick outcome's research-halt flag.
    """
    from windbreak.scheduler import loop as loop_module

    deps = _FakeDeps(kernel=_FakeKernel(mode=_FakeMode(name=mode_name)))
    outcome = _FakeOutcome(kernel_halted=kernel_halted, research_halted=research_halted)
    monkeypatch.setattr(loop_module, "build_paper_deps", lambda **_kwargs: deps)
    monkeypatch.setattr(loop_module, "run_single_tick", lambda _deps, *, beat: outcome)

    hook = _build_paper_on_beat(_paper_args(tmp_path), WindbreakConfig())

    assert hook(7) == BeatReport(
        mode=mode_name, halted=kernel_halted, research_halted=research_halted
    )


def test_the_paper_hooks_tick_failure_is_supervised_end_to_end(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A `store.append` failure inside the production hook is survived, loudly.

    Composes the real `_build_paper_on_beat` closure with the real `run_loop`
    and a real supervisor, and raises the exact `sqlite3.OperationalError` a
    full ledger volume produces -- #443's reproduction, minus the tmpfs.
    """
    caplog.set_level(logging.INFO)
    from windbreak.scheduler import loop as loop_module

    def _explode(_deps: object, *, beat: int) -> _FakeOutcome:
        """Fail the tick the way a full ledger volume does.

        Args:
            _deps: The (unused) dependency bundle.
            beat: The 1-based beat sequence number.

        Returns:
            Never returns.

        Raises:
            sqlite3.OperationalError: Always.
        """
        raise sqlite3.OperationalError(f"database or disk is full (beat {beat})")

    monkeypatch.setattr(
        loop_module,
        "build_paper_deps",
        lambda **_kwargs: _FakeDeps(kernel=_FakeKernel(mode=_FakeMode(name="PAPER"))),
    )
    monkeypatch.setattr(loop_module, "run_single_tick", _explode)
    supervisor, sink = _supervisor()

    run_loop(
        0,
        max_beats=2,
        on_beat=_build_paper_on_beat(_paper_args(tmp_path), WindbreakConfig()),
        supervisor=supervisor,
    )

    assert _heartbeat_lines(caplog) == [
        "mode=TICK_FAILED heartbeat seq=1",
        "mode=TICK_FAILED heartbeat seq=2",
    ]
    assert [message for _type, _severity, message in sink.delivered] == [
        "beat seq=1 failed: OperationalError: database or disk is full (beat 1)"
    ]


# --- issue #443: the failure is ledgered, not merely logged -----------------


class _FailingStore:
    """A `LedgerStore` whose `append` always raises, as a full volume does.

    Attributes:
        appended: Every event `append` was asked to persist, in order.
    """

    def __init__(self) -> None:
        """Initialize the store with an empty append log."""
        self.appended: list[Event] = []

    def append(self, event: Event) -> int:
        """Record the attempt then raise, as SQLite does on a full disk.

        Args:
            event: The event that would have been persisted.

        Returns:
            Never returns.

        Raises:
            sqlite3.OperationalError: Always.
        """
        self.appended.append(event)
        raise sqlite3.OperationalError("database or disk is full")

    def read_all(self) -> list[LedgerRecord]:
        """Return no records.

        Returns:
            An empty list.
        """
        return []

    def verify_chain(self) -> None:
        """Verify nothing; this double holds no chain."""

    def close(self) -> None:
        """Release nothing; this double holds no resources."""


def _alert_payloads(ledger_path: Path) -> list[dict[str, object]]:
    """Read every ledgered `AlertEmitted` payload, verifying the chain first.

    Args:
        ledger_path: The ledger database to read.

    Returns:
        One payload ``data`` mapping per `AlertEmitted` row, in ledger order.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    return [
        json.loads(record.payload_json)["data"]
        for record in records
        if record.event_type == "AlertEmitted"
    ]


def test_a_supervised_failure_is_appended_to_the_hash_chained_ledger(
    tmp_path: Path,
) -> None:
    """The alert a failing beat raises lands on the ledger, chain intact.

    #443's complaint was that "nothing is ledgered about the cause". The
    dispatcher's `LedgerWriter` seam is where that is closed.
    """
    ledger_path = tmp_path / "ledger.db"
    dispatcher = AlertDispatcher(
        sinks=[],
        ledger_writer=LedgerAlertWriter(ledger_path, component="pipeline"),
    )
    supervisor = BeatSupervisor(component="pipeline", dispatcher=dispatcher)

    run_loop(0, max_beats=2, on_beat=_raising_hook([]), supervisor=supervisor)

    assert _alert_payloads(ledger_path) == [
        {
            "severity": "critical",
            "message": "beat seq=1 failed: RuntimeError: database or disk is full",
        }
    ]


def test_a_ledger_that_cannot_be_written_never_takes_the_loop_down(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The full-disk case: ledgering the alert fails, and says so, loudly.

    On the very failure #443 reproduces -- a full ledger volume -- the attempt
    to *ledger* the alert fails too. That second failure must not become the
    thing that kills the daemon, and must not be silent either.
    """
    caplog.set_level(logging.INFO)
    store = _FailingStore()
    monkeypatch.setattr("windbreak.main.SqliteLedgerStore", lambda _path: store)
    dispatcher = AlertDispatcher(
        sinks=[],
        ledger_writer=LedgerAlertWriter(tmp_path / "ledger.db", component="pipeline"),
    )
    supervisor = BeatSupervisor(component="pipeline", dispatcher=dispatcher)
    seen: list[int] = []

    run_loop(0, max_beats=2, on_beat=_raising_hook(seen), supervisor=supervisor)

    assert seen == [1, 2]
    assert len(store.appended) == 1
    assert (
        "alert ledgering failed: OperationalError: database or disk is full"
        in _messages_at(caplog, logging.CRITICAL)
    )


def test_the_composed_supervisor_dispatches_through_the_configured_sinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_build_beat_supervisor` composes the real alert root, not a sink-less one.

    Issue #444 reports that the PAPER loop's own dispatcher is built with
    `sinks=[]`, so its alerts reach only the log-only fallback. `main` has the
    real composition root -- `_build_alert_dispatcher`, the sole caller of
    `build_sinks` -- and this pins that the beat supervisor is composed through
    it: a supervisor built over a second, sink-less dispatcher would deliver
    nothing to the recording sink.
    """
    sink = _RecordingSink()
    monkeypatch.setattr("windbreak.main.build_sinks", lambda *_a, **_k: [sink])
    args = argparse.Namespace(ledger_path=tmp_path / "ledger.db", process="pipeline")

    supervisor = _build_beat_supervisor(args, WindbreakConfig())
    supervisor.observe(1, _raising_hook([]))

    assert [message for _type, _severity, message in sink.delivered] == [
        "beat seq=1 failed: RuntimeError: database or disk is full"
    ]


def test_the_composed_supervisor_ledgers_to_the_configured_ledger_path(
    tmp_path: Path,
) -> None:
    """With `--ledger-path` the supervisor's alerts are hash-chained on disk."""
    ledger_path = tmp_path / "ledger.db"
    args = argparse.Namespace(ledger_path=ledger_path, process="pipeline")

    _build_beat_supervisor(args, WindbreakConfig()).observe(1, _raising_hook([]))

    assert _alert_payloads(ledger_path) == [
        {
            "severity": "critical",
            "message": "beat seq=1 failed: RuntimeError: database or disk is full",
        }
    ]


def test_the_composed_supervisor_writes_no_ledger_without_a_ledger_path(
    tmp_path: Path,
) -> None:
    """With no `--ledger-path` the supervisor creates no database file.

    A run that was never asked for a ledger must not grow one as a side effect
    of the new escalation path.
    """
    args = argparse.Namespace(ledger_path=None, process="pipeline")

    _build_beat_supervisor(args, WindbreakConfig()).observe(1, _raising_hook([]))

    assert list(tmp_path.iterdir()) == []


def test_the_composed_supervisor_stamps_the_process_component_on_its_log_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The escalation's log record carries the `--process` token, not a default.

    The ledgered row and the log record are stamped from two different places,
    so a hardcoded component in one of them is invisible to a test that only
    reads the other. This reads the record's own `component` extra, which is
    what every other heartbeat and shutdown line in this process is filtered by.
    The alert package's own log-only fallback record is filtered out by logger
    name: it is stamped `alerts`, which is correct and not what is under test.
    """
    caplog.set_level(logging.INFO)
    args = argparse.Namespace(ledger_path=None, process="order_gateway")

    _build_beat_supervisor(args, WindbreakConfig()).observe(1, _raising_hook([]))

    assert [
        record.__dict__["component"]
        for record in caplog.records
        if record.levelno == logging.CRITICAL and record.name == "windbreak"
    ] == ["order_gateway"]


def test_the_composed_supervisor_stamps_the_process_component_on_its_alerts(
    tmp_path: Path,
) -> None:
    """The ledgered alert carries the `--process` token, not a hardcoded one."""
    ledger_path = tmp_path / "ledger.db"
    args = argparse.Namespace(ledger_path=ledger_path, process="order_gateway")

    _build_beat_supervisor(args, WindbreakConfig()).observe(1, _raising_hook([]))

    store = SqliteLedgerStore(ledger_path)
    try:
        components = [record.component for record in store.read_all()]
    finally:
        store.close()

    assert components == ["order_gateway"]


# --- the supervised run refuses to start with an undeliverable alert path ---

#: The shared exchange fixtures the snapshot hook reads, resolved relative to
#: this file so the test does not depend on the pytest working directory.
_EXCHANGE_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "exchange"


def _snapshot_run_argv() -> list[str]:
    """Build the `run` argv for a one-beat loop with the snapshot hook wired.

    Returns:
        The argument vector `main` is invoked with.
    """
    return [
        "run",
        "--max-beats",
        "1",
        "--heartbeat-interval",
        "0",
        "--snapshot-fixture-dir",
        str(_EXCHANGE_FIXTURE_DIR),
    ]


def _logged_messages(capsys: pytest.CaptureFixture[str]) -> list[str]:
    """Extract the `msg` of every structured log line `main` wrote to stderr.

    `main` installs the JSON logging pipeline, which does not propagate to the
    `caplog` handler, so a test that drives the whole CLI reads the records
    back off stderr.

    Args:
        capsys: The pytest stdout/stderr capture fixture.

    Returns:
        One message per emitted log record, in order.
    """
    return [
        str(json.loads(line)["msg"])
        for line in capsys.readouterr().err.splitlines()
        if line
    ]


def test_a_run_with_an_undeliverable_alert_sink_fails_closed_at_startup(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sink that cannot be composed stops the run before the first beat.

    Discovering an undeliverable alert path at the moment a halt needs paging
    is the failure this forestalls: the escalation would be composed, dispatched
    into nothing, and the operator would learn about the halt never.
    """

    def _raise(*_args: object, **_kwargs: object) -> list[object]:
        """Refuse to compose any sink.

        Returns:
            Never returns.

        Raises:
            AlertSinkConfigError: Always.
        """
        raise AlertSinkConfigError("ntfy sink names unset NTFY_TOPIC_ENV")

    monkeypatch.setattr("windbreak.main.build_sinks", _raise)

    exit_code = main(_snapshot_run_argv())
    messages = _logged_messages(capsys)

    assert exit_code == 1
    assert "FATAL: ntfy sink names unset NTFY_TOPIC_ENV" in messages
    assert [message for message in messages if "heartbeat" in message] == []


def test_a_failing_beat_under_the_real_cli_pages_the_configured_sink(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: `run`'s own loop escalates through the composed dispatcher.

    Composing a supervisor and then not handing it to `run_loop` would leave the
    loop on its log-only default -- surviving, but paging nobody. Driving the
    whole `main` -> `_run_heartbeat` -> `run_loop` path with a hook that raises
    is what makes that wiring falsifiable.
    """
    sink = _RecordingSink()
    monkeypatch.setattr("windbreak.main.build_sinks", lambda *_a, **_k: [sink])
    monkeypatch.setattr(
        "windbreak.main._resolve_on_beat", lambda *_a, **_k: _raising_hook([])
    )

    exit_code = main(_snapshot_run_argv())
    messages = _logged_messages(capsys)

    assert exit_code == 0
    assert [message for _type, _severity, message in sink.delivered] == [
        "beat seq=1 failed: RuntimeError: database or disk is full"
    ]
    assert [message for message in messages if "heartbeat" in message] == [
        "mode=TICK_FAILED heartbeat seq=1"
    ]


def test_the_same_run_with_composable_sinks_beats_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The control: the default config composes, so the loop runs its one beat.

    Without this the fail-closed test above would pass just as well against a
    `run` that always exited 1.
    """
    exit_code = main(_snapshot_run_argv())
    messages = _logged_messages(capsys)

    assert exit_code == 0
    assert [message for message in messages if "heartbeat" in message] == [
        "mode=RESEARCH heartbeat seq=1"
    ]


# --- issue #447: the dashboard's no-evidence default must not claim health --


def test_the_dashboard_default_status_reports_unknown_never_research() -> None:
    """With no ledger, the dashboard has no evidence of any mode.

    Reporting `RESEARCH` there is the same lie as the heartbeat's hardcoded
    constant: it renders "I have never seen a heartbeat" as a healthy
    research-only loop.
    """
    status = _build_dashboard_status_source(None)()

    assert status.mode == MODE_UNKNOWN
    assert MODE_UNKNOWN == "UNKNOWN"
    assert status.mode != MODE_RESEARCH
    assert status.last_heartbeat is None


def test_the_dashboard_reports_halt_from_the_ledgers_mode_heartbeat(
    tmp_path: Path,
) -> None:
    """A HALTed kernel's ledgered mode reaches the dashboard status line.

    The riskkernel and scheduler sides already ledger the real mode; this pins
    that the dashboard surfaces it rather than the RESEARCH default.
    """
    from windbreak.ledger.events import ModeHeartbeat

    ledger_path = tmp_path / "ledger.db"
    store = SqliteLedgerStore(ledger_path)
    try:
        store.append(ModeHeartbeat(component="pipeline", mode="PAPER", beat=1))
        store.append(ModeHeartbeat(component="pipeline", mode="HALT", beat=2))
    finally:
        store.close()

    status = _build_dashboard_status_source(ledger_path)()

    assert status.mode == "HALT"
    assert status.last_heartbeat is not None


def test_the_dashboard_reports_unknown_when_the_ledger_has_no_heartbeat(
    tmp_path: Path,
) -> None:
    """An existing ledger with no `ModeHeartbeat` row is still no evidence."""
    ledger_path = tmp_path / "ledger.db"
    SqliteLedgerStore(ledger_path).close()

    status = _build_dashboard_status_source(ledger_path)()

    assert status.mode == MODE_UNKNOWN
    assert status.last_heartbeat is None
