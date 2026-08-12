"""Real ticks driven to failure and to HALT, through the real loop (#443/#447).

Both issues' acceptance criteria ask for integration tests, not unit doubles.
Every test here runs a *real* `run_single_tick` over the shared offline fixtures
through the *real* `_build_paper_on_beat` hook and the *real* `run_loop`.

1. `test_a_full_volume_mid_tick_leaves_the_loop_beating_and_says_so` (#443).
   Only the disk is faked, by the one wrapper that turns `EquitySampled`'s
   append -- the exact call in the issue's traceback (`loop.py:3038` ->
   `store.py:420`) -- into the `sqlite3.OperationalError` a full volume raises.
   Before the fix that raise unwound out of `run_loop` and killed the daemon
   after the first beat. The assertions are the whole survival contract: beat 2
   still runs, the failure is alerted exactly once and ledgered, the heartbeat
   line stops claiming RESEARCH, the *dashboard* stops reporting `PAPER`, and
   the hash chain the tick had already written survives -- a liveness failure,
   never a corruption one.
2. `test_the_same_tick_without_a_full_volume_reports_its_real_mode` -- the
   control, without which the survival test could pass against a loop that
   reports `TICK_FAILED` unconditionally.
3. `test_a_real_verification_breach_halts_the_kernel_and_pages_exactly_once`
   (#447). Nothing is faked at all: an order placed on the exchange behind the
   loop's back drives `_verification_stage` -> `run_verification_cycle` to a
   genuine breach, `deps.kernel.mode` to `Mode.HALT`, and the loop's heartbeat
   to `HALT` -- and keeps it there across later beats, which is the only way to
   exercise the once-per-transition dedupe against a real halt rather than a
   hand-built `BeatReport`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sqlite3
from datetime import date
from typing import TYPE_CHECKING

from tests.integration.conftest import FIXED_NOW_EPOCH_S, ledger_path_for
from windbreak.alerts import AlertDispatcher, LoggingLedgerWriter
from windbreak.ledger import SqliteLedgerStore
from windbreak.main import (
    BeatReport,
    BeatSupervisor,
    LedgerAlertWriter,
    LedgerModeWriter,
    _build_dashboard_status_source,
    _build_paper_on_beat,
    run_loop,
)
from windbreak.reports import maybe_write_weekly

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

    from windbreak.alerts import AlertSeverity, AlertType
    from windbreak.config.schema import WindbreakConfig
    from windbreak.ledger import Event
    from windbreak.ledger.store import LedgerRecord, LedgerStore

#: The event whose append the fake full volume refuses. `_equity_and_positions_stage`
#: appends it after the tick's `ModeHeartbeat`, so the tick is genuinely partway
#: through when the disk gives out -- the shape of issue #443's traceback.
_FULL_AT_EVENT_TYPE = "EquitySampled"

#: The `today` the report-fault expectation is derived with. Fixed and
#: arbitrary: it shapes only the dated report filename, and the unified
#: report-directory diagnosis (#551) names the directory, never the file -- so
#: no ISO week, timezone or clock reaches the expectation.
_A_REPORT_DATE = date(2026, 1, 7)


@dataclasses.dataclass
class _RecordingSink:
    """An `AlertSink` double recording every alert delivered through it.

    Attributes:
        name: The sink's identifier, as the `AlertSink` protocol requires.
        delivered: One `(type, severity, message)` triple per delivered alert.
    """

    name: str = "recording"
    delivered: list[tuple[AlertType, AlertSeverity, str]] = dataclasses.field(
        default_factory=list
    )

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


class _FullVolumeStore:
    """A `LedgerStore` delegating to a real one until the volume "fills".

    Attributes:
        refused: Every event type whose append was refused, in order.
    """

    def __init__(self, inner: LedgerStore, *, refuse: str) -> None:
        """Wrap a real store, refusing one event type's appends.

        Args:
            inner: The real store every other append is delegated to.
            refuse: The `event_type` whose append raises instead.
        """
        self._inner = inner
        self._refuse = refuse
        self.refused: list[str] = []

    def append(self, event: Event) -> int:
        """Append via the real store, or raise as a full volume does.

        Args:
            event: The event to persist.

        Returns:
            The sequence number the real store assigned.

        Raises:
            sqlite3.OperationalError: When ``event`` is the refused type.
        """
        if event.event_type == self._refuse:
            self.refused.append(event.event_type)
            raise sqlite3.OperationalError("database or disk is full")
        return self._inner.append(event)

    def read_all(self) -> list[LedgerRecord]:
        """Read every record from the real store.

        Returns:
            The real store's records, in ledger order.
        """
        return self._inner.read_all()

    def verify_chain(self) -> None:
        """Verify the real store's hash chain."""
        self._inner.verify_chain()

    def close(self) -> None:
        """Close the real store."""
        self._inner.close()


def _log_only_dispatcher() -> AlertDispatcher:
    """Build the sink-less dispatcher a no-sink deployment composes.

    Since issue #444 the PAPER hook takes its alert root as a parameter rather
    than hardcoding one. These scenarios patch `build_paper_deps` out entirely,
    so the root the hook is handed never reaches a verification cycle; the
    log-only dispatcher is the honest stand-in, and it is exactly what
    `build_paper_deps` composes when no sink is configured.

    Returns:
        A dispatcher whose `log-only` fallback carries every alert.
    """
    return AlertDispatcher(sinks=[], ledger_writer=LoggingLedgerWriter())


def _paper_args(tmp_path: Path, ledger_path: Path) -> argparse.Namespace:
    """Build the `run` namespace `_build_paper_on_beat` reads its flags from.

    Args:
        tmp_path: The per-test scratch directory.
        ledger_path: The tick ledger's path.

    Returns:
        A namespace carrying the four PAPER flags and no live ticker.
    """
    return argparse.Namespace(
        paper_books_dir=tmp_path / "books",
        cassette_path=tmp_path / "cassette.json",
        ledger_path=ledger_path,
        report_dir=tmp_path / "reports",
        paper_live_ticker=None,
        process="pipeline",
    )


def _fixed_clock() -> int:
    """Return the suite's fixed epoch second, so a tick is deterministic.

    Returns:
        The shared fixture epoch second.
    """
    return FIXED_NOW_EPOCH_S


def _alert_messages(ledger_path: Path) -> list[str]:
    """Read every ledgered alert message, verifying the chain first.

    Args:
        ledger_path: The alert ledger's path.

    Returns:
        One message per `AlertEmitted` row, in ledger order.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    return [
        str(json.loads(record.payload_json)["data"]["message"])
        for record in records
        if record.event_type == "AlertEmitted"
    ]


def test_a_full_volume_mid_tick_leaves_the_loop_beating_and_says_so(
    books_dir: Path,
    caplog: pytest.LogCaptureFixture,
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    research_tools_factory: Callable[[], object],
    tmp_path: Path,
) -> None:
    """A real tick whose ledger volume fills is survived, alerted, and ledgered.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        caplog: The pytest log capture fixture.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to hand the hook the disk-failing dependency bundle.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        research_tools_factory: Builds the offline research tools.
        tmp_path: The per-test scratch directory.
    """
    caplog.set_level(logging.INFO)
    from windbreak.scheduler import loop as loop_module

    tick_ledger_path = ledger_path_for(tmp_path, "tick.db")
    alert_ledger_path = ledger_path_for(tmp_path, "alerts.db")
    real_deps = loop_module.build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=tick_ledger_path,
        report_dir=report_dir,
        config=paper_config,
        research_tools=research_tools_factory(),
        clock=_fixed_clock,
    )
    store = _FullVolumeStore(real_deps.store, refuse=_FULL_AT_EVENT_TYPE)
    deps = dataclasses.replace(real_deps, store=store)
    monkeypatch.setattr(loop_module, "build_paper_deps", lambda **_kwargs: deps)
    supervisor = BeatSupervisor(
        component="pipeline",
        dispatcher=AlertDispatcher(
            sinks=[],
            ledger_writer=LedgerAlertWriter(alert_ledger_path, component="pipeline"),
        ),
        mode_writer=LedgerModeWriter(tick_ledger_path, component="pipeline"),
    )

    run_loop(
        0,
        max_beats=2,
        on_beat=_build_paper_on_beat(
            _paper_args(tmp_path, tick_ledger_path),
            paper_config,
            dispatcher=_log_only_dispatcher(),
        ),
        supervisor=supervisor,
    )

    assert store.refused == [_FULL_AT_EVENT_TYPE, _FULL_AT_EVENT_TYPE]
    assert [
        record.message
        for record in caplog.records
        if "heartbeat seq=" in record.message
    ] == [
        "mode=TICK_FAILED heartbeat seq=1",
        "mode=TICK_FAILED heartbeat seq=2",
    ]
    assert _alert_messages(alert_ledger_path) == [
        "beat seq=1 failed: OperationalError: database or disk is full"
    ]
    store.verify_chain()
    store.close()
    # The dashboard is the path that matters, and the one the verifier caught
    # still reporting health: `run_single_tick` stamps `ModeHeartbeat(PAPER)`
    # *before* `_equity_and_positions_stage` raises, so each failing beat left
    # a healthy row with a freshened timestamp for `_ledger_source` to read.
    # The supervisor's own row lands after the tick's, so the latest one names
    # what the beat actually did.
    assert _ledgered_modes(tick_ledger_path) == [
        "PAPER",
        "TICK_FAILED",
        "PAPER",
        "TICK_FAILED",
    ]
    status = _build_dashboard_status_source(tick_ledger_path)()
    assert status.mode == "TICK_FAILED"
    assert status.mode != "PAPER"


def _report_write_failure_text(report_dir: Path) -> str:
    """Return the `Type: message` a real report write raises against `report_dir`.

    Derived by making the call the tick makes -- `maybe_write_weekly`, on the
    same path -- and rendering it the way `BeatSupervisor.observe` renders a
    raising beat, so a platform whose wording differs moves the expectation
    with the behaviour. `OSError` is caught rather than the specific type, so
    this derives what the call *does* rather than presupposing it; the test
    asserts the type separately.

    `today` is fixed and arbitrary: it shapes only the dated report filename,
    which the report-directory diagnosis deliberately does not name.

    Args:
        report_dir: The report directory, currently a regular file.

    Returns:
        The exception's type name and message, as the supervisor renders them.

    Raises:
        AssertionError: If the call succeeded, meaning the fault under test was
            never in place and every assertion resting on it proves nothing.
    """
    try:
        maybe_write_weekly(report_dir, today=_A_REPORT_DATE)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    message = f"{report_dir} is a usable directory, so no fault was induced"
    raise AssertionError(message)


def test_a_report_volume_that_is_a_file_leaves_the_loop_beating_and_says_so(
    books_dir: Path,
    caplog: pytest.LogCaptureFixture,
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    research_tools_factory: Callable[[], object],
    tmp_path: Path,
) -> None:
    """A real tick whose report volume came back as a file is still survivable.

    The in-process counterpart of
    `tests/e2e/test_unattended_run.py::\
test_an_induced_report_volume_fault_is_survived_and_stays_loud`, and the guard
    that issue #551's unified diagnosis did not cost the #443/#444/#447
    behaviour it is carried by. Nothing is stubbed: the fault is induced from
    the filesystem, by replacing the report directory with a regular file after
    the dependency bundle is built, exactly as a bad bind mount presents.

    The escalation is asserted end to end and none of it by scraping a log for
    a substring: the alert message is *derived* by making the same call on the
    same path, the heartbeat lines stop claiming PAPER, the ledger carries the
    supervisor's row after the healthy row the tick had already stamped (#447's
    ordering), and beat 2 still runs, which is the survival #443 asks for.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        caplog: The pytest log capture fixture.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to hand the hook the real dependency bundle.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory, faulted below.
        research_tools_factory: Builds the offline research tools.
        tmp_path: The per-test scratch directory.
    """
    caplog.set_level(logging.INFO)
    from windbreak.scheduler import loop as loop_module

    tick_ledger_path = ledger_path_for(tmp_path, "tick.db")
    alert_ledger_path = ledger_path_for(tmp_path, "alerts.db")
    deps = loop_module.build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=tick_ledger_path,
        report_dir=report_dir,
        config=paper_config,
        research_tools=research_tools_factory(),
        clock=_fixed_clock,
    )
    monkeypatch.setattr(loop_module, "build_paper_deps", lambda **_kwargs: deps)
    supervisor = BeatSupervisor(
        component="pipeline",
        dispatcher=AlertDispatcher(
            sinks=[],
            ledger_writer=LedgerAlertWriter(alert_ledger_path, component="pipeline"),
        ),
        mode_writer=LedgerModeWriter(tick_ledger_path, component="pipeline"),
    )
    report_dir.write_text("", encoding="utf-8")

    run_loop(
        0,
        max_beats=2,
        on_beat=_build_paper_on_beat(
            _paper_args(tmp_path, tick_ledger_path),
            paper_config,
            dispatcher=_log_only_dispatcher(),
        ),
        supervisor=supervisor,
    )

    raised = _report_write_failure_text(report_dir)
    assert raised.startswith("WeeklyReportDirectoryError: ")
    assert str(report_dir) in raised
    assert _alert_messages(alert_ledger_path) == [f"beat seq=1 failed: {raised}"]
    assert [
        record.message
        for record in caplog.records
        if "heartbeat seq=" in record.message
    ] == [
        "mode=TICK_FAILED heartbeat seq=1",
        "mode=TICK_FAILED heartbeat seq=2",
    ]
    assert _ledgered_modes(tick_ledger_path) == [
        "PAPER",
        "TICK_FAILED",
        "PAPER",
        "TICK_FAILED",
    ]
    deps.store.verify_chain()
    deps.store.close()


def test_the_same_tick_without_a_full_volume_reports_its_real_mode(
    books_dir: Path,
    caplog: pytest.LogCaptureFixture,
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    research_tools_factory: Callable[[], object],
    tmp_path: Path,
) -> None:
    """The control: an unimpeded real tick heartbeats PAPER and pages nobody.

    Without this the survival test above could pass against a loop that reports
    `TICK_FAILED` unconditionally. It is also #447's acceptance criterion (a)
    against the real kernel: the mode in the log line is `deps.kernel.mode`,
    which for this tick is the same `PAPER` the `ModeHeartbeat` row carries.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        caplog: The pytest log capture fixture.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to hand the hook the real dependency bundle.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        research_tools_factory: Builds the offline research tools.
        tmp_path: The per-test scratch directory.
    """
    caplog.set_level(logging.INFO)
    from windbreak.scheduler import loop as loop_module

    tick_ledger_path = ledger_path_for(tmp_path, "tick.db")
    alert_ledger_path = ledger_path_for(tmp_path, "alerts.db")
    deps = loop_module.build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=tick_ledger_path,
        report_dir=report_dir,
        config=paper_config,
        research_tools=research_tools_factory(),
        clock=_fixed_clock,
    )
    monkeypatch.setattr(loop_module, "build_paper_deps", lambda **_kwargs: deps)
    supervisor = BeatSupervisor(
        component="pipeline",
        dispatcher=AlertDispatcher(
            sinks=[],
            ledger_writer=LedgerAlertWriter(alert_ledger_path, component="pipeline"),
        ),
    )

    run_loop(
        0,
        max_beats=1,
        on_beat=_build_paper_on_beat(
            _paper_args(tmp_path, tick_ledger_path),
            paper_config,
            dispatcher=_log_only_dispatcher(),
        ),
        supervisor=supervisor,
    )

    assert [
        record.message
        for record in caplog.records
        if "heartbeat seq=" in record.message
    ] == ["mode=PAPER heartbeat seq=1"]
    assert not alert_ledger_path.exists()
    deps.store.verify_chain()
    deps.store.close()


def _ledgered_modes(ledger_path: Path) -> list[str]:
    """Read every ledgered `ModeHeartbeat` mode, verifying the chain first.

    Args:
        ledger_path: The tick ledger's path.

    Returns:
        One mode token per `ModeHeartbeat` row, in ledger order.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    return [
        str(json.loads(record.payload_json)["data"]["mode"])
        for record in records
        if record.event_type == "ModeHeartbeat"
    ]


def test_a_real_verification_breach_halts_the_kernel_and_pages_exactly_once(
    books_dir: Path,
    caplog: pytest.LogCaptureFixture,
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    research_tools_factory: Callable[[], object],
    tmp_path: Path,
) -> None:
    """#447's acceptance criteria driven by a *real* kernel reaching HALT.

    Every other halt assertion in this PR hands `run_loop` a hand-constructed
    `BeatReport(mode="HALT", halted=True)` or a kernel double. A double cannot
    fail at the seam the issue is about: whether a genuine breach reaches
    `deps.kernel.mode`, whether `_on_beat` projects it, and whether the loop
    prints it. So this drives the kernel to `Mode.HALT` the way production
    would -- an order placed on the exchange behind the loop's back, which
    `_verification_stage` grades as a mismatch on the next tick -- and then
    keeps beating, because the halt persisting across beats is what makes the
    once-per-transition dedupe load-bearing rather than incidental.

    The one configuration this test overrides is the consecutive-mismatch
    auto-kill bar, raised above its own beat budget. Since issue #441 that bar
    genuinely binds the PAPER loop's kernel, so at the `RiskConfig()` default of
    three the *third* consecutive breach below would engage the kill switch and
    the fourth beat would report `KILLED` rather than the third persistent
    `HALT` this test's subject needs. That is correct behaviour, not a
    regression, and `tests/integration/test_paper_kill_switch.py` pins it at two
    different thresholds; raising the bar here isolates the supervisor's
    escalation dedupe from it. `RiskConfig()`'s shipped default is untouched.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        caplog: The pytest log capture fixture.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to hand the hook the shared dependency bundle.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        research_tools_factory: Builds the offline research tools.
        tmp_path: The per-test scratch directory.
    """
    caplog.set_level(logging.INFO)
    from windbreak.numeric.types import MoneyMicros
    from windbreak.riskkernel.modes import Mode
    from windbreak.scheduler import loop as loop_module

    tick_ledger_path = ledger_path_for(tmp_path, "tick.db")
    alert_ledger_path = ledger_path_for(tmp_path, "alerts.db")
    config = dataclasses.replace(
        paper_config,
        risk=dataclasses.replace(
            paper_config.risk, kill_after_consecutive_mismatches=5
        ),
    )
    deps = loop_module.build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=tick_ledger_path,
        report_dir=report_dir,
        config=config,
        research_tools=research_tools_factory(),
        clock=_fixed_clock,
    )
    monkeypatch.setattr(loop_module, "build_paper_deps", lambda **_kwargs: deps)
    sink = _RecordingSink()
    supervisor = BeatSupervisor(
        component="pipeline",
        dispatcher=AlertDispatcher(
            sinks=[sink],
            ledger_writer=LedgerAlertWriter(alert_ledger_path, component="pipeline"),
        ),
    )
    hook = _build_paper_on_beat(
        _paper_args(tmp_path, tick_ledger_path),
        config,
        dispatcher=_log_only_dispatcher(),
    )

    def _on_beat(seq: int) -> BeatReport | None:
        """Take cash off the venue with no execution behind it, then beat.

        Not an order placed behind the loop's back: the bookkeeper reads the
        venue's own fill and arrival logs, so it books whatever landed there
        regardless of who placed it, and since issue #423 booked the
        resting-order collateral too, such a movement reconciles in full. Cash
        that simply left is the movement fill accounting genuinely cannot
        explain.

        Args:
            seq: The 1-based beat sequence number.

        Returns:
            Whatever the real PAPER hook reports for this beat.
        """
        if seq == 2:
            opening = deps.exchange.balances
            deps.exchange.balances = type(opening)(
                total=MoneyMicros(opening.total.value - 7_000_000),
                available=MoneyMicros(opening.available.value - 7_000_000),
                fetched_at=opening.fetched_at,
            )
        return hook(seq)

    run_loop(0, max_beats=4, on_beat=_on_beat, supervisor=supervisor)

    assert deps.kernel.mode is Mode.HALT
    assert [
        record.message
        for record in caplog.records
        if "heartbeat seq=" in record.message
    ] == [
        "mode=PAPER heartbeat seq=1",
        "mode=HALT heartbeat seq=2",
        "mode=HALT heartbeat seq=3",
        "mode=HALT heartbeat seq=4",
    ]
    assert [message for _type, _severity, message in sink.delivered] == [
        "risk kernel HALT at beat seq=2 (mode=HALT)"
    ]
    assert _alert_messages(alert_ledger_path) == [
        "risk kernel HALT at beat seq=2 (mode=HALT)"
    ]
    assert _ledgered_modes(tick_ledger_path) == ["PAPER", "HALT", "HALT", "HALT"]
    event_types = [record.event_type for record in deps.store.read_all()]
    assert "VerificationMismatch" in event_types
    assert "VerificationMismatchHalt" in event_types
    deps.store.verify_chain()
    deps.store.close()
