"""`windbreak kill` actually stops the always-on PAPER loop (issue #441).

#144 wired a real `KillIntegration` into `windbreak/main.py`'s
`_build_risk_kernel` -- but that composition is reached only by
`windbreak run --process riskkernel`. The **PAPER loop builds its own kernel**,
and until this suite that one was constructed with `kill_integration=None`, so
`windbreak kill --state-dir <dir>` wrote a `KILL` file the always-on loop never
polled and `risk.kill_after_consecutive_mismatches` bound nothing on the kernel
that was actually trading. For a safety-critical kill switch an inert one is
worse than none: `docs/RUNBOOK.md` documented a control the running loop did
not honour.

Every test here drives the *real* chain -- `windbreak.main._resolve_on_beat` ->
`_build_paper_on_beat` -> `build_paper_deps` -> `run_loop` -> `_on_beat` ->
`run_single_tick` -- and the *real* `windbreak kill` / `windbreak rearm` CLI
entrypoints, never a hand-written `KILL` file or a hand-built bundle. Nothing
here asserts `deps.kernel.kill_integration is not None`: a kill integration
that is constructed but never polled is indistinguishable from `None` at
runtime, and that is precisely the defect class this backlog exists to remove.
The assertions are on observable consequence -- the kernel's mode, the ledgered
`KillEngaged`/`CancelAllDirective`/`AlertEmitted` rows, the alert the operator's
own sink received, and the exact count of markets the loop researched after the
kill.

What each test pins, and why it is shaped that way:

1. `test_the_operators_kill_file_stops_the_running_paper_loop` -- the kill
   fires. The `KILL` file is written by the real CLI *between* beats of one
   `run_loop` call, which is the operator's actual scenario (a kill arriving at
   a loop already running), and the "stopped trading" evidence is an exact count
   of `MarketSnapshotRecorded`/`ForecastCreated` rows: exactly one, from the
   beat before the kill. Counting intents would prove nothing here -- the
   offline forecast abstains, so this fixture emits zero intents whether killed
   or not, and an assertion that reads the same both ways is the coincidence
   trap this suite must not fall into.
2. `test_a_paper_loop_with_no_kill_file_keeps_trading` -- the kill does *not*
   fire. A guard that only ever vetoes is as wrong as one that never does, so
   the same three beats with no `KILL` file are pinned to the same exact counts
   in the other direction.
3. `test_the_operators_rearm_file_returns_the_killed_loop_to_paused` -- the
   watcher's other half is live too, through the real `windbreak rearm` and its
   verbatim typed phrase.
4. `test_consecutive_reconciliation_mismatches_auto_kill_at_the_configured_bar`
   -- `risk.kill_after_consecutive_mismatches` is genuinely bound, driven at two
   different configured thresholds so a hardcoded constant cannot pass, with the
   mode after *every* beat pinned: the auto-kill fires on the Nth consecutive
   breach and not on the (N-1)th.
5. `test_a_kill_releases_the_paper_loops_active_capital_reservations` -- the one
   test here driven at the composition seam rather than through a tick, because
   the offline fixture never mints a token and so never takes a reservation for
   a kill to release; see its docstring for why an end-to-end assertion would
   have compared zero with zero.
6. `test_an_engaged_kill_survives_a_restart_with_the_kill_file_deleted` -- the
   kill state is durable ledgered state (issue #123), not the `KILL` file: a
   rebuilt bundle over the same hash chain comes back `KILLED` even after the
   file is gone, and re-arms only on the phrase for the *replayed* kill
   sequence.
7. `test_a_composed_kill_ledgers_the_same_delivery_evidence_on_both_alert_rows`
   -- the two `AlertEmitted` rows one kill writes are payload-*identical*
   (issue #488), so neither denies a delivery the other reports and an auditor
   needs no undocumented convention to pick between them. Read back through a
   second connection to the file, chain verified, because that is what an
   auditor gets rather than the handle the run left open.
8. `test_a_composed_kill_whose_sink_fails_ledgers_the_fallback_not_its_text` --
   the same rule on the unhappy path, where a `deliveries: []` row would be
   indistinguishable from the truth, plus the closure guarantee: the raised
   `str(exc)` never reaches `ledger.db` or its WAL sidecar and the delivery
   mappings carry no fourth key.

Deliberately not covered here, and stated so the omission is a decision rather
than an oversight: the one `CancelAllDirective` a kill ledgers is not delivered
to the order gateway, because no `directive_sink` is wired -- on this path or on
`windbreak/main.py`'s LIVE one. Resting-order cancellation on kill is an audit
record, not yet an effect.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from tests.integration.conftest import FIXED_NOW_EPOCH_S, ledger_path_for
from windbreak.alerts.dispatch import AlertDispatcher, LoggingLedgerWriter
from windbreak.alerts.registry import AlertSeverity, AlertType
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.numeric.types import MoneyMicros
from windbreak.riskkernel.modes import Mode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig
    from windbreak.scheduler.loop import PaperTickDeps

#: Cash removed from the paper account with no execution behind it -- the one
#: venue movement fill accounting genuinely cannot explain, and therefore the
#: only reliable way to drive a reconciliation breach.
_UNEXPLAINED_CASH_MICROS = 7_000_000

#: The exact body `windbreak.riskkernel.kill` dispatches and ledgers on a kill.
#: Compared in full rather than by substring, because a substring assertion
#: passes just as well against a message that grew a second, wrong half.
_KILL_MESSAGE = "kill switch engaged; trading halted, positions held"

#: The exact scope the one kill-path `CancelAllDirective` carries: resting
#: orders only, never open positions (the SPEC S10.12 position-hold invariant).
_CANCEL_ALL_SCOPE = "all_open_orders"

#: The credential-bearing text an exploding sink raises, standing in for the
#: whole category issue #274 leaked: an ntfy destination URL whose topic
#: segment *is* the capability to page this deployment. `SinkOutcome.detail` is
#: `str(exc)`, so this is exactly the string a careless ledger projection would
#: hash into an unredactable chain.
_SINK_FAILURE_DETAIL = "POST https://ntfy.example/s3kr1t-488-topic refused"

#: The screened identity a sink whose `name` is not one this codebase defines
#: is recorded under. The suite's doubles are named `recording`, which is not a
#: registered sink, so every configured-sink row here reads as this token.
_UNREGISTERED = "unregistered"

#: The one delivery a kill's dispatch records when its single configured sink
#: accepts the page: screened identity, enumerated outcome, not the fallback.
_DELIVERED_BY_CONFIGURED_SINK = {
    "sink": _UNREGISTERED,
    "outcome": "delivered",
    "fallback": False,
}

#: The two deliveries a kill's dispatch records when its single configured sink
#: raises: the configured sink's classified failure, then the `log-only`
#: fallback that carried the page instead. `fallback: True` is the difference
#: between "the operator configured log-only" and "every real channel failed".
_DELIVERIES_AFTER_A_FAILED_SINK = [
    {"sink": _UNREGISTERED, "outcome": "errored", "fallback": False},
    {"sink": "log-only", "outcome": "delivered", "fallback": True},
]

#: The exact key set a ledgered delivery mapping may carry. There is no fourth
#: key, and in particular no `detail` (issues #274/#413).
_DELIVERY_KEYS = {"sink", "outcome", "fallback"}


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


@dataclass
class _ExplodingSink:
    """An `AlertSink` double that always raises token-bearing text.

    The unhappy half of the delivery record. Its exception message is the
    `str(exc)` that becomes `SinkOutcome.detail`, so it is simultaneously the
    thing the ledger must *summarise* (as an enumerated `errored`) and the
    thing the ledger must never *contain*.

    Attributes:
        name: The sink's identifier, as the `AlertSink` protocol requires.
        attempts: One `(type, severity, message)` triple per attempt, so a sink
            that was never reached is distinguishable from one that failed.
    """

    name: str = "recording"
    attempts: list[tuple[AlertType, AlertSeverity, str]] = field(default_factory=list)

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Record the attempt, then fail it with credential-bearing detail.

        Args:
            alert_type: The dispatched alert type.
            severity: The alert's severity.
            message: The alert body.

        Raises:
            RuntimeError: Always, carrying :data:`_SINK_FAILURE_DETAIL`.
        """
        self.attempts.append((alert_type, severity, message))
        raise RuntimeError(_SINK_FAILURE_DETAIL)


def _fixed_clock() -> int:
    """Return the suite's fixed epoch second, so a tick is deterministic.

    Returns:
        The shared fixture epoch second.
    """
    return FIXED_NOW_EPOCH_S


def _paper_args(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
) -> argparse.Namespace:
    """Build the `run` namespace the PAPER composition root reads its flags from.

    Args:
        books_dir: The paper-exchange fixture directory.
        cassette_path: The empty offline cassette.
        ledger_path: The tick ledger's path.
        report_dir: Where weekly-report stubs are written.

    Returns:
        A namespace carrying the four PAPER flags, no live ticker, and the
        `--process` token the alert root stamps its ledgered rows with.
    """
    return argparse.Namespace(
        paper_books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
        paper_live_ticker=None,
        process="pipeline",
    )


@pytest.fixture
def captured_deps(
    monkeypatch: pytest.MonkeyPatch, research_tools_factory: Callable[[], Any]
) -> Iterator[list[PaperTickDeps]]:
    """Capture the `PaperTickDeps` the real composition root builds.

    The wrapper is a *pass-through*, not a stand-in: it forwards every keyword
    `windbreak.main` supplies -- `dispatcher` above all -- to the real
    `build_paper_deps`, and only adds the two offline seams (`clock`,
    `research_tools`) that `_build_paper_on_beat` leaves at their wall-clock
    and network defaults.

    Args:
        monkeypatch: Used to install the pass-through wrapper.
        research_tools_factory: Builds the offline research tools double.

    Yields:
        The list the built bundle is appended to, one entry per build.
    """
    from windbreak.scheduler import loop as loop_module

    built: list[PaperTickDeps] = []
    real_build = loop_module.build_paper_deps

    def _capture(**kwargs: Any) -> PaperTickDeps:
        """Build the real bundle offline, recording it.

        Args:
            **kwargs: Every keyword the production call site supplied,
                forwarded verbatim.

        Returns:
            The real `PaperTickDeps`.
        """
        deps = real_build(
            **kwargs, clock=_fixed_clock, research_tools=research_tools_factory()
        )
        built.append(deps)
        return deps

    monkeypatch.setattr(loop_module, "build_paper_deps", _capture)
    yield built
    for deps in built:
        deps.store.close()


def _event_types(deps: PaperTickDeps) -> list[str]:
    """Return every ledgered event type, in ledger order.

    Args:
        deps: The wired bundle whose store is read.

    Returns:
        One event-type string per ledger row.
    """
    return [record.event_type for record in deps.store.read_all()]


def _payloads_of(deps: PaperTickDeps, event_type: str) -> list[dict[str, Any]]:
    """Return the payload `data` of every row of one event type, in order.

    Args:
        deps: The wired bundle whose store is read.
        event_type: The event type to project.

    Returns:
        One payload-data mapping per matching ledger row.
    """
    return [
        json.loads(record.payload_json)["data"]
        for record in deps.store.read_all()
        if record.event_type == event_type
    ]


def _alert_rows(deps: PaperTickDeps) -> list[tuple[str, str, str]]:
    """Return every ledgered alert as `(component, severity, message)`, in order.

    The component is projected deliberately. A kill writes *two* `AlertEmitted`
    rows, and they are not duplicates: the run's alert root ledgers the
    dispatch under the `--process` token, and `KillSwitch.kill` ledgers its own
    tamper-evident audit row under `riskkernel` after every fail-safe effect has
    landed (issue #287). Asserting on payloads alone could not tell one from the
    other, so a regression that dropped either would read as healthy.

    Args:
        deps: The wired bundle whose store is read.

    Returns:
        One triple per `AlertEmitted` row, in ledger order.
    """
    return [
        (
            str(record.component),
            str(json.loads(record.payload_json)["data"]["severity"]),
            str(json.loads(record.payload_json)["data"]["message"]),
        )
        for record in deps.store.read_all()
        if record.event_type == "AlertEmitted"
    ]


def _alert_rows_off_disk(ledger_path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Reopen the ledger *file* and return every `AlertEmitted` row, in order.

    Deliberately a second connection rather than `deps.store`: the assertion an
    auditor's reading has to survive is the one made against the database on
    disk, not against the handle the run happened to leave open. The chain is
    verified before anything is read, so a row that was appended out of chain
    fails here rather than being quietly projected.

    Args:
        ledger_path: The ledger database the run was given.

    Returns:
        One `(component, payload-data)` pair per `AlertEmitted` row, in ledger
        order. The component is kept because it is the *only* field the two
        rows one kill writes differ by.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    return [
        (str(record.component), dict(json.loads(record.payload_json)["data"]))
        for record in records
        if record.event_type == "AlertEmitted"
    ]


def _ledger_file_bytes(ledger_path: Path) -> bytes:
    """Return every byte of the ledger database *and* its SQLite sidecars.

    The store is WAL-journaled, so a freshly appended row lives in
    `ledger.db-wal` and not yet in `ledger.db`. A sweep that read only the
    latter would scan a file the new rows are not in and pass forever -- the
    false green a secrets test hit in PR #474.

    Args:
        ledger_path: The ledger database the run was given.

    Returns:
        The concatenation of `ledger.db` and every `ledger.db*` sidecar, in
        sorted name order.
    """
    return b"".join(
        path.read_bytes()
        for path in sorted(ledger_path.parent.glob(f"{ledger_path.name}*"))
    )


def _count_of(deps: PaperTickDeps, event_type: str) -> int:
    """Return how many rows of one event type the ledger holds.

    Args:
        deps: The wired bundle whose store is read.
        event_type: The event type to count.

    Returns:
        The exact row count.
    """
    return _event_types(deps).count(event_type)


def _move_the_venue_behind_the_loops_back(deps: PaperTickDeps) -> None:
    """Take cash off the exchange with no execution behind it.

    The same divergence `tests/integration/test_paper_verification.py` uses to
    drive a breach, and the only kind that still is one. *Trading* directly on
    the exchange is not: the bookkeeper reads the venue's own fill and arrival
    logs, so it books whatever landed there regardless of who placed it, and
    since issue #423 booked the resting-order collateral too, the whole
    movement reconciles. Cash that simply left explains nothing. It is taken off
    the opening balance every later reading derives from, so the divergence
    *persists* and every later cycle grades `BREACH` -- which is what makes a
    run of consecutive mismatches drivable at all.

    Args:
        deps: The wired bundle whose exchange is moved.
    """
    opening = deps.exchange.balances
    deps.exchange.balances = type(opening)(
        total=MoneyMicros(opening.total.value - _UNEXPLAINED_CASH_MICROS),
        available=MoneyMicros(opening.available.value - _UNEXPLAINED_CASH_MICROS),
        fetched_at=opening.fetched_at,
    )


def _run_beats(
    args: argparse.Namespace,
    config: WindbreakConfig,
    *,
    beats: int,
    between_beats: Callable[[int], None] | None = None,
) -> list[str]:
    """Run `beats` beats of the real loop, returning the mode after each.

    Drives the production chain end to end: the run's one composed alert root,
    the real `_resolve_on_beat` -> `_build_paper_on_beat` bundle, the real
    `BeatSupervisor`, and `run_loop` itself. `between_beats` is invoked with the
    just-finished beat number *after* each beat's hook returns, which is where
    an operator's `windbreak kill` lands: at a loop that is already running.

    Args:
        args: The `run` namespace carrying the PAPER flags.
        config: The loaded PAPER-ceilinged configuration.
        beats: How many beats to run.
        between_beats: Optional hook invoked after each beat with its 1-based
            sequence number.

    Returns:
        The kernel mode name observed after each beat, in beat order.
    """
    from windbreak.main import (
        _beat_alert_root,
        _build_beat_supervisor,
        _resolve_on_beat,
        run_loop,
    )

    alert_root = _beat_alert_root(args, config)
    on_beat = _resolve_on_beat(args, config, alert_root=alert_root)
    assert on_beat is not None
    modes: list[str] = []

    def _observed(seq: int) -> Any:
        """Run one real beat, record the mode it reported, then step the world.

        Args:
            seq: The 1-based beat sequence number.

        Returns:
            The real hook's `BeatReport`.
        """
        report = on_beat(seq)
        assert report is not None
        modes.append(report.mode)
        if between_beats is not None:
            between_beats(seq)
        return report

    run_loop(
        0.0,
        max_beats=beats,
        on_beat=_observed,
        supervisor=_build_beat_supervisor(args, dispatcher=alert_root()),
        component=args.process,
    )
    return modes


def test_the_operators_kill_file_stops_the_running_paper_loop(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    state_dir: Path,
    tmp_path: Path,
) -> None:
    """`windbreak kill` halts the always-on PAPER loop (issue #441's criterion).

    The `KILL` file is written by the real CLI after beat 1 of a three-beat
    `run_loop`, exactly as an operator would run it against a live daemon. The
    evidence is what the loop then *did*: one `MarketSnapshotRecorded` and one
    `ForecastCreated` in the whole run (beat 1's, before the kill), a
    `KILLED` mode heartbeat on beats 2 and 3, the closed kill-effect surface on
    the hash chain, and the `HALT_KILL` page in the operator's own sink.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures the bundle the composition root builds.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to make `build_sinks` yield the recording sink.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        state_dir: The kill/re-arm state directory `paper_config` points at.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.main import main

    sink = _RecordingSink()
    monkeypatch.setattr("windbreak.main.build_sinks", lambda *_a, **_k: (sink,))
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
    )

    def _operator_kills(seq: int) -> None:
        """Run the real `windbreak kill` CLI once, after the first beat.

        Args:
            seq: The just-finished beat's sequence number.
        """
        if seq == 1:
            assert main(["kill", "--state-dir", str(state_dir)]) == 0

    modes = _run_beats(args, paper_config, beats=3, between_beats=_operator_kills)

    deps = captured_deps[0]
    assert modes == ["PAPER", "KILLED", "KILLED"]
    assert deps.kernel.mode is Mode.KILLED
    assert [row["mode"] for row in _payloads_of(deps, "ModeHeartbeat")] == [
        "PAPER",
        "KILLED",
        "KILLED",
    ]
    assert _payloads_of(deps, "KillEngaged") == [
        {"trigger": "KILL_FILE", "kill_sequence": 1, "epoch": FIXED_NOW_EPOCH_S}
    ]
    assert _payloads_of(deps, "CancelAllDirective") == [{"scope": _CANCEL_ALL_SCOPE}]
    assert _alert_rows(deps) == [
        ("pipeline", AlertSeverity.CRITICAL.value, _KILL_MESSAGE),
        ("riskkernel", AlertSeverity.CRITICAL.value, _KILL_MESSAGE),
    ]
    assert sink.delivered == [
        (AlertType.HALT_KILL, AlertSeverity.CRITICAL, _KILL_MESSAGE)
    ]
    assert _count_of(deps, "MarketSnapshotRecorded") == 1
    assert _count_of(deps, "ForecastCreated") == 1
    assert _count_of(deps, "SelectorDecisionRecorded") == 1


def test_a_pending_kill_pre_empts_the_same_beats_verification_halt(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    state_dir: Path,
    tmp_path: Path,
) -> None:
    """The kill poll runs *before* the tick's verification cycle, not after.

    This pins the stage order, which is otherwise unobservable: whether the
    universe walk is skipped is read after the cycle either way, so a poll moved
    below `_verification_stage` would still stop the same beat, and a docstring
    claiming "first" would be claiming more than anything checked.

    The consequence that distinguishes them is what the beat *records*. With a
    `KILL` file already on disk the kernel is `KILLED` before the cycle runs, so
    `_halt_on_breach`'s dead-end guard applies and no `VerificationMismatchHalt`
    is written -- the loop never transitioned through `HALT` and its audit trail
    must not claim it did. Polled after the cycle, the same beat would halt
    first and then be killed, ledgering a halt that never happened. The
    `VerificationMismatch` row is still written in both readings, and that
    matters: a kill silences the *transition*, never the evidence.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures the bundle the composition root builds.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to make `build_sinks` yield the recording sink.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        state_dir: The kill/re-arm state directory `paper_config` points at.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.main import main

    monkeypatch.setattr(
        "windbreak.main.build_sinks", lambda *_a, **_k: (_RecordingSink(),)
    )
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
    )

    def _diverge_and_kill(seq: int) -> None:
        """Move the venue and engage the kill, both after the first beat.

        Args:
            seq: The just-finished beat's sequence number.
        """
        if seq == 1:
            _move_the_venue_behind_the_loops_back(captured_deps[0])
            assert main(["kill", "--state-dir", str(state_dir)]) == 0

    modes = _run_beats(args, paper_config, beats=2, between_beats=_diverge_and_kill)

    deps = captured_deps[0]
    assert modes == ["PAPER", "KILLED"]
    assert _count_of(deps, "VerificationMismatch") == 1
    assert _count_of(deps, "VerificationMismatchHalt") == 0
    assert [row["trigger"] for row in _payloads_of(deps, "KillEngaged")] == [
        "KILL_FILE"
    ]


def test_a_paper_loop_with_no_kill_file_keeps_trading(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    state_dir: Path,
    tmp_path: Path,
) -> None:
    """With no `KILL` file the very same three beats research every beat.

    The other direction of issue #441's acceptance bar. A kill switch that fires
    when it should is half the requirement; one that does *not* fire when the
    operator has asked for nothing is the other half, and a guard that only ever
    vetoes is as wrong as one that never does. Every count below is exact.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures the bundle the composition root builds.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to make `build_sinks` yield the recording sink.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        state_dir: The kill/re-arm state directory `paper_config` points at.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.riskkernel.kill import KILL_FILENAME

    sink = _RecordingSink()
    monkeypatch.setattr("windbreak.main.build_sinks", lambda *_a, **_k: (sink,))
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
    )

    modes = _run_beats(args, paper_config, beats=3)

    deps = captured_deps[0]
    assert not state_dir.joinpath(KILL_FILENAME).exists()
    assert modes == ["PAPER", "PAPER", "PAPER"]
    assert deps.kernel.mode is Mode.PAPER
    assert [row["mode"] for row in _payloads_of(deps, "ModeHeartbeat")] == [
        "PAPER",
        "PAPER",
        "PAPER",
    ]
    assert _payloads_of(deps, "KillEngaged") == []
    assert _payloads_of(deps, "CancelAllDirective") == []
    assert _alert_rows(deps) == []
    assert sink.delivered == []
    assert _count_of(deps, "MarketSnapshotRecorded") == 3
    assert _count_of(deps, "ForecastCreated") == 3
    assert _count_of(deps, "SelectorDecisionRecorded") == 3


def test_the_operators_rearm_file_returns_the_killed_loop_to_paused(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    state_dir: Path,
    tmp_path: Path,
) -> None:
    """`windbreak rearm`'s typed phrase brings the killed loop back to `PAUSED`.

    The kill file watcher's other half, driven through the real CLI: the phrase
    is typed verbatim on stdin exactly as the RUNBOOK instructs, the loop
    re-arms on the next beat, and the now-stale `KILL` file is cleared so the
    beat after that does not instantly re-kill.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures the bundle the composition root builds.
        cassette_path: The empty offline cassette.
        monkeypatch: Used for `build_sinks` and the typed re-arm phrase.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        state_dir: The kill/re-arm state directory `paper_config` points at.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.main import main
    from windbreak.riskkernel.kill import KILL_FILENAME, REARM_FILENAME

    monkeypatch.setattr(
        "windbreak.main.build_sinks", lambda *_a, **_k: (_RecordingSink(),)
    )
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
    )

    def _operator_kills_then_rearms(seq: int) -> None:
        """Kill after beat 1 and type the re-arm phrase after beat 2.

        Args:
            seq: The just-finished beat's sequence number.
        """
        if seq == 1:
            assert main(["kill", "--state-dir", str(state_dir)]) == 0
        if seq == 2:
            monkeypatch.setattr(
                "builtins.input",
                lambda: "RE-ARM KILL 1: I ACCEPT FULL RESPONSIBILITY",
            )
            assert main(["rearm", "--state-dir", str(state_dir)]) == 0

    modes = _run_beats(
        args, paper_config, beats=4, between_beats=_operator_kills_then_rearms
    )

    deps = captured_deps[0]
    assert modes == ["PAPER", "KILLED", "PAUSED", "PAUSED"]
    assert deps.kernel.mode is Mode.PAUSED
    assert _payloads_of(deps, "KillReArmed") == [{"kill_sequence": 1}]
    assert not state_dir.joinpath(KILL_FILENAME).exists()
    assert not state_dir.joinpath(REARM_FILENAME).exists()


@pytest.mark.parametrize(
    ("threshold", "expected_modes"),
    [
        (2, ["PAPER", "HALT", "KILLED", "KILLED"]),
        (3, ["PAPER", "HALT", "HALT", "KILLED"]),
    ],
)
def test_consecutive_reconciliation_mismatches_auto_kill_at_the_configured_bar(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    expected_modes: list[str],
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    state_dir: Path,
    threshold: int,
    tmp_path: Path,
) -> None:
    """`risk.kill_after_consecutive_mismatches` binds the PAPER loop's auto-kill.

    Two thresholds, because a single one cannot tell a bound configuration from
    a hardcoded constant. The mode after *every* beat is pinned, so the
    off-by-one is nailed in both directions: at a threshold of N the auto-kill
    fires on the Nth consecutive breach and the (N-1)th leaves the kernel merely
    `HALT`ed. The venue is moved once, after beat 1, and the divergence persists,
    so beats 2..4 each grade one `BREACH`.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures the bundle the composition root builds.
        cassette_path: The empty offline cassette.
        expected_modes: The mode expected after each of the four beats.
        monkeypatch: Used to make `build_sinks` yield the recording sink.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        state_dir: The kill/re-arm state directory `paper_config` points at.
        threshold: The configured consecutive-mismatch auto-kill bar.
        tmp_path: The per-test scratch directory.
    """
    import dataclasses

    from windbreak.riskkernel.kill import KILL_FILENAME

    sink = _RecordingSink()
    monkeypatch.setattr("windbreak.main.build_sinks", lambda *_a, **_k: (sink,))
    config = dataclasses.replace(
        paper_config,
        risk=dataclasses.replace(
            paper_config.risk, kill_after_consecutive_mismatches=threshold
        ),
    )
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
    )

    def _move_the_venue_once(seq: int) -> None:
        """Diverge the venue from the ledger after the first beat.

        Args:
            seq: The just-finished beat's sequence number.
        """
        if seq == 1:
            _move_the_venue_behind_the_loops_back(captured_deps[0])

    modes = _run_beats(args, config, beats=4, between_beats=_move_the_venue_once)

    deps = captured_deps[0]
    assert modes == expected_modes
    assert deps.kernel.mode is Mode.KILLED
    # An auto-kill leaves the same durable on-disk signal the operator's own
    # `windbreak kill` writes, so a restart with no readable ledger still comes
    # back dead. The switch creates the state directory to do it -- this one has
    # never existed until now.
    assert state_dir.joinpath(KILL_FILENAME).read_text(encoding="utf-8") == ""
    assert _payloads_of(deps, "KillEngaged") == [
        {
            "trigger": "AUTO_RECONCILIATION",
            "kill_sequence": 1,
            "epoch": FIXED_NOW_EPOCH_S,
        }
    ]
    # Filtered by *message* rather than compared whole, and deliberately not by
    # type: the `BeatSupervisor`'s own kernel-HALT escalation on the earlier
    # beats carries the same `HALT_KILL` type, so filtering on the type would
    # not isolate the kill at all. Within the kill's own message the count is
    # exact -- one page, never "at least one" -- so a kill that paged twice, or
    # once per later beat, fails here.
    assert [
        delivered for delivered in sink.delivered if delivered[2] == _KILL_MESSAGE
    ] == [(AlertType.HALT_KILL, AlertSeverity.CRITICAL, _KILL_MESSAGE)]


def test_a_kill_releases_the_paper_loops_active_capital_reservations(
    paper_config: WindbreakConfig,
) -> None:
    """A killed PAPER loop holds no live capital reservation (SPEC S10.12).

    Driven at the composition seam rather than through a whole tick, and
    deliberately so: this suite's offline forecast abstains, so the end-to-end
    fixture never mints an approval token and therefore never takes a
    reservation for a kill to release. An end-to-end assertion here would
    compare zero against zero and pass whether or not the reservation ledger
    were wired -- exactly the coincidence this suite exists to avoid. So the
    *production* composition function is called with a real `ReservationLedger`
    holding a real reservation, and the assertion is on the released row, its
    exact reason, and the state the ledger is left in.

    Args:
        paper_config: The PAPER-ceilinged configuration.
    """
    from windbreak.numeric import MoneyMicros
    from windbreak.riskkernel.kill import KillTrigger
    from windbreak.riskkernel.modes import ModeStateMachine
    from windbreak.riskkernel.process import InMemoryKernelLedgerWriter
    from windbreak.riskkernel.reservations import ReservationLedger
    from windbreak.scheduler.loop import _build_kill_integration

    writer = InMemoryKernelLedgerWriter()
    reservations = ReservationLedger(writer)
    reservations.reserve(
        "intent-441",
        MoneyMicros(2_500_000),
        "idem-441",
        expires_at=FIXED_NOW_EPOCH_S + 60,
    )
    integration = _build_kill_integration(
        paper_config,
        (),
        ModeStateMachine(mode_ceiling=Mode.PAPER, mode=Mode.PAPER),
        writer,
        _fixed_clock,
        AlertDispatcher(sinks=[], ledger_writer=LoggingLedgerWriter()),
        reservations,
    )

    integration.switch.kill(KillTrigger.CLI)

    assert [
        event.payload
        for event in writer.events
        if event.event_type == "ReservationReleased"
    ] == [{"intent_id": "intent-441", "reason": "kill_switch_engaged"}]
    assert reservations.total_reserved() == MoneyMicros(0)


def test_an_engaged_kill_survives_a_restart_with_the_kill_file_deleted(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    state_dir: Path,
    tmp_path: Path,
) -> None:
    """A killed PAPER loop comes back killed, from the ledger, not the file.

    Kill state is durable ledgered state (issue #123); the `KILL` file is
    belt-and-suspenders. So a second bundle built over the same hash chain --
    with the file deleted, so only the ledger can supply the answer -- must
    start `KILLED` rather than cheerfully resuming trading. Absent evidence must
    never read as healthy, and a deleted file is not a re-arm.

    The restarted loop is then re-armed through the real CLI with the phrase for
    kill sequence *1*, which is what proves the rebuilt `KillSwitch` restored its
    monotonic sequence from the chain rather than starting fresh at 0. Without
    that replay the phrase is rejected and the loop stays `KILLED`.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures the bundles the composition root builds.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to make `build_sinks` yield the recording sink.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        state_dir: The kill/re-arm state directory `paper_config` points at.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.main import main
    from windbreak.riskkernel.kill import KILL_FILENAME

    monkeypatch.setattr(
        "windbreak.main.build_sinks", lambda *_a, **_k: (_RecordingSink(),)
    )
    ledger_path = ledger_path_for(tmp_path)
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
    )

    def _operator_kills(seq: int) -> None:
        """Run the real `windbreak kill` CLI once, after the first beat.

        Args:
            seq: The just-finished beat's sequence number.
        """
        if seq == 1:
            assert main(["kill", "--state-dir", str(state_dir)]) == 0

    def _operator_rearms(seq: int) -> None:
        """Type the re-arm phrase for kill sequence 1 after the restart's beat 1.

        The phrase is the load-bearing half of this test. `KillSwitch.from_events`
        restores the *monotonic kill sequence* from the same chain, so a rebuilt
        switch expects `RE-ARM KILL 1`. A switch rebuilt without that replay
        would expect `RE-ARM KILL 0` and reject this phrase, leaving the kernel
        `KILLED` -- which is how this test tells a genuinely replayed switch from
        a fresh one that merely happens to sit beside a replayed-`KILLED` kernel.

        Args:
            seq: The just-finished beat's sequence number.
        """
        if seq == 1:
            monkeypatch.setattr(
                "builtins.input",
                lambda: "RE-ARM KILL 1: I ACCEPT FULL RESPONSIBILITY",
            )
            assert main(["rearm", "--state-dir", str(state_dir)]) == 0

    _run_beats(args, paper_config, beats=2, between_beats=_operator_kills)
    captured_deps[0].store.close()
    state_dir.joinpath(KILL_FILENAME).unlink()

    restarted = _run_beats(args, paper_config, beats=2, between_beats=_operator_rearms)

    assert not state_dir.joinpath(KILL_FILENAME).exists()
    assert restarted == ["KILLED", "PAUSED"]
    assert captured_deps[1].kernel.mode is Mode.PAUSED
    assert _payloads_of(captured_deps[1], "KillReArmed") == [{"kill_sequence": 1}]
    assert _count_of(captured_deps[1], "MarketSnapshotRecorded") == 2


def test_a_composed_kill_ledgers_the_same_delivery_evidence_on_both_alert_rows(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    state_dir: Path,
    tmp_path: Path,
) -> None:
    """Neither `AlertEmitted` row a kill writes denies the delivery (issue #488).

    One kill appends *two* `AlertEmitted` rows to the one chain, and both are
    load-bearing: the run's alert root ledgers the dispatch under the
    `--process` token, and `KillSwitch.kill` ledgers its own tamper-evident
    audit row under `riskkernel` after every fail-safe effect has landed
    (issue #287). Removing either would regress a landed guarantee, so two rows
    stay -- and the reader therefore needs a rule for which one to believe.

    The rule pinned here is the strongest available and the only one that needs
    no convention at all: **the two rows carry byte-identical payloads**, so
    either is authoritative and `component` records provenance rather than
    precedence. That holds by construction -- both project the *same*
    `AlertEmitted` object through `ledger_deliveries`, the single producer of
    ledgered delivery evidence -- and it is asserted here rather than merely
    documented, because a convention an auditor has to be told is not a rule
    the chain enforces.

    Before issue #488, `LedgerAlertWriter` dropped the report it was handed and
    stamped `deliveries: []`, `delivery_reported: false` -- not the silence a
    schema-1 row kept, but an explicit denial of evidence that was sitting in
    its own parameter. An auditor scanning by `event_type` rather than by
    `component` read a false negative for the very page that was delivered.

    Read off disk through a second connection, chain verified, because that is
    what an auditor gets.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures the bundle the composition root builds.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to make `build_sinks` yield the recording sink.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        state_dir: The kill/re-arm state directory `paper_config` points at.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.main import main

    sink = _RecordingSink()
    monkeypatch.setattr("windbreak.main.build_sinks", lambda *_a, **_k: (sink,))
    ledger_path = ledger_path_for(tmp_path)
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
    )

    def _operator_kills(seq: int) -> None:
        """Run the real `windbreak kill` CLI once, after the first beat.

        Args:
            seq: The just-finished beat's sequence number.
        """
        if seq == 1:
            assert main(["kill", "--state-dir", str(state_dir)]) == 0

    modes = _run_beats(args, paper_config, beats=2, between_beats=_operator_kills)

    delivered_payload = {
        "severity": AlertSeverity.CRITICAL.value,
        "message": _KILL_MESSAGE,
        "deliveries": [_DELIVERED_BY_CONFIGURED_SINK],
        "delivery_reported": True,
    }
    rows = _alert_rows_off_disk(ledger_path)
    assert modes == ["PAPER", "KILLED"]
    assert captured_deps[0].kernel.mode is Mode.KILLED
    assert sink.delivered == [
        (AlertType.HALT_KILL, AlertSeverity.CRITICAL, _KILL_MESSAGE)
    ]
    assert rows == [
        ("pipeline", delivered_payload),
        ("riskkernel", delivered_payload),
    ]


def test_a_composed_kill_whose_sink_fails_ledgers_the_fallback_not_its_text(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    state_dir: Path,
    tmp_path: Path,
) -> None:
    """The page that reached nobody real says so, in the closed vocabulary.

    The auditor's case that actually matters, and the one a `deliveries: []`
    row is indistinguishable from. The single configured sink raises, so the
    `log-only` fallback carries the kill page instead; both rows must record
    that -- `errored` for the configured sink, `delivered`/`fallback: true` for
    the fallback -- and both must record it *identically*, so the two-row
    reading rule holds on the unhappy path too and not only when everything
    worked.

    It is also the proof that forwarding the report did not widen what reaches
    the chain. `SinkOutcome.detail` is `str(exc)`, the shape that leaked whole
    token-bearing URLs in issue #274, and a hash chain can never be redacted.
    So the raised text is swept for across `ledger.db` *and* its WAL sidecar,
    and every delivery mapping's key set is pinned closed -- no `detail`, no
    fourth key. The sweep carries its own positive control: the kill message is
    asserted *present* in the same bytes, so a sweep that was scanning an empty
    or wrong corpus fails rather than passing forever.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures the bundle the composition root builds.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to make `build_sinks` yield the exploding sink.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        state_dir: The kill/re-arm state directory `paper_config` points at.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.main import main

    sink = _ExplodingSink()
    monkeypatch.setattr("windbreak.main.build_sinks", lambda *_a, **_k: (sink,))
    ledger_path = ledger_path_for(tmp_path)
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
    )

    def _operator_kills(seq: int) -> None:
        """Run the real `windbreak kill` CLI once, after the first beat.

        Args:
            seq: The just-finished beat's sequence number.
        """
        if seq == 1:
            assert main(["kill", "--state-dir", str(state_dir)]) == 0

    modes = _run_beats(args, paper_config, beats=2, between_beats=_operator_kills)

    failed_payload = {
        "severity": AlertSeverity.CRITICAL.value,
        "message": _KILL_MESSAGE,
        "deliveries": _DELIVERIES_AFTER_A_FAILED_SINK,
        "delivery_reported": True,
    }
    rows = _alert_rows_off_disk(ledger_path)
    on_disk = _ledger_file_bytes(ledger_path)
    assert modes == ["PAPER", "KILLED"]
    assert captured_deps[0].kernel.mode is Mode.KILLED
    assert sink.attempts == [
        (AlertType.HALT_KILL, AlertSeverity.CRITICAL, _KILL_MESSAGE)
    ]
    assert rows == [
        ("pipeline", failed_payload),
        ("riskkernel", failed_payload),
    ]
    assert [
        set(delivery)
        for _component, payload in rows
        for delivery in payload["deliveries"]
    ] == [_DELIVERY_KEYS, _DELIVERY_KEYS, _DELIVERY_KEYS, _DELIVERY_KEYS]
    assert _KILL_MESSAGE.encode() in on_disk
    assert _SINK_FAILURE_DETAIL.encode() not in on_disk
    assert b"s3kr1t-488" not in on_disk
