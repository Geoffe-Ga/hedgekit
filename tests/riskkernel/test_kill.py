"""Failing-first tests for windbreak.riskkernel.kill (issue #35, RED).

Issue #35 gives the Risk Kernel a single kill executor (:class:`KillSwitch`)
reachable from four triggers -- CLI, a `KILL` state-dir file, a dashboard
challenge/confirm handshake, and consecutive reconciliation-mismatch
auto-detection -- plus the typed-confirmation re-arm procedure that is the
*only* way back out of `KILLED`. `windbreak/riskkernel/kill.py` does not exist
yet, so the import below fails the whole module at collection with
`ModuleNotFoundError: No module named 'windbreak.riskkernel.kill'` -- the
expected Gate 1 RED state for issue #35. This file also pins three new
ledger events (`KillEngaged`, `CancelAllDirective`, `KillReArmed` in
`windbreak/ledger/events.py`), a new `RiskConfig.kill_after_consecutive_mismatches`
config field, a `RiskKernel`-level `KILLED` hard-veto and `kill_integration`
wiring, and a `windbreak kill` / `windbreak rearm` CLI pair -- none of which
exist yet either, so several individual imports below would independently
fail collection too (an `ImportError` on the not-yet-defined ledger event
classes, in particular) even once `kill.py` exists on its own.

Once every piece lands, this file pins: the kill-effect surface as a
*closed* set of event types (`KillEngaged`, `CancelAllDirective`,
`ReservationReleased`) that never carries a sell/close/submit/dump action;
kill/re-arm idempotency; the position-hold no-dump invariant under Hypothesis;
and the full SPEC S10.12 kill drill (open reservations, mid-run KILL-file
kill, then REARM-file re-arm restoring approval capability with no stale
reservation replay).

Issue #123 (durable kill state via ledger replay on kernel rebuild) adds a
pure fold, `kill_state_in`, returning a new `ReplayedKillState` dataclass;
`KillSwitch.from_events`, a classmethod restoring `active_kill_sequence` from
a replayed history; and extends `RiskKernel.from_events` (issue #33's
override-replay entrypoint) to also replay kill state, driving the shared
mode machine to `KILLED` on an unrearmed kill history. `ReplayedKillState`
and `kill_state_in` do not exist on the real, not-yet-updated `kill.py`
module yet, so the `from windbreak.riskkernel.kill import ...` block below
independently fails collection with `ImportError: cannot import name
'ReplayedKillState' from 'windbreak.riskkernel.kill'` -- the expected Gate 1
RED state for issue #123, on top of (and however the other pieces above have
already landed) whatever the rest of this file's imports pin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.riskkernel.conftest import make_context, make_intent
from windbreak.alerts.dispatch import AlertDispatcher, LoggingLedgerWriter
from windbreak.alerts.registry import AlertType
from windbreak.config import RiskConfig
from windbreak.connector.fake import FakeExchange
from windbreak.ledger.events import (
    EVENT_TYPES,
    CancelAllDirective,
    Event,
    KillEngaged,
    KillReArmed,
    ModeHeartbeat,
    SignificanceOverrideApplied,
)
from windbreak.main import main as windbreak_main
from windbreak.numeric.types import ContractCentis, MoneyMicros
from windbreak.riskkernel.demotion import DemotionTrigger
from windbreak.riskkernel.kill import (
    DashboardChallengeError,
    DashboardKillStub,
    KillFileWatcher,
    KillIntegration,
    KillSwitch,
    KillTrigger,
    ReconciliationMismatchMonitor,
    ReplayedKillState,
    kill_state_in,
)
from windbreak.riskkernel.modes import (
    REARM_CONFIRMATION_PHRASE,
    KillReArmError,
    Mode,
    ModeStateMachine,
)
from windbreak.riskkernel.process import InMemoryKernelLedgerWriter, RiskKernel
from windbreak.riskkernel.promotion import SIGNIFICANCE_OVERRIDE_ACK_PHRASE
from windbreak.riskkernel.reservations import (
    DuplicateReservationError,
    ReservationLedger,
)
from windbreak.riskkernel.verification import (
    LedgerExpectations,
    ReadOnlyVerifier,
    VerificationOutcome,
    VerificationTolerances,
)

#: The fixed "current instant" every `KillSwitch` built by `_build_switch`
#: reports, via an injected clock, so every `KillEngaged.epoch` assertion below
#: is an exact, deterministic int rather than an `isinstance` check against
#: real wall-clock time.
_FIXED_EPOCH_S = 1_700_000_000

#: A representative expiry far enough past `_FIXED_EPOCH_S` that no
#: reservation created in a test ever expires from under it.
_FAR_FUTURE_EXPIRY_S = 2_000_000_000

#: The kill sequence the issue #123 restart/restore tests pin as "the
#: sequence recorded before a restart": an arbitrary but fixed value well
#: above 1, so a test asserting the switch's *restored* sequence can never
#: coincidentally pass against a fresh switch's default starting sequence of
#: 0, nor against the "always starts its first kill at 1" behavior other
#: tests in this file pin.
_RESTORED_KILL_SEQUENCE = 4

#: An earlier, already-re-armed kill sequence preceding `_RESTORED_KILL_SEQUENCE`
#: in a multi-cycle history, so the end-to-end restart test replays a genuine
#: kill/re-arm/kill sequence rather than a single kill.
_EARLIER_KILL_SEQUENCE = 3

#: The closed set of event types the kill path may ever ledger (SPEC intent:
#: position-hold, never dump): the kill announcement, the one cancel-all
#: directive, and one release per active reservation.
_CLOSED_KILL_EVENT_TYPES = frozenset(
    {"KillEngaged", "CancelAllDirective", "ReservationReleased"}
)

#: Substrings that must never appear (case-insensitively) in a kill-path
#: event's payload keys or string values -- the position-hold invariant is that
#: the kill switch only cancels/releases, never sells, closes, submits, or
#: dumps anything.
_FORBIDDEN_ACTION_TOKENS = ("sell", "close", "submit", "dump")

#: Every non-KILLED `Mode`, used to sweep "rearm only valid from KILLED".
_NON_KILLED_MODES: tuple[Mode, ...] = tuple(
    mode for mode in Mode if mode is not Mode.KILLED
)

#: `tests/fixtures/verification/balance_breach` -- the same fixture
#: `tests/riskkernel/test_verification.py::_make_verifier` reads from,
#: resolved independently here (not a cross-module import of that helper) so
#: this module stays self-contained: a $90.00 observed available cash against
#: a $95.00 ledger expectation, a $5,000,000-micro drift, at zero tolerance.
_BALANCE_BREACH_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "verification" / "balance_breach"
)

#: The `balance_breach` fixture's ledger-expected available cash and position,
#: matching `test_verification.py`'s baseline exactly: the position side
#: matches the fixture's observed 500-centi KXFED-24DEC exactly (so it never
#: contributes to the breach), leaving the $5,000,000-micro cash drift as the
#: sole, deterministic breach cause.
_BALANCE_BREACH_EXPECTED_CASH = MoneyMicros(95_000_000)
_BALANCE_BREACH_EXPECTED_POSITIONS = {"KXFED-24DEC": ContractCentis(500)}

#: Zero tolerance on both dimensions, so the fixture's $5,000,000-micro cash
#: drift breaches outright -- no drift-vs-breach boundary ambiguity.
_ZERO_BALANCE_TOLERANCE = MoneyMicros(0)
_ZERO_POSITION_TOLERANCE = ContractCentis(0)


@dataclass
class _StaticExpectationSource:
    """A fake `ExpectationSource` that always returns one fixed snapshot."""

    expectations: LedgerExpectations

    def get_expectations(self) -> LedgerExpectations:
        """Return the fixed `LedgerExpectations`, ignoring all state."""
        return self.expectations


def _build_breach_verifier() -> ReadOnlyVerifier:
    """Build a real `ReadOnlyVerifier` that yields `BREACH` on every cycle.

    Mirrors `test_verification.py::_make_verifier("balance_breach", ...)`'s
    construction (a real `FakeExchange` over the `balance_breach` fixture, a
    static `ExpectationSource`, zero tolerances, and a real `AlertDispatcher`),
    inlined here rather than imported so this test module never reaches into
    another test module's private (`_`-prefixed) helper.

    Returns:
        A `ReadOnlyVerifier` whose `run_cycle` classifies `BREACH` every time
        it is called, driven purely by the fixture's fixed $5,000,000-micro
        cash drift at zero balance tolerance.
    """
    return ReadOnlyVerifier(
        connector=FakeExchange.from_fixture_dir(_BALANCE_BREACH_FIXTURE_DIR),
        expectation_source=_StaticExpectationSource(
            LedgerExpectations(
                expected_available_cash=_BALANCE_BREACH_EXPECTED_CASH,
                expected_positions=_BALANCE_BREACH_EXPECTED_POSITIONS,
                expected_open_order_ids=frozenset(),
            )
        ),
        tolerances=VerificationTolerances(
            balance_tolerance=_ZERO_BALANCE_TOLERANCE,
            position_tolerance=_ZERO_POSITION_TOLERANCE,
        ),
        dispatcher=AlertDispatcher([], ledger_writer=LoggingLedgerWriter()),
        ledger_writer=InMemoryKernelLedgerWriter(),
    )


class _FakeAlertSink:
    """A narrow `KillSwitch` alert-dispatcher test double.

    Records every dispatched `AlertType` in call order, so a test can assert
    both the exact count of `HALT_KILL` dispatches and that no other alert
    type was fired.
    """

    def __init__(self) -> None:
        """Initialize with an empty dispatch log."""
        self.dispatched: list[AlertType] = []

    def dispatch(self, alert_type: AlertType, message: str) -> None:
        """Record a dispatched alert type, ignoring its message body.

        Args:
            alert_type: The alert type dispatched.
            message: The alert body (unused; recorded calls key on type only).
        """
        del message
        self.dispatched.append(alert_type)

    def count(self, alert_type: AlertType) -> int:
        """Return how many times `alert_type` was dispatched.

        Args:
            alert_type: The alert type to count.

        Returns:
            The number of `dispatch` calls recorded for `alert_type`.
        """
        return self.dispatched.count(alert_type)


class _FakeDirectiveSink:
    """A narrow `KillSwitch` directive-sink test double.

    Records every `CancelAllDirective` handed to it, so a test can assert the
    directive was delivered to the order-gateway-facing seam in addition to
    being ledgered.
    """

    def __init__(self) -> None:
        """Initialize with an empty received-directives log."""
        self.received: list[CancelAllDirective] = []

    def submit(self, directive: CancelAllDirective) -> None:
        """Record a submitted cancel-all directive.

        Args:
            directive: The directive submitted for delivery.
        """
        self.received.append(directive)


def _build_switch(
    *,
    mode: Mode = Mode.LIVE,
    writer: InMemoryKernelLedgerWriter | None = None,
    reservation_ledger: ReservationLedger | None = None,
    directive_sink: _FakeDirectiveSink | None = None,
    state_dir: Path | None = None,
) -> tuple[KillSwitch, InMemoryKernelLedgerWriter, ModeStateMachine, _FakeAlertSink]:
    """Build a fully-wired `KillSwitch` plus its collaborators for a test.

    The switch is built with a fixed injected clock (`_FIXED_EPOCH_S`), so
    every `KillEngaged.epoch` this switch ever ledgers is deterministic.

    Args:
        mode: The mode machine's starting mode. Defaults to `Mode.LIVE`.
        writer: The ledger writer to wire in; a fresh
            `InMemoryKernelLedgerWriter` if omitted. Pass the same writer used
            to build `reservation_ledger` so kill-path and reservation-release
            events land in one shared, assertable log.
        reservation_ledger: An optional reservation ledger the switch releases
            on kill.
        directive_sink: An optional fake directive sink the switch hands its
            `CancelAllDirective` to.
        state_dir: An optional state directory the switch writes a `KILL` file
            into on kill.

    Returns:
        A `(switch, writer, mode_machine, alert_sink)` tuple.
    """
    effective_writer = writer if writer is not None else InMemoryKernelLedgerWriter()
    mode_machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=mode)
    alert_sink = _FakeAlertSink()
    switch = KillSwitch(
        mode_machine,
        effective_writer,
        alert_sink,
        reservation_ledger=reservation_ledger,
        directive_sink=directive_sink,
        state_dir=state_dir,
        clock=lambda: _FIXED_EPOCH_S,
    )
    return switch, effective_writer, mode_machine, alert_sink


# --- KillTrigger: the four named trigger sources --------------------------------


def test_kill_trigger_has_exactly_the_four_named_sources() -> None:
    """`KillTrigger` has exactly CLI/KILL_FILE/DASHBOARD/AUTO_RECONCILIATION."""
    assert {member.name for member in KillTrigger} == {
        "CLI",
        "KILL_FILE",
        "DASHBOARD",
        "AUTO_RECONCILIATION",
    }
    assert len(KillTrigger) == 4


# --- New ledger events: KillEngaged / CancelAllDirective / KillReArmed ----------


def test_kill_engaged_event_type_equals_its_class_name_and_is_registered() -> None:
    """`KillEngaged.event_type` is the literal string `"KillEngaged"`, and the
    class is reachable via `EVENT_TYPES["KillEngaged"]` for envelope replay.
    """
    event = KillEngaged(
        component="riskkernel", trigger="CLI", kill_sequence=1, epoch=_FIXED_EPOCH_S
    )

    assert event.event_type == "KillEngaged"
    assert event.payload == {
        "trigger": "CLI",
        "kill_sequence": 1,
        "epoch": _FIXED_EPOCH_S,
    }
    assert EVENT_TYPES["KillEngaged"] is KillEngaged


def test_cancel_all_directive_event_type_equals_its_class_name_and_is_registered() -> (
    None
):
    """`CancelAllDirective.event_type` is `"CancelAllDirective"`, its payload
    carries `scope="all_open_orders"`, and the class is registered.
    """
    event = CancelAllDirective(component="riskkernel", scope="all_open_orders")

    assert event.event_type == "CancelAllDirective"
    assert event.payload == {"scope": "all_open_orders"}
    assert EVENT_TYPES["CancelAllDirective"] is CancelAllDirective


def test_kill_rearmed_event_type_equals_its_class_name_and_is_registered() -> None:
    """`KillReArmed.event_type` is `"KillReArmed"` and the class is registered."""
    event = KillReArmed(component="riskkernel", kill_sequence=1)

    assert event.event_type == "KillReArmed"
    assert event.payload == {"kill_sequence": 1}
    assert EVENT_TYPES["KillReArmed"] is KillReArmed


# --- RiskConfig.kill_after_consecutive_mismatches -------------------------------


def test_risk_config_kill_after_consecutive_mismatches_defaults_to_three() -> None:
    """`RiskConfig().kill_after_consecutive_mismatches` defaults to 3."""
    assert RiskConfig().kill_after_consecutive_mismatches == 3


# --- KillSwitch.kill(): the four triggers, each producing one KillEngaged -------


@pytest.mark.parametrize("trigger", list(KillTrigger))
def test_kill_transitions_to_killed_and_records_one_kill_engaged_with_trigger_label(
    trigger: KillTrigger,
) -> None:
    """Every `KillTrigger` kills the same way: `.mode` becomes `KILLED` and
    exactly one `KillEngaged` event carries that trigger's name.
    """
    switch, writer, machine, _sink = _build_switch()
    assert switch.mode == Mode.LIVE

    switch.kill(trigger)

    assert machine.mode == Mode.KILLED
    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    assert len(kill_events) == 1
    assert kill_events[0].payload["trigger"] == trigger.name
    assert kill_events[0].payload["kill_sequence"] == 1
    assert kill_events[0].payload["epoch"] == _FIXED_EPOCH_S


def test_kill_file_trigger_kills_with_no_dashboard_object_ever_constructed(
    tmp_path: Path,
) -> None:
    """A `KILL` file kills the switch with zero `DashboardKillStub` object in
    existence anywhere -- the kill-file path has no HTTP/dashboard dependency.
    """
    switch, writer, machine, _sink = _build_switch(state_dir=tmp_path)
    watcher = KillFileWatcher(switch, tmp_path)
    (tmp_path / "KILL").write_text("", encoding="utf-8")

    watcher.poll_once(now_epoch_s=_FIXED_EPOCH_S)

    assert machine.mode == Mode.KILLED
    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    assert len(kill_events) == 1
    assert kill_events[0].payload["trigger"] == KillTrigger.KILL_FILE.name


def test_kill_file_presence_alone_triggers_kill_regardless_of_unreadable_content(
    tmp_path: Path,
) -> None:
    """A `KILL` file kills on presence alone: even non-UTF-8 garbage content
    (simulating an "unreadable" file) still kills, since the watcher must
    never need to parse the file -- fail toward dead on mere presence.
    """
    switch, _writer, machine, _sink = _build_switch(state_dir=tmp_path)
    watcher = KillFileWatcher(switch, tmp_path)
    (tmp_path / "KILL").write_bytes(b"\xff\xfe\x00not-valid-utf8-garbage")

    watcher.poll_once(now_epoch_s=_FIXED_EPOCH_S)

    assert machine.mode == Mode.KILLED


def test_kill_with_state_dir_wired_writes_a_kill_file(tmp_path: Path) -> None:
    """When a state dir is wired, `kill()` also writes a `KILL` file into it."""
    switch, _writer, _machine, _sink = _build_switch(state_dir=tmp_path)

    switch.kill(KillTrigger.CLI)

    assert (tmp_path / "KILL").exists()


def test_kill_creates_missing_state_dir_before_writing_kill_file(
    tmp_path: Path,
) -> None:
    """A non-CLI trigger against a not-yet-created `state_dir` still lands the
    `KILL` file: `kill()` creates the directory (parents included) rather than
    raising `FileNotFoundError`, so the fail-toward-dead file write can never be
    defeated by a missing directory (reviewer finding, PR #134).
    """
    missing_state_dir = tmp_path / "not" / "yet" / "created"
    assert not missing_state_dir.exists()
    switch, _writer, machine, _sink = _build_switch(state_dir=missing_state_dir)

    switch.kill(KillTrigger.AUTO_RECONCILIATION)

    assert (missing_state_dir / "KILL").exists()
    assert machine.mode is Mode.KILLED


# --- Kill-effect surface: multiple active reservations ---------------------------


def test_kill_releases_all_reservations_and_closes_the_event_surface() -> None:
    """From a state with three active reservations, `kill()`: releases every
    reservation (`total_reserved` becomes 0, `used_intent_ids` still
    remembers them), emits exactly one `CancelAllDirective` (ledgered and
    delivered to the sink), dispatches exactly one `HALT_KILL` alert, and the
    kill path's event-type surface is exactly the closed set.
    """
    writer = InMemoryKernelLedgerWriter()
    reservation_ledger = ReservationLedger(writer)
    directive_sink = _FakeDirectiveSink()
    switch, _writer, machine, sink = _build_switch(
        writer=writer,
        reservation_ledger=reservation_ledger,
        directive_sink=directive_sink,
    )
    reservation_ledger.reserve(
        "intent-a", MoneyMicros(1_000_000), "idem-a", expires_at=_FAR_FUTURE_EXPIRY_S
    )
    reservation_ledger.reserve(
        "intent-b", MoneyMicros(2_000_000), "idem-b", expires_at=_FAR_FUTURE_EXPIRY_S
    )
    reservation_ledger.reserve(
        "intent-c", MoneyMicros(3_000_000), "idem-c", expires_at=_FAR_FUTURE_EXPIRY_S
    )
    assert reservation_ledger.total_reserved() == MoneyMicros(6_000_000)

    # Baseline the shared ledger before the kill: the three `reserve()` calls
    # above each recorded a locked `ReservationCreated` event
    # (`tests/riskkernel/test_reservations.py::
    # test_reserve_emits_exactly_one_reservation_created_event` pins that
    # behavior), so those setup events sit in `writer.events` too. The
    # *kill-effect surface* this test pins is the closed set of event types the
    # kill path itself ledgers, so it is asserted against only the events the
    # kill appended -- `writer.events[events_before_kill:]` -- not the setup's
    # reservation-creation noise.
    events_before_kill = len(writer.events)

    switch.kill(KillTrigger.CLI)

    assert machine.mode == Mode.KILLED
    assert reservation_ledger.total_reserved() == MoneyMicros(0)
    assert reservation_ledger.used_intent_ids() == frozenset(
        {"intent-a", "intent-b", "intent-c"}
    )

    released_events = [
        event for event in writer.events if event.event_type == "ReservationReleased"
    ]
    assert len(released_events) == 3
    assert {event.payload["intent_id"] for event in released_events} == {
        "intent-a",
        "intent-b",
        "intent-c",
    }

    cancel_events = [
        event for event in writer.events if event.event_type == "CancelAllDirective"
    ]
    assert len(cancel_events) == 1
    assert cancel_events[0].payload["scope"] == "all_open_orders"
    assert len(directive_sink.received) == 1
    assert directive_sink.received[0].payload["scope"] == "all_open_orders"

    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    assert len(kill_events) == 1
    assert kill_events[0].payload["trigger"] == KillTrigger.CLI.name

    assert sink.count(AlertType.HALT_KILL) == 1

    kill_path_events = writer.events[events_before_kill:]
    assert {event.event_type for event in kill_path_events} == _CLOSED_KILL_EVENT_TYPES


# --- Idempotency: killing an already-KILLED switch does nothing ----------------


def test_second_kill_while_killed_is_a_no_op() -> None:
    """Calling `kill()` again while already `KILLED` is a pure no-op: no new
    ledger events, no new alert dispatch, and no exception.
    """
    writer = InMemoryKernelLedgerWriter()
    reservation_ledger = ReservationLedger(writer)
    switch, _writer, machine, sink = _build_switch(
        writer=writer, reservation_ledger=reservation_ledger
    )
    reservation_ledger.reserve(
        "intent-a", MoneyMicros(1_000_000), "idem-a", expires_at=_FAR_FUTURE_EXPIRY_S
    )
    switch.kill(KillTrigger.CLI)
    events_after_first_kill = list(writer.events)
    alert_count_after_first_kill = sink.count(AlertType.HALT_KILL)

    switch.kill(KillTrigger.DASHBOARD)

    assert writer.events == events_after_first_kill
    assert sink.count(AlertType.HALT_KILL) == alert_count_after_first_kill
    assert machine.mode == Mode.KILLED
    assert reservation_ledger.total_reserved() == MoneyMicros(0)


def test_kill_file_watcher_repeated_poll_with_a_persistent_kill_file_re_kills_nothing(
    tmp_path: Path,
) -> None:
    """A `KILL` file left in place across many polled beats kills exactly
    once: every subsequent `poll_once` call is a no-op.
    """
    switch, writer, machine, _sink = _build_switch(state_dir=tmp_path)
    watcher = KillFileWatcher(switch, tmp_path)
    (tmp_path / "KILL").write_text("", encoding="utf-8")

    for beat in range(5):
        watcher.poll_once(now_epoch_s=_FIXED_EPOCH_S + beat)

    assert machine.mode == Mode.KILLED
    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    assert len(kill_events) == 1


# --- expected_rearm_phrase: dynamic phrase embedding the sequence number --------


@given(sequence=st.integers(min_value=1, max_value=10_000))
def test_expected_rearm_phrase_contains_the_sequence_number(sequence: int) -> None:
    """`expected_rearm_phrase(sequence)` always contains `str(sequence)` --
    pinning the "sequence number must appear" contract without hard-coding
    the literal template text.
    """
    switch, _writer, _machine, _sink = _build_switch()

    phrase = switch.expected_rearm_phrase(sequence)

    assert str(sequence) in phrase


# --- rearm(): exact / mismatch / wrong-sequence / case-folded / wrong-state -----


def test_rearm_exact_phrase_moves_killed_to_paused_and_ledgers_rearmed() -> None:
    """A correctly typed re-arm confirmation moves `KILLED` -> `PAUSED` and
    ledgers exactly one `KillReArmed` event.
    """
    switch, writer, machine, _sink = _build_switch()
    switch.kill(KillTrigger.CLI)
    phrase = switch.expected_rearm_phrase(switch.active_kill_sequence)

    switch.rearm(phrase)

    assert machine.mode == Mode.PAUSED
    rearmed_events = [
        event for event in writer.events if event.event_type == "KillReArmed"
    ]
    assert len(rearmed_events) == 1


def test_rearm_with_a_mismatched_phrase_raises_and_records_nothing() -> None:
    """A wrong confirmation phrase raises `KillReArmError`, leaves `.mode` at
    `KILLED`, and ledgers no `KillReArmed` event.
    """
    switch, writer, machine, _sink = _build_switch()
    switch.kill(KillTrigger.CLI)
    events_before = list(writer.events)

    with pytest.raises(KillReArmError):
        switch.rearm("definitely the wrong phrase")

    assert machine.mode == Mode.KILLED
    assert writer.events == events_before
    assert not any(event.event_type == "KillReArmed" for event in writer.events)


def test_rearm_with_wrong_sequence_number_in_phrase_raises() -> None:
    """A phrase built from the wrong sequence number (correct template,
    wrong embedded int) is rejected just as any other mismatch is.
    """
    switch, _writer, machine, _sink = _build_switch()
    switch.kill(KillTrigger.CLI)
    wrong_phrase = switch.expected_rearm_phrase(switch.active_kill_sequence + 1)

    with pytest.raises(KillReArmError):
        switch.rearm(wrong_phrase)

    assert machine.mode == Mode.KILLED


def test_rearm_with_a_case_folded_phrase_raises_and_stays_killed() -> None:
    """A case-swapped confirmation phrase is rejected -- `rearm` must not
    case-fold the comparison.
    """
    switch, _writer, machine, _sink = _build_switch()
    switch.kill(KillTrigger.CLI)
    phrase = switch.expected_rearm_phrase(switch.active_kill_sequence)
    mismatched_phrase = phrase.swapcase()
    assert mismatched_phrase != phrase, (
        "fixture assumption: the rearm phrase must contain a cased character "
        "for this test to be meaningful"
    )

    with pytest.raises(KillReArmError):
        switch.rearm(mismatched_phrase)

    assert machine.mode == Mode.KILLED


@pytest.mark.parametrize("mode", _NON_KILLED_MODES)
def test_rearm_from_any_non_killed_mode_raises_kill_rearm_error(mode: Mode) -> None:
    """`rearm` is only ever valid from `KILLED`: called from any other mode it
    raises `KillReArmError` and leaves the machine in its current mode.
    """
    switch, _writer, machine, _sink = _build_switch(mode=mode)

    with pytest.raises(KillReArmError):
        switch.rearm(switch.expected_rearm_phrase(1))

    assert machine.mode == mode


def test_kill_sequence_strictly_increases_across_a_kill_rearm_kill_cycle() -> None:
    """The kill sequence is monotonic: a second kill (after a re-arm) carries
    a strictly larger `kill_sequence` than the first.
    """
    switch, writer, machine, _sink = _build_switch()

    switch.kill(KillTrigger.CLI)
    first_sequence = switch.active_kill_sequence
    switch.rearm(switch.expected_rearm_phrase(first_sequence))
    assert machine.mode == Mode.PAUSED

    switch.kill(KillTrigger.DASHBOARD)
    second_sequence = switch.active_kill_sequence

    assert second_sequence > first_sequence
    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    assert [event.payload["kill_sequence"] for event in kill_events] == [
        first_sequence,
        second_sequence,
    ]


def test_post_rearm_a_pre_kill_intent_id_is_still_duplicate_rejected() -> None:
    """After a successful re-arm, an intent id reserved *before* the kill is
    still permanently remembered: no stale-intent replay is ever possible.
    """
    writer = InMemoryKernelLedgerWriter()
    reservation_ledger = ReservationLedger(writer)
    switch, _writer, machine, _sink = _build_switch(
        writer=writer, reservation_ledger=reservation_ledger
    )
    reservation_ledger.reserve(
        "intent-x", MoneyMicros(1_000_000), "idem-x", expires_at=_FAR_FUTURE_EXPIRY_S
    )

    switch.kill(KillTrigger.CLI)
    switch.rearm(switch.expected_rearm_phrase(switch.active_kill_sequence))
    assert machine.mode == Mode.PAUSED

    with pytest.raises(DuplicateReservationError):
        reservation_ledger.reserve(
            "intent-x",
            MoneyMicros(500_000),
            "idem-x-replay",
            expires_at=_FAR_FUTURE_EXPIRY_S,
        )


# --- KILLED hard-veto at the RiskKernel level ------------------------------------


@pytest.mark.timeout(30)
def test_evaluate_intent_on_a_killed_kernel_returns_the_single_killed_reason() -> None:
    """A `KILLED` kernel's `evaluate_intent` short-circuits the check pipeline
    entirely: it returns `reasons == ("KILLED",)` -- the one hard-veto reason,
    never the usual multi-reason pipeline veto -- and records one
    `IntentVetoed` event. The identical pre-kill call must not produce that
    same single-reason signature (proving the pipeline really ran pre-kill).
    """
    writer = InMemoryKernelLedgerWriter()
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE)
    kernel = RiskKernel(writer, mode_machine=machine)
    intent = make_intent()
    context = make_context()

    pre_kill_decision = kernel.evaluate_intent(intent, context)
    assert pre_kill_decision.reasons != ("KILLED",)

    machine.transition(Mode.KILLED)
    decision = kernel.evaluate_intent(intent, context)

    assert decision.vetoed is True
    assert decision.reasons == ("KILLED",)
    assert decision.ledgered is True

    vetoed_events = [
        event for event in writer.events if event.event_type == "IntentVetoed"
    ]
    assert len(vetoed_events) == 2
    assert list(vetoed_events[-1].payload["reasons"]) == ["KILLED"]


@pytest.mark.timeout(30)
def test_evaluate_intent_after_rearm_to_paused_no_longer_reports_killed() -> None:
    """After a `KILLED` kernel's mode machine is re-armed to `PAUSED`,
    `evaluate_intent`'s reasons no longer contain `"KILLED"`.
    """
    writer = InMemoryKernelLedgerWriter()
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE)
    kernel = RiskKernel(writer, mode_machine=machine)
    intent = make_intent()
    context = make_context()

    machine.transition(Mode.KILLED)
    killed_decision = kernel.evaluate_intent(intent, context)
    assert killed_decision.reasons == ("KILLED",)

    # Drive the mode machine through the same two-step composition
    # `KillSwitch.rearm` performs internally (`machine.rearm(...)` then
    # `machine.transition(Mode.PAUSED)`) directly, so this test isolates the
    # kernel's `evaluate_intent` behavior from the kill-switch confirmation
    # mechanics pinned separately above.
    machine.rearm(REARM_CONFIRMATION_PHRASE)
    machine.transition(Mode.PAUSED)
    decision = kernel.evaluate_intent(intent, context)

    assert "KILLED" not in decision.reasons


# --- KILLED dead end: nothing but a ledgered re-arm ever escapes it -------------


@pytest.mark.timeout(30)
def test_killed_kernel_stays_killed_through_heartbeats_verification_and_demotions() -> (
    None
):
    """Once `KILLED`, heartbeats, verification cycles, and every demotion
    trigger firing leave the kernel `KILLED` absent a ledgered re-arm.
    """
    writer = InMemoryKernelLedgerWriter()
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE)
    kernel = RiskKernel(writer, mode_machine=machine)
    machine.transition(Mode.KILLED)

    kernel.run(max_beats=3, heartbeat_interval=0)
    kernel.run_verification_cycle()
    for trigger in DemotionTrigger:
        kernel.fire_demotion_trigger(trigger)

    assert machine.mode == Mode.KILLED


# --- ReconciliationMismatchMonitor: consecutive-BREACH auto-trigger -------------


def test_reconciliation_monitor_does_not_kill_before_nth_breach() -> None:
    """`N - 1` consecutive `BREACH` outcomes never kill the switch."""
    switch, _writer, machine, _sink = _build_switch()
    monitor = ReconciliationMismatchMonitor(switch, threshold=3)

    monitor.observe(VerificationOutcome.BREACH)
    monitor.observe(VerificationOutcome.BREACH)

    assert machine.mode != Mode.KILLED


def test_reconciliation_monitor_kills_on_exactly_the_nth_consecutive_breach() -> None:
    """The `N`-th consecutive `BREACH` outcome kills with trigger
    `AUTO_RECONCILIATION`.
    """
    switch, writer, machine, _sink = _build_switch()
    monitor = ReconciliationMismatchMonitor(switch, threshold=3)

    monitor.observe(VerificationOutcome.BREACH)
    monitor.observe(VerificationOutcome.BREACH)
    monitor.observe(VerificationOutcome.BREACH)

    assert machine.mode == Mode.KILLED
    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    assert len(kill_events) == 1
    assert kill_events[0].payload["trigger"] == KillTrigger.AUTO_RECONCILIATION.name


@pytest.mark.parametrize(
    "non_breach_outcome",
    [VerificationOutcome.CLEAN, VerificationOutcome.DRIFT_WITHIN_TOLERANCE],
)
def test_reconciliation_monitor_resets_its_count_on_any_non_breach_outcome(
    non_breach_outcome: VerificationOutcome,
) -> None:
    """Any non-`BREACH` outcome resets the consecutive-breach count to zero,
    so an alternating BREACH/non-BREACH/BREACH/... sequence never kills even
    when it runs well past the threshold.
    """
    switch, _writer, machine, _sink = _build_switch()
    monitor = ReconciliationMismatchMonitor(switch, threshold=3)

    for _ in range(5):
        monitor.observe(VerificationOutcome.BREACH)
        monitor.observe(non_breach_outcome)

    assert machine.mode != Mode.KILLED


def test_reconciliation_monitor_threshold_threaded_from_risk_config() -> None:
    """A monitor built with `RiskConfig().kill_after_consecutive_mismatches`
    kills on the default-configured 3rd consecutive breach, not before.
    """
    config = RiskConfig()
    switch, _writer, machine, _sink = _build_switch()
    monitor = ReconciliationMismatchMonitor(
        switch, threshold=config.kill_after_consecutive_mismatches
    )

    for _ in range(config.kill_after_consecutive_mismatches - 1):
        monitor.observe(VerificationOutcome.BREACH)
    assert machine.mode != Mode.KILLED

    monitor.observe(VerificationOutcome.BREACH)
    assert machine.mode == Mode.KILLED


@pytest.mark.timeout(30)
def test_sustained_verification_breach_through_kernel_auto_kills_on_nth_cycle() -> None:
    """A real breaching verifier, driven end-to-end through
    `RiskKernel.run_verification_cycle`, auto-kills via
    `_feed_mismatch_monitor` -> `ReconciliationMismatchMonitor.observe` on
    exactly the `N`-th consecutive breach -- not before, and not by any other
    path.

    Escalation semantics (issue #32 HALT-on-breach composed with issue #35
    auto-kill): each of the first `N - 1` breach cycles halts the kernel (the
    first breach transitions `LIVE` -> `HALT`; every subsequent breach up to
    the threshold is a no-op against that same `HALT`, since
    `_feed_mismatch_monitor` runs -- and does not yet trip the monitor's
    threshold -- before `_halt_on_breach` is even consulted). Only on the
    `N`-th consecutive breach does the monitor's threshold trip mid-cycle,
    driving the *shared* mode machine straight to `KILLED` from `HALT` (a
    legal safety-mode move) before `_halt_on_breach`'s own idempotency guard
    (`mode in {HALT, KILLED}`) sees it and no-ops -- so the kernel lands on
    `KILLED`, never bounces back to `HALT`.
    """
    threshold = RiskConfig().kill_after_consecutive_mismatches
    writer = InMemoryKernelLedgerWriter()
    mode_machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE)
    alert_sink = _FakeAlertSink()
    switch = KillSwitch(mode_machine, writer, alert_sink, clock=lambda: _FIXED_EPOCH_S)
    monitor = ReconciliationMismatchMonitor(switch, threshold=threshold)
    integration = KillIntegration(switch=switch, monitor=monitor)
    kernel = RiskKernel(
        writer,
        mode_machine=mode_machine,
        verifier=_build_breach_verifier(),
        clock=lambda: _FIXED_EPOCH_S,
        kill_integration=integration,
    )

    for _ in range(threshold - 1):
        kernel.run_verification_cycle()
    assert mode_machine.mode == Mode.HALT

    kernel.run_verification_cycle()

    assert mode_machine.mode == Mode.KILLED
    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    assert len(kill_events) == 1
    assert kill_events[0].payload["trigger"] == KillTrigger.AUTO_RECONCILIATION.name


# --- Replay-KILLED then breach: verification stays inert against a dead end -----


class _ObservingMonitor(ReconciliationMismatchMonitor):
    """A `ReconciliationMismatchMonitor` subclass recording every outcome it
    observes, on top of performing its normal auto-kill behavior -- proof
    that `RiskKernel._feed_mismatch_monitor` really fed a breaching cycle's
    outcome through, without reaching into any private monitor state.
    """

    def __init__(self, switch: KillSwitch, threshold: int) -> None:
        """Wire the spy exactly like its base, plus an empty observed log.

        Args:
            switch: The kill switch engaged once the threshold is reached.
            threshold: The number of consecutive breaches that auto-kills.
        """
        super().__init__(switch, threshold)
        self.observed: list[VerificationOutcome] = []

    def observe(self, outcome: VerificationOutcome) -> None:
        """Record `outcome`, then fold it through the base implementation.

        Args:
            outcome: The verification outcome to record and fold.
        """
        self.observed.append(outcome)
        super().observe(outcome)


def test_replay_killed_history_then_breach_stays_killed_with_no_halt_event() -> None:
    """A kernel rebuilt via `RiskKernel.from_events` over an unrearmed
    `KillEngaged` history comes back `KILLED`; running one breaching
    verification cycle against it leaves the kernel `KILLED` -- the
    verifier's own `VerificationMismatch` event still records unconditionally,
    the mismatch monitor still observes the `BREACH` outcome, but
    `_halt_on_breach`'s already-`KILLED` guard (`process.py:738`) means the
    cycle never records a `VerificationMismatchHalt` event: a `KILLED` kernel
    can never bounce to `HALT`.
    """
    events = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=_FIXED_EPOCH_S
        ),
    ]
    shared_machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.RESEARCH)
    writer = InMemoryKernelLedgerWriter()
    alert_sink = _FakeAlertSink()
    switch = KillSwitch.from_events(
        events, shared_machine, writer, alert_sink, clock=lambda: _FIXED_EPOCH_S
    )
    monitor = _ObservingMonitor(
        switch, threshold=RiskConfig().kill_after_consecutive_mismatches
    )
    integration = KillIntegration(switch=switch, monitor=monitor)
    # Built inline (not via `_build_breach_verifier`, whose own private
    # `InMemoryKernelLedgerWriter` would put `VerificationMismatch` on a
    # different log than `writer`) so the verifier's own event and the
    # kernel's kill/halt events all land in the one `writer` this test
    # inspects -- mirroring `test_verification.py::_make_verifier`'s
    # shared-writer wiring.
    breach_verifier = ReadOnlyVerifier(
        connector=FakeExchange.from_fixture_dir(_BALANCE_BREACH_FIXTURE_DIR),
        expectation_source=_StaticExpectationSource(
            LedgerExpectations(
                expected_available_cash=_BALANCE_BREACH_EXPECTED_CASH,
                expected_positions=_BALANCE_BREACH_EXPECTED_POSITIONS,
                expected_open_order_ids=frozenset(),
            )
        ),
        tolerances=VerificationTolerances(
            balance_tolerance=_ZERO_BALANCE_TOLERANCE,
            position_tolerance=_ZERO_POSITION_TOLERANCE,
        ),
        dispatcher=AlertDispatcher([], ledger_writer=LoggingLedgerWriter()),
        ledger_writer=writer,
    )
    kernel = RiskKernel.from_events(
        events,
        writer,
        mode_machine=shared_machine,
        verifier=breach_verifier,
        kill_integration=integration,
    )
    assert kernel.mode is Mode.KILLED

    kernel.run_verification_cycle()

    assert kernel.mode is Mode.KILLED
    mismatch_events = [
        event for event in writer.events if event.event_type == "VerificationMismatch"
    ]
    assert len(mismatch_events) == 1
    halt_events = [
        event
        for event in writer.events
        if event.event_type == "VerificationMismatchHalt"
    ]
    assert halt_events == []
    assert monitor.observed == [VerificationOutcome.BREACH]
    # The breach cycle records NO new `KillEngaged`: a single breach never
    # reaches the monitor's escalation threshold and an already-`KILLED` kernel
    # fires no fresh kill. The kernel is `KILLED` purely from the replayed
    # history, which `from_events` folds in-memory without re-ledgering it onto
    # this run's fresh writer -- so `writer.events` holds zero `KillEngaged`.
    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    assert kill_events == []


# --- DashboardKillStub: challenge/confirm handshake ------------------------------


def test_dashboard_kill_stub_confirm_with_the_correct_token_kills() -> None:
    """`confirm` with the exact token `request_challenge` issued kills with
    trigger `DASHBOARD`.
    """
    switch, writer, machine, _sink = _build_switch()
    stub = DashboardKillStub(switch)
    token = stub.request_challenge()

    stub.confirm(token)

    assert machine.mode == Mode.KILLED
    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    assert len(kill_events) == 1
    assert kill_events[0].payload["trigger"] == KillTrigger.DASHBOARD.name


def test_dashboard_confirm_with_wrong_token_raises_and_does_not_kill() -> None:
    """`confirm` with a token that was never issued raises
    `DashboardChallengeError` and never kills.
    """
    switch, _writer, machine, _sink = _build_switch()
    stub = DashboardKillStub(switch)
    stub.request_challenge()

    with pytest.raises(DashboardChallengeError):
        stub.confirm("a-token-that-was-never-issued")

    assert machine.mode != Mode.KILLED


def test_dashboard_kill_stub_confirm_rejects_a_reused_token() -> None:
    """A token, once confirmed, cannot be confirmed again: the second
    `confirm` call with the same (already-consumed) token raises
    `DashboardChallengeError`.
    """
    switch, _writer, machine, _sink = _build_switch()
    stub = DashboardKillStub(switch)
    token = stub.request_challenge()
    stub.confirm(token)
    assert machine.mode == Mode.KILLED

    with pytest.raises(DashboardChallengeError):
        stub.confirm(token)


# --- KillFileWatcher: REARM file handling ----------------------------------------


def test_kill_file_watcher_wrong_rearm_phrase_consumes_the_file_and_stays_killed(
    tmp_path: Path,
) -> None:
    """A `REARM` file carrying the wrong phrase is consumed (deleted) whether
    or not it succeeds -- and the mode stays `KILLED`.
    """
    switch, _writer, machine, _sink = _build_switch(state_dir=tmp_path)
    watcher = KillFileWatcher(switch, tmp_path)
    switch.kill(KillTrigger.CLI)
    (tmp_path / "REARM").write_text("wrong phrase entirely", encoding="utf-8")

    watcher.poll_once(now_epoch_s=_FIXED_EPOCH_S)

    assert machine.mode == Mode.KILLED
    assert not (tmp_path / "REARM").exists()


def test_kill_file_watcher_unreadable_rearm_file_is_consumed_and_stays_killed(
    tmp_path: Path,
) -> None:
    """A `REARM` file whose bytes are not valid UTF-8 (a read that *raises*, not
    merely a wrong phrase) must still fail toward dead: `poll_once` never
    propagates the decode error, the `REARM` file is always consumed (deleted),
    the stale `KILL` file is left in place, and the mode stays `KILLED` -- an
    unreadable re-arm file can never accidentally re-arm nor wedge the beat.
    """
    switch, _writer, machine, _sink = _build_switch(state_dir=tmp_path)
    watcher = KillFileWatcher(switch, tmp_path)
    switch.kill(KillTrigger.CLI)
    assert (tmp_path / "KILL").exists()
    (tmp_path / "REARM").write_bytes(b"\xff\xfe\x00not-valid-utf8-garbage")

    watcher.poll_once(now_epoch_s=_FIXED_EPOCH_S)

    assert machine.mode == Mode.KILLED
    assert not (tmp_path / "REARM").exists()
    assert (tmp_path / "KILL").exists()


def test_kill_file_watcher_correct_rearm_phrase_moves_to_paused_and_removes_both_files(
    tmp_path: Path,
) -> None:
    """A `REARM` file carrying the exact expected phrase moves `KILLED` ->
    `PAUSED` and removes *both* the `REARM` file and the now-stale `KILL`
    file -- so a later poll never instantly re-kills a freshly re-armed
    switch.
    """
    switch, _writer, machine, _sink = _build_switch(state_dir=tmp_path)
    watcher = KillFileWatcher(switch, tmp_path)
    switch.kill(KillTrigger.CLI)
    assert (tmp_path / "KILL").exists()
    phrase = switch.expected_rearm_phrase(switch.active_kill_sequence)
    (tmp_path / "REARM").write_text(phrase, encoding="utf-8")

    watcher.poll_once(now_epoch_s=_FIXED_EPOCH_S)

    assert machine.mode == Mode.PAUSED
    assert not (tmp_path / "REARM").exists()
    assert not (tmp_path / "KILL").exists()


# --- Position-hold no-dump invariant (Hypothesis) --------------------------------


@given(
    reservation_count=st.integers(min_value=0, max_value=5),
    pre_kill_mode=st.sampled_from(
        [Mode.RESEARCH, Mode.PAPER, Mode.LIVE_MICRO, Mode.LIVE]
    ),
)
def test_kill_path_event_surface_never_carries_a_sell_close_submit_or_dump_action(
    reservation_count: int, pre_kill_mode: Mode
) -> None:
    """Regardless of the starting mode or how many reservations are active,
    every event the kill path ledgers is one of the closed set's types, and
    no payload key or string value ever names a sell/close/submit/dump
    action -- the kill switch only cancels and releases, it never trades.
    """
    writer = InMemoryKernelLedgerWriter()
    reservation_ledger = ReservationLedger(writer)
    directive_sink = _FakeDirectiveSink()
    switch, _writer, _machine, _sink = _build_switch(
        mode=pre_kill_mode,
        writer=writer,
        reservation_ledger=reservation_ledger,
        directive_sink=directive_sink,
    )
    for index in range(reservation_count):
        reservation_ledger.reserve(
            f"intent-{index}",
            MoneyMicros(1_000_000),
            f"idem-{index}",
            expires_at=_FAR_FUTURE_EXPIRY_S,
        )

    # Baseline before the kill so the invariant below is asserted over exactly
    # the events the kill path ledgers: each `reserve()` above recorded a locked
    # `ReservationCreated` setup event into the shared writer (see
    # `test_reservations.py`), which is not part of the kill-effect surface.
    events_before_kill = len(writer.events)

    switch.kill(KillTrigger.CLI)

    for event in writer.events[events_before_kill:]:
        assert event.event_type in _CLOSED_KILL_EVENT_TYPES
        for key, value in event.payload.items():
            key_lower = str(key).lower()
            assert not any(token in key_lower for token in _FORBIDDEN_ACTION_TOKENS)
            if isinstance(value, str):
                value_lower = value.lower()
                assert not any(
                    token in value_lower for token in _FORBIDDEN_ACTION_TOKENS
                )


# --- Kill drill (SPEC S10.12): open reservations, mid-run kill, then re-arm ----


@pytest.mark.timeout(30)
def test_kill_drill_open_reservations_kill_file_mid_run_then_rearm_restores_capability(
    tmp_path: Path,
) -> None:
    """The full SPEC S10.12 kill drill: with open reservations and a pending
    pre-kill intent, a `KILL` file dropped mid-run halts everything (mode,
    reservations, directive, alert) inside one bounded `kernel.run` beat; a
    subsequent `REARM` file with the correct phrase restores `PAUSED` and
    approval capability (the hard `"KILLED"` veto reason disappears); and the
    pre-kill intent id is still permanently rejected as a duplicate -- no
    stale reservation replay survives the drill.
    """
    writer = InMemoryKernelLedgerWriter()
    reservation_ledger = ReservationLedger(writer)
    directive_sink = _FakeDirectiveSink()
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE)
    alert_sink = _FakeAlertSink()
    switch = KillSwitch(
        machine,
        writer,
        alert_sink,
        reservation_ledger=reservation_ledger,
        directive_sink=directive_sink,
        state_dir=tmp_path,
        clock=lambda: _FIXED_EPOCH_S,
    )
    watcher = KillFileWatcher(switch, tmp_path)
    integration = KillIntegration(switch=switch, watcher=watcher)
    kernel = RiskKernel(writer, mode_machine=machine, kill_integration=integration)

    reservation_ledger.reserve(
        "pending-1", MoneyMicros(1_000_000), "idem-p1", expires_at=_FAR_FUTURE_EXPIRY_S
    )
    reservation_ledger.reserve(
        "pending-2", MoneyMicros(2_000_000), "idem-p2", expires_at=_FAR_FUTURE_EXPIRY_S
    )

    pre_kill_intent = make_intent(
        intent_id="pre-kill-intent", idempotency_key="pre-kill-idem"
    )
    context = make_context()
    pre_kill_decision = kernel.evaluate_intent(pre_kill_intent, context)
    assert pre_kill_decision.reasons != ("KILLED",)

    (tmp_path / "KILL").write_text("", encoding="utf-8")
    kernel.run(max_beats=1, heartbeat_interval=0)

    assert machine.mode == Mode.KILLED
    assert reservation_ledger.total_reserved() == MoneyMicros(0)
    assert reservation_ledger.used_intent_ids() >= {"pending-1", "pending-2"}
    assert alert_sink.count(AlertType.HALT_KILL) == 1
    assert len(directive_sink.received) == 1

    killed_decision = kernel.evaluate_intent(pre_kill_intent, context)
    assert killed_decision.reasons == ("KILLED",)

    phrase = switch.expected_rearm_phrase(switch.active_kill_sequence)
    (tmp_path / "REARM").write_text(phrase, encoding="utf-8")
    watcher.poll_once(now_epoch_s=_FIXED_EPOCH_S + 1)

    assert machine.mode == Mode.PAUSED
    assert not (tmp_path / "REARM").exists()
    assert not (tmp_path / "KILL").exists()

    post_rearm_decision = kernel.evaluate_intent(pre_kill_intent, context)
    assert "KILLED" not in post_rearm_decision.reasons

    with pytest.raises(DuplicateReservationError):
        reservation_ledger.reserve(
            "pending-1",
            MoneyMicros(500_000),
            "idem-p1-replay",
            expires_at=_FAR_FUTURE_EXPIRY_S,
        )


# --- CLI: `windbreak kill` / `windbreak rearm` -------------------------------------


def test_main_kill_subcommand_writes_a_kill_file_and_exits_zero(tmp_path: Path) -> None:
    """`windbreak kill --state-dir DIR` writes a `KILL` file into `DIR` and
    exits 0.
    """
    exit_code = windbreak_main(["kill", "--state-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "KILL").exists()


def test_main_rearm_subcommand_writes_the_typed_phrase_verbatim_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`windbreak rearm --state-dir DIR` reads the confirmation phrase from its
    injected stdin reader and writes it *verbatim* (no stripping, no case
    change) to a `REARM` file in `DIR`, exiting 0.
    """
    typed_phrase = "RE-ARM KILL 7: I ACCEPT FULL RESPONSIBILITY  "
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: typed_phrase)

    exit_code = windbreak_main(["rearm", "--state-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "REARM").read_text(encoding="utf-8") == typed_phrase


# --- Defaults: the fallback clock and the idle (no-trigger) poll -----------------


def test_kill_without_an_injected_clock_stamps_an_integer_wall_clock_epoch() -> None:
    """A `KillSwitch` built with no `clock` falls back to the wall clock, whose
    epoch is a plain `int` (never a float -- SPEC S6.1) close to `time.time()`.
    """
    writer = InMemoryKernelLedgerWriter()
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE)
    switch = KillSwitch(machine, writer, _FakeAlertSink())

    before = int(time.time())
    switch.kill(KillTrigger.CLI)
    after = int(time.time())

    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    epoch = kill_events[0].payload["epoch"]
    assert isinstance(epoch, int)
    assert before <= epoch <= after


def test_poll_once_with_no_kill_file_and_a_live_switch_is_a_safe_no_op(
    tmp_path: Path,
) -> None:
    """When the switch is not KILLED and no `KILL` file is present, `poll_once`
    leaves the mode untouched and ledgers nothing -- the idle steady state.
    """
    switch, writer, machine, _sink = _build_switch(state_dir=tmp_path)
    watcher = KillFileWatcher(switch, tmp_path)

    watcher.poll_once(now_epoch_s=_FIXED_EPOCH_S)

    assert machine.mode == Mode.LIVE
    assert writer.events == []


# --- kill_state_in: the pure fold over kill/re-arm event history (issue #123) ---


def test_kill_state_in_empty_events_is_not_killed_with_sequence_zero() -> None:
    """`kill_state_in([])` folds to the not-killed, zero-sequence baseline --
    mirroring `override_applied_in([])`'s equivalent empty-history baseline.
    """
    assert kill_state_in([]) == ReplayedKillState(last_kill_sequence=0, killed=False)


def test_kill_state_in_detects_an_unrearmed_kill_engaged() -> None:
    """A lone `KillEngaged`, never re-armed, folds to `killed=True` at its own
    sequence number.
    """
    events = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=_FIXED_EPOCH_S
        ),
    ]

    assert kill_state_in(events) == ReplayedKillState(last_kill_sequence=1, killed=True)


def test_kill_state_in_ignores_unrelated_events() -> None:
    """A history of unrelated events (a heartbeat, a generic base `Event`)
    never trips `kill_state_in`, mirroring `test_override.py::
    test_override_applied_in_ignores_unrelated_events`'s equivalent fixture.
    """
    events: list[Event] = [
        ModeHeartbeat(component="riskkernel", mode="RESEARCH", beat=1),
        Event(
            event_type="Something",
            component="riskkernel",
            payload_schema_version=1,
            payload={},
        ),
    ]

    assert kill_state_in(events) == ReplayedKillState(
        last_kill_sequence=0, killed=False
    )


def test_kill_state_in_rearmed_kill_is_not_killed_but_keeps_last_sequence() -> None:
    """A kill immediately re-armed at the matching sequence nets
    `killed=False`, but the last sequence number survives the fold -- a
    re-arm never resets the monotonic counter, only clears the kill flag.
    """
    events = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=_FIXED_EPOCH_S
        ),
        KillReArmed(component="riskkernel", kill_sequence=1),
    ]

    assert kill_state_in(events) == ReplayedKillState(
        last_kill_sequence=1, killed=False
    )


def test_kill_state_in_multiple_cycles_restores_the_latest_sequence() -> None:
    """Two full kill/re-arm cycles fold to the *latest* sequence at each
    stage: `killed=True` after the second kill (not the first's stale
    sequence), and `killed=False` once that second kill is also re-armed.
    """
    events_after_second_kill = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=_FIXED_EPOCH_S
        ),
        KillReArmed(component="riskkernel", kill_sequence=1),
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=2, epoch=_FIXED_EPOCH_S
        ),
    ]

    assert kill_state_in(events_after_second_kill) == ReplayedKillState(
        last_kill_sequence=2, killed=True
    )

    events_after_second_rearm = [
        *events_after_second_kill,
        KillReArmed(component="riskkernel", kill_sequence=2),
    ]

    assert kill_state_in(events_after_second_rearm) == ReplayedKillState(
        last_kill_sequence=2, killed=False
    )


def test_kill_state_in_mismatched_rearm_sequence_stays_killed() -> None:
    """A `KillReArmed` whose sequence does not match the latest
    `KillEngaged` is a stale/mismatched record: it never un-kills.
    """
    events = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=2, epoch=_FIXED_EPOCH_S
        ),
        KillReArmed(component="riskkernel", kill_sequence=1),
    ]

    assert kill_state_in(events) == ReplayedKillState(last_kill_sequence=2, killed=True)


def test_kill_state_in_rearm_before_any_kill_is_not_killed() -> None:
    """A `KillReArmed` with no preceding `KillEngaged` at all leaves the fold
    at its not-killed, zero-sequence baseline -- there is nothing yet to
    clear, and the re-arm record itself never advances the sequence.
    """
    events = [KillReArmed(component="riskkernel", kill_sequence=1)]

    assert kill_state_in(events) == ReplayedKillState(
        last_kill_sequence=0, killed=False
    )


# --- RiskKernel.from_events: replaying kill state on rebuild (issue #123) -------


@pytest.mark.timeout(30)
def test_rebuilt_kernel_over_unrearmed_kill_history_comes_back_killed() -> None:
    """`RiskKernel.from_events` over an unrearmed kill history rebuilds
    already `KILLED` -- both in `.mode` and in `evaluate_intent`'s behavior,
    which must return the single `"KILLED"` reason exactly as
    `test_evaluate_intent_on_a_killed_kernel_returns_the_single_killed_reason`
    pins for an in-process kill.
    """
    events = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=_FIXED_EPOCH_S
        ),
    ]

    rebuilt = RiskKernel.from_events(events, InMemoryKernelLedgerWriter())

    assert rebuilt.mode is Mode.KILLED
    decision = rebuilt.evaluate_intent(make_intent(), make_context())
    assert decision.reasons == ("KILLED",)


def test_rebuilt_kernel_over_a_rearmed_history_keeps_the_passed_mode() -> None:
    """A net-re-armed kill history (a kill immediately followed by its own
    matching re-arm) never forces `KILLED` on rebuild: the rebuilt kernel
    keeps whatever mode the passed `mode_machine` was already in.
    """
    events = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=_FIXED_EPOCH_S
        ),
        KillReArmed(component="riskkernel", kill_sequence=1),
    ]
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.PAUSED)

    rebuilt = RiskKernel.from_events(
        events, InMemoryKernelLedgerWriter(), mode_machine=machine
    )

    assert rebuilt.mode is Mode.PAUSED


def test_rebuild_with_a_machine_already_in_killed_does_not_raise() -> None:
    """Rebuilding over an unrearmed kill history with a `mode_machine` that
    is already `KILLED` must never attempt a redundant `KILLED -> KILLED`
    transition (illegal on the ladder): it must not raise
    `IllegalModeTransitionError`, and the kernel comes back `KILLED`.
    """
    events = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=_FIXED_EPOCH_S
        ),
    ]
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.KILLED)

    rebuilt = RiskKernel.from_events(
        events, InMemoryKernelLedgerWriter(), mode_machine=machine
    )

    assert rebuilt.mode is Mode.KILLED


def test_rebuild_replays_override_and_kill_state_together() -> None:
    """One history recording both a significance override and an unrearmed
    kill rebuilds with *both* durable effects in force at once: `KILLED`
    mode and the `LIVE_MICRO` override ceiling -- the two replays (issue #33's
    override fold and issue #123's kill fold) compose independently.
    """
    events = [
        SignificanceOverrideApplied(
            component="riskkernel",
            operator_ack=SIGNIFICANCE_OVERRIDE_ACK_PHRASE,
            ceiling="LIVE_MICRO",
        ),
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=_FIXED_EPOCH_S
        ),
    ]
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.RESEARCH)

    rebuilt = RiskKernel.from_events(
        events, InMemoryKernelLedgerWriter(), mode_machine=machine
    )

    assert rebuilt.mode is Mode.KILLED
    assert rebuilt.mode_ceiling_effective is Mode.LIVE_MICRO


def test_from_events_accepts_a_single_pass_iterator() -> None:
    """`from_events` accepts a single-pass iterator, not just a list: it must
    materialize `events` internally rather than consuming it twice (once for
    the override fold, once for the kill fold) -- a naive double-consume of a
    generator would silently fail open, never observing the kill on the
    already-exhausted second pass.
    """
    events_iterator = iter(
        [
            KillEngaged(
                component="riskkernel",
                trigger="CLI",
                kill_sequence=1,
                epoch=_FIXED_EPOCH_S,
            ),
        ]
    )

    rebuilt = RiskKernel.from_events(events_iterator, InMemoryKernelLedgerWriter())

    assert rebuilt.mode is Mode.KILLED


# --- KillSwitch.from_events: re-arm authority survives a restart (issue #123) ---


def test_kill_switch_from_events_restores_the_active_kill_sequence_for_rearm() -> None:
    """A `KillSwitch` rebuilt via `from_events` over the same unrearmed kill
    history that drives a shared `RiskKernel.from_events`'s mode machine to
    `KILLED` restores the switch's own `active_kill_sequence`, so the
    operator's exact restored-sequence re-arm phrase succeeds and clears the
    shared machine.
    """
    events = [
        KillEngaged(
            component="riskkernel",
            trigger="CLI",
            kill_sequence=_RESTORED_KILL_SEQUENCE,
            epoch=_FIXED_EPOCH_S,
        ),
    ]
    shared_machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.RESEARCH)
    writer = InMemoryKernelLedgerWriter()
    RiskKernel.from_events(events, writer, mode_machine=shared_machine)
    assert shared_machine.mode is Mode.KILLED
    switch = KillSwitch.from_events(
        events, shared_machine, writer, _FakeAlertSink(), clock=lambda: _FIXED_EPOCH_S
    )

    switch.rearm(switch.expected_rearm_phrase(_RESTORED_KILL_SEQUENCE))

    assert shared_machine.mode == Mode.PAUSED
    rearmed_events = [
        event for event in writer.events if event.event_type == "KillReArmed"
    ]
    assert len(rearmed_events) == 1
    assert rearmed_events[0].payload["kill_sequence"] == _RESTORED_KILL_SEQUENCE


def test_rearm_after_restart_rejects_the_sequence_zero_phrase() -> None:
    """A restored switch's `active_kill_sequence` is the ledgered sequence,
    never a fresh 0: a phrase built for sequence 0 (the pre-restart default)
    is rejected, the shared machine stays `KILLED`, and nothing new is
    ledgered.
    """
    events = [
        KillEngaged(
            component="riskkernel",
            trigger="CLI",
            kill_sequence=_RESTORED_KILL_SEQUENCE,
            epoch=_FIXED_EPOCH_S,
        ),
    ]
    shared_machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.RESEARCH)
    writer = InMemoryKernelLedgerWriter()
    RiskKernel.from_events(events, writer, mode_machine=shared_machine)
    switch = KillSwitch.from_events(
        events, shared_machine, writer, _FakeAlertSink(), clock=lambda: _FIXED_EPOCH_S
    )
    events_before_rearm_attempt = list(writer.events)

    with pytest.raises(KillReArmError):
        switch.rearm(switch.expected_rearm_phrase(0))

    assert shared_machine.mode == Mode.KILLED
    assert writer.events == events_before_rearm_attempt


def test_kill_after_restart_increments_monotonically_from_the_restored_sequence() -> (
    None
):
    """Over a net-re-armed history ending at the restored sequence (so the
    machine is not `KILLED`), a switch rebuilt via `from_events` still
    restores `active_kill_sequence` unconditionally -- even though nothing is
    currently killed -- so its next `kill()` carries exactly
    `restored_sequence + 1`, never resetting back to 1.
    """
    events = [
        KillEngaged(
            component="riskkernel",
            trigger="CLI",
            kill_sequence=_RESTORED_KILL_SEQUENCE,
            epoch=_FIXED_EPOCH_S,
        ),
        KillReArmed(component="riskkernel", kill_sequence=_RESTORED_KILL_SEQUENCE),
    ]
    mode_machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.PAUSED)
    writer = InMemoryKernelLedgerWriter()
    switch = KillSwitch.from_events(
        events, mode_machine, writer, _FakeAlertSink(), clock=lambda: _FIXED_EPOCH_S
    )

    switch.kill(KillTrigger.CLI)

    kill_events = [
        event for event in writer.events if event.event_type == "KillEngaged"
    ]
    assert len(kill_events) == 1
    assert kill_events[0].payload["kill_sequence"] == _RESTORED_KILL_SEQUENCE + 1


def test_multi_cycle_history_ending_killed_rebuilds_end_to_end() -> None:
    """A multi-cycle history that nets `KILLED` at a sequence above 1 rebuilds
    correctly through *both* composed `from_events` methods over one shared
    machine: the kernel comes back `KILLED`, the switch restores the latest
    sequence so its exact re-arm phrase clears the machine, and the switch's
    next kill after that re-arm increments monotonically -- never resetting.
    """
    events = [
        KillEngaged(
            component="riskkernel",
            trigger="CLI",
            kill_sequence=_EARLIER_KILL_SEQUENCE,
            epoch=_FIXED_EPOCH_S,
        ),
        KillReArmed(component="riskkernel", kill_sequence=_EARLIER_KILL_SEQUENCE),
        KillEngaged(
            component="riskkernel",
            trigger="AUTO_RECONCILIATION",
            kill_sequence=_RESTORED_KILL_SEQUENCE,
            epoch=_FIXED_EPOCH_S,
        ),
    ]
    shared_machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.RESEARCH)
    writer = InMemoryKernelLedgerWriter()
    kernel = RiskKernel.from_events(events, writer, mode_machine=shared_machine)
    assert kernel.mode is Mode.KILLED
    switch = KillSwitch.from_events(
        events, shared_machine, writer, _FakeAlertSink(), clock=lambda: _FIXED_EPOCH_S
    )

    switch.rearm(switch.expected_rearm_phrase(_RESTORED_KILL_SEQUENCE))
    assert shared_machine.mode is Mode.PAUSED

    switch.kill(KillTrigger.CLI)

    assert switch.active_kill_sequence == _RESTORED_KILL_SEQUENCE + 1
