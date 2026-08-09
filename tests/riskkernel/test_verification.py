"""Failing-first tests for windbreak.riskkernel.verification (issue #32, RED).

Issue #32 gives the Risk Kernel read-only exchange verification (SPEC S5.2 /
S10.3): each cycle, `ReadOnlyVerifier.run_cycle` cross-checks a read-only
connector's exchange-verified balances / positions / open orders against
ledger-derived `LedgerExpectations`, classifies the result as
`VerificationOutcome.CLEAN` / `DRIFT_WITHIN_TOLERANCE` / `BREACH`, fires
`AlertType.JURISDICTION_UNKNOWN` for any held market whose jurisdiction status
is unknown, fires `AlertType.RECONCILIATION_MISMATCH` on a breach, and records
exactly one bare `Event` per cycle. `RiskKernel.run_verification_cycle`
transitions the kernel to `Mode.HALT` on a breach (unless already HALT or
KILLED, both illegal targets on the mode ladder), and `RiskKernel.evaluate_intent`
-- when a verifier is configured -- stamps the latest snapshot onto the
evaluated context and rewrites `account.exchange_verified_available_cash` /
`reconciliation_uncertainty_buffer` from it, so verification feeds the floor.

`windbreak/riskkernel/verification.py` does not exist yet, so every import
below fails collection with `ModuleNotFoundError: No module named
'windbreak.riskkernel.verification'` -- the expected Gate 1 RED state for
issue #32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.riskkernel.conftest import DEFAULT_NOW_EPOCH_S, make_context, make_intent
from windbreak.alerts.dispatch import AlertDispatcher, LoggingLedgerWriter
from windbreak.alerts.registry import AlertSeverity, AlertType, get_registration
from windbreak.connector.fake import FakeExchange
from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.models import (
    BalanceSemantics,
    BalanceSnapshot,
    ExchangeStatus,
    FeeModel,
    Fill,
    NormalizedMarket,
    OpenOrder,
    OrderBookSnapshot,
    Position,
)
from windbreak.connector.paper import PaperExchange, PaperOrderIntent
from windbreak.connector.semantics import (
    CancelCollateralRelease,
    FeeDebitTiming,
    FeeRounding,
    HaltedMarketBehavior,
    OrderCollateralInAvailable,
    OrderCollateralInTotal,
    PartialFillRepresentation,
    UnsettledProceeds,
)
from windbreak.ledger.events import (
    CancelAllDirective,
    ConfigLoaded,
    Event,
    KillEngaged,
    KillReArmed,
    PositionsSnapshotRecorded,
)
from windbreak.numeric.types import ContractCentis, MoneyMicros, PricePips
from windbreak.riskkernel.modes import Mode, ModeStateMachine
from windbreak.riskkernel.process import (
    InMemoryKernelLedgerWriter,
    RiskKernel,
    _default_clock,
)
from windbreak.riskkernel.verification import (
    LedgerExpectations,
    LedgerExpectationSource,
    ReadOnlyVerifier,
    VerificationOutcome,
    VerificationTolerances,
)

#: `tests/fixtures/verification/<scenario>` -- each a full `FakeExchange`
#: fixture directory copied from `tests/fixtures/exchange/` with exactly the
#: fields the scenario needs tweaked (see each fixture dir's own JSON).
_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "verification"

#: The baseline ledger-expected values shared by every scenario below unless
#: a test deliberately overrides one: the "clean"/"drift"/"breach" fixture
#: dirs all agree the ledger expects $95.00 available cash and a 500-centi
#: KXFED-24DEC position, with zero open orders (`FakeExchange.get_open_orders`
#: is hard-coded to return `()`, so the open-order dimension is driven purely
#: from the expectations side -- see the open-order tests below).
_BASELINE_EXPECTED_CASH = MoneyMicros(95_000_000)
_BASELINE_EXPECTED_POSITIONS = {"KXFED-24DEC": ContractCentis(500)}

#: Zero-drift tolerance singletons held at module scope (the scaled-int wrapper
#: types are frozen, so one shared instance is safe) so the helper builders can
#: default to them without a function call in an argument default (ruff B008).
_ZERO_TOLERANCE_MICROS = MoneyMicros(0)
_ZERO_TOLERANCE_CENTIS = ContractCentis(0)


@dataclass
class _StaticExpectationSource:
    """A fake `ExpectationSource` that always returns one fixed snapshot."""

    expectations: LedgerExpectations

    def get_expectations(self) -> LedgerExpectations:
        """Return the fixed `LedgerExpectations`, ignoring all state."""
        return self.expectations


@dataclass
class _RecordingSink:
    """A fake `AlertSink` that records every call without raising."""

    name: str = "recording"
    calls: list[tuple[AlertType, AlertSeverity, str]] = field(default_factory=list)

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Record the call."""
        self.calls.append((alert_type, severity, message))


def _fixture_path(name: str) -> Path:
    """Return the absolute path to a named verification fixture directory.

    Args:
        name: The scenario directory name under `tests/fixtures/verification/`.

    Returns:
        The absolute `Path` to that directory.
    """
    return _FIXTURES_DIR / name


def _make_verifier(
    fixture_name: str,
    *,
    expected_available_cash: MoneyMicros = _BASELINE_EXPECTED_CASH,
    expected_positions: dict[str, ContractCentis] | None = None,
    expected_open_order_ids: frozenset[str] = frozenset(),
    balance_tolerance: MoneyMicros = _ZERO_TOLERANCE_MICROS,
    position_tolerance: ContractCentis = _ZERO_TOLERANCE_CENTIS,
) -> tuple[ReadOnlyVerifier, InMemoryKernelLedgerWriter, _RecordingSink]:
    """Build a `ReadOnlyVerifier` wired to a named fixture, plus its spies.

    Args:
        fixture_name: The scenario directory under `tests/fixtures/verification/`.
        expected_available_cash: The ledger-expected available cash.
        expected_positions: The ledger-expected per-ticker positions; defaults
            to `_BASELINE_EXPECTED_POSITIONS`.
        expected_open_order_ids: The ledger-expected open-order id set.
        balance_tolerance: The balance-dimension tolerance.
        position_tolerance: The position-dimension tolerance.

    Returns:
        A `(verifier, kernel_ledger_writer, alert_sink)` triple, so a test can
        both run a cycle and inspect what it recorded/dispatched.
    """
    positions = (
        _BASELINE_EXPECTED_POSITIONS
        if expected_positions is None
        else expected_positions
    )
    ledger_writer = InMemoryKernelLedgerWriter()
    sink = _RecordingSink()
    dispatcher = AlertDispatcher([sink], ledger_writer=LoggingLedgerWriter())
    verifier = ReadOnlyVerifier(
        connector=FakeExchange.from_fixture_dir(_fixture_path(fixture_name)),
        expectation_source=_StaticExpectationSource(
            LedgerExpectations(
                expected_available_cash=expected_available_cash,
                expected_positions=positions,
                expected_open_order_ids=expected_open_order_ids,
            )
        ),
        tolerances=VerificationTolerances(
            balance_tolerance=balance_tolerance,
            position_tolerance=position_tolerance,
        ),
        dispatcher=dispatcher,
        ledger_writer=ledger_writer,
    )
    return verifier, ledger_writer, sink


def _kernel_with_verifier(
    fixture_name: str,
    *,
    mode: Mode = Mode.LIVE,
    expected_available_cash: MoneyMicros = _BASELINE_EXPECTED_CASH,
    expected_positions: dict[str, ContractCentis] | None = None,
    balance_tolerance: MoneyMicros = _ZERO_TOLERANCE_MICROS,
    position_tolerance: ContractCentis = _ZERO_TOLERANCE_CENTIS,
) -> tuple[RiskKernel, InMemoryKernelLedgerWriter, _RecordingSink]:
    """Build a `RiskKernel` wired to a verifier and a fixed injected clock.

    Args:
        fixture_name: The scenario directory under `tests/fixtures/verification/`.
        mode: The kernel's starting operating mode.
        expected_available_cash: The ledger-expected available cash.
        expected_positions: The ledger-expected per-ticker positions.
        balance_tolerance: The balance-dimension tolerance.
        position_tolerance: The position-dimension tolerance.

    Returns:
        A `(kernel, kernel_ledger_writer, alert_sink)` triple. The kernel's
        `clock` is a fixed lambda returning `DEFAULT_NOW_EPOCH_S`, never
        `time.time`, so every cycle is deterministic.
    """
    verifier, ledger_writer, sink = _make_verifier(
        fixture_name,
        expected_available_cash=expected_available_cash,
        expected_positions=expected_positions,
        balance_tolerance=balance_tolerance,
        position_tolerance=position_tolerance,
    )
    mode_machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=mode)
    kernel = RiskKernel(
        ledger_writer,
        mode_machine=mode_machine,
        verifier=verifier,
        clock=lambda: DEFAULT_NOW_EPOCH_S,
    )
    return kernel, ledger_writer, sink


def _events_of_type(
    writer: InMemoryKernelLedgerWriter, event_type: str
) -> list[object]:
    """Return every recorded event of one exact `event_type`, in order.

    Args:
        writer: The in-memory ledger writer to read from.
        event_type: The exact `event_type` string to filter on.

    Returns:
        The matching events, in recorded order.
    """
    return [event for event in writer.events if event.event_type == event_type]


# --- Clean pass ------------------------------------------------------------------


def test_clean_pass_yields_clean_outcome_with_zero_drift() -> None:
    """A `FakeExchange` snapshot that matches `LedgerExpectations` exactly
    yields `VerificationOutcome.CLEAN`, every per-dimension `ok` flag `True`,
    zero cash drift, and the snapshot carries the observed available cash.
    """
    verifier, _, _ = _make_verifier("clean")

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.outcome is VerificationOutcome.CLEAN
    assert snapshot.balance_ok is True
    assert snapshot.position_ok is True
    assert snapshot.open_order_ok is True
    assert snapshot.cash_drift == MoneyMicros(0)
    assert snapshot.exchange_verified_available_cash == MoneyMicros(95_000_000)
    assert snapshot.semantics_fully_known is True
    assert snapshot.verified_at_epoch_s == DEFAULT_NOW_EPOCH_S


def test_clean_pass_records_exactly_one_verification_passed_event() -> None:
    """A clean cycle records exactly one `VerificationPassed` event, correctly
    shaped, and dispatches no alert."""
    verifier, writer, sink = _make_verifier("clean")

    verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    passed_events = _events_of_type(writer, "VerificationPassed")
    assert len(passed_events) == 1
    event = passed_events[0]
    assert event.component == "riskkernel"
    assert event.payload_schema_version == 1
    assert sink.calls == []


def test_clean_pass_through_the_kernel_never_halts() -> None:
    """Running a clean verification cycle through `RiskKernel` never changes
    the operating mode."""
    kernel, _, _ = _kernel_with_verifier("clean", mode=Mode.LIVE)

    kernel.run_verification_cycle()

    assert kernel.mode is Mode.LIVE


# --- Tolerable drift ---------------------------------------------------------------


def test_drift_within_tolerance_yields_drift_outcome_and_nonzero_drift() -> None:
    """A small balance mismatch that is still within `balance_tolerance`
    yields `DRIFT_WITHIN_TOLERANCE`, `balance_ok=True`, and a nonzero
    `cash_drift` reflecting the exact mismatch."""
    verifier, _, sink = _make_verifier(
        "drift_within_tolerance", balance_tolerance=MoneyMicros(1_000)
    )

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.outcome is VerificationOutcome.DRIFT_WITHIN_TOLERANCE
    assert snapshot.balance_ok is True
    assert snapshot.cash_drift == MoneyMicros(500)
    assert sink.calls == []


def test_drift_within_tolerance_records_exactly_one_verification_drift_event() -> None:
    """A tolerable-drift cycle records exactly one `VerificationDrift` event
    and dispatches no alert (drift within tolerance is not a mismatch)."""
    verifier, writer, sink = _make_verifier(
        "drift_within_tolerance", balance_tolerance=MoneyMicros(1_000)
    )

    verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    drift_events = _events_of_type(writer, "VerificationDrift")
    assert len(drift_events) == 1
    event = drift_events[0]
    assert event.component == "riskkernel"
    assert event.payload_schema_version == 1
    assert sink.calls == []


def test_drift_within_tolerance_never_halts_through_the_kernel() -> None:
    """A tolerable-drift cycle through `RiskKernel` never changes the
    operating mode."""
    kernel, _, _ = _kernel_with_verifier(
        "drift_within_tolerance", mode=Mode.LIVE, balance_tolerance=MoneyMicros(1_000)
    )

    kernel.run_verification_cycle()

    assert kernel.mode is Mode.LIVE


def test_drift_reduces_usable_equity_so_the_floor_check_flips() -> None:
    """Metamorphic: given the *same* floor and the *same* `OrderIntent`, a
    kernel wired to a clean verifier passes `floor_invariant`, while an
    otherwise-identical kernel wired to a drifted verifier vetoes with
    `"worst-case equity below floor"` -- proving the verification snapshot's
    drift really does feed `account.exchange_verified_available_cash` /
    `reconciliation_uncertainty_buffer`, and thus the floor computation.

    Clean: cash 95_000_000, drift 0 -> equity 95_000_000; equity - cost
    (5_000_000) == floor (90_000_000): passes at exact equality.
    Drift: cash 94_999_500, drift 500 -> equity 94_999_000; equity - cost ==
    89_999_000 < floor (90_000_000): vetoes.
    """
    floor = MoneyMicros(90_000_000)
    intent = make_intent()
    context = make_context(floor=floor)

    clean_kernel, _, _ = _kernel_with_verifier("clean", mode=Mode.LIVE)
    drift_kernel, _, _ = _kernel_with_verifier(
        "drift_within_tolerance", mode=Mode.LIVE, balance_tolerance=MoneyMicros(1_000)
    )
    clean_kernel.run_verification_cycle()
    drift_kernel.run_verification_cycle()

    clean_decision = clean_kernel.evaluate_intent(intent, context)
    drift_decision = drift_kernel.evaluate_intent(intent, context)

    assert "worst-case equity below floor" not in clean_decision.reasons
    assert "worst-case equity below floor" in drift_decision.reasons


# --- Breach -> HALT ----------------------------------------------------------------


def test_balance_breach_yields_breach_outcome() -> None:
    """A balance mismatch beyond tolerance yields `BREACH` and
    `balance_ok=False`."""
    verifier, _, _ = _make_verifier(
        "balance_breach", balance_tolerance=MoneyMicros(1_000)
    )

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.outcome is VerificationOutcome.BREACH
    assert snapshot.balance_ok is False
    assert snapshot.cash_drift == MoneyMicros(5_000_000)


def test_balance_breach_dispatches_reconciliation_mismatch_and_ledgers_it() -> None:
    """A breach dispatches `AlertType.RECONCILIATION_MISMATCH` (severity
    CRITICAL, per the alert registry) and records exactly one
    `VerificationMismatch` event."""
    verifier, writer, sink = _make_verifier(
        "balance_breach", balance_tolerance=MoneyMicros(1_000)
    )

    verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    mismatch_calls = [
        call for call in sink.calls if call[0] is AlertType.RECONCILIATION_MISMATCH
    ]
    assert len(mismatch_calls) == 1
    assert (
        mismatch_calls[0][1]
        == get_registration(AlertType.RECONCILIATION_MISMATCH).severity
    )
    mismatch_events = _events_of_type(writer, "VerificationMismatch")
    assert len(mismatch_events) == 1
    assert mismatch_events[0].component == "riskkernel"
    assert mismatch_events[0].payload_schema_version == 1


def test_breach_through_the_kernel_halts_and_ledgers_the_transition() -> None:
    """A breach cycle run through `RiskKernel.run_verification_cycle`
    transitions the kernel to `Mode.HALT` and records exactly one
    `VerificationMismatchHalt` event, in addition to the verifier's own
    `VerificationMismatch` event."""
    kernel, writer, sink = _kernel_with_verifier(
        "balance_breach", mode=Mode.LIVE, balance_tolerance=MoneyMicros(1_000)
    )

    kernel.run_verification_cycle()

    assert kernel.mode is Mode.HALT
    assert len(_events_of_type(writer, "VerificationMismatch")) == 1
    halt_events = _events_of_type(writer, "VerificationMismatchHalt")
    assert len(halt_events) == 1
    assert halt_events[0].component == "riskkernel"
    assert halt_events[0].payload_schema_version == 1
    assert any(call[0] is AlertType.RECONCILIATION_MISMATCH for call in sink.calls)


# --- HALT idempotency --------------------------------------------------------------


def test_repeated_breach_while_already_halted_stays_halt_without_raising() -> None:
    """A second breach cycle, run while the kernel is already `HALT`, leaves
    the mode at `HALT` (a same-mode "transition" that `ModeStateMachine`
    itself would reject), raises nothing, and never records a second
    `VerificationMismatchHalt` event -- the halt only fires once, on the
    actual transition."""
    kernel, writer, _ = _kernel_with_verifier(
        "balance_breach", mode=Mode.LIVE, balance_tolerance=MoneyMicros(1_000)
    )

    kernel.run_verification_cycle()
    kernel.run_verification_cycle()

    assert kernel.mode is Mode.HALT
    assert len(_events_of_type(writer, "VerificationMismatchHalt")) == 1
    assert len(_events_of_type(writer, "VerificationMismatch")) == 2


def test_breach_while_killed_stays_killed_without_raising() -> None:
    """A breach cycle run while the kernel is `KILLED` leaves the mode at
    `KILLED` -- `HALT` is not a legal target from `KILLED` (a dead end on the
    mode ladder) -- and raises no `IllegalModeTransitionError`."""
    kernel, writer, _ = _kernel_with_verifier(
        "balance_breach", mode=Mode.KILLED, balance_tolerance=MoneyMicros(1_000)
    )

    kernel.run_verification_cycle()

    assert kernel.mode is Mode.KILLED
    assert _events_of_type(writer, "VerificationMismatchHalt") == []


# --- Position dimension (driven from expectations, over the "clean" fixture) ------
#
# The "clean" fixture's observed position is fixed at 500 centis of
# KXFED-24DEC; every test below holds the balance/open-order dimensions at
# their exact matching baseline and varies only `expected_positions` /
# `position_tolerance`, isolating the position dimension precisely.


def test_position_within_tolerance_yields_drift_outcome() -> None:
    """A per-ticker position drift within `position_tolerance` is `ok` and
    yields `DRIFT_WITHIN_TOLERANCE` (observed 500, expected 505, diff 5,
    tolerance 10)."""
    verifier, _, _ = _make_verifier(
        "clean",
        expected_positions={"KXFED-24DEC": ContractCentis(505)},
        position_tolerance=ContractCentis(10),
    )

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.position_ok is True
    assert snapshot.outcome is VerificationOutcome.DRIFT_WITHIN_TOLERANCE


def test_position_passes_at_exact_tolerance_boundary() -> None:
    """A per-ticker position diff exactly equal to `position_tolerance`
    passes (inclusive boundary, matching every other tolerance/ttl check in
    this codebase): observed 500, expected 510, diff 10, tolerance 10."""
    verifier, _, _ = _make_verifier(
        "clean",
        expected_positions={"KXFED-24DEC": ContractCentis(510)},
        position_tolerance=ContractCentis(10),
    )

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.position_ok is True


def test_position_vetoes_one_centi_over_the_tolerance_boundary() -> None:
    """One centi past the position tolerance boundary is a breach: observed
    500, expected 511, diff 11, tolerance 10."""
    verifier, _, _ = _make_verifier(
        "clean",
        expected_positions={"KXFED-24DEC": ContractCentis(511)},
        position_tolerance=ContractCentis(10),
    )

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.position_ok is False
    assert snapshot.outcome is VerificationOutcome.BREACH


def test_unexpected_ticker_held_is_a_breach() -> None:
    """A ticker the exchange reports a position in, but the ledger expects
    nothing for at all, is a breach: the observed 500-centi KXFED-24DEC
    position diffs against an implicit zero expectation, far exceeding a
    tight tolerance."""
    verifier, _, _ = _make_verifier(
        "clean",
        expected_positions={},
        position_tolerance=ContractCentis(10),
    )

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.position_ok is False
    assert snapshot.outcome is VerificationOutcome.BREACH


# --- Open-order dimension (driven from expectations; FakeExchange always ()) ------


def test_open_orders_exact_empty_match_passes() -> None:
    """An empty expected-open-order-id set matches `FakeExchange`'s
    hard-coded empty observed set exactly: `open_order_ok` is `True`."""
    verifier, _, _ = _make_verifier("clean", expected_open_order_ids=frozenset())

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.open_order_ok is True


def test_open_order_expected_but_absent_from_observed_is_a_breach() -> None:
    """An expected open-order id absent from the (always-empty) observed set
    is a breach -- open orders are discrete, so there is no tolerance: any
    mismatch at all is a breach, driven purely from the expectations side
    since `FakeExchange.get_open_orders` always returns `()`."""
    verifier, writer, sink = _make_verifier(
        "clean", expected_open_order_ids=frozenset({"order-not-on-exchange"})
    )

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.open_order_ok is False
    assert snapshot.outcome is VerificationOutcome.BREACH
    assert any(call[0] is AlertType.RECONCILIATION_MISMATCH for call in sink.calls)
    assert len(_events_of_type(writer, "VerificationMismatch")) == 1


@dataclass
class _ConnectorServingOpenOrders:
    """A read-only connector delegating to a `FakeExchange` but serving fixed
    positions and open orders.

    `FakeExchange.get_open_orders` is hard-coded to `()`, so this thin wrapper
    is the seam that exercises the verifier's *observed* open-order path (id and
    ticker extraction) with a non-empty resting-order set, without adding any
    trade-capable surface -- only the read-only methods `run_cycle` calls are
    overridden; everything else falls through to the inner fake.
    """

    inner: FakeExchange
    positions: tuple[Position, ...]
    open_orders: tuple[OpenOrder, ...]

    def get_positions(self) -> tuple[Position, ...]:
        """Return the fixed positions."""
        return self.positions

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Return the fixed open orders."""
        return self.open_orders

    def get_balances(self) -> BalanceSnapshot:
        """Delegate balances to the inner fake exchange."""
        return self.inner.get_balances()

    def get_balance_semantics(self) -> BalanceSemantics:
        """Delegate balance semantics to the inner fake exchange."""
        return self.inner.get_balance_semantics()

    def get_market(self, ticker: str) -> object:
        """Delegate market lookup to the inner fake exchange."""
        return self.inner.get_market(ticker)


def test_observed_open_order_matches_and_flags_its_unknown_jurisdiction() -> None:
    """A verifier reading a non-empty *observed* open-order set matches it by
    id against the ledger's expected set (so `open_order_ok` is `True` when they
    agree), and treats the order's market as a held market for the
    jurisdiction-unknown alert.

    The resting order rests in KXWEA-24DEC (jurisdiction "unknown" in the
    fixture), while the sole position is the eligible KXFED-24DEC, so the one
    `JURISDICTION_UNKNOWN` alert is attributable purely to the *order's* ticker
    -- the observed-open-order code path FakeExchange's hard-coded `()` cannot
    reach.
    """
    connector = _ConnectorServingOpenOrders(
        inner=FakeExchange.from_fixture_dir(_fixture_path("jurisdiction_unknown")),
        positions=(
            Position(
                ticker="KXFED-24DEC",
                quantity=ContractCentis(500),
                average_price=PricePips(4550),
            ),
        ),
        open_orders=(
            OpenOrder(
                id="order-1",
                ticker="KXWEA-24DEC",
                side="yes",
                price=PricePips(5000),
                quantity=ContractCentis(100),
            ),
        ),
    )
    ledger_writer = InMemoryKernelLedgerWriter()
    sink = _RecordingSink()
    verifier = ReadOnlyVerifier(
        connector=connector,
        expectation_source=_StaticExpectationSource(
            LedgerExpectations(
                expected_available_cash=_BASELINE_EXPECTED_CASH,
                expected_positions={"KXFED-24DEC": ContractCentis(500)},
                expected_open_order_ids=frozenset({"order-1"}),
            )
        ),
        tolerances=VerificationTolerances(
            balance_tolerance=MoneyMicros(0),
            position_tolerance=ContractCentis(0),
        ),
        dispatcher=AlertDispatcher([sink], ledger_writer=LoggingLedgerWriter()),
        ledger_writer=ledger_writer,
    )

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.open_order_ok is True
    assert snapshot.outcome is VerificationOutcome.CLEAN
    jurisdiction_calls = [
        call for call in sink.calls if call[0] is AlertType.JURISDICTION_UNKNOWN
    ]
    assert len(jurisdiction_calls) == 1
    assert "KXWEA-24DEC" in jurisdiction_calls[0][2]


# --- BalanceSemantics: unknown field surfaced, live-mode gate is checks.py's job --


def test_semantics_unknown_field_surfaces_as_not_fully_known() -> None:
    """A `balance_semantics.json` with one field left `UNKNOWN` surfaces as
    `semantics_fully_known=False` on the snapshot -- reconciliation itself
    still passes (balances/positions/open-orders all match), since the
    live-mode trading gate on unknown semantics is `checks.balance_reconciliation`'s
    job (see `tests/riskkernel/test_checks.py`), not the verifier's."""
    verifier, _, sink = _make_verifier("semantics_unknown_field")

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.semantics_fully_known is False
    assert snapshot.outcome is VerificationOutcome.CLEAN
    assert sink.calls == []


# --- Jurisdiction unknown: WARNING alert, no HALT ---------------------------------


def test_held_position_in_unknown_jurisdiction_dispatches_warning_alert() -> None:
    """A held position (KXWEA-24DEC, jurisdiction "unknown" in the fixture's
    `markets.json`) dispatches exactly one `AlertType.JURISDICTION_UNKNOWN`
    alert (severity WARNING, per the alert registry) -- reconciliation itself
    stays clean (both positions match their expectations exactly), and no
    `RECONCILIATION_MISMATCH` alert fires."""
    verifier, writer, sink = _make_verifier(
        "jurisdiction_unknown",
        expected_positions={
            "KXFED-24DEC": ContractCentis(500),
            "KXWEA-24DEC": ContractCentis(100),
        },
    )

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.outcome is VerificationOutcome.CLEAN
    jurisdiction_calls = [
        call for call in sink.calls if call[0] is AlertType.JURISDICTION_UNKNOWN
    ]
    assert len(jurisdiction_calls) == 1
    assert (
        jurisdiction_calls[0][1]
        == get_registration(AlertType.JURISDICTION_UNKNOWN).severity
    )
    assert "KXWEA-24DEC" in jurisdiction_calls[0][2]
    assert not any(call[0] is AlertType.RECONCILIATION_MISMATCH for call in sink.calls)
    assert _events_of_type(writer, "VerificationMismatch") == []


def test_jurisdiction_unknown_alert_never_halts_the_kernel() -> None:
    """A jurisdiction-unknown alert alone (outcome still CLEAN) never changes
    the kernel's operating mode -- only a `BREACH` outcome halts."""
    verifier, ledger_writer, _ = _make_verifier(
        "jurisdiction_unknown",
        expected_positions={
            "KXFED-24DEC": ContractCentis(500),
            "KXWEA-24DEC": ContractCentis(100),
        },
    )
    mode_machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE)
    kernel = RiskKernel(
        ledger_writer,
        mode_machine=mode_machine,
        verifier=verifier,
        clock=lambda: DEFAULT_NOW_EPOCH_S,
    )

    kernel.run_verification_cycle()

    assert kernel.mode is Mode.LIVE


# --- run(): the verification cycle fires once per beat when configured -----------


def test_kernel_run_invokes_the_verification_cycle_once_per_beat() -> None:
    """`RiskKernel.run(max_beats=N, ...)` invokes the verification cycle
    exactly N times when a verifier is configured -- one `VerificationPassed`
    event per beat, on top of the heartbeat events."""
    kernel, writer, _ = _kernel_with_verifier("clean", mode=Mode.LIVE)

    kernel.run(max_beats=3, heartbeat_interval=0)

    assert len(_events_of_type(writer, "VerificationPassed")) == 3


def test_default_clock_returns_a_nonnegative_int_off_the_float_path() -> None:
    """The kernel's default clock (used when no `clock` is injected) returns a
    whole-integer epoch second -- never a float -- so a verifier-configured
    kernel built without an explicit clock stays on the SPEC S6.1 no-float
    path."""
    now = _default_clock()

    assert isinstance(now, int)
    assert now > 0


def test_kernel_run_without_a_verifier_never_records_verification_events() -> None:
    """A `RiskKernel` built with no `verifier` (the pre-issue-#32 shape) never
    records any verification event, even across several beats -- the
    verifier-less kernel behaves exactly as before."""
    writer = InMemoryKernelLedgerWriter()
    kernel = RiskKernel(writer)

    kernel.run(max_beats=3, heartbeat_interval=0)

    assert _events_of_type(writer, "VerificationPassed") == []
    assert _events_of_type(writer, "VerificationDrift") == []
    assert _events_of_type(writer, "VerificationMismatch") == []


# --- evaluate_intent: fail-closed before the first cycle --------------------------


def test_evaluate_intent_stamps_none_verification_before_the_first_cycle() -> None:
    """With a verifier configured but `run_verification_cycle` never yet
    called, `evaluate_intent` stamps `verification=None` onto the effective
    context (fail-closed) rather than trusting the caller-supplied snapshot --
    so every reconciliation check vetoes on the missing snapshot."""
    kernel, _, _ = _kernel_with_verifier("clean", mode=Mode.LIVE)
    intent = make_intent()
    context = make_context()

    decision = kernel.evaluate_intent(intent, context)

    assert decision.vetoed is True
    assert "balance verification stale or missing" in decision.reasons
    assert "position verification stale or missing" in decision.reasons
    assert "open-order verification stale or missing" in decision.reasons


# --- Payload hygiene: every verification event payload is int/str only -----------


def test_every_verification_event_payload_is_int_str_or_bool_never_float() -> None:
    """Every payload value recorded by a verification event, across a clean,
    a tolerable-drift, and a breach cycle, is an `int`, `str`, or `bool` --
    never a `float`. Drift must be expressed as `.value` ints (SPEC S6.1)."""
    scenarios = [
        ("clean", {}),
        ("drift_within_tolerance", {"balance_tolerance": MoneyMicros(1_000)}),
        ("balance_breach", {"balance_tolerance": MoneyMicros(1_000)}),
    ]
    all_events: list[object] = []
    for fixture_name, tolerance_kwargs in scenarios:
        verifier, writer, _ = _make_verifier(fixture_name, **tolerance_kwargs)
        verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)
        all_events.extend(
            event
            for event in writer.events
            if event.event_type.startswith("Verification")
        )

    assert all_events, "expected at least one verification event to inspect"
    for event in all_events:
        for value in event.payload.values():
            assert not isinstance(value, float), f"{event.event_type}: {value!r}"
            assert isinstance(value, (int, str, bool)), f"{event.event_type}: {value!r}"


# --- LedgerExpectationSource (issue #288) -----------------------------------------
#
# `StartupBaselineExpectationSource` (issue #236) froze a connector's own
# startup snapshot because there was no ledger of "what the venue *should*
# hold" to read expectations from. `LedgerExpectationSource` replaces it: it
# folds the startup `history` once, at construction, into one frozen
# `LedgerExpectations`, per dimension --
#
#   * cash: the `exchange_verified_available_cash` of the LAST verification
#     event whose `event_type` is `"VerificationPassed"` or
#     `"VerificationDrift"` (a `"VerificationMismatch"` is IGNORED, so a
#     restart never re-baselines onto a breached value); else the connector's
#     `get_balances().available`.
#   * positions: the rows of the LAST `PositionsSnapshotRecorded` event,
#     mapped `{ticker: ContractCentis(quantity_centis)}`; else
#     `{p.ticker: p.quantity for p in connector.get_positions()}`.
#   * open orders: `frozenset()` only while the history ends KILLED and
#     unrearmed (`kill_state_in(history).killed` -- a `KillEngaged` with no
#     matching later `KillReArmed`, whose kill cancelled everything); once
#     re-armed (or never killed) it falls back to
#     `frozenset(o.id for o in connector.get_open_orders())`, so a past kill or
#     kill drill never permanently zeroes the expectation.
#
# Every dimension falls back independently, and -- exactly like the source it
# replaces -- the projection happens exactly once, at construction: a later
# connector mutation never changes what `get_expectations()` returns.
#
# `LedgerExpectationSource` does not exist yet, so every test below fails
# collection with `ImportError: cannot import name 'LedgerExpectationSource'
# from 'windbreak.riskkernel.verification'` -- the expected Gate 1 RED state
# for issue #288.

#: A fixed UTC instant for every `BalanceSnapshot.fetched_at` the stub
#: connectors below report; its exact value is irrelevant to every assertion.
_FIXED_DATETIME = datetime(2024, 1, 1, tzinfo=UTC)

#: A `BalanceSemantics` with every field a known (non-`UNKNOWN`) member,
#: reused by the stub connectors below wherever `get_balance_semantics` must
#: return something but no test actually inspects its value.
_FULLY_KNOWN_SEMANTICS = BalanceSemantics(
    open_order_collateral_in_total=OrderCollateralInTotal.EXCLUDED,
    open_order_collateral_in_available=OrderCollateralInAvailable.DEDUCTED_FROM_AVAILABLE,
    fee_debit_timing=FeeDebitTiming.AT_EXECUTION,
    fee_rounding=FeeRounding.EXACT,
    partial_fill_representation=PartialFillRepresentation.PER_FILL_RECORDS,
    cancel_collateral_release=CancelCollateralRelease.IMMEDIATE,
    unsettled_proceeds=UnsettledProceeds.INCLUDED_IMMEDIATELY,
    halted_market_behavior=HaltedMarketBehavior.NEW_ORDERS_REJECTED,
)


@dataclass
class _MutableBalanceConnector:
    """A minimal, mutable `MarketConnector` stub for exercising the
    connector-fallback side of `LedgerExpectationSource`'s per-dimension
    projection, and for proving it captures its snapshot once rather than
    reading the connector live on every call.

    Attributes:
        available: The account's current available cash, mutable so a test
            can change it after building a `LedgerExpectationSource` over this
            connector.
        positions: The account's fixed positions, returned verbatim by
            `get_positions` (empty by default, so a test that only cares
            about the cash or open-order dimension need not set it).
        open_orders: The account's fixed resting orders, returned verbatim by
            `get_open_orders` (empty by default).
    """

    available: MoneyMicros
    positions: tuple[Position, ...] = ()
    open_orders: tuple[OpenOrder, ...] = ()

    def get_balances(self) -> BalanceSnapshot:
        """Return the account's current, possibly-since-mutated available cash."""
        return BalanceSnapshot(
            total=self.available, available=self.available, fetched_at=_FIXED_DATETIME
        )

    def get_positions(self) -> tuple[Position, ...]:
        """Return the connector's fixed positions."""
        return self.positions

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Return the connector's fixed open orders."""
        return self.open_orders

    def get_balance_semantics(self) -> BalanceSemantics:
        """Return a fully-known `BalanceSemantics` (unused by this test)."""
        return _FULLY_KNOWN_SEMANTICS

    def list_markets(self) -> tuple[NormalizedMarket, ...]:
        """Return no markets; unused by this test."""
        return ()

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Raise; unused by this test."""
        raise UnknownMarketError(ticker)

    def get_order_book(self, ticker: str) -> OrderBookSnapshot:
        """Raise; unused by this test."""
        raise NotImplementedError(ticker)

    def get_exchange_status(self) -> ExchangeStatus:
        """Raise; unused by this test."""
        raise NotImplementedError

    def get_exchange_time(self) -> datetime:
        """Raise; unused by this test."""
        raise NotImplementedError

    def get_fills(self, since: datetime) -> tuple[Fill, ...]:
        """Return no fills; unused by this test."""
        del since
        return ()

    def get_fee_model(self, market_or_series: str) -> FeeModel:
        """Raise; unused by this test."""
        raise NotImplementedError(market_or_series)

    def place_order(self, normalized_intent: object, approval_token: object) -> object:
        """Raise; unused by this test."""
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> None:
        """Raise; unused by this test."""
        raise NotImplementedError(order_id)


def _verification_event(
    event_type: str, *, cash_micros: int, component: str = "riskkernel"
) -> Event:
    """Build a bare verification-cycle event carrying one cash observation.

    Mirrors the shape `ReadOnlyVerifier._record` emits (a bare `Event` whose
    `event_type` is one of `"VerificationPassed"` / `"VerificationDrift"` /
    `"VerificationMismatch"`), but populates only the one payload key
    `LedgerExpectationSource`'s cash projection folds
    (`exchange_verified_available_cash`) -- the other real-payload keys
    (`outcome`, `balance_ok`, ...) are irrelevant to the fold these tests
    exercise and are omitted for clarity.

    Args:
        event_type: The verification event's exact `event_type` string.
        cash_micros: The `exchange_verified_available_cash` value to carry,
            in micros.
        component: The ledger component to stamp the event with. Defaults to
            the kernel's own `"riskkernel"`, exactly as
            `ReadOnlyVerifier._record` stamps it, so only a test deliberately
            exercising the component filter ever passes anything else.

    Returns:
        The constructed bare `Event`.
    """
    return Event(
        event_type=event_type,
        component=component,
        payload_schema_version=1,
        payload={"exchange_verified_available_cash": cash_micros},
    )


def test_ledger_expectation_source_with_irrelevant_history_mirrors_the_connector() -> (
    None
):
    """With a history containing only an irrelevant event (`ConfigLoaded`),
    every dimension falls back to mirroring the connector exactly -- the same
    fallback semantics `StartupBaselineExpectationSource` always used."""
    connector = _MutableBalanceConnector(
        available=MoneyMicros(50_000_000),
        positions=(
            Position(
                ticker="KXFED-24DEC",
                quantity=ContractCentis(300),
                average_price=PricePips(4550),
            ),
        ),
        open_orders=(
            OpenOrder(
                id="order-9",
                ticker="KXFED-24DEC",
                side="yes",
                price=PricePips(5000),
                quantity=ContractCentis(50),
            ),
        ),
    )
    history = [ConfigLoaded(component="riskkernel", config_hash="abc", diff={})]

    source = LedgerExpectationSource(history, connector)
    expectations = source.get_expectations()

    assert expectations.expected_available_cash == connector.get_balances().available
    assert dict(expectations.expected_positions) == {
        position.ticker: position.quantity for position in connector.get_positions()
    }
    assert expectations.expected_open_order_ids == frozenset(
        order.id for order in connector.get_open_orders()
    )


def test_ledger_expectation_source_captures_snapshot_at_construction() -> None:
    """A connector mutated *after* construction never changes what an
    already-built `LedgerExpectationSource` reports: the projection happens
    once, at `__init__` time, never read live on a later `get_expectations()`
    call -- the same freeze guarantee `StartupBaselineExpectationSource` gave.
    """
    connector = _MutableBalanceConnector(available=MoneyMicros(1_000_000))
    source = LedgerExpectationSource([], connector)

    connector.available = MoneyMicros(2_000_000)

    assert source.get_expectations().expected_available_cash == MoneyMicros(1_000_000)


def test_ledger_expectation_source_cash_seeds_from_the_last_non_breach_event() -> None:
    """When `history` carries two non-breach verification events, the LAST
    one's cash wins -- over the connector *and* over the earlier event."""
    connector = _MutableBalanceConnector(available=MoneyMicros(999_000_000))
    history = [
        _verification_event("VerificationPassed", cash_micros=10_000_000),
        _verification_event("VerificationDrift", cash_micros=20_000_000),
    ]

    source = LedgerExpectationSource(history, connector)

    assert source.get_expectations().expected_available_cash == MoneyMicros(20_000_000)


def test_ledger_expectation_source_cash_ignores_a_trailing_breach_event() -> None:
    """A history ending in a `VerificationMismatch` after a
    `VerificationPassed`/`VerificationDrift` ignores the mismatch entirely:
    the seed stays at the prior clean/drift cash, never the breached value --
    a restart must never re-baseline onto a value already known to be wrong.
    """
    connector = _MutableBalanceConnector(available=MoneyMicros(1))
    history = [
        _verification_event("VerificationPassed", cash_micros=10_000_000),
        _verification_event("VerificationMismatch", cash_micros=999_000_000),
    ]

    source = LedgerExpectationSource(history, connector)

    assert source.get_expectations().expected_available_cash == MoneyMicros(10_000_000)


def test_ledger_expectation_source_positions_seed_from_the_last_snapshot() -> None:
    """The rows of the LAST `PositionsSnapshotRecorded` event win, mapped to
    `ContractCentis`: a later snapshot overrides an earlier one's ticker set
    entirely, not merely adding to it."""
    connector = _MutableBalanceConnector(available=MoneyMicros(1))
    history = [
        PositionsSnapshotRecorded(
            component="riskkernel",
            positions=[
                {
                    "ticker": "KXFED-24DEC",
                    "quantity_centis": 100,
                    "average_price_pips": 4500,
                }
            ],
        ),
        PositionsSnapshotRecorded(
            component="riskkernel",
            positions=[
                {
                    "ticker": "KXFED-24DEC",
                    "quantity_centis": 300,
                    "average_price_pips": 4600,
                },
                {
                    "ticker": "KXWEA-24DEC",
                    "quantity_centis": -50,
                    "average_price_pips": 5000,
                },
            ],
        ),
    ]

    source = LedgerExpectationSource(history, connector)
    expectations = source.get_expectations()

    assert dict(expectations.expected_positions) == {
        "KXFED-24DEC": ContractCentis(300),
        "KXWEA-24DEC": ContractCentis(-50),
    }


def test_ledger_expectation_source_open_orders_empty_while_killed_unrearmed() -> None:
    """A history ending in an unrearmed kill (a `KillEngaged` with no matching
    later `KillReArmed`, alongside its `CancelAllDirective`) expects zero open
    orders -- `frozenset()` -- even when the connector still reports resting
    orders: the kill cancelled everything and has not been re-armed, so nothing
    is expected to remain resting on the venue."""
    connector = _MutableBalanceConnector(
        available=MoneyMicros(1),
        open_orders=(
            OpenOrder(
                id="resting-1",
                ticker="KXFED-24DEC",
                side="yes",
                price=PricePips(5000),
                quantity=ContractCentis(10),
            ),
        ),
    )
    history = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=1_700_000_000
        ),
        CancelAllDirective(component="riskkernel", scope="all_open_orders"),
    ]

    source = LedgerExpectationSource(history, connector)

    assert source.get_expectations().expected_open_order_ids == frozenset()


def test_ledger_expectation_source_open_orders_fall_back_after_rearm() -> None:
    """A history whose last kill has been re-armed (`KillEngaged` -> its
    `CancelAllDirective` -> a matching `KillReArmed`) is no longer killed, so the
    open-order expectation falls back to the connector's *live* resting orders
    rather than staying empty. This is the post-kill/re-arm regression guard: a
    once-fired kill (or a routine kill drill) must NOT permanently zero the
    open-order expectation, which would make every later legitimately-resting
    order a false-positive breach for the life of the ledger."""
    connector = _MutableBalanceConnector(
        available=MoneyMicros(1),
        open_orders=(
            OpenOrder(
                id="rearmed-resting-1",
                ticker="KXFED-24DEC",
                side="yes",
                price=PricePips(5000),
                quantity=ContractCentis(10),
            ),
            OpenOrder(
                id="rearmed-resting-2",
                ticker="KXFED-24DEC",
                side="no",
                price=PricePips(4000),
                quantity=ContractCentis(20),
            ),
        ),
    )
    history = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=1_700_000_000
        ),
        CancelAllDirective(component="riskkernel", scope="all_open_orders"),
        KillReArmed(component="riskkernel", kill_sequence=1),
    ]

    source = LedgerExpectationSource(history, connector)

    assert source.get_expectations().expected_open_order_ids == frozenset(
        {"rearmed-resting-1", "rearmed-resting-2"}
    )


def test_cancel_all_directive_breaches_when_venue_still_shows_resting_orders() -> None:
    """Wired into a real `ReadOnlyVerifier`, an unrearmed kill history's
    zero-open-order expectation breaches against a venue that still reports a
    resting order: the balance and position dimensions are read straight off
    the same connector on both the expectation and the observation side (a
    tautological match, isolating the breach to the open-order dimension
    alone), yet `open_order_ok` is `False` and the outcome is `BREACH`.
    """
    connector = _ConnectorServingOpenOrders(
        inner=FakeExchange.from_fixture_dir(_fixture_path("clean")),
        positions=(
            Position(
                ticker="KXFED-24DEC",
                quantity=ContractCentis(500),
                average_price=PricePips(4550),
            ),
        ),
        open_orders=(
            OpenOrder(
                id="resting-1",
                ticker="KXFED-24DEC",
                side="yes",
                price=PricePips(5000),
                quantity=ContractCentis(10),
            ),
        ),
    )
    history = [
        KillEngaged(
            component="riskkernel", trigger="CLI", kill_sequence=1, epoch=1_700_000_000
        ),
        CancelAllDirective(component="riskkernel", scope="all_open_orders"),
    ]
    source = LedgerExpectationSource(history, connector)
    ledger_writer = InMemoryKernelLedgerWriter()
    sink = _RecordingSink()
    verifier = ReadOnlyVerifier(
        connector=connector,
        expectation_source=source,
        tolerances=VerificationTolerances(
            balance_tolerance=_ZERO_TOLERANCE_MICROS,
            position_tolerance=_ZERO_TOLERANCE_CENTIS,
        ),
        dispatcher=AlertDispatcher([sink], ledger_writer=LoggingLedgerWriter()),
        ledger_writer=ledger_writer,
    )

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.open_order_ok is False
    assert snapshot.outcome is VerificationOutcome.BREACH


def test_ledger_expectation_source_seeds_each_dimension_independently() -> None:
    """A history carrying only a `PositionsSnapshotRecorded` event (no
    verification event, no `CancelAllDirective`) seeds positions from the
    ledger while cash and open orders still fall back to the connector --
    proving each dimension's projection is independent, not all-or-nothing.
    """
    connector = _MutableBalanceConnector(
        available=MoneyMicros(77_000_000),
        positions=(
            Position(
                ticker="IGNORED-TICKER",
                quantity=ContractCentis(999),
                average_price=PricePips(1),
            ),
        ),
        open_orders=(
            OpenOrder(
                id="fallback-order",
                ticker="IGNORED-TICKER",
                side="yes",
                price=PricePips(1),
                quantity=ContractCentis(1),
            ),
        ),
    )
    history = [
        PositionsSnapshotRecorded(
            component="riskkernel",
            positions=[
                {
                    "ticker": "KXFED-24DEC",
                    "quantity_centis": 250,
                    "average_price_pips": 4500,
                }
            ],
        ),
    ]

    source = LedgerExpectationSource(history, connector)
    expectations = source.get_expectations()

    assert dict(expectations.expected_positions) == {"KXFED-24DEC": ContractCentis(250)}
    assert expectations.expected_available_cash == MoneyMicros(77_000_000)
    assert expectations.expected_open_order_ids == frozenset({"fallback-order"})


# --- Component-scoped ledger seeds (issue #288 review follow-up) ------------------
#
# The cash and position seeds must match on `event.component == "riskkernel"`,
# not on `event.event_type` alone. The PAPER scheduler loop records
# `PositionsSnapshotRecorded(component="scheduler", positions=...)` derived from
# a *simulated* `PaperExchange` (windbreak/scheduler/loop.py), and the documented
# deploy topology shares ONE ledger volume across processes -- so an event-type-only
# fold lets a LIVE risk kernel seed its safety-critical reconciliation baseline
# from simulated paper positions. The consequences are symmetric and both bad: a
# fake baseline that does not match the real venue is a full-size position breach
# (a spurious `AUTO_RECONCILIATION` auto-kill), and a fake baseline that happens
# to match the venue masks genuine drift the kernel exists to catch.
#
# The open-order dimension is deliberately NOT component-scoped; the last test
# below pins that asymmetry so a future "scope every dimension" cleanup cannot
# silently break the kill fold's single source of truth.


def test_ledger_expectation_source_positions_ignore_a_later_foreign_snapshot() -> None:
    """A `PositionsSnapshotRecorded` stamped with a foreign component never
    seeds the kernel's position baseline -- not even when it is the LAST
    snapshot in the history, where an event-type-only fold would let it
    overwrite the kernel's own. The kernel-stamped rows recorded before it are
    what survive, and the simulated paper ticker never appears at all."""
    connector = _MutableBalanceConnector(available=MoneyMicros(1))
    history = [
        PositionsSnapshotRecorded(
            component="riskkernel",
            positions=[
                {
                    "ticker": "KXFED-24DEC",
                    "quantity_centis": 250,
                    "average_price_pips": 4500,
                }
            ],
        ),
        PositionsSnapshotRecorded(
            component="scheduler",
            positions=[
                {
                    "ticker": "PAPER-SIM",
                    "quantity_centis": 12345,
                    "average_price_pips": 5000,
                }
            ],
        ),
    ]

    source = LedgerExpectationSource(history, connector)
    expectations = source.get_expectations()

    assert dict(expectations.expected_positions) == {"KXFED-24DEC": ContractCentis(250)}
    assert "PAPER-SIM" not in expectations.expected_positions


def test_ledger_expectation_source_positions_fall_back_when_only_foreign() -> None:
    """When the only `PositionsSnapshotRecorded` in history is foreign-stamped,
    the kernel's own ledger carries no snapshot at all, so the position
    baseline takes the same connector fallback an empty history takes: the
    connector's live `ContractCentis(500)` in `KXFED-24DEC`.

    That exact expected value is load-bearing -- it must be the connector's
    positions and NOT `{}`. An implementation that filters the foreign rows out
    of the fold but leaves its internal `rows` sentinel non-`None` would fall
    through to an empty snapshot instead of to the connector, expect a flat
    account, and turn every genuinely-held position into a full-size position
    breach on the very next verification cycle."""
    connector = _MutableBalanceConnector(
        available=MoneyMicros(1),
        positions=(
            Position(
                ticker="KXFED-24DEC",
                quantity=ContractCentis(500),
                average_price=PricePips(4550),
            ),
        ),
    )
    history = [
        PositionsSnapshotRecorded(
            component="scheduler",
            positions=[
                {
                    "ticker": "PAPER-SIM",
                    "quantity_centis": 12345,
                    "average_price_pips": 5000,
                }
            ],
        ),
    ]

    source = LedgerExpectationSource(history, connector)
    expectations = source.get_expectations()

    assert dict(expectations.expected_positions) == {"KXFED-24DEC": ContractCentis(500)}
    assert "PAPER-SIM" not in expectations.expected_positions


def test_ledger_expectation_source_positions_own_flat_snapshot_beats_foreign() -> None:
    """The kernel's own snapshot of a FLAT account (an empty row list) beats
    both a later foreign snapshot and the connector's live position.

    Two properties in one: today's event-type-only fold takes the later
    foreign rows, and once the seed is component-scoped this becomes the guard
    that the filter did not also turn an own *empty* snapshot into a *missing*
    one -- a flat snapshot is a real recorded fact, and must keep beating the
    connector rather than falling through to it."""
    connector = _MutableBalanceConnector(
        available=MoneyMicros(1),
        positions=(
            Position(
                ticker="KXFED-24DEC",
                quantity=ContractCentis(500),
                average_price=PricePips(4550),
            ),
        ),
    )
    history = [
        PositionsSnapshotRecorded(component="riskkernel", positions=[]),
        PositionsSnapshotRecorded(
            component="scheduler",
            positions=[
                {
                    "ticker": "PAPER-SIM",
                    "quantity_centis": 12345,
                    "average_price_pips": 5000,
                }
            ],
        ),
    ]

    source = LedgerExpectationSource(history, connector)
    expectations = source.get_expectations()

    assert dict(expectations.expected_positions) == {}


def test_ledger_expectation_source_cash_ignores_a_foreign_component_event() -> None:
    """A verification event stamped with a foreign component never seeds the
    cash baseline, even when it is the LAST non-breach verification event in
    the shared history: the kernel's own earlier `VerificationPassed` cash
    wins. Another process's `"VerificationPassed"` describes a different
    account entirely, so folding it would re-baseline a live kernel onto cash
    it has never held."""
    connector = _MutableBalanceConnector(available=MoneyMicros(1))
    history = [
        _verification_event("VerificationPassed", cash_micros=10_000_000),
        _verification_event(
            "VerificationPassed", cash_micros=999_000_000, component="scheduler"
        ),
    ]

    source = LedgerExpectationSource(history, connector)

    assert source.get_expectations().expected_available_cash == MoneyMicros(10_000_000)


def test_ledger_expectation_source_cash_falls_back_when_only_foreign() -> None:
    """When the only non-breach verification event in history is
    foreign-stamped, the kernel's own ledger carries no cash observation, so
    the cash baseline takes the connector fallback -- the connector's live
    `MoneyMicros(77_000_000)`, never the foreign event's figure."""
    connector = _MutableBalanceConnector(available=MoneyMicros(77_000_000))
    history = [
        _verification_event(
            "VerificationPassed", cash_micros=999_000_000, component="scheduler"
        ),
    ]

    source = LedgerExpectationSource(history, connector)

    assert source.get_expectations().expected_available_cash == MoneyMicros(77_000_000)


def test_ledger_expectation_source_open_orders_still_fold_a_foreign_kill() -> None:
    """A `KillEngaged` (plus its `CancelAllDirective`) stamped with a foreign
    component STILL empties the open-order expectation, even though the
    connector reports a resting order. The open-order dimension is deliberately
    NOT component-scoped, for two reasons.

    First, correctness of the fold: `_seed_open_order_ids` delegates to
    `kill_state_in`, the one canonical kill fold that `KillSwitch.from_events`
    and `RiskKernel.from_events` also run -- over the same unfiltered history --
    in `windbreak/main.py`. Component-scoping it at this single call site would
    let the open-order expectation disagree with the kill mode the kernel
    actually replays to.

    Second, direction of failure: a foreign kill event only makes the
    expectation MORE conservative (expect nothing resting), so the worst case
    is a breach on a live resting order, never a silently-missed one. It fails
    closed, which is the opposite of the cash and position seeds, where a
    foreign event can fabricate a permissive-looking baseline."""
    connector = _MutableBalanceConnector(
        available=MoneyMicros(1),
        open_orders=(
            OpenOrder(
                id="resting-1",
                ticker="KXFED-24DEC",
                side="yes",
                price=PricePips(5000),
                quantity=ContractCentis(10),
            ),
        ),
    )
    history = [
        KillEngaged(
            component="scheduler", trigger="CLI", kill_sequence=1, epoch=1_700_000_000
        ),
        CancelAllDirective(component="scheduler", scope="all_open_orders"),
    ]

    source = LedgerExpectationSource(history, connector)

    assert source.get_expectations().expected_open_order_ids == frozenset()


# --- Defensive-path coverage (issue #309) ----------------------------------------
#
# Three paths PR #308's Gate 2.5 self-review flagged as defensive-only and
# deferred: the cash seed's breach-only fallback, `_rows_to_positions`'s
# malformed-row narrowing, and the `isinstance(x, int)` checks that also accept
# `True`/`False`. They are covered here because the pre-v1.0.0 mutation gate
# (`./scripts/mutation.sh`) surfaces untested defensive branches as surviving
# mutants.
#
# The narrowing is not merely theoretical. Replay rebuilds *base* `Event`s from
# stored envelope JSON via `windbreak.ledger.store.events_from_records`, so
# `PositionsSnapshotRecorded`'s `list[dict[str, object]]` typing does not survive
# the round-trip -- on the read path `payload["positions"]` is whatever JSON a
# corrupt, older, or foreign writer left behind.


def _replayed_positions_snapshot(rows: object) -> Event:
    """Build a bare replayed positions snapshot carrying ``rows`` verbatim.

    The typed :class:`PositionsSnapshotRecorded` cannot express a malformed
    payload -- its ``positions`` field is ``list[dict[str, object]]`` -- but the
    replay path never constructs the typed class. This mirrors what
    ``events_from_records`` actually hands the projection, so the narrowing is
    tested against the shapes it really defends against.

    Args:
        rows: The ``positions`` payload value, well-formed or not.

    Returns:
        The constructed bare `Event`.
    """
    return Event(
        event_type="PositionsSnapshotRecorded",
        component="riskkernel",
        payload_schema_version=1,
        payload={"positions": rows},
    )


def test_ledger_expectation_source_cash_falls_back_when_only_breaches() -> None:
    """A history of *only* breach events leaves the cash seed unset, so the
    baseline takes the connector fallback.

    Distinct from the existing breach-ignored test, which pairs a breach with a
    non-breach event and asserts the non-breach cash wins: here there is no
    non-breach event at all, so the `seed is None` fallback is the branch under
    test. Re-baselining onto a cash figure already known to breach is exactly
    what this path exists to prevent.
    """
    connector = _MutableBalanceConnector(available=MoneyMicros(88_000_000))
    history = [
        _verification_event("VerificationMismatch", cash_micros=5_000_000),
        _verification_event("VerificationMismatch", cash_micros=6_000_000),
    ]

    source = LedgerExpectationSource(history, connector)

    assert source.get_expectations().expected_available_cash == MoneyMicros(88_000_000)


@pytest.mark.parametrize(
    "malformed",
    [250, "KXFED-24DEC", {"KXFED-24DEC": 250}],
    ids=["non-iterable-int", "bare-string", "mapping"],
)
def test_ledger_expectation_source_positions_narrow_a_non_list_payload(
    malformed: object,
) -> None:
    """A ``positions`` payload that is not a list narrows to a flat expectation.

    Note this does *not* fall back to the connector: the snapshot exists, so
    the internal rows sentinel is non-`None` and the narrowing returns an empty
    mapping. A corrupt snapshot therefore makes the kernel expect a flat
    account, which breaches loudly against any real holding rather than
    silently adopting a garbage baseline -- the fail-closed direction. The
    connector below deliberately holds a position so a wrong fallback would
    show up as that position leaking into the expectation.

    The `non-iterable-int` case is the one with real teeth, and is why this is
    parametrized rather than a single mapping payload: the string and mapping
    payloads still narrow to an empty mapping even with the list guard deleted
    (iterating them yields characters and keys, each skipped as a non-row), so
    they alone cannot detect that guard's removal. Iterating an `int` raises,
    so only that case actually fails if the guard is lost.
    """
    connector = _MutableBalanceConnector(
        available=MoneyMicros(1),
        positions=(
            Position(
                ticker="KXFED-24DEC",
                quantity=ContractCentis(500),
                average_price=PricePips(4550),
            ),
        ),
    )

    source = LedgerExpectationSource(
        [_replayed_positions_snapshot(malformed)], connector
    )

    assert dict(source.get_expectations().expected_positions) == {}


def test_ledger_expectation_source_positions_fall_back_on_a_null_payload() -> None:
    """A snapshot whose ``positions`` payload is absent -- or explicitly null --
    takes the *connector* fallback, not the empty narrowing.

    This is the boundary of the test above, and the two must not be conflated.
    A missing key and a JSON ``null`` both surface as `None` from
    `payload.get`, which is indistinguishable from "this snapshot recorded no
    positions fact at all" -- so the seed is left unset and the connector's live
    positions win. A payload that is present but the wrong *type* is a
    different thing: it is a recorded fact that cannot be read, and narrows to
    a flat expectation instead.
    """
    connector = _MutableBalanceConnector(
        available=MoneyMicros(1),
        positions=(
            Position(
                ticker="KXFED-24DEC",
                quantity=ContractCentis(500),
                average_price=PricePips(4550),
            ),
        ),
    )

    source = LedgerExpectationSource([_replayed_positions_snapshot(None)], connector)

    assert dict(source.get_expectations().expected_positions) == {
        "KXFED-24DEC": ContractCentis(500)
    }


def test_ledger_expectation_source_positions_skip_malformed_rows() -> None:
    """Individually malformed rows are skipped while well-formed rows around
    them still seed the baseline.

    Each row below trips a distinct guard in the narrowing -- a row that is not
    a mapping at all, a row whose ticker is not a string, and a row whose
    quantity is not an int -- and the one good row must survive all three.
    """
    connector = _MutableBalanceConnector(available=MoneyMicros(1))
    rows = [
        "not-a-row",
        {"ticker": 12345, "quantity_centis": 10},
        {"ticker": "KXBAD", "quantity_centis": "250"},
        {"ticker": "KXFED-24DEC", "quantity_centis": 250},
    ]

    source = LedgerExpectationSource([_replayed_positions_snapshot(rows)], connector)

    assert dict(source.get_expectations().expected_positions) == {
        "KXFED-24DEC": ContractCentis(250)
    }


def test_ledger_expectation_source_cash_rejects_a_bool_payload() -> None:
    """A boolean ``exchange_verified_available_cash`` never seeds the baseline.

    ``bool`` is a subclass of ``int``, so a bare ``isinstance(cash, int)``
    accepts ``True`` and seeds ``MoneyMicros(1)`` -- one micro-dollar -- from a
    payload carrying no cash figure at all. Essentially any real balance
    breaches against that fabricated baseline, so the value must be rejected
    and the connector fallback must fire instead.
    """
    connector = _MutableBalanceConnector(available=MoneyMicros(42_000_000))
    history = [
        Event(
            event_type="VerificationPassed",
            component="riskkernel",
            payload_schema_version=1,
            payload={"exchange_verified_available_cash": True},
        )
    ]

    source = LedgerExpectationSource(history, connector)

    assert source.get_expectations().expected_available_cash == MoneyMicros(42_000_000)


def test_ledger_expectation_source_positions_reject_a_bool_quantity() -> None:
    """A boolean ``quantity_centis`` never becomes a position.

    The same ``bool``-is-an-``int`` hole on the position path: unnarrowed,
    ``True`` smuggles ``ContractCentis(1)`` onto the scaled-int path for a
    ticker whose real size is unknown. The row must be skipped like any other
    malformed row, leaving only the well-formed one.
    """
    connector = _MutableBalanceConnector(available=MoneyMicros(1))
    rows = [
        {"ticker": "KXBOOL", "quantity_centis": True},
        {"ticker": "KXFED-24DEC", "quantity_centis": 250},
    ]

    source = LedgerExpectationSource([_replayed_positions_snapshot(rows)], connector)

    assert dict(source.get_expectations().expected_positions) == {
        "KXFED-24DEC": ContractCentis(250)
    }


# --- Issue #352: PAPER verification is falsifiable, not tautological -------------
#
# `LedgerExpectationSource` seeds every dimension from `component ==
# "riskkernel"` events, and the PAPER loop stamps `"scheduler"` on everything it
# writes, so a `LedgerExpectationSource(history, paper_exchange)` falls through
# to the connector on all three dimensions. That is only sound if the connector's
# *observation* is derived independently of the frozen *expectation*. Before
# issue #352 it was not: `PaperExchange.get_positions()` was hardcoded to `()`
# and `get_balances()` returned a static fixture, so expectation and observation
# were the same constant and `balance_ok` / `position_ok` could not fail for any
# reason -- three SPEC S10.3 checks passing on a comparison with no failure mode.
#
# The pair of assertions below is the whole point of the issue: the identical
# comparison must be able to land CLEAN *and* land BREACH, driven only by whether
# the exchange actually traded.

#: `tests/fixtures/books/<scenario>` -- the `PaperExchange` book-replay fixtures
#: shared with `tests/connector/test_paper_exchange.py`.
_BOOKS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "books"


def _paper_verifier(
    exchange: PaperExchange,
) -> tuple[ReadOnlyVerifier, InMemoryKernelLedgerWriter]:
    """Build a zero-tolerance verifier reconciling ``exchange`` against itself.

    The expectation source is constructed *now*, freezing the exchange's
    current state as the ledger baseline, exactly as a kernel startup would.

    Args:
        exchange: The paper exchange serving both the frozen expectation (at
            construction) and every later observation.

    Returns:
        The `(verifier, kernel_ledger_writer)` pair.
    """
    ledger_writer = InMemoryKernelLedgerWriter()
    verifier = ReadOnlyVerifier(
        connector=exchange,
        expectation_source=LedgerExpectationSource([], exchange),
        tolerances=VerificationTolerances(
            balance_tolerance=_ZERO_TOLERANCE_MICROS,
            position_tolerance=_ZERO_TOLERANCE_CENTIS,
        ),
        dispatcher=AlertDispatcher(
            [_RecordingSink()], ledger_writer=LoggingLedgerWriter()
        ),
        ledger_writer=ledger_writer,
    )
    return verifier, ledger_writer


def test_paper_verification_passes_while_the_exchange_has_not_traded() -> None:
    """An untraded paper exchange still matches the baseline frozen off it."""
    exchange = PaperExchange.from_fixture_dir(_BOOKS_DIR / "deep_walk")
    verifier, _ = _paper_verifier(exchange)

    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.balance_ok is True
    assert snapshot.position_ok is True
    assert snapshot.open_order_ok is True
    assert snapshot.outcome is VerificationOutcome.CLEAN


def test_paper_verification_breaches_once_the_exchange_actually_trades() -> None:
    """The same comparison BREACHES after a real fill -- so it is falsifiable.

    The order is sized to be fully absorbed (`deep_walk` step 1 offers a single
    4750 ask of 500 centis; a 100-centi buy is under both the requested size and
    the 25% participation cap), so *no* order rests and `open_order_ok` stays
    `True`. The breach is therefore attributable purely to the two dimensions
    that used to be tautologies: cash moved by 475_000 book cost + 20_000 fee,
    and a 100-centi MKT-DEEP position appeared where the frozen baseline
    expected none.
    """
    exchange = PaperExchange.from_fixture_dir(_BOOKS_DIR / "deep_walk")
    exchange.advance()
    verifier, _ = _paper_verifier(exchange)

    exchange.place_order(
        PaperOrderIntent("MKT-DEEP", "yes", PricePips(4750), ContractCentis(100)),
        approval_token=object(),
    )
    snapshot = verifier.run_cycle(now_epoch_s=DEFAULT_NOW_EPOCH_S)

    assert snapshot.open_order_ok is True
    assert snapshot.balance_ok is False
    assert snapshot.position_ok is False
    assert snapshot.outcome is VerificationOutcome.BREACH
    assert snapshot.cash_drift == MoneyMicros(495_000)
