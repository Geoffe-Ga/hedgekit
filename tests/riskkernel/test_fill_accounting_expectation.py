"""Failing-first tests for ledgered fill accounting (issue #365, RED).

`LedgerExpectationSource` freezes its baseline at construction, which is what
keeps the three reconciliation dimensions falsifiable rather than tautological
(issue #352: an expectation re-read off the connector each cycle is compared
against that same connector and can never fail). The documented consequence was
that the first real fill moved the venue off that frozen baseline permanently:
the next cycle graded ``BREACH``, the kernel HALTed, and only a process restart
cleared it -- see ``windbreak/scheduler/loop.py::_build_verifier``.

Issue #365 closes that with *ledgered fill accounting*. Each fill is booked
exactly once, at execution, into the hash-chained ledger as a signed cash and
position delta (a ``FillAccounted`` event), and the expectation advances by
folding those booked entries through a narrow ``FillAccountingFeed`` seam. The
booked entry is frozen at execution; the observation remains the venue's live
*aggregate* balance and position. The two are therefore still independent and
can still disagree -- which is exactly what
``test_a_booked_fill_never_reads_the_expectation_off_the_connector`` pins.

Fail closed: an entry whose accounting cannot be reconstructed is never skipped
past. The expectation stops advancing and latches, so the unexplained gap still
surfaces as divergence and still halts.

Neither ``FillAccounted`` nor the ``fill_accounting`` seam exists yet, so every
test below fails -- the expected Gate 1 RED state for issue #365.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from windbreak.alerts.dispatch import AlertDispatcher, LoggingLedgerWriter
from windbreak.connector.models import (
    BalanceSemantics,
    BalanceSnapshot,
    NormalizedMarket,
    OpenOrder,
    Position,
)
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
from windbreak.ledger.events import ConfigLoaded, Event, FillAccounted
from windbreak.numeric.types import ContractCentis, MoneyMicros, PricePips
from windbreak.riskkernel.process import InMemoryKernelLedgerWriter
from windbreak.riskkernel.verification import (
    LedgerExpectationSource,
    ReadOnlyVerifier,
    VerificationOutcome,
    VerificationTolerances,
)

if TYPE_CHECKING:
    from windbreak.alerts.registry import AlertSeverity, AlertType

#: The epoch second every cycle below is stamped at; its value is irrelevant to
#: every assertion, it only has to be deterministic.
_NOW_EPOCH_S = 1_700_000_000

#: A fixed UTC instant for every `BalanceSnapshot.fetched_at` reported below.
_FIXED_DATETIME = datetime(2024, 1, 1, tzinfo=UTC)

#: The market every booked fill below trades.
_TICKER = "KXFED-24DEC"

#: A `BalanceSemantics` with every field a known (non-`UNKNOWN`) member; no
#: assertion here inspects it, but `run_cycle` reads it on every pass.
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
class _MutableVenue:
    """A minimal, mutable `ReadOnlyVenueView` stub.

    Mutability is the point: a test builds the expectation source over this
    venue, then moves the venue, and asserts on whether the expectation
    followed. It must follow only what the *ledger* booked.

    Attributes:
        available: The account's current available cash.
        positions: The account's current positions.
        open_orders: The account's current resting orders.
    """

    available: MoneyMicros
    positions: tuple[Position, ...] = ()
    open_orders: tuple[OpenOrder, ...] = ()

    def get_balances(self) -> BalanceSnapshot:
        """Return the venue's current, possibly-since-mutated available cash."""
        return BalanceSnapshot(
            total=self.available, available=self.available, fetched_at=_FIXED_DATETIME
        )

    def get_positions(self) -> tuple[Position, ...]:
        """Return the venue's current positions."""
        return self.positions

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Return the venue's current resting orders."""
        return self.open_orders

    def get_balance_semantics(self) -> BalanceSemantics:
        """Return a fully-known `BalanceSemantics`."""
        return _FULLY_KNOWN_SEMANTICS

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Return an eligible market, so no jurisdiction alert ever fires.

        Args:
            ticker: The market to describe.

        Returns:
            An `"eligible"` `NormalizedMarket` for `ticker`; the jurisdiction
            dimension is out of scope for every test in this module.
        """
        return NormalizedMarket(
            exchange="paper",
            ticker=ticker,
            event_ticker="KXFED",
            title="Will the Fed cut?",
            resolution_criteria="Per the FOMC statement.",
            category="economics",
            close_time=_FIXED_DATETIME,
            expected_resolution_time=None,
            market_type="fully_collateralized_binary",
            price_tick_pips=1,
            min_order_contract_centis=1,
            fractional_trading_enabled=False,
            mutually_exclusive_group_id=None,
            jurisdiction_status="eligible",
            raw_exchange_payload_hash="0" * 64,
            volume_24h_micros=1_000_000,
        )


@dataclass
class _StubFeed:
    """A `FillAccountingFeed` stub handing out scripted batches, once each.

    Attributes:
        batches: The batches `drain` returns, oldest first. Each call pops the
            next one and every later call returns `()`, mirroring the real
            ledger-backed feed's drain-once cursor.
    """

    batches: list[tuple[Event, ...]] = field(default_factory=list)

    def drain(self) -> tuple[Event, ...]:
        """Return the next scripted batch, or `()` once exhausted."""
        if not self.batches:
            return ()
        return self.batches.pop(0)


@dataclass
class _RecordingSink:
    """A fake `AlertSink` recording every call without raising."""

    name: str = "recording"
    calls: list[tuple[AlertType, AlertSeverity, str]] = field(default_factory=list)

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Record the call."""
        self.calls.append((alert_type, severity, message))


def _booked_fill(
    *,
    fill_id: str = "paper-fill-1",
    ticker: str = _TICKER,
    cash_delta_micros: int = -5_000_000,
    position_delta_centis: int = 100,
) -> FillAccounted:
    """Build one `FillAccounted` entry booking a fill's two signed deltas.

    Args:
        fill_id: The venue's own identifier for the booked fill.
        ticker: The market that traded.
        cash_delta_micros: The signed available-cash movement, in micros.
        position_delta_centis: The signed YES-frame position movement, in
            contract-centis.

    Returns:
        The constructed `FillAccounted` event.
    """
    return FillAccounted(
        component="scheduler",
        fill_id=fill_id,
        ticker=ticker,
        cash_delta_micros=cash_delta_micros,
        position_delta_centis=position_delta_centis,
    )


def _run_cycle(
    venue: _MutableVenue, source: LedgerExpectationSource
) -> tuple[VerificationOutcome, InMemoryKernelLedgerWriter]:
    """Run one verification cycle of ``venue`` against ``source``.

    Args:
        venue: The read-only venue view the cycle observes.
        source: The expectation source the cycle diffs against.

    Returns:
        The cycle's graded outcome and the ledger writer it recorded through.
    """
    writer = InMemoryKernelLedgerWriter()
    verifier = ReadOnlyVerifier(
        connector=venue,
        expectation_source=source,
        tolerances=VerificationTolerances(
            balance_tolerance=MoneyMicros(0),
            position_tolerance=ContractCentis(0),
        ),
        dispatcher=AlertDispatcher(
            [_RecordingSink()], ledger_writer=LoggingLedgerWriter()
        ),
        ledger_writer=writer,
    )
    return verifier.run_cycle(_NOW_EPOCH_S).outcome, writer


def _held(centis: int) -> tuple[Position, ...]:
    """Return a one-row position tuple holding ``centis`` of `_TICKER`.

    Args:
        centis: The signed YES-frame holding, in contract-centis.

    Returns:
        The single-row position tuple.
    """
    return (
        Position(
            ticker=_TICKER,
            quantity=ContractCentis(centis),
            average_price=PricePips(5000),
        ),
    )


def test_booked_fill_advances_the_expectation_off_the_frozen_baseline() -> None:
    """A booked fill moves both ledger-derived dimensions by its own deltas.

    This is the whole of issue #365: without it the baseline is frozen at
    startup and the first fill breaches forever.
    """
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(batches=[(_booked_fill(),)])

    source = LedgerExpectationSource([], venue, fill_accounting=feed)
    expectations = source.get_expectations()

    assert expectations.expected_available_cash == MoneyMicros(95_000_000)
    assert dict(expectations.expected_positions) == {_TICKER: ContractCentis(100)}


def test_booked_fills_accumulate_across_successive_drains() -> None:
    """Two fills drained on two separate cycles both land, and compound."""
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(
        batches=[
            (_booked_fill(fill_id="paper-fill-1"),),
            (
                _booked_fill(
                    fill_id="paper-fill-2",
                    cash_delta_micros=-1_000_000,
                    position_delta_centis=25,
                ),
            ),
        ]
    )
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    source.get_expectations()
    expectations = source.get_expectations()

    assert expectations.expected_available_cash == MoneyMicros(94_000_000)
    assert dict(expectations.expected_positions) == {_TICKER: ContractCentis(125)}


def test_a_short_side_fill_advances_the_position_negatively() -> None:
    """Positions are booked in the YES frame, so a NO-side fill's delta is
    negative and nets against an existing long rather than adding to it."""
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(
        batches=[
            (
                _booked_fill(fill_id="paper-fill-1", position_delta_centis=100),
                _booked_fill(
                    fill_id="paper-fill-2",
                    cash_delta_micros=-2_000_000,
                    position_delta_centis=-40,
                ),
            )
        ]
    )
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    expectations = source.get_expectations()

    assert dict(expectations.expected_positions) == {_TICKER: ContractCentis(60)}
    assert expectations.expected_available_cash == MoneyMicros(93_000_000)


def test_a_booked_fill_never_reads_the_expectation_off_the_connector() -> None:
    """The expectation advances by the *ledgered* delta, never the venue's own
    report.

    The venue is moved to a cash figure the booked entry does not explain. Were
    the expectation re-derived from the venue the two would agree by
    construction and the cycle could never fail -- the issue #352 tautology.
    Because it advances only from ledgered evidence, the disagreement survives
    and the cycle grades BREACH.
    """
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(batches=[(_booked_fill(),)])
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    venue.available = MoneyMicros(80_000_000)
    venue.positions = _held(100)
    outcome, writer = _run_cycle(venue, source)

    assert outcome is VerificationOutcome.BREACH
    assert [event.event_type for event in writer.events] == ["VerificationMismatch"]


def test_a_venue_matching_the_booked_fill_grades_clean() -> None:
    """The complement of the disagreement test: a venue that moved exactly as
    the ledger booked reconciles clean, so a filling PAPER loop keeps ticking
    instead of halting on its first fill."""
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(batches=[(_booked_fill(),)])
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    venue.available = MoneyMicros(95_000_000)
    venue.positions = _held(100)
    outcome, writer = _run_cycle(venue, source)

    assert outcome is VerificationOutcome.CLEAN
    assert [event.event_type for event in writer.events] == ["VerificationPassed"]


def test_a_position_only_divergence_still_breaches_after_a_booked_fill() -> None:
    """Advancing cash correctly must not mask a position that moved further
    than the ledger can explain."""
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(batches=[(_booked_fill(),)])
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    venue.available = MoneyMicros(95_000_000)
    venue.positions = _held(300)
    outcome, _ = _run_cycle(venue, source)

    assert outcome is VerificationOutcome.BREACH


def test_an_unaccountable_booked_entry_refuses_to_advance_the_expectation() -> None:
    """An entry whose accounting cannot be reconstructed is never skipped past:
    the expectation stops advancing, so the unexplained gap still shows up as
    divergence and still halts (fail closed)."""
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    malformed = Event(
        event_type="FillAccounted",
        component="scheduler",
        payload_schema_version=1,
        payload={"fill_id": "paper-fill-1", "ticker": _TICKER},
    )
    feed = _StubFeed(batches=[(malformed,)])

    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    assert source.get_expectations().expected_available_cash == MoneyMicros(100_000_000)


def test_an_unaccountable_entry_latches_and_blocks_every_later_fill() -> None:
    """Once a gap is seen the expectation never advances again, even for
    well-formed entries drained afterwards: silently resuming past an
    unexplained gap would re-baseline onto books that cannot explain the
    venue."""
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    malformed = Event(
        event_type="FillAccounted",
        component="scheduler",
        payload_schema_version=1,
        payload={
            "fill_id": "paper-fill-1",
            "ticker": _TICKER,
            "cash_delta_micros": True,
            "position_delta_centis": 5,
        },
    )
    feed = _StubFeed(batches=[(malformed,), (_booked_fill(),)])
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    source.get_expectations()

    assert source.get_expectations().expected_available_cash == MoneyMicros(100_000_000)


def test_a_gap_mid_batch_keeps_the_fills_that_preceded_it() -> None:
    """Entries booked before the gap are real facts and still apply; only the
    advance *past* the gap is refused."""
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    malformed = Event(
        event_type="FillAccounted",
        component="scheduler",
        payload_schema_version=1,
        payload={"fill_id": "paper-fill-2", "ticker": _TICKER},
    )
    feed = _StubFeed(
        batches=[
            (
                _booked_fill(fill_id="paper-fill-1"),
                malformed,
                _booked_fill(fill_id="paper-fill-3", cash_delta_micros=-9_000_000),
            )
        ]
    )
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    assert source.get_expectations().expected_available_cash == MoneyMicros(95_000_000)


def test_a_foreign_event_type_on_the_feed_is_treated_as_a_gap() -> None:
    """The seam's contract is "this account's booked fill accounting". An event
    of any other type is an unreconstructable entry, not something to skip."""
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(
        batches=[
            (ConfigLoaded(component="scheduler", config_hash="abc", diff={}),),
            (_booked_fill(),),
        ]
    )
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    source.get_expectations()

    assert source.get_expectations().expected_available_cash == MoneyMicros(100_000_000)


def test_a_non_string_ticker_is_a_gap() -> None:
    """A ticker that is not a string cannot key a position, so the entry is
    unreconstructable rather than silently coerced."""
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    malformed = Event(
        event_type="FillAccounted",
        component="scheduler",
        payload_schema_version=1,
        payload={
            "fill_id": "paper-fill-1",
            "ticker": 7,
            "cash_delta_micros": -1,
            "position_delta_centis": 1,
        },
    )
    feed = _StubFeed(batches=[(malformed,)])
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    assert source.get_expectations().expected_available_cash == MoneyMicros(100_000_000)


def test_without_a_feed_the_expectation_stays_frozen_exactly_as_before() -> None:
    """Omitting the seam preserves the pre-#365 freeze guarantee verbatim, so a
    composition root that books no fills is completely unchanged."""
    venue = _MutableVenue(available=MoneyMicros(1_000_000))
    source = LedgerExpectationSource([], venue)

    venue.available = MoneyMicros(2_000_000)

    assert source.get_expectations() is source.get_expectations()
    assert source.get_expectations().expected_available_cash == MoneyMicros(1_000_000)


def test_an_empty_drain_leaves_the_expectation_object_identical() -> None:
    """A cycle with nothing new to fold must not rebuild the expectation."""
    venue = _MutableVenue(available=MoneyMicros(1_000_000))
    source = LedgerExpectationSource([], venue, fill_accounting=_StubFeed())

    assert source.get_expectations() is source.get_expectations()


def test_the_open_order_dimension_is_untouched_by_fill_accounting() -> None:
    """Fill accounting books cash and positions only. Venue order ids are never
    ledgered, so the open-order expectation stays exactly what the baseline
    captured -- a fill that retires a resting order still diverges, and still
    halts."""
    resting = OpenOrder(
        id="paper-order-1",
        ticker=_TICKER,
        side="yes",
        price=PricePips(5000),
        quantity=ContractCentis(100),
    )
    venue = _MutableVenue(available=MoneyMicros(100_000_000), open_orders=(resting,))
    feed = _StubFeed(batches=[(_booked_fill(),)])
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    expectations = source.get_expectations()

    assert expectations.expected_open_order_ids == frozenset({"paper-order-1"})
