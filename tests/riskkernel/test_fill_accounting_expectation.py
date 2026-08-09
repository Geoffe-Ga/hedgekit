"""Tests for the ledgered advance of the reconciliation baseline (#365, #390).

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

Issue #390 extends the same machinery to the third dimension. A ``FillAccounted``
now names the resting order it executed against, and a ``RestingOrderAccounted``
books an order's arrival on the venue's resting book, so the expectation can say
"this order rests because I placed it" and "this order is gone because a fill I
booked exhausted it". Before that, venue order ids were never ledgered at all:
an outright fill reconciled, but a partially filled order's surviving remainder
read as unexplained venue movement and halted the loop.

The advance is still never a relaxation. The open-order set moves only by booked
arrivals and booked fills, never by re-reading ``get_open_orders`` -- which is
the view the cycle compares against, and mirroring it would make that dimension
structurally incapable of failing. Both directions are pinned below: an order
the venue rests that nobody booked still breaches, and an order that vanishes
with no booked fill behind it still breaches.
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
from windbreak.ledger.events import (
    ConfigLoaded,
    Event,
    FillAccounted,
    RestingOrderAccounted,
)
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
    venue_order_id: str = "",
) -> FillAccounted:
    """Build one `FillAccounted` entry booking a fill's signed deltas.

    Args:
        fill_id: The venue's own identifier for the booked fill.
        ticker: The market that traded.
        cash_delta_micros: The signed available-cash movement, in micros.
        position_delta_centis: The signed YES-frame position movement, in
            contract-centis.
        venue_order_id: The resting order this fill executed against; `""` --
            the default -- for an outright taker execution that never rested.

    Returns:
        The constructed `FillAccounted` event.
    """
    return FillAccounted(
        component="scheduler",
        fill_id=fill_id,
        ticker=ticker,
        cash_delta_micros=cash_delta_micros,
        position_delta_centis=position_delta_centis,
        venue_order_id=venue_order_id,
    )


def _booked_resting_order(
    *,
    venue_order_id: str = "paper-order-1",
    ticker: str = _TICKER,
    resting_quantity_centis: int = 200,
) -> RestingOrderAccounted:
    """Build one `RestingOrderAccounted` entry booking an order's arrival.

    Args:
        venue_order_id: The venue's identifier for the order now resting.
        ticker: The market the order rests in.
        resting_quantity_centis: The quantity that came to rest, in
            contract-centis.

    Returns:
        The constructed `RestingOrderAccounted` event.
    """
    return RestingOrderAccounted(
        component="scheduler",
        venue_order_id=venue_order_id,
        ticker=ticker,
        resting_quantity_centis=resting_quantity_centis,
    )


def _resting(order_id: str, centis: int) -> OpenOrder:
    """Return one venue-reported resting order of `centis` on `_TICKER`.

    Args:
        order_id: The venue's identifier for the order.
        centis: The resting quantity, in contract-centis.

    Returns:
        The constructed `OpenOrder`.
    """
    return OpenOrder(
        id=order_id,
        ticker=_TICKER,
        side="yes",
        price=PricePips(5000),
        quantity=ContractCentis(centis),
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


def test_a_fill_naming_no_resting_order_leaves_the_open_order_set_alone() -> None:
    """An outright taker execution rests nothing and retires nothing.

    Such a fill carries no ``venue_order_id``, so it moves cash and positions
    and leaves the open-order dimension exactly where the baseline captured it.
    """
    venue = _MutableVenue(
        available=MoneyMicros(100_000_000),
        open_orders=(_resting("paper-order-1", 100),),
    )
    feed = _StubFeed(batches=[(_booked_fill(),)])
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    expectations = source.get_expectations()

    assert expectations.expected_open_order_ids == frozenset({"paper-order-1"})


def test_a_booked_resting_order_enters_the_open_order_expectation() -> None:
    """The headline of issue #390: a partial fill leaving a remainder is clean.

    A limit that crosses part of the book fills that part and rests the rest.
    Before #390 the expectation had no way to learn the remainder's venue order
    id -- ids were never ledgered at all -- so the surviving remainder read as
    unexplained venue movement and HALTed the loop the moment it rested an
    order.
    """
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(
        batches=[
            (
                _booked_fill(cash_delta_micros=-5_000_000, position_delta_centis=100),
                _booked_resting_order(resting_quantity_centis=200),
            )
        ]
    )
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    venue.available = MoneyMicros(95_000_000)
    venue.positions = _held(100)
    venue.open_orders = (_resting("paper-order-1", 200),)
    outcome, writer = _run_cycle(venue, source)

    assert outcome is VerificationOutcome.CLEAN
    assert [event.event_type for event in writer.events] == ["VerificationPassed"]


def test_a_fill_naming_its_resting_order_shrinks_then_retires_it() -> None:
    """A booked fill names the order it filled against and consumes it.

    The remainder shrinks by the fill's own size while it survives, and the id
    leaves the expectation exactly when the order is exhausted -- which is what
    lets an outright retirement reconcile instead of reading as an open order
    that vanished for no reason.
    """
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(
        batches=[
            (_booked_resting_order(resting_quantity_centis=300),),
            (
                _booked_fill(
                    fill_id="paper-fill-1",
                    cash_delta_micros=-5_000_000,
                    position_delta_centis=100,
                    venue_order_id="paper-order-1",
                ),
            ),
            (
                _booked_fill(
                    fill_id="paper-fill-2",
                    cash_delta_micros=-10_000_000,
                    position_delta_centis=200,
                    venue_order_id="paper-order-1",
                ),
            ),
        ]
    )
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    assert source.get_expectations().expected_open_order_ids == frozenset(
        {"paper-order-1"}
    )
    assert source.get_expectations().expected_open_order_ids == frozenset(
        {"paper-order-1"}
    )
    assert source.get_expectations().expected_open_order_ids == frozenset()


def test_a_resting_order_nobody_booked_still_breaches() -> None:
    """Non-vacuity: an order the venue rests that no booking explains halts.

    This is the guard against "advancing" the open-order dimension by mirroring
    ``get_open_orders`` into the expectation, which would compare the venue
    against itself and could never fail (issue #352).
    """
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    source = LedgerExpectationSource([], venue, fill_accounting=_StubFeed())

    venue.open_orders = (_resting("phantom-order-9", 300),)
    outcome, writer = _run_cycle(venue, source)

    assert outcome is VerificationOutcome.BREACH
    assert [event.event_type for event in writer.events] == ["VerificationMismatch"]


def test_a_resting_order_that_vanished_unexplained_still_breaches() -> None:
    """The mirror non-vacuity case: an order that left with no booked fill.

    A venue-side cancel or expiry retires an order without any execution behind
    it. Nothing in the books explains the disappearance, so the expectation
    keeps the id and the cycle breaches.
    """
    venue = _MutableVenue(
        available=MoneyMicros(100_000_000),
        open_orders=(_resting("paper-order-1", 300),),
    )
    source = LedgerExpectationSource([], venue, fill_accounting=_StubFeed())

    venue.open_orders = ()
    outcome, _ = _run_cycle(venue, source)

    assert outcome is VerificationOutcome.BREACH


def test_a_fill_retiring_less_than_the_venue_did_still_breaches() -> None:
    """A booking that under-explains the retirement does not absorb it.

    The books say the order shrank to 200; the venue dropped it entirely. Only
    the explained part of the movement is absorbed, so the disagreement survives.
    """
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(
        batches=[
            (
                _booked_resting_order(resting_quantity_centis=300),
                _booked_fill(
                    cash_delta_micros=-5_000_000,
                    position_delta_centis=100,
                    venue_order_id="paper-order-1",
                ),
            )
        ]
    )
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    venue.available = MoneyMicros(95_000_000)
    venue.positions = _held(100)
    venue.open_orders = ()
    outcome, _ = _run_cycle(venue, source)

    assert outcome is VerificationOutcome.BREACH


def test_a_fill_naming_an_order_the_books_never_rested_is_a_gap() -> None:
    """Fail closed: a fill cannot retire an order the expectation never held.

    Applying it would silently invent -- then discard -- a resting order, which
    is exactly the unexplained gap the latch exists to preserve.
    """
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(
        batches=[
            (_booked_fill(venue_order_id="never-rested-1"),),
            (_booked_fill(fill_id="paper-fill-2"),),
        ]
    )
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    source.get_expectations()

    assert source.get_expectations().expected_available_cash == MoneyMicros(100_000_000)


def test_a_malformed_resting_order_entry_is_a_gap() -> None:
    """A resting-order entry whose quantity is not a scaled int is a gap.

    Same fail-closed narrowing every other booked leaf gets: a malformed payload
    is not a fact, and the advance stops at it rather than stepping past it.
    """
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    malformed = Event(
        event_type="RestingOrderAccounted",
        component="scheduler",
        payload_schema_version=1,
        payload={
            "venue_order_id": "paper-order-1",
            "ticker": _TICKER,
            "resting_quantity_centis": "300",
        },
    )
    feed = _StubFeed(batches=[(malformed,), (_booked_fill(),)])
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    source.get_expectations()

    assert source.get_expectations().expected_available_cash == MoneyMicros(100_000_000)


def test_a_non_positive_booked_resting_quantity_is_a_gap() -> None:
    """An order booked as resting nothing is malformed, not a retirement.

    Retirement is expressed by a fill naming the order, never by booking a
    zero-sized arrival, so a non-positive quantity means the seam delivered
    something it should not have.
    """
    venue = _MutableVenue(available=MoneyMicros(100_000_000))
    feed = _StubFeed(
        batches=[(_booked_resting_order(resting_quantity_centis=0),), (_booked_fill(),)]
    )
    source = LedgerExpectationSource([], venue, fill_accounting=feed)

    source.get_expectations()

    assert source.get_expectations().expected_available_cash == MoneyMicros(100_000_000)


def test_the_open_order_baseline_still_carries_the_startup_capture() -> None:
    """Booking changes only the advance; the startup seed is untouched.

    The baseline is still the connector's resting-order set as captured once at
    construction, so an order already resting when the process started is
    expected without any booking at all.
    """
    venue = _MutableVenue(
        available=MoneyMicros(100_000_000),
        open_orders=(_resting("pre-existing-1", 500),),
    )
    source = LedgerExpectationSource([], venue, fill_accounting=_StubFeed())

    venue.open_orders = ()

    assert source.get_expectations().expected_open_order_ids == frozenset(
        {"pre-existing-1"}
    )
