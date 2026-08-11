"""Project venue executions into ledgered accounting entries (issues #365, #390).

The Risk Kernel never imports connector types (SPEC S5 Process-B trust
boundary), so it cannot read a :class:`~windbreak.connector.models.Fill` or an
:class:`~windbreak.connector.models.OpenOrder`. The scheduler is the composition
layer permitted to see both sides -- the same role
:mod:`windbreak.scheduler.eligibility` plays for market metadata -- and this
module is the one-way translation it performs for order activity: a venue
``Fill`` or a venue order arrival in, a hash-chained
:class:`~windbreak.ledger.events.FillAccounted` or
:class:`~windbreak.ledger.events.RestingOrderAccounted` entry out, and from there
only ``int``/``str`` payload leaves reach
:class:`~windbreak.riskkernel.verification.LedgerExpectationSource`.

Why book at all. The kernel's reconciliation baseline is frozen at process
start, which is what keeps the comparison falsifiable rather than the issue #352
tautology of grading the venue against itself. Frozen forever, though, the first
real fill moved the venue off that baseline permanently and the kernel HALTed
until a restart; and the first order left resting did the same to the open-order
dimension. Booking each execution and each arrival once, durably, gives the
expectation something to advance *from* that is not the connector: the entry is
written at the moment it happens and never rewritten, while the observation
stays the venue's live aggregate. A venue that moves by anything other than what
was booked still diverges and still halts.

What is booked is always a *discrete report* -- an execution, or an order's
arrival on the resting book -- never the account's aggregate state. That is the
whole safety property. Booking ``get_open_orders`` instead of arrivals would
mirror the very view the cycle then compares against, and the open-order check
could never fail.

Two halves:

* :class:`LedgerFillBookkeeper` -- the write side. Books each fill exactly once,
  keyed on the venue's own fill id, and each order arrival exactly once, keyed
  on the venue's own order id, so a re-polled report can never be double-booked
  into cash the venue only moved once or an order it only rested once.
* :class:`LedgerFillAccountingFeed` -- the read side, and where the trust
  decision lives. The kernel is handed only entries stamped by the component the
  composition root names as this account's bookkeeper, starting after the chain
  position the baseline was captured over. One named ``ledger`` volume is
  mounted across every process (``deploy/docker-compose.yml``) and
  :meth:`~windbreak.ledger.store.SqliteLedgerStore.read_all` returns the whole
  chain unscoped, so a ledger routinely carries other components' rows; folding
  one of those would describe an account this process has never held.

Every value on this path is a :mod:`windbreak.numeric` scaled integer or a plain
``int`` -- never a float (SPEC S6.1, enforced by ``scripts/lint_no_floats.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from windbreak.ledger.events import FillAccounted, RestingOrderAccounted
from windbreak.ledger.store import events_from_records

if TYPE_CHECKING:
    from collections.abc import Iterator

    from windbreak.connector.models import Fill, OpenOrder
    from windbreak.ledger.events import Event
    from windbreak.ledger.store import LedgerRecord
    from windbreak.numeric.types import MoneyMicros

#: The ledger ``event_type`` booked entries carry, matched verbatim by the
#: kernel's fold. It is the class name, as for every typed event.
_FILL_ACCOUNTED_EVENT_TYPE = "FillAccounted"

#: The ledger ``event_type`` booked order arrivals carry (issue #390).
_RESTING_ORDER_ACCOUNTED_EVENT_TYPE = "RestingOrderAccounted"

#: Every entry kind the feed hands the kernel, and therefore every kind the
#: kernel's fold must know how to apply. Anything else on the seam is a gap in
#: the books, never a row to skip.
_ACCOUNTING_EVENT_TYPES: tuple[str, ...] = (
    _RESTING_ORDER_ACCOUNTED_EVENT_TYPE,
    _FILL_ACCOUNTED_EVENT_TYPE,
)

#: An instant no real fill can precede, used as the ``get_fills`` lower bound.
#: The bookkeeper deliberately re-reads the venue's whole fill log every call
#: and de-duplicates on the fill *id* rather than advancing a timestamp cursor:
#: a taker walk emits one fill per consumed book level, all stamped at the same
#: venue instant, so a timestamp cursor would silently drop every level after
#: the first -- an under-booked expectation that fails *open* against cash the
#: account really spent.
_EPOCH_FLOOR = datetime(1970, 1, 1, tzinfo=UTC)


class FillSource(Protocol):
    """The venue seam the bookkeeper reads order activity through.

    Every method describes a *discrete report*, never the account's aggregate
    state: :meth:`get_fills` is the venue's execution report,
    :meth:`fill_cash_micros` is what one execution cost fee included, and
    :meth:`get_rested_orders` is the log of orders that came to rest. The
    account's balances, positions, and live resting book are read by the
    verification cycle, separately and later, which is what keeps the two sides
    independent.
    """

    def get_fills(self, since: datetime) -> tuple[Fill, ...]:
        """Return the fills executed strictly after ``since``.

        Args:
            since: The exclusive lower bound on execution time.

        Returns:
            The matching fills, in execution order.
        """
        ...

    def fill_cash_micros(self, fill: Fill, /) -> MoneyMicros:
        """Return the cash ``fill`` moved out of the account, in micros.

        Args:
            fill: The executed fill to price.

        Returns:
            The positive magnitude of the cash the fill consumed, book cost plus
            the fee charged on it.
        """
        ...

    def get_rested_orders(self) -> tuple[OpenOrder, ...]:
        """Return the append-only log of orders that came to rest.

        An *arrival report*, not the account's live resting book: an order that
        later fills or is cancelled leaves ``get_open_orders`` but never leaves
        this log. Booking arrivals rather than mirroring the live book is what
        keeps the open-order dimension falsifiable (issue #390); see
        :meth:`~windbreak.connector.paper.PaperExchange.get_rested_orders`.

        Returns:
            The orders that came to rest, in arrival order.
        """
        ...

    def resting_collateral_micros(self, order: OpenOrder, /) -> MoneyMicros:
        """Return the cash this venue withholds from ``available`` for ``order``.

        A *discrete report* about one order, like every other method here, never
        the account's aggregate withheld total: the reconciliation cycle
        compares against the aggregate ``available``, so advancing the
        expectation from that same aggregate would grade the venue against
        itself and could never fail (issue #352).

        Withholding is venue-dependent, which is exactly why
        :class:`~windbreak.connector.semantics.OrderCollateralInAvailable`
        exists, and each venue answers for its own convention here. A venue
        whose answer is not ``DEDUCTED_FROM_AVAILABLE`` withholds nothing and
        must return ``0``; the booking records that zero rather than inventing a
        reservation, because a booked reservation the venue never made would
        push the cash expectation below an ``available`` that never moved.

        Args:
            order: The order that came to rest, as reported by
                :meth:`get_rested_orders`; its quantity is the size that rested.

        Returns:
            The positive magnitude the venue withheld from ``available`` for
            this order, or ``MoneyMicros(0)`` on a venue that withholds nothing.
        """
        ...


class FillLedgerWriter(Protocol):
    """The append-only seam booked entries are written through."""

    def append(self, event: Event) -> int:
        """Append ``event`` to the hash chain.

        Args:
            event: The event to append.

        Returns:
            The appended record's sequence number.
        """
        ...


class FillRecordScan(Protocol):
    """The reverse-by-type read seam the feed streams booked entries through."""

    def iter_records_of_type_reversed(self, event_type: str) -> Iterator[LedgerRecord]:
        """Walk every record of ``event_type``, newest first.

        Args:
            event_type: The single event type to walk.

        Yields:
            The matching records in descending ``sequence_number`` order.
        """
        ...


def _position_delta_centis(fill: Fill) -> int:
    """Return ``fill``'s signed position movement, in the YES frame.

    Positions are reported in the YES frame -- positive is long YES, negative is
    long NO -- because a ticker gets exactly one row
    (:func:`windbreak.connector.paper._position_row`). The expectation's position
    dimension is diffed against those rows, so a booking must use the identical
    frame or a NO fill would read as a long and the drift would be double its
    true size.

    Args:
        fill: The executed fill.

    Returns:
        The signed movement in contract-centis: positive for a YES fill,
        negative for a NO fill.
    """
    if fill.side == "yes":
        return fill.quantity.value
    return -fill.quantity.value


class LedgerFillBookkeeper:
    """Books an account's fills and order arrivals into the ledger, once each.

    The de-duplication sets are per-process and per-instance, which is the
    correct scope: the feed's cursor starts at the chain head the expectation's
    baseline was captured over, so entries booked by an earlier process are
    already reflected in that baseline and are never folded again.
    """

    def __init__(
        self, writer: FillLedgerWriter, venue: FillSource, *, component: str
    ) -> None:
        """Bind the bookkeeper to one ledger and one venue.

        Args:
            writer: The append-only ledger seam entries are booked through.
            venue: The venue seam executions and their cost are read from.
            component: The ledger component to stamp bookings with. It must be
                the same label the paired :class:`LedgerFillAccountingFeed` is
                told to trust; that pairing is this account's identity on a
                ledger shared by several processes.
        """
        self._writer = writer
        self._venue = venue
        self._component = component
        self._booked: set[str] = set()
        self._booked_orders: set[str] = set()

    def book_new(self) -> int:
        """Book every arrival and execution not yet booked, and count them.

        Arrivals are booked *before* executions, and that order is load-bearing:
        the kernel's fold draws a named resting order down by the fill that
        consumed it, so an arrival landing after its own fill would present as a
        fill naming an order the expectation has never held -- an unexplained
        gap that latches the advance off. Within one call the venue's arrival
        log is always at least as complete as its fill log, because an order
        must rest before anything can fill against it.

        Returns:
            The number of entries appended; ``0`` when nothing new happened.
        """
        return self._book_arrivals() + self._book_fills()

    def _book_arrivals(self) -> int:
        """Book every order that has come to rest since the last call.

        The whole arrival log is re-read and de-duplicated on the venue's order
        id, exactly as :meth:`_book_fills` de-duplicates on the fill id: the log
        is small, and an id key cannot silently drop entries the way a timestamp
        cursor can.

        Each arrival also carries the collateral the venue withheld from
        ``available`` for that order (issue #423), asked of the venue rather
        than re-derived here for the same reason ``cash_delta_micros`` is asked
        of :meth:`FillSource.fill_cash_micros`: a second implementation of
        book-cost-plus-fee would drift from the venue the first time either
        rounding rule moved, and a bookkeeping drift is indistinguishable from
        the venue divergence reconciliation exists to catch.

        Returns:
            The number of arrival entries appended.
        """
        booked = 0
        for order in self._venue.get_rested_orders():
            if order.id in self._booked_orders:
                continue
            self._writer.append(
                RestingOrderAccounted(
                    component=self._component,
                    venue_order_id=order.id,
                    ticker=order.ticker,
                    resting_quantity_centis=order.quantity.value,
                    reserved_collateral_micros=(
                        self._venue.resting_collateral_micros(order).value
                    ),
                )
            )
            self._booked_orders.add(order.id)
            booked += 1
        return booked

    def _book_fills(self) -> int:
        """Book every execution not yet booked, and return how many were.

        Returns:
            The number of fill entries appended.
        """
        booked = 0
        for fill in self._venue.get_fills(_EPOCH_FLOOR):
            if fill.id in self._booked:
                continue
            self._writer.append(
                FillAccounted(
                    component=self._component,
                    fill_id=fill.id,
                    ticker=fill.ticker,
                    cash_delta_micros=-self._venue.fill_cash_micros(fill).value,
                    position_delta_centis=_position_delta_centis(fill),
                    venue_order_id=fill.order_id or "",
                )
            )
            self._booked.add(fill.id)
            booked += 1
        return booked


class LedgerFillAccountingFeed:
    """Streams this account's booked fill entries to the kernel, once each.

    Satisfies
    :class:`~windbreak.riskkernel.verification.FillAccountingFeed`. The walk runs
    newest-first over the ledger's ``(event_type, sequence_number)`` index and
    stops the moment it reaches the cursor, so an always-on loop pays for the
    entries booked since the last cycle rather than for the whole chain.
    """

    def __init__(
        self, scan: FillRecordScan, *, component: str, after_sequence: int
    ) -> None:
        """Bind the feed to one ledger, one component, and one starting cursor.

        Args:
            scan: The reverse-by-type ledger read seam.
            component: The only component whose bookings describe this account.
                Rows stamped by anything else belong to another process sharing
                the ledger volume and are never handed to the kernel.
            after_sequence: The chain position the expectation's baseline was
                captured over. Entries at or before it are already reflected in
                that baseline; folding them again would advance the expectation
                past cash the venue moved once.
        """
        self._scan = scan
        self._component = component
        self._cursor = after_sequence

    def drain(self) -> tuple[Event, ...]:
        """Return the entries booked since the last call, oldest first.

        Both booked entry kinds come back through this one seam, interleaved in
        chain order. The ledger's reverse-by-type scan takes one literal type at
        a time (so the package's SQL stays statically auditable), so each kind is
        walked separately and the two bounded results are merged on
        ``sequence_number``. Chain order is not cosmetic: the kernel draws a
        resting order down by the fill that consumed it, so an arrival must be
        folded before its own fill.

        Returns:
            The newly booked ``RestingOrderAccounted`` and ``FillAccounted``
            events; empty when nothing has been booked since the last call.
        """
        fresh: list[LedgerRecord] = []
        for event_type in _ACCOUNTING_EVENT_TYPES:
            fresh.extend(self._fresh_of_type(event_type))
        if not fresh:
            return ()
        fresh.sort(key=lambda record: record.sequence_number)
        self._cursor = fresh[-1].sequence_number
        return events_from_records(
            record for record in fresh if record.component == self._component
        )

    def _fresh_of_type(self, event_type: str) -> list[LedgerRecord]:
        """Return this type's records booked after the cursor, newest first.

        The walk stops the moment it reaches the cursor, so an always-on loop
        pays for the entries booked since the last cycle rather than for the
        whole chain.

        Args:
            event_type: The single event type to walk.

        Returns:
            The matching records above the cursor, in descending sequence order.
        """
        fresh: list[LedgerRecord] = []
        for record in self._scan.iter_records_of_type_reversed(event_type):
            if record.sequence_number <= self._cursor:
                break
            fresh.append(record)
        return fresh
