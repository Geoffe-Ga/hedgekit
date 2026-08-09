"""Project venue fills into ledgered accounting entries (issue #365).

The Risk Kernel never imports connector types (SPEC S5 Process-B trust
boundary), so it cannot read a :class:`~windbreak.connector.models.Fill`. The
scheduler is the composition layer permitted to see both sides -- the same role
:mod:`windbreak.scheduler.eligibility` plays for market metadata -- and this
module is the one-way translation it performs for executions: a venue ``Fill``
in, a hash-chained :class:`~windbreak.ledger.events.FillAccounted` entry out,
and from there only ``int``/``str`` payload leaves reach
:class:`~windbreak.riskkernel.verification.LedgerExpectationSource`.

Why book at all. The kernel's reconciliation baseline is frozen at process
start, which is what keeps the comparison falsifiable rather than the issue #352
tautology of grading the venue against itself. Frozen forever, though, the first
real fill moved the venue off that baseline permanently and the kernel HALTed
until a restart. Booking each execution once, durably, gives the expectation
something to advance *from* that is not the connector: the entry is written at
execution and never rewritten, while the observation stays the venue's live
aggregate. A venue that moves by anything other than what was booked still
diverges and still halts.

Two halves:

* :class:`LedgerFillBookkeeper` -- the write side. Books each fill exactly once,
  keyed on the venue's own fill id, so a re-polled execution report can never be
  double-booked into cash the venue only moved once.
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

from windbreak.ledger.events import FillAccounted
from windbreak.ledger.store import events_from_records

if TYPE_CHECKING:
    from collections.abc import Iterator

    from windbreak.connector.models import Fill
    from windbreak.ledger.events import Event
    from windbreak.ledger.store import LedgerRecord
    from windbreak.numeric.types import MoneyMicros

#: The ledger ``event_type`` booked entries carry, matched verbatim by the
#: kernel's fold. It is the class name, as for every typed event.
_FILL_ACCOUNTED_EVENT_TYPE = "FillAccounted"

#: An instant no real fill can precede, used as the ``get_fills`` lower bound.
#: The bookkeeper deliberately re-reads the venue's whole fill log every call
#: and de-duplicates on the fill *id* rather than advancing a timestamp cursor:
#: a taker walk emits one fill per consumed book level, all stamped at the same
#: venue instant, so a timestamp cursor would silently drop every level after
#: the first -- an under-booked expectation that fails *open* against cash the
#: account really spent.
_EPOCH_FLOOR = datetime(1970, 1, 1, tzinfo=UTC)


class FillSource(Protocol):
    """The venue seam the bookkeeper reads executions and their cost through.

    Both methods describe *executions*, never the account's aggregate state:
    :meth:`get_fills` is the venue's execution report and
    :meth:`fill_cash_micros` is what that one execution cost, fee included. The
    account's balances and positions are read by the verification cycle,
    separately and later, which is what keeps the two sides independent.
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
    """Books each of an account's venue fills into the ledger exactly once.

    The de-duplication set is per-process and per-instance, which is the correct
    scope: the feed's cursor starts at the chain head the expectation's baseline
    was captured over, so entries booked by an earlier process are already
    reflected in that baseline and are never folded again.
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

    def book_new(self) -> int:
        """Book every execution not yet booked, and return how many were.

        Returns:
            The number of entries appended; ``0`` when nothing new executed.
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

        Returns:
            The newly booked ``FillAccounted`` events; empty when nothing has
            been booked since the last call.
        """
        fresh: list[LedgerRecord] = []
        for record in self._scan.iter_records_of_type_reversed(
            _FILL_ACCOUNTED_EVENT_TYPE
        ):
            if record.sequence_number <= self._cursor:
                break
            fresh.append(record)
        if not fresh:
            return ()
        self._cursor = fresh[0].sequence_number
        fresh.reverse()
        return events_from_records(
            record for record in fresh if record.component == self._component
        )
