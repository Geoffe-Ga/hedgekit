"""Failing-first tests for the fill-accounting projection seam (issue #365, RED).

The Risk Kernel never imports connector types (SPEC S5 Process-B trust
boundary), so it cannot read a :class:`~windbreak.connector.models.Fill`. The
scheduler is the composition layer permitted to see both sides -- the same role
``windbreak/scheduler/eligibility.py`` plays for market metadata -- and
``windbreak/scheduler/fill_accounting.py`` is the one-way translation it
performs for executions: venue ``Fill`` in, ledgered ``FillAccounted`` entry
out, then plain ``int``/``str`` payload leaves back into the kernel's
expectation fold.

Two halves, both tested here:

* ``LedgerFillBookkeeper`` -- books each venue fill exactly once, keyed on the
  venue's own fill id, so a re-polled execution report can never be
  double-booked into the hash chain.
* ``LedgerFillAccountingFeed`` -- hands the kernel's expectation the entries
  booked *for this account*, once each. It is where the trust decision lives:
  which component is believed, and where the baseline's cursor starts.

``windbreak/scheduler/fill_accounting.py`` does not exist yet, so every import
below fails collection -- the expected Gate 1 RED state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from windbreak.connector.models import Fill
from windbreak.ledger.events import ConfigLoaded, Event
from windbreak.ledger.store import SqliteLedgerStore, events_from_records
from windbreak.numeric.types import ContractCentis, MoneyMicros, PricePips
from windbreak.scheduler.fill_accounting import (
    LedgerFillAccountingFeed,
    LedgerFillBookkeeper,
)

if TYPE_CHECKING:
    from pathlib import Path

#: The market every fill below trades.
_TICKER = "KXFED-24DEC"

#: The component the PAPER loop books its account's fill accounting under.
_COMPONENT = "scheduler"


def _fill(
    fill_id: str,
    *,
    side: str = "yes",
    price_pips: int = 5000,
    quantity_centis: int = 100,
) -> Fill:
    """Build one venue fill.

    Args:
        fill_id: The venue's own identifier for the fill.
        side: The side that traded, `"yes"` or `"no"`.
        price_pips: The execution price, in pips.
        quantity_centis: The executed size, in contract-centis.

    Returns:
        The constructed `Fill`.
    """
    assert side in {"yes", "no"}
    return Fill(
        id=fill_id,
        ticker=_TICKER,
        side="yes" if side == "yes" else "no",
        price=PricePips(price_pips),
        quantity=ContractCentis(quantity_centis),
        ts=datetime(2024, 6, 1, 12, tzinfo=UTC),
    )


@dataclass
class _StubVenue:
    """A minimal fill source standing in for a `PaperExchange`.

    Attributes:
        fills: Every fill the venue has executed, oldest first.
        cash_by_fill_id: The cash each fill moved out of the account, in
            micros, keyed by fill id; a fill absent from the mapping costs
            `_DEFAULT_CASH_MICROS`.
    """

    fills: list[Fill] = field(default_factory=list)
    cash_by_fill_id: dict[str, int] = field(default_factory=dict)

    def get_fills(self, since: datetime) -> tuple[Fill, ...]:
        """Return every executed fill strictly after ``since``."""
        return tuple(fill for fill in self.fills if fill.ts > since)

    def fill_cash_micros(self, fill: Fill) -> MoneyMicros:
        """Return the cash ``fill`` moved out of the account, in micros."""
        return MoneyMicros(self.cash_by_fill_id.get(fill.id, _DEFAULT_CASH_MICROS))


#: The cash a fill costs when a test does not care about the exact figure.
_DEFAULT_CASH_MICROS = 5_000_000


@pytest.fixture(name="store")
def _store(tmp_path: Path) -> SqliteLedgerStore:
    """Provide a fresh hash-chained ledger on disk.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Yields:
        The opened `SqliteLedgerStore`, closed on teardown.
    """
    opened = SqliteLedgerStore(tmp_path / "ledger.db")
    yield opened
    opened.close()


def _booked(store: SqliteLedgerStore) -> list[Event]:
    """Return every `FillAccounted` row on ``store``, oldest first.

    Read back through `events_from_records`, which rebuilds *base* `Event`
    objects rather than typed subclasses, so an entry is inspected exactly the
    way the kernel's fold inspects it: through its payload leaves.

    Args:
        store: The ledger to read.

    Returns:
        The booked entries, in chain order.
    """
    return [
        event
        for event in events_from_records(store.read_all())
        if event.event_type == "FillAccounted"
    ]


def test_a_yes_fill_books_a_negative_cash_and_positive_position_delta(
    store: SqliteLedgerStore,
) -> None:
    """Buying YES spends cash and lengthens the YES-frame position."""
    venue = _StubVenue(fills=[_fill("paper-fill-1")])
    bookkeeper = LedgerFillBookkeeper(store, venue, component=_COMPONENT)

    assert bookkeeper.book_new() == 1
    (entry,) = _booked(store)
    assert entry.payload["fill_id"] == "paper-fill-1"
    assert entry.payload["ticker"] == _TICKER
    assert entry.payload["cash_delta_micros"] == -_DEFAULT_CASH_MICROS
    assert entry.payload["position_delta_centis"] == 100
    assert entry.component == _COMPONENT


def test_a_no_fill_books_a_negative_position_delta(store: SqliteLedgerStore) -> None:
    """A NO fill is economically a short YES, so it books negatively -- the same
    YES frame `Position.quantity` reports, which is what the expectation's
    position dimension is diffed in."""
    venue = _StubVenue(fills=[_fill("paper-fill-1", side="no", quantity_centis=40)])
    bookkeeper = LedgerFillBookkeeper(store, venue, component=_COMPONENT)

    bookkeeper.book_new()

    (entry,) = _booked(store)
    assert entry.payload["position_delta_centis"] == -40
    assert entry.payload["cash_delta_micros"] == -_DEFAULT_CASH_MICROS


def test_a_fill_is_never_booked_twice(store: SqliteLedgerStore) -> None:
    """The venue's fill id is the idempotency key: re-polling the same
    execution report must not double-book it into the hash chain, which would
    advance the expectation past cash the venue never moved."""
    venue = _StubVenue(fills=[_fill("paper-fill-1")])
    bookkeeper = LedgerFillBookkeeper(store, venue, component=_COMPONENT)

    assert bookkeeper.book_new() == 1
    assert bookkeeper.book_new() == 0
    assert len(_booked(store)) == 1


def test_only_the_newly_executed_fills_are_booked(store: SqliteLedgerStore) -> None:
    """A later call books the fills that appeared since, and only those."""
    venue = _StubVenue(fills=[_fill("paper-fill-1")])
    bookkeeper = LedgerFillBookkeeper(store, venue, component=_COMPONENT)
    bookkeeper.book_new()

    venue.fills.append(_fill("paper-fill-2", quantity_centis=25))

    assert bookkeeper.book_new() == 1
    assert [entry.payload["fill_id"] for entry in _booked(store)] == [
        "paper-fill-1",
        "paper-fill-2",
    ]


def test_fills_sharing_an_execution_timestamp_are_all_booked(
    store: SqliteLedgerStore,
) -> None:
    """A taker walk emits one fill per consumed book level, all stamped at the
    same venue instant. Booking must key on the fill id, never on a timestamp
    cursor, or every level after the first would be silently dropped -- an
    under-booked expectation fails *open* against real spent cash."""
    venue = _StubVenue(fills=[_fill("paper-fill-1"), _fill("paper-fill-2")])
    bookkeeper = LedgerFillBookkeeper(store, venue, component=_COMPONENT)

    assert bookkeeper.book_new() == 2


def test_the_feed_drains_each_booked_entry_exactly_once(
    store: SqliteLedgerStore,
) -> None:
    """The seam's contract: entries not returned before, oldest first, never
    twice -- folding one entry twice would advance the expectation past cash
    the venue only moved once."""
    venue = _StubVenue(fills=[_fill("paper-fill-1")])
    bookkeeper = LedgerFillBookkeeper(store, venue, component=_COMPONENT)
    feed = LedgerFillAccountingFeed(store, component=_COMPONENT, after_sequence=0)
    bookkeeper.book_new()

    first = feed.drain()
    second = feed.drain()

    assert [event.payload["fill_id"] for event in first] == ["paper-fill-1"]
    assert second == ()


def test_the_feed_returns_entries_oldest_first(store: SqliteLedgerStore) -> None:
    """Order matters: the deltas compound, and a gap must stop the advance at
    the right point."""
    venue = _StubVenue(fills=[_fill("paper-fill-1"), _fill("paper-fill-2")])
    LedgerFillBookkeeper(store, venue, component=_COMPONENT).book_new()
    feed = LedgerFillAccountingFeed(store, component=_COMPONENT, after_sequence=0)

    assert [event.payload["fill_id"] for event in feed.drain()] == [
        "paper-fill-1",
        "paper-fill-2",
    ]


def test_the_feed_ignores_another_component_s_bookings(
    store: SqliteLedgerStore,
) -> None:
    """One `ledger` volume is mounted across every process, so a ledger routinely
    carries other components' rows. The feed hands the kernel only the account
    it was told to trust; a foreign booking is somebody else's account."""
    venue = _StubVenue(fills=[_fill("paper-fill-1")])
    LedgerFillBookkeeper(store, venue, component="somebody_else").book_new()
    feed = LedgerFillAccountingFeed(store, component=_COMPONENT, after_sequence=0)

    assert feed.drain() == ()


def test_the_feed_starts_after_its_cursor(store: SqliteLedgerStore) -> None:
    """Entries already reflected in the baseline the expectation captured must
    not be folded again. The composition root sets the cursor to the chain head
    it built that baseline over."""
    venue = _StubVenue(fills=[_fill("paper-fill-1")])
    LedgerFillBookkeeper(store, venue, component=_COMPONENT).book_new()
    head = store.head()
    assert head is not None
    feed = LedgerFillAccountingFeed(
        store, component=_COMPONENT, after_sequence=head.sequence_number
    )

    assert feed.drain() == ()


def test_the_feed_ignores_rows_of_every_other_type(store: SqliteLedgerStore) -> None:
    """Unrelated ledger traffic is not a gap in the books -- only a malformed
    *booking* is. The feed filters by type so ordinary loop chatter between two
    fills never latches the expectation off."""
    store.append(ConfigLoaded(component=_COMPONENT, config_hash="abc", diff={}))
    venue = _StubVenue(fills=[_fill("paper-fill-1")])
    LedgerFillBookkeeper(store, venue, component=_COMPONENT).book_new()
    store.append(ConfigLoaded(component=_COMPONENT, config_hash="def", diff={}))
    feed = LedgerFillAccountingFeed(store, component=_COMPONENT, after_sequence=0)

    drained = feed.drain()

    assert [event.event_type for event in drained] == ["FillAccounted"]


def test_a_booked_entry_survives_the_chain_verification(
    store: SqliteLedgerStore,
) -> None:
    """Booking writes through the ordinary append path, so the hash chain stays
    intact and the entries are as auditable as every other row."""
    venue = _StubVenue(fills=[_fill("paper-fill-1")])
    LedgerFillBookkeeper(store, venue, component=_COMPONENT).book_new()

    store.verify_chain()


def test_the_booked_payload_carries_no_leaf_that_is_not_int_or_str(
    store: SqliteLedgerStore,
) -> None:
    """SPEC S6.1: every recorded payload leaf is an `int` or a `str`, never a
    float and never a scaled-integer wrapper the ledger cannot serialize."""
    venue = _StubVenue(fills=[_fill("paper-fill-1")])
    LedgerFillBookkeeper(store, venue, component=_COMPONENT).book_new()

    (entry,) = _booked(store)
    assert all(isinstance(leaf, int | str) for leaf in entry.payload.values())


def test_an_event_drained_from_the_feed_is_what_the_kernel_folds(
    store: SqliteLedgerStore,
) -> None:
    """End-to-end shape check across the seam: what the bookkeeper writes is
    exactly what the expectation's fold reads back, keys and all."""
    venue = _StubVenue(fills=[_fill("paper-fill-1", quantity_centis=100)])
    LedgerFillBookkeeper(store, venue, component=_COMPONENT).book_new()
    feed = LedgerFillAccountingFeed(store, component=_COMPONENT, after_sequence=0)

    (event,) = feed.drain()

    assert isinstance(event, Event)
    assert event.payload["ticker"] == _TICKER
    assert event.payload["cash_delta_micros"] == -_DEFAULT_CASH_MICROS
    assert event.payload["position_delta_centis"] == 100
