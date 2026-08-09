"""Tests for the live-market-data paper session (issue #343).

Real books in, paper money out. Two objects carry that:

* :class:`windbreak.connector.live.MarketDataOnlyView` -- the credential-free
  read surface the live path is allowed to hold (SPEC S1.1 invariant 3). These
  tests pin the *absence* of the order and account methods, not merely that
  calling them fails.
* :class:`windbreak.connector.paper.LiveBookPaperExchange` -- a
  :class:`~windbreak.connector.paper.PaperExchange` whose books, market,
  status, venue clock, and fee schedule are read live on every call, while
  fills, positions, and balances stay simulated.

The assertions that matter most are the ones about *timestamps and prices*: a
live book's ``fetched_at`` is the venue's observation, never restamped, so the
existing freshness check can still veto a stale book; and a taker fill is
priced and dated off that same live book rather than a replayed fixture step.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.config import RiskConfig
from windbreak.connector.fees import FeeModel
from windbreak.connector.freshness import is_fresh
from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.live import MarketDataOnlyView, MarketDataSource
from windbreak.connector.models import (
    ExchangeStatus,
    NormalizedMarket,
    OrderBookLevel,
    OrderBookSnapshot,
)
from windbreak.connector.paper import LiveBookPaperExchange, PaperOrderIntent
from windbreak.numeric import ContractCentis, PricePips

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The account fixtures (opening balance + balance semantics) a live-book
#: session borrows. Its books and markets are never read in live mode -- only
#: the paper account's opening state is.
_ACCOUNT_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "books" / "deep_walk"

#: The single ticker the live session is bound to (#345 owns the universe).
_TICKER = "KXLIVE-25JAN"

#: The venue's own book timestamp. Deliberately distinct from every other
#: instant in this module so an assertion cannot pass by coincidence.
_VENUE_BOOK_AT = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)

#: The venue's own clock reading, distinct from both the book stamp and the
#: local observation clock so ``get_exchange_time`` cannot be satisfied by
#: either of them.
_VENUE_CLOCK = datetime(2025, 1, 2, 3, 4, 30, tzinfo=UTC)

#: The venue's status observation instant.
_VENUE_STATUS_AT = datetime(2025, 1, 2, 3, 4, 6, tzinfo=UTC)

#: The *local* observation clock. Never the venue's, which is the whole point
#: of the clock-skew check the loop runs.
_LOCAL_NOW = datetime(2025, 1, 2, 3, 5, 0, tzinfo=UTC)


def _level(price: int, quantity: int) -> OrderBookLevel:
    """Build one book level from plain pip/centi ints.

    Args:
        price: The level's price, in pips.
        quantity: The level's resting size, in contract-centis.

    Returns:
        The order-book level.
    """
    return OrderBookLevel(PricePips(price), ContractCentis(quantity))


def _book(
    *,
    asks: Sequence[tuple[int, int]],
    bids: Sequence[tuple[int, int]] = ((4400, 500),),
    fetched_at: datetime = _VENUE_BOOK_AT,
) -> OrderBookSnapshot:
    """Build a venue book snapshot from plain ``(price, quantity)`` pairs.

    Args:
        asks: The YES ask levels, best-first.
        bids: The YES bid levels, best-first.
        fetched_at: The venue's own observation instant for this book.

    Returns:
        The order-book snapshot.
    """
    return OrderBookSnapshot(
        ticker=_TICKER,
        yes_bids=tuple(_level(price, size) for price, size in bids),
        yes_asks=tuple(_level(price, size) for price, size in asks),
        fetched_at=fetched_at,
    )


#: A real Kalshi market carries ``jurisdiction_status="unknown"`` -- the venue
#: publishes no per-market eligibility signal -- so the double carries it too.
#: The kernel vetoing on it is correct behavior, not a gap this issue fills.
_MARKET = NormalizedMarket(
    exchange="kalshi",
    ticker=_TICKER,
    event_ticker="KXLIVE",
    title="A live market",
    resolution_criteria="Test double; not a real resolution.",
    category="Economics",
    close_time=datetime(2025, 6, 1, tzinfo=UTC),
    expected_resolution_time=None,
    market_type="fully_collateralized_binary",
    price_tick_pips=100,
    min_order_contract_centis=100,
    fractional_trading_enabled=False,
    mutually_exclusive_group_id=None,
    jurisdiction_status="unknown",
    raw_exchange_payload_hash="sha256:live0001",
    volume_24h_micros=5_000_000_000,
)

#: The venue's fee schedule, distinct from the account fixture's own
#: ``paper-test-v1`` schedule so a fee read that fell back to the fixture is
#: distinguishable from one that reached the venue.
_VENUE_FEE_MODEL = FeeModel(
    schedule_id="kalshi-live-v1",
    maker_fee_ppm=0,
    taker_fee_ppm=70_000,
    settlement_fee_ppm=0,
)


class _FakeVenue:
    """A scripted read-only market-data source standing in for a live venue.

    Every answer is mutable so a test can move the venue between two reads,
    which is how the "reads through rather than snapshotting" assertions are
    made falsifiable.
    """

    def __init__(self) -> None:
        """Initialize the venue at its default book, market, status, and clock."""
        self.book = _book(asks=((4500, 300), (4600, 200)))
        self.market = _MARKET
        self.status = ExchangeStatus(status="open", fetched_at=_VENUE_STATUS_AT)
        self.venue_time = _VENUE_CLOCK
        self.fee_model = _VENUE_FEE_MODEL
        self.book_reads = 0

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Return the scripted market, or raise for any other ticker.

        Args:
            ticker: The market ticker being looked up.

        Returns:
            The scripted normalized market.

        Raises:
            UnknownMarketError: If ``ticker`` is not the scripted one.
        """
        if ticker != self.market.ticker:
            raise UnknownMarketError(ticker)
        return self.market

    def get_order_book(self, ticker: str) -> OrderBookSnapshot:
        """Return the scripted book, counting the read.

        Args:
            ticker: The market ticker being looked up.

        Returns:
            The scripted order-book snapshot.

        Raises:
            UnknownMarketError: If ``ticker`` is not the scripted one.
        """
        if ticker != self.market.ticker:
            raise UnknownMarketError(ticker)
        self.book_reads += 1
        return self.book

    def get_exchange_status(self) -> ExchangeStatus:
        """Return the scripted exchange status."""
        return self.status

    def get_exchange_time(self) -> datetime:
        """Return the scripted venue clock."""
        return self.venue_time

    def get_fee_model(self, market_or_series: str) -> FeeModel:
        """Return the scripted fee schedule.

        Args:
            market_or_series: The market or series key (recorded, not routed).

        Returns:
            The scripted fee model.
        """
        del market_or_series
        return self.fee_model


@pytest.fixture
def venue() -> _FakeVenue:
    """Provide a fresh scripted venue."""
    return _FakeVenue()


@pytest.fixture
def exchange(venue: _FakeVenue) -> LiveBookPaperExchange:
    """Provide a live-book paper session over the scripted venue.

    Args:
        venue: The scripted market-data source.

    Returns:
        The live-book paper exchange, on a fixed local observation clock.
    """
    return LiveBookPaperExchange.from_account_dir(
        _ACCOUNT_DIR,
        market_data=MarketDataOnlyView(venue),
        ticker=_TICKER,
        clock=lambda: _LOCAL_NOW,
    )


# --- the credential-free market-data surface ------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        "place_order",
        "cancel_order",
        "get_balances",
        "get_positions",
        "get_open_orders",
        "get_fills",
    ],
)
def test_market_data_view_has_no_order_or_account_surface(method: str) -> None:
    """The live read path exposes market data and nothing else.

    `hasattr` is the assertion on purpose: SPEC S1.1 invariant 3 is that the
    read-only market-data path *cannot* trade or read the account, not that
    doing so fails loudly. A method that exists but raises is still reachable
    through duck-typed code.

    Args:
        method: The order/account method that must be absent.
    """
    view = MarketDataOnlyView(_FakeVenue())

    assert not hasattr(view, method)


def test_market_data_view_satisfies_the_market_data_protocol() -> None:
    """The view is a structural `MarketDataSource`, as is a full connector."""
    venue = _FakeVenue()

    assert isinstance(venue, MarketDataSource)
    assert isinstance(MarketDataOnlyView(venue), MarketDataSource)


def test_market_data_view_forwards_each_read_verbatim() -> None:
    """Every exposed read returns the wrapped source's own answer.

    A view that transformed an answer could report a venue state the venue
    never published -- the precise failure the wrapper exists to rule out.
    """
    venue = _FakeVenue()
    view = MarketDataOnlyView(venue)

    assert view.get_market(_TICKER) == venue.market
    assert view.get_order_book(_TICKER) == venue.book
    assert view.get_exchange_status() == venue.status
    assert view.get_exchange_time() == venue.venue_time
    assert view.get_fee_model(_TICKER) == venue.fee_model


def test_market_data_view_reads_through_rather_than_snapshotting() -> None:
    """A venue that moves is not hidden behind a frozen copy."""
    venue = _FakeVenue()
    view = MarketDataOnlyView(venue)
    venue.book = _book(asks=((4700, 100),))

    assert [level.price.value for level in view.get_order_book(_TICKER).yes_asks] == [
        4700
    ]


# --- real books ------------------------------------------------------------------


def test_get_order_book_refetches_the_venues_book_on_every_read(
    exchange: LiveBookPaperExchange, venue: _FakeVenue
) -> None:
    """Each read reaches the venue again rather than replaying one snapshot.

    Args:
        exchange: The live-book paper session.
        venue: The scripted venue behind it.
    """
    first = exchange.get_order_book(_TICKER)
    venue.book = _book(asks=((4700, 100),))
    second = exchange.get_order_book(_TICKER)

    assert [level.price.value for level in first.yes_asks] == [4500, 4600]
    assert [level.price.value for level in second.yes_asks] == [4700]
    assert venue.book_reads == 2


def test_get_order_book_preserves_the_venues_own_fetched_at(
    exchange: LiveBookPaperExchange,
) -> None:
    """The book's timestamp is the venue's observation, never our clock.

    Restamping it would make the book unfalsifiably fresh: `quote_freshness`
    is measured as ``now - fetched_at``, so a self-renewed stamp can only ever
    pass. Issue #369 fixed exactly that for the replay path.

    Args:
        exchange: The live-book paper session.
    """
    book = exchange.get_order_book(_TICKER)

    assert book.fetched_at == _VENUE_BOOK_AT
    assert book.fetched_at != _LOCAL_NOW


def test_a_stale_venue_book_still_fails_the_existing_freshness_check(
    exchange: LiveBookPaperExchange, venue: _FakeVenue
) -> None:
    """A book older than the quote ttl reads stale, and a recent one reads fresh.

    Both halves are asserted: a check that reported "stale" unconditionally
    would pass the first assertion while being useless.

    Args:
        exchange: The live-book paper session.
        venue: The scripted venue behind it.
    """
    ttl_seconds = RiskConfig().quote_ttl_seconds
    venue.book = _book(
        asks=((4500, 300),), fetched_at=_LOCAL_NOW - timedelta(seconds=ttl_seconds + 1)
    )
    stale = exchange.get_order_book(_TICKER)
    venue.book = _book(asks=((4500, 300),), fetched_at=_LOCAL_NOW)
    fresh = exchange.get_order_book(_TICKER)

    assert not is_fresh(stale.fetched_at, ttl_seconds=ttl_seconds, now=_LOCAL_NOW)
    assert is_fresh(fresh.fetched_at, ttl_seconds=ttl_seconds, now=_LOCAL_NOW)


def test_unknown_ticker_still_raises_unknown_market_error(
    exchange: LiveBookPaperExchange,
) -> None:
    """The venue's own refusal propagates unchanged.

    Args:
        exchange: The live-book paper session.
    """
    with pytest.raises(UnknownMarketError):
        exchange.get_order_book("KXABSENT-25JAN")


def test_market_status_and_venue_clock_all_come_from_the_venue(
    exchange: LiveBookPaperExchange,
) -> None:
    """Market, status, venue clock, and fee schedule are read live.

    `get_exchange_time` is the load-bearing one: answering it from the local
    clock would make `clock_skew_limit` compare the local clock with itself.

    Args:
        exchange: The live-book paper session.
    """
    assert exchange.get_market(_TICKER) == _MARKET
    assert exchange.get_exchange_status() == ExchangeStatus(
        status="open", fetched_at=_VENUE_STATUS_AT
    )
    assert exchange.get_exchange_time() == _VENUE_CLOCK
    assert exchange.get_fee_model(_TICKER) == _VENUE_FEE_MODEL


def test_markets_expose_exactly_the_bound_ticker(
    exchange: LiveBookPaperExchange,
) -> None:
    """The session's universe is the one bound ticker (#345 owns the rest).

    The PAPER composition root picks its ticker with ``next(iter(markets))``,
    so this mapping is load-bearing, not decorative.

    Args:
        exchange: The live-book paper session.
    """
    assert list(exchange.markets) == [_TICKER]
    assert exchange.list_markets() == (_MARKET,)


# --- paper fills -----------------------------------------------------------------


def test_taker_fill_is_priced_and_dated_off_the_live_book(
    exchange: LiveBookPaperExchange,
) -> None:
    """A crossing order fills at the venue's prices, stamped at the venue's time.

    Eligible ask depth is ``300 + 200 == 500`` centis, the 250_000 ppm
    participation cap floors that to 125, and the 100-centi request is smaller
    still, so the whole 100 comes off the 4500 level.

    Args:
        exchange: The live-book paper session.
    """
    placement = exchange.place_order(
        PaperOrderIntent(
            ticker=_TICKER,
            side="yes",
            price=PricePips(4600),
            quantity=ContractCentis(100),
        ),
        None,
    )

    assert [(fill.price.value, fill.quantity.value) for fill in placement.fills] == [
        (4500, 100)
    ]
    assert [fill.ts for fill in placement.fills] == [_VENUE_BOOK_AT]
    assert placement.resting_order is None


def test_a_moved_venue_book_moves_the_fill_price(
    exchange: LiveBookPaperExchange, venue: _FakeVenue
) -> None:
    """The fill tracks the live book, proving it is not a frozen boot snapshot.

    Args:
        exchange: The live-book paper session.
        venue: The scripted venue behind it.
    """
    venue.book = _book(asks=((4550, 400),))

    placement = exchange.place_order(
        PaperOrderIntent(
            ticker=_TICKER,
            side="yes",
            price=PricePips(4600),
            quantity=ContractCentis(100),
        ),
        None,
    )

    assert [fill.price.value for fill in placement.fills] == [4550]


def test_fills_positions_and_balances_stay_paper(
    exchange: LiveBookPaperExchange,
) -> None:
    """The account is folded from simulated fills, not read from the venue.

    ``100`` centis at ``4500`` pips is ``4_500_000`` micros of book cost plus
    the venue schedule's ``70_000`` ppm worst-case trading fee, all debited
    from the fixture's ``100_000_000``-micro opening balance.

    Args:
        exchange: The live-book paper session.
    """
    opening = exchange.get_balances().total.value
    exchange.place_order(
        PaperOrderIntent(
            ticker=_TICKER,
            side="yes",
            price=PricePips(4600),
            quantity=ContractCentis(100),
        ),
        None,
    )

    positions = exchange.get_positions()
    balances = exchange.get_balances()

    assert opening == 100_000_000
    assert [
        (row.ticker, row.quantity.value, row.average_price.value) for row in positions
    ] == [(_TICKER, 100, 4500)]
    assert balances.total.value < opening
    assert balances.fetched_at == _LOCAL_NOW


def test_a_resting_remainder_never_fills_without_a_venue_trade_tape(
    exchange: LiveBookPaperExchange,
) -> None:
    """No trade prints means no resting fill -- an absence, never an invention.

    A 200-centi request against 500 centis of eligible depth is capped to 125,
    leaving 75 resting. The live read path carries books, not a trade tape, so
    there is no evidence any resting order traded through; ``advance`` has
    nothing to replay and the order stays open rather than being credited with
    a fill nobody observed.

    Args:
        exchange: The live-book paper session.
    """
    placement = exchange.place_order(
        PaperOrderIntent(
            ticker=_TICKER,
            side="yes",
            price=PricePips(4600),
            quantity=ContractCentis(200),
        ),
        None,
    )
    advanced = exchange.advance()

    assert placement.resting_order is not None
    assert placement.resting_order.quantity.value == 75
    assert advanced is False
    assert exchange.get_open_orders() == (placement.resting_order,)
    assert len(exchange.get_fills(datetime(2020, 1, 1, tzinfo=UTC))) == 1


def test_live_session_is_never_re_dated_the_way_a_replay_is(
    exchange: LiveBookPaperExchange, venue: _FakeVenue
) -> None:
    """Live timestamps pass through untouched -- no replay anchor is applied.

    The recorded-session anchor (issue #369) exists because a recording's
    frozen literals are stale forever. Live data has no such problem, and
    shifting it would fabricate freshness the venue never claimed.

    Args:
        exchange: The live-book paper session.
        venue: The scripted venue behind it.
    """
    ancient = datetime(2021, 5, 5, tzinfo=UTC)
    venue.book = _book(asks=((4500, 300),), fetched_at=ancient)
    venue.venue_time = ancient

    assert exchange.get_order_book(_TICKER).fetched_at == ancient
    assert exchange.get_exchange_time() == ancient


def test_a_live_session_is_never_exhausted_the_way_a_replay_is(
    exchange: LiveBookPaperExchange,
) -> None:
    """Running the cursor off the end leaves the venue clock readable (#382).

    A replay stops answering once it has consumed its recording, because a
    recording substantiates the venue's clock only for the span it covers. A
    live session's single step is manufactured fresh from the venue on every
    read and its clock is the venue's own, so there is no recording to run out
    of -- `advance` reporting no further step says nothing about whether the
    venue can state its time.

    Args:
        exchange: The live-book paper session.
    """
    assert exchange.advance() is False

    assert exchange.get_exchange_time() == _VENUE_CLOCK
