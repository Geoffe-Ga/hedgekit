"""The product-type gate must not re-fetch the whole catalog per book read (#302).

``KalshiConnector.get_order_book`` gates on the market's product type before it
will contact ``/orderbook`` (issue #278), but the ``/orderbook`` payload carries
no ``market_type`` of its own -- so the gate has to learn the type from the
``/markets`` catalog. Read literally that meant one *fully paginated* catalog
walk per book read, which PR #408 turned from a latent inefficiency into a hot
path: the PAPER loop now screens a whole universe every tick and needs each
candidate's book, so the per-tick cost was ``(1 + universe) x catalog pages``.

These tests pin the fix at the only level that can prove it -- the count of HTTP
GETs the connector actually issues -- and, just as importantly, pin what the fix
is **not** allowed to buy that saving with:

* no book is ever cached, so every snapshot still carries the instant it was
  genuinely observed and a stale one still vetoes ``quote_freshness``;
* the allowlist gate stays fail-closed on the warm path -- a non-binary product
  and a ticker absent from the catalog are both refused *before* ``/orderbook``
  is contacted, exactly as they are on the cold path.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from tests.riskkernel.conftest import DEFAULT_NOW_EPOCH_S, make_context, make_intent
from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.kalshi.adapter import KalshiConnector
from windbreak.riskkernel.checks import DEFAULT_CHECKS

if TYPE_CHECKING:
    from windbreak.connector.kalshi.client import KalshiClient
    from windbreak.connector.snapshot import InMemoryEventLedgerWriter
    from windbreak.riskkernel.checks import Check

    from .conftest import FakeKalshiSession

#: The instant every advanceable clock in this module starts from.
_START = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

#: A ticker the recorded catalog offers as a binary *and* serves a book for.
_BINARY_TICKER = "KXFED-24DEC"

#: A ticker the recorded catalog offers as a non-binary (perpetual) product.
_PERPETUAL_TICKER = "FAKE-PERP"

#: A ticker no page of the recorded catalog lists at all.
_ABSENT_TICKER = "NOT-LISTED-ANYWHERE"


class _AdvanceableClock:
    """A manually advanced UTC clock, so a TTL is exercised without sleeping."""

    def __init__(self, start: datetime) -> None:
        """Initialize the clock at a fixed instant.

        Args:
            start: The instant the clock reports until it is advanced.
        """
        self._now = start

    def __call__(self) -> datetime:
        """Return the instant the clock currently reports."""
        return self._now

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward.

        Args:
            delta: How far forward to move.
        """
        self._now += delta


def _catalog_fetches(session: FakeKalshiSession) -> int:
    """Count the full ``/markets`` catalog pages the session was asked for.

    The per-market book route is ``/markets/{ticker}/orderbook``, so matching on
    a trailing ``/markets`` counts catalog walks only.

    Args:
        session: The recording fake session.

    Returns:
        How many catalog page requests were issued.
    """
    return sum(1 for call in session.calls if call["url"].endswith("/markets"))


def _orderbook_fetches(session: FakeKalshiSession) -> int:
    """Count the ``/orderbook`` requests the session was asked for.

    Args:
        session: The recording fake session.

    Returns:
        How many book requests were issued.
    """
    return sum(1 for call in session.calls if call["url"].endswith("/orderbook"))


def _check_named(name: str) -> Check:
    """Return the production risk check registered under ``name``.

    Args:
        name: The check's ``name`` attribute.

    Returns:
        The check callable from :data:`DEFAULT_CHECKS`.
    """
    return next(check for check in DEFAULT_CHECKS if check.name == name)


def test_repeated_book_reads_fetch_the_market_catalog_once(
    fake_kalshi_client: KalshiClient,
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
) -> None:
    """Five book reads within the TTL cost one catalog walk, not five.

    This is the issue's acceptance criterion stated as a number: the catalog is
    read for the product-type gate, and the product type of a given ticker is
    immutable venue reference data, so re-walking every page per book read buys
    nothing and scales the tick quadratically in universe size.
    """
    connector = KalshiConnector(
        fake_kalshi_client, ledger, clock=_AdvanceableClock(_START)
    )

    for _ in range(5):
        connector.get_order_book(_BINARY_TICKER)

    assert _catalog_fetches(fake_kalshi_session) == 1
    assert _orderbook_fetches(fake_kalshi_session) == 5


def test_a_screen_shaped_tick_fetches_one_catalog_for_the_whole_universe(
    fake_kalshi_client: KalshiClient,
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
) -> None:
    """``list_markets`` then a book per market costs exactly one catalog walk.

    This is the shape ``windbreak.scheduler.screening.screen_universe`` drives
    every tick. The catalog ``list_markets`` already walked is the same catalog
    the gate needs, so reusing it -- rather than re-walking it per candidate --
    is what collapses ``1 + universe`` walks into one.
    """
    connector = KalshiConnector(
        fake_kalshi_client, ledger, clock=_AdvanceableClock(_START)
    )

    for market in sorted(connector.list_markets(), key=lambda entry: entry.ticker):
        # Only one recorded market has a book fixture; the rest refuse or 404,
        # which is beside the point being measured -- the catalog walk count.
        with suppress(UnknownMarketError):
            connector.get_order_book(market.ticker)

    assert _catalog_fetches(fake_kalshi_session) == 1


def test_the_gate_is_refetched_once_its_ttl_has_elapsed(
    fake_kalshi_client: KalshiClient,
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
) -> None:
    """A gate older than its TTL is re-read, so a newly listed market is reachable.

    The saving is bounded rather than permanent: without an expiry a long-lived
    connector that only ever reads books would gate forever against the catalog
    it saw at startup, and a market listed afterwards could never be traded.
    """
    clock = _AdvanceableClock(_START)
    connector = KalshiConnector(
        fake_kalshi_client, ledger, clock=clock, gate_ttl=timedelta(seconds=30)
    )

    connector.get_order_book(_BINARY_TICKER)
    clock.advance(timedelta(seconds=29))
    connector.get_order_book(_BINARY_TICKER)
    assert _catalog_fetches(fake_kalshi_session) == 1

    clock.advance(timedelta(seconds=2))
    connector.get_order_book(_BINARY_TICKER)
    assert _catalog_fetches(fake_kalshi_session) == 2


def test_a_warm_gate_still_refuses_a_non_binary_product_before_the_book_route(
    fake_kalshi_client: KalshiClient,
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
) -> None:
    """The cached gate refuses a perpetual without ever contacting ``/orderbook``.

    The cold-path guarantee is pinned by
    ``test_get_order_book_refuses_non_binary_products_before_fetching_the_book``;
    this is the same guarantee on the path the cache introduced, because a cache
    that only fails closed while cold is not fail-closed.
    """
    connector = KalshiConnector(
        fake_kalshi_client, ledger, clock=_AdvanceableClock(_START)
    )
    connector.get_order_book(_BINARY_TICKER)
    before = _orderbook_fetches(fake_kalshi_session)

    with pytest.raises(UnknownMarketError):
        connector.get_order_book(_PERPETUAL_TICKER)

    assert _catalog_fetches(fake_kalshi_session) == 1
    assert _orderbook_fetches(fake_kalshi_session) == before


def test_a_warm_gate_still_refuses_a_ticker_the_catalog_never_listed(
    fake_kalshi_client: KalshiClient,
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
) -> None:
    """An unlisted ticker is refused from the cached gate, book route untouched.

    Absence from the catalog is refused rather than re-checked against the venue:
    refusing is the fail-closed answer, and the TTL bounds how long a genuinely
    new listing stays unreachable.
    """
    connector = KalshiConnector(
        fake_kalshi_client, ledger, clock=_AdvanceableClock(_START)
    )
    connector.get_order_book(_BINARY_TICKER)
    before = _orderbook_fetches(fake_kalshi_session)

    with pytest.raises(UnknownMarketError):
        connector.get_order_book(_ABSENT_TICKER)

    assert _catalog_fetches(fake_kalshi_session) == 1
    assert _orderbook_fetches(fake_kalshi_session) == before


def test_a_warm_gate_never_restamps_the_book_observation_instant(
    fake_kalshi_client: KalshiClient,
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
) -> None:
    """Every book still carries the instant *it* was fetched, never the gate's.

    This is the regression the cache must not introduce. ``quote_freshness`` and
    ``clock_skew_limit`` compare a quote's observed instant against now, so they
    can only veto while that instant is the truth. Caching the *gate* leaves the
    book untouched: each read still contacts ``/orderbook`` and is stamped at its
    own fetch instant, five seconds apart here, on a path where the catalog was
    demonstrably served from cache.
    """
    clock = _AdvanceableClock(_START)
    connector = KalshiConnector(fake_kalshi_client, ledger, clock=clock)

    first = connector.get_order_book(_BINARY_TICKER)
    clock.advance(timedelta(seconds=5))
    second = connector.get_order_book(_BINARY_TICKER)

    assert _catalog_fetches(fake_kalshi_session) == 1, "the gate must be warm here"
    assert first.fetched_at == _START
    assert second.fetched_at == _START + timedelta(seconds=5)


def test_a_stale_book_read_through_a_warm_gate_still_vetoes_quote_freshness(
    fake_kalshi_client: KalshiClient,
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
) -> None:
    """The kernel can still tell a stale book from a fresh one behind a warm gate.

    Both books below are read through the *same* cached gate -- pinned by the
    single catalog fetch -- yet ``quote_freshness`` vetoes the one observed an
    hour and a second before the evaluation instant and approves the one
    observed at it. That is the whole safety argument for caching the gate
    rather than the book: nothing the cache holds is ever compared against now,
    so the staleness veto keeps its teeth. The ``int(...timestamp())``
    conversion mirrors ``windbreak.scheduler.loop``'s own
    ``quote_snapshot_epoch_s=int(order_book.fetched_at.timestamp())``.
    """
    ttl_seconds = 3_600
    stale_instant = datetime.fromtimestamp(DEFAULT_NOW_EPOCH_S - ttl_seconds - 1, UTC)
    clock = _AdvanceableClock(stale_instant)
    connector = KalshiConnector(
        fake_kalshi_client, ledger, clock=clock, gate_ttl=timedelta(hours=2)
    )

    stale_book = connector.get_order_book(_BINARY_TICKER)
    clock.advance(timedelta(seconds=ttl_seconds + 1))
    fresh_book = connector.get_order_book(_BINARY_TICKER)

    assert _catalog_fetches(fake_kalshi_session) == 1, "both reads share one gate"
    quote_freshness = _check_named("quote_freshness")
    stale_result = quote_freshness(
        make_intent(),
        make_context(
            quote_snapshot_epoch_s=int(stale_book.fetched_at.timestamp()),
            now_epoch_s=DEFAULT_NOW_EPOCH_S,
            quote_ttl_seconds=ttl_seconds,
        ),
    )
    fresh_result = quote_freshness(
        make_intent(),
        make_context(
            quote_snapshot_epoch_s=int(fresh_book.fetched_at.timestamp()),
            now_epoch_s=DEFAULT_NOW_EPOCH_S,
            quote_ttl_seconds=ttl_seconds,
        ),
    )

    assert stale_result.vetoed is True
    assert stale_result.reason == "quote is stale or missing"
    assert fresh_result.vetoed is False
