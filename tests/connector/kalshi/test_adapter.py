"""Tests for windbreak.connector.kalshi.adapter (issues #17, #18): KalshiConnector.

Acceptance test: `list_markets()` excludes every refused (non-binary)
product and ledgers a `PRODUCT_REFUSED` event for each one; `get_market`,
`get_order_book`, `get_exchange_status`, and `get_exchange_time` are backed
by the recorded fixtures; `get_fee_model` and `get_balance_semantics` are
backed by the recorded series/semantics fixtures (issue #18); every remaining
trading/account method still raises `NotImplementedError` (order path is M4;
balances/positions/fills are issue #3).

`list_markets` also follows Kalshi's `cursor` pagination across every page of
`/markets` and `/events` (bounded by a hard max-page cap that raises
`KalshiPaginationError` rather than looping forever), and fails closed on a
single malformed binary by ledgering a `MARKET_MALFORMED` event and continuing
the scan instead of letting one bad payload abort the whole call.

Issue #106: a binary missing `volume_24h` degrades through the same
`MARKET_MALFORMED` path once `normalize_market` requires it.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from windbreak.connector.fees import FeeModel, UnknownFeeModelError
from windbreak.connector.interface import MarketConnector, UnknownMarketError
from windbreak.connector.kalshi import PRODUCT_REFUSED_EVENT, KalshiConnector
from windbreak.connector.kalshi.adapter import (
    KALSHI_BALANCE_SEMANTICS,
    MARKET_MALFORMED_EVENT,
    KalshiPaginationError,
    _fee_model_from_series,
)
from windbreak.connector.kalshi.client import KalshiClient
from windbreak.connector.kalshi.normalize import normalize_exchange_status
from windbreak.net.allowlist import OutboundAllowlist

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from windbreak.connector.snapshot import ConnectorEvent, InMemoryEventLedgerWriter

#: (method name, positional args) for every SPEC S7.2 method this issue does
#: not implement -- order path is M4; balances/positions/fills are issue #3.
_UNIMPLEMENTED_CALLS: tuple[tuple[str, tuple[object, ...]], ...] = (
    ("place_order", (object(), object())),
    ("cancel_order", ("some-order-id",)),
    ("get_balances", ()),
    ("get_positions", ()),
    ("get_open_orders", ()),
    ("get_fills", (datetime(2024, 1, 1, tzinfo=UTC),)),
)

#: The fee model `series_KXFED.json` normalizes to (issue #18): a 7% taker
#: rate, no maker fee, no settlement fee.
_EXPECTED_KXFED_FEE_MODEL = FeeModel(
    schedule_id="kxfed-standard-v1",
    maker_fee_ppm=0,
    taker_fee_ppm=70_000,
    settlement_fee_ppm=0,
)


class _RaisingLedgerWriter:
    """An `EventLedgerWriter` that always raises, simulating a broken ledger."""

    def record(self, event: ConnectorEvent) -> None:
        """Raise unconditionally.

        Args:
            event: The event that would have been recorded.
        """
        raise RuntimeError("ledger unavailable")


# --- list_markets: product gate + PRODUCT_REFUSED ledgering -----------------


def test_list_markets_excludes_refused_products_and_returns_only_binaries(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """Perpetual and scalar fixture products never appear in the result."""
    markets = kalshi_fixture_connector.list_markets()

    tickers = [market.ticker for market in markets]
    assert "FAKE-PERP" not in tickers
    assert "KXSCALAR-24DEC" not in tickers
    assert "KXFED-24DEC" in tickers


def test_list_markets_ledgers_a_product_refused_event_per_refused_market(
    kalshi_fixture_connector: KalshiConnector, ledger: InMemoryEventLedgerWriter
) -> None:
    """Every refused market is ledgered exactly once, with reason and hash."""
    kalshi_fixture_connector.list_markets()

    refusals = ledger.events_by_type(PRODUCT_REFUSED_EVENT)
    refused_tickers = {event.payload["ticker"] for event in refusals}

    assert refused_tickers == {"FAKE-PERP", "KXSCALAR-24DEC"}
    for event in refusals:
        assert event.payload["reason"]
        assert event.payload["raw_exchange_payload_hash"]
        assert "event_ticker" in event.payload


def test_list_markets_wires_mutually_exclusive_group_id_from_events_fixture(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """The event fixture's `mutually_exclusive` flag drives grouping end to end."""
    markets = {
        market.ticker: market for market in kalshi_fixture_connector.list_markets()
    }

    assert markets["KXFED-24DEC"].mutually_exclusive_group_id == "KXFED"
    assert markets["KXFED-24DEC-B75"].mutually_exclusive_group_id == "KXFED"
    assert markets["KXWEA-24DEC"].mutually_exclusive_group_id is None


def test_writer_raising_does_not_propagate_out_of_list_markets(
    fake_kalshi_client: KalshiClient,
    clock: Callable[[], datetime],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken ledger writer must not crash `list_markets` (isolation)."""
    caplog.set_level(logging.DEBUG)
    connector = KalshiConnector(fake_kalshi_client, _RaisingLedgerWriter(), clock=clock)

    markets = connector.list_markets()

    assert any(market.ticker == "KXFED-24DEC" for market in markets)
    assert any("ledger" in record.getMessage().lower() for record in caplog.records)


# --- get_market --------------------------------------------------------


@pytest.mark.parametrize("ticker", ["FAKE-PERP", "NOPE"])
def test_get_market_raises_unknown_for_refused_or_absent_ticker(
    kalshi_fixture_connector: KalshiConnector, ticker: str
) -> None:
    """A refused product and an absent ticker both raise `UnknownMarketError`."""
    with pytest.raises(UnknownMarketError):
        kalshi_fixture_connector.get_market(ticker)


def test_get_market_returns_the_normalized_binary_for_a_known_ticker(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """A known binary ticker returns its normalized market."""
    market = kalshi_fixture_connector.get_market("KXFED-24DEC")

    assert market.ticker == "KXFED-24DEC"
    assert market.market_type == "fully_collateralized_binary"


@pytest.mark.parametrize("ticker", ["KXFED-24DEC", "FAKE-PERP", "NOPE"])
def test_get_market_never_ledgers_product_refused_events(
    kalshi_fixture_connector: KalshiConnector,
    ledger: InMemoryEventLedgerWriter,
    ticker: str,
) -> None:
    """A single-ticker lookup must not ledger refusals for the whole venue.

    Only ``list_markets`` scans and refuses; ``get_market`` is a targeted
    lookup, so it emits no ``PRODUCT_REFUSED`` events regardless of whether the
    ticker is a binary, a refused product, or absent. This keeps the SPEC S7.1
    "one refusal per refused product" contract from being violated by repeated
    lookups flooding the ledger with duplicates.
    """
    with suppress(UnknownMarketError):
        kalshi_fixture_connector.get_market(ticker)

    assert ledger.events_by_type(PRODUCT_REFUSED_EVENT) == ()


# --- get_order_book ------------------------------------------------------


def test_get_order_book_raises_unknown_for_absent_ticker(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """An unrecorded ticker raises `UnknownMarketError`, not a raw API error."""
    with pytest.raises(UnknownMarketError):
        kalshi_fixture_connector.get_order_book("NOPE")


def test_get_order_book_fetched_at_equals_the_injected_clock(
    kalshi_fixture_connector: KalshiConnector, clock: Callable[[], datetime]
) -> None:
    """`fetched_at` comes from the injected clock, not wall-clock time."""
    book = kalshi_fixture_connector.get_order_book("KXFED-24DEC")

    assert book.fetched_at == clock()


@pytest.mark.parametrize("ticker", ["FAKE-PERP", "KXSCALAR-24DEC"])
def test_get_order_book_refuses_non_binary_products_before_fetching_the_book(
    kalshi_fixture_connector: KalshiConnector,
    fake_kalshi_session: Any,
    ticker: str,
) -> None:
    """A perpetual/scalar ticker is refused before the venue is ever asked.

    SPEC S7.1 rejects non-binary product surfaces unconditionally: a perpetual
    or scalar ticker must never reach the exchange's `/orderbook` route at
    all, mirroring the `gate_product` check `get_market` already applies.
    Asserting only `UnknownMarketError` here would be a false green -- today
    that error is already raised, but only because the fake session's
    orderbook route 404s for these tickers, not because the product gate
    fired first. The `.calls` assertion is what proves the gate runs before
    the fetch.
    """
    with pytest.raises(UnknownMarketError):
        kalshi_fixture_connector.get_order_book(ticker)

    orderbook_calls = [
        call for call in fake_kalshi_session.calls if call["url"].endswith("/orderbook")
    ]
    assert orderbook_calls == []


@pytest.mark.parametrize(
    "ticker", ["KXFED-24DEC", "FAKE-PERP", "KXSCALAR-24DEC", "NOPE"]
)
def test_get_order_book_never_ledgers_product_refused_events(
    kalshi_fixture_connector: KalshiConnector,
    ledger: InMemoryEventLedgerWriter,
    ticker: str,
) -> None:
    """A single-ticker order-book lookup must not ledger refusals for the venue.

    Like ``get_market``, ``get_order_book`` is a targeted lookup rather than a
    venue-wide scan, so it must never emit ``PRODUCT_REFUSED`` events -- doing
    so would let repeated single-ticker lookups flood the ledger with
    duplicates that only ``list_markets`` is meant to produce (SPEC S7.1 "one
    refusal per refused product").
    """
    with suppress(UnknownMarketError):
        kalshi_fixture_connector.get_order_book(ticker)

    assert ledger.events_by_type(PRODUCT_REFUSED_EVENT) == ()


def test_get_order_book_raises_unknown_when_allowed_binary_has_no_book(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """An allowed binary with no recorded book still raises `UnknownMarketError`.

    `KXWEA-24DEC` is an allowed binary in `markets.json` with no
    `orderbook_KXWEA-24DEC.json` fixture, so the fake session 404s its
    `/orderbook` route. This pins the `KalshiApiError` -> `UnknownMarketError`
    translation for a ticker that clears the product gate but still has no
    book at the venue, once the gate lets allowed binaries through to the
    fetch.
    """
    with pytest.raises(UnknownMarketError):
        kalshi_fixture_connector.get_order_book("KXWEA-24DEC")


# --- get_exchange_status -------------------------------------------------


@pytest.mark.parametrize(
    ("exchange_active", "trading_active", "expected_status"),
    [
        (True, True, "open"),
        (True, False, "paused"),
        (False, True, "closed"),
        (False, False, "closed"),
    ],
)
def test_exchange_status_active_flags_map_to_the_documented_status(
    exchange_active: bool, trading_active: bool, expected_status: str
) -> None:
    """Active-flag combinations map to open/paused/closed exactly as documented."""
    raw = {"exchange_active": exchange_active, "trading_active": trading_active}

    status = normalize_exchange_status(raw, datetime(2024, 1, 1, tzinfo=UTC))

    assert status.status == expected_status


def test_get_exchange_status_fetched_at_equals_the_injected_clock(
    kalshi_fixture_connector: KalshiConnector, clock: Callable[[], datetime]
) -> None:
    """The fixture's active flags map to `"open"`; `fetched_at` is the clock."""
    status = kalshi_fixture_connector.get_exchange_status()

    assert status.fetched_at == clock()
    assert status.status == "open"


# --- get_exchange_time ---------------------------------------------------


def test_get_exchange_time_returns_the_date_header_when_present(
    kalshi_fixture_connector: KalshiConnector, kalshi_fixture_server_date: datetime
) -> None:
    """The Date response header, parsed and UTC-normalized, wins over the clock."""
    server_time = kalshi_fixture_connector.get_exchange_time()

    assert server_time == kalshi_fixture_server_date


def test_get_exchange_time_falls_back_to_clock_when_date_header_absent(
    kalshi_connector_missing_date_header: KalshiConnector,
    clock: Callable[[], datetime],
) -> None:
    """A missing `Date` header falls back to the injected clock."""
    server_time = kalshi_connector_missing_date_header.get_exchange_time()

    assert server_time == clock()


# --- get_fee_model (issue #18) ---------------------------------------------


@pytest.mark.parametrize("market_or_series", ["KXFED-24DEC", "KXFED"])
def test_get_fee_model_resolves_the_series_ticker_from_a_market_or_series(
    kalshi_fixture_connector: KalshiConnector, market_or_series: str
) -> None:
    """A market ticker and its bare series ticker return the identical fee model."""
    fee_model = kalshi_fixture_connector.get_fee_model(market_or_series)

    assert fee_model == _EXPECTED_KXFED_FEE_MODEL


def test_get_fee_model_raises_unknown_fee_model_for_an_unrecognized_series(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """A series the venue does not recognize (a 404) fails closed."""
    with pytest.raises(UnknownFeeModelError):
        kalshi_fixture_connector.get_fee_model("NOPE-24DEC")


def test_get_fee_model_raises_unknown_fee_model_for_a_malformed_fee_type(
    kalshi_malformed_fee_connector: KalshiConnector,
) -> None:
    """An unrecognized `fee_type` fails closed rather than misreading the schedule."""
    with pytest.raises(UnknownFeeModelError):
        kalshi_malformed_fee_connector.get_fee_model("KXBAD-24DEC")


@pytest.mark.parametrize("bad_schedule_id", ["", 123])
def test_fee_model_from_series_rejects_a_non_string_schedule_id(
    bad_schedule_id: object,
) -> None:
    """An empty or non-str `fee_schedule_id` fails closed as UnknownFeeModelError.

    The schedule identifier is the fee model's provenance handle; a blank or
    wrong-typed value would silently degrade that provenance, so the parser
    rejects it with the same fail-closed error as any other malformed leaf
    rather than surfacing a bare ``ValueError`` from ``FeeModel``.
    """
    payload = {
        "series": {
            "fee_type": "quadratic",
            "fee_schedule_id": bad_schedule_id,
            "maker_fee_bps": 0,
            "taker_fee_bps": 700,
            "settlement_fee_bps": 0,
        }
    }

    with pytest.raises(UnknownFeeModelError):
        _fee_model_from_series(payload)


@pytest.mark.parametrize(
    "bad_leaf", ["maker_fee_bps", "taker_fee_bps", "settlement_fee_bps"]
)
def test_fee_model_from_series_rejects_a_negative_fee_leaf(bad_leaf: str) -> None:
    """A negative fee leaf fails closed as UnknownFeeModelError, not a raw ValueError.

    A ``*_fee_bps`` leaf of the right type but a negative value is still a
    schedule this adapter cannot faithfully model; it must be refused with the
    same fail-closed error as any other malformed leaf rather than surfacing the
    bare ``ValueError`` ``FeeModel`` raises for a negative ppm rate -- misreading
    a fee schedule is worse than admitting ignorance.
    """
    payload = {
        "series": {
            "fee_type": "quadratic",
            "fee_schedule_id": "KXFED-STD",
            "maker_fee_bps": 0,
            "taker_fee_bps": 700,
            "settlement_fee_bps": 0,
            bad_leaf: -1,
        }
    }

    with pytest.raises(UnknownFeeModelError):
        _fee_model_from_series(payload)


# --- get_balance_semantics (issue #18) --------------------------------------


def test_get_balance_semantics_returns_the_kalshi_module_constant(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """`get_balance_semantics` returns the shared `KALSHI_BALANCE_SEMANTICS` record."""
    assert kalshi_fixture_connector.get_balance_semantics() is KALSHI_BALANCE_SEMANTICS


# --- unimplemented surface + protocol conformance -------------------------


@pytest.mark.parametrize(("method_name", "args"), _UNIMPLEMENTED_CALLS)
def test_unimplemented_trading_and_account_methods_raise_not_implemented(
    kalshi_fixture_connector: KalshiConnector,
    method_name: str,
    args: tuple[object, ...],
) -> None:
    """Every trading/account method not yet wired raises `NotImplementedError`."""
    method = getattr(kalshi_fixture_connector, method_name)

    with pytest.raises(NotImplementedError):
        method(*args)


def test_kalshi_connector_satisfies_the_market_connector_protocol(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """`KalshiConnector` structurally satisfies the runtime-checkable protocol."""
    assert isinstance(kalshi_fixture_connector, MarketConnector)


# --- pagination: /markets and /events cursor-following ---------------------


def _binary_market(ticker: str, event_ticker: str, **overrides: Any) -> dict[str, Any]:
    """Build a minimal, fully-populated raw binary market payload.

    Args:
        ticker: The market ticker.
        event_ticker: The parent event ticker.
        **overrides: Fields to override or drop (a value of the sentinel below
            is not used; callers pass explicit keys to override defaults).

    Returns:
        A raw market mapping shaped like a `/markets` list entry.
    """
    market: dict[str, Any] = {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "market_type": "binary",
        "title": f"{ticker} title",
        "rules_primary": f"{ticker} rules",
        "category": "Test",
        "close_time": "2024-12-18T19:00:00Z",
        "expected_expiration_time": None,
        "tick_size": 1,
        "volume_24h": 1000,
    }
    market.update(overrides)
    return market


class _PagedResponse:
    """A minimal fake response carrying a scripted JSON page (no `Date`)."""

    def __init__(self, status_code: int, payload: Any) -> None:
        """Store the status code and JSON payload.

        Args:
            status_code: The HTTP status code to report.
            payload: The value `.json()` returns.
        """
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        """Return the scripted JSON page."""
        return self._payload


class _PaginatedSession:
    """Serve multi-page `/markets` and `/events` keyed on the `cursor` param.

    Each route maps to an ordered list of page payloads. A request with no
    `cursor` returns page 0; a request whose `cursor` param is the string index
    ``"n"`` returns page ``n``. Each page's own ``cursor`` field points to the
    next page index (or ``""`` on the last page), exactly as Kalshi paginates.
    """

    def __init__(self, pages_by_suffix: Mapping[str, list[dict[str, Any]]]) -> None:
        """Store the per-route page lists.

        Args:
            pages_by_suffix: Maps a URL suffix (``"/markets"`` / ``"/events"``)
                to its ordered list of page payloads.
        """
        self._pages = pages_by_suffix
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> _PagedResponse:
        """Return the page addressed by `url`'s route and the `cursor` param.

        Args:
            url: The request URL built by `KalshiClient`.
            params: Query parameters; the ``cursor`` key selects the page.
            timeout: The forwarded request timeout (recorded, not used).

        Returns:
            The scripted page response, or a 404 for an unknown route.
        """
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if url.endswith("/exchange/status"):
            # ``list_markets`` now consults exchange status first (issue #20's
            # maintenance check); serve an open venue so pagination proceeds.
            return _PagedResponse(
                200, {"exchange_active": True, "trading_active": True}
            )
        cursor = (params or {}).get("cursor")
        index = int(cursor) if cursor else 0
        for suffix, pages in self._pages.items():
            if url.endswith(suffix):
                return _PagedResponse(200, pages[index])
        return _PagedResponse(404, {"error": "not found"})


def _connector_over(
    session: _PaginatedSession,
    ledger: InMemoryEventLedgerWriter,
    clock: Callable[[], datetime],
) -> KalshiConnector:
    """Wire a `KalshiConnector` over a paginated fake session.

    Resilience is disabled (`resilience=None`): these adapter tests exercise
    pagination/normalization logic, not rate limiting, and one drives the
    1000-page safety cap -- the on-by-default token bucket would otherwise
    throttle that runaway walk with real sleeps. Resilience wiring is covered
    end-to-end in `test_client_resilience.py`.
    """
    client = KalshiClient(
        base_url="https://fake.test",
        timeout=5,
        session=session,
        resilience=None,
        allowlist=OutboundAllowlist(frozenset({"fake.test"})),
    )
    return KalshiConnector(client, ledger, clock=clock)


def test_list_markets_aggregates_every_market_page_via_cursor(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """Markets split across three `/markets` pages all appear in the result."""
    session = _PaginatedSession(
        {
            "/markets": [
                {"markets": [_binary_market("KX-A", "E1")], "cursor": "1"},
                {"markets": [_binary_market("KX-B", "E1")], "cursor": "2"},
                {"markets": [_binary_market("KX-C", "E1")], "cursor": ""},
            ],
            "/events": [{"events": [], "cursor": ""}],
        }
    )
    connector = _connector_over(session, ledger, clock)

    tickers = {market.ticker for market in connector.list_markets()}

    assert tickers == {"KX-A", "KX-B", "KX-C"}


def test_list_markets_aggregates_every_event_page_via_cursor(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """A grouping event living on the second `/events` page is still applied."""
    session = _PaginatedSession(
        {
            "/markets": [
                {"markets": [_binary_market("KX-A", "E2")], "cursor": ""},
            ],
            "/events": [
                {
                    "events": [{"event_ticker": "E1", "mutually_exclusive": True}],
                    "cursor": "1",
                },
                {
                    "events": [{"event_ticker": "E2", "mutually_exclusive": True}],
                    "cursor": "",
                },
            ],
        }
    )
    connector = _connector_over(session, ledger, clock)

    (market,) = connector.list_markets()

    assert market.mutually_exclusive_group_id == "E2"


def test_list_markets_raises_when_market_pagination_exceeds_the_cap(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """A never-terminating `/markets` cursor raises rather than looping forever."""

    class _NeverEndingSession:
        """A `/markets` route whose cursor never empties (a runaway venue)."""

        def get(
            self,
            url: str,
            *,
            params: Mapping[str, str] | None = None,
            timeout: int | None = None,
        ) -> _PagedResponse:
            """Serve an open status, empty `/events`, and a never-ending `/markets`."""
            if url.endswith("/exchange/status"):
                return _PagedResponse(
                    200, {"exchange_active": True, "trading_active": True}
                )
            if url.endswith("/events"):
                return _PagedResponse(200, {"events": [], "cursor": ""})
            return _PagedResponse(200, {"markets": [], "cursor": "more"})

    connector = _connector_over(
        _NeverEndingSession(),  # type: ignore[arg-type]
        ledger,
        clock,
    )

    with pytest.raises(KalshiPaginationError):
        connector.list_markets()


# --- fail-closed: one malformed binary must not abort the scan ------------


def test_list_markets_ledgers_malformed_binary_and_continues(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """A binary missing a required field is ledgered, not crash-propagated."""
    broken = _binary_market("KX-BAD", "E1")
    del broken["title"]
    session = _PaginatedSession(
        {
            "/markets": [
                {"markets": [broken, _binary_market("KX-GOOD", "E1")], "cursor": ""},
            ],
            "/events": [{"events": [], "cursor": ""}],
        }
    )
    connector = _connector_over(session, ledger, clock)

    tickers = {market.ticker for market in connector.list_markets()}
    malformed = ledger.events_by_type(MARKET_MALFORMED_EVENT)

    assert tickers == {"KX-GOOD"}
    assert {event.payload["ticker"] for event in malformed} == {"KX-BAD"}
    assert all(event.payload["raw_exchange_payload_hash"] for event in malformed)


def test_list_markets_ledgers_malformed_binary_with_bad_numeric_type(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """A non-integer `tick_size` on a binary fails closed to a ledgered event."""
    broken = _binary_market("KX-BAD", "E1", tick_size="not-a-number")
    session = _PaginatedSession(
        {
            "/markets": [{"markets": [broken], "cursor": ""}],
            "/events": [{"events": [], "cursor": ""}],
        }
    )
    connector = _connector_over(session, ledger, clock)

    assert connector.list_markets() == ()
    assert {
        event.payload["ticker"]
        for event in ledger.events_by_type(MARKET_MALFORMED_EVENT)
    } == {"KX-BAD"}


def test_list_markets_ledgers_malformed_binary_when_volume_24h_is_missing(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """A binary missing `volume_24h` is ledgered as MARKET_MALFORMED (issue #106).

    Mirrors `test_list_markets_ledgers_malformed_binary_and_continues`: a
    required field missing from an otherwise-allowed binary degrades to a
    ledgered `MARKET_MALFORMED` event and is skipped, never crash-propagated
    and never silently dropped.
    """
    broken = _binary_market("KX-BAD", "E1")
    del broken["volume_24h"]
    session = _PaginatedSession(
        {
            "/markets": [
                {"markets": [broken, _binary_market("KX-GOOD", "E1")], "cursor": ""},
            ],
            "/events": [{"events": [], "cursor": ""}],
        }
    )
    connector = _connector_over(session, ledger, clock)

    tickers = {market.ticker for market in connector.list_markets()}
    malformed = ledger.events_by_type(MARKET_MALFORMED_EVENT)

    assert tickers == {"KX-GOOD"}
    assert {event.payload["ticker"] for event in malformed} == {"KX-BAD"}
    assert all(event.payload["raw_exchange_payload_hash"] for event in malformed)


def test_get_market_ledgers_malformed_matching_binary_and_raises_unknown(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """A matching-but-malformed binary fails closed like `list_markets`.

    ``get_market`` finds the ticker but cannot normalize its payload; it must
    ledger a ``MARKET_MALFORMED`` event and raise ``UnknownMarketError`` rather
    than let the normalization error propagate uncaught.
    """
    broken = _binary_market("KX-BAD", "E1")
    del broken["title"]
    session = _PaginatedSession(
        {
            "/markets": [{"markets": [broken], "cursor": ""}],
            "/events": [{"events": [], "cursor": ""}],
        }
    )
    connector = _connector_over(session, ledger, clock)

    with pytest.raises(UnknownMarketError):
        connector.get_market("KX-BAD")

    malformed = ledger.events_by_type(MARKET_MALFORMED_EVENT)
    assert {event.payload["ticker"] for event in malformed} == {"KX-BAD"}
    assert all(event.payload["raw_exchange_payload_hash"] for event in malformed)
