"""Tests for windbreak.connector.fake.FakeExchange (issue #16; semantics: #18).

`FakeExchange` implements the full `MarketConnector` protocol from
`tests/fixtures/exchange/*.json`, wrapping every arithmetic-bearing value
(order-book prices/quantities, balances) in windbreak's scaled-integer unit
types at load time. `windbreak/connector/` does not exist yet, so importing it
fails collection with `ModuleNotFoundError: No module named
'windbreak.connector'` -- the expected Gate 1 RED state for issue #16. Once
that lands, this module additionally imports `windbreak.connector.semantics`
(issue #18), which does not exist yet either -- so this file stays RED via
`ModuleNotFoundError: No module named 'windbreak.connector.semantics'` until
both issues are implemented.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from windbreak.connector.fake import _parse_dt
from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.models import NormalizedMarket
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
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.connector.fake import FakeExchange

_ALL_FIXTURE_FILES = [
    "markets.json",
    "order_books.json",
    "exchange.json",
    "balances.json",
    "positions.json",
    "open_orders.json",
    "fills.json",
    "fee_models.json",
    "balance_semantics.json",
]


# --- The issue's worked example: KXFED-24DEC ------------------------------------


def test_kxfed_market_matches_the_issue_example(fake_exchange: FakeExchange) -> None:
    """KXFED-24DEC exercises every SPEC S6.2 field on a jurisdiction-eligible market."""
    market = fake_exchange.get_market("KXFED-24DEC")

    assert market.market_type == "fully_collateralized_binary"
    assert market.jurisdiction_status in ("eligible", "ineligible", "unknown")
    assert market.jurisdiction_status == "eligible"
    assert isinstance(market.price_tick_pips, int)
    assert isinstance(market.min_order_contract_centis, int)
    assert market.mutually_exclusive_group_id == "KXFED-24-GROUP"
    assert market.raw_exchange_payload_hash != ""


def test_get_market_unknown_ticker_raises_unknown_market_error(
    fake_exchange: FakeExchange,
) -> None:
    """An unrecognized ticker raises `UnknownMarketError`, not a bare `KeyError`."""
    with pytest.raises(UnknownMarketError):
        fake_exchange.get_market("NOT-A-REAL-TICKER")


def test_get_order_book_unknown_ticker_raises_unknown_market_error(
    fake_exchange: FakeExchange,
) -> None:
    """An unrecognized ticker raises `UnknownMarketError` from `get_order_book`."""
    with pytest.raises(UnknownMarketError):
        fake_exchange.get_order_book("NOT-A-REAL-TICKER")


def test_list_markets_returns_all_three_fixture_markets(
    fake_exchange: FakeExchange,
) -> None:
    """`list_markets` returns exactly the three tickers the fixtures define."""
    tickers = {market.ticker for market in fake_exchange.list_markets()}

    assert tickers == {"KXFED-24DEC", "KXBAN-24DEC", "KXWEA-24DEC"}


# --- Order book: prices and quantities are unit-wrapped -------------------------


def test_order_book_levels_are_wrapped_in_the_scaled_integer_unit_types(
    fake_exchange: FakeExchange,
) -> None:
    """Book prices are `PricePips` and quantities are `ContractCentis`."""
    book = fake_exchange.get_order_book("KXFED-24DEC")

    assert book.ticker == "KXFED-24DEC"
    assert book.yes_bids[0].price == PricePips(4500)
    assert book.yes_bids[0].quantity == ContractCentis(500)
    assert book.yes_asks[0].price == PricePips(4600)
    assert book.yes_asks[0].quantity == ContractCentis(300)
    assert isinstance(book.yes_bids[0].price, PricePips)
    assert isinstance(book.yes_bids[0].quantity, ContractCentis)


def test_order_book_for_market_with_no_liquidity_is_empty(
    fake_exchange: FakeExchange,
) -> None:
    """A market with no resting orders yields empty bid and ask tuples."""
    book = fake_exchange.get_order_book("KXBAN-24DEC")

    assert book.yes_bids == ()
    assert book.yes_asks == ()


# --- Fills: since-filtering and unit wrapping ------------------------------------


def test_get_fills_filters_strictly_after_since(fake_exchange: FakeExchange) -> None:
    """`get_fills` returns only the fills timestamped strictly after `since`."""
    since = datetime(2024, 11, 10, tzinfo=UTC)

    fills = fake_exchange.get_fills(since)

    assert [fill.id for fill in fills] == ["fill-2", "fill-3"]


def test_get_fills_excludes_a_fill_whose_ts_equals_since(
    fake_exchange: FakeExchange,
) -> None:
    """`since` is exclusive: a fill whose ts equals `since` is not returned."""
    since = datetime(2024, 11, 15, tzinfo=UTC)  # exactly fill-2's timestamp

    fills = fake_exchange.get_fills(since)

    assert [fill.id for fill in fills] == ["fill-3"]


def test_get_fills_since_after_every_fill_returns_nothing(
    fake_exchange: FakeExchange,
) -> None:
    """A `since` later than every fill returns an empty tuple, not an error."""
    since = datetime(2025, 1, 1, tzinfo=UTC)

    assert fake_exchange.get_fills(since) == ()


def test_fills_price_and_quantity_are_unit_wrapped(fake_exchange: FakeExchange) -> None:
    """Fill price and quantity are `PricePips` and `ContractCentis` values."""
    since = datetime(2024, 1, 1, tzinfo=UTC)

    fills = fake_exchange.get_fills(since)

    first = fills[0]
    assert isinstance(first.price, PricePips)
    assert isinstance(first.quantity, ContractCentis)
    assert first.price == PricePips(4500)
    assert first.quantity == ContractCentis(100)


# --- Balances, semantics, status, time -------------------------------------------


def test_get_balance_semantics_is_the_all_known_fixture(
    fake_exchange: FakeExchange,
) -> None:
    """The shared fixture records a known member for every field (issue #18)."""
    semantics = fake_exchange.get_balance_semantics()

    assert semantics.open_order_collateral_in_total is OrderCollateralInTotal.INCLUDED
    assert (
        semantics.open_order_collateral_in_available
        is OrderCollateralInAvailable.DEDUCTED_FROM_AVAILABLE
    )
    assert semantics.fee_debit_timing is FeeDebitTiming.AT_EXECUTION
    assert semantics.fee_rounding is FeeRounding.EXACT
    assert (
        semantics.partial_fill_representation
        is PartialFillRepresentation.PER_FILL_RECORDS
    )
    assert semantics.cancel_collateral_release is CancelCollateralRelease.IMMEDIATE
    assert semantics.unsettled_proceeds is UnsettledProceeds.INCLUDED_IMMEDIATELY
    assert semantics.halted_market_behavior is HaltedMarketBehavior.NEW_ORDERS_ACCEPTED
    assert semantics.is_fully_known() is True


def test_get_balances_wraps_amounts_in_money_micros(
    fake_exchange: FakeExchange,
) -> None:
    """Total and available balances are `MoneyMicros` holding the fixture amounts."""
    balances = fake_exchange.get_balances()

    assert balances.total == MoneyMicros(100_000_000)
    assert balances.available == MoneyMicros(95_000_000)
    assert isinstance(balances.total, MoneyMicros)


def test_get_exchange_status_is_deterministic_from_fixture(
    fake_exchange: FakeExchange,
) -> None:
    """Repeated status reads both return the fixture's recorded `open` status."""
    first = fake_exchange.get_exchange_status()
    second = fake_exchange.get_exchange_status()

    assert first.status == "open"
    assert first.status == second.status


def test_get_exchange_time_is_deterministic_from_fixture(
    fake_exchange: FakeExchange,
) -> None:
    """Repeated time reads both return the same fixed fixture instant."""
    first = fake_exchange.get_exchange_time()
    second = fake_exchange.get_exchange_time()

    assert first == second
    assert first == datetime(2024, 12, 1, tzinfo=UTC)


# --- Fee models -------------------------------------------------------------------


def test_get_fee_model_looks_up_by_ticker(fake_exchange: FakeExchange) -> None:
    """A ticker with its own schedule gets that schedule's maker/taker ppm rates."""
    fee_model = fake_exchange.get_fee_model("KXFED-24DEC")

    assert fee_model.schedule_id == "kxfed-promo-v1"
    assert fee_model.maker_fee_ppm == 0
    assert fee_model.taker_fee_ppm == 35_000
    assert isinstance(fee_model.taker_fee_ppm, int)


def test_get_fee_model_falls_back_to_default_for_unlisted_ticker(
    fake_exchange: FakeExchange,
) -> None:
    """A ticker with no schedule of its own falls back to the standard schedule."""
    fee_model = fake_exchange.get_fee_model("KXWEA-24DEC")

    assert fee_model.schedule_id == "standard-v1"
    assert fee_model.taker_fee_ppm == 70_000


# --- Positions and open orders ----------------------------------------------------


def test_get_positions_returns_the_fixture_position(
    fake_exchange: FakeExchange,
) -> None:
    """The one fixture position keeps a unit-typed quantity and average price."""
    positions = fake_exchange.get_positions()

    assert isinstance(positions, tuple)
    assert len(positions) == 1
    assert positions[0].ticker == "KXFED-24DEC"
    assert positions[0].quantity == ContractCentis(500)
    assert positions[0].average_price == PricePips(4550)


def test_get_open_orders_returns_the_empty_fixture(fake_exchange: FakeExchange) -> None:
    """The fixture records no open orders, so an empty tuple comes back."""
    open_orders = fake_exchange.get_open_orders()

    assert open_orders == ()


# --- Not-yet-implemented order actions --------------------------------------------


def test_place_order_raises_not_implemented(fake_exchange: FakeExchange) -> None:
    """`place_order` is not implemented yet and raises `NotImplementedError`."""
    with pytest.raises(NotImplementedError):
        fake_exchange.place_order(object(), object())


def test_cancel_order_raises_not_implemented(fake_exchange: FakeExchange) -> None:
    """`cancel_order` is not implemented yet and raises `NotImplementedError`."""
    with pytest.raises(NotImplementedError):
        fake_exchange.cancel_order("some-order-id")


# --- Fixture hygiene: no float leaf anywhere, every market round-trips -----------


def _walk_no_float(node: object) -> None:
    """Recursively assert that no value in `node` is a bare float."""
    if isinstance(node, dict):
        for value in node.values():
            _walk_no_float(value)
    elif isinstance(node, list):
        for item in node:
            _walk_no_float(item)
    else:
        assert type(node) is not float, f"float leaf found: {node!r}"


@pytest.mark.parametrize("filename", _ALL_FIXTURE_FILES)
def test_fixture_files_contain_no_float_leaf(fixture_dir: Path, filename: str) -> None:
    """No fixture file carries a bare float leaf anywhere in its JSON tree."""
    payload = json.loads((fixture_dir / filename).read_text(encoding="utf-8"))

    _walk_no_float(payload)


def test_every_market_fixture_round_trips_through_normalized_market(
    fake_exchange: FakeExchange,
) -> None:
    """Every loaded market is a valid, constructible NormalizedMarket."""
    markets = fake_exchange.list_markets()

    assert markets
    for market in markets:
        assert isinstance(market, NormalizedMarket)


# --- issue #392: the fixture loader refuses an offsetless timestamp -----------
#
# `FakeExchange` is a test double, but a double that disagrees with production
# teaches the wrong lesson: `windbreak/connector/kalshi/normalize.py` refuses an
# offsetless exchange timestamp rather than reading its wall clock as the
# host's local time, so the fixture loader must refuse identically. Both tests
# pin a UTC-05:00 host (`local_timezone_utc_minus_5`, tests/connector/
# conftest.py) because on a UTC host the misreading is the identity -- an
# unpinned assertion here could never fail.


@pytest.mark.usefixtures("local_timezone_utc_minus_5")
def test_offsetless_fixture_timestamp_is_refused_naming_the_field_and_value() -> None:
    """An offsetless fixture timestamp raises, naming both field and value.

    Pre-fix this returned `2024-12-19T00:00:00+00:00` under the pinned host --
    five hours late and a day later than the fixture says -- and `19:00Z` on a
    UTC host. A fixture whose meaning depends on the machine reading it is not
    a fixture, so the loader refuses instead of guessing an offset.
    """
    with pytest.raises(ValueError, match=r"close_time.*2024-12-18T19:00:00"):
        _parse_dt("2024-12-18T19:00:00", field="close_time")


@pytest.mark.usefixtures("local_timezone_utc_minus_5")
def test_offset_bearing_fixture_timestamp_converts_to_one_fixed_instant() -> None:
    """A timestamp carrying an offset converts by *its* offset, not the host's.

    The `-05:00` here coincides with the pinned host offset only by
    construction; what the assertion pins is that the string's own offset is
    what moves the clock, and that the refusal above did not over-reach into
    rejecting every non-`Z` timestamp.
    """
    assert _parse_dt("2024-12-18T14:00:00-05:00", field="close_time") == datetime(
        2024, 12, 18, 19, tzinfo=UTC
    )
