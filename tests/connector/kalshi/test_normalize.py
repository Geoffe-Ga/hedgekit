"""Tests for windbreak.connector.kalshi.normalize (issues #17, #106).

Pins the pure (no-I/O) normalization functions: `payload_hash`,
`gate_product`, `normalize_market`, `normalize_order_book`. All prices/sizes
convert to windbreak's fixed-point unit types (`PricePips`, `ContractCentis`)
with zero float intermediaries (SPEC S6.1).

Issue #106 adds `volume_24h_micros = int(raw["volume_24h"]) *
MICROS_PER_CONTRACT_NOTIONAL` (settlement-notional micro-dollars, exact
integer math). Until `normalize_market` sets that field, these new tests fail
with `AttributeError: 'NormalizedMarket' object has no attribute
'volume_24h_micros'` -- the expected Gate 1 RED state for issue #106.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from windbreak.connector.kalshi.normalize import (
    gate_product,
    normalize_market,
    normalize_order_book,
    payload_hash,
)
from windbreak.numeric import ContractCentis, PricePips

if TYPE_CHECKING:
    from windbreak.connector.kalshi.adapter import KalshiConnector

#: A fixed point in time for order-book normalization tests.
_FETCHED_AT = datetime(2024, 12, 1, tzinfo=UTC)


def _raw_binary(**overrides: Any) -> dict[str, Any]:
    """Build a raw Kalshi binary-market payload, with field overrides.

    Args:
        **overrides: Field values overriding the base binary-market payload.

    Returns:
        A raw Kalshi market mapping shaped like a `/markets` list entry.
    """
    base: dict[str, Any] = {
        "ticker": "KXFED-24DEC",
        "event_ticker": "KXFED",
        "market_type": "binary",
        "title": "Fed raises rates in December 2024?",
        "rules_primary": (
            "Resolves YES if the FOMC raises the federal funds rate at its "
            "December 2024 meeting."
        ),
        "category": "Economics",
        "close_time": "2024-12-18T19:00:00Z",
        "expected_expiration_time": "2024-12-18T20:00:00Z",
        "tick_size": 1,
        "volume_24h": 1234,
    }
    base.update(overrides)
    return base


def _raw_event(**overrides: Any) -> dict[str, Any]:
    """Build a raw Kalshi event payload, with field overrides.

    Args:
        **overrides: Field values overriding the base event payload.

    Returns:
        A raw Kalshi event mapping shaped like an `/events` list entry.
    """
    base: dict[str, Any] = {
        "event_ticker": "KXFED",
        "title": "Fed rate decisions",
        "mutually_exclusive": True,
    }
    base.update(overrides)
    return base


# --- normalize_market: cents->pips, fixed fields, jurisdiction --------------


def test_binary_market_normalizes_exact_cents_to_pips_and_fixed_fields() -> None:
    """A one-cent tick becomes exactly 100 pips; the fixed fields are pinned."""
    market = normalize_market(_raw_binary(tick_size=1), _raw_event())

    assert market.exchange == "kalshi"
    assert market.price_tick_pips == 100
    assert market.min_order_contract_centis == 100
    assert market.market_type == "fully_collateralized_binary"
    assert market.fractional_trading_enabled is False


def test_tick_size_absent_defaults_to_one_cent_tick() -> None:
    """A missing `tick_size` defaults to a 1-cent (100-pip) tick."""
    raw = _raw_binary()
    del raw["tick_size"]

    market = normalize_market(raw, None)

    assert market.price_tick_pips == 100


def test_tick_size_five_cents_scales_to_five_hundred_pips() -> None:
    """`tick_size` is an exact cents->pips multiplication (`* 100`)."""
    market = normalize_market(_raw_binary(tick_size=5), None)

    assert market.price_tick_pips == 500


def test_jurisdiction_status_is_always_unknown() -> None:
    """Kalshi exposes no eligibility signal, so jurisdiction is always unknown.

    SPEC S20 Q3: `jurisdiction_status` must be `"unknown"` for Kalshi, never
    inferred as `"eligible"` even when nothing in the payload suggests
    otherwise.
    """
    market = normalize_market(_raw_binary(), _raw_event())

    assert market.jurisdiction_status == "unknown"


@pytest.mark.parametrize(
    ("mutually_exclusive", "expected_group_id"),
    [(True, "KXFED"), (False, None)],
)
def test_group_id_reflects_event_mutual_exclusivity(
    mutually_exclusive: bool, expected_group_id: str | None
) -> None:
    """The group id is the event ticker iff the event is mutually exclusive."""
    market = normalize_market(
        _raw_binary(), _raw_event(mutually_exclusive=mutually_exclusive)
    )

    assert market.mutually_exclusive_group_id == expected_group_id


def test_group_id_is_none_when_no_event_is_supplied() -> None:
    """A standalone market with no event payload has no group id."""
    market = normalize_market(_raw_binary(), None)

    assert market.mutually_exclusive_group_id is None


def test_volume_24h_converts_to_exact_settlement_notional_micros() -> None:
    """`volume_24h` (whole settled contracts) converts to settlement micros.

    A raw `volume_24h` of 1234 (whole contracts, each worth up to $1 at
    settlement) becomes exactly `1_234_000_000` settlement-notional
    micro-dollars: an exact integer multiplication, never a float.
    """
    market = normalize_market(_raw_binary(volume_24h=1234), None)

    assert market.volume_24h_micros == 1_234_000_000


def test_volume_24h_zero_converts_to_zero_micros() -> None:
    """A zero `volume_24h` converts to exactly zero micros, not treated as absent."""
    market = normalize_market(_raw_binary(volume_24h=0), None)

    assert market.volume_24h_micros == 0


def test_raw_exchange_payload_hash_matches_payload_hash_of_the_raw_market() -> None:
    """The stored hash is exactly `payload_hash` applied to the raw market."""
    raw = _raw_binary()

    market = normalize_market(raw, None)

    assert market.raw_exchange_payload_hash == payload_hash(raw)


# --- issue #392: offsetless timestamps are refused, never read as host-local ---
#
# Every test below is pinned to a fixed UTC-05:00 host by
# `local_timezone_utc_minus_5` (tests/connector/conftest.py). The pin is what
# makes these assertions mean something: `datetime.fromisoformat` returns a
# *naive* datetime for an offsetless string, and `.astimezone(UTC)` on a naive
# value does not raise -- it reads the wall clock as the **host's** local time.
# On a UTC host (which CI is) that misreading is the identity, so an unpinned
# test here would be a permanent false green.


@pytest.mark.usefixtures("local_timezone_utc_minus_5")
def test_offsetless_close_time_is_refused_naming_the_field_and_value() -> None:
    """An offsetless `close_time` is refused, not silently read as host-local.

    `"2024-12-18T19:00:00"` carries no offset. The pre-fix code returned
    `2024-12-19T00:00:00+00:00` under this fixture's UTC-05:00 host -- five
    hours late and across the date boundary, which can make a closed market
    read as open -- while returning `19:00Z` on a UTC host. Neither is provable
    from the payload, so the parse refuses.

    The refusal must be unconditional, not "only when the host is not UTC": the
    pinned zone rules out an implementation that merely compares local time to
    UTC, and the raised `ValueError` rules out both the old silent shift and a
    "just assume UTC" fix that would return an unprovable instant.
    """
    raw = _raw_binary(close_time="2024-12-18T19:00:00")

    with pytest.raises(ValueError, match=r"close_time.*2024-12-18T19:00:00"):
        normalize_market(raw, None)


@pytest.mark.usefixtures("local_timezone_utc_minus_5")
def test_offsetless_expected_expiration_time_is_refused_naming_its_own_field() -> None:
    """The optional-timestamp path refuses too, and names *its* field.

    `_parse_optional_dt` preserves `None`, so its offsetless branch is a
    separate path from `close_time`'s. The error must name
    `expected_expiration_time` -- an operator reading the ledgered
    `MARKET_MALFORMED` event has only this message to locate the bad field.
    """
    raw = _raw_binary(expected_expiration_time="2024-12-18T20:00:00")

    with pytest.raises(ValueError, match=r"expected_expiration_time"):
        normalize_market(raw, None)


@pytest.mark.usefixtures("local_timezone_utc_minus_5")
def test_offset_bearing_timestamps_normalize_to_the_same_instant_on_any_host() -> None:
    """A timestamp that *does* carry an offset still converts, host-irrelevant.

    A non-UTC offset (`-05:00`) is used deliberately: it proves the conversion
    reads the offset in the string rather than the host's zone, and that the
    refusal above did not over-reach into rejecting every non-`Z` timestamp.
    """
    raw = _raw_binary(
        close_time="2024-12-18T14:00:00-05:00",
        expected_expiration_time="2024-12-18T20:00:00+00:00",
    )

    market = normalize_market(raw, None)

    assert market.close_time == datetime(2024, 12, 18, 19, tzinfo=UTC)
    assert market.expected_resolution_time == datetime(2024, 12, 18, 20, tzinfo=UTC)


@pytest.mark.usefixtures("local_timezone_utc_minus_5")
def test_absent_expected_expiration_time_stays_none_rather_than_refusing() -> None:
    """An *absent* optional timestamp is still `None`, not an offsetless refusal.

    Kalshi omits `expected_expiration_time` on markets with no announced
    resolution time. Absence is a stated fact about the market, not unprovable
    evidence about an instant, so it must keep flowing through untouched.
    """
    raw = _raw_binary()
    del raw["expected_expiration_time"]

    assert normalize_market(raw, None).expected_resolution_time is None


# --- payload_hash ------------------------------------------------------


def test_payload_hash_is_non_empty_sha256_hex() -> None:
    """The hash is a full 64-character lowercase-hex sha256 digest."""
    digest = payload_hash(_raw_binary())

    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_payload_hash_is_stable_under_key_reordering() -> None:
    """Canonical JSON hashing must not depend on Python dict insertion order."""
    ordered = {"a": 1, "b": 2, "c": 3}
    reordered = {"c": 3, "a": 1, "b": 2}

    assert payload_hash(ordered) == payload_hash(reordered)


def test_payload_hash_differs_across_different_payloads() -> None:
    """A changed value must change the hash."""
    assert payload_hash({"a": 1}) != payload_hash({"a": 2})


# --- gate_product: allowlist, not denylist ----------------------------------


def test_gate_product_passes_binary() -> None:
    """`"binary"` is the only market type that is not refused."""
    assert gate_product(_raw_binary(market_type="binary")) is None


@pytest.mark.parametrize("bad_type", ["perpetual", "scalar"])
def test_gate_product_refuses_non_binary_types_naming_the_type(bad_type: str) -> None:
    """A non-binary product is refused with a reason naming its type.

    This is an allowlist, not a denylist: any type other than `"binary"` is
    refused, so a newly-invented Kalshi product type is refused by default
    rather than silently normalized.
    """
    reason = gate_product(_raw_binary(market_type=bad_type))

    assert reason
    assert bad_type in reason


def test_gate_product_refuses_missing_market_type() -> None:
    """A market payload missing `market_type` entirely is refused, not skipped."""
    raw = _raw_binary()
    del raw["market_type"]

    reason = gate_product(raw)

    assert reason
    assert "missing" in reason.lower()


# --- normalize_order_book ----------------------------------------------


def test_yes_bids_sort_desc_and_no_bids_invert_into_asks_sorted_asc() -> None:
    """YES levels become bids (desc); NO levels invert to asks (asc).

    The raw `yes`/`no` lists are deliberately supplied out of price order so
    a normalize function that merely preserves input order (rather than
    truly sorting) fails this test.
    """
    raw = {
        "orderbook": {
            "yes": [[44, 250], [45, 100]],
            "no": [[52, 40], [55, 20]],
        }
    }

    book = normalize_order_book("KXFED-24DEC", raw, _FETCHED_AT)

    assert [level.price.value for level in book.yes_bids] == [4500, 4400]
    assert [level.quantity.value for level in book.yes_bids] == [10_000, 25_000]
    assert [level.price.value for level in book.yes_asks] == [4500, 4800]
    assert [level.quantity.value for level in book.yes_asks] == [2_000, 4_000]
    assert book.ticker == "KXFED-24DEC"
    assert book.fetched_at == _FETCHED_AT


def test_order_book_empty_or_absent_sides_become_empty_tuples() -> None:
    """No `yes`/`no` keys at all still yields empty (not missing) tuples."""
    raw = {"orderbook": {}}

    book = normalize_order_book("KXBAN-24DEC", raw, _FETCHED_AT)

    assert book.yes_bids == ()
    assert book.yes_asks == ()


def test_order_book_levels_are_pricepips_and_contractcentis() -> None:
    """Every level wraps windbreak's real fixed-point unit types."""
    raw = {"orderbook": {"yes": [[45, 100]], "no": [[52, 40]]}}

    book = normalize_order_book("KXFED-24DEC", raw, _FETCHED_AT)

    for level in book.yes_bids + book.yes_asks:
        assert isinstance(level.price, PricePips)
        assert isinstance(level.quantity, ContractCentis)


def test_orderbook_prices_are_fixed_point_pips_acceptance(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """The issue's acceptance test, reconciled to the real model API.

    `OrderBookLevel` exposes `.price` / `.quantity` (each a unit-wrapper with
    a `.value: int`) -- not the placeholder `.price_pips` / `.count_centis`
    attributes from the original issue draft, which do not exist on the real
    `windbreak.connector.models.OrderBookLevel`.
    """
    book = kalshi_fixture_connector.get_order_book("KXFED-24DEC")

    for level in book.yes_bids + book.yes_asks:
        assert isinstance(level.price.value, int)
        assert 0 < level.price.value < 10_000
        assert isinstance(level.quantity.value, int)
