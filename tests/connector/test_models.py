"""Tests for windbreak.connector.models (issues #16, #106): SPEC S6.2 models.

Pins `NormalizedMarket.__post_init__` validation (market_type, jurisdiction
enum, tick/min-order integrality and positivity, non-empty payload hash),
immutability, and `market_to_payload`'s JSON-safety (no float leaf anywhere,
datetimes rendered as ISO-8601 `Z` strings). `windbreak/connector/` does not
exist yet, so importing `windbreak.connector.models` fails collection with
`ModuleNotFoundError: No module named 'windbreak.connector'` -- the expected
Gate 1 RED state for issue #16.

Issue #106 adds `volume_24h_micros: int`, a plain non-negative int (NOT a
`MoneyMicros`) validated by a new `_require_nonnegative_unit_int` guard: bool
and non-int values still raise `TypeError`, negative raises `ValueError`, but
-- unlike the strictly-positive tick/min-order guard -- zero is legal. Until
that field and guard exist, every `_market()` call in this module raises
`TypeError: __init__() got an unexpected keyword argument 'volume_24h_micros'`
-- the expected Gate 1 RED state for issue #106.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime

import pytest

from windbreak.connector.models import NormalizedMarket, market_to_payload

_VALID_KWARGS: dict[str, object] = {
    "exchange": "fake-exchange",
    "ticker": "KXFED-24DEC",
    "event_ticker": "KXFED-24",
    "title": "Fed raises rates in December 2024?",
    "resolution_criteria": "Resolves YES if the FOMC raises rates.",
    "category": "economics",
    "close_time": datetime(2024, 12, 18, 19, tzinfo=UTC),
    "expected_resolution_time": None,
    "market_type": "fully_collateralized_binary",
    "price_tick_pips": 100,
    "min_order_contract_centis": 100,
    "fractional_trading_enabled": False,
    "mutually_exclusive_group_id": None,
    "jurisdiction_status": "eligible",
    "raw_exchange_payload_hash": "sha256:abc123",
    "volume_24h_micros": 1000000,
}


def _market(**overrides: object) -> NormalizedMarket:
    return NormalizedMarket(**{**_VALID_KWARGS, **overrides})


def test_valid_market_constructs_without_error() -> None:
    """The valid kwarg set constructs a market that keeps its field values."""
    market = _market()

    assert market.ticker == "KXFED-24DEC"
    assert market.jurisdiction_status == "eligible"


def test_bad_market_type_raises_value_error() -> None:
    """An unrecognized `market_type` raises ValueError naming the field."""
    with pytest.raises(ValueError, match="market_type"):
        _market(market_type="binary")


def test_bad_jurisdiction_status_raises_value_error() -> None:
    """A `jurisdiction_status` outside the enum raises ValueError naming it."""
    with pytest.raises(ValueError, match="jurisdiction_status"):
        _market(jurisdiction_status="maybe")


@pytest.mark.parametrize("field", ["price_tick_pips", "min_order_contract_centis"])
def test_bool_tick_or_min_order_raises_type_error(field: str) -> None:
    """A stray `bool` (an `int` subclass) must never masquerade as a tick size."""
    with pytest.raises(TypeError):
        _market(**{field: True})


@pytest.mark.parametrize("field", ["price_tick_pips", "min_order_contract_centis"])
def test_non_int_tick_or_min_order_raises_type_error(field: str) -> None:
    """A string tick or min-order size raises TypeError, never a coercion."""
    with pytest.raises(TypeError):
        _market(**{field: "100"})


@pytest.mark.parametrize("field", ["price_tick_pips", "min_order_contract_centis"])
@pytest.mark.parametrize("bad_value", [0, -1])
def test_non_positive_tick_or_min_order_raises_value_error(
    field: str, bad_value: int
) -> None:
    """Zero or negative tick and min-order sizes raise ValueError naming the field."""
    with pytest.raises(ValueError, match=field):
        _market(**{field: bad_value})


def test_empty_payload_hash_raises_value_error() -> None:
    """An empty `raw_exchange_payload_hash` raises ValueError naming the field."""
    with pytest.raises(ValueError, match="raw_exchange_payload_hash"):
        _market(raw_exchange_payload_hash="")


# --- volume_24h_micros (issue #106): a non-negative int, zero legal ---------


def test_zero_volume_24h_micros_is_accepted() -> None:
    """Unlike the strictly-positive tick/min-order guard, zero volume is legal."""
    market = _market(volume_24h_micros=0)

    assert market.volume_24h_micros == 0


def test_positive_volume_24h_micros_is_accepted() -> None:
    """A positive 24h volume is kept as the plain int micros it was given."""
    market = _market(volume_24h_micros=5_000_000_000)

    assert market.volume_24h_micros == 5_000_000_000


def test_negative_volume_24h_micros_raises_value_error() -> None:
    """A negative 24h volume raises ValueError naming `volume_24h_micros`."""
    with pytest.raises(ValueError, match="volume_24h_micros"):
        _market(volume_24h_micros=-1)


def test_bool_volume_24h_micros_raises_type_error() -> None:
    """A stray `bool` (an `int` subclass) must never masquerade as a volume."""
    with pytest.raises(TypeError):
        _market(volume_24h_micros=True)


def test_non_int_volume_24h_micros_raises_type_error() -> None:
    """A string 24h volume raises TypeError rather than being coerced to int."""
    with pytest.raises(TypeError):
        _market(volume_24h_micros="1000000")


def test_market_is_frozen() -> None:
    """A `NormalizedMarket` is frozen: assignment raises FrozenInstanceError."""
    market = _market()

    with pytest.raises(dataclasses.FrozenInstanceError):
        market.ticker = "OTHER"  # type: ignore[misc]


# --- market_to_payload: JSON-safety -----------------------------------------------


def test_market_to_payload_is_json_dumps_clean() -> None:
    """The payload round-trips unchanged through `json.dumps` and `json.loads`."""
    market = _market(expected_resolution_time=datetime(2024, 12, 18, 20, tzinfo=UTC))

    payload = market_to_payload(market)

    assert json.loads(json.dumps(payload)) == payload


def test_market_to_payload_datetimes_are_iso_z_strings() -> None:
    """Datetimes render as microsecond-precision ISO-8601 strings ending in `Z`."""
    market = _market(expected_resolution_time=datetime(2024, 12, 18, 20, tzinfo=UTC))

    payload = market_to_payload(market)

    assert payload["close_time"] == "2024-12-18T19:00:00.000000Z"
    assert payload["expected_resolution_time"] == "2024-12-18T20:00:00.000000Z"


def test_market_to_payload_handles_none_expected_resolution_time() -> None:
    """A missing expected resolution time stays `None` in the payload."""
    market = _market(expected_resolution_time=None)

    payload = market_to_payload(market)

    assert payload["expected_resolution_time"] is None


def test_market_to_payload_preserves_plain_int_unit_fields() -> None:
    """Tick and min-order sizes come through as exact `int` leaves, values intact."""
    market = _market(price_tick_pips=250, min_order_contract_centis=500)

    payload = market_to_payload(market)

    assert payload["price_tick_pips"] == 250
    assert payload["min_order_contract_centis"] == 500
    assert type(payload["price_tick_pips"]) is int
    assert type(payload["min_order_contract_centis"]) is int


def _assert_no_float_leaf(node: object) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _assert_no_float_leaf(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _assert_no_float_leaf(item)
    else:
        assert type(node) is not float, f"float leaf found in payload: {node!r}"


def test_market_to_payload_contains_no_float_leaf() -> None:
    """No float leaf appears anywhere in the serialized payload tree."""
    market = _market()

    payload = market_to_payload(market)

    _assert_no_float_leaf(payload)


def test_market_to_payload_preserves_volume_24h_micros_as_int_leaf() -> None:
    """The 24h volume serializes as an exact `int` leaf, never a float."""
    market = _market(volume_24h_micros=42)

    payload = market_to_payload(market)

    assert payload["volume_24h_micros"] == 42
    assert type(payload["volume_24h_micros"]) is int
