"""Tests for windbreak.connector.fees (issue #18): FeeModel's fee-bound math.

`windbreak/connector/fees.py` does not exist yet, so importing it fails
collection with `ModuleNotFoundError: No module named 'windbreak.connector.fees'`
-- the expected Gate 1 RED state for issue #18.

`FeeModel` carries a schedule's maker/taker/settlement rates in ppm and exposes
two upper-bound-fee calculators, both returning plain integer micros, both
computed entirely in integer arithmetic (this package is float-denylisted by
`scripts/lint_no_floats.py`):

    * `max_trading_fee_micros(price_pips, count_centis)` -- the worst-case
      per-trade fee, using the higher of the maker/taker rate applied to the
      quadratic `count * price * (1 - price)` bound (the maximum any resting
      or crossing fill at that price/size could cost), ceiling-rounded to the
      next whole cent (SPEC: fees always round in the exchange's favor, so the
      upper bound must too).
    * `max_settlement_fee_micros(count_centis)` -- the worst-case settlement
      fee over a $1/contract payout, likewise ceiling-rounded to the cent.

Every assertion below uses only `*`, `+`, `-`, and integer comparison -- never
`/` or a float literal.
"""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from windbreak.connector.fees import FeeModel

#: Bounds chosen to keep hypothesis examples fast while still covering the
#: full legal domain of each parameter.
_PRICE_STRATEGY = st.integers(min_value=0, max_value=10_000)
_COUNT_STRATEGY = st.integers(min_value=1, max_value=1_000_000)
_RATE_STRATEGY = st.integers(min_value=0, max_value=200_000)


def _model(rate_ppm: int, *, settlement_fee_ppm: int = 0) -> FeeModel:
    """Build a `FeeModel` with a given taker rate (maker pinned at 0)."""
    return FeeModel(
        schedule_id="prop-test-v1",
        maker_fee_ppm=0,
        taker_fee_ppm=rate_ppm,
        settlement_fee_ppm=settlement_fee_ppm,
    )


# --- FeeModel construction validation -----------------------------------------


def test_empty_schedule_id_raises_value_error() -> None:
    """An empty `schedule_id` raises ValueError naming the field."""
    with pytest.raises(ValueError, match="schedule_id"):
        FeeModel(schedule_id="", maker_fee_ppm=0, taker_fee_ppm=0, settlement_fee_ppm=0)


@pytest.mark.parametrize(
    "field", ["maker_fee_ppm", "taker_fee_ppm", "settlement_fee_ppm"]
)
def test_negative_fee_ppm_raises_value_error(field: str) -> None:
    """A negative ppm rate on any fee field raises ValueError naming that field."""
    kwargs: dict[str, object] = {
        "schedule_id": "s",
        "maker_fee_ppm": 0,
        "taker_fee_ppm": 0,
        "settlement_fee_ppm": 0,
        field: -1,
    }
    with pytest.raises(ValueError, match=field):
        FeeModel(**kwargs)


@pytest.mark.parametrize(
    "field", ["maker_fee_ppm", "taker_fee_ppm", "settlement_fee_ppm"]
)
def test_bool_fee_ppm_raises_type_error(field: str) -> None:
    """A `bool` fee rate raises TypeError on any field, despite `bool` being an int."""
    kwargs: dict[str, object] = {
        "schedule_id": "s",
        "maker_fee_ppm": 0,
        "taker_fee_ppm": 0,
        "settlement_fee_ppm": 0,
        field: True,
    }
    with pytest.raises(TypeError):
        FeeModel(**kwargs)


def test_fee_model_is_frozen() -> None:
    """A `FeeModel` is frozen: assigning a field raises FrozenInstanceError."""
    model = _model(70_000)
    # Assign through a dynamic attribute name so the test exercises the frozen
    # dataclass's runtime rejection without a static type-checker suppression.
    frozen_field = "taker_fee_ppm"

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(model, frozen_field, 0)


# --- max_trading_fee_micros: the exchange-documented golden example ----------


def test_max_trading_fee_micros_matches_the_exchange_documented_example() -> None:
    """The issue's worked example: a 7% taker rate, 50c price, 100 contracts."""
    model = FeeModel(
        schedule_id="kxfed-standard-v1",
        maker_fee_ppm=0,
        taker_fee_ppm=70_000,
        settlement_fee_ppm=0,
    )

    fee = model.max_trading_fee_micros(price_pips=5_000, count_centis=10_000)

    assert fee == 1_750_000
    assert fee >= 175_000


def test_max_trading_fee_micros_rounds_up_a_nonzero_remainder_to_a_whole_cent() -> None:
    """The smallest possible nonzero fee still ceils up to a full cent (10_000)."""
    model = _model(1)

    fee = model.max_trading_fee_micros(price_pips=1, count_centis=1)

    assert fee == 10_000


@pytest.mark.parametrize("price_pips", [0, 10_000])
def test_max_trading_fee_micros_is_zero_at_price_boundaries(price_pips: int) -> None:
    """At price 0 or 10_000 the quadratic bound `p * (10_000 - p)` is exactly 0."""
    model = _model(70_000)

    assert model.max_trading_fee_micros(price_pips=price_pips, count_centis=10_000) == 0


def test_max_trading_fee_micros_is_zero_for_a_zero_rate_schedule() -> None:
    """A zero-rate schedule charges nothing, with no round-up to a whole cent."""
    model = _model(0)

    assert model.max_trading_fee_micros(price_pips=5_000, count_centis=10_000) == 0


def test_max_trading_fee_micros_uses_the_higher_of_maker_and_taker() -> None:
    """The upper bound must use whichever rate is higher, regardless of side."""
    maker_higher = FeeModel(
        schedule_id="s", maker_fee_ppm=70_000, taker_fee_ppm=0, settlement_fee_ppm=0
    )
    taker_higher = FeeModel(
        schedule_id="s", maker_fee_ppm=0, taker_fee_ppm=70_000, settlement_fee_ppm=0
    )

    assert maker_higher.max_trading_fee_micros(
        price_pips=5_000, count_centis=10_000
    ) == taker_higher.max_trading_fee_micros(price_pips=5_000, count_centis=10_000)


# --- max_trading_fee_micros: validation ---------------------------------------


@pytest.mark.parametrize("price_pips", [-1, 10_001])
def test_max_trading_fee_micros_rejects_out_of_range_price(price_pips: int) -> None:
    """A price outside 0..10_000 pips raises ValueError naming `price_pips`."""
    model = _model(70_000)

    with pytest.raises(ValueError, match="price_pips"):
        model.max_trading_fee_micros(price_pips=price_pips, count_centis=10_000)


def test_max_trading_fee_micros_rejects_bool_price() -> None:
    """A `bool` price raises TypeError rather than being read as 0 or 1 pip."""
    model = _model(70_000)

    with pytest.raises(TypeError):
        model.max_trading_fee_micros(price_pips=True, count_centis=10_000)


@pytest.mark.parametrize("count_centis", [0, -1])
def test_max_trading_fee_micros_rejects_non_positive_count(count_centis: int) -> None:
    """A zero or negative count raises ValueError naming `count_centis`."""
    model = _model(70_000)

    with pytest.raises(ValueError, match="count_centis"):
        model.max_trading_fee_micros(price_pips=5_000, count_centis=count_centis)


def test_max_trading_fee_micros_rejects_bool_count() -> None:
    """A `bool` count raises TypeError rather than being read as one centi."""
    model = _model(70_000)

    with pytest.raises(TypeError):
        model.max_trading_fee_micros(price_pips=5_000, count_centis=True)


# --- max_trading_fee_micros: properties (integer math only) -------------------


@given(
    rate_ppm=_RATE_STRATEGY, count_centis=_COUNT_STRATEGY, price_pips=_PRICE_STRATEGY
)
def test_max_trading_fee_micros_is_a_true_ceiling_over_the_exact_product(
    rate_ppm: int, count_centis: int, price_pips: int
) -> None:
    """Scaled back up by 10**10, the result always dominates the exact product.

    `max_trading_fee_micros` is `ceil(numer / 1e14)` cents times 10_000 micros,
    so `result * 10**10 == ceil(numer / 1e14) * 1e14 >= numer` by the ceiling
    property -- verified here with cross-multiplication, never division.
    """
    model = _model(rate_ppm)

    result = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=count_centis
    )

    numer = rate_ppm * count_centis * price_pips * (10_000 - price_pips)
    assert result * 10**10 >= numer


@given(
    rate_ppm=_RATE_STRATEGY, count_centis=_COUNT_STRATEGY, price_pips=_PRICE_STRATEGY
)
def test_max_trading_fee_micros_is_always_a_whole_cent(
    rate_ppm: int, count_centis: int, price_pips: int
) -> None:
    """The trading-fee bound is always a whole cent: a multiple of 10_000 micros."""
    model = _model(rate_ppm)

    result = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=count_centis
    )

    assert result % 10_000 == 0


@given(
    rate_ppm=_RATE_STRATEGY,
    price_pips=_PRICE_STRATEGY,
    base_count=st.integers(min_value=1, max_value=500_000),
    delta=st.integers(min_value=0, max_value=500_000),
)
def test_max_trading_fee_micros_is_monotone_nondecreasing_in_count(
    rate_ppm: int, price_pips: int, base_count: int, delta: int
) -> None:
    """Raising the contract count never lowers the trading-fee bound."""
    model = _model(rate_ppm)

    smaller = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=base_count
    )
    larger = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=base_count + delta
    )

    assert larger >= smaller


@given(
    count_centis=_COUNT_STRATEGY,
    price_pips=_PRICE_STRATEGY,
    base_rate=st.integers(min_value=0, max_value=100_000),
    delta=st.integers(min_value=0, max_value=100_000),
)
def test_max_trading_fee_micros_is_monotone_nondecreasing_in_rate(
    count_centis: int, price_pips: int, base_rate: int, delta: int
) -> None:
    """Raising the fee rate never lowers the trading-fee bound."""
    smaller = _model(base_rate).max_trading_fee_micros(
        price_pips=price_pips, count_centis=count_centis
    )
    larger = _model(base_rate + delta).max_trading_fee_micros(
        price_pips=price_pips, count_centis=count_centis
    )

    assert larger >= smaller


@given(
    rate_ppm=_RATE_STRATEGY, count_centis=_COUNT_STRATEGY, price_pips=_PRICE_STRATEGY
)
def test_max_trading_fee_micros_is_symmetric_under_price_reflection(
    rate_ppm: int, count_centis: int, price_pips: int
) -> None:
    """`p * (10_000 - p)` is symmetric under `p -> 10_000 - p`; the fee is too."""
    model = _model(rate_ppm)
    mirrored_price = 10_000 - price_pips

    original = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=count_centis
    )
    mirrored = model.max_trading_fee_micros(
        price_pips=mirrored_price, count_centis=count_centis
    )

    assert original == mirrored


# --- max_settlement_fee_micros -------------------------------------------------


def test_max_settlement_fee_micros_is_zero_for_the_standard_schedule() -> None:
    """A schedule whose settlement rate is zero charges no settlement fee."""
    model = _model(70_000, settlement_fee_ppm=0)

    assert model.max_settlement_fee_micros(count_centis=10_000) == 0


def test_max_settlement_fee_micros_rounds_up_a_remainder_to_a_whole_cent() -> None:
    """333_333 ppm on 100 centis (1 contract): 33.3333 cents ceils to 34."""
    model = _model(0, settlement_fee_ppm=333_333)

    fee = model.max_settlement_fee_micros(count_centis=100)

    assert fee == 340_000


def test_max_settlement_fee_micros_is_exact_when_evenly_divisible() -> None:
    """An evenly divisible settlement fee is not ceiled past its exact value."""
    model = _model(0, settlement_fee_ppm=500_000)

    fee = model.max_settlement_fee_micros(count_centis=100)

    assert fee == 500_000


@pytest.mark.parametrize("count_centis", [0, -1])
def test_max_settlement_fee_micros_rejects_non_positive_count(
    count_centis: int,
) -> None:
    """A zero or negative count raises ValueError naming `count_centis`."""
    model = _model(70_000)

    with pytest.raises(ValueError, match="count_centis"):
        model.max_settlement_fee_micros(count_centis=count_centis)


def test_max_settlement_fee_micros_rejects_bool_count() -> None:
    """A `bool` count raises TypeError rather than being read as one centi."""
    model = _model(70_000)

    with pytest.raises(TypeError):
        model.max_settlement_fee_micros(count_centis=True)


@given(count_centis=_COUNT_STRATEGY, settlement_fee_ppm=_RATE_STRATEGY)
def test_max_settlement_fee_micros_is_a_true_ceiling_over_the_exact_product(
    count_centis: int, settlement_fee_ppm: int
) -> None:
    """`result * 10**6 >= settlement_fee_ppm * count_centis` (ceiling property)."""
    model = _model(0, settlement_fee_ppm=settlement_fee_ppm)

    result = model.max_settlement_fee_micros(count_centis=count_centis)

    assert result * 10**6 >= settlement_fee_ppm * count_centis * 10_000


@given(count_centis=_COUNT_STRATEGY, settlement_fee_ppm=_RATE_STRATEGY)
def test_max_settlement_fee_micros_is_always_a_whole_cent(
    count_centis: int, settlement_fee_ppm: int
) -> None:
    """The settlement-fee bound is always a whole cent: a multiple of 10_000 micros."""
    model = _model(0, settlement_fee_ppm=settlement_fee_ppm)

    result = model.max_settlement_fee_micros(count_centis=count_centis)

    assert result % 10_000 == 0
