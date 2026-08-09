"""Hypothesis property suite for `windbreak.connector.fills` (issue #19).

`windbreak.connector.fills` does not exist yet, so importing it fails
collection with `ModuleNotFoundError: No module named
'windbreak.connector.fills'` -- the expected Gate 1 RED state for issue #19.

SPEC S17.4 requires that no simulated fill is ever better than what walking
the recorded book allows, and that simulated cost never falls below the
book's own cost plus the modeled fee. This module backs that requirement with
five properties over randomly generated (but structurally valid) order
books, requested sizes, fee schedules, and pessimism parameters:

    (a) `book_cost` always equals an independently-computed best-first walk
        of the same levels for the realized `filled` quantity -- the walk
        never produces a cheaper fill than the recorded book allows.
    (b) `total_cost` never understates `book_cost + fee` (the haircut is
        never negative).
    (c) `filled` never exceeds the participation cap, which never exceeds the
        eligible (at-or-better) depth.
    (d) Raising `haircut_ppm` never lowers `total_cost` (and never changes
        how much fills -- the haircut is a pure cost adjustment).
    (e) A print stream containing only touches (price == limit, never
        strictly through) never fills a resting order.
    (f) Every consumed level's quantity never exceeds its own recorded
        level's quantity.

Every generated ask book is strictly ascending in price (best-first, no
duplicate price levels), which is both what a real order book guarantees and
what makes the "eligible levels are exactly a prefix" reasoning in (a)/(f)
exact rather than approximate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from hypothesis import given
from hypothesis import strategies as st

from windbreak.connector.fees import FeeModel
from windbreak.connector.fills import (
    RestingFillRequest,
    TradePrint,
    allocate_resting_fills,
    participation_cap,
    resting_fill_quantity,
    walk_taker_fill,
)
from windbreak.connector.models import OrderBookLevel
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Valid, non-degenerate prices (excludes the 0/10_000 edges where the fee
#: model's quadratic bound is trivially zero, so property (a)'s reference
#: walk stays informative).
_PRICE_STRATEGY = st.integers(min_value=1, max_value=9_999)

#: Contract-centis quantities, bounded to keep hypothesis examples fast.
_QTY_STRATEGY = st.integers(min_value=1, max_value=100_000)

#: Fee-schedule taker rates, matching the range used in
#: `tests/connector/kalshi/test_fees.py`'s own property suite.
_RATE_STRATEGY = st.integers(min_value=0, max_value=200_000)

#: Haircut ppm, allowed up to 100% (a fully-doubled cost is still valid).
_HAIRCUT_PPM_STRATEGY = st.integers(min_value=0, max_value=1_000_000)

#: Participation ppm; 0 is legal (always a zero fill) but excluded here so
#: the "cap <= depth" property in (c) stays exercised rather than trivial.
_PARTICIPATION_PPM_STRATEGY = st.integers(min_value=1, max_value=1_000_000)

#: A fixed timestamp for generated `TradePrint`s; irrelevant to every
#: property below, which reason only about price/quantity.
_TS = datetime(2025, 1, 1, tzinfo=UTC)


@st.composite
def _ascending_levels(draw: st.DrawFn) -> tuple[OrderBookLevel, ...]:
    """Build a best-first ask book: strictly increasing, unique prices.

    Sorting ascending guarantees that "levels at-or-better than a limit" is
    always exactly a prefix of the returned tuple -- the structural fact
    every property below relies on.
    """
    prices = draw(st.lists(_PRICE_STRATEGY, min_size=1, max_size=5, unique=True))
    prices.sort()
    quantities = draw(
        st.lists(_QTY_STRATEGY, min_size=len(prices), max_size=len(prices))
    )
    return tuple(
        OrderBookLevel(PricePips(p), ContractCentis(q))
        for p, q in zip(prices, quantities, strict=True)
    )


def _fee_model(rate_ppm: int) -> FeeModel:
    """Build a `FeeModel` with a given taker rate (maker/settlement pinned at 0)."""
    return FeeModel(
        schedule_id="prop-test-v1",
        maker_fee_ppm=0,
        taker_fee_ppm=rate_ppm,
        settlement_fee_ppm=0,
    )


def _reference_book_cost(
    levels: Sequence[OrderBookLevel], quantity: ContractCentis
) -> MoneyMicros:
    """Independently walk `levels` best-first for exactly `quantity` centis.

    Sums the exact `price * qty` product per level -- never a remainder, per
    `money_from_price_and_count`'s own derivation -- so this is a genuinely
    independent re-implementation of the book-cost half of `walk_taker_fill`,
    not a restatement of it.
    """
    remaining = quantity.value
    micros = 0
    for level in levels:
        if remaining <= 0:
            break
        take = min(remaining, level.quantity.value)
        micros += level.price.value * take
        remaining -= take
    return MoneyMicros(micros)


def _eligible_depth(levels: Sequence[OrderBookLevel], limit_price: int) -> int:
    """Sum the quantity of every level at-or-better (<=) than `limit_price`."""
    return sum(
        level.quantity.value for level in levels if level.price.value <= limit_price
    )


# --- (a): book_cost matches an independent best-first walk -------------------


@given(
    levels=_ascending_levels(),
    limit_price=_PRICE_STRATEGY,
    requested=_QTY_STRATEGY,
    rate_ppm=_RATE_STRATEGY,
    haircut_ppm=_HAIRCUT_PPM_STRATEGY,
    max_participation_ppm=_PARTICIPATION_PPM_STRATEGY,
)
def test_book_cost_matches_an_independent_best_first_walk(
    levels: tuple[OrderBookLevel, ...],
    limit_price: int,
    requested: int,
    rate_ppm: int,
    haircut_ppm: int,
    max_participation_ppm: int,
) -> None:
    """The walk never produces a fill cheaper than the recorded book allows."""
    result = walk_taker_fill(
        levels,
        PricePips(limit_price),
        ContractCentis(requested),
        _fee_model(rate_ppm),
        haircut_ppm=haircut_ppm,
        max_participation_ppm=max_participation_ppm,
    )

    assert result.book_cost == _reference_book_cost(levels, result.filled)


# --- (b): total_cost never understates book_cost + fee (haircut >= 0) -------


@given(
    levels=_ascending_levels(),
    limit_price=_PRICE_STRATEGY,
    requested=_QTY_STRATEGY,
    rate_ppm=_RATE_STRATEGY,
    haircut_ppm=_HAIRCUT_PPM_STRATEGY,
    max_participation_ppm=_PARTICIPATION_PPM_STRATEGY,
)
def test_total_cost_never_understates_book_cost_plus_fee(
    levels: tuple[OrderBookLevel, ...],
    limit_price: int,
    requested: int,
    rate_ppm: int,
    haircut_ppm: int,
    max_participation_ppm: int,
) -> None:
    """Total cost never falls below book cost plus fee, and the haircut is >= 0."""
    result = walk_taker_fill(
        levels,
        PricePips(limit_price),
        ContractCentis(requested),
        _fee_model(rate_ppm),
        haircut_ppm=haircut_ppm,
        max_participation_ppm=max_participation_ppm,
    )

    assert result.haircut.value >= 0
    assert result.total_cost.value >= result.book_cost.value + result.fee.value


# --- (c): filled <= participation cap <= eligible depth ---------------------


@given(
    levels=_ascending_levels(),
    limit_price=_PRICE_STRATEGY,
    requested=_QTY_STRATEGY,
    rate_ppm=_RATE_STRATEGY,
    haircut_ppm=_HAIRCUT_PPM_STRATEGY,
    max_participation_ppm=_PARTICIPATION_PPM_STRATEGY,
)
def test_filled_never_exceeds_the_participation_cap_or_eligible_depth(
    levels: tuple[OrderBookLevel, ...],
    limit_price: int,
    requested: int,
    rate_ppm: int,
    haircut_ppm: int,
    max_participation_ppm: int,
) -> None:
    """Filled quantity stays within the participation cap, itself within depth."""
    result = walk_taker_fill(
        levels,
        PricePips(limit_price),
        ContractCentis(requested),
        _fee_model(rate_ppm),
        haircut_ppm=haircut_ppm,
        max_participation_ppm=max_participation_ppm,
    )

    depth = _eligible_depth(levels, limit_price)
    cap = participation_cap(
        ContractCentis(depth), max_participation_ppm=max_participation_ppm
    )

    assert result.filled.value <= cap.value
    assert cap.value <= depth


# --- (d): raising haircut_ppm never lowers total_cost -----------------------


@given(
    levels=_ascending_levels(),
    limit_price=_PRICE_STRATEGY,
    requested=_QTY_STRATEGY,
    rate_ppm=_RATE_STRATEGY,
    lower_haircut_ppm=st.integers(min_value=0, max_value=500_000),
    haircut_delta=st.integers(min_value=0, max_value=500_000),
    max_participation_ppm=_PARTICIPATION_PPM_STRATEGY,
)
def test_total_cost_is_monotone_nondecreasing_in_haircut_ppm(
    levels: tuple[OrderBookLevel, ...],
    limit_price: int,
    requested: int,
    rate_ppm: int,
    lower_haircut_ppm: int,
    haircut_delta: int,
    max_participation_ppm: int,
) -> None:
    """The haircut is a pure cost adjustment: it never changes how much
    fills (fee/book_cost depend only on the consumed levels), so a higher
    haircut_ppm can only raise (or hold even) the total cost."""
    higher_haircut_ppm = lower_haircut_ppm + haircut_delta
    fee_model = _fee_model(rate_ppm)

    lower = walk_taker_fill(
        levels,
        PricePips(limit_price),
        ContractCentis(requested),
        fee_model,
        haircut_ppm=lower_haircut_ppm,
        max_participation_ppm=max_participation_ppm,
    )
    higher = walk_taker_fill(
        levels,
        PricePips(limit_price),
        ContractCentis(requested),
        fee_model,
        haircut_ppm=higher_haircut_ppm,
        max_participation_ppm=max_participation_ppm,
    )

    assert higher.filled == lower.filled
    assert higher.book_cost == lower.book_cost
    assert higher.fee == lower.fee
    assert higher.total_cost.value >= lower.total_cost.value


# --- (e): a touch-only print stream never fills a resting order ------------


@given(
    limit_price=_PRICE_STRATEGY,
    touch_quantities=st.lists(_QTY_STRATEGY, min_size=1, max_size=5),
    remaining=_QTY_STRATEGY,
    depth_at_or_better=_QTY_STRATEGY,
    max_participation_ppm=_PARTICIPATION_PPM_STRATEGY,
)
def test_touch_only_print_stream_never_fills_a_resting_order(
    limit_price: int,
    touch_quantities: list[int],
    remaining: int,
    depth_at_or_better: int,
    max_participation_ppm: int,
) -> None:
    """Every print in the stream trades exactly at the limit (a touch, never
    strictly through) -- no participation cap or remaining-size headroom can
    turn a touch into a fill."""
    prints = tuple(
        TradePrint(PricePips(limit_price), ContractCentis(qty), _TS)
        for qty in touch_quantities
    )

    fill = resting_fill_quantity(
        PricePips(limit_price),
        ContractCentis(remaining),
        prints,
        ContractCentis(depth_at_or_better),
        max_participation_ppm=max_participation_ppm,
    )

    assert fill == ContractCentis(0)


# --- (f): consumed slices never exceed their recorded level's own depth ----


@given(
    levels=_ascending_levels(),
    limit_price=_PRICE_STRATEGY,
    requested=_QTY_STRATEGY,
    rate_ppm=_RATE_STRATEGY,
    haircut_ppm=_HAIRCUT_PPM_STRATEGY,
    max_participation_ppm=_PARTICIPATION_PPM_STRATEGY,
)
def test_consumed_slices_never_exceed_their_recorded_level_depth(
    levels: tuple[OrderBookLevel, ...],
    limit_price: int,
    requested: int,
    rate_ppm: int,
    haircut_ppm: int,
    max_participation_ppm: int,
) -> None:
    """Because `levels` is strictly ascending, the eligible (at-or-better)
    levels are exactly a prefix, so `consumed` lines up positionally with
    that prefix of `levels`: same price, never more quantity than recorded."""
    result = walk_taker_fill(
        levels,
        PricePips(limit_price),
        ContractCentis(requested),
        _fee_model(rate_ppm),
        haircut_ppm=haircut_ppm,
        max_participation_ppm=max_participation_ppm,
    )

    for consumed_level, original_level in zip(result.consumed, levels, strict=False):
        assert consumed_level.price == original_level.price
        assert consumed_level.quantity.value <= original_level.quantity.value


# --- (g)/(h): a shared trade-through pool never double-counts real volume ----


@st.composite
def _resting_requests(draw: st.DrawFn) -> tuple[RestingFillRequest, ...]:
    """Build 1-4 same-side resting buys with independent limits/sizes/depths."""
    count = draw(st.integers(min_value=1, max_value=4))
    return tuple(
        RestingFillRequest(
            PricePips(draw(_PRICE_STRATEGY)),
            ContractCentis(draw(_QTY_STRATEGY)),
            ContractCentis(draw(_QTY_STRATEGY)),
        )
        for _ in range(count)
    )


@st.composite
def _trade_prints(draw: st.DrawFn) -> tuple[TradePrint, ...]:
    """Build 0-5 trade prints at arbitrary (price, quantity)."""
    count = draw(st.integers(min_value=0, max_value=5))
    return tuple(
        TradePrint(
            PricePips(draw(_PRICE_STRATEGY)), ContractCentis(draw(_QTY_STRATEGY)), _TS
        )
        for _ in range(count)
    )


@given(
    requests=_resting_requests(),
    prints=_trade_prints(),
    max_participation_ppm=_PARTICIPATION_PPM_STRATEGY,
)
def test_allocated_fills_never_exceed_recorded_trade_through_volume(
    requests: tuple[RestingFillRequest, ...],
    prints: tuple[TradePrint, ...],
    max_participation_ppm: int,
) -> None:
    """The pessimism invariant: the *sum* of every same-side resting fill can
    never exceed the volume that actually traded through some limit -- one
    real trade can never fill two orders in full (no self-flattering)."""
    allocated = allocate_resting_fills(
        requests, prints, max_participation_ppm=max_participation_ppm
    )

    highest_limit = max(request.limit.value for request in requests)
    reachable_volume = sum(
        print_.quantity.value for print_ in prints if print_.price.value < highest_limit
    )
    assert sum(fill.value for fill in allocated) <= reachable_volume


@given(
    requests=_resting_requests(),
    prints=_trade_prints(),
    max_participation_ppm=_PARTICIPATION_PPM_STRATEGY,
)
def test_shared_allocation_never_exceeds_an_isolated_resting_fill(
    requests: tuple[RestingFillRequest, ...],
    prints: tuple[TradePrint, ...],
    max_participation_ppm: int,
) -> None:
    """Sharing is only ever more pessimistic: no order's shared fill exceeds the
    fill it would have received alone against the full print stream."""
    allocated = allocate_resting_fills(
        requests, prints, max_participation_ppm=max_participation_ppm
    )

    for request, fill in zip(requests, allocated, strict=True):
        isolated = resting_fill_quantity(
            request.limit,
            request.remaining,
            prints,
            request.depth_at_or_better,
            max_participation_ppm=max_participation_ppm,
        )
        assert fill.value <= isolated.value
