"""Gate 1 RED, Hypothesis property suite for `windbreak.selector.sizing` (#45).

`windbreak/selector/sizing.py` does not exist yet, so importing it fails
collection with `ModuleNotFoundError: No module named 'windbreak.selector.sizing'`
-- the expected Gate 1 RED state for issue #45's sizing seam. The strategy
building a full `SelectorInputs` below also imports `PositionReadModelInput`
from `windbreak.selector.types`, which does not exist yet either -- the same
correct RED reason.

Five properties, from the narrowest (pure `dispersion_scale`/`kelly_size`
unit functions) to the broadest (`select` end-to-end over a generated book,
positions, and risk configuration):

    1. `kelly_size` is monotone non-decreasing in `net_edge_ppm`.
    2. `kelly_size` is exactly `ContractCentis(0)` whenever `net_edge_ppm <
       min_net_edge_ppm`, and strictly positive for a representative
       above-threshold input (guarding a degenerate always-zero
       implementation).
    3. `dispersion_scale` is monotone non-increasing in `d`, `g(0) ==
       1_000_000`, `g(d >= ceiling) == 0`, its range is always
       `[0, 1_000_000]`, and the pinned `ceiling <= 0` / `d < 0` behaviors
       hold for every input.
    4. Over generated books/exposures/risk configuration, a `select`-emitted
       size never exceeds any of the five notional-cap sizes, never
       violates participation at its own emitted marginal price, is always a
       whole-contract multiple (or zero), and is never negative.
    5. Whenever `select` emits an intent, the net edge recomputed at the
       emitted size is never below `min_net_edge_ppm` -- "never
       negative-EV-after-fees" is structural, not incidental.

Every generated value is a plain integer -- never a float -- per SPEC S6.1;
`windbreak.selector`/`windbreak.selector.sizing` are on
`scripts/lint_no_floats.py`'s denylist, and so is this test module's own
arithmetic (no bare `/`/`//`).
"""

from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from windbreak.config.schema import RiskConfig
from windbreak.connector.fees import FeeModel
from windbreak.connector.models import OrderBookLevel, OrderBookSnapshot
from windbreak.forecast.records import Citation, ForecastRecord
from windbreak.numeric import (
    ContractCentis,
    MoneyMicros,
    PricePips,
    RoundingDirection,
    divide,
)
from windbreak.selector import select
from windbreak.selector.correlation import (
    BUCKET_FED_POLICY,
    BucketExposureEntry,
    CorrelationTag,
)
from windbreak.selector.edge import EdgeFigures, compute_executable_edge
from windbreak.selector.sizing import (
    _depth_through_fill,
    _participation_fixed_point,
    dispersion_scale,
    kelly_size,
)
from windbreak.selector.types import (
    FeeModelInput,
    PositionReadModelInput,
    RiskConfigInput,
    SelectorInputs,
    SlippageModelInput,
)

#: A fixed reference instant every generated input is pinned to.
_INSTANT = datetime(2025, 3, 1, tzinfo=UTC)

_CITATION = Citation(
    url="https://example.com/sizing-property",
    content_hash="sha256:sizing-property-citation",
    quoted_text="A representative supporting citation for sizing properties.",
    publication_date=None,
    source_type="news_article",
)

#: A representative capital bound: wide enough to exercise a meaningfully
#: sized stake, narrow enough to keep hypothesis examples fast.
_CAPITAL_STRATEGY = st.integers(min_value=0, max_value=10**12)
_NET_EDGE_STRATEGY = st.integers(min_value=-1_000_000, max_value=1_000_000)
_PRICE_PPM_STRATEGY = st.integers(min_value=1, max_value=999_999)
_FRACTION_PPM_STRATEGY = st.integers(min_value=0, max_value=1_000_000)
_DISPERSION_D_STRATEGY = st.integers(min_value=-1_000_000, max_value=2_000_000)
_CEILING_STRATEGY = st.integers(min_value=-1_000_000, max_value=1_000_000)


# --- 1: kelly_size is monotone non-decreasing in net_edge_ppm ----------------


@given(
    net_edge_ppm=_NET_EDGE_STRATEGY,
    min_net_edge_ppm=st.integers(min_value=-1_000_000, max_value=1_000_000),
    executable_price_ppm=_PRICE_PPM_STRATEGY,
    kelly_fraction_ppm=_FRACTION_PPM_STRATEGY,
    dispersion_scale_ppm=st.integers(min_value=0, max_value=1_000_000),
    capital=_CAPITAL_STRATEGY,
    delta=st.integers(min_value=0, max_value=1_000_000),
)
def test_kelly_size_is_monotone_non_decreasing_in_net_edge(
    net_edge_ppm: int,
    min_net_edge_ppm: int,
    executable_price_ppm: int,
    kelly_fraction_ppm: int,
    dispersion_scale_ppm: int,
    capital: int,
    delta: int,
) -> None:
    """Increasing `net_edge_ppm` (all else fixed) never decreases the stake."""
    baseline = kelly_size(
        net_edge_ppm=net_edge_ppm,
        min_net_edge_ppm=min_net_edge_ppm,
        executable_price_ppm=executable_price_ppm,
        kelly_fraction_ppm=kelly_fraction_ppm,
        dispersion_scale_ppm=dispersion_scale_ppm,
        above_floor_capital_micros=MoneyMicros(capital),
    )
    increased = kelly_size(
        net_edge_ppm=net_edge_ppm + delta,
        min_net_edge_ppm=min_net_edge_ppm,
        executable_price_ppm=executable_price_ppm,
        kelly_fraction_ppm=kelly_fraction_ppm,
        dispersion_scale_ppm=dispersion_scale_ppm,
        above_floor_capital_micros=MoneyMicros(capital),
    )

    assert increased.value >= baseline.value


# --- 2: kelly_size's exactly-zero-below-threshold and existence guard -------


@given(
    min_net_edge_ppm=st.integers(min_value=0, max_value=1_000_000),
    shortfall=st.integers(min_value=1, max_value=1_000_000),
    executable_price_ppm=_PRICE_PPM_STRATEGY,
    kelly_fraction_ppm=st.integers(min_value=1, max_value=1_000_000),
    dispersion_scale_ppm=st.integers(min_value=1, max_value=1_000_000),
    capital=st.integers(min_value=1, max_value=10**12),
)
def test_kelly_size_is_exactly_zero_below_the_net_edge_floor(
    min_net_edge_ppm: int,
    shortfall: int,
    executable_price_ppm: int,
    kelly_fraction_ppm: int,
    dispersion_scale_ppm: int,
    capital: int,
) -> None:
    """`net_edge_ppm` strictly below `min_net_edge_ppm` always yields exactly
    `ContractCentis(0)`, for any other (sane) combination of inputs.
    """
    result = kelly_size(
        net_edge_ppm=min_net_edge_ppm - shortfall,
        min_net_edge_ppm=min_net_edge_ppm,
        executable_price_ppm=executable_price_ppm,
        kelly_fraction_ppm=kelly_fraction_ppm,
        dispersion_scale_ppm=dispersion_scale_ppm,
        above_floor_capital_micros=MoneyMicros(capital),
    )

    assert result == ContractCentis(0)


def test_kelly_size_is_strictly_positive_above_threshold_for_sane_inputs() -> None:
    """A representative, comfortably-above-threshold input yields a strictly
    positive stake -- guards a degenerate implementation that always returns
    zero regardless of input (which every property above would otherwise miss,
    since "zero" trivially satisfies both a monotonicity and a floor check).
    """
    result = kelly_size(
        net_edge_ppm=100_000,
        min_net_edge_ppm=30_000,
        executable_price_ppm=500_000,
        kelly_fraction_ppm=100_000,
        dispersion_scale_ppm=1_000_000,
        above_floor_capital_micros=MoneyMicros(1_000_000_000),
    )

    assert result.value > 0


# --- 3: dispersion_scale's shape invariants ----------------------------------


@given(
    d=_DISPERSION_D_STRATEGY,
    ceiling=_CEILING_STRATEGY,
    step=st.integers(min_value=0, max_value=10**6),
)
def test_dispersion_scale_is_monotone_non_increasing_in_d(
    d: int, ceiling: int, step: int
) -> None:
    """Increasing `d` (all else fixed) never increases the scale -- mirrors
    the issue's own example: `g(d, C) >= g(d + 1_000, C)` for any `d`/`C`.
    """
    assert dispersion_scale(d, ceiling) >= dispersion_scale(d + step, ceiling)


@given(ceiling=st.integers(min_value=1, max_value=1_000_000))
def test_dispersion_scale_at_zero_is_full_scale_for_any_positive_ceiling(
    ceiling: int,
) -> None:
    """`g(0, ceiling) == 1_000_000` for every positive ceiling."""
    assert dispersion_scale(0, ceiling) == 1_000_000


@given(
    ceiling=st.integers(min_value=1, max_value=1_000_000),
    overshoot=st.integers(min_value=0, max_value=1_000_000),
)
def test_dispersion_scale_at_or_past_the_ceiling_is_zero(
    ceiling: int, overshoot: int
) -> None:
    """`g(d, ceiling) == 0` for every `d >= ceiling`."""
    assert dispersion_scale(ceiling + overshoot, ceiling) == 0


@given(d=_DISPERSION_D_STRATEGY, ceiling=_CEILING_STRATEGY)
def test_dispersion_scale_range_is_always_zero_to_one_million(
    d: int, ceiling: int
) -> None:
    """`dispersion_scale`'s result is always within `[0, 1_000_000]`."""
    result = dispersion_scale(d, ceiling)

    assert 0 <= result <= 1_000_000


@given(ceiling=st.integers(min_value=-1_000_000, max_value=1_000_000))
def test_dispersion_scale_negative_d_is_always_full_scale(ceiling: int) -> None:
    """`d < 0` (never produced by a valid forecast, but defensively handled)
    is always treated as zero dispersion: full scale, for every ceiling.
    """
    assert dispersion_scale(-1, ceiling) == 1_000_000


@given(d=st.integers(min_value=1, max_value=2_000_000))
def test_dispersion_scale_non_positive_ceiling_with_positive_d_is_always_zero(
    d: int,
) -> None:
    """`ceiling <= 0` with `d > 0` is always fully zeroed, for every
    non-positive ceiling and every strictly positive `d`.
    """
    assert dispersion_scale(d, 0) == 0
    assert dispersion_scale(d, -1) == 0


# --- 4/5: end-to-end via select() over a generated book/positions/config ----


@st.composite
def _ascending_ask_levels(draw: st.DrawFn) -> tuple[OrderBookLevel, ...]:
    """Draw 1-3 strictly-increasing-price YES ask levels with ample depth.

    Sorting ascending guarantees "at-or-better than a limit" is exactly a
    prefix of the returned tuple, matching a real order book's invariant.
    """
    count = draw(st.integers(min_value=1, max_value=3))
    prices = draw(
        st.lists(
            st.integers(min_value=600, max_value=9_000),
            min_size=count,
            max_size=count,
            unique=True,
        )
    )
    prices.sort()
    quantities = draw(
        st.lists(
            st.integers(min_value=1_000, max_value=50_000),
            min_size=count,
            max_size=count,
        )
    )
    return tuple(
        OrderBookLevel(price=PricePips(p), quantity=ContractCentis(q))
        for p, q in zip(prices, quantities, strict=True)
    )


@st.composite
def _selector_inputs(draw: st.DrawFn) -> SelectorInputs:
    """Draw a valid, generously-passing-friendly `SelectorInputs`.

    Zero fee/slippage/research keeps the net edge exactly the
    probability/price gap, so the acceptance rate (and thus this property's
    statistical power) stays high without weakening any invariant checked.
    """
    asks = draw(_ascending_ask_levels())
    probability_ppm = draw(st.integers(min_value=400_000, max_value=1_000_000))
    vote_dispersion_ppm = draw(st.integers(min_value=0, max_value=150_000))

    kelly_fraction_ppm = draw(st.integers(min_value=10_000, max_value=500_000))
    dispersion_zero_ceiling_ppm = draw(
        st.integers(min_value=100_000, max_value=500_000)
    )
    max_participation_ppm = draw(st.integers(min_value=50_000, max_value=1_000_000))
    max_pos_market_pct_ppm = draw(st.integers(min_value=10_000, max_value=1_000_000))
    max_pos_event_pct_ppm = draw(st.integers(min_value=10_000, max_value=1_000_000))
    max_pos_bucket_pct_ppm = draw(st.integers(min_value=10_000, max_value=1_000_000))
    max_notional_per_day_micros = draw(st.integers(min_value=10**6, max_value=10**13))

    equity_micros = draw(st.integers(min_value=10**6, max_value=10**13))
    above_floor_capital_micros = draw(st.integers(min_value=0, max_value=equity_micros))
    market_exposure = draw(st.integers(min_value=0, max_value=equity_micros))
    event_exposure = draw(st.integers(min_value=0, max_value=equity_micros))
    bucket_exposure = draw(st.integers(min_value=0, max_value=equity_micros))
    total_exposure = draw(
        st.integers(
            min_value=max(market_exposure, event_exposure, bucket_exposure),
            max_value=equity_micros,
        )
    )
    total_deploy_cap_micros = draw(st.integers(min_value=0, max_value=equity_micros))
    notional_today = draw(
        st.integers(min_value=0, max_value=max_notional_per_day_micros)
    )

    forecast = ForecastRecord(
        forecast_id="fc-sizing-prop-0001",
        market_ticker="SIZING-PROP-TICKER",
        normalized_question_hash="sha256:sizing-prop-question",
        probability_ppm=probability_ppm,
        ci_low_ppm=0,
        ci_high_ppm=1,
        model_votes=(),
        vote_dispersion_ppm=vote_dispersion_ppm,
        rationale_markdown="n/a",
        citations=(_CITATION,),
        source_quality_notes=(),
        research_cost_micros=0,
        triage_stage="full",
        created_at=_INSTANT,
        forecast_horizon_hours=48,
        market_price_baseline_pips=asks[0].price.value,
        baseline_quote_snapshot_id="snap-sizing-prop-0001",
        coherence_group_sum_ppm=None,
        coherence_flag=False,
        abstention_reason=None,
        eligible_for_live=True,
    )
    order_book = OrderBookSnapshot(
        ticker="SIZING-PROP-TICKER",
        yes_bids=(),
        yes_asks=asks,
        fetched_at=_INSTANT,
    )
    fee_model = FeeModelInput(
        model=FeeModel(
            schedule_id="sizing-prop-fee-zero",
            maker_fee_ppm=0,
            taker_fee_ppm=0,
            settlement_fee_ppm=0,
        ),
        as_of=_INSTANT,
    )
    slippage_model = SlippageModelInput(
        model_id="sizing-prop-slippage-zero", per_contract_buffer_ppm=0
    )
    risk_config = RiskConfigInput(
        config=RiskConfig(
            kelly_fraction_ppm=kelly_fraction_ppm,
            dispersion_zero_ceiling_ppm=dispersion_zero_ceiling_ppm,
            max_participation_ppm=max_participation_ppm,
            max_pos_market_pct_ppm=max_pos_market_pct_ppm,
            max_pos_event_pct_ppm=max_pos_event_pct_ppm,
            max_pos_bucket_pct_ppm=max_pos_bucket_pct_ppm,
            max_notional_per_day_micros=max_notional_per_day_micros,
        ),
        config_hash="sha256:sizing-prop-risk",
    )
    positions = PositionReadModelInput(
        snapshot_id="positions-sizing-prop-0001",
        equity_micros=MoneyMicros(equity_micros),
        above_floor_capital_micros=MoneyMicros(above_floor_capital_micros),
        total_deploy_cap_micros=MoneyMicros(total_deploy_cap_micros),
        market_exposure=MoneyMicros(market_exposure),
        event_exposure=MoneyMicros(event_exposure),
        bucket_exposure=MoneyMicros(bucket_exposure),
        total_exposure=MoneyMicros(total_exposure),
        notional_today=MoneyMicros(notional_today),
    )
    # A single `fed-policy` peer carrying exactly `bucket_exposure` reproduces
    # that exposure through `select`'s SPEC S9.9 aggregation, so the value the
    # per-bucket cap actually clips against equals `positions.bucket_exposure`
    # (which `select` overrides with the identical aggregated figure) -- keeping
    # `_notional_cap_sizes`'s independent per-bucket cross-check exact.
    bucket_tag = CorrelationTag(
        bucket_id=BUCKET_FED_POLICY, source="llm", tagged_at=_INSTANT
    )
    bucket_peer = BucketExposureEntry(
        market_ticker="SIZING-PROP-PEER",
        exposure_micros=MoneyMicros(bucket_exposure),
        tags=(bucket_tag,),
    )
    return SelectorInputs(
        forecast=forecast,
        calibration_map_version="calib-sizing-prop-v1",
        order_book=order_book,
        fee_model=fee_model,
        slippage_model=slippage_model,
        positions=positions,
        risk_config=risk_config,
        correlation_tags=(bucket_tag,),
        bucket_peers=(bucket_peer,),
    )


def _depth_at_or_better(order_book: OrderBookSnapshot, marginal_price_pips: int) -> int:
    """Sum resting ask depth at a price at-or-better than `marginal_price_pips`.

    Args:
        order_book: The book to sum resting `yes_asks` depth over.
        marginal_price_pips: The limit price, in pips; a YES-buy level is
            "at-or-better" when its own price is `<=` this limit.

    Returns:
        The summed contract-centis depth across every qualifying level.
    """
    return sum(
        level.quantity.value
        for level in order_book.yes_asks
        if level.price.value <= marginal_price_pips
    )


def _notional_cap_sizes(inputs: SelectorInputs, executable_price_ppm: int) -> list[int]:
    """Recompute the five notional-cap sizes (in centis) `select` must honor.

    Independently re-derives each cap's headroom-to-centis conversion from
    `inputs`, per the architect's pinned formulas, so this is a genuine
    cross-check rather than a restatement of the implementation.

    Args:
        inputs: The selector inputs the emitted decision was computed from.
        executable_price_ppm: The final walk's executable price, in ppm.

    Returns:
        The five cap sizes, in contract-centis, in the pinned cap order
        (`per_market`, `per_event`, `per_bucket`, `total_deployed`,
        `daily_notional`).
    """
    positions = inputs.positions
    risk = inputs.risk_config.config

    def _pct_cap(pct_ppm: int, exposure: int) -> int:
        ceiling = divide(
            positions.equity_micros.value * pct_ppm,
            1_000_000,
            rounding=RoundingDirection.UNDERSTATE_EQUITY,
        )
        headroom = max(0, ceiling - exposure)
        return divide(
            headroom * 100,
            executable_price_ppm,
            rounding=RoundingDirection.UNDERSTATE_EQUITY,
        )

    def _absolute_cap(ceiling_micros: int, used_micros: int) -> int:
        headroom = max(0, ceiling_micros - used_micros)
        return divide(
            headroom * 100,
            executable_price_ppm,
            rounding=RoundingDirection.UNDERSTATE_EQUITY,
        )

    return [
        _pct_cap(risk.max_pos_market_pct_ppm, positions.market_exposure.value),
        _pct_cap(risk.max_pos_event_pct_ppm, positions.event_exposure.value),
        _pct_cap(risk.max_pos_bucket_pct_ppm, positions.bucket_exposure.value),
        _absolute_cap(
            positions.total_deploy_cap_micros.value, positions.total_exposure.value
        ),
        _absolute_cap(risk.max_notional_per_day_micros, positions.notional_today.value),
    ]


@given(inputs=_selector_inputs())
@settings(deadline=None, max_examples=50)
def test_emitted_size_respects_every_cap_and_is_a_nonnegative_whole_contract_multiple(
    inputs: SelectorInputs,
) -> None:
    """When `select` emits an intent, its size never exceeds any of the five
    notional-cap sizes, never violates participation at its own emitted
    marginal price, is always a whole-contract (i.e. a multiple of 100
    centis) multiple, and is never negative.
    """
    decision = select(inputs)
    if not decision.intents:
        return
    intent = decision.intents[0]
    size = intent.size.value

    assert size >= 0
    assert size % 100 == 0

    figures = compute_executable_edge(
        order_book=inputs.order_book,
        size=intent.size,
        forecast=inputs.forecast,
        fee_model=inputs.fee_model,
        slippage_model=inputs.slippage_model,
    )
    assert isinstance(figures, EdgeFigures)

    for cap_size in _notional_cap_sizes(inputs, figures.executable_price_ppm):
        assert size <= cap_size

    depth_at_or_better = _depth_at_or_better(inputs.order_book, intent.price.value)
    max_participation_ppm = inputs.risk_config.config.max_participation_ppm
    assert size * 1_000_000 <= depth_at_or_better * max_participation_ppm


@given(inputs=_selector_inputs())
@settings(deadline=None, max_examples=50)
def test_emitted_intent_is_never_negative_ev_after_fees(inputs: SelectorInputs) -> None:
    """When `select` emits an intent, the net edge recomputed at the emitted
    size is never below `min_net_edge_ppm` -- "never negative-EV-after-fees"
    is structural, not incidental to whatever size happened to be chosen.
    """
    decision = select(inputs)
    if not decision.intents:
        return
    intent = decision.intents[0]

    figures = compute_executable_edge(
        order_book=inputs.order_book,
        size=intent.size,
        forecast=inputs.forecast,
        fee_model=inputs.fee_model,
        slippage_model=inputs.slippage_model,
    )

    assert isinstance(figures, EdgeFigures)
    assert (
        figures.research_cost_adjusted_edge_ppm
        >= inputs.risk_config.config.min_net_edge_ppm
    )


@given(
    quantities=st.lists(
        st.integers(min_value=0, max_value=50_000), min_size=0, max_size=6
    ),
    max_participation_ppm=st.integers(min_value=1, max_value=1_000_000),
    size_centis=st.integers(min_value=0, max_value=500_000),
    lot=st.booleans(),
)
@settings(deadline=None, max_examples=300)
def test_participation_fixed_point_always_reaches_a_true_fixed_point(
    quantities: list[int],
    max_participation_ppm: int,
    size_centis: int,
    lot: bool,
) -> None:
    """The bounded iteration must *converge*, never merely run out of passes.

    Whatever it returns, the returned size takes no more than
    `max_participation_ppm` of the depth resting at-or-better than that very
    size's own marginal level -- the SPEC S9.6 participation invariant, restated
    at the answer rather than at the size it started from. This is the property
    the loop's iteration bound exists to guarantee: were the bound too small for
    some book, the loop would fall out mid-shrink and return an iterate still
    above its own cap, and this assertion is what catches it.

    The result is also never negative, never larger than the size asked for, and
    -- under `lot` -- always a whole-contract multiple.

    Args:
        quantities: Resting ask quantities, best-first, in contract-centis.
        max_participation_ppm: The participation share, in ppm.
        size_centis: The size to clip, in contract-centis.
        lot: Whether every iterate floors to a whole contract.
    """
    asks = tuple(
        OrderBookLevel(
            price=PricePips(4_000 + index), quantity=ContractCentis(quantity)
        )
        for index, quantity in enumerate(quantities)
    )

    result = _participation_fixed_point(
        size_centis, asks, max_participation_ppm, lot=lot
    )

    assert 0 <= result <= size_centis
    if lot:
        assert result % 100 == 0

    depth = _depth_through_fill(asks, result)
    cap = divide(
        max_participation_ppm * depth,
        1_000_000,
        rounding=RoundingDirection.UNDERSTATE_EQUITY,
    )
    if lot:
        cap = cap - (cap % 100)
    assert result <= cap
