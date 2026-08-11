"""Gate 1 RED tests for `windbreak.selector.execution_style` (issue #46).

`windbreak/selector/execution_style.py` does not exist yet, so every test below
fails collection with `ModuleNotFoundError: No module named
'windbreak.selector.execution_style'` -- the expected Gate 1 RED state for
issue #46's execution-style seam.

`decide_execution_style(inputs, size)` renders the architect's five-row
decision table into an `ExecutionStyleDecision`:

    row1: `yes_bids` empty OR `yes_asks` empty -> cross
    row2: spread (`best_ask.price - best_bid.price`, in pips) < 300 -> cross
    row3: rest price (`best_bid.price + 100`) outside
          `[min_open_price_pips, max_open_price_pips]` -> cross
    row4: net edge at the rest price < `min_net_edge_ppm` -> cross
    row5: else -> `rest_inside_spread`, priced at the rest price, carrying
          the risk config's `resting_order_ttl_seconds` /
          `cancel_on_move_ticks` verbatim

Every scenario below fixes the fee model and slippage buffer at zero and
uses the fixed test size 100 (contract-centis), at which
`_per_contract_ppm(total_micros, 100)` reduces to `total_micros` exactly (no
rounding remainder at any total, since multiplying by 100 then dividing by
100 cancels) -- so net edge at the rest price collapses to the clean
`probability_ppm - rest_price_pips * 100` used throughout this module's hand
computations. Two further tests exercise `select()` end-to-end: a wide-spread
all-pass scenario that rests (mirroring
`test_entry_conditions.py::test_select_emits_one_sized_intent_with_hand_
expected_deterministic_fields`'s own worked Kelly example, plus an added wide
bid level), and bundle A's own recorded (narrow-spread) book, which crosses.
"""

from __future__ import annotations

from datetime import UTC, datetime

from windbreak.config.schema import RiskConfig
from windbreak.connector.fees import FeeModel
from windbreak.connector.models import OrderBookLevel, OrderBookSnapshot
from windbreak.forecast.budget import FULL_PIPELINE_RESEARCH_COST_MICROS
from windbreak.forecast.records import Citation, ForecastRecord
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips
from windbreak.selector import SelectorInputs, select
from windbreak.selector.execution_style import (
    _REST_IMPROVEMENT_PIPS,
    _WIDE_SPREAD_MIN_PIPS,
    ExecutionStyleDecision,
    decide_execution_style,
)
from windbreak.selector.types import (
    FeeModelInput,
    PositionReadModelInput,
    RiskConfigInput,
    SlippageModelInput,
)

#: A fixed reference instant every timestamp in this module is pinned to;
#: freshness is not under test here, so nothing depends on its exact value.
_BASELINE_INSTANT = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

#: The fee model used everywhere in this module: a zero rate on every leg, so
#: `_fee_micros` is always exactly zero and net-edge-at-rest arithmetic
#: collapses to `probability_ppm - rest_price_pips * 100`.
_ZERO_FEE_MODEL = FeeModel(
    schedule_id="execstyle-test-fee-zero",
    maker_fee_ppm=0,
    taker_fee_ppm=0,
    settlement_fee_ppm=0,
)

#: The fixed size every `decide_execution_style` call below is evaluated at.
#: At 100 contract-centis, `_per_contract_ppm(total_micros, 100)` reduces to
#: `total_micros` exactly for any total (the `*100 / 100` scale cancels with
#: no remainder), keeping every hand computation in this module exact.
_TEST_SIZE = ContractCentis(100)

#: A deep resting quantity, used on every book level in this module so depth
#: never binds -- only price (spread, rest price, band) is ever under test.
_DEEP_QTY = ContractCentis(1_000_000)

_BASELINE_CITATION = Citation(
    url="https://example.com/execstyle-test",
    content_hash="sha256:execstyle-test-citation",
    quoted_text="Example quoted text supporting the execution-style forecast.",
    publication_date=None,
    source_type="news_article",
)


def _forecast(**overrides: object) -> ForecastRecord:
    """Build the baseline `ForecastRecord`: probability 500_000 ppm.

    Args:
        **overrides: Field values overriding the defaults below.

    Returns:
        The constructed, post-init-validated `ForecastRecord`.
    """
    defaults: dict[str, object] = {
        "forecast_id": "fc-execstyle-0001",
        "market_ticker": "EXECSTYLE-TICKER",
        "normalized_question_hash": "sha256:execstyle-question",
        "probability_ppm": 500_000,
        "ci_low_ppm": 100_000,
        "ci_high_ppm": 200_000,
        "model_votes": (),
        "vote_dispersion_ppm": 0,
        "rationale_markdown": "n/a",
        "citations": (_BASELINE_CITATION,),
        "source_quality_notes": (),
        "research_cost_micros": 0,
        "triage_stage": "full",
        "created_at": _BASELINE_INSTANT,
        "forecast_horizon_hours": 48,
        "market_price_baseline_pips": 4_000,
        "baseline_quote_snapshot_id": "snap-execstyle-0001",
        "coherence_group_sum_ppm": None,
        "coherence_flag": False,
        "abstention_reason": None,
        "eligible_for_live": True,
    }
    defaults.update(overrides)
    return ForecastRecord(**defaults)


def _order_book(**overrides: object) -> OrderBookSnapshot:
    """Build the default `OrderBookSnapshot`: bid 4_000 / ask 4_500 (wide,
    500-pip spread, both deep enough to never bind on size alone).

    Args:
        **overrides: Field values overriding the defaults below.

    Returns:
        The constructed `OrderBookSnapshot`.
    """
    defaults: dict[str, object] = {
        "ticker": "EXECSTYLE-TICKER",
        "yes_bids": (OrderBookLevel(price=PricePips(4_000), quantity=_DEEP_QTY),),
        "yes_asks": (OrderBookLevel(price=PricePips(4_500), quantity=_DEEP_QTY),),
        "fetched_at": _BASELINE_INSTANT,
    }
    defaults.update(overrides)
    return OrderBookSnapshot(**defaults)


def _fee_model(**overrides: object) -> FeeModelInput:
    """Build the baseline `FeeModelInput`: the zero-rate model above.

    Args:
        **overrides: Field values overriding the defaults below.

    Returns:
        The constructed `FeeModelInput`.
    """
    defaults: dict[str, object] = {
        "model": _ZERO_FEE_MODEL,
        "as_of": _BASELINE_INSTANT,
    }
    defaults.update(overrides)
    return FeeModelInput(**defaults)


def _slippage_model(**overrides: object) -> SlippageModelInput:
    """Build the baseline `SlippageModelInput`: a zero-ppm buffer.

    Args:
        **overrides: Field values overriding the defaults below.

    Returns:
        The constructed `SlippageModelInput`.
    """
    defaults: dict[str, object] = {
        "model_id": "execstyle-test-slippage-zero",
        "per_contract_buffer_ppm": 0,
    }
    defaults.update(overrides)
    return SlippageModelInput(**defaults)


def _positions(**overrides: object) -> PositionReadModelInput:
    """Build a generous `PositionReadModelInput` so no notional/participation
    cap ever binds in this module's `select()` integration scenarios.

    Args:
        **overrides: Field values overriding the generous defaults below.

    Returns:
        The constructed `PositionReadModelInput`.
    """
    defaults: dict[str, object] = {
        "snapshot_id": "positions-execstyle-0001",
        "equity_micros": MoneyMicros(1_000_000_000_000),
        "above_floor_capital_micros": MoneyMicros(1_000_000_000),
        "total_deploy_cap_micros": MoneyMicros(1_000_000_000_000),
        "market_exposure": MoneyMicros(0),
        "event_exposure": MoneyMicros(0),
        "bucket_exposure": MoneyMicros(0),
        "total_exposure": MoneyMicros(0),
        "notional_today": MoneyMicros(0),
    }
    defaults.update(overrides)
    return PositionReadModelInput(**defaults)


def _risk_config(**overrides: object) -> RiskConfigInput:
    """Build a `RiskConfigInput` over `RiskConfig` defaults plus overrides.

    Args:
        **overrides: `RiskConfig` field overrides.

    Returns:
        The constructed `RiskConfigInput`.
    """
    return RiskConfigInput(
        config=RiskConfig(**overrides), config_hash="sha256:risk-execstyle"
    )


def _inputs(
    *,
    forecast: ForecastRecord | None = None,
    order_book: OrderBookSnapshot | None = None,
    fee_model: FeeModelInput | None = None,
    slippage_model: SlippageModelInput | None = None,
    positions: PositionReadModelInput | None = None,
    risk_config: RiskConfigInput | None = None,
) -> SelectorInputs:
    """Assemble `SelectorInputs` from the baseline builders plus overrides.

    Args:
        forecast: Overriding `ForecastRecord`, or `None` for the baseline.
        order_book: Overriding `OrderBookSnapshot`, or `None` for the
            default wide (500-pip) bid/ask book.
        fee_model: Overriding `FeeModelInput`, or `None` for the zero-rate
            baseline.
        slippage_model: Overriding `SlippageModelInput`, or `None` for the
            zero-buffer baseline.
        positions: Overriding `PositionReadModelInput`, or `None` for the
            generous baseline.
        risk_config: Overriding `RiskConfigInput`, or `None` for unmodified
            `RiskConfig` defaults.

    Returns:
        The constructed `SelectorInputs`.
    """
    return SelectorInputs(
        forecast=forecast if forecast is not None else _forecast(),
        calibration_map_version="calib-execstyle-v1",
        order_book=order_book if order_book is not None else _order_book(),
        fee_model=fee_model if fee_model is not None else _fee_model(),
        slippage_model=(
            slippage_model if slippage_model is not None else _slippage_model()
        ),
        positions=positions if positions is not None else _positions(),
        risk_config=risk_config if risk_config is not None else _risk_config(),
        correlation_tags=(),
    )


# --- Fenced constants ---------------------------------------------------------


def test_fenced_constants_have_the_pinned_values() -> None:
    """The two fenced constants driving the decision table are pinned
    exactly: a 300-pip wide-spread floor and a 100-pip rest improvement.
    """
    assert _WIDE_SPREAD_MIN_PIPS == 300
    assert _REST_IMPROVEMENT_PIPS == 100


def test_decide_execution_style_returns_an_execution_style_decision() -> None:
    """`decide_execution_style` returns the dedicated result type, not a bare
    tuple or dict."""
    inputs = _inputs()

    decision = decide_execution_style(inputs, _TEST_SIZE)

    assert isinstance(decision, ExecutionStyleDecision)


# --- Row 1: empty bids or asks --------------------------------------------------


def test_decide_cross_when_yes_bids_is_empty() -> None:
    """Empty `yes_bids` forces `cross` regardless of a deep, wide ask book --
    there is no resting bid to improve on, so row 1 fires before spread,
    band, or edge are ever considered.
    """
    inputs = _inputs(order_book=_order_book(yes_bids=()))

    decision = decide_execution_style(inputs, _TEST_SIZE)

    assert decision.style == "cross"
    assert decision.resting_price_pips is None
    assert decision.resting_ttl_seconds is None
    assert decision.cancel_on_move_ticks is None


def test_decide_cross_when_yes_asks_is_empty() -> None:
    """Empty `yes_asks` likewise forces `cross` -- the mirror of the
    empty-bids case above."""
    inputs = _inputs(order_book=_order_book(yes_asks=()))

    decision = decide_execution_style(inputs, _TEST_SIZE)

    assert decision.style == "cross"
    assert decision.resting_price_pips is None


# --- Row 2: spread narrower than the wide-spread floor -------------------------


def test_decide_cross_when_spread_is_narrower_than_the_wide_spread_floor() -> None:
    """Bid 4_000 / ask 4_200 -> spread 200, strictly below the 300-pip
    `_WIDE_SPREAD_MIN_PIPS` floor -- row 2 fires: cross.
    """
    inputs = _inputs(
        order_book=_order_book(
            yes_bids=(OrderBookLevel(price=PricePips(4_000), quantity=_DEEP_QTY),),
            yes_asks=(OrderBookLevel(price=PricePips(4_200), quantity=_DEEP_QTY),),
        )
    )

    decision = decide_execution_style(inputs, _TEST_SIZE)

    assert decision.style == "cross"


def test_decide_cross_one_pip_below_the_wide_spread_boundary() -> None:
    """Bid 4_000 / ask 4_299 -> spread 299, one pip short of the inclusive
    300-pip boundary -- still row 2: cross.
    """
    inputs = _inputs(
        order_book=_order_book(
            yes_bids=(OrderBookLevel(price=PricePips(4_000), quantity=_DEEP_QTY),),
            yes_asks=(OrderBookLevel(price=PricePips(4_299), quantity=_DEEP_QTY),),
        )
    )

    decision = decide_execution_style(inputs, _TEST_SIZE)

    assert decision.style == "cross"


def test_decide_rests_at_the_inclusive_wide_spread_boundary() -> None:
    """Bid 4_000 / ask 4_300 -> spread exactly 300, the inclusive
    `_WIDE_SPREAD_MIN_PIPS` boundary -- wide enough to rest, not merely
    strictly greater. Rest price = 4_000 + 100 = 4_100 pips (inside the
    default [500, 9_500] band). Net edge at the rest price (zero fee/
    slippage/research at this test's fixed size):
        net_edge_ppm = 550_000 - 4_100*100 = 550_000 - 410_000 = 140_000
    comfortably clears the default `min_net_edge_ppm` (30_000).
    """
    inputs = _inputs(
        forecast=_forecast(probability_ppm=550_000),
        order_book=_order_book(
            yes_bids=(OrderBookLevel(price=PricePips(4_000), quantity=_DEEP_QTY),),
            yes_asks=(OrderBookLevel(price=PricePips(4_300), quantity=_DEEP_QTY),),
        ),
    )

    decision = decide_execution_style(inputs, _TEST_SIZE)

    assert decision.style == "rest_inside_spread"
    assert decision.resting_price_pips == PricePips(4_100)
    assert decision.resting_ttl_seconds == 900
    assert decision.cancel_on_move_ticks == 2


# --- Row 3: wide spread, rest price outside the open-price band ---------------


def test_decide_cross_when_the_rest_price_falls_below_the_open_price_band() -> None:
    """Bid 350 / ask 700 -> spread 350 (wide), but the rest price
    350 + 100 = 450 pips sits below the default 500-pip open-price floor --
    row 3 fires: cross, even though the spread alone would qualify as wide.
    """
    inputs = _inputs(
        order_book=_order_book(
            yes_bids=(OrderBookLevel(price=PricePips(350), quantity=_DEEP_QTY),),
            yes_asks=(OrderBookLevel(price=PricePips(700), quantity=_DEEP_QTY),),
        )
    )

    decision = decide_execution_style(inputs, _TEST_SIZE)

    assert decision.style == "cross"
    assert decision.resting_price_pips is None


# --- Row 4: net edge at the rest price vs. the floor ---------------------------


def test_decide_rests_when_net_edge_at_rest_exactly_equals_the_floor() -> None:
    """The default wide book (bid 4_000/ask 4_500, spread 500) rests at
    4_100 pips, inside the band. With `probability_ppm=440_000` and zero
    fee/slippage/research at this test's fixed size:
        net_edge_ppm = 440_000 - 4_100*100 = 440_000 - 410_000 = 30_000
    exactly the default `min_net_edge_ppm` (30_000) -- the row-4/row-5
    boundary is inclusive (only a strictly-below edge crosses), so this
    rests.
    """
    inputs = _inputs(forecast=_forecast(probability_ppm=440_000))

    decision = decide_execution_style(inputs, _TEST_SIZE)

    assert decision.style == "rest_inside_spread"
    assert decision.resting_price_pips == PricePips(4_100)


def test_decide_cross_when_net_edge_at_rest_is_one_ppm_below_the_floor() -> None:
    """One ppm below the boundary above (`probability_ppm=439_999` ->
    net_edge_ppm = 29_999) falls strictly below `min_net_edge_ppm` -- row 4
    fires: cross.
    """
    inputs = _inputs(forecast=_forecast(probability_ppm=439_999))

    decision = decide_execution_style(inputs, _TEST_SIZE)

    assert decision.style == "cross"
    assert decision.resting_price_pips is None


def test_the_rest_price_edge_carries_no_research_haircut() -> None:
    """`research_cost_micros` moves the rest/cross decision by exactly nothing.

    Issue #483 removed the research term from the entry edge
    (`windbreak.selector.edge`). `_net_edge_at_price_ppm` charges the resting
    price the *same* haircuts, so it had to lose the term too -- had only one
    side dropped it, a rest price would be judged against a systematically
    harsher edge than the cross price it is compared with, and
    `rest_inside_spread` would quietly stop being reachable at the boundary.

    The scenario is the exact row-4/row-5 boundary above: a net edge of 30_000
    ppm against the default `min_net_edge_ppm` of 30_000. Any haircut at all --
    the shipped charge amortized over this size is 3_000_000 ppm, but even one
    micro would do -- pushes it strictly below the floor and flips the verdict
    to `cross`. So this pins the absence of the term at the one point where its
    presence is detectable, rather than at a comfortable distance from any
    boundary.
    """
    charged = _inputs(
        forecast=_forecast(
            probability_ppm=440_000,
            research_cost_micros=FULL_PIPELINE_RESEARCH_COST_MICROS,
        )
    )
    free = _inputs(forecast=_forecast(probability_ppm=440_000))

    decision = decide_execution_style(charged, _TEST_SIZE)

    assert FULL_PIPELINE_RESEARCH_COST_MICROS > 0
    assert decision.style == "rest_inside_spread"
    assert decision.resting_price_pips == PricePips(4_100)
    assert decision == decide_execution_style(free, _TEST_SIZE)


# --- Row 5: happy path, with config threading ----------------------------------


def test_decide_rests_with_hand_computed_price_ttl_and_move_and_threads_config() -> (
    None
):
    """The default-book happy path (bid 4_000/ask 4_500, wide 500-pip
    spread, probability 500_000, zero fee/slippage/research) rests at
    4_100 pips with net edge 90_000 ppm (500_000 - 4_100*100), comfortably
    above the floor, and threads the risk config's
    `resting_order_ttl_seconds` / `cancel_on_move_ticks` verbatim into the
    decision -- overridden here to 1_234 / 5 (distinct from the `RiskConfig`
    defaults 900 / 2) so a stale hard-coded constant in the implementation
    cannot pass by coincidence.
    """
    inputs = _inputs(
        risk_config=_risk_config(
            resting_order_ttl_seconds=1_234, cancel_on_move_ticks=5
        )
    )

    decision = decide_execution_style(inputs, _TEST_SIZE)

    assert decision.style == "rest_inside_spread"
    assert decision.resting_price_pips == PricePips(4_100)
    assert decision.resting_ttl_seconds == 1_234
    assert decision.cancel_on_move_ticks == 5


# --- select()-level integration -------------------------------------------------


def test_select_emits_a_resting_intent_on_an_all_pass_wide_spread_scenario() -> None:
    """The default `_inputs()` scenario is shaped to hand-verifiably pass
    every SPEC S9.3 entry condition and Kelly-size to a real, nonzero
    position -- mirroring `test_entry_conditions.py`'s own worked example
    (probability 500_000, single deep 4_500-pip ask level, zero fee/
    slippage/research, generous positions) with an added deep 4_000-pip bid
    level -- so `select()`'s post-sizing `decide_execution_style` call has a
    genuine wide (500-pip) spread to rest inside.

    Probe-size (100-centi) entry-check figures (ask level 4_500/1_000_000,
    zero fee/slippage/research, probability 500_000): executable_price_ppm =
    450_000 (exact); net_edge_ppm = 50_000 >= 30_000 -> net_edge_min passes;
    annualized clears the hurdle by orders of magnitude; CI
    [100_000, 200_000] does not straddle 450_000; price 4_500 pips is in
    [500, 9_500] -- all twelve SPEC S9.3 conditions pass.

    Kelly sizing (identical arithmetic to `test_entry_conditions.py::test_
    select_emits_one_sized_intent_with_hand_expected_deterministic_fields`):
    g=1_000_000 (zero dispersion), stake_micros=9_090_909,
    raw_size_centis=2_020, no cap binds, floor-to-100 quantization -> final
    size 2_000. Re-walking the same flat 4_500-pip ask level at size=2_000:
    cost=9_000_000 micros; net_edge_at_final_size=50_000 >= 30_000 -> the
    final-size guard passes.

    `decide_execution_style` at size=2_000, over the added 4_000-pip bid
    level (spread = 4_500-4_000 = 500 >= 300, wide): rest price =
    4_000 + 100 = 4_100 pips (inside [500, 9_500]); net edge at the rest
    price (zero fee/slippage/research): 500_000 - 4_100*100 = 90_000 >=
    30_000 -> rests, with the default `RiskConfig` ttl (900) and
    cancel-on-move (2) threaded through.
    """
    inputs = _inputs()

    decision = select(inputs)

    assert len(decision.intents) == 1
    intent = decision.intents[0]
    assert intent.execution_style == "rest_inside_spread"
    assert intent.price == PricePips(4_100)
    assert intent.resting_ttl_seconds == 900
    assert intent.cancel_on_move_ticks == 2


def test_select_emits_a_crossing_intent_on_the_bundle_a_shaped_book(
    recorded_inputs_bundle_a: SelectorInputs,
) -> None:
    """Bundle A's own recorded book (bids best 4_400, asks best 4_600 --
    spread 200, narrower than the 300-pip wide-spread floor) already
    hand-verifiably passes all twelve SPEC S9.3 conditions and emits one
    sized intent (see `test_determinism_golden.py`); this pins that its
    narrow spread keeps `decide_execution_style` on the `cross` branch, with
    both resting fields left `None`.
    """
    decision = select(recorded_inputs_bundle_a)

    assert len(decision.intents) == 1
    intent = decision.intents[0]
    assert intent.execution_style == "cross"
    assert intent.resting_ttl_seconds is None
    assert intent.cancel_on_move_ticks is None
