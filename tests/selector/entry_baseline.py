"""The hand-verified all-pass selector baseline, shared by the entry tests.

Extracted verbatim from ``tests/selector/test_entry_conditions.py`` (issue #44)
when issue #483 added a second module that needs the same scenario. Two copies
of a hand-derived baseline drift, and a drifted baseline is worse than none: it
still passes, against a scenario nobody re-derived.

Baseline hand computation (probe size 100 centis, single ask level 5 000
pips / 1 000 centis, so the walk fills entirely at 5 000, exact)::

    executable_price_pips = 5000; executable_price_ppm = 500_000 (exact)
    gross_edge_ppm        = 600_000 - 500_000 = 100_000
    trading fee: rate=10_000, numerator=10_000*100*5000*5000
                 = 25_000_000_000_000
                 cents = ceil(25_000_000_000_000/1e14) = 1 -> 10_000 micros
                 fee_ppm = 10_000 (exact)
    fee_adjusted_edge_ppm = 100_000 - 10_000 = 90_000
    net_edge_ppm          = 90_000 - 2_000 (slippage buffer) = 88_000
    annualized            = floor(88_000*1_000_000*8760 / (500_000*48))
                          = floor(770_880_000_000_000 / 24_000_000)
                          = 32_120_000 (exact)

``net_edge_ppm`` is the last link in the chain since issue #483: the research
haircut that used to follow the slippage buffer is gone, because research cost
is a per-forecast cost and the probe is one contract. Governance of that spend
moved to the per-UTC-day ceiling.

At those figures every SPEC S9.3 condition passes: 88 000 clears
``min_net_edge_ppm`` (30 000); 32 120 000 clears ``annualized_hurdle_ppm +
idle_cash_apr_ppm`` (240 000); the CI [100 000, 200 000] does not straddle
500 000; ``T`` equals every source timestamp so each freshness check sees zero
age; the market is coherent, cited, priced inside [500, 9 500] and live
eligible.
"""

from __future__ import annotations

from datetime import UTC, datetime

from windbreak.config.schema import RiskConfig
from windbreak.connector.fees import FeeModel
from windbreak.connector.models import OrderBookLevel, OrderBookSnapshot
from windbreak.forecast.records import Citation, ForecastRecord
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips
from windbreak.selector import SelectorInputs
from windbreak.selector.edge import EdgeFigures, compute_executable_edge
from windbreak.selector.types import (
    FeeModelInput,
    PositionReadModelInput,
    RiskConfigInput,
    SlippageModelInput,
)

#: The reference instant every baseline timestamp (order book, forecast,
#: fee model) is pinned to, so `T = max(...)` collapses to this single value
#: and every freshness check starts from a known-fresh state.
BASELINE_INSTANT = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

#: ``select``'s fixed probe size (SPEC S9.1, issue #44): every entry-condition
#: test computes ``EdgeFigures`` at this same size, matching what ``select``
#: itself uses internally. Restated rather than imported from the selector's
#: module-private ``_PROBE_SIZE_CENTIS``; ``test_probe_size_matches_the_selector``
#: pins the two equal.
PROBE_SIZE = ContractCentis(100)

#: The single supporting citation every baseline forecast carries.
BASELINE_CITATION = Citation(
    url="https://example.com/entry-test",
    content_hash="sha256:entry-test-citation",
    quoted_text="Example quoted text supporting the baseline forecast.",
    publication_date=None,
    source_type="news_article",
)


def baseline_forecast(**overrides: object) -> ForecastRecord:
    """Build the baseline ``ForecastRecord``: probability 600 000, all-pass.

    Args:
        **overrides: Field values overriding the defaults below.

    Returns:
        The constructed, post-init-validated ``ForecastRecord``.
    """
    defaults: dict[str, object] = {
        "forecast_id": "fc-entry-0001",
        "market_ticker": "ENTRY-TICKER",
        "normalized_question_hash": "sha256:entry-question",
        "probability_ppm": 600_000,
        "ci_low_ppm": 100_000,
        "ci_high_ppm": 200_000,
        "model_votes": (),
        "vote_dispersion_ppm": 0,
        "rationale_markdown": "n/a",
        "citations": (BASELINE_CITATION,),
        "source_quality_notes": (),
        "research_cost_micros": 0,
        "triage_stage": "full",
        "created_at": BASELINE_INSTANT,
        "forecast_horizon_hours": 48,
        "market_price_baseline_pips": 5_000,
        "baseline_quote_snapshot_id": "snap-entry-0001",
        "coherence_group_sum_ppm": None,
        "coherence_flag": False,
        "abstention_reason": None,
        "eligible_for_live": True,
    }
    defaults.update(overrides)
    return ForecastRecord(**defaults)


def baseline_order_book(**overrides: object) -> OrderBookSnapshot:
    """Build the baseline ``OrderBookSnapshot``: one deep ask at 5 000 pips.

    Args:
        **overrides: Field values overriding the defaults below.

    Returns:
        The constructed ``OrderBookSnapshot``.
    """
    defaults: dict[str, object] = {
        "ticker": "ENTRY-TICKER",
        "yes_bids": (),
        "yes_asks": (
            OrderBookLevel(price=PricePips(5_000), quantity=ContractCentis(1_000)),
        ),
        "fetched_at": BASELINE_INSTANT,
    }
    defaults.update(overrides)
    return OrderBookSnapshot(**defaults)


def baseline_fee_model(**overrides: object) -> FeeModelInput:
    """Build the baseline ``FeeModelInput``: a 10 000 ppm taker rate.

    Args:
        **overrides: Field values overriding the defaults below.

    Returns:
        The constructed ``FeeModelInput``.
    """
    defaults: dict[str, object] = {
        "model": FeeModel(
            schedule_id="entry-test-fee",
            maker_fee_ppm=0,
            taker_fee_ppm=10_000,
            settlement_fee_ppm=0,
        ),
        "as_of": BASELINE_INSTANT,
    }
    defaults.update(overrides)
    return FeeModelInput(**defaults)


def baseline_slippage_model(**overrides: object) -> SlippageModelInput:
    """Build the baseline ``SlippageModelInput``: a 2 000 ppm buffer.

    Args:
        **overrides: Field values overriding the defaults below.

    Returns:
        The constructed ``SlippageModelInput``.
    """
    defaults: dict[str, object] = {
        "model_id": "entry-test-slippage",
        "per_contract_buffer_ppm": 2_000,
    }
    defaults.update(overrides)
    return SlippageModelInput(**defaults)


def baseline_positions(**overrides: object) -> PositionReadModelInput:
    """Build the baseline ``PositionReadModelInput``.

    Generous enough (huge equity/deploy-cap, zero exposures/notional) that none
    of the five notional caps or the participation cap ever bind in these
    scenarios -- only Kelly sizing (issue #45) or the fixed floor-to-100
    quantization determine the emitted size.

    Args:
        **overrides: Field values overriding the generous defaults below.

    Returns:
        The constructed ``PositionReadModelInput``.
    """
    defaults: dict[str, object] = {
        "snapshot_id": "positions-entry-0001",
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


def baseline_risk_config(**overrides: object) -> RiskConfigInput:
    """Build the baseline ``RiskConfigInput`` over unmodified ``RiskConfig`` defaults.

    Args:
        **overrides: ``RiskConfig`` field overrides.

    Returns:
        The constructed ``RiskConfigInput``, ``config_hash`` fixed for this suite.
    """
    return RiskConfigInput(
        config=RiskConfig(**overrides), config_hash="sha256:risk-entry"
    )


def baseline_inputs(
    *,
    forecast: ForecastRecord | None = None,
    order_book: OrderBookSnapshot | None = None,
    fee_model: FeeModelInput | None = None,
    slippage_model: SlippageModelInput | None = None,
    positions: PositionReadModelInput | None = None,
    risk_config: RiskConfigInput | None = None,
) -> SelectorInputs:
    """Assemble the baseline ``SelectorInputs``, all twelve conditions passing.

    Args:
        forecast: Overriding ``ForecastRecord``, or ``None`` for the baseline.
        order_book: Overriding ``OrderBookSnapshot``, or ``None`` for the baseline.
        fee_model: Overriding ``FeeModelInput``, or ``None`` for the baseline.
        slippage_model: Overriding ``SlippageModelInput``, or ``None`` for the
            baseline.
        positions: Overriding ``PositionReadModelInput``, or ``None`` for the
            generous baseline (issue #45; no cap ever binds by default).
        risk_config: Overriding ``RiskConfigInput``, or ``None`` for the baseline.

    Returns:
        The constructed ``SelectorInputs``.
    """
    return SelectorInputs(
        forecast=forecast if forecast is not None else baseline_forecast(),
        calibration_map_version="calib-entry-v1",
        order_book=order_book if order_book is not None else baseline_order_book(),
        fee_model=fee_model if fee_model is not None else baseline_fee_model(),
        slippage_model=(
            slippage_model if slippage_model is not None else baseline_slippage_model()
        ),
        positions=positions if positions is not None else baseline_positions(),
        risk_config=(
            risk_config if risk_config is not None else baseline_risk_config()
        ),
        correlation_tags=(),
    )


def probe_figures(inputs: SelectorInputs) -> EdgeFigures:
    """Compute ``EdgeFigures`` for ``inputs`` at the fixed probe size.

    Args:
        inputs: The selector inputs to compute edge figures for.

    Returns:
        The computed ``EdgeFigures``.

    Raises:
        AssertionError: If the probe fill declined rather than pricing, which
            would make every figure assertion downstream vacuous.
    """
    figures = compute_executable_edge(
        order_book=inputs.order_book,
        size=PROBE_SIZE,
        forecast=inputs.forecast,
        fee_model=inputs.fee_model,
        slippage_model=inputs.slippage_model,
    )
    assert isinstance(figures, EdgeFigures), f"probe declined: {figures!r}"
    return figures
