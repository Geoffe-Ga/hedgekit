"""The final-size net-edge gate in `windbreak.selector.select` (issue #249).

`select` prices a fixed 100-centis *probe* fill to run the twelve SPEC S9.3
entry conditions, then sizes the position and **re-prices the fill at that final
size**. A fill larger than the probe walks deeper into the book, so its
size-weighted price is worse and its net edge is smaller. The last gate before
an order is emitted (SPEC S9.5) rejects a trade whose edge no longer clears
`risk.min_net_edge_ppm` once the executable size is fixed -- the
`fail:net_edge_at_final_size` decline. Before this module, no test in the suite
ever took that arm: the whole `windbreak/selector/__init__.py` module sat at
88.89% with exactly that branch missing.

Every test here shares one book and one forecast, and varies only
`min_net_edge_ppm`. That isolation matters: a gate that always rejects is as
wrong as one that never does, so the boundary is pinned from **both** sides with
exact values.

Three separate haircuts are deliberately non-zero and mutually distinct -- a
1%-of-payout settlement fee, a 2_000-ppm slippage buffer, and a 3_000-micro
research cost. With all three at zero the four `EdgeFigures` edges collapse onto
one number, and a gate mutated to read `gross_edge_ppm` or
`slippage_adjusted_edge_ppm` instead of the research-cost-adjusted net edge
survives every test unchanged; that mutant was observed surviving before these
haircuts were added.

The book -- `yes_asks = ((4_500 pips, 100 centis), (5_000 pips, 900 centis))`,
no bids -- and the hand-derived arithmetic every assertion below rests on:

    probe (100 centis, level 1 only)
        cost      = 4_500 * 100                       = 450_000 micros
        price     = ceil(450_000 * 100 / 100)         = 450_000 ppm (4_500 pips)
        gross     = 500_000 - 450_000                 =  50_000 ppm
        fee       = ceil(10_000 * 100 / 10**6) cents  =  10_000 micros
                  -> ceil(10_000 * 100 / 100)         =  10_000 ppm
        research  = ceil(3_000 * 100 / 100)           =   3_000 ppm
        net       = 50_000 - 10_000 - 2_000 - 3_000   =  35_000 ppm
            -- clears every floor exercised below, so all twelve entry
               conditions pass and sizing runs.

    sizing (against the probe's figures)
        g(0, 200_000)                                 = 1_000_000 ppm
        stake     = floor(1_000_000_000 * 35_000 * 100_000 * 1_000_000
                          / ((1_000_000 - 450_000) * 10**12))
                  = floor(7_000_000_000 / 1_100)      = 6_363_636 micros
        raw       = floor(6_363_636 * 100 / 450_000)  =       1_414 centis
        participation clamp = floor(250_000 * 1_000 / 1_000_000) = 250 centis,
            which floors to the 200-centis whole-contract lot -- the binding cap
            is `participation`, and every notional cap (smallest: daily, 100_000
            centis at the 500_000-ppm deepest-ask reference price) is far larger.
        final     =                                             200 centis

    final size (200 centis: 100 @ 4_500 + 100 @ 5_000)
        cost      = 450_000 + 500_000                 = 950_000 micros
        price     = ceil(950_000 * 100 / 200)         = 475_000 ppm (4_750 pips)
        gross     = 500_000 - 475_000                 =  25_000 ppm
        fee       = ceil(10_000 * 200 / 10**6) cents  =  20_000 micros
                  -> ceil(20_000 * 100 / 200)         =  10_000 ppm
        fee-adj   = 25_000 - 10_000                   =  15_000 ppm
        slip-adj  = 15_000 - 2_000                    =  13_000 ppm
        research  = ceil(3_000 * 100 / 200)           =   1_500 ppm
        net       = 13_000 - 1_500                    =  11_500 ppm

The four final-size edges (25_000 / 15_000 / 13_000 / 11_500) are four different
numbers, as are the two net edges (probe 35_000, final 11_500) and the five
sizes in play (probe 100, raw 1_414, continuous participation 250, emitted 200,
book depth 1_000). No assertion below can pass by two quantities coinciding.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from windbreak.config.schema import RiskConfig
from windbreak.connector.fees import FeeModel
from windbreak.connector.models import OrderBookLevel, OrderBookSnapshot
from windbreak.forecast.records import Citation, ForecastRecord
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips
from windbreak.selector import SelectorInputs, select
from windbreak.selector.edge import EdgeFigures, compute_executable_edge
from windbreak.selector.types import (
    FeeModelInput,
    PositionReadModelInput,
    RiskConfigInput,
    SlippageModelInput,
)

#: A fixed reference instant every timestamp in this module is pinned to, so the
#: freshness conditions compare zero-length ages and never a wall clock.
_INSTANT = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

#: The probe size `select` prices the entry conditions against, in centis.
_PROBE_SIZE_CENTIS = 100

#: The net edge the *probe* fill prices at, in ppm (see the module docstring).
_PROBE_NET_EDGE_PPM = 35_000

#: The net edge the *final*, sized fill prices at, in ppm. A third of the
#: probe's, because the sized fill walks into the 5_000-pip level.
_FINAL_NET_EDGE_PPM = 11_500

#: The emitted size the sizing stage settles on, in contract-centis.
_FINAL_SIZE_CENTIS = 200

#: The pinned sizing reason the emit path appends (hand-derived in the module
#: docstring): raw 1_414 centis, full dispersion scale, participation binding.
_SIZING_REASON = (
    "sizing: raw_centis=1414 g_ppm=1000000 binding_cap=participation final_centis=200"
)

#: The twelve SPEC S9.3 entry conditions plus exactly one trailing decision
#: reason -- the shape both the accept and the reject path produce here.
_EXPECTED_REASON_COUNT = 13

_CITATION = Citation(
    url="https://example.com/final-size-net-edge",
    content_hash="sha256:final-size-net-edge-citation",
    quoted_text="Example quoted text supporting the final-size-gate forecast.",
    publication_date=None,
    source_type="news_article",
)


def _reject_reason(min_net_edge_ppm: int) -> str:
    """Render the exact decline reason the final-size gate emits.

    Args:
        min_net_edge_ppm: The configured net-edge floor, in ppm.

    Returns:
        The full ``fail:net_edge_at_final_size: ...`` reason string, naming the
        final-size net edge first and the floor second (the operand order is
        asserted, so a swapped detail is a failure, not a cosmetic difference).
    """
    return (
        f"fail:net_edge_at_final_size: net_edge_ppm={_FINAL_NET_EDGE_PPM} "
        f"min_net_edge_ppm={min_net_edge_ppm}"
    )


def _forecast() -> ForecastRecord:
    """Build the forecast: probability 500_000 ppm, 3_000 micros of research.

    Every field is chosen so all twelve SPEC S9.3 entry conditions pass at the
    probe price, leaving the final-size gate as the only thing that can decline.

    Returns:
        The constructed, post-init-validated `ForecastRecord`.
    """
    return ForecastRecord(
        forecast_id="fc-final-size-0001",
        market_ticker="FINAL-SIZE-TICKER",
        normalized_question_hash="sha256:final-size-question",
        probability_ppm=500_000,
        ci_low_ppm=100_000,
        ci_high_ppm=200_000,
        model_votes=(),
        vote_dispersion_ppm=0,
        rationale_markdown="n/a",
        citations=(_CITATION,),
        source_quality_notes=(),
        research_cost_micros=3_000,
        triage_stage="full",
        created_at=_INSTANT,
        forecast_horizon_hours=48,
        market_price_baseline_pips=4_500,
        baseline_quote_snapshot_id="snap-final-size-0001",
        coherence_group_sum_ppm=None,
        coherence_flag=False,
        abstention_reason=None,
        eligible_for_live=True,
    )


def _two_level_book() -> OrderBookSnapshot:
    """Build the two-level ask book the probe and the sized fill price against.

    The best level rests exactly the 100-centis probe size at 4_500 pips, so the
    probe never sees the 5_000-pip level behind it; any larger fill must.
    `yes_bids` is empty, which pins the execution style to `cross` (SPEC S9.7
    row 1) and therefore the emitted price to the walk's marginal level.

    Returns:
        The constructed `OrderBookSnapshot`.
    """
    return OrderBookSnapshot(
        ticker="FINAL-SIZE-TICKER",
        yes_bids=(),
        yes_asks=(
            OrderBookLevel(price=PricePips(4_500), quantity=ContractCentis(100)),
            OrderBookLevel(price=PricePips(5_000), quantity=ContractCentis(900)),
        ),
        fetched_at=_INSTANT,
    )


def _inputs(*, min_net_edge_ppm: int) -> SelectorInputs:
    """Assemble the shared `SelectorInputs`, varying only the net-edge floor.

    Args:
        min_net_edge_ppm: The configured net-edge floor, in ppm. The single
            knob every test in this module turns.

    Returns:
        The constructed `SelectorInputs`.
    """
    return SelectorInputs(
        forecast=_forecast(),
        calibration_map_version="calib-final-size-v1",
        order_book=_two_level_book(),
        fee_model=FeeModelInput(
            model=FeeModel(
                schedule_id="final-size-fee-settlement-only",
                maker_fee_ppm=0,
                taker_fee_ppm=0,
                settlement_fee_ppm=10_000,
            ),
            as_of=_INSTANT,
        ),
        slippage_model=SlippageModelInput(
            model_id="final-size-slippage", per_contract_buffer_ppm=2_000
        ),
        positions=PositionReadModelInput(
            snapshot_id="positions-final-size",
            equity_micros=MoneyMicros(1_000_000_000_000),
            above_floor_capital_micros=MoneyMicros(1_000_000_000),
            total_deploy_cap_micros=MoneyMicros(1_000_000_000_000),
            market_exposure=MoneyMicros(0),
            event_exposure=MoneyMicros(0),
            bucket_exposure=MoneyMicros(0),
            total_exposure=MoneyMicros(0),
            notional_today=MoneyMicros(0),
        ),
        risk_config=RiskConfigInput(
            config=RiskConfig(min_net_edge_ppm=min_net_edge_ppm),
            config_hash="sha256:risk-final-size",
        ),
        correlation_tags=(),
    )


def test_the_probe_and_the_sized_fill_price_at_four_distinct_edges_each() -> None:
    """The fixture actually separates every quantity the gate could be reading.

    Positive control for the whole module, asserted against
    `compute_executable_edge` directly at the two sizes `select` uses, so the
    separation is a property of the fixture rather than of the code under test.
    Two things are pinned. First, the probe and the final fill price at
    *different* net edges -- were they equal, every test below would pass for
    the wrong reason, the final-size gate being indistinguishable from the
    `net_edge_min` entry condition that already ran at the probe price. Second,
    the final fill's four chained edges are four different numbers, so a gate
    reading the gross, fee-adjusted, or slippage-adjusted figure instead of the
    research-cost-adjusted net edge is detectable at all.
    """
    inputs = _inputs(min_net_edge_ppm=30_000)

    probe = compute_executable_edge(
        order_book=inputs.order_book,
        size=ContractCentis(_PROBE_SIZE_CENTIS),
        forecast=inputs.forecast,
        fee_model=inputs.fee_model,
        slippage_model=inputs.slippage_model,
    )
    final = compute_executable_edge(
        order_book=inputs.order_book,
        size=ContractCentis(_FINAL_SIZE_CENTIS),
        forecast=inputs.forecast,
        fee_model=inputs.fee_model,
        slippage_model=inputs.slippage_model,
    )

    assert isinstance(probe, EdgeFigures)
    assert isinstance(final, EdgeFigures)
    assert probe.executable_price_ppm == 450_000
    assert final.executable_price_ppm == 475_000
    assert probe.research_cost_adjusted_edge_ppm == _PROBE_NET_EDGE_PPM
    assert final.gross_edge_ppm == 25_000
    assert final.fee_adjusted_edge_ppm == 15_000
    assert final.slippage_adjusted_edge_ppm == 13_000
    assert final.research_cost_adjusted_edge_ppm == _FINAL_NET_EDGE_PPM


def test_select_rejects_when_the_net_edge_does_not_survive_being_sized() -> None:
    """A probe edge of 35_000 ppm that decays to 11_500 ppm at the final size is
    rejected against the default 30_000-ppm floor: no intent, and the trailing
    reason is the exact `fail:net_edge_at_final_size` string naming both
    operands.

    This is the arm issue #249 found untested. The twelve entry reasons all read
    `pass:` -- proving the decision reached the post-sizing gate rather than
    being turned away by the `net_edge_min` entry condition, which saw the
    probe's healthy 35_000 ppm and passed.
    """
    decision = select(_inputs(min_net_edge_ppm=30_000))

    assert decision.intents == ()
    assert len(decision.reasons) == _EXPECTED_REASON_COUNT
    assert all(reason.startswith("pass:") for reason in decision.reasons[:-1])
    assert decision.reasons[-1] == _reject_reason(30_000)
    assert _SIZING_REASON not in decision.reasons


def test_select_emits_the_sized_intent_when_the_edge_exactly_meets_the_floor() -> None:
    """A final-size net edge exactly *equal* to the floor is admissible: the gate
    rejects on `<`, not `<=`, so 11_500 ppm against an 11_500-ppm floor emits the
    intent rather than declining.

    Pins the accept side with exact values -- 200 centis at the 5_000-pip
    marginal level, capped at the 950_000-micro fill cost plus its 20_000-micro
    worst-case fee -- so a gate mutated to reject at equality is caught by a
    missing intent, and one mutated to accept everything is caught by the
    reject-side test above.
    """
    decision = select(_inputs(min_net_edge_ppm=_FINAL_NET_EDGE_PPM))

    assert len(decision.intents) == 1
    intent = decision.intents[0]
    assert intent.size == ContractCentis(_FINAL_SIZE_CENTIS)
    assert intent.price == PricePips(5_000)
    assert intent.max_notional == MoneyMicros(970_000)
    assert intent.execution_style == "cross"
    assert len(decision.reasons) == _EXPECTED_REASON_COUNT
    assert decision.reasons[-1] == _SIZING_REASON


@pytest.mark.parametrize(
    ("min_net_edge_ppm", "expect_intent"),
    [
        (11_499, True),
        (_FINAL_NET_EDGE_PPM, True),
        (11_501, False),
    ],
)
def test_the_final_size_gate_turns_over_exactly_at_the_configured_floor(
    min_net_edge_ppm: int, expect_intent: bool
) -> None:
    """The gate's verdict flips between a floor of 11_500 and 11_501 ppm and
    nowhere else: one ppm below and one ppm at the final net edge both emit, one
    ppm above declines.

    A three-point sweep straddling the boundary by a single ppm, so a comparison
    mutated in either direction (`<=`, `>`, `>=`) or shifted by one changes at
    least one row. Every floor here is far below the probe's 35_000-ppm net edge
    and far below the final fill's other three edges (25_000 / 15_000 / 13_000),
    so the only quantity any row can be measuring is the *post-sizing*
    research-cost-adjusted net edge.

    Args:
        min_net_edge_ppm: The configured net-edge floor for this row, in ppm.
        expect_intent: Whether the row should emit the sized intent.
    """
    decision = select(_inputs(min_net_edge_ppm=min_net_edge_ppm))

    assert (len(decision.intents) == 1) is expect_intent
    expected_tail = (
        _SIZING_REASON if expect_intent else _reject_reason(min_net_edge_ppm)
    )
    assert decision.reasons[-1] == expected_tail
