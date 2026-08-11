"""The entry gate no longer amortizes a per-forecast charge over the probe (#483).

Research cost is incurred once per *forecast*, not once per *contract*, so
charging it against ``windbreak.selector``'s fixed 1.00-contract probe compared
a fixed cost against a per-unit return. A binary contract's gross edge over one
contract cannot exceed 1 000 000 ppm, and the shipped charge is 3 000 000
micros, so ``net_edge_min`` was unreachable for every market at every price with
any capital.

The owner decision of 2026-08-10 moves research-spend governance off the entry
gate and onto a configurable per-UTC-day ceiling (see
``tests/scheduler/test_research_spend_durability.py``). This module pins the
selector half of that: the edge chain carries no research term, so a forecast's
``research_cost_micros`` cannot move a single figure the entry conditions read.

Both directions are pinned. The real charge the pipeline books no longer
refuses a market that genuinely merits an intent, **and** a market whose edge
genuinely misses ``min_net_edge_ppm`` is still refused at exactly the same
floor, with exact values. ``min_net_edge_ppm`` is read at its production
default throughout: no threshold moves here.
"""

from __future__ import annotations

import dataclasses

from tests.selector.entry_baseline import (
    baseline_forecast,
    baseline_inputs,
    probe_figures,
)
from windbreak.config.schema import RiskConfig
from windbreak.forecast.budget import FULL_PIPELINE_RESEARCH_COST_MICROS
from windbreak.selector import select, serialize_decision
from windbreak.selector.edge import EdgeFigures
from windbreak.selector.entry import evaluate_entry_conditions

#: The baseline scenario's net edge, in ppm, hand-derived in
#: ``tests/selector/entry_baseline.py``: gross 100 000, less a 10 000 ppm fee
#: and a 2 000 ppm slippage buffer.
BASELINE_NET_EDGE_PPM = 88_000

#: The probability, in ppm, that lands the baseline's net edge exactly one ppm
#: below ``RiskConfig().min_net_edge_ppm``. The chain is
#: ``probability - executable(500 000) - fee(10 000) - slippage(2 000)``, so
#: ``541_999 - 512_000 == 29_999``.
ONE_PPM_SHORT_PROBABILITY_PPM = 541_999

#: The net edge, in ppm, that probability produces -- one ppm short of the
#: production floor, which is where the refusal must still happen.
ONE_PPM_SHORT_NET_EDGE_PPM = 29_999


def test_the_edge_chain_carries_no_research_term() -> None:
    """No :class:`EdgeFigures` field names research, derived from the dataclass.

    Read off ``dataclasses.fields`` rather than a hand-restated tuple, so a
    research term reintroduced under any name containing ``research`` fails
    this test instead of walking through a list nobody updated. The field set
    is asserted non-empty first, so the scan cannot pass by matching nothing.
    """
    names = {field.name for field in dataclasses.fields(EdgeFigures)}

    assert names
    assert not {name for name in names if "research" in name}
    assert "net_edge_ppm" in names


def test_the_real_pipeline_charge_leaves_every_entry_figure_untouched() -> None:
    """The charge the pipeline actually books moves no figure the gate reads.

    The cost is imported from its production definition site rather than
    restated, so the seam and the composition cannot silently disagree again
    (issue #483 acceptance criterion 6). Every one of the chained figures is
    compared between a zero-cost and a real-cost forecast, not merely the net
    edge -- a research term reintroduced into the annualized return alone would
    otherwise pass.
    """
    free = baseline_inputs(forecast=baseline_forecast(research_cost_micros=0))
    charged = baseline_inputs(
        forecast=baseline_forecast(
            research_cost_micros=FULL_PIPELINE_RESEARCH_COST_MICROS
        )
    )

    free_figures = probe_figures(free)
    charged_figures = probe_figures(charged)

    assert FULL_PIPELINE_RESEARCH_COST_MICROS == 3_000_000
    assert free_figures == charged_figures
    assert charged_figures.net_edge_ppm == BASELINE_NET_EDGE_PPM


def test_the_real_pipeline_charge_no_longer_refuses_a_meritorious_market() -> None:
    """A market that merits an intent gets one while carrying the real charge.

    ``min_net_edge_ppm`` is read at ``RiskConfig()``'s production default and
    asserted to be the value the pass is measured against, so this cannot be
    read as a loosened floor. The whole decision is then compared, serialized,
    against the same market carrying a zero charge: identical bytes are the
    strongest available statement that the gate no longer reads the charge at
    all.
    """
    charged = baseline_inputs(
        forecast=baseline_forecast(
            research_cost_micros=FULL_PIPELINE_RESEARCH_COST_MICROS
        )
    )
    free = baseline_inputs(forecast=baseline_forecast(research_cost_micros=0))

    decision = select(charged)

    assert len(decision.intents) == 1
    assert RiskConfig().min_net_edge_ppm == 30_000
    assert RiskConfig().min_net_edge_ppm <= BASELINE_NET_EDGE_PPM
    assert [reason for reason in decision.reasons if reason.startswith("fail:")] == []
    assert serialize_decision(decision) == serialize_decision(select(free))


def test_a_market_one_ppm_short_of_the_floor_is_still_refused() -> None:
    """The floor still refuses, at exactly one ppm below it, with exact values.

    The failing reason is compared for full string equality rather than a
    substring match, and the failing-check set is compared for equality rather
    than membership, so a gate that stopped refusing -- or refused for a
    different reason, or refused two conditions where one was expected -- fails
    here.
    """
    short = baseline_inputs(
        forecast=baseline_forecast(
            probability_ppm=ONE_PPM_SHORT_PROBABILITY_PPM,
            research_cost_micros=FULL_PIPELINE_RESEARCH_COST_MICROS,
        )
    )

    figures = probe_figures(short)
    decision = select(short)

    assert figures.net_edge_ppm == ONE_PPM_SHORT_NET_EDGE_PPM
    assert RiskConfig().min_net_edge_ppm - 1 == ONE_PPM_SHORT_NET_EDGE_PPM
    assert decision.intents == ()
    assert [reason for reason in decision.reasons if reason.startswith("fail:")] == [
        "fail:net_edge_min: net_edge_ppm=29999 min_net_edge_ppm=30000"
    ]
    assert [
        check.name
        for check in evaluate_entry_conditions(short, figures)
        if not check.passed
    ] == ["net_edge_min"]


def test_one_more_ppm_of_probability_clears_the_same_floor() -> None:
    """One ppm more probability passes, so the refusal above is the boundary.

    Without this the refusal could be produced by any condition that happens to
    sit far from its boundary; with it, the pair brackets ``min_net_edge_ppm``
    exactly and the floor is shown to be what decided both.
    """
    at_floor = baseline_inputs(
        forecast=baseline_forecast(
            probability_ppm=ONE_PPM_SHORT_PROBABILITY_PPM + 1,
            research_cost_micros=FULL_PIPELINE_RESEARCH_COST_MICROS,
        )
    )

    figures = probe_figures(at_floor)
    decision = select(at_floor)

    assert figures.net_edge_ppm == RiskConfig().min_net_edge_ppm
    assert len(decision.intents) == 1
    assert [reason for reason in decision.reasons if reason.startswith("fail:")] == []
