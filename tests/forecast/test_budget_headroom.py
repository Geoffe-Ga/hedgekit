"""The per-forecast ceiling's relationship to what a forecast can cost (issue #394).

`_RESEARCH_COST_MICROS` (the full pipeline's fixed research charge) and
`DEFAULT_PER_FORECAST_BUDGET_MICROS` (the ceiling that charge is checked
against) were once the *same number*, 3_000_000. A single research stage
therefore consumed the entire per-forecast ceiling, leaving exactly zero
headroom for the Stage-0 triage prior or the ensemble votes charged against the
same forecast -- so a correct triaged PROCEED run could not help but breach.
Issue #269's follow-up raised the *config* mirror to 6_000_000 but left the
engine constants at 3_000_000 and derived that 6_000_000 from a worst case that
omitted the Stage-0 charge entirely.

This module pins the relationship SPEC S16 now states, so that editing any one
term without re-deriving the ceiling fails the suite rather than silently
recreating the collision:

    per_forecast_micros = stage0_prior_cost
                        + full_pipeline_research_cost
                        + vote_ensemble_size * provider_retry.max_cost_micros

The intended headroom is *exactly zero* above that worst case. Anything lower
breaches on a healthy run (the defect above); anything higher is slack the
guard cannot see into, which is the "guard that exists but cannot fire" shape
this backlog has fixed a dozen times.

Two kinds of test live here and both are load-bearing:

* The *relationship* tests pin each term and the total as literals. They are
  what make the collision impossible to reintroduce quietly.
* The *boundary* tests prove the ceiling can still trip: a worst-case run fits
  exactly (the ceiling is inclusive), and one micro beyond it raises and
  ledgers. Without these, raising the ceiling would be indistinguishable from
  deleting the guard.

Before the fix this module fails at collection with `ImportError` -- the named
budget terms do not exist yet -- which is this suite's documented Gate 1 RED
state (see `test_budget.py`'s module docstring for the same convention).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from windbreak.config.schema import ForecastBudget, ProviderRetryConfig, ScreenerConfig
from windbreak.forecast.budget import (
    BUDGET_FORECAST_EXCEEDED_EVENT,
    DEFAULT_PER_FORECAST_BUDGET_MICROS,
    DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS,
    DEFAULT_VOTE_ENSEMBLE_SIZE,
    FULL_PIPELINE_RESEARCH_COST_MICROS,
    STAGE0_PRIOR_COST_MICROS,
    InMemoryBudgetLedger,
    PerForecastBudgetExceededError,
    ResearchBudget,
)
from windbreak.forecast.providers import DEFAULT_VOTE_ENSEMBLE

#: A timezone-aware instant for every charge below; the exact day is irrelevant
#: because no test here approaches the per-day ceiling.
_AT = datetime(2024, 12, 24, 12, 0, tzinfo=UTC)

#: The market ticker stamped on every charge's audit trail.
_TICKER = "MKT-HEADROOM"

#: What the full pipeline plus a worst-case ensemble costs, in micros -- i.e.
#: everything charged *after* the Stage-0 prior on a PROCEED-path run.
_POST_TRIAGE_WORST_CASE_MICROS = (
    FULL_PIPELINE_RESEARCH_COST_MICROS
    + DEFAULT_VOTE_ENSEMBLE_SIZE * DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS
)


# --- The documented relationship ----------------------------------------------


def test_the_per_forecast_ceiling_equals_its_documented_worst_case() -> None:
    """The ceiling is the exact sum of the four terms SPEC S16 names.

    Pinned as literals rather than recomputed from the module's own constants:
    a recomputation would agree with any edit and so could never fail, which is
    precisely the silent drift this test exists to catch.
    """
    assert STAGE0_PRIOR_COST_MICROS == 60_000
    assert FULL_PIPELINE_RESEARCH_COST_MICROS == 3_000_000
    assert DEFAULT_VOTE_ENSEMBLE_SIZE == 3
    assert DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS == 1_000_000
    assert DEFAULT_PER_FORECAST_BUDGET_MICROS == 6_060_000


def test_the_ceiling_is_the_sum_of_its_terms_and_not_merely_a_larger_number() -> None:
    """The ceiling is derived from the terms, leaving exactly zero headroom.

    This is the anti-slack assertion: a ceiling set comfortably above any
    reachable cost would pass every other test in this file while quietly
    reproducing the unfalsifiable guard issue #394 forbids.
    """
    assert DEFAULT_PER_FORECAST_BUDGET_MICROS == (
        STAGE0_PRIOR_COST_MICROS + _POST_TRIAGE_WORST_CASE_MICROS
    )


def test_the_stage0_prior_costs_two_percent_of_the_full_pipeline_it_avoids() -> None:
    """Stage-0 is priced against the pipeline it exists to skip, per SPEC S8.4.

    Deriving it from the *ceiling* instead would be self-referential: raising
    the ceiling would raise the Stage-0 charge, which would raise the required
    ceiling again.
    """
    assert STAGE0_PRIOR_COST_MICROS == FULL_PIPELINE_RESEARCH_COST_MICROS // 50


# --- The mirrors the relationship depends on -----------------------------------


def test_the_restated_ensemble_size_matches_the_real_default_ensemble() -> None:
    """Adding a fourth default vote member without raising the ceiling fails here."""
    assert len(DEFAULT_VOTE_ENSEMBLE) == DEFAULT_VOTE_ENSEMBLE_SIZE


def test_the_restated_member_ceiling_matches_the_retry_policy_default() -> None:
    """A member's spend is hard-bounded by the retry policy's affordability gate.

    `RetryingProvider` checks `accrued + price > max_cost_micros` before every
    attempt and re-checks the total on success, so no member can ever book more
    than this -- which is what makes the worst case a real bound rather than an
    estimate.
    """
    policy_ceiling_micros = ProviderRetryConfig().max_cost_micros
    assert policy_ceiling_micros == DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS


def test_the_operator_facing_ceiling_mirrors_the_engine_default() -> None:
    """Config and engine agree, so the live loop enforces the documented ceiling."""
    assert ForecastBudget().per_forecast_micros == DEFAULT_PER_FORECAST_BUDGET_MICROS


def test_the_per_tick_candidate_bound_is_what_the_two_ceilings_afford() -> None:
    """The screener's candidate bound stays the quotient of the two money ceilings.

    `max_candidates_per_tick` is a documented derivation, not a computed one
    (`ScreenerConfig` stores a literal `3`), so this assertion is the only thing
    keeping it honest: raising the per-forecast ceiling without revisiting the
    bound would let a tick plan to spend more than a whole day's budget.
    """
    budget = ForecastBudget()
    assert ScreenerConfig().max_candidates_per_tick == (
        budget.per_day_micros // budget.per_forecast_micros
    )


# --- The ceiling can still trip -------------------------------------------------


def test_a_worst_case_triaged_proceed_run_fits_exactly_under_the_ceiling() -> None:
    """The most expensive correct run is affordable, and books no breach event.

    Charged the way `run_triaged_forecast` charges it: Stage-0 through
    `charge_stage`, then the remainder of the forecast against the view it
    hands back, so the ceiling sees the aggregate.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(ledger=ledger)
    budget.ensure_day_open(at=_AT)

    remainder = budget.charge_stage(
        STAGE0_PRIOR_COST_MICROS, market_ticker=_TICKER, at=_AT
    )
    remainder.charge_forecast(
        _POST_TRIAGE_WORST_CASE_MICROS, market_ticker=_TICKER, at=_AT
    )

    assert ledger.events_by_type(BUDGET_FORECAST_EXCEEDED_EVENT) == ()


def test_one_micro_beyond_the_worst_case_trips_the_per_forecast_ceiling() -> None:
    """A run that legitimately exceeds the ceiling fails closed and ledgers it.

    The counterpart to the test above: together they pin the boundary from both
    sides, so a ceiling raised out of reach fails here rather than passing
    silently. The aggregate -- not the final stage's cost alone -- is what the
    error and its ledgered payload report.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(ledger=ledger)
    budget.ensure_day_open(at=_AT)

    remainder = budget.charge_stage(
        STAGE0_PRIOR_COST_MICROS, market_ticker=_TICKER, at=_AT
    )
    with pytest.raises(PerForecastBudgetExceededError) as excinfo:
        remainder.charge_forecast(
            _POST_TRIAGE_WORST_CASE_MICROS + 1, market_ticker=_TICKER, at=_AT
        )

    assert excinfo.value.cost_micros == DEFAULT_PER_FORECAST_BUDGET_MICROS + 1
    assert excinfo.value.budget_micros == DEFAULT_PER_FORECAST_BUDGET_MICROS
    events = ledger.events_by_type(BUDGET_FORECAST_EXCEEDED_EVENT)
    assert len(events) == 1
    assert events[0].payload["cost_micros"] == DEFAULT_PER_FORECAST_BUDGET_MICROS + 1
    assert events[0].payload["market_ticker"] == _TICKER
