"""Tests for windbreak.forecast.budget (issue #28): research budget enforcement.

Pins the SPEC S8.4/S16 budget contract: per-forecast and per-day micros
ceilings enforced fail-closed (the day-bucket charge lands *before* the
per-forecast raise, so a breached forecast still counts against the day),
`max_pages` bounding `bounded_web_research`'s fetch attempts, the UTC-day
rollover that resets the day bucket, the tracer invariant that an unused
budget produces a byte-identical record, and the cost-per-resolved-forecast
report's ceiling-division and zero-denominator handling.
`windbreak/forecast/budget.py` does not exist yet, so importing it below fails
collection with `ModuleNotFoundError: No module named 'windbreak.forecast.budget'`
-- the expected Gate 1 RED state for issue #28.

Local-double choice (`CountingFetchTransport`)
    `tests/forecast/conftest.py`'s `FixtureFetchTransport` returns
    deterministic, URL-derived content (`f"fixture content for {url}"`) but
    exposes no call counter, and it cannot be imported directly here (this
    project's rootdir-relative pytest import mode does not support
    cross-test-module imports -- see `tests/forecast/conftest.py`'s and
    `test_triage.py`'s module docstrings for the same convention applied
    elsewhere). `CountingFetchTransport` below is a small, self-contained
    local double that reproduces the identical deterministic content shape
    while counting every `fetch` call, so `max_pages` and per-day-halt tests
    can assert on real fetch-attempt counts without inventing a distinct
    content contract from the one every other fixture-backed citation in this
    package already self-verifies against.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.budget import (
    BUDGET_DAY_EXHAUSTED_EVENT,
    BUDGET_FORECAST_EXCEEDED_EVENT,
    COST_REPORT_EVENT,
    DEFAULT_MAX_PAGES,
    DEFAULT_PER_DAY_BUDGET_MICROS,
    DEFAULT_PER_FORECAST_BUDGET_MICROS,
    BudgetEvent,
    DailyBudgetExhaustedError,
    InMemoryBudgetLedger,
    PerForecastBudgetExceededError,
    ResearchBudget,
    report_research_costs,
)
from windbreak.forecast.pipeline import (
    bounded_web_research,
    decompose_subquestions,
    run_pipeline,
)
from windbreak.forecast.records import forecast_record_to_payload

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

    FakeVoteTransportFactory = Callable[..., object]
    ResearchToolsFactory = Callable[..., ResearchTools]

#: `windbreak.forecast.pipeline`'s private `_RESEARCH_COST_MICROS` stub cost for
#: a full run -- named here (rather than imported, since it is private) so
#: every budget-boundary assertion below reads against the same known figure.
_FULL_RUN_RESEARCH_COST_MICROS = 3_000_000


class CountingFetchTransport:
    """A `FetchTransport` double counting every fetch attempt (issue #28).

    Reproduces conftest's `FixtureFetchTransport` content shape locally (see
    the module docstring's "Local-double choice" note) while tracking a public
    `fetch_count`, so `max_pages` and per-day-halt tests can assert on the
    exact number of underlying fetch attempts.
    """

    def __init__(self) -> None:
        """Initialize the call counter at zero."""
        self.fetch_count = 0

    def fetch(self, url: str) -> str:
        """Return deterministic content for `url`, incrementing the counter.

        Args:
            url: The URL being fetched.

        Returns:
            Deterministic content derived only from `url`.
        """
        self.fetch_count += 1
        return f"fixture content for {url}"


def _assert_json_safe_leaves(node: object) -> None:
    """Recursively assert every leaf of `node` is an int, str, or bool.

    Args:
        node: A ledgered payload node (mapping, sequence, or scalar leaf).
    """
    if isinstance(node, dict):
        for value in node.values():
            _assert_json_safe_leaves(value)
    elif isinstance(node, list | tuple):
        for item in node:
            _assert_json_safe_leaves(item)
    else:
        assert isinstance(node, int | str | bool), f"non-leaf payload value: {node!r}"
        assert type(node) is not float, f"float leaf found in payload: {node!r}"


# --- Constants ----------------------------------------------------------------


def test_budget_default_constants_have_expected_values() -> None:
    """The three budget defaults are pinned to their SPEC S16 values.

    The per-forecast ceiling is *derived* (issue #394): it is the worst case a
    correct triaged PROCEED run can cost, with zero intended headroom above it.
    `test_budget_headroom.py` pins that derivation term by term; this assertion
    is the plain value check the rest of this module reads against.
    """
    assert DEFAULT_PER_FORECAST_BUDGET_MICROS == 6_060_000
    assert DEFAULT_PER_DAY_BUDGET_MICROS == 20_000_000
    assert DEFAULT_MAX_PAGES == 20


def test_budget_event_constants_have_expected_values() -> None:
    """The three budget ledger event-type strings are pinned exactly."""
    assert BUDGET_FORECAST_EXCEEDED_EVENT == "BUDGET_FORECAST_EXCEEDED"
    assert BUDGET_DAY_EXHAUSTED_EVENT == "BUDGET_DAY_EXHAUSTED"
    assert COST_REPORT_EVENT == "COST_REPORT"


# --- ResearchBudget: construction validation -----------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"per_forecast_micros": -1},
        {"per_day_micros": -1},
        {"max_pages": -1},
    ],
)
def test_research_budget_rejects_negative_inputs(kwargs: dict[str, int]) -> None:
    """Any negative budget or page-count input is rejected at construction."""
    with pytest.raises(ValueError):
        ResearchBudget(ledger=InMemoryBudgetLedger(), **kwargs)


def test_research_budget_max_pages_property_reflects_constructed_value() -> None:
    """The `max_pages` property returns exactly the constructed value."""
    budget = ResearchBudget(max_pages=5, ledger=InMemoryBudgetLedger())

    assert budget.max_pages == 5


# --- Per-forecast breach: fail-closed, day bucket still charged ---------------


def test_per_forecast_breach_raises_and_still_charges_the_day_bucket(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """A per-forecast breach raises `PerForecastBudgetExceededError`, ledgers
    exactly one `BUDGET_FORECAST_EXCEEDED` event, and the spend still lands in
    the day bucket -- proven by a subsequent `ensure_day_open` call raising.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(
        per_forecast_micros=_FULL_RUN_RESEARCH_COST_MICROS - 1,
        per_day_micros=_FULL_RUN_RESEARCH_COST_MICROS,
        ledger=ledger,
    )

    with pytest.raises(PerForecastBudgetExceededError) as exc_info:
        run_pipeline(
            market,
            baseline,
            transport=make_fake_vote_transport(),
            created_at=created_at,
            research_tools=research_tools,
            budget=budget,
        )

    assert exc_info.value.cost_micros == _FULL_RUN_RESEARCH_COST_MICROS
    assert exc_info.value.budget_micros == _FULL_RUN_RESEARCH_COST_MICROS - 1
    exceeded_events = ledger.events_by_type(BUDGET_FORECAST_EXCEEDED_EVENT)
    assert len(exceeded_events) == 1
    _assert_json_safe_leaves(exceeded_events[0].payload)

    with pytest.raises(DailyBudgetExhaustedError):
        budget.ensure_day_open(at=created_at)


def test_per_forecast_exact_budget_succeeds_with_zero_breach_events(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """A cost exactly equal to `per_forecast_micros` passes -- the boundary is
    inclusive, so a `>=` mutant on the raise condition would be caught.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(
        per_forecast_micros=_FULL_RUN_RESEARCH_COST_MICROS, ledger=ledger
    )

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        budget=budget,
    )

    assert record.eligible_for_live is True
    assert ledger.events_by_type(BUDGET_FORECAST_EXCEEDED_EVENT) == ()


# --- Per-day halt: before-any-research fail-closed -----------------------------


def test_per_day_halt_blocks_the_third_run_before_any_fetch(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """Two same-UTC-day runs exhaust a `6_000_000`-micros day budget; the third
    raises `DailyBudgetExhaustedError` before touching research at all.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(
        per_day_micros=2 * _FULL_RUN_RESEARCH_COST_MICROS, ledger=ledger
    )
    counting_transport = CountingFetchTransport()
    tools = research_tools_factory(
        cache_dir=tmp_path, fetch_transport=counting_transport
    )

    run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=tools,
        budget=budget,
    )
    run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=tools,
        budget=budget,
    )
    fetch_count_after_two_runs = counting_transport.fetch_count

    with pytest.raises(DailyBudgetExhaustedError) as exc_info:
        run_pipeline(
            market,
            baseline,
            transport=make_fake_vote_transport(),
            created_at=created_at,
            research_tools=tools,
            budget=budget,
        )

    assert counting_transport.fetch_count == fetch_count_after_two_runs
    assert exc_info.value.spent_micros == 2 * _FULL_RUN_RESEARCH_COST_MICROS
    assert exc_info.value.budget_micros == 2 * _FULL_RUN_RESEARCH_COST_MICROS
    exhausted_events = ledger.events_by_type(BUDGET_DAY_EXHAUSTED_EVENT)
    assert len(exhausted_events) == 1
    _assert_json_safe_leaves(exhausted_events[0].payload)


def test_utc_day_rollover_resets_the_spend_counter(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """Spend right up to `23:59:59` UTC blocks a same-day retry, but a run at
    the very next `00:00:00` UTC succeeds -- the day bucket resets, not decays.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(
        per_day_micros=_FULL_RUN_RESEARCH_COST_MICROS, ledger=ledger
    )
    late_on_day_one = datetime(2024, 12, 10, 23, 59, 59, tzinfo=UTC)
    start_of_day_two = datetime(2024, 12, 11, 0, 0, 0, tzinfo=UTC)
    tools = research_tools_factory(cache_dir=tmp_path)

    run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=late_on_day_one,
        research_tools=tools,
        budget=budget,
    )
    with pytest.raises(DailyBudgetExhaustedError):
        run_pipeline(
            market,
            baseline,
            transport=make_fake_vote_transport(),
            created_at=late_on_day_one,
            research_tools=tools,
            budget=budget,
        )

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=start_of_day_two,
        research_tools=tools,
        budget=budget,
    )

    assert record.eligible_for_live is True


# --- bounded_web_research: max_pages fetch bounding ----------------------------


def test_bounded_web_research_max_pages_one_yields_one_fetch_and_one_citation(
    market: NormalizedMarket,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """`max_pages=1` stops after exactly one fetch attempt, one citation."""
    counting_transport = CountingFetchTransport()
    tools = research_tools_factory(
        cache_dir=tmp_path, fetch_transport=counting_transport
    )
    subquestions = decompose_subquestions(market)

    citations = bounded_web_research(subquestions, tools=tools, max_pages=1)

    assert counting_transport.fetch_count == 1
    assert len(citations) == 1


def test_bounded_web_research_max_pages_zero_yields_zero_fetches(
    market: NormalizedMarket,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """`max_pages=0` performs zero fetches and gathers zero citations."""
    counting_transport = CountingFetchTransport()
    tools = research_tools_factory(
        cache_dir=tmp_path, fetch_transport=counting_transport
    )
    subquestions = decompose_subquestions(market)

    citations = bounded_web_research(subquestions, tools=tools, max_pages=0)

    assert counting_transport.fetch_count == 0
    assert citations == ()


def test_bounded_web_research_max_pages_none_is_unbounded_default(
    market: NormalizedMarket,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """`max_pages=None` is unchanged from today's behavior: 3 fetches for the
    default fixture's 3 subquestions.
    """
    counting_transport = CountingFetchTransport()
    tools = research_tools_factory(
        cache_dir=tmp_path, fetch_transport=counting_transport
    )
    subquestions = decompose_subquestions(market)

    citations = bounded_web_research(subquestions, tools=tools, max_pages=None)

    assert counting_transport.fetch_count == 3
    assert len(citations) == 3


def test_bounded_web_research_negative_max_pages_raises_value_error(
    market: NormalizedMarket,
    research_tools: ResearchTools,
) -> None:
    """A negative `max_pages` is a usage error, rejected loudly."""
    subquestions = decompose_subquestions(market)

    with pytest.raises(ValueError):
        bounded_web_research(subquestions, tools=research_tools, max_pages=-1)


# --- End-to-end threading: max_pages flows into eligibility --------------------


def test_budget_max_pages_one_threads_through_to_an_ineligible_record(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """A budget's `max_pages=1` yields a 1-citation record: fewer than the
    default `min_verified_citations=3`, so the record is ineligible (not
    abstained -- one citation still verifies).
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(max_pages=1, ledger=ledger)

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        budget=budget,
    )

    assert len(record.citations) == 1
    assert record.abstention_reason is None
    assert record.eligible_for_live is False


# --- Tracer invariant: an unused budget changes nothing ------------------------


def test_tracer_invariant_no_budget_and_generous_budget_are_byte_identical(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """Omitting `budget`, passing `budget=None`, and passing a generous budget
    all produce byte-identical payloads -- the tracer invariant holds.
    """
    ledger = InMemoryBudgetLedger()
    generous_budget = ResearchBudget(ledger=ledger)

    record_no_arg = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
    )
    record_explicit_none = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        budget=None,
    )
    record_with_budget = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        budget=generous_budget,
    )

    payload_no_arg = json.dumps(
        forecast_record_to_payload(record_no_arg), sort_keys=True
    )
    payload_none = json.dumps(
        forecast_record_to_payload(record_explicit_none), sort_keys=True
    )
    payload_with_budget = json.dumps(
        forecast_record_to_payload(record_with_budget), sort_keys=True
    )

    assert payload_no_arg == payload_none == payload_with_budget


# --- Cost report: ceiling division and zero-denominator omission --------------


def test_report_research_costs_exact_division(created_at: datetime) -> None:
    """An exactly-divisible cost yields an exact per-unit figure for both units."""
    ledger = InMemoryBudgetLedger()

    report = report_research_costs(
        total_research_cost_micros=9_000_000,
        resolved_forecast_count=3,
        profitable_trade_count=3,
        ledger=ledger,
        at=created_at,
    )

    assert report.total_research_cost_micros == 9_000_000
    assert report.resolved_forecast_count == 3
    assert report.profitable_trade_count == 3
    assert report.cost_per_resolved_forecast_micros == 3_000_000
    assert report.cost_per_profitable_trade_micros == 3_000_000


def test_report_research_costs_inexact_division_rounds_up(created_at: datetime) -> None:
    """A `10 / 3` division rounds UP (ceiling) to `4`, never truncating to `3`.

    A floor-division mutant would produce `3` here instead.
    """
    ledger = InMemoryBudgetLedger()

    report = report_research_costs(
        total_research_cost_micros=10,
        resolved_forecast_count=3,
        profitable_trade_count=3,
        ledger=ledger,
        at=created_at,
    )

    assert report.cost_per_resolved_forecast_micros == 4
    assert report.cost_per_profitable_trade_micros == 4


def test_report_research_costs_zero_denominator_yields_none_and_omits_key(
    created_at: datetime,
) -> None:
    """A zero `profitable_trade_count` yields a `None` field, and its key is
    entirely absent from the ledgered `COST_REPORT` payload.
    """
    ledger = InMemoryBudgetLedger()

    report = report_research_costs(
        total_research_cost_micros=9_000_000,
        resolved_forecast_count=3,
        profitable_trade_count=0,
        ledger=ledger,
        at=created_at,
    )

    assert report.cost_per_profitable_trade_micros is None
    assert report.cost_per_resolved_forecast_micros == 3_000_000
    events = ledger.events_by_type(COST_REPORT_EVENT)
    assert len(events) == 1
    payload = events[0].payload
    assert "cost_per_profitable_trade_micros" not in payload
    assert "cost_per_resolved_forecast_micros" in payload
    _assert_json_safe_leaves(payload)


def test_report_research_costs_negative_inputs_raise_value_error(
    created_at: datetime,
) -> None:
    """Any negative cost or count input is rejected."""
    ledger = InMemoryBudgetLedger()

    with pytest.raises(ValueError):
        report_research_costs(
            total_research_cost_micros=-1,
            resolved_forecast_count=1,
            profitable_trade_count=1,
            ledger=ledger,
            at=created_at,
        )


# --- InMemoryBudgetLedger mechanics ---------------------------------------------


def test_in_memory_budget_ledger_filters_by_type_and_preserves_record_order() -> None:
    """`events_by_type` filters correctly and preserves the recorded order."""
    ledger = InMemoryBudgetLedger()
    event_a = BudgetEvent(
        event_type=BUDGET_FORECAST_EXCEEDED_EVENT,
        payload={"n": 1},
        ts="2024-12-10T00:00:00.000000Z",
    )
    event_b = BudgetEvent(
        event_type=BUDGET_DAY_EXHAUSTED_EVENT,
        payload={"n": 2},
        ts="2024-12-10T00:00:01.000000Z",
    )
    event_c = BudgetEvent(
        event_type=BUDGET_FORECAST_EXCEEDED_EVENT,
        payload={"n": 3},
        ts="2024-12-10T00:00:02.000000Z",
    )

    ledger.record(event_a)
    ledger.record(event_b)
    ledger.record(event_c)

    assert ledger.events_by_type(BUDGET_FORECAST_EXCEEDED_EVENT) == (event_a, event_c)
    assert ledger.events_by_type(BUDGET_DAY_EXHAUSTED_EVENT) == (event_b,)


# --- charge_stage: multi-stage forecasts checked in aggregate (issue #348) -----


def test_charge_stage_checks_the_aggregate_against_the_per_forecast_ceiling() -> None:
    """Two stages that each fit under the ceiling still breach it together.

    A forecast that spends in two stages (Stage-0 triage, then the full
    pipeline) is still *one* forecast, so SPEC S16's per-forecast ceiling
    governs the total. Charging each stage through `charge_forecast`
    independently would clear both checks here, leaving a guard that cannot
    trip; the raised error and its ledgered event both name the aggregate.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(per_forecast_micros=1_000_000, ledger=ledger)
    at = datetime(2024, 12, 10, 12, 0, tzinfo=UTC)

    remainder = budget.charge_stage(600_000, market_ticker="KX-A", at=at)

    with pytest.raises(PerForecastBudgetExceededError) as exc_info:
        remainder.charge_forecast(600_000, market_ticker="KX-A", at=at)

    assert exc_info.value.cost_micros == 1_200_000
    assert exc_info.value.budget_micros == 1_000_000
    breach_events = ledger.events_by_type(BUDGET_FORECAST_EXCEEDED_EVENT)
    assert len(breach_events) == 1
    assert breach_events[0].payload["cost_micros"] == 1_200_000
    _assert_json_safe_leaves(breach_events[0].payload)


def test_charge_stage_aggregate_exactly_at_the_ceiling_passes() -> None:
    """A two-stage total exactly equal to the ceiling passes -- the aggregate
    boundary is inclusive, matching `charge_forecast`'s single-charge boundary,
    so a `>=` mutant on the raise condition is caught.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(per_forecast_micros=1_000_000, ledger=ledger)
    at = datetime(2024, 12, 10, 12, 0, tzinfo=UTC)

    remainder = budget.charge_stage(400_000, market_ticker="KX-A", at=at)
    remainder.charge_forecast(600_000, market_ticker="KX-A", at=at)

    assert ledger.events_by_type(BUDGET_FORECAST_EXCEEDED_EVENT) == ()


def test_charge_stage_charges_each_stage_to_the_shared_day_bucket_once() -> None:
    """Both stages land in the *same* day bucket, once each.

    The remainder budget is a view onto the same forecast, not a fresh budget:
    were its day bucket separate, the first stage's spend would vanish from the
    day's running total and the day ceiling could be overrun indefinitely.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(
        per_forecast_micros=1_000_000, per_day_micros=1_000_000, ledger=ledger
    )
    at = datetime(2024, 12, 10, 12, 0, tzinfo=UTC)

    remainder = budget.charge_stage(400_000, market_ticker="KX-A", at=at)
    remainder.charge_forecast(600_000, market_ticker="KX-A", at=at)

    with pytest.raises(DailyBudgetExhaustedError) as exc_info:
        budget.ensure_day_open(at=at)
    assert exc_info.value.spent_micros == 1_000_000


def test_charge_stage_does_not_leak_its_offset_into_the_next_forecast() -> None:
    """The already-spent offset is scoped to the one forecast that spent it.

    A budget outlives any single forecast, so charging a stage must not make
    every later forecast on that budget carry the first one's cost.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(
        per_forecast_micros=1_000_000, per_day_micros=10_000_000, ledger=ledger
    )
    at = datetime(2024, 12, 10, 12, 0, tzinfo=UTC)

    budget.charge_stage(900_000, market_ticker="KX-A", at=at)
    budget.charge_forecast(900_000, market_ticker="KX-B", at=at)

    assert ledger.events_by_type(BUDGET_FORECAST_EXCEEDED_EVENT) == ()


def test_charge_stage_raises_when_the_stage_alone_breaches_the_ceiling() -> None:
    """A first stage that alone exceeds the ceiling fails closed immediately --
    the remainder budget is never handed back, so no later stage can run on it.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(per_forecast_micros=1_000_000, ledger=ledger)
    at = datetime(2024, 12, 10, 12, 0, tzinfo=UTC)

    with pytest.raises(PerForecastBudgetExceededError) as exc_info:
        budget.charge_stage(1_000_001, market_ticker="KX-A", at=at)

    assert exc_info.value.cost_micros == 1_000_001
    assert len(ledger.events_by_type(BUDGET_FORECAST_EXCEEDED_EVENT)) == 1


def test_charge_stage_remainder_preserves_the_page_ceiling() -> None:
    """The remainder view keeps the forecast's `max_pages` ceiling, so the
    stage that follows is still bounded to the same page budget.
    """
    budget = ResearchBudget(max_pages=7, ledger=InMemoryBudgetLedger())
    at = datetime(2024, 12, 10, 12, 0, tzinfo=UTC)

    remainder = budget.charge_stage(1, market_ticker="KX-A", at=at)

    assert remainder.max_pages == 7
