"""The pipeline's research stage is skipped for a self-researching ensemble (#556).

`run_pipeline` used to run `bounded_web_research` + `verify_citations` and book
`FULL_PIPELINE_RESEARCH_COST_MICROS` on *every* full-pipeline forecast, with no
branch for who the ensemble members are. ADR-0005 S1(b) says a research
forecaster does its own web research server-side and therefore ignores the
pipeline-supplied quotes entirely, so an ensemble made of such members paid
twice: once, in real search/fetch egress, for research nobody read, and again
for its own provider call.

What this module pins, and why each assertion is shaped the way it is:

Two directions, never one
    Every "skipped" assertion here has a "still runs" twin over the *same*
    wiring, because a skip made unconditional is the regression that costs the
    most: an `LlmTransport` ensemble that stops researching votes on nothing.
    `test_llm_transport_ensemble_still_researches_and_books_the_stub` and
    `test_mixed_ensemble_still_researches_and_books_the_stub` are those twins,
    and the mixed case is the sharper of the two -- an ensemble holding one
    self-researching member alongside one no-tools LLM member must still
    research, since the LLM member has nothing to vote on otherwise.

Shape, never name
    The predicate answers "does this provider do its own research", which it
    reads off the provider's own declared
    `ForecastProvider.performs_own_research`. It is emphatically *not* "is this
    provider named futuresearch" and not "is this not a FixtureVoteProvider".
    `test_declared_shape_not_class_name_decides_the_skip` and
    `test_declared_shape_not_member_provider_name_decides_the_skip` drive the
    two dimensions apart deliberately: a self-researching double whose class
    name and configured `provider` string are ordinary, and a *non*-researching
    provider wired under a member whose `provider` string is `"futuresearch"`.
    Either name-shaped guard gets exactly one of them wrong.

Work, not just accounting
    Skipping the charge while still fetching would be strictly worse than the
    defect -- the money still leaves and the ledger stops saying so. So the
    "no research happened" assertions are made at the `SearchTransport` /
    `FetchTransport` seam (`_CountingSearchTransport` / `_CountingFetchTransport`
    below), counting the calls the sandbox actually made, never inferred from
    the resulting cost. Each such assertion is paired with a positive control
    in the same module so a counter that can only ever read zero would be
    caught.

Costs chosen so one charge and two charges cannot agree
    `_REPORTED_COST_USD` converts to `_REPORTED_COST_MICROS` (1_234_568), which
    differs from the $3.00 stub, from the $2.00 per-call fallback ceiling, and
    from their sum. A fixture whose provider cost happened to equal the stub
    would make "charged once" and "charged twice" produce numbers no assertion
    could tell apart.

Live eligibility, stated rather than inherited
    A self-researching run gathers no pipeline citations, so `verified_count`
    is 0 and the record is *not* live-eligible under SPEC S8.8's
    `min_verified_citations`. That is deliberate and asserted
    (`test_research_forecaster_record_is_not_live_eligible_on_zero_verified`):
    provider-reported citations stay audit-only, exactly as they already were,
    and nothing here widens a ceiling to make a cheaper run look better.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.forecast.conftest import FixtureFetchTransport, FixtureSearchTransport
from tests.forecast.providers.test_futuresearch import (
    _body,
    _config,
    _StubHttpTransport,
)
from windbreak.forecast.budget import (
    BUDGET_SPEND_RECORDED_EVENT,
    DEFAULT_MAX_PAGES,
    DEFAULT_PER_DAY_BUDGET_MICROS,
    DEFAULT_PER_FORECAST_BUDGET_MICROS,
    DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS,
    FULL_PIPELINE_RESEARCH_COST_MICROS,
    DailyBudgetExhaustedError,
    InMemoryBudgetLedger,
    ResearchBudget,
)
from windbreak.forecast.pipeline import run_pipeline
from windbreak.forecast.providers import (
    DEFAULT_MODEL_RATE_TABLE,
    DEFAULT_PROVIDER_PRICE_TABLE,
    EnsembleMember,
    FixtureVoteProvider,
    ForecastProvider,
    FutureSearchProvider,
    ProviderForecast,
    RetryingProvider,
    RetryPolicy,
)
from windbreak.forecast.providers.retry import (
    DEFAULT_BACKOFF_BASE_MS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TOTAL_DEADLINE_MS,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.cassettes import LlmTransport
    from windbreak.forecast.providers import EnsembleMemberLike
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sanitize import ResearchQuote

#: The `cost_usd` token the recorded FutureSearch response carries, as raw JSON.
#: Its last digit is deliberately below the micro grid so the conversion's
#: ROUND_CEILING rule is visible end-to-end rather than only in the unit test:
#: 1.2345671 USD is 1_234_567.1 micros, which ceilings to 1_234_568.
_REPORTED_COST_USD = "1.2345671"

#: What `_REPORTED_COST_USD` must become once ceilinged onto the micro grid.
#: Distinct from the $3.00 stub, from the $2.00 per-call fallback ceiling, and
#: from every sum of those, so a run charged once and a run charged twice can
#: never produce the same figure.
_REPORTED_COST_MICROS = 1_234_568

#: The pinned forecaster version `_config()`'s default accepts.
_PINNED_VERSION = "futuresearch-v1"

#: How many subquestions `decompose_subquestions` produces, and therefore how
#: many searches one researched run makes.
_SUBQUESTION_COUNT = 3

#: How many fetches one researched run makes: one per subquestion to build the
#: citation, then one per citation again when `verify_citations` independently
#: refetches it.
_RESEARCHED_FETCH_COUNT = 2 * _SUBQUESTION_COUNT


class _CountingSearchTransport:
    """A `SearchTransport` counting its calls, delegating the answer verbatim."""

    def __init__(self) -> None:
        """Wrap a fresh fixture transport and start with no recorded queries."""
        self._inner = FixtureSearchTransport()
        self.queries: list[str] = []

    def search(self, query: str) -> tuple[str, ...]:
        """Record `query` and return the fixture transport's candidate URLs.

        Args:
            query: The subquestion text being searched for.

        Returns:
            Whatever `FixtureSearchTransport` returns for `query`.
        """
        self.queries.append(query)
        return self._inner.search(query)


class _CountingFetchTransport:
    """A `FetchTransport` counting its calls, delegating the answer verbatim."""

    def __init__(self) -> None:
        """Wrap a fresh fixture transport and start with no recorded URLs."""
        self._inner = FixtureFetchTransport()
        self.urls: list[str] = []

    def fetch(self, url: str) -> str:
        """Record `url` and return the fixture transport's canned content.

        Args:
            url: The URL being fetched.

        Returns:
            Whatever `FixtureFetchTransport` returns for `url`.
        """
        self.urls.append(url)
        return self._inner.fetch(url)


class _SelfResearchingStubProvider:
    """A `ForecastProvider` declaring it researches for itself, over no network.

    Deliberately named nothing like any shipped research forecaster, and it
    reaches no HTTP seam at all: what makes the pipeline skip its research stage
    must be this class's `performs_own_research` declaration and nothing else.
    """

    #: This provider's research and vote are fused (ADR-0005 S1(b)).
    performs_own_research = True

    def __init__(self, member: EnsembleMemberLike, cost_micros: int) -> None:
        """Bind the provenance and reported cost every forecast carries.

        Args:
            member: The ensemble member whose provenance stamps the forecast.
            cost_micros: The cost this provider reports for its own call.
        """
        self._member = member
        self._cost_micros = cost_micros
        self.calls = 0

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[ResearchQuote, ...],
    ) -> ProviderForecast:
        """Return one canned forecast, ignoring the pipeline's quotes.

        Args:
            market: The (unused) market under forecast.
            baseline: The (unused) baseline quote snapshot.
            vote_index: The (unused) zero-based vote index.
            quotes: The pipeline's quotes -- ignored, per ADR-0005 S1(b).

        Returns:
            A fixed, schema-valid `ProviderForecast`.
        """
        del market, baseline, vote_index, quotes
        self.calls += 1
        return ProviderForecast(
            probability_ppm=612_000,
            rationale_summary="fused research and vote",
            citations=(),
            cost_micros=self._cost_micros,
            provider=self._member.provider,
            model_version=self._member.model_version,
            training_cutoff=self._member.training_cutoff,
            response_fingerprint="f" * 64,
        )


def _budget(ledger: InMemoryBudgetLedger) -> ResearchBudget:
    """Build a budget at the *production* default ceilings (issue #556 AC4).

    Every ceiling is passed explicitly from its shipped constant rather than
    left to the constructor's default, so this module's runs are visibly held
    to production limits and a test that only passed because a ceiling was
    widened could not exist here.

    Args:
        ledger: The in-memory budget ledger every charge is recorded to.

    Returns:
        A `ResearchBudget` at `DEFAULT_PER_FORECAST_BUDGET_MICROS`,
        `DEFAULT_PER_DAY_BUDGET_MICROS`, and `DEFAULT_MAX_PAGES`.
    """
    return ResearchBudget(
        per_forecast_micros=DEFAULT_PER_FORECAST_BUDGET_MICROS,
        per_day_micros=DEFAULT_PER_DAY_BUDGET_MICROS,
        max_pages=DEFAULT_MAX_PAGES,
        ledger=ledger,
    )


def _charges(ledger: InMemoryBudgetLedger) -> tuple[int, ...]:
    """Return every research charge the budget ledger recorded, in order.

    Args:
        ledger: The in-memory budget ledger a run charged against.

    Returns:
        Each `BUDGET_SPEND_RECORDED` event's `cost_micros`, in record order.
    """
    return tuple(
        int(str(event.payload["cost_micros"]))
        for event in ledger.events_by_type(BUDGET_SPEND_RECORDED_EVENT)
    )


def _futuresearch_provider(
    cost_usd: str | None = _REPORTED_COST_USD,
) -> tuple[FutureSearchProvider, _StubHttpTransport]:
    """Build the real `FutureSearchProvider` over a recorded response body.

    Args:
        cost_usd: The raw JSON token for the response's `cost_usd` field.

    Returns:
        The provider and the stub HTTP transport recording its calls.
    """
    http = _StubHttpTransport(_body(cost_usd=cost_usd))
    return FutureSearchProvider(http, _config()), http


def _tools(
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    search: _CountingSearchTransport,
    fetch: _CountingFetchTransport,
) -> object:
    """Build sandboxed `ResearchTools` over the two counting transports.

    Args:
        research_tools_factory: The package fixture factory.
        tmp_path: The per-test cache root.
        search: The counting search transport to inject.
        fetch: The counting fetch transport to inject.

    Returns:
        A `ResearchTools` whose every search/fetch is counted.
    """
    return research_tools_factory(
        cache_dir=tmp_path / "research-cache",
        search_transport=search,
        fetch_transport=fetch,
    )


def _single_member_factory(
    provider: ForecastProvider,
) -> Callable[[EnsembleMemberLike], ForecastProvider]:
    """Build a `provider_factory` returning one fixed provider for any member.

    Args:
        provider: The provider every member is driven through.

    Returns:
        A `provider_factory` closure.
    """

    def _factory(member: EnsembleMemberLike) -> ForecastProvider:
        """Return the fixed provider, ignoring `member`.

        Args:
            member: The (unused) ensemble member.

        Returns:
            The bound provider.
        """
        del member
        return provider

    return _factory


# --- AC1: one charge, at the provider's own reported figure ----------------------


def test_research_forecaster_ensemble_books_only_the_provider_reported_cost(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., LlmTransport],
) -> None:
    """One research charge, equal to the real provider's own reported `cost_usd`.

    Driven through the real `FutureSearchProvider` over a recorded response
    body, so the figure asserted is the one the provider's own Decimal->micros
    conversion produced -- not a constant this test chose. The two failure modes
    it separates are exactly the ones #556 names: `_REPORTED_COST_MICROS` alone
    (correct), versus the stub added on top of it (the defect).
    """
    provider, http = _futuresearch_provider()
    search, fetch = _CountingSearchTransport(), _CountingFetchTransport()
    ledger = InMemoryBudgetLedger()

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=_tools(research_tools_factory, tmp_path, search, fetch),
        min_ensemble_votes=1,
        budget=_budget(ledger),
        ensemble=(EnsembleMember("openai", "gpt-5-forecast", "2024-06-01"),),
        provider_factory=_single_member_factory(provider),
    )

    assert len(http.calls) == 1
    assert _charges(ledger) == (_REPORTED_COST_MICROS,)
    assert record.research_cost_micros == _REPORTED_COST_MICROS
    assert _REPORTED_COST_MICROS != FULL_PIPELINE_RESEARCH_COST_MICROS


def test_research_forecaster_record_is_not_live_eligible_on_zero_verified(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., LlmTransport],
) -> None:
    """A self-researching run cites nothing and stays live-ineligible (S8.8).

    Stated rather than inherited: skipping the pipeline's own research means
    zero *verified* citations, and provider-reported citations remain
    audit-only. So the run is cheaper and still cannot back a live order. This
    pins that the saving was not bought by relaxing the citation bar.
    """
    provider, _ = _futuresearch_provider()
    search, fetch = _CountingSearchTransport(), _CountingFetchTransport()

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=_tools(research_tools_factory, tmp_path, search, fetch),
        min_ensemble_votes=1,
        ensemble=(EnsembleMember("openai", "gpt-5-forecast", "2024-06-01"),),
        provider_factory=_single_member_factory(provider),
    )

    assert record.citations == ()
    assert record.eligible_for_live is False
    assert record.abstention_reason is None


# --- AC3: the work is gone, asserted at the transport seam -----------------------


def test_research_forecaster_run_makes_no_search_or_fetch_call(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., LlmTransport],
) -> None:
    """Zero egress: the sandbox's search and fetch transports are never called.

    Asserted at the transport seam, never inferred from the cost -- a fix that
    only stopped *recording* the spend would leave these counters non-zero. The
    positive control lives in
    `test_llm_transport_ensemble_still_researches_and_books_the_stub`, which
    drives the identical transports to non-zero counts.
    """
    provider, http = _futuresearch_provider()
    search, fetch = _CountingSearchTransport(), _CountingFetchTransport()

    run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=_tools(research_tools_factory, tmp_path, search, fetch),
        min_ensemble_votes=1,
        ensemble=(EnsembleMember("openai", "gpt-5-forecast", "2024-06-01"),),
        provider_factory=_single_member_factory(provider),
    )

    assert search.queries == []
    assert fetch.urls == []
    assert len(http.calls) == 1


# --- AC2: both directions; the mixed ensemble must not regress -------------------


def test_llm_transport_ensemble_still_researches_and_books_the_stub(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., LlmTransport],
) -> None:
    """The shipped no-tools ensemble still researches and still books $3.00 exactly.

    The positive control for every "no calls" assertion above, and the direction
    that must not regress: an `LlmTransport` ensemble votes only on the quotes
    the pipeline verified for it, so its research stage is not redundant and its
    charge is not a double payment.
    """
    search, fetch = _CountingSearchTransport(), _CountingFetchTransport()
    ledger = InMemoryBudgetLedger()

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=_tools(research_tools_factory, tmp_path, search, fetch),
        budget=_budget(ledger),
    )

    assert len(search.queries) == _SUBQUESTION_COUNT
    assert len(fetch.urls) == _RESEARCHED_FETCH_COUNT
    assert _charges(ledger) == (FULL_PIPELINE_RESEARCH_COST_MICROS,)
    assert record.research_cost_micros == FULL_PIPELINE_RESEARCH_COST_MICROS


def test_mixed_ensemble_still_researches_and_books_the_stub(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., LlmTransport],
) -> None:
    """One self-researching member beside one no-tools member still researches.

    The named regression of #556 AC2. The no-tools member has nothing to vote on
    unless the pipeline researches for it, so "any member researches for itself"
    is the wrong rule; only "every member does" may skip the stage. The charge is
    the stub *plus* the self-researching member's own reported cost, which is
    exactly one payment for each distinct piece of research performed.
    """
    llm_member = EnsembleMember("openai", "gpt-5-forecast", "2024-06-01")
    research_member = EnsembleMember("acme", "acme-forecaster-1", "2025-01-31")
    transport = make_fake_vote_transport()
    self_researching = _SelfResearchingStubProvider(
        research_member, _REPORTED_COST_MICROS
    )

    def _factory(member: EnsembleMemberLike) -> ForecastProvider:
        """Route the self-researching member to its stub, the other to a fixture.

        Args:
            member: The ensemble member being driven.

        Returns:
            The provider for `member`.
        """
        if member.model_version == research_member.model_version:
            return self_researching
        return FixtureVoteProvider(transport, member)

    search, fetch = _CountingSearchTransport(), _CountingFetchTransport()
    ledger = InMemoryBudgetLedger()

    record = run_pipeline(
        market,
        baseline,
        transport=transport,
        created_at=created_at,
        research_tools=_tools(research_tools_factory, tmp_path, search, fetch),
        budget=_budget(ledger),
        ensemble=(llm_member, research_member),
        provider_factory=_factory,
    )

    assert len(search.queries) == _SUBQUESTION_COUNT
    assert len(fetch.urls) == _RESEARCHED_FETCH_COUNT
    assert self_researching.calls == 1
    expected = FULL_PIPELINE_RESEARCH_COST_MICROS + _REPORTED_COST_MICROS
    assert _charges(ledger) == (expected,)
    assert record.research_cost_micros == expected


# --- Trap 4/7: the guard reads declared shape, never a name ----------------------


def test_declared_shape_not_class_name_decides_the_skip(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., LlmTransport],
) -> None:
    """A self-researching provider named nothing like a vendor still skips research.

    Half of the dimension-separating pair: this provider's class name carries no
    vendor, its member's `provider` string is `"acme"`, and it reaches no HTTP
    endpoint -- only its declared `performs_own_research` says what it is. Any
    guard keyed on a name gets this case wrong.
    """
    member = EnsembleMember("acme", "acme-forecaster-1", "2025-01-31")
    provider = _SelfResearchingStubProvider(member, _REPORTED_COST_MICROS)
    search, fetch = _CountingSearchTransport(), _CountingFetchTransport()
    ledger = InMemoryBudgetLedger()

    run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=_tools(research_tools_factory, tmp_path, search, fetch),
        min_ensemble_votes=1,
        budget=_budget(ledger),
        ensemble=(member,),
        provider_factory=_single_member_factory(provider),
    )

    assert search.queries == []
    assert fetch.urls == []
    assert _charges(ledger) == (_REPORTED_COST_MICROS,)


def test_declared_shape_not_member_provider_name_decides_the_skip(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., LlmTransport],
) -> None:
    """A member named `futuresearch`, driven by a no-tools provider, researches.

    The other half of the pair, and the one a name-keyed guard fails in the
    expensive direction: it would skip the research an ordinary LLM vote depends
    on, purely because the configured `provider` string reads `"futuresearch"`.
    """
    member = EnsembleMember("futuresearch", "gpt-5-forecast", "2024-06-01")
    transport = make_fake_vote_transport()
    search, fetch = _CountingSearchTransport(), _CountingFetchTransport()
    ledger = InMemoryBudgetLedger()

    run_pipeline(
        market,
        baseline,
        transport=transport,
        created_at=created_at,
        research_tools=_tools(research_tools_factory, tmp_path, search, fetch),
        min_ensemble_votes=1,
        budget=_budget(ledger),
        ensemble=(member,),
        provider_factory=_single_member_factory(FixtureVoteProvider(transport, member)),
    )

    assert len(search.queries) == _SUBQUESTION_COUNT
    assert len(fetch.urls) == _RESEARCHED_FETCH_COUNT
    assert _charges(ledger) == (FULL_PIPELINE_RESEARCH_COST_MICROS,)


# --- Trap 8: where the guard sits relative to the budget seam --------------------


def test_exhausted_day_still_halts_a_self_researching_run_before_any_vote(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., LlmTransport],
) -> None:
    """The daily cap still halts first, even when no pipeline research will run.

    The skip must sit *after* `ensure_day_open`, never before it: the day
    ceiling governs whether any money may be spent at all, including the
    provider's own. A guard hoisted above the budget seam would let an exhausted
    day keep paying a research forecaster forever.
    """
    member = EnsembleMember("acme", "acme-forecaster-1", "2025-01-31")
    provider = _SelfResearchingStubProvider(member, _REPORTED_COST_MICROS)
    search, fetch = _CountingSearchTransport(), _CountingFetchTransport()
    ledger = InMemoryBudgetLedger()
    day_key = created_at.date().isoformat()
    budget = ResearchBudget(
        per_forecast_micros=DEFAULT_PER_FORECAST_BUDGET_MICROS,
        per_day_micros=DEFAULT_PER_DAY_BUDGET_MICROS,
        max_pages=DEFAULT_MAX_PAGES,
        ledger=ledger,
        opening_spend_by_day={day_key: DEFAULT_PER_DAY_BUDGET_MICROS},
    )

    with pytest.raises(DailyBudgetExhaustedError):
        run_pipeline(
            market,
            baseline,
            transport=make_fake_vote_transport(),
            created_at=created_at,
            research_tools=_tools(research_tools_factory, tmp_path, search, fetch),
            min_ensemble_votes=1,
            budget=budget,
            ensemble=(member,),
            provider_factory=_single_member_factory(provider),
        )

    assert provider.calls == 0
    assert search.queries == []
    assert _charges(ledger) == ()


# --- Trap: an empty ensemble is not an ensemble that researches for itself -------


def test_empty_ensemble_still_researches_and_books_the_stub(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., LlmTransport],
) -> None:
    """No members at all is not "every member researches for itself".

    `all(())` is vacuously true, so a predicate written without a non-empty
    guard would silently stop researching for an empty ensemble -- a behavior
    change on a configuration that has nothing to do with research forecasters.
    """
    search, fetch = _CountingSearchTransport(), _CountingFetchTransport()
    ledger = InMemoryBudgetLedger()

    run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=_tools(research_tools_factory, tmp_path, search, fetch),
        min_ensemble_votes=1,
        budget=_budget(ledger),
        ensemble=(),
        provider_factory=_single_member_factory(
            _SelfResearchingStubProvider(
                EnsembleMember("acme", "acme-forecaster-1", "2025-01-31"),
                _REPORTED_COST_MICROS,
            )
        ),
    )

    assert len(search.queries) == _SUBQUESTION_COUNT
    assert _charges(ledger) == (FULL_PIPELINE_RESEARCH_COST_MICROS,)


# --- The declaration survives the wrapper #555 will wire it through --------------


def test_retrying_wrapper_forwards_the_self_research_declaration(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: Callable[..., object],
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., LlmTransport],
) -> None:
    """A research forecaster behind `RetryingProvider` still skips the stage.

    The composition root wraps every live provider in `RetryingProvider`
    (`windbreak.scheduler.provider_wiring`), so a declaration the wrapper
    swallowed would make this fix inert exactly where it is meant to fire. The
    wrapper forwards it, which is what lets the routing issue wire a research
    forecaster with no second edit here.
    """
    member = EnsembleMember("acme", "acme-forecaster-1", "2025-01-31")
    inner = _SelfResearchingStubProvider(member, _REPORTED_COST_MICROS)
    wrapped = RetryingProvider(
        inner,
        provider_name=member.provider,
        policy=RetryPolicy(
            max_attempts=DEFAULT_MAX_ATTEMPTS,
            total_deadline_ms=DEFAULT_TOTAL_DEADLINE_MS,
            backoff_base_ms=DEFAULT_BACKOFF_BASE_MS,
            max_cost_micros=DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS,
        ),
        price_table=DEFAULT_PROVIDER_PRICE_TABLE,
        rate_table=DEFAULT_MODEL_RATE_TABLE,
        monotonic_ms=lambda: 0,
        sleep_ms=lambda _ms: None,
    )
    search, fetch = _CountingSearchTransport(), _CountingFetchTransport()

    run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=_tools(research_tools_factory, tmp_path, search, fetch),
        min_ensemble_votes=1,
        ensemble=(member,),
        provider_factory=_single_member_factory(wrapped),
    )

    assert search.queries == []
    assert fetch.urls == []
    assert inner.calls == 1
