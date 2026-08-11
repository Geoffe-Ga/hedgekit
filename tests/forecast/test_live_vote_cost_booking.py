"""Tests for issue #399: a *successful* live vote must book its actual cost.

Before this issue, ``RetryingProvider`` accrued a provider's list price only
inside its ``except ProviderVoteError`` arm, so the price was charged once per
*failed* attempt and never for the attempt that actually succeeded. The inner
provider on the live path is a
:class:`~windbreak.forecast.providers.fixture.FixtureVoteProvider`, whose
``cost_micros`` is a hard ``0`` (correct for a cassette replay, which spends
nothing), and neither the Anthropic nor the OpenAI adapter parses any token
accounting out of the response envelope -- ``LlmTransport.complete`` returns
bare text. A live vote that succeeded on its first attempt therefore booked
``cost_micros == 0``.

That is a fail-open, not an accounting nit: ``ResearchBudget``'s per-forecast
and per-day ceilings are checked against what was *booked*, so a run whose
providers all succeed spent real money, booked none of it, and could not trip
its own ceiling however expensive it got.

The fix charges every attempt the wrapper makes, successful or not. Issue #451
then replaced *what* a successful attempt is charged: the list price was issue
#399's own sanctioned stopgap, and metering the response's reported token usage
at the model's configured rates is what finally makes the ceilings bound spend
rather than a count of attempts. Failed attempts -- which produce no response
and so no usage -- are still charged their list price. Neither figure can be
zero, so a live vote can never read as free: every mapped price and every rate
is validated strictly positive, an unmapped provider falls back to a positive
``unknown_provider_price_micros``, and an unmeasurable response falls back to a
positive ``unmetered_micros``.

Every test below is written against the public seams only. The two the issue
names as mandatory -- the per-forecast ceiling tripping on an all-*success*
run, and the day bucket totalling the actual charges across a mixed run --
drive the real :func:`~windbreak.forecast.pipeline.run_pipeline` through a
``provider_factory`` that wraps each ensemble member exactly the way
:func:`~windbreak.scheduler.provider_wiring.build_provider_factory` wraps it in
live mode, with ``transport=ForbiddenLiveTransport()`` proving no network is
touched. Both read their charge back through a ceiling trick (the repo's
established pattern -- see ``tests/forecast/test_budget.py``) rather than
reaching into ``ResearchBudget``'s private day map.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.budget import (
    DailyBudgetExhaustedError,
    InMemoryBudgetLedger,
    ModelRateTable,
    ModelTokenRate,
    PerForecastBudgetExceededError,
    ProviderPriceTable,
    ResearchBudget,
    TokenUsage,
)
from windbreak.forecast.cassettes import ForbiddenLiveTransport
from windbreak.forecast.pipeline import run_pipeline
from windbreak.forecast.providers import (
    DEFAULT_VOTE_ENSEMBLE,
    ProviderForecast,
)
from windbreak.forecast.providers.base import (
    ProviderCostOverrunError,
    ProviderTimeoutError,
)
from windbreak.forecast.providers.retry import RetryingProvider, RetryPolicy

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.providers import EnsembleMemberLike, ForecastProvider
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools
    from windbreak.forecast.sanitize import ResearchQuote

# --- Pinned figures ---------------------------------------------------------------

#: `windbreak.forecast.pipeline`'s private `_RESEARCH_COST_MICROS` stub cost for
#: a full run -- named locally (it is private), mirroring the identical
#: convention in `tests/forecast/test_budget.py` and
#: `tests/forecast/test_provider_failures.py`, so every exact-charge assertion
#: below reads against the same known figure.
_FULL_RUN_RESEARCH_COST_MICROS = 3_000_000

#: Per-attempt list price for the two default-ensemble OpenAI members.
_OPENAI_PRICE_MICROS = 500_000

#: Per-attempt list price for the single default-ensemble Anthropic member.
#: Deliberately *different* from `_OPENAI_PRICE_MICROS` so a run's total is
#: only correct if each vote is priced by its own provider rather than by one
#: flat per-vote figure.
_ANTHROPIC_PRICE_MICROS = 300_000

#: The fallback price charged for a provider absent from the table below. A
#: live vote for an unpriced provider must book this, never zero.
_UNKNOWN_PRICE_MICROS = 1_000_000

#: The price table every test here gates attempts through.
_PRICE_TABLE = ProviderPriceTable(
    prices_micros={
        "openai": _OPENAI_PRICE_MICROS,
        "anthropic": _ANTHROPIC_PRICE_MICROS,
    },
    unknown_provider_price_micros=_UNKNOWN_PRICE_MICROS,
)

#: The token usage every successful vote double below reports.
_VOTE_USAGE = TokenUsage(input_tokens=20_000, output_tokens=800)

#: What `_RATE_TABLE` charges `_VOTE_USAGE` for a `_MEMBER_A`/`_MEMBER_C`
#: (OpenAI) vote: ``20_000 * 2_000_000 + 800 * 20_000_000`` over a million.
#: Deliberately unequal to `_OPENAI_PRICE_MICROS` so a test cannot pass by
#: charging the pre-gate estimate instead of the meter.
_OPENAI_METERED_MICROS = 56_000

#: The same for the `_MEMBER_B` (Anthropic) vote, at that model's own rates:
#: ``20_000 * 4_000_000 + 800 * 25_000_000``. Distinct from the OpenAI figure,
#: so a run's total is only correct if each vote is metered at its own model's
#: rates rather than one flat figure.
_ANTHROPIC_METERED_MICROS = 100_000

#: The fail-closed charge for a vote whose cost cannot be derived. Distinct
#: from every price and every metered figure above.
_UNMETERED_MICROS = 900_000

#: The rate table every test here meters successful votes through.
_RATE_TABLE = ModelRateTable(
    rates={
        DEFAULT_VOTE_ENSEMBLE[0].model_version: ModelTokenRate(
            model_version=DEFAULT_VOTE_ENSEMBLE[0].model_version,
            input_micros_per_million_tokens=2_000_000,
            output_micros_per_million_tokens=20_000_000,
        ),
        DEFAULT_VOTE_ENSEMBLE[1].model_version: ModelTokenRate(
            model_version=DEFAULT_VOTE_ENSEMBLE[1].model_version,
            input_micros_per_million_tokens=4_000_000,
            output_micros_per_million_tokens=25_000_000,
        ),
        DEFAULT_VOTE_ENSEMBLE[2].model_version: ModelTokenRate(
            model_version=DEFAULT_VOTE_ENSEMBLE[2].model_version,
            input_micros_per_million_tokens=2_000_000,
            output_micros_per_million_tokens=20_000_000,
        ),
    },
    unmetered_micros=_UNMETERED_MICROS,
)

#: The three pinned default ensemble members (SPEC S6.3), indexed rather than
#: unpacked from the variable-length `DEFAULT_VOTE_ENSEMBLE` tuple.
_MEMBER_A = DEFAULT_VOTE_ENSEMBLE[0]
_MEMBER_B = DEFAULT_VOTE_ENSEMBLE[1]
_MEMBER_C = DEFAULT_VOTE_ENSEMBLE[2]

#: A bounded-retry policy generous in every dimension except the ones a given
#: test tightens, so only the behavior under test can fire.
_MAX_ATTEMPTS = 3
_TOTAL_DEADLINE_MS = 30_000
_BACKOFF_BASE_MS = 1_000
_MAX_COST_MICROS = 100_000_000

#: A per-forecast ceiling far above any figure charged here, used whenever a
#: test wants the run to *complete* so its day bucket can be read back.
_PERMISSIVE_PER_FORECAST_MICROS = 100_000_000

#: Surviving votes in the mixed run below: two successes, one timeout discard.
#: Clears the default `min_ensemble_votes=2` quorum, so the run completes.
_MIXED_RUN_SURVIVORS = 2


class _FakeClock:
    """A deterministic integer-millisecond clock for `RetryingProvider`.

    Mirrors `tests/forecast/test_provider_failures.py`'s helper of the same
    name (not imported across test modules per this package's rootdir-relative
    import convention -- see `tests/forecast/conftest.py`). No real
    `time.sleep` is ever invoked, so a retrying test is instant.
    """

    def __init__(self) -> None:
        """Initialize the clock at zero with an empty wait log."""
        self._now_ms = 0
        self.waits: list[int] = []

    def monotonic_ms(self) -> int:
        """Return the current internal clock reading, in milliseconds."""
        return self._now_ms

    def sleep_ms(self, milliseconds: int) -> None:
        """Advance the clock by `milliseconds` and record the wait.

        Args:
            milliseconds: How long to (fictitiously) sleep.
        """
        self.waits.append(milliseconds)
        self._now_ms += milliseconds


class _ZeroCostProvider:
    """A provider double returning a zero-cost forecast, as the fixture does.

    This is the shape that made issue #399 a fail-open: the live path's inner
    provider is a `FixtureVoteProvider`, which stamps `cost_micros=0` because
    it parses no token accounting from the response envelope.
    """

    def __init__(
        self, member: EnsembleMemberLike, *, usage: TokenUsage | None = _VOTE_USAGE
    ) -> None:
        """Store the member whose provenance stamps every returned forecast.

        Args:
            member: The ensemble member this provider votes for.
            usage: The token accounting every returned forecast reports
                (keyword-only); `None` models a response that reported none.
        """
        self._member = member
        self._usage = usage
        self.calls = 0

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[ResearchQuote, ...],
    ) -> ProviderForecast:
        """Record one call and return a valid, zero-cost, member-stamped vote.

        Args:
            market: The (unused) market under forecast.
            baseline: The (unused) baseline quote snapshot.
            vote_index: The (unused) zero-based vote index.
            quotes: The (unused) sanitized web quotes.

        Returns:
            A valid `ProviderForecast` whose `cost_micros` is zero.
        """
        del market, baseline, vote_index, quotes
        self.calls += 1
        return ProviderForecast(
            probability_ppm=500_000,
            rationale_summary="steady evidence",
            citations=(),
            cost_micros=0,
            provider=self._member.provider,
            model_version=self._member.model_version,
            training_cutoff=self._member.training_cutoff,
            response_fingerprint=hashlib.sha256(
                self._member.model_version.encode("utf-8")
            ).hexdigest(),
            usage=self._usage,
        )


class _TimingOutProvider:
    """A provider double raising a fresh `ProviderTimeoutError` every call."""

    def __init__(self) -> None:
        """Reset the call counter."""
        self.calls = 0

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[ResearchQuote, ...],
    ) -> ProviderForecast:
        """Record one call and time out.

        Args:
            market: The (unused) market under forecast.
            baseline: The (unused) baseline quote snapshot.
            vote_index: The (unused) zero-based vote index.
            quotes: The (unused) sanitized web quotes.

        Raises:
            ProviderTimeoutError: Always.
        """
        del market, baseline, vote_index, quotes
        self.calls += 1
        raise ProviderTimeoutError


def _policy(*, max_attempts: int = _MAX_ATTEMPTS) -> RetryPolicy:
    """Build a permissive bounded-retry policy.

    Args:
        max_attempts: The maximum provider attempts (keyword-only).

    Returns:
        A `RetryPolicy` generous in every dimension but `max_attempts`.
    """
    return RetryPolicy(
        max_attempts=max_attempts,
        total_deadline_ms=_TOTAL_DEADLINE_MS,
        backoff_base_ms=_BACKOFF_BASE_MS,
        max_cost_micros=_MAX_COST_MICROS,
    )


def _live_wrapped(
    inner: ForecastProvider,
    *,
    provider_name: str,
    price_table: ProviderPriceTable | None = None,
    rate_table: ModelRateTable | None = None,
) -> RetryingProvider:
    """Wrap `inner` exactly as the live composition root wraps a vote provider.

    Args:
        inner: The wrapped provider double.
        provider_name: The name priced against the table (keyword-only).
        price_table: The price table, defaulting to `_PRICE_TABLE`
            (keyword-only).
        rate_table: The per-model rate table, defaulting to `_RATE_TABLE`
            (keyword-only).

    Returns:
        The live-shaped `RetryingProvider`.
    """
    clock = _FakeClock()
    return RetryingProvider(
        inner,
        provider_name=provider_name,
        policy=_policy(),
        price_table=price_table if price_table is not None else _PRICE_TABLE,
        rate_table=rate_table if rate_table is not None else _RATE_TABLE,
        monotonic_ms=clock.monotonic_ms,
        sleep_ms=clock.sleep_ms,
    )


def _live_factory(
    routes: dict[str, ForecastProvider],
) -> Callable[[EnsembleMemberLike], ForecastProvider]:
    """Build a live-shaped `provider_factory` routing members by model version.

    Each routed provider is wrapped in a `RetryingProvider` priced by the
    member's *own* provider identifier, mirroring
    `windbreak.scheduler.provider_wiring.build_provider_factory`'s live branch.

    Args:
        routes: A `{model_version: ForecastProvider}` mapping covering every
            member the pipeline run will drive.

    Returns:
        A `provider_factory` closure.
    """

    def _factory(member: EnsembleMemberLike) -> ForecastProvider:
        """Return the live-wrapped provider routed for `member`.

        Args:
            member: The ensemble member being driven.

        Returns:
            The wrapped `ForecastProvider`.
        """
        return _live_wrapped(
            routes[member.model_version], provider_name=member.provider
        )

    return _factory


def _all_success_routes() -> dict[str, ForecastProvider]:
    """Route all three default members to a zero-cost success.

    Returns:
        A fresh `{model_version: ForecastProvider}` mapping.
    """
    return {
        member.model_version: _ZeroCostProvider(member)
        for member in (_MEMBER_A, _MEMBER_B, _MEMBER_C)
    }


def _mixed_routes() -> dict[str, ForecastProvider]:
    """Route the two OpenAI members to success and the Anthropic one to timeout.

    Two survivors clear the default `min_ensemble_votes=2` quorum, so the run
    completes and its day bucket can be read back.

    Returns:
        A fresh `{model_version: ForecastProvider}` mapping.
    """
    return {
        _MEMBER_A.model_version: _ZeroCostProvider(_MEMBER_A),
        _MEMBER_B.model_version: _TimingOutProvider(),
        _MEMBER_C.model_version: _ZeroCostProvider(_MEMBER_C),
    }


# --- Unit level: one successful vote books its price ------------------------------


def test_a_successful_live_vote_books_its_metered_cost_not_zero(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A first-attempt success books the successful attempt's metered cost.

    The regression this issue exists for: the inner provider reports a zero
    cost (it measures but never prices), so the wrapper is the only thing that
    can turn the call into money. Booking zero would make the vote read as
    free.
    """
    inner = _ZeroCostProvider(_MEMBER_A)
    retrying = _live_wrapped(inner, provider_name="openai")

    result = retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 1
    assert result.cost_micros == _OPENAI_METERED_MICROS


def test_a_successful_live_vote_is_metered_at_its_own_models_rates(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Each model's success books that model's own metered cost."""
    retrying = _live_wrapped(_ZeroCostProvider(_MEMBER_B), provider_name="anthropic")

    result = retrying.forecast(market, baseline, 0, ())

    assert result.cost_micros == _ANTHROPIC_METERED_MICROS


def test_a_successful_live_vote_reporting_no_usage_fails_closed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A response with no reported usage books the conservative fallback.

    Issue #399's fail-closed clause, carried forward to the metered path: where
    a vote's cost is not knowable, the vote is charged high rather than free --
    and, since issue #451, not at the list price either.
    """
    retrying = _live_wrapped(
        _ZeroCostProvider(_MEMBER_A, usage=None), provider_name="openai"
    )

    result = retrying.forecast(market, baseline, 0, ())

    assert result.cost_micros == _UNMETERED_MICROS


def test_an_unpriced_provider_is_still_gated_at_the_conservative_estimate(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """An unmapped provider's attempt is pre-gated at the high fallback.

    The property issue #399 established and issue #451 must not lose: the
    affordability pre-gate runs on the list price, and an unpriced provider
    gets `unknown_provider_price_micros` there -- so a member whose ceiling is
    one micro below that fallback cannot make even one attempt, and the inner
    provider is never called.
    """
    inner = _ZeroCostProvider(_MEMBER_A)
    clock = _FakeClock()
    retrying = RetryingProvider(
        inner,
        provider_name="a-provider-with-no-price",
        policy=RetryPolicy(
            max_attempts=_MAX_ATTEMPTS,
            total_deadline_ms=_TOTAL_DEADLINE_MS,
            backoff_base_ms=_BACKOFF_BASE_MS,
            max_cost_micros=_UNKNOWN_PRICE_MICROS - 1,
        ),
        price_table=_PRICE_TABLE,
        rate_table=_RATE_TABLE,
        monotonic_ms=clock.monotonic_ms,
        sleep_ms=clock.sleep_ms,
    )

    with pytest.raises(ProviderCostOverrunError) as excinfo:
        retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 0
    assert excinfo.value.ceiling_micros == _UNKNOWN_PRICE_MICROS - 1


def test_a_success_after_one_retry_books_every_attempt_it_made(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Two attempts (one timeout, one success) book one of each charge.

    Pins that fixing the success path did not double-count the failure path:
    the total is exactly one list price for the attempt that failed plus one
    metered charge for the attempt that returned a response.
    """

    class _FlakyProvider:
        """Times out once, then succeeds."""

        def __init__(self) -> None:
            """Reset the call counter."""
            self.calls = 0

        def forecast(
            self,
            market: NormalizedMarket,
            baseline: BaselineQuoteSnapshot,
            vote_index: int,
            quotes: tuple[ResearchQuote, ...],
        ) -> ProviderForecast:
            """Time out on the first call and succeed on the second.

            Args:
                market: The market under forecast.
                baseline: The baseline quote snapshot.
                vote_index: The zero-based vote index.
                quotes: The sanitized web quotes.

            Returns:
                A valid zero-cost forecast, on the second call onward.

            Raises:
                ProviderTimeoutError: On the first call.
            """
            self.calls += 1
            if self.calls == 1:
                raise ProviderTimeoutError
            return _ZeroCostProvider(_MEMBER_A).forecast(
                market, baseline, vote_index, quotes
            )

    inner = _FlakyProvider()
    retrying = _live_wrapped(inner, provider_name="openai")

    result = retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 2
    assert result.cost_micros == _OPENAI_PRICE_MICROS + _OPENAI_METERED_MICROS


# --- Issue #399 acceptance criterion 3: the ceiling trips on successes -------------


def test_the_per_forecast_ceiling_trips_on_a_run_of_successful_votes(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """An all-success run whose booked cost exceeds the ceiling halts.

    The test that makes the fix real. Before it, three successful votes booked
    zero, the run charged only the fixed research stub, and the per-forecast
    ceiling could not fire however many paid votes the run made -- so
    `ResearchBudget` was structurally unable to bound the healthy path.

    Every vote succeeds on its first attempt, so the run's charge is the stub
    plus exactly one list price per member. The ceiling sits one micro below
    that figure, which both proves the halt and pins the exact amount.
    """
    expected_charge = (
        _FULL_RUN_RESEARCH_COST_MICROS
        + _OPENAI_METERED_MICROS
        + _ANTHROPIC_METERED_MICROS
        + _OPENAI_METERED_MICROS
    )
    budget = ResearchBudget(
        per_forecast_micros=expected_charge - 1, ledger=InMemoryBudgetLedger()
    )

    with pytest.raises(PerForecastBudgetExceededError) as excinfo:
        run_pipeline(
            market,
            baseline,
            transport=ForbiddenLiveTransport(),
            created_at=created_at,
            research_tools=research_tools,
            budget=budget,
            provider_factory=_live_factory(_all_success_routes()),
        )

    assert excinfo.value.cost_micros == expected_charge
    assert excinfo.value.budget_micros == expected_charge - 1


def test_a_run_of_successful_votes_books_more_than_the_research_stub(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """A healthy all-success run books real vote spend, not just the stub.

    The positive companion to the ceiling test: under a ceiling generous
    enough for the run to finish, the day bucket still records every vote's
    price rather than collapsing to the fixed research stub.
    """
    expected_charge = (
        _FULL_RUN_RESEARCH_COST_MICROS
        + _OPENAI_METERED_MICROS
        + _ANTHROPIC_METERED_MICROS
        + _OPENAI_METERED_MICROS
    )
    budget = ResearchBudget(
        per_forecast_micros=_PERMISSIVE_PER_FORECAST_MICROS,
        per_day_micros=expected_charge,
        ledger=InMemoryBudgetLedger(),
    )

    run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        budget=budget,
        provider_factory=_live_factory(_all_success_routes()),
    )

    with pytest.raises(DailyBudgetExhaustedError) as excinfo:
        budget.ensure_day_open(at=created_at)
    assert excinfo.value.spent_micros == expected_charge
    assert excinfo.value.spent_micros > _FULL_RUN_RESEARCH_COST_MICROS


# --- Issue #399 acceptance criterion 4: the day bucket totals actual charges -------


def test_the_day_bucket_totals_actual_charges_across_a_mixed_run(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """The day bucket equals the summed charges of successes and failures.

    Two members succeed on their first attempt (one list price each) and one
    exhausts all three attempts on a persistent timeout (three list prices),
    on top of the fixed research stub. The day total is read back through the
    public `ensure_day_open` seam rather than the private day map.

    Before the fix the two successes contributed nothing, so the day bucket
    drifted below actual spend by exactly their price.
    """
    successes_micros = _OPENAI_METERED_MICROS + _OPENAI_METERED_MICROS
    failure_micros = _MAX_ATTEMPTS * _ANTHROPIC_PRICE_MICROS
    expected_total = _FULL_RUN_RESEARCH_COST_MICROS + successes_micros + failure_micros
    budget = ResearchBudget(
        per_forecast_micros=_PERMISSIVE_PER_FORECAST_MICROS,
        per_day_micros=expected_total,
        ledger=InMemoryBudgetLedger(),
    )

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        budget=budget,
        provider_factory=_live_factory(_mixed_routes()),
    )

    assert record.abstention_reason is None
    assert len(record.model_votes) == _MIXED_RUN_SURVIVORS
    with pytest.raises(DailyBudgetExhaustedError) as excinfo:
        budget.ensure_day_open(at=created_at)
    assert excinfo.value.spent_micros == expected_total
