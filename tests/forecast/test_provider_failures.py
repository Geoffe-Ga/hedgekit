"""Tests for issue #193's provider fault-injection taxonomy (Gate 1 RED).

Pins the new provider-failure taxonomy, retry/backoff semantics, per-attempt
pricing, and quorum/abstention behavior the chief architect's plan specifies:

* ``windbreak/forecast/providers/base.py`` gains a ``ProviderVoteError`` root
  (with a ``failure_code``/``response_fingerprint``/``cost_micros`` triple)
  that ``ProviderResponseRejectedError`` and ``ProviderVersionDriftError`` now
  subclass byte-identically, plus five new leaf error types
  (``ProviderTimeoutError``, ``ProviderRateLimitedError``,
  ``ProviderHTTPError``, ``ProviderMalformedResponseError``,
  ``ProviderCostOverrunError``).
* ``windbreak/forecast/budget.py`` gains ``ProviderPriceTable`` (a fail-closed,
  never-zero per-provider pricing map) and its pinned default.
* ``windbreak/forecast/providers/retry.py`` is an entirely new module: a
  ``RetryingProvider`` decorator implementing bounded retries, exponential
  backoff, an ``Retry-After``-aware rate-limit wait, a deadline clock, and an
  affordability pre-gate that raises ``ProviderCostOverrunError`` *before*
  ever calling the wrapped provider.
* ``windbreak/forecast/pipeline.py`` gains ``min_ensemble_votes`` and two new
  abstention reasons (``ensemble_quorum_not_met`` /
  ``provider_unavailable``) distinguishing "too few survivors" and "every
  discard was a transport fault" from the pre-existing
  ``all_votes_discarded``.

None of ``windbreak/forecast/providers/retry.py``,
``windbreak.forecast.budget.ProviderPriceTable``/``DEFAULT_PROVIDER_PRICE_TABLE``,
``windbreak.forecast.providers.base.ProviderVoteError`` and its five leaf
subclasses, or ``windbreak.forecast.pipeline``'s ``min_ensemble_votes`` seam
exist yet, so importing this module fails collection with
``ModuleNotFoundError: No module named 'windbreak.forecast.providers.retry'``
(the first unresolved import below) -- the expected Gate 1 RED state for
issue #193. Once those symbols land, the individual tests below still pin
exact string/int constants, exact retry counts and wait schedules, exact
budget charges (via the repo's established "one micro below the expected
charge" ceiling-trick pattern -- see ``tests/forecast/test_budget.py``), and
exact ledger payload shapes, so a partially-correct implementation fails on a
targeted ``AssertionError`` rather than a vague collection error.

Fault-injection strategy
    Every retry-semantics test drives a real ``RetryingProvider`` against a
    small, local ``ForecastProvider`` double (``_FailingProvider`` /
    ``_FlakyProvider`` / a plain success double) and a deterministic
    integer-millisecond ``_FakeClock`` -- never real ``time.sleep`` or
    ``datetime.now()``, so the whole suite is instant and reproducible. Every
    pipeline-integration test drives ``run_pipeline`` with a
    ``provider_factory`` routing each of the three pinned default ensemble
    members (keyed by their unique ``model_version``) to a fixed
    success/failure outcome, with ``transport=ForbiddenLiveTransport()``
    proving the fixture-transport seam is never touched on that path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.budget import (
    DEFAULT_PROVIDER_PRICE_TABLE,
    DEFAULT_UNKNOWN_PROVIDER_PRICE_MICROS,
    InMemoryBudgetLedger,
    PerForecastBudgetExceededError,
    ProviderPriceTable,
    ResearchBudget,
)
from windbreak.forecast.cassettes import ForbiddenLiveTransport
from windbreak.forecast.pipeline import (
    ABSTENTION_ALL_VOTES_DISCARDED,
    ABSTENTION_ENSEMBLE_QUORUM_NOT_MET,
    ABSTENTION_PROVIDER_UNAVAILABLE,
    DEFAULT_MIN_ENSEMBLE_VOTES,
    FORECAST_OUTPUT_DISCARDED_EVENT,
    InMemoryForecastLedger,
    run_pipeline,
)
from windbreak.forecast.providers import (
    DEFAULT_VOTE_ENSEMBLE,
    ProviderError,
    ProviderForecast,
    ProviderResponseRejectedError,
    ProviderVersionDriftError,
)
from windbreak.forecast.providers.base import (
    NO_RESPONSE_FINGERPRINT,
    PROVIDER_FAILURE_COST_OVERRUN,
    PROVIDER_FAILURE_RATE_LIMITED,
    PROVIDER_FAILURE_TIMEOUT,
    ProviderCostOverrunError,
    ProviderHTTPError,
    ProviderMalformedResponseError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderVoteError,
)
from windbreak.forecast.providers.retry import (
    DEFAULT_BACKOFF_BASE_MS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TOTAL_DEADLINE_MS,
    HTTP_TOO_MANY_REQUESTS,
    RetryingProvider,
    RetryPolicy,
    is_retryable_status,
)
from windbreak.forecast.records import forecast_record_to_payload
from windbreak.forecast.sanitize import (
    RESPONSE_FAILURE_HTTP_STATUS,
    RESPONSE_FAILURE_MALFORMED_VOTE_JSON,
    RESPONSE_FAILURE_VERSION_DRIFT,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.providers import (
        EnsembleMemberLike,
        ForecastProvider,
        ProviderCitation,
    )
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools
    from windbreak.forecast.sanitize import ResearchQuote

# --- Module-level literal pins (detect-secrets + DRY convention) -----------------
#
# Every new wire-format string/int this issue introduces is bound to a local
# constant here, then pinned against the imported production constant in
# `test_new_taxonomy_string_and_int_constants_are_pinned` below -- so a future
# accidental rename of e.g. `"provider_timeout"` is caught immediately rather
# than silently drifting the ledgered failure code.

_EXPECTED_PROVIDER_FAILURE_TIMEOUT = "provider_timeout"
_EXPECTED_PROVIDER_FAILURE_RATE_LIMITED = "provider_rate_limited"
_EXPECTED_PROVIDER_FAILURE_COST_OVERRUN = "provider_cost_overrun"
_EXPECTED_NO_RESPONSE_FINGERPRINT = "no_response"
_EXPECTED_DEFAULT_MAX_ATTEMPTS = 3
_EXPECTED_DEFAULT_TOTAL_DEADLINE_MS = 30_000
_EXPECTED_DEFAULT_BACKOFF_BASE_MS = 1_000
_EXPECTED_HTTP_TOO_MANY_REQUESTS = 429
_EXPECTED_DEFAULT_MIN_ENSEMBLE_VOTES = 2
_EXPECTED_ABSTENTION_ENSEMBLE_QUORUM_NOT_MET = "ensemble_quorum_not_met"
_EXPECTED_ABSTENTION_PROVIDER_UNAVAILABLE = "provider_unavailable"
_EXPECTED_ABSTENTION_ALL_VOTES_DISCARDED = "all_votes_discarded"
_EXPECTED_DEFAULT_UNKNOWN_PROVIDER_PRICE_MICROS = 1_000_000

#: `windbreak.forecast.pipeline`'s private `_RESEARCH_COST_MICROS` stub cost for
#: a full run -- named locally (it is private) mirroring
#: `tests/forecast/test_budget.py`'s identical convention, so every exact-charge
#: assertion below reads against the same known figure.
_FULL_RUN_RESEARCH_COST_MICROS = 3_000_000


def test_new_taxonomy_string_and_int_constants_are_pinned() -> None:
    """Every new literal failure-code / abstention-reason / retry-default
    constant this issue introduces is pinned exactly, so a future rename
    cannot silently drift the wire-format string or default value.
    """
    assert PROVIDER_FAILURE_TIMEOUT == _EXPECTED_PROVIDER_FAILURE_TIMEOUT
    assert PROVIDER_FAILURE_RATE_LIMITED == _EXPECTED_PROVIDER_FAILURE_RATE_LIMITED
    assert PROVIDER_FAILURE_COST_OVERRUN == _EXPECTED_PROVIDER_FAILURE_COST_OVERRUN
    assert NO_RESPONSE_FINGERPRINT == _EXPECTED_NO_RESPONSE_FINGERPRINT
    assert DEFAULT_MAX_ATTEMPTS == _EXPECTED_DEFAULT_MAX_ATTEMPTS
    assert DEFAULT_TOTAL_DEADLINE_MS == _EXPECTED_DEFAULT_TOTAL_DEADLINE_MS
    assert DEFAULT_BACKOFF_BASE_MS == _EXPECTED_DEFAULT_BACKOFF_BASE_MS
    assert HTTP_TOO_MANY_REQUESTS == _EXPECTED_HTTP_TOO_MANY_REQUESTS
    assert DEFAULT_MIN_ENSEMBLE_VOTES == _EXPECTED_DEFAULT_MIN_ENSEMBLE_VOTES
    assert (
        ABSTENTION_ENSEMBLE_QUORUM_NOT_MET
        == _EXPECTED_ABSTENTION_ENSEMBLE_QUORUM_NOT_MET
    )
    assert ABSTENTION_PROVIDER_UNAVAILABLE == _EXPECTED_ABSTENTION_PROVIDER_UNAVAILABLE
    assert ABSTENTION_ALL_VOTES_DISCARDED == _EXPECTED_ABSTENTION_ALL_VOTES_DISCARDED
    assert (
        DEFAULT_UNKNOWN_PROVIDER_PRICE_MICROS
        == _EXPECTED_DEFAULT_UNKNOWN_PROVIDER_PRICE_MICROS
    )


# --- Shared local test doubles ---------------------------------------------------


def _assert_json_safe_leaves(node: object) -> None:
    """Recursively assert every leaf of `node` is an int, str, or bool.

    Mirrors `tests/forecast/test_budget.py`'s helper of the same name (not
    imported across test modules per this package's rootdir-relative import
    convention -- see `tests/forecast/conftest.py`'s module docstring).

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


class _FakeClock:
    """A deterministic integer-millisecond clock for `RetryingProvider` tests.

    `monotonic_ms` returns the current internal clock; `sleep_ms` advances it
    by exactly the requested amount and records the wait -- no real
    `time.sleep` or `datetime.now()` is ever invoked, so every retry-backoff
    test is instant and fully reproducible (floats are banned repo-wide on
    `windbreak/forecast`; this clock is integer-only throughout).
    """

    def __init__(self, start_ms: int = 0) -> None:
        """Initialize the clock at `start_ms` with an empty wait log.

        Args:
            start_ms: The clock's initial reading, in milliseconds.
        """
        self._now_ms = start_ms
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


class _FailingProvider:
    """A `ForecastProvider` double that raises a fixed error on every call."""

    def __init__(self, error: BaseException) -> None:
        """Store the error every `forecast` call raises, and reset the counter.

        Args:
            error: The exception instance to raise, unmodified, every call.
        """
        self._error = error
        self.calls = 0

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[ResearchQuote, ...],
    ) -> ProviderForecast:
        """Record one call and raise the stored error, ignoring all arguments.

        Args:
            market: The (unused) market under forecast.
            baseline: The (unused) baseline quote snapshot.
            vote_index: The (unused) zero-based vote index.
            quotes: The (unused) sanitized web quotes.

        Raises:
            BaseException: The stored `self._error`, unconditionally.
        """
        self.calls += 1
        raise self._error


class _FlakyProvider:
    """A `ForecastProvider` double replaying a scripted outcome sequence.

    Each call pops the next outcome off the front of the script: a
    `BaseException` instance is raised, anything else is returned as the
    forecast result.
    """

    def __init__(self, outcomes: list[BaseException | ProviderForecast]) -> None:
        """Store the scripted outcome sequence and reset the call counter.

        Args:
            outcomes: The outcomes to replay, in call order.
        """
        self._outcomes = list(outcomes)
        self.calls = 0

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[ResearchQuote, ...],
    ) -> ProviderForecast:
        """Record one call and replay the next scripted outcome.

        Args:
            market: The (unused) market under forecast.
            baseline: The (unused) baseline quote snapshot.
            vote_index: The (unused) zero-based vote index.
            quotes: The (unused) sanitized web quotes.

        Returns:
            The next scripted `ProviderForecast`.

        Raises:
            BaseException: The next scripted error, if that is what is next.
        """
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _SucceedingProvider:
    """A `ForecastProvider` double returning one fixed forecast every call."""

    def __init__(self, forecast: ProviderForecast) -> None:
        """Store the forecast every `forecast` call returns.

        Args:
            forecast: The fixed `ProviderForecast` to return every time.
        """
        self._forecast = forecast
        self.calls = 0

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[ResearchQuote, ...],
    ) -> ProviderForecast:
        """Record one call and return the stored forecast, ignoring arguments.

        Args:
            market: The (unused) market under forecast.
            baseline: The (unused) baseline quote snapshot.
            vote_index: The (unused) zero-based vote index.
            quotes: The (unused) sanitized web quotes.

        Returns:
            The stored `ProviderForecast`, verbatim.
        """
        self.calls += 1
        return self._forecast


def _provider_forecast(
    *,
    probability_ppm: int = 500_000,
    rationale_summary: str = "steady evidence",
    citations: tuple[ProviderCitation, ...] = (),
    cost_micros: int = 0,
    provider: str = "openai",
    model_version: str = "gpt-5-forecast",
    training_cutoff: str = "2024-06-01",
    response_fingerprint: str = "f" * 64,
) -> ProviderForecast:
    """Build a valid `ProviderForecast`, defaulting every field.

    Args:
        probability_ppm: The forecast's probability estimate, in ppm.
        rationale_summary: The forecast's rationale summary.
        citations: The forecast's reported citations.
        cost_micros: The forecast's billed cost, in micros.
        provider: The producing provider identifier.
        model_version: The producing model's pinned version string.
        training_cutoff: The producing model's declared training cutoff.
        response_fingerprint: The sha256 fingerprint of the raw response.

    Returns:
        A valid `ProviderForecast` built from the given (or defaulted) fields.
    """
    return ProviderForecast(
        probability_ppm=probability_ppm,
        rationale_summary=rationale_summary,
        citations=citations,
        cost_micros=cost_micros,
        provider=provider,
        model_version=model_version,
        training_cutoff=training_cutoff,
        response_fingerprint=response_fingerprint,
    )


# --- Section 1: RetryingProvider unit tests ---------------------------------------

#: The provider name every retry test prices through `_TEST_PRICE_TABLE`.
_TEST_PROVIDER_NAME = "openai"

#: A deterministic, easy-to-eyeball per-attempt price for `_TEST_PROVIDER_NAME`.
_TEST_PRICE_MICROS = 100_000

#: A small, fully deterministic price table used by every retry unit test that
#: does not itself exercise unknown-provider pricing.
_TEST_PRICE_TABLE = ProviderPriceTable(
    prices_micros={_TEST_PROVIDER_NAME: _TEST_PRICE_MICROS},
    unknown_provider_price_micros=1_000_000,
)


def _retry_policy(
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    total_deadline_ms: int = DEFAULT_TOTAL_DEADLINE_MS,
    backoff_base_ms: int = DEFAULT_BACKOFF_BASE_MS,
    max_cost_micros: int = 100_000_000,
) -> RetryPolicy:
    """Build a `RetryPolicy`, defaulting every field to a large-permissive value.

    Args:
        max_attempts: The maximum retry attempts.
        total_deadline_ms: The total wall-clock deadline, in ms.
        backoff_base_ms: The exponential-backoff base, in ms.
        max_cost_micros: The affordability ceiling, in micros -- large by
            default so the affordability pre-gate only fires when a test
            deliberately tightens it.

    Returns:
        A constructed `RetryPolicy`.
    """
    return RetryPolicy(
        max_attempts=max_attempts,
        total_deadline_ms=total_deadline_ms,
        backoff_base_ms=backoff_base_ms,
        max_cost_micros=max_cost_micros,
    )


def _retrying_provider(
    inner: ForecastProvider,
    *,
    clock: _FakeClock,
    provider_name: str = _TEST_PROVIDER_NAME,
    policy: RetryPolicy | None = None,
    price_table: ProviderPriceTable | None = None,
) -> RetryingProvider:
    """Build a `RetryingProvider` wired to `clock`'s deterministic ms clock.

    Args:
        inner: The wrapped `ForecastProvider` double.
        clock: The fake clock supplying `monotonic_ms`/`sleep_ms`.
        provider_name: The provider name priced against `price_table`.
        policy: The retry policy; defaults to a large-permissive one.
        price_table: The per-attempt price table; defaults to
            `_TEST_PRICE_TABLE`.

    Returns:
        A constructed `RetryingProvider`.
    """
    return RetryingProvider(
        inner,
        provider_name=provider_name,
        policy=policy if policy is not None else _retry_policy(),
        price_table=price_table if price_table is not None else _TEST_PRICE_TABLE,
        monotonic_ms=clock.monotonic_ms,
        sleep_ms=clock.sleep_ms,
    )


def test_retry_default_constants_are_pinned() -> None:
    """The four `retry.py` module-level defaults are pinned to their exact
    documented values.
    """
    assert DEFAULT_MAX_ATTEMPTS == 3
    assert DEFAULT_TOTAL_DEADLINE_MS == 30_000
    assert DEFAULT_BACKOFF_BASE_MS == 1_000
    assert HTTP_TOO_MANY_REQUESTS == 429


@pytest.mark.parametrize("status_code", [429, 500, 550, 599])
def test_is_retryable_status_true_for_429_and_5xx_range(status_code: int) -> None:
    """`429` and every status in the inclusive `[500, 599]` range is retryable."""
    assert is_retryable_status(status_code) is True


@pytest.mark.parametrize("status_code", [200, 400, 404, 428, 499, 600])
def test_is_retryable_status_false_outside_the_retryable_set(status_code: int) -> None:
    """Every status outside `429` and the inclusive `[500, 599]` range is not
    retryable -- including the `428`/`499`/`600` off-by-one boundaries.
    """
    assert is_retryable_status(status_code) is False


@pytest.mark.parametrize(
    "field_name",
    ["max_attempts", "total_deadline_ms", "backoff_base_ms", "max_cost_micros"],
)
@pytest.mark.parametrize("bad_value", [0, -1])
def test_retry_policy_rejects_non_positive_fields(
    field_name: str, bad_value: int
) -> None:
    """Every `RetryPolicy` field is validated strictly positive; a `0` or
    negative value on any one field raises `ValueError`, mirroring the
    fail-closed money-path convention every other budget dataclass in this
    package uses.
    """
    kwargs: dict[str, int] = {
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
        "total_deadline_ms": DEFAULT_TOTAL_DEADLINE_MS,
        "backoff_base_ms": DEFAULT_BACKOFF_BASE_MS,
        "max_cost_micros": 100_000_000,
    }
    kwargs[field_name] = bad_value

    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_retrying_provider_exhausts_max_attempts_on_persistent_timeout(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A provider that always times out is retried up to `max_attempts` (3),
    then the same timeout error propagates with `cost_micros` equal to all
    three attempts' accrued price and the recorded exponential-backoff wait
    schedule between each attempt.
    """
    error = ProviderTimeoutError()
    inner = _FailingProvider(error)
    clock = _FakeClock()
    retrying = _retrying_provider(
        inner, clock=clock, policy=_retry_policy(max_attempts=3)
    )

    with pytest.raises(ProviderTimeoutError) as excinfo:
        retrying.forecast(market, baseline, 0, ())

    assert excinfo.value is error
    assert inner.calls == 3
    assert error.cost_micros == 3 * _TEST_PRICE_MICROS
    assert clock.waits == [DEFAULT_BACKOFF_BASE_MS, DEFAULT_BACKOFF_BASE_MS * 2]


def test_retrying_provider_recovers_after_one_timeout(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """One timeout then a successful response yields the inner forecast with
    `cost_micros` bumped by exactly one attempt's accrued price, after
    exactly two inner calls.
    """
    forecast = _provider_forecast(cost_micros=5_000)
    inner = _FlakyProvider([ProviderTimeoutError(), forecast])
    clock = _FakeClock()
    retrying = _retrying_provider(inner, clock=clock)

    result = retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 2
    assert result == dataclasses.replace(
        forecast, cost_micros=forecast.cost_micros + _TEST_PRICE_MICROS
    )


def test_retrying_provider_rate_limit_honors_retry_after_seconds(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `ProviderRateLimitedError(retry_after_seconds=N)` waits exactly
    `N * 1000` ms -- not the exponential-backoff schedule.
    """
    forecast = _provider_forecast()
    inner = _FlakyProvider([ProviderRateLimitedError(retry_after_seconds=7), forecast])
    clock = _FakeClock()
    retrying = _retrying_provider(
        inner, clock=clock, policy=_retry_policy(total_deadline_ms=60_000)
    )

    retrying.forecast(market, baseline, 0, ())

    assert clock.waits == [7_000]


def test_retrying_provider_retry_after_past_deadline_raises_without_sleeping(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `retry_after_seconds` that would push past the total deadline raises
    immediately -- the provider never sleeps past its budgeted deadline.
    """
    error = ProviderRateLimitedError(retry_after_seconds=100)
    inner = _FailingProvider(error)
    clock = _FakeClock()
    retrying = _retrying_provider(
        inner, clock=clock, policy=_retry_policy(total_deadline_ms=5_000)
    )

    with pytest.raises(ProviderRateLimitedError):
        retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 1
    assert clock.waits == []


def test_retrying_provider_deadline_exhaustion_stops_before_max_attempts(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A short total deadline halts retries before `max_attempts` is reached,
    even though the failure itself (a timeout) is otherwise retryable: the
    first backoff wait (1000ms) fits the 1500ms deadline, but the second
    (2000ms) would not.
    """
    error = ProviderTimeoutError()
    inner = _FailingProvider(error)
    clock = _FakeClock()
    retrying = _retrying_provider(
        inner,
        clock=clock,
        policy=_retry_policy(
            max_attempts=10, backoff_base_ms=1_000, total_deadline_ms=1_500
        ),
    )

    with pytest.raises(ProviderTimeoutError):
        retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 2
    assert clock.waits == [1_000]


def test_retrying_provider_retries_http_503(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A 5xx `ProviderHTTPError` is retryable: one 503 then success yields
    the inner forecast after exactly two inner calls.
    """
    forecast = _provider_forecast()
    inner = _FlakyProvider([ProviderHTTPError(503, "a" * 64), forecast])
    clock = _FakeClock()
    retrying = _retrying_provider(inner, clock=clock)

    result = retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 2
    assert result.cost_micros == forecast.cost_micros + _TEST_PRICE_MICROS


def test_retrying_provider_retries_http_429_using_backoff_not_retry_after(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A raw `ProviderHTTPError(429, ...)` -- distinct from the semantic
    `ProviderRateLimitedError` -- has no `retry_after_seconds`, so it waits
    the exponential-backoff schedule, never an HTTP-header-derived delay.
    """
    forecast = _provider_forecast()
    inner = _FlakyProvider(
        [ProviderHTTPError(HTTP_TOO_MANY_REQUESTS, "b" * 64), forecast]
    )
    clock = _FakeClock()
    retrying = _retrying_provider(inner, clock=clock)

    retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 2
    assert clock.waits == [DEFAULT_BACKOFF_BASE_MS]


def test_retrying_provider_does_not_retry_http_404(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A non-retryable status (`404`) is never retried: exactly one inner
    call, and the error's `cost_micros` reflects the single charged attempt.
    """
    error = ProviderHTTPError(404, "c" * 64)
    inner = _FailingProvider(error)
    clock = _FakeClock()
    retrying = _retrying_provider(inner, clock=clock)

    with pytest.raises(ProviderHTTPError):
        retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 1
    assert error.cost_micros == _TEST_PRICE_MICROS
    assert clock.waits == []


def test_retrying_provider_does_not_retry_malformed_response(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `ProviderMalformedResponseError` is never retried: exactly one inner
    call, and `cost_micros` reflects the single charged attempt.
    """
    error = ProviderMalformedResponseError("d" * 64)
    inner = _FailingProvider(error)
    clock = _FakeClock()
    retrying = _retrying_provider(inner, clock=clock)

    with pytest.raises(ProviderMalformedResponseError):
        retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 1
    assert error.cost_micros == _TEST_PRICE_MICROS


def test_retrying_provider_passes_through_response_rejected_without_retry(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `ProviderResponseRejectedError` (a screen-side rejection, not a
    transport fault) is never retried, and its `cost_micros` is stamped with
    the single charged attempt before re-raising.
    """
    error = ProviderResponseRejectedError(
        RESPONSE_FAILURE_MALFORMED_VOTE_JSON, "e" * 64
    )
    inner = _FailingProvider(error)
    clock = _FakeClock()
    retrying = _retrying_provider(inner, clock=clock)

    with pytest.raises(ProviderResponseRejectedError):
        retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 1
    assert error.cost_micros == _TEST_PRICE_MICROS


def test_retrying_provider_passes_through_version_drift_without_retry(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `ProviderVersionDriftError` is never retried, and its `cost_micros`
    is stamped with the single charged attempt before re-raising.
    """
    error = ProviderVersionDriftError("v9", ("v1", "v2"), "f" * 64)
    inner = _FailingProvider(error)
    clock = _FakeClock()
    retrying = _retrying_provider(inner, clock=clock)

    with pytest.raises(ProviderVersionDriftError):
        retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 1
    assert error.cost_micros == _TEST_PRICE_MICROS


def test_retrying_provider_does_not_retry_cost_overrun_error(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `ProviderCostOverrunError` raised by the inner provider is itself
    never retried (retrying an over-budget provider would only compound the
    overrun): exactly one inner call.
    """
    error = ProviderCostOverrunError(cost_micros=999, ceiling_micros=1)
    inner = _FailingProvider(error)
    clock = _FakeClock()
    retrying = _retrying_provider(inner, clock=clock)

    with pytest.raises(ProviderCostOverrunError):
        retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 1


def test_retrying_provider_affordability_pre_gate_blocks_before_first_attempt(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """When even one attempt's price would exceed `max_cost_micros`, the
    provider raises `ProviderCostOverrunError` *before* ever calling the
    wrapped provider -- proving no unbudgeted spend can occur.
    """
    inner = _FailingProvider(ProviderTimeoutError())
    clock = _FakeClock()
    retrying = _retrying_provider(
        inner,
        clock=clock,
        policy=_retry_policy(max_cost_micros=_TEST_PRICE_MICROS - 1),
    )

    with pytest.raises(ProviderCostOverrunError) as excinfo:
        retrying.forecast(market, baseline, 0, ())

    assert inner.calls == 0
    assert excinfo.value.cost_micros == 0
    assert excinfo.value.ceiling_micros == _TEST_PRICE_MICROS - 1


def test_retrying_provider_prices_unknown_provider_at_the_ceiling(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A provider name absent from the price table is charged the ceiling on
    a failed attempt, proving an unknown provider is never silently treated
    as free.
    """
    price_table = ProviderPriceTable(
        prices_micros={}, unknown_provider_price_micros=250_000
    )
    error = ProviderTimeoutError()
    inner = _FailingProvider(error)
    clock = _FakeClock()
    retrying = _retrying_provider(
        inner,
        clock=clock,
        provider_name="unlisted-provider",
        policy=_retry_policy(max_attempts=1),
        price_table=price_table,
    )

    with pytest.raises(ProviderTimeoutError):
        retrying.forecast(market, baseline, 0, ())

    assert error.cost_micros == 250_000


def test_retrying_provider_clean_success_is_byte_equal_to_inner_forecast(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A zero-failed-attempts success returns a forecast byte-equal to the
    inner provider's own result -- the retry wrapper is invisible on the
    happy path.
    """
    forecast = _provider_forecast(cost_micros=42)
    inner = _SucceedingProvider(forecast)
    clock = _FakeClock()
    retrying = _retrying_provider(inner, clock=clock)

    result = retrying.forecast(market, baseline, 0, ())

    assert result == forecast
    assert inner.calls == 1
    assert clock.waits == []


def test_retrying_provider_propagates_non_taxonomy_exception_untouched(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A plain, non-`ProviderVoteError` exception (an unrelated bug) is never
    caught, retried, or mutated -- it propagates exactly as raised.
    """

    class _BoomError(RuntimeError):
        """A non-taxonomy exception unrelated to the provider fault model."""

    boom = _BoomError("unexpected bug")
    inner = _FailingProvider(boom)
    clock = _FakeClock()
    retrying = _retrying_provider(inner, clock=clock)

    with pytest.raises(_BoomError) as excinfo:
        retrying.forecast(market, baseline, 0, ())

    assert excinfo.value is boom
    assert inner.calls == 1
    assert clock.waits == []


# --- Section 2: base.py taxonomy + budget price table --------------------------


def test_provider_vote_error_base_class_stores_all_four_fields() -> None:
    """`ProviderVoteError` stores `message`/`failure_code`/
    `response_fingerprint`/`cost_micros` and renders `message` as `str(error)`.
    """
    error = ProviderVoteError(
        "custom failure",
        failure_code="custom_code",
        response_fingerprint="z" * 64,
        cost_micros=42,
    )

    assert str(error) == "custom failure"
    assert error.failure_code == "custom_code"
    assert error.response_fingerprint == "z" * 64
    assert error.cost_micros == 42
    assert isinstance(error, ProviderError)


def test_provider_vote_error_defaults_cost_micros_to_zero() -> None:
    """`ProviderVoteError.cost_micros` defaults to `0` when omitted."""
    error = ProviderVoteError("x", failure_code="c", response_fingerprint="f")

    assert error.cost_micros == 0


def test_provider_response_rejected_error_constructor_is_byte_identical() -> None:
    """`ProviderResponseRejectedError`'s constructor signature, message text,
    and default `cost_micros=0` are byte-identical now that it subclasses
    `ProviderVoteError` -- and it is now both a `ProviderVoteError` and a
    `ProviderError`.
    """
    error = ProviderResponseRejectedError("http_status_error", "abc")

    assert error.failure_code == "http_status_error"
    assert error.response_fingerprint == "abc"
    assert error.cost_micros == 0
    assert str(error) == (
        "provider response rejected (http_status_error); fingerprint abc"
    )
    assert isinstance(error, ProviderVoteError)
    assert isinstance(error, ProviderError)


def test_provider_version_drift_error_constructor_is_byte_identical() -> None:
    """`ProviderVersionDriftError`'s constructor signature, message text, and
    default `cost_micros=0` are byte-identical now that it subclasses
    `ProviderVoteError` -- and it is now both a `ProviderVoteError` and a
    `ProviderError`.
    """
    error = ProviderVersionDriftError("v2", ("v1",), "xyz")

    assert error.failure_code == RESPONSE_FAILURE_VERSION_DRIFT
    assert error.response_fingerprint == "xyz"
    assert error.reported_version == "v2"
    assert error.pinned_versions == ("v1",)
    assert error.cost_micros == 0
    assert str(error) == "forecaster version 'v2' drifted from pinned set ('v1',)"
    assert isinstance(error, ProviderVoteError)
    assert isinstance(error, ProviderError)


def test_provider_timeout_error_taxonomy() -> None:
    """`ProviderTimeoutError()` carries the timeout failure code, the
    no-response sentinel fingerprint, and a zero default cost.
    """
    error = ProviderTimeoutError()

    assert isinstance(error, ProviderVoteError)
    assert error.failure_code == PROVIDER_FAILURE_TIMEOUT
    assert error.response_fingerprint == NO_RESPONSE_FINGERPRINT
    assert error.cost_micros == 0


def test_provider_rate_limited_error_defaults_and_carries_retry_after() -> None:
    """`ProviderRateLimitedError` defaults `retry_after_seconds` to `None`
    and exposes an explicit value when given one, alongside the rate-limit
    failure code and the no-response sentinel fingerprint.
    """
    no_hint = ProviderRateLimitedError()
    with_hint = ProviderRateLimitedError(retry_after_seconds=30)

    assert no_hint.retry_after_seconds is None
    assert no_hint.failure_code == PROVIDER_FAILURE_RATE_LIMITED
    assert no_hint.response_fingerprint == NO_RESPONSE_FINGERPRINT
    assert with_hint.retry_after_seconds == 30


def test_provider_rate_limited_error_rejects_bool_retry_after() -> None:
    """A `bool` (an `int` subclass) must never masquerade as
    `retry_after_seconds`, mirroring `_require_probability_ppm`'s convention.
    """
    with pytest.raises(TypeError):
        ProviderRateLimitedError(retry_after_seconds=True)


def test_provider_rate_limited_error_rejects_negative_retry_after() -> None:
    """A negative `retry_after_seconds` is rejected."""
    with pytest.raises(ValueError):
        ProviderRateLimitedError(retry_after_seconds=-1)


def test_provider_http_error_reuses_sanitize_http_status_failure_code() -> None:
    """`ProviderHTTPError` reuses `sanitize.RESPONSE_FAILURE_HTTP_STATUS` as
    its failure code (never inventing a parallel taxonomy) and exposes
    `status_code`.
    """
    error = ProviderHTTPError(503, "c" * 64)

    assert isinstance(error, ProviderVoteError)
    assert error.failure_code == RESPONSE_FAILURE_HTTP_STATUS
    assert error.status_code == 503
    assert error.response_fingerprint == "c" * 64


def test_provider_malformed_response_error_reuses_sanitize_malformed_code() -> None:
    """`ProviderMalformedResponseError` reuses
    `sanitize.RESPONSE_FAILURE_MALFORMED_VOTE_JSON` as its failure code.
    """
    error = ProviderMalformedResponseError("d" * 64)

    assert isinstance(error, ProviderVoteError)
    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON
    assert error.response_fingerprint == "d" * 64


def test_provider_cost_overrun_error_taxonomy() -> None:
    """`ProviderCostOverrunError` carries the cost-overrun failure code, the
    no-response sentinel fingerprint, and exposes both `cost_micros` and
    `ceiling_micros`.
    """
    error = ProviderCostOverrunError(cost_micros=500_000, ceiling_micros=400_000)

    assert isinstance(error, ProviderVoteError)
    assert error.failure_code == PROVIDER_FAILURE_COST_OVERRUN
    assert error.response_fingerprint == NO_RESPONSE_FINGERPRINT
    assert error.cost_micros == 500_000
    assert error.ceiling_micros == 400_000


def test_provider_price_table_returns_the_mapped_entry_for_a_known_provider() -> None:
    """A provider present in the price map returns its exact mapped price."""
    table = ProviderPriceTable(
        prices_micros={"openai": 10}, unknown_provider_price_micros=99
    )

    assert table.price_micros("openai") == 10


def test_provider_price_table_unknown_provider_prices_at_the_ceiling() -> None:
    """A provider absent from the price map falls back to the unknown ceiling."""
    table = ProviderPriceTable(
        prices_micros={"openai": 10}, unknown_provider_price_micros=99
    )

    assert table.price_micros("mystery-provider") == 99


@pytest.mark.parametrize("bad_entry", [0, -1])
def test_provider_price_table_rejects_a_non_positive_entry(bad_entry: int) -> None:
    """A zero or negative per-provider price is rejected -- never zero, since
    a zero-priced entry on this fail-closed money path would let a runaway
    provider spend for free.
    """
    with pytest.raises(ValueError):
        ProviderPriceTable(
            prices_micros={"openai": bad_entry}, unknown_provider_price_micros=99
        )


@pytest.mark.parametrize("bad_ceiling", [0, -1])
def test_provider_price_table_rejects_a_non_positive_ceiling(bad_ceiling: int) -> None:
    """A zero or negative unknown-provider ceiling is rejected."""
    with pytest.raises(ValueError):
        ProviderPriceTable(prices_micros={}, unknown_provider_price_micros=bad_ceiling)


def test_default_provider_price_table_pins_known_lookups_and_the_ceiling() -> None:
    """`DEFAULT_PROVIDER_PRICE_TABLE` prices every documented provider above
    zero, and an unrecognized provider prices at the pinned default ceiling.
    """
    assert DEFAULT_UNKNOWN_PROVIDER_PRICE_MICROS == 1_000_000
    assert DEFAULT_PROVIDER_PRICE_TABLE.price_micros("openai") > 0
    assert DEFAULT_PROVIDER_PRICE_TABLE.price_micros("anthropic") > 0
    assert DEFAULT_PROVIDER_PRICE_TABLE.price_micros("futuresearch") > 0
    assert (
        DEFAULT_PROVIDER_PRICE_TABLE.price_micros("a-totally-unknown-provider")
        == DEFAULT_UNKNOWN_PROVIDER_PRICE_MICROS
    )


# --- Section 3: run_pipeline fault-injection integration ------------------------

#: The three pinned default ensemble members (SPEC S6.3), indexed rather than
#: unpacked from the variable-length `DEFAULT_VOTE_ENSEMBLE` tuple.
_MEMBER_A = DEFAULT_VOTE_ENSEMBLE[0]
_MEMBER_B = DEFAULT_VOTE_ENSEMBLE[1]
_MEMBER_C = DEFAULT_VOTE_ENSEMBLE[2]


def _success_provider(member: EnsembleMemberLike) -> ForecastProvider:
    """Build a `_SucceedingProvider` returning a forecast stamped for `member`.

    Args:
        member: The ensemble member this provider's forecast is stamped with.

    Returns:
        A `_SucceedingProvider` returning a valid, member-stamped forecast.
    """
    return _SucceedingProvider(
        _provider_forecast(
            provider=member.provider,
            model_version=member.model_version,
            training_cutoff=member.training_cutoff,
            response_fingerprint=hashlib.sha256(
                member.model_version.encode("utf-8")
            ).hexdigest(),
        )
    )


def _routed_provider_factory(
    routes: dict[str, ForecastProvider],
) -> Callable[[EnsembleMemberLike], ForecastProvider]:
    """Build a `provider_factory` routing each member by its `model_version`.

    Args:
        routes: A `{model_version: ForecastProvider}` mapping covering every
            member the pipeline run will drive.

    Returns:
        A `provider_factory` closure looking `member.model_version` up in
        `routes`.
    """

    def _factory(member: EnsembleMemberLike) -> ForecastProvider:
        """Return the provider routed for `member`'s pinned model version.

        Args:
            member: The ensemble member being driven.

        Returns:
            The routed `ForecastProvider`.
        """
        return routes[member.model_version]

    return _factory


def _two_success_one_timeout_routes() -> dict[str, ForecastProvider]:
    """Route `_MEMBER_A`/`_MEMBER_C` to success and `_MEMBER_B` to a
    persistent timeout, fresh on every call (never a shared, stateful double
    across independent pipeline runs).

    Returns:
        A `{model_version: ForecastProvider}` mapping for the three default
        ensemble members.
    """
    return {
        _MEMBER_A.model_version: _success_provider(_MEMBER_A),
        _MEMBER_B.model_version: _FailingProvider(ProviderTimeoutError()),
        _MEMBER_C.model_version: _success_provider(_MEMBER_C),
    }


def test_pipeline_one_member_timeout_yields_full_live_record_with_one_discard(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """One of three ensemble members times out on every attempt; the other
    two survive, meeting the default `min_ensemble_votes=2` quorum, so the
    run produces a full, live-eligible 2-vote record with exactly one
    `FORECAST_OUTPUT_DISCARDED` event carrying the timeout failure code and
    the no-response sentinel fingerprint.
    """
    ledger = InMemoryForecastLedger()

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_factory=_routed_provider_factory(_two_success_one_timeout_routes()),
    )

    assert len(record.model_votes) == 2
    assert record.abstention_reason is None
    assert record.eligible_for_live is True
    discard_events = ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)
    assert len(discard_events) == 1
    event = discard_events[0]
    assert event.payload["failure"] == PROVIDER_FAILURE_TIMEOUT
    assert event.payload["response_fingerprint"] == NO_RESPONSE_FINGERPRINT
    assert set(event.payload) == {
        "market_ticker",
        "provider",
        "model_version",
        "vote_index",
        "failure",
        "response_fingerprint",
    }


def test_pipeline_one_member_timeout_charges_the_exact_research_cost(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """The one-timeout, two-survivor run's total research charge is exactly
    `_FULL_RUN_RESEARCH_COST_MICROS` (both survivors and the discarded
    `ProviderTimeoutError` cost zero), proven via the repo's established
    ceiling-trick pattern: a budget one micro below the expected charge
    raises with `cost_micros` equal to the exact figure.
    """
    ledger = InMemoryForecastLedger()
    budget_ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(
        per_forecast_micros=_FULL_RUN_RESEARCH_COST_MICROS - 1, ledger=budget_ledger
    )

    with pytest.raises(PerForecastBudgetExceededError) as excinfo:
        run_pipeline(
            market,
            baseline,
            transport=ForbiddenLiveTransport(),
            created_at=created_at,
            research_tools=research_tools,
            ledger=ledger,
            budget=budget,
            provider_factory=_routed_provider_factory(
                _two_success_one_timeout_routes()
            ),
        )

    assert excinfo.value.cost_micros == _FULL_RUN_RESEARCH_COST_MICROS


def test_pipeline_two_member_failures_trigger_quorum_abstention(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """Two of three members fail (one timeout, one malformed response),
    leaving a single survivor -- below the default `min_ensemble_votes=2`
    quorum -- so the run abstains with `ABSTENTION_ENSEMBLE_QUORUM_NOT_MET`
    rather than aggregating over a single vote, collapsing probability to
    the baseline.
    """
    ledger = InMemoryForecastLedger()
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _success_provider(_MEMBER_A),
        _MEMBER_B.model_version: _FailingProvider(ProviderTimeoutError()),
        _MEMBER_C.model_version: _FailingProvider(
            ProviderMalformedResponseError("e" * 64)
        ),
    }

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_factory=_routed_provider_factory(routes),
    )

    baseline_ppm = baseline.price_pips * 100
    assert record.abstention_reason == ABSTENTION_ENSEMBLE_QUORUM_NOT_MET
    assert record.eligible_for_live is False
    assert record.model_votes == ()
    assert record.probability_ppm == baseline_ppm
    assert record.ci_low_ppm == baseline_ppm
    assert record.ci_high_ppm == baseline_ppm
    assert len(ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)) == 2


def test_pipeline_all_transport_failures_yield_provider_unavailable_abstention(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """All three members fail with transport-class errors (timeout,
    rate-limited, retryable HTTP status): zero survivors, and every discard
    is transport-class, so the run abstains with the new, more specific
    `ABSTENTION_PROVIDER_UNAVAILABLE` reason -- not the pre-existing
    `ABSTENTION_ALL_VOTES_DISCARDED` -- with exactly three discard events and
    a baseline-collapsed probability.
    """
    ledger = InMemoryForecastLedger()
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _FailingProvider(ProviderTimeoutError()),
        _MEMBER_B.model_version: _FailingProvider(
            ProviderRateLimitedError(retry_after_seconds=5)
        ),
        _MEMBER_C.model_version: _FailingProvider(ProviderHTTPError(503, "g" * 64)),
    }

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_factory=_routed_provider_factory(routes),
    )

    baseline_ppm = baseline.price_pips * 100
    assert record.abstention_reason == ABSTENTION_PROVIDER_UNAVAILABLE
    assert record.eligible_for_live is False
    assert record.model_votes == ()
    assert record.probability_ppm == baseline_ppm
    discard_events = ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)
    assert len(discard_events) == 3
    failures = {event.payload["failure"] for event in discard_events}
    assert failures == {
        PROVIDER_FAILURE_TIMEOUT,
        PROVIDER_FAILURE_RATE_LIMITED,
        RESPONSE_FAILURE_HTTP_STATUS,
    }


def test_pipeline_mixed_zero_survivor_screen_and_transport_yields_all_votes_discarded(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """Zero survivors, but not every discard is transport-class (one member's
    response is screen-rejected, not a transport fault): the run abstains
    with the pre-existing `ABSTENTION_ALL_VOTES_DISCARDED`, not the new
    `ABSTENTION_PROVIDER_UNAVAILABLE` -- distinguishing "no provider could be
    reached" from "the providers responded, but every response was rejected".
    """
    ledger = InMemoryForecastLedger()
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _FailingProvider(
            ProviderResponseRejectedError(
                RESPONSE_FAILURE_MALFORMED_VOTE_JSON, "h" * 64
            )
        ),
        _MEMBER_B.model_version: _FailingProvider(ProviderTimeoutError()),
        _MEMBER_C.model_version: _FailingProvider(ProviderTimeoutError()),
    }

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_factory=_routed_provider_factory(routes),
    )

    assert record.abstention_reason == ABSTENTION_ALL_VOTES_DISCARDED
    assert record.eligible_for_live is False
    assert record.model_votes == ()
    assert len(ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)) == 3


def test_pipeline_all_non_retryable_http_failures_yield_all_votes_discarded(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """All three members fail with a *non-retryable* `ProviderHTTPError` (a
    404): every provider was reached and responded, so despite zero survivors
    this is not a transport wipeout. The run must abstain with the pre-existing
    `ABSTENTION_ALL_VOTES_DISCARDED` -- never `ABSTENTION_PROVIDER_UNAVAILABLE`,
    whose "no provider could be reached / retryable HTTP status" rationale would
    be factually false when the providers answered with a 404.
    """
    ledger = InMemoryForecastLedger()
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _FailingProvider(ProviderHTTPError(404, "i" * 64)),
        _MEMBER_B.model_version: _FailingProvider(ProviderHTTPError(403, "j" * 64)),
        _MEMBER_C.model_version: _FailingProvider(ProviderHTTPError(400, "k" * 64)),
    }

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_factory=_routed_provider_factory(routes),
    )

    assert record.abstention_reason == ABSTENTION_ALL_VOTES_DISCARDED
    assert record.eligible_for_live is False
    assert record.model_votes == ()
    assert len(ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)) == 3


def test_pipeline_mixed_retryable_and_non_retryable_http_yields_all_votes_discarded(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """A zero-survivor wipeout mixing a *retryable* HTTP 503 member with a
    *non-retryable* HTTP 404 member (and a timeout): because not every discard
    is transport-class -- the 404 means a provider was reached and responded --
    the run abstains with `ABSTENTION_ALL_VOTES_DISCARDED`, not the
    `ABSTENTION_PROVIDER_UNAVAILABLE` a same-status-code check would wrongly
    stamp when every failure code is `RESPONSE_FAILURE_HTTP_STATUS`.
    """
    ledger = InMemoryForecastLedger()
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _FailingProvider(ProviderHTTPError(503, "l" * 64)),
        _MEMBER_B.model_version: _FailingProvider(ProviderHTTPError(404, "m" * 64)),
        _MEMBER_C.model_version: _FailingProvider(ProviderTimeoutError()),
    }

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_factory=_routed_provider_factory(routes),
    )

    assert record.abstention_reason == ABSTENTION_ALL_VOTES_DISCARDED
    assert record.eligible_for_live is False
    assert record.model_votes == ()
    assert len(ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)) == 3


def test_pipeline_min_ensemble_votes_three_turns_a_two_survivor_run_into_abstention(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """Raising `min_ensemble_votes` to `3` turns the same two-survivor,
    one-timeout scenario that is a full record under the default quorum of
    `2` (see the discard test above) into a quorum abstention -- the knob
    genuinely changes the outcome.
    """
    ledger = InMemoryForecastLedger()

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_factory=_routed_provider_factory(_two_success_one_timeout_routes()),
        min_ensemble_votes=3,
    )

    assert record.abstention_reason == ABSTENTION_ENSEMBLE_QUORUM_NOT_MET


def test_run_pipeline_rejects_non_positive_min_ensemble_votes(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
    make_fake_vote_transport: Callable[..., object],
) -> None:
    """`min_ensemble_votes=0` is a usage error, rejected loudly before any
    stage runs.
    """
    with pytest.raises(ValueError):
        run_pipeline(
            market,
            baseline,
            transport=make_fake_vote_transport(),
            created_at=created_at,
            research_tools=research_tools,
            min_ensemble_votes=0,
        )


def test_pipeline_cost_overrun_member_run_survives_with_a_generous_budget(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """A member's provider raises `ProviderCostOverrunError` mid-run: the run
    never crashes -- the vote is discarded and ledgered with the
    `provider_cost_overrun` failure code, while the other two members'
    votes survive into a full record.
    """
    ledger = InMemoryForecastLedger()
    # A genuinely generous per-forecast ceiling: comfortably above the run's
    # full charge (`_FULL_RUN_RESEARCH_COST_MICROS` + the 750_000 discarded
    # cost-overrun member cost), so the fail-closed budget never trips and the
    # two surviving votes aggregate into a full record. The default ceiling
    # (`DEFAULT_PER_FORECAST_BUDGET_MICROS`, 3_000_000) is *below* that charge,
    # so it would raise -- which the adjacent ceiling-trick test asserts.
    generous_budget = ResearchBudget(
        per_forecast_micros=10_000_000, ledger=InMemoryBudgetLedger()
    )
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _success_provider(_MEMBER_A),
        _MEMBER_B.model_version: _FailingProvider(
            ProviderCostOverrunError(cost_micros=750_000, ceiling_micros=1_000_000)
        ),
        _MEMBER_C.model_version: _success_provider(_MEMBER_C),
    }

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        budget=generous_budget,
        provider_factory=_routed_provider_factory(routes),
    )

    assert len(record.model_votes) == 2
    assert record.abstention_reason is None
    discard_events = ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)
    assert len(discard_events) == 1
    assert discard_events[0].payload["failure"] == PROVIDER_FAILURE_COST_OVERRUN


def test_pipeline_cost_overrun_member_charges_its_accrued_cost_via_ceiling_trick(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """A discarded `ProviderCostOverrunError` member's `cost_micros` (750_000)
    is charged into the research budget *even though its vote never survives
    to aggregation* -- proven via the ceiling-trick pattern: a budget one
    micro below `_FULL_RUN_RESEARCH_COST_MICROS + 750_000` raises with
    `cost_micros` equal to that exact combined figure.
    """
    ledger = InMemoryForecastLedger()
    budget_ledger = InMemoryBudgetLedger()
    failed_member_cost_micros = 750_000
    exact_charge = _FULL_RUN_RESEARCH_COST_MICROS + failed_member_cost_micros
    tight_budget = ResearchBudget(
        per_forecast_micros=exact_charge - 1, ledger=budget_ledger
    )
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _success_provider(_MEMBER_A),
        _MEMBER_B.model_version: _FailingProvider(
            ProviderCostOverrunError(
                cost_micros=failed_member_cost_micros, ceiling_micros=1_000_000
            )
        ),
        _MEMBER_C.model_version: _success_provider(_MEMBER_C),
    }

    with pytest.raises(PerForecastBudgetExceededError) as excinfo:
        run_pipeline(
            market,
            baseline,
            transport=ForbiddenLiveTransport(),
            created_at=created_at,
            research_tools=research_tools,
            ledger=ledger,
            budget=tight_budget,
            provider_factory=_routed_provider_factory(routes),
        )

    assert excinfo.value.cost_micros == exact_charge


def test_pipeline_discard_ledger_hygiene_exact_keys_and_no_raw_body_leak(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """Every discard payload carries exactly the six documented keys, every
    leaf is JSON-safe (int/str/bool, never float), and the raw (never
    directly ledgered) malformed response body never appears as a substring
    anywhere in the serialized ledger -- only its fingerprint may leak.
    """
    raw_body = "SECRET-RAW-BODY-MARKER-not-a-real-secret-just-a-test-token"
    fingerprint = hashlib.sha256(raw_body.encode("utf-8")).hexdigest()
    ledger = InMemoryForecastLedger()
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _success_provider(_MEMBER_A),
        _MEMBER_B.model_version: _FailingProvider(
            ProviderMalformedResponseError(fingerprint)
        ),
        _MEMBER_C.model_version: _success_provider(_MEMBER_C),
    }

    run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_factory=_routed_provider_factory(routes),
    )

    discard_events = ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)
    assert len(discard_events) == 1
    payload = discard_events[0].payload
    assert set(payload) == {
        "market_ticker",
        "provider",
        "model_version",
        "vote_index",
        "failure",
        "response_fingerprint",
    }
    _assert_json_safe_leaves(payload)
    assert payload["response_fingerprint"] == fingerprint
    serialized = json.dumps(payload)
    assert raw_body not in serialized


# --- Section 4: tracer regression -- the pre-#193 happy path is unaffected -----


def test_tracer_default_happy_path_has_zero_discards_and_the_exact_research_cost(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
    make_fake_vote_transport: Callable[..., object],
) -> None:
    """The plain, no-`provider_factory` fixture-vote happy path -- entirely
    untouched by this issue's fault-injection taxonomy -- still yields zero
    discard events, an exact `research_cost_micros == 3_000_000` (pinned via
    the ceiling-trick pattern too), and two independent runs remain
    byte-equal. This guards the pre-#193 invariant end-to-end and should be
    among the first assertions in this file to turn green once the new
    modules exist, since it exercises no new parameter.
    """
    ledger = InMemoryForecastLedger()

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
    )

    assert record.research_cost_micros == _FULL_RUN_RESEARCH_COST_MICROS
    assert ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT) == ()

    budget_ledger = InMemoryBudgetLedger()
    tight_budget = ResearchBudget(
        per_forecast_micros=_FULL_RUN_RESEARCH_COST_MICROS - 1, ledger=budget_ledger
    )
    with pytest.raises(PerForecastBudgetExceededError) as excinfo:
        run_pipeline(
            market,
            baseline,
            transport=make_fake_vote_transport(),
            created_at=created_at,
            research_tools=research_tools,
            budget=tight_budget,
        )
    assert excinfo.value.cost_micros == _FULL_RUN_RESEARCH_COST_MICROS

    record_again = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
    )
    assert forecast_record_to_payload(record) == forecast_record_to_payload(
        record_again
    )
