"""A retry/backoff decorator around a single :class:`ForecastProvider` seam.

SPEC S8/S16 needs a hosted provider call to survive transient transport faults
(a timeout, a throttle, a 5xx) without either giving up on the first blip or
retrying forever into an unbounded spend. :class:`RetryingProvider` wraps any
inner :class:`~windbreak.forecast.providers.base.ForecastProvider` and adds
four bounded guards:

* **Bounded attempts** -- at most ``policy.max_attempts`` calls.
* **A total deadline** -- a retry is skipped when its wait would push past the
  run's budgeted wall-clock deadline.
* **Backoff / ``Retry-After``** -- exponential backoff between attempts, unless
  a :class:`~windbreak.forecast.providers.base.ProviderRateLimitedError` carries
  an explicit ``retry_after_seconds`` hint, which is honored instead.
* **An affordability pre-gate** -- before *every* attempt, the accrued spend
  plus the next attempt's list price is checked against
  ``policy.max_cost_micros``; an unaffordable attempt raises
  :class:`~windbreak.forecast.providers.base.ProviderCostOverrunError` *without*
  ever calling the inner provider.

Only a
:class:`~windbreak.forecast.providers.base.ProviderVoteError` is caught: a
non-taxonomy exception (a real bug) propagates untouched rather than being
silently swallowed by the retry loop. The module is stdlib-only and entirely
float-free (it sits on the money path guarded by ``scripts/lint_no_floats.py``):
all time is integer milliseconds through injected ``monotonic_ms``/``sleep_ms``
callables -- there is no ``time`` import and no real clock or sleep default, so
the whole retry schedule is deterministic and test-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from windbreak.forecast.providers.base import (
    ProviderCostOverrunError,
    ProviderHTTPError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderVoteError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.budget import ProviderPriceTable
    from windbreak.forecast.providers.base import ForecastProvider, ProviderForecast
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sanitize import ResearchQuote

#: The default maximum number of provider attempts (one initial call + retries).
DEFAULT_MAX_ATTEMPTS: Final = 3

#: The default total wall-clock deadline for all attempts, in milliseconds.
DEFAULT_TOTAL_DEADLINE_MS: Final = 30_000

#: The default exponential-backoff base wait, in milliseconds; attempt ``n``
#: waits ``backoff_base_ms << (n - 1)`` unless a ``Retry-After`` hint overrides.
DEFAULT_BACKOFF_BASE_MS: Final = 1_000

#: The HTTP "Too Many Requests" status -- the sole retryable status below 500.
HTTP_TOO_MANY_REQUESTS: Final = 429

#: Inclusive lower bound of the retryable HTTP 5xx server-error range.
_HTTP_SERVER_ERROR_FLOOR: Final = 500

#: Inclusive upper bound of the retryable HTTP 5xx server-error range.
_HTTP_SERVER_ERROR_CEILING: Final = 599

#: Milliseconds per second, converting a ``Retry-After`` seconds hint to ms.
_MS_PER_SECOND: Final = 1_000


def is_retryable_status(status_code: int) -> bool:
    """Return whether an HTTP status code warrants a retry.

    A status is retryable iff it is ``429`` (Too Many Requests) or lies in the
    inclusive ``[500, 599]`` server-error range -- transient, server-side
    conditions a later attempt may clear. Every 4xx below 429 and above it
    (a genuine client error) and any non-HTTP code is non-retryable.

    Args:
        status_code: The HTTP status code to classify.

    Returns:
        ``True`` if the status is retryable, else ``False``.
    """
    if status_code == HTTP_TOO_MANY_REQUESTS:
        return True
    return _HTTP_SERVER_ERROR_FLOOR <= status_code <= _HTTP_SERVER_ERROR_CEILING


def _is_retryable(error: ProviderVoteError) -> bool:
    """Return whether a caught provider failure is worth retrying.

    Only transport-class faults retry: a timeout, a rate-limit, or an HTTP
    error whose status :func:`is_retryable_status` accepts. A screen-side
    rejection (malformed/version-drift/response-rejected) or a cost overrun is
    never retried -- retrying would only re-poison or compound it.

    Args:
        error: The caught provider-vote failure.

    Returns:
        ``True`` if ``error`` is a retryable transport fault, else ``False``.
    """
    return isinstance(error, ProviderTimeoutError | ProviderRateLimitedError) or (
        isinstance(error, ProviderHTTPError) and is_retryable_status(error.status_code)
    )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """The bounded-retry policy a :class:`RetryingProvider` enforces.

    Every field is validated strictly positive at construction, matching the
    fail-closed convention the budget dataclasses use: a zero or negative
    attempt count, deadline, backoff, or cost ceiling is a usage error.

    Attributes:
        max_attempts: The maximum number of provider attempts.
        total_deadline_ms: The total wall-clock deadline for all attempts, in ms.
        backoff_base_ms: The exponential-backoff base wait, in ms.
        max_cost_micros: The affordability ceiling across all attempts, in micros.
    """

    max_attempts: int
    total_deadline_ms: int
    backoff_base_ms: int
    max_cost_micros: int

    def __post_init__(self) -> None:
        """Validate every field is strictly positive.

        Raises:
            ValueError: If any field is zero or negative.
        """
        fields = {
            "max_attempts": self.max_attempts,
            "total_deadline_ms": self.total_deadline_ms,
            "backoff_base_ms": self.backoff_base_ms,
            "max_cost_micros": self.max_cost_micros,
        }
        for name, value in fields.items():
            if value <= 0:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)


class RetryingProvider:
    """A :class:`ForecastProvider` decorator adding bounded retries and pricing.

    Wraps an inner provider, retrying transient transport faults with
    exponential backoff (or an honored ``Retry-After`` hint) up to a bounded
    attempt count and total deadline, while charging each attempt's list price
    against an affordability ceiling. On the happy path (zero failed attempts)
    the returned forecast is byte-equal to the inner provider's own result --
    the wrapper is invisible.
    """

    def __init__(
        self,
        inner: ForecastProvider,
        *,
        provider_name: str,
        policy: RetryPolicy,
        price_table: ProviderPriceTable,
        monotonic_ms: Callable[[], int],
        sleep_ms: Callable[[int], None],
    ) -> None:
        """Wire the wrapped provider, policy, pricing, and injected clock.

        Args:
            inner: The wrapped provider whose ``forecast`` is retried.
            provider_name: The name priced against ``price_table`` per attempt
                (keyword-only).
            policy: The bounded-retry policy to enforce (keyword-only).
            price_table: The per-attempt price table (keyword-only).
            monotonic_ms: An injected monotonic clock returning milliseconds
                (keyword-only); no real clock default -- the caller owns time.
            sleep_ms: An injected sleep taking a millisecond wait (keyword-only);
                no real sleep default -- the caller owns waiting.
        """
        self._inner = inner
        self._provider_name = provider_name
        self._policy = policy
        self._price_table = price_table
        self._monotonic_ms = monotonic_ms
        self._sleep_ms = sleep_ms

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[ResearchQuote, ...],
    ) -> ProviderForecast:
        """Obtain one forecast, retrying transient faults within budget.

        Before each attempt the affordability pre-gate refuses an attempt whose
        price would breach ``policy.max_cost_micros`` (raising
        :class:`ProviderCostOverrunError` without calling the inner provider).
        A caught retryable fault backs off (or honors a ``Retry-After`` hint)
        and retries while attempts, the deadline, and the budget all permit;
        otherwise its accrued ``cost_micros`` is stamped and it is re-raised. A
        non-:class:`ProviderVoteError` propagates untouched.

        Args:
            market: The market under forecast.
            baseline: The baseline quote snapshot.
            vote_index: The zero-based index of this vote in the ensemble.
            quotes: The sanitized web quotes threaded into the vote prompt.

        Returns:
            The inner forecast, its ``cost_micros`` bumped by the accrued price
            of any failed attempts.

        Raises:
            ProviderCostOverrunError: If an attempt (or the successful total)
                would breach the affordability ceiling.
            ProviderVoteError: If a non-retryable fault occurs, or a retryable
                fault exhausts the attempt or deadline budget.
        """
        deadline = self._monotonic_ms() + self._policy.total_deadline_ms
        price = self._price_table.price_micros(self._provider_name)
        accrued = 0
        attempt = 0
        while True:
            attempt += 1
            self._ensure_affordable(accrued, price)
            try:
                forecast = self._inner.forecast(market, baseline, vote_index, quotes)
            except ProviderVoteError as error:
                accrued += price
                if not self._retry_after_failure(error, attempt, deadline):
                    error.cost_micros = accrued
                    raise
                continue
            return self._finalize_success(forecast, accrued)

    def _ensure_affordable(self, accrued: int, price: int) -> None:
        """Refuse the next attempt when its price would breach the ceiling.

        Args:
            accrued: The spend already accrued across prior attempts, in micros.
            price: The next attempt's list price, in micros.

        Raises:
            ProviderCostOverrunError: If ``accrued + price`` exceeds the
                affordability ceiling; the inner provider is never called.
        """
        if accrued + price > self._policy.max_cost_micros:
            raise ProviderCostOverrunError(
                cost_micros=accrued, ceiling_micros=self._policy.max_cost_micros
            )

    def _retry_after_failure(
        self, error: ProviderVoteError, attempt: int, deadline: int
    ) -> bool:
        """Decide whether to retry after a caught failure, sleeping if so.

        A non-retryable fault, an exhausted attempt budget, or a wait that would
        push past ``deadline`` all stop the loop (return ``False``); otherwise
        the injected sleep is invoked for the computed wait and the loop
        continues (return ``True``).

        Args:
            error: The caught provider-vote failure.
            attempt: The 1-based number of the attempt that just failed.
            deadline: The absolute monotonic-ms deadline for all attempts.

        Returns:
            ``True`` if a retry was scheduled (and slept for), else ``False``.
        """
        if not _is_retryable(error):
            return False
        if attempt >= self._policy.max_attempts:
            return False
        wait = self._compute_wait(error, attempt)
        if self._monotonic_ms() + wait > deadline:
            return False
        self._sleep_ms(wait)
        return True

    def _compute_wait(self, error: ProviderVoteError, attempt: int) -> int:
        """Compute the wait before the next attempt, in milliseconds.

        A :class:`ProviderRateLimitedError` carrying an explicit
        ``retry_after_seconds`` hint waits exactly that many seconds (converted
        to ms); every other retryable fault waits the exponential-backoff
        schedule ``backoff_base_ms << (attempt - 1)``.

        Args:
            error: The caught provider-vote failure.
            attempt: The 1-based number of the attempt that just failed.

        Returns:
            The wait before the next attempt, in milliseconds.
        """
        if (
            isinstance(error, ProviderRateLimitedError)
            and error.retry_after_seconds is not None
        ):
            return error.retry_after_seconds * _MS_PER_SECOND
        return self._policy.backoff_base_ms << (attempt - 1)

    def _finalize_success(
        self, forecast: ProviderForecast, accrued: int
    ) -> ProviderForecast:
        """Fold accrued retry cost into a successful forecast, checking budget.

        Args:
            forecast: The inner provider's successful forecast.
            accrued: The spend accrued across failed attempts, in micros.

        Returns:
            ``forecast`` with ``cost_micros`` increased by ``accrued`` (byte-equal
            to ``forecast`` when ``accrued`` is zero).

        Raises:
            ProviderCostOverrunError: If the combined total breaches the ceiling.
        """
        total = forecast.cost_micros + accrued
        if total > self._policy.max_cost_micros:
            raise ProviderCostOverrunError(
                cost_micros=total, ceiling_micros=self._policy.max_cost_micros
            )
        return replace(forecast, cost_micros=total)
