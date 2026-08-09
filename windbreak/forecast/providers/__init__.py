"""The forecast provider seam: transport-agnostic vote collection (issue #184).

Re-exports the provider protocol and result types, the pinned default vote
ensemble, the network-free :class:`FixtureVoteProvider`, the hosted
research-forecaster :class:`FutureSearchProvider` (plus its HTTP record/replay
harness), and the pinned-LLM :class:`AnthropicMessagesTransport` /
:class:`OpenAiChatTransport` completion transports (issue #191) so callers
import from ``windbreak.forecast.providers`` rather than reaching into the
submodules.

Since issue #269 the provider *hardening* API is flat here too: the
:class:`~windbreak.forecast.providers.base.ProviderVoteError` failure taxonomy
(timeout / rate-limit / HTTP / malformed / version-drift / cost-overrun), the
deterministic :class:`RetryingProvider`/:class:`RetryPolicy` pair with its
:func:`is_retryable_status` classifier, and the fail-closed
:class:`~windbreak.forecast.budget.ProviderPriceTable`. Each is re-exported by
identity, so ``except providers.ProviderTimeoutError`` catches exactly what
:mod:`windbreak.forecast.providers.retry` raises.

Per the SPEC S8.3 sandbox boundary, nothing here imports ``windbreak.config``;
an ensemble member is accepted structurally via :class:`EnsembleMemberLike`.
"""

from __future__ import annotations

from windbreak.forecast.budget import (
    DEFAULT_PROVIDER_PRICE_TABLE,
    ProviderPriceTable,
)
from windbreak.forecast.providers.anthropic import (
    ANTHROPIC_MESSAGES_ENDPOINT,
    AnthropicMessagesTransport,
)
from windbreak.forecast.providers.base import (
    DEFAULT_VOTE_ENSEMBLE,
    EnsembleMember,
    EnsembleMemberLike,
    ForecastProvider,
    ProviderCitation,
    ProviderCostOverrunError,
    ProviderError,
    ProviderForecast,
    ProviderHTTPError,
    ProviderMalformedResponseError,
    ProviderRateLimitedError,
    ProviderResponseRejectedError,
    ProviderTimeoutError,
    ProviderVersionDriftError,
    ProviderVoteError,
    build_vote_prompt,
    fingerprint_response,
)
from windbreak.forecast.providers.fetch_live import (
    BodyTooLargeError,
    ContentTypeRejectedError,
    LiveFetchConfig,
    LiveFetchTransport,
    UnreachableUrlError,
)
from windbreak.forecast.providers.fixture import FixtureVoteProvider
from windbreak.forecast.providers.futuresearch import (
    FutureSearchProvider,
    FutureSearchProviderConfig,
)
from windbreak.forecast.providers.http_cassettes import (
    ForbiddenLiveHttpTransport,
    HttpRequest,
    HttpResponse,
    HttpTransport,
    RecordingHttpCassette,
    ReplayHttpCassette,
)
from windbreak.forecast.providers.openai import (
    OPENAI_CHAT_ENDPOINT,
    OpenAiChatTransport,
)
from windbreak.forecast.providers.retry import (
    RetryingProvider,
    RetryPolicy,
    is_retryable_status,
)
from windbreak.forecast.providers.search_live import (
    LiveSearchConfig,
    LiveSearchTransport,
)

__all__ = [
    "ANTHROPIC_MESSAGES_ENDPOINT",
    "DEFAULT_PROVIDER_PRICE_TABLE",
    "DEFAULT_VOTE_ENSEMBLE",
    "OPENAI_CHAT_ENDPOINT",
    "AnthropicMessagesTransport",
    "BodyTooLargeError",
    "ContentTypeRejectedError",
    "EnsembleMember",
    "EnsembleMemberLike",
    "FixtureVoteProvider",
    "ForbiddenLiveHttpTransport",
    "ForecastProvider",
    "FutureSearchProvider",
    "FutureSearchProviderConfig",
    "HttpRequest",
    "HttpResponse",
    "HttpTransport",
    "LiveFetchConfig",
    "LiveFetchTransport",
    "LiveSearchConfig",
    "LiveSearchTransport",
    "OpenAiChatTransport",
    "ProviderCitation",
    "ProviderCostOverrunError",
    "ProviderError",
    "ProviderForecast",
    "ProviderHTTPError",
    "ProviderMalformedResponseError",
    "ProviderPriceTable",
    "ProviderRateLimitedError",
    "ProviderResponseRejectedError",
    "ProviderTimeoutError",
    "ProviderVersionDriftError",
    "ProviderVoteError",
    "RecordingHttpCassette",
    "ReplayHttpCassette",
    "RetryPolicy",
    "RetryingProvider",
    "UnreachableUrlError",
    "build_vote_prompt",
    "fingerprint_response",
    "is_retryable_status",
]
