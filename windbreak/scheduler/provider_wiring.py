"""Live-vs-cassette forecast provider composition (issues #344, #269).

:func:`windbreak.scheduler.loop.build_paper_deps` chooses, once per process,
between the recorded offline replay cassette and the live pinned-LLM / research
transports epic #183 shipped. This module holds that choice's *construction*
so the composition root keeps stating policy rather than plumbing.

Three things live here:

* :class:`LiveProviderHttp` -- the per-seam bundle of live HTTP transports. It
  is a bundle rather than a single transport on purpose: each provider's
  credential is held inside its *own*
  :class:`~windbreak.net.live_http.LiveHttpTransport`, alongside its own
  single-host allowlist, so one vendor's key structurally cannot travel to
  another vendor's endpoint. Sharing one transport across providers would send
  the Anthropic key to OpenAI on the very first vote.
* :func:`retry_policy_from_config` / :func:`price_table_from_config` -- the
  config-to-engine adapters issue #269 needs. The forecast engine may not
  import ``windbreak.config`` (SPEC S8.3), so the translation happens out here,
  on the scheduler side of that boundary.
* :func:`build_provider_factory` -- the per-ensemble-member provider builder
  the pipeline drives votes through.

**Why only the live path is wrapped in a retrying, priced provider.** Replaying
a recorded cassette spends no money and suffers no transient transport faults.
Wrapping it would charge each vote a *list price* for a call that never
happened -- and
:class:`~windbreak.forecast.providers.retry.RetryingProvider` stamps that
accrued price onto a failed vote's ``cost_micros`` -- so an offline tick would
start reporting fabricated spend into the research budget and the cost ledger.
The price table exists to keep cost accounting honest; billing a replay would
be the first thing to make it dishonest. The invariant that matters is the
other direction, and it holds structurally: the live branch is the only place a
live transport is ever constructed, and it always wraps.

This module is on the money path (``scripts/lint_no_floats.py`` guards
``windbreak/scheduler``), so it is float-free: the retry schedule is whole
milliseconds and prices are whole micros throughout. The real wall-clock seam
it hands to the retry layer lives in :mod:`windbreak.net.live_http`, outside
the ban, because the final hop into ``time.sleep`` is necessarily fractional.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from windbreak.config.schema import (
    PROVIDER_TRANSPORT_CASSETTE,
    PROVIDER_TRANSPORT_LIVE,
)
from windbreak.forecast.budget import ProviderPriceTable
from windbreak.forecast.providers import (
    ANTHROPIC_MESSAGES_ENDPOINT,
    OPENAI_CHAT_ENDPOINT,
    AnthropicMessagesTransport,
    FixtureVoteProvider,
    LiveFetchConfig,
    LiveFetchTransport,
    LiveSearchConfig,
    LiveSearchTransport,
    OpenAiChatTransport,
    RetryingProvider,
    RetryPolicy,
)
from windbreak.forecast.sandbox import build_research_tools
from windbreak.net.live_http import RoutingLlmTransport, monotonic_ms, sleep_ms

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig
    from windbreak.forecast.cassettes import LlmTransport
    from windbreak.forecast.providers import (
        EnsembleMemberLike,
        ForecastProvider,
        HttpTransport,
    )
    from windbreak.forecast.sandbox import ResearchTools

#: Builds one vote provider for a (completion transport, ensemble member) pair.
#: The transport is a parameter rather than a captured value so the bundle's
#: swap seam keeps working; see :func:`build_provider_factory`.
ProviderFactory = Callable[["LlmTransport", "EnsembleMemberLike"], "ForecastProvider"]

#: The provider identifier each live LLM adapter is registered under. These
#: match the ``provider`` field of a configured vote-ensemble member and the
#: keys of ``windbreak.net.allowlist._FORECAST_PROVIDER_HOSTS``.
ANTHROPIC_PROVIDER = "anthropic"
OPENAI_PROVIDER = "openai"

#: How many candidate URLs a live search requests per subquestion.
_SEARCH_MAX_RESULTS = 10

#: Builds the live LLM adapter for each supported provider over its own HTTP
#: transport. A provider absent from this table has no live adapter and is
#: therefore unroutable, which fails closed at
#: :meth:`~windbreak.net.live_http.RoutingLlmTransport.complete`.
_LLM_ADAPTER_BUILDERS: Mapping[str, Callable[[HttpTransport], LlmTransport]] = {
    ANTHROPIC_PROVIDER: lambda http: AnthropicMessagesTransport(
        http, endpoint_url=ANTHROPIC_MESSAGES_ENDPOINT
    ),
    OPENAI_PROVIDER: lambda http: OpenAiChatTransport(
        http, endpoint_url=OPENAI_CHAT_ENDPOINT
    ),
}


@dataclass(frozen=True, slots=True)
class LiveProviderHttp:
    """The live HTTP seams a live forecast stage dials through.

    Each entry is a fully-formed transport carrying its own credential headers
    and its own single-host egress allowlist, built by the CLI composition root
    (which owns the process environment). Nothing here reads a secret: this
    module receives transports, never keys.

    Attributes:
        llm: The per-provider HTTP transport each pinned-LLM adapter dials
            through, keyed by provider identifier. A provider missing from this
            mapping has no live route and fails closed at vote time rather than
            borrowing another provider's transport -- and therefore another
            provider's credential.
        search: The HTTP transport live web search is issued over.
        fetch: The HTTP transport live page fetches are issued over.
    """

    llm: Mapping[str, HttpTransport]
    search: HttpTransport
    fetch: HttpTransport


def is_live_mode(config: WindbreakConfig) -> bool:
    """Return whether configuration selects the live provider transport.

    Args:
        config: The active configuration.

    Returns:
        ``True`` for :data:`PROVIDER_TRANSPORT_LIVE`, ``False`` for
        :data:`PROVIDER_TRANSPORT_CASSETTE`.

    Raises:
        ValueError: If the configured mode is neither. An unrecognized mode is
            an operator error, and silently choosing a default would discard a
            stated intent -- the same fail-closed reading the config loader
            already gives an unknown YAML key.
    """
    mode = config.forecast.provider_transport.mode
    if mode == PROVIDER_TRANSPORT_LIVE:
        return True
    if mode == PROVIDER_TRANSPORT_CASSETTE:
        return False
    msg = (
        f"unknown forecast.provider_transport.mode {mode!r}; expected "
        f"{PROVIDER_TRANSPORT_CASSETTE!r} or {PROVIDER_TRANSPORT_LIVE!r}"
    )
    raise ValueError(msg)


def build_live_llm_transport(provider_http: LiveProviderHttp) -> LlmTransport:
    """Build the provider-routing live completion transport.

    Args:
        provider_http: The live HTTP seams, one per provider.

    Returns:
        A :class:`~windbreak.net.live_http.RoutingLlmTransport` dispatching each
        vote to the adapter for its own provider.
    """
    adapters = {
        provider: _LLM_ADAPTER_BUILDERS[provider](http)
        for provider, http in provider_http.llm.items()
        if provider in _LLM_ADAPTER_BUILDERS
    }
    return RoutingLlmTransport(adapters)


def build_live_research_tools(
    config: WindbreakConfig, provider_http: LiveProviderHttp, cache_dir: Path
) -> ResearchTools:
    """Build sandboxed research tools over the live search/fetch transports.

    The sandbox's own host allowlist comes from
    ``config.forecast.research.allowed_research_hosts``, which defaults to
    empty: an unconfigured deployment therefore reaches *no* research host even
    in live mode, rather than inheriting a plausible-looking default.

    Args:
        config: The active configuration supplying endpoint, hosts, and budgets.
        provider_http: The live HTTP seams search and fetch are issued over.
        cache_dir: The root the fetch cache is jailed to.

    Returns:
        A capability-closed :class:`~windbreak.forecast.sandbox.ResearchTools`.
    """
    research = config.forecast.research
    return build_research_tools(
        allowed_hosts=research.allowed_research_hosts,
        cache_dir=cache_dir,
        search_transport=LiveSearchTransport(
            provider_http.search,
            LiveSearchConfig(
                endpoint_url=research.search_endpoint_url,
                max_results=_SEARCH_MAX_RESULTS,
            ),
        ),
        fetch_transport=LiveFetchTransport(
            provider_http.fetch,
            LiveFetchConfig(
                max_body_bytes=research.fetch_max_bytes,
                allowed_content_types=research.allowed_content_types,
            ),
        ),
    )


def retry_policy_from_config(config: WindbreakConfig) -> RetryPolicy:
    """Build the bounded-retry policy from configuration (issue #269).

    Args:
        config: The active configuration supplying the four bounds.

    Returns:
        The configured :class:`~windbreak.forecast.providers.retry.RetryPolicy`.

    Raises:
        ValueError: If any configured bound is not strictly positive --
            aborting startup rather than running an unbounded retry loop
            against a paid provider.
    """
    retry = config.forecast.provider_transport.retry
    return RetryPolicy(
        max_attempts=retry.max_attempts,
        total_deadline_ms=retry.total_deadline_ms,
        backoff_base_ms=retry.backoff_base_ms,
        max_cost_micros=retry.max_cost_micros,
    )


def price_table_from_config(config: WindbreakConfig) -> ProviderPriceTable:
    """Build the fail-closed per-provider price table from configuration (#269).

    Args:
        config: The active configuration supplying the prices.

    Returns:
        The configured
        :class:`~windbreak.forecast.budget.ProviderPriceTable`.

    Raises:
        ValueError: If any configured price is below one micro -- a zero-priced
            provider would evade its budget entirely.
    """
    transport = config.forecast.provider_transport
    return ProviderPriceTable(
        prices_micros={
            price.provider: price.price_micros for price in transport.prices
        },
        unknown_provider_price_micros=transport.unknown_provider_price_micros,
    )


def build_provider_factory(config: WindbreakConfig, *, live: bool) -> ProviderFactory:
    """Build the per-member vote-provider factory the pipeline drives.

    In cassette mode this returns the bare
    :class:`~windbreak.forecast.providers.fixture.FixtureVoteProvider` the
    pipeline would have built for itself, so the offline path stays
    byte-identical. In live mode every member is additionally wrapped in a
    :class:`~windbreak.forecast.providers.retry.RetryingProvider` carrying the
    configured policy, the configured price table, and the real
    integer-millisecond clock/sleep pair -- see this module's docstring for why
    the wrap is deliberately live-only.

    The completion transport is taken *per call* rather than captured here, so
    the factory stays a policy ("how a vote is wrapped") rather than a closure
    over one specific transport. That is what keeps
    ``dataclasses.replace(deps, transport=...)`` -- the swap seam the loop's
    tests drive a doubled vote transport through -- actually reaching the vote
    stage instead of silently voting against the transport that happened to be
    wired at composition time.

    Args:
        config: The active configuration supplying policy and prices.
        live: Whether the live provider transport is selected (keyword-only).

    Returns:
        A factory building one provider per (transport, ensemble member) pair.
    """
    if not live:
        return lambda transport, member: FixtureVoteProvider(transport, member)
    policy = retry_policy_from_config(config)
    price_table = price_table_from_config(config)

    def _live_provider(
        transport: LlmTransport, member: EnsembleMemberLike
    ) -> ForecastProvider:
        """Build one retried, priced live provider for ``member``.

        Args:
            transport: The completion transport the vote is obtained through.
            member: The vote-ensemble member to build a provider for.

        Returns:
            The wrapped provider.
        """
        return RetryingProvider(
            FixtureVoteProvider(transport, member),
            provider_name=member.provider,
            policy=policy,
            price_table=price_table,
            monotonic_ms=monotonic_ms,
            sleep_ms=sleep_ms,
        )

    return _live_provider
