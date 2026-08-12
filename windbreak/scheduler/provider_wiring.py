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
happened -- and since issue #399
:class:`~windbreak.forecast.providers.retry.RetryingProvider` prices *every*
attempt it makes, the successful one included -- so an offline tick would bill
a full ensemble's worth of fabricated spend into the research budget and the
cost ledger on every replayed run. The price table exists to keep cost
accounting honest; billing a replay would be the first thing to make it
dishonest.

The invariant that matters is the other direction, and it holds structurally:
the live branch is the only place a live transport is ever constructed, and it
always wraps. That is load-bearing for cost correctness, not just for retries.
The wrapped :class:`~windbreak.forecast.providers.fixture.FixtureVoteProvider`
reports ``cost_micros == 0`` -- truthfully for a replay, which spends nothing,
and by design for a live call, since that provider *measures* (it threads the
transport's reported token usage onto the forecast) but never prices. The retry
wrapper is therefore the only layer that turns a live vote into money, and a
live provider built without it would book every successful vote at zero.

Since issue #451 that wrapper needs two tables, not one, and
:func:`build_provider_factory` supplies both:
:func:`price_table_from_config` for the per-attempt affordability estimate the
pre-gate runs on, and :func:`rate_table_from_config` for the per-model token
rates a completed vote's reported usage is actually charged at. Wiring only the
first would restore the flat per-attempt charge issue #451 removed.

This module is on the money path (``scripts/lint_no_floats.py`` guards
``windbreak/scheduler``), so it is float-free: the retry schedule is whole
milliseconds and prices are whole micros throughout. The real wall-clock seam
it hands to the retry layer lives in :mod:`windbreak.net.live_http`, outside
the ban, because the final hop into ``time.sleep`` is necessarily fractional.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from windbreak.config.schema import (
    DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    PROVIDER_TRANSPORT_CASSETTE,
    PROVIDER_TRANSPORT_LIVE,
    REPLAY_CORPUS_DISABLED,
    REPLAY_CORPUS_REPLAY,
    UNCONFIGURED_PLACEHOLDER,
    ReplayCorpusConfig,
)
from windbreak.forecast.budget import (
    ModelRateTable,
    ModelTokenRate,
    ProviderPriceTable,
)
from windbreak.forecast.corpus import (
    CorpusResearchTransport,
    CorpusVoteTransport,
    ReplayCorpus,
    load_replay_corpus,
)
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

#: ``source=`` token logged when a configuration file selected this run's
#: replay-corpus section.
REPLAY_CORPUS_SOURCE_CONFIGURED = "configuration"

#: ``source=`` token logged when nothing selected it and the shipped default --
#: no corpus, an offline loop that cannot trade -- stands.
REPLAY_CORPUS_SOURCE_DEFAULT = "default"

#: The research egress host allowlisted for the offline default research tools.
#: The offline default never actually searches, so nothing is ever fetched
#: against it.
_DEFAULT_RESEARCH_HOST = "research.local"

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
        search: The HTTP transport live web search is issued over, or ``None``
            when live research is not configured. ``None`` is not a degraded
            live mode: research falls back to the offline transport that finds
            nothing, so the pipeline abstains on zero verified citations before
            any vote (SPEC S8.8). Live *providers* and live *research* are
            independently configured, and a deployment that has pinned an LLM
            but not a search endpoint must not be forced to invent one.
        fetch: The HTTP transport live page fetches are issued over, or ``None``
            alongside ``search``.
    """

    llm: Mapping[str, HttpTransport]
    search: HttpTransport | None
    fetch: HttpTransport | None


class OfflineResearchTransport:
    """A search/fetch transport that finds nothing (the offline default)."""

    def search(self, query: str) -> tuple[str, ...]:
        """Return no candidate URLs, unconditionally.

        Args:
            query: The (unused) subquestion text.

        Returns:
            An empty tuple, always.
        """
        del query
        return ()

    def fetch(self, url: str) -> str:
        """Never reached (search finds nothing); raises defensively.

        Args:
            url: The (unused) URL that would have been fetched.

        Raises:
            RuntimeError: Always -- reaching this is itself a wiring bug.
        """
        raise RuntimeError(
            f"offline research transport fetch unexpectedly called: {url!r}"
        )


def offline_research_tools(cache_dir: Path) -> ResearchTools:
    """Build the offline, no-network research bundle.

    Its transports find nothing, so the pipeline abstains on zero verified
    citations before any fetch or vote (SPEC S8.8) -- the offline PAPER
    contract, without a live network. Shared by the cassette path and by a live
    deployment that has not configured a research endpoint.

    The cache bound is the shipped default rather than the operator's
    ``forecast.research.cache_max_bytes``, and that is inert by construction:
    these transports find nothing, so no fetch ever runs and no cache entry is
    ever written. Taking the configured value here would imply this path can
    grow the cache, which it cannot.

    Args:
        cache_dir: The root the (never-written) fetch cache is jailed to.

    Returns:
        A capability-closed :class:`~windbreak.forecast.sandbox.ResearchTools`.
    """
    transport = OfflineResearchTransport()
    return build_research_tools(
        allowed_hosts=frozenset({_DEFAULT_RESEARCH_HOST}),
        cache_dir=cache_dir,
        search_transport=transport,
        fetch_transport=transport,
        max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    )


def replay_corpus_directory(config: WindbreakConfig) -> Path | None:
    """Return the committed corpus directory this deployment replays, or none.

    The one place ``forecast.replay_corpus`` is read into a decision, so the
    research half and the vote half of a corpus run can never disagree about
    whether one is selected -- which is the whole reason the two are a single
    token (issue #510). Closing research alone converts a graceful
    ``no_verified_citations`` abstention into a
    :class:`~windbreak.forecast.cassettes.CassetteMissError` out of the tick.

    Args:
        config: The active configuration.

    Returns:
        The corpus directory when :data:`REPLAY_CORPUS_REPLAY` is selected, else
        ``None``.

    Raises:
        ValueError: If the mode is unrecognized, or the replay mode is selected
            without naming a directory. Both refuse to start rather than
            silently degrading to the offline default, which would leave an
            operator who asked for a demonstrable run watching a loop that
            abstains forever and says nothing about why.
    """
    settings = config.forecast.replay_corpus
    if settings.mode == REPLAY_CORPUS_DISABLED:
        return None
    if settings.mode != REPLAY_CORPUS_REPLAY:
        msg = (
            f"unknown forecast.replay_corpus.mode {settings.mode!r}; expected "
            f"{REPLAY_CORPUS_DISABLED!r} or {REPLAY_CORPUS_REPLAY!r}"
        )
        raise ValueError(msg)
    if settings.corpus_dir == UNCONFIGURED_PLACEHOLDER:
        msg = (
            f"forecast.replay_corpus.mode is {REPLAY_CORPUS_REPLAY!r} but "
            f"forecast.replay_corpus.corpus_dir is still "
            f"{UNCONFIGURED_PLACEHOLDER!r}; name the committed corpus directory "
            f"or select {REPLAY_CORPUS_DISABLED!r}"
        )
        raise ValueError(msg)
    return Path(settings.corpus_dir)


def replay_corpus_source(config: WindbreakConfig) -> str:
    """Return which source decided this run's replay-corpus section.

    A configuration-only leaf has exactly two sources, and an operator reading
    the startup line needs to know which one they got: a section that differs
    from the shipped default was written down somewhere, and one that matches it
    was not written down at all. Derived by comparison against
    :class:`~windbreak.config.schema.ReplayCorpusConfig`'s own defaults rather
    than by a flag the loader would have to remember to set.

    Args:
        config: The active configuration.

    Returns:
        :data:`REPLAY_CORPUS_SOURCE_CONFIGURED` or
        :data:`REPLAY_CORPUS_SOURCE_DEFAULT`.
    """
    if config.forecast.replay_corpus == ReplayCorpusConfig():
        return REPLAY_CORPUS_SOURCE_DEFAULT
    return REPLAY_CORPUS_SOURCE_CONFIGURED


def load_corpus(directory: Path) -> ReplayCorpus:
    """Load the committed corpus at ``directory``.

    A thin re-export so the composition root reaches the loader through this
    module, beside the other two transport choices, rather than importing the
    forecast engine's file layout directly.

    Args:
        directory: The corpus directory.

    Returns:
        The loaded corpus.

    Raises:
        CorpusFormatError: If the directory is not a well-formed corpus; see
            :func:`~windbreak.forecast.corpus.load_replay_corpus`.
    """
    return load_replay_corpus(directory)


def build_corpus_research_tools(corpus: ReplayCorpus, cache_dir: Path) -> ResearchTools:
    """Build the sandboxed research bundle over a committed corpus.

    The sandbox's egress allowlist is **derived from the corpus itself** --
    exactly the hosts it holds recorded documents for -- rather than configured
    beside it. A second, transcribed host list could grant a host the corpus
    cannot serve, and there is nothing for it to grant: these transports read
    committed files and never open a socket.

    The cache bound is the shipped default rather than the operator's
    ``forecast.research.cache_max_bytes``, for the same reason
    :func:`offline_research_tools` takes it: a replayed fetch writes a bounded,
    committed body, so this path cannot grow the cache the way a live one can,
    and taking the live ceiling here would imply otherwise.

    Args:
        corpus: The loaded corpus to serve.
        cache_dir: The root the fetch cache is jailed to.

    Returns:
        A capability-closed :class:`~windbreak.forecast.sandbox.ResearchTools`.
    """
    transport = CorpusResearchTransport(corpus)
    return build_research_tools(
        allowed_hosts=corpus.hosts(),
        cache_dir=cache_dir,
        search_transport=transport,
        fetch_transport=transport,
        max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    )


def build_corpus_vote_transport(corpus: ReplayCorpus) -> LlmTransport:
    """Build the vote transport serving a committed corpus's recorded votes.

    Args:
        corpus: The loaded corpus to serve.

    Returns:
        A :class:`~windbreak.forecast.corpus.CorpusVoteTransport`.
    """
    return CorpusVoteTransport(corpus)


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


def routable_live_providers() -> frozenset[str]:
    """Return every provider this composition root can route a live vote to.

    The authoritative answer to "what does ``mode: live`` actually support",
    shared by the startup guard, its error message, and the configuration
    defaults, so those three can never drift into advertising a provider the
    root would refuse.

    Returns:
        The live-routable provider identifiers.
    """
    return frozenset(_LLM_ADAPTER_BUILDERS)


def live_vote_providers(config: WindbreakConfig) -> tuple[str, ...]:
    """Return the distinct providers the configured vote ensemble draws on.

    Order-preserving and de-duplicated, because the default ensemble names one
    provider twice (two OpenAI models) and only one transport per provider is
    ever built.

    Args:
        config: The active configuration.

    Returns:
        Each distinct provider identifier, in first-appearance order.
    """
    return tuple(
        dict.fromkeys(member.provider for member in config.forecast.vote_ensemble)
    )


def _require_routable(config: WindbreakConfig, provider_http: LiveProviderHttp) -> None:
    """Refuse a live ensemble naming a provider this deployment cannot route.

    ``EnsembleMemberConfig.provider`` is a free string, so nothing else stops an
    operator adding a provider with no live adapter -- ``futuresearch``, whose
    provider is a :class:`ForecastProvider` rather than a completion transport
    and so does not ride this seam at all, or simply a typo. Left unchecked that
    surfaces as a per-vote
    :class:`~windbreak.forecast.providers.base.ProviderNotRoutableError`
    mid-tick; refusing here turns it into a clean startup failure naming the
    provider, which is what an operator can actually act on.

    Args:
        config: The active configuration supplying the vote ensemble.
        provider_http: The live HTTP seams supplied for it.

    Raises:
        ValueError: If any ensemble provider has no live adapter, or has one but
            no HTTP seam was built for it.
    """
    routable = sorted(routable_live_providers())
    for provider in live_vote_providers(config):
        if provider not in _LLM_ADAPTER_BUILDERS:
            msg = (
                f"forecast.vote_ensemble names provider {provider!r}, which has "
                f"no live transport; live-routable providers are {routable}. "
                f"Remove it from the ensemble or select the 'cassette' transport"
            )
            raise ValueError(msg)
        if provider not in provider_http.llm:
            msg = (
                f"forecast.vote_ensemble names provider {provider!r} but no live "
                f"HTTP seam was built for it; supply one or remove the member"
            )
            raise ValueError(msg)


def build_live_llm_transport(
    config: WindbreakConfig,
    provider_http: LiveProviderHttp,
    *,
    validate: bool = True,
) -> LlmTransport:
    """Build the provider-routing live completion transport.

    Args:
        config: The active configuration supplying the vote ensemble.
        provider_http: The live HTTP seams, one per provider.
        validate: Whether to refuse an ensemble naming an unroutable provider
            (keyword-only, default ``True``). Only a test proving the
            defence-in-depth layer behind that refusal passes ``False``.

    Returns:
        A :class:`~windbreak.net.live_http.RoutingLlmTransport` dispatching each
        vote to the adapter for its own provider.

    Raises:
        ValueError: If ``validate`` and the ensemble names a provider with no
            live route; see :func:`_require_routable`.
    """
    if validate:
        _require_routable(config, provider_http)
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

    This is the *only* path that can grow the fetch cache, so it is the path
    that carries the operator's ``forecast.research.cache_max_bytes`` into it
    (issue #453). Removing that one keyword is what
    ``test_the_configured_cap_reaches_the_live_cache`` exists to catch.

    Args:
        config: The active configuration supplying endpoint, hosts, and budgets.
        provider_http: The live HTTP seams search and fetch are issued over.
        cache_dir: The root the fetch cache is jailed to.

    Returns:
        A capability-closed :class:`~windbreak.forecast.sandbox.ResearchTools`;
        the offline no-network bundle when live research is unconfigured.

    Raises:
        ValueError: If ``forecast.research.cache_max_bytes`` is not a positive
            byte count; see
            :class:`~windbreak.forecast.sandbox.ResearchCache`.
    """
    research = config.forecast.research
    if provider_http.search is None or provider_http.fetch is None:
        return offline_research_tools(cache_dir)
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
        max_bytes=research.cache_max_bytes,
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


def rate_table_from_config(config: WindbreakConfig) -> ModelRateTable:
    """Build the fail-closed per-model metered-cost table from config (#451).

    Args:
        config: The active configuration supplying the token rates.

    Returns:
        The configured :class:`~windbreak.forecast.budget.ModelRateTable`.

    Raises:
        ValueError: If any configured token rate, or the unmetered fallback, is
            below one micro -- a zero-rated model would bill nothing however
            many tokens it consumed, which is the unbounded-spend hole metering
            exists to close.
    """
    transport = config.forecast.provider_transport
    return ModelRateTable(
        rates={
            price.model_version: ModelTokenRate(
                model_version=price.model_version,
                input_micros_per_million_tokens=price.input_micros_per_million_tokens,
                output_micros_per_million_tokens=price.output_micros_per_million_tokens,
            )
            for price in transport.token_prices
        },
        unmetered_micros=transport.unmetered_response_micros,
    )


def build_provider_factory(config: WindbreakConfig, *, live: bool) -> ProviderFactory:
    """Build the per-member vote-provider factory the pipeline drives.

    In cassette mode this returns the bare
    :class:`~windbreak.forecast.providers.fixture.FixtureVoteProvider` the
    pipeline would have built for itself, so the offline path stays
    byte-identical. In live mode every member is additionally wrapped in a
    :class:`~windbreak.forecast.providers.retry.RetryingProvider` carrying the
    configured policy, the configured price table, the configured per-model
    token rate table (issue #451), and the real integer-millisecond clock/sleep
    pair -- see this module's docstring for why the wrap is deliberately
    live-only.

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
    rate_table = rate_table_from_config(config)

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
            rate_table=rate_table,
            monotonic_ms=monotonic_ms,
            sleep_ms=sleep_ms,
        )

    return _live_provider
