"""Config-to-engine provider wiring (issues #344, #269).

:mod:`windbreak.scheduler.provider_wiring` sits on the SPEC S8.3 boundary: the
forecast engine may not import ``windbreak.config``, so every translation from
an operator's configuration into an engine object happens here. These tests
drive that translation directly, where
``tests/integration/test_paper_live_providers.py`` drives it through the whole
composition root.

Nothing here reaches a network.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from windbreak.config.schema import (
    PROVIDER_TRANSPORT_CASSETTE,
    PROVIDER_TRANSPORT_LIVE,
    EnsembleMemberConfig,
    ForecastConfig,
    ModelTokenPrice,
    ProviderPrice,
    ProviderRetryConfig,
    ProviderTransportConfig,
    ResearchSettings,
    WindbreakConfig,
)
from windbreak.connector.models import NormalizedMarket
from windbreak.forecast.budget import TokenUsage
from windbreak.forecast.cassettes import Completion
from windbreak.forecast.providers import FixtureVoteProvider, RetryingProvider
from windbreak.forecast.records import BaselineQuoteSnapshot
from windbreak.scheduler.provider_wiring import (
    LiveProviderHttp,
    OfflineResearchTransport,
    build_live_llm_transport,
    build_live_research_tools,
    build_provider_factory,
    is_live_mode,
    offline_research_tools,
    price_table_from_config,
    rate_table_from_config,
    retry_policy_from_config,
)

if TYPE_CHECKING:
    from pathlib import Path


class _StubHttpTransport:
    """An `HttpTransport` double that is never expected to be dialed."""

    def send(self, request: object) -> object:
        """Fail loudly if dialed.

        Args:
            request: The (rejected) request.

        Raises:
            AssertionError: Always.
        """
        msg = f"unexpected dial: {request!r}"
        raise AssertionError(msg)


class _StubMember:
    """A structural `EnsembleMemberLike` naming one provider.

    Attributes:
        provider: The provider identifier.
        model_version: The pinned model version.
        training_cutoff: The declared training cutoff.
    """

    def __init__(self, provider: str) -> None:
        """Store the provider and fixed provenance.

        Args:
            provider: The provider identifier.
        """
        self.provider = provider
        self.model_version = "pinned-for-test"
        self.training_cutoff = "2025-01-01"


#: A schema-valid vote body the metered wiring tests drive a real vote through.
_VOTE_JSON = (
    '{"probability_ppm": 500000, "rationale_summary": "steady", "abstain": false}'
)


class _StubLlmTransport:
    """An `LlmTransport` double returning a fixed completion."""

    def __init__(
        self, *, response: str = "{}", usage: TokenUsage | None = None
    ) -> None:
        """Store the canned completion text and its reported token usage.

        Args:
            response: The completion text every call returns (keyword-only).
            usage: The token accounting every completion reports, or `None` for
                a response that reported none (keyword-only).
        """
        self._response = response
        self._usage = usage

    def complete(self, request: object) -> Completion:
        """Return a fixed completion.

        Args:
            request: The (unused) completion request.

        Returns:
            The stored `Completion`.
        """
        del request
        return Completion(text=self._response, usage=self._usage)


def _market() -> NormalizedMarket:
    """Build a minimal, valid market for a wiring-level vote.

    Returns:
        A `NormalizedMarket` the vote prompt builder accepts.
    """
    return NormalizedMarket(
        exchange="fake-exchange",
        ticker="KXWIRE-01",
        event_ticker="KXWIRE",
        title="Does the configured rate reach the meter?",
        resolution_criteria="Resolves YES if it does.",
        category="economics",
        close_time=datetime(2024, 12, 18, 19, tzinfo=UTC),
        expected_resolution_time=None,
        market_type="fully_collateralized_binary",
        price_tick_pips=100,
        min_order_contract_centis=100,
        fractional_trading_enabled=False,
        mutually_exclusive_group_id=None,
        jurisdiction_status="eligible",
        raw_exchange_payload_hash="sha256:abc123",
        volume_24h_micros=0,
    )


def _baseline() -> BaselineQuoteSnapshot:
    """Build the baseline quote snapshot the vote prompt renders.

    Returns:
        A `BaselineQuoteSnapshot` for `_market`.
    """
    return BaselineQuoteSnapshot(
        snapshot_id="snap-wiring-0001",
        price_pips=4500,
        fetched_at=datetime(2024, 12, 10, 12, tzinfo=UTC),
    )


def _config(**transport: object) -> WindbreakConfig:
    """Build a configuration whose provider transport section is overridden.

    Args:
        **transport: `ProviderTransportConfig` field overrides.

    Returns:
        The configuration under test.
    """
    base = WindbreakConfig()
    forecast = dataclasses.replace(
        base.forecast, provider_transport=ProviderTransportConfig(**transport)
    )
    return dataclasses.replace(base, forecast=forecast)


def _live_http(
    llm: dict[str, object] | None = None, *, research: bool = False
) -> LiveProviderHttp:
    """Build a live seam bundle over stub transports.

    Args:
        llm: The per-provider transports, or `None` for both known providers.
        research: Whether to supply live research transports.

    Returns:
        The bundle under test.
    """
    stub = _StubHttpTransport()
    return LiveProviderHttp(
        llm=llm if llm is not None else {"anthropic": stub, "openai": stub},
        search=stub if research else None,
        fetch=stub if research else None,
        futuresearch=None,
    )


# --- Mode selection ---------------------------------------------------------------


def test_cassette_mode_reads_as_not_live() -> None:
    """The default mode is the offline one."""
    assert is_live_mode(_config(mode=PROVIDER_TRANSPORT_CASSETTE)) is False


def test_live_mode_reads_as_live() -> None:
    """The explicit live token selects live."""
    assert is_live_mode(_config(mode=PROVIDER_TRANSPORT_LIVE)) is True


def test_an_unknown_mode_refuses_rather_than_defaulting() -> None:
    """Silently choosing a default would discard a stated operator intent."""
    with pytest.raises(ValueError, match="unknown"):
        is_live_mode(_config(mode="almost-live"))


def test_the_unknown_mode_message_names_both_valid_modes() -> None:
    """An operator who mistyped needs to be told what was expected."""
    with pytest.raises(ValueError) as excinfo:
        is_live_mode(_config(mode="almost-live"))

    assert PROVIDER_TRANSPORT_CASSETTE in str(excinfo.value)
    assert PROVIDER_TRANSPORT_LIVE in str(excinfo.value)


# --- Retry policy and price table translation -------------------------------------


def test_the_retry_policy_carries_every_configured_bound() -> None:
    """All four bounds reach the engine's policy, none defaulted away."""
    policy = retry_policy_from_config(
        _config(
            retry=ProviderRetryConfig(
                max_attempts=7,
                total_deadline_ms=12_345,
                backoff_base_ms=13,
                max_cost_micros=4_242_424,
            )
        )
    )

    assert policy.max_attempts == 7
    assert policy.total_deadline_ms == 12_345
    assert policy.backoff_base_ms == 13
    assert policy.max_cost_micros == 4_242_424


def test_a_non_positive_retry_bound_refuses_to_start() -> None:
    """An unbounded retry loop against a paid provider must not be buildable."""
    with pytest.raises(ValueError, match="max_attempts"):
        retry_policy_from_config(_config(retry=ProviderRetryConfig(max_attempts=0)))


def test_the_price_table_carries_every_configured_price() -> None:
    """Operator prices reach the table verbatim."""
    table = price_table_from_config(
        _config(
            prices=(ProviderPrice("anthropic", 111), ProviderPrice("openai", 222)),
            unknown_provider_price_micros=999,
        )
    )

    assert table.price_micros("anthropic") == 111
    assert table.price_micros("openai") == 222


def test_an_unpriced_provider_falls_back_to_the_configured_ceiling() -> None:
    """An unknown provider is priced high, never free."""
    table = price_table_from_config(
        _config(
            prices=(ProviderPrice("anthropic", 111),),
            unknown_provider_price_micros=999,
        )
    )

    assert table.price_micros("a-provider-nobody-priced") == 999


def test_a_zero_price_refuses_to_start() -> None:
    """A zero list price would let a provider evade its budget entirely."""
    with pytest.raises(ValueError):
        price_table_from_config(_config(prices=(ProviderPrice("anthropic", 0),)))


# --- Metered rate table (issue #451) ------------------------------------------------


def test_the_rate_table_carries_every_configured_token_rate() -> None:
    """Operator token rates reach the metered table verbatim."""
    table = rate_table_from_config(
        _config(
            token_prices=(
                ModelTokenPrice("model-a", 3_000_000, 15_000_000),
                ModelTokenPrice("model-b", 1_000_000, 4_000_000),
            ),
            unmetered_response_micros=888,
        )
    )

    assert (
        table.micros_for(
            model_version="model-a",
            usage=TokenUsage(input_tokens=1_000, output_tokens=100),
        )
        == 4_500
    )
    assert (
        table.micros_for(
            model_version="model-b",
            usage=TokenUsage(input_tokens=1_000, output_tokens=100),
        )
        == 1_400
    )


def test_an_unrated_model_falls_back_to_the_configured_unmetered_charge() -> None:
    """A model the operator never rated is charged high, never free."""
    table = rate_table_from_config(
        _config(
            token_prices=(ModelTokenPrice("model-a", 3_000_000, 15_000_000),),
            unmetered_response_micros=888,
        )
    )

    assert (
        table.micros_for(
            model_version="a-model-nobody-rated",
            usage=TokenUsage(input_tokens=1_000, output_tokens=100),
        )
        == 888
    )


def test_a_zero_token_rate_refuses_to_start() -> None:
    """A zero-rated model would bill nothing however many tokens it burned."""
    with pytest.raises(ValueError):
        rate_table_from_config(
            _config(token_prices=(ModelTokenPrice("model-a", 0, 15_000_000),))
        )


def test_a_zero_unmetered_charge_refuses_to_start() -> None:
    """A zero fail-closed charge would make every unmeasurable vote free."""
    with pytest.raises(ValueError):
        rate_table_from_config(_config(unmetered_response_micros=0))


# --- Provider factory ---------------------------------------------------------------


def test_the_offline_factory_builds_a_bare_fixture_provider() -> None:
    """Replaying a recording is neither priced nor retried."""
    factory = build_provider_factory(_config(), live=False, provider_http=None)

    provider = factory(_StubLlmTransport(), _StubMember("openai"))

    assert isinstance(provider, FixtureVoteProvider)


def test_the_live_factory_wraps_in_a_retrying_provider() -> None:
    """Every live vote runs under the configured bounded-retry policy."""
    factory = build_provider_factory(
        _config(mode=PROVIDER_TRANSPORT_LIVE), live=True, provider_http=_live_http()
    )

    provider = factory(_StubLlmTransport(), _StubMember("openai"))

    assert isinstance(provider, RetryingProvider)


def test_the_live_factory_meters_a_vote_at_the_configured_token_rates() -> None:
    """The configured rate table actually reaches the vote that is charged.

    The wiring test, not a table test: it drives a real vote through the
    factory the composition root returns and asserts the *charge*, so deleting
    the ``rate_table=`` argument in ``build_provider_factory`` fails here even
    though every table-level test above still passes. The configured rate is
    deliberately unlike the configured list price, so the assertion cannot be
    satisfied by the pre-gate estimate.
    """
    factory = build_provider_factory(
        _config(
            mode=PROVIDER_TRANSPORT_LIVE,
            prices=(ProviderPrice("openai", 200_000),),
            token_prices=(ModelTokenPrice("pinned-for-test", 3_000_000, 15_000_000),),
            unmetered_response_micros=888_000,
        ),
        live=True,
        provider_http=_live_http(),
    )
    transport = _StubLlmTransport(
        response=_VOTE_JSON, usage=TokenUsage(input_tokens=1_000, output_tokens=100)
    )

    provider = factory(transport, _StubMember("openai"))
    result = provider.forecast(_market(), _baseline(), 0, ())

    assert result.cost_micros == 4_500


def test_the_live_factory_fails_closed_on_a_vote_reporting_no_usage() -> None:
    """A live vote whose response reported no usage books the configured bound.

    Proves the fail-closed figure is wired too, not merely the rates: without
    it a response with no token accounting would be charged nothing.
    """
    factory = build_provider_factory(
        _config(
            mode=PROVIDER_TRANSPORT_LIVE,
            prices=(ProviderPrice("openai", 200_000),),
            token_prices=(ModelTokenPrice("pinned-for-test", 3_000_000, 15_000_000),),
            unmetered_response_micros=888_000,
        ),
        live=True,
        provider_http=_live_http(),
    )
    transport = _StubLlmTransport(response=_VOTE_JSON)

    provider = factory(transport, _StubMember("openai"))
    result = provider.forecast(_market(), _baseline(), 0, ())

    assert result.cost_micros == 888_000


def test_the_factory_takes_the_transport_per_call() -> None:
    """The factory is a policy, not a closure over one transport.

    This is what keeps ``dataclasses.replace(deps, transport=...)`` reaching the
    vote stage instead of voting against whatever was wired at composition time.
    """
    factory = build_provider_factory(_config(), live=False, provider_http=None)
    first, second = _StubLlmTransport(), _StubLlmTransport()

    assert factory(first, _StubMember("openai")) is not factory(
        second, _StubMember("openai")
    )


# --- Live LLM routing ---------------------------------------------------------------


def _ensemble_config(*providers: str) -> WindbreakConfig:
    """Build a live-mode config whose vote ensemble names ``providers``.

    Args:
        *providers: The provider identifiers the ensemble draws on.

    Returns:
        The configuration under test.
    """
    base = WindbreakConfig()
    forecast = dataclasses.replace(
        base.forecast,
        provider_transport=ProviderTransportConfig(mode=PROVIDER_TRANSPORT_LIVE),
        vote_ensemble=tuple(
            EnsembleMemberConfig(provider, f"{provider}-pinned", "2025-01-01")
            for provider in providers
        ),
    )
    return dataclasses.replace(base, forecast=forecast)


def test_every_known_provider_gets_an_adapter() -> None:
    """Both providers with a live adapter are routable."""
    transport = build_live_llm_transport(
        _ensemble_config("anthropic", "openai"), _live_http()
    )

    assert transport.complete is not None


def test_a_provider_with_no_adapter_is_discarded_as_a_vote_failure() -> None:
    """An unroutable provider fails as a *vote*, never as a bare `KeyError`.

    A `KeyError` is caught by neither `RetryingProvider.forecast` nor
    `pipeline._collect_provider_forecasts` (both catch `ProviderVoteError`
    only), so it would escape `run_pipeline` and crash the whole tick. Raising a
    taxonomy leaf instead means the vote is discarded like any other failure and
    the run degrades to quorum abstention.
    """
    from windbreak.forecast.cassettes import LlmRequest
    from windbreak.forecast.providers import ProviderNotRoutableError

    transport = build_live_llm_transport(
        _ensemble_config("a-provider-with-no-adapter"),
        _live_http({"a-provider-with-no-adapter": _StubHttpTransport()}),
        validate=False,
    )

    with pytest.raises(ProviderNotRoutableError) as excinfo:
        transport.complete(
            LlmRequest(
                provider="a-provider-with-no-adapter",
                model_version="v",
                prompt="p",
            )
        )

    assert "a-provider-with-no-adapter" in str(excinfo.value)


def test_an_unroutable_provider_failure_is_a_vote_error() -> None:
    """It must sit under the type both discard paths actually catch."""
    from windbreak.forecast.cassettes import LlmRequest
    from windbreak.forecast.providers import ProviderVoteError

    transport = build_live_llm_transport(
        _ensemble_config("nope"),
        _live_http({"nope": _StubHttpTransport()}),
        validate=False,
    )

    with pytest.raises(ProviderVoteError):
        transport.complete(LlmRequest(provider="nope", model_version="v", prompt="p"))


def test_an_unroutable_provider_failure_is_not_retried() -> None:
    """A missing route will not appear on a second attempt."""
    from windbreak.forecast.providers import ProviderNotRoutableError
    from windbreak.forecast.providers.retry import _is_retryable

    assert not _is_retryable(ProviderNotRoutableError("nope"))


# --- Startup validation: an unroutable ensemble refuses to start ------------------


def test_a_live_ensemble_naming_an_unroutable_provider_refuses_to_start() -> None:
    """The misconfiguration is refused cleanly, before any tick runs."""
    with pytest.raises(ValueError, match="futuresearch"):
        build_live_llm_transport(
            _ensemble_config("futuresearch"),
            _live_http({"futuresearch": _StubHttpTransport()}),
        )


def test_the_refusal_names_the_routable_providers() -> None:
    """An operator who typo'd a provider needs to be told what is routable."""
    with pytest.raises(ValueError) as excinfo:
        build_live_llm_transport(
            _ensemble_config("anthropi"), _live_http({"anthropi": _StubHttpTransport()})
        )

    assert "anthropic" in str(excinfo.value)


def test_an_ensemble_provider_absent_from_the_supplied_seams_refuses() -> None:
    """A routable provider still needs a transport actually built for it."""
    with pytest.raises(ValueError, match="openai"):
        build_live_llm_transport(
            _ensemble_config("openai"), _live_http({"anthropic": _StubHttpTransport()})
        )


def test_a_fully_routable_live_ensemble_builds() -> None:
    """The supported configuration is unaffected by the new guard."""
    transport = build_live_llm_transport(
        _ensemble_config("anthropic", "openai"), _live_http()
    )

    assert transport is not None


def test_a_provider_with_no_adapter_is_dropped_rather_than_guessed_at() -> None:
    """An unknown provider has no live route, so it fails closed at vote time."""
    from windbreak.forecast.cassettes import LlmRequest
    from windbreak.forecast.providers import ProviderNotRoutableError

    transport = build_live_llm_transport(
        _ensemble_config("anthropic"), _live_http(), validate=False
    )

    with pytest.raises(ProviderNotRoutableError):
        transport.complete(
            LlmRequest(provider="futuresearch", model_version="v", prompt="p")
        )


# --- Offline research fallback ------------------------------------------------------


def test_the_offline_research_transport_finds_nothing() -> None:
    """Finding nothing is what makes the pipeline abstain before any vote."""
    assert OfflineResearchTransport().search("any subquestion") == ()


def test_the_offline_research_transport_refuses_a_fetch() -> None:
    """Search finds nothing, so reaching fetch at all is a wiring bug."""
    with pytest.raises(RuntimeError, match="unexpectedly called"):
        OfflineResearchTransport().fetch("https://research.local/anything")


def test_offline_research_tools_are_capability_closed(tmp_path: Path) -> None:
    """The offline bundle builds without a network and searches to empty."""
    tools = offline_research_tools(tmp_path / "cache")

    assert tools.search("a subquestion") == ()


def test_live_mode_without_a_research_endpoint_falls_back_to_offline(
    tmp_path: Path,
) -> None:
    """Live providers and live research are configured independently.

    A deployment that pinned an LLM but no search endpoint must not be forced to
    invent one; research finds nothing and the pipeline abstains on zero
    verified citations, which fails closed.
    """
    tools = build_live_research_tools(
        _config(mode=PROVIDER_TRANSPORT_LIVE), _live_http(), tmp_path / "cache"
    )

    assert tools.search("a subquestion") == ()


def test_configured_live_research_builds_the_live_transports(
    tmp_path: Path,
) -> None:
    """With both halves supplied the sandbox is wired to the live seams."""
    base = WindbreakConfig()
    forecast = ForecastConfig(
        provider_transport=ProviderTransportConfig(mode=PROVIDER_TRANSPORT_LIVE),
        research=ResearchSettings(
            search_endpoint_url="https://search.example.com/search",
            allowed_research_hosts=("research.example.com",),
        ),
    )
    config = dataclasses.replace(base, forecast=forecast)

    tools = build_live_research_tools(
        config, _live_http(research=True), tmp_path / "cache"
    )

    assert tools is not None


# --- The research forecaster's config-to-engine translation (issue #555) -------

#: The version the research-forecaster members below pin, and the one the
#: section admits, unless a test overrides one of the two.
_RESEARCH_VERSION = "fs-2026-06-01"

#: The environment variable the research-forecaster section names, and an
#: alternate one used to prove the leaf is carried rather than defaulted. Bound
#: to constants, never written as literals beside an ``api_key_env`` keyword:
#: the repo's secret scanner reads that shape as a credential, and renaming the
#: fixture is the structural fix the hook asks for (PRs #260/#282).
_RESEARCH_ENV_VAR = "FUTURESEARCH_API_KEY"
_ALTERNATE_ENV_VAR = "A_DIFFERENT_VARIABLE_NAME"


def _research_forecaster_config(
    *,
    members: int = 1,
    model_version: str = _RESEARCH_VERSION,
    endpoint_url: str = "https://futuresearch.example/v1/forecast",
    pinned_versions: tuple[str, ...] = (_RESEARCH_VERSION,),
    env_var: str = _RESEARCH_ENV_VAR,
    per_call_ceiling_micros: int = 1_250_000,
    reject_on_version_drift: bool = True,
) -> WindbreakConfig:
    """Build a live config selecting the hosted research forecaster.

    Args:
        members: How many `futuresearch` vote-ensemble members to name.
        model_version: The version each member pins.
        endpoint_url: The section's endpoint.
        pinned_versions: The versions the section admits.
        env_var: The environment variable the section names.
        per_call_ceiling_micros: The section's fail-closed per-call charge.
        reject_on_version_drift: The section's drift policy.

    Returns:
        The configuration under test.
    """
    from windbreak.config.schema import FutureSearchProviderSettings

    base = WindbreakConfig()
    forecast = dataclasses.replace(
        base.forecast,
        provider_transport=ProviderTransportConfig(mode=PROVIDER_TRANSPORT_LIVE),
        vote_ensemble=tuple(
            EnsembleMemberConfig("futuresearch", model_version, "server-managed")
            for _ in range(members)
        ),
        futuresearch=FutureSearchProviderSettings(
            endpoint_url=endpoint_url,
            pinned_forecaster_versions=pinned_versions,
            api_key_env=env_var,
            per_call_ceiling_micros=per_call_ceiling_micros,
            reject_on_version_drift=reject_on_version_drift,
        ),
    )
    return dataclasses.replace(base, forecast=forecast)


def _research_seams(*, seam: object = None) -> LiveProviderHttp:
    """Build a seam bundle carrying a research-forecaster transport.

    Args:
        seam: The research-forecaster HTTP transport, or `None` for a stub.

    Returns:
        The bundle under test.
    """
    stub = _StubHttpTransport()
    return LiveProviderHttp(
        llm={},
        search=None,
        fetch=None,
        futuresearch=seam if seam is not None else stub,
    )


def test_every_configured_research_forecaster_leaf_reaches_the_engine() -> None:
    """All five leaves cross the SPEC S8.3 boundary, none defaulted silently.

    The section was read by nothing in `windbreak/` before issue #555. Asserting
    the whole translated record -- rather than one field -- is what makes a leaf
    dropped from this adapter a failure instead of a silent revert to the
    engine's own dataclass default.
    """
    from windbreak.forecast.providers import FutureSearchProviderConfig
    from windbreak.scheduler.provider_wiring import futuresearch_config_from_config

    resolved = futuresearch_config_from_config(
        _research_forecaster_config(
            endpoint_url="https://pinned.example/v1/forecast",
            pinned_versions=(_RESEARCH_VERSION, "fs-2026-07-01"),
            env_var=_ALTERNATE_ENV_VAR,
            per_call_ceiling_micros=1_750_000,
            reject_on_version_drift=False,
        )
    )

    assert resolved == FutureSearchProviderConfig(
        endpoint_url="https://pinned.example/v1/forecast",
        pinned_forecaster_versions=(_RESEARCH_VERSION, "fs-2026-07-01"),
        api_key_env=_ALTERNATE_ENV_VAR,
        per_call_ceiling_micros=1_750_000,
        reject_on_version_drift=False,
    )


def test_no_translated_leaf_equals_the_engines_own_default() -> None:
    """Guards the assertion above against a coincidence.

    Every override in that test must differ from `FutureSearchProviderConfig`'s
    own default, or an adapter that dropped the leaf entirely would still
    produce the expected record.
    """
    from windbreak.forecast.providers import FutureSearchProviderConfig
    from windbreak.scheduler.provider_wiring import futuresearch_config_from_config

    resolved = futuresearch_config_from_config(
        _research_forecaster_config(
            endpoint_url="https://pinned.example/v1/forecast",
            pinned_versions=(_RESEARCH_VERSION, "fs-2026-07-01"),
            env_var=_ALTERNATE_ENV_VAR,
            per_call_ceiling_micros=1_750_000,
            reject_on_version_drift=False,
        )
    )
    engine_default = FutureSearchProviderConfig(
        endpoint_url="", pinned_forecaster_versions=()
    )

    assert resolved.api_key_env != engine_default.api_key_env
    assert resolved.per_call_ceiling_micros != engine_default.per_call_ceiling_micros
    assert resolved.reject_on_version_drift != engine_default.reject_on_version_drift


def test_the_live_factory_builds_the_research_forecaster_for_its_member() -> None:
    """A `futuresearch` member is routed to its own provider, not the seam."""
    from windbreak.forecast.providers import FutureSearchProvider

    factory = build_provider_factory(
        _research_forecaster_config(), live=True, provider_http=_research_seams()
    )

    provider = factory(_StubLlmTransport(), _StubMember("futuresearch"))

    assert isinstance(provider, FutureSearchProvider)


def test_the_live_factory_still_wraps_a_completion_member_beside_it() -> None:
    """One research forecaster in the ensemble must not unwrap the LLM members.

    A guard that dispatched on the *configuration* rather than the member would
    agree with this one for a single-family ensemble and disagree here.
    """
    factory = build_provider_factory(
        _research_forecaster_config(), live=True, provider_http=_research_seams()
    )

    provider = factory(_StubLlmTransport(), _StubMember("openai"))

    assert isinstance(provider, RetryingProvider)


def test_a_live_factory_without_the_seam_bundle_refuses() -> None:
    """A live factory that cannot reach the seams is not built at all.

    Without this the composition root could drop `provider_http=` and still
    return a factory that looks wired and silently routes no research
    forecaster.
    """
    with pytest.raises(ValueError, match="provider_http"):
        build_provider_factory(
            _config(mode=PROVIDER_TRANSPORT_LIVE), live=True, provider_http=None
        )


def test_the_research_forecaster_is_not_wrapped_in_the_metering_layer() -> None:
    """Its reported cost must reach the ledger unchanged.

    `RetryingProvider._finalize_success` adds the rate table's fail-closed
    `unmetered_micros` to any response carrying no token usage, and a research
    response reports dollars rather than tokens -- so wrapping would add a flat
    constant on top of a measurement.
    """
    factory = build_provider_factory(
        _research_forecaster_config(), live=True, provider_http=_research_seams()
    )

    provider = factory(_StubLlmTransport(), _StubMember("futuresearch"))

    assert not isinstance(provider, RetryingProvider)
