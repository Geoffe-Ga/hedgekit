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
from typing import TYPE_CHECKING

import pytest

from windbreak.config.schema import (
    PROVIDER_TRANSPORT_CASSETTE,
    PROVIDER_TRANSPORT_LIVE,
    EnsembleMemberConfig,
    ForecastConfig,
    ProviderPrice,
    ProviderRetryConfig,
    ProviderTransportConfig,
    ResearchSettings,
    WindbreakConfig,
)
from windbreak.forecast.providers import FixtureVoteProvider, RetryingProvider
from windbreak.scheduler.provider_wiring import (
    LiveProviderHttp,
    OfflineResearchTransport,
    build_live_llm_transport,
    build_live_research_tools,
    build_provider_factory,
    is_live_mode,
    offline_research_tools,
    price_table_from_config,
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


class _StubLlmTransport:
    """An `LlmTransport` double returning a fixed completion."""

    def complete(self, request: object) -> str:
        """Return a fixed completion.

        Args:
            request: The (unused) completion request.

        Returns:
            A fixed string.
        """
        del request
        return "{}"


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


# --- Provider factory ---------------------------------------------------------------


def test_the_offline_factory_builds_a_bare_fixture_provider() -> None:
    """Replaying a recording is neither priced nor retried."""
    factory = build_provider_factory(_config(), live=False)

    provider = factory(_StubLlmTransport(), _StubMember("openai"))

    assert isinstance(provider, FixtureVoteProvider)


def test_the_live_factory_wraps_in_a_retrying_provider() -> None:
    """Every live vote runs under the configured bounded-retry policy."""
    factory = build_provider_factory(_config(mode=PROVIDER_TRANSPORT_LIVE), live=True)

    provider = factory(_StubLlmTransport(), _StubMember("openai"))

    assert isinstance(provider, RetryingProvider)


def test_the_factory_takes_the_transport_per_call() -> None:
    """The factory is a policy, not a closure over one transport.

    This is what keeps ``dataclasses.replace(deps, transport=...)`` reaching the
    vote stage instead of voting against whatever was wired at composition time.
    """
    factory = build_provider_factory(_config(), live=False)
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
