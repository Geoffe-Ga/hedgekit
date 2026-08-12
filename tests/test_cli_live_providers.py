"""Resolving the live provider HTTP seams at the CLI composition root (#344).

``windbreak.scheduler.loop.build_paper_deps`` *selects* the live transport, but
something has to *build* it -- and building it is the one step that touches the
process environment, because configuration may only ever name the environment
variable a key is read from, never the key. That step belongs here, at the CLI
edge, exactly where ``_resolve_paper_market_data`` already resolves the live
venue surface.

These tests pin the properties that keep going live safe:

* **Nothing is built, and no environment variable is read, in cassette mode.**
  The default deployment resolves ``None`` and never looks for a key.
* **Every provider host is screened against the deployment's own allowlist
  before a session exists** (SPEC S15), mirroring ``_resolve_paper_market_data``
  -- a host nobody declared refuses to start rather than being dialed.
* **Each provider gets a single-host allowlist of its own.** Defence in depth:
  even holding the Anthropic transport, an attacker-supplied endpoint cannot
  make it dial OpenAI's host and hand over the wrong key.
* **A missing key fails closed, naming the variable and never the value.**
"""

from __future__ import annotations

import dataclasses

import pytest

from windbreak.config.schema import (
    PROVIDER_TRANSPORT_CASSETTE,
    PROVIDER_TRANSPORT_LIVE,
    ProviderTransportConfig,
    WindbreakConfig,
)
from windbreak.main import _resolve_provider_http
from windbreak.net.allowlist import EgressDeniedError

#: The environment variables the default configuration names.
_ANTHROPIC_ENV = "ANTHROPIC_API_KEY"
_OPENAI_ENV = "OPENAI_API_KEY"

#: The environment variable the hosted research forecaster's section names.
_FUTURESEARCH_ENV = "FUTURESEARCH_API_KEY"

#: Distinctive markers exported in place of live credentials. Named as canaries
#: rather than as keys so the repo's secret scanner is never asked to tell a
#: fixture apart from the real thing; the assertions check they never surface.
_ANTHROPIC_CANARY = "anthropic-env-canary-0001"
_OPENAI_CANARY = "openai-env-canary-0001"
_FUTURESEARCH_CANARY = "futuresearch-env-canary-0001"

#: The research forecaster endpoint the tests below configure. Its host reaches
#: the deployment allowlist only because a vote-ensemble member names the
#: provider -- both fields, or no egress.
_FUTURESEARCH_ENDPOINT = "https://futuresearch.example/v1/forecast"


def _config(mode: str = PROVIDER_TRANSPORT_LIVE) -> WindbreakConfig:
    """Build a configuration selecting ``mode``.

    Args:
        mode: The provider transport mode.

    Returns:
        The configuration under test.
    """
    base = WindbreakConfig()
    forecast = dataclasses.replace(
        base.forecast, provider_transport=ProviderTransportConfig(mode=mode)
    )
    return dataclasses.replace(base, forecast=forecast)


@pytest.fixture
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Export both provider keys for the duration of one test.

    Args:
        monkeypatch: The pytest environment patcher.
    """
    monkeypatch.setenv(_ANTHROPIC_ENV, _ANTHROPIC_CANARY)
    monkeypatch.setenv(_OPENAI_ENV, _OPENAI_CANARY)


def test_cassette_mode_resolves_no_live_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default deployment builds nothing at all."""
    monkeypatch.delenv(_ANTHROPIC_ENV, raising=False)
    monkeypatch.delenv(_OPENAI_ENV, raising=False)

    assert _resolve_provider_http(_config(PROVIDER_TRANSPORT_CASSETTE)) is None


def test_cassette_mode_reads_no_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An offline deployment need not have any provider key exported at all."""
    monkeypatch.delenv(_ANTHROPIC_ENV, raising=False)
    monkeypatch.delenv(_OPENAI_ENV, raising=False)

    _resolve_provider_http(_config(PROVIDER_TRANSPORT_CASSETTE))


@pytest.mark.usefixtures("_keys")
def test_live_mode_builds_a_transport_for_each_ensemble_provider() -> None:
    """Both providers the default vote ensemble draws on are routable."""
    resolved = _resolve_provider_http(_config())

    assert resolved is not None
    assert set(resolved.llm) == {"anthropic", "openai"}


@pytest.mark.usefixtures("_keys")
def test_each_provider_transport_refuses_another_providers_host() -> None:
    """A single-host allowlist each: the Anthropic key cannot reach OpenAI."""
    from windbreak.forecast.providers import HttpRequest

    resolved = _resolve_provider_http(_config())
    assert resolved is not None

    with pytest.raises(EgressDeniedError):
        resolved.llm["anthropic"].send(
            HttpRequest(
                method="POST",
                url="https://api.openai.com/v1/chat/completions",
                body="{}",
            )
        )


def test_a_missing_key_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexported key aborts startup rather than dialing unauthenticated."""
    monkeypatch.setenv(_OPENAI_ENV, _OPENAI_CANARY)
    monkeypatch.delenv(_ANTHROPIC_ENV, raising=False)

    with pytest.raises(ValueError, match=_ANTHROPIC_ENV):
        _resolve_provider_http(_config())


def test_a_missing_key_error_names_the_variable_not_any_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal must not leak a neighbouring provider's exported secret."""
    monkeypatch.setenv(_OPENAI_ENV, _OPENAI_CANARY)
    monkeypatch.delenv(_ANTHROPIC_ENV, raising=False)

    with pytest.raises(ValueError) as excinfo:
        _resolve_provider_http(_config())

    assert _OPENAI_CANARY not in str(excinfo.value)


def _ensemble_config(*providers: str) -> WindbreakConfig:
    """Build a live config whose vote ensemble names exactly ``providers``.

    Args:
        *providers: The provider identifiers the ensemble draws on.

    Returns:
        The configuration under test.
    """
    from windbreak.config.schema import EnsembleMemberConfig

    base = _config()
    forecast = dataclasses.replace(
        base.forecast,
        vote_ensemble=tuple(
            EnsembleMemberConfig(provider, f"{provider}-pinned", "2025-01-01")
            for provider in providers
        ),
    )
    return dataclasses.replace(base, forecast=forecast)


def _futuresearch_config() -> WindbreakConfig:
    """Build a live config naming one LLM member and one research forecaster.

    Returns:
        The configuration under test, with a real research-forecaster endpoint
        and the pinned version its member declares.
    """
    from windbreak.config.schema import FutureSearchProviderSettings

    base = _ensemble_config("anthropic", "futuresearch")
    forecast = dataclasses.replace(
        base.forecast,
        futuresearch=FutureSearchProviderSettings(
            endpoint_url=_FUTURESEARCH_ENDPOINT,
            pinned_forecaster_versions=("futuresearch-pinned",),
            api_key_env=_FUTURESEARCH_ENV,
        ),
    )
    return dataclasses.replace(base, forecast=forecast)


def test_only_the_configured_ensembles_providers_need_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An Anthropic-only ensemble must not demand an OpenAI key.

    Building a transport for a provider nobody configured is what made an
    unroutable ensemble member possible in the first place.
    """
    monkeypatch.setenv(_ANTHROPIC_ENV, _ANTHROPIC_CANARY)
    monkeypatch.delenv(_OPENAI_ENV, raising=False)

    resolved = _resolve_provider_http(_ensemble_config("anthropic"))

    assert resolved is not None
    assert set(resolved.llm) == {"anthropic"}


@pytest.mark.usefixtures("_keys")
def test_an_ensemble_naming_an_unroutable_provider_refuses_to_start() -> None:
    """A provider with no live route at all is refused, naming it.

    ``futuresearch`` was this test's example until issue #555 made it routable
    over its own HTTP seam; a typo'd vendor name is the case that remains.
    """
    with pytest.raises(ValueError, match="not-a-vendor"):
        _resolve_provider_http(_ensemble_config("anthropic", "not-a-vendor"))


@pytest.mark.usefixtures("_keys")
def test_the_research_forecaster_gets_its_own_credentialed_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``futuresearch`` member builds its own seam, not an LLM one (#555).

    The provider is a `ForecastProvider` over HTTP rather than a completion
    transport, so it must never appear in the routed ``llm`` mapping -- a
    transport parked there would be silently filtered out by
    ``build_live_llm_transport`` and route nothing.
    """
    monkeypatch.setenv(_FUTURESEARCH_ENV, _FUTURESEARCH_CANARY)

    resolved = _resolve_provider_http(_futuresearch_config())

    assert resolved is not None
    assert set(resolved.llm) == {"anthropic"}
    assert resolved.futuresearch is not None


@pytest.mark.usefixtures("_keys")
def test_an_unset_research_forecaster_key_refuses_naming_only_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing key fails closed, naming the variable and never a value."""
    monkeypatch.delenv(_FUTURESEARCH_ENV, raising=False)

    with pytest.raises(ValueError) as excinfo:
        _resolve_provider_http(_futuresearch_config())

    message = str(excinfo.value)
    assert _FUTURESEARCH_ENV in message
    assert _ANTHROPIC_CANARY not in message


def test_no_research_forecaster_member_demands_no_research_forecaster_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment that never selected it must not be asked for its key."""
    monkeypatch.setenv(_ANTHROPIC_ENV, _ANTHROPIC_CANARY)
    monkeypatch.delenv(_FUTURESEARCH_ENV, raising=False)

    resolved = _resolve_provider_http(_ensemble_config("anthropic"))

    assert resolved is not None
    assert resolved.futuresearch is None


@pytest.mark.usefixtures("_keys")
def test_the_unroutable_refusal_names_what_is_routable() -> None:
    """An operator who typo'd needs to be told the routable set."""
    with pytest.raises(ValueError) as excinfo:
        _resolve_provider_http(_ensemble_config("anthropi"))

    assert "anthropic" in str(excinfo.value)


@pytest.mark.usefixtures("_keys")
def test_a_repeated_provider_builds_one_transport() -> None:
    """The default ensemble names OpenAI twice; one transport is enough."""
    resolved = _resolve_provider_http(_ensemble_config("openai", "openai"))

    assert resolved is not None
    assert set(resolved.llm) == {"openai"}


def test_an_undeclared_provider_host_refuses_to_start() -> None:
    """A host this deployment's allowlist omits must not be dialed.

    Asserted against ``_live_http_for`` directly rather than through
    ``_resolve_provider_http``. Since transports are now built only for
    ``vote_ensemble`` providers, and ``allowlist_from_config`` derives its
    provider hosts from that very field, the two can no longer disagree by
    configuration -- this screen is defence in depth behind that agreement, and
    the honest way to exercise defence in depth is to aim at it directly.
    """
    from windbreak.main import _live_http_for

    base = _config()
    forecast = dataclasses.replace(base.forecast, vote_ensemble=(), ensemble=())
    config = dataclasses.replace(base, forecast=forecast)

    with pytest.raises(ValueError, match="allowlist"):
        _live_http_for(
            "https://api.anthropic.com/v1/messages", {}, config, timeout_seconds=30
        )


def test_the_undeclared_host_refusal_names_the_host() -> None:
    """An operator needs to know which host was not declared."""
    from windbreak.main import _live_http_for

    base = _config()
    forecast = dataclasses.replace(base.forecast, vote_ensemble=(), ensemble=())
    config = dataclasses.replace(base, forecast=forecast)

    with pytest.raises(ValueError) as excinfo:
        _live_http_for(
            "https://api.anthropic.com/v1/messages", {}, config, timeout_seconds=30
        )

    assert "api.anthropic.com" in str(excinfo.value)
