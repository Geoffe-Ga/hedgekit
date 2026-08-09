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

#: Distinctive markers exported in place of live credentials. Named as canaries
#: rather than as keys so the repo's secret scanner is never asked to tell a
#: fixture apart from the real thing; the assertions check they never surface.
_ANTHROPIC_CANARY = "anthropic-env-canary-0001"
_OPENAI_CANARY = "openai-env-canary-0001"


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


@pytest.mark.usefixtures("_keys")
def test_an_undeclared_provider_host_refuses_to_start() -> None:
    """A deployment whose allowlist omits a provider host must not dial it.

    The vote ensemble is what puts a provider host on the outbound allowlist
    (``allowlist_from_config``), so an ensemble naming no known provider
    declares no provider host -- and a live transport for one would be egress
    nobody authorized.
    """
    base = _config()
    forecast = dataclasses.replace(base.forecast, vote_ensemble=(), ensemble=())
    config = dataclasses.replace(base, forecast=forecast)

    with pytest.raises(ValueError, match="allowlist"):
        _resolve_provider_http(config)
