"""Provider transport / retry / pricing configuration (issues #344, #269).

Issue #344 needs configuration to *select* the forecast provider transport, and
issue #269 needs the ``RetryingProvider`` policy and ``ProviderPriceTable`` to
be config-sourced rather than hardcoded at the injection site. These tests pin
the three properties that make that safe:

1. **The cassette is the default.** ``WindbreakConfig()`` selects the recorded
   replay transport, so CI -- which builds the default config -- never acquires
   a live network dependency by omission. Turning on live providers must be an
   explicit, written act.
2. **Config carries secret *names*, never secret values.** Every config leaf is
   flattened verbatim into the hash-chained ledger through ``diff_configs`` ->
   ``ConfigLoaded``, so an API key in configuration is an API key in the audit
   trail forever. The provider transport section names environment variables.
3. **The mirrored defaults really mirror.** ``config.schema`` is deliberately
   dependency-free (it imports nothing from ``windbreak``), so the price and
   retry defaults are written out here and their equality with the forecast
   engine's own :data:`DEFAULT_PROVIDER_PRICE_TABLE` / retry constants is
   pinned by test -- the same idiom ``_default_vote_ensemble`` and
   ``ProviderGateConfig`` already use.
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
from windbreak.config.versioning import flatten
from windbreak.forecast.budget import DEFAULT_PROVIDER_PRICE_TABLE
from windbreak.forecast.providers.retry import (
    DEFAULT_BACKOFF_BASE_MS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TOTAL_DEADLINE_MS,
)


def test_the_default_configuration_selects_the_cassette_transport() -> None:
    """An omitted section leaves CI offline: the replay cassette is the default."""
    assert WindbreakConfig().forecast.provider_transport.mode == (
        PROVIDER_TRANSPORT_CASSETTE
    )


def test_the_two_transport_modes_are_distinct_named_constants() -> None:
    """The live mode is a distinct, explicitly-spelled token."""
    assert PROVIDER_TRANSPORT_CASSETTE != PROVIDER_TRANSPORT_LIVE


def test_retry_defaults_mirror_the_forecast_engines_own_constants() -> None:
    """The config-sourced retry policy defaults equal the engine's documented ones."""
    retry = ProviderTransportConfig().retry

    assert retry.max_attempts == DEFAULT_MAX_ATTEMPTS
    assert retry.total_deadline_ms == DEFAULT_TOTAL_DEADLINE_MS
    assert retry.backoff_base_ms == DEFAULT_BACKOFF_BASE_MS


def test_price_defaults_agree_with_the_engines_default_price_table() -> None:
    """Where the two tables overlap they must not disagree on a price."""
    configured = {
        price.provider: price.price_micros for price in ProviderTransportConfig().prices
    }
    engine = dict(DEFAULT_PROVIDER_PRICE_TABLE.prices_micros)

    assert all(engine[provider] == price for provider, price in configured.items())


def test_price_defaults_cover_exactly_the_live_routable_providers() -> None:
    """The default table must not advertise a provider the root would refuse.

    ``futuresearch`` is priced by the *engine's* table but has no routed
    completion transport (its provider does its own research and is not an
    ``LlmTransport``), so listing it here would imply live support that startup
    validation then rejects.
    """
    from windbreak.scheduler.provider_wiring import routable_live_providers

    configured = {price.provider for price in ProviderTransportConfig().prices}

    assert configured == routable_live_providers()


def test_unknown_provider_fallback_price_mirrors_the_engines_own() -> None:
    """An unpriced provider stays conservatively priced, never free."""
    assert (
        ProviderTransportConfig().unknown_provider_price_micros
        == DEFAULT_PROVIDER_PRICE_TABLE.unknown_provider_price_micros
    )


def test_every_configured_price_is_strictly_positive() -> None:
    """A zero or negative list price would let a provider evade its budget."""
    assert all(price.price_micros > 0 for price in ProviderTransportConfig().prices)


def test_api_key_leaves_name_environment_variables_never_hold_secrets() -> None:
    """Each key leaf is an env-var *name*, matching the repo's ``*_api_key_env``.

    Asserted by *shape* rather than against a literal: an upper-snake-case
    identifier is what an environment variable name looks like, and no real
    credential does. A value here would be flattened into the hash-chained
    ledger by ``diff_configs`` and be unremovable from the audit trail.
    """
    transport = ProviderTransportConfig()
    anthropic_var = transport.anthropic_api_key_env
    openai_var = transport.openai_api_key_env

    for variable in (anthropic_var, openai_var):
        assert variable.isupper()
        assert variable.replace("_", "").isalnum()
    assert "ANTHROPIC" in anthropic_var
    assert "OPENAI" in openai_var


def test_no_provider_transport_leaf_is_spelled_as_a_bare_secret() -> None:
    """No leaf name suggests it holds a key itself rather than naming its var."""
    suspicious = [
        f.name
        for f in dataclasses.fields(ProviderTransportConfig)
        if "api_key" in f.name and not f.name.endswith("_api_key_env")
    ]

    assert suspicious == []


def test_the_section_flattens_into_the_ledger_diff_surface() -> None:
    """The new leaves reach ``ConfigLoaded`` like every other config leaf."""
    flat = flatten(WindbreakConfig())

    assert flat["forecast.provider_transport.mode"] == PROVIDER_TRANSPORT_CASSETTE
    assert flat["forecast.provider_transport.retry.max_attempts"] == (
        DEFAULT_MAX_ATTEMPTS
    )


def test_the_section_is_frozen() -> None:
    """Configuration is immutable once loaded, like every other section."""
    transport = ProviderTransportConfig()

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        transport.mode = PROVIDER_TRANSPORT_LIVE


def test_request_timeout_is_whole_seconds() -> None:
    """The dial timeout is an integer, keeping the money path float-free."""
    timeout = ProviderTransportConfig().request_timeout_seconds

    assert isinstance(timeout, int)
    assert timeout > 0
