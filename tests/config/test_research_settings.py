"""Tests for `ForecastConfig.research` (issue #192, the live-research config seam).

Pins `windbreak.config.schema.ResearchSettings` -- the config-schema section
backing the new `windbreak.forecast.providers.search_live.LiveSearchTransport`
/ `windbreak.forecast.providers.fetch_live.LiveFetchTransport` pair -- and its
`forecast.research` YAML round-trip through `load_config`: every field parses
via the loader's existing generic dataclass coercion (no bespoke loader code
required, exactly like `forecast.futuresearch`), every field falls back to its
schema default when the section or key is omitted, and an unknown key under
`forecast.research` is fatal, naming the full dotted path. Per SPEC §6.1's
integer-units invariant, `ResearchSettings` carries only integer/string/tuple
leaves -- never a float.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from windbreak.config import ConfigError, load_config
from windbreak.config.schema import ForecastConfig, ResearchSettings

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any


# Fake environment-variable *names* (not credentials) exercised by the
# fixtures below. `search_api_key_env` names the env var an operator points at
# their real search-API key -- it never holds a secret. Binding these to
# constants (mirroring `tests/config/test_futuresearch_provider_config.py`)
# keeps the literals off the `search_api_key_env = "..."` assignment lines so
# detect-secrets' keyword heuristic cannot misread an env-var *name* as a
# hard-coded credential.
_CUSTOM_ENV_VAR = "CUSTOM_RESEARCH_KEY_ENV"
_DEFAULT_ENV_VAR = "RESEARCH_SEARCH_API_KEY"
_YAML_ENV_VAR = "MY_RESEARCH_SEARCH_KEY"


# --- ResearchSettings: construction, immutability, defaults ----------------------


def test_research_settings_construction_preserves_its_fields() -> None:
    """A fully-specified `ResearchSettings` preserves every field."""
    settings = ResearchSettings(
        search_endpoint_url="https://search.example/v1/search",
        search_api_key_env=_CUSTOM_ENV_VAR,
        allowed_research_hosts=("research.example", "news.example"),
        fetch_timeout_seconds=15,
        fetch_max_bytes=500_000,
        allowed_content_types=("text/html",),
    )

    assert settings.search_endpoint_url == "https://search.example/v1/search"
    assert settings.search_api_key_env == _CUSTOM_ENV_VAR
    assert settings.allowed_research_hosts == ("research.example", "news.example")
    assert settings.fetch_timeout_seconds == 15
    assert settings.fetch_max_bytes == 500_000
    assert settings.allowed_content_types == ("text/html",)


def test_research_settings_is_frozen() -> None:
    """Mutating a constructed `ResearchSettings` raises."""
    settings = ResearchSettings()

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        settings.search_endpoint_url = "https://elsewhere.example"  # type: ignore[misc]


def test_research_settings_defaults_are_fail_closed_operator_placeholders() -> None:
    """The bare no-arg default matches the "operator must configure this"
    idiom `AlertSink`/`ModelRef`/`FutureSearchProviderSettings` all use for a
    field with no safe real-world default, and fails *closed*:
    `allowed_research_hosts` defaults empty, so an unconfigured deployment's
    live-research egress allowlist contributes zero hosts rather than an
    invented, plausible-looking one.
    """
    settings = ResearchSettings()

    assert settings.search_endpoint_url == "configured-by-operator"
    assert settings.search_api_key_env == _DEFAULT_ENV_VAR
    assert settings.allowed_research_hosts == ()
    assert isinstance(settings.fetch_timeout_seconds, int)
    assert isinstance(settings.fetch_max_bytes, int)
    assert settings.fetch_timeout_seconds > 0
    assert settings.fetch_max_bytes > 0
    assert isinstance(settings.allowed_content_types, tuple)


def test_research_settings_leaves_are_never_float() -> None:
    """Every leaf on the default `ResearchSettings` is an int, str, or tuple
    of str -- never a `float` (SPEC §6.1's integer-units invariant).
    """
    settings = ResearchSettings()

    for field_def in dataclasses.fields(settings):
        value = getattr(settings, field_def.name)
        assert not isinstance(value, float)


def test_forecast_config_default_research_is_the_bare_settings_default() -> None:
    """`ForecastConfig()`'s default `research` is a bare
    `ResearchSettings()`.
    """
    assert ForecastConfig().research == ResearchSettings()


def test_forecast_config_default_triage_model_is_unchanged_by_this_addition() -> None:
    """Adding `ResearchSettings` does not touch the pre-existing SPEC §16
    `triage_model` default (`ModelRef("cheapest-adequate",
    "pinned-by-operator")`), verbatim.
    """
    from windbreak.config.schema import ModelRef

    assert ForecastConfig().triage_model == ModelRef(
        "cheapest-adequate", "pinned-by-operator"
    )


# --- load_config: YAML round-trips into a typed ResearchSettings -----------------


def test_load_config_parses_research_block_from_yaml(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """A config file naming `forecast.research` parses into a typed
    `ResearchSettings`, via the loader's existing generic dataclass coercion.
    """
    config_path = write_config(
        tmp_path,
        {
            "forecast": {
                "research": {
                    "search_endpoint_url": "https://search.example/v1/search",
                    "search_api_key_env": _YAML_ENV_VAR,
                    "allowed_research_hosts": ["research.example", "news.example"],
                    "fetch_timeout_seconds": 12,
                    "fetch_max_bytes": 750_000,
                    "allowed_content_types": ["text/html", "text/plain"],
                }
            }
        },
    )

    config = load_config(config_path)

    assert config.forecast.research == ResearchSettings(
        search_endpoint_url="https://search.example/v1/search",
        search_api_key_env=_YAML_ENV_VAR,
        allowed_research_hosts=("research.example", "news.example"),
        fetch_timeout_seconds=12,
        fetch_max_bytes=750_000,
        allowed_content_types=("text/html", "text/plain"),
    )
    assert isinstance(config.forecast.research, ResearchSettings)


def test_load_config_research_defaults_when_key_omitted(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """A config file that never mentions `forecast.research` at all -- not
    even an empty `forecast:` section -- still falls back to the built-in
    default settings.
    """
    config_path = write_config(tmp_path, {"mode_ceiling": "paper"})

    config = load_config(config_path)

    assert config.forecast.research == ResearchSettings()


def test_load_config_research_defaults_when_forecast_section_partial(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """A `forecast:` section present but silent on `research` still falls
    back to the built-in default for that one field, while other `forecast`
    keys in the same section still apply.
    """
    config_path = write_config(tmp_path, {"forecast": {"triage_threshold_ppm": 75_000}})

    config = load_config(config_path)

    assert config.forecast.research == ResearchSettings()
    assert config.forecast.triage_threshold_ppm == 75_000


def test_unknown_key_under_research_is_fatal(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """An unknown key under `forecast.research` is fatal, naming the full
    dotted path, exactly like every other config section.
    """
    config_path = write_config(tmp_path, {"forecast": {"research": {"bogus_key": 1}}})

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    message = str(excinfo.value)
    assert "forecast.research.bogus_key" in message
    assert "unknown keys are fatal per SPEC §16" in message
