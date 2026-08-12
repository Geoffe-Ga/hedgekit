"""Tests for `ForecastConfig.futuresearch` (issue #189, the FutureSearch config seam).

Pins `windbreak.config.schema.FutureSearchProviderSettings` -- the config-schema
mirror of
`windbreak.forecast.providers.futuresearch.FutureSearchProviderConfig` -- and
its `forecast.futuresearch` YAML round-trip through `load_config`: every field
parses from YAML via the loader's existing generic dataclass coercion (no
bespoke loader code required, exactly like every other nested-dataclass config
section), every field falls back to the "operator must fill this in"
placeholder idiom when the section or key is omitted, and an unknown key under
`forecast.futuresearch` is fatal, naming the full dotted path, exactly like
every other config section.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from windbreak.config import ConfigError, load_config
from windbreak.config.schema import ForecastConfig, FutureSearchProviderSettings

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any


# Fake environment-variable *names* (not credentials) exercised by the fixtures
# below. `api_key_env` names the env var an operator points at their real key --
# it never holds a secret. Binding these to constants keeps the literals off the
# `api_key_env = "..."` assignment lines so detect-secrets' keyword heuristic
# cannot misread an env-var name as a hard-coded credential.
_CUSTOM_ENV_VAR = "CUSTOM_KEY_ENV"
_DEFAULT_ENV_VAR = "FUTURESEARCH_API_KEY"
_YAML_ENV_VAR = "MY_FUTURESEARCH_KEY"


# --- FutureSearchProviderSettings: construction, immutability, defaults ----------


def test_futuresearch_provider_settings_construction_preserves_its_fields() -> None:
    """A fully-specified `FutureSearchProviderSettings` preserves every field."""
    settings = FutureSearchProviderSettings(
        endpoint_url="https://futuresearch.example/v1/forecast",
        pinned_forecaster_versions=("futuresearch-v1", "futuresearch-v2"),
        api_key_env=_CUSTOM_ENV_VAR,
        per_call_ceiling_micros=1_000_000,
        reject_on_version_drift=False,
    )

    assert settings.endpoint_url == "https://futuresearch.example/v1/forecast"
    assert settings.pinned_forecaster_versions == (
        "futuresearch-v1",
        "futuresearch-v2",
    )
    assert settings.api_key_env == _CUSTOM_ENV_VAR
    assert settings.per_call_ceiling_micros == 1_000_000
    assert settings.reject_on_version_drift is False


def test_futuresearch_provider_settings_is_frozen() -> None:
    """Mutating a constructed `FutureSearchProviderSettings` raises."""
    settings = FutureSearchProviderSettings()

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        settings.endpoint_url = "https://elsewhere.example"  # type: ignore[misc]


def test_futuresearch_provider_settings_defaults_are_operator_placeholders() -> None:
    """The bare no-arg default matches the documented operator-fill-in idiom,
    mirroring `AlertSink`/`ModelRef`'s "operator must configure this" fields
    elsewhere in the schema.
    """
    settings = FutureSearchProviderSettings()

    assert settings.endpoint_url == "configured-by-operator"
    assert settings.pinned_forecaster_versions == ("pinned-by-operator",)
    assert settings.api_key_env == _DEFAULT_ENV_VAR
    assert settings.per_call_ceiling_micros == 2_000_000
    assert settings.reject_on_version_drift is True


def test_forecast_config_default_futuresearch_is_the_bare_settings_default() -> None:
    """`ForecastConfig()`'s default `futuresearch` is a bare
    `FutureSearchProviderSettings()`.
    """
    assert ForecastConfig().futuresearch == FutureSearchProviderSettings()


# --- load_config: YAML round-trips into a typed FutureSearchProviderSettings -----


def test_load_config_parses_futuresearch_block_from_yaml(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """A config file naming `forecast.futuresearch` parses into a typed
    `FutureSearchProviderSettings`, via the loader's existing generic
    dataclass coercion -- no bespoke loader code required.
    """
    config_path = write_config(
        tmp_path,
        {
            "forecast": {
                "futuresearch": {
                    "endpoint_url": "https://futuresearch.example/v1/forecast",
                    "pinned_forecaster_versions": ["futuresearch-v1"],
                    "api_key_env": _YAML_ENV_VAR,
                    "per_call_ceiling_micros": 1_500_000,
                    "reject_on_version_drift": False,
                }
            }
        },
    )

    config = load_config(config_path)

    assert config.forecast.futuresearch == FutureSearchProviderSettings(
        endpoint_url="https://futuresearch.example/v1/forecast",
        pinned_forecaster_versions=("futuresearch-v1",),
        api_key_env=_YAML_ENV_VAR,
        per_call_ceiling_micros=1_500_000,
        reject_on_version_drift=False,
    )
    assert isinstance(config.forecast.futuresearch, FutureSearchProviderSettings)


def test_load_config_futuresearch_defaults_when_key_omitted(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """A config file that never mentions `forecast.futuresearch` at all --
    not even an empty `forecast:` section -- still falls back to the built-in
    default settings.
    """
    config_path = write_config(tmp_path, {"mode_ceiling": "paper"})

    config = load_config(config_path)

    assert config.forecast.futuresearch == FutureSearchProviderSettings()


def test_load_config_futuresearch_defaults_when_forecast_section_partial(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """A `forecast:` section present but silent on `futuresearch` still falls
    back to the built-in default for that one field, while other `forecast`
    keys in the same section still apply.
    """
    config_path = write_config(tmp_path, {"forecast": {"triage_threshold_ppm": 75_000}})

    config = load_config(config_path)

    assert config.forecast.futuresearch == FutureSearchProviderSettings()
    assert config.forecast.triage_threshold_ppm == 75_000


def test_unknown_key_under_futuresearch_is_fatal(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """An unknown key under `forecast.futuresearch` is fatal, naming the full
    dotted path, exactly like every other config section.
    """
    config_path = write_config(
        tmp_path, {"forecast": {"futuresearch": {"bogus_key": 1}}}
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    message = str(excinfo.value)
    assert "forecast.futuresearch.bogus_key" in message
    assert "unknown keys are fatal per SPEC §16" in message


# --- The section names a variable; it never holds a key (issue #555) -----------


def test_the_api_key_leaf_names_an_environment_variable_not_a_secret() -> None:
    """The key leaf is an env-var *name*, matching the repo's ``*_api_key_env``.

    Asserted by shape rather than against a literal: an upper-snake-case
    identifier is what an environment variable name looks like, and no real
    credential does. Since issue #555 this section is actually consumed by the
    composition root, so the rule now has teeth it did not have while the
    section was inert -- a value here would be flattened verbatim into the
    hash-chained ``ConfigLoaded`` event and be unremovable from the audit trail.
    """
    variable = FutureSearchProviderSettings().api_key_env

    assert variable.isupper()
    assert variable.replace("_", "").isalnum()
    assert "FUTURESEARCH" in variable


def test_no_futuresearch_leaf_is_spelled_as_a_bare_secret() -> None:
    """No leaf name suggests it holds a key itself rather than naming its var."""
    suspicious = [
        field.name
        for field in dataclasses.fields(FutureSearchProviderSettings)
        if ("key" in field.name or "token" in field.name or "secret" in field.name)
        and not field.name.endswith("_env")
    ]

    assert suspicious == []


def test_only_the_cli_composition_root_reads_the_key_variables_value() -> None:
    """``os.environ`` is reachable from ``windbreak.main`` alone on this path.

    The rule the alert sinks and the LLM transports already follow, restated as
    a corpus scan so a later edit cannot quietly move the read into the
    scheduler or the forecast engine, where the value would sit beside the
    ledger writer. The scan is anchored on a module that *does* match, so it
    cannot pass by finding nothing.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "windbreak"
    pattern = re.compile(r"\bapi_key_env\b")
    sources = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
    }
    mentions = {name for name, text in sources.items() if pattern.search(text)}
    readers = {name for name in mentions if "os.environ" in sources[name]}

    assert mentions, "the corpus scan found no `api_key_env` at all"
    assert readers == {"main.py"}
