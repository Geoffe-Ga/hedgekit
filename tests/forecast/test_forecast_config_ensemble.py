"""Tests for the `ForecastConfig.vote_ensemble` config seam (issue #184).

Pins two things: (1) `windbreak.config.schema.EnsembleMemberConfig` -- a new
frozen dataclass naming one vote-ensemble member's `provider` /
`model_version` / `training_cutoff` -- and (2) `ForecastConfig.vote_ensemble`,
a new `tuple[EnsembleMemberConfig, ...]` field that the loader parses
generically (via `windbreak.config.loader`'s existing dataclass/tuple
coercion, with no new loader code required) exactly like every other nested
tuple-of-dataclass field (e.g. `AlertsConfig.sinks`). Omitting the key from a
config file falls back to a built-in default of today's three pipeline
ensemble members, so no existing config file need change.

Placed under `tests/forecast/` rather than `tests/config/` to respect this
issue's scope fence (the config *seam* this issue adds is forecast-specific;
the loader itself is untouched).

`windbreak.config.schema` has no `EnsembleMemberConfig` yet, so importing it
below fails collection with `ImportError: cannot import name
'EnsembleMemberConfig' from 'windbreak.config.schema'` -- the expected Gate 1
RED state for issue #184.

Issue #191 repins `_DEFAULT_VOTE_ENSEMBLE` (and the production
`_default_vote_ensemble()`/`DEFAULT_VOTE_ENSEMBLE` it mirrors) from the
pre-#191 placeholder triple to the real, pinned three-provider triple
(`gpt-5-2025-08-07` / `claude-sonnet-4-5-20250929` / `gpt-5-mini-2025-08-07`).
Until #191 lands, `ForecastConfig()`'s actual `vote_ensemble` default is
still the old triple, so the tests below asserting equality with the new
`_DEFAULT_VOTE_ENSEMBLE` fail with an `AssertionError` naming the mismatched
tuple, not a collection error -- the expected Gate 1 RED state for issue
#191.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
import yaml

from windbreak.config import load_config
from windbreak.config.schema import EnsembleMemberConfig, ForecastConfig, ModelRef
from windbreak.forecast.providers import DEFAULT_VOTE_ENSEMBLE

if TYPE_CHECKING:
    from pathlib import Path

#: The #191 pinned three-provider vote-ensemble triple (mirroring
#: `windbreak.forecast.providers.DEFAULT_VOTE_ENSEMBLE`), which
#: `ForecastConfig.vote_ensemble` must default to.
_DEFAULT_VOTE_ENSEMBLE = (
    EnsembleMemberConfig("openai", "gpt-5-2025-08-07", "2024-09-30"),
    EnsembleMemberConfig("anthropic", "claude-sonnet-4-5-20250929", "2025-07-31"),
    EnsembleMemberConfig("openai", "gpt-5-mini-2025-08-07", "2024-05-31"),
)


def _write_yaml(tmp_path: Path, mapping: dict[str, object]) -> Path:
    """Dump `mapping` as YAML to a fresh `config.yaml` under `tmp_path`.

    Args:
        tmp_path: The directory to write the config file under.
        mapping: The raw configuration mapping to serialize.

    Returns:
        The path to the written YAML file.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return config_path


# --- EnsembleMemberConfig: construction and immutability -------------------------


def test_ensemble_member_config_construction_preserves_its_fields() -> None:
    """A valid `EnsembleMemberConfig` constructs and preserves every field."""
    member = EnsembleMemberConfig(
        provider="openai", model_version="gpt-5-forecast", training_cutoff="2024-06-01"
    )

    assert member.provider == "openai"
    assert member.model_version == "gpt-5-forecast"
    assert member.training_cutoff == "2024-06-01"


def test_ensemble_member_config_is_frozen() -> None:
    """Mutating a constructed `EnsembleMemberConfig` raises."""
    member = EnsembleMemberConfig("openai", "gpt-5-forecast", "2024-06-01")

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        member.provider = "anthropic"  # type: ignore[misc]


# --- ForecastConfig.vote_ensemble: built-in default -------------------------------


def test_forecast_config_default_vote_ensemble_is_the_three_pipeline_members() -> None:
    """`ForecastConfig()`'s default `vote_ensemble` is today's pinned three
    pipeline members, so wiring the config default into `collect_model_votes`
    changes nothing about the pre-#184 pipeline's behavior.
    """
    assert ForecastConfig().vote_ensemble == _DEFAULT_VOTE_ENSEMBLE


def test_default_vote_ensemble_mirrors_the_forecast_engine_default() -> None:
    """`windbreak.forecast.providers.DEFAULT_VOTE_ENSEMBLE` and
    `ForecastConfig()`'s default `vote_ensemble` stay mirror-equal in
    provenance (`provider`/`model_version`/`training_cutoff`), so wiring
    either into the vote stage yields identical ensemble provenance. Compared
    field-by-field (not via `==`) because `EnsembleMember` and
    `EnsembleMemberConfig` are distinct dataclasses across the SPEC S8.3
    sandbox boundary -- generated dataclass equality never holds across
    distinct types even when every field matches.
    """
    engine_provenance = tuple(
        (member.provider, member.model_version, member.training_cutoff)
        for member in DEFAULT_VOTE_ENSEMBLE
    )
    config_provenance = tuple(
        (member.provider, member.model_version, member.training_cutoff)
        for member in ForecastConfig().vote_ensemble
    )
    assert engine_provenance == config_provenance


def test_forecast_config_existing_ensemble_field_is_unchanged() -> None:
    """Adding `vote_ensemble` leaves the pre-existing `ensemble` field (the
    triage/promotion `ModelRef` ensemble) completely untouched.
    """
    assert ForecastConfig().ensemble == (
        ModelRef("anthropic", "pinned-by-operator"),
        ModelRef("openai", "pinned-by-operator"),
    )


# --- load_config: YAML parses into typed EnsembleMemberConfig tuples -------------


def test_load_config_parses_vote_ensemble_from_yaml(tmp_path: Path) -> None:
    """A config file naming `forecast.vote_ensemble` parses into typed
    `EnsembleMemberConfig` members, in file order, via the loader's existing
    generic dataclass/tuple coercion -- no bespoke loader code required.
    """
    config_path = _write_yaml(
        tmp_path,
        {
            "forecast": {
                "vote_ensemble": [
                    {
                        "provider": "openai",
                        "model_version": "custom-model-a",
                        "training_cutoff": "2025-01-01",
                    },
                    {
                        "provider": "anthropic",
                        "model_version": "custom-model-b",
                        "training_cutoff": "2025-02-01",
                    },
                ]
            }
        },
    )

    config = load_config(config_path)

    assert config.forecast.vote_ensemble == (
        EnsembleMemberConfig("openai", "custom-model-a", "2025-01-01"),
        EnsembleMemberConfig("anthropic", "custom-model-b", "2025-02-01"),
    )
    assert all(
        isinstance(member, EnsembleMemberConfig)
        for member in config.forecast.vote_ensemble
    )


def test_load_config_vote_ensemble_defaults_when_key_omitted(tmp_path: Path) -> None:
    """A config file that never mentions `forecast.vote_ensemble` at all --
    not even an empty `forecast:` section -- still falls back to the built-in
    three-member default.
    """
    config_path = _write_yaml(tmp_path, {"mode_ceiling": "paper"})

    config = load_config(config_path)

    assert config.forecast.vote_ensemble == _DEFAULT_VOTE_ENSEMBLE


def test_load_config_vote_ensemble_defaults_when_forecast_section_partial(
    tmp_path: Path,
) -> None:
    """A `forecast:` section present but silent on `vote_ensemble` still falls
    back to the built-in default for that one field, while other `forecast`
    keys in the same section still apply.
    """
    config_path = _write_yaml(tmp_path, {"forecast": {"triage_threshold_ppm": 75_000}})

    config = load_config(config_path)

    assert config.forecast.vote_ensemble == _DEFAULT_VOTE_ENSEMBLE
    assert config.forecast.triage_threshold_ppm == 75_000
