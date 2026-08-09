"""Tests for the `windbreak.config` schema (issue #11, SPEC S16).

Pins the shape of `WindbreakConfig`: its defaults are the SPEC S16
example verbatim, it is immutable, it holds no float leaves (SPEC
S6.1: integer units only), and the lone float in the SPEC S16 YAML
(`bootstrap_confidence`) loads as an integer parts-per-million field.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from windbreak.config import WindbreakConfig, load_config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _leaves(value: object) -> Iterator[object]:
    """Yield every non-container leaf value reachable from `value`.

    Recurses through dataclass instances (via their declared fields),
    dict values, and list/tuple elements, so it walks the full config
    tree regardless of nesting depth.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from _leaves(getattr(value, f.name))
    elif isinstance(value, dict):
        for v in value.values():
            yield from _leaves(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _leaves(v)
    else:
        yield value


def test_default_config_matches_spec16_example(spec16_path: Path) -> None:
    """WindbreakConfig()'s defaults ARE the SPEC S16 example, verbatim."""
    assert WindbreakConfig() == load_config(spec16_path)


def test_config_is_immutable() -> None:
    """Mutating a nested config field raises FrozenInstanceError."""
    cfg = WindbreakConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.risk.min_net_edge_ppm = 1


def test_no_float_fields_anywhere() -> None:
    """No config value is ever a Python float (SPEC S6.1: integers only)."""
    cfg = WindbreakConfig()

    leaves = list(_leaves(cfg))

    assert leaves
    assert not any(isinstance(leaf, float) for leaf in leaves)


def test_bootstrap_confidence_maps_to_ppm_int(spec16_path: Path) -> None:
    """The lone float in SPEC S16 loads as an integer ppm field."""
    cfg = load_config(spec16_path)

    assert cfg.evaluation.bootstrap_confidence_ppm == 950000
    assert isinstance(cfg.evaluation.bootstrap_confidence_ppm, int)
    assert not isinstance(cfg.evaluation.bootstrap_confidence_ppm, bool)


def test_default_config_dashboard_port_is_8080() -> None:
    """`WindbreakConfig().dashboard.port` defaults to 8080 (issue #79).

    Matches the reserved `127.0.0.1:8080` compose publish (SPEC §14: the
    dashboard's host is never configurable, only its port). `DashboardConfig`
    does not exist yet, so this fails with `ImportError` at the local import
    below -- scoped to this one test so every other test in this file keeps
    collecting and passing.
    """
    from windbreak.config.schema import DashboardConfig

    cfg = WindbreakConfig()

    assert isinstance(cfg.dashboard, DashboardConfig)
    assert cfg.dashboard.port == 8080


def test_dashboard_config_is_immutable() -> None:
    """`DashboardConfig`, like every other config section, is frozen."""
    from windbreak.config.schema import DashboardConfig

    section = DashboardConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        section.port = 1


def test_provider_gate_config_defaults_match_evaluation_thresholds() -> None:
    """`ProviderGateConfig`'s defaults (150 resolved / 10000 ppm skill) match
    `EvaluationConfig`'s own promotion thresholds -- the same statistical bar
    applied to a per-provider live-eligibility gate (issue #194).
    `ProviderGateConfig` does not exist yet, so this fails with `ImportError`
    at the local import below -- scoped to this one test so every other test
    in this file keeps collecting and passing.
    """
    from windbreak.config.schema import ProviderGateConfig

    section = ProviderGateConfig()
    defaults = WindbreakConfig()

    assert section.min_resolved == 150
    assert section.min_brier_skill_ppm == 10000
    assert section.min_resolved == defaults.evaluation.min_resolved_for_calibration
    assert section.min_brier_skill_ppm == defaults.evaluation.brier_skill_required_ppm


def test_provider_gate_config_is_immutable() -> None:
    """`ProviderGateConfig`, like every other config section, is frozen."""
    from windbreak.config.schema import ProviderGateConfig

    section = ProviderGateConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        section.min_resolved = 1


def test_forecast_config_provider_gate_defaults_to_provider_gate_config() -> None:
    """`ForecastConfig.provider_gate` defaults to a fresh `ProviderGateConfig()`."""
    from windbreak.config.schema import ProviderGateConfig

    cfg = WindbreakConfig()

    assert isinstance(cfg.forecast.provider_gate, ProviderGateConfig)
    assert cfg.forecast.provider_gate == ProviderGateConfig()


def test_risk_config_verification_tolerance_fields_default_to_zero() -> None:
    """`RiskConfig.verification_balance_tolerance_micros` and
    `.verification_position_tolerance_centis` both default to `0` -- a
    fail-closed exact-match default, matching `_build_risk_kernel`'s
    zero-tolerance composition when a config leaves them unset (issue #236).

    Neither field exists on `RiskConfig` yet, so this fails with
    `AttributeError: 'RiskConfig' object has no attribute
    'verification_balance_tolerance_micros'` -- the expected Gate 1 RED state
    for issue #236.
    """
    cfg = WindbreakConfig()

    assert cfg.risk.verification_balance_tolerance_micros == 0
    assert cfg.risk.verification_position_tolerance_centis == 0
