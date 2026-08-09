"""Tests for the `windbreak.config` loader (issue #11, SPEC S16).

Covers the full example loading correctly, partial configs falling
back to defaults, unknown keys being fatal at every nesting depth,
type/bool rejection, the one nullable field, the `bootstrap_confidence`
range check, and the three ways a config file itself can be invalid
(missing, malformed YAML, non-mapping root).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pytest

from windbreak.config import ConfigError, WindbreakConfig, load_config

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

#: The eight mapping-valued top-level sections (`mode_ceiling` is a
#: bare string scalar and has no nested keys to inject an unknown key
#: into, so it is exercised only via the top-level (`None`) case).
SECTION_NAMES = [
    "exchange",
    "capital",
    "risk",
    "screener",
    "forecast",
    "evaluation",
    "ops",
    "alerts",
]


def test_full_spec16_example_loads(spec16_path: Path) -> None:
    """Loading the full SPEC S16 example populates every section."""
    cfg = load_config(spec16_path)

    assert cfg.risk.require_human_ack_above_micros is None
    assert cfg.forecast.ensemble[1].provider == "openai"
    assert cfg.screener.horizon_days.max == 120
    assert cfg.alerts.sinks[0].type == "ntfy"
    assert cfg.capital.floor_micros == 1000000000
    assert cfg.exchange.provider == "kalshi"


def test_partial_config_fills_defaults(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """A config specifying only mode_ceiling still fills every other default."""
    defaults = WindbreakConfig()
    config_path = write_config(tmp_path, {"mode_ceiling": "paper"})

    cfg = load_config(config_path)

    assert cfg.mode_ceiling == "paper"
    assert cfg.exchange == defaults.exchange
    assert cfg.capital == defaults.capital
    assert cfg.risk == defaults.risk
    assert cfg.screener == defaults.screener
    assert cfg.forecast == defaults.forecast
    assert cfg.evaluation == defaults.evaluation
    assert cfg.ops == defaults.ops
    assert cfg.alerts == defaults.alerts


@pytest.mark.parametrize("section", [*SECTION_NAMES, None])
def test_unknown_key_fatal_per_section(
    section: str | None,
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """An unknown key anywhere in the schema is fatal, naming its full path."""
    mapping = copy.deepcopy(spec16_dict)
    if section is None:
        mapping["bogus_key"] = 1
        expected_path = "bogus_key"
    else:
        mapping[section]["bogus_key"] = 1
        expected_path = f"{section}.bogus_key"
    config_path = write_config(tmp_path, mapping)

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    message = str(excinfo.value)
    assert expected_path in message
    assert "unknown keys are fatal per SPEC §16" in message


def test_unknown_key_in_nested_mapping(
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """Unknown-key detection recurses into nested inline mappings."""
    horizon_mapping = copy.deepcopy(spec16_dict)
    horizon_mapping["screener"]["horizon_days"]["bogus_key"] = 1
    horizon_path = write_config(tmp_path, horizon_mapping)

    with pytest.raises(ConfigError) as horizon_excinfo:
        load_config(horizon_path)

    assert "screener.horizon_days.bogus_key" in str(horizon_excinfo.value)

    budget_mapping = copy.deepcopy(spec16_dict)
    budget_mapping["forecast"]["budget"]["bogus_key"] = 1
    budget_path = write_config(tmp_path, budget_mapping)

    with pytest.raises(ConfigError) as budget_excinfo:
        load_config(budget_path)

    assert "forecast.budget.bogus_key" in str(budget_excinfo.value)


def test_scalar_where_tuple_expected_rejected(
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """A scalar where a list/tuple field is expected is rejected, not iterated."""
    mapping = copy.deepcopy(spec16_dict)
    mapping["exchange"]["product_allowlist"] = "predictions"
    config_path = write_config(tmp_path, mapping)

    with pytest.raises(ConfigError, match=r"exchange\.product_allowlist"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("section", "key", "bad_value", "expected_match"),
    [
        ("screener", "horizon_days", 5, r"screener\.horizon_days"),
        (None, "exchange", "kalshi", r"^exchange:"),
    ],
    ids=["nested-section", "top-level-section"],
)
def test_non_mapping_where_section_expected_rejected(
    section: str | None,
    key: str,
    bad_value: object,
    expected_match: str,
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """A scalar where a nested-dataclass section is expected is rejected."""
    mapping = copy.deepcopy(spec16_dict)
    if section is None:
        mapping[key] = bad_value
    else:
        mapping[section][key] = bad_value
    config_path = write_config(tmp_path, mapping)

    with pytest.raises(ConfigError, match=expected_match):
        load_config(config_path)


def test_unknown_key_inside_list_item_dataclass_rejected(
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """An unknown key inside a list-of-dataclass element is fatal by index."""
    mapping = copy.deepcopy(spec16_dict)
    mapping["forecast"]["ensemble"][0]["bogus_key"] = 1
    config_path = write_config(tmp_path, mapping)

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    message = str(excinfo.value)
    assert "forecast.ensemble.0.bogus_key" in message
    assert "unknown keys are fatal per SPEC §16" in message


def test_wrong_type_int_field_rejected(
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """A string where an int is expected is rejected, naming the field."""
    mapping = copy.deepcopy(spec16_dict)
    mapping["risk"]["max_orders_per_hour"] = "twenty"
    config_path = write_config(tmp_path, mapping)

    with pytest.raises(ConfigError, match=r"risk\.max_orders_per_hour"):
        load_config(config_path)


def test_bool_rejected_where_int_expected(
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """A bool is rejected where an int is expected, despite subclassing int."""
    mapping = copy.deepcopy(spec16_dict)
    mapping["risk"]["max_orders_per_hour"] = True
    config_path = write_config(tmp_path, mapping)

    with pytest.raises(ConfigError, match=r"risk\.max_orders_per_hour"):
        load_config(config_path)


def test_nullable_field_accepts_null_and_int(
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """require_human_ack_above_micros accepts null or an int, not other types."""
    null_mapping = copy.deepcopy(spec16_dict)
    null_mapping["risk"]["require_human_ack_above_micros"] = None
    null_path = write_config(tmp_path, null_mapping)
    assert load_config(null_path).risk.require_human_ack_above_micros is None

    int_mapping = copy.deepcopy(spec16_dict)
    int_mapping["risk"]["require_human_ack_above_micros"] = 123
    int_path = write_config(tmp_path, int_mapping)
    assert load_config(int_path).risk.require_human_ack_above_micros == 123

    bad_mapping = copy.deepcopy(spec16_dict)
    bad_mapping["risk"]["require_human_ack_above_micros"] = "soon"
    bad_path = write_config(tmp_path, bad_mapping)
    with pytest.raises(ConfigError, match=r"risk\.require_human_ack_above_micros"):
        load_config(bad_path)


@pytest.mark.parametrize("bad_value", [0.9555555, "high", 1.5])
def test_bootstrap_confidence_rejects_unrepresentable(
    bad_value: object,
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """Values that cannot map cleanly to a ppm int are rejected."""
    mapping = copy.deepcopy(spec16_dict)
    mapping["evaluation"]["bootstrap_confidence"] = bad_value
    config_path = write_config(tmp_path, mapping)

    with pytest.raises(ConfigError, match=r"evaluation\.bootstrap_confidence"):
        load_config(config_path)


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_bootstrap_confidence_rejects_non_finite(
    bad_value: float,
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """A non-finite bootstrap_confidence raises ConfigError, not OverflowError.

    YAML's safe loader accepts ``.inf``/``-.inf``/``.nan`` as float literals;
    without a finiteness guard, ``.inf`` slips past the exact-ppm check and
    crashes ``int(Decimal("Infinity"))`` with an uncaught ``OverflowError``.
    """
    mapping = copy.deepcopy(spec16_dict)
    mapping["evaluation"]["bootstrap_confidence"] = bad_value
    config_path = write_config(tmp_path, mapping)

    with pytest.raises(ConfigError, match=r"evaluation\.bootstrap_confidence"):
        load_config(config_path)


def test_bootstrap_confidence_rejects_negative(
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """A negative bootstrap_confidence is rejected by the lower-bound check."""
    mapping = copy.deepcopy(spec16_dict)
    mapping["evaluation"]["bootstrap_confidence"] = -0.1
    config_path = write_config(tmp_path, mapping)

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    message = str(excinfo.value)
    assert "evaluation.bootstrap_confidence" in message
    assert "must be between 0 and 1" in message


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    """Loading a nonexistent path raises ConfigError, not a bare OSError."""
    missing_path = tmp_path / "does-not-exist.yaml"

    with pytest.raises(ConfigError):
        load_config(missing_path)


def test_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    """Invalid YAML syntax raises ConfigError, not a bare yaml.YAMLError."""
    bad_path = tmp_path / "malformed.yaml"
    bad_path.write_text(":\n  - unbalanced", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(bad_path)


def test_non_mapping_root_raises_config_error(tmp_path: Path) -> None:
    """A YAML document whose root is a list, not a mapping, is rejected."""
    list_root_path = tmp_path / "list_root.yaml"
    list_root_path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(list_root_path)


def test_invalid_utf8_file_raises_config_error(tmp_path: Path) -> None:
    """A file with invalid UTF-8 bytes raises ConfigError, not UnicodeDecodeError.

    ``UnicodeDecodeError`` is a ``ValueError`` subclass, not an ``OSError``, so
    decoding failures must be caught explicitly to honour the module invariant
    that no raw traceback escapes to the operator.
    """
    bad_path = tmp_path / "invalid_utf8.yaml"
    bad_path.write_bytes(b"\xff\xfe invalid utf-8 bytes")

    with pytest.raises(ConfigError):
        load_config(bad_path)


def test_dashboard_port_loads_from_yaml(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """YAML `dashboard: {port: 9090}` loads to `config.dashboard.port == 9090`.

    The `dashboard` schema field does not exist yet, so today this fails
    closed with `ConfigError: unknown configuration key(s): dashboard` --
    the top-level `dashboard` section itself is unrecognized -- rather than
    successfully loading 9090 (issue #79).
    """
    config_path = write_config(tmp_path, {"dashboard": {"port": 9090}})

    cfg = load_config(config_path)

    assert cfg.dashboard.port == 9090


def test_dashboard_host_key_is_unknown_and_fatal(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """`dashboard: {host: ...}` is an unknown key, naming `dashboard.host`.

    The dashboard's bind host is never configurable (SPEC §14: loopback-only);
    this unknown-key rejection is the structural guarantee that pins it. Today
    the loader doesn't even recognize `dashboard` as a valid top-level section,
    so the raised `ConfigError` names the bare `dashboard` key (not the nested
    `dashboard.host` path this test expects) -- a legitimate RED mismatch, not
    a typo (issue #79).
    """
    config_path = write_config(tmp_path, {"dashboard": {"host": "0.0.0.0"}})

    with pytest.raises(ConfigError, match=r"dashboard\.host"):
        load_config(config_path)


def test_dashboard_non_int_port_is_a_config_error(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """A non-int `dashboard.port` value is rejected, naming `dashboard.port`."""
    config_path = write_config(tmp_path, {"dashboard": {"port": "not-an-int"}})

    with pytest.raises(ConfigError, match=r"dashboard\.port"):
        load_config(config_path)


def test_unknown_key_in_forecast_provider_gate_is_fatal(
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """An unknown key under `forecast.provider_gate` is fatal (issue #194).

    `ForecastConfig` has no `provider_gate` field yet, so today the whole
    `forecast.provider_gate` mapping itself is the unrecognized key, and the
    raised `ConfigError` names the bare `forecast.provider_gate` path -- not
    the nested `forecast.provider_gate.bogus_key` this test expects once the
    schema gains the field. A legitimate RED mismatch, not a typo (mirrors
    `test_dashboard_host_key_is_unknown_and_fatal`'s issue #79 precedent).
    """
    mapping = copy.deepcopy(spec16_dict)
    mapping["forecast"]["provider_gate"] = {
        "min_resolved": 150,
        "min_brier_skill_ppm": 10000,
        "bogus_key": 1,
    }
    config_path = write_config(tmp_path, mapping)

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    message = str(excinfo.value)
    assert "forecast.provider_gate.bogus_key" in message
    assert "unknown keys are fatal per SPEC §16" in message


def test_verification_tolerance_fields_load_from_yaml(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """YAML `risk.verification_balance_tolerance_micros` /
    `.verification_position_tolerance_centis` load onto `RiskConfig` (issue
    #236): `_build_risk_kernel` reads these two fields to compose the live
    verifier's `VerificationTolerances`.

    `RiskConfig` has neither field yet, so today both are unrecognized keys
    and `load_config` raises `ConfigError: unknown configuration key(s):
    risk.verification_balance_tolerance_micros, ...` rather than loading the
    values -- the expected Gate 1 RED state for issue #236.
    """
    config_path = write_config(
        tmp_path,
        {
            "risk": {
                "verification_balance_tolerance_micros": 500,
                "verification_position_tolerance_centis": 3,
            }
        },
    )

    cfg = load_config(config_path)

    assert cfg.risk.verification_balance_tolerance_micros == 500
    assert cfg.risk.verification_position_tolerance_centis == 3


def test_forecast_provider_gate_override_round_trips(
    spec16_dict: dict[str, Any],
    tmp_path: Path,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """An explicit `forecast.provider_gate` override loads and round-trips.

    `ForecastConfig` has no `provider_gate` field yet, so today this raises
    `ConfigError` for the still-unrecognized `forecast.provider_gate` key
    (issue #194) rather than successfully loading the override.
    """
    mapping = copy.deepcopy(spec16_dict)
    mapping["forecast"]["provider_gate"] = {
        "min_resolved": 200,
        "min_brier_skill_ppm": 15000,
    }
    config_path = write_config(tmp_path, mapping)

    cfg = load_config(config_path)

    assert cfg.forecast.provider_gate.min_resolved == 200
    assert cfg.forecast.provider_gate.min_brier_skill_ppm == 15000
