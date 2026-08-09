"""Tests for `windbreak.config` hashing and diffing (issue #11, SPEC S16).

Every configuration version must be ledgerable via a stable,
key-order-independent SHA-256 hash and a path-level diff against the
previous version, plus a human-readable rendering of that diff.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

import yaml

from windbreak.config import (
    ModelRef,
    WindbreakConfig,
    config_hash,
    diff_configs,
    flatten,
    format_diff,
    load_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_config_hash_is_sha256_hex() -> None:
    """config_hash returns a 64-character lowercase hex SHA-256 digest."""
    digest = config_hash(WindbreakConfig())

    assert len(digest) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_hash_stable_across_yaml_key_order(
    spec16_dict: dict[str, object], tmp_path: Path
) -> None:
    """Reordering top-level YAML keys does not change the resulting hash."""
    sorted_path = tmp_path / "sorted.yaml"
    sorted_path.write_text(
        yaml.safe_dump(spec16_dict, sort_keys=True), encoding="utf-8"
    )

    reversed_mapping = dict(reversed(list(spec16_dict.items())))
    reversed_path = tmp_path / "reversed.yaml"
    reversed_path.write_text(
        yaml.safe_dump(reversed_mapping, sort_keys=False), encoding="utf-8"
    )

    sorted_hash = config_hash(load_config(sorted_path))
    reversed_hash = config_hash(load_config(reversed_path))
    assert sorted_hash == reversed_hash


def test_hash_changes_when_value_changes() -> None:
    """Changing a single leaf value changes the config hash."""
    default_cfg = WindbreakConfig()
    changed_risk = dataclasses.replace(default_cfg.risk, min_net_edge_ppm=40000)
    changed_cfg = dataclasses.replace(default_cfg, risk=changed_risk)

    assert config_hash(changed_cfg) != config_hash(default_cfg)


def test_diff_reports_exactly_changed_paths() -> None:
    """diff_configs reports exactly the paths that changed, with old/new values."""
    old_cfg = WindbreakConfig()
    new_risk = dataclasses.replace(old_cfg.risk, min_net_edge_ppm=40000)
    new_ops = dataclasses.replace(
        old_cfg.ops, min_free_disk_mb=old_cfg.ops.min_free_disk_mb + 1
    )
    new_cfg = dataclasses.replace(old_cfg, risk=new_risk, ops=new_ops)

    diff = diff_configs(old_cfg, new_cfg)

    assert set(diff.changed) == {"risk.min_net_edge_ppm", "ops.min_free_disk_mb"}
    assert diff.changed["risk.min_net_edge_ppm"] == (30000, 40000)
    assert diff.changed["ops.min_free_disk_mb"] == (
        old_cfg.ops.min_free_disk_mb,
        old_cfg.ops.min_free_disk_mb + 1,
    )
    assert not diff.added
    assert not diff.removed


def test_diff_reports_added_removed_list_paths() -> None:
    """diff_configs reports appended/removed tuple entries as indexed paths."""
    old_cfg = WindbreakConfig()
    extra_model = ModelRef(provider="mistral", model="pinned-by-operator")
    new_ensemble = (*old_cfg.forecast.ensemble, extra_model)
    new_forecast = dataclasses.replace(old_cfg.forecast, ensemble=new_ensemble)
    new_cfg = dataclasses.replace(old_cfg, forecast=new_forecast)

    added_diff = diff_configs(old_cfg, new_cfg)
    assert set(added_diff.added) == {
        "forecast.ensemble.2.provider",
        "forecast.ensemble.2.model",
    }

    removed_diff = diff_configs(new_cfg, old_cfg)
    assert set(removed_diff.removed) == {
        "forecast.ensemble.2.provider",
        "forecast.ensemble.2.model",
    }


def test_diff_identical_configs_is_empty() -> None:
    """Diffing a config against itself yields an empty, no-op diff."""
    cfg = WindbreakConfig()

    diff = diff_configs(cfg, cfg)

    assert diff.is_empty
    assert not diff.added
    assert not diff.removed
    assert not diff.changed


def test_flatten_pins_ensemble_and_vote_ensemble_as_distinct_dotted_paths() -> None:
    """Decision pin (issue #240): ``forecast.ensemble`` (legacy SPEC S16
    triage/promotion set) and ``forecast.vote_ensemble`` (vote-stage
    per-member provenance, issues #184/#191) are two distinct fields, neither
    a rename of the other -- both dotted leaf paths must appear in the
    flattened config, protecting config-hash stability against a future
    rename/merge of the two fields.
    """
    flat = flatten(WindbreakConfig())

    assert "forecast.ensemble.0.provider" in flat
    assert "forecast.vote_ensemble.0.provider" in flat


def test_format_diff_human_readable() -> None:
    """format_diff renders a readable line per changed path, or a no-op marker."""
    old_cfg = WindbreakConfig()
    new_risk = dataclasses.replace(old_cfg.risk, min_net_edge_ppm=40000)
    new_cfg = dataclasses.replace(old_cfg, risk=new_risk)

    changed_text = format_diff(diff_configs(old_cfg, new_cfg))
    assert "~ risk.min_net_edge_ppm: 30000 -> 40000" in changed_text

    empty_text = format_diff(diff_configs(old_cfg, old_cfg))
    assert "(no changes)" in empty_text
