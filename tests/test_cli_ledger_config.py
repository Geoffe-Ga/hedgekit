"""Tests wiring `windbreak run --ledger-path` to a real ConfigLoaded row (issue #74).

Before issue #74, `--ledger-path` alone (i.e. supplied without the other
three PAPER-loop flags) created no ledger at all -- ledgering was reserved
for the fully-activated PAPER loop (issue #48). This module pins the new
contract: any successful config load, with or without `--config`, appends
exactly one `ConfigLoaded` event to the ledger at `--ledger-path` as the
first record (`sequence_number == 1`), before any PAPER events. A failed
`--config` load stays fail-closed: the store is never opened, so no ledger
file is ever created.

Placed flat under `tests/` (matching the sibling `test_cli_config.py`)
rather than under `tests/config/`, whose `conftest.py` fixtures are scoped
to that subtree only; this module reaches `tests/config/spec16_example.yaml`
directly by path, following that sibling's documented convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from windbreak.config import config_hash, load_config, load_default_config
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.main import main

if TYPE_CHECKING:
    import pytest

#: The checked-in SPEC S16 example, shared with tests/config/conftest.py.
_SPEC16_PATH = Path(__file__).parent / "config" / "spec16_example.yaml"

#: The event_type discriminators the PAPER loop ledgers (issue #48); none of
#: these may appear when only `--ledger-path` is supplied without the other
#: three PAPER flags.
_PAPER_EVENT_TYPES = frozenset(
    {
        "MarketSnapshotRecorded",
        "EquitySampled",
        "SelectorDecisionRecorded",
        "ForecastCreated",
    }
)


def test_run_with_config_and_ledger_path_ledgers_one_config_loaded_at_seq_1(
    tmp_path: Path,
) -> None:
    """`run --config X --ledger-path L` appends one ConfigLoaded as record 1."""
    ledger_path = tmp_path / "ledger.db"
    expected_hash = config_hash(load_config(_SPEC16_PATH))

    exit_code = main(
        [
            "run",
            "--config",
            str(_SPEC16_PATH),
            "--ledger-path",
            str(ledger_path),
            "--heartbeat-interval",
            "0",
            "--max-beats",
            "1",
        ]
    )

    assert exit_code == 0
    store = SqliteLedgerStore(ledger_path)
    store.verify_chain()
    records = store.read_all()
    store.close()
    assert len(records) == 1
    record = records[0]
    assert record.sequence_number == 1
    assert record.event_type == "ConfigLoaded"
    assert record.component == "pipeline"
    data = json.loads(record.payload_json)["data"]
    assert data["config_hash"] == expected_hash
    assert {r.event_type for r in records}.isdisjoint(_PAPER_EVENT_TYPES)


def test_run_with_config_and_process_riskkernel_ledgers_that_component(
    tmp_path: Path,
) -> None:
    """`--process riskkernel` stamps the ledgered ConfigLoaded's component.

    Since issue #235 the live kernel persists its own events (heartbeats,
    verdicts) to the same ledger through a real ``KernelLedgerWriter``, so the
    ConfigLoaded row is no longer the ledger's only record. It stays the first
    row -- config is ledgered before the kernel loop is entered -- and is the
    row whose ``component`` this test pins; the trailing kernel rows are the
    #235 persistence behavior, not a regression.
    """
    ledger_path = tmp_path / "ledger.db"

    exit_code = main(
        [
            "run",
            "--config",
            str(_SPEC16_PATH),
            "--process",
            "riskkernel",
            "--ledger-path",
            str(ledger_path),
            "--heartbeat-interval",
            "0",
            "--max-beats",
            "1",
        ]
    )

    assert exit_code == 0
    store = SqliteLedgerStore(ledger_path)
    records = store.read_all()
    store.close()
    assert records[0].event_type == "ConfigLoaded"
    assert records[0].component == "riskkernel"
    # #235: the kernel now persists a ModeHeartbeat to the same hash chain.
    assert any(record.event_type == "ModeHeartbeat" for record in records[1:])


def test_run_without_config_ledgers_default_config_with_empty_diff(
    tmp_path: Path,
) -> None:
    """No `--config` still ledgers one ConfigLoaded, diffed against itself."""
    ledger_path = tmp_path / "ledger.db"
    expected_hash = config_hash(load_default_config())

    exit_code = main(
        [
            "run",
            "--ledger-path",
            str(ledger_path),
            "--heartbeat-interval",
            "0",
            "--max-beats",
            "1",
        ]
    )

    assert exit_code == 0
    store = SqliteLedgerStore(ledger_path)
    records = store.read_all()
    store.close()
    assert len(records) == 1
    data = json.loads(records[0].payload_json)["data"]
    assert data["config_hash"] == expected_hash
    assert data["diff"] == {"added": {}, "removed": {}, "changed": {}}


def test_rebuild_after_ledgered_run_is_deterministic_across_runs(
    tmp_path: Path,
) -> None:
    """`rebuild` folds the ledgered ConfigLoaded into a byte-identical file."""
    ledger_path = tmp_path / "ledger.db"
    expected_hash = config_hash(load_config(_SPEC16_PATH))
    exit_code = main(
        [
            "run",
            "--config",
            str(_SPEC16_PATH),
            "--ledger-path",
            str(ledger_path),
            "--heartbeat-interval",
            "0",
            "--max-beats",
            "1",
        ]
    )
    assert exit_code == 0

    output_dir_1 = tmp_path / "read_models_1"
    output_dir_2 = tmp_path / "read_models_2"
    rebuild_exit_1 = main(
        [
            "rebuild",
            "--ledger-path",
            str(ledger_path),
            "--output-dir",
            str(output_dir_1),
        ]
    )
    rebuild_exit_2 = main(
        [
            "rebuild",
            "--ledger-path",
            str(ledger_path),
            "--output-dir",
            str(output_dir_2),
        ]
    )

    assert rebuild_exit_1 == 0
    assert rebuild_exit_2 == 0
    config_versions_1 = output_dir_1 / "config_versions.json"
    config_versions_2 = output_dir_2 / "config_versions.json"
    entries = json.loads(config_versions_1.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["config_hash"] == expected_hash
    assert config_versions_1.read_bytes() == config_versions_2.read_bytes()


def test_run_with_missing_config_and_ledger_path_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fatal `--config` error never opens the ledger: fail-closed, no DB file."""
    missing_path = tmp_path / "nope.yaml"
    ledger_path = tmp_path / "ledger.db"

    exit_code = main(
        [
            "run",
            "--config",
            str(missing_path),
            "--ledger-path",
            str(ledger_path),
            "--max-beats",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FATAL:" in captured.err
    assert "seq=" not in captured.err
    assert not ledger_path.exists()


def test_run_without_ledger_path_creates_no_database_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tracer invariant re-pin: omitting `--ledger-path` creates no `*.db` file."""
    monkeypatch.chdir(tmp_path)

    exit_code = main(["run", "--heartbeat-interval", "0", "--max-beats", "1"])

    assert exit_code == 0
    assert list(tmp_path.glob("*.db")) == []
