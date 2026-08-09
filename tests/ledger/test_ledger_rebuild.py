"""Tests for folding the ledger into read models (issue #13, `rebuild`).

`rebuild` is a pure fold over `read_all()` after `verify_chain()` passes.
These tests pin: exact byte-for-byte determinism of the three read-model
files across repeated rebuilds of the same ledger, that only those three
files are ever written, that `AlertEmitted` and unrecognized event types
are silently skipped rather than erroring, that an empty ledger still
produces valid (empty) read models, and that a tampered ledger's
corruption propagates as `ChainIntegrityError` instead of silently
producing a plausible-looking but wrong read model.

Issue #40 adds a third, always-written read model: `gateway_events.json`,
a chronological `{seq, created_at, event_type, data}` projection of the
seven Order Gateway / crash-recovery event types (`OrderTransitionLedgered`,
`SubmissionRefused`, `ReduceOnlyRefused`, `ReduceOnlyViolation`,
`ReconciliationHalted`, `ReconciliationHealed`, `RecoveryCompleted`),
mirroring `config_versions.json`/`mode_history.json`'s own
always-written-even-when-empty contract. The pre-existing two read models'
content is unaffected by gateway events mixed into the same ledger.

Issue #58 (PR #199 review fix) wires two more, already-defined projection
functions -- `execution_quality_read_model` / `live_divergence_read_model` --
into `rebuild()` itself: today `rebuild()` never calls either one, so
`execution_quality.json` / `live_divergence.json` are never written at all
(not even empty). The tests below pin that both files are always written
(empty-but-valid on an empty ledger) and hold exactly the
`ExecutionQualityRecorded` / `LiveDivergenceSampled` rows, in ledger order,
once `rebuild()` is wired up.

Issue #76 pins that `rebuild()` fails loudly on a missing ledger path.
`SqliteLedgerStore.__init__` opens its connection with `sqlite3.connect`,
which *creates* the database file if it is absent -- so today `rebuild()` on
a nonexistent `ledger_path` silently produces an empty, "clean" ledger and
writes a full set of empty read models instead of erroring. The test below
pins the corrected contract: `rebuild()` raises `FileNotFoundError` naming
the missing path, and neither the ledger file nor any read model is created
as a side effect of the failed attempt.

Issue #180 pins that a ledger mixing `ConfigLoaded`/`ModeHeartbeat` with the
three evaluation events moved into `windbreak.ledger.events`
(`GatePlanRegistered`, `GatePlanChanged`, `GateComputationMismatch`) still
rebuilds cleanly and that those three events are projection-neutral: `rebuild`
has no dedicated read model for them (mirroring
`test_rebuild_skips_unknown_event_types_without_error`'s
skip-without-erroring contract for genuinely unknown types, but for these
three *named* types this module simply does not yet project), so
`config_versions.json`/`mode_history.json` come out byte-identical to a
control ledger holding only the `ConfigLoaded`/`ModeHeartbeat` rows. The test
imports the three new event classes locally (mirroring this module's own
`mode_history_read_model` local-import precedent below) so this single test's
`ImportError` does not break collection of the rest of the file.

Issue #195 grows `rebuild()`'s always-written output-file set to ten:
`canary_status.json` (the latest-per-provider `CanaryVerdictRecorded`
projection) and `forecasts.json` (every `ForecastCreated` row, for the
weekly-report/dashboard fleet-cost and abstention-rate fold) join the eight
already documented above. `test_rebuild_writes_only_the_documented_read_model_files`
is updated in place (a CONSCIOUS, minimal update -- `rebuild()`'s output-file
set is exhaustively pinned there, so a ninth/tenth file must extend that same
list) to include both; dedicated projection-function tests for
`canary_status_read_model`/`forecasts_read_model` live in
`tests/ledger/test_canary_rebuild.py`, mirroring this module's own
`test_scheduler_rebuild.py` sibling-file precedent for a single issue's new
projections.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import TYPE_CHECKING

import pytest

from windbreak.evaluation.execution_quality import (
    ExecutionQualityRecord,
    ExecutionQualityRecorded,
)
from windbreak.evaluation.live_divergence import (
    LiveDivergenceBreached,
    LiveDivergenceSampled,
)
from windbreak.ledger.events import (
    AlertEmitted,
    ConfigLoaded,
    ModeHeartbeat,
    OrderTransitionLedgered,
    ReconciliationHalted,
    ReconciliationHealed,
    RecoveryCompleted,
    ReduceOnlyRefused,
    ReduceOnlyViolation,
    SubmissionRefused,
    canonical_json,
)
from windbreak.ledger.rebuild import rebuild
from windbreak.ledger.store import (
    ChainIntegrityError,
    SqliteLedgerStore,
    compute_event_hash,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path


def _populate_interleaved_ledger(store: SqliteLedgerStore) -> None:
    """Append a fixed, interleaved sequence of all three M0 event types.

    Sequence numbers: 1=ConfigLoaded, 2=ModeHeartbeat, 3=AlertEmitted,
    4=ModeHeartbeat, 5=ConfigLoaded.
    """
    store.append(
        ConfigLoaded(component="pipeline", config_hash="hash-1", diff={"a": 1})
    )
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.append(AlertEmitted(component="alerts", severity="low", message="noop"))
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=2))
    store.append(
        ConfigLoaded(component="pipeline", config_hash="hash-2", diff={"a": 2})
    )


def test_rebuild_writes_only_the_documented_read_model_files(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """Only the documented read models ever appear in output_dir.

    Every read model is written unconditionally (empty here where no source
    events are present) -- the original three plus the three PAPER-loop read
    models issue #48 adds (`positions.json`, `equity_curve.json`,
    `selector_decisions.json`), the two live-divergence read models
    issue #58 wires up (`execution_quality.json`, `live_divergence.json`,
    PR #199 review fix), and the per-provider vote-cost read model issue #281
    adds (`provider_vote_costs.json`).
    """
    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _populate_interleaved_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    produced = sorted(path.name for path in output_dir.iterdir())
    assert produced == [
        "canary_status.json",
        "config_versions.json",
        "equity_curve.json",
        "execution_quality.json",
        "forecasts.json",
        "gateway_events.json",
        "live_divergence.json",
        "mode_history.json",
        "positions.json",
        "provider_vote_costs.json",
        "selector_decisions.json",
    ]


def test_rebuild_is_byte_for_byte_deterministic_across_repeated_runs(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """Rebuilding the same ledger into two different output dirs is byte-identical."""
    db_path = tmp_path / "ledger.db"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _populate_interleaved_ledger(store)
    store.close()

    first_dir = tmp_path / "out1"
    second_dir = tmp_path / "out2"
    rebuild(db_path, first_dir)
    rebuild(db_path, second_dir)

    for name in ("config_versions.json", "gateway_events.json", "mode_history.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_rebuild_config_versions_contains_only_config_loaded_projection_in_order(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """config_versions.json holds exactly the ConfigLoaded rows, in seq order."""
    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _populate_interleaved_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    config_versions = json.loads((output_dir / "config_versions.json").read_text())
    assert [entry["seq"] for entry in config_versions] == [1, 5]
    assert [entry["config_hash"] for entry in config_versions] == ["hash-1", "hash-2"]
    assert [entry["diff"] for entry in config_versions] == [{"a": 1}, {"a": 2}]
    assert all("created_at" in entry for entry in config_versions)


def test_rebuild_mode_history_contains_only_mode_heartbeat_projection_in_order(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """mode_history.json holds exactly the ModeHeartbeat rows, in seq order."""
    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _populate_interleaved_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    mode_history = json.loads((output_dir / "mode_history.json").read_text())
    assert [entry["seq"] for entry in mode_history] == [2, 4]
    assert [entry["mode"] for entry in mode_history] == ["RESEARCH", "RESEARCH"]
    assert [entry["beat"] for entry in mode_history] == [1, 2]
    assert all("created_at" in entry for entry in mode_history)


def test_rebuild_read_models_are_canonical_json_with_one_trailing_newline(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """Each read model is canonical JSON bytes ending in exactly one newline."""
    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _populate_interleaved_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    for name in ("config_versions.json", "gateway_events.json", "mode_history.json"):
        raw = (output_dir / name).read_bytes()
        assert raw.endswith(b"\n")
        assert not raw.endswith(b"\n\n")
        body = raw[:-1].decode("utf-8")
        assert body == canonical_json(json.loads(body))


def test_rebuild_on_empty_ledger_produces_valid_empty_read_models(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """An empty ledger still produces well-formed, empty read models.

    Includes `execution_quality.json` / `live_divergence.json` (issue #58,
    PR #199 review fix): both currently FAIL with `FileNotFoundError` because
    `rebuild()` never writes either file today, empty ledger or not.
    """
    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.close()

    rebuild(db_path, output_dir)

    assert json.loads((output_dir / "config_versions.json").read_text()) == []
    assert json.loads((output_dir / "gateway_events.json").read_text()) == []
    assert json.loads((output_dir / "mode_history.json").read_text()) == []
    assert json.loads((output_dir / "execution_quality.json").read_text()) == []
    assert json.loads((output_dir / "live_divergence.json").read_text()) == []
    assert json.loads((output_dir / "canary_status.json").read_text()) == []
    assert json.loads((output_dir / "forecasts.json").read_text()) == []


def test_rebuild_skips_unknown_event_types_without_error(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """An event_type absent from EVENT_TYPES is silently skipped, not an error."""
    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()

    created_at = "2099-01-01T00:00:00.000000+00:00"
    payload_json = canonical_json(
        {"component": "future", "data": {"whatever": True}, "schema_version": 1}
    )
    conn = sqlite3.connect(db_path)
    try:
        prev_hash = conn.execute(
            "SELECT event_hash FROM ledger WHERE sequence_number = 1"
        ).fetchone()[0]
        event_hash = compute_event_hash(
            2, "FutureEvent", created_at, payload_json, prev_hash
        )
        conn.execute(
            "INSERT INTO ledger ("
            "sequence_number, event_type, created_at, component, "
            "payload_json, payload_schema_version, prev_hash, event_hash"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                2,
                "FutureEvent",
                created_at,
                "future",
                payload_json,
                1,
                prev_hash,
                event_hash,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    rebuild(db_path, output_dir)

    assert json.loads((output_dir / "config_versions.json").read_text()) == []
    assert json.loads((output_dir / "gateway_events.json").read_text()) == []
    mode_history = json.loads((output_dir / "mode_history.json").read_text())
    assert len(mode_history) == 1
    assert mode_history[0]["beat"] == 1


def _populate_gateway_and_recovery_events_ledger(store: SqliteLedgerStore) -> None:
    """Append M0 events interleaved with all seven gateway/recovery events.

    Sequence numbers: 1=ConfigLoaded, 2=OrderTransitionLedgered,
    3=ModeHeartbeat, 4=SubmissionRefused, 5=ReduceOnlyRefused, 6=AlertEmitted,
    7=ReduceOnlyViolation, 8=ReconciliationHalted, 9=ReconciliationHealed,
    10=RecoveryCompleted.

    Args:
        store: The ledger store to append the fixed sequence into.
    """
    store.append(ConfigLoaded(component="pipeline", config_hash="hash-1", diff={}))
    store.append(
        OrderTransitionLedgered(
            component="order_gateway",
            client_order_id="coid-1",
            from_state="INTENT_CREATED",
            event="APPROVE",
            to_state="APPROVED",
        )
    )
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.append(
        SubmissionRefused(
            component="order_gateway", client_order_id="coid-2", reason="closed"
        )
    )
    store.append(
        ReduceOnlyRefused(
            component="order_gateway",
            client_order_id="coid-3",
            ticker="MKT-DEEP",
            held_centis=500,
            inflight_closing_centis=0,
            requested_close_centis=600,
            reason="reduce_only",
        )
    )
    store.append(AlertEmitted(component="alerts", severity="low", message="noop"))
    store.append(
        ReduceOnlyViolation(
            component="order_gateway",
            client_order_id="coid-4",
            ticker="MKT-DEEP",
            held_centis=500,
            filled_centis=600,
            net_centis=-100,
        )
    )
    store.append(
        ReconciliationHalted(
            component="order_gateway",
            reason="foreign_open_order",
            ticker="MKT-DEEP",
            venue_order_id="paper-order-9",
            client_order_id="",
            detail="untracked order discovered on the venue",
        )
    )
    store.append(
        ReconciliationHealed(
            component="order_gateway",
            client_order_id="coid-5",
            action="fill_confirmed",
            detail="matched an out-of-band fill",
        )
    )
    store.append(
        RecoveryCompleted(component="order_gateway", orders_reconciled=3, halted=False)
    )


def test_rebuild_gateway_events_contains_only_the_seven_gateway_rows_in_order(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """gateway_events.json holds exactly the 7 gateway/recovery rows, in order.

    Each entry is the `{seq, created_at, event_type, data}` shape; the
    pre-existing `config_versions.json`/`mode_history.json` projections are
    unaffected by the gateway events mixed into the same ledger.
    """
    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _populate_gateway_and_recovery_events_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    gateway_events = json.loads((output_dir / "gateway_events.json").read_text())
    assert [entry["seq"] for entry in gateway_events] == [2, 4, 5, 7, 8, 9, 10]
    assert [entry["event_type"] for entry in gateway_events] == [
        "OrderTransitionLedgered",
        "SubmissionRefused",
        "ReduceOnlyRefused",
        "ReduceOnlyViolation",
        "ReconciliationHalted",
        "ReconciliationHealed",
        "RecoveryCompleted",
    ]
    assert all("created_at" in entry for entry in gateway_events)
    assert gateway_events[0]["data"]["client_order_id"] == "coid-1"
    assert gateway_events[-1]["data"] == {"orders_reconciled": 3, "halted": False}

    config_versions = json.loads((output_dir / "config_versions.json").read_text())
    assert [entry["seq"] for entry in config_versions] == [1]
    mode_history = json.loads((output_dir / "mode_history.json").read_text())
    assert [entry["seq"] for entry in mode_history] == [3]


def _populate_execution_quality_and_divergence_ledger(store: SqliteLedgerStore) -> None:
    """Append M0 events interleaved with one execution-quality and one
    live-divergence event (issue #58, PR #199 review fix).

    Sequence numbers: 1=ConfigLoaded, 2=ExecutionQualityRecorded,
    3=ModeHeartbeat, 4=LiveDivergenceSampled.

    Args:
        store: The ledger store to append the fixed sequence into.
    """
    store.append(ConfigLoaded(component="pipeline", config_hash="hash-1", diff={}))
    record = ExecutionQualityRecord(
        fill_id="F-rebuild-1",
        market_ticker="MKT-REBUILD",
        side="YES",
        filled_centis=100,
        actual_cost_micros=1_100_000,
        modeled_cost_micros=1_000_000,
        model_version="pfm-rebuild-test",
        created_sequence=1,
    )
    store.append(ExecutionQualityRecorded(component="evaluation", record=record))
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.append(
        LiveDivergenceSampled(
            component="evaluation",
            sample={
                "live_slippage_ratio_ppm": 1_100_000,
                "live_brier_degradation_ppm": "UNDEFINED",
                "plan_hash": "hash-rebuild-test",
            },
        )
    )


def test_rebuild_execution_quality_and_live_divergence_projections_are_wired(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`rebuild()` also projects `ExecutionQualityRecorded` /
    `LiveDivergenceSampled` rows into `execution_quality.json` /
    `live_divergence.json`, in ledger order (issue #58, PR #199 review fix).

    Today `rebuild()` never calls `execution_quality_read_model` /
    `live_divergence_read_model` even though both are already defined, so
    reading either file currently raises `FileNotFoundError`.
    """
    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _populate_execution_quality_and_divergence_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    execution_quality = json.loads((output_dir / "execution_quality.json").read_text())
    assert [entry["seq"] for entry in execution_quality] == [2]
    assert execution_quality[0]["event_type"] == "ExecutionQualityRecorded"
    assert execution_quality[0]["data"]["fill_id"] == "F-rebuild-1"
    assert execution_quality[0]["data"]["actual_cost_micros"] == 1_100_000
    assert execution_quality[0]["data"]["modeled_cost_micros"] == 1_000_000
    assert all("created_at" in entry for entry in execution_quality)

    live_divergence = json.loads((output_dir / "live_divergence.json").read_text())
    assert [entry["seq"] for entry in live_divergence] == [4]
    assert live_divergence[0]["event_type"] == "LiveDivergenceSampled"
    assert live_divergence[0]["data"]["live_slippage_ratio_ppm"] == 1_100_000
    assert live_divergence[0]["data"]["plan_hash"] == "hash-rebuild-test"
    assert all("created_at" in entry for entry in live_divergence)

    # The pre-existing read models are unaffected by the two new event types
    # mixed into the same ledger.
    config_versions = json.loads((output_dir / "config_versions.json").read_text())
    assert [entry["seq"] for entry in config_versions] == [1]
    mode_history = json.loads((output_dir / "mode_history.json").read_text())
    assert [entry["seq"] for entry in mode_history] == [3]


def _populate_live_divergence_ledger_with_breach(store: SqliteLedgerStore) -> None:
    """Append one `LiveDivergenceSampled` row immediately followed by one
    `LiveDivergenceBreached` row for the same run (Gate 4 round-2 review fix,
    Fix 2: `LiveDivergenceBreached` must appear in the dashboard divergence
    projection).

    Sequence numbers: 1=LiveDivergenceSampled, 2=LiveDivergenceBreached.

    Args:
        store: The ledger store to append the fixed sequence into.
    """
    sample = {
        "live_slippage_ratio_ppm": 1_600_000,
        "live_brier_degradation_ppm": 60_000,
        "plan_hash": "hash-breach-test",
    }
    store.append(LiveDivergenceSampled(component="evaluation", sample=sample))
    store.append(
        LiveDivergenceBreached(
            component="evaluation",
            sample=sample,
            trigger="LIVE_PAPER_SLIPPAGE_DIVERGENCE",
        )
    )


def test_rebuild_live_divergence_projection_includes_breached_rows(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`live_divergence.json` holds BOTH `LiveDivergenceSampled` AND
    `LiveDivergenceBreached` rows, in ledger order, and the breach row's
    `data` carries its firing `trigger` (Gate 4 round-2 review fix, Fix 2).

    `LiveDivergenceBreached` is the durable, operator-facing audit trail for
    the SPEC S10.10 automatic-demotion gate. Today
    `live_divergence_read_model` filters on the `LiveDivergenceSampled` event
    type alone (`_LIVE_DIVERGENCE_SAMPLED`), so the `LiveDivergenceBreached`
    row is silently dropped from the read model entirely -- this fails on
    `assert [entry["seq"] for entry in live_divergence] == [1, 2]` (today it
    is `[1]`, missing the seq-2 breach row).
    """
    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _populate_live_divergence_ledger_with_breach(store)
    store.close()

    rebuild(db_path, output_dir)

    live_divergence = json.loads((output_dir / "live_divergence.json").read_text())

    assert [entry["seq"] for entry in live_divergence] == [1, 2]
    assert live_divergence[0]["event_type"] == "LiveDivergenceSampled"
    assert live_divergence[1]["event_type"] == "LiveDivergenceBreached"
    assert live_divergence[1]["data"]["trigger"] == "LIVE_PAPER_SLIPPAGE_DIVERGENCE"
    assert live_divergence[1]["data"]["live_slippage_ratio_ppm"] == 1_600_000
    assert all("created_at" in entry for entry in live_divergence)


def test_rebuild_on_tampered_ledger_raises_chain_integrity_error(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """rebuild verifies the chain first, so a corrupt ledger never projects."""
    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _populate_interleaved_ledger(store)
    store.close()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE ledger SET event_hash = ? WHERE sequence_number = 3",
            ("0" * 64,),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ChainIntegrityError):
        rebuild(db_path, output_dir)


def test_rebuild_on_missing_ledger_path_raises_file_not_found_error(
    tmp_path: Path,
) -> None:
    """rebuild() on a nonexistent ledger_path fails loudly, not silently (issue #76).

    `SqliteLedgerStore.__init__` opens `ledger_path` via `sqlite3.connect`,
    which creates the file if it does not exist -- so without an explicit
    existence check, `rebuild()` on a missing path today succeeds, silently
    fabricating an empty ledger and a full set of empty read models. This
    pins the corrected contract: a `FileNotFoundError` naming the missing
    path, with neither the ledger file nor any read model created as a
    side effect of the failed call.
    """
    missing_ledger_path = tmp_path / "missing.db"
    output_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_ledger_path))):
        rebuild(missing_ledger_path, output_dir)

    assert not missing_ledger_path.exists()
    assert not (output_dir / "config_versions.json").exists()


def test_mode_history_read_model_projects_mode_heartbeats_in_ledger_order(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`mode_history_read_model` projects every `ModeHeartbeat`, in ledger order.

    Issue #79 wires the dashboard's ledger-backed status source through a new
    *public* `mode_history_read_model`, wrapping the existing private
    `_mode_projection` the same way `equity_curve_read_model` etc. already
    wrap `_gateway_projection`. It does not exist yet -- only `_mode_projection`
    does -- so this fails with `ImportError` at the local import below, scoped
    to this one test so the rest of this module keeps collecting and passing.
    """
    from windbreak.ledger.rebuild import mode_history_read_model

    db_path = tmp_path / "ledger.db"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ConfigLoaded(component="pipeline", config_hash="hash-1", diff={}))
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.append(ModeHeartbeat(component="pipeline", mode="PAPER", beat=2))
    store.verify_chain()
    records = store.read_all()
    store.close()

    rows = mode_history_read_model(records)

    assert [row["seq"] for row in rows] == [2, 3]
    assert [row["mode"] for row in rows] == ["RESEARCH", "PAPER"]
    assert [row["beat"] for row in rows] == [1, 2]
    assert all("created_at" in row for row in rows)


# --- Issue #180: the three evaluation-defined events moved into ---------------
# --- `windbreak.ledger.events` are projection-neutral in `rebuild()` ----------


def _append_config_and_mode_heartbeat(store: SqliteLedgerStore) -> None:
    """Append the two M0 rows shared by the main and control ledgers below.

    Sequence numbers: 1=ConfigLoaded, 2=ModeHeartbeat.

    Args:
        store: The ledger store to append the fixed sequence into.
    """
    store.append(
        ConfigLoaded(component="pipeline", config_hash="hash-1", diff={"a": 1})
    )
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))


def test_rebuild_config_and_mode_projections_are_neutral_to_moved_evaluation_events(
    tmp_path: Path,
) -> None:
    """`config_versions.json`/`mode_history.json` are unaffected by the three
    evaluation-defined events moved into `windbreak.ledger.events` (issue #180):
    `GatePlanRegistered`, `GatePlanChanged`, `GateComputationMismatch`.

    A MAIN ledger holds the same `ConfigLoaded`/`ModeHeartbeat` rows as a
    CONTROL ledger -- seq 1 and 2 in both -- with the three moved event types
    appended afterward (seq 3-5), mixing all three into the same ledger.
    `rebuild()` has no dedicated read model for any of the three, so they fall
    through the same skip-without-erroring path
    `test_rebuild_skips_unknown_event_types_without_error` pins for genuinely
    unknown types: `config_versions.json`/`mode_history.json` come out
    byte-identical between the two ledgers, proving the three moved types are
    projection-neutral. Appending the shared rows first (rather than
    interleaving the three new rows in between them) keeps their `seq` --
    which the read-model entries carry verbatim -- identical between the two
    ledgers; each store also gets its own fresh deterministic clock (the same
    reproducible, call-indexed behavior `deterministic_clock` provides) so
    the shared seq-1/seq-2 rows carry identical `created_at` timestamps too,
    despite the main ledger's three extra rows appended afterward.

    The three new classes are imported locally, mirroring this module's own
    `mode_history_read_model` precedent above, so this single test's
    `ImportError` -- they do not exist in `windbreak.ledger.events` yet --
    does not break collection of the rest of this file.
    """
    from tests.ledger.conftest import DeterministicClock
    from windbreak.ledger.events import (
        GateComputationMismatch,
        GatePlanChanged,
        GatePlanRegistered,
    )

    plan_fields: dict[str, object] = {
        "metric_windows": [["brier", "all"]],
        "min_resolved_for_calibration": 150,
        "promotion_min_resolved": 300,
        "promotion_min_independent_event_groups": 100,
        "brier_skill_required_ppm": 10_000,
        "bootstrap_confidence_ppm": 950_000,
        "live_rolling_window_size": 100,
        "live_slippage_ratio_limit_ppm": 1_500_000,
        "live_brier_degradation_band_ppm": 50_000,
        "observation_window": "latest_before_close",
        "baseline_scheme": "executable_price_at_baseline_snapshot",
        "clustering_scheme": "event_correlation_group",
        "paper_fill_model_version": "pfm-v1",
    }
    plan_hash = "a" * 64
    previous_plan_hash = "b" * 64

    main_db_path = tmp_path / "main.db"
    control_db_path = tmp_path / "control.db"
    main_out = tmp_path / "main_out"
    control_out = tmp_path / "control_out"

    main_store = SqliteLedgerStore(main_db_path, now=DeterministicClock())
    _append_config_and_mode_heartbeat(main_store)
    main_store.append(
        GatePlanRegistered(
            component="evaluation",
            **plan_fields,
            plan_hash=plan_hash,
            paper_clock_start=1_700_000_000,
        )
    )
    main_store.append(
        GatePlanChanged(
            component="evaluation",
            **plan_fields,
            plan_hash=plan_hash,
            paper_clock_start=1_700_000_100,
            previous_plan_hash=previous_plan_hash,
        )
    )
    main_store.append(
        GateComputationMismatch(
            component="evaluation",
            plan_hash=plan_hash,
            tolerance=1,
            mismatches=[
                {
                    "name": "brier",
                    "window": "latest_before_close",
                    "python_value": 54_000,
                    "sql_value": 55_000,
                }
            ],
        )
    )
    main_store.close()

    control_store = SqliteLedgerStore(control_db_path, now=DeterministicClock())
    _append_config_and_mode_heartbeat(control_store)
    control_store.close()

    rebuild(main_db_path, main_out)
    rebuild(control_db_path, control_out)

    for name in ("config_versions.json", "mode_history.json"):
        assert (main_out / name).read_bytes() == (control_out / name).read_bytes()
