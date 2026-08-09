"""Tests for the three new PAPER-loop read-model projections (issue #48, RED).

`windbreak.ledger.rebuild.rebuild` does not yet write `positions.json`,
`equity_curve.json`, or `selector_decisions.json`, and `windbreak.ledger.events`
does not yet define the event types those projections fold -- so every test
below fails collection or assertion with either `ImportError` (the new event
types) or a missing/empty-in-the-wrong-way output file -- the expected Gate 1
RED state for issue #48.

Mirrors `tests/ledger/test_ledger_rebuild.py`'s own idiom throughout: the
`deterministic_clock`/`ledger_store_factory` fixtures from
`tests/ledger/conftest.py`, canonical-JSON-with-trailing-newline output,
determinism across repeated rebuilds, always-written-even-when-empty (`[]`),
and silent skip of unrecognized event types.

Read-model shapes pinned here (this test module's own invented, minimal
contract, since the issue names the three files but not their exact rows):

* `positions.json` -- a list holding at most one entry, the *latest*
  `PositionsSnapshotRecorded` projected in the same `{seq, created_at,
  event_type, data}` shape `gateway_events.json` already uses; `[]` when no
  such event has ever been ledgered.
* `equity_curve.json` -- every `EquitySampled` row, in ledger order, same shape.
* `selector_decisions.json` -- every `SelectorDecisionRecorded`,
  `IntentApproved`, and `IntentVetoed` row, interleaved in ledger order, same
  shape (the latter two are bare `Event`s the Risk Kernel already emits, not
  new typed classes -- see `windbreak/riskkernel/process.py`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from windbreak.ledger.store import SqliteLedgerStore


def _append_full_scheduler_ledger(store: SqliteLedgerStore) -> None:
    """Append a fixed, interleaved sequence exercising all three new projections.

    Sequence numbers: 1=MarketSnapshotRecorded, 2=EquitySampled,
    3=SelectorDecisionRecorded, 4=IntentVetoed (bare Event), 5=EquitySampled,
    6=PositionsSnapshotRecorded, 7=ModeHeartbeat (unrelated, must not leak
    into any of the three new projections), 8=PositionsSnapshotRecorded
    (a second, *later* snapshot -- `positions.json` must hold only this one).

    Args:
        store: The ledger store to append the fixed sequence into.
    """
    from windbreak.ledger.events import (
        EquitySampled,
        Event,
        MarketSnapshotRecorded,
        ModeHeartbeat,
        PositionsSnapshotRecorded,
        SelectorDecisionRecorded,
    )

    store.append(
        MarketSnapshotRecorded(
            component="scheduler",
            ticker="MKT-DEEP",
            best_bid_pips=4500,
            best_ask_pips=4600,
            fetched_at_epoch_s=1_700_000_000,
        )
    )
    store.append(
        EquitySampled(
            component="scheduler",
            equity_micros=1_000_000_000,
            floor_micros=0,
            epoch_s=1_700_000_000,
        )
    )
    store.append(
        SelectorDecisionRecorded(
            component="scheduler",
            forecast_id="fc-0001",
            market_ticker="MKT-DEEP",
            intent_count=1,
            reasons=["pass:net_edge_min"],
        )
    )
    store.append(
        Event(
            event_type="IntentVetoed",
            component="riskkernel",
            payload_schema_version=1,
            payload={
                "intent_id": "intent-0001",
                "reasons": ["balance verification stale or missing"],
            },
        )
    )
    store.append(
        EquitySampled(
            component="scheduler",
            equity_micros=1_000_500_000,
            floor_micros=0,
            epoch_s=1_700_000_060,
        )
    )
    store.append(
        PositionsSnapshotRecorded(
            component="scheduler",
            positions=[
                {
                    "ticker": "MKT-DEEP",
                    "quantity_centis": 100,
                    "average_price_pips": 4600,
                }
            ],
        )
    )
    store.append(ModeHeartbeat(component="scheduler", mode="PAPER", beat=1))
    store.append(
        PositionsSnapshotRecorded(
            component="scheduler",
            positions=[
                {
                    "ticker": "MKT-DEEP",
                    "quantity_centis": 200,
                    "average_price_pips": 4600,
                }
            ],
        )
    )


def test_rebuild_writes_the_three_new_read_model_files(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`rebuild` unconditionally writes `positions.json`, `equity_curve.json`,
    and `selector_decisions.json` alongside the pre-existing three.
    """
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _append_full_scheduler_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    produced = sorted(path.name for path in output_dir.iterdir())
    assert "positions.json" in produced
    assert "equity_curve.json" in produced
    assert "selector_decisions.json" in produced


def test_rebuild_is_byte_for_byte_deterministic_for_the_new_read_models(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """Rebuilding the same ledger twice yields byte-identical new read models."""
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _append_full_scheduler_ledger(store)
    store.close()

    first_dir = tmp_path / "out1"
    second_dir = tmp_path / "out2"
    rebuild(db_path, first_dir)
    rebuild(db_path, second_dir)

    for name in ("positions.json", "equity_curve.json", "selector_decisions.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_positions_json_holds_only_the_single_latest_snapshot(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`positions.json` holds exactly one entry: the *latest*
    `PositionsSnapshotRecorded` (seq 8), never the earlier seq-6 snapshot.
    """
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _append_full_scheduler_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    positions = json.loads((output_dir / "positions.json").read_text())
    assert len(positions) == 1
    assert positions[0]["seq"] == 8
    assert positions[0]["data"]["positions"][0]["quantity_centis"] == 200


def test_equity_curve_json_contains_every_equity_sampled_row_in_order(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`equity_curve.json` holds both `EquitySampled` rows (seq 2 and 5), in order."""
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _append_full_scheduler_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    equity_curve = json.loads((output_dir / "equity_curve.json").read_text())
    assert [entry["seq"] for entry in equity_curve] == [2, 5]
    assert [entry["data"]["equity_micros"] for entry in equity_curve] == [
        1_000_000_000,
        1_000_500_000,
    ]


def test_selector_decisions_json_interleaves_selector_and_intent_events(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`selector_decisions.json` holds `SelectorDecisionRecorded` (seq 3) and
    the bare `IntentVetoed` (seq 4), in ledger order.
    """
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _append_full_scheduler_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    selector_decisions = json.loads(
        (output_dir / "selector_decisions.json").read_text()
    )
    assert [entry["seq"] for entry in selector_decisions] == [3, 4]
    assert [entry["event_type"] for entry in selector_decisions] == [
        "SelectorDecisionRecorded",
        "IntentVetoed",
    ]


def test_new_read_models_are_empty_list_on_an_empty_ledger(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """An empty ledger still produces the three new, well-formed empty read models."""
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.close()

    rebuild(db_path, output_dir)

    assert json.loads((output_dir / "positions.json").read_text()) == []
    assert json.loads((output_dir / "equity_curve.json").read_text()) == []
    assert json.loads((output_dir / "selector_decisions.json").read_text()) == []


def test_new_read_models_are_canonical_json_with_one_trailing_newline(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """Each new read model is canonical JSON bytes ending in exactly one newline."""
    from windbreak.ledger.events import canonical_json
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _append_full_scheduler_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    for name in ("positions.json", "equity_curve.json", "selector_decisions.json"):
        raw = (output_dir / name).read_bytes()
        assert raw.endswith(b"\n")
        assert not raw.endswith(b"\n\n")
        body = raw[:-1].decode("utf-8")
        assert body == canonical_json(json.loads(body))


def test_unrelated_mode_heartbeat_never_leaks_into_any_new_projection(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """The unrelated seq-7 `ModeHeartbeat` never appears in any of the three
    new read models -- each stays scoped to its own event type(s).
    """
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    _append_full_scheduler_ledger(store)
    store.close()

    rebuild(db_path, output_dir)

    for name in ("positions.json", "equity_curve.json", "selector_decisions.json"):
        rows = json.loads((output_dir / name).read_text())
        assert all(row["event_type"] != "ModeHeartbeat" for row in rows)
