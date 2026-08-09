"""Tests for `windbreak.scheduler.weekly_data` (issue #188).

Pins `weekly_report_body(records, *, today) -> str`: the pure function that
folds a whole `SqliteLedgerStore.read_all()` result into a rendered weekly
report, wiring REAL `EvaluationInputs`/`FixtureForecast`s and a real
`CostMeter` in place of the #55 `evaluation=None, costs=None` placeholder
`windbreak.scheduler.loop.run_single_tick` writes today.

Every `ForecastCreated` record folds into a `FixtureForecast` with
`created_sequence = record.sequence_number`, `baseline_executable_price_pips`
and the research cost read verbatim off the payload, `traded=False`,
`live=False`. The fold always gates against `EvaluationInputs(forecasts=...,
resolutions={}, temporal=TemporalContext(deployment_sequence=0,
resolution_sequences={}))` -- a whole-ledger fold has no ground-truth
resolutions available at all -- so every folded forecast is temporally
admitted (`created_sequence` is always `>= 1 > 0 ==
deployment_sequence`) and then rejected by the evaluation gate's own
`UNRESOLVED` reason (`market_ticker not in resolutions`), landing in the
rendered report's `== rejections ==` ledger rather than silently vanishing.
A legacy (pre-#188, `payload_schema_version == 1`) `ForecastCreated` row --
missing `research_cost_micros` / `market_price_baseline_pips` -- fails closed
with a `ValueError` naming the missing key, never a silent zero-cost default
or a bare `KeyError`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from typing import TYPE_CHECKING

import pytest

from windbreak.ledger.events import EquitySampled, ModeHeartbeat
from windbreak.ledger.store import SqliteLedgerStore

if TYPE_CHECKING:
    from pathlib import Path

#: The component label every ledgered event in this suite is stamped with.
_COMPONENT = "scheduler"


def _append_forecast(
    store: SqliteLedgerStore,
    *,
    forecast_id: str,
    market_ticker: str,
    research_cost_micros: int,
    market_price_baseline_pips: int,
    probability_ppm: int = 500_000,
    eligible_for_live: bool = False,
    abstention_reason: str | None = "no_verified_citations",
) -> None:
    """Append one v2-shaped `ForecastCreated` record to `store`.

    Args:
        store: The ledger store to append into.
        forecast_id: The forecast's deterministic id.
        market_ticker: The market the forecast is for.
        research_cost_micros: The forecast's research spend, in micros.
        market_price_baseline_pips: The baseline executable price, in pips.
        probability_ppm: The forecast probability, in ppm.
        eligible_for_live: Whether the forecast may back a live order.
        abstention_reason: Why the engine abstained, or `None` when traded.
    """
    from windbreak.ledger.events import ForecastCreated

    store.append(
        ForecastCreated(
            component=_COMPONENT,
            forecast_id=forecast_id,
            market_ticker=market_ticker,
            probability_ppm=probability_ppm,
            eligible_for_live=eligible_for_live,
            abstention_reason=abstention_reason,
            research_cost_micros=research_cost_micros,
            market_price_baseline_pips=market_price_baseline_pips,
        )
    )


def _write_legacy_v1_forecast_created_row(
    db_path: Path, *, payload: dict[str, object] | None = None, schema_version: int = 1
) -> None:
    """Hand-insert a legacy/partial `ForecastCreated` row directly via `sqlite3`.

    Reproduces exactly what a ledger row written before this issue (or a
    partially-migrated one) looks like on disk: the current `ForecastCreated`
    constructor always stamps the new v2 five-plus-two-field shape and cannot
    itself produce a row missing a v2 key, so this bypasses it entirely and
    writes the raw envelope. Chain hash correctness is irrelevant here --
    `weekly_report_body` never calls `verify_chain` -- so `prev_hash`/
    `event_hash` are filler.

    Args:
        db_path: The (not yet created) SQLite database path to write into.
        payload: The raw ``data`` payload to store; defaults to the pre-#188
            v1 shape missing both `research_cost_micros` and
            `market_price_baseline_pips`.
        schema_version: The `payload_schema_version` to stamp on the row.
    """
    legacy_payload = payload or {
        "forecast_id": "fc-legacy",
        "market_ticker": "MKT-LEGACY",
        "probability_ppm": 500_000,
        "eligible_for_live": False,
        "abstention_reason": None,
    }
    envelope = {
        "component": _COMPONENT,
        "data": legacy_payload,
        "schema_version": schema_version,
    }
    envelope_json = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ledger ("
            "sequence_number INTEGER PRIMARY KEY, "
            "event_type TEXT NOT NULL, "
            "created_at TEXT NOT NULL, "
            "component TEXT NOT NULL, "
            "payload_json TEXT NOT NULL, "
            "payload_schema_version INTEGER NOT NULL, "
            "prev_hash TEXT NOT NULL, "
            "event_hash TEXT NOT NULL"
            ")"
        )
        conn.execute(
            "INSERT INTO ledger (sequence_number, event_type, created_at, "
            "component, payload_json, payload_schema_version, prev_hash, "
            "event_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                "ForecastCreated",
                "2026-01-01T00:00:00.000000+00:00",
                _COMPONENT,
                envelope_json,
                schema_version,
                "0" * 64,
                "f" * 64,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. A folded, unresolved forecast lands in the Evaluation section's
#    rejection ledger, never silently dropped.
# ---------------------------------------------------------------------------


def test_weekly_report_body_names_the_folded_forecast_under_rejections(
    tmp_path: Path,
) -> None:
    """A single `ForecastCreated` record, folded with no ground-truth
    resolutions available, renders under the `## Evaluation` section's
    `== rejections ==` ledger, naming its forecast id and market ticker.
    """
    from windbreak.scheduler.weekly_data import weekly_report_body

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    _append_forecast(
        store,
        forecast_id="fc-0001",
        market_ticker="MKT-DEEP",
        research_cost_micros=1_500_000,
        market_price_baseline_pips=4600,
    )

    body = weekly_report_body(store.read_all(), today=date(2026, 1, 5))

    evaluation_section = body.split("## Evaluation", 1)[1].split("## Cost meter", 1)[0]
    assert "== rejections ==" in evaluation_section
    assert "fc-0001" in evaluation_section
    assert "MKT-DEEP" in evaluation_section


def test_weekly_report_body_renders_one_rejection_line_per_folded_forecast(
    tmp_path: Path,
) -> None:
    """Two distinct `ForecastCreated` records each land their own rejection
    line -- none merged, none dropped.
    """
    from windbreak.scheduler.weekly_data import weekly_report_body

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    _append_forecast(
        store,
        forecast_id="fc-A",
        market_ticker="MKT-A",
        research_cost_micros=1_000_000,
        market_price_baseline_pips=4500,
    )
    _append_forecast(
        store,
        forecast_id="fc-B",
        market_ticker="MKT-B",
        research_cost_micros=2_000_000,
        market_price_baseline_pips=5100,
    )

    body = weekly_report_body(store.read_all(), today=date(2026, 1, 5))

    evaluation_section = body.split("## Evaluation", 1)[1].split("## Cost meter", 1)[0]
    assert "fc-A" in evaluation_section
    assert "fc-B" in evaluation_section
    assert evaluation_section.count("== rejections ==") == 1
    rejection_lines = [
        line
        for line in evaluation_section.splitlines()
        if line.startswith("EVALUATION_RECORD_REJECTED")
    ]
    assert len(rejection_lines) == 2


def test_weekly_report_body_preserves_ledger_order_in_the_rejection_ledger(
    tmp_path: Path,
) -> None:
    """The fold honors each forecast's own ledger provenance (its
    `created_sequence`, sourced from the record's `sequence_number`): two
    forecasts appended in a fixed order render in that same order in the
    rejection ledger, not e.g. reversed or re-sorted by id.
    """
    from windbreak.scheduler.weekly_data import weekly_report_body

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    _append_forecast(
        store,
        forecast_id="fc-first",
        market_ticker="MKT-FIRST",
        research_cost_micros=1_000_000,
        market_price_baseline_pips=4500,
    )
    _append_forecast(
        store,
        forecast_id="fc-second",
        market_ticker="MKT-SECOND",
        research_cost_micros=1_000_000,
        market_price_baseline_pips=4500,
    )

    body = weekly_report_body(store.read_all(), today=date(2026, 1, 5))

    evaluation_section = body.split("## Evaluation", 1)[1].split("## Cost meter", 1)[0]
    assert evaluation_section.index("fc-first") < evaluation_section.index("fc-second")


# ---------------------------------------------------------------------------
# 2. Cost meter: the total equals the summed research_cost_micros.
# ---------------------------------------------------------------------------


def test_weekly_report_body_cost_meter_total_equals_summed_research_cost(
    tmp_path: Path,
) -> None:
    """The Cost meter section's total equals the sum of every folded
    `ForecastCreated`'s `research_cost_micros` -- unconditionally on
    evaluation admission, since research spend is incurred regardless of
    whether the market later resolves.
    """
    from windbreak.scheduler.weekly_data import weekly_report_body

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    _append_forecast(
        store,
        forecast_id="fc-0001",
        market_ticker="MKT-A",
        research_cost_micros=1_000_000,
        market_price_baseline_pips=4500,
    )
    _append_forecast(
        store,
        forecast_id="fc-0002",
        market_ticker="MKT-B",
        research_cost_micros=2_500_007,
        market_price_baseline_pips=5000,
    )

    body = weekly_report_body(store.read_all(), today=date(2026, 1, 5))

    cost_section = body.split("## Cost meter", 1)[1]
    assert str(1_000_000 + 2_500_007) in cost_section


def test_weekly_report_body_ignores_non_forecast_created_records_in_the_cost_total(
    tmp_path: Path,
) -> None:
    """A non-`ForecastCreated` record (e.g. `EquitySampled`, `ModeHeartbeat`)
    interleaved in the ledger contributes nothing to the cost total and does
    not crash the fold.
    """
    from windbreak.scheduler.weekly_data import weekly_report_body

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    store.append(
        EquitySampled(
            component=_COMPONENT, equity_micros=10_000_000, floor_micros=0, epoch_s=1
        )
    )
    _append_forecast(
        store,
        forecast_id="fc-0001",
        market_ticker="MKT-A",
        research_cost_micros=3_000_000,
        market_price_baseline_pips=4500,
    )
    store.append(ModeHeartbeat(component=_COMPONENT, mode="PAPER", beat=1))

    body = weekly_report_body(store.read_all(), today=date(2026, 1, 5))

    cost_section = body.split("## Cost meter", 1)[1]
    assert "3000000" in cost_section


# ---------------------------------------------------------------------------
# 3. A ledger with no ForecastCreated records at all renders the fallback,
#    never crashes.
# ---------------------------------------------------------------------------


def test_weekly_report_body_over_a_ledger_with_no_forecasts_never_crashes(
    tmp_path: Path,
) -> None:
    """A ledger holding only non-forecast events folds to zero forecasts.
    `weekly_report_body` always wires a genuinely built (if metric-empty)
    evaluation report and cost meter -- never the #55 `evaluation=None,
    costs=None` placeholder -- so both new sections still render their real
    track/cost-meter structure rather than crashing on an empty fold; with
    zero forecasts there is nothing to reject, so the rejection ledger is
    absent entirely (not an empty, header-only section).
    """
    from windbreak.scheduler.weekly_data import weekly_report_body

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    store.append(
        EquitySampled(component=_COMPONENT, equity_micros=0, floor_micros=0, epoch_s=1)
    )

    body = weekly_report_body(store.read_all(), today=date(2026, 1, 5))

    assert "## Evaluation" in body
    assert "## Cost meter" in body
    assert "== forecast ==" in body
    assert "== rejections ==" not in body


# ---------------------------------------------------------------------------
# 4. Fail closed on a legacy (pre-#188) v1 ForecastCreated payload.
# ---------------------------------------------------------------------------


def test_weekly_report_body_raises_value_error_on_a_legacy_v1_forecast_created_payload(
    tmp_path: Path,
) -> None:
    """A legacy v1 `ForecastCreated` row -- missing `research_cost_micros` --
    fails closed with a `ValueError` naming the missing key, rather than
    silently defaulting to zero cost or crashing with a bare `KeyError`.
    """
    from windbreak.scheduler.weekly_data import weekly_report_body

    db_path = tmp_path / "legacy.db"
    _write_legacy_v1_forecast_created_row(db_path)
    store = SqliteLedgerStore(db_path)

    with pytest.raises(ValueError, match="research_cost_micros"):
        weekly_report_body(store.read_all(), today=date(2026, 1, 5))


# ---------------------------------------------------------------------------
# 5. Real equity/position/decision wiring into the three original #48 stub
#    sections (issue #255): `weekly_report_body` builds their bodies via the
#    canonical `equity_curve_read_model` / `positions_read_model` /
#    `selector_decisions_read_model` rebuild folds over the same `records`
#    it already holds, and passes them through to `render_weekly_report`.
# ---------------------------------------------------------------------------


def test_weekly_report_body_wires_equity_position_decision_sections(
    tmp_path: Path,
) -> None:
    """A ledger carrying one `EquitySampled`, one `PositionsSnapshotRecorded`,
    and one `SelectorDecisionRecorded` record renders their content under the
    three respective stub headings -- the signed equity buffer, the position's
    ticker, and the decision's reason -- instead of the `No data yet.`
    placeholder those three sections still hardcode today.
    """
    from windbreak.ledger.events import (
        PositionsSnapshotRecorded,
        SelectorDecisionRecorded,
    )
    from windbreak.scheduler.weekly_data import weekly_report_body

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    store.append(
        EquitySampled(
            component=_COMPONENT,
            equity_micros=11_000_000,
            floor_micros=10_000_000,
            epoch_s=1700,
        )
    )
    store.append(
        PositionsSnapshotRecorded(
            component=_COMPONENT,
            positions=[
                {
                    "ticker": "MKT-DEEP",
                    "quantity_centis": 200,
                    "average_price_pips": 4600,
                }
            ],
        )
    )
    store.append(
        SelectorDecisionRecorded(
            component=_COMPONENT,
            forecast_id="fc-0001",
            market_ticker="MKT-DEEP",
            intent_count=1,
            reasons=["net_edge_below_minimum"],
        )
    )

    body = weekly_report_body(store.read_all(), today=date(2026, 1, 5))

    equity_section = body.split("## Equity vs floor", 1)[1].split("## Positions", 1)[0]
    positions_section = body.split("## Positions", 1)[1].split("## Decisions", 1)[0]
    decisions_section = body.split("## Decisions", 1)[1].split("## Evaluation", 1)[0]

    assert "buffer_micros=+1000000" in equity_section
    assert "No data yet." not in equity_section
    assert "MKT-DEEP" in positions_section
    assert "No data yet." not in positions_section
    assert "net_edge_below_minimum" in decisions_section
    assert "No data yet." not in decisions_section


def test_weekly_report_body_positions_section_uses_only_the_latest_snapshot(
    tmp_path: Path,
) -> None:
    """Two `PositionsSnapshotRecorded` records fold to only the latest
    snapshot's content under `## Positions` -- the earlier snapshot's ticker
    never appears, mirroring `positions_read_model`'s own latest-wins
    contract.
    """
    from windbreak.ledger.events import PositionsSnapshotRecorded
    from windbreak.scheduler.weekly_data import weekly_report_body

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    store.append(
        PositionsSnapshotRecorded(
            component=_COMPONENT,
            positions=[
                {"ticker": "MKT-OLD", "quantity_centis": 10, "average_price_pips": 100}
            ],
        )
    )
    store.append(
        PositionsSnapshotRecorded(
            component=_COMPONENT,
            positions=[
                {"ticker": "MKT-NEW", "quantity_centis": 20, "average_price_pips": 200}
            ],
        )
    )

    body = weekly_report_body(store.read_all(), today=date(2026, 1, 5))

    positions_section = body.split("## Positions", 1)[1].split("## Decisions", 1)[0]
    assert "MKT-NEW" in positions_section
    assert "MKT-OLD" not in positions_section


def test_weekly_report_body_over_an_empty_ledger_renders_honest_zero_counts(
    tmp_path: Path,
) -> None:
    """An empty ledger (zero records) renders the three sections' honest
    zero-count bodies (`equity_samples=0`, `snapshots=0`,
    `decision_events=0`) -- never the `No data yet.` placeholder those three
    headings hardcode today, and never a crash.
    """
    from windbreak.scheduler.weekly_data import weekly_report_body

    store = SqliteLedgerStore(tmp_path / "ledger.db")

    body = weekly_report_body(store.read_all(), today=date(2026, 1, 5))

    equity_section = body.split("## Equity vs floor", 1)[1].split("## Positions", 1)[0]
    positions_section = body.split("## Positions", 1)[1].split("## Decisions", 1)[0]
    decisions_section = body.split("## Decisions", 1)[1].split("## Evaluation", 1)[0]

    assert "equity_samples=0" in equity_section
    assert "snapshots=0" in positions_section
    assert "decision_events=0" in decisions_section
    assert "No data yet." not in equity_section
    assert "No data yet." not in positions_section
    assert "No data yet." not in decisions_section


def test_weekly_report_body_names_the_missing_baseline_key_when_only_it_is_absent(
    tmp_path: Path,
) -> None:
    """A partial row carrying `research_cost_micros` but missing only
    `market_price_baseline_pips` fails closed with a `ValueError` naming *that*
    key -- proving the second fail-closed check names its own missing key,
    independently of the first (which validates research cost and would
    short-circuit if both were absent).
    """
    from windbreak.scheduler.weekly_data import weekly_report_body

    db_path = tmp_path / "partial.db"
    _write_legacy_v1_forecast_created_row(
        db_path,
        payload={
            "forecast_id": "fc-partial",
            "market_ticker": "MKT-PARTIAL",
            "probability_ppm": 500_000,
            "eligible_for_live": False,
            "abstention_reason": None,
            "research_cost_micros": 3_000_000,
        },
        schema_version=2,
    )
    store = SqliteLedgerStore(db_path)

    with pytest.raises(ValueError, match="market_price_baseline_pips"):
        weekly_report_body(store.read_all(), today=date(2026, 1, 5))
