"""Tests for the two new canary/forecast read-model projections (issue #195, RED).

`windbreak.ledger.events` does not yet define `CanaryVerdictRecorded`, and
`windbreak.ledger.rebuild` does not yet define `canary_status_read_model` /
`forecasts_read_model` (nor does `rebuild()` write `canary_status.json` /
`forecasts.json`) -- so every test below fails with either `ImportError`
(the new event type or the new projection functions) or a missing output
file -- the expected Gate 1 RED state for issue #195.

Every new symbol is imported locally inside each test (mirroring
`tests/ledger/test_ledger_rebuild.py`'s own
`test_mode_history_read_model_projects_mode_heartbeats_in_ledger_order` /
`test_rebuild_config_and_mode_projections_are_neutral_to_moved_evaluation_events`
local-import precedent) so one new symbol's absence never breaks collection
of the rest of this file.

Read-model shapes pinned here (this module's own minimal, documented
contract, mirroring `tests/ledger/test_scheduler_rebuild.py`'s own
issue-scoped invention):

* `canary_status.json` -- the LATEST `CanaryVerdictRecorded` row per
  provider, in the same `{seq, created_at, event_type, data}` shape every
  other projection uses; a provider that appears more than once keeps only
  its most recently ledgered verdict, at that provider's first-seen list
  position (mirrors a Python dict's own "update in place, keep original
  position" semantics -- the simplest, most literal "latest wins" contract).
  `[]` when no such event has ever been ledgered.
* `forecasts.json` -- every `ForecastCreated` row, in ledger order, same
  `{seq, created_at, event_type, data}` shape (feeds the weekly-report/
  dashboard fleet cost-per-forecast and abstention-rate fold).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from windbreak.ledger.store import LedgerRecord


def _sample_verdict_kwargs(
    *, provider: str, status: str, drift_kind: str = "", drift_score_ppm: int = 0
) -> dict[str, object]:
    """Build a minimal, valid `CanaryVerdictRecorded` constructor kwargs dict.

    Args:
        provider: The provider identifier.
        status: The verdict status (`"OK"`/`"ANSWER_DRIFT"`/`"VERSION_DRIFT"`).
        drift_kind: The drift kind (`""` for a clean `OK` verdict).
        drift_score_ppm: The scored drift, in ppm.

    Returns:
        The kwargs (sans `component`) `CanaryVerdictRecorded` accepts.
    """
    return {
        "provider": provider,
        "status": status,
        "drift_kind": drift_kind,
        "drift_score_ppm": drift_score_ppm,
        "tolerance_ppm": 50_000,
        "reported_version": "v1",
        "pinned_versions": ["v1"],
    }


# --- canary_status_read_model: pure fold, latest-per-provider wins ------------


def test_canary_status_read_model_empty_input_returns_empty_list() -> None:
    """No records at all yields an empty list, not an error."""
    from windbreak.ledger.rebuild import canary_status_read_model

    assert canary_status_read_model([]) == []


def test_canary_status_read_model_latest_per_provider_wins_on_duplicate(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """A provider ledgered more than once keeps only its LATEST verdict, at
    that provider's first-seen position; a distinct provider is unaffected.
    """
    from windbreak.ledger.events import CanaryVerdictRecorded
    from windbreak.ledger.rebuild import canary_status_read_model
    from windbreak.ledger.store import SqliteLedgerStore

    store = SqliteLedgerStore(tmp_path / "ledger.db", now=deterministic_clock)
    store.append(
        CanaryVerdictRecorded(
            component="scheduler",
            **_sample_verdict_kwargs(provider="futuresearch", status="OK"),
        )
    )
    store.append(
        CanaryVerdictRecorded(
            component="scheduler",
            **_sample_verdict_kwargs(provider="anthropic", status="OK"),
        )
    )
    store.append(
        CanaryVerdictRecorded(
            component="scheduler",
            **_sample_verdict_kwargs(
                provider="futuresearch",
                status="ANSWER_DRIFT",
                drift_kind="answer",
                drift_score_ppm=90_000,
            ),
        )
    )
    store.verify_chain()
    records: list[LedgerRecord] = store.read_all()
    store.close()

    rows = canary_status_read_model(records)

    assert [row["data"]["provider"] for row in rows] == ["futuresearch", "anthropic"]
    assert [row["seq"] for row in rows] == [3, 2]
    assert rows[0]["data"]["status"] == "ANSWER_DRIFT"
    assert rows[0]["data"]["drift_score_ppm"] == 90_000
    assert rows[1]["data"]["status"] == "OK"


# --- forecasts_read_model: pure fold, every ForecastCreated row ---------------


def test_forecasts_read_model_empty_input_returns_empty_list() -> None:
    """No records at all yields an empty list, not an error."""
    from windbreak.ledger.rebuild import forecasts_read_model

    assert forecasts_read_model([]) == []


def test_forecasts_read_model_projects_every_forecast_created_row_in_ledger_order(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """Every `ForecastCreated` row is projected, in ledger order, unaffected
    by an unrelated event mixed into the same ledger.
    """
    from windbreak.ledger.events import ForecastCreated, ModeHeartbeat
    from windbreak.ledger.rebuild import forecasts_read_model
    from windbreak.ledger.store import SqliteLedgerStore

    store = SqliteLedgerStore(tmp_path / "ledger.db", now=deterministic_clock)
    store.append(
        ForecastCreated(
            component="scheduler",
            forecast_id="fc-0001",
            market_ticker="MKT-DEEP",
            probability_ppm=500_000,
            eligible_for_live=True,
            abstention_reason=None,
            research_cost_micros=1_000_000,
            market_price_baseline_pips=4_500,
        )
    )
    store.append(ModeHeartbeat(component="scheduler", mode="PAPER", beat=1))
    store.append(
        ForecastCreated(
            component="scheduler",
            forecast_id="fc-0002",
            market_ticker="MKT-DEEP",
            probability_ppm=600_000,
            eligible_for_live=False,
            abstention_reason="insufficient_citations",
            research_cost_micros=2_000_000,
            market_price_baseline_pips=4_600,
        )
    )
    store.verify_chain()
    records: list[LedgerRecord] = store.read_all()
    store.close()

    rows = forecasts_read_model(records)

    assert [row["seq"] for row in rows] == [1, 3]
    assert [row["data"]["forecast_id"] for row in rows] == ["fc-0001", "fc-0002"]
    assert rows[1]["data"]["abstention_reason"] == "insufficient_citations"
    assert all("created_at" in row for row in rows)


# --- rebuild(): both projections are wired in ----------------------------------


def test_rebuild_writes_canary_status_and_forecasts_read_models(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`rebuild()` writes both `canary_status.json` and `forecasts.json`,
    each holding the rows their dedicated projection function would produce.
    """
    import json

    from windbreak.ledger.events import CanaryVerdictRecorded, ForecastCreated
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(
        ForecastCreated(
            component="scheduler",
            forecast_id="fc-0001",
            market_ticker="MKT-DEEP",
            probability_ppm=500_000,
            eligible_for_live=True,
            abstention_reason=None,
            research_cost_micros=1_000_000,
            market_price_baseline_pips=4_500,
        )
    )
    store.append(
        CanaryVerdictRecorded(
            component="scheduler",
            **_sample_verdict_kwargs(provider="futuresearch", status="OK"),
        )
    )
    store.close()

    rebuild(db_path, output_dir)

    forecasts = json.loads((output_dir / "forecasts.json").read_text())
    assert [entry["data"]["forecast_id"] for entry in forecasts] == ["fc-0001"]

    canary_status = json.loads((output_dir / "canary_status.json").read_text())
    assert [entry["data"]["provider"] for entry in canary_status] == ["futuresearch"]
    assert canary_status[0]["data"]["status"] == "OK"
