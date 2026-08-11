"""The provider gate's bootstrap is transient now, not terminal (issue #440).

PR #481 mapped this as barrier 5 between the shipped PAPER loop and one order
intent: `_build_provider_gate` reads `provider-track-records.json` and never
recomputes it, and nothing in the repository wrote that file, so every provider
was unproven forever and `config.forecast.provider_gate.min_resolved` could
never be approached.

This suite proves the end state through the *real composition* rather than a
seam beside it (the composition trap): a real hash-chained ledger carrying real
`ForecastCreated` / `ProviderVoteRecorded` rows and a real operator-ingested
`MarketResolved` row, folded by the real `windbreak evaluate-providers` CLI
verb into a real artifact, read back by the real `_build_provider_gate`.

Four states are pinned, and they are deliberately not symmetric:

1. **Bootstrap** -- no artifact yet, the provider is held.
2. **Proven** -- the verb runs over sufficient evidence and the *same*
   `_build_provider_gate` now admits the provider.
3. **Real-but-insufficient** -- the verb writes exact, honest values that miss a
   bar, and the provider is held *by the bar* rather than by the record's
   absence. PR #481 found this is the only way either bar can be tested at all:
   deleting the artifact leaves a provider unproven whatever the bar says, so
   the bars themselves go untested.
4. **Evidence removed** -- fold a ledger with the settlement row gone and the
   artifact goes back to `{}`, so the transition in (2) is attributable to the
   ingested resolution and to nothing else.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from windbreak.config.schema import (
    ForecastConfig,
    ProviderGateConfig,
    WindbreakConfig,
)
from windbreak.evaluation.ingest import MarketResolved
from windbreak.evaluation.resolution import ResolutionOutcome
from windbreak.evaluation.track_records import PROVIDER_TRACK_RECORD_FILENAME
from windbreak.forecast.pipeline import VOTE_OUTCOME_VOTED
from windbreak.ledger.events import ConfigLoaded, ForecastCreated, ProviderVoteRecorded
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.main import main
from windbreak.scheduler.loop import _build_provider_gate

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from windbreak.ledger.events import Event

#: The component label stamped on every event this suite ledgers.
_COMPONENT = "scheduler"

#: The instant the scripted ledger clock starts from.
_BASE = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)

#: The provider whose track record this suite drives from held to proven.
_PROVIDER = "openai"

#: The forecast probability every scored forecast here carries, in ppm.
_PROBABILITY_PPM = 250_000

#: The recorded baseline executable price, in pips (460_000 ppm).
_BASELINE_PIPS = 4600

#: Exact ppm-per-pip factor, restated so the expectation below is arithmetic
#: done here rather than a value read back off the metric under test.
_PPM_PER_PIP = 100

#: One whole probability in ppm.
_PPM_SCALE = 1_000_000


def _expected_skill_ppm() -> int:
    """Compute the Brier skill every forecast in this suite earns, in ppm.

    Each forecast carries the same probability and baseline and settles ``NO``,
    so the pooled skill equals the single-forecast skill. Derived here from the
    probability, the baseline price and the ingested outcome -- never read back
    off the artifact or the gate this suite checks (#422).

    Returns:
        The Brier skill the artifact must record, in ppm.
    """
    forecast_term = (_PROBABILITY_PPM - 0) ** 2
    baseline_term = (_BASELINE_PIPS * _PPM_PER_PIP - 0) ** 2
    return (baseline_term - forecast_term) * _PPM_SCALE // baseline_term


#: The exact skill the artifact must record for `_PROVIDER`.
_EXPECTED_SKILL_PPM = _expected_skill_ppm()


def _stepping_instants() -> Iterator[datetime]:
    """Yield ascending one-second-apart instants, one per ledger append.

    Yields:
        Each `created_at` instant in append order.
    """
    step = 0
    while True:
        yield _BASE + timedelta(seconds=step)
        step += 1


def _config(*, min_resolved: int, min_brier_skill_ppm: int) -> WindbreakConfig:
    """Return a config whose provider-gate bars are set explicitly.

    Args:
        min_resolved: The resolved-forecast bar to put in force.
        min_brier_skill_ppm: The Brier-skill bar, in ppm, to put in force.

    Returns:
        The constructed configuration.
    """
    return dataclasses.replace(
        WindbreakConfig(),
        forecast=ForecastConfig(
            provider_gate=ProviderGateConfig(
                min_resolved=min_resolved,
                min_brier_skill_ppm=min_brier_skill_ppm,
            )
        ),
    )


def _write_ledger(db_path: Path, events: list[Event]) -> None:
    """Append `events` to a fresh ledger stamping ascending instants.

    Args:
        db_path: Where the SQLite ledger database is created.
        events: The events to append, in order.
    """
    clock = _stepping_instants()
    store = SqliteLedgerStore(db_path, now=lambda: next(clock))
    try:
        for event in events:
            store.append(event)
    finally:
        store.close()


def _scored_run_events(*, forecast_count: int) -> list[Event]:
    """Build the event stream one PAPER run plus one operator ingest leaves.

    Each forecast is on its own market so no observation window collapses two
    of them into one, and every market settles ``NO`` at an instant after every
    forecast row -- so no forecast is backdated and all of them score.

    Args:
        forecast_count: How many forecasts (and markets) to produce.

    Returns:
        The events, in append order.
    """
    events: list[Event] = [
        ConfigLoaded(component=_COMPONENT, config_hash="deadbeef", diff={})
    ]
    for index in range(forecast_count):
        ticker = f"M{index}"
        events.append(
            ForecastCreated(
                component=_COMPONENT,
                forecast_id=f"f{index}",
                market_ticker=ticker,
                probability_ppm=_PROBABILITY_PPM,
                eligible_for_live=False,
                abstention_reason=None,
                research_cost_micros=1_000_000,
                market_price_baseline_pips=_BASELINE_PIPS,
            )
        )
        events.append(
            ProviderVoteRecorded(
                component=_COMPONENT,
                forecast_id=f"f{index}",
                market_ticker=ticker,
                provider=_PROVIDER,
                model_version="v1",
                vote_index=0,
                cost_micros=1_000,
                outcome=VOTE_OUTCOME_VOTED,
                failure_code="",
            )
        )
    # Every settlement instant is at or after the last forecast row's instant,
    # so the projection places each market's answer past every forecast.
    settled_at = _BASE + timedelta(seconds=len(events))
    for index in range(forecast_count):
        events.append(
            MarketResolved(
                component="cli",
                market_ticker=f"M{index}",
                outcome=ResolutionOutcome.NO,
                resolved_at=settled_at,
                source="test settlement notice",
            )
        )
    return events


def _run_verb(ledger_path: Path, report_dir: Path) -> int:
    """Run the `evaluate-providers` CLI verb over a ledger and report dir.

    Args:
        ledger_path: The ledger database to fold.
        report_dir: The evaluation-artifact directory to write into.

    Returns:
        The verb's process exit code.
    """
    return main(
        [
            "evaluate-providers",
            "--ledger-path",
            str(ledger_path),
            "--report-dir",
            str(report_dir),
        ]
    )


def _artifact(report_dir: Path) -> dict[str, dict[str, int]]:
    """Read the written track-record artifact back as parsed JSON.

    Args:
        report_dir: The evaluation-artifact directory.

    Returns:
        The artifact's decoded contents.
    """
    text = (report_dir / PROVIDER_TRACK_RECORD_FILENAME).read_text(encoding="utf-8")
    decoded: dict[str, dict[str, int]] = json.loads(text)
    return decoded


def test_the_gate_opens_only_after_the_producer_folds_real_resolved_evidence(
    tmp_path: Path,
) -> None:
    """Held with no artifact; proven once the verb folds the settled forecasts.

    The same `_build_provider_gate` is asked twice over the same directory, so
    the only thing that changed between the two verdicts is the artifact the
    producer wrote from the ledger's own resolved outcomes.
    """
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    ledger_path = tmp_path / "ledger.db"
    _write_ledger(ledger_path, _scored_run_events(forecast_count=3))
    config = _config(min_resolved=3, min_brier_skill_ppm=_EXPECTED_SKILL_PPM)

    bootstrap_gate = _build_provider_gate(report_dir, config)
    assert bootstrap_gate.is_provider_proven(_PROVIDER) is False
    assert not (report_dir / PROVIDER_TRACK_RECORD_FILENAME).exists()

    assert _run_verb(ledger_path, report_dir) == 0

    assert _artifact(report_dir) == {
        _PROVIDER: {
            "provider": _PROVIDER,
            "resolved_count": 3,
            "brier_skill_ppm": _EXPECTED_SKILL_PPM,
        }
    }
    assert _build_provider_gate(report_dir, config).is_provider_proven(_PROVIDER)


def test_a_real_but_insufficient_record_is_held_by_the_resolved_count_bar(
    tmp_path: Path,
) -> None:
    """The record exists and carries exact values; the count bar holds it."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    ledger_path = tmp_path / "ledger.db"
    _write_ledger(ledger_path, _scored_run_events(forecast_count=3))
    assert _run_verb(ledger_path, report_dir) == 0

    assert _artifact(report_dir)[_PROVIDER]["resolved_count"] == 3
    gate = _build_provider_gate(
        report_dir, _config(min_resolved=4, min_brier_skill_ppm=_EXPECTED_SKILL_PPM)
    )
    assert gate.is_provider_proven(_PROVIDER) is False
    assert gate.unproven_providers([_PROVIDER]) == (_PROVIDER,)


def test_a_real_but_insufficient_record_is_held_by_the_brier_skill_bar(
    tmp_path: Path,
) -> None:
    """The count clears; the skill bar one ppm above the earned skill holds it."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    ledger_path = tmp_path / "ledger.db"
    _write_ledger(ledger_path, _scored_run_events(forecast_count=3))
    assert _run_verb(ledger_path, report_dir) == 0

    assert _artifact(report_dir)[_PROVIDER]["brier_skill_ppm"] == _EXPECTED_SKILL_PPM
    gate = _build_provider_gate(
        report_dir, _config(min_resolved=3, min_brier_skill_ppm=_EXPECTED_SKILL_PPM + 1)
    )
    assert gate.is_provider_proven(_PROVIDER) is False


def test_removing_the_ingested_settlement_closes_the_gate_again(
    tmp_path: Path,
) -> None:
    """Fold the same run without its settlements and the gate shuts."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    config = _config(min_resolved=3, min_brier_skill_ppm=_EXPECTED_SKILL_PPM)

    settled_path = tmp_path / "settled.db"
    _write_ledger(settled_path, _scored_run_events(forecast_count=3))
    assert _run_verb(settled_path, report_dir) == 0
    assert _build_provider_gate(report_dir, config).is_provider_proven(_PROVIDER)

    unsettled_path = tmp_path / "unsettled.db"
    _write_ledger(
        unsettled_path,
        [
            event
            for event in _scored_run_events(forecast_count=3)
            if not isinstance(event, MarketResolved)
        ],
    )
    assert _run_verb(unsettled_path, report_dir) == 0

    assert _artifact(report_dir) == {}
    assert _build_provider_gate(report_dir, config).is_provider_proven(_PROVIDER) is (
        False
    )


def test_the_verb_refuses_a_report_dir_that_does_not_exist(tmp_path: Path) -> None:
    """A mistyped report dir is refused, not created behind the loop's back."""
    ledger_path = tmp_path / "ledger.db"
    _write_ledger(ledger_path, _scored_run_events(forecast_count=1))
    missing = tmp_path / "nope"

    assert _run_verb(ledger_path, missing) == 1
    assert not missing.exists()


def test_the_verb_refuses_a_ledger_path_that_does_not_exist(tmp_path: Path) -> None:
    """A mistyped ledger path is refused rather than folded as an empty run.

    Opening a `SqliteLedgerStore` on a missing path creates an empty database,
    which folds to `{}` -- a cheerful exit 0 that would *shut* a gate the
    operator believed they were opening, and would write the artifact from a
    ledger that is not the loop's.
    """
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    assert _run_verb(tmp_path / "absent.db", report_dir) == 1
    assert not (report_dir / PROVIDER_TRACK_RECORD_FILENAME).exists()
    assert not (tmp_path / "absent.db").exists()
