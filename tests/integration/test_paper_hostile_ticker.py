"""A hostile market ticker never reaches the hash chain via `ForecastCreated`.

Issue #525, RED. `windbreak/scheduler/loop.py::_forecast_stage` appended
`ForecastCreated(market_ticker=forecast.market_ticker)` straight off the record,
and that record's ticker was `market.ticker` verbatim
(`windbreak/forecast/pipeline.py::build_forecast_record`). Nothing screened
between them, so a market whose ticker forges an untrusted-data delimiter landed
unaltered in an **append-only** chain -- permanently, on the route every
successful forecast takes rather than only on a discard.

Everything here drives the *shipped* composition (`build_paper_deps` ->
`run_single_tick`) over a generated two-market universe, never a hand-built
`ForecastCreated`. That is deliberate: the defect was a wiring defect, so a test
that constructs the event itself would pass over code the loop never runs.

The universe is generated into `tmp_path` from the committed
`two_ticker_isolation` fixture rather than committed as a fixture of its own, so
no forged delimiter bytes enter the repository.

Ticker order is load-bearing. `screen_universe` walks `sorted(..., key=ticker)`
and `_HOSTILE_TICKER` begins with `<` (0x3C), which sorts *before* the clean
control's `M` (0x4D). The refused market is therefore reached first, so the
positive control -- the clean ticker being present in the chain -- doubles as
proof that refusing one market did not stop the walk.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from typing import TYPE_CHECKING

import pytest

from tests.integration.conftest import (
    FIXED_NOW_EPOCH_S,
    FIXTURE_SCREENER_CONFIG,
    ledger_path_for,
)
from windbreak.config.schema import (
    CapitalConfig,
    CorrelationConfig,
    OpsConfig,
    RiskConfig,
    WindbreakConfig,
)
from windbreak.forecast.sanitize import DATA_BLOCK_BEGIN

if TYPE_CHECKING:
    from pathlib import Path

#: The clean control market, kept verbatim from the committed fixture. Its
#: presence in the swept corpus is what proves the sweep is looking at real rows
#: rather than at nothing at all.
_CLEAN_TICKER = "MKT-ISO-A"

#: The fixture market whose ticker is rewritten hostile. Its book and metadata
#: are otherwise untouched, so it screens *in* exactly as the clean one does --
#: the difference under test is the ticker's bytes and nothing else.
_REWRITTEN_TICKER = "MKT-ISO-B"

#: The distinguishing substring of the hostile ticker, chosen so it appears
#: nowhere else in the corpus. Asserting on this rather than on the whole ticker
#: catches a *partially* scrubbed value too: a ledgered ticker that dropped the
#: delimiter but kept the attacker's own text is still attacker text on an
#: unredactable chain.
_HOSTILE_MARKER = "MKT-EVIL-9F3"

#: The hostile ticker itself: a forged untrusted-data delimiter spliced onto an
#: otherwise ticker-shaped string. Sorts before `_CLEAN_TICKER` (see the module
#: docstring) so the refused market is walked first.
_HOSTILE_TICKER = f"{DATA_BLOCK_BEGIN} {_HOSTILE_MARKER}"

#: The shape a ledgered `forecast_id` must have: a lowercase sha256 digest.
#: Pinned so the byte-identity assertion's derived `forecast_id` is still a
#: real claim about the value rather than a hole in the comparison.
_FORECAST_ID_SHAPE = re.compile(r"[0-9a-f]{64}")


def _fixed_clock() -> int:
    """Return the fixed, non-advancing epoch second every tick here runs at."""
    return FIXED_NOW_EPOCH_S


def _hostile_books_dir(source: Path, destination: Path) -> Path:
    """Copy the two-market fixture, rewriting one ticker to a hostile one.

    Args:
        source: The committed `two_ticker_isolation` books-fixture directory.
        destination: The `tmp_path` directory to build the universe in.

    Returns:
        The generated books directory.
    """
    shutil.copytree(source, destination)
    markets_path = destination / "markets.json"
    markets = json.loads(markets_path.read_text(encoding="utf-8"))
    for market in markets:
        if market["ticker"] == _REWRITTEN_TICKER:
            market["ticker"] = _HOSTILE_TICKER
    markets_path.write_text(json.dumps(markets), encoding="utf-8")
    sessions_path = destination / "sessions.json"
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    sessions[_HOSTILE_TICKER] = sessions.pop(_REWRITTEN_TICKER)
    sessions_path.write_text(json.dumps(sessions), encoding="utf-8")
    return destination


def _config(state_dir: Path) -> WindbreakConfig:
    """Build the PAPER-ceilinged config these scenarios tick under.

    Args:
        state_dir: The per-test kill/re-arm state directory.

    Returns:
        The assembled configuration.
    """
    return WindbreakConfig(
        mode_ceiling="paper",
        capital=CapitalConfig(floor_micros=0),
        risk=RiskConfig(),
        screener=FIXTURE_SCREENER_CONFIG,
        correlation=CorrelationConfig(),
        ops=OpsConfig(state_dir=str(state_dir)),
    )


def _rows_from_disk(ledger_path: Path) -> list[tuple[str, str]]:
    """Read every persisted row back through a *fresh* connection.

    Read across `ledger.db` **and** its `-wal` sidecar: the store runs
    WAL-journaled, so freshly appended rows live in the sidecar until a
    checkpoint. Slurping `ledger.db` alone sweeps a corpus missing exactly the
    rows under test and passes forever -- the false green PR #474 shipped.
    Opening a second SQLite connection reads both, which is why the reader is
    SQLite rather than `read_bytes`.

    Args:
        ledger_path: The `ledger.db` path the store was created at.

    Returns:
        One `(event_type, payload_json)` pair per row, in ledger order.
    """
    connection = sqlite3.connect(ledger_path)
    try:
        return [
            (str(event_type), str(payload_json))
            for event_type, payload_json in connection.execute(
                "SELECT event_type, payload_json FROM ledger ORDER BY sequence_number"
            ).fetchall()
        ]
    finally:
        connection.close()


@pytest.fixture
def hostile_universe(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
):
    """Run one shipped PAPER tick over a universe holding a hostile ticker.

    Args:
        two_ticker_books_dir: The committed two-market books fixture.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        research_tools_factory: Builds the offline research-tools double.
        tmp_path: The pytest scratch directory.

    Returns:
        A `(deps, outcome, rows)` triple: the wired bundle, the tick's outcome,
        and every ledger row read back from disk across `ledger.db*`.
    """
    from windbreak.scheduler.loop import build_paper_deps, run_single_tick

    ledger_path = ledger_path_for(tmp_path)
    deps = build_paper_deps(
        books_dir=_hostile_books_dir(two_ticker_books_dir, tmp_path / "books"),
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
        config=_config(tmp_path / "state"),
        research_tools=research_tools_factory(),
        clock=_fixed_clock,
    )
    outcome = run_single_tick(deps, beat=1)
    deps.store.verify_chain()

    # Read while the store is still open, so the read genuinely crosses
    # `ledger.db` and the `-wal` sidecar the fresh rows are still sitting in.
    sidecar = ledger_path.with_name(f"{ledger_path.name}-wal")
    assert sidecar.exists()
    assert _CLEAN_TICKER.encode() in sidecar.read_bytes()
    assert _CLEAN_TICKER.encode() not in ledger_path.read_bytes()
    rows = _rows_from_disk(ledger_path)

    deps.store.close()
    return deps, outcome, rows


def test_no_forecast_created_row_carries_the_hostile_ticker(hostile_universe) -> None:
    """The hostile ticker's bytes are in no `ForecastCreated` row on the chain.

    The positive control is inside the same corpus the sweep scans: the clean
    ticker must be found in a `ForecastCreated` row, so a sweep over an empty
    or wrong corpus cannot pass.
    """
    _deps, _outcome, rows = hostile_universe

    forecast_rows = [
        payload for event_type, payload in rows if event_type == "ForecastCreated"
    ]

    # Positive control first: the corpus is real and the row kind exists in it.
    assert forecast_rows
    assert any(_CLEAN_TICKER in payload for payload in forecast_rows)

    # The sweep itself.
    assert not any(_HOSTILE_MARKER in payload for payload in forecast_rows)
    assert not any(DATA_BLOCK_BEGIN in payload for payload in forecast_rows)


def test_refusing_one_market_does_not_stop_the_universe_walk(hostile_universe) -> None:
    """The clean market behind the refused one is still forecast, and no halt.

    The refused market is walked *first* (see the module docstring), so this is
    what separates "refuse and skip" from "refuse and stop": a refusal routed
    through `_run_universe`'s budget-halt door would break the walk and the
    clean market -- the only forecast this tick produces -- would never happen.
    """
    _deps, outcome, _rows = hostile_universe

    assert outcome.research_halted is False
    assert len(outcome.forecast_ids) == 1
    assert outcome.candidate_tickers == (_HOSTILE_TICKER, _CLEAN_TICKER)


def test_the_rows_that_still_carry_a_hostile_ticker_are_exactly_the_known_two(
    hostile_universe,
) -> None:
    """State the residual, derived from the corpus rather than hand-restated.

    After this fix, no forecast-stage row can carry an unscreened ticker. Two
    *earlier* stages still can, and both are outside issue #525's lane: the
    screener's own `ScreenDecisionRecorded` and the loop's
    `MarketSnapshotRecorded`, each appended before the forecast stage is
    reached. They are filed separately rather than fixed here.

    The set below is computed from the swept rows, so the day a new row kind
    starts carrying attacker text this fails -- which a hand-written list of
    "paths that are fine" could never do.
    """
    _deps, _outcome, rows = hostile_universe

    leaking_types = {
        event_type for event_type, payload in rows if _HOSTILE_MARKER in payload
    }

    assert leaking_types == {"ScreenDecisionRecorded", "MarketSnapshotRecorded"}


def test_a_clean_ticker_still_ledgers_byte_identically(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A clean universe's `ForecastCreated` payloads are unchanged, in full.

    Full equality on every key of both rows, not "no exception raised": the
    screen must add no artefact, drop no key, and change no value on the path
    every ordinary market takes. Each literal below was captured from a run of
    the pre-fix code.

    `forecast_id` is the one field asserted against a *derived* expectation
    rather than a literal, and deliberately so on two counts. A ledgered id is
    a bare 64-character digest, which `detect-secrets` flags as a high-entropy
    string, and pasting one into a test would launder that hook rather than
    satisfy it. It also makes the assertion stronger: the row's id must equal
    the id `run_single_tick` *reported*, which reaches the caller by a wholly
    different route (`_run_candidate`'s return value) than the payload does, so
    a fix that ledgered one market's id under another's would fail here. Its
    shape and distinctness are pinned separately, so "derived" never degrades
    into "unasserted".
    """
    from windbreak.scheduler.loop import build_paper_deps, run_single_tick

    ledger_path = ledger_path_for(tmp_path)
    deps = build_paper_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
        config=_config(tmp_path / "state"),
        research_tools=research_tools_factory(),
        clock=_fixed_clock,
    )

    outcome = run_single_tick(deps, beat=1)
    deps.store.verify_chain()
    rows = _rows_from_disk(ledger_path)
    deps.store.close()

    payloads = [
        json.loads(payload)["data"]
        for event_type, payload in rows
        if event_type == "ForecastCreated"
    ]

    first_id, second_id = outcome.forecast_ids
    assert first_id != second_id
    assert all(_FORECAST_ID_SHAPE.fullmatch(value) for value in (first_id, second_id))
    assert payloads == [
        {
            "forecast_id": first_id,
            "market_ticker": "MKT-ISO-A",
            "probability_ppm": 440_000,
            "eligible_for_live": False,
            "abstention_reason": "no_verified_citations",
            "research_cost_micros": 3_000_000,
            "market_price_baseline_pips": 4_400,
        },
        {
            "forecast_id": second_id,
            "market_ticker": "MKT-ISO-B",
            "probability_ppm": 340_000,
            "eligible_for_live": False,
            "abstention_reason": "no_verified_citations",
            "research_cost_micros": 3_000_000,
            "market_price_baseline_pips": 3_400,
        },
    ]
