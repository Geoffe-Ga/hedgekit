"""End-to-end tests for `windbreak correct-resolution` (issue #484).

Everything here drives the **real** CLI over a **real** `SqliteLedgerStore` on
disk and folds the result through the **real**
`windbreak.scheduler.weekly_data.weekly_report_body`. That is deliberate and it
is the point of the module. Three defects in this area survived a full suite of
green unit tests because each one was tested at a seam: a parser test proves
parsing, a fold test proves folding, and neither proves that the verb an
operator types moves the number the weekly report prints. The composition is
what was broken, so the composition is what is asserted.

WHAT THE VERB IS FOR

`ingest-resolution` (#439) is operator-typed, and a single wrong `--outcome`
exits 0. Nothing contradicts it, so #482's guard -- which refuses a
*conflicting* re-ingest -- never fires, and the ledger is append-only: the bad
settlement stood forever and every metric folded from it was wrong.
`correct-resolution` appends one `SettlementReversed` row that supersedes the
row it names. Nothing is redacted; the wrong claim stays on the chain and is
named in the rendered report.

THE NEGATIVE IS ASSERTED, NOT ASSUMED

`test_the_corrected_outcome_moves_the_metric_and_the_uncorrected_one_does_not`
folds the identical records twice -- once whole, once with only the
`SettlementReversed` row removed -- and asserts two *different* exact Brier
values. A correction whose outcome scored the same as the original would be a
green test proving nothing (proven trap 3), so the two values are 562_500 and
62_500 and the test asserts they differ.

REFUSALS ARE AT THE VERB, NEVER AT THE FOLD

#482's lesson: a fail-closed guard that fires on a non-contradiction and cannot
be recovered from is worse than the gap it closes -- that guard is why this
issue exists. Every refusal below exits 1 and writes **nothing**, so the
operator retypes one command and the loop never stops beating. And a market can
be corrected more than once
(`test_a_second_correction_names_the_first_and_moves_the_metric_again`),
because a correction path that works exactly once is the same permanent trap
one command later.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.evaluation.ingest import (
    MARKET_RESOLVED_EVENT_TYPE,
    SETTLEMENT_REVERSED_EVENT_TYPE,
    MarketResolved,
)
from windbreak.evaluation.resolution import ResolutionOutcome
from windbreak.ledger.events import ConfigLoaded, ForecastCreated
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.main import build_parser, main
from windbreak.scheduler.weekly_data import weekly_report_body

if TYPE_CHECKING:
    from windbreak.ledger.store import LedgerRecord

#: The market every test here settles and then corrects.
_TICKER = "MKT-A"

#: The report date stamped into every rendered body.
_TODAY = date(2026, 3, 8)

#: The forecast probability, in ppm, every seeded forecast carries.
_PROBABILITY_PPM = 250_000

#: Mean Brier of one `_PROBABILITY_PPM` forecast on a market that settled `no`:
#: `(250_000 - 0)^2 / 1_000_000`, computed from the forecast's probability and
#: the outcome rather than read back off the report the assertion checks.
_BRIER_IF_NO = (_PROBABILITY_PPM - 0) ** 2 // 1_000_000

#: The same forecast scored against `yes`: `(250_000 - 1_000_000)^2 / 1_000_000`.
_BRIER_IF_YES = (_PROBABILITY_PPM - 1_000_000) ** 2 // 1_000_000

#: The provenance label the first (wrong) ingest carries.
_ORIGINAL_SOURCE = "kalshi settlement notice 2026-03-01"

#: The provenance label the correction carries. Distinct from
#: `_ORIGINAL_SOURCE` so a raw byte scan of the database files can tell which
#: of the two rows it is looking at.
_CORRECTION_SOURCE = "kalshi settlement correction notice 2026-03-05"


def _base_instant() -> datetime:
    """Return the fixed instant every seeded row and settlement is offset from.

    Anchored 30 days in the past rather than at a hardcoded calendar date
    because the CLI stamps its own rows with the real clock: the seeded rows
    must sort *before* the rows `main()` appends, or the temporal projection
    would be measuring the test harness rather than the gate. Whole seconds, so
    every derived instant renders exactly.

    Returns:
        The base instant, timezone-aware and truncated to the second.
    """
    return (datetime.now(UTC) - timedelta(days=30)).replace(microsecond=0)


def _seed(ledger_path: Path, *, forecast_at: datetime, deployed_at: datetime) -> None:
    """Seed a ledger with a deployment marker and one forecast, at fixed stamps.

    Args:
        ledger_path: The ledger database to create.
        forecast_at: The `created_at` stamped on the `ForecastCreated` row.
        deployed_at: The `created_at` stamped on the `ConfigLoaded` marker.
    """
    stamps = iter([deployed_at, forecast_at])
    store = SqliteLedgerStore(ledger_path, now=lambda: next(stamps))
    try:
        store.append(
            ConfigLoaded(component="scheduler", config_hash="deadbeef", diff={})
        )
        store.append(
            ForecastCreated(
                component="scheduler",
                forecast_id="fc-a",
                market_ticker=_TICKER,
                probability_ppm=_PROBABILITY_PPM,
                eligible_for_live=False,
                abstention_reason=None,
                research_cost_micros=1_000_000,
                market_price_baseline_pips=4600,
            )
        )
        store.verify_chain()
    finally:
        store.close()


def _ingest_argv(
    ledger_path: Path, *, outcome: str, resolved_at: datetime, ticker: str = _TICKER
) -> list[str]:
    """Build a complete `ingest-resolution` argument vector.

    Args:
        ledger_path: The ledger database the verb appends into.
        outcome: The `--outcome` token.
        resolved_at: The settlement instant.
        ticker: The `--market-ticker` value.

    Returns:
        The argument vector for `main()`.
    """
    return [
        "ingest-resolution",
        "--ledger-path",
        str(ledger_path),
        "--market-ticker",
        ticker,
        "--outcome",
        outcome,
        "--resolved-at",
        resolved_at.isoformat(),
        "--source",
        _ORIGINAL_SOURCE,
    ]


def _correct_argv(
    ledger_path: Path,
    *,
    superseded: int,
    outcome: str,
    resolved_at: datetime,
    ticker: str = _TICKER,
    source: str = _CORRECTION_SOURCE,
) -> list[str]:
    """Build a complete `correct-resolution` argument vector.

    Args:
        ledger_path: The ledger database the verb appends into.
        superseded: The `--superseded-sequence-number` value.
        outcome: The corrected `--outcome` token.
        resolved_at: The corrected settlement instant.
        ticker: The `--market-ticker` value.
        source: The `--source` provenance of the correction.

    Returns:
        The argument vector for `main()`.
    """
    return [
        "correct-resolution",
        "--ledger-path",
        str(ledger_path),
        "--market-ticker",
        ticker,
        "--superseded-sequence-number",
        str(superseded),
        "--outcome",
        outcome,
        "--resolved-at",
        resolved_at.isoformat(),
        "--source",
        source,
    ]


def _rows(ledger_path: Path) -> list[tuple[str, str]]:
    """Read every ledger row's event type and envelope straight from SQLite.

    Args:
        ledger_path: The ledger database to read.

    Returns:
        One `(event_type, payload_json)` pair per row, in sequence order.
    """
    conn = sqlite3.connect(ledger_path)
    try:
        return [
            (str(row[0]), str(row[1]))
            for row in conn.execute(
                "SELECT event_type, payload_json FROM ledger ORDER BY sequence_number"
            )
        ]
    finally:
        conn.close()


def _records(ledger_path: Path) -> list[LedgerRecord]:
    """Read the full ledger back through the real store, verifying the chain.

    Args:
        ledger_path: The ledger database to read.

    Returns:
        Every persisted record, in ascending sequence order.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        return store.read_all()
    finally:
        store.close()


def _on_disk_bytes(ledger_path: Path) -> bytes:
    """Concatenate every `ledger.db*` file's bytes, WAL sidecar included.

    A `ledger.db`-only read is a false green: SQLite's WAL journal keeps freshly
    committed rows in the `-wal` sidecar until a checkpoint, so the newest row
    -- exactly the one under test -- can be absent from the main database file.

    Args:
        ledger_path: The ledger database path (its siblings are globbed).

    Returns:
        Every matching file's contents, concatenated.
    """
    sidecars = sorted(ledger_path.parent.glob(f"{ledger_path.name}*"))
    assert ledger_path in sidecars
    raw = b"".join(path.read_bytes() for path in sidecars)
    assert raw != b""
    return raw


def _fatal_messages(captured: str) -> str:
    """Decode the `msg` field of every structured log line on stderr.

    Args:
        captured: The captured stderr text.

    Returns:
        The decoded messages, newline-joined.
    """
    return "\n".join(
        str(json.loads(line).get("msg", "")) for line in captured.splitlines() if line
    )


def _metric_line(body: str, name: str) -> str:
    """Return the single rendered line for one metric name.

    Args:
        body: The rendered weekly-report body.
        name: The metric's registry name.

    Returns:
        The one matching `name [window] = value` line.

    Raises:
        AssertionError: If the body does not carry exactly one such line.
    """
    matches = [line for line in body.splitlines() if line.startswith(f"{name} [")]
    assert len(matches) == 1, f"expected exactly one {name!r} line, got {matches}"
    return matches[0]


def _rejection_lines(body: str) -> list[str]:
    """Return every rendered temporal-integrity rejection line.

    Args:
        body: The rendered weekly-report body.

    Returns:
        The rejection ledger's lines, in render order.
    """
    return [
        line
        for line in body.splitlines()
        if line.startswith("EVALUATION_RECORD_REJECTED")
    ]


# ---------------------------------------------------------------------------
# 1. The argument surface: every flag required, nothing defaulted.
# ---------------------------------------------------------------------------


def test_correct_resolution_parses_all_six_required_arguments() -> None:
    """The verb parses its path, ticker, superseded row, outcome, instant, source."""
    moment = datetime(2026, 3, 2, 9, 30, tzinfo=UTC)
    args = build_parser().parse_args(
        _correct_argv(
            Path("ledger.db"), superseded=3, outcome="yes", resolved_at=moment
        )
    )

    assert args.command == "correct-resolution"
    assert args.market_ticker == _TICKER
    assert args.superseded_sequence_number == 3
    assert args.outcome == "yes"
    assert args.resolved_at == "2026-03-02T09:30:00+00:00"
    assert args.source == _CORRECTION_SOURCE


@pytest.mark.parametrize(
    "dropped",
    [
        "--ledger-path",
        "--market-ticker",
        "--superseded-sequence-number",
        "--outcome",
        "--resolved-at",
        "--source",
    ],
)
def test_correct_resolution_requires_every_argument(
    dropped: str, tmp_path: Path
) -> None:
    """Omitting any one argument is a usage error, never a defaulted correction.

    Args:
        dropped: The option (and its value) removed from the argument vector.
        tmp_path: Pytest's per-test temporary directory.
    """
    argv = _correct_argv(
        tmp_path / "ledger.db",
        superseded=1,
        outcome="yes",
        resolved_at=datetime(2026, 3, 2, 9, 30, tzinfo=UTC),
    )
    index = argv.index(dropped)
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)

    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 2. The append: one row, distinguishable from a first ingest, chain intact.
# ---------------------------------------------------------------------------


def test_a_correction_appends_one_reversal_row_naming_the_row_it_supersedes(
    tmp_path: Path,
) -> None:
    """The corrected claim reaches disk as its own event type, superseding row 3.

    Acceptance criterion 1, end to end: the row names the superseded
    `sequence_number`, the corrected evidentiary fields, and the correction's
    own provenance -- and it is a *different* event type from the first ingest,
    which is how a reader tells an extraordinary act from an ordinary one.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    base = _base_instant()
    ledger_path = tmp_path / "ledger.db"
    _seed(ledger_path, deployed_at=base, forecast_at=base + timedelta(hours=1))
    settled_at = base + timedelta(hours=2)
    assert main(_ingest_argv(ledger_path, outcome="no", resolved_at=settled_at)) == 0

    exit_code = main(
        _correct_argv(ledger_path, superseded=3, outcome="yes", resolved_at=settled_at)
    )

    assert exit_code == 0
    rows = _rows(ledger_path)
    assert [event_type for event_type, _ in rows] == [
        "ConfigLoaded",
        "ForecastCreated",
        MARKET_RESOLVED_EVENT_TYPE,
        SETTLEMENT_REVERSED_EVENT_TYPE,
    ]
    assert json.loads(rows[3][1])["data"] == {
        "market_ticker": _TICKER,
        "superseded_sequence_number": 3,
        "outcome": "yes",
        "resolved_at": settled_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")
        + "Z",
        "source": _CORRECTION_SOURCE,
    }
    assert len(_records(ledger_path)) == 4


def test_the_correction_reaches_the_database_files_on_disk(tmp_path: Path) -> None:
    """The correction's bytes are in `ledger.db*`, WAL sidecar included.

    The positive control is the *original* provenance label: a scan that found
    neither string would pass vacuously over an empty or unwritten file, so the
    test asserts the superseded row is still physically there too. Nothing was
    redacted -- that is the append-only guarantee, checked at the byte level.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    base = _base_instant()
    ledger_path = tmp_path / "ledger.db"
    _seed(ledger_path, deployed_at=base, forecast_at=base + timedelta(hours=1))
    settled_at = base + timedelta(hours=2)
    assert main(_ingest_argv(ledger_path, outcome="no", resolved_at=settled_at)) == 0

    assert (
        main(
            _correct_argv(
                ledger_path, superseded=3, outcome="yes", resolved_at=settled_at
            )
        )
        == 0
    )

    raw = _on_disk_bytes(ledger_path)
    assert _CORRECTION_SOURCE.encode() in raw
    assert _ORIGINAL_SOURCE.encode() in raw
    assert SETTLEMENT_REVERSED_EVENT_TYPE.encode() in raw


# ---------------------------------------------------------------------------
# 3. The metric moves -- and does not move without the correction.
# ---------------------------------------------------------------------------


def test_the_corrected_outcome_moves_the_metric_and_the_uncorrected_one_does_not(
    tmp_path: Path,
) -> None:
    """`brier` reads the corrected outcome's value, and the wrong one without it.

    Acceptance criterion 2, end to end and in both directions. The two Brier
    values are asserted to differ, so neither number can be coming from a
    fixture coincidence, and the second fold differs from the first by exactly
    one removed row.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    base = _base_instant()
    ledger_path = tmp_path / "ledger.db"
    _seed(ledger_path, deployed_at=base, forecast_at=base + timedelta(hours=1))
    settled_at = base + timedelta(hours=2)
    assert main(_ingest_argv(ledger_path, outcome="no", resolved_at=settled_at)) == 0
    before = weekly_report_body(_records(ledger_path), today=_TODAY)

    assert (
        main(
            _correct_argv(
                ledger_path, superseded=3, outcome="yes", resolved_at=settled_at
            )
        )
        == 0
    )

    records = _records(ledger_path)
    after = weekly_report_body(records, today=_TODAY)
    without_correction = [
        record
        for record in records
        if record.event_type != SETTLEMENT_REVERSED_EVENT_TYPE
    ]
    assert len(without_correction) == len(records) - 1
    negative = weekly_report_body(without_correction, today=_TODAY)

    assert _BRIER_IF_YES != _BRIER_IF_NO
    assert _BRIER_IF_NO == 62_500
    assert _BRIER_IF_YES == 562_500
    assert (
        _metric_line(before, "brier") == f"brier [latest_before_close] = {_BRIER_IF_NO}"
    )
    assert (
        _metric_line(after, "brier") == f"brier [latest_before_close] = {_BRIER_IF_YES}"
    )
    assert _metric_line(negative, "brier") == (
        f"brier [latest_before_close] = {_BRIER_IF_NO}"
    )


def test_the_report_names_the_superseded_claim_and_omits_the_section_without_one(
    tmp_path: Path,
) -> None:
    """A correction that leaves no trace is the failure mode being prevented.

    The rendered report gains a `## Resolution corrections` section naming both
    rows and both claims -- and carries no such section at all until a
    correction exists, so its presence is itself the signal.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    base = _base_instant()
    ledger_path = tmp_path / "ledger.db"
    _seed(ledger_path, deployed_at=base, forecast_at=base + timedelta(hours=1))
    settled_at = base + timedelta(hours=2)
    assert main(_ingest_argv(ledger_path, outcome="no", resolved_at=settled_at)) == 0
    before = weekly_report_body(_records(ledger_path), today=_TODAY)

    assert (
        main(
            _correct_argv(
                ledger_path, superseded=3, outcome="yes", resolved_at=settled_at
            )
        )
        == 0
    )

    after = weekly_report_body(_records(ledger_path), today=_TODAY)
    stamp = settled_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    assert "## Resolution corrections" not in before
    assert "## Resolution corrections" in after
    assert (
        f"RESOLUTION_CORRECTED {_TICKER} superseded_sequence_number=3 "
        f"outcome='no' resolved_at={stamp} -> correction_sequence_number=4 "
        f"outcome='yes' resolved_at={stamp} source={_CORRECTION_SOURCE!r}"
    ) in after


def test_a_second_correction_names_the_first_and_moves_the_metric_again(
    tmp_path: Path,
) -> None:
    """The path is not single-use: correcting a correction works and is visible.

    A mechanism that can be walked exactly once is the same permanent trap this
    issue exists to remove, reached one command later.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    base = _base_instant()
    ledger_path = tmp_path / "ledger.db"
    _seed(ledger_path, deployed_at=base, forecast_at=base + timedelta(hours=1))
    settled_at = base + timedelta(hours=2)
    assert main(_ingest_argv(ledger_path, outcome="no", resolved_at=settled_at)) == 0
    assert (
        main(
            _correct_argv(
                ledger_path, superseded=3, outcome="yes", resolved_at=settled_at
            )
        )
        == 0
    )

    exit_code = main(
        _correct_argv(
            ledger_path,
            superseded=4,
            outcome="no",
            resolved_at=settled_at,
            source="venue support ticket 4471",
        )
    )

    assert exit_code == 0
    body = weekly_report_body(_records(ledger_path), today=_TODAY)
    assert (
        _metric_line(body, "brier") == f"brier [latest_before_close] = {_BRIER_IF_NO}"
    )
    assert body.count("RESOLUTION_CORRECTED") == 2


# ---------------------------------------------------------------------------
# 4. Temporal integrity survives a correction, in both directions.
# ---------------------------------------------------------------------------


def test_correcting_the_instant_later_makes_a_backdated_forecast_scorable(
    tmp_path: Path,
) -> None:
    """A forecast refused `backdated` is scored once the true instant is recorded.

    The first ingest claims the market settled 30 minutes after deployment --
    *before* the forecast was created -- so the forecast could have peeked and
    is refused. The correction moves the instant to an hour after the forecast,
    with the outcome unchanged, so the metric moves on the instant alone.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    base = _base_instant()
    ledger_path = tmp_path / "ledger.db"
    _seed(ledger_path, deployed_at=base, forecast_at=base + timedelta(hours=1))
    assert (
        main(
            _ingest_argv(
                ledger_path, outcome="no", resolved_at=base + timedelta(minutes=30)
            )
        )
        == 0
    )
    before = weekly_report_body(_records(ledger_path), today=_TODAY)

    assert (
        main(
            _correct_argv(
                ledger_path,
                superseded=3,
                outcome="no",
                resolved_at=base + timedelta(hours=2),
            )
        )
        == 0
    )

    after = weekly_report_body(_records(ledger_path), today=_TODAY)
    assert _rejection_lines(before) == [
        f"EVALUATION_RECORD_REJECTED fc-a {_TICKER} backdated"
    ]
    assert _metric_line(before, "brier") == "brier [latest_before_close] = UNDEFINED"
    assert _rejection_lines(after) == []
    assert (
        _metric_line(after, "brier") == f"brier [latest_before_close] = {_BRIER_IF_NO}"
    )


def test_correcting_the_instant_earlier_refuses_a_forecast_that_was_scored(
    tmp_path: Path,
) -> None:
    """The mirror direction: a scored forecast becomes `backdated` on correction.

    #439's whole point is that a forecast may only be scored against a
    resolution that postdates it. A correction changes what a market settled
    to *and when*, so the gate must re-adjudicate -- in this direction it must
    take a score away, which is the direction an operator would rather not see
    and therefore the one most worth pinning.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    base = _base_instant()
    ledger_path = tmp_path / "ledger.db"
    _seed(ledger_path, deployed_at=base, forecast_at=base + timedelta(hours=1))
    assert (
        main(
            _ingest_argv(
                ledger_path, outcome="no", resolved_at=base + timedelta(hours=2)
            )
        )
        == 0
    )
    before = weekly_report_body(_records(ledger_path), today=_TODAY)

    assert (
        main(
            _correct_argv(
                ledger_path,
                superseded=3,
                outcome="no",
                resolved_at=base + timedelta(minutes=30),
            )
        )
        == 0
    )

    after = weekly_report_body(_records(ledger_path), today=_TODAY)
    assert _rejection_lines(before) == []
    assert _metric_line(before, "brier") == (
        f"brier [latest_before_close] = {_BRIER_IF_NO}"
    )
    assert _rejection_lines(after) == [
        f"EVALUATION_RECORD_REJECTED fc-a {_TICKER} backdated"
    ]
    assert _metric_line(after, "brier") == "brier [latest_before_close] = UNDEFINED"


# ---------------------------------------------------------------------------
# 5. Refusals: at the verb, exit 1, nothing written.
# ---------------------------------------------------------------------------


def _seeded_and_ingested(tmp_path: Path) -> tuple[Path, datetime]:
    """Seed a ledger and ingest one `no` resolution for `_TICKER`.

    Args:
        tmp_path: Pytest's per-test temporary directory.

    Returns:
        The ledger path paired with the settlement instant that was ingested.
    """
    base = _base_instant()
    ledger_path = tmp_path / "ledger.db"
    _seed(ledger_path, deployed_at=base, forecast_at=base + timedelta(hours=1))
    settled_at = base + timedelta(hours=2)
    assert main(_ingest_argv(ledger_path, outcome="no", resolved_at=settled_at)) == 0
    return ledger_path, settled_at


def test_a_correction_naming_a_row_that_is_not_the_current_claim_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Acceptance criterion 3: naming a non-resolution row exits 1, writes nothing.

    Row 2 is the `ForecastCreated`. The refusal names the row that *does* carry
    the claim, so the operator's next command is written for them -- a refusal
    with no way forward is the trap this issue exists to close.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        capsys: Captures the structured `FATAL` diagnostic.
    """
    ledger_path, settled_at = _seeded_and_ingested(tmp_path)
    stamp = settled_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    capsys.readouterr()

    exit_code = main(
        _correct_argv(ledger_path, superseded=2, outcome="yes", resolved_at=settled_at)
    )

    assert exit_code == 1
    assert len(_rows(ledger_path)) == 3
    assert _fatal_messages(capsys.readouterr().err) == (
        f"FATAL: refusing to correct market_ticker='{_TICKER}': "
        "--superseded-sequence-number=2 does not name the row carrying this "
        "market's current resolution, which is sequence_number=3 "
        f"(outcome='no' resolved_at={stamp}). A correction supersedes exactly "
        "the row it names, so naming any other position would leave two rows "
        "claiming one market with no rule for which wins. Nothing was written. "
        "Re-run with --superseded-sequence-number=3."
    )


def test_a_correction_of_an_already_superseded_row_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Acceptance criterion 3: a row already reversed cannot be reversed again.

    The recovery is spelled out in the refusal: name the correction row
    instead, which `test_a_second_correction_names_the_first_and_moves_the_
    metric_again` proves works.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        capsys: Captures the structured `FATAL` diagnostic.
    """
    ledger_path, settled_at = _seeded_and_ingested(tmp_path)
    assert (
        main(
            _correct_argv(
                ledger_path, superseded=3, outcome="yes", resolved_at=settled_at
            )
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        _correct_argv(ledger_path, superseded=3, outcome="no", resolved_at=settled_at)
    )

    assert exit_code == 1
    assert len(_rows(ledger_path)) == 4
    assert "--superseded-sequence-number=3 does not name the row carrying this" in (
        _fatal_messages(capsys.readouterr().err)
    )


def test_a_correction_of_a_market_with_no_resolution_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """There is nothing to supersede until the market has been ingested at all.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        capsys: Captures the structured `FATAL` diagnostic.
    """
    ledger_path, settled_at = _seeded_and_ingested(tmp_path)
    capsys.readouterr()

    exit_code = main(
        _correct_argv(
            ledger_path,
            superseded=3,
            outcome="yes",
            resolved_at=settled_at,
            ticker="MKT-B",
        )
    )

    assert exit_code == 1
    assert len(_rows(ledger_path)) == 3
    assert _fatal_messages(capsys.readouterr().err) == (
        "FATAL: refusing to correct market_ticker='MKT-B': this ledger carries "
        "no resolution for that market, so there is nothing to supersede. "
        "Record it with `windbreak ingest-resolution` instead. Nothing was "
        "written."
    )


def test_a_correction_that_changes_nothing_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A no-op reversal is refused: it would launder an unexplained act on-chain.

    Reversing a settlement is extraordinary. One that restates the claim it
    supersedes changes no metric and leaves an unexplained reversal in the
    audit trail, so the verb declines rather than recording it. Nothing is
    bricked by declining -- the operator simply had nothing to correct.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        capsys: Captures the structured `FATAL` diagnostic.
    """
    ledger_path, settled_at = _seeded_and_ingested(tmp_path)
    stamp = settled_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    capsys.readouterr()

    exit_code = main(
        _correct_argv(ledger_path, superseded=3, outcome="no", resolved_at=settled_at)
    )

    assert exit_code == 1
    assert len(_rows(ledger_path)) == 3
    assert _fatal_messages(capsys.readouterr().err) == (
        f"FATAL: refusing to correct market_ticker='{_TICKER}': this call "
        f"claims outcome='no' resolved_at={stamp}, which is exactly what "
        "sequence_number=3 already carries. A correction that changes nothing "
        "would record an unexplained reversal in an append-only audit trail "
        "without moving a single metric. Nothing was written."
    )


def test_a_correction_with_a_naive_instant_is_refused_before_the_ledger_is_opened(
    tmp_path: Path,
    local_timezone_utc_minus_5: None,
) -> None:
    """An offsetless `--resolved-at` exits 1 and leaves the ledger untouched.

    The process timezone is pinned five hours off UTC, so a host that quietly
    read the naive wall clock as local time would store a different instant
    from one that read it as UTC -- a difference no UTC-running CI host could
    otherwise see.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    ledger_path, _ = _seeded_and_ingested(tmp_path)
    argv = _correct_argv(
        ledger_path,
        superseded=3,
        outcome="yes",
        resolved_at=datetime(2026, 3, 2, 9, 30, tzinfo=UTC),
    )
    argv[argv.index("--resolved-at") + 1] = "2026-03-02T09:30:00"

    exit_code = main(argv)

    assert exit_code == 1
    assert len(_rows(ledger_path)) == 3


def test_a_correction_with_a_blank_source_is_refused(tmp_path: Path) -> None:
    """An unattributed correction exits 1 and writes nothing.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path, settled_at = _seeded_and_ingested(tmp_path)

    exit_code = main(
        _correct_argv(
            ledger_path,
            superseded=3,
            outcome="yes",
            resolved_at=settled_at,
            source="   ",
        )
    )

    assert exit_code == 1
    assert len(_rows(ledger_path)) == 3


def test_a_correction_into_an_unfoldable_ledger_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ledger this verb did not write is not made worse by appending to it.

    Two contradicting `MarketResolved` rows can only reach a ledger by some
    path other than these verbs, and once there the weekly fold cannot read it
    at all. Appending a correction would not help -- the fold fails before it
    ever reaches the correction -- so the verb refuses and quotes the fold's
    own diagnosis rather than writing a row that cannot be read.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        capsys: Captures the structured `FATAL` diagnostic.
    """
    base = _base_instant()
    ledger_path = tmp_path / "ledger.db"
    settled_at = base + timedelta(hours=2)
    store = SqliteLedgerStore(ledger_path)
    try:
        for outcome in (ResolutionOutcome.NO, ResolutionOutcome.YES):
            store.append(
                MarketResolved(
                    component="operator",
                    market_ticker=_TICKER,
                    outcome=outcome,
                    resolved_at=settled_at,
                    source="hand-written",
                )
            )
    finally:
        store.close()
    capsys.readouterr()

    exit_code = main(
        _correct_argv(ledger_path, superseded=1, outcome="yes", resolved_at=settled_at)
    )

    assert exit_code == 1
    assert len(_rows(ledger_path)) == 2
    messages = _fatal_messages(capsys.readouterr().err)
    assert messages.startswith(
        f"FATAL: refusing to correct market_ticker='{_TICKER}': this ledger's "
        "existing MarketResolved rows cannot be folded -- MarketResolved at "
        "sequence_number=2 contradicts an earlier resolution of "
        f"market_ticker='{_TICKER}':"
    )
    assert messages.endswith("Nothing was written.")


def test_a_refused_correction_leaves_the_weekly_fold_working(tmp_path: Path) -> None:
    """The loop keeps beating after every refusal: the fold still renders.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path, settled_at = _seeded_and_ingested(tmp_path)

    assert (
        main(
            _correct_argv(
                ledger_path, superseded=1, outcome="yes", resolved_at=settled_at
            )
        )
        == 1
    )

    body = weekly_report_body(_records(ledger_path), today=_TODAY)
    assert "## Cost meter" in body
    assert (
        _metric_line(body, "brier") == f"brier [latest_before_close] = {_BRIER_IF_NO}"
    )
