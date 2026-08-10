"""Tests for the `windbreak ingest-resolution` CLI subcommand (issue #439).

This verb is the whole ingestion mechanism: the smallest surface that lets a
resolved outcome enter the running system from *outside* the forecast records
it grades. It appends exactly one `MarketResolved` row to the hash-chained
ledger; the weekly fold the always-on loop already runs picks it up on the next
tick, with no scheduler change at all.

The refusals matter as much as the append. An instant with no UTC offset is
unprovable evidence and is refused rather than reinterpreted against whatever
timezone the operator's shell happens to carry, and a refused ingest must leave
the ledger byte-for-byte unchanged -- a half-ingested resolution would be worse
than none, because the report would then quote a number nothing supports.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from windbreak.evaluation.ingest import MARKET_RESOLVED_EVENT_TYPE
from windbreak.ledger.events import ModeHeartbeat
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.main import build_parser, main

#: A well-formed settlement instant, with an explicit UTC offset.
_RESOLVED_AT = "2026-03-01T12:00:00+00:00"


def _ingest_argv(ledger_path: Path, *, resolved_at: str = _RESOLVED_AT) -> list[str]:
    """Build a complete `ingest-resolution` argument vector.

    Args:
        ledger_path: The ledger database the verb appends into.
        resolved_at: The settlement instant to pass through.

    Returns:
        The argument vector for `main()`.
    """
    return [
        "ingest-resolution",
        "--ledger-path",
        str(ledger_path),
        "--market-ticker",
        "MKT-A",
        "--outcome",
        "no",
        "--resolved-at",
        resolved_at,
        "--source",
        "kalshi-settlement-notice",
    ]


def _ledger_rows(ledger_path: Path) -> list[tuple[str, str]]:
    """Read every ledger row's event type and envelope, without the store.

    Reading through `sqlite3` directly proves the row reached disk rather than
    an in-process buffer.

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


def test_ingest_resolution_parses_all_five_required_arguments() -> None:
    """The verb parses its ledger path, ticker, outcome, instant and source."""
    args = build_parser().parse_args(_ingest_argv(Path("ledger.db")))

    assert args.command == "ingest-resolution"
    assert args.ledger_path == Path("ledger.db")
    assert args.market_ticker == "MKT-A"
    assert args.outcome == "no"
    assert args.resolved_at == _RESOLVED_AT
    assert args.source == "kalshi-settlement-notice"


@pytest.mark.parametrize(
    "dropped",
    ["--ledger-path", "--market-ticker", "--outcome", "--resolved-at", "--source"],
)
def test_ingest_resolution_requires_every_argument(dropped: str) -> None:
    """Omitting any one argument is a usage error, never a defaulted ingest.

    Args:
        dropped: The option (and its value) removed from the argument vector.
    """
    argv = _ingest_argv(Path("ledger.db"))
    index = argv.index(dropped)
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)

    assert exc_info.value.code == 2


def test_ingest_resolution_rejects_an_outcome_outside_the_binary_vocabulary() -> None:
    """An `--outcome` token other than `yes`/`no` is a usage error."""
    argv = _ingest_argv(Path("ledger.db"))
    argv[argv.index("--outcome") + 1] = "maybe"

    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)

    assert exc_info.value.code == 2


def test_ingest_resolution_appends_exactly_one_market_resolved_row(
    tmp_path: Path,
) -> None:
    """A well-formed ingest writes one row carrying the operator's four fields.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path = tmp_path / "ledger.db"

    exit_code = main(_ingest_argv(ledger_path))

    assert exit_code == 0
    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1
    event_type, payload_json = rows[0]
    assert event_type == MARKET_RESOLVED_EVENT_TYPE
    assert json.loads(payload_json)["data"] == {
        "market_ticker": "MKT-A",
        "outcome": "no",
        "resolved_at": "2026-03-01T12:00:00.000000Z",
        "source": "kalshi-settlement-notice",
    }


def test_ingest_resolution_normalizes_a_non_utc_offset_to_the_same_instant(
    tmp_path: Path,
) -> None:
    """An offset-carrying instant is stored as the same moment rendered in UTC.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path = tmp_path / "ledger.db"

    exit_code = main(_ingest_argv(ledger_path, resolved_at="2026-03-01T07:00:00-05:00"))

    assert exit_code == 0
    payload = json.loads(_ledger_rows(ledger_path)[0][1])["data"]
    assert payload["resolved_at"] == "2026-03-01T12:00:00.000000Z"
    assert datetime.fromisoformat("2026-03-01T07:00:00-05:00") == datetime(
        2026, 3, 1, 12, 0, 0, tzinfo=UTC
    )


def test_ingest_resolution_refuses_a_naive_instant_and_writes_nothing(
    tmp_path: Path,
    local_timezone_utc_minus_5: None,
) -> None:
    """An offsetless `--resolved-at` exits 1 and leaves the ledger untouched.

    The local timezone is pinned five hours off UTC, so a host that quietly
    read the naive wall clock as local time would store a different instant
    from one that read it as UTC -- a difference no UTC-running CI host could
    otherwise see.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    ledger_path = tmp_path / "ledger.db"
    SqliteLedgerStore(ledger_path).close()

    exit_code = main(_ingest_argv(ledger_path, resolved_at="2026-03-01T12:00:00"))

    assert exit_code == 1
    assert _ledger_rows(ledger_path) == []


def test_ingest_resolution_refuses_an_unparseable_instant_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """A `--resolved-at` that is not ISO-8601 exits 1 without touching the ledger.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path = tmp_path / "ledger.db"
    SqliteLedgerStore(ledger_path).close()

    exit_code = main(_ingest_argv(ledger_path, resolved_at="last Tuesday"))

    assert exit_code == 1
    assert _ledger_rows(ledger_path) == []


def test_ingest_resolution_refuses_a_blank_source_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """A whitespace-only `--source` exits 1: an unattributed outcome is refused.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path = tmp_path / "ledger.db"
    SqliteLedgerStore(ledger_path).close()
    argv = _ingest_argv(ledger_path)
    argv[argv.index("--source") + 1] = "   "

    exit_code = main(argv)

    assert exit_code == 1
    assert _ledger_rows(ledger_path) == []


def test_ingest_resolution_appends_to_an_existing_chain_and_keeps_it_valid(
    tmp_path: Path,
) -> None:
    """Ingesting extends the existing chain rather than replacing it.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path = tmp_path / "ledger.db"
    store = SqliteLedgerStore(ledger_path)
    store.append(ModeHeartbeat(component="scheduler", mode="PAPER", beat=1))
    store.close()

    exit_code = main(_ingest_argv(ledger_path))

    assert exit_code == 0
    assert [event_type for event_type, _ in _ledger_rows(ledger_path)] == [
        "ModeHeartbeat",
        MARKET_RESOLVED_EVENT_TYPE,
    ]
    verifier = SqliteLedgerStore(ledger_path)
    try:
        verifier.verify_chain()
    finally:
        verifier.close()
