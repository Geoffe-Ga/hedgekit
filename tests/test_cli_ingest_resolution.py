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

One refusal has to happen *here* and nowhere else. The ledger is append-only
and `weekly_report_body` folds it on every tick, so a row contradicting an
earlier resolution is not a bad record but a permanent stop on the always-on
loop, which re-ingesting the correct value cannot undo. The verb therefore
reads the existing resolutions back before it appends anything, through the
same fold the tick uses, and exits 1 without writing. The mirror of that: a
call differing only in the free-text `--source` label is *not* a contradiction
and is accepted, because provenance is not a claim about what the market did.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from windbreak.evaluation.ingest import (
    MARKET_RESOLVED_EVENT_TYPE,
    MarketResolved,
    ingested_resolutions_from_records,
)
from windbreak.evaluation.resolution import ResolutionOutcome
from windbreak.ledger.events import ModeHeartbeat
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.main import build_parser, main
from windbreak.scheduler.weekly_data import weekly_report_body

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


# ---------------------------------------------------------------------------
# The verb refuses a conflicting append rather than letting the fold refuse it.
# ---------------------------------------------------------------------------


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


def _ingest(
    ledger_path: Path,
    *,
    outcome: str = "no",
    resolved_at: str = _RESOLVED_AT,
    source: str = "kalshi-settlement-notice",
) -> int:
    """Run one `ingest-resolution` call against `MKT-A`.

    Args:
        ledger_path: The ledger database the verb appends into.
        outcome: The `--outcome` token.
        resolved_at: The `--resolved-at` instant.
        source: The `--source` provenance label.

    Returns:
        The verb's exit code.
    """
    argv = _ingest_argv(ledger_path, resolved_at=resolved_at)
    argv[argv.index("--outcome") + 1] = outcome
    argv[argv.index("--source") + 1] = source
    return main(argv)


def test_a_second_ingest_differing_only_in_source_is_accepted(
    tmp_path: Path,
) -> None:
    """Retyping the free-text provenance label is not a contradiction.

    `source` says where an operator read the settlement, never what the market
    did, so the two spellings below make one claim. Both rows land -- the
    ledger is append-only and records what was actually run -- and the fold
    reads one resolution back off them.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path = tmp_path / "ledger.db"

    first = _ingest(ledger_path, source="kalshi settlement notice")
    second = _ingest(ledger_path, source="kalshi-settlement-notice")

    assert (first, second) == (0, 0)
    assert len(_ledger_rows(ledger_path)) == 2
    store = SqliteLedgerStore(ledger_path)
    try:
        resolutions = ingested_resolutions_from_records(store.read_all())
    finally:
        store.close()
    assert len(resolutions) == 1
    assert resolutions[0].source == "kalshi settlement notice"


def test_a_contradicting_ingest_exits_one_and_appends_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second call claiming a different outcome is refused before it writes.

    This is the whole point of checking at the verb. The ledger is append-only
    and `weekly_report_body` folds it on every tick, so letting this row land
    would stop the loop permanently -- and re-ingesting the correct value could
    not undo it.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        capsys: Captures the structured `FATAL` diagnostic.
    """
    ledger_path = tmp_path / "ledger.db"
    assert _ingest(ledger_path, outcome="no") == 0
    capsys.readouterr()

    exit_code = _ingest(ledger_path, outcome="yes")

    assert exit_code == 1
    assert len(_ledger_rows(ledger_path)) == 1
    assert json.loads(_ledger_rows(ledger_path)[0][1])["data"]["outcome"] == "no"
    assert _fatal_messages(capsys.readouterr().err) == (
        "FATAL: refusing to ingest market_ticker='MKT-A': it already resolved "
        "on this ledger with outcome='no' "
        "resolved_at=2026-03-01T12:00:00.000000Z, and this call claims "
        "outcome='yes' resolved_at=2026-03-01T12:00:00.000000Z. The ledger is "
        "append-only, so a contradicting row could never be un-written and "
        "every later weekly fold -- one per tick -- would refuse to read it. "
        "Nothing was written. Re-run with the values the ledger already "
        "carries; correcting a genuinely wrong recorded outcome needs the "
        "settlement-reversal path, which does not exist yet (issue #484) and "
        "cannot be improvised by ingesting again."
    )


def test_an_ingest_at_a_different_instant_is_refused(tmp_path: Path) -> None:
    """A second call moving the settlement instant is refused, exactly.

    The instant decides which forecasts could have peeked, so a different one
    is a different claim -- compared at full microsecond precision.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path = tmp_path / "ledger.db"
    assert _ingest(ledger_path, resolved_at="2026-03-01T12:00:00+00:00") == 0

    exit_code = _ingest(ledger_path, resolved_at="2026-03-01T12:00:00.000001+00:00")

    assert exit_code == 1
    assert len(_ledger_rows(ledger_path)) == 1


def test_an_ingest_restating_the_same_instant_at_another_offset_is_accepted(
    tmp_path: Path,
) -> None:
    """`07:00-05:00` and `12:00+00:00` are one instant, so this is idempotent.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path = tmp_path / "ledger.db"
    assert _ingest(ledger_path, resolved_at="2026-03-01T12:00:00+00:00") == 0

    exit_code = _ingest(ledger_path, resolved_at="2026-03-01T07:00:00-05:00")

    assert exit_code == 0
    assert len(_ledger_rows(ledger_path)) == 2


def test_an_ingest_into_an_already_contradictory_ledger_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ledger this verb did not write is not made worse by appending to it.

    Two conflicting rows can only reach a ledger by some path other than this
    verb, and once there the weekly fold cannot read it at all. Appending a
    third row would not help, so the verb refuses and quotes the fold's own
    diagnosis.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        capsys: Captures the structured `FATAL` diagnostic.
    """
    ledger_path = tmp_path / "ledger.db"
    store = SqliteLedgerStore(ledger_path)
    try:
        for outcome in (ResolutionOutcome.NO, ResolutionOutcome.YES):
            store.append(
                MarketResolved(
                    component="operator",
                    market_ticker="MKT-A",
                    outcome=outcome,
                    resolved_at=datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC),
                    source="hand-written",
                )
            )
    finally:
        store.close()
    capsys.readouterr()

    exit_code = _ingest(ledger_path, outcome="no")

    assert exit_code == 1
    assert len(_ledger_rows(ledger_path)) == 2
    messages = _fatal_messages(capsys.readouterr().err)
    assert messages.startswith(
        "FATAL: refusing to ingest market_ticker='MKT-A': this ledger's "
        "existing MarketResolved rows cannot be folded -- MarketResolved at "
        "sequence_number=2 contradicts an earlier resolution of "
        "market_ticker='MKT-A':"
    )
    assert messages.endswith("Nothing was written.")


def test_a_refused_ingest_leaves_the_weekly_fold_working(tmp_path: Path) -> None:
    """The loop keeps beating: the fold still renders after a refused call.

    The end-to-end shape of the bug this closes. Two `ingest-resolution` calls
    that differ only in a mistyped field used to both exit 0 and leave every
    subsequent `weekly_report_body` -- and therefore every tick -- raising
    forever. Now the second call is refused and the fold still returns a body.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path = tmp_path / "ledger.db"
    assert _ingest(ledger_path, outcome="no") == 0

    assert _ingest(ledger_path, outcome="yes") == 1

    store = SqliteLedgerStore(ledger_path)
    try:
        body = weekly_report_body(store.read_all(), today=date(2026, 3, 8))
    finally:
        store.close()
    assert "## Cost meter" in body


def test_a_different_market_is_unaffected_by_an_existing_resolution(
    tmp_path: Path,
) -> None:
    """The conflict check is per-market, not per-ledger.

    A ledger accumulates one resolution per settled market, so a check that
    compared against whichever resolution it found first would refuse every
    market after the first one -- and the operator would be locked out of
    ingesting at exactly the point the harness started having enough data to
    say anything.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    ledger_path = tmp_path / "ledger.db"
    assert _ingest(ledger_path, outcome="no") == 0
    argv = _ingest_argv(ledger_path)
    argv[argv.index("--market-ticker") + 1] = "MKT-B"
    argv[argv.index("--outcome") + 1] = "yes"

    exit_code = main(argv)

    assert exit_code == 0
    store = SqliteLedgerStore(ledger_path)
    try:
        resolutions = ingested_resolutions_from_records(store.read_all())
    finally:
        store.close()
    assert [
        (resolution.market_ticker, resolution.outcome) for resolution in resolutions
    ] == [("MKT-A", ResolutionOutcome.NO), ("MKT-B", ResolutionOutcome.YES)]
