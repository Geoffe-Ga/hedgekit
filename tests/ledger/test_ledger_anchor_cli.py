"""Tests for the `windbreak anchor` / `windbreak verify` CLI subcommands (issue #75).

Mirrors `tests/ledger/test_ledger_rebuild_cli.py`'s style: pins the CLI-level
contract that `anchor` and `verify` each require `--ledger-path` and
`--anchor-path` Path arguments, that `main()` dispatches to
`anchor_command`/`verify_command` and surfaces their exit codes (0 clean, 1
on failure with the offending detail on stderr), and that the pre-existing
`run`/`rebuild` subcommands' parsing is unaffected by this addition.

Issue #217 adds the empty-ledger contract to that list: `anchor` distinguishes
a *missing* ledger (a refusal, exit 1, nothing created) from an *existing but
empty* one (exit 0 with an explicit "nothing anchored" notice on stderr), and
still anchors a populated ledger to the exact same bytes as before -- pinned
here by full-line equality, not by an exit code alone.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.ledger.anchor import anchor_head
from windbreak.ledger.events import GENESIS_PREV_HASH, ModeHeartbeat
from windbreak.ledger.store import SqliteLedgerStore, compute_event_hash
from windbreak.main import build_parser, main

if TYPE_CHECKING:
    from collections.abc import Callable

#: The chained ``event_hash`` of the single record every anchoring test below
#: appends: ``ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1)``
#: stamped by ``deterministic_clock``'s fixed 2024-01-01T00:00:00+00:00 epoch.
#: Built here from the §12 digest preimage written out field by field, rather
#: than read back from the store, so the anchor's *contents* are checked
#: against a value the anchoring path never supplied.
_HEAD_HASH_AT_SEQUENCE_ONE = compute_event_hash(
    sequence_number=1,
    event_type="ModeHeartbeat",
    created_at="2024-01-01T00:00:00.000000+00:00",
    payload_json=(
        '{"component":"pipeline","data":{"beat":1,"mode":"RESEARCH"},'
        '"schema_version":1}'
    ),
    prev_hash=GENESIS_PREV_HASH,
)


def test_build_parser_parses_anchor_with_required_args() -> None:
    """`anchor --ledger-path X --anchor-path Y` parses both as Path objects."""
    parser = build_parser()

    args = parser.parse_args(
        ["anchor", "--ledger-path", "ledger.db", "--anchor-path", "anchors.jsonl"]
    )

    assert args.command == "anchor"
    assert args.ledger_path == Path("ledger.db")
    assert args.anchor_path == Path("anchors.jsonl")


def test_build_parser_anchor_requires_ledger_path() -> None:
    """Omitting `--ledger-path` on `anchor` is a usage error (SystemExit(2))."""
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["anchor", "--anchor-path", "anchors.jsonl"])

    assert exc_info.value.code == 2


def test_build_parser_anchor_requires_anchor_path() -> None:
    """Omitting `--anchor-path` on `anchor` is a usage error (SystemExit(2))."""
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["anchor", "--ledger-path", "ledger.db"])

    assert exc_info.value.code == 2


def test_build_parser_anchor_requires_both_args_when_neither_is_given() -> None:
    """`anchor` with no flags at all is a usage error (SystemExit(2))."""
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["anchor"])

    assert exc_info.value.code == 2


def test_build_parser_parses_verify_with_required_args() -> None:
    """`verify --ledger-path X --anchor-path Y` parses both as Path objects."""
    parser = build_parser()

    args = parser.parse_args(
        ["verify", "--ledger-path", "ledger.db", "--anchor-path", "anchors.jsonl"]
    )

    assert args.command == "verify"
    assert args.ledger_path == Path("ledger.db")
    assert args.anchor_path == Path("anchors.jsonl")


def test_build_parser_verify_requires_ledger_path() -> None:
    """Omitting `--ledger-path` on `verify` is a usage error (SystemExit(2))."""
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["verify", "--anchor-path", "anchors.jsonl"])

    assert exc_info.value.code == 2


def test_build_parser_verify_requires_anchor_path() -> None:
    """Omitting `--anchor-path` on `verify` is a usage error (SystemExit(2))."""
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["verify", "--ledger-path", "ledger.db"])

    assert exc_info.value.code == 2


def test_build_parser_verify_requires_both_args_when_neither_is_given() -> None:
    """`verify` with no flags at all is a usage error (SystemExit(2))."""
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["verify"])

    assert exc_info.value.code == 2


def test_main_anchor_returns_zero_and_writes_one_anchor_line(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`main(["anchor", ...])` returns 0 and appends exactly one anchor line."""
    db_path = tmp_path / "ledger.db"
    anchor_path = tmp_path / "anchors.jsonl"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()

    exit_code = main(
        ["anchor", "--ledger-path", str(db_path), "--anchor-path", str(anchor_path)]
    )

    assert exit_code == 0
    assert len(anchor_path.read_text(encoding="utf-8").splitlines()) == 1


def test_main_anchor_returns_one_on_tampered_ledger_and_reports_sequence_number(
    tmp_path: Path,
    deterministic_clock: Callable[[], datetime],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main(["anchor", ...])` returns 1 and reports the tampered sequence position."""
    db_path = tmp_path / "ledger.db"
    anchor_path = tmp_path / "anchors.jsonl"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE ledger SET event_hash = ? WHERE sequence_number = 1", ("0" * 64,)
        )
        conn.commit()
    finally:
        conn.close()

    exit_code = main(
        ["anchor", "--ledger-path", str(db_path), "--anchor-path", str(anchor_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "sequence_number=1" in captured.err
    assert not anchor_path.exists()


def test_main_verify_returns_zero_on_clean_anchored_chain(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`main(["verify", ...])` returns 0 for an untampered, anchored chain."""
    db_path = tmp_path / "ledger.db"
    anchor_path = tmp_path / "anchors.jsonl"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    anchor_head(db_path, anchor_path)

    exit_code = main(
        ["verify", "--ledger-path", str(db_path), "--anchor-path", str(anchor_path)]
    )

    assert exit_code == 0


def test_main_verify_returns_one_on_tampered_chain_and_reports_sequence_number(
    tmp_path: Path,
    deterministic_clock: Callable[[], datetime],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main(["verify", ...])` returns 1 and prints `sequence_number=<n>` on a
    plain chain-integrity break (unrelated to anchoring: verify_chain fails first).
    """
    db_path = tmp_path / "ledger.db"
    anchor_path = tmp_path / "anchors.jsonl"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    anchor_head(db_path, anchor_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE ledger SET event_hash = ? WHERE sequence_number = 1", ("0" * 64,)
        )
        conn.commit()
    finally:
        conn.close()

    exit_code = main(
        ["verify", "--ledger-path", str(db_path), "--anchor-path", str(anchor_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "sequence_number=1" in captured.err


def test_main_verify_returns_one_on_mismatched_anchor_and_reports_sequence_number(
    tmp_path: Path,
    deterministic_clock: Callable[[], datetime],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main(["verify", ...])` returns 1 and reports the mismatched anchor's
    sequence_number on a tail-rewrite a bare chain check alone would miss.
    """
    db_path = tmp_path / "ledger.db"
    anchor_path = tmp_path / "anchors.jsonl"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    for beat in range(1, 4):
        store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=beat))
    store.close()
    anchor_head(db_path, anchor_path)  # anchors seq=3

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM ledger WHERE sequence_number >= 2")
        conn.commit()
    finally:
        conn.close()

    exit_code = main(
        ["verify", "--ledger-path", str(db_path), "--anchor-path", str(anchor_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "sequence_number=3" in captured.err


def test_main_verify_returns_one_on_missing_anchor_file(
    tmp_path: Path,
    deterministic_clock: Callable[[], datetime],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main(["verify", ...])` returns 1 with a stderr message when the anchor
    file itself is missing (fail closed: no anchors is never "verified").
    """
    db_path = tmp_path / "ledger.db"
    anchor_path = tmp_path / "does_not_exist.jsonl"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()

    exit_code = main(
        ["verify", "--ledger-path", str(db_path), "--anchor-path", str(anchor_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.strip() != ""


def _anchor(db_path: Path, anchor_path: Path) -> int:
    """Run the shipped `anchor` verb through `main`, as an operator's cron does.

    Args:
        db_path: The ``--ledger-path`` value to pass.
        anchor_path: The ``--anchor-path`` value to pass.

    Returns:
        The process exit code `main` returned.
    """
    return main(
        ["anchor", "--ledger-path", str(db_path), "--anchor-path", str(anchor_path)]
    )


def test_main_anchor_refuses_a_missing_ledger_path_and_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mistyped `--ledger-path` exits 1 with what/why/next and creates no ledger.

    This is the misconfigured-cron half of issue #217. Before the fix the store
    *created* the database at the typo'd path, so the run exited 0, wrote no
    anchor, and left a decoy ledger behind -- indistinguishable from success.
    """
    db_path = tmp_path.joinpath("typo.db")
    anchor_path = tmp_path.joinpath("anchors.jsonl")

    exit_code = _anchor(db_path, anchor_path)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err == (
        f"ledger not found at {db_path}: anchor reads an existing ledger and "
        "will not create one. Check --ledger-path, or run the pipeline first "
        "to produce a ledger.\n"
    )
    assert captured.out == ""
    assert not db_path.exists()
    assert not anchor_path.exists()


def test_main_anchor_reports_an_empty_ledger_then_anchors_once_it_has_a_head(
    tmp_path: Path,
    deterministic_clock: Callable[[], datetime],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An existing-but-empty ledger exits 0 with an explicit "nothing anchored".

    Two phases, because "no anchor file was written" is only meaningful next to
    a run that writes one: phase one anchors an empty ledger and pins the notice
    and the absent file; phase two -- the positive control -- appends one record
    to that same ledger and pins the resulting anchor line exactly. Without the
    control, the phase-one assertions would hold just as well against an
    `anchor` verb that had stopped working entirely.
    """
    db_path = tmp_path.joinpath("ledger.db")
    anchor_path = tmp_path.joinpath("anchors.jsonl")
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.close()

    empty_exit_code = _anchor(db_path, anchor_path)
    empty_capture = capsys.readouterr()

    assert empty_exit_code == 0
    assert empty_capture.err == (
        f"nothing anchored: the ledger at {db_path} holds no records, so there "
        f"is no head hash to pin and {anchor_path} was not written. Re-run "
        "anchor once the pipeline has appended its first event; if you "
        "expected records already, check --ledger-path.\n"
    )
    assert empty_capture.out == ""
    assert not anchor_path.exists()

    reopened = SqliteLedgerStore(db_path, now=deterministic_clock)
    reopened.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    reopened.close()

    anchored_exit_code = _anchor(db_path, anchor_path)
    anchored_capture = capsys.readouterr()

    assert anchored_exit_code == 0
    assert anchored_capture.err == ""
    lines = anchor_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_hash"] == _HEAD_HASH_AT_SEQUENCE_ONE
    assert json.loads(lines[0])["sequence_number"] == 1


def test_main_anchor_writes_the_exact_anchor_line_for_a_populated_ledger(
    tmp_path: Path,
    deterministic_clock: Callable[[], datetime],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A populated ledger still anchors to the exact bytes, silently, exit 0.

    The other direction of issue #217: the new empty-ledger reporting must not
    change what a real anchor writes. Asserts the whole line -- canonical JSON,
    sorted keys, compact separators, one trailing newline -- with the head hash
    pinned to a literal, and re-verifies the chain afterwards.
    """
    db_path = tmp_path.joinpath("ledger.db")
    anchor_path = tmp_path.joinpath("anchors.jsonl")
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()

    before = datetime.now(UTC)
    exit_code = _anchor(db_path, anchor_path)
    after = datetime.now(UTC)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == ""
    text = anchor_path.read_text(encoding="utf-8")
    assert len(text.splitlines()) == 1
    record = json.loads(text)
    assert record["sequence_number"] == 1
    assert record["event_hash"] == _HEAD_HASH_AT_SEQUENCE_ONE
    anchored_at = datetime.fromisoformat(record["anchored_at"])
    assert before <= anchored_at <= after
    assert text == (
        '{"anchored_at":"'
        + record["anchored_at"]
        + '","event_hash":"'
        + _HEAD_HASH_AT_SEQUENCE_ONE
        + '","sequence_number":1}\n'
    )
    reopened = SqliteLedgerStore(db_path, now=deterministic_clock)
    try:
        reopened.verify_chain()
    finally:
        reopened.close()


def test_build_parser_run_and_rebuild_subcommand_parsing_is_unchanged() -> None:
    """Adding `anchor`/`verify` must not disturb the pre-existing subcommands."""
    run_args = build_parser().parse_args(["run"])
    rebuild_args = build_parser().parse_args(
        ["rebuild", "--ledger-path", "ledger.db", "--output-dir", "out"]
    )

    assert run_args.command == "run"
    assert run_args.heartbeat_interval == 5.0
    assert run_args.max_beats is None
    assert rebuild_args.command == "rebuild"
    assert rebuild_args.ledger_path == Path("ledger.db")
