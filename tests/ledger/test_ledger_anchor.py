"""Tests for `windbreak.ledger.anchor` -- head-hash anchoring (issue #75).

The SQLite hash-chained ledger is only tamper-evident *relative to a trusted
head*: `SqliteLedgerStore.verify_chain()` alone cannot distinguish a
legitimately short chain from one whose tail was truncated and re-chained by
a writer with raw DB access. Anchoring closes that gap by appending the
chain's head `(sequence_number, event_hash)` to an append-only,
JSON-lines anchor file (`anchor_head`), and independently checking every
anchored position against the live chain (`verify_anchors`).

**THE ACCEPTANCE TEST** is the two `test_anchor_detects_*` cases below: a
tail-rewrite -- whether a pure truncation or a truncate-and-re-chain forgery
of identical length -- passes `verify_chain()` but is caught by
`verify_anchors()`.

Tampering is injected via a raw `sqlite3` connection directly against the
on-disk table (the literal `ledger` table name, per this suite's established
convention -- see `tests/ledger/conftest.py` -- keeps the SQL fully literal
and avoids bandit B608 false positives), since the public `LedgerStore` API
has no mutation method.

Fail-closed contract pinned here for `verify_anchors`: a missing anchor
file, or a present-but-empty one (zero anchor records), is never treated as
"nothing to verify" -- both raise `AnchorFormatError(line_number=1)`, the
same error a malformed anchor line raises (naming its 1-based line).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from windbreak.ledger.anchor import (
    AnchorFormatError,
    AnchorMismatchError,
    anchor_head,
    read_anchors,
    verify_anchors,
)
from windbreak.ledger.events import ModeHeartbeat, canonical_json
from windbreak.ledger.store import (
    ChainIntegrityError,
    SqliteLedgerStore,
    compute_event_hash,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: Number of rows in the acceptance test's baseline chain.
_ACCEPTANCE_CHAIN_LENGTH = 5

#: Sequence position both acceptance-test forgeries truncate back to (rows
#: 4 and 5 are removed/re-chained; rows 1-3 remain untouched).
_TRUNCATE_FROM_SEQUENCE = 4


def _build_five_row_chain(
    ledger_store_factory: Callable[..., SqliteLedgerStore], db_name: str
) -> None:
    """Append five ModeHeartbeat events via the public API, then close the store.

    Args:
        ledger_store_factory: The `ledger_store_factory` fixture.
        db_name: The database filename to build the chain at.
    """
    store = ledger_store_factory(db_name)
    for beat in range(1, _ACCEPTANCE_CHAIN_LENGTH + 1):
        store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=beat))
    store.close()


def _delete_rows_sequence_ge_four(db_path: Path) -> None:
    """Delete every row with sequence_number >= 4 via raw sqlite3 (tamper-only).

    Args:
        db_path: Path to the SQLite database file to tamper with.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM ledger WHERE sequence_number >= 4")
        conn.commit()
    finally:
        conn.close()


def _forge_rows_four_and_five(db_path: Path) -> str:
    """Re-chain forged rows 4 and 5 atop the untouched row 3 (tamper-only).

    Rows 4 and 5 are fully internally consistent -- each row's `event_hash`
    is recomputed via `compute_event_hash` over its own forged fields and the
    predecessor's hash, exactly as the real store would compute it -- but
    their payload differs from the originals, so the resulting head hash
    differs from the one already anchored.

    Args:
        db_path: Path to the SQLite database file to tamper with. Row 3 must
            already exist and be untouched.

    Returns:
        The forged row 5's `event_hash` (the forged chain's new head).
    """
    conn = sqlite3.connect(db_path)
    try:
        prev_hash = conn.execute(
            "SELECT event_hash FROM ledger WHERE sequence_number = 3"
        ).fetchone()[0]
        created_at_four = "2099-01-01T00:00:00.000000+00:00"
        payload_json_four = canonical_json(
            {
                "component": "pipeline",
                "data": {"mode": "RESEARCH", "beat": 999},
                "schema_version": 1,
            }
        )
        event_hash_four = compute_event_hash(
            4, "ModeHeartbeat", created_at_four, payload_json_four, prev_hash
        )
        conn.execute(
            "INSERT INTO ledger ("
            "sequence_number, event_type, created_at, component, "
            "payload_json, payload_schema_version, prev_hash, event_hash"
            ") VALUES (4, 'ModeHeartbeat', ?, 'pipeline', ?, 1, ?, ?)",
            (created_at_four, payload_json_four, prev_hash, event_hash_four),
        )
        created_at_five = "2099-01-01T00:00:01.000000+00:00"
        payload_json_five = canonical_json(
            {
                "component": "pipeline",
                "data": {"mode": "RESEARCH", "beat": 998},
                "schema_version": 1,
            }
        )
        event_hash_five = compute_event_hash(
            5, "ModeHeartbeat", created_at_five, payload_json_five, event_hash_four
        )
        conn.execute(
            "INSERT INTO ledger ("
            "sequence_number, event_type, created_at, component, "
            "payload_json, payload_schema_version, prev_hash, event_hash"
            ") VALUES (5, 'ModeHeartbeat', ?, 'pipeline', ?, 1, ?, ?)",
            (created_at_five, payload_json_five, event_hash_four, event_hash_five),
        )
        conn.commit()
    finally:
        conn.close()
    return event_hash_five


def _tamper_row_one_event_hash(db_path: Path) -> None:
    """Corrupt row 1's `event_hash` via raw sqlite3, breaking the whole chain.

    Args:
        db_path: Path to the SQLite database file to tamper with.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE ledger SET event_hash = ? WHERE sequence_number = 1", ("0" * 64,)
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# THE ACCEPTANCE TEST: a tail-rewrite is detected by anchor comparison.
# --------------------------------------------------------------------------


def test_anchor_detects_pure_truncation_tail_rewrite(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
    tmp_path: Path,
) -> None:
    """ACCEPTANCE (variant a): pure truncation past the anchored head is caught.

    Anchors the head at sequence_number=5, then deletes rows 4 and 5 so the
    live chain becomes a genuinely valid, self-consistent 3-row chain --
    `verify_chain()` alone cannot distinguish this from a chain that was
    always 3 rows long. Only `verify_anchors()` can: the position anchored at
    seq=5 no longer exists.
    """
    db_name = "truncate_only.db"
    _build_five_row_chain(ledger_store_factory, db_name)
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"

    anchor_head(db_path, anchor_path)
    _delete_rows_sequence_ge_four(db_path)

    reopened = SqliteLedgerStore(db_path)
    try:
        reopened.verify_chain()  # a genuinely valid, shorter chain passes
        records = reopened.read_all()
    finally:
        reopened.close()
    assert len(records) == _TRUNCATE_FROM_SEQUENCE - 1

    with pytest.raises(AnchorMismatchError) as exc_info:
        verify_anchors(db_path, anchor_path)

    assert exc_info.value.sequence_number == _ACCEPTANCE_CHAIN_LENGTH


def test_anchor_detects_truncate_and_rechain_to_same_length(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
    tmp_path: Path,
) -> None:
    """ACCEPTANCE (variant b): a same-length truncate+re-chain forgery is caught.

    Anchors the head at sequence_number=5, deletes rows 4 and 5, then
    re-inserts forged rows 4 and 5 that are fully internally consistent (own
    hashes correctly recomputed) but carry different payloads than the
    originals -- so the forged head's hash differs from the anchored one.
    `verify_chain()` cannot tell this 5-row forged chain from the original
    5-row chain; only anchor comparison can, because the hash at the
    anchored position (seq=5) no longer matches.
    """
    db_name = "rechain.db"
    _build_five_row_chain(ledger_store_factory, db_name)
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"

    anchor_head(db_path, anchor_path)
    original_head = read_anchors(anchor_path)[0]

    _delete_rows_sequence_ge_four(db_path)
    forged_head_hash = _forge_rows_four_and_five(db_path)

    assert forged_head_hash != original_head.event_hash

    reopened = SqliteLedgerStore(db_path)
    try:
        reopened.verify_chain()  # internally consistent forged chain still passes
        records = reopened.read_all()
    finally:
        reopened.close()
    assert len(records) == _ACCEPTANCE_CHAIN_LENGTH
    assert records[-1].sequence_number == _ACCEPTANCE_CHAIN_LENGTH
    assert records[-1].event_hash == forged_head_hash

    with pytest.raises(AnchorMismatchError) as exc_info:
        verify_anchors(db_path, anchor_path)

    assert exc_info.value.sequence_number == _ACCEPTANCE_CHAIN_LENGTH


# --------------------------------------------------------------------------
# anchor_head: happy path and append-only discipline.
# --------------------------------------------------------------------------


def test_anchor_head_appends_exactly_one_well_formed_json_line(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """One anchor_head() call appends exactly one line pinned to the head.

    The line's `sequence_number`/`event_hash` equal the head row's exactly,
    and its `anchored_at` is a parseable ISO-8601 timestamp.
    """
    db_name = "single_anchor.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    head_record = store.read_all()[-1]
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"

    anchor_head(db_path, anchor_path)

    lines = anchor_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["sequence_number"] == head_record.sequence_number
    assert record["event_hash"] == head_record.event_hash
    datetime.fromisoformat(record["anchored_at"])  # must parse as ISO-8601


def test_anchor_head_writes_canonical_json_sorted_compact_with_trailing_newline(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """The written line is canonical JSON: sorted keys, compact separators, one \\n."""
    db_name = "canonical.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    head_record = store.read_all()[-1]
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"

    anchor_head(db_path, anchor_path)

    raw = anchor_path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    body = raw.decode("utf-8").rstrip("\n")
    parsed = json.loads(body)
    expected = canonical_json(
        {
            "anchored_at": parsed["anchored_at"],
            "event_hash": head_record.event_hash,
            "sequence_number": head_record.sequence_number,
        }
    )
    assert body == expected
    assert ", " not in body
    assert ": " not in body


def test_anchor_head_twice_appends_two_lines_first_line_byte_unchanged(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """Anchoring twice appends a second line, leaving the first unchanged."""
    db_name = "double_anchor.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"

    anchor_head(db_path, anchor_path)
    first_line_after_first_call = anchor_path.read_text(encoding="utf-8").splitlines()[
        0
    ]

    reopened = SqliteLedgerStore(db_path)
    reopened.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=2))
    reopened.close()
    anchor_head(db_path, anchor_path)

    lines = anchor_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == first_line_after_first_call


def test_anchor_head_on_empty_ledger_appends_nothing_and_returns_none(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """An empty ledger appends nothing, raises nothing, and returns None (#217).

    The `None` is the whole signal: `anchor_head` is a library function, so it
    reports "no head to anchor" to its caller rather than printing, and
    `anchor_command` is what turns that into the operator-facing notice.
    """
    db_name = "empty.db"
    ledger_store_factory(db_name)
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"

    anchored = anchor_head(db_path, anchor_path)

    assert anchored is None
    assert not anchor_path.exists()


def test_anchor_head_returns_the_head_it_anchored(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """A populated ledger returns the very head whose hash was written (#217)."""
    db_name = "returns_head.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    head_record = store.read_all()[-1]
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"

    anchored = anchor_head(db_path, anchor_path)

    assert anchored is not None
    assert anchored.sequence_number == head_record.sequence_number
    assert anchored.event_hash == head_record.event_hash


def test_anchor_head_refuses_a_missing_ledger_path_and_creates_no_database(
    tmp_path: Path,
) -> None:
    """A `ledger_path` naming no file raises and creates neither DB nor anchor (#217).

    Opening a `SqliteLedgerStore` creates the database, so without the guard a
    mistyped path left a decoy empty ledger behind and anchored nothing.
    """
    db_path = tmp_path / "never_created.db"
    anchor_path = tmp_path / "anchors.jsonl"

    with pytest.raises(FileNotFoundError) as exc_info:
        anchor_head(db_path, anchor_path)

    assert str(exc_info.value) == (
        f"ledger not found at {db_path}: anchor reads an existing ledger and "
        "will not create one. Check --ledger-path, or run the pipeline first "
        "to produce a ledger."
    )
    assert not db_path.exists()
    assert not anchor_path.exists()


def test_anchor_head_refuses_a_ledger_path_that_is_a_directory(
    tmp_path: Path,
) -> None:
    """A directory at `ledger_path` is a missing ledger, not an openable one (#217).

    `Path.exists()` would accept it and hand SQLite an unopenable path; the
    guard tests `is_file()` so the operator reads the same guidance either way.
    """
    db_path = tmp_path / "a_directory.db"
    db_path.mkdir()
    anchor_path = tmp_path / "anchors.jsonl"

    with pytest.raises(FileNotFoundError):
        anchor_head(db_path, anchor_path)

    assert not anchor_path.exists()


def test_anchor_head_on_broken_chain_raises_and_creates_no_file(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """A tampered chain fails verify_chain inside anchor_head and appends nothing."""
    db_name = "broken.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"

    _tamper_row_one_event_hash(db_path)

    with pytest.raises(ChainIntegrityError):
        anchor_head(db_path, anchor_path)

    assert not anchor_path.exists()


def test_anchor_head_on_broken_chain_leaves_existing_anchor_file_byte_unchanged(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """A later broken-chain anchor attempt never mutates a pre-existing anchor file."""
    db_name = "broken_after_good.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"

    anchor_head(db_path, anchor_path)
    before = anchor_path.read_bytes()

    _tamper_row_one_event_hash(db_path)

    with pytest.raises(ChainIntegrityError):
        anchor_head(db_path, anchor_path)

    assert anchor_path.read_bytes() == before


def test_anchor_head_creates_missing_anchor_parent_directory(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """anchor_head() creates the anchor file's parent directory if absent."""
    db_name = "parent_dir.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "nested" / "anchors" / "anchors.jsonl"

    anchor_head(db_path, anchor_path)

    assert anchor_path.exists()
    assert len(anchor_path.read_text(encoding="utf-8").splitlines()) == 1


# --------------------------------------------------------------------------
# read_anchors: parsing contract.
# --------------------------------------------------------------------------


def test_read_anchors_returns_one_record_per_line_in_file_order(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """read_anchors() parses each JSON line into a record exposing the three
    anchor fields, in file (append) order.
    """
    db_name = "read_anchors.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"
    anchor_head(db_path, anchor_path)

    reopened = SqliteLedgerStore(db_path)
    reopened.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=2))
    reopened.close()
    anchor_head(db_path, anchor_path)

    records = read_anchors(anchor_path)

    assert len(records) == 2
    assert records[0].sequence_number == 1
    assert records[1].sequence_number == 2
    assert records[0].event_hash != records[1].event_hash
    assert isinstance(records[0].anchored_at, str)


_MALFORMED_ANCHOR_LINES = [
    pytest.param("this is not json{", id="invalid-json"),
    pytest.param(
        json.dumps(
            {"anchored_at": "2024-01-01T00:00:00+00:00", "event_hash": "a" * 64}
        ),
        id="missing-sequence-number-key",
    ),
    pytest.param(
        json.dumps(
            {
                "anchored_at": "2024-01-01T00:00:00+00:00",
                "event_hash": "a" * 64,
                "sequence_number": "not-an-int",
            }
        ),
        id="wrong-type-sequence-number",
    ),
    pytest.param(
        json.dumps(
            {
                "anchored_at": "2024-01-01T00:00:00+00:00",
                "event_hash": "a" * 64,
                "sequence_number": 0,
            }
        ),
        id="zero-sequence-number",
    ),
    pytest.param(
        json.dumps(
            {
                "anchored_at": "2024-01-01T00:00:00+00:00",
                "event_hash": "a" * 64,
                "sequence_number": -1,
            }
        ),
        id="negative-sequence-number",
    ),
]


@pytest.mark.parametrize("malformed_line", _MALFORMED_ANCHOR_LINES)
def test_read_anchors_raises_anchor_format_error_on_malformed_line(
    tmp_path: Path, malformed_line: str
) -> None:
    """A malformed anchor line raises AnchorFormatError naming its 1-based line.

    The file's first line is well-formed so the failure is unambiguously
    attributable to the second (malformed) line, pinning 1-based numbering.
    """
    anchor_path = tmp_path / "malformed.jsonl"
    good_line = canonical_json(
        {
            "anchored_at": "2024-01-01T00:00:00+00:00",
            "event_hash": "b" * 64,
            "sequence_number": 1,
        }
    )
    anchor_path.write_text(good_line + "\n" + malformed_line + "\n", encoding="utf-8")

    with pytest.raises(AnchorFormatError) as exc_info:
        read_anchors(anchor_path)

    assert exc_info.value.line_number == 2


# --------------------------------------------------------------------------
# verify_anchors: happy paths.
# --------------------------------------------------------------------------


def test_verify_anchors_passes_for_single_anchor_and_untampered_chain(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """A single anchor against its own untampered chain verifies cleanly."""
    db_name = "happy_single.db"
    store = ledger_store_factory(db_name)
    for beat in range(1, 4):
        store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=beat))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"
    anchor_head(db_path, anchor_path)

    verify_anchors(db_path, anchor_path)  # must not raise


def test_verify_anchors_passes_when_chain_legitimately_grows_past_the_anchor(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """A chain that legitimately grows past a stale anchor still verifies."""
    db_name = "grown.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"
    anchor_head(db_path, anchor_path)

    reopened = SqliteLedgerStore(db_path)
    reopened.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=2))
    reopened.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=3))
    reopened.close()

    verify_anchors(db_path, anchor_path)  # must not raise


def test_verify_anchors_passes_for_multiple_anchors_across_growth(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """Two anchors, taken before and after legitimate growth, both verify."""
    db_name = "multi_anchor.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"
    anchor_head(db_path, anchor_path)

    reopened = SqliteLedgerStore(db_path)
    reopened.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=2))
    reopened.close()
    anchor_head(db_path, anchor_path)

    verify_anchors(db_path, anchor_path)  # must not raise
    assert len(anchor_path.read_text(encoding="utf-8").splitlines()) == 2


# --------------------------------------------------------------------------
# verify_anchors: fail-closed and error-attribution edge cases.
# --------------------------------------------------------------------------


def test_verify_anchors_fails_closed_on_missing_anchor_file(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """A missing anchor file fails closed: "no anchors" is never "verified"."""
    db_name = "no_anchor_file.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(AnchorFormatError) as exc_info:
        verify_anchors(db_path, anchor_path)

    assert exc_info.value.line_number == 1


def test_verify_anchors_fails_closed_on_empty_anchor_file(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """A present-but-empty anchor file (zero anchor records) also fails closed."""
    db_name = "empty_anchor_file.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "empty.jsonl"
    anchor_path.write_text("", encoding="utf-8")

    with pytest.raises(AnchorFormatError) as exc_info:
        verify_anchors(db_path, anchor_path)

    assert exc_info.value.line_number == 1


def test_verify_anchors_raises_anchor_format_error_on_malformed_anchor_line(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """verify_anchors surfaces a malformed anchor line as AnchorFormatError."""
    db_name = "malformed_verify.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "malformed.jsonl"
    anchor_path.write_text("not json at all\n", encoding="utf-8")

    with pytest.raises(AnchorFormatError) as exc_info:
        verify_anchors(db_path, anchor_path)

    assert exc_info.value.line_number == 1


def test_verify_anchors_raises_on_anchor_past_the_live_head(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """An anchor whose sequence_number exceeds the live head is a mismatch."""
    db_name = "past_head.db"
    store = ledger_store_factory(db_name)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"
    forged_far_future_seq = 42
    anchor_path.write_text(
        canonical_json(
            {
                "anchored_at": "2024-01-01T00:00:00+00:00",
                "event_hash": "c" * 64,
                "sequence_number": forged_far_future_seq,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AnchorMismatchError) as exc_info:
        verify_anchors(db_path, anchor_path)

    assert exc_info.value.sequence_number == forged_far_future_seq


def test_verify_anchors_fails_closed_on_non_positive_anchor_sequence_number(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """A non-positive anchored sequence_number fails closed, never negative-indexed.

    A structurally malformed anchor claiming ``sequence_number=0`` must raise
    :class:`AnchorFormatError` at its line, not be silently accepted and then
    resolved via ``records[0 - 1]`` -- Python negative indexing -- into the
    live chain's *last* row, which would check the wrong position (and could be
    coaxed into a false pass). The chain here is untampered, so any leak of the
    old fail-open path would surface as a spurious clean verify.
    """
    db_name = "non_positive_seq.db"
    store = ledger_store_factory(db_name)
    for beat in range(1, 4):
        store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=beat))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"
    anchor_path.write_text(
        canonical_json(
            {
                "anchored_at": "2024-01-01T00:00:00+00:00",
                "event_hash": "e" * 64,
                "sequence_number": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AnchorFormatError) as exc_info:
        verify_anchors(db_path, anchor_path)

    assert exc_info.value.line_number == 1


def test_verify_anchors_raises_when_ledger_is_empty_but_anchor_exists(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """Any anchor at all against an empty live ledger is a missing-position mismatch."""
    db_name = "empty_but_anchored.db"
    ledger_store_factory(db_name)
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"
    anchor_path.write_text(
        canonical_json(
            {
                "anchored_at": "2024-01-01T00:00:00+00:00",
                "event_hash": "d" * 64,
                "sequence_number": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AnchorMismatchError) as exc_info:
        verify_anchors(db_path, anchor_path)

    assert exc_info.value.sequence_number == 1


def test_verify_anchors_reports_the_first_anchor_record_not_a_later_one(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """A rollback past two anchors is reported at the FIRST anchor record in the
    file (seq=5), not the later one (seq=8), pinning file-order scanning.

    Builds an 8-row chain, anchors at seq=5 then again at seq=8, then rolls
    the live chain back to 3 rows (both anchors are now past the head). The
    first anchor record encountered when scanning the file top-to-bottom is
    the seq=5 record, so that is the one `verify_anchors` must report.
    """
    db_name = "rollback.db"
    store = ledger_store_factory(db_name)
    for beat in range(1, _ACCEPTANCE_CHAIN_LENGTH + 1):
        store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=beat))
    store.close()
    db_path = tmp_path / db_name
    anchor_path = tmp_path / "anchors.jsonl"
    anchor_head(db_path, anchor_path)  # anchors seq=5

    reopened = SqliteLedgerStore(db_path)
    for beat in range(_ACCEPTANCE_CHAIN_LENGTH + 1, 9):
        reopened.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=beat))
    reopened.close()
    anchor_head(db_path, anchor_path)  # anchors seq=8

    _delete_rows_sequence_ge_four(db_path)  # rollback: live head is now 3

    with pytest.raises(AnchorMismatchError) as exc_info:
        verify_anchors(db_path, anchor_path)

    assert exc_info.value.sequence_number == _ACCEPTANCE_CHAIN_LENGTH


# --------------------------------------------------------------------------
# Exception contracts, tested in isolation for mutation resistance.
# --------------------------------------------------------------------------


def test_anchor_mismatch_error_message_contains_sequence_number() -> None:
    """AnchorMismatchError's str() mirrors ChainIntegrityError's message contract."""
    error = AnchorMismatchError(7)

    assert error.sequence_number == 7
    assert "sequence_number=7" in str(error)


def test_anchor_format_error_exposes_line_number_attribute() -> None:
    """AnchorFormatError carries the offending 1-based line_number."""
    error = AnchorFormatError(3)

    assert error.line_number == 3
