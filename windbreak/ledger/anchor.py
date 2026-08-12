"""Head-hash anchoring for tail-rewrite tamper detection (issue #75).

:meth:`~windbreak.ledger.store.SqliteLedgerStore.verify_chain` proves a ledger
is *internally* consistent, but it cannot tell a legitimately short chain from
one whose tail a writer with raw database access truncated and re-chained: both
verify cleanly. Anchoring closes that gap by recording the chain's head
``(sequence_number, event_hash)`` to an append-only, JSON-lines *anchor file*
(:func:`anchor_head`) and independently checking every anchored position back
against the live chain (:func:`verify_anchors`). A head hash that no longer
matches its anchor -- or an anchored position that no longer exists -- is a
tail rewrite, reported as :class:`AnchorMismatchError` at the offending
sequence number.

The anchor file fails *closed*: a missing or empty file is never read as
"nothing to verify" -- both surface as :class:`AnchorFormatError`, the same
error a malformed line raises.

**Trust boundary (read before relying on this control).** Anchoring only moves
the trust root; it does not eliminate it. The guarantee is strictly
*conditional*: it holds only while the anchor file is protected from the same
writer who can reach the ledger database. An attacker who controls **both** the
ledger DB **and** the anchor file can truncate the chain, re-chain a forged
tail, and simply append a fresh anchor pinning the forged head -- both
:func:`anchor_head` and :func:`verify_anchors` would then pass. The tamper
evidence is therefore only as strong as the *separation* between the two sinks:
put the anchor file on a separately-permissioned volume, an append-only /
write-once medium, or a remote/off-host sink the ledger writer cannot rewrite.
On the same filesystem, under the same principal, this detects accidental
corruption and unsophisticated tampering, not a determined writer with full
local access.

``anchor_command``/``verify_command`` adapt the two operations to the
``windbreak anchor`` / ``windbreak verify`` CLI verbs, returning 0 on success
and 1 (with the offending detail on stderr) on any domain failure. ``anchor``
additionally separates the two ways it can end up writing no anchor at all
(issue #217): a ``--ledger-path`` naming no existing file is refused with exit
1, while an existing ledger holding no records exits 0 with an explicit
"nothing anchored" notice. Neither is silent, because a scheduled anchor that
prints nothing and exits 0 is indistinguishable from one that worked.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from windbreak.ledger.events import canonical_json
from windbreak.ledger.store import ChainIntegrityError, SqliteLedgerStore

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable
    from pathlib import Path

    from windbreak.ledger.store import ChainHead, LedgerRecord

#: JSON key holding an anchor record's 1-based chain position.
_SEQUENCE_NUMBER_KEY = "sequence_number"

#: JSON key holding an anchor record's chained SHA-256 head hash.
_EVENT_HASH_KEY = "event_hash"

#: JSON key holding an anchor record's ISO-8601 anchoring timestamp.
_ANCHORED_AT_KEY = "anchored_at"

#: ``isoformat`` precision stamped on an anchor's ``anchored_at``, mirroring the
#: microsecond precision the store stamps on each record's ``created_at``.
_TIMESTAMP_PRECISION = "microseconds"

#: The 1-based line number reported when the anchor file itself is missing or
#: holds zero records -- the fail-closed contract treats "no anchors" as a
#: malformed first line rather than a silent pass.
_FAIL_CLOSED_LINE_NUMBER = 1

#: The smallest structurally valid anchored chain position. Chain sequence
#: numbers are 1-based (:mod:`windbreak.ledger.store`), so any anchor claiming
#: ``sequence_number < 1`` is malformed: accepting it would let a
#: zero/negative value index the live-record list from the *end*
#: (``records[seq - 1]``), checking the wrong row instead of failing closed.
_MIN_SEQUENCE_NUMBER = 1


def _missing_ledger_message(ledger_path: Path) -> str:
    """Build the refusal shown when ``--ledger-path`` names no existing file.

    Deliberately mirrors :func:`windbreak.ledger.rebuild.rebuild`'s
    missing-path wording (issue #76): both verbs *read* a ledger, neither
    creates one, and an operator who mistyped the flag should read the same
    sentence whichever one they ran.

    Args:
        ledger_path: The path that does not name an existing file.

    Returns:
        The what/why/next refusal text.
    """
    return (
        f"ledger not found at {ledger_path}: anchor reads an existing ledger "
        "and will not create one. Check --ledger-path, or run the pipeline "
        "first to produce a ledger."
    )


def _empty_ledger_message(ledger_path: Path, anchor_path: Path) -> str:
    """Build the notice shown when the ledger exists but holds no records.

    Not a refusal: a ledger that has genuinely never been appended to has no
    head, and anchoring nothing is the correct outcome (issue #217). What was
    wrong was doing it in silence, which read as a successful anchor.

    Args:
        ledger_path: The empty ledger that was read.
        anchor_path: The anchor file that was consequently not written.

    Returns:
        The what/why/next notice text.
    """
    return (
        f"nothing anchored: the ledger at {ledger_path} holds no records, so "
        f"there is no head hash to pin and {anchor_path} was not written. "
        "Re-run anchor once the pipeline has appended its first event; if you "
        "expected records already, check --ledger-path."
    )


def _default_clock() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    Returns:
        ``datetime.now`` in the UTC timezone.
    """
    return datetime.now(UTC)


@dataclass(frozen=True)
class AnchorRecord:
    """One anchored chain head, as persisted to a line of the anchor file.

    Attributes:
        sequence_number: The anchored head's 1-based chain position.
        event_hash: The anchored head's chained SHA-256 ``event_hash``.
        anchored_at: The ISO-8601 instant the anchor was written.
    """

    sequence_number: int
    event_hash: str
    anchored_at: str


class AnchorMismatchError(Exception):
    """Raised when a live chain no longer matches one of its anchors.

    Mirrors :class:`~windbreak.ledger.store.ChainIntegrityError`'s message
    contract so both surface an offending position uniformly.

    Attributes:
        sequence_number: The anchored position whose live hash is absent or no
            longer matches the anchored one.
    """

    def __init__(self, sequence_number: int) -> None:
        """Initialize the error with the offending anchored position.

        Args:
            sequence_number: The anchored sequence position that failed to
                match the live chain.
        """
        self.sequence_number = sequence_number
        super().__init__(f"ledger anchor mismatch at sequence_number={sequence_number}")


class AnchorFormatError(Exception):
    """Raised when an anchor file line cannot be parsed as an anchor record.

    Covers invalid JSON, a missing required key, or a wrongly-typed value, and
    -- per the fail-closed contract -- a missing or empty anchor file (reported
    at line 1).

    Attributes:
        line_number: The offending 1-based line number in the anchor file.
    """

    def __init__(self, line_number: int) -> None:
        """Initialize the error with the offending 1-based line number.

        Args:
            line_number: The 1-based line number that failed to parse.
        """
        self.line_number = line_number
        super().__init__(f"malformed anchor record at line_number={line_number}")


def _parse_anchor_line(line: str, line_number: int) -> AnchorRecord:
    """Parse one anchor-file line into an :class:`AnchorRecord`.

    Args:
        line: The raw line text (without its trailing newline).
        line_number: The line's 1-based position, reported on failure.

    Returns:
        The parsed anchor record.

    Raises:
        AnchorFormatError: If the line is not valid JSON, is missing a required
            key, carries a wrongly-typed value, or claims a non-positive
            ``sequence_number`` (chain positions are 1-based).
    """
    try:
        parsed: object = json.loads(line)
    except json.JSONDecodeError as exc:
        raise AnchorFormatError(line_number) from exc
    if not isinstance(parsed, dict):
        raise AnchorFormatError(line_number)
    sequence_number = parsed.get(_SEQUENCE_NUMBER_KEY)
    event_hash = parsed.get(_EVENT_HASH_KEY)
    anchored_at = parsed.get(_ANCHORED_AT_KEY)
    if (
        not isinstance(sequence_number, int)
        or isinstance(sequence_number, bool)
        or sequence_number < _MIN_SEQUENCE_NUMBER
        or not isinstance(event_hash, str)
        or not isinstance(anchored_at, str)
    ):
        raise AnchorFormatError(line_number)
    return AnchorRecord(
        sequence_number=sequence_number,
        event_hash=event_hash,
        anchored_at=anchored_at,
    )


def read_anchors(anchor_path: Path) -> list[AnchorRecord]:
    """Read an anchor file into its :class:`AnchorRecord` list, in file order.

    Each non-blank line is parsed as canonical anchor JSON. The read fails
    closed: a missing file, or one holding zero anchor records, raises
    :class:`AnchorFormatError` at line 1 rather than returning an empty list,
    so "no anchors" can never be mistaken for "verified".

    Args:
        anchor_path: Path to the append-only JSON-lines anchor file.

    Returns:
        The parsed anchor records, in append (file) order.

    Raises:
        AnchorFormatError: If the file is missing or empty, or any line is
            malformed (reporting that line's 1-based number).
    """
    if not anchor_path.exists():
        raise AnchorFormatError(_FAIL_CLOSED_LINE_NUMBER)
    lines = anchor_path.read_text(encoding="utf-8").splitlines()
    records: list[AnchorRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        records.append(_parse_anchor_line(line, line_number))
    if not records:
        raise AnchorFormatError(_FAIL_CLOSED_LINE_NUMBER)
    return records


def _read_verified_head(ledger_path: Path) -> ChainHead | None:
    """Verify the chain at ``ledger_path`` and return its head.

    Args:
        ledger_path: Path to the SQLite ledger database.

    Returns:
        The verified chain's head, or ``None`` for an empty ledger.

    Raises:
        FileNotFoundError: If ``ledger_path`` does not point at an existing
            file. Opening a :class:`SqliteLedgerStore` *creates* the database,
            so without this guard a mistyped ``--ledger-path`` would produce a
            fresh empty ledger there and anchor nothing (issue #217).
        ChainIntegrityError: If the ledger's hash chain fails verification.
    """
    if not ledger_path.is_file():
        raise FileNotFoundError(_missing_ledger_message(ledger_path))
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        return store.head()
    finally:
        store.close()


def _append_anchor_line(
    anchor_path: Path, head: ChainHead, now: Callable[[], datetime]
) -> None:
    """Append one canonical anchor line pinning ``head`` to the anchor file.

    Creates the anchor file's parent directory if absent and appends a single
    canonical-JSON line (sorted keys, compact separators, one trailing
    newline) in ``"a"`` mode, so earlier anchors are never rewritten.

    Args:
        anchor_path: Path to the append-only JSON-lines anchor file.
        head: The verified chain head to anchor.
        now: Clock supplying the ``anchored_at`` timestamp.
    """
    # ``Path.parent`` + ``mkdir`` (never the ``/`` operator) keeps this
    # money-path module clear of the no-float lint's blanket true-division ban
    # (SPEC S6.1); the append is byte-identical either way.
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    record = canonical_json(
        {
            _ANCHORED_AT_KEY: now().isoformat(timespec=_TIMESTAMP_PRECISION),
            _EVENT_HASH_KEY: head.event_hash,
            _SEQUENCE_NUMBER_KEY: head.sequence_number,
        }
    )
    with anchor_path.open("a", encoding="utf-8") as anchor_file:
        anchor_file.write(record + "\n")


def anchor_head(
    ledger_path: Path,
    anchor_path: Path,
    *,
    now: Callable[[], datetime] = _default_clock,
) -> ChainHead | None:
    """Verify the ledger and append its head to the anchor file.

    Refuses a ``ledger_path`` that names no existing file, because opening the
    store would otherwise *create* one and anchor its emptiness. Then verifies
    the hash chain, so a broken chain raises :class:`ChainIntegrityError` and
    nothing is appended (and no anchor file is created if it did not already
    exist). An empty ledger appends nothing and returns ``None`` -- there is no
    head to anchor; reporting that outcome belongs to the caller, since this is
    a library function and the operator-facing wording is
    :func:`anchor_command`'s (issue #217). Otherwise exactly one
    canonical-JSON line pinning the head ``(sequence_number, event_hash)`` is
    appended.

    Args:
        ledger_path: Path to the SQLite ledger database.
        anchor_path: Path to the append-only JSON-lines anchor file.
        now: Clock supplying the anchor's ``anchored_at`` timestamp. Injectable
            for deterministic tests.

    Returns:
        The head that was anchored, or ``None`` when the ledger held no records
        and nothing was written.

    Raises:
        FileNotFoundError: If ``ledger_path`` does not point at an existing
            file.
        ChainIntegrityError: If the ledger's hash chain fails verification.
    """
    head = _read_verified_head(ledger_path)
    if head is None:
        return None
    _append_anchor_line(anchor_path, head, now)
    return head


def _check_anchor(anchor: AnchorRecord, records: list[LedgerRecord]) -> None:
    """Check one anchor against the live chain, raising on a violation.

    Args:
        anchor: The anchored head to check.
        records: The verified live records, in ascending sequence order;
            contiguity lets ``records[seq - 1]`` be the row at ``seq``.

    Raises:
        AnchorMismatchError: If the anchored position exceeds the live head, or
            the live hash at that position differs from the anchored one.
    """
    if anchor.sequence_number > len(records):
        raise AnchorMismatchError(anchor.sequence_number)
    live = records[anchor.sequence_number - 1]
    if live.event_hash != anchor.event_hash:
        raise AnchorMismatchError(anchor.sequence_number)


def verify_anchors(ledger_path: Path, anchor_path: Path) -> None:
    """Verify every anchor against the live chain, raising on the first breach.

    Verifies the hash chain first (a broken chain raises
    :class:`ChainIntegrityError`), then reads the anchor file -- failing closed
    on a missing or empty one -- and checks each anchor, in file order, against
    the live chain. The first anchored position that no longer exists or whose
    live hash no longer matches raises :class:`AnchorMismatchError`.

    Args:
        ledger_path: Path to the SQLite ledger database.
        anchor_path: Path to the append-only JSON-lines anchor file.

    Raises:
        ChainIntegrityError: If the ledger's hash chain fails verification.
        AnchorFormatError: If the anchor file is missing, empty, or malformed.
        AnchorMismatchError: If any anchored position no longer matches the
            live chain (reporting the first such position in file order).
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    for anchor in read_anchors(anchor_path):
        _check_anchor(anchor, records)


def anchor_command(args: Namespace) -> int:
    """Run :func:`anchor_head` for the CLI, mapping each outcome to an exit code.

    Three outcomes, deliberately kept distinct (issue #217), because a
    scheduled ``anchor`` that reports success while writing nothing is only
    discovered at ``verify`` time, when the window it was meant to cover is
    already gone:

    * a **missing** ``--ledger-path`` is a refusal -- exit 1, nothing created,
      the same wording ``rebuild`` uses for the same mistake;
    * an **existing but empty** ledger is a reportable no-op -- exit 0, with an
      explicit "nothing anchored" notice on stderr, because a ledger that has
      genuinely never been appended to is not a failure;
    * anything else anchors the head and stays silent.

    Args:
        args: Parsed CLI arguments exposing ``ledger_path`` and ``anchor_path``.

    Returns:
        0 on a clean anchor and on an empty ledger (the latter with the
        nothing-anchored notice on stderr); 1 if the ledger path does not
        exist or the chain fails verification (with the missing-path guidance
        or the offending ``sequence_number`` on stderr).
    """
    try:
        head = anchor_head(args.ledger_path, args.anchor_path)
    except (ChainIntegrityError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 1
    if head is None:
        print(
            _empty_ledger_message(args.ledger_path, args.anchor_path),
            file=sys.stderr,
        )
    return 0


def verify_command(args: Namespace) -> int:
    """Run :func:`verify_anchors` for the CLI, mapping failure to an exit code.

    Args:
        args: Parsed CLI arguments exposing ``ledger_path`` and ``anchor_path``.

    Returns:
        0 when every anchor verifies; 1 on a chain-integrity break, a malformed
        or missing anchor file, or an anchor mismatch (with the offending
        detail printed to stderr).
    """
    try:
        verify_anchors(args.ledger_path, args.anchor_path)
    except (ChainIntegrityError, AnchorMismatchError, AnchorFormatError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
