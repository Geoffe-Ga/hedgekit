"""SQLite-backed, append-only, hash-chained ledger store (SPEC S5.1, §12).

Persists a tamper-evident log of :class:`~windbreak.ledger.events.Event`
records. Each row carries a monotonically increasing ``sequence_number``
starting at 1 and a SHA-256 ``event_hash`` that chains to its
predecessor's hash (the first row chains to
:data:`~windbreak.ledger.events.GENESIS_PREV_HASH`). Because any change to a
persisted row breaks the chain, :meth:`SqliteLedgerStore.verify_chain` can
detect corruption of any single column.

The store only ever inserts and reads rows -- it exposes no mutation path
by design, which is what makes the log trustworthy as an audit trail. The
package's SQL is statically checked to remain insert-and-select only.

Beyond the four :class:`LedgerStore` operations every store supports, a store
may additionally declare the *optional*, narrow :class:`LatestRecordLookup`
capability: an indexed reverse read answering "the newest record of these event
types" without replaying the whole log (issue #246). It is deliberately a
separate protocol rather than three more lines on :class:`LedgerStore`, because
:class:`LedgerStore` is satisfied structurally by hand-rolled doubles that must
keep working untouched; a consumer duck-type dispatches onto the capability when
present and falls back to a full scan when it is not.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from windbreak.ledger.events import GENESIS_PREV_HASH, Event

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable
    from pathlib import Path

#: DDL creating the eight-column §12 ledger row if it does not yet exist.
_CREATE_TABLE_SQL = (
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

_INSERT_SQL = (
    "INSERT INTO ledger ("
    "sequence_number, event_type, created_at, component, "
    "payload_json, payload_schema_version, prev_hash, event_hash"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

_SELECT_ALL_SQL = (
    "SELECT sequence_number, event_type, created_at, component, "
    "payload_json, payload_schema_version, prev_hash, event_hash "
    "FROM ledger ORDER BY sequence_number"
)

#: Covering index backing :meth:`SqliteLedgerStore.latest_record_of_types`.
#: Without it, "newest row of type X" walks the primary key backwards until it
#: meets a match -- and the rows that read is asked about (a gate-plan
#: registration, written once at startup) sit near the *front* of a ledger that
#: grows without bound, so the walk degenerates to a full scan. The composite
#: ``(event_type, sequence_number)`` order lets SQLite seek straight to the last
#: entry for a type. Created on first use, like the table itself.
_CREATE_EVENT_TYPE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS ledger_event_type_sequence "
    "ON ledger (event_type, sequence_number)"
)

_SELECT_LAST_SQL = (
    "SELECT sequence_number, event_hash FROM ledger "
    "ORDER BY sequence_number DESC LIMIT 1"
)

#: Newest row of ONE event type. Held to a single, fully literal statement with
#: one bound parameter -- rather than a dynamically assembled ``IN (?, ?, ...)``
#: -- so the package's SQL stays statically auditable and free of string-built
#: queries; a caller wanting several types issues this read once per type and
#: keeps the highest sequence number.
_SELECT_LATEST_OF_TYPE_SQL = (
    "SELECT sequence_number, event_type, created_at, component, "
    "payload_json, payload_schema_version, prev_hash, event_hash "
    "FROM ledger WHERE event_type = ? ORDER BY sequence_number DESC LIMIT 1"
)


def _default_clock() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    Returns:
        ``datetime.now`` in the UTC timezone.
    """
    return datetime.now(UTC)


def compute_event_hash(
    sequence_number: int,
    event_type: str,
    created_at: str,
    payload_json: str,
    prev_hash: str,
) -> str:
    """Compute a record's chained SHA-256 hash from its §12 fields.

    Hashes the exact concatenation
    ``str(sequence_number) + event_type + created_at + payload_json +
    prev_hash``, so the digest binds the record's position, type,
    timestamp, payload, and its link to the predecessor's hash.

    Args:
        sequence_number: The record's 1-based position in the chain.
        event_type: The record's event type discriminator.
        created_at: The record's ISO-8601 creation timestamp.
        payload_json: The canonical envelope JSON persisted for the record.
        prev_hash: The predecessor's ``event_hash`` (genesis for the first).

    Returns:
        The 64-character hex SHA-256 digest.
    """
    digest_input = (
        str(sequence_number) + event_type + created_at + payload_json + prev_hash
    )
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LedgerRecord:
    """One persisted ledger row: the eight §12 columns, read back verbatim.

    Attributes:
        sequence_number: The record's 1-based position in the chain.
        event_type: The record's event type discriminator.
        created_at: The record's ISO-8601 creation timestamp.
        component: The producing component projected from the envelope.
        payload_json: The canonical envelope JSON stored for the record.
        payload_schema_version: Payload schema version projected from the
            envelope.
        prev_hash: The predecessor's ``event_hash``.
        event_hash: This record's chained hash.
    """

    sequence_number: int
    event_type: str
    created_at: str
    component: str
    payload_json: str
    payload_schema_version: int
    prev_hash: str
    event_hash: str


def events_from_records(records: Iterable[LedgerRecord]) -> tuple[Event, ...]:
    """Reconstruct base :class:`~windbreak.ledger.events.Event` objects from rows.

    The read-side companion of :meth:`SqliteLedgerStore.append`: each persisted
    :class:`LedgerRecord` carries a canonical envelope JSON of
    ``{"component", "data", "schema_version"}``, from which a base ``Event`` is
    rebuilt so a restarting :class:`~windbreak.riskkernel.process.RiskKernel` or
    :class:`~windbreak.riskkernel.kill.KillSwitch` can fold real ledger history
    (issue #235).

    The result is a materialized tuple, never a lazy generator: a folding caller
    (e.g. ``RiskKernel.from_events``) walks the history more than once, and an
    exhausted single-pass iterator would silently fold to nothing on the second
    walk -- failing open on a safety-critical replay. Reconstruction is
    fail-closed: a malformed envelope (not a mapping, or missing any of
    ``component`` / ``data`` / ``schema_version``) raises ``ValueError`` rather
    than fabricating a wrong event, chaining the underlying decode/lookup error
    as its cause.

    Args:
        records: The persisted records to reconstruct events from, in order.

    Returns:
        The reconstructed events as a tuple, one per input record, in order.

    Raises:
        ValueError: If any record's envelope is malformed or missing a required
            key (fail-closed).
    """
    return tuple(_event_from_record(record) for record in records)


def _event_from_record(record: LedgerRecord) -> Event:
    """Reconstruct one base ``Event`` from a persisted record's envelope.

    Args:
        record: The record whose ``payload_json`` envelope is rebuilt into an
            ``Event``.

    Returns:
        The reconstructed :class:`~windbreak.ledger.events.Event`.

    Raises:
        ValueError: If the envelope is malformed (not a mapping) or missing any
            of ``component`` / ``data`` / ``schema_version`` (fail-closed),
            chaining the underlying error as its cause.
    """
    try:
        envelope: dict[str, object] = json.loads(record.payload_json)
        component = envelope["component"]
        payload = envelope["data"]
        schema_version = envelope["schema_version"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        msg = f"malformed ledger envelope at sequence_number={record.sequence_number}"
        raise ValueError(msg) from exc
    return Event(
        event_type=record.event_type,
        component=cast("str", component),
        payload_schema_version=cast("int", schema_version),
        payload=cast("dict[str, object]", payload),
    )


@dataclass(frozen=True)
class ChainHead:
    """The current head (last row) of the ledger's hash chain.

    Attributes:
        sequence_number: The head row's 1-based position in the chain.
        event_hash: The head row's chained SHA-256 ``event_hash``.
    """

    sequence_number: int
    event_hash: str


class ChainIntegrityError(Exception):
    """Raised when the ledger's hash chain fails verification.

    Attributes:
        sequence_number: The expected sequence position at which the first
            violation was detected.
    """

    def __init__(self, sequence_number: int) -> None:
        """Initialize the error with the offending sequence position.

        Args:
            sequence_number: The expected sequence position of the first
                detected violation.
        """
        self.sequence_number = sequence_number
        super().__init__(
            f"ledger chain integrity violation at sequence_number={sequence_number}"
        )


class LedgerStore(Protocol):
    """Structural interface for an append-only, hash-chained ledger."""

    def append(self, event: Event) -> int:
        """Append an event and return its assigned sequence number."""

    def read_all(self) -> list[LedgerRecord]:
        """Return every persisted record in ascending sequence order."""

    def verify_chain(self) -> None:
        """Verify the hash chain, raising ``ChainIntegrityError`` on tamper."""

    def close(self) -> None:
        """Release the underlying storage resources."""


@runtime_checkable
class LatestRecordLookup(Protocol):
    """Optional capability: an indexed reverse read over event types (issue #246).

    Kept deliberately narrow and *separate* from :class:`LedgerStore`. Consumers
    (currently
    :func:`windbreak.evaluation.preregistration.latest_gate_plan_registration`)
    ``isinstance``-dispatch onto this protocol when a store provides it and fall
    back to a full :meth:`LedgerStore.read_all` scan when it does not, so a store
    that never grew the method -- every hand-rolled double in the test suite --
    keeps working unchanged.

    ``runtime_checkable`` makes the ``isinstance`` dispatch legal; note it
    verifies only that the attribute exists, never its signature, which is the
    intended duck-type semantics here.
    """

    def latest_record_of_types(
        self, event_types: Collection[str]
    ) -> LedgerRecord | None:
        """Return the highest-sequence record whose type is in ``event_types``."""


class SqliteLedgerStore:
    """A :class:`LedgerStore` persisted to a WAL-journaled SQLite database.

    The connection runs in autocommit mode so each append can wrap its
    insert in an explicit ``BEGIN IMMEDIATE`` transaction, and the ledger
    table is created on first use if absent.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        now: Callable[[], datetime] = _default_clock,
    ) -> None:
        """Open (or create) the ledger database at ``db_path``.

        Args:
            db_path: Filesystem path to the SQLite database file.
            now: Clock returning the timezone-aware datetime stamped as each
                record's ``created_at``. Injectable for deterministic tests.
        """
        self._now = now
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_EVENT_TYPE_INDEX_SQL)

    def append(self, event: Event) -> int:
        """Append an event as the next record in the chain.

        Args:
            event: The event to persist.

        Returns:
            The sequence number assigned to the new record.

        Raises:
            Exception: Any failure mid-append (e.g. from hashing or the INSERT)
                rolls back the ``BEGIN IMMEDIATE`` transaction — releasing the
                write lock and leaving the ledger unchanged — and re-raises the
                original exception unchanged.
        """
        created_at = self._now().isoformat(timespec="microseconds")
        payload_json = event.envelope_json
        # ``BEGIN IMMEDIATE`` stays outside the ``try`` so that a failure to
        # acquire the write lock leaves no half-open transaction to roll back.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            last = self._conn.execute(_SELECT_LAST_SQL).fetchone()
            if last is None:
                sequence_number = 1
                prev_hash = GENESIS_PREV_HASH
            else:
                sequence_number = int(last[0]) + 1
                prev_hash = str(last[1])
            event_hash = compute_event_hash(
                sequence_number, event.event_type, created_at, payload_json, prev_hash
            )
            self._conn.execute(
                _INSERT_SQL,
                (
                    sequence_number,
                    event.event_type,
                    created_at,
                    event.component,
                    payload_json,
                    event.payload_schema_version,
                    prev_hash,
                    event_hash,
                ),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return sequence_number

    def read_all(self) -> list[LedgerRecord]:
        """Return every record in ascending sequence order.

        Returns:
            The persisted records as :class:`LedgerRecord` instances.
        """
        rows = self._conn.execute(_SELECT_ALL_SQL).fetchall()
        return [LedgerRecord(*row) for row in rows]

    def latest_record_of_types(
        self, event_types: Collection[str]
    ) -> LedgerRecord | None:
        """Return the newest record whose ``event_type`` is in ``event_types``.

        Satisfies the optional :class:`LatestRecordLookup` capability. Issues one
        index-backed ``ORDER BY sequence_number DESC LIMIT 1`` read per requested
        type and keeps the highest-sequence hit, so a caller looking for "the
        latest gate-plan registration" no longer materializes the entire ledger
        on every promotion attempt (issue #246). Types are queried in sorted
        order purely so the emitted statement sequence is deterministic -- the
        answer is order-independent by construction.

        An empty ``event_types`` matches nothing and returns ``None`` without
        touching the database, rather than emitting a degenerate query.

        Args:
            event_types: The event types to match. Duplicates are collapsed.

        Returns:
            The matching :class:`LedgerRecord` with the largest
            ``sequence_number``, or ``None`` when no row matches.
        """
        latest: LedgerRecord | None = None
        for event_type in sorted(set(event_types)):
            row = self._conn.execute(
                _SELECT_LATEST_OF_TYPE_SQL, (event_type,)
            ).fetchone()
            if row is None:
                continue
            record = LedgerRecord(*row)
            if latest is None or record.sequence_number > latest.sequence_number:
                latest = record
        return latest

    def verify_chain(self) -> None:
        """Verify sequence contiguity and hash linkage across the chain.

        Raises:
            ChainIntegrityError: On the first row whose sequence number,
                recomputed hash, predecessor link, or envelope projection
                does not match, reporting that row's expected position.
        """
        expected_prev_hash = GENESIS_PREV_HASH
        expected_seq = 1
        for record in self.read_all():
            self._verify_row(record, expected_seq, expected_prev_hash)
            expected_prev_hash = record.event_hash
            expected_seq += 1

    def head(self) -> ChainHead | None:
        """Return the current chain head, or ``None`` for an empty ledger.

        Reads the highest-sequence row via the same ``_SELECT_LAST_SQL`` the
        append path uses to find its predecessor, so the head reported here is
        exactly the row the next append would chain onto.

        Returns:
            A :class:`ChainHead` pinned to the last row's ``sequence_number``
            and ``event_hash``, or ``None`` when no rows have been appended.
        """
        row = self._conn.execute(_SELECT_LAST_SQL).fetchone()
        if row is None:
            return None
        return ChainHead(sequence_number=int(row[0]), event_hash=str(row[1]))

    def _verify_row(
        self, record: LedgerRecord, expected_seq: int, expected_prev_hash: str
    ) -> None:
        """Verify one record against its expected position and predecessor.

        Args:
            record: The record to verify.
            expected_seq: The sequence number this position must hold.
            expected_prev_hash: The predecessor's ``event_hash``.

        Raises:
            ChainIntegrityError: If sequence, hash, or link checks fail.
        """
        if record.sequence_number != expected_seq:
            raise ChainIntegrityError(expected_seq)
        recomputed = compute_event_hash(
            record.sequence_number,
            record.event_type,
            record.created_at,
            record.payload_json,
            record.prev_hash,
        )
        if recomputed != record.event_hash:
            raise ChainIntegrityError(expected_seq)
        if record.prev_hash != expected_prev_hash:
            raise ChainIntegrityError(expected_seq)
        self._verify_envelope(record, expected_seq)

    def _verify_envelope(self, record: LedgerRecord, expected_seq: int) -> None:
        """Verify a record's column projections match its stored envelope.

        Args:
            record: The record whose ``component`` and schema version are
                checked against its ``payload_json`` envelope.
            expected_seq: The sequence position reported on mismatch.

        Raises:
            ChainIntegrityError: If a projected column disagrees with the
                envelope, or the envelope is malformed or missing a required
                key.
        """
        try:
            envelope: dict[str, object] = json.loads(record.payload_json)
            projections_match = (
                record.component == envelope["component"]
                and record.payload_schema_version == envelope["schema_version"]
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ChainIntegrityError(expected_seq) from exc
        if not projections_match:
            raise ChainIntegrityError(expected_seq)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
