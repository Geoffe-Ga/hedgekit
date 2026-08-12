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
may additionally declare *optional*, narrow capability protocols that answer a
targeted question without replaying the whole log:

* :class:`LatestRecordLookup` -- "the newest record of these event types",
  as one indexed reverse read (issue #246).
* :class:`ReverseTypeScan` -- a lazy newest-first walk over one event type, for
  the questions that need more than the newest row but far less than the whole
  log, and whose consumer can stop early (issue #370).

Each is deliberately a separate protocol rather than a few more lines on
:class:`LedgerStore`, because :class:`LedgerStore` is satisfied structurally by
hand-rolled doubles that must keep working untouched; a consumer duck-type
dispatches onto the capability when present and falls back to a full scan when
it is not.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from windbreak.ledger.events import GENESIS_PREV_HASH, Event

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator
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

#: Every row of ONE event type, newest first. Same statement as
#: ``_SELECT_LATEST_OF_TYPE_SQL`` without the ``LIMIT``, and backed by the same
#: composite index, so SQLite seeks straight to that type's last entry and walks
#: it backwards. Deliberately unbounded in SQL and bounded by the *consumer*
#: instead: how far back is far enough is a question only the caller's predicate
#: can answer (see :class:`ReverseTypeScan`), and the rows are streamed one at a
#: time so stopping early genuinely stops the read.
_SELECT_OF_TYPE_REVERSED_SQL = (
    "SELECT sequence_number, event_type, created_at, component, "
    "payload_json, payload_schema_version, prev_hash, event_hash "
    "FROM ledger WHERE event_type = ? ORDER BY sequence_number DESC"
)


_LOGGER = logging.getLogger(__name__)

#: How long a process opening the ledger waits for a sibling's SQLite write
#: lock before failing closed, in milliseconds (issue #328).
#:
#: WHAT IT BOUNDS, EXACTLY. Each *contended statement*, not the open as a whole.
#: SQLite's ``busy_timeout`` is a per-statement budget, and the retry deadline
#: in :func:`_enter_wal_mode` bounds that one statement's retrying. The open
#: issues three statements that can contend, so a holder that keeps re-taking
#: the lock between them can stretch the whole open toward three times this
#: value before it fails. That is still bounded -- the point of the budget is
#: that no path can hang -- but "the open gives up after ten seconds" would be
#: the optimistic reading, and the honest one is "no single contended statement
#: waits longer than ten seconds".
#:
#: WHAT IT MUST SURVIVE. Opening the ledger is a *write*, and
#: ``deploy/docker-compose.yml`` starts ``pipeline``, ``riskkernel`` and
#: ``order-gateway`` simultaneously against one shared volume, so the shipped
#: startup path is several processes racing for the same lock. Every write
#: windbreak itself holds is one row inside its own ``BEGIN IMMEDIATE`` ...
#: ``COMMIT`` -- there is no long-running write transaction anywhere in this
#: package -- and a four-way contended open measured while fixing #328 cleared
#: in at most two attempts and a few milliseconds. Ten seconds is three orders
#: of magnitude of headroom over that, which is the point: the budget must not
#: be a number the shipped deployment can plausibly reach.
#:
#: WHAT IT MUST NOT BECOME. Under ``restart: on-failure`` a hang is worse than
#: a crash, because a container that never exits never gets its cause into the
#: log. So the budget also has to be short enough that a genuinely wedged
#: holder -- a stopped process, an abandoned transaction, a stale lock on a
#: network filesystem -- turns into a *readable* failure quickly. Ten seconds
#: per contended statement keeps even the three-statement worst case inside the
#: half-minute an operator spends deciding whether a service is slow or stuck,
#: so the ``FATAL`` line arrives while they are still looking.
#:
#: WHEN IT IS EXCEEDED, the open raises :class:`LedgerLockedError` after
#: logging it as ``FATAL`` at ``CRITICAL``. The capability fails closed; the
#: cause is never silent.
LEDGER_BUSY_TIMEOUT_MILLIS = 10_000

#: Nanoseconds per millisecond. The retry deadline is computed in integer
#: nanoseconds from :func:`time.monotonic_ns`, because this package is on the
#: no-float path (SPEC S6.1/S17.3) and a timeout in milliseconds is an integer.
_NANOS_PER_MILLI = 1_000_000

#: The ``timeout`` handed to :func:`sqlite3.connect`. Zero on purpose: the
#: standard library otherwise supplies 5.0 seconds of busy timeout that nothing
#: in windbreak chose, named, or could see. Suppressing it leaves exactly one
#: source for the value actually in force -- the ``PRAGMA busy_timeout`` set
#: immediately afterwards.
_NO_IMPLICIT_BUSY_TIMEOUT = 0

#: Put the connection's database into write-ahead logging.
_JOURNAL_MODE_WAL_SQL = "PRAGMA journal_mode=WAL"

#: Take and immediately release SQLite's write lock. Used only as a *wait*:
#: unlike ``PRAGMA journal_mode``, ``BEGIN IMMEDIATE`` is routed through the
#: busy handler, so blocking on it costs no CPU and honours ``busy_timeout``.
_TAKE_WRITE_LOCK_SQL = "BEGIN IMMEDIATE"
_RELEASE_WRITE_LOCK_SQL = "ROLLBACK"

#: The SQLite result codes that mean "another connection holds a lock I need",
#: and nothing else. Every other ``sqlite3.OperationalError`` -- an unopenable
#: path, a read-only mount, a full disk -- reaches the caller with its own
#: message, because a confident wrong diagnosis is worse than a bare one.
_LOCK_CONTENTION_ERRORCODES = (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


class LedgerLockedError(Exception):
    """Raised when opening the ledger lost the race for its write lock (#328).

    The failure this replaces was an unhandled ``sqlite3.OperationalError``
    raised out of a ``PRAGMA`` inside the constructor: exit 1, a traceback
    naming a pragma, and no ``FATAL`` line -- so an operator watching a
    ``restart: on-failure`` service saw a crash loop with no cause. This error
    is logged as ``FATAL`` at ``CRITICAL`` before it is raised, and its message
    names the ledger, the budget that was exhausted, and SQLite's own wording.
    """

    def __init__(self, db_path: Path, busy_timeout_millis: int, cause: str) -> None:
        """Initialize the error with the ledger, the budget, and SQLite's cause.

        Args:
            db_path: The ledger that could not be opened.
            busy_timeout_millis: The budget that was exhausted waiting for it.
            cause: SQLite's own description of the refusal.
        """
        super().__init__(
            f"cannot open the ledger at {db_path}: another process still held "
            f"the SQLite write lock after {busy_timeout_millis} ms ({cause}). "
            "Stop whichever windbreak process is using the same --ledger-path, "
            "or let it finish, then start this one again."
        )


def _is_lock_contention(exc: sqlite3.OperationalError) -> bool:
    """Report whether ``exc`` is SQLite refusing on a lock someone else holds.

    Args:
        exc: The error SQLite raised.

    Returns:
        ``True`` when its result code is ``SQLITE_BUSY`` or ``SQLITE_LOCKED``.
    """
    return exc.sqlite_errorcode in _LOCK_CONTENTION_ERRORCODES


def _wait_out_the_write_lock(conn: sqlite3.Connection) -> None:
    """Block until the write lock is free, then release it again.

    The retry in :func:`_enter_wal_mode` must not spin: a hot loop against a
    genuinely wedged holder would burn a core for the whole budget. Taking and
    releasing the lock is the wait, because ``BEGIN IMMEDIATE`` *is* routed
    through SQLite's busy handler and so blocks at ``busy_timeout`` inside
    SQLite itself.

    Its own failure is not a second failure channel: when the lock is still
    held at the end of the busy timeout this returns, and the caller re-checks
    its deadline and either retries or fails loudly. Nothing is swallowed that
    the caller does not go on to decide about.

    Args:
        conn: The connection to wait on.
    """
    # Suppression hides nothing: the caller re-checks its deadline immediately
    # after this returns and either retries or raises loudly, so a lock still
    # held here becomes a reported failure rather than a silent one.
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute(_TAKE_WRITE_LOCK_SQL)
        conn.execute(_RELEASE_WRITE_LOCK_SQL)


def _enter_wal_mode(conn: sqlite3.Connection, *, deadline_ns: int) -> None:
    """Put the connection's database into WAL journaling, waiting if contended.

    This is the one statement in the open sequence that ``busy_timeout`` cannot
    help. ``PRAGMA journal_mode=WAL`` acquires the database's exclusive lock and
    SQLite does **not** route that acquisition through the busy handler, so with
    any other connection holding the write lock it is refused at once, at any
    busy timeout whatsoever. Measured while fixing issue #328: four simultaneous
    openers of one brand-new database still lost 28 of 160 openers to ``database
    is locked`` at a 10 000 ms busy timeout, and 123 of 160 at zero. So the wait
    for this statement is windbreak's to run, and it is bounded by the same
    budget everything else in the open uses.

    Args:
        conn: The connection to convert.
        deadline_ns: The :func:`time.monotonic_ns` reading past which the wait
            is over and the refusal is final.

    Raises:
        sqlite3.OperationalError: If the conversion is refused for a reason that
            is not lock contention, or is still refused at the deadline. The
            caller translates a contended refusal into
            :class:`LedgerLockedError`.
    """
    while True:
        try:
            conn.execute(_JOURNAL_MODE_WAL_SQL)
        except sqlite3.OperationalError as exc:
            if not _is_lock_contention(exc) or time.monotonic_ns() >= deadline_ns:
                raise
            _wait_out_the_write_lock(conn)
        else:
            return


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


@runtime_checkable
class ReverseTypeScan(Protocol):
    """Optional capability: a lazy newest-first walk over one type (issue #370).

    The sibling of :class:`LatestRecordLookup`, for the questions that need more
    than the single newest row but far less than the whole log. Its first
    consumer,
    :func:`windbreak.scheduler.loop.read_start_of_day_equity_micros`, wants the
    *earliest* ``EquitySampled`` row of the current UTC day -- which the
    latest-only lookup cannot answer, and which the PAPER loop asks on every
    tick of a deployment meant to run for weeks.

    Walking newest-first is what makes that bounded: the consumer holds a
    recency predicate, so it can stop the moment it crosses the day boundary and
    pay O(rows since that boundary) instead of O(ledger). The walk must
    therefore be **lazy** -- yielding one record at a time rather than
    materializing the type's rows -- or the consumer's early stop would save
    only the decoding, not the read.

    Kept narrow and separate from :class:`LedgerStore` for the same reason
    :class:`LatestRecordLookup` is: consumers ``isinstance``-dispatch onto it
    when a store provides it and fall back to a full
    :meth:`LedgerStore.read_all` fold when it does not, so every hand-rolled
    double keeps working unchanged.
    """

    def iter_records_of_type_reversed(self, event_type: str) -> Iterator[LedgerRecord]:
        """Yield records of ``event_type`` newest first, one at a time."""


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
        busy_timeout_millis: int = LEDGER_BUSY_TIMEOUT_MILLIS,
    ) -> None:
        """Open (or create) the ledger database at ``db_path``.

        Opening is a *write*: the journal mode, the table and the index are all
        set here, so a process opening a ledger a sibling is writing has to wait
        for the write lock rather than die on it (issue #328). Each of those
        three statements waits up to ``busy_timeout_millis`` and then fails
        closed, loudly -- the budget is per contended statement, not per open.

        Args:
            db_path: Filesystem path to the SQLite database file.
            now: Clock returning the timezone-aware datetime stamped as each
                record's ``created_at``. Injectable for deterministic tests.
            busy_timeout_millis: How long to wait for a sibling process's write
                lock before giving up, in milliseconds. Defaults to
                :data:`LEDGER_BUSY_TIMEOUT_MILLIS`, which is the value the
                shipped deployment runs on.

        Raises:
            ValueError: If ``busy_timeout_millis`` is negative. SQLite reads a
                negative busy timeout as "never time out", and a ledger open
                that can hang for ever is the failure this parameter exists to
                stop.
            LedgerLockedError: If the write lock was still held by another
                process when the budget ran out.
            sqlite3.OperationalError: If the database could not be opened for
                any reason that is *not* lock contention -- an unopenable path,
                a read-only mount, a full disk -- reported unchanged.
        """
        if busy_timeout_millis < 0:
            raise ValueError(
                f"busy_timeout_millis must not be negative, got "
                f"{busy_timeout_millis}: a negative SQLite busy timeout means "
                "waiting for ever, which is not a way to open a ledger"
            )
        self._now = now
        self._conn = sqlite3.connect(
            db_path, isolation_level=None, timeout=_NO_IMPLICIT_BUSY_TIMEOUT
        )
        try:
            self._open_schema(busy_timeout_millis)
        except sqlite3.OperationalError as exc:
            self._conn.close()
            if not _is_lock_contention(exc):
                raise
            error = LedgerLockedError(db_path, busy_timeout_millis, str(exc))
            _LOGGER.critical("FATAL: %s", error)
            raise error from exc

    def _open_schema(self, busy_timeout_millis: int) -> None:
        """Set the busy timeout, enter WAL journaling, and ensure the schema.

        The order is load-bearing. ``PRAGMA busy_timeout`` comes first because
        every statement after it can contend, and the two ``CREATE ... IF NOT
        EXISTS`` statements are routed through SQLite's busy handler and so wait
        on it. ``PRAGMA journal_mode=WAL`` is not, which is why it gets its own
        bounded retry rather than trusting the timeout it ignores.

        Args:
            busy_timeout_millis: The whole open's budget, in milliseconds.
        """
        self._conn.execute(f"PRAGMA busy_timeout={busy_timeout_millis}")
        deadline_ns = time.monotonic_ns() + busy_timeout_millis * _NANOS_PER_MILLI
        _enter_wal_mode(self._conn, deadline_ns=deadline_ns)
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_EVENT_TYPE_INDEX_SQL)

    def append(self, event: Event) -> int:
        """Append an event as the next record in the chain.

        Args:
            event: The event to persist.

        Returns:
            The sequence number assigned to the new record.

        Raises:
            Exception: Any failure mid-append (e.g. from hashing, the INSERT, or
                the COMMIT) rolls back the ``BEGIN IMMEDIATE`` transaction —
                releasing the write lock and leaving the ledger unchanged — and
                re-raises the **original** exception unchanged. The rollback is
                best-effort and can never displace that original: when SQLite has
                already discarded the transaction itself (as it does on
                ``SQLITE_FULL``), the resulting ``cannot rollback - no
                transaction is active`` is suppressed so the caller still reads
                the real condition, e.g. ``database or disk is full``
                (issue #448).
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
            # The rollback is cleanup, never a second failure channel. SQLite
            # discards the transaction itself on conditions like ``SQLITE_FULL``,
            # so this ``ROLLBACK`` can raise ``cannot rollback - no transaction
            # is active`` -- and an exception raised inside an ``except`` block
            # REPLACES the one being handled, leaving the operator reading a
            # transaction-management error where "database or disk is full"
            # belongs (issue #448). Suppression is scoped to the rollback alone,
            # so the original always propagates.
            with contextlib.suppress(sqlite3.OperationalError):
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

    def iter_records_of_type_reversed(self, event_type: str) -> Iterator[LedgerRecord]:
        """Walk every record of ``event_type`` newest first, streaming one row.

        Satisfies the optional :class:`ReverseTypeScan` capability. The read runs
        against the same ``(event_type, sequence_number)`` index
        :meth:`latest_record_of_types` uses, so rows of other types are never
        visited at all, and rows are pulled from the cursor one at a time rather
        than fetched in bulk -- a consumer that stops after the first few (the
        PAPER loop's daily-loss baseline stops at the UTC day boundary, issue
        #370) really does leave the rest unread.

        Abandoning the walk part-way is the expected case, not an error: the
        cursor is released with the generator, and WAL journaling lets a
        subsequent append proceed regardless.

        Args:
            event_type: The single event type to walk. One type, not a
                collection, so the statement stays fully literal and the
                package's SQL statically auditable.

        Yields:
            The matching :class:`LedgerRecord` instances in descending
            ``sequence_number`` order.
        """
        for row in self._conn.execute(_SELECT_OF_TYPE_REVERSED_SQL, (event_type,)):
            yield LedgerRecord(*row)

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
