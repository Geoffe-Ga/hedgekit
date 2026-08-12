"""Opening a ledger a sibling process is writing (issue #328).

``SqliteLedgerStore.__init__`` is not a read. It issues ``PRAGMA
journal_mode=WAL``, ``CREATE TABLE IF NOT EXISTS`` and ``CREATE INDEX IF NOT
EXISTS``, all of which take SQLite's write lock, and
``deploy/docker-compose.yml`` starts ``pipeline``, ``riskkernel`` and
``order-gateway`` simultaneously against one shared volume under ``restart:
on-failure``. So "a process opened the ledger while a sibling held the write
lock" is the shipped startup path, not a rare race.

Two *different* contended statements, which this module keeps apart because
SQLite treats them differently and only one of them is fixed by a timeout:

* ``CREATE TABLE IF NOT EXISTS`` is routed through SQLite's busy handler, so
  ``PRAGMA busy_timeout`` really does make it wait. This is the statement PR
  #511's reproduction hit.
* ``PRAGMA journal_mode=WAL`` takes the database's exclusive lock and SQLite
  does **not** consult the busy handler for it. It raises immediately, at any
  busy timeout whatsoever -- measured while writing this module: four
  simultaneous openers of one brand-new database, released from a barrier,
  still lost 28 of 160 openers to ``database is locked`` at a 10 000 ms busy
  timeout, and 123 of 160 at zero. That is the statement the traceback on issue
  #328 names, and no value of ``busy_timeout`` alone would have repaired it.

So each mechanism gets its own contended-statement test in both directions: it
waits and then succeeds when the lock frees inside the budget, and it fails
closed -- loudly, with the cause readable -- when it does not.

Determinism
-----------

Nothing here sleeps and nothing asserts on elapsed wall-clock. Every wait is
bounded by :data:`JOIN_TIMEOUT_SECONDS` and every assertion is made on the
*outcome*.

The rendezvous is causal rather than hopeful, which matters: a test that simply
signalled "about to open" and freed the lock could lose the race to a store that
never waited, and would then pass against the unfixed store. So the holder is
freed only once SQLite's own statement trace reports the opening thread
*beginning* the contended statement -- the trace fires as a statement starts,
before it blocks. In the WAL direction the awaited statement is the retry's own
``BEGIN IMMEDIATE`` wait, which exists only because the conversion was refused,
so reaching it *is* the retry having run.

The negative direction needs no rendezvous at all: the budget is zero and the
lock is never freed, so the refusal is immediate by construction.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import TYPE_CHECKING, Any

import pytest

from windbreak.ledger.events import ConfigLoaded
from windbreak.ledger.store import (
    _CREATE_EVENT_TYPE_INDEX_SQL as CREATE_INDEX_SQL,
)
from windbreak.ledger.store import (
    _CREATE_TABLE_SQL as CREATE_TABLE_SQL,
)
from windbreak.ledger.store import (
    _JOURNAL_MODE_WAL_SQL as JOURNAL_MODE_WAL_SQL,
)
from windbreak.ledger.store import (
    _TAKE_WRITE_LOCK_SQL as TAKE_WRITE_LOCK_SQL,
)
from windbreak.ledger.store import (
    LEDGER_BUSY_TIMEOUT_MILLIS,
    LedgerLockedError,
    SqliteLedgerStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: Bound on every wait in this module. Generous on purpose: it is a deadlock
#: guard, never a measurement, so no assertion depends on how long it takes.
JOIN_TIMEOUT_SECONDS = 60.0

#: The busy timeout the standard library silently supplies when
#: ``sqlite3.connect`` is given no ``timeout``: 5.0 seconds, in milliseconds.
#: Pinned here because it is the value that used to govern this store by
#: accident -- nothing in windbreak chose it, nothing named it, and it is not
#: what the store must be governed by.
STDLIB_DEFAULT_BUSY_TIMEOUT_MILLIS = 5_000

#: The refusal budget for the negative direction: no wait at all.
NO_WAIT_MILLIS = 0

#: The component stamped on the row every positive test appends, to show the
#: store that waited is a working store and not merely a constructed object.
COMPONENT = "pipeline"


def _held_write_lock(db_path: Path, *, in_wal: bool) -> sqlite3.Connection:
    """Open a second connection and hold the ledger's write lock on it.

    An ordinary second connection is exactly what a sibling process is, so
    nothing about this reproduction is special to a test.

    Args:
        db_path: The ledger file to lock.
        in_wal: Whether to put the database into WAL journaling first. ``False``
            leaves it in SQLite's default rollback-journal mode, which is what
            makes the opener's ``PRAGMA journal_mode=WAL`` the contended
            statement rather than its ``CREATE TABLE``.

    Returns:
        The connection holding the write lock. The caller frees it.
    """
    holder = sqlite3.connect(db_path, isolation_level=None)
    if in_wal:
        holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("BEGIN IMMEDIATE")
    return holder


def _trace_every_new_connection(
    monkeypatch: pytest.MonkeyPatch, trace: Callable[[str], None]
) -> None:
    """Install ``trace`` on every SQLite connection opened from here on.

    The store makes its own connection, so this is the only way to watch the
    statements it really executes. Applied *after* a test's lock holder is
    already open, so the holder is never traced.

    Args:
        monkeypatch: pytest's attribute patcher, scoped to one test.
        trace: Called with each SQL statement as SQLite begins running it --
            crucially *before* the statement blocks, which is what makes it
            usable as a rendezvous.
    """
    real_connect = sqlite3.connect

    def tracing_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        """Open a connection and attach the trace callback to it."""
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracing_connect)


def _assert_opens_and_works_while_locked(
    db_path: Path,
    holder: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    *,
    blocks_on: str,
) -> None:
    """Open and exercise the contended ledger, freeing ``holder`` once it blocks.

    The rendezvous is causal, not a guess. SQLite's statement trace fires as a
    statement *begins* running, so waiting for ``blocks_on`` proves the opening
    thread has reached the contended statement and is about to sit on the lock
    this thread still holds. Only then is the lock freed. Without that, a
    release could win the race to a store that never waited at all, and the test
    would pass on an unfixed store.

    The store's whole lifetime stays on the opening thread: a SQLite connection
    may only be used from the thread that made it, so a store handed back across
    the boundary could not be appended to, verified, or even closed.

    A constructor that merely returned would prove nothing about the connection
    behind it, so the opening thread appends a row, reads it back, and verifies
    the hash chain before closing. The verification matters twice over here: the
    chain is what makes the ledger evidence, and this open raced a writer.

    Args:
        db_path: The contended ledger.
        holder: The connection holding the write lock, freed and closed here.
        monkeypatch: pytest's attribute patcher, used to trace the store's own
            connection.
        blocks_on: The exact statement the opener must reach before the lock is
            freed -- the statement this direction is about.
    """
    outcome: list[str] = []
    failures: list[Exception] = []
    reached = threading.Event()

    def note(statement: str) -> None:
        """Signal once the opener reaches the statement under test."""
        if statement == blocks_on:
            reached.set()

    _trace_every_new_connection(monkeypatch, note)

    def open_and_use_the_ledger() -> None:
        """Open the contended ledger, append to it, and verify its chain."""
        try:
            store = SqliteLedgerStore(db_path)
        except Exception as exc:
            failures.append(exc)
            reached.set()
            return
        try:
            store.append(ConfigLoaded(component=COMPONENT, config_hash="a", diff={}))
            records = store.read_all()
            store.verify_chain()
            outcome.extend(
                f"{record.sequence_number}:{record.component}" for record in records
            )
        finally:
            store.close()

    opener = threading.Thread(target=open_and_use_the_ledger, name="ledger-opener")
    opener.start()
    try:
        assert reached.wait(timeout=JOIN_TIMEOUT_SECONDS)
        # It cannot have finished: this thread still holds the write lock the
        # statement it just reached needs. Asserted *before* the release, so a
        # store that died on contention is caught here rather than below.
        assert opener.is_alive()
        holder.rollback()
    finally:
        holder.close()
        opener.join(timeout=JOIN_TIMEOUT_SECONDS)

    assert not opener.is_alive()
    assert failures == []
    assert outcome == [f"1:{COMPONENT}"]


def test_the_open_busy_timeout_is_windbreaks_own_value_not_the_standard_librarys(
    tmp_path: Path,
) -> None:
    """The store's connection carries the module's busy timeout, chosen here.

    ``sqlite3.connect`` supplies 5 000 ms when given no ``timeout``, so before
    issue #328 this store waited five seconds and then died on a value no part
    of windbreak had chosen or written down. The store now passes ``timeout=0``
    and sets its own, so the busy timeout in force has exactly one source: a
    raw connection to the same file reports the standard library's value, the
    store's reports windbreak's.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    db_path = tmp_path / "ledger.db"
    raw = sqlite3.connect(db_path)
    try:
        assert raw.execute("PRAGMA busy_timeout").fetchone() == (
            STDLIB_DEFAULT_BUSY_TIMEOUT_MILLIS,
        )
    finally:
        raw.close()

    store = SqliteLedgerStore(db_path)
    try:
        assert store._conn.execute("PRAGMA busy_timeout").fetchone() == (
            LEDGER_BUSY_TIMEOUT_MILLIS,
        )
        assert LEDGER_BUSY_TIMEOUT_MILLIS != STDLIB_DEFAULT_BUSY_TIMEOUT_MILLIS
    finally:
        store.close()


def test_the_open_emits_exactly_these_statements_in_exactly_this_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order of the open sequence is the fix, so it is asserted as a whole.

    ``PRAGMA busy_timeout`` has to come **first**: every statement after it can
    contend, and the two ``CREATE ... IF NOT EXISTS`` statements wait on it. Set
    later it would still be readable off the finished connection while having
    protected nothing, which is the shape of a fix that is not one.

    The sequence is compared for equality, from the real ``__init__``, against
    the production statement constants rather than a restatement of them -- so a
    reordering, an extra statement, or a missing one all fail here.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: pytest's attribute patcher, used to trace the statements.
    """
    statements: list[str] = []
    _trace_every_new_connection(monkeypatch, statements.append)

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    store.close()

    assert statements == [
        f"PRAGMA busy_timeout={LEDGER_BUSY_TIMEOUT_MILLIS}",
        JOURNAL_MODE_WAL_SQL,
        CREATE_TABLE_SQL,
        CREATE_INDEX_SQL,
    ]


def test_the_open_budget_is_the_value_this_deployment_was_sized_for(
    tmp_path: Path,
) -> None:
    """Ten seconds, pinned: changing it must be a decision, not a drift.

    Chosen against two bounds, both of which a silent edit would break. It must
    be far longer than anything windbreak itself holds the write lock for --
    every write in this package is one row inside its own transaction, and a
    four-way contended open measured while fixing #328 cleared in at most two
    attempts and a few milliseconds -- or the shipped simultaneous start fails.
    And it must be far shorter than an operator's patience with a service that
    never comes up, because under a restart policy a hang produces no log line
    at all while a bounded refusal produces one every cycle.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    assert LEDGER_BUSY_TIMEOUT_MILLIS == 10_000

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        assert store._conn.execute("PRAGMA busy_timeout").fetchone() == (10_000,)
    finally:
        store.close()


def test_an_opener_waits_out_a_sibling_holding_the_lock_over_the_schema_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A WAL ledger whose table is missing: the opener waits, then opens.

    This is PR #511's reproduction exactly -- the holder puts the file into WAL
    and takes ``BEGIN IMMEDIATE``, so the opener's contended statement is
    ``CREATE TABLE IF NOT EXISTS``, the statement its traceback died on. It is
    the direction ``busy_timeout`` is responsible for.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: pytest's attribute patcher, used for the rendezvous.
    """
    db_path = tmp_path / "ledger.db"
    holder = _held_write_lock(db_path, in_wal=True)
    _assert_opens_and_works_while_locked(
        db_path, holder, monkeypatch, blocks_on=CREATE_TABLE_SQL
    )


def test_an_opener_waits_out_a_sibling_holding_the_lock_over_the_wal_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rollback-journal ledger under a held lock: the opener waits, then opens.

    The other contended statement, and the one no busy timeout can help:
    ``PRAGMA journal_mode=WAL`` is refused immediately while another connection
    holds the write lock, because SQLite does not route the exclusive-lock
    acquisition through the busy handler. Only the store's own bounded retry
    makes this succeed, so this test is what falsifies "a busy timeout was
    enough".

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: pytest's attribute patcher, used for the rendezvous.
    """
    db_path = tmp_path / "ledger.db"
    holder = _held_write_lock(db_path, in_wal=False)
    # The rendezvous is the retry's own wait statement, which exists only
    # because the conversion was refused: reaching it *is* the retry running.
    _assert_opens_and_works_while_locked(
        db_path, holder, monkeypatch, blocks_on=TAKE_WRITE_LOCK_SQL
    )


def test_a_lock_that_never_frees_fails_closed_and_loudly_on_the_schema_create(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The negative direction for the schema create: no wait, no ledger, loud.

    With the budget at zero the wait is removed, so the same setup that
    succeeds above must now refuse -- and refuse *readably*. Before issue #328
    this path exited 1 with an unhandled ``sqlite3.OperationalError`` and no
    ``FATAL`` line anywhere in the process's output, which is the least
    diagnosable failure available under ``restart: on-failure``.

    Args:
        tmp_path: pytest's per-test temporary directory.
        caplog: pytest's log capture, holding the ``FATAL`` critical.
    """
    db_path = tmp_path / "ledger.db"
    holder = _held_write_lock(db_path, in_wal=True)
    caplog.set_level(logging.CRITICAL, logger="windbreak.ledger.store")
    try:
        with pytest.raises(LedgerLockedError) as caught:
            SqliteLedgerStore(db_path, busy_timeout_millis=NO_WAIT_MILLIS)
    finally:
        holder.rollback()
        holder.close()

    expected = (
        f"cannot open the ledger at {db_path}: another process still held the "
        "SQLite write lock after 0 ms (database is locked). Stop whichever "
        "windbreak process is using the same --ledger-path, or let it finish, "
        "then start this one again."
    )
    assert type(caught.value) is LedgerLockedError
    assert str(caught.value) == expected
    assert type(caught.value.__cause__) is sqlite3.OperationalError
    criticals = [
        record for record in caplog.records if record.levelno == logging.CRITICAL
    ]
    assert len(criticals) == 1
    assert criticals[0].getMessage() == f"FATAL: {expected}"


def test_a_lock_that_never_frees_fails_closed_and_loudly_on_the_wal_conversion(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The negative direction for the WAL conversion: the retry is bounded.

    The retry that makes the conversion survive contention must not become a
    process that waits for ever -- under ``restart: on-failure`` a hang is
    worse than a crash, because nothing ever reaches the log. With the budget
    spent the retry stops and the same loud refusal is raised.

    Args:
        tmp_path: pytest's per-test temporary directory.
        caplog: pytest's log capture, holding the ``FATAL`` critical.
    """
    db_path = tmp_path / "ledger.db"
    holder = _held_write_lock(db_path, in_wal=False)
    caplog.set_level(logging.CRITICAL, logger="windbreak.ledger.store")
    try:
        with pytest.raises(LedgerLockedError) as caught:
            SqliteLedgerStore(db_path, busy_timeout_millis=NO_WAIT_MILLIS)
    finally:
        holder.rollback()
        holder.close()

    expected = (
        f"cannot open the ledger at {db_path}: another process still held the "
        "SQLite write lock after 0 ms (database is locked). Stop whichever "
        "windbreak process is using the same --ledger-path, or let it finish, "
        "then start this one again."
    )
    assert type(caught.value) is LedgerLockedError
    assert str(caught.value) == expected
    criticals = [
        record for record in caplog.records if record.levelno == logging.CRITICAL
    ]
    assert len(criticals) == 1
    assert criticals[0].getMessage() == f"FATAL: {expected}"


def test_a_failure_that_is_not_contention_reaches_the_caller_unchanged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unopenable ledger is not reported as a lock another process holds.

    A guard that answers "another process held the write lock" to every
    ``sqlite3.OperationalError`` would hand the operator a confident wrong
    diagnosis for a wrong path, a read-only mount, or a full disk -- strictly
    worse than the bare error it replaced. Only ``SQLITE_BUSY``/``SQLITE_LOCKED``
    are translated; everything else propagates with its own message and logs no
    ``FATAL``.

    Args:
        tmp_path: pytest's per-test temporary directory.
        caplog: pytest's log capture, asserted empty.
    """
    not_a_database_file = tmp_path / "ledger.db"
    not_a_database_file.mkdir()
    caplog.set_level(logging.CRITICAL, logger="windbreak.ledger.store")

    with pytest.raises(sqlite3.OperationalError) as caught:
        SqliteLedgerStore(not_a_database_file)

    assert type(caught.value) is sqlite3.OperationalError
    assert str(caught.value) == "unable to open database file"
    assert caplog.records == []


def test_a_negative_busy_timeout_is_refused_before_anything_is_opened(
    tmp_path: Path,
) -> None:
    """A negative budget would mean "wait for ever", so it is refused.

    SQLite reads a negative ``busy_timeout`` as "never time out". Accepting one
    would turn the loud, bounded refusal this issue is about back into the
    silent hang it replaced, so it is rejected before the database file is
    touched at all.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    db_path = tmp_path / "ledger.db"

    with pytest.raises(ValueError) as caught:
        SqliteLedgerStore(db_path, busy_timeout_millis=-1)

    assert type(caught.value) is ValueError
    assert str(caught.value) == (
        "busy_timeout_millis must not be negative, got -1: a negative SQLite "
        "busy timeout means waiting for ever, which is not a way to open a "
        "ledger"
    )
    assert not db_path.exists()
