"""Tests for `windbreak.ledger.store.SqliteLedgerStore` (issue #13, #75).

Pins the on-disk contract from SPEC §12: monotonically increasing
sequence numbers starting at 1, hash chaining from a fixed genesis
`prev_hash`, WAL journaling, exact round-trip of every persisted column,
and durability across a store close/reopen cycle -- since the ledger
must survive process restarts.

The `compute_event_hash` known-answer test is the mutation-killer for
the hashing formula itself: it independently recomputes the SHA-256
digest in the test body using the exact §12 concatenation and asserts
byte-for-byte equality with the production function's output.

The `head()` tests (issue #75) pin the read used by anchor-head
computation: the current chain head as a `ChainHead(sequence_number,
event_hash)`, or `None` for an empty ledger.

The rollback tests (issue #76) pin `append`'s transactional contract: a
failure injected between `BEGIN IMMEDIATE` and `COMMIT` (via a monkeypatched
`compute_event_hash`) must propagate to the caller *and* leave the
transaction released, so the ledger is byte-for-byte unchanged and the next
append continues the chain normally. Today `append` has no
try/ROLLBACK, so the injected failure leaves the SQLite connection sitting
inside an un-released `BEGIN IMMEDIATE`; the very next `append` call fails
with `sqlite3.OperationalError: cannot start a transaction within a
transaction` instead of succeeding -- that is the FAILING-for-the-right-reason
symptom these tests pin.

The commit-time rollback test (issue #76 hardening) drives the harder
*post-INSERT* case: it lets the SELECT-last read, `compute_event_hash`, and
the `INSERT` all run for real, then forces `COMMIT` *itself* to fail, pinning
that `append` still ROLLBACKs -- rather than COMMITs -- the already-inserted
row. A mutant swapping `append`'s except-clause `ROLLBACK` for `COMMIT` would
persist that row and so fail this test's snapshot-equality assertion.

Issue #235 (replay durable kill state on `windbreak run --process riskkernel`
startup) adds `events_from_records`, the read-side companion this module does
not yet define: reconstructing base `Event`s from persisted `LedgerRecord`s so
a rebuilt `RiskKernel`/`KillSwitch` can fold real ledger history. It does not
exist on the real, not-yet-updated `store.py` module yet, so importing it
below fails collection with `ImportError: cannot import name
'events_from_records' from 'windbreak.ledger.store'` -- the expected Gate 1
RED state for issue #235.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from windbreak.ledger.events import (
    GENESIS_PREV_HASH,
    ConfigLoaded,
    Event,
    ModeHeartbeat,
    canonical_json,
)
from windbreak.ledger.store import (
    ChainHead,
    LatestRecordLookup,
    LedgerRecord,
    ReverseTypeScan,
    SqliteLedgerStore,
    compute_event_hash,
    events_from_records,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import Any, NoReturn


def test_append_returns_monotonically_increasing_sequence_numbers(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """The 1st, 2nd, and 3rd appended events receive sequence numbers 1, 2, 3."""
    store = ledger_store_factory()
    event = ConfigLoaded(component="pipeline", config_hash="abc", diff={})

    first = store.append(event)
    second = store.append(event)
    third = store.append(event)

    assert (first, second, third) == (1, 2, 3)


def test_first_record_prev_hash_is_genesis(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """The very first appended row links back to the genesis sentinel."""
    store = ledger_store_factory()
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))

    records = store.read_all()

    assert records[0].prev_hash == GENESIS_PREV_HASH


def test_each_record_prev_hash_links_to_predecessor_event_hash(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """Every non-genesis row's prev_hash equals its predecessor's event_hash."""
    store = ledger_store_factory()
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=2))

    records = store.read_all()

    assert records[1].prev_hash == records[0].event_hash
    assert records[2].prev_hash == records[1].event_hash


def test_compute_event_hash_matches_independent_sha256_known_answer() -> None:
    """KNOWN-ANSWER: pins the exact §12 concatenation formula byte-for-byte.

    The expected digest is computed here, independently of production
    code, from the literal concatenation
    ``str(sequence_number) + event_type + created_at + payload_json + prev_hash``.
    """
    sequence_number = 7
    event_type = "ConfigLoaded"
    created_at = "2024-01-01T00:00:00.000000+00:00"
    payload_json = (
        '{"component":"pipeline","data":{"config_hash":"abc"},"schema_version":1}'
    )
    prev_hash = "f" * 64

    expected = hashlib.sha256(
        (
            str(sequence_number) + event_type + created_at + payload_json + prev_hash
        ).encode("utf-8")
    ).hexdigest()

    actual = compute_event_hash(
        sequence_number, event_type, created_at, payload_json, prev_hash
    )

    assert actual == expected
    assert len(actual) == 64


def test_compute_event_hash_changes_when_sequence_number_changes() -> None:
    """A different sequence_number produces a different digest.

    Kills constant-fold mutants that ignore sequence_number entirely.
    """
    common_args = ("ConfigLoaded", "2024-01-01T00:00:00.000000+00:00", "{}", "f" * 64)

    assert compute_event_hash(1, *common_args) != compute_event_hash(2, *common_args)


def test_store_uses_wal_journal_mode(
    ledger_store_factory: Callable[..., SqliteLedgerStore], tmp_path: Path
) -> None:
    """The store opens its SQLite connection in WAL journal mode."""
    ledger_store_factory("wal_check.db")

    conn = sqlite3.connect(tmp_path / "wal_check.db")
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()

    assert mode.lower() == "wal"


def test_read_all_round_trips_all_eight_persisted_fields(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """read_all() reconstructs every LedgerRecord field exactly as persisted."""
    store = ledger_store_factory("roundtrip.db")
    event = ConfigLoaded(component="pipeline", config_hash="deadbeef", diff={"x": 1})

    store.append(event)
    records = store.read_all()

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, LedgerRecord)
    assert record.sequence_number == 1
    assert record.event_type == "ConfigLoaded"
    assert record.component == "pipeline"
    assert record.payload_schema_version == 1
    assert record.prev_hash == GENESIS_PREV_HASH
    assert record.event_hash == compute_event_hash(
        record.sequence_number,
        record.event_type,
        record.created_at,
        record.payload_json,
        record.prev_hash,
    )
    assert '"config_hash":"deadbeef"' in record.payload_json


def test_read_all_returns_records_in_ascending_sequence_order(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """read_all() returns rows ordered 1..N, not insertion or arbitrary order."""
    store = ledger_store_factory("ordering.db")
    for beat in range(1, 4):
        store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=beat))

    records = store.read_all()

    assert [record.sequence_number for record in records] == [1, 2, 3]


def test_data_persists_across_store_close_and_reopen(
    tmp_path: Path, deterministic_clock: Callable[[], object]
) -> None:
    """A fresh SqliteLedgerStore on the same path sees prior appends."""
    db_path = tmp_path / "reopen.db"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))
    store.close()

    reopened = SqliteLedgerStore(db_path, now=deterministic_clock)
    try:
        records = reopened.read_all()
    finally:
        reopened.close()

    assert len(records) == 1
    assert records[0].sequence_number == 1
    assert records[0].event_type == "ConfigLoaded"


def test_append_after_reopen_continues_the_same_sequence_and_hash_chain(
    tmp_path: Path, deterministic_clock: Callable[[], object]
) -> None:
    """Appending after a reopen continues sequence numbers and prev_hash linkage."""
    db_path = tmp_path / "continue.db"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))
    store.close()

    reopened = SqliteLedgerStore(db_path, now=deterministic_clock)
    try:
        second_seq = reopened.append(
            ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1)
        )
        records = reopened.read_all()
    finally:
        reopened.close()

    assert second_seq == 2
    assert records[1].prev_hash == records[0].event_hash


def test_append_without_injected_clock_uses_default_clock_for_created_at(
    tmp_path: Path,
) -> None:
    """With no `now=` override, append() stamps `created_at` via the real UTC clock.

    Every other test in this suite injects `DeterministicClock`; this is the
    one place `_default_clock` itself runs, so it must independently verify
    the timestamp it produces is a parseable, UTC, microsecond-precision
    ISO-8601 string.
    """
    db_path = tmp_path / "default_clock.db"
    store = SqliteLedgerStore(db_path)
    try:
        store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))
        records = store.read_all()
    finally:
        store.close()

    created_at = records[0].created_at
    parsed = datetime.fromisoformat(created_at)

    assert parsed.utcoffset() == timedelta(0)
    assert created_at.endswith("+00:00")
    fractional_and_offset = created_at.split(".", 1)[1]
    microseconds_digits = fractional_and_offset.split("+", 1)[0]
    assert len(microseconds_digits) == 6
    assert microseconds_digits.isdigit()


def test_head_returns_none_for_empty_ledger(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """head() on a ledger with zero appended rows returns None."""
    store = ledger_store_factory()

    assert store.head() is None


def test_head_returns_chain_head_matching_the_last_appended_row(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """head() returns a ChainHead pinned to the Nth appended row's seq/hash.

    Builds a 5-row chain and asserts head() equals `ChainHead(5,
    <the 5th row's event_hash>)` exactly -- a mutant that returns the wrong
    row (e.g. the first, or an off-by-one) is caught by dataclass equality.
    """
    store = ledger_store_factory()
    for beat in range(1, 6):
        store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=beat))

    head = store.head()
    last_record = store.read_all()[-1]

    assert head == ChainHead(sequence_number=5, event_hash=last_record.event_hash)


def test_head_matches_the_last_read_all_row_exactly(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """head()'s fields equal the last read_all() row's sequence_number/event_hash."""
    store = ledger_store_factory()
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))

    records = store.read_all()
    head = store.head()

    assert head is not None
    assert head.sequence_number == records[-1].sequence_number
    assert head.event_hash == records[-1].event_hash


def test_head_reflects_the_new_head_after_a_further_append(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """head() called again after a further append reports the new head, not the old."""
    store = ledger_store_factory()
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))
    first_head = store.head()

    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    second_head = store.head()

    assert first_head is not None
    assert first_head == ChainHead(sequence_number=1, event_hash=first_head.event_hash)
    assert second_head is not None
    assert second_head.sequence_number == 2
    assert second_head.event_hash != first_head.event_hash


def test_chain_head_is_frozen() -> None:
    """ChainHead is an immutable dataclass: assigning to a field raises."""
    head = ChainHead(sequence_number=1, event_hash="a" * 64)

    with pytest.raises(dataclasses.FrozenInstanceError):
        head.sequence_number = 2  # type: ignore[misc]


def _raise_mid_append_failure(
    sequence_number: int,
    event_type: str,
    created_at: str,
    payload_json: str,
    prev_hash: str,
) -> NoReturn:
    """Simulate a failure between `BEGIN IMMEDIATE` and `COMMIT` (issue #76).

    Matches `compute_event_hash`'s exact signature so it can replace the
    function wholesale via `monkeypatch.setattr`, firing at the same point
    `append` calls it: after the SELECT-last read, before the INSERT.

    Args:
        sequence_number: Unused; present only to mirror the replaced
            function's signature.
        event_type: Unused; see `sequence_number`.
        created_at: Unused; see `sequence_number`.
        payload_json: Unused; see `sequence_number`.
        prev_hash: Unused; see `sequence_number`.

    Raises:
        RuntimeError: Always, with a fixed injected-failure message.
    """
    raise RuntimeError("injected mid-append failure")


def test_append_rolls_back_and_recovers_after_mid_transaction_failure(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure between BEGIN and COMMIT propagates, rolls back, and heals.

    Appends one good event, snapshots the ledger, then injects a failure
    (via a monkeypatched `compute_event_hash`) inside the second append's
    open transaction. The failure must propagate as `RuntimeError` rather
    than being swallowed, the ledger must be byte-for-byte unchanged by the
    failed append, and -- critically -- a further append afterwards must
    succeed and continue the chain at sequence 2. Without an explicit
    ROLLBACK, the failed append leaves the connection's `BEGIN IMMEDIATE`
    open, so this final append instead raises `sqlite3.OperationalError:
    cannot start a transaction within a transaction`.
    """
    store = ledger_store_factory()
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))
    snapshot_before_failure = store.read_all()
    monkeypatch.setattr(
        "windbreak.ledger.store.compute_event_hash", _raise_mid_append_failure
    )

    with pytest.raises(RuntimeError, match="injected mid-append failure"):
        store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))

    monkeypatch.undo()
    recovery_sequence = store.append(
        ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=2)
    )

    assert recovery_sequence == 2
    records = store.read_all()
    assert len(records) == 2
    assert records[:-1] == snapshot_before_failure
    store.verify_chain()


def test_append_rolls_back_after_mid_transaction_failure_on_empty_ledger(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure on the very first append (the `last is None` branch) also heals.

    Injects the same mid-transaction failure on an empty ledger's first
    append, then undoes the patch and appends again: the recovered append
    must land at sequence 1 with the genesis `prev_hash`, exactly as if the
    failed attempt had never happened.
    """
    store = ledger_store_factory()
    monkeypatch.setattr(
        "windbreak.ledger.store.compute_event_hash", _raise_mid_append_failure
    )

    with pytest.raises(RuntimeError, match="injected mid-append failure"):
        store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))

    monkeypatch.undo()
    recovery_sequence = store.append(
        ConfigLoaded(component="pipeline", config_hash="abc", diff={})
    )

    assert recovery_sequence == 1
    records = store.read_all()
    assert len(records) == 1
    assert records[0].prev_hash == GENESIS_PREV_HASH


class _CommitFailingConnection:
    """A `sqlite3` connection proxy that fails one `COMMIT` on demand (issue #76).

    Wraps a real :class:`sqlite3.Connection`, delegating `execute` and `close`
    verbatim, except that once :attr:`armed` is set the *next* `COMMIT`
    statement raises :class:`RuntimeError` (and disarms itself). Because
    `BEGIN IMMEDIATE`, the SELECT-last read, `compute_event_hash`, and the
    `INSERT` all still run for real, the injected failure lands *after* a row
    has been inserted into the open transaction -- the post-INSERT
    (commit-time) fault a `ROLLBACK`->`COMMIT` mutant of `append`'s
    except-clause would otherwise survive.

    Attributes:
        armed: When `True`, the next `COMMIT` execute raises instead of
            committing; the proxy clears it so later commits proceed.
    """

    def __init__(self, real: sqlite3.Connection) -> None:
        """Wrap `real` with the one-shot COMMIT fault disarmed.

        Args:
            real: The genuine SQLite connection every call delegates to.
        """
        self._real = real
        self.armed = False

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        """Delegate to the real connection, failing one armed `COMMIT`.

        Args:
            sql: The SQL statement to execute.
            *args: Optional bound parameters, forwarded verbatim.

        Returns:
            The real connection's cursor for every delegated statement.

        Raises:
            RuntimeError: On the first `COMMIT` seen while armed, simulating a
                commit-time failure after a successful INSERT; the proxy
                disarms itself so the recovery append can commit normally.
        """
        if self.armed and sql.strip().upper().startswith("COMMIT"):
            self.armed = False
            raise RuntimeError("injected commit failure")
        return self._real.execute(sql, *args)

    def close(self) -> None:
        """Close the wrapped real connection."""
        self._real.close()


def test_append_rolls_back_when_commit_itself_fails_after_insert(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A COMMIT-time failure (post-INSERT) still ROLLBACKs the inserted row.

    Pins the load-bearing append-only property at the point the two
    `compute_event_hash`-based rollback tests cannot reach. Those inject the
    failure *before* the INSERT, so no row is ever written and a mutant
    swapping `append`'s except-clause `ROLLBACK` for `COMMIT` would still pass
    them. This test instead lets the SELECT-last read, hashing, and `INSERT`
    all run for real, then forces `COMMIT` *itself* to fail via a one-shot
    connection proxy. The injected `RuntimeError` must propagate, and
    `append`'s `ROLLBACK` must undo the already-inserted row -- so `read_all`
    is byte-for-byte the pre-failure snapshot, the connection heals (a further
    append commits at sequence 2), and `verify_chain` sees a contiguous 1..2
    chain. A `ROLLBACK`->`COMMIT` mutant would instead persist the failed row,
    failing the snapshot-equality assertion -- which is what kills it.
    """
    real_connect = sqlite3.connect
    created_proxies: list[_CommitFailingConnection] = []

    def fake_connect(*args: Any, **kwargs: Any) -> _CommitFailingConnection:
        proxy = _CommitFailingConnection(real_connect(*args, **kwargs))
        created_proxies.append(proxy)
        return proxy

    monkeypatch.setattr(sqlite3, "connect", fake_connect)

    store = ledger_store_factory()
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))
    snapshot_before_failure = store.read_all()

    created_proxies[-1].armed = True
    with pytest.raises(RuntimeError, match="injected commit failure"):
        store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))

    assert store.read_all() == snapshot_before_failure

    recovery_sequence = store.append(
        ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=2)
    )

    assert recovery_sequence == 2
    assert [record.sequence_number for record in store.read_all()] == [1, 2]
    store.verify_chain()


# --- events_from_records: reconstructing Events from persisted rows (#235) -----


def test_events_from_records_round_trips_a_typed_events_fields(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """`events_from_records` reconstructs a base `Event` whose `event_type`,
    `component`, `payload_schema_version`, and `payload` exactly match the
    original typed event appended through the store -- the round trip issue
    #235's ledger-replay startup path depends on.
    """
    store = ledger_store_factory()
    original = ModeHeartbeat(component="riskkernel", mode="RESEARCH", beat=1)
    store.append(original)

    events = events_from_records(store.read_all())

    assert isinstance(events, tuple)
    assert len(events) == 1
    rebuilt = events[0]
    assert rebuilt.event_type == original.event_type
    assert rebuilt.component == original.component
    assert rebuilt.payload_schema_version == original.payload_schema_version
    assert rebuilt.payload == original.payload


def test_events_from_records_of_an_empty_ledger_returns_an_empty_tuple(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """An empty ledger's records fold to an empty tuple, not `None` or a
    list -- so a caller can iterate the result unconditionally.
    """
    store = ledger_store_factory()

    assert events_from_records(store.read_all()) == ()


def test_events_from_records_raises_value_error_on_an_envelope_missing_data() -> None:
    """A record whose envelope is missing the required `"data"` key raises
    `ValueError` -- the fail-closed contract issue #235's kill-replay startup
    path depends on: a corrupt or malformed envelope must never silently
    reconstruct a wrong `Event` (e.g. one with a fabricated empty payload).
    """
    malformed_payload_json = canonical_json(
        {"component": "riskkernel", "schema_version": 1}
    )
    record = LedgerRecord(
        sequence_number=1,
        event_type="ModeHeartbeat",
        created_at="2024-01-01T00:00:00.000000+00:00",
        component="riskkernel",
        payload_json=malformed_payload_json,
        payload_schema_version=1,
        prev_hash=GENESIS_PREV_HASH,
        event_hash="a" * 64,
    )

    with pytest.raises(ValueError):
        events_from_records([record])


# --- issue #246: indexed reverse lookup of the newest record of given types -----
#
# `latest_gate_plan_registration` scans the WHOLE ledger (`read_all()`) at least
# twice per promotion attempt just to find the last registration row.
# `SqliteLedgerStore.latest_record_of_types` answers that question with a single
# indexed `ORDER BY sequence_number DESC LIMIT 1` read, and is declared through
# the NARROW, optional `LatestRecordLookup` capability protocol -- deliberately
# separate from `LedgerStore`, which several hand-rolled test doubles implement
# structurally and which must therefore stay exactly as wide as it is.
#
# Neither the method nor the protocol exists yet, so this module's import of
# `LatestRecordLookup` fails collection with `ImportError: cannot import name
# 'LatestRecordLookup' from 'windbreak.ledger.store'` -- the expected Gate 1 RED
# state for issue #246.


def test_latest_record_of_types_returns_none_for_an_empty_ledger(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """An empty ledger has no record of any type, so the lookup returns `None`
    rather than raising -- the caller distinguishes "absent" from "corrupt".
    """
    store = ledger_store_factory()

    assert store.latest_record_of_types({"ConfigLoaded"}) is None


def test_latest_record_of_types_returns_none_when_no_row_matches(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """A non-empty ledger holding no row of the requested types returns `None`,
    never the newest row of some other type (which would be a wrong answer the
    fail-closed gate-plan read would then trust).
    """
    store = ledger_store_factory()
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))

    assert store.latest_record_of_types({"ModeHeartbeat"}) is None


def test_latest_record_of_types_returns_the_highest_sequence_match(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """Among several matching rows the newest (highest `sequence_number`) wins,
    and every one of the eight persisted columns round-trips identically to the
    same row read back through `read_all` -- the scan this read supersedes.
    """
    store = ledger_store_factory()
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=2))

    latest = store.latest_record_of_types({"ModeHeartbeat"})

    assert latest == store.read_all()[2]


def test_latest_record_of_types_spans_every_requested_type(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """With more than one requested type the newest row across the WHOLE set
    wins -- the property the two-event gate-plan vocabulary
    (`GatePlanRegistered` / `GatePlanChanged`) depends on.
    """
    store = ledger_store_factory()
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))

    latest = store.latest_record_of_types({"ModeHeartbeat", "ConfigLoaded"})

    assert latest is not None
    assert latest.sequence_number == 2
    assert latest.event_type == "ConfigLoaded"


def test_latest_record_of_types_returns_none_for_an_empty_type_set(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """Asking for no types at all matches nothing, so the lookup returns `None`
    without emitting a degenerate `IN ()` query.
    """
    store = ledger_store_factory()
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))

    assert store.latest_record_of_types(frozenset()) is None


def test_sqlite_ledger_store_satisfies_the_optional_lookup_capability() -> None:
    """`SqliteLedgerStore` structurally satisfies `LatestRecordLookup`, which is
    how `latest_gate_plan_registration` duck-type dispatches onto the indexed
    read instead of the full scan.
    """
    assert issubclass(SqliteLedgerStore, LatestRecordLookup)


def test_latest_record_lookup_is_not_satisfied_by_a_bare_ledger_store() -> None:
    """A hand-rolled `LedgerStore` double that predates the capability does NOT
    satisfy `LatestRecordLookup` -- proof the new capability is separately
    declared and that widening `LedgerStore` itself was not required.
    """

    class _BareStore:
        """A minimal structural `LedgerStore` with no indexed lookup."""

        def append(self, event: Event) -> int:
            """Raise; this double is never appended to."""
            raise NotImplementedError(event)

        def read_all(self) -> list[LedgerRecord]:
            """Return no records."""
            return []

        def verify_chain(self) -> None:
            """Do nothing; the double has no chain."""

        def close(self) -> None:
            """Do nothing; the double holds no resources."""

    assert not isinstance(_BareStore(), LatestRecordLookup)


# --- issue #370: bounded reverse scan of ONE event type -------------------------
#
# `windbreak.scheduler.loop.start_of_day_equity_micros` needs the EARLIEST
# `EquitySampled` row of the current UTC day, which `latest_record_of_types`
# cannot answer -- it returns the newest row, and reading the newest sample would
# raise the daily-loss threshold every time equity grew intraday. Answering it
# without a full `read_all()` on every tick needs a LAZY, newest-first walk over
# rows of one type, so the consumer can stop the moment it crosses the day
# boundary.
#
# `iter_records_of_type_reversed` and the narrow `ReverseTypeScan` capability it
# is declared through do not exist yet, so this module's import of
# `ReverseTypeScan` fails collection with `ImportError: cannot import name
# 'ReverseTypeScan' from 'windbreak.ledger.store'` -- the expected Gate 1 RED
# state for issue #370.


def test_iter_records_of_type_reversed_yields_only_that_type_newest_first(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """The walk yields rows of the requested type ONLY, in descending sequence
    order. Newest-first is the whole point: a consumer bounded by a recency
    predicate can stop early only if the newest rows arrive first.
    """
    store = ledger_store_factory()
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=2))

    walked = list(store.iter_records_of_type_reversed("ModeHeartbeat"))

    assert [record.sequence_number for record in walked] == [3, 1]
    assert walked == [store.read_all()[2], store.read_all()[0]]


def test_iter_records_of_type_reversed_is_empty_when_no_row_matches(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """A type with no rows walks to nothing rather than raising, so an absent
    answer stays distinguishable from a corrupt one.
    """
    store = ledger_store_factory()
    store.append(ConfigLoaded(component="pipeline", config_hash="abc", diff={}))

    assert list(store.iter_records_of_type_reversed("ModeHeartbeat")) == []


def test_iter_records_of_type_reversed_is_lazy_and_abandonable(
    ledger_store_factory: Callable[..., SqliteLedgerStore],
) -> None:
    """The walk is a true `Iterator`, not a materialized list, and abandoning it
    after one row leaves the store fully usable.

    Laziness is the load-bearing half of the bound: a `fetchall()` behind this
    method would still read every row of the type out of SQLite, so the
    consumer's early stop would save JSON decoding and nothing else. Abandoning
    matters because early-stopping is the normal case -- if the unfinished read
    left a lock behind, the very next `append` would fail.
    """
    store = ledger_store_factory()
    for beat in range(5):
        store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=beat))

    walk: Iterator[LedgerRecord] = store.iter_records_of_type_reversed("ModeHeartbeat")
    first = next(walk)
    del walk

    assert not isinstance(store.iter_records_of_type_reversed("ModeHeartbeat"), list)
    assert first.sequence_number == 5
    assert store.append(ConfigLoaded(component="p", config_hash="a", diff={})) == 6


def test_sqlite_ledger_store_satisfies_the_reverse_type_scan_capability() -> None:
    """`SqliteLedgerStore` structurally satisfies `ReverseTypeScan`, which is how
    the start-of-day equity read dispatches onto the bounded walk instead of the
    full scan.
    """
    assert issubclass(SqliteLedgerStore, ReverseTypeScan)


def test_reverse_type_scan_is_not_satisfied_by_a_bare_ledger_store() -> None:
    """A hand-rolled `LedgerStore` double does NOT satisfy `ReverseTypeScan` --
    proof this second capability is separately declared too, so `LedgerStore`
    stayed exactly as wide as it was and every existing double still works.
    """

    class _BareStore:
        """A minimal structural `LedgerStore` with no reverse walk."""

        def append(self, event: Event) -> int:
            """Raise; this double is never appended to."""
            raise NotImplementedError(event)

        def read_all(self) -> list[LedgerRecord]:
            """Return no records."""
            return []

        def verify_chain(self) -> None:
            """Do nothing; the double has no chain."""

        def close(self) -> None:
            """Do nothing; the double holds no resources."""

    assert not isinstance(_BareStore(), ReverseTypeScan)
