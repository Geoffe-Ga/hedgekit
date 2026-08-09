"""Indexed reverse lookup for the latest gate-plan registration (issue #246).

`latest_gate_plan_registration` replays the WHOLE ledger via `read_all()` to
find the single newest `GatePlanRegistered`/`GatePlanChanged` row, and
`RiskKernel.request_promotion` drives it at least twice per PAPER promotion
attempt (once through `anchor_gate_evidence`, once through the gate build). It
now duck-type dispatches onto the optional
`windbreak.ledger.store.LatestRecordLookup` capability -- an indexed
`ORDER BY sequence_number DESC LIMIT 1` read -- whenever the store declares it,
and falls back to the existing scan whenever it does not.

The capability is deliberately a SEPARATE, narrow protocol rather than three
more lines on `LedgerStore`: several hand-rolled `LedgerStore` doubles across
this suite satisfy that protocol structurally, and widening it would break every
one of them. `test_falls_back_to_the_full_scan_without_the_capability` is the
direct proof that the fallback path is still live.

RED today: `windbreak.ledger.store` exports no `LatestRecordLookup`, so this
module fails collection with `ImportError: cannot import name
'LatestRecordLookup' from 'windbreak.ledger.store'`. Once it exists, the
dispatch test next fails as `AssertionError` -- `latest_gate_plan_registration`
still calls `read_all()` -- which is the behavioral RED this issue closes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from windbreak.config import EvaluationConfig
from windbreak.evaluation.preregistration import (
    build_gate_plan,
    latest_gate_plan_registration,
    register_gate_plan,
)
from windbreak.ledger.store import LatestRecordLookup, SqliteLedgerStore

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from windbreak.ledger.events import Event
    from windbreak.ledger.store import LedgerRecord

#: A fixed epoch for every registration below, so no test depends on the wall
#: clock.
_FIXED_EPOCH_S = 1_700_000_000

#: The pinned fill-model version (SPEC §17.4); its value never affects the
#: lookup, so one constant is reused throughout.
_PAPER_FILL_MODEL_VERSION = "indexed-lookup-test-v1"


def _registered_store(tmp_path: Path, name: str = "ledger.db") -> SqliteLedgerStore:
    """Open a store at `tmp_path / name` holding one registered gate plan.

    Args:
        tmp_path: The per-test temporary directory the database is rooted in.
        name: The database filename.

    Returns:
        An open `SqliteLedgerStore` whose ledger holds exactly one
        `GatePlanRegistered` row.
    """
    store = SqliteLedgerStore(tmp_path / name)
    plan = build_gate_plan(
        EvaluationConfig(), paper_fill_model_version=_PAPER_FILL_MODEL_VERSION
    )
    register_gate_plan(plan, store, now=lambda: _FIXED_EPOCH_S)
    return store


class _ScanCountingStore:
    """A `LedgerStore` delegate that counts how often `read_all` is called.

    Wraps a real, hash-chained `SqliteLedgerStore` so every answer stays
    genuine; only the call accounting is added. Whether the wrapper forwards
    `latest_record_of_types` decides which of the two lookup paths
    `latest_gate_plan_registration` takes, which is exactly what these tests
    pin.

    Attributes:
        read_all_calls: How many times `read_all` has been called.
    """

    def __init__(self, inner: SqliteLedgerStore) -> None:
        """Wrap `inner`, starting the scan counter at zero.

        Args:
            inner: The real store every call is delegated to.
        """
        self._inner = inner
        self.read_all_calls = 0

    def append(self, event: Event) -> int:
        """Delegate the append.

        Args:
            event: The event to persist.

        Returns:
            The sequence number the inner store assigned.
        """
        return self._inner.append(event)

    def read_all(self) -> list[LedgerRecord]:
        """Delegate the full scan, counting the call.

        Returns:
            Every persisted record in ascending sequence order.
        """
        self.read_all_calls += 1
        return self._inner.read_all()

    def verify_chain(self) -> None:
        """Delegate chain verification."""
        self._inner.verify_chain()

    def close(self) -> None:
        """Delegate the close."""
        self._inner.close()


class _IndexedScanCountingStore(_ScanCountingStore):
    """A `_ScanCountingStore` that also declares the optional lookup capability."""

    def latest_record_of_types(
        self, event_types: Collection[str]
    ) -> LedgerRecord | None:
        """Delegate the indexed reverse lookup.

        Args:
            event_types: The event types to match.

        Returns:
            The highest-sequence matching record, or `None` when none matches.
        """
        return self._inner.latest_record_of_types(event_types)


def test_dispatches_to_the_indexed_lookup_without_scanning(tmp_path: Path) -> None:
    """A store declaring `LatestRecordLookup` is read through the indexed
    lookup: the registration comes back correctly and `read_all` is never
    called, so promotion attempts no longer cost O(ledger).
    """
    inner = _registered_store(tmp_path)
    store = _IndexedScanCountingStore(inner)
    try:
        registration = latest_gate_plan_registration(store)

        assert registration is not None
        assert registration.paper_clock_start == _FIXED_EPOCH_S
        assert store.read_all_calls == 0
    finally:
        store.close()


def test_falls_back_to_the_full_scan_without_the_capability(tmp_path: Path) -> None:
    """A store that does NOT declare the capability still resolves the same
    registration, through the original `read_all` scan -- the fallback every
    hand-rolled `LedgerStore` double in this suite depends on.
    """
    inner = _registered_store(tmp_path)
    store = _ScanCountingStore(inner)
    try:
        registration = latest_gate_plan_registration(store)

        assert registration is not None
        assert registration.paper_clock_start == _FIXED_EPOCH_S
        assert store.read_all_calls == 1
    finally:
        store.close()


def test_both_paths_agree_on_the_latest_of_two_registrations(tmp_path: Path) -> None:
    """After a `GatePlanChanged` supersedes the first plan, the indexed lookup
    and the scan return the SAME registration -- the newest one. A reverse
    lookup that returned the first row would silently pin promotion to a stale
    plan, the exact anti-Goodhart failure the gate exists to prevent.
    """
    inner = _registered_store(tmp_path)
    changed_plan = build_gate_plan(
        EvaluationConfig(promotion_min_resolved=999),
        paper_fill_model_version=_PAPER_FILL_MODEL_VERSION,
    )
    register_gate_plan(changed_plan, inner, now=lambda: _FIXED_EPOCH_S + 1)
    indexed = _IndexedScanCountingStore(inner)
    scanning = _ScanCountingStore(inner)
    try:
        from_index = latest_gate_plan_registration(indexed)
        from_scan = latest_gate_plan_registration(scanning)

        assert from_index == from_scan
        assert from_index is not None
        assert from_index.plan.promotion_min_resolved == 999
    finally:
        inner.close()


def test_the_capability_is_narrow_enough_to_leave_ledger_store_doubles_alone(
    tmp_path: Path,
) -> None:
    """The scan-only double is a valid `LedgerStore` yet is NOT a
    `LatestRecordLookup`: the capability is separately declared, so no existing
    double had to grow a method to keep working.
    """
    inner = _registered_store(tmp_path)
    store = _ScanCountingStore(inner)
    try:
        assert not isinstance(store, LatestRecordLookup)
        assert isinstance(_IndexedScanCountingStore(inner), LatestRecordLookup)
    finally:
        store.close()
