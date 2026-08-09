"""Shared component: the append-only, hash-chained ledger (SPEC S5.1).

Provides the tamper-evident event log every process writes to and that
Evaluation and the Dashboard read from. Events (:class:`ConfigLoaded`,
:class:`ModeHeartbeat`, :class:`AlertEmitted`) are appended to a
:class:`SqliteLedgerStore`, which chains each record's SHA-256 hash to its
predecessor so :meth:`SqliteLedgerStore.verify_chain` can detect any
single-column corruption. :func:`rebuild` folds a verified ledger into
derived read models.

Head-hash anchoring (issue #75) closes the one gap ``verify_chain`` cannot:
:func:`anchor_head` records the chain head to an append-only anchor file and
:func:`verify_anchors` flags any live chain that a tail rewrite has moved away
from its anchors (:class:`AnchorMismatchError`), failing closed on a missing or
malformed anchor file (:class:`AnchorFormatError`).

The Order Gateway / crash-recovery event vocabulary (issue #38-#40) lives here
too, so a persisted envelope can be reconstructed from :data:`EVENT_TYPES`
regardless of which package produced it: :class:`OrderTransitionLedgered`,
:class:`SubmissionRefused`, :class:`ReduceOnlyRefused`,
:class:`ReduceOnlyViolation`, :class:`ReconciliationHalted`,
:class:`ReconciliationHealed`, and :class:`RecoveryCompleted`.

Example:
    >>> from pathlib import Path
    >>> store = SqliteLedgerStore(Path("ledger.db"))
    >>> event = ConfigLoaded(component="pipeline", config_hash="abc", diff={})
    >>> seq = store.append(event)
    >>> store.verify_chain()
    >>> store.close()
    >>> rebuild(Path("ledger.db"), Path("read_models"))
"""

from __future__ import annotations

from windbreak.ledger.anchor import (
    AnchorFormatError,
    AnchorMismatchError,
    AnchorRecord,
    anchor_command,
    anchor_head,
    read_anchors,
    verify_anchors,
    verify_command,
)
from windbreak.ledger.events import (
    EVENT_TYPES,
    GENESIS_PREV_HASH,
    AlertEmitted,
    ConfigLoaded,
    EquitySampled,
    Event,
    ExchangeStatusObserved,
    ForecastCreated,
    MarketSnapshotRecorded,
    ModeHeartbeat,
    OrderTransitionLedgered,
    PipelineHeartbeatRecorded,
    PositionsSnapshotRecorded,
    ProviderVoteRecorded,
    ReconciliationHalted,
    ReconciliationHealed,
    RecoveryCompleted,
    ReduceOnlyRefused,
    ReduceOnlyViolation,
    ResearchBudgetHalted,
    ScreenDecisionRecorded,
    SelectorDecisionRecorded,
    SubmissionRefused,
    canonical_json,
)
from windbreak.ledger.rebuild import rebuild, rebuild_command
from windbreak.ledger.store import (
    ChainHead,
    ChainIntegrityError,
    LatestRecordLookup,
    LedgerRecord,
    LedgerStore,
    SqliteLedgerStore,
    compute_event_hash,
    events_from_records,
)

__all__ = [
    "EVENT_TYPES",
    "GENESIS_PREV_HASH",
    "AlertEmitted",
    "AnchorFormatError",
    "AnchorMismatchError",
    "AnchorRecord",
    "ChainHead",
    "ChainIntegrityError",
    "ConfigLoaded",
    "EquitySampled",
    "Event",
    "ExchangeStatusObserved",
    "ForecastCreated",
    "LatestRecordLookup",
    "LedgerRecord",
    "LedgerStore",
    "MarketSnapshotRecorded",
    "ModeHeartbeat",
    "OrderTransitionLedgered",
    "PipelineHeartbeatRecorded",
    "PositionsSnapshotRecorded",
    "ProviderVoteRecorded",
    "ReconciliationHalted",
    "ReconciliationHealed",
    "RecoveryCompleted",
    "ReduceOnlyRefused",
    "ReduceOnlyViolation",
    "ResearchBudgetHalted",
    "ScreenDecisionRecorded",
    "SelectorDecisionRecorded",
    "SqliteLedgerStore",
    "SubmissionRefused",
    "anchor_command",
    "anchor_head",
    "canonical_json",
    "compute_event_hash",
    "events_from_records",
    "read_anchors",
    "rebuild",
    "rebuild_command",
    "verify_anchors",
    "verify_command",
]
