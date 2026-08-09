"""Event types and canonical serialization for the hash-chained ledger.

This module defines the M0 event vocabulary that every windbreak process
records into the append-only ledger, plus the two serialization
primitives the store hashes over:

- :func:`canonical_json` -- a deterministic, whitespace-free JSON encoding
  whose output depends only on a value's contents, never on dict insertion
  order, so identical events always hash identically.
- :func:`utc_now_iso` -- a UTC ISO-8601 timestamp with microsecond
  precision, used as each record's ``created_at``.

Each concrete event (:class:`ConfigLoaded`, :class:`ModeHeartbeat`,
:class:`AlertEmitted`, and the growing family of gateway, PAPER-loop, and
evaluation events -- the three evaluation-defined
:class:`GatePlanRegistered`/:class:`GatePlanChanged`/
:class:`GateComputationMismatch` events live here too, per issue #180, so the
evaluation package stays a one-way, acyclic consumer of this module) is a frozen
dataclass with an ergonomic, typed constructor that derives its ``event_type``
(the class name), its ``payload_schema_version``, and its ``payload`` dict. The
:attr:`Event.envelope_json` property wraps those into the persisted
envelope ``{"component", "data", "schema_version"}``, and
:data:`EVENT_TYPES` maps each ``event_type`` string back to its class so a
persisted envelope can be reconstructed as
``EVENT_TYPES[event_type](component=..., **data)``.

Example:
    >>> event = ConfigLoaded(component="pipeline", config_hash="abc", diff={})
    >>> event.event_type
    'ConfigLoaded'
    >>> event.payload
    {'config_hash': 'abc', 'diff': {}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

#: All-zero, SHA-256-width sentinel used as the ``prev_hash`` of the first
#: record in a chain, since it has no predecessor to link back to.
GENESIS_PREV_HASH = "0" * 64

#: Schema version stamped on every M0 event payload. Bump when a payload's
#: shape changes so old and new records remain distinguishable.
_SCHEMA_VERSION = 1


def canonical_json(obj: dict[str, object]) -> str:
    """Serialize a mapping to deterministic, whitespace-free JSON.

    Keys are emitted in sorted order and separators carry no spaces, so the
    output is a byte-stable function of the value's contents alone --
    independent of dict insertion order. This is the exact form the ledger
    hashes over.

    Args:
        obj: The mapping to serialize.

    Returns:
        The canonical JSON encoding of ``obj``.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 microsecond timestamp.

    Returns:
        A string such as ``"2024-01-01T00:00:00.000000+00:00"``.
    """
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Event:
    """A ledger event carrying its type, source, schema version, and payload.

    Attributes:
        event_type: Discriminator naming the concrete event kind.
        component: The process or subsystem that produced the event.
        payload_schema_version: Version of the payload's shape.
        payload: The event's type-specific data.
    """

    event_type: str
    component: str
    payload_schema_version: int
    payload: dict[str, object]

    @property
    def envelope_json(self) -> str:
        """Return the canonical JSON envelope persisted for this event.

        Returns:
            Canonical JSON of ``{"component", "data", "schema_version"}``,
            where ``data`` is this event's payload.
        """
        return canonical_json(
            {
                "component": self.component,
                "data": self.payload,
                "schema_version": self.payload_schema_version,
            }
        )


def _derive_typed_event(
    event: Event,
    payload: dict[str, object],
    *,
    schema_version: int = _SCHEMA_VERSION,
) -> None:
    """Populate the derived ``Event`` fields on a frozen typed subclass.

    Sets ``event_type`` to the concrete class name, ``payload_schema_version``
    to ``schema_version`` (the module-wide default for every event but the one
    that overrides it), and ``payload`` to the assembled dict, using
    ``object.__setattr__`` because the instances are frozen.

    Args:
        event: The freshly constructed typed event to populate.
        payload: The type-specific payload assembled by the subclass.
        schema_version: The payload schema version to stamp; defaults to the
            module-wide :data:`_SCHEMA_VERSION`. Only an event whose payload
            shape has diverged from its v1 form (``ForecastCreated``, #188)
            supplies an override.
    """
    object.__setattr__(event, "event_type", type(event).__name__)
    object.__setattr__(event, "payload_schema_version", schema_version)
    object.__setattr__(event, "payload", payload)


@dataclass(frozen=True)
class ConfigLoaded(Event):
    """Records that a component loaded a configuration revision.

    Attributes:
        config_hash: Content hash identifying the loaded configuration.
        diff: The change from the previously active configuration.
    """

    config_hash: str
    diff: dict[str, object]
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "config_hash": self.config_hash,
            "diff": self.diff,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ModeHeartbeat(Event):
    """Records a periodic liveness beat for a component's operating mode.

    Attributes:
        mode: The operating mode reported by the beat.
        beat: The monotonically increasing heartbeat counter.
    """

    mode: str
    beat: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {"mode": self.mode, "beat": self.beat}
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class AlertEmitted(Event):
    """Records that a component emitted an operational alert.

    Attributes:
        severity: The alert's severity label.
        message: Human-readable description of the alert.
    """

    severity: str
    message: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "severity": self.severity,
            "message": self.message,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class PromotionEvaluated(Event):
    """Records the outcome of one Risk Kernel promotion-gate evaluation.

    Attributes:
        source_mode: The mode promotion was requested from (``Mode.name``).
        target_mode: The mode promotion was toward (``Mode.name``).
        approved: Whether every gate criterion passed.
        override_bypassed: Whether an active significance override promoted
            despite the mandatory significance criterion failing -- ``True``
            only when the override rescued a promotion the raw evaluation
            (``approved is False``) would otherwise have blocked.
        evidence: The evaluated ``GateEvidence`` snapshot as a payload.
        results: One per-criterion result payload, in gate order.
    """

    source_mode: str
    target_mode: str
    approved: bool
    override_bypassed: bool
    evidence: dict[str, object]
    results: list[dict[str, object]]
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "source_mode": self.source_mode,
            "target_mode": self.target_mode,
            "approved": self.approved,
            "override_bypassed": self.override_bypassed,
            "evidence": self.evidence,
            "results": self.results,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class SignificanceOverrideApplied(Event):
    """Records the one-way significance-gate override (SPEC S5.1).

    Attributes:
        operator_ack: The exact acknowledgement phrase the operator typed.
        ceiling: The override's mode ceiling (always ``"LIVE_MICRO"``).
    """

    operator_ack: str
    ceiling: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "operator_ack": self.operator_ack,
            "ceiling": self.ceiling,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class DemotionTriggerFired(Event):
    """Records one Risk Kernel demotion-trigger firing.

    Attributes:
        trigger: The firing trigger (``DemotionTrigger.name``).
        action: The trigger's mapped action (``DemotionAction.name``).
        from_mode: The mode at firing time (``Mode.name``).
        to_mode: The mode after resolution (``Mode.name``; equals ``from_mode``
            on a no-op firing).
        transitioned: Whether the firing actually moved the mode.
    """

    trigger: str
    action: str
    from_mode: str
    to_mode: str
    transitioned: bool
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "trigger": self.trigger,
            "action": self.action,
            "from_mode": self.from_mode,
            "to_mode": self.to_mode,
            "transitioned": self.transitioned,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class KillEngaged(Event):
    """Records that the Risk Kernel kill switch engaged (issue #35).

    The single announcement event every kill emits, whichever of the four
    triggers fired it. It never carries a sell/close/submit/dump action: a kill
    only halts and cancels, never trades (SPEC position-hold invariant).

    Attributes:
        trigger: The firing trigger's name (``KillTrigger.name``).
        kill_sequence: The monotonic, strictly-increasing kill counter, so a
            re-arm and a subsequent kill are always distinguishable.
        epoch: The wall-clock instant of the kill, in whole epoch seconds (an
            ``int``, never a float -- SPEC S6.1).
    """

    trigger: str
    kill_sequence: int
    epoch: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "trigger": self.trigger,
            "kill_sequence": self.kill_sequence,
            "epoch": self.epoch,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class CancelAllDirective(Event):
    """Records the kill switch's one cancel-all-open-orders directive (issue #35).

    The kill switch cancels resting orders; it never closes or sells the
    positions those orders would have touched (SPEC position-hold invariant),
    so the scope names only open *orders*.

    Attributes:
        scope: The cancellation scope (always ``"all_open_orders"``).
    """

    scope: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {"scope": self.scope}
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class KillReArmed(Event):
    """Records a successful typed-confirmation re-arm out of KILLED (issue #35).

    Attributes:
        kill_sequence: The kill counter of the kill this re-arm cleared, tying
            the re-arm back to its originating kill in the audit trail.
    """

    kill_sequence: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {"kill_sequence": self.kill_sequence}
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class OrderTransitionLedgered(Event):
    """Records one Order Gateway state-machine transition (issue #38).

    Attributes:
        client_order_id: The content-addressed id of the intent this transition
            belongs to (see :func:`~windbreak.order_gateway.client_order_id`).
        from_state: The state the transition moved from (``OrderState.name``).
        event: The event that drove the transition (``OrderEvent.name``).
        to_state: The state the transition moved to (``OrderState.name``).
    """

    client_order_id: str
    from_state: str
    event: str
    to_state: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "client_order_id": self.client_order_id,
            "from_state": self.from_state,
            "event": self.event,
            "to_state": self.to_state,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class SubmissionRefused(Event):
    """Records a submission refused before any transition or submit (issue #38).

    Emitted when the Gateway declines an intent up front -- e.g. the exchange is
    not open, or crash recovery has not yet completed -- so the token is never
    verified or consumed and the submitter is never called.

    Attributes:
        client_order_id: The content-addressed id of the refused intent (see
            :func:`~windbreak.order_gateway.client_order_id`).
        reason: A short human-readable reason for the refusal.
    """

    client_order_id: str
    reason: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "client_order_id": self.client_order_id,
            "reason": self.reason,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ReduceOnlyRefused(Event):
    """Records a close refused for exceeding its closeable headroom (issue #39).

    Emitted when the Gateway declines a ``SELL_TO_CLOSE`` whose size exceeds the
    held position net of in-flight closes -- *before* the token is verified or
    consumed, so a refusal never burns the token's single use. The five count
    fields pin the exact numbers the reduce-only verdict was computed from (see
    :class:`~windbreak.order_gateway.reduce_only.PositionSnapshot`).

    Attributes:
        client_order_id: The content-addressed id of the refused close (see
            :func:`~windbreak.order_gateway.client_order_id`).
        ticker: The market ticker the close targeted.
        held_centis: The net held position for ``ticker``, in contract-centis.
        inflight_closing_centis: The sum of closes already in flight for
            ``ticker``, in contract-centis.
        requested_close_centis: The refused close's size, in contract-centis.
        reason: A short machine-readable reason (always ``"reduce_only"``).
    """

    client_order_id: str
    ticker: str
    held_centis: int
    inflight_closing_centis: int
    requested_close_centis: int
    reason: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "client_order_id": self.client_order_id,
            "ticker": self.ticker,
            "held_centis": self.held_centis,
            "inflight_closing_centis": self.inflight_closing_centis,
            "requested_close_centis": self.requested_close_centis,
            "reason": self.reason,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ReduceOnlyViolation(Event):
    """Records a post-fill net-short breach that halts the Gateway (issue #39).

    Emitted when a close filled more than was held, leaving the position
    net-short (``net_centis < 0``). This is the fail-closed halt trigger (SPEC
    S11.5): the Gateway records this event and then refuses all further work.
    Crash recovery (issue #40) folds this durable fact and stays halted -- there
    is no un-halt event.

    Attributes:
        client_order_id: The content-addressed id of the offending close (see
            :func:`~windbreak.order_gateway.client_order_id`).
        ticker: The market ticker the close targeted.
        held_centis: The net held position observed for ``ticker`` after the
            fill, in contract-centis.
        filled_centis: The quantity the venue reported filled, in
            contract-centis.
        net_centis: ``held_centis - filled_centis``, negative on a breach.
    """

    client_order_id: str
    ticker: str
    held_centis: int
    filled_centis: int
    net_centis: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "client_order_id": self.client_order_id,
            "ticker": self.ticker,
            "held_centis": self.held_centis,
            "filled_centis": self.filled_centis,
            "net_centis": self.net_centis,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ReconciliationHalted(Event):
    """Records a crash-recovery/reconciler mismatch that halts the Gateway (#40).

    Emitted the fail-closed instant reconciliation cannot safely resolve the
    venue against the durable ledger/WAL (SPEC S3.2/S11.4: when in doubt, halt).
    ``reason`` is a closed set of exactly three members: ``"foreign_open_order"``
    (a resting order with no durable trace), ``"vanished_order_no_fill"`` (a
    tracked order gone with no corroborating fill), and ``"ambiguous_match"`` (a
    placed order whose completing WAL-ack was never written, so it cannot be
    correlated). Inapplicable id fields carry the ``""`` sentinel so the payload
    round-trips through ``EVENT_TYPES[t](component=..., **data)`` without a
    ``None``-typed field.

    Attributes:
        reason: The closed-set halt reason (see above).
        ticker: The market ticker the offending order/fill was on, or ``""``.
        venue_order_id: The venue's resting-order id involved, or ``""``.
        client_order_id: The correlated content-addressed id, or ``""`` when the
            mismatch could not be tied to a known intent.
        detail: A short human-readable diagnostic describing the mismatch.
    """

    reason: str
    ticker: str
    venue_order_id: str
    client_order_id: str
    detail: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "reason": self.reason,
            "ticker": self.ticker,
            "venue_order_id": self.venue_order_id,
            "client_order_id": self.client_order_id,
            "detail": self.detail,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ReconciliationHealed(Event):
    """Records a benign reconciliation heal (issue #40).

    Emitted when reconciliation confirms an out-of-band but *expected* effect --
    a missed fill on a Gateway-placed order, or the safe disposition of an intent
    that never reached the venue -- so the order's ledgered lifecycle can be
    advanced without a halt.

    Attributes:
        client_order_id: The content-addressed id of the healed intent (see
            :func:`~windbreak.order_gateway.client_order_id`).
        action: A short machine-readable label for the heal that was applied
            (e.g. ``"fill_confirmed"``).
        detail: A short human-readable diagnostic describing the heal.
    """

    client_order_id: str
    action: str
    detail: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "client_order_id": self.client_order_id,
            "action": self.action,
            "detail": self.detail,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class RecoveryCompleted(Event):
    """Records that crash recovery finished reconciling the venue (issue #40).

    The final record a clean ``OrderGateway.recover()`` writes, doubling as the
    anchor for the next recovery's fills-since checkpoint. When ``halted`` is
    ``True`` the Gateway completed recovery in a fail-closed state and never
    accepts approvals.

    Attributes:
        orders_reconciled: The number of tracked orders reconciled this recovery.
        halted: Whether recovery finished with the Gateway halted.
    """

    orders_reconciled: int
    halted: bool
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "orders_reconciled": self.orders_reconciled,
            "halted": self.halted,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class MarketFreeze(Event):
    """Records a strict beyond-N-ticks move freezing a whole ticker (issue #41).

    Emitted by the adverse-selection sweeper the instant a resting order's
    side-matched top of book has gapped strictly beyond ``threshold_ticks``
    ticks from that order's own captured baseline: the whole ticker is frozen
    and every resting order on it is cancelled. Exactly one of these is ledgered
    per frozen ticker per sweep, carrying the first breaching order's baseline
    and the shared observed reference. ``event_type`` is the literal class name
    ``"MarketFreeze"``, derived like every other concrete event via
    :func:`_derive_typed_event` (never a shouty-snake-case variant).

    Attributes:
        ticker: The market ticker that was frozen.
        trigger: The machine-readable trigger label (always ``"cancel_on_move"``).
        baseline_price_pips: The first breaching order's captured limit, in pips.
        observed_price_pips: The side-matched top of book at freeze time, in pips.
        threshold_ticks: The policy's move threshold, in ticks.
        price_tick_pips: The market's price tick, in pips (an int -- SPEC S6.1).
        epoch: The wall-clock instant of the freeze, in whole epoch seconds.
    """

    ticker: str
    trigger: str
    baseline_price_pips: int
    observed_price_pips: int
    threshold_ticks: int
    price_tick_pips: int
    epoch: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "ticker": self.ticker,
            "trigger": self.trigger,
            "baseline_price_pips": self.baseline_price_pips,
            "observed_price_pips": self.observed_price_pips,
            "threshold_ticks": self.threshold_ticks,
            "price_tick_pips": self.price_tick_pips,
            "epoch": self.epoch,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ReturnToScreener(Event):
    """Records a frozen ticker's orders returned to re-screening (issue #41).

    The companion to :class:`MarketFreeze`: emitted once per frozen ticker per
    sweep, after every resting order on the ticker has been cancelled, marking
    the ticker as handed back to manual/algorithmic re-screening.

    Attributes:
        ticker: The market ticker whose orders were returned to the screener.
        reason: The machine-readable reason (always ``"market_freeze"``).
        epoch: The wall-clock instant of the return, in whole epoch seconds.
    """

    ticker: str
    reason: str
    epoch: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "ticker": self.ticker,
            "reason": self.reason,
            "epoch": self.epoch,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class MarketSnapshotRecorded(Event):
    """Records one PAPER-loop market snapshot's top of book (issue #48).

    The best bid/ask are scaled-integer pips (never a float, SPEC S6.1) and each
    is ``None`` for a missing (empty) book side, so a one-sided or empty book is
    representable without fabricating a zero price.

    Attributes:
        ticker: The market the snapshot is for.
        best_bid_pips: The top-of-book best YES bid, in pips, or ``None``.
        best_ask_pips: The top-of-book best YES ask, in pips, or ``None``.
        fetched_at_epoch_s: When the book was fetched, in whole epoch seconds.
    """

    ticker: str
    best_bid_pips: int | None
    best_ask_pips: int | None
    fetched_at_epoch_s: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "ticker": self.ticker,
            "best_bid_pips": self.best_bid_pips,
            "best_ask_pips": self.best_ask_pips,
            "fetched_at_epoch_s": self.fetched_at_epoch_s,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ScreenDecisionRecorded(Event):
    """Records one PAPER-loop screening verdict for a market (issue #48).

    Attributes:
        ticker: The market the verdict is for.
        eligible: Whether the market passed screening.
        blocked_by: The screening filters that blocked it (empty when eligible).
    """

    ticker: str
    eligible: bool
    blocked_by: list[str]
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "ticker": self.ticker,
            "eligible": self.eligible,
            "blocked_by": self.blocked_by,
        }
        _derive_typed_event(self, payload)


#: ``ForecastCreated``'s payload schema version (issue #188). Bumped to ``2``
#: -- the first M0-family event whose version is not the module-wide
#: :data:`_SCHEMA_VERSION` -- when the two cost/baseline fields the weekly
#: evaluation/cost-meter fold reads were added, so a v1-shaped row already on
#: disk is distinguishable from a v2 one without inspecting the payload's keys.
_FORECAST_CREATED_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ForecastCreated(Event):
    """Records one PAPER-loop forecast's headline figures (issue #48, #188).

    Issue #188 adds ``research_cost_micros`` and ``market_price_baseline_pips``
    -- the two fields the weekly evaluation/cost-meter fold reads verbatim off
    the ledgered payload -- and bumps ``payload_schema_version`` to ``2`` (see
    :data:`_FORECAST_CREATED_SCHEMA_VERSION`). Both new fields are scaled ints,
    never floats (SPEC S6.1).

    Attributes:
        forecast_id: The forecast's deterministic id.
        market_ticker: The market the forecast is for.
        probability_ppm: The forecast probability, in parts-per-million.
        eligible_for_live: Whether the forecast may back a live order.
        abstention_reason: Why the engine abstained, or ``None`` when it did not.
        research_cost_micros: The forecast's research spend, in micros.
        market_price_baseline_pips: The baseline executable price, in pips.
    """

    forecast_id: str
    market_ticker: str
    probability_ppm: int
    eligible_for_live: bool
    abstention_reason: str | None
    research_cost_micros: int
    market_price_baseline_pips: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "forecast_id": self.forecast_id,
            "market_ticker": self.market_ticker,
            "probability_ppm": self.probability_ppm,
            "eligible_for_live": self.eligible_for_live,
            "abstention_reason": self.abstention_reason,
            "research_cost_micros": self.research_cost_micros,
            "market_price_baseline_pips": self.market_price_baseline_pips,
        }
        _derive_typed_event(
            self, payload, schema_version=_FORECAST_CREATED_SCHEMA_VERSION
        )


@dataclass(frozen=True)
class SelectorDecisionRecorded(Event):
    """Records one PAPER-loop selector decision (issue #48).

    Attributes:
        forecast_id: The originating forecast's id.
        market_ticker: The market the decision is for.
        intent_count: How many normalized intents the selector emitted.
        reasons: The selector's pinned reasons for its verdict.
    """

    forecast_id: str
    market_ticker: str
    intent_count: int
    reasons: list[str]
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "forecast_id": self.forecast_id,
            "market_ticker": self.market_ticker,
            "intent_count": self.intent_count,
            "reasons": self.reasons,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class EquitySampled(Event):
    """Records one PAPER-loop equity sample against the floor (issue #48).

    Every field is a scaled int (never a float, SPEC S6.1): ``equity_micros`` and
    ``floor_micros`` are money-micros and ``epoch_s`` is whole epoch seconds.

    Attributes:
        equity_micros: The sampled account equity, in money-micros.
        floor_micros: The configured equity floor, in money-micros.
        epoch_s: When the sample was taken, in whole epoch seconds.
    """

    equity_micros: int
    floor_micros: int
    epoch_s: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "equity_micros": self.equity_micros,
            "floor_micros": self.floor_micros,
            "epoch_s": self.epoch_s,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class PositionsSnapshotRecorded(Event):
    """Records one PAPER-loop snapshot of open positions (issue #48).

    Attributes:
        positions: One JSON-safe row per held position (empty when flat). Each
            row's numeric fields (``quantity_centis``/``average_price_pips``) are
            scaled ints, never floats.
    """

    positions: list[dict[str, object]]
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {"positions": self.positions}
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class DrillCompleted(Event):
    """Records the graded outcome of one operational drill (issue #59).

    The single event ``windbreak.drills.framework.run_drill`` appends to the
    operational ledger for every graded drill result, win or lose, so the
    "CI runs every drill" guarantee is auditable. Like every other concrete
    event its ``event_type`` is the literal class name ``"DrillCompleted"``
    (never a shouty-snake-case variant), derived via :func:`_derive_typed_event`.

    Attributes:
        drill: The drill's registry name (e.g. ``"kill-rearm"``).
        passed: Whether the drill's graded result passed.
        evidence: The drill's JSON-serializable evidence payload; it never
            carries secret material (drills that touch credentials report only
            variable names, booleans, and fingerprints).
    """

    drill: str
    passed: bool
    evidence: dict[str, object]
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "drill": self.drill,
            "passed": self.passed,
            "evidence": self.evidence,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class GatePlanRegistered(Event):
    """Records the first registration of a gate plan into the ledger (issue #13).

    Moved here from :mod:`windbreak.evaluation.preregistration` (issue #180) so
    the evaluation package stays a one-way runtime consumer of this module. The
    constructor's fields ARE the flattened payload keys -- the thirteen canonical
    :meth:`~windbreak.evaluation.preregistration.GatePlan.canonical_dict` keys
    plus ``plan_hash``/``paper_clock_start`` -- so the persisted payload never
    carries a separate ``plan_dict`` wrapper and round-trips through
    ``EVENT_TYPES[event_type](component=..., **data)`` by construction.

    Attributes:
        metric_windows: The ``[name, window]`` metric/window catalogue, as the
            JSON list-of-two-element-lists form.
        min_resolved_for_calibration: Minimum resolved forecasts before
            calibration statistics are computed.
        promotion_min_resolved: Minimum resolved forecasts required to promote.
        promotion_min_independent_event_groups: Minimum independent event groups
            required to promote.
        brier_skill_required_ppm: Required Brier skill score, in ppm.
        bootstrap_confidence_ppm: Bootstrap confidence level, in ppm.
        live_rolling_window_size: Rolling-window size for the live-divergence
            gates.
        live_slippage_ratio_limit_ppm: Live-vs-paper slippage ratio ceiling, in
            ppm.
        live_brier_degradation_band_ppm: Allowed LIVE-over-PAPER rolling Brier
            degradation, in ppm.
        observation_window: The headline observation window value.
        baseline_scheme: The named executable-price baseline scheme.
        clustering_scheme: The named event-correlation clustering scheme.
        paper_fill_model_version: The paper fill-model version pinned into the
            plan's identity (SPEC §17.4).
        plan_hash: The registered plan's content hash.
        paper_clock_start: The whole-epoch-second instant the paper clock started
            for this plan.
    """

    metric_windows: list[list[str]]
    min_resolved_for_calibration: int
    promotion_min_resolved: int
    promotion_min_independent_event_groups: int
    brier_skill_required_ppm: int
    bootstrap_confidence_ppm: int
    live_rolling_window_size: int
    live_slippage_ratio_limit_ppm: int
    live_brier_degradation_band_ppm: int
    observation_window: str
    baseline_scheme: str
    clustering_scheme: str
    paper_fill_model_version: str
    plan_hash: str
    paper_clock_start: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the flattened payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "metric_windows": self.metric_windows,
            "min_resolved_for_calibration": self.min_resolved_for_calibration,
            "promotion_min_resolved": self.promotion_min_resolved,
            "promotion_min_independent_event_groups": (
                self.promotion_min_independent_event_groups
            ),
            "brier_skill_required_ppm": self.brier_skill_required_ppm,
            "bootstrap_confidence_ppm": self.bootstrap_confidence_ppm,
            "live_rolling_window_size": self.live_rolling_window_size,
            "live_slippage_ratio_limit_ppm": self.live_slippage_ratio_limit_ppm,
            "live_brier_degradation_band_ppm": self.live_brier_degradation_band_ppm,
            "observation_window": self.observation_window,
            "baseline_scheme": self.baseline_scheme,
            "clustering_scheme": self.clustering_scheme,
            "paper_fill_model_version": self.paper_fill_model_version,
            "plan_hash": self.plan_hash,
            "paper_clock_start": self.paper_clock_start,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class GatePlanChanged(Event):
    """Records a change from one registered gate plan to a different one (#13).

    The companion to :class:`GatePlanRegistered`, moved here from
    :mod:`windbreak.evaluation.preregistration` (issue #180). Carries every field
    :class:`GatePlanRegistered` does plus ``previous_plan_hash`` linking back to
    the plan this one replaced; like it, the constructor's fields ARE the
    flattened payload keys (no ``plan_dict`` wrapper).

    Attributes:
        metric_windows: The ``[name, window]`` metric/window catalogue, as the
            JSON list-of-two-element-lists form.
        min_resolved_for_calibration: Minimum resolved forecasts before
            calibration statistics are computed.
        promotion_min_resolved: Minimum resolved forecasts required to promote.
        promotion_min_independent_event_groups: Minimum independent event groups
            required to promote.
        brier_skill_required_ppm: Required Brier skill score, in ppm.
        bootstrap_confidence_ppm: Bootstrap confidence level, in ppm.
        live_rolling_window_size: Rolling-window size for the live-divergence
            gates.
        live_slippage_ratio_limit_ppm: Live-vs-paper slippage ratio ceiling, in
            ppm.
        live_brier_degradation_band_ppm: Allowed LIVE-over-PAPER rolling Brier
            degradation, in ppm.
        observation_window: The headline observation window value.
        baseline_scheme: The named executable-price baseline scheme.
        clustering_scheme: The named event-correlation clustering scheme.
        paper_fill_model_version: The paper fill-model version pinned into the
            plan's identity (SPEC §17.4).
        plan_hash: The new plan's content hash.
        paper_clock_start: The whole-epoch-second instant the paper clock reset
            to on this change (strictly later than the prior registration's).
        previous_plan_hash: The content hash of the plan this one replaced.
    """

    metric_windows: list[list[str]]
    min_resolved_for_calibration: int
    promotion_min_resolved: int
    promotion_min_independent_event_groups: int
    brier_skill_required_ppm: int
    bootstrap_confidence_ppm: int
    live_rolling_window_size: int
    live_slippage_ratio_limit_ppm: int
    live_brier_degradation_band_ppm: int
    observation_window: str
    baseline_scheme: str
    clustering_scheme: str
    paper_fill_model_version: str
    plan_hash: str
    paper_clock_start: int
    previous_plan_hash: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the flattened payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "metric_windows": self.metric_windows,
            "min_resolved_for_calibration": self.min_resolved_for_calibration,
            "promotion_min_resolved": self.promotion_min_resolved,
            "promotion_min_independent_event_groups": (
                self.promotion_min_independent_event_groups
            ),
            "brier_skill_required_ppm": self.brier_skill_required_ppm,
            "bootstrap_confidence_ppm": self.bootstrap_confidence_ppm,
            "live_rolling_window_size": self.live_rolling_window_size,
            "live_slippage_ratio_limit_ppm": self.live_slippage_ratio_limit_ppm,
            "live_brier_degradation_band_ppm": self.live_brier_degradation_band_ppm,
            "observation_window": self.observation_window,
            "baseline_scheme": self.baseline_scheme,
            "clustering_scheme": self.clustering_scheme,
            "paper_fill_model_version": self.paper_fill_model_version,
            "plan_hash": self.plan_hash,
            "paper_clock_start": self.paper_clock_start,
            "previous_plan_hash": self.previous_plan_hash,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class GateComputationMismatch(Event):
    """Records that the SQL and Python gate paths disagreed on a crosscheck (#55).

    Moved here verbatim from :mod:`windbreak.evaluation.crosscheck` (issue #180)
    so the evaluation package stays a one-way runtime consumer of this module.

    Attributes:
        plan_hash: The gate plan's content hash the run was scored under.
        tolerance: The integer tolerance the comparison used.
        mismatches: One entry per disagreeing metric, each shaped
            ``{"name", "window", "python_value", "sql_value"}`` with any sentinel
            rendered by its ``.name``.
    """

    plan_hash: str
    tolerance: int
    mismatches: list[dict[str, object]]
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "plan_hash": self.plan_hash,
            "tolerance": self.tolerance,
            "mismatches": self.mismatches,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class CanaryVerdictRecorded(Event):
    """Records one provider's canary verdict (fleet observability, issue #195).

    The scheduler's ``run_canaries`` composition root appends one of these per
    provider per canary battery run (SPEC S8.4/S16 extended per-provider), so a
    later ledger fold surfaces each provider's live drift status. Like every
    concrete event its ``event_type`` is the literal class name
    ``"CanaryVerdictRecorded"``, derived via :func:`_derive_typed_event`, and
    every payload leaf is int/str/bool/list -- never a float.

    Attributes:
        provider: The provider this verdict is for.
        status: The verdict status
            (``ProviderCanaryStatus.name``: ``"OK"``/``"ANSWER_DRIFT"``/
            ``"VERSION_DRIFT"``).
        drift_kind: The drift kind (``"answer"``, ``"version"``, or ``""`` for a
            clean ``OK`` verdict -- never ``None``, matching this module's
            inapplicable-string convention).
        drift_score_ppm: The scored answer-drift distance, in ppm.
        tolerance_ppm: The drift tolerance the score was gated against, in ppm.
        reported_version: The forecaster version the provider reported.
        pinned_versions: The provider's operator-pinned version strings (plural:
            a pin set may accept more than one accepted version).
    """

    provider: str
    status: str
    drift_kind: str
    drift_score_ppm: int
    tolerance_ppm: int
    reported_version: str
    pinned_versions: list[str]
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "provider": self.provider,
            "status": self.status,
            "drift_kind": self.drift_kind,
            "drift_score_ppm": self.drift_score_ppm,
            "tolerance_ppm": self.tolerance_ppm,
            "reported_version": self.reported_version,
            "pinned_versions": self.pinned_versions,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class PromotionBlocked(Event):
    """Records a fail-closed PAPER->LIVE_MICRO promotion attempt (issue #244).

    Emitted when a PAPER promotion is refused before any gate is evaluated
    because no readable pre-registered gate plan was available (the fail-closed
    path of issue #185, whose deliberate no-event default this event opts into
    per follow-up #185). Like every concrete event its ``event_type`` is the
    literal class name ``"PromotionBlocked"``, derived via
    :func:`_derive_typed_event`, and every payload leaf is a ``str``. It is only
    emitted when the Risk Kernel is opted in via its ``ledger_blocked_promotions``
    flag; the default kernel fails closed silently as before.

    Attributes:
        source_mode: The mode promotion was requested from (``Mode.name``;
            ``"PAPER"`` on the only path that emits this event).
        target_mode: The mode promotion was toward (``Mode.name``;
            ``"LIVE_MICRO"``), stamped from the ladder even though the live gate
            -- and thus its ``.target`` -- could not be built.
        reason: The human-readable fail-closed message from the raised
            ``GatePlanUnavailableError`` (e.g. why no plan was readable).
    """

    source_mode: str
    target_mode: str
    reason: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "source_mode": self.source_mode,
            "target_mode": self.target_mode,
            "reason": self.reason,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ProviderVoteRecorded(Event):
    """Records one ensemble member's per-vote cost outcome (issue #281).

    The scheduler's ``_forecast_stage`` composition root appends one of these
    per ensemble member driven per paper tick (per-provider vote-cost signal),
    so a later ledger fold surfaces each provider's charged spend, abstention
    rate, and cost-per-forecast. Like every concrete event its ``event_type``
    is the literal class name ``"ProviderVoteRecorded"``, derived via
    :func:`_derive_typed_event`, and every payload leaf is int/str -- never a
    float, never ``None``.

    Attributes:
        forecast_id: The forecast this vote belongs to.
        market_ticker: The forecast's market ticker.
        provider: The provider identifier that cast the vote (the default
            ensemble repeats a provider across distinct model versions).
        model_version: The provider's pinned model version (unique per member).
        vote_index: The zero-based index of this vote in the driven ensemble.
        cost_micros: The vote's billed cost, in micros (charged even when the
            vote was discarded).
        outcome: The vote outcome (``"voted"``/``"abstained"``/``"discarded"``).
        failure_code: The discard failure code, or ``""`` for a non-discard
            (``"voted"``/``"abstained"``) -- never ``None``, matching this
            module's inapplicable-string convention (see
            :class:`CanaryVerdictRecorded` ``drift_kind``).
    """

    forecast_id: str
    market_ticker: str
    provider: str
    model_version: str
    vote_index: int
    cost_micros: int
    outcome: str
    failure_code: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "forecast_id": self.forecast_id,
            "market_ticker": self.market_ticker,
            "provider": self.provider,
            "model_version": self.model_version,
            "vote_index": self.vote_index,
            "cost_micros": self.cost_micros,
            "outcome": self.outcome,
            "failure_code": self.failure_code,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ProviderGateHeld(Event):
    """Records a per-provider track-record live-eligibility hold (issue #305).

    Appended by the scheduler's ``_forecast_stage`` composition root when the
    tick's :class:`~windbreak.forecast.providers.track_record.ProviderTrackRecordGate`
    finds any voting provider unproven and therefore forces the forecast
    live-ineligible (SPEC S13/S16 promotion bars, S19's no-unmeasured-edge
    mandate). The hold is the durable answer to "why did this otherwise healthy
    forecast not qualify to back a live order", and naming the providers plus
    the two bars they missed makes the decision reconstructable from the audit
    trail alone -- without it, a held forecast is indistinguishable from one
    that simply had thin citations. Like every concrete event its
    ``event_type`` is the literal class name ``"ProviderGateHeld"``, derived via
    :func:`_derive_typed_event`, and every payload leaf is int/str -- never a
    float, never ``None``.

    The votes themselves are *not* held: an unproven provider keeps forecasting
    (that is the only way its paper track record ever accrues), so this row sits
    alongside, never instead of, the tick's :class:`ProviderVoteRecorded` rows.

    Attributes:
        forecast_id: The forecast whose live eligibility was withheld.
        market_ticker: That forecast's market ticker.
        unproven_providers: The unproven voting providers, comma-joined in the
            sorted, deduplicated order the gate screens them in (e.g.
            ``"anthropic,openai"``). A scalar string rather than a list so the
            payload stays flat, matching this module's leaf convention.
        unproven_count: How many distinct providers ``unproven_providers``
            names -- carried explicitly so a fold never has to re-split the
            joined string to count them.
        min_resolved: The resolved-forecast bar in force for this decision,
            from ``config.forecast.provider_gate.min_resolved``.
        min_brier_skill_ppm: The Brier-skill bar in force for this decision, in
            ppm, from ``config.forecast.provider_gate.min_brier_skill_ppm``.
    """

    forecast_id: str
    market_ticker: str
    unproven_providers: str
    unproven_count: int
    min_resolved: int
    min_brier_skill_ppm: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "forecast_id": self.forecast_id,
            "market_ticker": self.market_ticker,
            "unproven_providers": self.unproven_providers,
            "unproven_count": self.unproven_count,
            "min_resolved": self.min_resolved,
            "min_brier_skill_ppm": self.min_brier_skill_ppm,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ResearchBudgetHalted(Event):
    """Records a fail-closed research halt on a budget ceiling (issue #339).

    Appended by the scheduler's PAPER composition root when the always-on loop's
    :class:`~windbreak.forecast.budget.ResearchBudget` refuses to spend: either
    the UTC day's cumulative research budget is exhausted, or one forecast's
    research cost strictly exceeds the per-forecast ceiling. The halt is the
    durable audit record that research stopped *and why*, so an operator can
    tell a budget halt apart from a quiet loop. Like every concrete event its
    ``event_type`` is the literal class name ``"ResearchBudgetHalted"``, derived
    via :func:`_derive_typed_event`, and every payload leaf is int/str -- never
    a float, never ``None``.

    Attributes:
        market_ticker: The breaching forecast's market ticker for a
            ``"per_forecast"`` halt, or ``""`` for a ``"per_day"`` halt, which
            is not attributable to any single market -- never ``None``, matching
            this module's inapplicable-string convention (see
            :class:`ProviderVoteRecorded` ``failure_code``).
        halt_kind: Which ceiling halted research: ``"per_day"`` or
            ``"per_forecast"``.
        utc_day: The halt's UTC calendar day, as an ISO ``YYYY-MM-DD`` string;
            the same key the budget buckets spend under.
        spent_micros: For a ``"per_day"`` halt, the day's cumulative spend at
            the halt; for a ``"per_forecast"`` halt, the single breaching
            forecast's research cost. In micros.
        budget_micros: The ceiling that was breached, in micros.
    """

    market_ticker: str
    halt_kind: str
    utc_day: str
    spent_micros: int
    budget_micros: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "market_ticker": self.market_ticker,
            "halt_kind": self.halt_kind,
            "utc_day": self.utc_day,
            "spent_micros": self.spent_micros,
            "budget_micros": self.budget_micros,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ExchangeStatusObserved(Event):
    """Records one observation of the exchange's trading status (issue #342).

    Appended by the scheduler each tick, at the instant the status is read, so
    the evidence the ``exchange_status_ok`` check consumed is auditable after
    the fact. The ``status`` string is the connector's own value, never
    synthesized; ``observed_at_epoch_s`` is when the observation happened, which
    is what the freshness check measures against.

    Attributes:
        status: The exchange's raw trading status (``"open"``/``"paused"``/
            ``"closed"``), verbatim from the connector.
        observed_at_epoch_s: Epoch second the status was observed.
    """

    status: str
    observed_at_epoch_s: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {
            "status": self.status,
            "observed_at_epoch_s": self.observed_at_epoch_s,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class PipelineHeartbeatRecorded(Event):
    """Records that the tick pipeline was alive at an instant (issue #342).

    Stamped once per tick, after the snapshot stage has proven the pipeline is
    genuinely running, and read by the ``pipeline_heartbeat_ok`` check. It is a
    distinct concept from :class:`ModeHeartbeat`, which carries the operating
    mode and beat number but no instant, and which is appended too late in the
    tick to serve as evidence for the approve stage.

    Attributes:
        heartbeat_epoch_s: Epoch second at which the pipeline was observed
            alive.
    """

    heartbeat_epoch_s: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        payload: dict[str, object] = {"heartbeat_epoch_s": self.heartbeat_epoch_s}
        _derive_typed_event(self, payload)


#: Maps each event_type string to its class, so a persisted envelope can be
#: reconstructed as ``EVENT_TYPES[event_type](component=..., **data)``.
EVENT_TYPES: dict[str, type[Event]] = {
    "ConfigLoaded": ConfigLoaded,
    "ModeHeartbeat": ModeHeartbeat,
    "AlertEmitted": AlertEmitted,
    "PromotionEvaluated": PromotionEvaluated,
    "SignificanceOverrideApplied": SignificanceOverrideApplied,
    "DemotionTriggerFired": DemotionTriggerFired,
    "KillEngaged": KillEngaged,
    "CancelAllDirective": CancelAllDirective,
    "KillReArmed": KillReArmed,
    "OrderTransitionLedgered": OrderTransitionLedgered,
    "SubmissionRefused": SubmissionRefused,
    "ReduceOnlyRefused": ReduceOnlyRefused,
    "ReduceOnlyViolation": ReduceOnlyViolation,
    "ReconciliationHalted": ReconciliationHalted,
    "ReconciliationHealed": ReconciliationHealed,
    "RecoveryCompleted": RecoveryCompleted,
    "MarketFreeze": MarketFreeze,
    "ReturnToScreener": ReturnToScreener,
    "MarketSnapshotRecorded": MarketSnapshotRecorded,
    "ScreenDecisionRecorded": ScreenDecisionRecorded,
    "ForecastCreated": ForecastCreated,
    "SelectorDecisionRecorded": SelectorDecisionRecorded,
    "EquitySampled": EquitySampled,
    "PositionsSnapshotRecorded": PositionsSnapshotRecorded,
    "DrillCompleted": DrillCompleted,
    "GatePlanRegistered": GatePlanRegistered,
    "GatePlanChanged": GatePlanChanged,
    "GateComputationMismatch": GateComputationMismatch,
    "CanaryVerdictRecorded": CanaryVerdictRecorded,
    "PromotionBlocked": PromotionBlocked,
    "ProviderVoteRecorded": ProviderVoteRecorded,
    "ProviderGateHeld": ProviderGateHeld,
    "ResearchBudgetHalted": ResearchBudgetHalted,
    "ExchangeStatusObserved": ExchangeStatusObserved,
    "PipelineHeartbeatRecorded": PipelineHeartbeatRecorded,
}
