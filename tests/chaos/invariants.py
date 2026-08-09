"""Pure SPEC S11.5 invariant checkers for the Order Gateway chaos suite (#42).

Four invariants must hold after *every* chaos scenario reaches quiescence
(``OrderGateway.recover()`` -> ``Reconciler.run_once()``/``run()`` to fixpoint
-> ``Sweeper.sweep_once()``/``run()`` to fixpoint):

    1. :func:`assert_no_duplicate_live_orders` -- no two currently-resting
       venue orders share one content-addressed ``client_order_id``.
    2. :func:`assert_no_tokenless_orders` -- every currently-resting venue
       order traces back to a Gateway-verified, WAL-journalled submission (or
       is explicitly flagged by a durable ``ReconciliationHalted`` -- a
       fail-closed halt is convergence, not a live, unaccounted-for order).
    3. :func:`assert_no_net_short_positions` -- no ticker's held quantity is
       negative, unless a durable ``ReduceOnlyViolation`` halt names that
       *exact* ticker (the fail-closed halt *is* the safe outcome the
       invariant demands, but only for the ticker it names -- an unrelated
       ticker's short with no violation naming it still fails).
    4. :func:`assert_reservations_balanced` -- every ``client_order_id``'s
       ledgered ``OrderTransitionLedgered`` history replays as a single legal
       state-machine chain from ``INTENT_CREATED``. This is the observable
       proxy for the Gateway's in-flight-closing "reservation" bookkeeping
       (issue #39/#40): every place where that in-memory tally is opened
       (an ``ACK`` transition) or released (``FILL``/``CANCEL``) is *also* a
       ledgered transition, so a reservation opened or released more than
       once (or out of sequence) necessarily leaves a corrupted per-coid
       chain -- the only durable, purely-external symptom such a bug can
       leave, since the tally itself is a private Gateway attribute this
       module never reaches into.

Every function here is pure: it reads only *public*, durable state -- the
ledger (:class:`~windbreak.ledger.store.LedgerRecord`, the same source
:meth:`~windbreak.order_gateway.gateway.OrderGateway.recover` and
:class:`~windbreak.order_gateway.reconciler.Reconciler` fold), the write-ahead
log (:class:`~windbreak.order_gateway.wal.WalRecord`, equally durable and
public via ``WriteAheadLog.read_all()``), and the venue's live truth
(:class:`~windbreak.connector.models.OpenOrder`/``Position``, read exactly as
:meth:`~windbreak.connector.paper.PaperExchange.get_open_orders`/
``get_positions`` return them). None ever reaches into a Gateway private
attribute (``_acks``, ``_tracked``, ``_inflight_closing``, ...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from windbreak.order_gateway.state_machine import (
    IllegalTransitionError,
    OrderEvent,
    OrderState,
    transition,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from windbreak.connector.models import OpenOrder, Position
    from windbreak.ledger.store import LedgerRecord
    from windbreak.order_gateway.wal import WalRecord

#: The ledgered event type recording one order-lifecycle state transition.
_ORDER_TRANSITION = "OrderTransitionLedgered"

#: The ledgered event type recording a fail-closed reconciliation/recovery halt.
_RECONCILIATION_HALTED = "ReconciliationHalted"

#: The ledgered event type recording a fail-closed post-fill net-short halt.
_REDUCE_ONLY_VIOLATION = "ReduceOnlyViolation"


@dataclass(frozen=True, slots=True)
class GatewaySnapshot:
    """The complete public-state snapshot the four SPEC S11.5 invariants read.

    Every field is durable/observable state only -- reconciliation's own
    inputs, never a Gateway private attribute.

    Attributes:
        ledger_records: Every durable ledger row, in ascending sequence order.
        wal_records: Every durable write-ahead record, in append order.
        open_orders: The venue's currently resting orders.
        positions: The currently held positions, as reported by a wired
            reduce-only position source; empty when enforcement is off, and
            empty on an untraded venue (the paper exchange folds its holdings
            out of its own fill log, see
            :meth:`~windbreak.connector.paper.PaperExchange.get_positions`).
    """

    ledger_records: Sequence[LedgerRecord]
    wal_records: Sequence[WalRecord] = ()
    open_orders: Sequence[OpenOrder] = ()
    positions: Sequence[Position] = ()


def _payload_data(record: LedgerRecord) -> dict[str, object]:
    """Return one ledger record's decoded ``data`` payload.

    Args:
        record: The ledger record to decode.

    Returns:
        The record's ``data`` mapping, decoded from its ``payload_json``.
    """
    return cast("dict[str, object]", json.loads(record.payload_json)["data"])


def _order_id_to_coid(wal_records: Sequence[WalRecord]) -> dict[str, str]:
    """Map every WAL-acked venue order id to its owning ``client_order_id``.

    Args:
        wal_records: The write-ahead log records to scan.

    Returns:
        A mapping from venue ``order_id`` to ``client_order_id``, built from
        every ack record that left a resting order behind (``order_id`` is
        not ``None``).
    """
    mapping: dict[str, str] = {}
    for record in wal_records:
        if record.kind == "ack" and record.order_id is not None:
            mapping[record.order_id] = record.client_order_id
    return mapping


def _halted_venue_order_ids(ledger_records: Sequence[LedgerRecord]) -> frozenset[str]:
    """Return every venue order id named by a durable reconciliation halt.

    Args:
        ledger_records: The ledger records to scan.

    Returns:
        The set of ``venue_order_id`` values carried by every
        ``ReconciliationHalted`` record whose id is non-empty.
    """
    ids: set[str] = set()
    for record in ledger_records:
        if record.event_type != _RECONCILIATION_HALTED:
            continue
        order_id = str(_payload_data(record).get("venue_order_id", ""))
        if order_id:
            ids.add(order_id)
    return frozenset(ids)


def _reduce_only_halted_tickers(
    ledger_records: Sequence[LedgerRecord],
) -> frozenset[str]:
    """Return every ticker named by a durable ``ReduceOnlyViolation`` halt.

    Args:
        ledger_records: The ledger records to scan.

    Returns:
        The set of ``ticker`` values carried by every ``ReduceOnlyViolation``
        record whose ticker is non-empty. Each such record names exactly the
        one ticker whose close overshot the held position (see
        :class:`~windbreak.ledger.events.ReduceOnlyViolation`); the halt it
        latches is process-wide (the Gateway refuses all further work), but
        the invariant only excuses a net-short on the *named* ticker -- an
        unrelated ticker's transient short, with no violation naming it, is
        never a symptom this halt explains.
    """
    tickers: set[str] = set()
    for record in ledger_records:
        if record.event_type != _REDUCE_ONLY_VIOLATION:
            continue
        ticker = str(_payload_data(record).get("ticker", ""))
        if ticker:
            tickers.add(ticker)
    return frozenset(tickers)


def assert_no_duplicate_live_orders(snapshot: GatewaySnapshot) -> None:
    """Assert no two currently-resting venue orders share one ``client_order_id``.

    Content-addressed ``client_order_id``s are the Gateway's anti-duplicate
    mechanism (SPEC S11.2): a genuine double-submission of the same economic
    intent would show up here as two currently-open venue orders both tracing
    (via their WAL ack records) back to the identical coid.

    Args:
        snapshot: The public-state snapshot to check.

    Raises:
        AssertionError: If any ``client_order_id`` correlates to more than one
            currently-open venue order.
    """
    coid_by_order_id = _order_id_to_coid(snapshot.wal_records)
    order_ids_by_coid: dict[str, list[str]] = {}
    for order in snapshot.open_orders:
        coid = coid_by_order_id.get(order.id)
        if coid is None:
            continue
        order_ids_by_coid.setdefault(coid, []).append(order.id)
    duplicates = {coid: ids for coid, ids in order_ids_by_coid.items() if len(ids) > 1}
    assert not duplicates, (
        f"duplicate live venue orders detected for client_order_id(s): {duplicates!r}"
    )


def assert_no_tokenless_orders(snapshot: GatewaySnapshot) -> None:
    """Assert every currently-resting venue order traces to a verified token.

    An order counts as accounted for when a WAL ack record correlates its
    venue ``order_id`` back to a Gateway-journalled submission. A resting
    order with no such trace is only acceptable when it has *already* been
    flagged by a durable ``ReconciliationHalted`` naming that exact venue
    order id: per SPEC S3.2/S11.4 (when in doubt, halt), the Gateway's
    fail-closed response to an unaccounted-for order *is* the safe,
    convergent outcome -- it never autonomously treats it as valid, and this
    checker mirrors that by excusing only the already-flagged case.

    Args:
        snapshot: The public-state snapshot to check.

    Raises:
        AssertionError: If any currently-open venue order has neither a WAL
            ack trace nor a durable reconciliation halt naming it.
    """
    coid_by_order_id = _order_id_to_coid(snapshot.wal_records)
    halted_order_ids = _halted_venue_order_ids(snapshot.ledger_records)
    unaccounted = [
        order.id
        for order in snapshot.open_orders
        if order.id not in coid_by_order_id and order.id not in halted_order_ids
    ]
    assert not unaccounted, (
        "live venue order(s) with no Gateway-verified token trace, and not "
        f"flagged by a durable ReconciliationHalted: {unaccounted!r}"
    )


def assert_no_net_short_positions(snapshot: GatewaySnapshot) -> None:
    """Assert no ticker's held position is standing net-short.

    A post-fill net-short breach halts the Gateway fail-closed (a durable
    ``ReduceOnlyViolation``, SPEC S11.5); that halt latch is itself the
    convergent, safe outcome the invariant demands, so it excuses the check
    -- but *only* for the ticker the violation names. ``ReduceOnlyViolation``
    carries the one ``ticker`` whose close overshot its held position (see
    :class:`~windbreak.ledger.events.ReduceOnlyViolation`); although the halt
    it latches is process-wide (the Gateway refuses all further work), that
    says nothing about any *other* ticker's position, so a short standing on
    an unrelated, unnamed ticker still fails this check -- mirroring how
    :func:`assert_no_tokenless_orders` excuses only the specific venue order
    a ``ReconciliationHalted`` names, never every resting order globally.

    Args:
        snapshot: The public-state snapshot to check.

    Raises:
        AssertionError: If any position is negative on a ticker not named by
            a ``ReduceOnlyViolation`` halt latched in the ledger.
    """
    halted_tickers = _reduce_only_halted_tickers(snapshot.ledger_records)
    shorts = {
        position.ticker: position.quantity.value
        for position in snapshot.positions
        if position.quantity.value < 0 and position.ticker not in halted_tickers
    }
    assert not shorts, (
        f"net-short position(s) standing with no halt latched: {shorts!r}"
    )


def assert_reservations_balanced(snapshot: GatewaySnapshot) -> None:
    """Assert every coid's ledgered transition history is a single legal chain.

    Replays each ``client_order_id``'s ledgered ``(from_state, event)`` pairs,
    in ledger (append) order, through the pure :func:`transition` function,
    starting at ``INTENT_CREATED``. A reservation-relevant transition
    (``ACK`` opens the in-flight-closing tally; ``FILL``/``CANCEL`` releases
    it) recorded more than once, or out of sequence, breaks this replay --
    either the recorded ``from_state`` disagrees with the replayed state, or
    the move is outright illegal -- which is exactly the observable symptom a
    double-open or double-release bug would leave (see the module docstring).

    Args:
        snapshot: The public-state snapshot to check.

    Raises:
        AssertionError: If any coid's ledgered chain is illegal or internally
            inconsistent.
    """
    by_coid: dict[str, list[dict[str, object]]] = {}
    for record in snapshot.ledger_records:
        if record.event_type != _ORDER_TRANSITION:
            continue
        data = _payload_data(record)
        by_coid.setdefault(str(data["client_order_id"]), []).append(data)
    for coid, transitions in by_coid.items():
        _assert_legal_chain(coid, transitions)


def _assert_legal_chain(coid: str, transitions: list[dict[str, object]]) -> None:
    """Replay one coid's ledgered transitions, asserting a single legal chain.

    Args:
        coid: The client-order-id whose transitions are being replayed.
        transitions: The coid's ledgered ``OrderTransitionLedgered`` payloads,
            in ledger (append) order.

    Raises:
        AssertionError: If the replay disagrees with any ledgered
            ``from_state``/``to_state``, or an illegal move is ledgered.
    """
    state = OrderState.INTENT_CREATED
    for data in transitions:
        assert data["from_state"] == state.name, (
            f"coid {coid!r}: ledgered from_state {data['from_state']!r} "
            f"disagrees with the replayed state {state.name!r} -- a "
            "reservation-relevant transition was re-recorded out of "
            "sequence (a double-open or double-release)"
        )
        event = OrderEvent[str(data["event"])]
        try:
            state = transition(state, event)
        except IllegalTransitionError as exc:
            raise AssertionError(
                f"coid {coid!r}: illegal replayed transition "
                f"{state.name} --{event.name}--> ? (ledgered to_state was "
                f"{data['to_state']!r})"
            ) from exc
        assert data["to_state"] == state.name, (
            f"coid {coid!r}: ledgered to_state {data['to_state']!r} "
            f"disagrees with the pure transition() result {state.name!r}"
        )


def assert_all_invariants(snapshot: GatewaySnapshot) -> None:
    """Assert all four SPEC S11.5 invariants hold for ``snapshot``.

    Args:
        snapshot: The public-state snapshot to check.

    Raises:
        AssertionError: If any of the four invariants is violated.
    """
    assert_no_duplicate_live_orders(snapshot)
    assert_no_tokenless_orders(snapshot)
    assert_no_net_short_positions(snapshot)
    assert_reservations_balanced(snapshot)
