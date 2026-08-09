"""Failing-first tests for capital reservations and the approval pipeline
(issue #31, RED).

Issue #31 gives the Risk Kernel a `ReservationLedger` (SPEC S5.3/S10.6):
monotonic per-approval sequence numbers, forever-remembered intent-id and
idempotency-key uniqueness (even past release), decrease-only adjustment,
time-bounded expiry, and one ledgered `Event` per mutation -- plus an
`ApprovalPipeline` that stamps ledger-sourced state onto the evaluated
context, reserves capital, and issues a signed, single-use approval token
only when the check pipeline does not veto.

`windbreak/riskkernel/reservations.py` does not exist yet, so every import
below fails collection with `ModuleNotFoundError` -- the expected Gate 1 RED
state for issue #31.

The concurrency test (T4) is fully deterministic: a `threading.Barrier`
releases every worker at once, `join(timeout=...)` bounds every wait, there
is no `sleep`, and the outcome (exactly `k` of `N` approved) follows from the
ledger's single lock plus a monotonically increasing headroom consumption,
regardless of thread interleaving.
"""

from __future__ import annotations

import threading

import pytest

from tests.riskkernel.conftest import make_context, make_intent
from windbreak.numeric.types import MoneyMicros
from windbreak.riskkernel import checks as checks_module
from windbreak.riskkernel.process import InMemoryKernelLedgerWriter
from windbreak.riskkernel.reservations import (
    ApprovalPipeline,
    DuplicateReservationError,
    ReservationLedger,
)
from windbreak.riskkernel.signing import SigningKeyHandle
from windbreak.riskkernel.tokens import TokenIssuer

#: A fixed, valid (>=32-byte) signing key shared by every pipeline test below.
_KEY_MATERIAL = b"k" * 32

#: The default intent's worst-case cost under `make_context()`'s permissive
#: defaults (zero fees, zero rounding buffer): 5000 pips * 1000 centis ==
#: 5_000_000 micros. Documented and relied upon in `tests/riskkernel/conftest.py`
#: and `tests/riskkernel/test_checks.py`.
_DEFAULT_INTENT_COST_MICROS = 5_000_000

#: The default account cash / equity-relevant balances
#: (`tests/riskkernel/conftest.py`'s `_DEFAULT_EQUITY_MICROS`).
_DEFAULT_EQUITY_MICROS = 1_000_000_000


# --- ReservationLedger: monotonic sequence numbers ------------------------------


def test_reserve_assigns_monotonic_sequence_numbers_starting_at_one() -> None:
    """Sequential `reserve()` calls receive sequence numbers 1, 2, 3, ..."""
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())

    first = ledger.reserve("intent-1", MoneyMicros(100), "idem-1", expires_at=1_000)
    second = ledger.reserve("intent-2", MoneyMicros(100), "idem-2", expires_at=1_000)
    third = ledger.reserve("intent-3", MoneyMicros(100), "idem-3", expires_at=1_000)

    assert (first.sequence_number, second.sequence_number, third.sequence_number) == (
        1,
        2,
        3,
    )


# --- ReservationLedger: forever-remembered uniqueness ---------------------------


def test_reserve_duplicate_intent_id_raises_even_after_release() -> None:
    """An `intent_id` already used -- even by a since-released reservation --
    raises `DuplicateReservationError` forever.
    """
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(100), "idem-1", expires_at=1_000)
    ledger.release("intent-1", reason="test cleanup")

    with pytest.raises(DuplicateReservationError):
        ledger.reserve("intent-1", MoneyMicros(50), "idem-2", expires_at=2_000)


def test_reserve_duplicate_idempotency_key_raises_even_after_release() -> None:
    """An `idempotency_key` already used -- even by a since-released
    reservation -- raises `DuplicateReservationError` forever.
    """
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(100), "idem-1", expires_at=1_000)
    ledger.release("intent-1", reason="test cleanup")

    with pytest.raises(DuplicateReservationError):
        ledger.reserve("intent-2", MoneyMicros(50), "idem-1", expires_at=2_000)


def test_reserve_distinct_ids_and_keys_never_collide() -> None:
    """Reservations with distinct intent ids and idempotency keys never
    raise, and both accumulate in `total_reserved()`."""
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())

    ledger.reserve("intent-1", MoneyMicros(100), "idem-1", expires_at=1_000)
    ledger.reserve("intent-2", MoneyMicros(200), "idem-2", expires_at=1_000)

    assert ledger.total_reserved() == MoneyMicros(300)


# --- ReservationLedger: release drops total_reserved, keeps ids used -----------


def test_release_drops_total_reserved_but_keeps_ids_used_forever() -> None:
    """`release()` zeroes the reservation's contribution to
    `total_reserved()`, but its intent id and idempotency key remain in the
    "used" sets forever.
    """
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(500), "idem-1", expires_at=1_000)

    ledger.release("intent-1", reason="cancelled")

    assert ledger.total_reserved() == MoneyMicros(0)
    assert "intent-1" in ledger.used_intent_ids()
    assert "idem-1" in ledger.used_idempotency_keys()


# --- ReservationLedger: adjust is decrease-only ---------------------------------


def test_adjust_decreases_an_active_reservations_amount() -> None:
    """A valid decrease (`0 < new < current`) updates `total_reserved()`."""
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(500), "idem-1", expires_at=1_000)

    ledger.adjust("intent-1", MoneyMicros(300))

    assert ledger.total_reserved() == MoneyMicros(300)


def test_adjust_rejects_an_increase() -> None:
    """A `remaining_amount` above the current reservation raises."""
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(500), "idem-1", expires_at=1_000)

    with pytest.raises(ValueError, match="intent-1"):
        ledger.adjust("intent-1", MoneyMicros(600))


def test_adjust_rejects_an_amount_equal_to_the_current_reservation() -> None:
    """A `remaining_amount` equal to the current amount is not a decrease and
    raises -- the invariant is strict (`new < current`).
    """
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(500), "idem-1", expires_at=1_000)

    with pytest.raises(ValueError):
        ledger.adjust("intent-1", MoneyMicros(500))


def test_adjust_rejects_a_zero_amount() -> None:
    """A `remaining_amount` of zero raises -- use `release()` to zero a
    reservation, not `adjust()`."""
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(500), "idem-1", expires_at=1_000)

    with pytest.raises(ValueError):
        ledger.adjust("intent-1", MoneyMicros(0))


def test_adjust_rejects_an_unknown_intent_id() -> None:
    """Adjusting an intent id with no active reservation raises."""
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())

    with pytest.raises(ValueError, match="unknown-intent"):
        ledger.adjust("unknown-intent", MoneyMicros(1))


def test_adjust_rejects_a_released_reservations_intent_id() -> None:
    """Adjusting a since-released reservation raises -- `adjust()` only
    operates on active reservations."""
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(500), "idem-1", expires_at=1_000)
    ledger.release("intent-1", reason="cancelled")

    with pytest.raises(ValueError):
        ledger.adjust("intent-1", MoneyMicros(100))


# --- ReservationLedger: expire_due -----------------------------------------------


def test_expire_due_releases_only_reservations_whose_expiry_has_passed() -> None:
    """`expire_due` releases exactly the reservations whose `expires_at` is
    due, leaving unexpired reservations untouched."""
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(100), "idem-1", expires_at=1_000)
    ledger.reserve("intent-2", MoneyMicros(200), "idem-2", expires_at=2_000)

    ledger.expire_due(now_epoch_s=1_500)

    assert ledger.total_reserved() == MoneyMicros(200)


def test_expire_due_releases_a_reservation_exactly_at_its_expiry() -> None:
    """`now_epoch_s == expires_at` is due (inclusive boundary)."""
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(100), "idem-1", expires_at=1_000)

    ledger.expire_due(now_epoch_s=1_000)

    assert ledger.total_reserved() == MoneyMicros(0)


def test_expire_due_keeps_a_reservation_one_second_before_expiry() -> None:
    """`now_epoch_s == expires_at - 1` is not yet due."""
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve("intent-1", MoneyMicros(100), "idem-1", expires_at=1_000)

    ledger.expire_due(now_epoch_s=999)

    assert ledger.total_reserved() == MoneyMicros(100)


# --- ReservationLedger: exactly one correctly-shaped Event per mutation --------


def test_reserve_emits_exactly_one_reservation_created_event() -> None:
    """`reserve()` records exactly one `ReservationCreated` event, with
    integer (`.value`) payload fields."""
    writer = InMemoryKernelLedgerWriter()
    ledger = ReservationLedger(writer)

    reservation = ledger.reserve(
        "intent-1", MoneyMicros(500), "idem-1", expires_at=1_000
    )

    events = [
        event for event in writer.events if event.event_type == "ReservationCreated"
    ]
    assert len(events) == 1
    event = events[0]
    assert event.component == "riskkernel"
    assert event.payload_schema_version == 1
    assert event.payload["intent_id"] == "intent-1"
    assert event.payload["amount"] == 500
    assert event.payload["idempotency_key"] == "idem-1"
    assert event.payload["expires_at"] == 1_000
    assert event.payload["sequence_number"] == reservation.sequence_number


def test_release_emits_exactly_one_reservation_released_event() -> None:
    """`release()` records exactly one `ReservationReleased` event."""
    writer = InMemoryKernelLedgerWriter()
    ledger = ReservationLedger(writer)
    ledger.reserve("intent-1", MoneyMicros(500), "idem-1", expires_at=1_000)

    ledger.release("intent-1", reason="cancelled")

    events = [
        event for event in writer.events if event.event_type == "ReservationReleased"
    ]
    assert len(events) == 1
    event = events[0]
    assert event.component == "riskkernel"
    assert event.payload_schema_version == 1
    assert event.payload["intent_id"] == "intent-1"
    assert event.payload["reason"] == "cancelled"


def test_adjust_emits_exactly_one_reservation_adjusted_event() -> None:
    """`adjust()` records exactly one `ReservationAdjusted` event."""
    writer = InMemoryKernelLedgerWriter()
    ledger = ReservationLedger(writer)
    ledger.reserve("intent-1", MoneyMicros(500), "idem-1", expires_at=1_000)

    ledger.adjust("intent-1", MoneyMicros(300))

    events = [
        event for event in writer.events if event.event_type == "ReservationAdjusted"
    ]
    assert len(events) == 1
    event = events[0]
    assert event.component == "riskkernel"
    assert event.payload_schema_version == 1
    assert event.payload["intent_id"] == "intent-1"
    assert event.payload["remaining_amount"] == 300


def test_expire_due_emits_one_reservation_released_event_per_expired_reservation() -> (
    None
):
    """`expire_due` emits one `ReservationReleased` event for each
    reservation it releases -- not a single batched event."""
    writer = InMemoryKernelLedgerWriter()
    ledger = ReservationLedger(writer)
    ledger.reserve("intent-1", MoneyMicros(100), "idem-1", expires_at=1_000)
    ledger.reserve("intent-2", MoneyMicros(100), "idem-2", expires_at=1_000)

    ledger.expire_due(now_epoch_s=1_000)

    events = [
        event for event in writer.events if event.event_type == "ReservationReleased"
    ]
    assert len(events) == 2
    assert {event.payload["intent_id"] for event in events} == {"intent-1", "intent-2"}


# --- ApprovalPipeline: veto path -------------------------------------------------


def test_approval_pipeline_veto_reserves_nothing_and_issues_no_token() -> None:
    """A vetoed intent yields no reservation, no token, and consumes no
    ledger sequence number.

    The veto is sourced from a real dimension -- an unknown jurisdiction, which
    `jurisdiction_product_eligibility` fails closed on. Before issue #340 this
    test took its veto for free from that check's unconditional stub arm, so a
    fully permissive context sufficed; now such a context approves, and the
    assertions below would fail by *reserving capital and minting a token*.
    """
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    handle = SigningKeyHandle(_KEY_MATERIAL)
    issuer = TokenIssuer(handle)
    pipeline = ApprovalPipeline(ledger, issuer, config_hash="cfg-hash-1")
    intent = make_intent()
    context = make_context(jurisdiction_status=None)

    outcome = pipeline.approve(intent, context)

    assert outcome.decision.vetoed is True
    assert outcome.token is None
    assert ledger.total_reserved() == MoneyMicros(0)
    assert ledger.used_intent_ids() == frozenset()
    assert ledger.used_idempotency_keys() == frozenset()

    # No sequence number was consumed by the veto: the next successful
    # reservation still starts at 1.
    reservation = ledger.reserve(
        "intent-after-veto", MoneyMicros(1), "idem-after-veto", expires_at=9_999
    )
    assert reservation.sequence_number == 1


def test_approval_pipeline_success_issues_a_token_and_reserves_capital(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An approving pipeline reserves the cost, issues a token, and ledgers it.

    `ApprovalPipeline.approve` reserves the worst-case cost, issues a signed
    token whose claims carry the pipeline-computed `expires_at` /
    `max_fee_micros` / `kernel_sequence_number`, and records an
    `ApprovalTokenIssued` event.

    Since issue #340 promoted the last SPEC S10.3 check, a permissive context
    reaches this branch unaided. The pipeline is still stubbed here, mirroring
    `tests/riskkernel/test_process_isolation.py`'s identical technique, so this
    test pins the reservation/token-issuance contract in isolation from the
    check sequence.
    """
    approved = checks_module.Decision(vetoed=False, reasons=())
    monkeypatch.setattr(
        checks_module, "evaluate_intent", lambda intent, context: approved
    )

    writer = InMemoryKernelLedgerWriter()
    ledger = ReservationLedger(writer)
    handle = SigningKeyHandle(_KEY_MATERIAL)
    issuer = TokenIssuer(handle)
    now_epoch_s = 1_700_000_000
    ttl_seconds = 60
    pipeline = ApprovalPipeline(
        ledger, issuer, ttl_seconds=ttl_seconds, config_hash="cfg-hash-1"
    )
    intent = make_intent()
    context = make_context(
        now_epoch_s=now_epoch_s,
        max_trading_fee=MoneyMicros(300_000),
        max_settlement_fee=MoneyMicros(150_000),
    )

    outcome = pipeline.approve(intent, context)

    assert outcome.decision.vetoed is False
    assert outcome.token is not None
    claims = outcome.token.claims
    assert claims.intent_id == intent.intent_id
    assert claims.idempotency_key == intent.idempotency_key
    assert claims.config_hash == "cfg-hash-1"
    assert claims.expires_at == now_epoch_s + ttl_seconds
    assert claims.max_fee_micros == MoneyMicros(450_000)
    assert claims.kernel_sequence_number == 1

    # The reserved amount is the *full* worst-case cost (notional + both fee
    # bounds + rounding buffer), not the bare notional: notional 5_000_000 +
    # trading fee 300_000 + settlement fee 150_000 + buffer 0.
    assert ledger.total_reserved() == MoneyMicros(5_450_000)

    issued_events = [
        event for event in writer.events if event.event_type == "ApprovalTokenIssued"
    ]
    assert len(issued_events) == 1
    assert issued_events[0].component == "riskkernel"
    assert issued_events[0].payload_schema_version == 1
    assert issued_events[0].payload["intent_id"] == intent.intent_id


def test_approval_pipeline_stamps_ledger_state_onto_the_evaluated_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before evaluating checks, the pipeline stamps
    `account.pending_kernel_reservations`, `used_intent_ids`, and
    `used_idempotency_keys` from the ledger's *current* state onto a copy of
    the caller's context -- never the caller-supplied values -- so a check
    reading exposure or uniqueness always sees ledger-truth, and the
    caller's original context object is left untouched.
    """
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ledger.reserve(
        "prior-intent", MoneyMicros(1_000_000), "prior-idem", expires_at=9_999
    )

    captured_contexts: list[object] = []

    def _spy(intent: object, context: object) -> checks_module.Decision:
        captured_contexts.append(context)
        return checks_module.Decision(vetoed=False, reasons=())

    monkeypatch.setattr(checks_module, "evaluate_intent", _spy)

    handle = SigningKeyHandle(_KEY_MATERIAL)
    issuer = TokenIssuer(handle)
    pipeline = ApprovalPipeline(ledger, issuer, config_hash="cfg-hash-1")
    intent = make_intent(intent_id="new-intent", idempotency_key="new-idem")
    caller_context = make_context()

    pipeline.approve(intent, caller_context)

    assert len(captured_contexts) == 1
    effective = captured_contexts[0]
    assert effective.account.pending_kernel_reservations == MoneyMicros(1_000_000)
    assert effective.used_intent_ids == frozenset({"prior-intent"})
    assert effective.used_idempotency_keys == frozenset({"prior-idem"})
    # The caller's own context object is never mutated.
    assert caller_context.account.pending_kernel_reservations == MoneyMicros(0)
    assert caller_context.used_intent_ids == frozenset()
    assert caller_context.used_idempotency_keys == frozenset()


# --- T4: headroom admits exactly k of N under concurrent contention -------------


def test_approval_pipeline_admits_exactly_k_of_n_under_headroom_contention() -> None:
    """Launch N=8 threads through `ApprovalPipeline.approve`, released at
    once by a `threading.Barrier`, against a headroom sized for exactly 3
    successes (`equity - floor == 3 * cost`, floor-only checks active).
    Regardless of thread interleaving, exactly 3 tokens are issued,
    `total_reserved()` never exceeds the headroom, and the remaining 5 are
    vetoed citing the floor -- the ledger's single lock makes the outcome
    deterministic no matter which thread wins each race.
    """
    cost_micros = _DEFAULT_INTENT_COST_MICROS
    headroom_micros = 3 * cost_micros
    floor_micros = _DEFAULT_EQUITY_MICROS - headroom_micros

    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    handle = SigningKeyHandle(_KEY_MATERIAL)
    issuer = TokenIssuer(handle)
    pipeline = ApprovalPipeline(ledger, issuer, config_hash="cfg-hash-1")

    # Isolate the floor invariant from every other SPEC S10.3 check, so an
    # unrelated veto cannot mask the headroom behavior this test targets.
    original_evaluate_intent = checks_module.evaluate_intent
    floor_only_checks = tuple(
        check
        for check in checks_module.DEFAULT_CHECKS
        if check.name == "floor_invariant"
    )

    def _floor_only_evaluate(intent: object, context: object) -> checks_module.Decision:
        return original_evaluate_intent(intent, context, checks=floor_only_checks)

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    outcomes: list[object | None] = [None] * n_threads
    errors: list[Exception] = []

    def _worker(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            intent = make_intent(
                intent_id=f"intent-{index}", idempotency_key=f"idem-{index}"
            )
            context = make_context(
                floor=MoneyMicros(floor_micros),
                exchange_verified_available_cash=MoneyMicros(_DEFAULT_EQUITY_MICROS),
            )
            outcomes[index] = pipeline.approve(intent, context)
        except Exception as exc:
            # Surfaced to the main thread (asserted on below) rather than
            # letting a worker-thread exception vanish silently.
            errors.append(exc)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(checks_module, "evaluate_intent", _floor_only_evaluate)

        threads = [
            threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads), "a worker thread hung"
    assert not errors, errors

    approved = [
        outcome for outcome in outcomes if outcome is not None and outcome.token
    ]
    vetoed = [
        outcome for outcome in outcomes if outcome is not None and outcome.token is None
    ]

    assert len(approved) == 3
    assert len(vetoed) == 5
    assert all(
        any("floor" in reason for reason in outcome.decision.reasons)
        for outcome in vetoed
    )
    assert ledger.total_reserved() == MoneyMicros(headroom_micros)
    assert ledger.total_reserved() <= MoneyMicros(headroom_micros)
