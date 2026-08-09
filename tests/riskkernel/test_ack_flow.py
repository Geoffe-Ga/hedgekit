"""Failing-first tests for the human-ack orchestration coordinator (issue #57,
RED).

Issue #57's plan flagged that the Risk Kernel has real ``human_ack_satisfied``
check logic (issue #34) and a real ``HumanAckQueue`` (issue #34), but nothing
*orchestrates* them: nothing turns an over-threshold veto into a HELD state
that later resubmits once an operator grants it, and nothing watches a
filesystem drop-box for an operator's grant. This file specifies that missing
piece: a new module, ``windbreak/riskkernel/ack_flow.py``, which does not
exist yet, so the import below fails collection with
``ModuleNotFoundError: No module named 'windbreak.riskkernel.ack_flow'`` --
the expected Gate 1 RED state for issue #57.

Proposed public shape (the implementation specialist must build to this
exactly, or confirm/rename via the handoff):

* ``AckGatedApprovalPipeline(pipeline: ApprovalPipeline, ack_queue:
  HumanAckQueue)`` -- wraps an existing, already-built
  ``ApprovalPipeline``/``HumanAckQueue`` pair (composition, not
  reimplementation).

  * ``.submit(intent, context) -> AckGatedOutcome`` -- stamps the queue's
    currently-granted intent ids onto a copy of ``context`` (mirroring
    ``ApprovalPipeline._effective_context``'s ledger-stamping precedent),
    runs the wrapped pipeline, and:

    - no veto -> ``AckGatedOutcome(token=..., decision=..., held=False,
      pending_ack=None)``;
    - vetoed with *exactly* the one reason
      ``"human acknowledgement required"`` (``_HumanAckSatisfied``'s exact
      string) -> opens a ``PendingHumanAck`` via ``ack_queue.request_ack``
      and returns ``AckGatedOutcome(token=None, decision=..., held=True,
      pending_ack=...)``;
    - vetoed for that reason *plus* any other -> a plain veto, fail-closed:
      ``held=False``, ``pending_ack=None``, no ack requested.

  * ``.expire_due(now) -> None`` -- delegates to ``ack_queue.expire_due``.

  Judgment call (flagged for implementation to confirm): ``submit`` has no
  clock of its own, so ``ack_queue.request_ack``'s ``now`` argument is
  ``context.now_epoch_s`` -- the same instant the check pipeline itself
  reads. Every test below that computes an expiry boundary
  (``context.now_epoch_s + ack_ttl_seconds``) relies on this exact wiring.

* ``AckGatedOutcome`` -- a frozen dataclass: ``token: SignedApprovalToken |
  None``, ``decision: Decision``, ``held: bool``, ``pending_ack:
  PendingHumanAck | None``.

* ``AckFileWatcher(queue: HumanAckQueue, state_dir: Path)`` -- mirrors
  ``windbreak.riskkernel.kill.KillFileWatcher``'s poll-once, always-consume
  shape, but over a directory of files rather than two fixed names:
  ``.poll_once(now) -> None`` grants every pending approval named by a file
  under ``<state_dir>/acks/`` and always removes the file afterward,
  whether the grant succeeded, the id was never issued
  (``UnknownApprovalError``), or it had already lapsed (``AckLapsedError``)
  -- fail-closed, never raising.

This file also proves the consequence on the Gateway side: an intent HELD
by ``AckGatedApprovalPipeline`` carries no approval token at all, and even a
token fabricated for it (signed under a key that is not the Risk Kernel's
real signing key, since no real approval was ever granted) is rejected by
``windbreak.order_gateway.tokens.verify_and_consume`` before the injected
``OrderSubmitter`` is ever called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.riskkernel.conftest import make_context, make_intent
from windbreak.numeric.types import MoneyMicros
from windbreak.riskkernel import checks as checks_module
from windbreak.riskkernel.ack_flow import (
    AckFileWatcher,
    AckGatedApprovalPipeline,
)
from windbreak.riskkernel.floor import worst_case_cost
from windbreak.riskkernel.human_ack import AckLapsedError, HumanAckQueue
from windbreak.riskkernel.process import InMemoryKernelLedgerWriter
from windbreak.riskkernel.reservations import ApprovalPipeline, ReservationLedger
from windbreak.riskkernel.signing import SigningKeyHandle
from windbreak.riskkernel.tokens import TokenIssuer
from windbreak.tokens.verify import ApprovalTokenClaims

if TYPE_CHECKING:
    from pathlib import Path

#: A fixed, valid (>=32-byte) signing key every real ``ApprovalPipeline``
#: below issues tokens under.
_KEY_MATERIAL = b"k" * 32

#: A deliberately different, equally-fake key: never the real Gateway
#: verification key, simulating "no legitimate approval was ever granted".
_REAL_GATEWAY_KEY = b"g" * 32
_FORGER_KEY = b"f" * 32  # pragma: allowlist secret

#: The exact `_HumanAckSatisfied` veto reason (windbreak/riskkernel/checks.py).
_HUMAN_ACK_VETO_REASON = "human acknowledgement required"

#: An obviously-fake, low-entropy 32-hex-char id, never a real approval id.
_UNKNOWN_APPROVAL_ID = "aa" * 16


def _isolate_checks(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Patch ``evaluate_intent`` to run only the named SPEC S10.3 check(s).

    Mirrors ``tests/riskkernel/test_reservations.py``'s T4 isolation
    technique and ``tests/riskkernel/test_micro_cap_properties.py``'s copy of
    it: 1 of the 24 SPEC S10.3 checks is still an unconditional-veto stub,
    which would otherwise mask every behavior this file targets.

    Args:
        monkeypatch: The active monkeypatch fixture.
        *names: The ``Check.name`` value(s) to keep, in SPEC S10.3 order.
    """
    original_evaluate_intent = checks_module.evaluate_intent
    kept = tuple(check for check in checks_module.DEFAULT_CHECKS if check.name in names)

    def _kept_only_evaluate(intent: object, context: object) -> checks_module.Decision:
        return original_evaluate_intent(intent, context, checks=kept)

    monkeypatch.setattr(checks_module, "evaluate_intent", _kept_only_evaluate)


def _build_coordinator(
    *, ack_ttl_seconds: int = 3_600
) -> tuple[
    ReservationLedger,
    HumanAckQueue,
    InMemoryKernelLedgerWriter,
    AckGatedApprovalPipeline,
]:
    """Build a fresh ledger/pipeline/queue/coordinator quartet for one test.

    Args:
        ack_ttl_seconds: The human-ack queue's operator-response window.

    Returns:
        ``(ledger, ack_queue, ack_writer, coordinator)``, where ``ack_writer``
        is the queue's own (separate) event log, so a test can assert on
        acknowledgement events without wading through reservation events.
    """
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    issuer = TokenIssuer(SigningKeyHandle(_KEY_MATERIAL))
    pipeline = ApprovalPipeline(ledger, issuer, config_hash="cfg-hash-ack-flow")
    ack_writer = InMemoryKernelLedgerWriter()
    ack_queue = HumanAckQueue(
        writer=ack_writer, releaser=ledger, ttl_seconds=ack_ttl_seconds
    )
    coordinator = AckGatedApprovalPipeline(pipeline, ack_queue)
    return ledger, ack_queue, ack_writer, coordinator


def _expected_cost() -> MoneyMicros:
    """Return `make_intent()`'s worst-case cost under `make_context()`'s
    permissive defaults (zero fees, zero rounding buffer): 5000 pips *
    1000 centis == 5,000,000 micros -- the same constant documented in
    `tests/riskkernel/test_reservations.py`.
    """
    intent = make_intent()
    return worst_case_cost(
        intent.price,
        intent.size,
        max_trading_fee=MoneyMicros(0),
        max_settlement_fee=MoneyMicros(0),
        rounding_buffer=MoneyMicros(0),
    )


# --- submit(): the only-reason-is-human-ack HELD path ---------------------------


def test_over_threshold_intent_with_only_human_ack_veto_is_held_with_no_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-threshold LIVE intent whose *only* veto is
    `human_ack_satisfied` is HELD, not plainly vetoed: `submit` returns no
    token, `held` is True, a `PendingHumanAck` is opened whose
    `worst_case_cost` matches the SPEC S10.4 cost exactly, a
    `HumanAckRequested` event is ledgered, and -- because
    `human_ack_satisfied` vetoes *before* any capital is reserved -- nothing
    is reserved against the intent yet.
    """
    _isolate_checks(monkeypatch, "human_ack_satisfied")
    ledger, _queue, ack_writer, coordinator = _build_coordinator()
    context = make_context(require_human_ack_above_micros=MoneyMicros(1_000_000))
    intent = make_intent()

    outcome = coordinator.submit(intent, context)

    assert outcome.token is None
    assert outcome.held is True
    assert outcome.pending_ack is not None
    assert outcome.pending_ack.intent_id == intent.intent_id
    assert outcome.pending_ack.worst_case_cost == _expected_cost()
    requested = [
        event for event in ack_writer.events if event.event_type == "HumanAckRequested"
    ]
    assert len(requested) == 1
    assert requested[0].payload["intent_id"] == intent.intent_id
    assert ledger.total_reserved() == MoneyMicros(0)


def test_after_grant_resubmitting_the_same_intent_issues_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once an operator grants the pending acknowledgement, resubmitting the
    identical intent runs the full pipeline -- now stamped with the queue's
    acknowledged intent ids -- and issues a token.
    """
    _isolate_checks(monkeypatch, "human_ack_satisfied")
    _ledger, ack_queue, _ack_writer, coordinator = _build_coordinator()
    context = make_context(require_human_ack_above_micros=MoneyMicros(1_000_000))
    intent = make_intent()

    held_outcome = coordinator.submit(intent, context)
    assert held_outcome.held is True
    assert held_outcome.pending_ack is not None

    ack_queue.grant(
        approval_id=held_outcome.pending_ack.approval_id, now=context.now_epoch_s
    )
    granted_outcome = coordinator.submit(intent, context)

    assert granted_outcome.token is not None
    assert granted_outcome.held is False
    assert granted_outcome.decision.vetoed is False


def test_resubmitting_while_the_ack_is_still_pending_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resubmitting the same intent before its acknowledgement is granted or
    lapsed re-returns the *existing* pending request rather than opening a
    second one: `held` stays True, the `approval_id` is unchanged, and exactly
    one `HumanAckRequested` event is ever ledgered.
    """
    _isolate_checks(monkeypatch, "human_ack_satisfied")
    _ledger, _ack_queue, ack_writer, coordinator = _build_coordinator()
    context = make_context(require_human_ack_above_micros=MoneyMicros(1_000_000))
    intent = make_intent()

    first = coordinator.submit(intent, context)
    second = coordinator.submit(intent, context)

    assert first.pending_ack is not None
    assert second.pending_ack is not None
    assert second.held is True
    assert second.token is None
    assert second.pending_ack.approval_id == first.pending_ack.approval_id
    requested = [
        event for event in ack_writer.events if event.event_type == "HumanAckRequested"
    ]
    assert len(requested) == 1


def test_multi_reason_veto_is_not_held_and_requests_no_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A veto carrying `human_ack_satisfied`'s reason *plus* another failing
    check (a stale/missing quote) is a plain veto, fail-closed: `held` is
    False, no `PendingHumanAck` is opened, and no `HumanAckRequested` event
    is ledgered -- an operator is never asked to bless an intent that is
    independently broken for another reason.
    """
    _isolate_checks(monkeypatch, "human_ack_satisfied", "quote_freshness")
    _ledger, _queue, ack_writer, coordinator = _build_coordinator()
    context = make_context(
        require_human_ack_above_micros=MoneyMicros(1_000_000),
        quote_snapshot_epoch_s=None,
    )
    intent = make_intent()

    outcome = coordinator.submit(intent, context)

    assert outcome.token is None
    assert outcome.held is False
    assert outcome.pending_ack is None
    assert len(outcome.decision.reasons) == 2
    assert _HUMAN_ACK_VETO_REASON in outcome.decision.reasons
    assert [
        event for event in ack_writer.events if event.event_type == "HumanAckRequested"
    ] == []


# --- expire_due(): the inclusive-boundary lapse --------------------------------


def test_expire_due_lapses_the_pending_ack_and_a_later_grant_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`AckGatedApprovalPipeline.expire_due` lapses a still-pending
    acknowledgement at its inclusive ttl boundary: a `HumanAckLapsed` event
    is ledgered, and a later `grant` for that same approval id raises
    `AckLapsedError`.
    """
    _isolate_checks(monkeypatch, "human_ack_satisfied")
    ack_ttl_seconds = 100
    _ledger, ack_queue, ack_writer, coordinator = _build_coordinator(
        ack_ttl_seconds=ack_ttl_seconds
    )
    context = make_context(require_human_ack_above_micros=MoneyMicros(1_000_000))
    intent = make_intent()
    held_outcome = coordinator.submit(intent, context)
    assert held_outcome.pending_ack is not None
    approval_id = held_outcome.pending_ack.approval_id
    lapse_at = context.now_epoch_s + ack_ttl_seconds

    coordinator.expire_due(lapse_at)

    lapsed = [
        event for event in ack_writer.events if event.event_type == "HumanAckLapsed"
    ]
    assert len(lapsed) == 1
    assert lapsed[0].payload["approval_id"] == approval_id
    with pytest.raises(AckLapsedError):
        ack_queue.grant(approval_id=approval_id, now=lapse_at + 1)


# --- Gateway refusal proof: a HELD intent never reaches the Gateway ------------


def test_a_held_intents_forged_token_is_rejected_by_the_gateway_and_never_submitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A HELD intent (proven above) carries no legitimately-issued approval
    token at all. Fabricating one anyway -- signed under a key that is not
    the Risk Kernel's real signing key, since no real approval was ever
    granted -- is still rejected by
    `windbreak.order_gateway.tokens.verify_and_consume` as
    `SubmitOutcome.REJECTED_TOKEN`, and the injected `OrderSubmitter` is
    never invoked. This composes issue #57's ack-gating with the existing
    Gateway verification boundary (issues #31/#37) to show the two halves
    close the loop end-to-end: an unacked order can never reach the
    exchange.
    """
    from windbreak.order_gateway.gateway import OrderGateway, SubmitOutcome

    _isolate_checks(monkeypatch, "human_ack_satisfied")
    _ledger, _queue, _ack_writer, coordinator = _build_coordinator()
    context = make_context(require_human_ack_above_micros=MoneyMicros(1_000_000))
    intent = make_intent()

    held_outcome = coordinator.submit(intent, context)
    assert held_outcome.token is None, "fixture assumption: the intent is HELD"

    forged_claims = ApprovalTokenClaims(
        intent_id=intent.intent_id,
        market_ticker=intent.market_ticker,
        outcome=intent.outcome,
        action=intent.action,
        limit_price_pips=intent.price,
        count_centis=intent.size,
        max_fee_micros=MoneyMicros(0),
        expires_at=context.now_epoch_s + 3_600,
        idempotency_key=intent.idempotency_key,
        config_hash="cfg-hash-forged",
        kernel_sequence_number=1,
    )
    forged_token = TokenIssuer(SigningKeyHandle(_FORGER_KEY)).issue(forged_claims)

    class _SpySubmitter:
        """An `OrderSubmitter` that fails the test if ever called."""

        def submit(self, intent: object, token: object) -> object:
            """Fail the test: the Gateway must never call this."""
            del intent, token
            raise AssertionError(
                "an unacked (HELD) intent's forged token must never reach the submitter"
            )

    gateway = OrderGateway(
        _SpySubmitter(),
        verification_key=_REAL_GATEWAY_KEY,
        clock=lambda: context.now_epoch_s,
    )

    result = gateway.process_intent(intent, forged_token)

    assert result.outcome is SubmitOutcome.REJECTED_TOKEN
    assert result.ack is None


# --- AckFileWatcher: presence-driven grant, always-consume ----------------------


def test_ack_file_watcher_grants_the_named_pending_approval_and_removes_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file at `<state_dir>/acks/<approval_id>` grants that pending
    acknowledgement and is removed on `poll_once`.
    """
    _isolate_checks(monkeypatch, "human_ack_satisfied")
    _ledger, ack_queue, _ack_writer, coordinator = _build_coordinator()
    context = make_context(require_human_ack_above_micros=MoneyMicros(1_000_000))
    intent = make_intent()
    held_outcome = coordinator.submit(intent, context)
    assert held_outcome.pending_ack is not None
    approval_id = held_outcome.pending_ack.approval_id
    acks_dir = tmp_path / "acks"
    acks_dir.mkdir()
    (acks_dir / approval_id).write_text("", encoding="utf-8")
    watcher = AckFileWatcher(ack_queue, tmp_path)

    watcher.poll_once(context.now_epoch_s)

    assert intent.intent_id in ack_queue.acknowledged_intent_ids(context.now_epoch_s)
    assert not (acks_dir / approval_id).exists()


def test_ack_file_watcher_unknown_id_grants_nothing_and_is_consumed_without_raising(
    tmp_path: Path,
) -> None:
    """A file naming an approval id that was never issued is removed and
    grants nothing -- fail-closed, never a crash.
    """
    ledger = ReservationLedger(InMemoryKernelLedgerWriter())
    ack_queue = HumanAckQueue(writer=InMemoryKernelLedgerWriter(), releaser=ledger)
    acks_dir = tmp_path / "acks"
    acks_dir.mkdir()
    (acks_dir / _UNKNOWN_APPROVAL_ID).write_text("", encoding="utf-8")
    watcher = AckFileWatcher(ack_queue, tmp_path)

    watcher.poll_once(0)

    assert not (acks_dir / _UNKNOWN_APPROVAL_ID).exists()
    assert ack_queue.acknowledged_intent_ids(0) == frozenset()


def test_ack_file_watcher_lapsed_id_grants_nothing_and_is_consumed_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file naming an approval id that has already lapsed is removed and
    grants nothing -- the pending request is gone, not resurrected.
    """
    _isolate_checks(monkeypatch, "human_ack_satisfied")
    ack_ttl_seconds = 10
    _ledger, ack_queue, _ack_writer, coordinator = _build_coordinator(
        ack_ttl_seconds=ack_ttl_seconds
    )
    context = make_context(require_human_ack_above_micros=MoneyMicros(1_000_000))
    intent = make_intent()
    held_outcome = coordinator.submit(intent, context)
    assert held_outcome.pending_ack is not None
    approval_id = held_outcome.pending_ack.approval_id
    lapse_at = context.now_epoch_s + ack_ttl_seconds
    ack_queue.expire_due(lapse_at)
    acks_dir = tmp_path / "acks"
    acks_dir.mkdir()
    (acks_dir / approval_id).write_text("", encoding="utf-8")
    watcher = AckFileWatcher(ack_queue, tmp_path)

    watcher.poll_once(lapse_at)

    assert not (acks_dir / approval_id).exists()
    assert intent.intent_id not in ack_queue.acknowledged_intent_ids(lapse_at)


def test_ack_file_watcher_is_a_noop_when_the_drop_box_dir_does_not_exist(
    tmp_path: Path,
) -> None:
    """`poll_once` returns quietly before any `ack` file has ever been written.

    The `<state_dir>/acks/` directory only exists once the CLI or dashboard has
    dropped a grant, so a beat that polls before then must fail closed as a
    no-op rather than raise.
    """
    _ledger, ack_queue, _ack_writer, _coordinator = _build_coordinator()
    watcher = AckFileWatcher(ack_queue, tmp_path)

    watcher.poll_once(0)

    assert not (tmp_path / "acks").exists()


def test_ack_file_watcher_ignores_a_non_file_entry_in_the_drop_box(
    tmp_path: Path,
) -> None:
    """A non-file entry (e.g. a subdirectory) in the drop-box is left untouched.

    Only files name approval ids; `poll_once` skips anything that is not a file
    rather than treating its name as an id or attempting to unlink it.
    """
    _ledger, ack_queue, _ack_writer, _coordinator = _build_coordinator()
    acks_dir = tmp_path / "acks"
    acks_dir.mkdir()
    stray_subdir = acks_dir / "not-an-approval-file"
    stray_subdir.mkdir()
    watcher = AckFileWatcher(ack_queue, tmp_path)

    watcher.poll_once(0)

    assert stray_subdir.is_dir()
