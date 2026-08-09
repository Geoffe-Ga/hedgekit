"""Failing-first tests for crash recovery (issue #40, RED).

`windbreak/order_gateway/wal.py` and `windbreak/order_gateway/recovery.py` do not
exist yet, so the module-level imports below fail collection with
`ModuleNotFoundError: No module named 'windbreak.order_gateway.wal'` (or
`.recovery`) -- the expected Gate 1 RED state for issue #40. `OrderGateway`
also does not yet accept the `wal`/`ledger_reader`/`reconciliation_source`
constructor keywords, expose `.recover()`/`.accepting_approvals`/`.halted`, or
export `SubmitOutcome.REFUSED_RECOVERY_PENDING`; once the module-level import
above is satisfied, those gaps surface as `TypeError`/`AttributeError` instead.

This module pins the crash-recovery contract:

    * A write-ahead log (`WriteAheadLog`) durably journals an intent *before*
      the Gateway proceeds, and the just-placed order's ack *before* the
      Gateway ledgers `SUBMIT`/`ACK` -- so a crash at any of the seven durable
      write points along `APPROVE -> REQUEST_SUBMISSION -> (place) ->
      SUBMIT -> ACK` (WAL-intent, APPROVE, REQUEST_SUBMISSION, exchange
      placement, WAL-ack, SUBMIT, ACK) leaves the paper exchange with never
      more than one order for that attempt, and a fresh `OrderGateway`'s
      `.recover()` over the same durable ledger/WAL/exchange always leaves the
      per-`client_order_id` ledgered transition history a *legal* state-machine
      chain (never corrupt), and the chain's hash integrity intact.
    * Wiring any of `wal`/`ledger_reader`/`reconciliation_source` gates
      `accepting_approvals` to `False` from construction until `.recover()`
      completes; a brand-new intent presented before that point is refused
      `REFUSED_RECOVERY_PENDING` (ledgering `SubmissionRefused(reason=
      "recovery_pending")`) *without* consuming its token -- the identical
      token still ACKs once `.recover()` has run.
    * An open order on the exchange with zero corroborating WAL/ledger trace
      (a `"foreign_open_order"`) halts recovery and the Gateway fail-closed:
      `.halted` and `.accepting_approvals is False` forever after, and every
      subsequent `process_intent` raises `GatewayHaltedError`. Likewise an
      exchange placement whose completing WAL-ack was never durably written
      (the crash lands strictly between the exchange accepting the order and
      the Gateway journaling its ack) is undecidable from the paper
      exchange's bare `OpenOrder` (which carries no `client_order_id`) and
      halts with reason `"ambiguous_match"` rather than guess.
    * A prior in-process `ReduceOnlyViolation` halt (issue #39) is a durable
      ledger fact: a fresh, restarted Gateway's `.recover()` folds it and
      stays halted -- there is no un-halt event.
    * The reduce-only in-flight-closing tally (issue #39) is rebuilt from the
      ledger/WAL on `.recover()`: a still-resting close's size keeps shrinking
      the closeable remainder across a restart, while a *fully settled* close
      (no resting remainder left) is retired, returning its full headroom.
    * `.recover()` never leaves `accepting_approvals` `True` while a halt
      condition stands (SPEC S11.4 ordering).
    * A journalled intent whose payload no longer re-derives the
      `client_order_id` it was stored under is *refused*, never reconciled
      under either id: `read_all()` raises `ValueError` naming both ids by
      role, one corrupt record poisons the whole read (no partial recovery),
      and a Gateway whose `.recover()` hits one stays shut -- re-presenting
      the same intent is refused `REFUSED_RECOVERY_PENDING` rather than
      submitted a second time (issue #252).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from tests.order_gateway.conftest import (
    DEFAULT_MARKET_TICKER,
    DEFAULT_NOW_EPOCH_S,
    KEY_MATERIAL,
    issue_matching_token,
    make_intent,
)
from tests.order_gateway.test_reduce_only import (
    _ForcedFillSubmitter,
    _position,
    _StubPositionSource,
)
from windbreak.connector.paper import PaperOrderIntent
from windbreak.ledger.events import canonical_json
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.numeric.types import ContractCentis, PricePips
from windbreak.order_gateway.client_order_id import client_order_id
from windbreak.order_gateway.gateway import (
    GatewayHaltedError,
    OrderGateway,
    PaperSubmitter,
    SubmitOutcome,
)
from windbreak.order_gateway.ledger_writer import SqliteGatewayLedgerWriter
from windbreak.order_gateway.recovery import RecoveryReport
from windbreak.order_gateway.state_machine import OrderEvent, OrderState, transition
from windbreak.order_gateway.wal import WriteAheadLog

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from windbreak.connector.paper import PaperExchange
    from windbreak.ledger.events import Event
    from windbreak.ledger.store import LedgerRecord
    from windbreak.order_gateway.gateway import (
        GatewayPositionSource,
        OrderSubmitter,
        SubmissionAck,
    )
    from windbreak.order_gateway.ledger_writer import GatewayLedgerWriter
    from windbreak.order_gateway.wal import WalRecord
    from windbreak.riskkernel.checks import OrderIntent
    from windbreak.tokens.verify import SignedApprovalToken


def _build_recovering_gateway(
    exchange: PaperExchange,
    store: SqliteLedgerStore,
    wal: WriteAheadLog,
    *,
    position_source: GatewayPositionSource | None = None,
    submitter: OrderSubmitter | None = None,
) -> OrderGateway:
    """Build an `OrderGateway` wired for the issue #40 recovery surface.

    Args:
        exchange: The paper exchange the Gateway submits through and
            reconciles against (`reconciliation_source`).
        store: The SQLite-backed ledger, used as both `ledger_writer`
            (wrapped in `SqliteGatewayLedgerWriter`) and `ledger_reader`.
        wal: The write-ahead log the Gateway durably journals through.
        position_source: Optional reduce-only position source (issue #39).
        submitter: Optional override submitter; defaults to a real
            `PaperSubmitter` over `exchange`.

    Returns:
        A fully wired `OrderGateway`, not yet recovered (`accepting_approvals`
        is `False` until `.recover()` runs).
    """
    return OrderGateway(
        submitter if submitter is not None else PaperSubmitter(exchange),
        verification_key=KEY_MATERIAL,
        clock=lambda: DEFAULT_NOW_EPOCH_S,
        ledger_writer=SqliteGatewayLedgerWriter(store),
        wal=wal,
        ledger_reader=store,
        reconciliation_source=exchange,
        position_source=position_source,
    )


def _assert_ledger_transitions_form_a_legal_chain_per_coid(
    records: list[LedgerRecord],
) -> None:
    """Replay every `OrderTransitionLedgered` payload, grouped by coid.

    Folds each `client_order_id`'s ledgered `(from_state, event)` pairs, in
    sequence order, through the real `transition()` function: an illegal move
    raises `IllegalTransitionError` (failing the test), and a ledgered
    `to_state` that disagrees with what `transition()` computes fails an
    explicit assertion. This is a recovery-agnostic consistency check: it
    holds regardless of whether `.recover()` completes a stalled coid's
    missing transitions or leaves them as-is.

    Args:
        records: Every ledger record to scan (only `OrderTransitionLedgered`
            rows are folded; everything else is ignored).
    """
    by_coid: dict[str, list[dict[str, object]]] = {}
    for record in records:
        if record.event_type != "OrderTransitionLedgered":
            continue
        data = json.loads(record.payload_json)["data"]
        by_coid.setdefault(str(data["client_order_id"]), []).append(data)
    for transitions in by_coid.values():
        state = OrderState.INTENT_CREATED
        for data in transitions:
            assert data["from_state"] == state.name
            event = OrderEvent[str(data["event"])]
            state = transition(state, event)
            assert data["to_state"] == state.name


class SimulatedCrashError(Exception):
    """Raised by a test-local wrapper seam to simulate a mid-submission crash."""


class _KillSwitch:
    """Shared counter raising `SimulatedCrashError` on its Nth `tick()`.

    Threaded into every crash-simulating wrapper below so one shared count
    spans the WAL, ledger writer, and submitter seams: their combined call
    order for a single intent is exactly the seven durable-write points the
    kill matrix parametrizes over (WAL-intent, APPROVE, REQUEST_SUBMISSION,
    exchange placement, WAL-ack, SUBMIT, ACK).
    """

    def __init__(self, kill_after: int) -> None:
        """Initialize, tracking ticks against the configured kill point.

        Args:
            kill_after: The 1-based tick count that raises.
        """
        self._kill_after = kill_after
        self._count = 0
        self._armed = True

    def tick(self, label: str) -> None:
        """Record one durable write, raising if it is the configured Nth.

        Args:
            label: A short description of the seam that just wrote durably,
                folded into the raised error for diagnosability.

        Raises:
            SimulatedCrashError: On exactly the `kill_after`-th tick (once armed).
        """
        self._count += 1
        if self._armed and self._count == self._kill_after:
            raise SimulatedCrashError(
                f"simulated crash immediately after {label} "
                f"(durable write #{self._count})"
            )

    def disarm(self) -> None:
        """Suspend crashing (ticks still count) around the clean boot recovery.

        A clean boot `.recover()` legitimately ledgers one `RecoveryCompleted`
        checkpoint of its own, which is not one of the seven per-intent write
        points the matrix parametrizes over; disarming keeps that write from
        ever standing in for the `kill_after`-th per-intent write.
        """
        self._armed = False

    def rearm(self) -> None:
        """Re-arm crashing and reset the counter, isolating the single intent."""
        self._armed = True
        self._count = 0


class _CrashingWal:
    """A `WriteAheadLog`-shaped wrapper ticking a shared `_KillSwitch`."""

    def __init__(self, inner: WriteAheadLog, kill_switch: _KillSwitch) -> None:
        """Bind the wrapper to the real WAL and the shared kill switch.

        Args:
            inner: The real `WriteAheadLog` every call delegates to first.
            kill_switch: The shared counter ticked after each durable append.
        """
        self._inner = inner
        self._kill_switch = kill_switch

    def append_intent(self, intent: OrderIntent, client_order_id_: str) -> None:
        """Durably append the intent, then tick the shared kill switch.

        Args:
            intent: The order intent to journal.
            client_order_id_: The intent's content-addressed id.
        """
        self._inner.append_intent(intent, client_order_id_)
        self._kill_switch.tick("wal_intent")

    def append_ack(
        self, client_order_id_: str, order_id: str | None, filled: ContractCentis
    ) -> None:
        """Durably append the ack, then tick the shared kill switch.

        Args:
            client_order_id_: The intent's content-addressed id.
            order_id: The venue's resting-order id, or `None`.
            filled: The quantity filled immediately, in contract-centis.
        """
        self._inner.append_ack(client_order_id_, order_id, filled)
        self._kill_switch.tick("wal_ack")

    def read_all(self) -> tuple[WalRecord, ...]:
        """Delegate straight to the real WAL's `read_all()`.

        Returns:
            Whatever the real `WriteAheadLog.read_all()` returns.
        """
        return self._inner.read_all()


class _CrashingLedgerWriter:
    """A `GatewayLedgerWriter`-shaped wrapper ticking a shared `_KillSwitch`."""

    def __init__(self, inner: GatewayLedgerWriter, kill_switch: _KillSwitch) -> None:
        """Bind the wrapper to the real ledger writer and the kill switch.

        Args:
            inner: The real `GatewayLedgerWriter` every call delegates to.
            kill_switch: The shared counter ticked after each durable write.
        """
        self._inner = inner
        self._kill_switch = kill_switch

    def record(self, event: Event) -> None:
        """Durably record `event`, then tick the shared kill switch.

        Args:
            event: The ledger event to record.
        """
        self._inner.record(event)
        self._kill_switch.tick(f"ledger_{event.event_type}")


class _CrashingSubmitter:
    """An `OrderSubmitter`-shaped wrapper ticking a shared `_KillSwitch`."""

    def __init__(self, inner: OrderSubmitter, kill_switch: _KillSwitch) -> None:
        """Bind the wrapper to the real submitter and the kill switch.

        Args:
            inner: The real `OrderSubmitter` every call delegates to first.
            kill_switch: The shared counter ticked after each placement.
        """
        self._inner = inner
        self._kill_switch = kill_switch

    def submit(self, intent: OrderIntent, token: SignedApprovalToken) -> SubmissionAck:
        """Durably place the order, then tick the shared kill switch.

        Args:
            intent: The verified order intent to submit.
            token: The approval token that authorized it.

        Returns:
            The real submitter's `SubmissionAck`.
        """
        ack = self._inner.submit(intent, token)
        self._kill_switch.tick("submit")
        return ack


# --- 1. Kill-at-every-edge matrix ---------------------------------------------

#: (kill_after, id, expected `gateway.halted` after a fresh `.recover()`).
#: Every point through "after WAL-ack" (5) onward has a durable, completing
#: WAL-ack record explaining any exchange side effect, so recovery resumes
#: cleanly. Exactly one point -- "after exchange place, pre-WAL-ack" (4) --
#: leaves an exchange order with zero corroborating durable record (the
#: paper exchange's `OpenOrder` carries no `client_order_id` to correlate
#: against), which recovery cannot safely resolve and must halt
#: `"ambiguous_match"` rather than guess.
_KILL_MATRIX = (
    (1, "after-wal-intent", False),
    (2, "after-approve-ledger-write", False),
    (3, "after-request-submission-write", False),
    (4, "after-exchange-place-pre-wal-ack", True),
    (5, "after-wal-ack", False),
    (6, "after-submit-write", False),
    (7, "after-ack-write", False),
)


@pytest.mark.parametrize(
    "kill_after,expected_halted",
    [(point[0], point[2]) for point in _KILL_MATRIX],
    ids=[point[1] for point in _KILL_MATRIX],
)
def test_recover_after_crash_at_every_durable_write_point(
    tmp_path: Path,
    paper_exchange: PaperExchange,
    kill_after: int,
    expected_halted: bool,
) -> None:
    """A crash at any of the seven durable write points recovers consistently.

    A fresh `OrderGateway` over the pre-crash `.recover()`s cleanly (nothing
    to reconcile yet), then a brand-new intent is driven through
    `process_intent` under crash-simulating wrappers configured to raise
    immediately after the `kill_after`-th durable write. A second, fresh
    `OrderGateway` over the *same* SQLite ledger, WAL file, and paper
    exchange then `.recover()`s: the exchange never ends up with more than
    one order for the attempt, the ledger's hash chain still verifies, every
    coid's ledgered transition history is a legal state-machine chain, and
    `.halted` matches this kill point's expected resolution.
    """
    db_path = tmp_path / "ledger.db"
    wal_path = tmp_path / "wal.jsonl"
    store = SqliteLedgerStore(db_path)
    wal = WriteAheadLog(wal_path)
    kill_switch = _KillSwitch(kill_after)
    crashing_wal = _CrashingWal(wal, kill_switch)
    crashing_writer = _CrashingLedgerWriter(
        SqliteGatewayLedgerWriter(store), kill_switch
    )
    crashing_submitter = _CrashingSubmitter(PaperSubmitter(paper_exchange), kill_switch)
    gateway = OrderGateway(
        crashing_submitter,
        verification_key=KEY_MATERIAL,
        clock=lambda: DEFAULT_NOW_EPOCH_S,
        ledger_writer=crashing_writer,
        wal=crashing_wal,
        ledger_reader=store,
        reconciliation_source=paper_exchange,
    )
    # A clean boot recovery ledgers its own RecoveryCompleted checkpoint through
    # the crashing writer; disarm around it so the seven parametrized kill points
    # are exactly the single intent's durable writes, not the boot's bookkeeping
    # write.
    kill_switch.disarm()
    boot_report = gateway.recover()
    assert boot_report.halted is False
    assert gateway.accepting_approvals is True
    kill_switch.rearm()

    intent = make_intent(idempotency_key=f"idem-kill-{kill_after}")
    token = issue_matching_token(intent)

    with pytest.raises(SimulatedCrashError):
        gateway.process_intent(intent, token)

    store.close()

    # "Restart": a fresh gateway over the same durable state, no crash wrappers.
    fresh_store = SqliteLedgerStore(db_path)
    fresh_wal = WriteAheadLog(wal_path)
    fresh_gateway = _build_recovering_gateway(paper_exchange, fresh_store, fresh_wal)

    report = fresh_gateway.recover()

    assert isinstance(report, RecoveryReport)
    fresh_store.verify_chain()
    assert len(paper_exchange.get_open_orders()) <= 1
    assert fresh_gateway.halted is expected_halted
    _assert_ledger_transitions_form_a_legal_chain_per_coid(fresh_store.read_all())
    if expected_halted:
        assert fresh_gateway.accepting_approvals is False
        halts = [
            r for r in fresh_store.read_all() if r.event_type == "ReconciliationHalted"
        ]
        assert len(halts) == 1
        data = json.loads(halts[0].payload_json)["data"]
        assert data["reason"] == "ambiguous_match"
    else:
        assert fresh_gateway.accepting_approvals is True
    fresh_store.close()


# --- 2. Refuse-until-recovered --------------------------------------------------


def test_process_intent_before_recover_refuses_recovery_pending_and_token_survives(
    tmp_path: Path, paper_exchange: PaperExchange
) -> None:
    """`process_intent` before `.recover()` refuses; the token is untouched.

    A recovery-wired Gateway starts with `accepting_approvals is False`. A
    brand-new intent presented before `.recover()` returns
    `REFUSED_RECOVERY_PENDING`, ledgers exactly one `SubmissionRefused(reason=
    "recovery_pending")`, and never consumes the token: the *identical*
    (intent, token) pair ACKs once `.recover()` has run.
    """
    db_path = tmp_path / "ledger.db"
    wal_path = tmp_path / "wal.jsonl"
    store = SqliteLedgerStore(db_path)
    wal = WriteAheadLog(wal_path)
    gateway = _build_recovering_gateway(paper_exchange, store, wal)

    assert gateway.accepting_approvals is False

    intent = make_intent(idempotency_key="idem-recovery-pending")
    token = issue_matching_token(intent)

    refused = gateway.process_intent(intent, token)

    assert refused.outcome is SubmitOutcome.REFUSED_RECOVERY_PENDING
    assert refused.ack is None
    refusals = [r for r in store.read_all() if r.event_type == "SubmissionRefused"]
    assert len(refusals) == 1
    data = json.loads(refusals[0].payload_json)["data"]
    assert data["reason"] == "recovery_pending"
    assert data["client_order_id"] == client_order_id(intent)

    report = gateway.recover()

    assert isinstance(report, RecoveryReport)
    assert report.halted is False
    assert gateway.accepting_approvals is True

    acked = gateway.process_intent(intent, token)

    assert acked.outcome is SubmitOutcome.ACKED
    store.close()


def test_any_single_recovery_dependency_wired_gates_accepting_approvals(
    tmp_path: Path, paper_exchange: PaperExchange
) -> None:
    """Wiring even *one* recovery dependency gates `accepting_approvals`.

    Wiring only `wal` (with `ledger_reader`/`reconciliation_source` both left
    `None`) is still enough to gate `accepting_approvals` to `False`
    immediately at construction, before `.recover()` ever runs.
    """
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    gateway = OrderGateway(
        PaperSubmitter(paper_exchange), verification_key=KEY_MATERIAL, wal=wal
    )

    assert gateway.accepting_approvals is False


def test_gateway_without_recovery_deps_accepts_immediately_and_recover_is_a_no_op(
    paper_exchange: PaperExchange,
) -> None:
    """Omitting all three recovery deps preserves the pre-issue-#40 surface.

    `wal`/`ledger_reader`/`reconciliation_source` all default `None`, so
    `accepting_approvals` is `True` from construction (never gated) and
    `.recover()` is a harmless no-op reporting nothing to reconcile.
    """
    gateway = OrderGateway(
        PaperSubmitter(paper_exchange), verification_key=KEY_MATERIAL
    )

    assert gateway.accepting_approvals is True
    assert gateway.halted is False

    report = gateway.recover()

    assert isinstance(report, RecoveryReport)
    assert report.halted is False
    assert gateway.accepting_approvals is True


# --- 3. Foreign order halts ----------------------------------------------------


def test_foreign_open_order_with_zero_trace_halts_recovery_and_the_gateway(
    tmp_path: Path, paper_exchange: PaperExchange
) -> None:
    """An untracked resting order (no WAL/ledger trace at all) halts.

    Placed directly on the exchange (bypassing the Gateway entirely), a
    `.recover()` over a ledger and WAL that never mention it halts: the last
    ledger record is `ReconciliationHalted(reason="foreign_open_order")`, the
    Gateway is `.halted` and stops `accepting_approvals`, and a subsequent
    `process_intent` raises `GatewayHaltedError`.
    """
    db_path = tmp_path / "ledger.db"
    wal_path = tmp_path / "wal.jsonl"
    paper_exchange.place_order(
        PaperOrderIntent(
            ticker=DEFAULT_MARKET_TICKER,
            side="yes",
            price=PricePips(4000),
            quantity=ContractCentis(10),
        ),
        object(),
    )
    store = SqliteLedgerStore(db_path)
    wal = WriteAheadLog(wal_path)
    gateway = _build_recovering_gateway(paper_exchange, store, wal)

    report = gateway.recover()

    assert report.halted is True
    assert gateway.halted is True
    assert gateway.accepting_approvals is False
    records = store.read_all()
    assert records[-1].event_type == "ReconciliationHalted"
    data = json.loads(records[-1].payload_json)["data"]
    assert data["reason"] == "foreign_open_order"
    assert data["ticker"] == DEFAULT_MARKET_TICKER

    later_intent = make_intent(idempotency_key="idem-after-foreign-halt")
    later_token = issue_matching_token(later_intent)
    with pytest.raises(GatewayHaltedError):
        gateway.process_intent(later_intent, later_token)
    store.close()


def test_recovery_never_sets_accepting_approvals_true_while_halted(
    tmp_path: Path, paper_exchange: PaperExchange
) -> None:
    """SPEC S11.4: `accepting_approvals` and `.halted` never both hold true.

    Mirrors the foreign-order scenario, asserting the ordering invariant
    directly rather than each flag in isolation.
    """
    db_path = tmp_path / "ledger.db"
    wal_path = tmp_path / "wal.jsonl"
    paper_exchange.place_order(
        PaperOrderIntent(
            ticker=DEFAULT_MARKET_TICKER,
            side="yes",
            price=PricePips(4000),
            quantity=ContractCentis(5),
        ),
        object(),
    )
    store = SqliteLedgerStore(db_path)
    wal = WriteAheadLog(wal_path)
    gateway = _build_recovering_gateway(paper_exchange, store, wal)

    gateway.recover()

    assert gateway.halted is True
    assert gateway.accepting_approvals is False
    assert not (gateway.halted and gateway.accepting_approvals)
    store.close()


# --- 4. Durable halt latch (#39 handoff) ---------------------------------------


def test_reduce_only_violation_halt_latches_durably_across_restart(
    paper_exchange: PaperExchange, tmp_path: Path
) -> None:
    """A `ReduceOnlyViolation` halt (issue #39) survives a restart.

    Drives a post-fill net-short breach (reusing `test_reduce_only.py`'s
    overshooting-fill stub submitter and stub position source) through a
    recovery-wired Gateway, then restarts: a fresh Gateway's `.recover()`
    folds the persisted `ReduceOnlyViolation` and stays halted -- there is no
    un-halt event.
    """
    db_path = tmp_path / "ledger.db"
    wal_path = tmp_path / "wal.jsonl"
    store = SqliteLedgerStore(db_path)
    wal = WriteAheadLog(wal_path)
    position_source = _StubPositionSource((_position(500),))
    submitter = _ForcedFillSubmitter(filled=ContractCentis(600))
    gateway = _build_recovering_gateway(
        paper_exchange,
        store,
        wal,
        position_source=position_source,
        submitter=submitter,
    )
    boot_report = gateway.recover()
    assert boot_report.halted is False
    assert gateway.accepting_approvals is True

    intent = make_intent(
        action="sell_to_close",
        size=ContractCentis(500),
        idempotency_key="idem-durable-violation",
    )
    token = issue_matching_token(intent)

    with pytest.raises(GatewayHaltedError):
        gateway.process_intent(intent, token)

    violations = [r for r in store.read_all() if r.event_type == "ReduceOnlyViolation"]
    assert len(violations) == 1
    store.close()

    fresh_store = SqliteLedgerStore(db_path)
    fresh_wal = WriteAheadLog(wal_path)
    fresh_gateway = _build_recovering_gateway(paper_exchange, fresh_store, fresh_wal)

    report = fresh_gateway.recover()

    assert report.halted is True
    assert fresh_gateway.halted is True
    assert fresh_gateway.accepting_approvals is False

    later_intent = make_intent(idempotency_key="idem-after-restart-violation-halt")
    later_token = issue_matching_token(later_intent)
    with pytest.raises(GatewayHaltedError):
        fresh_gateway.process_intent(later_intent, later_token)
    fresh_store.close()


# --- 5. Durable in-flight-closes tally (#39 handoff) ---------------------------


def test_inflight_closing_tally_survives_restart_for_a_still_resting_close(
    paper_exchange: PaperExchange, tmp_path: Path
) -> None:
    """A still-resting close's in-flight tally is rebuilt across a restart.

    A first 200-centis close against a 500-centis held position rests
    150-centis (the deep_walk fixture's participation cap fills only 50).
    After a restart, a second, 301-centis close exceeds the rebuilt
    remaining headroom (500 - 200 = 300) and is refused, with the rebuilt
    `inflight_closing_centis` reflected verbatim in `position_snapshot`.
    """
    db_path = tmp_path / "ledger.db"
    wal_path = tmp_path / "wal.jsonl"
    store = SqliteLedgerStore(db_path)
    wal = WriteAheadLog(wal_path)
    position_source = _StubPositionSource((_position(500),))
    gateway = _build_recovering_gateway(
        paper_exchange, store, wal, position_source=position_source
    )
    assert gateway.recover().halted is False

    first_close = make_intent(
        action="sell_to_close",
        size=ContractCentis(200),
        idempotency_key="idem-partial-resting-close",
    )
    first_token = issue_matching_token(first_close)
    first_result = gateway.process_intent(first_close, first_token)
    assert first_result.outcome is SubmitOutcome.ACKED
    assert first_result.ack is not None
    assert first_result.ack.filled == ContractCentis(50)
    assert first_result.ack.order_id is not None
    store.close()

    fresh_store = SqliteLedgerStore(db_path)
    fresh_wal = WriteAheadLog(wal_path)
    fresh_position_source = _StubPositionSource((_position(500),))
    fresh_gateway = _build_recovering_gateway(
        paper_exchange, fresh_store, fresh_wal, position_source=fresh_position_source
    )

    report = fresh_gateway.recover()
    assert report.halted is False
    assert fresh_gateway.accepting_approvals is True

    second_close = make_intent(
        action="sell_to_close",
        size=ContractCentis(301),
        idempotency_key="idem-second-close-after-restart",
    )
    second_token = issue_matching_token(second_close)
    second_result = fresh_gateway.process_intent(second_close, second_token)

    assert second_result.outcome is SubmitOutcome.REFUSED_REDUCE_ONLY
    assert second_result.position_snapshot is not None
    assert second_result.position_snapshot.held_centis == 500
    assert second_result.position_snapshot.inflight_closing_centis == 200
    assert second_result.position_snapshot.requested_close_centis == 301
    fresh_store.close()


def test_inflight_closing_tally_is_retired_on_restart_once_a_close_fully_settles(
    paper_exchange: PaperExchange, tmp_path: Path
) -> None:
    """A fully-settled close's in-flight tally is retired, not rebuilt.

    A 50-centis close fully fills immediately (no resting remainder) --
    "settled". After a restart, a subsequent 450-centis close against the
    now-450-centis-held position (the live feed has caught up) ACKs, which
    is only possible if the settled close's in-flight tally was retired to
    zero rather than left standing.
    """
    db_path = tmp_path / "ledger.db"
    wal_path = tmp_path / "wal.jsonl"
    store = SqliteLedgerStore(db_path)
    wal = WriteAheadLog(wal_path)
    position_source = _StubPositionSource((_position(500),))
    gateway = _build_recovering_gateway(
        paper_exchange, store, wal, position_source=position_source
    )
    assert gateway.recover().halted is False

    settled_close = make_intent(
        action="sell_to_close",
        size=ContractCentis(50),
        idempotency_key="idem-fully-settled-close",
    )
    settled_token = issue_matching_token(settled_close)
    settled_result = gateway.process_intent(settled_close, settled_token)

    assert settled_result.outcome is SubmitOutcome.ACKED
    assert settled_result.ack is not None
    assert settled_result.ack.filled == ContractCentis(50)
    assert settled_result.ack.order_id is None
    store.close()

    fresh_store = SqliteLedgerStore(db_path)
    fresh_wal = WriteAheadLog(wal_path)
    # The live position feed has caught up to reflect the settled close.
    fresh_position_source = _StubPositionSource((_position(450),))
    fresh_gateway = _build_recovering_gateway(
        paper_exchange, fresh_store, fresh_wal, position_source=fresh_position_source
    )

    report = fresh_gateway.recover()
    assert report.halted is False

    next_close = make_intent(
        action="sell_to_close",
        size=ContractCentis(450),
        idempotency_key="idem-close-after-retirement",
    )
    next_token = issue_matching_token(next_close)
    next_result = fresh_gateway.process_intent(next_close, next_token)

    assert next_result.outcome is SubmitOutcome.ACKED
    fresh_store.close()


# --- 6. Pristine restart reports zero reconciled -------------------------------


def test_recover_on_pristine_state_reports_zero_reconciled_and_ledgers_completion(
    paper_exchange: PaperExchange, tmp_path: Path
) -> None:
    """`.recover()` over an empty ledger/WAL reports nothing to reconcile.

    Ledgers exactly one `RecoveryCompleted(orders_reconciled=0, halted=False)`
    as the final record.
    """
    db_path = tmp_path / "ledger.db"
    wal_path = tmp_path / "wal.jsonl"
    store = SqliteLedgerStore(db_path)
    wal = WriteAheadLog(wal_path)
    gateway = _build_recovering_gateway(paper_exchange, store, wal)

    report = gateway.recover()

    assert isinstance(report, RecoveryReport)
    assert report.orders_reconciled == 0
    assert report.halted is False
    assert gateway.accepting_approvals is True
    records = store.read_all()
    assert records
    assert records[-1].event_type == "RecoveryCompleted"
    data = json.loads(records[-1].payload_json)["data"]
    assert data["orders_reconciled"] == 0
    assert data["halted"] is False
    store.close()


# --- 7. WAL contract, pinned directly ------------------------------------------


def test_wal_round_trips_intent_and_ack_across_reopen(tmp_path: Path) -> None:
    """Appended intent/ack records survive closing and reopening the WAL."""
    wal_path = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(wal_path)
    intent = make_intent(idempotency_key="idem-wal-roundtrip")
    coid = client_order_id(intent)

    wal.append_intent(intent, coid)
    wal.append_ack(coid, "paper-order-99", ContractCentis(50))

    reopened = WriteAheadLog(wal_path)
    records = reopened.read_all()

    assert len(records) == 2
    intent_record, ack_record = records
    assert intent_record.client_order_id == coid
    assert intent_record.intent == intent
    assert isinstance(intent_record.intent.price, PricePips)
    assert isinstance(intent_record.intent.size, ContractCentis)
    assert ack_record.client_order_id == coid
    assert ack_record.order_id == "paper-order-99"
    assert ack_record.filled == ContractCentis(50)


def test_wal_is_append_only_jsonl_one_record_per_line(tmp_path: Path) -> None:
    """Every append lands as exactly one independently-parseable JSON line."""
    wal_path = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(wal_path)
    intent = make_intent(idempotency_key="idem-wal-jsonl")
    coid = client_order_id(intent)

    wal.append_intent(intent, coid)
    wal.append_ack(coid, "paper-order-1", ContractCentis(10))

    lines = wal_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)


# --- 8. WAL corruption is refused, never guessed (issue #252) -------------------

#: A syntactically well-formed but wrong client-order-id (64 lowercase hex
#: characters, like a real SHA-256 digest) to overwrite a journalled id with.
#: It cannot collide with any real derivation: `client_order_id` hashes the
#: intent's nine fields, and no fixture below hashes to a repeating "dead".
_FOREIGN_COID = "dead" * 16

#: The width of a `client_order_id`: a 64-character lowercase-hex SHA-256
#: digest. Asserted as a precondition below so the near-miss tamperings that
#: flip the first/last digit really are flipping the ends of a full digest.
_COID_HEX_LENGTH = 64

#: The price a tampered payload is rewritten to, in pips -- deliberately
#: different from `make_intent`'s 4600-pip default so the re-derived id is a
#: genuinely different digest rather than an accidental match.
_TAMPERED_PRICE_PIPS = 4700

#: Three distinct sizes (in contract-centis) for the multi-record journal, so
#: the three records differ in a money-path field as well as in their
#: idempotency key: no two can hash alike, and a mixed-up record is visible.
_MULTI_RECORD_SIZES = (100, 150, 175)


def _read_wal_objects(wal_path: Path) -> list[dict[str, object]]:
    """Parse the journal into one mutable mapping per JSONL line.

    Args:
        wal_path: Path to the JSONL journal.

    Returns:
        One decoded mapping per non-empty line, in append order.
    """
    text = wal_path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _rewrite_wal_objects(wal_path: Path, objects: list[dict[str, object]]) -> None:
    """Rewrite the journal from `objects`, one canonical-JSON line each.

    Round-tripping through `canonical_json` -- the same encoder
    `WriteAheadLog._append` uses -- means a tampered file is byte-shaped
    exactly like a genuine one, so the read path rejects it on the
    content-addressed id check and not incidentally on malformed JSON.

    Args:
        wal_path: Path to the JSONL journal to overwrite.
        objects: The record mappings to write, in order.
    """
    wal_path.write_text(
        "".join(canonical_json(obj) + "\n" for obj in objects), encoding="utf-8"
    )


def _with_flipped_hex_digit(coid: str, index: int) -> str:
    """Return `coid` with the hex digit at `index` swapped for a different one.

    Used to build a *near-miss* id that differs from the true one in exactly
    one character. A guard that compared only a prefix (or only a suffix) of
    the two ids would wave such a record through, so flipping each end in turn
    is what proves the whole 64-character digest is compared.

    Args:
        coid: The client-order-id to perturb.
        index: The character position to change.

    Returns:
        A 64-character hex id differing from `coid` at exactly `index`.
    """
    replacement = "1" if coid[index] == "0" else "0"
    return coid[:index] + replacement + coid[index + 1 :]


def _expected_corruption_message(rederived: str, journalled: str) -> str:
    """Build the exact `ValueError` message a corrupt journal must raise with.

    Spelled out here rather than imported from `wal.py` so the assertion is an
    independent statement of the contract: a reworded message, or one that
    swaps which id is reported as re-derived and which as journalled, fails
    these tests instead of silently agreeing with itself.

    Args:
        rederived: The id re-derived from the journalled intent payload.
        journalled: The id the record was actually stored under.

    Returns:
        The exact expected `str(ValueError)`.
    """
    return (
        f"write-ahead log corrupt: intent re-derives client_order_id "
        f"{rederived!r} but was journalled under {journalled!r}"
    )


def _assert_exactly_value_error(error: BaseException, expected_message: str) -> None:
    """Assert `error` is a plain `ValueError` carrying exactly `expected_message`.

    The type check is `is ValueError`, not `isinstance`: a malformed journal
    line makes `json.loads` raise `json.JSONDecodeError`, which *subclasses*
    `ValueError`, so a bare `pytest.raises(ValueError)` would pass for
    entirely the wrong reason -- the record never reaching the integrity check
    at all. Pinning the exact type and the exact message is what makes these
    tests evidence that the corruption guard fired.

    Args:
        error: The exception raised by the read path.
        expected_message: The exact message it must carry.
    """
    assert type(error) is ValueError
    assert str(error) == expected_message


@pytest.mark.parametrize(
    "flip_index",
    [None, 0, _COID_HEX_LENGTH - 1],
    ids=["wholly-foreign-id", "first-digit-differs", "last-digit-differs"],
)
def test_read_all_refuses_an_intent_record_whose_journalled_coid_was_tampered(
    tmp_path: Path, flip_index: int | None
) -> None:
    """Rewriting the stored id (payload intact) is refused, not silently trusted.

    The journalled payload still re-derives its true id, so the two sides
    disagree and the read must fail closed rather than adopt either one: a
    WAL that recovered under the wrong `client_order_id` could resubmit or
    orphan a live order. The message must report the *payload's* id as
    re-derived and the *tampered* id as journalled.

    Three tamperings, because "the ids differ" is not the same claim as "the
    guard compares them in full": besides a wholly foreign id, the stored id
    is perturbed in only its first digit and then in only its last. A guard
    that compared a prefix of the two digests would accept the last-digit
    near miss, and one that compared a suffix would accept the first-digit
    near miss -- both of them silently recovering an order under an id that
    is not its own.
    """
    wal_path = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(wal_path)
    intent = make_intent(idempotency_key="idem-wal-coid-tampered")
    true_coid = client_order_id(intent)
    assert len(true_coid) == _COID_HEX_LENGTH
    wrong_coid = (
        _FOREIGN_COID
        if flip_index is None
        else _with_flipped_hex_digit(true_coid, flip_index)
    )
    assert wrong_coid != true_coid
    assert len(wrong_coid) == _COID_HEX_LENGTH
    wal.append_intent(intent, true_coid)

    objects = _read_wal_objects(wal_path)
    objects[0]["client_order_id"] = wrong_coid
    _rewrite_wal_objects(wal_path, objects)

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value, _expected_corruption_message(true_coid, wrong_coid)
    )


def test_read_all_refuses_an_intent_record_whose_payload_was_tampered(
    tmp_path: Path,
) -> None:
    """Rewriting a money-path field (id intact) is refused, not silently trusted.

    The mirror image of the tampered-id case: here the *payload* is the
    altered side, so the re-derived id is the tampered one and the journalled
    id is the true one -- exactly the opposite roles. Asserting both
    directions pins which id the message reports as which, so an
    implementation that swapped them (or that resolved the mismatch by
    trusting whichever side it read first) cannot pass both tests.
    """
    wal_path = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(wal_path)
    intent = make_intent(idempotency_key="idem-wal-payload-tampered")
    journalled_coid = client_order_id(intent)
    wal.append_intent(intent, journalled_coid)

    objects = _read_wal_objects(wal_path)
    payload = cast("dict[str, object]", objects[0]["intent"])
    assert payload["price"] != _TAMPERED_PRICE_PIPS
    payload["price"] = _TAMPERED_PRICE_PIPS
    _rewrite_wal_objects(wal_path, objects)

    tampered_intent = make_intent(
        idempotency_key="idem-wal-payload-tampered",
        price=PricePips(_TAMPERED_PRICE_PIPS),
    )
    tampered_coid = client_order_id(tampered_intent)
    assert tampered_coid != journalled_coid

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value, _expected_corruption_message(tampered_coid, journalled_coid)
    )


@pytest.mark.parametrize(
    "corrupt_index",
    range(len(_MULTI_RECORD_SIZES)),
    ids=["first-record", "middle-record", "last-record"],
)
def test_read_all_refuses_the_whole_journal_wherever_the_corrupt_record_sits(
    tmp_path: Path, corrupt_index: int
) -> None:
    """One corrupt record poisons the entire read -- there is no partial recovery.

    Three sound records are journalled and read back intact (so the raise
    below can only be caused by the tampering, not by an always-failing read).
    Exactly one is then corrupted, at each position in turn. `read_all()` must
    raise every time: with the corruption last, two perfectly good records
    precede it and are still not returned; with it first or in the middle, a
    guard that only checked the opening record would let a corrupt journal
    through. Handing back a truncated prefix would be the worst outcome of
    all -- a Gateway recovering from a silently shortened WAL has no durable
    ack for orders it already placed.
    """
    wal_path = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(wal_path)
    intents = [
        make_intent(idempotency_key=f"idem-wal-multi-{size}", size=ContractCentis(size))
        for size in _MULTI_RECORD_SIZES
    ]
    coids = [client_order_id(intent) for intent in intents]
    assert len(set(coids)) == len(_MULTI_RECORD_SIZES)
    for intent, coid in zip(intents, coids, strict=True):
        wal.append_intent(intent, coid)

    sound = WriteAheadLog(wal_path).read_all()
    assert [record.client_order_id for record in sound] == coids

    objects = _read_wal_objects(wal_path)
    objects[corrupt_index]["client_order_id"] = _FOREIGN_COID
    _rewrite_wal_objects(wal_path, objects)

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_corruption_message(coids[corrupt_index], _FOREIGN_COID),
    )


def test_recover_over_a_corrupt_wal_keeps_the_gateway_shut_and_resubmits_nothing(
    tmp_path: Path, paper_exchange: PaperExchange
) -> None:
    """A corrupt journal stops recovery dead: no ledger write, no resubmission.

    The end-to-end consequence the guard exists for. One intent is driven all
    the way to `ACKED`, leaving a partially filled order resting on the
    exchange, and its journalled id is then tampered. A fresh Gateway over the
    same durable state must refuse rather than guess: `.recover()` raises the
    corruption `ValueError`, writes nothing to the ledger, and -- critically
    -- leaves `accepting_approvals` `False`, so re-presenting the *same*
    intent and token is refused `REFUSED_RECOVERY_PENDING` instead of being
    submitted a second time. The exchange still shows exactly the one resting
    order it had before recovery, not two.
    """
    db_path = tmp_path / "ledger.db"
    wal_path = tmp_path / "wal.jsonl"
    store = SqliteLedgerStore(db_path)
    gateway = _build_recovering_gateway(paper_exchange, store, WriteAheadLog(wal_path))
    assert gateway.recover().halted is False

    intent = make_intent(idempotency_key="idem-wal-corrupt-recover")
    token = issue_matching_token(intent)
    true_coid = client_order_id(intent)
    assert gateway.process_intent(intent, token).outcome is SubmitOutcome.ACKED
    resting_before = paper_exchange.get_open_orders()
    assert len(resting_before) == 1
    ledger_rows_before = len(store.read_all())
    store.close()

    objects = _read_wal_objects(wal_path)
    objects[0]["client_order_id"] = _FOREIGN_COID
    _rewrite_wal_objects(wal_path, objects)

    fresh_store = SqliteLedgerStore(db_path)
    fresh_gateway = _build_recovering_gateway(
        paper_exchange, fresh_store, WriteAheadLog(wal_path)
    )
    assert fresh_gateway.accepting_approvals is False

    with pytest.raises(ValueError) as excinfo:
        fresh_gateway.recover()

    _assert_exactly_value_error(
        excinfo.value, _expected_corruption_message(true_coid, _FOREIGN_COID)
    )
    assert fresh_gateway.accepting_approvals is False
    assert len(fresh_store.read_all()) == ledger_rows_before

    replayed = fresh_gateway.process_intent(intent, token)

    assert replayed.outcome is SubmitOutcome.REFUSED_RECOVERY_PENDING
    assert replayed.ack is None
    assert paper_exchange.get_open_orders() == resting_before
    refusals = [
        row for row in fresh_store.read_all() if row.event_type == "SubmissionRefused"
    ]
    assert len(refusals) == 1
    assert json.loads(refusals[0].payload_json)["data"]["reason"] == "recovery_pending"
    fresh_store.verify_chain()
    fresh_store.close()


# --- 9. A record the WAL cannot fully validate is refused (issue #427) ----------

#: The exact field set a journalled *intent* record carries. Spelled out here
#: rather than imported from `wal.py` so these tests are an independent
#: statement of the on-disk contract: a read path that quietly widened or
#: narrowed the accepted shape fails them instead of agreeing with itself.
_INTENT_RECORD_FIELDS = ("kind", "client_order_id", "intent")

#: The exact field set a journalled *ack* record carries.
_ACK_RECORD_FIELDS = ("kind", "client_order_id", "order_id", "filled")

#: The exact nine fields an intent record's `intent` payload carries.
_INTENT_PAYLOAD_FIELDS = (
    "intent_id",
    "market_ticker",
    "outcome",
    "action",
    "price",
    "size",
    "max_notional",
    "implied_probability",
    "idempotency_key",
)

#: A venue order id for the hand-built ack records below.
_ACK_ORDER_ID = "venue-order-427"

#: An immediately-filled quantity, in contract-centis, for hand-built acks.
_ACK_FILLED_CENTIS = 25


def _write_wal_text(wal_path: Path, text: str) -> None:
    """Overwrite the journal with exactly `text`, byte for byte.

    Used where the corruption under test is *not* a well-formed record -- a
    blank line, a torn line -- and so cannot be expressed as a mapping to
    re-encode.

    Args:
        wal_path: Path to the JSONL journal to overwrite.
        text: The exact file content to write.
    """
    wal_path.write_text(text, encoding="utf-8")


def _journalled_intent_object(wal_path: Path, key: str) -> dict[str, object]:
    """Journal one genuine intent record and return its decoded mapping.

    The record is produced by the real `append_intent`, so a tampering applied
    to the returned mapping perturbs exactly one thing about an otherwise
    byte-genuine record.

    Args:
        wal_path: Path to the journal to append to (overwritten by callers).
        key: The idempotency key distinguishing this intent.

    Returns:
        The decoded mapping of the record just appended (the journal's last),
        so this may be called more than once to build a multi-record journal.
    """
    intent = make_intent(idempotency_key=key)
    WriteAheadLog(wal_path).append_intent(intent, client_order_id(intent))
    return _read_wal_objects(wal_path)[-1]


def _sound_intent_object(wal_path: Path) -> dict[str, object]:
    """Journal one sound intent record to sit *before* the corrupt one.

    Several refusals below deliberately place their corrupt record on line 2
    rather than line 1. Every message the read path emits names a line, and a
    message that hard-coded line 1 -- or that reported the position of the
    line it started from rather than the one that failed -- would pass a suite
    that only ever corrupted the opening record while misdirecting an operator
    on every real journal.

    Args:
        wal_path: Path to the journal to append to (overwritten by callers).

    Returns:
        The decoded mapping of a genuine intent record.
    """
    return _journalled_intent_object(wal_path, "idem-wal-preceding-sound")


def _ack_object(coid: str) -> dict[str, object]:
    """Build a well-formed ack record mapping for `coid`.

    Args:
        coid: The client-order-id the ack correlates.

    Returns:
        A mapping with exactly the four fields an ack record carries.
    """
    return {
        "kind": "ack",
        "client_order_id": coid,
        "order_id": _ACK_ORDER_ID,
        "filled": _ACK_FILLED_CENTIS,
    }


def _expected_missing_kind_message(lineno: int) -> str:
    """Build the exact message a record with no `kind` field must be refused with.

    Args:
        lineno: The 1-based line the offending record sits on.

    Returns:
        The exact expected `str(ValueError)`.
    """
    return (
        f"write-ahead log corrupt: the record on line {lineno} is missing "
        f"its 'kind' field"
    )


def _expected_unrecognised_kind_message(lineno: int, kind: object) -> str:
    """Build the exact message an unrecognised `kind` must be refused with.

    The offending value is named, so an operator reading the log learns *what*
    the journal claimed rather than being handed an incidental `KeyError` from
    somewhere further down the read path.

    Args:
        lineno: The 1-based line the offending record sits on.
        kind: The unrecognised discriminator value, as decoded from JSON.

    Returns:
        The exact expected `str(ValueError)`.
    """
    return (
        f"write-ahead log corrupt: the record on line {lineno} has unrecognised "
        f"kind {kind!r} (expected 'intent' or 'ack')"
    )


def _expected_field_set_message(
    where: str, actual: Iterable[str], expected: Iterable[str]
) -> str:
    """Build the exact message a wrong field set must be refused with.

    Both sides are reported sorted, so the message is deterministic regardless
    of JSON key order.

    Args:
        where: Which part of which line is malformed, e.g. `"the ack record on
            line 2"`.
        actual: The fields the record actually carries.
        expected: The fields its declared kind requires.

    Returns:
        The exact expected `str(ValueError)`.
    """
    return (
        f"write-ahead log corrupt: {where} carries fields {sorted(actual)} "
        f"(expected {sorted(expected)})"
    )


def _expected_not_an_object_message(where: str) -> str:
    """Build the exact message a non-object JSON value must be refused with.

    Args:
        where: Which part of which line is malformed.

    Returns:
        The exact expected `str(ValueError)`.
    """
    return f"write-ahead log corrupt: {where} is not a JSON object"


def _expected_malformed_line_message(lineno: int) -> str:
    """Build the exact message an unparseable line must be refused with.

    Args:
        lineno: The 1-based line that could not be decoded.

    Returns:
        The exact expected `str(ValueError)`.
    """
    return f"write-ahead log corrupt: line {lineno} is not a well-formed JSON record"


def test_read_all_reads_back_an_ack_that_left_nothing_resting(tmp_path: Path) -> None:
    """A `null` `order_id` is a genuine record shape and stays readable.

    The positive control for the exact-field-set rule below: an ack whose
    placement left nothing resting journals `order_id: null`, which is a
    *present* field carrying `None`. A validator that confused "absent" with
    "null" would refuse this perfectly sound record and take the Gateway down
    on every restart after a fully-filled order.
    """
    wal_path = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(wal_path)
    intent = make_intent(idempotency_key="idem-wal-null-order-id")
    coid = client_order_id(intent)
    wal.append_intent(intent, coid)
    wal.append_ack(coid, None, ContractCentis(_ACK_FILLED_CENTIS))

    records = WriteAheadLog(wal_path).read_all()

    assert len(records) == 2
    assert records[1].kind == "ack"
    assert records[1].client_order_id == coid
    assert records[1].order_id is None
    assert records[1].filled == ContractCentis(_ACK_FILLED_CENTIS)


@pytest.mark.parametrize(
    "kind",
    ["snapshot", "", "Intent", "ack ", "INTENT", None, 0, ["ack"]],
    ids=[
        "future-record-type",
        "empty-string",
        "capitalised-intent",
        "trailing-space-ack",
        "upper-case-intent",
        "json-null",
        "json-integer",
        "json-array",
    ],
)
def test_read_all_refuses_a_record_whose_kind_is_unrecognised(
    tmp_path: Path, kind: object
) -> None:
    """An unrecognised `kind` is refused by name, never assumed to be an ack.

    Every case below is otherwise a *fully well-formed ack record* -- the four
    ack fields, all present and all well-typed -- with only the discriminator
    replaced. So nothing else in the read path can account for the refusal:
    the record is turned away because its `kind` was not recognised, and for no
    other reason. Before issue #427 each of these was read straight down the
    ack branch, which validates nothing at all.

    The near misses matter as much as the obvious ones. `"Intent"`, `"INTENT"`
    and `"ack "` differ from a legal discriminator by case or by one trailing
    space; a guard that lower-cased, stripped, or prefix-matched the value
    would wave them through. A non-string `kind` (`null`, an integer, an array)
    must be refused rather than crash the comparison.
    """
    wal_path = tmp_path / "wal.jsonl"
    sound = _sound_intent_object(wal_path)
    intent = make_intent(idempotency_key="idem-wal-unknown-kind")
    record = _ack_object(client_order_id(intent))
    record["kind"] = kind
    _rewrite_wal_objects(wal_path, [sound, record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value, _expected_unrecognised_kind_message(2, kind)
    )


def test_read_all_refuses_a_record_carrying_no_kind_at_all(tmp_path: Path) -> None:
    """A record with no `kind` field is refused, not treated as an ack.

    Distinct from a record whose `kind` is `null`: here the discriminator is
    absent entirely, and the message says so rather than reporting a `None`
    value the journal never contained.
    """
    wal_path = tmp_path / "wal.jsonl"
    sound = _sound_intent_object(wal_path)
    intent = make_intent(idempotency_key="idem-wal-no-kind")
    record = _ack_object(client_order_id(intent))
    del record["kind"]
    _rewrite_wal_objects(wal_path, [sound, record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(excinfo.value, _expected_missing_kind_message(2))


def test_read_all_refuses_an_intent_record_relabelled_as_an_ack(
    tmp_path: Path,
) -> None:
    """Flipping `kind` to `"ack"` no longer routes an intent past the id guard.

    The headline defect of issue #427. `"ack"` *is* a recognised kind, so the
    discriminator check alone cannot catch this -- what catches it is that the
    record still carries an `intent` field and carries neither of the two
    fields an ack requires. A one-word edit on disk used to be enough to skip
    the `client_order_id` re-derivation entirely.
    """
    wal_path = tmp_path / "wal.jsonl"
    record = _journalled_intent_object(wal_path, "idem-wal-relabelled-ack")
    record["kind"] = "ack"
    _rewrite_wal_objects(wal_path, [record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the ack record on line 1", record, _ACK_RECORD_FIELDS
        ),
    )


def test_read_all_refuses_an_intent_relabelled_as_an_ack_with_ack_fields_grafted_on(
    tmp_path: Path,
) -> None:
    """The relabelling that used to be *silently accepted* is now refused.

    The bare relabel above at least crashed (on a bare `KeyError('order_id')`
    -- fail-closed by accident). This one did not: graft the two missing ack
    fields onto the relabelled record and the old read path returned it
    happily, as an ack under whatever `client_order_id` the journal claimed,
    with the intent payload never validated and never even looked at. That is a
    silent accept, and it is exactly how an order gets recovered under an
    identity that is not its own.

    The leftover `intent` field is what gives it away: a record must carry
    *exactly* the fields its kind requires, so an extra one is corruption even
    when every required field is present and well-formed.
    """
    wal_path = tmp_path / "wal.jsonl"
    record = _journalled_intent_object(wal_path, "idem-wal-relabelled-grafted")
    record["kind"] = "ack"
    record["order_id"] = _ACK_ORDER_ID
    record["filled"] = _ACK_FILLED_CENTIS
    _rewrite_wal_objects(wal_path, [record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the ack record on line 1", record, _ACK_RECORD_FIELDS
        ),
    )


def test_read_all_refuses_an_ack_record_relabelled_as_an_intent(
    tmp_path: Path,
) -> None:
    """The mirror relabelling is refused too, and names the intent shape.

    An ack claiming to be an intent has no payload to re-derive an id from.
    Pinning both directions stops an implementation from validating one kind's
    shape and trusting the other's.
    """
    wal_path = tmp_path / "wal.jsonl"
    sound = _sound_intent_object(wal_path)
    intent = make_intent(idempotency_key="idem-wal-relabelled-intent")
    record = _ack_object(client_order_id(intent))
    record["kind"] = "intent"
    _rewrite_wal_objects(wal_path, [sound, record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the intent record on line 2", record, _INTENT_RECORD_FIELDS
        ),
    )


@pytest.mark.parametrize("dropped", _ACK_RECORD_FIELDS[1:])
def test_read_all_refuses_an_ack_record_missing_a_field_it_requires(
    tmp_path: Path, dropped: str
) -> None:
    """A missing ack field is a named refusal, never a bare `KeyError`.

    `kind` is excluded from the sweep because dropping it is the
    no-discriminator case pinned separately above. Each of the remaining
    fields is dropped in turn: before issue #427, dropping `order_id` or
    `filled` raised `KeyError`, which names a Python dict key and says nothing
    about the journal being corrupt, and dropping `client_order_id` raised the
    same for a record the reader had already decided to trust.
    """
    wal_path = tmp_path / "wal.jsonl"
    intent = make_intent(idempotency_key=f"idem-wal-ack-no-{dropped}")
    record = _ack_object(client_order_id(intent))
    del record[dropped]
    _rewrite_wal_objects(wal_path, [record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the ack record on line 1", record, _ACK_RECORD_FIELDS
        ),
    )


def test_read_all_refuses_an_ack_record_carrying_an_unexpected_field(
    tmp_path: Path,
) -> None:
    """A field this build does not know about is refused, not ignored.

    A newer build's extra field (a checksum, a venue timestamp) is evidence
    this reader cannot fully validate the record. Fail-closed means refusing
    it, not reading the subset it happens to recognise and discarding the rest.
    """
    wal_path = tmp_path / "wal.jsonl"
    sound = _sound_intent_object(wal_path)
    intent = make_intent(idempotency_key="idem-wal-ack-extra-field")
    record = _ack_object(client_order_id(intent))
    record["venue_checksum"] = "0badc0de"
    _rewrite_wal_objects(wal_path, [sound, record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the ack record on line 2", record, _ACK_RECORD_FIELDS
        ),
    )


@pytest.mark.parametrize(
    ("renamed", "replacement"),
    [
        ("order_id", "venue_order_id"),
        ("filled", "filled_centis"),
        ("client_order_id", "coid"),
    ],
)
def test_read_all_refuses_an_ack_record_whose_field_was_renamed(
    tmp_path: Path, renamed: str, replacement: str
) -> None:
    """A renamed field is refused even though the field *count* is unchanged.

    The near miss that separates "the record has the right number of fields"
    from "the record has the right fields". Every other corruption in this
    section adds or removes a field, so a validator that compared only how many
    fields a record carried would pass all of them -- and then index the
    original name and raise the bare `KeyError` this issue exists to abolish.
    Renaming keeps the count at four and changes only which names are there.
    """
    wal_path = tmp_path / "wal.jsonl"
    intent = make_intent(idempotency_key=f"idem-wal-ack-renamed-{renamed}")
    record = _ack_object(client_order_id(intent))
    record[replacement] = record.pop(renamed)
    assert len(record) == len(_ACK_RECORD_FIELDS)
    _rewrite_wal_objects(wal_path, [record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the ack record on line 1", record, _ACK_RECORD_FIELDS
        ),
    )


def test_read_all_refuses_an_intent_record_whose_payload_field_was_renamed(
    tmp_path: Path,
) -> None:
    """Renaming the `intent` field is refused, count unchanged (three fields).

    The same near miss one level up: the record still declares `kind`
    `"intent"` and still carries three fields, but the payload is no longer
    where an intent record's payload lives.
    """
    wal_path = tmp_path / "wal.jsonl"
    record = _journalled_intent_object(wal_path, "idem-wal-record-renamed")
    record["payload"] = record.pop("intent")
    assert len(record) == len(_INTENT_RECORD_FIELDS)
    _rewrite_wal_objects(wal_path, [record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the intent record on line 1", record, _INTENT_RECORD_FIELDS
        ),
    )


@pytest.mark.parametrize(
    ("renamed", "replacement"),
    [("price", "price_pips"), ("size", "quantity"), ("intent_id", "id")],
)
def test_read_all_refuses_an_intent_payload_whose_field_was_renamed(
    tmp_path: Path, renamed: str, replacement: str
) -> None:
    """A renamed payload field is refused, all nine still present by count.

    The most dangerous of the three near misses: the `client_order_id` is a
    digest over these nine fields under these nine names, so a payload whose
    field was renamed cannot re-derive its recorded id even in principle. A
    count-only check would send it to `_intent_from_payload`, which would raise
    a bare `KeyError` for the original name -- never reaching the id guard the
    record was supposed to be checked against.
    """
    wal_path = tmp_path / "wal.jsonl"
    record = _journalled_intent_object(wal_path, f"idem-wal-payload-ren-{renamed}")
    payload = cast("dict[str, object]", record["intent"])
    payload[replacement] = payload.pop(renamed)
    assert len(payload) == len(_INTENT_PAYLOAD_FIELDS)
    _rewrite_wal_objects(wal_path, [record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the intent payload on line 1", payload, _INTENT_PAYLOAD_FIELDS
        ),
    )


@pytest.mark.parametrize("dropped", _INTENT_PAYLOAD_FIELDS)
def test_read_all_refuses_an_intent_payload_missing_one_of_its_nine_fields(
    tmp_path: Path, dropped: str
) -> None:
    """Every one of the nine payload fields is required, by name.

    All nine are swept rather than a representative one, because the failure
    mode differs field by field: before issue #427 each raised its own bare
    `KeyError` from inside `_intent_from_payload`, and a validator that checked
    only the fields it happened to read first would leave the rest exposed.
    """
    wal_path = tmp_path / "wal.jsonl"
    record = _journalled_intent_object(wal_path, f"idem-wal-payload-no-{dropped}")
    payload = cast("dict[str, object]", record["intent"])
    del payload[dropped]
    _rewrite_wal_objects(wal_path, [record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the intent payload on line 1", payload, _INTENT_PAYLOAD_FIELDS
        ),
    )


def test_read_all_refuses_an_intent_payload_carrying_an_unexpected_field(
    tmp_path: Path,
) -> None:
    """An extra payload field is refused rather than dropped on the floor.

    It matters more here than on an ack: the `client_order_id` is derived from
    the intent's nine fields, so a tenth field is content the recorded digest
    cannot possibly attest to. Ignoring it would mean recovering an order whose
    journalled description and whose verified identity disagree.
    """
    wal_path = tmp_path / "wal.jsonl"
    sound = _sound_intent_object(wal_path)
    record = _journalled_intent_object(wal_path, "idem-wal-payload-extra-field")
    payload = cast("dict[str, object]", record["intent"])
    payload["stop_price"] = 4200
    _rewrite_wal_objects(wal_path, [sound, record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the intent payload on line 2", payload, _INTENT_PAYLOAD_FIELDS
        ),
    )


@pytest.mark.parametrize(
    "payload",
    [None, 4600, "intent", ["intent_id"]],
    ids=["json-null", "json-integer", "json-string", "json-array"],
)
def test_read_all_refuses_an_intent_record_whose_payload_is_not_an_object(
    tmp_path: Path, payload: object
) -> None:
    """A non-object `intent` payload is refused before anything indexes it.

    Indexing a string by field name yields `TypeError`, and indexing a list
    yields another; neither says the journal is corrupt, and neither is a
    refusal the read path decided to make.
    """
    wal_path = tmp_path / "wal.jsonl"
    sound = _sound_intent_object(wal_path)
    record = _journalled_intent_object(wal_path, "idem-wal-payload-not-object")
    record["intent"] = payload
    _rewrite_wal_objects(wal_path, [sound, record])

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_not_an_object_message("the intent payload on line 2"),
    )


@pytest.mark.parametrize(
    "line",
    ["null", "4600", '"intent"', "[]", '["kind", "intent"]'],
    ids=["json-null", "json-integer", "json-string", "empty-array", "array"],
)
def test_read_all_refuses_a_line_that_is_not_a_json_object(
    tmp_path: Path, line: str
) -> None:
    """A line that parses but is not an object is refused as corruption.

    It is well-formed JSON, so the decode succeeds and the record reaches the
    shape checks with no mapping to check. Fail-closed: refuse.
    """
    wal_path = tmp_path / "wal.jsonl"
    sound = _sound_intent_object(wal_path)
    _write_wal_text(wal_path, f"{canonical_json(sound)}\n{line}\n")

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value, _expected_not_an_object_message("line 2")
    )


def test_read_all_refuses_a_blank_line_instead_of_silently_skipping_it(
    tmp_path: Path,
) -> None:
    """A blank line is corruption, not whitespace to be tolerated (issue #427).

    Nothing in this journal's writer can produce one: `_append` writes exactly
    one canonical-JSON line plus one newline per record. A blank line therefore
    means the file is no longer what the writer wrote, and the read path used
    to skip it and hand back the surrounding records as though the journal were
    intact. The blank sits between two sound records here, so a reader that
    skipped it would return two perfectly good records and report success.

    The line number must be the blank one, not the first: a refusal that always
    blamed line 1 would send an operator to the wrong record.
    """
    wal_path = tmp_path / "wal.jsonl"
    intent = make_intent(idempotency_key="idem-wal-blank-line")
    coid = client_order_id(intent)
    wal = WriteAheadLog(wal_path)
    wal.append_intent(intent, coid)
    wal.append_ack(coid, _ACK_ORDER_ID, ContractCentis(_ACK_FILLED_CENTIS))
    sound_lines = wal_path.read_text(encoding="utf-8").splitlines()
    assert len(WriteAheadLog(wal_path).read_all()) == 2

    _write_wal_text(wal_path, f"{sound_lines[0]}\n\n{sound_lines[1]}\n")

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(excinfo.value, _expected_malformed_line_message(2))


def test_read_all_refuses_a_torn_trailing_line_as_a_plain_value_error(
    tmp_path: Path,
) -> None:
    """A crash-torn tail is refused with the WAL's own error, not JSON's.

    The realistic corruption: the process died part-way through appending, so
    the final line is a prefix of a record. The read path already stopped here,
    but only incidentally -- `json.JSONDecodeError` *subclasses* `ValueError`,
    so the failure looked like a refusal without any code having refused
    anything. Pinning `type(...) is ValueError` and the exact message is what
    tells the two apart.

    The tear is at line 2 of 2 and one sound record precedes it, so the
    tolerant alternative -- returning the records before the tear -- is
    available and must not be taken.
    """
    wal_path = tmp_path / "wal.jsonl"
    intent = make_intent(idempotency_key="idem-wal-torn-tail")
    coid = client_order_id(intent)
    wal = WriteAheadLog(wal_path)
    wal.append_intent(intent, coid)
    wal.append_ack(coid, _ACK_ORDER_ID, ContractCentis(_ACK_FILLED_CENTIS))
    sound_lines = wal_path.read_text(encoding="utf-8").splitlines()

    torn = sound_lines[1][: len(sound_lines[1]) // 2]
    assert torn
    _write_wal_text(wal_path, f"{sound_lines[0]}\n{torn}")

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(excinfo.value, _expected_malformed_line_message(2))


def test_read_all_refuses_a_torn_line_fused_with_the_append_that_followed_it(
    tmp_path: Path,
) -> None:
    """A tear that a later append ran into is refused, and it is why we refuse.

    This is the case that decides the torn-line rule. A torn line carries no
    terminating newline, so the *next* append lands on the same line and fuses
    with it: the damage is no longer at the tail, and the record that fused
    onto it -- a perfectly good one, durably written after the restart -- is
    destroyed along with it. A reader that tolerated a torn tail by truncating
    at the tear would therefore silently discard sound, later records too. So
    the WAL refuses every unparseable line wherever it sits.
    """
    wal_path = tmp_path / "wal.jsonl"
    intent = make_intent(idempotency_key="idem-wal-fused-tear")
    coid = client_order_id(intent)
    wal = WriteAheadLog(wal_path)
    wal.append_intent(intent, coid)
    sound_line = wal_path.read_text(encoding="utf-8").splitlines()[0]

    _write_wal_text(wal_path, sound_line[: len(sound_line) // 2])
    wal.append_ack(coid, _ACK_ORDER_ID, ContractCentis(_ACK_FILLED_CENTIS))
    assert len(wal_path.read_text(encoding="utf-8").splitlines()) == 1

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(excinfo.value, _expected_malformed_line_message(1))


@pytest.mark.parametrize("bad_index", range(len(_MULTI_RECORD_SIZES)))
def test_read_all_reports_the_line_the_unreadable_record_actually_sits_on(
    tmp_path: Path, bad_index: int
) -> None:
    """The reported line number tracks the offending record's position.

    Three sound records are journalled and read back intact, then exactly one
    is replaced by an unparseable line, at each position in turn. A read path
    that hard-coded a line number, counted from zero, or skipped blank lines
    while numbering would misreport at least one of these -- and an operator
    told to inspect the wrong line of a corrupt journal is being misdirected at
    precisely the worst moment.
    """
    wal_path = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(wal_path)
    for size in _MULTI_RECORD_SIZES:
        intent = make_intent(
            idempotency_key=f"idem-wal-lineno-{size}", size=ContractCentis(size)
        )
        wal.append_intent(intent, client_order_id(intent))
    lines = wal_path.read_text(encoding="utf-8").splitlines()
    assert len(WriteAheadLog(wal_path).read_all()) == len(_MULTI_RECORD_SIZES)

    lines[bad_index] = "{not json"
    _write_wal_text(wal_path, "".join(f"{line}\n" for line in lines))

    with pytest.raises(ValueError) as excinfo:
        WriteAheadLog(wal_path).read_all()

    _assert_exactly_value_error(
        excinfo.value, _expected_malformed_line_message(bad_index + 1)
    )


def test_recover_over_a_relabelled_wal_record_keeps_the_gateway_shut(
    tmp_path: Path, paper_exchange: PaperExchange
) -> None:
    """A relabelled record stops recovery dead, exactly as a tampered id does.

    The end-to-end consequence, mirroring the issue #252 case that PR #426
    pinned but corrupting the journal the way issue #427 describes: one intent
    is driven to `ACKED`, leaving a partially filled order resting, and its
    journalled record's `kind` is then flipped to `"ack"`. A fresh Gateway must
    refuse rather than guess -- `.recover()` raises, writes nothing to the
    ledger, and leaves `accepting_approvals` `False`, so re-presenting the same
    intent and token is refused `REFUSED_RECOVERY_PENDING` instead of being
    submitted a second time. The exchange still shows the one resting order it
    had, not two.
    """
    db_path = tmp_path / "ledger.db"
    wal_path = tmp_path / "wal.jsonl"
    store = SqliteLedgerStore(db_path)
    gateway = _build_recovering_gateway(paper_exchange, store, WriteAheadLog(wal_path))
    assert gateway.recover().halted is False

    intent = make_intent(idempotency_key="idem-wal-relabel-recover")
    token = issue_matching_token(intent)
    assert gateway.process_intent(intent, token).outcome is SubmitOutcome.ACKED
    resting_before = paper_exchange.get_open_orders()
    assert len(resting_before) == 1
    ledger_rows_before = len(store.read_all())
    store.close()

    objects = _read_wal_objects(wal_path)
    objects[0]["kind"] = "ack"
    _rewrite_wal_objects(wal_path, objects)

    fresh_store = SqliteLedgerStore(db_path)
    fresh_gateway = _build_recovering_gateway(
        paper_exchange, fresh_store, WriteAheadLog(wal_path)
    )
    assert fresh_gateway.accepting_approvals is False

    with pytest.raises(ValueError) as excinfo:
        fresh_gateway.recover()

    _assert_exactly_value_error(
        excinfo.value,
        _expected_field_set_message(
            "the ack record on line 1", objects[0], _ACK_RECORD_FIELDS
        ),
    )
    assert fresh_gateway.accepting_approvals is False
    assert len(fresh_store.read_all()) == ledger_rows_before

    replayed = fresh_gateway.process_intent(intent, token)

    assert replayed.outcome is SubmitOutcome.REFUSED_RECOVERY_PENDING
    assert replayed.ack is None
    assert paper_exchange.get_open_orders() == resting_before
    fresh_store.verify_chain()
    fresh_store.close()
