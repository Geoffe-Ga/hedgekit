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
