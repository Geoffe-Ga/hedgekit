"""Failing-first tests for gate-plan-sourced PAPER promotion (issue #185, RED).

Issue #185 rewires the PAPER -> LIVE_MICRO promotion gate to consume the
pre-registered, content-addressed :class:`~windbreak.evaluation.preregistration.
GatePlan` -- read from the ledger at promotion time via
:func:`~windbreak.evaluation.preregistration.latest_gate_plan_registration` --
instead of the live :class:`~windbreak.config.EvaluationConfig`, and fails
closed (raising the new ``GatePlanUnavailableError``) whenever no verified plan
is available. RESEARCH -> PAPER and LIVE_MICRO -> LIVE stay sourced from pinned
module constants, built eagerly at construction, and never consult the plan
store.

Neither ``windbreak.riskkernel.promotion.paper_gate_from_thresholds`` nor
``windbreak.riskkernel.promotion.GatePlanUnavailableError`` exist yet, and
``RiskKernel.__init__``/``RiskKernel.from_events`` do not yet accept a
``gate_plan_store`` keyword (nor has ``evaluation_config`` been removed), so
this file fails collection today with an ``ImportError: cannot import name
'GatePlanUnavailableError' from 'windbreak.riskkernel.promotion'`` -- the
expected Gate 1 RED state for issue #185. Once the promotion-module symbols
exist, `_kernel_at`'s ``gate_plan_store=`` keyword would next raise
``TypeError: RiskKernel.__init__() got an unexpected keyword argument
'gate_plan_store'`` until ``process.py`` is updated too.

ASSUMPTIONS this file makes explicit (renegotiate with the architect if wrong):

1. ``paper_gate_from_thresholds`` is a *pure* function (no ledger access) that
   returns a full ten-criterion PAPER -> LIVE_MICRO :class:`PromotionGate`,
   identical in every other criterion to :func:`build_promotion_gates`'s own
   PAPER gate -- only the three plan-sourced thresholds are parameters.
2. ``GatePlanUnavailableError`` derives from ``Exception`` (not a subclass of
   any existing riskkernel error), matching ``OverrideAcknowledgementError``'s
   own shape.
3. The kernel reads the gate plan lazily, once per ``request_promotion`` call
   on the PAPER rung -- never cached across calls -- so a
   ``GatePlanChanged`` registered between two promotion attempts is picked up
   by the very next attempt without rebuilding the kernel.

Issue #244 (RED -- neither ``windbreak.ledger.events.PromotionBlocked`` nor
``RiskKernel``'s ``ledger_blocked_promotions`` keyword exist yet, so this file
fails collection today with an ``ImportError: cannot import name
'PromotionBlocked' from 'windbreak.ledger.events'``) adds an opt-in audit
event on the fail-closed PAPER promotion path: with
``ledger_blocked_promotions=True``, a ``GatePlanUnavailableError`` raised by
``_gate_for`` before any ledger write records exactly one
``PromotionBlocked(source_mode="PAPER", target_mode="LIVE_MICRO",
reason=str(err))`` and then re-raises, mode unchanged. The default
(``False``) is the pre-#244 behavior: no event. A new module-private
constant, ``windbreak.riskkernel.process._PAPER_PROMOTION_TARGET``, names the
PAPER rung's promotion target and is drift-guarded against
``windbreak.riskkernel.modes._next_rung(Mode.PAPER)``.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from windbreak.config import EvaluationConfig
from windbreak.evaluation.preregistration import build_gate_plan, register_gate_plan
from windbreak.ledger.events import PromotionBlocked
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.riskkernel.modes import Mode, ModeStateMachine, _next_rung
from windbreak.riskkernel.process import (
    _PAPER_PROMOTION_TARGET,
    InMemoryKernelLedgerWriter,
    RiskKernel,
)
from windbreak.riskkernel.promotion import (
    SIGNIFICANCE_OVERRIDE_ACK_PHRASE,
    GateEvidence,
    GatePlanUnavailableError,
    build_promotion_gates,
    paper_gate_from_thresholds,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from windbreak.ledger.events import Event
    from windbreak.ledger.store import LedgerRecord, LedgerStore

#: The default (SPEC S16) `EvaluationConfig`, used as the plan-building
#: baseline whenever a test does not deliberately probe a different config.
_DEFAULT_CONFIG = EvaluationConfig()

#: The `paper_fill_model_version` every plan built in this file pins (SPEC
#: §17.4); its value is irrelevant to promotion evaluation, so a single fixed
#: string is reused everywhere.
_PAPER_FILL_MODEL_VERSION = "gate-plan-test-v1"


def _counter_clock(start: int = 1_700_000_000) -> Callable[[], int]:
    """Build a strictly monotonically increasing int clock.

    Each call to the returned callable advances by exactly one whole second
    from `start`, so back-to-back `register_gate_plan` calls sharing one clock
    always satisfy the strictly-later-epoch precondition for a
    `GatePlanChanged` registration.

    Args:
        start: The first epoch second the clock returns.

    Returns:
        A zero-argument callable returning successive whole epoch seconds.
    """
    counter = itertools.count(start)

    def _clock() -> int:
        return next(counter)

    return _clock


def _register(
    store: LedgerStore,
    evaluation: EvaluationConfig,
    *,
    now: Callable[[], int],
    paper_fill_model_version: str = _PAPER_FILL_MODEL_VERSION,
) -> None:
    """Build a `GatePlan` from `evaluation` and register it into `store`.

    Args:
        store: The ledger to register the plan into.
        evaluation: The config the plan's thresholds are snapshotted from.
        now: The clock supplying the registration's paper-clock epoch.
        paper_fill_model_version: The pinned fill-model version (SPEC §17.4).
    """
    plan = build_gate_plan(
        evaluation, paper_fill_model_version=paper_fill_model_version
    )
    register_gate_plan(plan, store, now=now)


def _kernel_at(
    mode: Mode,
    *,
    ceiling: Mode = Mode.LIVE,
    gate_plan_store: LedgerStore | None = None,
    ledger_blocked_promotions: bool = False,
) -> RiskKernel:
    """Build a `RiskKernel` parked at `mode`, ceilinged at `ceiling`.

    Args:
        mode: The starting operating mode.
        ceiling: The configured `mode_ceiling`.
        gate_plan_store: The ledger the kernel reads its PAPER gate plan from,
            or `None` to leave the kernel with no plan source wired.
        ledger_blocked_promotions: Whether the kernel should ledger a
            `PromotionBlocked` audit event on a fail-closed PAPER promotion
            attempt (issue #244). Defaults to `False` (current behavior).

    Returns:
        A `RiskKernel` wired to a fresh `InMemoryKernelLedgerWriter`.
    """
    machine = ModeStateMachine(mode_ceiling=ceiling, mode=mode)
    return RiskKernel(
        InMemoryKernelLedgerWriter(),
        mode_machine=machine,
        gate_plan_store=gate_plan_store,
        ledger_blocked_promotions=ledger_blocked_promotions,
    )


def _paper_evidence(
    *,
    resolved: int = 300,
    independent_groups: int = 100,
    brier_skill_ppm: int = 10_000,
    significance_ci_lower_ppm: int = 1,
) -> GateEvidence:
    """Build PAPER->LIVE_MICRO evidence, the three plan-sourced fields free.

    Every non-plan-sourced criterion (P&L, window, drawdown, calibration band,
    kernel-invariant count) is pinned at its passing value, so only the four
    caller-supplied fields (the three plan-sourced thresholds' evidence, plus
    the mandatory significance criterion) can flip a test's verdict.

    Args:
        resolved: `resolved_realtime_forecast_count` evidence value.
        independent_groups: `independent_event_group_count` evidence value.
        brier_skill_ppm: `brier_skill_ppm` evidence value.
        significance_ci_lower_ppm: `brier_skill_ci_lower_ppm` evidence value
            (the mandatory, overridable significance criterion).

    Returns:
        The assembled `GateEvidence`.
    """
    return GateEvidence(
        resolved_realtime_forecast_count=resolved,
        independent_event_group_count=independent_groups,
        brier_skill_ppm=brier_skill_ppm,
        brier_skill_ci_lower_ppm=significance_ci_lower_ppm,
        paper_pnl_net_micro_usd=1,
        paper_window_days=90,
        paper_max_drawdown_ppm=0,
        calibration_slope_ppm=1_000_000,
        kernel_invariant_failure_count=0,
    )


#: A `GateEvidence` snapshot satisfying every RESEARCH->PAPER criterion,
#: irrelevant to the plan store (RESEARCH's gate is plan-independent).
_RESEARCH_ALL_PASSING = GateEvidence(
    forecast_count=50,
    adversarial_suite_green=True,
    days_without_unhandled_errors=14,
    ledger_rebuild_verified=True,
)

#: A `GateEvidence` snapshot satisfying every LIVE_MICRO->LIVE criterion,
#: irrelevant to the plan store (LIVE_MICRO's gate is plan-independent).
_LIVE_MICRO_ALL_PASSING = GateEvidence(
    live_micro_days=60,
    live_slippage_vs_paper_ppm=0,
    live_brier_degradation_ppm=0,
    reconciliation_halt_count=0,
    invariant_violation_count=0,
    operator_confirmation=True,
)


class _StaticLedgerStore:
    """A minimal, read-only `LedgerStore` stub serving fixed records.

    Used to hand the kernel a doctored `GatePlanRegistered` record without
    round-tripping through a real, hash-chained `SqliteLedgerStore` (which
    would refuse to persist an already-tampered row).
    """

    def __init__(self, records: list[LedgerRecord]) -> None:
        """Initialize the stub with the fixed records `read_all` returns.

        Args:
            records: The records `read_all` always returns, verbatim.
        """
        self._records = records

    def append(self, event: Event) -> int:
        """Refuse to append; this stub is read-only.

        Args:
            event: Ignored.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("_StaticLedgerStore is read-only")

    def read_all(self) -> list[LedgerRecord]:
        """Return the fixed records this stub was built with."""
        return list(self._records)

    def verify_chain(self) -> None:
        """No-op: this stub carries no real hash chain to verify."""
        return None

    def close(self) -> None:
        """No-op: this stub owns no real resource."""
        return None


def _tampered_gate_plan_store(tmp_path: Path) -> LedgerStore:
    """Register one real plan, then serve a hash-mismatched copy of its record.

    Registers a genuine plan into a throwaway `SqliteLedgerStore`, reads back
    its single persisted record, mutates one plan-payload field (leaving the
    stored `plan_hash` untouched), and wraps the doctored record in a
    `_StaticLedgerStore` -- so `latest_gate_plan_registration` recomputes a
    content hash that no longer matches the stored one, exactly the "corrupt
    or tampered payload" case `_registration_from_record` fails closed on.

    Args:
        tmp_path: The pytest tmp-path directory to root the throwaway store in.

    Returns:
        A `LedgerStore` stub whose single record fails hash verification.
    """
    real_store = SqliteLedgerStore(tmp_path / "tampered_source.db")
    _register(real_store, _DEFAULT_CONFIG, now=_counter_clock())
    record = real_store.read_all()[0]
    real_store.close()

    envelope = json.loads(record.payload_json)
    envelope["data"]["promotion_min_resolved"] = (
        envelope["data"]["promotion_min_resolved"] + 1
    )
    tampered_record = dataclasses.replace(record, payload_json=json.dumps(envelope))
    return _StaticLedgerStore([tampered_record])


# --- Fail-closed: no store, empty store, corrupt registration -------------------


def test_promotion_raises_gate_plan_unavailable_when_no_store_is_wired() -> None:
    """`gate_plan_store=None` on a PAPER kernel raises `GatePlanUnavailableError`
    before any `PromotionEvaluated` event is recorded, leaving the mode
    unchanged -- never falling back to a live `EvaluationConfig`.
    """
    kernel = _kernel_at(Mode.PAPER, ceiling=Mode.LIVE, gate_plan_store=None)

    with pytest.raises(GatePlanUnavailableError):
        kernel.request_promotion(_paper_evidence())

    assert kernel.ledger_writer.events == []
    assert kernel.mode is Mode.PAPER


def test_promotion_raises_gate_plan_unavailable_when_store_has_no_registration(
    tmp_path: Path,
) -> None:
    """A wired but empty gate-plan store (no `GatePlanRegistered` ever
    appended) raises `GatePlanUnavailableError` identically to no store at
    all: zero events recorded, mode unchanged.
    """
    store = SqliteLedgerStore(tmp_path / "empty.db")
    kernel = _kernel_at(Mode.PAPER, ceiling=Mode.LIVE, gate_plan_store=store)

    with pytest.raises(GatePlanUnavailableError):
        kernel.request_promotion(_paper_evidence())

    assert kernel.ledger_writer.events == []
    assert kernel.mode is Mode.PAPER
    store.close()


def test_promotion_raises_gate_plan_unavailable_with_cause_on_tampered_registration(
    tmp_path: Path,
) -> None:
    """A hash-mismatched (tampered) registration raises `GatePlanUnavailableError`
    whose `__cause__` is the underlying `ValueError`/`TypeError` the ledger read
    raised, and still records zero events and leaves the mode unchanged.
    """
    store = _tampered_gate_plan_store(tmp_path)
    kernel = _kernel_at(Mode.PAPER, ceiling=Mode.LIVE, gate_plan_store=store)

    with pytest.raises(GatePlanUnavailableError) as exc_info:
        kernel.request_promotion(_paper_evidence())

    assert isinstance(exc_info.value.__cause__, (ValueError, TypeError))
    assert kernel.ledger_writer.events == []
    assert kernel.mode is Mode.PAPER


# --- Plan overrides config, both directions (mutation-resistant) ----------------


def test_stricter_registered_plan_rejects_evidence_the_default_config_would_pass(
    tmp_path: Path,
) -> None:
    """A plan whose `promotion_min_resolved` (301) is stricter than the
    `EvaluationConfig()` default (300) rejects evidence that clears the
    default (300 resolved forecasts) but not the plan: proves the kernel
    reads the *plan*, not the live config.
    """
    store = SqliteLedgerStore(tmp_path / "strict.db")
    _register(
        store,
        dataclasses.replace(_DEFAULT_CONFIG, promotion_min_resolved=301),
        now=_counter_clock(),
    )
    kernel = _kernel_at(Mode.PAPER, ceiling=Mode.LIVE, gate_plan_store=store)

    decision = kernel.request_promotion(_paper_evidence(resolved=300))

    assert decision.approved is False
    assert kernel.mode is Mode.PAPER
    store.close()


def test_looser_registered_plan_approves_evidence_the_default_config_would_reject(
    tmp_path: Path,
) -> None:
    """A plan whose `promotion_min_resolved` (299) is looser than the
    `EvaluationConfig()` default (300) approves evidence that fails the
    default (299 resolved forecasts) but clears the plan: proves the kernel
    never silently falls back to the live config's stricter default.
    """
    store = SqliteLedgerStore(tmp_path / "loose.db")
    _register(
        store,
        dataclasses.replace(_DEFAULT_CONFIG, promotion_min_resolved=299),
        now=_counter_clock(),
    )
    kernel = _kernel_at(Mode.PAPER, ceiling=Mode.LIVE, gate_plan_store=store)

    decision = kernel.request_promotion(_paper_evidence(resolved=299))

    assert decision.approved is True
    assert kernel.mode is Mode.LIVE_MICRO
    store.close()


# --- Mid-run GatePlanChanged is consumed on the very next attempt ---------------


def test_a_registered_loose_plan_alone_approves_the_qualifying_evidence(
    tmp_path: Path,
) -> None:
    """Baseline half of the plan-change pair: a single loose registration
    (299) approves evidence pinned at exactly 299 resolved forecasts.
    """
    store = SqliteLedgerStore(tmp_path / "single_loose.db")
    _register(
        store,
        dataclasses.replace(_DEFAULT_CONFIG, promotion_min_resolved=299),
        now=_counter_clock(),
    )
    kernel = _kernel_at(Mode.PAPER, ceiling=Mode.LIVE, gate_plan_store=store)

    decision = kernel.request_promotion(_paper_evidence(resolved=299))

    assert decision.approved is True
    assert kernel.mode is Mode.LIVE_MICRO
    store.close()


def test_a_subsequent_stricter_registration_flips_the_identical_evidence_to_rejected(
    tmp_path: Path,
) -> None:
    """Registering a stricter plan (301) *after* an initial loose one (299),
    both on one strictly-advancing clock, means the very next
    `request_promotion` call re-reads the ledger and rejects the identical
    evidence (299) that the loose plan alone would have approved -- the
    kernel never caches a stale plan across calls.
    """
    store = SqliteLedgerStore(tmp_path / "changed.db")
    clock = _counter_clock()
    _register(
        store,
        dataclasses.replace(_DEFAULT_CONFIG, promotion_min_resolved=299),
        now=clock,
    )
    _register(
        store,
        dataclasses.replace(_DEFAULT_CONFIG, promotion_min_resolved=301),
        now=clock,
    )
    kernel = _kernel_at(Mode.PAPER, ceiling=Mode.LIVE, gate_plan_store=store)

    decision = kernel.request_promotion(_paper_evidence(resolved=299))

    assert decision.approved is False
    assert kernel.mode is Mode.PAPER
    store.close()


# --- Ledger audit echoes the registered plan's thresholds, not config's --------


def test_promotion_evaluated_results_echo_the_registered_plan_thresholds(
    tmp_path: Path,
) -> None:
    """The `PromotionEvaluated` event's `results` entries for the three
    plan-sourced criteria carry the *registered plan's* threshold values
    (250/80/9000), not the `EvaluationConfig()` defaults (300/100/10000) --
    proving the audit trail itself reflects the plan, not the live config.
    """
    store = SqliteLedgerStore(tmp_path / "audit.db")
    _register(
        store,
        dataclasses.replace(
            _DEFAULT_CONFIG,
            promotion_min_resolved=250,
            promotion_min_independent_event_groups=80,
            brier_skill_required_ppm=9_000,
        ),
        now=_counter_clock(),
    )
    kernel = _kernel_at(Mode.PAPER, ceiling=Mode.LIVE, gate_plan_store=store)
    evidence = _paper_evidence(
        resolved=250, independent_groups=80, brier_skill_ppm=9_000
    )

    kernel.request_promotion(evidence)

    events = [
        event
        for event in kernel.ledger_writer.events
        if event.event_type == "PromotionEvaluated"
    ]
    assert len(events) == 1
    results_by_id = {
        result["criterion_id"]: result for result in events[0].payload["results"]
    }
    assert results_by_id["paper_resolved_forecasts"]["threshold"] == 250
    assert results_by_id["paper_independent_event_groups"]["threshold"] == 80
    assert results_by_id["paper_brier_skill"]["threshold"] == 9_000
    store.close()


# --- Plan-independence of the other two rungs -----------------------------------


def test_research_to_paper_promotes_with_no_gate_plan_store_wired() -> None:
    """RESEARCH->PAPER never reads the plan store: it promotes on all-passing
    evidence even with `gate_plan_store=None`.
    """
    kernel = _kernel_at(Mode.RESEARCH, ceiling=Mode.LIVE, gate_plan_store=None)

    decision = kernel.request_promotion(_RESEARCH_ALL_PASSING)

    assert decision.approved is True
    assert kernel.mode is Mode.PAPER


def test_live_micro_to_live_promotes_with_no_gate_plan_store_wired() -> None:
    """LIVE_MICRO->LIVE never reads the plan store: it promotes on all-passing
    evidence even with `gate_plan_store=None`.
    """
    kernel = _kernel_at(Mode.LIVE_MICRO, ceiling=Mode.LIVE, gate_plan_store=None)

    decision = kernel.request_promotion(_LIVE_MICRO_ALL_PASSING)

    assert decision.approved is True
    assert kernel.mode is Mode.LIVE


# --- Override regression: significance bypass still works with a plan wired ----


def test_significance_override_still_bypasses_with_a_registered_gate_plan(
    tmp_path: Path,
) -> None:
    """With a gate plan registered and evidence failing *only* the mandatory
    significance criterion, an active significance override still bypasses
    exactly as it does without a plan wired: `override_bypassed=True` and the
    mode advances to LIVE_MICRO.
    """
    store = SqliteLedgerStore(tmp_path / "override.db")
    _register(store, _DEFAULT_CONFIG, now=_counter_clock())
    kernel = _kernel_at(Mode.PAPER, ceiling=Mode.LIVE, gate_plan_store=store)
    evidence = _paper_evidence(significance_ci_lower_ppm=0)
    kernel.apply_ledgered_override(SIGNIFICANCE_OVERRIDE_ACK_PHRASE)

    decision = kernel.request_promotion(evidence)

    assert decision.approved is False
    assert kernel.mode is Mode.LIVE_MICRO
    events = [
        event
        for event in kernel.ledger_writer.events
        if event.event_type == "PromotionEvaluated"
    ]
    assert len(events) == 1
    assert events[0].payload["override_bypassed"] is True
    store.close()


# --- promotion.py pure unit: paper_gate_from_thresholds + DRY guard ------------


def test_paper_gate_from_thresholds_stamps_the_three_thresholds_by_criterion_id() -> (
    None
):
    """`paper_gate_from_thresholds` stamps each of its three keyword
    thresholds onto the correctly-identified criterion, matched by
    `criterion_id` (not positionally).
    """
    gate = paper_gate_from_thresholds(
        promotion_min_resolved=111,
        promotion_min_independent_event_groups=222,
        brier_skill_required_ppm=333,
    )

    by_id = {criterion.criterion_id: criterion for criterion in gate.criteria}
    assert by_id["paper_resolved_forecasts"].threshold == 111
    assert by_id["paper_independent_event_groups"].threshold == 222
    assert by_id["paper_brier_skill"].threshold == 333
    assert gate.source is Mode.PAPER
    assert gate.target is Mode.LIVE_MICRO


def test_paper_gate_from_thresholds_has_ten_criteria() -> None:
    """`paper_gate_from_thresholds` produces the same ten-criterion shape as
    `build_promotion_gates(...)[Mode.PAPER]` -- it is a thin threshold-only
    parameterization of the same gate, not a divergent second definition.
    """
    gate = paper_gate_from_thresholds(
        promotion_min_resolved=300,
        promotion_min_independent_event_groups=100,
        brier_skill_required_ppm=10_000,
    )

    assert len(gate.criteria) == 10


def test_paper_gate_from_thresholds_matches_build_promotion_gates() -> None:
    """DRY guard: feeding `paper_gate_from_thresholds` the same three
    threshold values `build_promotion_gates(EvaluationConfig())` would use
    produces an *equal* `PromotionGate` -- `_paper_to_live_micro_gate` must
    delegate to `paper_gate_from_thresholds`, not maintain a second,
    independently-drifting criteria tuple.
    """
    config = _DEFAULT_CONFIG
    via_config = build_promotion_gates(config)[Mode.PAPER]

    via_thresholds = paper_gate_from_thresholds(
        promotion_min_resolved=config.promotion_min_resolved,
        promotion_min_independent_event_groups=(
            config.promotion_min_independent_event_groups
        ),
        brier_skill_required_ppm=config.brier_skill_required_ppm,
    )

    assert via_thresholds == via_config


def test_gate_plan_unavailable_error_is_exported_from_riskkernel_package() -> None:
    """`GatePlanUnavailableError` (design-contract item 4) is re-exported from
    `windbreak.riskkernel`'s package root, mirroring
    `OverrideAcknowledgementError`'s own existing re-export.
    """
    import windbreak.riskkernel as riskkernel_package

    assert riskkernel_package.GatePlanUnavailableError is GatePlanUnavailableError
    assert "GatePlanUnavailableError" in riskkernel_package.__all__


# --- Import-order regression: guards a fragile evaluation<->riskkernel cycle ---


def test_importing_riskkernel_process_then_evaluation_live_divergence_succeeds() -> (
    None
):
    """Importing `windbreak.riskkernel.process` (which must read a gate plan
    from `windbreak.evaluation.preregistration`) before
    `windbreak.evaluation.live_divergence` (which imports
    `windbreak.riskkernel.demotion`) succeeds in a clean subprocess -- this
    import order must never deadlock in a partially-initialized-module cycle.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import windbreak.riskkernel.process; "
            "import windbreak.evaluation.live_divergence",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_importing_evaluation_live_divergence_then_riskkernel_process_succeeds() -> (
    None
):
    """The reverse import order -- `windbreak.evaluation.live_divergence`
    before `windbreak.riskkernel.process` -- also succeeds in a clean
    subprocess, so neither import order can ever trip the cycle.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import windbreak.evaluation.live_divergence; "
            "import windbreak.riskkernel.process",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


# --- Issue #244: optional PromotionBlocked audit event on fail-closed PAPER ----
# --- promotion, gated by `ledger_blocked_promotions` -----------------------------


def _promotion_blocked_events(kernel: RiskKernel) -> list[PromotionBlocked]:
    """Return every `PromotionBlocked` event recorded on `kernel`'s writer.

    Args:
        kernel: The kernel whose in-memory ledger writer is inspected.

    Returns:
        The recorded `PromotionBlocked` events, in recording order.
    """
    return [
        event
        for event in kernel.ledger_writer.events
        if isinstance(event, PromotionBlocked)
    ]


def test_ledger_blocked_promotions_records_promotion_blocked_when_no_store_wired() -> (
    None
):
    """With `ledger_blocked_promotions=True` and no plan store wired, a PAPER
    promotion attempt still raises `GatePlanUnavailableError`, but now records
    exactly one `PromotionBlocked` event carrying the source/target modes and
    the raised error's message, and the mode stays unchanged.
    """
    kernel = _kernel_at(
        Mode.PAPER,
        ceiling=Mode.LIVE,
        gate_plan_store=None,
        ledger_blocked_promotions=True,
    )

    with pytest.raises(GatePlanUnavailableError):
        kernel.request_promotion(_paper_evidence())

    blocked = _promotion_blocked_events(kernel)
    assert len(blocked) == 1
    assert blocked[0].payload["source_mode"] == "PAPER"
    assert blocked[0].payload["target_mode"] == "LIVE_MICRO"
    assert "no gate plan store wired" in blocked[0].payload["reason"]
    assert kernel.mode is Mode.PAPER


def test_ledger_blocked_promotions_records_promotion_blocked_when_store_empty(
    tmp_path: Path,
) -> None:
    """With `ledger_blocked_promotions=True` and a wired-but-empty plan store,
    the fail-closed promotion attempt records one `PromotionBlocked` whose
    reason names the missing registration.
    """
    store = SqliteLedgerStore(tmp_path / "empty_blocked.db")
    kernel = _kernel_at(
        Mode.PAPER,
        ceiling=Mode.LIVE,
        gate_plan_store=store,
        ledger_blocked_promotions=True,
    )

    with pytest.raises(GatePlanUnavailableError):
        kernel.request_promotion(_paper_evidence())

    blocked = _promotion_blocked_events(kernel)
    assert len(blocked) == 1
    assert "no registered gate plan" in blocked[0].payload["reason"]
    store.close()


def test_ledger_blocked_promotions_records_promotion_blocked_on_tampered_registration(
    tmp_path: Path,
) -> None:
    """With `ledger_blocked_promotions=True` and a tampered registration, the
    fail-closed promotion attempt still records one `PromotionBlocked` whose
    reason names the unreadable registration, and the raised error's
    `__cause__` is still the underlying `ValueError`/`TypeError` -- ledgering
    the audit event never swallows or replaces the original cause chain.
    """
    store = _tampered_gate_plan_store(tmp_path)
    kernel = _kernel_at(
        Mode.PAPER,
        ceiling=Mode.LIVE,
        gate_plan_store=store,
        ledger_blocked_promotions=True,
    )

    with pytest.raises(GatePlanUnavailableError) as exc_info:
        kernel.request_promotion(_paper_evidence())

    blocked = _promotion_blocked_events(kernel)
    assert len(blocked) == 1
    assert "unreadable" in blocked[0].payload["reason"]
    assert isinstance(exc_info.value.__cause__, (ValueError, TypeError))


def test_ledger_blocked_promotions_records_nothing_on_a_successful_promotion(
    tmp_path: Path,
) -> None:
    """With `ledger_blocked_promotions=True` but a valid registered plan and
    passing evidence, promotion proceeds normally: the kernel records exactly
    one `PromotionEvaluated` and zero `PromotionBlocked` -- the flag must
    never fire a false positive on the success path.
    """
    store = SqliteLedgerStore(tmp_path / "blocked_flag_success.db")
    _register(store, _DEFAULT_CONFIG, now=_counter_clock())
    kernel = _kernel_at(
        Mode.PAPER,
        ceiling=Mode.LIVE,
        gate_plan_store=store,
        ledger_blocked_promotions=True,
    )

    decision = kernel.request_promotion(_paper_evidence())

    assert decision.approved is True
    assert kernel.mode is Mode.LIVE_MICRO
    evaluated = [
        event
        for event in kernel.ledger_writer.events
        if event.event_type == "PromotionEvaluated"
    ]
    assert len(evaluated) == 1
    assert _promotion_blocked_events(kernel) == []
    store.close()


def test_ledger_blocked_promotions_defaults_to_off_recording_nothing() -> None:
    """The default (omitted `ledger_blocked_promotions`) is `False`: a PAPER
    kernel's fail-closed promotion attempt records no events at all,
    identically to the pre-#244 behavior -- an explicit regression guard for
    the default-off contract.
    """
    kernel = _kernel_at(Mode.PAPER, ceiling=Mode.LIVE, gate_plan_store=None)

    with pytest.raises(GatePlanUnavailableError):
        kernel.request_promotion(_paper_evidence())

    assert kernel.ledger_writer.events == []


def test_ledger_blocked_promotions_is_plumbed_through_from_events() -> None:
    """`RiskKernel.from_events(..., ledger_blocked_promotions=True)` forwards
    the flag verbatim: a kernel rebuilt via `from_events` and parked at PAPER
    still records one `PromotionBlocked` on the fail-closed path, proving the
    flag survives the replay constructor rather than only `__init__`.
    """
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.PAPER)
    kernel = RiskKernel.from_events(
        [],
        InMemoryKernelLedgerWriter(),
        mode_machine=machine,
        gate_plan_store=None,
        ledger_blocked_promotions=True,
    )

    with pytest.raises(GatePlanUnavailableError):
        kernel.request_promotion(_paper_evidence())

    assert len(_promotion_blocked_events(kernel)) == 1


def test_paper_promotion_target_constant_matches_next_rung_of_paper() -> None:
    """Drift guard: the named `_PAPER_PROMOTION_TARGET` module constant stays
    equal to the mode ladder's own one-rung-up target for `Mode.PAPER`, so the
    two never silently diverge if the ladder is ever reordered.
    """
    assert _PAPER_PROMOTION_TARGET is _next_rung(Mode.PAPER)


def test_promotion_blocked_event_round_trips_through_a_real_sqlite_ledger(
    tmp_path: Path,
) -> None:
    """A ledger containing a `PromotionBlocked` record still verifies and
    round-trips cleanly through a real, hash-chained `SqliteLedgerStore` --
    proving the new event type is a first-class, chain-safe ledger citizen,
    not merely a shape that satisfies the in-memory writer.
    """
    store = SqliteLedgerStore(tmp_path / "promotion_blocked_tolerance.db")
    store.append(
        PromotionBlocked(
            component="riskkernel",
            source_mode="PAPER",
            target_mode="LIVE_MICRO",
            reason="no gate plan store wired; promotion blocked (fail-closed)",
        )
    )

    store.verify_chain()

    records = store.read_all()
    assert len(records) == 1
    payload = json.loads(records[0].payload_json)
    assert payload["data"]["source_mode"] == "PAPER"
    store.close()
