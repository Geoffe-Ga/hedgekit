"""Process B: the Risk Kernel veto authority (SPEC S5.1-S5.3).

The Risk Kernel is the sole holder of the approval-token **signing** key and
runs with read-only exchange credentials (SPEC S5.2). It validates normalized
order intents, reserves capital, and signs single-use approval tokens. Per the
SPEC S5.3 import boundary, only this package may import the approval-token
signing key handle; a CI-enforced import-linter check (wired via
scripts/check-all.sh, issue #91) enforces that rule alongside the stronger
pure-stdlib AST test.

This package root re-exports the mode machine (:mod:`~windbreak.riskkernel.modes`),
the pre-trade checks (:mod:`~windbreak.riskkernel.checks`), the process skeleton
(:mod:`~windbreak.riskkernel.process`), the gate-plan-anchored promotion-evidence
producer (:mod:`~windbreak.riskkernel.evidence`), the capital-reservation ledger
and approval pipeline (:mod:`~windbreak.riskkernel.reservations`), and the
approval-token issuer (:mod:`~windbreak.riskkernel.tokens`). It deliberately does
**not** re-export :mod:`windbreak.riskkernel.signing`: the signing key handle is
reachable only via its fully qualified module path, preserving the SPEC S5.3
boundary.
"""

from windbreak.riskkernel.checks import (
    DEFAULT_CHECKS,
    Check,
    CheckResult,
    Decision,
    OrderIntent,
    evaluate_intent,
)
from windbreak.riskkernel.context import (
    AccountState,
    EvaluationContext,
    ExchangeTradingStatus,
    FeeBounds,
    JurisdictionStatus,
    MarketView,
    ProductType,
    RiskLimits,
)
from windbreak.riskkernel.demotion import (
    TRIGGER_ACTIONS,
    DemotionAction,
    DemotionTrigger,
    resolve_demotion,
)
from windbreak.riskkernel.evidence import (
    SECONDS_PER_DAY,
    anchor_gate_evidence,
    anchored_paper_window_days,
)
from windbreak.riskkernel.floor import worst_case_cost, worst_case_equity
from windbreak.riskkernel.governance import (
    DEFAULT_FLOOR_LOWER_COOL_OFF_SECONDS,
    ChangeOrigin,
    CoolOffActiveError,
    FloorGovernance,
    ForbiddenOriginError,
    LoweringAlreadyPendingError,
    NonceMismatchError,
    NoPendingLowerError,
    PendingFloorLower,
)
from windbreak.riskkernel.human_ack import (
    DEFAULT_HUMAN_ACK_TTL_SECONDS,
    AckLapsedError,
    DuplicateAckRequestError,
    HumanAckQueue,
    PendingHumanAck,
    Releaser,
    UnknownApprovalError,
)
from windbreak.riskkernel.kill import (
    DashboardChallengeError,
    DashboardKillStub,
    DirectiveSink,
    KillFileWatcher,
    KillIntegration,
    KillSwitch,
    KillTrigger,
    ReconciliationMismatchMonitor,
    ReplayedKillState,
    kill_state_in,
)
from windbreak.riskkernel.modes import (
    REARM_CONFIRMATION_PHRASE,
    IllegalModeTransitionError,
    KillReArmError,
    Mode,
    ModeCeilingExceededError,
    ModeStateMachine,
)
from windbreak.riskkernel.process import (
    InMemoryKernelLedgerWriter,
    KernelLedgerWriter,
    LoggingKernelLedgerWriter,
    RiskKernel,
)
from windbreak.riskkernel.promotion import (
    SIGNIFICANCE_OVERRIDE_ACK_PHRASE,
    Comparison,
    CriterionResult,
    GateCriterion,
    GateEvidence,
    GatePlanUnavailableError,
    OverrideAcknowledgementError,
    PromotionDecision,
    PromotionGate,
    build_promotion_gates,
    constant_promotion_gates,
    effective_mode_ceiling,
    evaluate_promotion,
    override_applied_in,
    paper_gate_from_thresholds,
)
from windbreak.riskkernel.reservations import (
    ApprovalOutcome,
    ApprovalPipeline,
    DuplicateReservationError,
    Reservation,
    ReservationLedger,
)
from windbreak.riskkernel.tokens import DEFAULT_TOKEN_TTL_SECONDS, TokenIssuer

__all__ = [
    "DEFAULT_CHECKS",
    "DEFAULT_FLOOR_LOWER_COOL_OFF_SECONDS",
    "DEFAULT_HUMAN_ACK_TTL_SECONDS",
    "DEFAULT_TOKEN_TTL_SECONDS",
    "REARM_CONFIRMATION_PHRASE",
    "SECONDS_PER_DAY",
    "SIGNIFICANCE_OVERRIDE_ACK_PHRASE",
    "TRIGGER_ACTIONS",
    "AccountState",
    "AckLapsedError",
    "ApprovalOutcome",
    "ApprovalPipeline",
    "ChangeOrigin",
    "Check",
    "CheckResult",
    "Comparison",
    "CoolOffActiveError",
    "CriterionResult",
    "DashboardChallengeError",
    "DashboardKillStub",
    "Decision",
    "DemotionAction",
    "DemotionTrigger",
    "DirectiveSink",
    "DuplicateAckRequestError",
    "DuplicateReservationError",
    "EvaluationContext",
    "ExchangeTradingStatus",
    "FeeBounds",
    "FloorGovernance",
    "ForbiddenOriginError",
    "GateCriterion",
    "GateEvidence",
    "GatePlanUnavailableError",
    "HumanAckQueue",
    "IllegalModeTransitionError",
    "InMemoryKernelLedgerWriter",
    "JurisdictionStatus",
    "KernelLedgerWriter",
    "KillFileWatcher",
    "KillIntegration",
    "KillReArmError",
    "KillSwitch",
    "KillTrigger",
    "LoggingKernelLedgerWriter",
    "LoweringAlreadyPendingError",
    "MarketView",
    "Mode",
    "ModeCeilingExceededError",
    "ModeStateMachine",
    "NoPendingLowerError",
    "NonceMismatchError",
    "OrderIntent",
    "OverrideAcknowledgementError",
    "PendingFloorLower",
    "PendingHumanAck",
    "ProductType",
    "PromotionDecision",
    "PromotionGate",
    "ReconciliationMismatchMonitor",
    "Releaser",
    "ReplayedKillState",
    "Reservation",
    "ReservationLedger",
    "RiskKernel",
    "RiskLimits",
    "TokenIssuer",
    "UnknownApprovalError",
    "anchor_gate_evidence",
    "anchored_paper_window_days",
    "build_promotion_gates",
    "constant_promotion_gates",
    "effective_mode_ceiling",
    "evaluate_intent",
    "evaluate_promotion",
    "kill_state_in",
    "override_applied_in",
    "paper_gate_from_thresholds",
    "resolve_demotion",
    "worst_case_cost",
    "worst_case_equity",
]
