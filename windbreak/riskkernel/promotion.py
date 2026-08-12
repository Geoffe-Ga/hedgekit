"""Promotion-gate machinery for the Risk Kernel (SPEC S5.1, S10.9).

The Risk Kernel advances up the operating-mode ladder only when a pinned set of
:class:`PromotionGate`\\s is satisfied: RESEARCH -> PAPER, PAPER -> LIVE_MICRO,
and LIVE_MICRO -> LIVE. Each gate is a tuple of :class:`GateCriterion`\\s, each a
single ``(evidence_field, comparison, threshold)`` predicate evaluated against a
:class:`GateEvidence` snapshot by the pure, non-short-circuiting
:func:`evaluate_promotion`.

This module is deliberately pure data plus evaluation: it imports no ledger
machinery, so it can be unit-tested and reasoned about in isolation. The
kernel-level entrypoints (:meth:`RiskKernel.request_promotion` and the ledgered
significance override) live in :mod:`windbreak.riskkernel.process` and build on
these primitives. Note that nothing under ``windbreak/`` constructs a
:class:`GateEvidence` or calls ``request_promotion``, so no shipped path
evaluates any of these gates against real evidence (issue #542).

The PAPER -> LIVE_MICRO gate's three plan-sourced thresholds (resolved-count,
independent-event-group, and Brier-skill) are no longer read from a live
:class:`~windbreak.config.EvaluationConfig` at promotion time (issue #185).
Instead the kernel builds that gate lazily from the pre-registered,
content-addressed gate plan via :func:`paper_gate_from_thresholds`, failing
closed with :class:`GatePlanUnavailableError` when no readable plan exists. The
RESEARCH -> PAPER and LIVE_MICRO -> LIVE gates stay pinned to module constants
(built eagerly via :func:`constant_promotion_gates`); :func:`build_promotion_gates`
remains for the pure-evaluation unit suites that still source the PAPER
thresholds from a config.

Evaluation is table-driven over the :class:`Comparison` enum (see
:data:`_COMPARATORS`) rather than a branch tree, keeping each function's
cyclomatic complexity low and making the comparison operators mutation-resistant
(each is exercised on both sides of its boundary by the test suite).

The whole package is float-free (SPEC S6.1): every evidence value and threshold
is an ``int`` or ``bool``, and boolean evidence is coerced to ``int`` (``0``/
``1``) so a single integer comparison table covers every criterion.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from windbreak.riskkernel.modes import _LADDER_RANK, Mode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from windbreak.config import EvaluationConfig
    from windbreak.ledger.events import Event

# --- SPEC S10.9 promotion thresholds (pinned module constants) -------------------
#
# TODO(#114): migrate the numeric thresholds below into
# ``EvaluationConfig`` -- they are SPEC S10.9 figures with no config home yet
# (paper max drawdown, calibration-slope band, live slippage/degradation maxima,
# and the RESEARCH forecast/error-free-day/window-day minimums). Editing
# ``windbreak/config/schema.py`` is out of this issue's fence, so a dedicated
# follow-up issue (#114) tracks it. The three config-sourced
# thresholds (resolved/independent-groups/Brier-skill) are already read from
# ``EvaluationConfig`` in :func:`build_promotion_gates`.

#: RESEARCH -> PAPER: minimum resolved forecasts before paper trading.
_RESEARCH_MIN_FORECASTS = 50

#: RESEARCH -> PAPER: minimum consecutive days without an unhandled error.
_RESEARCH_MIN_ERROR_FREE_DAYS = 14

#: PAPER -> LIVE_MICRO: minimum paper-trading observation window, in days.
_PAPER_MIN_WINDOW_DAYS = 90

#: PAPER -> LIVE_MICRO: paper max drawdown must be strictly below this (30%).
_PAPER_MAX_DRAWDOWN_THRESHOLD_PPM = 300_000

#: PAPER -> LIVE_MICRO: acceptable calibration-slope band, low and high edges.
_CALIBRATION_SLOPE_LOW_PPM = 800_000
_CALIBRATION_SLOPE_HIGH_PPM = 1_200_000

#: LIVE_MICRO -> LIVE: minimum live-micro trading duration, in days.
_LIVE_MICRO_MIN_DAYS = 60

#: LIVE_MICRO -> LIVE: max acceptable live-vs-paper slippage and Brier drift.
_LIVE_SLIPPAGE_MAX_PPM = 50_000
_LIVE_BRIER_DEGRADATION_MAX_PPM = 20_000

#: ``GT`` this floor means "strictly positive" (P&L, CI-lower significance).
_STRICT_POSITIVE_FLOOR = 0

#: ``EQ`` this means "must be exactly zero" (invariant/halt counts).
_FORBIDDEN_COUNT = 0

#: ``IS_TRUE`` compares a coerced bool (``0``/``1``) against this floor.
_TRUTHY_THRESHOLD = 1

#: The exact, case-sensitive phrase an operator must type to apply the one-way
#: significance-gate override. It contains cased characters, so any case-folded
#: or case-swapped near-miss is rejected verbatim.
SIGNIFICANCE_OVERRIDE_ACK_PHRASE = (
    "OVERRIDE SIGNIFICANCE GATE: I ACCEPT REDUCED STATISTICAL CONFIDENCE"
)

#: The highest mode the significance override can ever reach: never ``LIVE``.
OVERRIDE_CEILING = Mode.LIVE_MICRO

#: ``event_type`` string of the ledgered significance-override marker event.
_OVERRIDE_EVENT_TYPE = "SignificanceOverrideApplied"


class OverrideAcknowledgementError(Exception):
    """Raised when the significance-override acknowledgement phrase is wrong."""


class GatePlanUnavailableError(Exception):
    """Raised when promotion cannot read a registered gate plan (fail-closed)."""


class Comparison(enum.Enum):
    """The six comparison operators a :class:`GateCriterion` may apply."""

    GE = enum.auto()
    GT = enum.auto()
    LE = enum.auto()
    LT = enum.auto()
    EQ = enum.auto()
    IS_TRUE = enum.auto()


#: Table-driven comparison evaluators, one per :class:`Comparison` member. Each
#: takes the coerced-integer observed value and the criterion's integer
#: threshold and returns whether the criterion passes. ``IS_TRUE`` treats the
#: coerced bool (``0``/``1``) as passing iff it reaches the truthy floor.
_COMPARATORS: Mapping[Comparison, Callable[[int, int], bool]] = {
    Comparison.GE: lambda observed, threshold: observed >= threshold,
    Comparison.GT: lambda observed, threshold: observed > threshold,
    Comparison.LE: lambda observed, threshold: observed <= threshold,
    Comparison.LT: lambda observed, threshold: observed < threshold,
    Comparison.EQ: lambda observed, threshold: observed == threshold,
    Comparison.IS_TRUE: lambda observed, threshold: observed >= threshold,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class GateEvidence:
    """A keyword-only, failing-closed snapshot of promotion-readiness evidence.

    Every field defaults to its failing value (``0`` for counts/measurements,
    ``False`` for flags), so a bare ``GateEvidence()`` satisfies no real
    criterion. All values are ``int`` or ``bool`` (SPEC S6.1 float-free);
    per-million (``_ppm``) and micro-USD (``_micro_usd``) fields carry scaled
    integers.

    Attributes:
        forecast_count: Resolved forecasts produced in research.
        adversarial_suite_green: Whether the adversarial test suite passes.
        days_without_unhandled_errors: Consecutive error-free operating days.
        ledger_rebuild_verified: Whether a ledger rebuild reproduced state.
        resolved_realtime_forecast_count: Resolved real-time forecasts.
        independent_event_group_count: Independent event groups forecast.
        brier_skill_ppm: Brier skill score, in parts-per-million.
        brier_skill_ci_lower_ppm: Lower confidence bound on Brier skill (ppm).
        brier_skill_ci_upper_ppm: Upper confidence bound on Brier skill (ppm).
        paper_pnl_net_micro_usd: Net paper-trading P&L, in micro-USD.
        paper_window_days: Length of the paper-trading window, in days.
        paper_max_drawdown_ppm: Worst paper drawdown, in parts-per-million.
        calibration_slope_ppm: Calibration slope, in parts-per-million.
        kernel_invariant_failure_count: Kernel invariant failures observed.
        live_micro_days: Days spent trading in live-micro mode.
        live_slippage_vs_paper_ppm: Live-vs-paper slippage, in ppm.
        live_brier_degradation_ppm: Live Brier degradation vs paper, in ppm.
        reconciliation_halt_count: Reconciliation-triggered halts observed.
        invariant_violation_count: Live invariant violations observed.
        operator_confirmation: Whether an operator confirmed the promotion.
    """

    forecast_count: int = 0
    adversarial_suite_green: bool = False
    days_without_unhandled_errors: int = 0
    ledger_rebuild_verified: bool = False
    resolved_realtime_forecast_count: int = 0
    independent_event_group_count: int = 0
    brier_skill_ppm: int = 0
    brier_skill_ci_lower_ppm: int = 0
    brier_skill_ci_upper_ppm: int = 0
    paper_pnl_net_micro_usd: int = 0
    paper_window_days: int = 0
    paper_max_drawdown_ppm: int = 0
    calibration_slope_ppm: int = 0
    kernel_invariant_failure_count: int = 0
    live_micro_days: int = 0
    live_slippage_vs_paper_ppm: int = 0
    live_brier_degradation_ppm: int = 0
    reconciliation_halt_count: int = 0
    invariant_violation_count: int = 0
    operator_confirmation: bool = False

    def to_payload(self) -> dict[str, object]:
        """Project every field into a JSON-safe, field-keyed mapping.

        Returns:
            A dict keyed by field name, values being the raw ``int``/``bool``.
        """
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


@dataclass(frozen=True)
class GateCriterion:
    """A single promotion predicate over one evidence field.

    Attributes:
        criterion_id: Stable identifier for this criterion.
        evidence_field: The :class:`GateEvidence` field this criterion reads.
        comparison: The :class:`Comparison` operator applied.
        threshold: The integer threshold the observed value is compared against.
        description: Human-readable explanation of the criterion.
        overridable: Whether an active significance override may bypass this
            criterion when it fails. Marks the sole criterion the one-way
            significance override is permitted to promote past (SPEC S10.9);
            every other criterion keeps the default ``False`` and always blocks
            promotion when it fails, override or not.
    """

    criterion_id: str
    evidence_field: str
    comparison: Comparison
    threshold: int
    description: str
    overridable: bool = False

    def to_payload(self) -> dict[str, object]:
        """Project the criterion into a JSON-safe, field-keyed mapping.

        Returns:
            A dict keyed by field name, with ``comparison`` rendered as its
            ``.name`` string rather than the raw enum member.
        """
        return {
            "criterion_id": self.criterion_id,
            "evidence_field": self.evidence_field,
            "comparison": self.comparison.name,
            "threshold": self.threshold,
            "description": self.description,
            "overridable": self.overridable,
        }


@dataclass(frozen=True)
class PromotionGate:
    """The ordered criteria guarding one ladder promotion.

    Attributes:
        source: The mode being promoted from.
        target: The mode being promoted to.
        criteria: The criteria, all of which must pass for approval.
    """

    source: Mode
    target: Mode
    criteria: tuple[GateCriterion, ...]

    def to_payload(self) -> dict[str, object]:
        """Project the gate into a JSON-safe mapping.

        Returns:
            A dict with ``source``/``target`` rendered as mode ``.name``
            strings and ``criteria`` as a list of nested criterion payloads.
        """
        return {
            "source": self.source.name,
            "target": self.target.name,
            "criteria": [criterion.to_payload() for criterion in self.criteria],
        }


@dataclass(frozen=True)
class CriterionResult:
    """The evaluated outcome of one :class:`GateCriterion`.

    Attributes:
        criterion_id: The evaluated criterion's identifier.
        observed: The coerced-integer observed value (bools become ``0``/``1``).
        threshold: The criterion's threshold, echoed for audit.
        comparison: The comparison that was applied.
        passed: Whether the criterion passed.
    """

    criterion_id: str
    observed: int
    threshold: int
    comparison: Comparison
    passed: bool


@dataclass(frozen=True)
class PromotionDecision:
    """The full, non-short-circuiting outcome of evaluating one gate.

    Attributes:
        source: The gate's source mode.
        target: The gate's target mode.
        approved: Whether every criterion passed.
        results: One :class:`CriterionResult` per gate criterion, in gate order.
    """

    source: Mode
    target: Mode
    approved: bool
    results: tuple[CriterionResult, ...]


def _research_to_paper_gate() -> PromotionGate:
    """Build the RESEARCH -> PAPER gate from pinned module constants.

    Returns:
        The four-criterion RESEARCH -> PAPER :class:`PromotionGate`.
    """
    criteria = (
        GateCriterion(
            criterion_id="research_min_forecasts",
            evidence_field="forecast_count",
            comparison=Comparison.GE,
            threshold=_RESEARCH_MIN_FORECASTS,
            description="at least 50 resolved forecasts",
        ),
        GateCriterion(
            criterion_id="research_error_free_days",
            evidence_field="days_without_unhandled_errors",
            comparison=Comparison.GE,
            threshold=_RESEARCH_MIN_ERROR_FREE_DAYS,
            description="at least 14 consecutive error-free days",
        ),
        GateCriterion(
            criterion_id="research_adversarial_suite_green",
            evidence_field="adversarial_suite_green",
            comparison=Comparison.IS_TRUE,
            threshold=_TRUTHY_THRESHOLD,
            description="adversarial test suite is green",
        ),
        GateCriterion(
            criterion_id="research_ledger_rebuild_verified",
            evidence_field="ledger_rebuild_verified",
            comparison=Comparison.IS_TRUE,
            threshold=_TRUTHY_THRESHOLD,
            description="ledger rebuild reproduced kernel state",
        ),
    )
    return PromotionGate(source=Mode.RESEARCH, target=Mode.PAPER, criteria=criteria)


def paper_gate_from_thresholds(
    *,
    promotion_min_resolved: int,
    promotion_min_independent_event_groups: int,
    brier_skill_required_ppm: int,
) -> PromotionGate:
    """Build the PAPER -> LIVE_MICRO gate, stamping three caller-supplied thresholds.

    This is the pure, ledger-free seam the kernel uses to build the PAPER gate
    lazily from a pre-registered gate plan (issue #185): the three plan-sourced
    thresholds are parameters, every other criterion is pinned to the SPEC S10.9
    module constants. :func:`_paper_to_live_micro_gate` delegates here, so the
    ten-criterion shape has exactly one definition (no drift).

    Args:
        promotion_min_resolved: Threshold for the resolved-real-time-forecast
            count criterion (``paper_resolved_forecasts``).
        promotion_min_independent_event_groups: Threshold for the
            independent-event-group count criterion
            (``paper_independent_event_groups``).
        brier_skill_required_ppm: Threshold for the Brier-skill floor criterion
            (``paper_brier_skill``), in parts-per-million.

    Returns:
        The ten-criterion PAPER -> LIVE_MICRO :class:`PromotionGate`.
    """
    criteria = (
        GateCriterion(
            criterion_id="paper_resolved_forecasts",
            evidence_field="resolved_realtime_forecast_count",
            comparison=Comparison.GE,
            threshold=promotion_min_resolved,
            description="enough resolved real-time forecasts",
        ),
        GateCriterion(
            criterion_id="paper_independent_event_groups",
            evidence_field="independent_event_group_count",
            comparison=Comparison.GE,
            threshold=promotion_min_independent_event_groups,
            description="enough independent event groups",
        ),
        GateCriterion(
            criterion_id="paper_brier_skill",
            evidence_field="brier_skill_ppm",
            comparison=Comparison.GE,
            threshold=brier_skill_required_ppm,
            description="Brier skill meets the required floor",
        ),
        GateCriterion(
            criterion_id="paper_brier_skill_significance",
            evidence_field="brier_skill_ci_lower_ppm",
            comparison=Comparison.GT,
            threshold=_STRICT_POSITIVE_FLOOR,
            description="Brier-skill confidence interval excludes zero",
            overridable=True,
        ),
        GateCriterion(
            criterion_id="paper_pnl_positive",
            evidence_field="paper_pnl_net_micro_usd",
            comparison=Comparison.GT,
            threshold=_STRICT_POSITIVE_FLOOR,
            description="net paper P&L is positive",
        ),
        GateCriterion(
            criterion_id="paper_window_days",
            evidence_field="paper_window_days",
            comparison=Comparison.GE,
            threshold=_PAPER_MIN_WINDOW_DAYS,
            description="paper window is long enough",
        ),
        GateCriterion(
            criterion_id="paper_max_drawdown",
            evidence_field="paper_max_drawdown_ppm",
            comparison=Comparison.LT,
            threshold=_PAPER_MAX_DRAWDOWN_THRESHOLD_PPM,
            description="paper max drawdown stays under the ceiling",
        ),
        GateCriterion(
            criterion_id="paper_calibration_slope_low",
            evidence_field="calibration_slope_ppm",
            comparison=Comparison.GE,
            threshold=_CALIBRATION_SLOPE_LOW_PPM,
            description="calibration slope at or above the band's low edge",
        ),
        GateCriterion(
            criterion_id="paper_calibration_slope_high",
            evidence_field="calibration_slope_ppm",
            comparison=Comparison.LE,
            threshold=_CALIBRATION_SLOPE_HIGH_PPM,
            description="calibration slope at or below the band's high edge",
        ),
        GateCriterion(
            criterion_id="paper_kernel_invariant_failures",
            evidence_field="kernel_invariant_failure_count",
            comparison=Comparison.EQ,
            threshold=_FORBIDDEN_COUNT,
            description="no kernel invariant failures",
        ),
    )
    return PromotionGate(source=Mode.PAPER, target=Mode.LIVE_MICRO, criteria=criteria)


def _paper_to_live_micro_gate(evaluation: EvaluationConfig) -> PromotionGate:
    """Build the PAPER -> LIVE_MICRO gate, config-sourcing three thresholds.

    A thin delegate to :func:`paper_gate_from_thresholds` that reads the three
    plan-sourced thresholds off an :class:`~windbreak.config.EvaluationConfig`,
    kept so :func:`build_promotion_gates` stays byte-identical for the
    pure-evaluation unit suites.

    Args:
        evaluation: The evaluation config supplying the resolved-count,
            independent-group, and Brier-skill thresholds.

    Returns:
        The ten-criterion PAPER -> LIVE_MICRO :class:`PromotionGate`.
    """
    return paper_gate_from_thresholds(
        promotion_min_resolved=evaluation.promotion_min_resolved,
        promotion_min_independent_event_groups=(
            evaluation.promotion_min_independent_event_groups
        ),
        brier_skill_required_ppm=evaluation.brier_skill_required_ppm,
    )


def _live_micro_to_live_gate() -> PromotionGate:
    """Build the LIVE_MICRO -> LIVE gate from pinned module constants.

    Returns:
        The six-criterion LIVE_MICRO -> LIVE :class:`PromotionGate`.
    """
    criteria = (
        GateCriterion(
            criterion_id="live_micro_days",
            evidence_field="live_micro_days",
            comparison=Comparison.GE,
            threshold=_LIVE_MICRO_MIN_DAYS,
            description="enough days in live-micro mode",
        ),
        GateCriterion(
            criterion_id="live_micro_slippage",
            evidence_field="live_slippage_vs_paper_ppm",
            comparison=Comparison.LE,
            threshold=_LIVE_SLIPPAGE_MAX_PPM,
            description="live-vs-paper slippage within tolerance",
        ),
        GateCriterion(
            criterion_id="live_micro_brier_degradation",
            evidence_field="live_brier_degradation_ppm",
            comparison=Comparison.LE,
            threshold=_LIVE_BRIER_DEGRADATION_MAX_PPM,
            description="live Brier degradation within tolerance",
        ),
        GateCriterion(
            criterion_id="live_micro_reconciliation_halts",
            evidence_field="reconciliation_halt_count",
            comparison=Comparison.EQ,
            threshold=_FORBIDDEN_COUNT,
            description="no reconciliation halts",
        ),
        GateCriterion(
            criterion_id="live_micro_invariant_violations",
            evidence_field="invariant_violation_count",
            comparison=Comparison.EQ,
            threshold=_FORBIDDEN_COUNT,
            description="no invariant violations",
        ),
        GateCriterion(
            criterion_id="live_micro_operator_confirmation",
            evidence_field="operator_confirmation",
            comparison=Comparison.IS_TRUE,
            threshold=_TRUTHY_THRESHOLD,
            description="operator confirmed the promotion",
        ),
    )
    return PromotionGate(source=Mode.LIVE_MICRO, target=Mode.LIVE, criteria=criteria)


def build_promotion_gates(evaluation: EvaluationConfig) -> Mapping[Mode, PromotionGate]:
    """Build the three promotion gates, keyed by their source mode.

    Args:
        evaluation: The evaluation config supplying the PAPER -> LIVE_MICRO
            gate's three config-sourced thresholds; the remaining thresholds
            are pinned module constants (SPEC S10.9).

    Returns:
        A mapping of source :class:`Mode` (``RESEARCH``/``PAPER``/
        ``LIVE_MICRO``) to its :class:`PromotionGate`.
    """
    return {
        Mode.RESEARCH: _research_to_paper_gate(),
        Mode.PAPER: _paper_to_live_micro_gate(evaluation),
        Mode.LIVE_MICRO: _live_micro_to_live_gate(),
    }


def constant_promotion_gates() -> Mapping[Mode, PromotionGate]:
    """Build the two config-independent promotion gates, keyed by source mode.

    Returns only the RESEARCH -> PAPER and LIVE_MICRO -> LIVE gates, both pinned
    entirely to SPEC S10.9 module constants. The PAPER -> LIVE_MICRO gate is
    deliberately absent: the kernel builds it lazily from the registered gate
    plan at promotion time (issue #185), never eagerly at construction, so no
    live :class:`~windbreak.config.EvaluationConfig` is consulted to stand a
    kernel up.

    Returns:
        A mapping of :class:`Mode` (``RESEARCH`` and ``LIVE_MICRO`` only) to its
        constant :class:`PromotionGate`.
    """
    return {
        Mode.RESEARCH: _research_to_paper_gate(),
        Mode.LIVE_MICRO: _live_micro_to_live_gate(),
    }


def _evaluate_criterion(
    criterion: GateCriterion, evidence: GateEvidence
) -> CriterionResult:
    """Evaluate one criterion against the evidence via the comparison table.

    Args:
        criterion: The criterion to evaluate.
        evidence: The evidence snapshot to read the observed value from.

    Returns:
        The :class:`CriterionResult`, with bool evidence coerced to ``0``/``1``.
    """
    observed = int(getattr(evidence, criterion.evidence_field))
    comparator = _COMPARATORS[criterion.comparison]
    return CriterionResult(
        criterion_id=criterion.criterion_id,
        observed=observed,
        threshold=criterion.threshold,
        comparison=criterion.comparison,
        passed=comparator(observed, criterion.threshold),
    )


def evaluate_promotion(
    gate: PromotionGate, evidence: GateEvidence
) -> PromotionDecision:
    """Evaluate every criterion of ``gate`` against ``evidence``, no short-circuit.

    Args:
        gate: The promotion gate to evaluate.
        evidence: The evidence snapshot to evaluate it against.

    Returns:
        A :class:`PromotionDecision` with one result per criterion (in gate
        order); ``approved`` is true only if every criterion passed.
    """
    results = tuple(
        _evaluate_criterion(criterion, evidence) for criterion in gate.criteria
    )
    approved = all(result.passed for result in results)
    return PromotionDecision(
        source=gate.source, target=gate.target, approved=approved, results=results
    )


def effective_mode_ceiling(configured: Mode, override_applied: bool) -> Mode:
    """Return the effective ceiling: the ladder-rank MIN of configured and cap.

    Without an override the configured ceiling stands unchanged. With an
    override, the effective ceiling is the lower-ranked of the configured
    ceiling and :data:`OVERRIDE_CEILING` (``LIVE_MICRO``) -- so a ceiling
    already at or below ``LIVE_MICRO`` is unaffected, and only ``LIVE`` is
    capped down.

    Args:
        configured: The statically-configured mode ceiling (a ladder mode).
        override_applied: Whether the significance override is in force.

    Returns:
        The effective ceiling mode.
    """
    if not override_applied:
        return configured
    if _LADDER_RANK[configured] <= _LADDER_RANK[OVERRIDE_CEILING]:
        return configured
    return OVERRIDE_CEILING


def override_applied_in(events: Iterable[Event]) -> bool:
    """Return whether any event marks the significance override as applied.

    Args:
        events: The event history to scan (e.g. for restart replay).

    Returns:
        True if any event's ``event_type`` is ``SignificanceOverrideApplied``.
    """
    return any(event.event_type == _OVERRIDE_EVENT_TYPE for event in events)
