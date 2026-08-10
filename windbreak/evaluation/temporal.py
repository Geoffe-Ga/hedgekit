"""Temporal-integrity enforcement for the evaluation harness (SPEC-EPIC_07, #52).

This module makes training-data leakage into gate metrics *structurally*
impossible. The temporal coordinate is exclusively the monotonic integer
``sequence_number`` on the append-only ledger -- there is no wall-clock and no
float anywhere on the value path, only exact integer comparisons.

A forecast is rejected, at ingestion, when it cannot be an honest prediction:

    * ``PRE_DEPLOYMENT`` -- it has no recorded creation sequence, or its
      creation sequence is at or before the system's deployment sequence.
    * ``BACKDATED`` -- its market has a known resolution sequence ``r`` and the
      forecast was created at or after ``r`` (it could have peeked at the
      answer).
    * ``UNRESOLVED`` -- its market never resolved, so it can never enter a
      headline metric.

The three reasons are checked in exactly that precedence, so each rejected
record is ledgered with exactly one reason. Every rejection is recorded as an
immutable :class:`RejectionEvent` rather than silently dropped.

The intra-package dependency graph stays one-way and acyclic: this module
imports the dependency-free :mod:`windbreak.evaluation.resolution` leaf at
runtime and references :mod:`windbreak.evaluation.registry`'s carrier types only
under :data:`typing.TYPE_CHECKING`. Because it therefore cannot construct an
:class:`~windbreak.evaluation.registry.EvaluationInputs`, the reconstruction seam
that rebuilds admitted inputs lives in the registry, which imports this module
at runtime.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from windbreak.evaluation.resolution import SettlementEventType
from windbreak.timekeeping import require_aware

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from datetime import datetime
    from typing import Any, Final

    from windbreak.evaluation.registry import EvaluationInputs, FixtureForecast
    from windbreak.evaluation.resolution import SettlementEvent

#: The immutable event-type token stamped on every ledgered rejection record.
EVALUATION_RECORD_REJECTED: Final[str] = "EVALUATION_RECORD_REJECTED"

#: The sequence a settlement instant projects to when there is no ledger
#: position to project it onto at all. Real sequence numbers start at ``1``, so
#: ``0`` makes every forecast ``created_sequence >= 0`` -- i.e. BACKDATED --
#: rather than scored against an ordering nothing established (fail closed).
_UNPROVABLE_RESOLUTION_SEQUENCE = 0

#: JSON top-level key holding the ordered mode-transition ledger.
_MODE_TRANSITIONS_KEY = "mode_transitions"
#: JSON field carrying a mode transition's global sequence number.
_SEQUENCE_NUMBER_FIELD = "sequence_number"


class RejectionReason(enum.Enum):
    """Why a forecast was rejected by the temporal-integrity gate.

    The three members are mutually exclusive and exhaustive: there is no
    escape-hatch "skipped" or "ignored" member a caller could select to slip a
    leaked record past the gate. The lowercase token values mirror
    :class:`~windbreak.evaluation.resolution.ResolutionOutcome` and
    :class:`~windbreak.evaluation.resolution.SettlementEventType`.
    """

    BACKDATED = "backdated"
    PRE_DEPLOYMENT = "pre_deployment"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class RejectionEvent:
    """One immutable ledger entry recording a rejected forecast.

    Attributes:
        forecast_id: The rejected forecast's identifier.
        market_ticker: The market the rejected forecast named.
        reason: Why the forecast was rejected.
        created_sequence: The forecast's creation sequence, or ``None`` when it
            carried no recorded provenance.
        deployment_sequence: The system deployment sequence gated against, or
            ``None`` when no temporal context applied.
        resolution_sequence: The market's known resolution sequence, present for
            a ``BACKDATED`` rejection (and for a ``PRE_DEPLOYMENT`` forecast on
            an already-resolved market) and ``None`` for an ``UNRESOLVED`` one.
        event_type: The fixed :data:`EVALUATION_RECORD_REJECTED` token; not a
            constructor parameter, so no caller can forge a different token.
    """

    forecast_id: str
    market_ticker: str
    reason: RejectionReason
    created_sequence: int | None
    deployment_sequence: int | None
    resolution_sequence: int | None
    event_type: str = field(init=False)

    def __post_init__(self) -> None:
        """Stamp the fixed event-type token and enforce coherence.

        ``event_type`` is set here (rather than via a plain ``field`` default)
        so the immutable :data:`EVALUATION_RECORD_REJECTED` token is applied
        reliably under ``slots=True`` and can never be forged at construction.

        Raises:
            ValueError: If a ``BACKDATED`` rejection carries no
                ``resolution_sequence`` (the reason presupposes a known
                resolution), or if an ``UNRESOLVED`` rejection carries one (an
                unresolved market cannot have a resolution sequence); the
                message names the ``resolution_sequence`` field.
        """
        object.__setattr__(self, "event_type", EVALUATION_RECORD_REJECTED)
        if (
            self.reason is RejectionReason.BACKDATED
            and self.resolution_sequence is None
        ):
            raise ValueError(
                "BACKDATED rejection requires a non-None resolution_sequence; "
                "the reason presupposes a known market resolution"
            )
        if (
            self.reason is RejectionReason.UNRESOLVED
            and self.resolution_sequence is not None
        ):
            raise ValueError(
                "UNRESOLVED rejection requires resolution_sequence to be None; "
                "an unresolved market carries no resolution sequence"
            )


@dataclass(frozen=True, slots=True)
class TemporalContext:
    """The temporal coordinates one evaluation run gates its forecasts against.

    Attributes:
        deployment_sequence: The sequence number at or before which a forecast
            is pre-deployment and cannot be honest.
        resolution_sequences: The first-settlement sequence of each resolved
            market, keyed by ``market_ticker``.
    """

    deployment_sequence: int
    resolution_sequences: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class TemporalGateResult:
    """The outcome of gating one run's forecasts for temporal integrity.

    Attributes:
        admitted_forecasts: The forecasts that passed the gate, in fixture
            order.
        rejections: The ledgered rejection events, in fixture order.
    """

    admitted_forecasts: tuple[FixtureForecast, ...]
    rejections: tuple[RejectionEvent, ...]


def _is_pre_deployment(created_sequence: int | None, deployment_sequence: int) -> bool:
    """Report whether a forecast predates (or lacks provenance before) deployment.

    Args:
        created_sequence: The forecast's creation sequence, or ``None``.
        deployment_sequence: The system deployment sequence.

    Returns:
        ``True`` when the creation sequence is missing or at/before deployment.
    """
    return created_sequence is None or created_sequence <= deployment_sequence


def _is_backdated(
    created_sequence: int | None, resolution_sequence: int | None
) -> bool:
    """Report whether a forecast was created at or after its market's resolution.

    Args:
        created_sequence: The forecast's creation sequence, or ``None``.
        resolution_sequence: The market's known resolution sequence, or ``None``
            when the market has never resolved.

    Returns:
        ``True`` only when both sequences are known and creation is at or after
        resolution.
    """
    if created_sequence is None or resolution_sequence is None:
        return False
    return created_sequence >= resolution_sequence


def _classify(
    forecast: FixtureForecast,
    resolutions: Mapping[str, object],
    temporal: TemporalContext,
) -> RejectionReason | None:
    """Classify one forecast against the temporal gate, or admit it.

    The three reasons are evaluated in fixed precedence
    (``PRE_DEPLOYMENT`` > ``BACKDATED`` > ``UNRESOLVED``) so exactly one reason
    is ever assigned to a rejected record.

    Args:
        forecast: The forecast to classify.
        resolutions: The run's ground-truth resolutions, keyed by ticker.
        temporal: The temporal context to gate against.

    Returns:
        The single :class:`RejectionReason` that applies, or ``None`` to admit.
    """
    resolution_sequence = temporal.resolution_sequences.get(forecast.market_ticker)
    if _is_pre_deployment(forecast.created_sequence, temporal.deployment_sequence):
        return RejectionReason.PRE_DEPLOYMENT
    if _is_backdated(forecast.created_sequence, resolution_sequence):
        return RejectionReason.BACKDATED
    if forecast.market_ticker not in resolutions:
        return RejectionReason.UNRESOLVED
    return None


def _build_rejection(
    forecast: FixtureForecast,
    reason: RejectionReason,
    temporal: TemporalContext,
) -> RejectionEvent:
    """Build the ledger entry for one rejected forecast.

    Args:
        forecast: The rejected forecast.
        reason: The reason it was rejected.
        temporal: The temporal context it was gated against.

    Returns:
        The immutable :class:`RejectionEvent`, carrying the market's known
        resolution sequence (``None`` for an unresolved market). An
        ``UNRESOLVED`` rejection always carries ``None`` -- even when the
        settlement-derived ``resolution_sequences`` map holds a sequence for
        the ticker -- so the event stays coherent with
        :class:`RejectionEvent`'s guard and is ledgered, never raised.
    """
    resolution_sequence = (
        None
        if reason is RejectionReason.UNRESOLVED
        else temporal.resolution_sequences.get(forecast.market_ticker)
    )
    return RejectionEvent(
        forecast_id=forecast.forecast_id,
        market_ticker=forecast.market_ticker,
        reason=reason,
        created_sequence=forecast.created_sequence,
        deployment_sequence=temporal.deployment_sequence,
        resolution_sequence=resolution_sequence,
    )


def _require_temporal_context(inputs: EvaluationInputs) -> TemporalContext:
    """Return the run's temporal context, failing closed when it is absent.

    Args:
        inputs: The evaluation inputs whose temporal context is required.

    Returns:
        The non-``None`` :class:`TemporalContext`.

    Raises:
        ValueError: If ``inputs.temporal`` is ``None`` while forecasts are
            present -- the gate never silently skips a populated run.
    """
    temporal = inputs.temporal
    if temporal is None:
        raise ValueError(
            "enforce_temporal_integrity requires a TemporalContext to gate "
            f"{len(inputs.forecasts)} forecast(s); inputs.temporal was None"
        )
    return temporal


def enforce_temporal_integrity(inputs: EvaluationInputs) -> TemporalGateResult:
    """Gate a run's forecasts for temporal integrity, ledgering every rejection.

    This is the single choke point through which forecasts must pass before any
    metric scores them. It takes exactly one argument and has no bypass flag, so
    a caller cannot opt out of the gate.

    Args:
        inputs: The evaluation inputs to gate. When ``inputs.temporal`` is
            ``None`` and there are no forecasts, the empty result is returned;
            when it is ``None`` and forecasts are present, a
            :class:`ValueError` is raised (fail-closed, never a silent skip).

    Returns:
        A :class:`TemporalGateResult` carrying the admitted forecasts and the
        rejection ledger, both in fixture order.

    Raises:
        ValueError: If ``inputs.temporal`` is ``None`` while forecasts are
            present.
    """
    if inputs.temporal is None and not inputs.forecasts:
        return TemporalGateResult(admitted_forecasts=(), rejections=())
    temporal = _require_temporal_context(inputs)
    admitted: list[FixtureForecast] = []
    rejections: list[RejectionEvent] = []
    for forecast in inputs.forecasts:
        reason = _classify(forecast, inputs.resolutions, temporal)
        if reason is None:
            admitted.append(forecast)
        else:
            rejections.append(_build_rejection(forecast, reason, temporal))
    return TemporalGateResult(
        admitted_forecasts=tuple(admitted), rejections=tuple(rejections)
    )


def resolution_sequences_from_events(
    events: Iterable[SettlementEvent],
) -> Mapping[str, int]:
    """Fold a settlement stream into each market's first-settlement sequence.

    The temporal answer -- "when could this market's outcome first have been
    known" -- is fixed at a market's *first-ever* ``SETTLEMENT`` event.
    ``SETTLEMENT_REVERSED`` events are ignored for positioning, and a later
    re-settlement never overwrites the first, so a reversed-and-resettled market
    keeps its original sequence.

    Args:
        events: The settlement events, in stream order.

    Returns:
        A mapping from ``market_ticker`` to its first-settlement sequence
        number; an empty stream yields an empty mapping.
    """
    sequences: dict[str, int] = {}
    for event in events:
        is_first_settlement = (
            event.event_type is SettlementEventType.SETTLEMENT
            and event.market_ticker not in sequences
        )
        if is_first_settlement:
            sequences[event.market_ticker] = event.sequence_number
    return sequences


def _first_position_at_or_after(
    positions: tuple[tuple[int, datetime], ...], instant: datetime
) -> int:
    """Return the earliest ledger position whose instant is at or after ``instant``.

    Args:
        positions: The ledger's ``(sequence_number, created_at)`` pairs, sorted
            ascending by sequence number.
        instant: The settlement instant to locate.

    Returns:
        The sequence number of the first position at or after ``instant``; one
        past the ledger's head when the instant postdates every position
        (nothing recorded could have known the answer yet); and
        :data:`_UNPROVABLE_RESOLUTION_SEQUENCE` when there are no positions at
        all.
    """
    for sequence_number, moment in positions:
        if moment >= instant:
            return sequence_number
    if not positions:
        return _UNPROVABLE_RESOLUTION_SEQUENCE
    return positions[-1][0] + 1


def resolution_sequences_from_instants(
    ledger_instants: Iterable[tuple[int, datetime]],
    resolution_instants: Mapping[str, datetime],
) -> Mapping[str, int]:
    """Project wall-clock settlement instants onto the ledger's sequence axis.

    The temporal gate's coordinate is the monotonic integer ``sequence_number``
    -- exact, float-free, and unfakeable. A resolution ingested from the outside
    world carries a wall-clock instant instead, so the two must be reconciled
    before the gate can adjudicate anything. The reconciliation is: a market's
    answer could first have been known at the earliest ledger position stamped
    at or after the instant it settled.

    Doing it this way is what makes a *late* ingest safe. Keying the gate on the
    position of the ``MarketResolved`` row itself would place the answer at the
    moment the operator noticed rather than the moment the market settled, and
    every forecast made in between -- each of which could have peeked -- would
    be silently admitted and scored. This function ignores where the resolution
    row landed entirely.

    Args:
        ledger_instants: The ledger's ``(sequence_number, created_at)`` pairs.
            Order is immaterial; they are sorted by sequence number here.
        resolution_instants: Each resolved market's settlement instant, keyed by
            ``market_ticker``.

    Returns:
        A mapping from ``market_ticker`` to the sequence number at or after
        which its answer was knowable; empty when no market resolved.

    Raises:
        ValueError: If any ledger instant or settlement instant carries no UTC
            offset. Comparing an offsetless instant would silently shift it by
            the host's offset -- and this comparison decides whether a forecast
            is scored or refused -- so it is refused rather than normalized
            against a guessed timezone.
    """
    positions = tuple(sorted(ledger_instants))
    for _, moment in positions:
        require_aware(moment, "ledger_instant")
    sequences: dict[str, int] = {}
    for market_ticker, instant in resolution_instants.items():
        require_aware(instant, "resolution_instant")
        sequences[market_ticker] = _first_position_at_or_after(positions, instant)
    return sequences


def _transition_sequence(entry: Mapping[str, Any]) -> int:
    """Extract and validate one mode transition's sequence number.

    Args:
        entry: One decoded ``mode_transitions`` entry.

    Returns:
        The entry's ``sequence_number``.

    Raises:
        TypeError: If the ``sequence_number`` is a ``bool`` (an ``int`` subclass
            that must not masquerade as a position) or is not an ``int`` at all
            -- mirroring the ``_IntUnit`` guard; the message names the
            ``sequence_number`` field.
    """
    number = entry[_SEQUENCE_NUMBER_FIELD]
    if isinstance(number, bool) or not isinstance(number, int):
        raise TypeError(
            f"sequence_number requires a non-bool int, got {type(number).__name__}"
        )
    validated: int = number
    return validated


def deployment_sequence_from_fixture(fixture: Mapping[str, Any]) -> int:
    """Read the deployment sequence from a fixture's ``mode_transitions`` block.

    Deployment is the earliest transition ever recorded, so the deployment
    sequence is the minimum ``sequence_number`` across all transitions,
    regardless of list order.

    Args:
        fixture: The decoded fixture payload, carrying a non-empty
            ``mode_transitions`` list.

    Returns:
        The minimum ``sequence_number`` among the mode transitions.

    Raises:
        ValueError: If the ``mode_transitions`` key is missing or its list is
            empty -- there is no deployment point to gate against; the message
            names the ``mode_transitions`` block.
        TypeError: If any ``sequence_number`` is a ``bool`` or not an ``int``.
    """
    transitions = fixture.get(_MODE_TRANSITIONS_KEY)
    if not transitions:
        raise ValueError(
            f"fixture requires a non-empty {_MODE_TRANSITIONS_KEY!r} block; "
            "there is no deployment sequence to gate against"
        )
    return min(_transition_sequence(entry) for entry in transitions)
