"""Per-provider canary battery runner (SPEC S8.4/S16, issue #195).

The pinned-canary-model gate (:mod:`windbreak.forecast.canary`) checks one
global model for silent drift. This module layers a *fleet-observability*
battery on top: one :class:`ProviderCanarySpec` per forecast provider, each
paired with a small, injected :class:`ProviderCanaryObserver` that gathers one
:class:`ProviderCanaryObservation` (its per-question observed ppm and reported
forecaster version). :func:`run_provider_canaries` reduces the battery to one
:class:`ProviderCanaryVerdict` per provider, driving the per-provider extension
of :class:`~windbreak.forecast.canary.CanaryGate`:

* a reported version off the provider's pinned set is a ``VERSION_DRIFT`` breach
  (via :meth:`~windbreak.forecast.canary.CanaryGate.apply_version_drift`);
* an observed-probability distance past tolerance is an ``ANSWER_DRIFT`` breach
  (via :func:`~windbreak.forecast.canary.score_canary_run` +
  :meth:`~windbreak.forecast.canary.CanaryGate.apply_run`);
* an observer that raises ``ProviderVersionDriftError`` is treated as
  ``VERSION_DRIFT``, fail-closed, without ever scoring answer drift over a
  (nonexistent) observation.

``VERSION_DRIFT`` takes status priority over ``ANSWER_DRIFT``. The module is
float-free and, per the SPEC S8.3 sandbox boundary, imports neither
``windbreak.config`` nor ``windbreak.ledger`` -- the forecast-to-ledger bridge
lives in the :mod:`windbreak.scheduler.canaries` composition root.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from windbreak.forecast.canary import score_canary_run
from windbreak.forecast.providers.base import ProviderVersionDriftError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from windbreak.forecast.canary import (
        CanaryAlertEmitter,
        CanaryGate,
        CanaryLedgerWriter,
        CanaryQuestion,
    )


@dataclass(frozen=True, slots=True)
class ProviderCanaryObservation:
    """One provider's observed canary answers and reported version.

    Attributes:
        observed_ppm: The per-question observed probability, in ppm, keyed by
            question id (integer leaves only, never a float).
        reported_version: The forecaster version the provider reported for this
            observation, checked against the spec's pinned set.
    """

    observed_ppm: Mapping[str, int]
    reported_version: str


class ProviderCanaryObserver(Protocol):
    """The seam that gathers one provider's canary observation.

    A concrete observer wraps whatever transport reaches a hosted forecaster or
    an LLM-vote member; the offline tests inject small fakes. An observer may
    fail closed by raising
    :class:`~windbreak.forecast.providers.base.ProviderVersionDriftError`, which
    :func:`run_provider_canaries` treats as version drift.
    """

    def observe(self, spec: ProviderCanarySpec) -> ProviderCanaryObservation:
        """Gather one observation for ``spec``'s provider.

        Args:
            spec: The provider canary spec being observed.

        Returns:
            The provider's observation.

        Raises:
            ProviderVersionDriftError: If the provider reports a forecaster
                version off its pinned set (fail-closed version drift).
        """
        ...


@dataclass(frozen=True, slots=True)
class ProviderCanarySpec:
    """One provider's canary battery specification.

    Attributes:
        provider: The provider identifier this spec checks.
        questions: The reference questions this provider is scored against.
        pinned_versions: The operator-pinned forecaster versions considered
            valid for this provider (plural: a pin set may accept more than one).
        observer: The observer gathering this provider's observation.
    """

    provider: str
    questions: tuple[CanaryQuestion, ...]
    pinned_versions: tuple[str, ...]
    observer: ProviderCanaryObserver


class ProviderCanaryStatus(Enum):
    """The graded outcome of one provider's canary battery run.

    Attributes:
        OK: Neither version nor answer drift; the provider stays live-eligible.
        ANSWER_DRIFT: An observed-probability distance breached tolerance.
        VERSION_DRIFT: A reported version was off the pinned set (takes priority
            over answer drift).
    """

    OK = "OK"
    ANSWER_DRIFT = "ANSWER_DRIFT"
    VERSION_DRIFT = "VERSION_DRIFT"


@dataclass(frozen=True, slots=True)
class ProviderCanaryVerdict:
    """One provider's reduced canary verdict.

    Attributes:
        provider: The provider this verdict is for.
        status: The graded :class:`ProviderCanaryStatus`.
        drift_score_ppm: The scored answer-drift distance, in ppm (``0`` when no
            observation was scored, e.g. a fail-closed version drift).
        worst_question_id: The id of the worst-scoring question, or ``""`` when
            no observation was scored.
        reported_version: The forecaster version the provider reported.
    """

    provider: str
    status: ProviderCanaryStatus
    drift_score_ppm: int
    worst_question_id: str
    reported_version: str


def _run_one_spec(
    spec: ProviderCanarySpec,
    *,
    gate: CanaryGate,
    alerts: CanaryAlertEmitter,
    ledger: CanaryLedgerWriter,
    checked_at: datetime,
) -> ProviderCanaryVerdict:
    """Run one provider's canary spec into its verdict.

    An observer that raises
    :class:`~windbreak.forecast.providers.base.ProviderVersionDriftError` is
    treated as fail-closed version drift: the gate is driven by the error's
    reported/pinned versions and no answer scoring is attempted. Otherwise the
    reported version is gated first (version drift takes priority) and the
    observed answers are scored second.

    Args:
        spec: The provider canary spec to run.
        gate: The per-provider canary gate to drive.
        alerts: The alert emitter breaches dispatch through.
        ledger: The canary-event ledger writer.
        checked_at: When the battery was checked.

    Returns:
        The provider's reduced verdict.
    """
    try:
        observation = spec.observer.observe(spec)
    except ProviderVersionDriftError as error:
        gate.apply_version_drift(
            spec.provider,
            error.reported_version,
            error.pinned_versions,
            checked_at=checked_at,
            alerts=alerts,
            ledger=ledger,
        )
        return ProviderCanaryVerdict(
            provider=spec.provider,
            status=ProviderCanaryStatus.VERSION_DRIFT,
            drift_score_ppm=0,
            worst_question_id="",
            reported_version=error.reported_version,
        )
    version_drifted = gate.apply_version_drift(
        spec.provider,
        observation.reported_version,
        spec.pinned_versions,
        checked_at=checked_at,
        alerts=alerts,
        ledger=ledger,
    )
    scored = score_canary_run(spec.questions, observation.observed_ppm)
    answer_drifted = gate.apply_run(
        scored,
        checked_at=checked_at,
        alerts=alerts,
        ledger=ledger,
        provider=spec.provider,
    )
    status = _verdict_status(
        version_drifted=version_drifted, answer_drifted=answer_drifted
    )
    return ProviderCanaryVerdict(
        provider=spec.provider,
        status=status,
        drift_score_ppm=scored.drift_score_ppm,
        worst_question_id=scored.worst_question_id,
        reported_version=observation.reported_version,
    )


def _verdict_status(
    *, version_drifted: bool, answer_drifted: bool
) -> ProviderCanaryStatus:
    """Reduce the two breach flags to a single status (version takes priority).

    Args:
        version_drifted: Whether the reported version breached its pin.
        answer_drifted: Whether the observed answers breached tolerance.

    Returns:
        ``VERSION_DRIFT`` if the version drifted, else ``ANSWER_DRIFT`` if the
        answers drifted, else ``OK``.
    """
    if version_drifted:
        return ProviderCanaryStatus.VERSION_DRIFT
    if answer_drifted:
        return ProviderCanaryStatus.ANSWER_DRIFT
    return ProviderCanaryStatus.OK


def run_provider_canaries(
    specs: tuple[ProviderCanarySpec, ...],
    *,
    gate: CanaryGate,
    alerts: CanaryAlertEmitter,
    ledger: CanaryLedgerWriter,
    checked_at: datetime,
) -> tuple[ProviderCanaryVerdict, ...]:
    """Run a provider canary battery, one verdict per provider (SPEC S8.4/S16).

    Each spec is observed and reduced to a :class:`ProviderCanaryVerdict` in
    spec order, driving the per-provider :class:`~windbreak.forecast.canary.CanaryGate`
    so a drifting provider is blocked from live eligibility while its siblings
    stay open. An all-clean provider ledgers one ``CANARY_OK`` (inside
    :meth:`~windbreak.forecast.canary.CanaryGate.apply_run`) and dispatches no
    alert.

    Args:
        specs: The provider canary specs to run, one per provider.
        gate: The per-provider canary gate to drive (keyword-only).
        alerts: The alert emitter breaches dispatch through (keyword-only).
        ledger: The canary-event ledger writer (keyword-only).
        checked_at: When the battery was checked (keyword-only).

    Returns:
        One verdict per spec, in spec order.
    """
    return tuple(
        _run_one_spec(
            spec, gate=gate, alerts=alerts, ledger=ledger, checked_at=checked_at
        )
        for spec in specs
    )
