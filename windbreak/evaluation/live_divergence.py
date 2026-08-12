"""Per-run live-vs-paper divergence monitor (issue #58, SPEC §10.9/§10.10).

:func:`monitor_live_divergence` scores the two live-divergence series --
``live_slippage_ratio`` (cost slippage of real fills vs the paper model) and
``live_brier_degradation`` (rolling LIVE-over-PAPER forecast-skill decay) --
against a pre-registered :class:`~windbreak.evaluation.preregistration.GatePlan`'s
thresholds, and turns a breach into a call on the ``fire_trigger`` seam its
caller injects.

**Nothing under ``windbreak/`` calls :func:`monitor_live_divergence` today**, and
nothing passes it a kernel method, so no shipped path turns a divergence breach
into an actual demotion: the monitor is exercised only by the test suite (issue
#542). Whether this monitor should demote the kernel automatically, or surface
the breach for an operator to act on, is an open decision recorded on #542;
until it is made, read every "demotes" below as "would demote, once wired".

Every call appends exactly one :class:`LiveDivergenceSampled` (both series
values -- sentinels rendered by name -- thresholds, window size, cohort counts,
and ``plan_hash``), regardless of outcome. Each series that breaches its
threshold appends exactly one :class:`LiveDivergenceBreached` (the same snapshot
plus the firing trigger name), fires exactly one ``AlertSeverity.CRITICAL``
alert, and calls ``fire_trigger`` with the matching
:class:`~windbreak.riskkernel.demotion.DemotionTrigger`
(``LIVE_PAPER_SLIPPAGE_DIVERGENCE`` for slippage, ``ROLLING_BRIER_DEGRADATION``
for the Brier band). Both breaching in one run fires both triggers -- the
fail-safe "double one-rung demotion" reading -- never a single conflated event.

A PAPER-only run (no LIVE forecast, no execution record) yields both series
``UNDEFINED``, appends only the sampled event, and fires nothing: the demoability
tracer's ordinary early-deployment state. A recorded fill whose ``model_version``
disagrees with the plan's ``paper_fill_model_version`` fails closed
(:class:`ValueError`) before anything is ledgered.

The two typed events derive their base
:class:`~windbreak.ledger.events.Event` fields through a LOCAL
:func:`_derive_typed_event` (the house pattern from
:mod:`windbreak.evaluation.preregistration` / :mod:`windbreak.evaluation.crosscheck`),
so this issue never touches the ledger's central ``EVENT_TYPES`` map. Only the
leaf :class:`~windbreak.riskkernel.demotion.DemotionTrigger` is imported from the
risk-kernel package, keeping the edge thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from windbreak.alerts.registry import AlertSeverity
from windbreak.evaluation.cohorts import UndefinedBrier
from windbreak.evaluation.execution_quality import require_model_version
from windbreak.evaluation.registry import NotImplementedSentinel, registered_metrics
from windbreak.ledger.events import Event
from windbreak.riskkernel.demotion import DemotionTrigger

if TYPE_CHECKING:
    from windbreak.evaluation.preregistration import GatePlan
    from windbreak.evaluation.registry import EvaluationInputs, MetricValue
    from windbreak.ledger.store import LedgerStore
    from windbreak.riskkernel.modes import Mode

#: Payload schema version stamped on this module's events. Replicated locally
#: (rather than imported from :mod:`windbreak.ledger.events`'s private copy, out
#: of this issue's scope) so a payload-shape change here can be versioned
#: independently.
_SCHEMA_VERSION = 1

#: The registry name of the cost-slippage series this monitor scores.
_SLIPPAGE_RATIO_METRIC = "live_slippage_ratio"

#: The registry name of the rolling Brier-degradation series this monitor scores.
_BRIER_DEGRADATION_METRIC = "live_brier_degradation"


class AlertHook(Protocol):
    """A sink the monitor fires a severity/message pair into on a breach."""

    def __call__(self, severity: AlertSeverity, message: str) -> None:
        """Emit one alert.

        Args:
            severity: The alert's severity.
            message: The alert's human-readable message.
        """


class DemotionFirer(Protocol):
    """A one-argument callable firing a demotion trigger.

    Structurally matches
    :meth:`windbreak.riskkernel.process.RiskKernel.fire_demotion_trigger`, so a
    bound kernel method *would* pass directly, while a test double is an
    ordinary callable -- keeping the risk-kernel edge a thin, leaf dependency.
    No production code currently supplies either: the only implementations are
    test doubles (issue #542).
    """

    def __call__(self, trigger: DemotionTrigger) -> Mode | None:
        """Fire one demotion trigger.

        Args:
            trigger: The demotion trigger to fire.

        Returns:
            The resolved destination mode, or ``None`` for a no-op firing.
        """


def _render_metric(value: MetricValue) -> object:
    """Render a metric value for a JSON-safe payload entry.

    Args:
        value: The computed series value from the reference path.

    Returns:
        The sentinel's ``.name`` for a sentinel (e.g. ``"UNDEFINED"``), else the
        ``int`` itself -- mirroring :func:`windbreak.evaluation.crosscheck`'s
        ``_render_value``.
    """
    if isinstance(value, (UndefinedBrier, NotImplementedSentinel)):
        return value.name
    return value


@dataclass(frozen=True, slots=True)
class _DivergenceSnapshot:
    """One run's two series values plus the thresholds and cohort counts.

    Attributes:
        slippage_ratio: The ``live_slippage_ratio`` value (``int`` or a sentinel).
        brier_degradation: The ``live_brier_degradation`` value (``int`` or a
            sentinel).
        slippage_limit_ppm: The plan's slippage-ratio ceiling, in ppm.
        brier_band_ppm: The plan's Brier-degradation band, in ppm.
        rolling_window_size: The plan's rolling-window size.
        execution_record_count: Number of execution-quality records this run saw.
        live_forecast_count: Number of LIVE-track forecasts this run saw.
        paper_forecast_count: Number of PAPER-track forecasts this run saw.
        plan_hash: The scored gate plan's content hash.
    """

    slippage_ratio: MetricValue
    brier_degradation: MetricValue
    slippage_limit_ppm: int
    brier_band_ppm: int
    rolling_window_size: int
    execution_record_count: int
    live_forecast_count: int
    paper_forecast_count: int
    plan_hash: str

    def payload(self) -> dict[str, object]:
        """Return the JSON-safe sampled/breached snapshot payload.

        Returns:
            A fresh dict naming both series values (sentinels rendered by name),
            the two thresholds, the window size, the three cohort counts, and the
            plan hash.
        """
        return {
            "live_slippage_ratio_ppm": _render_metric(self.slippage_ratio),
            "live_brier_degradation_ppm": _render_metric(self.brier_degradation),
            "live_slippage_ratio_limit_ppm": self.slippage_limit_ppm,
            "live_brier_degradation_band_ppm": self.brier_band_ppm,
            "live_rolling_window_size": self.rolling_window_size,
            "execution_record_count": self.execution_record_count,
            "live_forecast_count": self.live_forecast_count,
            "paper_forecast_count": self.paper_forecast_count,
            "plan_hash": self.plan_hash,
        }


def _derive_typed_event(event: Event, payload: dict[str, object]) -> None:
    """Populate the derived :class:`~windbreak.ledger.events.Event` fields.

    Replicates the ledger module's private derivation locally (its ``EVENT_TYPES``
    map is out of this issue's scope): sets ``event_type`` to the concrete class
    name, ``payload_schema_version`` to this module's schema version, and
    ``payload`` to the assembled dict, via ``object.__setattr__`` because the
    events are frozen.

    Args:
        event: The freshly constructed typed event to populate.
        payload: The type-specific payload assembled by the subclass.
    """
    object.__setattr__(event, "event_type", type(event).__name__)
    object.__setattr__(event, "payload_schema_version", _SCHEMA_VERSION)
    object.__setattr__(event, "payload", payload)


@dataclass(frozen=True)
class LiveDivergenceSampled(Event):
    """Records one live-divergence sample (appended on every monitor run).

    Attributes:
        sample: The snapshot payload naming both series values, thresholds,
            window size, cohort counts, and plan hash.
    """

    sample: dict[str, object]
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload and derive the base ``Event`` fields."""
        _derive_typed_event(self, dict(self.sample))


@dataclass(frozen=True)
class LiveDivergenceBreached(Event):
    """Records one breached live-divergence series (one per firing trigger).

    Attributes:
        sample: The same snapshot payload the sampled event carried.
        trigger: The firing :class:`~windbreak.riskkernel.demotion.DemotionTrigger`'s
            name.
    """

    sample: dict[str, object]
    trigger: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Assemble the payload (snapshot + trigger) and derive base fields."""
        payload: dict[str, object] = {**self.sample, "trigger": self.trigger}
        _derive_typed_event(self, payload)


def _sample_divergence(inputs: EvaluationInputs, plan: GatePlan) -> _DivergenceSnapshot:
    """Score both series and capture the run's snapshot.

    Args:
        inputs: The (temporally-admitted) evaluation inputs to score.
        plan: The gate plan supplying the thresholds and window size.

    Returns:
        The populated :class:`_DivergenceSnapshot`.
    """
    specs = registered_metrics()
    return _DivergenceSnapshot(
        slippage_ratio=specs[_SLIPPAGE_RATIO_METRIC].compute(inputs),
        brier_degradation=specs[_BRIER_DEGRADATION_METRIC].compute(inputs),
        slippage_limit_ppm=plan.live_slippage_ratio_limit_ppm,
        brier_band_ppm=plan.live_brier_degradation_band_ppm,
        rolling_window_size=plan.live_rolling_window_size,
        execution_record_count=len(inputs.execution_records),
        live_forecast_count=sum(1 for forecast in inputs.forecasts if forecast.live),
        paper_forecast_count=sum(
            1 for forecast in inputs.forecasts if not forecast.live
        ),
        plan_hash=plan.plan_hash,
    )


def _exceeds(value: MetricValue, limit: int) -> bool:
    """Return whether a real ``int`` series value strictly exceeds a limit.

    Args:
        value: The series value (``int`` or a sentinel).
        limit: The threshold to compare against.

    Returns:
        ``True`` only when ``value`` is a real (non-``bool``) ``int`` strictly
        greater than ``limit``; a sentinel (``UNDEFINED``) never breaches. The
        comparison is strict ``>``, so a series value exactly equal to its
        threshold does NOT breach (the limit is the last passing value).
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > limit


def _breached_triggers(
    snapshot: _DivergenceSnapshot, plan: GatePlan
) -> list[DemotionTrigger]:
    """Return the demotion triggers the snapshot's breached series demand.

    Args:
        snapshot: The scored run snapshot.
        plan: The gate plan supplying the thresholds.

    Returns:
        The matching triggers, in series order (slippage before Brier); empty
        when neither series breaches.
    """
    triggers: list[DemotionTrigger] = []
    if _exceeds(snapshot.slippage_ratio, plan.live_slippage_ratio_limit_ppm):
        triggers.append(DemotionTrigger.LIVE_PAPER_SLIPPAGE_DIVERGENCE)
    if _exceeds(snapshot.brier_degradation, plan.live_brier_degradation_band_ppm):
        triggers.append(DemotionTrigger.ROLLING_BRIER_DEGRADATION)
    return triggers


def _emit_breach(
    trigger: DemotionTrigger,
    snapshot: _DivergenceSnapshot,
    store: LedgerStore,
    alert: AlertHook,
    fire_trigger: DemotionFirer,
    component: str,
) -> None:
    """Ledger one breach, fire one critical alert, then fire the trigger.

    The order -- breach event, alert, then ``fire_trigger`` -- lands the
    ``LiveDivergenceBreached`` immediately before the kernel's resulting
    ``DemotionTriggerFired`` in one ordered ledger.

    Args:
        trigger: The demotion trigger the breached series demands.
        snapshot: The scored run snapshot the breach is stamped with.
        store: The append-only ledger the breach event is written to.
        alert: The sink the single critical alert is fired into.
        fire_trigger: The demotion firer supplied by the caller. No production
            caller exists today, so in the shipped system this is never the
            kernel's own method (issue #542).
        component: The producing component recorded on the event.
    """
    store.append(
        LiveDivergenceBreached(
            component=component, sample=snapshot.payload(), trigger=trigger.name
        )
    )
    alert(
        AlertSeverity.CRITICAL,
        f"live-vs-paper divergence breach: {trigger.name}",
    )
    fire_trigger(trigger)


def monitor_live_divergence(
    inputs: EvaluationInputs,
    *,
    plan: GatePlan,
    store: LedgerStore,
    alert: AlertHook,
    fire_trigger: DemotionFirer,
    component: str,
) -> None:
    """Score the live-divergence series and fire ``fire_trigger`` on any breach.

    Args:
        inputs: The (temporally-admitted) evaluation inputs to score.
        plan: The pre-registered gate plan supplying the thresholds and window.
        store: The append-only ledger the sampled and breach events are written
            to.
        alert: The sink one critical alert is fired into per breached series.
        fire_trigger: The demotion firer invoked once per breached series. It
            would be the kernel's ``fire_demotion_trigger`` in a wired
            deployment; no production code calls this function or supplies that
            method today, so the demotion does not happen in the shipped system
            (issue #542).
        component: The producing component recorded on the appended events.

    Raises:
        ValueError: If any execution record's ``model_version`` disagrees with
            ``plan.paper_fill_model_version`` -- the run fails closed before
            anything is ledgered.
    """
    require_model_version(inputs.execution_records, plan.paper_fill_model_version)
    snapshot = _sample_divergence(inputs, plan)
    store.append(LiveDivergenceSampled(component=component, sample=snapshot.payload()))
    for trigger in _breached_triggers(snapshot, plan):
        _emit_breach(trigger, snapshot, store, alert, fire_trigger, component)
