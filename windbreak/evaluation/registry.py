"""Metric registry and typed inputs for the evaluation harness (#49, #51, #53).

This module owns SPEC-EPIC_07's three-track evaluation vocabulary -- the
:class:`Track` taxonomy, the typed :class:`FixtureForecast` /
:class:`EvaluationInputs` carriers, the :class:`MetricSpec` shape, and the
:func:`registered_metrics` catalogue -- and wires each spec's ``compute`` to its
real arithmetic. :class:`ObservationWindow` is re-exported from its canonical
home in :mod:`windbreak.evaluation.windows` (#53), so
``registry.ObservationWindow is windows.ObservationWindow`` and every existing
import site keeps working unchanged.

As of issue #51 the seven forecast-track metrics (``brier``,
``brier_skill_vs_executable_price``, ``log_score``,
``expected_calibration_error``, ``calibration_slope``,
``calibration_intercept``, ``sharpness``) delegate to
:mod:`windbreak.evaluation.metrics`; issue #53 makes
``traded_vs_skipped_brier_delta`` a real computation too, delegating to
:mod:`windbreak.evaluation.cohorts`. These are one-way runtime edges with no
cycle: ``metrics`` and ``cohorts`` reference this module's types only under
``TYPE_CHECKING``.

One metric remains a deliberate stub whose ``compute`` returns the
:data:`NOT_IMPLEMENTED` sentinel (a distinct :class:`NotImplementedSentinel`
value, never ``None`` and never a stray ``int``) so the renderer prints the
literal ``NOT_IMPLEMENTED`` rather than omitting the row: the execution-track
``fill_vs_model_slippage``.

A fully-implemented metric can still be *undefined* for a given input rather
than unimplemented: ``traded_vs_skipped_brier_delta`` is undefined whenever the
window holds zero ``TRADED`` or zero ``SKIPPED`` forecasts (an ordinary
early-deployment state), so its adapter catches
:class:`~windbreak.evaluation.cohorts.EmptyCohortError` and returns the distinct
:data:`~windbreak.evaluation.cohorts.UNDEFINED` sentinel -- keeping "not yet
built" (``NOT_IMPLEMENTED``) and "built but undefined here" (``UNDEFINED``)
nominally separate, and never letting the empty-cohort case crash the report.
"""

from __future__ import annotations

import enum
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import windbreak.evaluation.cohorts as cohorts
import windbreak.evaluation.execution_quality as execution_quality
import windbreak.evaluation.metrics as metrics
from windbreak.evaluation.temporal import enforce_temporal_integrity
from windbreak.evaluation.windows import (
    HEADLINE_OBSERVATION_WINDOW,
    ObservationWindow,
    resolve_window,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Final

    from windbreak.evaluation.execution_quality import ExecutionQualityRecord
    from windbreak.evaluation.resolution import ResolutionOutcome
    from windbreak.evaluation.temporal import RejectionEvent, TemporalContext
    from windbreak.numeric.types import ProbabilityPpm

#: The rolling-window size the live-divergence metrics truncate their most-recent
#: cohort to, keyed off ``created_sequence`` descending. This constant is the
#: single source of truth for the window: the SQL reproduction binds the plan's
#: ``live_rolling_window_size`` into its ``LIMIT``, and
#: :func:`~windbreak.evaluation.preregistration.build_gate_plan` validates
#: fail-closed that ``EvaluationConfig.live_rolling_window_size`` equals this
#: constant, so the Python reference and the SQL reproduction always score the
#: identical window (#58). Changing the window is a code change to this constant,
#: which re-registers the pre-registered gate plan (§13.6 clock reset) -- it is
#: not a freely operator-tunable config knob.
LIVE_ROLLING_WINDOW_SIZE: int = 100

#: Inclusive lower bound of a valid ``probability_ppm`` (0.0 as parts-per-million).
_PROBABILITY_PPM_MIN = 0
#: Inclusive upper bound of a valid ``probability_ppm`` (1.0 as parts-per-million).
_PROBABILITY_PPM_MAX = 1_000_000


class Track(enum.Enum):
    """The three evaluation tracks a metric can belong to (SPEC-EPIC_07).

    Defined in report order -- forecast quality first, then selection quality,
    then execution quality -- so iterating the enum yields the fixed section
    order the renderer emits.
    """

    FORECAST = "forecast"
    SELECTION = "selection"
    EXECUTION = "execution"


class NotImplementedSentinel(enum.Enum):
    """Single-valued sentinel marking a metric whose ``compute`` is a stub.

    A dedicated enum (rather than ``None`` or a magic string) keeps an
    unimplemented metric value nominally distinct from every real ``int``
    measurement, so ``value is NOT_IMPLEMENTED`` is an unambiguous test and the
    renderer can print the literal ``NOT_IMPLEMENTED`` for it.
    """

    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


#: The sentinel value returned by every not-yet-implemented metric ``compute``.
NOT_IMPLEMENTED: Final = NotImplementedSentinel.NOT_IMPLEMENTED

#: A computed metric value: a ppm-scaled ``int`` measurement, the
#: :data:`NOT_IMPLEMENTED` sentinel when the metric's arithmetic is still a stub,
#: or the :data:`~windbreak.evaluation.cohorts.UNDEFINED` sentinel when a metric
#: is implemented but genuinely undefined for the given inputs (e.g. an empty
#: cohort). All three are nominally distinct enums, so ``value is <sentinel>`` is
#: an unambiguous test and the renderer prints the literal sentinel name.
MetricValue = int | NotImplementedSentinel | cohorts.UndefinedBrier


@dataclass(frozen=True, slots=True)
class FixtureForecast:
    """One forecast row from a known-answer evaluation fixture.

    Carries the fields the evaluation metrics read for a single forecast: its
    identity, the market it named, its probability, whether it was eligible for
    and actually taken as a live trade, any abstention reason, and the baseline
    executable price the skill metric compares against.

    Attributes:
        forecast_id: Stable identifier of the forecast record.
        market_ticker: Ticker of the market this forecast is about.
        probability_ppm: Forecast probability in parts-per-million (0..1_000_000).
        eligible_for_live: Whether the forecast passed live-eligibility gates.
        abstention_reason: Why the forecast was skipped, or ``None`` if traded.
        traded: Whether a live trade was actually taken on this forecast.
        baseline_executable_price_pips: Reference executable price, in pips,
            the skill metric measures the forecast against.
        correlation_group_id: Identifier of the correlation cluster this
            forecast belongs to, or ``None`` when the market is its own
            singleton cluster (the clustered-bootstrap resampling unit, #51).
        created_sequence: The forecast's creation sequence on the append-only
            ledger, or ``None`` when it carried no recorded provenance (which
            the temporal gate treats as fail-closed pre-deployment, #52).
        live: Whether this forecast is on the LIVE track (vs the PAPER track).
            Defaults to ``False`` so every pre-#58 construction stays valid; the
            live-divergence metrics partition on this flag (#58).
    """

    forecast_id: str
    market_ticker: str
    probability_ppm: ProbabilityPpm
    eligible_for_live: bool
    abstention_reason: str | None
    traded: bool
    baseline_executable_price_pips: int
    correlation_group_id: str | None = None
    created_sequence: int | None = None
    live: bool = False

    def __post_init__(self) -> None:
        """Validate the numeric invariants of the forecast row.

        Raises:
            TypeError: If ``baseline_executable_price_pips`` or a non-``None``
                ``created_sequence`` is a ``bool`` (an ``int`` subclass that
                must not masquerade as a number) or is not an ``int`` at all --
                mirroring the ``_IntUnit`` guard in
                :mod:`windbreak.numeric.types`; the message names the field.
            ValueError: If ``probability_ppm`` falls outside the inclusive
                ``[0, 1_000_000]`` ppm range; the message names the field.
        """
        price = self.baseline_executable_price_pips
        if isinstance(price, bool) or not isinstance(price, int):
            raise TypeError(
                "baseline_executable_price_pips requires a non-bool int, "
                f"got {type(price).__name__}"
            )
        created = self.created_sequence
        if created is not None and (
            isinstance(created, bool) or not isinstance(created, int)
        ):
            raise TypeError(
                "created_sequence requires a non-bool int, "
                f"got {type(created).__name__}"
            )
        ppm = self.probability_ppm.value
        if not _PROBABILITY_PPM_MIN <= ppm <= _PROBABILITY_PPM_MAX:
            raise ValueError(
                "probability_ppm must be within "
                f"[{_PROBABILITY_PPM_MIN}, {_PROBABILITY_PPM_MAX}], got {ppm}"
            )


@dataclass(frozen=True, slots=True)
class EvaluationInputs:
    """The immutable inputs one evaluation run scores metrics over.

    Attributes:
        forecasts: The forecast rows to score, in fixture order.
        resolutions: Ground-truth outcomes keyed by ``market_ticker``.
        temporal: The temporal context the run's forecasts are gated against
            (#52), or ``None`` for a run that carries no temporal coordinates
            (e.g. a renderer or stub unit test over empty inputs).
        execution_records: The live-fill execution-quality records the
            ``live_slippage_ratio`` metric folds (#58). Defaults to ``()`` so
            every pre-#58 construction stays valid; unlike forecasts, fills are
            not forecast-gated, so the temporal gate carries them through
            unchanged.
    """

    forecasts: tuple[FixtureForecast, ...]
    resolutions: Mapping[str, ResolutionOutcome]
    temporal: TemporalContext | None = None
    execution_records: tuple[ExecutionQualityRecord, ...] = ()


def gate_evaluation_inputs(
    inputs: EvaluationInputs,
) -> tuple[EvaluationInputs, tuple[RejectionEvent, ...]]:
    """Gate inputs for temporal integrity, returning admitted inputs + ledger.

    Runs :func:`~windbreak.evaluation.temporal.enforce_temporal_integrity` and
    reconstructs an :class:`EvaluationInputs` carrying only the admitted
    forecasts while preserving the original ``resolutions`` and ``temporal``
    context, so a metric never observes a rejected record. This module owns the
    reconstruction seam because :mod:`windbreak.evaluation.temporal` cannot
    import :class:`EvaluationInputs` without forming a cycle.

    Args:
        inputs: The raw evaluation inputs to gate.

    Returns:
        A ``(admitted_inputs, rejections)`` pair: the inputs narrowed to the
        admitted forecasts, and the rejection ledger in fixture order.

    Raises:
        ValueError: If ``inputs.temporal`` is ``None`` while forecasts are
            present (propagated from the gate; there is no silent skip).
    """
    result = enforce_temporal_integrity(inputs)
    admitted = EvaluationInputs(
        forecasts=result.admitted_forecasts,
        resolutions=inputs.resolutions,
        temporal=inputs.temporal,
        execution_records=inputs.execution_records,
    )
    return admitted, result.rejections


#: The observation window every forecast-track metric is scored over (S13.4);
#: the compute adapters resolve it over the inputs before delegating. Sourced
#: from its canonical home in :mod:`windbreak.evaluation.windows` so the choice is
#: defined once and shared with the report's selection-detail computation.
_FORECAST_WINDOW = HEADLINE_OBSERVATION_WINDOW


def _windowed(inputs: EvaluationInputs, window: ObservationWindow) -> EvaluationInputs:
    """Narrow an inputs' forecasts to the slice the observation ``window`` admits.

    This is the seam that makes an :class:`ObservationWindow` genuinely
    load-bearing rather than a decorative label: it applies
    :func:`windbreak.evaluation.windows.resolve_window` to
    ``inputs.forecasts`` so a multi-forecast-per-market input collapses to the
    single declared observation per market *before* the window-agnostic scorer
    in :mod:`windbreak.evaluation.metrics` sees it. Because every temporally
    admitted forecast carries a non-``None`` ``created_sequence`` (the temporal
    gate rejects ``None`` as pre-deployment), a per-market window never meets the
    fail-closed ``None`` path here on a production compute path.

    Args:
        inputs: The (temporally admitted) evaluation inputs to narrow.
        window: The observation window to resolve the forecasts under.

    Returns:
        A new :class:`EvaluationInputs` carrying the window-selected forecasts
        and the original ``resolutions`` and ``temporal`` context.
    """
    selected = resolve_window(inputs.forecasts, window=window)
    return EvaluationInputs(
        forecasts=selected.forecasts,
        resolutions=inputs.resolutions,
        temporal=inputs.temporal,
        execution_records=inputs.execution_records,
    )


def _compute_brier(inputs: EvaluationInputs) -> MetricValue:
    """Compute the forecast-track mean Brier score, in ppm.

    Args:
        inputs: The evaluation inputs to score.

    Returns:
        The mean Brier score delegated to :func:`windbreak.evaluation.metrics`.
    """
    return metrics.mean_brier(
        _windowed(inputs, _FORECAST_WINDOW), window=_FORECAST_WINDOW
    )


def _compute_brier_skill_vs_executable_price(inputs: EvaluationInputs) -> MetricValue:
    """Compute the headline Brier skill versus the executable-price baseline.

    Args:
        inputs: The evaluation inputs to score.

    Returns:
        The Brier skill in ppm delegated to :func:`windbreak.evaluation.metrics`.
    """
    return metrics.brier_skill(
        _windowed(inputs, _FORECAST_WINDOW), window=_FORECAST_WINDOW
    )


def _compute_log_score(inputs: EvaluationInputs) -> MetricValue:
    """Compute the forecast-track mean logarithmic score, in micro-nats.

    Args:
        inputs: The evaluation inputs to score.

    Returns:
        The mean log score delegated to :func:`windbreak.evaluation.metrics`.
    """
    return metrics.mean_log_score(
        _windowed(inputs, _FORECAST_WINDOW), window=_FORECAST_WINDOW
    )


def _compute_expected_calibration_error(inputs: EvaluationInputs) -> MetricValue:
    """Compute the forecast-track expected calibration error, in ppm.

    Args:
        inputs: The evaluation inputs to score.

    Returns:
        The ECE delegated to :func:`windbreak.evaluation.metrics`.
    """
    return metrics.expected_calibration_error(
        _windowed(inputs, _FORECAST_WINDOW), window=_FORECAST_WINDOW
    )


def _compute_calibration_slope(inputs: EvaluationInputs) -> MetricValue:
    """Compute the forecast-track calibration slope, in ppm.

    Args:
        inputs: The evaluation inputs to score.

    Returns:
        The calibration slope delegated to :func:`windbreak.evaluation.metrics`.
    """
    return metrics.calibration_slope(
        _windowed(inputs, _FORECAST_WINDOW), window=_FORECAST_WINDOW
    )


def _compute_calibration_intercept(inputs: EvaluationInputs) -> MetricValue:
    """Compute the forecast-track calibration intercept, in ppm.

    Args:
        inputs: The evaluation inputs to score.

    Returns:
        The calibration intercept delegated to
        :func:`windbreak.evaluation.metrics`.
    """
    return metrics.calibration_intercept(
        _windowed(inputs, _FORECAST_WINDOW), window=_FORECAST_WINDOW
    )


def _compute_sharpness(inputs: EvaluationInputs) -> MetricValue:
    """Compute the forecast-track sharpness (forecast variance), in ppm.

    Args:
        inputs: The evaluation inputs to score.

    Returns:
        The sharpness delegated to :func:`windbreak.evaluation.metrics`.
    """
    return metrics.sharpness(
        _windowed(inputs, _FORECAST_WINDOW), window=_FORECAST_WINDOW
    )


def _compute_traded_vs_skipped_brier_delta(inputs: EvaluationInputs) -> MetricValue:
    """Compute the selection-track traded-vs-skipped Brier delta, in ppm (#53).

    Args:
        inputs: The evaluation inputs to score.

    Returns:
        ``mean_brier(SKIPPED) - mean_brier(TRADED)`` in ppm, delegated to
        :func:`windbreak.evaluation.cohorts.traded_vs_skipped_brier_delta`; a
        negative value flags that skipped forecasts outperformed traded ones. If
        either cohort has no resolved records in the window -- an ordinary
        early-deployment state (nothing traded yet, or nothing skipped) -- the
        delta is genuinely undefined, so the
        :data:`~windbreak.evaluation.cohorts.UNDEFINED` sentinel is returned
        rather than letting the exception crash the whole report. The catch is
        scoped to :class:`~windbreak.evaluation.cohorts.EmptyCohortError` alone,
        so any other invalid-input ``ValueError`` still propagates.
    """
    try:
        return cohorts.traded_vs_skipped_brier_delta(inputs, window=_FORECAST_WINDOW)
    except cohorts.EmptyCohortError:
        return cohorts.UNDEFINED


def _compute_fill_vs_model_slippage(inputs: EvaluationInputs) -> MetricValue:
    """Execution-track fill-vs-model slippage stub.

    Args:
        inputs: The evaluation inputs (ignored at this tracer-code stage).

    Returns:
        The :data:`NOT_IMPLEMENTED` sentinel until a later issue wires the
        execution-track slippage.
    """
    del inputs  # Tracer stub: the execution-track slippage lands in a later issue.
    return NOT_IMPLEMENTED


def _windowed_execution_records(
    inputs: EvaluationInputs,
) -> tuple[ExecutionQualityRecord, ...]:
    """Return the most-recent execution records within the rolling window.

    Args:
        inputs: The evaluation inputs carrying the execution-quality records.

    Returns:
        The :data:`LIVE_ROLLING_WINDOW_SIZE` records with the highest
        ``created_sequence`` (descending), so the ratio scores only the current
        window -- matching the SQL path's ``ORDER BY created_sequence DESC
        LIMIT`` truncation.
    """
    ordered = sorted(
        inputs.execution_records,
        key=lambda record: record.created_sequence,
        reverse=True,
    )
    return tuple(ordered[:LIVE_ROLLING_WINDOW_SIZE])


def _compute_live_slippage_ratio(inputs: EvaluationInputs) -> MetricValue:
    """Compute the execution-track live-vs-paper slippage ratio, in ppm (#58).

    Args:
        inputs: The evaluation inputs to score.

    Returns:
        The rolling-window slippage ratio delegated to
        :func:`windbreak.evaluation.execution_quality.live_slippage_ratio`, or the
        :data:`~windbreak.evaluation.cohorts.UNDEFINED` sentinel when no execution
        record exists (an ordinary early-deployment state). A non-empty window
        whose modeled-cost sum is ``0`` is likewise empty-but-valid and degrades
        to the ``UNDEFINED`` sentinel, so the divergence monitor always samples
        and never breaches on that degenerate window. The catch is scoped to
        :class:`~windbreak.evaluation.execution_quality.ZeroModeledCostError`
        alone, so any other invalid-input ``ValueError`` still propagates.
    """
    try:
        return execution_quality.live_slippage_ratio(
            _windowed_execution_records(inputs)
        )
    except execution_quality.ZeroModeledCostError:
        return cohorts.UNDEFINED


def _compute_live_brier_degradation(inputs: EvaluationInputs) -> MetricValue:
    """Compute the LIVE-vs-PAPER rolling Brier degradation, in ppm (#58).

    Args:
        inputs: The evaluation inputs to score.

    Returns:
        ``mean_brier(LIVE_window) - mean_brier(PAPER)`` delegated to
        :func:`windbreak.evaluation.cohorts.live_brier_degradation`; a positive
        value flags the LIVE track scoring worse than the PAPER baseline. If
        either cohort has no resolved records -- an ordinary early-deployment
        state -- the :data:`~windbreak.evaluation.cohorts.UNDEFINED` sentinel is
        returned rather than letting the empty-cohort case crash the report. The
        catch is scoped to :class:`~windbreak.evaluation.cohorts.EmptyCohortError`
        alone, so any other invalid-input ``ValueError`` still propagates.
    """
    try:
        return cohorts.live_brier_degradation(
            inputs, window=_FORECAST_WINDOW, window_size=LIVE_ROLLING_WINDOW_SIZE
        )
    except cohorts.EmptyCohortError:
        return cohorts.UNDEFINED


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """The static definition of one evaluation metric.

    Attributes:
        name: Unique metric name, used as its registry key and render label.
        track: The :class:`Track` this metric reports under.
        window: The :class:`ObservationWindow` the metric observes.
        compute: Callable turning :class:`EvaluationInputs` into a
            :data:`MetricValue` (an ``int`` measurement or the sentinel).
    """

    name: str
    track: Track
    window: ObservationWindow
    compute: Callable[[EvaluationInputs], MetricValue]

    def __post_init__(self) -> None:
        """Wrap ``compute`` so every call is temporally gated at the choke point.

        The original ``compute`` is replaced with a wrapper that first routes
        its inputs through :func:`gate_evaluation_inputs` and only ever calls
        the original with the *admitted* inputs. This is the single, mandatory
        choke point: there is no ungated call path and no opt-out, so no
        ``MetricSpec`` -- registered or freshly constructed -- can observe a
        rejected record. Re-gating already-admitted inputs is idempotent, so the
        wrap is safe to apply even when the caller has pre-gated.
        """
        original = self.compute

        @functools.wraps(original)
        def gated_compute(inputs: EvaluationInputs) -> MetricValue:
            """Gate inputs then delegate to the original ``compute``.

            A metrics-level
            :class:`~windbreak.evaluation.metrics.NoResolvedForecastsError`
            (raised when a fold has no resolved forecasts at all -- an ordinary
            whole-ledger, pre-resolution state, #188) is caught here and mapped
            to the :data:`~windbreak.evaluation.cohorts.UNDEFINED` sentinel,
            mirroring the per-metric ``EmptyCohortError``/``ZeroModeledCostError``
            adapters, so no registered metric raises over unresolved-only inputs.
            The catch is scoped to that one exception type; any other
            invalid-input ``ValueError`` still propagates.

            Its siblings in the
            :class:`~windbreak.evaluation.metrics.UndefinedMetricError` family
            (#439) are deliberately NOT caught here. They are mapped to the same
            sentinel one layer up, in
            :func:`windbreak.evaluation.report._build_tracks`, because the other
            consumer of a registered spec --
            :func:`windbreak.evaluation.crosscheck.crosscheck_gates` -- must
            keep seeing the raise: its whole job is to notice when the Python
            and SQL paths disagree, and it renders a raising reference metric as
            its own ``PYTHON_COMPUTE_FAILED`` sentinel. Swallowing the raise
            here would have made that guard unreachable.

            Args:
                inputs: The raw evaluation inputs handed to the metric.

            Returns:
                The original metric's value over the temporally-admitted inputs,
                or the ``UNDEFINED`` sentinel when nothing resolved.
            """
            admitted, _ = gate_evaluation_inputs(inputs)
            try:
                return original(admitted)
            except metrics.NoResolvedForecastsError:
                return cohorts.UNDEFINED

        object.__setattr__(self, "compute", gated_compute)


#: The registry key of the headline forecast-skill metric the renderer gates the
#: "no edge" banner on.
HEADLINE_SKILL_METRIC: Final[str] = "brier_skill_vs_executable_price"


def _seed_metric_specs() -> list[MetricSpec]:
    """Build the fixed list of seed :class:`MetricSpec`s for the harness.

    Issue #51 turns ``brier`` and ``brier_skill_vs_executable_price`` into real
    computations and adds five more forecast-track metrics (``log_score``,
    ``expected_calibration_error``, ``calibration_slope``,
    ``calibration_intercept``, ``sharpness``); issue #53 makes
    ``traded_vs_skipped_brier_delta`` real too. Only ``fill_vs_model_slippage``
    remains a stub. Issue #58 adds the two live-divergence metrics
    (``live_slippage_ratio``, ``live_brier_degradation``).

    Returns:
        The eleven metric specifications, spanning all three :class:`Track`s.
    """
    return [
        MetricSpec(
            name="brier",
            track=Track.FORECAST,
            window=ObservationWindow.LATEST_BEFORE_CLOSE,
            compute=_compute_brier,
        ),
        MetricSpec(
            name=HEADLINE_SKILL_METRIC,
            track=Track.FORECAST,
            window=ObservationWindow.LATEST_BEFORE_CLOSE,
            compute=_compute_brier_skill_vs_executable_price,
        ),
        MetricSpec(
            name="log_score",
            track=Track.FORECAST,
            window=ObservationWindow.LATEST_BEFORE_CLOSE,
            compute=_compute_log_score,
        ),
        MetricSpec(
            name="expected_calibration_error",
            track=Track.FORECAST,
            window=ObservationWindow.LATEST_BEFORE_CLOSE,
            compute=_compute_expected_calibration_error,
        ),
        MetricSpec(
            name="calibration_slope",
            track=Track.FORECAST,
            window=ObservationWindow.LATEST_BEFORE_CLOSE,
            compute=_compute_calibration_slope,
        ),
        MetricSpec(
            name="calibration_intercept",
            track=Track.FORECAST,
            window=ObservationWindow.LATEST_BEFORE_CLOSE,
            compute=_compute_calibration_intercept,
        ),
        MetricSpec(
            name="sharpness",
            track=Track.FORECAST,
            window=ObservationWindow.LATEST_BEFORE_CLOSE,
            compute=_compute_sharpness,
        ),
        MetricSpec(
            name="traded_vs_skipped_brier_delta",
            track=Track.SELECTION,
            window=ObservationWindow.LATEST_BEFORE_CLOSE,
            compute=_compute_traded_vs_skipped_brier_delta,
        ),
        MetricSpec(
            name="fill_vs_model_slippage",
            track=Track.EXECUTION,
            window=ObservationWindow.TRADE_TRIGGERING,
            compute=_compute_fill_vs_model_slippage,
        ),
        MetricSpec(
            name="live_slippage_ratio",
            track=Track.EXECUTION,
            window=ObservationWindow.TRADE_TRIGGERING,
            compute=_compute_live_slippage_ratio,
        ),
        # ``live_brier_degradation`` is ``Track.FORECAST`` (a forecast-quality
        # Brier delta), whereas its "live-divergence" sibling
        # ``live_slippage_ratio`` above is ``Track.EXECUTION`` (an
        # execution-quality cost ratio) -- the two do NOT share a track.
        MetricSpec(
            name="live_brier_degradation",
            track=Track.FORECAST,
            window=ObservationWindow.LATEST_BEFORE_CLOSE,
            compute=_compute_live_brier_degradation,
        ),
    ]


def registered_metrics() -> Mapping[str, MetricSpec]:
    """Return the seed metric catalogue keyed by unique metric name.

    Returns:
        A mapping from each metric's ``name`` to its :class:`MetricSpec`.

    Raises:
        ValueError: If two specs share a ``name`` -- a construction-time
            invariant guarding against a silent drop that a naive
            list-to-dict conversion would hide.
    """
    registry: dict[str, MetricSpec] = {}
    for spec in _seed_metric_specs():
        if spec.name in registry:
            raise ValueError(f"duplicate metric name in registry: {spec.name!r}")
        registry[spec.name] = spec
    return registry
