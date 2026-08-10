"""Forecast-track statistical metrics for the evaluation harness (#51).

SPEC-EPIC_07 S13.5's forecast-quality machinery, computed with *zero floats*:
every statistic is exact integer arithmetic over the parts-per-million (ppm)
integers carried on :class:`~windbreak.evaluation.registry.EvaluationInputs`, and
every reported value is a ppm-scaled ``int`` produced through the sanctioned
:func:`windbreak.numeric.rounding.divide` with an explicit conservative rounding
direction. There is no ``float()``, no ``math.log``, and no bare true-division on
the value path anywhere in this module.

The scalar metrics (:func:`mean_brier`, :func:`mean_log_score`,
:func:`brier_skill`, :func:`expected_calibration_error`,
:func:`calibration_slope`, :func:`calibration_intercept`, :func:`sharpness`)
each take ``(inputs, *, window)`` and return a single ``int``. The rich reports
(:func:`reliability_diagram`, :func:`price_bucket_report`,
:func:`edge_bucket_report`) return tuples of frozen dataclasses.

Only forecasts whose ``market_ticker`` resolves enter any metric (S13.6):
forecasts on unresolved or absent tickers are excluded, and an empty resolved
set raises :class:`ValueError` rather than silently scoring zero observations.

This module imports :class:`~windbreak.evaluation.registry.EvaluationInputs` and
its collaborators only under :data:`typing.TYPE_CHECKING`, keeping the
registry->metrics dependency a one-way *annotation-only* edge with no runtime
import cycle; :class:`~windbreak.evaluation.resolution.ResolutionOutcome` is the
one runtime import, from the dependency-free resolution leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from windbreak.evaluation.resolution import ResolutionOutcome
from windbreak.numeric.rounding import RoundingDirection, divide

if TYPE_CHECKING:
    from windbreak.evaluation.registry import (
        EvaluationInputs,
        FixtureForecast,
    )
    from windbreak.evaluation.windows import ObservationWindow

#: One whole probability expressed in ppm (1.0 == 1_000_000 ppm). Also the
#: ppm-scaling factor used to lift a mean/ratio back into ppm space.
PPM_SCALE = 1_000_000
#: A ``YES`` resolution as a ppm outcome (certainty); ``NO`` is ``0``.
OUTCOME_YES_PPM = 1_000_000
#: Exact ppm-per-pip factor: one pip (1e-4 payout-$) is 100 ppm of probability.
BASELINE_PPM_PER_PIP = 100
#: Full binary payout, in pips ($1.00): a winning 1-contract long-yes pays this.
PAYOUT_PIPS = 10_000

#: Number of equal-width probability bins for calibration reports.
ECE_BIN_COUNT = 10
#: Width of one calibration bin, in ppm (``PPM_SCALE // ECE_BIN_COUNT``).
_BIN_WIDTH_PPM = 100_000

#: Number of price-decile buckets in :func:`price_bucket_report`.
PRICE_BUCKET_COUNT = 10
#: Width of one price bucket, in pips.
_PRICE_BUCKET_WIDTH_PIPS = 1_000
#: The eleven contiguous price-bucket edges, in pips: ``0, 1000, ..., 10_000``.
PRICE_BUCKET_EDGES_PIPS = tuple(
    index * _PRICE_BUCKET_WIDTH_PIPS for index in range(PRICE_BUCKET_COUNT + 1)
)

#: The seven symmetric edge-bucket boundaries, in ppm, for
#: :func:`edge_bucket_report`; the six buckets they delimit widen away from 0.
EDGE_BUCKET_EDGES_PPM = (
    -1_000_000,
    -100_000,
    -50_000,
    0,
    50_000,
    100_000,
    1_000_000,
)


class UndefinedMetricError(ValueError):
    """Raised when a metric has no value over otherwise-valid inputs (#439).

    This is the narrow vocabulary for "the mathematics does not define an
    answer here", as distinct from "these inputs are malformed". The registry's
    ``gated_compute`` choke point catches this type and maps it to the
    ``UNDEFINED`` sentinel; every other :class:`ValueError` -- a missing
    temporal context, an out-of-range probability -- still propagates, so a
    genuinely broken input is never quietly rendered as a dash.

    The distinction became load-bearing when issue #439 gave the always-on
    loop's weekly fold real ground truth. Three registered metrics are
    undefined over inputs the fold now routinely produces: the calibration
    slope and intercept need a non-zero forecast variance, which a *single*
    resolved forecast can never have; the skill ratio needs a baseline that
    made some error; and the log score diverges on a probability of exactly
    zero. Each previously raised a bare :class:`ValueError` that nothing
    caught, which was unreachable only because the fold hardcoded
    ``resolutions={}`` and :class:`NoResolvedForecastsError` always fired
    first. The first market ever to resolve would otherwise have taken the
    whole weekly report down with it.

    Subclassing :class:`ValueError` keeps every pre-existing
    ``pytest.raises(ValueError)`` expectation passing unchanged.
    """


class NoResolvedForecastsError(UndefinedMetricError):
    """Raised when a metric is asked to score an empty resolved set (#188).

    The original member of the :class:`UndefinedMetricError` family: a fold
    with nothing resolved yet is an ordinary whole-ledger, pre-resolution
    state, not an error, so the registry's ``gated_compute`` choke point maps
    it to the ``UNDEFINED`` sentinel (mirroring the existing
    :class:`~windbreak.evaluation.cohorts.EmptyCohortError` adapter) rather
    than crashing the fold. It remains a distinct type so a caller that cares
    specifically about "nothing resolved" can still say so.
    """


#: ``floor(ln(2) * 10**18)``: natural log of 2 scaled by 1e18 for the integer
#: log-score. ``ln(2) = 0.693147180559945309417232...``; truncating at 18
#: fractional decimal digits gives ``693147180559945309``. The residual error is
#: below 1e-18 in nats, far tighter than any ppm-scaled result can observe.
_LN2_SCALED_E18 = 693_147_180_559_945_309
#: The decimal scale (1e18) that :data:`_LN2_SCALED_E18` is expressed in.
_LN2_DECIMAL_SCALE = 1_000_000_000_000_000_000
#: Number of fractional bits extracted for the fixed-point ``log2`` mantissa in
#: :func:`_ln_micro_nats`; 64 bits leaves the truncation error negligible.
_LOG2_FRACTION_BITS = 64
#: Divisor lifting the log-score result from ``log2 * 1e18``-scaled nats to
#: micro-nats: ``2**_LOG2_FRACTION_BITS * (1e18 / 1e6)``.
_LOG_SCORE_DENOMINATOR = (1 << _LOG2_FRACTION_BITS) * (_LN2_DECIMAL_SCALE // PPM_SCALE)


@dataclass(frozen=True, slots=True)
class _ScoredPair:
    """One resolved forecast joined to its market's ground-truth outcome.

    Attributes:
        probability_ppm: The forecast probability, in ppm.
        outcome_ppm: The resolved outcome as ppm certainty (``0`` or
            :data:`OUTCOME_YES_PPM`).
        baseline_ppm: The executable-price baseline probability, in ppm
            (``baseline_executable_price_pips * BASELINE_PPM_PER_PIP``).
        baseline_pips: The executable ask price, in pips (the trade fill price).
        traded: Whether a live trade was actually taken on this forecast.
        market_ticker: The forecast's market ticker (its singleton cluster key).
        correlation_group_id: The forecast's correlation group, or ``None`` when
            it is its own singleton cluster.
    """

    probability_ppm: int
    outcome_ppm: int
    baseline_ppm: int
    baseline_pips: int
    traded: bool
    market_ticker: str
    correlation_group_id: str | None


def _scored_pair(forecast: FixtureForecast, outcome: ResolutionOutcome) -> _ScoredPair:
    """Join one forecast to its resolved outcome into a :class:`_ScoredPair`.

    Args:
        forecast: The forecast row to score.
        outcome: The ground-truth outcome of the forecast's market.

    Returns:
        The scored pair carrying ppm-scaled probability, outcome, and baseline.
    """
    outcome_ppm = OUTCOME_YES_PPM if outcome is ResolutionOutcome.YES else 0
    return _ScoredPair(
        probability_ppm=forecast.probability_ppm.value,
        outcome_ppm=outcome_ppm,
        baseline_ppm=forecast.baseline_executable_price_pips * BASELINE_PPM_PER_PIP,
        baseline_pips=forecast.baseline_executable_price_pips,
        traded=forecast.traded,
        market_ticker=forecast.market_ticker,
        correlation_group_id=forecast.correlation_group_id,
    )


def _scored_pairs(
    inputs: EvaluationInputs, *, window: ObservationWindow
) -> tuple[_ScoredPair, ...]:
    """Join every *resolved* forecast to its outcome, excluding the unresolved.

    Forecasts whose ``market_ticker`` has no entry in ``inputs.resolutions`` are
    dropped (S13.6): they never enter a headline metric. The ``window`` argument
    is part of every metric's signature as a declared label only; this function
    scores whatever forecasts it is handed and never selects a slice itself.
    Per-market slice selection for multi-forecast-per-market inputs is the
    caller's responsibility: the registry forecast-track adapters (via
    ``_windowed``) and the cohort functions (via ``_windowed_cohort_forecasts``)
    apply :func:`windbreak.evaluation.windows.resolve_window` before scoring, so
    the forecasts they hand in are already the window's chosen records.

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        One :class:`_ScoredPair` per resolved forecast, in fixture order.

    Raises:
        ValueError: If no forecast resolves -- a metric must never be computed
            over an empty resolved set.
    """
    del window  # Label only; the caller already applied windows.resolve_window (#53).
    pairs: list[_ScoredPair] = []
    for forecast in inputs.forecasts:
        outcome = inputs.resolutions.get(forecast.market_ticker)
        if outcome is None:
            continue
        pairs.append(_scored_pair(forecast, outcome))
    if not pairs:
        raise NoResolvedForecastsError(
            "no resolved forecasts to score (empty resolved set)"
        )
    return tuple(pairs)


def _forecast_term(pair: _ScoredPair) -> int:
    """Return the forecast's squared error ``(p - o)^2``, in ppm-squared.

    Args:
        pair: The scored forecast/outcome pair.

    Returns:
        The exact squared difference of probability and outcome, in ppm^2.
    """
    return (pair.probability_ppm - pair.outcome_ppm) ** 2


def _baseline_term(pair: _ScoredPair) -> int:
    """Return the baseline's squared error ``(baseline - o)^2``, in ppm-squared.

    Args:
        pair: The scored forecast/outcome pair.

    Returns:
        The exact squared difference of baseline and outcome, in ppm^2.
    """
    return (pair.baseline_ppm - pair.outcome_ppm) ** 2


@dataclass(frozen=True, slots=True)
class ForecastTerms:
    """One resolved forecast's Brier terms plus its cluster identity.

    Exposed so the clustered bootstrap can pool terms by cluster without reaching
    into this module's private scoring internals.

    Attributes:
        forecast_term: The forecast's squared error ``(p - o)^2``, in ppm^2.
        baseline_term: The baseline's squared error ``(baseline - o)^2``, ppm^2.
        correlation_group_id: The forecast's correlation group, or ``None``.
        market_ticker: The forecast's market ticker (its singleton cluster key).
    """

    forecast_term: int
    baseline_term: int
    correlation_group_id: str | None
    market_ticker: str


def resolved_forecast_terms(
    inputs: EvaluationInputs, *, window: ObservationWindow
) -> tuple[ForecastTerms, ...]:
    """Return the per-forecast Brier terms and cluster identity, resolved only.

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        One :class:`ForecastTerms` per resolved forecast, in fixture order.

    Raises:
        ValueError: If no forecast resolves.
    """
    return tuple(
        ForecastTerms(
            forecast_term=_forecast_term(pair),
            baseline_term=_baseline_term(pair),
            correlation_group_id=pair.correlation_group_id,
            market_ticker=pair.market_ticker,
        )
        for pair in _scored_pairs(inputs, window=window)
    )


def _skill_from_term_sums(forecast_sum: int, baseline_sum: int) -> int:
    """Convert forecast/baseline Brier-term sums into a skill score, in ppm.

    Brier skill is ``1 - forecast_sum / baseline_sum``; rearranged over a common
    denominator it is ``(baseline_sum - forecast_sum) / baseline_sum``, lifted to
    ppm and floored (``UNDERSTATE_EQUITY``) so a marginal skill never rounds up
    into a falsely-positive edge.

    Args:
        forecast_sum: Sum of forecast Brier terms, in ppm^2.
        baseline_sum: Sum of baseline Brier terms, in ppm^2.

    Returns:
        The Brier skill, in ppm (``PPM_SCALE`` == perfect; may be negative).

    Raises:
        ValueError: If ``baseline_sum`` is zero -- the ratio, and thus skill, is
            undefined against a baseline that made no error.
    """
    if baseline_sum == 0:
        raise UndefinedMetricError(
            "baseline Brier-term sum is zero; skill is undefined"
        )
    return divide(
        (baseline_sum - forecast_sum) * PPM_SCALE,
        baseline_sum,
        rounding=RoundingDirection.UNDERSTATE_EQUITY,
    )


def mean_brier(inputs: EvaluationInputs, *, window: ObservationWindow) -> int:
    """Compute the mean Brier score over the resolved forecasts, in ppm.

    The Brier score is the mean squared error ``mean((p - o)^2)``; in ppm space
    that is ``sum((p - o)^2) / (n * PPM_SCALE)``, rounded up
    (``OVERSTATE_COST``) so a forecast's measured error is never understated.

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        The mean Brier score, in ppm (``0`` best, ``PPM_SCALE`` worst).

    Raises:
        ValueError: If no forecast resolves.
    """
    pairs = _scored_pairs(inputs, window=window)
    total = sum(_forecast_term(pair) for pair in pairs)
    return divide(
        total, len(pairs) * PPM_SCALE, rounding=RoundingDirection.OVERSTATE_COST
    )


def brier_skill(inputs: EvaluationInputs, *, window: ObservationWindow) -> int:
    """Compute Brier skill versus the executable-price baseline, in ppm.

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        The Brier skill, in ppm (``PPM_SCALE`` == perfect; may be negative).

    Raises:
        ValueError: If no forecast resolves, or the baseline made no error.
    """
    pairs = _scored_pairs(inputs, window=window)
    forecast_sum = sum(_forecast_term(pair) for pair in pairs)
    baseline_sum = sum(_baseline_term(pair) for pair in pairs)
    return _skill_from_term_sums(forecast_sum, baseline_sum)


def _ln_micro_nats(arg_ppm: int) -> int:
    """Return the surprisal ``-ln(arg_ppm / PPM_SCALE)`` in micro-nats (>= 0).

    Computed from scratch with pure integers: ``arg_ppm / PPM_SCALE`` in ``(0, 1]``
    has a non-negative surprisal ``ln(PPM_SCALE / arg_ppm)``. The ratio is first
    reduced to ``[1, 2)`` (tracking the integer ``log2`` part), then
    :data:`_LOG2_FRACTION_BITS` fractional ``log2`` bits are extracted by repeated
    fixed-point squaring (square, compare to 2, halve). The resulting scaled
    ``log2`` is multiplied by :data:`_LN2_SCALED_E18` (``ln 2``) and reduced to
    micro-nats, rounded up (``OVERSTATE_COST``) so a penalty is never understated.

    Args:
        arg_ppm: The probability of the *observed* outcome, in ppm.

    Returns:
        The surprisal in micro-nats (nats * ``PPM_SCALE``); ``0`` when the
        observed outcome was forecast with certainty (``arg_ppm == PPM_SCALE``).

    Raises:
        ValueError: If ``arg_ppm`` is ``0`` -- a certain-and-wrong forecast has an
            infinite log penalty and cannot be scored.
    """
    if arg_ppm == 0:
        raise UndefinedMetricError(
            "log-score probability is 0 (certain-wrong); -ln(0) diverges"
        )
    if arg_ppm == PPM_SCALE:
        return 0
    log2_fixed = _log2_reciprocal_fixed(arg_ppm)
    return divide(
        log2_fixed * _LN2_SCALED_E18,
        _LOG_SCORE_DENOMINATOR,
        rounding=RoundingDirection.OVERSTATE_COST,
    )


def _log2_reciprocal_fixed(arg_ppm: int) -> int:
    """Return ``log2(PPM_SCALE / arg_ppm) * 2**_LOG2_FRACTION_BITS`` as an int.

    Args:
        arg_ppm: A probability in ppm, strictly in ``(0, PPM_SCALE)``.

    Returns:
        The base-2 logarithm of the reciprocal ratio, scaled by
        ``2**_LOG2_FRACTION_BITS`` and truncated to an integer.
    """
    numerator, denominator, integer_part = _normalise_ratio(PPM_SCALE, arg_ppm)
    mantissa = (numerator << _LOG2_FRACTION_BITS) // denominator
    fraction = _log2_fraction_bits(mantissa)
    return (integer_part << _LOG2_FRACTION_BITS) + fraction


def _normalise_ratio(numerator: int, denominator: int) -> tuple[int, int, int]:
    """Halve a ``>= 1`` ratio into ``[1, 2)``, counting the integer log2 part.

    Args:
        numerator: The ratio numerator (``>= denominator``).
        denominator: The ratio denominator (positive).

    Returns:
        A ``(numerator, denominator, integer_part)`` triple where the (mutated)
        denominator scales the ratio into ``[1, 2)`` and ``integer_part`` is the
        number of halvings applied -- i.e. ``floor(log2(numerator/denominator))``.
    """
    integer_part = 0
    while numerator >= 2 * denominator:
        denominator *= 2
        integer_part += 1
    return numerator, denominator, integer_part


def _log2_fraction_bits(mantissa: int) -> int:
    """Extract the fractional ``log2`` bits of a ``[1, 2)`` fixed-point mantissa.

    Args:
        mantissa: ``m * 2**_LOG2_FRACTION_BITS`` for some ``m`` in ``[1, 2)``.

    Returns:
        The fractional part of ``log2(m)`` scaled by ``2**_LOG2_FRACTION_BITS``,
        built one bit at a time by repeated squaring (each square compares the
        result against 2 and halves it back into ``[1, 2)``).
    """
    fraction = 0
    for _ in range(_LOG2_FRACTION_BITS):
        mantissa = (mantissa * mantissa) >> _LOG2_FRACTION_BITS
        fraction <<= 1
        if mantissa >= (2 << _LOG2_FRACTION_BITS):
            mantissa >>= 1
            fraction |= 1
    return fraction


def _observed_argument(pair: _ScoredPair) -> int:
    """Return the ppm probability the forecast assigned to the *observed* outcome.

    Args:
        pair: The scored forecast/outcome pair.

    Returns:
        ``p`` for a ``YES`` outcome, or ``PPM_SCALE - p`` for a ``NO`` outcome.
    """
    if pair.outcome_ppm == OUTCOME_YES_PPM:
        return pair.probability_ppm
    return PPM_SCALE - pair.probability_ppm


def mean_log_score(inputs: EvaluationInputs, *, window: ObservationWindow) -> int:
    """Compute the mean logarithmic score over the resolved forecasts.

    The per-forecast term is the surprisal of the observed outcome
    (``-ln(p)`` on ``YES``, ``-ln(1 - p)`` on ``NO``); the report is their mean,
    in micro-nats, rounded up (``OVERSTATE_COST``).

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        The mean log score, in micro-nats (>= 0; lower is better).

    Raises:
        ValueError: If no forecast resolves, or a forecast was certain and wrong.
    """
    pairs = _scored_pairs(inputs, window=window)
    total = sum(_ln_micro_nats(_observed_argument(pair)) for pair in pairs)
    return divide(total, len(pairs), rounding=RoundingDirection.OVERSTATE_COST)


def _bin_index(probability_ppm: int) -> int:
    """Return the calibration-bin index for a probability, clamped to the top bin.

    Args:
        probability_ppm: The forecast probability, in ppm.

    Returns:
        ``min(probability_ppm // _BIN_WIDTH_PPM, ECE_BIN_COUNT - 1)`` so a
        probability of exactly ``PPM_SCALE`` lands in the final bin.
    """
    return min(probability_ppm // _BIN_WIDTH_PPM, ECE_BIN_COUNT - 1)


def expected_calibration_error(
    inputs: EvaluationInputs, *, window: ObservationWindow
) -> int:
    """Compute the expected calibration error over 10 equal-width bins, in ppm.

    ECE is ``sum_b (n_b / n) * |mean_confidence_b - frequency_b|``. Because the
    per-bin weight ``n_b / n`` and the per-bin mean's ``1 / n_b`` cancel, the
    exact value reduces to ``sum_b |sum_p_b - yes_b * PPM_SCALE| / n``, rounded up
    (``OVERSTATE_COST``) so miscalibration is never understated.

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        The expected calibration error, in ppm.

    Raises:
        ValueError: If no forecast resolves.
    """
    pairs = _scored_pairs(inputs, window=window)
    probability_sums = [0] * ECE_BIN_COUNT
    yes_counts = [0] * ECE_BIN_COUNT
    for pair in pairs:
        index = _bin_index(pair.probability_ppm)
        probability_sums[index] += pair.probability_ppm
        if pair.outcome_ppm == OUTCOME_YES_PPM:
            yes_counts[index] += 1
    total = sum(
        abs(probability_sums[index] - yes_counts[index] * PPM_SCALE)
        for index in range(ECE_BIN_COUNT)
    )
    return divide(total, len(pairs), rounding=RoundingDirection.OVERSTATE_COST)


@dataclass(frozen=True, slots=True)
class _OlsSums:
    """The five running sums an ordinary-least-squares fit of outcome on forecast.

    Attributes:
        count: Number of resolved forecasts, ``n``.
        probability_sum: ``sum(p)``, in ppm.
        outcome_sum: ``sum(o)``, in ppm.
        probability_square_sum: ``sum(p^2)``, in ppm^2.
        product_sum: ``sum(p * o)``, in ppm^2.
    """

    count: int
    probability_sum: int
    outcome_sum: int
    probability_square_sum: int
    product_sum: int

    @property
    def variance_numerator(self) -> int:
        """Return ``n * sum(p^2) - sum(p)^2`` (``n^2`` times the variance)."""
        return self.count * self.probability_square_sum - self.probability_sum**2

    @property
    def covariance_numerator(self) -> int:
        """Return ``n * sum(p*o) - sum(p) * sum(o)`` (``n^2`` times covariance)."""
        return self.count * self.product_sum - self.probability_sum * self.outcome_sum


def _ols_sums(pairs: tuple[_ScoredPair, ...]) -> _OlsSums:
    """Accumulate the OLS running sums over the scored pairs.

    Args:
        pairs: The resolved forecast/outcome pairs.

    Returns:
        The populated :class:`_OlsSums`.
    """
    return _OlsSums(
        count=len(pairs),
        probability_sum=sum(pair.probability_ppm for pair in pairs),
        outcome_sum=sum(pair.outcome_ppm for pair in pairs),
        probability_square_sum=sum(pair.probability_ppm**2 for pair in pairs),
        product_sum=sum(pair.probability_ppm * pair.outcome_ppm for pair in pairs),
    )


def _require_forecast_variance(sums: _OlsSums) -> int:
    """Return the non-zero variance numerator, or raise if the forecast is flat.

    Args:
        sums: The OLS running sums.

    Returns:
        The (strictly non-zero) variance numerator.

    Raises:
        ValueError: If the forecast variance is zero (every probability equal),
            leaving the OLS slope undefined.
    """
    variance_numerator = sums.variance_numerator
    if variance_numerator == 0:
        raise UndefinedMetricError(
            "forecast variance is zero; calibration slope is undefined"
        )
    return variance_numerator


def calibration_slope(inputs: EvaluationInputs, *, window: ObservationWindow) -> int:
    """Compute the OLS calibration slope of outcome on forecast, in ppm.

    The slope is ``cov(p, o) / var(p)``; with the shared ``n^2`` factor cancelling
    it is ``covariance_numerator / variance_numerator``, lifted to ppm and floored
    (``UNDERSTATE_EQUITY``). A perfectly calibrated forecaster has slope
    ``PPM_SCALE`` (1.0).

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        The calibration slope, in ppm (``PPM_SCALE`` == 1.0).

    Raises:
        ValueError: If no forecast resolves, or the forecast variance is zero.
    """
    sums = _ols_sums(_scored_pairs(inputs, window=window))
    variance_numerator = _require_forecast_variance(sums)
    return divide(
        sums.covariance_numerator * PPM_SCALE,
        variance_numerator,
        rounding=RoundingDirection.UNDERSTATE_EQUITY,
    )


def calibration_intercept(
    inputs: EvaluationInputs, *, window: ObservationWindow
) -> int:
    """Compute the OLS calibration intercept of outcome on forecast, in ppm.

    The intercept is ``mean(o) - slope * mean(p)``; expressed exactly over the
    common denominator ``n * variance_numerator`` it is
    ``(sum(o) * var_num - cov_num * sum(p)) / (n * var_num)``, floored
    (``UNDERSTATE_EQUITY``). A perfectly calibrated forecaster has intercept ``0``.

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        The calibration intercept, in ppm.

    Raises:
        ValueError: If no forecast resolves, or the forecast variance is zero.
    """
    sums = _ols_sums(_scored_pairs(inputs, window=window))
    variance_numerator = _require_forecast_variance(sums)
    numerator = (
        sums.outcome_sum * variance_numerator
        - sums.covariance_numerator * sums.probability_sum
    )
    return divide(
        numerator,
        sums.count * variance_numerator,
        rounding=RoundingDirection.UNDERSTATE_EQUITY,
    )


def sharpness(inputs: EvaluationInputs, *, window: ObservationWindow) -> int:
    """Compute the sharpness (variance of the forecast probabilities), in ppm.

    Variance is ``mean((p - mean_p)^2) = (n * sum(p^2) - sum(p)^2) / n^2``; the
    ppm-scaled report is that divided by ``PPM_SCALE`` again, floored
    (``UNDERSTATE_EQUITY``), i.e.
    ``variance_numerator / (n^2 * PPM_SCALE)``.

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        The sharpness, in ppm.

    Raises:
        ValueError: If no forecast resolves.
    """
    sums = _ols_sums(_scored_pairs(inputs, window=window))
    return divide(
        sums.variance_numerator,
        sums.count * sums.count * PPM_SCALE,
        rounding=RoundingDirection.UNDERSTATE_EQUITY,
    )


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One equal-width probability bin of a reliability diagram.

    Attributes:
        bin_low_ppm: Inclusive lower probability edge of the bin, in ppm.
        bin_high_ppm: Exclusive upper probability edge of the bin, in ppm.
        count: Number of resolved forecasts that fell in the bin.
        mean_forecast_ppm: Mean forecast probability in the bin, in ppm
            (``0`` for an empty bin).
        observed_frequency_ppm: Observed ``YES`` frequency in the bin, in ppm
            (``0`` for an empty bin).
    """

    bin_low_ppm: int
    bin_high_ppm: int
    count: int
    mean_forecast_ppm: int
    observed_frequency_ppm: int


def _mean_or_zero(total: int, count: int) -> int:
    """Return ``floor(total / count)`` in ppm, or ``0`` for an empty group.

    Args:
        total: The summed numerator.
        count: The group size; ``0`` yields ``0`` rather than dividing.

    Returns:
        The floored mean, or ``0`` when ``count`` is ``0``.
    """
    if count == 0:
        return 0
    return divide(total, count, rounding=RoundingDirection.UNDERSTATE_EQUITY)


def _observed_frequency_ppm(yes_count: int, count: int) -> int:
    """Return the ``YES`` frequency of a group, in ppm, or ``0`` if empty.

    Args:
        yes_count: Number of ``YES`` outcomes in the group.
        count: The group size.

    Returns:
        ``yes_count * PPM_SCALE / count`` floored, or ``0`` when empty.
    """
    return _mean_or_zero(yes_count * PPM_SCALE, count)


def _mean_brier_ppm(term_sum: int, count: int) -> int:
    """Return the mean Brier score of a group, in ppm, or ``0`` if empty.

    Args:
        term_sum: Sum of the group's Brier terms, in ppm^2.
        count: The group size.

    Returns:
        ``term_sum / (count * PPM_SCALE)`` rounded up, or ``0`` when empty.
    """
    if count == 0:
        return 0
    return divide(
        term_sum, count * PPM_SCALE, rounding=RoundingDirection.OVERSTATE_COST
    )


def _traded_pnl_pips(pair: _ScoredPair) -> int:
    """Return a traded forecast's 1-contract long-yes PnL, in pips.

    A traded forecast buys one yes contract at the executable ask and collects the
    full payout on a ``YES`` outcome; an untraded forecast contributes nothing.

    Args:
        pair: The scored forecast/outcome pair.

    Returns:
        ``payout - ask`` for a traded pair (``payout`` is ``PAYOUT_PIPS`` on
        ``YES`` else ``0``); ``0`` if the pair was not traded.
    """
    if not pair.traded:
        return 0
    payout = PAYOUT_PIPS if pair.outcome_ppm == OUTCOME_YES_PPM else 0
    return payout - pair.baseline_pips


def reliability_diagram(
    inputs: EvaluationInputs, *, window: ObservationWindow
) -> tuple[ReliabilityBin, ...]:
    """Build the 10-bin reliability diagram over the resolved forecasts.

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        Ten contiguous :class:`ReliabilityBin`s spanning ``[0, PPM_SCALE)``.

    Raises:
        ValueError: If no forecast resolves.
    """
    pairs = _scored_pairs(inputs, window=window)
    counts = [0] * ECE_BIN_COUNT
    probability_sums = [0] * ECE_BIN_COUNT
    yes_counts = [0] * ECE_BIN_COUNT
    for pair in pairs:
        index = _bin_index(pair.probability_ppm)
        counts[index] += 1
        probability_sums[index] += pair.probability_ppm
        if pair.outcome_ppm == OUTCOME_YES_PPM:
            yes_counts[index] += 1
    return tuple(
        ReliabilityBin(
            bin_low_ppm=index * _BIN_WIDTH_PPM,
            bin_high_ppm=(index + 1) * _BIN_WIDTH_PPM,
            count=counts[index],
            mean_forecast_ppm=_mean_or_zero(probability_sums[index], counts[index]),
            observed_frequency_ppm=_observed_frequency_ppm(
                yes_counts[index], counts[index]
            ),
        )
        for index in range(ECE_BIN_COUNT)
    )


@dataclass(frozen=True, slots=True)
class _BucketAccumulator:
    """An immutable running tally for one price or edge bucket.

    Attributes:
        count: Number of resolved forecasts in the bucket.
        key_sum: Sum of the bucket's key axis (forecast ppm, or edge ppm).
        yes_count: Number of ``YES`` outcomes in the bucket.
        brier_term_sum: Sum of the bucket's Brier terms, in ppm^2.
        pnl_pips: Running PnL over the bucket's *traded* forecasts, in pips.
    """

    count: int = 0
    key_sum: int = 0
    yes_count: int = 0
    brier_term_sum: int = 0
    pnl_pips: int = 0

    def add(self, pair: _ScoredPair, *, key_value: int) -> _BucketAccumulator:
        """Return a new accumulator folding one scored pair into this bucket.

        Args:
            pair: The scored forecast/outcome pair to add.
            key_value: The pair's value on the bucket's key axis.

        Returns:
            The updated accumulator (this type is immutable).
        """
        is_yes = pair.outcome_ppm == OUTCOME_YES_PPM
        return _BucketAccumulator(
            count=self.count + 1,
            key_sum=self.key_sum + key_value,
            yes_count=self.yes_count + (1 if is_yes else 0),
            brier_term_sum=self.brier_term_sum + _forecast_term(pair),
            pnl_pips=self.pnl_pips + _traded_pnl_pips(pair),
        )


@dataclass(frozen=True, slots=True)
class PriceBucket:
    """One executable-price decile bucket of :func:`price_bucket_report`.

    Attributes:
        bucket_low_pips: Inclusive lower price edge of the bucket, in pips.
        bucket_high_pips: Exclusive upper price edge of the bucket, in pips.
        count: Number of resolved forecasts in the bucket.
        mean_forecast_ppm: Mean forecast probability in the bucket, in ppm.
        observed_frequency_ppm: Observed ``YES`` frequency in the bucket, in ppm.
        brier_ppm: Mean Brier score in the bucket, in ppm.
        pnl_pips: PnL over the bucket's *traded* forecasts only, in pips.
    """

    bucket_low_pips: int
    bucket_high_pips: int
    count: int
    mean_forecast_ppm: int
    observed_frequency_ppm: int
    brier_ppm: int
    pnl_pips: int


def _price_bucket_index(baseline_pips: int) -> int:
    """Return the price-decile index for a baseline ask price, clamped to the top.

    Args:
        baseline_pips: The executable ask price, in pips.

    Returns:
        ``min(baseline_pips // _PRICE_BUCKET_WIDTH_PIPS, PRICE_BUCKET_COUNT - 1)``.
    """
    return min(baseline_pips // _PRICE_BUCKET_WIDTH_PIPS, PRICE_BUCKET_COUNT - 1)


def price_bucket_report(
    inputs: EvaluationInputs, *, window: ObservationWindow
) -> tuple[PriceBucket, ...]:
    """Bucket the resolved forecasts into 10 executable-price deciles (S9.4).

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        Ten contiguous :class:`PriceBucket`s spanning ``[0, 10_000)`` pips.

    Raises:
        ValueError: If no forecast resolves.
    """
    pairs = _scored_pairs(inputs, window=window)
    buckets = [_BucketAccumulator() for _ in range(PRICE_BUCKET_COUNT)]
    for pair in pairs:
        index = _price_bucket_index(pair.baseline_pips)
        buckets[index] = buckets[index].add(pair, key_value=pair.probability_ppm)
    return tuple(
        PriceBucket(
            bucket_low_pips=PRICE_BUCKET_EDGES_PIPS[index],
            bucket_high_pips=PRICE_BUCKET_EDGES_PIPS[index + 1],
            count=bucket.count,
            mean_forecast_ppm=_mean_or_zero(bucket.key_sum, bucket.count),
            observed_frequency_ppm=_observed_frequency_ppm(
                bucket.yes_count, bucket.count
            ),
            brier_ppm=_mean_brier_ppm(bucket.brier_term_sum, bucket.count),
            pnl_pips=bucket.pnl_pips,
        )
        for index, bucket in enumerate(buckets)
    )


@dataclass(frozen=True, slots=True)
class EdgeBucket:
    """One symmetric edge bucket of :func:`edge_bucket_report`.

    Attributes:
        bucket_low_ppm: Inclusive lower edge boundary of the bucket, in ppm.
        bucket_high_ppm: Exclusive upper edge boundary of the bucket, in ppm.
        count: Number of resolved forecasts in the bucket.
        mean_edge_ppm: Mean edge (``p - baseline``) in the bucket, in ppm.
        brier_ppm: Mean Brier score in the bucket, in ppm.
        observed_frequency_ppm: Observed ``YES`` frequency in the bucket, in ppm.
        pnl_pips: PnL over the bucket's *traded* forecasts only, in pips.
    """

    bucket_low_ppm: int
    bucket_high_ppm: int
    count: int
    mean_edge_ppm: int
    brier_ppm: int
    observed_frequency_ppm: int
    pnl_pips: int


def _edge_ppm(pair: _ScoredPair) -> int:
    """Return the forecast's edge over its baseline, in ppm.

    Args:
        pair: The scored forecast/outcome pair.

    Returns:
        ``probability_ppm - baseline_ppm``.
    """
    return pair.probability_ppm - pair.baseline_ppm


def _edge_bucket_index(edge_ppm: int) -> int:
    """Return the symmetric edge-bucket index for an edge value.

    Buckets are half-open ``[low, high)`` except the final bucket, which is closed
    at the top so a maximal ``+PPM_SCALE`` edge lands in it.

    Args:
        edge_ppm: The forecast edge, in ppm.

    Returns:
        The index into :data:`EDGE_BUCKET_EDGES_PPM`'s buckets.
    """
    last = len(EDGE_BUCKET_EDGES_PPM) - 2
    for index in range(last + 1):
        low = EDGE_BUCKET_EDGES_PPM[index]
        high = EDGE_BUCKET_EDGES_PPM[index + 1]
        if low <= edge_ppm < high:
            return index
    return last


def edge_bucket_report(
    inputs: EvaluationInputs, *, window: ObservationWindow
) -> tuple[EdgeBucket, ...]:
    """Bucket the resolved forecasts into six symmetric edge buckets.

    Args:
        inputs: The evaluation inputs to score.
        window: The declared observation-window label, carried through unused by
            this window-agnostic scorer. Per-market slice selection, when
            applicable, is the caller's responsibility, applied via
            :func:`windbreak.evaluation.windows.resolve_window` before scoring.

    Returns:
        Six :class:`EdgeBucket`s over the :data:`EDGE_BUCKET_EDGES_PPM` boundaries.

    Raises:
        ValueError: If no forecast resolves.
    """
    pairs = _scored_pairs(inputs, window=window)
    bucket_count = len(EDGE_BUCKET_EDGES_PPM) - 1
    buckets = [_BucketAccumulator() for _ in range(bucket_count)]
    for pair in pairs:
        edge = _edge_ppm(pair)
        index = _edge_bucket_index(edge)
        buckets[index] = buckets[index].add(pair, key_value=edge)
    return tuple(
        EdgeBucket(
            bucket_low_ppm=EDGE_BUCKET_EDGES_PPM[index],
            bucket_high_ppm=EDGE_BUCKET_EDGES_PPM[index + 1],
            count=bucket.count,
            mean_edge_ppm=_mean_or_zero(bucket.key_sum, bucket.count),
            brier_ppm=_mean_brier_ppm(bucket.brier_term_sum, bucket.count),
            observed_frequency_ppm=_observed_frequency_ppm(
                bucket.yes_count, bucket.count
            ),
            pnl_pips=bucket.pnl_pips,
        )
        for index, bucket in enumerate(buckets)
    )
