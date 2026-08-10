"""A mathematically undefined metric renders UNDEFINED, never crashes (#439).

Three registered metrics are undefined over inputs that are otherwise perfectly
valid:

* `calibration_slope` / `calibration_intercept` need `var(p) != 0`, which a
  single resolved forecast can never have;
* `brier_skill_vs_executable_price` needs a baseline that made *some* error;
* `log_score` diverges on a probability of exactly `0`.

Each raised a bare `ValueError` that nothing caught on the report path. That
was unreachable while the weekly fold hardcoded `resolutions={}` --
`NoResolvedForecastsError` always fired first and *was* caught -- and became
reachable the moment issue #439 gave the fold real ground truth. The first
market ever to resolve would have taken the whole weekly report down with it,
because one resolved forecast has zero forecast variance by definition.

`UndefinedMetricError` is the narrow vocabulary for "this metric has no value
over these inputs", distinct from "these inputs are malformed". The *report*
maps the former to the `UNDEFINED` sentinel and still lets the latter
propagate: an undefined metric must read as undefined, never as a number and
never as silence.

The mapping deliberately does NOT live in the registry's `gated_compute` choke
point. The other consumer of a registered spec, `crosscheck_gates`, exists to
notice Python-vs-SQL disagreement and renders a raising reference metric as its
own `PYTHON_COMPUTE_FAILED` sentinel; swallowing the raise at the choke point
would have left that guard unreachable -- a guard made inert by the very change
meant to strengthen the surrounding code.
"""

from __future__ import annotations

import pytest

from windbreak.evaluation.cohorts import UNDEFINED
from windbreak.evaluation.metrics import (
    NoResolvedForecastsError,
    UndefinedMetricError,
    brier_skill,
    calibration_intercept,
    calibration_slope,
    mean_log_score,
)
from windbreak.evaluation.registry import (
    EvaluationInputs,
    FixtureForecast,
    MetricSpec,
    MetricValue,
    Track,
    registered_metrics,
)
from windbreak.evaluation.report import build_evaluation_report
from windbreak.evaluation.resolution import ResolutionOutcome
from windbreak.evaluation.temporal import TemporalContext
from windbreak.evaluation.windows import ObservationWindow
from windbreak.numeric.types import ProbabilityPpm

#: The window every metric here is asked for.
_WINDOW = ObservationWindow.LATEST_BEFORE_CLOSE

#: The metrics that go undefined over a single resolved forecast.
_UNDEFINED_OVER_ONE_FORECAST = ("calibration_slope", "calibration_intercept")


def _forecast(
    *,
    forecast_id: str = "fc-1",
    market_ticker: str = "MKT-A",
    probability_ppm: int = 250_000,
    baseline_pips: int = 4_600,
) -> FixtureForecast:
    """Build one admitted-by-default forecast row.

    Args:
        forecast_id: The forecast's identifier.
        market_ticker: The market the forecast names.
        probability_ppm: The forecast probability, in ppm.
        baseline_pips: The baseline executable price, in pips.

    Returns:
        The assembled `FixtureForecast`, provenanced at sequence 2.
    """
    return FixtureForecast(
        forecast_id=forecast_id,
        market_ticker=market_ticker,
        probability_ppm=ProbabilityPpm(probability_ppm),
        eligible_for_live=False,
        abstention_reason=None,
        traded=False,
        baseline_executable_price_pips=baseline_pips,
        created_sequence=2,
    )


def _inputs(
    *forecasts: FixtureForecast, **outcomes: ResolutionOutcome
) -> EvaluationInputs:
    """Build temporally-admitting inputs over `forecasts` and `outcomes`.

    Args:
        forecasts: The forecast rows to score.
        outcomes: Each market's settled outcome, keyed by ticker.

    Returns:
        The assembled `EvaluationInputs`.
    """
    return EvaluationInputs(
        forecasts=forecasts,
        resolutions=dict(outcomes),
        temporal=TemporalContext(
            deployment_sequence=1,
            resolution_sequences=dict.fromkeys(outcomes, 99),
        ),
    )


def test_no_resolved_forecasts_error_is_an_undefined_metric_error() -> None:
    """The pre-existing empty-resolved-set guard is one of these, by type."""
    assert issubclass(NoResolvedForecastsError, UndefinedMetricError)
    assert issubclass(UndefinedMetricError, ValueError)


def test_calibration_slope_over_one_forecast_raises_undefined_metric_error() -> None:
    """One resolved forecast has zero forecast variance, so the slope has no value."""
    inputs = _inputs(_forecast(), **{"MKT-A": ResolutionOutcome.NO})

    with pytest.raises(UndefinedMetricError) as exc_info:
        calibration_slope(inputs, window=_WINDOW)

    assert type(exc_info.value) is UndefinedMetricError
    assert str(exc_info.value) == (
        "forecast variance is zero; calibration slope is undefined"
    )


def test_calibration_intercept_over_one_forecast_raises_undefined_metric_error() -> (
    None
):
    """The intercept shares the slope's variance requirement, and its refusal."""
    inputs = _inputs(_forecast(), **{"MKT-A": ResolutionOutcome.NO})

    with pytest.raises(UndefinedMetricError) as exc_info:
        calibration_intercept(inputs, window=_WINDOW)

    assert type(exc_info.value) is UndefinedMetricError


def test_brier_skill_against_a_perfect_baseline_raises_undefined_metric_error() -> None:
    """A baseline that made no error leaves the skill ratio with no denominator."""
    inputs = _inputs(_forecast(baseline_pips=0), **{"MKT-A": ResolutionOutcome.NO})

    with pytest.raises(UndefinedMetricError) as exc_info:
        brier_skill(inputs, window=_WINDOW)

    assert type(exc_info.value) is UndefinedMetricError
    assert str(exc_info.value) == "baseline Brier-term sum is zero; skill is undefined"


def test_log_score_on_a_certain_wrong_forecast_raises_undefined_metric_error() -> None:
    """`-ln(0)` diverges, so a certain-and-wrong forecast has no log score."""
    inputs = _inputs(_forecast(probability_ppm=0), **{"MKT-A": ResolutionOutcome.YES})

    with pytest.raises(UndefinedMetricError) as exc_info:
        mean_log_score(inputs, window=_WINDOW)

    assert type(exc_info.value) is UndefinedMetricError
    assert str(exc_info.value) == (
        "log-score probability is 0 (certain-wrong); -ln(0) diverges"
    )


@pytest.mark.parametrize("metric_name", _UNDEFINED_OVER_ONE_FORECAST)
def test_the_report_renders_an_undefined_metric_rather_than_crashing(
    metric_name: str,
) -> None:
    """`build_evaluation_report` prints the sentinel instead of raising.

    This is the property the weekly fold depends on: the first market ever to
    resolve leaves exactly one resolved forecast, and the report must print
    `UNDEFINED` for the metrics that genuinely have no value over it rather
    than take the whole tick down.

    Args:
        metric_name: The registry name of the metric under test.
    """
    inputs = _inputs(_forecast(), **{"MKT-A": ResolutionOutcome.NO})

    report = build_evaluation_report(inputs)

    values = {
        result.name: result.value for track in report.tracks for result in track.metrics
    }
    assert metric_name in values
    assert values[metric_name] is UNDEFINED
    assert values["brier"] == 62_500
    assert f"{metric_name} [latest_before_close] = UNDEFINED" in report.render_text()


def test_the_registry_choke_point_still_lets_the_raise_reach_the_crosscheck() -> None:
    """The mapping lives in the report, not in `gated_compute`.

    `crosscheck_gates` renders a raising reference metric as its own
    `PYTHON_COMPUTE_FAILED` sentinel, which is how a Python-vs-SQL disagreement
    stays visible. Swallowing `UndefinedMetricError` at the registry choke point
    would have made that guard unreachable, so the raise must still escape a
    registered spec's `compute`.
    """
    inputs = _inputs(_forecast(), **{"MKT-A": ResolutionOutcome.NO})

    with pytest.raises(UndefinedMetricError):
        registered_metrics()["calibration_slope"].compute(inputs)


def test_a_malformed_input_value_error_is_not_rendered_as_undefined() -> None:
    """The report's catch is by type, not a blanket `except ValueError`.

    A missing temporal context over a populated run is an invalid-input
    `ValueError`, not an undefined metric, and must still reach the caller
    rather than being quietly printed as a dash.
    """
    inputs = EvaluationInputs(
        forecasts=(_forecast(),),
        resolutions={"MKT-A": ResolutionOutcome.NO},
        temporal=None,
    )

    with pytest.raises(ValueError) as exc_info:
        build_evaluation_report(inputs)

    assert not isinstance(exc_info.value, UndefinedMetricError)
    assert "requires a TemporalContext" in str(exc_info.value)


def test_an_invalid_input_raise_is_not_swallowed_by_the_report_helper() -> None:
    """The report's catch is `UndefinedMetricError`, not a blanket `ValueError`.

    Widening it would render every broken metric as a dash, which is exactly
    the failure mode this issue exists to remove: a report cell that reads
    "no value" when the truth is "the computation is broken". A spec whose
    `compute` raises a plain `ValueError` must therefore still propagate
    through the report's per-metric helper.

    The synthetic spec goes through the real `MetricSpec` constructor, so it
    carries the registry's temporal choke-point wrapper exactly as a registered
    metric does.
    """
    from windbreak.evaluation.report import _metric_value_or_undefined

    def _broken(inputs: EvaluationInputs) -> MetricValue:
        """Raise a plain, non-undefined `ValueError`.

        Args:
            inputs: The gated evaluation inputs (unused).

        Returns:
            Never returns.

        Raises:
            ValueError: Always, standing in for a genuinely broken metric.
        """
        del inputs
        raise ValueError("metric implementation is broken")

    spec = MetricSpec(
        name="broken",
        track=Track.FORECAST,
        window=_WINDOW,
        compute=_broken,
    )
    inputs = _inputs(_forecast(), **{"MKT-A": ResolutionOutcome.NO})

    with pytest.raises(ValueError) as exc_info:
        _metric_value_or_undefined(spec, inputs)

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == "metric implementation is broken"
