"""Failing-first tests for `render_weekly_report` / `generate_weekly_report`
(issue #55, RED).

Issue #195 adds a fourth optional section, `## Providers` (fleet
observability): `render_weekly_report` gains a keyword-only
`provider_lines: str | None = None` -- the pre-rendered body
`windbreak.reports.providers.render_provider_lines` produces, embedded
verbatim, mirroring `evaluation`/`costs`'s own "pre-built object, `None` ->
`No data yet.` fallback" contract exactly for `evaluation` (whose body is
likewise produced by a call the caller makes, not by `render_weekly_report`
itself: `evaluation.render_text()`). This is a CONSCIOUS, minimal update to
three pre-existing "no data yet" occurrence-count assertions below (each
grows by exactly one, since every existing call site in this file omits the
new keyword and therefore hits its `None` -> fallback default) -- noted at
each touched assertion.

Neither symbol exists yet on `windbreak.evaluation.report`, so every test below
imports them as the FIRST statement inside the test body (matching this
package's established RED convention; see `test_preregistration.py` /
`test_cohorts.py`) so each test collects and fails independently on its own
`ImportError: cannot import name 'render_weekly_report' from
'windbreak.evaluation.report'` (or `'generate_weekly_report'`).

Pins issue #55's weekly-report extension of the #48 stub
(`windbreak.reports.weekly`):

- `render_weekly_report(*, today, evaluation, costs) -> str` is pure markdown:
  it always preserves the stub's three original headings (`## Equity vs
  floor`, `## Positions`, `## Decisions`, each still `No data yet.` -- this
  issue does not wire that data), and adds two new sections: `## Evaluation`
  (the verbatim output of `evaluation.render_text()` when `evaluation` is not
  `None`, else `No data yet.`) and `## Cost meter` (the `str()` rendering of
  each of `CostMeter`'s three `MoneyMicros` fields when `costs` is not `None`,
  else `No data yet.`).
- `generate_weekly_report(output_dir, *, today, evaluation, costs) -> Path`
  delegates naming and ISO-week idempotence to
  `windbreak.reports.weekly.maybe_write_weekly`, passing the rendered body
  through; calling it twice within the same ISO week returns the same
  already-written file untouched.

Issue #188 extracts the tail of `run_evaluation` (gate -> power -> cohorts ->
abstentions -> tracks) into a new public
`build_evaluation_report(inputs: EvaluationInputs) ->
EvaluationReport`, so the scheduler's weekly fold can build a report straight
from a whole-ledger fold's `EvaluationInputs` without going through the
fixture-file loader. It must set `power=None` (catching the new
`windbreak.evaluation.metrics.NoResolvedForecastsError`) when every forecast
is unresolved, and `_render_cost_meter` gains a `total_research_cost_micros`
line plus the three denominator counts, so a real (if still pre-resolution)
`CostMeter` renders observably differently from the bare `costs=None`
fallback even when every per-unit `MoneyMicros` field is `n/a`.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from windbreak.evaluation.registry import Track
from windbreak.evaluation.report import EvaluationReport, TrackReport
from windbreak.numeric.types import MoneyMicros

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The epic-wide known-answer fixture shared by issues #49-#55, reused here
#: (issue #188) to pin `build_evaluation_report`'s byte-identical extraction
#: from `run_evaluation`'s tail.
_SYNTHETIC_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "synthetic_known_answer.json"
)


def _empty_evaluation_report() -> EvaluationReport:
    """Build a minimal, valid `EvaluationReport` with no metrics in any track.

    Returns:
        An `EvaluationReport` whose `render_text()` renders only the three
        bare `== <track> ==` section headers, with no metric lines and no
        `No data yet.` substring anywhere -- so this fixture cannot be
        confused with either side's own "no data" fallback text.
    """
    return EvaluationReport(
        tracks=(
            TrackReport(name=Track.FORECAST.value, metrics=()),
            TrackReport(name=Track.SELECTION.value, metrics=()),
            TrackReport(name=Track.EXECUTION.value, metrics=()),
        )
    )


def _sample_cost_meter() -> object:
    """Build a `CostMeter` with three distinct, non-zero `MoneyMicros` fields.

    Returns:
        A `CostMeter` whose three money fields each render to a distinct,
        greppable `str()` value.
    """
    from windbreak.evaluation.costs import CostMeter

    return CostMeter(
        total_research_cost_micros=7_000_000,
        resolved_forecast_count=2,
        profitable_trade_count=1,
        trade_count=2,
        cost_per_resolved_forecast_micros=MoneyMicros(3_500_000),
        cost_per_profitable_trade_micros=MoneyMicros(7_000_000),
        cost_adjusted_expectancy_micros=MoneyMicros(-1_500_000),
    )


# ---------------------------------------------------------------------------
# 1. render_weekly_report: both sections populated.
# ---------------------------------------------------------------------------


def test_render_weekly_report_includes_stub_headings_and_both_new_sections() -> None:
    """Populated `evaluation`/`costs` render both new sections' real content.

    The three original stub headings (`Equity vs floor`, `Positions`,
    `Decisions`) still render `No data yet.` -- this issue does not wire that
    data -- so exactly 3 `No data yet.` occurrences remain (none contributed
    by the new sections, since both fixtures carry real content).
    """
    from windbreak.evaluation.report import render_weekly_report

    evaluation = _empty_evaluation_report()
    costs = _sample_cost_meter()
    today = date(2024, 3, 4)

    body = render_weekly_report(today=today, evaluation=evaluation, costs=costs)

    assert "# Weekly report 2024-03-04" in body
    assert "## Equity vs floor" in body
    assert "## Positions" in body
    assert "## Decisions" in body
    assert "## Evaluation" in body
    assert "## Cost meter" in body

    assert evaluation.render_text() in body
    assert str(MoneyMicros(3_500_000)) in body
    assert str(MoneyMicros(7_000_000)) in body
    assert str(MoneyMicros(-1_500_000)) in body

    # Issue #195: a fourth "## Providers" section joins the three original
    # stub headings, defaulting to "No data yet." here since this call
    # supplies no `provider_lines` -- 3 stub sections + 1 Providers fallback.
    assert body.count("No data yet.") == 4


# ---------------------------------------------------------------------------
# 2. render_weekly_report: both None -> fallback text under both sections.
# ---------------------------------------------------------------------------


def test_render_weekly_report_shows_no_data_yet_for_both_none_arguments() -> None:
    """`evaluation=None, costs=None` fall back to `No data yet.` in both
    new sections, on top of the three the stub always carries -- five total.
    """
    from windbreak.evaluation.report import render_weekly_report

    today = date(2024, 3, 4)

    body = render_weekly_report(today=today, evaluation=None, costs=None)

    assert "## Evaluation" in body
    assert "## Cost meter" in body
    # Issue #195: was 5 (3 stub + evaluation + costs); +1 for the new
    # "## Providers" section's own `provider_lines=None` -> fallback.
    assert body.count("No data yet.") == 6


def test_render_weekly_report_evaluation_and_costs_fall_back_independently() -> None:
    """`evaluation=None` with `costs` populated renders the cost section's
    real content alongside the evaluation section's fallback (and vice
    versa) -- the two sections are independent, not all-or-nothing.
    """
    from windbreak.evaluation.report import render_weekly_report

    today = date(2024, 3, 4)
    costs = _sample_cost_meter()

    body_costs_only = render_weekly_report(today=today, evaluation=None, costs=costs)
    assert str(MoneyMicros(3_500_000)) in body_costs_only
    # Issue #195: 3 stub sections + the evaluation section's own fallback +
    # the new Providers section's own fallback (no `provider_lines` supplied
    # here either) = 5 (was 4).
    assert body_costs_only.count("No data yet.") == 5

    evaluation = _empty_evaluation_report()
    body_evaluation_only = render_weekly_report(
        today=today, evaluation=evaluation, costs=None
    )
    assert evaluation.render_text() in body_evaluation_only
    assert body_evaluation_only.count("No data yet.") == 5


# ---------------------------------------------------------------------------
# 2b. render_weekly_report: the new `## Providers` section (issue #195).
# ---------------------------------------------------------------------------


def test_render_weekly_report_providers_section_defaults_to_no_data_yet() -> None:
    """With no `provider_lines` supplied, the `## Providers` section renders
    the same `No data yet.` fallback every other data-less section uses --
    never an error or a silently omitted heading.
    """
    from windbreak.evaluation.report import render_weekly_report

    body = render_weekly_report(today=date(2024, 3, 4), evaluation=None, costs=None)

    assert "## Providers" in body
    providers_section = body.split("## Providers", 1)[1]
    assert "No data yet." in providers_section


def test_render_weekly_report_embeds_provider_lines_verbatim_when_supplied() -> None:
    """A supplied `provider_lines` string is embedded verbatim under
    `## Providers`, exactly like `evaluation.render_text()` is embedded under
    `## Evaluation`.
    """
    from windbreak.evaluation.report import render_weekly_report

    provider_lines = (
        "provider=futuresearch resolved=212 brier_skill_ppm=+14200 "
        "cost_per_forecast=n/a abstain_rate=9% canary=OK"
    )

    body = render_weekly_report(
        today=date(2024, 3, 4),
        evaluation=None,
        costs=None,
        provider_lines=provider_lines,
    )

    assert "## Providers" in body
    assert provider_lines in body
    providers_section = body.split("## Providers", 1)[1]
    assert "No data yet." not in providers_section


# ---------------------------------------------------------------------------
# 2c. render_weekly_report: the equity/position/decision line params
#     (issue #255) embed verbatim under their existing #48 stub headings,
#     mirroring the `## Providers` embed-or-fallback contract (2b above) for
#     the three ORIGINAL stub sections instead of a fourth new one. Every
#     call site above omits these three new keywords too, so none of the
#     "No data yet." occurrence-count assertions in sections 1-2 change: the
#     three stub headings already contributed exactly one "No data yet." each
#     to those counts, and still do when their new param defaults to `None`.
# ---------------------------------------------------------------------------


def test_render_weekly_report_equity_lines_section_defaults_to_no_data_yet() -> None:
    """With no `equity_lines` supplied, `## Equity vs floor` still renders the
    shared `No data yet.` fallback -- never an error or an omitted heading.
    """
    from windbreak.evaluation.report import render_weekly_report

    body = render_weekly_report(today=date(2024, 3, 4), evaluation=None, costs=None)

    section = body.split("## Equity vs floor", 1)[1].split("## Positions", 1)[0]
    assert "No data yet." in section


def test_render_weekly_report_embeds_equity_lines_verbatim_when_supplied() -> None:
    """A supplied `equity_lines` string is embedded verbatim under `## Equity
    vs floor`, exactly like `provider_lines` is embedded under `## Providers`.
    """
    from windbreak.evaluation.report import render_weekly_report

    equity_lines = (
        "equity_samples=1\n"
        "epoch_s=1700 equity_micros=11000000 floor_micros=10000000 "
        "buffer_micros=+1000000"
    )

    body = render_weekly_report(
        today=date(2024, 3, 4),
        evaluation=None,
        costs=None,
        equity_lines=equity_lines,
    )

    section = body.split("## Equity vs floor", 1)[1].split("## Positions", 1)[0]
    assert equity_lines in section
    assert "No data yet." not in section


def test_render_weekly_report_position_lines_section_defaults_to_no_data_yet() -> None:
    """With no `position_lines` supplied, `## Positions` still renders the
    shared `No data yet.` fallback.
    """
    from windbreak.evaluation.report import render_weekly_report

    body = render_weekly_report(today=date(2024, 3, 4), evaluation=None, costs=None)

    section = body.split("## Positions", 1)[1].split("## Decisions", 1)[0]
    assert "No data yet." in section


def test_render_weekly_report_embeds_position_lines_verbatim_when_supplied() -> None:
    """A supplied `position_lines` string is embedded verbatim under
    `## Positions`.
    """
    from windbreak.evaluation.report import render_weekly_report

    position_lines = (
        "snapshots=1\n"
        "open_positions=1\n"
        "ticker=MKT-DEEP quantity_centis=200 average_price_pips=4600"
    )

    body = render_weekly_report(
        today=date(2024, 3, 4),
        evaluation=None,
        costs=None,
        position_lines=position_lines,
    )

    section = body.split("## Positions", 1)[1].split("## Decisions", 1)[0]
    assert position_lines in section
    assert "No data yet." not in section


def test_render_weekly_report_decision_lines_section_defaults_to_no_data_yet() -> None:
    """With no `decision_lines` supplied, `## Decisions` still renders the
    shared `No data yet.` fallback.
    """
    from windbreak.evaluation.report import render_weekly_report

    body = render_weekly_report(today=date(2024, 3, 4), evaluation=None, costs=None)

    section = body.split("## Decisions", 1)[1].split("## Evaluation", 1)[0]
    assert "No data yet." in section


def test_render_weekly_report_embeds_decision_lines_verbatim_when_supplied() -> None:
    """A supplied `decision_lines` string is embedded verbatim under
    `## Decisions`.
    """
    from windbreak.evaluation.report import render_weekly_report

    decision_lines = (
        "decision_events=1\n"
        "event=SelectorDecisionRecorded subject=MKT-DEEP "
        "reasons=net_edge_below_minimum"
    )

    body = render_weekly_report(
        today=date(2024, 3, 4),
        evaluation=None,
        costs=None,
        decision_lines=decision_lines,
    )

    section = body.split("## Decisions", 1)[1].split("## Evaluation", 1)[0]
    assert decision_lines in section
    assert "No data yet." not in section


def test_render_weekly_report_equity_position_decision_lines_are_independent() -> None:
    """The three new params are independent, not all-or-nothing: supplying
    only `equity_lines` leaves `## Positions`/`## Decisions` on their own
    fallback, and supplying all three lands each body under its own heading
    with no cross-contamination between sections.
    """
    from windbreak.evaluation.report import render_weekly_report

    today = date(2024, 3, 4)

    equity_only_body = render_weekly_report(
        today=today, evaluation=None, costs=None, equity_lines="equity_samples=0"
    )
    equity_section = equity_only_body.split("## Equity vs floor", 1)[1].split(
        "## Positions", 1
    )[0]
    positions_section = equity_only_body.split("## Positions", 1)[1].split(
        "## Decisions", 1
    )[0]
    decisions_section = equity_only_body.split("## Decisions", 1)[1].split(
        "## Evaluation", 1
    )[0]
    assert "equity_samples=0" in equity_section
    assert "No data yet." not in equity_section
    assert "No data yet." in positions_section
    assert "No data yet." in decisions_section

    all_three_body = render_weekly_report(
        today=today,
        evaluation=None,
        costs=None,
        equity_lines="equity_samples=0",
        position_lines="snapshots=0",
        decision_lines="decision_events=0",
    )
    equity_section = all_three_body.split("## Equity vs floor", 1)[1].split(
        "## Positions", 1
    )[0]
    positions_section = all_three_body.split("## Positions", 1)[1].split(
        "## Decisions", 1
    )[0]
    decisions_section = all_three_body.split("## Decisions", 1)[1].split(
        "## Evaluation", 1
    )[0]
    assert "equity_samples=0" in equity_section
    assert "snapshots=0" not in equity_section
    assert "snapshots=0" in positions_section
    assert "decision_events=0" not in positions_section
    assert "decision_events=0" in decisions_section
    assert "equity_samples=0" not in decisions_section
    assert "No data yet." not in equity_section
    assert "No data yet." not in positions_section
    assert "No data yet." not in decisions_section


# ---------------------------------------------------------------------------
# 3. generate_weekly_report: writes the dated file; ISO-week idempotence.
# ---------------------------------------------------------------------------


def test_generate_weekly_report_writes_dated_file_with_the_rendered_body(
    tmp_path: Path,
) -> None:
    """`generate_weekly_report` writes `weekly-YYYY-MM-DD.md` with the body
    `render_weekly_report` would have produced for the same arguments (this
    direct invocation is exactly what a scheduler hook would call).
    """
    from windbreak.evaluation.report import (
        generate_weekly_report,
        render_weekly_report,
    )

    today = date(2024, 3, 4)
    evaluation = _empty_evaluation_report()
    costs = _sample_cost_meter()

    path = generate_weekly_report(
        tmp_path, today=today, evaluation=evaluation, costs=costs
    )

    assert path == tmp_path / "weekly-2024-03-04.md"
    assert path.exists()
    assert path.read_text(encoding="utf-8") == render_weekly_report(
        today=today, evaluation=evaluation, costs=costs
    )


def test_generate_weekly_report_is_idempotent_within_the_same_iso_week(
    tmp_path: Path,
) -> None:
    """A second call within the same ISO calendar week returns the first
    file untouched -- routed through `windbreak.reports.weekly.maybe_write_weekly`
    -- even though the second call's `today` and arguments differ.

    2024-03-04 is a Monday and 2024-03-05 a Tuesday of the same ISO week.
    """
    from windbreak.evaluation.report import generate_weekly_report

    first = generate_weekly_report(
        tmp_path, today=date(2024, 3, 4), evaluation=None, costs=None
    )
    first_body = first.read_text(encoding="utf-8")

    second = generate_weekly_report(
        tmp_path,
        today=date(2024, 3, 5),
        evaluation=_empty_evaluation_report(),
        costs=_sample_cost_meter(),
    )

    assert second == first
    assert second.read_text(encoding="utf-8") == first_body


def test_generate_weekly_report_writes_a_new_file_the_following_iso_week(
    tmp_path: Path,
) -> None:
    """A call in a later ISO week writes a second, distinctly-named file."""
    from windbreak.evaluation.report import generate_weekly_report

    first = generate_weekly_report(
        tmp_path, today=date(2024, 3, 4), evaluation=None, costs=None
    )
    second = generate_weekly_report(
        tmp_path, today=date(2024, 3, 11), evaluation=None, costs=None
    )

    assert first != second
    assert first.exists()
    assert second.exists()


# ---------------------------------------------------------------------------
# 4. build_evaluation_report: extracted from run_evaluation's tail (issue #188).
# ---------------------------------------------------------------------------


def _unresolved_inputs(count: int) -> object:
    """Build `EvaluationInputs` carrying `count` forecasts, none resolved.

    Args:
        count: How many distinct, temporally-admissible-but-unresolved
            forecasts to build.

    Returns:
        The typed `EvaluationInputs`, with an empty `resolutions` mapping and
        a `TemporalContext` that admits every forecast past the deployment
        gate (so each is rejected for `UNRESOLVED`, not `PRE_DEPLOYMENT`).
    """
    from windbreak.evaluation.registry import EvaluationInputs, FixtureForecast
    from windbreak.evaluation.temporal import TemporalContext
    from windbreak.numeric.types import ProbabilityPpm

    forecasts = tuple(
        FixtureForecast(
            forecast_id=f"fc-{index}",
            market_ticker=f"MKT-{index}",
            probability_ppm=ProbabilityPpm(500_000),
            eligible_for_live=True,
            abstention_reason=None,
            traded=False,
            baseline_executable_price_pips=5_000,
            created_sequence=index + 1,
        )
        for index in range(count)
    )
    return EvaluationInputs(
        forecasts=forecasts,
        resolutions={},
        temporal=TemporalContext(deployment_sequence=0, resolution_sequences={}),
    )


def test_build_evaluation_report_over_unresolved_inputs_sets_power_none() -> None:
    """`power` is `None` when every forecast is unresolved (issue #188): the
    metrics-level `NoResolvedForecastsError` the bootstrap/power path raises
    on an empty resolved set is caught here, exactly as the forecast-track
    metrics' own `gated_compute` adapter catches it, rather than crashing
    `build_evaluation_report` outright.
    """
    from windbreak.evaluation.report import build_evaluation_report

    report = build_evaluation_report(_unresolved_inputs(3))

    assert report.power is None


def test_build_report_unresolved_ledgers_one_rejection_per_forecast() -> None:
    """All N unresolved forecasts are ledgered as rejections, none silently
    dropped.
    """
    from windbreak.evaluation.report import build_evaluation_report

    report = build_evaluation_report(_unresolved_inputs(3))

    assert len(report.rejections) == 3
    assert {rejection.forecast_id for rejection in report.rejections} == {
        "fc-0",
        "fc-1",
        "fc-2",
    }


def test_build_report_unresolved_forecast_track_all_undefined() -> None:
    """Every forecast-track metric renders the `UNDEFINED` sentinel over an
    all-unresolved fold, never a raised exception (issue #188's
    `NoResolvedForecastsError` -> `UNDEFINED` adapter at the registry choke
    point).
    """
    from windbreak.evaluation.cohorts import UNDEFINED
    from windbreak.evaluation.registry import Track as RegistryTrack
    from windbreak.evaluation.report import build_evaluation_report

    report = build_evaluation_report(_unresolved_inputs(2))

    forecast_track = next(
        track for track in report.tracks if track.name == RegistryTrack.FORECAST.value
    )
    assert forecast_track.metrics, "expected at least one forecast-track metric"
    for metric in forecast_track.metrics:
        assert metric.value is UNDEFINED, f"{metric.name} = {metric.value!r}"


def _inputs_from_fixture_payload(payload: Mapping[str, Any]) -> object:
    """Build typed `EvaluationInputs` from the shared fixture, via public API
    only (mirrors `run_evaluation`'s own internal, private loader) so this
    test does not reach into `windbreak.evaluation.report`'s private helpers.

    Args:
        payload: The decoded known-answer fixture payload.

    Returns:
        The typed `EvaluationInputs`, including the fixture's temporal
        context.
    """
    from windbreak.evaluation.registry import EvaluationInputs, FixtureForecast
    from windbreak.evaluation.resolution import (
        resolutions_from_fixture,
        settlement_events_from_fixture,
    )
    from windbreak.evaluation.temporal import (
        TemporalContext,
        deployment_sequence_from_fixture,
        resolution_sequences_from_events,
    )
    from windbreak.numeric.types import ProbabilityPpm

    forecasts = tuple(
        FixtureForecast(
            forecast_id=entry["forecast_id"],
            market_ticker=entry["market_ticker"],
            probability_ppm=ProbabilityPpm(entry["probability_ppm"]),
            eligible_for_live=entry["eligible_for_live"],
            abstention_reason=entry["abstention_reason"],
            traded=entry["traded"],
            baseline_executable_price_pips=entry["baseline_executable_price_pips"],
            correlation_group_id=entry.get("correlation_group_id"),
            created_sequence=entry.get("created_sequence"),
        )
        for entry in payload["forecasts"]
    )
    resolutions = resolutions_from_fixture(payload)
    resolution_sequences = resolution_sequences_from_events(
        settlement_events_from_fixture(payload)
    )
    deployment_sequence = deployment_sequence_from_fixture(payload)
    temporal = TemporalContext(
        deployment_sequence=deployment_sequence,
        resolution_sequences=resolution_sequences,
    )
    return EvaluationInputs(
        forecasts=forecasts, resolutions=resolutions, temporal=temporal
    )


def test_build_report_matches_run_evaluation_over_fixture() -> None:
    """`build_evaluation_report`, fed the identical typed inputs `run_evaluation`
    itself builds from the shared known-answer fixture, renders byte-for-byte
    the same report text (issue #188): extracting the function from
    `run_evaluation`'s tail must change structure, never content.
    """
    from windbreak.evaluation.report import build_evaluation_report, run_evaluation

    payload = json.loads(_SYNTHETIC_FIXTURE.read_text(encoding="utf-8"))
    inputs = _inputs_from_fixture_payload(payload)

    direct_report = build_evaluation_report(inputs)
    loaded_report = run_evaluation(fixture_path=_SYNTHETIC_FIXTURE)

    assert direct_report.render_text() == loaded_report.render_text()


# ---------------------------------------------------------------------------
# 5. _render_cost_meter: total + counts make a wired meter observably
#    different from `costs=None`, even when every per-unit field is `n/a`
#    (issue #188).
# ---------------------------------------------------------------------------


def _zero_denominator_cost_meter() -> object:
    """Build a `CostMeter` with a non-zero total but every count at `0`.

    Returns:
        A `CostMeter` whose three per-unit `MoneyMicros` fields are all
        `None` (rendering `n/a`) -- exactly what
        `windbreak.evaluation.costs.aggregate_research_costs` returns for a
        whole-ledger fold before any market has resolved or any trade has
        been taken -- while `total_research_cost_micros` is a distinct,
        non-zero, greppable value.
    """
    from windbreak.evaluation.costs import CostMeter

    return CostMeter(
        total_research_cost_micros=4_200_007,
        resolved_forecast_count=0,
        profitable_trade_count=0,
        trade_count=0,
        cost_per_resolved_forecast_micros=None,
        cost_per_profitable_trade_micros=None,
        cost_adjusted_expectancy_micros=None,
    )


def test_cost_meter_prints_total_when_all_per_unit_na() -> None:
    """The Cost meter section prints `total_research_cost_micros` even when
    every per-unit field is `n/a` -- otherwise a real, wired-but-pre-resolution
    meter would render byte-identically to the three bare `n/a` lines a
    meter with genuinely zero research spend would also produce, losing the
    one piece of information (total spend already incurred) this state
    actually carries.
    """
    from windbreak.evaluation.report import render_weekly_report

    costs = _zero_denominator_cost_meter()
    today = date(2024, 3, 4)

    body = render_weekly_report(today=today, evaluation=None, costs=costs)
    cost_section = body.split("## Cost meter", 1)[1]

    assert str(costs.total_research_cost_micros) in cost_section
    assert cost_section.count("n/a") == 3


def test_render_weekly_report_cost_meter_prints_the_three_denominator_counts() -> None:
    """`resolved_forecast_count`, `profitable_trade_count`, and `trade_count`
    each render as text in the Cost meter section, not just the three
    per-unit `MoneyMicros` fields.
    """
    from windbreak.evaluation.costs import CostMeter
    from windbreak.evaluation.report import render_weekly_report

    costs = CostMeter(
        total_research_cost_micros=9_000_003,
        resolved_forecast_count=601,
        profitable_trade_count=402,
        trade_count=1103,
        cost_per_resolved_forecast_micros=MoneyMicros(1_500_000),
        cost_per_profitable_trade_micros=MoneyMicros(2_250_000),
        cost_adjusted_expectancy_micros=MoneyMicros(300_000),
    )
    today = date(2024, 3, 4)

    body = render_weekly_report(today=today, evaluation=None, costs=costs)
    cost_section = body.split("## Cost meter", 1)[1]

    assert "601" in cost_section
    assert "402" in cost_section
    assert "1103" in cost_section
    assert "9000003" in cost_section
