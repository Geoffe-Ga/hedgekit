"""Whole-ledger weekly-report fold for the always-on PAPER loop (issue #188).

:func:`weekly_report_body` folds an entire ``SqliteLedgerStore.read_all()``
result into a rendered weekly report, wiring *real*
:class:`~windbreak.evaluation.registry.EvaluationInputs` /
:class:`~windbreak.evaluation.registry.FixtureForecast`s and a real
:class:`~windbreak.evaluation.costs.CostMeter` in place of the #55
``evaluation=None, costs=None`` placeholder the scheduler wrote before.

Issue #255 additionally wires the three original #48 stub sections
(``## Equity vs floor``, ``## Positions``, ``## Decisions``) with real data:
the same ``records`` are folded through the canonical
:func:`~windbreak.ledger.rebuild.equity_curve_read_model` /
:func:`~windbreak.ledger.rebuild.positions_read_model` /
:func:`~windbreak.ledger.rebuild.selector_decisions_read_model` builders and
rendered by the :mod:`windbreak.reports.sections` line renderers, replacing the
``No data yet.`` placeholder those three sections previously hardcoded.

Every ``ForecastCreated`` record folds into a :class:`FixtureForecast` with
``created_sequence`` sourced from the record's ``sequence_number``, its
``baseline_executable_price_pips`` and research cost read verbatim off the
payload, ``traded=False`` and ``live=False``. A whole-ledger fold has no
ground-truth resolutions available at all, so the fold always gates against an
empty ``resolutions`` and a ``TemporalContext`` with ``deployment_sequence=0``:
every folded forecast is temporally admitted (``created_sequence >= 1 > 0``) and
then rejected by the evaluation gate's own ``UNRESOLVED`` reason, landing in the
rendered report's ``== rejections ==`` ledger rather than silently vanishing.

A legacy (pre-#188, ``payload_schema_version == 1``) ``ForecastCreated`` row --
missing ``research_cost_micros`` / ``market_price_baseline_pips`` -- fails closed
with a :class:`ValueError` naming the missing key, never a silent zero-cost
default or a bare :class:`KeyError`.

Every value stays on the integer money/probability path (SPEC S6.1): this
package is on ``scripts/lint_no_floats.py``'s denylist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from windbreak.evaluation.costs import aggregate_research_costs
from windbreak.evaluation.registry import EvaluationInputs, FixtureForecast
from windbreak.evaluation.report import build_evaluation_report, render_weekly_report
from windbreak.evaluation.temporal import TemporalContext
from windbreak.ledger.rebuild import (
    equity_curve_read_model,
    positions_read_model,
    selector_decisions_read_model,
)
from windbreak.numeric.types import ProbabilityPpm
from windbreak.reports.sections import (
    render_decision_lines,
    render_equity_lines,
    render_position_lines,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date
    from typing import Any

    from windbreak.ledger.store import LedgerRecord

#: The ledger event type discriminating a PAPER-loop forecast record.
_FORECAST_CREATED_EVENT_TYPE = "ForecastCreated"

#: Envelope key under which a ledgered event's typed payload is nested (mirrors
#: :func:`windbreak.ledger.rebuild._gateway_projection`'s own ``["data"]`` read).
_PAYLOAD_DATA_KEY = "data"

#: The v2 ``ForecastCreated`` research-cost payload key the fold reads verbatim.
_RESEARCH_COST_KEY = "research_cost_micros"

#: The v2 ``ForecastCreated`` baseline-price payload key the fold reads verbatim.
_BASELINE_PIPS_KEY = "market_price_baseline_pips"

#: The deployment sequence a whole-ledger fold gates against: ``0``, so every
#: folded forecast (``created_sequence >= 1``) is temporally admitted and then
#: rejected ``UNRESOLVED`` (the fold carries no ground-truth resolutions).
_FOLD_DEPLOYMENT_SEQUENCE = 0


@dataclass(frozen=True, slots=True)
class _ForecastCostRow:
    """A lightweight research-cost source folded from one ``ForecastCreated`` row.

    Carries only the two attributes
    :func:`~windbreak.evaluation.costs.aggregate_research_costs` reads (its
    structural ``_ResearchCostSource`` surface), so the fold never reconstructs
    a full :class:`~windbreak.forecast.records.ForecastRecord` just to meter
    research spend.

    Attributes:
        market_ticker: The market the forecast is for.
        research_cost_micros: The forecast's research spend, in micros.
    """

    market_ticker: str
    research_cost_micros: int


def _require_payload_int(data: Mapping[str, Any], key: str) -> int:
    """Return a required integer payload field, failing closed on a legacy row.

    Args:
        data: The ``ForecastCreated`` event's typed payload.
        key: The required key -- a v2-only cost/baseline field.

    Returns:
        The key's integer value.

    Raises:
        ValueError: If ``key`` is absent -- a legacy (pre-#188, v1) row -- with
            a message naming the missing key, rather than a silent zero default
            or a bare :class:`KeyError`.
    """
    if key not in data:
        raise ValueError(
            f"legacy v1 ForecastCreated payload is missing {key!r}; "
            "cannot fold a pre-#188 row"
        )
    return int(data[key])


def _forecast_from_record(
    record: LedgerRecord,
) -> tuple[FixtureForecast, _ForecastCostRow]:
    """Fold one ``ForecastCreated`` ledger row into a forecast and cost source.

    The research cost is read before the baseline price so a legacy row missing
    both fails closed naming ``research_cost_micros`` first.

    Args:
        record: The ``ForecastCreated`` ledger row to fold.

    Returns:
        The typed :class:`FixtureForecast` (temporally provenanced by the
        record's ``sequence_number``) paired with its :class:`_ForecastCostRow`.

    Raises:
        ValueError: If the payload is a legacy v1 row missing a v2 cost/baseline
            key (message names the missing key).
    """
    envelope: dict[str, Any] = json.loads(record.payload_json)
    data: dict[str, Any] = envelope[_PAYLOAD_DATA_KEY]
    research_cost = _require_payload_int(data, _RESEARCH_COST_KEY)
    baseline_pips = _require_payload_int(data, _BASELINE_PIPS_KEY)
    forecast = FixtureForecast(
        forecast_id=data["forecast_id"],
        market_ticker=data["market_ticker"],
        probability_ppm=ProbabilityPpm(data["probability_ppm"]),
        eligible_for_live=data["eligible_for_live"],
        abstention_reason=data["abstention_reason"],
        traded=False,
        baseline_executable_price_pips=baseline_pips,
        created_sequence=record.sequence_number,
        live=False,
    )
    cost_row = _ForecastCostRow(
        market_ticker=data["market_ticker"],
        research_cost_micros=research_cost,
    )
    return forecast, cost_row


def weekly_report_body(records: list[LedgerRecord], *, today: date) -> str:
    """Fold a whole-ledger read into a rendered weekly report body (#188).

    Every ``ForecastCreated`` record is folded into a :class:`FixtureForecast`
    and a research-cost source; non-forecast records are ignored. The fold gates
    against no resolutions (a whole-ledger fold has no ground truth), so every
    folded forecast lands in the report's ``== rejections ==`` ledger, and the
    cost meter's total is the summed research spend of every folded forecast.

    Args:
        records: The full ledger read (``SqliteLedgerStore.read_all()``), in
            append order.
        today: The report date stamped into the rendered body.

    Returns:
        The rendered markdown weekly-report body.

    Raises:
        ValueError: If any ``ForecastCreated`` record is a legacy v1 row missing
            a v2 cost/baseline key (message names the missing key).
    """
    forecasts: list[FixtureForecast] = []
    cost_rows: list[_ForecastCostRow] = []
    for record in records:
        if record.event_type != _FORECAST_CREATED_EVENT_TYPE:
            continue
        forecast, cost_row = _forecast_from_record(record)
        forecasts.append(forecast)
        cost_rows.append(cost_row)
    inputs = EvaluationInputs(
        forecasts=tuple(forecasts),
        resolutions={},
        temporal=TemporalContext(
            deployment_sequence=_FOLD_DEPLOYMENT_SEQUENCE,
            resolution_sequences={},
        ),
    )
    cost_meter = aggregate_research_costs(
        cost_rows, resolutions={}, trade_pnls_micros={}
    )
    return render_weekly_report(
        today=today,
        evaluation=build_evaluation_report(inputs),
        costs=cost_meter,
        equity_lines=render_equity_lines(equity_curve_read_model(records)),
        position_lines=render_position_lines(positions_read_model(records)),
        decision_lines=render_decision_lines(selector_decisions_read_model(records)),
    )
