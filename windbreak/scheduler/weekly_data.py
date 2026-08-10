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
payload, ``traded=False`` and ``live=False``.

Issue #439 gave the fold real ground truth. Until then it hardcoded
``resolutions={}`` and ``deployment_sequence=0``, so every folded forecast was
temporally admitted and then immediately rejected ``UNRESOLVED`` -- which meant
every metric in the always-on loop's only evaluation consumer read ``UNDEFINED``
after seven days exactly as it did after seven minutes. Both coordinates are now
read off the same ledger the forecasts come from:

* ``resolutions`` folds the ``MarketResolved`` rows an operator ingests through
  ``windbreak ingest-resolution`` (see :mod:`windbreak.evaluation.ingest`). A
  market with no such row is still rejected ``UNRESOLVED``, as before.
* ``TemporalContext.resolution_sequences`` projects each resolution's *instant*
  onto the ledger's sequence axis
  (:func:`~windbreak.evaluation.temporal.resolution_sequences_from_instants`),
  so a forecast created after its market settled is refused ``BACKDATED`` even
  when the resolution was ingested long afterwards.
* ``TemporalContext.deployment_sequence`` is the ledger's own deployment
  marker: the first ``ConfigLoaded`` row, which every real run writes before any
  loop event. A fold carrying no marker cannot prove when the system deployed,
  so it fails closed to the ledger's head -- refusing every forecast
  ``PRE_DEPLOYMENT`` rather than admitting them on trust.

Every rejection still lands in the rendered report's ``== rejections ==`` ledger
rather than silently vanishing.

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
from datetime import datetime
from typing import TYPE_CHECKING

from windbreak.evaluation.costs import aggregate_research_costs
from windbreak.evaluation.ingest import (
    ingested_resolutions_from_records,
    resolution_instants,
    resolution_outcomes,
)
from windbreak.evaluation.registry import EvaluationInputs, FixtureForecast
from windbreak.evaluation.report import build_evaluation_report, render_weekly_report
from windbreak.evaluation.temporal import (
    TemporalContext,
    resolution_sequences_from_instants,
)
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

    from windbreak.evaluation.ingest import IngestedResolution
    from windbreak.ledger.store import LedgerRecord

#: The ledger event type discriminating a PAPER-loop forecast record.
_FORECAST_CREATED_EVENT_TYPE = "ForecastCreated"

#: The ledger event type marking the system's first recorded act of a run: the
#: config load that ``windbreak run`` persists before any loop event. It is the
#: ledger analogue of a fixture's ``mode_transitions`` block, and the earliest
#: one is the deployment point the temporal gate measures forecasts against.
_CONFIG_LOADED_EVENT_TYPE = "ConfigLoaded"

#: Envelope key under which a ledgered event's typed payload is nested (mirrors
#: :func:`windbreak.ledger.rebuild._gateway_projection`'s own ``["data"]`` read).
_PAYLOAD_DATA_KEY = "data"

#: The v2 ``ForecastCreated`` research-cost payload key the fold reads verbatim.
_RESEARCH_COST_KEY = "research_cost_micros"

#: The v2 ``ForecastCreated`` baseline-price payload key the fold reads verbatim.
_BASELINE_PIPS_KEY = "market_price_baseline_pips"

#: The deployment sequence an *empty* fold gates against. Real sequence numbers
#: start at ``1``, so ``0`` admits everything -- which is inert here precisely
#: because a fold with no records has no forecasts to admit. A non-empty fold
#: never uses this value: it either finds a deployment marker or fails closed to
#: its own head (:func:`_deployment_sequence`).
_EMPTY_FOLD_DEPLOYMENT_SEQUENCE = 0


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


def _deployment_sequence(records: list[LedgerRecord]) -> int:
    """Return the deployment sequence this fold gates its forecasts against.

    Deployment is the earliest ``ConfigLoaded`` row: ``windbreak run`` persists
    the config-load events it captured *before* opening the PAPER loop, so on a
    live ledger that row is the system's first recorded act. Taking the minimum
    (rather than the newest) mirrors
    :func:`~windbreak.evaluation.temporal.deployment_sequence_from_fixture`,
    which takes the minimum ``mode_transitions`` sequence: a restart writes
    another marker, and deployment is still the first one.

    A fold that carries no marker cannot prove when the system deployed, so it
    fails closed to the ledger's head -- every forecast is then at or before the
    deployment sequence and is refused ``PRE_DEPLOYMENT``, visibly, in the
    report's rejection ledger. Admitting them instead would let a ledger seeded
    with simulated forecasts score as if the system had made them live.

    Args:
        records: The full ledger read, in append order.

    Returns:
        The earliest ``ConfigLoaded`` sequence number; else the fold's highest
        sequence number; else :data:`_EMPTY_FOLD_DEPLOYMENT_SEQUENCE` when the
        fold is empty (and so has no forecasts to gate).
    """
    markers = [
        record.sequence_number
        for record in records
        if record.event_type == _CONFIG_LOADED_EVENT_TYPE
    ]
    if markers:
        return min(markers)
    return max(
        (record.sequence_number for record in records),
        default=_EMPTY_FOLD_DEPLOYMENT_SEQUENCE,
    )


def _record_instant(record: LedgerRecord) -> tuple[int, datetime]:
    """Return one record's ``(sequence_number, created_at)`` position.

    Args:
        record: The ledger row to read.

    Returns:
        The row's sequence number paired with its parsed creation instant.

    Raises:
        ValueError: If ``created_at`` is not an ISO-8601 instant, naming the
            offending row so it is locatable on a live ledger.
    """
    try:
        moment = datetime.fromisoformat(record.created_at)
    except ValueError as exc:
        raise ValueError(
            f"ledger row at sequence_number={record.sequence_number} carries an "
            f"unparseable created_at: {record.created_at!r}"
        ) from exc
    return record.sequence_number, moment


def _temporal_context(
    records: list[LedgerRecord], resolutions: tuple[IngestedResolution, ...]
) -> TemporalContext:
    """Build the temporal context this fold gates against.

    The ledger's ``created_at`` column is parsed only when at least one market
    has actually resolved: with no ground truth there is no instant to project,
    and a fold that never resolves anything should not be able to fail on a
    malformed timestamp in a row it does not consult.

    Args:
        records: The full ledger read, in append order.
        resolutions: The ground truth folded out of the same records.

    Returns:
        The :class:`TemporalContext` carrying the real deployment sequence and
        the instant-derived resolution sequences.

    Raises:
        ValueError: If a row's ``created_at``, or a resolution's instant, is
            unparseable or carries no UTC offset.
    """
    instants = resolution_instants(resolutions)
    sequences: Mapping[str, int] = {}
    if instants:
        sequences = resolution_sequences_from_instants(
            (_record_instant(record) for record in records), instants
        )
    return TemporalContext(
        deployment_sequence=_deployment_sequence(records),
        resolution_sequences=sequences,
    )


def weekly_report_body(records: list[LedgerRecord], *, today: date) -> str:
    """Fold a whole-ledger read into a rendered weekly report body (#188).

    Every ``ForecastCreated`` record is folded into a :class:`FixtureForecast`
    and a research-cost source; non-forecast records are ignored. Every
    ``MarketResolved`` record is folded into the run's ground truth, and every
    forecast is then gated against it: one whose market never resolved is still
    rejected ``UNRESOLVED``, one created after its market settled is rejected
    ``BACKDATED``, and one that predates the ledger's deployment marker is
    rejected ``PRE_DEPLOYMENT``. Each rejection lands in the report's
    ``== rejections ==`` ledger. The cost meter's total is the summed research
    spend of every folded forecast, admitted or not -- the money was spent
    either way. Its ``resolved forecasts`` count is on the same footing: a
    forecast counts once its *market* resolved, whether or not the temporal
    gate admitted it for scoring, so in a fold carrying rejections that count
    is strictly larger than the number of forecasts a metric was computed over.
    Both denominators are defensible; this one answers "what did research cost
    per question that got an answer", which is the question the meter is for,
    and it is asserted at exactly that divergence in
    ``tests/scheduler/test_weekly_resolution_fold.py``.

    Args:
        records: The full ledger read (``SqliteLedgerStore.read_all()``), in
            append order.
        today: The report date stamped into the rendered body.

    Returns:
        The rendered markdown weekly-report body.

    Raises:
        ValueError: If any ``ForecastCreated`` record is a legacy v1 row missing
            a v2 cost/baseline key (message names the missing key); if a
            ``MarketResolved`` record is malformed or contradicts an earlier
            resolution of the same market; or if an instant a resolution is
            gated against is unparseable or carries no UTC offset.
    """
    forecasts: list[FixtureForecast] = []
    cost_rows: list[_ForecastCostRow] = []
    for record in records:
        if record.event_type != _FORECAST_CREATED_EVENT_TYPE:
            continue
        forecast, cost_row = _forecast_from_record(record)
        forecasts.append(forecast)
        cost_rows.append(cost_row)
    resolutions = ingested_resolutions_from_records(records)
    outcomes = resolution_outcomes(resolutions)
    inputs = EvaluationInputs(
        forecasts=tuple(forecasts),
        resolutions=outcomes,
        temporal=_temporal_context(records, resolutions),
    )
    cost_meter = aggregate_research_costs(
        cost_rows, resolutions=outcomes, trade_pnls_micros={}
    )
    return render_weekly_report(
        today=today,
        evaluation=build_evaluation_report(inputs),
        costs=cost_meter,
        equity_lines=render_equity_lines(equity_curve_read_model(records)),
        position_lines=render_position_lines(positions_read_model(records)),
        decision_lines=render_decision_lines(selector_decisions_read_model(records)),
    )
