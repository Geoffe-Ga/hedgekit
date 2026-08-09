"""SPEC S8.4 two-stage triage with per-forecast cost ledgering.

The forecast engine spends most of its budget on the twelve-stage full pipeline
(:func:`windbreak.forecast.pipeline.run_pipeline`). SPEC S8.4 puts a cheap
Stage-0 *prior* in front of it: a single model call yields a rough
probability, and the expensive pipeline runs *only if* that prior diverges from
the executable-price baseline by at least the triage threshold (or an operator
flag / refresh forces it). When the prior stays within the band, the run stops
early and emits a schema-valid ``triage_only`` :class:`ForecastRecord` carrying
just the Stage-0 cost -- never a live-eligible record, since no research backed
it.

Every decision is ledgered through the :class:`TriageLedgerWriter` seam (a
dependency-injection point modeled on
:class:`windbreak.connector.snapshot.EventLedgerWriter`): a ``TRIAGE_STOP`` or
``TRIAGE_PROCEED`` :class:`TriageEvent` whose payload leaves are exact
``int``/``str``/``bool`` values -- never a float, per the package-wide no-float
convention ``scripts/lint_no_floats.py`` enforces. All arithmetic here is
integer-only for the same reason.

The PROCEED path additionally threads an optional
:class:`windbreak.forecast.pipeline.ForecastLedgerWriter` *discard* ledger into
the full pipeline, so a triage-path run ledgers discarded model outputs
identically to a direct :func:`windbreak.forecast.pipeline.run_pipeline` call.
This is a separate seam from the ``TriageLedgerWriter`` above (which records
only the STOP/PROCEED decision itself).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol

from windbreak.forecast.cassettes import LlmRequest
from windbreak.forecast.pipeline import (
    _forecast_horizon_hours,
    normalize_question,
    outside_view_base_rate,
    run_pipeline,
)
from windbreak.forecast.records import ForecastRecord

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.cassettes import LlmTransport
    from windbreak.forecast.pipeline import ForecastLedgerWriter
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

#: Stage-0 divergence threshold, in ppm (SPEC S16 default). The full pipeline
#: runs only when the prior is at least this far from the baseline.
TRIAGE_THRESHOLD_PPM = 50_000

#: The per-forecast research *budget*, in micros (SPEC S16
#: ``budget.per_forecast_micros``). This is the spending ceiling for a whole
#: forecast; it is deliberately distinct from
#: ``windbreak.forecast.pipeline``'s private ``_RESEARCH_COST_MICROS`` stub
#: *cost* for a full run, which coincidentally equals the same figure today.
PER_FORECAST_BUDGET_MICROS = 3_000_000

#: Divisor turning the per-forecast budget into the Stage-0 cost ceiling: the
#: cheap prior may consume at most 2% (1/50) of the budget.
_STAGE0_COST_DIVISOR = 50

#: The fixed Stage-0 prior cost, in micros (2% of the per-forecast budget).
_STAGE0_COST_MICROS = PER_FORECAST_BUDGET_MICROS // _STAGE0_COST_DIVISOR

#: Event type recorded when a run stops at the triage stage (no full pipeline).
TRIAGE_STOP_EVENT = "TRIAGE_STOP"

#: Event type recorded when a run proceeds through the full pipeline.
TRIAGE_PROCEED_EVENT = "TRIAGE_PROCEED"

#: Lowest legal ppm prior (0.0 probability).
_MIN_PRIOR_PPM = 0

#: Highest legal ppm prior (1.0 probability).
_MAX_PRIOR_PPM = 1_000_000

#: The triage-only stage tag (SPEC S8.4), used both as the record's
#: ``triage_stage`` and as the id-namespacing tag so a triage-only forecast id
#: can never collide with a full record for identical provenance.
_TRIAGE_ONLY_STAGE: Final = "triage_only"

#: Deterministic rationale stamped on every triage-only record.
_TRIAGE_RATIONALE_MD = (
    "## Triage-only forecast\n\n"
    "The Stage-0 prior fell within the triage band, so the full research "
    "pipeline was not run. This record is live-ineligible.\n"
)


class TriageModel(NamedTuple):
    """The Stage-0 triage model's provenance strings.

    Mirrors :class:`windbreak.forecast.providers.EnsembleMember`: pinning the
    provider/version keeps the single Stage-0 request byte-stable across runs.
    A plain structural NamedTuple, it is the injection point an operator-facing
    config->forecast wiring layer builds from a
    :class:`windbreak.config.schema.ModelRef`-shaped ``(provider, model)`` pair
    -- so ``windbreak.forecast.triage`` never needs to import
    ``windbreak.config`` (the SPEC S8.3 sandbox boundary).

    Attributes:
        provider: The LLM provider identifier.
        model_version: The pinned model version string.
    """

    provider: str
    model_version: str


#: The single pinned model that produces the Stage-0 prior by default; omitting
#: the ``model=``/``triage_model=`` keyword is byte-identical to passing this.
_DEFAULT_TRIAGE_MODEL = TriageModel("openai", "gpt-5-triage-mini")


@dataclass(frozen=True, slots=True)
class TriagePrior:
    """The cheap Stage-0 prior and its cost (SPEC S8.4).

    Attributes:
        prior_ppm: The Stage-0 probability estimate, in ppm.
        cost_micros: The Stage-0 research cost, in micros.
    """

    prior_ppm: int
    cost_micros: int


@dataclass(frozen=True, slots=True)
class TriageEvent:
    """One recorded triage decision (mirrors ``ConnectorEvent``).

    Attributes:
        event_type: The event kind (``TRIAGE_STOP`` or ``TRIAGE_PROCEED``).
        payload: The JSON-safe event body (int/str/bool leaves only).
        ts: ISO-8601 UTC timestamp of when the event was created.
    """

    event_type: str
    payload: Mapping[str, object]
    ts: str


class TriageLedgerWriter(Protocol):
    """The seam through which a triage decision is persisted."""

    def record(self, event: TriageEvent) -> None:
        """Persist a triage event.

        Args:
            event: The event to persist.
        """
        ...


class InMemoryTriageLedger:
    """A :class:`TriageLedgerWriter` that retains events in memory for tests."""

    def __init__(self) -> None:
        """Initialize with an empty event log."""
        self._events: list[TriageEvent] = []

    def record(self, event: TriageEvent) -> None:
        """Append a triage event to the in-memory log.

        Args:
            event: The event to retain.
        """
        self._events.append(event)

    def events_by_type(self, event_type: str) -> tuple[TriageEvent, ...]:
        """Return every retained event of a given type, in record order.

        Args:
            event_type: The event kind to filter by.

        Returns:
            The matching events.
        """
        return tuple(event for event in self._events if event.event_type == event_type)


def _iso_z(moment: datetime) -> str:
    """Render a datetime as ISO-8601 UTC with a trailing ``Z``.

    Follows the local-``_iso_z`` precedent in ``pipeline.py`` and ``records.py``
    (each module defines its own) rather than reaching across for a shared one.

    Args:
        moment: The (timezone-aware) datetime to render; normalized to UTC.

    Returns:
        A string like ``2024-12-10T12:00:00.000000Z``.
    """
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _stage0_prompt(market: NormalizedMarket, baseline: BaselineQuoteSnapshot) -> str:
    """Build the deterministic Stage-0 prompt from question + baseline.

    Args:
        market: The market under triage.
        baseline: The baseline quote snapshot.

    Returns:
        A deterministic prompt string keyed on the normalized-question hash.
    """
    question_hash = normalize_question(market)
    return (
        f"Stage-0 triage: estimate the resolution probability in ppm for "
        f"question {question_hash}; baseline {baseline.price_pips} pips."
    )


def _parse_prior_ppm(response: str) -> int:
    """Parse a Stage-0 response into a validated ppm prior, fail-closed.

    The response must be a bare integer string within ``[0, 1_000_000]``. A
    non-integer (e.g. ``"0.52"`` or ``"maybe"``) or an out-of-range value fails
    loudly rather than silently defaulting; the message names the offending
    text.

    Args:
        response: The raw Stage-0 completion text.

    Returns:
        The parsed ppm prior.

    Raises:
        ValueError: If ``response`` is not an integer or falls outside
            ``[0, 1_000_000]``.
    """
    try:
        value = int(response)
    except ValueError as exc:
        message = f"stage-0 prior must be an integer ppm string, got {response!r}"
        raise ValueError(message) from exc
    if not _MIN_PRIOR_PPM <= value <= _MAX_PRIOR_PPM:
        message = (
            f"stage-0 prior {response!r} is outside "
            f"[{_MIN_PRIOR_PPM}, {_MAX_PRIOR_PPM}]"
        )
        raise ValueError(message)
    return value


def run_stage0_prior(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    *,
    transport: LlmTransport,
    model: TriageModel = _DEFAULT_TRIAGE_MODEL,
) -> TriagePrior:
    """Run the cheap Stage-0 prior with a single model call (SPEC S8.4).

    Issues exactly one deterministic :class:`LlmRequest` on ``model``, parses
    the response into a validated ppm prior (fail-closed on a non-integer or
    out-of-range value), and pairs it with the fixed Stage-0 cost. Omitting
    ``model`` uses the pinned :data:`_DEFAULT_TRIAGE_MODEL`, byte-identical to
    the request built before ``model`` existed as a keyword.

    Args:
        market: The market under triage.
        baseline: The baseline quote snapshot.
        transport: The LLM transport for the single Stage-0 call (keyword-only).
        model: The triage model whose provider/version drive the sent request;
            defaults to the pinned :data:`_DEFAULT_TRIAGE_MODEL`.

    Returns:
        The Stage-0 prior and its cost.

    Raises:
        ValueError: If the response is not an integer ppm within
            ``[0, 1_000_000]``.
    """
    request = LlmRequest(
        provider=model.provider,
        model_version=model.model_version,
        prompt=_stage0_prompt(market, baseline),
    )
    response = transport.complete(request)
    prior_ppm = _parse_prior_ppm(response)
    return TriagePrior(prior_ppm=prior_ppm, cost_micros=_STAGE0_COST_MICROS)


def should_run_full_pipeline(
    prior_ppm: int,
    baseline_ppm: int,
    *,
    triage_threshold_ppm: int,
    operator_flagged: bool,
    refresh_triggered: bool,
) -> bool:
    """Decide whether the expensive full pipeline should run (SPEC S8.4).

    The pipeline runs when the prior diverges from the baseline by at least the
    threshold (a ``>=`` boundary, per SPEC), or when an operator flag or a
    refresh trigger forces it.

    Args:
        prior_ppm: The Stage-0 prior probability, in ppm.
        baseline_ppm: The executable-price baseline probability, in ppm.
        triage_threshold_ppm: The minimum divergence that forces the pipeline.
        operator_flagged: Whether an operator forced a full run.
        refresh_triggered: Whether a refresh forced a full run.

    Returns:
        ``True`` if the full pipeline should run, else ``False``.
    """
    return (
        abs(prior_ppm - baseline_ppm) >= triage_threshold_ppm
        or operator_flagged
        or refresh_triggered
    )


@dataclass(frozen=True, slots=True)
class _TriageContext:
    """The gating inputs threaded from decision to payload building.

    Attributes:
        market_ticker: The triaged market's ticker.
        prior: The Stage-0 prior and its cost.
        baseline_ppm: The executable-price baseline probability, in ppm.
        triage_threshold_ppm: The divergence threshold used for the decision.
        operator_flagged: Whether an operator forced a full run.
        refresh_triggered: Whether a refresh forced a full run.
        triage_model: The Stage-0 triage model, stamped into the ledger payload
            for model provenance.
    """

    market_ticker: str
    prior: TriagePrior
    baseline_ppm: int
    triage_threshold_ppm: int
    operator_flagged: bool
    refresh_triggered: bool
    triage_model: TriageModel


def _base_payload(context: _TriageContext) -> dict[str, object]:
    """Build the JSON-safe payload leaves common to both event types.

    Args:
        context: The gating inputs of the triage decision.

    Returns:
        A mapping of int/str/bool leaves (never a float).
    """
    return {
        "market_ticker": context.market_ticker,
        "prior_ppm": context.prior.prior_ppm,
        "baseline_ppm": context.baseline_ppm,
        "triage_threshold_ppm": context.triage_threshold_ppm,
        "operator_flagged": context.operator_flagged,
        "refresh_triggered": context.refresh_triggered,
        "triage_cost_micros": context.prior.cost_micros,
        "triage_provider": context.triage_model.provider,
        "triage_model_version": context.triage_model.model_version,
    }


def _triage_forecast_id(
    question_hash: str, snapshot_id: str, created_at: datetime
) -> str:
    """Derive a deterministic, stage-namespaced triage-only forecast id.

    The ``triage_only`` tag is folded into the canonical JSON so a triage-only
    id can never collide with a full record's id for identical provenance.

    Args:
        question_hash: The normalized-question hash.
        snapshot_id: The baseline snapshot identifier.
        created_at: The forecast creation instant.

    Returns:
        A sha256 hex digest over the canonical JSON of the provenance tuple.
    """
    canonical = json.dumps(
        [question_hash, snapshot_id, _iso_z(created_at), _TRIAGE_ONLY_STAGE],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_triage_only_record(
    *,
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    prior: TriagePrior,
    created_at: datetime,
) -> ForecastRecord:
    """Assemble a schema-valid ``triage_only`` record from the Stage-0 prior.

    The prior collapses the point estimate and both confidence bounds onto the
    same ppm value (there is no ensemble), carries only the Stage-0 cost, and
    is permanently live-ineligible.

    Args:
        market: The market under triage.
        baseline: The baseline quote snapshot.
        prior: The Stage-0 prior and its cost.
        created_at: The forecast creation instant.

    Returns:
        A schema-valid, immutable triage-only forecast record.
    """
    question_hash = normalize_question(market)
    return ForecastRecord(
        forecast_id=_triage_forecast_id(
            question_hash, baseline.snapshot_id, created_at
        ),
        market_ticker=market.ticker,
        normalized_question_hash=question_hash,
        probability_ppm=prior.prior_ppm,
        ci_low_ppm=prior.prior_ppm,
        ci_high_ppm=prior.prior_ppm,
        model_votes=(),
        vote_dispersion_ppm=0,
        rationale_markdown=_TRIAGE_RATIONALE_MD,
        citations=(),
        source_quality_notes=(),
        research_cost_micros=prior.cost_micros,
        triage_stage=_TRIAGE_ONLY_STAGE,
        created_at=created_at,
        forecast_horizon_hours=_forecast_horizon_hours(market, created_at),
        market_price_baseline_pips=baseline.price_pips,
        baseline_quote_snapshot_id=baseline.snapshot_id,
        coherence_group_sum_ppm=None,
        coherence_flag=False,
        abstention_reason=None,
        eligible_for_live=False,
    )


def _run_stop_path(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    context: _TriageContext,
    *,
    created_at: datetime,
    ledger: TriageLedgerWriter,
) -> ForecastRecord:
    """Handle the STOP path: build the triage-only record and ledger a STOP.

    The full transport is never touched on this path.

    Args:
        market: The market under triage.
        baseline: The baseline quote snapshot.
        context: The gating inputs of the triage decision.
        created_at: The forecast creation instant.
        ledger: The triage-event ledger writer.

    Returns:
        The triage-only forecast record.
    """
    record = _build_triage_only_record(
        market=market, baseline=baseline, prior=context.prior, created_at=created_at
    )
    ledger.record(
        TriageEvent(TRIAGE_STOP_EVENT, _base_payload(context), _iso_z(created_at))
    )
    return record


def _run_proceed_path(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    context: _TriageContext,
    *,
    created_at: datetime,
    ledger: TriageLedgerWriter,
    full_transport: LlmTransport,
    research_tools: ResearchTools,
    discard_ledger: ForecastLedgerWriter | None = None,
) -> ForecastRecord:
    """Handle the PROCEED path: run the full pipeline and fold in triage cost.

    Args:
        market: The market under triage.
        baseline: The baseline quote snapshot.
        context: The gating inputs of the triage decision.
        created_at: The forecast creation instant.
        ledger: The triage-event ledger writer.
        full_transport: The transport for the full pipeline's vote stage.
        research_tools: The sandboxed research tools threaded into the full
            pipeline's Stage-5 bounded web research.
        discard_ledger: The optional forecast-event ledger for the full
            pipeline's vote-discard events, or ``None`` to record nothing
            (mirrors ``run_pipeline``'s ``ledger`` seam). Default ``None`` is a
            strict no-op.

    Returns:
        The full forecast record with the Stage-0 cost folded into its total.
    """
    full_record = run_pipeline(
        market,
        baseline,
        transport=full_transport,
        created_at=created_at,
        research_tools=research_tools,
        ledger=discard_ledger,
    )
    folded = replace(
        full_record,
        research_cost_micros=full_record.research_cost_micros
        + context.prior.cost_micros,
    )
    payload = _base_payload(context)
    payload["full_cost_micros"] = full_record.research_cost_micros
    payload["total_research_cost_micros"] = folded.research_cost_micros
    ledger.record(TriageEvent(TRIAGE_PROCEED_EVENT, payload, _iso_z(created_at)))
    return folded


def run_triaged_pipeline(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    *,
    triage_transport: LlmTransport,
    full_transport: LlmTransport,
    ledger: TriageLedgerWriter,
    created_at: datetime,
    research_tools: ResearchTools,
    discard_ledger: ForecastLedgerWriter | None = None,
    triage_threshold_ppm: int = TRIAGE_THRESHOLD_PPM,
    operator_flagged: bool = False,
    refresh_triggered: bool = False,
    triage_model: TriageModel = _DEFAULT_TRIAGE_MODEL,
) -> ForecastRecord:
    """Run the two-stage triaged pipeline into a forecast record (SPEC S8.4).

    Runs the cheap Stage-0 prior, then decides via :func:`should_run_full_pipeline`
    whether to stop (emit a live-ineligible ``triage_only`` record, never
    touching ``full_transport``) or proceed (run the full pipeline and fold the
    Stage-0 cost into the record's total). Exactly one ``TRIAGE_STOP`` or
    ``TRIAGE_PROCEED`` event is ledgered per run. Given identical inputs and
    ``created_at``, two runs produce equal records and event trails.

    Args:
        market: The market under triage.
        baseline: The baseline quote snapshot the forecast is struck against.
        triage_transport: The transport for the single Stage-0 call.
        full_transport: The transport for the full pipeline's vote stage;
            untouched on the STOP path.
        ledger: The triage-event ledger writer.
        created_at: The injected creation instant, for determinism.
        research_tools: The sandboxed research tools threaded into the full
            pipeline's Stage-5 bounded web research; untouched on the STOP path.
        discard_ledger: The optional forecast-event ledger for the full
            pipeline's vote-discard events, threaded through on the PROCEED path
            so a triage-path run ledgers discarded model outputs identically to
            a direct ``run_pipeline`` call (mirrors ``run_pipeline``'s ``ledger``
            seam). Default ``None`` is a strict no-op -- untouched on the STOP
            path, where no vote can be discarded -- leaving output byte-for-byte
            unchanged.
        triage_threshold_ppm: The divergence threshold forcing the full run.
        operator_flagged: Whether an operator forces a full run.
        refresh_triggered: Whether a refresh forces a full run.
        triage_model: The Stage-0 triage model, threaded into
            :func:`run_stage0_prior` and stamped into the ledger payload;
            defaults to the pinned :data:`_DEFAULT_TRIAGE_MODEL`.

    Returns:
        The produced, immutable forecast record.
    """
    prior = run_stage0_prior(
        market, baseline, transport=triage_transport, model=triage_model
    )
    baseline_ppm = outside_view_base_rate(baseline)
    context = _TriageContext(
        market_ticker=market.ticker,
        prior=prior,
        baseline_ppm=baseline_ppm,
        triage_threshold_ppm=triage_threshold_ppm,
        operator_flagged=operator_flagged,
        refresh_triggered=refresh_triggered,
        triage_model=triage_model,
    )
    if should_run_full_pipeline(
        prior.prior_ppm,
        baseline_ppm,
        triage_threshold_ppm=triage_threshold_ppm,
        operator_flagged=operator_flagged,
        refresh_triggered=refresh_triggered,
    ):
        return _run_proceed_path(
            market,
            baseline,
            context,
            created_at=created_at,
            ledger=ledger,
            full_transport=full_transport,
            research_tools=research_tools,
            discard_ledger=discard_ledger,
        )
    return _run_stop_path(
        market, baseline, context, created_at=created_at, ledger=ledger
    )
