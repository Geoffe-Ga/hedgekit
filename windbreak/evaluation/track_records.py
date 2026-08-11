"""The evaluation pass that writes `provider-track-records.json` (issue #440).

WHAT WAS MISSING

:func:`windbreak.scheduler.loop._build_provider_gate` reads each provider's
resolved-forecast count and Brier skill out of one strict-JSON artifact and
never recomputes them, by design. Before this module **nothing in the
repository wrote that artifact**: a whole-tree grep for its filename returned
the constant and two documents describing the read. Both documents called the
empty case a transient bootstrap -- ``docs/RUNBOOK.md``'s "**No artifact yet**
is the bootstrap case", ``EVALUATION.md``'s "Until an evaluation pass writes
the file" -- but with no producer the bootstrap was *terminal*: every provider
unproven forever, every full forecast held back from live eligibility forever,
and ``config.forecast.provider_gate.min_resolved`` unreachable by construction.
This module is that evaluation pass.

WHO WRITES IT, AND WHEN

An operator-run CLI verb, ``windbreak evaluate-providers``
(:func:`windbreak.main._run_evaluate_providers`), not a tick stage.

Ground truth enters this system only through ``windbreak ingest-resolution``
(issue #439): there is no venue settlement feed, so a market's outcome is known
to the running loop exactly when an operator ingests it. A scoring pass that
ran automatically every tick would therefore re-derive the same answer from the
same evidence until the operator acted anyway, while adding a per-tick failure
surface to the always-on loop for a fold that today raises on a malformed or
self-contradicting ledger. Putting the pass at the same cadence as its only
input keeps the evidence chain one operator-driven step wide, and keeps a bad
ledger costing one exit-1 command rather than every subsequent beat.

The gate itself stays what its docstring says it is: a **process-lived read
model** built once, at :func:`windbreak.scheduler.loop.build_paper_deps`. A
freshly written artifact is therefore picked up by the loop's *next start*, and
``docs/RUNBOOK.md``'s "Prove a provider before it can back a live order"
section states that restart in the procedure rather than leaving it implied.

WHAT A PROVIDER'S RECORD ACTUALLY MEASURES, AND WHAT IT DOES NOT

The durable ledger attributes a forecast to a provider through
``ProviderVoteRecorded``: one row per ensemble member driven, carrying the
member's ``provider`` and its ``outcome``. Only a ``"voted"`` outcome counts
here -- an ``"abstained"`` member answered but cast no vote, and a
``"discarded"`` member's vote never reached aggregation, so neither backed the
forecast that was produced.

What that row does **not** carry is the member's own probability. So a
provider's Brier skill here is the skill of *the forecasts it backed*, scored
on the aggregate probability those forecasts were published with -- not the
provider's isolated calibration, which no ledger row records. That is the
honest description of the measurement and it is deliberately narrower than the
metric's name suggests, per SPEC S19's no-unmeasured-edge-claims mandate. It is
also exactly the question the gate asks: *may this provider's vote back a live
order* -- answered by whether the orders its votes backed have historically
beaten the executable-price baseline. Two providers that vote on identical
market sets earn identical records; they diverge exactly where their
abstentions, discards, and vote coverage diverge. Recording each member's own
probability would need a new ``ProviderVoteRecorded`` payload leaf and is
tracked separately.

FAIL CLOSED, IN BOTH DIRECTIONS

The direction of danger is asymmetric. A permanently-held gate blocks a launch;
a permanently-open one lets an unmeasured provider back a live order. So:

* A provider with no ``"voted"`` row, no *admitted* resolved forecast, or an
  **undefined** skill (a baseline that made no error, so the ratio has no
  denominator) is **omitted from the document entirely**. Absent means unproven
  to :func:`~windbreak.forecast.providers.track_record.parse_track_records`,
  and there is no value that could honestly be written for a measurement that
  does not exist -- a ``0`` would be a fabricated observation that a bar of
  ``0`` would happily promote.
* A provider with *real but insufficient* evidence is written with its exact
  values, not omitted. That distinction is load-bearing: PR #481 found that
  deleting the artifact leaves a provider unproven whatever the bars say, so
  only a recorded-but-short record can test the bars themselves.
* Forecasts are gated for temporal integrity first
  (:func:`~windbreak.evaluation.registry.gate_evaluation_inputs`), so a
  forecast created after its market settled, or before the ledger's deployment
  marker, credits nobody. The gate that decides who may trade must not be
  openable with evidence the evaluation harness itself refuses.
* A malformed row raises, naming the offending key and its sequence number. The
  verb catches it, writes nothing, and exits 1; a half-written artifact would
  abort the loop's next startup instead.

RELATION TO THE WEEKLY FOLD

The resolutions, their sequence projection, and the temporal gate are the very
functions :func:`windbreak.scheduler.weekly_data.weekly_report_body` folds with
(:mod:`windbreak.evaluation.ingest`, :mod:`windbreak.evaluation.temporal`), so
the two passes cannot disagree about what settled or when. The
``ForecastCreated`` projection and the deployment-marker rule are stated again
here because ``weekly_data``'s are module-private; extracting the shared fold
belongs to whoever next needs a third caller, and until then the rule is stated
once in each place with this cross-reference rather than silently forked.

Every quantity is an integer count or an integer ppm -- never a float.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from typing import TYPE_CHECKING, Final

from windbreak.evaluation.ingest import (
    ingested_resolutions_from_records,
    resolution_instants,
    resolution_outcomes,
)
from windbreak.evaluation.metrics import UndefinedMetricError, brier_skill
from windbreak.evaluation.registry import (
    EvaluationInputs,
    FixtureForecast,
    gate_evaluation_inputs,
)
from windbreak.evaluation.temporal import (
    TemporalContext,
    resolution_sequences_from_instants,
)
from windbreak.evaluation.windows import HEADLINE_OBSERVATION_WINDOW, resolve_window
from windbreak.forecast.pipeline import VOTE_OUTCOME_VOTED
from windbreak.forecast.providers.track_record import ProviderTrackRecord
from windbreak.ledger.events import ConfigLoaded, ForecastCreated, ProviderVoteRecorded
from windbreak.numeric.types import ProbabilityPpm

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path
    from typing import Any

    from windbreak.ledger.store import LedgerRecord

#: The strict-JSON artifact this pass writes and
#: :func:`windbreak.scheduler.loop._build_provider_gate` reads, resolved inside
#: the ``report_dir`` the loop's other evaluation artifacts live in. Defined
#: here, in the module that *writes* it, and imported by the reader, so the
#: producer and the consumer can never name two different files.
PROVIDER_TRACK_RECORD_FILENAME: Final = "provider-track-records.json"

#: Suffix of the scratch file the document is written to before being moved
#: into place, so a reader never observes a half-written artifact.
_SCRATCH_SUFFIX: Final = ".tmp"

#: Ledger event-type discriminators, each taken from the concrete event class
#: so a renamed event breaks this fold loudly instead of silently matching
#: nothing (which would read as "no evidence" -- a fail-open shape here).
_FORECAST_CREATED_EVENT_TYPE: Final = ForecastCreated.__name__
_PROVIDER_VOTE_RECORDED_EVENT_TYPE: Final = ProviderVoteRecorded.__name__
_CONFIG_LOADED_EVENT_TYPE: Final = ConfigLoaded.__name__

#: Envelope key under which a ledgered event's typed payload is nested.
_PAYLOAD_DATA_KEY: Final = "data"

#: The ``ForecastCreated`` payload leaves this fold reads.
_FORECAST_ID_KEY: Final = "forecast_id"
_MARKET_TICKER_KEY: Final = "market_ticker"
_PROBABILITY_PPM_KEY: Final = "probability_ppm"
_ELIGIBLE_FOR_LIVE_KEY: Final = "eligible_for_live"
_ABSTENTION_REASON_KEY: Final = "abstention_reason"
_BASELINE_PIPS_KEY: Final = "market_price_baseline_pips"

#: Every ``ForecastCreated`` payload key the fold requires, in the order a
#: malformed row is checked against so the first missing one is named. Exposed
#: so a test can assert the set against a real event's own payload rather than
#: against a second hand-written list.
REQUIRED_FORECAST_PAYLOAD_KEYS: Final = (
    _FORECAST_ID_KEY,
    _MARKET_TICKER_KEY,
    _PROBABILITY_PPM_KEY,
    _ELIGIBLE_FOR_LIVE_KEY,
    _ABSTENTION_REASON_KEY,
    _BASELINE_PIPS_KEY,
)

#: The ``ProviderVoteRecorded`` payload leaves this fold reads.
_VOTE_PROVIDER_KEY: Final = "provider"
_VOTE_OUTCOME_KEY: Final = "outcome"

#: The deployment sequence an *empty* fold gates against. Real sequence numbers
#: start at ``1``, so ``0`` admits everything -- inert here precisely because a
#: fold with no records has no forecasts to admit.
_EMPTY_FOLD_DEPLOYMENT_SEQUENCE: Final = 0


def _payload_data(record: LedgerRecord) -> dict[str, Any]:
    """Return one record's typed payload, unwrapped from its envelope.

    Args:
        record: The ledger row to read.

    Returns:
        The row's ``data`` mapping.
    """
    envelope: dict[str, Any] = json.loads(record.payload_json)
    data: dict[str, Any] = envelope[_PAYLOAD_DATA_KEY]
    return data


def _require_forecast_keys(data: Mapping[str, Any], sequence_number: int) -> None:
    """Check that a ``ForecastCreated`` payload carries every key this fold reads.

    Args:
        data: The row's typed payload.
        sequence_number: The row's ledger position, named in the error.

    Raises:
        ValueError: If any required key is absent -- a legacy (pre-#188, v1)
            row, or a payload whose leaves have been renamed -- naming the
            first missing key rather than raising a bare :class:`KeyError` or
            defaulting to a zero baseline that would score as real evidence.
    """
    for key in REQUIRED_FORECAST_PAYLOAD_KEYS:
        if key not in data:
            raise ValueError(
                f"ForecastCreated payload at sequence_number={sequence_number} "
                f"is missing {key!r}"
            )


def _forecast_from_record(record: LedgerRecord) -> FixtureForecast:
    """Fold one ``ForecastCreated`` row into the forecast the metrics score.

    ``traded`` and ``live`` are ``False``: this pass scores the PAPER track, and
    no ledger row on it records a live fill against a forecast.

    Args:
        record: The ``ForecastCreated`` ledger row to fold.

    Returns:
        The typed :class:`~windbreak.evaluation.registry.FixtureForecast`,
        provenanced by the row's own ``sequence_number``.

    Raises:
        ValueError: If the payload is missing a key this fold reads (see
            :func:`_require_forecast_keys`).
    """
    data = _payload_data(record)
    _require_forecast_keys(data, record.sequence_number)
    return FixtureForecast(
        forecast_id=data[_FORECAST_ID_KEY],
        market_ticker=data[_MARKET_TICKER_KEY],
        probability_ppm=ProbabilityPpm(data[_PROBABILITY_PPM_KEY]),
        eligible_for_live=data[_ELIGIBLE_FOR_LIVE_KEY],
        abstention_reason=data[_ABSTENTION_REASON_KEY],
        traded=False,
        baseline_executable_price_pips=int(data[_BASELINE_PIPS_KEY]),
        created_sequence=record.sequence_number,
        live=False,
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


def _deployment_sequence(records: Iterable[LedgerRecord]) -> int:
    """Return the deployment sequence this fold gates its forecasts against.

    The same rule
    :func:`windbreak.scheduler.weekly_data._deployment_sequence` applies, so a
    forecast the weekly report refuses as ``pre_deployment`` cannot earn a
    provider a track record either: deployment is the *earliest*
    ``ConfigLoaded`` row, and a fold carrying no marker cannot prove when the
    system deployed, so it fails closed to the ledger's head -- refusing every
    forecast rather than admitting them on trust.

    Args:
        records: The full ledger read, in append order.

    Returns:
        The earliest ``ConfigLoaded`` sequence number; else the fold's highest
        sequence number; else :data:`_EMPTY_FOLD_DEPLOYMENT_SEQUENCE`.
    """
    materialized = tuple(records)
    markers = [
        record.sequence_number
        for record in materialized
        if record.event_type == _CONFIG_LOADED_EVENT_TYPE
    ]
    if markers:
        return min(markers)
    return max(
        (record.sequence_number for record in materialized),
        default=_EMPTY_FOLD_DEPLOYMENT_SEQUENCE,
    )


def evaluation_inputs_from_records(
    records: Iterable[LedgerRecord],
) -> EvaluationInputs:
    """Fold a whole-ledger read into the inputs a per-provider score is cut from.

    Args:
        records: The full ledger read, in append order.

    Returns:
        The :class:`~windbreak.evaluation.registry.EvaluationInputs` carrying
        every ``ForecastCreated`` row, the ground truth an operator ingested,
        and the temporal coordinates both are adjudicated against.

    Raises:
        ValueError: If a ``ForecastCreated`` row is missing a key this fold
            reads, a ``MarketResolved`` row is malformed or contradicts an
            earlier resolution of the same market, or an instant the projection
            needs is unparseable or carries no UTC offset.
    """
    materialized = tuple(records)
    resolutions = ingested_resolutions_from_records(materialized)
    instants = resolution_instants(resolutions)
    sequences: Mapping[str, int] = {}
    if instants:
        sequences = resolution_sequences_from_instants(
            (_record_instant(record) for record in materialized), instants
        )
    return EvaluationInputs(
        forecasts=tuple(
            _forecast_from_record(record)
            for record in materialized
            if record.event_type == _FORECAST_CREATED_EVENT_TYPE
        ),
        resolutions=resolution_outcomes(resolutions),
        temporal=TemporalContext(
            deployment_sequence=_deployment_sequence(materialized),
            resolution_sequences=sequences,
        ),
    )


def voted_forecast_ids_by_provider(
    records: Iterable[LedgerRecord],
) -> Mapping[str, frozenset[str]]:
    """Attribute each forecast to the providers whose votes actually backed it.

    Only a ``"voted"`` outcome attributes: an ``"abstained"`` member answered
    but cast no vote, and a ``"discarded"`` member's vote never reached
    aggregation, so neither stands behind the published forecast.

    Args:
        records: The full ledger read, in append order.

    Returns:
        Each provider's set of backed ``forecast_id``s; providers that never
        cast a surviving vote are absent.
    """
    backed: dict[str, set[str]] = {}
    for record in records:
        if record.event_type != _PROVIDER_VOTE_RECORDED_EVENT_TYPE:
            continue
        data = _payload_data(record)
        if data[_VOTE_OUTCOME_KEY] != VOTE_OUTCOME_VOTED:
            continue
        backed.setdefault(str(data[_VOTE_PROVIDER_KEY]), set()).add(
            str(data[_FORECAST_ID_KEY])
        )
    return {provider: frozenset(ids) for provider, ids in backed.items()}


def _track_record_for(
    provider: str, backed: frozenset[str], admitted: EvaluationInputs
) -> ProviderTrackRecord | None:
    """Score one provider over the admitted forecasts its votes backed.

    Args:
        provider: The provider being scored.
        backed: The ``forecast_id``s the provider's surviving votes backed.
        admitted: The temporally admitted, resolved forecasts of the whole run.

    Returns:
        The provider's :class:`ProviderTrackRecord`, or ``None`` when it has no
        admitted resolved forecast or its skill is undefined -- both of which
        are omitted from the artifact so the gate reads the provider as
        unproven rather than as a fabricated measurement.
    """
    selected = tuple(
        forecast for forecast in admitted.forecasts if forecast.forecast_id in backed
    )
    windowed = resolve_window(selected, window=HEADLINE_OBSERVATION_WINDOW).forecasts
    if not windowed:
        return None
    scoped = EvaluationInputs(
        forecasts=windowed,
        resolutions=admitted.resolutions,
        temporal=admitted.temporal,
    )
    try:
        skill = brier_skill(scoped, window=HEADLINE_OBSERVATION_WINDOW)
    except UndefinedMetricError:
        return None
    return ProviderTrackRecord(
        provider=provider,
        resolved_count=len(windowed),
        brier_skill_ppm=skill,
    )


def provider_track_records(
    records: Iterable[LedgerRecord],
) -> tuple[ProviderTrackRecord, ...]:
    """Fold a whole-ledger read into every provider's earned track record.

    Args:
        records: The full ledger read, in append order.

    Returns:
        One :class:`ProviderTrackRecord` per provider that earned a defined
        score, sorted by provider. A provider with no admitted resolved
        forecast, or an undefined skill, is absent (fail closed).

    Raises:
        ValueError: If the ledger cannot be folded (see
            :func:`evaluation_inputs_from_records`).
    """
    materialized = tuple(records)
    admitted, _ = gate_evaluation_inputs(evaluation_inputs_from_records(materialized))
    backed_by_provider = voted_forecast_ids_by_provider(materialized)
    scored = (
        _track_record_for(provider, backed_by_provider[provider], admitted)
        for provider in sorted(backed_by_provider)
    )
    return tuple(record for record in scored if record is not None)


def _entry(record: ProviderTrackRecord) -> dict[str, object]:
    """Project one record into its JSON entry, deriving every key.

    The entry's keys come from :func:`dataclasses.fields`, never a restated
    list, so a field added to :class:`ProviderTrackRecord` cannot silently be
    dropped from the written artifact. ``provider`` is emitted alongside the
    two measurements: it is a key
    :func:`~windbreak.forecast.providers.track_record.parse_track_records`
    permits, and the reader takes the provider identity from the outer key
    regardless.

    Args:
        record: The track record to project.

    Returns:
        The record's JSON entry.
    """
    return {
        field.name: getattr(record, field.name) for field in dataclasses.fields(record)
    }


def render_track_record_document(records: Iterable[ProviderTrackRecord]) -> str:
    """Render track records as the strict-JSON document the gate parses.

    Args:
        records: The track records to render.

    Returns:
        The document text, provider-sorted and newline-terminated. An empty
        input renders ``{}``: a pass that ran and proved nothing, which is a
        different claim from an artifact that was never written and a stronger
        one than one that was silently left stale.
    """
    document = {record.provider: _entry(record) for record in records}
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def write_provider_track_records(
    report_dir: Path, records: Iterable[ProviderTrackRecord]
) -> Path:
    """Publish the provider gate's artifact from already-folded track records.

    Publication is separated from the fold on purpose: the caller folds first,
    so a ledger that cannot be folded is refused with nothing written at all,
    rather than after the previous artifact has already been replaced.

    The document is written to a scratch file beside the artifact and moved
    into place, so a concurrent reader -- the loop starting up, say -- either
    sees the previous document or the new one, never a truncated one that would
    abort its startup.

    The write is a *replacement*, never a merge: a provider that no longer
    earns a record disappears from the artifact, so a stale promotion cannot
    outlive the evidence that justified it.

    Args:
        report_dir: The existing evaluation-artifact directory to write into.
        records: The track records to publish.

    Returns:
        The path the artifact was written to.
    """
    document = render_track_record_document(records)
    artifact = report_dir / PROVIDER_TRACK_RECORD_FILENAME
    scratch = report_dir / (PROVIDER_TRACK_RECORD_FILENAME + _SCRATCH_SUFFIX)
    scratch.write_text(document, encoding="utf-8")
    scratch.replace(artifact)
    return artifact
