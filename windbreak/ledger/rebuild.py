"""Fold the ledger into derived read models (SPEC S5.1, issue #13).

``rebuild`` is a pure projection over a verified ledger: it verifies the
hash chain, then folds the records in sequence order into eleven byte-stable
read-model files -- ``config_versions.json`` (the ``ConfigLoaded`` rows),
``mode_history.json`` (the ``ModeHeartbeat`` rows), ``gateway_events.json``
(the chronological Order Gateway / crash-recovery events, issue #40), the
three PAPER-loop projections ``positions.json`` / ``equity_curve.json`` /
``selector_decisions.json`` (issue #48), the two live-divergence
projections ``execution_quality.json`` / ``live_divergence.json`` (issue #58),
the two fleet-observability projections ``canary_status.json`` (the
latest-per-provider ``CanaryVerdictRecorded``) / ``forecasts.json`` (every
``ForecastCreated`` row, issue #195), and the per-provider vote-cost aggregate
``provider_vote_costs.json`` (issue #281).
``AlertEmitted`` and any unrecognized event types are skipped. Because
verification runs first, a corrupt ledger raises :class:`ChainIntegrityError`
instead of producing a plausible-but-wrong projection.

``rebuild_command`` adapts ``rebuild`` to the ``windbreak rebuild`` CLI,
returning 0 on success and 1 when the chain fails verification or the ledger
path does not exist -- printing the offending ``sequence_number`` or the
missing-path guidance to stderr, respectively.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, cast

from windbreak.ledger.store import ChainIntegrityError, SqliteLedgerStore

if TYPE_CHECKING:
    from argparse import Namespace
    from pathlib import Path

    from windbreak.ledger.store import LedgerRecord

#: Read-model filename holding the ``ConfigLoaded`` projection.
_CONFIG_VERSIONS_FILENAME = "config_versions.json"

#: Read-model filename holding the ``ModeHeartbeat`` projection.
_MODE_HISTORY_FILENAME = "mode_history.json"

#: Read-model filename holding the chronological Order Gateway / crash-recovery
#: event projection (issue #40).
_GATEWAY_EVENTS_FILENAME = "gateway_events.json"

_CONFIG_LOADED = "ConfigLoaded"
_MODE_HEARTBEAT = "ModeHeartbeat"

#: The Order Gateway / crash-recovery event types projected, verbatim, into
#: ``gateway_events.json`` (issue #38-#40), in chronological ledger order.
_GATEWAY_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "OrderTransitionLedgered",
        "SubmissionRefused",
        "ReduceOnlyRefused",
        "ReduceOnlyViolation",
        "ReconciliationHalted",
        "ReconciliationHealed",
        "RecoveryCompleted",
    }
)

#: Read-model filename holding the latest open-positions snapshot (issue #48).
_POSITIONS_FILENAME = "positions.json"

#: Read-model filename holding every equity sample in ledger order (issue #48).
_EQUITY_CURVE_FILENAME = "equity_curve.json"

#: Read-model filename holding the interleaved selector/intent decision trail
#: (issue #48).
_SELECTOR_DECISIONS_FILENAME = "selector_decisions.json"

_POSITIONS_SNAPSHOT_RECORDED = "PositionsSnapshotRecorded"
_EQUITY_SAMPLED = "EquitySampled"

#: The PAPER-loop selector/intent event types projected, in ledger order, into
#: ``selector_decisions.json`` (issue #48): the scheduler's own
#: ``SelectorDecisionRecorded`` plus the two bare intent-verdict events the Risk
#: Kernel already emits (``IntentApproved``/``IntentVetoed``).
_SELECTOR_DECISION_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "SelectorDecisionRecorded",
        "IntentApproved",
        "IntentVetoed",
    }
)


def _config_projection(record: LedgerRecord) -> dict[str, object]:
    """Project a ``ConfigLoaded`` record into its read-model entry.

    Args:
        record: A ``ConfigLoaded`` ledger record.

    Returns:
        The ``{seq, created_at, config_hash, diff}`` read-model entry.
    """
    data = json.loads(record.payload_json)["data"]
    return {
        "seq": record.sequence_number,
        "created_at": record.created_at,
        "config_hash": data["config_hash"],
        "diff": data["diff"],
    }


def _mode_projection(record: LedgerRecord) -> dict[str, object]:
    """Project a ``ModeHeartbeat`` record into its read-model entry.

    Args:
        record: A ``ModeHeartbeat`` ledger record.

    Returns:
        The ``{seq, created_at, mode, beat}`` read-model entry.
    """
    data = json.loads(record.payload_json)["data"]
    return {
        "seq": record.sequence_number,
        "created_at": record.created_at,
        "mode": data["mode"],
        "beat": data["beat"],
    }


def mode_history_read_model(records: list[LedgerRecord]) -> list[dict[str, object]]:
    """Project every ``ModeHeartbeat`` row, in ledger order, into its read model.

    Wraps the private :func:`_mode_projection` the same way
    :func:`equity_curve_read_model` wraps :func:`_gateway_projection`, so the
    dashboard's ledger-backed status source (issue #79) and ``windbreak
    rebuild`` fold the mode history identically.

    Args:
        records: The verified ledger records, in sequence order.

    Returns:
        One ``{seq, created_at, mode, beat}`` entry per ``ModeHeartbeat`` record.
    """
    return [
        _mode_projection(record)
        for record in records
        if record.event_type == _MODE_HEARTBEAT
    ]


def _gateway_projection(record: LedgerRecord) -> dict[str, object]:
    """Project a gateway/recovery record into its read-model entry.

    Args:
        record: A ledger record whose ``event_type`` is a gateway/recovery type.

    Returns:
        The ``{seq, created_at, event_type, data}`` read-model entry, carrying
        the event's full persisted payload verbatim under ``data``.
    """
    data = json.loads(record.payload_json)["data"]
    return {
        "seq": record.sequence_number,
        "created_at": record.created_at,
        "event_type": record.event_type,
        "data": data,
    }


def positions_read_model(records: list[LedgerRecord]) -> list[dict[str, object]]:
    """Project the latest ``PositionsSnapshotRecorded`` into ``positions.json``.

    Holds at most one entry -- the single most recent snapshot -- so a reader
    sees the account's current positions, not its whole history. An empty list
    when no such event has ever been ledgered.

    Args:
        records: The verified ledger records, in sequence order.

    Returns:
        A list with the latest snapshot's ``{seq, created_at, event_type, data}``
        entry, or an empty list.
    """
    snapshots = [
        _gateway_projection(record)
        for record in records
        if record.event_type == _POSITIONS_SNAPSHOT_RECORDED
    ]
    return snapshots[-1:]


def equity_curve_read_model(records: list[LedgerRecord]) -> list[dict[str, object]]:
    """Project every ``EquitySampled`` row, in ledger order, into ``equity_curve.json``.

    Args:
        records: The verified ledger records, in sequence order.

    Returns:
        One ``{seq, created_at, event_type, data}`` entry per equity sample.
    """
    return [
        _gateway_projection(record)
        for record in records
        if record.event_type == _EQUITY_SAMPLED
    ]


def selector_decisions_read_model(
    records: list[LedgerRecord],
) -> list[dict[str, object]]:
    """Project the interleaved selector/intent trail into ``selector_decisions.json``.

    Folds each ``SelectorDecisionRecorded`` plus the bare
    ``IntentApproved``/``IntentVetoed`` verdicts the Risk Kernel emits, in ledger
    order, so a reader can follow each forecast's decision through to its
    approval or veto.

    Args:
        records: The verified ledger records, in sequence order.

    Returns:
        One ``{seq, created_at, event_type, data}`` entry per selector/intent
        event, interleaved in ledger order.
    """
    return [
        _gateway_projection(record)
        for record in records
        if record.event_type in _SELECTOR_DECISION_EVENT_TYPES
    ]


#: Read-model filename holding the per-fill execution-quality comparison
#: projection (issue #58).
_EXECUTION_QUALITY_FILENAME = "execution_quality.json"

#: Read-model filename holding the per-run live-divergence sample projection
#: (issue #58).
_LIVE_DIVERGENCE_FILENAME = "live_divergence.json"

#: The live-divergence ledger event types projected into their read models
#: (issue #58): the per-fill execution-quality comparison and the per-run
#: divergence sample.
_EXECUTION_QUALITY_RECORDED = "ExecutionQualityRecorded"
_LIVE_DIVERGENCE_SAMPLED = "LiveDivergenceSampled"
_LIVE_DIVERGENCE_BREACHED = "LiveDivergenceBreached"


def execution_quality_read_model(
    records: list[LedgerRecord],
) -> list[dict[str, object]]:
    """Project every ``ExecutionQualityRecorded`` row, in ledger order (issue #58).

    Args:
        records: The verified ledger records, in sequence order.

    Returns:
        One ``{seq, created_at, event_type, data}`` entry per execution-quality
        comparison.
    """
    return [
        _gateway_projection(record)
        for record in records
        if record.event_type == _EXECUTION_QUALITY_RECORDED
    ]


#: Read-model filename holding the latest-per-provider canary-verdict
#: projection (issue #195).
_CANARY_STATUS_FILENAME = "canary_status.json"

#: Read-model filename holding every ``ForecastCreated`` row, in ledger order
#: (issue #195), for the weekly-report/dashboard fleet-cost/abstention fold.
_FORECASTS_FILENAME = "forecasts.json"

_CANARY_VERDICT_RECORDED = "CanaryVerdictRecorded"
_FORECAST_CREATED = "ForecastCreated"

#: Read-model filename holding the per-provider vote-cost aggregate projection
#: (issue #281).
_PROVIDER_VOTE_COSTS_FILENAME = "provider_vote_costs.json"

_PROVIDER_VOTE_RECORDED = "ProviderVoteRecorded"

#: The vote outcome that marks an abstention, counted into ``abstain_count``.
_OUTCOME_ABSTAINED = "abstained"

#: Parts-per-million scale: an abstention rate is reported as an integer ppm
#: (``abstain_count * _PPM_SCALE // vote_count``), never a float ratio, keeping
#: this money-path module float-free (SPEC S6.1).
_PPM_SCALE = 1_000_000


def canary_status_read_model(
    records: list[LedgerRecord],
) -> list[dict[str, object]]:
    """Project the LATEST ``CanaryVerdictRecorded`` per provider (issue #195).

    Folds the ledger keeping only each provider's most recently ledgered
    verdict, at that provider's first-seen list position -- exactly a Python
    dict's own "reassign the value in place, keep the original key position"
    semantics, the simplest literal "latest wins" contract. An empty list when
    no such event has ever been ledgered.

    Args:
        records: The verified ledger records, in sequence order.

    Returns:
        One ``{seq, created_at, event_type, data}`` entry per provider, holding
        that provider's latest verdict, in first-seen order.
    """
    latest: dict[object, dict[str, object]] = {}
    for record in records:
        if record.event_type != _CANARY_VERDICT_RECORDED:
            continue
        data = json.loads(record.payload_json)["data"]
        latest[data["provider"]] = {
            "seq": record.sequence_number,
            "created_at": record.created_at,
            "event_type": record.event_type,
            "data": data,
        }
    return list(latest.values())


def forecasts_read_model(records: list[LedgerRecord]) -> list[dict[str, object]]:
    """Project every ``ForecastCreated`` row, in ledger order (issue #195).

    Feeds the weekly-report/dashboard fleet cost-per-forecast and
    abstention-rate fold.

    Args:
        records: The verified ledger records, in sequence order.

    Returns:
        One ``{seq, created_at, event_type, data}`` entry per forecast.
    """
    return [
        _gateway_projection(record)
        for record in records
        if record.event_type == _FORECAST_CREATED
    ]


class _ProviderVoteAggregate:
    """Mutable accumulator folding one provider's ``ProviderVoteRecorded`` rows.

    Accumulates the running totals a single provider's read-model row is
    derived from, so the fold in :func:`provider_vote_costs_read_model` stays a
    flat, single-branch loop. ``forecast_count`` deliberately counts DISTINCT
    ``forecast_id`` values (the default ensemble votes one provider twice per
    forecast), so ``vote_count`` may exceed ``forecast_count``.

    Attributes:
        cost_micros_total: Summed ``cost_micros`` across every outcome (charged
            spend, including discarded votes).
        vote_count: The number of ``ProviderVoteRecorded`` rows for the provider.
        abstain_count: The rows whose ``outcome`` is ``"abstained"``.
    """

    def __init__(self) -> None:
        """Start every running total at zero, with no forecasts seen yet."""
        self.cost_micros_total = 0
        self.vote_count = 0
        self.abstain_count = 0
        self._forecast_ids: set[str] = set()

    def add(self, data: dict[str, object]) -> None:
        """Fold one vote's persisted payload into the running totals.

        Args:
            data: A ``ProviderVoteRecorded`` payload's ``data`` dict.
        """
        self.cost_micros_total += cast("int", data["cost_micros"])
        self.vote_count += 1
        if data["outcome"] == _OUTCOME_ABSTAINED:
            self.abstain_count += 1
        self._forecast_ids.add(str(data["forecast_id"]))

    def as_row(self, provider: str) -> dict[str, object]:
        """Derive the provider's read-model row from the accumulated totals.

        Args:
            provider: The provider identifier this aggregate folded.

        Returns:
            The ``{provider, cost_micros_total, vote_count, abstain_count,
            forecast_count, cost_per_forecast_micros, abstain_rate_ppm}`` row.
            Integer floor division only -- ``vote_count`` and ``forecast_count``
            are both ``>= 1`` for any row that exists, so neither denominator is
            ever zero.
        """
        forecast_count = len(self._forecast_ids)
        return {
            "provider": provider,
            "cost_micros_total": self.cost_micros_total,
            "vote_count": self.vote_count,
            "abstain_count": self.abstain_count,
            "forecast_count": forecast_count,
            "cost_per_forecast_micros": self.cost_micros_total // forecast_count,
            "abstain_rate_ppm": self.abstain_count * _PPM_SCALE // self.vote_count,
        }


def provider_vote_costs_read_model(
    records: list[LedgerRecord],
) -> list[dict[str, object]]:
    """Fold every ``ProviderVoteRecorded`` into one per-provider aggregate (#281).

    Unlike this package's passthrough projections, this fold produces one
    AGGREGATE row per provider (not one row per event), in first-seen provider
    order, summing charged spend and deriving integer cost-per-forecast and
    abstention-rate figures. An empty list when no such event has ever been
    ledgered, and no row at all for a provider with zero events (so no
    zero-denominator row is ever produced).

    Args:
        records: The verified ledger records, in sequence order.

    Returns:
        One aggregate row per provider that has ever been ledgered, in
        first-seen order.
    """
    aggregates: dict[str, _ProviderVoteAggregate] = {}
    for record in records:
        if record.event_type != _PROVIDER_VOTE_RECORDED:
            continue
        data = json.loads(record.payload_json)["data"]
        provider = str(data["provider"])
        if provider not in aggregates:
            aggregates[provider] = _ProviderVoteAggregate()
        aggregates[provider].add(data)
    return [aggregate.as_row(provider) for provider, aggregate in aggregates.items()]


def live_divergence_read_model(
    records: list[LedgerRecord],
) -> list[dict[str, object]]:
    """Project the live-divergence audit trail, in ledger order (issue #58).

    Folds both ``LiveDivergenceSampled`` (every monitor run) and
    ``LiveDivergenceBreached`` (one per firing SPEC S10.10 automatic-demotion
    trigger) rows, preserving ledger order, so an operator sees each sample
    alongside any breach it triggered. A breach row's ``data`` carries the
    sampled snapshot plus the firing ``trigger`` name.

    Args:
        records: The verified ledger records, in sequence order.

    Returns:
        One ``{seq, created_at, event_type, data}`` entry per sampled or
        breached divergence row.
    """
    return [
        _gateway_projection(record)
        for record in records
        if record.event_type in {_LIVE_DIVERGENCE_SAMPLED, _LIVE_DIVERGENCE_BREACHED}
    ]


def _write_read_model(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a read model as canonical JSON bytes with one trailing newline.

    Args:
        path: Destination file path.
        rows: The read-model entries to serialize.
    """
    body = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    path.write_bytes(body.encode("utf-8") + b"\n")


def rebuild(ledger_path: Path, output_dir: Path) -> None:
    """Verify the ledger and fold it into the eleven read-model files.

    Writes ``config_versions.json``, ``mode_history.json``,
    ``gateway_events.json``, the three PAPER-loop projections
    (``positions.json``, ``equity_curve.json``, ``selector_decisions.json``,
    issue #48), the two live-divergence projections
    (``execution_quality.json``, ``live_divergence.json``, issue #58), the two
    fleet-observability projections (``canary_status.json``,
    ``forecasts.json``, issue #195), and the per-provider vote-cost aggregate
    (``provider_vote_costs.json``, issue #281); each is written
    unconditionally, empty where its source events are absent.

    Args:
        ledger_path: Path to the SQLite ledger database.
        output_dir: Directory to write the read models into; created if
            absent.

    Raises:
        FileNotFoundError: If ``ledger_path`` does not point at an existing
            file; rebuild reads an existing ledger and never creates one.
        ChainIntegrityError: If the ledger's hash chain fails verification.
    """
    if not ledger_path.is_file():
        raise FileNotFoundError(
            f"ledger not found at {ledger_path}: rebuild reads an existing "
            "ledger and will not create one. Check --ledger-path, or run the "
            "pipeline first to produce a ledger."
        )
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()

    config_versions = [
        _config_projection(record)
        for record in records
        if record.event_type == _CONFIG_LOADED
    ]
    mode_history = mode_history_read_model(records)
    gateway_events = [
        _gateway_projection(record)
        for record in records
        if record.event_type in _GATEWAY_EVENT_TYPES
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    # ``Path.joinpath`` (not the ``/`` operator) keeps this module clear of the
    # no-float lint's blanket true-division ban on money-path packages (SPEC
    # S6.1); path joining is byte-identical either way.
    _write_read_model(output_dir.joinpath(_CONFIG_VERSIONS_FILENAME), config_versions)
    _write_read_model(output_dir.joinpath(_MODE_HISTORY_FILENAME), mode_history)
    _write_read_model(output_dir.joinpath(_GATEWAY_EVENTS_FILENAME), gateway_events)
    _write_read_model(
        output_dir.joinpath(_POSITIONS_FILENAME), positions_read_model(records)
    )
    _write_read_model(
        output_dir.joinpath(_EQUITY_CURVE_FILENAME), equity_curve_read_model(records)
    )
    _write_read_model(
        output_dir.joinpath(_SELECTOR_DECISIONS_FILENAME),
        selector_decisions_read_model(records),
    )
    _write_read_model(
        output_dir.joinpath(_EXECUTION_QUALITY_FILENAME),
        execution_quality_read_model(records),
    )
    _write_read_model(
        output_dir.joinpath(_LIVE_DIVERGENCE_FILENAME),
        live_divergence_read_model(records),
    )
    _write_read_model(
        output_dir.joinpath(_CANARY_STATUS_FILENAME),
        canary_status_read_model(records),
    )
    _write_read_model(
        output_dir.joinpath(_FORECASTS_FILENAME), forecasts_read_model(records)
    )
    _write_read_model(
        output_dir.joinpath(_PROVIDER_VOTE_COSTS_FILENAME),
        provider_vote_costs_read_model(records),
    )


def rebuild_command(args: Namespace) -> int:
    """Run ``rebuild`` for the CLI, mapping failures to an exit code.

    Args:
        args: Parsed CLI arguments exposing ``ledger_path`` and
            ``output_dir`` paths.

    Returns:
        0 on a clean rebuild; 1 if the chain fails verification or the ledger
        path does not exist (with the offending ``sequence_number`` or the
        missing-path guidance printed to stderr).
    """
    try:
        rebuild(args.ledger_path, args.output_dir)
    except (ChainIntegrityError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
