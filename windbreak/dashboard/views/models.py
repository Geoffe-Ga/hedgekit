"""The dashboard read-model bundle and its ledger-backed source factory (issue #48).

:class:`DashboardReadModels` is the immutable "current view" the three PAPER-loop
routes render: the latest positions, the equity curve, and the interleaved
selector/intent decisions, each a list of the read-model rows
:mod:`windbreak.ledger.rebuild` produces. An empty bundle is the documented
"no data yet" input. :func:`build_ledger_read_models_source` folds a verified
ledger database into a zero-arg source callable, reusing the very projection
functions ``windbreak rebuild`` writes so the dashboard never re-derives its own
view of the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from windbreak.forecast.providers.track_record import ProviderTrackRecord

#: One read-model row: the ``{seq, created_at, event_type, data}`` shape every
#: :mod:`windbreak.ledger.rebuild` projection emits.
ReadModelRow = dict[str, object]

#: The old-ledger fallback for a figure the fold cannot supply: a provider
#: absent from the ``provider_vote_costs`` fold (a pre-#281 ledger, or one with
#: zero vote-cost rows for that provider) keeps this ``n/a`` placeholder for its
#: ``abstain_rate_ppm``/``cost_per_forecast`` rather than a fabricated ``0``.
_NOT_AVAILABLE = "n/a"


@dataclass(frozen=True)
class DashboardReadModels:
    """The immutable read-model bundle the three PAPER-loop views render.

    Attributes:
        positions: The latest positions-snapshot rows (at most one).
        equity_curve: Every equity-sample row, in ledger order.
        decisions: The interleaved selector/intent decision rows, in ledger
            order.
        execution_quality: Every ``ExecutionQualityRecorded`` row, in ledger
            order (issue #58); defaults to empty so a pre-#58 construction stays
            valid.
        live_divergence: Every ``LiveDivergenceSampled`` and
            ``LiveDivergenceBreached`` row, in ledger order (issue #58); breach
            rows carry the firing trigger. Defaults to empty.
        provider_panel: The fleet-observability provider-panel rows (issue #195):
            one ``"provider"``-kind row per provider plus an optional
            ``"fleet"``-kind cost-summary row. Defaults to empty so a pre-#195
            construction stays valid.
    """

    positions: list[ReadModelRow]
    equity_curve: list[ReadModelRow]
    decisions: list[ReadModelRow]
    execution_quality: list[ReadModelRow] = field(default_factory=list)
    live_divergence: list[ReadModelRow] = field(default_factory=list)
    provider_panel: list[ReadModelRow] = field(default_factory=list)


def _row_data(row: ReadModelRow) -> dict[str, object]:
    """Return one projection row's ``data`` payload, narrowed for indexing.

    Args:
        row: A ``{seq, created_at, event_type, data}`` projection row.

    Returns:
        The row's ``data`` payload as a ``dict[str, object]``.
    """
    return cast("dict[str, object]", row["data"])


def _resolved_and_skill(
    provider: object,
    track_records: dict[str, ProviderTrackRecord] | None,
) -> tuple[int | str, int | str]:
    """Resolve one provider's ``(resolved, brier_skill_ppm)`` panel pair.

    When a track-record fold is supplied and it covers this provider, both
    figures are the real integers from #194's read model -- a negative Brier
    skill included verbatim (the honesty invariant), never suppressed. A
    provider absent from the fold (or no fold at all) keeps the ``n/a``
    placeholder rather than a fabricated figure.

    Args:
        provider: The provider identity from the canary-status row (an
            ``object`` off the projection payload; only ``str`` keys can match).
        track_records: The parsed provider -> track-record map, or ``None`` when
            no track-record artifact was wired.

    Returns:
        The ``(resolved, brier_skill_ppm)`` pair: real integers when covered,
        else the ``n/a`` placeholder for each.
    """
    if track_records is not None and isinstance(provider, str):
        record = track_records.get(provider)
        if record is not None:
            return record.resolved_count, record.brier_skill_ppm
    return _NOT_AVAILABLE, _NOT_AVAILABLE


def _vote_cost_by_provider(
    vote_cost_rows: list[ReadModelRow] | None,
) -> dict[str, ReadModelRow]:
    """Index the ``provider_vote_costs`` fold rows by provider identity.

    Args:
        vote_cost_rows: The ``provider_vote_costs.json`` aggregate rows, or
            ``None`` when no vote-cost fold was threaded through.

    Returns:
        The rows keyed by their ``provider`` field, or an empty map when no
        fold was supplied.
    """
    if vote_cost_rows is None:
        return {}
    return {cast("str", row["provider"]): row for row in vote_cost_rows}


def _abstain_rate_and_cost(
    provider: object,
    vote_cost_by_provider: dict[str, ReadModelRow],
) -> tuple[int | str, int | str]:
    """Resolve one provider's ``(abstain_rate_ppm, cost_per_forecast)`` pair.

    Mirrors :func:`_resolved_and_skill`'s covered/uncovered contract: a provider
    the vote-cost fold covers gets the real integer figures verbatim; one it
    does not cover (or no fold at all) keeps the ``n/a`` placeholder rather than
    a fabricated ``0``.

    Args:
        provider: The provider identity from the canary-status row (an
            ``object`` off the projection payload; only ``str`` keys can match).
        vote_cost_by_provider: The vote-cost fold rows keyed by provider.

    Returns:
        The ``(abstain_rate_ppm, cost_per_forecast)`` pair: real integers when
        covered, else the ``n/a`` placeholder for each.
    """
    if isinstance(provider, str):
        row = vote_cost_by_provider.get(provider)
        if row is not None:
            return (
                cast("int", row["abstain_rate_ppm"]),
                cast("int", row["cost_per_forecast_micros"]),
            )
    return _NOT_AVAILABLE, _NOT_AVAILABLE


def _provider_summary_rows(
    canary_rows: list[ReadModelRow],
    track_records: dict[str, ProviderTrackRecord] | None = None,
    *,
    vote_cost_rows: list[ReadModelRow] | None = None,
) -> list[ReadModelRow]:
    """Compose one ``"provider"`` panel row per latest-canary-status row.

    The provider identity and its live canary status come from the ledger fold.
    The resolved count and Brier skill come from the optional ``track_records``
    fold (#194): real integers for a covered provider, ``n/a`` for one the fold
    does not cover (or when no fold is wired). The abstention rate and
    per-provider cost come from the optional ``vote_cost_rows`` fold (#281) the
    same way: real integers for a covered provider, ``n/a`` otherwise.

    Args:
        canary_rows: The ``canary_status.json`` projection rows.
        track_records: The parsed provider -> track-record map, or ``None`` to
            leave every provider's ``resolved``/``brier_skill_ppm`` as ``n/a``.
        vote_cost_rows: The ``provider_vote_costs.json`` aggregate rows (#281),
            or ``None`` to leave every provider's ``abstain_rate_ppm``/
            ``cost_per_forecast`` as ``n/a``.

    Returns:
        One provider-panel row per provider, in first-seen order.
    """
    vote_cost_by_provider = _vote_cost_by_provider(vote_cost_rows)
    rows: list[ReadModelRow] = []
    for canary_row in canary_rows:
        data = _row_data(canary_row)
        resolved, brier_skill_ppm = _resolved_and_skill(data["provider"], track_records)
        abstain_rate_ppm, cost_per_forecast = _abstain_rate_and_cost(
            data["provider"], vote_cost_by_provider
        )
        rows.append(
            {
                "kind": "provider",
                "provider": data["provider"],
                "resolved": resolved,
                "brier_skill_ppm": brier_skill_ppm,
                "canary_status": data["status"],
                "abstain_rate_ppm": abstain_rate_ppm,
                "cost_per_forecast": cost_per_forecast,
            }
        )
    return rows


def _fleet_cost_row(forecast_rows: list[ReadModelRow]) -> ReadModelRow | None:
    """Compose the fleet-wide cost-summary row from the forecasts projection.

    The fleet cost-per-forecast IS derivable in aggregate (total research spend
    over the forecast count, integer-divided); cost-per-resolved needs
    resolution data this fold does not carry, so it stays ``None`` (rendered
    ``n/a``).

    Args:
        forecast_rows: The ``forecasts.json`` projection rows.

    Returns:
        A ``"fleet"``-kind row, or ``None`` when no forecast has been ledgered.
    """
    if not forecast_rows:
        return None
    total_micros = sum(
        cast("int", _row_data(row)["research_cost_micros"]) for row in forecast_rows
    )
    return {
        "kind": "fleet",
        "cost_per_forecast_micros": total_micros // len(forecast_rows),
        "cost_per_resolved_micros": None,
    }


def _compose_provider_panel(
    canary_rows: list[ReadModelRow],
    forecast_rows: list[ReadModelRow],
    track_records: dict[str, ProviderTrackRecord] | None = None,
    *,
    vote_cost_rows: list[ReadModelRow] | None = None,
) -> list[ReadModelRow]:
    """Compose the full provider panel from the fleet projections.

    Args:
        canary_rows: The ``canary_status.json`` projection rows.
        forecast_rows: The ``forecasts.json`` projection rows.
        track_records: The parsed provider -> track-record map (#194), or
            ``None`` to leave every provider's ``resolved``/``brier_skill_ppm``
            as ``n/a``.
        vote_cost_rows: The ``provider_vote_costs.json`` aggregate rows (#281),
            or ``None`` to leave every provider's ``abstain_rate_ppm``/
            ``cost_per_forecast`` as ``n/a``.

    Returns:
        The provider summary rows followed by the fleet cost row (when present).
    """
    panel = _provider_summary_rows(
        canary_rows, track_records, vote_cost_rows=vote_cost_rows
    )
    fleet_row = _fleet_cost_row(forecast_rows)
    if fleet_row is not None:
        panel.append(fleet_row)
    return panel


def build_ledger_read_models_source(
    ledger_path: Path,
    *,
    track_record_path: Path | None = None,
) -> Callable[[], DashboardReadModels]:
    """Build a zero-arg source folding a verified ledger into read models.

    The returned callable opens the ledger fresh on every invocation, verifies
    its hash chain, and folds it into a :class:`DashboardReadModels` via the same
    projection functions ``windbreak rebuild`` writes -- so every request reflects
    live ledger truth and a corrupt ledger fails closed
    (:class:`~windbreak.ledger.store.ChainIntegrityError`) rather than rendering a
    plausible-but-wrong view.

    When ``track_record_path`` is supplied, each invocation also re-reads that M6
    track-record artifact and folds #194's real per-provider ``resolved``/
    ``brier_skill_ppm`` into the provider panel (a negative Brier skill included
    verbatim). A malformed artifact propagates ``parse_track_records``'s
    fail-closed :class:`ValueError` rather than rendering a plausible-but-wrong
    skill. When it is ``None`` (the default), those figures stay ``n/a`` --
    the pre-#194 behavior, unchanged.

    Args:
        ledger_path: Path to the SQLite ledger database to project.
        track_record_path: Optional path to an M6 track-record artifact JSON
            file (keyword-only). When ``None``, per-provider
            ``resolved``/``brier_skill_ppm`` stay ``n/a``.

    Returns:
        A callable suitable for
        :func:`windbreak.dashboard.app.create_server`'s ``read_models_source``.
    """
    from windbreak.forecast.providers.track_record import parse_track_records
    from windbreak.ledger.rebuild import (
        canary_status_read_model,
        equity_curve_read_model,
        execution_quality_read_model,
        forecasts_read_model,
        live_divergence_read_model,
        positions_read_model,
        provider_vote_costs_read_model,
        selector_decisions_read_model,
    )
    from windbreak.ledger.store import SqliteLedgerStore

    def _source() -> DashboardReadModels:
        """Fold the ledger (and any track-record artifact) into a fresh bundle."""
        track_records = (
            parse_track_records(track_record_path.read_text())
            if track_record_path is not None
            else None
        )
        store = SqliteLedgerStore(ledger_path)
        try:
            store.verify_chain()
            records = store.read_all()
        finally:
            store.close()
        return DashboardReadModels(
            positions=positions_read_model(records),
            equity_curve=equity_curve_read_model(records),
            decisions=selector_decisions_read_model(records),
            execution_quality=execution_quality_read_model(records),
            live_divergence=live_divergence_read_model(records),
            provider_panel=_compose_provider_panel(
                canary_status_read_model(records),
                forecasts_read_model(records),
                track_records,
                vote_cost_rows=provider_vote_costs_read_model(records),
            ),
        )

    return _source
