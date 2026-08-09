"""Weekly-report provider section renderer (fleet observability, issue #195).

A pure renderer (mirroring
:func:`windbreak.evaluation.report._render_cost_meter`'s no-I/O shape) producing
one line per provider plus two fleet cost lines, embedded verbatim under the
weekly report's ``## Providers`` section. The worked-example line is, byte for
byte::

    provider=futuresearch resolved=212 brier_skill_ppm=+14200
    cost_per_forecast=n/a abstain_rate=9% canary=OK

Per-provider ``cost_per_forecast`` and ``abstain_rate`` are real-or-``None``
(issue #281 lands per-provider cost attribution): ``None`` -- never the string
``"n/a"`` -- is the not-yet-attributed sentinel, rendered ``n/a`` through the
``_micros_or_na`` pattern, so a not-yet-covered provider's row is representable
without ever routing a string into integer division. The two FLEET cost figures
ARE derivable in aggregate and render as their own lines. Every figure is an
integer (ppm/micros/whole percent); the module is float-free and true-division
free (SPEC S6.1), on ``scripts/lint_no_floats.py``'s denylist -- the
abstention-rate percent is an exact integer floor-division of the ppm rate.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Parts-per-million per whole percent, for the exact-integer abstention-rate
#: conversion (1% == 10_000 ppm).
_PPM_PER_PERCENT = 10_000

#: The not-yet-attributed sentinel a ``None`` figure renders as (issue #281).
_NOT_AVAILABLE = "n/a"


@dataclass(frozen=True, slots=True)
class ProviderReportRow:
    """One provider's weekly-report summary row.

    Attributes:
        provider: The provider identifier.
        resolved: How many of the provider's forecasts have resolved.
        brier_skill_ppm: The provider's Brier skill over baseline, in ppm
            (signed: negative means below baseline).
        abstain_rate_ppm: The provider's abstention rate, in ppm, or ``None``
            when not yet attributed (issue #281): ``None``, never the string
            ``"n/a"``, is the sentinel, so it is never routed into the
            percent floor-division.
        cost_per_forecast_micros: The provider's per-forecast cost, in micros,
            or ``None`` when not yet attributed (issue #281).
        canary_status: The provider's latest canary status
            (``"OK"``/``"ANSWER_DRIFT"``/``"VERSION_DRIFT"``).
    """

    provider: str
    resolved: int
    brier_skill_ppm: int
    abstain_rate_ppm: int | None
    cost_per_forecast_micros: int | None
    canary_status: str


@dataclass(frozen=True, slots=True)
class FleetCostSummary:
    """The fleet-wide cost figures, derivable in aggregate.

    Attributes:
        cost_per_forecast_micros: The fleet cost per forecast, in micros, or
            ``None`` when not yet derivable.
        cost_per_resolved_micros: The fleet cost per resolved forecast, in
            micros, or ``None`` when nothing has resolved yet.
    """

    cost_per_forecast_micros: int | None
    cost_per_resolved_micros: int | None


def _micros_or_na(value: int | None) -> str:
    """Render a micros figure, or ``n/a`` when it is unset (``None``).

    Args:
        value: The figure in micros, or ``None``.

    Returns:
        The decimal string, or ``n/a`` for ``None``.
    """
    return _NOT_AVAILABLE if value is None else str(value)


def _abstain_rate_or_na(abstain_rate_ppm: int | None) -> str:
    """Render an abstention rate as a whole percent, or ``n/a`` when unset.

    The ``None`` sentinel is checked BEFORE the floor-division, so the string
    ``"n/a"`` is never routed into integer division (the ``"n/a" // 10_000``
    crash issue #281's sentinel makes structurally impossible).

    Args:
        abstain_rate_ppm: The abstention rate in ppm, or ``None``.

    Returns:
        The ``<percent>%`` string, or ``n/a`` for ``None``.
    """
    if abstain_rate_ppm is None:
        return _NOT_AVAILABLE
    return f"{abstain_rate_ppm // _PPM_PER_PERCENT}%"


def _render_row(row: ProviderReportRow) -> str:
    """Render one provider row into its pinned weekly-report line.

    Args:
        row: The provider report row to render.

    Returns:
        The ``provider=.. resolved=.. brier_skill_ppm=+/-..
        cost_per_forecast=<micros|n/a> abstain_rate=<percent%|n/a> canary=..``
        line.
    """
    return (
        f"provider={row.provider} resolved={row.resolved} "
        f"brier_skill_ppm={row.brier_skill_ppm:+d} "
        f"cost_per_forecast={_micros_or_na(row.cost_per_forecast_micros)} "
        f"abstain_rate={_abstain_rate_or_na(row.abstain_rate_ppm)} "
        f"canary={row.canary_status}"
    )


def render_provider_lines(
    rows: tuple[ProviderReportRow, ...], *, fleet: FleetCostSummary
) -> str:
    """Render the provider section: one line per provider, then two fleet lines.

    Args:
        rows: The provider report rows, one per provider (possibly empty).
        fleet: The fleet-wide cost summary (keyword-only).

    Returns:
        The rendered section body: each provider's line followed by the
        ``fleet_cost_per_forecast_micros`` and ``fleet_cost_per_resolved_micros``
        lines (``n/a`` when a figure is unset). Zero provider rows still renders
        the two fleet lines; the caller owns any "No data yet." empty-state.
    """
    forecast_cost = _micros_or_na(fleet.cost_per_forecast_micros)
    resolved_cost = _micros_or_na(fleet.cost_per_resolved_micros)
    lines = [_render_row(row) for row in rows]
    lines.append(f"fleet_cost_per_forecast_micros={forecast_cost}")
    lines.append(f"fleet_cost_per_resolved_micros={resolved_cost}")
    return "\n".join(lines)
