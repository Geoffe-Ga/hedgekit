"""Renderer for the dashboard's ``/providers`` fleet-observability panel (#195).

Renders the provider-panel read-model rows into an HTML section, mirroring
:mod:`windbreak.dashboard.views.decisions`'s pure-renderer idiom. Two row kinds
share one list (mirroring ``gateway_events.json``'s own multi-type-in-one
projection, discriminated by a ``kind`` key here): a ``"provider"`` row (one
per provider's summary line -- provider id, resolved count, Brier skill, canary
status, and the per-provider abstention rate / ``cost_per_forecast`` the
``provider_vote_costs`` fold now supplies, issue #281, falling back to ``n/a``
only for a provider that fold does not cover) and a ``"fleet"`` row (the two
fleet-wide cost figures that ARE derivable in aggregate).

Every ledger-derived value flows through
:func:`windbreak.dashboard.views._html.escape` before output -- provider
identifiers are operator-supplied config today but must stay defensively
escaped like every other dashboard-rendered string. A negative Brier skill is
rendered verbatim, in markup structurally identical to a positive one: the panel
never dims, omits, or conditionally styles a value based on its sign (the
honesty invariant).
"""

from __future__ import annotations

from windbreak.dashboard.views._html import escape, section

#: The section heading the provider panel renders under.
_TITLE = "Providers"

#: The row-``kind`` discriminator marking a fleet-wide cost summary row.
_FLEET_KIND = "fleet"

#: The not-yet-attributed fallback for a provider the ``provider_vote_costs``
#: fold (issue #281) does not cover: an old ledger with no such events, or one
#: with zero vote-cost rows for that provider.
_NOT_AVAILABLE = "n/a"


def _format_signed(value: object) -> str:
    """Render an integer with an explicit sign, else stringify verbatim.

    A real ``int`` renders with an explicit ``+``/``-`` so the sign is never
    ambiguous between "not measured" and "at or above baseline"; a non-int
    (already-formatted or sentinel) value is stringified unchanged.

    Args:
        value: The value to render (typically a ppm Brier skill).

    Returns:
        The signed decimal string for an ``int``, else ``str(value)``.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:+d}"
    return str(value)


def _format_micros(value: object) -> str:
    """Render a micros cost, or ``n/a`` when it is unset (``None``).

    Args:
        value: The cost in micros, or ``None`` when not yet derivable.

    Returns:
        The decimal string, or ``n/a`` for ``None`` (never the literal
        ``None``).
    """
    return _NOT_AVAILABLE if value is None else str(value)


def _render_provider_row(row: dict[str, object]) -> str:
    """Render one per-provider summary row into an escaped HTML fragment.

    Args:
        row: One ``"provider"``-kind panel row.

    Returns:
        An HTML fragment with the provider's summary fields, all ledger-derived
        values HTML-escaped and the Brier skill rendered verbatim with its sign.
    """
    provider = escape(row.get("provider", ""))
    resolved = escape(row.get("resolved", ""))
    brier = escape(_format_signed(row.get("brier_skill_ppm")))
    canary = escape(row.get("canary_status", ""))
    abstain = escape(row.get("abstain_rate_ppm", ""))
    cost = escape(row.get("cost_per_forecast", _NOT_AVAILABLE))
    return (
        "<article>"
        f"<h3>{provider}</h3>"
        "<dl>"
        f"<dt>resolved</dt><dd>{resolved}</dd>"
        f"<dt>brier_skill_ppm</dt><dd>{brier}</dd>"
        f"<dt>canary</dt><dd>{canary}</dd>"
        f"<dt>abstain_rate_ppm</dt><dd>{abstain}</dd>"
        f"<dt>cost_per_forecast</dt><dd>{cost}</dd>"
        "</dl>"
        "</article>"
    )


def _render_fleet_row(row: dict[str, object]) -> str:
    """Render the fleet-wide cost summary row into an escaped HTML fragment.

    Args:
        row: One ``"fleet"``-kind panel row.

    Returns:
        An HTML fragment with the fleet cost-per-forecast and cost-per-resolved
        figures (``n/a`` when unset), all values HTML-escaped.
    """
    forecast_cost = escape(_format_micros(row.get("cost_per_forecast_micros")))
    resolved_cost = escape(_format_micros(row.get("cost_per_resolved_micros")))
    return (
        "<article>"
        "<h3>fleet</h3>"
        "<dl>"
        f"<dt>cost_per_forecast_micros</dt><dd>{forecast_cost}</dd>"
        f"<dt>cost_per_resolved_micros</dt><dd>{resolved_cost}</dd>"
        "</dl>"
        "</article>"
    )


def _render_panel_row(row: dict[str, object]) -> str:
    """Render one panel row, dispatching on its ``kind`` discriminator.

    Args:
        row: One provider-panel read-model row.

    Returns:
        The rendered fleet fragment for a ``"fleet"`` row, else the provider
        fragment.
    """
    if row.get("kind") == _FLEET_KIND:
        return _render_fleet_row(row)
    return _render_provider_row(row)


def render_provider_panel(rows: list[dict[str, object]]) -> str:
    """Render the fleet-observability provider panel into an HTML section.

    Args:
        rows: The provider-panel read-model rows; an empty list renders the
            shared "No data yet." placeholder rather than an error or an empty
            table.

    Returns:
        An HTML section fragment listing each provider's summary and the fleet
        cost lines, all ledger-derived values HTML-escaped.
    """
    body = [_render_panel_row(row) for row in rows]
    return section(_TITLE, body)
