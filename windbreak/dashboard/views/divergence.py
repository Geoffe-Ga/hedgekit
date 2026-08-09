"""Renderer for the live-vs-paper divergence view (issue #58).

Renders every ``LiveDivergenceSampled`` and ``LiveDivergenceBreached`` read-model
row into one row of a column-labelled HTML table
(``<table>``/``<thead>``/``<tbody>``, issue #275) placing the two divergence
series beside their thresholds and the firing trigger. Each series value, each
threshold, and the trigger name is drawn from the row payload and HTML-escaped
(:func:`windbreak.dashboard.views._html.escape`) before output; a sentinel value
(e.g. ``"UNDEFINED"``) renders verbatim. Sampled rows carry no ``trigger`` and
render the :data:`_MISSING` placeholder in that cell rather than dropping it, so
every row keeps the same cells under the same headers; breach rows render the
escaped firing trigger name so an operator can see which threshold fired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from windbreak.dashboard.views._html import table

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The section heading the divergence view renders under.
_TITLE = "Live vs paper divergence"

#: Rendered for a payload field a (minimal) sampled row does not carry, so a
#: missing threshold never renders a bare ``None`` and never silently shortens
#: the row out of alignment with its headers.
_MISSING = "n/a"

#: The payload keys rendered, in column order. One tuple serves as both the
#: header labels and the cell lookup keys so a header can never drift out of
#: step with the value beneath it.
_COLUMNS = (
    "live_slippage_ratio_ppm",
    "live_slippage_ratio_limit_ppm",
    "live_brier_degradation_ppm",
    "live_brier_degradation_band_ppm",
    "trigger",
)


def _divergence_values(row: dict[str, object]) -> tuple[object, ...]:
    """Extract one divergence row's cell values in column order.

    Args:
        row: One ``live_divergence`` read-model row
            (``{seq, created_at, event_type, data}``).

    Returns:
        The row's :data:`_COLUMNS` values, unescaped (the table helper escapes
        every cell), with :data:`_MISSING` standing in for any key the payload
        does not carry -- a sampled row has no ``trigger``.
    """
    data = cast("Mapping[str, object]", row["data"])
    return tuple(data.get(column, _MISSING) for column in _COLUMNS)


def render_live_divergence(rows: list[dict[str, object]]) -> str:
    """Render the live-divergence read model into an HTML section.

    Args:
        rows: The ``live_divergence`` read-model rows, in ledger order; an empty
            list renders the shared "no data yet" placeholder.

    Returns:
        An HTML section containing a table of each sampled or breached row's two
        series against their thresholds plus the firing trigger (the
        :data:`_MISSING` placeholder for sampled rows), under :data:`_COLUMNS`
        headers, all values HTML-escaped.
    """
    return table(_TITLE, _COLUMNS, [_divergence_values(row) for row in rows])
