"""Renderer for the PAPER-loop equity-vs-floor view (issue #48).

Renders every ``EquitySampled`` read-model row into one row of a column-labelled
HTML table (``<table>``/``<thead>``/``<tbody>``, issue #275) pairing the sampled
equity with the configured floor. Every ledger-derived value is HTML-escaped
(:func:`windbreak.dashboard.views._html.escape`) before output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from windbreak.dashboard.views._html import table

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The section heading the equity view renders under.
_TITLE = "Equity vs floor"

#: The payload keys rendered, in column order. One tuple serves as both the
#: header labels and the cell lookup keys so a header can never drift out of
#: step with the value beneath it.
_COLUMNS = ("epoch_s", "equity_micros", "floor_micros")


def _equity_values(row: dict[str, object]) -> tuple[object, ...]:
    """Extract one equity sample's cell values in column order.

    Args:
        row: One ``equity_curve.json`` read-model row.

    Returns:
        The sample's :data:`_COLUMNS` values, unescaped (the table helper
        escapes every cell).
    """
    data = cast("Mapping[str, object]", row["data"])
    return tuple(data.get(column) for column in _COLUMNS)


def render_equity_vs_floor(rows: list[dict[str, object]]) -> str:
    """Render the equity curve into an HTML section.

    Args:
        rows: The ``equity_curve.json`` read-model rows, in ledger order; an
            empty list renders the "no data yet" placeholder.

    Returns:
        An HTML section containing a table of each sample's equity and floor
        under :data:`_COLUMNS` headers, all values HTML-escaped.
    """
    return table(_TITLE, _COLUMNS, [_equity_values(row) for row in rows])
