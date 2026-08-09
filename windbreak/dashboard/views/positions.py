"""Renderer for the PAPER-loop positions view (issue #48).

Renders the latest ``PositionsSnapshotRecorded`` read-model row into a
column-labelled HTML table (``<table>``/``<thead>``/``<tbody>``, issue #275) of
each held position's ticker, quantity, and average price. Every ledger-derived
value is HTML-escaped (:func:`windbreak.dashboard.views._html.escape`) before
output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from windbreak.dashboard.views._html import table

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The section heading the positions view renders under.
_TITLE = "Positions"

#: The position keys rendered, in column order. One tuple serves as both the
#: header labels and the cell lookup keys so a header can never drift out of
#: step with the value beneath it.
_COLUMNS = ("ticker", "quantity_centis", "average_price_pips")


def _position_values(position: Mapping[str, object]) -> tuple[object, ...]:
    """Extract one position's cell values in column order.

    Args:
        position: One position mapping from a snapshot's ``data.positions``.

    Returns:
        The position's :data:`_COLUMNS` values, unescaped (the table helper
        escapes every cell).
    """
    return tuple(position.get(column) for column in _COLUMNS)


def render_positions(rows: list[dict[str, object]]) -> str:
    """Render the latest positions snapshot into an HTML section.

    Args:
        rows: The ``positions.json`` read-model rows (at most one, the latest
            snapshot); an empty list renders the "no data yet" placeholder.

    Returns:
        An HTML section containing a table of each held position under
        :data:`_COLUMNS` headers, all values HTML-escaped.
    """
    if not rows:
        return table(_TITLE, _COLUMNS, [])
    data = cast("Mapping[str, object]", rows[-1]["data"])
    positions = cast("list[Mapping[str, object]]", data.get("positions", []))
    return table(_TITLE, _COLUMNS, [_position_values(row) for row in positions])
