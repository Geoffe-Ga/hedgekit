"""Renderer for the live execution-quality view (issue #58).

Renders every ``ExecutionQualityRecorded`` read-model row into one row of a
column-labelled HTML table (``<table>``/``<thead>``/``<tbody>``, issue #275)
giving a fill's identity and its live-vs-paper cost slippage. Every
ledger-derived value is HTML-escaped
(:func:`windbreak.dashboard.views._html.escape`) before output -- a fill id is
forecast/venue-adjacent and therefore an XSS surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from windbreak.dashboard.views._html import table

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The section heading the execution-quality view renders under.
_TITLE = "Execution quality (live vs paper)"

#: The payload keys rendered, in column order. One tuple serves as both the
#: header labels and the cell lookup keys so a header can never drift out of
#: step with the value beneath it.
_COLUMNS = ("fill_id", "market_ticker", "slippage_micros")


def _execution_values(row: dict[str, object]) -> tuple[object, ...]:
    """Extract one execution-quality record's cell values in column order.

    Args:
        row: One ``execution_quality`` read-model row
            (``{seq, created_at, event_type, data}``).

    Returns:
        The record's :data:`_COLUMNS` values, unescaped (the table helper
        escapes every cell).
    """
    data = cast("Mapping[str, object]", row["data"])
    return tuple(data.get(column) for column in _COLUMNS)


def render_execution_quality(rows: list[dict[str, object]]) -> str:
    """Render the execution-quality read model into an HTML section.

    Args:
        rows: The ``execution_quality`` read-model rows, in ledger order; an
            empty list renders the shared "no data yet" placeholder.

    Returns:
        An HTML section containing a table of each fill's slippage under
        :data:`_COLUMNS` headers, all values HTML-escaped.
    """
    return table(_TITLE, _COLUMNS, [_execution_values(row) for row in rows])
