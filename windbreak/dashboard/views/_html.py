"""Shared HTML-rendering helpers for the PAPER-loop dashboard views (issue #48).

Every ledger-derived string flows through :func:`escape` before it reaches
output -- selector/veto reasons and market tickers are forecast/LLM-adjacent and
therefore an XSS surface -- so the view renderers never interpolate a raw ledger
value into HTML. :func:`section` wraps a title and body rows into a labelled
section, rendering the shared :data:`NO_DATA_PLACEHOLDER` when there is nothing
to show (mirroring :mod:`windbreak.dashboard.app`'s ``never``-placeholder
precedent for a missing heartbeat).

:func:`table` is the single owner of tabular markup (issue #275). The row views
previously emitted bare ``<tr>``/``<td>`` straight into a ``<section>``; per the
HTML tree-construction spec a browser foster-parents those out of their parent,
so an operator saw unlabelled, run-together numbers instead of rows. Routing all
four row views through one helper means the ``<table>``/``<thead>``/``<tbody>``
scaffolding and the per-cell escaping exist in exactly one place and cannot
drift apart per view.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The readable placeholder every view renders when its read model is empty.
NO_DATA_PLACEHOLDER = "No data yet."


def escape(value: object) -> str:
    """Return ``value`` stringified and HTML-escaped for safe interpolation.

    Args:
        value: The (possibly ledger-derived, possibly non-string) value to
            render. Coerced to ``str`` first so an integer count or a hostile
            string are both escaped uniformly.

    Returns:
        The HTML-escaped string form of ``value``.
    """
    return html.escape(str(value))


def section(title: str, body_rows: list[str]) -> str:
    """Wrap a section title and its already-escaped body rows into HTML.

    Args:
        title: The section heading (a fixed, trusted literal from the caller).
        body_rows: The rendered, already-escaped body lines; an empty list
            renders the shared "no data yet" placeholder instead.

    Returns:
        A ``<section>`` HTML fragment.
    """
    inner = f"<p>{NO_DATA_PLACEHOLDER}</p>" if not body_rows else "\n".join(body_rows)
    return f"<section>\n<h2>{escape(title)}</h2>\n{inner}\n</section>\n"


def _data_row(values: Sequence[object]) -> str:
    """Render one row of values as an escaped ``<tr>`` of ``<td>`` cells.

    Args:
        values: The row's cell values in column order. Every value is escaped
            here, so no caller ever hand-builds a ``<td>`` and no caller can
            forget to escape one.

    Returns:
        An HTML ``<tr>`` element.
    """
    cells = "".join(f"<td>{escape(value)}</td>" for value in values)
    return f"<tr>{cells}</tr>"


def table(
    title: str, headers: Sequence[str], value_rows: Sequence[Sequence[object]]
) -> str:
    """Render titled, column-labelled tabular data as a well-formed table.

    The rows are wrapped in a real ``<table>`` with a ``<thead>`` header row and
    a ``<tbody>``, because a ``<tr>`` outside a table is foster-parented by the
    browser and its cells collapse into unlabelled running text -- the operator
    then cannot tell which number is which (issue #275).

    Args:
        title: The section heading (a fixed, trusted literal from the caller).
        headers: The column labels, in the same order the callers' rows supply
            their values.
        value_rows: One sequence of raw (unescaped) cell values per data row; an
            empty sequence renders the shared "no data yet" placeholder instead
            of an empty table, so an absent read model never reads as a table
            whose rows happen to be missing.

    Returns:
        A ``<section>`` HTML fragment containing the ``<table>``, or the
        placeholder section when there are no rows.
    """
    if not value_rows:
        return section(title, [])
    header_cells = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "\n".join(_data_row(values) for values in value_rows)
    markup = (
        "<table>\n"
        f"<thead><tr>{header_cells}</tr></thead>\n"
        f"<tbody>\n{body}\n</tbody>\n"
        "</table>"
    )
    return section(title, [markup])
