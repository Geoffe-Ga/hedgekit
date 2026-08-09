"""Weekly-report equity/position/decision section renderers (issue #255).

Three pure renderers -- mirroring
:func:`windbreak.reports.providers.render_provider_lines`'s own shape and tone
(``key=value`` lines, integer-only, no floats, no true-division; the caller owns
the ``No data yet.`` empty-state fallback) -- turning the ledger read-model rows
:mod:`windbreak.ledger.rebuild` already projects into the bodies embedded
verbatim under the weekly report's three original #48 stub headings
(``## Equity vs floor``, ``## Positions``, ``## Decisions``):

- :func:`render_equity_lines` over ``equity_curve.json`` rows: a leading
  ``equity_samples=<n>`` count line, then one signed
  ``buffer_micros`` sample line per row, in ledger order.
- :func:`render_position_lines` over ``positions.json`` rows (at most one, the
  latest snapshot): ``snapshots=<0|1>`` and -- only when a snapshot is present --
  ``open_positions=<n>`` and one line per held position.
- :func:`render_decision_lines` over ``selector_decisions.json`` rows: a leading
  ``decision_events=<n>`` count line, then one ``event=.. subject=.. reasons=..``
  line per row, in ledger order.

Every figure is an integer in scaled units (micros/centis/pips); the module is
float-free and true-division free (SPEC S6.1), as the ``windbreak/reports``
package is on ``scripts/lint_no_floats.py``'s denylist.

Security: every ledger-derived string embedded verbatim into the markdown report
(a position ``ticker``, a decision ``subject``, each decision ``reason``) is a
forecast/LLM-adjacent value, exactly as
:mod:`windbreak.dashboard.views.decisions`'s HTML-escape docstring flags for the
HTML side of this same data. Each such string is passed through
:func:`_neutralize_newlines` so an embedded ``CR``/``LF`` cannot forge a ``## ``
markdown heading (or inject an extra line) once the body is embedded verbatim.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The literal rendered for a decision row that carries no reasons at all --
#: distinct from a blank ``reasons=`` so an empty reason list reads honestly.
_NO_REASONS = "none"

#: The separator joining a decision row's two-or-more reasons into one field.
_REASON_SEPARATOR = "; "

#: Matches one-or-more consecutive ``CR``/``LF`` characters, collapsed to a
#: single space by :func:`_neutralize_newlines` so a ``\r\n`` pair (or a run)
#: cannot inject a line break or forge a markdown heading.
_NEWLINE_RUN = re.compile(r"[\r\n]+")


def _neutralize_newlines(value: str) -> str:
    """Collapse embedded ``CR``/``LF`` runs in a ledger-derived string to a space.

    Args:
        value: The raw, forecast/LLM-adjacent ledger string to sanitize.

    Returns:
        ``value`` with every run of one-or-more ``\\r``/``\\n`` characters
        replaced by a single space, so the string cannot forge a ``## `` heading
        or inject an extra line once embedded verbatim into the markdown report.
    """
    return _NEWLINE_RUN.sub(" ", value)


def _equity_sample_line(row: dict[str, object]) -> str:
    """Render one ``equity_curve.json`` row into its pinned sample line.

    Args:
        row: One ``{seq, created_at, event_type, data}`` equity read-model row,
            whose ``data`` carries integer ``equity_micros``/``floor_micros``/
            ``epoch_s`` fields.

    Returns:
        The ``epoch_s=.. equity_micros=.. floor_micros=.. buffer_micros=<+/-..>``
        line, where ``buffer_micros`` is the signed integer difference
        ``equity_micros - floor_micros``.
    """
    data = cast("Mapping[str, object]", row["data"])
    equity_micros = cast("int", data["equity_micros"])
    floor_micros = cast("int", data["floor_micros"])
    epoch_s = cast("int", data["epoch_s"])
    buffer_micros = equity_micros - floor_micros
    return (
        f"epoch_s={epoch_s} equity_micros={equity_micros} "
        f"floor_micros={floor_micros} buffer_micros={buffer_micros:+d}"
    )


def render_equity_lines(rows: list[dict[str, object]]) -> str:
    """Render the equity section: a sample count, then one line per sample.

    Args:
        rows: The ``equity_curve.json`` read-model rows, in ledger order
            (possibly empty).

    Returns:
        The rendered section body: a leading ``equity_samples=<n>`` count line
        followed by one signed-buffer sample line per row, in the same order as
        ``rows``. An empty list renders only ``equity_samples=0``.
    """
    lines = [f"equity_samples={len(rows)}"]
    lines.extend(_equity_sample_line(row) for row in rows)
    return "\n".join(lines)


def _position_line(position: Mapping[str, object]) -> str:
    """Render one held position into its pinned ``ticker=.. ..`` line.

    Args:
        position: One ``{ticker, quantity_centis, average_price_pips}`` entry
            from the latest snapshot's ``positions`` list.

    Returns:
        The ``ticker=.. quantity_centis=.. average_price_pips=..`` line, with the
        ledger-derived ``ticker`` newline-neutralized.
    """
    ticker = _neutralize_newlines(cast("str", position["ticker"]))
    quantity_centis = cast("int", position["quantity_centis"])
    average_price_pips = cast("int", position["average_price_pips"])
    return (
        f"ticker={ticker} quantity_centis={quantity_centis} "
        f"average_price_pips={average_price_pips}"
    )


def render_position_lines(rows: list[dict[str, object]]) -> str:
    """Render the positions section from the latest snapshot (at most one row).

    Args:
        rows: The ``positions.json`` read-model rows -- ``positions_read_model``
            holds at most one, the latest snapshot. The renderer trusts that
            contract and never merges or dedupes across rows.

    Returns:
        The rendered section body. An empty list renders only ``snapshots=0``
        (no snapshot ever recorded). A present snapshot renders ``snapshots=1``,
        an ``open_positions=<n>`` count, then one line per held position in the
        snapshot's order -- a flat snapshot rendering ``open_positions=0`` and no
        position lines, distinct from the ``snapshots=0`` no-data state.
    """
    if not rows:
        return "snapshots=0"
    data = cast("Mapping[str, object]", rows[0]["data"])
    positions = cast("list[Mapping[str, object]]", data["positions"])
    lines = ["snapshots=1", f"open_positions={len(positions)}"]
    lines.extend(_position_line(position) for position in positions)
    return "\n".join(lines)


def _decision_subject(data: Mapping[str, object]) -> str:
    """Return a decision row's subject: its market ticker, else its intent id.

    Replicates the subject rule from
    :func:`windbreak.dashboard.views.decisions._subject` (a
    ``SelectorDecisionRecorded`` carries a ``market_ticker``; a bare
    ``IntentApproved``/``IntentVetoed`` verdict carries only an ``intent_id``)
    without importing a dashboard view into the reports package.

    Args:
        data: The row's ``data`` payload.

    Returns:
        The newline-neutralized ``market_ticker`` when present and truthy, else
        the newline-neutralized ``intent_id``.
    """
    subject = data.get("market_ticker") or data.get("intent_id")
    return _neutralize_newlines(str(subject))


def _decision_reasons(data: Mapping[str, object]) -> str:
    """Render a decision row's reasons into its pinned ``reasons=`` field value.

    Args:
        data: The row's ``data`` payload, whose ``reasons`` list is absent when
            the row carries none.

    Returns:
        The row's reasons -- each newline-neutralized -- joined by ``'; '``, or
        the literal ``none`` when the ``reasons`` key is empty or absent.
    """
    reasons = cast("list[object]", data.get("reasons", []))
    if not reasons:
        return _NO_REASONS
    return _REASON_SEPARATOR.join(
        _neutralize_newlines(str(reason)) for reason in reasons
    )


def _decision_line(row: dict[str, object]) -> str:
    """Render one ``selector_decisions.json`` row into its pinned line.

    Args:
        row: One ``{seq, created_at, event_type, data}`` decision read-model row.

    Returns:
        The ``event=.. subject=.. reasons=..`` line, with the ledger-derived
        event type, subject, and reasons all newline-neutralized.
    """
    event_type = _neutralize_newlines(cast("str", row["event_type"]))
    data = cast("Mapping[str, object]", row["data"])
    subject = _decision_subject(data)
    reasons = _decision_reasons(data)
    return f"event={event_type} subject={subject} reasons={reasons}"


def render_decision_lines(rows: list[dict[str, object]]) -> str:
    """Render the decisions section: an event count, then one line per event.

    Args:
        rows: The ``selector_decisions.json`` read-model rows (interleaved
            ``SelectorDecisionRecorded``/``IntentApproved``/``IntentVetoed``), in
            ledger order (possibly empty).

    Returns:
        The rendered section body: a leading ``decision_events=<n>`` count line
        followed by one ``event=.. subject=.. reasons=..`` line per row, in the
        same interleaved order as ``rows``. An empty list renders only
        ``decision_events=0``.
    """
    lines = [f"decision_events={len(rows)}"]
    lines.extend(_decision_line(row) for row in rows)
    return "\n".join(lines)
