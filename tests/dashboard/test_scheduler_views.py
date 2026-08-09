"""Failing-first tests for the new dashboard views package (issue #48, RED).

`windbreak.dashboard.views` does not exist yet, so every import below fails
collection with `ModuleNotFoundError: No module named
'windbreak.dashboard.views'` -- the expected Gate 1 RED state for issue #48.

Each renderer is a pure function over the *read-model row shape*
`windbreak.ledger.rebuild` already produces for every projection
(`{seq, created_at, event_type, data}`), so the dashboard reuses the same
projections `windbreak rebuild` writes rather than re-deriving its own view of
the ledger. Every renderer must `html.escape` any ledger-derived string before
interpolating it -- selector/veto reasons are forecast/LLM-adjacent and
therefore an XSS surface (mirrors `windbreak/dashboard/app.py`'s own existing
`html.escape` treatment of `mode`/`last_heartbeat`).

The two table views (`render_positions`, `render_equity_vs_floor`) are asserted
through `tests.dashboard.html_structure.assert_data_table` rather than by
substring (issue #275): a substring assertion over cell text cannot see whether
a `<table>` wraps the rows at all, which is precisely how the orphan-`<tr>`
markup reached production.
"""

from __future__ import annotations

import dataclasses

import pytest

from tests.dashboard.html_structure import assert_data_table


def test_dashboard_read_models_is_frozen() -> None:
    """`DashboardReadModels` is an immutable value object."""
    from windbreak.dashboard.views import DashboardReadModels

    read_models = DashboardReadModels(positions=[], equity_curve=[], decisions=[])

    with pytest.raises(dataclasses.FrozenInstanceError):
        read_models.positions = [{"whatever": True}]  # type: ignore[misc]


def test_dashboard_read_models_defaults_to_empty_when_constructed_bare() -> None:
    """An empty `DashboardReadModels` is the documented "no data yet" input."""
    from windbreak.dashboard.views import DashboardReadModels

    read_models = DashboardReadModels(positions=[], equity_curve=[], decisions=[])

    assert read_models.positions == []
    assert read_models.equity_curve == []
    assert read_models.decisions == []


def test_render_positions_shows_no_data_yet_placeholder_when_empty() -> None:
    """An empty positions read model renders a readable placeholder, not a
    crash or an empty table.
    """
    from windbreak.dashboard.views import render_positions

    html = render_positions([])

    assert "no data yet" in html.lower()


def test_render_positions_renders_a_labelled_table_of_each_position() -> None:
    """A populated positions read model renders a real, labelled table.

    Asserted structurally (issue #275): the previous version of this test only
    checked that ``"MKT-DEEP"`` and ``"200"`` appeared *somewhere* in the string,
    which passed against the orphan ``<tr>``/``<td>`` markup this view used to
    emit -- markup a browser foster-parents out of its section and renders as
    run-together text. Deleting the ``<table>`` wrapper now fails this test.
    """
    from windbreak.dashboard.views import render_positions

    rows = [
        {
            "seq": 8,
            "created_at": "2026-01-01T00:00:00.000000+00:00",
            "event_type": "PositionsSnapshotRecorded",
            "data": {
                "positions": [
                    {
                        "ticker": "MKT-DEEP",
                        "quantity_centis": 200,
                        "average_price_pips": 4600,
                    }
                ]
            },
        }
    ]

    html = render_positions(rows)

    assert_data_table(
        html,
        headers=["ticker", "quantity_centis", "average_price_pips"],
        rows=[["MKT-DEEP", "200", "4600"]],
    )


def test_render_equity_vs_floor_shows_no_data_yet_placeholder_when_empty() -> None:
    """An empty equity-curve read model renders a readable placeholder."""
    from windbreak.dashboard.views import render_equity_vs_floor

    html = render_equity_vs_floor([])

    assert "no data yet" in html.lower()


def test_render_equity_vs_floor_renders_a_labelled_table_of_each_sample() -> None:
    """A populated equity-curve read model renders a real, labelled table.

    Asserted structurally (issue #275) for the same reason as the positions
    view: ``"1000000000" in html`` passed against orphan ``<td>`` markup with no
    ``<table>`` ancestor, so it could never have caught the missing structure.
    """
    from windbreak.dashboard.views import render_equity_vs_floor

    rows = [
        {
            "seq": 2,
            "created_at": "2026-01-01T00:00:00.000000+00:00",
            "event_type": "EquitySampled",
            "data": {
                "equity_micros": 1_000_000_000,
                "floor_micros": 0,
                "epoch_s": 1_700_000_000,
            },
        }
    ]

    html = render_equity_vs_floor(rows)

    assert_data_table(
        html,
        headers=["epoch_s", "equity_micros", "floor_micros"],
        rows=[["1700000000", "1000000000", "0"]],
    )


def test_render_decisions_shows_no_data_yet_placeholder_when_empty() -> None:
    """An empty decisions read model renders a readable placeholder."""
    from windbreak.dashboard.views import render_decisions

    html = render_decisions([])

    assert "no data yet" in html.lower()


def test_render_decisions_renders_reasons_for_a_selector_decision_row() -> None:
    """A `SelectorDecisionRecorded` row renders its market ticker and reasons."""
    from windbreak.dashboard.views import render_decisions

    rows = [
        {
            "seq": 3,
            "created_at": "2026-01-01T00:00:00.000000+00:00",
            "event_type": "SelectorDecisionRecorded",
            "data": {
                "forecast_id": "fc-0001",
                "market_ticker": "MKT-DEEP",
                "intent_count": 0,
                "reasons": ["fail:net_edge_min: net_edge_ppm=-500"],
            },
        }
    ]

    html = render_decisions(rows)

    assert "MKT-DEEP" in html
    assert "net_edge_min" in html


def test_render_decisions_escapes_a_hostile_reason_string() -> None:
    """A hostile (forged-HTML) reason string is rendered escaped, never raw.

    Selector/veto reasons flow from forecast/LLM-adjacent input; rendering
    them unescaped would be a stored-XSS vector (mirrors
    `tests/dashboard/test_app.py::test_status_fields_are_html_escaped`).
    """
    from windbreak.dashboard.views import render_decisions

    rows = [
        {
            "seq": 4,
            "created_at": "2026-01-01T00:00:00.000000+00:00",
            "event_type": "IntentVetoed",
            "data": {
                "intent_id": "intent-0001",
                "reasons": ["<script>alert(1)</script>"],
            },
        }
    ]

    html = render_decisions(rows)

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>" not in html


def test_render_decisions_escapes_a_hostile_ticker_string() -> None:
    """A hostile ticker/forecast_id string is also rendered escaped."""
    from windbreak.dashboard.views import render_decisions

    rows = [
        {
            "seq": 3,
            "created_at": "2026-01-01T00:00:00.000000+00:00",
            "event_type": "SelectorDecisionRecorded",
            "data": {
                "forecast_id": "fc-0001",
                "market_ticker": "<img src=x onerror=alert(1)>",
                "intent_count": 0,
                "reasons": [],
            },
        }
    ]

    html = render_decisions(rows)

    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img" in html
