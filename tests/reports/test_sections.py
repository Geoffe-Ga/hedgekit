"""Tests for `windbreak.reports.sections` (issue #255, RED).

`windbreak/reports/sections.py` does not exist yet, so every import below
fails collection with `ModuleNotFoundError: No module named
'windbreak.reports.sections'` -- the expected Gate 1 RED state for issue
#255.

Three pure renderers (mirroring `windbreak.reports.providers.render_provider_lines`'s
own shape/tone: key=value lines, integer-only, no floats, no true-division,
caller owns the "No data yet." fallback), each consuming the ledger read-model
rows `windbreak.ledger.rebuild` already projects:

- `render_equity_lines(rows) -> str` over `equity_curve.json` rows
  (`data = {equity_micros, floor_micros, epoch_s}`): a leading
  `equity_samples=<n>` count line, then one
  `epoch_s=.. equity_micros=.. floor_micros=.. buffer_micros=<+/-..>` line per
  row, in ledger order. `buffer_micros` is the signed integer
  `equity_micros - floor_micros`.
- `render_position_lines(rows) -> str` over `positions.json` rows (at most
  one, the latest snapshot; `data = {positions: [{ticker, quantity_centis,
  average_price_pips}]}`): `snapshots=<0|1>`, then -- only when a snapshot is
  present -- `open_positions=<n>` and one `ticker=.. quantity_centis=..
  average_price_pips=..` line per held position, in order.
- `render_decision_lines(rows) -> str` over `selector_decisions.json` rows
  (interleaved `SelectorDecisionRecorded`/`IntentApproved`/`IntentVetoed`):
  `decision_events=<n>`, then one `event=.. subject=.. reasons=..` line per
  row, in ledger order. `subject` is `data.get("market_ticker") or
  data.get("intent_id")` (copied from
  `windbreak.dashboard.views.decisions._subject`); `reasons` is
  `data["reasons"]` joined with `'; '`, or the literal `none` when empty or
  absent.

Every ledger-derived string (ticker, subject, each reason) collapses embedded
CR/LF to a single space before rendering, so a hostile reason like
`"\\n## Forged"` can never forge a markdown heading in the weekly report these
lines are embedded into verbatim (security-critical: selector/veto reasons are
forecast/LLM-adjacent input, exactly as
`windbreak.dashboard.views.decisions`'s own HTML-escape docstring already
flags for the HTML side of this same data).
"""

from __future__ import annotations


def _equity_row(
    *,
    seq: int = 1,
    equity_micros: int,
    floor_micros: int,
    epoch_s: int,
) -> dict[str, object]:
    """Build one hand-built `equity_curve.json` read-model row."""
    return {
        "seq": seq,
        "created_at": "2026-01-01T00:00:00.000000+00:00",
        "event_type": "EquitySampled",
        "data": {
            "equity_micros": equity_micros,
            "floor_micros": floor_micros,
            "epoch_s": epoch_s,
        },
    }


def _positions_row(
    *, seq: int = 1, positions: list[dict[str, object]]
) -> dict[str, object]:
    """Build one hand-built `positions.json` read-model row."""
    return {
        "seq": seq,
        "created_at": "2026-01-01T00:00:00.000000+00:00",
        "event_type": "PositionsSnapshotRecorded",
        "data": {"positions": positions},
    }


def _decision_row(
    *,
    seq: int = 1,
    event_type: str = "SelectorDecisionRecorded",
    data: dict[str, object],
) -> dict[str, object]:
    """Build one hand-built `selector_decisions.json` read-model row."""
    return {
        "seq": seq,
        "created_at": "2026-01-01T00:00:00.000000+00:00",
        "event_type": event_type,
        "data": data,
    }


# ---------------------------------------------------------------------------
# render_equity_lines
# ---------------------------------------------------------------------------


def test_render_equity_lines_empty_renders_zero_count_and_no_further_lines() -> None:
    """An empty rows list renders only `equity_samples=0`, no sample line."""
    from windbreak.reports.sections import render_equity_lines

    text = render_equity_lines([])

    assert text == "equity_samples=0"


def test_render_equity_lines_single_row_renders_the_exact_line_format() -> None:
    """One row renders the pinned count line then the pinned sample line,
    byte for byte.
    """
    from windbreak.reports.sections import render_equity_lines

    row = _equity_row(equity_micros=11_000_000, floor_micros=10_000_000, epoch_s=1700)

    text = render_equity_lines([row])

    assert text == (
        "equity_samples=1\n"
        "epoch_s=1700 equity_micros=11000000 floor_micros=10000000 "
        "buffer_micros=+1000000"
    )


def test_render_equity_lines_multiple_rows_preserve_ledger_order() -> None:
    """Two rows render two sample lines in the same order as `rows`, never
    reordered (e.g. by `epoch_s`).
    """
    from windbreak.reports.sections import render_equity_lines

    rows = [
        _equity_row(
            seq=1, equity_micros=9_000_000, floor_micros=10_000_000, epoch_s=100
        ),
        _equity_row(
            seq=2, equity_micros=12_000_000, floor_micros=10_000_000, epoch_s=200
        ),
    ]

    text = render_equity_lines(rows)

    assert text == (
        "equity_samples=2\n"
        "epoch_s=100 equity_micros=9000000 floor_micros=10000000 "
        "buffer_micros=-1000000\n"
        "epoch_s=200 equity_micros=12000000 floor_micros=10000000 "
        "buffer_micros=+2000000"
    )


def test_render_equity_lines_buffer_is_signed_positive_when_equity_exceeds_floor() -> (
    None
):
    """Equity above the floor renders an explicit `+` sign, not a bare
    unsigned number.
    """
    from windbreak.reports.sections import render_equity_lines

    row = _equity_row(equity_micros=10_500_000, floor_micros=10_000_000, epoch_s=1)

    text = render_equity_lines([row])

    assert "buffer_micros=+500000" in text
    assert "buffer_micros=500000" not in text


def test_render_equity_lines_buffer_is_signed_negative_when_equity_below_floor() -> (
    None
):
    """Equity below the floor renders the exact `-` sign, never suppressed or
    rendered as `n/a`.
    """
    from windbreak.reports.sections import render_equity_lines

    row = _equity_row(equity_micros=9_500_000, floor_micros=10_000_000, epoch_s=1)

    text = render_equity_lines([row])

    assert "buffer_micros=-500000" in text


def test_render_equity_lines_buffer_is_signed_plus_zero_at_the_floor() -> None:
    """Equity exactly at the floor renders the boundary buffer as `+0`, never a
    bare `0` -- the `:+d` sign is present even at the zero boundary.
    """
    from windbreak.reports.sections import render_equity_lines

    row = _equity_row(equity_micros=10_000_000, floor_micros=10_000_000, epoch_s=1)

    text = render_equity_lines([row])

    assert "buffer_micros=+0" in text


def test_render_equity_lines_output_is_integer_only_no_decimal_points() -> None:
    """Every numeric field renders as a plain integer -- no floats, no `.`
    decimal points anywhere in the rendered text (SPEC S6.1).
    """
    from windbreak.reports.sections import render_equity_lines

    row = _equity_row(equity_micros=11_000_000, floor_micros=10_000_000, epoch_s=1700)

    text = render_equity_lines([row])

    assert "." not in text


# ---------------------------------------------------------------------------
# render_position_lines
# ---------------------------------------------------------------------------


def test_render_position_lines_empty_renders_zero_snapshots_and_no_further_lines() -> (
    None
):
    """No snapshot ever recorded renders only `snapshots=0` -- never a bare
    `open_positions=0`, which is reserved for a real-but-flat snapshot."""
    from windbreak.reports.sections import render_position_lines

    text = render_position_lines([])

    assert text == "snapshots=0"
    assert "open_positions=" not in text


def test_render_position_lines_flat_snapshot_renders_open_positions_zero() -> None:
    """A real snapshot recording zero held positions (flat) renders
    `snapshots=1` and `open_positions=0`, distinct from both the `snapshots=0`
    no-data state and a populated snapshot -- no ticker rows follow.
    """
    from windbreak.reports.sections import render_position_lines

    row = _positions_row(positions=[])

    text = render_position_lines([row])

    assert text == "snapshots=1\nopen_positions=0"
    assert "ticker=" not in text


def test_render_position_lines_populated_snapshot_renders_count_and_rows_in_order() -> (
    None
):
    """A snapshot holding two positions renders `open_positions=2` and both
    position rows, in the same order as the snapshot's `positions` list.
    """
    from windbreak.reports.sections import render_position_lines

    row = _positions_row(
        positions=[
            {"ticker": "MKT-A", "quantity_centis": 200, "average_price_pips": 4600},
            {"ticker": "MKT-B", "quantity_centis": 50, "average_price_pips": 5100},
        ]
    )

    text = render_position_lines([row])

    assert text == (
        "snapshots=1\n"
        "open_positions=2\n"
        "ticker=MKT-A quantity_centis=200 average_price_pips=4600\n"
        "ticker=MKT-B quantity_centis=50 average_price_pips=5100"
    )


def test_render_position_lines_only_the_single_row_present_is_rendered() -> None:
    """`positions_read_model` already holds at most one row (the latest
    snapshot); the renderer trusts that contract and never merges/dedupes
    across rows -- proven here by a single distinctive row rendering exactly
    once.
    """
    from windbreak.reports.sections import render_position_lines

    row = _positions_row(
        positions=[
            {"ticker": "MKT-ONLY", "quantity_centis": 1, "average_price_pips": 1}
        ]
    )

    text = render_position_lines([row])

    assert text.count("ticker=") == 1


def test_render_position_lines_ticker_newline_forgery_is_neutralized() -> None:
    """A ticker carrying embedded CR/LF cannot forge a markdown heading (or
    inject an extra line) once embedded verbatim into the weekly report.
    """
    from windbreak.reports.sections import render_position_lines

    row = _positions_row(
        positions=[
            {
                "ticker": "MKT\n## Forged",
                "quantity_centis": 1,
                "average_price_pips": 1,
            }
        ]
    )

    text = render_position_lines([row])

    assert not any(line.startswith("## ") for line in text.splitlines())
    assert "\n## Forged" not in text


# ---------------------------------------------------------------------------
# render_decision_lines
# ---------------------------------------------------------------------------


def test_render_decision_lines_empty_renders_zero_count_and_no_further_lines() -> None:
    """No decision ever recorded renders only `decision_events=0`."""
    from windbreak.reports.sections import render_decision_lines

    text = render_decision_lines([])

    assert text == "decision_events=0"


def test_render_decision_lines_selector_decision_subject_is_the_market_ticker() -> None:
    """A `SelectorDecisionRecorded` row's subject is its `market_ticker`
    (copied subject rule from `windbreak.dashboard.views.decisions._subject`).
    """
    from windbreak.reports.sections import render_decision_lines

    row = _decision_row(
        event_type="SelectorDecisionRecorded",
        data={
            "forecast_id": "fc-0001",
            "market_ticker": "MKT-DEEP",
            "intent_count": 1,
            "reasons": [],
        },
    )

    text = render_decision_lines([row])

    assert "event=SelectorDecisionRecorded subject=MKT-DEEP reasons=none" in text


def test_render_decision_lines_bare_intent_approved_subject_is_the_intent_id() -> None:
    """A bare `IntentApproved` row (no `market_ticker` key at all) falls back
    to its `intent_id` as the subject.
    """
    from windbreak.reports.sections import render_decision_lines

    row = _decision_row(
        event_type="IntentApproved",
        data={"intent_id": "intent-001", "reasons": []},
    )

    text = render_decision_lines([row])

    assert "event=IntentApproved subject=intent-001 reasons=none" in text


def test_render_decision_lines_bare_intent_vetoed_subject_is_the_intent_id() -> None:
    """A bare `IntentVetoed` row likewise falls back to its `intent_id`."""
    from windbreak.reports.sections import render_decision_lines

    row = _decision_row(
        event_type="IntentVetoed",
        data={
            "intent_id": "intent-002",
            "reasons": ["exchange status stale or missing"],
        },
    )

    text = render_decision_lines([row])

    assert (
        "event=IntentVetoed subject=intent-002 reasons=exchange status stale or missing"
    ) in text


def test_render_decision_lines_multiple_reasons_are_joined_with_semicolon_space() -> (
    None
):
    """Two-or-more reasons render joined with the literal `'; '` separator."""
    from windbreak.reports.sections import render_decision_lines

    row = _decision_row(
        event_type="IntentVetoed",
        data={
            "intent_id": "intent-003",
            "reasons": [
                "exchange status stale or missing",
                "pipeline heartbeat stale or missing",
            ],
        },
    )

    text = render_decision_lines([row])

    assert (
        "reasons=exchange status stale or missing; pipeline heartbeat stale or missing"
    ) in text


def test_render_decision_lines_absent_reasons_key_renders_the_literal_none() -> None:
    """A row whose `data` carries no `reasons` key at all (not even an empty
    list) still renders `reasons=none`, never a crash or `reasons=` blank.
    """
    from windbreak.reports.sections import render_decision_lines

    row = _decision_row(
        event_type="SelectorDecisionRecorded",
        data={"forecast_id": "fc-0002", "market_ticker": "MKT-A", "intent_count": 0},
    )

    text = render_decision_lines([row])

    assert "reasons=none" in text


def test_render_decision_lines_preserves_ledger_order_across_event_types() -> None:
    """A `SelectorDecisionRecorded` followed by its `IntentVetoed` verdict
    renders in that same interleaved order, both counted.
    """
    from windbreak.reports.sections import render_decision_lines

    rows = [
        _decision_row(
            seq=1,
            event_type="SelectorDecisionRecorded",
            data={
                "forecast_id": "fc-0003",
                "market_ticker": "MKT-C",
                "intent_count": 1,
                "reasons": [],
            },
        ),
        _decision_row(
            seq=2,
            event_type="IntentVetoed",
            data={"intent_id": "intent-004", "reasons": ["net_edge_below_minimum"]},
        ),
    ]

    text = render_decision_lines(rows)

    assert text.startswith("decision_events=2\n")
    selector_index = text.index("event=SelectorDecisionRecorded")
    vetoed_index = text.index("event=IntentVetoed")
    assert selector_index < vetoed_index


def test_render_decision_lines_reason_newline_forgery_is_neutralized() -> None:
    """A hostile reason carrying an embedded newline followed by a markdown
    heading marker (`"\\n## Forged"`) can never forge a heading once embedded
    verbatim into the weekly report -- CR/LF is collapsed to a single space.
    """
    from windbreak.reports.sections import render_decision_lines

    row = _decision_row(
        event_type="IntentVetoed",
        data={"intent_id": "intent-005", "reasons": ["\n## Forged heading"]},
    )

    text = render_decision_lines([row])

    assert not any(line.startswith("## ") for line in text.splitlines())
    assert "\n## Forged heading" not in text


def test_render_decision_lines_subject_newline_forgery_is_neutralized() -> None:
    """A hostile `market_ticker` subject carrying an embedded CRLF/heading
    payload is likewise neutralized.
    """
    from windbreak.reports.sections import render_decision_lines

    row = _decision_row(
        event_type="SelectorDecisionRecorded",
        data={
            "forecast_id": "fc-0004",
            "market_ticker": "MKT\r\n## Forged",
            "intent_count": 0,
            "reasons": [],
        },
    )

    text = render_decision_lines([row])

    assert not any(line.startswith("## ") for line in text.splitlines())
    assert "\r\n## Forged" not in text
    assert "\n## Forged" not in text


def test_render_decision_lines_neutralizes_every_reason_not_just_the_first() -> None:
    """Neutralization applies to each reason in a multi-reason list, so a hostile
    payload in a middle/trailing reason is collapsed exactly like a leading one --
    there is no first-only escape path.
    """
    from windbreak.reports.sections import render_decision_lines

    row = _decision_row(
        event_type="SelectorDecisionRecorded",
        data={
            "forecast_id": "fc-0005",
            "market_ticker": "MKT",
            "intent_count": 0,
            "reasons": ["clean reason", "\n## Forged middle", "\r\n## Forged tail"],
        },
    )

    text = render_decision_lines([row])

    assert not any(line.startswith("## ") for line in text.splitlines())
    assert "\n## Forged" not in text
    assert "\r\n## Forged" not in text
