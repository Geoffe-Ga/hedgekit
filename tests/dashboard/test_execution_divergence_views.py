"""Failing-first tests for the dashboard's `/execution` + `/divergence` routes
(issue #58, RED).

`windbreak.dashboard.views.DashboardReadModels` does not yet carry the two new
attributes these routes render, and `windbreak.dashboard.app`'s `_VIEWS` table
does not yet route `/execution` or `/divergence`, so every test below that
seeds those attributes fails at fixture-construction time with `TypeError:
__init__() got an unexpected keyword argument 'execution_quality'` (or
`'live_divergence'`); every test that merely requests one of the two new
paths against the *unmodified* `_VIEWS` table gets today's ordinary 404
instead of the eventual 401/200 -- either way the expected Gate 1 RED state
for issue #58's dashboard surface. Reuses `tests/dashboard/test_app.py`'s
`TEST_TOKEN`/`_bearer`/`_get` helpers (DRY), mirroring
`test_app_scheduler_routes.py`'s own reuse of the same helpers.

Pins, mirroring `test_app_scheduler_routes.py`'s already-proven contract for
`/positions`/`/equity`/`/decisions`:

- `/execution` and `/divergence` are gated behind the same bearer auth as `/`
  (401 without/with-wrong bearer, 200 with the correct one).
- With no read-models source wired, both routes still 200 with a "no data
  yet" placeholder, never a 404 or a 500.
- A hostile (forged-HTML) value flowing through either view's rendered rows
  is escaped, never raw.
- `/` itself is unaffected by the two new routes existing alongside it.

ASSUMPTION this file pins (the architecture plan names the two new routes and
their source views but not `DashboardReadModels`' exact new attribute names):
the read-model bundle grows two new list-of-row attributes,
`execution_quality` and `live_divergence`, mirroring the existing
`positions`/`equity_curve`/`decisions` naming (module name singular/plural
matching the read-model's own shape, not the route path). If the real
implementation names these differently, this is a design point to reconcile,
not a signal to silently rename the assertions to match whichever lands
first.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from tests.dashboard.html_structure import assert_data_table
from tests.dashboard.test_app import TEST_TOKEN, _bearer, _get

if TYPE_CHECKING:
    import http.server
    from collections.abc import Iterator

pytestmark = pytest.mark.timeout(15)


def _make_read_models_source(rows: dict[str, list[dict[str, object]]]):
    """Build a zero-arg callable returning a fixed `DashboardReadModels`
    seeded with `execution_quality`/`live_divergence` rows in addition to the
    three pre-existing read models (left empty here).

    Args:
        rows: A mapping of `execution_quality`/`live_divergence` to their
            read-model row lists.

    Returns:
        A callable suitable for `create_server(read_models_source=...)`.
    """

    def _source() -> object:
        from windbreak.dashboard.views import DashboardReadModels

        return DashboardReadModels(
            positions=[],
            equity_curve=[],
            decisions=[],
            execution_quality=rows.get("execution_quality", []),
            live_divergence=rows.get("live_divergence", []),
        )

    return _source


@pytest.fixture
def dashboard_server_with_execution_divergence_rows() -> Iterator[
    tuple[http.server.ThreadingHTTPServer, tuple[str, int]]
]:
    """Start a dashboard server seeded with execution/divergence read-model rows."""
    from windbreak.dashboard.app import DashboardStatus, create_server

    rows = {
        "execution_quality": [
            {
                "seq": 5,
                "created_at": "2026-01-01T00:00:00.000000+00:00",
                "event_type": "ExecutionQualityRecorded",
                "data": {
                    "fill_id": "F-<script>alert(1)</script>",
                    "market_ticker": "MKT-EXEC",
                    "slippage_micros": 100_000,
                },
            }
        ],
        "live_divergence": [
            {
                "seq": 9,
                "created_at": "2026-01-01T00:00:00.000000+00:00",
                "event_type": "LiveDivergenceSampled",
                "data": {
                    "live_slippage_ratio_ppm": 1_100_000,
                    "live_brier_degradation_ppm": "UNDEFINED",
                },
            }
        ],
    }
    server = create_server(
        token=TEST_TOKEN,
        status_source=lambda: DashboardStatus(mode="LIVE_MICRO", last_heartbeat=None),
        read_models_source=_make_read_models_source(rows),
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, server.server_address
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("path", ["/execution", "/divergence"])
class TestNewRoutesShareTheExistingBearerAuth:
    """Every new route is gated behind the same auth as `/`."""

    def test_missing_bearer_returns_401(
        self,
        path: str,
        dashboard_server_with_execution_divergence_rows: tuple[
            http.server.ThreadingHTTPServer, tuple[str, int]
        ],
    ) -> None:
        """No `Authorization` header at all is unauthenticated on the new route."""
        _server, address = dashboard_server_with_execution_divergence_rows

        status, headers, _body = _get(address, path)

        assert status == 401
        assert "WWW-Authenticate" in headers

    def test_wrong_bearer_returns_401(
        self,
        path: str,
        dashboard_server_with_execution_divergence_rows: tuple[
            http.server.ThreadingHTTPServer, tuple[str, int]
        ],
    ) -> None:
        """An incorrect bearer token is rejected on the new route too."""
        _server, address = dashboard_server_with_execution_divergence_rows

        status, headers, _body = _get(address, path, headers=_bearer("wrong-token"))

        assert status == 401
        assert "WWW-Authenticate" in headers

    def test_correct_bearer_returns_200(
        self,
        path: str,
        dashboard_server_with_execution_divergence_rows: tuple[
            http.server.ThreadingHTTPServer, tuple[str, int]
        ],
    ) -> None:
        """The correct bearer token is accepted on the new route."""
        _server, address = dashboard_server_with_execution_divergence_rows

        status, _headers, _body = _get(address, path, headers=_bearer(TEST_TOKEN))

        assert status == 200


def test_execution_route_renders_the_seeded_ticker(
    dashboard_server_with_execution_divergence_rows: tuple[
        http.server.ThreadingHTTPServer, tuple[str, int]
    ],
) -> None:
    """`/execution` renders the seeded read model's market ticker."""
    _server, address = dashboard_server_with_execution_divergence_rows

    status, _headers, body = _get(address, "/execution", headers=_bearer(TEST_TOKEN))

    assert status == 200
    assert "MKT-EXEC" in body


def test_execution_route_escapes_a_hostile_fill_id(
    dashboard_server_with_execution_divergence_rows: tuple[
        http.server.ThreadingHTTPServer, tuple[str, int]
    ],
) -> None:
    """`/execution` escapes a hostile ledger-derived fill id."""
    _server, address = dashboard_server_with_execution_divergence_rows

    status, _headers, body = _get(address, "/execution", headers=_bearer(TEST_TOKEN))

    assert status == 200
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "<script>" not in body


def test_execution_route_serves_a_well_formed_labelled_table(
    dashboard_server_with_execution_divergence_rows: tuple[
        http.server.ThreadingHTTPServer, tuple[str, int]
    ],
) -> None:
    """The served `/execution` page is a real table, not orphan rows (#275).

    Asserted over the *whole served page*, so it covers `_VIEW_TEMPLATE`'s
    wrapping as well as the renderer: every `<tr>`/`<td>` must sit inside a
    single `<table>` with a header row naming the columns. The hostile fill id
    is expected back as cell *text*; had it been interpolated raw it would parse
    as a `<script>` element instead and the cell text would not match.
    """
    _server, address = dashboard_server_with_execution_divergence_rows

    status, _headers, body = _get(address, "/execution", headers=_bearer(TEST_TOKEN))

    assert status == 200
    assert_data_table(
        body,
        headers=["fill_id", "market_ticker", "slippage_micros"],
        rows=[["F-<script>alert(1)</script>", "MKT-EXEC", "100000"]],
    )


def test_render_live_divergence_renders_a_labelled_table_of_each_row() -> None:
    """The divergence renderer emits a labelled table, placeholders included.

    A sampled row carries neither threshold nor `trigger`, so those cells render
    the `n/a` placeholder -- structurally identical cells, never omitted, so the
    columns stay aligned with their headers (issue #275).
    """
    from windbreak.dashboard.views import render_live_divergence

    rows = [
        {
            "seq": 9,
            "created_at": "2026-01-01T00:00:00.000000+00:00",
            "event_type": "LiveDivergenceSampled",
            "data": {
                "live_slippage_ratio_ppm": 1_100_000,
                "live_brier_degradation_ppm": "UNDEFINED",
            },
        }
    ]

    html = render_live_divergence(rows)

    assert_data_table(
        html,
        headers=[
            "live_slippage_ratio_ppm",
            "live_slippage_ratio_limit_ppm",
            "live_brier_degradation_ppm",
            "live_brier_degradation_band_ppm",
            "trigger",
        ],
        rows=[["1100000", "n/a", "UNDEFINED", "n/a", "n/a"]],
    )


def test_divergence_route_renders_the_seeded_ratio(
    dashboard_server_with_execution_divergence_rows: tuple[
        http.server.ThreadingHTTPServer, tuple[str, int]
    ],
) -> None:
    """`/divergence` renders the seeded slippage-ratio threshold value."""
    _server, address = dashboard_server_with_execution_divergence_rows

    status, _headers, body = _get(address, "/divergence", headers=_bearer(TEST_TOKEN))

    assert status == 200
    assert "1100000" in body or "1_100_000" in body


@pytest.mark.parametrize("path", ["/execution", "/divergence"])
def test_new_route_renders_no_data_yet_when_read_models_source_is_none(
    path: str,
) -> None:
    """With no `read_models_source` wired (the default), the route still 200s
    with a "no data yet" placeholder, never a 404 or a 500 -- mirroring
    `windbreak/dashboard/app.py`'s existing `last_heartbeat=None` ->
    `"never"`-placeholder precedent.
    """
    from windbreak.dashboard.app import DashboardStatus, create_server

    server = create_server(
        token=TEST_TOKEN,
        status_source=lambda: DashboardStatus(mode="RESEARCH", last_heartbeat=None),
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _headers, body = _get(
            server.server_address, path, headers=_bearer(TEST_TOKEN)
        )

        assert status == 200
        assert "no data yet" in body.lower()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_divergence_route_renders_and_escapes_the_breach_trigger() -> None:
    """`/divergence` renders a `LiveDivergenceBreached` row's `trigger`,
    HTML-escaped -- the primary operator-facing audit trail for the SPEC
    S10.10 automatic-demotion gate (Gate 4 round-2 review fix, Fix 2).

    Today `render_live_divergence`'s `_divergence_row` renders only the four
    fixed sampled-series/threshold fields (`live_slippage_ratio_ppm`,
    `live_slippage_ratio_limit_ppm`, `live_brier_degradation_ppm`,
    `live_brier_degradation_band_ppm`); it never reads a row's `trigger` key
    at all, so this fails: neither the escaped nor the raw trigger text
    appears anywhere in the response body.
    """
    from windbreak.dashboard.app import DashboardStatus, create_server

    hostile_trigger = "LIVE_PAPER_SLIPPAGE_DIVERGENCE<script>alert(1)</script>"
    rows = {
        "live_divergence": [
            {
                "seq": 9,
                "created_at": "2026-01-01T00:00:00.000000+00:00",
                "event_type": "LiveDivergenceBreached",
                "data": {
                    "live_slippage_ratio_ppm": 1_600_000,
                    "live_slippage_ratio_limit_ppm": 1_500_000,
                    "live_brier_degradation_ppm": "UNDEFINED",
                    "live_brier_degradation_band_ppm": 50_000,
                    "trigger": hostile_trigger,
                },
            }
        ],
    }
    server = create_server(
        token=TEST_TOKEN,
        status_source=lambda: DashboardStatus(mode="LIVE_MICRO", last_heartbeat=None),
        read_models_source=_make_read_models_source(rows),
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _headers, body = _get(
            server.server_address, "/divergence", headers=_bearer(TEST_TOKEN)
        )

        assert status == 200
        assert (
            "LIVE_PAPER_SLIPPAGE_DIVERGENCE&lt;script&gt;alert(1)&lt;/script&gt;"
            in body
        )
        assert "<script>" not in body
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_root_path_is_unaffected_by_the_two_new_routes_existing(
    dashboard_server_with_execution_divergence_rows: tuple[
        http.server.ThreadingHTTPServer, tuple[str, int]
    ],
) -> None:
    """`/` still renders the ordinary status page, unaffected by `/execution`
    and `/divergence` existing alongside it.
    """
    _server, address = dashboard_server_with_execution_divergence_rows

    status, _headers, body = _get(address, "/", headers=_bearer(TEST_TOKEN))

    assert status == 200
    assert "windbreak dashboard" in body
