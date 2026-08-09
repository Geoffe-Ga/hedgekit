"""Tests for the dashboard's `/providers` panel (issue #195, RED).

`windbreak.dashboard.views.providers` does not exist yet, so every import
below fails collection with `ModuleNotFoundError: No module named
'windbreak.dashboard.views.providers'` -- the expected Gate 1 RED state for
issue #195.

`render_provider_panel` is a pure renderer over the fleet-observability
panel rows, mirroring `windbreak/dashboard/views/decisions.py`'s own
single-positional-list-arg shape (`Callable[[list[dict[str, object]]], str]`)
so it plugs into `windbreak/dashboard/app.py`'s existing `_ViewSpec`
machinery unchanged. Two row kinds share one list (mirroring
`gateway_events.json`'s own multi-event-type-in-one-projection precedent,
discriminated by an `event_type` key there and a `kind` key here): a
`"provider"` row (one per provider's summary line) and an optional
`"fleet"` row (the two fleet-wide cost-per-forecast/cost-per-resolved
lines that -- unlike the per-provider `cost_per_forecast`, which is
permanently `"n/a"`, issue #281 -- ARE derivable in aggregate).

`DashboardReadModels` gains a `provider_panel: list[ReadModelRow] =
field(default_factory=list)` (issue #195); the route registration and
"No data yet." empty-state tests mirror
`tests/dashboard/test_app_scheduler_routes.py`'s own established pattern for
a new PAPER-loop view (reusing its `TEST_TOKEN`/`_bearer`/`_get` helpers, DRY).

`build_ledger_read_models_source` gains a new keyword-only
`track_record_path: Path | None = None` (issue #195): when supplied, it
points at an M6 track-record artifact JSON file in the shape
`windbreak.forecast.providers.track_record.parse_track_records` accepts, and
each `"provider"`-kind panel row for a provider the artifact covers carries
the REAL `resolved`/`brier_skill_ppm` from that read model instead of the
`"n/a"` placeholder -- including a NEGATIVE skill, rendered verbatim (the
honesty invariant). A provider present in canary status but absent from the
artifact stays `"n/a"` (no fabricated skill). `track_record_path` does not
yet exist on `build_ledger_read_models_source`, so every test below that
passes it fails with `TypeError: build_ledger_read_models_source() got an
unexpected keyword argument 'track_record_path'` -- the expected Gate 1 RED
state for this addition.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from typing import TYPE_CHECKING

import pytest

from tests.dashboard.test_app import TEST_TOKEN, _bearer, _get

if TYPE_CHECKING:
    import http.server
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.timeout(15)

#: A per-provider row with a POSITIVE Brier skill, for the prominence-parity
#: test below.
_POSITIVE_ROW = {
    "kind": "provider",
    "provider": "openai",
    "resolved": 212,
    "brier_skill_ppm": 14_200,
    "canary_status": "OK",
    "abstain_rate_ppm": 90_000,
    "cost_per_forecast": "n/a",
}

#: The identical row shape, but with a NEGATIVE Brier skill and a different
#: provider name -- everything else held constant, for the byte-structural
#: prominence-parity comparison.
_NEGATIVE_ROW = {
    "kind": "provider",
    "provider": "anthropic",
    "resolved": 212,
    "brier_skill_ppm": -2_100,
    "canary_status": "OK",
    "abstain_rate_ppm": 90_000,
    "cost_per_forecast": "n/a",
}


def test_render_provider_panel_shows_no_data_yet_placeholder_when_empty() -> None:
    """Zero rows renders a readable placeholder, never an error or an empty
    table (the empty-state invariant the `/providers` route also pins).
    """
    from windbreak.dashboard.views.providers import render_provider_panel

    html = render_provider_panel([])

    assert "no data yet" in html.lower()


def test_render_provider_panel_renders_provider_summary_fields() -> None:
    """A populated provider row renders its provider id, resolved count,
    Brier skill, canary status, abstention rate, and the permanent
    per-provider `cost_per_forecast=n/a` (issue #281 is NOT this issue).
    """
    from windbreak.dashboard.views.providers import render_provider_panel

    html = render_provider_panel([_POSITIVE_ROW])

    assert "openai" in html
    assert "212" in html
    assert "14200" in html
    assert "OK" in html
    assert "90000" in html or "90%" in html
    assert "n/a" in html


def test_render_provider_panel_negative_brier_skill_renders_verbatim() -> None:
    """A negative Brier skill renders its exact signed value, never suppressed
    or rounded away.
    """
    from windbreak.dashboard.views.providers import render_provider_panel

    html = render_provider_panel([_NEGATIVE_ROW])

    assert "-2100" in html


def test_render_provider_panel_negative_skill_as_prominent_as_positive() -> None:
    """The honesty invariant: a negative-skill row's markup has the exact same
    shape as a positive-skill row's -- only the provider name and the skill
    value itself differ. No conditional styling, dimming, or omission may be
    applied based on the sign of `brier_skill_ppm`.
    """
    from windbreak.dashboard.views.providers import render_provider_panel

    html_positive = render_provider_panel([_POSITIVE_ROW])
    html_negative = render_provider_panel([_NEGATIVE_ROW])

    # Normalize away an optional explicit "+" sign convention on the positive
    # value (renderer's choice) before comparing structure, so this test pins
    # "no suppression of negative values" without also constraining whether a
    # positive value is prefixed with "+".
    normalized_positive = html_positive.replace("+14200", "14200")
    normalized_negative = html_negative.replace("-2100", "14200").replace(
        "anthropic", "openai"
    )
    assert normalized_negative == normalized_positive


def test_render_provider_panel_escapes_a_hostile_provider_string() -> None:
    """A hostile (forged-HTML) provider identifier is rendered escaped, never
    raw -- provider identifiers are operator-supplied config today but must
    stay defensively escaped like every other dashboard-rendered string.
    """
    from windbreak.dashboard.views.providers import render_provider_panel

    row = {**_POSITIVE_ROW, "provider": "<script>alert(1)</script>"}

    html = render_provider_panel([row])

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_render_provider_panel_renders_fleet_cost_lines_when_present() -> None:
    """A `"fleet"`-kind row renders the fleet-wide cost-per-forecast and
    cost-per-resolved figures, distinct from the per-provider `n/a`.
    """
    from windbreak.dashboard.views.providers import render_provider_panel

    fleet_row = {
        "kind": "fleet",
        "cost_per_forecast_micros": 125_000,
        "cost_per_resolved_micros": 250_000,
    }

    html = render_provider_panel([_POSITIVE_ROW, fleet_row])

    assert "125000" in html
    assert "250000" in html


def test_render_provider_panel_fleet_costs_render_n_a_when_none() -> None:
    """A `"fleet"` row with unset (`None`) costs renders `n/a`, never `None`
    or a crash.
    """
    from windbreak.dashboard.views.providers import render_provider_panel

    fleet_row = {
        "kind": "fleet",
        "cost_per_forecast_micros": None,
        "cost_per_resolved_micros": None,
    }

    html = render_provider_panel([fleet_row])

    assert "None" not in html
    assert "n/a" in html


# --- DashboardReadModels.provider_panel: default-empty backward compat -------


def test_dashboard_read_models_provider_panel_defaults_to_empty_list() -> None:
    """`DashboardReadModels` gains `provider_panel` defaulting to `[]`, so
    every pre-#195 bare construction (`positions=[], equity_curve=[],
    decisions=[]`, no `provider_panel` kwarg) stays valid.
    """
    from windbreak.dashboard.views import DashboardReadModels

    read_models = DashboardReadModels(positions=[], equity_curve=[], decisions=[])

    assert read_models.provider_panel == []


def test_dashboard_read_models_provider_panel_accepts_explicit_rows_and_is_frozen() -> (
    None
):
    """`provider_panel` accepts an explicit row list at construction (not just
    the empty default) and, like every other field, cannot be reassigned
    afterward. Constructing with the keyword is itself the load-bearing
    assertion here: today it raises `TypeError` (`provider_panel` is not yet
    a recognized keyword), never merely a generic frozen-instance error a
    bare, field-less dataclass would already raise for any attribute name.
    """
    from windbreak.dashboard.views import DashboardReadModels

    seeded_row = {"kind": "provider", "provider": "openai"}
    read_models = DashboardReadModels(
        positions=[], equity_curve=[], decisions=[], provider_panel=[seeded_row]
    )

    assert read_models.provider_panel == [seeded_row]
    with pytest.raises(dataclasses.FrozenInstanceError):
        read_models.provider_panel = []  # type: ignore[misc]


# --- _provider_summary_rows / _compose_provider_panel: vote_cost_rows (#281) --
#
# `windbreak.dashboard.views.models._provider_summary_rows` and
# `_compose_provider_panel` do not yet accept a `vote_cost_rows` keyword, so
# every call below fails with `TypeError: _provider_summary_rows() got an
# unexpected keyword argument 'vote_cost_rows'` -- the expected Gate 1 RED
# state for issue #281. Once the keyword lands, a provider covered by the
# `provider_vote_costs_read_model` fold gets REAL ints for
# `abstain_rate_ppm`/`cost_per_forecast` (never the `"n/a"` placeholder), and
# an uncovered provider keeps `"n/a"` for both -- mirroring
# `_resolved_and_skill`'s own covered/uncovered #194 pattern exactly.

#: A minimal `canary_status.json`-shaped row for provider `"openai"`.
_OPENAI_CANARY_ROW = {
    "seq": 1,
    "created_at": "2024-01-01T00:00:00.000000+00:00",
    "event_type": "CanaryVerdictRecorded",
    "data": {"provider": "openai", "status": "OK"},
}

#: A minimal `canary_status.json`-shaped row for provider `"anthropic"`
#: (deliberately absent from the vote-cost fold in the "uncovered" tests).
_ANTHROPIC_CANARY_ROW = {
    "seq": 2,
    "created_at": "2024-01-01T00:00:01.000000+00:00",
    "event_type": "CanaryVerdictRecorded",
    "data": {"provider": "anthropic", "status": "OK"},
}

#: A `provider_vote_costs.json`-shaped aggregate row covering `"openai"`.
_OPENAI_VOTE_COST_ROW = {
    "provider": "openai",
    "cost_micros_total": 1_000,
    "vote_count": 2,
    "abstain_count": 1,
    "forecast_count": 1,
    "cost_per_forecast_micros": 1_000,
    "abstain_rate_ppm": 500_000,
}


def test_provider_summary_rows_covered_provider_gets_real_vote_cost_ints() -> None:
    """A provider present in the `vote_cost_rows` fold gets REAL ints for
    `cost_per_forecast`/`abstain_rate_ppm`, copied verbatim off the fold's own
    `cost_per_forecast_micros`/`abstain_rate_ppm` -- never `"n/a"`.
    """
    from windbreak.dashboard.views.models import _provider_summary_rows

    rows = _provider_summary_rows(
        [_OPENAI_CANARY_ROW], vote_cost_rows=[_OPENAI_VOTE_COST_ROW]
    )

    assert len(rows) == 1
    assert rows[0]["provider"] == "openai"
    assert rows[0]["cost_per_forecast"] == 1_000
    assert rows[0]["abstain_rate_ppm"] == 500_000
    assert type(rows[0]["cost_per_forecast"]) is int
    assert type(rows[0]["abstain_rate_ppm"]) is int


def test_provider_summary_rows_uncovered_provider_stays_n_a() -> None:
    """A provider present in canary status but ABSENT from the vote-cost fold
    keeps the `"n/a"` placeholder for both figures -- old-ledger tolerance,
    never a fabricated `0`.
    """
    from windbreak.dashboard.views.models import _provider_summary_rows

    rows = _provider_summary_rows(
        [_ANTHROPIC_CANARY_ROW], vote_cost_rows=[_OPENAI_VOTE_COST_ROW]
    )

    assert len(rows) == 1
    assert rows[0]["provider"] == "anthropic"
    assert rows[0]["cost_per_forecast"] == "n/a"
    assert rows[0]["abstain_rate_ppm"] == "n/a"


def test_provider_summary_rows_default_vote_cost_rows_stays_n_a() -> None:
    """With no `vote_cost_rows` supplied at all, every provider stays `"n/a"`
    for both figures -- the pre-#281 behavior, unchanged (backward compat for
    any caller that has not yet threaded the new fold through).
    """
    from windbreak.dashboard.views.models import _provider_summary_rows

    rows = _provider_summary_rows([_OPENAI_CANARY_ROW])

    assert rows[0]["cost_per_forecast"] == "n/a"
    assert rows[0]["abstain_rate_ppm"] == "n/a"


def test_compose_provider_panel_threads_vote_cost_rows_into_provider_summary() -> None:
    """`_compose_provider_panel` gains a `vote_cost_rows` keyword and threads
    it straight through to `_provider_summary_rows`, so a covered provider's
    panel row carries the real fold ints end-to-end from the two source
    projections.
    """
    from windbreak.dashboard.views.models import _compose_provider_panel

    panel = _compose_provider_panel(
        [_OPENAI_CANARY_ROW], [], vote_cost_rows=[_OPENAI_VOTE_COST_ROW]
    )

    provider_row = next(row for row in panel if row["kind"] == "provider")
    assert provider_row["cost_per_forecast"] == 1_000
    assert provider_row["abstain_rate_ppm"] == 500_000


# --- /providers route: bearer-gated, "no data yet" empty state --------------


@pytest.fixture
def dashboard_server_with_provider_panel() -> Iterator[
    tuple[http.server.ThreadingHTTPServer, tuple[str, int]]
]:
    """Start a dashboard server wired with a fixed, non-empty provider panel."""
    from windbreak.dashboard.app import DashboardStatus, create_server
    from windbreak.dashboard.views import DashboardReadModels

    def _source() -> DashboardReadModels:
        return DashboardReadModels(
            positions=[],
            equity_curve=[],
            decisions=[],
            provider_panel=[_POSITIVE_ROW],
        )

    server = create_server(
        token=TEST_TOKEN,
        status_source=lambda: DashboardStatus(mode="PAPER", last_heartbeat=None),
        read_models_source=_source,
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


@pytest.fixture
def dashboard_server_with_zero_providers_configured() -> Iterator[
    tuple[http.server.ThreadingHTTPServer, tuple[str, int]]
]:
    """Start a dashboard server whose read-models source is wired but returns
    a genuinely EMPTY provider panel -- "zero providers configured", never an
    unwired `read_models_source=None` default.
    """
    from windbreak.dashboard.app import DashboardStatus, create_server
    from windbreak.dashboard.views import DashboardReadModels

    def _source() -> DashboardReadModels:
        return DashboardReadModels(
            positions=[], equity_curve=[], decisions=[], provider_panel=[]
        )

    server = create_server(
        token=TEST_TOKEN,
        status_source=lambda: DashboardStatus(mode="PAPER", last_heartbeat=None),
        read_models_source=_source,
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


def test_providers_route_requires_bearer_auth(
    dashboard_server_with_provider_panel: tuple[
        http.server.ThreadingHTTPServer, tuple[str, int]
    ],
) -> None:
    """`/providers` is gated behind the same bearer auth as every other route."""
    _server, address = dashboard_server_with_provider_panel

    status, headers, _body = _get(address, "/providers")

    assert status == 401
    assert "WWW-Authenticate" in headers


def test_providers_route_renders_the_seeded_provider(
    dashboard_server_with_provider_panel: tuple[
        http.server.ThreadingHTTPServer, tuple[str, int]
    ],
) -> None:
    """With a correct bearer token, `/providers` returns 200 and renders the
    seeded provider's summary line.
    """
    _server, address = dashboard_server_with_provider_panel

    status, _headers, body = _get(address, "/providers", headers=_bearer(TEST_TOKEN))

    assert status == 200
    assert "openai" in body
    assert "14200" in body


def test_providers_route_returns_200_no_data_yet_with_zero_providers_configured(
    dashboard_server_with_zero_providers_configured: tuple[
        http.server.ThreadingHTTPServer, tuple[str, int]
    ],
) -> None:
    """ZERO providers configured is still a 200 with a "No data yet." panel --
    never an error page or an empty table (the empty-state invariant).
    """
    _server, address = dashboard_server_with_zero_providers_configured

    status, _headers, body = _get(address, "/providers", headers=_bearer(TEST_TOKEN))

    assert status == 200
    assert "no data yet" in body.lower()


def test_providers_route_returns_200_no_data_yet_when_read_models_source_unwired() -> (
    None
):
    """With NO `read_models_source` wired at all (the default `None`),
    `/providers` still 200s with the "no data yet" placeholder, mirroring
    every other PAPER-loop view's unwired-seam contract.
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
            server.server_address, "/providers", headers=_bearer(TEST_TOKEN)
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 200
    assert "no data yet" in body.lower()


# --- build_ledger_read_models_source: track_record_path fold (issue #195) ---


def _seed_provider_canary_ledger(tmp_path: Path, providers: list[str]) -> Path:
    """Seed a fresh SQLite ledger with one clean `CanaryVerdictRecorded` per
    provider, mirroring `tests/ledger/test_canary_rebuild.py`'s own
    `_sample_verdict_kwargs` shape, so `canary_status_read_model` yields one
    provider-panel row per provider for the `track_record_path` fold tests
    below.

    Args:
        tmp_path: The test's isolated tmp directory.
        providers: The provider identifiers to seed, in ledger order.

    Returns:
        The path to the seeded SQLite ledger database.
    """
    from windbreak.ledger.events import CanaryVerdictRecorded
    from windbreak.ledger.store import SqliteLedgerStore

    ledger_path = tmp_path / "ledger.db"
    store = SqliteLedgerStore(ledger_path)
    for provider in providers:
        store.append(
            CanaryVerdictRecorded(
                component="scheduler",
                provider=provider,
                status="OK",
                drift_kind="",
                drift_score_ppm=0,
                tolerance_ppm=50_000,
                reported_version="v1",
                pinned_versions=["v1"],
            )
        )
    store.close()
    return ledger_path


def _write_track_record_artifact(
    tmp_path: Path,
    records: dict[str, dict[str, object]],
    *,
    name: str = "track_record.json",
) -> Path:
    """Write a track-record artifact JSON file in the shape
    `windbreak.forecast.providers.track_record.parse_track_records` accepts.

    Args:
        tmp_path: The test's isolated tmp directory.
        records: The provider -> `{resolved_count, brier_skill_ppm}` mapping.
        name: The artifact's filename.

    Returns:
        The path to the written artifact file.
    """
    artifact_path = tmp_path / name
    artifact_path.write_text(json.dumps(records))
    return artifact_path


def _provider_rows_by_name(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Index a provider panel's `"provider"`-kind rows by provider identity.

    Args:
        rows: The composed `provider_panel` rows (provider and fleet kinds
            mixed together).

    Returns:
        The `"provider"`-kind rows keyed by their `provider` field.
    """
    by_provider: dict[str, dict[str, object]] = {}
    for row in rows:
        if row.get("kind") == "provider":
            by_provider[str(row["provider"])] = row
    return by_provider


def test_build_ledger_read_models_source_track_record_shows_real_skill_and_resolved(
    tmp_path: Path,
) -> None:
    """A `track_record_path` artifact covering canary-status providers backs
    the composed panel with the REAL `brier_skill_ppm`/`resolved` from #194's
    read model -- including a NEGATIVE skill, rendered verbatim, never
    suppressed or coerced to `"n/a"` (the honesty invariant).
    """
    from windbreak.dashboard.views import build_ledger_read_models_source

    ledger_path = _seed_provider_canary_ledger(tmp_path, ["openai", "anthropic"])
    track_record_path = _write_track_record_artifact(
        tmp_path,
        {
            "openai": {"resolved_count": 212, "brier_skill_ppm": 14_200},
            "anthropic": {"resolved_count": 180, "brier_skill_ppm": -2_100},
        },
    )

    source = build_ledger_read_models_source(
        ledger_path, track_record_path=track_record_path
    )
    read_models = source()

    rows_by_provider = _provider_rows_by_name(read_models.provider_panel)
    assert rows_by_provider["openai"]["resolved"] == 212
    assert rows_by_provider["openai"]["brier_skill_ppm"] == 14_200
    assert rows_by_provider["anthropic"]["resolved"] == 180
    assert rows_by_provider["anthropic"]["brier_skill_ppm"] == -2_100


def test_build_ledger_read_models_source_track_record_path_missing_provider_stays_n_a(
    tmp_path: Path,
) -> None:
    """A provider present in canary status but ABSENT from the track-record
    artifact stays `"n/a"` for `resolved`/`brier_skill_ppm` -- no fabricated
    skill for a provider #194's read model has never measured -- while a
    covered provider in the same panel gets its real figures.
    """
    from windbreak.dashboard.views import build_ledger_read_models_source

    ledger_path = _seed_provider_canary_ledger(tmp_path, ["openai", "futuresearch"])
    track_record_path = _write_track_record_artifact(
        tmp_path, {"openai": {"resolved_count": 212, "brier_skill_ppm": 14_200}}
    )

    source = build_ledger_read_models_source(
        ledger_path, track_record_path=track_record_path
    )
    read_models = source()

    rows_by_provider = _provider_rows_by_name(read_models.provider_panel)
    assert rows_by_provider["futuresearch"]["resolved"] == "n/a"
    assert rows_by_provider["futuresearch"]["brier_skill_ppm"] == "n/a"
    assert rows_by_provider["openai"]["resolved"] == 212
    assert rows_by_provider["openai"]["brier_skill_ppm"] == 14_200


def test_build_ledger_read_models_source_default_track_record_path_stays_n_a(
    tmp_path: Path,
) -> None:
    """Backward compat: with NO `track_record_path` supplied (the default),
    per-provider `resolved`/`brier_skill_ppm` stay `"n/a"` -- the pre-existing
    behavior, unchanged. This call carries no new keyword, so it is not
    expected to fail alongside the tests above.
    """
    from windbreak.dashboard.views import build_ledger_read_models_source

    ledger_path = _seed_provider_canary_ledger(tmp_path, ["openai"])

    source = build_ledger_read_models_source(ledger_path)
    read_models = source()

    rows_by_provider = _provider_rows_by_name(read_models.provider_panel)
    assert rows_by_provider["openai"]["resolved"] == "n/a"
    assert rows_by_provider["openai"]["brier_skill_ppm"] == "n/a"


def test_build_ledger_read_models_source_malformed_track_record_raises_value_error(
    tmp_path: Path,
) -> None:
    """A malformed track-record artifact (a JSON float leaf where an integer
    Brier skill is required) fails closed with the same `ValueError`
    `parse_track_records` itself raises -- never silently truncated, ignored,
    or rendered as a plausible-but-wrong skill.
    """
    from windbreak.dashboard.views import build_ledger_read_models_source

    ledger_path = _seed_provider_canary_ledger(tmp_path, ["openai"])
    malformed_path = _write_track_record_artifact(
        tmp_path,
        {"openai": {"resolved_count": 212, "brier_skill_ppm": 14_200.5}},
        name="malformed_track_record.json",
    )

    source = build_ledger_read_models_source(
        ledger_path, track_record_path=malformed_path
    )

    with pytest.raises(ValueError, match="float"):
        source()


# --- build_ledger_read_models_source: real vote-cost figures (issue #281) ----
#
# `windbreak.ledger.events.ProviderVoteRecorded` does not exist yet, so
# `_seed_provider_vote_costs` below fails at call time with `ImportError:
# cannot import name 'ProviderVoteRecorded' from 'windbreak.ledger.events'`
# -- the expected Gate 1 RED state for issue #281.


def _seed_provider_vote_costs(
    ledger_path: Path, *, provider: str, forecast_id: str, cost_micros: int
) -> None:
    """Append one clean, non-abstaining `ProviderVoteRecorded` row to an
    already-seeded ledger, so `provider_vote_costs_read_model` folds a real
    aggregate for `provider`.

    Args:
        ledger_path: The already-seeded SQLite ledger path.
        provider: The provider identifier the vote is stamped for.
        forecast_id: The forecast this vote belongs to.
        cost_micros: The vote's billed cost, in micros.
    """
    from windbreak.ledger.events import ProviderVoteRecorded
    from windbreak.ledger.store import SqliteLedgerStore

    store = SqliteLedgerStore(ledger_path)
    store.append(
        ProviderVoteRecorded(
            component="scheduler",
            forecast_id=forecast_id,
            market_ticker="MKT-DEEP",
            provider=provider,
            model_version="gpt-5-2025-08-07",
            vote_index=0,
            cost_micros=cost_micros,
            outcome="voted",
            failure_code="",
        )
    )
    store.close()


def test_build_ledger_source_shows_real_cost_and_abstain_for_covered(
    tmp_path: Path,
) -> None:
    """A provider covered by the ledger's `ProviderVoteRecorded` rows gets
    REAL `cost_per_forecast`/`abstain_rate_ppm` ints in the composed dashboard
    panel -- never the `"n/a"` placeholder -- proving the vote-cost fold is
    actually wired into `build_ledger_read_models_source`, not just the
    `_compose_provider_panel` unit-level contract above.
    """
    from windbreak.dashboard.views import build_ledger_read_models_source

    ledger_path = _seed_provider_canary_ledger(tmp_path, ["openai"])
    _seed_provider_vote_costs(
        ledger_path, provider="openai", forecast_id="fc-0001", cost_micros=1_500
    )

    source = build_ledger_read_models_source(ledger_path)
    read_models = source()

    rows_by_provider = _provider_rows_by_name(read_models.provider_panel)
    assert rows_by_provider["openai"]["cost_per_forecast"] == 1_500
    assert rows_by_provider["openai"]["abstain_rate_ppm"] == 0
    assert type(rows_by_provider["openai"]["cost_per_forecast"]) is int


def test_build_ledger_read_models_source_uncovered_provider_still_shows_n_a(
    tmp_path: Path,
) -> None:
    """A provider present in canary status but with ZERO `ProviderVoteRecorded`
    rows keeps the `"n/a"` placeholder for both figures -- old-ledger
    tolerance, never a fabricated `0` -- while a covered provider in the same
    panel gets its real figures.
    """
    from windbreak.dashboard.views import build_ledger_read_models_source

    ledger_path = _seed_provider_canary_ledger(tmp_path, ["openai", "anthropic"])
    _seed_provider_vote_costs(
        ledger_path, provider="openai", forecast_id="fc-0001", cost_micros=1_500
    )

    source = build_ledger_read_models_source(ledger_path)
    read_models = source()

    rows_by_provider = _provider_rows_by_name(read_models.provider_panel)
    assert rows_by_provider["anthropic"]["cost_per_forecast"] == "n/a"
    assert rows_by_provider["anthropic"]["abstain_rate_ppm"] == "n/a"
    assert rows_by_provider["openai"]["cost_per_forecast"] == 1_500
