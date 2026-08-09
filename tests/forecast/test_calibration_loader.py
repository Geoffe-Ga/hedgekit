"""Tests for windbreak.forecast.calibration (issue #194): versioned,
temporally-safe calibration maps (SPEC S8... calibration/temporal-integrity
seam).

Pins `CalibrationMap`'s deterministic integer piecewise-linear interpolation
(floor-division rounding, endpoint clamping) and its `__post_init__`
validation, `ensure_temporal_integrity`'s "no calibration map trained on the
future" guard -- the issue's own worked example: a map "trained" 2026-09-01
must never calibrate a forecast created 2026-07-01 -- and
`load_calibration_map`'s single entry point wiring construction and the
integrity check together. Also pins `run_pipeline`'s new `calibration_map`
seam: a wired fitted map's exact application is ledgered as
`CALIBRATION_MAP_APPLIED`, a future-dated map fails the whole run closed, and
`calibration_map=None` (the default) stays a byte-identical, zero-event
no-op. `windbreak/forecast/calibration.py` does not exist yet, so importing
`windbreak.forecast.calibration` fails collection with
`ModuleNotFoundError: No module named 'windbreak.forecast.calibration'` --
the expected Gate 1 RED state for issue #194.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.calibration import (
    IDENTITY_CALIBRATION_MAP,
    IDENTITY_MAP_ID,
    IDENTITY_MAP_VERSION,
    CalibrationMap,
    TemporalIntegrityError,
    ensure_temporal_integrity,
    load_calibration_map,
)
from windbreak.forecast.pipeline import (
    CALIBRATION_MAP_APPLIED_EVENT,
    InMemoryForecastLedger,
    aggregate_median,
    apply_calibration_map,
    collect_model_votes,
    run_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

    #: See `tests/forecast/conftest.py`'s "Sandbox-transport fixture choice"
    #: note for why `make_fake_vote_transport` is typed structurally here.
    FakeVoteTransportFactory = Callable[..., object]

#: Issue #194's own worked example: a calibration map "trained" on data
#: through this date...
_FUTURE_MAP_VERSION = "2026-09-01"

#: ...must never calibrate a forecast created on this earlier date.
JULY_1 = datetime(2026, 7, 1, tzinfo=UTC)

#: A calibration map "trained" the same day as a forecast's creation date --
#: the equal-date boundary, which `ensure_temporal_integrity` must allow.
_SAME_DAY_VERSION = "2026-07-01"

#: A later time on `_SAME_DAY_VERSION`'s own calendar date, proving the
#: equal-date check compares dates, not full timestamps.
_SAME_DAY_LATER = datetime(2026, 7, 1, 23, 59, 59, tzinfo=UTC)

#: One calendar day strictly after `JULY_1`'s date -- the boundary
#: `ensure_temporal_integrity` must reject.
_NEXT_DAY_VERSION = "2026-07-02"

#: A version string that is not a valid ISO-8601 date at all.
_MALFORMED_VERSION = "not-a-date"

#: A non-`"v0"` map id/version pair used only for the interpolation and
#: `__post_init__` validation tests below (no temporal-integrity check runs
#: in those tests).
_FITTED_MAP_ID = "fitted-v1"
_FITTED_MAP_VERSION = "2024-12-01"

#: Two-breakpoint fitted map spanning only the interior of the ppm domain
#: (200_000..800_000), so both the below-first-breakpoint and
#: above-last-breakpoint clamp paths are independently observable.
_INTERIOR_ENTRIES: tuple[tuple[int, int], ...] = (
    (200_000, 100_000),
    (800_000, 900_000),
)

#: The fixture pipeline's baseline/`created_at` are both dated 2024-12-10
#: (see `tests/forecast/conftest.py`); a map "trained" the same day is
#: therefore integrity-safe to wire into a full pipeline run.
_PIPELINE_SAFE_MAP_VERSION = "2024-12-10"

#: One calendar day after the pipeline fixtures' `created_at` -- integrity-
#: unsafe, and must make `run_pipeline` raise closed.
_PIPELINE_FUTURE_MAP_VERSION = "2024-12-11"


def _fitted_map(
    *, version: str = _FITTED_MAP_VERSION, map_id: str = _FITTED_MAP_ID
) -> CalibrationMap:
    """Build the shared interior-breakpoint fitted map for these tests.

    Args:
        version: The map's version string; defaults to a fixed past date.
        map_id: The map's identifier; defaults to a fixed id.

    Returns:
        A `CalibrationMap` over `_INTERIOR_ENTRIES`.
    """
    return CalibrationMap(map_id=map_id, version=version, entries=_INTERIOR_ENTRIES)


# --- CalibrationMap.apply: deterministic piecewise-linear interpolation -----------


def test_fitted_map_apply_clamps_below_first_breakpoint_to_its_output() -> None:
    """A probability below the lowest breakpoint clamps to that breakpoint's
    output, never extrapolating past it."""
    assert _fitted_map().apply(0) == 100_000


def test_fitted_map_apply_clamps_above_last_breakpoint_to_its_output() -> None:
    """A probability above the highest breakpoint clamps to that
    breakpoint's output."""
    assert _fitted_map().apply(1_000_000) == 900_000


def test_fitted_map_apply_is_exact_at_breakpoints() -> None:
    """Applying the map at an exact breakpoint input returns its output
    unchanged (no interpolation drift at the pinned points)."""
    fitted = _fitted_map()

    assert fitted.apply(200_000) == 100_000
    assert fitted.apply(800_000) == 900_000


def test_fitted_map_apply_interpolates_with_floor_rounding() -> None:
    """450_000 sits between the two breakpoints; the true fractional output
    is 433_333.33..., pinned to FLOOR (433_333), never CEIL (433_334)."""
    assert _fitted_map().apply(450_000) == 433_333


def test_fitted_map_apply_is_identity_when_entries_empty() -> None:
    """A map with no entries at all (the identity-map shape) leaves an
    in-range probability unchanged."""
    empty_map = CalibrationMap(map_id=_FITTED_MAP_ID, version=_FITTED_MAP_VERSION)

    assert empty_map.apply(450_000) == 450_000


# --- CalibrationMap.__post_init__ validation ---------------------------------------


def test_calibration_map_rejects_empty_map_id() -> None:
    """A blank `map_id` is never a valid calibration map identity."""
    with pytest.raises(ValueError, match="map_id"):
        CalibrationMap(map_id="", version=_FITTED_MAP_VERSION)


def test_calibration_map_rejects_empty_version() -> None:
    """A blank `version` is never a valid calibration map identity."""
    with pytest.raises(ValueError, match="version"):
        CalibrationMap(map_id=_FITTED_MAP_ID, version="")


def test_calibration_map_rejects_non_increasing_breakpoint_inputs() -> None:
    """A descending breakpoint sequence violates the strictly-increasing-
    inputs invariant."""
    with pytest.raises(ValueError):
        CalibrationMap(
            map_id=_FITTED_MAP_ID,
            version=_FITTED_MAP_VERSION,
            entries=((300_000, 100_000), (200_000, 200_000)),
        )


def test_calibration_map_rejects_duplicate_breakpoint_inputs() -> None:
    """Two breakpoints sharing the same input is not strictly increasing."""
    with pytest.raises(ValueError):
        CalibrationMap(
            map_id=_FITTED_MAP_ID,
            version=_FITTED_MAP_VERSION,
            entries=((200_000, 100_000), (200_000, 200_000)),
        )


@pytest.mark.parametrize(
    "entries",
    [
        ((-1, 100_000),),
        ((1_000_001, 100_000),),
        ((200_000, -1),),
        ((200_000, 1_000_001),),
    ],
    ids=["negative-input", "over-max-input", "negative-output", "over-max-output"],
)
def test_calibration_map_rejects_out_of_range_breakpoint_values(
    entries: tuple[tuple[int, int], ...],
) -> None:
    """Every breakpoint value (input or output) must lie within
    `[0, 1_000_000]`."""
    with pytest.raises(ValueError):
        CalibrationMap(
            map_id=_FITTED_MAP_ID, version=_FITTED_MAP_VERSION, entries=entries
        )


# --- Identity map -------------------------------------------------------------------


def test_identity_calibration_map_has_expected_id_version_and_entries() -> None:
    """The module-level identity map singleton carries the pinned id,
    version, and empty entry set."""
    assert IDENTITY_CALIBRATION_MAP.map_id == IDENTITY_MAP_ID
    assert IDENTITY_CALIBRATION_MAP.version == IDENTITY_MAP_VERSION
    assert IDENTITY_CALIBRATION_MAP.entries == ()


def test_identity_calibration_map_apply_is_identity_within_domain() -> None:
    """The identity map returns any in-domain probability unchanged,
    including at both domain boundaries."""
    assert IDENTITY_CALIBRATION_MAP.apply(0) == 0
    assert IDENTITY_CALIBRATION_MAP.apply(1_000_000) == 1_000_000
    assert IDENTITY_CALIBRATION_MAP.apply(450_000) == 450_000


def test_identity_calibration_map_apply_clamps_out_of_domain() -> None:
    """The identity map defensively clamps an out-of-domain input back into
    `[0, 1_000_000]` rather than returning it unchanged."""
    assert IDENTITY_CALIBRATION_MAP.apply(-1) == 0
    assert IDENTITY_CALIBRATION_MAP.apply(1_000_001) == 1_000_000


# --- ensure_temporal_integrity: the "no future-trained map" guard -----------------


def test_ensure_temporal_integrity_allows_equal_date() -> None:
    """A map version dated the same calendar day as the forecast's creation
    (but an earlier time) is allowed -- the date comparison is date-only,
    not full-timestamp."""
    equal_day_map = CalibrationMap(map_id=_FITTED_MAP_ID, version=_SAME_DAY_VERSION)

    ensure_temporal_integrity(equal_day_map, forecast_created_at=_SAME_DAY_LATER)


def test_ensure_temporal_integrity_rejects_strictly_future_date() -> None:
    """A map version dated even one calendar day after the forecast's
    creation date is rejected."""
    tomorrow_map = CalibrationMap(map_id=_FITTED_MAP_ID, version=_NEXT_DAY_VERSION)

    with pytest.raises(TemporalIntegrityError):
        ensure_temporal_integrity(tomorrow_map, forecast_created_at=JULY_1)


def test_ensure_temporal_integrity_malformed_version_raises_plain_value_error() -> None:
    """A version string that is not a valid ISO-8601 date raises a plain
    `ValueError` (from `date.fromisoformat`), never a `TemporalIntegrityError`
    -- the two failure modes must stay distinguishable."""
    bad_map = CalibrationMap(map_id=_FITTED_MAP_ID, version=_MALFORMED_VERSION)

    with pytest.raises(ValueError, match=_MALFORMED_VERSION) as excinfo:
        ensure_temporal_integrity(bad_map, forecast_created_at=JULY_1)

    assert not isinstance(excinfo.value, TemporalIntegrityError)


def test_ensure_temporal_integrity_v0_sentinel_always_passes() -> None:
    """The `"v0"` version sentinel always passes, regardless of how far its
    "date" would otherwise diverge from the forecast's creation date."""
    absurdly_early = datetime(1900, 1, 1, tzinfo=UTC)

    ensure_temporal_integrity(
        IDENTITY_CALIBRATION_MAP, forecast_created_at=absurdly_early
    )


# --- load_calibration_map: the single construct-and-validate entry point ----------


def test_load_calibration_map_rejects_future_dated_version_issue_example() -> None:
    """Issue #194's own worked example, verbatim: a map "trained" 2026-09-01
    must never calibrate a forecast created 2026-07-01."""
    with pytest.raises(TemporalIntegrityError):
        load_calibration_map(version=_FUTURE_MAP_VERSION, forecast_created_at=JULY_1)


def test_load_calibration_map_default_version_v0_returns_identity_equivalent() -> None:
    """Loading with the default `map_id`/`entries` and the `"v0"` sentinel
    version yields an identity-equivalent map, regardless of `created_at`."""
    result = load_calibration_map(
        version=IDENTITY_MAP_VERSION, forecast_created_at=JULY_1
    )

    assert result.map_id == IDENTITY_MAP_ID
    assert result.version == IDENTITY_MAP_VERSION
    assert result.entries == ()
    assert result.apply(450_000) == 450_000


def test_load_calibration_map_returns_fitted_map_with_supplied_entries() -> None:
    """A caller-supplied `map_id`/`entries` pair, with a safely-past version,
    constructs and passes integrity, carrying those exact fields through."""
    result = load_calibration_map(
        version=_FITTED_MAP_VERSION,
        forecast_created_at=JULY_1,
        map_id=_FITTED_MAP_ID,
        entries=_INTERIOR_ENTRIES,
    )

    assert result.map_id == _FITTED_MAP_ID
    assert result.version == _FITTED_MAP_VERSION
    assert result.entries == _INTERIOR_ENTRIES


# --- pipeline.apply_calibration_map: the two-arg stage function -------------------


def test_apply_calibration_map_stage_function_none_preserves_identity_clamp() -> None:
    """Passing `calibration_map=None` (the default) preserves the pre-#194
    identity-clamp behavior byte-for-byte."""
    assert apply_calibration_map(450_000, None) == 450_000
    assert apply_calibration_map(1_000_001, None) == 1_000_000


def test_apply_calibration_map_stage_function_delegates_to_wired_map() -> None:
    """Passing a wired `CalibrationMap` delegates to its own `.apply`."""
    fitted = _fitted_map()

    assert apply_calibration_map(450_000, fitted) == fitted.apply(450_000)


# --- run_pipeline integration: the calibration_map seam ----------------------------


def test_run_pipeline_wires_calibration_map_and_ledgers_applied_event(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """A wired, integrity-safe fitted map is applied to the aggregate median
    and ledgered as one `CALIBRATION_MAP_APPLIED` event whose payload leaves
    are the exact int/str values `CalibrationMap.apply` itself would produce
    -- pinning the wiring, not re-deriving the interpolation math by hand.
    """
    fitted = _fitted_map(version=_PIPELINE_SAFE_MAP_VERSION)
    ledger = InMemoryForecastLedger()
    votes = collect_model_votes(market, baseline, transport=make_fake_vote_transport())
    expected_input_ppm = aggregate_median(votes).probability_ppm
    expected_output_ppm = fitted.apply(expected_input_ppm)

    run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        calibration_map=fitted,
        ledger=ledger,
    )

    events = ledger.events_by_type(CALIBRATION_MAP_APPLIED_EVENT)
    assert len(events) == 1
    assert events[0].payload == {
        "map_id": _FITTED_MAP_ID,
        "map_version": _PIPELINE_SAFE_MAP_VERSION,
        "input_ppm": expected_input_ppm,
        "output_ppm": expected_output_ppm,
    }
    # Sanity: this fitted map is genuinely non-identity over the canned
    # votes' median, so the assertion above is not vacuously true.
    assert expected_output_ppm != expected_input_ppm


def test_run_pipeline_future_dated_calibration_map_raises_temporal_integrity_error(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """Wiring a calibration map "trained" after the forecast's own creation
    date fails the whole run closed."""
    future_map = _fitted_map(version=_PIPELINE_FUTURE_MAP_VERSION)

    with pytest.raises(TemporalIntegrityError):
        run_pipeline(
            market,
            baseline,
            transport=make_fake_vote_transport(),
            created_at=created_at,
            research_tools=research_tools,
            calibration_map=future_map,
        )


def test_run_pipeline_calibration_map_none_records_zero_calibration_events(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """`calibration_map=None` (the default), with a ledger wired, records
    zero `CALIBRATION_MAP_APPLIED` events -- a strict no-op."""
    ledger = InMemoryForecastLedger()

    run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
    )

    assert ledger.events_by_type(CALIBRATION_MAP_APPLIED_EVENT) == ()
