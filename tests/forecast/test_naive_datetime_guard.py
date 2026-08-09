"""Naive datetimes are refused at the boundary, not read as host-local (#397).

``datetime.astimezone()`` does not raise on a naive value: it reinterprets the
wall clock as the **host's** local time. Three sites let that misreading change
a *decision* rather than merely a log string:

1. ``calibration.ensure_temporal_integrity`` -- the look-ahead-bias gate.
2. ``budget.ResearchBudget`` -- the daily spend bucket, keyed by calendar day.
3. ``canary.CanaryGate`` -- the live-eligibility drift window's ordering.

Every test here **pins the process timezone**. CI runs UTC, where the naive and
correct paths agree exactly, so an unpinned value assertion is a permanent
false green. Both directions are pinned deliberately, because they are not
symmetric:

- **West of UTC** (``local_timezone_utc_minus_5``): a naive late-evening wall
  clock reads as the *next* UTC day, pushing ``created_on`` later than the true
  creation date, so a map trained the following day slips past
  ``trained_on > created_on``. The gate **fails open** -- future information
  enters a backtest silently.
- **East of UTC** (``local_timezone_utc_plus_5``): a naive early-morning wall
  clock reads as the *previous* UTC day, so a legitimately same-day-trained map
  is rejected. The gate **fails closed** -- a false alarm, not a breach.

Pinning only the eastward direction would mistake the safe failure for
correctness. The fix must refuse the unprovable instant on *either* host.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.budget import InMemoryBudgetLedger, ResearchBudget
from windbreak.forecast.calibration import (
    CalibrationMap,
    TemporalIntegrityError,
    ensure_temporal_integrity,
    load_calibration_map,
)
from windbreak.forecast.canary import (
    CanaryGate,
    CanaryRunResult,
    InMemoryCanaryLedger,
)

if TYPE_CHECKING:
    from windbreak.alerts import AlertType

#: A naive late-evening wall clock. Its true calendar day is 2024-12-10, but a
#: UTC-05:00 host reads it as 2024-12-11 -- one day *later*.
NAIVE_EVENING = datetime(2024, 12, 10, 23, 30)

#: A naive early-morning wall clock. Its true calendar day is 2024-12-10, but a
#: UTC+05:00 host reads it as 2024-12-09 -- one day *earlier*.
NAIVE_MORNING = datetime(2024, 12, 10, 1, 30)

_AWARE = "must be timezone-aware"


class RecordingAlertEmitter:
    """A canary alert emitter that records every dispatch."""

    def __init__(self) -> None:
        """Start with no recorded dispatches."""
        self.dispatched: list[tuple[AlertType, str]] = []

    def dispatch(self, alert_type: AlertType, message: str) -> object:
        """Record one dispatched alert.

        Args:
            alert_type: The alert type dispatched.
            message: The alert body.

        Returns:
            ``None``; the seam's return value is never inspected.
        """
        self.dispatched.append((alert_type, message))
        return None


# --- Site 1: the look-ahead-bias gate ----------------------------------------


def test_look_ahead_gate_refuses_a_naive_instant_west_of_utc(
    local_timezone_utc_minus_5: None,
) -> None:
    """A naive creation instant is refused rather than read as host-local.

    This is the fail-**open** direction. Before the guard, ``created_on`` read
    as 2024-12-11 on this host, so a map trained 2024-12-11 cleared
    ``trained_on > created_on`` and was admitted -- future information into a
    backtest, silently.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    trained_next_day = CalibrationMap("m1", "2024-12-11")

    with pytest.raises(ValueError, match=_AWARE) as excinfo:
        ensure_temporal_integrity(trained_next_day, forecast_created_at=NAIVE_EVENING)

    assert type(excinfo.value) is ValueError
    assert "forecast_created_at" in str(excinfo.value)


def test_the_instant_the_west_skew_admitted_is_a_real_look_ahead_breach(
    local_timezone_utc_minus_5: None,
) -> None:
    """The very map the naive west path admitted is a genuine integrity breach.

    Declaring the *same* wall clock aware makes the gate reject it as
    look-ahead bias. That contrast is the whole defect: the map's admissibility
    depended on whether the caller happened to attach a timezone, not on any
    fact about the forecast.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    trained_next_day = CalibrationMap("m1", "2024-12-11")

    with pytest.raises(TemporalIntegrityError, match="2024-12-11"):
        ensure_temporal_integrity(
            trained_next_day, forecast_created_at=NAIVE_EVENING.replace(tzinfo=UTC)
        )


def test_look_ahead_gate_refuses_a_naive_instant_east_of_utc(
    local_timezone_utc_plus_5: None,
) -> None:
    """A naive creation instant is refused east of UTC too.

    This is the fail-**closed** direction: before the guard this raised
    ``TemporalIntegrityError`` -- a *false alarm* on a perfectly legitimate
    same-day-trained map, because ``created_on`` read as 2024-12-09. The
    assertion below discriminates on both type and message, because
    ``TemporalIntegrityError`` subclasses ``ValueError`` and would otherwise
    satisfy a bare ``pytest.raises(ValueError)`` before the fix landed.

    Args:
        local_timezone_utc_plus_5: Pins the process timezone east of UTC.
    """
    trained_same_day = CalibrationMap("m1", "2024-12-10")

    with pytest.raises(ValueError, match=_AWARE) as excinfo:
        ensure_temporal_integrity(trained_same_day, forecast_created_at=NAIVE_MORNING)

    assert type(excinfo.value) is ValueError
    assert not isinstance(excinfo.value, TemporalIntegrityError)


def test_a_same_day_trained_map_is_admitted_when_the_instant_is_aware(
    local_timezone_utc_plus_5: None,
) -> None:
    """An aware instant admits the same-day map the east skew wrongly rejected.

    A same-day-trained map is integrity-safe by design (the comparison is
    strict). Pinning this proves the eastward guard is refusing *naivety*, not
    quietly tightening the gate.

    Args:
        local_timezone_utc_plus_5: Pins the process timezone east of UTC.
    """
    ensure_temporal_integrity(
        CalibrationMap("m1", "2024-12-10"),
        forecast_created_at=NAIVE_MORNING.replace(tzinfo=UTC),
    )


@pytest.mark.parametrize(
    "pin", ["local_timezone_utc_minus_5", "local_timezone_utc_plus_5"]
)
def test_the_refusal_does_not_depend_on_the_host_timezone(
    pin: str, request: pytest.FixtureRequest
) -> None:
    """The same naive instant is refused on either side of UTC.

    A fix that merely happened to be conservative on one host would pass one
    direction and fail the other.

    Args:
        pin: The name of the timezone-pinning fixture to activate.
        request: Pytest's fixture request, used to activate ``pin``.
    """
    request.getfixturevalue(pin)

    with pytest.raises(ValueError, match=_AWARE):
        ensure_temporal_integrity(
            CalibrationMap("m1", "2024-12-10"), forecast_created_at=NAIVE_EVENING
        )


def test_load_calibration_map_refuses_a_naive_instant(
    local_timezone_utc_minus_5: None,
) -> None:
    """The map-loading entry point inherits the gate's refusal.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    with pytest.raises(ValueError, match=_AWARE):
        load_calibration_map(version="2024-12-11", forecast_created_at=NAIVE_EVENING)


def test_the_identity_map_still_bypasses_the_date_comparison() -> None:
    """The ``v0`` sentinel short-circuits before any date work, as before.

    The guard must not disturb the identity map's fast path for aware inputs.
    """
    ensure_temporal_integrity(
        CalibrationMap("identity", "v0"),
        forecast_created_at=datetime(2024, 12, 10, 23, 30, tzinfo=UTC),
    )


# --- Site 2: the daily spend bucket ------------------------------------------


def _budget(ledger: InMemoryBudgetLedger) -> ResearchBudget:
    """Build a budget with room to spend.

    Args:
        ledger: The ledger to record budget events into.

    Returns:
        A research budget with generous ceilings.
    """
    return ResearchBudget(
        per_forecast_micros=1_000_000, per_day_micros=1_000_000, ledger=ledger
    )


def test_ensure_day_open_refuses_a_naive_instant(
    local_timezone_utc_minus_5: None,
) -> None:
    """A naive instant cannot open a day bucket.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    with pytest.raises(ValueError, match=_AWARE):
        _budget(InMemoryBudgetLedger()).ensure_day_open(at=NAIVE_EVENING)


def test_charge_forecast_refuses_a_naive_instant(
    local_timezone_utc_minus_5: None,
) -> None:
    """A naive instant cannot bucket a charge.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    with pytest.raises(ValueError, match=_AWARE):
        _budget(InMemoryBudgetLedger()).charge_forecast(
            100, market_ticker="MKT", at=NAIVE_EVENING
        )


def test_charge_stage_refuses_a_naive_instant(
    local_timezone_utc_minus_5: None,
) -> None:
    """A naive instant cannot bucket a staged charge.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    with pytest.raises(ValueError, match=_AWARE):
        _budget(InMemoryBudgetLedger()).charge_stage(
            100, market_ticker="MKT", at=NAIVE_EVENING
        )


def test_the_naive_evening_charge_would_have_landed_on_the_wrong_day(
    local_timezone_utc_minus_5: None,
) -> None:
    """The aware instant buckets to 2024-12-10; the naive one read as 2024-12-11.

    Spend booked to the wrong calendar day silently resets or double-counts the
    daily ceiling across the UTC-midnight boundary. This pins the *correct*
    bucket for the aware instant on a west-of-UTC host, so the day the naive
    path would have chosen is demonstrably a different one.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(
        per_forecast_micros=1_000_000, per_day_micros=1, ledger=ledger
    )

    budget.charge_forecast(5, market_ticker="MKT", at=NAIVE_EVENING.replace(tzinfo=UTC))
    with pytest.raises(Exception, match="2024-12-10") as excinfo:
        budget.ensure_day_open(at=NAIVE_EVENING.replace(tzinfo=UTC))

    assert "2024-12-11" not in str(excinfo.value)


# --- Site 3: the canary drift window -----------------------------------------


def _breaching_run() -> CanaryRunResult:
    """Build a canary result that breaches the default tolerance.

    Returns:
        A run result whose drift score exceeds the gate tolerance used here.
    """
    return CanaryRunResult(
        distances_ppm={"q1": 90_000}, drift_score_ppm=90_000, worst_question_id="q1"
    )


def test_apply_run_refuses_a_naive_checked_at(
    local_timezone_utc_minus_5: None,
) -> None:
    """A naive ``checked_at`` cannot open a drift window.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    with pytest.raises(ValueError, match="checked_at " + _AWARE):
        CanaryGate(drift_tolerance_ppm=50_000).apply_run(
            _breaching_run(),
            checked_at=NAIVE_EVENING,
            alerts=RecordingAlertEmitter(),
            ledger=InMemoryCanaryLedger(),
        )


def test_apply_version_drift_refuses_a_naive_checked_at(
    local_timezone_utc_minus_5: None,
) -> None:
    """A naive ``checked_at`` cannot open a version-drift window.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    with pytest.raises(ValueError, match="checked_at " + _AWARE):
        CanaryGate().apply_version_drift(
            "openai",
            "gpt-x",
            ("gpt-y",),
            checked_at=NAIVE_EVENING,
            alerts=RecordingAlertEmitter(),
            ledger=InMemoryCanaryLedger(),
        )


def test_acknowledge_refuses_a_naive_acked_at(
    local_timezone_utc_minus_5: None,
) -> None:
    """A naive ``acked_at`` cannot close a drift window.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    with pytest.raises(ValueError, match="acked_at " + _AWARE):
        CanaryGate().acknowledge(acked_at=NAIVE_EVENING, ledger=InMemoryCanaryLedger())


def test_is_live_blocked_refuses_a_naive_created_at(
    local_timezone_utc_minus_5: None,
) -> None:
    """A naive ``created_at`` cannot be tested against a drift window.

    Before the guard this compared bare wall clocks against an aware drift
    instant, raising ``TypeError`` deep inside the gate on a naive/aware mix --
    or, on a naive/naive mix, silently comparing host-local wall clocks.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    with pytest.raises(ValueError, match="created_at " + _AWARE):
        CanaryGate().is_live_blocked(created_at=NAIVE_EVENING)
