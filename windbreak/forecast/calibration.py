"""SPEC S8.2 versioned, temporally-safe calibration maps (stage 11).

A *calibration map* corrects a raw aggregate probability toward the frequency
its historical forecasts actually resolved at, via a deterministic integer
piecewise-linear interpolation over a small table of ``(input_ppm, output_ppm)``
breakpoints. Every quantity is a bare parts-per-million integer and every
rounding decision is a single floor division (``//``), so no float ever enters
the probability path guarded by ``scripts/lint_no_floats.py``.

Two invariants keep a fitted map honest:

* *Domain safety* -- :meth:`CalibrationMap.apply` clamps below the first and
  above the last breakpoint to those endpoints' outputs (never extrapolating)
  and defensively clamps its final result into ``[0, 1_000_000]``.
* *Temporal integrity* -- :func:`ensure_temporal_integrity` rejects a map whose
  training-date version is strictly after the forecast's own creation date, so
  a map "trained on the future" can never leak hindsight into a past forecast.
  The ``"v0"`` identity sentinel always passes: it corrects nothing.

This module is pure (no I/O) and, per the SPEC S8.3 sandbox boundary, imports
nothing from :mod:`windbreak.config`; its ppm bounds are mirrored as local
constants (the ``canary.py`` convention).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, date
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from datetime import datetime

#: The identity calibration map's stable identifier.
IDENTITY_MAP_ID: Final = "identity"

#: The identity calibration map's version sentinel; it always passes temporal
#: integrity and corrects nothing (no resolved forecasts exist yet to fit).
IDENTITY_MAP_VERSION: Final = "v0"

#: Lowest legal ppm value (0.0 probability), for breakpoint inputs/outputs.
_MIN_PPM: Final = 0

#: Highest legal ppm value (1.0 probability), for breakpoint inputs/outputs.
_MAX_PPM: Final = 1_000_000


class TemporalIntegrityError(ValueError):
    """Raised when a calibration map is trained after its forecast's creation.

    A subclass of :class:`ValueError` so callers may catch it broadly, yet a
    distinct type so a genuine future-training breach stays distinguishable from
    a merely malformed version string (which surfaces as a plain
    :class:`ValueError` from :func:`datetime.date.fromisoformat`).
    """


def _clamp_ppm(value: int) -> int:
    """Clamp an integer into the legal ppm domain ``[0, 1_000_000]``.

    Args:
        value: The candidate ppm value.

    Returns:
        ``value`` clamped into ``[0, 1_000_000]``.
    """
    return max(_MIN_PPM, min(value, _MAX_PPM))


def _validate_breakpoint(
    input_ppm: int, output_ppm: int, previous_input: int | None
) -> None:
    """Validate one breakpoint's range and its ordering against its predecessor.

    Args:
        input_ppm: The breakpoint's input probability, in ppm.
        output_ppm: The breakpoint's calibrated output, in ppm.
        previous_input: The prior breakpoint's input, or ``None`` for the first.

    Raises:
        ValueError: If either value is outside ``[0, 1_000_000]`` or the input is
            not strictly greater than ``previous_input``.
    """
    for value in (input_ppm, output_ppm):
        if not _MIN_PPM <= value <= _MAX_PPM:
            msg = (
                f"calibration breakpoint value must be within "
                f"[{_MIN_PPM}, {_MAX_PPM}], got {value}"
            )
            raise ValueError(msg)
    if previous_input is not None and input_ppm <= previous_input:
        msg = (
            f"calibration breakpoint inputs must be strictly increasing, "
            f"got {input_ppm} after {previous_input}"
        )
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CalibrationMap:
    """A versioned integer piecewise-linear probability calibration (SPEC S8.2).

    Attributes:
        map_id: The map's stable, non-empty identifier.
        version: The map's non-empty version. Either the ``"v0"`` identity
            sentinel or an ISO-8601 training date the temporal-integrity guard
            compares against a forecast's creation date.
        entries: The ascending ``(input_ppm, output_ppm)`` breakpoints. Empty
            (the default) is the identity shape: :meth:`apply` returns any input
            unchanged after a defensive clamp.
    """

    map_id: str
    version: str
    entries: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        """Validate the identity strings and breakpoint invariants.

        Raises:
            ValueError: If ``map_id`` or ``version`` is empty, the breakpoint
                inputs are not strictly increasing, or any breakpoint value
                falls outside ``[0, 1_000_000]``. Each message names the field.
        """
        if not self.map_id:
            msg = "map_id must be non-empty"
            raise ValueError(msg)
        if not self.version:
            msg = "version must be non-empty"
            raise ValueError(msg)
        self._validate_entries()

    def _validate_entries(self) -> None:
        """Guard the breakpoint range and strictly-increasing-inputs invariant.

        Raises:
            ValueError: If any input/output is outside ``[0, 1_000_000]`` or the
                inputs are not strictly increasing.
        """
        previous_input: int | None = None
        for input_ppm, output_ppm in self.entries:
            _validate_breakpoint(input_ppm, output_ppm, previous_input)
            previous_input = input_ppm

    def apply(self, probability_ppm: int) -> int:
        """Calibrate one probability by integer piecewise-linear interpolation.

        An empty map is the identity (the input is returned after a defensive
        clamp). Otherwise an input at or below the first breakpoint clamps to its
        output and one at or above the last clamps to its output; an interior
        input is interpolated between its bracketing breakpoints with a single
        floor division. The final result is clamped into ``[0, 1_000_000]``.

        Args:
            probability_ppm: The raw aggregate probability, in ppm.

        Returns:
            The calibrated probability, in ppm.
        """
        if not self.entries:
            return _clamp_ppm(probability_ppm)
        return _clamp_ppm(self._interpolate(probability_ppm))

    def _interpolate(self, probability_ppm: int) -> int:
        """Interpolate ``probability_ppm`` over the (non-empty) breakpoints.

        Args:
            probability_ppm: The raw probability, in ppm.

        Returns:
            The interpolated (endpoint-clamped) output, in ppm, before the
            caller's final domain clamp.
        """
        first_input, first_output = self.entries[0]
        last_input, last_output = self.entries[-1]
        if probability_ppm <= first_input:
            return first_output
        if probability_ppm >= last_input:
            return last_output
        inputs = [entry[0] for entry in self.entries]
        upper = bisect_right(inputs, probability_ppm)
        x0, y0 = self.entries[upper - 1]
        x1, y1 = self.entries[upper]
        return y0 + (probability_ppm - x0) * (y1 - y0) // (x1 - x0)


#: The v0 identity calibration map singleton: no breakpoints, corrects nothing.
IDENTITY_CALIBRATION_MAP: Final = CalibrationMap(IDENTITY_MAP_ID, IDENTITY_MAP_VERSION)


def ensure_temporal_integrity(
    calibration_map: CalibrationMap, *, forecast_created_at: datetime
) -> None:
    """Reject a calibration map trained after its forecast's creation date.

    The ``"v0"`` identity sentinel always passes. Otherwise the version is parsed
    as an ISO-8601 date and compared -- date-only, not full-timestamp -- against
    the forecast's UTC creation date: a training date strictly after it breaches
    temporal integrity. An equal date is allowed (a same-day-trained map is
    integrity-safe).

    Args:
        calibration_map: The map whose version encodes its training date.
        forecast_created_at: The forecast's (timezone-aware) creation instant
            (keyword-only); normalized to its UTC calendar date for comparison.

    Raises:
        TemporalIntegrityError: If the map's training date is strictly after the
            forecast's creation date.
        ValueError: If a non-``"v0"`` version is not a valid ISO-8601 date
            (propagated unwrapped from :func:`datetime.date.fromisoformat`).
    """
    if calibration_map.version == IDENTITY_MAP_VERSION:
        return
    trained_on = date.fromisoformat(calibration_map.version)
    created_on = forecast_created_at.astimezone(UTC).date()
    if trained_on > created_on:
        msg = (
            f"calibration map {calibration_map.map_id!r} trained {trained_on} is "
            f"after the forecast's creation date {created_on}"
        )
        raise TemporalIntegrityError(msg)


def load_calibration_map(
    *,
    version: str,
    forecast_created_at: datetime,
    map_id: str = IDENTITY_MAP_ID,
    entries: tuple[tuple[int, int], ...] = (),
) -> CalibrationMap:
    """Construct a calibration map and enforce its temporal integrity.

    The single entry point that both builds a :class:`CalibrationMap` and runs
    :func:`ensure_temporal_integrity` against a forecast's creation date, so a
    future-trained map is rejected before it can calibrate anything.

    Args:
        version: The map's version (``"v0"`` sentinel or an ISO-8601 training
            date), keyword-only.
        forecast_created_at: The forecast's creation instant the map's training
            date is checked against (keyword-only).
        map_id: The map's identifier (keyword-only); defaults to the identity id.
        entries: The ascending ``(input_ppm, output_ppm)`` breakpoints
            (keyword-only); empty by default (the identity shape).

    Returns:
        The constructed, integrity-checked calibration map.

    Raises:
        TemporalIntegrityError: If the map's training date is after
            ``forecast_created_at``.
        ValueError: If the map's fields are invalid or the version is a
            malformed (non-``"v0"``) date string.
    """
    calibration_map = CalibrationMap(map_id=map_id, version=version, entries=entries)
    ensure_temporal_integrity(calibration_map, forecast_created_at=forecast_created_at)
    return calibration_map
