"""The timezone-awareness invariant and the shared UTC renderer (issue #397).

``datetime.astimezone()`` does not raise on a naive value. It reinterprets the
wall clock as the **host's** local time, so the same input yields a different
instant on a developer's laptop than on a UTC-running CI runner -- and yields
the *correct* answer on UTC, which is precisely why no CI test can catch the
misreading. The skew is not even symmetric: west of UTC a naive late-evening
instant reads as the *next* calendar day, east of UTC a naive early-morning one
reads as the *previous* day.

:func:`require_aware` is the single boundary check that refuses such a value.
It **refuses rather than repairs**: assuming UTC for an unprovable instant would
trade a loud wrong answer for a quiet one, letting a value nothing established
as UTC read as healthy. That matches the existing offsetless-timestamp refusals
in ``windbreak/connector/kalshi/normalize.py`` and ``windbreak/connector/fake.py``,
and ``windbreak/forecast/pubdate.py``'s degrade-to-``None`` rule.

:func:`iso_z` is the one UTC ``Z`` renderer for the whole codebase. Six modules
previously each defined a byte-identical private ``_iso_z``, so the guard would
have had to be added -- and kept -- in six places; a seventh copy would have
reintroduced the defect. Rendering an audit-trail timestamp is exactly where a
host-dependent instant must never be written, so the guard lives inside it.

This module deliberately imports nothing from ``windbreak``: both
``windbreak.connector`` and ``windbreak.forecast`` depend on it, so it must stay
a leaf.
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["iso_z", "require_aware"]


def require_aware(value: datetime, field_name: str) -> None:
    """Reject a datetime that carries no UTC offset.

    "Naive" means ``tzinfo is None`` *or* a ``tzinfo`` whose ``utcoffset()``
    returns ``None`` -- Python's own definition, and both shapes in one call.
    Testing ``tzinfo is None`` alone was issue #346: the second shape fell
    through to ``astimezone()``, which treats it exactly like a naive value.

    Args:
        value: The candidate instant.
        field_name: The owning parameter or field's name, surfaced in the error.

    Raises:
        ValueError: If ``value`` carries no UTC offset. The message names
            ``field_name`` and the offending value.
    """
    if value.utcoffset() is None:
        raise ValueError(
            f"{field_name} must be timezone-aware, got naive "
            f"{value.isoformat()}. An offsetless instant is unprovable "
            "evidence -- reading its wall clock as the host's local time would "
            "silently shift it by the host's offset, and across a calendar-day "
            "boundary that changes the answer -- so it is refused rather than "
            "normalized against a guessed timezone."
        )


def iso_z(moment: datetime) -> str:
    """Render a timezone-aware datetime as ISO-8601 UTC with a trailing ``Z``.

    Args:
        moment: The instant to render; normalized to UTC.

    Returns:
        A string like ``2024-12-10T12:00:00.000000Z``.

    Raises:
        ValueError: If ``moment`` carries no UTC offset. An audit trail must
            never record an instant whose meaning depends on which machine
            wrote it.
    """
    require_aware(moment, "moment")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
