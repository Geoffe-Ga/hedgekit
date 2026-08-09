"""Tests for the shared timezone-awareness guard (issue #397).

``datetime.astimezone()`` does not raise on a naive value -- it silently reads
the wall clock as the *host's* local time. :func:`windbreak.timekeeping.require_aware`
is the single boundary check that refuses such a value before it can reach a
comparison, a calendar-day bucket, or a ledgered timestamp.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone, tzinfo

import pytest

from windbreak.timekeeping import iso_z, require_aware


class _OffsetlessTimezone(tzinfo):
    """A ``tzinfo`` whose ``utcoffset()`` returns ``None``.

    This is the second naive shape (issue #346): ``tzinfo is not None`` yet the
    value carries no offset, so it is naive *in effect* and ``astimezone()``
    still reads it as host-local. A guard testing ``tzinfo is None`` alone
    misses it entirely, which is why the check consults ``utcoffset()``.
    """

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        """Return no offset at all.

        Args:
            dt: The datetime being interrogated; ignored.

        Returns:
            ``None``, always -- the whole point of this stub.
        """
        return None

    def tzname(self, dt: datetime | None) -> str | None:
        """Return no zone name.

        Args:
            dt: The datetime being interrogated; ignored.

        Returns:
            ``None``, always.
        """
        return None

    def dst(self, dt: datetime | None) -> timedelta | None:
        """Return no DST offset.

        Args:
            dt: The datetime being interrogated; ignored.

        Returns:
            ``None``, always.
        """
        return None


def test_require_aware_accepts_a_utc_datetime() -> None:
    """An aware UTC datetime passes the guard without raising."""
    require_aware(datetime(2024, 12, 10, 23, 30, tzinfo=UTC), "created_at")


def test_require_aware_accepts_a_non_utc_offset() -> None:
    """Awareness, not UTC-ness, is the invariant: a fixed offset passes."""
    moment = datetime(2024, 12, 10, 23, 30, tzinfo=timezone(timedelta(hours=-5)))
    require_aware(moment, "created_at")


def test_require_aware_rejects_a_naive_datetime() -> None:
    """A ``tzinfo``-less datetime is refused, naming the field and the value."""
    with pytest.raises(
        ValueError, match="created_at must be timezone-aware"
    ) as excinfo:
        require_aware(datetime(2024, 12, 10, 23, 30), "created_at")
    assert "2024-12-10T23:30:00" in str(excinfo.value)


def test_require_aware_rejects_a_tzinfo_whose_utcoffset_is_none() -> None:
    """The second naive shape (#346) is refused too, not just ``tzinfo is None``.

    A guard written as ``value.tzinfo is None`` would wave this through even
    though ``astimezone()`` treats it exactly like a naive value.
    """
    moment = datetime(2024, 12, 10, 23, 30, tzinfo=_OffsetlessTimezone())
    assert moment.tzinfo is not None
    with pytest.raises(ValueError, match="fetched_at must be timezone-aware"):
        require_aware(moment, "fetched_at")


def test_require_aware_does_not_assume_utc_for_a_naive_value() -> None:
    """The guard refuses; it never repairs a naive value by assuming UTC.

    Assuming UTC would trade a loud wrong answer for a quiet one: an instant
    nothing proved is UTC would then read as healthy. The only safe response to
    an unprovable instant is to refuse it.
    """
    naive = datetime(2024, 12, 10, 23, 30)
    with pytest.raises(ValueError, match="must be timezone-aware"):
        require_aware(naive, "at")
    assert naive.tzinfo is None


def test_iso_z_renders_a_utc_instant() -> None:
    """An aware UTC datetime renders as an ISO-8601 ``Z`` string."""
    assert iso_z(datetime(2024, 12, 10, 12, 0, tzinfo=UTC)) == (
        "2024-12-10T12:00:00.000000Z"
    )


def test_iso_z_normalizes_a_non_utc_offset_to_utc() -> None:
    """A non-UTC aware instant is converted, not merely relabelled."""
    moment = datetime(2024, 12, 10, 7, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert iso_z(moment) == "2024-12-10T12:00:00.000000Z"


def test_iso_z_refuses_a_naive_datetime(
    local_timezone_utc_minus_5: None,
) -> None:
    """A naive datetime is refused rather than rendered against the host's zone.

    Without the guard this returns ``2024-12-11T04:30:00.000000Z`` on a UTC-5
    host and ``2024-12-10T23:30:00.000000Z`` on a UTC host -- the same audit
    trail entry reading differently depending on which machine wrote it. The
    timezone pin is what makes this test able to fail; on UTC-running CI the
    naive and correct renderings are identical.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    with pytest.raises(ValueError, match="moment must be timezone-aware"):
        iso_z(datetime(2024, 12, 10, 23, 30))
