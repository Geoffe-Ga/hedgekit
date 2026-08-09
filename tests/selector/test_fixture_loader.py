"""Tests for tests.selector.fixture_loader's timestamp parsing (issue #392).

`fixture_loader` turns the checked-in `tests/selector/fixtures/bundle_*.json`
bundles into selector inputs, and its `_parse_dt` had the same shape as the
production bug this issue fixes: `datetime.fromisoformat(value).astimezone(UTC)`
returns a *naive* datetime for an offsetless string, and `.astimezone(UTC)` on a
naive value does not raise -- it reads the wall clock as the **host's** local
time. Every bundle timestamp carries a `Z` today, so the loader happened to be
correct; nothing checked that it stayed so, and a bundle edited without an offset
would have silently meant a different instant on every host.

Both tests pin a fixed UTC-05:00 host via `local_timezone_utc_minus_5`
(tests/conftest.py). Without the pin these assertions could never fail in CI,
which runs UTC: there the misreading is the identity.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.selector.fixture_loader import _parse_dt, _parse_optional_dt


@pytest.mark.usefixtures("local_timezone_utc_minus_5")
def test_offsetless_bundle_timestamp_is_refused_naming_the_field_and_value() -> None:
    """An offsetless bundle timestamp raises, naming both field and value.

    Pre-fix this returned `2024-12-10T17:00:00+00:00` under the pinned host and
    `12:00Z` on a UTC host -- two different instants from one checked-in
    fixture. The refusal makes the bundle's meaning a property of the bundle.
    """
    with pytest.raises(ValueError, match=r"created_at.*2024-12-10T12:00:00"):
        _parse_dt("2024-12-10T12:00:00", field="created_at")


@pytest.mark.usefixtures("local_timezone_utc_minus_5")
def test_offset_bearing_bundle_timestamp_converts_to_one_fixed_instant() -> None:
    """A timestamp carrying an offset converts by *its* offset, not the host's.

    A non-UTC offset is used deliberately: it proves the conversion reads the
    string's offset rather than the pinned host zone, and that the refusal above
    did not over-reach into rejecting every non-`Z` timestamp. The optional
    wrapper is exercised alongside it, since `None` must still pass through as
    an absent value rather than tripping the offset check.
    """
    assert _parse_dt("2024-12-10T07:00:00-05:00", field="created_at") == datetime(
        2024, 12, 10, 12, tzinfo=UTC
    )
    assert _parse_optional_dt(None, field="publication_date") is None
