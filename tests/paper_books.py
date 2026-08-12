"""Shared preparation of a committed books fixture for a re-enacted replay.

Several test modules copy a committed ``tests/fixtures/books`` directory and
move the *inputs* the SPEC §16 screen measures -- the close time, the resting
depth, the opening balance -- to values that genuinely satisfy the production
thresholds. The close time is the one that cannot be written as a plain
instant, and this module exists to say why once instead of four times.

:class:`~windbreak.connector.paper.PaperExchange` re-enacts a recording from a
``replay_anchor``: it computes one offset -- the anchor minus the earliest
recorded book -- and shifts every recorded instant by it, the market calendar
included since issue #510. So a fixture's ``close_time`` is not "when this
market closes"; it is **when this market closed relative to the books beside
it**, and the run's clock supplies the rest. Writing an absolute
``now + 30 days`` into ``markets.json`` therefore lands 30 days after the
anchor *plus the whole recording's age*, which for a 2025 fixture replayed in
2027 is two years outside the ``[2, 120]``-day window.

:func:`set_close_time` takes the only quantity that survives the re-enactment:
whole days from the recording's own leading book. A market written 30 days
after the recording closes 30 days after the beat that replays it, whenever
that beat runs -- which is exactly the property that lets a committed fixture
hold a horizon at all.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: The timestamp format every committed books fixture writes its instants in.
FIXTURE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def fixture_instant(raw: str) -> datetime:
    """Parse one committed fixture timestamp as a UTC-aware instant.

    Args:
        raw: The recorded timestamp text.

    Returns:
        The parsed instant, in UTC.
    """
    return datetime.strptime(raw, FIXTURE_TIME_FORMAT).replace(tzinfo=UTC)


def recording_origin(books: Path) -> datetime:
    """Return the earliest recorded book instant in a books fixture.

    The recording's own zero point, and the one
    :func:`~windbreak.connector.paper._replay_offset` measures its shift from.
    Read out of the fixture rather than restated, so a fixture whose session is
    re-recorded moves this with it.

    Args:
        books: The books directory to read ``sessions.json`` from.

    Returns:
        The earliest ``book.fetched_at`` across every recorded ticker.
    """
    sessions = json.loads((books / "sessions.json").read_text(encoding="utf-8"))
    return min(
        fixture_instant(str(step["book"]["fetched_at"]))
        for steps in sessions.values()
        for step in steps
    )


def set_close_time(books: Path, *, days_after_recording: int) -> None:
    """Re-date the fixture market's close relative to its own recording.

    Args:
        books: The books directory to rewrite.
        days_after_recording: Whole days from the recording's leading book to
            the market's close. Negative values put the close *before* the
            recording, which is how a fixture is made to fail the horizon
            window on a measured reading rather than on a stale literal.
    """
    closes_at = recording_origin(books) + timedelta(days=days_after_recording)
    path = books / "markets.json"
    markets = json.loads(path.read_text(encoding="utf-8"))
    markets[0]["close_time"] = closes_at.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    path.write_text(json.dumps(markets, indent=2), encoding="utf-8")
