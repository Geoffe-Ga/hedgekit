"""A re-enacted recording's markets must be re-enacted too (#510, door 3).

:class:`~windbreak.connector.paper.PaperExchange` accepts a ``replay_anchor``
(issue #369) declaring the instant a recording is being re-enacted from, and
shifts *every recorded instant* by the one offset that puts the earliest
recorded book on that anchor -- books, trade prints (issue #369), and the venue
clock (issue #377). Its market metadata was the exception:
``self.markets = markets`` assigned the loaded mapping verbatim, so a
re-enactment moved the tape and left the calendar behind.

The consequence is not cosmetic, and it is the third closed door under issue
#510. ``close_time`` is what
:func:`~windbreak.screener.filters.horizon_filter` measures against the run's
clock, so a committed books fixture could hold a *book* that stays fresh
forever and could not hold a *horizon* that stays open at all: its close is a
literal, the clock is not, and every committed fixture therefore drifts out of
the production ``[2, 120]``-day window and stays out. That is why
``tests/integration/test_shipped_cli_hermetic_forecast.py`` had to re-date
``markets.json`` from the test before it could reach the screen, and why no
committed configuration could reach a forecast without one.

After this module, a close time is what it always claimed to be inside a
recording: an *offset from when the recording was made*. A fixture that closed
thirty days after its books were captured closes thirty days after the beat
that replays them, whenever that beat runs.

Both halves are pinned, and each is derived from the fixture rather than
transcribed:

* :func:`test_an_anchored_replay_moves_the_close_time_by_the_recordings_offset`
  reads the recording's own origin and close out of the committed JSON and
  asserts the shifted close is exactly ``anchor + (close - origin)``.
* :func:`test_a_verbatim_replay_leaves_every_market_instant_untouched` is the
  negative: with no anchor there is no declared mapping onto the run's clock,
  nothing may move, and the recorded literals survive identically. Live mode
  passes no anchor, so this is the branch that keeps a venue's own metadata
  from being rewritten by a paper offset.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.conftest import pinned_local_timezone
from windbreak.config.schema import HorizonDays
from windbreak.connector.paper import PaperExchange
from windbreak.screener import horizon_filter

#: The repository root, resolved from this file rather than the process's cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed books fixture the two shipped command lines both name.
SHIPPED_BOOKS = REPO_ROOT / "tests" / "fixtures" / "books" / "deep_walk"

#: That fixture's sole market.
TICKER = "MKT-DEEP"

#: An arbitrary, fixed re-enactment instant. Nothing here reads the host clock,
#: so the module is timezone- and calendar-independent (2027-01-15T08:00:00Z).
ANCHOR = datetime(2027, 1, 15, 8, 0, 0, tzinfo=UTC)

#: The format every committed fixture writes its instants in.
FIXTURE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

#: The settlement lag written onto a market for the resolution-instant test.
RESOLUTION_LAG = timedelta(days=7)


def _fixture_instant(raw: str) -> datetime:
    """Parse one committed fixture timestamp as a UTC-aware instant.

    Args:
        raw: The recorded timestamp text.

    Returns:
        The parsed instant, in UTC.
    """
    return datetime.strptime(raw, FIXTURE_TIME_FORMAT).replace(tzinfo=UTC)


def _recorded_close() -> datetime:
    """Return the committed market's recorded close time.

    Returns:
        The ``close_time`` literal in the fixture's ``markets.json``.
    """
    markets = json.loads((SHIPPED_BOOKS / "markets.json").read_text(encoding="utf-8"))
    return _fixture_instant(str(markets[0]["close_time"]))


def _recorded_origin() -> datetime:
    """Return the earliest recorded book instant across the committed session.

    The recording's own origin, read from the fixture rather than restated, so
    this module measures the same zero point
    :func:`~windbreak.connector.paper._replay_offset` measures.

    Returns:
        The earliest ``book.fetched_at`` in the fixture's ``sessions.json``.
    """
    sessions = json.loads((SHIPPED_BOOKS / "sessions.json").read_text(encoding="utf-8"))
    return min(
        _fixture_instant(str(step["book"]["fetched_at"]))
        for steps in sessions.values()
        for step in steps
    )


def test_an_anchored_replay_moves_the_close_time_by_the_recordings_offset() -> None:
    """An anchored replay re-dates the market by the recording's own offset.

    The claim under test is an equality, not an inequality: the shifted close
    sits exactly as far after the anchor as the recorded close sat after the
    recorded origin. Asserting the preserved *interval* rather than a literal
    is what makes this a statement about the re-enactment rather than about one
    fixture's numbers, and it is the property the horizon filter needs -- a
    fixture's horizon is whatever the recording said it was, forever.

    The recorded close is asserted to differ from the shifted one first, so the
    equality below cannot be satisfied by an offset that is accidentally zero.
    """
    recorded_close = _recorded_close()
    recorded_horizon = recorded_close - _recorded_origin()

    exchange = PaperExchange.from_fixture_dir(SHIPPED_BOOKS, replay_anchor=ANCHOR)
    market = exchange.get_market(TICKER)

    assert recorded_horizon > timedelta(0)
    assert market.close_time != recorded_close
    assert market.close_time == ANCHOR + recorded_horizon
    assert market.close_time - ANCHOR == recorded_horizon
    assert exchange.list_markets() == (market,)


def test_the_re_enacted_horizon_is_the_recorded_horizon() -> None:
    """The horizon filter measures the recording's own horizon, at any anchor.

    The point of door 3 stated as the screen sees it. ``deep_walk`` records a
    close 151 days after its books, which is outside the production ``[2,
    120]``-day window, so it is still refused -- and refused for a *measured*
    reason rather than because its literal went stale. The same measurement is
    taken at two anchors a year apart and comes back identical, which is the
    property no committed fixture had before: a recorded horizon that does not
    decay.
    """
    window = HorizonDays()
    recorded_horizon = _recorded_close() - _recorded_origin()
    later_anchor = ANCHOR + timedelta(days=365)

    at_anchor = horizon_filter(
        PaperExchange.from_fixture_dir(SHIPPED_BOOKS, replay_anchor=ANCHOR).get_market(
            TICKER
        ),
        now=ANCHOR,
        min_days=window.min,
        max_days=window.max,
    )
    a_year_later = horizon_filter(
        PaperExchange.from_fixture_dir(
            SHIPPED_BOOKS, replay_anchor=later_anchor
        ).get_market(TICKER),
        now=later_anchor,
        min_days=window.min,
        max_days=window.max,
    )

    assert at_anchor.measured == recorded_horizon.days
    assert a_year_later.measured == recorded_horizon.days
    assert at_anchor.measured == 151
    assert at_anchor.passed is False
    assert a_year_later.passed is False
    assert window.max == 120


@pytest.mark.parametrize("zone", ["EST5", "XXX-5", "UTC"])
@pytest.mark.parametrize(
    "anchor",
    [
        datetime(2027, 1, 15, 4, 30, tzinfo=UTC),
        datetime(2027, 1, 15, 5, 30, tzinfo=UTC),
        datetime(2027, 1, 15, 18, 30, tzinfo=UTC),
        datetime(2027, 1, 15, 19, 30, tzinfo=UTC),
    ],
)
def test_local_midnight_does_not_move_the_re_enacted_horizon(
    zone: str, anchor: datetime
) -> None:
    """The re-enacted horizon is the same measurement in every process timezone.

    Issue #510's fix makes a fixture's close time a function of the run's clock,
    which is exactly the shape that hides a local-vs-UTC defect: CI runs UTC and
    cannot see one. The four anchors straddle *local* midnight in both a
    UTC-05:00 and a UTC+05:00 process (04:30Z/05:30Z and 18:30Z/19:30Z), so a
    shift or a horizon computed against a naive local reading would put two of
    them on a different calendar day from their neighbour.

    The measured horizon is asserted **equal across every combination**, derived
    from the recording rather than restated, and equal to the interval the
    fixture itself records. Three zones rather than two because a test that only
    ever compares two wrong answers to each other proves nothing; ``UTC`` is the
    control.

    Args:
        zone: The pinned process timezone.
        anchor: The re-enactment instant.
    """
    recorded_horizon = _recorded_close() - _recorded_origin()
    window = HorizonDays()

    with pinned_local_timezone(zone):
        market = PaperExchange.from_fixture_dir(
            SHIPPED_BOOKS, replay_anchor=anchor
        ).get_market(TICKER)
        verdict = horizon_filter(
            market, now=anchor, min_days=window.min, max_days=window.max
        )

    assert market.close_time == anchor + recorded_horizon
    assert verdict.measured == recorded_horizon.days
    assert verdict.measured == 151


def test_the_expected_resolution_time_moves_by_the_same_offset(
    tmp_path: Path,
) -> None:
    """A market's resolution instant shifts with its close, by the one offset.

    ``expected_resolution_time`` is the market's *other* recorded instant, and
    :class:`~windbreak.connector.models.NormalizedMarket` enforces that it is
    never before the close. Shifting one and not the other would either newly
    violate that invariant or silently invent a settlement lag the recording
    never had, so the two are asserted to move by an identical delta and the
    recorded gap between them to survive exactly.

    No committed fixture carries one today, which is precisely why this is
    written: an unexercised branch is indistinguishable from a wrong one.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    books = tmp_path / "books"
    shutil.copytree(SHIPPED_BOOKS, books)
    path = books / "markets.json"
    markets = json.loads(path.read_text(encoding="utf-8"))
    recorded_close = _fixture_instant(str(markets[0]["close_time"]))
    recorded_resolution = recorded_close + RESOLUTION_LAG
    markets[0]["expected_resolution_time"] = (
        recorded_resolution.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    )
    path.write_text(json.dumps(markets, indent=2), encoding="utf-8")

    market = PaperExchange.from_fixture_dir(books, replay_anchor=ANCHOR).get_market(
        TICKER
    )

    assert market.expected_resolution_time is not None
    assert market.expected_resolution_time != recorded_resolution
    assert market.expected_resolution_time - recorded_resolution == (
        market.close_time - recorded_close
    )
    assert market.expected_resolution_time - market.close_time == RESOLUTION_LAG


def test_a_verbatim_replay_leaves_every_market_instant_untouched() -> None:
    """With no anchor nothing is re-dated, so the recorded literals survive.

    A verbatim replay declares no mapping between the recording and the run's
    clock, so there is no offset to apply and applying one would fabricate a
    calendar the fixture never claimed. This is also the live branch:
    :func:`~windbreak.scheduler.loop.build_paper_exchange` passes no
    ``replay_anchor`` in live mode, and a venue's own market metadata must
    reach the screen exactly as the venue reported it.
    """
    exchange = PaperExchange.from_fixture_dir(SHIPPED_BOOKS)
    market = exchange.get_market(TICKER)

    assert market.close_time == _recorded_close()
    assert market.expected_resolution_time is None
