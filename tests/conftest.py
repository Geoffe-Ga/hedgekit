"""Process-wide state hygiene shared across the test suite (issues #65, #392).

``windbreak.main._install_signal_handlers`` mutates process-global signal
dispositions via ``signal.signal(...)`` without saving or restoring the
previous handler. Any test that exercises that path -- directly, or
indirectly through ``windbreak.main.main`` -- can leave a later test, or
the pytest process itself, with a hijacked SIGINT/SIGTERM handler. This
conftest installs an autouse fixture that snapshots and restores both
dispositions around every test in the suite, so no individual test module
has to opt in.

Issue #392 adds the opt-in ``local_timezone_utc_minus_5`` fixture. It lives here
rather than in one package's conftest because three packages need the same pin:
``tests/connector/kalshi`` (the ``Date``-header and Kalshi-timestamp guards),
``tests/connector`` (the ``FakeExchange`` fixture loader), and ``tests/selector``
(the bundle loader).

Issue #397 adds its east-of-UTC counterpart, ``local_timezone_utc_plus_5``, and
factors both onto :func:`pinned_local_timezone`. A naive datetime misread as
local time skews *opposite* directions on either side of UTC, and only the
westward skew fails open; pinning both keeps the safe direction from being
mistaken for a correct one.
"""

from __future__ import annotations

import os
import signal
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def preserved_signal_handlers() -> Iterator[None]:
    """Snapshot and restore the SIGINT/SIGTERM dispositions around a block.

    The restoration happens in a ``finally`` clause, so it runs whether the
    wrapped block completes normally or raises.

    Yields:
        None. The caller's code runs with whatever SIGINT/SIGTERM handlers
        were in effect on entry still installed; it is free to replace
        them, and both dispositions are restored to their entry values on
        exit.
    """
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


@pytest.fixture(autouse=True)
def restore_signal_handlers() -> Iterator[None]:
    """Save and restore SIGINT/SIGTERM handlers around every test.

    ``_install_signal_handlers`` mutates process-global signal
    dispositions; without this fixture a failing assertion (or any test
    that installs handlers and never restores them) could leave later
    tests, or the pytest process itself, with a hijacked SIGINT or SIGTERM
    handler. Autouse means every test in the suite gets this protection
    without opting in explicitly.
    """
    with preserved_signal_handlers():
        yield


@contextmanager
def pinned_local_timezone(zone: str) -> Iterator[None]:
    """Pin the process's local timezone to ``zone`` for the wrapped block.

    Args:
        zone: A POSIX ``TZ`` value. Note the POSIX sign convention is inverted
            relative to the UTC offset it names: ``"EST5"`` is UTC-05:00 and
            ``"XXX-5"`` is UTC+05:00. Both are fixed-offset (no DST rule), so
            they hold on every date.

    Yields:
        None, with ``TZ`` pinned. The previous ``TZ`` and the interpreter's
        cached zone are both restored on exit, whether or not the block raises.
    """
    previous = os.environ.get("TZ")
    os.environ["TZ"] = zone
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = previous
        time.tzset()


@pytest.fixture
def local_timezone_utc_minus_5() -> Iterator[None]:
    """Pin the process's local timezone to a fixed UTC-05:00 for one test.

    `datetime.astimezone()` silently interprets a *naive* datetime as **local**
    time. A test distinguishing "read as UTC" from "reinterpreted as local" is
    therefore a false green on a UTC host -- which most CI runners are -- since
    both paths then yield the same instant. Pinning a fixed-offset zone (`EST5`
    carries no DST rule, so it is UTC-05:00 on every date) makes the correct
    and buggy results differ by five hours on every host, in CI and locally
    alike.

    Shared here rather than per-module because several packages need the same
    pin: issue #346's `Date`-header guard in `kalshi/client.py`, and issue
    #392's offsetless-timestamp refusal in `kalshi/normalize.py`,
    `connector/fake.py`, and `tests/selector/fixture_loader.py`.

    Yields:
        None, with `TZ` pinned. The previous `TZ` and the interpreter's cached
        zone are both restored on exit, whether or not the test raises.
    """
    with pinned_local_timezone("EST5"):
        yield


@pytest.fixture
def local_timezone_utc_plus_5() -> Iterator[None]:
    """Pin the process's local timezone to a fixed UTC+05:00 for one test.

    The east-of-UTC counterpart to :func:`local_timezone_utc_minus_5`, added by
    issue #397. The two directions skew a misread naive datetime *opposite*
    ways across a calendar-day boundary, and only one of them is dangerous:

    - **West (UTC-05:00)**: a naive late-evening wall clock reads as the *next*
      UTC day. In `calibration.ensure_temporal_integrity` that pushes
      `created_on` later than the forecast's true creation date, so a map
      trained the following day slips past `trained_on > created_on` -- the
      gate **fails open**, admitting future information into a backtest.
    - **East (UTC+05:00)**: a naive early-morning wall clock reads as the
      *previous* UTC day, so a legitimately same-day-trained map is rejected --
      the gate **fails closed**, which is merely a false alarm.

    Both directions are pinned in the regression tests so the safe direction is
    never mistaken for correctness: a fix must refuse the unprovable instant on
    *either* host, not merely happen to be conservative on one of them.

    Yields:
        None, with `TZ` pinned. The previous `TZ` and the interpreter's cached
        zone are both restored on exit, whether or not the test raises.
    """
    with pinned_local_timezone("XXX-5"):
        yield
