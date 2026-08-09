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
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "EST5"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = previous
        time.tzset()
