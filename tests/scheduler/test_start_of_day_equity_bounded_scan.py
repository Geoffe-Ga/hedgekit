"""Bounded derivation of the daily-loss baseline (issue #370).

`_approve_stage` derives `equity_start_of_day` by folding `EquitySampled` rows
out of `deps.store.read_all()`, so EVERY tick of a loop built for continuous
multi-week operation scans the whole ledger and JSON-decodes every row -- and
the ledger only grows, one appended sample per tick among the rest.

`read_start_of_day_equity_micros` replaces that with a bounded, newest-first
walk over `EquitySampled` rows alone (`ReverseTypeScan.iter_records_of_type_
reversed`), stopping the moment it crosses into a previous UTC day. Its cost is
therefore O(samples taken today) -- bounded by ticks-per-day -- rather than
O(ledger).

The capability is deliberately a SEPARATE, narrow protocol rather than more
lines on `LedgerStore`, exactly as issue #246 established for
`LatestRecordLookup`: hand-rolled `LedgerStore` doubles across this suite
satisfy that protocol structurally and must keep working untouched, so the
consumer duck-type dispatches and falls back to the existing
`start_of_day_equity_micros` full fold when the capability is absent.

What must NOT change is the semantics the bound is buying speed for:

* **Earliest**-of-day, never latest. Reading the most recent sample would raise
  the `daily_loss_limit` threshold every time equity grew intraday, quietly
  widening the limit as the day went well.
* The no-sample-yet **veto**. A UTC day with no sample -- including the day's
  first approval, since equity is sampled *after* approving -- must yield no
  baseline at all, never a permissive zero.

RED today: `windbreak.scheduler.loop` exports no
`read_start_of_day_equity_micros` and `windbreak.ledger.store` no
`ReverseTypeScan`, so this module fails collection with `ImportError`. Once
both exist, `test_the_walk_is_bounded_by_todays_samples` is the behavioral RED
this issue closes -- it is the assertion that pins the improvement so it cannot
silently regress to a full scan later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.scheduler.conftest import ScanCountingStore, WalkCountingStore
from windbreak.ledger.events import EquitySampled, ExchangeStatusObserved
from windbreak.ledger.store import ReverseTypeScan, SqliteLedgerStore
from windbreak.numeric import MoneyMicros
from windbreak.scheduler.loop import (
    read_start_of_day_equity_micros,
    start_of_day_equity_micros,
)

if TYPE_CHECKING:
    from pathlib import Path

#: A fixed "now" so no test depends on the wall clock. 2023-11-14T22:13:20Z --
#: late enough in its UTC day that whole hours can be subtracted without
#: crossing midnight.
_NOW_EPOCH_S = 1_700_000_000

#: One UTC day, in whole seconds.
_ONE_DAY_S = 86_400

#: How many *previous-day* samples each bounded-scan test buries under today's.
#: Stands in for the unbounded history an always-on loop accumulates.
_PRIOR_DAY_SAMPLES = 200


def _ledger(path: Path, samples: tuple[tuple[int, int], ...]) -> SqliteLedgerStore:
    """Append `(epoch_s, equity_micros)` samples to a fresh ledger, in order.

    Interleaves an `ExchangeStatusObserved` row after every sample, mirroring
    the real tick (which appends several non-sample rows per beat) and so
    proving the walk's type filter: those rows must not count against the bound.

    Args:
        path: Where the ledger database is created.
        samples: The `(epoch_s, equity_micros)` pairs to append, in order.

    Returns:
        The opened `SqliteLedgerStore`.
    """
    store = SqliteLedgerStore(path)
    for epoch_s, equity_micros in samples:
        store.append(
            EquitySampled(
                component="scheduler",
                equity_micros=equity_micros,
                floor_micros=0,
                epoch_s=epoch_s,
            )
        )
        store.append(
            ExchangeStatusObserved(
                component="scheduler", status="open", observed_at_epoch_s=epoch_s
            )
        )
    return store


def _prior_day_samples() -> tuple[tuple[int, int], ...]:
    """Return `_PRIOR_DAY_SAMPLES` samples spread across the previous UTC day.

    Returns:
        `(epoch_s, equity_micros)` pairs, oldest first, all stamped yesterday.
    """
    return tuple(
        (_NOW_EPOCH_S - _ONE_DAY_S + index, 500_000_000 + index)
        for index in range(_PRIOR_DAY_SAMPLES)
    )


def test_the_walk_is_bounded_by_todays_samples(tmp_path: Path) -> None:
    """The baseline is derived from today's samples plus ONE boundary row --
    never the whole ledger.

    This is the pin: 200 previous-day samples (and 203 interleaved non-sample
    rows) sit under today's three, yet the consumer pulls exactly four records
    and never calls `read_all`. A regression to a full scan, or to draining the
    walk instead of stopping at the day boundary, fails here.
    """
    inner = _ledger(
        tmp_path / "ledger.db",
        (
            *_prior_day_samples(),
            (_NOW_EPOCH_S - 7_200, 90_000_000),
            (_NOW_EPOCH_S - 3_600, 120_000_000),
            (_NOW_EPOCH_S, 140_000_000),
        ),
    )
    store = WalkCountingStore(inner)
    try:
        baseline = read_start_of_day_equity_micros(store, now_epoch_s=_NOW_EPOCH_S)

        assert baseline == MoneyMicros(90_000_000)
        assert store.records_walked == 4
        assert store.read_all_calls == 0
    finally:
        store.close()


def test_falls_back_to_the_full_fold_without_the_capability(tmp_path: Path) -> None:
    """A store that does NOT declare the capability still resolves the same
    baseline, through the original `read_all` fold -- the fallback every
    hand-rolled `LedgerStore` double in this suite depends on.
    """
    inner = _ledger(
        tmp_path / "ledger.db",
        (
            *_prior_day_samples(),
            (_NOW_EPOCH_S - 7_200, 90_000_000),
            (_NOW_EPOCH_S, 140_000_000),
        ),
    )
    store = ScanCountingStore(inner)
    try:
        baseline = read_start_of_day_equity_micros(store, now_epoch_s=_NOW_EPOCH_S)

        assert baseline == MoneyMicros(90_000_000)
        assert store.read_all_calls == 1
    finally:
        store.close()


def test_both_paths_return_the_earliest_sample_of_the_day(tmp_path: Path) -> None:
    """Both paths agree, and both return the day's EARLIEST sample rather than
    its latest.

    Returning the latest would silently raise the `daily_loss_limit` threshold
    every time equity grew intraday -- the loosening the earliest-of-day rule
    exists to prevent -- so this asserts the smaller, older figure explicitly.
    """
    inner = _ledger(
        tmp_path / "ledger.db",
        (
            (_NOW_EPOCH_S - 7_200, 90_000_000),
            (_NOW_EPOCH_S - 3_600, 120_000_000),
            (_NOW_EPOCH_S, 140_000_000),
        ),
    )
    walking = WalkCountingStore(inner)
    scanning = ScanCountingStore(inner)
    try:
        from_walk = read_start_of_day_equity_micros(walking, now_epoch_s=_NOW_EPOCH_S)
        from_scan = read_start_of_day_equity_micros(scanning, now_epoch_s=_NOW_EPOCH_S)

        assert from_walk == from_scan == MoneyMicros(90_000_000)
        assert from_walk == start_of_day_equity_micros(
            inner.read_all(), now_epoch_s=_NOW_EPOCH_S
        )
    finally:
        inner.close()


def test_the_earliest_sample_wins_even_when_appended_last(tmp_path: Path) -> None:
    """Within the day the baseline is chosen by stamped `epoch_s`, not by append
    position, so a clock that steps backwards mid-day cannot promote a later
    sample to the baseline.

    The bounded walk sees the day's samples newest-first, so this also proves it
    keeps comparing stamps rather than seizing the first row it happens to pull.
    """
    inner = _ledger(
        tmp_path / "ledger.db",
        (
            *_prior_day_samples(),
            (_NOW_EPOCH_S, 140_000_000),
            (_NOW_EPOCH_S - 7_200, 90_000_000),
        ),
    )
    store = WalkCountingStore(inner)
    try:
        baseline = read_start_of_day_equity_micros(store, now_epoch_s=_NOW_EPOCH_S)

        assert baseline == MoneyMicros(90_000_000)
    finally:
        store.close()


def test_a_day_with_no_sample_yet_still_has_no_baseline(tmp_path: Path) -> None:
    """Yesterday's samples are not today's baseline: the walk crosses the day
    boundary and returns `None`, so `daily_loss_limit` keeps vetoing.

    This is the day's-first-approval case -- equity is sampled *after*
    approving, so the first tick of every UTC day reads a ledger holding only
    previous-day samples. Carrying one forward would trade against a stale
    baseline instead of failing closed.
    """
    inner = _ledger(tmp_path / "ledger.db", _prior_day_samples())
    store = WalkCountingStore(inner)
    try:
        assert read_start_of_day_equity_micros(store, now_epoch_s=_NOW_EPOCH_S) is None
        assert store.records_walked == 1
    finally:
        store.close()


def test_an_empty_ledger_has_no_baseline(tmp_path: Path) -> None:
    """No sample at all means no baseline -- never a permissive zero-by-default,
    which would be indistinguishable from a genuine reading of zero equity.
    """
    inner = _ledger(tmp_path / "ledger.db", ())
    store = WalkCountingStore(inner)
    try:
        assert read_start_of_day_equity_micros(store, now_epoch_s=_NOW_EPOCH_S) is None
    finally:
        store.close()


def test_a_future_dated_sample_does_not_end_the_walk(tmp_path: Path) -> None:
    """A sample stamped on a LATER UTC day is skipped, not treated as the day
    boundary.

    Only a row that *predates* today ends the walk. Stopping on a forward clock
    blip instead would discard today's genuine baseline and leave the day
    unbaselined for as long as the blip sat at the head of the ledger.
    """
    inner = _ledger(
        tmp_path / "ledger.db",
        (
            *_prior_day_samples(),
            (_NOW_EPOCH_S - 7_200, 90_000_000),
            (_NOW_EPOCH_S + _ONE_DAY_S, 999_000_000),
        ),
    )
    store = WalkCountingStore(inner)
    try:
        baseline = read_start_of_day_equity_micros(store, now_epoch_s=_NOW_EPOCH_S)

        assert baseline == MoneyMicros(90_000_000)
        assert store.records_walked == 3
    finally:
        store.close()


def test_the_capability_leaves_ledger_store_doubles_alone(tmp_path: Path) -> None:
    """The scan-only double is a valid `LedgerStore` yet is NOT a
    `ReverseTypeScan`: the capability is separately declared, so no existing
    double had to grow a method to keep working.
    """
    inner = _ledger(tmp_path / "ledger.db", ())
    try:
        assert not isinstance(ScanCountingStore(inner), ReverseTypeScan)
        assert isinstance(WalkCountingStore(inner), ReverseTypeScan)
    finally:
        inner.close()
