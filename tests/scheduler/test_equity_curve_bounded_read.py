"""Bounded, exactly-equal read of the equity curve (issue #516).

`read_equity_curve` folds the account's all-time high-water mark and its newest
sampled reading out of every `EquitySampled` row the ledger holds, on every
tick. The newest reading is O(1) on a newest-first walk. The mark is not: a
maximum over an all-time series has no recency predicate to stop on, so PR
#515 shipped a walk that drains the whole series -- and the series gains one
row per beat, for the life of a deployment epic #455 means to run unattended
for a week.

The trap is that the obvious bound is wrong in the dangerous direction.
Truncating the walk to a recent window silently *lowers* the mark, and
`trailing_drawdown_limit`'s threshold is a ppm share of that mark, so a lower
mark is a looser cap. A cap that stops binding is the exact failure #513/#514
were filed for. `test_a_truncated_walk_understates_the_mark_by_699_999_501`
is the positive control that pins how far wrong the naive fix goes on this
module's own corpus, so nothing here can pass by coincidence.

The bound this module pins instead is a **resumable watermark**, not a window.
`EquityCurveCursor` remembers the fold of every sample up to and including one
`sequence_number`; the next read walks newest-first and stops the moment it
reaches a row at or below that watermark, because everything from there down
is already inside the remembered maximum. The answer is therefore *exactly*
the unbounded one, by

    max(all samples) == max(max(samples <= w), max(samples > w))

and not approximately it -- which is why every assertion below compares the
bounded answer to a brute-force whole-ledger fold rather than to a tolerance.

Three properties PR #515 pinned must survive the memo:

* **Restart.** The cursor is per-process and starts cold, so a restarted loop
  re-folds the mark from the ledger's own rows rather than resetting to
  whatever it samples first. `test_a_restarted_process_recovers_the_same_mark`
  reopens the database behind a fresh cursor and asserts the same micros.
* **All-time ratchet.** The watermark bounds *how far back the walk goes*, not
  *how far back the mark counts*: the remembered maximum already covers
  everything below it. The corpus below puts its peak at index 3 of hundreds,
  so a mark that forgot anything old would report a different number.
* **The fallback.** A store not declaring `ReverseTypeScan` -- every
  hand-rolled double in this suite -- keeps folding `read_all()`, and must
  still answer identically.

RED before the fix: `windbreak.scheduler.loop` exports no `EquityCurveCursor`,
so this module fails collection with `ImportError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests.scheduler.conftest import ScanCountingStore, WalkCountingStore
from windbreak.ledger.events import EquitySampled, ExchangeStatusObserved
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.numeric import MoneyMicros
from windbreak.scheduler.loop import (
    EquityCurveCursor,
    equity_curve_micros,
    read_equity_curve,
)

if TYPE_CHECKING:
    from pathlib import Path

#: A fixed instant so no test here depends on the wall clock:
#: 2023-11-14T22:13:20Z. The curve fold is all-time and reads no calendar, so
#: this only has to be stable, not placed against a day boundary.
_NOW_EPOCH_S = 1_700_000_000

#: The two corpus sizes AC1 is measured at: N and 10N samples.
_N_SAMPLES = 50
_TEN_N_SAMPLES = 500

#: The oldest sample, deliberately the series minimum and NOT the peak.
_OLDEST_MICROS = 100_000_000

#: The all-time high-water mark, and the index it sits at. Index 3 of hundreds
#: makes it neither the newest sample nor the oldest, so a bound that kept only
#: the newest rows and a bound that kept only the oldest both miss it.
_PEAK_MICROS = 900_000_000
_PEAK_INDEX = 3

#: Every other sample is `_ORDINARY_BASE_MICROS + index`: strictly increasing,
#: all distinct, and -- for every corpus size this module builds -- hundreds of
#: millions of micros below the peak, so an understated mark can never coincide
#: with the true one.
_ORDINARY_BASE_MICROS = 200_000_000

#: The window a naive "just bound the walk" fix would keep. Ten rows of the
#: 500-row corpus, i.e. a window that provably excludes the peak.
_NAIVE_WINDOW_SAMPLES = 10

#: What that naive window reports as the mark on the 500-sample corpus, and how
#: far below the truth it lands. Both pinned to the micro: the whole issue is
#: that the failure is a *quiet* number, not an exception.
_NAIVE_WINDOW_MARK_MICROS = 200_000_499
_NAIVE_WINDOW_SHORTFALL_MICROS = 699_999_501

#: What a warm cursor pulls per read once one new sample has been appended: the
#: new row, plus the one row at the watermark that ends the walk.
_WARM_ROWS_WALKED = 2

#: The component every row below is stamped with.
_COMPONENT = "scheduler"


def _sample_micros(count: int) -> tuple[int, ...]:
    """Return `count` sampled equities, oldest first.

    Args:
        count: How many samples the corpus holds. Must exceed `_PEAK_INDEX`.

    Returns:
        The equity readings, in micros, in append order.
    """
    return tuple(
        _OLDEST_MICROS
        if index == 0
        else _PEAK_MICROS
        if index == _PEAK_INDEX
        else _ORDINARY_BASE_MICROS + index
        for index in range(count)
    )


def _ledger(path: Path, equities: tuple[int, ...]) -> SqliteLedgerStore:
    """Append `equities` to a fresh ledger, one sample per reading, in order.

    An `ExchangeStatusObserved` row is interleaved after every sample, the way
    a real tick appends several non-sample rows per beat, so the walk's type
    filter is exercised rather than assumed: those rows must never count
    against the bound.

    Args:
        path: Where the ledger database is created.
        equities: The equity readings, in micros, in append order.

    Returns:
        The opened `SqliteLedgerStore`, its chain verified.
    """
    store = SqliteLedgerStore(path)
    for index, equity_micros in enumerate(equities):
        store.append(
            EquitySampled(
                component=_COMPONENT,
                equity_micros=equity_micros,
                floor_micros=0,
                epoch_s=_NOW_EPOCH_S + index,
            )
        )
        store.append(
            ExchangeStatusObserved(
                component=_COMPONENT,
                status="open",
                observed_at_epoch_s=_NOW_EPOCH_S + index,
            )
        )
    store.verify_chain()
    return store


def _append_one(store: SqliteLedgerStore, equity_micros: int, index: int) -> None:
    """Append one further sample, the way a later tick would.

    Args:
        store: The ledger to append to.
        equity_micros: The reading to record, in micros.
        index: The sample's position in the series, stamping its instant.
    """
    store.append(
        EquitySampled(
            component=_COMPONENT,
            equity_micros=equity_micros,
            floor_micros=0,
            epoch_s=_NOW_EPOCH_S + index,
        )
    )
    store.verify_chain()


@dataclass(frozen=True, slots=True)
class _WalkCost:
    """What one corpus cost to read cold and then warm.

    Attributes:
        cold: Rows pulled by the first read, against a fresh cursor.
        warm: Rows pulled by the second read, after one further sample was
            appended -- the steady-state per-tick cost.
        curve_marks: The high-water mark each of the two reads answered, in
            micros.
    """

    cold: int
    warm: int
    curve_marks: tuple[int, int]


def _walk_cost(path: Path, count: int) -> _WalkCost:
    """Measure the cold and warm row counts for a `count`-sample corpus.

    Args:
        path: Where the ledger database is created.
        count: How many samples the corpus holds before the warm read.

    Returns:
        The measured `_WalkCost`.
    """
    inner = _ledger(path, _sample_micros(count))
    store = WalkCountingStore(inner)
    cursor = EquityCurveCursor()
    try:
        cold_curve = read_equity_curve(store, cursor=cursor)
        cold = store.records_walked
        _append_one(inner, _ORDINARY_BASE_MICROS + count, count)
        warm_curve = read_equity_curve(store, cursor=cursor)
        warm = store.records_walked - cold
        assert cold_curve is not None
        assert warm_curve is not None
        assert store.read_all_calls == 0
        return _WalkCost(
            cold=cold,
            warm=warm,
            curve_marks=(
                cold_curve.high_water_mark.value,
                warm_curve.high_water_mark.value,
            ),
        )
    finally:
        store.close()


class TestTheCorpusDiscriminates:
    """No figure here does double duty, so no two behaviours share an answer."""

    def test_the_peak_is_neither_the_newest_sample_nor_the_oldest(self) -> None:
        """The shape every equality below rests on, asserted rather than assumed."""
        equities = _sample_micros(_TEN_N_SAMPLES)
        assert len(equities) == _TEN_N_SAMPLES
        assert max(equities) == _PEAK_MICROS
        assert equities.index(_PEAK_MICROS) == _PEAK_INDEX
        assert equities[0] != _PEAK_MICROS
        assert equities[-1] != _PEAK_MICROS
        assert equities[0] == min(equities)

    def test_every_sample_in_the_corpus_is_distinct(self) -> None:
        """Repeated readings would let an understated mark equal a true one."""
        equities = _sample_micros(_TEN_N_SAMPLES)
        assert len(set(equities)) == _TEN_N_SAMPLES

    def test_a_truncated_walk_understates_the_mark_by_699_999_501(self) -> None:
        """The naive fix, priced to the micro.

        A walk bounded to the newest ten samples reports 200_000_499 where the
        truth is 900_000_000 -- a mark 699_999_501 micros too low, and so a
        `trailing_drawdown_limit` threshold 12.5% of that much too small. The
        cap does not fail loudly there; it simply stops reaching.
        """
        equities = _sample_micros(_TEN_N_SAMPLES)
        truncated = max(equities[-_NAIVE_WINDOW_SAMPLES:])
        assert truncated == _NAIVE_WINDOW_MARK_MICROS
        assert truncated < _PEAK_MICROS
        assert _PEAK_MICROS - truncated == _NAIVE_WINDOW_SHORTFALL_MICROS


class TestThePerTickCostIsBounded:
    """AC1: rows pulled per read stop scaling with the age of the ledger."""

    def test_the_warm_read_pulls_the_same_rows_at_n_and_at_ten_n(
        self, tmp_path: Path
    ) -> None:
        """The measurement, not an assertion about the code.

        The cold read is the startup fold and scales with the corpus, exactly
        as a re-fold from the ledger must. Every read after it pulls two rows
        -- the tick's own new sample, and the one row at the watermark that
        ends the walk -- whether the ledger holds fifty samples or five
        hundred. That is the property the seven-day soak needs.
        """
        small = _walk_cost(tmp_path / "small.db", _N_SAMPLES)
        large = _walk_cost(tmp_path / "large.db", _TEN_N_SAMPLES)

        assert small.cold == _N_SAMPLES
        assert large.cold == _TEN_N_SAMPLES
        assert large.cold == small.cold * 10
        assert small.warm == _WARM_ROWS_WALKED
        assert large.warm == _WARM_ROWS_WALKED

    def test_the_bounded_read_answers_the_same_mark_cold_and_warm(
        self, tmp_path: Path
    ) -> None:
        """A cheaper read that changed the answer would be no fix at all."""
        large = _walk_cost(tmp_path / "large.db", _TEN_N_SAMPLES)

        assert large.curve_marks == (_PEAK_MICROS, _PEAK_MICROS)


class TestTheBoundedMarkEqualsTheUnboundedFold:
    """AC2: exact equality with a brute-force fold, never a tolerance."""

    def test_a_warm_cursor_still_answers_the_whole_ledgers_maximum(
        self, tmp_path: Path
    ) -> None:
        """Fifty later samples, none near the peak, and the mark does not move.

        The peak sits at index 3 and every sample appended after the cursor
        went warm is hundreds of millions of micros below it, so a walk that
        stopped anywhere short of the remembered watermark would answer a
        different number -- and this asserts the number, against a fold of the
        store's own `read_all()`.
        """
        inner = _ledger(tmp_path / "ledger.db", _sample_micros(_TEN_N_SAMPLES))
        store = WalkCountingStore(inner)
        cursor = EquityCurveCursor()
        try:
            read_equity_curve(store, cursor=cursor)
            for index in range(_TEN_N_SAMPLES, _TEN_N_SAMPLES + _N_SAMPLES):
                _append_one(inner, _ORDINARY_BASE_MICROS + index, index)
            warm = read_equity_curve(store, cursor=cursor)
            unbounded = equity_curve_micros(inner.read_all())

            assert warm is not None
            assert warm.high_water_mark == MoneyMicros(_PEAK_MICROS)
            assert warm == unbounded
        finally:
            store.close()

    def test_the_newest_reading_is_the_newest_row(self, tmp_path: Path) -> None:
        """AC5: the O(1) half of the fold is untouched by the memo."""
        inner = _ledger(tmp_path / "ledger.db", _sample_micros(_N_SAMPLES))
        store = WalkCountingStore(inner)
        cursor = EquityCurveCursor()
        try:
            read_equity_curve(store, cursor=cursor)
            _append_one(inner, _OLDEST_MICROS, _N_SAMPLES)
            warm = read_equity_curve(store, cursor=cursor)

            assert warm is not None
            assert warm.latest == MoneyMicros(_OLDEST_MICROS)
            assert warm.high_water_mark == MoneyMicros(_PEAK_MICROS)
        finally:
            store.close()

    def test_a_new_peak_ratchets_the_mark_up(self, tmp_path: Path) -> None:
        """The memo remembers a maximum, so a higher sample must still raise it.

        The half that keeps `test_a_warm_cursor_still_answers_the_whole_ledgers
        _maximum` from passing on a cursor that simply froze its answer.
        """
        inner = _ledger(tmp_path / "ledger.db", _sample_micros(_N_SAMPLES))
        store = WalkCountingStore(inner)
        cursor = EquityCurveCursor()
        try:
            read_equity_curve(store, cursor=cursor)
            _append_one(inner, _PEAK_MICROS + 1, _N_SAMPLES)
            warm = read_equity_curve(store, cursor=cursor)

            assert warm is not None
            assert warm.high_water_mark == MoneyMicros(_PEAK_MICROS + 1)
            assert warm == equity_curve_micros(inner.read_all())
        finally:
            store.close()

    def test_a_restarted_process_recovers_the_same_mark(self, tmp_path: Path) -> None:
        """The mark is memoized per process, never persisted as a shortcut.

        A second store object opened on the same database, behind a cursor that
        has never read anything, re-folds the ledger's own rows and recovers
        exactly the peak the warm cursor was reporting.
        """
        inner = _ledger(tmp_path / "ledger.db", _sample_micros(_N_SAMPLES))
        cursor = EquityCurveCursor()
        warm = read_equity_curve(inner, cursor=cursor)
        inner.close()

        reopened = SqliteLedgerStore(tmp_path / "ledger.db")
        try:
            restarted = read_equity_curve(reopened, cursor=EquityCurveCursor())

            assert warm is not None
            assert restarted == warm
            assert restarted is not None
            assert restarted.high_water_mark == MoneyMicros(_PEAK_MICROS)
        finally:
            reopened.close()


class TestTheFallbackPathIsUnchanged:
    """A store lacking the optional capability is a real configuration."""

    def test_a_scan_only_store_folds_the_whole_ledger_and_agrees(
        self, tmp_path: Path
    ) -> None:
        """Every hand-rolled `LedgerStore` double takes this path.

        It keeps calling `read_all` -- once per read, unbounded, exactly as
        before -- and must answer the same micros as the bounded walk.
        """
        inner = _ledger(tmp_path / "ledger.db", _sample_micros(_N_SAMPLES))
        store = ScanCountingStore(inner)
        cursor = EquityCurveCursor()
        try:
            first = read_equity_curve(store, cursor=cursor)
            _append_one(inner, _ORDINARY_BASE_MICROS + _N_SAMPLES, _N_SAMPLES)
            second = read_equity_curve(store, cursor=cursor)

            assert store.read_all_calls == 2
            assert first is not None
            assert second is not None
            assert second.high_water_mark == MoneyMicros(_PEAK_MICROS)
            assert second.latest == MoneyMicros(_ORDINARY_BASE_MICROS + _N_SAMPLES)
            assert second == equity_curve_micros(inner.read_all())
        finally:
            store.close()

    def test_a_cursorless_read_is_the_pre_existing_whole_series_fold(
        self, tmp_path: Path
    ) -> None:
        """Callers that pass no cursor keep paying, and answering, exactly as before."""
        inner = _ledger(tmp_path / "ledger.db", _sample_micros(_N_SAMPLES))
        store = WalkCountingStore(inner)
        try:
            first = read_equity_curve(store)
            walked_once = store.records_walked
            second = read_equity_curve(store)

            assert walked_once == _N_SAMPLES
            assert store.records_walked - walked_once == _N_SAMPLES
            assert first == second == equity_curve_micros(inner.read_all())
        finally:
            store.close()


class TestAnUnsampledLedgerStillRefuses:
    """AC: absence stays `None`, so `trailing_drawdown_limit` keeps vetoing."""

    def test_an_empty_ledger_answers_none_cold_and_warm(self, tmp_path: Path) -> None:
        """A cursor that has folded nothing must not manufacture a reading."""
        inner = _ledger(tmp_path / "ledger.db", ())
        store = WalkCountingStore(inner)
        cursor = EquityCurveCursor()
        try:
            assert read_equity_curve(store, cursor=cursor) is None
            assert read_equity_curve(store, cursor=cursor) is None
            assert cursor.covered_through == 0
        finally:
            store.close()

    def test_the_first_sample_after_an_empty_read_is_seen(self, tmp_path: Path) -> None:
        """A cold cursor that answered `None` must not latch that refusal."""
        inner = _ledger(tmp_path / "ledger.db", ())
        store = WalkCountingStore(inner)
        cursor = EquityCurveCursor()
        try:
            assert read_equity_curve(store, cursor=cursor) is None
            _append_one(inner, _PEAK_MICROS, 0)
            curve = read_equity_curve(store, cursor=cursor)

            assert curve is not None
            assert curve.high_water_mark == MoneyMicros(_PEAK_MICROS)
            assert curve.latest == MoneyMicros(_PEAK_MICROS)
        finally:
            store.close()
