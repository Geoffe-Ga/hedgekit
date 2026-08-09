"""The PAPER tick never steps the replay cursor (issue #387, SPEC S7.5.1).

`PaperExchange.advance()` had no production caller at all, so an always-on
PAPER run replayed step 0 forever. Issue #387 settled that this is the intended
semantics rather than an oversight: a tick reads, prices, and fills against the
single recorded step the cursor stands on, and the cursor is not a function of
the beat counter. See SPEC S7.5.1 for why -- in one line, a cursor stepped per
tick moves the anchored book stamp at the recording's cadence while the clock
moves at the loop's, so every freshness and skew reading downstream would be
measuring the gap between two unrelated cadences.

`tests/connector/test_paper_replay_semantics.py` pins that at the connector.
This module pins it where it would actually be broken: by driving *real*
`run_single_tick` calls -- not the exchange, not a stage -- and reading the
answer out of the hash-chained ledger the tick itself wrote. A test that
inspected `deps.exchange` after calling a stage helper directly would step
around the very code that could regress.

The second test is the positive control, and it is what makes the first one
evidence. It runs the identical three ticks with one explicit `advance()`
wedged between the first and the second, and asserts the ledger *changes*: a
different best ask and a different snapshot instant. Without it, "all three
snapshots agree" would be equally satisfied by a fixture whose two steps carry
the same book, or by a payload that never carried the book at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.integration.conftest import (
    FIXED_NOW_EPOCH_S,
    ledger_path_for,
    read_event_type_payload_pairs,
)

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig

#: The single ticker the `deep_walk` fixture records.
_TICKER = "MKT-DEEP"

#: `deep_walk`'s recorded step spacing, in seconds: step 0 at 00:00:00 and
#: step 1 at 00:05:00. Far coarser than any beat interval, which is the whole
#: reason a per-tick advance cannot track the wall clock.
_STEP_SPACING_S = 300

#: The loop's default beat interval (`windbreak/main.py`), rounded to whole
#: seconds: how far the clock really moves between the ticks driven below.
_BEAT_INTERVAL_S = 5

#: Step 0's best YES ask, in pips (two ask levels: 4600 x 200, 4700 x 1000).
_STEP_0_BEST_ASK_PIPS = 4600

#: Step 1's best YES ask, in pips (a single ask level: 4750 x 500). Distinct
#: from step 0's, so which step a tick read is readable off the ledger.
_STEP_1_BEST_ASK_PIPS = 4750

#: How many ticks each scenario drives.
_TICKS = 3


class _BeatClock:
    """A clock that moves one beat interval when the caller says so.

    Deliberately not auto-advancing per read: `build_paper_deps` reads it to
    anchor the replay and every stage reads it again within a tick, so a clock
    that moved on each read would make "the tick's instant" meaningless and
    the anchor unpredictable.

    Attributes:
        now_epoch_s: The current reading, in whole epoch seconds.
    """

    def __init__(self, start_epoch_s: int) -> None:
        """Start the clock at ``start_epoch_s``.

        Args:
            start_epoch_s: The initial reading, in whole epoch seconds.
        """
        self.now_epoch_s = start_epoch_s

    def __call__(self) -> int:
        """Return the current reading, in whole epoch seconds."""
        return self.now_epoch_s

    def beat(self) -> None:
        """Move the clock forward by one beat interval."""
        self.now_epoch_s += _BEAT_INTERVAL_S


def _build_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    research_tools_factory,
    clock: _BeatClock,
):
    """Wire one `PaperTickDeps` over the offline `deep_walk` fixtures.

    Args:
        books_dir: The `deep_walk` books-fixture directory.
        cassette_path: The (empty) recorded-cassette path.
        ledger_path: Where the tick's `SqliteLedgerStore` is created.
        report_dir: Where weekly-report stubs are written.
        config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
        clock: The beat clock the deps -- and the replay anchor -- read.

    Returns:
        A fully wired `PaperTickDeps`.
    """
    from windbreak.scheduler.loop import build_paper_deps

    return build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
        config=config,
        research_tools=research_tools_factory(),
        clock=clock,
    )


def _snapshot_rows(records: list[object]) -> list[tuple[int | None, int]]:
    """Project the ledger into one `(best_ask_pips, fetched_at_epoch_s)` per snapshot.

    Args:
        records: The `LedgerRecord` sequence from `store.read_all()`.

    Returns:
        The ticker's market-snapshot rows, in ledger order.
    """
    return [
        (payload["best_ask_pips"], payload["fetched_at_epoch_s"])
        for event_type, payload in read_event_type_payload_pairs(records)
        if event_type == "MarketSnapshotRecorded" and payload["ticker"] == _TICKER
    ]


def test_three_ticks_all_price_against_the_recordings_first_step(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Every tick ledgers step 0's book, however far the clock has moved.

    The clock genuinely advances between ticks -- 10 seconds over the three --
    so a cursor keyed to the beat counter would be visible here as a second
    distinct snapshot. The exchange is re-read afterwards to show the same
    thing from the venue's side: nothing about the replay moved.
    """
    from windbreak.scheduler.loop import run_single_tick

    clock = _BeatClock(FIXED_NOW_EPOCH_S)
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
        clock=clock,
    )

    for beat in range(1, _TICKS + 1):
        run_single_tick(deps, beat=beat)
        clock.beat()

    deps.store.verify_chain()
    assert (
        _snapshot_rows(deps.store.read_all())
        == [(_STEP_0_BEST_ASK_PIPS, FIXED_NOW_EPOCH_S)] * _TICKS
    )
    assert clock.now_epoch_s == FIXED_NOW_EPOCH_S + _TICKS * _BEAT_INTERVAL_S
    book = deps.exchange.get_order_book(_TICKER)
    assert book.yes_asks[0].price.value == _STEP_0_BEST_ASK_PIPS
    assert int(book.fetched_at.timestamp()) == FIXED_NOW_EPOCH_S


def test_an_advance_between_ticks_is_visible_in_the_very_same_ledger(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """The positive control: stepping the cursor changes what the ledger records.

    Identical to the scenario above except for one explicit `advance()` after
    the first tick. Ticks two and three then ledger step 1's book -- a
    different price at a different instant -- which is exactly what the first
    test asserts never happens on its own. It also shows the second-order
    consequence the decision turns on: the instant ledgered for tick 2 is 295
    seconds *after* the clock reading that tick was priced at, a book from the
    future.
    """
    from windbreak.scheduler.loop import run_single_tick

    clock = _BeatClock(FIXED_NOW_EPOCH_S)
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
        clock=clock,
    )

    for beat in range(1, _TICKS + 1):
        run_single_tick(deps, beat=beat)
        clock.beat()
        if beat == 1:
            deps.exchange.advance()

    deps.store.verify_chain()
    advanced_epoch_s = FIXED_NOW_EPOCH_S + _STEP_SPACING_S
    assert _snapshot_rows(deps.store.read_all()) == [
        (_STEP_0_BEST_ASK_PIPS, FIXED_NOW_EPOCH_S),
        (_STEP_1_BEST_ASK_PIPS, advanced_epoch_s),
        (_STEP_1_BEST_ASK_PIPS, advanced_epoch_s),
    ]
    assert advanced_epoch_s > FIXED_NOW_EPOCH_S + _BEAT_INTERVAL_S
