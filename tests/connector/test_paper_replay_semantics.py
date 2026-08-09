"""Pins the PAPER replay cursor as *stationary* in production (issue #387).

Issue #387 is a decision, not a bug: nothing in `windbreak/` ever called
`PaperExchange.advance()`, so the always-on PAPER loop sat on replay step 0
forever -- and nobody had decided whether that was the intent or an oversight.
The decision recorded in SPEC S7.5.1 is **frozen**: a tick prices, fills, and
allocates against the one recorded step the cursor stands on, and the cursor is
not a function of the tick counter.

This module pins the three halves of that answer that live in the connector:

1. **The SPEC says so.** S7.5.1 exists and states the normative rule. A code
   change flipping the answer without moving the SPEC leaves the two disagreeing
   and this suite red.
2. **No production module steps the cursor.** `advance()` is harness API --
   the fill-model suites drive the recorded tape through it -- so the guard is
   over `windbreak/`, not over the loop alone. Wiring it into *any* production
   caller fails here, not just wiring it into `run_single_tick`.
3. **The rejected reading would misdate the book against the clock.** A cursor
   stepped once per tick moves the (anchored) book stamp by the recording's own
   step spacing while the wall clock moves by the tick interval. Those are
   unrelated rates, so `quote_freshness` would stop measuring staleness and
   start measuring the mismatch between them -- vetoing a book dated *after*
   the instant it is being priced at. That is the concrete harm the decision
   turns on, so it is asserted rather than merely asserted-about in prose.

The trailing-step class is issue #387's acceptance criterion 3 in test form: a
frozen cursor still *uses* the recording's later steps, because their instants
are what bound the span the replay can substantiate a venue clock over (issue
#382). Truncating the tape shortens that span, which is what makes "the format
keeps carrying them" an answer rather than an excuse.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from windbreak.connector import paper
from windbreak.connector.freshness import is_fresh

#: The repository root, reached from `tests/connector/`.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The normative SPEC section this decision is recorded in.
_SPEC_PATH = _REPO_ROOT / "plans" / "SPEC_v3.md"

#: The production package the "no caller" guard scans.
_PACKAGE_ROOT = _REPO_ROOT / "windbreak"

#: A zero-argument `.advance()` call. `PaperExchange.advance` is the only
#: no-argument `advance` in the codebase, so this matches its call sites and
#: not `verification.py`'s `_advance(event)` / `_advanced(...)` helpers.
_ADVANCE_CALL = re.compile(r"\.advance\(\s*\)")

#: The instant every anchored replay in this module is re-enacted from.
_ANCHOR = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)

#: `deep_walk`'s recorded step spacing: step 0 at 00:00:00, step 1 at 00:05:00.
_STEP_SPACING = timedelta(minutes=5)

#: A tick cadence far finer than the recording's step spacing -- the ordinary
#: case, since a recorded tape is sampled in minutes and the loop beats in
#: seconds. It is what makes an advanced book *future*-dated.
_TICK_INTERVAL = timedelta(seconds=5)


def _deep_walk(
    books_fixture_dir: Path, *, now: datetime, anchored: bool = True
) -> paper.PaperExchange:
    """Build the two-step `deep_walk` replay against a fixed observation clock.

    Args:
        books_fixture_dir: The shared books-fixture root.
        now: The instant the exchange's own clock reports, fixed for the test.
        anchored: Whether to re-enact the recording from :data:`_ANCHOR`.

    Returns:
        The loaded paper exchange, positioned at step 0.
    """
    return paper.PaperExchange.from_fixture_dir(
        books_fixture_dir / "deep_walk",
        clock=lambda: now,
        replay_anchor=_ANCHOR if anchored else None,
    )


def _truncated_deep_walk(
    books_fixture_dir: Path, tmp_path: Path, *, steps: int, now: datetime
) -> paper.PaperExchange:
    """Build a `deep_walk` replay keeping only its first ``steps`` steps.

    Args:
        books_fixture_dir: The shared books-fixture root.
        tmp_path: The pytest scratch directory the copy is made in.
        steps: How many leading recorded steps to keep.
        now: The instant the exchange's own clock reports, fixed for the test.

    Returns:
        The loaded paper exchange over the shortened recording.
    """
    directory = tmp_path / f"deep_walk_{steps}"
    shutil.copytree(books_fixture_dir / "deep_walk", directory)
    sessions_path = directory / "sessions.json"
    recorded = json.loads(sessions_path.read_text(encoding="utf-8"))
    sessions_path.write_text(
        json.dumps({ticker: tape[:steps] for ticker, tape in recorded.items()}),
        encoding="utf-8",
    )
    return paper.PaperExchange.from_fixture_dir(
        directory, clock=lambda: now, replay_anchor=_ANCHOR
    )


class TestTheDecisionIsWrittenDown:
    """SPEC S7.5.1 records the answer, and no production caller contradicts it."""

    def test_spec_section_7_5_1_declares_the_cursor_stationary(self) -> None:
        """S7.5.1 exists and states the normative rule in so many words.

        The section heading alone would be satisfied by a section saying the
        opposite, so the two load-bearing words are asserted too: the cursor is
        *stationary*, and it is not advanced *per tick*.
        """
        spec = _SPEC_PATH.read_text(encoding="utf-8")
        heading = "### 7.5.1 Replay cursor semantics (normative)"

        assert heading in spec
        section = spec.split(heading, 1)[1].split("\n### ", 1)[0]
        assert "stationary" in section
        assert "never advances the replay cursor" in section

    def test_no_production_module_steps_the_replay_cursor(self) -> None:
        """`advance()` has no caller anywhere under `windbreak/`.

        The scan covers the whole package rather than `scheduler/loop.py`
        alone, because "the loop does not advance the tape" is not the claim --
        "production does not" is. It excludes only the definition site.
        """
        definition = _PACKAGE_ROOT / "connector" / "paper.py"
        callers = sorted(
            module.relative_to(_REPO_ROOT).as_posix()
            for module in _PACKAGE_ROOT.rglob("*.py")
            if module != definition
            and _ADVANCE_CALL.search(module.read_text(encoding="utf-8"))
        )

        assert callers == []

    def test_the_definition_site_itself_holds_no_self_call(self) -> None:
        """`paper.py` defines `advance()` and never calls it either.

        The guard above exempts the defining module, so this closes the one
        file that exemption opens: a `self.advance()` inside, say,
        `get_order_book` would step the tape on every production read and the
        scan would not see it.
        """
        source = (_PACKAGE_ROOT / "connector" / "paper.py").read_text(encoding="utf-8")

        assert "def advance(self)" in source
        assert not _ADVANCE_CALL.search(source)


class TestTrailingStepsBoundTheSubstantiatedSpan:
    """A stationary cursor still uses the tape's later steps (issue #387 AC3)."""

    def test_the_full_recording_answers_inside_its_trailing_gap(
        self, books_fixture_dir: Path
    ) -> None:
        """299 seconds in -- one second short of the last recorded book -- the
        two-step recording still substantiates the venue clock.
        """
        now = _ANCHOR + _STEP_SPACING - timedelta(seconds=1)
        exchange = _deep_walk(books_fixture_dir, now=now)

        assert exchange.get_exchange_time() == _ANCHOR

    def test_dropping_the_trailing_step_refuses_at_that_same_instant(
        self, books_fixture_dir: Path, tmp_path: Path
    ) -> None:
        """The same instant, over the same tape minus its last step, refuses.

        This is the difference the two tests exist to show: the *only* thing
        that changed is a step the stationary cursor never reads, and the venue
        clock's readability changed with it. A recording format that stopped
        carrying its trailing steps would silently extend nothing and shorten
        the span every anchored run is measured against.
        """
        now = _ANCHOR + _STEP_SPACING - timedelta(seconds=1)
        exchange = _truncated_deep_walk(books_fixture_dir, tmp_path, steps=1, now=now)

        with pytest.raises(paper.ReplayExhaustedError) as excinfo:
            exchange.get_exchange_time()

        assert type(excinfo.value) is paper.ReplayExhaustedError
        assert str(excinfo.value) == (
            "the run has outlived its recording, which observed no instant "
            f"past {_ANCHOR.isoformat()}"
        )

    def test_the_truncated_recording_still_answers_before_its_own_end(
        self, books_fixture_dir: Path, tmp_path: Path
    ) -> None:
        """The one-step tape is not simply always exhausted.

        Without this, the test above would pass for a truncation that broke the
        recording outright rather than for the span shortening by exactly the
        trailing gap.
        """
        exchange = _truncated_deep_walk(
            books_fixture_dir, tmp_path, steps=1, now=_ANCHOR
        )

        assert exchange.get_exchange_time() == _ANCHOR


class TestAnOverAdvancedCursorClampsToTheLastStep:
    """What `advance()` past the end means -- reachable only from a harness."""

    def test_reading_a_book_past_the_last_step_serves_the_last_step(
        self, books_fixture_dir: Path
    ) -> None:
        """A cursor run off the end keeps answering with the final recorded book.

        `_current_step` clamps to the last index, and under the stationary
        decision nothing in production can reach that clamp: only `advance()`
        moves the cursor and no production caller exists. It is still the
        connector's answer to a harness that over-steps, so it is pinned here
        rather than left to an `IndexError` nobody has stated. Found because
        removing the clamp altogether left the whole existing suite green.
        """
        exchange = _deep_walk(books_fixture_dir, now=_ANCHOR)
        exchange.advance()
        assert exchange.advance() is False

        book = exchange.get_order_book("MKT-DEEP")

        assert book.yes_asks[0].price.value == 4750
        assert book.fetched_at == _ANCHOR + _STEP_SPACING


class TestAdvancingPerTickWouldMisdateTheBook:
    """Why the rejected reading is rejected: it breaks `quote_freshness`."""

    def test_the_stationary_book_ages_forward_from_the_anchor(
        self, books_fixture_dir: Path
    ) -> None:
        """A frozen step's book is exactly as old as the run is long.

        Fresh at the ttl boundary, stale one second past it: the age
        `quote_freshness` measures is the run's own elapsed time, which is a
        real quantity about a real recording -- "this tape observed the venue
        once, and you are this far from that observation".
        """
        ttl_seconds = 10
        at_boundary = _ANCHOR + timedelta(seconds=ttl_seconds)
        book = _deep_walk(books_fixture_dir, now=at_boundary).get_order_book("MKT-DEEP")

        assert book.fetched_at == _ANCHOR
        assert is_fresh(book.fetched_at, ttl_seconds=ttl_seconds, now=at_boundary)
        assert not is_fresh(
            book.fetched_at,
            ttl_seconds=ttl_seconds,
            now=at_boundary + timedelta(seconds=1),
        )

    def test_a_per_tick_advance_dates_the_book_after_the_tick_pricing_it(
        self, books_fixture_dir: Path
    ) -> None:
        """One step per tick outruns the wall clock, and freshness cannot help.

        The recording samples every 5 minutes; the loop beats every 5 seconds.
        A cursor stepped once per tick therefore hands tick 2 a book stamped
        295 seconds *after* the instant tick 2 is pricing at -- and a
        future-dated quote is refused for any ttl at all, however generous.
        `quote_freshness` would then be reporting the gap between two cadences,
        not the age of a price.
        """
        tick_two = _ANCHOR + _TICK_INTERVAL
        exchange = _deep_walk(books_fixture_dir, now=tick_two)
        exchange.advance()  # the rejected reading, applied once

        book = exchange.get_order_book("MKT-DEEP")

        assert book.fetched_at == _ANCHOR + _STEP_SPACING
        assert book.fetched_at > tick_two
        assert not is_fresh(book.fetched_at, ttl_seconds=86_400, now=tick_two)

    def test_the_two_steps_are_distinguishable_books(
        self, books_fixture_dir: Path
    ) -> None:
        """The fixture's two steps differ in price *and* in instant.

        The positive control for every assertion above and for the loop-level
        suite: were the two steps' books identical, "the cursor never moved"
        and "the cursor moved" would produce the same observation and nothing
        here would be measuring anything.
        """
        exchange = _deep_walk(books_fixture_dir, now=_ANCHOR)
        first = exchange.get_order_book("MKT-DEEP")
        exchange.advance()
        second = exchange.get_order_book("MKT-DEEP")

        assert first.yes_asks[0].price.value == 4600
        assert second.yes_asks[0].price.value == 4750
        assert second.fetched_at - first.fetched_at == _STEP_SPACING
