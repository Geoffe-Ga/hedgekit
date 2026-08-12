"""The quote-freshness guard #369 added, falsified by a drifting clock (#502).

``_approve_stage`` threads the order book's own ``fetched_at`` as the quote
snapshot epoch (issue #369), and its docstring makes a strong claim about why:
feeding it this stage's ``now_epoch_s`` instead made ``_is_stale(now, now, ttl)``
``False`` for every ttl, so ``quote_freshness`` was "incapable of ever vetoing --
worse than the absent datum it replaced".

**Nothing falsified that claim.** Measured on ``origin/main`` at 23a2860 by the
issue #450 lane (PR #501): substituting ``now_epoch_s`` back in for
``quote_snapshot_epoch_s`` survived every test in ``tests/`` outside the
``e2e``/``deploy``/``docs``/``toolchain`` tiers. Not because the guard is
inert -- because every suite reaching that stage injects a *fixed* clock, and
:func:`~windbreak.scheduler.loop.build_paper_deps` anchors the replay to a
reading from that same clock. The book's ``fetched_at`` and the approval stage's
second clock read were therefore the same integer, and the two expressions were
numerically identical. A degenerate fixture, not an equivalence:
``tests/integration/test_paper_real_approval_fill.py`` says so from the other
side, under "What this scenario cannot distinguish".

What breaks the coincidence
---------------------------

A clock that genuinely **advances between the two readings**. The advance cannot
happen before the tick starts: the selector is clock-free (SPEC S9.1) and judges
freshness against ``T = max(order_book.fetched_at, forecast.created_at,
fee_model.as_of)``, so a clock already ahead of the book at tick start stamps
``forecast.created_at`` ahead of it too and the selector's own
``quote_snapshot_fresh`` blocks the intent before any approval exists to veto.

So the drift is placed where a real deployment would put it, and where
``_approve_stage``'s docstring already says the second clock read exists to
catch it: **inside the forecast stage**. :class:`_SlowVoteTransport` wraps the
same near-certain vote transport the fill scenario uses and advances
:class:`_DriftingClock` on its first call -- a vote round that took
``drift`` seconds. Everything struck before it (the screened book, the tick's
``created_at``, the fee model's ``as_of``) keeps the tick-start instant, so the
selector still admits the intent; everything read after it -- ``_approve_stage``'s
deliberate second clock read -- sees the later one.

Every test below asserts the coincidence really is broken before reading any
verdict: the ledgered ``MarketSnapshotRecorded.fetched_at_epoch_s`` is the
tick-start instant while beat 2's ``ExchangeStatusObserved.observed_at_epoch_s``
-- taken two statements before the second clock read, off the same clock -- is
the drifted one. Without that pair, a harness whose clock silently failed to
advance would read as a passing test of nothing.

Both directions, at the boundary
--------------------------------

A guard that only ever vetoes is as wrong as one that never does, so the drift
is swept across ``RiskConfig().quote_ttl_seconds`` (10):

* one second inside (9) -- no quote veto,
* exactly at the ttl (10) -- no quote veto, since ``_is_stale`` compares
  ``now - timestamp > ttl``,
* one second past it (11) -- the quote veto fires.

The ledgered ``IntentVetoed`` reasons are compared for **exact list equality**,
never membership: ``quote_freshness`` is not the only check the drift trips
(``clock_skew_max_seconds`` is 2, so the anchored replay's venue clock is out of
tolerance from 3 seconds of drift onward), and a ``match=``-style partial check
would pass on any veto at all. Exact equality is also what makes the *inside*
cases load-bearing -- they assert the quote reason is **absent** from a list that
does contain another reason, so they fail if the guard starts over-vetoing.

:func:`test_no_drift_fills_through_the_same_slow_vote_harness` is the harness's
positive control: with the drift set to zero, the identical composition reaches a
real approval and a 25 000-centi fill. Without it, "the drifted tick vetoed"
could not be told apart from "this harness never fills".

The sibling epoch fields
------------------------

``_approve_stage`` is the only production caller of
:func:`~windbreak.scheduler.loop.build_evaluation_context`, and it threads five
observation epochs onto the context. Under a fixed clock **four of the five**
were unfalsifiable for exactly the reason ``quote_snapshot_epoch_s`` was, and the
drift harness closes all four:

* ``exchange_clock_epoch_s`` (issue #377) -- killed by every case here from 3
  seconds of drift on: fed ``now_epoch_s`` the skew is 0 and the veto vanishes.
* ``forecast_epoch_s`` (issue #380) -- ttl 3600, so it needs the long-drift case
  in :func:`test_a_drift_past_every_ttl_ages_out_the_forecast_and_the_heartbeat`.
* ``pipeline_heartbeat_epoch_s`` -- ttl 60, killed by that same case.
* ``exchange_status_epoch_s`` -- **still a surviving mutant, and reported here
  rather than papered over.** It is *not* an equivalence.
  :meth:`~windbreak.connector.paper.PaperExchange.get_exchange_status` restamps
  ``fetched_at`` from the loop's own clock on every read (deliberately: a status
  records *when we observed*), and ``_approve_stage`` takes that read two
  statements before its second clock read. Nothing in a fixture-mode tick can
  advance a clock *between* two adjacent statements, so no drift this harness
  places anywhere can separate them -- but a real wall clock crossing a second
  boundary there, and
  :class:`~windbreak.connector.paper.LiveBookPaperExchange`, which passes the
  venue's own instant through untouched, both can. Killing it needs an exchange
  seam that charges latency to the status read, which is issue #342's surface
  rather than #369's; it is filed with this reproduction rather than absorbed.
"""

from __future__ import annotations

import dataclasses
import shutil
from typing import TYPE_CHECKING

import pytest

from tests.scheduler.test_paper_intent_barriers import (
    FIXED_NOW_EPOCH_S,
    FRESH_LEDGER_VETO_REASONS,
    SHIPPED_BOOKS,
    SHIPPED_CASSETTE,
    SIZED_FILL_CENTIS,
    TICKER,
    _bucketed_config,
    _finding_research_tools,
    _NearCertainVoteTransport,
    _rows,
    _tradeable_books,
    _write_track_records,
)
from windbreak.config.schema import OpsConfig, RiskConfig
from windbreak.scheduler.loop import build_paper_deps, run_single_tick

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.forecast.cassettes import Completion, LlmRequest
    from windbreak.scheduler.loop import PaperTickDeps

#: The quote ttl the drifts below are struck around, read off the shipped
#: config rather than restated, so a changed threshold moves the whole sweep
#: instead of leaving three stale literals behind.
QUOTE_TTL_SECONDS = RiskConfig().quote_ttl_seconds

#: The clock-skew tolerance, likewise derived. Every drift below exceeds it,
#: which is why `clock_skew_limit` appears in every drifted reason list.
CLOCK_SKEW_MAX_SECONDS = RiskConfig().clock_skew_max_seconds

#: The forecast ttl `windbreak.scheduler.loop` maps into `RiskLimits`
#: (`RiskConfig` carries a quote ttl but no forecast ttl).
FORECAST_TTL_SECONDS = 3600

#: The pipeline-heartbeat ttl the same mapping supplies.
PIPELINE_HEARTBEAT_TTL_SECONDS = 60

#: The span the `deep_walk` recording covers, in seconds: its two book instants
#: are five minutes apart. Past this an anchored replay has outlived its
#: recording and refuses to report a venue clock at all (issue #382).
RECORDED_SPAN_SECONDS = 300

#: A drift one second inside the quote ttl: the book is old, but not too old.
DRIFT_INSIDE_TTL_S = QUOTE_TTL_SECONDS - 1

#: A drift of exactly the quote ttl. `_is_stale` vetoes on `now - stamp > ttl`,
#: so the boundary itself is admitted -- pinned here rather than assumed.
DRIFT_AT_TTL_S = QUOTE_TTL_SECONDS

#: A drift one second past the quote ttl: the first drift that vetoes on the
#: quote, and the one the surviving mutant of issue #369 dies on.
DRIFT_PAST_TTL_S = QUOTE_TTL_SECONDS + 1

#: A drift one second past the *forecast* ttl, which is also past the heartbeat
#: ttl and past the recorded span. The sibling-field case.
DRIFT_PAST_FORECAST_TTL_S = FORECAST_TTL_SECONDS + 1

#: `_QuoteFreshness`'s verbatim veto reason.
QUOTE_STALE_REASON = "quote is stale or missing"

#: `_ClockSkewLimit`'s verbatim veto reason when the venue clock is readable
#: but out of tolerance.
CLOCK_SKEW_REASON = "clock skew exceeds limit"

#: `_ClockSkewLimit`'s verbatim veto reason once the anchored replay has
#: outlived its recording and reports no venue clock at all (issue #382).
CLOCK_UNKNOWN_REASON = "exchange clock unknown"

#: `_ForecastFreshness`'s verbatim veto reason.
FORECAST_STALE_REASON = "forecast is stale or missing"

#: `_PipelineHeartbeatOk`'s verbatim veto reason.
HEARTBEAT_STALE_REASON = "pipeline heartbeat stale or missing"

#: The three reconciliation checks' verbatim veto reasons, which the long-drift
#: case also trips: the tick's verification snapshot is taken at tick start and
#: its ttl is 3600 seconds.
VERIFICATION_STALE_REASONS = (
    "balance verification stale or missing",
    "position verification stale or missing",
    "open-order verification stale or missing",
)

#: The exact `IntentVetoed` reasons beat 2 ledgers at each swept drift, in SPEC
#: S10.3 evaluation order. Observed, not predicted -- and the *only* difference
#: between the last two rows is the quote reason, which is the whole claim.
VETO_REASONS_BY_DRIFT = {
    DRIFT_INSIDE_TTL_S: [CLOCK_SKEW_REASON],
    DRIFT_AT_TTL_S: [CLOCK_SKEW_REASON],
    DRIFT_PAST_TTL_S: [QUOTE_STALE_REASON, CLOCK_SKEW_REASON],
}


class _DriftingClock:
    """An injected epoch-second clock that advances once, mid-tick.

    Holds one instant and hands it to every reader -- the exchange's replay
    anchor and observation stamp, and the loop's own reads -- exactly as
    :data:`~tests.scheduler.test_paper_intent_barriers.FIXED_NOW_EPOCH_S` does,
    until :meth:`advance` applies whatever drift the test armed. One instant,
    two readings: that is the whole apparatus.

    Attributes:
        now: The current reading, in epoch seconds.
        pending_drift_s: The advance :meth:`advance` will apply on its next
            call, in seconds. Zero (the default) makes this an ordinary fixed
            clock.
    """

    def __init__(self) -> None:
        """Start at the shared fixed instant with no drift armed."""
        self.now = FIXED_NOW_EPOCH_S
        self.pending_drift_s = 0

    def __call__(self) -> int:
        """Return the current reading.

        Returns:
            The clock's instant, in epoch seconds.
        """
        return self.now

    def advance(self) -> None:
        """Apply the armed drift, once, and disarm.

        Idempotent by construction: the drift is consumed, so the three vote
        calls one forecast makes advance the clock once between them rather
        than three times.
        """
        self.now += self.pending_drift_s
        self.pending_drift_s = 0


class _SlowVoteTransport:
    """The near-certain vote transport, with the drift charged to its first call.

    Composes
    :class:`~tests.scheduler.test_paper_intent_barriers._NearCertainVoteTransport`
    rather than re-inventing a vote body, so the votes this suite's forecasts are
    built from are byte-identical to the ones the fill scenario uses and the only
    difference between the two compositions is how long the round took.
    """

    def __init__(self, clock: _DriftingClock) -> None:
        """Bind the transport to the clock its latency is charged to.

        Args:
            clock: The drifting clock advanced on the first completion.
        """
        self._clock = clock
        self._inner = _NearCertainVoteTransport()

    def complete(self, request: LlmRequest) -> Completion:
        """Advance the clock by the armed drift, then answer the vote.

        Args:
            request: The completion request, forwarded unmodified.

        Returns:
            The inner transport's own near-certain vote.
        """
        self._clock.advance()
        return self._inner.complete(request)


def _drifting_deps(tmp_path: Path, clock: _DriftingClock) -> PaperTickDeps:
    """Wire the shipped PAPER composition over a clock that can drift.

    Identical to ``tests/integration/test_paper_real_approval_fill.py``'s
    ``_emitting_deps`` in every respect but two: the injected clock is
    :class:`_DriftingClock` rather than a constant, and the vote transport is
    :class:`_SlowVoteTransport` rather than the bare near-certain one. No
    threshold moves; that suite's
    ``test_only_the_correlation_declaration_and_state_dir_leave_the_defaults``
    derives that claim for the same configuration.

    Args:
        tmp_path: pytest's per-test temporary directory.
        clock: The drifting clock the exchange and the loop both read.

    Returns:
        The wired dependencies, over a fresh ledger.
    """
    books = tmp_path / "books"
    shutil.copytree(SHIPPED_BOOKS, books)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    _tradeable_books(books)
    _write_track_records(report_dir)
    deps = build_paper_deps(
        books_dir=books,
        cassette_path=SHIPPED_CASSETTE,
        ledger_path=tmp_path / "ledger.db",
        report_dir=report_dir,
        config=dataclasses.replace(
            _bucketed_config(), ops=OpsConfig(state_dir=str(tmp_path / "state"))
        ),
        research_tools=_finding_research_tools(tmp_path / "cache"),
        clock=clock,
    )
    return dataclasses.replace(deps, transport=_SlowVoteTransport(clock))


def _payloads(deps: PaperTickDeps, event: str) -> list[dict[str, object]]:
    """Return every ledgered payload of one event type, in ledger order.

    Args:
        deps: The wired dependencies whose store is read.
        event: The event type wanted.

    Returns:
        That event's ``data`` payloads.
    """
    return [data for event_type, data in _rows(deps) if event_type == event]


def _nth_index(events: list[str], event: str, nth: int) -> int:
    """Return the ledger position of the ``nth`` occurrence of ``event``.

    Args:
        events: The ledgered event types, in sequence order.
        event: The event type wanted.
        nth: Which occurrence, counting from one.

    Returns:
        That occurrence's index.

    Raises:
        AssertionError: If the ledger holds fewer than ``nth`` of them, which
            would otherwise make every ordering claim below vacuous.
    """
    positions = [index for index, name in enumerate(events) if name == event]
    assert len(positions) >= nth, f"expected >= {nth} {event}, got {len(positions)}"
    return positions[nth - 1]


def _assert_book_and_stage_instants(deps: PaperTickDeps, *, drift_s: int) -> None:
    """Assert beat 2 struck its book and its approval clock exactly ``drift_s`` apart.

    The guard against this whole module degenerating back into the fixture it
    exists to replace. The book's instant is read from beat 2's own
    ``MarketSnapshotRecorded``, and the approval stage's timeline from beat 2's
    ``ExchangeStatusObserved`` -- restamped from the loop's clock two statements
    before ``_approve_stage``'s second clock read, so it is that read's instant.
    A harness whose clock silently failed to advance fails here rather than
    reading as a passing test of nothing.

    Args:
        deps: The wired dependencies whose ledger is read.
        drift_s: The drift the vote transport was armed with.

    Raises:
        AssertionError: If either beat is missing its row, or if the gap between
            the two instants is not exactly ``drift_s``.
    """
    snapshots = _payloads(deps, "MarketSnapshotRecorded")
    statuses = _payloads(deps, "ExchangeStatusObserved")
    assert len(snapshots) == 2
    assert len(statuses) == 2
    assert snapshots[1]["fetched_at_epoch_s"] == FIXED_NOW_EPOCH_S
    assert statuses[1]["observed_at_epoch_s"] == FIXED_NOW_EPOCH_S + drift_s


def _run_two_beats(deps: PaperTickDeps, clock: _DriftingClock, drift_s: int) -> int:
    """Run beat 1 undrifted, then beat 2 with ``drift_s`` charged to its votes.

    Beat 1 is deliberately undrifted and deliberately present. It vetoes on
    ``daily_loss_limit`` and ``trailing_drawdown_limit`` alike -- a fresh
    ledger carries no ``EquitySampled`` row, so the day has no equity baseline
    and the account has no high-water mark -- and it ledgers the sample beat 2
    measures against. Without it, every beat-2 reason list below would carry
    both transients too and the sweep would be reading three effects at once.

    Args:
        deps: The wired dependencies.
        clock: The drifting clock, armed between the beats.
        drift_s: The drift charged to beat 2's vote round, in seconds.

    Returns:
        Beat 2's filled quantity, in contract-centis.
    """
    run_single_tick(deps, beat=1)
    clock.pending_drift_s = drift_s
    return run_single_tick(deps, beat=2).filled_centis


def test_the_swept_drifts_straddle_the_shipped_quote_ttl() -> None:
    """The three swept drifts really are ttl-1, ttl and ttl+1 around 10 seconds.

    Pinned before any of them is used, because the sweep's whole claim is about
    a boundary: a ttl that quietly moved would leave three drifts that all sit
    on the same side of it and a suite that still passed. The skew tolerance is
    pinned for the same reason -- every drifted reason list below contains
    ``clock_skew_limit`` precisely because 9 exceeds 2.
    """
    assert QUOTE_TTL_SECONDS == 10
    assert CLOCK_SKEW_MAX_SECONDS == 2
    assert (DRIFT_INSIDE_TTL_S, DRIFT_AT_TTL_S, DRIFT_PAST_TTL_S) == (9, 10, 11)
    assert sorted(VETO_REASONS_BY_DRIFT) == [9, 10, 11]
    assert DRIFT_PAST_FORECAST_TTL_S > FORECAST_TTL_SECONDS
    assert DRIFT_PAST_FORECAST_TTL_S > PIPELINE_HEARTBEAT_TTL_SECONDS
    assert DRIFT_PAST_FORECAST_TTL_S > RECORDED_SPAN_SECONDS


@pytest.mark.parametrize("drift_s", sorted(VETO_REASONS_BY_DRIFT))
def test_the_quote_veto_fires_only_past_the_ttl_under_a_drifting_clock(
    tmp_path: Path, drift_s: int
) -> None:
    """A book one second past its ttl vetoes; at the ttl and inside it, it does not.

    Issue #502's acceptance criteria 1 and 2, through the real
    :func:`~windbreak.scheduler.loop.run_single_tick` over the shipped
    composition -- nothing here calls ``_approve_stage``,
    ``build_evaluation_context`` or ``deps.approval.decide``, so the guard is
    entered by production code or not at all.

    The reasons are compared as an exact list, in SPEC S10.3 evaluation order.
    That is what makes all three rows load-bearing at once: the two inside the
    ttl assert the quote reason is *absent* from a list that is not empty, and
    the one past it asserts it is *present* and first. Substituting the stage's
    own ``now_epoch_s`` for the book's ``fetched_at`` collapses the third list
    onto the first two, which is precisely the mutant that survived the whole
    suite before this module existed.

    Args:
        tmp_path: pytest's per-test temporary directory.
        drift_s: The drift charged to beat 2's vote round, in seconds.
    """
    clock = _DriftingClock()
    deps = _drifting_deps(tmp_path, clock)

    filled = _run_two_beats(deps, clock, drift_s)

    _assert_book_and_stage_instants(deps, drift_s=drift_s)
    vetoes = _payloads(deps, "IntentVetoed")
    assert len(vetoes) == 2
    assert vetoes[0]["reasons"] == FRESH_LEDGER_VETO_REASONS
    assert vetoes[1]["reasons"] == VETO_REASONS_BY_DRIFT[drift_s]
    # The veto really stopped the order: nothing was reserved, no token minted,
    # and nothing filled on either beat.
    assert filled == 0
    assert _payloads(deps, "ApprovalTokenIssued") == []
    assert _payloads(deps, "ReservationCreated") == []
    assert _payloads(deps, "IntentApproved") == []
    deps.store.verify_chain()


def test_no_drift_fills_through_the_same_slow_vote_harness(tmp_path: Path) -> None:
    """With the drift set to zero, the identical composition approves and fills.

    The positive control the sweep above is worthless without. Every other test
    in this module asserts a veto, and a harness that vetoed for some reason of
    its own -- a vote transport the pipeline rejected, a clock object the
    exchange could not read -- would satisfy all of them while proving nothing.
    Here the same :class:`_SlowVoteTransport` and the same
    :class:`_DriftingClock` are wired with ``pending_drift_s`` left at zero, and
    beat 2 reaches a real approval and a 25 000-centi fill.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    clock = _DriftingClock()
    deps = _drifting_deps(tmp_path, clock)

    filled = _run_two_beats(deps, clock, 0)

    _assert_book_and_stage_instants(deps, drift_s=0)
    assert clock.now == FIXED_NOW_EPOCH_S
    assert filled == SIZED_FILL_CENTIS
    # Beat 1's fresh-ledger transient is the only veto across both beats.
    assert [payload["reasons"] for payload in _payloads(deps, "IntentVetoed")] == [
        FRESH_LEDGER_VETO_REASONS
    ]
    assert [payload["reasons"] for payload in _payloads(deps, "IntentApproved")] == [[]]
    assert len(_payloads(deps, "ApprovalTokenIssued")) == 1
    assert _payloads(deps, "PositionsSnapshotRecorded")[-1]["positions"] != []
    deps.store.verify_chain()


def test_a_drift_past_every_ttl_ages_out_the_forecast_and_the_heartbeat(
    tmp_path: Path,
) -> None:
    """The three sibling epoch fields #502 asks about, falsified by the same lever.

    ``_approve_stage`` threads five observation epochs, and the module docstring
    explains why four of them were unfalsifiable under a fixed clock for exactly
    the reason ``quote_snapshot_epoch_s`` was. One drift past the longest ttl
    closes the three that a ten-second one cannot reach, and the exact reason
    list is what kills each substitution:

    * ``forecast_epoch_s`` (issue #380) -- the forecast is stamped with the
      tick's own ``created_at``, so a vote round of 3 601 seconds ages it past
      its 3 600-second ttl. Fed the stage's ``now_epoch_s`` it would be zero
      seconds old and this reason would vanish.
    * ``pipeline_heartbeat_epoch_s`` -- stamped by ``_heartbeat_stage`` at tick
      start, so the same drift ages it past its 60-second ttl.
    * ``exchange_clock_epoch_s`` (issue #377) -- the drift also carries the run
      past the recorded span, so the anchored replay refuses to report a venue
      clock at all (issue #382) and the reason changes from
      :data:`CLOCK_SKEW_REASON` to :data:`CLOCK_UNKNOWN_REASON`. That the two
      differ is itself the evidence the venue clock is being read rather than
      the local one.

    The three reconciliation reasons are here because the tick's verification
    snapshot is taken at tick start and shares the 3 600-second ttl -- observed
    and recorded, not sought.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    clock = _DriftingClock()
    deps = _drifting_deps(tmp_path, clock)

    filled = _run_two_beats(deps, clock, DRIFT_PAST_FORECAST_TTL_S)

    _assert_book_and_stage_instants(deps, drift_s=DRIFT_PAST_FORECAST_TTL_S)
    vetoes = _payloads(deps, "IntentVetoed")
    assert len(vetoes) == 2
    assert vetoes[1]["reasons"] == [
        *VERIFICATION_STALE_REASONS,
        QUOTE_STALE_REASON,
        FORECAST_STALE_REASON,
        CLOCK_UNKNOWN_REASON,
        HEARTBEAT_STALE_REASON,
    ]
    assert filled == 0
    assert _payloads(deps, "ApprovalTokenIssued") == []
    deps.store.verify_chain()


def test_the_stale_quote_is_the_one_this_tick_was_priced_against(
    tmp_path: Path,
) -> None:
    """The drifted veto names the intent struck on the book that went stale.

    A one-market fixture makes it easy to assert *a* veto without ever
    establishing *which* observation went stale. Three ties close that gap:
    both beats snapshot :data:`TICKER`, the vetoed intent's id carries beat 2's
    own ``ForecastCreated`` id (the selector mints ``<forecast_id>:...``), and
    that veto row is ledgered after beat 2's snapshot rather than before it.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    clock = _DriftingClock()
    deps = _drifting_deps(tmp_path, clock)

    _run_two_beats(deps, clock, DRIFT_PAST_TTL_S)

    events = [event for event, _ in _rows(deps)]
    tickers = [
        payload["ticker"] for payload in _payloads(deps, "MarketSnapshotRecorded")
    ]
    forecasts = _payloads(deps, "ForecastCreated")
    vetoes = _payloads(deps, "IntentVetoed")
    assert tickers == [TICKER, TICKER]
    assert len(forecasts) == 2
    assert forecasts[1]["market_ticker"] == TICKER
    assert vetoes[1]["intent_id"] == f"{forecasts[1]['forecast_id']}:yes:buy:sized"
    second_snapshot = _nth_index(events, "MarketSnapshotRecorded", 2)
    assert _nth_index(events, "IntentVetoed", 2) > second_snapshot
