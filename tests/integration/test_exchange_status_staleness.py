"""``exchange_status_ok``'s staleness half, falsified by a latent venue (#506).

``_approve_stage`` threads five observation epochs onto the evaluation context.
Issue #502 / PR #505 closed four of them with a drifting clock
(``tests/integration/test_quote_freshness_drift.py``). The fifth --
``exchange_status_epoch_s``, the datum issue #342 added and SPEC S7.3 requires --
survived: substituting the stage's own ``now_epoch_s`` for it left all 29 tests
in that module, ``tests/integration/test_paper_real_approval_fill.py`` and
``tests/scheduler/test_paper_intent_barriers.py`` green, so the check's staleness
half was measured by nothing.

Why a drifting clock cannot reach it
------------------------------------

Not an equivalence -- a structural blind spot, and it lives in *production*
rather than in a fixture.
:meth:`~windbreak.connector.paper.PaperExchange.get_exchange_status` restamps
``fetched_at`` from the loop's own clock on every read, and ``_approve_stage``
takes that read two statements before its second clock read. Under any clock the
loop injects, the status epoch and the stage's clock read are therefore equal *by
construction*: no drift placed anywhere can separate two adjacent statements.
:func:`test_the_fixture_exchange_cannot_express_a_stale_status` pins that as an
executable claim rather than leaving it as prose, so the blind spot is
falsifiable too.

The restamping is correct and is not touched here. A committed fixture's status
carries a frozen literal while an in-process answer genuinely happened now; the
:meth:`~windbreak.connector.paper.PaperExchange.get_exchange_time` docstring
explains at length (issue #377) why the *venue clock* must not be restamped the
same way. Nothing in ``windbreak/`` changes for this module.

The seam that does reach it already exists
------------------------------------------

:meth:`~windbreak.connector.paper.LiveBookPaperExchange.get_exchange_status`
returns the venue's observation **verbatim** -- built by issue #343 precisely
because "renewing it would make ``exchange_status_ok``'s staleness bound
unfalsifiable". That is the production seam this module drives, reached through
``build_paper_deps``'s shipped ``market_data``/``live_ticker`` pair. No new
production surface was needed and none was added; the gap was that nothing had
ever pointed a stale status at it.

The honest model: latency on the status read
--------------------------------------------

On a live venue ``get_exchange_status`` is an HTTP round trip, so the instant the
venue observed its status is necessarily *older* than the clock reading taken
after the answer arrives. :class:`_LatentVenue` is exactly that and nothing more:
every market read delegates to the same shipped ``deep_walk`` fixture the fill
scenario uses, and only the status carries an instant
:data:`_LatentVenue.latency_s` seconds behind the run clock. The clock itself is
the ordinary fixed one, so nothing else in the tick moves -- the book, the
forecast, the heartbeat and the venue clock all stay fresh, and the swept reason
lists below contain the status reason or nothing at all.

Both directions, at the boundary
--------------------------------

``_is_stale`` vetoes on ``now - timestamp > ttl``, so the ttl itself is
admissible. The sweep straddles it, and the two fresh rows are not near-misses:
they reach a real approval and a 25 000-centi fill.

* one second inside (29 s) -- approved, empty reason list, fills;
* exactly at the ttl (30 s) -- approved, empty reason list, fills;
* one second past it (31 s) -- vetoed on exactly ``["exchange status stale or
  missing"]`` and nothing else, and fills nothing.

Reason lists are compared for exact equality, never membership. Because the
latency is charged only to the status read, the vetoing row's list has exactly
one entry, which is the strongest form this evidence can take: the guard under
test is the *only* thing that fired.
"""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from tests.scheduler.test_paper_intent_barriers import (
    DAILY_LOSS_VETO_REASON,
    FIXED_NOW_EPOCH_S,
    SHIPPED_BOOKS,
    SHIPPED_CASSETTE,
    SIZED_FILL_CENTIS,
    TICKER,
    _bucketed_config,
    _finding_research_tools,
    _fixed_clock,
    _fixed_now,
    _NearCertainVoteTransport,
    _rows,
    _tradeable_books,
    _write_track_records,
)
from windbreak.config.schema import OpsConfig
from windbreak.connector.paper import LiveBookPaperExchange, PaperExchange
from windbreak.scheduler.loop import _build_limits, build_paper_deps, run_single_tick

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from windbreak.connector.models import (
        ExchangeStatus,
        FeeModel,
        NormalizedMarket,
        OrderBookSnapshot,
    )
    from windbreak.scheduler.loop import PaperTickDeps

#: The exchange-status ttl the loop actually wires, read out of the production
#: config-to-limits mapping rather than restated as a literal. A threshold that
#: moved would move the whole sweep below instead of leaving three stale numbers
#: behind.
STATUS_TTL_SECONDS = _build_limits(
    _bucketed_config(),
    frozenset({TICKER}),
    exposure_provable=True,
    notional_provable=True,
    orders_provable=True,
).exchange_status_ttl_seconds

#: A round trip one second inside the ttl: the status is old, but not too old.
LATENCY_INSIDE_TTL_S = STATUS_TTL_SECONDS - 1

#: A round trip of exactly the ttl. ``_is_stale`` vetoes on ``now - stamp >
#: ttl``, so the boundary itself is admitted -- pinned, not assumed.
LATENCY_AT_TTL_S = STATUS_TTL_SECONDS

#: A round trip one second past the ttl: the first latency that vetoes, and the
#: one the surviving mutant of issue #342 dies on.
LATENCY_PAST_TTL_S = STATUS_TTL_SECONDS + 1

#: ``_ExchangeStatusOk``'s verbatim veto reason.
STATUS_STALE_REASON = "exchange status stale or missing"

#: The exact ``IntentVetoed`` reasons beat 2 ledgers at each swept latency, and
#: the empty list beat 2 *approves* with. Observed, not predicted. The only
#: difference between the last row and the two before it is the status reason,
#: which is the entire claim of this module.
APPROVAL_REASONS_BY_LATENCY: dict[int, list[str]] = {
    LATENCY_INSIDE_TTL_S: [],
    LATENCY_AT_TTL_S: [],
    LATENCY_PAST_TTL_S: [STATUS_STALE_REASON],
}

#: The quantity beat 2 fills, in contract-centis, when the status is fresh.
#: Shared with the fill scenario so the two harnesses cannot silently diverge.
FRESH_FILL_CENTIS = SIZED_FILL_CENTIS


class _LatentVenue:
    """The shipped fixture venue, answering its status one round trip late.

    A :class:`~windbreak.connector.live.MarketDataSource` and nothing more. Four
    of the five reads delegate verbatim to a
    :class:`~windbreak.connector.paper.PaperExchange` loaded from the very same
    ``deep_walk`` directory and anchored to the very same instant the fixture
    path uses, so the books, the market document and the fee schedule this
    module's tick reasons about are byte-identical to
    ``tests/integration/test_paper_real_approval_fill.py``'s. That is what makes
    the swept verdicts attributable to the status alone.

    The fifth read is the whole apparatus. ``get_exchange_status`` reports the
    venue's own observation instant as :attr:`latency_s` seconds before the run
    clock -- what a venue that answered an HTTP request that long ago is, by
    definition, reporting. It is deliberately *not* modelled as a mutable clock
    the read advances: a fixed clock leaves every other observation in the tick
    fresh, so nothing but the status can move the verdict.

    ``get_exchange_time`` returns the run clock, modelling a venue whose clock
    agrees with ours. That keeps ``clock_skew_limit`` -- which fires two checks
    before ``exchange_status_ok`` in SPEC S10.3 order -- out of every reason list
    below, and the venue-clock guard is measured by issue #377's own suite
    rather than borrowed here.

    Attributes:
        latency_s: The status read's round-trip time, in seconds.
    """

    def __init__(self, fixture: PaperExchange, *, latency_s: int) -> None:
        """Bind a latent status read onto the shipped fixture's market data.

        Args:
            fixture: The anchored fixture exchange every market read delegates
                to.
            latency_s: The status read's round-trip time, in seconds.
        """
        self._fixture = fixture
        self.latency_s = latency_s

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Return the fixture's market document, unmodified.

        Args:
            ticker: The market to look up.

        Returns:
            The fixture's normalized market.
        """
        return self._fixture.get_market(ticker)

    def get_order_book(self, ticker: str) -> OrderBookSnapshot:
        """Return the fixture's anchored book, unmodified.

        Args:
            ticker: The market to look up.

        Returns:
            The fixture's order-book snapshot, carrying the anchored instant.
        """
        return self._fixture.get_order_book(ticker)

    def get_fee_model(self, market_or_series: str) -> FeeModel:
        """Return the fixture's fee schedule, unmodified.

        Args:
            market_or_series: A market ticker or bare series key.

        Returns:
            The fixture's fee model.
        """
        return self._fixture.get_fee_model(market_or_series)

    def get_exchange_time(self) -> datetime:
        """Return a venue clock agreeing exactly with the run clock.

        Returns:
            The run's fixed instant.
        """
        return _fixed_now()

    def get_exchange_status(self) -> ExchangeStatus:
        """Return the fixture's status, observed one round trip ago.

        The ``status`` value comes verbatim from the fixture -- an ``open``
        venue, so nothing but the *age* of the observation is under test.

        Returns:
            The fixture's status, stamped :attr:`latency_s` seconds before the
            run clock.
        """
        return replace(
            self._fixture.exchange_status,
            fetched_at=_fixed_now() - timedelta(seconds=self.latency_s),
        )


def _anchored_fixture(books: Path) -> PaperExchange:
    """Load the fixture exchange the live venue's market reads delegate to.

    Built exactly as :func:`~windbreak.scheduler.loop._build_paper_exchange`
    builds the fixture path's own exchange -- same clock, same replay anchor --
    so a book this module screens is the same book the fixture-mode suites
    screen.

    Args:
        books: The rewritten ``deep_walk`` directory.

    Returns:
        The anchored fixture exchange.
    """
    return PaperExchange.from_fixture_dir(
        books, clock=_fixed_now, replay_anchor=_fixed_now()
    )


def _latent_deps(tmp_path: Path, *, latency_s: int) -> PaperTickDeps:
    """Wire the shipped live-book composition over a venue with a latent status.

    Identical to ``tests/integration/test_paper_real_approval_fill.py``'s
    ``_emitting_deps`` in every respect but one: ``market_data``/``live_ticker``
    are supplied, so ``build_paper_deps`` returns a
    :class:`~windbreak.connector.paper.LiveBookPaperExchange` that passes the
    venue's status instant through untouched. No threshold moves.

    Args:
        tmp_path: pytest's per-test temporary directory.
        latency_s: The status read's round-trip time, in seconds.

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
        clock=_fixed_clock,
        market_data=_LatentVenue(_anchored_fixture(books), latency_s=latency_s),
        live_ticker=TICKER,
    )
    return dataclasses.replace(deps, transport=_NearCertainVoteTransport())


def _payloads(deps: PaperTickDeps, event: str) -> list[dict[str, object]]:
    """Return every ledgered payload of one event type, in ledger order.

    Args:
        deps: The wired dependencies whose store is read.
        event: The event type wanted.

    Returns:
        That event's ``data`` payloads.
    """
    return [data for event_type, data in _rows(deps) if event_type == event]


def _assert_status_and_book_instants(deps: PaperTickDeps, *, latency_s: int) -> None:
    """Assert the status instant really is ``latency_s`` behind, and the book is not.

    The guard against this module degenerating into the coincidence it exists to
    break. Beat 2's ``ExchangeStatusObserved`` carries ``_approve_stage``'s
    ``status_epoch_s`` verbatim, and its ``MarketSnapshotRecorded`` carries the
    book the same beat priced against; the first must sit ``latency_s`` seconds
    before the run clock and the second exactly on it. A harness whose venue
    silently restamped the status would fail here rather than reading as a
    passing test of nothing.

    Args:
        deps: The wired dependencies whose ledger is read.
        latency_s: The status read's round-trip time, in seconds.

    Raises:
        AssertionError: If either beat is missing its row, or if the two
            instants are not exactly ``latency_s`` apart.
    """
    snapshots = _payloads(deps, "MarketSnapshotRecorded")
    statuses = _payloads(deps, "ExchangeStatusObserved")
    assert len(snapshots) == 2
    assert len(statuses) == 2
    assert snapshots[1]["fetched_at_epoch_s"] == FIXED_NOW_EPOCH_S
    assert statuses[1]["observed_at_epoch_s"] == FIXED_NOW_EPOCH_S - latency_s


def _run_two_beats(deps: PaperTickDeps) -> int:
    """Run beat 1, then beat 2, and return beat 2's fill.

    Beat 1 is deliberately present. It vetoes on ``daily_loss_limit`` -- a fresh
    ledger carries no ``EquitySampled`` row, so the day has no equity baseline --
    and ledgers the sample beat 2 measures against. Without it every beat-2
    reason list below would carry that transient too and the sweep would be
    reading two effects at once.

    Args:
        deps: The wired dependencies.

    Returns:
        Beat 2's filled quantity, in contract-centis.
    """
    run_single_tick(deps, beat=1)
    return run_single_tick(deps, beat=2).filled_centis


def test_the_swept_latencies_straddle_the_wired_status_ttl() -> None:
    """The three swept latencies really are ttl-1, ttl and ttl+1 around 30 seconds.

    Pinned before any of them is used, because the sweep's whole claim is about a
    boundary: a ttl that quietly moved would leave three latencies sitting on the
    same side of it and a module that still passed.
    """
    assert STATUS_TTL_SECONDS == 30
    assert (LATENCY_INSIDE_TTL_S, LATENCY_AT_TTL_S, LATENCY_PAST_TTL_S) == (29, 30, 31)
    assert sorted(APPROVAL_REASONS_BY_LATENCY) == [29, 30, 31]


def test_the_live_seam_passes_the_venues_status_instant_through_untouched(
    tmp_path: Path,
) -> None:
    """The production property this whole module rests on, asserted directly.

    ``build_paper_deps`` must hand back a
    :class:`~windbreak.connector.paper.LiveBookPaperExchange`, and that class
    must report the venue's own observation instant rather than the run clock.
    If a future change made it restamp the way its fixture-mode parent does, the
    sweep below would silently go green on a status that is fresh by
    construction -- exactly the defect this module exists to close -- so the
    property is pinned here rather than inferred from a passing verdict.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    deps = _latent_deps(tmp_path, latency_s=LATENCY_PAST_TTL_S)

    exchange = deps.exchange

    assert isinstance(exchange, LiveBookPaperExchange)
    assert exchange.get_exchange_status().fetched_at == _fixed_now() - timedelta(
        seconds=LATENCY_PAST_TTL_S
    )
    assert exchange.get_order_book(TICKER).fetched_at == _fixed_now()


def test_the_fixture_exchange_cannot_express_a_stale_status(tmp_path: Path) -> None:
    """The blind spot itself, made falsifiable instead of merely described.

    ``tests/integration/test_quote_freshness_drift.py`` reports this survivor in
    prose: no drift its clock can place separates ``_approve_stage``'s status
    read from its second clock read, because
    :meth:`~windbreak.connector.paper.PaperExchange.get_exchange_status` restamps
    from that same clock. Handed a fixture whose committed status is a frozen
    literal *decades* stale, the fixture exchange still answers "now" -- so the
    fixture path cannot express a stale status at any drift, and this module's
    live seam is not a stylistic preference but the only lever there is.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    books = tmp_path / "books"
    shutil.copytree(SHIPPED_BOOKS, books)
    _tradeable_books(books)
    fixture = _anchored_fixture(books)

    restamped = fixture.get_exchange_status()

    assert restamped.fetched_at == _fixed_now()
    # The committed literal the restamping discards is genuinely far outside
    # every ttl, so "it answered now" is a real substitution, not a no-op.
    assert abs(
        int(fixture.exchange_status.fetched_at.timestamp()) - FIXED_NOW_EPOCH_S
    ) > (STATUS_TTL_SECONDS)


@pytest.mark.parametrize("latency_s", sorted(APPROVAL_REASONS_BY_LATENCY))
def test_the_status_veto_fires_only_past_the_ttl_under_a_latent_venue(
    tmp_path: Path, latency_s: int
) -> None:
    """A status one second past its ttl vetoes; at the ttl and inside it, it does not.

    The issue's acceptance criteria 1 and 2, through the real
    :func:`~windbreak.scheduler.loop.run_single_tick` over the shipped
    composition -- nothing here calls ``_approve_stage``,
    ``build_evaluation_context`` or ``deps.approval.decide``, so the guard is
    entered by production code or not at all.

    Both directions are load-bearing and both are exact. The two fresh rows
    assert an *approval* whose reason list is empty and a 25 000-centi fill, so a
    guard that started over-vetoing fails here; the stale row asserts a veto
    whose reason list is exactly ``[STATUS_STALE_REASON]``, so a guard that
    stopped vetoing fails there. Substituting the stage's own ``now_epoch_s``
    for ``exchange_status_epoch_s`` collapses the third row onto the first two,
    which is precisely the mutant that survived every suite before this module
    existed.

    Args:
        tmp_path: pytest's per-test temporary directory.
        latency_s: The status read's round-trip time, in seconds.
    """
    deps = _latent_deps(tmp_path, latency_s=latency_s)

    filled = _run_two_beats(deps)

    _assert_status_and_book_instants(deps, latency_s=latency_s)
    expected = APPROVAL_REASONS_BY_LATENCY[latency_s]
    approvals = _payloads(deps, "IntentApproved")
    vetoes = _payloads(deps, "IntentVetoed")
    if expected == []:
        assert [payload["reasons"] for payload in approvals] == [[]]
        assert [payload["reasons"] for payload in vetoes] == [[DAILY_LOSS_VETO_REASON]]
        assert filled == FRESH_FILL_CENTIS
        assert len(_payloads(deps, "ApprovalTokenIssued")) == 1
    else:
        assert approvals == []
        assert len(vetoes) == 2
        assert vetoes[1]["reasons"] == expected
        assert filled == 0
        assert _payloads(deps, "ApprovalTokenIssued") == []
        assert _payloads(deps, "ReservationCreated") == []
    deps.store.verify_chain()


def test_the_stale_status_veto_names_the_intent_this_tick_struck(
    tmp_path: Path,
) -> None:
    """The vetoed intent is beat 2's own, ledgered after beat 2's status read.

    A one-market fixture makes it easy to assert *a* veto without ever
    establishing *which* tick it belongs to. Three ties close that gap: both
    beats snapshot :data:`TICKER`, the vetoed intent's id carries beat 2's
    ``ForecastCreated`` id (the selector mints ``<forecast_id>:...``), and that
    veto row is ledgered after beat 2's ``ExchangeStatusObserved`` rather than
    before it.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    deps = _latent_deps(tmp_path, latency_s=LATENCY_PAST_TTL_S)

    _run_two_beats(deps)

    events = [event for event, _ in _rows(deps)]
    tickers = [
        payload["ticker"] for payload in _payloads(deps, "MarketSnapshotRecorded")
    ]
    forecasts = _payloads(deps, "ForecastCreated")
    vetoes = _payloads(deps, "IntentVetoed")
    assert tickers == [TICKER, TICKER]
    assert len(forecasts) == 2
    # Counted before it is indexed: without this a tick that stopped vetoing
    # would fail below with an `IndexError` rather than saying what went wrong.
    assert len(vetoes) == 2
    assert vetoes[1]["intent_id"] == f"{forecasts[1]['forecast_id']}:yes:buy:sized"
    second_status = _nth_index(events, "ExchangeStatusObserved", 2)
    assert _nth_index(events, "IntentVetoed", 2) > second_status


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
            would otherwise make the ordering claim above vacuous.
    """
    positions = [index for index, name in enumerate(events) if name == event]
    assert len(positions) >= nth, f"expected >= {nth} {event}, got {len(positions)}"
    return positions[nth - 1]


def test_the_staleness_measurement_is_the_same_west_of_utc(
    tmp_path: Path, local_timezone_utc_minus_5: None
) -> None:
    """A fresh status stays fresh on a UTC-05:00 host, which CI is not.

    ``_approve_stage`` measures the status age as ``now_epoch_s -
    int(observed.fetched_at.timestamp())``. Both terms are offset-invariant
    *provided* the venue's instant stays timezone-aware: a naive stamp
    reinterpreted as local time would shift by 18 000 seconds, six hundred times
    the 30-second ttl, and this tick's approval would become a
    ``STATUS_STALE_REASON`` veto. CI runs UTC and cannot see that, so the pin is
    a fixed-offset zone with no DST rule rather than a hope.

    The *fresh* case is the sharp one: a misread instant can only push the age
    out, so an approval here is the direction that cannot pass by accident.

    Args:
        tmp_path: pytest's per-test temporary directory.
        local_timezone_utc_minus_5: Pins the process ``TZ`` to ``EST5``.
    """
    deps = _latent_deps(tmp_path, latency_s=LATENCY_INSIDE_TTL_S)

    filled = _run_two_beats(deps)

    _assert_status_and_book_instants(deps, latency_s=LATENCY_INSIDE_TTL_S)
    assert [payload["reasons"] for payload in _payloads(deps, "IntentApproved")] == [[]]
    assert filled == FRESH_FILL_CENTIS
    deps.store.verify_chain()
