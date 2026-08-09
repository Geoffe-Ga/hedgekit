"""The PAPER composition root wired onto live Kalshi books (issue #343, RED).

PR #384 built the connector half -- `MarketDataSource`, `MarketDataOnlyView`,
`build_kalshi_market_data`, and `LiveBookPaperExchange` -- and deliberately
stopped before the composition root, because a *partial* wire would be worse
than none: snapshotting live books while the gateway, reconciler, and
verification view still held the fixture exchange would have the loop reason
about one venue and fill against another.

These tests specify the total wire. `build_paper_deps` grows a
`market_data`/`live_ticker` pair; when it is supplied every downstream
consumer -- the gateway's submitter, its status and reconciliation sources, the
`Reconciler`, the read-only verification view, and `PaperTickDeps.exchange`
itself -- must hold the *same* `LiveBookPaperExchange` instance. That identity
is asserted directly (`test_every_consumer_holds_the_one_live_exchange`), so a
future half-wire fails here rather than in production.

Two further properties are load-bearing and both fail *closed*:

* **No re-dating.** The fixture path anchors a recording's frozen literals to
  the run's clock (issue #369). A live book must not be anchored: passing the
  venue's own `fetched_at` through untouched is what lets `quote_freshness`
  genuinely veto a stale book. `test_a_live_book_keeps_the_venues_own_stamp`
  pins that by giving the venue an observation instant an hour and a half
  before the run's clock and asserting the book still carries the venue's.
* **No half-supplied pair.** A `market_data` without a `live_ticker` (or the
  reverse) refuses to build rather than silently falling back to fixtures --
  the same "half-wiring is actively wrong" argument, enforced at the API.

Every test drives the *real* `KalshiConnector` over the recorded-fixture
session `tests/connector/kalshi/conftest.py` already ships, so these are real
Kalshi-shaped books and a real normalization path, and nothing touches the
network (SPEC S17.1: CI is offline and deterministic).

RED, before the implementation exists: `build_paper_deps` has no `market_data`
parameter, so every test below fails with
`TypeError: build_paper_deps() got an unexpected keyword argument
'market_data'`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from tests.connector.kalshi.conftest import FakeKalshiSession
from tests.integration.conftest import ledger_path_for, read_event_type_payload_pairs
from windbreak.connector.live import build_kalshi_market_data
from windbreak.connector.paper import LiveBookPaperExchange, PaperExchange
from windbreak.connector.snapshot import InMemoryEventLedgerWriter
from windbreak.net.allowlist import OutboundAllowlist

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig
    from windbreak.connector.live import MarketDataOnlyView

#: The market the recorded Kalshi fixtures publish a book and a series for.
_TICKER = "KXFED-24DEC"

#: The instant the *venue* observed its book at -- the clock handed to the
#: connector. Deliberately well before `_RUN_AT` so "the book carries the
#: venue's stamp" cannot pass by coincidence with "the book was re-dated".
_VENUE_OBSERVED_AT = datetime(2024, 12, 1, 0, 0, 0, tzinfo=UTC)

#: The *run's* clock for the wiring tests: a different instant entirely.
_RUN_AT = datetime(2024, 12, 1, 1, 30, 0, tzinfo=UTC)

#: An allowlist admitting exactly the production Kalshi host, so the factory is
#: exercised against its real base URL while the injected session keeps every
#: request offline.
_KALSHI_ALLOWLIST = OutboundAllowlist(frozenset({"api.elections.kalshi.com"}))


def _run_clock() -> int:
    """Return the wiring tests' fixed run clock, in epoch seconds.

    Returns:
        The epoch second of `_RUN_AT`.
    """
    return int(_RUN_AT.timestamp())


def _aligned_clock() -> int:
    """Return a run clock aligned with the venue's own observation instant.

    The end-to-end tick uses this so the venue's book is genuinely fresh and
    the clock-skew check compares two agreeing clocks -- exercising the live
    path past its freshness gates rather than stopping at them.

    Returns:
        The epoch second of `_VENUE_OBSERVED_AT`.
    """
    return int(_VENUE_OBSERVED_AT.timestamp())


@pytest.fixture
def live_market_data() -> MarketDataOnlyView:
    """Provide the live market-data view over the recorded Kalshi fixtures.

    Returns:
        A `MarketDataOnlyView` over a real `KalshiConnector` whose session is
        the recorded-fixture double, stamping every observation at
        `_VENUE_OBSERVED_AT`.
    """
    return build_kalshi_market_data(
        environment="production",
        allowlist=_KALSHI_ALLOWLIST,
        ledger_writer=InMemoryEventLedgerWriter(),
        session=FakeKalshiSession(),
        clock=lambda: _VENUE_OBSERVED_AT,
    )


def _build_live_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    research_tools_factory,
    market_data: MarketDataOnlyView,
    clock=_run_clock,
):
    """Build one `PaperTickDeps` whose market data is the live venue view.

    Args:
        books_dir: The `deep_walk` fixture directory; in live mode only its
            account fixtures (opening balances, balance semantics) are read.
        cassette_path: The (empty) recorded-cassette path.
        ledger_path: Where the tick's `SqliteLedgerStore` is created.
        report_dir: Where weekly-report stubs would be written.
        config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
        market_data: The live venue view the session reads through.
        clock: The injected epoch-second run clock.

    Returns:
        A fully wired `PaperTickDeps` in live-book mode.
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
        market_data=market_data,
        live_ticker=_TICKER,
    )


def test_omitting_market_data_keeps_the_fixture_path(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """The default is byte-identical: no `market_data`, no live session.

    Acceptance criterion "default (no flag) behavior is byte-identical to
    today's fixture path" -- pinned as the *absence* of a live session, not
    merely as a passing tick.
    """
    from windbreak.scheduler.loop import build_paper_deps

    deps = build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools=research_tools_factory(),
        clock=_run_clock,
    )

    assert isinstance(deps.exchange, PaperExchange)
    assert not isinstance(deps.exchange, LiveBookPaperExchange)


def test_live_market_data_builds_a_live_book_session(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    live_market_data: MarketDataOnlyView,
    tmp_path: Path,
) -> None:
    """Supplying the pair binds the loop to a live-book session on that ticker."""
    deps = _build_live_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
        market_data=live_market_data,
    )

    assert isinstance(deps.exchange, LiveBookPaperExchange)
    # `live_ticker` still binds exactly one market, and since issue #345 that is
    # observable only as the universe the tick will screen: the deps bundle no
    # longer carries a ticker of its own, so the exchange's market set is what
    # says which market the live session was bound to.
    assert tuple(deps.exchange.markets) == (_TICKER,)
    assert deps.exchange.market_data is live_market_data


def test_every_consumer_holds_the_one_live_exchange(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    live_market_data: MarketDataOnlyView,
    tmp_path: Path,
) -> None:
    """The wire is *total*: one exchange instance, held by every consumer.

    This is the test PR #384 said had to exist before the flag could. A wire
    that fed live books to the snapshot while the gateway filled against a
    fixture exchange would let the loop reason about one venue and trade on
    another; identity, not type, is what rules that out.
    """
    deps = _build_live_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
        market_data=live_market_data,
    )

    exchange = deps.exchange
    assert deps.gateway._submitter._exchange is exchange
    assert deps.gateway._status_source is exchange
    assert deps.gateway._reconciliation_source is exchange
    assert deps.reconciler._source is exchange
    assert deps.verification_view._connector is exchange


def test_a_live_book_keeps_the_venues_own_stamp(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    live_market_data: MarketDataOnlyView,
    tmp_path: Path,
) -> None:
    """No replay anchor in live mode: the venue's `fetched_at` survives.

    The fixture path re-dates a recording to the run's clock (issue #369)
    because its frozen literals are stale against every ttl forever. Applying
    that to a live book would fabricate freshness the venue never claimed and
    make `quote_freshness` unable to ever veto. The book must therefore carry
    the venue's instant, an hour and a half behind this run's clock.
    """
    deps = _build_live_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
        market_data=live_market_data,
    )

    book = deps.exchange.get_order_book(_TICKER)

    assert book.fetched_at == _VENUE_OBSERVED_AT
    assert book.fetched_at != _RUN_AT


def test_market_data_without_a_ticker_is_refused(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    live_market_data: MarketDataOnlyView,
    tmp_path: Path,
) -> None:
    """A venue with no market named refuses to build rather than guessing.

    A `MarketDataSource` cannot enumerate the venue's catalog by design (the
    market universe is issue #345), so there is no ticker to fall back to --
    and falling back to the *fixture* directory's first market would run a
    live-book session against a market nobody asked for.
    """
    from windbreak.scheduler.loop import build_paper_deps

    with pytest.raises(ValueError, match="live_ticker"):
        build_paper_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            ledger_path=ledger_path_for(tmp_path),
            report_dir=report_dir,
            config=paper_config,
            research_tools=research_tools_factory(),
            clock=_run_clock,
            market_data=live_market_data,
        )


def test_a_ticker_without_market_data_is_refused(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A ticker with no venue refuses rather than silently using fixtures.

    Quietly serving fixture books to an operator who asked for a live market
    is the exact failure mode this issue exists to close.
    """
    from windbreak.scheduler.loop import build_paper_deps

    with pytest.raises(ValueError, match="market_data"):
        build_paper_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            ledger_path=ledger_path_for(tmp_path),
            report_dir=report_dir,
            config=paper_config,
            research_tools=research_tools_factory(),
            clock=_run_clock,
            live_ticker=_TICKER,
        )


def test_a_live_tick_runs_end_to_end_through_the_kalshi_connector(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    live_market_data: MarketDataOnlyView,
    tmp_path: Path,
) -> None:
    """One full tick, driven off real Kalshi-shaped books, ledgers and verifies.

    The issue's headline acceptance criterion. The recorded book normalizes to
    YES bids `4500 x 10_000` / `4400 x 25_000` centis and a single YES ask
    `4800 x 4_000` centis, and it is *that* book the tick's `MarketSnapshot`
    row must carry -- proof the loop's snapshot came from the venue rather
    than from `tests/fixtures/books/deep_walk`.
    """
    deps = _build_live_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
        market_data=live_market_data,
        clock=_aligned_clock,
    )

    from windbreak.scheduler.loop import run_single_tick

    outcome = run_single_tick(deps, beat=1)

    assert outcome is not None
    deps.store.verify_chain()
    snapshots = [
        payload
        for event_type, payload in read_event_type_payload_pairs(deps.store.read_all())
        if event_type == "MarketSnapshotRecorded"
    ]
    assert len(snapshots) == 1
    assert snapshots[0]["ticker"] == _TICKER
    assert snapshots[0]["best_bid_pips"] == 4500
    assert snapshots[0]["best_ask_pips"] == 4800
    assert snapshots[0]["fetched_at_epoch_s"] == int(_VENUE_OBSERVED_AT.timestamp())
