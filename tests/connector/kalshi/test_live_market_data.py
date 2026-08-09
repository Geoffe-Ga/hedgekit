"""Tests for the Kalshi live-market-data composition root (issue #343).

`build_kalshi_market_data` is the one place a real `KalshiConnector` becomes
reachable from outside its own package: it resolves the environment's API base,
screens that base against the deployment's outbound allowlist, and hands back a
`MarketDataOnlyView` -- a surface with no order and no account method on it at
all.

Every test here runs through the recorded-fixture session, so nothing dials the
network (SPEC S17.1: CI is offline and deterministic).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.connector.kalshi.client import KALSHI_API_BASE, KALSHI_DEMO_API_BASE
from windbreak.connector.live import MarketDataOnlyView, build_kalshi_market_data
from windbreak.connector.paper import LiveBookPaperExchange
from windbreak.net.allowlist import OutboundAllowlist

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.connector.snapshot import InMemoryEventLedgerWriter

    from .conftest import FakeKalshiSession

#: The account fixtures a live-book session borrows its opening balance and
#: balance semantics from; its books and markets come from the venue.
_ACCOUNT_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "books" / "deep_walk"

#: The recorded market the fixture session serves a book for.
_TICKER = "KXFED-24DEC"

#: An allowlist admitting exactly the two real Kalshi API hosts, so the factory
#: is exercised against its *real* base URLs while the injected session keeps
#: every request offline.
_KALSHI_ALLOWLIST = OutboundAllowlist(
    frozenset({"api.elections.kalshi.com", "demo-api.kalshi.co"})
)


def test_production_environment_resolves_the_production_api_base(
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
    clock: Callable[[], datetime],
) -> None:
    """A ``production`` deployment dials the production base and nothing else.

    Args:
        fake_kalshi_session: The recorded-fixture session (no network).
        ledger: The event ledger refusals are recorded through.
        clock: The connector's injected observation clock.
    """
    market_data = build_kalshi_market_data(
        environment="production",
        allowlist=_KALSHI_ALLOWLIST,
        ledger_writer=ledger,
        session=fake_kalshi_session,
        clock=clock,
    )

    market_data.get_exchange_status()

    assert [call["url"] for call in fake_kalshi_session.calls] == [
        f"{KALSHI_API_BASE}/exchange/status"
    ]


def test_demo_environment_resolves_the_demo_api_base(
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
    clock: Callable[[], datetime],
) -> None:
    """A ``demo`` deployment dials the demo base (SPEC S16 ``demo | production``).

    Args:
        fake_kalshi_session: The recorded-fixture session (no network).
        ledger: The event ledger refusals are recorded through.
        clock: The connector's injected observation clock.
    """
    market_data = build_kalshi_market_data(
        environment="demo",
        allowlist=_KALSHI_ALLOWLIST,
        ledger_writer=ledger,
        session=fake_kalshi_session,
        clock=clock,
    )

    market_data.get_exchange_status()

    assert [call["url"] for call in fake_kalshi_session.calls] == [
        f"{KALSHI_DEMO_API_BASE}/exchange/status"
    ]


def test_an_unrecognized_environment_is_refused(
    fake_kalshi_session: FakeKalshiSession, ledger: InMemoryEventLedgerWriter
) -> None:
    """An unknown environment fails closed rather than guessing a base URL.

    Args:
        fake_kalshi_session: The recorded-fixture session (no network).
        ledger: The event ledger refusals are recorded through.
    """
    with pytest.raises(ValueError, match="unknown exchange environment 'staging'"):
        build_kalshi_market_data(
            environment="staging",
            allowlist=_KALSHI_ALLOWLIST,
            ledger_writer=ledger,
            session=fake_kalshi_session,
        )


def test_an_api_base_off_the_allowlist_is_refused(
    fake_kalshi_session: FakeKalshiSession, ledger: InMemoryEventLedgerWriter
) -> None:
    """The deployment's own allowlist still gates the venue host (SPEC S15).

    A deployment whose config declares only the demo host must not reach
    production merely because the factory knows its URL.

    Args:
        fake_kalshi_session: The recorded-fixture session (no network).
        ledger: The event ledger refusals are recorded through.
    """
    demo_only = OutboundAllowlist(frozenset({"demo-api.kalshi.co"}))

    with pytest.raises(ValueError, match="outbound allowlist"):
        build_kalshi_market_data(
            environment="production",
            allowlist=demo_only,
            ledger_writer=ledger,
            session=fake_kalshi_session,
        )


def test_the_built_surface_is_market_data_only(
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
    clock: Callable[[], datetime],
) -> None:
    """The factory hands back a view with no order or account method.

    Args:
        fake_kalshi_session: The recorded-fixture session (no network).
        ledger: The event ledger refusals are recorded through.
        clock: The connector's injected observation clock.
    """
    market_data = build_kalshi_market_data(
        environment="production",
        allowlist=_KALSHI_ALLOWLIST,
        ledger_writer=ledger,
        session=fake_kalshi_session,
        clock=clock,
    )

    assert isinstance(market_data, MarketDataOnlyView)
    assert not hasattr(market_data, "place_order")
    assert not hasattr(market_data, "get_balances")


def test_the_default_observation_clock_is_timezone_aware_utc(
    fake_kalshi_session: FakeKalshiSession, ledger: InMemoryEventLedgerWriter
) -> None:
    """Omitting ``clock`` stamps observations on the wall clock, in UTC.

    A naive datetime here would make every ``now - fetched_at`` freshness
    subtraction raise ``TypeError`` instead of vetoing, turning a stale-data
    check into a crash.

    Args:
        fake_kalshi_session: The recorded-fixture session (no network).
        ledger: The event ledger refusals are recorded through.
    """
    market_data = build_kalshi_market_data(
        environment="production",
        allowlist=_KALSHI_ALLOWLIST,
        ledger_writer=ledger,
        session=fake_kalshi_session,
    )

    before = datetime.now(UTC)
    observed = market_data.get_exchange_status().fetched_at
    after = datetime.now(UTC)

    assert observed.tzinfo is not None
    assert before <= observed <= after


def test_a_paper_session_trades_against_the_real_connectors_book(
    fake_kalshi_session: FakeKalshiSession,
    ledger: InMemoryEventLedgerWriter,
    clock: Callable[[], datetime],
    kalshi_fixture_server_date: datetime,
) -> None:
    """Real books through the Kalshi read path, paper money out.

    The recorded book is ``yes: [[45, 100], [44, 250]]`` and ``no: [[52, 40]]``,
    which normalizes to YES bids ``4500 x 10_000`` / ``4400 x 25_000`` centis and
    a single YES ask ``4800 x 4_000`` centis. The market's jurisdiction is
    ``unknown`` because Kalshi publishes no per-market eligibility signal -- the
    kernel vetoing on that is correct, not a gap.

    Args:
        fake_kalshi_session: The recorded-fixture session (no network).
        ledger: The event ledger refusals are recorded through.
        clock: The connector's injected observation clock.
        kalshi_fixture_server_date: The venue clock the fixture's ``Date``
            header publishes.
    """
    market_data = build_kalshi_market_data(
        environment="production",
        allowlist=_KALSHI_ALLOWLIST,
        ledger_writer=ledger,
        session=fake_kalshi_session,
        clock=clock,
    )
    exchange = LiveBookPaperExchange.from_account_dir(
        _ACCOUNT_DIR, market_data=market_data, ticker=_TICKER
    )

    book = exchange.get_order_book(_TICKER)

    assert list(exchange.markets) == [_TICKER]
    assert exchange.get_market(_TICKER).jurisdiction_status == "unknown"
    assert [(level.price.value, level.quantity.value) for level in book.yes_bids] == [
        (4500, 10_000),
        (4400, 25_000),
    ]
    assert [(level.price.value, level.quantity.value) for level in book.yes_asks] == [
        (4800, 4_000)
    ]
    assert exchange.get_fee_model(_TICKER).schedule_id == "kxfed-standard-v1"
    assert exchange.get_exchange_time() == kalshi_fixture_server_date
