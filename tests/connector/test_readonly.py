"""Tests for the narrow read-only venue view (issue #353, SPEC S1.1 inv. 3).

`ReadOnlyConnectorView` exists for exactly one reason: the verification path
must not be able to trade. These tests pin both halves of that -- the write
surface is genuinely absent (not merely "raises if called"), and every read it
does expose returns the wrapped connector's own answer verbatim, so the view
can never report a venue state the connector does not.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from windbreak.connector.paper import PaperExchange
from windbreak.connector.readonly import ReadOnlyConnectorView, ReadOnlyVenueView

#: The shared `deep_walk` books fixture (sole ticker `MKT-DEEP`).
_BOOKS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "books" / "deep_walk"

#: The sole ticker in that fixture.
_TICKER = "MKT-DEEP"

#: Every method the view is required to forward.
_READ_METHODS = (
    "get_balances",
    "get_balance_semantics",
    "get_positions",
    "get_open_orders",
)


#: A fixed observation instant. `get_balances` stamps `fetched_at` from the
#: injected clock, so a wall clock would make two successive reads differ by
#: microseconds and turn the forwarding assertion into a flake.
_FIXED_NOW = datetime(2025, 1, 1, tzinfo=UTC)


@pytest.fixture
def exchange() -> PaperExchange:
    """Provide a paper exchange over the shared books fixture.

    Returns:
        The loaded `PaperExchange`, on a fixed clock.
    """
    return PaperExchange.from_fixture_dir(_BOOKS_DIR, clock=lambda: _FIXED_NOW)


@pytest.mark.parametrize("method", ["place_order", "cancel_order"])
def test_view_does_not_expose_the_wrapped_connectors_write_surface(
    exchange: PaperExchange, method: str
) -> None:
    """The view has no order-placing attribute at all.

    `hasattr` is the assertion on purpose: a method that exists but raises
    would still be reachable through duck-typed code, and the invariant is that
    the verification path *cannot* trade, not that trading fails loudly.

    Args:
        exchange: The wrapped paper exchange.
        method: The write-surface method name that must be absent.
    """
    view = ReadOnlyConnectorView(exchange)

    assert hasattr(exchange, method)
    assert not hasattr(view, method)


@pytest.mark.parametrize("method", _READ_METHODS)
def test_view_forwards_each_read_verbatim(exchange: PaperExchange, method: str) -> None:
    """Every exposed read returns the wrapped connector's own answer.

    Args:
        exchange: The wrapped paper exchange.
        method: The read method under test.
    """
    view = ReadOnlyConnectorView(exchange)

    assert getattr(view, method)() == getattr(exchange, method)()


def test_view_forwards_get_market_by_ticker(exchange: PaperExchange) -> None:
    """`get_market` forwards its ticker argument and the connector's answer.

    Args:
        exchange: The wrapped paper exchange.
    """
    view = ReadOnlyConnectorView(exchange)

    assert view.get_market(_TICKER) == exchange.get_market(_TICKER)


def test_view_and_connector_both_satisfy_the_read_only_protocol(
    exchange: PaperExchange,
) -> None:
    """A full connector still satisfies the narrowed protocol structurally.

    That is what lets existing callers keep passing a whole connector while a
    composition root that wants the guarantee passes the view instead.

    Args:
        exchange: The wrapped paper exchange.
    """
    assert isinstance(exchange, ReadOnlyVenueView)
    assert isinstance(ReadOnlyConnectorView(exchange), ReadOnlyVenueView)


def test_view_reflects_later_connector_state_rather_than_a_frozen_copy(
    exchange: PaperExchange,
) -> None:
    """The view reads through, so a venue that moves is not hidden behind it.

    A snapshotting wrapper would quietly turn every later reconciliation into a
    comparison against stale data -- the exact "cannot fail" failure mode the
    verification path must not acquire.

    Args:
        exchange: The wrapped paper exchange.
    """
    from windbreak.connector.paper import PaperOrderIntent
    from windbreak.numeric import ContractCentis, PricePips

    view = ReadOnlyConnectorView(exchange)
    before = view.get_balances().available.value

    exchange.place_order(
        PaperOrderIntent(
            ticker=_TICKER,
            side="yes",
            price=PricePips(4600),
            quantity=ContractCentis(100),
        ),
        None,
    )

    assert view.get_balances().available.value < before
    assert view.get_positions() == exchange.get_positions()
