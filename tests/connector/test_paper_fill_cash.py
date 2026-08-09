"""Failing-first tests for `PaperExchange.fill_cash_micros` (issue #365, RED).

Ledgered fill accounting books what each execution cost the account, so the
kernel's reconciliation expectation can advance from ledgered evidence instead
of freezing at process start. That booking is only useful if the figure booked
is the *same* figure the venue's own balance moved by -- otherwise every filling
loop accumulates a phantom drift and halts anyway, just more slowly.

So the venue reports it. `fill_cash_micros` is the per-fill half of the exact
arithmetic `get_balances` folds across the whole fill log, exposed rather than
re-derived by the caller: a caller reimplementing book-cost-plus-fee would drift
from the simulator the first time either rounding rule changed, and drift here
is indistinguishable from the real divergence verification exists to catch.

The invariant test below is the anti-drift guard: summing this method across
every executed fill must equal the account's total cash movement, exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from windbreak.connector.paper import PaperExchange, PaperOrderIntent
from windbreak.numeric.types import ContractCentis, MoneyMicros, PricePips

if TYPE_CHECKING:
    from pathlib import Path

#: The shared books-fixture scenario every test below loads: a book deep enough
#: that a marketable order walks several levels and emits several fills.
_SCENARIO = "deep_walk"


@pytest.fixture(name="exchange")
def _exchange(books_fixture_dir: Path) -> PaperExchange:
    """Provide a paper exchange loaded from the shared books fixture.

    Args:
        books_fixture_dir: The shared books-fixture root.

    Returns:
        The loaded `PaperExchange`.
    """
    return PaperExchange.from_fixture_dir(books_fixture_dir / _SCENARIO)


def _first_ticker(exchange: PaperExchange) -> str:
    """Return a ticker the fixture has a session for.

    Args:
        exchange: The loaded paper exchange.

    Returns:
        One tradeable ticker.
    """
    return next(iter(exchange.sessions))


def test_fill_cash_micros_is_positive_for_an_executed_fill(
    exchange: PaperExchange,
) -> None:
    """The method reports a magnitude, not a signed movement: the bookkeeper
    applies the sign, so the venue never has to guess the account's convention.
    """
    ticker = _first_ticker(exchange)
    exchange.place_order(
        PaperOrderIntent(
            ticker=ticker,
            side="yes",
            price=PricePips(9900),
            quantity=ContractCentis(50),
        ),
        approval_token=None,
    )
    fills = exchange.get_fills(since=exchange.get_exchange_time().replace(year=1971))

    assert fills
    assert all(exchange.fill_cash_micros(fill).value > 0 for fill in fills)


def test_summed_fill_cash_equals_the_accounts_whole_cash_movement(
    exchange: PaperExchange,
) -> None:
    """The anti-drift invariant, and the reason this method exists at all.

    If the per-fill figure ever stopped summing to the venue's own balance
    movement, every booked expectation would diverge from the venue by the
    difference and a correctly-operating loop would halt on a bookkeeping bug
    rather than a real one.
    """
    ticker = _first_ticker(exchange)
    opening = exchange.balances.total
    exchange.place_order(
        PaperOrderIntent(
            ticker=ticker,
            side="yes",
            price=PricePips(9900),
            quantity=ContractCentis(50),
        ),
        approval_token=None,
    )
    fills = exchange.get_fills(since=exchange.get_exchange_time().replace(year=1971))

    spent = sum(exchange.fill_cash_micros(fill).value for fill in fills)

    assert MoneyMicros(opening.value - spent) == exchange.get_balances().total


def test_an_unfilled_account_has_no_fill_cash_to_report(
    exchange: PaperExchange,
) -> None:
    """A flat account reports no fills, so nothing is ever booked for it and the
    expectation stays exactly at its baseline."""
    assert (
        exchange.get_fills(since=exchange.get_exchange_time().replace(year=1971)) == ()
    )
