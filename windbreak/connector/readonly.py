"""The narrow read-only venue surface the verification path is allowed to hold.

SPEC S1.1 invariant 3 says the verification path never holds trade-scope
credentials. :class:`~windbreak.connector.interface.MarketConnector` cannot
express that: it bundles ``place_order``/``cancel_order`` in with the read
methods, so *typing* a verifier against it hands the verifier the write surface
and leaves the invariant to convention. :class:`ReadOnlyVenueView` is the
subset the verification path actually calls -- the five account/market reads --
and :class:`ReadOnlyConnectorView` is the concrete adapter that exposes exactly
those five and nothing else.

Handing :class:`ReadOnlyConnectorView` to a verifier makes the invariant
structural rather than advisory: the object it holds has no order-placing
method to call, so no amount of later drift inside the verifier can submit an
order. That matters most in PAPER, where the very same
:class:`~windbreak.connector.paper.PaperExchange` instance both fills the
loop's orders and answers the verification cycle's reads.

Any :class:`~windbreak.connector.interface.MarketConnector` structurally
satisfies :class:`ReadOnlyVenueView`, so existing callers that pass a whole
connector keep type-checking; the wrapper is what a *composition root* uses
when it wants the guarantee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from windbreak.connector.models import (
        BalanceSemantics,
        BalanceSnapshot,
        NormalizedMarket,
        OpenOrder,
        Position,
    )


@runtime_checkable
class ReadOnlyVenueView(Protocol):
    """The exact venue surface a read-only verification cycle reads.

    These five methods are everything
    :class:`~windbreak.riskkernel.verification.ReadOnlyVerifier` and
    :class:`~windbreak.riskkernel.verification.LedgerExpectationSource` call.
    The protocol deliberately stops there: no ``place_order``, no
    ``cancel_order``, and no market-data method the verification path has no
    business reaching for.
    """

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Return the market for ``ticker``.

        Args:
            ticker: The market ticker to look up.

        Returns:
            The normalized market.

        Raises:
            UnknownMarketError: If no market has that ticker.
        """
        ...

    def get_balances(self) -> BalanceSnapshot:
        """Return the account's current balances."""
        ...

    def get_balance_semantics(self) -> BalanceSemantics:
        """Return the venue's balance-interpretation semantics."""
        ...

    def get_positions(self) -> tuple[Position, ...]:
        """Return the account's open positions."""
        ...

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Return the account's resting open orders."""
        ...


class ReadOnlyConnectorView:
    """A :class:`ReadOnlyVenueView` over a connector, hiding its write surface.

    Composition, not inheritance, is the point: the wrapped connector is held
    on a private attribute and only the five read methods are re-exposed, so
    the view has no ``place_order``/``cancel_order`` attribute at all --
    ``hasattr(view, "place_order")`` is ``False``, not merely "raises if
    called". Every method delegates verbatim, so the view can never report a
    *different* venue state from the connector it wraps; it only reports less
    of it.
    """

    __slots__ = ("_connector",)

    def __init__(self, connector: ReadOnlyVenueView) -> None:
        """Wrap ``connector``, re-exposing only its read-only venue surface.

        Args:
            connector: The connector (typically a full
                :class:`~windbreak.connector.interface.MarketConnector`) whose
                reads this view forwards.
        """
        self._connector = connector

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Return the wrapped connector's market for ``ticker``.

        Args:
            ticker: The market ticker to look up.

        Returns:
            The normalized market.
        """
        return self._connector.get_market(ticker)

    def get_balances(self) -> BalanceSnapshot:
        """Return the wrapped connector's current balances.

        Returns:
            The balance snapshot.
        """
        return self._connector.get_balances()

    def get_balance_semantics(self) -> BalanceSemantics:
        """Return the wrapped connector's balance semantics.

        Returns:
            The balance semantics.
        """
        return self._connector.get_balance_semantics()

    def get_positions(self) -> tuple[Position, ...]:
        """Return the wrapped connector's open positions.

        Returns:
            The open positions.
        """
        return self._connector.get_positions()

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Return the wrapped connector's resting orders.

        Returns:
            The resting open orders.
        """
        return self._connector.get_open_orders()
