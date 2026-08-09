"""The credential-free live market-data surface the PAPER loop reads (issue #343).

The PAPER loop's market data has until now come from a fixture directory, so the
Kalshi adapter -- normalization, fee model, freshness, resilience, egress
allowlist, all built and tested -- was reachable from nowhere outside its own
package. This module is the seam that connects them: it names the exact venue
reads a paper session needs (:class:`MarketDataSource`), hands back a wrapper
that exposes those and *only* those (:class:`MarketDataOnlyView`), and builds
one over a real :class:`~windbreak.connector.kalshi.adapter.KalshiConnector`
(:func:`build_kalshi_market_data`).

Three properties are load-bearing, and each is a fail-closed one:

**No credentials, structurally.** SPEC S1.1 invariant 3 keeps trade-scope
credentials out of everything but the Order Gateway.
:class:`~windbreak.connector.kalshi.client.KalshiClient` models public read-only
market access and never attaches an auth header at all, so this path holds no
key to leak -- there is deliberately no ``*_api_key_env`` indirection here,
because there is no key. :class:`MarketDataOnlyView` then makes the *other* half
structural the way :class:`~windbreak.connector.readonly.ReadOnlyConnectorView`
does for the verification path: the object a live paper session holds has no
``place_order`` attribute to call and no account read to reach, so no later
drift inside the session can submit an order or mistake the venue's account for
its own simulated one.

**No implicit egress.** The venue base URL is resolved from the deployment's
declared environment (SPEC S16's ``demo | production``) and then screened
against the deployment's own :class:`~windbreak.net.allowlist.OutboundAllowlist`
before any session exists (SPEC S15). Knowing a URL is not permission to dial
it: an allowlist derived from one environment refuses the other's host.

**No offline surprises.** The ``session`` seam is injectable, so the suite drives
this whole path against recorded fixtures and CI never touches the network.

This module is on ``scripts/lint_no_floats.py``'s denylist: it does no
arithmetic, so no float or true division appears here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from windbreak.connector.kalshi.adapter import KalshiConnector
from windbreak.connector.kalshi.client import (
    KALSHI_API_BASE,
    KALSHI_DEMO_API_BASE,
    KalshiClient,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.connector.fees import FeeModel
    from windbreak.connector.kalshi.client import Session
    from windbreak.connector.models import (
        ExchangeStatus,
        NormalizedMarket,
        OrderBookSnapshot,
    )
    from windbreak.connector.snapshot import EventLedgerWriter
    from windbreak.net.allowlist import OutboundAllowlist

__all__ = [
    "MarketDataOnlyView",
    "MarketDataSource",
    "build_kalshi_market_data",
]

#: SPEC S16's ``exchange.environment`` values mapped to their API base. A
#: deployment naming anything else is refused rather than defaulted to
#: production: silently dialing the real venue because an environment string was
#: misspelled is the costliest possible interpretation of a typo.
_ENVIRONMENT_API_BASES: Final[dict[str, str]] = {
    "production": KALSHI_API_BASE,
    "demo": KALSHI_DEMO_API_BASE,
}


def _utc_now() -> datetime:
    """Return the current instant as a timezone-aware UTC datetime.

    Mirrors the module-local default clocks in
    :mod:`windbreak.connector.kalshi.adapter` and
    :mod:`windbreak.connector.paper`: every component that stamps an
    observation owns its own default so a test can replace it wholesale.

    Returns:
        The current UTC instant.
    """
    return datetime.now(UTC)


@runtime_checkable
class MarketDataSource(Protocol):
    """The exact venue surface a live-book paper session reads.

    These five methods are everything
    :class:`~windbreak.connector.paper.LiveBookPaperExchange` asks a venue for.
    The protocol deliberately stops there: no ``place_order``, no
    ``cancel_order``, and none of the account reads (balances, positions, open
    orders, fills) -- in a live-book paper session those describe the
    *simulator's* account, not the venue's, and reading the venue's would blend
    two different sets of books into one.

    Any full :class:`~windbreak.connector.interface.MarketConnector`
    structurally satisfies this, so existing callers keep type-checking; the
    concrete :class:`MarketDataOnlyView` is what a composition root passes when
    it wants the narrowing to be structural rather than advisory.
    """

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Return the venue's normalized market for ``ticker``.

        Args:
            ticker: The market ticker to look up.

        Returns:
            The normalized market.
        """
        ...

    def get_order_book(self, ticker: str) -> OrderBookSnapshot:
        """Return the venue's current order book for ``ticker``.

        Args:
            ticker: The market ticker to look up.

        Returns:
            The order-book snapshot, carrying the venue's own ``fetched_at``.
        """
        ...

    def get_exchange_status(self) -> ExchangeStatus:
        """Return the venue's current trading status."""
        ...

    def get_exchange_time(self) -> datetime:
        """Return the venue's own clock reading."""
        ...

    def get_fee_model(self, market_or_series: str) -> FeeModel:
        """Return the venue's fee schedule for a market or series.

        Args:
            market_or_series: A market ticker or a bare series ticker.

        Returns:
            The normalized fee model.
        """
        ...


class MarketDataOnlyView:
    """A :class:`MarketDataSource` over a connector, hiding everything else.

    Composition, not inheritance, is the point: the wrapped connector is held on
    a private attribute and only the five market-data reads are re-exposed, so
    the view has no ``place_order``/``cancel_order`` attribute and no account
    read at all -- ``hasattr(view, "place_order")`` is ``False``, not merely
    "raises if called". Every method delegates verbatim, so the view can never
    report a venue state the connector it wraps does not; it only reports less
    of it.

    The sibling of :class:`~windbreak.connector.readonly.ReadOnlyConnectorView`,
    narrowed to a different question: that one is the account surface a
    verification cycle may hold, this one is the market surface a live-book
    paper session may hold.
    """

    __slots__ = ("_source",)

    def __init__(self, source: MarketDataSource) -> None:
        """Wrap ``source``, re-exposing only its market-data reads.

        Args:
            source: The connector (typically a full
                :class:`~windbreak.connector.interface.MarketConnector`) whose
                market-data reads this view forwards.
        """
        self._source = source

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Return the wrapped source's market for ``ticker``.

        Args:
            ticker: The market ticker to look up.

        Returns:
            The normalized market.
        """
        return self._source.get_market(ticker)

    def get_order_book(self, ticker: str) -> OrderBookSnapshot:
        """Return the wrapped source's current book for ``ticker``.

        Args:
            ticker: The market ticker to look up.

        Returns:
            The order-book snapshot, timestamp included, verbatim.
        """
        return self._source.get_order_book(ticker)

    def get_exchange_status(self) -> ExchangeStatus:
        """Return the wrapped source's exchange status.

        Returns:
            The exchange status.
        """
        return self._source.get_exchange_status()

    def get_exchange_time(self) -> datetime:
        """Return the wrapped source's venue clock reading.

        Returns:
            The venue's clock, in UTC.
        """
        return self._source.get_exchange_time()

    def get_fee_model(self, market_or_series: str) -> FeeModel:
        """Return the wrapped source's fee schedule.

        Args:
            market_or_series: A market ticker or a bare series ticker.

        Returns:
            The normalized fee model.
        """
        return self._source.get_fee_model(market_or_series)


def build_kalshi_market_data(
    *,
    environment: str,
    allowlist: OutboundAllowlist,
    ledger_writer: EventLedgerWriter,
    session: Session | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> MarketDataOnlyView:
    """Build the live Kalshi market-data surface for one deployment.

    Resolves SPEC S16's ``exchange.environment`` to its API base, hands that base
    to :class:`~windbreak.connector.kalshi.client.KalshiClient` -- which screens
    it against ``allowlist`` *before* creating a session, so an undeclared host
    is refused with no network call at all -- and narrows the resulting
    connector to :class:`MarketDataOnlyView`.

    Nothing here reads a credential, and there is no ``*_api_key_env`` parameter
    by design: Kalshi's market-data routes are public, the client never attaches
    an auth header, and a configuration leaf naming a real secret would be
    flattened verbatim by
    :func:`~windbreak.config.versioning.diff_configs` into the append-only
    hash-chained ledger.

    Args:
        environment: The deployment's exchange environment; ``"production"`` or
            ``"demo"``.
        allowlist: The deployment's outbound-egress allowlist, which the
            resolved API base's host must be on (SPEC S15).
        ledger_writer: The seam refused products and connector halts are
            recorded through.
        session: An injected ``requests``-like session; ``None`` lets the client
            create a real one. Every test injects a recorded-fixture session, so
            CI stays offline.
        clock: The connector's observation clock, stamped on the books and
            statuses it normalizes.

    Returns:
        The market-data-only view over a live Kalshi connector.

    Raises:
        ValueError: If ``environment`` is not a recognized SPEC S16 value, or
            the resolved API base's host is not on ``allowlist``. Either way
            the deployment refuses to build a venue reader rather than guessing
            which venue it meant.
    """
    try:
        base_url = _ENVIRONMENT_API_BASES[environment]
    except KeyError as exc:
        known = ", ".join(sorted(_ENVIRONMENT_API_BASES))
        raise ValueError(
            f"unknown exchange environment {environment!r}; expected one of {known}"
        ) from exc
    client = KalshiClient(base_url, session=session, allowlist=allowlist)
    return MarketDataOnlyView(KalshiConnector(client, ledger_writer, clock=clock))
