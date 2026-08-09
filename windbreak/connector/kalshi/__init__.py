"""The Kalshi connector: a public, read-only adapter over Kalshi's v2 REST API.

Per SPEC S5.2 this package models only *public, read-only market access* and
carries no trade credentials. It targets Kalshi's current API generation
(SPEC S7.1) at :data:`~windbreak.connector.kalshi.client.KALSHI_API_BASE`, or
:data:`~windbreak.connector.kalshi.client.KALSHI_DEMO_API_BASE` for a ``demo``
deployment (SPEC S16's ``exchange.environment``), and delivers
the read-only market-data surface: :class:`KalshiConnector` normalizes markets,
order books, exchange status, and server time, applying a binary-only product
allowlist and ledgering a :data:`PRODUCT_REFUSED_EVENT` for every refused
product. ``list_markets`` follows Kalshi's ``cursor`` pagination across every
page of ``/markets`` and ``/events`` (bounded by a hard cap that raises
:class:`KalshiPaginationError` rather than looping forever) and fails closed on
a single unnormalizable binary by ledgering a :data:`MARKET_MALFORMED_EVENT`.

``get_fee_model`` and ``get_balance_semantics`` are implemented (issue #18).

Scope fence -- methods that intentionally raise :class:`NotImplementedError`
until later work wires them:

    * ``place_order`` / ``cancel_order`` -- the order path (milestone M4).
    * ``get_balances`` / ``get_positions`` / ``get_open_orders`` /
      ``get_fills`` -- the remaining account access (issue #3).

Everything on the price/money path uses :mod:`windbreak.numeric` scaled-integer
types -- never floats (enforced by ``scripts/lint_no_floats.py``).
"""

from windbreak.connector.kalshi.adapter import (
    KalshiConnector,
    KalshiPaginationError,
)
from windbreak.connector.kalshi.client import (
    KalshiApiError,
    KalshiClient,
    KalshiResponse,
)
from windbreak.connector.kalshi.normalize import (
    MARKET_MALFORMED_EVENT,
    PRODUCT_REFUSED_EVENT,
)

__all__ = [
    "MARKET_MALFORMED_EVENT",
    "PRODUCT_REFUSED_EVENT",
    "KalshiApiError",
    "KalshiClient",
    "KalshiConnector",
    "KalshiPaginationError",
    "KalshiResponse",
]
