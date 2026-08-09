"""The recorded-fixture contract-test MATRIX over `MarketConnector` reads (#21).

SPEC S7.2 declares thirteen `MarketConnector` methods; this module pins the
**eleven read methods** across both shipped adapters --
:class:`~windbreak.connector.kalshi.adapter.KalshiConnector` and
:class:`~windbreak.connector.paper.PaperExchange` -- leaving out
``place_order``/``cancel_order`` (the two *trading* methods, wired by a later
milestone and already covered by their own dedicated deferral tests). The
eleven read methods are: ``list_markets``, ``get_market``, ``get_order_book``,
``get_exchange_status``, ``get_exchange_time``, ``get_balance_semantics``,
``get_balances``, ``get_positions``, ``get_open_orders``, ``get_fills``, and
``get_fee_model``.

Unlike a RED-first TDD suite, every test here **pins CURRENT merged
behavior** and is expected to *pass* today; a genuine failure is a real
contract discrepancy to report, not a target to chase.

Design: a declarative case table
---------------------------------
The matrix is a dict, ``_MATRIX``, keyed by ``(connector, endpoint,
scenario)`` over 2 connectors x 11 endpoints x 5 scenarios
(``happy``/``error``/``rate_limit``/``malformed``/``schema_drift``) = **110
logical cells**. Each cell is either:

* a zero-argument runnable callable pinning a concrete assertion, or
* an explicit :class:`NotApplicable` marker carrying a *reason* -- never a
  bare ``pytest.mark.skip`` (forbidden by the anti-bypass rule).

``test_matrix_cell`` (parametrized via ``pytest_generate_tests`` over every
registered key, with ids like ``kalshi-get_order_book-schema_drift``) either
executes the runnable or asserts the N/A marker carries a non-empty reason,
so every one of the 110 cells is a real, passing pytest item --
``test_matrix_accounts_for_every_endpoint_scenario_cell`` is the completeness
meta-test proving no cell silently vanished.

Harness reuse
--------------
Rather than re-deriving the Kalshi fault-injection harness, this module
imports the plain (non-fixture) helper classes straight from
`tests/connector/kalshi/conftest.py` -- `FakeKalshiSession` (routes recorded
fixtures by URL suffix, 404s unknown), `RecordingSleeper`, and `FakeIntClock`
-- exactly as `test_adapter.py` / `test_client_resilience.py` compose them,
just invoked directly (a pytest fixture function cannot be called outside
the fixture-injection system) instead of via pytest's dependency injection,
since every matrix cell is a plain zero-argument callable, not a fixture
consumer. `ScriptedFaultSession`'s response queue is *global* FIFO
(fine for the linear call sequences in `test_client_resilience.py`, but this
module's cells frequently need one route to fault while sibling routes
(`/exchange/status`, and -- since `get_order_book` gates the product
allowlist via a `/markets` fetch before fetching the book, #278 -- `/markets`
too) stay clean); this module's local `_RouteQueueSession` gives each
URL-suffix route its own independent queue for exactly that reason. The two
schema-drift fixtures
(`tests/fixtures/exchange/kalshi/faults/orderbook_drift_money_fee.json` and
`orderbook_drift_cosmetic.json`) are reused verbatim for `get_order_book`'s
schema-drift cell and its cosmetic-tolerance supplement; every other
endpoint's schema-drift cell mutates an in-memory copy of its recorded
fixture rather than adding a new checked-in file (no new fixtures were
added by this module).

Float / fixed-point preservation guard
---------------------------------------
`test_kalshi_happy_path_values_carry_no_float_leaf` and
`test_paper_happy_path_values_carry_no_float_leaf` recursively walk every
happy-path return value (dataclass fields, tuple/list elements, mapping
values) and assert no ``float`` instance appears anywhere (datetimes, ints,
and windbreak's scaled-integer unit types are all fine) -- the SPEC S17.6
connector-boundary acceptance check.
`test_lint_no_floats_passes_over_windbreak_connector` additionally runs
`scripts/lint_no_floats.py` over `windbreak/connector` as a
light, additive AST-level guard (the repo-wide float-lint test already
covers this; this is not a duplicate suite).
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import random
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest

from tests.connector.kalshi.conftest import (
    FakeIntClock,
    FakeKalshiSession,
    RecordingSleeper,
)
from windbreak.connector.fees import UnknownFeeModelError
from windbreak.connector.interface import MarketConnector, UnknownMarketError
from windbreak.connector.kalshi.adapter import KALSHI_BALANCE_SEMANTICS, KalshiConnector
from windbreak.connector.kalshi.client import KalshiApiError, KalshiClient
from windbreak.connector.kalshi.normalize import MARKET_MALFORMED_EVENT
from windbreak.connector.paper import PaperExchange
from windbreak.connector.resilience import ResiliencePolicy, ResilientCaller
from windbreak.connector.semantics import PartialFillRepresentation
from windbreak.connector.snapshot import InMemoryEventLedgerWriter
from windbreak.connector.validation import (
    SCHEMA_ANOMALY_EVENT,
    SchemaAnomalyHaltError,
    SchemaValidator,
    kalshi_default_schema_registry,
)
from windbreak.net.allowlist import OutboundAllowlist
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips

if TYPE_CHECKING:
    import types
    from collections.abc import Callable

# =============================================================================
# Constants: fixture paths, the endpoint/connector/scenario axes, fixed values
# =============================================================================

#: `tests/connector/test_contract_matrix.py` -> `tests/connector` -> `tests/`.
_KALSHI_FIXTURE_DIR: Final = (
    Path(__file__).resolve().parent.parent / "fixtures" / "exchange" / "kalshi"
)
_BOOKS_FIXTURE_DIR: Final = (
    Path(__file__).resolve().parent.parent / "fixtures" / "books"
)
_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_LINT_SCRIPT_PATH: Final = _REPO_ROOT / "scripts" / "lint_no_floats.py"

_FAKE_BASE_URL: Final = "https://fake-kalshi.test"

#: Allowlist admitting the fake host, now that ``KalshiClient`` enforces its
#: base URL host at construction (issue #57).
_FAKE_ALLOWLIST: Final = OutboundAllowlist(frozenset({"fake-kalshi.test"}))

#: The injected connector clock every cell below uses; distinct from the
#: fixed `Date` header `FakeKalshiSession` serves so a test can tell them apart.
_CLOCK_FIXED: Final = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

#: The `Date` header every `FakeKalshiSession`-backed response carries
#: (mirrors `tests/connector/kalshi/conftest.py`'s `_FIXED_SERVER_DATE`).
_FIXTURE_SERVER_DATE: Final = datetime(2024, 12, 1, tzinfo=UTC)

#: `KalshiConnector`'s exact current message for every deferred account method
#: (`windbreak.connector.kalshi.adapter._ACCOUNT_DEFERRAL`, copied verbatim so
#: this pin does not depend on importing a private module constant).
_EXPECTED_ACCOUNT_DEFERRAL_MESSAGE: Final = (
    "account access (balances, positions, open orders, fills) is deferred to issue #3"
)

_ENDPOINTS: Final[tuple[str, ...]] = (
    "list_markets",
    "get_market",
    "get_order_book",
    "get_exchange_status",
    "get_exchange_time",
    "get_balance_semantics",
    "get_balances",
    "get_positions",
    "get_open_orders",
    "get_fills",
    "get_fee_model",
)
_CONNECTORS: Final[tuple[str, ...]] = ("kalshi", "paper")
_SCENARIOS: Final[tuple[str, ...]] = (
    "happy",
    "error",
    "rate_limit",
    "malformed",
    "schema_drift",
)
_NON_HAPPY_SCENARIOS: Final[tuple[str, ...]] = (
    "error",
    "rate_limit",
    "malformed",
    "schema_drift",
)


def _clock() -> datetime:
    """Return the fixed connector clock every cell in this module injects."""
    return _CLOCK_FIXED


def _read_kalshi_fixture(name: str) -> Any:
    """Parse one recorded Kalshi JSON fixture by filename.

    Args:
        name: The fixture file's name, e.g. ``"markets.json"``.

    Returns:
        The parsed JSON.
    """
    return json.loads((_KALSHI_FIXTURE_DIR / name).read_text(encoding="utf-8"))


# =============================================================================
# NotApplicable: the explicit, reasoned "this cell does not apply" marker
# =============================================================================


@dataclass(frozen=True, slots=True)
class NotApplicable:
    """An explicit, reasoned "this cell does not apply" marker.

    Replaces `pytest.mark.skip` (forbidden by the anti-bypass rule): every one
    of the 110 `(connector, endpoint, scenario)` cells in `_MATRIX` is either a
    zero-argument runnable assertion or one of these, so
    `test_matrix_accounts_for_every_endpoint_scenario_cell` can assert nothing
    silently vanished.

    Attributes:
        reason: Why this cell cannot be exercised. Must be non-empty.
    """

    reason: str


# =============================================================================
# Minimal fake HTTP plumbing (self-contained; no real network)
# =============================================================================


class _Resp:
    """A minimal stand-in for a `requests.Response`."""

    def __init__(self, status_code: int, payload: Any) -> None:
        """Initialize with a status code and a `.json()` payload.

        Args:
            status_code: The HTTP status code to report.
            payload: The value `.json()` returns.
        """
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        """Return the scripted payload."""
        return self._payload


class _JsonRaisingResp:
    """A response whose `.json()` raises, simulating a truncated/malformed body."""

    def __init__(self, exc: Exception) -> None:
        """Initialize with the exception `.json()` raises.

        Args:
            exc: The exception `.json()` raises on every call.
        """
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._exc = exc

    def json(self) -> Any:
        """Raise the scripted parse failure."""
        raise self._exc


class _RouteQueueSession:
    """Serve each URL-suffix route its own independent response queue.

    Unlike `ScriptedFaultSession`'s single global FIFO (every call, regardless
    of URL, pops the next response), this session matches each call's trailing
    URL segment against a registered suffix and serves *that route's own*
    queue -- so a fault can be scripted on exactly one route (e.g. a 429 then
    a recovering 200 on `/markets`) while a sibling route (`/exchange/status`)
    serves its single clean response repeatedly. A route whose queue holds
    more than one entry pops one per call; a single-entry queue repeats
    (models a steady-state response, or a persistent fault). An unmatched
    suffix 404s, mirroring `FakeKalshiSession`'s fail-closed default.
    """

    def __init__(self, routes: Mapping[str, list[Any]]) -> None:
        """Initialize with each route's ordered response queue.

        Args:
            routes: Maps a URL suffix (e.g. `"/markets"`) to its ordered list
                of scripted responses.
        """
        self._queues = {suffix: list(responses) for suffix, responses in routes.items()}
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        timeout: int | None = None,
    ) -> Any:
        """Record the call and serve the matched route's next response.

        Args:
            url: The full request URL `KalshiClient` built.
            params: Forwarded query parameters (recorded, not used to route).
            timeout: The forwarded request timeout (recorded, not used).

        Returns:
            The next (or repeated, steady-state) response for the matched
            route, or a 404 when no registered suffix matches.
        """
        self.calls.append(url)
        for suffix, queue in self._queues.items():
            if url.endswith(suffix):
                return queue.pop(0) if len(queue) > 1 else queue[0]
        return _Resp(404, {"error": "not found"})


# =============================================================================
# Shared construction helpers
# =============================================================================


def _resilience_policy(**overrides: int) -> ResiliencePolicy:
    """Build a generous `ResiliencePolicy`, overridable per cell.

    Args:
        **overrides: Field values overriding the generous defaults below.

    Returns:
        The constructed policy.
    """
    fields: dict[str, int] = {
        "bucket_capacity": 1_000,
        "refill_interval_seconds": 10,
        "max_attempts": 3,
        "base_backoff_seconds": 1,
        "max_backoff_seconds": 30,
        "max_jitter_seconds": 0,
        "failure_threshold": 10,
        "cooldown_seconds": 60,
    }
    fields.update(overrides)
    return ResiliencePolicy(**fields)


def _rate_limited_client(
    routes: Mapping[str, list[Any]], sleeper: RecordingSleeper
) -> KalshiClient:
    """Build a `KalshiClient` over a `_RouteQueueSession`, driven by a real caller.

    Wires a real `ResilientCaller` (deterministic `FakeIntClock` /
    `RecordingSleeper` / seeded RNG) so a scripted 429-then-200 route recovers
    end to end and the sleeper's recorded calls prove a backoff wait happened
    -- mirroring how `test_client_resilience.py` wires the same three seams.

    Args:
        routes: Per-suffix response queues for the underlying `_RouteQueueSession`.
        sleeper: The recording sleeper the caller's backoff waits are recorded on.

    Returns:
        The wired client.
    """
    session = _RouteQueueSession(routes)
    resilience = ResilientCaller(
        _resilience_policy(max_attempts=3),
        InMemoryEventLedgerWriter(),
        clock=FakeIntClock(),
        sleeper=sleeper,
        rng=random.Random(20260704),
        wall_clock=_clock,
    )
    return KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=resilience,
    )


def _build_fixture_connector() -> KalshiConnector:
    """Build a `KalshiConnector` over the recorded fixtures via `FakeKalshiSession`."""
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=FakeKalshiSession(),
    )
    return KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)


def _binary_market(ticker: str, event_ticker: str, **overrides: Any) -> dict[str, Any]:
    """Build a minimal, fully-populated raw binary market payload.

    Args:
        ticker: The market ticker.
        event_ticker: The parent event ticker.
        **overrides: Fields to override.

    Returns:
        A raw market mapping shaped like a `/markets` list entry.
    """
    market: dict[str, Any] = {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "market_type": "binary",
        "title": f"{ticker} title",
        "rules_primary": f"{ticker} rules",
        "category": "Test",
        "close_time": "2024-12-18T19:00:00Z",
        "expected_expiration_time": None,
        "tick_size": 1,
        "volume_24h": 7500,
    }
    market.update(overrides)
    return market


def _paper_exchange() -> PaperExchange:
    """Build a fresh `PaperExchange` from the `deep_walk` fixture scenario."""
    return PaperExchange.from_fixture_dir(_BOOKS_FIXTURE_DIR / "deep_walk")


# =============================================================================
# Kalshi cells: list_markets
# =============================================================================


def _kalshi_list_markets_happy() -> None:
    connector = _build_fixture_connector()

    markets = connector.list_markets()

    tickers = {m.ticker for m in markets}
    assert "KXFED-24DEC" in tickers
    assert "FAKE-PERP" not in tickers
    assert "KXSCALAR-24DEC" not in tickers
    market = next(m for m in markets if m.ticker == "KXFED-24DEC")
    assert market.price_tick_pips == 100  # tick_size 1 cent -> 100 pips
    assert market.min_order_contract_centis == 100


def _kalshi_list_markets_error() -> None:
    session = _RouteQueueSession(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/events": [_Resp(200, {"events": []})],
            "/markets": [_Resp(400, {"error": "bad request"})],
        }
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=None,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(KalshiApiError) as exc_info:
        connector.list_markets()

    assert exc_info.value.status_code == 400


def _kalshi_list_markets_rate_limit() -> None:
    sleeper = RecordingSleeper()
    client = _rate_limited_client(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/events": [_Resp(200, _read_kalshi_fixture("events.json"))],
            "/markets": [
                _Resp(429, {"error": "slow down"}),
                _Resp(200, _read_kalshi_fixture("markets.json")),
            ],
        },
        sleeper,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    markets = connector.list_markets()

    assert "KXFED-24DEC" in {m.ticker for m in markets}
    assert sleeper.calls  # a backoff wait was recorded


def _kalshi_list_markets_malformed() -> None:
    broken = _binary_market("KX-BAD", "E1")
    del broken["title"]
    good = _binary_market("KX-GOOD", "E1")
    session = _RouteQueueSession(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/events": [_Resp(200, {"events": []})],
            "/markets": [_Resp(200, {"markets": [broken, good], "cursor": ""})],
        }
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=None,
    )
    ledger = InMemoryEventLedgerWriter()
    connector = KalshiConnector(client, ledger, clock=_clock)

    markets = connector.list_markets()

    assert {m.ticker for m in markets} == {"KX-GOOD"}
    malformed = ledger.events_by_type(MARKET_MALFORMED_EVENT)
    assert {event.payload["ticker"] for event in malformed} == {"KX-BAD"}


def _kalshi_list_markets_schema_drift() -> None:
    mutated_markets = copy.deepcopy(_read_kalshi_fixture("markets.json"))
    mutated_markets["unexpected_top_level_field"] = "drift"
    session = _RouteQueueSession(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/events": [_Resp(200, _read_kalshi_fixture("events.json"))],
            "/markets": [_Resp(200, mutated_markets)],
        }
    )
    ledger = InMemoryEventLedgerWriter()
    validator = SchemaValidator(
        kalshi_default_schema_registry(), ledger, wall_clock=_clock
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        validator=validator,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(SchemaAnomalyHaltError):
        connector.list_markets()

    (event,) = ledger.events_by_type(SCHEMA_ANOMALY_EVENT)
    assert "unexpected_top_level_field" in event.payload["fields"]


# =============================================================================
# Kalshi cells: get_market
# =============================================================================


def _kalshi_get_market_happy() -> None:
    connector = _build_fixture_connector()

    market = connector.get_market("KXFED-24DEC")

    assert market.ticker == "KXFED-24DEC"
    assert market.market_type == "fully_collateralized_binary"


def _kalshi_get_market_error() -> None:
    connector = _build_fixture_connector()

    with pytest.raises(UnknownMarketError):
        connector.get_market("NOPE")


def _kalshi_get_market_rate_limit() -> None:
    sleeper = RecordingSleeper()
    client = _rate_limited_client(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/markets": [
                _Resp(429, {"error": "slow down"}),
                _Resp(200, _read_kalshi_fixture("markets.json")),
            ],
            "/events": [_Resp(200, _read_kalshi_fixture("events.json"))],
        },
        sleeper,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    market = connector.get_market("KXFED-24DEC")

    assert market.ticker == "KXFED-24DEC"
    assert sleeper.calls


def _kalshi_get_market_malformed() -> None:
    broken = _binary_market("KX-BAD", "E1")
    del broken["title"]
    session = _RouteQueueSession(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/markets": [_Resp(200, {"markets": [broken], "cursor": ""})],
            "/events": [_Resp(200, {"events": []})],
        }
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=None,
    )
    ledger = InMemoryEventLedgerWriter()
    connector = KalshiConnector(client, ledger, clock=_clock)

    with pytest.raises(UnknownMarketError):
        connector.get_market("KX-BAD")

    malformed = ledger.events_by_type(MARKET_MALFORMED_EVENT)
    assert {event.payload["ticker"] for event in malformed} == {"KX-BAD"}


def _kalshi_get_market_schema_drift() -> None:
    mutated_markets = copy.deepcopy(_read_kalshi_fixture("markets.json"))
    mutated_markets["unexpected_top_level_field"] = "drift"
    session = _RouteQueueSession(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/markets": [_Resp(200, mutated_markets)],
        }
    )
    ledger = InMemoryEventLedgerWriter()
    validator = SchemaValidator(
        kalshi_default_schema_registry(), ledger, wall_clock=_clock
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        validator=validator,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(SchemaAnomalyHaltError):
        connector.get_market("KXFED-24DEC")

    (event,) = ledger.events_by_type(SCHEMA_ANOMALY_EVENT)
    assert "unexpected_top_level_field" in event.payload["fields"]


# =============================================================================
# Kalshi cells: get_order_book
# =============================================================================


def _kalshi_get_order_book_happy() -> None:
    connector = _build_fixture_connector()

    book = connector.get_order_book("KXFED-24DEC")

    assert book.ticker == "KXFED-24DEC"
    assert book.yes_bids[0].price == PricePips(4500)
    assert book.yes_bids[0].quantity == ContractCentis(10_000)
    assert book.yes_asks[0].price == PricePips(4800)
    assert book.yes_asks[0].quantity == ContractCentis(4_000)
    assert book.fetched_at == _clock()


def _kalshi_get_order_book_error() -> None:
    connector = _build_fixture_connector()

    with pytest.raises(UnknownMarketError):
        connector.get_order_book("NOPE")


def _kalshi_get_order_book_rate_limit() -> None:
    sleeper = RecordingSleeper()
    client = _rate_limited_client(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/markets": [_Resp(200, _read_kalshi_fixture("markets.json"))],
            "/orderbook": [
                _Resp(429, {"error": "slow down"}),
                _Resp(200, _read_kalshi_fixture("orderbook_KXFED-24DEC.json")),
            ],
        },
        sleeper,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    book = connector.get_order_book("KXFED-24DEC")

    assert book.ticker == "KXFED-24DEC"
    assert sleeper.calls


def _kalshi_get_order_book_malformed() -> None:
    session = _RouteQueueSession(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/markets": [_Resp(200, _read_kalshi_fixture("markets.json"))],
            "/orderbook": [_JsonRaisingResp(ValueError("truncated body"))],
        }
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience_policy=_resilience_policy(
            max_attempts=2,
            base_backoff_seconds=0,
            max_backoff_seconds=0,
            max_jitter_seconds=0,
            failure_threshold=99,
            cooldown_seconds=999,
        ),
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(ValueError, match="truncated body"):
        connector.get_order_book("KXFED-24DEC")


def _kalshi_get_order_book_schema_drift() -> None:
    drift_payload = _read_kalshi_fixture("faults/orderbook_drift_money_fee.json")
    session = _RouteQueueSession(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/markets": [_Resp(200, _read_kalshi_fixture("markets.json"))],
            "/orderbook": [_Resp(200, drift_payload)],
        }
    )
    ledger = InMemoryEventLedgerWriter()
    validator = SchemaValidator(
        kalshi_default_schema_registry(), ledger, wall_clock=_clock
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        validator=validator,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(SchemaAnomalyHaltError):
        connector.get_order_book("KXFED-24DEC")

    (event,) = ledger.events_by_type(SCHEMA_ANOMALY_EVENT)
    assert "fee" in event.payload["fields"]


# =============================================================================
# Kalshi cells: get_exchange_status / get_exchange_time
# =============================================================================


def _kalshi_get_exchange_status_happy() -> None:
    connector = _build_fixture_connector()

    status = connector.get_exchange_status()

    assert status.status == "open"
    assert status.fetched_at == _clock()


def _kalshi_get_exchange_status_error() -> None:
    session = _RouteQueueSession(
        {"/exchange/status": [_Resp(400, {"error": "bad request"})]}
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=None,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(KalshiApiError) as exc_info:
        connector.get_exchange_status()

    assert exc_info.value.status_code == 400


def _kalshi_get_exchange_status_rate_limit() -> None:
    sleeper = RecordingSleeper()
    client = _rate_limited_client(
        {
            "/exchange/status": [
                _Resp(429, {"error": "slow down"}),
                _Resp(200, {"exchange_active": True, "trading_active": True}),
            ]
        },
        sleeper,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    status = connector.get_exchange_status()

    assert status.status == "open"
    assert sleeper.calls


def _kalshi_get_exchange_status_malformed() -> None:
    session = _RouteQueueSession(
        {"/exchange/status": [_JsonRaisingResp(ValueError("truncated body"))]}
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience_policy=_resilience_policy(
            max_attempts=2,
            base_backoff_seconds=0,
            max_backoff_seconds=0,
            max_jitter_seconds=0,
            failure_threshold=99,
            cooldown_seconds=999,
        ),
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(ValueError, match="truncated body"):
        connector.get_exchange_status()


def _kalshi_get_exchange_status_schema_drift() -> None:
    mutated_status = {"exchange_active": True, "trading_active": True, "unexpected": 1}
    session = _RouteQueueSession({"/exchange/status": [_Resp(200, mutated_status)]})
    ledger = InMemoryEventLedgerWriter()
    validator = SchemaValidator(
        kalshi_default_schema_registry(), ledger, wall_clock=_clock
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        validator=validator,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(SchemaAnomalyHaltError):
        connector.get_exchange_status()

    (event,) = ledger.events_by_type(SCHEMA_ANOMALY_EVENT)
    assert "unexpected" in event.payload["fields"]


def _kalshi_get_exchange_time_happy() -> None:
    connector = _build_fixture_connector()

    server_time = connector.get_exchange_time()

    assert server_time == _FIXTURE_SERVER_DATE


def _kalshi_get_exchange_time_error() -> None:
    session = _RouteQueueSession(
        {"/exchange/status": [_Resp(400, {"error": "bad request"})]}
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=None,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(KalshiApiError) as exc_info:
        connector.get_exchange_time()

    assert exc_info.value.status_code == 400


def _kalshi_get_exchange_time_rate_limit() -> None:
    sleeper = RecordingSleeper()
    client = _rate_limited_client(
        {
            "/exchange/status": [
                _Resp(429, {"error": "slow down"}),
                _Resp(200, {"exchange_active": True, "trading_active": True}),
            ]
        },
        sleeper,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    server_time = connector.get_exchange_time()

    # The scripted response carries no `Date` header -> falls back to the clock.
    assert server_time == _clock()
    assert sleeper.calls


def _kalshi_get_exchange_time_malformed() -> None:
    session = _RouteQueueSession(
        {"/exchange/status": [_JsonRaisingResp(ValueError("truncated body"))]}
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience_policy=_resilience_policy(
            max_attempts=2,
            base_backoff_seconds=0,
            max_backoff_seconds=0,
            max_jitter_seconds=0,
            failure_threshold=99,
            cooldown_seconds=999,
        ),
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(ValueError, match="truncated body"):
        connector.get_exchange_time()


def _kalshi_get_exchange_time_schema_drift() -> None:
    mutated_status = {"exchange_active": True, "trading_active": True, "unexpected": 1}
    session = _RouteQueueSession({"/exchange/status": [_Resp(200, mutated_status)]})
    ledger = InMemoryEventLedgerWriter()
    validator = SchemaValidator(
        kalshi_default_schema_registry(), ledger, wall_clock=_clock
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        validator=validator,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(SchemaAnomalyHaltError):
        connector.get_exchange_time()

    (event,) = ledger.events_by_type(SCHEMA_ANOMALY_EVENT)
    assert "unexpected" in event.payload["fields"]


# =============================================================================
# Kalshi cells: get_balance_semantics (constant; no transport)
# =============================================================================


def _kalshi_get_balance_semantics_happy() -> None:
    connector = _build_fixture_connector()

    assert connector.get_balance_semantics() is KALSHI_BALANCE_SEMANTICS


# =============================================================================
# Kalshi cells: get_balances / get_positions / get_open_orders / get_fills
# (deferred to issue #3: happy == the current NotImplementedError contract)
# =============================================================================


def _kalshi_get_balances_happy() -> None:
    connector = _build_fixture_connector()

    with pytest.raises(NotImplementedError) as exc_info:
        connector.get_balances()

    assert str(exc_info.value) == _EXPECTED_ACCOUNT_DEFERRAL_MESSAGE


def _kalshi_get_positions_happy() -> None:
    connector = _build_fixture_connector()

    with pytest.raises(NotImplementedError) as exc_info:
        connector.get_positions()

    assert str(exc_info.value) == _EXPECTED_ACCOUNT_DEFERRAL_MESSAGE


def _kalshi_get_open_orders_happy() -> None:
    connector = _build_fixture_connector()

    with pytest.raises(NotImplementedError) as exc_info:
        connector.get_open_orders()

    assert str(exc_info.value) == _EXPECTED_ACCOUNT_DEFERRAL_MESSAGE


def _kalshi_get_fills_happy() -> None:
    connector = _build_fixture_connector()

    with pytest.raises(NotImplementedError) as exc_info:
        connector.get_fills(datetime(2024, 1, 1, tzinfo=UTC))

    assert str(exc_info.value) == _EXPECTED_ACCOUNT_DEFERRAL_MESSAGE


# =============================================================================
# Kalshi cells: get_fee_model
# =============================================================================


def _kalshi_get_fee_model_happy() -> None:
    connector = _build_fixture_connector()

    fee_model = connector.get_fee_model("KXFED-24DEC")

    assert fee_model.schedule_id == "kxfed-standard-v1"
    assert fee_model.maker_fee_ppm == 0
    assert fee_model.taker_fee_ppm == 70_000
    assert fee_model.settlement_fee_ppm == 0


def _kalshi_get_fee_model_error() -> None:
    connector = _build_fixture_connector()

    with pytest.raises(UnknownFeeModelError):
        connector.get_fee_model("NOPE-24DEC")


def _kalshi_get_fee_model_rate_limit() -> None:
    sleeper = RecordingSleeper()
    client = _rate_limited_client(
        {
            "/series/KXFED": [
                _Resp(429, {"error": "slow down"}),
                _Resp(200, _read_kalshi_fixture("series_KXFED.json")),
            ]
        },
        sleeper,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    fee_model = connector.get_fee_model("KXFED-24DEC")

    assert fee_model.schedule_id == "kxfed-standard-v1"
    assert sleeper.calls


def _kalshi_get_fee_model_malformed() -> None:
    session = _RouteQueueSession(
        {"/series/KXBAD": [_Resp(200, _read_kalshi_fixture("series_KXBAD.json"))]}
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=None,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(UnknownFeeModelError):
        connector.get_fee_model("KXBAD-24DEC")


def _kalshi_get_fee_model_schema_drift() -> None:
    mutated_series = copy.deepcopy(_read_kalshi_fixture("series_KXFED.json"))
    mutated_series["series"]["unexpected_leaf"] = 1
    session = _RouteQueueSession({"/series/KXFED": [_Resp(200, mutated_series)]})
    ledger = InMemoryEventLedgerWriter()
    validator = SchemaValidator(
        kalshi_default_schema_registry(), ledger, wall_clock=_clock
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        validator=validator,
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    with pytest.raises(SchemaAnomalyHaltError):
        connector.get_fee_model("KXFED-24DEC")

    (event,) = ledger.events_by_type(SCHEMA_ANOMALY_EVENT)
    assert "unexpected_leaf" in event.payload["fields"]


# =============================================================================
# Paper cells: happy (all 11 endpoints)
# =============================================================================


def _paper_list_markets_happy() -> None:
    exchange = _paper_exchange()

    markets = exchange.list_markets()

    assert {m.ticker for m in markets} == {"MKT-DEEP"}


def _paper_get_market_happy() -> None:
    exchange = _paper_exchange()

    market = exchange.get_market("MKT-DEEP")

    assert market.ticker == "MKT-DEEP"


def _paper_get_order_book_happy() -> None:
    exchange = _paper_exchange()

    book = exchange.get_order_book("MKT-DEEP")

    assert book.yes_bids[0].price == PricePips(4500)
    assert book.yes_bids[0].quantity == ContractCentis(300)
    assert book.yes_asks[0].price == PricePips(4600)
    assert book.yes_asks[0].quantity == ContractCentis(200)
    assert book.fetched_at == datetime(2025, 1, 1, tzinfo=UTC)


def _paper_get_exchange_status_happy() -> None:
    exchange = _paper_exchange()

    assert exchange.get_exchange_status().status == "open"


def _paper_get_exchange_time_happy() -> None:
    exchange = _paper_exchange()

    assert exchange.get_exchange_time() == datetime(2025, 1, 1, tzinfo=UTC)


def _paper_get_balance_semantics_happy() -> None:
    exchange = _paper_exchange()

    semantics = exchange.get_balance_semantics()

    expected = PartialFillRepresentation.PER_FILL_RECORDS
    assert semantics.partial_fill_representation is expected


def _paper_get_balances_happy() -> None:
    exchange = _paper_exchange()

    balances = exchange.get_balances()

    assert balances.total == MoneyMicros(100_000_000)
    assert balances.available == MoneyMicros(100_000_000)


def _paper_get_positions_happy() -> None:
    exchange = _paper_exchange()

    assert exchange.get_positions() == ()


def _paper_get_open_orders_happy() -> None:
    exchange = _paper_exchange()

    assert exchange.get_open_orders() == ()


def _paper_get_fills_happy() -> None:
    exchange = _paper_exchange()

    assert exchange.get_fills(datetime(2000, 1, 1, tzinfo=UTC)) == ()


def _paper_get_fee_model_happy() -> None:
    exchange = _paper_exchange()

    fee_model = exchange.get_fee_model("MKT-DEEP")

    assert fee_model.schedule_id == "paper-test-v1"
    assert fee_model.taker_fee_ppm == 70_000


# =============================================================================
# Paper cells: error (only the two ticker-taking reads have a failure path)
# =============================================================================


def _paper_get_market_error() -> None:
    exchange = _paper_exchange()

    with pytest.raises(UnknownMarketError):
        exchange.get_market("NOPE")


def _paper_get_order_book_error() -> None:
    exchange = _paper_exchange()

    with pytest.raises(UnknownMarketError):
        exchange.get_order_book("NOPE")


# =============================================================================
# Paper cells: malformed (from_fixture_dir loads everything eagerly)
# =============================================================================


def _paper_list_markets_malformed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        shutil.copytree(_BOOKS_FIXTURE_DIR / "deep_walk", tmp_dir, dirs_exist_ok=True)
        markets_path = tmp_dir / "markets.json"
        markets = json.loads(markets_path.read_text(encoding="utf-8"))
        del markets[0]["title"]  # drop a required key -> KeyError at load time
        markets_path.write_text(json.dumps(markets), encoding="utf-8")

        with pytest.raises(KeyError):
            PaperExchange.from_fixture_dir(tmp_dir)


def _paper_get_order_book_malformed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        shutil.copytree(_BOOKS_FIXTURE_DIR / "deep_walk", tmp_dir, dirs_exist_ok=True)
        sessions_path = tmp_dir / "sessions.json"
        sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
        # A float money/price leaf -> PricePips.__post_init__ rejects non-int.
        sessions["MKT-DEEP"][0]["book"]["yes_bids"][0]["price"] = 4500.5
        sessions_path.write_text(json.dumps(sessions), encoding="utf-8")

        with pytest.raises(TypeError):
            PaperExchange.from_fixture_dir(tmp_dir)


# =============================================================================
# _MATRIX: registration
# =============================================================================

_MATRIX: dict[tuple[str, str, str], Callable[[], None] | NotApplicable] = {
    ("kalshi", "list_markets", "happy"): _kalshi_list_markets_happy,
    ("kalshi", "list_markets", "error"): _kalshi_list_markets_error,
    ("kalshi", "list_markets", "rate_limit"): _kalshi_list_markets_rate_limit,
    ("kalshi", "list_markets", "malformed"): _kalshi_list_markets_malformed,
    ("kalshi", "list_markets", "schema_drift"): _kalshi_list_markets_schema_drift,
    ("kalshi", "get_market", "happy"): _kalshi_get_market_happy,
    ("kalshi", "get_market", "error"): _kalshi_get_market_error,
    ("kalshi", "get_market", "rate_limit"): _kalshi_get_market_rate_limit,
    ("kalshi", "get_market", "malformed"): _kalshi_get_market_malformed,
    ("kalshi", "get_market", "schema_drift"): _kalshi_get_market_schema_drift,
    ("kalshi", "get_order_book", "happy"): _kalshi_get_order_book_happy,
    ("kalshi", "get_order_book", "error"): _kalshi_get_order_book_error,
    ("kalshi", "get_order_book", "rate_limit"): _kalshi_get_order_book_rate_limit,
    ("kalshi", "get_order_book", "malformed"): _kalshi_get_order_book_malformed,
    ("kalshi", "get_order_book", "schema_drift"): _kalshi_get_order_book_schema_drift,
    ("kalshi", "get_exchange_status", "happy"): _kalshi_get_exchange_status_happy,
    ("kalshi", "get_exchange_status", "error"): _kalshi_get_exchange_status_error,
    (
        "kalshi",
        "get_exchange_status",
        "rate_limit",
    ): _kalshi_get_exchange_status_rate_limit,
    (
        "kalshi",
        "get_exchange_status",
        "malformed",
    ): _kalshi_get_exchange_status_malformed,
    (
        "kalshi",
        "get_exchange_status",
        "schema_drift",
    ): _kalshi_get_exchange_status_schema_drift,
    ("kalshi", "get_exchange_time", "happy"): _kalshi_get_exchange_time_happy,
    ("kalshi", "get_exchange_time", "error"): _kalshi_get_exchange_time_error,
    ("kalshi", "get_exchange_time", "rate_limit"): _kalshi_get_exchange_time_rate_limit,
    ("kalshi", "get_exchange_time", "malformed"): _kalshi_get_exchange_time_malformed,
    (
        "kalshi",
        "get_exchange_time",
        "schema_drift",
    ): _kalshi_get_exchange_time_schema_drift,
    ("kalshi", "get_balance_semantics", "happy"): _kalshi_get_balance_semantics_happy,
    ("kalshi", "get_balances", "happy"): _kalshi_get_balances_happy,
    ("kalshi", "get_positions", "happy"): _kalshi_get_positions_happy,
    ("kalshi", "get_open_orders", "happy"): _kalshi_get_open_orders_happy,
    ("kalshi", "get_fills", "happy"): _kalshi_get_fills_happy,
    ("kalshi", "get_fee_model", "happy"): _kalshi_get_fee_model_happy,
    ("kalshi", "get_fee_model", "error"): _kalshi_get_fee_model_error,
    ("kalshi", "get_fee_model", "rate_limit"): _kalshi_get_fee_model_rate_limit,
    ("kalshi", "get_fee_model", "malformed"): _kalshi_get_fee_model_malformed,
    ("kalshi", "get_fee_model", "schema_drift"): _kalshi_get_fee_model_schema_drift,
    ("paper", "list_markets", "happy"): _paper_list_markets_happy,
    ("paper", "get_market", "happy"): _paper_get_market_happy,
    ("paper", "get_order_book", "happy"): _paper_get_order_book_happy,
    ("paper", "get_exchange_status", "happy"): _paper_get_exchange_status_happy,
    ("paper", "get_exchange_time", "happy"): _paper_get_exchange_time_happy,
    ("paper", "get_balance_semantics", "happy"): _paper_get_balance_semantics_happy,
    ("paper", "get_balances", "happy"): _paper_get_balances_happy,
    ("paper", "get_positions", "happy"): _paper_get_positions_happy,
    ("paper", "get_open_orders", "happy"): _paper_get_open_orders_happy,
    ("paper", "get_fills", "happy"): _paper_get_fills_happy,
    ("paper", "get_fee_model", "happy"): _paper_get_fee_model_happy,
    ("paper", "get_market", "error"): _paper_get_market_error,
    ("paper", "get_order_book", "error"): _paper_get_order_book_error,
    ("paper", "list_markets", "malformed"): _paper_list_markets_malformed,
    ("paper", "get_order_book", "malformed"): _paper_get_order_book_malformed,
}

#: The four kalshi account-access methods deferred to issue #3 (SPEC S7.2):
#: every non-happy scenario is N/A because no transport is ever dialed.
_KALSHI_DEFERRED_ACCOUNT_ENDPOINTS: Final[tuple[str, ...]] = (
    "get_balances",
    "get_positions",
    "get_open_orders",
    "get_fills",
)

for _endpoint in _KALSHI_DEFERRED_ACCOUNT_ENDPOINTS:
    for _scenario in _NON_HAPPY_SCENARIOS:
        _MATRIX[("kalshi", _endpoint, _scenario)] = NotApplicable(
            f"{_endpoint} is deferred to issue #3 (see the 'happy' cell's "
            "NotImplementedError contract); no transport is dialed, so this "
            "scenario cannot occur."
        )

for _scenario in _NON_HAPPY_SCENARIOS:
    _MATRIX[("kalshi", "get_balance_semantics", _scenario)] = NotApplicable(
        "get_balance_semantics returns a static in-memory record; no "
        "transport is dialed, so this scenario cannot occur."
    )

#: Reasons for the 9 paper endpoints with no defined "error" contract (only
#: `get_market` / `get_order_book` can fail on an unknown ticker).
_PAPER_ERROR_NA_REASONS: Final[dict[str, str]] = {
    "list_markets": (
        "list_markets takes no argument and always returns the loaded "
        "fixture's markets; there is no ticker input that can be unknown."
    ),
    "get_exchange_status": "returns the static fixture status; no failure path exists.",
    "get_exchange_time": "returns the static fixture time; no failure path exists.",
    "get_balance_semantics": (
        "returns the static fixture semantics; no failure path exists."
    ),
    "get_balances": "returns the static fixture balances; no failure path exists.",
    "get_positions": "always returns an empty tuple; no failure path exists.",
    "get_open_orders": (
        "returns the current resting-order list; a fresh exchange has no "
        "failure path here."
    ),
    "get_fills": "returns a filtered tuple for any `since`; no failure path exists.",
    "get_fee_model": (
        "an unrecognized ticker falls back to the 'default' schedule (see "
        "the 'happy' cell); there is no unknown-fee-model failure path."
    ),
}

for _endpoint, _reason in _PAPER_ERROR_NA_REASONS.items():
    _MATRIX[("paper", _endpoint, "error")] = NotApplicable(_reason)

#: The 9 paper endpoints whose "malformed" cell is N/A: `PaperExchange.from_fixture_dir`
#: loads every fixture eagerly at construction, so a corrupt fixture fails
#: closed before *any* endpoint can be exercised -- pinned once for
#: `list_markets` (a dropped required key) and once for `get_order_book`
#: (a float money/price leaf), not redundantly for every other read.
for _endpoint in _ENDPOINTS:
    if _endpoint in ("list_markets", "get_order_book"):
        continue
    _MATRIX[("paper", _endpoint, "malformed")] = NotApplicable(
        "PaperExchange.from_fixture_dir loads every fixture eagerly at "
        "construction time; a corrupt fixture fails closed before any "
        "endpoint can be exercised -- see the list_markets / get_order_book "
        "malformed cells for the two representative failure modes."
    )

#: Paper has no network transport at all: every endpoint's rate_limit and
#: schema_drift cell is N/A for the identical reason.
for _endpoint in _ENDPOINTS:
    _MATRIX[("paper", _endpoint, "rate_limit")] = NotApplicable(
        "PaperExchange replays local fixtures; there is no network "
        "transport to rate-limit."
    )
    _MATRIX[("paper", _endpoint, "schema_drift")] = NotApplicable(
        "PaperExchange replays local fixtures; there is no wire payload to drift."
    )


# =============================================================================
# The parametrized matrix test + completeness meta-test
# =============================================================================


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize `test_matrix_cell` over every registered `_MATRIX` entry.

    Args:
        metafunc: The pytest metafunc for the collected test function.
    """
    if "cell_key" in metafunc.fixturenames:
        keys = sorted(_MATRIX)
        ids = [
            f"{connector}-{endpoint}-{scenario}"
            for connector, endpoint, scenario in keys
        ]
        metafunc.parametrize("cell_key", keys, ids=ids)


def test_matrix_cell(cell_key: tuple[str, str, str]) -> None:
    """Execute one matrix cell: a runnable assertion, or an explicit N/A marker.

    Args:
        cell_key: The `(connector, endpoint, scenario)` key to execute.
    """
    entry = _MATRIX[cell_key]
    if isinstance(entry, NotApplicable):
        assert entry.reason, f"{cell_key} is marked N/A with no reason"
        return
    entry()


def test_matrix_accounts_for_every_endpoint_scenario_cell() -> None:
    """All 2 x 11 x 5 = 110 `(connector, endpoint, scenario)` cells are accounted for.

    Every cell in `_MATRIX` must be one of the 110 valid keys, no more, no
    fewer.
    """
    expected = {
        (connector, endpoint, scenario)
        for connector in _CONNECTORS
        for endpoint in _ENDPOINTS
        for scenario in _SCENARIOS
    }

    assert len(expected) == 110
    assert set(_MATRIX) == expected


def test_matrix_reports_the_runnable_versus_not_applicable_split() -> None:
    """Sanity guard on the matrix's shape: 50 runnable cells, 60 N/A cells."""
    runnable = sum(
        1 for entry in _MATRIX.values() if not isinstance(entry, NotApplicable)
    )
    not_applicable = sum(
        1 for entry in _MATRIX.values() if isinstance(entry, NotApplicable)
    )

    assert runnable == 50
    assert not_applicable == 60
    assert runnable + not_applicable == 110


# =============================================================================
# Supplements: protocol conformance, cosmetic-drift tolerance
# =============================================================================


def test_kalshi_connector_satisfies_the_market_connector_protocol() -> None:
    """`KalshiConnector` structurally satisfies the runtime-checkable protocol."""
    assert isinstance(_build_fixture_connector(), MarketConnector)


def test_paper_exchange_satisfies_the_market_connector_protocol() -> None:
    """`PaperExchange` structurally satisfies the runtime-checkable protocol."""
    assert isinstance(_paper_exchange(), MarketConnector)


def test_kalshi_order_book_cosmetic_schema_drift_is_tolerated_not_halted() -> None:
    """The cosmetic sibling of the money-drift matrix cell: warns, never halts.

    Supplements `("kalshi", "get_order_book", "schema_drift")` (the money-field
    drift that halts) with the other documented half of SPEC S3 principle 3:
    `orderbook_drift_cosmetic.json`'s allowlisted extra field only warns, and
    the call still returns.
    """
    drift_payload = _read_kalshi_fixture("faults/orderbook_drift_cosmetic.json")
    session = _RouteQueueSession(
        {
            "/exchange/status": [
                _Resp(200, {"exchange_active": True, "trading_active": True})
            ],
            "/markets": [_Resp(200, _read_kalshi_fixture("markets.json"))],
            "/orderbook": [_Resp(200, drift_payload)],
        }
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL, allowlist=_FAKE_ALLOWLIST, timeout=5, session=session
    )
    connector = KalshiConnector(client, InMemoryEventLedgerWriter(), clock=_clock)

    book = connector.get_order_book("KXFED-24DEC")

    assert book.ticker == "KXFED-24DEC"


# =============================================================================
# Float / fixed-point preservation guard (SPEC S17.6 connector-boundary check)
# =============================================================================


def _iter_leaves(obj: object, path: str = "$") -> list[tuple[str, object]]:
    """Recursively enumerate every non-container leaf reachable from `obj`.

    Recurses into dataclass fields (including windbreak's scaled-integer unit
    types, themselves frozen dataclasses wrapping a single `.value: int`),
    mapping values, and list/tuple elements. Every other value -- datetimes,
    plain ints, strings, bools, enum members, `None` -- is a leaf.

    Args:
        obj: The value to walk.
        path: The accumulated path rendering, for a readable failure message.

    Returns:
        Every `(path, leaf_value)` pair reachable from `obj`.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        leaves: list[tuple[str, object]] = []
        for field in dataclasses.fields(obj):
            leaf_path = f"{path}.{field.name}"
            leaves.extend(_iter_leaves(getattr(obj, field.name), leaf_path))
        return leaves
    if isinstance(obj, Mapping):
        leaves = []
        for key, value in obj.items():
            leaves.extend(_iter_leaves(value, f"{path}[{key!r}]"))
        return leaves
    if isinstance(obj, (list, tuple)):
        leaves = []
        for index, value in enumerate(obj):
            leaves.extend(_iter_leaves(value, f"{path}[{index}]"))
        return leaves
    return [(path, obj)]


def _assert_no_float_leaves(obj: object, path: str = "$") -> None:
    """Assert no leaf reachable from `obj` is a `float` instance.

    Args:
        obj: The value to walk.
        path: The accumulated path rendering, for a readable failure message.
    """
    for leaf_path, value in _iter_leaves(obj, path):
        message = f"float leaf found at {leaf_path}: {value!r}"
        assert not isinstance(value, float), message


def test_kalshi_happy_path_values_carry_no_float_leaf() -> None:
    """Every implemented Kalshi read's happy-path value is entirely float-free."""
    connector = _build_fixture_connector()

    values: list[object] = [
        connector.list_markets(),
        connector.get_market("KXFED-24DEC"),
        connector.get_order_book("KXFED-24DEC"),
        connector.get_exchange_status(),
        connector.get_exchange_time(),
        connector.get_balance_semantics(),
        connector.get_fee_model("KXFED-24DEC"),
    ]

    for value in values:
        _assert_no_float_leaves(value)


def test_paper_happy_path_values_carry_no_float_leaf() -> None:
    """Every `PaperExchange` read's happy-path value is entirely float-free."""
    exchange = _paper_exchange()

    values: list[object] = [
        exchange.list_markets(),
        exchange.get_market("MKT-DEEP"),
        exchange.get_order_book("MKT-DEEP"),
        exchange.get_exchange_status(),
        exchange.get_exchange_time(),
        exchange.get_balance_semantics(),
        exchange.get_balances(),
        exchange.get_positions(),
        exchange.get_open_orders(),
        exchange.get_fills(datetime(2000, 1, 1, tzinfo=UTC)),
        exchange.get_fee_model("MKT-DEEP"),
    ]

    for value in values:
        _assert_no_float_leaves(value)


def test_lint_no_floats_passes_over_windbreak_connector() -> None:
    """The AST float-lint additionally guards `windbreak/connector` cleanly.

    Loads `scripts/lint_no_floats.py` by file path (it lives outside the
    `windbreak` package), mirroring `tests/numeric/test_float_lint.py`'s
    `importlib.util.spec_from_file_location` pattern. This is a light,
    additive connector-owned regression guard -- the repo-wide float-lint
    test already covers the same package, so this is not a duplicate suite.
    """
    spec = importlib.util.spec_from_file_location("lint_no_floats", _LINT_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module: types.ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    exit_code = module.main([str(_REPO_ROOT / "windbreak" / "connector")])

    assert exit_code == 0
