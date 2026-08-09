"""End-to-end tests for issue #20's client/adapter wiring.

Ties `windbreak.connector.resilience` and `windbreak.connector.validation`
into `KalshiClient` / `KalshiConnector`, exercised through the fake sessions
in `tests/connector/kalshi/conftest.py`:

* `KalshiClient.get()` runs the transport+parse through a `ResilientCaller`
  (on by default -- see `build_default_resilient_caller` -- or an injected one;
  `resilience=None` opts out to raw single-attempt transport), then always
  runs the (on-by-default) `SchemaValidator` -- a `SchemaAnomalyHaltError`
  from that validator step runs *outside* the retry loop: it is never retried
  and never counted against the circuit breaker. Retryable `5xx` / `429` /
  malformed-body faults are retried, then either recover or fail closed.
* `KalshiConnector._ensure_operational()` fetches exchange status at the top
  of every snapshot fetch (`list_markets` / `get_market` / `get_order_book`);
  a non-`"open"` status ledgers one `CONNECTOR_HALT` (`reason="maintenance"`)
  and raises `MaintenanceHaltError` before any further transport happens.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from windbreak.connector.kalshi.adapter import KalshiConnector
from windbreak.connector.kalshi.client import KalshiApiError, KalshiClient
from windbreak.connector.resilience import (
    CONNECTOR_HALT_EVENT,
    ConnectorHaltError,
    MaintenanceHaltError,
    ResiliencePolicy,
    ResilientCaller,
    build_default_resilient_caller,
)
from windbreak.connector.validation import (
    SCHEMA_ANOMALY_EVENT,
    SchemaAnomalyHaltError,
    SchemaValidator,
    kalshi_default_schema_registry,
)
from windbreak.net.allowlist import OutboundAllowlist

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import datetime

    from windbreak.connector.snapshot import InMemoryEventLedgerWriter

#: `tests/connector/kalshi/test_client_resilience.py` -> `tests/` ->
#: `tests/fixtures/exchange/kalshi/`.
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "exchange" / "kalshi"

_FAKE_BASE_URL = "https://fake-kalshi.test"

#: Allowlist admitting the fake host, now that ``KalshiClient`` enforces its
#: base URL host at construction (issue #57).
_FAKE_ALLOWLIST = OutboundAllowlist(frozenset({"fake-kalshi.test"}))


def _read_fixture(relative_name: str) -> Any:
    """Parse one recorded or fault-drift Kalshi JSON fixture.

    Args:
        relative_name: The path relative to `tests/fixtures/exchange/kalshi/`,
            e.g. `"faults/orderbook_drift_money_fee.json"`.

    Returns:
        The parsed JSON.
    """
    return json.loads((_FIXTURE_DIR / relative_name).read_text(encoding="utf-8"))


def _resilience_policy(**overrides: int) -> ResiliencePolicy:
    """Build a generous `ResiliencePolicy`, overridable per test.

    Args:
        **overrides: Field values overriding the defaults below.

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
        "failure_threshold": 3,
        "cooldown_seconds": 60,
    }
    fields.update(overrides)
    return ResiliencePolicy(**fields)


class _SingleRouteSession:
    """Serves one fixed payload to every `.get()` call, regardless of URL.

    Used for the schema-drift tests below, which only need one route (the
    order book) and care about the number of underlying HTTP calls, not
    per-route dispatch.
    """

    def __init__(self, status_code: int, payload: Any) -> None:
        """Initialize with the single response every call returns.

        Args:
            status_code: The HTTP status code every call reports.
            payload: The value `.json()` returns on every call.
        """
        self._status_code = status_code
        self._payload = payload
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        timeout: int | None = None,
    ) -> _SingleRouteSession._Response:
        """Record the call and return the fixed scripted response.

        Args:
            url: The full request URL `KalshiClient` built.
            params: Forwarded query parameters (unused).
            timeout: The forwarded request timeout (unused).

        Returns:
            The fixed scripted response.
        """
        self.calls.append(url)
        return self._Response(self._status_code, self._payload)

    class _Response:
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


class _MaintenanceSession:
    """Serves a scripted `/exchange/status` flag pair plus the real fixtures.

    Backs `_ensure_operational` tests: the order book and markets/events
    routes serve the real recorded fixtures (`markets.json` / `events.json` /
    `orderbook_KXFED-24DEC.json`), so only the exchange-status flags vary
    between test cases.
    """

    def __init__(self, *, exchange_active: bool, trading_active: bool) -> None:
        """Initialize with the exchange-status flag pair to serve.

        Args:
            exchange_active: The `exchange_active` flag to report.
            trading_active: The `trading_active` flag to report.
        """
        self._status = {
            "exchange_active": exchange_active,
            "trading_active": trading_active,
        }
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        timeout: int | None = None,
    ) -> _MaintenanceSession._Response:
        """Route by URL suffix to the status flags or a fixture data payload.

        Args:
            url: The full request URL `KalshiClient` built.
            params: Forwarded query parameters (unused).
            timeout: The forwarded request timeout (unused).

        Returns:
            The scripted response for the matched route.
        """
        self.calls.append(url)
        if url.endswith("/exchange/status"):
            return self._Response(200, self._status)
        if url.endswith("/orderbook"):
            return self._Response(200, _read_fixture("orderbook_KXFED-24DEC.json"))
        if url.endswith("/markets"):
            return self._Response(200, _read_fixture("markets.json"))
        if url.endswith("/events"):
            return self._Response(200, _read_fixture("events.json"))
        return self._Response(404, {"error": "not found"})

    class _Response:
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


# =============================================================================
# ResilientCaller wired through KalshiClient.get()
# =============================================================================


def test_get_recovers_after_two_5xx_then_a_200(
    scripted_fault_session: Callable[[list[Any]], Any],
    queued_fault_response: Callable[..., Any],
    ledger: InMemoryEventLedgerWriter,
    fake_int_clock: Callable[[], int],
    recording_sleeper: Callable[[int], None],
    seeded_rng: random.Random,
    clock: Callable[[], datetime],
) -> None:
    """Two transient 500s followed by a 200 succeed, transparently retried."""
    orderbook_payload = _read_fixture("orderbook_KXFED-24DEC.json")
    session = scripted_fault_session(
        [
            queued_fault_response(500, {"error": "boom"}),
            queued_fault_response(500, {"error": "boom"}),
            queued_fault_response(200, orderbook_payload),
        ]
    )
    resilience = ResilientCaller(
        _resilience_policy(max_attempts=3),
        ledger,
        clock=fake_int_clock,
        sleeper=recording_sleeper,
        rng=seeded_rng,
        wall_clock=clock,
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=resilience,
    )

    response = client.get("markets", "KXFED-24DEC", "orderbook")

    assert response.payload == orderbook_payload
    assert len(session.calls) == 3


def test_persistent_5xx_exhausts_retries_and_eventually_trips_the_breaker(
    scripted_fault_session: Callable[[list[Any]], Any],
    queued_fault_response: Callable[..., Any],
    ledger: InMemoryEventLedgerWriter,
    fake_int_clock: Callable[[], int],
    recording_sleeper: Callable[[int], None],
    seeded_rng: random.Random,
    clock: Callable[[], datetime],
) -> None:
    """A venue stuck returning 500s exhausts retries, then halts via the breaker."""
    always_500 = [queued_fault_response(500, {"error": "boom"}) for _ in range(20)]
    session = scripted_fault_session(always_500)
    resilience = ResilientCaller(
        _resilience_policy(max_attempts=2, failure_threshold=2, cooldown_seconds=999),
        ledger,
        clock=fake_int_clock,
        sleeper=recording_sleeper,
        rng=seeded_rng,
        wall_clock=clock,
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=resilience,
    )

    with pytest.raises(KalshiApiError):
        client.get(
            "markets", "KXFED-24DEC", "orderbook"
        )  # call #1: 1st breaker failure
    with pytest.raises(KalshiApiError):
        client.get("markets", "KXFED-24DEC", "orderbook")  # call #2: trips OPEN

    with pytest.raises(ConnectorHaltError):
        client.get("markets", "KXFED-24DEC", "orderbook")  # call #3: breaker OPEN

    assert len(ledger.events_by_type(CONNECTOR_HALT_EVENT)) == 1


def test_get_recovers_after_a_429_then_a_200(
    scripted_fault_session: Callable[[list[Any]], Any],
    queued_fault_response: Callable[..., Any],
    ledger: InMemoryEventLedgerWriter,
    fake_int_clock: Callable[[], int],
    recording_sleeper: Callable[[int], None],
    seeded_rng: random.Random,
    clock: Callable[[], datetime],
) -> None:
    """A `429` rate-limit response is retried end-to-end, then a `200` succeeds."""
    orderbook_payload = _read_fixture("orderbook_KXFED-24DEC.json")
    session = scripted_fault_session(
        [
            queued_fault_response(429, {"error": "slow down"}),
            queued_fault_response(200, orderbook_payload),
        ]
    )
    resilience = ResilientCaller(
        _resilience_policy(max_attempts=3),
        ledger,
        clock=fake_int_clock,
        sleeper=recording_sleeper,
        rng=seeded_rng,
        wall_clock=clock,
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=resilience,
    )

    response = client.get("markets", "KXFED-24DEC", "orderbook")

    assert response.payload == orderbook_payload
    assert len(session.calls) == 2  # 429 retried, then 200


def test_get_recovers_after_a_malformed_body_then_a_200(
    scripted_fault_session: Callable[[list[Any]], Any],
    queued_fault_response: Callable[..., Any],
    ledger: InMemoryEventLedgerWriter,
    fake_int_clock: Callable[[], int],
    recording_sleeper: Callable[[int], None],
    seeded_rng: random.Random,
    clock: Callable[[], datetime],
) -> None:
    """A malformed/truncated body (`.json()` raises) is retried end-to-end.

    Drives the transport's real `response.json()` call through `KalshiClient`
    via `QueuedFaultResponse.json_raises`: the parse failure surfaces as a
    retryable transport error, so the next `200` recovers transparently.
    """
    orderbook_payload = _read_fixture("orderbook_KXFED-24DEC.json")
    session = scripted_fault_session(
        [
            queued_fault_response(
                200, json_raises=ValueError("Expecting value: line 1 column 1 (char 0)")
            ),
            queued_fault_response(200, orderbook_payload),
        ]
    )
    resilience = ResilientCaller(
        _resilience_policy(max_attempts=3),
        ledger,
        clock=fake_int_clock,
        sleeper=recording_sleeper,
        rng=seeded_rng,
        wall_clock=clock,
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=resilience,
    )

    response = client.get("markets", "KXFED-24DEC", "orderbook")

    assert response.payload == orderbook_payload
    assert len(session.calls) == 2  # malformed body retried, then 200


def test_persistent_malformed_body_exhausts_retries_and_surfaces(
    scripted_fault_session: Callable[[list[Any]], Any],
    queued_fault_response: Callable[..., Any],
    ledger: InMemoryEventLedgerWriter,
    fake_int_clock: Callable[[], int],
    recording_sleeper: Callable[[int], None],
    seeded_rng: random.Random,
    clock: Callable[[], datetime],
) -> None:
    """A body that never parses fails closed: the parse error surfaces, no data."""
    session = scripted_fault_session(
        [
            queued_fault_response(200, json_raises=ValueError("truncated body"))
            for _ in range(3)
        ]
    )
    resilience = ResilientCaller(
        _resilience_policy(max_attempts=3),
        ledger,
        clock=fake_int_clock,
        sleeper=recording_sleeper,
        rng=seeded_rng,
        wall_clock=clock,
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=resilience,
    )

    with pytest.raises(ValueError, match="truncated body"):
        client.get("markets", "KXFED-24DEC", "orderbook")

    assert len(session.calls) == 3  # every attempt made, then surfaced


# =============================================================================
# Resilience is on by default: a plain KalshiClient is protected (issue #20)
# =============================================================================


def test_default_client_trips_the_breaker_under_a_5xx_storm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A plain client (no injected caller) gets live breaker protection.

    Constructs `KalshiClient` with *no* explicit `ResilientCaller` -- only a
    tuned `resilience_policy` -- so the on-by-default caller is what protects
    it. A persistent 5xx storm exhausts retries on each call, trips the breaker
    on the second, and refuses the third live with `ConnectorHaltError`. The
    zero-second backoff/jitter keeps the real `time.sleep` instantaneous.
    """
    caplog.set_level(logging.INFO)
    session = _SingleRouteSession(500, {"error": "boom"})
    policy = _resilience_policy(
        max_attempts=2,
        failure_threshold=2,
        cooldown_seconds=3_600,
        base_backoff_seconds=0,
        max_backoff_seconds=0,
        max_jitter_seconds=0,
        bucket_capacity=1_000,
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience_policy=policy,
    )

    with pytest.raises(KalshiApiError):
        client.get("exchange", "status")  # call #1: 1st breaker failure
    with pytest.raises(KalshiApiError):
        client.get("exchange", "status")  # call #2: trips the breaker OPEN
    with pytest.raises(ConnectorHaltError):
        client.get("exchange", "status")  # call #3: refused live, no transport

    assert any(CONNECTOR_HALT_EVENT in record.getMessage() for record in caplog.records)


def test_default_client_token_bucket_gates_real_client_requests(
    scripted_fault_session: Callable[[list[Any]], Any],
    queued_fault_response: Callable[..., Any],
    ledger: InMemoryEventLedgerWriter,
    fake_int_clock: Callable[[], int],
    recording_sleeper: Callable[[int], None],
    seeded_rng: random.Random,
    clock: Callable[[], datetime],
) -> None:
    """The default-built caller's token bucket gates real `KalshiClient.get`s.

    Uses the exact `build_default_resilient_caller` seam the client wires by
    default, with injected timing so it is deterministic: a single-token bucket
    lets the first request through and forces the second to wait one full refill
    interval -- proving rate limiting governs real client traffic, not just a
    standalone `TokenBucket`.
    """
    orderbook_payload = _read_fixture("orderbook_KXFED-24DEC.json")
    session = scripted_fault_session(
        [
            queued_fault_response(200, orderbook_payload),
            queued_fault_response(200, orderbook_payload),
        ]
    )
    caller = build_default_resilient_caller(
        ledger,
        policy=_resilience_policy(
            bucket_capacity=1, refill_interval_seconds=5, max_attempts=1
        ),
        clock=fake_int_clock,
        sleeper=recording_sleeper,
        rng=seeded_rng,
        wall_clock=clock,
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=caller,
    )

    client.get("markets", "KXFED-24DEC", "orderbook")  # consumes the sole token
    client.get("markets", "KXFED-24DEC", "orderbook")  # bucket empty: waits a refill

    assert recording_sleeper.calls == [5]


# =============================================================================
# SchemaValidator wired through KalshiClient.get(): drift halts outside retry
# =============================================================================


def test_money_field_drift_halts_and_bypasses_the_retry_loop_entirely(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """An unexpected money/risk field halts on the *first* call: never retried.

    Wires a `ResilientCaller` that would happily retry a transport failure,
    proving the `SchemaAnomalyHaltError` short-circuits before ever reaching it:
    exactly one underlying HTTP call is made, and the breaker is untouched.
    """
    drift_payload = _read_fixture("faults/orderbook_drift_money_fee.json")
    session = _SingleRouteSession(200, drift_payload)
    resilience = ResilientCaller(
        _resilience_policy(max_attempts=5, failure_threshold=1),
        ledger,
        clock=lambda: 0,
        sleeper=lambda seconds: None,
        rng=random.Random(1),
        wall_clock=clock,
    )
    validator = SchemaValidator(
        kalshi_default_schema_registry(), ledger, wall_clock=clock
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL,
        allowlist=_FAKE_ALLOWLIST,
        timeout=5,
        session=session,
        resilience=resilience,
        validator=validator,
    )

    with pytest.raises(SchemaAnomalyHaltError):
        client.get("markets", "KXFED-24DEC", "orderbook")

    assert len(session.calls) == 1  # never retried
    assert ledger.events_by_type(CONNECTOR_HALT_EVENT) == ()  # breaker never touched
    (event,) = ledger.events_by_type(SCHEMA_ANOMALY_EVENT)
    assert "fee" in event.payload["fields"]

    # The breaker is provably untouched: a second call still raises the same
    # SchemaAnomalyHaltError, not a ConnectorHaltError from a (nonexistent) trip.
    with pytest.raises(SchemaAnomalyHaltError):
        client.get("markets", "KXFED-24DEC", "orderbook")


def test_cosmetic_field_drift_only_warns_and_the_call_still_succeeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cosmetic-allowlisted extra field warns but the response still returns."""
    caplog.set_level(logging.WARNING)
    drift_payload = _read_fixture("faults/orderbook_drift_cosmetic.json")
    session = _SingleRouteSession(200, drift_payload)
    client = KalshiClient(
        base_url=_FAKE_BASE_URL, allowlist=_FAKE_ALLOWLIST, timeout=5, session=session
    )

    response = client.get("markets", "KXFED-24DEC", "orderbook")

    assert response.payload == drift_payload
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_default_validator_validates_the_standard_fixtures_clean(
    fake_kalshi_client: KalshiClient,
) -> None:
    """The client's on-by-default validator never rejects the recorded fixtures.

    Regression guard: wiring an on-by-default `SchemaValidator` into
    `KalshiClient` must not break a plain, un-overridden client fetching the
    real recorded fixtures (`markets.json` / `events.json` /
    `orderbook_KXFED-24DEC.json` / `exchange_status.json` / `series_KXFED.json`).
    """
    assert fake_kalshi_client.get("markets").payload is not None
    assert fake_kalshi_client.get("events").payload is not None
    assert fake_kalshi_client.get("exchange", "status").payload is not None
    assert fake_kalshi_client.get("series", "KXFED").payload is not None
    assert (
        fake_kalshi_client.get("markets", "KXFED-24DEC", "orderbook").payload
        is not None
    )


# =============================================================================
# KalshiConnector._ensure_operational(): maintenance suspension
# =============================================================================


@pytest.mark.parametrize(
    ("exchange_active", "trading_active"),
    [(True, False), (False, False)],  # paused, closed
)
def test_get_order_book_raises_maintenance_halt_when_not_open(
    ledger: InMemoryEventLedgerWriter,
    clock: Callable[[], datetime],
    exchange_active: bool,
    trading_active: bool,
) -> None:
    """`get_order_book` refuses to proceed while the exchange isn't `"open"`."""
    session = _MaintenanceSession(
        exchange_active=exchange_active, trading_active=trading_active
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL, allowlist=_FAKE_ALLOWLIST, timeout=5, session=session
    )
    connector = KalshiConnector(client, ledger, clock=clock)

    with pytest.raises(MaintenanceHaltError):
        connector.get_order_book("KXFED-24DEC")

    (event,) = ledger.events_by_type(CONNECTOR_HALT_EVENT)
    assert event.payload["reason"] == "maintenance"
    assert not any(url.endswith("/orderbook") for url in session.calls)


@pytest.mark.parametrize(
    ("exchange_active", "trading_active"),
    [(True, False), (False, False)],  # paused, closed
)
def test_list_markets_raises_maintenance_halt_when_not_open(
    ledger: InMemoryEventLedgerWriter,
    clock: Callable[[], datetime],
    exchange_active: bool,
    trading_active: bool,
) -> None:
    """`list_markets` refuses to proceed while the exchange isn't `"open"`."""
    session = _MaintenanceSession(
        exchange_active=exchange_active, trading_active=trading_active
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL, allowlist=_FAKE_ALLOWLIST, timeout=5, session=session
    )
    connector = KalshiConnector(client, ledger, clock=clock)

    with pytest.raises(MaintenanceHaltError):
        connector.list_markets()

    (event,) = ledger.events_by_type(CONNECTOR_HALT_EVENT)
    assert event.payload["reason"] == "maintenance"
    assert not any(url.endswith("/markets") for url in session.calls)


@pytest.mark.parametrize(
    ("exchange_active", "trading_active"),
    [(True, False), (False, False)],  # paused, closed
)
def test_get_market_raises_maintenance_halt_when_not_open(
    ledger: InMemoryEventLedgerWriter,
    clock: Callable[[], datetime],
    exchange_active: bool,
    trading_active: bool,
) -> None:
    """`get_market` (a single-ticker snapshot) also halts while not `"open"`.

    Closes the fail-closed gap where single-ticker lookups could fetch live
    data during a maintenance window while `list_markets`/`get_order_book`
    correctly halted.
    """
    session = _MaintenanceSession(
        exchange_active=exchange_active, trading_active=trading_active
    )
    client = KalshiClient(
        base_url=_FAKE_BASE_URL, allowlist=_FAKE_ALLOWLIST, timeout=5, session=session
    )
    connector = KalshiConnector(client, ledger, clock=clock)

    with pytest.raises(MaintenanceHaltError):
        connector.get_market("KXFED-24DEC")

    (event,) = ledger.events_by_type(CONNECTOR_HALT_EVENT)
    assert event.payload["reason"] == "maintenance"
    assert not any(url.endswith("/markets") for url in session.calls)


def test_get_market_proceeds_normally_when_exchange_is_open(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """An `"open"` exchange status lets `get_market` fetch the real catalog."""
    session = _MaintenanceSession(exchange_active=True, trading_active=True)
    client = KalshiClient(
        base_url=_FAKE_BASE_URL, allowlist=_FAKE_ALLOWLIST, timeout=5, session=session
    )
    connector = KalshiConnector(client, ledger, clock=clock)

    # The `/markets` route now serves the real recorded catalog, so the known
    # binary ticker is found and normalized -- never a MaintenanceHaltError,
    # and the venue was consulted (the market fetch was reached).
    market = connector.get_market("KXFED-24DEC")

    assert market.ticker == "KXFED-24DEC"
    assert market.market_type == "fully_collateralized_binary"
    assert ledger.events_by_type(CONNECTOR_HALT_EVENT) == ()
    assert any(url.endswith("/markets") for url in session.calls)


def test_get_order_book_proceeds_normally_when_exchange_is_open(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """An `"open"` exchange status never triggers `MaintenanceHaltError`."""
    session = _MaintenanceSession(exchange_active=True, trading_active=True)
    client = KalshiClient(
        base_url=_FAKE_BASE_URL, allowlist=_FAKE_ALLOWLIST, timeout=5, session=session
    )
    connector = KalshiConnector(client, ledger, clock=clock)

    book = connector.get_order_book("KXFED-24DEC")

    assert book.ticker == "KXFED-24DEC"
    assert ledger.events_by_type(CONNECTOR_HALT_EVENT) == ()


def test_list_markets_proceeds_normally_when_exchange_is_open(
    ledger: InMemoryEventLedgerWriter, clock: Callable[[], datetime]
) -> None:
    """An `"open"` exchange status lets `list_markets` fetch the real catalog."""
    session = _MaintenanceSession(exchange_active=True, trading_active=True)
    client = KalshiClient(
        base_url=_FAKE_BASE_URL, allowlist=_FAKE_ALLOWLIST, timeout=5, session=session
    )
    connector = KalshiConnector(client, ledger, clock=clock)

    markets = connector.list_markets()

    assert markets
    assert any(market.ticker == "KXFED-24DEC" for market in markets)
    assert ledger.events_by_type(CONNECTOR_HALT_EVENT) == ()
