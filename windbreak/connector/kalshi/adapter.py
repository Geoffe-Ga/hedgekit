"""The :class:`KalshiConnector` adapter over Kalshi's public read-only surface.

:class:`KalshiConnector` implements the SPEC S7.2
:class:`~windbreak.connector.interface.MarketConnector` protocol against a
:class:`~windbreak.connector.kalshi.client.KalshiClient`. It exposes only the
public, read-only market-data methods this issue delivers -- ``list_markets``,
``get_market``, ``get_order_book``, ``get_exchange_status``, and
``get_exchange_time`` -- normalizing each raw payload through
:mod:`windbreak.connector.kalshi.normalize`. ``list_markets`` applies the binary
allowlist and ledgers one ``PRODUCT_REFUSED`` event per refused product through
an injected :class:`~windbreak.connector.snapshot.EventLedgerWriter`, isolating a
broken writer so one bad record never aborts a run.

The fee-model and balance-semantics methods are implemented here (issue #18):
:meth:`KalshiConnector.get_fee_model` resolves a series' fee schedule and
:meth:`KalshiConnector.get_balance_semantics` returns the recorded
:data:`KALSHI_BALANCE_SEMANTICS` record. The remaining trading and account
methods (order path is milestone M4; balances, positions, open orders, fills
are issue #3) still raise :class:`NotImplementedError` until those issues wire
them. This module sits on the money path guarded by
``scripts/lint_no_floats.py``: no ``/`` or ``float`` appears anywhere.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

from windbreak.connector.fees import FeeModel, UnknownFeeModelError
from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.kalshi.client import KalshiApiError
from windbreak.connector.kalshi.normalize import (
    MARKET_MALFORMED_EVENT,
    PRODUCT_REFUSED_EVENT,
    gate_product,
    normalize_exchange_status,
    normalize_market,
    normalize_order_book,
    payload_hash,
)
from windbreak.connector.resilience import CONNECTOR_HALT_EVENT, MaintenanceHaltError
from windbreak.connector.semantics import (
    BalanceSemantics,
    CancelCollateralRelease,
    FeeDebitTiming,
    FeeRounding,
    HaltedMarketBehavior,
    OrderCollateralInAvailable,
    OrderCollateralInTotal,
    PartialFillRepresentation,
    UnsettledProceeds,
)
from windbreak.connector.snapshot import ConnectorEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from windbreak.connector.kalshi.client import KalshiClient
    from windbreak.connector.models import (
        BalanceSnapshot,
        ExchangeStatus,
        Fill,
        NormalizedMarket,
        OpenOrder,
        OrderBookSnapshot,
        Position,
    )
    from windbreak.connector.snapshot import EventLedgerWriter

_LOGGER = logging.getLogger("windbreak.connector.kalshi")

#: Hard upper bound on pages a single cursor-paginated fetch will follow. A
#: healthy Kalshi catalog is far below this; the cap exists solely so a venue
#: that returns a never-emptying cursor (a bug or a runaway loop) fails loudly
#: via :class:`KalshiPaginationError` instead of spinning forever.
_MAX_PAGES: Final = 1000

#: Malformed-market normalization failures are total functions of the payload,
#: so these are the only exception types :meth:`KalshiConnector.list_markets`
#: degrades to a ledgered ``MARKET_MALFORMED`` event: a missing required field
#: (``KeyError``), an unparseable value (``ValueError``, e.g. a bad timestamp),
#: or a wrong-typed money leaf (``TypeError`` from a scaled-integer unit).
_MALFORMED_MARKET_ERRORS: Final = (KeyError, ValueError, TypeError)

#: Message naming the milestone that wires the order path (place/cancel).
_ORDER_PATH_DEFERRAL: Final = "the order path is deferred to milestone M4"

#: Message naming the issue that wires the remaining account access.
_ACCOUNT_DEFERRAL: Final = (
    "account access (balances, positions, open orders, fills) is deferred to issue #3"
)

#: One basis point is 100 ppm (1e-4 vs 1e-6), so a bps rate scales to ppm by
#: multiplying by 100 (e.g. 700 bps -> 70_000 ppm) -- integer math only.
_BPS_TO_PPM: Final = 100

#: The only fee-formula family whose schedule the SPEC fee-bound math models; a
#: series advertising any other ``fee_type`` fails closed as unknown.
_QUADRATIC_FEE_TYPE: Final = "quadratic"

#: A market ticker's series ticker is the segment before the first ``-`` (e.g.
#: ``"KXFED-24DEC"`` -> ``"KXFED"``); a bare series ticker has no separator.
_SERIES_TICKER_SEPARATOR: Final = "-"

#: The exchange status that permits market-data fetches to proceed; any other
#: status (paused/closed) suspends fetches via :class:`MaintenanceHaltError`.
_OPEN_STATUS: Final = "open"

#: The ``reason`` stamped on the ``CONNECTOR_HALT`` event when the exchange is
#: not open, distinguishing a maintenance suspension from a breaker trip.
_MAINTENANCE_REASON: Final = "maintenance"

#: Kalshi's recorded balance-interpretation semantics: five fields pinned to
#: documented behavior, three left ``UNKNOWN`` for lack of public evidence. See
#: ``tests/fixtures/exchange/kalshi/README.md`` for the field-by-field evidence.
KALSHI_BALANCE_SEMANTICS: Final[BalanceSemantics] = BalanceSemantics(
    open_order_collateral_in_total=OrderCollateralInTotal.UNKNOWN,
    open_order_collateral_in_available=(
        OrderCollateralInAvailable.DEDUCTED_FROM_AVAILABLE
    ),
    fee_debit_timing=FeeDebitTiming.AT_EXECUTION,
    fee_rounding=FeeRounding.UP_TO_NEXT_CENT,
    partial_fill_representation=PartialFillRepresentation.PER_FILL_RECORDS,
    cancel_collateral_release=CancelCollateralRelease.UNKNOWN,
    unsettled_proceeds=UnsettledProceeds.EXCLUDED_UNTIL_CREDITED,
    halted_market_behavior=HaltedMarketBehavior.UNKNOWN,
)


def _bps_to_ppm(value: object, field_name: str) -> int:
    """Convert a basis-point leaf to ppm, failing closed on a non-int leaf.

    A JSON fee leaf must be a true integer count of basis points; a ``bool``,
    a ``float``, or any other type means the schedule is not the shape this
    adapter understands, so it is refused rather than misread.

    Args:
        value: The raw ``*_fee_bps`` leaf from the series document.
        field_name: The leaf's key, surfaced in the failure message.

    Returns:
        The rate in ppm (``value * 100``).

    Raises:
        UnknownFeeModelError: If ``value`` is a ``bool`` or not an ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise UnknownFeeModelError(
            f"series fee leaf {field_name!r} must be an int, got {type(value).__name__}"
        )
    return value * _BPS_TO_PPM


def _fee_model_from_series(payload: Mapping[str, Any]) -> FeeModel:
    """Parse a Kalshi ``/series/{ticker}`` document into a :class:`FeeModel`.

    Args:
        payload: The parsed series-document JSON body.

    Returns:
        The normalized fee model.

    Raises:
        UnknownFeeModelError: If the document is missing the ``series`` block or
            a fee field, advertises a non-quadratic ``fee_type``, or carries a
            non-integer or negative fee leaf.
    """
    try:
        series = payload["series"]
        fee_type = series["fee_type"]
        schedule_id = series["fee_schedule_id"]
        maker_bps = series["maker_fee_bps"]
        taker_bps = series["taker_fee_bps"]
        settlement_bps = series["settlement_fee_bps"]
    except (KeyError, TypeError) as exc:
        raise UnknownFeeModelError(
            f"series document is missing a required fee field: {exc}"
        ) from exc
    if fee_type != _QUADRATIC_FEE_TYPE:
        raise UnknownFeeModelError(
            f"unsupported fee_type {fee_type!r}; only {_QUADRATIC_FEE_TYPE!r} "
            "is modeled"
        )
    if not isinstance(schedule_id, str) or not schedule_id:
        raise UnknownFeeModelError(
            f"series fee_schedule_id must be a non-empty str, got {schedule_id!r}"
        )
    try:
        return FeeModel(
            schedule_id=schedule_id,
            maker_fee_ppm=_bps_to_ppm(maker_bps, "maker_fee_bps"),
            taker_fee_ppm=_bps_to_ppm(taker_bps, "taker_fee_bps"),
            settlement_fee_ppm=_bps_to_ppm(settlement_bps, "settlement_fee_bps"),
        )
    except (TypeError, ValueError) as exc:
        raise UnknownFeeModelError(
            f"series fee schedule is not a shape this adapter models: {exc}"
        ) from exc


def _utc_now() -> datetime:
    """Return the current time as a UTC datetime (the default connector clock)."""
    return datetime.now(UTC)


def _iso_timestamp(moment: datetime) -> str:
    """Render a datetime as ISO-8601 UTC with a trailing ``Z``.

    Args:
        moment: The (timezone-aware) datetime to render; normalized to UTC.

    Returns:
        A string like ``2025-06-01T12:00:00.000000Z``.
    """
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class KalshiPaginationError(RuntimeError):
    """Raised when a cursor-paginated fetch exceeds :data:`_MAX_PAGES`.

    Kalshi's ``/markets`` and ``/events`` endpoints paginate with an opaque
    ``cursor``; the connector follows it page by page until the cursor empties.
    If it never empties within the safety cap -- a venue bug or a cursor that
    loops on itself -- the connector refuses to spin forever and raises this,
    surfacing the runaway loudly rather than hanging a caller.
    """

    def __init__(self, endpoint: str, max_pages: int) -> None:
        """Initialize with the endpoint and the cap that was exceeded.

        Args:
            endpoint: The path segment that would not stop paginating.
            max_pages: The page cap that was hit.
        """
        self.endpoint = endpoint
        self.max_pages = max_pages
        super().__init__(
            f"{endpoint!r} pagination exceeded the {max_pages}-page safety cap; "
            "refusing to follow an unbounded cursor"
        )


class KalshiConnector:
    """A read-only :class:`MarketConnector` backed by :class:`KalshiClient`."""

    def __init__(
        self,
        client: KalshiClient,
        ledger_writer: EventLedgerWriter,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Initialize the connector.

        Args:
            client: The HTTPS JSON client the connector fetches through.
            ledger_writer: The seam that records ``PRODUCT_REFUSED`` events.
            clock: Returns "now"; injected so snapshots are deterministic in
                tests. Defaults to wall-clock UTC.
        """
        self._client = client
        self._ledger_writer = ledger_writer
        self._clock = clock

    def _ensure_operational(self) -> None:
        """Fail closed while the exchange is not open for trading.

        Fetches the current exchange status; when it is anything other than
        open, ledgers one ``CONNECTOR_HALT`` event (reason maintenance) and
        refuses to proceed, so no market-data transport runs against a paused
        or closed venue.

        Raises:
            MaintenanceHaltError: If the exchange status is not open.
        """
        status = self.get_exchange_status()
        if status.status == _OPEN_STATUS:
            return
        event = ConnectorEvent(
            event_type=CONNECTOR_HALT_EVENT,
            payload={"reason": _MAINTENANCE_REASON, "status": status.status},
            ts=_iso_timestamp(self._clock()),
        )
        self._record(event)
        raise MaintenanceHaltError(
            f"exchange is not open for trading (status={status.status!r})"
        )

    def _paginate(self, endpoint: str, item_key: str) -> list[Mapping[str, Any]]:
        """Follow a Kalshi ``cursor`` across every page of one list endpoint.

        Fetches ``endpoint`` repeatedly, passing each response's non-empty
        ``cursor`` as the ``cursor`` query parameter of the next request, and
        concatenates the ``item_key`` list from every page. The walk is bounded
        by :data:`_MAX_PAGES`: a cursor that never empties raises
        :class:`KalshiPaginationError` rather than looping forever.

        Args:
            endpoint: The single path segment to fetch (``"markets"`` /
                ``"events"``).
            item_key: The response key whose list is aggregated across pages.

        Returns:
            Every item from every page, in page-then-position order.

        Raises:
            KalshiPaginationError: If the cursor is still non-empty after
                :data:`_MAX_PAGES` pages.
        """
        items: list[Mapping[str, Any]] = []
        cursor = ""
        for _ in range(_MAX_PAGES):
            params = {"cursor": cursor} if cursor else None
            payload = cast(
                "Mapping[str, Any]", self._client.get(endpoint, params=params).payload
            )
            items.extend(payload.get(item_key, []))
            cursor = payload.get("cursor") or ""
            if not cursor:
                return items
        raise KalshiPaginationError(endpoint, _MAX_PAGES)

    def _raw_markets(self) -> list[Mapping[str, Any]]:
        """Fetch every page of ``/markets`` and return the raw market payloads."""
        return self._paginate("markets", "markets")

    def _event_index(self) -> dict[str, Mapping[str, Any]]:
        """Fetch every page of ``/events``, indexing raw events by event ticker."""
        return {
            event["event_ticker"]: event for event in self._paginate("events", "events")
        }

    def list_markets(self) -> tuple[NormalizedMarket, ...]:
        """Return every allowed binary market, ledgering each refusal.

        Non-binary products are excluded and recorded as a single
        ``PRODUCT_REFUSED`` event each. An allowed binary that fails to
        normalize (a missing field or wrong-typed leaf) is likewise excluded
        and recorded as a ``MARKET_MALFORMED`` event, so one bad payload never
        aborts the scan yet is never silently dropped. Every page of both
        ``/markets`` and ``/events`` is followed via their ``cursor``. Only the
        normalized binaries are returned.

        Returns:
            The normalized binary markets the venue currently offers.

        Raises:
            MaintenanceHaltError: If the exchange is not open for trading.
        """
        self._ensure_operational()
        events = self._event_index()
        normalized: list[NormalizedMarket] = []
        for raw in self._raw_markets():
            reason = gate_product(raw)
            if reason is not None:
                self._record_refusal(raw, reason)
                continue
            try:
                market = normalize_market(raw, events.get(raw["event_ticker"]))
            except _MALFORMED_MARKET_ERRORS as exc:
                self._record_malformed(raw, exc)
                continue
            normalized.append(market)
        return tuple(normalized)

    def _record_refusal(self, raw: Mapping[str, Any], reason: str) -> None:
        """Ledger one ``PRODUCT_REFUSED`` event for a refused market.

        Args:
            raw: The refused raw market payload.
            reason: Why the product was refused.
        """
        event = ConnectorEvent(
            event_type=PRODUCT_REFUSED_EVENT,
            payload={
                "ticker": raw.get("ticker"),
                "event_ticker": raw.get("event_ticker"),
                "reason": reason,
                "raw_exchange_payload_hash": payload_hash(raw),
            },
            ts=_iso_timestamp(self._clock()),
        )
        self._record(event)

    def _record_malformed(self, raw: Mapping[str, Any], exc: Exception) -> None:
        """Ledger one ``MARKET_MALFORMED`` event for an unnormalizable binary.

        Args:
            raw: The binary market payload that failed to normalize.
            exc: The normalization failure, named in the ledgered reason.
        """
        event = ConnectorEvent(
            event_type=MARKET_MALFORMED_EVENT,
            payload={
                "ticker": raw.get("ticker"),
                "event_ticker": raw.get("event_ticker"),
                "reason": f"{type(exc).__name__}: {exc}",
                "raw_exchange_payload_hash": payload_hash(raw),
            },
            ts=_iso_timestamp(self._clock()),
        )
        self._record(event)

    def _record(self, event: ConnectorEvent) -> None:
        """Record an event via the writer, isolating any failure.

        Args:
            event: The event to record. A raising writer is logged and
                swallowed so one bad record never aborts a run.
        """
        try:
            self._ledger_writer.record(event)
        except Exception as exc:
            _LOGGER.warning(
                "event ledger writer failed to record %s event: %s",
                event.event_type,
                exc,
                extra={"component": "connector.kalshi"},
            )

    def _gated_raw_market(self, ticker: str) -> Mapping[str, Any]:
        """Return the raw market payload for ``ticker`` iff it is an allowed binary.

        Scans :meth:`_raw_markets` for the matching ticker and applies the
        product allowlist, mirroring the gate :meth:`get_market` enforces so
        every single-ticker read path refuses a non-binary (margin/perp/
        derivative) surface identically (SPEC §7.1).

        Args:
            ticker: The market ticker to locate and gate.

        Returns:
            The raw market payload when ``ticker`` is an offered, allowed binary.

        Raises:
            UnknownMarketError: If the ticker is absent from the catalog or its
                product type is refused by the allowlist. Deliberately does not
                ledger a ``PRODUCT_REFUSED`` event: a targeted lookup must not
                flood the ledger (contrast :meth:`list_markets`), matching
                :meth:`get_market`.
        """
        for raw in self._raw_markets():
            if raw.get("ticker") != ticker:
                continue
            if gate_product(raw) is not None:
                raise UnknownMarketError(ticker)
            return raw
        raise UnknownMarketError(ticker)

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Return the normalized binary market for ``ticker``.

        A single-ticker snapshot fetch, so it suspends on a maintenance window
        exactly like :meth:`list_markets` / :meth:`get_order_book`: it consults
        exchange status first and fails closed if the venue is not open, rather
        than returning live market data from a paused/closed exchange (SPEC §3
        principle 3). The product allowlist is applied by
        :meth:`_gated_raw_market` before any normalization.

        Args:
            ticker: The market ticker to look up.

        Returns:
            The normalized market.

        Raises:
            MaintenanceHaltError: If the exchange is not open for trading.
            UnknownMarketError: If the ticker is refused, not offered, or its
                payload is malformed (fail closed: the malformed binary is
                ledgered as a ``MARKET_MALFORMED`` event before raising).
        """
        self._ensure_operational()
        raw = self._gated_raw_market(ticker)
        events = self._event_index()
        try:
            return normalize_market(raw, events.get(raw["event_ticker"]))
        except _MALFORMED_MARKET_ERRORS as exc:
            self._record_malformed(raw, exc)
            raise UnknownMarketError(ticker) from exc

    def get_order_book(self, ticker: str) -> OrderBookSnapshot:
        """Return the current YES order book for ``ticker``.

        Applies the product allowlist via :meth:`_gated_raw_market` before
        contacting the venue, so a non-binary product's book is never fetched
        (fail closed, SPEC §7.1). The market payload is read purely to gate:
        the ``/orderbook`` route carries no ``market_type`` of its own.

        Args:
            ticker: The market ticker to look up.

        Returns:
            The order-book snapshot, stamped with the injected clock.

        Raises:
            MaintenanceHaltError: If the exchange is not open for trading.
            UnknownMarketError: If the ticker is refused (a non-binary product),
                not offered, or the venue has no book for it.
        """
        self._ensure_operational()
        self._gated_raw_market(ticker)  # refuse non-binary/absent before fetching
        try:
            response = self._client.get("markets", ticker, "orderbook")
        except KalshiApiError as exc:
            raise UnknownMarketError(ticker) from exc
        raw = cast("Mapping[str, object]", response.payload)
        return normalize_order_book(ticker, raw, self._clock())

    def get_exchange_status(self) -> ExchangeStatus:
        """Return the exchange's current trading status."""
        response = self._client.get("exchange", "status")
        raw = cast("Mapping[str, object]", response.payload)
        return normalize_exchange_status(raw, self._clock())

    def get_exchange_time(self) -> datetime:
        """Return the exchange server time, falling back to the clock.

        Returns:
            The response ``Date`` header when present; otherwise the injected
            clock's current time.
        """
        response = self._client.get("exchange", "status")
        if response.server_date is not None:
            return response.server_date
        return self._clock()

    def get_balance_semantics(self) -> BalanceSemantics:
        """Return Kalshi's recorded balance-interpretation semantics.

        Returns:
            The shared :data:`KALSHI_BALANCE_SEMANTICS` record (five documented
            fields, three ``UNKNOWN``).
        """
        return KALSHI_BALANCE_SEMANTICS

    def get_balances(self) -> BalanceSnapshot:
        """Raise; account access is deferred (issue #3).

        Raises:
            NotImplementedError: Always; see :data:`_ACCOUNT_DEFERRAL`.
        """
        raise NotImplementedError(_ACCOUNT_DEFERRAL)

    def get_positions(self) -> tuple[Position, ...]:
        """Raise; account access is deferred (issue #3).

        Raises:
            NotImplementedError: Always; see :data:`_ACCOUNT_DEFERRAL`.
        """
        raise NotImplementedError(_ACCOUNT_DEFERRAL)

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Raise; account access is deferred (issue #3).

        Raises:
            NotImplementedError: Always; see :data:`_ACCOUNT_DEFERRAL`.
        """
        raise NotImplementedError(_ACCOUNT_DEFERRAL)

    def get_fills(self, since: datetime) -> tuple[Fill, ...]:
        """Raise; account access is deferred (issue #3).

        Args:
            since: Unused exclusive lower bound on fill timestamps.

        Raises:
            NotImplementedError: Always; see :data:`_ACCOUNT_DEFERRAL`.
        """
        raise NotImplementedError(_ACCOUNT_DEFERRAL)

    def get_fee_model(self, market_or_series: str) -> FeeModel:
        """Return the fee schedule for a market or its series, failing closed.

        Resolves the series ticker (the segment before the first ``-``), fetches
        its ``/series/{ticker}`` document, and normalizes the schedule. Both a
        market ticker (``"KXFED-24DEC"``) and its bare series (``"KXFED"``)
        resolve to the same fee model.

        Deliberately *not* gated by the maintenance-window check: a fee schedule
        is static reference data, not a live market-data snapshot, so it stays
        readable while the exchange is paused/closed (it never returns tradeable
        prices). Only the snapshot-fetching methods (``list_markets`` /
        ``get_market`` / ``get_order_book``) suspend during maintenance.

        Args:
            market_or_series: A market ticker or a bare series ticker.

        Returns:
            The normalized fee model for the resolved series.

        Raises:
            UnknownFeeModelError: If the series is unrecognized (a 404) or its
                document is malformed / advertises an unsupported ``fee_type``.
        """
        series_ticker = market_or_series.split(_SERIES_TICKER_SEPARATOR, 1)[0]
        try:
            response = self._client.get("series", series_ticker)
        except KalshiApiError as exc:
            raise UnknownFeeModelError(
                f"no fee schedule for series {series_ticker!r}"
            ) from exc
        raw = cast("Mapping[str, Any]", response.payload)
        return _fee_model_from_series(raw)

    def place_order(self, normalized_intent: object, approval_token: object) -> object:
        """Raise; the order path is deferred (milestone M4).

        Args:
            normalized_intent: Unused normalized order intent.
            approval_token: Unused risk-kernel approval token.

        Raises:
            NotImplementedError: Always; see :data:`_ORDER_PATH_DEFERRAL`.
        """
        raise NotImplementedError(_ORDER_PATH_DEFERRAL)

    def cancel_order(self, order_id: str) -> None:
        """Raise; the order path is deferred (milestone M4).

        Args:
            order_id: Unused venue order identifier.

        Raises:
            NotImplementedError: Always; see :data:`_ORDER_PATH_DEFERRAL`.
        """
        raise NotImplementedError(_ORDER_PATH_DEFERRAL)
