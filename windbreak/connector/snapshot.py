"""The market-snapshot task and its event-ledger persistence seam.

:class:`MarketSnapshotTask` walks every market a connector offers and records,
per market, a ``MARKET_SNAPSHOT`` event (the market projected via
:func:`~windbreak.connector.models.market_to_payload`), then hands the market
and its order book to the injected screener via ``screen_book``. The screener --
not this task -- is the single ``SCREEN_DECISION`` emitter, writing to the same
shared ledger writer. Events flow through the :class:`EventLedgerWriter`
protocol -- a dependency-injection seam with no ``windbreak.ledger`` dependency,
mirroring :class:`windbreak.alerts.dispatch.LedgerWriter`. A broken writer is
isolated: its failure (whether raised by a snapshot write or by the screener's
own decision write) is logged and swallowed so one bad record can never abort a
run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol

from windbreak.connector.models import market_to_payload
from windbreak.timekeeping import iso_z

if TYPE_CHECKING:
    from collections.abc import Mapping

    from windbreak.connector.interface import MarketConnector
    from windbreak.connector.models import NormalizedMarket, OrderBookSnapshot
    from windbreak.screener import ScreenResult

#: Event type recorded for each market's normalized snapshot.
MARKET_SNAPSHOT_EVENT: Final = "MARKET_SNAPSHOT"

#: Event type recorded for each market's screening verdict.
SCREEN_DECISION_EVENT: Final = "SCREEN_DECISION"

_LOGGER = logging.getLogger("windbreak.connector")


def utc_now_iso(moment: datetime | None = None) -> str:
    """Render a moment as ISO-8601 UTC with a trailing ``Z``.

    Every connector ledger ``ts`` flows through here, including the values
    produced by the injected ``clock`` / ``wall_clock`` callables that
    :class:`windbreak.screener.screener.Screener` and
    ``windbreak.connector.validation.SchemaValidator`` take as *required,
    undefaulted* seams. Those seams never validate what they return, so this is
    the boundary that refuses a naive instant before it is written to an audit
    trail (issue #397).

    Args:
        moment: The datetime to render, normalized to UTC; it **must** be
            timezone-aware. Defaults to the current wall-clock UTC time when
            ``None``.

    Returns:
        A string like ``2026-07-04T12:00:00.000000Z``.

    Raises:
        ValueError: If ``moment`` is provided and carries no UTC offset.
    """
    if moment is None:
        return iso_z(datetime.now(UTC))
    return iso_z(moment)


@dataclass(frozen=True, slots=True)
class ConnectorEvent:
    """One recorded connector event.

    Attributes:
        event_type: The event kind (e.g. ``MARKET_SNAPSHOT``).
        payload: The JSON-safe event body.
        ts: ISO-8601 UTC timestamp of when the event was created.
    """

    event_type: str
    payload: Mapping[str, object]
    ts: str


class EventLedgerWriter(Protocol):
    """The seam through which a connector event is persisted."""

    def record(self, event: ConnectorEvent) -> None:
        """Persist a connector event.

        Args:
            event: The event to persist.
        """
        ...


class MarketScreener(Protocol):
    """The minimal screening contract :class:`MarketSnapshotTask` depends on."""

    def screen_book(
        self, market: NormalizedMarket, order_book: OrderBookSnapshot
    ) -> ScreenResult:
        """Screen a market against its order book and ledger the decision.

        Args:
            market: The market to screen.
            order_book: The market's order-book snapshot.

        Returns:
            The aggregate screening result.
        """
        ...


class LoggingEventLedgerWriter:
    """An :class:`EventLedgerWriter` that logs events instead of persisting them.

    Stands in until a real ledger provides a persisting writer; it emits on the
    module ``windbreak.connector`` logger with the event type in the message so
    operators (and the snapshot wiring) can see each event.
    """

    def record(self, event: ConnectorEvent) -> None:
        """Log a connector event as a single structured line.

        Args:
            event: The event to log.
        """
        _LOGGER.info(
            "connector event recorded event_type=%s ts=%s",
            event.event_type,
            event.ts,
            extra={
                "component": "connector",
                "event_type": event.event_type,
                "ts": event.ts,
            },
        )


class InMemoryEventLedgerWriter:
    """An :class:`EventLedgerWriter` that retains events in memory for tests."""

    def __init__(self) -> None:
        """Initialize with an empty event log."""
        self._events: list[ConnectorEvent] = []

    def record(self, event: ConnectorEvent) -> None:
        """Append a connector event to the in-memory log.

        Args:
            event: The event to retain.
        """
        self._events.append(event)

    def events_by_type(self, event_type: str) -> tuple[ConnectorEvent, ...]:
        """Return every retained event of a given type, in record order.

        Args:
            event_type: The event kind to filter by.

        Returns:
            The matching events.
        """
        return tuple(event for event in self._events if event.event_type == event_type)


class MarketSnapshotTask:
    """Record a snapshot and screening decision for every offered market."""

    def __init__(
        self,
        connector: MarketConnector,
        screener: MarketScreener,
        writer: EventLedgerWriter,
    ) -> None:
        """Initialize the task.

        Args:
            connector: The source of markets to snapshot.
            screener: The screener producing each market's verdict.
            writer: The event-ledger writer that records each event.
        """
        self._connector = connector
        self._screener = screener
        self._writer = writer

    def run_once(self) -> None:
        """Record a snapshot and screen each market the connector offers."""
        for market in self._connector.list_markets():
            self._record_snapshot(market)
            self._screen_market(market)

    def _record_snapshot(self, market: NormalizedMarket) -> None:
        """Record one market's normalized snapshot event.

        Args:
            market: The market to snapshot.
        """
        event = ConnectorEvent(
            event_type=MARKET_SNAPSHOT_EVENT,
            payload=market_to_payload(market),
            ts=utc_now_iso(),
        )
        self._record(event)

    def _screen_market(self, market: NormalizedMarket) -> None:
        """Screen one market against its order book, isolating any failure.

        The injected screener is the single ``SCREEN_DECISION`` emitter; it
        writes to the shared ledger writer itself. A raising writer (surfaced
        here through the screener) is logged and swallowed, mirroring
        :meth:`_record`, so one bad decision write never aborts the run.

        Args:
            market: The market to screen.
        """
        try:
            self._screener.screen_book(
                market, self._connector.get_order_book(market.ticker)
            )
        except Exception as exc:
            _LOGGER.warning(
                "failed to screen market %s (book fetch or decision write): %s",
                market.ticker,
                exc,
                extra={"component": "connector"},
            )

    def _record(self, event: ConnectorEvent) -> None:
        """Record an event via the writer, isolating any failure.

        Args:
            event: The event to record. A raising writer is logged and
                swallowed so one bad record never aborts the run.
        """
        try:
            self._writer.record(event)
        except Exception as exc:
            _LOGGER.warning(
                "event ledger writer failed to record %s event: %s",
                event.event_type,
                exc,
                extra={"component": "connector"},
            )
