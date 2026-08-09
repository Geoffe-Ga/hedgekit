"""A fixture-backed :class:`MarketConnector` for tests and local runs.

:class:`FakeExchange` loads a directory of JSON fixtures into the SPEC S6.2
models, wrapping every arithmetic-bearing value (order-book prices/quantities,
balances, fill and position amounts) in windbreak's scaled-integer unit types at
load time. Because those wrappers reject non-``int`` inputs, a stray float in a
fixture fails loudly rather than silently entering the money path. It exposes
only public, read-only market data (SPEC S5.2); ``place_order`` and
``cancel_order`` raise :class:`NotImplementedError` until a later issue wires
real trading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.models import (
    BalanceSemantics,
    BalanceSnapshot,
    ExchangeStatus,
    FeeModel,
    Fill,
    NormalizedMarket,
    OrderBookLevel,
    OrderBookSnapshot,
    Position,
)
from windbreak.connector.semantics import (
    CancelCollateralRelease,
    FeeDebitTiming,
    FeeRounding,
    HaltedMarketBehavior,
    OrderCollateralInAvailable,
    OrderCollateralInTotal,
    PartialFillRepresentation,
    UnsettledProceeds,
)
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, Literal

    from windbreak.connector.models import OpenOrder

#: Fee-schedule key used when a market has no ticker-specific schedule.
_DEFAULT_FEE_KEY: Final = "default"

#: Narrow a JSON status string to the :class:`ExchangeStatus` literal domain.
_STATUS_BY_NAME: Final[dict[str, Literal["open", "paused", "closed"]]] = {
    "open": "open",
    "paused": "paused",
    "closed": "closed",
}

#: Narrow a JSON side string to the YES/NO literal domain.
_SIDE_BY_NAME: Final[dict[str, Literal["yes", "no"]]] = {"yes": "yes", "no": "no"}


def _read_json(path: Path) -> Any:
    """Parse a JSON fixture file into native Python objects.

    Args:
        path: The fixture file to read.

    Returns:
        The parsed JSON (typed ``Any`` because a fixture may be a list or an
        object; callers narrow it as they build typed models).
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_dt(value: str, *, field: str) -> datetime:
    """Parse an offset-bearing ISO-8601 fixture timestamp into a UTC datetime.

    The offset is required, exactly as in
    :func:`windbreak.connector.kalshi.normalize._parse_dt` -- a test double that
    accepts what production refuses teaches the wrong contract. ``fromisoformat``
    returns a *naive* datetime for an offsetless string and ``.astimezone(UTC)``
    does not raise on one: it reads the wall clock as the **host's** local time,
    so the same checked-in fixture would mean different instants on a
    developer's laptop and in UTC-running CI (issue #392). A fixture whose
    meaning depends on the machine reading it is not a fixture, so this refuses
    rather than guessing an offset.

    Args:
        value: An ISO-8601 string carrying a UTC offset, e.g.
            ``2024-12-01T00:00:00.000000Z``.
        field: The fixture field name ``value`` came from, named in the error so
            the offending entry is locatable.

    Returns:
        The timezone-aware datetime, normalized to UTC.

    Raises:
        ValueError: If ``value`` is not parseable as ISO-8601, or parses to a
            naive (offsetless) datetime.
    """
    parsed = datetime.fromisoformat(value)
    # Naive means `tzinfo is None` *or* a `tzinfo` whose `utcoffset()` returns
    # None -- Python's own definition, reported by `utcoffset()` in one call.
    if parsed.utcoffset() is None:
        raise ValueError(
            f"Fixture timestamp field {field!r} carries no UTC offset: {value!r}. "
            "An offsetless timestamp would be read against the host's local "
            "timezone, so the fixture is refused rather than loaded against a "
            "guessed offset."
        )
    return parsed.astimezone(UTC)


def _parse_optional_dt(value: str | None, *, field: str) -> datetime | None:
    """Parse an optional offset-bearing ISO-8601 timestamp, preserving ``None``.

    Args:
        value: An ISO-8601 string carrying a UTC offset, or None.
        field: The fixture field name ``value`` came from, named in the error.

    Returns:
        The parsed UTC datetime, or None when ``value`` is None.

    Raises:
        ValueError: If ``value`` is present and offsetless or unparseable.
    """
    return None if value is None else _parse_dt(value, field=field)


def _market_from_dict(data: Mapping[str, Any]) -> NormalizedMarket:
    """Build a :class:`NormalizedMarket` from one raw fixture entry."""
    return NormalizedMarket(
        exchange=data["exchange"],
        ticker=data["ticker"],
        event_ticker=data["event_ticker"],
        title=data["title"],
        resolution_criteria=data["resolution_criteria"],
        category=data["category"],
        close_time=_parse_dt(data["close_time"], field="close_time"),
        expected_resolution_time=_parse_optional_dt(
            data["expected_resolution_time"], field="expected_resolution_time"
        ),
        market_type=data["market_type"],
        price_tick_pips=data["price_tick_pips"],
        min_order_contract_centis=data["min_order_contract_centis"],
        fractional_trading_enabled=data["fractional_trading_enabled"],
        mutually_exclusive_group_id=data["mutually_exclusive_group_id"],
        jurisdiction_status=data["jurisdiction_status"],
        raw_exchange_payload_hash=data["raw_exchange_payload_hash"],
        volume_24h_micros=data["volume_24h_micros"],
    )


def _level_from_dict(data: Mapping[str, Any]) -> OrderBookLevel:
    """Build a unit-wrapped :class:`OrderBookLevel` from a raw fixture entry."""
    return OrderBookLevel(
        price=PricePips(data["price"]), quantity=ContractCentis(data["quantity"])
    )


def _book_from_dict(ticker: str, data: Mapping[str, Any]) -> OrderBookSnapshot:
    """Build an :class:`OrderBookSnapshot` from a raw fixture entry."""
    return OrderBookSnapshot(
        ticker=ticker,
        yes_bids=tuple(_level_from_dict(level) for level in data["yes_bids"]),
        yes_asks=tuple(_level_from_dict(level) for level in data["yes_asks"]),
        fetched_at=_parse_dt(data["fetched_at"], field="fetched_at"),
    )


def _fill_from_dict(data: Mapping[str, Any]) -> Fill:
    """Build a unit-wrapped :class:`Fill` from a raw fixture entry."""
    return Fill(
        id=data["id"],
        ticker=data["ticker"],
        side=_SIDE_BY_NAME[data["side"]],
        price=PricePips(data["price"]),
        quantity=ContractCentis(data["quantity"]),
        ts=_parse_dt(data["ts"], field="ts"),
    )


def _position_from_dict(data: Mapping[str, Any]) -> Position:
    """Build a unit-wrapped :class:`Position` from a raw fixture entry."""
    return Position(
        ticker=data["ticker"],
        quantity=ContractCentis(data["quantity"]),
        average_price=PricePips(data["average_price"]),
    )


def _fee_model_from_dict(data: Mapping[str, Any]) -> FeeModel:
    """Build a :class:`FeeModel` from a raw fixture entry."""
    return FeeModel(
        schedule_id=data["schedule_id"],
        maker_fee_ppm=data["maker_fee_ppm"],
        taker_fee_ppm=data["taker_fee_ppm"],
        settlement_fee_ppm=data["settlement_fee_ppm"],
    )


def _load_markets(directory: Path) -> dict[str, NormalizedMarket]:
    """Load ``markets.json`` into a ticker-keyed mapping of markets."""
    entries = _read_json(directory.joinpath("markets.json"))
    markets = [_market_from_dict(entry) for entry in entries]
    return {market.ticker: market for market in markets}


def _load_order_books(directory: Path) -> dict[str, OrderBookSnapshot]:
    """Load ``order_books.json`` into a ticker-keyed mapping of books."""
    data = _read_json(directory.joinpath("order_books.json"))
    return {ticker: _book_from_dict(ticker, book) for ticker, book in data.items()}


def _load_exchange(directory: Path) -> tuple[ExchangeStatus, datetime]:
    """Load ``exchange.json`` into a status and a server time."""
    data = _read_json(directory.joinpath("exchange.json"))
    status = ExchangeStatus(
        status=_STATUS_BY_NAME[data["status"]],
        fetched_at=_parse_dt(data["status_fetched_at"], field="status_fetched_at"),
    )
    return status, _parse_dt(data["exchange_time"], field="exchange_time")


def _load_balances(directory: Path) -> BalanceSnapshot:
    """Load ``balances.json`` into a unit-wrapped :class:`BalanceSnapshot`."""
    data = _read_json(directory.joinpath("balances.json"))
    return BalanceSnapshot(
        total=MoneyMicros(data["total"]),
        available=MoneyMicros(data["available"]),
        fetched_at=_parse_dt(data["fetched_at"], field="fetched_at"),
    )


def _load_balance_semantics(directory: Path) -> BalanceSemantics:
    """Load ``balance_semantics.json`` into a :class:`BalanceSemantics`.

    Each field value is the *name* of an enum member (e.g. ``"UP_TO_NEXT_CENT"``)
    looked up via ``EnumClass[name]``, so a typo'd or invented member name in a
    hand-edited fixture raises ``KeyError`` loudly rather than silently coercing
    to a default.
    """
    data = _read_json(directory.joinpath("balance_semantics.json"))
    return BalanceSemantics(
        open_order_collateral_in_total=OrderCollateralInTotal[
            data["open_order_collateral_in_total"]
        ],
        open_order_collateral_in_available=OrderCollateralInAvailable[
            data["open_order_collateral_in_available"]
        ],
        fee_debit_timing=FeeDebitTiming[data["fee_debit_timing"]],
        fee_rounding=FeeRounding[data["fee_rounding"]],
        partial_fill_representation=PartialFillRepresentation[
            data["partial_fill_representation"]
        ],
        cancel_collateral_release=CancelCollateralRelease[
            data["cancel_collateral_release"]
        ],
        unsettled_proceeds=UnsettledProceeds[data["unsettled_proceeds"]],
        halted_market_behavior=HaltedMarketBehavior[data["halted_market_behavior"]],
    )


def _load_positions(directory: Path) -> tuple[Position, ...]:
    """Load ``positions.json`` into a tuple of unit-wrapped positions."""
    data = _read_json(directory.joinpath("positions.json"))
    return tuple(_position_from_dict(entry) for entry in data)


def _load_fills(directory: Path) -> tuple[Fill, ...]:
    """Load ``fills.json`` into a tuple of unit-wrapped fills."""
    data = _read_json(directory.joinpath("fills.json"))
    return tuple(_fill_from_dict(entry) for entry in data)


def _load_fee_models(directory: Path) -> dict[str, FeeModel]:
    """Load ``fee_models.json`` into a key-keyed mapping of fee models."""
    data = _read_json(directory.joinpath("fee_models.json"))
    return {key: _fee_model_from_dict(value) for key, value in data.items()}


@dataclass(frozen=True, slots=True)
class FakeExchange:
    """A :class:`MarketConnector` served entirely from JSON fixtures.

    Attributes:
        markets: Ticker-keyed normalized markets.
        order_books: Ticker-keyed order-book snapshots.
        exchange_status: The exchange's trading status.
        exchange_time: The exchange's server time.
        balances: The account's balances.
        balance_semantics: The venue's balance-interpretation semantics.
        positions: The account's open positions.
        fills: The account's fills, in fixture order.
        fee_models: Fee schedules keyed by market ticker (plus a ``default``).
    """

    markets: Mapping[str, NormalizedMarket]
    order_books: Mapping[str, OrderBookSnapshot]
    exchange_status: ExchangeStatus
    exchange_time: datetime
    balances: BalanceSnapshot
    balance_semantics: BalanceSemantics
    positions: tuple[Position, ...]
    fills: tuple[Fill, ...]
    fee_models: Mapping[str, FeeModel]

    @classmethod
    def from_fixture_dir(cls, path: str | Path) -> FakeExchange:
        """Build a :class:`FakeExchange` from a directory of JSON fixtures.

        Args:
            path: The directory holding the ``*.json`` exchange fixtures.

        Returns:
            A fully loaded fake exchange.
        """
        directory = Path(path)
        status, exchange_time = _load_exchange(directory)
        return cls(
            markets=_load_markets(directory),
            order_books=_load_order_books(directory),
            exchange_status=status,
            exchange_time=exchange_time,
            balances=_load_balances(directory),
            balance_semantics=_load_balance_semantics(directory),
            positions=_load_positions(directory),
            fills=_load_fills(directory),
            fee_models=_load_fee_models(directory),
        )

    def list_markets(self) -> tuple[NormalizedMarket, ...]:
        """Return every fixture market."""
        return tuple(self.markets.values())

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Return the market for ``ticker`` or raise ``UnknownMarketError``."""
        try:
            return self.markets[ticker]
        except KeyError as exc:
            raise UnknownMarketError(ticker) from exc

    def get_order_book(self, ticker: str) -> OrderBookSnapshot:
        """Return the order book for ``ticker`` or raise ``UnknownMarketError``."""
        try:
            return self.order_books[ticker]
        except KeyError as exc:
            raise UnknownMarketError(ticker) from exc

    def get_exchange_status(self) -> ExchangeStatus:
        """Return the fixture exchange status."""
        return self.exchange_status

    def get_exchange_time(self) -> datetime:
        """Return the fixture exchange time."""
        return self.exchange_time

    def get_balance_semantics(self) -> BalanceSemantics:
        """Return the fixture balance semantics."""
        return self.balance_semantics

    def get_balances(self) -> BalanceSnapshot:
        """Return the fixture balances."""
        return self.balances

    def get_positions(self) -> tuple[Position, ...]:
        """Return the fixture positions."""
        return self.positions

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Return the account's resting open orders (none in the fake book)."""
        return ()

    def get_fills(self, since: datetime) -> tuple[Fill, ...]:
        """Return every fill executed strictly after ``since``.

        Args:
            since: The exclusive lower bound; only fills with ``ts > since``
                are returned.

        Returns:
            The matching fills, in fixture order.
        """
        return tuple(fill for fill in self.fills if fill.ts > since)

    def get_fee_model(self, market_or_series: str) -> FeeModel:
        """Return the fee schedule for a ticker, falling back to ``default``.

        Args:
            market_or_series: The market ticker (or series key) to look up.

        Returns:
            The ticker-specific fee model when present, else the default one.
        """
        return self.fee_models.get(market_or_series, self.fee_models[_DEFAULT_FEE_KEY])

    def place_order(self, normalized_intent: object, approval_token: object) -> object:
        """Reject order placement; real trading is a later issue.

        Args:
            normalized_intent: Unused normalized order intent.
            approval_token: Unused risk-kernel approval token.

        Raises:
            NotImplementedError: Always; the fake exchange is read-only.
        """
        raise NotImplementedError("FakeExchange does not support placing orders")

    def cancel_order(self, order_id: str) -> None:
        """Reject order cancellation; real trading is a later issue.

        Args:
            order_id: Unused venue order identifier.

        Raises:
            NotImplementedError: Always; the fake exchange is read-only.
        """
        raise NotImplementedError("FakeExchange does not support cancelling orders")
