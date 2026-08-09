"""The loop's venue reads and refusals feeding the exposure projection (#407).

`tests/scheduler/test_exposure.py` pins the pure projection and
`tests/scheduler/test_concentration_binds.py` pins the cap binding both ways.
What is left, and what this module covers, is the wiring between them: the loop
reading the venue, and every path where it declines because the venue or the
operator left it unable to prove what the account holds.

These branches matter more than their line count suggests. A fail-closed path
nothing exercises is indistinguishable from one that silently passes -- which
is the whole defect issue #407 exists to remove, so leaving its replacement
untested would reproduce it one layer up.

The doubles here are deliberately minimal duck-typed stubs rather than a full
`PaperTickDeps`: these functions read exactly three things (`exchange`,
`config`, `store`) plus the candidate, and a stub that supplies only those
cannot pass by accident through some unrelated real behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from windbreak.config.schema import (
    CorrelationConfig,
    CorrelationTagConfig,
    WindbreakConfig,
)
from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.models import Position
from windbreak.connector.paper import TwoSidedPositionError
from windbreak.numeric import ContractCentis, PricePips
from windbreak.scheduler.exposure import declared_correlation_tags
from windbreak.scheduler.loop import (
    _select_stage,
    _unbucketable_decision,
    read_candidate_exposure,
)
from windbreak.selector.correlation import BUCKET_WEATHER

#: The offset-aware instant every declared tag below carries.
_TAGGED_AT = "2026-03-01T00:00:00+00:00"

#: The market the stub loop is configured to trade.
_TICKER = "KXRAINNYC-26MAR01"

#: A second market the account may hold, for the undeclared-peer path.
_PEER = "KXSNOWCHI-26MAR01"


@dataclass
class _StubMarket:
    """The `NormalizedMarket` fields the exposure wiring reads.

    Attributes:
        ticker: The market's own ticker.
        event_ticker: The parent event's ticker.
    """

    ticker: str
    event_ticker: str


@dataclass
class _StubCandidate:
    """The `MarketCandidate` surface the exposure wiring reads.

    The candidate carries its already-screened market, so the target's event
    ticker needs no second venue read -- and therefore cannot disagree with the
    one the screen judged.

    Attributes:
        market: The screened market's metadata.
    """

    market: _StubMarket

    @property
    def ticker(self) -> str:
        """Return the candidate's market ticker.

        Returns:
            The ticker of the screened market.
        """
        return self.market.ticker


def _candidate(ticker: str = _TICKER) -> _StubCandidate:
    """Build a candidate for one market.

    Args:
        ticker: The market being sized.

    Returns:
        The assembled stub candidate.
    """
    return _StubCandidate(
        market=_StubMarket(ticker=ticker, event_ticker=f"{ticker}-EVENT")
    )


@dataclass
class _StubExchange:
    """A venue answering only the three calls the exposure wiring makes.

    Attributes:
        positions: The rows `get_positions` returns.
        two_sided: When true, `get_positions` raises
            :class:`~windbreak.connector.paper.TwoSidedPositionError`.
        unknown_tickers: Tickers `get_market` refuses to describe.
    """

    positions: tuple[Position, ...] = ()
    two_sided: bool = False
    unknown_tickers: frozenset[str] = frozenset()

    def get_positions(self) -> tuple[Position, ...]:
        """Return the account's rows, or refuse to describe them.

        Returns:
            The configured position rows.

        Raises:
            TwoSidedPositionError: When configured to refuse.
        """
        if self.two_sided:
            raise TwoSidedPositionError(ticker=_TICKER)
        return self.positions

    def get_market(self, ticker: str) -> _StubMarket:
        """Return a market's event ticker, or refuse to describe it.

        Args:
            ticker: The market to describe.

        Returns:
            The stub market.

        Raises:
            UnknownMarketError: When ``ticker`` is configured unknown.
        """
        if ticker in self.unknown_tickers:
            raise UnknownMarketError(ticker)
        return _StubMarket(ticker=ticker, event_ticker=f"{ticker}-EVENT")


@dataclass
class _StubStore:
    """A ledger collecting appended events for assertion.

    Attributes:
        appended: Every event appended, in order.
    """

    appended: list[Any] = field(default_factory=list)

    def append(self, event: Any) -> None:
        """Record one event.

        Args:
            event: The event to record.
        """
        self.appended.append(event)


@dataclass
class _StubDeps:
    """The three `PaperTickDeps` attributes the exposure wiring reads.

    Attributes:
        exchange: The venue double.
        config: The configuration supplying declared buckets.
        store: The ledger double.
    """

    exchange: _StubExchange
    config: WindbreakConfig
    store: _StubStore


@dataclass
class _StubForecast:
    """The one `ForecastRecord` field a declining decision reads.

    Attributes:
        forecast_id: The forecast's identifier.
    """

    forecast_id: str


def _config(*tickers: str) -> WindbreakConfig:
    """Build a configuration declaring each ticker into the weather bucket.

    Args:
        tickers: The markets to declare.

    Returns:
        The assembled configuration.
    """
    return WindbreakConfig(
        correlation=CorrelationConfig(
            tags=tuple(
                CorrelationTagConfig(
                    ticker=ticker, bucket_ids=(BUCKET_WEATHER,), tagged_at=_TAGGED_AT
                )
                for ticker in tickers
            )
        )
    )


def _position(ticker: str) -> Position:
    """Build one held position row.

    Args:
        ticker: The market held.

    Returns:
        The assembled :class:`~windbreak.connector.models.Position`.
    """
    return Position(
        ticker=ticker,
        quantity=ContractCentis(1_000),
        average_price=PricePips(5_000),
    )


def _deps(exchange: _StubExchange, config: WindbreakConfig) -> _StubDeps:
    """Assemble the stub dependency bundle.

    Args:
        exchange: The venue double.
        config: The configuration to read buckets from.

    Returns:
        The assembled stub bundle.
    """
    return _StubDeps(exchange=exchange, config=config, store=_StubStore())


class TestReadTickExposureRefuses:
    """Every path where the tick cannot prove what the account holds."""

    def test_a_two_sided_holding_refuses(self) -> None:
        """A venue that will not describe a two-sided ticker yields `None`.

        The connector declines because the only single-row answer is a netted
        one, and a netted YES-plus-NO reports flat while both legs and their
        collateral are live. Reading that as zero exposure would be the
        fabricated healthy zero the connector just refused to invent.
        """
        deps = _deps(_StubExchange(two_sided=True), _config(_TICKER))
        assert read_candidate_exposure(deps, _candidate()) is None

    def test_an_undescribable_holding_refuses(self) -> None:
        """A held ticker the venue will not describe yields `None`."""
        deps = _deps(
            _StubExchange(
                positions=(_position(_PEER),), unknown_tickers=frozenset({_PEER})
            ),
            _config(_TICKER, _PEER),
        )
        assert read_candidate_exposure(deps, _candidate()) is None

    def test_the_target_is_never_re_read_from_the_venue(self) -> None:
        """The candidate's own screened metadata supplies the target's event.

        The venue is configured to refuse `_TICKER` outright. The projection
        still succeeds, which is the assertion: the target's event ticker comes
        off the already-screened `MarketCandidate`, so no second venue read
        exists that could disagree with the one the screen judged. Only
        *holdings* are looked up.
        """
        deps = _deps(
            _StubExchange(unknown_tickers=frozenset({_TICKER})), _config(_TICKER)
        )
        assert read_candidate_exposure(deps, _candidate()) is not None

    def test_an_undeclared_holding_refuses(self) -> None:
        """A held ticker with no declared bucket yields `None`.

        The venue described it perfectly well; the operator has not classified
        it. It could be in the target's bucket, so the target's bucket exposure
        is unprovable while it is held.
        """
        deps = _deps(_StubExchange(positions=(_position(_PEER),)), _config(_TICKER))
        assert read_candidate_exposure(deps, _candidate()) is None


class TestReadTickExposureProjects:
    """The provable paths, so the refusals above are not unconditional."""

    def test_a_flat_declared_account_projects_zeros(self) -> None:
        """A declared target with no holdings projects, at zero."""
        deps = _deps(_StubExchange(), _config(_TICKER))
        projection = read_candidate_exposure(deps, _candidate())
        assert projection is not None
        assert projection.total_exposure.value == 0

    def test_a_declared_holding_is_priced_into_every_term(self) -> None:
        """A 1_000-centi holding at 5_000 pips prices to 5_000_000 micros.

        Both markets are in the weather bucket, so the peer's value reaches the
        target through `bucket_exposure` while `market_exposure` stays zero --
        different tickers, different events.
        """
        deps = _deps(
            _StubExchange(positions=(_position(_PEER),)), _config(_TICKER, _PEER)
        )
        projection = read_candidate_exposure(deps, _candidate())
        assert projection is not None
        assert projection.total_exposure.value == 5_000_000
        assert projection.bucket_exposure.value == 5_000_000
        assert projection.market_exposure.value == 0
        assert projection.event_exposure.value == 0


class TestSelectStageDeclines:
    """An unprovable tick declines with a stated reason, and emits nothing."""

    def test_declines_and_ledgers_the_reason(self) -> None:
        """`_select_stage` with no exposure emits no intent but says why."""
        deps = _deps(_StubExchange(two_sided=True), _config(_TICKER))
        decision = _select_stage(
            deps,
            _candidate(),
            _StubForecast(forecast_id="forecast-407"),
            None,
            None,
        )
        assert decision.intents == ()
        assert len(decision.reasons) == 1
        assert decision.reasons[0].startswith("unprovable_exposure:")
        assert _TICKER in decision.reasons[0]
        # The decline must reach the ledger, or a reader cannot tell "declined,
        # here is why" from "nothing ran" -- the contract SelectorDecision
        # documents.
        assert len(deps.store.appended) == 1
        assert deps.store.appended[0].intent_count == 0

    def test_the_decline_carries_the_forecast_it_declined(self) -> None:
        """The declining decision echoes the forecast id for traceability."""
        decision = _unbucketable_decision(
            _candidate(), _StubForecast(forecast_id="forecast-407")
        )
        assert decision.forecast_id == "forecast-407"
        assert decision.market_ticker == _TICKER


class TestUnparseableTaggedAtIsRefused:
    """A malformed declared instant raises rather than defaulting."""

    def test_non_iso_tagged_at_raises(self) -> None:
        """A `tagged_at` that is not ISO-8601 at all is refused by name."""
        config = CorrelationConfig(
            tags=(
                CorrelationTagConfig(
                    ticker=_TICKER,
                    bucket_ids=(BUCKET_WEATHER,),
                    tagged_at="last Tuesday",
                ),
            )
        )
        with pytest.raises(ValueError, match="not an ISO-8601 instant"):
            declared_correlation_tags(config, _TICKER)
