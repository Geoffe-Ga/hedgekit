"""`daily_loss_limit` and `trailing_drawdown_limit` bind on ledgered equity.

Closes the last two of the five hardcoded-zero risk terms (#513, #514), after
the exposure quartet (#407), the day's booked notional (#415) and the trailing
hour's routed orders (#491). All five lived in one function,
`_account_from_verification`, and all five had the same shape: a config-driven
cap fed a permissive literal by the scheduler's context composition.

`tests/riskkernel/test_checks.py` already pins both caps' arithmetic to the
micro. What it cannot prove is that anything ever *feeds* them:

* `realized_loss_today=MoneyMicros(0)` made `daily_loss_limit` evaluate
  `0 >= ppm_of(equity_start_of_day, 20_000)`, which is false for every account
  whose day has a ledgered baseline -- the loop's normal steady state.
* `equity_high_water_mark=MoneyMicros(0)` made `trailing_drawdown_limit`
  evaluate `0 - equity >= ppm_of(0, 100_000)`, i.e. `negative >= 0`, false for
  every solvent account.

So these tests exercise the *wiring*, from hash-chained `EquitySampled` rows
through the two folds and `build_evaluation_context` into the real check
instances -- and, because a fold nobody calls is the same defect one layer up,
through the very `_approve_stage` call site that composes the context in
production. A test that built an `AccountState` by hand would pass identically
before and after this change.

Both figures are read off one series: the ledger's own equity samples. That is
also why `trailing_drawdown_limit` now measures against
`AccountState.sampled_equity` rather than worst-case equity. A drawdown is the
distance between two readings of *one* quantity; worst-case equity assumes every
open position resolves against us (SPEC glossary), so it falls by a position's
full cost the moment one is opened. Measured against the sampled peak it would
report a 12% "drawdown" for an account that had merely deployed 12% of its
equity -- pre-empting `max_pos_total_pct_ppm` (1_000_000 ppm by default) at a
tenth of its configured reach, and vetoing on deployment rather than on loss.
`test_deploying_capital_is_not_a_drawdown` is the case that pins that.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from windbreak.config.schema import CapitalConfig, RiskConfig, WindbreakConfig
from windbreak.connector.models import (
    BalanceSnapshot,
    ExchangeStatus,
    OrderBookLevel,
    OrderBookSnapshot,
    Position,
)
from windbreak.ledger.events import EquitySampled
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips, ProbabilityPpm
from windbreak.riskkernel.checks import DEFAULT_CHECKS, OrderIntent
from windbreak.scheduler.loop import (
    _EQUITY_MICROS_KEY,
    _EQUITY_SAMPLED_EVENT_TYPE,
    _SAMPLE_EPOCH_KEY,
    _approve_stage,
    _equity_and_positions_stage,
    day_equity_micros,
    equity_curve_micros,
    read_day_equity,
    read_equity_curve,
    read_realized_loss_today_micros,
    read_start_of_day_equity_micros,
)

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.riskkernel.context import EvaluationContext

#: The tick's instant: 2026-03-01T05:30:00Z. Chosen so the UTC day under
#: measurement began five and a half hours ago while the *local* day on a
#: UTC-05:00 host began only half an hour ago -- the boundary a fold bucketing
#: on the host's calendar would move.
_NOW_EPOCH_S = 1_772_343_000

#: A sample stamped on the previous UTC day, 2026-02-28T20:00:00Z. It is the
#: highest reading in every fixture below, so it is the all-time high-water
#: mark; and it must never be the day's baseline.
_YESTERDAY_EVENING_EPOCH_S = 1_772_308_800

#: 2026-03-01T04:00:00Z -- inside the tick's UTC day, but 2026-02-28T23:00
#: local on a UTC-05:00 host. The day's first sample is deliberately placed
#: here: a fold reading the host's calendar would drop it and promote a later,
#: lower sample to the baseline.
_TODAY_FIRST_EPOCH_S = 1_772_337_600

#: 2026-03-01T04:45:00Z and 2026-03-01T05:15:00Z: the day's middle and last
#: samples, both unambiguously inside the day on either host.
_TODAY_MIDDLE_EPOCH_S = 1_772_340_300
_TODAY_LAST_EPOCH_S = 1_772_342_100

#: The configured daily loss limit: 2.5%, deliberately not `RiskConfig`'s 2%
#: default, so a fix that read the default instead of the configuration would
#: miss every boundary below.
_DAILY_LOSS_LIMIT_PCT_PPM = 25_000

#: The configured trailing drawdown limit: 12.5%, again not the 10% default.
_MAX_DRAWDOWN_PCT_PPM = 125_000

#: The day's baseline equity ($400) and the loss threshold 2.5% of it buys.
_BASELINE_MICROS = 400_000_000
_LOSS_THRESHOLD_MICROS = 10_000_000

#: The all-time peak ($500, yesterday evening) and the drawdown threshold
#: 12.5% of it buys.
_PEAK_MICROS = 500_000_000
_DRAWDOWN_THRESHOLD_MICROS = 62_500_000

#: The day's trough that puts realized loss exactly at the threshold, and the
#: one micro above it that leaves the loss one micro short.
_TROUGH_AT_LIMIT_MICROS = 390_000_000
_TROUGH_BELOW_LIMIT_MICROS = 390_000_001

#: The latest sample that puts drawdown from the peak exactly at the threshold,
#: and the one micro above it that leaves drawdown one micro short.
_LATEST_AT_LIMIT_MICROS = 437_500_000
_LATEST_BELOW_LIMIT_MICROS = 437_500_001

#: The two veto strings under test, asserted in full rather than matched.
_LOSS_VETO = "daily loss limit reached"
_DRAWDOWN_VETO = "trailing drawdown limit reached"

#: The market every fixture below trades.
_TICKER = "KXRAINNYC-26MAR01"

#: The component every fixture stamps on its rows.
_COMPONENT = "scheduler"

#: The configured capital floor: zero, so `floor_invariant` cannot supply a
#: veto that masks the one under test.
_FLOOR_MICROS = 0


def _sample(*, equity_micros: int, epoch_s: int) -> EquitySampled:
    """Build one ledgered equity sample.

    Args:
        equity_micros: The sampled equity, in micros.
        epoch_s: The instant the sample attests to, in whole epoch seconds.

    Returns:
        The assembled :class:`~windbreak.ledger.events.EquitySampled`.
    """
    return EquitySampled(
        component=_COMPONENT,
        equity_micros=equity_micros,
        floor_micros=_FLOOR_MICROS,
        epoch_s=epoch_s,
    )


def _store(tmp_path: Path, samples: tuple[tuple[int, int], ...]) -> SqliteLedgerStore:
    """Open a ledger holding one equity sample per ``(epoch_s, equity)`` pair.

    The rows are appended in the order given, so the last pair is the newest
    row by sequence number -- which is what the curve fold reads as "latest".

    Args:
        tmp_path: The directory the database file is created in.
        samples: The ``(epoch_s, equity_micros)`` pairs to append, in order.

    Returns:
        The opened store, holding exactly those rows.
    """
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    for epoch_s, equity_micros in samples:
        store.append(_sample(equity_micros=equity_micros, epoch_s=epoch_s))
    return store


def _day(
    trough_micros: int, *, latest_micros: int | None = None
) -> tuple[tuple[int, int], ...]:
    """Return yesterday's peak plus a day opening at the baseline and sinking.

    Args:
        trough_micros: The day's lowest sampled equity, in micros.
        latest_micros: The day's final sample, in micros. Defaults to the
            trough, i.e. a day that has not recovered.

    Returns:
        The ``(epoch_s, equity_micros)`` pairs to append.
    """
    last = trough_micros if latest_micros is None else latest_micros
    return (
        (_YESTERDAY_EVENING_EPOCH_S, _PEAK_MICROS),
        (_TODAY_FIRST_EPOCH_S, _BASELINE_MICROS),
        (_TODAY_MIDDLE_EPOCH_S, trough_micros),
        (_TODAY_LAST_EPOCH_S, last),
    )


def _config() -> WindbreakConfig:
    """Build the configuration both caps under test are read from.

    Returns:
        The assembled :class:`~windbreak.config.schema.WindbreakConfig`.
    """
    return WindbreakConfig(
        capital=CapitalConfig(floor_micros=_FLOOR_MICROS),
        risk=RiskConfig(
            daily_loss_limit_pct_ppm=_DAILY_LOSS_LIMIT_PCT_PPM,
            max_drawdown_pct_ppm=_MAX_DRAWDOWN_PCT_PPM,
        ),
    )


def _intent() -> OrderIntent:
    """Build the order whose approval the two caps decide.

    Returns:
        The assembled :class:`~windbreak.riskkernel.checks.OrderIntent`.
    """
    return OrderIntent(
        intent_id="intent-513-514",
        market_ticker=_TICKER,
        outcome="yes",
        action="buy",
        price=PricePips(5_000),
        size=ContractCentis(1_000),
        max_notional=MoneyMicros(1_000_000_000),
        implied_probability=ProbabilityPpm(500_000),
        idempotency_key="key-513-514",
    )


def _order_book() -> OrderBookSnapshot:
    """Build a deep two-sided book stamped at the tick's instant.

    Returns:
        The assembled :class:`~windbreak.connector.models.OrderBookSnapshot`.
    """
    level = OrderBookLevel(price=PricePips(5_000), quantity=ContractCentis(1_000_000))
    return OrderBookSnapshot(
        ticker=_TICKER,
        yes_bids=(level,),
        yes_asks=(level,),
        fetched_at=datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC),
    )


@dataclass
class _StubCandidate:
    """The two `MarketCandidate` attributes `_approve_stage` reads.

    Attributes:
        ticker: The market being approved for.
        order_book: The book the participation cap is measured against.
    """

    ticker: str
    order_book: OrderBookSnapshot


@dataclass
class _StubDecision:
    """The one `SelectorDecision` attribute `_approve_stage` reads.

    Attributes:
        intents: The intents this tick emitted.
    """

    intents: tuple[OrderIntent, ...]


@dataclass
class _StubForecast:
    """The one `ForecastRecord` attribute `_approve_stage` reads.

    Attributes:
        created_at: The instant the forecast was created.
    """

    created_at: datetime


@dataclass
class _StubExchange:
    """A venue answering only the calls the two stages under test make.

    Every answer is deliberately permissive, so no check other than the one
    under test can supply the veto.

    Attributes:
        positions: The holdings the venue reports.
        available_micros: The available cash the venue reports.
    """

    positions: tuple[Position, ...] = ()
    available_micros: int = _PEAK_MICROS

    def get_market(self, ticker: str) -> None:
        """Decline to describe the market, which no cap under test reads.

        Args:
            ticker: The market asked about.

        Returns:
            ``None`` -- the eligibility projections fail closed on it, and
            neither cap under test consults them.
        """
        del ticker
        return None

    def get_exchange_status(self) -> ExchangeStatus:
        """Report the venue open, observed at the tick's instant.

        Returns:
            The assembled :class:`~windbreak.connector.models.ExchangeStatus`.
        """
        return ExchangeStatus(
            status="open", fetched_at=datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC)
        )

    def get_exchange_time(self) -> datetime:
        """Report the venue's clock in perfect agreement with ours.

        Returns:
            The tick's instant.
        """
        return datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC)

    def get_positions(self) -> tuple[Position, ...]:
        """Report the configured holdings.

        Returns:
            The venue's positions, empty by default.
        """
        return self.positions

    def get_balances(self) -> BalanceSnapshot:
        """Report the configured available cash.

        Returns:
            The assembled :class:`~windbreak.connector.models.BalanceSnapshot`.
        """
        cash = MoneyMicros(self.available_micros)
        return BalanceSnapshot(
            total=cash,
            available=cash,
            fetched_at=datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC),
        )


@dataclass
class _StubKernel:
    """The one `RiskKernel` attribute `_approve_stage` reads.

    Attributes:
        latest_verification: The kernel's newest verification snapshot.
    """

    latest_verification: None = None


@dataclass(frozen=True)
class _NoTokenOutcome:
    """The `ApprovalOutcome` field `_approve_stage` reads.

    Attributes:
        token: The minted approval token, always absent here.
    """

    token: None = None


#: The single vetoing outcome the capturing seam hands back.
_NO_TOKEN = _NoTokenOutcome()


@dataclass
class _CapturingApproval:
    """An approval seam that records the context and mints nothing.

    Attributes:
        contexts: Every context `_approve_stage` composed, in order.
    """

    contexts: list[EvaluationContext] = field(default_factory=list)

    def decide(self, intent: OrderIntent, context: EvaluationContext) -> Any:
        """Record the composed context and veto.

        Args:
            intent: The intent under approval, unused.
            context: The context `_approve_stage` composed.

        Returns:
            An outcome carrying no token, so no order is ever routed.
        """
        del intent
        self.contexts.append(context)
        return _NO_TOKEN


@dataclass
class _StubDeps:
    """The `PaperTickDeps` attributes the two stages under test read.

    Attributes:
        config: The configuration supplying both caps.
        store: The real hash-chained ledger both folds read.
        exchange: The venue double.
        approval: The capturing approval seam.
        kernel: The kernel double supplying the verification snapshot.
    """

    config: WindbreakConfig
    store: SqliteLedgerStore
    exchange: _StubExchange
    approval: _CapturingApproval
    kernel: _StubKernel

    def clock(self) -> int:
        """Return the tick's fixed instant.

        Returns:
            `_NOW_EPOCH_S`.
        """
        return _NOW_EPOCH_S


def _deps(store: SqliteLedgerStore, exchange: _StubExchange) -> Any:
    """Assemble the dependency double the production stages run against.

    Args:
        store: The ledger holding the equity samples under test.
        exchange: The venue double.

    Returns:
        The assembled `_StubDeps`, typed loosely so it can stand in for the
        real frozen `PaperTickDeps` without reimplementing its whole surface.
    """
    return _StubDeps(
        config=_config(),
        store=store,
        exchange=exchange,
        approval=_CapturingApproval(),
        kernel=_StubKernel(),
    )


def _composed_context(
    store: SqliteLedgerStore, *, exchange: _StubExchange | None = None
) -> EvaluationContext:
    """Drive the real `_approve_stage` and return the context it composed.

    This is the composition proof: nothing here hand-builds an `AccountState`
    or calls `build_evaluation_context` directly. Both figures reach their
    checks only if the production call site folds them out of `store`.

    Args:
        store: The ledger holding this account's equity samples.
        exchange: The venue double, flat and solvent by default.

    Returns:
        The single :class:`~windbreak.riskkernel.context.EvaluationContext`
        `_approve_stage` composed for the tick's one intent.
    """
    deps = _deps(store, _StubExchange() if exchange is None else exchange)
    candidate: Any = _StubCandidate(ticker=_TICKER, order_book=_order_book())
    decision: Any = _StubDecision(intents=(_intent(),))
    forecast: Any = _StubForecast(
        created_at=datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC)
    )
    _approve_stage(deps, candidate, decision, _NOW_EPOCH_S, forecast, None)
    assert len(deps.approval.contexts) == 1
    return deps.approval.contexts[0]


def _verdict(context: EvaluationContext, name: str) -> str | None:
    """Run one real check over a composed context.

    Args:
        context: The context `_approve_stage` composed.
        name: The check to run, by its own registered name.

    Returns:
        The check's veto reason, or ``None`` when it approved.
    """
    fees = dataclasses.replace(
        context.fees,
        max_trading_fee=MoneyMicros(0),
        max_settlement_fee=MoneyMicros(0),
    )
    check = next(c for c in DEFAULT_CHECKS if c.name == name)
    result = check(_intent(), dataclasses.replace(context, fees=fees))
    return result.reason if result.vetoed else None


class TestTheFixturesRestOnTheArithmeticTheyClaim:
    """Pin every figure, so a fixture edit cannot silently align a pair."""

    def test_the_tick_sits_where_the_two_calendars_disagree(self) -> None:
        """05:30Z is 00:30 local on UTC-05:00, so the local day is younger."""
        assert datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC) == datetime(
            2026, 3, 1, 5, 30, tzinfo=UTC
        )
        assert datetime.fromtimestamp(_TODAY_FIRST_EPOCH_S, tz=UTC) == datetime(
            2026, 3, 1, 4, 0, tzinfo=UTC
        )
        assert datetime.fromtimestamp(_YESTERDAY_EVENING_EPOCH_S, tz=UTC) == datetime(
            2026, 2, 28, 20, 0, tzinfo=UTC
        )

    def test_the_loss_boundary_is_one_micro_wide(self) -> None:
        """2.5% of the $400 baseline is 10_000_000 micros, to the micro."""
        assert _BASELINE_MICROS * _DAILY_LOSS_LIMIT_PCT_PPM // 1_000_000 == (
            _LOSS_THRESHOLD_MICROS
        )
        assert _BASELINE_MICROS - _TROUGH_AT_LIMIT_MICROS == _LOSS_THRESHOLD_MICROS
        assert (
            _BASELINE_MICROS - _TROUGH_BELOW_LIMIT_MICROS == _LOSS_THRESHOLD_MICROS - 1
        )

    def test_the_drawdown_boundary_is_one_micro_wide(self) -> None:
        """12.5% of the $500 peak is 62_500_000 micros, to the micro."""
        assert _PEAK_MICROS * _MAX_DRAWDOWN_PCT_PPM // 1_000_000 == (
            _DRAWDOWN_THRESHOLD_MICROS
        )
        assert _PEAK_MICROS - _LATEST_AT_LIMIT_MICROS == _DRAWDOWN_THRESHOLD_MICROS
        assert (
            _PEAK_MICROS - _LATEST_BELOW_LIMIT_MICROS == _DRAWDOWN_THRESHOLD_MICROS - 1
        )

    def test_the_two_thresholds_cannot_be_confused_for_each_other(self) -> None:
        """No figure here does double duty, so neither check can borrow the
        other's evidence and still look right.
        """
        figures = {
            _BASELINE_MICROS,
            _PEAK_MICROS,
            _LOSS_THRESHOLD_MICROS,
            _DRAWDOWN_THRESHOLD_MICROS,
            _TROUGH_AT_LIMIT_MICROS,
            _LATEST_AT_LIMIT_MICROS,
        }
        assert len(figures) == 6

    def test_neither_configured_share_is_the_shipped_default(self) -> None:
        """A fix reading `RiskConfig()` instead of the tick's configuration
        would miss every boundary in this module.
        """
        assert RiskConfig().daily_loss_limit_pct_ppm != _DAILY_LOSS_LIMIT_PCT_PPM
        assert RiskConfig().max_drawdown_pct_ppm != _MAX_DRAWDOWN_PCT_PPM


class TestTheLoopFeedsTodaysRealizedLoss:
    """The loss the production call site composes is the ledger's own."""

    def test_the_composed_account_carries_the_days_measured_loss(
        self, tmp_path: Path
    ) -> None:
        """400_000_000 opened, 390_000_000 at the trough: 10_000_000 lost."""
        context = _composed_context(_store(tmp_path, _day(_TROUGH_AT_LIMIT_MICROS)))
        assert context.account.realized_loss_today == MoneyMicros(
            _LOSS_THRESHOLD_MICROS
        )
        assert context.account.equity_start_of_day == MoneyMicros(_BASELINE_MICROS)

    def test_yesterdays_higher_sample_is_not_todays_baseline(
        self, tmp_path: Path
    ) -> None:
        """Yesterday closed at $500 and the day opened at $400.

        Folding yesterday in would raise the baseline to the peak and report a
        110_000_000 loss against a 12_500_000 threshold, so every approving
        case in this module would flip to a veto.
        """
        context = _composed_context(_store(tmp_path, _day(_TROUGH_BELOW_LIMIT_MICROS)))
        assert context.account.equity_start_of_day == MoneyMicros(_BASELINE_MICROS)
        assert context.account.realized_loss_today == MoneyMicros(
            _LOSS_THRESHOLD_MICROS - 1
        )

    def test_the_configured_share_is_what_the_limits_carry(
        self, tmp_path: Path
    ) -> None:
        """The threshold is a share of the baseline, not of a default."""
        context = _composed_context(_store(tmp_path, _day(_TROUGH_AT_LIMIT_MICROS)))
        assert context.limits.daily_loss_limit_pct_ppm == _DAILY_LOSS_LIMIT_PCT_PPM


class TestTheDailyLossLimitBindsBothWays:
    """The pair that proves the cap is alive rather than merely present."""

    def test_vetoes_a_day_whose_loss_reached_the_threshold(
        self, tmp_path: Path
    ) -> None:
        """10_000_000 lost >= 10_000_000 threshold: veto, by exact reason."""
        context = _composed_context(_store(tmp_path, _day(_TROUGH_AT_LIMIT_MICROS)))
        assert _verdict(context, "daily_loss_limit") == _LOSS_VETO

    def test_approves_one_micro_below_the_threshold(self, tmp_path: Path) -> None:
        """9_999_999 lost < 10_000_000 threshold: approve.

        The half that stops "fail closed" from degenerating into "veto
        everything". Identical to the case above but for one micro of trough.
        """
        context = _composed_context(_store(tmp_path, _day(_TROUGH_BELOW_LIMIT_MICROS)))
        assert _verdict(context, "daily_loss_limit") is None


class TestARecoveryDoesNotUnbookTheDaysLoss:
    """The day's loss is measured to its trough, not to its latest sample."""

    def test_equity_recovering_to_the_baseline_still_vetoes(
        self, tmp_path: Path
    ) -> None:
        """SPEC S10.10 pauses a daily-loss breach *to the next UTC day*.

        A loss measured against the latest sample would un-book itself the
        moment the mark recovered, so a volatile day could cross the limit
        repeatedly and trade on every rebound. The day here sinks to the
        threshold and returns to its opening equity; the cap must still bind.
        """
        samples = _day(_TROUGH_AT_LIMIT_MICROS, latest_micros=_BASELINE_MICROS)
        context = _composed_context(_store(tmp_path, samples))
        assert context.account.realized_loss_today == MoneyMicros(
            _LOSS_THRESHOLD_MICROS
        )
        assert _verdict(context, "daily_loss_limit") == _LOSS_VETO


class TestTheLoopFeedsTheEquityHighWaterMark:
    """The mark and the current reading are two points on one series."""

    def test_the_composed_account_carries_the_all_time_peak(
        self, tmp_path: Path
    ) -> None:
        """Yesterday's $500 is the highest sample the ledger holds."""
        context = _composed_context(_store(tmp_path, _day(_LATEST_AT_LIMIT_MICROS)))
        assert context.account.equity_high_water_mark == MoneyMicros(_PEAK_MICROS)

    def test_the_composed_account_carries_the_newest_sample(
        self, tmp_path: Path
    ) -> None:
        """The current reading is the newest row, not the newest *stamp*."""
        context = _composed_context(_store(tmp_path, _day(_LATEST_AT_LIMIT_MICROS)))
        assert context.account.sampled_equity == MoneyMicros(_LATEST_AT_LIMIT_MICROS)

    def test_a_peak_from_an_earlier_day_still_counts(self, tmp_path: Path) -> None:
        """The mark is all-time, so it ratchets and never resets at midnight.

        A mark rebuilt from today's samples alone would read $400 here, put the
        threshold at 50_000_000, and approve the veto case below -- a "trailing
        drawdown" that forgives itself every UTC midnight.
        """
        curve = read_equity_curve(_store(tmp_path, _day(_LATEST_AT_LIMIT_MICROS)))
        assert curve is not None
        assert curve.high_water_mark == MoneyMicros(_PEAK_MICROS)
        assert curve.high_water_mark > MoneyMicros(_BASELINE_MICROS)

    def test_the_mark_survives_a_process_restart(self, tmp_path: Path) -> None:
        """A high-water mark that resets on restart is not a high-water mark.

        It is not held in memory: it is re-folded from the ledger's own rows,
        so a second store object opened on the same database recovers exactly
        the same peak. That is the durability the #442 daily-budget lesson asks
        for, without a second place to keep state.
        """
        _store(tmp_path, _day(_LATEST_AT_LIMIT_MICROS))
        reopened = SqliteLedgerStore(tmp_path / "ledger.db")
        curve = read_equity_curve(reopened)
        assert curve is not None
        assert curve.high_water_mark == MoneyMicros(_PEAK_MICROS)
        assert curve.latest == MoneyMicros(_LATEST_AT_LIMIT_MICROS)

    def test_the_configured_share_is_what_the_limits_carry(
        self, tmp_path: Path
    ) -> None:
        """The threshold is a share of the mark, not of a default."""
        context = _composed_context(_store(tmp_path, _day(_LATEST_AT_LIMIT_MICROS)))
        assert context.limits.max_drawdown_pct_ppm == _MAX_DRAWDOWN_PCT_PPM


class TestTheTrailingDrawdownLimitBindsBothWays:
    """The pair that proves the cap is alive rather than merely present."""

    def test_vetoes_a_drawdown_that_reached_the_threshold(self, tmp_path: Path) -> None:
        """500_000_000 - 437_500_000 == 62_500_000 threshold: veto."""
        context = _composed_context(_store(tmp_path, _day(_LATEST_AT_LIMIT_MICROS)))
        assert _verdict(context, "trailing_drawdown_limit") == _DRAWDOWN_VETO

    def test_approves_one_micro_below_the_threshold(self, tmp_path: Path) -> None:
        """62_499_999 drawn down < 62_500_000 threshold: approve.

        Identical to the case above but for one micro of the newest sample.
        """
        context = _composed_context(_store(tmp_path, _day(_LATEST_BELOW_LIMIT_MICROS)))
        assert _verdict(context, "trailing_drawdown_limit") is None

    def test_deploying_capital_is_not_a_drawdown(self, tmp_path: Path) -> None:
        """The dimension the check measures in, pinned.

        The venue reports a $250 holding, so worst-case equity -- which assumes
        every open position resolves against us -- is half the sampled equity
        and 50% below the mark. Measured against *that*, the cap would veto an
        account that had merely deployed capital, pre-empting the configured
        concentration caps and vetoing on deployment rather than on loss.
        Measured on the equity series, this account has drawn down nothing.
        """
        samples = ((_TODAY_FIRST_EPOCH_S, _PEAK_MICROS),)
        held = Position(
            ticker=_TICKER,
            quantity=ContractCentis(50_000),
            average_price=PricePips(5_000),
        )
        exchange = _StubExchange(
            positions=(held,), available_micros=_PEAK_MICROS - 250_000_000
        )
        context = _composed_context(_store(tmp_path, samples), exchange=exchange)
        assert context.account.sampled_equity == MoneyMicros(_PEAK_MICROS)
        assert context.account.exchange_verified_available_cash == MoneyMicros(0)
        assert _verdict(context, "trailing_drawdown_limit") is None


class TestAnUnsampledLedgerFailsClosed:
    """Neither cap may pass on an account whose equity nothing attests to."""

    def test_a_fresh_ledger_folds_to_no_reading_at_all(self, tmp_path: Path) -> None:
        """`None`, never a zero dressed up as an observation.

        Each named read is asserted, not just the day fold behind them: an
        unsampled day and a day that provably lost nothing map onto the same
        `AccountState`, so only the contract at this seam distinguishes "no
        evidence" from "evidence of no loss". Letting the reads answer zero
        would be indistinguishable here today and permissive the moment either
        term stops being paired with the baseline it is measured against.
        """
        store = _store(tmp_path, ())
        assert read_day_equity(store, now_epoch_s=_NOW_EPOCH_S) is None
        assert read_start_of_day_equity_micros(store, now_epoch_s=_NOW_EPOCH_S) is None
        assert read_realized_loss_today_micros(store, now_epoch_s=_NOW_EPOCH_S) is None
        assert read_equity_curve(store) is None

    def test_a_fresh_ledger_vetoes_both_caps(self, tmp_path: Path) -> None:
        """The paired zeros veto: `0 >= ppm_of(0)` for each cap.

        Carrying "unprovable" as zero is safe here precisely *because* both
        sides of each comparison come from the same fold. A zero loss against a
        zero baseline reaches its threshold, and a zero mark against a zero
        current reading reaches its own -- which is why neither needs the
        limits-side seam #407/#415/#491 used, where only one side was folded.
        """
        context = _composed_context(_store(tmp_path, ()))
        assert context.account.equity_start_of_day == MoneyMicros(0)
        assert context.account.realized_loss_today == MoneyMicros(0)
        assert context.account.equity_high_water_mark == MoneyMicros(0)
        assert context.account.sampled_equity == MoneyMicros(0)
        assert _verdict(context, "daily_loss_limit") == _LOSS_VETO
        assert _verdict(context, "trailing_drawdown_limit") == _DRAWDOWN_VETO

    def test_one_sample_lifts_both_vetoes(self, tmp_path: Path) -> None:
        """Failing closed is conditional on absent evidence, not permanent.

        The half that separates "fail closed" from "broken": the very first
        ledgered sample gives each cap a base to measure against, and both
        release.
        """
        store = _store(tmp_path, ((_TODAY_FIRST_EPOCH_S, _BASELINE_MICROS),))
        context = _composed_context(store)
        assert _verdict(context, "daily_loss_limit") is None
        assert _verdict(context, "trailing_drawdown_limit") is None

    def test_a_day_with_no_sample_keeps_the_baseline_veto(self, tmp_path: Path) -> None:
        """PR #481's unknown-baseline veto, unchanged.

        Yesterday's sample proves a mark but says nothing about today, so the
        loss limit keeps vetoing on the unknown baseline while the drawdown
        limit -- which has the evidence it needs -- does not.
        """
        store = _store(tmp_path, ((_YESTERDAY_EVENING_EPOCH_S, _PEAK_MICROS),))
        context = _composed_context(store)
        assert context.account.equity_start_of_day == MoneyMicros(0)
        assert _verdict(context, "daily_loss_limit") == _LOSS_VETO
        assert _verdict(context, "trailing_drawdown_limit") is None


class TestTheDayBucketDoesNotMoveWithTheHostsTimezone:
    """CI runs UTC, where a locally-bucketed day and a correct one agree.

    The day's first sample is stamped 04:00Z, which is 23:00 on the *previous*
    day for a UTC-05:00 host and 09:00 the same day for a UTC+05:00 one. A fold
    reading the host's calendar would drop it west of UTC, promote the trough
    to baseline, and report no loss at all.
    """

    def test_the_loss_is_identical_west_of_utc(
        self, tmp_path: Path, local_timezone_utc_minus_5: None
    ) -> None:
        """UTC-05:00, where the tick's local wall clock reads 00:30."""
        del local_timezone_utc_minus_5
        context = _composed_context(_store(tmp_path, _day(_TROUGH_AT_LIMIT_MICROS)))
        assert context.account.equity_start_of_day == MoneyMicros(_BASELINE_MICROS)
        assert context.account.realized_loss_today == MoneyMicros(
            _LOSS_THRESHOLD_MICROS
        )
        assert _verdict(context, "daily_loss_limit") == _LOSS_VETO

    def test_the_loss_is_identical_east_of_utc(
        self, tmp_path: Path, local_timezone_utc_plus_5: None
    ) -> None:
        """UTC+05:00, which skews a misread instant the opposite way."""
        del local_timezone_utc_plus_5
        context = _composed_context(_store(tmp_path, _day(_TROUGH_AT_LIMIT_MICROS)))
        assert context.account.realized_loss_today == MoneyMicros(
            _LOSS_THRESHOLD_MICROS
        )
        assert _verdict(context, "daily_loss_limit") == _LOSS_VETO

    def test_the_boundary_still_releases_west_of_utc(
        self, tmp_path: Path, local_timezone_utc_minus_5: None
    ) -> None:
        """The approving direction holds under the pin too, to the micro."""
        del local_timezone_utc_minus_5
        context = _composed_context(_store(tmp_path, _day(_TROUGH_BELOW_LIMIT_MICROS)))
        assert _verdict(context, "daily_loss_limit") is None

    def test_the_drawdown_is_identical_west_of_utc(
        self, tmp_path: Path, local_timezone_utc_minus_5: None
    ) -> None:
        """The mark spans two UTC days, so it is the fold most exposed here."""
        del local_timezone_utc_minus_5
        context = _composed_context(_store(tmp_path, _day(_LATEST_AT_LIMIT_MICROS)))
        assert context.account.equity_high_water_mark == MoneyMicros(_PEAK_MICROS)
        assert _verdict(context, "trailing_drawdown_limit") == _DRAWDOWN_VETO


class TestTheTwoReadPathsAgree:
    """The indexed walk is an optimization, never a behavioral fork."""

    def test_the_day_walk_and_the_whole_ledger_fold_agree(self, tmp_path: Path) -> None:
        """Both paths answer identically on a day straddling the boundary."""
        store = _store(tmp_path, _day(_TROUGH_AT_LIMIT_MICROS))
        assert read_day_equity(store, now_epoch_s=_NOW_EPOCH_S) == day_equity_micros(
            store.read_all(), now_epoch_s=_NOW_EPOCH_S
        )

    def test_the_curve_walk_and_the_whole_ledger_fold_agree(
        self, tmp_path: Path
    ) -> None:
        """Newest-first and oldest-first must reach the same peak and latest."""
        store = _store(tmp_path, _day(_LATEST_AT_LIMIT_MICROS))
        assert read_equity_curve(store) == equity_curve_micros(store.read_all())

    def test_both_paths_refuse_the_same_unsampled_ledger(self, tmp_path: Path) -> None:
        """The refusal is a property of the fold, not of one read path."""
        store = _store(tmp_path, ())
        assert day_equity_micros(store.read_all(), now_epoch_s=_NOW_EPOCH_S) is None
        assert equity_curve_micros(store.read_all()) is None

    def test_the_baseline_read_is_the_day_folds_own_answer(
        self, tmp_path: Path
    ) -> None:
        """One walk defines the baseline and the loss, so they cannot disagree.

        The threshold is a share of the baseline and the loss is measured from
        it; folding them separately would let a clock anomaly move one without
        the other.
        """
        store = _store(tmp_path, _day(_TROUGH_AT_LIMIT_MICROS))
        day = read_day_equity(store, now_epoch_s=_NOW_EPOCH_S)
        assert day is not None
        assert read_start_of_day_equity_micros(store, now_epoch_s=_NOW_EPOCH_S) == (
            day.baseline
        )
        assert read_realized_loss_today_micros(store, now_epoch_s=_NOW_EPOCH_S) == (
            day.realized_loss
        )


class TestTheRowsProductionWritesAreTheRowsFolded:
    """A positive control written by the loop's own equity stage."""

    def test_the_sample_the_equity_stage_writes_is_the_sample_read_back(
        self, tmp_path: Path
    ) -> None:
        """`_equity_and_positions_stage` is what appends every real sample.

        Building the row by hand alone would prove only that the folds agree
        with this module's fixture: they would keep passing after a payload
        rename that stopped production ever being read.
        """
        store = _store(tmp_path, ())
        deps = _deps(store, _StubExchange(available_micros=_BASELINE_MICROS))
        sampled = _equity_and_positions_stage(deps, _TODAY_MIDDLE_EPOCH_S)
        assert sampled == _BASELINE_MICROS
        curve = read_equity_curve(store)
        assert curve is not None
        assert curve.latest == MoneyMicros(_BASELINE_MICROS)
        day = read_day_equity(store, now_epoch_s=_NOW_EPOCH_S)
        assert day is not None
        assert day.baseline == MoneyMicros(_BASELINE_MICROS)

    def test_the_event_type_and_payload_keys_are_derived_not_restated(self) -> None:
        """A rename must break this test, not silently disconnect a cap."""
        written = _sample(equity_micros=_BASELINE_MICROS, epoch_s=_TODAY_FIRST_EPOCH_S)
        assert written.event_type == _EQUITY_SAMPLED_EVENT_TYPE
        assert set(written.payload) >= {_EQUITY_MICROS_KEY, _SAMPLE_EPOCH_KEY}
