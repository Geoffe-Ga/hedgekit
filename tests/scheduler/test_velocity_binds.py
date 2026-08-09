"""`velocity_limits`' daily-notional cap binds on real booked fills (#415).

`tests/riskkernel/test_checks.py` already pins the check's arithmetic to the
micro. What it cannot prove is that anything ever *feeds* it: the scheduler
passed `notional_today=MoneyMicros(0)` unconditionally, so the daily cap ran on
an account that looked untraded no matter how much the loop had routed, reported
approval, and proved nothing.

So these tests exercise the *wiring*, from hash-chained `FillAccounted` rows
through the UTC-day fold and `build_evaluation_context` into the real
`velocity_limits` instance. A test that built an `AccountState` by hand would
pass identically before and after this change.

Three properties are load-bearing and each has its own case:

* The window comes from the **recorded** instant (`LedgerRecord.created_at`,
  which is bound into the chain hash) and never from a read-time clock. The
  yesterday-stamped fill below is sized so that counting it would flip the
  approving case into a veto.
* A naive `created_at` is **refused**, not assumed UTC. West of UTC a naive
  evening stamp reads as the next calendar day, which is the shape PR #405 found
  failing open.
* The cap binds **and** releases. Without the approving direction a veto test is
  indistinguishable from vetoing everything, which is not fail-closed but
  broken.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from windbreak.config.schema import CapitalConfig, RiskConfig, WindbreakConfig
from windbreak.ledger.events import FillAccounted
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.numeric import (
    ContractCentis,
    MoneyMicros,
    PricePips,
    ProbabilityPpm,
)
from windbreak.riskkernel.checks import DEFAULT_CHECKS, OrderIntent
from windbreak.riskkernel.context import EvaluationContext, ExchangeTradingStatus
from windbreak.scheduler.loop import (
    build_evaluation_context,
    notional_today_micros,
    read_notional_today_micros,
)

if TYPE_CHECKING:
    from pathlib import Path

#: The tick's instant: 2026-03-01T12:00:00Z. Every stamp below is placed
#: relative to the UTC day it names.
_NOW_EPOCH_S = 1_772_366_400

#: Two stamps inside that UTC day, and one inside the previous one. The
#: previous-day stamp is half an hour before midnight: close enough that a fold
#: keyed on anything but the calendar day would swallow it.
_TODAY_MORNING = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
_TODAY_LATER = datetime(2026, 3, 1, 11, 0, tzinfo=UTC)
_YESTERDAY_LATE = datetime(2026, 2, 28, 23, 30, tzinfo=UTC)

#: A stamp carrying no offset at all. Read as UTC it falls inside the tick's
#: day; read on a UTC-5 host it falls inside the *next* one. Nothing establishes
#: which, so the fold must refuse it rather than pick.
_NAIVE_EVENING = datetime(2026, 3, 1, 20, 0, tzinfo=UTC).replace(tzinfo=None)

#: The configured daily cap, in micros ($10).
_MAX_NOTIONAL_PER_DAY_MICROS = 10_000_000

#: The order under evaluation: 1_000 contract-centis at 5_000 pips with both fee
#: bounds zero costs 5_000_000 micros.
_INTENT_SIZE_CENTIS = 1_000
_INTENT_PRICE_PIPS = 5_000
_INTENT_COST_MICROS = 5_000_000

#: A day whose booked notional leaves room for the order:
#: 4_000_000 + 5_000_000 == 9_000_000 <= 10_000_000.
_LEGITIMATE_DAY_MICROS = 4_000_000

#: A day whose booked notional does not:
#: 6_000_000 + 5_000_000 == 11_000_000 > 10_000_000.
_OVERTRADED_DAY_MICROS = 6_000_000

#: Yesterday's booked notional. Deliberately large enough that folding it into
#: today would push even `_LEGITIMATE_DAY_MICROS` over the cap, so the approving
#: case below fails if the window ever stops being a calendar day.
_YESTERDAY_MICROS = 9_000_000

#: The component every fixture stamps on its rows.
_COMPONENT = "scheduler"


def _fill(cash_delta_micros: int, *, fill_id: str) -> FillAccounted:
    """Build one booked fill moving ``cash_delta_micros`` out of cash.

    Args:
        cash_delta_micros: The signed cash movement, in micros. Negative is a
            buy, which is how the fixtures below spend the day's notional.
        fill_id: The venue's identifier for the fill.

    Returns:
        The assembled :class:`~windbreak.ledger.events.FillAccounted`.
    """
    return FillAccounted(
        component=_COMPONENT,
        fill_id=fill_id,
        ticker="KXRAINNYC-26MAR01",
        cash_delta_micros=cash_delta_micros,
        position_delta_centis=100,
    )


def _store(
    tmp_path: Path, stamped: tuple[tuple[datetime, int], ...]
) -> SqliteLedgerStore:
    """Open a ledger holding one booked fill per ``(stamp, cash_delta)`` pair.

    The stamp is the store's injected clock reading at append time, so it lands
    in the row's ``created_at`` -- the field the chain hash covers and the fold
    reads.

    Args:
        tmp_path: The directory the database file is created in.
        stamped: The ``(created_at, cash_delta_micros)`` pairs to append, in
            order.

    Returns:
        The opened store, holding exactly those rows.
    """
    readings = iter(stamp for stamp, _ in stamped)
    store = SqliteLedgerStore(tmp_path / "ledger.db", now=lambda: next(readings))
    for index, (_, cash_delta_micros) in enumerate(stamped):
        store.append(_fill(cash_delta_micros, fill_id=f"fill-{index}"))
    return store


def _legitimate_day() -> tuple[tuple[datetime, int], ...]:
    """Return a day booking `_LEGITIMATE_DAY_MICROS`, plus yesterday's trading.

    Returns:
        The ``(created_at, cash_delta_micros)`` pairs to append.
    """
    return (
        (_YESTERDAY_LATE, -_YESTERDAY_MICROS),
        (_TODAY_MORNING, -2_500_000),
        (_TODAY_LATER, -1_500_000),
    )


def _overtraded_day() -> tuple[tuple[datetime, int], ...]:
    """Return a day booking `_OVERTRADED_DAY_MICROS`, plus yesterday's trading.

    Returns:
        The ``(created_at, cash_delta_micros)`` pairs to append.
    """
    return (
        (_YESTERDAY_LATE, -_YESTERDAY_MICROS),
        (_TODAY_MORNING, -2_500_000),
        (_TODAY_LATER, -3_500_000),
    )


def _config() -> WindbreakConfig:
    """Build a configuration whose daily notional cap is `10_000_000` micros.

    Returns:
        The assembled :class:`~windbreak.config.schema.WindbreakConfig`.
    """
    return WindbreakConfig(
        capital=CapitalConfig(floor_micros=0),
        risk=RiskConfig(
            max_notional_per_day_micros=_MAX_NOTIONAL_PER_DAY_MICROS,
        ),
    )


def _context(notional_today: MoneyMicros | None) -> EvaluationContext:
    """Compose the tick's context around a folded day's notional.

    Args:
        notional_today: The day's booked notional, or ``None`` when the fold
            could not establish it.

    Returns:
        The composed context, with both fee bounds pinned to zero so an
        unprovable cost cannot mask the veto under test.
    """
    context = build_evaluation_context(
        _config(),
        now_epoch_s=_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({"KXRAINNYC-26MAR01"}),
        market=None,
        exchange_status=ExchangeTradingStatus.OPEN,
        exchange_status_epoch_s=_NOW_EPOCH_S,
        pipeline_heartbeat_epoch_s=_NOW_EPOCH_S,
        quote_snapshot_epoch_s=_NOW_EPOCH_S,
        exchange_clock_epoch_s=_NOW_EPOCH_S,
        forecast_epoch_s=_NOW_EPOCH_S,
        open_position=None,
        equity_start_of_day=MoneyMicros(500_000_000),
        visible_depth=ContractCentis(1_000_000),
        exposure=None,
        notional_today=notional_today,
    )
    fees = dataclasses.replace(
        context.fees,
        max_trading_fee=MoneyMicros(0),
        max_settlement_fee=MoneyMicros(0),
    )
    return dataclasses.replace(context, fees=fees)


def _verdict(notional_today: MoneyMicros | None) -> str | None:
    """Run the real `velocity_limits` over a context built from that fold.

    Args:
        notional_today: The day's booked notional, or ``None`` when unprovable.

    Returns:
        The check's veto reason, or ``None`` when it approved.
    """
    intent = OrderIntent(
        intent_id="intent-415",
        market_ticker="KXRAINNYC-26MAR01",
        outcome="yes",
        action="buy",
        price=PricePips(_INTENT_PRICE_PIPS),
        size=ContractCentis(_INTENT_SIZE_CENTIS),
        max_notional=MoneyMicros(_MAX_NOTIONAL_PER_DAY_MICROS),
        implied_probability=ProbabilityPpm(500_000),
        idempotency_key="key-415",
    )
    check = next(c for c in DEFAULT_CHECKS if c.name == "velocity_limits")
    result = check(intent, _context(notional_today))
    return result.reason if result.vetoed else None


class TestTheFoldReadsTheRecordedInstant:
    """The window comes from `created_at`, which the chain hash covers."""

    def test_sums_todays_booked_fills_and_excludes_yesterdays(
        self, tmp_path: Path
    ) -> None:
        """2_500_000 + 1_500_000 == 4_000_000; yesterday's 9_000_000 is out."""
        store = _store(tmp_path, _legitimate_day())
        folded = read_notional_today_micros(store, now_epoch_s=_NOW_EPOCH_S)
        assert folded == MoneyMicros(_LEGITIMATE_DAY_MICROS)

    def test_an_overtraded_day_folds_to_its_exact_total(self, tmp_path: Path) -> None:
        """2_500_000 + 3_500_000 == 6_000_000, again without yesterday's."""
        store = _store(tmp_path, _overtraded_day())
        folded = read_notional_today_micros(store, now_epoch_s=_NOW_EPOCH_S)
        assert folded == MoneyMicros(_OVERTRADED_DAY_MICROS)

    def test_a_day_with_no_booked_fill_is_a_provable_zero(self, tmp_path: Path) -> None:
        """No fill today is evidence of no notional, not absence of evidence.

        This is where the day's notional differs from `equity_start_of_day`: an
        unsampled day has no baseline to read, but an untraded day has provably
        routed nothing, so the cap works from the very first tick instead of
        vetoing until something is booked.
        """
        store = _store(tmp_path, ((_YESTERDAY_LATE, -_YESTERDAY_MICROS),))
        assert read_notional_today_micros(store, now_epoch_s=_NOW_EPOCH_S) == (
            MoneyMicros(0)
        )

    def test_a_credit_and_a_debit_both_consume_the_days_notional(
        self, tmp_path: Path
    ) -> None:
        """Notional routed is the size of the movement, not its sign.

        A sell releases cash and a buy consumes it; both routed an order, so the
        day's budget must count each at its magnitude. Summing signed deltas
        would let a sell *refund* budget a buy had spent.
        """
        store = _store(
            tmp_path, ((_TODAY_MORNING, -2_500_000), (_TODAY_LATER, 1_500_000))
        )
        assert read_notional_today_micros(store, now_epoch_s=_NOW_EPOCH_S) == (
            MoneyMicros(4_000_000)
        )

    def test_the_reverse_walk_and_the_whole_ledger_fold_agree(
        self, tmp_path: Path
    ) -> None:
        """The indexed read path is an optimization, never a behavioral fork."""
        store = _store(tmp_path, _overtraded_day())
        assert read_notional_today_micros(store, now_epoch_s=_NOW_EPOCH_S) == (
            notional_today_micros(store.read_all(), now_epoch_s=_NOW_EPOCH_S)
        )


class TestANaiveStampIsRefused:
    """An offsetless `created_at` is unprovable evidence, so it is refused."""

    def test_naive_created_at_folds_to_none_rather_than_assumed_utc(
        self, tmp_path: Path
    ) -> None:
        """Reading 2026-03-01T20:00 as UTC would book it into the tick's day.

        West of UTC that same wall clock is 2026-03-02, so which day it belongs
        to depends on the host. `require_aware` refuses rather than repairs, and
        the fold reports the day as unestablished.
        """
        store = _store(tmp_path, ((_NAIVE_EVENING, -2_500_000),))
        assert read_notional_today_micros(store, now_epoch_s=_NOW_EPOCH_S) is None

    def test_one_naive_row_makes_the_whole_day_unprovable(self, tmp_path: Path) -> None:
        """A day is not partially provable: an unreadable row could be today's.

        Folding the readable rows alone would report a total smaller than the
        evidence supports, and under-reporting a cap's consumption is exactly
        the permissive direction.
        """
        store = _store(
            tmp_path, ((_TODAY_MORNING, -2_500_000), (_NAIVE_EVENING, -1_500_000))
        )
        assert read_notional_today_micros(store, now_epoch_s=_NOW_EPOCH_S) is None


class TestVelocityLimitsBindsBothWays:
    """The pair that proves the cap is alive rather than merely present."""

    def test_the_cases_rest_on_the_arithmetic_they_claim(self) -> None:
        """Pin the figures, so a fixture edit cannot silently align the pair."""
        assert _INTENT_SIZE_CENTIS * _INTENT_PRICE_PIPS == _INTENT_COST_MICROS
        assert (
            _LEGITIMATE_DAY_MICROS + _INTENT_COST_MICROS <= _MAX_NOTIONAL_PER_DAY_MICROS
        )
        assert (
            _OVERTRADED_DAY_MICROS + _INTENT_COST_MICROS > _MAX_NOTIONAL_PER_DAY_MICROS
        )
        assert (
            _LEGITIMATE_DAY_MICROS + _YESTERDAY_MICROS + _INTENT_COST_MICROS
            > _MAX_NOTIONAL_PER_DAY_MICROS
        )

    def test_vetoes_a_day_already_over_the_notional_cap(self, tmp_path: Path) -> None:
        """6_000_000 booked + 5_000_000 cost > 10_000_000 cap: veto."""
        store = _store(tmp_path, _overtraded_day())
        folded = read_notional_today_micros(store, now_epoch_s=_NOW_EPOCH_S)
        assert _verdict(folded) == "daily notional cap exceeded"

    def test_approves_a_day_legitimately_under_it(self, tmp_path: Path) -> None:
        """4_000_000 booked + 5_000_000 cost <= 10_000_000 cap: approve.

        The half that stops "fail closed" from degenerating into "veto
        everything". Identical to the case above but for one fill's size -- and
        it also fails if yesterday's 9_000_000 ever leaks into today's window.
        """
        store = _store(tmp_path, _legitimate_day())
        folded = read_notional_today_micros(store, now_epoch_s=_NOW_EPOCH_S)
        assert _verdict(folded) is None


class TestAnUnprovableDayFailsClosed:
    """A day whose notional cannot be established must not pass."""

    def test_an_unprovable_day_zeroes_the_configured_daily_cap(self) -> None:
        """`None` states the fact in the limits, not as a fabricated zero.

        `AccountState.notional_today` is a `MoneyMicros` with no `None`, so
        carrying "unprovable" there would mean writing the permissive zero this
        issue removes -- the same seam #414 established for exposure.
        """
        assert _context(None).limits.max_notional_per_day == MoneyMicros(0)

    def test_an_unprovable_day_vetoes_any_positive_cost(self) -> None:
        """The zeroed cap is what the check reads, so the order cannot route."""
        assert _verdict(None) == "daily notional cap exceeded"

    def test_a_provable_day_keeps_the_configured_cap(self) -> None:
        """The zeroing is conditional on absent evidence, not unconditional."""
        context = _context(MoneyMicros(_LEGITIMATE_DAY_MICROS))
        assert context.limits.max_notional_per_day == (
            MoneyMicros(_MAX_NOTIONAL_PER_DAY_MICROS)
        )
        assert context.account.notional_today == MoneyMicros(_LEGITIMATE_DAY_MICROS)
