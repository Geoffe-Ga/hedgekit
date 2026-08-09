"""`concentration_limits` binds on real exposure and passes on a good one (#407).

The point of issue #407 is not that the check's arithmetic is wrong --
`tests/riskkernel/test_checks.py` already pins that to the micro. It is that the
scheduler fed the check four zeros, so the arithmetic ran on an account that
looked untouched no matter what it held, reported approval, and proved nothing.

So these tests exercise the *wiring*, from priced holdings through
`project_exposure` and `build_evaluation_context` into the real
`concentration_limits` instance. A test that built an `AccountState` by hand
would pass identically before and after this change and would therefore prove
nothing either.

Both directions are required, and the second matters as much as the first: a cap
that vetoed unconditionally would not be failing closed, it would be broken.
Every figure below is pinned exactly, and the two cases differ *only* in the
size of the held peer position.
"""

from __future__ import annotations

import dataclasses

from windbreak.config.schema import (
    CapitalConfig,
    CorrelationConfig,
    CorrelationTagConfig,
    RiskConfig,
    WindbreakConfig,
)
from windbreak.numeric import (
    ContractCentis,
    MoneyMicros,
    PricePips,
    ProbabilityPpm,
)
from windbreak.riskkernel.checks import DEFAULT_CHECKS, OrderIntent
from windbreak.riskkernel.context import ExchangeTradingStatus
from windbreak.scheduler.exposure import HeldMarket, project_exposure
from windbreak.scheduler.loop import build_evaluation_context
from windbreak.selector.correlation import BUCKET_WEATHER

#: The offset-aware instant every declared tag below carries.
_TAGGED_AT = "2026-03-01T00:00:00+00:00"

#: The market being sized this tick.
_TARGET = "KXRAINNYC-26MAR01"

#: A different market in a different event, declared into the *same* correlation
#: bucket. It is the whole mechanism under test: without a declared bucket it
#: could not be aggregated, and before #407 it was never aggregated at all.
_PEER = "KXSNOWCHI-26MAR01"

#: Worst-case equity for every case below, in micros ($500). With the account
#: state `build_evaluation_context` composes, worst-case equity is exactly the
#: verified available cash, so this is also the balance the fixture reports.
_EQUITY_MICROS = 500_000_000

#: The per-bucket cap: 10% of equity == 50_000_000 micros of headroom.
_BUCKET_PCT_PPM = 100_000

#: 10% of `_EQUITY_MICROS`. Named so the two cases below can be read against it
#: rather than against a bare literal.
_BUCKET_CEILING_MICROS = 50_000_000

#: A peer holding that leaves the target's bucket *under* its ceiling once the
#: order's cost is added: 40_000_000 + 5_000_000 == 45_000_000 <= 50_000_000.
_LEGITIMATE_PEER_MICROS = 40_000_000

#: A peer holding that pushes the target's bucket *over* it:
#: 48_000_000 + 5_000_000 == 53_000_000 > 50_000_000.
_OVERCONCENTRATED_PEER_MICROS = 48_000_000

#: The order under evaluation: 1_000 contract-centis at 5_000 pips, with both
#: fee bounds zero, costs 5_000_000 micros -- the figure both arithmetic
#: comments above depend on.
_INTENT_SIZE_CENTIS = 1_000
_INTENT_PRICE_PIPS = 5_000
_INTENT_COST_MICROS = 5_000_000

#: A clock reading shared by the context and its freshness inputs, so no
#: staleness check fires and `concentration_limits` is the only check under test.
_NOW_EPOCH_S = 1_772_000_000


def _config() -> WindbreakConfig:
    """Build a configuration declaring both markets into the weather bucket.

    Returns:
        The assembled :class:`~windbreak.config.schema.WindbreakConfig`.
    """
    return WindbreakConfig(
        capital=CapitalConfig(floor_micros=0),
        risk=RiskConfig(max_pos_bucket_pct_ppm=_BUCKET_PCT_PPM),
        correlation=CorrelationConfig(
            tags=(
                CorrelationTagConfig(
                    ticker=_TARGET, bucket_ids=(BUCKET_WEATHER,), tagged_at=_TAGGED_AT
                ),
                CorrelationTagConfig(
                    ticker=_PEER, bucket_ids=(BUCKET_WEATHER,), tagged_at=_TAGGED_AT
                ),
            )
        ),
    )


def _bucket_exposure_micros(peer_micros: int) -> int:
    """Project a single peer holding into the target's bucket exposure.

    Args:
        peer_micros: The peer position's mark value, in micros.

    Returns:
        The projected bucket exposure, in micros.
    """
    projection = project_exposure(
        (
            HeldMarket(
                ticker=_PEER,
                event_ticker="KXSNOWCHI",
                value_micros=MoneyMicros(peer_micros),
            ),
        ),
        target_ticker=_TARGET,
        target_event_ticker="KXRAINNYC",
        correlation=_config().correlation,
    )
    assert projection is not None
    return projection.bucket_exposure.value


def _verdict(peer_micros: int) -> str | None:
    """Run the real `concentration_limits` over a context built from holdings.

    Args:
        peer_micros: The peer position's mark value, in micros.

    Returns:
        The check's veto reason, or ``None`` when it approved.
    """
    config = _config()
    projection = project_exposure(
        (
            HeldMarket(
                ticker=_PEER,
                event_ticker="KXSNOWCHI",
                value_micros=MoneyMicros(peer_micros),
            ),
        ),
        target_ticker=_TARGET,
        target_event_ticker="KXRAINNYC",
        correlation=config.correlation,
    )
    context = build_evaluation_context(
        config,
        now_epoch_s=_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({_TARGET}),
        market=None,
        exchange_status=ExchangeTradingStatus.OPEN,
        exchange_status_epoch_s=_NOW_EPOCH_S,
        pipeline_heartbeat_epoch_s=_NOW_EPOCH_S,
        quote_snapshot_epoch_s=_NOW_EPOCH_S,
        exchange_clock_epoch_s=_NOW_EPOCH_S,
        forecast_epoch_s=_NOW_EPOCH_S,
        open_position=None,
        equity_start_of_day=MoneyMicros(_EQUITY_MICROS),
        visible_depth=ContractCentis(1_000_000),
        exposure=projection,
        notional_today=MoneyMicros(0),
    )
    # Worst-case equity is what the cap is a share of, so it is set here rather
    # than left at the composed zero -- a zero-equity account makes every cap
    # zero and both cases below would veto for a reason unrelated to bucketing.
    account = dataclasses.replace(
        context.account,
        exchange_verified_available_cash=MoneyMicros(_EQUITY_MICROS),
    )
    # A provable fee bound: `concentration_limits` vetoes an unprovable cost as
    # `"unprovable"` before it ever reaches the exposure dimensions, and that
    # veto would mask the one under test.
    fees = dataclasses.replace(
        context.fees,
        max_trading_fee=MoneyMicros(0),
        max_settlement_fee=MoneyMicros(0),
    )
    context = dataclasses.replace(context, account=account, fees=fees)
    intent = OrderIntent(
        intent_id="intent-407",
        market_ticker=_TARGET,
        outcome="yes",
        action="buy",
        price=PricePips(_INTENT_PRICE_PIPS),
        size=ContractCentis(_INTENT_SIZE_CENTIS),
        max_notional=MoneyMicros(_EQUITY_MICROS),
        implied_probability=ProbabilityPpm(500_000),
        idempotency_key="key-407",
    )
    check = next(c for c in DEFAULT_CHECKS if c.name == "concentration_limits")
    result = check(intent, context)
    return result.reason if result.vetoed else None


class TestBucketExposureReachesTheKernel:
    """The projected figure is the one the cap compares, to the micro."""

    def test_peer_exposure_is_aggregated_into_the_targets_bucket(self) -> None:
        """A peer in the same declared bucket contributes its full value.

        Before #407 this was `0` for every peer and every holding, because
        `correlation_tags=()` made `effective_buckets(())` empty and
        `aggregate_bucket_exposure` short-circuited to `(0, None)`.
        """
        assert _bucket_exposure_micros(_OVERCONCENTRATED_PEER_MICROS) == 48_000_000

    def test_order_cost_and_ceiling_are_the_values_the_cases_assume(self) -> None:
        """Pin the two arithmetic facts the veto/approve split rests on.

        Without this, a change to the fixture's price or equity could silently
        move both cases to the same side of the ceiling and the pair below would
        still pass while testing nothing.
        """
        assert _INTENT_SIZE_CENTIS * _INTENT_PRICE_PIPS == _INTENT_COST_MICROS
        assert _EQUITY_MICROS * _BUCKET_PCT_PPM // 1_000_000 == _BUCKET_CEILING_MICROS
        assert (
            _OVERCONCENTRATED_PEER_MICROS + _INTENT_COST_MICROS > _BUCKET_CEILING_MICROS
        )
        assert _LEGITIMATE_PEER_MICROS + _INTENT_COST_MICROS <= _BUCKET_CEILING_MICROS


class TestConcentrationLimitsBindsBothWays:
    """The pair that proves the cap is alive rather than merely present."""

    def test_vetoes_a_real_over_concentration(self) -> None:
        """48_000_000 held + 5_000_000 cost > 50_000_000 ceiling: veto."""
        assert _verdict(_OVERCONCENTRATED_PEER_MICROS) == "concentration limit exceeded"

    def test_approves_a_legitimate_position(self) -> None:
        """40_000_000 held + 5_000_000 cost <= 50_000_000 ceiling: approve.

        The half that stops "fail closed" from degenerating into "veto
        everything". Identical to the case above but for the peer's size.
        """
        assert _verdict(_LEGITIMATE_PEER_MICROS) is None


class TestUnprovableExposureFailsClosed:
    """An account whose exposure cannot be established must not pass."""

    def test_undeclared_market_vetoes_rather_than_reading_as_empty(self) -> None:
        """With no declared bucket the projection is `None` and the cap vetoes.

        `build_evaluation_context` zeroes the four concentration caps for a
        `None` projection, which is the only place "unprovable" can be said:
        `AccountState`'s exposure terms have no `None`, so carrying the fact
        there would mean writing the permissive zero this issue removes.
        """
        config = dataclasses.replace(_config(), correlation=CorrelationConfig())
        projection = project_exposure(
            (),
            target_ticker=_TARGET,
            target_event_ticker="KXRAINNYC",
            correlation=config.correlation,
        )
        assert projection is None
        context = build_evaluation_context(
            config,
            now_epoch_s=_NOW_EPOCH_S,
            verification=None,
            instrument_whitelist=frozenset({_TARGET}),
            market=None,
            exchange_status=ExchangeTradingStatus.OPEN,
            exchange_status_epoch_s=_NOW_EPOCH_S,
            pipeline_heartbeat_epoch_s=_NOW_EPOCH_S,
            quote_snapshot_epoch_s=_NOW_EPOCH_S,
            exchange_clock_epoch_s=_NOW_EPOCH_S,
            forecast_epoch_s=_NOW_EPOCH_S,
            open_position=None,
            equity_start_of_day=MoneyMicros(_EQUITY_MICROS),
            visible_depth=ContractCentis(1_000_000),
            exposure=projection,
            notional_today=MoneyMicros(0),
        )
        assert context.limits.max_pos_market_pct_ppm == 0
        assert context.limits.max_pos_event_pct_ppm == 0
        assert context.limits.max_pos_bucket_pct_ppm == 0
        assert context.limits.max_pos_total_pct_ppm == 0

    def test_a_provable_projection_keeps_the_configured_caps(self) -> None:
        """The zeroing is conditional on absent evidence, not unconditional."""
        config = _config()
        projection = project_exposure(
            (),
            target_ticker=_TARGET,
            target_event_ticker="KXRAINNYC",
            correlation=config.correlation,
        )
        context = build_evaluation_context(
            config,
            now_epoch_s=_NOW_EPOCH_S,
            verification=None,
            instrument_whitelist=frozenset({_TARGET}),
            market=None,
            exchange_status=ExchangeTradingStatus.OPEN,
            exchange_status_epoch_s=_NOW_EPOCH_S,
            pipeline_heartbeat_epoch_s=_NOW_EPOCH_S,
            quote_snapshot_epoch_s=_NOW_EPOCH_S,
            exchange_clock_epoch_s=_NOW_EPOCH_S,
            forecast_epoch_s=_NOW_EPOCH_S,
            open_position=None,
            equity_start_of_day=MoneyMicros(_EQUITY_MICROS),
            visible_depth=ContractCentis(1_000_000),
            exposure=projection,
            notional_today=MoneyMicros(0),
        )
        assert context.limits.max_pos_bucket_pct_ppm == _BUCKET_PCT_PPM
