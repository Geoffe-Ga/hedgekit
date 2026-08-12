"""The full evaluation context a Risk Kernel pre-trade check reads (SPEC S10).

Issue #30 promotes the 24 SPEC S10.3 checks from reading an
:class:`~windbreak.riskkernel.checks.OrderIntent` alone to reading an
:class:`EvaluationContext`: the intent *plus* everything a check needs to make
a real risk decision -- the operating mode, the configured risk limits, the
current account state, the market view, the fee upper bounds, and the wall
clock. The context is assembled from five frozen, slotted dataclasses so that:

    * each concern (limits vs. live account state vs. market data vs. fees)
      owns its own immutable value object, and
    * a check can be handed the whole context yet still only *read* it -- no
      check can mutate the state it is evaluating.

Every numeric field is a :mod:`windbreak.numeric` scaled-integer type (or a
plain ``int`` count / epoch second), never a float (SPEC S6.1). Optional
market and fee fields are ``None`` when the datum is unavailable, and a check
that depends on a ``None`` field fails closed (vetoes) rather than guessing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from windbreak.numeric.types import ContractCentis, MoneyMicros, PricePips
    from windbreak.riskkernel.modes import Mode
    from windbreak.riskkernel.verification import VerificationSnapshot


class ExchangeTradingStatus(enum.Enum):
    """The exchange's trading status, as the kernel's ``exchange_status_ok`` reads it.

    This mirrors 1:1 the connector's
    :attr:`windbreak.connector.models.ExchangeStatus.status`
    ``Literal["open", "paused", "closed"]`` domain, but is deliberately **not**
    imported from the connector: keeping the kernel's own enum preserves the
    SPEC S5 Process-B trust boundary (the kernel context never imports connector
    types) and keeps this context module all-int-epoch, free of any ``datetime``
    the connector models carry. "Unknown" or "missing" is modeled as ``None`` on
    :class:`MarketView` -- never as an enum member here -- so the absence of a
    status can only ever fail closed (veto), never masquerade as a tradable
    state. Only :attr:`OPEN` is tradable.

    Attributes:
        OPEN: The exchange is open for trading -- the sole tradable status.
        PAUSED: The exchange has paused trading; not tradable.
        CLOSED: The exchange is closed; not tradable.
    """

    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"


class JurisdictionStatus(enum.Enum):
    """Whether a market is jurisdiction-eligible, as the kernel's check reads it.

    Mirrors the connector's
    :attr:`windbreak.connector.models.NormalizedMarket.jurisdiction_status`
    ``Literal["eligible", "ineligible", "unknown"]`` domain, but deliberately
    **not** imported from the connector, for the same SPEC S5 Process-B reason
    :class:`ExchangeTradingStatus` documents.

    Note the missing third member. The connector's ``"unknown"`` maps to ``None``
    on :class:`MarketView` rather than to an enum member, so an unproven
    jurisdiction has no value it could ever be compared equal to
    :attr:`ELIGIBLE`. Absence can only fail closed (SPEC S1.1 invariant 1).

    Attributes:
        ELIGIBLE: The market is proven jurisdiction-eligible -- the sole
            tradable status.
        INELIGIBLE: The market is proven ineligible; never tradable.
    """

    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


class ProductType(enum.Enum):
    """The product shape a market is, as the kernel's eligibility check reads it.

    Mirrors the connector's
    :attr:`windbreak.connector.models.NormalizedMarket.market_type`, which is a
    single-valued ``Literal["fully_collateralized_binary"]``. This enum has one
    member for the same reason: every other raw exchange product is refused
    before normalization and can never reach a
    :class:`~windbreak.connector.models.NormalizedMarket`, so a second member
    would be a value no production path can produce.

    An unrecognized or absent product maps to ``None`` on :class:`MarketView`,
    never to a member, so it can only fail closed.

    Attributes:
        FULLY_COLLATERALIZED_BINARY: A fully collateralized binary contract --
            the sole tradable product (SPEC S1.1 invariant 2).
    """

    FULLY_COLLATERALIZED_BINARY = "fully_collateralized_binary"


@dataclass(frozen=True, slots=True)
class FeeBounds:
    """Worst-case fee upper bounds for the order under evaluation.

    Each bound is ``None`` when the kernel cannot yet prove an upper bound on
    that fee; a check that needs the bound fails closed on ``None`` rather than
    assuming a fee of zero.

    Attributes:
        max_trading_fee: The worst-case trading fee, in micros, or ``None`` if
            no upper bound is provable.
        max_settlement_fee: The worst-case settlement fee, in micros, or
            ``None`` if no upper bound is provable.
    """

    max_trading_fee: MoneyMicros | None
    max_settlement_fee: MoneyMicros | None


@dataclass(frozen=True, slots=True)
class AccountState:
    """A snapshot of the account state the risk checks evaluate against.

    Groups the worst-case-equity inputs, the day's reference equities and
    realized loss, the four concentration-exposure dimensions, and the velocity
    counters, all as scaled integers.

    Attributes:
        exchange_verified_available_cash: Exchange-confirmed available cash, in
            micros.
        guaranteed_terminal_value_of_positions: Guaranteed resolution value of
            open positions, in micros.
        pending_kernel_reservations: Capital reserved against in-flight
            approvals, in micros.
        unresolved_fee_upper_bounds: Upper bound on not-yet-finalized fees, in
            micros.
        reconciliation_uncertainty_buffer: Buffer covering unreconciled state,
            in micros.
        equity_start_of_day: Equity at the start of the trading day, in micros.
        equity_high_water_mark: The highest equity ever reached, in micros.
            All-time and ratcheting: a *trailing* drawdown trails a peak that
            never resets, so a mark rebuilt each day would forgive every
            drawdown at midnight.
        sampled_equity: The account's most recently sampled equity, in micros
            -- cash plus the marked value of open positions, the same series
            ``equity_start_of_day`` and ``equity_high_water_mark`` are points
            on. ``trailing_drawdown_limit`` measures against this rather than
            against worst-case equity (§10.4) because a drawdown is the
            distance between two readings of *one* quantity: worst-case equity
            assumes every open position resolves against us, so it falls by a
            position's full cost the moment one is opened, and a "drawdown"
            measured from the sampled peak down to it would fire on capital
            deployment rather than on loss (issue #514).
        realized_loss_today: Loss realized so far today, in micros: the
            distance from ``equity_start_of_day`` down to the day's lowest
            sample. Measured to the trough rather than to the latest reading so
            a rebound cannot un-book a breach SPEC S10.10 pauses to the next
            UTC day (issue #513).
        market_exposure: Current exposure to the single market, in micros.
        event_exposure: Current exposure to the parent event, in micros.
        bucket_exposure: Current exposure to the correlation bucket, in micros.
        total_exposure: Current total portfolio exposure, in micros.
        orders_last_hour: Orders placed in the trailing hour.
        notional_today: Notional traded so far today, in micros.
    """

    exchange_verified_available_cash: MoneyMicros
    guaranteed_terminal_value_of_positions: MoneyMicros
    pending_kernel_reservations: MoneyMicros
    unresolved_fee_upper_bounds: MoneyMicros
    reconciliation_uncertainty_buffer: MoneyMicros
    equity_start_of_day: MoneyMicros
    equity_high_water_mark: MoneyMicros
    sampled_equity: MoneyMicros
    realized_loss_today: MoneyMicros
    market_exposure: MoneyMicros
    event_exposure: MoneyMicros
    bucket_exposure: MoneyMicros
    total_exposure: MoneyMicros
    orders_last_hour: int
    notional_today: MoneyMicros


@dataclass(frozen=True, slots=True)
class MarketView:
    """The market-data view a check evaluates freshness, depth, and skew from.

    Every field is ``None`` when the datum is unavailable; the freshness, skew,
    participation, and reduce-only checks each fail closed on the ``None`` they
    depend on.

    Attributes:
        quote_snapshot_epoch_s: Epoch second the quote was snapshotted, or
            ``None``.
        forecast_epoch_s: Epoch second the forecast was produced, or ``None``.
        visible_depth: Visible order-book depth, in contract-centis, or
            ``None``.
        exchange_clock_epoch_s: The exchange's own clock, in epoch seconds, or
            ``None``.
        open_position: The current open position, in contract-centis, or
            ``None`` if flat / unknown.
        exchange_status: The exchange's trading status, or ``None`` when the
            datum is unavailable; ``exchange_status_ok`` fails closed on the
            ``None`` (an unknown status can never read as tradable). Issue #110.
        exchange_status_epoch_s: Epoch second the exchange status was observed,
            or ``None``; ``exchange_status_ok`` fails closed when it is ``None``
            or older than ``exchange_status_ttl_seconds``. Issue #110.
        jurisdiction_status: Whether the market is proven jurisdiction-eligible,
            or ``None`` when the datum is unknown, absent, or unrecognized;
            ``jurisdiction_product_eligibility`` fails closed on the ``None``
            (SPEC S6.2). Issue #340.
        product_type: The market's product shape, or ``None`` when it is absent
            or not a recognized tradable product;
            ``jurisdiction_product_eligibility`` fails closed on the ``None``
            (SPEC S1.1 invariant 2). Issue #340.

    Unlike ``exchange_status``, the two eligibility fields carry no companion
    ``_epoch_s`` and no ttl. That asymmetry is deliberate, not an oversight: an
    exchange's trading status is a live feed that goes stale, whereas a market's
    jurisdiction and product shape are near-immutable attributes re-read from the
    market object on every tick.
    """

    quote_snapshot_epoch_s: int | None
    forecast_epoch_s: int | None
    visible_depth: ContractCentis | None
    exchange_clock_epoch_s: int | None
    open_position: ContractCentis | None
    exchange_status: ExchangeTradingStatus | None
    exchange_status_epoch_s: int | None
    jurisdiction_status: JurisdictionStatus | None
    product_type: ProductType | None


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """The configured risk limits every check compares the account against.

    Percentage limits are in parts-per-million (ppm); a limit of ``1_000_000``
    ppm is 100%. Prices are in pips and monetary limits in micros.

    Attributes:
        floor: The equity floor an open must clear, in micros.
        instrument_whitelist: The set of tradable market tickers.
        micro_cap: The LIVE_MICRO total-exposure ceiling, in micros.
        min_open_price: The minimum admissible open price, in pips.
        max_open_price: The maximum admissible open price, in pips.
        max_participation_ppm: Max share of visible depth an order may take, in
            ppm.
        max_pos_market_pct_ppm: Max market exposure as a share of equity, ppm.
        max_pos_event_pct_ppm: Max event exposure as a share of equity, ppm.
        max_pos_bucket_pct_ppm: Max bucket exposure as a share of equity, ppm.
        max_pos_total_pct_ppm: Max total exposure as a share of equity, ppm.
        daily_loss_limit_pct_ppm: Daily realized-loss limit as a share of
            start-of-day equity, ppm.
        max_drawdown_pct_ppm: Max trailing drawdown from the high-water mark, as
            a share of that mark, ppm.
        max_orders_per_hour: Max orders admissible in a trailing hour.
        max_notional_per_day: Max notional admissible in a day, in micros.
        quote_ttl_seconds: Max admissible quote age, in seconds.
        forecast_ttl_seconds: Max admissible forecast age, in seconds.
        clock_skew_max_seconds: Max admissible exchange-clock skew, in seconds.
        rounding_buffer: The worst-case-cost rounding buffer, in micros.
        verification_ttl_seconds: Max admissible age of a verification snapshot,
            in seconds, before the reconciliation checks treat it as stale and
            fail closed (issue #32).
        require_human_ack_above_micros: The worst-case-cost threshold above
            which an order needs a human acknowledgement, in micros, or ``None``
            when no human-ack gate is configured (the permissive default, under
            which ``human_ack_satisfied`` always approves). Issue #34.
        exchange_status_ttl_seconds: Max admissible age of the exchange-status
            feed, in seconds, before ``exchange_status_ok`` treats it as stale
            and fails closed (issue #110).
        pipeline_heartbeat_ttl_seconds: Max admissible age of the pipeline
            heartbeat, in seconds, before ``pipeline_heartbeat_ok`` treats it as
            stale and fails closed (issue #110).
    """

    floor: MoneyMicros
    instrument_whitelist: frozenset[str]
    micro_cap: MoneyMicros
    min_open_price: PricePips
    max_open_price: PricePips
    max_participation_ppm: int
    max_pos_market_pct_ppm: int
    max_pos_event_pct_ppm: int
    max_pos_bucket_pct_ppm: int
    max_pos_total_pct_ppm: int
    daily_loss_limit_pct_ppm: int
    max_drawdown_pct_ppm: int
    max_orders_per_hour: int
    max_notional_per_day: MoneyMicros
    quote_ttl_seconds: int
    forecast_ttl_seconds: int
    clock_skew_max_seconds: int
    rounding_buffer: MoneyMicros
    verification_ttl_seconds: int
    require_human_ack_above_micros: MoneyMicros | None
    exchange_status_ttl_seconds: int
    pipeline_heartbeat_ttl_seconds: int


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """The full, immutable context a pre-trade check reads (SPEC S10).

    Bundles the operating mode with the four value objects a check needs to
    reach a verdict. Frozen and slotted, so a check can be handed the whole
    context yet can never mutate the state it is judging.

    Attributes:
        mode: The current operating mode.
        limits: The configured risk limits.
        account: The current account-state snapshot.
        market: The current market-data view.
        fees: The worst-case fee upper bounds.
        now_epoch_s: The kernel's current wall clock, in epoch seconds.
        used_intent_ids: Every intent id the reservation ledger has ever seen,
            for the ``approval_token_uniqueness`` check. Required with no
            production default: a forgotten wiring must fail loudly, never open.
        used_idempotency_keys: Every idempotency key the reservation ledger has
            ever seen, for the ``idempotency_key_uniqueness`` check. Required
            with no production default, for the same fail-loud reason.
        verification: The latest read-only exchange-verification snapshot, or
            ``None`` when no cycle has run yet, for the reconciliation checks
            (issue #32). Required with no production default: a forgotten wiring
            must fail loudly (the checks fail closed on ``None``), never open.
        acknowledged_intent_ids: Every intent id with a granted human
            acknowledgement, for the ``human_ack_satisfied`` check (issue #34).
            Required with no production default: a forgotten wiring must fail
            loudly, never open.
        pipeline_heartbeat_epoch_s: The last pipeline liveness beat, in epoch
            seconds, or ``None``, for the ``pipeline_heartbeat_ok`` check
            (issue #110). Required with no production default so a forgotten
            wiring fails loudly (the check fails closed on ``None``), never open
            -- the SPEC threat T5 dead-man's switch.
    """

    mode: Mode
    limits: RiskLimits
    account: AccountState
    market: MarketView
    fees: FeeBounds
    now_epoch_s: int
    used_intent_ids: frozenset[str]
    used_idempotency_keys: frozenset[str]
    verification: VerificationSnapshot | None
    acknowledged_intent_ids: frozenset[str]
    pipeline_heartbeat_epoch_s: int | None
