"""Pre-trade veto checks for the Risk Kernel (SPEC S10.3).

This module ships the 24 SPEC S10.3 pre-trade checks. Issues #30, #31, #32,
#34, #110, and #340 give all 24 real logic -- instrument whitelist, mode/ceiling,
the floor invariant, balance/position/open-order reconciliation (#32), fee-bound
presence, concentration, daily loss, trailing drawdown, velocity, quote/forecast
freshness, price band, participation cap, human-ack satisfaction (#34),
approval-token and idempotency-key uniqueness (#31), clock skew, exchange-status
and pipeline-heartbeat liveness (#110), reduce-only provability, and
jurisdiction/product eligibility (#340) -- each reading a full
:class:`EvaluationContext`. No check is a stub: every one of the 24 can both
approve and veto on evidence.

Every check is a small, pure callable taking ``(intent, context)`` and
returning a :class:`CheckResult`; :func:`evaluate_intent` runs the whole
sequence fail-closed: a check that *raises* is converted into a veto reason
(``"{name}: error: {exc}"``) rather than propagating, and the checks after it
still run. The 24-check order is pinned by :data:`_SPEC_10_3_CHECK_NAMES`, and
:data:`DEFAULT_CHECKS` is assembled from a name-to-check table so that order can
never drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from windbreak.numeric import RoundingDirection, divide
from windbreak.riskkernel.context import (
    ExchangeTradingStatus,
    JurisdictionStatus,
    ProductType,
)
from windbreak.riskkernel.floor import worst_case_cost, worst_case_equity
from windbreak.riskkernel.modes import Mode

if TYPE_CHECKING:
    from windbreak.numeric.types import (
        ContractCentis,
        MoneyMicros,
        PricePips,
        ProbabilityPpm,
    )
    from windbreak.riskkernel.context import AccountState, EvaluationContext


#: Denominator taking a parts-per-million (ppm) share back to a whole fraction.
_PPM_DENOMINATOR = 1_000_000

#: The trade actions treated as *opening* a new position (notional at risk).
_OPENING_ACTIONS: frozenset[str] = frozenset({"buy"})

#: The trade actions treated as *closing* an existing position (proceeds, not
#: notional, so the floor invariant charges only fees + buffer).
_CLOSING_ACTIONS: frozenset[str] = frozenset({"sell_to_close"})

#: The reason an intent whose action is neither an open nor a provable close is
#: vetoed by every action-branching check -- the kernel cannot prove its risk.
_UNPROVABLE_REASON = "unprovable"

#: The trading modes in which unknown balance semantics block live trading: a
#: verified balance cannot be trusted for a real order while any
#: ``BalanceSemantics`` field is ``UNKNOWN`` (issue #32).
_LIVE_TRADING_MODES: frozenset[Mode] = frozenset({Mode.LIVE_MICRO, Mode.LIVE})

#: The distinct veto reason ``balance_reconciliation`` raises when live trading
#: is attempted while balance semantics are not fully known.
_SEMANTICS_UNKNOWN_REASON = "balance semantics not fully known in live mode"

#: The veto reason ``human_ack_satisfied`` raises for an unacknowledged
#: over-threshold live order. Public so the ack-flow coordinator
#: (:mod:`windbreak.riskkernel.ack_flow`) can key its HELD-vs-vetoed decision off
#: this exact string without duplicating the literal.
HUMAN_ACK_REQUIRED_REASON = "human acknowledgement required"

#: The exchange trading statuses in which an order may be placed. Only
#: :attr:`ExchangeTradingStatus.OPEN` is tradable; ``PAUSED`` and ``CLOSED`` (and
#: an unknown ``None``) all fail closed via ``exchange_status_ok`` (issue #110).
_TRADABLE_EXCHANGE_STATUSES: frozenset[ExchangeTradingStatus] = frozenset(
    {ExchangeTradingStatus.OPEN}
)

#: The product types this system may ever trade. Fully collateralized binaries
#: only -- SPEC S1.1 invariant 2 and LEGAL_AND_COMPLIANCE.md make this a hard,
#: non-configurable constraint, so it lives here rather than in ``RiskLimits``:
#: no operator config edit can widen it. An unrecognized or absent product is
#: ``None`` on :class:`MarketView` and fails closed (issue #340).
_TRADABLE_PRODUCT_TYPES: frozenset[ProductType] = frozenset(
    {ProductType.FULLY_COLLATERALIZED_BINARY}
)


class _UnprovableCostError(Exception):
    """Raised when an order's worst-case cost cannot be proven (fail-closed).

    :func:`_order_cost` raises this when either fee upper bound is absent, so a
    cost that cannot be established fails *closed* rather than being silently
    assumed. Every cost-consuming check catches it and vetoes as
    :data:`_UNPROVABLE_REASON`; as defense in depth, :func:`evaluate_intent`'s
    fail-closed wrapper would also convert an uncaught instance into a veto.

    This deliberately replaces a former ``assert`` narrowing the two optional
    fee bounds: an ``assert`` is stripped under ``python -O``, which on the
    money-critical cost path would let a missing bound flow into the arithmetic
    as though present -- an unacceptable silent-failure risk. A raised
    exception cannot be optimized away.
    """


@dataclass(frozen=True)
class OrderIntent:
    """A normalized order intent submitted to the Risk Kernel for veto review.

    The dataclass is ``frozen`` (immutable): assigning to any attribute -- a
    declared field or an undeclared name -- raises
    :class:`dataclasses.FrozenInstanceError`, itself an :class:`AttributeError`
    subclass, so no attribute can ever be added or mutated after construction.
    ``slots`` is deliberately not enabled here: on CPython, combining
    ``frozen`` with ``slots`` routes an undeclared-attribute assignment through
    a stale ``super()`` cell (CPython issue #91126) that raises ``TypeError``
    instead of the immutability error, so plain ``frozen`` gives the stronger,
    correct rejection.

    Every numeric field is a :mod:`windbreak.numeric` scaled-integer type, never
    a float (SPEC S6.1); the identity fields are plain strings.

    Attributes:
        intent_id: The intent's unique identifier.
        market_ticker: The exchange ticker the intent targets.
        outcome: The market outcome the intent trades (e.g. ``"yes"``).
        action: The trade action (e.g. ``"buy"``).
        price: The limit price, in pips.
        size: The contract count, in centis.
        max_notional: The notional cap, in money-micros.
        implied_probability: The forecast-implied probability, in ppm.
        idempotency_key: The caller-supplied idempotency key, read by the
            ``idempotency_key_uniqueness`` check to reject a duplicate submission
            (issue #31).
    """

    intent_id: str
    market_ticker: str
    outcome: str
    action: str
    price: PricePips
    size: ContractCentis
    max_notional: MoneyMicros
    implied_probability: ProbabilityPpm
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of a single pre-trade check.

    Attributes:
        vetoed: Whether the check vetoes the intent.
        reason: A short human-readable reason for the verdict.
    """

    vetoed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class Decision:
    """The Risk Kernel's aggregate verdict over an intent.

    Attributes:
        vetoed: Whether any check vetoed the intent.
        reasons: One reason per vetoing (or raising) check, in evaluation
            order.
        ledgered: Whether this decision has been recorded to the ledger. Pure
            :func:`evaluate_intent` leaves this ``False``; the process-level
            kernel sets it ``True`` once the veto event is persisted.
    """

    vetoed: bool
    reasons: tuple[str, ...]
    ledgered: bool = False


class Check(Protocol):
    """A pre-trade check: a named callable returning a :class:`CheckResult`.

    ``name`` is a read-only property so both a frozen-dataclass field (like
    :class:`_ReconciliationCheck`) and a plain class attribute satisfy the
    protocol; a bare ``name: str`` would demand a *settable* attribute that a
    frozen check cannot provide.
    """

    @property
    def name(self) -> str:
        """The SPEC S10.3 check name."""

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Evaluate ``intent`` against ``context`` and return this verdict.

        Args:
            intent: The order intent to evaluate.
            context: The full evaluation context.

        Returns:
            The check's :class:`CheckResult`.
        """
        ...


# --- Shared helpers ---------------------------------------------------------------


def _approve(reason: str = "ok") -> CheckResult:
    """Return a non-vetoing :class:`CheckResult`.

    Args:
        reason: An optional human-readable note; unused by callers that pass.

    Returns:
        A :class:`CheckResult` that does not veto.
    """
    return CheckResult(vetoed=False, reason=reason)


def _veto(reason: str) -> CheckResult:
    """Return a vetoing :class:`CheckResult` carrying ``reason``.

    Args:
        reason: The human-readable reason for the veto.

    Returns:
        A vetoing :class:`CheckResult`.
    """
    return CheckResult(vetoed=True, reason=reason)


def _ppm_of(base: int, ppm: int) -> int:
    """Return ``base * ppm / 1_000_000``, rounded down (floor).

    The share is floored via :func:`~windbreak.numeric.divide` with
    :data:`RoundingDirection.UNDERSTATE_EQUITY`, so a dropped remainder can only
    ever *shrink* a permissive threshold -- never inflate one -- keeping every
    ppm-scaled limit conservative.

    Args:
        base: The scaled-integer base value (e.g. equity or depth ``.value``).
        ppm: The share, in parts per million.

    Returns:
        The floored ppm share of ``base``.
    """
    return divide(
        base * ppm, _PPM_DENOMINATOR, rounding=RoundingDirection.UNDERSTATE_EQUITY
    )


def _equity_of(account: AccountState) -> MoneyMicros:
    """Compute the worst-case equity from an :class:`AccountState`.

    Args:
        account: The account-state snapshot to read the five equity terms from.

    Returns:
        The worst-case equity, in micros.
    """
    return worst_case_equity(
        exchange_verified_available_cash=account.exchange_verified_available_cash,
        guaranteed_terminal_value_of_positions=(
            account.guaranteed_terminal_value_of_positions
        ),
        pending_kernel_reservations=account.pending_kernel_reservations,
        unresolved_fee_upper_bounds=account.unresolved_fee_upper_bounds,
        reconciliation_uncertainty_buffer=account.reconciliation_uncertainty_buffer,
    )


def _is_derisking_close(intent: OrderIntent, context: EvaluationContext) -> bool:
    """Return whether ``intent`` is a *provably* de-risking close (#100).

    A close is provably de-risking iff it is a closing action against a known
    open position that it cannot overshoot -- the exact invariant
    :meth:`_ReduceOnlyProvable.__call__`'s close branch enforces: the action is
    in :data:`_CLOSING_ACTIONS`, an open position is on record
    (``context.market.open_position is not None``), and the order size does not
    exceed it (``intent.size <= open_position``). One shared definition,
    consumed by four checks (:class:`_ReduceOnlyProvable`,
    :class:`_ConcentrationLimits`, :class:`_ModePermissionCeiling`, and
    :class:`_VelocityLimits`), so the reduce-only test can never drift between
    them.

    SPEC S9.8 permits a ``SELL_TO_CLOSE`` only from the kill path, drawdown
    de-risking, or an operator command; SPEC S10.4 requires that "for closes,
    worst-case cost must be provably non-increasing". A close that satisfies
    this predicate can therefore only ever *reduce* exposure, so the three
    aggregate-cap checks exempt it: the safety valve that reduces exposure must
    never be blocked by the very cap it is reducing.

    This is defense in depth, not a shortcut. These caps run *before*
    ``reduce_only_provable`` in the SPEC S10.3 sequence, so each must establish
    the de-risking property itself rather than deferring to that later check --
    the ``sell_to_close`` label alone is never trusted; only a size provably
    within a known open position earns the exemption.

    Args:
        intent: The order intent supplying the action and size.
        context: The evaluation context supplying the open position on record.

    Returns:
        ``True`` iff the intent is a provable de-risking close, else ``False``.
    """
    open_position = context.market.open_position
    return (
        intent.action in _CLOSING_ACTIONS
        and open_position is not None
        and intent.size <= open_position
    )


def _order_cost(intent: OrderIntent, context: EvaluationContext) -> MoneyMicros:
    """Compute an order's full worst-case cost, requiring present fee bounds.

    This always charges the full worst-case notional (SPEC S10.4 opening-buy
    formula), regardless of ``intent.action``. On the *non-exempt* path the
    aggregate-cap checks (:class:`_ModePermissionCeiling` LIVE_MICRO cap,
    :class:`_ConcentrationLimits`, :class:`_VelocityLimits`) use it deliberately:
    over-charging a ``SELL_TO_CLOSE`` that is *not* provably de-risking against a
    cap can only bias toward a veto, never toward approving fresh risk, which is
    the conservative side for a headroom check. A provably de-risking close is
    instead exempted outright (:func:`_is_derisking_close`) before its cost is
    ever computed, since such a close can only reduce exposure. Both the floor
    invariant and those three caps thus special-case a provable close --
    :class:`_FloorInvariant` by charging only fees plus buffer (S10.4), the caps
    by full exemption -- so a risk-reducing exit is never wrongly blocked.

    Both fee upper bounds must be present for the cost to be provable. When
    either is ``None`` the cost is indeterminate, so this raises
    :class:`_UnprovableCostError` -- an explicit fail-closed guard rather than
    an ``assert`` (which ``python -O`` strips, silently letting a missing bound
    reach the arithmetic). Every cost-consuming check catches the error and
    vetoes; :func:`evaluate_intent` would also convert an uncaught instance
    into a veto.

    Args:
        intent: The order intent supplying price and size.
        context: The evaluation context supplying fee bounds and the buffer.

    Returns:
        The worst-case cost, in micros.

    Raises:
        _UnprovableCostError: If either fee upper bound is ``None``, so an
            unprovable cost fails closed instead of being silently assumed.
    """
    trading = context.fees.max_trading_fee
    settlement = context.fees.max_settlement_fee
    if trading is None or settlement is None:
        raise _UnprovableCostError
    return worst_case_cost(
        intent.price,
        intent.size,
        max_trading_fee=trading,
        max_settlement_fee=settlement,
        rounding_buffer=context.limits.rounding_buffer,
    )


def _is_stale(timestamp: int | None, now: int, ttl_seconds: int) -> bool:
    """Return whether a timestamp is missing, in the future, or past its ttl.

    Args:
        timestamp: The datum's epoch second, or ``None`` if unavailable.
        now: The current epoch second.
        ttl_seconds: The maximum admissible age, in seconds.

    Returns:
        ``True`` if the timestamp is ``None``, later than ``now``, or older
        than ``ttl_seconds``.
    """
    if timestamp is None or timestamp > now:
        return True
    return now - timestamp > ttl_seconds


# --- The 24 real checks (SPEC S10.3) ---------------------------------------------


class _InstrumentWhitelist:
    """Veto any intent whose market ticker is not on the whitelist."""

    name = "instrument_whitelist"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff the intent's ticker is in the configured whitelist.

        Args:
            intent: The order intent to evaluate.
            context: The evaluation context supplying the whitelist.

        Returns:
            A vetoing result if the ticker is absent, else an approval.
        """
        if intent.market_ticker in context.limits.instrument_whitelist:
            return _approve()
        return _veto(f"ticker {intent.market_ticker} not on whitelist")


class _JurisdictionProductEligibility:
    """Veto unless the market is provably jurisdiction- and product-eligible (#340)."""

    name = "jurisdiction_product_eligibility"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff the market is proven eligible and a tradable product.

        Absence is checked *first*: an unknown, missing, or unrecognized
        jurisdiction arrives as ``None`` and can never be read as eligible, so an
        exchange that publishes no eligibility signal fails closed exactly like a
        missing quote (SPEC S1.1 invariant 1). Only once eligibility is proven is
        the product consulted, and only a fully collateralized binary
        (:data:`_TRADABLE_PRODUCT_TYPES`) approves -- SPEC S6.2 forbids a live
        order on a non-eligible market, and SPEC S1.1 invariant 2 forbids any
        other product shape.

        This check deliberately does not *alert* on an unknown jurisdiction.
        Checks are pure ``(intent, context) -> CheckResult`` callables with no
        emitter, and the SPEC S6.2 alert duty is already discharged upstream at
        the preflight and verification layers; raising it here would break
        :func:`evaluate_intent`'s purity for all 24 checks and triple-alert.

        Args:
            intent: The order intent (unused -- eligibility is a property of the
                market, not of the order).
            context: The evaluation context supplying the market view.

        Returns:
            A veto naming which datum was unprovable or disqualifying, else an
            approval.
        """
        del intent
        if context.market.jurisdiction_status is None:
            return _veto("jurisdiction eligibility unknown or missing")
        if context.market.jurisdiction_status is not JurisdictionStatus.ELIGIBLE:
            return _veto("market not jurisdiction-eligible")
        if context.market.product_type not in _TRADABLE_PRODUCT_TYPES:
            return _veto("product not provably a fully collateralized binary")
        return _approve()


class _ModePermissionCeiling:
    """Veto trading in a non-trading mode, or above the LIVE_MICRO cap."""

    name = "mode_permission_ceiling"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff the mode may trade and the LIVE_MICRO cap is respected.

        The mode gate is absolute and runs first: a non-trading mode vetoes as
        ``"mode ... may not trade"`` even for a provable de-risking close, so the
        exposure safety valve can never become a bypass for the trading-mode
        gate. Only *inside* the LIVE_MICRO branch, and only for the exposure-cap
        arithmetic, is a provable de-risking close (:func:`_is_derisking_close`)
        exempt: it can only reduce exposure, so it approves regardless of the
        micro-cap term and without ever computing its cost (#100).

        Args:
            intent: The order intent to evaluate.
            context: The evaluation context supplying mode, exposure, and cap.

        Returns:
            A vetoing result if the mode may not trade, the LIVE_MICRO cost is
            unprovable (a missing fee bound vetoes as ``"unprovable"``), or the
            cap is exceeded; an approval for a provable de-risking close in
            LIVE_MICRO or for any order in a permitted non-LIVE_MICRO mode; else
            an approval when the cap is respected.
        """
        if context.mode not in {Mode.PAPER, Mode.LIVE_MICRO, Mode.LIVE}:
            return _veto(f"mode {context.mode.name} may not trade")
        if context.mode is not Mode.LIVE_MICRO:
            return _approve()
        if _is_derisking_close(intent, context):
            return _approve()
        try:
            cost = _order_cost(intent, context)
        except _UnprovableCostError:
            return _veto(_UNPROVABLE_REASON)
        # The ceiling must count capital already reserved against other in-flight
        # approvals (``pending_kernel_reservations``, stamped from ledger truth by
        # ``ApprovalPipeline._effective_context`` inside the ledger lock), not just
        # settled exposure: two intents each under the cap alone can otherwise
        # jointly breach it. All-integer ``MoneyMicros`` math (SPEC S6.1).
        projected_exposure = (
            context.account.total_exposure
            + context.account.pending_kernel_reservations
            + cost
        )
        if projected_exposure > context.limits.micro_cap:
            return _veto("live-micro exposure ceiling exceeded")
        return _approve()


class _FloorInvariant:
    """Veto an intent whose worst-case equity would fall below the floor."""

    name = "floor_invariant"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff worst-case equity less cost still clears the floor.

        A missing fee bound (either kind) or an action that is neither an open
        nor a provable close makes the cost indeterminate and vetoes as
        ``"unprovable"``. Opens are charged the full worst-case cost (notional +
        fees + buffer); provable closes are charged fees + buffer only, since a
        close realizes proceeds rather than committing notional.

        Args:
            intent: The order intent to evaluate.
            context: The evaluation context supplying account, fees, and floor.

        Returns:
            An approval when ``equity - cost >= floor`` (equality approves), a
            veto when it does not, or a ``"unprovable"`` veto when the cost is
            indeterminate.
        """
        trading = context.fees.max_trading_fee
        settlement = context.fees.max_settlement_fee
        if trading is None or settlement is None:
            return _veto(_UNPROVABLE_REASON)
        cost = self._cost(intent, context, trading, settlement)
        if cost is None:
            return _veto(_UNPROVABLE_REASON)
        equity = _equity_of(context.account)
        if (equity - cost) >= context.limits.floor:
            return _approve()
        return _veto("worst-case equity below floor")

    def _cost(
        self,
        intent: OrderIntent,
        context: EvaluationContext,
        trading: MoneyMicros,
        settlement: MoneyMicros,
    ) -> MoneyMicros | None:
        """Return the floor-invariant cost, or ``None`` for an unknown action.

        Args:
            intent: The order intent to evaluate.
            context: The evaluation context supplying the rounding buffer.
            trading: The (present) worst-case trading fee, in micros.
            settlement: The (present) worst-case settlement fee, in micros.

        Returns:
            The full worst-case cost for an open, the fees-plus-buffer cost for
            a provable close, or ``None`` if the action is neither.
        """
        if intent.action in _OPENING_ACTIONS:
            return worst_case_cost(
                intent.price,
                intent.size,
                max_trading_fee=trading,
                max_settlement_fee=settlement,
                rounding_buffer=context.limits.rounding_buffer,
            )
        if intent.action in _CLOSING_ACTIONS:
            return trading + settlement + context.limits.rounding_buffer
        return None


class _FeeUpperBoundPresent:
    """Veto when no trading-fee upper bound is provable."""

    name = "fee_upper_bound_present"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff a trading-fee upper bound is present.

        Args:
            intent: The order intent (unused).
            context: The evaluation context supplying the fee bounds.

        Returns:
            A vetoing result if the trading-fee bound is ``None``, else an
            approval.
        """
        del intent
        if context.fees.max_trading_fee is None:
            return _veto("no trading-fee upper bound")
        return _approve()


class _SettlementFeeUpperBound:
    """Veto when no settlement-fee upper bound is provable."""

    name = "settlement_fee_upper_bound"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff a settlement-fee upper bound is present.

        Args:
            intent: The order intent (unused).
            context: The evaluation context supplying the fee bounds.

        Returns:
            A vetoing result if the settlement-fee bound is ``None``, else an
            approval.
        """
        del intent
        if context.fees.max_settlement_fee is None:
            return _veto("no settlement-fee upper bound")
        return _approve()


class _ConcentrationLimits:
    """Veto when any exposure dimension plus cost exceeds its equity share."""

    name = "concentration_limits"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff every exposure dimension stays within its ppm cap.

        A provable de-risking close (:func:`_is_derisking_close`) is fully
        exempt and approves as the very first step -- regardless of how far any
        exposure dimension is over its cap, and even when a fee bound is missing,
        because the exemption short-circuits before :func:`_order_cost` is ever
        called, so an exempt close never consumes cost. Such a close can only
        reduce exposure, so the cap it is reducing must never veto it (#100).

        Args:
            intent: The order intent supplying price and size.
            context: The evaluation context supplying account and caps.

        Returns:
            An approval for a provable de-risking close; otherwise a vetoing
            result if the cost is unprovable (a missing fee bound vetoes as
            ``"unprovable"``) or if any of market/event/bucket/total exposure
            plus cost exceeds its floored share of worst-case equity, else
            approval.
        """
        if _is_derisking_close(intent, context):
            return _approve()
        try:
            cost = _order_cost(intent, context)
        except _UnprovableCostError:
            return _veto(_UNPROVABLE_REASON)
        equity = _equity_of(context.account).value
        account = context.account
        limits = context.limits
        dimensions = (
            (account.market_exposure, limits.max_pos_market_pct_ppm),
            (account.event_exposure, limits.max_pos_event_pct_ppm),
            (account.bucket_exposure, limits.max_pos_bucket_pct_ppm),
            (account.total_exposure, limits.max_pos_total_pct_ppm),
        )
        for exposure, cap_ppm in dimensions:
            if exposure.value + cost.value > _ppm_of(equity, cap_ppm):
                return _veto("concentration limit exceeded")
        return _approve()


class _DailyLossLimit:
    """Veto once today's realized loss reaches its equity-relative limit."""

    name = "daily_loss_limit"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff realized loss is strictly below the daily limit.

        Args:
            intent: The order intent (unused).
            context: The evaluation context supplying loss and the limit.

        Returns:
            A vetoing result once realized loss reaches (``>=``) the floored
            ppm share of start-of-day equity, else an approval.
        """
        del intent
        account = context.account
        threshold = _ppm_of(
            account.equity_start_of_day.value, context.limits.daily_loss_limit_pct_ppm
        )
        if account.realized_loss_today.value >= threshold:
            return _veto("daily loss limit reached")
        return _approve()


class _TrailingDrawdownLimit:
    """Veto once drawdown from the high-water mark reaches its limit.

    Both terms are points on the ledger's own equity series
    (:class:`~windbreak.riskkernel.context.AccountState.sampled_equity` and the
    running maximum over it), because a drawdown is the distance between two
    readings of *one* quantity. It is deliberately not measured against
    worst-case equity: that figure assumes every open position resolves against
    us, so it falls by a position's full cost the instant one is opened, and a
    drawdown taken from the sampled peak down to it would report a 12.5%
    "drawdown" for an account that had merely deployed 12.5% of its equity --
    vetoing on deployment rather than on loss, and pre-empting the configured
    concentration caps at a fraction of their reach (issue #514).
    """

    name = "trailing_drawdown_limit"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff drawdown is strictly below the trailing limit.

        Args:
            intent: The order intent (unused).
            context: The evaluation context supplying the mark, the current
                sampled equity, and the limit.

        Returns:
            A vetoing result once ``high_water_mark - sampled_equity`` reaches
            (``>=``) the floored ppm share of the mark, else approval. An
            account with no ledgered sample carries both terms as zero, so
            ``0 >= 0`` vetoes until the first sample lands.
        """
        del intent
        mark = context.account.equity_high_water_mark.value
        drawdown = mark - context.account.sampled_equity.value
        threshold = _ppm_of(mark, context.limits.max_drawdown_pct_ppm)
        if drawdown >= threshold:
            return _veto("trailing drawdown limit reached")
        return _approve()


class _VelocityLimits:
    """Veto when an order would breach the hourly or daily velocity caps."""

    name = "velocity_limits"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff both the hourly-order and daily-notional caps hold.

        The hourly-order-count cap is checked first and needs no cost: it is
        runaway-order protection and still applies to a de-risking close, which
        can flood the exchange with cancels/replacements just as an open can. A
        provable de-risking close (:func:`_is_derisking_close`) is then exempt
        from the daily-notional term only: it can only reduce exposure, so the
        notional budget it would otherwise consume must not veto it. That
        exemption short-circuits before :func:`_order_cost` is ever called, so an
        exempt close with hourly headroom approves even when a fee bound is
        missing -- cost is never consumed on that path (#100).

        Args:
            intent: The order intent supplying price and size.
            context: The evaluation context supplying counters and caps.

        Returns:
            A vetoing result if this order would exceed the hourly order cap
            (applies to closes too); an approval for a provable de-risking close
            past the hourly gate; otherwise a vetoing result if the cost is
            unprovable (a missing fee bound vetoes as ``"unprovable"``) or if the
            daily notional cap would be exceeded, else an approval.
        """
        account = context.account
        limits = context.limits
        if account.orders_last_hour + 1 > limits.max_orders_per_hour:
            return _veto("hourly order cap exceeded")
        if _is_derisking_close(intent, context):
            return _approve()
        try:
            cost = _order_cost(intent, context)
        except _UnprovableCostError:
            return _veto(_UNPROVABLE_REASON)
        if (account.notional_today + cost) > limits.max_notional_per_day:
            return _veto("daily notional cap exceeded")
        return _approve()


class _QuoteFreshness:
    """Veto when the quote snapshot is missing, future-dated, or stale."""

    name = "quote_freshness"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff the quote is present and within its ttl.

        Args:
            intent: The order intent (unused).
            context: The evaluation context supplying the quote and its ttl.

        Returns:
            A vetoing result if the quote is missing, in the future, or older
            than its ttl, else an approval.
        """
        del intent
        if _is_stale(
            context.market.quote_snapshot_epoch_s,
            context.now_epoch_s,
            context.limits.quote_ttl_seconds,
        ):
            return _veto("quote is stale or missing")
        return _approve()


class _ForecastFreshness:
    """Veto when the forecast is missing, future-dated, or stale."""

    name = "forecast_freshness"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff the forecast is present and within its ttl.

        Args:
            intent: The order intent (unused).
            context: The evaluation context supplying the forecast and its ttl.

        Returns:
            A vetoing result if the forecast is missing, in the future, or
            older than its ttl, else an approval.
        """
        del intent
        if _is_stale(
            context.market.forecast_epoch_s,
            context.now_epoch_s,
            context.limits.forecast_ttl_seconds,
        ):
            return _veto("forecast is stale or missing")
        return _approve()


class _PriceBandCompliance:
    """Veto an open priced outside the configured band; closes are exempt."""

    name = "price_band_compliance"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff an open is within the band; provable closes pass.

        Args:
            intent: The order intent supplying action and price.
            context: The evaluation context supplying the price band.

        Returns:
            For opens, a vetoing result outside the inclusive band; for closes,
            an approval; for any other action, a veto.
        """
        if intent.action in _CLOSING_ACTIONS:
            return _approve()
        if intent.action not in _OPENING_ACTIONS:
            return _veto(_UNPROVABLE_REASON)
        limits = context.limits
        if limits.min_open_price <= intent.price <= limits.max_open_price:
            return _approve()
        return _veto("price outside open band")


class _ParticipationCapCompliance:
    """Veto when the order takes more than the permitted share of depth."""

    name = "participation_cap_compliance"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff the order size is within the participation cap.

        Args:
            intent: The order intent supplying size.
            context: The evaluation context supplying depth and the cap.

        Returns:
            A vetoing result if depth is unknown or the size exceeds the floored
            ppm share of visible depth, else an approval.
        """
        depth = context.market.visible_depth
        if depth is None:
            return _veto("visible depth unknown")
        if intent.size.value > _ppm_of(
            depth.value, context.limits.max_participation_ppm
        ):
            return _veto("participation cap exceeded")
        return _approve()


class _ClockSkewLimit:
    """Veto when the exchange clock skew exceeds the configured maximum."""

    name = "clock_skew_limit"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff the exchange clock is within the skew limit.

        Args:
            intent: The order intent (unused).
            context: The evaluation context supplying the clock and the limit.

        Returns:
            A vetoing result if the exchange clock is unknown or skewed (in
            either direction) beyond the limit, else an approval.
        """
        del intent
        exchange_clock = context.market.exchange_clock_epoch_s
        if exchange_clock is None:
            return _veto("exchange clock unknown")
        skew = abs(context.now_epoch_s - exchange_clock)
        if skew > context.limits.clock_skew_max_seconds:
            return _veto("clock skew exceeds limit")
        return _approve()


class _ExchangeStatusOk:
    """Veto when the exchange status is stale, unknown, or not open (#110)."""

    name = "exchange_status_ok"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff a fresh exchange status reports the exchange OPEN.

        Staleness is checked *first*, before the status value, so a stale or
        unknown (``None``) status can never be read as tradable: an absent
        status feed fails closed exactly like a missing quote. Only once the
        status is proven fresh is its value consulted, and only
        :attr:`ExchangeTradingStatus.OPEN`
        (:data:`_TRADABLE_EXCHANGE_STATUSES`) approves; ``PAUSED`` / ``CLOSED``
        veto (SPEC S7.3 / SPEC S10.3).

        Args:
            intent: The order intent (unused).
            context: The evaluation context supplying the market status, its
                epoch, the clock, and the status ttl.

        Returns:
            A ``"exchange status stale or missing"`` veto when the status is
            ``None`` or its epoch is missing, future-dated, or past its ttl; an
            ``"exchange not open for trading"`` veto for a fresh non-open
            status; else an approval.
        """
        del intent
        if context.market.exchange_status is None or _is_stale(
            context.market.exchange_status_epoch_s,
            context.now_epoch_s,
            context.limits.exchange_status_ttl_seconds,
        ):
            return _veto("exchange status stale or missing")
        if context.market.exchange_status not in _TRADABLE_EXCHANGE_STATUSES:
            return _veto("exchange not open for trading")
        return _approve()


class _PipelineHeartbeatOk:
    """Veto when the pipeline heartbeat is stale or missing (#110, threat T5)."""

    name = "pipeline_heartbeat_ok"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff the pipeline heartbeat is present and within its ttl.

        This is the SPEC S10.3 threat-T5 dead-man's switch: a missing
        (``None``), future-dated, or stale heartbeat means the pipeline can no
        longer prove it is alive, so the check fails closed rather than letting
        an order through on a silent pipeline.

        Args:
            intent: The order intent (unused).
            context: The evaluation context supplying the heartbeat epoch, the
                clock, and the heartbeat ttl.

        Returns:
            A ``"pipeline heartbeat stale or missing"`` veto when the heartbeat
            is ``None``, in the future, or older than its ttl, else an approval.
        """
        del intent
        if _is_stale(
            context.pipeline_heartbeat_epoch_s,
            context.now_epoch_s,
            context.limits.pipeline_heartbeat_ttl_seconds,
        ):
            return _veto("pipeline heartbeat stale or missing")
        return _approve()


class _HumanAckSatisfied:
    """Veto a live over-threshold order lacking a human acknowledgement (#34)."""

    name = "human_ack_satisfied"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve unless a live over-threshold order lacks an acknowledgement.

        Real capital is only at risk in the live modes, so RESEARCH/PAPER always
        approve; a ``None`` threshold means no human-ack gate is configured and
        also always approves. Otherwise the order's worst-case cost is compared
        against the threshold (inclusive): a cost at or below it approves, and a
        cost strictly above it approves only when the intent id is already
        acknowledged. A missing fee bound makes the cost unprovable and vetoes
        fail-closed as ``"unprovable"``, exactly like every cost-consuming check.

        Args:
            intent: The order intent supplying the id, price, and size.
            context: The evaluation context supplying the mode, threshold, fee
                bounds, and acknowledged-intent-id set.

        Returns:
            An approval outside the live modes or with no configured threshold; a
            ``"unprovable"`` veto when the cost is indeterminate; a
            ``"human acknowledgement required"`` veto for an unacknowledged
            over-threshold cost; else an approval.
        """
        threshold = context.limits.require_human_ack_above_micros
        if threshold is None or context.mode not in _LIVE_TRADING_MODES:
            return _approve()
        try:
            cost = _order_cost(intent, context)
        except _UnprovableCostError:
            return _veto(_UNPROVABLE_REASON)
        if cost > threshold and intent.intent_id not in context.acknowledged_intent_ids:
            return _veto(HUMAN_ACK_REQUIRED_REASON)
        return _approve()


class _ApprovalTokenUniqueness:
    """Veto an intent whose id already has an issued approval token (#31)."""

    name = "approval_token_uniqueness"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff the intent id has not been reserved/approved before.

        Args:
            intent: The order intent supplying the intent id.
            context: The evaluation context supplying the used-intent-id set.

        Returns:
            A vetoing result if the intent id is already in
            ``context.used_intent_ids``, else an approval.
        """
        if intent.intent_id in context.used_intent_ids:
            return _veto("approval token already issued for intent id")
        return _approve()


class _IdempotencyKeyUniqueness:
    """Veto an intent whose idempotency key was already used (#31)."""

    name = "idempotency_key_uniqueness"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff the idempotency key has not been used before.

        Args:
            intent: The order intent supplying the idempotency key.
            context: The evaluation context supplying the used-key set.

        Returns:
            A vetoing result if the idempotency key is already in
            ``context.used_idempotency_keys``, else an approval.
        """
        if intent.idempotency_key in context.used_idempotency_keys:
            return _veto("idempotency key already used")
        return _approve()


class _ReduceOnlyProvable:
    """Veto a close that cannot be proven to reduce the open position."""

    name = "reduce_only_provable"

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve opens; approve closes only if provably reduce-only.

        Args:
            intent: The order intent supplying action and size.
            context: The evaluation context supplying the open position.

        Returns:
            An approval for opens; for closes, an approval only when an open
            position is on record and the size does not exceed it; a veto for
            any other action.
        """
        if intent.action in _OPENING_ACTIONS:
            return _approve()
        if intent.action not in _CLOSING_ACTIONS:
            return _veto(_UNPROVABLE_REASON)
        if _is_derisking_close(intent, context):
            return _approve()
        return _veto("close is not provably reduce-only")


@dataclass(frozen=True, slots=True)
class _ReconciliationCheck:
    """A per-dimension reconciliation check over the verification snapshot (#32).

    All three reconciliation dimensions share one shape: fail closed on a
    missing or stale snapshot (reusing :func:`_is_stale`, exactly as the
    freshness and clock-skew checks do), then veto when this check's own
    per-dimension ``ok`` flag is False. The balance dimension alone additionally
    refuses live trading (LIVE_MICRO / LIVE) while balance semantics are not
    fully known; the position and open-order dimensions ignore semantics.

    Attributes:
        name: The SPEC S10.3 check name.
        ok_attr: The :class:`VerificationSnapshot` boolean field this dimension
            reads (``"balance_ok"`` / ``"position_ok"`` / ``"open_order_ok"``).
        stale_reason: The veto reason for a missing or stale snapshot.
        mismatch_reason: The veto reason when the dimension flag is False.
        gate_semantics: Whether to additionally gate live trading on fully-known
            balance semantics (only the balance dimension does).
    """

    name: str
    ok_attr: str
    stale_reason: str
    mismatch_reason: str
    gate_semantics: bool

    def __call__(self, intent: OrderIntent, context: EvaluationContext) -> CheckResult:
        """Approve iff a fresh snapshot marks this dimension reconciled.

        Args:
            intent: The order intent (unused).
            context: The evaluation context supplying the verification snapshot,
                the ttl, and the operating mode.

        Returns:
            A stale-reason veto on a missing or stale snapshot; the dimension's
            mismatch-reason veto when its ``ok`` flag is False; for the balance
            dimension, a distinct semantics veto in live modes when semantics
            are not fully known; else an approval.
        """
        del intent
        snapshot = context.verification
        if snapshot is None or _is_stale(
            snapshot.verified_at_epoch_s,
            context.now_epoch_s,
            context.limits.verification_ttl_seconds,
        ):
            return _veto(self.stale_reason)
        if not getattr(snapshot, self.ok_attr):
            return _veto(self.mismatch_reason)
        if (
            self.gate_semantics
            and not snapshot.semantics_fully_known
            and context.mode in _LIVE_TRADING_MODES
        ):
            return _veto(_SEMANTICS_UNKNOWN_REASON)
        return _approve()


#: The three issue-#32 reconciliation checks, one per verified dimension.
_RECONCILIATION_CHECKS: tuple[_ReconciliationCheck, ...] = (
    _ReconciliationCheck(
        name="balance_reconciliation",
        ok_attr="balance_ok",
        stale_reason="balance verification stale or missing",
        mismatch_reason="balance reconciliation mismatch",
        gate_semantics=True,
    ),
    _ReconciliationCheck(
        name="position_reconciliation",
        ok_attr="position_ok",
        stale_reason="position verification stale or missing",
        mismatch_reason="position reconciliation mismatch",
        gate_semantics=False,
    ),
    _ReconciliationCheck(
        name="open_order_reconciliation",
        ok_attr="open_order_ok",
        stale_reason="open-order verification stale or missing",
        mismatch_reason="open-order reconciliation mismatch",
        gate_semantics=False,
    ),
)


# --- Assembly: the pinned SPEC S10.3 sequence ------------------------------------


#: The SPEC S10.3 check names, in the exact order they must be evaluated.
_SPEC_10_3_CHECK_NAMES: tuple[str, ...] = (
    "instrument_whitelist",
    "jurisdiction_product_eligibility",
    "mode_permission_ceiling",
    "floor_invariant",
    "balance_reconciliation",
    "position_reconciliation",
    "open_order_reconciliation",
    "fee_upper_bound_present",
    "settlement_fee_upper_bound",
    "concentration_limits",
    "daily_loss_limit",
    "trailing_drawdown_limit",
    "velocity_limits",
    "quote_freshness",
    "forecast_freshness",
    "price_band_compliance",
    "participation_cap_compliance",
    "human_ack_satisfied",
    "approval_token_uniqueness",
    "idempotency_key_uniqueness",
    "clock_skew_limit",
    "exchange_status_ok",
    "pipeline_heartbeat_ok",
    "reduce_only_provable",
)

#: The 24 real checks, keyed by SPEC S10.3 name for order-independent assembly.
_REAL_CHECKS: tuple[Check, ...] = (
    _InstrumentWhitelist(),
    _JurisdictionProductEligibility(),
    _ModePermissionCeiling(),
    _FloorInvariant(),
    *_RECONCILIATION_CHECKS,
    _FeeUpperBoundPresent(),
    _SettlementFeeUpperBound(),
    _ConcentrationLimits(),
    _DailyLossLimit(),
    _TrailingDrawdownLimit(),
    _VelocityLimits(),
    _QuoteFreshness(),
    _ForecastFreshness(),
    _PriceBandCompliance(),
    _ParticipationCapCompliance(),
    _HumanAckSatisfied(),
    _ApprovalTokenUniqueness(),
    _IdempotencyKeyUniqueness(),
    _ClockSkewLimit(),
    _ExchangeStatusOk(),
    _PipelineHeartbeatOk(),
    _ReduceOnlyProvable(),
)


#: Name-to-check lookup spanning all 24 checks; the pinned
#: :data:`_SPEC_10_3_CHECK_NAMES` sequence selects and orders them.
_CHECK_BY_NAME: dict[str, Check] = {check.name: check for check in _REAL_CHECKS}

#: The default pre-trade check sequence, in exact SPEC S10.3 order.
DEFAULT_CHECKS: tuple[Check, ...] = tuple(
    _CHECK_BY_NAME[name] for name in _SPEC_10_3_CHECK_NAMES
)


def _run_check(
    check: Check, intent: OrderIntent, context: EvaluationContext
) -> str | None:
    """Run one check fail-closed, returning its veto reason or ``None``.

    A check that raises is converted into a veto reason rather than
    propagating, so one buggy check can never let an intent through or abort
    the whole evaluation.

    Args:
        check: The check to run.
        intent: The order intent to evaluate.
        context: The full evaluation context.

    Returns:
        The veto reason string if the check vetoes or raises, else ``None``.
    """
    try:
        result = check(intent, context)
    except Exception as exc:  # Fail-closed: a raising check becomes a veto.
        return f"{check.name}: error: {exc}"
    return result.reason if result.vetoed else None


def evaluate_intent(
    intent: OrderIntent,
    context: EvaluationContext,
    checks: tuple[Check, ...] = DEFAULT_CHECKS,
) -> Decision:
    """Evaluate an intent against every check, fail-closed.

    Args:
        intent: The order intent to evaluate.
        context: The full evaluation context every check reads.
        checks: The checks to run, in order. Defaults to :data:`DEFAULT_CHECKS`.

    Returns:
        A :class:`Decision` carrying one reason per vetoing (or raising) check,
        in evaluation order; ``vetoed`` is True iff any reason was collected.
    """
    reasons: list[str] = []
    for check in checks:
        reason = _run_check(check, intent, context)
        if reason is not None:
            reasons.append(reason)
    return Decision(vetoed=bool(reasons), reasons=tuple(reasons))
