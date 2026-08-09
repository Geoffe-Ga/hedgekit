"""Dispersion-scaled fractional-Kelly sizing and cap clipping (SPEC S9.5/S9.6).

Three separately testable pure stages the selector chains after every SPEC S9.3
entry condition passes, plus a small result type:

    * :func:`dispersion_scale` -- the linear ``g()`` ramp of SPEC S9.6 that
      discounts the Kelly stake as the ensemble's vote dispersion rises toward a
      configured ceiling, in ppm.
    * :func:`kelly_size` -- the binary-contract fractional-Kelly stake (SPEC
      S9.5): full-Kelly ``f* = edge / (1 - P)`` scaled by the configured Kelly
      fraction and the dispersion scale, in contract-centis. The numerator is the
      *research-cost-adjusted net* edge, so "never negative-EV-after-fees" is
      structural -- a below-floor net edge sizes to exactly zero.
    * :func:`clip_to_caps` -- clips a raw stake down through the five SPEC S9.6
      notional caps (in a pinned order), the participation cap, and the exchange
      minimum-order-size quantization, naming which cap bound.

Every size-affecting division is integer, routed through
:func:`windbreak.numeric.divide` with an explicit
:data:`~windbreak.numeric.RoundingDirection.UNDERSTATE_EQUITY` (floor): a stake
or a cap headroom is never *over*-stated, so the selector always errs toward the
smaller, safer size. There is no ``float``, no bare ``/`` or ``//`` anywhere;
this module is on ``scripts/lint_no_floats.py``'s denylist.

Two seams are deliberately fenced here, matching :mod:`windbreak.selector.entry`'s
``_SCREENER_SEAM_DETAIL`` style:

    * **Mode ceiling and live-micro cap.** SPEC S10.2's mode-gated ceilings (the
      operating-mode ceiling and the LIVE_MICRO absolute exposure cap) are
      *not* mirrored here: :class:`~windbreak.selector.types.SelectorInputs`
      carries no operating-mode field, so the selector cannot know the active
      mode. The Kernel enforces both caps authoritatively (SPEC S10.2); the
      selector's job is only to size conservatively beneath them.
    * **Exchange minimum-order size.** The whole-contract quantization here uses
      a fixed :data:`_EXCHANGE_MIN_ORDER_CENTIS` constant (one whole contract).
      A future seam will read the venue's real per-market minimum-order size
      from the normalized market metadata (SPEC S6.2); until that metadata is
      threaded into ``SelectorInputs``, one whole contract is the conservative
      floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from windbreak.numeric import ContractCentis, RoundingDirection, divide

if TYPE_CHECKING:
    from windbreak.config.schema import RiskConfig
    from windbreak.connector.models import OrderBookLevel, OrderBookSnapshot
    from windbreak.numeric import MoneyMicros
    from windbreak.selector.types import PositionReadModelInput

#: Full ppm scale: ``1_000_000`` ppm is 1.0 (100%). Both the dispersion ramp's
#: numerator and every ``* pct_ppm`` / ``* participation_ppm`` fraction divide by
#: this to return to an absolute quantity.
_PPM_ONE = 1_000_000

#: The fused Kelly denominator's scale factor. The stake numerator carries three
#: ppm fractions (net edge, Kelly fraction, dispersion scale), each ``1e-6``, so
#: ``10**12`` cancels two of them while the third pairs with the ``(1e6 - price)``
#: complement -- see :func:`kelly_size`'s derivation.
_KELLY_DENOMINATOR_SCALE = 10**12

#: Contract-centis per whole contract: a size in centis is ``contracts * 100``.
#: Converting a money headroom (micros) at a ppm price into a size multiplies by
#: this before dividing by the price in ppm.
_CENTIS_PER_CONTRACT = 100

#: The exchange minimum order size (and whole-contract quantization step), in
#: contract-centis: one whole contract. A raw stake is floored to a multiple of
#: this, and anything below it sizes to zero. See the module docstring's fenced
#: exchange-minimum seam.
_EXCHANGE_MIN_ORDER_CENTIS = 100

#: The name flagged when the final quantization takes a sub-minimum survivor to
#: zero (distinct from ``None``, which flags an unclipped survivor).
_EXCHANGE_MIN_CAP_NAME = "exchange_min_order"

#: Every size-affecting division floors (understates the size): a stake, a cap
#: headroom, or a participation limit is never over-stated.
_SIZE_ROUNDING = RoundingDirection.UNDERSTATE_EQUITY


def dispersion_scale(vote_dispersion_ppm: int, dispersion_zero_ceiling_ppm: int) -> int:
    """Return the SPEC S9.6 dispersion scale ``g(d, ceiling)``, in ppm.

    A linear ramp that discounts the Kelly stake as the ensemble's vote
    dispersion rises: ``g(0) == 1_000_000`` (no discount), ``g(ceiling) == 0``
    (fully zeroed), monotone non-increasing in ``d`` between, with a result
    always within ``[0, 1_000_000]``. The two degenerate boundaries are handled
    explicitly: a negative dispersion (never produced by a valid forecast) is
    treated as zero dispersion, and a non-positive ceiling admits no ramp -- any
    positive dispersion against it zeroes the scale, while zero dispersion keeps
    full scale.

    Args:
        vote_dispersion_ppm: The ensemble's vote dispersion ``d``, in ppm.
        dispersion_zero_ceiling_ppm: The ceiling at (or past) which the scale
            reaches zero, in ppm.

    Returns:
        The dispersion scale, in ppm, within ``[0, 1_000_000]``.
    """
    if vote_dispersion_ppm < 0:
        return _PPM_ONE
    if dispersion_zero_ceiling_ppm <= 0:
        return _PPM_ONE if vote_dispersion_ppm <= 0 else 0
    if vote_dispersion_ppm >= dispersion_zero_ceiling_ppm:
        return 0
    numerator = (dispersion_zero_ceiling_ppm - vote_dispersion_ppm) * _PPM_ONE
    return divide(numerator, dispersion_zero_ceiling_ppm, rounding=_SIZE_ROUNDING)


def kelly_size(
    *,
    net_edge_ppm: int,
    min_net_edge_ppm: int,
    executable_price_ppm: int,
    kelly_fraction_ppm: int,
    dispersion_scale_ppm: int,
    above_floor_capital_micros: MoneyMicros,
) -> ContractCentis:
    """Return the dispersion-scaled fractional-Kelly stake (SPEC S9.5), in centis.

    For a fully-collateralized binary contract priced at ``P`` (in ppm-of-$1),
    full Kelly is ``f* = edge / (1 - P)``. The staked capital is that fraction of
    ``above_floor_capital_micros``, scaled by the configured Kelly fraction and
    the dispersion scale, and the size is that stake bought at ``P``. Using the
    research-cost-adjusted *net* edge as the numerator makes "never
    negative-EV-after-fees" structural: a non-positive net edge (fractional
    Kelly stakes only on a strictly positive edge), a net edge below
    ``min_net_edge_ppm``, or a degenerate price outside ``(0, 1_000_000)`` ppm
    (a zero-pip fill has no stake, a full-dollar fill leaves no ``1 - P``
    denominator) each sizes to exactly zero, so the "never negative" guarantee
    holds for any operand signs.

    The fused stake division carries the units exactly (all integer):
    ``capital(micros) * net_edge_ppm * kelly_fraction_ppm * dispersion_scale_ppm``
    over ``(1_000_000 - executable_price_ppm) * 10**12`` yields the stake in
    micros; ``stake * 100 / executable_price_ppm`` re-expresses it as
    contract-centis. Both divisions floor (``UNDERSTATE_EQUITY``).

    Args:
        net_edge_ppm: The research-cost-adjusted net edge, in ppm.
        min_net_edge_ppm: The configured net-edge floor, in ppm; a net edge
            below it sizes to zero.
        executable_price_ppm: The executable fill price, in ppm-of-$1.
        kelly_fraction_ppm: The configured fractional-Kelly multiplier, in ppm.
        dispersion_scale_ppm: The dispersion scale ``g`` (see
            :func:`dispersion_scale`), in ppm.
        above_floor_capital_micros: The capital the stake sizes against, in
            micros.

    Returns:
        The raw fractional-Kelly size, in contract-centis (never negative).
    """
    if net_edge_ppm <= 0 or net_edge_ppm < min_net_edge_ppm:
        return ContractCentis(0)
    if executable_price_ppm <= 0 or executable_price_ppm >= _PPM_ONE:
        return ContractCentis(0)
    stake_numerator = (
        above_floor_capital_micros.value
        * net_edge_ppm
        * kelly_fraction_ppm
        * dispersion_scale_ppm
    )
    stake_denominator = (_PPM_ONE - executable_price_ppm) * _KELLY_DENOMINATOR_SCALE
    stake_micros = divide(stake_numerator, stake_denominator, rounding=_SIZE_ROUNDING)
    size_centis = divide(
        stake_micros * _CENTIS_PER_CONTRACT,
        executable_price_ppm,
        rounding=_SIZE_ROUNDING,
    )
    return ContractCentis(size_centis)


@dataclass(frozen=True, slots=True)
class CapClipResult:
    """The clipped size and the cap (if any) that bound it (SPEC S9.6).

    Attributes:
        size: The final, whole-contract-quantized size, in contract-centis
            (zero when a cap or the exchange minimum drove it below one lot).
        binding_cap: The pinned-order-first name of the cap whose limit equalled
            the clipped minimum (kept even when that cap drove the size to zero,
            so a saturated per-bucket cap stays named), ``"exchange_min_order"``
            when no cap bound but the whole-lot quantization zeroed a naturally
            sub-lot raw size, or ``None`` when the raw size survived every cap
            unclipped (only the routine whole-lot flooring applied).
    """

    size: ContractCentis
    binding_cap: str | None


def _pct_ceiling_micros(equity_micros: int, pct_ppm: int) -> int:
    """Return a percentage-of-equity concentration ceiling, in micros.

    Args:
        equity_micros: Total account equity, in micros.
        pct_ppm: The ceiling as a share of equity, in ppm.

    Returns:
        The ceiling ``equity * pct`` floored to micros (``UNDERSTATE_EQUITY``).
    """
    return divide(equity_micros * pct_ppm, _PPM_ONE, rounding=_SIZE_ROUNDING)


def _headroom_cap_centis(
    ceiling_micros: int, used_micros: int, executable_price_ppm: int
) -> int:
    """Convert a money ceiling and its used amount into a cap size, in centis.

    Args:
        ceiling_micros: The cap's money ceiling, in micros.
        used_micros: The amount already used against the ceiling, in micros.
        executable_price_ppm: The executable fill price, in ppm-of-$1 (positive).

    Returns:
        The headroom ``max(0, ceiling - used)`` re-expressed as a size in
        contract-centis, floored (``UNDERSTATE_EQUITY``).
    """
    headroom_micros = max(0, ceiling_micros - used_micros)
    return divide(
        headroom_micros * _CENTIS_PER_CONTRACT,
        executable_price_ppm,
        rounding=_SIZE_ROUNDING,
    )


def _notional_cap_limits(
    positions: PositionReadModelInput,
    risk: RiskConfig,
    executable_price_ppm: int,
    *,
    bucket_cap_name: str,
) -> list[tuple[str, int]]:
    """Return the five SPEC S9.6 notional-cap ``(name, limit)`` pairs, in order.

    Args:
        positions: The account capital/exposure figures the caps read.
        risk: The risk configuration supplying the cap percentages and limits.
        executable_price_ppm: The executable fill price, in ppm-of-$1.
        bucket_cap_name: The name the per-bucket cap is reported under. Defaults
            to the bare ``"per_bucket"`` at the public boundary; ``select()``
            passes ``"per_bucket:<bucket-id>"`` so the pinned sizing reason names
            the specific correlation bucket that bound (SPEC S9.9).

    Returns:
        The five caps in the pinned order ``per_market``, ``per_event``,
        ``per_bucket``, ``total_deployed``, ``daily_notional``; each ``limit`` is
        a size in contract-centis.
    """
    equity = positions.equity_micros.value

    def _pct(pct_ppm: int, exposure_micros: int) -> int:
        ceiling = _pct_ceiling_micros(equity, pct_ppm)
        return _headroom_cap_centis(ceiling, exposure_micros, executable_price_ppm)

    def _absolute(ceiling_micros: int, used_micros: int) -> int:
        return _headroom_cap_centis(ceiling_micros, used_micros, executable_price_ppm)

    market = _pct(risk.max_pos_market_pct_ppm, positions.market_exposure.value)
    event = _pct(risk.max_pos_event_pct_ppm, positions.event_exposure.value)
    bucket = _pct(risk.max_pos_bucket_pct_ppm, positions.bucket_exposure.value)
    total = _absolute(
        positions.total_deploy_cap_micros.value, positions.total_exposure.value
    )
    daily = _absolute(risk.max_notional_per_day_micros, positions.notional_today.value)
    return [
        ("per_market", market),
        ("per_event", event),
        (bucket_cap_name, bucket),
        ("total_deployed", total),
        ("daily_notional", daily),
    ]


def _floor_lot(size_centis: int) -> int:
    """Floor a size to a whole multiple of the exchange minimum order size.

    Args:
        size_centis: A size, in contract-centis (non-negative).

    Returns:
        The largest multiple of :data:`_EXCHANGE_MIN_ORDER_CENTIS` not exceeding
        ``size_centis``.
    """
    return size_centis - (size_centis % _EXCHANGE_MIN_ORDER_CENTIS)


def _depth_through_fill(yes_asks: tuple[OrderBookLevel, ...], fill_centis: int) -> int:
    """Return the cumulative ask depth at-or-better than a fill's marginal level.

    Walks ``yes_asks`` best-first, accumulating resting depth, and returns the
    running total through the first level that completes (or, for an empty book
    or an over-deep fill, the last level reached). Because the book's asks are
    best-first with ascending prices, this cumulative depth is exactly the depth
    resting at a price at-or-better than the fill's marginal level -- the
    quantity the participation cap measures a fill against.

    Args:
        yes_asks: The market's resting YES asks, best-first.
        fill_centis: The prospective fill size, in contract-centis.

    Returns:
        The cumulative contract-centis depth through the marginal level.
    """
    cumulative = 0
    for level in yes_asks:
        cumulative += level.quantity.value
        if cumulative >= fill_centis:
            return cumulative
    return cumulative


def _participation_fixed_point(
    size_centis: int,
    yes_asks: tuple[OrderBookLevel, ...],
    max_participation_ppm: int,
    *,
    lot: bool,
) -> int:
    """Solve the SPEC S9.6 participation cap by bounded fixed-point iteration.

    A fill may take at most ``max_participation_ppm`` of the depth resting
    at-or-better than its own marginal level -- but shrinking the fill can move
    the marginal level shallower, tightening the very depth the cap is measured
    against. Starting from ``min(size, participation of the total depth)``, each
    pass recomputes the marginal level's at-or-better depth and the cap it
    permits; the iterate can only shrink, so it converges in at most one pass per
    ask level (bounded -- never an unbounded loop). With ``lot`` set, every
    iterate is floored to a whole contract, so the returned size honors
    participation *at its own quantized marginal level* -- the form the emitted
    size must take.

    One pass per ask level is not merely a *safe* bound but a *sufficient* one,
    which is why this loop iterates over ``yes_asks`` itself and keeps no spare
    pass to fall out of. Two facts give it. A pass that does not converge must
    move the marginal level strictly shallower -- were the marginal level
    unchanged, the recomputed cap would equal the iterate just assigned from it,
    which converges. And an iterate marginal at the *deepest* level converges
    immediately, because there the cap is the starting clamp, which the iterate
    by construction never exceeds. So an ``n``-level book converges within ``n``
    passes; an empty book rests no depth, admits no fill, is already clamped to
    zero, and needs none. An earlier ``n + 1`` bound left a pass that no book
    could reach -- dead, and untestable by construction (issue #167). The
    invariant it was insuring is now asserted directly, over generated books, by
    ``test_participation_fixed_point_always_reaches_a_true_fixed_point``.

    Args:
        size_centis: The size to clip, in contract-centis (non-negative).
        yes_asks: The market's resting YES asks, best-first.
        max_participation_ppm: The maximum share of at-or-better depth a fill may
            take, in ppm.
        lot: Whether to floor every iterate to a whole contract. ``False`` gives
            the continuous limit used to name a binding cap; ``True`` gives the
            whole-contract size the selector emits.

    Returns:
        The participation-limited size, in contract-centis.
    """
    total_depth = sum(level.quantity.value for level in yes_asks)
    clamp = divide(
        max_participation_ppm * total_depth, _PPM_ONE, rounding=_SIZE_ROUNDING
    )
    current = min(size_centis, clamp)
    if lot:
        current = _floor_lot(current)
    for _ in yes_asks:
        depth = _depth_through_fill(yes_asks, current)
        cap = divide(max_participation_ppm * depth, _PPM_ONE, rounding=_SIZE_ROUNDING)
        if lot:
            cap = _floor_lot(cap)
        if current <= cap:
            break
        current = cap
    return current


def _binding_cap_name(
    raw_centis: int, ordered_limits: list[tuple[str, int]], clipped_minimum: int
) -> str | None:
    """Name the pinned-order-first cap whose limit equals the clipped minimum.

    Args:
        raw_centis: The raw (pre-cap) size, in contract-centis.
        ordered_limits: The cap ``(name, limit)`` pairs, in pinned order.
        clipped_minimum: The minimum across the raw size and every cap limit.

    Returns:
        The first cap name whose limit equals ``clipped_minimum``, or ``None``
        when the raw size was the minimum (no cap reduced it).
    """
    if raw_centis <= clipped_minimum:
        return None
    return next(name for name, limit in ordered_limits if limit == clipped_minimum)


def clip_to_caps(
    raw_size: ContractCentis,
    *,
    executable_price_ppm: int,
    order_book: OrderBookSnapshot,
    risk_config: RiskConfig,
    positions: PositionReadModelInput,
    bucket_cap_name: str = "per_bucket",
) -> CapClipResult:
    """Clip a raw stake through the SPEC S9.6 caps and quantization.

    Applies, in the pinned order, the five notional caps (``per_market``,
    ``per_event``, ``per_bucket``, ``total_deployed``, ``daily_notional``), then
    the participation cap, then the whole-contract exchange-minimum quantization.
    The binding cap is named from the *continuous* cap limits so the name
    reflects which economic constraint dominated, while the emitted size is the
    whole-contract participation fixed point of the notional-clipped size, so it
    honors participation at its own quantized marginal level (SPEC S9.6).

    When the final, lot-floored size drops below one whole contract, the result
    is zeroed. Its reported cap distinguishes two causes (SPEC S9.9 divergence):
    if a real cap already bound the *continuous* size (``binding_cap`` is not
    ``None``) that cap's own name survives -- a saturated per-bucket cap that
    drove the size to zero stays named ``per_bucket:<bucket-id>``, not masked as
    a generic exchange-minimum floor. Only when no cap bound and the raw size was
    itself merely sub-lot is it flagged ``exchange_min_order``.

    Args:
        raw_size: The raw fractional-Kelly size to clip, in contract-centis.
        executable_price_ppm: The executable fill price, in ppm-of-$1 (positive).
        order_book: The market's order-book snapshot, for the participation cap's
            resting-depth walk.
        risk_config: The risk configuration supplying the cap thresholds.
        positions: The account capital/exposure figures the notional caps read.
        bucket_cap_name: The name the per-bucket cap is reported under; defaults
            to the bare ``"per_bucket"`` so every pre-#47 call site and golden is
            unchanged. ``select()`` passes ``"per_bucket:<bucket-id>"`` (SPEC
            S9.9).

    Returns:
        A :class:`CapClipResult` carrying the final size and the binding cap's
        name (or ``None``).
    """
    raw_centis = raw_size.value
    yes_asks = order_book.yes_asks
    part_ppm = risk_config.max_participation_ppm

    notional_limits = _notional_cap_limits(
        positions, risk_config, executable_price_ppm, bucket_cap_name=bucket_cap_name
    )
    after_notional = min(raw_centis, *(limit for _, limit in notional_limits))

    continuous_participation = _participation_fixed_point(
        after_notional, yes_asks, part_ppm, lot=False
    )
    ordered_limits = [*notional_limits, ("participation", continuous_participation)]
    clipped_minimum = min(raw_centis, *(limit for _, limit in ordered_limits))
    binding_cap = _binding_cap_name(raw_centis, ordered_limits, clipped_minimum)

    final_centis = _participation_fixed_point(
        after_notional, yes_asks, part_ppm, lot=True
    )
    if final_centis < _EXCHANGE_MIN_ORDER_CENTIS:
        zeroed_cap = binding_cap if binding_cap is not None else _EXCHANGE_MIN_CAP_NAME
        return CapClipResult(size=ContractCentis(0), binding_cap=zeroed_cap)
    return CapClipResult(size=ContractCentis(final_centis), binding_cap=binding_cap)
