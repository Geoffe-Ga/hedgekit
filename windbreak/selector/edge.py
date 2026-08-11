"""Fee-aware executable-edge arithmetic for the selector (SPEC S9.2, issue #44).

:func:`compute_executable_edge` walks a market's ``yes_asks`` best-first to the
requested size, prices the fill at its *executable* (size-weighted) cost -- never
the midpoint or a single level's quote -- and chains the signed
ppm-of-$1-per-contract edge figures off it: gross, fee-adjusted, net (the
fee-adjusted edge less the slippage buffer), and the annualized expected return.
When the book cannot fill the size it returns :class:`InsufficientDepth` naming
the shortfall rather than raising, and when a fill prices but cannot be
annualized (a 0-pip price or a zero-hour forecast horizon would zero the
annualization denominator) it returns :class:`NonAnnualizable` -- so both a
shallow book and an undecidable fill are decidable non-entries, not crashes.

NO RESEARCH TERM (issue #483, owner decision 2026-08-10)

The chain used to carry a fifth figure, ``research_cost_adjusted_edge_ppm``:
the forecast's whole ``research_cost_micros`` amortized over the priced size.
:func:`windbreak.selector.select` prices its entry probe at a fixed **1.00
contract**, so at the gate that haircut was the entire per-forecast charge --
3 000 000 ppm at the shipped
:data:`~windbreak.forecast.budget.FULL_PIPELINE_RESEARCH_COST_MICROS`. A binary
contract's gross edge over one contract cannot exceed 1 000 000 ppm, so the net
edge was at most -2 000 000 ppm for **every** market, at every price, with any
capital, and ``net_edge_min`` (30 000) was arithmetically unreachable.

That is a category error rather than a calibration one: research cost is
incurred once per *forecast* and the edge is a per-*contract* return, so no
value of the charge makes the comparison mean anything at a one-contract probe.
Governance of research spend moved to the configurable per-UTC-day ceiling
(:class:`windbreak.forecast.budget.ResearchBudget`), which bounds the quantity
that actually needs bounding -- total spend -- and leaves this module to judge a
fill on its own economics. ``ForecastRecord.research_cost_micros`` is still
recorded, ledgered and reported; it is simply no longer subtracted from a
per-contract edge.

Unit bridge (the one place scales cross): a price is in pips (1e-4 $/contract)
and a size in centis (1e-2 contracts), so ``price.value * count.value`` is an
*exact* micros (1e-6 $) product with no remainder. A per-contract cost of
``total_micros`` over ``size_centis`` re-expressed in ppm-of-$1 is
``total_micros * 100 / size_centis`` (1 pip == 100 ppm), the sole role of
:data:`_PPM_PER_PIP`. Every division routes through
:func:`windbreak.numeric.divide` with an explicit conservative direction -- costs
:data:`~windbreak.numeric.RoundingDirection.OVERSTATE_COST` (ceiling, never
understate what a fill costs), the annualized return
:data:`~windbreak.numeric.RoundingDirection.UNDERSTATE_EQUITY` (floor toward
negative infinity, never overstate what a position earns). This module is on
``scripts/lint_no_floats.py``'s denylist: no float, no bare ``/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from windbreak.numeric import MoneyMicros, PricePips, RoundingDirection, divide

if TYPE_CHECKING:
    from windbreak.connector.models import OrderBookLevel, OrderBookSnapshot
    from windbreak.forecast.records import ForecastRecord
    from windbreak.numeric import ContractCentis
    from windbreak.selector.types import FeeModelInput, SlippageModelInput

#: Ppm-of-$1 per pip: a pip is 1e-4 $ and a ppm is 1e-6 $, so one pip is 100
#: ppm. Multiplying a micros cost by this before dividing by a centis count
#: yields that per-contract cost as a ppm-of-$1 price (see the unit bridge in
#: the module docstring).
_PPM_PER_PIP = 100

#: Parts-per-million scale: the annualized return is the per-horizon edge
#: fraction (``net_edge_ppm / executable_price_ppm``, both already ppm)
#: re-expressed back in ppm, so the numerator carries this 1e6 factor.
_PPM_SCALE = 1_000_000

#: Hours in a (365-day) year: the ``8760 / forecast_horizon_hours`` factor that
#: annualizes a single-horizon expected return (SPEC S9.2).
_HOURS_PER_YEAR = 8760

#: Costs round up so an edge is never overstated by understating what a fill,
#: fee, or research spend costs.
_COST_ROUNDING = RoundingDirection.OVERSTATE_COST

#: The annualized return floors toward negative infinity so a positive figure is
#: never overstated and a negative one is never truncated toward zero.
_RETURN_ROUNDING = RoundingDirection.UNDERSTATE_EQUITY


@dataclass(frozen=True, slots=True)
class InsufficientDepth:
    """The book could not fill the requested size (SPEC S9.2 non-entry).

    Returned -- never raised -- when the resting ``yes_asks`` depth is strictly
    less than the requested size, so a shallow book is a decidable non-entry
    the selector can render into a reason rather than an exception.

    Attributes:
        required_centis: The size the walk needed to fill, in contract-centis.
        available_centis: The total resting ask depth available, in
            contract-centis (strictly less than ``required_centis``).
    """

    required_centis: int
    available_centis: int

    @property
    def reason(self) -> str:
        """Return the pinned reason string naming the depth shortfall.

        Returns:
            ``"insufficient_book_depth: required=<R> available=<A>"``.
        """
        return (
            f"insufficient_book_depth: required={self.required_centis} "
            f"available={self.available_centis}"
        )


@dataclass(frozen=True, slots=True)
class NonAnnualizable:
    """The fill priced but its single-horizon return cannot be annualized.

    Returned -- never raised -- when a fill completes but the annualization
    denominator ``executable_price_ppm * forecast_horizon_hours`` would be
    zero: a 0-pip fill prices at ``executable_price_ppm == 0``, and a valid
    :class:`~windbreak.forecast.records.ForecastRecord` may carry
    ``forecast_horizon_hours == 0`` (its ``__post_init__`` does not forbid it).
    Either makes the single-horizon expected return undefined, so this is a
    decidable non-entry the selector renders into a reason rather than an
    uncaught ``ZeroDivisionError`` (SPEC S9.2: non-fillable/undecidable is a
    NON-ENTRY, not a crash). Distinct from :class:`InsufficientDepth`: the book
    *did* fill the size, so this is not a depth shortfall.

    Attributes:
        executable_price_ppm: The fill's executable price, in ppm-of-$1;
            non-positive when this decline is returned.
        forecast_horizon_hours: The forecast horizon, in hours; non-positive
            when this decline is returned.
    """

    executable_price_ppm: int
    forecast_horizon_hours: int

    @property
    def reason(self) -> str:
        """Return the pinned reason string naming the non-annualizable fill.

        Returns:
            ``"non_annualizable: executable_price_ppm=<P> horizon_hours=<H>"``.
        """
        return (
            f"non_annualizable: executable_price_ppm={self.executable_price_ppm} "
            f"horizon_hours={self.forecast_horizon_hours}"
        )


@dataclass(frozen=True, slots=True)
class EdgeFigures:
    """The SPEC S9.2 executable-edge figures for one priced fill.

    The ``*_edge_ppm`` / ``*_return_ppm`` figures are signed ppm-of-$1 per
    contract, chained conservatively off the executable (size-weighted) fill
    price. The three unit-typed fields carry the fill's raw cost and prices.

    Attributes:
        gross_edge_ppm: Forecast probability minus executable price, in ppm.
        fee_adjusted_edge_ppm: Gross edge less the per-contract fee, in ppm.
        net_edge_ppm: The net edge: fee-adjusted less the slippage buffer, and
            the last link in the chain since issue #483 removed the research
            haircut (see this module's docstring). This is the figure
            ``net_edge_min`` gates on and Kelly sizes against.
        annualized_expected_return_ppm: The net edge annualized over the
            forecast horizon (floored toward negative infinity).
        executable_price_ppm: The size-weighted fill price, in ppm-of-$1
            (ceiling) -- the fine-grained price ``gross_edge_ppm`` is chained
            off, carried so entry conditions compare the confidence interval
            against the same price the edge does (SPEC S9.3), never a coarser
            pips-reconstructed value.
        executable_price_pips: The size-weighted fill price, in pips (ceiling).
        executable_cost_micros: The exact total fill cost, in micros.
        marginal_price_pips: The deepest level's price the walk reached, in pips.
    """

    gross_edge_ppm: int
    fee_adjusted_edge_ppm: int
    net_edge_ppm: int
    annualized_expected_return_ppm: int
    executable_price_ppm: int
    executable_price_pips: PricePips
    executable_cost_micros: MoneyMicros
    marginal_price_pips: PricePips


def _walk_book_cost(
    yes_asks: tuple[OrderBookLevel, ...], size_centis: int
) -> tuple[int, PricePips] | None:
    """Walk asks best-first to ``size_centis``, returning cost and marginal.

    Takes ``min(remaining, level.quantity)`` at each level, accumulating the
    exact micros ``price.value * taken`` (no rounding), until the size is filled.

    Args:
        yes_asks: The market's resting YES asks, best-first.
        size_centis: The size to fill, in contract-centis (positive).

    Returns:
        ``(total_cost_micros, marginal_price_pips)`` on a complete fill, where
        the marginal price is the deepest level the walk reached; ``None`` if
        the total resting depth is strictly less than ``size_centis``.
    """
    remaining = size_centis
    total_cost = 0
    marginal: PricePips | None = None
    for level in yes_asks:
        if remaining <= 0:
            break
        taken = min(remaining, level.quantity.value)
        total_cost += level.price.value * taken
        marginal = level.price
        remaining -= taken
    if remaining > 0 or marginal is None:
        return None
    return total_cost, marginal


def _per_contract_ppm(total_micros: int, size_centis: int) -> int:
    """Re-express a total micros cost as a per-contract ppm-of-$1 price.

    Args:
        total_micros: A total cost, in micros (1e-6 $).
        size_centis: The contract count the cost spans, in centis (positive).

    Returns:
        The per-contract cost in ppm-of-$1, rounded up (``OVERSTATE_COST``).
    """
    return divide(total_micros * _PPM_PER_PIP, size_centis, rounding=_COST_ROUNDING)


def _fee_micros(fee_model: FeeModelInput, price_pips: int, size_centis: int) -> int:
    """Return the worst-case trading-plus-settlement fee for a fill, in micros.

    Feeding the ceiling VWAP into ``FeeModel.max_trading_fee_micros`` (whose
    quadratic peaks at price=5000) can, for a VWAP above 5000, marginally
    understate the raw fee bound; this is immaterial because ``fees.py`` rounds
    that bound up to a whole cent (``OVERSTATE_COST``), which dominates the
    sub-pip effect.

    Args:
        fee_model: The fee schedule carrier to price with.
        price_pips: The executable fill price, in pips.
        size_centis: The fill size, in contract-centis (positive).

    Returns:
        The summed worst-case trading and settlement fee bounds, in micros.
    """
    trading = fee_model.model.max_trading_fee_micros(price_pips, size_centis)
    settlement = fee_model.model.max_settlement_fee_micros(size_centis)
    return trading + settlement


def _is_annualizable(executable_price_ppm: int, horizon_hours: int) -> bool:
    """Return whether a fill's single-horizon return can be annualized.

    The annualization denominator is ``executable_price_ppm * horizon_hours``
    (:func:`_annualize`), well-defined only when both factors are strictly
    positive. A 0-pip fill prices at ``executable_price_ppm == 0`` and a valid
    :class:`~windbreak.forecast.records.ForecastRecord` may carry
    ``horizon_hours == 0``, either of which would zero the denominator.

    Args:
        executable_price_ppm: The fill's executable price, in ppm-of-$1.
        horizon_hours: The forecast horizon, in hours.

    Returns:
        ``True`` when both factors are strictly positive, else ``False``.
    """
    return executable_price_ppm > 0 and horizon_hours > 0


def _annualize(net_edge_ppm: int, executable_price_ppm: int, horizon_hours: int) -> int:
    """Annualize a single-horizon net edge over the forecast horizon (S9.2).

    Args:
        net_edge_ppm: The research-cost-adjusted net edge, in ppm (may be
            negative).
        executable_price_ppm: The executable fill price, in ppm-of-$1 (positive).
        horizon_hours: The forecast horizon, in hours (positive).

    Returns:
        The annualized expected return, in ppm, floored toward negative infinity
        (``UNDERSTATE_EQUITY``) so a negative edge is never truncated toward zero
        and a positive one is never overstated.

    Note:
        Callers must guard the zero-denominator case (``executable_price_ppm``
        or ``horizon_hours`` non-positive) before calling; a non-annualizable
        fill is declined as a non-entry upstream (see
        :func:`compute_executable_edge`), so the denominator is always positive
        here.
    """
    numerator = net_edge_ppm * _PPM_SCALE * _HOURS_PER_YEAR
    denominator = executable_price_ppm * horizon_hours
    return divide(numerator, denominator, rounding=_RETURN_ROUNDING)


def compute_executable_edge(
    order_book: OrderBookSnapshot,
    size: ContractCentis,
    forecast: ForecastRecord,
    fee_model: FeeModelInput,
    slippage_model: SlippageModelInput,
) -> EdgeFigures | InsufficientDepth | NonAnnualizable:
    """Price a fill and chain the executable-edge figures off it (SPEC S9.2).

    Walks ``order_book.yes_asks`` best-first to ``size`` (SPEC S9.2): the
    executable price is the size-weighted cost of the actual fill, so a two-level
    ``(4500@10_000, 4700@10_000)`` walk over ``15_000`` prices at the VWAP
    ``4567``, not the ``4600`` midpoint. Fees and the slippage buffer are then
    subtracted in ppm, and the resulting net edge is annualized over the
    forecast horizon. The forecast's ``research_cost_micros`` is deliberately
    *not* subtracted -- see this module's docstring and issue #483. All
    arithmetic is integer with conservative rounding (costs up, the annualized
    return floored); the unit bridge is documented on the module.

    Args:
        order_book: The market's order-book snapshot to fill against.
        size: The size to price, in contract-centis (positive).
        forecast: The forecast supplying probability and horizon.
        fee_model: The fee schedule carrier to charge the fill with.
        slippage_model: The per-contract slippage buffer to subtract.

    Returns:
        :class:`EdgeFigures` on a complete fill; :class:`InsufficientDepth`
        naming the shortfall when the resting ask depth is below ``size``; or
        :class:`NonAnnualizable` when the fill completed but its return cannot
        be annualized because ``executable_price_ppm`` or the forecast horizon
        is non-positive (guarding the ``_annualize`` denominator so a 0-pip
        book level or a zero-hour horizon declines as a non-entry rather than
        raising ``ZeroDivisionError``).
    """
    walked = _walk_book_cost(order_book.yes_asks, size.value)
    if walked is None:
        available = sum(level.quantity.value for level in order_book.yes_asks)
        return InsufficientDepth(required_centis=size.value, available_centis=available)
    total_cost, marginal = walked
    size_centis = size.value

    executable_price_pips = PricePips(
        divide(total_cost, size_centis, rounding=_COST_ROUNDING)
    )
    executable_price_ppm = _per_contract_ppm(total_cost, size_centis)
    horizon_hours = forecast.forecast_horizon_hours
    if not _is_annualizable(executable_price_ppm, horizon_hours):
        return NonAnnualizable(
            executable_price_ppm=executable_price_ppm,
            forecast_horizon_hours=horizon_hours,
        )
    gross_edge_ppm = forecast.probability_ppm - executable_price_ppm

    fee_ppm = _per_contract_ppm(
        _fee_micros(fee_model, executable_price_pips.value, size_centis), size_centis
    )
    fee_adjusted_edge_ppm = gross_edge_ppm - fee_ppm
    net_edge_ppm = fee_adjusted_edge_ppm - slippage_model.per_contract_buffer_ppm

    return EdgeFigures(
        gross_edge_ppm=gross_edge_ppm,
        fee_adjusted_edge_ppm=fee_adjusted_edge_ppm,
        net_edge_ppm=net_edge_ppm,
        annualized_expected_return_ppm=_annualize(
            net_edge_ppm,
            executable_price_ppm,
            horizon_hours,
        ),
        executable_price_ppm=executable_price_ppm,
        executable_price_pips=executable_price_pips,
        executable_cost_micros=MoneyMicros(total_cost),
        marginal_price_pips=marginal,
    )
