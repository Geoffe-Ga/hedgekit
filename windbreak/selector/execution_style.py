"""Execution-style decision for an opening selector intent (SPEC S9.7/S9.4).

SPEC S9.7 fixes the default execution style at ``cross``: a fundamentals bot
resting passively is a free option for faster traders, who pick it off the
instant the fundamentals move. ``rest_inside_spread`` is the deliberate
exception -- posting a passive order priced one improvement inside the book --
permitted *only* when the spread is genuinely wide AND the forecast edge still
persists at the improved (worse-for-us) resting price. SPEC S9.4 supplies the
open-price band the resting price must fall inside.

:func:`decide_execution_style` renders that policy as a five-row decision table
(any row firing falls back to ``cross``; only surviving all four rests):

    row1: ``yes_bids`` empty OR ``yes_asks`` empty -> cross (nothing to rest on)
    row2: spread (``best_ask - best_bid``, pips) < ``_WIDE_SPREAD_MIN_PIPS`` -> cross
    row3: rest price (``best_bid + _REST_IMPROVEMENT_PIPS``) outside the SPEC S9.4
          open-price band ``[min_open_price_pips, max_open_price_pips]`` -> cross
    row4: net edge at the rest price < ``min_net_edge_ppm`` -> cross
    row5: else -> ``rest_inside_spread`` at the rest price, carrying the risk
          config's ``resting_order_ttl_seconds`` / ``cancel_on_move_ticks``

The net-edge-at-rest arithmetic reuses :mod:`windbreak.selector.edge`'s private
``_fee_micros`` / ``_per_contract_ppm`` (the same intra-package private reuse
:mod:`windbreak.selector` already applies to ``_fee_micros``), so a resting price
is charged the identical fee/slippage/research haircut the entry edge is. All
arithmetic is integer; the pips->ppm bridge (1 pip == 100 ppm) is the sole role
of :data:`_PPM_PER_PIP`, and every division routes through
:func:`windbreak.numeric.divide` inside those two reused helpers. This module is
on ``scripts/lint_no_floats.py``'s denylist: no float, no bare ``/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from windbreak.numeric import PricePips
from windbreak.selector.edge import _fee_micros, _per_contract_ppm

if TYPE_CHECKING:
    from windbreak.numeric import ContractCentis
    from windbreak.selector.types import ExecutionStyle, SelectorInputs

#: Minimum bid/ask spread (in pips, inclusive) for a spread to count as "wide"
#: enough to rest inside rather than cross (SPEC S9.7). Hard-coded here as the
#: config seam SPEC S16 will later expose alongside the other execution knobs
#: (mirroring :data:`windbreak.selector.entry._FEE_MODEL_TTL`'s fencing).
_WIDE_SPREAD_MIN_PIPS = 300

#: How far inside the spread a resting order improves on the best bid, in pips:
#: the passive rest price is ``best_bid + _REST_IMPROVEMENT_PIPS``. Fenced here
#: as the same future SPEC S16 config seam as :data:`_WIDE_SPREAD_MIN_PIPS`.
_REST_IMPROVEMENT_PIPS = 100

#: Ppm-of-$1 per pip: a pip is 1e-4 $ and a ppm is 1e-6 $, so one pip is 100 ppm.
#: Converts a rest price in pips into the ppm-of-$1 the net-edge figure subtracts.
_PPM_PER_PIP = 100


@dataclass(frozen=True, slots=True)
class ExecutionStyleDecision:
    """The chosen execution style and its resting parameters (SPEC S9.7).

    Attributes:
        style: ``"cross"`` (take liquidity now) or ``"rest_inside_spread"``.
        resting_price_pips: The passive rest price, in pips, when resting; ``None``
            when crossing.
        resting_ttl_seconds: The resting order's ttl, in seconds, when resting;
            ``None`` when crossing.
        cancel_on_move_ticks: The adverse-move cancel guard, in ticks, when
            resting; ``None`` when crossing.
    """

    style: ExecutionStyle
    resting_price_pips: PricePips | None
    resting_ttl_seconds: int | None
    cancel_on_move_ticks: int | None


def _cross_decision() -> ExecutionStyleDecision:
    """Return the plain crossing decision, with every resting field ``None``.

    Returns:
        A ``"cross"`` :class:`ExecutionStyleDecision`.
    """
    return ExecutionStyleDecision(
        style="cross",
        resting_price_pips=None,
        resting_ttl_seconds=None,
        cancel_on_move_ticks=None,
    )


def _spread_pips(best_bid: PricePips, best_ask: PricePips) -> int:
    """Return the bid/ask spread, in pips.

    Args:
        best_bid: The best resting bid price, in pips.
        best_ask: The best resting ask price, in pips.

    Returns:
        ``best_ask - best_bid``, in pips.
    """
    return best_ask.value - best_bid.value


def _net_edge_at_price_ppm(
    inputs: SelectorInputs, rest_price: PricePips, size: ContractCentis
) -> int:
    """Return the net edge at a resting price, in ppm-of-$1 (SPEC S9.2/S9.7).

    Charges the resting price the identical worst-case fee and slippage buffer
    the entry edge charges its executable fill (SPEC S9.2), reusing
    :mod:`windbreak.selector.edge`'s helpers so the two figures stay consistent.
    Since issue #483 that chain carries no research haircut on either side; had
    only one side dropped it, a rest price would be judged against a
    systematically harsher edge than the cross price it is compared with, and
    ``rest_inside_spread`` would quietly stop being reachable. The ``rest_price``
    pip price is converted to ppm via :data:`_PPM_PER_PIP` (1 pip == 100 ppm).

    Args:
        inputs: The selector inputs supplying probability, fee model, and
            slippage buffer.
        rest_price: The passive rest price to evaluate the edge at, in pips.
        size: The fill size the fee haircut amortizes over, in centis.

    Returns:
        The net edge at ``rest_price``, in ppm-of-$1 (may be negative).
    """
    fee_ppm = _per_contract_ppm(
        _fee_micros(inputs.fee_model, rest_price.value, size.value), size.value
    )
    return (
        inputs.forecast.probability_ppm
        - rest_price.value * _PPM_PER_PIP
        - fee_ppm
        - inputs.slippage_model.per_contract_buffer_ppm
    )


def decide_execution_style(
    inputs: SelectorInputs, size: ContractCentis
) -> ExecutionStyleDecision:
    """Decide whether an opening intent crosses or rests (SPEC S9.7/S9.4).

    Renders the five-row decision table documented on this module: the default is
    ``cross``, and only a wide spread whose in-band resting price still clears the
    net-edge floor at ``size`` earns ``rest_inside_spread`` (stamped with the risk
    config's ttl / cancel-on-move guards).

    Args:
        inputs: The selector inputs carrying the order book, fee/slippage models,
            forecast, and risk config the decision reads.
        size: The final fill size the net-edge-at-rest haircuts amortize over, in
            contract-centis.

    Returns:
        The :class:`ExecutionStyleDecision`: a resting decision with a priced,
        in-band, edge-clearing rest price, or ``cross`` when any row fires.
    """
    book = inputs.order_book
    if not book.yes_bids or not book.yes_asks:
        return _cross_decision()
    best_bid = book.yes_bids[0].price
    best_ask = book.yes_asks[0].price
    if _spread_pips(best_bid, best_ask) < _WIDE_SPREAD_MIN_PIPS:
        return _cross_decision()
    rest_price = PricePips(best_bid.value + _REST_IMPROVEMENT_PIPS)
    risk = inputs.risk_config.config
    below_band = rest_price.value < risk.min_open_price_pips
    above_band = rest_price.value > risk.max_open_price_pips
    if below_band or above_band:
        return _cross_decision()
    if _net_edge_at_price_ppm(inputs, rest_price, size) < risk.min_net_edge_ppm:
        return _cross_decision()
    return ExecutionStyleDecision(
        style="rest_inside_spread",
        resting_price_pips=rest_price,
        resting_ttl_seconds=risk.resting_order_ttl_seconds,
        cancel_on_move_ticks=risk.cancel_on_move_ticks,
    )
