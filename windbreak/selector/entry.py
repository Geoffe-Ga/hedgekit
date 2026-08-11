"""SPEC S9.3 entry conditions gating a selector intent (issue #44).

:func:`evaluate_entry_conditions` renders the twelve pinned SPEC S9.3 named
conditions into a ``tuple[EntryCheck, ...]`` -- one check per name, in the fixed
evaluation order the ledger and the golden-determinism harness depend on. Each
condition is a small, branch-free helper reading only what it needs from the
:class:`~windbreak.selector.types.SelectorInputs` and the already-computed
:class:`~windbreak.selector.edge.EdgeFigures`; the aggregator assembles them in
order. The selector is pure and clock-free (SPEC S9.1), so every freshness
condition measures age against a *reference instant* derived from the inputs'
own timestamps -- ``T = max(order_book.fetched_at, forecast.created_at,
fee_model.as_of)`` -- never against ``datetime.now``. All temporal comparisons
are on :class:`~datetime.timedelta` objects, never their float
``.total_seconds()``. This module is on ``scripts/lint_no_floats.py``'s
denylist: no float, no bare ``/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from windbreak.selector.edge import EdgeFigures
    from windbreak.selector.types import SelectorInputs

#: Fee-model freshness ttl (SPEC S9.3 ``fee_model_current``): a fee schedule
#: older than this is treated as stale. Hard-coded here as the config seam SPEC
#: §16 will later expose alongside the other risk ttls.
_FEE_MODEL_TTL = timedelta(hours=24)

#: The reason detail naming the screener as the upstream owner of the two
#: eligibility conditions the selector cannot itself evaluate: market
#: jurisdiction/category metadata is not threaded into ``SelectorInputs`` (SPEC
#: S9.1), so these pass vacuously here and are enforced upstream by the screener.
_SCREENER_SEAM_DETAIL = (
    "enforced upstream by the screener (market metadata not in inputs)"
)


@dataclass(frozen=True, slots=True)
class EntryCheck:
    """One SPEC S9.3 entry condition's verdict.

    Attributes:
        name: The pinned SPEC S9.3 condition name.
        passed: Whether the condition holds (``True`` admits, ``False`` blocks).
        detail: A short, deterministic explanation carrying the operative values.
    """

    name: str
    passed: bool
    detail: str


def _reference_instant(inputs: SelectorInputs) -> datetime:
    """Return the clock-free reference instant ``T`` freshness is judged from.

    ``T = max(order_book.fetched_at, forecast.created_at, fee_model.as_of)`` --
    the latest fact the selector holds -- so no freshness check ever compares
    against a wall clock the pure selector may not read (SPEC S9.1).

    Args:
        inputs: The selector inputs carrying the three source timestamps.

    Returns:
        The reference instant ``T``.
    """
    return max(
        inputs.order_book.fetched_at,
        inputs.forecast.created_at,
        inputs.fee_model.as_of,
    )


def _net_edge_min(inputs: SelectorInputs, figures: EdgeFigures) -> EntryCheck:
    """Check the net edge clears the configured floor (SPEC S9.3)."""
    floor = inputs.risk_config.config.min_net_edge_ppm
    net = figures.net_edge_ppm
    return EntryCheck(
        name="net_edge_min",
        passed=net >= floor,
        detail=f"net_edge_ppm={net} min_net_edge_ppm={floor}",
    )


def _annualized_hurdle(inputs: SelectorInputs, figures: EdgeFigures) -> EntryCheck:
    """Check the annualized return beats the hurdle plus idle-cash APR (SPEC S9.2).

    The hurdle is ``annualized_hurdle_ppm + idle_cash_apr_ppm``: an entry must
    beat parked capital, not merely zero (SPEC S9.2).
    """
    risk = inputs.risk_config.config
    hurdle = risk.annualized_hurdle_ppm + risk.idle_cash_apr_ppm
    annualized = figures.annualized_expected_return_ppm
    return EntryCheck(
        name="annualized_hurdle",
        passed=annualized >= hurdle,
        detail=f"annualized_ppm={annualized} hurdle_ppm={hurdle}",
    )


def _ci_straddles_executable_price(
    inputs: SelectorInputs, figures: EdgeFigures
) -> EntryCheck:
    """Check the CI does not straddle the executable price (SPEC S9.3).

    Fails when the forecast confidence interval contains the executable price
    (inclusive on both bounds): a price the CI straddles is not decisively
    mispriced. Compares against ``figures.executable_price_ppm`` -- the fine
    ppm price ``gross_edge`` is chained off -- not a coarser
    ``executable_price_pips * 100`` reconstruction (0-99 ppm higher), so the
    CI/price comparison stays consistent with how the edge prices the same fill
    and a near-bound price cannot false-admit.
    """
    forecast = inputs.forecast
    price_ppm = figures.executable_price_ppm
    straddles = forecast.ci_low_ppm <= price_ppm <= forecast.ci_high_ppm
    return EntryCheck(
        name="ci_straddles_executable_price",
        passed=not straddles,
        detail=(
            f"ci=[{forecast.ci_low_ppm},{forecast.ci_high_ppm}] "
            f"executable_price_ppm={price_ppm}"
        ),
    )


def _quote_snapshot_fresh(inputs: SelectorInputs, reference: datetime) -> EntryCheck:
    """Check the quote snapshot is within its ttl (SPEC S9.3)."""
    age = reference - inputs.order_book.fetched_at
    ttl = timedelta(seconds=inputs.risk_config.config.quote_ttl_seconds)
    return EntryCheck(
        name="quote_snapshot_fresh",
        passed=age <= ttl,
        detail=f"age={age} quote_ttl={ttl}",
    )


def _forecast_fresh(inputs: SelectorInputs, reference: datetime) -> EntryCheck:
    """Check the reference instant is within the forecast horizon (SPEC S9.3)."""
    forecast = inputs.forecast
    deadline = forecast.created_at + timedelta(hours=forecast.forecast_horizon_hours)
    return EntryCheck(
        name="forecast_fresh",
        passed=reference <= deadline,
        detail=f"reference={reference.isoformat()} deadline={deadline.isoformat()}",
    )


def _fee_model_current(inputs: SelectorInputs, reference: datetime) -> EntryCheck:
    """Check the fee model is younger than its ttl (SPEC S9.3)."""
    age = reference - inputs.fee_model.as_of
    return EntryCheck(
        name="fee_model_current",
        passed=age <= _FEE_MODEL_TTL,
        detail=f"age={age} fee_model_ttl={_FEE_MODEL_TTL}",
    )


def _market_coherent(inputs: SelectorInputs) -> EntryCheck:
    """Check the forecast was not flagged incoherent (SPEC S9.3)."""
    flagged = inputs.forecast.coherence_flag
    return EntryCheck(
        name="market_coherent",
        passed=not flagged,
        detail=f"coherence_flag={flagged}",
    )


def _citation_support(inputs: SelectorInputs) -> EntryCheck:
    """Check the forecast carries at least one supporting citation (SPEC S9.3)."""
    count = len(inputs.forecast.citations)
    return EntryCheck(
        name="citation_support",
        passed=count > 0,
        detail=f"citation_count={count}",
    )


def _jurisdiction_eligible() -> EntryCheck:
    """Pass jurisdiction eligibility vacuously (SPEC S9.3; enforced by the screener).

    Market jurisdiction metadata is not threaded into ``SelectorInputs`` (SPEC
    S9.1), so this named condition passes here and is enforced upstream by the
    screener.
    """
    return EntryCheck(
        name="jurisdiction_eligible", passed=True, detail=_SCREENER_SEAM_DETAIL
    )


def _category_eligible() -> EntryCheck:
    """Pass category eligibility vacuously (SPEC S9.3; enforced by the screener).

    Market category metadata is not threaded into ``SelectorInputs`` (SPEC
    S9.1), so this named condition passes here and is enforced upstream by the
    screener.
    """
    return EntryCheck(
        name="category_eligible", passed=True, detail=_SCREENER_SEAM_DETAIL
    )


def _price_within_bands(inputs: SelectorInputs, figures: EdgeFigures) -> EntryCheck:
    """Check the executable price sits inside the open-price band (SPEC S9.4).

    On failure the ``detail`` leads with a greppable token -- ``price_below_min_
    open_band`` (price below the floor) or ``price_above_max_open_band`` (price
    above the ceiling), issue #46 -- so a downstream ledger reader can grep the
    exact band-fail direction from the rendered ``fail:price_within_bands: ...``
    reason, rather than reconstructing it from a bare price value. The pass-path
    ``detail`` is unchanged (the goldens depend on its byte-stable form).
    """
    risk = inputs.risk_config.config
    price = figures.executable_price_pips.value
    band = f"band=[{risk.min_open_price_pips},{risk.max_open_price_pips}]"
    if price < risk.min_open_price_pips:
        return EntryCheck(
            name="price_within_bands",
            passed=False,
            detail=f"price_below_min_open_band executable_price_pips={price} {band}",
        )
    if price > risk.max_open_price_pips:
        return EntryCheck(
            name="price_within_bands",
            passed=False,
            detail=f"price_above_max_open_band executable_price_pips={price} {band}",
        )
    return EntryCheck(
        name="price_within_bands",
        passed=True,
        detail=f"executable_price_pips={price} {band}",
    )


def _forecast_live_eligible(inputs: SelectorInputs) -> EntryCheck:
    """Check the forecast is eligible to back a live order (SPEC S9.3).

    Always required, regardless of paper/live mode: ``SelectorInputs`` carries
    no mode field yet, so this condition cannot be relaxed for a paper-mode
    evaluation and every decision is held to the live-eligibility bar.
    """
    eligible = inputs.forecast.eligible_for_live
    return EntryCheck(
        name="forecast_live_eligible",
        passed=eligible,
        detail=f"eligible_for_live={eligible}",
    )


def evaluate_entry_conditions(
    inputs: SelectorInputs, figures: EdgeFigures
) -> tuple[EntryCheck, ...]:
    """Evaluate the twelve SPEC S9.3 entry conditions in their pinned order.

    Args:
        inputs: The selector inputs carrying the forecast, order book, fee
            model, and risk configuration the conditions read.
        figures: The executable-edge figures computed for the same evaluation.

    Returns:
        Exactly twelve :class:`EntryCheck` results, one per named SPEC S9.3
        condition, in the fixed evaluation order (a decision admits only when
        every one of them passes).
    """
    reference = _reference_instant(inputs)
    return (
        _net_edge_min(inputs, figures),
        _annualized_hurdle(inputs, figures),
        _ci_straddles_executable_price(inputs, figures),
        _quote_snapshot_fresh(inputs, reference),
        _forecast_fresh(inputs, reference),
        _fee_model_current(inputs, reference),
        _market_coherent(inputs),
        _citation_support(inputs),
        _jurisdiction_eligible(),
        _category_eligible(),
        _price_within_bands(inputs, figures),
        _forecast_live_eligible(inputs),
    )
