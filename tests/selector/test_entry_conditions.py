"""Gate 1 RED tests for `windbreak.selector.entry` (issue #44, SPEC S9.3).

`windbreak/selector/entry.py` does not exist yet, so every test below fails
collection with `ModuleNotFoundError: No module named 'windbreak.selector.entry'`
-- the expected Gate 1 RED state for issue #44's entry-condition seam.

`evaluate_entry_conditions(inputs, figures)` renders the twelve SPEC S9.3
named conditions -- `net_edge_min`, `annualized_hurdle`,
`ci_straddles_executable_price`, `quote_snapshot_fresh`, `forecast_fresh`,
`fee_model_current`, `market_coherent`, `citation_support`,
`jurisdiction_eligible`, `category_eligible`, `price_within_bands`,
`forecast_live_eligible` -- into a `tuple[EntryCheck, ...]`, one check per
name, in that pinned order. This module hand-verifies a baseline scenario on
which all twelve pass, then flips exactly one input at a time (documenting the
two cases where the domain types themselves force a second check to flip
alongside it) to pin each condition's failure boundary -- including the
inclusive CI-straddle boundary and the "just below threshold" boundaries for
`net_edge_min` and `annualized_hurdle`.

A final test exercises `windbreak.selector.select` end-to-end over the all-pass
baseline: the emitted intent's price/size/max_notional/idempotency_key are
hand-derived (the idempotency key via the same
`hashlib.sha256(canonical_json(...))` primitive `client_order_id` already uses
elsewhere in this repo, applied to exactly the six named fields), and two
in-process `select` calls over the same inputs serialize byte-identically.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

from tests.selector.entry_baseline import (
    BASELINE_INSTANT,
    PROBE_SIZE,
    baseline_fee_model,
    baseline_forecast,
    baseline_inputs,
    baseline_order_book,
    baseline_slippage_model,
)
from windbreak.config.schema import RiskConfig
from windbreak.connector.fees import FeeModel
from windbreak.connector.models import OrderBookLevel
from windbreak.ledger.events import canonical_json
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips, ProbabilityPpm
from windbreak.selector import SelectorInputs, select, serialize_decision
from windbreak.selector.edge import EdgeFigures, compute_executable_edge
from windbreak.selector.entry import evaluate_entry_conditions

#: The exact SPEC S9.3 condition names, in the architect's pinned order.
_EXPECTED_CHECK_NAMES: tuple[str, ...] = (
    "net_edge_min",
    "annualized_hurdle",
    "ci_straddles_executable_price",
    "quote_snapshot_fresh",
    "forecast_fresh",
    "fee_model_current",
    "market_coherent",
    "citation_support",
    "jurisdiction_eligible",
    "category_eligible",
    "price_within_bands",
    "forecast_live_eligible",
)


def _figures_for(inputs: SelectorInputs) -> EdgeFigures:
    """Compute `EdgeFigures` for `inputs` at the fixed probe size.

    Args:
        inputs: The selector inputs to compute edge figures for.

    Returns:
        The computed `EdgeFigures`.

    Raises:
        AssertionError: If the book cannot fill the probe size (none of this
            module's fixtures are that shallow).
    """
    result = compute_executable_edge(
        order_book=inputs.order_book,
        size=PROBE_SIZE,
        forecast=inputs.forecast,
        fee_model=inputs.fee_model,
        slippage_model=inputs.slippage_model,
    )
    assert isinstance(result, EdgeFigures), f"expected EdgeFigures, got {result!r}"
    return result


def _failed_names(inputs: SelectorInputs) -> set[str]:
    """Return the set of failing `EntryCheck` names for `inputs`.

    Args:
        inputs: The selector inputs to evaluate.

    Returns:
        The set of `EntryCheck.name` values whose `passed` is `False`.
    """
    figures = _figures_for(inputs)
    checks = evaluate_entry_conditions(inputs, figures)
    return {check.name for check in checks if not check.passed}


# --- Baseline: hand-verified all-pass ----------------------------------------
#
# Baseline hand computation (probe size=100, single ask level 5000 pips/1000
# centis -> filled entirely at 5000, exact):
#   executable_price_pips = 5000; executable_price_ppm = 500_000 (exact)
#   gross_edge_ppm = 600_000 - 500_000 = 100_000
#   trading fee: rate=10_000, numerator=10_000*100*5000*5000=25_000_000_000_000,
#     cents=ceil(25_000_000_000_000/1e14)=1 -> 10_000 micros; fee_ppm=10_000 (exact)
#   fee_adjusted_edge_ppm = 100_000 - 10_000 = 90_000
#   slippage_adjusted_edge_ppm = 90_000 - 2_000 = 88_000
#   net_edge_ppm = 88_000 (no research haircut since issue #483)
#   net_edge_ppm(88_000) >= min_net_edge_ppm(30_000) -> net_edge_min passes
#   annualized = floor(88_000*1_000_000*8760 / (500_000*48))
#              = floor(770_880_000_000_000 / 24_000_000) = 32_120_000 (exact)
#   32_120_000 >= annualized_hurdle_ppm(200_000)+idle_cash_apr_ppm(40_000)=
#     240_000 -> annualized_hurdle passes
#   ci [100_000, 200_000] does not straddle executable_price_ppm(500_000)
#     -> ci_straddles_executable_price passes
#   T = BASELINE_INSTANT for order_book.fetched_at, forecast.created_at, and
#     fee_model.as_of alike -> every freshness check passes at zero age
#   coherence_flag=False, citations non-empty, price 5000 in [500, 9500],
#     eligible_for_live=True -> the remaining checks pass


def test_all_twelve_conditions_pass_on_the_baseline_scenario() -> None:
    """The hand-verified baseline passes every named SPEC S9.3 condition, in
    the pinned evaluation order (a deterministic count of exactly twelve).
    """
    inputs = baseline_inputs()
    figures = _figures_for(inputs)

    checks = evaluate_entry_conditions(inputs, figures)

    assert tuple(check.name for check in checks) == _EXPECTED_CHECK_NAMES
    assert all(check.passed for check in checks)


def test_jurisdiction_and_category_checks_are_vacuous_placeholders() -> None:
    """`jurisdiction_eligible` and `category_eligible` always pass -- market
    jurisdiction/category metadata is not threaded into `SelectorInputs` (SPEC
    S9.1) -- so this pins that they are intentional, documented placeholders
    (their `detail` names the screener as the upstream enforcer) rather than
    silent, unintentional no-ops: both are present, both pass, and both
    details name the screener seam explicitly.
    """
    inputs = baseline_inputs()
    figures = _figures_for(inputs)

    checks = evaluate_entry_conditions(inputs, figures)
    by_name = {check.name: check for check in checks}

    for name in ("jurisdiction_eligible", "category_eligible"):
        check = by_name[name]
        assert check.passed is True
        assert check.detail  # non-empty: a real (if vacuous) explanation
        assert "screener" in check.detail


# --- net_edge_min -------------------------------------------------------------


def test_net_edge_min_fails_just_below_threshold() -> None:
    """A 60_001 ppm slippage buffer drives the net edge to 29_999 -- one ppm
    below `min_net_edge_ppm`'s default 30_000 -- failing only `net_edge_min`.

    The buffer is a plain ppm subtraction with no rounding, and it replaces
    (rather than adds to) the baseline's own 2_000, so
    `fee_adjusted(90_000) - 60_001 == 29_999` precisely. The buffer replaced
    `research_cost_micros` as this boundary's lever when issue #483 removed the
    research haircut from the chain: a per-forecast cost amortized over a
    one-contract probe made `net_edge_min` unreachable for every market, so the
    charge no longer moves the edge at all and can no longer be used to walk it
    to a boundary.
    """
    inputs = baseline_inputs(
        slippage_model=baseline_slippage_model(per_contract_buffer_ppm=60_001)
    )

    figures = _figures_for(inputs)
    assert figures.net_edge_ppm == 29_999
    assert _failed_names(inputs) == {"net_edge_min"}


def test_net_edge_min_passes_one_ppm_higher() -> None:
    """One ppm less slippage clears the same floor, bracketing the boundary.

    Without this the refusal above could come from any condition sitting far
    from its own boundary; with it, the pair straddles `min_net_edge_ppm`
    exactly and no condition fails on the passing side.
    """
    inputs = baseline_inputs(
        slippage_model=baseline_slippage_model(per_contract_buffer_ppm=60_000)
    )

    figures = _figures_for(inputs)
    assert figures.net_edge_ppm == RiskConfig().min_net_edge_ppm
    assert _failed_names(inputs) == set()


# --- annualized_hurdle ---------------------------------------------------------


def test_annualized_hurdle_fails_just_below_threshold() -> None:
    """Stretching `forecast_horizon_hours` to 6_425 (net edge unchanged at
    88_000) drives the annualized figure to 239_962 -- below the default
    `annualized_hurdle_ppm + idle_cash_apr_ppm` (240_000) -- failing only
    `annualized_hurdle`. A *longer* horizon only relaxes `forecast_fresh`
    (`T <= created_at + horizon`), so no other check is disturbed.

    annualized = floor(88_000*1_000_000*8760 / (500_000*6425))
               = floor(770_880_000_000_000 / 3_212_500_000) = 239_962
    """
    inputs = baseline_inputs(forecast=baseline_forecast(forecast_horizon_hours=6_425))

    figures = _figures_for(inputs)
    assert figures.annualized_expected_return_ppm == 239_962
    assert _failed_names(inputs) == {"annualized_hurdle"}


# --- ci_straddles_executable_price: the inclusive lower boundary -------------


def test_ci_straddles_executable_price_fails_at_the_inclusive_lower_boundary() -> None:
    """`ci_low_ppm` set exactly equal to the executable price (500_000 ppm)
    straddles inclusively -- the check must fail at the boundary itself, not
    only strictly inside the interval.
    """
    inputs = baseline_inputs(
        forecast=baseline_forecast(ci_low_ppm=500_000, ci_high_ppm=600_000)
    )

    assert _failed_names(inputs) == {"ci_straddles_executable_price"}


# --- ci_straddles_executable_price: the inclusive upper boundary -------------


def test_ci_straddles_executable_price_fails_at_the_inclusive_upper_boundary() -> None:
    """`ci_high_ppm` set exactly equal to the baseline executable price
    (500_000 ppm) straddles inclusively -- the mirror of the lower-boundary
    test above, pinning that the upper bound is equally inclusive.
    """
    inputs = baseline_inputs(
        forecast=baseline_forecast(ci_low_ppm=100_000, ci_high_ppm=500_000)
    )

    assert _failed_names(inputs) == {"ci_straddles_executable_price"}


def test_ci_straddle_uses_finer_ppm_price_not_coarse_pips_reconstruction() -> None:
    """Regression for blocker 2: `_ci_straddles_executable_price` must compare
    against `figures.executable_price_ppm` -- the fine ppm price `gross_edge`
    is chained off -- never a coarser `executable_price_pips * 100`
    reconstruction. The baseline fixture is an exact single-level fill (no
    pips/ppm rounding remainder), so it cannot distinguish the two; this test
    engineers a two-level fill with a nonzero pips-rounding remainder so the
    fine ppm value and the coarse pips*100 reconstruction genuinely differ,
    and pins that the straddle check uses the finer one.

    Book (size=100 probe, two levels: 5000 pips/99 centis, then 5050
    pips/1 centi):
        cost = 5000*99 + 5050*1 = 495_000 + 5_050 = 500_050 micros (exact
            per-level accumulation, no rounding)
        executable_price_pips = ceil(500_050 / 100) = 5_001
            (100*5000=500_000, remainder 50 -> rounds up)
        executable_price_ppm  = 500_050 exactly (dividing by size_centis=100
            always exactly cancels the *100 ppm-per-pip scale factor at this
            probe size, so the ppm figure keeps the exact remainder the pips
            figure rounded away)

    The OLD (buggy) coarse reconstruction would have been
    `executable_price_pips.value * 100 == 500_100` -- strictly ABOVE the true
    500_050. Setting `ci_high_ppm=500_050` (exactly the true price) would
    NOT have straddled under that coarse comparison
    (`100_000 <= 500_100 <= 500_050` is false) while it DOES straddle under
    the correct fine-ppm comparison (`100_000 <= 500_050 <= 500_050` is true)
    -- exactly the case the old comparison would have gotten wrong.
    """
    inputs = baseline_inputs(
        order_book=baseline_order_book(
            yes_asks=(
                OrderBookLevel(price=PricePips(5_000), quantity=ContractCentis(99)),
                OrderBookLevel(price=PricePips(5_050), quantity=ContractCentis(1)),
            )
        ),
        forecast=baseline_forecast(ci_low_ppm=100_000, ci_high_ppm=500_050),
    )

    figures = _figures_for(inputs)
    assert figures.executable_price_pips == PricePips(5_001)
    assert figures.executable_price_ppm == 500_050
    assert figures.executable_price_pips.value * 100 == 500_100  # the old, wrong value

    assert _failed_names(inputs) == {"ci_straddles_executable_price"}


# --- quote_snapshot_fresh -----------------------------------------------------


def test_quote_snapshot_fresh_fails_when_the_quote_is_stale() -> None:
    """Moving `order_book.fetched_at` 100s before the reference instant `T`
    exceeds the default 10s `quote_ttl_seconds`, failing only
    `quote_snapshot_fresh`.
    """
    stale_fetched_at = BASELINE_INSTANT - timedelta(seconds=100)
    inputs = baseline_inputs(
        order_book=baseline_order_book(fetched_at=stale_fetched_at)
    )

    assert _failed_names(inputs) == {"quote_snapshot_fresh"}


# --- forecast_fresh ------------------------------------------------------------


def test_forecast_fresh_fails_once_past_created_at_plus_horizon() -> None:
    """A `created_at` far enough in the past that `T` exceeds
    `created_at + forecast_horizon_hours` fails only `forecast_fresh`.
    """
    stale_created_at = BASELINE_INSTANT - timedelta(days=365)
    inputs = baseline_inputs(forecast=baseline_forecast(created_at=stale_created_at))

    assert _failed_names(inputs) == {"forecast_fresh"}


# --- fee_model_current ---------------------------------------------------------


def test_fee_model_current_fails_when_the_fee_model_is_older_than_24h() -> None:
    """A `fee_model.as_of` 25h before the reference instant `T` exceeds the
    24h fee-model ttl, failing only `fee_model_current`.
    """
    stale_as_of = BASELINE_INSTANT - timedelta(hours=25)
    inputs = baseline_inputs(fee_model=baseline_fee_model(as_of=stale_as_of))

    assert _failed_names(inputs) == {"fee_model_current"}


# --- market_coherent (+ the forced forecast_live_eligible side effect) ------


def test_coherence_flag_fails_both_market_coherent_and_forecast_live_eligible() -> None:
    """`ForecastRecord.__post_init__` forces `eligible_for_live=False` whenever
    `coherence_flag` is `True` (raising `ValueError` otherwise), so this
    scenario necessarily fails BOTH `market_coherent` (the flag itself) and
    `forecast_live_eligible` (the forced-False eligibility) -- a coupling
    intrinsic to the domain type, not a loose test.
    """
    inputs = baseline_inputs(
        forecast=baseline_forecast(coherence_flag=True, eligible_for_live=False)
    )

    assert _failed_names(inputs) == {"market_coherent", "forecast_live_eligible"}


# --- citation_support ----------------------------------------------------------


def test_citation_support_fails_with_zero_citations() -> None:
    """An empty `citations` tuple fails only `citation_support`."""
    inputs = baseline_inputs(forecast=baseline_forecast(citations=()))

    assert _failed_names(inputs) == {"citation_support"}


# --- price_within_bands: below the floor --------------------------------------


def test_price_within_bands_fails_below_the_minimum_open_price() -> None:
    """A 400-pip executable price (below the default 500-pip floor) fails
    only `price_within_bands`; using the baseline fee (10_000 ppm taker) and
    slippage (2_000 ppm) unchanged, net edge (548_000) and the annualized
    figure stay comfortably above their thresholds, and the CI
    (100_000-200_000 ppm) still does not straddle the new 40_000-ppm
    executable price -- a genuine single-field flip (only the ask price).

    cost = 400*100 = 40_000 micros (exact); executable_price_ppm = 40_000
    gross_edge_ppm = 600_000 - 40_000 = 560_000
    trading fee: numerator=10_000*100*400*9600=3_840_000_000_000,
      cents=ceil(0.0384)=1 -> 10_000 micros; fee_ppm=10_000 (exact)
    fee_adjusted=550_000; slippage_adjusted=548_000; net_edge=548_000
    """
    inputs = baseline_inputs(
        order_book=baseline_order_book(
            yes_asks=(
                OrderBookLevel(price=PricePips(400), quantity=ContractCentis(1_000)),
            )
        )
    )

    figures = _figures_for(inputs)
    assert figures.executable_price_pips == PricePips(400)
    assert _failed_names(inputs) == {"price_within_bands"}


# --- price_within_bands: above the ceiling -------------------------------------


def test_price_within_bands_fails_above_the_maximum_open_price() -> None:
    """A 9_600-pip executable price (above the default 9_500-pip ceiling)
    fails only `price_within_bands`.

    Unlike the lower-band test, pushing the price this high against the
    baseline fixed-rate fee model would unavoidably drag `net_edge_min` down
    too (a 9_600-pip fill leaves little room for any fee against a 600_000 ppm
    probability), so this scenario is a deliberate, documented multi-field
    construction -- not a single-input flip: `probability_ppm` is raised to
    995_000 and fees/slippage are zeroed to keep `net_edge_min` and
    `annualized_hurdle` comfortably passing.

    cost = 9600*100 = 960_000 micros (exact); executable_price_ppm = 960_000
    gross_edge_ppm = 995_000 - 960_000 = 35_000 = net_edge_ppm (zero fee/
      slippage/research)
    annualized = floor(35_000*1_000_000*8760 / (960_000*48)) = 6_653_645
      (>= 240_000 -> annualized_hurdle passes)
    ci [100_000, 200_000] does not straddle 960_000 -> ci check passes
    """
    inputs = baseline_inputs(
        forecast=baseline_forecast(probability_ppm=995_000),
        order_book=baseline_order_book(
            yes_asks=(
                OrderBookLevel(price=PricePips(9_600), quantity=ContractCentis(1_000)),
            )
        ),
        fee_model=baseline_fee_model(
            model=FeeModel(
                schedule_id="entry-test-fee-zero",
                maker_fee_ppm=0,
                taker_fee_ppm=0,
                settlement_fee_ppm=0,
            )
        ),
        slippage_model=baseline_slippage_model(per_contract_buffer_ppm=0),
    )

    figures = _figures_for(inputs)
    assert figures.executable_price_pips == PricePips(9_600)
    assert figures.net_edge_ppm == 35_000
    assert _failed_names(inputs) == {"price_within_bands"}


# --- price_within_bands: boundary passes + greppable failure tokens (#46) ---


def test_price_within_bands_passes_at_the_inclusive_minimum_boundary() -> None:
    """A 500-pip executable price (exactly the default floor) passes
    `price_within_bands` -- the lower bound is inclusive, not only the
    already-tested 400-pip failure one pip further out.

    Reusing the baseline forecast/fee/slippage unchanged (probability
    600_000, taker 10_000 ppm, slippage buffer 2_000 ppm):
        cost = 500*100 = 50_000 micros (exact); executable_price_ppm = 50_000
        gross_edge_ppm = 600_000 - 50_000 = 550_000
        trading fee: numerator=10_000*100*500*9_500=4_750_000_000_000,
          cents=ceil(0.0475)=1 -> 10_000 micros; fee_ppm=10_000 (exact)
        fee_adjusted=540_000; slippage_adjusted=538_000; net_edge=538_000
          (>= 30_000 -> net_edge_min passes)
        ci [100_000,200_000] does not straddle 50_000 (50_000 < 100_000)
    """
    inputs = baseline_inputs(
        order_book=baseline_order_book(
            yes_asks=(
                OrderBookLevel(price=PricePips(500), quantity=ContractCentis(1_000)),
            )
        )
    )

    figures = _figures_for(inputs)
    assert figures.executable_price_pips == PricePips(500)
    checks = evaluate_entry_conditions(inputs, figures)
    assert tuple(check.name for check in checks) == _EXPECTED_CHECK_NAMES
    assert all(check.passed for check in checks)


def test_price_within_bands_passes_at_the_inclusive_maximum_boundary() -> None:
    """A 9_500-pip executable price (exactly the default ceiling) passes
    `price_within_bands` -- the mirror of the minimum-boundary test above,
    reusing the existing 9_600-pip failure scenario's probability/fee/
    slippage shape (probability 995_000, zero fee/slippage) one pip inside
    the ceiling instead of one pip outside it.

    cost = 9_500*100 = 950_000 micros (exact); executable_price_ppm = 950_000
    gross_edge_ppm = 995_000 - 950_000 = 45_000 = net_edge_ppm (zero fee/
      slippage/research)
    net_edge_ppm(45_000) >= min_net_edge_ppm(30_000) -> net_edge_min passes
    ci [100_000,200_000] does not straddle 950_000
    """
    inputs = baseline_inputs(
        forecast=baseline_forecast(probability_ppm=995_000),
        order_book=baseline_order_book(
            yes_asks=(
                OrderBookLevel(price=PricePips(9_500), quantity=ContractCentis(1_000)),
            )
        ),
        fee_model=baseline_fee_model(
            model=FeeModel(
                schedule_id="entry-test-fee-zero",
                maker_fee_ppm=0,
                taker_fee_ppm=0,
                settlement_fee_ppm=0,
            )
        ),
        slippage_model=baseline_slippage_model(per_contract_buffer_ppm=0),
    )

    figures = _figures_for(inputs)
    assert figures.executable_price_pips == PricePips(9_500)
    checks = evaluate_entry_conditions(inputs, figures)
    assert tuple(check.name for check in checks) == _EXPECTED_CHECK_NAMES
    assert all(check.passed for check in checks)


def test_price_within_bands_fails_one_pip_below_floor_with_a_greppable_detail() -> None:
    """A 499-pip executable price (one pip below the default 500-pip floor)
    fails only `price_within_bands`, and its `detail` now leads with the
    greppable token `price_below_min_open_band` (issue #46) rather than a
    bare `executable_price_pips=...` reconstruction.

    cost = 499*100 = 49_900 micros (exact); executable_price_ppm = 49_900
    gross_edge_ppm = 600_000 - 49_900 = 550_100
    trading fee: numerator=10_000*100*499*9_501=4_740_999_000_000,
      cents=ceil(0.04740999)=1 -> 10_000 micros; fee_ppm=10_000 (exact)
    fee_adjusted=540_100; slippage_adjusted=538_100; net_edge=538_100
      (>= 30_000 -> only price_within_bands fails)
    """
    inputs = baseline_inputs(
        order_book=baseline_order_book(
            yes_asks=(
                OrderBookLevel(price=PricePips(499), quantity=ContractCentis(1_000)),
            )
        )
    )

    figures = _figures_for(inputs)
    assert figures.executable_price_pips == PricePips(499)
    checks = evaluate_entry_conditions(inputs, figures)
    by_name = {check.name: check for check in checks}

    assert _failed_names(inputs) == {"price_within_bands"}
    assert by_name["price_within_bands"].detail.startswith("price_below_min_open_band")


def test_price_within_bands_fails_one_pip_above_ceiling_with_a_greppable_detail() -> (
    None
):
    """A 9_501-pip executable price (one pip above the default 9_500-pip
    ceiling) fails only `price_within_bands`, and its `detail` now contains
    the greppable token `price_above_max_open_band` (issue #46), mirroring
    the below-floor test above.

    cost = 9_501*100 = 950_100 micros (exact); executable_price_ppm=950_100
    gross_edge_ppm = 995_000 - 950_100 = 44_900 = net_edge_ppm (zero fee/
      slippage/research)
    net_edge_ppm(44_900) >= min_net_edge_ppm(30_000) -> only
      price_within_bands fails
    """
    inputs = baseline_inputs(
        forecast=baseline_forecast(probability_ppm=995_000),
        order_book=baseline_order_book(
            yes_asks=(
                OrderBookLevel(price=PricePips(9_501), quantity=ContractCentis(1_000)),
            )
        ),
        fee_model=baseline_fee_model(
            model=FeeModel(
                schedule_id="entry-test-fee-zero",
                maker_fee_ppm=0,
                taker_fee_ppm=0,
                settlement_fee_ppm=0,
            )
        ),
        slippage_model=baseline_slippage_model(per_contract_buffer_ppm=0),
    )

    figures = _figures_for(inputs)
    assert figures.executable_price_pips == PricePips(9_501)
    checks = evaluate_entry_conditions(inputs, figures)
    by_name = {check.name: check for check in checks}

    assert _failed_names(inputs) == {"price_within_bands"}
    assert "price_above_max_open_band" in by_name["price_within_bands"].detail


def test_select_renders_a_greppable_fail_reason_for_a_below_band_price() -> None:
    """`select`'s rendered reason for the below-floor scenario above starts
    with the pinned, greppable `fail:price_within_bands: price_below_min_
    open_band` prefix -- not just `evaluate_entry_conditions`'s own detail,
    but the exact string a downstream ledger reader greps for.
    """
    inputs = baseline_inputs(
        order_book=baseline_order_book(
            yes_asks=(
                OrderBookLevel(price=PricePips(499), quantity=ContractCentis(1_000)),
            )
        )
    )

    decision = select(inputs)

    assert decision.intents == ()
    matching = [
        reason
        for reason in decision.reasons
        if reason.startswith("fail:price_within_bands: price_below_min_open_band")
    ]
    assert len(matching) == 1


# --- forecast_live_eligible ------------------------------------------------------


def test_forecast_live_eligible_fails_when_not_eligible_for_live() -> None:
    """`eligible_for_live=False` (with no coherence flag or abstention reason)
    fails only `forecast_live_eligible`.
    """
    inputs = baseline_inputs(forecast=baseline_forecast(eligible_for_live=False))

    assert _failed_names(inputs) == {"forecast_live_eligible"}


# --- select()-level: hand-expected deterministic intent fields --------------


def test_select_emits_one_sized_intent_with_hand_expected_deterministic_fields() -> (
    None
):
    """On an all-pass scenario shaped for hand-verifiable Kelly sizing
    (issue #45), `select` emits exactly one intent whose
    price/size/max_notional/idempotency_key match hand-derived values, and
    two in-process `select` calls over the same inputs serialize
    byte-identically.

    This scenario overrides three baseline fields to keep the sizing
    arithmetic clean (matching the chief architect's own worked Kelly
    example): `probability_ppm=500_000` (baseline default 600_000), a single
    deep 4_500-pip ask level (baseline default 5_000-pip/1_000-centi), and
    zero-rate fee/slippage (baseline defaults taker=10_000/buffer=2_000).
    `vote_dispersion_ppm` stays the baseline default (0) and
    `research_cost_micros` stays the baseline default (0) -- and since
    issue #483 it would not move a figure at any value.

    Probe-size (100-centi) entry-check figures:
        cost = 4_500*100 = 450_000 micros (exact); executable_price_ppm =
            450_000 (exact, zero fee and zero slippage)
        gross_edge_ppm = 500_000-450_000 = 50_000 = net_edge_ppm (probe)
        net_edge_min: 50_000 >= 30_000 -> passes
        annualized = floor(50_000*1_000_000*8760 / (450_000*48))
                   = floor(438_000_000_000_000 / 21_600_000) = 20_277_777
            >= 240_000 -> annualized_hurdle passes
        ci [100_000,200_000] does not straddle 450_000; price_within_bands:
            4_500 pips in [500,9_500] -- all twelve conditions pass.

    Kelly sizing (g=dispersion_scale(0, 200_000)=1_000_000 at zero
    dispersion; kelly_fraction_ppm=100_000 default; capital=1_000_000_000
    from `baseline_positions`'s default `above_floor_capital_micros`):
        stake_micros = divide(1_000_000_000*50_000*100_000*1_000_000,
                               550_000*10**12, floor)
                     = divide(5*10**24, 5.5*10**17, floor) = 9_090_909
        size_centis  = divide(9_090_909*100, 450_000, floor)
                     = divide(909_090_900, 450_000, floor) = 2_020
            (450_000*2_020=909_000_000; remainder=90_900 < 450_000)
        No notional/participation cap binds (equity/deploy-cap huge, zero
        exposures, single deep ask level) -> `binding_cap=None`; the routine
        floor-to-100 quantization takes 2_020 -> 2_000 (final size).

    Re-walking the same flat 4_500-pip level at the final size=2_000:
        cost = 4_500*2_000 = 9_000_000 micros (exact); executable_price_ppm
            = 450_000 (exact); fee=0 (zero-rate model)
        net_edge_at_final_size = 500_000-450_000 = 50_000 >= 30_000 -> the
            final-size guard passes.
        price = marginal_price_pips = 4_500 (the only level walked)
        size = 2_000 centis
        max_notional = executable_cost_micros(9_000_000) + fee_micros(0)
                     = 9_000_000
        intent_id suffix is `:sized` (not `:probe`), reflecting real sizing.
    idempotency_key = sha256(canonical_json({forecast_id, market_ticker,
        outcome, action, price.value, size.value})).hexdigest() -- the same
        `hashlib.sha256(canonical_json(...))` primitive
        `windbreak.order_gateway.client_order_id` already uses elsewhere in
        this repo, applied here to exactly the six named fields (never
        derived by calling `select` itself).
    """
    inputs = baseline_inputs(
        forecast=baseline_forecast(probability_ppm=500_000),
        order_book=baseline_order_book(
            yes_asks=(
                OrderBookLevel(
                    price=PricePips(4_500), quantity=ContractCentis(1_000_000)
                ),
            )
        ),
        fee_model=baseline_fee_model(
            model=FeeModel(
                schedule_id="entry-test-fee-zero",
                maker_fee_ppm=0,
                taker_fee_ppm=0,
                settlement_fee_ppm=0,
            )
        ),
        slippage_model=baseline_slippage_model(per_contract_buffer_ppm=0),
    )

    decision = select(inputs)

    assert len(decision.intents) == 1
    intent = decision.intents[0]
    assert intent.intent_id == "fc-entry-0001:yes:buy:sized"
    assert intent.market_ticker == "ENTRY-TICKER"
    assert intent.outcome == "yes"
    assert intent.action == "buy"
    assert intent.price == PricePips(4_500)
    assert intent.size == ContractCentis(2_000)
    assert intent.max_notional == MoneyMicros(9_000_000)
    assert intent.implied_probability == ProbabilityPpm(500_000)

    expected_key_fields: dict[str, object] = {
        "forecast_id": "fc-entry-0001",
        "market_ticker": "ENTRY-TICKER",
        "outcome": "yes",
        "action": "buy",
        "price": 4_500,
        "size": 2_000,
    }
    expected_key = hashlib.sha256(
        canonical_json(expected_key_fields).encode("utf-8")
    ).hexdigest()
    assert intent.idempotency_key == expected_key

    assert decision.reasons[12] == (
        "sizing: raw_centis=2020 g_ppm=1000000 binding_cap=none final_centis=2000"
    )

    assert serialize_decision(select(inputs)) == serialize_decision(decision)


def test_select_declines_with_insufficient_depth_reason_and_no_intents() -> None:
    """When the book cannot fill the fixed 100-centi probe, `select` returns no
    intents and a single reason echoing `InsufficientDepth.reason` verbatim.

    An empty `yes_asks` tuple leaves zero resting depth against the 100-centi
    probe, so `compute_executable_edge` returns `InsufficientDepth(
    required_centis=100, available_centis=0)` and `select` short-circuits
    before ever evaluating the SPEC S9.3 entry conditions.
    """
    inputs = baseline_inputs(order_book=baseline_order_book(yes_asks=()))

    decision = select(inputs)

    assert decision.intents == ()
    assert decision.reasons == ("insufficient_book_depth: required=100 available=0",)


def test_select_declines_with_non_annualizable_reason_and_no_intents() -> None:
    """When the probe fill prices fine but its return cannot be annualized (a
    zero-hour forecast horizon), `select` returns no intents and a single
    reason echoing `NonAnnualizable.reason` verbatim -- proving `select`
    declines rather than raising `ZeroDivisionError`, and short-circuits
    before ever evaluating the SPEC S9.3 entry conditions (a
    `ci_straddles_executable_price` fail, say, is never also rendered here).

    The baseline order book (5000 pips/1000 centis) fills the 100-centi probe
    exactly: cost = 5000*100 = 500_000 micros; executable_price_ppm =
    ceil(500_000*100 / 100) = 500_000 (exact). With
    `forecast_horizon_hours=0`, `compute_executable_edge` returns
    `NonAnnualizable(executable_price_ppm=500_000, forecast_horizon_hours=0)`.
    """
    inputs = baseline_inputs(forecast=baseline_forecast(forecast_horizon_hours=0))

    decision = select(inputs)

    assert decision.intents == ()
    assert decision.reasons == (
        "non_annualizable: executable_price_ppm=500000 horizon_hours=0",
    )
