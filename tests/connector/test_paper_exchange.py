"""Failing-first tests for `windbreak.connector.fills` / `.paper` (issue #19).

SPEC S7.5 (`PaperExchange`), S17.4 (the paper-fill realism model, normative),
and S9.5 (participation caps) define a pessimistic fill simulator: taker
orders walk the recorded book and pay the live fee schedule plus a slippage
haircut (default +25% of modeled fees); resting orders fill only when the
recorded market *trades through* the limit price (a touch is never a fill).
Neither `windbreak.connector.fills` nor `windbreak.connector.paper` exists yet,
so importing either fails collection with `ModuleNotFoundError: No module
named 'windbreak.connector.fills'` -- the expected Gate 1 RED state for issue
#19. (`windbreak.connector` itself, along with `.models`, `.fees`,
`.semantics`, `.interface`, and `.fake`, already exists -- issues #16/#18 are
merged -- so every *other* import below resolves cleanly.)

Every golden number pinned here is hand-derived from `FeeModel`'s own
documented formula (`rate_ppm * count * price * (10_000 - price)`, ceil-
rounded to the cent -- see `windbreak/connector/fees.py`) plus the haircut
formula `ceil(fee * haircut_ppm / 1_000_000)`, so a mutation in the walk, the
participation cap, or the haircut arithmetic flips a concrete assertion
rather than a vague "no exception raised" check.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from windbreak.connector import fills, paper
from windbreak.connector.fees import FeeModel
from windbreak.connector.interface import MarketConnector, UnknownMarketError
from windbreak.connector.models import OrderBookLevel
from windbreak.connector.semantics import (
    BalanceSemantics,
    CancelCollateralRelease,
    FeeDebitTiming,
    FeeRounding,
    HaltedMarketBehavior,
    OrderCollateralInAvailable,
    OrderCollateralInTotal,
    PartialFillRepresentation,
    UnsettledProceeds,
)
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.connector.paper import PaperExchange

#: A fixed timestamp for `TradePrint`s that don't need a specific value.
_TS = datetime(2025, 1, 1, tzinfo=UTC)

#: The `since` bound used by every `get_fills` assertion below: strictly
#: before every fixture's session timestamps, so "everything so far" is
#: exactly what each scenario produced.
_SINCE = datetime(2024, 1, 1, tzinfo=UTC)


def _fee_model(taker_fee_ppm: int = 70_000) -> FeeModel:
    """Build a `FeeModel` with a given taker rate (maker/settlement pinned at 0)."""
    return FeeModel(
        schedule_id="paper-test-v1",
        maker_fee_ppm=0,
        taker_fee_ppm=taker_fee_ppm,
        settlement_fee_ppm=0,
    )


# =============================================================================
# windbreak.connector.fills: pure taker-walk / resting-fill primitives
# =============================================================================


class TestModuleConstants:
    """Pins the fills module's default ppm constants and model-version hash."""

    def test_default_haircut_ppm_is_25_percent(self) -> None:
        assert fills.DEFAULT_FEE_HAIRCUT_PPM == 250_000

    def test_default_max_participation_ppm_is_25_percent(self) -> None:
        assert fills.DEFAULT_MAX_PARTICIPATION_PPM == 250_000

    def test_paper_fill_model_version_is_a_sha256_hex_digest(self) -> None:
        """A sha256 hex digest is exactly 64 lowercase hex characters."""
        version = fills.PAPER_FILL_MODEL_VERSION

        assert isinstance(version, str)
        assert len(version) == 64
        assert version == version.lower()
        int(version, 16)  # raises ValueError if any character isn't hex

    def test_paper_module_reexports_the_identical_model_version(self) -> None:
        """`paper.py` re-exports the same constant, not a redefinition."""
        assert paper.PAPER_FILL_MODEL_VERSION == fills.PAPER_FILL_MODEL_VERSION


class TestParticipationCap:
    """Pins `participation_cap`: floor-rounded, always against the trader."""

    def test_exact_quarter_of_an_even_depth(self) -> None:
        cap = fills.participation_cap(
            ContractCentis(1000), max_participation_ppm=250_000
        )

        assert cap == ContractCentis(250)

    def test_rounds_down_a_fractional_quarter(self) -> None:
        """333 * 250_000 / 1_000_000 == 83.25 -- floors to 83, never up."""
        cap = fills.participation_cap(
            ContractCentis(333), max_participation_ppm=250_000
        )

        assert cap == ContractCentis(83)

    def test_tiny_depth_rounds_the_cap_to_zero(self) -> None:
        """3 * 250_000 / 1_000_000 == 0.75 -- floors to 0, not up to 1."""
        cap = fills.participation_cap(ContractCentis(3), max_participation_ppm=250_000)

        assert cap == ContractCentis(0)

    def test_full_participation_returns_the_full_depth(self) -> None:
        cap = fills.participation_cap(
            ContractCentis(300), max_participation_ppm=1_000_000
        )

        assert cap == ContractCentis(300)

    def test_zero_participation_always_yields_zero(self) -> None:
        cap = fills.participation_cap(ContractCentis(1000), max_participation_ppm=0)

        assert cap == ContractCentis(0)

    def test_returns_a_contract_centis(self) -> None:
        cap = fills.participation_cap(
            ContractCentis(1000), max_participation_ppm=250_000
        )

        assert isinstance(cap, ContractCentis)


class TestWalkTakerFill:
    """Pins `walk_taker_fill`'s pessimistic multi-level taker walk.

    Convention: `levels` is the ask side, best-first (ascending price); a buy
    at `limit` crosses every level whose price is at-or-better (<=) the limit,
    walked in order, capped at `max_participation_ppm` of the *eligible*
    (at-or-better) depth, rounded down. `total_cost` always overstates
    (`book_cost + fee + haircut`, each individually rounded against the
    trader), matching SPEC S17.4's "pessimistic" mandate.
    """

    def test_participation_cap_binds_within_the_first_level(self) -> None:
        """Depth 1000 -> cap 250 (25%); the single level has plenty of room."""
        levels = (OrderBookLevel(PricePips(4600), ContractCentis(1000)),)

        result = fills.walk_taker_fill(
            levels,
            PricePips(4600),
            ContractCentis(1000),
            _fee_model(),
            haircut_ppm=250_000,
            max_participation_ppm=250_000,
        )

        assert result.filled == ContractCentis(250)
        assert result.consumed == (
            OrderBookLevel(PricePips(4600), ContractCentis(250)),
        )
        # book_cost: 4600 * 250 == 1_150_000 micros exactly (price * qty, no remainder).
        assert result.book_cost == MoneyMicros(1_150_000)
        # fee: ceil(70_000 * 250 * 4600 * 5400 / 1e14) == ceil(4.347) == 5 cents.
        assert result.fee == MoneyMicros(50_000)
        # haircut: ceil(50_000 * 250_000 / 1e6) == 12_500 (exact: 50_000 / 4).
        assert result.haircut == MoneyMicros(12_500)
        assert result.total_cost == MoneyMicros(1_212_500)

    def test_multi_level_walk_spans_two_levels_with_a_partial_second(self) -> None:
        """Eligible depth 200+1000=1200 -> cap 300; consumes level 0 fully (200)
        then 100 of level 1's 1000, landing exactly on the cap."""
        levels = (
            OrderBookLevel(PricePips(4600), ContractCentis(200)),
            OrderBookLevel(PricePips(4700), ContractCentis(1000)),
        )

        result = fills.walk_taker_fill(
            levels,
            PricePips(4700),
            ContractCentis(1000),
            _fee_model(),
            haircut_ppm=250_000,
            max_participation_ppm=250_000,
        )

        assert result.filled == ContractCentis(300)
        assert result.consumed == (
            OrderBookLevel(PricePips(4600), ContractCentis(200)),
            OrderBookLevel(PricePips(4700), ContractCentis(100)),
        )
        # book_cost: 4600*200 + 4700*100 == 920_000 + 470_000 == 1_390_000.
        assert result.book_cost == MoneyMicros(1_390_000)
        # fee: ceil(4600 level: 3.4776 -> 4 cents) + ceil(4700 level: 1.7437 -> 2 cents)
        #    == 40_000 + 20_000 == 60_000 micros.
        assert result.fee == MoneyMicros(60_000)
        # haircut: ceil(60_000 * 250_000 / 1e6) == 15_000 (exact: 60_000 / 4).
        assert result.haircut == MoneyMicros(15_000)
        assert result.total_cost == MoneyMicros(1_465_000)

    def test_requested_smaller_than_cap_and_depth_fills_exactly_requested(self) -> None:
        """Cap (250) and depth (1000) both exceed requested (100): requested binds."""
        levels = (OrderBookLevel(PricePips(4600), ContractCentis(1000)),)

        result = fills.walk_taker_fill(
            levels,
            PricePips(4600),
            ContractCentis(100),
            _fee_model(),
            haircut_ppm=250_000,
            max_participation_ppm=250_000,
        )

        assert result.filled == ContractCentis(100)
        assert result.book_cost == MoneyMicros(460_000)  # 4600 * 100
        # fee: ceil(70_000 * 100 * 4600 * 5400 / 1e14) == ceil(1.7388) == 2 cents.
        assert result.fee == MoneyMicros(20_000)
        assert result.haircut == MoneyMicros(5_000)  # ceil(20_000 * 250_000 / 1e6)
        assert result.total_cost == MoneyMicros(485_000)

    def test_a_level_worse_than_the_limit_is_never_walked(self) -> None:
        """The 4800 level is strictly worse than the 4700 limit and must be
        entirely excluded from both `consumed` and the eligible-depth
        computation feeding the participation cap."""
        levels = (
            OrderBookLevel(PricePips(4600), ContractCentis(300)),
            OrderBookLevel(PricePips(4800), ContractCentis(500)),
        )

        result = fills.walk_taker_fill(
            levels,
            PricePips(4700),
            ContractCentis(1000),
            _fee_model(),
            haircut_ppm=250_000,
            max_participation_ppm=250_000,
        )

        # Eligible depth is 300 (only the 4600 level); cap = floor(300*0.25) = 75.
        assert result.filled == ContractCentis(75)
        assert result.consumed == (OrderBookLevel(PricePips(4600), ContractCentis(75)),)
        assert result.book_cost == MoneyMicros(345_000)  # 4600 * 75
        assert result.fee == MoneyMicros(20_000)  # ceil(1.3041) == 2 cents
        assert result.haircut == MoneyMicros(5_000)
        assert result.total_cost == MoneyMicros(370_000)

    def test_max_participation_ppm_override_allows_a_full_fill(self) -> None:
        """100% participation removes the cap entirely: full depth can fill."""
        levels = (OrderBookLevel(PricePips(4600), ContractCentis(300)),)

        result = fills.walk_taker_fill(
            levels,
            PricePips(4600),
            ContractCentis(300),
            _fee_model(),
            haircut_ppm=250_000,
            max_participation_ppm=1_000_000,
        )

        assert result.filled == ContractCentis(300)
        assert result.book_cost == MoneyMicros(1_380_000)  # 4600 * 300
        # fee: ceil(70_000 * 300 * 4600 * 5400 / 1e14) == ceil(5.2164) == 6 cents.
        assert result.fee == MoneyMicros(60_000)
        assert result.haircut == MoneyMicros(15_000)
        assert result.total_cost == MoneyMicros(1_455_000)

    def test_haircut_ppm_override_rounds_up_a_true_remainder(self) -> None:
        """A non-default haircut_ppm (33.3333%) forces a real ceiling: the fee
        model's own documented minimal-fee example (price=1, count=1,
        taker_fee_ppm=1 -> fee == 10_000 exactly, per
        `windbreak/connector/fees.py`'s worked example) times 333_333 ppm is
        3333.33 cents-of-micros, which must ceil to 3334, not floor to 3333.
        """
        levels = (OrderBookLevel(PricePips(1), ContractCentis(1)),)

        result = fills.walk_taker_fill(
            levels,
            PricePips(1),
            ContractCentis(1),
            _fee_model(taker_fee_ppm=1),
            haircut_ppm=333_333,
            max_participation_ppm=1_000_000,
        )

        assert result.filled == ContractCentis(1)
        assert result.book_cost == MoneyMicros(1)  # 1 * 1, exact
        assert result.fee == MoneyMicros(10_000)
        assert result.haircut == MoneyMicros(3_334)
        assert result.total_cost == MoneyMicros(13_335)

    def test_zero_haircut_ppm_leaves_total_cost_as_book_cost_plus_fee(self) -> None:
        levels = (OrderBookLevel(PricePips(4600), ContractCentis(1000)),)

        result = fills.walk_taker_fill(
            levels,
            PricePips(4600),
            ContractCentis(100),
            _fee_model(),
            haircut_ppm=0,
            max_participation_ppm=250_000,
        )

        assert result.haircut == MoneyMicros(0)
        assert result.total_cost == result.book_cost + result.fee

    def test_tiny_depth_that_rounds_the_cap_to_zero_yields_a_costless_zero_fill(
        self,
    ) -> None:
        """3 centis of depth * 25% floors to a cap of 0: nothing fills, and no
        fee/haircut is charged against a zero-sized trade."""
        levels = (OrderBookLevel(PricePips(4600), ContractCentis(3)),)

        result = fills.walk_taker_fill(
            levels,
            PricePips(4600),
            ContractCentis(1000),
            _fee_model(),
            haircut_ppm=250_000,
            max_participation_ppm=250_000,
        )

        assert result.filled == ContractCentis(0)
        assert result.consumed == ()
        assert result.book_cost == MoneyMicros(0)
        assert result.fee == MoneyMicros(0)
        assert result.haircut == MoneyMicros(0)
        assert result.total_cost == MoneyMicros(0)


class TestWalkTakerFillNoSide:
    """A NO buy crosses the complement-transformed `yes_bids` book.

    `walk_taker_fill` is side-agnostic: `PaperExchange.place_order` is
    responsible for complementing a NO order's limit and the yes-bid levels
    it walks (`PricePips(10_000 - p)`) before calling this function. This
    pins that convention with a hand-computed example so the NO-side glue
    code in `paper.py` has an unambiguous reference.
    """

    def test_no_buy_walks_the_complemented_yes_bids(self) -> None:
        # yes_bids, best-first (highest price first): 6000/500, then 5900/1000.
        yes_bids = (
            OrderBookLevel(PricePips(6000), ContractCentis(500)),
            OrderBookLevel(PricePips(5900), ContractCentis(1000)),
        )
        # Complementing each level (10_000 - price) yields the NO-ask book,
        # already in best-first (ascending) order because yes_bids was
        # descending: 4000/500, then 4100/1000.
        no_ask_levels = tuple(
            OrderBookLevel(PricePips(10_000 - level.price.value), level.quantity)
            for level in yes_bids
        )
        no_limit = PricePips(4050)

        result = fills.walk_taker_fill(
            no_ask_levels,
            no_limit,
            ContractCentis(1000),
            _fee_model(),
            haircut_ppm=250_000,
            max_participation_ppm=250_000,
        )

        # Only the 4000-priced level is at-or-better than 4050; its depth is
        # 500, so the 25% participation cap is 125.
        assert result.consumed == (
            OrderBookLevel(PricePips(4000), ContractCentis(125)),
        )
        assert result.filled == ContractCentis(125)
        assert result.book_cost == MoneyMicros(500_000)  # 4000 * 125
        # fee: ceil(70_000 * 125 * 4000 * 6000 / 1e14) == ceil(2.1) == 3 cents.
        assert result.fee == MoneyMicros(30_000)
        assert result.haircut == MoneyMicros(7_500)  # ceil(30_000 * 250_000 / 1e6)
        assert result.total_cost == MoneyMicros(537_500)


class TestRestingFillQuantity:
    """Pins `resting_fill_quantity`: touch != fill; trade-through, capped.

    Convention (matches a resting BUY / bid at `limit`): a print strictly
    below `limit` is a trade-through and its full quantity counts toward the
    fill; a print at or above `limit` (touch or irrelevant) contributes
    nothing. The realized fill is `min(remaining, participation_cap(
    depth_at_or_better), total trade-through volume)`.
    """

    def test_touch_is_not_a_fill(self) -> None:
        prints = (fills.TradePrint(PricePips(4200), ContractCentis(1000), _TS),)

        fill = fills.resting_fill_quantity(
            PricePips(4200),
            ContractCentis(1000),
            prints,
            ContractCentis(500),
            max_participation_ppm=250_000,
        )

        assert fill == ContractCentis(0)

    def test_prints_above_the_limit_are_irrelevant_to_a_resting_buy(self) -> None:
        prints = (fills.TradePrint(PricePips(4250), ContractCentis(5000), _TS),)

        fill = fills.resting_fill_quantity(
            PricePips(4200),
            ContractCentis(1000),
            prints,
            ContractCentis(100_000),
            max_participation_ppm=250_000,
        )

        assert fill == ContractCentis(0)

    def test_participation_cap_binds_the_trade_through_fill(self) -> None:
        """depth_at_or_better 500 -> cap 125; the print's 2000 centis and the
        1000-centis remaining both exceed that cap."""
        prints = (fills.TradePrint(PricePips(4150), ContractCentis(2000), _TS),)

        fill = fills.resting_fill_quantity(
            PricePips(4200),
            ContractCentis(1000),
            prints,
            ContractCentis(500),
            max_participation_ppm=250_000,
        )

        assert fill == ContractCentis(125)

    def test_trade_through_volume_binds_when_smaller_than_the_cap(self) -> None:
        prints = (fills.TradePrint(PricePips(4150), ContractCentis(100), _TS),)

        fill = fills.resting_fill_quantity(
            PricePips(4200),
            ContractCentis(300),
            prints,
            ContractCentis(10_000),  # cap = 2500, far above the 100-centi print
            max_participation_ppm=250_000,
        )

        assert fill == ContractCentis(100)

    def test_remaining_order_size_binds_when_it_is_the_smallest_bound(self) -> None:
        prints = (fills.TradePrint(PricePips(4100), ContractCentis(10_000), _TS),)

        fill = fills.resting_fill_quantity(
            PricePips(4200),
            ContractCentis(50),
            prints,
            ContractCentis(100_000),
            max_participation_ppm=250_000,
        )

        assert fill == ContractCentis(50)

    def test_multiple_trade_through_prints_sum_their_volume(self) -> None:
        prints = (
            fills.TradePrint(PricePips(4100), ContractCentis(50), _TS),
            fills.TradePrint(PricePips(4150), ContractCentis(30), _TS),
        )

        fill = fills.resting_fill_quantity(
            PricePips(4200),
            ContractCentis(1000),
            prints,
            ContractCentis(100_000),
            max_participation_ppm=250_000,
        )

        assert fill == ContractCentis(80)

    def test_a_touch_print_mixed_with_a_through_print_counts_only_the_through_volume(
        self,
    ) -> None:
        prints = (
            fills.TradePrint(PricePips(4200), ContractCentis(999), _TS),  # touch
            fills.TradePrint(PricePips(4150), ContractCentis(40), _TS),  # through
        )

        fill = fills.resting_fill_quantity(
            PricePips(4200),
            ContractCentis(1000),
            prints,
            ContractCentis(100_000),
            max_participation_ppm=250_000,
        )

        assert fill == ContractCentis(40)

    def test_returns_a_contract_centis(self) -> None:
        prints = (fills.TradePrint(PricePips(4150), ContractCentis(10), _TS),)

        fill = fills.resting_fill_quantity(
            PricePips(4200),
            ContractCentis(100),
            prints,
            ContractCentis(100_000),
            max_participation_ppm=250_000,
        )

        assert isinstance(fill, ContractCentis)


class TestTradeThroughFillTs:
    """Pins `trade_through_fill_ts`: a resting fill occurs at the *last* print
    that trades through the limit, and a touch print never counts."""

    def test_returns_the_latest_trade_through_prints_timestamp(self) -> None:
        early = datetime(2025, 1, 1, 0, 1, tzinfo=UTC)
        late = datetime(2025, 1, 1, 0, 2, tzinfo=UTC)
        prints = (
            fills.TradePrint(PricePips(4150), ContractCentis(30), early),
            fills.TradePrint(PricePips(4100), ContractCentis(50), late),
        )

        assert fills.trade_through_fill_ts(PricePips(4200), prints) == late

    def test_a_touch_print_is_ignored_when_choosing_the_timestamp(self) -> None:
        through = datetime(2025, 1, 1, 0, 1, tzinfo=UTC)
        touch_later = datetime(2025, 1, 1, 0, 5, tzinfo=UTC)
        prints = (
            fills.TradePrint(PricePips(4150), ContractCentis(30), through),  # through
            fills.TradePrint(PricePips(4200), ContractCentis(99), touch_later),  # touch
        )

        assert fills.trade_through_fill_ts(PricePips(4200), prints) == through

    def test_no_trade_through_print_raises_value_error(self) -> None:
        prints = (fills.TradePrint(PricePips(4200), ContractCentis(99), _TS),)  # touch

        with pytest.raises(ValueError, match="no trade-through print"):
            fills.trade_through_fill_ts(PricePips(4200), prints)


class TestWalkTakerFillSkipsZeroQuantityLevels:
    """A recorded book level with `quantity == 0` is walked past, not crashed.

    Real exchange snapshots can carry a stale, about-to-be-removed zero-size
    level; `FeeModel.max_trading_fee_micros` rejects a zero count, so the walk
    must skip such a level rather than ask the fee model to price a zero fill.
    """

    def test_a_zero_quantity_eligible_level_is_skipped_not_crashed(self) -> None:
        levels = (
            OrderBookLevel(PricePips(4300), ContractCentis(0)),  # stale, zero size
            OrderBookLevel(PricePips(4350), ContractCentis(1000)),
        )

        result = fills.walk_taker_fill(
            levels,
            PricePips(4400),
            ContractCentis(1000),
            _fee_model(),
            haircut_ppm=0,
            max_participation_ppm=1_000_000,  # full participation isolates the skip
        )

        # The zero-size 4300 level contributes no depth and is never consumed;
        # the fill comes entirely from the 4350 level.
        assert result.filled == ContractCentis(1000)
        assert [level.price for level in result.consumed] == [PricePips(4350)]
        assert result.consumed[0].quantity == ContractCentis(1000)


class TestAllocateRestingFills:
    """Pins `allocate_resting_fills`: same-side resting buys share one shrinking
    trade-through volume pool, claimed in price priority, so their fills can
    never jointly exceed the recorded trade-through volume (no double-count).
    """

    def test_a_single_request_matches_resting_fill_quantity(self) -> None:
        prints = (fills.TradePrint(PricePips(4150), ContractCentis(2000), _TS),)
        request = fills.RestingFillRequest(
            PricePips(4200), ContractCentis(1000), ContractCentis(500)
        )

        (allocated,) = fills.allocate_resting_fills(
            (request,), prints, max_participation_ppm=250_000
        )

        expected = fills.resting_fill_quantity(
            PricePips(4200),
            ContractCentis(1000),
            prints,
            ContractCentis(500),
            max_participation_ppm=250_000,
        )
        assert allocated == expected == ContractCentis(125)

    def test_a_single_trade_fills_only_the_more_aggressive_order(self) -> None:
        """One 1000-centi print through both limits, with generous depth/caps:
        the highest limit (4700) claims the whole 1000, leaving nothing for the
        4600 order -- not 1000 each (which would double the recorded volume)."""
        prints = (fills.TradePrint(PricePips(4500), ContractCentis(1000), _TS),)
        requests = (
            fills.RestingFillRequest(
                PricePips(4700), ContractCentis(1000), ContractCentis(100_000)
            ),
            fills.RestingFillRequest(
                PricePips(4600), ContractCentis(1000), ContractCentis(100_000)
            ),
        )

        allocated = fills.allocate_resting_fills(
            requests, prints, max_participation_ppm=1_000_000
        )

        assert allocated == (ContractCentis(1000), ContractCentis(0))

    def test_price_priority_is_independent_of_request_order(self) -> None:
        """The same two orders, listed 4600-then-4700, still let 4700 claim the
        pool first -- priority follows the limit price, not insertion order."""
        prints = (fills.TradePrint(PricePips(4500), ContractCentis(1000), _TS),)
        requests = (
            fills.RestingFillRequest(
                PricePips(4600), ContractCentis(1000), ContractCentis(100_000)
            ),
            fills.RestingFillRequest(
                PricePips(4700), ContractCentis(1000), ContractCentis(100_000)
            ),
        )

        allocated = fills.allocate_resting_fills(
            requests, prints, max_participation_ppm=1_000_000
        )

        assert allocated == (ContractCentis(0), ContractCentis(1000))

    def test_lower_priced_volume_is_preserved_for_the_less_aggressive_order(
        self,
    ) -> None:
        """Prints 4650x100 and 4500x100: the 4700 order (needs 50) claims from
        the highest-priced through-print first, leaving the 4500x100 -- the only
        volume the 4600 order can reach -- intact for it."""
        prints = (
            fills.TradePrint(PricePips(4650), ContractCentis(100), _TS),
            fills.TradePrint(PricePips(4500), ContractCentis(100), _TS),
        )
        requests = (
            fills.RestingFillRequest(
                PricePips(4700), ContractCentis(50), ContractCentis(100_000)
            ),
            fills.RestingFillRequest(
                PricePips(4600), ContractCentis(200), ContractCentis(100_000)
            ),
        )

        allocated = fills.allocate_resting_fills(
            requests, prints, max_participation_ppm=1_000_000
        )

        assert allocated == (ContractCentis(50), ContractCentis(100))

    def test_participation_cap_still_binds_each_order_independently(self) -> None:
        """When the participation cap (not the pool) binds, both orders fill up
        to their cap -- sharing only kicks in when trade volume is the binding
        constraint."""
        prints = (fills.TradePrint(PricePips(4150), ContractCentis(2000), _TS),)
        requests = (
            fills.RestingFillRequest(
                PricePips(4200), ContractCentis(1000), ContractCentis(500)
            ),
            fills.RestingFillRequest(
                PricePips(4180), ContractCentis(1000), ContractCentis(500)
            ),
        )

        allocated = fills.allocate_resting_fills(
            requests, prints, max_participation_ppm=250_000
        )

        # cap = 25% of 500 = 125 each; pool (2000) is never the binding constraint.
        assert allocated == (ContractCentis(125), ContractCentis(125))


# =============================================================================
# windbreak.connector.paper.PaperExchange: the replay-driven MarketConnector
# =============================================================================


class TestPaperExchangeProtocolConformance:
    """`PaperExchange` structurally satisfies `MarketConnector` (SPEC S7.2)."""

    def test_satisfies_the_market_connector_protocol(
        self, paper_exchange: PaperExchange
    ) -> None:
        assert isinstance(paper_exchange, MarketConnector)

    def test_defaults_match_the_fills_module_constants(
        self, paper_exchange: PaperExchange
    ) -> None:
        assert paper_exchange.haircut_ppm == fills.DEFAULT_FEE_HAIRCUT_PPM
        assert (
            paper_exchange.max_participation_ppm == fills.DEFAULT_MAX_PARTICIPATION_PPM
        )

    def test_from_fixture_dir_accepts_ppm_overrides(
        self, books_fixture_dir: Path
    ) -> None:
        overridden = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "deep_walk",
            haircut_ppm=0,
            max_participation_ppm=1_000_000,
        )

        assert overridden.haircut_ppm == 0
        assert overridden.max_participation_ppm == 1_000_000


class TestPaperExchangeConstructorRejectsInconsistentSemantics:
    """The #18 consistency requirement: partial fills must be per-fill records.

    A `BalanceSemantics` record whose `partial_fill_representation` is not
    `PER_FILL_RECORDS` is incompatible with a taker walk that can span
    multiple book levels (each level needs its own `Fill.price`), so
    construction must reject it rather than silently aggregating.
    """

    def test_rejects_a_fixture_whose_partial_fill_representation_is_aggregated(
        self, books_fixture_dir: Path, tmp_path: Path
    ) -> None:
        broken_dir = tmp_path / "books"
        shutil.copytree(books_fixture_dir / "deep_walk", broken_dir)
        original = json.loads(
            (books_fixture_dir / "deep_walk" / "balance_semantics.json").read_text(
                encoding="utf-8"
            )
        )
        broken = {**original, "partial_fill_representation": "AGGREGATED"}
        (broken_dir / "balance_semantics.json").write_text(
            json.dumps(broken), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="partial_fill_representation"):
            paper.PaperExchange.from_fixture_dir(broken_dir)


class TestPaperExchangeOrderBookCursor:
    """`advance()` steps the replay cursor through a session's book snapshots."""

    def test_get_order_book_starts_at_the_first_step(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")

        book = exchange.get_order_book("MKT-DEEP")

        assert [level.price.value for level in book.yes_asks] == [4600, 4700]

    def test_advance_moves_the_cursor_to_the_next_book_and_returns_true(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")

        advanced = exchange.advance()
        book = exchange.get_order_book("MKT-DEEP")

        assert advanced is True
        assert [level.price.value for level in book.yes_asks] == [4750]

    def test_advance_returns_false_once_the_session_is_exhausted(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")
        exchange.advance()  # consumes step 0's book -> now positioned at step 1

        assert exchange.advance() is False

    def test_get_order_book_unknown_ticker_raises_unknown_market_error(
        self, paper_exchange: PaperExchange
    ) -> None:
        with pytest.raises(UnknownMarketError):
            paper_exchange.get_order_book("NOT-A-REAL-TICKER")


class TestPaperExchangeCrossingOrders:
    """A crossing order fills immediately via the pessimistic taker walk."""

    def test_crossing_buy_emits_one_fill_record_per_consumed_level(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")

        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-DEEP", "yes", PricePips(4700), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        recorded = exchange.get_fills(_SINCE)
        assert [(fill.price.value, fill.quantity.value) for fill in recorded] == [
            (4600, 200),
            (4700, 100),
        ]
        assert all(
            fill.side == "yes" and fill.ticker == "MKT-DEEP" for fill in recorded
        )

    def test_crossing_buy_fills_are_stamped_at_the_book_snapshot_time(
        self, books_fixture_dir: Path
    ) -> None:
        """A taker order executes *now*, against *now's* book, so its fills are
        stamped at the current book snapshot time (unlike resting fills, which
        are stamped at the later trade print that triggers them)."""
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")

        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-DEEP", "yes", PricePips(4700), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        book_time = exchange.get_order_book("MKT-DEEP").fetched_at
        recorded = exchange.get_fills(_SINCE)
        assert recorded
        assert all(fill.ts == book_time for fill in recorded)

    def test_crossing_buy_rests_the_unfilled_remainder_at_its_own_limit(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")

        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-DEEP", "yes", PricePips(4700), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        resting = [o for o in exchange.get_open_orders() if o.ticker == "MKT-DEEP"]
        assert len(resting) == 1
        assert resting[0].price == PricePips(4700)
        assert resting[0].quantity == ContractCentis(700)  # 1000 requested - 300 filled

    def test_a_fully_absorbed_crossing_order_leaves_no_resting_remainder(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")
        exchange.advance()  # step 1's book: single ask level 4750/500

        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-DEEP", "yes", PricePips(4750), ContractCentis(100)
            ),
            approval_token=object(),
        )

        recorded = exchange.get_fills(_SINCE)
        assert [(fill.price.value, fill.quantity.value) for fill in recorded] == [
            (4750, 100)
        ]
        assert [o for o in exchange.get_open_orders() if o.ticker == "MKT-DEEP"] == []

    def test_non_crossing_buy_rests_without_any_immediate_fill(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")

        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-DEEP", "yes", PricePips(4000), ContractCentis(100)
            ),
            approval_token=object(),
        )

        assert exchange.get_fills(_SINCE) == ()
        resting = [o for o in exchange.get_open_orders() if o.ticker == "MKT-DEEP"]
        assert resting[0].quantity == ContractCentis(100)

    def test_approval_token_is_accepted_and_ignored(
        self, books_fixture_dir: Path
    ) -> None:
        """Paper trading has no real risk-kernel gate: any object is accepted."""
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")

        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-DEEP", "yes", PricePips(4000), ContractCentis(100)
            ),
            approval_token="not-a-real-approval-token",
        )

        assert len(exchange.get_open_orders()) == 1

    def test_place_order_rejects_a_malformed_intent(
        self, paper_exchange: PaperExchange
    ) -> None:
        with pytest.raises(TypeError):
            paper_exchange.place_order(object(), approval_token=object())

    def test_place_order_unknown_ticker_raises_unknown_market_error(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")

        with pytest.raises(UnknownMarketError):
            exchange.place_order(
                paper.PaperOrderIntent(
                    "NOT-A-REAL-TICKER", "yes", PricePips(4000), ContractCentis(100)
                ),
                approval_token=object(),
            )

    def test_cancel_order_removes_a_resting_order(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-DEEP", "yes", PricePips(4000), ContractCentis(100)
            ),
            approval_token=object(),
        )
        order_id = exchange.get_open_orders()[0].id

        exchange.cancel_order(order_id)

        assert exchange.get_open_orders() == ()


class TestPaperExchangeRestingOrderTouchIsNotAFill:
    """The issue's load-bearing scenario: a touch print never fills a resting
    order, even though the print's price exactly equals the resting limit."""

    def test_touch_is_not_a_fill(self, books_fixture_dir: Path) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "touch_not_fill"
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-TOUCH", "yes", PricePips(4200), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        advanced = exchange.advance()  # processes the touch-only print

        assert advanced is True
        assert exchange.get_fills(_SINCE) == ()
        resting = exchange.get_open_orders()
        assert len(resting) == 1
        assert resting[0].quantity == ContractCentis(1000)  # untouched


class TestPaperExchangeRestingOrderTradeThrough:
    """A print strictly through the resting limit fills at the order's own
    limit price (never better), capped by the recorded book's own
    at-or-better depth via the 25% participation cap."""

    def test_trade_through_fills_at_the_orders_own_limit_price(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "trade_through"
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-THROUGH", "yes", PricePips(4200), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        exchange.advance()

        recorded = exchange.get_fills(_SINCE)
        assert len(recorded) == 1
        # depth_at_or_better (the recorded 4200 bid level) is 500 -> cap 125;
        # both the print's 2000-centi volume and the order's 1000 remaining
        # exceed that cap, so the participation cap binds at exactly 125.
        assert recorded[0].price == PricePips(4200)
        assert recorded[0].quantity == ContractCentis(125)
        assert recorded[0].side == "yes"
        assert recorded[0].ticker == "MKT-THROUGH"

    def test_trade_through_leaves_the_correct_remaining_resting_quantity(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "trade_through"
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-THROUGH", "yes", PricePips(4200), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        exchange.advance()

        resting = [o for o in exchange.get_open_orders() if o.ticker == "MKT-THROUGH"]
        assert len(resting) == 1
        assert resting[0].quantity == ContractCentis(875)  # 1000 - 125 filled

    def test_trade_through_fill_is_stamped_at_the_triggering_prints_time(
        self, books_fixture_dir: Path
    ) -> None:
        """A resting fill's ``ts`` is the trade print that caused it, not the
        book snapshot time. The ``trade_through`` fixture's step-0 book is
        stamped ``00:00:00`` but its trade-through print prints at ``00:01:00``;
        the fill only became possible because of that later print, so ``Fill.ts``
        (documented as *when the fill occurred*) must reflect the print time."""
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "trade_through"
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-THROUGH", "yes", PricePips(4200), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        exchange.advance()

        recorded = exchange.get_fills(_SINCE)
        assert len(recorded) == 1
        assert recorded[0].ts == datetime(2025, 1, 1, 0, 1, 0, tzinfo=UTC)

    def test_trade_through_fill_is_visible_to_a_cursor_after_the_book(
        self, books_fixture_dir: Path
    ) -> None:
        """A consumer polling ``get_fills(since=cursor)`` with a cursor between
        the book snapshot (``00:00:00``) and the triggering print (``00:01:00``)
        still sees the fill, because it is stamped at the print time. Stamping it
        at the earlier book time would silently drop it from this poll."""
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "trade_through"
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-THROUGH", "yes", PricePips(4200), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        exchange.advance()

        cursor = datetime(2025, 1, 1, 0, 0, 30, tzinfo=UTC)
        assert len(exchange.get_fills(cursor)) == 1


class TestPaperExchangeConcurrentRestingOrders:
    """Two resting buys on one ticker share a single step's recorded trade
    volume: their fills must never jointly exceed it (SPEC S17.4 pessimism --
    a backtest can never flatter itself), and the more aggressive limit claims
    first. The `concurrent_resting` fixture prints only 100 centis through both
    resting limits while the book depth is deep enough that the participation
    cap never binds, so the trade-through volume is the binding constraint.
    """

    def test_a_single_trade_never_fills_two_orders_beyond_its_own_volume(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "concurrent_resting"
        )
        # Both rest fully (limits below the 4400 ask; no immediate taker fill).
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-CONCURRENT", "yes", PricePips(4200), ContractCentis(1000)
            ),
            approval_token=object(),
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-CONCURRENT", "yes", PricePips(4100), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        exchange.advance()  # a single 4050 x 100 print trades through both limits

        recorded = exchange.get_fills(_SINCE)
        total_filled = sum(fill.quantity.value for fill in recorded)
        # The recorded trade was 100 centis; the two orders must not conjure 200.
        assert total_filled == 100

    def test_the_more_aggressive_limit_claims_the_shared_volume_first(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "concurrent_resting"
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-CONCURRENT", "yes", PricePips(4200), ContractCentis(1000)
            ),
            approval_token=object(),
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-CONCURRENT", "yes", PricePips(4100), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        exchange.advance()

        recorded = exchange.get_fills(_SINCE)
        # Only the 4200 (more aggressive) order fills; the 4100 order gets nothing.
        assert len(recorded) == 1
        assert recorded[0].price == PricePips(4200)
        assert recorded[0].quantity == ContractCentis(100)

        remaining = {
            order.price: order.quantity for order in exchange.get_open_orders()
        }
        assert remaining[PricePips(4200)] == ContractCentis(900)  # 1000 - 100
        assert remaining[PricePips(4100)] == ContractCentis(1000)  # untouched


class TestPaperExchangeReadOnlySurface:
    """Smoke coverage for the remaining SPEC S7.2 read-only methods."""

    def test_list_markets_returns_the_fixture_market(
        self, paper_exchange: PaperExchange
    ) -> None:
        tickers = {market.ticker for market in paper_exchange.list_markets()}

        assert tickers == {"MKT-DEEP"}

    def test_get_market_unknown_ticker_raises_unknown_market_error(
        self, paper_exchange: PaperExchange
    ) -> None:
        with pytest.raises(UnknownMarketError):
            paper_exchange.get_market("NOT-A-REAL-TICKER")

    def test_get_balances_matches_the_fixture(
        self, paper_exchange: PaperExchange
    ) -> None:
        balances = paper_exchange.get_balances()

        assert balances.total == MoneyMicros(100_000_000)
        assert balances.available == MoneyMicros(100_000_000)

    def test_get_exchange_status_is_open(self, paper_exchange: PaperExchange) -> None:
        assert paper_exchange.get_exchange_status().status == "open"

    def test_get_fee_model_falls_back_to_the_default_schedule(
        self, paper_exchange: PaperExchange
    ) -> None:
        fee_model = paper_exchange.get_fee_model("MKT-DEEP")

        assert fee_model.taker_fee_ppm == 70_000

    def test_get_positions_returns_a_tuple(self, paper_exchange: PaperExchange) -> None:
        assert isinstance(paper_exchange.get_positions(), tuple)

    def test_get_open_orders_starts_empty(self, paper_exchange: PaperExchange) -> None:
        assert paper_exchange.get_open_orders() == ()

    def test_get_exchange_time_returns_the_fixture_exchange_time(
        self, paper_exchange: PaperExchange
    ) -> None:
        assert paper_exchange.get_exchange_time() == _TS

    def test_get_balance_semantics_returns_the_fixture_semantics(
        self, paper_exchange: PaperExchange
    ) -> None:
        expected = BalanceSemantics(
            open_order_collateral_in_total=OrderCollateralInTotal.INCLUDED,
            open_order_collateral_in_available=(
                OrderCollateralInAvailable.DEDUCTED_FROM_AVAILABLE
            ),
            fee_debit_timing=FeeDebitTiming.AT_EXECUTION,
            fee_rounding=FeeRounding.EXACT,
            partial_fill_representation=PartialFillRepresentation.PER_FILL_RECORDS,
            cancel_collateral_release=CancelCollateralRelease.IMMEDIATE,
            unsettled_proceeds=UnsettledProceeds.INCLUDED_IMMEDIATELY,
            halted_market_behavior=HaltedMarketBehavior.NEW_ORDERS_ACCEPTED,
        )

        assert paper_exchange.get_balance_semantics() == expected


class TestPaperExchangeAdvancePastExhaustion:
    """A cursor already exhausted by a prior `advance()` is skipped, not re-run."""

    def test_advance_after_exhaustion_keeps_returning_false_without_raising(
        self, books_fixture_dir: Path
    ) -> None:
        """`deep_walk` has exactly two steps: the first `advance()` consumes
        step 0 (returns True, one step remains), the second consumes step 1
        (returns False, cursor now == len(steps)), and a third call must hit
        the exhausted-cursor `continue` branch -- returning False again
        without touching any resting order or raising."""
        exchange = paper.PaperExchange.from_fixture_dir(books_fixture_dir / "deep_walk")

        first = exchange.advance()
        second = exchange.advance()
        third = exchange.advance()

        assert first is True
        assert second is False
        assert third is False


class TestPaperExchangeNoSideTaker:
    """A NO order crosses the complement of the recorded `yes_bids`.

    Hand-computed golden numbers (mirroring
    `TestWalkTakerFillNoSide.test_no_buy_walks_the_complemented_yes_bids`):
    `yes_bids` (6000, 500), (5900, 1000) complement to a best-first NO-ask
    book of (4000, 500), (4100, 1000). A NO buy at limit 4050 only reaches
    the 4000-complement level (depth 500 -> 25% cap 125), so the fill is
    exactly 125 at the complemented price 4000, and the remainder rests as a
    NO `OpenOrder` at the *original* limit, 4050.
    """

    def test_no_order_walks_the_complemented_yes_bids_and_rests_the_remainder(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "no_side_taker"
        )

        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-NOTAKER", "no", PricePips(4050), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        recorded = exchange.get_fills(_SINCE)
        assert [(fill.price.value, fill.quantity.value) for fill in recorded] == [
            (4000, 125)
        ]
        assert all(
            fill.side == "no" and fill.ticker == "MKT-NOTAKER" for fill in recorded
        )
        resting = [o for o in exchange.get_open_orders() if o.ticker == "MKT-NOTAKER"]
        assert len(resting) == 1
        assert resting[0].side == "no"
        assert resting[0].price == PricePips(4050)  # the order's own limit, not 4000
        assert resting[0].quantity == ContractCentis(875)  # 1000 requested - 125 filled


class TestPaperExchangeNoSideRestingFill:
    """A resting NO order fills when the complemented print trades through.

    `_no_resting_inputs` complements both the trade prints and the depth
    source (`yes_asks`, not `yes_bids`): a resting NO bid at 4200 has its
    at-or-better depth drawn from `yes_asks` levels whose complement
    (`10_000 - price`) is `>= 4200` (only the 5800 level: complement 4200,
    depth 500 -> 25% cap 125), and its trade-through volume drawn from prints
    whose complemented price is strictly below 4200 (the 5850 print
    complements to 4150, which qualifies).
    """

    def test_resting_no_order_fills_at_its_own_limit_from_complemented_depth(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "no_side_resting"
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-NORESTING", "no", PricePips(4200), ContractCentis(1000)
            ),
            approval_token=object(),
        )

        exchange.advance()

        recorded = exchange.get_fills(_SINCE)
        assert len(recorded) == 1
        assert recorded[0].price == PricePips(4200)  # the order's own limit
        assert recorded[0].quantity == ContractCentis(125)
        assert recorded[0].side == "no"
        assert recorded[0].ticker == "MKT-NORESTING"
        resting = [o for o in exchange.get_open_orders() if o.ticker == "MKT-NORESTING"]
        assert len(resting) == 1
        assert resting[0].quantity == ContractCentis(875)  # 1000 - 125 filled


class TestPaperExchangeRestingOrderFullyConsumed:
    """A trade-through fill that meets or exceeds the remaining size drops
    the order entirely: `_resting_fill` returns None rather than a
    zero-or-negative-quantity survivor."""

    def test_a_fill_covering_the_full_remainder_removes_the_resting_order(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "resting_full_consume"
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-FULLCONSUME", "yes", PricePips(4200), ContractCentis(100)
            ),
            approval_token=object(),
        )

        exchange.advance()

        recorded = exchange.get_fills(_SINCE)
        assert [(fill.price.value, fill.quantity.value) for fill in recorded] == [
            (4200, 100)
        ]
        survivors = [
            o for o in exchange.get_open_orders() if o.ticker == "MKT-FULLCONSUME"
        ]
        assert survivors == []


class TestPaperExchangeMultiTickerRestingIsolation:
    """`_process_resting_fills` only touches resting orders on its own ticker.

    A resting order on a different ticker must survive one ticker's
    processing untouched (the `order.ticker != ticker` branch), even though
    both tickers' current steps are consumed by the same `advance()` call.
    """

    def test_advancing_one_ticker_leaves_the_others_resting_order_untouched(
        self, books_fixture_dir: Path
    ) -> None:
        exchange = paper.PaperExchange.from_fixture_dir(
            books_fixture_dir / "two_ticker_isolation"
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-ISO-A", "yes", PricePips(4200), ContractCentis(100)
            ),
            approval_token=object(),
        )
        exchange.place_order(
            paper.PaperOrderIntent(
                "MKT-ISO-B", "yes", PricePips(3000), ContractCentis(200)
            ),
            approval_token=object(),
        )

        exchange.advance()

        # Ticker A's order trades through fully (depth 500 -> cap 125 >= 100
        # remaining, through-volume 2000 >= 100) and is dropped.
        resting_a = [o for o in exchange.get_open_orders() if o.ticker == "MKT-ISO-A"]
        assert resting_a == []
        # Ticker B has no trade prints at all: its resting order must be
        # completely unaffected by ticker A's processing in the same call.
        resting_b = [o for o in exchange.get_open_orders() if o.ticker == "MKT-ISO-B"]
        assert len(resting_b) == 1
        assert resting_b[0].quantity == ContractCentis(200)  # untouched
        recorded = exchange.get_fills(_SINCE)
        assert [(f.ticker, f.price.value, f.quantity.value) for f in recorded] == [
            ("MKT-ISO-A", 4200, 100)
        ]


# --- Issue #342: the status observation is stamped at read time -------------


def test_get_exchange_status_stamps_fetched_at_from_the_injected_clock(
    books_fixture_dir: Path,
) -> None:
    """`get_exchange_status` reports *when it was observed*, not the fixture literal.

    The committed fixtures all carry `status_fetched_at: 2025-01-01T00:00:00Z`,
    which is far outside any sane freshness TTL. Returning that verbatim would
    make `exchange_status_ok` veto forever on a healthy exchange. The paper
    exchange answers synchronously and in-process, so the observation genuinely
    did occur now -- stamping it is honest, and it mirrors
    `KalshiConnector.get_exchange_status`, which likewise does not trust the
    venue's own timestamp.

    Args:
        books_fixture_dir: The shared books-fixture root.
    """
    observed_at = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    exchange = paper.PaperExchange.from_fixture_dir(
        books_fixture_dir / "deep_walk", clock=lambda: observed_at
    )

    assert exchange.get_exchange_status().fetched_at == observed_at


def test_get_exchange_status_keeps_the_fixture_status_value(
    books_fixture_dir: Path, tmp_path: Path
) -> None:
    """The status *value* is never synthesized -- only its timestamp is stamped.

    This is the line between honest and fabricated evidence: hardcoding `open`
    would make the unhealthy-exchange veto unreachable, so a `paused` fixture
    must still report `paused`.

    Args:
        books_fixture_dir: The shared books-fixture root.
        tmp_path: Scratch directory for the mutated copy.
    """
    paused_dir = tmp_path / "paused_books"
    shutil.copytree(books_fixture_dir / "deep_walk", paused_dir)
    exchange_json = paused_dir / "exchange.json"
    payload = json.loads(exchange_json.read_text(encoding="utf-8"))
    payload["status"] = "paused"
    exchange_json.write_text(json.dumps(payload), encoding="utf-8")

    exchange = paper.PaperExchange.from_fixture_dir(paused_dir)

    assert exchange.get_exchange_status().status == "paused"
