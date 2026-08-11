"""Gate 1 RED tests for `windbreak.selector.edge` (issue #44, SPEC S9.2).

`windbreak/selector/edge.py` does not exist yet, so every test below fails
collection with `ModuleNotFoundError: No module named 'windbreak.selector.edge'`
-- the expected Gate 1 RED state for issue #44's edge-figures seam. Once the
implementation specialist lands `edge.py`, these tests pin its exact contract:

`compute_executable_edge(order_book, size, forecast, fee_model, slippage_model)`
walks the book's `yes_asks` best-first, taking `min(remaining, level.quantity)`
at each level and accumulating the *exact* micros `level.price.value *
taken.value` (no rounding: 1e-4 $ * 1e-2 contracts = 1e-6 $ = micros exactly),
until either `size` is filled or the book runs out -- in which case it returns
`InsufficientDepth` naming the shortfall, never raises. On a successful walk it
returns `EdgeFigures`, the signed-ppm-of-$1-per-contract figures chained
through SPEC S9.2's formulas:

    executable_price_pips = ceil(total_cost_micros / size_centis)
    executable_price_ppm  = ceil(total_cost_micros * 100 / size_centis)
    gross_edge_ppm         = probability_ppm - executable_price_ppm
    fee_ppm                = ceil(fee_micros * 100 / size_centis)
    fee_adjusted_edge_ppm  = gross_edge_ppm - fee_ppm
    net_edge_ppm           = fee_adjusted_edge_ppm - per_contract_buffer_ppm
    annualized_expected_return_ppm =
        floor(net_edge_ppm * 1_000_000 * 8760
              / (executable_price_ppm * forecast_horizon_hours))

Issue #483 removed the sixth line the chain used to carry -- a
`research_ppm = ceil(research_cost_micros * 100 / size_centis)` haircut between
the slippage buffer and the annualization. Research is a per-forecast cost and
this is a per-contract edge, and `select`'s entry probe prices a single
contract, so that subtraction made `net_edge_min` unreachable for every market
at every price. `test_research_cost_micros_moves_no_figure` pins its absence
against the real charge the pipeline books.

Every expected number below is hand-computed in integer arithmetic in each
test's own comment/docstring -- never derived by calling
`compute_executable_edge` itself -- so a wrong rounding direction anywhere in
the chain fails the assertion for the right reason.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from windbreak.connector.fees import FeeModel
from windbreak.connector.models import OrderBookLevel, OrderBookSnapshot
from windbreak.forecast.budget import FULL_PIPELINE_RESEARCH_COST_MICROS
from windbreak.forecast.records import ForecastRecord
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips
from windbreak.selector.edge import (
    EdgeFigures,
    InsufficientDepth,
    NonAnnualizable,
    compute_executable_edge,
)
from windbreak.selector.types import FeeModelInput, SlippageModelInput

if TYPE_CHECKING:
    from collections.abc import Iterable

#: A fixed reference instant reused by every fixture below; `compute_executable_
#: edge` reads no clock (SPEC S9.1), so its exact value is arbitrary but must be
#: consistent for a given test.
_T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)


def _order_book(levels: Iterable[tuple[int, int]]) -> OrderBookSnapshot:
    """Build an `OrderBookSnapshot` whose `yes_asks` are the given levels.

    Args:
        levels: `(price_pips, quantity_centis)` pairs, best-first.

    Returns:
        The constructed order-book snapshot, with empty `yes_bids`.
    """
    asks = tuple(
        OrderBookLevel(price=PricePips(price), quantity=ContractCentis(qty))
        for price, qty in levels
    )
    return OrderBookSnapshot(
        ticker="EDGE-TICKER", yes_bids=(), yes_asks=asks, fetched_at=_T0
    )


def _forecast(**overrides: object) -> ForecastRecord:
    """Build a `ForecastRecord` for edge-computation tests, with sane defaults.

    Args:
        **overrides: Field values overriding the defaults below.

    Returns:
        The constructed, post-init-validated `ForecastRecord`. Fields the edge
        computation never reads (citations, model votes, CI bounds, ...) are
        given innocuous placeholder values.
    """
    defaults: dict[str, object] = {
        "forecast_id": "fc-edge-0001",
        "market_ticker": "EDGE-TICKER",
        "normalized_question_hash": "sha256:edge-question",
        "probability_ppm": 650_000,
        "ci_low_ppm": 100_000,
        "ci_high_ppm": 200_000,
        "model_votes": (),
        "vote_dispersion_ppm": 0,
        "rationale_markdown": "n/a",
        "citations": (),
        "source_quality_notes": (),
        "research_cost_micros": 0,
        "triage_stage": "full",
        "created_at": _T0,
        "forecast_horizon_hours": 48,
        "market_price_baseline_pips": 5_000,
        "baseline_quote_snapshot_id": "snap-edge-0001",
        "coherence_group_sum_ppm": None,
        "coherence_flag": False,
        "abstention_reason": None,
        "eligible_for_live": True,
    }
    defaults.update(overrides)
    return ForecastRecord(**defaults)


def _fee_model_input(
    *, taker_fee_ppm: int, settlement_fee_ppm: int = 0
) -> FeeModelInput:
    """Build a `FeeModelInput` with a given taker rate (maker pinned at 0).

    Args:
        taker_fee_ppm: The schedule's taker rate, in ppm.
        settlement_fee_ppm: The schedule's settlement rate, in ppm.

    Returns:
        The constructed `FeeModelInput`, `as_of` pinned to `_T0`.
    """
    model = FeeModel(
        schedule_id="edge-test-fee",
        maker_fee_ppm=0,
        taker_fee_ppm=taker_fee_ppm,
        settlement_fee_ppm=settlement_fee_ppm,
    )
    return FeeModelInput(model=model, as_of=_T0)


def _slippage_input(*, per_contract_buffer_ppm: int) -> SlippageModelInput:
    """Build a `SlippageModelInput` with a given per-contract buffer.

    Args:
        per_contract_buffer_ppm: The buffer to subtract, in ppm.

    Returns:
        The constructed `SlippageModelInput`.
    """
    return SlippageModelInput(
        model_id="edge-test-slippage", per_contract_buffer_ppm=per_contract_buffer_ppm
    )


# --- The canonical VWAP-not-midpoint walk ------------------------------------


def test_canonical_walk_yields_vwap_not_midpoint() -> None:
    """The two-level (4500@10_000, 4700@10_000) walk over size 15_000 proves
    the executable price is the size-weighted VWAP (4567), never the midpoint
    (4600) nor either level's own price (4500 or 4700).

    Hand computation:
        cost = 4500*10_000 + 4700*5_000 = 45_000_000 + 23_500_000
             = 68_500_000 micros (exact: price.value * count.value at every
               level, no rounding until the final ceil-division)
        executable_price_pips = ceil(68_500_000 / 15_000)
                               = ceil(4566.6667) = 4567
        executable_price_ppm  = ceil(68_500_000*100 / 15_000)
                               = ceil(456_666.667) = 456_667
        gross_edge_ppm = probability_ppm(620_000) - 456_667 = 163_333
    """
    book = _order_book([(4500, 10_000), (4700, 10_000)])
    forecast = _forecast(probability_ppm=620_000)
    fee_model = _fee_model_input(taker_fee_ppm=0)
    slippage = _slippage_input(per_contract_buffer_ppm=0)

    result = compute_executable_edge(
        order_book=book,
        size=ContractCentis(15_000),
        forecast=forecast,
        fee_model=fee_model,
        slippage_model=slippage,
    )

    assert isinstance(result, EdgeFigures)
    assert result.executable_cost_micros == MoneyMicros(68_500_000)
    assert result.executable_price_pips == PricePips(4567)
    assert result.executable_price_pips != PricePips(4566)
    assert result.executable_price_pips != PricePips(4600)  # not the midpoint
    assert result.marginal_price_pips == PricePips(4700)  # deepest level walked
    assert result.gross_edge_ppm == 163_333
    # Zero fee and zero slippage: every downstream figure equals gross edge.
    assert result.fee_adjusted_edge_ppm == 163_333
    assert result.net_edge_ppm == 163_333


# --- Single-level exact fill (no remainder) ----------------------------------


def test_single_level_exact_fill_has_no_remainder() -> None:
    """A size exactly filled within one level divides its cost evenly:
    5000 pips * 1000 centis = 5_000_000 micros / 1000 centis = 5000 pips exact.
    """
    book = _order_book([(5_000, 1_000)])
    forecast = _forecast(probability_ppm=650_000)
    fee_model = _fee_model_input(taker_fee_ppm=0)
    slippage = _slippage_input(per_contract_buffer_ppm=0)

    result = compute_executable_edge(
        order_book=book,
        size=ContractCentis(1_000),
        forecast=forecast,
        fee_model=fee_model,
        slippage_model=slippage,
    )

    assert isinstance(result, EdgeFigures)
    assert result.executable_cost_micros == MoneyMicros(5_000_000)
    assert result.executable_price_pips == PricePips(5_000)
    assert result.marginal_price_pips == PricePips(5_000)


# --- Exact-boundary fill (size == total depth, spanning two levels) ---------


def test_exact_boundary_fill_consumes_every_level_fully() -> None:
    """Size exactly equal to total depth across two levels fills both levels
    fully -- not `InsufficientDepth`, since depth == size is the boundary, not
    a shortfall -- with the marginal price at the deepest (second) level.

    cost = 4000*300 + 4100*200 = 1_200_000 + 820_000 = 2_020_000 micros
    executable_price_pips = ceil(2_020_000 / 500) = 4040 (exact: 500*4040
        == 2_020_000, no remainder)
    """
    book = _order_book([(4_000, 300), (4_100, 200)])
    forecast = _forecast(probability_ppm=650_000)
    fee_model = _fee_model_input(taker_fee_ppm=0)
    slippage = _slippage_input(per_contract_buffer_ppm=0)

    result = compute_executable_edge(
        order_book=book,
        size=ContractCentis(500),
        forecast=forecast,
        fee_model=fee_model,
        slippage_model=slippage,
    )

    assert isinstance(result, EdgeFigures)
    assert result.executable_cost_micros == MoneyMicros(2_020_000)
    assert result.executable_price_pips == PricePips(4_040)
    assert result.marginal_price_pips == PricePips(4_100)


# --- InsufficientDepth: size exceeds available depth -------------------------


def test_insufficient_depth_when_size_exceeds_total_ask_depth() -> None:
    """A size larger than the total resting ask depth (100 available, 150
    requested) returns `InsufficientDepth` naming the exact required/available
    centis counts and the pinned reason string -- never raises.
    """
    book = _order_book([(5_000, 100)])
    forecast = _forecast()
    fee_model = _fee_model_input(taker_fee_ppm=0)
    slippage = _slippage_input(per_contract_buffer_ppm=0)

    result = compute_executable_edge(
        order_book=book,
        size=ContractCentis(150),
        forecast=forecast,
        fee_model=fee_model,
        slippage_model=slippage,
    )

    assert isinstance(result, InsufficientDepth)
    assert result.required_centis == 150
    assert result.available_centis == 100
    assert result.reason == "insufficient_book_depth: required=150 available=100"


def test_insufficient_depth_when_book_has_no_asks() -> None:
    """An empty `yes_asks` book has zero available depth: still a named
    `InsufficientDepth`, not a crash on an empty sequence.
    """
    book = _order_book([])
    forecast = _forecast()
    fee_model = _fee_model_input(taker_fee_ppm=0)
    slippage = _slippage_input(per_contract_buffer_ppm=0)

    result = compute_executable_edge(
        order_book=book,
        size=ContractCentis(50),
        forecast=forecast,
        fee_model=fee_model,
        slippage_model=slippage,
    )

    assert isinstance(result, InsufficientDepth)
    assert result.required_centis == 50
    assert result.available_centis == 0
    assert result.reason == "insufficient_book_depth: required=50 available=0"


# --- NonAnnualizable: a zero-hour forecast horizon --------------------------


def test_non_annualizable_when_forecast_horizon_is_zero() -> None:
    """A book that fills the size fully (positive executable price) paired with
    a `forecast_horizon_hours=0` forecast returns `NonAnnualizable`, never
    raises `ZeroDivisionError`: the fill priced fine, only the annualization
    denominator (`executable_price_ppm * horizon_hours`) is zero because the
    horizon factor is zero.

    Hand computation (size=100 centis, single level 5000 pips/100 centis):
        cost = 5000*100 = 500_000 micros (exact)
        executable_price_ppm = ceil(500_000*100 / 100) = 500_000 (exact:
            dividing by size_centis=100 always exactly cancels the *100 scale
            factor, so this per-contract-sized fill has no remainder)
    """
    book = _order_book([(5_000, 100)])
    forecast = _forecast(forecast_horizon_hours=0)
    fee_model = _fee_model_input(taker_fee_ppm=0)
    slippage = _slippage_input(per_contract_buffer_ppm=0)

    result = compute_executable_edge(
        order_book=book,
        size=ContractCentis(100),
        forecast=forecast,
        fee_model=fee_model,
        slippage_model=slippage,
    )

    expected_reason = "non_annualizable: executable_price_ppm=500000 horizon_hours=0"

    assert isinstance(result, NonAnnualizable)
    assert result.executable_price_ppm == 500_000
    assert result.forecast_horizon_hours == 0
    assert result.reason == expected_reason


# --- NonAnnualizable: a zero-pip (fully free) executable price ---------------


def test_non_annualizable_when_executable_price_is_zero() -> None:
    """A book whose sole walked level prices at `PricePips(0)` (permitted --
    `fees.py` accepts the inclusive `[0, 10_000]` pip range) fills the size
    fully at zero cost, so `executable_price_ppm == 0` with a positive horizon:
    still a decidable `NonAnnualizable`, never a `ZeroDivisionError`, this time
    because the price factor of the annualization denominator is zero rather
    than the horizon factor (the companion to the zero-horizon case above).

    Hand computation (size=100 centis, single level 0 pips/100 centis):
        cost = 0*100 = 0 micros (exact)
        executable_price_ppm = ceil(0*100 / 100) = 0
    """
    book = _order_book([(0, 100)])
    forecast = _forecast(forecast_horizon_hours=48)
    fee_model = _fee_model_input(taker_fee_ppm=0)
    slippage = _slippage_input(per_contract_buffer_ppm=0)

    result = compute_executable_edge(
        order_book=book,
        size=ContractCentis(100),
        forecast=forecast,
        fee_model=fee_model,
        slippage_model=slippage,
    )

    expected_reason = "non_annualizable: executable_price_ppm=0 horizon_hours=48"

    assert isinstance(result, NonAnnualizable)
    assert result.executable_price_ppm == 0
    assert result.forecast_horizon_hours == 48
    assert result.reason == expected_reason


# --- The full chain, engineered for a nonzero remainder at every single -----
# --- division -- a wrong rounding direction anywhere fails. -----------------


def test_full_chain_with_nonzero_remainders_pins_rounding_direction() -> None:
    """Every ceil/floor division in the chain has a nonzero remainder here, so
    a wrong rounding direction anywhere changes at least one asserted figure.

    Book walk (size=300 centis over two levels):
        level 1: 200 centis @ 5000 pips -> 1_000_000 micros
        level 2: 100 centis @ 5200 pips ->   520_000 micros
        total cost = 1_520_000 micros; marginal level = 5200 pips (deepest)

        executable_price_pips = ceil(1_520_000 / 300)
                               = ceil(5066.667) = 5067   (300*5066=1_519_800,
                                 remainder 200)
        executable_price_ppm  = ceil(1_520_000*100 / 300)
                               = ceil(506_666.667) = 506_667   (300*506_666=
                                 151_999_800, remainder 200)

    Forecast probability_ppm = 650_000:
        gross_edge_ppm = 650_000 - 506_667 = 143_333

    Fee model: taker_fee_ppm=80_000, settlement_fee_ppm=17, at
    price_pips=5067, count_centis=300:
        trading numerator = 80_000*300*5067*(10_000-5067)
                           = 80_000*300*5067*4933 = 599_892_264_000_000
        trading cents = ceil(599_892_264_000_000 / 1e14) = 6 (5.999 -> 6)
                       -> 60_000 micros
        settlement numerator = 17*300 = 5_100
        settlement cents = ceil(5_100 / 1e6) = 1 -> 10_000 micros
        fee_micros = 60_000 + 10_000 = 70_000
        fee_ppm = ceil(70_000*100 / 300) = ceil(23_333.33) = 23_334
                 (300*23_333=6_999_900, remainder 100)
        fee_adjusted_edge_ppm = 143_333 - 23_334 = 119_999

    Slippage buffer = 4_999 (plain subtraction, no rounding):
        net_edge_ppm = 119_999 - 4_999 = 115_000

    The forecast still carries `research_cost_micros=778`, and every figure
    above is computed as though it did not: since issue #483 the chain has no
    research term, so that 778 must move nothing. Were the old
    `research_ppm = ceil(778*100 / 300) = 260` haircut still applied, the net
    edge would read 114_740 and the annualized figure 19_837_929 -- both
    asserted *not* to be the answer below, so a reinstated haircut fails here
    rather than shifting a number nobody re-derived.

    Annualized (forecast_horizon_hours=100):
        annualized = floor(115_000 * 1_000_000 * 8760 / (506_667 * 100))
                   = floor(1_007_400_000_000_000 / 50_666_700)
                   = 19_882_881
        (verified: 50_666_700*19_882_881 + 33_237_300 == 1_007_400_000_000_000,
         with remainder 33_237_300 < 50_666_700)
    """
    book = _order_book([(5_000, 200), (5_200, 100)])
    forecast = _forecast(
        probability_ppm=650_000,
        research_cost_micros=778,
        forecast_horizon_hours=100,
    )
    fee_model = _fee_model_input(taker_fee_ppm=80_000, settlement_fee_ppm=17)
    slippage = _slippage_input(per_contract_buffer_ppm=4_999)

    result = compute_executable_edge(
        order_book=book,
        size=ContractCentis(300),
        forecast=forecast,
        fee_model=fee_model,
        slippage_model=slippage,
    )

    assert isinstance(result, EdgeFigures)
    assert result.executable_cost_micros == MoneyMicros(1_520_000)
    assert result.executable_price_pips == PricePips(5_067)
    assert result.marginal_price_pips == PricePips(5_200)
    assert result.gross_edge_ppm == 143_333
    assert result.fee_adjusted_edge_ppm == 119_999
    assert result.net_edge_ppm == 115_000
    assert result.net_edge_ppm != 114_740
    assert result.annualized_expected_return_ppm == 19_882_881
    assert result.annualized_expected_return_ppm != 19_837_929


def test_research_cost_micros_moves_no_figure() -> None:
    """Every figure is identical whatever `research_cost_micros` carries (#483).

    The charge compared is the production constant the full pipeline actually
    books, imported from its definition site rather than restated, so the seam
    and the composition cannot disagree again. Whole-dataclass equality is
    asserted rather than the net edge alone: a research term reintroduced into
    only the annualized return would survive a narrower check.

    The zero-cost figures are asserted to be a real priced fill first, so the
    equality cannot hold vacuously by both sides declining.
    """
    book = _order_book([(5_000, 200), (5_200, 100)])
    fee_model = _fee_model_input(taker_fee_ppm=80_000, settlement_fee_ppm=17)
    slippage = _slippage_input(per_contract_buffer_ppm=4_999)

    figures = [
        compute_executable_edge(
            order_book=book,
            size=ContractCentis(300),
            forecast=_forecast(
                probability_ppm=650_000,
                research_cost_micros=cost,
                forecast_horizon_hours=100,
            ),
            fee_model=fee_model,
            slippage_model=slippage,
        )
        for cost in (0, FULL_PIPELINE_RESEARCH_COST_MICROS)
    ]

    assert isinstance(figures[0], EdgeFigures)
    assert figures[0].net_edge_ppm == 115_000
    assert figures[0] == figures[1]


# --- Annualization: 8760/horizon scaling, and the negative-edge floor -------


def test_annualized_expected_return_scales_inversely_with_horizon() -> None:
    """Doubling `forecast_horizon_hours` (48 -> 96) exactly halves the
    annualized figure (both divide evenly here), pinning the
    `8760 / forecast_horizon_hours` scaling relationship.

    net_edge_ppm = probability_ppm(600_000) - executable_price_ppm(500_000)
                 = 100_000 (zero fee/slippage/research)
    annualized(48) = floor(100_000*1_000_000*8760 / (500_000*48))
                    = floor(876_000_000_000_000 / 24_000_000) = 36_500_000
                    (exact: 24_000_000*36_500_000 == 876_000_000_000_000)
    annualized(96) = floor(876_000_000_000_000 / 48_000_000) = 18_250_000
                    (exact, exactly half of the 48h figure)
    """
    book = _order_book([(5_000, 100)])
    fee_model = _fee_model_input(taker_fee_ppm=0)
    slippage = _slippage_input(per_contract_buffer_ppm=0)

    short_horizon = compute_executable_edge(
        order_book=book,
        size=ContractCentis(100),
        forecast=_forecast(probability_ppm=600_000, forecast_horizon_hours=48),
        fee_model=fee_model,
        slippage_model=slippage,
    )
    long_horizon = compute_executable_edge(
        order_book=book,
        size=ContractCentis(100),
        forecast=_forecast(probability_ppm=600_000, forecast_horizon_hours=96),
        fee_model=fee_model,
        slippage_model=slippage,
    )

    assert isinstance(short_horizon, EdgeFigures)
    assert isinstance(long_horizon, EdgeFigures)
    assert short_horizon.annualized_expected_return_ppm == 36_500_000
    assert long_horizon.annualized_expected_return_ppm == 18_250_000


def test_annualized_expected_return_floors_toward_negative_infinity() -> None:
    """A negative net edge over a non-evenly-dividing denominator floors
    toward negative infinity (Python `//` semantics), never truncates toward
    zero.

    Book: single ask 6000 pips / 100 centis, size=100 -> cost = 600_000 micros
    exact; executable_price_ppm = 600_000 exact. probability_ppm=100_000 (far
    below price) with zero fee/slippage/research:
        gross_edge_ppm = 100_000 - 600_000 = -500_000 = net_edge_ppm

    forecast_horizon_hours=7:
        numerator   = -500_000 * 1_000_000 * 8760 = -4_380_000_000_000_000
        denominator = 600_000 * 7 = 4_200_000
        exact fraction = -1_042_857_142.857142...
        floor -> -1_042_857_143 (NOT -1_042_857_142, which would be
                 truncation-toward-zero -- the exact bug this test guards
                 against)
        verified: 4_200_000*(-1_042_857_143) == -4_380_000_000_600_000 <=
                  -4_380_000_000_000_000 < 4_200_000*(-1_042_857_142) ==
                  -4_379_999_996_400_000
    """
    book = _order_book([(6_000, 100)])
    forecast = _forecast(probability_ppm=100_000, forecast_horizon_hours=7)
    fee_model = _fee_model_input(taker_fee_ppm=0)
    slippage = _slippage_input(per_contract_buffer_ppm=0)

    result = compute_executable_edge(
        order_book=book,
        size=ContractCentis(100),
        forecast=forecast,
        fee_model=fee_model,
        slippage_model=slippage,
    )

    assert isinstance(result, EdgeFigures)
    assert result.net_edge_ppm == -500_000
    assert result.annualized_expected_return_ppm == -1_042_857_143
    assert result.annualized_expected_return_ppm != -1_042_857_142


#: The SPEC document whose §9.2 figure list must match `EdgeFigures`.
_SPEC_PATH = Path(__file__).resolve().parents[2] / "plans" / "SPEC_v3.md"

#: The §9.2 sentence whose backticked figure names are the SPEC's claim about
#: what the selector computes.
_SPEC_FIGURES_PREFIX = "The selector walks the actual book to the proposed size"


def _spec_figure_names() -> list[str]:
    """Return the figure names SPEC §9.2 says the selector computes.

    Returns:
        The backticked identifiers from §9.2's opening sentence, in order.
    """
    sentences = [
        line
        for line in _SPEC_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith(_SPEC_FIGURES_PREFIX)
    ]
    assert len(sentences) == 1, f"expected one §9.2 sentence, got {len(sentences)}"
    return re.findall(r"`([a-z_]+)`", sentences[0])


def test_spec_9_2_names_exactly_the_figures_the_chain_computes() -> None:
    """SPEC §9.2's figure list and `EdgeFigures`' own fields are one list.

    Nothing enforced this before, and it drifted: §9.2 went on listing
    `research_cost_adjusted_edge` after issue #483 deleted the figure, so the
    document that agent briefs and reviewers read as authoritative described a
    haircut the code no longer applies -- the phantom-gate failure mode this
    repo has closed four times in its own tooling.

    The comparison runs both ways, as a set equality, so a figure added to the
    chain without a SPEC update fails just as loudly as a figure removed from
    it. The `_ppm` suffix is the code's unit convention rather than part of the
    figure's name, so it is bridged here and nowhere else.
    """
    documented = {f"{name}_ppm" for name in _spec_figure_names()}

    computed = {
        field.name
        for field in dataclasses.fields(EdgeFigures)
        if field.name.endswith(("_edge_ppm", "_return_ppm"))
    }

    assert documented == computed
    assert "research_cost_adjusted_edge_ppm" not in computed
    assert len(computed) == 4
