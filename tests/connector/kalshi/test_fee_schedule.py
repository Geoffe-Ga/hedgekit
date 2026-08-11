"""Checks `FeeModel` against Kalshi's published fee schedule (issue #452).

Issue #452 asked whether `FeeModel.max_trading_fee_micros`' single quadratic
form plus `max(maker_fee_ppm, taker_fee_ppm)` can *understate* a worst-case
fee -- which it can, but only for a venue whose maker and taker fees have
different functional forms. This module answers that question against the
venue rather than against memory.

`tests/fixtures/exchange/kalshi/fee_schedule_published.json` transcribes
Kalshi's own fee-schedule PDF (effective 2026-02-05). Every trading fee on it
-- taker, maker, and the S&P500/Nasdaq-100 special rate -- is the *same*
quadratic `round up(rate x C x P x (1-P))`, differing only in `rate`. Because
one shared, rate-monotone form makes "the fee of the higher rate" and "the
higher of the two fees" the same number, `max(maker_fee_ppm, taker_fee_ppm)`
is a sound upper bound for this venue, and issue #452's understatement cannot
occur while that stays true.

The tests below make that finding load-bearing rather than prose:

    * the published fee tables are replayed row by row against `FeeModel`, so
      a change to the rate, the form, the scale, or the rounding breaks a test
      that cites a printed number;
    * the max-of-computed-fees semantics issue #452 asks for is asserted
      directly, so a bound that took the *lower* of the two fees would fail
      even though the existing suite's equal-outcome test would not;
    * the shared-form premise itself is asserted against the recorded
      formulas, so the day Kalshi publishes a second form, this test -- not a
      live order -- is what notices.

Every assertion uses `*`, `+`, `-`, `//` and integer comparison only; there is
no `/` and no float literal anywhere in this module.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from windbreak.connector.fees import FeeModel
from windbreak.connector.snapshot import InMemoryEventLedgerWriter
from windbreak.connector.validation import (
    SCHEMA_ANOMALY_EVENT,
    SchemaAnomalyHaltError,
    SchemaValidator,
    kalshi_default_schema_registry,
)

#: The transcribed fee-schedule record under test.
_SCHEDULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "exchange"
    / "kalshi"
    / "fee_schedule_published.json"
)

#: Pips per cent: a price in pips is 1e-4 dollars, a cent is 1e-2 dollars.
_PIPS_PER_CENT = 100

#: Contract-centis per whole contract (a centi is 1e-2 contracts).
_CENTIS_PER_CONTRACT = 100

#: Micros per cent: a micro is 1e-6 dollars, a cent 1e-2 dollars.
_MICROS_PER_CENT = 10_000

#: One basis point is 100 ppm, matching the adapter's `_bps_to_ppm`.
_PPM_PER_BPS = 100

#: The denominator taking `rate_ppm * centis * pips * pips` to whole cents.
_CENT_DENOMINATOR = 10**14

#: Row count both published tables must have, asserted before any row-wise
#: claim so a truncated or empty table can never make those claims vacuous.
_PUBLISHED_ROW_COUNT = 21


def _load_schedule() -> dict[str, Any]:
    """Return the transcribed published fee schedule as a mapping."""
    with _SCHEDULE_PATH.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


#: The transcribed record, loaded once at import so parametrization can read it.
_SCHEDULE = _load_schedule()

#: Published rates, in ppm, keyed by the schedule's own formula names.
_TAKER_PPM = _SCHEDULE["formulas"]["taker"]["rate_bps"] * _PPM_PER_BPS
_MAKER_PPM = _SCHEDULE["formulas"]["maker"]["rate_bps"] * _PPM_PER_BPS
_INDEX_PPM = _SCHEDULE["formulas"]["index"]["rate_bps"] * _PPM_PER_BPS


def _model(*, maker_ppm: int, taker_ppm: int) -> FeeModel:
    """Build a `FeeModel` with explicit maker and taker rates in ppm."""
    return FeeModel(
        schedule_id="kalshi-published-2026-02-05",
        maker_fee_ppm=maker_ppm,
        taker_fee_ppm=taker_ppm,
        settlement_fee_ppm=0,
    )


def _quadratic_fee_micros(rate_ppm: int, price_pips: int, count_centis: int) -> int:
    """Return `round up(rate x C x P x (1-P))` in micros, by integer math.

    This is the schedule's printed formula re-derived independently of
    `FeeModel`, so a max-of-fees claim can be checked without asking
    `FeeModel` what each individual fee is. `-(-n // d)` is the exact integer
    ceiling for a non-negative numerator.

    Args:
        rate_ppm: The fee rate, in parts-per-million.
        price_pips: The contract price, in pips.
        count_centis: The size, in contract-centis.

    Returns:
        The ceiling-to-the-cent fee, in micros.
    """
    numerator = rate_ppm * count_centis * price_pips * (10_000 - price_pips)
    return -(-numerator // _CENT_DENOMINATOR) * _MICROS_PER_CENT


def _rows(table_key: str) -> list[dict[str, int]]:
    """Return one published fee table's rows.

    Args:
        table_key: The schedule key naming the table.

    Returns:
        The table's rows, each a mapping of price and both fee columns.
    """
    rows: list[dict[str, int]] = _SCHEDULE[table_key]
    return rows


# --- the recorded schedule itself ---------------------------------------------


def test_recorded_schedule_names_its_source_and_effective_date() -> None:
    """The record cites the venue's own PDF and the date it took effect."""
    assert _SCHEDULE["source_url"] == "https://kalshi.com/docs/kalshi-fee-schedule.pdf"
    assert _SCHEDULE["effective_date"] == "2026-02-05"
    assert _SCHEDULE["rounding"] == "round up = rounds to the next cent"


@pytest.mark.parametrize(
    "table_key", ["general_trading_fees_table", "index_trading_fees_table"]
)
def test_each_published_table_has_every_printed_row(table_key: str) -> None:
    """Both tables carry all 21 printed prices, so no row-wise test is vacuous."""
    rows = _rows(table_key)

    assert len(rows) == _PUBLISHED_ROW_COUNT
    assert [row["price_cents"] for row in rows] == [
        1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99
    ]  # fmt: skip


def test_every_published_trading_fee_shares_one_quadratic_form() -> None:
    """Taker, maker and index fees are all `rate x C x P x (1-P)`; settlement is none.

    This is the premise issue #452 turns on: one shared, rate-monotone form is
    exactly what makes `max(maker_fee_ppm, taker_fee_ppm)` a sound bound. The
    forms are read off the record rather than restated, so a second form
    appearing in the schedule fails here.
    """
    forms = {name: block["form"] for name, block in _SCHEDULE["formulas"].items()}

    assert forms == {
        "taker": "quadratic",
        "maker": "quadratic",
        "index": "quadratic",
        "settlement": "none",
    }


def test_the_published_maker_rate_is_below_the_published_taker_rate() -> None:
    """Maker is a quarter of taker (175 bps vs 700 bps), both strictly positive."""
    assert _MAKER_PPM == 17_500
    assert _TAKER_PPM == 70_000
    assert _INDEX_PPM == 35_000
    assert _MAKER_PPM < _TAKER_PPM


def test_the_published_schedule_charges_no_settlement_fee() -> None:
    """The schedule states there is no settlement fee, so its rate is zero."""
    assert _SCHEDULE["formulas"]["settlement"]["rate_bps"] == 0
    assert (
        _SCHEDULE["formulas"]["settlement"]["quoted"] == "There is no settlement fee."
    )


# --- FeeModel replays the published tables -------------------------------------


@pytest.mark.parametrize("row", _rows("general_trading_fees_table"))
def test_fee_model_reproduces_the_published_general_table_row(
    row: dict[str, int],
) -> None:
    """At 700 bps, `FeeModel` returns each printed general-market fee exactly."""
    model = _model(maker_ppm=0, taker_ppm=_TAKER_PPM)
    price_pips = row["price_cents"] * _PIPS_PER_CENT

    one_contract = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=_CENTIS_PER_CONTRACT
    )
    hundred_contracts = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=100 * _CENTIS_PER_CONTRACT
    )

    assert one_contract == row["fee_1_contract_cents"] * _MICROS_PER_CENT
    assert hundred_contracts == row["fee_100_contracts_cents"] * _MICROS_PER_CENT


@pytest.mark.parametrize("row", _rows("index_trading_fees_table"))
def test_fee_model_reproduces_the_published_index_table_row(
    row: dict[str, int],
) -> None:
    """At 350 bps, `FeeModel` returns each printed S&P500/Nasdaq-100 fee exactly."""
    model = _model(maker_ppm=0, taker_ppm=_INDEX_PPM)
    price_pips = row["price_cents"] * _PIPS_PER_CENT

    one_contract = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=_CENTIS_PER_CONTRACT
    )
    hundred_contracts = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=100 * _CENTIS_PER_CONTRACT
    )

    assert one_contract == row["fee_1_contract_cents"] * _MICROS_PER_CENT
    assert hundred_contracts == row["fee_100_contracts_cents"] * _MICROS_PER_CENT


# --- the bound is the max of the computed fees, not the fee of the max rate ----


def test_the_bound_is_the_taker_fee_when_both_published_rates_are_carried() -> None:
    """With maker 175 bps and taker 700 bps, the bound is the larger, computed fee.

    At 50c on 100 contracts the two fees are far apart -- $1.75 taker against
    $0.44 maker -- so a bound that took the lower fee, or the lower rate, would
    return a different number here.
    """
    model = _model(maker_ppm=_MAKER_PPM, taker_ppm=_TAKER_PPM)
    price_pips = 50 * _PIPS_PER_CENT
    count_centis = 100 * _CENTIS_PER_CONTRACT

    bound = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=count_centis
    )

    assert _quadratic_fee_micros(_MAKER_PPM, price_pips, count_centis) == 440_000
    assert _quadratic_fee_micros(_TAKER_PPM, price_pips, count_centis) == 1_750_000
    assert bound == 1_750_000


def test_the_bound_is_side_agnostic_when_the_published_rates_are_swapped() -> None:
    """Carrying 700 bps as maker and 175 bps as taker returns the same bound.

    The published rates are the same two numbers whichever field holds them, so
    the worst case must not depend on which side is nominally the higher one.
    """
    price_pips = 50 * _PIPS_PER_CENT
    count_centis = 100 * _CENTIS_PER_CONTRACT

    as_published = _model(
        maker_ppm=_MAKER_PPM, taker_ppm=_TAKER_PPM
    ).max_trading_fee_micros(price_pips=price_pips, count_centis=count_centis)
    swapped = _model(maker_ppm=_TAKER_PPM, taker_ppm=_MAKER_PPM).max_trading_fee_micros(
        price_pips=price_pips, count_centis=count_centis
    )

    assert as_published == 1_750_000
    assert swapped == 1_750_000


@pytest.mark.parametrize("price_cents", [1, 2, 98, 99])
def test_the_bound_at_the_price_extremes_dominates_both_published_fees(
    price_cents: int,
) -> None:
    """Near P=0 and P=1 -- where forms diverge most -- the bound still dominates.

    Issue #452's worked example lives at 99c: it feared the bound could come
    back *smaller* than an individually computed fee there. With both published
    rates carried, the bound equals the taker fee and strictly exceeds the maker
    fee at every one of these prices.
    """
    model = _model(maker_ppm=_MAKER_PPM, taker_ppm=_TAKER_PPM)
    price_pips = price_cents * _PIPS_PER_CENT
    count_centis = 100 * _CENTIS_PER_CONTRACT

    bound = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=count_centis
    )

    maker_fee = _quadratic_fee_micros(_MAKER_PPM, price_pips, count_centis)
    taker_fee = _quadratic_fee_micros(_TAKER_PPM, price_pips, count_centis)
    assert bound == taker_fee
    assert bound > maker_fee


@given(
    maker_ppm=st.integers(min_value=1, max_value=200_000),
    taker_ppm=st.integers(min_value=1, max_value=200_000),
    price_pips=st.integers(min_value=0, max_value=10_000),
    count_centis=st.integers(min_value=1, max_value=1_000_000),
)
def test_the_bound_equals_the_max_of_the_two_independently_computed_fees(
    maker_ppm: int, taker_ppm: int, price_pips: int, count_centis: int
) -> None:
    """Over the whole price range, the bound is the max of both computed fees.

    Each fee is computed here from the schedule's printed formula rather than
    from `FeeModel`, so the equality pins the combination semantics issue #452
    asks for: the *maximum of the fees*, never the fee of the minimum rate.
    The rates are forced distinct so no example can pass by the two sides
    coinciding.
    """
    assume(maker_ppm != taker_ppm)
    model = _model(maker_ppm=maker_ppm, taker_ppm=taker_ppm)

    bound = model.max_trading_fee_micros(
        price_pips=price_pips, count_centis=count_centis
    )

    maker_fee = _quadratic_fee_micros(maker_ppm, price_pips, count_centis)
    taker_fee = _quadratic_fee_micros(taker_ppm, price_pips, count_centis)
    assert bound == max(maker_fee, taker_fee)
    assert bound >= maker_fee
    assert bound >= taker_fee


# --- the guard that fails closed if the shared-form premise ever decays --------
#
# The finding above -- "every published trading fee shares one quadratic form,
# so `max(maker_ppm, taker_ppm)` is sound" -- is only safe while it stays true.
# `ACCOUNTING.md` names the series schema's empty cosmetic allowlist as what
# turns a venue publishing a *second* form into a halt rather than a silently
# ignored field. That claim was prose only: widening the allowlist left the
# whole suite green. These tests make it load-bearing.

#: The recorded clean `/series/{ticker}` document, read rather than restated.
_SERIES_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "exchange"
    / "kalshi"
    / "series_KXFED.json"
)

#: The endpoint path the series fee-schedule document is fetched from.
_SERIES_SEGMENTS = ("series", "KXFED")


def _series_wall_clock() -> datetime:
    """Return a fixed UTC instant so a ledgered anomaly's `ts` is deterministic."""
    return datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)


def _clean_series_payload() -> dict[str, Any]:
    """Return the recorded, drift-free series document as a mutable mapping."""
    with _SERIES_FIXTURE_PATH.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


def test_the_series_fee_block_grants_no_cosmetic_fields() -> None:
    """The series fee block tolerates *no* extra field, so none can be ignored.

    A cosmetic field is one the validator only warns about. Any fee-bearing
    field admitted there would let a new per-side fee form reach `FeeModel`
    unread, which is precisely the understatement issue #452 is about. The
    allowlist is read back through the public registry, so widening it by any
    name -- not merely a name guessed here -- fails this test.
    """
    registry = kalshi_default_schema_registry()

    series_schema = registry.schema_for(_SERIES_SEGMENTS)
    assert series_schema is not None
    assert series_schema.cosmetic == frozenset()
    block_schema = series_schema.recognized["series"]
    assert block_schema is not None
    assert block_schema.cosmetic == frozenset()


def test_a_per_side_fee_form_field_on_the_series_document_halts_and_ledgers() -> None:
    """A second fee form appearing on the series document halts, not an order.

    Driven through the real `SchemaValidator` and the real Kalshi registry --
    not a hand-built schema -- so this fails if the wiring, not merely the
    allowlist, stops fail-closing. The clean fixture is asserted to pass first,
    so the halt below is attributable to the added field alone.
    """
    ledger = InMemoryEventLedgerWriter()
    validator = SchemaValidator(
        kalshi_default_schema_registry(), ledger, wall_clock=_series_wall_clock
    )
    clean = _clean_series_payload()
    validator.validate(_SERIES_SEGMENTS, clean)
    assert ledger.events_by_type(SCHEMA_ANOMALY_EVENT) == ()

    drifted = _clean_series_payload()
    drifted["series"]["maker_fee_type"] = "linear"

    with pytest.raises(SchemaAnomalyHaltError) as excinfo:
        validator.validate(_SERIES_SEGMENTS, drifted)

    assert str(excinfo.value) == (
        "schema 'kalshi.series.block' v1 halted on unrecognized field(s): "
        "maker_fee_type"
    )
    (event,) = ledger.events_by_type(SCHEMA_ANOMALY_EVENT)
    assert event.payload["schema_key"] == "kalshi.series.block"
    assert event.payload["version"] == 1
    assert tuple(event.payload["fields"]) == ("maker_fee_type",)
