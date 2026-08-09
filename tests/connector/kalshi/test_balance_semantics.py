"""Tests for windbreak.connector.semantics (issue #18): the BalanceSemantics contract.

`windbreak/connector/semantics.py` does not exist yet, so importing it fails
collection with `ModuleNotFoundError: No module named
'windbreak.connector.semantics'` -- the expected Gate 1 RED state for issue #18.

Pins: each of the eight SPEC-required semantics enums carries an explicit
`UNKNOWN` member (never inferred, never silently defaulted); the exact member
set per enum; `BalanceSemantics.is_fully_known()`'s per-field boolean logic;
and the eight recorded Kalshi field values (five KNOWN, three UNKNOWN) that
back the venue's evidence table in
`tests/fixtures/exchange/kalshi/README.md`.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from typing import TYPE_CHECKING

import pytest

from windbreak.connector.fake import FakeExchange
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

if TYPE_CHECKING:
    import enum
    from pathlib import Path

    from windbreak.connector.kalshi.adapter import KalshiConnector

#: Every SPEC-required semantics enum, for parametrized cross-enum checks.
_ENUM_CLASSES = (
    OrderCollateralInTotal,
    OrderCollateralInAvailable,
    FeeDebitTiming,
    FeeRounding,
    PartialFillRepresentation,
    CancelCollateralRelease,
    UnsettledProceeds,
    HaltedMarketBehavior,
)

#: The exact member-name set each enum must carry -- no more, no less.
_EXPECTED_MEMBERS: dict[type, frozenset[str]] = {
    OrderCollateralInTotal: frozenset({"INCLUDED", "EXCLUDED", "UNKNOWN"}),
    OrderCollateralInAvailable: frozenset(
        {"DEDUCTED_FROM_AVAILABLE", "INCLUDED_IN_AVAILABLE", "UNKNOWN"}
    ),
    FeeDebitTiming: frozenset({"AT_EXECUTION", "AT_SETTLEMENT", "UNKNOWN"}),
    FeeRounding: frozenset({"UP_TO_NEXT_CENT", "EXACT", "UNKNOWN"}),
    PartialFillRepresentation: frozenset({"PER_FILL_RECORDS", "AGGREGATED", "UNKNOWN"}),
    CancelCollateralRelease: frozenset({"IMMEDIATE", "DELAYED", "UNKNOWN"}),
    UnsettledProceeds: frozenset(
        {"EXCLUDED_UNTIL_CREDITED", "INCLUDED_IMMEDIATELY", "UNKNOWN"}
    ),
    HaltedMarketBehavior: frozenset(
        {"NEW_ORDERS_REJECTED", "NEW_ORDERS_ACCEPTED", "UNKNOWN"}
    ),
}

#: A fully-known `BalanceSemantics`, one non-UNKNOWN member per field, used to
#: pin `is_fully_known()`'s True branch and as a base for single-field flips.
_ALL_KNOWN_KWARGS: dict[str, object] = {
    "open_order_collateral_in_total": OrderCollateralInTotal.INCLUDED,
    "open_order_collateral_in_available": (
        OrderCollateralInAvailable.DEDUCTED_FROM_AVAILABLE
    ),
    "fee_debit_timing": FeeDebitTiming.AT_EXECUTION,
    "fee_rounding": FeeRounding.UP_TO_NEXT_CENT,
    "partial_fill_representation": PartialFillRepresentation.PER_FILL_RECORDS,
    "cancel_collateral_release": CancelCollateralRelease.IMMEDIATE,
    "unsettled_proceeds": UnsettledProceeds.EXCLUDED_UNTIL_CREDITED,
    "halted_market_behavior": HaltedMarketBehavior.NEW_ORDERS_REJECTED,
}

#: This field's UNKNOWN member, keyed by field name, for the single-flip test.
_UNKNOWN_BY_FIELD: dict[str, object] = {
    "open_order_collateral_in_total": OrderCollateralInTotal.UNKNOWN,
    "open_order_collateral_in_available": OrderCollateralInAvailable.UNKNOWN,
    "fee_debit_timing": FeeDebitTiming.UNKNOWN,
    "fee_rounding": FeeRounding.UNKNOWN,
    "partial_fill_representation": PartialFillRepresentation.UNKNOWN,
    "cancel_collateral_release": CancelCollateralRelease.UNKNOWN,
    "unsettled_proceeds": UnsettledProceeds.UNKNOWN,
    "halted_market_behavior": HaltedMarketBehavior.UNKNOWN,
}


def _semantics(**overrides: object) -> BalanceSemantics:
    return BalanceSemantics(**{**_ALL_KNOWN_KWARGS, **overrides})


# --- Every enum has an explicit UNKNOWN member, and only the spec'd members --


@pytest.mark.parametrize("enum_cls", _ENUM_CLASSES)
def test_every_semantics_enum_has_an_explicit_unknown_member(
    enum_cls: type[enum.Enum],
) -> None:
    """Every one of the eight semantics enums documents an explicit UNKNOWN."""
    assert "UNKNOWN" in enum_cls.__members__


@pytest.mark.parametrize("enum_cls", _ENUM_CLASSES)
def test_enum_member_names_match_the_spec_exactly(
    enum_cls: type[enum.Enum],
) -> None:
    """Each enum carries exactly its spec'd member set -- no extra, no missing."""
    member_names = frozenset(enum_cls.__members__)
    assert member_names == _EXPECTED_MEMBERS[enum_cls]


# --- BalanceSemantics: shape, immutability, is_fully_known() -----------------


def test_balance_semantics_field_order_matches_the_spec() -> None:
    """Field order is pinned so downstream positional construction stays stable."""
    field_names = tuple(field.name for field in dataclasses.fields(BalanceSemantics))

    assert field_names == (
        "open_order_collateral_in_total",
        "open_order_collateral_in_available",
        "fee_debit_timing",
        "fee_rounding",
        "partial_fill_representation",
        "cancel_collateral_release",
        "unsettled_proceeds",
        "halted_market_behavior",
    )


def test_balance_semantics_is_frozen() -> None:
    """A `BalanceSemantics` is frozen: assignment raises FrozenInstanceError."""
    semantics = _semantics()
    # Assign through a dynamic attribute name so the test exercises the frozen
    # dataclass's runtime rejection without a static type-checker suppression.
    frozen_field = "fee_rounding"

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(semantics, frozen_field, FeeRounding.EXACT)


def test_is_fully_known_true_when_every_field_is_a_known_member() -> None:
    """A record with no UNKNOWN field reports `is_fully_known()` as True."""
    assert _semantics().is_fully_known() is True


@pytest.mark.parametrize("field_name", sorted(_ALL_KNOWN_KWARGS))
def test_is_fully_known_false_when_exactly_one_field_is_unknown(
    field_name: str,
) -> None:
    """Flipping any single field to UNKNOWN alone must flip the overall verdict."""
    semantics = _semantics(**{field_name: _UNKNOWN_BY_FIELD[field_name]})

    assert semantics.is_fully_known() is False


def test_is_fully_known_false_when_every_field_is_unknown() -> None:
    """An all-UNKNOWN record reports `is_fully_known()` as False."""
    semantics = _semantics(**_UNKNOWN_BY_FIELD)

    assert semantics.is_fully_known() is False


# --- The issue's worked example: Kalshi excludes proceeds until credited -----


def test_kalshi_unsettled_proceeds_matches_the_issue_example(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """The issue's verbatim example: Kalshi never credits proceeds early."""
    semantics = kalshi_fixture_connector.get_balance_semantics()

    assert semantics.unsettled_proceeds is UnsettledProceeds.EXCLUDED_UNTIL_CREDITED


# --- Kalshi's full recorded record: five KNOWN, three UNKNOWN ---------------


def test_kalshi_balance_semantics_known_fields(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """The five fields Kalshi's documented behavior lets us pin as KNOWN."""
    semantics = kalshi_fixture_connector.get_balance_semantics()

    assert semantics.fee_rounding is FeeRounding.UP_TO_NEXT_CENT
    assert semantics.fee_debit_timing is FeeDebitTiming.AT_EXECUTION
    assert (
        semantics.partial_fill_representation
        is PartialFillRepresentation.PER_FILL_RECORDS
    )
    assert semantics.unsettled_proceeds is UnsettledProceeds.EXCLUDED_UNTIL_CREDITED
    assert (
        semantics.open_order_collateral_in_available
        is OrderCollateralInAvailable.DEDUCTED_FROM_AVAILABLE
    )


def test_kalshi_balance_semantics_unknown_fields(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """The three fields Kalshi documents no public evidence for stay UNKNOWN."""
    semantics = kalshi_fixture_connector.get_balance_semantics()

    assert semantics.open_order_collateral_in_total is OrderCollateralInTotal.UNKNOWN
    assert semantics.cancel_collateral_release is CancelCollateralRelease.UNKNOWN
    assert semantics.halted_market_behavior is HaltedMarketBehavior.UNKNOWN


def test_kalshi_balance_semantics_is_not_fully_known(
    kalshi_fixture_connector: KalshiConnector,
) -> None:
    """Three undocumented fields mean Kalshi's record is never fully known."""
    semantics = kalshi_fixture_connector.get_balance_semantics()

    assert semantics.is_fully_known() is False


# --- FakeExchange: an all-known fixture record round-trips -------------------


def test_fake_exchange_balance_semantics_is_fully_known(
    fake_exchange: FakeExchange,
) -> None:
    """The shared fixture's semantics record has no UNKNOWN field left."""
    semantics = fake_exchange.get_balance_semantics()

    assert semantics.is_fully_known() is True


def test_fake_exchange_raises_loudly_on_unrecognized_balance_semantics_value(
    tmp_path: Path, fixture_dir: Path
) -> None:
    """An unrecognized enum-member-name value in the fixture fails loudly.

    A venue integration is only as trustworthy as its loudest failure mode: a
    typo'd or invented member name (e.g. from a hand-edited fixture) must never
    silently coerce to a default -- it must raise.
    """
    broken_dir = tmp_path / "exchange"
    shutil.copytree(fixture_dir, broken_dir)
    original = json.loads(
        (fixture_dir / "balance_semantics.json").read_text(encoding="utf-8")
    )
    broken = {**original, "fee_rounding": "NOT_A_REAL_MEMBER"}
    (broken_dir / "balance_semantics.json").write_text(
        json.dumps(broken), encoding="utf-8"
    )

    with pytest.raises((KeyError, ValueError)):
        FakeExchange.from_fixture_dir(broken_dir)
