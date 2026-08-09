"""Failing-first tests for windbreak.numeric.types (issue #12, SPEC S6.1).

Pins the four frozen int-wrapper value types -- PricePips (1e-4
payout-dollars), ContractCentis (1e-2 contracts), MoneyMicros (1e-6
dollars), ProbabilityPpm (1e-6 probability) -- covering construction
validation, within-unit arithmetic/comparisons, cross-unit isolation,
display formatting, and the canonical price*count -> money conversion.

Pinned display formats (implementer must match exactly; see the
individual test docstrings below for the derivation of each):
    MoneyMicros(1_370_100)  -> "$1.370100"
    PricePips(4567)         -> "$0.4567"
    PricePips(50)           -> "$0.0050"   (half a cent)
    ContractCentis(300)     -> "3.00"
    ProbabilityPpm(456700)  -> "45.6700%"
All are sign-prefixed separately from a divmod() on the absolute value,
so floor-division-on-negatives can never corrupt the digits.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from windbreak.numeric.rounding import RoundingDirection
from windbreak.numeric.types import (
    ContractCentis,
    MoneyMicros,
    PricePips,
    ProbabilityPpm,
    money_from_price_and_count,
)

#: All four SPEC S6.1 unit types, for parametrized within-unit tests.
UNIT_TYPES = (PricePips, ContractCentis, MoneyMicros, ProbabilityPpm)

#: Every ordered pair of *distinct* unit types, for cross-unit isolation tests.
CROSS_UNIT_PAIRS = [
    (left, right)
    for left, right in itertools.product(UNIT_TYPES, repeat=2)
    if left is not right
]


# --- Construction validation -------------------------------------------------


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_construction_rejects_bool_true(unit_type) -> None:
    """bool is an int subclass in Python; every unit type must reject it.

    Silently accepting `True` as `1` would let a stray boolean flag
    corrupt a money/price/probability value without any signal.
    """
    with pytest.raises(TypeError):
        unit_type(True)


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_construction_rejects_bool_false(unit_type) -> None:
    """Explicit False case: bool(0) must not silently become value=0."""
    with pytest.raises(TypeError):
        unit_type(False)


@pytest.mark.parametrize("bad_value", [1.5, "1", None, [1], 1.0, 0.0])
@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_construction_rejects_non_int(unit_type, bad_value) -> None:
    """Only true `int` values are accepted at construction -- never float,
    str, None, or other containers, even when they look numeric."""
    with pytest.raises(TypeError):
        unit_type(bad_value)


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_construction_accepts_int_and_round_trips_through_value(unit_type) -> None:
    """A valid int constructs cleanly and is recoverable via `.value`."""
    instance = unit_type(42)

    assert instance.value == 42


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_construction_accepts_negative_int(unit_type) -> None:
    """Negative values are valid (e.g. a debit, a short position)."""
    instance = unit_type(-7)

    assert instance.value == -7


# --- Within-unit arithmetic --------------------------------------------------


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_addition_within_unit_returns_same_type_and_sum(unit_type) -> None:
    """Adding two same-unit values returns that unit type holding the sum."""
    result = unit_type(7) + unit_type(5)

    assert type(result) is unit_type
    assert result.value == 12


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_subtraction_within_unit_returns_same_type_and_difference(unit_type) -> None:
    """Subtracting same-unit values returns that unit type with the difference."""
    result = unit_type(7) - unit_type(5)

    assert type(result) is unit_type
    assert result.value == 2


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_negation_within_unit_returns_same_type_and_negated_value(unit_type) -> None:
    """Negating a unit value returns the same type with the value negated."""
    result = -unit_type(7)

    assert type(result) is unit_type
    assert result.value == -7


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_scalar_multiplication_returns_same_type_and_product(unit_type) -> None:
    """A unit value times a dimensionless int scales within the same unit.

    e.g. ContractCentis(100) * 3 == ContractCentis(300); the dimension is
    unchanged, so the result is the same unit type (SPEC S6.1 "scalar
    multiplication").
    """
    result = unit_type(7) * 3

    assert type(result) is unit_type
    assert result.value == 21


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_scalar_right_multiplication_is_commutative(unit_type) -> None:
    """`scalar * value` (via __rmul__) equals `value * scalar`."""
    result = 3 * unit_type(7)

    assert type(result) is unit_type
    assert result.value == 21


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
@pytest.mark.parametrize("scalar,expected", [(0, 0), (-4, -28)])
def test_scalar_multiplication_zero_and_negative(unit_type, scalar, expected) -> None:
    """Scaling by zero or a negative int is well-defined and stays in-unit."""
    result = unit_type(7) * scalar

    assert type(result) is unit_type
    assert result.value == expected


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_scalar_multiplication_rejects_bool(unit_type) -> None:
    """bool is an int subclass but is not a valid scalar (mirrors construction).

    Silently accepting `True` as the factor `1` would let a stray boolean
    flag masquerade as a scale factor without any signal.
    """
    with pytest.raises(TypeError):
        unit_type(7) * True
    with pytest.raises(TypeError):
        True * unit_type(7)


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_scalar_multiplication_rejects_float(unit_type) -> None:
    """Multiplying by a float would put a float on the money path -- forbidden."""
    with pytest.raises(TypeError):
        unit_type(7) * 2.5
    with pytest.raises(TypeError):
        2.5 * unit_type(7)


@pytest.mark.parametrize("left_type,right_type", CROSS_UNIT_PAIRS)
def test_cross_unit_multiplication_raises_type_error(left_type, right_type) -> None:
    """A unit times another unit changes the dimension and must fail loudly."""
    with pytest.raises(TypeError):
        left_type(2) * right_type(3)


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_same_unit_multiplication_raises_type_error(unit_type) -> None:
    """Even same-unit * same-unit is dimensionally nonsensical and must raise."""
    with pytest.raises(TypeError):
        unit_type(2) * unit_type(3)


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_within_unit_equality_and_ordering(unit_type) -> None:
    """Same-unit values compare by payload across <, <=, >, >=, == and !=."""
    small, big, big_copy = unit_type(3), unit_type(9), unit_type(9)

    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big == big_copy
    assert small != big
    assert not big < small
    assert not (big < big_copy)


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
@given(a=st.integers(), b=st.integers(), c=st.integers())
def test_within_unit_addition_is_associative(unit_type, a: int, b: int, c: int) -> None:
    """(x + y) + z == x + (y + z) for within-unit addition (SPEC S17.3).

    A metamorphic property required by issue #12: because each unit carries
    an exact integer payload, addition is exactly integer addition and must
    be associative for every combination of operands, with the result always
    the same unit type (never a float or a decayed int).
    """
    x, y, z = unit_type(a), unit_type(b), unit_type(c)

    left_assoc = (x + y) + z
    right_assoc = x + (y + z)

    assert left_assoc == right_assoc
    assert type(left_assoc) is unit_type
    assert type(right_assoc) is unit_type


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
@given(value=st.integers(), scalar=st.integers())
def test_scalar_multiplication_never_returns_float(
    unit_type, value: int, scalar: int
) -> None:
    """Scalar multiplication always yields the same unit type, never a float.

    Backs the SPEC S6.1 "no operation returns float" invariant for the
    scalar-multiply operation across the full integer domain.
    """
    result = unit_type(value) * scalar

    assert type(result) is unit_type
    assert result.value == value * scalar


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_units_are_hashable_and_usable_as_keys(unit_type) -> None:
    """Frozen value types must stay hashable (set/dict membership).

    Pins the hashability the frozen dataclass grants, so a future refactor
    that adds a custom ``__eq__`` without a matching ``__hash__`` (which
    would silently make instances unhashable) fails loudly here.
    """
    assert {unit_type(9)} == {unit_type(9)}
    assert hash(unit_type(9)) == hash(unit_type(9))
    assert {unit_type(9): "x"}[unit_type(9)] == "x"


# --- Cross-unit isolation -----------------------------------------------------


@pytest.mark.parametrize("left_type,right_type", CROSS_UNIT_PAIRS)
def test_cross_unit_addition_raises_type_error(left_type, right_type) -> None:
    """Every ordered pair of distinct unit types must refuse to add.

    e.g. PricePips(1) + ContractCentis(1) is nonsensical (dollars-per-
    contract plus a contract count) and must fail loudly, not silently
    produce a garbage int.
    """
    with pytest.raises(TypeError):
        left_type(1) + right_type(1)


@pytest.mark.parametrize("left_type,right_type", CROSS_UNIT_PAIRS)
def test_cross_unit_subtraction_raises_type_error(left_type, right_type) -> None:
    """Every ordered pair of distinct unit types must refuse to subtract."""
    with pytest.raises(TypeError):
        left_type(1) - right_type(1)


@pytest.mark.parametrize("left_type,right_type", CROSS_UNIT_PAIRS)
def test_cross_unit_equality_is_false_not_an_error(left_type, right_type) -> None:
    """Cross-unit equality is a well-defined False, never a TypeError.

    Unlike arithmetic (which is nonsensical across units and must raise),
    `==` must remain total: two values of different unit types are simply
    unequal, the same way `1 == "1"` is False rather than an error.
    """
    assert (left_type(1) == right_type(1)) is False
    assert (left_type(1) != right_type(1)) is True


@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_true_division_is_unsupported(unit_type) -> None:
    """The base type never defines `__truediv__`; `/` must raise TypeError.

    This is the type-level enforcement of "no floats on the money path":
    dividing two same-unit values would need a float (or a new unit) to
    represent the result, and the base deliberately never allows it.
    """
    with pytest.raises(TypeError):
        unit_type(10) / unit_type(2)


# --- Display formatting -------------------------------------------------------


def test_money_micros_str_positive() -> None:
    """1_370_100 micros == $1.370100 (1,000,000 micros per dollar)."""
    assert str(MoneyMicros(1_370_100)) == "$1.370100"


def test_money_micros_str_zero() -> None:
    """Zero micros renders as `$0.000000`, keeping all six fraction digits."""
    assert str(MoneyMicros(0)) == "$0.000000"


def test_money_micros_str_negative() -> None:
    """Sign is prefixed before the '$', digits are the absolute value."""
    assert str(MoneyMicros(-500_000)) == "-$0.500000"


def test_price_pips_str_typical() -> None:
    """4567 pips == $0.4567 (10,000 pips per dollar)."""
    assert str(PricePips(4567)) == "$0.4567"


def test_price_pips_str_half_cent() -> None:
    """50 pips == half a cent == $0.0050, zero-padded to 4 digits."""
    assert str(PricePips(50)) == "$0.0050"


def test_price_pips_str_negative() -> None:
    """A negative PricePips renders sign-first as `-$0.0050`."""
    assert str(PricePips(-50)) == "-$0.0050"


def test_contract_centis_str_typical() -> None:
    """300 centis == 3.00 contracts (100 centis per contract, no unit suffix)."""
    assert str(ContractCentis(300)) == "3.00"


def test_contract_centis_str_sub_unit() -> None:
    """A single centi renders as the sub-unit quantity `0.01`."""
    assert str(ContractCentis(1)) == "0.01"


def test_contract_centis_str_negative() -> None:
    """A negative ContractCentis renders sign-first as `-3.00`."""
    assert str(ContractCentis(-300)) == "-3.00"


def test_probability_ppm_str_typical() -> None:
    """456700 ppm == 45.6700% (10,000 ppm per percentage point)."""
    assert str(ProbabilityPpm(456_700)) == "45.6700%"


def test_probability_ppm_str_negative() -> None:
    """A negative ProbabilityPpm renders sign-first as `-45.6700%`."""
    assert str(ProbabilityPpm(-456_700)) == "-45.6700%"


def test_probability_ppm_str_zero() -> None:
    """Zero ppm renders as `0.0000%`, keeping four fraction digits."""
    assert str(ProbabilityPpm(0)) == "0.0000%"


# --- money_from_price_and_count: the canonical conversion --------------------


def test_money_from_price_and_count_canonical_example_overstate() -> None:
    """SPEC S6.1 derivation: price(pips) x count(centis) = money(micros) exactly.

    price = pips * 1e-4 $/contract; count = centis * 1e-2 contracts;
    price * count = pips * centis * 1e-6 dollars == (pips * centis) micros
    -- an exact integer product, with no rounding loss whatsoever. For
    PricePips(4567) and ContractCentis(300) (i.e. 3.00 contracts):
    4567 * 300 = 1_370_100 micros == $1.370100. (NOT the issue's originally
    stated 137_010_000, which was a 100x error that treated 300 centis as
    300 whole contracts instead of 3.00 contracts.)
    """
    result = money_from_price_and_count(
        PricePips(4567),
        ContractCentis(300),
        rounding=RoundingDirection.OVERSTATE_COST,
    )

    assert result == MoneyMicros(1_370_100)


def test_money_from_price_and_count_canonical_example_understate() -> None:
    """The conversion is exact integer multiplication -- both directions agree."""
    result = money_from_price_and_count(
        PricePips(4567),
        ContractCentis(300),
        rounding=RoundingDirection.UNDERSTATE_EQUITY,
    )

    assert result == MoneyMicros(1_370_100)


def test_money_from_price_and_count_rounding_is_keyword_only() -> None:
    """`rounding` must be required and keyword-only, never positional."""
    with pytest.raises(TypeError):
        money_from_price_and_count(
            PricePips(1), ContractCentis(1), RoundingDirection.OVERSTATE_COST
        )


def test_money_from_price_and_count_rounding_is_required() -> None:
    """Omitting `rounding` entirely must fail loudly, not default silently."""
    with pytest.raises(TypeError):
        money_from_price_and_count(PricePips(1), ContractCentis(1))


@given(
    price=st.integers(min_value=0, max_value=100_000),
    count=st.integers(min_value=0, max_value=100_000),
)
def test_money_from_price_and_count_is_exact_product_both_directions(
    price: int, count: int
) -> None:
    """price(pips) x count(centis) is exactly (price*count) micros -- no loss.

    Because the unit scaling cancels exactly (1e-4 * 1e-2 == 1e-6), this
    conversion never has a remainder to round away, so OVERSTATE_COST and
    UNDERSTATE_EQUITY must produce the identical MoneyMicros value
    (metamorphic property: over == exact == under).
    """
    over = money_from_price_and_count(
        PricePips(price),
        ContractCentis(count),
        rounding=RoundingDirection.OVERSTATE_COST,
    )
    under = money_from_price_and_count(
        PricePips(price),
        ContractCentis(count),
        rounding=RoundingDirection.UNDERSTATE_EQUITY,
    )

    assert over.value == price * count
    assert under.value == price * count
    assert over == under
