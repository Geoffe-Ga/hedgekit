"""Fail-closed projection of connector eligibility onto kernel enums (#340, RED).

`windbreak.scheduler.eligibility` does not exist yet, so every test here fails at
import with `ModuleNotFoundError` -- the intended Gate 1 RED for issue #340.

Why this module exists at all: the Risk Kernel must not import connector types
(SPEC S5 Process-B trust boundary), so the scheduler -- the one composition layer
permitted to see both sides -- translates the connector's
`Literal["eligible", "ineligible", "unknown"]` into the kernel's own enum, mapping
anything it cannot prove eligible onto `None`.

The projection deliberately takes the **raw field value** (`str | None`) rather
than a `NormalizedMarket`. `NormalizedMarket.__post_init__` rejects out-of-set
values, so a market-taking signature would make the adversarial-string cases
below unconstructible -- and those cases are exactly where the fail-closed
contract's branch coverage and mutation strength live.
"""

from __future__ import annotations

import pytest

from windbreak.riskkernel.context import JurisdictionStatus, ProductType
from windbreak.scheduler.eligibility import project_jurisdiction, project_product_type

#: Raw values the connector's `jurisdiction_status` Literal can actually hold,
#: paired with the kernel value each must project onto. `"unknown"` is the
#: load-bearing case: it is a legal connector value that must NOT survive as an
#: enum member, because an enum member could be compared equal to eligible.
_JURISDICTION_CASES = (
    ("eligible", JurisdictionStatus.ELIGIBLE),
    ("ineligible", JurisdictionStatus.INELIGIBLE),
    ("unknown", None),
    (None, None),
)

#: Values that are not in the connector's closed set at all -- typos, casing
#: drift, whitespace, a future literal nobody has taught this table about. Each
#: must fail closed rather than raise or leak through.
_ADVERSARIAL_JURISDICTIONS = (
    "ELIGIBLE",
    "Eligible",
    "eligible ",
    " eligible",
    "",
    "pending",
)

#: The single product type the connector can produce, plus the raw Kalshi values
#: that are refused before normalization and so must never project to a product.
_PRODUCT_CASES = (
    ("fully_collateralized_binary", ProductType.FULLY_COLLATERALIZED_BINARY),
    ("binary", None),
    ("perpetual", None),
    ("scalar", None),
    ("", None),
    (None, None),
)


@pytest.mark.parametrize(("raw", "expected"), _JURISDICTION_CASES)
def test_project_jurisdiction_maps_the_closed_set(
    raw: str | None, expected: JurisdictionStatus | None
) -> None:
    """Each connector jurisdiction literal projects onto its kernel counterpart.

    Args:
        raw: The connector's raw `jurisdiction_status` value.
        expected: The kernel value it must project onto.
    """
    assert project_jurisdiction(raw) is expected


@pytest.mark.parametrize("raw", _ADVERSARIAL_JURISDICTIONS)
def test_project_jurisdiction_fails_closed_on_anything_unrecognized(raw: str) -> None:
    """An unrecognized jurisdiction value projects to `None`, never to eligible.

    This is an allowlist, not a denylist: a value the table has never been taught
    yields `None` with no exception and no branch anyone can forget to write.

    Args:
        raw: A value outside the connector's closed set.
    """
    assert project_jurisdiction(raw) is None


@pytest.mark.parametrize(("raw", "expected"), _PRODUCT_CASES)
def test_project_product_type_maps_only_the_collateralized_binary(
    raw: str | None, expected: ProductType | None
) -> None:
    """Only the fully collateralized binary projects to a product type.

    Args:
        raw: The connector's raw `market_type` value.
        expected: The kernel value it must project onto.
    """
    assert project_product_type(raw) is expected


def test_a_normalized_kalshi_market_never_projects_to_an_eligible_jurisdiction() -> (
    None
):
    """A real Kalshi market projects to `None`, so the kernel vetoes it.

    Standing regression guard, and the honest boundary of issue #340: Kalshi
    exposes no jurisdiction signal, so `normalize_market` stamps every market
    `"unknown"`. Approval on a live Kalshi path therefore remains impossible by
    construction -- only a market carrying real eligibility metadata (the paper
    fixture books do) can clear this check. If a later change makes this test
    fail, someone has invented an eligibility signal the exchange never gave us.
    """
    from windbreak.connector.kalshi.normalize import _JURISDICTION_UNKNOWN

    assert project_jurisdiction(_JURISDICTION_UNKNOWN) is None
