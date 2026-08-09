"""Project connector eligibility metadata onto the kernel's own enums (#340).

The Risk Kernel never imports connector types (SPEC S5 Process-B trust
boundary), so it cannot read a
:class:`~windbreak.connector.models.NormalizedMarket` directly. The scheduler is
the composition layer permitted to see both sides, and this module is the
one-way translation it performs.

Both projections are **allowlists**, not denylists: a value the table has not
been taught -- the connector's own ``"unknown"``, a future literal, a typo, a
casing difference, or ``None`` -- maps to ``None``. There is no raise and no
branch a later author can forget to write, and ``None`` is precisely what
``jurisdiction_product_eligibility`` fails closed on (SPEC S1.1 invariant 1,
SPEC S6.2).

Each function takes the **raw field value** rather than the market, so the
fail-closed contract can be exercised with values
``NormalizedMarket.__post_init__`` would refuse to construct.
"""

from __future__ import annotations

from windbreak.riskkernel.context import JurisdictionStatus, ProductType

#: The only raw ``jurisdiction_status`` values that project onto a kernel value.
#: The connector's third literal, ``"unknown"``, is deliberately absent: it must
#: become ``None`` so it can never be compared equal to ``ELIGIBLE``.
_JURISDICTION_BY_RAW: dict[str, JurisdictionStatus] = {
    "eligible": JurisdictionStatus.ELIGIBLE,
    "ineligible": JurisdictionStatus.INELIGIBLE,
}

#: The only raw ``market_type`` value that projects onto a kernel product.
_PRODUCT_TYPE_BY_RAW: dict[str, ProductType] = {
    "fully_collateralized_binary": ProductType.FULLY_COLLATERALIZED_BINARY,
}


def project_jurisdiction(status: str | None) -> JurisdictionStatus | None:
    """Return the kernel jurisdiction status for a raw connector value.

    Args:
        status: The market's raw ``jurisdiction_status``, or ``None`` when no
            market is available at all.

    Returns:
        :attr:`JurisdictionStatus.ELIGIBLE` or
        :attr:`JurisdictionStatus.INELIGIBLE` for a recognized value; ``None``
        for ``"unknown"``, ``None``, or anything unrecognized -- which the
        kernel's eligibility check vetoes.
    """
    if status is None:
        return None
    return _JURISDICTION_BY_RAW.get(status)


def project_product_type(market_type: str | None) -> ProductType | None:
    """Return the kernel product type for a raw connector value.

    Args:
        market_type: The market's raw ``market_type``, or ``None`` when no
            market is available at all.

    Returns:
        :attr:`ProductType.FULLY_COLLATERALIZED_BINARY` for the one recognized
        product; ``None`` for ``None`` or anything unrecognized -- which the
        kernel's eligibility check vetoes.
    """
    if market_type is None:
        return None
    return _PRODUCT_TYPE_BY_RAW.get(market_type)
