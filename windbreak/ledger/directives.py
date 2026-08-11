"""The closed vocabulary a *directive* delivery may be recorded as (issue #480).

The kill switch's one ``CancelAllDirective`` row proved the directive was
*emitted*. It looked identical whether the venue cancelled every resting order
or the connector raised on the first one, so an audit could not establish that
anything was actually cancelled -- the same emission-is-not-delivery gap issue
#413 closed for ``AlertEmitted``, one layer over.

This module is that gap's closure, and it draws the same line issue #274 forced:
a connector failure detail is ``str(exc)`` from an arbitrary venue client -- the
exact shape that leaked whole token-bearing URLs -- and a hash chain is
append-only, so nothing written into it can ever be redacted. So a
:class:`DirectiveDelivery` has **no free-form field at all**: two counts the
code owns end to end, and an enumerated :class:`DirectiveOutcome` *derived* from
them. Venue order ids are arbitrary venue-supplied text and are deliberately not
among them either; ``cancelled``/``failed`` say how many, never which.

It sits below both :mod:`windbreak.riskkernel.kill` (which ledgers the record)
and :mod:`windbreak.order_gateway.cancel_all` (which produces it), importing
neither, so the seam between them carries data rather than a package dependency
(SPEC S5: the risk kernel imports no connector types).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class DirectiveOutcome(enum.Enum):
    """How a directive sink answered one directive -- the closed vocabulary.

    ``DELIVERED`` is the only member that means the venue took the whole
    instruction; every other member means it did not, and the distinction
    between them is diagnostic only. A reader that treats anything but
    ``DELIVERED`` as success would be failing open.
    """

    DELIVERED = "delivered"
    PARTIAL = "partial"
    REFUSED = "refused"
    ERRORED = "errored"


@dataclass(frozen=True, slots=True)
class DirectiveDelivery:
    """What a directive sink did with one cancel-all directive.

    Attributes:
        cancelled: How many resting orders the venue accepted a cancellation
            for. Zero with a zero ``failed`` means there was nothing resting --
            a vacuously complete delivery, not a silent one.
        failed: How many resting orders the venue refused or raised on. The
            *reason* is deliberately not recorded: it would be venue-supplied
            text on an unredactable chain (issue #274).
        errored: Whether the sink itself failed before it could attempt any
            order, leaving both counts unknown rather than zero. Kept separate
            from ``failed`` because "nothing was resting" and "the sink never
            got as far as looking" are opposite facts that both count zero.
    """

    cancelled: int = 0
    failed: int = 0
    errored: bool = False

    @property
    def fully_delivered(self) -> bool:
        """Return whether the venue took the entire instruction.

        Returns:
            True only when the sink ran to completion and refused nothing.
            An errored sink is never fully delivered whatever its counts say:
            absent evidence must not read as healthy.
        """
        return not self.errored and self.failed == 0

    @property
    def outcome(self) -> DirectiveOutcome:
        """Return the enumerated outcome this delivery amounts to.

        Derived from the counts rather than tracked alongside them, so the word
        an auditor reads can never drift from the numbers beside it.

        Returns:
            The closed :class:`DirectiveOutcome` member.
        """
        if self.errored:
            return DirectiveOutcome.ERRORED
        if self.failed == 0:
            return DirectiveOutcome.DELIVERED
        if self.cancelled == 0:
            return DirectiveOutcome.REFUSED
        return DirectiveOutcome.PARTIAL

    def as_payload(self) -> dict[str, object]:
        """Return this delivery as the closed mapping a ledger payload holds.

        Returns:
            A three-key mapping of the two counts and the derived outcome's
            enum value. There is no fourth key, and in particular no venue
            order id and no failure text.
        """
        return {
            "cancelled": self.cancelled,
            "failed": self.failed,
            "outcome": self.outcome.value,
        }


def ledger_directive_delivery(delivery: DirectiveDelivery | None) -> dict[str, object]:
    """Project a directive delivery into the closed rows a ledger may hold.

    The single producer of ledgered directive-delivery evidence, mirroring
    :func:`windbreak.alerts.dispatch.ledger_deliveries`: everything that reaches
    an append-only chain about *what a venue did with a directive* passes
    through here, so there is one place to audit rather than one per call site.

    Args:
        delivery: The delivery to project, or None when no directive sink was
            wired at all and there is therefore no evidence either way.

    Returns:
        The closed three-key mapping, or an empty mapping when ``delivery`` is
        None. Empty means *unreported*, never delivered.
    """
    if delivery is None:
        return {}
    return delivery.as_payload()
