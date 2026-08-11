"""The consumer of the kill switch's cancel-all directive (issue #480).

Engaging the kill switch wrote one
:class:`~windbreak.ledger.events.CancelAllDirective` to the hash-chained
ledger, and **nothing consumed it**. No ``directive_sink`` was wired on either
composition path, so on kill an order resting at the venue was an audit record
rather than an effect: a live instruction that could still fill after the
operator had killed the system and walked away. This module is the thing that
was missing.

**Why it lives here.** SPEC S5.2/S5.3 put trade-capable venue access in the
Order Gateway and nowhere else, and SPEC S5 keeps connector types out of the
Risk Kernel entirely. So the kernel emits the directive as data across the
narrow :class:`~windbreak.riskkernel.kill.DirectiveSink` seam, and the order-
gateway side -- this module -- is what turns it into venue calls. The seam
carries :class:`~windbreak.ledger.directives.DirectiveDelivery` back, a type
neither package owns, so the kernel learns what happened without either side
importing the other.

**Why it never raises.** :class:`VenueCancelAllSink` mirrors
:class:`~windbreak.alerts.dispatch.AlertDispatcher`'s discipline exactly: one
broken order can never stop the rest from being attempted, and the whole sink
can never take down the kill path that called it. A kill's remaining fail-safe
effects -- the reservation release, the operator page, the ``KILL`` file -- must
all still happen when a venue misbehaves. Failure is reported, loudly, rather
than thrown: as a count on an enumerated outcome the kill switch ledgers and
names on the page.

**What never reaches the chain.** A venue client's rejection is ``str(exc)``
from an arbitrary third party, and a venue order id is arbitrary venue-supplied
text; both are the shape that leaked whole token-bearing URLs in issue #274,
and a hash chain can never be redacted. Neither is returned from here. They are
logged -- a log line may hold what a chain may not -- and the sink's report
carries only counts and a closed outcome.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from windbreak.ledger.directives import DirectiveDelivery

if TYPE_CHECKING:
    from windbreak.connector.models import OpenOrder
    from windbreak.ledger.events import CancelAllDirective

_LOGGER = logging.getLogger("windbreak.order_gateway.cancel_all")

#: Component label used on this module's log records.
_COMPONENT = "order_gateway"


class RestingOrderVenue(Protocol):
    """The venue seam a cancel-all directive is delivered through.

    Exactly the two methods a cancel-all needs, and no more: the resting orders
    to act on, and the call that cancels one. Structural, so every
    :class:`~windbreak.connector.interface.MarketConnector` satisfies it without
    this module naming a concrete connector -- and so a test double is a class
    with two methods rather than a fifteen-method stub.
    """

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Return the account's resting open orders."""
        ...

    def cancel_order(self, order_id: str) -> None:
        """Cancel a resting order by its identifier.

        Args:
            order_id: The venue order identifier to cancel.
        """
        ...


class VenueCancelAllSink:
    """Delivers the kill switch's one cancel-all directive to a venue.

    The :class:`~windbreak.riskkernel.kill.DirectiveSink` implementation the
    live compositions wire (``windbreak/scheduler/loop.py`` for the always-on
    PAPER loop, ``windbreak/main.py`` for ``--process riskkernel``). It reads
    the resting orders once, attempts every one of them, and reports counts.

    It cancels **orders only**. Nothing here reads, closes, or sells a position:
    the kill switch holds positions by design, and an adapter that reached for
    them would break that invariant from outside the kernel.
    """

    __slots__ = ("_venue",)

    def __init__(self, venue: RestingOrderVenue) -> None:
        """Wire the sink to the venue it cancels resting orders at.

        Args:
            venue: The trade-capable venue seam. This is deliberately the raw
                connector rather than a read-only view: cancelling is the whole
                point, and the read-only views elsewhere in the compositions
                exist to keep *other* stages away from exactly this capability.
        """
        self._venue = venue

    def submit(self, directive: CancelAllDirective) -> DirectiveDelivery:
        """Cancel every resting order at the venue, reporting what happened.

        The directive's scope is ``all_open_orders`` and there is no other, so
        it carries no instruction this method has to branch on; it is accepted
        to satisfy the seam and to keep the call site honest about what is being
        delivered.

        Args:
            directive: The cancel-all directive being delivered. Unused: the
                scope is the whole instruction.

        Returns:
            A :class:`~windbreak.ledger.directives.DirectiveDelivery` counting
            the orders the venue accepted and refused -- errored, with unknown
            counts, when the venue could not even be asked what was resting.
        """
        del directive
        orders = self._resting_orders()
        if orders is None:
            return DirectiveDelivery(errored=True)
        cancelled = sum(1 for order in orders if self._cancel(order.id))
        return DirectiveDelivery(cancelled=cancelled, failed=len(orders) - cancelled)

    def _resting_orders(self) -> tuple[OpenOrder, ...] | None:
        """Read the venue's resting orders once, or report that it could not.

        Returns:
            The resting orders, or ``None`` when the venue raised -- which is a
            different fact from an empty tuple, and must not be counted as
            "there was nothing to cancel".
        """
        try:
            return self._venue.get_open_orders()
        except Exception as exc:
            _LOGGER.critical(
                "cancel-all could not read the venue's resting orders: %s",
                exc,
                extra={"component": _COMPONENT},
            )
            return None

    def _cancel(self, order_id: str) -> bool:
        """Cancel one resting order, converting any failure into a False.

        Args:
            order_id: The venue order id to cancel.

        Returns:
            True when the venue accepted the cancellation, False when it
            raised. The exception -- and the order id -- go to the log and
            never to the caller, because the caller's report is ledgered.
        """
        try:
            self._venue.cancel_order(order_id)
        except Exception as exc:
            _LOGGER.critical(
                "cancel-all failed for resting order %s: %s",
                order_id,
                exc,
                extra={"component": _COMPONENT},
            )
            return False
        return True
