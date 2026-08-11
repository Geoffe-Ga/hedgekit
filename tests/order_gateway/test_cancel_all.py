"""The venue-side consumer of the kill switch's cancel-all directive (#480).

`VenueCancelAllSink` is the thing that was missing: until issue #480 the one
`CancelAllDirective` a kill ledgered was consumed by nothing, so a resting order
survived a kill as a live instruction that could still fill after the operator
had walked away.

What this module pins, and why each shape is what it is:

* **The venue is actually called**, with every resting order's id, in book
  order -- asserted against the double's own record, never against a return
  value the sink could have invented.
* **One bad order never stops the rest.** The sink mirrors
  `AlertDispatcher`'s isolation discipline: a venue that refuses order two
  still gets asked about order three. Three orders with exactly one refusal is
  the smallest fixture that can tell "kept going" from "stopped at the first
  failure" *and* from "counted them all as failed".
* **Failure never escapes.** A kill's remaining fail-safe effects run after the
  sink returns, so nothing here may raise -- not a refusing `cancel_order`, not
  a venue that cannot even be asked what is resting.
* **Nothing venue-supplied comes back.** The returned `DirectiveDelivery` is
  ledgered onto an append-only chain, so it carries counts and a closed
  outcome; the order ids and the rejection text stay in the log.
* **Positions are never touched.** The kill switch holds positions by design,
  and an adapter reaching for them would break that invariant from outside the
  kernel, so the double fails loudly if its position surface is read at all.

The counts in every fixture below are deliberately distinct from one another
(three orders, one failure, two cancellations), so no assertion can pass by two
different quantities happening to coincide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from windbreak.connector.models import OpenOrder
from windbreak.ledger.directives import DirectiveOutcome
from windbreak.ledger.events import CancelAllDirective
from windbreak.numeric.types import ContractCentis, PricePips
from windbreak.order_gateway.cancel_all import VenueCancelAllSink

if TYPE_CHECKING:
    from collections.abc import Iterable

    from windbreak.order_gateway.cancel_all import RestingOrderVenue

#: The three resting order ids every fixture below starts with. Three, so the
#: "kept going after a refusal" assertion has a third order to have reached.
_ORDER_IDS = ("venue-a", "venue-b", "venue-c")

#: The id whose cancellation the refusing venue rejects. The *middle* one, so a
#: sink that stopped at the first failure and one that ran to completion differ
#: in what they attempted, not merely in what they counted.
_REFUSED_ID = "venue-b"

#: Credential-bearing text a venue client raises, standing in for the category
#: issue #274 leaked: a URL whose path segment *is* a capability. Nothing
#: derived from it may appear in anything the sink returns.
_VENUE_FAILURE_DETAIL = "POST https://venue.example/v1/s3kr1t-480-token/cancel 403"

#: The one directive the kill switch ever emits.
_DIRECTIVE = CancelAllDirective(component="riskkernel", scope="all_open_orders")


@dataclass
class _RecordingVenue:
    """A `RestingOrderVenue` double recording every cancellation attempt.

    Attributes:
        refuse: The order ids whose cancellation this venue rejects, by
            raising credential-bearing text.
        attempted: Every order id `cancel_order` was called with, in call
            order -- so "was asked" is observable separately from "succeeded".
        remaining: The ids still resting, in book order.
    """

    refuse: frozenset[str] = field(default_factory=frozenset)
    attempted: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=lambda: list(_ORDER_IDS))

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Return the orders still resting, in book order."""
        return tuple(_open_order(order_id) for order_id in self.remaining)

    def cancel_order(self, order_id: str) -> None:
        """Record the attempt, then retire the order or refuse it.

        Args:
            order_id: The venue order id to cancel.

        Raises:
            RuntimeError: If ``order_id`` is in :attr:`refuse`, carrying
                :data:`_VENUE_FAILURE_DETAIL`.
        """
        self.attempted.append(order_id)
        if order_id in self.refuse:
            raise RuntimeError(_VENUE_FAILURE_DETAIL)
        self.remaining.remove(order_id)

    def get_positions(self) -> tuple[object, ...]:
        """Raise; a cancel-all must never read, let alone touch, a position.

        Raises:
            AssertionError: Always -- reaching this is the position-hold
                invariant being broken from outside the kernel.
        """
        raise AssertionError("cancel-all must never read positions")


@dataclass
class _UnreadableVenue:
    """A `RestingOrderVenue` double that cannot even be asked what is resting.

    The distinct failure mode from a refusing cancel: the sink never learns how
    many orders there were, so zero counts here would be a lie rather than a
    fact.

    Attributes:
        cancelled: Every id `cancel_order` was called with; must stay empty.
    """

    cancelled: list[str] = field(default_factory=list)

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Raise, as an unreachable venue does.

        Raises:
            RuntimeError: Always, carrying :data:`_VENUE_FAILURE_DETAIL`.
        """
        raise RuntimeError(_VENUE_FAILURE_DETAIL)

    def cancel_order(self, order_id: str) -> None:
        """Record a cancellation that should never happen.

        Args:
            order_id: The order id unexpectedly cancelled.
        """
        self.cancelled.append(order_id)


def _open_order(order_id: str) -> OpenOrder:
    """Build one resting order with the given id.

    Args:
        order_id: The venue order id.

    Returns:
        A resting YES order; only its ``id`` matters to the sink.
    """
    return OpenOrder(
        id=order_id,
        ticker="MKT-480",
        side="yes",
        price=PricePips(4_400),
        quantity=ContractCentis(100),
    )


def _payload_strings(payload: dict[str, object]) -> Iterable[str]:
    """Yield every string reachable in a ledger payload mapping.

    Args:
        payload: The payload to walk.

    Yields:
        Each string key and each string value, at any nesting depth.
    """
    for key, value in payload.items():
        yield str(key)
        if isinstance(value, str):
            yield value


def test_the_sink_cancels_every_resting_order_at_the_venue() -> None:
    """Every resting order is cancelled, in book order, and the book empties.

    The whole point of issue #480, asserted at the venue rather than at the
    ledger row that claimed it for months while it was untrue.
    """
    venue = _RecordingVenue()

    delivery = VenueCancelAllSink(venue).submit(_DIRECTIVE)

    assert tuple(venue.attempted) == _ORDER_IDS
    assert venue.remaining == []
    assert delivery.cancelled == 3
    assert delivery.failed == 0
    assert delivery.errored is False
    assert delivery.outcome is DirectiveOutcome.DELIVERED
    assert delivery.fully_delivered is True


def test_one_refused_order_does_not_stop_the_sink_asking_about_the_rest() -> None:
    """A venue that refuses the middle order is still asked about the last one.

    One broken order can never take down the others, exactly as one broken
    alert sink can never take down another. Asserted on what the venue was
    *asked* -- all three ids -- because a sink that gave up after the refusal
    would report the same `failed` count while leaving a live order resting.
    """
    venue = _RecordingVenue(refuse=frozenset({_REFUSED_ID}))

    delivery = VenueCancelAllSink(venue).submit(_DIRECTIVE)

    assert tuple(venue.attempted) == _ORDER_IDS
    assert venue.remaining == [_REFUSED_ID]
    assert delivery.cancelled == 2
    assert delivery.failed == 1
    assert delivery.outcome is DirectiveOutcome.PARTIAL
    assert delivery.fully_delivered is False


def test_a_venue_refusing_every_order_reports_refused_and_never_raises() -> None:
    """Every refusal is counted, nothing escapes, and nothing reads delivered."""
    venue = _RecordingVenue(refuse=frozenset(_ORDER_IDS))

    delivery = VenueCancelAllSink(venue).submit(_DIRECTIVE)

    assert tuple(venue.attempted) == _ORDER_IDS
    assert venue.remaining == list(_ORDER_IDS)
    assert delivery.cancelled == 0
    assert delivery.failed == 3
    assert delivery.outcome is DirectiveOutcome.REFUSED
    assert delivery.fully_delivered is False


def test_an_empty_book_is_a_complete_delivery_not_a_silent_one() -> None:
    """Nothing resting cancels nothing, and says so with zero counts.

    Vacuous completion is the correct reading -- there was nothing to cancel --
    and it is distinguishable from every failure mode by `errored is False`
    with a zero `failed`.
    """
    venue = _RecordingVenue(remaining=[])

    delivery = VenueCancelAllSink(venue).submit(_DIRECTIVE)

    assert venue.attempted == []
    assert delivery.cancelled == 0
    assert delivery.failed == 0
    assert delivery.errored is False
    assert delivery.outcome is DirectiveOutcome.DELIVERED
    assert delivery.fully_delivered is True


def test_a_venue_that_cannot_be_read_errors_rather_than_reporting_zero() -> None:
    """An unreadable book is unknown, never "there was nothing to cancel".

    Zero counts with `errored is False` is the positive claim that the venue
    was asked and had nothing resting. A venue that raised on the *question*
    has made no such claim, so it must be `errored` -- and it must still not
    raise, because the kill's page and `KILL` file come after this call.
    """
    venue = _UnreadableVenue()

    delivery = VenueCancelAllSink(venue).submit(_DIRECTIVE)

    assert venue.cancelled == []
    assert delivery.errored is True
    assert delivery.outcome is DirectiveOutcome.ERRORED
    assert delivery.fully_delivered is False


@pytest.mark.parametrize(
    "venue",
    [
        _RecordingVenue(refuse=frozenset({_REFUSED_ID})),
        _RecordingVenue(refuse=frozenset(_ORDER_IDS)),
        _UnreadableVenue(),
    ],
    ids=["partial", "refused", "unreadable"],
)
def test_no_venue_text_survives_into_the_ledger_payload(
    venue: RestingOrderVenue,
) -> None:
    """Nothing a venue said reaches the mapping the chain will hash (#274).

    Every failure path is swept, not just one, because the closure has to hold
    on all of them. The positive control is the assertion that the payload is
    *non-empty* and carries the exact three keys: a sweep over an empty mapping
    would pass forever while proving nothing.

    Args:
        venue: The failing venue double this case sweeps.
    """
    delivery = VenueCancelAllSink(venue).submit(_DIRECTIVE)
    payload = delivery.as_payload()
    strings = list(_payload_strings(payload))

    assert set(payload) == {"cancelled", "failed", "outcome"}
    assert strings != []
    assert all("s3kr1t-480-token" not in text for text in strings)
    assert all("venue.example" not in text for text in strings)
    assert all(order_id not in strings for order_id in _ORDER_IDS)
