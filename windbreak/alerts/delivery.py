"""The closed vocabulary a delivery outcome may be recorded as (issue #413).

A kill-switch ``AlertEmitted`` row proves an alert was *emitted*; it says
nothing about whether any sink accepted it. Recording per-sink outcomes closes
that gap, but the obvious payload -- ``SinkOutcome.detail``, which is
``str(exc)`` from an arbitrary sink -- is the exact shape that leaked whole
token-bearing URLs in issue #274, and a hash chain is append-only: nothing
written into it can ever be redacted.

:class:`DeliveryOutcome` is therefore the *only* description of a failure the
chain ever sees. It is a four-member enumeration the code controls end to end,
so no sink-supplied text can travel on it however a sink fails.

This module deliberately imports nothing from :mod:`windbreak.alerts`: both
:mod:`windbreak.alerts.sinks` (which classifies its own transport failures) and
:mod:`windbreak.alerts.dispatch` (which projects them for the ledger) depend on
it, so it must sit below both.
"""

from __future__ import annotations

import enum


class DeliveryOutcome(enum.Enum):
    """How one sink answered one alert -- the closed, chain-safe vocabulary.

    ``DELIVERED`` is the only member that means the alert was accepted; every
    other member means it was not, and the distinction between them is
    diagnostic only. A reader that treats anything but ``DELIVERED`` as success
    would be failing open.
    """

    DELIVERED = "delivered"
    REFUSED = "refused"
    TIMED_OUT = "timed_out"
    ERRORED = "errored"
