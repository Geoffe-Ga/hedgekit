"""The closed delivery vocabulary the hash chain is allowed to hold (issue #413).

A kill-switch ``AlertEmitted`` row proves the alert was *emitted*. It proves
nothing about whether any sink accepted it, so an audit cannot establish that
anyone was told. Recording per-sink outcomes closes that gap -- but
``SinkOutcome.detail`` is ``str(exc)`` from an arbitrary sink, the exact shape
that leaked whole token-bearing URLs in issue #274, and a hash chain is
append-only: nothing written into it can ever be redacted.

So the chain-facing projection is *closed*: a sink identity drawn from the set
of sink classes the code itself defines, plus a :class:`DeliveryOutcome` drawn
from a four-member enum. This module pins that closure -- every failure mode a
sink can present maps onto one of the four members, an unregistered sink name
is replaced rather than echoed, and the projection drops ``detail`` entirely.

RED before the implementation: :mod:`windbreak.alerts.delivery` does not exist,
so this module fails at collection with ``ModuleNotFoundError``.
"""

from __future__ import annotations

import dataclasses

import pytest

from windbreak.alerts.delivery import DeliveryOutcome
from windbreak.alerts.dispatch import (
    UNREGISTERED_SINK,
    AlertDispatcher,
    LoggingLedgerWriter,
    SinkDelivery,
    SinkOutcome,
    ledger_deliveries,
)
from windbreak.alerts.registry import AlertSeverity, AlertType
from windbreak.alerts.sinks import (
    DesktopSink,
    LogOnlySink,
    NtfySink,
    SinkSendError,
    SmtpSink,
    WebhookSink,
    WebhookSinkConfig,
    classify_transport_failure,
    registered_sink_names,
)
from windbreak.net.allowlist import OutboundAllowlist

#: The allowlisted host every webhook built here posts to.
_WEBHOOK_HOST = "hooks.example.com"

#: A token-bearing destination of exactly the shape issue #274 found echoed
#: whole out of a denial message. Used as a sink's exception text so a test can
#: prove the closed projection carries no fragment of it.
_TOKEN_URL = f"https://{_WEBHOOK_HOST}/services/s3cr3t-token?key=abc123"

#: Spelled once so the `dataclasses.fields` guard below reads as a field check
#: rather than as a second construction.
_DELIVERED = DeliveryOutcome.DELIVERED


class _RaisingSink:
    """An `AlertSink` that always fails with a caller-supplied exception."""

    def __init__(self, name: str, error: Exception) -> None:
        """Initialize the sink.

        Args:
            name: The sink's reported ``name``.
            error: The exception every ``send`` raises.
        """
        self.name = name
        self._error = error

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Raise the configured exception.

        Args:
            alert_type: The alert type (ignored).
            severity: The alert's severity (ignored).
            message: The alert body (ignored).

        Raises:
            Exception: Always -- the exception this sink was built with.
        """
        del alert_type, severity, message
        raise self._error


class _AcceptingSink:
    """An `AlertSink` named ``webhook`` that always accepts the alert."""

    name = WebhookSink.name

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Accept and discard the alert.

        Args:
            alert_type: The alert type (ignored).
            severity: The alert's severity (ignored).
            message: The alert body (ignored).
        """
        del alert_type, severity, message


def test_the_delivery_vocabulary_is_exactly_four_closed_members() -> None:
    """`DeliveryOutcome` is the closed four-member enumeration issue #413 names."""
    assert {member.name: member.value for member in DeliveryOutcome} == {
        "DELIVERED": "delivered",
        "REFUSED": "refused",
        "TIMED_OUT": "timed_out",
        "ERRORED": "errored",
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("read timed out"), DeliveryOutcome.TIMED_OUT),
        (ConnectionRefusedError("connection refused"), DeliveryOutcome.REFUSED),
        (ValueError("something else"), DeliveryOutcome.ERRORED),
    ],
)
def test_a_raw_transport_failure_maps_onto_the_closed_vocabulary(
    error: Exception, expected: DeliveryOutcome
) -> None:
    """Every transport exception classifies by *type*, never by its message.

    Args:
        error: The raw transport exception to classify.
        expected: The closed outcome it must map onto.
    """
    assert classify_transport_failure(error) is expected


def test_a_sink_send_error_defaults_to_errored_and_carries_its_own_outcome() -> None:
    """`SinkSendError` declares its closed outcome, defaulting to ERRORED."""
    assert SinkSendError("boom").outcome is DeliveryOutcome.ERRORED
    assert (
        SinkSendError("nope", outcome=DeliveryOutcome.REFUSED).outcome
        is DeliveryOutcome.REFUSED
    )


def test_registered_sink_names_are_derived_from_the_sink_classes() -> None:
    """The closed sink vocabulary is derived from `sinks`, never hand-restated.

    Derived, so a sink class added to the module joins the vocabulary without
    anyone remembering to update a list -- the drift a hand-restated tuple
    invites. The set is asserted non-empty first: a screen against an empty
    vocabulary would reject every name and pass vacuously.
    """
    names = registered_sink_names()

    assert names
    assert names == {
        NtfySink.name,
        WebhookSink.name,
        SmtpSink.name,
        DesktopSink.name,
        LogOnlySink.name,
    }


def test_an_unregistered_sink_name_is_replaced_rather_than_echoed() -> None:
    """A sink whose `name` is not in the vocabulary cannot write it to the chain.

    ``name`` is an arbitrary string on the `AlertSink` protocol, so it is a
    second free-form channel into an append-only chain alongside ``detail``.
    Substituting a fixed token keeps the outcome recorded while closing it.
    """
    delivery = SinkDelivery(
        sink=_TOKEN_URL, outcome=DeliveryOutcome.ERRORED, fallback=False
    )

    assert delivery.sink == UNREGISTERED_SINK
    assert UNREGISTERED_SINK not in registered_sink_names()


def test_a_registered_sink_name_survives_the_screen_verbatim() -> None:
    """The screen replaces only unregistered names, never every name."""
    delivery = SinkDelivery(
        sink=WebhookSink.name, outcome=DeliveryOutcome.DELIVERED, fallback=False
    )

    assert delivery.sink == WebhookSink.name


def test_the_closed_payload_has_exactly_three_keys_and_no_detail() -> None:
    """`SinkDelivery.as_payload` emits the closed three-key shape, and only it."""
    payload = SinkDelivery(
        sink=LogOnlySink.name, outcome=DeliveryOutcome.DELIVERED, fallback=True
    ).as_payload()

    assert payload == {
        "sink": "log-only",
        "outcome": "delivered",
        "fallback": True,
    }


def test_the_dispatcher_projects_a_failed_sink_without_its_exception_text() -> None:
    """A dispatched alert's `deliveries` carry the outcome but not the detail.

    The unredacted `str(exc)` stays available on ``outcomes`` for the log line;
    ``deliveries`` is the projection the append-only chain is handed, and no
    fragment of the exception text survives it.
    """
    sink = _RaisingSink(WebhookSink.name, SinkSendError(_TOKEN_URL))
    dispatcher = AlertDispatcher([sink], ledger_writer=LoggingLedgerWriter())

    event = dispatcher.dispatch(AlertType.HALT_KILL, "halted")

    assert event.outcomes[0].detail == _TOKEN_URL
    assert event.deliveries == (
        SinkDelivery(
            sink=WebhookSink.name, outcome=DeliveryOutcome.ERRORED, fallback=False
        ),
        SinkDelivery(
            sink=LogOnlySink.name, outcome=DeliveryOutcome.DELIVERED, fallback=True
        ),
    )
    assert (
        "detail"
        not in SinkDelivery(
            sink=WebhookSink.name, outcome=DeliveryOutcome.ERRORED, fallback=False
        ).as_payload()
    )


def test_the_fallback_flag_marks_only_the_fallback_attempt() -> None:
    """`fallback` is True on the fallback sink alone, never on a configured one.

    Without it a ledgered ``log-only: delivered`` is ambiguous between "the
    operator configured log-only" and "every real channel failed and this is
    all that is left" -- which is the whole question an audit asks.
    """
    dispatcher = AlertDispatcher(
        [_AcceptingSink()], ledger_writer=LoggingLedgerWriter()
    )

    event = dispatcher.dispatch(AlertType.HALT_KILL, "halted")

    assert [delivery.fallback for delivery in event.deliveries] == [False]
    assert event.deliveries[0].outcome is DeliveryOutcome.DELIVERED


def test_sink_outcome_ok_is_derived_from_its_closed_outcome() -> None:
    """`SinkOutcome.ok` is a view of `outcome`, so the two can never disagree."""
    assert SinkOutcome(sink="a", outcome=DeliveryOutcome.DELIVERED).ok is True
    assert SinkOutcome(sink="a", outcome=DeliveryOutcome.REFUSED).ok is False
    assert SinkOutcome(sink="a", outcome=DeliveryOutcome.TIMED_OUT).ok is False
    assert SinkOutcome(sink="a", outcome=DeliveryOutcome.ERRORED).ok is False
    assert "ok" not in {
        f.name for f in dataclasses.fields(SinkOutcome(sink="a", outcome=_DELIVERED))
    }


def test_a_non_2xx_webhook_response_is_recorded_as_refused() -> None:
    """A destination that answers and declines classifies REFUSED, not ERRORED.

    This is the one path that makes REFUSED reachable from a real sink, so the
    four-member vocabulary is four *live* members rather than three plus a
    decorative one.
    """
    sink = WebhookSink(
        WebhookSinkConfig(url=f"https://{_WEBHOOK_HOST}/incoming"),
        transport=lambda url, body, headers: 503,
        allowlist=OutboundAllowlist(frozenset({_WEBHOOK_HOST})),
    )
    dispatcher = AlertDispatcher([sink], ledger_writer=LoggingLedgerWriter())

    event = dispatcher.dispatch(AlertType.HALT_KILL, "halted")

    assert event.deliveries[0] == SinkDelivery(
        sink=WebhookSink.name, outcome=DeliveryOutcome.REFUSED, fallback=False
    )


def test_ledger_deliveries_of_no_report_is_empty_rather_than_invented() -> None:
    """A dispatcher that reports nothing yields no rows, never a fabricated one.

    Fail closed: absent delivery evidence must read as absent, never as a
    delivery that was never observed.
    """
    assert ledger_deliveries(None) == []


def test_ledger_deliveries_projects_every_attempt_in_order() -> None:
    """The chain-facing projection preserves attempt order, one entry per sink."""
    sink = _RaisingSink(WebhookSink.name, TimeoutError("read timed out"))
    dispatcher = AlertDispatcher([sink], ledger_writer=LoggingLedgerWriter())

    event = dispatcher.dispatch(AlertType.HALT_KILL, "halted")

    assert ledger_deliveries(event) == [
        {"sink": "webhook", "outcome": "timed_out", "fallback": False},
        {"sink": "log-only", "outcome": "delivered", "fallback": True},
    ]
