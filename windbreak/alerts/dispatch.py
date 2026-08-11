"""The alert dispatcher: fan one alert out to many sinks, isolating failures.

:class:`AlertDispatcher` sends an alert to every configured sink, converting
each sink's success or failure into a :class:`SinkOutcome`. A broken sink can
never take down another sink, the caller, or the ledger writer. When no sink
succeeds (including the empty-sink-list edge case), a fallback sink -- a
:class:`~windbreak.alerts.sinks.LogOnlySink` by default -- fires so an alert is
never silently lost.

Ledger persistence of the resulting :class:`AlertEmitted` (issue #13) is wired
through the :class:`LedgerWriter` protocol; this module ships a
:class:`LoggingLedgerWriter` that only logs, with no ``windbreak.ledger``
dependency.

**What may reach an append-only chain (issue #413).** :attr:`SinkOutcome.detail`
is ``str(exc)`` from an arbitrary sink -- the shape that leaked whole
token-bearing URLs in issue #274 -- and a hash chain can never be redacted. So
this module draws a line: ``outcomes`` is the full, unredacted record a log
line may use, and :attr:`AlertEmitted.deliveries` is the *closed* projection of
it that a ledger may hold. A :class:`SinkDelivery` carries a sink identity
screened against :func:`~windbreak.alerts.sinks.registered_sink_names` and a
:class:`~windbreak.alerts.delivery.DeliveryOutcome` from a four-member
enumeration -- no free-form field at all. :func:`ledger_deliveries` is the one
function that turns a dispatch into ledger payload rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol

from windbreak.alerts.delivery import DeliveryOutcome
from windbreak.alerts.registry import get_registration
from windbreak.alerts.sinks import (
    LogOnlySink,
    SinkSendError,
    classify_transport_failure,
    registered_sink_names,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from windbreak.alerts.registry import AlertSeverity, AlertType
    from windbreak.alerts.sinks import AlertSink

_LOGGER = logging.getLogger("windbreak.alerts")

#: The sink identity recorded when a sink's ``name`` is not one this codebase
#: defines. ``name`` is an arbitrary string on the :class:`AlertSink` protocol,
#: so it is a second free-form channel into an unredactable chain alongside
#: ``detail``; substituting a fixed token keeps the outcome on the record while
#: closing the channel. Chosen to collide with no real sink name -- a property
#: :func:`~windbreak.alerts.sinks.registered_sink_names` is asserted against.
UNREGISTERED_SINK: Final = "unregistered"


def _failure_outcome(exc: BaseException) -> DeliveryOutcome:
    """Classify a sink failure into the closed delivery vocabulary.

    A :class:`~windbreak.alerts.sinks.SinkSendError` already carries the
    outcome the sink chose for itself; anything else -- a sink that raised
    something other than the documented error -- is classified by exception
    type, never by message text.

    Args:
        exc: The exception the sink raised.

    Returns:
        The closed :class:`~windbreak.alerts.delivery.DeliveryOutcome`.
    """
    if isinstance(exc, SinkSendError):
        return exc.outcome
    return classify_transport_failure(exc)


@dataclass(frozen=True)
class SinkOutcome:
    """The result of attempting to deliver an alert through one sink.

    Attributes:
        sink: The sink's ``name``, verbatim and unscreened.
        outcome: The closed delivery outcome. ``ok`` is a view of it, so the
            two can never disagree.
        detail: Failure detail when delivery failed, else None. Arbitrary
            sink-supplied text; safe for a log line, never for the chain.
        fallback: Whether this attempt was the fallback sink rather than a
            configured one.
    """

    sink: str
    outcome: DeliveryOutcome
    detail: str | None = None
    fallback: bool = False

    @property
    def ok(self) -> bool:
        """Return whether this sink accepted the alert.

        Returns:
            True only for :attr:`DeliveryOutcome.DELIVERED`; every other member
            means the alert was not accepted.
        """
        return self.outcome is DeliveryOutcome.DELIVERED


@dataclass(frozen=True)
class SinkDelivery:
    """One sink's chain-safe delivery record: identity plus a closed outcome.

    Deliberately has no free-form field. ``sink`` is screened at construction
    against the sink names this codebase defines, so a sink cannot smuggle text
    into an append-only ledger through its own ``name``.

    Attributes:
        sink: The screened sink identity, or :data:`UNREGISTERED_SINK`.
        outcome: The closed delivery outcome.
        fallback: Whether this was the fallback sink rather than a configured
            one -- the difference between "the operator configured log-only"
            and "every real channel failed and this is all that is left".
    """

    sink: str
    outcome: DeliveryOutcome
    fallback: bool

    def __post_init__(self) -> None:
        """Replace an unrecognized sink identity with the closed token."""
        if self.sink not in registered_sink_names():
            object.__setattr__(self, "sink", UNREGISTERED_SINK)

    def as_payload(self) -> dict[str, object]:
        """Return this delivery as the closed mapping a ledger payload holds.

        Returns:
            A three-key mapping of the screened sink identity, the outcome's
            enum value, and the fallback flag. There is no fourth key, and in
            particular no ``detail``.
        """
        return {
            "sink": self.sink,
            "outcome": self.outcome.value,
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class AlertEmitted:
    """A record of one dispatched alert and every sink's outcome.

    Attributes:
        alert_type: The dispatched alert type.
        severity: The alert's severity.
        message: The alert body.
        outcomes: One outcome per attempted sink, in order. Carries unredacted
            sink detail; never persist it to a hash-chained ledger.
        ts: ISO-8601 UTC timestamp of dispatch.
    """

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    outcomes: tuple[SinkOutcome, ...]
    ts: str

    @property
    def deliveries(self) -> tuple[SinkDelivery, ...]:
        """Return the chain-safe projection of :attr:`outcomes`.

        Derived from ``outcomes`` rather than tracked alongside it, so the
        redacted view can never drift from the record it redacts.

        Returns:
            One :class:`SinkDelivery` per attempted sink, in attempt order.
        """
        return tuple(
            SinkDelivery(
                sink=outcome.sink, outcome=outcome.outcome, fallback=outcome.fallback
            )
            for outcome in self.outcomes
        )


class AlertDeliveryReport(Protocol):
    """The chain-safe view of one dispatch: closed per-sink deliveries only.

    The narrowed return type of the kill switch's dispatcher seam (issue #413).
    Structural, so both the real :class:`AlertEmitted` and a test double fit --
    but it exposes ``deliveries`` and nothing else, so a consumer typed against
    it cannot reach :attr:`SinkOutcome.detail` even by mistake. The closure
    against unredactable disclosure is therefore checked by mypy at the seam,
    not only by a test downstream of it.
    """

    @property
    def deliveries(self) -> tuple[SinkDelivery, ...]:
        """Return one closed delivery record per attempted sink, in order."""
        ...


def ledger_deliveries(report: AlertDeliveryReport | None) -> list[dict[str, object]]:
    """Project a dispatch into the closed rows a ledger payload may hold.

    The single producer of ledgered delivery evidence: everything that reaches
    an append-only chain about *who accepted an alert* passes through here, so
    there is one place to audit rather than one per call site.

    Args:
        report: The dispatch to project, or None when the dispatcher reported
            no delivery evidence at all.

    Returns:
        One closed three-key mapping per attempted sink, in attempt order; an
        empty list when ``report`` is None. Empty means *unreported*, never
        delivered -- absent evidence must never read as healthy.
    """
    if report is None:
        return []
    return [delivery.as_payload() for delivery in report.deliveries]


class LedgerWriter(Protocol):
    """The seam through which an emitted alert is persisted (issue #13)."""

    def record(self, event: AlertEmitted) -> None:
        """Persist an emitted-alert event.

        Args:
            event: The event to persist.
        """
        ...


class LoggingLedgerWriter:
    """A :class:`LedgerWriter` that logs events instead of persisting them.

    Stands in until the real ledger (issue #13) provides a persisting
    :class:`LedgerWriter`; it emits on the module ``windbreak.alerts`` logger.
    """

    def record(self, event: AlertEmitted) -> None:
        """Log the emitted-alert event as a single structured line.

        Args:
            event: The event to log.
        """
        summary = ", ".join(
            f"{outcome.sink}=ok:{outcome.ok}" for outcome in event.outcomes
        )
        _LOGGER.info(
            "alert emitted type=%s severity=%s message=%s",
            event.alert_type.value,
            event.severity.value,
            event.message,
            extra={
                "component": "alerts",
                "event": "AlertEmitted",
                "alert_type": event.alert_type.value,
                "severity": event.severity.value,
                "outcomes": summary,
            },
        )


def _utc_now_iso() -> str:
    """Return the current UTC time as ISO-8601 with a trailing ``Z``.

    Returns:
        A string like ``2026-07-04T12:00:00.000000Z``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class AlertDispatcher:
    """Fan an alert out to every sink, isolating and recording each outcome."""

    def __init__(
        self,
        sinks: Sequence[AlertSink],
        *,
        ledger_writer: LedgerWriter,
        fallback: AlertSink | None = None,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            sinks: The sinks to attempt for each alert, in order.
            ledger_writer: The writer that records each emitted event.
            fallback: The sink to fire when no primary sink succeeds.
                Defaults to a :class:`~windbreak.alerts.sinks.LogOnlySink`.
        """
        self._sinks = sinks
        self._ledger_writer = ledger_writer
        self._fallback: AlertSink = fallback if fallback is not None else LogOnlySink()

    def dispatch(self, alert_type: AlertType, message: str) -> AlertEmitted:
        """Send an alert to every sink, firing the fallback if none succeed.

        Args:
            alert_type: The alert type to dispatch.
            message: The alert body.

        Returns:
            The :class:`AlertEmitted` event describing every sink outcome.
        """
        severity = get_registration(alert_type).severity
        outcomes = [
            self._attempt(sink, alert_type, severity, message) for sink in self._sinks
        ]
        if not any(outcome.ok for outcome in outcomes):
            outcomes.append(
                self._attempt(
                    self._fallback, alert_type, severity, message, fallback=True
                )
            )
        event = AlertEmitted(
            alert_type=alert_type,
            severity=severity,
            message=message,
            outcomes=tuple(outcomes),
            ts=_utc_now_iso(),
        )
        self._record(event)
        return event

    def _attempt(
        self,
        sink: AlertSink,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        *,
        fallback: bool = False,
    ) -> SinkOutcome:
        """Attempt one sink, converting any exception into a failed outcome.

        Args:
            sink: The sink to send through.
            alert_type: The alert type to dispatch.
            severity: The alert's severity.
            message: The alert body.
            fallback: Whether this attempt is the fallback sink rather than a
                configured one.

        Returns:
            A delivered :class:`SinkOutcome` on success, or a failed one
            carrying both the closed outcome and the (unredacted, log-only)
            exception detail when the sink raises.
        """
        try:
            sink.send(alert_type, severity, message)
        except Exception as exc:
            _LOGGER.warning(
                "alert sink %r failed: %s",
                sink.name,
                exc,
                extra={"component": "alerts", "sink": sink.name},
            )
            return SinkOutcome(
                sink=sink.name,
                outcome=_failure_outcome(exc),
                detail=str(exc),
                fallback=fallback,
            )
        return SinkOutcome(
            sink=sink.name, outcome=DeliveryOutcome.DELIVERED, fallback=fallback
        )

    def _record(self, event: AlertEmitted) -> None:
        """Record an event via the ledger writer, never letting it raise.

        Args:
            event: The event to record.
        """
        try:
            self._ledger_writer.record(event)
        except Exception as exc:
            _LOGGER.warning(
                "ledger writer failed to record alert: %s",
                exc,
                extra={"component": "alerts"},
            )


def dispatch_hook(
    dispatcher: AlertDispatcher, alert_type: AlertType
) -> Callable[[AlertSeverity, str], None]:
    """Bind a dispatcher to the crosscheck's alert seam for one alert type.

    The returned callable is the seam
    :func:`windbreak.evaluation.crosscheck.crosscheck_gates` fires a mismatch
    into: it takes a severity/message pair and delivers the message through
    ``dispatcher`` under the pre-bound ``alert_type``. Severity is authoritative
    from the registry, so a caller-supplied severity that disagrees with the
    registration is logged as a WARNING and otherwise ignored -- the actual
    dispatch always uses the registry-derived severity that
    :meth:`AlertDispatcher.dispatch` looks up itself.

    The closure never raises: the registry lookup and
    :meth:`AlertDispatcher.dispatch` (which internally isolates every sink and
    ledger failure) are both non-raising, so ``crosscheck_gates`` can call it
    uncaught as its documented never-raising ``AlertHook``.

    Args:
        dispatcher: The dispatcher every alert is delivered through.
        alert_type: The alert type bound into the returned closure.

    Returns:
        A ``(severity, message) -> None`` callable that structurally satisfies
        :class:`windbreak.evaluation.crosscheck.AlertHook` (and
        :class:`windbreak.evaluation.live_divergence.AlertHook`) without the
        alerts package importing :mod:`windbreak.evaluation`. This is the
        producer side of the same structural-satisfaction boundary the
        :class:`windbreak.forecast.canary.CanaryAlertEmitter` precedent draws
        from the consumer side (there the consumer declares the protocol a real
        dispatcher satisfies; here the alerts package hands back a closure that
        satisfies the consumer's protocol) -- neither side imports the other.
    """
    registered_severity = get_registration(alert_type).severity

    def _hook(severity: AlertSeverity, message: str) -> None:
        """Dispatch ``message`` under the bound alert type, never raising.

        Args:
            severity: The caller-supplied severity; ignored for dispatch but
                warned about when it disagrees with the registered severity.
            message: The alert body to deliver.
        """
        if severity is not registered_severity:
            _LOGGER.warning(
                "alert hook severity %s disagrees with the registration for %s; "
                "dispatching at the registered %s",
                severity,
                alert_type,
                registered_severity,
                extra={"component": "alerts", "alert_type": alert_type.value},
            )
        dispatcher.dispatch(alert_type, message)

    return _hook
