"""Shared component: operator alerting primitives.

Defines the alert catalog (:class:`AlertType`, :class:`AlertSeverity`,
:data:`ALERT_REGISTRY`), the delivery channels (:class:`NtfySink`,
:class:`WebhookSink`, :class:`SmtpSink`, :class:`DesktopSink`,
:class:`LogOnlySink`), and the :class:`AlertDispatcher` that fans an alert out
to those channels while isolating failures. :func:`dispatch_hook` binds a
dispatcher to the crosscheck's ``(severity, message) -> None`` alert seam.

:func:`~windbreak.alerts.factory.build_sinks` is the config-driven composition
seam: it turns the ``alerts.sinks`` section of a
:class:`~windbreak.config.schema.WindbreakConfig` into the concrete ``*Sink``
instances a dispatcher fans out to, resolving each destination from the
environment variable configuration names for it (destinations are never config
leaves: a leaf is persisted verbatim into the hash-chained ``ConfigLoaded``
ledger event) and screening every one against the deployment's egress allowlist
(issue #274). It is deliberately *not*
re-exported from this package: importing it pulls :mod:`windbreak.config` and
:mod:`windbreak.net` in behind it, and the alert primitives below must stay
usable without either.

Example:
    >>> from windbreak.alerts import AlertDispatcher, AlertType, LoggingLedgerWriter
    >>> dispatcher = AlertDispatcher(sinks=[], ledger_writer=LoggingLedgerWriter())
    >>> event = dispatcher.dispatch(AlertType.MODE_CHANGE, "switched to PAPER")

One dependency-injection seam is still wired by a successor issue: the ledger
(issue #13) will provide a real :class:`LedgerWriter` that persists each
:class:`AlertEmitted` in place of :class:`LoggingLedgerWriter`.
"""

from windbreak.alerts.dispatch import (
    AlertDispatcher,
    AlertEmitted,
    LedgerWriter,
    LoggingLedgerWriter,
    SinkOutcome,
    dispatch_hook,
)
from windbreak.alerts.registry import (
    ALERT_REGISTRY,
    AlertRegistration,
    AlertSeverity,
    AlertType,
    cli_token,
    get_registration,
)
from windbreak.alerts.sinks import (
    DesktopSink,
    LogOnlySink,
    NtfySink,
    NtfySinkConfig,
    SinkSendError,
    SmtpSink,
    SmtpSinkConfig,
    WebhookSink,
    WebhookSinkConfig,
)

__all__ = [
    "ALERT_REGISTRY",
    "AlertDispatcher",
    "AlertEmitted",
    "AlertRegistration",
    "AlertSeverity",
    "AlertType",
    "DesktopSink",
    "LedgerWriter",
    "LogOnlySink",
    "LoggingLedgerWriter",
    "NtfySink",
    "NtfySinkConfig",
    "SinkOutcome",
    "SinkSendError",
    "SmtpSink",
    "SmtpSinkConfig",
    "WebhookSink",
    "WebhookSinkConfig",
    "cli_token",
    "dispatch_hook",
    "get_registration",
]
