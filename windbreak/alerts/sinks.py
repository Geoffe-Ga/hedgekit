"""Outbound alert channels (sinks) and their default network transports.

Each sink implements the :class:`AlertSink` protocol -- a ``name`` plus
``send(alert_type, severity, message)`` -- and delegates its actual network
call to an injectable transport, so tests exercise the wiring without touching
a socket. Every delivery failure surfaces uniformly as :class:`SinkSendError`.

Each ``*SinkConfig`` is intentionally a plain dataclass with no
``windbreak.config`` dependency: the config-to-sink mapping lives one module
over, in :mod:`windbreak.alerts.factory`, so this module stays usable (and
testable) without a configuration at all.

Every sink that dials the network -- :class:`NtfySink`, :class:`WebhookSink`,
and :class:`SmtpSink` -- requires an :class:`~windbreak.net.allowlist.OutboundAllowlist`
and screens its destination at *construction*, so an off-allowlist host is
refused before a single packet leaves. :class:`DesktopSink` is exempt because it
never leaves the machine, and :class:`LogOnlySink` because it never leaves the
process.
"""

from __future__ import annotations

import http.client
import json
import logging
import smtplib
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.message import EmailMessage
from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import urlsplit, urlunsplit

from windbreak.alerts.delivery import DeliveryOutcome
from windbreak.alerts.registry import AlertSeverity

if TYPE_CHECKING:
    from windbreak.alerts.registry import AlertType
    from windbreak.net.allowlist import OutboundAllowlist

#: Seconds to wait for an alert transport (HTTPS or SMTP) to respond.
_TRANSPORT_TIMEOUT_SECONDS: Final = 10.0

#: Lowest HTTP status code considered a success (inclusive).
_HTTP_OK_MIN: Final = 200

#: Lowest HTTP status code considered a failure (i.e. the first non-2xx code).
_HTTP_OK_EXCLUSIVE_MAX: Final = 300

#: ntfy priority header value for each severity (1 = min, 5 = max).
_NTFY_PRIORITY: Final[Mapping[AlertSeverity, str]] = {
    AlertSeverity.INFO: "3",
    AlertSeverity.WARNING: "4",
    AlertSeverity.CRITICAL: "5",
}


def classify_transport_failure(exc: BaseException) -> DeliveryOutcome:
    """Map a raw transport exception onto the closed delivery vocabulary.

    Classification is by exception *type* alone, never by parsing the
    exception's message: a message is arbitrary sink-supplied text (issue #274)
    and must never become the thing a fail-closed audit record depends on.

    Args:
        exc: The exception a transport raised.

    Returns:
        :attr:`DeliveryOutcome.TIMED_OUT` for a timeout,
        :attr:`DeliveryOutcome.REFUSED` for a refused connection, and
        :attr:`DeliveryOutcome.ERRORED` for everything else -- the fail-closed
        default, since an unrecognized failure is still a failure.
    """
    if isinstance(exc, TimeoutError):
        return DeliveryOutcome.TIMED_OUT
    if isinstance(exc, ConnectionRefusedError):
        return DeliveryOutcome.REFUSED
    return DeliveryOutcome.ERRORED


def registered_sink_names() -> frozenset[str]:
    """Return the closed set of sink names this module defines.

    Derived by introspecting the module rather than hand-restated, so a sink
    class added here joins the vocabulary without anyone remembering to update
    a list -- the drift a mirrored tuple invites. A class qualifies when it is
    defined in this module and carries both a string ``name`` and a callable
    ``send``; :class:`AlertSink` itself is excluded because its ``name`` is an
    annotation with no value.

    Returns:
        Every concrete sink's ``name``, as the vocabulary a chain-facing sink
        identity is screened against.
    """
    names: set[str] = set()
    for obj in list(globals().values()):
        if not isinstance(obj, type) or obj.__module__ != __name__:
            continue
        name = getattr(obj, "name", None)
        if isinstance(name, str) and callable(getattr(obj, "send", None)):
            names.add(name)
    return frozenset(names)


class SinkSendError(Exception):
    """Raised when an alert sink fails to deliver a message.

    Attributes:
        outcome: The closed :class:`DeliveryOutcome` this failure is recorded
            as. Carried on the exception -- rather than inferred later from its
            message -- so the one description of the failure that reaches the
            append-only ledger is chosen by the code that knows what happened.
    """

    def __init__(
        self, message: str, *, outcome: DeliveryOutcome = DeliveryOutcome.ERRORED
    ) -> None:
        """Initialize the error with its message and closed outcome.

        Args:
            message: The human-readable failure detail. May contain arbitrary
                transport text, including a full destination URL, so it is
                never persisted to the hash-chained ledger.
            outcome: The closed outcome this failure is recorded as. Defaults
                to :attr:`DeliveryOutcome.ERRORED`, the fail-closed choice.
        """
        super().__init__(message)
        self.outcome = outcome


class AlertSink(Protocol):
    """The contract every alert delivery channel implements.

    Attributes:
        name: A short, non-empty identifier for the channel.
    """

    name: str

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Deliver one alert through this channel.

        Args:
            alert_type: The kind of alert being delivered.
            severity: The alert's severity.
            message: The human-readable alert body.

        Raises:
            SinkSendError: If delivery fails.
        """
        ...


@dataclass(frozen=True)
class NtfySinkConfig:
    """Connection settings for an ntfy topic sink.

    Attributes:
        base_url: The ntfy server base URL (must be ``https://``).
        topic: The ntfy topic to publish alerts to.
    """

    base_url: str
    topic: str


@dataclass(frozen=True)
class WebhookSinkConfig:
    """Connection settings for a generic JSON webhook sink.

    Attributes:
        url: The ``https://`` endpoint to POST alert payloads to.
    """

    url: str


@dataclass(frozen=True)
class SmtpSinkConfig:
    """Connection settings for an SMTP email sink.

    Attributes:
        host: The SMTP server hostname.
        port: The SMTP server port.
        sender: The envelope/from address.
        recipients: The recipient addresses.
    """

    host: str
    port: int
    sender: str
    recipients: tuple[str, ...]


#: Signature of an HTTP transport: ``(url, body, headers) -> status_code``.
HttpTransport = Callable[[str, bytes, Mapping[str, str]], int]

#: Signature of an SMTP transport: ``(config, message) -> None``.
SmtpTransport = Callable[[SmtpSinkConfig, EmailMessage], None]


def _request_path(url: str) -> str:
    """Extract the request-path (with any query) from an absolute URL.

    Args:
        url: The absolute URL to split.

    Returns:
        The path-and-query portion of ``url`` (for example ``/alerts``);
        ``urlunsplit`` omits the ``?`` when the query is empty.
    """
    parts = urlsplit(url)
    return urlunsplit(("", "", parts.path, parts.query, ""))


def _https_post(url: str, body: bytes, headers: Mapping[str, str]) -> int:
    """POST a body over HTTPS and return the response status code.

    Args:
        url: The target URL; must use the ``https`` scheme.
        body: The raw request body.
        headers: The request headers.

    Returns:
        The HTTP response status code (always 2xx on return).

    Raises:
        SinkSendError: If ``url`` is not ``https://`` (recorded as the
            fail-closed :attr:`DeliveryOutcome.ERRORED`, because nothing was
            ever dialled) or the response is not a 2xx status (recorded as
            :attr:`DeliveryOutcome.REFUSED` -- the destination answered and
            declined, which is a different operational fact from an
            unrecognized failure and is what an operator reads off the ledger).
    """
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise SinkSendError(f"refusing non-https URL: {url!r}")
    context = ssl.create_default_context()
    connection = http.client.HTTPSConnection(
        parts.netloc, timeout=_TRANSPORT_TIMEOUT_SECONDS, context=context
    )
    try:
        connection.request("POST", _request_path(url), body, dict(headers))
        status = connection.getresponse().status
    finally:
        connection.close()
    if not _HTTP_OK_MIN <= status < _HTTP_OK_EXCLUSIVE_MAX:
        raise SinkSendError(
            f"HTTPS POST to {url!r} returned status {status}",
            outcome=DeliveryOutcome.REFUSED,
        )
    return status


def _smtp_send(config: SmtpSinkConfig, message: EmailMessage) -> None:
    """Send an email via SMTP, upgrading the connection with STARTTLS.

    Args:
        config: The SMTP connection settings.
        message: The message to send.

    Raises:
        OSError: If the SMTP connection cannot be established.
        smtplib.SMTPException: If the SMTP conversation (STARTTLS or send)
            fails.
        ssl.SSLError: If the TLS handshake fails.

    Note:
        These transport/TLS errors propagate to the caller
        (:meth:`SmtpSink.send`), which wraps them in :class:`SinkSendError`,
        for parity with :func:`_https_post`.
    """
    context = ssl.create_default_context()
    with smtplib.SMTP(
        config.host, config.port, timeout=_TRANSPORT_TIMEOUT_SECONDS
    ) as client:
        client.starttls(context=context)
        client.send_message(message)


def _send_http(
    transport: HttpTransport, url: str, body: bytes, headers: Mapping[str, str]
) -> None:
    """Invoke an HTTP transport, translating any failure to SinkSendError.

    Args:
        transport: The HTTP transport to call.
        url: The target URL.
        body: The request body.
        headers: The request headers.

    Raises:
        SinkSendError: If the transport raises or returns a non-2xx status,
            carrying the closed :class:`DeliveryOutcome` the failure is
            recorded as. A transport that raises :class:`SinkSendError` has
            already chosen that outcome for itself and propagates *unchanged* --
            re-classifying it here would discard the choice and flatten the
            shipped transport's ``REFUSED`` for a declining destination
            (:func:`_https_post`) back to ``ERRORED``. Anything else a transport
            raises is classified by exception *type*; a transport that instead
            *returns* a non-2xx status (permitted by the
            :data:`HttpTransport` signature, though :func:`_https_post` raises
            rather than returning one) is recorded ``REFUSED`` on the same
            reasoning.
    """
    try:
        status = transport(url, body, headers)
    except SinkSendError:
        raise
    except Exception as exc:
        raise SinkSendError(str(exc), outcome=classify_transport_failure(exc)) from exc
    if not _HTTP_OK_MIN <= status < _HTTP_OK_EXCLUSIVE_MAX:
        raise SinkSendError(
            f"transport for {url!r} returned status {status}",
            outcome=DeliveryOutcome.REFUSED,
        )


#: The real HTTPS transport every network sink defaults to. Exposed under a
#: public name so :mod:`windbreak.alerts.factory` can hand it through as an
#: ordinary argument instead of branching on "was a test transport injected?".
DEFAULT_HTTP_TRANSPORT: Final[HttpTransport] = _https_post

#: The real SMTP transport :class:`SmtpSink` defaults to; public for the same
#: reason as :data:`DEFAULT_HTTP_TRANSPORT`.
DEFAULT_SMTP_TRANSPORT: Final[SmtpTransport] = _smtp_send


class NtfySink:
    """Publish alerts to an ntfy topic over HTTPS."""

    name = "ntfy"

    def __init__(
        self,
        config: NtfySinkConfig,
        *,
        transport: HttpTransport = _https_post,
        allowlist: OutboundAllowlist,
    ) -> None:
        """Initialize the sink, validating the configured host up front.

        Args:
            config: The ntfy server and topic settings.
            transport: The HTTP transport to use. Defaults to
                :func:`_https_post`.
            allowlist: The required outbound-network allowlist the configured
                ``base_url`` host (and its ``https`` scheme) must satisfy. The
                allowlist is a constructor argument rather than something the
                sink derives from its own config, so the permitted host set
                stays an independent operator declaration the destination has
                to clear (see :mod:`windbreak.net.allowlist`).

        Raises:
            EgressDeniedError: If ``config.base_url``'s host is off the
                allowlist or its scheme is not http(s) -- rejected at
                construction, before any network call.
        """
        allowlist.require(config.base_url)
        self._config = config
        self._transport = transport

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """POST the alert message to the configured ntfy topic.

        Args:
            alert_type: The kind of alert being delivered.
            severity: The alert's severity.
            message: The human-readable alert body.

        Raises:
            SinkSendError: If the HTTP POST fails.
        """
        url = f"{self._config.base_url}/{self._config.topic}"
        headers = {
            "X-Alert-Type": alert_type.value,
            "X-Alert-Severity": severity.value,
            "Title": f"windbreak {alert_type.value}",
            "Priority": _NTFY_PRIORITY[severity],
        }
        _send_http(self._transport, url, message.encode("utf-8"), headers)


class WebhookSink:
    """POST a JSON alert payload to a generic webhook over HTTPS."""

    name = "webhook"

    def __init__(
        self,
        config: WebhookSinkConfig,
        *,
        transport: HttpTransport = _https_post,
        allowlist: OutboundAllowlist,
    ) -> None:
        """Initialize the sink, validating the configured endpoint up front.

        Args:
            config: The webhook endpoint settings.
            transport: The HTTP transport to use. Defaults to
                :func:`_https_post`.
            allowlist: The required outbound-network allowlist ``config.url``
                must satisfy. Required, with no default, because a webhook URL
                is operator-supplied config: without this screen a mistyped or
                tampered ``url`` would let alert delivery reach an internal host
                (SSRF). Kept identical to :class:`NtfySink`'s contract so no
                sibling sink is the weak one.

        Raises:
            EgressDeniedError: If ``config.url``'s host is off the allowlist or
                its scheme is not http(s) -- rejected at construction, before
                any network call.
        """
        allowlist.require(config.url)
        self._config = config
        self._transport = transport

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """POST a JSON ``{type, severity, message}`` payload to the webhook.

        Args:
            alert_type: The kind of alert being delivered.
            severity: The alert's severity.
            message: The human-readable alert body.

        Raises:
            SinkSendError: If the HTTP POST fails.
        """
        body = json.dumps(
            {
                "type": alert_type.value,
                "severity": severity.value,
                "message": message,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        _send_http(self._transport, self._config.url, body, headers)


class SmtpSink:
    """Deliver alerts as email via SMTP with STARTTLS."""

    name = "smtp"

    def __init__(
        self,
        config: SmtpSinkConfig,
        *,
        transport: SmtpTransport = _smtp_send,
        allowlist: OutboundAllowlist,
    ) -> None:
        """Initialize the sink, validating the configured relay host up front.

        Args:
            config: The SMTP connection and addressing settings.
            transport: The SMTP transport to use. Defaults to
                :func:`_smtp_send`.
            allowlist: The required outbound-network allowlist ``config.host``
                must satisfy. SMTP has no URL to screen, so the bare host goes
                through :meth:`~windbreak.net.allowlist.OutboundAllowlist.require_host`;
                without it SMTP would be the one outbound path with no egress
                check at all.

        Raises:
            EgressDeniedError: If ``config.host`` is off the allowlist --
                rejected at construction, before any connection is opened.
        """
        allowlist.require_host(config.host)
        self._config = config
        self._transport = transport

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Build an alert email and hand it to the SMTP transport.

        Args:
            alert_type: The kind of alert being delivered.
            severity: The alert's severity.
            message: The human-readable alert body.

        Raises:
            SinkSendError: If the SMTP transport fails.
        """
        email = self._build_message(alert_type, severity, message)
        try:
            self._transport(self._config, email)
        except Exception as exc:
            raise SinkSendError(
                str(exc), outcome=classify_transport_failure(exc)
            ) from exc

    def _build_message(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> EmailMessage:
        """Build the alert email addressed from and to the configured parties.

        Args:
            alert_type: The kind of alert being delivered.
            severity: The alert's severity.
            message: The human-readable alert body.

        Returns:
            A fully addressed :class:`~email.message.EmailMessage`.
        """
        email = EmailMessage()
        email["From"] = self._config.sender
        email["To"] = ", ".join(self._config.recipients)
        email["Subject"] = f"[windbreak] {alert_type.value} ({severity.value})"
        email.set_content(message)
        return email


class DesktopSink:
    """Raise a local desktop notification through an injected notifier."""

    name = "desktop"

    def __init__(self, notifier: Callable[[str, str], None] | None = None) -> None:
        """Initialize the sink.

        Args:
            notifier: A ``(title, body) -> None`` callable that raises the
                notification. When None, the sink cannot deliver.
        """
        self._notifier = notifier

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Raise a desktop notification for the alert.

        Args:
            alert_type: The kind of alert being delivered.
            severity: The alert's severity.
            message: The human-readable alert body.

        Raises:
            SinkSendError: If no notifier was configured.
        """
        if self._notifier is None:
            raise SinkSendError("no desktop notifier configured")
        title = f"windbreak {alert_type.value}"
        body = f"[{severity.value}] {message}"
        self._notifier(title, body)


class LogOnlySink:
    """The always-available fallback sink that only logs the alert."""

    name = "log-only"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the sink.

        Args:
            logger: The logger to emit on. Defaults to ``windbreak.alerts``.
        """
        self._logger = logger or logging.getLogger("windbreak.alerts")

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Log the alert at the severity-mapped level with its fields.

        Args:
            alert_type: The kind of alert being delivered.
            severity: The alert's severity.
            message: The human-readable alert body.
        """
        self._logger.log(
            severity.to_log_level(),
            message,
            extra={
                "component": "alerts",
                "alert_type": alert_type.value,
                "severity": severity.value,
            },
        )
