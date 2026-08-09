"""Build the concrete alert sinks a configuration asks for (issue #274).

This is the composition seam between :mod:`windbreak.config` and
:mod:`windbreak.alerts.sinks`: :func:`build_sinks` turns each
:class:`~windbreak.config.schema.AlertSink` entry into a live sink instance that
:class:`~windbreak.alerts.dispatch.AlertDispatcher` fans alerts out to. Before
it existed every production dispatcher was built with ``sinks=[]``, so operator
alerts never left the box.

Three rules govern the mapping, and all three are about not lying to the
operator:

1. **Unconfigured is skipped, loudly.** A sink entry still holding
   :data:`~windbreak.config.schema.UNCONFIGURED_PLACEHOLDER` in a field it needs
   cannot deliver anything, so it builds nothing and logs a WARNING naming its
   type. The alert still surfaces: with no sink built, the dispatcher's
   ``log-only`` fallback fires. This mirrors the fail-closed placeholder idiom
   :class:`~windbreak.config.schema.ResearchSettings` already uses, and it is
   why the shipped SPEC S16 default (one placeholder ntfy sink) does not make
   every process refuse to start.
2. **Misconfigured is fatal.** An unknown sink type, an off-allowlist
   destination, or a sink that cannot possibly deliver (a ``desktop`` entry with
   no notifier) raises :class:`AlertSinkConfigError` at composition. These are
   operator mistakes, not absent configuration, and degrading them to log-only
   would hide a broken alerting path behind a healthy-looking process.
3. **Destinations are secrets.** An ntfy topic is a bearer capability and a
   webhook URL can embed a token, so no message this module raises or logs ever
   contains more of a destination than its hostname.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from windbreak.alerts.sinks import (
    DEFAULT_HTTP_TRANSPORT,
    DEFAULT_SMTP_TRANSPORT,
    DesktopSink,
    NtfySink,
    NtfySinkConfig,
    SmtpSink,
    SmtpSinkConfig,
    WebhookSink,
    WebhookSinkConfig,
)
from windbreak.config.schema import UNCONFIGURED_PLACEHOLDER
from windbreak.net.allowlist import EgressDeniedError

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.alerts.sinks import (
        AlertSink,
        HttpTransport,
        SmtpTransport,
    )
    from windbreak.config.schema import AlertsConfig
    from windbreak.config.schema import AlertSink as AlertSinkSpec
    from windbreak.net.allowlist import OutboundAllowlist

_LOGGER = logging.getLogger("windbreak.alerts")

#: The sink ``type`` token for each supported channel, so a typo in a config
#: file is compared against one authoritative spelling.
_NTFY: Final = "ntfy"
_WEBHOOK: Final = "webhook"
_SMTP: Final = "smtp"
_DESKTOP: Final = "desktop"

#: Every recognized ``type``, named in the error an unknown type raises so the
#: operator is told what the valid choices are.
_KNOWN_TYPES: Final = (_NTFY, _WEBHOOK, _SMTP, _DESKTOP)


class AlertSinkConfigError(Exception):
    """Raised when an alert sink is configured in a way that cannot work.

    Distinct from a skipped *unconfigured* sink: this signals an operator
    mistake (an unknown type, an undeliverable sink, a destination off the
    egress allowlist) that must stop the process rather than silently degrade
    alerting to the log-only fallback.
    """


def _is_configured(value: str) -> bool:
    """Return whether an operator has actually filled a destination field in.

    Args:
        value: The raw field value from the configuration.

    Returns:
        ``True`` unless the value is empty or still the shipped
        :data:`~windbreak.config.schema.UNCONFIGURED_PLACEHOLDER`.
    """
    return bool(value) and value != UNCONFIGURED_PLACEHOLDER


def _skip_unconfigured(sink_type: str, missing: str) -> None:
    """Log that an unconfigured sink was skipped, naming no destination.

    Args:
        sink_type: The sink's configured ``type``.
        missing: The name of the field the operator has not filled in. A field
            *name* only -- never its value, which may be a topic or token.
    """
    _LOGGER.warning(
        "alert sink %r is unconfigured (%s not set) and was skipped; "
        "alerts fall back to log-only",
        sink_type,
        missing,
        extra={"component": "alerts", "sink": sink_type},
    )


def _denied(sink_type: str, destination: str) -> AlertSinkConfigError:
    """Translate an egress denial into an error that leaks no destination.

    :class:`~windbreak.net.allowlist.EgressDeniedError`'s own message quotes the
    whole URL, which for a webhook may carry a token in its path or query. Every
    caller raises this replacement with ``from None`` so that URL-bearing
    message cannot resurface in a traceback either.

    Args:
        sink_type: The sink's configured ``type``.
        destination: The denied URL or bare host.

    Returns:
        The :class:`AlertSinkConfigError` to raise in the denial's place.
    """
    host = urlsplit(destination).hostname or destination
    return AlertSinkConfigError(
        f"alert sink {sink_type!r} destination host {host!r} is not on the "
        "outbound egress allowlist; declare it in alerts.allowed_hosts"
    )


def _build_ntfy(
    spec: AlertSinkSpec, allowlist: OutboundAllowlist, transport: HttpTransport
) -> AlertSink | None:
    """Build an ntfy sink, or ``None`` when the entry is unconfigured.

    Args:
        spec: The configuration entry to build from.
        allowlist: The egress allowlist ``base_url``'s host must clear.
        transport: The HTTP transport the sink delivers through.

    Returns:
        The sink, or ``None`` if ``base_url`` or ``topic`` is still a
        placeholder.

    Raises:
        AlertSinkConfigError: If ``base_url``'s host is off the allowlist.
    """
    if not _is_configured(spec.base_url):
        _skip_unconfigured(spec.type, "base_url")
        return None
    if not _is_configured(spec.topic):
        _skip_unconfigured(spec.type, "topic")
        return None
    config = NtfySinkConfig(base_url=spec.base_url, topic=spec.topic)
    try:
        return NtfySink(config, transport=transport, allowlist=allowlist)
    except EgressDeniedError:
        raise _denied(spec.type, spec.base_url) from None


def _build_webhook(
    spec: AlertSinkSpec, allowlist: OutboundAllowlist, transport: HttpTransport
) -> AlertSink | None:
    """Build a webhook sink, or ``None`` when the entry is unconfigured.

    Args:
        spec: The configuration entry to build from.
        allowlist: The egress allowlist ``url``'s host must clear.
        transport: The HTTP transport the sink delivers through.

    Returns:
        The sink, or ``None`` if ``url`` is still a placeholder.

    Raises:
        AlertSinkConfigError: If ``url``'s host is off the allowlist.
    """
    if not _is_configured(spec.url):
        _skip_unconfigured(spec.type, "url")
        return None
    try:
        return WebhookSink(
            WebhookSinkConfig(url=spec.url), transport=transport, allowlist=allowlist
        )
    except EgressDeniedError:
        raise _denied(spec.type, spec.url) from None


def _build_smtp(
    spec: AlertSinkSpec, allowlist: OutboundAllowlist, transport: SmtpTransport
) -> AlertSink | None:
    """Build an SMTP sink, or ``None`` when the entry is unconfigured.

    Args:
        spec: The configuration entry to build from.
        allowlist: The egress allowlist the relay ``host`` must clear.
        transport: The SMTP transport the sink delivers through.

    Returns:
        The sink, or ``None`` if the relay host, sender, or recipient list is
        unset.

    Raises:
        AlertSinkConfigError: If the relay host is off the allowlist.
    """
    smtp = spec.smtp
    if not _is_configured(smtp.host):
        _skip_unconfigured(spec.type, "smtp.host")
        return None
    if not _is_configured(smtp.sender):
        _skip_unconfigured(spec.type, "smtp.sender")
        return None
    if not smtp.recipients:
        _skip_unconfigured(spec.type, "smtp.recipients")
        return None
    config = SmtpSinkConfig(
        host=smtp.host,
        port=smtp.port,
        sender=smtp.sender,
        recipients=smtp.recipients,
    )
    try:
        return SmtpSink(config, transport=transport, allowlist=allowlist)
    except EgressDeniedError:
        raise _denied(spec.type, smtp.host) from None


def _build_desktop(
    spec: AlertSinkSpec, notifier: Callable[[str, str], None] | None
) -> AlertSink:
    """Build a desktop sink, requiring a notifier that can actually deliver.

    Args:
        spec: The configuration entry to build from.
        notifier: The ``(title, body) -> None`` callable that raises the
            notification.

    Returns:
        The desktop sink.

    Raises:
        AlertSinkConfigError: If no notifier was supplied. A notifier-less
            :class:`~windbreak.alerts.sinks.DesktopSink` raises on every send,
            which would quietly downgrade this process's alerting to the
            fallback; refusing at composition surfaces the gap at startup.
    """
    if notifier is None:
        raise AlertSinkConfigError(
            f"alert sink {spec.type!r} is configured but this process supplies "
            "no desktop notifier, so it could never deliver an alert"
        )
    return DesktopSink(notifier)


def _build_one(
    spec: AlertSinkSpec,
    *,
    allowlist: OutboundAllowlist,
    http_transport: HttpTransport,
    smtp_transport: SmtpTransport,
    desktop_notifier: Callable[[str, str], None] | None,
) -> AlertSink | None:
    """Dispatch one configuration entry to its type's builder.

    Args:
        spec: The configuration entry to build from.
        allowlist: The egress allowlist every network destination must clear.
        http_transport: The HTTP transport ntfy/webhook sinks deliver through.
        smtp_transport: The SMTP transport an smtp sink delivers through.
        desktop_notifier: The notifier a desktop sink delivers through.

    Returns:
        The built sink, or ``None`` when the entry is unconfigured.

    Raises:
        AlertSinkConfigError: If ``spec.type`` is unrecognized, or the type's
            builder rejects the entry.
    """
    if spec.type == _NTFY:
        return _build_ntfy(spec, allowlist, http_transport)
    if spec.type == _WEBHOOK:
        return _build_webhook(spec, allowlist, http_transport)
    if spec.type == _SMTP:
        return _build_smtp(spec, allowlist, smtp_transport)
    if spec.type == _DESKTOP:
        return _build_desktop(spec, desktop_notifier)
    raise AlertSinkConfigError(
        f"unknown alert sink type {spec.type!r}; expected one of "
        f"{', '.join(_KNOWN_TYPES)}"
    )


def build_sinks(
    alerts: AlertsConfig,
    *,
    allowlist: OutboundAllowlist,
    http_transport: HttpTransport = DEFAULT_HTTP_TRANSPORT,
    smtp_transport: SmtpTransport = DEFAULT_SMTP_TRANSPORT,
    desktop_notifier: Callable[[str, str], None] | None = None,
) -> tuple[AlertSink, ...]:
    """Build every deliverable sink the alerts configuration declares.

    Args:
        alerts: The configuration's alerts section.
        allowlist: The deployment egress allowlist every network destination is
            screened against, normally
            :func:`windbreak.net.allowlist.allowlist_from_config`. Required, so
            no caller can compose an unscreened alert path.
        http_transport: The HTTP transport ntfy and webhook sinks deliver
            through. Injectable so tests exercise the real wiring without a
            socket.
        smtp_transport: The SMTP transport an smtp sink delivers through.
        desktop_notifier: The ``(title, body) -> None`` callable a desktop sink
            delivers through. ``None`` (the default for every headless process)
            makes a configured desktop sink an error rather than a silent no-op.

    Returns:
        One sink per deliverable entry, in configuration order. Empty when
        nothing is configured, which is the only case in which the dispatcher's
        ``log-only`` fallback should ever fire.

    Raises:
        AlertSinkConfigError: If any entry names an unknown type, targets a host
            off the egress allowlist, or cannot deliver as configured.
    """
    built = (
        _build_one(
            spec,
            allowlist=allowlist,
            http_transport=http_transport,
            smtp_transport=smtp_transport,
            desktop_notifier=desktop_notifier,
        )
        for spec in alerts.sinks
    )
    return tuple(sink for sink in built if sink is not None)
