"""End-to-end alert delivery: configuration -> concrete sink (issue #274).

Every test here drives the *real* composition path -- a
:class:`~windbreak.config.schema.WindbreakConfig`, the outbound allowlist
derived from it, :func:`~windbreak.alerts.factory.build_sinks`, and a real
:class:`~windbreak.alerts.dispatch.AlertDispatcher` -- and asserts the alert
lands in an injected fake transport rather than in the ``log-only`` fallback.
Zero sockets and zero SMTP connections are opened: only the transport seam is
faked, so the config->sink->dispatch wiring itself is genuinely exercised.

RED before the implementation: ``windbreak.alerts.factory`` does not exist, so
this module fails at collection with ``ModuleNotFoundError``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING

import pytest

from windbreak.alerts.dispatch import AlertDispatcher, LoggingLedgerWriter
from windbreak.alerts.factory import AlertSinkConfigError, build_sinks
from windbreak.alerts.registry import AlertType
from windbreak.config.schema import (
    UNCONFIGURED_PLACEHOLDER,
    AlertsConfig,
    AlertSink,
    SmtpSinkSettings,
    WindbreakConfig,
)
from windbreak.net.allowlist import allowlist_from_config

if TYPE_CHECKING:
    from collections.abc import Mapping
    from email.message import EmailMessage

    from windbreak.alerts.sinks import SmtpSinkConfig

#: The ntfy host every configured-ntfy test publishes to.
_NTFY_HOST = "ntfy.example.com"

#: The webhook host every configured-webhook test posts to.
_WEBHOOK_HOST = "hooks.example.com"

#: A webhook path carrying an operator secret, used to prove no error message
#: or log line ever echoes anything past the host.
_WEBHOOK_SECRET_PATH = "/services/s3cr3t-token"

#: The SMTP relay host every configured-smtp test hands to its transport.
_SMTP_HOST = "smtp.example.com"

#: The operator secret embedded in :data:`_HOSTLESS_WEBHOOK_URL`'s query string.
_HOSTLESS_URL_TOKEN = "abc123"

#: A plausible operator typo: one slash too many leaves the netloc empty, so
#: `urlsplit(...).hostname` is `None` even though the URL still carries a path
#: and a token. Used to prove a denial with no parseable host redacts the whole
#: destination rather than falling back to echoing it.
_HOSTLESS_WEBHOOK_URL = f"https:///{_WEBHOOK_HOST}/incoming?token={_HOSTLESS_URL_TOKEN}"


class _RecordingHttpTransport:
    """An :data:`~windbreak.alerts.sinks.HttpTransport` double.

    Records each ``(url, body, headers)`` triple and returns a canned status,
    so a sink's delivery is observable without a socket.
    """

    def __init__(self, status: int = 200) -> None:
        """Initialize the transport.

        Args:
            status: The HTTP status code every call returns.
        """
        self.status = status
        self.calls: list[tuple[str, bytes, Mapping[str, str]]] = []

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str]) -> int:
        """Record one POST and return the canned status.

        Args:
            url: The target URL.
            body: The request body.
            headers: The request headers.

        Returns:
            The canned status code.
        """
        self.calls.append((url, body, headers))
        return self.status


class _RecordingSmtpTransport:
    """A :data:`~windbreak.alerts.sinks.SmtpTransport` double recording sends."""

    def __init__(self) -> None:
        """Initialize the transport with an empty call log."""
        self.calls: list[tuple[SmtpSinkConfig, EmailMessage]] = []

    def __call__(self, config: SmtpSinkConfig, message: EmailMessage) -> None:
        """Record one message instead of speaking SMTP.

        Args:
            config: The SMTP connection settings the sink was built with.
            message: The message the sink built.
        """
        self.calls.append((config, message))


def _config_with(alerts: AlertsConfig) -> WindbreakConfig:
    """Return the default configuration with its alerts section replaced.

    Args:
        alerts: The alerts section to substitute.

    Returns:
        A `WindbreakConfig` identical to the defaults but for `alerts`.
    """
    return dataclasses.replace(WindbreakConfig(), alerts=alerts)


def _ntfy_config() -> WindbreakConfig:
    """Return a configuration carrying one fully-configured ntfy sink."""
    return _config_with(
        AlertsConfig(
            sinks=(
                AlertSink(
                    type="ntfy", topic="windbreak-ops", base_url=f"https://{_NTFY_HOST}"
                ),
            ),
            allowed_hosts=(_NTFY_HOST,),
        )
    )


# --- 1. The configured sink actually receives the alert -----------------------


def test_configured_ntfy_sink_receives_dispatched_alert() -> None:
    """A configured ntfy sink -- not the fallback -- delivers the alert.

    Drives the whole seam the issue says is missing: `AlertsConfig` ->
    `build_sinks` -> `AlertDispatcher.dispatch` -> the injected transport.
    """
    config = _ntfy_config()
    transport = _RecordingHttpTransport()

    sinks = build_sinks(
        config.alerts,
        allowlist=allowlist_from_config(config),
        http_transport=transport,
    )
    dispatcher = AlertDispatcher(sinks, ledger_writer=LoggingLedgerWriter())
    event = dispatcher.dispatch(AlertType.MODE_CHANGE, "switched to PAPER")

    assert [outcome.sink for outcome in event.outcomes] == ["ntfy"]
    assert event.outcomes[0].ok is True
    assert len(transport.calls) == 1
    url, body, headers = transport.calls[0]
    assert url == f"https://{_NTFY_HOST}/windbreak-ops"
    assert body == b"switched to PAPER"
    assert headers["X-Alert-Type"] == "mode change"


def test_configured_sink_suppresses_the_log_only_fallback() -> None:
    """`LogOnlySink` never fires while a configured sink succeeds."""
    config = _ntfy_config()

    sinks = build_sinks(
        config.alerts,
        allowlist=allowlist_from_config(config),
        http_transport=_RecordingHttpTransport(),
    )
    event = AlertDispatcher(sinks, ledger_writer=LoggingLedgerWriter()).dispatch(
        AlertType.VETO, "vetoed"
    )

    assert "log-only" not in [outcome.sink for outcome in event.outcomes]


def test_log_only_fallback_fires_when_no_sink_is_configured() -> None:
    """The shipped placeholder config builds no sink, so the fallback fires."""
    config = WindbreakConfig()

    sinks = build_sinks(config.alerts, allowlist=allowlist_from_config(config))
    event = AlertDispatcher(sinks, ledger_writer=LoggingLedgerWriter()).dispatch(
        AlertType.VETO, "vetoed"
    )

    assert sinks == ()
    assert [outcome.sink for outcome in event.outcomes] == ["log-only"]


def test_unconfigured_sink_is_skipped_with_a_warning_naming_only_its_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A placeholder sink is skipped loudly, and the warning leaks no topic.

    Absent evidence must not read as healthy evidence: skipping is visible in
    the log. The topic is an ntfy capability token, so the warning names the
    sink *type* only.
    """
    config = _config_with(
        AlertsConfig(sinks=(AlertSink(type="ntfy", topic="s3cr3t-topic"),))
    )

    with caplog.at_level(logging.WARNING, logger="windbreak.alerts"):
        sinks = build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    assert sinks == ()
    assert any(
        record.levelno == logging.WARNING and "ntfy" in record.getMessage()
        for record in caplog.records
    )
    assert "s3cr3t-topic" not in caplog.text


# --- 2. Webhook: same allowlist as ntfy, and its rejection path ---------------


def test_configured_webhook_sink_delivers_a_json_payload() -> None:
    """An allowlisted webhook receives the JSON `{type, severity, message}`."""
    config = _config_with(
        AlertsConfig(
            sinks=(
                AlertSink(
                    type="webhook",
                    url=f"https://{_WEBHOOK_HOST}{_WEBHOOK_SECRET_PATH}",
                ),
            ),
            allowed_hosts=(_WEBHOOK_HOST,),
        )
    )
    transport = _RecordingHttpTransport()

    sinks = build_sinks(
        config.alerts,
        allowlist=allowlist_from_config(config),
        http_transport=transport,
    )
    AlertDispatcher(sinks, ledger_writer=LoggingLedgerWriter()).dispatch(
        AlertType.DISK_HALT, "disk full"
    )

    assert len(transport.calls) == 1
    url, body, _ = transport.calls[0]
    assert url == f"https://{_WEBHOOK_HOST}{_WEBHOOK_SECRET_PATH}"
    assert json.loads(body) == {
        "type": "disk halt",
        "severity": "critical",
        "message": "disk full",
    }


def test_webhook_url_off_the_allowlist_is_rejected_at_build_time() -> None:
    """A non-allowlisted webhook URL fails closed before any dispatch.

    This is the SSRF guard the issue asks for: `WebhookSink` must enforce the
    same `OutboundAllowlist` as `NtfySink`, and enforce it at construction so
    an internal host is never dialed even once.
    """
    config = _config_with(
        AlertsConfig(
            sinks=(AlertSink(type="webhook", url="https://169.254.169.254/latest"),),
            allowed_hosts=(_WEBHOOK_HOST,),
        )
    )

    with pytest.raises(AlertSinkConfigError) as exc_info:
        build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    assert "169.254.169.254" in str(exc_info.value)


def test_webhook_rejection_message_never_echoes_the_url_past_its_host() -> None:
    """A denial names the host but never the secret-bearing path or query."""
    config = _config_with(
        AlertsConfig(
            sinks=(
                AlertSink(
                    type="webhook",
                    url=f"https://internal.example.org{_WEBHOOK_SECRET_PATH}",
                ),
            ),
        )
    )

    with pytest.raises(AlertSinkConfigError) as exc_info:
        build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    assert "s3cr3t-token" not in str(exc_info.value)
    assert "internal.example.org" in str(exc_info.value)


def test_webhook_rejection_message_never_echoes_an_unparseable_url() -> None:
    """A denied webhook URL with no parseable host still leaks nothing.

    The triple-slash typo (`https:///host/...`) leaves `urlsplit` with an empty
    netloc, so there is no hostname to name. The message must fall back to
    saying *which sink and field* to fix rather than echoing the destination:
    the raw URL carries the token, and `main.py` logs this message verbatim via
    `_LOGGER.critical("FATAL: %s", exc)` onto stderr.
    """
    config = _config_with(
        AlertsConfig(
            sinks=(AlertSink(type="webhook", url=_HOSTLESS_WEBHOOK_URL),),
            allowed_hosts=(_WEBHOOK_HOST,),
        )
    )

    with pytest.raises(AlertSinkConfigError) as exc_info:
        build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    message = str(exc_info.value)
    assert _HOSTLESS_URL_TOKEN not in message
    assert _HOSTLESS_WEBHOOK_URL not in message
    assert "/incoming" not in message
    assert "webhook" in message
    assert "url" in message


def test_ntfy_rejection_message_never_echoes_an_unparseable_base_url() -> None:
    """The same hostless-destination redaction applies to an ntfy `base_url`.

    `base_url` and `topic` are both bearer capabilities, so the ntfy branch must
    name its field without echoing the value, exactly as the webhook branch does.
    """
    config = _config_with(
        AlertsConfig(
            sinks=(
                AlertSink(
                    type="ntfy",
                    base_url=_HOSTLESS_WEBHOOK_URL,
                    topic="secret-topic",
                ),
            ),
            allowed_hosts=(_NTFY_HOST,),
        )
    )

    with pytest.raises(AlertSinkConfigError) as exc_info:
        build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    message = str(exc_info.value)
    assert _HOSTLESS_URL_TOKEN not in message
    assert _HOSTLESS_WEBHOOK_URL not in message
    assert "ntfy" in message
    assert "base_url" in message


def test_smtp_rejection_message_redacts_a_host_that_is_not_a_bare_host() -> None:
    """A URL mistyped into `smtp.host` is redacted, not echoed.

    `smtp.host` is a bare hostname by contract, and a bare host is safe to name
    in full -- but only *because* it is bare. An operator who pastes a full URL
    into the field would otherwise have its query string echoed into a FATAL
    log line, since a bare-host denial has no URL parsing to strip it.
    """
    config = _config_with(
        AlertsConfig(
            sinks=(
                AlertSink(
                    type="smtp",
                    smtp=SmtpSinkSettings(
                        host=_HOSTLESS_WEBHOOK_URL,
                        sender="alerts@example.com",
                        recipients=("ops@example.com",),
                    ),
                ),
            ),
            allowed_hosts=(_SMTP_HOST,),
        )
    )

    with pytest.raises(AlertSinkConfigError) as exc_info:
        build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    message = str(exc_info.value)
    assert _HOSTLESS_URL_TOKEN not in message
    assert _HOSTLESS_WEBHOOK_URL not in message
    assert "smtp.host" in message


def test_smtp_rejection_message_still_names_a_genuine_bare_host() -> None:
    """Redacting the malformed case must not blind the ordinary one.

    An off-allowlist bare relay host carries no secret, and naming it is the
    whole remediation: the operator needs to know which host to declare.
    """
    config = _config_with(
        AlertsConfig(
            sinks=(
                AlertSink(
                    type="smtp",
                    smtp=SmtpSinkSettings(
                        host="relay.internal.example",
                        sender="alerts@example.com",
                        recipients=("ops@example.com",),
                    ),
                ),
            ),
            allowed_hosts=(_SMTP_HOST,),
        )
    )

    with pytest.raises(AlertSinkConfigError) as exc_info:
        build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    message = str(exc_info.value)
    assert "relay.internal.example" in message
    assert "alerts.allowed_hosts" in message


# --- 3. SMTP and desktop -----------------------------------------------------


def test_configured_smtp_sink_delivers_an_email() -> None:
    """An allowlisted SMTP relay receives the built alert email."""
    config = _config_with(
        AlertsConfig(
            sinks=(
                AlertSink(
                    type="smtp",
                    smtp=SmtpSinkSettings(
                        host=_SMTP_HOST,
                        port=587,
                        sender="windbreak@example.com",
                        recipients=("ops@example.com",),
                    ),
                ),
            ),
            allowed_hosts=(_SMTP_HOST,),
        )
    )
    transport = _RecordingSmtpTransport()

    sinks = build_sinks(
        config.alerts,
        allowlist=allowlist_from_config(config),
        smtp_transport=transport,
    )
    AlertDispatcher(sinks, ledger_writer=LoggingLedgerWriter()).dispatch(
        AlertType.BACKUP_FAILURE, "backup failed"
    )

    assert len(transport.calls) == 1
    sent_config, message = transport.calls[0]
    assert sent_config.host == _SMTP_HOST
    assert message["To"] == "ops@example.com"


def test_smtp_host_off_the_allowlist_is_rejected_at_build_time() -> None:
    """An SMTP relay host must clear the same declared-host allowlist."""
    config = _config_with(
        AlertsConfig(
            sinks=(
                AlertSink(
                    type="smtp",
                    smtp=SmtpSinkSettings(
                        host="relay.internal",
                        sender="windbreak@example.com",
                        recipients=("ops@example.com",),
                    ),
                ),
            ),
            allowed_hosts=(_SMTP_HOST,),
        )
    )

    with pytest.raises(AlertSinkConfigError) as exc_info:
        build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    assert "relay.internal" in str(exc_info.value)


def test_configured_desktop_sink_delivers_through_the_injected_notifier() -> None:
    """A desktop sink is built only when a real notifier is supplied."""
    notified: list[tuple[str, str]] = []
    config = _config_with(AlertsConfig(sinks=(AlertSink(type="desktop"),)))

    sinks = build_sinks(
        config.alerts,
        allowlist=allowlist_from_config(config),
        desktop_notifier=lambda title, body: notified.append((title, body)),
    )
    AlertDispatcher(sinks, ledger_writer=LoggingLedgerWriter()).dispatch(
        AlertType.VETO, "vetoed"
    )

    assert notified == [("windbreak veto", "[warning] vetoed")]


def test_desktop_sink_without_a_notifier_is_rejected_at_build_time() -> None:
    """A desktop sink with no notifier can never deliver, so it fails closed.

    Building an undeliverable sink would silently degrade every alert to the
    fallback at dispatch time; rejecting at composition makes the gap visible
    at startup instead.
    """
    config = _config_with(AlertsConfig(sinks=(AlertSink(type="desktop"),)))

    with pytest.raises(AlertSinkConfigError) as exc_info:
        build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    assert "desktop" in str(exc_info.value)


# --- 4. Fail-closed on anything unrecognized ---------------------------------


def test_unknown_sink_type_is_rejected() -> None:
    """An unrecognized sink type is fatal, never silently dropped."""
    config = _config_with(AlertsConfig(sinks=(AlertSink(type="carrier-pigeon"),)))

    with pytest.raises(AlertSinkConfigError) as exc_info:
        build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    assert "carrier-pigeon" in str(exc_info.value)


def test_ntfy_host_off_the_allowlist_is_rejected_at_build_time() -> None:
    """A configured ntfy base URL still has to clear the allowlist."""
    config = _config_with(
        AlertsConfig(
            sinks=(
                AlertSink(
                    type="ntfy", topic="ops", base_url="https://evil.example.net"
                ),
            ),
            allowed_hosts=(_NTFY_HOST,),
        )
    )

    with pytest.raises(AlertSinkConfigError) as exc_info:
        build_sinks(config.alerts, allowlist=allowlist_from_config(config))

    assert "evil.example.net" in str(exc_info.value)


def test_every_configured_sink_is_built_in_order() -> None:
    """Multiple configured sinks all reach the dispatcher, in config order."""
    config = _config_with(
        AlertsConfig(
            sinks=(
                AlertSink(type="ntfy", topic="ops", base_url=f"https://{_NTFY_HOST}"),
                AlertSink(type="webhook", url=f"https://{_WEBHOOK_HOST}/hook"),
            ),
            allowed_hosts=(_NTFY_HOST, _WEBHOOK_HOST),
        )
    )

    sinks = build_sinks(
        config.alerts,
        allowlist=allowlist_from_config(config),
        http_transport=_RecordingHttpTransport(),
    )

    assert [sink.name for sink in sinks] == ["ntfy", "webhook"]


# --- 5. The allowlist itself now derives the operator-declared alert hosts ----


def test_allowlist_from_config_admits_the_declared_alert_hosts() -> None:
    """`alerts.allowed_hosts` entries join the deployment egress allowlist."""
    config = _config_with(AlertsConfig(allowed_hosts=(_NTFY_HOST.upper(),)))

    allowlist_from_config(config).require(f"https://{_NTFY_HOST}/topic")


def test_allowlist_from_config_admits_no_alert_host_by_default() -> None:
    """An operator who declares no alert host gets no alert egress at all."""
    from windbreak.net.allowlist import EgressDeniedError

    with pytest.raises(EgressDeniedError):
        allowlist_from_config(WindbreakConfig()).require(f"https://{_NTFY_HOST}/t")


def test_placeholder_constant_is_the_shipped_alert_sink_default() -> None:
    """The skip rule keys off the same placeholder the schema ships."""
    assert WindbreakConfig().alerts.sinks[0].base_url == UNCONFIGURED_PLACEHOLDER
