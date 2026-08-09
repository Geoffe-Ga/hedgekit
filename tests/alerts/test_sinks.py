"""Tests for windbreak.alerts.sinks (issue #14): outbound alert channels.

Every sink is exercised through an injected transport double -- zero real
sockets, zero real SMTP connections. The default transports (`_https_post`,
`_smtp_send`) are covered separately by swapping `http.client.HTTPSConnection`
/ `smtplib.SMTP` for fakes, so the wiring between "sink" and "network call"
is pinned without ever touching the network.

None of `windbreak.alerts.sinks`'s public names exist yet, so importing this
module fails at collection with `ModuleNotFoundError` -- the expected RED
state for issue #14's Gate 1.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import logging
import smtplib
from email.message import EmailMessage
from typing import TYPE_CHECKING, ClassVar

import pytest

from windbreak.alerts.registry import AlertSeverity, AlertType
from windbreak.alerts.sinks import (
    _TRANSPORT_TIMEOUT_SECONDS,
    DesktopSink,
    LogOnlySink,
    NtfySink,
    NtfySinkConfig,
    SinkSendError,
    SmtpSink,
    SmtpSinkConfig,
    WebhookSink,
    WebhookSinkConfig,
    _https_post,
    _smtp_send,
)
from windbreak.net.allowlist import EgressDeniedError, OutboundAllowlist

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Allowlist admitting the ntfy host every ``TestNtfySink`` construction uses,
#: now that ``NtfySink`` requires an explicit outbound allowlist (issue #57).
_NTFY_ALLOWLIST = OutboundAllowlist(frozenset({"ntfy.example.com"}))

#: Allowlist admitting the webhook host every ``TestWebhookSink`` construction
#: uses. ``WebhookSink`` requires the same explicit allowlist ``NtfySink`` does
#: (issue #274), so a config-supplied URL can never reach an internal host.
_WEBHOOK_ALLOWLIST = OutboundAllowlist(frozenset({"hooks.example.com"}))

#: Allowlist admitting the SMTP relay host every ``TestSmtpSink`` construction
#: uses; ``SmtpSink`` screens its bare host through ``require_host`` (issue
#: #274) so SMTP is not the one unscreened outbound path.
_SMTP_ALLOWLIST = OutboundAllowlist(frozenset({"smtp.example.com"}))


class _FakeHTTPSResponse:
    """Minimal stand-in for `http.client.HTTPResponse`."""

    def __init__(self, status: int) -> None:
        self.status = status

    def read(self) -> bytes:
        """Return an empty response body."""
        return b""


class _FakeHTTPSConnection:
    """Records the request made through it; returns a canned status.

    Stands in for `http.client.HTTPSConnection` so `_https_post` can be
    exercised end-to-end without opening a real TLS socket.
    """

    instances: ClassVar[list[_FakeHTTPSConnection]] = []
    default_status: ClassVar[int] = 200

    def __init__(
        self, host: str, *, timeout: float | None = None, context: object = None
    ) -> None:
        self.host = host
        self.timeout = timeout
        self.context = context
        self.requests: list[tuple[str, str, bytes, Mapping[str, str]]] = []
        self.status = type(self).default_status
        type(self).instances.append(self)

    def request(
        self, method: str, path: str, body: bytes, headers: Mapping[str, str]
    ) -> None:
        """Record the outgoing request."""
        self.requests.append((method, path, body, headers))

    def getresponse(self) -> _FakeHTTPSResponse:
        """Return the canned response."""
        return _FakeHTTPSResponse(self.status)

    def close(self) -> None:
        """No-op close for API parity with the real connection."""


class _FakeSMTP:
    """Records starttls/send_message calls; stands in for `smtplib.SMTP`."""

    instances: ClassVar[list[_FakeSMTP]] = []

    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.starttls_calls: list[object] = []
        self.sent_messages: list[EmailMessage] = []
        type(self).instances.append(self)

    def __enter__(self) -> _FakeSMTP:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def starttls(self, *, context: object = None) -> None:
        """Record the TLS context used to upgrade the connection."""
        self.starttls_calls.append(context)

    def send_message(self, message: EmailMessage) -> None:
        """Record the message that would have been sent."""
        self.sent_messages.append(message)


class TestNtfySink:
    """Tests for `NtfySink.send` against an injected `HttpTransport`."""

    def test_send_success_posts_expected_url_body_and_headers(self) -> None:
        """A 200 response posts once with the topic URL, message, and metadata."""
        calls: list[tuple[str, bytes, Mapping[str, str]]] = []

        def spy_transport(url: str, body: bytes, headers: Mapping[str, str]) -> int:
            calls.append((url, body, dict(headers)))
            return 200

        config = NtfySinkConfig(base_url="https://ntfy.example.com", topic="alerts")
        sink = NtfySink(config, transport=spy_transport, allowlist=_NTFY_ALLOWLIST)

        sink.send(AlertType.HALT_KILL, AlertSeverity.CRITICAL, "kill switch engaged")

        assert len(calls) == 1
        url, body, headers = calls[0]
        assert url == "https://ntfy.example.com/alerts"
        assert b"kill switch engaged" in body
        assert headers["X-Alert-Type"] == AlertType.HALT_KILL.value
        assert headers["X-Alert-Severity"] == AlertSeverity.CRITICAL.value

    def test_send_failure_status_raises_sink_send_error(self) -> None:
        """A non-2xx transport response surfaces as `SinkSendError`."""

        def failing_transport(url: str, body: bytes, headers: Mapping[str, str]) -> int:
            return 500

        config = NtfySinkConfig(base_url="https://ntfy.example.com", topic="alerts")
        sink = NtfySink(config, transport=failing_transport, allowlist=_NTFY_ALLOWLIST)

        with pytest.raises(SinkSendError):
            sink.send(AlertType.VETO, AlertSeverity.WARNING, "vetoed")

    def test_send_transport_raising_oserror_raises_sink_send_error(self) -> None:
        """A transport-level `OSError` becomes a `SinkSendError`, not a crash."""

        def raising_transport(url: str, body: bytes, headers: Mapping[str, str]) -> int:
            raise OSError("connection refused")

        config = NtfySinkConfig(base_url="https://ntfy.example.com", topic="alerts")
        sink = NtfySink(config, transport=raising_transport, allowlist=_NTFY_ALLOWLIST)

        with pytest.raises(SinkSendError):
            sink.send(AlertType.VETO, AlertSeverity.WARNING, "vetoed")


class TestNtfySinkAllowlist:
    """Failing-first tests for `NtfySink`'s required outbound allowlist gate
    (issue #57, RED).

    `NtfySink.__init__` does not yet accept (much less *require*) an
    `allowlist` keyword, so every construction call below that passes one
    fails with `TypeError: __init__() got an unexpected keyword argument
    'allowlist'` -- the expected Gate 1 RED state for issue #57.

    NOTE (flagged for implementation to confirm): making `allowlist`
    *required* (no default) means every pre-existing `NtfySink(config, ...)`
    call in `TestNtfySink` above will need updating once this lands -- a
    wide blast radius on this file's existing (currently-green) tests, out
    of scope for this test-authorship pass.
    """

    def test_construction_against_an_off_list_host_raises_and_never_transports(
        self,
    ) -> None:
        """An `NtfySink` built against a `base_url` host absent from the
        supplied `allowlist` raises at construction, and the transport spy
        is never invoked.
        """
        from windbreak.net.allowlist import EgressDeniedError, OutboundAllowlist

        calls: list[tuple[str, bytes, Mapping[str, str]]] = []

        def spy_transport(url: str, body: bytes, headers: Mapping[str, str]) -> int:
            calls.append((url, body, dict(headers)))
            return 200

        allowlist = OutboundAllowlist(frozenset({"some-other-host.example.com"}))
        config = NtfySinkConfig(base_url="https://ntfy.example.com", topic="alerts")

        with pytest.raises(EgressDeniedError):
            NtfySink(config, transport=spy_transport, allowlist=allowlist)

        assert calls == []

    def test_construction_against_an_allowlisted_host_still_constructs_and_sends(
        self,
    ) -> None:
        """An `NtfySink` whose `base_url` host *is* on the allowlist
        constructs cleanly and sends exactly as before.
        """
        from windbreak.net.allowlist import OutboundAllowlist

        calls: list[tuple[str, bytes, Mapping[str, str]]] = []

        def spy_transport(url: str, body: bytes, headers: Mapping[str, str]) -> int:
            calls.append((url, body, dict(headers)))
            return 200

        allowlist = OutboundAllowlist(frozenset({"ntfy.example.com"}))
        config = NtfySinkConfig(base_url="https://ntfy.example.com", topic="alerts")
        sink = NtfySink(config, transport=spy_transport, allowlist=allowlist)

        sink.send(AlertType.HALT_KILL, AlertSeverity.CRITICAL, "kill switch engaged")

        assert len(calls) == 1


class TestWebhookSink:
    """Tests for `WebhookSink.send` against an injected `HttpTransport`."""

    def test_send_success_posts_json_payload_with_type_severity_message(self) -> None:
        """A 200 response posts JSON carrying type, severity, and message."""
        calls: list[tuple[str, bytes, Mapping[str, str]]] = []

        def spy_transport(url: str, body: bytes, headers: Mapping[str, str]) -> int:
            calls.append((url, body, dict(headers)))
            return 200

        config = WebhookSinkConfig(url="https://hooks.example.com/incoming")
        sink = WebhookSink(
            config, transport=spy_transport, allowlist=_WEBHOOK_ALLOWLIST
        )

        sink.send(AlertType.DISK_HALT, AlertSeverity.CRITICAL, "disk full")

        assert len(calls) == 1
        url, body, _headers = calls[0]
        assert url == "https://hooks.example.com/incoming"
        payload = json.loads(body)
        assert payload["type"] == AlertType.DISK_HALT.value
        assert payload["severity"] == AlertSeverity.CRITICAL.value
        assert payload["message"] == "disk full"

    def test_send_failure_status_raises_sink_send_error(self) -> None:
        """A non-2xx transport response surfaces as `SinkSendError`."""

        def failing_transport(url: str, body: bytes, headers: Mapping[str, str]) -> int:
            return 503

        config = WebhookSinkConfig(url="https://hooks.example.com/incoming")
        sink = WebhookSink(
            config, transport=failing_transport, allowlist=_WEBHOOK_ALLOWLIST
        )

        with pytest.raises(SinkSendError):
            sink.send(AlertType.VETO, AlertSeverity.WARNING, "vetoed")

    def test_send_transport_raising_raises_sink_send_error(self) -> None:
        """A transport-level exception becomes a `SinkSendError`."""

        def raising_transport(url: str, body: bytes, headers: Mapping[str, str]) -> int:
            raise OSError("dns failure")

        config = WebhookSinkConfig(url="https://hooks.example.com/incoming")
        sink = WebhookSink(
            config, transport=raising_transport, allowlist=_WEBHOOK_ALLOWLIST
        )

        with pytest.raises(SinkSendError):
            sink.send(AlertType.VETO, AlertSeverity.WARNING, "vetoed")

    def test_construction_rejects_a_url_off_the_allowlist(self) -> None:
        """An off-allowlist webhook URL is refused before any POST (issue #274).

        Closes the SSRF gap the sink shipped with: `WebhookSink` used to accept
        any config-supplied `https` URL, so a mistyped or tampered endpoint
        could reach an internal host. It now screens at construction exactly as
        `NtfySink` does -- the transport is never even consulted.
        """
        transport_calls: list[str] = []

        def spy_transport(url: str, body: bytes, headers: Mapping[str, str]) -> int:
            transport_calls.append(url)
            return 200

        config = WebhookSinkConfig(url="https://169.254.169.254/latest/meta-data")

        with pytest.raises(EgressDeniedError):
            WebhookSink(config, transport=spy_transport, allowlist=_WEBHOOK_ALLOWLIST)

        assert transport_calls == []

    def test_construction_rejects_a_non_https_url(self) -> None:
        """A `file://` webhook endpoint is refused by the allowlist's scheme check."""
        config = WebhookSinkConfig(url="file:///etc/passwd")

        with pytest.raises(EgressDeniedError):
            WebhookSink(
                config, transport=lambda *_args: 200, allowlist=_WEBHOOK_ALLOWLIST
            )


class TestDefaultHttpsPost:
    """Tests for the default `_https_post` transport implementation."""

    def setup_method(self) -> None:
        """Reset the fake connection's recorded instances and default status."""
        _FakeHTTPSConnection.instances = []
        _FakeHTTPSConnection.default_status = 200

    def test_happy_path_uses_httpsconnection_and_returns_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 2xx response is returned as the status code, via HTTPSConnection."""
        monkeypatch.setattr(http.client, "HTTPSConnection", _FakeHTTPSConnection)

        status = _https_post(
            "https://ntfy.example.com/alerts", b"payload", {"X-Test": "1"}
        )

        assert status == 200
        assert len(_FakeHTTPSConnection.instances) == 1
        conn = _FakeHTTPSConnection.instances[0]
        assert conn.host == "ntfy.example.com"
        assert conn.context is not None
        assert conn.timeout == _TRANSPORT_TIMEOUT_SECONDS
        method, path, body, headers = conn.requests[0]
        assert method == "POST"
        assert path == "/alerts"
        assert body == b"payload"
        assert headers["X-Test"] == "1"

    def test_rejects_non_https_url_before_connecting(self) -> None:
        """A non-`https://` URL is rejected without attempting a connection."""
        with pytest.raises(SinkSendError):
            _https_post("http://ntfy.example.com/alerts", b"payload", {})

    def test_non_2xx_status_raises_sink_send_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-2xx HTTPS response is surfaced as `SinkSendError`."""
        _FakeHTTPSConnection.default_status = 500
        monkeypatch.setattr(http.client, "HTTPSConnection", _FakeHTTPSConnection)

        with pytest.raises(SinkSendError):
            _https_post("https://ntfy.example.com/alerts", b"payload", {})


class TestSmtpSink:
    """Tests for `SmtpSink.send` against an injected `SmtpTransport`."""

    def test_send_success_builds_email_and_calls_transport(self) -> None:
        """The built `EmailMessage` carries sender, recipients, and alert content."""
        calls: list[tuple[SmtpSinkConfig, EmailMessage]] = []

        def spy_transport(config: SmtpSinkConfig, message: EmailMessage) -> None:
            calls.append((config, message))

        config = SmtpSinkConfig(
            host="smtp.example.com",
            port=587,
            sender="alerts@example.com",
            recipients=("ops@example.com", "oncall@example.com"),
        )
        sink = SmtpSink(config, transport=spy_transport, allowlist=_SMTP_ALLOWLIST)

        sink.send(AlertType.BACKUP_FAILURE, AlertSeverity.WARNING, "backup job failed")

        assert len(calls) == 1
        sent_config, message = calls[0]
        assert sent_config == config
        assert message["From"] == "alerts@example.com"
        assert message["To"] == "ops@example.com, oncall@example.com"
        assert AlertType.BACKUP_FAILURE.value in message["Subject"]
        assert AlertSeverity.WARNING.value in message["Subject"]
        assert "backup job failed" in message.get_content()

    def test_send_transport_raising_raises_sink_send_error(self) -> None:
        """A transport-level exception becomes a `SinkSendError`."""

        def raising_transport(config: SmtpSinkConfig, message: EmailMessage) -> None:
            raise OSError("smtp down")

        config = SmtpSinkConfig(
            host="smtp.example.com",
            port=587,
            sender="a@example.com",
            recipients=("b@example.com",),
        )
        sink = SmtpSink(config, transport=raising_transport, allowlist=_SMTP_ALLOWLIST)

        with pytest.raises(SinkSendError):
            sink.send(AlertType.VETO, AlertSeverity.WARNING, "vetoed")

    def test_construction_rejects_a_relay_host_off_the_allowlist(self) -> None:
        """An off-allowlist SMTP relay is refused before any connection (#274).

        SMTP has no URL to screen, so the bare host goes through
        `OutboundAllowlist.require_host`; without it SMTP would be the one
        outbound alert path with no egress check at all.
        """
        transport_calls: list[SmtpSinkConfig] = []

        def spy_transport(config: SmtpSinkConfig, message: EmailMessage) -> None:
            transport_calls.append(config)

        config = SmtpSinkConfig(
            host="relay.internal",
            port=25,
            sender="a@example.com",
            recipients=("b@example.com",),
        )

        with pytest.raises(EgressDeniedError):
            SmtpSink(config, transport=spy_transport, allowlist=_SMTP_ALLOWLIST)

        assert transport_calls == []


class TestDefaultSmtpSend:
    """Tests for the default `_smtp_send` transport implementation."""

    def setup_method(self) -> None:
        """Reset the fake SMTP client's recorded instances."""
        _FakeSMTP.instances = []

    def test_upgrades_to_tls_then_sends_the_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_smtp_send` connects, calls `starttls`, then `send_message`."""
        monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
        config = SmtpSinkConfig(
            host="smtp.example.com",
            port=587,
            sender="a@example.com",
            recipients=("b@example.com",),
        )
        message = EmailMessage()
        message["Subject"] = "test"

        _smtp_send(config, message)

        assert len(_FakeSMTP.instances) == 1
        conn = _FakeSMTP.instances[0]
        assert conn.host == "smtp.example.com"
        assert conn.port == 587
        assert conn.timeout == _TRANSPORT_TIMEOUT_SECONDS
        assert len(conn.starttls_calls) == 1
        assert conn.starttls_calls[0] is not None
        assert conn.sent_messages == [message]


class TestDesktopSink:
    """Tests for `DesktopSink.send` and its no-notifier failure mode."""

    def test_send_calls_notifier_with_title_and_body(self) -> None:
        """The injected notifier receives a title and body derived from the alert."""
        calls: list[tuple[str, str]] = []

        def spy_notifier(title: str, body: str) -> None:
            calls.append((title, body))

        sink = DesktopSink(notifier=spy_notifier)

        sink.send(AlertType.CANARY_DRIFT, AlertSeverity.INFO, "canary drifted 3%")

        assert len(calls) == 1
        title, body = calls[0]
        assert AlertType.CANARY_DRIFT.value in title
        assert "canary drifted 3%" in body

    def test_send_without_notifier_raises_sink_send_error(self) -> None:
        """No desktop notifier available means the sink cannot deliver."""
        sink = DesktopSink(notifier=None)

        with pytest.raises(SinkSendError):
            sink.send(AlertType.CANARY_DRIFT, AlertSeverity.INFO, "canary drifted 3%")

    def test_default_constructor_has_no_notifier_and_raises(self) -> None:
        """`DesktopSink()` with no argument behaves like `notifier=None`."""
        sink = DesktopSink()

        with pytest.raises(SinkSendError):
            sink.send(AlertType.VETO, AlertSeverity.WARNING, "vetoed")


class TestLogOnlySink:
    """Tests for `LogOnlySink.send`, the always-available fallback sink."""

    def test_emits_at_severity_mapped_level_with_alert_fields(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """CRITICAL severity produces a CRITICAL record carrying alert fields."""
        caplog.set_level(logging.DEBUG)
        sink = LogOnlySink()

        sink.send(AlertType.DRAWDOWN_DEMOTION, AlertSeverity.CRITICAL, "demoted to L2")

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.CRITICAL
        assert record.component == "alerts"
        assert record.alert_type == AlertType.DRAWDOWN_DEMOTION.value
        assert record.severity == AlertSeverity.CRITICAL.value
        assert "demoted to L2" in record.message

    def test_uses_the_injected_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        """A custom logger passed to the constructor is the one that emits."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger("windbreak.test.custom_alerts")
        sink = LogOnlySink(logger=logger)

        sink.send(AlertType.VETO, AlertSeverity.WARNING, "vetoed")

        assert len(caplog.records) == 1
        assert caplog.records[0].name == "windbreak.test.custom_alerts"


@pytest.mark.parametrize(
    "sink_factory",
    [
        lambda: NtfySink(
            NtfySinkConfig(base_url="https://n.example.com", topic="t"),
            transport=lambda *_args: 200,
            allowlist=OutboundAllowlist(frozenset({"n.example.com"})),
        ),
        lambda: WebhookSink(
            WebhookSinkConfig(url="https://w.example.com"),
            transport=lambda *_args: 200,
            allowlist=OutboundAllowlist(frozenset({"w.example.com"})),
        ),
        lambda: SmtpSink(
            SmtpSinkConfig(
                host="h", port=25, sender="a@example.com", recipients=("b@example.com",)
            ),
            transport=lambda *_args: None,
            allowlist=OutboundAllowlist(frozenset({"h"})),
        ),
        lambda: DesktopSink(notifier=lambda *_args: None),
        lambda: LogOnlySink(),
    ],
)
def test_every_sink_exposes_a_nonempty_name_attribute(sink_factory: object) -> None:
    """Every concrete sink satisfies the `AlertSink` protocol's `name: str`."""
    sink = sink_factory()  # type: ignore[operator]

    assert isinstance(sink.name, str)
    assert sink.name != ""


@pytest.mark.parametrize(
    "config",
    [
        NtfySinkConfig(base_url="https://n.example.com", topic="t"),
        WebhookSinkConfig(url="https://w.example.com"),
        SmtpSinkConfig(
            host="h", port=25, sender="a@example.com", recipients=("b@example.com",)
        ),
    ],
)
def test_sink_config_dataclasses_are_frozen(config: object) -> None:
    """Every sink config dataclass is immutable once constructed."""
    field_name = next(iter(dataclasses.fields(config))).name

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, field_name, getattr(config, field_name))
