"""Tests for the alert-sink schema fields the sink factory reads (issue #274).

Before this issue `AlertSink` carried only ``type``/``topic``, which is thinner
than every concrete ``windbreak.alerts.sinks.*SinkConfig`` shape -- so no
webhook or SMTP sink could be described in configuration at all. These tests
pin the widened schema: its YAML round-trip, its fail-closed placeholder
defaults, and that unknown keys stay fatal inside the new nesting level.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from windbreak.config import ConfigError, WindbreakConfig, load_config
from windbreak.config.schema import (
    UNCONFIGURED_PLACEHOLDER,
    AlertSink,
    SmtpSinkSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any


def test_default_alert_sink_destinations_are_all_placeholders() -> None:
    """The shipped sink is deliberately undeliverable until an operator acts.

    Every destination field defaults to the "operator must fill this in"
    placeholder, so `build_sinks` constructs nothing and alerts fall back to
    log-only rather than being POSTed at an invented endpoint.
    """
    sink = WindbreakConfig().alerts.sinks[0]

    assert sink.type == "ntfy"
    assert sink.topic == UNCONFIGURED_PLACEHOLDER
    assert sink.base_url == UNCONFIGURED_PLACEHOLDER
    assert sink.url == UNCONFIGURED_PLACEHOLDER
    assert sink.smtp == SmtpSinkSettings()


def test_default_smtp_settings_are_unconfigured() -> None:
    """An untouched SMTP block names no relay, no sender, and no recipient."""
    smtp = SmtpSinkSettings()

    assert smtp.host == UNCONFIGURED_PLACEHOLDER
    assert smtp.sender == UNCONFIGURED_PLACEHOLDER
    assert smtp.recipients == ()
    assert smtp.port == 587


def test_default_allowed_hosts_is_empty() -> None:
    """An operator who declares no alert host gets no alert egress."""
    assert WindbreakConfig().alerts.allowed_hosts == ()


def test_alerts_section_round_trips_every_sink_shape(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """One YAML alerts section describes ntfy, webhook, and SMTP sinks."""
    config_path = write_config(
        tmp_path,
        {
            "alerts": {
                "allowed_hosts": ["ntfy.example.com", "hooks.example.com"],
                "sinks": [
                    {
                        "type": "ntfy",
                        "topic": "windbreak-ops",
                        "base_url": "https://ntfy.example.com",
                    },
                    {"type": "webhook", "url": "https://hooks.example.com/incoming"},
                    {
                        "type": "smtp",
                        "smtp": {
                            "host": "smtp.example.com",
                            "port": 587,
                            "sender": "windbreak@example.com",
                            "recipients": ["ops@example.com"],
                        },
                    },
                ],
            }
        },
    )

    alerts = load_config(config_path).alerts

    assert alerts.allowed_hosts == ("ntfy.example.com", "hooks.example.com")
    assert alerts.sinks[0] == AlertSink(
        type="ntfy", topic="windbreak-ops", base_url="https://ntfy.example.com"
    )
    assert alerts.sinks[1].url == "https://hooks.example.com/incoming"
    assert alerts.sinks[2].smtp == SmtpSinkSettings(
        host="smtp.example.com",
        port=587,
        sender="windbreak@example.com",
        recipients=("ops@example.com",),
    )


def test_unknown_key_inside_a_sink_is_fatal(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """A misspelled sink field is rejected, never silently ignored.

    Silently dropping `base_ur1` would leave the sink unconfigured and the
    operator believing alerts were wired -- the exact failure this issue exists
    to remove.
    """
    config_path = write_config(
        tmp_path,
        {"alerts": {"sinks": [{"type": "ntfy", "base_ur1": "https://ntfy.example"}]}},
    )

    with pytest.raises(ConfigError, match="base_ur1"):
        load_config(config_path)


def test_unknown_key_inside_the_smtp_block_is_fatal(
    tmp_path: Path, write_config: Callable[[Path, dict[str, Any]], Path]
) -> None:
    """Unknown-keys-are-fatal reaches the newly nested SMTP mapping too."""
    config_path = write_config(
        tmp_path,
        {"alerts": {"sinks": [{"type": "smtp", "smtp": {"hostname": "relay"}}]}},
    )

    with pytest.raises(ConfigError, match="hostname"):
        load_config(config_path)


def test_sink_type_has_no_default() -> None:
    """`type` stays required: a sink that names no transport is malformed.

    Every *destination* field gained a placeholder default so a half-filled
    entry fails closed, but `type` deliberately did not -- there is no
    defensible default channel to guess at. Checked through the dataclass's
    own field metadata rather than a construction call, so this stays a
    statement about the schema and not about `TypeError` plumbing.
    """
    fields_by_name = {field.name: field for field in dataclasses.fields(AlertSink)}

    assert fields_by_name["type"].default is dataclasses.MISSING
    assert fields_by_name["type"].default_factory is dataclasses.MISSING
    assert fields_by_name["topic"].default == UNCONFIGURED_PLACEHOLDER
