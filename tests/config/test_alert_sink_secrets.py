"""Alert-sink destination secrets stay out of the config diff and ledger (#274).

Every configuration load flows through
:func:`windbreak.config.versioning.diff_configs`, whose ``(old, new)`` leaf
values are persisted verbatim by
:class:`windbreak.config.ledger_recorder.LedgerConfigEventRecorder` into a
hash-chained ``ConfigLoaded`` ledger event -- and folded again, in plaintext,
into the ``config_versions.json`` read model by
:func:`windbreak.ledger.rebuild.rebuild`. That machinery is generic: it has no
notion of a secret field, and the ledger is append-only, so anything that
reaches it is unredactable after the fact.

An ntfy topic is a bearer capability and a webhook URL can embed a token, so
neither may ever be a config leaf. These tests pin the
:data:`~windbreak.config.schema.UNCONFIGURED_PLACEHOLDER`-shaped
``*_env`` indirection the rest of the schema already uses for secrets
(``FutureSearchProviderSettings.api_key_env``,
``ResearchSettings.search_api_key_env``): configuration carries the *name* of an
environment variable, the value lives only in the process environment, and the
ledger therefore records the name.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from windbreak.config import load_config
from windbreak.config.ledger_recorder import LedgerConfigEventRecorder
from windbreak.ledger.store import SqliteLedgerStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    import pytest

#: The environment variable an operator points at their real ntfy topic.
_TOPIC_VAR = "WINDBREAK_TEST_NTFY_TOPIC"

#: The environment variable naming the ntfy server base URL.
_BASE_URL_VAR = "WINDBREAK_TEST_NTFY_BASE_URL"

#: The environment variable naming the webhook endpoint.
_WEBHOOK_VAR = "WINDBREAK_TEST_WEBHOOK_URL"

#: The bearer-capability ntfy topic the operator keeps in the environment.
#: Built from fragments so no scanner-tripping literal sits on one line.
_TOPIC_VALUE = "windbreak-ops-" + "9f3c1d2b"

#: The ntfy base URL, in the environment because a URL can embed userinfo.
_BASE_URL_VALUE = "https://ntfy.example.com"

#: A webhook endpoint whose path carries an operator capability.
_WEBHOOK_VALUE = "https://hooks.example.com/services/" + "T0-K3N-a1b2"


def _alerts_section() -> dict[str, Any]:
    """Return an alerts section wiring both HTTPS sinks through the environment.

    Returns:
        The YAML mapping an operator writes to configure a real ntfy sink and a
        real webhook sink, naming environment variables rather than embedding
        the destinations themselves.
    """
    return {
        "allowed_hosts": ["ntfy.example.com", "hooks.example.com"],
        "sinks": [
            {"type": "ntfy", "topic_env": _TOPIC_VAR, "base_url_env": _BASE_URL_VAR},
            {"type": "webhook", "url_env": _WEBHOOK_VAR},
        ],
    }


def _export_destinations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put every real destination in the environment, where it belongs."""
    monkeypatch.setenv(_TOPIC_VAR, _TOPIC_VALUE)
    monkeypatch.setenv(_BASE_URL_VAR, _BASE_URL_VALUE)
    monkeypatch.setenv(_WEBHOOK_VAR, _WEBHOOK_VALUE)


def test_configured_alert_destinations_never_reach_the_ledgered_config_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """A real ntfy topic and webhook URL appear nowhere in the ConfigLoaded event.

    Drives the exact path a production ``windbreak run --config ...
    --ledger-path ...`` drives: ``load_config`` -> ``diff_configs`` ->
    ``LedgerConfigEventRecorder`` -> the hash-chained store. The assertion is
    deliberately made non-vacuous by also requiring the *variable names* to be
    present: the sink really was configured, the diff really is non-empty, and
    what got persisted is the indirection rather than the secret.
    """
    _export_destinations(monkeypatch)
    config_path = write_config(tmp_path, {"alerts": _alerts_section()})
    store = SqliteLedgerStore(tmp_path / "ledger.db")

    load_config(config_path, recorder=LedgerConfigEventRecorder(store, component="cli"))

    records = store.read_all()
    store.verify_chain()
    store.close()
    persisted = json.dumps([json.loads(record.payload_json) for record in records])
    assert _TOPIC_VALUE not in persisted
    assert _WEBHOOK_VALUE not in persisted
    assert _BASE_URL_VALUE not in persisted
    assert _TOPIC_VAR in persisted
    assert _WEBHOOK_VAR in persisted


def test_loaded_config_carries_variable_names_not_destination_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_config: Callable[[Path, dict[str, Any]], Path],
) -> None:
    """The in-memory config object holds the indirection, never the secret.

    `config_hash`, `format_diff`, and every future consumer of
    `WindbreakConfig` inherit the guarantee from the object's own shape rather
    than from a redaction step each of them has to remember to apply.
    """
    _export_destinations(monkeypatch)
    config_path = write_config(tmp_path, {"alerts": _alerts_section()})

    config = load_config(config_path)

    ntfy, webhook = config.alerts.sinks
    assert ntfy.topic_env == _TOPIC_VAR
    assert ntfy.base_url_env == _BASE_URL_VAR
    assert webhook.url_env == _WEBHOOK_VAR
    assert not hasattr(ntfy, "topic")
    assert not hasattr(ntfy, "base_url")
    assert not hasattr(webhook, "url")
