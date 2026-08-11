"""The kill-switch audit row records delivery, not just emission (issue #413).

PR #412 (issue #287) made the kill switch ledger one `AlertEmitted` row proving
the `HALT_KILL` page was *emitted*. If every sink fails -- pager down, webhook
500, SMTP refused -- the fallback still fires and that row looks identical to
the all-succeeded case, so a post-incident audit can establish that the kill
fired and why, but not that anyone was told.

#412 stopped short deliberately, and this module preserves the reason:
`SinkOutcome.detail` is `str(exc)` from an arbitrary sink -- the exact shape
that leaked token-bearing URLs in issue #274 -- and a hash chain is
append-only, so nothing written into it can ever be redacted. The fix is
therefore a *closed* payload: a sink identity drawn from the sink classes the
code defines, plus a four-member `DeliveryOutcome`, and never `str(exc)`.

These tests pin, against a real `AlertDispatcher` over real sinks:

- the all-succeeded and all-failed cases are distinguishable in the ledger;
- no substring of a token-bearing URL a sink raises with reaches the chain
  (read across `ledger.db*`, because `PRAGMA journal_mode=WAL` leaves a fresh
  row in the sidecar and a search of `ledger.db` alone passes vacuously);
- a dispatcher that reports nothing records *unreported*, never delivered;
- #412's ordering and raise-don't-swallow behaviour still hold.

RED before the implementation: `windbreak.alerts.delivery` does not exist and
the ledger `AlertEmitted` carries no delivery fields, so this module fails at
collection with `ModuleNotFoundError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from windbreak.alerts.dispatch import AlertDispatcher, LoggingLedgerWriter
from windbreak.alerts.registry import AlertSeverity, AlertType
from windbreak.alerts.sinks import WebhookSink, WebhookSinkConfig
from windbreak.ledger.store import SqliteLedgerStore, events_from_records
from windbreak.net.allowlist import OutboundAllowlist
from windbreak.riskkernel.kill import AlertDispatcherProtocol, KillSwitch, KillTrigger
from windbreak.riskkernel.modes import Mode, ModeStateMachine
from windbreak.riskkernel.process import (
    InMemoryKernelLedgerWriter,
    PersistingKernelLedgerWriter,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from windbreak.ledger.events import Event

#: The fixed epoch every switch below reports, so no assertion depends on the
#: wall clock.
_FIXED_EPOCH_S = 1_700_000_000

#: The allowlisted host every real `WebhookSink` here posts to.
_WEBHOOK_HOST = "hooks.example.com"

#: The operator's bearer token, embedded in the webhook destination. Kept as
#: its own constant so a test can assert on the token independently of the URL
#: that carries it.
_WEBHOOK_TOKEN = "s3cr3t-bearer-abc123"

#: A token-bearing webhook destination: the shape issue #274 found echoed whole
#: out of an error path, and the shape `_send_http` still raises verbatim
#: (`SinkSendError(str(exc))`) when its transport fails.
_WEBHOOK_URL = f"https://{_WEBHOOK_HOST}/services/{_WEBHOOK_TOKEN}?key=k1"

#: The shortest fragment of the destination a leak-detector should catch. Any
#: substring search that misses this would miss the whole URL too.
_TOKEN_FRAGMENT = _WEBHOOK_TOKEN

#: The closed key set the ledgered `AlertEmitted` payload may ever hold. A
#: fifth key is a schema change, and this set is what fails when one appears.
_CLOSED_ALERT_PAYLOAD_KEYS = frozenset(
    {"severity", "message", "deliveries", "delivery_reported"}
)

#: The two keys issue #413 added. Everything outside them is what a pre-#413
#: row already held, and is what made the failed and delivered rows identical.
_DELIVERY_KEYS = frozenset({"deliveries", "delivery_reported"})


def _webhook(*, failing: bool) -> WebhookSink:
    """Build a real `WebhookSink` over a token-bearing URL and a fake transport.

    Only the HTTP transport is replaced; the allowlist screen, the URL, the
    JSON body and the `SinkSendError` wrapping are all the real ones, because
    the leak this guards against travels on the real error path.

    Args:
        failing: When True the transport raises with the whole destination URL
            in its message, exactly as a `http.client` failure would.

    Returns:
        A `WebhookSink` whose `send` either succeeds or raises.
    """

    def _transport(url: str, body: bytes, headers: Mapping[str, str]) -> int:
        """Answer the POST, or fail with the destination in the message.

        Args:
            url: The destination posted to.
            body: The request body (ignored).
            headers: The request headers (ignored).

        Returns:
            HTTP 200 when this transport is not the failing one.

        Raises:
            OSError: When failing, carrying the whole destination URL.
        """
        del body, headers
        if failing:
            raise OSError(f"connection reset posting to {url}")
        return 200

    return WebhookSink(
        WebhookSinkConfig(url=_WEBHOOK_URL),
        transport=_transport,
        allowlist=OutboundAllowlist(frozenset({_WEBHOOK_HOST})),
    )


class _SilentDispatcher:
    """A `KillSwitch` alert dispatcher that reports no delivery evidence.

    The shape every pre-#413 test double had: it pages, and says nothing about
    whether the page landed. The audit row must record that absence as absence.
    """

    def __init__(self) -> None:
        """Initialize with an empty dispatch log."""
        self.dispatched: list[AlertType] = []

    def dispatch(self, alert_type: AlertType, message: str) -> None:
        """Record the dispatch and report nothing about its delivery.

        Args:
            alert_type: The alert type dispatched.
            message: The alert body (ignored).
        """
        del message
        self.dispatched.append(alert_type)


def _kill_with(
    dispatcher: AlertDispatcherProtocol, writer: InMemoryKernelLedgerWriter
) -> None:
    """Engage a fresh kill switch wired to `dispatcher`, recording to `writer`.

    Args:
        dispatcher: The alert-dispatcher seam the switch pages through.
        writer: The kernel ledger writer every kill-path event lands in.
    """
    switch = KillSwitch(
        ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE),
        writer,
        dispatcher,
        clock=lambda: _FIXED_EPOCH_S,
    )
    switch.kill(KillTrigger.CLI)


def _alert_payload(events: Sequence[Event]) -> dict[str, object]:
    """Return the single `AlertEmitted` payload in `events`.

    Args:
        events: The recorded kill-path events.

    Returns:
        That row's payload mapping.
    """
    rows = [event for event in events if event.event_type == "AlertEmitted"]
    assert len(rows) == 1
    return rows[0].payload


def _dispatcher_over(sink: WebhookSink | None) -> AlertDispatcher:
    """Build a real `AlertDispatcher` over zero or one real webhook sink.

    Args:
        sink: The one configured sink, or None for a sink-less dispatcher.

    Returns:
        A real dispatcher with the real `LogOnlySink` fallback.
    """
    return AlertDispatcher(
        [] if sink is None else [sink], ledger_writer=LoggingLedgerWriter()
    )


def test_the_ledger_alone_shows_which_sink_accepted_the_kill_page() -> None:
    """A delivered page ledgers the accepting sink and its closed outcome.

    Acceptance criterion 1: a reader with only the ledger can reconstruct that
    at least one sink accepted the alert -- here, the configured webhook, with
    no fallback entry because none was needed.
    """
    writer = InMemoryKernelLedgerWriter()

    _kill_with(_dispatcher_over(_webhook(failing=False)), writer)

    assert _alert_payload(writer.events)["deliveries"] == [
        {"sink": "webhook", "outcome": "delivered", "fallback": False}
    ]
    assert _alert_payload(writer.events)["delivery_reported"] is True


def test_a_total_delivery_failure_is_distinguishable_from_a_delivered_page() -> None:
    """All-sinks-failed and all-sinks-succeeded ledger *different* rows.

    Acceptance criterion 3, and the whole point of issue #413: before it, both
    cases produced a byte-identical `AlertEmitted` row. Both payloads are pinned
    exactly, and the *rest* of each row is then asserted still identical -- so
    the distinction is pinned to the two keys #413 added rather than leaking out
    of some other field that happened to differ.
    """
    failed_writer = InMemoryKernelLedgerWriter()
    delivered_writer = InMemoryKernelLedgerWriter()

    _kill_with(_dispatcher_over(_webhook(failing=True)), failed_writer)
    _kill_with(_dispatcher_over(_webhook(failing=False)), delivered_writer)

    failed = _alert_payload(failed_writer.events)
    delivered = _alert_payload(delivered_writer.events)
    assert failed["deliveries"] == [
        {"sink": "webhook", "outcome": "errored", "fallback": False},
        {"sink": "log-only", "outcome": "delivered", "fallback": True},
    ]
    assert delivered["deliveries"] == [
        {"sink": "webhook", "outcome": "delivered", "fallback": False}
    ]
    # `failed != delivered` would be implied by the two exact pins above and so
    # could never fail. What is *not* implied: that every field outside the two
    # keys #413 added is still identical between the two rows -- which is both
    # the reason the pre-#413 rows were indistinguishable and the guarantee that
    # the new distinction is carried by the delivery evidence and nothing else.
    # Guarded first against a payload reduced to the delivery keys alone, which
    # would otherwise compare two empty mappings and pass vacuously.
    assert set(failed) - _DELIVERY_KEYS
    assert {key: failed[key] for key in set(failed) - _DELIVERY_KEYS} == {
        key: delivered[key] for key in set(delivered) - _DELIVERY_KEYS
    }


def test_a_dispatcher_that_reports_nothing_ledgers_absence_not_delivery() -> None:
    """No delivery report ledgers `delivery_reported: false` and no rows.

    Fail closed: a dispatcher that cannot say whether the page landed must
    never produce a row a reader could mistake for a delivered one.
    """
    writer = InMemoryKernelLedgerWriter()
    dispatcher = _SilentDispatcher()

    _kill_with(dispatcher, writer)

    payload = _alert_payload(writer.events)
    assert dispatcher.dispatched == [AlertType.HALT_KILL]
    assert payload["delivery_reported"] is False
    assert payload["deliveries"] == []


def test_the_kill_alert_payload_stays_a_closed_four_key_shape() -> None:
    """The ledgered payload holds exactly the four closed keys, never a fifth."""
    writer = InMemoryKernelLedgerWriter()

    _kill_with(_dispatcher_over(_webhook(failing=True)), writer)

    assert set(_alert_payload(writer.events)) == _CLOSED_ALERT_PAYLOAD_KEYS


def test_no_substring_of_a_token_bearing_sink_url_reaches_the_kill_chain(
    tmp_path: Path,
) -> None:
    """A failing sink's exception text never lands in the append-only chain.

    Acceptance criterion 2, and the reason #412 refused to persist outcomes at
    all. The sink raises with the whole destination -- token and query string --
    in its message, and `_send_http` re-raises that verbatim as
    `SinkSendError(str(exc))`, so the unredactable text genuinely exists on the
    `SinkOutcome` this row is derived from.

    The search reads `ledger.db*`, not `ledger.db`: `SqliteLedgerStore` sets
    `PRAGMA journal_mode=WAL`, so a freshly appended row lives in the `-wal`
    sidecar until a checkpoint and a search of the database file alone would
    pass while the secret sat on disk (found live as mutation M9 of PR #474).
    """
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    sink = _webhook(failing=True)
    dispatcher = _dispatcher_over(sink)
    switch = KillSwitch(
        ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE),
        PersistingKernelLedgerWriter(store),
        dispatcher,
        clock=lambda: _FIXED_EPOCH_S,
    )

    switch.kill(KillTrigger.AUTO_RECONCILIATION)

    # `verify_chain()` returns None and raises on a broken chain, so it is
    # called bare: asserting on its return value would assert nothing.
    store.verify_chain()
    replayed = events_from_records(store.read_all())
    payload = _alert_payload(replayed)
    # Guard first, so the substring sweep below cannot pass vacuously against a
    # run that never dialled the sink or never wrote the row.
    assert payload["deliveries"] == [
        {"sink": "webhook", "outcome": "errored", "fallback": False},
        {"sink": "log-only", "outcome": "delivered", "fallback": True},
    ]
    blob = b"".join(path.read_bytes() for path in sorted(tmp_path.glob("ledger.db*")))
    assert blob
    for secret in (_WEBHOOK_URL, _TOKEN_FRAGMENT, _WEBHOOK_HOST, "connection reset"):
        assert secret.encode("utf-8") not in blob


class _KillFileWatchingWriter:
    """Records, per appended event, whether the `KILL` file existed *yet*.

    The event list alone cannot see the ordering this pins: moving the audit
    append above `_write_kill_file()` leaves `AlertEmitted` last in the list all
    the same, and the file exists by the time the test looks either way. Sampling
    the filesystem *at append time* is what makes the ordering observable.
    """

    def __init__(self, state_dir: Path) -> None:
        """Initialize with an empty log and the directory the `KILL` file lands in.

        Args:
            state_dir: The kill switch's state directory.
        """
        self._kill_file = state_dir / "KILL"
        self.events: list[Event] = []
        self.kill_file_seen: dict[str, bool] = {}

    def record(self, event: Event) -> None:
        """Retain `event` alongside whether the `KILL` file was on disk yet.

        Args:
            event: The kernel event to persist.
        """
        self.kill_file_seen[event.event_type] = self._kill_file.exists()
        self.events.append(event)


def test_the_delivery_bearing_row_is_still_the_kill_paths_last_write(
    tmp_path: Path,
) -> None:
    """#412's ordering survives: the audit append stays after every fail-safe.

    A failing append must never be able to skip the halt, the cancel-all, the
    release, the page or the on-disk `KILL` file, so the row carrying delivery
    evidence is written last -- exactly where the emission-only row was.

    The `KILL`-file half of that claim is asserted at append time rather than
    after the fact: `kill_file_seen` is False when `KillEngaged` is written and
    True when `AlertEmitted` is, which is false for any reordering that hoists
    the audit append above the kill file.
    """
    writer = _KillFileWatchingWriter(tmp_path)
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE)
    switch = KillSwitch(
        machine,
        writer,
        _dispatcher_over(_webhook(failing=True)),
        state_dir=tmp_path,
        clock=lambda: _FIXED_EPOCH_S,
    )

    switch.kill(KillTrigger.CLI)

    assert [event.event_type for event in writer.events] == [
        "KillEngaged",
        "CancelAllDirective",
        "AlertEmitted",
    ]
    assert writer.kill_file_seen == {
        "KillEngaged": False,
        "CancelAllDirective": False,
        "AlertEmitted": True,
    }
    assert machine.mode is Mode.KILLED
    assert (tmp_path / "KILL").exists()


class _RefusingKernelWriter:
    """A kernel ledger writer that refuses the audit row and keeps the rest."""

    def __init__(self) -> None:
        """Initialize with an empty log of the events that did persist."""
        self.events: list[Event] = []

    def record(self, event: Event) -> None:
        """Retain `event` unless it is the audit row, which raises.

        Args:
            event: The kernel event to persist.

        Raises:
            OSError: If `event` is the `AlertEmitted` audit row.
        """
        if event.event_type == "AlertEmitted":
            raise OSError("ledger append failed")
        self.events.append(event)


def test_a_delivery_bearing_audit_append_still_raises_rather_than_swallowing(
    tmp_path: Path,
) -> None:
    """#412's raise-don't-swallow survives the schema bump.

    A kill switch whose audit record silently fails to write is worse than one
    that never claimed to write, so the append stays unguarded -- and because
    it is last, the halt, the page and the `KILL` file are already done.
    """
    writer = _RefusingKernelWriter()
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE)
    switch = KillSwitch(
        machine,
        writer,
        _dispatcher_over(_webhook(failing=False)),
        state_dir=tmp_path,
        clock=lambda: _FIXED_EPOCH_S,
    )

    with pytest.raises(OSError, match="ledger append failed") as raised:
        switch.kill(KillTrigger.CLI)

    assert type(raised.value) is OSError
    assert str(raised.value) == "ledger append failed"
    assert machine.mode is Mode.KILLED
    assert (tmp_path / "KILL").exists()
    assert [event.event_type for event in writer.events] == [
        "KillEngaged",
        "CancelAllDirective",
    ]


def test_the_bumped_row_still_stamps_riskkernel_at_the_registered_severity(
    tmp_path: Path,
) -> None:
    """The row names its own component, the registered severity, and schema 2.

    SPEC S5: `KillSwitch` books its own `riskkernel`-stamped events, forging no
    other component's stamp, and the position-hold invariant means no string it
    ledgers names a sell/close/submit/dump action. The schema version is pinned
    to 2 because a v1 row carries no delivery keys at all: stamping them apart
    keeps "written before delivery was recorded" distinguishable from "written
    after, and nothing was delivered".
    """
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    switch = KillSwitch(
        ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE),
        PersistingKernelLedgerWriter(store),
        _dispatcher_over(_webhook(failing=True)),
        clock=lambda: _FIXED_EPOCH_S,
    )

    switch.kill(KillTrigger.CLI)

    store.verify_chain()
    rows = [
        record for record in store.read_all() if record.event_type == "AlertEmitted"
    ]
    assert len(rows) == 1
    assert rows[0].payload_schema_version == 2
    assert rows[0].component == "riskkernel"
    payload = _alert_payload(events_from_records(store.read_all()))
    assert payload["severity"] == AlertSeverity.CRITICAL.value
    body = str(payload["message"])
    assert not any(
        token in body.lower() for token in ("sell", "close", "submit", "dump")
    )
