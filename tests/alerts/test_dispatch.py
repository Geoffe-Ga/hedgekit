"""Tests for windbreak.alerts.dispatch (issues #14, #186): the alert dispatcher.

`AlertDispatcher` fans a single alert out to every configured sink,
isolating failures so a broken sink never takes down another sink, the
caller, or the ledger. These tests pin:

- the happy path (all sinks ok, ledger recorded once);
- partial failure (isolation, no premature fallback);
- total failure (fallback fires, is itself recorded);
- the empty-sink-list edge case (zero successes, same as total failure);
- ledger-writer failure (logged, never re-raised).

`AlertDispatcher`/`AlertEmitted`/`SinkOutcome`/`LoggingLedgerWriter` are
re-exported from `windbreak.alerts`; none of them exist yet, so importing
this module fails at collection with `ModuleNotFoundError` -- the expected
RED state for issue #14's Gate 1.

Issue #186 adds `dispatch_hook(dispatcher, alert_type)`: a factory binding
`windbreak.evaluation.crosscheck`'s `AlertHook` seam
(`(severity, message) -> None`) to a real `AlertDispatcher`, whose
`dispatch(alert_type, message)` derives severity from the registry rather
than trusting the caller's `severity` argument. `dispatch_hook` does not
exist yet, so the tests pinning it below import it locally (inside each
test body) rather than at module scope, so they fail on their own
`ImportError` without breaking collection of the rest of this
already-passing suite -- the expected RED state for issue #186's Gate 1.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from windbreak.alerts import (
    AlertDispatcher,
    AlertEmitted,
    LoggingLedgerWriter,
    SinkOutcome,
)
from windbreak.alerts.delivery import DeliveryOutcome
from windbreak.alerts.registry import AlertSeverity, AlertType, get_registration
from windbreak.alerts.sinks import LogOnlySink

if TYPE_CHECKING:
    from windbreak.evaluation.crosscheck import AlertHook

_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


@dataclass
class _SucceedingSink:
    """A fake `AlertSink` that always succeeds and records its calls."""

    name: str
    calls: list[tuple[AlertType, AlertSeverity, str]] = field(default_factory=list)

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Record the call without raising."""
        self.calls.append((alert_type, severity, message))


@dataclass
class _FailingSink:
    """A fake `AlertSink` that always raises after recording its calls."""

    name: str
    calls: list[tuple[AlertType, AlertSeverity, str]] = field(default_factory=list)

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Record the call, then raise to simulate a broken channel."""
        self.calls.append((alert_type, severity, message))
        raise RuntimeError(f"{self.name} send failed")


class _SpyLedgerWriter:
    """A fake `LedgerWriter` that records every event it is given."""

    def __init__(self) -> None:
        self.recorded: list[AlertEmitted] = []

    def record(self, event: AlertEmitted) -> None:
        """Record the event without raising."""
        self.recorded.append(event)


class _RaisingLedgerWriter:
    """A fake `LedgerWriter` that always raises, simulating a broken ledger."""

    def record(self, event: AlertEmitted) -> None:
        """Raise unconditionally."""
        raise RuntimeError("ledger unavailable")


@pytest.mark.parametrize("alert_type", [AlertType.MODE_CHANGE, AlertType.HALT_KILL])
def test_dispatch_happy_path_records_ok_outcomes_and_ledger_event(
    alert_type: AlertType,
) -> None:
    """Two healthy sinks both succeed; the ledger writer records once."""
    sink_a = _SucceedingSink("a")
    sink_b = _SucceedingSink("b")
    ledger = _SpyLedgerWriter()
    dispatcher = AlertDispatcher([sink_a, sink_b], ledger_writer=ledger)

    event = dispatcher.dispatch(alert_type, "hello")

    assert event.alert_type == alert_type
    assert event.severity == get_registration(alert_type).severity
    assert event.message == "hello"
    assert event.outcomes == (
        SinkOutcome(sink="a", outcome=DeliveryOutcome.DELIVERED),
        SinkOutcome(sink="b", outcome=DeliveryOutcome.DELIVERED),
    )
    assert ledger.recorded == [event]


def test_dispatch_partial_failure_is_isolated_and_does_not_fire_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One failing sink does not raise, and does not trigger the fallback."""
    caplog.set_level(logging.DEBUG)
    sink_a = _FailingSink("a")
    sink_b = _SucceedingSink("b")
    ledger = _SpyLedgerWriter()
    fallback = _SucceedingSink("fallback")
    dispatcher = AlertDispatcher(
        [sink_a, sink_b], ledger_writer=ledger, fallback=fallback
    )

    event = dispatcher.dispatch(AlertType.VETO, "vetoed")

    assert [outcome.ok for outcome in event.outcomes] == [False, True]
    assert fallback.calls == []
    assert any("a" in record.getMessage() for record in caplog.records)


def test_dispatch_all_sinks_failing_fires_the_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When every sink fails, the fallback fires and is itself recorded."""
    caplog.set_level(logging.DEBUG)
    sink_a = _FailingSink("a")
    sink_b = _FailingSink("b")
    ledger = _SpyLedgerWriter()
    fallback = _SucceedingSink("fallback")
    dispatcher = AlertDispatcher(
        [sink_a, sink_b], ledger_writer=ledger, fallback=fallback
    )

    event = dispatcher.dispatch(AlertType.DISK_HALT, "disk full")

    assert [outcome.ok for outcome in event.outcomes] == [False, False, True]
    assert event.outcomes[-1].sink == "fallback"
    assert fallback.calls == [
        (
            AlertType.DISK_HALT,
            get_registration(AlertType.DISK_HALT).severity,
            "disk full",
        )
    ]
    assert ledger.recorded[0].outcomes[-1] == event.outcomes[-1]


def test_dispatch_empty_sink_list_fires_the_fallback() -> None:
    """Zero configured sinks means zero successes, so the fallback fires."""
    ledger = _SpyLedgerWriter()
    fallback = _SucceedingSink("fallback")
    dispatcher: AlertDispatcher = AlertDispatcher(
        [], ledger_writer=ledger, fallback=fallback
    )

    event = dispatcher.dispatch(AlertType.SCHEMA_ANOMALY, "schema drift")

    assert event.outcomes == (
        SinkOutcome(sink="fallback", outcome=DeliveryOutcome.DELIVERED, fallback=True),
    )
    assert fallback.calls == [
        (
            AlertType.SCHEMA_ANOMALY,
            get_registration(AlertType.SCHEMA_ANOMALY).severity,
            "schema drift",
        )
    ]


def test_dispatch_default_fallback_is_a_log_only_sink(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without an explicit fallback, the default fallback is a `LogOnlySink`."""
    caplog.set_level(logging.DEBUG)
    dispatcher: AlertDispatcher = AlertDispatcher([], ledger_writer=_SpyLedgerWriter())

    event = dispatcher.dispatch(AlertType.VETO, "vetoed")

    assert event.outcomes[-1].sink == LogOnlySink().name
    assert any("vetoed" in record.getMessage() for record in caplog.records)


def test_dispatch_ledger_writer_raising_does_not_propagate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken ledger writer is logged, never raised into the caller."""
    caplog.set_level(logging.DEBUG)
    sink = _SucceedingSink("a")
    dispatcher = AlertDispatcher([sink], ledger_writer=_RaisingLedgerWriter())

    event = dispatcher.dispatch(AlertType.VETO, "vetoed")

    assert event.outcomes == (SinkOutcome(sink="a", outcome=DeliveryOutcome.DELIVERED),)
    assert any("ledger" in record.getMessage().lower() for record in caplog.records)


def test_alert_emitted_ts_is_iso_utc_and_outcomes_is_a_tuple() -> None:
    """`AlertEmitted.ts` is ISO-UTC and `.outcomes` is an immutable tuple."""
    sink = _SucceedingSink("a")
    dispatcher = AlertDispatcher([sink], ledger_writer=_SpyLedgerWriter())

    event = dispatcher.dispatch(AlertType.VETO, "vetoed")

    assert _ISO_UTC.match(event.ts)
    assert isinstance(event.outcomes, tuple)


def test_logging_ledger_writer_records_event_as_a_structured_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`LoggingLedgerWriter.record` logs the event's type/severity/message."""
    caplog.set_level(logging.INFO)
    writer = LoggingLedgerWriter()
    event = AlertEmitted(
        alert_type=AlertType.VETO,
        severity=AlertSeverity.WARNING,
        message="vetoed",
        outcomes=(SinkOutcome(sink="a", outcome=DeliveryOutcome.DELIVERED),),
        ts="2026-01-01T00:00:00.000000Z",
    )

    writer.record(event)

    assert len(caplog.records) == 1
    text = caplog.records[0].getMessage()
    assert AlertType.VETO.value in text
    assert "vetoed" in text


def test_sink_outcome_detail_defaults_to_none() -> None:
    """`SinkOutcome.detail` is optional and defaults to `None`."""
    outcome = SinkOutcome(sink="a", outcome=DeliveryOutcome.DELIVERED)

    assert outcome.detail is None


def test_sink_outcome_is_frozen() -> None:
    """`SinkOutcome` instances cannot be mutated after construction."""
    outcome = SinkOutcome(sink="a", outcome=DeliveryOutcome.DELIVERED)

    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.sink = "b"  # type: ignore[misc]


def test_alert_emitted_is_frozen() -> None:
    """`AlertEmitted` instances cannot be mutated after construction."""
    event = AlertEmitted(
        alert_type=AlertType.VETO,
        severity=AlertSeverity.WARNING,
        message="x",
        outcomes=(),
        ts="2026-01-01T00:00:00.000000Z",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.message = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# dispatch_hook (issue #186): binding the crosscheck's AlertHook seam to a
# real AlertDispatcher.
# ---------------------------------------------------------------------------


def test_dispatch_hook_dispatches_through_the_wrapped_dispatcher(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Calling the returned hook delivers exactly one alert via the ledger.

    The `AlertEmitted` the ledger receives carries the bound `alert_type`,
    the registry-derived `AlertSeverity.CRITICAL` severity, and the message
    verbatim. Because the caller-supplied severity agrees with the
    registration, no severity-disagreement WARNING is logged.
    """
    from windbreak.alerts import dispatch_hook

    caplog.set_level(logging.WARNING, logger="windbreak.alerts")
    sink = _SucceedingSink("a")
    ledger = _SpyLedgerWriter()
    dispatcher = AlertDispatcher([sink], ledger_writer=ledger)

    hook = dispatch_hook(dispatcher, AlertType.GATE_COMPUTATION_MISMATCH)
    hook(AlertSeverity.CRITICAL, "some message")

    assert len(ledger.recorded) == 1
    event = ledger.recorded[0]
    assert event.alert_type == AlertType.GATE_COMPUTATION_MISMATCH
    assert event.severity == AlertSeverity.CRITICAL
    assert event.message == "some message"
    assert [
        record
        for record in caplog.records
        if record.name == "windbreak.alerts" and record.levelno == logging.WARNING
    ] == []


def test_dispatch_hook_disagreeing_severity_warns_but_still_dispatches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hook-supplied severity that disagrees with the registry never wins.

    Calling the hook with `AlertSeverity.INFO` against
    `GATE_COMPUTATION_MISMATCH` (registered CRITICAL) logs a WARNING on the
    `windbreak.alerts` logger, never raises, and still dispatches -- with the
    registry-derived CRITICAL severity, since `AlertDispatcher.dispatch`
    always derives severity itself.
    """
    from windbreak.alerts import dispatch_hook

    caplog.set_level(logging.WARNING, logger="windbreak.alerts")
    sink = _SucceedingSink("a")
    ledger = _SpyLedgerWriter()
    dispatcher = AlertDispatcher([sink], ledger_writer=ledger)
    hook = dispatch_hook(dispatcher, AlertType.GATE_COMPUTATION_MISMATCH)

    hook(AlertSeverity.INFO, "some message")

    warnings = [
        record
        for record in caplog.records
        if record.name == "windbreak.alerts" and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert len(ledger.recorded) == 1
    assert ledger.recorded[0].severity == AlertSeverity.CRITICAL


def test_dispatch_hook_satisfies_the_crosscheck_alert_hook_protocol() -> None:
    """A `dispatch_hook` closure structurally satisfies `crosscheck.AlertHook`.

    Mirrors `tests/forecast/test_canary.py`'s structural-satisfaction pattern:
    a typed assignment pins that mypy accepts the closure as an `AlertHook`
    without the alerts package importing `windbreak.evaluation`.
    """
    from windbreak.alerts import dispatch_hook

    sink = _SucceedingSink("a")
    ledger = _SpyLedgerWriter()
    dispatcher = AlertDispatcher([sink], ledger_writer=ledger)

    hook: AlertHook = dispatch_hook(dispatcher, AlertType.GATE_COMPUTATION_MISMATCH)

    assert callable(hook)
    hook(AlertSeverity.CRITICAL, "structural check")
    assert len(ledger.recorded) == 1
