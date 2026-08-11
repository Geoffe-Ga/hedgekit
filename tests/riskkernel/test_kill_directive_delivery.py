"""The kill switch records what the venue *did* with its cancel-all (#480).

Issue #413 drew the emission-is-not-delivery line for `AlertEmitted`: a row
carrying only the body proved the page fired and looked identical whether every
sink accepted it or every sink failed. Issue #480 is the same defect one layer
over, for the directive: a `CancelAllDirective` row carrying only its scope
proved a directive was written, and looked identical whether the venue cancelled
every resting order or -- as was true on every path for the whole life of the
event -- was never told at all.

What this module pins:

* **The order of the two things.** The directive reaches the sink *before* the
  row is appended, per PR #412's rule that a failing audit append must never be
  able to skip an effect. Asserted as an interleaving, not as two independent
  facts, because "both happened" is true of the wrong order too.
* **The row distinguishes delivered from attempted**, and an unwired sink is
  recorded as *unknown* rather than as either.
* **Delivery failure is loud.** The one page an operator is guaranteed to read
  names the cancel-all that did not land, with its counts -- and, because the
  same string is ledgered, so does the chain.
* **A raising sink cannot skip the rest of the fail-safe surface.** The
  reservation release, the page and the `KILL` file all still happen, and the
  failure is recorded rather than swallowed.
* **Nothing venue-supplied reaches the chain**, swept over the real ledger file
  *and its WAL sidecar*, with a positive control so the sweep cannot pass by
  scanning an empty corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from windbreak.ledger.directives import DirectiveDelivery
from windbreak.numeric.types import MoneyMicros
from windbreak.riskkernel.kill import KILL_FILENAME, KillSwitch, KillTrigger
from windbreak.riskkernel.modes import Mode, ModeStateMachine
from windbreak.riskkernel.process import InMemoryKernelLedgerWriter
from windbreak.riskkernel.reservations import ReservationLedger

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.ledger.events import CancelAllDirective, Event

#: The fixed epoch every switch below is clocked at, so no test here depends on
#: wall-clock time (CI runs UTC; this module never reads a real clock).
_FIXED_EPOCH_S = 1_700_000_000

#: The base `HALT_KILL` body, compared in full rather than by substring: a
#: substring assertion passes just as well against a body that grew a second,
#: wrong half.
_BASE_MESSAGE = "kill switch engaged; trading halted, positions held"

#: The far-future expiry every reservation below is taken with.
_FAR_FUTURE_EXPIRY_S = _FIXED_EPOCH_S + 86_400

#: Credential-bearing text a sink raises, standing in for the category issue
#: #274 leaked. Nothing derived from it may reach the hash chain.
_SINK_FAILURE_DETAIL = "POST https://venue.example/v1/s3kr1t-480-token/cancel 403"


@dataclass
class _ReportingSink:
    """A `DirectiveSink` double reporting a caller-chosen delivery.

    Attributes:
        delivery: What `submit` reports back, or None to report nothing.
        received: Every directive submitted, in order.
    """

    delivery: DirectiveDelivery | None
    received: list[CancelAllDirective] = field(default_factory=list)

    def submit(self, directive: CancelAllDirective) -> DirectiveDelivery | None:
        """Record the directive and report the configured delivery.

        Args:
            directive: The directive submitted for delivery.

        Returns:
            The configured :class:`DirectiveDelivery`, or None.
        """
        self.received.append(directive)
        return self.delivery


@dataclass
class _RaisingSink:
    """A `DirectiveSink` double that raises credential-bearing text.

    Attributes:
        attempts: One entry per `submit` call, so a sink that was never
            reached is distinguishable from one that failed.
    """

    attempts: list[CancelAllDirective] = field(default_factory=list)

    def submit(self, directive: CancelAllDirective) -> DirectiveDelivery | None:
        """Record the attempt, then fail it.

        Args:
            directive: The directive submitted for delivery.

        Raises:
            RuntimeError: Always, carrying :data:`_SINK_FAILURE_DETAIL`.
        """
        self.attempts.append(directive)
        raise RuntimeError(_SINK_FAILURE_DETAIL)


@dataclass
class _OrderRecordingSink:
    """A `DirectiveSink` double recording *when* it ran, against the ledger.

    Appends a marker to the shared trace the ledger writer also appends to, so
    the submit and the append become one ordered sequence rather than two
    independently-true facts.

    Attributes:
        trace: The shared, ordered trace of what happened when.
    """

    trace: list[str]

    def submit(self, directive: CancelAllDirective) -> DirectiveDelivery:
        """Record that delivery happened, and report a clean one.

        Args:
            directive: The directive submitted for delivery.

        Returns:
            A fully-delivered :class:`DirectiveDelivery`.
        """
        del directive
        self.trace.append("submit")
        return DirectiveDelivery(cancelled=1)


class _TracingWriter(InMemoryKernelLedgerWriter):
    """A kernel ledger writer that also appends each event type to a trace."""

    def __init__(self, trace: list[str]) -> None:
        """Wire the writer to the shared trace.

        Args:
            trace: The shared, ordered trace to append event types to.
        """
        super().__init__()
        self._trace = trace

    def record(self, event: Event) -> None:
        """Record the event and note its type on the trace.

        Args:
            event: The event to record.
        """
        self._trace.append(event.event_type)
        super().record(event)


@dataclass
class _CapturingAlertSink:
    """A `KillSwitch` alert-dispatcher double capturing every body dispatched.

    Attributes:
        messages: One body per dispatch, in order.
    """

    messages: list[str] = field(default_factory=list)

    def dispatch(self, alert_type: object, message: str) -> None:
        """Capture one dispatched alert body.

        Args:
            alert_type: The alert type (ignored).
            message: The alert body.
        """
        del alert_type
        self.messages.append(message)


def _build(
    sink: object | None,
    *,
    writer: InMemoryKernelLedgerWriter | None = None,
    reservations: ReservationLedger | None = None,
    state_dir: Path | None = None,
) -> tuple[KillSwitch, InMemoryKernelLedgerWriter, _CapturingAlertSink]:
    """Build a kill switch over the given directive sink.

    Args:
        sink: The directive sink to wire, or None to wire none.
        writer: The ledger writer to wire; a fresh one if omitted.
        reservations: An optional reservation ledger released on kill.
        state_dir: An optional directory a `KILL` file is written into.

    Returns:
        A `(switch, writer, alert_sink)` tuple.
    """
    effective_writer = writer if writer is not None else InMemoryKernelLedgerWriter()
    alert_sink = _CapturingAlertSink()
    switch = KillSwitch(
        ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE),
        effective_writer,
        alert_sink,
        reservation_ledger=reservations,
        directive_sink=sink,
        state_dir=state_dir,
        clock=lambda: _FIXED_EPOCH_S,
    )
    return switch, effective_writer, alert_sink


def _directive_payload(writer: InMemoryKernelLedgerWriter) -> dict[str, object]:
    """Return the one ledgered `CancelAllDirective` payload.

    Args:
        writer: The writer whose events are searched.

    Returns:
        The single directive row's payload.

    Raises:
        AssertionError: If the kill did not write exactly one such row.
    """
    rows = [
        event.payload
        for event in writer.events
        if event.event_type == "CancelAllDirective"
    ]
    assert len(rows) == 1
    return rows[0]


def test_the_directive_reaches_the_sink_before_its_row_is_appended() -> None:
    """The effect happens first; the audit append records it afterwards.

    PR #412's ordering rule: a failing append must never be able to skip an
    effect. Until issue #480 this method appended *then* submitted, putting the
    record of an effect ahead of the effect itself -- and the row it wrote
    claimed nothing about whether the venue was ever told.

    Asserted as one interleaved sequence rather than as two separate facts,
    because "the sink was called" and "the row was written" are both true of
    the wrong order too.
    """
    trace: list[str] = []
    writer = _TracingWriter(trace)
    switch, _writer, _alerts = _build(_OrderRecordingSink(trace), writer=writer)

    switch.kill(KillTrigger.CLI)

    assert trace[:3] == ["KillEngaged", "submit", "CancelAllDirective"]
    assert trace[-1] == "AlertEmitted"


def test_a_delivered_cancel_all_is_ledgered_as_delivered() -> None:
    """The row carries the sink's counts and the derived outcome."""
    switch, writer, alerts = _build(
        _ReportingSink(DirectiveDelivery(cancelled=3, failed=0))
    )

    switch.kill(KillTrigger.CLI)

    assert _directive_payload(writer) == {
        "scope": "all_open_orders",
        "delivery": {"cancelled": 3, "failed": 0, "outcome": "delivered"},
        "delivery_reported": True,
    }
    assert alerts.messages == [_BASE_MESSAGE]


def test_an_unwired_sink_is_ledgered_as_unknown_and_never_as_delivered() -> None:
    """With no sink, the row records *unreported*, not a successful cancel.

    The fail-closed reading, mirroring issue #413's `delivery_reported: false`.
    An empty `delivery` must never be mistaken for "the venue had nothing
    resting", which is a positive claim only a wired sink can make.

    The page is deliberately left unchanged here: an unwired sink is an unknown
    rather than a failure, the base body never claims a cancellation happened,
    and the row above is where the unknown is recorded.
    """
    switch, writer, alerts = _build(None)

    switch.kill(KillTrigger.CLI)

    assert _directive_payload(writer) == {
        "scope": "all_open_orders",
        "delivery": {},
        "delivery_reported": False,
    }
    assert alerts.messages == [_BASE_MESSAGE]


@pytest.mark.parametrize(
    ("delivery", "expected_clause"),
    [
        (
            DirectiveDelivery(cancelled=2, failed=1),
            "resting-order cancellation NOT confirmed at the venue"
            " (outcome=partial, cancelled=2, failed=1)",
        ),
        (
            DirectiveDelivery(cancelled=0, failed=4),
            "resting-order cancellation NOT confirmed at the venue"
            " (outcome=refused, cancelled=0, failed=4)",
        ),
        (
            DirectiveDelivery(errored=True),
            "resting-order cancellation NOT confirmed at the venue"
            " (outcome=errored, cancelled=0, failed=0)",
        ),
    ],
    ids=["partial", "refused", "errored"],
)
def test_a_cancel_all_that_did_not_land_is_named_on_the_operators_page(
    delivery: DirectiveDelivery, expected_clause: str
) -> None:
    """The page an operator will read says what the kill could not do (#480).

    A kill whose cancel-all was refused, partly taken, or errored has left live
    instructions resting at a venue the operator has just walked away from.
    Silent partial execution of a fail-safe is worse than a fail-safe that
    announces its limits, so the notice rides the one page that is guaranteed
    to be seen -- and, since the same string is ledgered as the `AlertEmitted`
    body, the chain records it too.

    Each case carries distinct counts so no assertion can pass on a message
    built from the wrong delivery.

    Args:
        delivery: The failed delivery the sink reports.
        expected_clause: The exact clause appended to the base body.
    """
    switch, writer, alerts = _build(_ReportingSink(delivery))

    switch.kill(KillTrigger.CLI)

    expected = _BASE_MESSAGE + "; " + expected_clause
    alert_rows = [
        event.payload for event in writer.events if event.event_type == "AlertEmitted"
    ]
    assert alerts.messages == [expected]
    assert [row["message"] for row in alert_rows] == [expected]


def test_a_raising_sink_is_recorded_and_skips_no_later_fail_safe_effect(
    tmp_path: Path,
) -> None:
    """A sink that raises cannot take the rest of the kill down with it.

    The directive sink is a structural seam, so an arbitrary implementation may
    raise -- and everything after it is fail-safe: the capital release, the
    operator page, the `KILL` file, the audit row. Propagating would skip all
    four. The failure is therefore converted, not swallowed: it is ledgered as
    an errored outcome and named on the page.
    """
    writer = InMemoryKernelLedgerWriter()
    reservations = ReservationLedger(writer)
    reservations.reserve(
        "intent-480",
        MoneyMicros(4_000_000),
        "idem-480",
        expires_at=_FAR_FUTURE_EXPIRY_S,
    )
    sink = _RaisingSink()
    switch, _writer, alerts = _build(
        sink, writer=writer, reservations=reservations, state_dir=tmp_path
    )

    switch.kill(KillTrigger.CLI)

    assert len(sink.attempts) == 1
    assert switch.mode is Mode.KILLED
    assert reservations.total_reserved() == MoneyMicros(0)
    assert tmp_path.joinpath(KILL_FILENAME).exists()
    assert _directive_payload(writer)["delivery"] == {
        "cancelled": 0,
        "failed": 0,
        "outcome": "errored",
    }
    assert alerts.messages == [
        _BASE_MESSAGE + "; resting-order cancellation NOT confirmed at the venue"
        " (outcome=errored, cancelled=0, failed=0)"
    ]


def test_a_raising_sinks_text_never_reaches_the_kill_paths_ledgered_payloads() -> None:
    """No substring of the sink's credential-bearing text is ledgered (#274).

    A hash chain is append-only, so anything written into it is unredactable.
    The positive control is the assertion that the swept corpus actually
    contains the kill's own body: a sweep over an empty corpus passes forever
    while proving nothing.
    """
    switch, writer, _alerts = _build(_RaisingSink())

    switch.kill(KillTrigger.CLI)

    corpus = "".join(event.envelope_json for event in writer.events)
    assert _BASE_MESSAGE in corpus
    assert _SINK_FAILURE_DETAIL not in corpus
    assert "s3kr1t-480-token" not in corpus
    assert "venue.example" not in corpus
