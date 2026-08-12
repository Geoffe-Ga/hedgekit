"""Resolution ingestion: how a settled outcome enters the system (issue #439).

Before this module nothing in the running system ever learned that a market
resolved. :mod:`windbreak.evaluation.resolution` could *parse* resolutions, but
its only loader read a known-answer test fixture, so the always-on loop's one
evaluation consumer --
:func:`windbreak.scheduler.weekly_data.weekly_report_body` -- folded against a
hardcoded ``resolutions={}`` and every metric it printed read ``UNDEFINED``
forever. Those figures were not "not enough data yet": they were structurally
unreachable, so the paper track record could neither pass nor fail a promotion
gate.

This module is the missing seam, and it is deliberately the *smallest* one that
closes the loop: a :class:`MarketResolved` ledger event, an operator-driven
``windbreak ingest-resolution`` CLI verb that appends one, and the fold
(:func:`ingested_resolutions_from_records`) that reads them back. There is no
new tick stage and no venue-polling settlement feed -- polling a venue's
settlement surface is a much larger change tracked separately -- because the
hash-chained ledger the loop already reads on every tick is a sufficient
transport: an ingested row is picked up by the very next weekly fold with no
scheduler change at all.

WHY THE INSTANT, AND NOT JUST THE OUTCOME

A resolution carries the exact instant the market settled, not merely its
answer. Temporal integrity (SPEC S1.1-6) is the point of the evaluation
harness: a forecast may only be scored against a resolution that *postdates*
it, because a forecast made after the answer was knowable is not a prediction.
The temporal gate itself is sequence-based by design (no wall clock on the
value path), so the settlement instant is projected onto the ledger's sequence
axis by
:func:`windbreak.evaluation.temporal.resolution_sequences_from_instants` before
it gates anything. Keying the gate on the *ingesting row's* own position
instead would be the exact failure this module exists to prevent: an operator
who ingests a week late would silently score a week of forecasts that already
knew the answer.

Every refusal here is loud. A naive ``resolved_at``, an unknown outcome token,
a missing payload key, a blank provenance label, and a second *contradicting*
resolution for one market all raise :class:`ValueError` naming the offending
field, rather than defaulting to something that reads healthy.

WHAT COUNTS AS A CONTRADICTION

Only the *evidentiary* fields do: ``market_ticker``, ``outcome`` and
``resolved_at`` (see :func:`resolutions_conflict`). Those three are what the
harness adjudicates against -- the outcome is the ground truth a forecast is
scored on, and the instant is the coordinate
:func:`windbreak.evaluation.temporal.resolution_sequences_from_instants`
projects onto the sequence axis to decide which forecasts could have peeked.
``source`` is deliberately excluded: it is a free-text audit label naming where
an operator read the settlement, so ``"kalshi settlement notice"`` and
``"kalshi-settlement-notice"`` are two spellings of one provenance, never two
claims about what the market did. Comparing whole frozen records -- which is
what this module shipped first -- reported that retyping to a differing label
as a contradiction *about the settled outcome*, which was both false and, at the
fold, unrecoverable. Both rows stay on the append-only ledger and stay
auditable; the fold's projection carries the first row's label.

``resolved_at`` is compared at full microsecond precision, offset-normalized
(so ``12:00:00+00:00`` and ``07:00:00-05:00`` are the same instant, as
:class:`~datetime.datetime` equality already has it). It is not rounded to the
second, because two instants inside one second can project onto *different*
ledger positions and therefore admit or refuse different forecasts: rounding
would silently merge two claims that gate differently. Exactness costs a
mistyping operator one retyped command, because the ``ingest-resolution`` verb
refuses a conflicting call *before* it writes anything (issue #439 review) --
it never costs them a stalled loop.

CORRECTING A WRONG FIRST INGEST (ISSUE #484)

The contradiction guard closed half of the failure mode: a *conflicting*
re-ingest is refused before it is written, so a mistyped second call can no
longer stop the loop. The other half stayed open, because nothing contradicts a
**first** ingest that was simply wrong. Nothing refused it, the ledger is
append-only, and so the bad settlement -- and every metric folded from it --
stood forever.

:class:`SettlementReversed` is the correction, and it is deliberately *not* a
redaction: the wrong row stays on the chain, and a later row supersedes it. Two
rows then describe one market, so the precedence rule is explicit rather than
conventional (PR #500's standard):

    A ``SettlementReversed`` row supersedes exactly the row whose
    ``sequence_number`` it names, and only when that row is the one currently
    carrying the market's claim. It then wins. A later ``MarketResolved`` row
    names nothing and therefore supersedes nothing -- it is either an
    idempotent restatement or a contradiction, and the contradiction is
    refused, exactly as before.

So "later wins" is never a bare append-order convention here: a row only
supersedes what it explicitly claims to supersede. :func:`_fold_correction_row`
is the single place that rule lives, and
``tests/evaluation/test_resolution_correction.py`` asserts it in both
directions.

The event is named after
:class:`~windbreak.evaluation.resolution.SettlementEventType`'s
``SETTLEMENT_REVERSED`` member on purpose. That member already existed, but only
in the *fixture* settlement vocabulary -- a parallel path nothing on the ledger
emitted or consumed, which is exactly why ``ingest.py``'s old error message
could name a real symbol that no operator could ever act on (issue #484's
correcting comment). Reusing the word keeps one meaning of "reversal" across
both paths, and
``test_a_ledger_correction_means_what_the_fixture_vocabulary_means`` proves the
agreement through the independent :class:`ResolutionTracker` state machine
rather than asserting it in prose.

Where the fixture vocabulary spends two events -- a reversal that clears, then
a corrected settlement -- the ledger spends one row carrying both halves. The
always-on loop folds the whole ledger on *every* tick, so a market observable
in a cleared-but-not-yet-re-settled state would silently drop out of scoring
for as long as the operator took to type a second command, and a crash between
the two would drop it forever. One row is atomic against a reader that never
stops reading.

Every refusal is at the verb (:func:`windbreak.main._run_correct_resolution`),
never only at the fold, and a market can be corrected again and again -- the
next correction names the correction row. A mechanism that works exactly once
would be the same permanent trap this module exists to remove, reached one
command later.

THE CHECK BELONGS AT THE VERB, NOT ONLY AT THE FOLD

The ledger is append-only: a contradicting row can never be un-written, and
this fold runs inside ``weekly_report_body`` on every tick. A raise here is
therefore *terminal* -- which is why
:func:`windbreak.main._run_ingest_resolution` reads the existing resolutions
back through this very function and refuses a conflicting append at ingest
time, exit 1 and nothing written. The guard below remains as the last line of
defense for a ledger this verb did not write, and its message no longer offers
a remedy that does not exist.

The typed event derives its base :class:`~windbreak.ledger.events.Event` fields
through a LOCAL ``_derive_typed_event`` -- the house pattern from
:mod:`windbreak.evaluation.preregistration`,
:mod:`windbreak.evaluation.crosscheck` and
:mod:`windbreak.evaluation.live_divergence` -- so this issue never touches the
ledger's central ``EVENT_TYPES`` map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from windbreak.evaluation.resolution import ResolutionOutcome
from windbreak.ledger.events import Event
from windbreak.timekeeping import iso_z, require_aware

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any

    from windbreak.ledger.store import LedgerRecord

#: The immutable event-type token every ingested resolution is ledgered under.
#: Equal to :class:`MarketResolved`'s class name by construction, since the
#: local derivation stamps ``type(event).__name__``.
MARKET_RESOLVED_EVENT_TYPE = "MarketResolved"

#: Payload schema version stamped on this module's event. Replicated locally
#: (rather than imported from :mod:`windbreak.ledger.events`'s private copy) so
#: a payload-shape change here can be versioned independently.
_SCHEMA_VERSION = 1

#: Envelope key under which a ledgered event's typed payload is nested.
_PAYLOAD_DATA_KEY = "data"

#: Payload key naming the market that settled.
_MARKET_TICKER_KEY = "market_ticker"
#: Payload key carrying the settled outcome token (``"yes"`` / ``"no"``).
_OUTCOME_KEY = "outcome"
#: Payload key carrying the ISO-8601 instant the market settled.
_RESOLVED_AT_KEY = "resolved_at"
#: Payload key carrying where the operator read the settlement.
_SOURCE_KEY = "source"

#: Payload key naming the ledger row a correction supersedes.
_SUPERSEDED_SEQUENCE_KEY = "superseded_sequence_number"

#: The immutable event-type token every correction is ledgered under. Named
#: after :class:`~windbreak.evaluation.resolution.SettlementEventType`'s
#: ``SETTLEMENT_REVERSED`` member deliberately -- see this module's docstring
#: on why the ledger path reuses the fixture path's word rather than coining a
#: second one. Equal to :class:`SettlementReversed`'s class name by
#: construction, since the local derivation stamps ``type(event).__name__``.
SETTLEMENT_REVERSED_EVENT_TYPE = "SettlementReversed"

#: The payload keys every ``MarketResolved`` row must carry, in the order a
#: malformed row is checked against so the first missing one is named.
_REQUIRED_PAYLOAD_KEYS = (
    _MARKET_TICKER_KEY,
    _OUTCOME_KEY,
    _RESOLVED_AT_KEY,
    _SOURCE_KEY,
)

#: The payload keys every ``SettlementReversed`` row must carry: the row it
#: supersedes, plus the same evidentiary triple and provenance label a
#: resolution carries, because a correction *is* a resolution claim -- one that
#: happens to name what it replaces.
_REQUIRED_CORRECTION_PAYLOAD_KEYS = (
    _SUPERSEDED_SEQUENCE_KEY,
    *_REQUIRED_PAYLOAD_KEYS,
)

#: The lowest position a real ledger row can occupy. ``sequence_number`` is
#: 1-based, so anything at or below zero names no row at all.
_FIRST_LEDGER_POSITION = 1


def _derive_typed_event(event: Event, payload: dict[str, object]) -> None:
    """Populate the derived :class:`~windbreak.ledger.events.Event` fields.

    Replicates the ledger module's private derivation locally (its
    ``EVENT_TYPES`` map is out of this issue's scope): sets ``event_type`` to
    the concrete class name, ``payload_schema_version`` to this module's schema
    version, and ``payload`` to the assembled dict, via ``object.__setattr__``
    because the event is frozen.

    Args:
        event: The freshly constructed typed event to populate.
        payload: The type-specific payload assembled by the subclass.
    """
    object.__setattr__(event, "event_type", type(event).__name__)
    object.__setattr__(event, "payload_schema_version", _SCHEMA_VERSION)
    object.__setattr__(event, "payload", payload)


@dataclass(frozen=True)
class MarketResolved(Event):
    """Records that a binary event market settled, and when.

    Attributes:
        market_ticker: The market that settled.
        outcome: The settled ground-truth outcome.
        resolved_at: The exact, timezone-aware instant the market settled --
            the temporal coordinate a forecast must predate to be scorable
            against this resolution.
        source: Where the operator read the settlement (the claim's
            provenance). Never blank: an outcome nothing attributes cannot be
            audited.
    """

    market_ticker: str
    outcome: ResolutionOutcome
    resolved_at: datetime
    source: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Validate the resolution, then assemble the payload and base fields.

        Raises:
            ValueError: If ``resolved_at`` carries no UTC offset (an instant
                whose meaning depends on the ingesting host is not evidence),
                if ``market_ticker`` is blank (it could never join to a
                forecast), or if ``source`` is blank (an unattributed outcome
                is unprovable).
        """
        require_aware(self.resolved_at, _RESOLVED_AT_KEY)
        if not self.market_ticker.strip():
            raise ValueError(
                "market_ticker must be a non-blank market identifier; a "
                "resolution naming no market can never join to a forecast"
            )
        if not self.source.strip():
            raise ValueError(
                "source must be a non-blank provenance label; an outcome "
                "nothing attributes is unprovable evidence and is refused "
                "rather than ingested"
            )
        payload: dict[str, object] = {
            _MARKET_TICKER_KEY: self.market_ticker,
            _OUTCOME_KEY: self.outcome.value,
            _RESOLVED_AT_KEY: iso_z(self.resolved_at),
            _SOURCE_KEY: self.source,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class SettlementReversed(Event):
    """Records that an earlier settlement was wrong, and what it should be.

    The correction mechanism for issue #484. The ledger is append-only, so a
    wrong :class:`MarketResolved` row can never be redacted -- this event does
    not try to. It is a *later row that supersedes an earlier one*, naming the
    exact position it replaces so no reader ever has to guess which of two
    claims about one market wins.

    Attributes:
        market_ticker: The market whose settlement is being corrected.
        superseded_sequence_number: The ledger position of the row this
            correction replaces -- the row currently carrying the market's
            claim, which is its first ingest or its most recent correction.
        outcome: The corrected ground-truth outcome.
        resolved_at: The corrected, timezone-aware settlement instant. A
            correction may move the instant as well as the answer, and the
            temporal gate re-adjudicates against the corrected one.
        source: Where the correction was read (its own provenance), never the
            superseded row's. Overturning a settled outcome is the one act on
            this ledger that most needs attribution.
    """

    market_ticker: str
    superseded_sequence_number: int
    outcome: ResolutionOutcome
    resolved_at: datetime
    source: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Validate the correction, then assemble the payload and base fields.

        Raises:
            TypeError: If ``superseded_sequence_number`` is a ``bool`` -- an
                ``int`` subclass that must not masquerade as row 1.
            ValueError: If ``resolved_at`` carries no UTC offset, if
                ``market_ticker`` is blank, if ``source`` is blank, or if
                ``superseded_sequence_number`` is not a positive ledger
                position.
        """
        require_aware(self.resolved_at, _RESOLVED_AT_KEY)
        self._validate_target()
        if not self.market_ticker.strip():
            raise ValueError(
                "market_ticker must be a non-blank market identifier; a "
                "correction naming no market can never join to a resolution"
            )
        if not self.source.strip():
            raise ValueError(
                "source must be a non-blank provenance label; a correction "
                "nothing attributes cannot be audited, and overturning a "
                "settled outcome is the one act on this ledger that most "
                "needs to be"
            )
        payload: dict[str, object] = {
            _MARKET_TICKER_KEY: self.market_ticker,
            _SUPERSEDED_SEQUENCE_KEY: self.superseded_sequence_number,
            _OUTCOME_KEY: self.outcome.value,
            _RESOLVED_AT_KEY: iso_z(self.resolved_at),
            _SOURCE_KEY: self.source,
        }
        _derive_typed_event(self, payload)

    def _validate_target(self) -> None:
        """Check that the superseded position could name a real ledger row.

        Raises:
            TypeError: If the position is a ``bool``, mirroring the
                ``SettlementEvent`` guard: ``True`` is an ``int`` subclass and
                would otherwise silently name row 1.
            ValueError: If the position is not positive.
        """
        target = self.superseded_sequence_number
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError(
                f"{_SUPERSEDED_SEQUENCE_KEY} requires a non-bool int, got "
                f"{type(target).__name__}"
            )
        if target < _FIRST_LEDGER_POSITION:
            raise ValueError(
                f"{_SUPERSEDED_SEQUENCE_KEY} must be a positive ledger "
                f"position, got {target}; ledger sequence numbers start at "
                f"{_FIRST_LEDGER_POSITION}, so this names no row at all"
            )


@dataclass(frozen=True, slots=True)
class IngestedResolution:
    """One market's ingested ground truth, read back off the ledger.

    Attributes:
        market_ticker: The market that settled.
        outcome: The settled ground-truth outcome.
        resolved_at: The timezone-aware instant the market settled.
        source: Where the operator read the settlement.
    """

    market_ticker: str
    outcome: ResolutionOutcome
    resolved_at: datetime
    source: str


def _evidentiary_claim(
    resolution: IngestedResolution,
) -> tuple[str, ResolutionOutcome, datetime]:
    """Project a resolution onto the fields that make it evidence.

    Args:
        resolution: The resolution to project.

    Returns:
        Its ``(market_ticker, outcome, resolved_at)`` triple. ``source`` is
        excluded by construction: see this module's docstring for why a
        differing provenance label is not a contradiction.
    """
    return (resolution.market_ticker, resolution.outcome, resolution.resolved_at)


def resolutions_conflict(
    existing: IngestedResolution, candidate: IngestedResolution
) -> bool:
    """Report whether two resolutions make incompatible claims.

    The single authority on what "contradicting" means, shared by the ledger
    fold and by the ``ingest-resolution`` verb's pre-append check, so the verb
    refuses exactly the appends the fold would later refuse to read.

    Args:
        existing: The resolution already on the ledger.
        candidate: The resolution being compared against it.

    Returns:
        ``True`` when the two disagree on ticker, outcome or settlement
        instant; ``False`` when they agree on all three, whatever their
        ``source`` labels say.
    """
    return _evidentiary_claim(existing) != _evidentiary_claim(candidate)


def describe_resolution_claim(resolution: IngestedResolution) -> str:
    """Render a resolution's evidentiary claim for an operator-facing message.

    Args:
        resolution: The resolution to describe.

    Returns:
        A ``outcome=... resolved_at=...`` fragment carrying exact values, so a
        refusal shows the operator what differs rather than only that
        something did.
    """
    return (
        f"{_OUTCOME_KEY}={resolution.outcome.value!r} "
        f"{_RESOLVED_AT_KEY}={iso_z(resolution.resolved_at)}"
    )


def ingested_resolution_of(
    event: MarketResolved | SettlementReversed,
) -> IngestedResolution:
    """Project a not-yet-appended event into the resolution the fold will read.

    Lets the ``ingest-resolution`` and ``correct-resolution`` verbs compare what
    they are about to write against what is already on the ledger using the same
    type, and therefore the same equality, the tick's fold uses. Both event
    types project through here because a correction *is* a resolution claim --
    one that additionally names the row it replaces -- so the two verbs cannot
    drift into two notions of what "the same claim" means. The projection is
    exact rather than approximate: :func:`windbreak.timekeeping.iso_z` renders
    microseconds and :meth:`datetime.datetime.fromisoformat` reads them back, so
    the round trip through the ledger returns an equal instant --
    ``test_a_round_tripped_event_folds_back_to_its_own_projection`` pins that.

    Args:
        event: The ``MarketResolved`` or ``SettlementReversed`` event about to
            be appended.

    Returns:
        The :class:`IngestedResolution` a fold of the appended row would yield.
    """
    return IngestedResolution(
        market_ticker=event.market_ticker,
        outcome=event.outcome,
        resolved_at=event.resolved_at,
        source=event.source,
    )


def _parsed_instant(text: str) -> datetime | None:
    """Parse an ISO-8601 instant, reporting failure as ``None``.

    Returning ``None`` rather than raising lets each caller phrase its own
    locatable message (the CLI names the flag, the ledger fold names the row)
    without either restating the other's wording.

    Args:
        text: The candidate ISO-8601 instant.

    Returns:
        The parsed datetime, or ``None`` when ``text`` is not ISO-8601. The
        result may still be naive; awareness is a separate check.
    """
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_resolution_instant(text: str) -> datetime:
    """Parse an operator-supplied settlement instant, failing closed.

    Args:
        text: The ISO-8601 instant as typed on the command line.

    Returns:
        The parsed, timezone-aware instant.

    Raises:
        ValueError: If ``text`` is not an ISO-8601 instant, or carries no UTC
            offset -- an offsetless instant would be read against whatever zone
            the operator's shell happens to carry, so it is refused rather than
            guessed.
    """
    moment = _parsed_instant(text)
    if moment is None:
        raise ValueError(
            f"{_RESOLVED_AT_KEY} is not an ISO-8601 instant: {text!r} "
            "(expected e.g. 2026-03-01T12:00:00+00:00)"
        )
    require_aware(moment, _RESOLVED_AT_KEY)
    return moment


def _require_payload_keys(
    data: Mapping[str, Any],
    sequence_number: int,
    *,
    label: str,
    keys: tuple[str, ...],
) -> None:
    """Check that a persisted payload carries every required key.

    Args:
        data: The row's typed payload.
        sequence_number: The row's ledger position, named in the error.
        label: The row's event type, named in the error so a malformed
            correction is not misreported as a malformed resolution.
        keys: The keys the row must carry, in the order they are checked so
            the first missing one is the one named.

    Raises:
        ValueError: If any required key is absent, naming the first one
            missing rather than raising a bare :class:`KeyError`.
    """
    for key in keys:
        if key not in data:
            raise ValueError(
                f"{label} payload at sequence_number={sequence_number} "
                f"is missing {key!r}"
            )


def _outcome_from_payload(
    token: object, sequence_number: int, *, label: str
) -> ResolutionOutcome:
    """Parse a persisted outcome token, failing closed on an unknown value.

    Args:
        token: The raw ``outcome`` value off the payload.
        sequence_number: The row's ledger position, named in the error.
        label: The row's event type, named in the error.

    Returns:
        The typed :class:`~windbreak.evaluation.resolution.ResolutionOutcome`.

    Raises:
        ValueError: If ``token`` is not one of the binary outcome tokens.
    """
    try:
        return ResolutionOutcome(token)
    except ValueError as exc:
        raise ValueError(
            f"{label} payload at sequence_number={sequence_number} "
            f"carries an unknown {_OUTCOME_KEY}: {token!r} (expected one of "
            f"{[member.value for member in ResolutionOutcome]})"
        ) from exc


def _instant_from_payload(
    token: object, sequence_number: int, *, label: str
) -> datetime:
    """Parse a persisted settlement instant, failing closed twice over.

    Args:
        token: The raw ``resolved_at`` value off the payload.
        sequence_number: The row's ledger position, named in the error.
        label: The row's event type, named in the error.

    Returns:
        The parsed, timezone-aware instant.

    Raises:
        ValueError: If ``token`` is not a string, is not an ISO-8601 instant,
            or carries no UTC offset.
    """
    moment = _parsed_instant(token) if isinstance(token, str) else None
    if moment is None:
        raise ValueError(
            f"{label} payload at sequence_number={sequence_number} "
            f"carries an unparseable {_RESOLVED_AT_KEY}: {token!r} "
            "(expected an ISO-8601 instant)"
        )
    require_aware(moment, _RESOLVED_AT_KEY)
    return moment


def _payload_of(record: LedgerRecord) -> Mapping[str, Any]:
    """Return one persisted row's typed payload, unwrapped from its envelope.

    Args:
        record: The ledger row to read.

    Returns:
        The row's ``data`` mapping.
    """
    envelope: dict[str, Any] = json.loads(record.payload_json)
    data: dict[str, Any] = envelope[_PAYLOAD_DATA_KEY]
    return data


def _claim_from_payload(
    data: Mapping[str, Any], sequence_number: int, *, label: str
) -> IngestedResolution:
    """Rebuild the evidentiary claim a persisted payload carries.

    Shared by the resolution and correction readers, because a correction *is*
    a resolution claim -- one that additionally names the row it replaces. One
    reader means the two row types can never drift into parsing the same four
    fields two different ways.

    Args:
        data: The row's typed payload.
        sequence_number: The row's ledger position, named in any error.
        label: The row's event type, named in any error.

    Returns:
        The typed claim the payload carries.

    Raises:
        ValueError: If the payload carries an unknown outcome token or an
            unparseable/naive instant.
    """
    return IngestedResolution(
        market_ticker=str(data[_MARKET_TICKER_KEY]),
        outcome=_outcome_from_payload(data[_OUTCOME_KEY], sequence_number, label=label),
        resolved_at=_instant_from_payload(
            data[_RESOLVED_AT_KEY], sequence_number, label=label
        ),
        source=str(data[_SOURCE_KEY]),
    )


def _resolution_from_record(record: LedgerRecord) -> IngestedResolution:
    """Rebuild one :class:`IngestedResolution` from a persisted row.

    Args:
        record: The ``MarketResolved`` ledger row to read.

    Returns:
        The typed resolution the row carries.

    Raises:
        ValueError: If the row's payload is missing a key, carries an unknown
            outcome token, or carries an unparseable/naive instant. Each
            message names both the offending field and the row's sequence
            number, so a malformed row is locatable on a live ledger.
    """
    data = _payload_of(record)
    sequence_number = record.sequence_number
    _require_payload_keys(
        data,
        sequence_number,
        label=MARKET_RESOLVED_EVENT_TYPE,
        keys=_REQUIRED_PAYLOAD_KEYS,
    )
    return _claim_from_payload(data, sequence_number, label=MARKET_RESOLVED_EVENT_TYPE)


@dataclass(frozen=True, slots=True)
class _PersistedCorrection:
    """A ``SettlementReversed`` row read back off the ledger.

    Attributes:
        superseded_sequence_number: The ledger position this row supersedes.
        claim: The corrected resolution this row asserts in its place.
    """

    superseded_sequence_number: int
    claim: IngestedResolution


def _correction_from_record(record: LedgerRecord) -> _PersistedCorrection:
    """Rebuild one persisted correction from a ``SettlementReversed`` row.

    Args:
        record: The ledger row to read.

    Returns:
        The typed correction the row carries.

    Raises:
        ValueError: If the row's payload is missing a key, names a
            non-integer superseded position, carries an unknown outcome token,
            or carries an unparseable/naive instant.
    """
    data = _payload_of(record)
    sequence_number = record.sequence_number
    _require_payload_keys(
        data,
        sequence_number,
        label=SETTLEMENT_REVERSED_EVENT_TYPE,
        keys=_REQUIRED_CORRECTION_PAYLOAD_KEYS,
    )
    target = data[_SUPERSEDED_SEQUENCE_KEY]
    if isinstance(target, bool) or not isinstance(target, int):
        raise ValueError(
            f"{SETTLEMENT_REVERSED_EVENT_TYPE} payload at "
            f"sequence_number={sequence_number} carries a non-integer "
            f"{_SUPERSEDED_SEQUENCE_KEY}: {target!r}"
        )
    return _PersistedCorrection(
        superseded_sequence_number=target,
        claim=_claim_from_payload(
            data, sequence_number, label=SETTLEMENT_REVERSED_EVENT_TYPE
        ),
    )


@dataclass(frozen=True, slots=True)
class _MarketClaim:
    """The row currently carrying one market's authoritative resolution.

    Attributes:
        sequence_number: That row's ledger position -- the only value a
            correction may name.
        resolution: The claim the row makes.
    """

    sequence_number: int
    resolution: IngestedResolution


@dataclass(frozen=True, slots=True)
class ResolutionCorrection:
    """One superseded resolution and the correction that replaced it.

    Attributes:
        market_ticker: The corrected market.
        superseded_sequence_number: The ledger position of the row that no
            longer carries the market's claim.
        superseded: The claim that row made -- still on the append-only
            ledger, never redacted, and named in the rendered report.
        correction_sequence_number: The ledger position of the
            ``SettlementReversed`` row that superseded it.
        corrected: The claim that superseded it, carrying the correction's own
            provenance label.
    """

    market_ticker: str
    superseded_sequence_number: int
    superseded: IngestedResolution
    correction_sequence_number: int
    corrected: IngestedResolution


@dataclass(frozen=True, slots=True)
class ResolutionFold:
    """Everything one ledger read says about ground truth.

    Attributes:
        resolutions: One authoritative :class:`IngestedResolution` per market,
            in the order each market was first ingested, already reflecting
            every correction.
        corrections: Every correction the ledger carries, in append order.
        claim_sequence_numbers: The ledger position of the row currently
            carrying each market's claim -- the only value a further
            correction may name.
    """

    resolutions: tuple[IngestedResolution, ...]
    corrections: tuple[ResolutionCorrection, ...]
    claim_sequence_numbers: Mapping[str, int]

    def resolution_of(self, market_ticker: str) -> IngestedResolution | None:
        """Return one market's authoritative resolution, or ``None``.

        Args:
            market_ticker: The market to look up.

        Returns:
            The market's current resolution, or ``None`` when this ledger
            carries none for it.
        """
        return next(
            (
                resolution
                for resolution in self.resolutions
                if resolution.market_ticker == market_ticker
            ),
            None,
        )


def _contradicting_resolution_message(
    record: LedgerRecord, existing: IngestedResolution, candidate: IngestedResolution
) -> str:
    """Phrase the fold's refusal of a second, contradicting resolution row.

    Args:
        record: The offending row.
        existing: The claim the ledger already carried.
        candidate: The claim this row makes.

    Returns:
        The refusal message.
    """
    return (
        f"MarketResolved at sequence_number={record.sequence_number} "
        "contradicts an earlier resolution of "
        f"{_MARKET_TICKER_KEY}={candidate.market_ticker!r}: the ledger "
        f"already carries {describe_resolution_claim(existing)}, this "
        f"row carries {describe_resolution_claim(candidate)}. "
        "`windbreak ingest-resolution` refuses a conflicting append "
        "before writing anything, so a ledger carrying both rows was "
        "not written by it; the ledger is append-only and neither row "
        "can be un-written. A genuinely wrong recorded outcome is "
        "superseded with `windbreak correct-resolution`, never re-ingested. "
        "See docs/RUNBOOK.md, 'Ingesting a resolved outcome'."
    )


def _stale_correction_message(
    record: LedgerRecord, correction: _PersistedCorrection, claim: _MarketClaim | None
) -> str:
    """Phrase the fold's refusal of a correction naming the wrong row.

    Args:
        record: The offending ``SettlementReversed`` row.
        correction: The correction that row carries.
        claim: The market's actual current claim, or ``None`` when it has none.

    Returns:
        The refusal message, naming the row that does carry the claim so the
        offending row is locatable against a live ledger.
    """
    locator = (
        "that market has no resolution on this ledger"
        if claim is None
        else f"that is sequence_number={claim.sequence_number}"
    )
    return (
        f"{SETTLEMENT_REVERSED_EVENT_TYPE} at "
        f"sequence_number={record.sequence_number} names "
        f"{_SUPERSEDED_SEQUENCE_KEY}="
        f"{correction.superseded_sequence_number}, which does not carry the "
        f"current resolution of "
        f"{_MARKET_TICKER_KEY}={correction.claim.market_ticker!r} ({locator}). "
        "A correction supersedes exactly the row it names, so a row naming "
        "any other position is refused rather than allowed to win on append "
        "order. `windbreak correct-resolution` refuses such a call before "
        "writing anything, so a ledger carrying this row was not written by "
        "it. See docs/RUNBOOK.md, 'Correcting a wrong resolution'."
    )


def _fold_resolution_row(claims: dict[str, _MarketClaim], record: LedgerRecord) -> None:
    """Fold one ``MarketResolved`` row into the running claim map.

    Args:
        claims: The per-market claim map, mutated in place.
        record: The row to fold.

    Raises:
        ValueError: If the row is malformed, or contradicts the market's
            current claim. A row restating that claim -- however its ``source``
            is spelled -- is idempotent and leaves the map untouched, so the
            claim's own row keeps carrying it.
    """
    resolution = _resolution_from_record(record)
    existing = claims.get(resolution.market_ticker)
    if existing is None:
        claims[resolution.market_ticker] = _MarketClaim(
            sequence_number=record.sequence_number, resolution=resolution
        )
    elif resolutions_conflict(existing.resolution, resolution):
        raise ValueError(
            _contradicting_resolution_message(record, existing.resolution, resolution)
        )


def _fold_correction_row(
    claims: dict[str, _MarketClaim], record: LedgerRecord
) -> ResolutionCorrection:
    """Fold one ``SettlementReversed`` row into the running claim map.

    This is the precedence rule, in one place: the correction supersedes
    exactly the row whose position it names, and only when that row is the one
    currently carrying the market's claim. A row naming anything else never
    wins on recency -- it is refused.

    Args:
        claims: The per-market claim map, mutated in place.
        record: The row to fold.

    Returns:
        The :class:`ResolutionCorrection` pairing the superseded claim with the
        one that replaced it, so the report can name both.

    Raises:
        ValueError: If the row is malformed, or names a position that does not
            carry the market's current claim.
    """
    correction = _correction_from_record(record)
    ticker = correction.claim.market_ticker
    claim = claims.get(ticker)
    if claim is None or claim.sequence_number != correction.superseded_sequence_number:
        raise ValueError(_stale_correction_message(record, correction, claim))
    claims[ticker] = _MarketClaim(
        sequence_number=record.sequence_number, resolution=correction.claim
    )
    return ResolutionCorrection(
        market_ticker=ticker,
        superseded_sequence_number=claim.sequence_number,
        superseded=claim.resolution,
        correction_sequence_number=record.sequence_number,
        corrected=correction.claim,
    )


def fold_resolutions(records: Iterable[LedgerRecord]) -> ResolutionFold:
    """Fold a ledger read into the ground truth it carries, corrections applied.

    Rows of any other event type are ignored. A market may be ingested more
    than once -- an operator re-running the verb, or a replayed script -- and a
    re-ingest making the same evidentiary claim is idempotent, including one
    that spells its ``source`` label differently (see this module's docstring on
    what counts as a contradiction). A re-ingest claiming a *different* outcome
    or settlement instant is not: letting the later row win would let a mistyped
    outcome overwrite a correct one with no trace in the report, so it raises
    instead.

    A wrong *first* ingest is overturned the other way (issue #484): a
    ``SettlementReversed`` row names the position it supersedes and carries the
    corrected claim, and the superseded claim is returned alongside so the
    report can name it. Nothing is redacted; both rows stay on the chain.

    Args:
        records: The ledger rows to fold, in append order.

    Returns:
        The :class:`ResolutionFold` carrying each market's authoritative
        resolution, every correction, and the position of the row currently
        carrying each market's claim.

    Raises:
        ValueError: If a row is malformed (see :func:`_resolution_from_record`
            and :func:`_correction_from_record`), if two ``MarketResolved``
            rows make different evidentiary claims about one market, or if a
            ``SettlementReversed`` row names a position that does not carry its
            market's current claim.
    """
    claims: dict[str, _MarketClaim] = {}
    corrections: list[ResolutionCorrection] = []
    for record in records:
        if record.event_type == MARKET_RESOLVED_EVENT_TYPE:
            _fold_resolution_row(claims, record)
        elif record.event_type == SETTLEMENT_REVERSED_EVENT_TYPE:
            corrections.append(_fold_correction_row(claims, record))
    return ResolutionFold(
        resolutions=tuple(claim.resolution for claim in claims.values()),
        corrections=tuple(corrections),
        claim_sequence_numbers={
            ticker: claim.sequence_number for ticker, claim in claims.items()
        },
    )


def ingested_resolutions_from_records(
    records: Iterable[LedgerRecord],
) -> tuple[IngestedResolution, ...]:
    """Fold a ledger read into the ground truth it carries.

    The #439 entry point, kept because every existing reader calls it. It is a
    thin projection of :func:`fold_resolutions` rather than a second fold, so a
    caller reading only the resolutions can never disagree with one reading the
    corrections about what a market settled to.

    Args:
        records: The ledger rows to fold, in append order.

    Returns:
        One :class:`IngestedResolution` per distinct market, in the order each
        market was first ingested, already reflecting every correction.

    Raises:
        ValueError: Whatever :func:`fold_resolutions` raises.
    """
    return fold_resolutions(records).resolutions


def render_correction_lines(
    corrections: Iterable[ResolutionCorrection],
) -> str | None:
    """Render the weekly report's resolution-correction ledger.

    Each line carries exact values on both sides of the arrow -- both ledger
    positions, both outcomes, both instants -- because "this market was
    corrected" without the superseded claim is precisely the trace-free
    correction the contradiction guard exists to prevent.

    Args:
        corrections: The corrections a fold reported, in append order.

    Returns:
        The rendered lines, or ``None`` when there are none. ``None`` omits the
        report section entirely rather than printing an empty one: a heading
        that appears on every report is a heading operators learn to skip, and
        the presence of this one is itself the signal that ground truth was
        overturned.
    """
    lines = [
        f"RESOLUTION_CORRECTED {correction.market_ticker} "
        f"{_SUPERSEDED_SEQUENCE_KEY}={correction.superseded_sequence_number} "
        f"{describe_resolution_claim(correction.superseded)} -> "
        f"correction_sequence_number={correction.correction_sequence_number} "
        f"{describe_resolution_claim(correction.corrected)} "
        f"{_SOURCE_KEY}={correction.corrected.source!r}"
        for correction in corrections
    ]
    if not lines:
        return None
    return "\n".join(lines)


def resolution_outcomes(
    resolutions: Iterable[IngestedResolution],
) -> Mapping[str, ResolutionOutcome]:
    """Project ingested resolutions into the harness's ticker-keyed mapping.

    Args:
        resolutions: The ingested resolutions to project.

    Returns:
        A mapping from ``market_ticker`` to its settled outcome -- the same
        shape :func:`windbreak.evaluation.resolution.resolutions_from_fixture`
        returns, so the ledger path and the fixture path feed the metrics
        identically.
    """
    return {resolution.market_ticker: resolution.outcome for resolution in resolutions}


def resolution_instants(
    resolutions: Iterable[IngestedResolution],
) -> Mapping[str, datetime]:
    """Project ingested resolutions into their ticker-keyed settlement instants.

    Args:
        resolutions: The ingested resolutions to project.

    Returns:
        A mapping from ``market_ticker`` to the instant it settled, ready for
        :func:`windbreak.evaluation.temporal.resolution_sequences_from_instants`
        to project onto the ledger's sequence axis.
    """
    return {
        resolution.market_ticker: resolution.resolved_at for resolution in resolutions
    }
