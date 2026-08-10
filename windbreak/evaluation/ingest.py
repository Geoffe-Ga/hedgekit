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

#: The payload keys every ``MarketResolved`` row must carry, in the order a
#: malformed row is checked against so the first missing one is named.
_REQUIRED_PAYLOAD_KEYS = (
    _MARKET_TICKER_KEY,
    _OUTCOME_KEY,
    _RESOLVED_AT_KEY,
    _SOURCE_KEY,
)


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


def _require_payload_keys(data: Mapping[str, Any], sequence_number: int) -> None:
    """Check that a persisted payload carries every required key.

    Args:
        data: The row's typed payload.
        sequence_number: The row's ledger position, named in the error.

    Raises:
        ValueError: If any required key is absent, naming the first one
            missing rather than raising a bare :class:`KeyError`.
    """
    for key in _REQUIRED_PAYLOAD_KEYS:
        if key not in data:
            raise ValueError(
                f"MarketResolved payload at sequence_number={sequence_number} "
                f"is missing {key!r}"
            )


def _outcome_from_payload(token: object, sequence_number: int) -> ResolutionOutcome:
    """Parse a persisted outcome token, failing closed on an unknown value.

    Args:
        token: The raw ``outcome`` value off the payload.
        sequence_number: The row's ledger position, named in the error.

    Returns:
        The typed :class:`~windbreak.evaluation.resolution.ResolutionOutcome`.

    Raises:
        ValueError: If ``token`` is not one of the binary outcome tokens.
    """
    try:
        return ResolutionOutcome(token)
    except ValueError as exc:
        raise ValueError(
            f"MarketResolved payload at sequence_number={sequence_number} "
            f"carries an unknown {_OUTCOME_KEY}: {token!r} (expected one of "
            f"{[member.value for member in ResolutionOutcome]})"
        ) from exc


def _instant_from_payload(token: object, sequence_number: int) -> datetime:
    """Parse a persisted settlement instant, failing closed twice over.

    Args:
        token: The raw ``resolved_at`` value off the payload.
        sequence_number: The row's ledger position, named in the error.

    Returns:
        The parsed, timezone-aware instant.

    Raises:
        ValueError: If ``token`` is not a string, is not an ISO-8601 instant,
            or carries no UTC offset.
    """
    moment = _parsed_instant(token) if isinstance(token, str) else None
    if moment is None:
        raise ValueError(
            f"MarketResolved payload at sequence_number={sequence_number} "
            f"carries an unparseable {_RESOLVED_AT_KEY}: {token!r} "
            "(expected an ISO-8601 instant)"
        )
    require_aware(moment, _RESOLVED_AT_KEY)
    return moment


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
    envelope: dict[str, Any] = json.loads(record.payload_json)
    data: dict[str, Any] = envelope[_PAYLOAD_DATA_KEY]
    sequence_number = record.sequence_number
    _require_payload_keys(data, sequence_number)
    return IngestedResolution(
        market_ticker=str(data[_MARKET_TICKER_KEY]),
        outcome=_outcome_from_payload(data[_OUTCOME_KEY], sequence_number),
        resolved_at=_instant_from_payload(data[_RESOLVED_AT_KEY], sequence_number),
        source=str(data[_SOURCE_KEY]),
    )


def ingested_resolutions_from_records(
    records: Iterable[LedgerRecord],
) -> tuple[IngestedResolution, ...]:
    """Fold a ledger read into the ground truth it carries.

    Rows of any other event type are ignored. A market may be ingested more
    than once -- an operator re-running the verb, or a replayed script -- and a
    byte-identical re-ingest is idempotent. A *contradicting* re-ingest is not:
    letting the later row win would let a mistyped outcome overwrite a correct
    one with no trace in the report, so it raises instead. Correcting a settled
    market is what the settlement-reversal vocabulary in
    :mod:`windbreak.evaluation.resolution` is for.

    Args:
        records: The ledger rows to fold, in append order.

    Returns:
        One :class:`IngestedResolution` per distinct market, in the order each
        market was first ingested.

    Raises:
        ValueError: If a ``MarketResolved`` row is malformed (see
            :func:`_resolution_from_record`), or if two rows carry different
            resolutions for one market.
    """
    resolutions: dict[str, IngestedResolution] = {}
    for record in records:
        if record.event_type != MARKET_RESOLVED_EVENT_TYPE:
            continue
        resolution = _resolution_from_record(record)
        existing = resolutions.get(resolution.market_ticker)
        if existing is None:
            resolutions[resolution.market_ticker] = resolution
        elif existing != resolution:
            raise ValueError(
                f"MarketResolved at sequence_number={record.sequence_number} "
                "contradicts an earlier resolution of "
                f"{_MARKET_TICKER_KEY}={resolution.market_ticker!r}; a settled "
                "market is corrected through a settlement reversal, never by a "
                "second conflicting ingest"
            )
    return tuple(resolutions.values())


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
