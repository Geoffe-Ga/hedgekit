"""Durable research spend and a runtime-adjustable daily ceiling (#442, #483).

Before this module the per-UTC-day research ceiling was not a per-day ceiling.
:class:`windbreak.forecast.budget.ResearchBudget` kept its counter in an
instance ``dict``, ``_build_research_budget`` constructed a fresh one from
configuration on every process start, and ``charge_forecast`` ledgered only
*breaches* -- so nothing on disk said what the day had already spent.
``deploy/docker-compose.yml`` sets ``restart: on-failure`` and every
``deploy/systemd/*.service`` sets ``Restart=on-failure``, so the ceiling was a
**per-process** ceiling and the process count per day is unbounded.

That was survivable only by accident, because issue #483's per-trade research
charge held spend near zero by refusing every market. Removing that charge
deletes the accidental brake, which is why #442 and #483 land together: a cap
that resets on restart is not a cap.

WHAT THIS MODULE IS

The hash-chained ledger is already the durable, append-only state store the
loop reads on every tick, so it is the transport. Two typed rows, and the folds
that read them back:

* :class:`ResearchSpendRecorded` -- one row per *successful* charge, carrying
  the UTC day it bucketed to. :func:`spend_by_day_from_records` folds them into
  the ``{utc_day: micros}`` map a rebuilt budget opens with, so the day's spend
  is rehydrated rather than reset.
* :class:`ResearchBudgetCapSet` -- an operator's runtime instruction to change
  the per-UTC-day ceiling, appended by ``windbreak set-research-budget``.
  :func:`effective_per_day_micros` folds them, **last row wins**, falling back
  to the configured ceiling when the ledger carries none.

The loop re-reads both on every tick, so a cap change is picked up on the next
beat without a restart and a second process's spend is picked up too.

LAST ROW WINS, AND NOTHING CONFLICTS

There is deliberately no contradiction check between two cap rows, and that is
the lesson of PR #482 applied rather than restated. That verb shipped an
equality check that included a free-text provenance label, so two ingests
differing only in wording were read as contradicting each other -- and because
the fold ran on every tick against an append-only ledger, the mistake was
terminal. A cap is not evidence about the outside world; it is an instruction,
and a later instruction supersedes an earlier one. Two cap rows can therefore
never conflict, so no fold here can ever brick a tick over one.

The rows can still be *malformed*, and that fails closed and loudly: a spend
row whose payload cannot be read is not skipped, because skipping it would
undercount the day and re-open a ceiling that should be shut -- the exact
runaway this module exists to prevent. ``windbreak set-research-budget``
validates before it appends, so it cannot write a row this fold refuses.

Both events derive their base :class:`~windbreak.ledger.events.Event` fields
through a LOCAL ``_derive_typed_event``, the house pattern from
:mod:`windbreak.evaluation.ingest` and :mod:`windbreak.evaluation.crosscheck`,
so this change never touches the ledger's central ``EVENT_TYPES`` map. Money is
integer micros throughout and no float appears.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from windbreak.ledger.store import LedgerRecord

from windbreak.ledger.events import Event

#: The immutable event-type token every recorded research charge is ledgered
#: under. Equal to :class:`ResearchSpendRecorded`'s class name by construction,
#: since the local derivation stamps ``type(event).__name__``.
RESEARCH_SPEND_RECORDED_EVENT_TYPE: Final = "ResearchSpendRecorded"

#: The immutable event-type token every operator cap change is ledgered under.
RESEARCH_BUDGET_CAP_SET_EVENT_TYPE: Final = "ResearchBudgetCapSet"

#: Payload schema version stamped on this module's events. Replicated locally
#: (rather than imported from :mod:`windbreak.ledger.events`'s private copy) so
#: a payload-shape change here can be versioned independently.
_SCHEMA_VERSION: Final = 1

#: Envelope key under which a ledgered event's typed payload is nested.
_PAYLOAD_DATA_KEY: Final = "data"

#: Payload key carrying the ISO ``YYYY-MM-DD`` UTC day a charge bucketed to.
_UTC_DAY_KEY: Final = "utc_day"
#: Payload key carrying a charge's cost, in micros.
_COST_MICROS_KEY: Final = "cost_micros"
#: Payload key carrying the market a charge was incurred researching.
_MARKET_TICKER_KEY: Final = "market_ticker"
#: Payload key carrying an operator-set per-UTC-day ceiling, in micros.
_PER_DAY_MICROS_KEY: Final = "per_day_micros"
#: Payload key carrying the operator's free-text note on a cap change.
_NOTE_KEY: Final = "note"


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


def _require_non_negative_micros(value: int, field_name: str) -> None:
    """Guard that a micros field is non-negative.

    Args:
        value: The candidate micros integer.
        field_name: The owning field's name, surfaced in the error message.

    Raises:
        ValueError: If ``value`` is negative. A negative spend would *credit*
            the day's counter and a negative ceiling would be unenforceable, so
            both are refused rather than clamped.
    """
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")


def _require_non_blank(value: str, field_name: str) -> None:
    """Guard that an identifier/label field carries something.

    Args:
        value: The candidate string.
        field_name: The owning field's name, surfaced in the error message.

    Raises:
        ValueError: If ``value`` is blank or whitespace-only.
    """
    if not value.strip():
        raise ValueError(f"{field_name} must be non-blank")


@dataclass(frozen=True)
class ResearchSpendRecorded(Event):
    """Records one successful research charge against the UTC day's budget.

    Appended by the scheduler's budget ledger writer on *every* charge, not
    only on a breach. That is the whole point of issue #442: without a durable
    row per charge there is nothing for a restarted process to fold, so its day
    counter starts at zero however much the previous process spent.

    Attributes:
        utc_day: The charge's UTC calendar day, as an ISO ``YYYY-MM-DD``
            string -- the same key
            :class:`windbreak.forecast.budget.ResearchBudget` buckets spend
            under, carried rather than re-derived so the fold cannot bucket a
            row differently from the charge that wrote it.
        market_ticker: The market whose forecast incurred the charge.
        cost_micros: The charge, in micros. Never negative.
    """

    utc_day: str
    market_ticker: str
    cost_micros: int
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Validate the charge, then assemble the payload and base fields.

        Raises:
            ValueError: If ``utc_day`` is blank (an unbucketable charge could
                never be folded back) or ``cost_micros`` is negative (which
                would credit the day's counter).
        """
        _require_non_blank(self.utc_day, _UTC_DAY_KEY)
        _require_non_negative_micros(self.cost_micros, _COST_MICROS_KEY)
        payload: dict[str, object] = {
            _UTC_DAY_KEY: self.utc_day,
            _MARKET_TICKER_KEY: self.market_ticker,
            _COST_MICROS_KEY: self.cost_micros,
        }
        _derive_typed_event(self, payload)


@dataclass(frozen=True)
class ResearchBudgetCapSet(Event):
    """Records an operator changing the per-UTC-day research ceiling.

    The runtime half of "settable without a source-code change": appended by
    ``windbreak set-research-budget`` while the loop keeps beating, and read by
    the next tick. A later row supersedes an earlier one; two rows can never
    contradict, so no fold of them can ever refuse to read (see this module's
    docstring).

    Attributes:
        per_day_micros: The new per-UTC-day research ceiling, in micros. Never
            negative. ``0`` is meaningful and allowed: it stops all further
            research spend for every day, which is the shape an operator wants
            when a cost anomaly is under investigation.
        note: Why the operator changed it, recorded verbatim for the audit
            trail. Never blank -- a money ceiling nothing explains cannot be
            reviewed -- and never compared against anything, so its wording can
            never make two rows disagree.
    """

    per_day_micros: int
    note: str
    event_type: str = field(init=False)
    payload_schema_version: int = field(init=False)
    payload: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        """Validate the ceiling, then assemble the payload and base fields.

        Raises:
            ValueError: If ``per_day_micros`` is negative (an unenforceable
                ceiling) or ``note`` is blank.
        """
        _require_non_negative_micros(self.per_day_micros, _PER_DAY_MICROS_KEY)
        _require_non_blank(self.note, _NOTE_KEY)
        payload: dict[str, object] = {
            _PER_DAY_MICROS_KEY: self.per_day_micros,
            _NOTE_KEY: self.note,
        }
        _derive_typed_event(self, payload)


def _payload_of(record: LedgerRecord) -> dict[str, Any]:
    """Return one row's typed payload.

    Args:
        record: The ledger row to read.

    Returns:
        The row's ``data`` payload.
    """
    envelope: dict[str, Any] = json.loads(record.payload_json)
    data: dict[str, Any] = envelope[_PAYLOAD_DATA_KEY]
    return data


def _int_field(record: LedgerRecord, data: Mapping[str, Any], key: str) -> int:
    """Read a required integer payload field, failing closed and locatably.

    Args:
        record: The row being read, whose sequence number locates a failure.
        data: The row's typed payload.
        key: The payload key to read.

    Returns:
        The integer value.

    Raises:
        ValueError: If the key is absent or its value is not an integer. Read
            by subscript-with-check rather than ``.get(key, 0)``: a defaulted
            zero here would silently undercount the day's spend, which is the
            precise failure this module exists to close.
    """
    if key not in data:
        raise ValueError(
            f"{record.event_type} payload at "
            f"sequence_number={record.sequence_number} is missing {key!r}"
        )
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"{record.event_type} payload at "
            f"sequence_number={record.sequence_number} carries a non-integer "
            f"{key!r}: {value!r}"
        )
    return value


def _str_field(record: LedgerRecord, data: Mapping[str, Any], key: str) -> str:
    """Read a required string payload field, failing closed and locatably.

    Args:
        record: The row being read, whose sequence number locates a failure.
        data: The row's typed payload.
        key: The payload key to read.

    Returns:
        The string value.

    Raises:
        ValueError: If the key is absent or its value is not a string.
    """
    if key not in data:
        raise ValueError(
            f"{record.event_type} payload at "
            f"sequence_number={record.sequence_number} is missing {key!r}"
        )
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(
            f"{record.event_type} payload at "
            f"sequence_number={record.sequence_number} carries a non-string "
            f"{key!r}: {value!r}"
        )
    return value


def spend_by_day_from_records(
    records: Iterable[LedgerRecord],
) -> Mapping[str, int]:
    """Fold every recorded research charge into per-UTC-day totals.

    The rehydration half of issue #442. Rows of any other event type are
    ignored; every ``ResearchSpendRecorded`` row is summed into its own carried
    ``utc_day`` bucket, so a ledger written across a UTC midnight yields two
    buckets and a restart inside one day resumes with that day's real total.

    Args:
        records: The ledger rows to fold, in append order.

    Returns:
        A ``{utc_day: total_micros}`` mapping over every day the ledger carries
        a charge for. Days with no charges are absent, which a budget reads as
        zero.

    Raises:
        ValueError: If a ``ResearchSpendRecorded`` row's payload is missing a
            key or carries a wrongly-typed one. It fails closed rather than
            skipping the row: an unreadable charge dropped from the fold would
            undercount the day and re-open a ceiling that should be shut.
    """
    totals: dict[str, int] = {}
    for record in records:
        if record.event_type != RESEARCH_SPEND_RECORDED_EVENT_TYPE:
            continue
        data = _payload_of(record)
        day = _str_field(record, data, _UTC_DAY_KEY)
        cost = _int_field(record, data, _COST_MICROS_KEY)
        totals[day] = totals.get(day, 0) + cost
    return totals


def effective_per_day_micros(
    records: Iterable[LedgerRecord], *, configured_micros: int
) -> int:
    """Return the per-UTC-day ceiling in force, ledger overriding configuration.

    The runtime-adjustment half of the owner's 2026-08-10 decision. The
    *latest* ``ResearchBudgetCapSet`` row wins, because a cap is an instruction
    and a later instruction supersedes an earlier one -- never a merge, never a
    conflict. With no such row the configured ceiling stands, so a deployment
    that has never run the verb behaves exactly as it did before.

    Precedence is deliberately this way round rather than "whichever is
    smaller": the operator must be able to *raise* the cap on a live loop
    (issue #483 acceptance criterion 3 requires proving exactly that), and a
    minimum rule would make raising it impossible without a restart.
    ``windbreak run`` logs which source won, so the precedence is visible
    rather than inferred.

    Args:
        records: The ledger rows to fold, in append order.
        configured_micros: The ceiling from configuration (itself overridable
            at startup by ``windbreak run --research-per-day-micros``), used
            when the ledger carries no cap row (keyword-only).

    Returns:
        The per-UTC-day research ceiling in force, in micros.

    Raises:
        ValueError: If a ``ResearchBudgetCapSet`` row's payload is missing
            ``per_day_micros`` or carries a wrongly-typed one.
    """
    effective = configured_micros
    for record in records:
        if record.event_type != RESEARCH_BUDGET_CAP_SET_EVENT_TYPE:
            continue
        data = _payload_of(record)
        effective = _int_field(record, data, _PER_DAY_MICROS_KEY)
    return effective
