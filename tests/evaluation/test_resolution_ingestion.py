"""Tests for resolution ingestion and its instant/sequence projection (#439).

Nothing in the running system ever learned that a market resolved: the only
loader in `windbreak.evaluation.resolution` reads a known-answer *fixture*, so
the always-on loop's fold hardcoded `resolutions={}` and every evaluation
metric read `UNDEFINED` forever. `windbreak.evaluation.ingest` is the missing
seam -- a `MarketResolved` ledger event carrying the settled outcome, the
*instant* the market settled, and the provenance of that claim.

The instant is the load-bearing part. The temporal gate
(`windbreak.evaluation.temporal`) is sequence-based by design (SPEC S1.1-6, no
wall clock on the value path), so a wall-clock resolution instant must be
*projected* onto the ledger's sequence axis before it can gate anything:
`resolution_sequences_from_instants` answers "at what ledger position could
this market's answer first have been known". That projection is what makes a
late-ingested resolution still refuse a forecast made after the market
actually settled -- if the gate keyed on the ingesting row's own position
instead, an operator who ingested a week late would silently score a week of
forecasts that already knew the answer.

Every instant in this module is exact and timezone-aware; a naive one is
refused at the boundary (`windbreak.timekeeping.require_aware`) rather than
reinterpreted against the host's zone.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from windbreak.evaluation.ingest import (
    MARKET_RESOLVED_EVENT_TYPE,
    IngestedResolution,
    MarketResolved,
    ingested_resolution_of,
    ingested_resolutions_from_records,
    parse_resolution_instant,
    resolution_outcomes,
    resolutions_conflict,
)
from windbreak.evaluation.resolution import ResolutionOutcome
from windbreak.evaluation.temporal import resolution_sequences_from_instants
from windbreak.ledger.store import LedgerRecord, SqliteLedgerStore

if TYPE_CHECKING:
    from pathlib import Path

#: The component label stamped on every event this suite ledgers.
_COMPONENT = "operator"

#: The exact instant the fixtures treat as "the market settled".
_RESOLVED_AT = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


def _record(
    *,
    sequence_number: int,
    event_type: str,
    created_at: str,
    payload: dict[str, object],
) -> LedgerRecord:
    """Build one `LedgerRecord` with a hand-written envelope.

    Args:
        sequence_number: The row's position in the chain.
        event_type: The row's event-type discriminator.
        created_at: The row's ISO-8601 creation timestamp.
        payload: The typed payload nested under the envelope's `data` key.

    Returns:
        The assembled `LedgerRecord`; the hash columns are filler, since no
        consumer under test verifies the chain.
    """
    envelope = {"component": _COMPONENT, "data": payload, "schema_version": 1}
    return LedgerRecord(
        sequence_number=sequence_number,
        event_type=event_type,
        created_at=created_at,
        component=_COMPONENT,
        payload_json=json.dumps(envelope, sort_keys=True, separators=(",", ":")),
        payload_schema_version=1,
        prev_hash="0" * 64,
        event_hash="f" * 64,
    )


def _resolved_payload(
    *,
    market_ticker: str = "MKT-A",
    outcome: str = "no",
    resolved_at: str = "2026-03-01T12:00:00.000000Z",
    source: str = "kalshi-settlement-notice",
) -> dict[str, object]:
    """Build a well-formed `MarketResolved` payload, overridable field by field.

    Args:
        market_ticker: The market the resolution names.
        outcome: The settled outcome token.
        resolved_at: The ISO-8601 settlement instant.
        source: Where the operator read the settlement.

    Returns:
        The four-key payload dict.
    """
    return {
        "market_ticker": market_ticker,
        "outcome": outcome,
        "resolved_at": resolved_at,
        "source": source,
    }


# ---------------------------------------------------------------------------
# 1. The MarketResolved ledger event: shape, provenance, and boundary refusals.
# ---------------------------------------------------------------------------


def test_market_resolved_event_stamps_its_type_and_four_field_payload() -> None:
    """`MarketResolved` derives the fixed event type and a four-key payload."""
    event = MarketResolved(
        component=_COMPONENT,
        market_ticker="MKT-A",
        outcome=ResolutionOutcome.NO,
        resolved_at=_RESOLVED_AT,
        source="kalshi-settlement-notice",
    )

    assert event.event_type == MARKET_RESOLVED_EVENT_TYPE
    assert event.event_type == "MarketResolved"
    assert event.payload == {
        "market_ticker": "MKT-A",
        "outcome": "no",
        "resolved_at": "2026-03-01T12:00:00.000000Z",
        "source": "kalshi-settlement-notice",
    }


def test_market_resolved_refuses_a_naive_resolution_instant() -> None:
    """A `resolved_at` with no UTC offset is refused, never read as local time."""
    with pytest.raises(ValueError) as exc_info:
        MarketResolved(
            component=_COMPONENT,
            market_ticker="MKT-A",
            outcome=ResolutionOutcome.NO,
            resolved_at=datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None),
            source="kalshi-settlement-notice",
        )

    assert type(exc_info.value) is ValueError
    assert "resolved_at must be timezone-aware" in str(exc_info.value)


def test_market_resolved_refuses_a_blank_source() -> None:
    """A blank `source` is refused: an unprovenanced outcome is not evidence."""
    with pytest.raises(ValueError) as exc_info:
        MarketResolved(
            component=_COMPONENT,
            market_ticker="MKT-A",
            outcome=ResolutionOutcome.NO,
            resolved_at=_RESOLVED_AT,
            source="   ",
        )

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == (
        "source must be a non-blank provenance label; an outcome nothing "
        "attributes is unprovable evidence and is refused rather than ingested"
    )


def test_market_resolved_refuses_a_blank_market_ticker() -> None:
    """A blank `market_ticker` is refused: it can never join to a forecast."""
    with pytest.raises(ValueError) as exc_info:
        MarketResolved(
            component=_COMPONENT,
            market_ticker=" ",
            outcome=ResolutionOutcome.NO,
            resolved_at=_RESOLVED_AT,
            source="kalshi-settlement-notice",
        )

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == (
        "market_ticker must be a non-blank market identifier; a resolution "
        "naming no market can never join to a forecast"
    )


def test_market_resolved_round_trips_through_a_real_ledger(tmp_path: Path) -> None:
    """An appended `MarketResolved` reads back as an `IngestedResolution`."""
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    store.append(
        MarketResolved(
            component=_COMPONENT,
            market_ticker="MKT-A",
            outcome=ResolutionOutcome.YES,
            resolved_at=_RESOLVED_AT,
            source="kalshi-settlement-notice",
        )
    )

    resolutions = ingested_resolutions_from_records(store.read_all())

    assert resolutions == (
        IngestedResolution(
            market_ticker="MKT-A",
            outcome=ResolutionOutcome.YES,
            resolved_at=_RESOLVED_AT,
            source="kalshi-settlement-notice",
        ),
    )


# ---------------------------------------------------------------------------
# 2. The record fold: it reads only MarketResolved rows, and fails closed.
# ---------------------------------------------------------------------------


def test_ingested_resolutions_ignores_every_other_event_type() -> None:
    """Non-`MarketResolved` rows contribute nothing and never crash the fold."""
    records = [
        _record(
            sequence_number=1,
            event_type="ConfigLoaded",
            created_at="2026-03-01T00:00:00.000000+00:00",
            payload={"config_hash": "abc", "diff": {}},
        ),
        _record(
            sequence_number=2,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-01T13:00:00.000000+00:00",
            payload=_resolved_payload(),
        ),
    ]

    resolutions = ingested_resolutions_from_records(records)

    assert len(resolutions) == 1
    assert resolutions[0].market_ticker == "MKT-A"


@pytest.mark.parametrize(
    "missing_key",
    ["market_ticker", "outcome", "resolved_at", "source"],
)
def test_ingested_resolutions_fails_closed_on_a_missing_payload_key(
    missing_key: str,
) -> None:
    """A `MarketResolved` row missing any key raises, naming the missing key."""
    payload = _resolved_payload()
    del payload[missing_key]
    records = [
        _record(
            sequence_number=1,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-01T13:00:00.000000+00:00",
            payload=payload,
        )
    ]

    with pytest.raises(ValueError) as exc_info:
        ingested_resolutions_from_records(records)

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == (
        f"MarketResolved payload at sequence_number=1 is missing {missing_key!r}"
    )


def test_ingested_resolutions_fails_closed_on_an_unknown_outcome_token() -> None:
    """An outcome token outside `{yes, no}` raises, naming the offending token."""
    records = [
        _record(
            sequence_number=4,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-01T13:00:00.000000+00:00",
            payload=_resolved_payload(outcome="maybe"),
        )
    ]

    with pytest.raises(ValueError) as exc_info:
        ingested_resolutions_from_records(records)

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == (
        "MarketResolved payload at sequence_number=4 carries an unknown "
        "outcome: 'maybe' (expected one of ['yes', 'no'])"
    )


def test_ingested_resolutions_fails_closed_on_an_unparseable_instant() -> None:
    """A `resolved_at` that is not an ISO-8601 instant raises, naming the field."""
    records = [
        _record(
            sequence_number=7,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-01T13:00:00.000000+00:00",
            payload=_resolved_payload(resolved_at="last Tuesday"),
        )
    ]

    with pytest.raises(ValueError) as exc_info:
        ingested_resolutions_from_records(records)

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == (
        "MarketResolved payload at sequence_number=7 carries an unparseable "
        "resolved_at: 'last Tuesday' (expected an ISO-8601 instant)"
    )


def test_ingested_resolutions_fails_closed_on_a_naive_instant_on_disk() -> None:
    """A hand-written row whose `resolved_at` has no offset is refused."""
    records = [
        _record(
            sequence_number=9,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-01T13:00:00.000000+00:00",
            payload=_resolved_payload(resolved_at="2026-03-01T12:00:00.000000"),
        )
    ]

    with pytest.raises(ValueError) as exc_info:
        ingested_resolutions_from_records(records)

    assert type(exc_info.value) is ValueError
    assert "resolved_at must be timezone-aware" in str(exc_info.value)


def test_ingested_resolutions_tolerates_an_identical_re_ingest() -> None:
    """Re-ingesting the byte-identical resolution is idempotent, not an error."""
    payload = _resolved_payload()
    records = [
        _record(
            sequence_number=1,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-01T13:00:00.000000+00:00",
            payload=dict(payload),
        ),
        _record(
            sequence_number=2,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-02T13:00:00.000000+00:00",
            payload=dict(payload),
        ),
    ]

    resolutions = ingested_resolutions_from_records(records)

    assert len(resolutions) == 1
    assert resolutions[0].outcome is ResolutionOutcome.NO


def test_ingested_resolutions_fails_closed_on_a_contradicting_re_ingest() -> None:
    """A second, *different* resolution for one market raises rather than wins.

    Silently letting the later row win would let a mis-typed outcome overwrite
    a correct one with no trace in the report. The message quotes both claims
    exactly and names no remedy that does not exist: there is no settlement
    reversal to perform (issue #484), and the verb refuses this append before
    it is ever written.
    """
    records = [
        _record(
            sequence_number=1,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-01T13:00:00.000000+00:00",
            payload=_resolved_payload(outcome="no"),
        ),
        _record(
            sequence_number=2,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-02T13:00:00.000000+00:00",
            payload=_resolved_payload(outcome="yes"),
        ),
    ]

    with pytest.raises(ValueError) as exc_info:
        ingested_resolutions_from_records(records)

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == (
        "MarketResolved at sequence_number=2 contradicts an earlier resolution "
        "of market_ticker='MKT-A': the ledger already carries outcome='no' "
        "resolved_at=2026-03-01T12:00:00.000000Z, this row carries "
        "outcome='yes' resolved_at=2026-03-01T12:00:00.000000Z. "
        "`windbreak ingest-resolution` refuses a conflicting append before "
        "writing anything, so a ledger carrying both rows was not written by "
        "it; the ledger is append-only and neither row can be un-written. See "
        "docs/RUNBOOK.md, 'Ingesting a resolved outcome'."
    )
    assert "settlement reversal" not in str(exc_info.value)


def test_a_re_ingest_differing_only_in_source_is_not_a_contradiction() -> None:
    """Retyping the free-text provenance label is idempotent, not a conflict.

    `source` records where an operator read the settlement; it says nothing
    about what the market did. Comparing whole frozen records made
    `"kalshi settlement notice"` and `"kalshi-settlement-notice"` a reported
    contradiction *about the settled outcome*, which was false -- and, because
    this fold runs on every tick over an append-only ledger, terminal.
    """
    records = [
        _record(
            sequence_number=1,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-01T13:00:00.000000+00:00",
            payload=_resolved_payload(source="kalshi settlement notice"),
        ),
        _record(
            sequence_number=2,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-02T13:00:00.000000+00:00",
            payload=_resolved_payload(source="kalshi-settlement-notice"),
        ),
    ]

    resolutions = ingested_resolutions_from_records(records)

    assert resolutions == (
        IngestedResolution(
            market_ticker="MKT-A",
            outcome=ResolutionOutcome.NO,
            resolved_at=_RESOLVED_AT,
            source="kalshi settlement notice",
        ),
    )


def test_a_re_ingest_one_microsecond_apart_is_a_contradiction() -> None:
    """The settlement instant is compared exactly, not rounded to the second.

    Two instants inside one second can project onto different ledger positions
    and therefore admit or refuse different forecasts, so a microsecond's
    difference is a genuine disagreement about which forecasts could have
    peeked -- not a formatting nit to be smoothed over.
    """
    records = [
        _record(
            sequence_number=1,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-01T13:00:00.000000+00:00",
            payload=_resolved_payload(resolved_at="2026-03-01T12:00:00.000000Z"),
        ),
        _record(
            sequence_number=2,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-02T13:00:00.000000+00:00",
            payload=_resolved_payload(resolved_at="2026-03-01T12:00:00.000001Z"),
        ),
    ]

    with pytest.raises(ValueError) as exc_info:
        ingested_resolutions_from_records(records)

    assert str(exc_info.value).startswith(
        "MarketResolved at sequence_number=2 contradicts an earlier resolution "
        "of market_ticker='MKT-A': the ledger already carries outcome='no' "
        "resolved_at=2026-03-01T12:00:00.000000Z, this row carries "
        "outcome='no' resolved_at=2026-03-01T12:00:00.000001Z."
    )


def test_the_same_instant_written_at_two_offsets_is_not_a_contradiction() -> None:
    """Offset spelling is not a claim: `07:00-05:00` and `12:00Z` are one instant."""
    records = [
        _record(
            sequence_number=1,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-01T13:00:00.000000+00:00",
            payload=_resolved_payload(resolved_at="2026-03-01T12:00:00.000000Z"),
        ),
        _record(
            sequence_number=2,
            event_type=MARKET_RESOLVED_EVENT_TYPE,
            created_at="2026-03-02T13:00:00.000000+00:00",
            payload=_resolved_payload(resolved_at="2026-03-01T07:00:00.000000-05:00"),
        ),
    ]

    resolutions = ingested_resolutions_from_records(records)

    assert len(resolutions) == 1
    assert resolutions[0].resolved_at == _RESOLVED_AT


def test_resolutions_conflict_ignores_source_and_compares_the_other_three() -> None:
    """The shared conflict predicate reads exactly three fields, and not `source`."""
    base = IngestedResolution(
        market_ticker="MKT-A",
        outcome=ResolutionOutcome.NO,
        resolved_at=_RESOLVED_AT,
        source="kalshi-settlement-notice",
    )

    assert resolutions_conflict(base, replace(base, source="a wholly other label")) is (
        False
    )
    assert resolutions_conflict(base, replace(base, market_ticker="MKT-B")) is True
    assert resolutions_conflict(base, replace(base, outcome=ResolutionOutcome.YES)) is (
        True
    )
    assert (
        resolutions_conflict(
            base, replace(base, resolved_at=_RESOLVED_AT + timedelta(microseconds=1))
        )
        is True
    )


def test_a_round_tripped_event_folds_back_to_its_own_projection(
    tmp_path: Path,
) -> None:
    """`ingested_resolution_of` equals what a fold of the appended row returns.

    The `ingest-resolution` verb compares its candidate against the ledger
    *before* appending, so its projection of an unwritten event must equal what
    the tick's fold would later read back. The instant carries microseconds
    precisely so this cannot be assumed: `iso_z` writes `%f` and
    `fromisoformat` reads it, and this pins that the round trip is lossless.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    event = MarketResolved(
        component=_COMPONENT,
        market_ticker="MKT-A",
        outcome=ResolutionOutcome.NO,
        resolved_at=_RESOLVED_AT + timedelta(microseconds=123_456),
        source="kalshi-settlement-notice",
    )
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        store.append(event)
        folded = ingested_resolutions_from_records(store.read_all())
    finally:
        store.close()

    assert folded == (ingested_resolution_of(event),)
    assert folded[0].resolved_at.microsecond == 123_456


def test_resolution_outcomes_projects_the_ticker_keyed_outcome_mapping() -> None:
    """`resolution_outcomes` yields the shape the evaluation harness scores."""
    resolutions = (
        IngestedResolution(
            market_ticker="MKT-A",
            outcome=ResolutionOutcome.YES,
            resolved_at=_RESOLVED_AT,
            source="notice",
        ),
        IngestedResolution(
            market_ticker="MKT-B",
            outcome=ResolutionOutcome.NO,
            resolved_at=_RESOLVED_AT,
            source="notice",
        ),
    )

    assert resolution_outcomes(resolutions) == {
        "MKT-A": ResolutionOutcome.YES,
        "MKT-B": ResolutionOutcome.NO,
    }


# ---------------------------------------------------------------------------
# 3. The instant -> sequence projection: both directions, exact instants.
# ---------------------------------------------------------------------------


def test_projection_maps_an_instant_to_the_first_position_at_or_after_it() -> None:
    """The projected sequence is the earliest row whose instant is >= the answer.

    Positions 1..4 sit one hour apart. A market settling at 02:00 could first
    have been known at position 3, so a forecast at position 2 predates the
    answer and one at position 3 does not.
    """
    base = datetime(2026, 3, 1, tzinfo=UTC)
    positions = [(index, base + timedelta(hours=index - 1)) for index in range(1, 5)]

    sequences = resolution_sequences_from_instants(
        positions, {"MKT-A": base + timedelta(hours=2)}
    )

    assert sequences == {"MKT-A": 3}


def test_projection_ignores_where_the_resolution_row_itself_landed() -> None:
    """A late-ingested resolution still gates on when the market *settled*.

    The resolution row lands last (position 5) but names an instant at
    position 2. The projection must return 2, not 5: keying on the ingesting
    row's own position would silently admit every forecast made between the
    settlement and the operator noticing it.
    """
    base = datetime(2026, 3, 1, tzinfo=UTC)
    positions = [
        (1, base),
        (2, base + timedelta(hours=1)),
        (3, base + timedelta(hours=2)),
        (4, base + timedelta(hours=3)),
        (5, base + timedelta(days=7)),
    ]

    sequences = resolution_sequences_from_instants(
        positions, {"MKT-A": base + timedelta(hours=1)}
    )

    assert sequences == {"MKT-A": 2}


def test_projection_places_a_future_resolution_one_past_the_ledger_head() -> None:
    """An instant after every recorded row cannot have leaked into any of them."""
    base = datetime(2026, 3, 1, tzinfo=UTC)
    positions = [(1, base), (2, base + timedelta(hours=1))]

    sequences = resolution_sequences_from_instants(
        positions, {"MKT-A": base + timedelta(days=365)}
    )

    assert sequences == {"MKT-A": 3}


def test_projection_over_no_positions_fails_closed_to_sequence_zero() -> None:
    """With no ledger positions the ordering is unprovable, so nothing is safe.

    Sequence numbers start at 1, so a projected `0` makes every forecast
    `created_sequence >= 0` -- refused rather than scored against an ordering
    nothing established.
    """
    sequences = resolution_sequences_from_instants(
        (), {"MKT-A": datetime(2026, 3, 1, tzinfo=UTC)}
    )

    assert sequences == {"MKT-A": 0}


def test_projection_refuses_a_naive_ledger_instant() -> None:
    """A ledger position carrying an offsetless instant is refused, not compared."""
    with pytest.raises(ValueError) as exc_info:
        resolution_sequences_from_instants(
            [(1, _RESOLVED_AT.replace(tzinfo=None))],
            {"MKT-A": datetime(2026, 3, 1, tzinfo=UTC)},
        )

    assert type(exc_info.value) is ValueError
    assert "ledger_instant must be timezone-aware" in str(exc_info.value)


def test_projection_refuses_a_naive_resolution_instant() -> None:
    """A resolution instant with no offset is refused, not compared."""
    with pytest.raises(ValueError) as exc_info:
        resolution_sequences_from_instants(
            [(1, datetime(2026, 3, 1, tzinfo=UTC))],
            {"MKT-A": _RESOLVED_AT.replace(tzinfo=None)},
        )

    assert type(exc_info.value) is ValueError
    assert "resolution_instant must be timezone-aware" in str(exc_info.value)


def test_projection_is_insensitive_to_the_order_positions_are_supplied_in() -> None:
    """Positions are ordered by sequence number, not by the caller's iteration.

    The projection answers "the *earliest* ledger position at or after this
    instant", which is only the earliest if the walk is in sequence order. A
    caller handing the pairs over in some other order -- a set, a dict, a
    reversed scan -- would otherwise get the first *supplied* match instead.
    """
    base = datetime(2026, 3, 1, tzinfo=UTC)
    shuffled = [
        (3, base + timedelta(hours=2)),
        (1, base),
        (4, base + timedelta(hours=3)),
        (2, base + timedelta(hours=1)),
    ]

    sequences = resolution_sequences_from_instants(
        shuffled, {"MKT-A": base + timedelta(hours=1)}
    )

    assert sequences == {"MKT-A": 2}


def test_projection_returns_no_entry_for_a_market_that_never_resolved() -> None:
    """An empty resolution mapping projects to an empty sequence mapping."""
    base = datetime(2026, 3, 1, tzinfo=UTC)

    assert resolution_sequences_from_instants([(1, base)], {}) == {}


# ---------------------------------------------------------------------------
# `parse_resolution_instant`: the CLI's own boundary, exercised directly.
# ---------------------------------------------------------------------------


def test_parse_resolution_instant_returns_the_aware_instant_it_was_given() -> None:
    """A well-formed offset-carrying instant parses to that exact moment."""
    assert parse_resolution_instant("2026-03-01T07:00:00-05:00") == _RESOLVED_AT


def test_parse_resolution_instant_refuses_a_non_iso_string() -> None:
    """A string that is not ISO-8601 is named, with the expected shape shown."""
    with pytest.raises(ValueError) as exc_info:
        parse_resolution_instant("last Tuesday")

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == (
        "resolved_at is not an ISO-8601 instant: 'last Tuesday' "
        "(expected e.g. 2026-03-01T12:00:00+00:00)"
    )


def test_parse_resolution_instant_refuses_a_parseable_but_naive_instant(
    local_timezone_utc_minus_5: None,
) -> None:
    """An offsetless instant is refused rather than read as the host's zone.

    This is the guard `MarketResolved.__post_init__` also carries, so deleting
    it left the whole suite green -- defense in depth that nothing exercised.
    Here it is exercised on its own, with the process timezone pinned five
    hours off UTC so a host that quietly read the wall clock as local time
    would produce a different instant instead of an error.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    with pytest.raises(ValueError) as exc_info:
        parse_resolution_instant("2026-03-01T12:00:00")

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value).startswith(
        "resolved_at must be timezone-aware, got naive 2026-03-01T12:00:00."
    )
