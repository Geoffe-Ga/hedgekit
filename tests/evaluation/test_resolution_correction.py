"""A wrong first `ingest-resolution` becomes correctable, on-chain (#484).

Issue #439 shipped the only path ground truth takes into the running system,
and issue #482 closed half of its failure mode: a *contradicting* re-ingest is
refused at the verb, so a mistyped second call can no longer brick the loop.
The other half stayed open. Nothing contradicts a **first** ingest that was
simply wrong, so nothing refuses it, and the ledger is append-only -- the bad
settlement stood forever and every metric folded from it was wrong.

This module pins the correction mechanism. The ledger cannot be redacted, so a
correction is not a deletion: it is a later row that **supersedes** an earlier
one, and every reader must agree on which one wins.

THE PRECEDENCE RULE, STATED ONCE

A `SettlementReversed` row supersedes exactly the row whose
`sequence_number` it names, and only that row. The superseding row wins. A
later `MarketResolved` row that names nothing never wins -- it is either an
idempotent restatement of the market's current claim or a contradiction, and
the contradiction is refused, exactly as #482 left it. "Later wins" is
therefore not a bare append-order convention: **a row only supersedes what it
explicitly claims to supersede.** PR #500's standard is that two rows
describing one event must either be payload-identical or carry an explicit,
asserted precedence; this suite asserts it in both directions
(`test_the_correction_wins_over_the_row_it_names` and
`test_a_reversal_that_names_nothing_current_is_refused_by_the_fold`).

WHY `SettlementReversed`, AND NOT A NEW WORD

`SettlementEventType.SETTLEMENT_REVERSED` already existed in the *fixture*
settlement vocabulary, disconnected from the ledger path (issue #484's own
correcting comment). The ledger event here is named after it deliberately, and
`test_a_ledger_correction_means_what_the_fixture_vocabulary_means` proves the
two paths agree rather than asserting it: the same correction, hand-expressed
in the fixture vocabulary as a `SETTLEMENT_REVERSED` followed by a corrected
`SETTLEMENT`, folds through the independent `ResolutionTracker` state machine
to the same outcome. The expectation comes from the other implementation, not
from the one under test (#422).

WHY THE ROW CARRIES THE CORRECTED CLAIM

The fixture vocabulary's reversal carries no outcome, and re-settlement is a
second event. On the ledger it is one row, because the always-on loop folds the
whole ledger on *every* tick: a market observable in a cleared-but-not-yet-
re-settled state would silently drop out of the weekly report's scoring for as
long as the operator took to type a second command, and a crash between the two
would leave it dropped forever. One row is atomic against a reader that never
stops reading.

NOT A SECOND BRICK

#482's lesson is that a fail-closed guard firing on a non-contradiction, with
no way back, is worse than the gap it closes. Every refusal here is at the
verb (`tests/test_cli_correct_resolution.py`), and a market can be corrected
again and again: `test_a_second_correction_supersedes_the_first` is the
regression test for exactly that, because a mechanism that works once is the
same permanent trap in slower motion.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from windbreak.evaluation.ingest import (
    MARKET_RESOLVED_EVENT_TYPE,
    SETTLEMENT_REVERSED_EVENT_TYPE,
    IngestedResolution,
    MarketResolved,
    ResolutionCorrection,
    SettlementReversed,
    fold_resolutions,
    ingested_resolutions_from_records,
    render_correction_lines,
    resolution_outcomes,
)
from windbreak.evaluation.resolution import (
    ResolutionOutcome,
    ResolutionTracker,
    SettlementEvent,
    SettlementEventType,
)
from windbreak.ledger.events import Event, ModeHeartbeat
from windbreak.ledger.store import LedgerRecord, SqliteLedgerStore

if TYPE_CHECKING:
    from pathlib import Path

#: The market every fold in this module corrects.
_TICKER = "MKT-A"

#: The instant the first (wrong) ingest claimed the market settled.
_WRONG_INSTANT = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)

#: The instant the correction claims it actually settled. Deliberately a
#: *different* instant from `_WRONG_INSTANT`, and by more than a rounding: a
#: correction whose instant coincided with the original would let a fold that
#: ignored the instant entirely still pass (proven trap 3).
_CORRECTED_INSTANT = datetime(2026, 3, 2, 9, 30, 0, tzinfo=UTC)

#: The component label stamped on every event here.
_COMPONENT = "operator"


def _resolved(
    *,
    market_ticker: str = _TICKER,
    outcome: ResolutionOutcome = ResolutionOutcome.NO,
    resolved_at: datetime = _WRONG_INSTANT,
    source: str = "kalshi settlement notice",
) -> MarketResolved:
    """Build one operator-ingested `MarketResolved` event.

    Args:
        market_ticker: The market that settled.
        outcome: The settled outcome claimed.
        resolved_at: The settlement instant claimed.
        source: The provenance label.

    Returns:
        The assembled event.
    """
    return MarketResolved(
        component=_COMPONENT,
        market_ticker=market_ticker,
        outcome=outcome,
        resolved_at=resolved_at,
        source=source,
    )


def _reversed(
    *,
    market_ticker: str = _TICKER,
    superseded_sequence_number: int = 1,
    outcome: ResolutionOutcome = ResolutionOutcome.YES,
    resolved_at: datetime = _CORRECTED_INSTANT,
    source: str = "kalshi settlement notice, corrected 2026-03-05",
) -> SettlementReversed:
    """Build one operator-issued `SettlementReversed` correction event.

    Args:
        market_ticker: The market being corrected.
        superseded_sequence_number: The ledger row this correction supersedes.
        outcome: The corrected outcome.
        resolved_at: The corrected settlement instant.
        source: The provenance of the correction itself.

    Returns:
        The assembled event.
    """
    return SettlementReversed(
        component=_COMPONENT,
        market_ticker=market_ticker,
        superseded_sequence_number=superseded_sequence_number,
        outcome=outcome,
        resolved_at=resolved_at,
        source=source,
    )


def _ledger(tmp_path: Path, *events: Event) -> list[LedgerRecord]:
    """Append events to a real ledger and read every row back off disk.

    Nothing here uses an in-memory list of events: the fold under test reads
    persisted rows, and a projection that only ever round-trips through Python
    objects proves nothing about what a later tick will read. The chain is
    verified bare after the appends, so a fold asserted below is always a fold
    of an intact chain.

    Args:
        tmp_path: The per-test temporary directory.
        events: The events to append, in order.

    Returns:
        Every persisted record, in ascending sequence order.
    """
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        for event in events:
            store.append(event)
        store.verify_chain()
        return store.read_all()
    finally:
        store.close()


def _correction_record(*, sequence_number: int, dropped: str) -> LedgerRecord:
    """Build a persisted correction row with one payload key removed.

    A malformed row can only reach a live ledger from outside this verb, so it
    is constructed here directly rather than appended -- the event constructor
    would refuse to build it, which is the point.

    Args:
        sequence_number: The row's ledger position, named in the error.
        dropped: The payload key to omit.

    Returns:
        The malformed :class:`LedgerRecord`.
    """
    data = {
        "market_ticker": _TICKER,
        "superseded_sequence_number": 1,
        "outcome": "yes",
        "resolved_at": "2026-03-02T09:30:00.000000Z",
        "source": "corrected",
    }
    del data[dropped]
    return LedgerRecord(
        sequence_number=sequence_number,
        event_type=SETTLEMENT_REVERSED_EVENT_TYPE,
        created_at="2026-03-05T00:00:00.000000+00:00",
        component=_COMPONENT,
        payload_json=json.dumps(
            {"component": _COMPONENT, "data": data, "schema_version": 1}
        ),
        payload_schema_version=1,
        prev_hash="0" * 64,
        event_hash="1" * 64,
    )


# ---------------------------------------------------------------------------
# 1. The event: what one correction row actually carries.
# ---------------------------------------------------------------------------


def test_a_correction_row_carries_the_superseded_row_and_the_corrected_claim() -> None:
    """The payload names the row superseded, the new claim, and its provenance.

    Acceptance criterion 1: an operator records the correction through an
    append-only event naming the superseded row's `sequence_number`, the
    corrected evidentiary fields, and a provenance label for the correction.
    """
    event = _reversed(superseded_sequence_number=7)

    assert event.event_type == SETTLEMENT_REVERSED_EVENT_TYPE
    assert event.payload == {
        "market_ticker": "MKT-A",
        "superseded_sequence_number": 7,
        "outcome": "yes",
        "resolved_at": "2026-03-02T09:30:00.000000Z",
        "source": "kalshi settlement notice, corrected 2026-03-05",
    }


def test_a_correction_is_a_different_event_type_from_a_first_ingest() -> None:
    """A reader tells a correction from a first ingest by the row's own type.

    A reversal is an extraordinary act. It must never be indistinguishable
    from an ordinary ingest on the chain, or an operator could launder a bad
    outcome by making the correction look like the original claim.
    """
    assert SETTLEMENT_REVERSED_EVENT_TYPE == "SettlementReversed"
    assert SETTLEMENT_REVERSED_EVENT_TYPE != MARKET_RESOLVED_EVENT_TYPE
    assert _resolved().event_type == MARKET_RESOLVED_EVENT_TYPE
    assert _reversed().event_type == SETTLEMENT_REVERSED_EVENT_TYPE


def test_a_correction_with_a_naive_instant_is_refused() -> None:
    """An offsetless corrected instant is refused rather than guessed."""
    with pytest.raises(ValueError) as exc_info:
        _reversed(resolved_at=datetime(2026, 3, 2, 9, 30, 0))

    assert str(exc_info.value) == (
        "resolved_at must be timezone-aware, got naive 2026-03-02T09:30:00. "
        "An offsetless instant is unprovable evidence -- reading its wall "
        "clock as the host's local time would silently shift it by the host's "
        "offset, and across a calendar-day boundary that changes the answer "
        "-- so it is refused rather than normalized against a guessed "
        "timezone."
    )


def test_a_correction_with_a_blank_source_is_refused() -> None:
    """A correction nothing attributes is refused: it is the extraordinary act."""
    with pytest.raises(ValueError) as exc_info:
        _reversed(source="   ")

    assert str(exc_info.value) == (
        "source must be a non-blank provenance label; a correction nothing "
        "attributes cannot be audited, and overturning a settled outcome is "
        "the one act on this ledger that most needs to be"
    )


def test_a_correction_with_a_blank_market_ticker_is_refused() -> None:
    """A correction naming no market could never join to a resolution."""
    with pytest.raises(ValueError) as exc_info:
        _reversed(market_ticker="  ")

    assert str(exc_info.value) == (
        "market_ticker must be a non-blank market identifier; a correction "
        "naming no market can never join to a resolution"
    )


@pytest.mark.parametrize("sequence_number", [0, -1])
def test_a_correction_naming_a_non_positive_row_is_refused(
    sequence_number: int,
) -> None:
    """Ledger sequence numbers start at 1, so 0 and below name no row at all.

    Args:
        sequence_number: The impossible sequence number under test.
    """
    with pytest.raises(ValueError) as exc_info:
        _reversed(superseded_sequence_number=sequence_number)

    assert str(exc_info.value) == (
        "superseded_sequence_number must be a positive ledger position, got "
        f"{sequence_number}; ledger sequence numbers start at 1, so this "
        "names no row at all"
    )


def test_a_correction_naming_a_boolean_row_is_refused() -> None:
    """`True` is an `int` subclass and must not masquerade as row 1."""
    with pytest.raises(TypeError) as exc_info:
        _reversed(superseded_sequence_number=True)

    assert str(exc_info.value) == (
        "superseded_sequence_number requires a non-bool int, got bool"
    )


# ---------------------------------------------------------------------------
# 2. The fold: precedence, and what a reader sees.
# ---------------------------------------------------------------------------


def test_the_correction_wins_over_the_row_it_names(tmp_path: Path) -> None:
    """The superseding row is authoritative; the superseded row stays on-chain.

    This is the precedence rule, asserted rather than assumed. Both rows are
    still on the append-only ledger -- nothing was redacted -- and the fold
    reports exactly one resolution for the market, carrying the *corrected*
    outcome, the *corrected* instant, and the *correction's* provenance.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(tmp_path, _resolved(), _reversed(superseded_sequence_number=1))

    fold = fold_resolutions(records)

    assert [record.event_type for record in records] == [
        MARKET_RESOLVED_EVENT_TYPE,
        SETTLEMENT_REVERSED_EVENT_TYPE,
    ]
    assert fold.resolutions == (
        IngestedResolution(
            market_ticker="MKT-A",
            outcome=ResolutionOutcome.YES,
            resolved_at=_CORRECTED_INSTANT,
            source="kalshi settlement notice, corrected 2026-03-05",
        ),
    )
    assert resolution_outcomes(fold.resolutions) == {"MKT-A": ResolutionOutcome.YES}


def test_the_superseded_claim_survives_the_fold_and_names_both_rows(
    tmp_path: Path,
) -> None:
    """A correction that leaves no trace is the failure mode being prevented.

    Acceptance criterion 2: the superseded claim stays visible. The fold
    carries it forward with both sequence numbers, so a reader can go back to
    the exact rows.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(tmp_path, _resolved(), _reversed(superseded_sequence_number=1))

    fold = fold_resolutions(records)

    assert fold.corrections == (
        ResolutionCorrection(
            market_ticker="MKT-A",
            superseded_sequence_number=1,
            superseded=IngestedResolution(
                market_ticker="MKT-A",
                outcome=ResolutionOutcome.NO,
                resolved_at=_WRONG_INSTANT,
                source="kalshi settlement notice",
            ),
            correction_sequence_number=2,
            corrected=IngestedResolution(
                market_ticker="MKT-A",
                outcome=ResolutionOutcome.YES,
                resolved_at=_CORRECTED_INSTANT,
                source="kalshi settlement notice, corrected 2026-03-05",
            ),
        ),
    )


def test_an_uncorrected_fold_reports_no_corrections(tmp_path: Path) -> None:
    """The negative control: without a reversal row nothing is reported corrected.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(tmp_path, _resolved())

    fold = fold_resolutions(records)

    assert fold.corrections == ()
    assert fold.resolutions[0].outcome == ResolutionOutcome.NO
    assert fold.resolutions[0].resolved_at == _WRONG_INSTANT


def test_the_legacy_fold_entry_point_still_returns_only_resolutions(
    tmp_path: Path,
) -> None:
    """`ingested_resolutions_from_records` keeps its shape and sees corrections.

    Every #439 caller reads through this function. It must keep returning the
    same tuple type *and* must not have been left behind on the uncorrected
    claim, or two readers of one ledger would disagree about what settled.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(tmp_path, _resolved(), _reversed(superseded_sequence_number=1))

    resolutions = ingested_resolutions_from_records(records)

    assert resolutions == fold_resolutions(records).resolutions
    assert resolutions[0].outcome == ResolutionOutcome.YES


def test_a_second_correction_supersedes_the_first(tmp_path: Path) -> None:
    """Correcting a correction works: the mechanism is not single-use.

    A path that can be walked exactly once is the same permanent trap as no
    path at all, arrived at one command later. The second correction names the
    *first correction's* row, because that is the row carrying the market's
    claim by then.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    third_instant = datetime(2026, 3, 3, 18, 0, 0, tzinfo=UTC)
    records = _ledger(
        tmp_path,
        _resolved(),
        _reversed(superseded_sequence_number=1),
        _reversed(
            superseded_sequence_number=2,
            outcome=ResolutionOutcome.NO,
            resolved_at=third_instant,
            source="venue support ticket 4471",
        ),
    )

    fold = fold_resolutions(records)

    assert fold.resolutions == (
        IngestedResolution(
            market_ticker="MKT-A",
            outcome=ResolutionOutcome.NO,
            resolved_at=third_instant,
            source="venue support ticket 4471",
        ),
    )
    assert [
        (correction.superseded_sequence_number, correction.correction_sequence_number)
        for correction in fold.corrections
    ] == [(1, 2), (2, 3)]


def test_a_correction_leaves_other_markets_untouched(tmp_path: Path) -> None:
    """Correction is per-market, exactly as the #482 conflict check is.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(
        tmp_path,
        _resolved(),
        _resolved(market_ticker="MKT-B", outcome=ResolutionOutcome.YES),
        _reversed(superseded_sequence_number=1),
    )

    fold = fold_resolutions(records)

    assert resolution_outcomes(fold.resolutions) == {
        "MKT-A": ResolutionOutcome.YES,
        "MKT-B": ResolutionOutcome.YES,
    }
    assert [correction.market_ticker for correction in fold.corrections] == ["MKT-A"]


def test_a_reingest_restating_the_corrected_claim_is_idempotent(
    tmp_path: Path,
) -> None:
    """After a correction the corrected claim is the one a re-ingest must match.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(
        tmp_path,
        _resolved(),
        _reversed(superseded_sequence_number=1),
        _resolved(
            outcome=ResolutionOutcome.YES,
            resolved_at=_CORRECTED_INSTANT,
            source="retyped label",
        ),
    )

    fold = fold_resolutions(records)

    assert fold.resolutions[0].outcome == ResolutionOutcome.YES
    assert (
        fold.resolutions[0].source == "kalshi settlement notice, corrected 2026-03-05"
    )


def test_a_reingest_restating_the_superseded_claim_is_refused_by_the_fold(
    tmp_path: Path,
) -> None:
    """The corrected claim cannot be quietly undone by re-ingesting the old one.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(
        tmp_path,
        _resolved(),
        _reversed(superseded_sequence_number=1),
        _resolved(),
    )

    with pytest.raises(ValueError) as exc_info:
        fold_resolutions(records)

    assert str(exc_info.value).startswith(
        "MarketResolved at sequence_number=3 contradicts an earlier resolution "
        "of market_ticker='MKT-A':"
    )


# ---------------------------------------------------------------------------
# 3. The fold's last line of defense (the verb refuses these first).
# ---------------------------------------------------------------------------


def test_a_reversal_that_names_nothing_current_is_refused_by_the_fold(
    tmp_path: Path,
) -> None:
    """A reversal naming a row that carries no current claim cannot be folded.

    This is the complement of the precedence rule: append order alone never
    supersedes anything. A row that does not name the market's current claim
    is refused rather than silently winning on recency.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(
        tmp_path,
        ModeHeartbeat(component="scheduler", mode="PAPER", beat=1),
        _resolved(),
        _reversed(superseded_sequence_number=1),
    )

    with pytest.raises(ValueError) as exc_info:
        fold_resolutions(records)

    assert str(exc_info.value) == (
        "SettlementReversed at sequence_number=3 names "
        "superseded_sequence_number=1, which does not carry the current "
        "resolution of market_ticker='MKT-A' (that is sequence_number=2). A "
        "correction supersedes exactly the row it names, so a row naming any "
        "other position is refused rather than allowed to win on append order. "
        "`windbreak correct-resolution` refuses such a call before writing "
        "anything, so a ledger carrying this row was not written by it. See "
        "docs/RUNBOOK.md, 'Correcting a wrong resolution'."
    )


def test_a_reversal_of_an_unresolved_market_is_refused_by_the_fold(
    tmp_path: Path,
) -> None:
    """There is nothing to supersede until the market has a claim at all.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(tmp_path, _reversed(superseded_sequence_number=1))

    with pytest.raises(ValueError) as exc_info:
        fold_resolutions(records)

    assert str(exc_info.value).startswith(
        "SettlementReversed at sequence_number=1 names "
        "superseded_sequence_number=1, which does not carry the current "
        "resolution of market_ticker='MKT-A' (that market has no resolution "
        "on this ledger)."
    )


def test_a_second_reversal_naming_the_already_superseded_row_is_refused(
    tmp_path: Path,
) -> None:
    """A row that has itself been superseded can never be superseded again.

    Acceptance criterion 3's second clause, at the fold. The recovery is not
    "give up": it is to name the correction row instead, which
    `test_a_second_correction_supersedes_the_first` proves works.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(
        tmp_path,
        _resolved(),
        _reversed(superseded_sequence_number=1),
        _reversed(superseded_sequence_number=1, outcome=ResolutionOutcome.NO),
    )

    with pytest.raises(ValueError) as exc_info:
        fold_resolutions(records)

    assert str(exc_info.value).startswith(
        "SettlementReversed at sequence_number=3 names "
        "superseded_sequence_number=1, which does not carry the current "
        "resolution of market_ticker='MKT-A' (that is sequence_number=2)."
    )


@pytest.mark.parametrize(
    "dropped",
    [
        "market_ticker",
        "superseded_sequence_number",
        "outcome",
        "resolved_at",
        "source",
    ],
)
def test_a_correction_payload_missing_a_key_names_it_and_the_row(dropped: str) -> None:
    """A malformed persisted correction is locatable, never a bare `KeyError`.

    Args:
        dropped: The payload key removed before the fold reads the row.
    """
    record = _correction_record(sequence_number=4, dropped=dropped)

    with pytest.raises(ValueError) as exc_info:
        fold_resolutions([record])

    assert str(exc_info.value) == (
        f"SettlementReversed payload at sequence_number=4 is missing {dropped!r}"
    )


# ---------------------------------------------------------------------------
# 4. What the report shows, and what the fixture vocabulary agrees it means.
# ---------------------------------------------------------------------------


def test_the_rendered_correction_line_names_both_claims_and_both_rows(
    tmp_path: Path,
) -> None:
    """The report line carries exact values, not merely "something changed".

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(tmp_path, _resolved(), _reversed(superseded_sequence_number=1))

    lines = render_correction_lines(fold_resolutions(records).corrections)

    assert lines == (
        "RESOLUTION_CORRECTED MKT-A superseded_sequence_number=1 "
        "outcome='no' resolved_at=2026-03-01T12:00:00.000000Z -> "
        "correction_sequence_number=2 outcome='yes' "
        "resolved_at=2026-03-02T09:30:00.000000Z "
        "source='kalshi settlement notice, corrected 2026-03-05'"
    )


def test_no_corrections_render_no_lines_at_all() -> None:
    """An empty correction ledger renders nothing, so the section stays absent.

    A `## Resolution corrections` heading printed on every report -- almost
    always empty -- trains an operator to skip it. Its *presence* is the
    signal, so it is rendered only when a correction exists.
    """
    assert render_correction_lines(()) is None


def test_a_ledger_correction_means_what_the_fixture_vocabulary_means(
    tmp_path: Path,
) -> None:
    """The two paths do not drift into two meanings of "reversal".

    The same correction is expressed twice: once as the ledger rows this issue
    ships, and once by hand in the pre-existing *fixture* settlement vocabulary
    -- a `SETTLEMENT`, then a `SETTLEMENT_REVERSED` clearing it, then a
    corrected `SETTLEMENT`. `ResolutionTracker` is an independent state machine
    that knows nothing about this issue, so it is a real oracle: the
    expectation comes from the other implementation, not from the one under
    test.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    records = _ledger(tmp_path, _resolved(), _reversed(superseded_sequence_number=1))
    fixture_stream = (
        SettlementEvent(
            sequence_number=1,
            event_type=SettlementEventType.SETTLEMENT,
            market_ticker=_TICKER,
            outcome=ResolutionOutcome.NO,
        ),
        SettlementEvent(
            sequence_number=2,
            event_type=SettlementEventType.SETTLEMENT_REVERSED,
            market_ticker=_TICKER,
            outcome=None,
        ),
        SettlementEvent(
            sequence_number=3,
            event_type=SettlementEventType.SETTLEMENT,
            market_ticker=_TICKER,
            outcome=ResolutionOutcome.YES,
        ),
    )

    tracker = ResolutionTracker.from_ledger(fixture_stream)

    assert tracker.resolved_outcomes() == {"MKT-A": ResolutionOutcome.YES}
    assert resolution_outcomes(fold_resolutions(records).resolutions) == (
        tracker.resolved_outcomes()
    )
    assert tracker.get(_TICKER).reversal_count == 1
