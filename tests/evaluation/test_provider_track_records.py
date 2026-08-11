"""The producer that finally writes `provider-track-records.json` (issue #440).

`_build_provider_gate` reads that artifact and never recomputes it, and before
this module *nothing in the repository wrote it*: a whole-tree grep found the
filename constant and two documents describing the read. The docs called the
empty case a transient bootstrap, but with no producer it was terminal --
every provider unproven forever, and the two bars
(`provider_gate.min_resolved` / `provider_gate.min_brier_skill_ppm`)
unreachable by construction.

These tests pin the producer's fold. The expectation never comes from the
thing being checked (#422): every asserted count and Brier skill is computed
here from the forecast's own `probability_ppm`, its recorded baseline price,
and the outcome an operator ingested -- three inputs, none of which is the
gate's prior state or the artifact being written.

Both directions of the danger are pinned, because they are not symmetric. A
provider that has *not* earned a record must stay held; an artifact that
materialised with optimistic defaults would convert a permanently-closed gate
into a permanently-open one, which is strictly worse than the bug it fixes. So
there are tests for: no record at all, a real-but-insufficient record (the only
way the *bars* themselves can be tested -- deleting the artifact leaves a
provider unproven whatever the bar says), and a sufficient one.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from windbreak.evaluation.ingest import MarketResolved
from windbreak.evaluation.resolution import ResolutionOutcome
from windbreak.evaluation.track_records import (
    PROVIDER_TRACK_RECORD_FILENAME,
    REQUIRED_FORECAST_PAYLOAD_KEYS,
    provider_track_records,
    render_track_record_document,
    write_provider_track_records,
)
from windbreak.forecast.pipeline import (
    VOTE_OUTCOME_ABSTAINED,
    VOTE_OUTCOME_DISCARDED,
    VOTE_OUTCOME_VOTED,
)
from windbreak.forecast.providers.track_record import (
    ProviderTrackRecord,
    parse_track_records,
)
from windbreak.ledger.events import ConfigLoaded, ForecastCreated, ProviderVoteRecorded
from windbreak.ledger.store import SqliteLedgerStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from windbreak.ledger.events import Event
    from windbreak.ledger.store import LedgerRecord

#: The component label stamped on every event this suite ledgers.
_COMPONENT = "scheduler"

#: The instant the scripted ledger clock starts from; each append steps one
#: second, so a settlement instant can be aimed at an exact row.
_BASE = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)

#: The forecast probability every scored forecast here carries, in ppm.
_PROBABILITY_PPM = 250_000

#: The recorded baseline executable price, in pips, every scored forecast is
#: measured against. 4600 pips is 460_000 ppm of probability.
_BASELINE_PIPS = 4600

#: Exact ppm-per-pip factor, restated from
#: :data:`windbreak.evaluation.metrics.BASELINE_PPM_PER_PIP` so the expectation
#: below is arithmetic done here rather than a value read back off the metric.
_PPM_PER_PIP = 100

#: One whole probability in ppm.
_PPM_SCALE = 1_000_000

#: The two providers the fixtures vote with.
_PROVEN_PROVIDER = "openai"
_OTHER_PROVIDER = "anthropic"


def _brier_skill_of_one_no_market_ppm() -> int:
    """Compute the exact Brier skill of one `_PROBABILITY_PPM` forecast on NO.

    Derived from the forecast's probability, its recorded baseline price and
    the ingested outcome -- never from the artifact or the gate this suite
    checks. Brier skill is ``1 - sum(forecast_term) / sum(baseline_term)``,
    lifted to ppm and floored so a marginal edge never rounds up.

    Returns:
        The Brier skill of a single such forecast, in ppm.
    """
    forecast_term = (_PROBABILITY_PPM - 0) ** 2
    baseline_term = (_BASELINE_PIPS * _PPM_PER_PIP - 0) ** 2
    return (baseline_term - forecast_term) * _PPM_SCALE // baseline_term


#: The exact skill a single scored forecast earns; ``704631`` ppm, computed
#: above from the three inputs rather than restated as a literal.
_ONE_FORECAST_SKILL_PPM = _brier_skill_of_one_no_market_ppm()


def _stepping_instants(count: int) -> Iterator[datetime]:
    """Yield `count` one-second-apart instants from :data:`_BASE`.

    Args:
        count: How many instants to yield.

    Yields:
        Each instant in ascending order, one per ledger append.
    """
    for step in range(count):
        yield _BASE + timedelta(seconds=step)


def _instant_at(row: int) -> datetime:
    """Return the instant the scripted clock stamps on the ``row``-th append.

    Args:
        row: The 1-based append position.

    Returns:
        The `created_at` instant that append receives.
    """
    return _BASE + timedelta(seconds=row - 1)


class _ScriptedLedger:
    """A ledger whose `created_at` stamps follow a fixed one-per-append script.

    The temporal projection the producer applies reads each row's `created_at`,
    so a suite that cannot choose those stamps cannot distinguish "the
    settlement instant excluded this forecast" from "the append order happened
    to exclude it".
    """

    def __init__(self, db_path: Path) -> None:
        """Open a store stamping ascending one-second instants.

        Args:
            db_path: The SQLite database path to open or create.
        """
        self._clock = _stepping_instants(count=512)
        self.store = SqliteLedgerStore(db_path, now=lambda: next(self._clock))

    def append(self, event: Event) -> int:
        """Append one event, stamping the script's next instant.

        Args:
            event: The event to persist.

        Returns:
            The sequence number assigned to the new record.
        """
        return self.store.append(event)

    def read_all(self) -> list[LedgerRecord]:
        """Return every persisted record in ascending sequence order.

        Returns:
            The full ledger read.
        """
        return self.store.read_all()

    def close(self) -> None:
        """Close the underlying store."""
        self.store.close()


def _config_loaded() -> ConfigLoaded:
    """Build the deployment marker every real run writes as its first row.

    Returns:
        A `ConfigLoaded` event.
    """
    return ConfigLoaded(component=_COMPONENT, config_hash="deadbeef", diff={})


def _forecast(
    *,
    forecast_id: str,
    market_ticker: str,
    probability_ppm: int = _PROBABILITY_PPM,
    baseline_pips: int = _BASELINE_PIPS,
) -> ForecastCreated:
    """Build one v2-shaped `ForecastCreated` event.

    Args:
        forecast_id: The forecast's deterministic id.
        market_ticker: The market the forecast is for.
        probability_ppm: The forecast probability, in ppm.
        baseline_pips: The recorded baseline executable price, in pips.

    Returns:
        The assembled `ForecastCreated` event.
    """
    return ForecastCreated(
        component=_COMPONENT,
        forecast_id=forecast_id,
        market_ticker=market_ticker,
        probability_ppm=probability_ppm,
        eligible_for_live=False,
        abstention_reason=None,
        research_cost_micros=1_000_000,
        market_price_baseline_pips=baseline_pips,
    )


def _vote(
    *,
    forecast_id: str,
    market_ticker: str,
    provider: str,
    outcome: str = VOTE_OUTCOME_VOTED,
) -> ProviderVoteRecorded:
    """Build one `ProviderVoteRecorded` row attributing a forecast to a provider.

    Args:
        forecast_id: The forecast the vote belongs to.
        market_ticker: The forecast's market.
        provider: The provider that cast the vote.
        outcome: The vote outcome token.

    Returns:
        The assembled `ProviderVoteRecorded` event.
    """
    return ProviderVoteRecorded(
        component=_COMPONENT,
        forecast_id=forecast_id,
        market_ticker=market_ticker,
        provider=provider,
        model_version="v1",
        vote_index=0,
        cost_micros=1_000,
        outcome=outcome,
        failure_code="",
    )


def _resolved(
    *, market_ticker: str, resolved_at: datetime, outcome: ResolutionOutcome
) -> MarketResolved:
    """Build one operator-ingested `MarketResolved` event.

    Args:
        market_ticker: The market that settled.
        resolved_at: The instant it settled.
        outcome: The settled ground-truth outcome.

    Returns:
        The assembled `MarketResolved` event.
    """
    return MarketResolved(
        component="cli",
        market_ticker=market_ticker,
        outcome=outcome,
        resolved_at=resolved_at,
        source="test settlement notice",
    )


def _one_scored_forecast_ledger(
    tmp_path: Path,
    *,
    voters: tuple[str, ...] = (_PROVEN_PROVIDER,),
    vote_outcomes: tuple[str, ...] = (VOTE_OUTCOME_VOTED,),
) -> list[LedgerRecord]:
    """Ledger one forecast, its votes, and the settlement that scores it.

    Row order is `ConfigLoaded` (the deployment marker), `ForecastCreated`,
    one `ProviderVoteRecorded` per voter, then `MarketResolved` -- the shape a
    real run plus one `windbreak ingest-resolution` call leaves behind.

    Args:
        tmp_path: The directory the ledger database is created in.
        voters: The providers that cast a vote on the forecast.
        vote_outcomes: Each voter's vote outcome, positionally paired.

    Returns:
        The full ledger read.
    """
    ledger = _ScriptedLedger(tmp_path / "ledger.db")
    try:
        ledger.append(_config_loaded())
        ledger.append(_forecast(forecast_id="f1", market_ticker="M1"))
        for provider, outcome in zip(voters, vote_outcomes, strict=True):
            ledger.append(
                _vote(
                    forecast_id="f1",
                    market_ticker="M1",
                    provider=provider,
                    outcome=outcome,
                )
            )
        settlement_row = 3 + len(voters)
        ledger.append(
            _resolved(
                market_ticker="M1",
                resolved_at=_instant_at(settlement_row),
                outcome=ResolutionOutcome.NO,
            )
        )
        return ledger.read_all()
    finally:
        ledger.close()


def test_a_voting_provider_earns_the_exact_count_and_skill_its_forecast_scored(
    tmp_path: Path,
) -> None:
    """One resolved forecast yields one record with hand-computed values."""
    records = provider_track_records(_one_scored_forecast_ledger(tmp_path))
    assert records == (
        ProviderTrackRecord(
            provider=_PROVEN_PROVIDER,
            resolved_count=1,
            brier_skill_ppm=_ONE_FORECAST_SKILL_PPM,
        ),
    )


def test_a_provider_whose_market_never_settled_earns_no_record(
    tmp_path: Path,
) -> None:
    """Removing the settlement row removes the record entirely (the negative)."""
    with_settlement = _one_scored_forecast_ledger(tmp_path)
    without_settlement = [
        record for record in with_settlement if record.event_type != "MarketResolved"
    ]
    assert provider_track_records(with_settlement) != ()
    assert provider_track_records(without_settlement) == ()


def test_a_provider_that_abstained_is_not_credited_with_that_forecast(
    tmp_path: Path,
) -> None:
    """An abstaining voter earns nothing while the voting one earns the record."""
    records = provider_track_records(
        _one_scored_forecast_ledger(
            tmp_path,
            voters=(_PROVEN_PROVIDER, _OTHER_PROVIDER),
            vote_outcomes=(VOTE_OUTCOME_VOTED, VOTE_OUTCOME_ABSTAINED),
        )
    )
    assert tuple(record.provider for record in records) == (_PROVEN_PROVIDER,)


def test_a_provider_whose_vote_was_discarded_is_not_credited_with_that_forecast(
    tmp_path: Path,
) -> None:
    """A discarded vote never reached aggregation, so it backs nothing."""
    records = provider_track_records(
        _one_scored_forecast_ledger(
            tmp_path,
            voters=(_PROVEN_PROVIDER, _OTHER_PROVIDER),
            vote_outcomes=(VOTE_OUTCOME_VOTED, VOTE_OUTCOME_DISCARDED),
        )
    )
    assert tuple(record.provider for record in records) == (_PROVEN_PROVIDER,)


def test_a_forecast_created_after_its_market_settled_earns_nothing(
    tmp_path: Path,
) -> None:
    """A backdated forecast could have peeked, so it credits no provider."""
    ledger = _ScriptedLedger(tmp_path / "ledger.db")
    try:
        ledger.append(_config_loaded())
        # The market settled at row 2's instant, so the row-3 forecast that
        # follows it is at-or-after the settlement position: BACKDATED.
        ledger.append(_forecast(forecast_id="f0", market_ticker="M0"))
        ledger.append(_forecast(forecast_id="f1", market_ticker="M1"))
        ledger.append(
            _vote(forecast_id="f1", market_ticker="M1", provider=_PROVEN_PROVIDER)
        )
        ledger.append(
            _resolved(
                market_ticker="M1",
                resolved_at=_instant_at(2),
                outcome=ResolutionOutcome.NO,
            )
        )
        records = ledger.read_all()
    finally:
        ledger.close()
    assert provider_track_records(records) == ()


def test_a_forecast_at_or_before_the_deployment_marker_earns_nothing(
    tmp_path: Path,
) -> None:
    """A pre-deployment forecast is not evidence the running system produced."""
    ledger = _ScriptedLedger(tmp_path / "ledger.db")
    try:
        ledger.append(_forecast(forecast_id="f1", market_ticker="M1"))
        ledger.append(
            _vote(forecast_id="f1", market_ticker="M1", provider=_PROVEN_PROVIDER)
        )
        # The deployment marker lands *after* the forecast, so the forecast is
        # at-or-before deployment and is refused.
        ledger.append(_config_loaded())
        ledger.append(
            _resolved(
                market_ticker="M1",
                resolved_at=_instant_at(4),
                outcome=ResolutionOutcome.NO,
            )
        )
        records = ledger.read_all()
    finally:
        ledger.close()
    assert provider_track_records(records) == ()


def test_a_ledger_with_no_deployment_marker_credits_nobody(
    tmp_path: Path,
) -> None:
    """With no `ConfigLoaded` row the fold cannot date deployment, so it holds.

    A ledger carrying no deployment marker cannot prove *when* the system it
    describes went live, so every forecast on it might predate the code being
    scored. The fold answers that by gating against the ledger's own head, which
    is at-or-after every forecast in it -- refusing them all. The alternative,
    admitting them on trust, is the fail-open shape: it would let a ledger
    assembled by hand promote a provider on forecasts no deployment ever made.
    """
    ledger = _ScriptedLedger(tmp_path / "ledger.db")
    try:
        ledger.append(_forecast(forecast_id="f1", market_ticker="M1"))
        ledger.append(
            _vote(forecast_id="f1", market_ticker="M1", provider=_PROVEN_PROVIDER)
        )
        ledger.append(
            _resolved(
                market_ticker="M1",
                resolved_at=_instant_at(3),
                outcome=ResolutionOutcome.NO,
            )
        )
        records = ledger.read_all()
    finally:
        ledger.close()
    assert all(record.event_type != "ConfigLoaded" for record in records)
    assert provider_track_records(records) == ()


def test_a_restart_does_not_retroactively_void_the_evidence_before_it(
    tmp_path: Path,
) -> None:
    """Deployment is the *earliest* marker, so a restart keeps prior evidence.

    A long-running loop writes one `ConfigLoaded` row per start, so a ledger
    that has been restarted carries several. Dating deployment from the latest
    of them would move the pre-deployment cutoff forward on every restart,
    silently voiding every forecast made before it -- a provider would lose its
    earned track record for no reason but an operator bouncing the process.
    Taking the earliest keeps the cutoff where the evidence actually begins.
    """
    ledger = _ScriptedLedger(tmp_path / "ledger.db")
    try:
        ledger.append(_config_loaded())
        ledger.append(_forecast(forecast_id="f1", market_ticker="M1"))
        ledger.append(
            _vote(forecast_id="f1", market_ticker="M1", provider=_PROVEN_PROVIDER)
        )
        # The loop is restarted, writing a second deployment marker *after* the
        # forecast that has already been made.
        ledger.append(_config_loaded())
        ledger.append(
            _resolved(
                market_ticker="M1",
                resolved_at=_instant_at(5),
                outcome=ResolutionOutcome.NO,
            )
        )
        records = ledger.read_all()
    finally:
        ledger.close()
    markers = [record for record in records if record.event_type == "ConfigLoaded"]
    assert len(markers) == 2
    assert provider_track_records(records) == (
        ProviderTrackRecord(
            provider=_PROVEN_PROVIDER,
            resolved_count=1,
            brier_skill_ppm=_ONE_FORECAST_SKILL_PPM,
        ),
    )


def test_two_forecasts_on_one_market_count_once_not_twice(
    tmp_path: Path,
) -> None:
    """A re-forecast market is one resolved observation, not two.

    A loop that revisits a market produces several `ForecastCreated` rows for
    it, and exactly one of them settles -- the market resolves once. Counting
    each row would let a provider clear `min_resolved` by re-forecasting a
    handful of markets rather than by being measured against many, which is the
    fail-open direction the bar exists to prevent. The producer resolves the
    headline observation window (`LATEST_BEFORE_CLOSE`) first, so each market
    contributes its single last-before-close observation.
    """
    ledger = _ScriptedLedger(tmp_path / "ledger.db")
    try:
        ledger.append(_config_loaded())
        for forecast_id in ("f1", "f2"):
            ledger.append(_forecast(forecast_id=forecast_id, market_ticker="M1"))
            ledger.append(
                _vote(
                    forecast_id=forecast_id,
                    market_ticker="M1",
                    provider=_PROVEN_PROVIDER,
                )
            )
        ledger.append(
            _resolved(
                market_ticker="M1",
                resolved_at=_instant_at(6),
                outcome=ResolutionOutcome.NO,
            )
        )
        records = ledger.read_all()
    finally:
        ledger.close()
    forecasts = [record for record in records if record.event_type == "ForecastCreated"]
    assert len(forecasts) == 2
    assert provider_track_records(records) == (
        ProviderTrackRecord(
            provider=_PROVEN_PROVIDER,
            resolved_count=1,
            brier_skill_ppm=_ONE_FORECAST_SKILL_PPM,
        ),
    )


def test_an_unparseable_created_at_is_refused_by_row_rather_than_guessed(
    tmp_path: Path,
) -> None:
    """A corrupt timestamp raises naming its row, never a silently dropped one.

    The temporal projection that decides which forecasts predate a settlement is
    built from every row's `created_at`. A row whose stamp cannot be parsed
    therefore makes the ordering unknowable, and skipping it would quietly move
    every later forecast's position -- possibly admitting a backdated one. The
    fold refuses the whole ledger instead, naming the row so it is locatable.
    """
    records = _one_scored_forecast_ledger(tmp_path)
    corrupted = [
        dataclasses.replace(record, created_at="not-an-instant")
        if record.sequence_number == 2
        else record
        for record in records
    ]
    with pytest.raises(ValueError) as caught:
        provider_track_records(corrupted)
    assert type(caught.value) is ValueError
    assert str(caught.value) == (
        "ledger row at sequence_number=2 carries an unparseable created_at: "
        "'not-an-instant'"
    )


def test_a_provider_whose_baseline_made_no_error_earns_no_record(
    tmp_path: Path,
) -> None:
    """An undefined skill is omitted, never written as a fabricated zero."""
    ledger = _ScriptedLedger(tmp_path / "ledger.db")
    try:
        ledger.append(_config_loaded())
        # A 0-pip baseline on a market that settled NO makes the baseline's
        # Brier term zero, so the skill ratio has no denominator.
        ledger.append(_forecast(forecast_id="f1", market_ticker="M1", baseline_pips=0))
        ledger.append(
            _vote(forecast_id="f1", market_ticker="M1", provider=_PROVEN_PROVIDER)
        )
        ledger.append(
            _resolved(
                market_ticker="M1",
                resolved_at=_instant_at(4),
                outcome=ResolutionOutcome.NO,
            )
        )
        records = ledger.read_all()
    finally:
        ledger.close()
    assert provider_track_records(records) == ()


def test_an_unsettled_market_does_not_inflate_the_resolved_count(
    tmp_path: Path,
) -> None:
    """`resolved_count` counts settled forecasts only -- the fail-open direction.

    A provider backing three forecasts of which one has settled must record
    ``1``, not ``3``: `min_resolved` exists to demand *resolved* evidence, and a
    count inflated by open markets would promote a provider on forecasts nothing
    has scored. What keeps the count honest is that the producer takes it from
    the forecasts :func:`~windbreak.evaluation.registry.gate_evaluation_inputs`
    *admitted*, and the temporal gate rejects an unsettled forecast as
    ``UNRESOLVED`` before it is ever counted. That is the same set
    :func:`~windbreak.evaluation.metrics.brier_skill` scores, so the count and
    the skill published beside it can never describe different forecasts.

    This pins the coupling rather than the arithmetic: a producer that counted
    ``EvaluationInputs.forecasts`` before gating -- or scored the gated set but
    counted the ungated one -- would report ``3`` here while publishing a skill
    computed from one forecast.
    """
    ledger = _ScriptedLedger(tmp_path / "ledger.db")
    try:
        ledger.append(_config_loaded())
        for index in (0, 1, 2):
            ledger.append(_forecast(forecast_id=f"f{index}", market_ticker=f"M{index}"))
            ledger.append(
                _vote(
                    forecast_id=f"f{index}",
                    market_ticker=f"M{index}",
                    provider=_PROVEN_PROVIDER,
                )
            )
        # Only M0 ever settles; M1 and M2 are still open.
        ledger.append(
            _resolved(
                market_ticker="M0",
                resolved_at=_instant_at(8),
                outcome=ResolutionOutcome.NO,
            )
        )
        records = ledger.read_all()
    finally:
        ledger.close()
    assert provider_track_records(records) == (
        ProviderTrackRecord(
            provider=_PROVEN_PROVIDER,
            resolved_count=1,
            brier_skill_ppm=_ONE_FORECAST_SKILL_PPM,
        ),
    )


def test_the_rendered_document_reads_back_through_the_gate_s_own_parser(
    tmp_path: Path,
) -> None:
    """What the producer writes is exactly what `parse_track_records` accepts."""
    records = provider_track_records(_one_scored_forecast_ledger(tmp_path))
    parsed = parse_track_records(render_track_record_document(records))
    assert parsed == {record.provider: record for record in records}


def test_the_rendered_entry_keys_are_derived_from_the_record_s_own_fields(
    tmp_path: Path,
) -> None:
    """Every `ProviderTrackRecord` field is emitted; none is hand-restated."""
    records = provider_track_records(_one_scored_forecast_ledger(tmp_path))
    assert records != ()
    document = json.loads(render_track_record_document(records))
    expected_keys = {field.name for field in dataclasses.fields(ProviderTrackRecord)}
    assert expected_keys == {"provider", "resolved_count", "brier_skill_ppm"}
    for entry in document.values():
        assert set(entry) == expected_keys


def test_an_empty_fold_renders_an_empty_document_not_an_absent_one(
    tmp_path: Path,
) -> None:
    """Nothing proven is written as `{}` -- a pass that ran and found nothing."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    written = write_provider_track_records(report_dir, [])
    assert written == report_dir / PROVIDER_TRACK_RECORD_FILENAME
    assert json.loads(written.read_text(encoding="utf-8")) == {}


def test_the_writer_replaces_a_stale_artifact_rather_than_merging_into_it(
    tmp_path: Path,
) -> None:
    """A provider that no longer qualifies disappears from the artifact."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    stale = report_dir / PROVIDER_TRACK_RECORD_FILENAME
    stale.write_text(
        json.dumps({"ghost": {"resolved_count": 9_999, "brier_skill_ppm": 999_999}}),
        encoding="utf-8",
    )
    write_provider_track_records(
        report_dir, provider_track_records(_one_scored_forecast_ledger(tmp_path))
    )
    assert set(json.loads(stale.read_text(encoding="utf-8"))) == {_PROVEN_PROVIDER}


def test_the_forecast_payload_keys_the_fold_reads_are_the_ones_the_event_writes() -> (
    None
):
    """The fold's required payload keys exist on a real `ForecastCreated`.

    Guards against the hand-restated-key drift that a fold over a persisted
    payload invites: if the event renames a leaf, this fails rather than the
    fold silently reading `None`.
    """
    payload = _forecast(forecast_id="f1", market_ticker="M1").payload
    assert REQUIRED_FORECAST_PAYLOAD_KEYS != ()
    assert set(REQUIRED_FORECAST_PAYLOAD_KEYS) <= set(payload)


def test_a_legacy_forecast_row_missing_a_v2_key_fails_closed_by_name(
    tmp_path: Path,
) -> None:
    """A pre-#188 row raises naming the key, never a silent zero baseline."""
    records = _one_scored_forecast_ledger(tmp_path)
    stripped: list[LedgerRecord] = []
    for record in records:
        if record.event_type != "ForecastCreated":
            stripped.append(record)
            continue
        envelope = json.loads(record.payload_json)
        del envelope["data"]["market_price_baseline_pips"]
        stripped.append(dataclasses.replace(record, payload_json=json.dumps(envelope)))
    with pytest.raises(ValueError) as caught:
        provider_track_records(stripped)
    assert type(caught.value) is ValueError
    assert str(caught.value) == (
        "ForecastCreated payload at sequence_number=2 is missing "
        "'market_price_baseline_pips'"
    )
