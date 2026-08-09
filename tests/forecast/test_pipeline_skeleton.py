"""Tests for windbreak.forecast.pipeline (issue #22): the SPEC S8.2 stub wiring.

Pins the four issue-mandated pipeline behaviors: (a) a schema-valid
`ForecastRecord` comes out the other end, (b) identical inputs produce
byte-identical output, (c) the output record is truly immutable, and (d)
replaying over a `ForbiddenLiveTransport` with a fully-populated cassette
completes purely from recorded responses while an empty cassette fails closed
with `CassetteMissError` -- never a live fallback. `windbreak/forecast/` does
not exist yet, so importing `windbreak.forecast.pipeline` fails collection with
`ModuleNotFoundError: No module named 'windbreak.forecast'` -- the expected
Gate 1 RED state for issue #22.

Also pins the issue #184 vote-parsing seam directly on `collect_model_votes`:
vote probabilities must come from the *response* (a structured, integer-ppm
JSON vote), never from `baseline ± fixed offset`; a response whose
`probability_ppm` is not a true integer is discarded and ledgered exactly
like an injection-tainted response; and an explicit `ensemble` override
threads its members' provenance onto the resulting votes. Until issue #184
lands, `collect_model_votes` has no `ensemble` keyword, `windbreak.forecast.
sanitize` has no `RESPONSE_FAILURE_NON_INTEGER_PROBABILITY` constant, and
`windbreak.config.schema` has no `EnsembleMemberConfig` -- so the three tests
in the "vote probability comes from the response" section below import those
two new names *locally* (deferred to call time, mirroring
`tests/forecast/conftest.py`'s "Sandbox-transport fixture choice" pattern) so
this module keeps collecting cleanly for every pre-existing test above; only
those three new tests themselves fail, with an `ImportError` naming the
missing symbol or (once the symbols exist) a `TypeError` on the still-missing
`ensemble=` keyword -- the expected Gate 1 RED state for issue #184.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.cassettes import (
    CassetteMissError,
    ForbiddenLiveTransport,
    LiveCallForbiddenError,
    LlmRequest,
    RecordingCassette,
    ReplayCassette,
)
from windbreak.forecast.pipeline import (
    FORECAST_OUTPUT_DISCARDED_EVENT,
    InMemoryForecastLedger,
    aggregate_median,
    apply_calibration_map,
    build_forecast_record,
    collect_model_votes,
    run_pipeline,
    shrink_toward_baseline,
)
from windbreak.forecast.records import forecast_record_to_payload

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

    #: `make_fake_vote_transport` (see tests/forecast/conftest.py) is a factory
    #: for `FakeVoteTransport`, a network-free `LlmTransport` double defined in
    #: the conftest module (not part of the `windbreak` package under test), so
    #: it is typed structurally here rather than imported by name. It also
    #: accepts an optional `responses` tuple override, hence `Callable[...]`
    #: rather than the no-argument `Callable[[], object]`.
    FakeVoteTransportFactory = Callable[..., object]


# --- (a) schema-valid record ------------------------------------------------------


def test_run_pipeline_produces_schema_valid_forecast_record(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """A full pipeline run yields a ForecastRecord that satisfies its invariants."""
    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
    )

    assert record.triage_stage == "full"
    assert 0 <= record.probability_ppm <= 1_000_000
    assert record.ci_low_ppm <= record.probability_ppm <= record.ci_high_ppm
    assert record.market_price_baseline_pips == baseline.price_pips
    assert record.baseline_quote_snapshot_id == baseline.snapshot_id
    assert len(record.model_votes) == 3
    assert all(vote.response_fingerprint for vote in record.model_votes)


# --- (b) byte-determinism ---------------------------------------------------------


def test_run_pipeline_is_byte_deterministic_for_identical_inputs(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """Two runs with identical inputs produce `==` records and identical JSON."""
    record_a = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
    )
    record_b = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
    )

    assert record_a == record_b
    payload_a = json.dumps(forecast_record_to_payload(record_a), sort_keys=True)
    payload_b = json.dumps(forecast_record_to_payload(record_b), sort_keys=True)
    assert payload_a == payload_b


# --- (c) mutation raises -----------------------------------------------------------


def test_run_pipeline_output_is_immutable(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """Mutating any field of the real pipeline's output record raises."""
    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
    )

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        record.probability_ppm = 0


# --- (d) replay over ForbiddenLiveTransport + fail-closed empty cassette ---------


def test_replay_over_forbidden_live_transport_completes_from_recording(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
    tmp_path: Path,
) -> None:
    """Recording with a fake transport, then replaying with a `ReplayCassette`,
    completes purely from the cassette and matches the original recording
    byte-for-byte -- the structural proof that replay never reaches a network.
    """
    cassette_path = tmp_path / "recorded_votes.json"
    recorder = RecordingCassette(
        transport=make_fake_vote_transport(), path=cassette_path
    )
    recorded = run_pipeline(
        market,
        baseline,
        transport=recorder,
        created_at=created_at,
        research_tools=research_tools,
    )

    replay = ReplayCassette.from_path(cassette_path)
    replayed = run_pipeline(
        market,
        baseline,
        transport=replay,
        created_at=created_at,
        research_tools=research_tools,
    )

    assert replayed == recorded

    # ForbiddenLiveTransport is constructed here only to prove, in isolation,
    # that if the run above had ever fallen through to a live call it would
    # have blown up loudly. `replay` above never delegates to a transport at
    # all -- `replayed` completing is itself the proof no stage reached the
    # network.
    with pytest.raises(LiveCallForbiddenError):
        ForbiddenLiveTransport().complete(
            LlmRequest(provider="probe", model_version="probe", prompt="probe")
        )


def test_run_pipeline_with_live_transport_directly_raises_forbidden_error(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """Wiring `ForbiddenLiveTransport` in directly (no cassette at all) still
    fails closed -- proving `collect_model_votes` really does call
    `transport.complete` rather than silently succeeding without it.
    """
    with pytest.raises(LiveCallForbiddenError):
        run_pipeline(
            market,
            baseline,
            transport=ForbiddenLiveTransport(),
            created_at=created_at,
            research_tools=research_tools,
        )


def test_run_pipeline_with_empty_cassette_raises_cassette_miss_error(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
    tmp_path: Path,
) -> None:
    """An empty cassette fails closed with `CassetteMissError` -- never a live
    fallback, per the fail-closed design contract.
    """
    empty_path = tmp_path / "empty_votes.json"
    empty_path.write_text("{}", encoding="utf-8")
    empty_cassette = ReplayCassette.from_path(empty_path)

    with pytest.raises(CassetteMissError):
        run_pipeline(
            market,
            baseline,
            transport=empty_cassette,
            created_at=created_at,
            research_tools=research_tools,
        )


def test_build_forecast_record_incoherent_and_eligible_raises_value_error(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
) -> None:
    """A non-None `coherence_sum` sets `coherence_flag=True` while
    `build_forecast_record` still passes `eligible_for_live=True` -- an
    invalid combination `ForecastRecord` must reject (#25).
    """
    votes = collect_model_votes(market, baseline, transport=make_fake_vote_transport())
    aggregate = aggregate_median(votes)

    with pytest.raises(ValueError):
        build_forecast_record(
            market=market,
            baseline=baseline,
            created_at=created_at,
            question_hash="sha256:question-hash",
            probability_ppm=aggregate.probability_ppm,
            aggregate=aggregate,
            votes=votes,
            citations=(),
            source_notes=(),
            rationale="## Rationale\n\nStub rationale for issue #25 RED test.\n",
            coherence_sum=1_500_000,
        )


# --- Stage-function arithmetic (stages 11-12) -------------------------------------
#
# In a full `run_pipeline` run the vote median is *derived from* the baseline, so
# `shrink_toward_baseline` always receives an estimate equal to its baseline and
# the terminal `_clamp_between` is a structural no-op -- no end-to-end fixture can
# distinguish the shrink/clamp math. These direct unit tests exercise the public
# stage functions with distinguishing inputs so the calibration/shrinkage
# arithmetic and its ppm-domain clamp are behaviorally pinned (mutation robustness
# for Gate 3), without changing the deferred real math (#25).


def test_shrink_toward_baseline_blends_estimate_toward_baseline() -> None:
    """Shrinkage moves an estimate a fixed integer fraction toward the baseline."""
    # lambda = 250_000 ppm: (200_000*750_000 + 1_000_000*250_000) // 1_000_000.
    shrunk = shrink_toward_baseline(200_000, 1_000_000)

    assert shrunk == 400_000
    assert 200_000 < shrunk < 1_000_000


def test_shrink_toward_baseline_is_identity_when_estimate_equals_baseline() -> None:
    """Shrinking an estimate toward an equal baseline leaves it unchanged."""
    assert shrink_toward_baseline(600_000, 600_000) == 600_000


def test_apply_calibration_map_is_identity_within_domain() -> None:
    """The v0 calibration map returns an in-range probability unchanged."""
    assert apply_calibration_map(450_000) == 450_000


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1_000_001, 1_000_000), (-1, 0)],
)
def test_apply_calibration_map_clamps_out_of_domain(value: int, expected: int) -> None:
    """The calibration map defensively clamps back into ``[0, 1_000_000]``."""
    assert apply_calibration_map(value) == expected


# --- Vote probability comes from the response, not the baseline (issue #184) -----
#
# `windbreak.forecast.sanitize.RESPONSE_FAILURE_NON_INTEGER_PROBABILITY` and
# `windbreak.config.schema.EnsembleMemberConfig` do not exist yet, so each test
# below imports them locally rather than at module scope (see this module's
# docstring) -- only these three tests fail collection/execution, never the
# tests above.


def test_vote_probability_comes_from_response_not_baseline(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    make_fake_vote_transport: FakeVoteTransportFactory,
) -> None:
    """Each surviving vote's `probability_ppm` is parsed from its response's
    structured JSON, not derived from `baseline ± fixed offset`.

    `baseline.price_pips == 4500` maps to a 450_000 ppm baseline, so the old
    `baseline ± 10_000 ppm` derivation would have produced exactly
    `(440_000, 450_000, 460_000)` regardless of what the responses said. Here
    every response carries a probability far from that baseline-derived triple
    (123_000 / 456_000 / 789_000), so a pipeline still deriving from the
    baseline would fail this test's *first* assertion, and one still deriving
    from the baseline by coincidence would still fail the explicit negation
    below.
    """
    responses = (
        '{"probability_ppm": 123000, "rationale_summary": "alpha evidence", '
        '"abstain": false}',
        '{"probability_ppm": 456000, "rationale_summary": "beta evidence", '
        '"abstain": false}',
        '{"probability_ppm": 789000, "rationale_summary": "gamma evidence", '
        '"abstain": false}',
    )
    transport = make_fake_vote_transport(responses)

    votes = collect_model_votes(market, baseline, transport=transport)

    observed = [vote.probability_ppm for vote in votes]
    assert observed == [123_000, 456_000, 789_000]
    assert observed != [440_000, 450_000, 460_000]


def test_non_integer_probability_is_discarded_and_ledgered(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
) -> None:
    """A response whose `probability_ppm` is not a true integer (`0.47`) is
    discarded and ledgered -- exactly like an injection-tainted response --
    leaving the other two valid votes to survive, with exactly one
    `FORECAST_OUTPUT_DISCARDED` event carrying the new failure code and a
    fingerprint, never the raw (float-carrying) response text.
    """
    from windbreak.forecast.sanitize import RESPONSE_FAILURE_NON_INTEGER_PROBABILITY

    tainted_response = (
        '{"probability_ppm": 0.47, "rationale_summary": "bad evidence", '
        '"abstain": false}'
    )
    responses = (
        '{"probability_ppm": 300000, "rationale_summary": "alpha evidence", '
        '"abstain": false}',
        tainted_response,
        '{"probability_ppm": 700000, "rationale_summary": "gamma evidence", '
        '"abstain": false}',
    )
    transport = make_fake_vote_transport(responses)
    ledger = InMemoryForecastLedger()

    votes = collect_model_votes(
        market,
        baseline,
        transport=transport,
        ledger=ledger,
        created_at=created_at,
    )

    assert [vote.probability_ppm for vote in votes] == [300_000, 700_000]
    events = ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)
    assert len(events) == 1
    event = events[0]
    assert event.payload["failure"] == RESPONSE_FAILURE_NON_INTEGER_PROBABILITY
    assert event.payload["vote_index"] == 1
    assert (
        event.payload["response_fingerprint"]
        == hashlib.sha256(tainted_response.encode("utf-8")).hexdigest()
    )
    for value in event.payload.values():
        assert "0.47" not in str(value)


def test_collect_model_votes_custom_ensemble_overrides_default_and_call_count(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    make_fake_vote_transport: FakeVoteTransportFactory,
) -> None:
    """A caller-supplied `ensemble` overrides the built-in three-member default:
    the resulting votes carry that tuple's own provenance, in order, and
    `collect_model_votes` issues exactly `len(ensemble)` transport calls (two
    here, never the default three).
    """
    from windbreak.config.schema import EnsembleMemberConfig

    custom_ensemble = (
        EnsembleMemberConfig("openai", "custom-model-a", "2025-01-01"),
        EnsembleMemberConfig("anthropic", "custom-model-b", "2025-02-01"),
    )
    responses = (
        '{"probability_ppm": 111111, "rationale_summary": "alpha evidence", '
        '"abstain": false}',
        '{"probability_ppm": 222222, "rationale_summary": "beta evidence", '
        '"abstain": false}',
    )
    transport = _CountingTransport(make_fake_vote_transport(responses))

    votes = collect_model_votes(
        market, baseline, transport=transport, ensemble=custom_ensemble
    )

    assert transport.call_count == 2
    assert [vote.provider for vote in votes] == ["openai", "anthropic"]
    assert [vote.model_version for vote in votes] == [
        "custom-model-a",
        "custom-model-b",
    ]
    assert [vote.declared_training_cutoff for vote in votes] == [
        "2025-01-01",
        "2025-02-01",
    ]
    assert [vote.probability_ppm for vote in votes] == [111_111, 222_222]


def test_collect_model_votes_excludes_abstaining_member_from_returned_votes(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    make_fake_vote_transport: FakeVoteTransportFactory,
) -> None:
    """`collect_model_votes` excludes an abstaining member's vote from its
    returned tuple (issue #241): the middle response abstains, so exactly
    two (not three) votes survive, and neither surviving vote carries the
    abstainer's 500_000 ppm probability. `ParsedVote.abstain` is parsed
    today but dead-ends before it ever reaches `collect_model_votes`'
    return value, so this currently fails with all three votes surviving.
    """
    responses = (
        '{"probability_ppm": 300000, "rationale_summary": "alpha evidence", '
        '"abstain": false}',
        '{"probability_ppm": 500000, "rationale_summary": "beta evidence", '
        '"abstain": true}',
        '{"probability_ppm": 700000, "rationale_summary": "gamma evidence", '
        '"abstain": false}',
    )
    transport = make_fake_vote_transport(responses)

    votes = collect_model_votes(market, baseline, transport=transport)

    assert [vote.probability_ppm for vote in votes] == [300_000, 700_000]
    assert len(votes) == 2


class _CountingTransport:
    """An `LlmTransport` double counting calls, then delegating (mirrors
    `tests/forecast/test_triage.py`'s `_CountingTransport`).
    """

    def __init__(self, transport: object) -> None:
        """Store the delegate transport and reset the call counter.

        Args:
            transport: The underlying transport to delegate every call to.
        """
        self._transport = transport
        self.call_count = 0

    def complete(self, request: LlmRequest) -> str:
        """Record one call, then delegate to the wrapped transport.

        Args:
            request: The completion request to forward unchanged.

        Returns:
            The wrapped transport's response.
        """
        self.call_count += 1
        return self._transport.complete(request)  # type: ignore[attr-defined]
