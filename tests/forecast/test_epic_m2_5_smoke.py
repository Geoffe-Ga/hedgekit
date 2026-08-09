"""Epic #183 M2.5 end-to-end acceptance smoke suite (issue #284, verification).

Every M2.5-epic-relevant collaborator (`run_pipeline` and its full stage
chain, the cassette record/replay harness, the per-provider vote-cost ledger
seam (issue #281), the response-side injection screen (SPEC S8.5), the
ensemble quorum-abstention taxonomy (issue #241), and the per-provider
track-record gate (issue #194)) already exists and is already exercised,
piecewise, by this package's other test modules. This module's job is
narrower and different in kind: it drives
`windbreak.forecast.pipeline.run_pipeline` over the existing fixtures once
per epic-acceptance criterion and pins every resulting surface -- vote
provenance, the point estimate, citations, the research-cost constant, the
per-vote cost ledger, and byte-identical record/replay determinism -- onto a
*single* record per test, so a future change that silently breaks the epic's
end-to-end contract (even while every piecewise unit test keeps passing) is
caught here.

TDD posture
    This is a **verification** suite, not new-feature TDD. Every production
    collaborator it drives already ships; Gate 1's "RED" for this issue is
    simply this file's own absence, and the expected first-run outcome is
    GREEN, not a red failure pinning a not-yet-built behavior. The
    anti-regression discipline TDD's red-first cycle normally buys is
    transferred onto **assertion strength** instead: every assertion below
    pins an exact value, count, constant, or byte-equal payload -- never an
    existence-only (`citations` is truthy) or shape-only (a dict came back)
    check that a mutant could satisfy by accident. Every expected value
    quoted here (the three canned vote probabilities, the pinned ensemble's
    provenance triple, the fixture provider's zero cost, the research-cost
    stub constant, the provider-gate's default thresholds) was derived by
    hand from the epic's acceptance criteria and this package's own
    already-pinned constants *before* this suite was ever run against the
    real pipeline -- never copied off an observed first-run output. Had a
    hand-derived expectation disagreed with the pipeline's honest first-run
    behavior, the mismatch would have been diagnosed: a wrong expectation
    gets fixed here with documented reasoning (see test 1's
    shrink-to-baseline-is-identity note below); a genuine behavior gap would
    instead stop this suite and get reported rather than bend an assertion or
    touch production code. In this run every hand-derived expectation matched
    on the first honest execution.

Fixture and doubles choices
    `market`/`baseline`/`created_at`/`research_tools`/`research_tools_factory`
    /`diverse_markets` all come from `tests/forecast/conftest.py` unchanged --
    see that module's own docstring for why they are constructed directly
    rather than loaded from a fixture file. `FakeVoteTransport` and
    `DIVERGENT_VOTE_RESPONSES` are imported by name from that conftest module
    (the one cross-test-module import this repo's convention blesses); every
    other double below (`_SucceedingProvider`, `_FailingProvider`,
    `_routed_provider_factory`) is defined module-locally, mirroring
    `tests/forecast/test_provider_vote_costing.py`'s and
    `tests/forecast/test_provider_failures.py`'s identical convention (no
    supported cross-test-module import for private helpers under this
    project's rootdir-relative pytest import mode).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tests.forecast.conftest import DIVERGENT_VOTE_RESPONSES, FakeVoteTransport
from windbreak.forecast.cassettes import (
    ForbiddenLiveTransport,
    RecordingCassette,
    ReplayCassette,
)
from windbreak.forecast.pipeline import (
    ABSTENTION_ALL_VOTES_DISCARDED,
    ABSTENTION_ENSEMBLE_QUORUM_NOT_MET,
    FORECAST_OUTPUT_DISCARDED_EVENT,
    PROVIDER_GATE_HELD_EVENT,
    PROVIDER_VOTE_COSTED_EVENT,
    VOTE_OUTCOME_VOTED,
    InMemoryForecastLedger,
    run_pipeline,
)
from windbreak.forecast.providers import DEFAULT_VOTE_ENSEMBLE, ProviderForecast
from windbreak.forecast.providers.base import ProviderTimeoutError
from windbreak.forecast.providers.fixture import FixtureVoteProvider
from windbreak.forecast.providers.track_record import (
    InMemoryTrackRecordSource,
    ProviderTrackRecordGate,
)
from windbreak.forecast.records import forecast_record_to_payload
from windbreak.forecast.sanitize import DATA_BLOCK_BEGIN

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.providers.base import EnsembleMemberLike, ForecastProvider
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

#: The default ensemble's three members, indexed rather than unpacked from
#: the variable-length `DEFAULT_VOTE_ENSEMBLE` tuple -- mirrors
#: `tests/forecast/test_provider_vote_costing.py`'s and
#: `tests/forecast/test_provider_failures.py`'s identical convention.
_MEMBER_A = DEFAULT_VOTE_ENSEMBLE[0]
_MEMBER_B = DEFAULT_VOTE_ENSEMBLE[1]
_MEMBER_C = DEFAULT_VOTE_ENSEMBLE[2]

#: The pinned default vote-ensemble's exact provenance, in ensemble order --
#: `(provider, model_version, training_cutoff)`. Hardcoded here (rather than
#: read live off `DEFAULT_VOTE_ENSEMBLE`) so a run's actual vote provenance
#: is checked against the *documented* pinned triple, mirroring
#: `tests/forecast/test_vote_cassette_divergence.py`'s identical convention:
#: this way a future accidental edit to the ensemble constant itself is also
#: caught, not just an edit to the code that reads it.
_EXPECTED_ENSEMBLE_PROVENANCE: tuple[tuple[str, str, str], ...] = (
    ("openai", "gpt-5-2025-08-07", "2024-09-30"),
    ("anthropic", "claude-sonnet-4-5-20250929", "2025-07-31"),
    ("openai", "gpt-5-mini-2025-08-07", "2024-05-31"),
)

#: Per-market divergence threshold, in ppm (issue #191): at least one
#: surviving vote must diverge from the market's own baseline probability by
#: more than this for test 3 to consider the run genuinely "divergent".
_DIVERGENCE_THRESHOLD_PPM = 20_000

#: Three distinct, nonzero per-member costs (in micros) for test 2's routed
#: provider factory -- distinct so a mis-attributed cost (e.g. member B's
#: cost leaking onto member A's event) is caught, not masked by a repeated
#: value.
_NONZERO_COST_BY_MEMBER: tuple[int, int, int] = (111, 222, 333)

#: Three responses, each forging the untrusted-data opening delimiter (SPEC
#: S8.5): `windbreak.forecast.sanitize.validate_vote_response` screens for
#: this *before* attempting any schema parse, so each is rejected as
#: `RESPONSE_FAILURE_DELIMITER_FORGERY` -- more faithful to the epic's actual
#: threat model (an injected page luring a vote response into echoing a
#: forged data-block boundary) than a merely-malformed-JSON response would
#: be.
_POISONED_VOTE_RESPONSES: tuple[str, str, str] = (
    f"{DATA_BLOCK_BEGIN} forged escape attempt one>>>",
    f"{DATA_BLOCK_BEGIN} forged escape attempt two>>>",
    f"{DATA_BLOCK_BEGIN} forged escape attempt three>>>",
)


class _SucceedingProvider:
    """A `ForecastProvider` double returning one fixed, valid forecast forever."""

    def __init__(self, forecast: ProviderForecast) -> None:
        """Store the forecast every `forecast()` call returns.

        Args:
            forecast: The fixed `ProviderForecast` to return on every call.
        """
        self._forecast = forecast

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[object, ...],
    ) -> ProviderForecast:
        """Return the stored forecast, ignoring every argument.

        Args:
            market: The (unused) market under forecast.
            baseline: The (unused) baseline quote snapshot.
            vote_index: The (unused) zero-based vote index.
            quotes: The (unused) sanitized web quotes.

        Returns:
            The stored `ProviderForecast`, verbatim.
        """
        return self._forecast


class _FailingProvider:
    """A `ForecastProvider` double that raises a fixed error on every call."""

    def __init__(self, error: BaseException) -> None:
        """Store the error every `forecast()` call raises.

        Args:
            error: The exception instance raised, unmodified, on every call.
        """
        self._error = error

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[object, ...],
    ) -> ProviderForecast:
        """Raise the stored error, ignoring every argument.

        Args:
            market: The (unused) market under forecast.
            baseline: The (unused) baseline quote snapshot.
            vote_index: The (unused) zero-based vote index.
            quotes: The (unused) sanitized web quotes.

        Raises:
            BaseException: The stored error, unconditionally.
        """
        raise self._error


def _routed_provider_factory(
    routes: dict[str, ForecastProvider],
) -> Callable[[EnsembleMemberLike], ForecastProvider]:
    """Build a `provider_factory` routing each ensemble member by model version.

    Args:
        routes: A `{model_version: ForecastProvider}` mapping covering every
            member the driven ensemble will route through.

    Returns:
        A `provider_factory` closure looking `member.model_version` up in
        `routes`.
    """

    def _factory(member: EnsembleMemberLike) -> ForecastProvider:
        """Return the provider routed for `member`'s pinned model version.

        Args:
            member: The ensemble member being driven.

        Returns:
            The routed `ForecastProvider`.
        """
        return routes[member.model_version]

    return _factory


def _clean_forecast(
    member: EnsembleMemberLike, *, cost_micros: int
) -> ProviderForecast:
    """Build a valid, non-abstaining `ProviderForecast` stamped for `member`.

    Args:
        member: The ensemble member this forecast's provenance is stamped
            with.
        cost_micros: The forecast's billed cost, in micros.

    Returns:
        A schema-valid `ProviderForecast` with a fixed mid-range probability.
    """
    return ProviderForecast(
        probability_ppm=500_000,
        rationale_summary="steady evidence",
        citations=(),
        cost_micros=cost_micros,
        provider=member.provider,
        model_version=member.model_version,
        training_cutoff=member.training_cutoff,
        response_fingerprint="f" * 64,
        abstain=False,
    )


# --- Test 1: crit 1 -- the bundled epic-surface record ----------------------


def test_screened_market_record_bundles_epic_surface(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
    tmp_path: Path,
) -> None:
    """One full-pipeline run over the default fixtures bundles every M2.5
    epic-acceptance surface (crit 1) onto a single record: vote provenance
    and probabilities, the median point estimate, citations and source
    notes, the research-cost stub constant, the per-provider vote-cost
    ledger (issue #281), and byte-identical record/replay determinism
    (SPEC S8.2 / S8.6 / S8.8 / S8.9).
    """
    ledger = InMemoryForecastLedger()
    cassette_path = tmp_path / "cassette.json"

    record = run_pipeline(
        market,
        baseline,
        transport=RecordingCassette(transport=FakeVoteTransport(), path=cassette_path),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
    )

    assert record.triage_stage == "full"
    assert record.abstention_reason is None
    assert record.eligible_for_live is True

    assert len(record.model_votes) == 3
    assert {vote.probability_ppm for vote in record.model_votes} == {
        440_000,
        450_000,
        460_000,
    }
    provenance = tuple(
        (vote.provider, vote.model_version, vote.declared_training_cutoff)
        for vote in record.model_votes
    )
    assert provenance == _EXPECTED_ENSEMBLE_PROVENANCE
    assert {provider for provider, _, _ in provenance} == {"anthropic", "openai"}

    # The three canned vote probabilities are symmetric around the market's
    # own 450_000 ppm baseline, so stage 12's shrink-to-baseline blend is a
    # no-op identity here: the sorted median IS the final point estimate.
    assert record.probability_ppm == 450_000

    assert len(record.citations) == 3
    verified_notes = [
        note for note in record.source_quality_notes if "verified" in note
    ]
    assert len(verified_notes) == 3

    # `_RESEARCH_COST_MICROS` is a private pipeline constant (SPEC stub); pin
    # its documented value directly rather than importing it.
    assert record.research_cost_micros == 3_000_000

    cost_events = ledger.events_by_type(PROVIDER_VOTE_COSTED_EVENT)
    assert len(cost_events) == 3
    cost_events_by_index = {
        event.payload["vote_index"]: event.payload for event in cost_events
    }
    for index, (provider, model_version, _) in enumerate(_EXPECTED_ENSEMBLE_PROVENANCE):
        assert cost_events_by_index[index] == {
            "market_ticker": market.ticker,
            "provider": provider,
            "model_version": model_version,
            "vote_index": index,
            # The fixture provider's cost is definitionally zero.
            # windbreak/forecast/providers/fixture.py:44
            "cost_micros": 0,
            "outcome": VOTE_OUTCOME_VOTED,
            "failure_code": "",
        }

    replay_record_1 = run_pipeline(
        market,
        baseline,
        transport=ReplayCassette.from_path(cassette_path),
        created_at=created_at,
        research_tools=research_tools,
    )
    replay_record_2 = run_pipeline(
        market,
        baseline,
        transport=ReplayCassette.from_path(cassette_path),
        created_at=created_at,
        research_tools=research_tools,
    )

    assert replay_record_1 == replay_record_2
    payload_1 = json.dumps(forecast_record_to_payload(replay_record_1), sort_keys=True)
    payload_2 = json.dumps(forecast_record_to_payload(replay_record_2), sort_keys=True)
    assert payload_1 == payload_2
    # The ledger and transport mode differ (recording vs. replay) but neither
    # is part of `ForecastRecord`'s own fields, so the record produced while
    # recording equals both pure-replay records byte-for-byte.
    assert record == replay_record_1


# --- Test 2: crit 1 complement -- nonzero per-provider vote costs -----------


def test_nonzero_per_provider_vote_costs_ledger_through_run_pipeline(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """A routed provider factory's distinct nonzero per-member costs ledger
    exactly, over `run_pipeline` -- the complement to test 1's all-zero
    costs (`FixtureVoteProvider`'s cost is definitionally zero, so crit 1's
    "nonzero cost" acceptance property needs a provider double that actually
    charges something). `ForbiddenLiveTransport` is wired as `transport` and
    never raises, proving the provider_factory path never touches it.
    """
    ledger = InMemoryForecastLedger()
    routes: dict[str, ForecastProvider] = {
        member.model_version: _SucceedingProvider(
            _clean_forecast(member, cost_micros=cost)
        )
        for member, cost in zip(
            (_MEMBER_A, _MEMBER_B, _MEMBER_C), _NONZERO_COST_BY_MEMBER, strict=True
        )
    }

    run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_factory=_routed_provider_factory(routes),
    )

    events = ledger.events_by_type(PROVIDER_VOTE_COSTED_EVENT)
    assert len(events) == 3
    cost_by_model_version = {
        event.payload["model_version"]: event.payload["cost_micros"] for event in events
    }
    assert cost_by_model_version == {
        _MEMBER_A.model_version: 111,
        _MEMBER_B.model_version: 222,
        _MEMBER_C.model_version: 333,
    }
    assert all(event.payload["outcome"] == VOTE_OUTCOME_VOTED for event in events)


# --- Test 3: crit 2 -- divergent votes on one bundled record -----------------


def test_divergent_record_carries_rationale_and_citations(
    diverse_markets: tuple[tuple[NormalizedMarket, BaselineQuoteSnapshot], ...],
    created_at: datetime,
    research_tools_factory: Callable[..., ResearchTools],
    tmp_path: Path,
) -> None:
    """The Fed-rate market's three mutually-diverging votes (issue #191, crit
    2) land on the *same* record that also carries a non-empty rationale and
    three independently-verified citations -- the epic's "one bundled
    record", not three disjoint properties checked on three separate runs.
    """
    market, baseline = diverse_markets[0]
    baseline_ppm = baseline.price_pips * 100
    responses = DIVERGENT_VOTE_RESPONSES[market.ticker]
    research_tools = research_tools_factory(cache_dir=tmp_path / "research-cache")
    cassette_path = tmp_path / "cassette.json"

    record = run_pipeline(
        market,
        baseline,
        transport=RecordingCassette(
            transport=FakeVoteTransport(responses=responses), path=cassette_path
        ),
        created_at=created_at,
        research_tools=research_tools,
    )

    assert baseline_ppm == 450_000
    # The Fed market's three responses are all non-abstaining (conftest
    # `DIVERGENT_VOTE_RESPONSES[_FED_TICKER]`), so all three survive; pin the
    # exact surviving set (matching test 1's rigor) rather than only that one
    # vote diverges.
    assert {vote.probability_ppm for vote in record.model_votes} == {
        440_000,
        455_000,
        490_000,
    }
    assert any(
        abs(vote.probability_ppm - baseline_ppm) > _DIVERGENCE_THRESHOLD_PPM
        for vote in record.model_votes
    )
    # `_build_rationale` (pipeline.py) always opens with this exact heading, so
    # pin a stable prefix rather than an existence-only non-empty check.
    assert record.rationale_markdown.startswith("## Rationale")
    assert len(record.citations) == 3
    assert record.abstention_reason is None
    assert record.eligible_for_live is True


# --- Test 4: crit 3 (thin) -- single-vendor failure abstains ----------------


def test_single_vendor_failure_abstains_quorum_not_met(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """Two of the three default members (both `"openai"`) time out, leaving
    a single `"anthropic"` survivor -- below the default
    `min_ensemble_votes=2` quorum -- so the run abstains
    `ABSTENTION_ENSEMBLE_QUORUM_NOT_MET` rather than aggregate over one vote
    (issue #241's own precedence rule 4).
    """
    ledger = InMemoryForecastLedger()
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _FailingProvider(ProviderTimeoutError()),
        _MEMBER_B.model_version: FixtureVoteProvider(FakeVoteTransport(), _MEMBER_B),
        _MEMBER_C.model_version: _FailingProvider(ProviderTimeoutError()),
    }

    record = run_pipeline(
        market,
        baseline,
        # Every member is routed through `provider_factory`, so the outer
        # transport is never touched: wire the forbidden one to prove it.
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_factory=_routed_provider_factory(routes),
    )

    assert record.abstention_reason == ABSTENTION_ENSEMBLE_QUORUM_NOT_MET
    assert record.model_votes == ()
    assert record.probability_ppm == 450_000
    assert record.eligible_for_live is False
    assert len(ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)) == 2


# --- Test 5: crit 4 (thin) -- poisoned votes all discarded -------------------


def test_poisoned_votes_all_discarded_abstains(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """Three responses that each forge the untrusted-data opening delimiter
    (SPEC S8.5) are all screen-rejected before any schema parse is even
    attempted, leaving zero survivors from a non-transport-class wipeout --
    so the run abstains `ABSTENTION_ALL_VOTES_DISCARDED`, not the
    transport-only `ABSTENTION_PROVIDER_UNAVAILABLE` reason.
    """
    ledger = InMemoryForecastLedger()

    record = run_pipeline(
        market,
        baseline,
        transport=FakeVoteTransport(responses=_POISONED_VOTE_RESPONSES),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
    )

    assert record.abstention_reason == ABSTENTION_ALL_VOTES_DISCARDED
    assert record.model_votes == ()
    assert record.eligible_for_live is False
    assert len(ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)) == 3


# --- Test 6: crit 5 -- unproven provider held back from live -----------------


def test_unproven_provider_held_back_from_live(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """An empty track-record source (issue #194) leaves both default-ensemble
    provider families -- `"anthropic"` and `"openai"` -- unproven, holding
    the run back from live eligibility while every vote still runs (only
    eligibility is forced, never the votes themselves), and ledgering
    exactly one `PROVIDER_GATE_HELD` event naming both.
    """
    ledger = InMemoryForecastLedger()
    gate = ProviderTrackRecordGate(InMemoryTrackRecordSource([]))

    record = run_pipeline(
        market,
        baseline,
        transport=FakeVoteTransport(),
        created_at=created_at,
        research_tools=research_tools,
        ledger=ledger,
        provider_gate=gate,
    )

    assert record.eligible_for_live is False
    assert record.abstention_reason is None
    assert len(record.model_votes) == 3

    events = ledger.events_by_type(PROVIDER_GATE_HELD_EVENT)
    assert len(events) == 1
    # The gate's default thresholds are pinned as literals (not imported) so an
    # accidental edit to `DEFAULT_MIN_RESOLVED` / `DEFAULT_MIN_BRIER_SKILL_PPM`
    # in track_record.py fails this equality rather than moving both sides
    # together and escaping the pin -- the same rigor as the provenance triple.
    assert events[0].payload == {
        "unproven_providers": "anthropic,openai",
        "unproven_count": 2,
        "min_resolved": 150,
        "min_brier_skill_ppm": 10_000,
    }
