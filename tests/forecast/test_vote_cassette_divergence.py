"""Cross-market vote-divergence coverage for the LLM cassette harness (#191).

Drives `run_pipeline` over three genuinely different markets (a Fed-rate
market, a weather market, and a presidential-election market -- see
`tests/forecast/conftest.py`'s `diverse_markets` fixture and
`DIVERGENT_VOTE_RESPONSES` constant), each wired to a `FakeVoteTransport`
returning three hand-authored, mutually-diverging vote responses wrapped in a
`RecordingCassette`, then replayed twice through `ReplayCassette` -- pinning
that the record/replay harness stays fully deterministic even when the three
ensemble votes actually disagree with each other and with the market
baseline (not just when they happen to be close, as the pre-#191
`test_pipeline_skeleton.py` coverage exercises with a single fixed market).

Each surviving vote's provenance is checked against the new #191 pinned
vote-ensemble triple (`windbreak.forecast.providers.DEFAULT_VOTE_ENSEMBLE`
and its config-schema mirror) -- today that ensemble is still the pre-#191
triple, so this module's provenance assertions fail with an `AssertionError`
naming the mismatched provider/model_version/training_cutoff, not a
collection error, the expected Gate 1 RED state for issue #191.

A separate, single fail-closed test drives the same default market/baseline
fixtures over `ForbiddenLiveTransport`, proving the vote stage is actually
reached (the transport really is called) yet never completes a live call.
That test exercises an already-guaranteed structural invariant (no new
production code is required for it to pass), so it is included as
regression coverage locking in behavior this issue must not break, not as a
RED pin.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.forecast.conftest import DIVERGENT_VOTE_RESPONSES
from windbreak.forecast.cassettes import (
    ForbiddenLiveTransport,
    LiveCallForbiddenError,
    RecordingCassette,
    ReplayCassette,
)
from windbreak.forecast.pipeline import run_pipeline
from windbreak.forecast.records import forecast_record_to_payload

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

#: Per-market divergence threshold, in ppm: at least one surviving vote must
#: diverge from that market's own baseline probability by more than this.
_DIVERGENCE_THRESHOLD_PPM = 20_000

#: The #191 pinned default vote-ensemble triple's expected provenance, in
#: ensemble order -- `(provider, model_version, training_cutoff)`. Hardcoded
#: here (rather than imported live) so a run's actual vote provenance is
#: checked against the *documented* new pinned triple, not merely against
#: whatever `DEFAULT_VOTE_ENSEMBLE` happens to currently hold.
_EXPECTED_ENSEMBLE_PROVENANCE: tuple[tuple[str, str, str], ...] = (
    ("openai", "gpt-5-2025-08-07", "2024-09-30"),
    ("anthropic", "claude-sonnet-4-5-20250929", "2025-07-31"),
    ("openai", "gpt-5-mini-2025-08-07", "2024-05-31"),
)


@pytest.mark.parametrize("market_index", [0, 1, 2])
def test_divergent_votes_record_and_replay_across_diverse_markets(
    market_index: int,
    diverse_markets: tuple[tuple[NormalizedMarket, BaselineQuoteSnapshot], ...],
    created_at: datetime,
    research_tools_factory: Callable[..., ResearchTools],
    make_fake_vote_transport: Callable[..., object],
    tmp_path: Path,
) -> None:
    """One market's three diverging votes record, then replay twice,
    deterministically -- with correct per-member provenance throughout.
    """
    market, baseline = diverse_markets[market_index]
    baseline_ppm = baseline.price_pips * 100
    responses = DIVERGENT_VOTE_RESPONSES[market.ticker]
    research_tools = research_tools_factory(cache_dir=tmp_path / "research-cache")
    cassette_path = tmp_path / "cassette.json"

    recording_transport = RecordingCassette(
        transport=make_fake_vote_transport(responses=responses), path=cassette_path
    )
    record = run_pipeline(
        market,
        baseline,
        transport=recording_transport,
        created_at=created_at,
        research_tools=research_tools,
    )

    # (a) At least two distinct probabilities survive.
    assert len({vote.probability_ppm for vote in record.model_votes}) >= 2
    # (b) At least one vote diverges from the market's own baseline by more
    # than the threshold.
    assert any(
        abs(vote.probability_ppm - baseline_ppm) > _DIVERGENCE_THRESHOLD_PPM
        for vote in record.model_votes
    )
    # (d) Every surviving vote carries a non-empty fingerprint and the #191
    # pinned ensemble's exact provenance, in order. A member whose response
    # abstained (issue #241) is excluded from the record, so the expected
    # provenance is only the non-abstaining members' -- for the elections
    # market that is the first two members; Fed and weather keep all three.
    expected_provenance = tuple(
        provenance
        for provenance, response in zip(
            _EXPECTED_ENSEMBLE_PROVENANCE, responses, strict=True
        )
        if not json.loads(response)["abstain"]
    )
    assert len(record.model_votes) == len(expected_provenance)
    for vote, (provider, model_version, training_cutoff) in zip(
        record.model_votes, expected_provenance, strict=True
    ):
        assert vote.response_fingerprint
        assert vote.provider == provider
        assert vote.model_version == model_version
        assert vote.declared_training_cutoff == training_cutoff

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

    # (c) Two independent replay runs are equal and byte-identical.
    assert replay_record_1 == replay_record_2
    payload_1 = json.dumps(forecast_record_to_payload(replay_record_1), sort_keys=True)
    payload_2 = json.dumps(forecast_record_to_payload(replay_record_2), sort_keys=True)
    assert payload_1 == payload_2


def test_forbidden_live_transport_never_completes_a_live_call(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """Driving the default market/baseline over `ForbiddenLiveTransport`
    reaches the vote stage (the transport really is invoked) yet raises
    `LiveCallForbiddenError` instead of ever completing a live call.
    """
    with pytest.raises(LiveCallForbiddenError):
        run_pipeline(
            market,
            baseline,
            transport=ForbiddenLiveTransport(),
            created_at=created_at,
            research_tools=research_tools,
        )
