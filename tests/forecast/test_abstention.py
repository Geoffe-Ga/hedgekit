"""Tests for scored abstention (issue #26): `is_live_eligible` and the
pipeline's citation-backed abstention path.

Pins three layers of the SPEC S8.5/S8.8 abstention contract: (1) the pure
`windbreak.forecast.records.is_live_eligible` gate and its four independent
ineligibility triggers, (2) the new `ForecastRecord.__post_init__` invariant
forbidding `abstention_reason` and `eligible_for_live=True` together, and (3)
`run_pipeline`'s citation-verification-driven abstention: zero verified
citations always abstains (even with `min_verified_citations=0`, an ABSOLUTE
precedence pin), a non-zero-but-below-threshold verified count still produces
a full, stored, live-*in*eligible record, and the tracer/default-fixture path
meeting the default threshold is live-eligible. `windbreak/forecast/citations.py`
does not exist yet (nor do `is_live_eligible`,
`ForecastRecord`'s new invariant, or `run_pipeline`'s new
`min_verified_citations` keyword), so importing them below fails collection --
the expected Gate 1 RED state for issue #26.

Local valid-kwargs template choice
    `_forecast_record_kwargs` below is a small, self-contained, valid
    `ForecastRecord` keyword template -- deliberately *not* imported from
    `test_records.py`'s `_VALID_FORECAST_RECORD_KWARGS` (cross-test-module
    imports are not supported under this project's rootdir-relative pytest
    import mode; see `tests/forecast/conftest.py`'s and `test_triage.py`'s
    docstrings for the same convention applied elsewhere). `test_records.py`
    needs no change for this issue: its existing template already pins
    `abstention_reason=None` / `eligible_for_live=True`, which is the
    *sanctioned* pairing; the new invariant under test here only fires on the
    (currently untested) `abstention_reason="..."` + `eligible_for_live=True`
    combination, so the two test modules' fixtures compose without conflict.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast import (
    ForbiddenLiveTransport,
    InMemoryTriageLedger,
    run_triaged_pipeline,
)
from windbreak.forecast.citations import FAILURE_CONTENT_HASH_MISMATCH
from windbreak.forecast.pipeline import (
    ABSTENTION_NO_VERIFIED_CITATIONS,
    DEFAULT_MIN_VERIFIED_CITATIONS,
    run_pipeline,
)
from windbreak.forecast.records import (
    ForecastRecord,
    forecast_record_to_payload,
    is_live_eligible,
)
from windbreak.forecast.triage import TRIAGE_PROCEED_EVENT

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.budget import ResearchBudget
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

    FakeVoteTransportFactory = Callable[..., object]
    ResearchToolsFactory = Callable[..., ResearchTools]
    RaisingFetchTransportFactory = Callable[[], object]
    MutatingRefetchTransportFactory = Callable[..., object]

_VALID_FORECAST_RECORD_KWARGS: dict[str, object] = {
    "forecast_id": "fc-abstention-0001",
    "market_ticker": "KXFED-24DEC",
    "normalized_question_hash": "sha256:question-hash",
    "probability_ppm": 450_000,
    "ci_low_ppm": 450_000,
    "ci_high_ppm": 450_000,
    "model_votes": (),
    "vote_dispersion_ppm": 0,
    "rationale_markdown": "## Rationale\n\nStub rationale for issue #26 RED test.\n",
    "citations": (),
    "source_quality_notes": (),
    "research_cost_micros": 3_000_000,
    "triage_stage": "full",
    "created_at": datetime(2024, 12, 10, 12, 0, tzinfo=UTC),
    "forecast_horizon_hours": 48,
    "market_price_baseline_pips": 4500,
    "baseline_quote_snapshot_id": "snap-0001",
    "coherence_group_sum_ppm": None,
    "coherence_flag": False,
    "abstention_reason": None,
    "eligible_for_live": True,
}


def _record(**overrides: object) -> ForecastRecord:
    """Build a `ForecastRecord` from the local valid-kwargs template.

    Args:
        **overrides: Field overrides layered on `_VALID_FORECAST_RECORD_KWARGS`.

    Returns:
        A constructed `ForecastRecord`.
    """
    return ForecastRecord(**{**_VALID_FORECAST_RECORD_KWARGS, **overrides})


class _EmptySearchTransport:
    """A `SearchTransport` double that finds no candidate URL for any query.

    Mirrors `test_sandbox.py`'s local double of the same name.
    """

    def search(self, query: str) -> tuple[str, ...]:
        """Return an empty tuple, ignoring `query`.

        Args:
            query: The (unused) subquestion text.

        Returns:
            An empty tuple -- the "no candidate found" case.
        """
        return ()


def _assert_no_float_leaf(node: object) -> None:
    """Recursively assert that no leaf of `node` is a `float` instance.

    Args:
        node: A JSON-safe payload node (mapping, sequence, or scalar).
    """
    if isinstance(node, dict):
        for value in node.values():
            _assert_no_float_leaf(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _assert_no_float_leaf(item)
    else:
        assert type(node) is not float, f"float leaf found in payload: {node!r}"


# --- Constants ------------------------------------------------------------------


def test_default_min_verified_citations_constant_is_three() -> None:
    """`DEFAULT_MIN_VERIFIED_CITATIONS` pins the SPEC-mandated default of 3."""
    assert DEFAULT_MIN_VERIFIED_CITATIONS == 3


def test_abstention_no_verified_citations_constant_value() -> None:
    """`ABSTENTION_NO_VERIFIED_CITATIONS` pins the exact abstention-reason string."""
    assert ABSTENTION_NO_VERIFIED_CITATIONS == "no_verified_citations"


# --- is_live_eligible: pure-gate truth table -------------------------------------


def test_is_live_eligible_true_when_count_equals_min() -> None:
    """The count-equals-min boundary is eligible (an inclusive `>=` gate)."""
    result = is_live_eligible(
        verified_citation_count=3,
        min_verified_citations=3,
        triage_stage="full",
        coherence_flag=False,
        abstention_reason=None,
    )

    assert result is True


def test_is_live_eligible_false_when_count_one_below_min() -> None:
    """One verified citation short of the minimum is ineligible."""
    result = is_live_eligible(
        verified_citation_count=2,
        min_verified_citations=3,
        triage_stage="full",
        coherence_flag=False,
        abstention_reason=None,
    )

    assert result is False


def test_is_live_eligible_true_when_count_exceeds_min_and_no_trigger_fires() -> None:
    """A comfortably-above-min count, with every trigger absent, is eligible.

    Confirms the composed guard is not over-broad.
    """
    result = is_live_eligible(
        verified_citation_count=5,
        min_verified_citations=3,
        triage_stage="full",
        coherence_flag=False,
        abstention_reason=None,
    )

    assert result is True


@pytest.mark.parametrize(
    ("triage_stage", "coherence_flag", "abstention_reason"),
    [
        ("triage_only", False, None),
        ("full", True, None),
        ("full", False, "no_verified_citations"),
    ],
)
def test_is_live_eligible_each_trigger_independently_forces_ineligibility(
    triage_stage: str, coherence_flag: bool, abstention_reason: str | None
) -> None:
    """Each of the three non-count triggers independently forces `False`, even
    when `verified_citation_count` comfortably exceeds `min_verified_citations`.
    """
    result = is_live_eligible(
        verified_citation_count=10,
        min_verified_citations=3,
        triage_stage=triage_stage,
        coherence_flag=coherence_flag,
        abstention_reason=abstention_reason,
    )

    assert result is False


@pytest.mark.parametrize("field", ["verified_citation_count", "min_verified_citations"])
def test_is_live_eligible_bool_count_field_raises_type_error(field: str) -> None:
    """A stray `bool` for either count field must never masquerade as an int."""
    kwargs: dict[str, object] = {
        "verified_citation_count": 3,
        "min_verified_citations": 3,
        "triage_stage": "full",
        "coherence_flag": False,
        "abstention_reason": None,
    }
    kwargs[field] = True

    with pytest.raises(TypeError):
        is_live_eligible(**kwargs)


@pytest.mark.parametrize("field", ["verified_citation_count", "min_verified_citations"])
def test_is_live_eligible_negative_count_field_raises_value_error(field: str) -> None:
    """A negative count field is rejected as an invalid count."""
    kwargs: dict[str, object] = {
        "verified_citation_count": 3,
        "min_verified_citations": 3,
        "triage_stage": "full",
        "coherence_flag": False,
        "abstention_reason": None,
    }
    kwargs[field] = -1

    with pytest.raises(ValueError):
        is_live_eligible(**kwargs)


# --- ForecastRecord: new abstention/eligibility invariant ------------------------


def test_forecast_record_abstained_and_eligible_for_live_raises_value_error() -> None:
    """A non-`None` `abstention_reason` paired with `eligible_for_live=True`
    is an invalid combination and must raise (mirrors the existing
    `triage_stage`/`coherence_flag` ineligibility invariants).
    """
    with pytest.raises(ValueError, match=r"abstention_reason|eligible_for_live"):
        _record(abstention_reason="no_verified_citations", eligible_for_live=True)


def test_forecast_record_abstained_and_ineligible_constructs_fine() -> None:
    """The sanctioned pairing (abstained AND ineligible) still constructs."""
    record = _record(abstention_reason="no_verified_citations", eligible_for_live=False)

    assert record.abstention_reason == "no_verified_citations"
    assert record.eligible_for_live is False


# --- run_pipeline: abstention precedence and thresholds --------------------------


def test_run_pipeline_all_unreachable_citations_abstains_before_vote_transport(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: ResearchToolsFactory,
    make_raising_fetch_transport: RaisingFetchTransportFactory,
    tmp_path: Path,
) -> None:
    """Every candidate URL being unreachable yields zero verified citations,
    so the run abstains -- and does so *before* `collect_model_votes`:
    `ForbiddenLiveTransport` as the vote transport never raises
    `LiveCallForbiddenError`, the structural proof the vote stage is skipped.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path, fetch_transport=make_raising_fetch_transport()
    )

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=tools,
    )

    assert record.abstention_reason == ABSTENTION_NO_VERIFIED_CITATIONS
    assert record.eligible_for_live is False
    assert record.triage_stage == "full"
    assert record.model_votes == ()


def test_run_pipeline_empty_search_results_abstain(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """A search transport that finds no candidate URLs gathers zero citations,
    which also abstains (zero citations gathered implies zero verified).
    """
    tools = research_tools_factory(
        cache_dir=tmp_path, search_transport=_EmptySearchTransport()
    )

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=tools,
    )

    assert record.abstention_reason == ABSTENTION_NO_VERIFIED_CITATIONS
    assert record.eligible_for_live is False
    assert record.citations == ()
    assert record.model_votes == ()


def test_run_pipeline_zero_min_verified_citations_still_abstains_on_zero_verified(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """Abstention precedence is ABSOLUTE: even `min_verified_citations=0`
    still abstains when zero citations verify -- the threshold knob can lower
    the eligibility bar, but it can never turn "zero evidence" into eligible.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path, search_transport=_EmptySearchTransport()
    )

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=tools,
        min_verified_citations=0,
    )

    assert record.abstention_reason == ABSTENTION_NO_VERIFIED_CITATIONS
    assert record.eligible_for_live is False


def test_run_pipeline_partial_verification_below_default_min_is_full_but_ineligible(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: ResearchToolsFactory,
    make_mutating_refetch_transport: MutatingRefetchTransportFactory,
    make_fake_vote_transport: FakeVoteTransportFactory,
    tmp_path: Path,
) -> None:
    """Two of three citations verifying (below the default min of 3) still
    produces and stores a *full* record -- just not a live-eligible one, and
    not abstained.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path,
        fetch_transport=make_mutating_refetch_transport(stable_urls=2),
    )

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=tools,
    )

    assert record.triage_stage == "full"
    assert record.eligible_for_live is False
    assert record.abstention_reason is None
    assert len(record.citations) == 3
    assert any(
        "unverified" in note and FAILURE_CONTENT_HASH_MISMATCH in note
        for note in record.source_quality_notes
    )


def test_run_pipeline_partial_verification_meeting_lowered_min_is_live_eligible(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: ResearchToolsFactory,
    make_mutating_refetch_transport: MutatingRefetchTransportFactory,
    make_fake_vote_transport: FakeVoteTransportFactory,
    tmp_path: Path,
) -> None:
    """Lowering `min_verified_citations` to match the verified count flips the
    same partial-verification scenario to live-eligible: the config knob works.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path,
        fetch_transport=make_mutating_refetch_transport(stable_urls=2),
    )

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=tools,
        min_verified_citations=2,
    )

    assert record.eligible_for_live is True
    assert record.abstention_reason is None


# --- Tracer pin: default fixtures meet the default threshold ---------------------


def test_default_fixture_run_is_live_eligible_and_byte_deterministic(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """Under the default conftest fixtures (3 subquestions, all fetches
    self-verify), the default run is live-eligible, never abstains, and two
    runs over fresh transports produce equal records and byte-identical JSON.
    """
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

    assert record_a.eligible_for_live is True
    assert record_a.abstention_reason is None
    assert record_a == record_b
    payload_a = json.dumps(forecast_record_to_payload(record_a), sort_keys=True)
    payload_b = json.dumps(forecast_record_to_payload(record_b), sort_keys=True)
    assert payload_a == payload_b


# --- Ledgerability: abstained records are JSON-safe ------------------------------


def test_abstained_record_payload_is_json_dumps_clean_and_float_free(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """An abstained record's payload round-trips through `json.dumps`, carries
    a non-`None` `abstention_reason`, and has no float leaf anywhere.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path, search_transport=_EmptySearchTransport()
    )
    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=tools,
    )

    payload = forecast_record_to_payload(record)

    assert json.dumps(payload)
    assert payload["abstention_reason"] is not None
    _assert_no_float_leaf(payload)


# --- Integration: abstained records are ledgered like any forecast --------------


def test_triaged_pipeline_abstained_record_is_ledgered_as_proceed(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools_factory: ResearchToolsFactory,
    make_raising_fetch_transport: RaisingFetchTransportFactory,
    make_fake_vote_transport: FakeVoteTransportFactory,
    tmp_path: Path,
    research_budget: ResearchBudget,
) -> None:
    """`operator_flagged=True` forces the PROCEED path regardless of the
    Stage-0 prior; when the full pipeline it proceeds into abstains (every
    citation unreachable), that abstained record is still ledgered as a
    normal `TRIAGE_PROCEED` event -- abstention is a property of the
    *forecast record*, not a reason to skip ledgering.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path, fetch_transport=make_raising_fetch_transport()
    )
    ledger = InMemoryTriageLedger()

    record = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("600000",)),
        full_transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        research_tools=tools,
        operator_flagged=True,
        budget=research_budget,
    )

    assert record.abstention_reason == ABSTENTION_NO_VERIFIED_CITATIONS
    assert record.eligible_for_live is False
    proceed_events = ledger.events_by_type(TRIAGE_PROCEED_EVENT)
    assert len(proceed_events) == 1
