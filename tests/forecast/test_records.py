"""Tests for windbreak.forecast.records (issue #22): the SPEC S6.3 record schema.

Pins `ForecastRecord.__post_init__` validation (ppm range and non-bool-int
integrality, non-empty identifiers, the `triage_stage` closed set),
immutability of `ForecastRecord`, `ModelVote`, `Citation`, and
`BaselineQuoteSnapshot`, and `forecast_record_to_payload`'s JSON-safety (no
float leaf anywhere, datetimes as ISO-8601 `Z` strings, tuple-of-dataclass
fields projected as lists of dicts). `windbreak/forecast/` does not exist yet,
so importing `windbreak.forecast.records` fails collection with
`ModuleNotFoundError: No module named 'windbreak.forecast'` -- the expected
Gate 1 RED state for issue #22.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from datetime import UTC, datetime

import pytest

from windbreak.forecast.records import (
    BaselineQuoteSnapshot,
    Citation,
    ForecastRecord,
    ModelVote,
    forecast_record_to_payload,
)
from windbreak.forecast.sanitize import (
    DATA_BLOCK_BEGIN,
    DATA_BLOCK_END,
    TOOL_CALL_MARKERS,
    screen_single_line_text,
)

_VALID_MODEL_VOTE_KWARGS: dict[str, object] = {
    "provider": "openai",
    "model_version": "gpt-5-forecast",
    "declared_training_cutoff": "2024-06-01",
    "probability_ppm": 650_000,
    "response_fingerprint": "sha256:deadbeef",
}

_VALID_CITATION_KWARGS: dict[str, object] = {
    "url": "https://example.com/article",
    "content_hash": "sha256:cafebabe",
    "quoted_text": "The Fed is expected to raise rates.",
    "publication_date": datetime(2024, 11, 1, tzinfo=UTC),
    "source_type": "news_article",
}

_VALID_BASELINE_KWARGS: dict[str, object] = {
    "snapshot_id": "snap-0001",
    "price_pips": 4500,
    "fetched_at": datetime(2024, 12, 10, 12, 0, tzinfo=UTC),
}


def _model_vote(**overrides: object) -> ModelVote:
    return ModelVote(**{**_VALID_MODEL_VOTE_KWARGS, **overrides})


def _citation(**overrides: object) -> Citation:
    return Citation(**{**_VALID_CITATION_KWARGS, **overrides})


def _baseline(**overrides: object) -> BaselineQuoteSnapshot:
    return BaselineQuoteSnapshot(**{**_VALID_BASELINE_KWARGS, **overrides})


_VALID_FORECAST_RECORD_KWARGS: dict[str, object] = {
    "forecast_id": "fc-0001",
    "market_ticker": "KXFED-24DEC",
    "normalized_question_hash": "sha256:question-hash",
    "probability_ppm": 620_000,
    "ci_low_ppm": 550_000,
    "ci_high_ppm": 690_000,
    "model_votes": (
        _model_vote(probability_ppm=600_000),
        _model_vote(probability_ppm=620_000),
        _model_vote(probability_ppm=640_000),
    ),
    "vote_dispersion_ppm": 40_000,
    "rationale_markdown": "## Rationale\n\nBased on polling trends.",
    "citations": (_citation(),),
    "source_quality_notes": ("primary source", "cross-checked"),
    "research_cost_micros": 125_000,
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
    return ForecastRecord(**{**_VALID_FORECAST_RECORD_KWARGS, **overrides})


# --- ForecastRecord: construction and validation ---------------------------------


def test_valid_forecast_record_constructs_without_error() -> None:
    """A fully valid ForecastRecord constructs and preserves its fields."""
    record = _record()

    assert record.forecast_id == "fc-0001"
    assert record.triage_stage == "full"
    assert len(record.model_votes) == 3


@pytest.mark.parametrize("field", ["probability_ppm", "ci_low_ppm", "ci_high_ppm"])
@pytest.mark.parametrize("bad_value", [-1, 1_000_001])
def test_out_of_range_ppm_field_raises_value_error(field: str, bad_value: int) -> None:
    """probability_ppm / ci_low_ppm / ci_high_ppm must stay within [0, 1e6]."""
    with pytest.raises(ValueError, match=field):
        _record(**{field: bad_value})


@pytest.mark.parametrize("field", ["probability_ppm", "ci_low_ppm", "ci_high_ppm"])
def test_bool_ppm_field_raises_type_error(field: str) -> None:
    """A stray `bool` (an `int` subclass) must never masquerade as a ppm value."""
    with pytest.raises(TypeError):
        _record(**{field: True})


@pytest.mark.parametrize("field", ["forecast_id", "market_ticker"])
def test_empty_identifier_field_raises_value_error(field: str) -> None:
    """`forecast_id` and `market_ticker` must both be non-empty."""
    with pytest.raises(ValueError, match=field):
        _record(**{field: ""})


def test_bad_triage_stage_raises_value_error() -> None:
    """`triage_stage` outside {"triage_only", "full"} raises `ValueError`."""
    with pytest.raises(ValueError, match="triage_stage"):
        _record(triage_stage="not_a_real_stage")


@pytest.mark.parametrize("triage_stage", ["triage_only", "full"])
def test_valid_triage_stage_values_are_accepted(triage_stage: str) -> None:
    """Both sanctioned `triage_stage` values construct without error.

    `eligible_for_live` is pinned per stage (rather than left at the
    fixture's `full`-record default) so this test stays valid alongside the
    `triage_stage`/`eligible_for_live` invariant pinned below (issue #23):
    a `triage_only` record is never `eligible_for_live`.
    """
    record = _record(
        triage_stage=triage_stage, eligible_for_live=(triage_stage == "full")
    )

    assert record.triage_stage == triage_stage


def test_coherence_flag_and_eligible_for_live_both_true_raises() -> None:
    """A record cannot be both incoherence-flagged and live-eligible (#25):
    `coherence_flag=True` must forbid `eligible_for_live=True`.
    """
    with pytest.raises(ValueError, match=r"coherence_flag|eligible_for_live"):
        _record(coherence_flag=True, eligible_for_live=True)


def test_triage_only_eligible_for_live_combination_raises_value_error() -> None:
    """A `triage_stage="triage_only"` record can never be `eligible_for_live` (#23).

    A triage-only record was never backed by the full pipeline's research, so
    it must never be eligible to back a live order; the error message must
    name both offending fields so a caller can see the conflicting pair at a
    glance.
    """
    with pytest.raises(ValueError) as exc_info:
        _record(triage_stage="triage_only", eligible_for_live=True)

    message = str(exc_info.value)
    assert "triage_stage" in message
    assert "eligible_for_live" in message


@pytest.mark.parametrize(
    ("triage_stage", "eligible_for_live"),
    [("triage_only", False), ("full", True)],
)
def test_sanctioned_triage_stage_eligible_for_live_combinations_construct(
    triage_stage: str, eligible_for_live: bool
) -> None:
    """Both sanctioned `(triage_stage, eligible_for_live)` pairings construct fine.

    Regression coverage for #23: a `triage_only` record that correctly
    declares itself ineligible, and a `full` record that correctly declares
    itself eligible, must both continue to construct without error.
    """
    record = _record(triage_stage=triage_stage, eligible_for_live=eligible_for_live)

    assert record.triage_stage == triage_stage
    assert record.eligible_for_live is eligible_for_live


@pytest.mark.parametrize(
    ("triage_stage", "coherence_flag"),
    [
        ("triage_only", True),
        ("triage_only", False),
        ("full", True),
    ],
)
def test_any_live_ineligibility_trigger_forbids_eligible_for_live(
    triage_stage: str, coherence_flag: bool
) -> None:
    """The two live-ineligibility triggers compose (#23 + #25).

    `triage_stage="triage_only"` and `coherence_flag=True` each independently
    force live-ineligibility, so any record carrying either trigger -- or both
    together -- must reject `eligible_for_live=True`.
    """
    with pytest.raises(ValueError, match=r"eligible_for_live|triage_stage"):
        _record(
            triage_stage=triage_stage,
            coherence_flag=coherence_flag,
            eligible_for_live=True,
        )


def test_coherence_flagged_full_record_is_live_eligible_when_not_flagged() -> None:
    """Neither trigger firing leaves a `full` record free to be live-eligible.

    Confirms the composed guard is not over-broad: a `full`, coherent record
    (both triggers absent) still constructs with `eligible_for_live=True`.
    """
    record = _record(triage_stage="full", coherence_flag=False, eligible_for_live=True)

    assert record.eligible_for_live is True


def test_forecast_record_is_frozen() -> None:
    """Mutating any field of a constructed ForecastRecord raises."""
    record = _record()

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        record.probability_ppm = 999_999


# --- ModelVote: construction and validation --------------------------------------


def test_model_vote_constructs_with_valid_fields() -> None:
    """A valid ModelVote constructs and preserves its fields."""
    vote = _model_vote()

    assert vote.provider == "openai"
    assert vote.probability_ppm == 650_000


@pytest.mark.parametrize("bad_value", [-1, 1_000_001])
def test_model_vote_out_of_range_probability_raises_value_error(
    bad_value: int,
) -> None:
    """A ModelVote's probability_ppm must stay within [0, 1_000_000]."""
    with pytest.raises(ValueError, match="probability_ppm"):
        _model_vote(probability_ppm=bad_value)


def test_model_vote_bool_probability_raises_type_error() -> None:
    """A stray `bool` must never masquerade as a ModelVote probability."""
    with pytest.raises(TypeError):
        _model_vote(probability_ppm=True)


def test_model_vote_is_frozen() -> None:
    """Mutating any field of a constructed ModelVote raises."""
    vote = _model_vote()

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        vote.provider = "anthropic"


# --- Citation: construction -------------------------------------------------------


def test_citation_constructs_with_valid_fields() -> None:
    """A valid Citation constructs and preserves its fields."""
    citation = _citation()

    assert citation.source_type == "news_article"
    assert citation.url == "https://example.com/article"


def test_citation_accepts_none_publication_date() -> None:
    """A Citation with an unknown publication date accepts `None`."""
    citation = _citation(publication_date=None)

    assert citation.publication_date is None


def test_citation_is_frozen() -> None:
    """Mutating any field of a constructed Citation raises."""
    citation = _citation()

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        citation.url = "https://example.com/other"


# --- BaselineQuoteSnapshot: construction and validation --------------------------


def test_baseline_quote_snapshot_constructs_with_valid_fields() -> None:
    """A valid BaselineQuoteSnapshot constructs and preserves its fields."""
    baseline = _baseline()

    assert baseline.price_pips == 4500
    assert baseline.snapshot_id == "snap-0001"


@pytest.mark.parametrize("bad_value", [0, -1])
def test_baseline_non_positive_price_pips_raises_value_error(bad_value: int) -> None:
    """`price_pips` must be a strictly positive integer."""
    with pytest.raises(ValueError, match="price_pips"):
        _baseline(price_pips=bad_value)


def test_baseline_bool_price_pips_raises_type_error() -> None:
    """A stray `bool` must never masquerade as `price_pips`."""
    with pytest.raises(TypeError):
        _baseline(price_pips=True)


def test_baseline_empty_snapshot_id_raises_value_error() -> None:
    """`snapshot_id` must be non-empty."""
    with pytest.raises(ValueError, match="snapshot_id"):
        _baseline(snapshot_id="")


def test_baseline_quote_snapshot_is_frozen() -> None:
    """Mutating any field of a constructed BaselineQuoteSnapshot raises."""
    baseline = _baseline()

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        baseline.snapshot_id = "other"


# --- forecast_record_to_payload: JSON-safety -------------------------------------


def _assert_no_float_leaf(node: object) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _assert_no_float_leaf(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _assert_no_float_leaf(item)
    else:
        assert type(node) is not float, f"float leaf found in payload: {node!r}"


def test_forecast_record_to_payload_is_json_dumps_clean() -> None:
    """The payload round-trips cleanly through `json.dumps`/`json.loads`."""
    record = _record()

    payload = forecast_record_to_payload(record)

    assert json.loads(json.dumps(payload)) == payload


def test_forecast_record_to_payload_datetimes_are_iso_z_strings() -> None:
    """`created_at` is rendered as an ISO-8601 `Z` string."""
    record = _record()

    payload = forecast_record_to_payload(record)

    assert payload["created_at"] == "2024-12-10T12:00:00.000000Z"


def test_forecast_record_to_payload_renders_model_votes_as_list_of_dicts() -> None:
    """`model_votes` (a tuple of ModelVote) becomes a list of plain dicts."""
    record = _record()

    payload = forecast_record_to_payload(record)
    votes = payload["model_votes"]

    assert isinstance(votes, list)
    assert len(votes) == 3
    assert all(isinstance(vote, dict) for vote in votes)
    assert votes[0]["provider"] == "openai"
    assert votes[0]["probability_ppm"] == 600_000


def test_forecast_record_to_payload_renders_citations_as_list_of_dicts() -> None:
    """`citations` (a tuple of Citation) becomes a list of plain dicts, with
    the citation's own datetime field also projected as an ISO-8601 `Z` string.
    """
    record = _record()

    payload = forecast_record_to_payload(record)
    citations = payload["citations"]

    assert isinstance(citations, list)
    assert len(citations) == 1
    assert citations[0]["url"] == "https://example.com/article"
    assert citations[0]["publication_date"] == "2024-11-01T00:00:00.000000Z"


def test_payload_renders_source_quality_notes_as_list_of_str() -> None:
    """`source_quality_notes` (a tuple of str) becomes a plain list of str."""
    record = _record()

    payload = forecast_record_to_payload(record)

    assert payload["source_quality_notes"] == ["primary source", "cross-checked"]


def test_forecast_record_to_payload_preserves_none_fields() -> None:
    """Optional fields that are `None` stay `None` in the payload."""
    record = _record(coherence_group_sum_ppm=None, abstention_reason=None)

    payload = forecast_record_to_payload(record)

    assert payload["coherence_group_sum_ppm"] is None
    assert payload["abstention_reason"] is None


def test_forecast_record_to_payload_contains_no_float_leaf() -> None:
    """No leaf anywhere in the payload is a `float`."""
    record = _record()

    payload = forecast_record_to_payload(record)

    _assert_no_float_leaf(payload)


# --- The ticker cannot be unscreened, by construction (issue #525) ----------------
#
# `windbreak/scheduler/loop.py` appends `ForecastCreated(market_ticker=
# forecast.market_ticker)` straight off the record, into an append-only hash
# chain nothing can ever redact. Screening at that append would be coverage by
# convention -- one call, deletable, and re-added by hand for every future
# consumer. Screening here makes it coverage by construction: no `ForecastRecord`
# exists carrying a ticker the provider seam would refuse, so no consumer of one
# can leak it.
#
# The corpus below is derived from `sanitize`'s own public constants, so a screen
# that learns a new artifact is exercised here without this file being edited.

_HOSTILE_TICKER_CANDIDATES: tuple[str, ...] = (
    f"KXFED{DATA_BLOCK_BEGIN}",
    f"KXFED{DATA_BLOCK_END}",
    *(f"KXFED{marker}" for marker in sorted(TOOL_CALL_MARKERS)),
    "KXFED\n24DEC",
    "KXFED\r24DEC",
)

_CLEAN_TICKER_CANDIDATES: tuple[str, ...] = ("KXFED-24DEC", "MKT-ISO-A", "A")

_TICKER_CANDIDATES: tuple[str, ...] = (
    *_CLEAN_TICKER_CANDIDATES,
    *_HOSTILE_TICKER_CANDIDATES,
)


def test_the_ticker_corpus_exercises_both_verdicts() -> None:
    """Guard the parametrization itself: both arms below are really reached.

    A corpus that screened clean on every entry would make the equivalence test
    vacuously true, and a corpus that screened hostile on every entry would let
    a record type that refuses *everything* pass. Both halves are asserted
    against the screen, not against a hand-written count.
    """
    verdicts = {
        ticker: screen_single_line_text(ticker) for ticker in _TICKER_CANDIDATES
    }

    assert {ticker for ticker, v in verdicts.items() if v is None} == set(
        _CLEAN_TICKER_CANDIDATES
    )
    assert {ticker for ticker, v in verdicts.items() if v is not None} == set(
        _HOSTILE_TICKER_CANDIDATES
    )


@pytest.mark.parametrize("ticker", _TICKER_CANDIDATES)
def test_a_record_admits_a_ticker_exactly_when_the_seam_screen_does(
    ticker: str,
) -> None:
    """`ForecastRecord` accepts a ticker iff `screen_single_line_text` does.

    Stated as an equivalence against the screen rather than as a list of banned
    strings: a guard wired to a *narrower* check would leak exactly the
    artifacts the provider seam already knows are hostile, and would still pass
    a test that only listed today's tokens.
    """
    verdict = screen_single_line_text(ticker)

    if verdict is None:
        assert _record(market_ticker=ticker).market_ticker == ticker
        return
    with pytest.raises(ValueError, match=re.escape(f"market_ticker: {verdict}")):
        _record(market_ticker=ticker)


def test_a_refused_ticker_is_never_echoed_into_the_error_message() -> None:
    """The refusal names the verdict and a digest, never the attacker's bytes.

    An exception message travels into logs and, via a caller that formats it,
    could travel further. Repeating the ticker there would defeat the point of
    keeping it off the chain.
    """
    ticker = f"KXFED{DATA_BLOCK_BEGIN} MKT-EVIL-9F3"

    with pytest.raises(ValueError) as refusal:
        _record(market_ticker=ticker)

    message = str(refusal.value)
    assert "MKT-EVIL-9F3" not in message
    assert DATA_BLOCK_BEGIN not in message
    assert hashlib.sha256(ticker.encode("utf-8")).hexdigest() in message
