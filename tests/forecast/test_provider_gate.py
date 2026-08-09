"""Tests for windbreak.forecast.providers.track_record (issue #194): a
per-provider track-record live-eligibility gate.

Pins `ProviderTrackRecordGate.is_provider_proven`'s ">=" boundary on both
`resolved_count` and `brier_skill_ppm` (missing-record and negative-skill both
fail closed to "unproven"), `unproven_providers`'s sorted/deduped output,
`parse_track_records`'s fail-closed strict-JSON read model, and `run_pipeline`'s
new `provider_gate` seam: an unproven voting provider forces
`eligible_for_live=False` and ledgers exactly one `PROVIDER_GATE_HELD` event
(even when a breached `CanaryGate` *also* blocks -- no short-circuit), while
every voting provider's votes still run and are recorded regardless of
eligibility. `windbreak/forecast/providers/track_record.py` does not exist
yet, so importing it fails collection with `ModuleNotFoundError: No module
named 'windbreak.forecast.providers.track_record'` -- the expected Gate 1 RED
state for issue #194.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.canary import CanaryGate, CanaryRunResult, InMemoryCanaryLedger
from windbreak.forecast.pipeline import (
    CALIBRATION_MAP_APPLIED_EVENT,
    PROVIDER_GATE_HELD_EVENT,
    InMemoryForecastLedger,
    run_pipeline,
)
from windbreak.forecast.providers.track_record import (
    DEFAULT_MIN_BRIER_SKILL_PPM,
    DEFAULT_MIN_RESOLVED,
    InMemoryTrackRecordSource,
    ProviderTrackRecord,
    ProviderTrackRecordGate,
    parse_track_records,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

    #: See `tests/forecast/conftest.py`'s "Sandbox-transport fixture choice"
    #: note for why `make_fake_vote_transport` is typed structurally here.
    FakeVoteTransportFactory = Callable[..., object]

#: The default vote ensemble's two distinct providers (SPEC S6.3 / issue
#: #191's `DEFAULT_VOTE_ENSEMBLE`): two OpenAI members and one Anthropic
#: member. A track-record source must cover both for a full run to be proven.
_PROVIDER_OPENAI = "openai"
_PROVIDER_ANTHROPIC = "anthropic"

#: A third, non-voting provider name used only to exercise
#: `unproven_providers`'s handling of a research-forecaster-style provider
#: that never actually appears among a run's votes.
_PROVIDER_FUTURESEARCH = "futuresearch"

#: A provider name with no track record at all, for the fail-closed
#: missing-record cases.
_PROVIDER_UNKNOWN = "unknown-provider"

#: The three canned vote probabilities `tests/forecast/conftest.py`'s
#: `CANNED_VOTE_RESPONSES` produce, in call order.
_CANNED_VOTE_PROBABILITIES_PPM = (440_000, 450_000, 460_000)


class _NoOpAlertEmitter:
    """A `CanaryAlertEmitter` double that drops every dispatched call.

    Mirrors `tests/forecast/test_canary.py`'s `RecordingAlertEmitter` shape
    but discards the call, since these tests only care about the resulting
    `CanaryGate` state, never the alert content.
    """

    def dispatch(self, alert_type: object, message: str) -> object:
        """Discard the call and return an opaque sentinel.

        Args:
            alert_type: The (unused) alert type dispatched.
            message: The (unused) alert body.

        Returns:
            A sentinel object; callers never inspect this seam's return value.
        """
        return object()


def _breached_canary_gate(*, checked_at: datetime) -> CanaryGate:
    """Build a `CanaryGate` already breached at `checked_at` (mirrors
    `tests/forecast/test_canary.py`'s drifted-gate pipeline fixture).

    Args:
        checked_at: The instant the breach is recorded at.

    Returns:
        A `CanaryGate` whose `is_live_blocked` is `True` from `checked_at`
        onward.
    """
    gate = CanaryGate()
    result = CanaryRunResult(
        distances_ppm={"q1": 100_000}, drift_score_ppm=100_000, worst_question_id="q1"
    )
    gate.apply_run(
        result,
        checked_at=checked_at,
        alerts=_NoOpAlertEmitter(),
        ledger=InMemoryCanaryLedger(),
    )
    return gate


# --- ProviderTrackRecord -------------------------------------------------------------


def test_provider_track_record_rejects_empty_provider() -> None:
    """A blank provider identifier is never a valid track record."""
    with pytest.raises(ValueError, match="provider"):
        ProviderTrackRecord(provider="", resolved_count=10, brier_skill_ppm=0)


def test_provider_track_record_rejects_negative_resolved_count() -> None:
    """`resolved_count` must be non-negative."""
    with pytest.raises(ValueError, match="resolved_count"):
        ProviderTrackRecord(
            provider=_PROVIDER_OPENAI, resolved_count=-1, brier_skill_ppm=0
        )


def test_provider_track_record_allows_negative_brier_skill_ppm() -> None:
    """A negative Brier skill (worse than baseline) is a valid, if unproven,
    track record -- it must construct without raising."""
    record = ProviderTrackRecord(
        provider=_PROVIDER_OPENAI, resolved_count=500, brier_skill_ppm=-5_000
    )

    assert record.brier_skill_ppm == -5_000


# --- InMemoryTrackRecordSource --------------------------------------------------------


def test_in_memory_track_record_source_returns_record_for_known_provider() -> None:
    """The source returns the exact record it was constructed with, and
    `None` for a provider it was never given."""
    record = ProviderTrackRecord(
        provider=_PROVIDER_OPENAI, resolved_count=200, brier_skill_ppm=20_000
    )
    source = InMemoryTrackRecordSource([record])

    assert source.track_record_for(_PROVIDER_OPENAI) == record
    assert source.track_record_for(_PROVIDER_UNKNOWN) is None


# --- parse_track_records: fail-closed strict-JSON read model -------------------------

_VALID_TRACK_RECORD_JSON = (
    '{"openai": {"resolved_count": 200, "brier_skill_ppm": 20000}, '
    '"anthropic": {"resolved_count": 150, "brier_skill_ppm": 10000}}'
)

_FLOAT_LEAF_TRACK_RECORD_JSON = (
    '{"openai": {"resolved_count": 200, "brier_skill_ppm": 20000.5}}'
)

_BOOL_WHERE_INT_TRACK_RECORD_JSON = (
    '{"openai": {"resolved_count": true, "brier_skill_ppm": 20000}}'
)

_UNKNOWN_KEY_TRACK_RECORD_JSON = (
    '{"openai": {"resolved_count": 200, "brier_skill_ppm": 20000, "bogus_key": 1}}'
)

#: A JSON document whose root is an array, not the required provider object.
_NON_OBJECT_ROOT_TRACK_RECORD_JSON = "[1, 2, 3]"

#: A provider entry that is a bare scalar rather than the required mapping.
_NON_MAPPING_ENTRY_TRACK_RECORD_JSON = '{"openai": 200}'

#: A provider entry that omits the required `brier_skill_ppm` measurement.
_MISSING_MEASUREMENT_TRACK_RECORD_JSON = '{"openai": {"resolved_count": 200}}'


def test_parse_track_records_round_trips_valid_json() -> None:
    """Valid JSON parses into exactly the expected `ProviderTrackRecord`
    mapping, keyed by provider."""
    records = parse_track_records(_VALID_TRACK_RECORD_JSON)

    assert records == {
        _PROVIDER_OPENAI: ProviderTrackRecord(
            provider=_PROVIDER_OPENAI, resolved_count=200, brier_skill_ppm=20_000
        ),
        _PROVIDER_ANTHROPIC: ProviderTrackRecord(
            provider=_PROVIDER_ANTHROPIC, resolved_count=150, brier_skill_ppm=10_000
        ),
    }


def test_parse_track_records_rejects_float_leaf() -> None:
    """A JSON float leaf (`20000.5`) is fail-closed rejected, never silently
    truncated to an int."""
    with pytest.raises(ValueError):
        parse_track_records(_FLOAT_LEAF_TRACK_RECORD_JSON)


def test_parse_track_records_rejects_bool_where_int_expected() -> None:
    """A JSON `true`/`false` where an integer count is expected is rejected,
    since a bare bool is a `bool` subclass of `int` and would otherwise slip
    through unchecked."""
    with pytest.raises((TypeError, ValueError)):
        parse_track_records(_BOOL_WHERE_INT_TRACK_RECORD_JSON)


def test_parse_track_records_rejects_unknown_per_entry_key() -> None:
    """An unrecognized key inside one provider's entry is fatal."""
    with pytest.raises(ValueError, match="bogus_key"):
        parse_track_records(_UNKNOWN_KEY_TRACK_RECORD_JSON)


def test_parse_track_records_rejects_non_object_root() -> None:
    """A JSON document whose root is not a provider object is fail-closed:
    a malformed artifact can never be silently read as an empty record set."""
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_track_records(_NON_OBJECT_ROOT_TRACK_RECORD_JSON)


def test_parse_track_records_rejects_non_mapping_entry() -> None:
    """A provider entry that is a bare scalar (not a mapping) is fatal, so a
    malformed entry can never promote a provider."""
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_track_records(_NON_MAPPING_ENTRY_TRACK_RECORD_JSON)


def test_parse_track_records_rejects_missing_measurement() -> None:
    """An entry omitting a required measurement is fatal: an absent
    `brier_skill_ppm` can never be read as a passing (or zero) skill."""
    with pytest.raises(ValueError, match="brier_skill_ppm"):
        parse_track_records(_MISSING_MEASUREMENT_TRACK_RECORD_JSON)


# --- ProviderTrackRecordGate: is_provider_proven boundary ----------------------------


def test_is_provider_proven_exact_boundary_is_proven() -> None:
    """A record exactly at both thresholds (`>=` on each) is proven."""
    source = InMemoryTrackRecordSource(
        [
            ProviderTrackRecord(
                provider=_PROVIDER_OPENAI,
                resolved_count=DEFAULT_MIN_RESOLVED,
                brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM,
            )
        ]
    )
    gate = ProviderTrackRecordGate(source)

    assert gate.is_provider_proven(_PROVIDER_OPENAI) is True


def test_is_provider_proven_one_below_resolved_threshold_is_unproven() -> None:
    """One resolved forecast short of the threshold is unproven."""
    source = InMemoryTrackRecordSource(
        [
            ProviderTrackRecord(
                provider=_PROVIDER_OPENAI,
                resolved_count=DEFAULT_MIN_RESOLVED - 1,
                brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM,
            )
        ]
    )
    gate = ProviderTrackRecordGate(source)

    assert gate.is_provider_proven(_PROVIDER_OPENAI) is False


def test_is_provider_proven_one_below_skill_threshold_is_unproven() -> None:
    """One ppm short of the Brier-skill threshold is unproven."""
    source = InMemoryTrackRecordSource(
        [
            ProviderTrackRecord(
                provider=_PROVIDER_OPENAI,
                resolved_count=DEFAULT_MIN_RESOLVED,
                brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM - 1,
            )
        ]
    )
    gate = ProviderTrackRecordGate(source)

    assert gate.is_provider_proven(_PROVIDER_OPENAI) is False


def test_is_provider_proven_missing_record_is_unproven() -> None:
    """A provider with no track record at all fails closed to unproven."""
    gate = ProviderTrackRecordGate(InMemoryTrackRecordSource([]))

    assert gate.is_provider_proven(_PROVIDER_UNKNOWN) is False


def test_gate_min_resolved_and_min_brier_skill_ppm_properties_are_read_only() -> None:
    """The gate's thresholds are exposed read-only, carrying the caller's
    override values."""
    gate = ProviderTrackRecordGate(
        InMemoryTrackRecordSource([]), min_resolved=200, min_brier_skill_ppm=5_000
    )

    assert gate.min_resolved == 200
    assert gate.min_brier_skill_ppm == 5_000
    with pytest.raises(AttributeError):
        gate.min_resolved = 1  # type: ignore[misc]


def test_gate_default_thresholds_match_module_constants() -> None:
    """With no override, the gate's thresholds are the module defaults."""
    gate = ProviderTrackRecordGate(InMemoryTrackRecordSource([]))

    assert gate.min_resolved == DEFAULT_MIN_RESOLVED
    assert gate.min_brier_skill_ppm == DEFAULT_MIN_BRIER_SKILL_PPM


# --- ProviderTrackRecordGate.unproven_providers: sorted, deduped, fail-closed -------


def test_unproven_providers_sorted_deduped_and_fail_closed() -> None:
    """A mix of one proven provider, one too-few-resolved provider, one
    negative-Brier-skill provider, a duplicated input, and one entirely
    missing provider yields the unproven set, sorted and deduped."""
    source = InMemoryTrackRecordSource(
        [
            ProviderTrackRecord(
                provider=_PROVIDER_OPENAI, resolved_count=200, brier_skill_ppm=20_000
            ),
            ProviderTrackRecord(
                provider=_PROVIDER_ANTHROPIC, resolved_count=12, brier_skill_ppm=40_000
            ),
            ProviderTrackRecord(
                provider=_PROVIDER_FUTURESEARCH,
                resolved_count=500,
                brier_skill_ppm=-1_000,
            ),
        ]
    )
    gate = ProviderTrackRecordGate(source)

    result = gate.unproven_providers(
        [
            _PROVIDER_FUTURESEARCH,
            _PROVIDER_FUTURESEARCH,
            _PROVIDER_OPENAI,
            _PROVIDER_ANTHROPIC,
            _PROVIDER_UNKNOWN,
        ]
    )

    assert result == (_PROVIDER_ANTHROPIC, _PROVIDER_FUTURESEARCH, _PROVIDER_UNKNOWN)


def test_unproven_providers_empty_when_every_provider_proven() -> None:
    """A fully-proven provider set yields an empty unproven tuple."""
    source = InMemoryTrackRecordSource(
        [
            ProviderTrackRecord(
                provider=_PROVIDER_OPENAI,
                resolved_count=DEFAULT_MIN_RESOLVED,
                brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM,
            ),
            ProviderTrackRecord(
                provider=_PROVIDER_ANTHROPIC,
                resolved_count=DEFAULT_MIN_RESOLVED,
                brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM,
            ),
        ]
    )
    gate = ProviderTrackRecordGate(source)

    assert gate.unproven_providers([_PROVIDER_OPENAI, _PROVIDER_ANTHROPIC]) == ()


# --- run_pipeline integration: the provider_gate seam --------------------------------


def test_run_pipeline_unproven_provider_forces_ineligible_but_votes_still_run(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """Issue #194's own worked example, adapted to `run_pipeline`: an
    unproven provider (12 resolved, 40_000 ppm skill) among the default vote
    ensemble forces `eligible_for_live=False` and ledgers exactly one
    `PROVIDER_GATE_HELD` event -- while every vote still runs and is
    recorded (only eligibility is forced, never the votes themselves).
    """
    source = InMemoryTrackRecordSource(
        [
            ProviderTrackRecord(
                provider=_PROVIDER_OPENAI, resolved_count=12, brier_skill_ppm=40_000
            ),
            ProviderTrackRecord(
                provider=_PROVIDER_ANTHROPIC,
                resolved_count=DEFAULT_MIN_RESOLVED,
                brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM,
            ),
        ]
    )
    gate = ProviderTrackRecordGate(source)
    ledger = InMemoryForecastLedger()

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        provider_gate=gate,
        ledger=ledger,
    )

    assert record.eligible_for_live is False
    assert record.abstention_reason is None
    assert record.triage_stage == "full"
    assert [vote.probability_ppm for vote in record.model_votes] == list(
        _CANNED_VOTE_PROBABILITIES_PPM
    )

    events = ledger.events_by_type(PROVIDER_GATE_HELD_EVENT)
    assert len(events) == 1
    assert events[0].payload == {
        "unproven_providers": _PROVIDER_OPENAI,
        "unproven_count": 1,
        "min_resolved": DEFAULT_MIN_RESOLVED,
        "min_brier_skill_ppm": DEFAULT_MIN_BRIER_SKILL_PPM,
    }


def test_run_pipeline_proven_providers_at_exact_boundary_opens_gate(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """Every voting provider proven at the exact boundary opens the gate:
    live-eligible (given the fixture's default 3 verified citations and an
    open canary gate), with zero `PROVIDER_GATE_HELD` events."""
    source = InMemoryTrackRecordSource(
        [
            ProviderTrackRecord(
                provider=_PROVIDER_OPENAI,
                resolved_count=DEFAULT_MIN_RESOLVED,
                brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM,
            ),
            ProviderTrackRecord(
                provider=_PROVIDER_ANTHROPIC,
                resolved_count=DEFAULT_MIN_RESOLVED,
                brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM,
            ),
        ]
    )
    gate = ProviderTrackRecordGate(source)
    ledger = InMemoryForecastLedger()

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        provider_gate=gate,
        ledger=ledger,
    )

    assert record.eligible_for_live is True
    assert ledger.events_by_type(PROVIDER_GATE_HELD_EVENT) == ()


def test_run_pipeline_provider_gate_held_ledgered_even_when_canary_also_blocks(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """A breached canary gate ALSO blocking the run must never short-circuit
    the provider-gate check: `PROVIDER_GATE_HELD` is still ledgered exactly
    once, and the run is ineligible for both independent reasons."""
    canary_gate = _breached_canary_gate(checked_at=created_at)
    source = InMemoryTrackRecordSource(
        [
            ProviderTrackRecord(
                provider=_PROVIDER_OPENAI, resolved_count=12, brier_skill_ppm=40_000
            ),
            ProviderTrackRecord(
                provider=_PROVIDER_ANTHROPIC,
                resolved_count=DEFAULT_MIN_RESOLVED,
                brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM,
            ),
        ]
    )
    provider_gate = ProviderTrackRecordGate(source)
    ledger = InMemoryForecastLedger()

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        canary_gate=canary_gate,
        provider_gate=provider_gate,
        ledger=ledger,
    )

    assert record.eligible_for_live is False
    held_events = ledger.events_by_type(PROVIDER_GATE_HELD_EVENT)
    assert len(held_events) == 1
    assert held_events[0].payload["unproven_providers"] == _PROVIDER_OPENAI


# --- Tracer invariant: both new seams default to a byte-identical no-op -----------


def test_run_pipeline_new_seams_default_to_byte_identical_tracer_path(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """With both `calibration_map` and `provider_gate` left at their `None`
    defaults, a run is `==` to a run that never passes those keywords at
    all, and a wired ledger gains zero events of either new type."""
    ledger = InMemoryForecastLedger()

    record_with_explicit_none = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        calibration_map=None,
        provider_gate=None,
        ledger=ledger,
    )
    record_without_new_kwargs = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
    )

    assert record_with_explicit_none == record_without_new_kwargs
    assert ledger.events_by_type(CALIBRATION_MAP_APPLIED_EVENT) == ()
    assert ledger.events_by_type(PROVIDER_GATE_HELD_EVENT) == ()
