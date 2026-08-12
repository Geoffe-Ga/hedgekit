"""Tests for `windbreak.forecast.pipeline`'s abstention-rationale guard and its
three optional-observability `None` arms (issue #314).

An abstention that records no rationale is an unexplained decision in the audit
trail, so `_build_abstention_record` refuses an `abstention_reason` it has no
registered rationale for rather than shipping a mismatched one. The three
optional seams -- the vote loop's discard recorder, the calibration-map ledger,
and the provider-gate ledger -- each default to `None`, and each `None` arm must
be a pure *observability* no-op: the pipeline still discards the failing vote,
still charges its cost, still applies the calibration map, and still holds the
provider gate when nothing is wired to watch. A `None` ledger that also dropped
the *decision* would be the fail-open shape this backlog is clearing.

Every ledgerless assertion below is paired with a wired-ledger control run, so
"the two agree" is proven against a case where the ledger genuinely emits, and
against a third run where the seam is absent entirely -- never against a value
that would coincide either way.

Local-doubles choice
    This module defines its own `_SucceedingProvider`/`_FailingProvider`/
    `_routed_provider_factory` doubles rather than importing them from a sibling
    test module: cross-test-module imports are not supported under this
    project's rootdir-relative pytest import mode (see
    `tests/forecast/conftest.py` and `tests/forecast/test_provider_failures.py`
    for the same convention applied elsewhere).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.budget import (
    FULL_PIPELINE_RESEARCH_COST_MICROS,
    InMemoryBudgetLedger,
    PerForecastBudgetExceededError,
    ResearchBudget,
)
from windbreak.forecast.calibration import CalibrationMap
from windbreak.forecast.cassettes import ForbiddenLiveTransport
from windbreak.forecast.pipeline import (
    _ABSTENTION_RATIONALE_BY_REASON,
    ABSTENTION_ALL_VOTES_DISCARDED,
    ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED,
    ABSTENTION_ENSEMBLE_QUORUM_NOT_MET,
    ABSTENTION_NO_VERIFIED_CITATIONS,
    ABSTENTION_PROVIDER_UNAVAILABLE,
    CALIBRATION_MAP_APPLIED_EVENT,
    FORECAST_OUTPUT_DISCARDED_EVENT,
    PROVIDER_GATE_HELD_EVENT,
    InMemoryForecastLedger,
    _apply_and_record_calibration,
    _build_abstention_record,
    _provider_gate_open,
    collect_model_votes,
    normalize_question,
    run_pipeline,
)
from windbreak.forecast.providers import DEFAULT_VOTE_ENSEMBLE, ProviderForecast
from windbreak.forecast.providers.base import ProviderVoteError
from windbreak.forecast.providers.track_record import (
    DEFAULT_MIN_BRIER_SKILL_PPM,
    DEFAULT_MIN_RESOLVED,
    InMemoryTrackRecordSource,
    ProviderTrackRecord,
    ProviderTrackRecordGate,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.pipeline import ModelVote
    from windbreak.forecast.providers import EnsembleMemberLike, ForecastProvider
    from windbreak.forecast.records import BaselineQuoteSnapshot, ForecastRecord
    from windbreak.forecast.sandbox import ResearchTools

    #: See `tests/forecast/conftest.py`'s "Sandbox-transport fixture choice"
    #: note for why `make_fake_vote_transport` is typed structurally here.
    FakeVoteTransportFactory = Callable[..., object]

#: An `abstention_reason` deliberately absent from
#: `_ABSTENTION_RATIONALE_BY_REASON` -- the shape a future abstention path would
#: have if its author forgot to register a rationale alongside it.
_UNREGISTERED_ABSTENTION_REASON = "reason_with_no_registered_rationale"

#: The guard's exact rendered message for `_UNREGISTERED_ABSTENTION_REASON`,
#: pinned in full (not as a substring) so a drifting message body is caught.
_EXPECTED_GUARD_MESSAGE = (
    "No rationale registered for abstention reason "
    "'reason_with_no_registered_rationale'"
)

#: Every abstention reason the pipeline can stamp, in declaration order.
_REGISTERED_ABSTENTION_REASONS: tuple[str, ...] = (
    ABSTENTION_NO_VERIFIED_CITATIONS,
    ABSTENTION_ALL_VOTES_DISCARDED,
    ABSTENTION_ENSEMBLE_QUORUM_NOT_MET,
    ABSTENTION_PROVIDER_UNAVAILABLE,
    ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED,
)

#: The three pinned default ensemble members (SPEC S6.3). `_MEMBER_A` and
#: `_MEMBER_C` are both `"openai"` with distinct `model_version`s.
_MEMBER_A = DEFAULT_VOTE_ENSEMBLE[0]
_MEMBER_B = DEFAULT_VOTE_ENSEMBLE[1]
_MEMBER_C = DEFAULT_VOTE_ENSEMBLE[2]

#: The default ensemble's two distinct provider names.
_PROVIDER_OPENAI = "openai"
_PROVIDER_ANTHROPIC = "anthropic"

#: Three deliberately distinct vote costs, so a dropped or swapped leaf changes
#: the charged total rather than coinciding with another leaf's value.
_MEMBER_A_COST_MICROS = 100
_MEMBER_C_COST_MICROS = 300

#: The discarded vote's cost -- distinct from, and far larger than, both
#: surviving costs, so "the discard cost was tallied" is a different number from
#: every other combination of leaves.
_DISCARD_COST_MICROS = 700_000

#: The exact per-forecast research charge a two-survivor, one-discard run makes
#: when the discarded vote's cost is tallied (the behavior under test).
_EXPECTED_CHARGE_MICROS = (
    FULL_PIPELINE_RESEARCH_COST_MICROS
    + _MEMBER_A_COST_MICROS
    + _MEMBER_C_COST_MICROS
    + _DISCARD_COST_MICROS
)

#: The two surviving members' probabilities, in call order.
_MEMBER_A_PROBABILITY_PPM = 400_000
_MEMBER_C_PROBABILITY_PPM = 600_000

#: A fitted, two-breakpoint calibration map spanning only the interior of the
#: ppm domain, mirroring `tests/forecast/test_calibration_loader.py`'s map.
_FITTED_MAP_ID = "fitted-v1"

#: The fixture pipeline's `created_at` is 2024-12-10, so a map "trained" the
#: same day is temporally safe to wire into a full run.
_PIPELINE_SAFE_MAP_VERSION = "2024-12-10"

_INTERIOR_ENTRIES: tuple[tuple[int, int], ...] = (
    (200_000, 100_000),
    (800_000, 900_000),
)

#: An input ppm the fitted map above genuinely moves (it is not a fixed point),
#: so "calibrated" and "uncalibrated" are two different numbers.
_CALIBRATION_INPUT_PPM = 450_000


def _fitted_map(version: str = _PIPELINE_SAFE_MAP_VERSION) -> CalibrationMap:
    """Build the shared interior-breakpoint fitted calibration map.

    Args:
        version: The map's version string (an ISO date); defaults to the
            pipeline-safe same-day version.

    Returns:
        A `CalibrationMap` over `_INTERIOR_ENTRIES`.
    """
    return CalibrationMap(
        map_id=_FITTED_MAP_ID, version=version, entries=_INTERIOR_ENTRIES
    )


class _SucceedingProvider:
    """A `ForecastProvider` double returning one fixed forecast every call."""

    def __init__(self, forecast: ProviderForecast) -> None:
        """Store the forecast every `forecast` call returns.

        Args:
            forecast: The fixed `ProviderForecast` to return every time.
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
    """A `ForecastProvider` double raising a fixed error on every call."""

    def __init__(self, error: ProviderVoteError) -> None:
        """Store the error every `forecast` call raises.

        Args:
            error: The exception instance to raise, unmodified, every call.
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
            ProviderVoteError: The stored `self._error`, unconditionally.
        """
        raise self._error


def _routed_provider_factory(
    routes: dict[str, ForecastProvider],
) -> Callable[[EnsembleMemberLike], ForecastProvider]:
    """Build a `provider_factory` routing each member by its `model_version`.

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


def _member_provider(
    member: EnsembleMemberLike, *, probability_ppm: int, cost_micros: int
) -> ForecastProvider:
    """Build a `_SucceedingProvider` returning a clean, member-stamped forecast.

    Args:
        member: The ensemble member this provider's forecast is stamped with.
        probability_ppm: The forecast's probability estimate, in ppm.
        cost_micros: The forecast's billed cost, in micros.

    Returns:
        A `_SucceedingProvider` returning a valid, member-stamped forecast.
    """
    return _SucceedingProvider(
        ProviderForecast(
            probability_ppm=probability_ppm,
            rationale_summary="steady evidence",
            citations=(),
            cost_micros=cost_micros,
            provider=member.provider,
            model_version=member.model_version,
            training_cutoff=member.training_cutoff,
            response_fingerprint="f" * 64,
            abstain=False,
        )
    )


def _two_survivors_one_discard_routes() -> dict[str, ForecastProvider]:
    """Route members A and C to clean votes and member B to a discarded one.

    Returns:
        A `{model_version: ForecastProvider}` mapping for the three default
        ensemble members, each with a distinct cost.
    """
    return {
        _MEMBER_A.model_version: _member_provider(
            _MEMBER_A,
            probability_ppm=_MEMBER_A_PROBABILITY_PPM,
            cost_micros=_MEMBER_A_COST_MICROS,
        ),
        _MEMBER_B.model_version: _FailingProvider(
            ProviderVoteError(
                "rejected",
                failure_code="malformed_vote_json",
                response_fingerprint="a" * 64,
                cost_micros=_DISCARD_COST_MICROS,
            )
        ),
        _MEMBER_C.model_version: _member_provider(
            _MEMBER_C,
            probability_ppm=_MEMBER_C_PROBABILITY_PPM,
            cost_micros=_MEMBER_C_COST_MICROS,
        ),
    }


def _track_record_gate(*, openai_resolved: int) -> ProviderTrackRecordGate:
    """Build a gate whose Anthropic record is proven and OpenAI's is tunable.

    Args:
        openai_resolved: The OpenAI provider's resolved-market count. Below
            `DEFAULT_MIN_RESOLVED` the gate holds; at it, the gate opens.

    Returns:
        A `ProviderTrackRecordGate` over an in-memory two-provider source.
    """
    return ProviderTrackRecordGate(
        InMemoryTrackRecordSource(
            [
                ProviderTrackRecord(
                    provider=_PROVIDER_OPENAI,
                    resolved_count=openai_resolved,
                    brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM,
                ),
                ProviderTrackRecord(
                    provider=_PROVIDER_ANTHROPIC,
                    resolved_count=DEFAULT_MIN_RESOLVED,
                    brier_skill_ppm=DEFAULT_MIN_BRIER_SKILL_PPM,
                ),
            ]
        )
    )


def _canned_votes(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    make_fake_vote_transport: FakeVoteTransportFactory,
) -> tuple[ModelVote, ...]:
    """Collect the three canned fixture votes the provider gate screens.

    Args:
        market: The market under forecast.
        baseline: The baseline quote snapshot.
        make_fake_vote_transport: The canned-vote transport factory.

    Returns:
        The three surviving default-ensemble votes, in call order.
    """
    return collect_model_votes(market, baseline, transport=make_fake_vote_transport())


# --- The abstention-rationale guard -----------------------------------------


def test_build_abstention_record_refuses_a_reason_with_no_registered_rationale(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
) -> None:
    """An `abstention_reason` absent from `_ABSTENTION_RATIONALE_BY_REASON`
    raises `ValueError` naming the offending reason, chained from the raw
    `KeyError`, rather than shipping an unexplained abstention.

    The raised type is pinned with `type(...) is ValueError` (not
    `isinstance`), so a future subclass -- or an unrelated `ValueError`
    subclass raised from deeper in record construction -- cannot pass this
    test by accident, and the message is compared in full, not by substring.
    """
    with pytest.raises(ValueError) as excinfo:
        _build_abstention_record(
            market=market,
            baseline=baseline,
            created_at=created_at,
            question_hash=normalize_question(market),
            citations=(),
            abstention_reason=_UNREGISTERED_ABSTENTION_REASON,
            research_cost_micros=FULL_PIPELINE_RESEARCH_COST_MICROS,
        )

    error = excinfo.value
    assert type(error) is ValueError
    assert str(error) == _EXPECTED_GUARD_MESSAGE
    assert type(error.__cause__) is KeyError
    assert error.__cause__.args == (_UNREGISTERED_ABSTENTION_REASON,)


def test_build_abstention_record_gives_each_registered_reason_its_own_rationale(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
) -> None:
    """The positive control for the guard above: every registered reason builds
    a live-ineligible record carrying *its own* rationale.

    The five rationales are asserted pairwise distinct, so a mapping that
    collapsed them onto one shared body -- the exact "mismatched rationale" the
    guard exists to prevent -- fails here rather than passing on a coincidence.
    """
    question_hash = normalize_question(market)

    records: dict[str, ForecastRecord] = {
        reason: _build_abstention_record(
            market=market,
            baseline=baseline,
            created_at=created_at,
            question_hash=question_hash,
            citations=(),
            abstention_reason=reason,
            research_cost_micros=FULL_PIPELINE_RESEARCH_COST_MICROS,
        )
        for reason in _REGISTERED_ABSTENTION_REASONS
    }

    assert set(_ABSTENTION_RATIONALE_BY_REASON) == set(_REGISTERED_ABSTENTION_REASONS)
    rationales = [records[reason].rationale_markdown for reason in records]
    assert len(set(rationales)) == len(_REGISTERED_ABSTENTION_REASONS)
    for reason, record in records.items():
        assert record.rationale_markdown == _ABSTENTION_RATIONALE_BY_REASON[reason]
        assert record.abstention_reason == reason
        assert record.eligible_for_live is False


# --- The calibration-map ledger's `None` arm --------------------------------


def test_apply_and_record_calibration_without_a_ledger_returns_the_calibrated_ppm(
    created_at: datetime,
) -> None:
    """With `ledger=None` the calibration still runs: the returned ppm is the
    map's own applied value, byte-identical to the ledger-wired call, while the
    wired call additionally emits exactly one `CALIBRATION_MAP_APPLIED` event.

    The map is asserted to genuinely move `_CALIBRATION_INPUT_PPM`, so
    "ledgerless returns the calibrated value" cannot pass by the calibrated and
    uncalibrated values coinciding.
    """
    fitted = _fitted_map()
    expected_ppm = fitted.apply(_CALIBRATION_INPUT_PPM)
    ledger = InMemoryForecastLedger()

    ledgerless_ppm = _apply_and_record_calibration(
        _CALIBRATION_INPUT_PPM, fitted, None, created_at
    )
    ledgered_ppm = _apply_and_record_calibration(
        _CALIBRATION_INPUT_PPM, fitted, ledger, created_at
    )

    assert expected_ppm != _CALIBRATION_INPUT_PPM
    assert ledgerless_ppm == expected_ppm
    assert ledgerless_ppm == ledgered_ppm
    events = ledger.events_by_type(CALIBRATION_MAP_APPLIED_EVENT)
    assert len(events) == 1
    assert events[0].payload == {
        "map_id": _FITTED_MAP_ID,
        "map_version": _PIPELINE_SAFE_MAP_VERSION,
        "input_ppm": _CALIBRATION_INPUT_PPM,
        "output_ppm": expected_ppm,
    }


def test_run_pipeline_applies_the_calibration_map_with_no_ledger_wired(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """End-to-end: a wired map with `ledger=None` produces the *same record* as
    the ledger-wired run, and that record differs from an uncalibrated run --
    so the ledgerless arm is proven to still calibrate, not merely to not
    crash.
    """
    fitted = _fitted_map()
    ledger = InMemoryForecastLedger()

    ledgerless = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        calibration_map=fitted,
    )
    ledgered = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        calibration_map=fitted,
        ledger=ledger,
    )
    uncalibrated = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
    )

    assert ledgerless == ledgered
    assert ledgerless.probability_ppm != uncalibrated.probability_ppm
    assert len(ledger.events_by_type(CALIBRATION_MAP_APPLIED_EVENT)) == 1


# --- The provider-gate ledger's `None` arm ----------------------------------


def test_provider_gate_open_without_a_ledger_still_holds_an_unproven_provider(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
) -> None:
    """With `ledger=None` an unproven voting provider still holds the gate
    closed (`False`), identically to the ledger-wired call, which additionally
    emits exactly one `PROVIDER_GATE_HELD` event naming the provider.
    """
    votes = _canned_votes(market, baseline, make_fake_vote_transport)
    gate = _track_record_gate(openai_resolved=DEFAULT_MIN_RESOLVED - 1)
    ledger = InMemoryForecastLedger()

    ledgerless_open = _provider_gate_open(gate, votes, None, created_at)
    ledgered_open = _provider_gate_open(gate, votes, ledger, created_at)

    assert ledgerless_open is False
    assert ledgerless_open == ledgered_open
    events = ledger.events_by_type(PROVIDER_GATE_HELD_EVENT)
    assert len(events) == 1
    assert events[0].payload == {
        "unproven_providers": _PROVIDER_OPENAI,
        "unproven_count": 1,
        "min_resolved": DEFAULT_MIN_RESOLVED,
        "min_brier_skill_ppm": DEFAULT_MIN_BRIER_SKILL_PPM,
    }


def test_provider_gate_open_without_a_ledger_opens_for_proven_providers(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
) -> None:
    """The control for the hold above: with the *same* ledgerless call shape,
    a gate whose providers are all proven at the exact boundary returns `True`
    -- so the `False` above is a verdict on the track record, not a constant.
    """
    votes = _canned_votes(market, baseline, make_fake_vote_transport)
    proven_gate = _track_record_gate(openai_resolved=DEFAULT_MIN_RESOLVED)

    assert _provider_gate_open(proven_gate, votes, None, created_at) is True
    assert _provider_gate_open(None, votes, None, created_at) is True


def test_run_pipeline_holds_the_provider_gate_with_no_ledger_wired(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """End-to-end: an unproven voting provider with `ledger=None` still forces
    `eligible_for_live=False`, producing the same record as the ledger-wired
    run, while a proven gate over the identical ledgerless run is live-eligible
    -- the hold is a real verdict, not an unconditional `False`.
    """
    ledger = InMemoryForecastLedger()

    ledgerless_held = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        provider_gate=_track_record_gate(openai_resolved=DEFAULT_MIN_RESOLVED - 1),
    )
    ledgered_held = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        provider_gate=_track_record_gate(openai_resolved=DEFAULT_MIN_RESOLVED - 1),
        ledger=ledger,
    )
    ledgerless_open = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
        provider_gate=_track_record_gate(openai_resolved=DEFAULT_MIN_RESOLVED),
    )

    assert ledgerless_held.eligible_for_live is False
    assert ledgerless_held == ledgered_held
    assert ledgerless_open.eligible_for_live is True
    assert len(ledger.events_by_type(PROVIDER_GATE_HELD_EVENT)) == 1


# --- The vote loop's discard-recorder `None` arm ----------------------------


def test_collect_model_votes_discards_a_failing_vote_with_no_ledger_wired(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
) -> None:
    """With `ledger=None` a `ProviderVoteError` still discards exactly that one
    vote and lets the loop continue: the two survivors are returned, in call
    order, identically to the ledger-wired run (which also ledgers one
    `FORECAST_OUTPUT_DISCARDED` event).
    """
    ledger = InMemoryForecastLedger()

    ledgerless_votes = collect_model_votes(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        provider_factory=_routed_provider_factory(_two_survivors_one_discard_routes()),
    )
    ledgered_votes = collect_model_votes(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        provider_factory=_routed_provider_factory(_two_survivors_one_discard_routes()),
    )

    assert [vote.probability_ppm for vote in ledgerless_votes] == [
        _MEMBER_A_PROBABILITY_PPM,
        _MEMBER_C_PROBABILITY_PPM,
    ]
    assert ledgerless_votes == ledgered_votes
    assert len(ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)) == 1


def test_run_pipeline_charges_the_discarded_vote_cost_with_no_ledger_wired(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """With `ledger=None` the discarded vote's spend is still tallied into the
    budget seam: the run's charge is research + both survivors + the discard,
    proven by the repo's ceiling trick (a budget one micro short raises with
    `cost_micros` equal to the exact figure).

    The three vote costs are deliberately distinct (100 / 300 / 700_000), so
    dropping the discard leaf -- the shape a `recorder is None` arm that also
    skipped the accumulation would have -- lands on a different total rather
    than coinciding with the correct one.
    """
    budget = ResearchBudget(
        per_forecast_micros=_EXPECTED_CHARGE_MICROS - 1,
        ledger=InMemoryBudgetLedger(),
    )

    with pytest.raises(PerForecastBudgetExceededError) as excinfo:
        run_pipeline(
            market,
            baseline,
            transport=ForbiddenLiveTransport(),
            created_at=created_at,
            research_tools=research_tools,
            budget=budget,
            provider_factory=_routed_provider_factory(
                _two_survivors_one_discard_routes()
            ),
        )

    assert excinfo.value.cost_micros == _EXPECTED_CHARGE_MICROS
