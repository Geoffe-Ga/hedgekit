"""Tests for issue #241: consume `ParsedVote.abstain` in vote aggregation.

`ParsedVote.abstain` (`windbreak/forecast/sanitize.py`) has always been parsed
but dead-ended: `FixtureVoteProvider.forecast`
(`windbreak/forecast/providers/fixture.py`) ignores `parsed.abstain` when
building `ProviderForecast`, `ProviderForecast`
(`windbreak/forecast/providers/base.py`) carries no `abstain` field, and the
vote-aggregation path (`_collect_votes` / `_vote_shortfall_reason` /
`aggregate_votes`) never sees it. Net effect: a member returning
`{"abstain": true, "probability_ppm": 500000}` still contributes 500_000 to the
median instead of being excluded.

This module pins the chief architect's fix contract end-to-end:

* `ProviderForecast` gains a final, defaulted `abstain: bool = False` field.
* `collect_model_votes` excludes an abstaining member's vote from its returned
  tuple.
* `_vote_shortfall_reason` gains a 4th `abstain_count` parameter with
  first-match-wins precedence: quorum-met (with abstainers present) beats
  everything and returns `None`; any abstention beats the pre-existing
  transport/screen-discard classification and returns the new
  `ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED`; only when `abstain_count == 0` does
  the pre-#241 `PROVIDER_UNAVAILABLE` / `ALL_VOTES_DISCARDED` /
  `ENSEMBLE_QUORUM_NOT_MET` taxonomy apply, unchanged.
* An abstaining member's cost is still charged (it *did* call), but its
  provider-reported citations never enter the record's audit trail and it
  never contributes a vote to the median.

`windbreak.forecast.pipeline.ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED` does not
exist yet, so importing it below fails collection with
`ImportError: cannot import name 'ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED'` --
the expected Gate 1 RED state for issue #241. Once that constant (and the
behavior it names) lands, the individual tests below pin the exact
`abstain_count`-precedence rules, the exact excluded-vote/excluded-citation
contract, and the exact rationale-registry entry, so a partially-correct
implementation fails on a targeted `AssertionError` rather than a vague
collection error.

Local-doubles choice
    This module defines its own minimal `_SucceedingProvider` /
    `_routed_provider_factory` / `_MEMBER_A`/`_MEMBER_B`/`_MEMBER_C` doubles
    rather than importing them from `tests/forecast/test_provider_failures.py`
    -- cross-test-module imports are not supported under this project's
    rootdir-relative pytest import mode (see `tests/forecast/conftest.py`'s
    and `test_triage.py`'s docstrings for the same convention applied
    elsewhere).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.budget import (
    InMemoryBudgetLedger,
    PerForecastBudgetExceededError,
    ResearchBudget,
)
from windbreak.forecast.cassettes import ForbiddenLiveTransport
from windbreak.forecast.pipeline import (
    _ABSTENTION_RATIONALE_BY_REASON,
    ABSTENTION_ALL_VOTES_DISCARDED,
    ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED,
    ABSTENTION_ENSEMBLE_QUORUM_NOT_MET,
    ABSTENTION_PROVIDER_UNAVAILABLE,
    _vote_shortfall_reason,
    run_pipeline,
)
from windbreak.forecast.providers import (
    DEFAULT_VOTE_ENSEMBLE,
    ProviderCitation,
    ProviderForecast,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.providers import EnsembleMemberLike, ForecastProvider
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

#: `windbreak.forecast.pipeline`'s private `_RESEARCH_COST_MICROS` stub cost for
#: a full run -- named locally (it is private), mirroring
#: `tests/forecast/test_provider_failures.py`'s identical convention, so every
#: exact-charge assertion below reads against the same known figure.
_FULL_RUN_RESEARCH_COST_MICROS = 3_000_000

#: The three pinned default ensemble members (SPEC S6.3), indexed rather than
#: unpacked from the variable-length `DEFAULT_VOTE_ENSEMBLE` tuple (mirrors
#: `test_provider_failures.py`'s identical convention).
_MEMBER_A = DEFAULT_VOTE_ENSEMBLE[0]
_MEMBER_B = DEFAULT_VOTE_ENSEMBLE[1]
_MEMBER_C = DEFAULT_VOTE_ENSEMBLE[2]


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


def _routed_provider_factory(
    routes: dict[str, ForecastProvider],
) -> Callable[[EnsembleMemberLike], ForecastProvider]:
    """Build a `provider_factory` routing each member by its `model_version`.

    Args:
        routes: A `{model_version: ForecastProvider}` mapping covering every
            member the pipeline run will drive.

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
    member: EnsembleMemberLike,
    *,
    probability_ppm: int,
    abstain: bool,
    cost_micros: int = 0,
    citations: tuple[ProviderCitation, ...] = (),
) -> ForecastProvider:
    """Build a `_SucceedingProvider` returning a clean, member-stamped forecast.

    Args:
        member: The ensemble member this provider's forecast is stamped with.
        probability_ppm: The forecast's probability estimate, in ppm.
        abstain: Whether the returned forecast abstains.
        cost_micros: The forecast's billed cost, in micros.
        citations: The forecast's provider-reported citations.

    Returns:
        A `_SucceedingProvider` returning a valid, member-stamped forecast.
    """
    fingerprint_source = f"{member.model_version}-abstain={abstain}"
    return _SucceedingProvider(
        ProviderForecast(
            probability_ppm=probability_ppm,
            rationale_summary="steady evidence",
            citations=citations,
            cost_micros=cost_micros,
            provider=member.provider,
            model_version=member.model_version,
            training_cutoff=member.training_cutoff,
            response_fingerprint=hashlib.sha256(
                fingerprint_source.encode("utf-8")
            ).hexdigest(),
            abstain=abstain,
        )
    )


# --- run_pipeline: an abstainer is excluded from the median, fingerprints, --------
# --- and citations, but its cost is still charged (contract 3 and 5) -------------


def test_abstaining_member_excluded_from_median_fingerprint_and_citations(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """One of three members abstains; the record's point estimate reflects
    the surviving two voters' own median, never the 3-vote median that would
    include the abstainer, and the abstainer's fingerprint and
    provider-reported citation are both absent from the record.

    Member A votes 200_000 ppm and member C votes 900_000 ppm: their 2-voter
    median is 550_000 ppm, shrinking to a final point estimate of 525_000
    ppm. A 3-vote median including the abstainer's 500_000 ppm would instead
    be 500_000 ppm, shrinking to 487_500 ppm -- the two outcomes are
    numerically distinct, so this test fails today (the abstainer still
    counts) and will pass once abstention is honored.
    """
    abstain_citation = ProviderCitation(
        url="https://research.local/abstainer-source",
        publication_date=None,
        quoted_text="an abstaining member's reported source",
    )
    abstain_fingerprint = hashlib.sha256(
        f"{_MEMBER_B.model_version}-abstain=True".encode()
    ).hexdigest()
    abstain_forecast = ProviderForecast(
        probability_ppm=500_000,
        rationale_summary="steady evidence",
        citations=(abstain_citation,),
        cost_micros=250_000,
        provider=_MEMBER_B.provider,
        model_version=_MEMBER_B.model_version,
        training_cutoff=_MEMBER_B.training_cutoff,
        response_fingerprint=abstain_fingerprint,
        abstain=True,
    )
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _member_provider(
            _MEMBER_A, probability_ppm=200_000, abstain=False
        ),
        _MEMBER_B.model_version: _SucceedingProvider(abstain_forecast),
        _MEMBER_C.model_version: _member_provider(
            _MEMBER_C, probability_ppm=900_000, abstain=False
        ),
    }
    generous_budget = ResearchBudget(
        per_forecast_micros=10_000_000, ledger=InMemoryBudgetLedger()
    )

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        budget=generous_budget,
        provider_factory=_routed_provider_factory(routes),
    )

    assert record.abstention_reason is None
    assert record.eligible_for_live is True
    assert len(record.model_votes) == 2
    assert abstain_fingerprint not in {
        vote.response_fingerprint for vote in record.model_votes
    }
    assert record.probability_ppm == 525_000
    assert record.ci_low_ppm == 200_000
    assert record.ci_high_ppm == 900_000
    assert abstain_citation.url not in {citation.url for citation in record.citations}


def test_abstaining_member_cost_still_charged_via_ceiling_trick(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """The abstaining member's own `cost_micros` (250_000) is charged into
    the research budget even though its vote never survives to aggregation:
    it *called* the provider, so it is billed like any other vote -- proven
    via the repo's established ceiling-trick pattern (see
    `tests/forecast/test_provider_failures.py`): a budget one micro below the
    expected combined charge raises with `cost_micros` equal to the exact
    figure.
    """
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _member_provider(
            _MEMBER_A, probability_ppm=200_000, abstain=False
        ),
        _MEMBER_B.model_version: _member_provider(
            _MEMBER_B, probability_ppm=500_000, abstain=True, cost_micros=250_000
        ),
        _MEMBER_C.model_version: _member_provider(
            _MEMBER_C, probability_ppm=900_000, abstain=False
        ),
    }
    exact_charge = _FULL_RUN_RESEARCH_COST_MICROS + 250_000
    tight_budget = ResearchBudget(
        per_forecast_micros=exact_charge - 1, ledger=InMemoryBudgetLedger()
    )

    with pytest.raises(PerForecastBudgetExceededError) as excinfo:
        run_pipeline(
            market,
            baseline,
            transport=ForbiddenLiveTransport(),
            created_at=created_at,
            research_tools=research_tools,
            budget=tight_budget,
            provider_factory=_routed_provider_factory(routes),
        )

    assert excinfo.value.cost_micros == exact_charge


# --- _vote_shortfall_reason: abstain_count precedence (unit-level, contract 3) ---
#
# The last three parametrized cases (`regression_*` ids) are critical
# regression pins: they guard against the new abstain-aware rule shadowing
# the pre-#241 taxonomy when no member actually abstained
# (`abstain_count == 0`).


@pytest.mark.parametrize(
    (
        "vote_count",
        "min_ensemble_votes",
        "discard_transport_flags",
        "abstain_count",
        "expected_reason",
    ),
    [
        pytest.param(2, 2, (), 1, None, id="quorum_met_with_abstainer_present"),
        pytest.param(
            1,
            2,
            (),
            1,
            ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED,
            id="below_quorum_with_one_abstainer",
        ),
        pytest.param(
            0,
            2,
            (),
            3,
            ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED,
            id="zero_survivors_all_abstained",
        ),
        pytest.param(
            0,
            2,
            (True,),
            1,
            ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED,
            id="zero_survivors_mixed_abstain_and_transport_discard",
        ),
        pytest.param(
            0,
            2,
            (True, True),
            0,
            ABSTENTION_PROVIDER_UNAVAILABLE,
            id="regression_zero_abstain_all_transport_discards",
        ),
        pytest.param(
            0,
            2,
            (True, False),
            0,
            ABSTENTION_ALL_VOTES_DISCARDED,
            id="regression_zero_abstain_mixed_screen_discard",
        ),
        pytest.param(
            1,
            2,
            (),
            0,
            ABSTENTION_ENSEMBLE_QUORUM_NOT_MET,
            id="regression_zero_abstain_one_survivor_below_quorum",
        ),
    ],
)
def test_vote_shortfall_reason_abstain_count_precedence(
    vote_count: int,
    min_ensemble_votes: int,
    discard_transport_flags: tuple[bool, ...],
    abstain_count: int,
    expected_reason: str | None,
) -> None:
    """`_vote_shortfall_reason`'s abstain-aware precedence (issue #241).

    Quorum met (even with an abstainer present) returns `None`; any
    nonzero `abstain_count` below quorum returns the new
    `ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED` -- beating even a transport
    wipeout in the mix (rule 2 beats rule 3); and `abstain_count == 0`
    leaves every pre-#241 reason (`PROVIDER_UNAVAILABLE`,
    `ALL_VOTES_DISCARDED`, `ENSEMBLE_QUORUM_NOT_MET`) exactly as it was.
    """
    result = _vote_shortfall_reason(
        vote_count,
        min_ensemble_votes,
        discard_transport_flags,
        abstain_count=abstain_count,
    )

    assert result == expected_reason


# --- End-to-end abstention record for the new reason (contract 3, item D) --------


def test_run_pipeline_all_members_abstain_yields_members_abstained_record(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """All three ensemble members cleanly abstain (no failures, no discards):
    the run abstains with the new `ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED`
    reason, a baseline-collapsed probability, `model_votes == ()`, permanent
    live-ineligibility, and a rationale that is both the exact registered
    prose for this reason and distinct from every other abstention reason's
    prose -- no borrowed text.
    """
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _member_provider(
            _MEMBER_A, probability_ppm=500_000, abstain=True
        ),
        _MEMBER_B.model_version: _member_provider(
            _MEMBER_B, probability_ppm=500_000, abstain=True
        ),
        _MEMBER_C.model_version: _member_provider(
            _MEMBER_C, probability_ppm=500_000, abstain=True
        ),
    }

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        provider_factory=_routed_provider_factory(routes),
    )

    baseline_ppm = baseline.price_pips * 100
    assert record.abstention_reason == ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED
    assert record.eligible_for_live is False
    assert record.model_votes == ()
    assert record.probability_ppm == baseline_ppm
    assert record.ci_low_ppm == baseline_ppm
    assert record.ci_high_ppm == baseline_ppm
    expected_rationale = _ABSTENTION_RATIONALE_BY_REASON[
        ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED
    ]
    all_votes_discarded_rationale = _ABSTENTION_RATIONALE_BY_REASON[
        ABSTENTION_ALL_VOTES_DISCARDED
    ]
    quorum_not_met_rationale = _ABSTENTION_RATIONALE_BY_REASON[
        ABSTENTION_ENSEMBLE_QUORUM_NOT_MET
    ]
    provider_unavailable_rationale = _ABSTENTION_RATIONALE_BY_REASON[
        ABSTENTION_PROVIDER_UNAVAILABLE
    ]
    assert record.rationale_markdown == expected_rationale
    assert record.rationale_markdown != all_votes_discarded_rationale
    assert record.rationale_markdown != quorum_not_met_rationale
    assert record.rationale_markdown != provider_unavailable_rationale


# --- Rationale registry: the new reason has its own distinct entry (contract E) --


def test_abstention_reason_string_value_is_pinned() -> None:
    """`ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED`'s wire-format string is exactly
    `"ensemble_members_abstained"`, so a future rename cannot silently drift
    the machine-readable reason stamped on abstained records.
    """
    assert ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED == "ensemble_members_abstained"


def test_abstention_rationale_registry_has_members_abstained_entry() -> None:
    """`ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED` is registered in
    `_ABSTENTION_RATIONALE_BY_REASON` and maps to its own distinct prose --
    never borrowing another reason's rationale text, so the audit trail's
    prose always matches why the engine actually abstained.
    """
    assert ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED in _ABSTENTION_RATIONALE_BY_REASON
    rationale = _ABSTENTION_RATIONALE_BY_REASON[ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED]
    other_rationales = [
        text
        for reason, text in _ABSTENTION_RATIONALE_BY_REASON.items()
        if reason != ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED
    ]
    assert rationale not in other_rationales
