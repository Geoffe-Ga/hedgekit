"""SPEC S8.2 forecast pipeline wiring (offline, deterministic stub stages).

This module implements the twelve pipeline stages of SPEC S8.2 as discrete,
typed functions and threads them together in :func:`run_pipeline`. The spec's
S8.2 diagram (``plans/SPEC_v3.md:292-301``) reads as an eleven-arrow chain but
carries *twelve* arrow segments once the terminal
"schema-validated ForecastRecord" step is counted; the twelve stage functions
below map one-to-one onto those segments.

The stage bodies are deterministic, network-free stubs (identity and
fixture-derived logic) with real control flow -- the milestone is a correct,
byte-deterministic wiring skeleton, not modeled forecasting. Determinism is
load-bearing: ``created_at`` is injected (never ``datetime.now()``), all math
is integer-only, and :func:`collect_model_votes` is the sole stage that
touches the :class:`~windbreak.forecast.cassettes.LlmTransport` seam, so wiring
a :class:`~windbreak.forecast.cassettes.ForbiddenLiveTransport` (or an empty
replay cassette) fails the run closed rather than silently succeeding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol

from windbreak.forecast.calibration import ensure_temporal_integrity
from windbreak.forecast.citations import (
    content_hash_of,
    count_verified,
    verify_citations,
)
from windbreak.forecast.ensemble import aggregate_votes
from windbreak.forecast.providers import (
    DEFAULT_VOTE_ENSEMBLE,
    FixtureVoteProvider,
)
from windbreak.forecast.providers.base import (
    ProviderHTTPError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderVoteError,
)
from windbreak.forecast.providers.retry import is_retryable_status
from windbreak.forecast.pubdate import extract_publication_date
from windbreak.forecast.records import (
    Citation,
    ForecastRecord,
    ModelVote,
    is_live_eligible,
)
from windbreak.forecast.sanitize import (
    MAX_QUOTE_WORDS,
    ResearchQuote,
    extract_quote,
    sanitize_content,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.budget import ResearchBudget
    from windbreak.forecast.calibration import CalibrationMap
    from windbreak.forecast.canary import CanaryGate
    from windbreak.forecast.cassettes import LlmTransport
    from windbreak.forecast.citations import CitationVerdict
    from windbreak.forecast.ensemble import VoteAggregate
    from windbreak.forecast.providers import (
        EnsembleMemberLike,
        ForecastProvider,
        ProviderCitation,
        ProviderForecast,
    )
    from windbreak.forecast.providers.track_record import ProviderTrackRecordGate
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

#: One full probability (1.0) expressed in parts-per-million; also the clamp
#: ceiling and the shrinkage denominator.
_PPM_SCALE = 1_000_000

#: Lowest legal ppm probability (the clamp floor).
_MIN_PPM = 0

#: Exact pips-to-ppm factor: pips are 1e-4 and ppm 1e-6, so a binary market
#: price in pips maps to a probability in ppm by multiplying by 100.
_PIPS_TO_PPM = 100

#: Fixed shrinkage weight toward the market baseline (SPEC S16
#: ``shrink_to_market_lambda_ppm`` default), applied as integer math.
_SHRINK_LAMBDA_PPM = 250_000

#: Deterministic subquestion prefixes for the decomposition stage.
_SUBQUESTION_PREFIXES: tuple[str, ...] = (
    "Base rate for",
    "Recent evidence on",
    "Counter-signal for",
)

#: Deterministic stub research cost for a full-pipeline run, in micros.
_RESEARCH_COST_MICROS = 3_000_000

#: Hours per day and seconds per hour, for the integer horizon computation.
_HOURS_PER_DAY = 24
_SECONDS_PER_HOUR = 3_600

#: Default minimum independently-verified citations a full record needs to be
#: live-eligible (SPEC S16 ``ForecastConfig.min_verified_citations`` default).
DEFAULT_MIN_VERIFIED_CITATIONS: Final = 3

#: Abstention reason stamped when a full run gathers zero verified citations.
ABSTENTION_NO_VERIFIED_CITATIONS: Final = "no_verified_citations"

#: Abstention reason stamped when every ensemble vote is discarded by the
#: response-side injection screen (SPEC S8.5), leaving nothing to aggregate.
ABSTENTION_ALL_VOTES_DISCARDED: Final = "all_votes_discarded"

#: Abstention reason stamped when at least one vote survived but the surviving
#: count falls below the required ensemble quorum (``min_ensemble_votes``), so
#: aggregating over too few members would be less trustworthy than abstaining.
ABSTENTION_ENSEMBLE_QUORUM_NOT_MET: Final = "ensemble_quorum_not_met"

#: Abstention reason stamped when zero votes survived *and* every discard was a
#: transport-class fault (timeout / rate-limit / retryable HTTP status): no
#: provider could be reached at all, distinct from providers that responded but
#: were screen-rejected (:data:`ABSTENTION_ALL_VOTES_DISCARDED`).
ABSTENTION_PROVIDER_UNAVAILABLE: Final = "provider_unavailable"

#: Abstention reason stamped when one or more ensemble members explicitly
#: abstained from voting (SPEC S6.3), leaving fewer voting members than the
#: required ensemble quorum. Distinct from a screen discard or a transport
#: fault: the member was reached and answered, but declined to cast a usable
#: vote, so it is charged yet excluded from the median (issue #241).
ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED: Final = "ensemble_members_abstained"

#: The default minimum surviving ensemble votes a full record needs; below this
#: the run abstains with :data:`ABSTENTION_ENSEMBLE_QUORUM_NOT_MET`.
DEFAULT_MIN_ENSEMBLE_VOTES: Final = 2

#: Event type ledgered when one model vote is discarded as injection-tainted.
FORECAST_OUTPUT_DISCARDED_EVENT: Final = "FORECAST_OUTPUT_DISCARDED"

#: Event type buffered per ensemble member for the per-provider vote-cost audit
#: signal (issue #281): the composition root folds these into ledgered
#: ``ProviderVoteRecorded`` events after the tick's ``ForecastCreated``.
PROVIDER_VOTE_COSTED_EVENT: Final = "PROVIDER_VOTE_COSTED"

#: ``outcome`` on a surviving, non-abstaining vote's ``PROVIDER_VOTE_COSTED``.
VOTE_OUTCOME_VOTED: Final = "voted"

#: ``outcome`` on a surviving vote that explicitly abstained (SPEC S6.3): it
#: answered, but chose not to vote, so it never reaches the median.
VOTE_OUTCOME_ABSTAINED: Final = "abstained"

#: ``outcome`` on a discarded vote (the provider raised ``ProviderVoteError``):
#: its cost was still charged, but its vote never reached aggregation.
VOTE_OUTCOME_DISCARDED: Final = "discarded"

#: Event type ledgered when a wired calibration map is applied to the aggregate
#: median, recording the exact pre-/post-calibration ppm (SPEC S8.2 stage 11).
CALIBRATION_MAP_APPLIED_EVENT: Final = "CALIBRATION_MAP_APPLIED"

#: Event type ledgered when the provider-track-record gate holds a run back from
#: live eligibility because one or more voting providers are unproven (#194).
PROVIDER_GATE_HELD_EVENT: Final = "PROVIDER_GATE_HELD"

#: ``source_type`` stamped on a citation a provider *reported* (never one this
#: pipeline independently verified): it carries provider-claimed provenance, so
#: it is retained for audit but deliberately excluded from the verified-citation
#: live-eligibility count (SPEC S8.8 stays anchored to verified citations).
PROVIDER_REPORTED_SOURCE_TYPE: Final = "provider_reported"

#: Sha256 content-hash prefix, matching ``citations.content_hash_of``'s scheme.
_SHA256_PREFIX: Final = "sha256:"

#: Deterministic rationale stamped on a zero-verified-citation abstention
#: record (mirrors ``triage._TRIAGE_RATIONALE_MD``).
_ABSTENTION_RATIONALE_NO_VERIFIED_CITATIONS_MD: Final = (
    "## Abstained forecast\n\n"
    "The full pipeline ran, but no gathered citation could be independently "
    "verified, so the engine abstained. This record is live-ineligible.\n"
)

#: Deterministic rationale stamped when citations *were* verified but every
#: ensemble vote was rejected or discarded for a non-transport reason. This is a
#: catch-all bucket: the response-side injection screen (SPEC S8.5),
#: forecaster-version drift, a cost overrun, a malformed response, or a
#: non-retryable provider HTTP status (e.g. 4xx) can all land here, so the prose
#: must name the whole set rather than claim the injection screen alone. It must
#: never borrow the no-verified-citations rationale: that would misreport why the
#: engine abstained in the exact scenario S8.5 adds.
_ABSTENTION_RATIONALE_ALL_VOTES_DISCARDED_MD: Final = (
    "## Abstained forecast\n\n"
    "The full pipeline ran and citations were independently verified, but "
    "every ensemble vote was rejected or discarded (response-side injection "
    "screen, forecaster-version drift, cost overrun, or a non-retryable provider "
    "error), leaving nothing to aggregate, so the engine abstained. This record "
    "is live-ineligible.\n"
)

#: Deterministic rationale stamped when at least one vote survived but too few
#: did to meet the required ensemble quorum. Truthful about the shortfall: some
#: members survived, but fewer than the ensemble quorum required, so the engine
#: abstained rather than trust an under-quorum aggregation.
_ABSTENTION_RATIONALE_ENSEMBLE_QUORUM_NOT_MET_MD: Final = (
    "## Abstained forecast\n\n"
    "The full pipeline ran and citations were independently verified, but "
    "fewer ensemble members survived the response-side screen than the required "
    "ensemble quorum, so the engine abstained rather than aggregate over too "
    "few votes. This record is live-ineligible.\n"
)

#: Deterministic rationale stamped when zero votes survived and every discard
#: was a transport-class fault: no provider could be reached at all (distinct
#: from providers that responded but were screen-rejected).
_ABSTENTION_RATIONALE_PROVIDER_UNAVAILABLE_MD: Final = (
    "## Abstained forecast\n\n"
    "The full pipeline ran and citations were independently verified, but every "
    "ensemble member failed with a transport-class fault (timeout, rate-limit, "
    "or a retryable HTTP status), so no provider could be reached and the engine "
    "abstained. This record is live-ineligible.\n"
)

#: Deterministic rationale stamped when one or more ensemble members explicitly
#: abstained from voting, leaving too few voting members to meet the ensemble
#: quorum. Truthful in every shortfall case that includes at least one
#: abstention (mixed with discards or not): the pipeline ran and citations were
#: verified, but abstaining members declined to cast usable votes, so the engine
#: abstained rather than aggregate over too few votes.
_ABSTENTION_RATIONALE_ENSEMBLE_MEMBERS_ABSTAINED_MD: Final = (
    "## Abstained forecast\n\n"
    "The full pipeline ran and citations were independently verified, but one "
    "or more ensemble members explicitly abstained from voting, leaving fewer "
    "voting members than the required ensemble quorum, so the engine abstained "
    "rather than aggregate over too few votes. This record is live-ineligible.\n"
)

#: Maps each abstention reason to its human-readable rationale, so the audit
#: trail's prose always matches the machine-readable ``abstention_reason``.
_ABSTENTION_RATIONALE_BY_REASON: Final[Mapping[str, str]] = {
    ABSTENTION_NO_VERIFIED_CITATIONS: _ABSTENTION_RATIONALE_NO_VERIFIED_CITATIONS_MD,
    ABSTENTION_ALL_VOTES_DISCARDED: _ABSTENTION_RATIONALE_ALL_VOTES_DISCARDED_MD,
    ABSTENTION_ENSEMBLE_QUORUM_NOT_MET: (
        _ABSTENTION_RATIONALE_ENSEMBLE_QUORUM_NOT_MET_MD
    ),
    ABSTENTION_PROVIDER_UNAVAILABLE: _ABSTENTION_RATIONALE_PROVIDER_UNAVAILABLE_MD,
    ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED: (
        _ABSTENTION_RATIONALE_ENSEMBLE_MEMBERS_ABSTAINED_MD
    ),
}


@dataclass(frozen=True, slots=True)
class ForecastEvent:
    """One recorded forecast-engine decision (mirrors ``TriageEvent``).

    Attributes:
        event_type: The event kind (e.g. ``FORECAST_OUTPUT_DISCARDED``).
        payload: The JSON-safe event body (int/str/bool leaves only -- never a
            float and never the raw model response text).
        ts: ISO-8601 UTC timestamp of when the event was created.
    """

    event_type: str
    payload: Mapping[str, object]
    ts: str


class ForecastLedgerWriter(Protocol):
    """The seam through which a forecast-engine decision is persisted."""

    def record(self, event: ForecastEvent) -> None:
        """Persist a forecast event.

        Args:
            event: The event to persist.
        """
        ...


class InMemoryForecastLedger:
    """A :class:`ForecastLedgerWriter` retaining events in memory for tests."""

    def __init__(self) -> None:
        """Initialize with an empty event log."""
        self._events: list[ForecastEvent] = []

    def record(self, event: ForecastEvent) -> None:
        """Append a forecast event to the in-memory log.

        Args:
            event: The event to retain.
        """
        self._events.append(event)

    def events_by_type(self, event_type: str) -> tuple[ForecastEvent, ...]:
        """Return every retained event of a given type, in record order.

        Args:
            event_type: The event kind to filter by.

        Returns:
            The matching events.
        """
        return tuple(event for event in self._events if event.event_type == event_type)


@dataclass(frozen=True, slots=True)
class _DiscardRecorder:
    """Binds a ledger writer to a fixed timestamp for discard events.

    Built once per :func:`collect_model_votes` call (only when a ledger is
    wired), so the loop that screens votes never has to re-thread an optional
    ``created_at`` -- the timestamp is already resolved and non-optional here.

    Attributes:
        ledger: The forecast-event ledger writer to persist through.
        ts: The pre-rendered ISO-8601 UTC timestamp stamped on every event.
    """

    ledger: ForecastLedgerWriter
    ts: str

    def record_discard(
        self,
        *,
        market_ticker: str,
        member: EnsembleMemberLike,
        vote_index: int,
        failure: str,
        response_fingerprint: str,
    ) -> None:
        """Ledger one discarded vote with a fingerprint-only payload.

        The raw response never enters the payload -- only its sha256 fingerprint
        (computed by the provider and passed in here) does -- so a tainted
        response cannot leak through the audit trail.

        Args:
            market_ticker: The forecast market's ticker.
            member: The ensemble member whose vote was discarded.
            vote_index: The zero-based index of the discarded vote.
            failure: The ``RESPONSE_FAILURE_*`` code the screen returned.
            response_fingerprint: The rejected response's sha256 fingerprint.
        """
        payload: dict[str, object] = {
            "market_ticker": market_ticker,
            "provider": member.provider,
            "model_version": member.model_version,
            "vote_index": vote_index,
            "failure": failure,
            "response_fingerprint": response_fingerprint,
        }
        self.ledger.record(
            ForecastEvent(FORECAST_OUTPUT_DISCARDED_EVENT, payload, self.ts)
        )

    def record_vote_cost(
        self,
        *,
        market_ticker: str,
        member: EnsembleMemberLike,
        vote_index: int,
        cost_micros: int,
        outcome: str,
        failure_code: str,
    ) -> None:
        """Buffer one per-vote cost event for the vote-cost audit signal (#281).

        Emitted once per ensemble member driven -- one ``"voted"``,
        ``"abstained"``, or ``"discarded"`` outcome each -- so the composition
        root can fold each provider's charged spend, whatever its vote outcome.
        Unlike :meth:`record_discard`, this never carries a response
        fingerprint: it is a cost/outcome signal, not a tainted-response audit.

        Args:
            market_ticker: The forecast market's ticker.
            member: The ensemble member whose vote this cost belongs to.
            vote_index: The zero-based index of the vote in the driven ensemble.
            cost_micros: The vote's billed cost, in micros (charged even when
                the vote was discarded).
            outcome: The vote outcome (:data:`VOTE_OUTCOME_VOTED` /
                :data:`VOTE_OUTCOME_ABSTAINED` / :data:`VOTE_OUTCOME_DISCARDED`).
            failure_code: The discard failure code, or ``""`` for a non-discard.
        """
        payload: dict[str, object] = {
            "market_ticker": market_ticker,
            "provider": member.provider,
            "model_version": member.model_version,
            "vote_index": vote_index,
            "cost_micros": cost_micros,
            "outcome": outcome,
            "failure_code": failure_code,
        }
        self.ledger.record(ForecastEvent(PROVIDER_VOTE_COSTED_EVENT, payload, self.ts))


def _clamp_ppm(value: int) -> int:
    """Clamp an integer into the legal ppm domain ``[0, 1_000_000]``.

    Args:
        value: The candidate ppm value.

    Returns:
        ``value`` clamped into ``[0, 1_000_000]``.
    """
    return max(_MIN_PPM, min(value, _PPM_SCALE))


def _clamp_between(value: int, low: int, high: int) -> int:
    """Clamp ``value`` into the inclusive ``[low, high]`` interval.

    Args:
        value: The candidate value.
        low: The interval floor.
        high: The interval ceiling.

    Returns:
        ``value`` clamped into ``[low, high]``.
    """
    return max(low, min(value, high))


def _iso_z(moment: datetime) -> str:
    """Render a datetime as ISO-8601 UTC with a trailing ``Z``.

    Args:
        moment: The (timezone-aware) datetime to render; normalized to UTC.

    Returns:
        A string like ``2024-12-10T12:00:00.000000Z``.
    """
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _baseline_probability_ppm(baseline: BaselineQuoteSnapshot) -> int:
    """Convert a baseline pip price into a clamped ppm probability.

    Args:
        baseline: The baseline quote snapshot.

    Returns:
        The baseline price as a probability in ppm.
    """
    return _clamp_ppm(baseline.price_pips * _PIPS_TO_PPM)


# --- SPEC S8.2 stages (twelve discrete steps) ------------------------------------


def normalize_question(market: NormalizedMarket) -> str:
    """Stage 1: normalize the question into a stable content hash.

    Args:
        market: The market under forecast.

    Returns:
        A sha256 hex hash of the title plus resolution criteria.
    """
    digest_source = market.title + market.resolution_criteria
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()


def extract_resolution_criteria(market: NormalizedMarket) -> str:
    """Stage 2: extract the market's resolution criteria (identity stub).

    Args:
        market: The market under forecast.

    Returns:
        The market's resolution-criteria prose.
    """
    return market.resolution_criteria


def outside_view_base_rate(baseline: BaselineQuoteSnapshot) -> int:
    """Stage 3: take the outside-view base rate from the baseline price.

    Args:
        baseline: The baseline quote snapshot.

    Returns:
        The baseline price as a ppm probability.
    """
    return _baseline_probability_ppm(baseline)


def decompose_subquestions(market: NormalizedMarket) -> tuple[str, ...]:
    """Stage 4: decompose the question into deterministic subquestions.

    Args:
        market: The market under forecast.

    Returns:
        One subquestion per configured prefix.
    """
    return tuple(f"{prefix} {market.title}" for prefix in _SUBQUESTION_PREFIXES)


def _page_budget_reached(max_pages: int | None, pages_fetched: int) -> bool:
    """Return whether the per-forecast page budget has been reached.

    Args:
        max_pages: The fetch-attempt ceiling, or ``None`` for unbounded.
        pages_fetched: How many fetch attempts have already been made.

    Returns:
        ``True`` when ``max_pages`` is set and already met; ``False`` otherwise
        (including the unbounded ``None`` case).
    """
    return max_pages is not None and pages_fetched >= max_pages


def bounded_web_research(
    subquestions: tuple[str, ...],
    *,
    tools: ResearchTools,
    max_pages: int | None = None,
) -> tuple[Citation, ...]:
    """Stage 5: gather citations through the sandboxed research tools.

    Each subquestion is searched for a candidate URL, whose content is fetched
    through the egress-allowlisted :meth:`ResearchTools.fetch` -- so a search
    result off the allowlist raises :class:`EgressDeniedError` here, on the pipeline
    path itself (a policy violation fails the run closed, never silently). An
    *unreachable* source (the transport raising an ``OSError`` such as
    ``ConnectionError``) is not a policy violation but a dead link, so it is
    skipped: it simply contributes no citation, letting the downstream
    verification stage abstain on an evidence-free run. A dead-link fetch still
    *counts* as one page against ``max_pages`` -- the attempt was made -- so a
    run cannot evade its page budget by hitting unreachable URLs. Each citation's
    ``content_hash`` stays over the *raw* fetched content, while its
    ``quoted_text`` is the sanitized excerpt (SPEC S8.5):
    :func:`windbreak.forecast.sanitize.sanitize_content` strips scripts, hidden
    payloads, and forged delimiters, and
    :func:`windbreak.forecast.sanitize.extract_quote` caps it at
    :data:`~windbreak.forecast.sanitize.MAX_QUOTE_WORDS` words. Because
    :func:`windbreak.forecast.citations.verify_citation` rehashes the raw
    refetch and re-checks the sanitized quote as a raw substring, a page whose
    injection sits *after* the quote window still self-verifies (the quote is a
    contiguous raw substring), whereas a hidden span, ``<script>``, or forged
    delimiter *inside* the window breaks that substring property -- so the
    citation fails to verify and the run fails closed (abstains) rather than
    prompting an LLM with poisoned text. Each citation's ``publication_date`` is
    :func:`windbreak.forecast.pubdate.extract_publication_date` run over the
    *raw* fetched content (before sanitization strips its ``<script>`` blocks),
    so a page carrying a real JSON-LD/meta date is stamped with that
    timezone-aware date and a dateless page degrades to ``None`` -- never a
    fabricated constant. The resulting citations are deterministic (the fixture
    transports derive URL and content from their inputs) and integer-free.

    Args:
        subquestions: The decomposed subquestions to research.
        tools: The sandboxed research tools (search/fetch capabilities).
        max_pages: The maximum number of fetch attempts this run may make;
            ``None`` (the default) is unbounded, preserving the pre-budget
            behavior byte-for-byte. Each ``tools.fetch`` call -- including one
            that raises ``OSError`` -- counts as one page.

    Returns:
        One deterministic citation per reachable subquestion candidate URL, up
        to the ``max_pages`` fetch-attempt ceiling.

    Raises:
        ValueError: If ``max_pages`` is negative.
    """
    if max_pages is not None and max_pages < 0:
        msg = f"max_pages must be non-negative or None, got {max_pages}"
        raise ValueError(msg)
    citations: list[Citation] = []
    pages_fetched = 0
    for subquestion in subquestions:
        if _page_budget_reached(max_pages, pages_fetched):
            break
        urls = tools.search(subquestion)
        if not urls:
            continue
        url = urls[0]
        pages_fetched += 1
        try:
            content = tools.fetch(url)
        except OSError:
            continue
        citations.append(
            Citation(
                url=url,
                content_hash=content_hash_of(content),
                quoted_text=extract_quote(
                    sanitize_content(content), max_words=MAX_QUOTE_WORDS
                ),
                publication_date=extract_publication_date(content),
                source_type="research_note",
            )
        )
    return tuple(citations)


def _source_note(verdict: CitationVerdict) -> str:
    """Render one truthful source-quality note from a citation verdict.

    Args:
        verdict: The citation's verification verdict.

    Returns:
        A note naming the source type and URL, prefixed ``verified`` when the
        citation verified or ``unverified (<failure>)`` when it did not.
    """
    citation = verdict.citation
    if verdict.verified:
        return f"verified {citation.source_type} at {citation.url}"
    return f"unverified ({verdict.failure}) {citation.source_type} at {citation.url}"


def assess_source_reliability(verdicts: tuple[CitationVerdict, ...]) -> tuple[str, ...]:
    """Stage 6: note the verified reliability of each gathered source.

    The note is truthful about verification: a verdict that failed to verify is
    labelled ``unverified`` with its failure code, never silently reported as
    verified.

    Args:
        verdicts: The per-citation verification verdicts.

    Returns:
        One source-quality note per verdict.
    """
    return tuple(_source_note(verdict) for verdict in verdicts)


def adversarial_counterargument(subquestions: tuple[str, ...]) -> str:
    """Stage 7: form the adversarial counterargument pass.

    Args:
        subquestions: The decomposed subquestions.

    Returns:
        A deterministic counterargument summary.
    """
    return "; ".join(f"counterpoint on {subquestion}" for subquestion in subquestions)


def _build_discard_recorder(
    ledger: ForecastLedgerWriter | None, created_at: datetime | None
) -> _DiscardRecorder | None:
    """Resolve the optional ledger/timestamp pair into a discard recorder.

    Args:
        ledger: The forecast-event ledger writer, or ``None`` to record nothing.
        created_at: The creation instant events are stamped with; required
            whenever ``ledger`` is supplied.

    Returns:
        A :class:`_DiscardRecorder` when a ledger is wired, else ``None``.

    Raises:
        ValueError: If ``ledger`` is supplied without ``created_at`` -- an
            event could not be timestamped, so the call fails loudly rather
            than fabricating an instant.
    """
    if ledger is None:
        return None
    if created_at is None:
        raise ValueError(
            "collect_model_votes requires created_at when a ledger is supplied"
        )
    return _DiscardRecorder(ledger=ledger, ts=_iso_z(created_at))


def _build_model_vote(forecast: ProviderForecast) -> ModelVote:
    """Assemble one :class:`ModelVote` from a provider's parsed forecast.

    The vote's probability comes from the parsed response (via ``forecast``),
    never from the baseline, and its provenance and fingerprint are threaded
    straight through from the provider.

    Args:
        forecast: The provider's structured forecast for this vote.

    Returns:
        The assembled model vote.
    """
    return ModelVote(
        provider=forecast.provider,
        model_version=forecast.model_version,
        declared_training_cutoff=forecast.training_cutoff,
        probability_ppm=forecast.probability_ppm,
        response_fingerprint=forecast.response_fingerprint,
    )


def _resolve_vote_ensemble(
    ensemble: tuple[EnsembleMemberLike, ...] | None,
) -> tuple[EnsembleMemberLike, ...]:
    """Resolve the caller's ensemble override, or the pinned default.

    Args:
        ensemble: A caller-supplied ensemble, or ``None`` for the default.

    Returns:
        The supplied ensemble unchanged, or the package-default
        :data:`~windbreak.forecast.providers.DEFAULT_VOTE_ENSEMBLE`.
    """
    if ensemble is not None:
        return ensemble
    return DEFAULT_VOTE_ENSEMBLE


def _build_provider(
    provider_factory: Callable[[EnsembleMemberLike], ForecastProvider] | None,
    transport: LlmTransport,
    member: EnsembleMemberLike,
) -> ForecastProvider:
    """Resolve the provider driving one ensemble member's vote.

    With no ``provider_factory`` (the default) the pre-existing network-free
    :class:`~windbreak.forecast.providers.FixtureVoteProvider` is built over
    ``transport`` exactly as before -- byte-identical behavior. A supplied
    factory instead builds the provider from the member alone (an HTTP-backed
    :class:`~windbreak.forecast.providers.FutureSearchProvider`, say), so
    ``transport`` is never touched on that path.

    Args:
        provider_factory: The caller's provider factory, or ``None`` for the
            default fixture provider.
        transport: The LLM transport the default fixture provider votes through.
        member: The ensemble member driving this vote.

    Returns:
        The provider to obtain this member's forecast from.
    """
    if provider_factory is None:
        return FixtureVoteProvider(transport, member)
    return provider_factory(member)


def _is_transport_failure(error: ProviderVoteError) -> bool:
    """Return whether a discarded vote's failure means no provider was reached.

    A discard is *transport-class* -- meaning the provider could not be reached
    or answered only with a transient, retryable condition -- iff it is a
    timeout, a rate-limit, or an HTTP error whose status
    :func:`~windbreak.forecast.providers.retry.is_retryable_status` accepts
    (``429`` or ``5xx``). This is the pipeline-classification twin of the retry
    layer's :func:`~windbreak.forecast.providers.retry._is_retryable` predicate:
    a *non-retryable* HTTP status (a ``4xx`` such as 400/403/404) means the
    provider **was** reached and responded, so it is deliberately *not*
    transport-class -- classifying it as such would stamp a zero-survivor run
    ``provider_unavailable`` and ledger a rationale claiming "no provider could
    be reached", which a 4xx response makes factually false. Every screen-side
    rejection (malformed / version-drift / response-rejected) and every cost
    overrun is likewise non-transport.

    Args:
        error: The caught, discarded provider-vote failure to classify.

    Returns:
        ``True`` iff ``error`` is a transport-class fault, else ``False``.
    """
    return isinstance(error, ProviderTimeoutError | ProviderRateLimitedError) or (
        isinstance(error, ProviderHTTPError) and is_retryable_status(error.status_code)
    )


@dataclass(frozen=True, slots=True)
class _VoteCollection:
    """The outcome of driving every ensemble member's provider once.

    Bundles the surviving forecasts with the aggregate cost and per-discard
    transport classification of the *discarded* votes, so :func:`run_pipeline`
    can both charge the failed spend and classify a zero/under-quorum-survivor
    abstention -- while :func:`collect_model_votes` keeps its narrower,
    forecasts-only contract.

    Attributes:
        forecasts: The surviving provider forecasts, in call order.
        discarded_cost_micros: The summed ``cost_micros`` of every discarded
            vote's failure, charged into the research budget even though those
            votes never reach aggregation.
        discard_transport_flags: Each discarded vote's
            :func:`_is_transport_failure` result, in discard order, used to
            distinguish a transport-only wipeout (no provider reached) from a
            screen-side or non-retryable-HTTP one.
    """

    forecasts: tuple[ProviderForecast, ...]
    discarded_cost_micros: int
    discard_transport_flags: tuple[bool, ...]


def _survival_outcome(forecast: ProviderForecast) -> str:
    """Classify a surviving forecast's vote-cost outcome (issue #281).

    A surviving vote either voted or explicitly abstained (SPEC S6.3); both
    carry an empty ``failure_code`` (they answered -- an abstention is a "no
    vote" answer, never a failure).

    Args:
        forecast: The surviving provider forecast.

    Returns:
        :data:`VOTE_OUTCOME_ABSTAINED` if the forecast abstained, else
        :data:`VOTE_OUTCOME_VOTED`.
    """
    return VOTE_OUTCOME_ABSTAINED if forecast.abstain else VOTE_OUTCOME_VOTED


def _collect_provider_forecasts(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    *,
    transport: LlmTransport,
    quotes: tuple[ResearchQuote, ...],
    recorder: _DiscardRecorder | None,
    members: tuple[EnsembleMemberLike, ...],
    provider_factory: Callable[[EnsembleMemberLike], ForecastProvider] | None,
) -> _VoteCollection:
    """Drive each member's provider into a :class:`_VoteCollection`.

    One provider per member (see :func:`_build_provider`) produces one forecast;
    any per-vote failure crossing the seam
    (:class:`~windbreak.forecast.providers.base.ProviderVoteError` -- a
    screen-side rejection, a version drift, or a transport-class timeout/
    rate-limit/HTTP/cost-overrun fault) is *discarded* rather than crashing the
    whole run (#189, #193). A discarded vote is ledgered (when a recorder is
    wired) with a fingerprint-only payload, its ``cost_micros`` accumulated for
    the budget seam, and its :func:`_is_transport_failure` classification
    collected so the caller can tell a transport wipeout (no provider reached)
    from a screen-side or non-retryable-HTTP rejection. The result holds between
    zero and ``len(members)`` surviving forecasts, in call order.

    Args:
        market: The market under forecast.
        baseline: The baseline quote snapshot.
        transport: The LLM transport the default fixture provider votes through.
        quotes: The sanitized web quotes threaded into each fixture vote prompt.
        recorder: The discard recorder, or ``None`` to ledger nothing.
        members: The resolved ensemble members to drive, in order.
        provider_factory: The caller's provider factory, or ``None`` for the
            default fixture provider.

    Returns:
        The surviving forecasts plus the discarded votes' aggregate cost and
        per-discard transport-class flags.
    """
    forecasts: list[ProviderForecast] = []
    discarded_cost_micros = 0
    discard_transport_flags: list[bool] = []
    for index, member in enumerate(members):
        provider = _build_provider(provider_factory, transport, member)
        try:
            forecast = provider.forecast(market, baseline, index, quotes)
        except ProviderVoteError as failed:
            discarded_cost_micros += failed.cost_micros
            discard_transport_flags.append(_is_transport_failure(failed))
            if recorder is not None:
                recorder.record_discard(
                    market_ticker=market.ticker,
                    member=member,
                    vote_index=index,
                    failure=failed.failure_code,
                    response_fingerprint=failed.response_fingerprint,
                )
                recorder.record_vote_cost(
                    market_ticker=market.ticker,
                    member=member,
                    vote_index=index,
                    cost_micros=failed.cost_micros,
                    outcome=VOTE_OUTCOME_DISCARDED,
                    failure_code=failed.failure_code,
                )
            continue
        forecasts.append(forecast)
        if recorder is not None:
            recorder.record_vote_cost(
                market_ticker=market.ticker,
                member=member,
                vote_index=index,
                cost_micros=forecast.cost_micros,
                outcome=_survival_outcome(forecast),
                failure_code="",
            )
    return _VoteCollection(
        forecasts=tuple(forecasts),
        discarded_cost_micros=discarded_cost_micros,
        discard_transport_flags=tuple(discard_transport_flags),
    )


def collect_model_votes(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    *,
    transport: LlmTransport,
    quotes: tuple[ResearchQuote, ...] = (),
    ledger: ForecastLedgerWriter | None = None,
    created_at: datetime | None = None,
    ensemble: tuple[EnsembleMemberLike, ...] | None = None,
    provider_factory: Callable[[EnsembleMemberLike], ForecastProvider] | None = None,
) -> tuple[ModelVote, ...]:
    """Stage 8: collect the surviving, injection-screened model votes.

    This is the only stage that touches the transport seam. It drives one
    provider per ensemble member (a
    :class:`~windbreak.forecast.providers.FixtureVoteProvider` by default, or a
    caller-supplied one via ``provider_factory``) -- ``len(ensemble)`` calls in
    call order (three by default) -- screening each response and parsing the
    structured vote. A response that forges a delimiter, lures a tool call, or
    fails schema validation (e.g. a non-integer ``probability_ppm``) raises
    :class:`~windbreak.forecast.providers.ProviderResponseRejectedError`; that
    vote is *discarded* (never trusted, never retried) and, when a ledger is
    wired, recorded as a :data:`FORECAST_OUTPUT_DISCARDED_EVENT` with a
    fingerprint-only payload. A provider-reported forecaster-version drift under
    the strict policy
    (:class:`~windbreak.forecast.providers.ProviderVersionDriftError`) is
    discarded per-vote in exactly the same way -- one vote dropped and ledgered
    ``FORECAST_OUTPUT_DISCARDED`` with the version-drift failure code and the
    drifted response's fingerprint -- never a whole-run crash (#189). Each
    surviving vote's ``probability_ppm`` is parsed
    from its response (SPEC S6.3), not derived from the baseline, while its
    fingerprint records provider drift (T14). The returned tuple therefore holds
    between zero and ``len(ensemble)`` votes. This stage never charges the failed
    votes' cost: the discarded spend is owned by :func:`run_pipeline`, which holds
    the budget seam and charges it there.

    Args:
        market: The market under forecast.
        baseline: The baseline quote snapshot.
        transport: The LLM transport (recording, replay, or forbidden-live).
        quotes: The sanitized web quotes to thread into each vote prompt as
            untrusted-data blocks (keyword-only; empty by default).
        ledger: The forecast-event ledger writer for discard events, or
            ``None`` to record nothing (keyword-only).
        created_at: The creation instant discard events are stamped with
            (keyword-only); required whenever ``ledger`` is supplied.
        ensemble: The vote ensemble to drive, or ``None`` (keyword-only) for the
            pinned default three-member ensemble. Its length sets the number of
            transport calls, and each member's provenance stamps its vote.
        provider_factory: An optional factory building the provider for a member
            (keyword-only). ``None`` (the default) builds a
            :class:`~windbreak.forecast.providers.FixtureVoteProvider` over
            ``transport``, byte-identically to before.

    Returns:
        The surviving, non-abstaining ensemble votes, in call order (0 to
        ``len(ensemble)``). A member that explicitly abstained (SPEC S6.3) is
        excluded from this tuple even though its cost was charged upstream, so
        it never contributes to the median.

    Raises:
        ValueError: If ``ledger`` is supplied without ``created_at``.
    """
    recorder = _build_discard_recorder(ledger, created_at)
    members = _resolve_vote_ensemble(ensemble)
    collection = _collect_provider_forecasts(
        market,
        baseline,
        transport=transport,
        quotes=quotes,
        recorder=recorder,
        members=members,
        provider_factory=provider_factory,
    )
    voting_forecasts, _ = _partition_abstaining(collection.forecasts)
    return tuple(_build_model_vote(forecast) for forecast in voting_forecasts)


def aggregate_median(votes: tuple[ModelVote, ...]) -> VoteAggregate:
    """Stage 9: aggregate votes into a median with confidence bounds (S8.6).

    Delegates to :func:`windbreak.forecast.ensemble.aggregate_votes`, which owns
    the integer-median and exclusive-median IQR math; ``vote_dispersion_ppm``
    is now the inter-quartile spread rather than the raw max-minus-min range.

    Args:
        votes: The ensemble votes to aggregate.

    Returns:
        The median probability with low/high bounds and IQR dispersion.
    """
    return aggregate_votes(votes)


def normalize_coherence(market: NormalizedMarket) -> int | None:
    """Stage 10: normalize probability coherence within a group (S8.7 stub).

    A single-market v1 forecast has no mutually-exclusive peer group, so there
    is no cross-outcome probability sum to normalize; the group sum is
    therefore undefined. Structural group-coherence is post-v1 (SPEC S8.7).

    Args:
        market: The market under forecast (its group membership determines
            whether coherence applies; v1 markets always stand alone).

    Returns:
        Always ``None`` in v1: there is no group sum to report.
    """
    return None


def apply_calibration_map(
    probability_ppm: int, calibration_map: CalibrationMap | None = None
) -> int:
    """Stage 11: apply the versioned calibration map to an aggregate probability.

    With no map (the default) the v0 identity behavior is preserved byte-for-
    byte: the value is returned unchanged after a defensive clamp back into the
    ppm domain. A wired :class:`~windbreak.forecast.calibration.CalibrationMap`
    instead corrects the value through its own integer piecewise-linear
    interpolation, re-clamped defensively.

    Args:
        probability_ppm: The aggregated probability, in ppm.
        calibration_map: The fitted calibration map to apply, or ``None`` (the
            default) for the identity-clamp behavior.

    Returns:
        The calibrated probability, in ppm.
    """
    if calibration_map is None:
        return _clamp_ppm(probability_ppm)
    return _clamp_ppm(calibration_map.apply(probability_ppm))


def shrink_toward_baseline(probability_ppm: int, baseline_ppm: int) -> int:
    """Stage 12: shrink the estimate toward the market baseline (integer λ).

    Blends the estimate with the baseline at a fixed weight using integer
    math only (floor division), so no float ever enters the probability path.

    Args:
        probability_ppm: The calibrated probability, in ppm.
        baseline_ppm: The market baseline probability, in ppm.

    Returns:
        The shrunk probability, in ppm.
    """
    blended = (
        probability_ppm * (_PPM_SCALE - _SHRINK_LAMBDA_PPM)
        + baseline_ppm * _SHRINK_LAMBDA_PPM
    ) // _PPM_SCALE
    return _clamp_ppm(blended)


def _build_rationale(criteria: str, counterpoints: str) -> str:
    """Assemble the deterministic rationale markdown.

    Args:
        criteria: The extracted resolution criteria.
        counterpoints: The adversarial counterargument summary.

    Returns:
        A markdown rationale string.
    """
    return (
        "## Rationale\n\n"
        f"Resolution criteria: {criteria}\n\n"
        f"Counterarguments considered: {counterpoints}\n"
    )


def _forecast_id(question_hash: str, snapshot_id: str, created_at: datetime) -> str:
    """Derive a deterministic forecast id from its provenance fields.

    Args:
        question_hash: The normalized-question hash.
        snapshot_id: The baseline snapshot identifier.
        created_at: The forecast creation instant.

    Returns:
        A sha256 hex digest over the canonical JSON of the three inputs.
    """
    canonical = json.dumps(
        [question_hash, snapshot_id, _iso_z(created_at)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _forecast_horizon_hours(market: NormalizedMarket, created_at: datetime) -> int:
    """Compute whole hours from creation to the market's close (integer math).

    Args:
        market: The market under forecast.
        created_at: The forecast creation instant.

    Returns:
        The horizon in whole hours.
    """
    delta = market.close_time - created_at
    return delta.days * _HOURS_PER_DAY + delta.seconds // _SECONDS_PER_HOUR


def build_forecast_record(
    *,
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    question_hash: str,
    probability_ppm: int,
    aggregate: VoteAggregate,
    votes: tuple[ModelVote, ...],
    citations: tuple[Citation, ...],
    source_notes: tuple[str, ...],
    rationale: str,
    coherence_sum: int | None,
    eligible_for_live: bool = True,
) -> ForecastRecord:
    """Assemble and validate the final :class:`ForecastRecord`.

    Args:
        market: The market under forecast.
        baseline: The baseline quote snapshot.
        created_at: The forecast creation instant.
        question_hash: The normalized-question hash.
        probability_ppm: The final, bound-clamped probability, in ppm.
        aggregate: The median vote aggregation (bounds and dispersion).
        votes: The ensemble votes.
        citations: The verified supporting citations.
        source_notes: The source-quality notes.
        rationale: The rationale markdown.
        coherence_sum: The coherence group sum, or None.
        eligible_for_live: Whether the record may back a live order; defaults to
            ``True`` and is overridden by the caller's live-eligibility gate.

    Returns:
        A schema-valid, immutable forecast record.
    """
    return ForecastRecord(
        forecast_id=_forecast_id(question_hash, baseline.snapshot_id, created_at),
        market_ticker=market.ticker,
        normalized_question_hash=question_hash,
        probability_ppm=probability_ppm,
        ci_low_ppm=aggregate.ci_low_ppm,
        ci_high_ppm=aggregate.ci_high_ppm,
        model_votes=votes,
        vote_dispersion_ppm=aggregate.vote_dispersion_ppm,
        rationale_markdown=rationale,
        citations=citations,
        source_quality_notes=source_notes,
        research_cost_micros=_RESEARCH_COST_MICROS,
        triage_stage="full",
        created_at=created_at,
        forecast_horizon_hours=_forecast_horizon_hours(market, created_at),
        market_price_baseline_pips=baseline.price_pips,
        baseline_quote_snapshot_id=baseline.snapshot_id,
        coherence_group_sum_ppm=coherence_sum,
        coherence_flag=coherence_sum is not None,
        abstention_reason=None,
        eligible_for_live=eligible_for_live,
    )


def _build_abstention_record(
    *,
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    question_hash: str,
    citations: tuple[Citation, ...],
    abstention_reason: str,
) -> ForecastRecord:
    """Assemble a schema-valid abstention record (mirrors the triage-only one).

    The baseline collapses the point estimate and both confidence bounds onto
    the same ppm value (no ensemble ran), the gathered citations are retained
    for audit, and the record is permanently live-ineligible with its
    abstention reason set.

    Args:
        market: The market under forecast.
        baseline: The baseline quote snapshot.
        created_at: The forecast creation instant.
        question_hash: The normalized-question hash.
        citations: The gathered citations, retained for audit.
        abstention_reason: Why the engine abstained. Must be a known reason.

    Returns:
        A schema-valid, immutable abstention forecast record.

    Raises:
        ValueError: If ``abstention_reason`` has no registered rationale, so a
            new abstention path can never silently ship a mismatched rationale.
    """
    try:
        rationale_markdown = _ABSTENTION_RATIONALE_BY_REASON[abstention_reason]
    except KeyError as exc:
        raise ValueError(
            f"No rationale registered for abstention reason {abstention_reason!r}"
        ) from exc
    baseline_ppm = _baseline_probability_ppm(baseline)
    return ForecastRecord(
        forecast_id=_forecast_id(question_hash, baseline.snapshot_id, created_at),
        market_ticker=market.ticker,
        normalized_question_hash=question_hash,
        probability_ppm=baseline_ppm,
        ci_low_ppm=baseline_ppm,
        ci_high_ppm=baseline_ppm,
        model_votes=(),
        vote_dispersion_ppm=0,
        rationale_markdown=rationale_markdown,
        citations=citations,
        source_quality_notes=(),
        research_cost_micros=_RESEARCH_COST_MICROS,
        triage_stage="full",
        created_at=created_at,
        forecast_horizon_hours=_forecast_horizon_hours(market, created_at),
        market_price_baseline_pips=baseline.price_pips,
        baseline_quote_snapshot_id=baseline.snapshot_id,
        coherence_group_sum_ppm=None,
        coherence_flag=False,
        abstention_reason=abstention_reason,
        eligible_for_live=False,
    )


def _all_transport_discards(flags: tuple[bool, ...]) -> bool:
    """Return whether every discard was a transport-class fault (≥1 discard).

    Args:
        flags: Each discarded vote's :func:`_is_transport_failure` result, in
            discard order.

    Returns:
        ``True`` iff at least one vote was discarded and every discard was
        transport-class.
    """
    return bool(flags) and all(flags)


def _partition_abstaining(
    forecasts: tuple[ProviderForecast, ...],
) -> tuple[tuple[ProviderForecast, ...], int]:
    """Split provider forecasts into voting survivors and the abstainer count.

    An abstaining member is charged like any other (it did call the provider),
    but declined to cast a usable vote (SPEC S6.3), so it is dropped from the
    aggregation and its provider-reported citations never reach the audit trail.

    Args:
        forecasts: Every surviving provider forecast, in call order.

    Returns:
        A ``(voting_forecasts, abstain_count)`` pair: the non-abstaining
        forecasts (in call order) and the number of abstaining forecasts.
    """
    voting = tuple(forecast for forecast in forecasts if not forecast.abstain)
    return voting, len(forecasts) - len(voting)


def _vote_shortfall_reason(
    vote_count: int,
    min_ensemble_votes: int,
    discard_transport_flags: tuple[bool, ...],
    *,
    abstain_count: int,
) -> str | None:
    """Classify a vote shortfall into its abstention reason, or ``None``.

    First-match-wins precedence (issue #241):

    1. ``vote_count >= min_ensemble_votes`` -> ``None``: quorum is met (even with
       an abstainer in the mix), so the run proceeds to full aggregation.
    2. ``abstain_count > 0`` ->
       :data:`ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED`: any explicit abstention
       below quorum takes precedence over the transport/screen taxonomy, even
       when a transport wipeout is also in the discard mix.
    3. ``vote_count == 0`` (and no abstainers): an all-transport wipeout means no
       provider could be reached (:data:`ABSTENTION_PROVIDER_UNAVAILABLE`),
       whereas any non-transport discard -- a screen-side rejection or a
       non-retryable HTTP response -- keeps
       :data:`ABSTENTION_ALL_VOTES_DISCARDED`.
    4. Otherwise (survivors present but below quorum, no abstainers) the run
       abstains :data:`ABSTENTION_ENSEMBLE_QUORUM_NOT_MET`.

    Args:
        vote_count: The number of surviving (non-abstaining) votes.
        min_ensemble_votes: The minimum surviving votes a full record requires.
        discard_transport_flags: Each discarded vote's
            :func:`_is_transport_failure` result, in discard order.
        abstain_count: The number of members that explicitly abstained
            (keyword-only).

    Returns:
        The abstention reason to stamp, or ``None`` when quorum is met.
    """
    if vote_count >= min_ensemble_votes:
        return None
    if abstain_count > 0:
        return ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED
    if vote_count == 0:
        if _all_transport_discards(discard_transport_flags):
            return ABSTENTION_PROVIDER_UNAVAILABLE
        return ABSTENTION_ALL_VOTES_DISCARDED
    return ABSTENTION_ENSEMBLE_QUORUM_NOT_MET


def _live_gate_open(canary_gate: CanaryGate | None, created_at: datetime) -> bool:
    """Return whether the canary gate permits live eligibility for a record.

    Extracted so :func:`run_pipeline` stays a flat, low-complexity wiring
    function: the gate is open (returns ``True``) when there is no gate at all
    or the gate is not blocking a record created at ``created_at``.

    Args:
        canary_gate: The optional canary drift gate; ``None`` means no gate.
        created_at: The record's creation instant.

    Returns:
        ``True`` if the gate permits live eligibility, else ``False``.
    """
    return canary_gate is None or not canary_gate.is_live_blocked(created_at=created_at)


def _apply_and_record_calibration(
    probability_ppm: int,
    calibration_map: CalibrationMap | None,
    ledger: ForecastLedgerWriter | None,
    created_at: datetime,
) -> int:
    """Apply the calibration map (if any), ledgering the exact ppm transition.

    Extracted so :func:`run_pipeline` stays a flat, low-complexity wiring
    function. With no map this is the byte-identical identity clamp and records
    nothing (even with a ledger wired). With a map it first enforces temporal
    integrity -- so a future-trained map fails the whole run closed -- then
    applies it and, when a ledger is wired, records one
    :data:`CALIBRATION_MAP_APPLIED_EVENT` carrying the pre-/post-calibration ppm.

    Args:
        probability_ppm: The aggregate median probability to calibrate, in ppm.
        calibration_map: The fitted map to apply, or ``None`` for the identity.
        ledger: The forecast-event ledger writer, or ``None`` to record nothing.
        created_at: The forecast creation instant (temporal-integrity anchor and
            event timestamp).

    Returns:
        The calibrated probability, in ppm.

    Raises:
        TemporalIntegrityError: If ``calibration_map`` is trained after
            ``created_at``.
    """
    if calibration_map is None:
        return apply_calibration_map(probability_ppm)
    ensure_temporal_integrity(calibration_map, forecast_created_at=created_at)
    calibrated_ppm = apply_calibration_map(probability_ppm, calibration_map)
    if ledger is not None:
        payload: dict[str, object] = {
            "map_id": calibration_map.map_id,
            "map_version": calibration_map.version,
            "input_ppm": probability_ppm,
            "output_ppm": calibrated_ppm,
        }
        ledger.record(
            ForecastEvent(CALIBRATION_MAP_APPLIED_EVENT, payload, _iso_z(created_at))
        )
    return calibrated_ppm


def _provider_gate_open(
    provider_gate: ProviderTrackRecordGate | None,
    votes: tuple[ModelVote, ...],
    ledger: ForecastLedgerWriter | None,
    created_at: datetime,
) -> bool:
    """Return whether the provider gate permits live eligibility for a run.

    Mirrors :func:`_live_gate_open`: the gate is open (returns ``True``) when
    there is no gate at all or every voting provider is proven. When one or more
    voting providers are unproven the gate holds the run back (returns
    ``False``) and, if a ledger is wired, records one
    :data:`PROVIDER_GATE_HELD_EVENT` naming the unproven providers. Computed as a
    statement -- never short-circuited against the canary gate -- so the hold is
    always ledgered, even when another gate also blocks the run.

    Args:
        provider_gate: The optional per-provider track-record gate; ``None``
            means no gate.
        votes: The surviving ensemble votes whose providers are screened.
        ledger: The forecast-event ledger writer, or ``None`` to record nothing.
        created_at: The forecast creation instant (event timestamp).

    Returns:
        ``True`` if the gate permits live eligibility, else ``False``.
    """
    if provider_gate is None:
        return True
    unproven = provider_gate.unproven_providers(vote.provider for vote in votes)
    if not unproven:
        return True
    if ledger is not None:
        payload: dict[str, object] = {
            "unproven_providers": ",".join(unproven),
            "unproven_count": len(unproven),
            "min_resolved": provider_gate.min_resolved,
            "min_brier_skill_ppm": provider_gate.min_brier_skill_ppm,
        }
        ledger.record(
            ForecastEvent(PROVIDER_GATE_HELD_EVENT, payload, _iso_z(created_at))
        )
    return False


def _full_run_eligible_for_live(
    *,
    canary_gate: CanaryGate | None,
    provider_gate_ok: bool,
    created_at: datetime,
    verified_count: int,
    min_verified_citations: int,
    coherence_sum: int | None,
) -> bool:
    """Combine every live-eligibility gate for a full (non-abstaining) run.

    A full run is live-eligible only when the canary drift gate is open, the
    per-provider track-record gate is open (``provider_gate_ok``), *and* the
    citation/stage/coherence invariants of :func:`is_live_eligible` all hold.
    Extracted from :func:`run_pipeline` so the wiring function stays a flat,
    low-complexity chain; the ``provider_gate_ok`` argument is pre-computed by
    the caller (never short-circuited) so a held provider gate is always
    ledgered even when the canary gate also blocks.

    Args:
        canary_gate: The optional canary drift gate; ``None`` means no gate.
        provider_gate_ok: Whether the provider track-record gate is open,
            pre-computed by :func:`_provider_gate_open`.
        created_at: The record's creation instant.
        verified_count: How many citations independently verified.
        min_verified_citations: The minimum verified citations required.
        coherence_sum: The coherence group sum, or ``None`` when the market
            stands alone.

    Returns:
        ``True`` if the full run may back a live order, else ``False``.
    """
    return (
        _live_gate_open(canary_gate, created_at)
        and provider_gate_ok
        and is_live_eligible(
            verified_citation_count=verified_count,
            min_verified_citations=min_verified_citations,
            triage_stage="full",
            coherence_flag=coherence_sum is not None,
            abstention_reason=None,
        )
    )


def _reported_citation(citation: ProviderCitation) -> Citation:
    """Map one provider-reported citation into an audit-trail :class:`Citation`.

    The ``content_hash`` is a sha256 over the citation's own reported provenance
    (its ``{url, publication_date, quoted_text}``) -- a marker of *provider-
    reported* provenance, explicitly NOT an independently verified content hash;
    the ``source_type`` is stamped :data:`PROVIDER_REPORTED_SOURCE_TYPE` so this
    citation is retained for audit yet never counts toward the verified-citation
    live-eligibility gate (SPEC S8.8).

    Args:
        citation: The provider-reported citation to map.

    Returns:
        The audit-trail citation.
    """
    publication_iso = (
        _iso_z(citation.publication_date)
        if citation.publication_date is not None
        else None
    )
    canonical = json.dumps(
        {
            "url": citation.url,
            "publication_date": publication_iso,
            "quoted_text": citation.quoted_text,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Citation(
        url=citation.url,
        content_hash=f"{_SHA256_PREFIX}{digest}",
        quoted_text=citation.quoted_text,
        publication_date=citation.publication_date,
        source_type=PROVIDER_REPORTED_SOURCE_TYPE,
    )


def _reported_citations(
    forecasts: tuple[ProviderForecast, ...],
) -> tuple[Citation, ...]:
    """Flatten every surviving forecast's reported citations, in order.

    Args:
        forecasts: The surviving provider forecasts.

    Returns:
        Every provider-reported citation mapped to an audit-trail
        :class:`Citation`, in forecast-then-citation order.
    """
    return tuple(
        _reported_citation(citation)
        for forecast in forecasts
        for citation in forecast.citations
    )


def _charge_research(
    budget: ResearchBudget | None,
    cost_micros: int,
    market: NormalizedMarket,
    created_at: datetime,
) -> None:
    """Charge the run's research cost against ``budget`` when one is wired.

    Args:
        budget: The research budget, or ``None`` for a no-op (the tracer path).
        cost_micros: The research cost to charge, in micros.
        market: The market under forecast, for the audit trail.
        created_at: The run's creation instant, bucketing the spend to a day.

    Raises:
        PerForecastBudgetExceededError: If ``cost_micros`` exceeds the
            per-forecast ceiling.
    """
    if budget is not None:
        budget.charge_forecast(cost_micros, market_ticker=market.ticker, at=created_at)


def _open_budget_day(budget: ResearchBudget | None, created_at: datetime) -> int | None:
    """Open the budget's UTC day and return stage 5's page ceiling.

    Args:
        budget: The research budget, or ``None`` for the no-op tracer path.
        created_at: The run's creation instant selecting the budget day.

    Returns:
        The budget's ``max_pages`` ceiling bounding stage 5's fetches, or
        ``None`` when no budget is wired (an unbounded fetch, exactly as before).

    Raises:
        DailyBudgetExhaustedError: If ``budget`` is supplied and its UTC day is
            already exhausted; raised before any research.
    """
    if budget is None:
        return None
    budget.ensure_day_open(at=created_at)
    return budget.max_pages


def _verified_quotes(
    verdicts: tuple[CitationVerdict, ...],
) -> tuple[ResearchQuote, ...]:
    """Thread only the *verified* citations' quotes into the vote prompts.

    Args:
        verdicts: Every gathered citation's verification verdict.

    Returns:
        One :class:`ResearchQuote` per verified verdict (SPEC S8.5), in verdict
        order; an unverified citation contributes no quote.
    """
    return tuple(
        ResearchQuote(url=verdict.citation.url, text=verdict.citation.quoted_text)
        for verdict in verdicts
        if verdict.verified
    )


@dataclass(frozen=True, slots=True)
class _VoteBundle:
    """The vote stage's survivors paired with any quorum-shortfall verdict.

    Attributes:
        votes: The surviving ensemble votes, in call order, ready to aggregate.
        forecasts: The voting (non-abstaining) provider forecasts backing
            ``votes``, whose provider-reported citations land in the record's
            audit trail. An abstaining member is charged but never appears here,
            so its reported citations are never cited and its provider is never
            screened by the provider gate.
        shortfall_reason: The abstention reason when too few votes survived to
            aggregate over (see :func:`_vote_shortfall_reason`), or ``None`` when
            the quorum is met and aggregation may proceed.
    """

    votes: tuple[ModelVote, ...]
    forecasts: tuple[ProviderForecast, ...]
    shortfall_reason: str | None


def _collect_votes(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    *,
    transport: LlmTransport,
    quotes: tuple[ResearchQuote, ...],
    created_at: datetime,
    min_ensemble_votes: int,
    ledger: ForecastLedgerWriter | None,
    budget: ResearchBudget | None,
    ensemble: tuple[EnsembleMemberLike, ...] | None,
    provider_factory: Callable[[EnsembleMemberLike], ForecastProvider] | None,
) -> _VoteBundle:
    """Drive the vote stage, charge its spend, and classify any shortfall.

    Collects one forecast per ensemble member (discarding per-vote failures),
    charges the run's full research spend -- the fixed stub cost plus every
    surviving *and* discarded vote's cost (an abstaining member is charged too:
    it did call the provider) -- into ``budget``. It then partitions off any
    explicitly abstaining members (:func:`_partition_abstaining`): only the
    voting survivors build model votes and back the bundle's forecasts, while the
    abstainer count feeds the shortfall classification. The bundle's shortfall
    reason decides whether the voting-survivor count clears
    ``min_ensemble_votes``.

    Args:
        market: The market under forecast.
        baseline: The baseline quote snapshot the votes are struck against.
        transport: The LLM transport the default fixture provider votes through.
        quotes: The sanitized, verified web quotes threaded into each vote prompt.
        created_at: The run's creation instant, bucketing the budget spend.
        min_ensemble_votes: The minimum surviving votes a full record requires.
        ledger: The ledger writer for vote-discard events, or ``None``.
        budget: The research budget to charge, or ``None`` for a no-op.
        ensemble: The vote ensemble to drive, or ``None`` for the pinned default.
        provider_factory: The caller's provider factory, or ``None`` for the
            default fixture provider.

    Returns:
        The surviving votes and forecasts paired with the shortfall abstention
        reason, which is ``None`` when the quorum is met.

    Raises:
        PerForecastBudgetExceededError: If the run's research cost exceeds the
            per-forecast ceiling.
    """
    collection = _collect_provider_forecasts(
        market,
        baseline,
        transport=transport,
        quotes=quotes,
        recorder=_build_discard_recorder(ledger, created_at),
        members=_resolve_vote_ensemble(ensemble),
        provider_factory=provider_factory,
    )
    provider_cost_micros = sum(
        forecast.cost_micros for forecast in collection.forecasts
    )
    _charge_research(
        budget,
        _RESEARCH_COST_MICROS + provider_cost_micros + collection.discarded_cost_micros,
        market,
        created_at,
    )
    voting_forecasts, abstain_count = _partition_abstaining(collection.forecasts)
    votes = tuple(_build_model_vote(forecast) for forecast in voting_forecasts)
    shortfall_reason = _vote_shortfall_reason(
        len(votes),
        min_ensemble_votes,
        collection.discard_transport_flags,
        abstain_count=abstain_count,
    )
    return _VoteBundle(
        votes=votes, forecasts=voting_forecasts, shortfall_reason=shortfall_reason
    )


def _aggregate_into_record(
    bundle: _VoteBundle,
    *,
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    question_hash: str,
    citations: tuple[Citation, ...],
    criteria: str,
    counterpoints: str,
    base_rate_ppm: int,
    source_notes: tuple[str, ...],
    canary_gate: CanaryGate | None,
    verified_count: int,
    min_verified_citations: int,
    calibration_map: CalibrationMap | None,
    ledger: ForecastLedgerWriter | None,
    provider_gate: ProviderTrackRecordGate | None,
) -> ForecastRecord:
    """Aggregate the surviving votes into the final, live-gated record.

    Runs the median aggregation, calibration, baseline shrinkage, and bound
    clamp into the point estimate, then assembles the schema-valid record. A
    wired ``calibration_map`` is applied (ledgering the ppm transition) after
    temporal-integrity is enforced; a wired ``provider_gate`` holds the run back
    from live eligibility (ledgering a hold) when any voting provider is
    unproven. Live eligibility is the AND of the canary gate staying open, the
    provider gate staying open, and the verified citation count clearing
    ``min_verified_citations`` (SPEC S8.8).

    Args:
        bundle: The surviving votes and forecasts from the vote stage.
        market: The market under forecast.
        baseline: The baseline quote snapshot the forecast is struck against.
        created_at: The run's creation instant.
        question_hash: The normalized-question hash.
        citations: The verified supporting citations; the surviving forecasts'
            provider-reported citations are appended for the audit trail.
        criteria: The extracted resolution criteria, for the rationale.
        counterpoints: The adversarial counterargument, for the rationale.
        base_rate_ppm: The outside-view base rate the estimate shrinks toward.
        source_notes: The per-citation source-quality notes.
        canary_gate: The optional canary drift gate ANDed into live eligibility.
        verified_count: The independently-verified citation count.
        min_verified_citations: The minimum verified citations for live
            eligibility.
        calibration_map: The optional fitted calibration map; ``None`` is the
            byte-identical identity that records nothing.
        ledger: The forecast-event ledger writer, or ``None`` to record nothing;
            carries the calibration-applied and provider-gate-held events.
        provider_gate: The optional per-provider track-record gate ANDed into
            live eligibility; ``None`` means no gate.

    Returns:
        The produced, immutable, live-gated forecast record.

    Raises:
        TemporalIntegrityError: If ``calibration_map`` is trained after
            ``created_at`` (a future-dated map fails the whole run closed).
    """
    aggregate = aggregate_median(bundle.votes)
    coherence_sum = normalize_coherence(market)
    calibrated_ppm = _apply_and_record_calibration(
        aggregate.probability_ppm, calibration_map, ledger, created_at
    )
    shrunk_ppm = shrink_toward_baseline(calibrated_ppm, base_rate_ppm)
    probability_ppm = _clamp_between(
        shrunk_ppm, aggregate.ci_low_ppm, aggregate.ci_high_ppm
    )
    rationale = _build_rationale(criteria, counterpoints)
    # Computed as a statement (never short-circuited) so a PROVIDER_GATE_HELD is
    # ledgered even when the canary gate also blocks the run.
    provider_gate_ok = _provider_gate_open(
        provider_gate, bundle.votes, ledger, created_at
    )
    return build_forecast_record(
        market=market,
        baseline=baseline,
        created_at=created_at,
        question_hash=question_hash,
        probability_ppm=probability_ppm,
        aggregate=aggregate,
        votes=bundle.votes,
        citations=citations + _reported_citations(bundle.forecasts),
        source_notes=source_notes,
        rationale=rationale,
        coherence_sum=coherence_sum,
        eligible_for_live=_full_run_eligible_for_live(
            canary_gate=canary_gate,
            provider_gate_ok=provider_gate_ok,
            created_at=created_at,
            verified_count=verified_count,
            min_verified_citations=min_verified_citations,
            coherence_sum=coherence_sum,
        ),
    )


def run_pipeline(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    *,
    transport: LlmTransport,
    created_at: datetime,
    research_tools: ResearchTools,
    min_verified_citations: int = DEFAULT_MIN_VERIFIED_CITATIONS,
    min_ensemble_votes: int = DEFAULT_MIN_ENSEMBLE_VOTES,
    ledger: ForecastLedgerWriter | None = None,
    budget: ResearchBudget | None = None,
    canary_gate: CanaryGate | None = None,
    ensemble: tuple[EnsembleMemberLike, ...] | None = None,
    provider_factory: Callable[[EnsembleMemberLike], ForecastProvider] | None = None,
    calibration_map: CalibrationMap | None = None,
    provider_gate: ProviderTrackRecordGate | None = None,
) -> ForecastRecord:
    """Run the twelve-stage pipeline into a schema-valid forecast record.

    All stages run offline and deterministically; only
    :func:`collect_model_votes` touches ``transport``, so wiring a forbidden
    or empty transport fails the run closed rather than reaching a network.
    Stage 5 reaches the web only through ``research_tools``, whose egress
    allowlist can make the run raise :class:`EgressDeniedError` before any vote.
    Each gathered citation is then independently re-verified (SPEC S8.8); when
    *zero* verify, the run abstains with
    :data:`ABSTENTION_NO_VERIFIED_CITATIONS` -- returning a live-ineligible
    record *before* the vote stage, so ``transport`` is never touched. Only the
    *verified* citations' sanitized quotes are threaded into the vote prompts
    (SPEC S8.5). Each vote response is itself injection-screened, and any
    transport-class fault (timeout, rate-limit, HTTP, cost overrun) is caught
    the same way; a discarded vote is ledgered through ``ledger`` and its cost
    charged into the budget. An explicitly abstaining member (SPEC S6.3) is
    still charged but excluded from the median, its citations, and the provider
    gate. A vote shortfall then abstains rather than aggregate over too few
    members (see :func:`_vote_shortfall_reason`): any abstainer below quorum
    abstains :data:`ABSTENTION_ENSEMBLE_MEMBERS_ABSTAINED`; with no abstainers,
    zero survivors from an all-transport wipeout abstains
    :data:`ABSTENTION_PROVIDER_UNAVAILABLE`; zero survivors with any screen-side
    rejection abstains :data:`ABSTENTION_ALL_VOTES_DISCARDED`; and survivors
    below ``min_ensemble_votes`` abstain
    :data:`ABSTENTION_ENSEMBLE_QUORUM_NOT_MET`. Otherwise live eligibility is
    gated on the verified count meeting ``min_verified_citations`` *and* the
    canary gate not blocking the record.

    The optional ``ledger``, ``budget``, and ``canary_gate`` seams all default
    to ``None``, a strict no-op: with no ledger, no budget, and no gate (or an
    unexhausted budget and an open gate) the run behaves exactly as before,
    byte-for-byte. When a budget is supplied it halts before any research once
    the day is exhausted, bounds stage 5's fetches to ``budget.max_pages``, and
    charges the run's research cost (raising fail-closed on a per-forecast
    overrun). Given identical inputs
    and ``created_at``, two runs produce equal records and byte-identical
    payloads.

    Args:
        market: The market under forecast.
        baseline: The baseline quote snapshot the forecast is struck against.
        transport: The LLM transport for the vote stage (keyword-only).
        created_at: The injected creation instant, for determinism
            (keyword-only; never ``datetime.now()``).
        research_tools: The sandboxed research tools threaded into stage 5's
            bounded web research (keyword-only).
        min_verified_citations: The minimum independently-verified citations a
            record needs to be live-eligible (keyword-only). Abstention on zero
            verified citations takes absolute precedence over this knob.
        min_ensemble_votes: The minimum surviving ensemble votes a full record
            requires (keyword-only). With survivors present but below this
            quorum the run abstains :data:`ABSTENTION_ENSEMBLE_QUORUM_NOT_MET`
            rather than aggregate over too few members. Must be at least ``1``.
        ledger: The forecast-event ledger writer for vote-discard events, or
            ``None`` to record nothing (keyword-only).
        budget: The optional research budget guarding day/per-forecast spend and
            bounding stage 5's page fetches (keyword-only). ``None`` is a no-op.
        canary_gate: The optional canary drift gate ANDed into live eligibility
            (keyword-only). ``None`` (or an open gate) is a no-op.
        ensemble: The vote ensemble to drive the vote stage with, or ``None``
            (keyword-only) for the pinned default three-member ensemble
            (:data:`~windbreak.forecast.providers.DEFAULT_VOTE_ENSEMBLE`).
        provider_factory: An optional factory building the vote provider per
            member (keyword-only). ``None`` (the default) drives the pre-existing
            fixture-vote path byte-identically; a supplied factory instead drives
            each member through a caller's provider (e.g. an HTTP-backed
            :class:`~windbreak.forecast.providers.FutureSearchProvider`), whose
            reported citations land in the record as
            :data:`PROVIDER_REPORTED_SOURCE_TYPE` entries (audit-only, never
            counted toward live eligibility) and whose reported cost is charged
            into the run's research budget.
        calibration_map: The optional fitted calibration map applied at stage 11
            (keyword-only). ``None`` (the default) is the byte-identical identity
            clamp and records no event; a wired map is temporal-integrity checked
            against ``created_at``, applied to the aggregate median, and ledgered
            as a :data:`CALIBRATION_MAP_APPLIED_EVENT`.
        provider_gate: The optional per-provider track-record gate ANDed into
            live eligibility (keyword-only). ``None`` (the default) is a no-op; a
            wired gate forces ``eligible_for_live=False`` and ledgers one
            :data:`PROVIDER_GATE_HELD_EVENT` when any voting provider is unproven,
            independently of the canary gate (never short-circuited). It never
            changes which votes run.

    Returns:
        The produced, immutable forecast record.

    Raises:
        ValueError: If ``min_ensemble_votes`` is below ``1``; a usage error
            rejected loudly before any stage runs.
        DailyBudgetExhaustedError: If ``budget`` is supplied and the run's UTC
            day is already exhausted; raised before any research.
        PerForecastBudgetExceededError: If ``budget`` is supplied and the run's
            research cost exceeds the per-forecast ceiling.
        TemporalIntegrityError: If ``calibration_map`` is supplied and trained
            after the forecast's own ``created_at`` (a future-dated map fails the
            whole run closed).
    """
    if min_ensemble_votes < 1:
        msg = f"min_ensemble_votes must be at least 1, got {min_ensemble_votes}"
        raise ValueError(msg)
    max_pages = _open_budget_day(budget, created_at)
    question_hash = normalize_question(market)
    criteria = extract_resolution_criteria(market)
    base_rate_ppm = outside_view_base_rate(baseline)
    subquestions = decompose_subquestions(market)
    citations = bounded_web_research(
        subquestions, tools=research_tools, max_pages=max_pages
    )
    verdicts = verify_citations(research_tools, citations, as_of=created_at)
    verified_count = count_verified(verdicts)
    if verified_count == 0:
        _charge_research(budget, _RESEARCH_COST_MICROS, market, created_at)
        return _build_abstention_record(
            market=market,
            baseline=baseline,
            created_at=created_at,
            question_hash=question_hash,
            citations=citations,
            abstention_reason=ABSTENTION_NO_VERIFIED_CITATIONS,
        )
    source_notes = assess_source_reliability(verdicts)
    counterpoints = adversarial_counterargument(subquestions)
    bundle = _collect_votes(
        market,
        baseline,
        transport=transport,
        quotes=_verified_quotes(verdicts),
        created_at=created_at,
        min_ensemble_votes=min_ensemble_votes,
        ledger=ledger,
        budget=budget,
        ensemble=ensemble,
        provider_factory=provider_factory,
    )
    if bundle.shortfall_reason is not None:
        return _build_abstention_record(
            market=market,
            baseline=baseline,
            created_at=created_at,
            question_hash=question_hash,
            citations=citations,
            abstention_reason=bundle.shortfall_reason,
        )
    return _aggregate_into_record(
        bundle,
        market=market,
        baseline=baseline,
        created_at=created_at,
        question_hash=question_hash,
        citations=citations,
        criteria=criteria,
        counterpoints=counterpoints,
        base_rate_ppm=base_rate_ppm,
        source_notes=source_notes,
        canary_gate=canary_gate,
        verified_count=verified_count,
        min_verified_citations=min_verified_citations,
        calibration_map=calibration_map,
        ledger=ledger,
        provider_gate=provider_gate,
    )
