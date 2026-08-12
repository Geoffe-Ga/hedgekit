"""SPEC S6.3 forecast record schema and its JSON-safe projection.

Every model here is a frozen, slotted dataclass describing one facet of a
produced forecast: the ensemble :class:`ModelVote`s, their supporting
:class:`Citation`s, the :class:`BaselineQuoteSnapshot` the forecast was struck
against, and the immutable :class:`ForecastRecord` that ties them together.
Probability-bearing quantities are carried as bare parts-per-million integers
(``probability_ppm`` etc.) -- never floats -- so this package sits on the
probability/money path guarded by ``scripts/lint_no_floats.py``.

:class:`ForecastRecord` validates its range, integrality, non-emptiness, and
closed-set invariants in ``__post_init__`` so a malformed record fails loudly
at construction. Since issue #525 that includes screening ``market_ticker``
through the same untrusted-text screen the provider seam refuses a market on:
the record is what the scheduler reads a ``ForecastCreated`` row's ticker off,
and that row lands in an append-only chain, so the invariant belongs on the type
rather than at each append. :func:`forecast_record_to_payload` renders a record into a
JSON-safe mapping (datetimes as ISO-8601 ``Z`` strings, tuples of dataclasses
as lists of dicts, no float leaf anywhere) for ledger/event emission.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from datetime import datetime
from typing import TYPE_CHECKING

from windbreak.forecast.sanitize import screen_single_line_text
from windbreak.timekeeping import iso_z, require_aware

if TYPE_CHECKING:
    from typing import Literal

#: Lowest legal parts-per-million probability (0.0 probability).
_MIN_PPM = 0

#: Highest legal parts-per-million probability (1.0 probability).
_MAX_PPM = 1_000_000

#: The triage-only stage tag (SPEC S8.4): a record backed solely by the cheap
#: Stage-0 prior, never by the full pipeline's research. Named here because it
#: participates in more than one ``__post_init__`` invariant (the closed-set
#: membership check and the live-eligibility guard below).
_TRIAGE_ONLY_STAGE = "triage_only"

#: The full-pipeline stage tag (SPEC S8.4).
_FULL_STAGE = "full"

#: The closed set of triage stages a forecast record may carry (SPEC S8.4).
_TRIAGE_STAGES: frozenset[str] = frozenset({_TRIAGE_ONLY_STAGE, _FULL_STAGE})

#: The ppm-domain fields of :class:`ForecastRecord` sharing one range rule.
_PPM_FIELDS: tuple[str, ...] = ("probability_ppm", "ci_low_ppm", "ci_high_ppm")


def _require_ppm(value: int, field_name: str) -> None:
    """Guard that a ppm field is a true integer within ``[0, 1_000_000]``.

    The bool/int convention mirrors :func:`windbreak.connector.models`'s unit
    guards: a stray ``bool`` (an ``int`` subclass) must never masquerade as a
    probability, so it is rejected before the range check.

    Args:
        value: The candidate parts-per-million integer.
        field_name: The owning field's name, surfaced in the error message.

    Raises:
        TypeError: If ``value`` is a ``bool`` or is not an ``int``.
        ValueError: If ``value`` is outside ``[0, 1_000_000]``. The message
            names the offending ``field_name``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be a non-bool int, got {type(value).__name__}"
        )
    if not _MIN_PPM <= value <= _MAX_PPM:
        raise ValueError(
            f"{field_name} must be within [{_MIN_PPM}, {_MAX_PPM}], got {value}"
        )


def _require_positive_unit_int(value: int, field_name: str) -> None:
    """Guard that a plain-int unit field is a true, strictly positive integer.

    Args:
        value: The candidate integer.
        field_name: The owning field's name, surfaced in the error message.

    Raises:
        TypeError: If ``value`` is a ``bool`` or is not an ``int``.
        ValueError: If ``value`` is not strictly positive.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be a non-bool int, got {type(value).__name__}"
        )
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


def _require_non_empty(value: str, field_name: str) -> None:
    """Reject an empty string identifier.

    Args:
        value: The candidate identifier.
        field_name: The owning field's name, surfaced in the error message.

    Raises:
        ValueError: If ``value`` is empty. The message names ``field_name``.
    """
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _require_screened_ticker(value: str) -> None:
    """Refuse a ``market_ticker`` the provider seam would refuse (issue #525).

    A market's ticker is upstream-sourced and attacker-influenced;
    :class:`~windbreak.connector.models.NormalizedMarket` validates enum-shaped
    and numeric fields only, so nothing between the exchange and here screens it.
    A :class:`ForecastRecord` is what the scheduler's composition root reads
    ``market_ticker`` off to build the ``ForecastCreated`` row it appends to an
    **append-only** hash chain -- and nothing written there can ever be redacted.

    Screening at that append would be coverage by *convention*: one deletable
    call, re-added by hand for every future consumer of a record. Screening at
    construction makes it coverage by *construction* -- no record carrying an
    unscreened ticker exists for a consumer to leak, which is the same argument
    that moved the metadata screen to the provider seam in issue #462.

    The screen is :func:`~windbreak.forecast.sanitize.screen_single_line_text`,
    the identical function :func:`
    ~windbreak.forecast.providers.base.screen_market_metadata` refuses the
    market's ticker with. Wiring a narrower check here would admit exactly the
    artifacts the seam already knows are hostile.

    The refusal names the verdict and the field's sha256 digest, never the bytes:
    an exception message travels into logs, and repeating the ticker there would
    defeat keeping it off the chain.

    Args:
        value: The candidate market ticker.

    Raises:
        ValueError: If the ticker forges an untrusted-data delimiter, embeds a
            tool-call lure, or carries a line terminator.
    """
    verdict = screen_single_line_text(value)
    if verdict is None:
        return
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    raise ValueError(
        f"market_ticker: {verdict} (sha256:{digest}); a ticker the market"
        " metadata screen refuses is refused here too, never recorded"
    )


def _require_non_negative_count(value: int, field_name: str) -> None:
    """Guard that a count field is a true, non-negative integer.

    Mirrors :func:`_require_ppm`'s bool/int convention: a stray ``bool`` (an
    ``int`` subclass) must never masquerade as a citation count, so it is
    rejected before the range check.

    Args:
        value: The candidate count.
        field_name: The owning field's name, surfaced in the error message.

    Raises:
        TypeError: If ``value`` is a ``bool`` or is not an ``int``.
        ValueError: If ``value`` is negative.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be a non-bool int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")


def is_live_eligible(
    *,
    verified_citation_count: int,
    min_verified_citations: int,
    triage_stage: str,
    coherence_flag: bool,
    abstention_reason: str | None,
) -> bool:
    """Decide whether a forecast may back a live order (SPEC S8.8).

    Live eligibility requires enough independently verified citations *and* the
    absence of every ineligibility trigger. Four independent conditions each
    force ineligibility: a triage-only stage (never backed by full research), a
    raised coherence flag (an incoherent forecast), a set abstention reason (the
    engine declined to stand behind the estimate), or a verified-citation count
    below the configured minimum.

    Args:
        verified_citation_count: How many citations independently verified.
        min_verified_citations: The minimum verified citations required.
        triage_stage: The record's triage stage (``"triage_only"`` or
            ``"full"``).
        coherence_flag: Whether the forecast was flagged incoherent.
        abstention_reason: Why the engine abstained, or ``None``.

    Returns:
        ``True`` if the forecast is live-eligible, else ``False``.

    Raises:
        TypeError: If either count field is a ``bool`` or non-``int``.
        ValueError: If either count field is negative.
    """
    _require_non_negative_count(verified_citation_count, "verified_citation_count")
    _require_non_negative_count(min_verified_citations, "min_verified_citations")
    ineligible = (
        triage_stage == _TRIAGE_ONLY_STAGE
        or coherence_flag
        or abstention_reason is not None
        or verified_citation_count < min_verified_citations
    )
    return not ineligible


@dataclass(frozen=True, slots=True)
class ModelVote:
    """One ensemble member's structured probability vote (SPEC S6.3 / S8.6).

    Attributes:
        provider: The LLM provider that produced the vote.
        model_version: The pinned model version string.
        declared_training_cutoff: The model's self-declared training cutoff.
        probability_ppm: The member's probability estimate, in ppm.
        response_fingerprint: A content fingerprint of the raw response, for
            silent-drift detection (T14).
    """

    provider: str
    model_version: str
    declared_training_cutoff: str
    probability_ppm: int
    response_fingerprint: str

    def __post_init__(self) -> None:
        """Validate the ppm range and integrality invariant.

        Raises:
            TypeError: If ``probability_ppm`` is a ``bool`` or non-``int``.
            ValueError: If ``probability_ppm`` is outside ``[0, 1_000_000]``.
        """
        _require_ppm(self.probability_ppm, "probability_ppm")


@dataclass(frozen=True, slots=True)
class Citation:
    """A single verified source backing a forecast (SPEC S6.3 / S8.8).

    Attributes:
        url: The source URL.
        content_hash: Hash of the retrieved content, for provenance.
        quoted_text: The short quoted excerpt (<= 25 words per S8.5).
        publication_date: The source's publication date, or None when unknown.
        source_type: The source's category (e.g. ``news_article``).
    """

    url: str
    content_hash: str
    quoted_text: str
    publication_date: datetime | None
    source_type: str


@dataclass(frozen=True, slots=True)
class BaselineQuoteSnapshot:
    """The executable-price baseline a forecast is struck against (SPEC S8.1).

    Attributes:
        snapshot_id: The snapshot's unique identifier.
        price_pips: The baseline executable price, in pips (a positive int).
        fetched_at: When the baseline snapshot was taken.
    """

    snapshot_id: str
    price_pips: int
    fetched_at: datetime

    def __post_init__(self) -> None:
        """Validate the price positivity and identifier invariants.

        Raises:
            TypeError: If ``price_pips`` is a ``bool`` or non-``int``.
            ValueError: If ``price_pips`` is non-positive, ``snapshot_id`` is
                empty, or ``fetched_at`` carries no UTC offset (issue #397).
        """
        _require_positive_unit_int(self.price_pips, "price_pips")
        _require_non_empty(self.snapshot_id, "snapshot_id")
        require_aware(self.fetched_at, "fetched_at")


@dataclass(frozen=True, slots=True)
class ForecastRecord:
    """An immutable forecast produced by the engine (SPEC S6.3).

    Records are never mutated after creation; calibration produces derived
    records rather than editing originals.

    Attributes:
        forecast_id: The record's unique identifier.
        market_ticker: The forecasted market's ticker.
        normalized_question_hash: Hash of the normalized question text.
        probability_ppm: The aggregated probability estimate, in ppm.
        ci_low_ppm: Lower confidence bound, in ppm.
        ci_high_ppm: Upper confidence bound, in ppm.
        model_votes: The individual ensemble votes.
        vote_dispersion_ppm: Spread of the votes, feeding sizing (S9.6).
        rationale_markdown: Human-readable rationale.
        citations: The verified supporting citations.
        source_quality_notes: Free-form notes on source quality.
        research_cost_micros: Total research cost, in micros.
        triage_stage: Whether the record is triage-only or full.
        created_at: When the forecast was created.
        forecast_horizon_hours: Hours until the forecast's horizon.
        market_price_baseline_pips: The baseline executable price, in pips.
        baseline_quote_snapshot_id: The baseline snapshot's identifier.
        coherence_group_sum_ppm: Group probability sum, or None when the
            market stands alone (S8.7).
        coherence_flag: Whether the forecast was flagged incoherent.
        abstention_reason: Why the engine abstained, or None.
        eligible_for_live: Whether the record may back a live order.
    """

    forecast_id: str
    market_ticker: str
    normalized_question_hash: str
    probability_ppm: int
    ci_low_ppm: int
    ci_high_ppm: int
    model_votes: tuple[ModelVote, ...]
    vote_dispersion_ppm: int
    rationale_markdown: str
    citations: tuple[Citation, ...]
    source_quality_notes: tuple[str, ...]
    research_cost_micros: int
    triage_stage: Literal["triage_only", "full"]
    created_at: datetime
    forecast_horizon_hours: int
    market_price_baseline_pips: int
    baseline_quote_snapshot_id: str
    coherence_group_sum_ppm: int | None
    coherence_flag: bool
    abstention_reason: str | None
    eligible_for_live: bool

    def __post_init__(self) -> None:
        """Validate the ppm ranges, identifiers, and triage invariants.

        Raises:
            TypeError: If any ppm field is a ``bool`` or non-``int``.
            ValueError: If any ppm field is out of range, ``forecast_id`` or
                ``market_ticker`` is empty, ``market_ticker`` fails the
                market-metadata screen (see :func:`_require_screened_ticker`,
                issue #525), ``created_at`` carries no UTC offset
                (issue #397), ``triage_stage`` is unrecognized,
                or the record is ``eligible_for_live`` while any
                live-ineligibility trigger holds. Three independent triggers each
                force live-ineligibility: ``triage_stage == "triage_only"`` (a
                triage-only record was never backed by the full pipeline's
                research), ``coherence_flag`` being ``True`` (an incoherent
                forecast must never back a live order), and ``abstention_reason``
                being non-``None`` (the engine declined to stand behind the
                estimate). Each message names the offending field(s).
        """
        for field_name in _PPM_FIELDS:
            _require_ppm(getattr(self, field_name), field_name)
        _require_non_empty(self.forecast_id, "forecast_id")
        _require_non_empty(self.market_ticker, "market_ticker")
        _require_screened_ticker(self.market_ticker)
        require_aware(self.created_at, "created_at")
        if self.triage_stage not in _TRIAGE_STAGES:
            allowed = ", ".join(sorted(_TRIAGE_STAGES))
            raise ValueError(
                f"triage_stage must be one of {{{allowed}}}, got {self.triage_stage!r}"
            )
        if self.triage_stage == _TRIAGE_ONLY_STAGE and self.eligible_for_live:
            raise ValueError(
                f"triage_stage={self.triage_stage!r} records are permanently "
                "live-ineligible; eligible_for_live must be False"
            )
        if self.coherence_flag and self.eligible_for_live:
            raise ValueError(
                "eligible_for_live must be False when coherence_flag is True"
            )
        if self.abstention_reason is not None and self.eligible_for_live:
            raise ValueError(
                "eligible_for_live must be False when abstention_reason is set"
            )


def _jsonable(value: object) -> object:
    """Convert one record value into a JSON-safe node, recursively.

    Datetimes become ISO-8601 ``Z`` strings; tuples become lists; nested
    :class:`ModelVote` / :class:`Citation` dataclasses become dicts of their
    own projected fields; every other value (str, int, bool, None) is already
    JSON-safe and passes through unchanged. No float is ever produced.

    Args:
        value: The raw value to project.

    Returns:
        The JSON-safe projection of ``value``.
    """
    if isinstance(value, datetime):
        return iso_z(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, ModelVote | Citation):
        return {
            field_def.name: _jsonable(getattr(value, field_def.name))
            for field_def in fields(value)
        }
    return value


def forecast_record_to_payload(record: ForecastRecord) -> dict[str, object]:
    """Project a forecast record into a JSON-safe mapping by field name.

    The mapping is stable and lossless: keys are the dataclass field names
    verbatim, datetimes are ISO-8601 ``Z`` strings, tuple-of-dataclass fields
    become lists of dicts, ``None`` stays ``None``, and every ppm integer
    remains an int -- there is never a float leaf anywhere.

    Args:
        record: The forecast record to project.

    Returns:
        A JSON-serializable mapping of every field of ``record``.
    """
    return {
        field_def.name: _jsonable(getattr(record, field_def.name))
        for field_def in fields(record)
    }
