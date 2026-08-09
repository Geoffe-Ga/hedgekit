"""Project held positions onto the exposure figures the risk caps read (#407).

Two caps in this codebase ran on every tick, reported success, and proved
nothing, because nothing ever fed them an input:

* ``concentration_limits`` (:mod:`windbreak.riskkernel.checks`) compares four
  exposure dimensions against their shares of worst-case equity, and
  :func:`windbreak.scheduler.loop._account_from_verification` handed it four
  zeros. Zero is *permissive* -- ``0 + cost > share`` is false for any sane
  cost -- so the check could not veto for one market or for many.
* The selector's per-bucket notional cap (:mod:`windbreak.selector.sizing`)
  aggregates ``bucket_peers`` against the target's ``correlation_tags``, and
  nothing in ``windbreak/`` had ever constructed a
  :class:`~windbreak.selector.correlation.CorrelationTag`. The loop passed
  ``correlation_tags=()``, so ``effective_buckets(()) == ()`` and
  ``aggregate_bucket_exposure`` returned ``(0, None)`` however many markets a
  tick screened.

This module is the seam that feeds both from evidence the account actually
holds. It is the scheduler's, not the kernel's: SPEC S5 keeps the Risk Kernel
from importing connector types, and the scheduler is the one composition layer
permitted to see both sides -- the same reasoning that puts
:mod:`windbreak.scheduler.eligibility` here.

It is nonetheless **pure**. It takes already-resolved :class:`HeldMarket`
triples rather than connector ``Position`` rows or a live exchange, so it
imports no connector type at all, the caller keeps the single venue read, and
the fail-closed contract can be exercised without a venue.

**Where the exposure comes from.** The account's *held positions*, priced by
the caller. Not the tick's screened candidates: a candidate that never fills
carries no exposure, while a holding does, so holdings are both the stronger
evidence and the figure available today.

**The fail-closed rule.** A market whose bucket the operator has not declared is
unbucketable, and an unbucketable market must not size as though its bucket were
empty -- :func:`project_exposure` returns ``None`` (refuse) rather than a
projection carrying a fabricated zero. That covers the target market *and* every
held peer: a holding nobody has classified could be in the target's bucket, so
while it is held the target's bucket exposure is unprovable. The remedy is one
configuration line, not a softer default.

An *empty* holdings tuple is a different thing entirely and does project. The
venue answered and reported no rows -- an observed absence, which the
connector's contract defines as genuinely flat, exactly as an empty book is
zero depth rather than unknown depth (issue #364). Absent evidence and evidence
of absence are not the same fact, and only the first refuses.

Every figure here is an integer in micros (SPEC S6.1); this module performs only
summation, so there is no division to round.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from windbreak.numeric import MoneyMicros
from windbreak.selector.correlation import (
    BucketExposureEntry,
    CorrelationTag,
    aggregate_bucket_exposure,
    effective_buckets,
)
from windbreak.timekeeping import require_aware

if TYPE_CHECKING:
    from windbreak.config.schema import CorrelationConfig


@dataclass(frozen=True, slots=True)
class HeldMarket:
    """One held position paired with the venue metadata that classifies it.

    The caller resolves these: it reads the venue once for the account's
    positions, prices each, and asks the venue for each ticker's parent event.
    Passing the resolved triple rather than a connector ``Position`` keeps this
    module free of connector imports and lets the fail-closed cases below be
    constructed without a venue.

    Attributes:
        ticker: The held market's own ticker.
        event_ticker: The parent event's ticker, as the venue reports it. This
            is a factual venue field, unlike a correlation bucket, so reading it
            from the venue claims nothing the venue did not say.
        value_micros: The position's mark value, in micros, priced by the
            caller. Non-negative: it is an exposure magnitude, and a signed
            figure would let a long NO cancel a long YES into an understated
            bucket total when both legs and their collateral are live.
    """

    ticker: str
    event_ticker: str
    value_micros: MoneyMicros


@dataclass(frozen=True, slots=True)
class ExposureProjection:
    """The exposure figures a tick's risk caps read, all provable (#407).

    Produced only when every held ticker *and* the target carry a declared
    correlation bucket; see the module docstring for why an undeclared market
    refuses instead.

    Attributes:
        market_exposure: Value held in the target's own ticker, in micros.
        event_exposure: Value held across the target's parent event, in micros.
        bucket_exposure: The largest per-bucket peer total among the target's
            effective buckets, in micros -- the same figure
            :func:`~windbreak.selector.correlation.aggregate_bucket_exposure`
            hands the selector, computed here for the kernel, which has no peer
            aggregation of its own. SPEC S9.9 requires exactly this: "Selector
            and Kernel independently enforce per-bucket caps (defense in
            depth)".
        total_exposure: Value held across every market, in micros.
        bucket_peers: One entry per holding, for the selector to aggregate.
        target_tags: The target market's own declared tags, for the selector to
            resolve via :func:`~windbreak.selector.correlation.effective_buckets`.
    """

    market_exposure: MoneyMicros
    event_exposure: MoneyMicros
    bucket_exposure: MoneyMicros
    total_exposure: MoneyMicros
    bucket_peers: tuple[BucketExposureEntry, ...]
    target_tags: tuple[CorrelationTag, ...]


def _parse_tagged_at(raw: str, ticker: str) -> datetime:
    """Parse an operator-declared ``tagged_at``, refusing a naive value.

    Args:
        raw: The ISO-8601 timestamp the operator declared.
        ticker: The market the declaration belongs to, for the error message.

    Returns:
        The parsed, offset-aware instant.

    Raises:
        ValueError: If ``raw`` is not ISO-8601, or carries no UTC offset --
            refused at this boundary (issue #397) rather than read as
            host-local, which would make a tag's provenance depend on which
            machine loaded the configuration.
    """
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            f"correlation tagged_at for {ticker!r} is not an ISO-8601 instant: {raw!r}"
        ) from exc
    require_aware(moment, f"correlation tagged_at for {ticker!r}")
    return moment


def declared_correlation_tags(
    correlation: CorrelationConfig, ticker: str
) -> tuple[CorrelationTag, ...]:
    """Return the operator-declared correlation tags for one market (S9.9).

    The tags carry ``source="human"``, which is the literal truth rather than a
    convenient label: an operator writing a bucket into the configuration is
    the human that :data:`~windbreak.selector.correlation.TagSource` names, and
    SPEC S9.9 calls tagging "LLM-assisted ... human-overridable, stored as
    data". No third source value is invented; see
    :class:`~windbreak.config.schema.CorrelationTagConfig` for why deriving a
    bucket from venue metadata was rejected.

    Args:
        correlation: The operator's declared assignments.
        ticker: The market whose tags to resolve.

    Returns:
        The market's tags, or ``()`` when the operator has declared none --
        which callers must treat as unbucketable, never as an empty bucket.

    Raises:
        ValueError: If a declared ``tagged_at`` is naive or unparseable, or a
            declared bucket id is outside the seed taxonomy (raised by
            :class:`~windbreak.selector.correlation.CorrelationTag` itself, so
            a typo'd bucket fails loudly instead of silently creating one).
    """
    return tuple(
        CorrelationTag(
            bucket_id=bucket_id,
            source="human",
            tagged_at=_parse_tagged_at(entry.tagged_at, entry.ticker),
        )
        for entry in correlation.tags
        if entry.ticker == ticker
        for bucket_id in entry.bucket_ids
    )


def project_exposure(
    held: tuple[HeldMarket, ...],
    *,
    target_ticker: str,
    target_event_ticker: str,
    correlation: CorrelationConfig,
) -> ExposureProjection | None:
    """Project holdings onto the four exposure terms, or refuse (#407).

    Args:
        held: The account's priced holdings. Empty is evidence of a flat
            account and projects zeros; it is not absent evidence.
        target_ticker: The market being sized this tick.
        target_event_ticker: The target's parent event, as the venue reports it.
        correlation: The operator's declared bucket assignments.

    Returns:
        The projection, or ``None`` when the target or any holding carries no
        declared correlation bucket -- the fail-closed answer, because a
        projection with a zeroed bucket term would let the per-bucket cap pass
        on evidence nobody has.
    """
    target_tags = declared_correlation_tags(correlation, target_ticker)
    if not target_tags:
        return None
    peers: list[BucketExposureEntry] = []
    for holding in held:
        peer_tags = declared_correlation_tags(correlation, holding.ticker)
        if not peer_tags:
            return None
        peers.append(
            BucketExposureEntry(
                market_ticker=holding.ticker,
                exposure_micros=holding.value_micros,
                tags=peer_tags,
            )
        )
    bucket_used, _ = aggregate_bucket_exposure(
        effective_buckets(target_tags), tuple(peers)
    )
    return ExposureProjection(
        market_exposure=MoneyMicros(
            sum(h.value_micros.value for h in held if h.ticker == target_ticker)
        ),
        event_exposure=MoneyMicros(
            sum(
                h.value_micros.value
                for h in held
                if h.event_ticker == target_event_ticker
            )
        ),
        bucket_exposure=MoneyMicros(bucket_used),
        total_exposure=MoneyMicros(sum(h.value_micros.value for h in held)),
        bucket_peers=tuple(peers),
        target_tags=target_tags,
    )
