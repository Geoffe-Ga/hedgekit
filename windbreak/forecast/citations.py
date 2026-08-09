"""SPEC S8.8 citation verification and its structured verdicts.

The forecast engine may only stand a citation behind a live-eligible forecast
once that citation has been *independently re-verified* against its own backing
source: the stored content hash must still match a fresh fetch, the quoted
excerpt (SPEC S8.5) must still be present in that content, any publication date
must be timezone-aware and not in the future, and the source type must be one
of the closed set SPEC S8.8 recognizes. :func:`verify_citation` performs those
checks in a fixed, first-failure order and always returns a
:class:`CitationVerdict` -- an unreachable URL (a denied egress or a transport
``OSError``) is an *unverified* result, never an uncaught exception, so a dead
source can never silently pass as verified.

Every fetch goes through the egress-guarded :meth:`ResearchTools.fetch`
capability (SPEC S8.3), never a raw transport, so citation verification can
never sidestep the sandbox's allowlist. This module never touches a float: it
sits on the probability/money path guarded by ``scripts/lint_no_floats.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from windbreak.forecast.sandbox import EgressDeniedError

if TYPE_CHECKING:
    from datetime import datetime

    from windbreak.forecast.records import Citation
    from windbreak.forecast.sandbox import ResearchTools

#: Failure code: the citation's URL could not be fetched (egress denied or a
#: transport error). An unreachable source is unverified, never an exception.
FAILURE_UNREACHABLE: Final = "unreachable"

#: Failure code: a fresh fetch's content hash disagrees with the stored one, so
#: the citation no longer provenances the content it claims.
FAILURE_CONTENT_HASH_MISMATCH: Final = "content_hash_mismatch"

#: Failure code: the quoted excerpt (SPEC S8.5) is absent from -- or empty
#: against -- the refetched content.
FAILURE_QUOTE_NOT_FOUND: Final = "quote_not_found"

#: Failure code: the publication date is naive (timezone-less) or later than
#: the verification instant, either of which makes it untrustworthy.
FAILURE_PUBLICATION_DATE_INVALID: Final = "publication_date_invalid"

#: Failure code: the source type is not one of the SPEC S8.8 recognized kinds.
FAILURE_UNKNOWN_SOURCE_TYPE: Final = "unknown_source_type"

#: The closed set of source types SPEC S8.8 will verify (SPEC S8.8).
KNOWN_SOURCE_TYPES: frozenset[str] = frozenset(
    {"news_article", "research_note", "primary_source"}
)


def content_hash_of(content: str) -> str:
    """Return a namespaced sha256 content hash for citation provenance.

    The single canonical content-hash implementation for the forecast package;
    ``pipeline.bounded_web_research`` reuses it so a citation's stored hash and
    the hash recomputed here during verification are byte-for-byte comparable.

    Args:
        content: The content to hash.

    Returns:
        A ``sha256:``-prefixed, 64-character lowercase hex digest.
    """
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CitationVerdict:
    """The outcome of verifying a single citation (SPEC S8.8).

    Attributes:
        citation: The citation that was verified.
        verified: Whether the citation passed every check.
        failure: The first failing check's code, or ``None`` when verified.
            ``verified`` is exactly ``failure is None`` -- the two fields can
            never disagree.
    """

    citation: Citation
    verified: bool
    failure: str | None

    def __post_init__(self) -> None:
        """Enforce that ``verified`` is exactly ``failure is None``.

        Raises:
            ValueError: If ``verified`` is ``True`` with a non-``None``
                ``failure``, or ``False`` with a ``None`` ``failure`` -- an
                internally inconsistent verdict.
        """
        if self.verified == (self.failure is not None):
            raise ValueError(
                "verified must be exactly (failure is None); got "
                f"verified={self.verified!r}, failure={self.failure!r}"
            )


def _publication_date_invalid(
    publication_date: datetime | None, as_of: datetime
) -> bool:
    """Return whether a citation's publication date fails the S8.8 date check.

    A ``None`` date is valid (the "where available" clause): there is nothing
    to distrust. A present date is invalid when it is naive (offsetless, so not
    comparable to the aware ``as_of``) or later than ``as_of`` (a source cannot
    postdate the verification instant).

    Args:
        publication_date: The citation's publication date, or ``None``.
        as_of: The verification instant (timezone-aware).

    Returns:
        ``True`` if the date is present and invalid, else ``False``.
    """
    if publication_date is None:
        return False
    # Naive means `tzinfo is None` *or* a `tzinfo` whose `utcoffset()` returns
    # None -- Python's own definition. Testing `tzinfo is None` alone (issue
    # #346) let the second shape fall through to the comparison below and raise
    # `TypeError: can't compare offset-naive and offset-aware datetimes`,
    # aborting the whole S8.8 verification pass on precisely the input this
    # check exists to reject. `utcoffset()` covers both shapes in one call: it
    # already returns None when `tzinfo` is None. Rejecting -- rather than
    # assuming an offset -- matches `windbreak/forecast/pubdate.py`, which
    # degrades an offsetless date instead of fabricating a timezone for it.
    if publication_date.utcoffset() is None:
        return True
    return publication_date > as_of


def _first_failure(content: str, citation: Citation, as_of: datetime) -> str | None:
    """Return the first failing check's code for a fetched citation, or ``None``.

    The checks run in the fixed SPEC S8.8 order -- content hash, quote presence,
    publication date, source type -- and the first that fails wins, so a
    citation bad on several dimensions reports the earliest one deterministically.

    Args:
        content: The freshly fetched content backing the citation.
        citation: The citation being verified.
        as_of: The verification instant (timezone-aware).

    Returns:
        The first failure code, or ``None`` if every check passes.
    """
    if content_hash_of(content) != citation.content_hash:
        return FAILURE_CONTENT_HASH_MISMATCH
    if not citation.quoted_text or citation.quoted_text not in content:
        return FAILURE_QUOTE_NOT_FOUND
    if _publication_date_invalid(citation.publication_date, as_of):
        return FAILURE_PUBLICATION_DATE_INVALID
    if citation.source_type not in KNOWN_SOURCE_TYPES:
        return FAILURE_UNKNOWN_SOURCE_TYPE
    return None


def verify_citation(
    tools: ResearchTools, citation: Citation, *, as_of: datetime
) -> CitationVerdict:
    """Verify one citation against a fresh fetch of its source (SPEC S8.8).

    The citation's URL is refetched through the egress-guarded
    :meth:`ResearchTools.fetch`; a denied egress or a transport ``OSError`` is
    caught and reported as :data:`FAILURE_UNREACHABLE` (an unverified verdict,
    never a raised exception -- the egress boundary is never silently
    bypassed). On a successful fetch, the S8.8 checks run in fixed first-failure
    order via :func:`_first_failure`.

    Args:
        tools: The sandboxed research tools whose ``fetch`` capability is used.
        citation: The citation to verify.
        as_of: The verification instant, for the publication-date check
            (keyword-only, timezone-aware).

    Returns:
        The citation's verdict.
    """
    try:
        content = tools.fetch(citation.url)
    except (EgressDeniedError, OSError):
        return CitationVerdict(citation, verified=False, failure=FAILURE_UNREACHABLE)
    failure = _first_failure(content, citation, as_of)
    return CitationVerdict(citation, verified=failure is None, failure=failure)


def verify_citations(
    tools: ResearchTools, citations: tuple[Citation, ...], *, as_of: datetime
) -> tuple[CitationVerdict, ...]:
    """Verify each citation in a tuple, preserving order (SPEC S8.8).

    Args:
        tools: The sandboxed research tools whose ``fetch`` capability is used.
        citations: The citations to verify, in order.
        as_of: The verification instant (keyword-only, timezone-aware).

    Returns:
        One verdict per citation, in the same order.
    """
    return tuple(
        verify_citation(tools, citation, as_of=as_of) for citation in citations
    )


def count_verified(verdicts: tuple[CitationVerdict, ...]) -> int:
    """Count how many verdicts verified (SPEC S8.8).

    Args:
        verdicts: The verdicts to tally.

    Returns:
        The number of verdicts whose ``verified`` flag is ``True``.
    """
    return sum(1 for verdict in verdicts if verdict.verified)
