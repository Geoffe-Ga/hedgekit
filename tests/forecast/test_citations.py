"""Tests for windbreak.forecast.citations (issue #26): citation verification.

Pins `verify_citation`'s SPEC S8.5/S8.8 contract: fetch a citation's URL
through the egress-guarded `ResearchTools.fetch` capability (never a raw
transport), and check -- in a FIXED first-failure order -- content-hash
integrity, quote presence, publication-date validity (where available), and
source-type membership in the known set. An unreachable URL (a raised
`EgressDeniedError` or `OSError`) is an *unverified* result, never an
uncaught exception: `verify_citation` always returns a `CitationVerdict`.
`windbreak/forecast/citations.py` does not exist yet, so importing it below
fails collection with `ModuleNotFoundError: No module named
'windbreak.forecast.citations'` -- the expected Gate 1 RED state for
issue #26.

Fixture-construction choice (`_StaticFetchTransport`, `_citation`)
    Each test builds its own `Citation` directly (mirroring
    `test_records.py`'s `_citation(**overrides)` pattern) over a
    `research_tools_factory`-built `ResearchTools` wired to a small, local
    `_StaticFetchTransport` returning one fixed content string for any URL
    (mirroring `test_sandbox.py`'s own local double of the same name -- each
    test module defines its own rather than reaching across files, per this
    package's established convention; see `tests/forecast/conftest.py`'s
    module docstring). This gives each test full, independent control of
    exactly one failing dimension at a time (mutation hardening): flip only
    the field the test targets, leaving every other field green, so no single
    mutant in `verify_citation`'s check order can survive undetected.
"""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.citations import (
    FAILURE_CONTENT_HASH_MISMATCH,
    FAILURE_PUBLICATION_DATE_INVALID,
    FAILURE_QUOTE_NOT_FOUND,
    FAILURE_UNKNOWN_SOURCE_TYPE,
    FAILURE_UNREACHABLE,
    KNOWN_SOURCE_TYPES,
    CitationVerdict,
    content_hash_of,
    count_verified,
    verify_citation,
    verify_citations,
)
from windbreak.forecast.records import Citation

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from windbreak.forecast.sandbox import ResearchTools

    ResearchToolsFactory = Callable[..., ResearchTools]
    RaisingFetchTransportFactory = Callable[[], object]

#: The single host every `research_tools_factory`-built tools instance
#: allowlists by default (see `tests/forecast/conftest.py`'s
#: `_DEFAULT_ALLOWED_HOST`); reproduced here as a literal, matching
#: `test_sandbox.py`'s own convention of not reaching into conftest's private
#: constant.
_ALLOWED_HOST = "research.local"

#: Content returned by every green-path `_StaticFetchTransport` fixture below.
_GREEN_CONTENT = "The Federal Reserve raised interest rates in December 2024."

#: A quote that is a genuine substring of `_GREEN_CONTENT`.
_GREEN_QUOTE = "raised interest rates"

_VALID_CITATION_KWARGS: dict[str, object] = {
    "url": f"https://{_ALLOWED_HOST}/article",
    "content_hash": content_hash_of(_GREEN_CONTENT),
    "quoted_text": _GREEN_QUOTE,
    "publication_date": datetime(2024, 11, 1, tzinfo=UTC),
    "source_type": "news_article",
}


def _citation(**overrides: object) -> Citation:
    """Build a `Citation` from the green-path defaults, with overrides.

    Args:
        **overrides: Field overrides layered on `_VALID_CITATION_KWARGS`.

    Returns:
        A constructed `Citation`.
    """
    return Citation(**{**_VALID_CITATION_KWARGS, **overrides})


class _OffsetlessTimezone(tzinfo):
    """A `tzinfo` whose `utcoffset` is `None`, so its datetimes are still naive.

    Python defines a datetime as naive when `tzinfo is None` *or* when
    `tzinfo.utcoffset(dt)` returns `None`. Only a hand-written subclass
    produces that second shape -- neither `datetime.fromisoformat` nor
    `email.utils.parsedate_to_datetime` ever yields one -- which is exactly why
    it needs a double here (issue #346). Defined locally rather than shared,
    per this package's convention (see the module docstring).

    Only `utcoffset` is overridden: it is the single method the naive-ness test
    and the `>` comparison consult, so supplying `tzname`/`dst` would add
    behaviour no test exercises.
    """

    def utcoffset(self, dt: datetime | None) -> None:
        """Return `None` -- this zone declines to define an offset.

        Args:
            dt: The datetime being interrogated; unused, since the answer is
                unconditional.

        Returns:
            `None`, always.
        """
        return None


class _StaticFetchTransport:
    """A `FetchTransport` returning one fixed content string for any URL.

    Mirrors `test_sandbox.py`'s local double of the same name.
    """

    def __init__(self, content: str = _GREEN_CONTENT) -> None:
        """Store the fixed content every `fetch` call will return.

        Args:
            content: The content string to return for any URL.
        """
        self._content = content

    def fetch(self, url: str) -> str:
        """Return the fixed content, ignoring `url`.

        Args:
            url: The (unused) URL being fetched.

        Returns:
            `self._content`, verbatim.
        """
        return self._content


def _tools(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    *,
    fetch_transport: object | None = None,
) -> ResearchTools:
    """Build a `ResearchTools` over a fixed-content fetch transport.

    Args:
        research_tools_factory: The conftest-provided tools factory.
        tmp_path: The pytest-provided temporary cache directory.
        fetch_transport: The `FetchTransport` to inject; defaults to a
            `_StaticFetchTransport` returning `_GREEN_CONTENT`.

    Returns:
        A `ResearchTools` sandboxed to `_ALLOWED_HOST`.
    """
    return research_tools_factory(
        cache_dir=tmp_path,
        allowed_hosts=frozenset({_ALLOWED_HOST}),
        fetch_transport=fetch_transport or _StaticFetchTransport(),
    )


# --- content_hash_of ---------------------------------------------------------------


def test_content_hash_of_has_sha256_prefix_and_64_hex_chars() -> None:
    """`content_hash_of` returns a `sha256:`-prefixed, 64-char lowercase hex digest."""
    result = content_hash_of("hello world")

    assert result.startswith("sha256:")
    digest = result.removeprefix("sha256:")
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_content_hash_of_is_deterministic() -> None:
    """Hashing the same content twice yields identical hashes."""
    assert content_hash_of("same content") == content_hash_of("same content")


def test_content_hash_of_distinct_content_yields_distinct_hash() -> None:
    """Hashing two different strings yields two different hashes."""
    assert content_hash_of("content a") != content_hash_of("content b")


# --- verify_citation: the all-green path --------------------------------------------


def test_all_green_citation_is_verified(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """A citation whose hash, quote, date, and source_type are all valid verifies."""
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation()

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is True
    assert verdict.failure is None


def test_verifying_same_citation_twice_yields_equal_verdicts(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """Verification is deterministic: re-verifying yields an equal verdict."""
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation()

    verdict_a = verify_citation(tools, citation, as_of=created_at)
    verdict_b = verify_citation(tools, citation, as_of=created_at)

    assert verdict_a == verdict_b


# --- verify_citation: unreachable (fetch failure is not an error) ------------------


def test_dead_url_yields_unreachable_failure_without_raising(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
    make_raising_fetch_transport: RaisingFetchTransportFactory,
) -> None:
    """A dead URL (fetch raises `ConnectionError`) yields an unverified verdict.

    No exception escapes `verify_citation` -- the "unverified, not an error"
    contract.
    """
    tools = _tools(
        research_tools_factory,
        tmp_path,
        fetch_transport=make_raising_fetch_transport(),
    )
    citation = _citation()

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is False
    assert verdict.failure == FAILURE_UNREACHABLE


def test_off_allowlist_url_yields_unreachable_via_egress_denied(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """A URL off the tools' egress allowlist is unreachable, never bypassed.

    `tools.fetch` raises `EgressDeniedError` for an off-allowlist host; that
    must surface as an unverified verdict, proving the egress boundary is
    never sidestepped by citation verification.
    """
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(url="https://evil.example/article")

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is False
    assert verdict.failure == FAILURE_UNREACHABLE


def test_unreachable_and_bad_quote_yields_unreachable_first(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
    make_raising_fetch_transport: RaisingFetchTransportFactory,
) -> None:
    """An unreachable URL takes precedence over an also-bad quote.

    Fetch failure happens before any content-derived check can even run, so
    the fixed check order pins `FAILURE_UNREACHABLE` regardless of how bad the
    citation's other fields are.
    """
    tools = _tools(
        research_tools_factory,
        tmp_path,
        fetch_transport=make_raising_fetch_transport(),
    )
    citation = _citation(quoted_text="this text can never be checked")

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.failure == FAILURE_UNREACHABLE


# --- verify_citation: content-hash mismatch -----------------------------------------


def test_content_hash_mismatch_yields_content_hash_mismatch_failure(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """A citation whose stored hash disagrees with the refetched content fails."""
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(content_hash=content_hash_of("a different document entirely"))

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is False
    assert verdict.failure == FAILURE_CONTENT_HASH_MISMATCH


def test_hash_mismatch_takes_precedence_over_bad_quote_and_bad_date(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """A citation that is simultaneously hash-mismatched, quote-absent, and
    date-invalid (naive AND future) still fails on the hash first -- the
    FIXED check-order contract.
    """
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(
        content_hash=content_hash_of("a different document entirely"),
        quoted_text="not present anywhere in the fetched content",
        publication_date=datetime(2099, 1, 1),  # naive AND after as_of
    )

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.failure == FAILURE_CONTENT_HASH_MISMATCH


# --- verify_citation: quote presence -------------------------------------------------


def test_quote_not_found_yields_quote_not_found_failure(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """A quote that is not a substring of the fetched content fails to verify.

    The issue's canonical example.
    """
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(quoted_text="this exact phrase is not in the content")

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is False
    assert verdict.failure == FAILURE_QUOTE_NOT_FOUND


def test_empty_quoted_text_yields_quote_not_found_failure(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """An empty `quoted_text` is treated the same as an absent quote."""
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(quoted_text="")

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is False
    assert verdict.failure == FAILURE_QUOTE_NOT_FOUND


# --- verify_citation: publication date ("where available") -------------------------


def test_none_publication_date_skips_the_check_and_still_verifies(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """A `None` publication date skips the date check entirely (per "where
    available"); a citation with every other field green still verifies.
    """
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(publication_date=None)

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is True
    assert verdict.failure is None


def test_future_publication_date_yields_publication_date_invalid_failure(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """A publication date after `as_of` fails the date-validity check."""
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(publication_date=datetime(2099, 1, 1, tzinfo=UTC))

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is False
    assert verdict.failure == FAILURE_PUBLICATION_DATE_INVALID


def test_naive_publication_date_yields_publication_date_invalid_failure(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """A naive (tz-less) publication date fails the date-validity check, even
    when its calendar value would otherwise precede `as_of`.
    """
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(publication_date=datetime(2024, 11, 1))

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is False
    assert verdict.failure == FAILURE_PUBLICATION_DATE_INVALID


def test_offsetless_tzinfo_publication_date_is_rejected_not_raised(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """A publication date carrying a `tzinfo` whose `utcoffset()` is `None` is
    naive by Python's own definition, so the date check *rejects* it (issue
    #346) rather than raising `TypeError` from the `>` against the aware
    `as_of` -- which would abort the whole verification pass on the exact input
    the check exists to catch.

    The calendar value deliberately precedes `as_of`, so the naive-ness test is
    the only thing that can fail this citation: were the guard to consult a
    fabricated offset instead of rejecting, the date would read as valid and
    the assertions below would flip rather than merely stop raising.
    """
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(
        publication_date=datetime(2024, 11, 1, tzinfo=_OffsetlessTimezone())
    )

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is False
    assert verdict.failure == FAILURE_PUBLICATION_DATE_INVALID


# --- verify_citation: source_type ----------------------------------------------------


def test_unknown_source_type_yields_unknown_source_type_failure(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """A `source_type` outside `KNOWN_SOURCE_TYPES` fails the final check."""
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(source_type="anonymous_blog_post")

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is False
    assert verdict.failure == FAILURE_UNKNOWN_SOURCE_TYPE


@pytest.mark.parametrize(
    "source_type", ["news_article", "research_note", "primary_source"]
)
def test_each_known_source_type_is_accepted(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
    source_type: str,
) -> None:
    """Every member of `KNOWN_SOURCE_TYPES` is individually accepted."""
    tools = _tools(research_tools_factory, tmp_path)
    citation = _citation(source_type=source_type)

    verdict = verify_citation(tools, citation, as_of=created_at)

    assert verdict.verified is True


def test_known_source_types_is_the_expected_closed_set() -> None:
    """`KNOWN_SOURCE_TYPES` is exactly the three SPEC-named source kinds."""
    assert (
        frozenset({"news_article", "research_note", "primary_source"})
        == KNOWN_SOURCE_TYPES
    )


# --- verify_citations / count_verified: batch behavior ------------------------------


def test_verify_citations_returns_one_verdict_per_citation_in_order(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """`verify_citations` maps each citation to a verdict, order preserved."""
    tools = _tools(research_tools_factory, tmp_path)
    green = _citation(url=f"https://{_ALLOWED_HOST}/green")
    bad_hash = _citation(
        url=f"https://{_ALLOWED_HOST}/bad-hash",
        content_hash=content_hash_of("mismatched content"),
    )

    verdicts = verify_citations(tools, (green, bad_hash), as_of=created_at)

    assert len(verdicts) == 2
    assert verdicts[0].citation == green
    assert verdicts[0].verified is True
    assert verdicts[1].citation == bad_hash
    assert verdicts[1].failure == FAILURE_CONTENT_HASH_MISMATCH


def test_verify_citations_on_empty_tuple_returns_empty_tuple(
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
    created_at: datetime,
) -> None:
    """Verifying zero citations returns zero verdicts."""
    tools = _tools(research_tools_factory, tmp_path)

    verdicts = verify_citations(tools, (), as_of=created_at)

    assert verdicts == ()


def test_count_verified_of_no_verified_verdicts_is_zero() -> None:
    """`count_verified` returns 0 when every verdict is unverified."""
    citation = _citation()
    verdicts = (
        CitationVerdict(citation=citation, verified=False, failure=FAILURE_UNREACHABLE),
        CitationVerdict(
            citation=citation, verified=False, failure=FAILURE_QUOTE_NOT_FOUND
        ),
    )

    assert count_verified(verdicts) == 0


def test_count_verified_of_some_verified_verdicts_is_the_exact_count() -> None:
    """`count_verified` counts only the verified verdicts, not the total."""
    citation = _citation()
    verdicts = (
        CitationVerdict(citation=citation, verified=True, failure=None),
        CitationVerdict(citation=citation, verified=False, failure=FAILURE_UNREACHABLE),
        CitationVerdict(citation=citation, verified=True, failure=None),
    )

    assert count_verified(verdicts) == 2


def test_count_verified_of_all_verified_verdicts_is_the_full_length() -> None:
    """`count_verified` returns the full length when every verdict verifies."""
    citation = _citation()
    verdicts = (
        CitationVerdict(citation=citation, verified=True, failure=None),
        CitationVerdict(citation=citation, verified=True, failure=None),
    )

    assert count_verified(verdicts) == 2


def test_count_verified_of_empty_tuple_is_zero() -> None:
    """`count_verified` of an empty tuple is 0."""
    assert count_verified(()) == 0


# --- CitationVerdict: consistency invariant -----------------------------------------


def test_citation_verdict_verified_true_with_failure_raises_value_error() -> None:
    """`verified=True` paired with a non-`None` `failure` is an inconsistent
    verdict and must raise.
    """
    citation = _citation()

    with pytest.raises(ValueError, match=r"verified|failure"):
        CitationVerdict(citation=citation, verified=True, failure=FAILURE_UNREACHABLE)


def test_citation_verdict_verified_false_with_no_failure_raises_value_error() -> None:
    """`verified=False` paired with `failure=None` is an inconsistent verdict
    and must raise.
    """
    citation = _citation()

    with pytest.raises(ValueError, match=r"verified|failure"):
        CitationVerdict(citation=citation, verified=False, failure=None)
