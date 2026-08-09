"""Tests for `bounded_web_research`'s real publication-date extraction (#192).

Issue #192 deletes `windbreak.forecast.pipeline`'s private, fixed
`_CITATION_PUBLICATION_DATE` stub constant and replaces it with
`publication_date=windbreak.forecast.pubdate.extract_publication_date(content)`,
computed over each gathered page's **raw** fetched content -- *before*
`windbreak.forecast.sanitize.sanitize_content` strips it down to visible text.
Pins three behaviors the pre-#192 fixed-date stub could never distinguish:

1. A page carrying a real, extractable date yields a citation stamped with
   that (timezone-aware) date.
2. A page carrying no extractable date yields `publication_date=None` --
   never a fabricated constant.
3. Extraction happens on the *raw* content, not the sanitized one: a date
   living inside a `<script type="application/ld+json">` block would be
   entirely discarded by `sanitize_content` (every `<script>` element's
   subtree is suppressed, regardless of its `type` attribute), so a
   post-sanitize extraction would wrongly yield `None` here.

`windbreak/forecast/pubdate.py` does not exist yet, so importing it fails
collection with `ModuleNotFoundError` -- the expected Gate 1 RED state for
issue #192.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from windbreak.forecast.pipeline import bounded_web_research, decompose_subquestions

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.sandbox import ResearchTools

    ResearchToolsFactory = Callable[..., ResearchTools]

#: A page carrying a genuine JSON-LD `datePublished`, served for every URL.
_DATED_PAGE = (
    '<html><head><script type="application/ld+json">'
    '{"@context": "https://schema.org", "@type": "NewsArticle", '
    '"datePublished": "2024-11-20T09:00:00Z"}</script></head>'
    "<body><p>Ordinary article prose with no dates in the visible text.</p>"
    "</body></html>"
)

#: A page carrying no extractable date at all, served for every URL.
_DATELESS_PAGE = (
    "<html><head><title>No date here</title></head>"
    "<body><p>Ordinary article prose with no dates anywhere.</p></body></html>"
)


class _FixedContentFetchTransport:
    """A `FetchTransport` double serving one fixed page for every URL."""

    def __init__(self, page: str) -> None:
        """Store the page every `fetch` call returns.

        Args:
            page: The raw content to serve verbatim for any URL.
        """
        self._page = page

    def fetch(self, url: str) -> str:
        """Return the fixed page, ignoring `url` entirely.

        Args:
            url: The (unused) URL being fetched.

        Returns:
            `self._page`, verbatim.
        """
        return self._page


def test_bounded_web_research_stamps_the_extracted_publication_date(
    market: NormalizedMarket,
    tmp_path: Path,
    research_tools_factory: ResearchToolsFactory,
) -> None:
    """A page carrying a real, extractable JSON-LD date yields a citation
    stamped with that exact timezone-aware datetime.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path, fetch_transport=_FixedContentFetchTransport(_DATED_PAGE)
    )
    subquestions = decompose_subquestions(market)

    citations = bounded_web_research(subquestions, tools=tools)

    assert citations
    assert all(
        citation.publication_date == datetime(2024, 11, 20, 9, 0, 0, tzinfo=UTC)
        for citation in citations
    )


def test_bounded_web_research_dateless_page_yields_none_publication_date(
    market: NormalizedMarket,
    tmp_path: Path,
    research_tools_factory: ResearchToolsFactory,
) -> None:
    """A page carrying no extractable date at all yields `publication_date is
    None` -- never a fabricated fixed-date constant.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path,
        fetch_transport=_FixedContentFetchTransport(_DATELESS_PAGE),
    )
    subquestions = decompose_subquestions(market)

    citations = bounded_web_research(subquestions, tools=tools)

    assert citations
    assert all(citation.publication_date is None for citation in citations)


def test_bounded_web_research_extracts_from_raw_content_before_sanitizing(
    market: NormalizedMarket,
    tmp_path: Path,
    research_tools_factory: ResearchToolsFactory,
) -> None:
    """The date lives inside a `<script type="application/ld+json">` block,
    which `sanitize_content` entirely discards (every `<script>` subtree is
    suppressed regardless of its `type`). A citation still carries the
    extracted date, proving `bounded_web_research` runs date extraction on the
    *raw* fetched content, not the sanitized excerpt -- a post-sanitize
    extraction would wrongly yield `None` for every citation here.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path, fetch_transport=_FixedContentFetchTransport(_DATED_PAGE)
    )
    subquestions = decompose_subquestions(market)

    citations = bounded_web_research(subquestions, tools=tools)

    assert citations
    assert all(citation.publication_date is not None for citation in citations)


def test_bounded_web_research_citation_source_type_still_research_note(
    market: NormalizedMarket,
    tmp_path: Path,
    research_tools_factory: ResearchToolsFactory,
) -> None:
    """The publication-date extraction change leaves `source_type` unchanged:
    every gathered citation is still stamped `"research_note"`.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path, fetch_transport=_FixedContentFetchTransport(_DATED_PAGE)
    )
    subquestions = decompose_subquestions(market)

    citations = bounded_web_research(subquestions, tools=tools)

    assert citations
    assert all(citation.source_type == "research_note" for citation in citations)


def test_pipeline_module_no_longer_defines_the_fixed_publication_date_constant() -> (
    None
):
    """`windbreak.forecast.pipeline._CITATION_PUBLICATION_DATE` is deleted --
    the module no longer carries any fixed-date stub for citation provenance.
    """
    import windbreak.forecast.pipeline as pipeline_module

    assert not hasattr(pipeline_module, "_CITATION_PUBLICATION_DATE")
