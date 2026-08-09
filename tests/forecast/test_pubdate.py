"""Tests for windbreak.forecast.pubdate (issue #192).

Pins `extract_publication_date(html: str) -> datetime | None`'s full
extraction contract:

* Three sources, in a fixed priority order: (1) a JSON-LD `<script
  type="application/ld+json">` block's `datePublished` key, (2) a
  `<meta property="article:published_time" content="...">` tag, (3) a
  `<meta name="date" content="...">` tag. The first source present *and
  parseable* wins; a present-but-malformed higher-priority source falls
  through to the next one rather than aborting the whole extraction.
* Only a genuinely timezone-aware `datetime` is ever returned -- parsed via
  `datetime.fromisoformat` with a trailing `Z` normalized to `+00:00`, mirroring
  `windbreak.forecast.providers.futuresearch._parse_publication_date`'s
  Z-suffix handling. A naive datetime, a date-only string, a malformed string,
  or the complete absence of any of the three sources all return `None` --
  **never** a fabricated timezone and never a raised exception: a poisoned or
  merely sloppy page must not crash the pipeline's stage-5 citation-gathering
  loop.
* The module never touches a Python `float`: a JSON-LD block is parsed with
  `parse_float=Decimal` and a non-finite-constant-rejecting hook, exactly
  like every other untrusted-JSON parse in this package.

`windbreak/forecast/pubdate.py` does not exist yet, so importing it fails
collection with `ModuleNotFoundError` -- the expected Gate 1 RED state for
issue #192.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from windbreak.forecast.pubdate import extract_publication_date


def _json_ld(date_published: str) -> str:
    """Build a minimal JSON-LD `<script>` block carrying `datePublished`.

    Args:
        date_published: The raw `datePublished` string value.

    Returns:
        The `<script type="application/ld+json">...</script>` HTML fragment.
    """
    return (
        '<script type="application/ld+json">'
        '{"@context": "https://schema.org", "@type": "NewsArticle", '
        f'"datePublished": "{date_published}"}}'
        "</script>"
    )


def _article_meta(content: str) -> str:
    """Build an `article:published_time` `<meta>` tag.

    Args:
        content: The raw `content` attribute value.

    Returns:
        The `<meta property="article:published_time" content="...">` tag.
    """
    return f'<meta property="article:published_time" content="{content}">'


def _date_meta(content: str) -> str:
    """Build a `name="date"` `<meta>` tag.

    Args:
        content: The raw `content` attribute value.

    Returns:
        The `<meta name="date" content="...">` tag.
    """
    return f'<meta name="date" content="{content}">'


# --- Each source, in isolation -----------------------------------------------


def test_extracts_from_json_ld_date_published() -> None:
    """A page carrying only a JSON-LD `datePublished` extracts that date."""
    html = f"<html><head>{_json_ld('2024-03-15T10:00:00Z')}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 3, 15, 10, 0, 0, tzinfo=UTC)


def test_extracts_from_article_published_time_meta() -> None:
    """A page carrying only an `article:published_time` meta tag extracts it."""
    html = f"<html><head>{_article_meta('2024-06-01T08:30:00Z')}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 6, 1, 8, 30, 0, tzinfo=UTC)


def test_extracts_from_name_date_meta() -> None:
    """A page carrying only a `name="date"` meta tag extracts it."""
    html = f"<html><head>{_date_meta('2024-09-20T14:00:00Z')}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 9, 20, 14, 0, 0, tzinfo=UTC)


# --- Fixed priority order ------------------------------------------------------


def test_json_ld_takes_priority_over_both_meta_tags() -> None:
    """When all three sources are present, JSON-LD `datePublished` wins."""
    html = (
        "<html><head>"
        f"{_json_ld('2024-01-01T00:00:00Z')}"
        f"{_article_meta('2024-02-02T00:00:00Z')}"
        f"{_date_meta('2024-03-03T00:00:00Z')}"
        "</head></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2024, 1, 1, tzinfo=UTC)


def test_article_published_time_takes_priority_over_name_date() -> None:
    """With no JSON-LD present, `article:published_time` wins over `name="date"`."""
    html = (
        "<html><head>"
        f"{_article_meta('2024-02-02T00:00:00Z')}"
        f"{_date_meta('2024-03-03T00:00:00Z')}"
        "</head></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2024, 2, 2, tzinfo=UTC)


def test_malformed_json_ld_falls_through_to_article_meta() -> None:
    """A higher-priority source that is present but unparseable falls through
    to the next source in priority order, rather than aborting to `None`.
    """
    html = (
        "<html><head>"
        f"{_json_ld('not-a-real-date')}"
        f"{_article_meta('2024-05-05T00:00:00Z')}"
        "</head></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2024, 5, 5, tzinfo=UTC)


def test_json_ld_block_missing_date_published_key_falls_through() -> None:
    """A JSON-LD block present but lacking `datePublished` entirely falls
    through to the next source.
    """
    json_ld_no_date = (
        '<script type="application/ld+json">'
        '{"@context": "https://schema.org", "@type": "NewsArticle"}'
        "</script>"
    )
    meta = _date_meta("2024-07-07T00:00:00Z")
    html = f"<html><head>{json_ld_no_date}{meta}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 7, 7, tzinfo=UTC)


# --- Timezone-aware only: never a fabricated offset ---------------------------


def test_returns_none_for_a_naive_datetime_string() -> None:
    """A datetime string with no UTC offset (no `Z`, no `+HH:MM`) is naive --
    never coerced into a fabricated timezone.
    """
    html = f"<html><head>{_article_meta('2024-03-15T10:00:00')}</head></html>"

    assert extract_publication_date(html) is None


def test_returns_none_for_a_date_only_string() -> None:
    """A bare date (no time component at all) parses as naive -- also `None`."""
    html = f"<html><head>{_article_meta('2024-03-15')}</head></html>"

    assert extract_publication_date(html) is None


def test_returns_none_for_a_malformed_date_string() -> None:
    """Free text that is not any recognizable date format returns `None`."""
    html = f"<html><head>{_article_meta('not even a date')}</head></html>"

    assert extract_publication_date(html) is None


def test_accepts_z_suffix_and_normalizes_to_a_utc_offset() -> None:
    """A trailing `Z` is normalized to `+00:00`, mirroring the futuresearch
    provider's own publication-date parsing.
    """
    html = f"<html><head>{_article_meta('2024-03-15T10:00:00Z')}</head></html>"

    result = extract_publication_date(html)

    assert result is not None
    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(0)


def test_accepts_an_explicit_non_utc_offset_suffix() -> None:
    """An explicit non-UTC offset (e.g. `+05:30`) is accepted and preserved as
    a timezone-aware datetime -- it need not be UTC to count as "aware".
    """
    html = f"<html><head>{_article_meta('2024-03-15T10:00:00+05:30')}</head></html>"

    result = extract_publication_date(html)

    assert result is not None
    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(hours=5, minutes=30)


# --- Absence: no exception, just None ------------------------------------------


def test_returns_none_when_no_source_is_present_at_all() -> None:
    """A page carrying none of the three sources returns `None`."""
    html = "<html><head><title>No dates here</title></head><body>hello</body></html>"

    assert extract_publication_date(html) is None


def test_returns_none_for_empty_string_input() -> None:
    """An empty string returns `None`, never an exception."""
    assert extract_publication_date("") is None


def test_returns_none_and_never_raises_for_malformed_html() -> None:
    """Unclosed/malformed markup never raises -- `html.parser.HTMLParser` is
    tag-soup tolerant, and an unparseable date source still degrades to
    `None`.
    """
    html = '<html><head><meta property="article:published_time" content="'

    assert extract_publication_date(html) is None


# --- Non-finite JSON-LD constants are rejected, never crash --------------------


def test_json_ld_with_non_finite_constant_falls_through_without_raising() -> None:
    """A JSON-LD block containing a non-finite JSON constant (`Infinity`) in
    an unrelated field is rejected by the parse hook (never materialized as a
    Python float) and never raises -- extraction falls through to the next
    available source instead of crashing the pipeline.
    """
    poisoned_json_ld = (
        '<script type="application/ld+json">'
        '{"datePublished": "2024-01-01T00:00:00Z", "score": Infinity}'
        "</script>"
    )
    meta = _date_meta("2024-08-08T00:00:00Z")
    html = f"<html><head>{poisoned_json_ld}{meta}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 8, 8, tzinfo=UTC)


def test_extraction_over_realistic_full_page_with_visible_body_text() -> None:
    """A realistic full page (JSON-LD in the head, ordinary prose in the body)
    still extracts the JSON-LD date correctly -- the parser is not confused
    by unrelated markup.
    """
    html = (
        "<html><head>"
        f"{_json_ld('2024-11-11T09:15:00Z')}"
        "<title>Some Headline</title>"
        "</head><body>"
        "<p>Ordinary article prose that has nothing to do with dates.</p>"
        "</body></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2024, 11, 11, 9, 15, 0, tzinfo=UTC)
