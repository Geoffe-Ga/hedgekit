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

Issue #313 added the meta/JSON-LD extraction branches at the bottom of this
module: a `<meta>` with no usable `content`, a `<meta>` matching neither date
shape, and a JSON-LD block whose top-level value is a list rather than a bare
object. One caveat is recorded there rather than hidden: the `content is None`
early return in `_capture_meta` is an **equivalent mutant** through the public
API -- deleting it writes `None` over an already-`None` first-wins sentinel, so
no page can distinguish the two. It is covered but not killable; see the note
above `test_content_less_article_meta_does_not_suppress_a_later_date_meta`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

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


# --- Issue #313: meta/JSON-LD extraction branches ------------------------------
#
# Every decoy below carries a *parseable* date that differs from the date the
# page really publishes. A decoy holding free text (or the same date) could not
# tell "the wrong tag was preferred" apart from "no tag was read at all" -- both
# would yield the correct answer for the wrong reason.


def _named_meta(name: str, content: str) -> str:
    """Build a `<meta name="..." content="...">` tag with an arbitrary name.

    Args:
        name: The raw `name` attribute value.
        content: The raw `content` attribute value.

    Returns:
        The `<meta name="..." content="...">` tag.
    """
    return f'<meta name="{name}" content="{content}">'


def _propertied_meta(prop: str, content: str) -> str:
    """Build a `<meta property="..." content="...">` tag with any property.

    Args:
        prop: The raw `property` attribute value.
        content: The raw `content` attribute value.

    Returns:
        The `<meta property="..." content="...">` tag.
    """
    return f'<meta property="{prop}" content="{content}">'


def _list_json_ld(*entries: str) -> str:
    """Build a JSON-LD `<script>` block whose top-level value is a JSON list.

    Args:
        *entries: Raw JSON texts to place, in order, inside the list.

    Returns:
        The `<script type="application/ld+json">[...]</script>` HTML fragment.
    """
    return f'<script type="application/ld+json">[{", ".join(entries)}]</script>'


# --- A meta tag carrying no usable `content` ----------------------------------
#
# Honest caveat: `_capture_meta`'s `if content is None: return` cannot be killed
# through the public API. Deleting it assigns `None` to a collector that is
# already `None`, and "first one wins" is decided by `is None`, so the two
# versions agree on every page (verified by differential comparison over 1110
# generated meta combinations). These tests pin the *observable* contract --
# a content-less tag does not consume the first-wins slot and does not suppress
# a later real date -- which is what a future refactor could actually break.


def test_content_less_article_meta_does_not_suppress_a_later_date_meta() -> None:
    """A date-shaped meta with no `content` attribute at all is skipped.

    An `article:published_time` meta whose `content` attribute is entirely
    absent carries no date, so extraction must continue to the `name="date"`
    meta rather than latching onto the empty tag. The two dates differ, so a
    wrong answer here is a *different* datetime, not the same one.
    """
    html = (
        "<html><head>"
        '<meta property="article:published_time">'
        f"{_date_meta('2024-10-10T06:00:00Z')}"
        "</head></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2024, 10, 10, 6, 0, 0, tzinfo=UTC)


def test_valueless_content_attribute_is_treated_as_absent() -> None:
    """A bare `content` attribute (no `=value`) parses as `None`, not `""`.

    `HTMLParser` reports `<meta ... content>` as the pair `("content", None)`,
    a different shape from the missing-attribute case above. It must reach the
    same conclusion: no date here, keep looking.
    """
    html = (
        "<html><head>"
        '<meta property="article:published_time" content>'
        f"{_article_meta('2025-01-02T03:04:05Z')}"
        "</head></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_page_whose_only_date_meta_lacks_content_yields_none() -> None:
    """With no other source, a content-less date meta degrades to `None`.

    Fail closed: an unreadable publication date is reported as absent, never
    filled in from the clock, the epoch, or the fetch time.
    """
    html = '<html><head><meta property="article:published_time"></head></html>'

    assert extract_publication_date(html) is None


# --- A meta tag matching neither date shape -----------------------------------


def test_meta_matching_neither_date_shape_is_ignored() -> None:
    """A meta that is neither date shape leaves both collectors untouched.

    `name="viewport"` carries a `content` value, so it gets past the
    content-presence check and must then fall out of *both* the
    `article:published_time` and the `name="date"` arms.
    """
    html = f"<html><head>{_named_meta('viewport', 'width=device-width')}</head></html>"

    assert extract_publication_date(html) is None


def test_a_date_valued_meta_under_the_wrong_name_is_not_harvested() -> None:
    """Only `name="date"` is read -- not any meta that happens to hold a date.

    `author` and `description` here both carry perfectly parseable timestamps.
    If the `name` comparison were dropped or widened, first-wins would capture
    2024-04-04 and the page would be dated six months early.
    """
    html = (
        "<html><head>"
        f"{_named_meta('author', '2024-04-04T00:00:00Z')}"
        f"{_named_meta('description', '2024-05-05T00:00:00Z')}"
        f"{_date_meta('2024-10-10T00:00:00Z')}"
        "</head></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2024, 10, 10, tzinfo=UTC)


def test_a_non_published_property_meta_is_not_read_as_published_time() -> None:
    """`og:updated_time` and `article:modified_time` are not publication dates.

    A *modified* time is systematically later than the publication time it
    would be mistaken for, which is precisely the direction that lets a source
    look newer than it is.
    """
    html = (
        "<html><head>"
        f"{_propertied_meta('og:updated_time', '2023-03-03T00:00:00Z')}"
        f"{_propertied_meta('article:modified_time', '2023-04-04T00:00:00Z')}"
        f"{_article_meta('2024-12-12T00:00:00Z')}"
        "</head></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2024, 12, 12, tzinfo=UTC)


def test_a_bare_content_meta_with_no_name_or_property_is_ignored() -> None:
    """A meta carrying only `content` names no source and is not harvested."""
    html = (
        "<html><head>"
        '<meta content="2024-04-04T00:00:00Z">'
        f"{_date_meta('2024-10-10T00:00:00Z')}"
        "</head></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2024, 10, 10, tzinfo=UTC)


# --- A JSON-LD block whose top-level value is a list --------------------------


def test_extracts_date_published_from_a_list_form_json_ld_block() -> None:
    """A top-level JSON *list* of objects is read like a bare object.

    The page also carries a meta date, deliberately different, so this asserts
    the list arm actually ran: had the list gone unread, extraction would have
    fallen through and returned the meta's 2024-06-06 instead.
    """
    json_ld = _list_json_ld(
        '{"@type": "WebPage", "name": "Landing"}',
        '{"@type": "NewsArticle", "datePublished": "2024-02-14T07:00:00Z"}',
    )
    html = f"<html><head>{json_ld}{_date_meta('2024-06-06T00:00:00Z')}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 2, 14, 7, 0, 0, tzinfo=UTC)


def test_list_form_json_ld_skips_non_object_entries() -> None:
    """Strings and numbers inside a JSON-LD list are skipped, not dereferenced.

    Scraped JSON-LD is hostile input: a list of mixed types must not crash the
    stage-5 citation loop on its way to the one object that has a date.
    """
    json_ld = _list_json_ld(
        '"https://example.test/schema"',
        "42",
        "null",
        '{"datePublished": "2024-08-21T12:30:00Z"}',
    )
    html = f"<html><head>{json_ld}{_date_meta('2024-06-06T00:00:00Z')}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 8, 21, 12, 30, 0, tzinfo=UTC)


def test_first_list_entry_with_a_date_published_wins() -> None:
    """Within one list, the earliest-positioned `datePublished` is taken.

    Both entries carry a valid but different date, so taking the wrong one is
    observable rather than coincidentally equal.
    """
    json_ld = _list_json_ld(
        '{"datePublished": "2024-03-03T00:00:00Z"}',
        '{"datePublished": "2024-09-09T00:00:00Z"}',
    )
    html = f"<html><head>{json_ld}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 3, 3, tzinfo=UTC)


def test_a_list_entry_with_an_unparseable_date_falls_through_to_the_next() -> None:
    """A list entry whose `datePublished` is junk yields to the next candidate.

    A present-but-unparseable higher-priority source must not abort the whole
    extraction -- the same fall-through contract the bare-object form has.
    """
    json_ld = _list_json_ld(
        '{"datePublished": "yesterday-ish"}',
        '{"datePublished": "2024-07-19T18:45:00Z"}',
    )
    html = f"<html><head>{json_ld}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 7, 19, 18, 45, 0, tzinfo=UTC)


def test_an_empty_list_json_ld_block_falls_through_to_the_meta_tags() -> None:
    """An empty JSON-LD list yields no candidates and never raises."""
    html = (
        "<html><head>"
        f"{_list_json_ld()}{_date_meta('2024-06-06T00:00:00Z')}"
        "</head></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2024, 6, 6, tzinfo=UTC)


def test_a_list_json_ld_with_no_date_published_anywhere_falls_through() -> None:
    """A list of dateless objects falls through rather than returning `None`."""
    json_ld = _list_json_ld('{"@type": "WebPage"}', '{"@type": "Organization"}')
    html = f"<html><head>{json_ld}{_date_meta('2024-06-06T00:00:00Z')}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 6, 6, tzinfo=UTC)


def test_a_scalar_top_level_json_ld_block_yields_no_candidates() -> None:
    """A JSON-LD block that is neither an object nor a list is ignored.

    `"just a string"` and `42` are valid JSON, so the block parses cleanly and
    then matches neither the object nor the list shape. It must yield nothing
    and let the meta tags answer, rather than raising on a `.get` that a bare
    scalar does not have.
    """
    scalar_blocks = (
        '<script type="application/ld+json">"https://example.test/"</script>'
        '<script type="application/ld+json">42</script>'
    )
    html = (
        f"<html><head>{scalar_blocks}{_date_meta('2024-06-06T00:00:00Z')}</head></html>"
    )

    result = extract_publication_date(html)

    assert result == datetime(2024, 6, 6, tzinfo=UTC)


def test_a_non_string_date_published_in_a_list_entry_is_refused() -> None:
    """A numeric `datePublished` is not coerced into a date.

    An epoch-looking integer must not be guessed at (seconds? milliseconds?
    whose epoch?). It is skipped, and the next real source is used.
    """
    json_ld = _list_json_ld('{"datePublished": 1710500400}')
    html = f"<html><head>{json_ld}{_date_meta('2024-06-06T00:00:00Z')}</head></html>"

    result = extract_publication_date(html)

    assert result == datetime(2024, 6, 6, tzinfo=UTC)


# --- Timezone: a naive pubdate is refused, never read as host-local time -------
#
# `datetime.fromisoformat` returns a **naive** datetime for an offsetless
# string, and `.astimezone()` on a naive value silently reads the wall clock as
# the *host's* local time. CI runs UTC, where the correct and misreading paths
# agree exactly, so an unpinned assertion here would be a permanent false green
# (issue #392, PR #396). Each test below pins the process timezone.


def test_naive_json_ld_date_is_refused_west_of_utc(
    local_timezone_utc_minus_5: None,
) -> None:
    """An offsetless JSON-LD date is refused, not localized, west of UTC.

    This is the dangerous direction: read as host-local, 2024-12-18T19:00 on a
    UTC-05:00 host becomes 2024-12-19T00:00Z -- a *different calendar day*, and
    a later one. A source stamped a day late can pass as prior evidence for a
    forecast it actually postdates. The page's only other source is deliberately
    absent so the refusal itself is what is asserted.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    json_ld = _json_ld("2024-12-18T19:00:00")
    html = f"<html><head>{json_ld}</head></html>"

    assert extract_publication_date(html) is None


def test_naive_meta_date_is_refused_east_of_utc(
    local_timezone_utc_plus_5: None,
) -> None:
    """An offsetless meta date is refused east of UTC too.

    The mirror skew: 2024-12-19T02:00 read as host-local on a UTC+05:00 host
    becomes 2024-12-18T21:00Z, the *previous* day. Pinning only the westward
    host would let a merely-conservative misreading pass for correctness.

    Args:
        local_timezone_utc_plus_5: Pins the process timezone east of UTC.
    """
    html = f"<html><head>{_article_meta('2024-12-19T02:00:00')}</head></html>"

    assert extract_publication_date(html) is None


def test_an_offset_bearing_date_is_preserved_not_converted_to_host_local(
    local_timezone_utc_minus_5: None,
) -> None:
    """An explicit offset survives extraction unshifted, on a non-UTC host.

    Equality alone cannot catch a stray `.astimezone()`: a converted datetime
    still compares equal because it names the same instant. The wall-clock
    fields and the offset are asserted individually, so a normalization to the
    host's zone (which would read 05:30 at UTC-05:00) fails.

    Args:
        local_timezone_utc_minus_5: Pins the process timezone west of UTC.
    """
    json_ld = _json_ld("2024-12-18T19:00:00+05:30")
    html = f"<html><head>{json_ld}</head></html>"

    result = extract_publication_date(html)

    assert result is not None
    assert result.utcoffset() == timedelta(hours=5, minutes=30)
    assert (result.year, result.month, result.day) == (2024, 12, 18)
    assert (result.hour, result.minute) == (19, 0)
    assert result == datetime(
        2024, 12, 18, 19, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
