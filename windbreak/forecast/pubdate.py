"""Best-effort publication-date extraction from a raw fetched page (SPEC S8.5).

:func:`extract_publication_date` pulls a *timezone-aware* publication
:class:`~datetime.datetime` out of a fetched HTML page for citation provenance,
degrading to ``None`` -- never raising, never fabricating a timezone -- on any
absent, malformed, or naive source. It runs on the *raw* fetched content
(before :func:`windbreak.forecast.sanitize.sanitize_content` strips every
``<script>`` subtree), so a JSON-LD block's date survives to be read here.

Three sources are consulted in a fixed priority order; the first that yields a
timezone-aware datetime wins, and a present-but-unparseable higher-priority
source falls through to the next rather than aborting the whole extraction:

1. A JSON-LD ``<script type="application/ld+json">`` block's ``datePublished``
   (tolerating either a top-level object or a list of objects).
2. A ``<meta property="article:published_time" content="...">`` tag.
3. A ``<meta name="date" content="...">`` tag.

Each candidate string is parsed with :meth:`datetime.datetime.fromisoformat`
after normalizing a trailing ``Z`` to ``+00:00`` (mirroring
:func:`windbreak.forecast.providers.futuresearch._parse_publication_date`). Only
a genuinely timezone-aware result is returned: a naive, date-only, or malformed
value degrades to ``None``. A JSON-LD block is parsed with ``parse_float=Decimal``
and a non-finite-constant-rejecting hook, so the module never touches a Python
``float`` and a poisoned block (e.g. a stray ``Infinity``) falls through
without raising -- it never crashes the pipeline's stage-5 citation loop.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from html.parser import HTMLParser
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The ``<script>`` ``type`` attribute marking an embedded JSON-LD block.
_JSON_LD_TYPE = "application/ld+json"

#: The JSON-LD key holding a publication date.
_DATE_PUBLISHED_KEY = "datePublished"

#: The ``<meta property=...>`` value naming an article's published time.
_ARTICLE_PUBLISHED_PROPERTY = "article:published_time"

#: The ``<meta name=...>`` value naming a generic date.
_DATE_NAME = "date"

#: The UTC-offset spelling ``datetime.fromisoformat`` accepts in place of ``Z``.
_UTC_SUFFIX = "+00:00"
_ZULU_SUFFIX = "Z"


def _reject_constant(token: str) -> NoReturn:
    """Reject a non-finite JSON constant token (``Infinity``/``NaN``).

    Installed as ``json.loads(..., parse_constant=...)`` so a non-standard
    constant -- which ``json.loads`` would otherwise materialize as a real
    Python ``float`` -- fails the parse instead of smuggling a float in.

    Args:
        token: The non-finite constant token the parser encountered.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"non-finite JSON constant is banned, got {token!r}")


class _PublicationDateParser(HTMLParser):
    """A tag-soup-tolerant collector of the three publication-date sources.

    Accumulates every ``application/ld+json`` script block's text and the first
    ``article:published_time`` / ``name="date"`` meta ``content`` value, so
    :func:`extract_publication_date` can resolve them in priority order.
    """

    def __init__(self) -> None:
        """Initialize with empty source collectors."""
        super().__init__()
        self.json_ld_blocks: list[str] = []
        self.article_published_time: str | None = None
        self.name_date: str | None = None
        self._in_json_ld = False
        self._current_json_ld: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Enter a JSON-LD block or capture a date-bearing meta tag.

        Args:
            tag: The lowercased element name.
            attrs: The element's ``(name, value)`` attribute pairs.
        """
        attr_map = dict(attrs)
        if tag == "script" and attr_map.get("type") == _JSON_LD_TYPE:
            self._in_json_ld = True
            self._current_json_ld = []
        elif tag == "meta":
            self._capture_meta(attr_map)

    def handle_endtag(self, tag: str) -> None:
        """Close an open JSON-LD block, retaining its accumulated text.

        Args:
            tag: The lowercased element name.
        """
        if tag == "script" and self._in_json_ld:
            self.json_ld_blocks.append("".join(self._current_json_ld))
            self._in_json_ld = False

    def handle_data(self, data: str) -> None:
        """Accumulate text while inside an open JSON-LD block.

        Args:
            data: The character data between tags.
        """
        if self._in_json_ld:
            self._current_json_ld.append(data)

    def _capture_meta(self, attr_map: dict[str, str | None]) -> None:
        """Record the first article-published-time / name-date meta content.

        Args:
            attr_map: The meta element's attribute map.
        """
        content = attr_map.get("content")
        if content is None:
            return
        if (
            attr_map.get("property") == _ARTICLE_PUBLISHED_PROPERTY
            and self.article_published_time is None
        ):
            self.article_published_time = content
        elif attr_map.get("name") == _DATE_NAME and self.name_date is None:
            self.name_date = content


def _json_ld_objects(parsed: object) -> Iterator[dict[str, object]]:
    """Yield the JSON-LD objects in a parsed block (a bare object or a list).

    Args:
        parsed: The parsed JSON-LD value.

    Yields:
        Each mapping in the block, so a top-level object or a list of objects
        are handled uniformly.
    """
    if isinstance(parsed, dict):
        yield parsed
    elif isinstance(parsed, list):
        yield from (item for item in parsed if isinstance(item, dict))


def _json_ld_date_candidates(blocks: list[str]) -> Iterator[str]:
    """Yield each JSON-LD ``datePublished`` string, skipping unparseable blocks.

    Args:
        blocks: The collected ``application/ld+json`` block texts.

    Yields:
        Every string-valued ``datePublished`` found, in block order. A block
        that fails the float-free JSON parse is silently skipped.
    """
    for block in blocks:
        try:
            parsed = json.loads(
                block, parse_float=Decimal, parse_constant=_reject_constant
            )
        except ValueError:
            continue
        for obj in _json_ld_objects(parsed):
            value = obj.get(_DATE_PUBLISHED_KEY)
            if isinstance(value, str):
                yield value


def _parse_aware(value: str) -> datetime | None:
    """Parse an ISO-8601 string into a timezone-aware datetime, or ``None``.

    A trailing ``Z`` is normalized to ``+00:00`` before parsing. A naive,
    date-only, or malformed value degrades to ``None`` -- never a fabricated
    timezone.

    Args:
        value: The candidate ISO-8601 datetime string.

    Returns:
        The parsed timezone-aware datetime, or ``None``.
    """
    try:
        parsed = datetime.fromisoformat(value.replace(_ZULU_SUFFIX, _UTC_SUFFIX))
    except ValueError:
        return None
    if parsed.utcoffset() is None:
        return None
    return parsed


def extract_publication_date(html: str) -> datetime | None:
    """Extract a timezone-aware publication date from a page, or ``None``.

    Consults the three sources in priority order (JSON-LD ``datePublished``,
    ``article:published_time`` meta, ``name="date"`` meta) and returns the first
    that parses to a timezone-aware datetime. Absent, naive, or malformed
    sources degrade to ``None``; the function never raises.

    Args:
        html: The raw fetched page markup.

    Returns:
        The extracted timezone-aware publication datetime, or ``None``.
    """
    parser = _PublicationDateParser()
    parser.feed(html)
    candidates = (
        *_json_ld_date_candidates(parser.json_ld_blocks),
        parser.article_published_time,
        parser.name_date,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        parsed = _parse_aware(candidate)
        if parsed is not None:
            return parsed
    return None
