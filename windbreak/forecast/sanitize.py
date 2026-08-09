"""SPEC S8.5 prompt-injection defenses for untrusted web content (threat T1).

Bounded web research (``pipeline.bounded_web_research``) fetches attacker-
controlled pages. This module is the response-side firewall between those raw
bytes and any LLM prompt: it strips a page down to its *visible* text
(discarding ``<script>``/``<style>`` and CSS/ARIA-hidden payloads), neutralizes
any forged untrusted-data delimiter, caps the surviving excerpt at
:data:`MAX_QUOTE_WORDS` words, wraps that excerpt in an explicitly-labelled data
block (:func:`wrap_data_block`), and validates a model's *own* vote response for
delimiter forgery or a tool-call lure (:func:`validate_vote_response`).

The module is pure and stdlib-only (``html.parser.HTMLParser``, no new
dependency) and never touches a float -- it sits on the probability/money path
guarded by ``scripts/lint_no_floats.py``, so every operation here is a
string/int one.

The sanitize/raw-hash contract is deliberate. ``sanitize_content`` returns
*visible* text with hidden subtrees removed, while a citation's content hash
(``pipeline.bounded_web_research``) stays over the *raw* bytes and
``citations.verify_citation`` re-checks the sanitized quote as a raw substring.
So a page whose attack sits *after* the first :data:`MAX_QUOTE_WORDS` clean
words yields a quote that is still a contiguous raw substring (it verifies and
the run proceeds), whereas a page with a hidden span, a ``<script>`` block, or a
forged delimiter *inside* that first window has its removal break the raw-
substring property -- the citation then fails to verify and the run fails
closed, exactly the intended defense.

For that raw-substring re-check to hold on *benign* pages, sanitization must not
rewrite the bytes it keeps: entity and character references are preserved
verbatim (``&amp;`` stays ``&amp;``), never decoded, so ordinary content such as
"S&amp;P 500" re-verifies instead of spuriously abstaining. Leaving entities
encoded also neutralizes an entity-encoded delimiter forgery for free -- it stays
inert ``&lt;&lt;&lt;`` text rather than decoding into a real breakout token. The
one carried-forward caveat is whitespace: the raw content is assumed already
whitespace-normalized, since ``sanitize_content`` collapses internal whitespace
runs and that collapse (like a malformed, semicolon-less entity) would break the
raw-substring re-check and fail closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Final

#: Opening token of an untrusted-data block wrapping a web quote in a prompt.
DATA_BLOCK_BEGIN: Final = "<<<UNTRUSTED-DATA"

#: Closing token of an untrusted-data block wrapping a web quote in a prompt.
DATA_BLOCK_END: Final = "<<<END-UNTRUSTED-DATA>>>"

#: Maximum words retained in a sanitized quote excerpt (SPEC S8.5 length cap).
MAX_QUOTE_WORDS: Final = 25

#: Response-failure code: the model returned an empty/whitespace-only vote.
RESPONSE_FAILURE_EMPTY: Final = "empty_response"

#: Response-failure code: the model's response forges an untrusted-data
#: delimiter (an attempt to break out of, or fake, a data block).
RESPONSE_FAILURE_DELIMITER_FORGERY: Final = "delimiter_forgery"

#: Response-failure code: the model's response embeds a tool-call lure (a JSON
#: key that would coax a caller into dispatching an unrequested tool call).
RESPONSE_FAILURE_TOOL_CALL_LURE: Final = "tool_call_lure"

#: The JSON key tokens a tool-call lure uses; a vote response carrying any of
#: them is discarded rather than trusted (SPEC S8.5). Matching is a plain
#: substring test, so a legitimate vote response that happens to contain the
#: quoted literal ``"tool"`` is discarded conservatively -- an intentional
#: fail-closed tradeoff, and only a discard signal (probability never derives
#: from response text); broader marker coverage is tracked as a hardening
#: follow-up.
TOOL_CALL_MARKERS: Final[frozenset[str]] = frozenset(
    {'"tool"', '"tool_call"', '"function_call"'}
)

#: Response-failure code: the response is not a schema-valid vote object at all
#: (not JSON, not a JSON object, a required key is missing, or a required value
#: is of the wrong JSON type entirely).
RESPONSE_FAILURE_MALFORMED_VOTE_JSON: Final = "malformed_vote_json"

#: Response-failure code: ``probability_ppm`` is numeric-shaped but not a true
#: integer -- a JSON float, or a ``bool`` (an ``int`` subclass that must never
#: masquerade as a probability, mirroring ``records._require_ppm``).
RESPONSE_FAILURE_NON_INTEGER_PROBABILITY: Final = "non_integer_probability"

#: Response-failure code: ``probability_ppm`` is a true integer but falls
#: outside the inclusive ``[0, 1_000_000]`` ppm domain.
RESPONSE_FAILURE_PROBABILITY_OUT_OF_RANGE: Final = "probability_out_of_range"

#: Response-failure code: a *present* ``rationale_summary`` violates a content
#: constraint (not a string, empty, or longer than :data:`MAX_RATIONALE_CHARS`).
RESPONSE_FAILURE_INVALID_RATIONALE: Final = "invalid_rationale"

#: Response-failure code: the vote object carries a top-level key outside the
#: closed schema set, mirroring ``config.loader``'s unknown-key-is-fatal rule.
RESPONSE_FAILURE_UNKNOWN_VOTE_KEY: Final = "unknown_vote_key"

#: Response-failure code: the forecaster reported a version outside the
#: operator-pinned set under the strict drift policy -- discarded per-vote
#: (never crashes the whole run), mirroring every other rejection path.
RESPONSE_FAILURE_VERSION_DRIFT: Final = "forecaster_version_drift"

#: Response-failure code: the transport returned a non-2xx status; the response
#: is rejected fast rather than relied upon to fail body-schema validation.
RESPONSE_FAILURE_HTTP_STATUS: Final = "http_status_error"

#: Maximum characters a ``rationale_summary`` may carry (SPEC S6.3 length cap).
#: A model rationale is a short justification, never a document, so a generous
#: bound still rejects a response trying to smuggle a large payload past the
#: schema layer.
MAX_RATIONALE_CHARS: Final = 2_000

#: The vote schema's ``probability_ppm`` key.
_VOTE_PPM_KEY: Final = "probability_ppm"

#: The vote schema's ``rationale_summary`` key.
_VOTE_RATIONALE_KEY: Final = "rationale_summary"

#: The vote schema's ``abstain`` key.
_VOTE_ABSTAIN_KEY: Final = "abstain"

#: The closed set of top-level keys a schema-valid vote object may carry.
_REQUIRED_VOTE_KEYS: Final[frozenset[str]] = frozenset(
    {_VOTE_PPM_KEY, _VOTE_RATIONALE_KEY, _VOTE_ABSTAIN_KEY}
)

#: Lowest legal parts-per-million probability (inclusive).
_MIN_PPM: Final = 0

#: Highest legal parts-per-million probability (inclusive).
_MAX_PPM: Final = 1_000_000

#: HTML elements whose entire text subtree is discarded during sanitization:
#: executable, styling, or off-DOM containers that never carry visible prose.
_SUPPRESSED_TAGS: Final[frozenset[str]] = frozenset(
    {"script", "style", "template", "noscript", "iframe"}
)

#: Void HTML elements: they have no text subtree, so they are never pushed onto
#: the open-element stack (they carry no content to suppress or keep).
_VOID_TAGS: Final[frozenset[str]] = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

#: Space-stripped inline-``style`` fragments that mark an element as hidden; an
#: element whose ``style`` contains any of them has its subtree discarded.
_HIDDEN_STYLE_TOKENS: Final[frozenset[str]] = frozenset(
    {"display:none", "visibility:hidden", "font-size:0"}
)


@dataclass(frozen=True, slots=True)
class ResearchQuote:
    """One sanitized web quote paired with the URL it was fetched from.

    Attributes:
        url: The URL the sandbox actually fetched, never a URL parsed out of
            the page content -- so a citation-URL spoof buried in the text can
            never redirect where a quote claims to come from.
        text: The sanitized excerpt, at most :data:`MAX_QUOTE_WORDS` words.
    """

    url: str
    text: str


@dataclass(frozen=True, slots=True)
class ParsedVote:
    """A parsed, schema-valid structured vote response (SPEC S6.3).

    Produced by :func:`parse_vote_response` only after
    :func:`validate_vote_response` has cleared both the injection screen and the
    schema layer, so every field here is already known to be well-formed.

    Attributes:
        probability_ppm: The member's integer probability estimate, in ppm,
            within ``[0, 1_000_000]``.
        rationale_summary: The bounded, non-empty free-text rationale summary.
        abstain: Whether the member abstained from casting a usable vote.
    """

    probability_ppm: int
    rationale_summary: str
    abstain: bool


@dataclass(slots=True)
class _ElementFrame:
    """One open element on the sanitizer's tag stack.

    Attributes:
        tag: The (lowercased) element tag name.
        suppressed: Whether this element is a text-suppressing tag (script,
            style, and the like).
        hidden: Whether this element is hidden (via ``hidden``,
            ``aria-hidden``, or an inline ``style``).
    """

    tag: str
    suppressed: bool
    hidden: bool


def _style_hides(value: str | None) -> bool:
    """Return whether an inline ``style`` value marks its element hidden.

    Args:
        value: The raw ``style`` attribute value, or ``None`` for a valueless
            attribute.

    Returns:
        ``True`` if the space-stripped, lowercased style contains any hidden
        token (``display:none`` / ``visibility:hidden`` / ``font-size:0``).
    """
    if value is None:
        return False
    condensed = value.lower().replace(" ", "")
    return any(token in condensed for token in _HIDDEN_STYLE_TOKENS)


def _attrs_hidden(attrs: list[tuple[str, str | None]]) -> bool:
    """Return whether an element's attributes mark it visually hidden.

    Args:
        attrs: The element's ``(name, value)`` attributes as HTMLParser yields
            them (names already lowercased, values verbatim or ``None``).

    Returns:
        ``True`` if a bare ``hidden`` attribute, ``aria-hidden="true"``, or a
        hiding inline ``style`` is present.
    """
    for name, value in attrs:
        if name == "hidden":
            return True
        if name == "aria-hidden" and value is not None and value.lower() == "true":
            return True
        if name == "style" and _style_hides(value):
            return True
    return False


class _VisibleTextParser(HTMLParser):
    """An HTMLParser that collects only the *visible* character data of a page.

    Text inside a suppressed subtree (``<script>``, ``<style>``, ...) or a
    hidden element (``hidden`` / ``aria-hidden`` / a hiding inline ``style``)
    is dropped; everything else is retained. Entity and character references are
    kept *verbatim* in their original encoded form (``convert_charrefs=False``,
    re-emitted by :meth:`handle_entityref` / :meth:`handle_charref`). Preserving
    them, rather than decoding, serves the raw-hash contract in two ways: a
    benign ``&amp;`` / ``&#8217;`` stays a byte-for-byte substring of the raw
    fetched content so the citation re-verifies (no spurious abstention), and an
    entity-encoded delimiter forgery (``&lt;&lt;&lt;UNTRUSTED-DATA``) stays inert
    -- it never decodes into a real, breakout-capable delimiter token.
    """

    def __init__(self) -> None:
        """Initialize an empty open-element stack and visible-text buffer."""
        super().__init__(convert_charrefs=False)
        self._stack: list[_ElementFrame] = []
        self._parts: list[str] = []

    @property
    def _hidden(self) -> bool:
        """Return whether any currently-open element suppresses its text.

        Returns:
            ``True`` if some open frame is a suppressed tag or a hidden
            element, so character data must not be collected.
        """
        return any(frame.suppressed or frame.hidden for frame in self._stack)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Push a non-void element onto the stack, recording its visibility.

        Args:
            tag: The element's (lowercased) tag name.
            attrs: The element's attributes.
        """
        if tag in _VOID_TAGS:
            return
        self._stack.append(
            _ElementFrame(
                tag=tag,
                suppressed=tag in _SUPPRESSED_TAGS,
                hidden=_attrs_hidden(attrs),
            )
        )

    def handle_endtag(self, tag: str) -> None:
        """Pop back to the most recent matching start tag, if any.

        Popping the matched frame *and* any still-open frames nested inside it
        keeps the stack consistent against unclosed inner tags; a stray end tag
        with no matching start is ignored (tag-soup tolerant).

        Args:
            tag: The (lowercased) closing tag name.
        """
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        """Collect character data unless an open element suppresses it.

        Args:
            data: The character data between tags.
        """
        if not self._hidden:
            self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        """Re-emit a named entity reference (``&name;``) verbatim, unencoded.

        With ``convert_charrefs=False`` the parser reports each named reference
        here instead of decoding it; re-emitting the original ``&name;`` source
        keeps visible text byte-identical to the raw fetched content (so a
        benign entity re-verifies) and leaves an entity-encoded delimiter forgery
        inert.

        Args:
            name: The entity name between ``&`` and ``;`` (e.g. ``"amp"``).
        """
        if not self._hidden:
            self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        """Re-emit a numeric character reference (``&#name;``) verbatim.

        Args:
            name: The reference body after ``&#`` (a decimal string such as
                ``"8217"`` or a hex string such as ``"x2019"``).
        """
        if not self._hidden:
            self._parts.append(f"&#{name};")

    def visible_text(self) -> str:
        """Return the concatenated visible character data collected so far.

        Returns:
            Every retained data run, joined in document order.
        """
        return "".join(self._parts)


def _collapse_whitespace(text: str) -> str:
    """Collapse every run of whitespace in ``text`` to single spaces.

    Args:
        text: The text to normalize.

    Returns:
        ``text`` with leading/trailing whitespace stripped and internal runs
        collapsed to one space each.
    """
    return " ".join(text.split())


def _neutralize_delimiters(text: str) -> str:
    """Replace every residual untrusted-data delimiter token with a space.

    A space is inserted (not an empty string) so two fragments that were only
    separated by a token can never abut and reassemble a fresh token.

    Args:
        text: The post-parse visible text.

    Returns:
        ``text`` with each :data:`DATA_BLOCK_BEGIN` / :data:`DATA_BLOCK_END`
        occurrence replaced by a single space.
    """
    return text.replace(DATA_BLOCK_BEGIN, " ").replace(DATA_BLOCK_END, " ")


def sanitize_content(content: str) -> str:
    """Reduce raw fetched page content to neutralized, visible text (S8.5).

    Fast identity path: content with no ``<`` can carry no HTML markup and no
    literal delimiter token (both delimiter tokens contain ``<``), so it is
    only whitespace-collapsed -- any entity it holds is left encoded, which is
    exactly what the raw-hash contract needs. Otherwise the content is parsed
    for *visible* text -- suppressed subtrees (``<script>`` and kin) and hidden
    elements are dropped, while entities are kept verbatim (never decoded) -- and
    any literal delimiter token that survives parsing is neutralized before a
    final whitespace collapse.

    Args:
        content: The raw fetched page content.

    Returns:
        The neutralized, whitespace-collapsed visible text.
    """
    if "<" not in content:
        return _collapse_whitespace(content)
    parser = _VisibleTextParser()
    parser.feed(content)
    parser.close()
    neutralized = _neutralize_delimiters(parser.visible_text())
    return _collapse_whitespace(neutralized)


def extract_quote(sanitized: str, *, max_words: int = MAX_QUOTE_WORDS) -> str:
    """Take at most ``max_words`` words from already-sanitized text (S8.5).

    Args:
        sanitized: The sanitized text (as returned by :func:`sanitize_content`).
        max_words: The maximum number of words to retain (defaults to
            :data:`MAX_QUOTE_WORDS`).

    Returns:
        The first ``max_words`` whitespace-separated words joined by single
        spaces; an empty string for empty input.
    """
    return " ".join(sanitized.split()[:max_words])


def _contains_delimiter(text: str) -> bool:
    """Return whether ``text`` contains either untrusted-data delimiter token.

    Args:
        text: The text to inspect.

    Returns:
        ``True`` if :data:`DATA_BLOCK_BEGIN` or :data:`DATA_BLOCK_END` appears.
    """
    return DATA_BLOCK_BEGIN in text or DATA_BLOCK_END in text


def wrap_data_block(*, url: str, quote: str) -> str:
    """Wrap a sanitized quote in a labelled untrusted-data block (S8.5).

    Args:
        url: The URL the quote was fetched from.
        quote: The sanitized quote excerpt.

    Returns:
        The quote framed between the opening and closing delimiter tokens, with
        the source URL on the opening line.

    Raises:
        ValueError: If ``url`` contains a delimiter token, a newline, or a
            double-quote character (any of which could break the opening
            line's structure), or if ``quote`` contains a delimiter token.
    """
    if _contains_delimiter(url) or "\n" in url or '"' in url:
        raise ValueError(
            f"url must not contain a delimiter token, a newline, or a "
            f"double-quote character: {url!r}"
        )
    if _contains_delimiter(quote):
        raise ValueError(f"quote must not contain a delimiter token: {quote!r}")
    return f'{DATA_BLOCK_BEGIN} url="{url}">>>\n{quote}\n{DATA_BLOCK_END}'


def _probability_failure(value: object) -> str | None:
    """Return the failure code for a ``probability_ppm`` value, or ``None``.

    The bool/float/int ordering is load-bearing: ``bool`` is an ``int``
    subclass, so it must be rejected as non-integer *before* the ``int`` branch
    ever runs, and a JSON ``float`` is numeric-shaped but the wrong numeric kind
    (never silently truncated). Any non-numeric JSON type (string, list, object,
    ``null``) is malformed rather than non-integer.

    Args:
        value: The raw ``probability_ppm`` value parsed from the response.

    Returns:
        A ``RESPONSE_FAILURE_*`` code, or ``None`` when the value is a valid
        in-range integer.
    """
    if isinstance(value, bool):
        return RESPONSE_FAILURE_NON_INTEGER_PROBABILITY
    if isinstance(value, int):
        if _MIN_PPM <= value <= _MAX_PPM:
            return None
        return RESPONSE_FAILURE_PROBABILITY_OUT_OF_RANGE
    if isinstance(value, float):
        return RESPONSE_FAILURE_NON_INTEGER_PROBABILITY
    return RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def _rationale_failure(value: object) -> str | None:
    """Return the failure code for a ``rationale_summary`` value, or ``None``.

    The non-string guard runs first so ``len`` is never called on a non-string
    (e.g. a JSON ``null`` or number), failing closed on a present-but-bad value.

    Args:
        value: The raw ``rationale_summary`` value parsed from the response.

    Returns:
        :data:`RESPONSE_FAILURE_INVALID_RATIONALE` when the value is not a
        non-empty string of at most :data:`MAX_RATIONALE_CHARS`, else ``None``.
    """
    if not isinstance(value, str) or not value or len(value) > MAX_RATIONALE_CHARS:
        return RESPONSE_FAILURE_INVALID_RATIONALE
    return None


def _abstain_failure(value: object) -> str | None:
    """Return the failure code for an ``abstain`` value, or ``None`` if valid.

    ``abstain`` must be a genuine JSON ``bool`` (validated by ``isinstance``,
    never by truthiness): a wrong-typed value -- a string, a number, ``null``,
    or a nested container that could smuggle an *unbounded* payload past the
    :data:`MAX_RATIONALE_CHARS` cap (the only bounded free field) -- is rejected
    as malformed rather than coerced, so no non-boolean ever reaches
    :class:`ParsedVote`. The check mirrors the wrong-JSON-type handling of a
    non-numeric ``probability_ppm``.

    Args:
        value: The raw ``abstain`` value parsed from the response.

    Returns:
        :data:`RESPONSE_FAILURE_MALFORMED_VOTE_JSON` when ``value`` is not a
        ``bool``, else ``None``.
    """
    if not isinstance(value, bool):
        return RESPONSE_FAILURE_MALFORMED_VOTE_JSON
    return None


def _schema_failure(response: str) -> str | None:
    """Return the vote-schema failure code for a response, or ``None`` if valid.

    Runs only after the injection screen has cleared (see
    :func:`validate_vote_response`). Checks run in a fixed order: JSON parse,
    object shape, required keys present, closed key set, then per-field value
    constraints (probability before rationale before abstain).

    Args:
        response: The injection-clean raw vote-completion text.

    Returns:
        A ``RESPONSE_FAILURE_*`` code for the first failing check, or ``None``.
    """
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return RESPONSE_FAILURE_MALFORMED_VOTE_JSON
    if not isinstance(payload, dict):
        return RESPONSE_FAILURE_MALFORMED_VOTE_JSON
    if not _REQUIRED_VOTE_KEYS.issubset(payload):
        return RESPONSE_FAILURE_MALFORMED_VOTE_JSON
    if set(payload) - _REQUIRED_VOTE_KEYS:
        return RESPONSE_FAILURE_UNKNOWN_VOTE_KEY
    return (
        _probability_failure(payload[_VOTE_PPM_KEY])
        or _rationale_failure(payload[_VOTE_RATIONALE_KEY])
        or _abstain_failure(payload[_VOTE_ABSTAIN_KEY])
    )


def screen_untrusted_text(text: str) -> str | None:
    """Screen an arbitrary piece of untrusted text for injection artifacts.

    The standalone SPEC S8.5 injection screen, exposing the module's existing
    delimiter-forgery and tool-call-lure checks (in that fixed order) over *any*
    untrusted text -- a whole response body, a single parsed rationale, one
    citation quote -- so a caller need not duplicate the delimiter/tool-lure
    logic. Unlike :func:`validate_vote_response`, emptiness is never flagged
    here: a blank field is a field-specific validity concern a caller checks
    separately, not an injection artifact.

    Args:
        text: The untrusted text to screen.

    Returns:
        :data:`RESPONSE_FAILURE_DELIMITER_FORGERY` if ``text`` forges an
        untrusted-data delimiter, else :data:`RESPONSE_FAILURE_TOOL_CALL_LURE`
        if it embeds any tool-call marker, else ``None``.
    """
    if _contains_delimiter(text):
        return RESPONSE_FAILURE_DELIMITER_FORGERY
    if any(marker in text for marker in TOOL_CALL_MARKERS):
        return RESPONSE_FAILURE_TOOL_CALL_LURE
    return None


def validate_vote_response(response: str) -> str | None:
    """Screen a vote response for injection artifacts, then schema (S8.5/S6.3).

    Checks run in a fixed first-failure order. The pre-existing SPEC S8.5
    injection screen runs *first* -- empty/whitespace-only, then (via
    :func:`screen_untrusted_text`) a forged delimiter token, then a tool-call
    lure -- so a response that is otherwise schema-valid JSON but carries an
    injection artifact is still flagged by the injection code, never masked by
    the schema layer. Only a response that clears the injection screen is then
    schema-validated (SPEC S6.3) as a structured vote object. The first failure
    wins.

    Args:
        response: The raw vote-completion text.

    Returns:
        A ``RESPONSE_FAILURE_*`` code for the first failing check, or ``None``
        when the response is both injection-clean and schema-valid.
    """
    if not response.strip():
        return RESPONSE_FAILURE_EMPTY
    injection_failure = screen_untrusted_text(response)
    if injection_failure is not None:
        return injection_failure
    return _schema_failure(response)


def parse_vote_response(response: str) -> ParsedVote:
    """Parse a validated vote response into a typed :class:`ParsedVote`, fail-closed.

    Validates first via :func:`validate_vote_response` and raises on any failure
    rather than silently coercing (mirroring
    ``windbreak.forecast.cassettes._reject_float``'s float-leaf ban): a float
    ``probability_ppm``, for instance, is rejected as non-integer, never
    truncated. A clean response is re-parsed into its exact fields.

    Args:
        response: The raw vote-completion text.

    Returns:
        The parsed, schema-valid vote.

    Raises:
        ValueError: If ``response`` fails the injection screen or the schema
            check; the message names the offending ``RESPONSE_FAILURE_*`` code.
    """
    failure = validate_vote_response(response)
    if failure is not None:
        raise ValueError(f"vote response failed validation: {failure}")
    payload = json.loads(response)
    return ParsedVote(
        probability_ppm=payload[_VOTE_PPM_KEY],
        rationale_summary=payload[_VOTE_RATIONALE_KEY],
        abstain=payload[_VOTE_ABSTAIN_KEY],
    )
