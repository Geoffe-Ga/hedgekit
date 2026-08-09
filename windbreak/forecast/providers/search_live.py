"""A live-shaped :class:`SearchTransport` over the HTTP record/replay seam (S8.9).

:class:`LiveSearchTransport` obtains candidate research URLs for a subquestion
through the injected
:class:`~windbreak.forecast.providers.http_cassettes.HttpTransport` seam -- fake,
recording, replay, or forbidden-live -- so a live search composes with the
offline record/replay harness exactly like
:class:`~windbreak.forecast.providers.futuresearch.FutureSearchProvider`. The
transport is dependency-injected, so this module never imports ``requests`` (the
live recorder lives in ``scripts/record_research_cassettes.py``) and, per the
SPEC S8.3 sandbox boundary, never imports ``windbreak.config``. It is stdlib-only
and float-free.

The request body is a canonical, KEY-FREE JSON object of exactly
``{max_results, query}`` (sorted keys, space-free separators) -- mirroring
:func:`windbreak.forecast.providers.futuresearch._canonical_request_body`'s "no
API-key material ever enters a hashed or persisted request" discipline: an API
key a live transport injects at send time lives only in a send-time header, of
which :class:`~windbreak.forecast.providers.http_cassettes.HttpRequest` has
none.

The response is processed in a fixed order, mirroring ``futuresearch``'s
"injection screen before schema" precedent, but -- unlike ``futuresearch``,
which *raises* -- every failure returns ``()`` rather than raising, because
:func:`windbreak.forecast.pipeline.bounded_web_research` calls ``tools.search``
with no surrounding ``try``/``except``: an empty tuple simply means "no
candidate URL for this subquestion".

1. A non-2xx status returns ``()``.
2. :func:`windbreak.forecast.sanitize.screen_untrusted_text` over the *entire*
   raw body -- a delimiter forgery or tool-call lure anywhere in the bytes
   returns ``()`` before any JSON-structure trust is extended.
3. JSON-parse with ``parse_float=Decimal`` and a non-finite-constant-rejecting
   hook; a malformed body, a non-object body, or a non-finite constant returns
   ``()``.
4. ``results`` must be a list of ``str`` -- returned as a tuple, else ``()``
   (a partially-extractable array is treated as wholly malformed, never
   partially trusted).
5. Each *decoded* ``results`` URL is re-screened with
   :func:`~windbreak.forecast.sanitize.screen_untrusted_text`: the step-2
   screen runs on the raw bytes, so a delimiter or tool-call marker smuggled
   through JSON ``\\uXXXX`` escapes (which ``json.loads`` decodes into a real
   token) is only visible post-parse; any tainted URL fails the whole batch to
   ``()`` before it can reach the vote prompt's ``wrap_data_block``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Final, NoReturn

from windbreak.forecast.providers.http_cassettes import HttpRequest
from windbreak.forecast.sanitize import screen_untrusted_text

if TYPE_CHECKING:
    from windbreak.forecast.providers.http_cassettes import HttpTransport

#: The HTTP method every search request uses.
_SEARCH_METHOD = "POST"

#: The request-body field naming the query text.
_QUERY_KEY = "query"

#: The request-body field naming the requested result count.
_MAX_RESULTS_KEY = "max_results"

#: The response-body field holding the candidate-URL array.
_RESULTS_KEY = "results"

#: Inclusive lower bound of the HTTP success (2xx) status range.
_HTTP_SUCCESS_MIN: Final = 200

#: Exclusive upper bound of the HTTP success (2xx) status range.
_HTTP_SUCCESS_MAX_EXCLUSIVE: Final = 300


@dataclass(frozen=True, slots=True)
class LiveSearchConfig:
    """The pinned configuration one :class:`LiveSearchTransport` searches under.

    Attributes:
        endpoint_url: The search endpoint every request is POSTed to.
        max_results: The requested result count, threaded into the canonical
            request body.
    """

    endpoint_url: str
    max_results: int


def _reject_constant(token: str) -> NoReturn:
    """Reject a non-finite JSON constant token (``Infinity``/``NaN``).

    Installed as ``json.loads(..., parse_constant=...)`` so a non-standard
    constant -- which ``json.loads`` would otherwise materialize as a real
    Python ``float`` -- fails the parse instead of smuggling a float onto the
    money/probability path.

    Args:
        token: The non-finite constant token the parser encountered.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"non-finite JSON constant is banned, got {token!r}")


def _canonical_request_body(query: str, max_results: int) -> str:
    """Serialize the query and result count into a deterministic request body.

    Keys are sorted and separators are space-free, matching the ledger's
    canonical JSON form so the body (and thus the request hash) is byte-stable
    and carries no key material.

    Args:
        query: The subquestion text to search for.
        max_results: The requested result count.

    Returns:
        The canonical JSON request-body text.
    """
    return json.dumps(
        {_MAX_RESULTS_KEY: max_results, _QUERY_KEY: query},
        sort_keys=True,
        separators=(",", ":"),
    )


def _extract_results(raw_body: str) -> tuple[str, ...]:
    """Extract the string-URL ``results`` tuple from a response body, or ``()``.

    Runs the whole-body injection screen first, then a float-free JSON parse,
    then the ``results`` shape check; any failure returns ``()`` rather than
    raising.

    Args:
        raw_body: The raw response body text.

    Returns:
        The candidate URLs in order, or ``()`` on any screen/parse/shape failure.
    """
    if screen_untrusted_text(raw_body) is not None:
        return ()
    try:
        payload = json.loads(
            raw_body, parse_float=Decimal, parse_constant=_reject_constant
        )
    except ValueError:
        return ()
    if not isinstance(payload, dict):
        return ()
    results = payload.get(_RESULTS_KEY)
    if not isinstance(results, list) or not all(
        isinstance(entry, str) for entry in results
    ):
        return ()
    # Re-screen each *decoded* URL: the whole-body screen above runs on the raw
    # bytes, so a delimiter/tool-call marker smuggled through JSON ``\uXXXX``
    # escapes (which ``json.loads`` decodes into a real token) would otherwise
    # slip past it and reach the vote prompt's ``wrap_data_block``. Any tainted
    # URL fails the whole batch closed -- discard-not-repair, fingerprint-only.
    if any(screen_untrusted_text(entry) is not None for entry in results):
        return ()
    return tuple(results)


class LiveSearchTransport:
    """A :class:`SearchTransport` obtaining candidate URLs over the HTTP seam."""

    def __init__(self, transport: HttpTransport, config: LiveSearchConfig) -> None:
        """Bind the HTTP transport and pinned search configuration.

        Args:
            transport: The HTTP transport (fake, recording, replay, or
                forbidden-live) each search is sent through.
            config: The pinned endpoint and result-count configuration.
        """
        self._transport = transport
        self._config = config

    def search(self, query: str) -> tuple[str, ...]:
        """Search for ``query`` and return its candidate URLs, fail-to-empty.

        Args:
            query: The subquestion text to search for.

        Returns:
            The candidate URLs in order, or ``()`` when the response is non-2xx,
            injection-tainted, malformed, or carries a non-string-URL ``results``
            array. Never raises for a *served* bad response.
        """
        request = HttpRequest(
            method=_SEARCH_METHOD,
            url=self._config.endpoint_url,
            body=_canonical_request_body(query, self._config.max_results),
        )
        response = self._transport.send(request)
        if not (
            _HTTP_SUCCESS_MIN <= response.status_code < _HTTP_SUCCESS_MAX_EXCLUSIVE
        ):
            return ()
        return _extract_results(response.body)
