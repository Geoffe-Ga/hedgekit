"""Tests for windbreak.forecast.providers.search_live (issue #192).

Pins the live-search `SearchTransport` implementation's full contract:

* `search` builds a canonical, KEY-FREE POST request body -- sorted-key,
  space-free JSON of exactly `{query, max_results}` (mirroring
  `windbreak.forecast.providers.futuresearch._canonical_request_body`'s
  "no API-key material ever enters a hashed/persisted request" discipline) --
  and sends it through the injected `HttpTransport`.
* The response is processed in a fixed order: (a)
  `windbreak.forecast.sanitize.screen_untrusted_text` over the *entire* raw
  response body first (so a delimiter forgery or tool-call lure embedded
  anywhere in the raw bytes is caught before any JSON parsing is attempted);
  (b) JSON-parse with `parse_float=Decimal` and a non-finite-constant
  (`Infinity`/`-Infinity`/`NaN`) rejection hook; (c) extraction of a
  string-URL `results` array.
* Unlike `windbreak.forecast.providers.futuresearch.FutureSearchProvider`
  (which *raises* `ProviderResponseRejectedError` on a bad response), a
  non-2xx status, an injection-artifact body, or a malformed body all make
  `search` return `()` -- never raise -- because
  `windbreak.forecast.pipeline.bounded_web_research` calls `tools.search`
  with no surrounding `try`/`except` at all (an empty result tuple is simply
  "no candidate URL for this subquestion", the existing behavior for any
  search transport).

`windbreak/forecast/providers/search_live.py` does not exist yet, so importing
it fails collection with `ModuleNotFoundError` -- the expected Gate 1 RED
state for issue #192.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.cassettes import CassetteMissError
from windbreak.forecast.providers.http_cassettes import (
    ForbiddenLiveHttpTransport,
    HttpRequest,
    HttpResponse,
    RecordingHttpCassette,
    ReplayHttpCassette,
)
from windbreak.forecast.providers.search_live import (
    LiveSearchConfig,
    LiveSearchTransport,
)
from windbreak.forecast.sanitize import DATA_BLOCK_BEGIN

if TYPE_CHECKING:
    from pathlib import Path

#: The endpoint every test config below POSTs to.
_ENDPOINT_URL = "https://search.example/v1/search"

#: The query text every default test call searches for.
_QUERY = "Fed rate decision December 2024"


def _config(
    *, endpoint_url: str = _ENDPOINT_URL, max_results: int = 5
) -> LiveSearchConfig:
    """Build a `LiveSearchConfig` test double.

    Args:
        endpoint_url: The search endpoint URL override.
        max_results: The requested-result-count override.

    Returns:
        The constructed `LiveSearchConfig`.
    """
    return LiveSearchConfig(endpoint_url=endpoint_url, max_results=max_results)


def _results_body(urls: list[object]) -> str:
    """Build a raw `{"results": [...]}` response body.

    Args:
        urls: The raw JSON-encodable values for the `results` array.

    Returns:
        The serialized response body text.
    """
    return json.dumps({"results": urls})


class _StubHttpTransport:
    """A minimal `HttpTransport` double returning one fixed response verbatim."""

    def __init__(self, body: str, *, status_code: int = 200) -> None:
        """Store the fixed response every `send` call returns.

        Args:
            body: The fixed raw response body text to return.
            status_code: The fixed HTTP status code to return.
        """
        self._body = body
        self._status_code = status_code
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        """Record one call and return the fixed response, ignoring `request`.

        Args:
            request: The (recorded, otherwise unused) HTTP request.

        Returns:
            `HttpResponse(self._status_code, self._body)`, verbatim.
        """
        self.calls.append(request)
        return HttpResponse(self._status_code, self._body)


# --- Request shape: canonical, key-free, sorted-key POST body --------------------


def test_search_sends_a_canonical_sorted_key_post_body() -> None:
    """The request body is exactly the sorted-key, compact-separator canonical
    JSON of `{max_results, query}` -- never an API key or any other field.
    """
    transport = _StubHttpTransport(_results_body([]))
    live_search = LiveSearchTransport(transport, _config(max_results=7))

    live_search.search(_QUERY)

    expected_body = json.dumps(
        {"max_results": 7, "query": _QUERY}, sort_keys=True, separators=(",", ":")
    )
    assert transport.calls[0].method == "POST"
    assert transport.calls[0].url == _ENDPOINT_URL
    assert transport.calls[0].body == expected_body


def test_search_request_body_carries_no_headers_or_key_material() -> None:
    """The sent `HttpRequest` -- like every `HttpRequest` -- has no `headers`
    field at all, so an API key a live transport injects at send time can
    never be hashed into the request or persisted to a cassette.
    """
    transport = _StubHttpTransport(_results_body([]))
    live_search = LiveSearchTransport(transport, _config())

    live_search.search(_QUERY)

    field_names = {field.name for field in dataclasses.fields(HttpRequest)}
    assert field_names == {"method", "url", "body"}


def test_search_two_calls_with_different_queries_hash_differently() -> None:
    """Two distinct queries build two distinct, independently-hashable
    requests -- the query really is threaded into the canonical body.
    """
    transport = _StubHttpTransport(_results_body([]))
    live_search = LiveSearchTransport(transport, _config())

    live_search.search("query one")
    live_search.search("query two")

    assert transport.calls[0].request_hash() != transport.calls[1].request_hash()


# --- Happy path: URL extraction ---------------------------------------------------


def test_search_returns_the_results_array_as_a_url_tuple() -> None:
    """A clean response's `results` array becomes the returned URL tuple, in
    order.
    """
    urls = ["https://research.local/a", "https://research.local/b"]
    transport = _StubHttpTransport(_results_body(urls))
    live_search = LiveSearchTransport(transport, _config())

    result = live_search.search(_QUERY)

    assert result == ("https://research.local/a", "https://research.local/b")


def test_search_empty_results_array_returns_empty_tuple() -> None:
    """A well-formed but empty `results` array returns `()`, not an error."""
    transport = _StubHttpTransport(_results_body([]))
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


# --- Fail-closed to (), never raise: non-2xx, malformed, injection ---------------


@pytest.mark.parametrize("status_code", [404, 500, 503])
def test_search_non_2xx_status_returns_empty_tuple_without_raising(
    status_code: int,
) -> None:
    """A non-2xx status yields `()`, never a raised exception -- `search` has
    no `try`/`except` wrapper at its one call site.
    """
    transport = _StubHttpTransport(
        _results_body(["https://research.local/a"]), status_code=status_code
    )
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


def test_search_malformed_json_body_returns_empty_tuple() -> None:
    """A response body that is not valid JSON at all yields `()`."""
    transport = _StubHttpTransport("not json at all")
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


def test_search_results_field_missing_returns_empty_tuple() -> None:
    """A well-formed JSON object with no `results` key at all yields `()`."""
    transport = _StubHttpTransport(json.dumps({"status": "ok"}))
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


def test_search_results_not_a_list_returns_empty_tuple() -> None:
    """A `results` value that is present but not a JSON array yields `()`."""
    transport = _StubHttpTransport(json.dumps({"results": "not-a-list"}))
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


def test_search_results_containing_a_non_string_element_returns_empty_tuple() -> None:
    """A `results` array holding a non-string element (e.g. a nested object)
    yields `()` -- the whole response is treated as malformed, never a
    partially-extracted tuple.
    """
    transport = _StubHttpTransport(
        _results_body(["https://research.local/a", {"not": "a-url-string"}])
    )
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


def test_search_non_finite_json_constant_returns_empty_tuple() -> None:
    """A non-finite JSON constant (`Infinity`) anywhere in the body is
    rejected by the parse hook, never materialized as a Python float, and
    yields `()`.
    """
    body = '{"results": [], "score": Infinity}'
    transport = _StubHttpTransport(body)
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


def test_search_response_body_not_a_json_object_returns_empty_tuple() -> None:
    """A response body that is valid JSON but not an object (a bare array)
    yields `()`.
    """
    transport = _StubHttpTransport("[]")
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


# --- Fixed order: whole-body injection screen runs before JSON parsing ----------


def test_search_delimiter_forgery_in_a_valid_url_string_yields_empty_tuple() -> None:
    """A `results` entry that is syntactically a valid JSON string but embeds
    a literal untrusted-data delimiter token is still caught: the raw-body
    injection screen runs *before* any JSON-structure trust is extended, so
    this response yields `()` rather than a "clean" URL tuple containing the
    delimiter-bearing string.
    """
    poisoned_url = f"https://research.local/{DATA_BLOCK_BEGIN}"
    body = _results_body([poisoned_url])
    assert json.loads(body)  # sanity: the body is syntactically valid JSON.
    transport = _StubHttpTransport(body)
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


def test_search_json_escaped_delimiter_in_a_url_string_yields_empty_tuple() -> None:
    """A delimiter token hidden from the raw-body substring screen via JSON
    ``\\uXXXX`` escapes -- which ``json.dumps`` never emits but a hostile
    search server can -- must still yield ``()``.

    The raw body carries ``\\u003c\\u003c\\u003cUNTRUSTED-DATA`` rather than the
    literal ``<<<UNTRUSTED-DATA``, so the whole-body ``screen_untrusted_text``
    substring check passes; only after ``json.loads`` decodes the escapes does
    the ``results`` URL carry the real delimiter. Left unscreened, that URL
    would flow through fetch/verify into ``sanitize.wrap_data_block``, which
    raises an unhandled ``ValueError`` that echoes the raw attacker URL --
    crashing the run and leaking raw untrusted text into the traceback. The
    post-decode URL screen keeps the module's "every failure returns ()"
    contract instead: discard-not-repair, fingerprint-only.
    """
    # Encode every "<" as its six-character JSON unicode escape (backslash, u,
    # 0, 0, 3, c) -- a form json.dumps never emits but a hostile server can --
    # so the raw body hides the delimiter from a literal substring screen.
    literal_url = f"https://research.local/{DATA_BLOCK_BEGIN}"
    escaped_url = literal_url.replace("<", "\\u003c")
    body = f'{{"results": ["{escaped_url}"]}}'
    # The raw body does NOT literally contain the delimiter (it is escaped)...
    assert DATA_BLOCK_BEGIN not in body
    # ...yet json.loads decodes the escapes back into the real delimiter token.
    assert json.loads(body)["results"][0] == literal_url
    transport = _StubHttpTransport(body)
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


def test_search_tool_call_marker_anywhere_in_body_yields_empty_tuple() -> None:
    """A tool-call-lure marker anywhere in the raw response body is caught by
    the whole-body screen, even outside the `results` array.
    """
    body = json.dumps({"results": ["https://research.local/a"], "extra": "tool"})
    transport = _StubHttpTransport(body)
    live_search = LiveSearchTransport(transport, _config())

    assert live_search.search(_QUERY) == ()


# --- Record/replay: composes with the offline harness ----------------------------


def test_search_over_replay_http_cassette_round_trips(tmp_path: Path) -> None:
    """Recording a `LiveSearchTransport` run, then replaying the persisted
    cassette, reproduces the identical result tuple.
    """
    cassette_path = tmp_path / "search_cassette.json"
    recorder = RecordingHttpCassette(
        transport=_StubHttpTransport(_results_body(["https://research.local/a"])),
        path=cassette_path,
    )
    recorded = LiveSearchTransport(recorder, _config()).search(_QUERY)

    replay_search = LiveSearchTransport(
        ReplayHttpCassette.from_path(cassette_path), _config()
    )
    replayed = replay_search.search(_QUERY)

    assert replayed == recorded == ("https://research.local/a",)


def test_search_over_empty_replay_cassette_raises_cassette_miss_error() -> None:
    """An unrecorded request fails closed via `CassetteMissError` -- proving
    `search` really does call `transport.send` rather than short-circuiting.
    A cassette miss is a harness-level failure, distinct from the
    fail-to-empty-tuple contract for a *served* bad response.
    """
    live_search = LiveSearchTransport(ReplayHttpCassette({}), _config())

    with pytest.raises(CassetteMissError):
        live_search.search(_QUERY)


def test_search_over_forbidden_live_transport_fails_closed() -> None:
    """Driving `LiveSearchTransport` directly over `ForbiddenLiveHttpTransport`
    fails closed, confirming the transport is genuinely invoked.
    """
    from windbreak.forecast.cassettes import LiveCallForbiddenError

    live_search = LiveSearchTransport(ForbiddenLiveHttpTransport(), _config())

    with pytest.raises(LiveCallForbiddenError):
        live_search.search(_QUERY)
