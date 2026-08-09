"""Tests for windbreak.forecast.providers.anthropic (issue #191).

Pins `AnthropicMessagesTransport`: an `LlmTransport` (SPEC S8.9) over the
Anthropic Messages HTTP API. Its request body is a deterministic, canonical
(sorted-key, no-space-separator) JSON object naming exactly `max_tokens`,
`messages` (a single `user` turn carrying the prompt verbatim),
`model` (the request's pinned `model_version`), and an *integer* `temperature`
of `0` -- never a float `0.0`, so the probability/money-path float ban is
respected even in a request body that never itself carries a probability.

A non-2xx HTTP status is rejected fast, before any body parsing, via
`ProviderResponseRejectedError(RESPONSE_FAILURE_HTTP_STATUS, fingerprint)` --
fingerprint-only, the raw response text never leaking into the exception. A
clean response is parsed with `json.loads(..., parse_float=Decimal,
parse_constant=<reject Infinity/NaN>)`, float-free and fail-closed on any
non-finite JSON constant anywhere in the envelope. `content[0]["text"]` (the
first `type == "text"` content block) is extracted and returned verbatim as
the completion text; any other envelope shape (not an object, missing/
malformed `content`, a non-text block, a missing/non-string `text`) is
rejected as `RESPONSE_FAILURE_MALFORMED_VOTE_JSON`.

`windbreak.forecast.providers.anthropic` does not exist yet, so importing it
below fails collection with `ModuleNotFoundError: No module named
'windbreak.forecast.providers.anthropic'` -- the expected Gate 1 RED state
for issue #191.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.cassettes import LlmRequest
from windbreak.forecast.providers import (
    EnsembleMember,
    FixtureVoteProvider,
    HttpResponse,
    ProviderForecast,
    ProviderResponseRejectedError,
)
from windbreak.forecast.providers.anthropic import AnthropicMessagesTransport
from windbreak.forecast.sanitize import (
    RESPONSE_FAILURE_HTTP_STATUS,
    RESPONSE_FAILURE_MALFORMED_VOTE_JSON,
)

if TYPE_CHECKING:
    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.providers import HttpRequest
    from windbreak.forecast.records import BaselineQuoteSnapshot

#: The endpoint every test transport below POSTs to.
_ENDPOINT_URL = "https://api.anthropic.com/v1/messages"

#: The `max_tokens` cap every test transport below is constructed with.
_MAX_TOKENS = 1024

#: The pinned Anthropic model version exercised by these tests, matching the
#: new #191 default vote-ensemble member's `model_version`.
_MODEL_VERSION = "claude-sonnet-4-5-20250929"

#: An arbitrary, fixed prompt text: `AnthropicMessagesTransport` only ever
#: relays `LlmRequest.prompt` verbatim, so its own tests never need a real
#: `build_vote_prompt` call.
_PROMPT_TEXT = "Estimate the resolution probability. Respond as JSON."

#: A valid #184 vote-schema JSON string, used as the "model's" completion
#: text in the end-to-end `FixtureVoteProvider` test.
_VALID_VOTE_JSON = (
    '{"probability_ppm": 611111, "rationale_summary": "steady corroborating '
    'evidence", "abstain": false}'
)


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


def _envelope_body(text: str = "a completion") -> str:
    """Build a well-formed Anthropic Messages response envelope.

    Args:
        text: The completion text embedded in the sole `text` content block.

    Returns:
        The raw JSON response body text.
    """
    return json.dumps({"content": [{"type": "text", "text": text}]})


def _transport(body: str, *, status_code: int = 200) -> AnthropicMessagesTransport:
    """Build an `AnthropicMessagesTransport` over a stub returning `body`.

    Args:
        body: The fixed raw response body the underlying stub returns.
        status_code: The fixed HTTP status code the stub returns.

    Returns:
        The constructed transport, wired to a fresh `_StubHttpTransport`.
    """
    return AnthropicMessagesTransport(
        _StubHttpTransport(body, status_code=status_code),
        endpoint_url=_ENDPOINT_URL,
        max_tokens=_MAX_TOKENS,
    )


def _request(prompt: str = _PROMPT_TEXT) -> LlmRequest:
    """Build a fixed `LlmRequest` for the transport's own unit tests.

    Args:
        prompt: The prompt text to carry, verbatim.

    Returns:
        An `LlmRequest` pinned to `_MODEL_VERSION`.
    """
    return LlmRequest(provider="anthropic", model_version=_MODEL_VERSION, prompt=prompt)


# --- Happy path: returns the exact embedded completion text ----------------------


def test_complete_happy_path_returns_the_exact_embedded_completion_text() -> None:
    """A clean envelope's `content[0]["text"]` is returned verbatim."""
    transport = _transport(_envelope_body("the exact completion text"))

    result = transport.complete(_request())

    assert result == "the exact completion text"


# --- Request body: canonical, deterministic, integer temperature -----------------


def test_request_body_is_canonical_json_with_integer_temperature() -> None:
    """The request body is exactly the sorted-key, compact-separator canonical
    JSON of `{max_tokens, messages, model, temperature}`, with `temperature`
    an *integer* `0` -- never a float `0.0`.
    """
    http_transport = _StubHttpTransport(_envelope_body())
    transport = AnthropicMessagesTransport(
        http_transport, endpoint_url=_ENDPOINT_URL, max_tokens=_MAX_TOKENS
    )

    transport.complete(_request())

    expected_body = json.dumps(
        {
            "max_tokens": _MAX_TOKENS,
            "messages": [{"content": _PROMPT_TEXT, "role": "user"}],
            "model": _MODEL_VERSION,
            "temperature": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert http_transport.calls[0].method == "POST"
    assert http_transport.calls[0].url == _ENDPOINT_URL
    assert http_transport.calls[0].body == expected_body
    assert '"temperature":0' in http_transport.calls[0].body
    assert '"temperature":0.0' not in http_transport.calls[0].body


def test_two_calls_with_equal_requests_build_byte_identical_bodies() -> None:
    """Two `complete` calls over an equal `LlmRequest` build byte-identical
    request bodies, hashing to the identical `request_hash()`.
    """
    stub_a = _StubHttpTransport(_envelope_body())
    stub_b = _StubHttpTransport(_envelope_body())
    transport_a = AnthropicMessagesTransport(
        stub_a, endpoint_url=_ENDPOINT_URL, max_tokens=_MAX_TOKENS
    )
    transport_b = AnthropicMessagesTransport(
        stub_b, endpoint_url=_ENDPOINT_URL, max_tokens=_MAX_TOKENS
    )
    request = _request()

    transport_a.complete(request)
    transport_b.complete(request)

    assert stub_a.calls[0].body == stub_b.calls[0].body
    assert stub_a.calls[0].request_hash() == stub_b.calls[0].request_hash()


# --- Non-2xx status: rejected fast, fingerprint-only --------------------------------


def test_non_2xx_status_is_rejected_with_http_status_failure_code() -> None:
    """A non-2xx status raises `ProviderResponseRejectedError` carrying
    `RESPONSE_FAILURE_HTTP_STATUS` and a fingerprint of the raw body -- never
    the raw body text itself.
    """
    body = "irrelevant body content that must never leak into the exception"
    transport = _transport(body, status_code=500)

    with pytest.raises(ProviderResponseRejectedError) as excinfo:
        transport.complete(_request())

    assert excinfo.value.failure_code == RESPONSE_FAILURE_HTTP_STATUS
    assert body not in str(excinfo.value)
    assert len(excinfo.value.response_fingerprint) == 64
    assert excinfo.value.response_fingerprint in str(excinfo.value)
    assert (
        excinfo.value.response_fingerprint
        == hashlib.sha256(body.encode("utf-8")).hexdigest()
    )


# --- Malformed envelopes: rejected as malformed_vote_json -------------------------


@pytest.mark.parametrize(
    "body",
    [
        "not even json",
        "[]",
        '{"content": "not-a-list"}',
        '{"content": []}',
        "{}",
        '{"content": [123]}',
        '{"content": [{"type": "image", "text": "x"}]}',
        '{"content": [{"type": "text"}]}',
        '{"content": [{"type": "text", "text": 123}]}',
    ],
    ids=[
        "not-json-at-all",
        "bare-array",
        "content-not-a-list",
        "content-empty-list",
        "content-key-missing",
        "content-element-not-an-object",
        "content-element-wrong-type",
        "content-element-missing-text",
        "content-element-text-not-a-string",
    ],
)
def test_malformed_envelope_is_rejected_as_malformed_vote_json(body: str) -> None:
    """Every documented malformed-envelope shape is rejected as
    `RESPONSE_FAILURE_MALFORMED_VOTE_JSON`, never silently coerced.
    """
    transport = _transport(body)

    with pytest.raises(ProviderResponseRejectedError) as excinfo:
        transport.complete(_request())

    assert excinfo.value.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_non_finite_json_constant_anywhere_in_envelope_is_rejected() -> None:
    """A non-finite JSON constant (`Infinity`/`-Infinity`/`NaN`) anywhere in
    the envelope is rejected, never materialized as a Python float.
    """
    body = (
        '{"content": [{"type": "text", "text": "ok"}], "usage": {"tokens": Infinity}}'
    )
    transport = _transport(body)

    with pytest.raises(ProviderResponseRejectedError) as excinfo:
        transport.complete(_request())

    assert excinfo.value.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


# --- End-to-end: FixtureVoteProvider over the real adapter ------------------------


def test_fixture_vote_provider_over_anthropic_transport_yields_validated_forecast(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """`FixtureVoteProvider(AnthropicMessagesTransport(...), member)` end-to-end:
    a clean, schema-valid completion embedded in a clean envelope yields a
    fully-validated `ProviderForecast` carrying the member's provenance and a
    fingerprint of the *extracted completion text* (never the whole envelope).
    """
    transport = _transport(_envelope_body(_VALID_VOTE_JSON))
    member = EnsembleMember("anthropic", _MODEL_VERSION, "2025-07-31")
    provider = FixtureVoteProvider(transport, member)

    result = provider.forecast(market, baseline, 0, ())

    assert isinstance(result, ProviderForecast)
    assert result.probability_ppm == 611_111
    assert result.rationale_summary == "steady corroborating evidence"
    assert result.provider == "anthropic"
    assert result.model_version == _MODEL_VERSION
    assert result.training_cutoff == "2025-07-31"
    assert (
        result.response_fingerprint
        == hashlib.sha256(_VALID_VOTE_JSON.encode("utf-8")).hexdigest()
    )
