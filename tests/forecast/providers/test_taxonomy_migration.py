"""Live adapters raise the specific vote-failure taxonomy (issue #269, item 3).

PR #268 shipped :class:`~windbreak.forecast.providers.base.ProviderHTTPError`,
:class:`~windbreak.forecast.providers.base.ProviderMalformedResponseError`, and
the rest of the :class:`~windbreak.forecast.providers.base.ProviderVoteError`
taxonomy, but the live adapters kept raising the generic
:class:`~windbreak.forecast.providers.base.ProviderResponseRejectedError` for
*every* failure. That flattening is what makes the hardening inert:
:class:`~windbreak.forecast.providers.retry.RetryingProvider` classifies a
retryable transport fault by *type* (and, for HTTP, by ``status_code``), so an
adapter that reports a retryable ``503`` as an unretryable screen rejection can
never be retried at all.

These tests pin the differentiation at the adapter boundary:

* a non-2xx status raises :class:`ProviderHTTPError` carrying the *actual*
  status code, so :func:`~windbreak.forecast.providers.retry.is_retryable_status`
  can tell a transient ``503``/``429`` from a permanent ``400``;
* an unparseable body raises :class:`ProviderMalformedResponseError`, which is
  screen-side and must stay unretryable -- retrying only re-poisons it;
* the wire-format ``failure_code`` on each is *unchanged*, so the discard/ledger
  path keeps emitting exactly the codes it emitted before. The taxonomy is a
  finer type, not a new wire vocabulary.

Every adapter here is driven through a stub transport; nothing reaches a network.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.cassettes import LlmRequest
from windbreak.forecast.providers import (
    HttpResponse,
    ProviderHTTPError,
    ProviderMalformedResponseError,
    ProviderResponseRejectedError,
    ProviderVoteError,
    is_retryable_status,
)
from windbreak.forecast.providers.anthropic import AnthropicMessagesTransport
from windbreak.forecast.providers.openai import OpenAiChatTransport
from windbreak.forecast.sanitize import (
    RESPONSE_FAILURE_HTTP_STATUS,
    RESPONSE_FAILURE_MALFORMED_VOTE_JSON,
)

if TYPE_CHECKING:
    from windbreak.forecast.providers import HttpRequest

#: Endpoints the stub adapters below are pointed at (never dialed).
_ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

#: The response-token cap every adapter under test is built with.
_MAX_TOKENS = 1024

#: An arbitrary fixed prompt: these adapters relay `LlmRequest.prompt` verbatim.
_PROMPT_TEXT = "Estimate the resolution probability. Respond as JSON."


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

    def send(self, request: HttpRequest) -> HttpResponse:
        """Return the fixed response, ignoring `request`.

        Args:
            request: The (unused) HTTP request.

        Returns:
            The fixed `HttpResponse`.
        """
        del request
        return HttpResponse(self._status_code, self._body)


def _request() -> LlmRequest:
    """Build the single completion request every test below drives.

    Returns:
        A fixed `LlmRequest`.
    """
    return LlmRequest(
        provider="anthropic",
        model_version="claude-sonnet-4-5-20250929",
        prompt=_PROMPT_TEXT,
    )


def _anthropic(body: str, *, status_code: int) -> AnthropicMessagesTransport:
    """Build an Anthropic adapter over a stub returning `body`/`status_code`.

    Args:
        body: The fixed raw response body.
        status_code: The fixed HTTP status code.

    Returns:
        The adapter under test.
    """
    return AnthropicMessagesTransport(
        _StubHttpTransport(body, status_code=status_code),
        endpoint_url=_ANTHROPIC_ENDPOINT,
        max_tokens=_MAX_TOKENS,
    )


def _openai(body: str, *, status_code: int) -> OpenAiChatTransport:
    """Build an OpenAI adapter over a stub returning `body`/`status_code`.

    Args:
        body: The fixed raw response body.
        status_code: The fixed HTTP status code.

    Returns:
        The adapter under test.
    """
    return OpenAiChatTransport(
        _StubHttpTransport(body, status_code=status_code),
        endpoint_url=_OPENAI_ENDPOINT,
        max_tokens=_MAX_TOKENS,
    )


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_anthropic_retryable_status_raises_a_retryable_http_error(
    status_code: int,
) -> None:
    """A transient status surfaces as `ProviderHTTPError` the retry layer retries."""
    with pytest.raises(ProviderHTTPError) as excinfo:
        _anthropic("upstream unavailable", status_code=status_code).complete(_request())

    assert excinfo.value.status_code == status_code
    assert is_retryable_status(excinfo.value.status_code)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_anthropic_client_error_status_raises_an_unretryable_http_error(
    status_code: int,
) -> None:
    """A permanent 4xx carries its own status, which the retry layer refuses."""
    with pytest.raises(ProviderHTTPError) as excinfo:
        _anthropic("bad request", status_code=status_code).complete(_request())

    assert excinfo.value.status_code == status_code
    assert not is_retryable_status(excinfo.value.status_code)


def test_openai_non_success_status_raises_the_http_error_with_its_status() -> None:
    """The OpenAI adapter differentiates identically to the Anthropic one."""
    with pytest.raises(ProviderHTTPError) as excinfo:
        _openai("upstream unavailable", status_code=503).complete(_request())

    assert excinfo.value.status_code == 503


def test_http_status_failure_code_is_unchanged_by_the_finer_type() -> None:
    """The wire-format code stays `RESPONSE_FAILURE_HTTP_STATUS` for the ledger."""
    with pytest.raises(ProviderHTTPError) as excinfo:
        _anthropic("upstream unavailable", status_code=500).complete(_request())

    assert excinfo.value.failure_code == RESPONSE_FAILURE_HTTP_STATUS


def test_http_error_is_a_vote_error_so_the_pipeline_still_discards_it() -> None:
    """It stays a `ProviderVoteError`, so per-vote discard behaviour is intact."""
    with pytest.raises(ProviderVoteError):
        _anthropic("upstream unavailable", status_code=500).complete(_request())


def test_http_error_is_not_a_screen_side_rejection() -> None:
    """A transport fault is no longer conflated with a screen rejection.

    This is the discrimination the whole migration exists for: before #269 an
    ``except ProviderResponseRejectedError`` caught a retryable 503, which is
    precisely how the retry layer was rendered inert.
    """
    with pytest.raises(ProviderHTTPError) as excinfo:
        _anthropic("upstream unavailable", status_code=503).complete(_request())

    assert not isinstance(excinfo.value, ProviderResponseRejectedError)


@pytest.mark.parametrize("body", ["not even json", "[]", '"a bare string"', "17"])
def test_anthropic_unparseable_body_raises_the_malformed_error(body: str) -> None:
    """A 2xx whose body is not a JSON object is malformed, not an HTTP fault."""
    with pytest.raises(ProviderMalformedResponseError) as excinfo:
        _anthropic(body, status_code=200).complete(_request())

    assert excinfo.value.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_openai_unparseable_body_raises_the_malformed_error() -> None:
    """The OpenAI adapter reports a malformed envelope with the same type."""
    with pytest.raises(ProviderMalformedResponseError):
        _openai("not even json", status_code=200).complete(_request())


def test_malformed_error_is_not_an_http_error_so_it_is_never_retried() -> None:
    """A screen-side rejection must not be reachable as a transport fault."""
    with pytest.raises(ProviderMalformedResponseError) as excinfo:
        _anthropic("not even json", status_code=200).complete(_request())

    assert not isinstance(excinfo.value, ProviderHTTPError)


def test_a_missing_envelope_key_still_rejects_rather_than_reporting_http() -> None:
    """A well-formed 2xx JSON object missing its content stays screen-side."""
    with pytest.raises(ProviderVoteError) as excinfo:
        _anthropic(json.dumps({"unexpected": "shape"}), status_code=200).complete(
            _request()
        )

    assert not isinstance(excinfo.value, ProviderHTTPError)
