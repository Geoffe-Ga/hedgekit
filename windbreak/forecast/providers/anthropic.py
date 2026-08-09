"""An :class:`LlmTransport` over the Anthropic Messages HTTP API (issue #191).

:class:`AnthropicMessagesTransport` adapts the Anthropic Messages endpoint to the
forecast engine's :class:`~windbreak.forecast.cassettes.LlmTransport` seam (SPEC
S8.9), so a :class:`~windbreak.forecast.providers.fixture.FixtureVoteProvider`
can drive a live Anthropic model through the same path a fake, recording, or
replay transport uses. Its request body is a deterministic, canonical
(sorted-key, no-space-separator) JSON object naming exactly ``max_tokens``,
``messages`` (a single ``user`` turn carrying the prompt verbatim), ``model``,
and an *integer* ``temperature`` of ``0`` -- shared with the OpenAI adapter
through :mod:`windbreak.forecast.providers._llm_http`.

The response is fetched, HTTP-status-screened, and JSON-parsed by that shared
plumbing (float-free, fail-closed on any non-finite constant); this module adds
only the Anthropic-specific extraction of ``content[0]["text"]`` (the first
``type == "text"`` content block). Any other envelope shape is rejected as
:data:`~windbreak.forecast.sanitize.RESPONSE_FAILURE_MALFORMED_VOTE_JSON`,
carrying only a fingerprint of the untrusted text, never the raw bytes.

The module is stdlib-only and float-free, never imports ``windbreak.config``
(SPEC S8.3), never imports ``requests`` (the transport is injected), and never
reads the process environment or names an API key -- the live recorder in
``scripts/record_vote_cassettes.py`` owns all of that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from windbreak.forecast.providers._llm_http import (
    build_chat_request_body,
    fetch_envelope,
    reject,
    require_first_element,
    require_object,
    require_text,
)
from windbreak.forecast.sanitize import RESPONSE_FAILURE_MALFORMED_VOTE_JSON

if TYPE_CHECKING:
    from windbreak.forecast.cassettes import LlmRequest
    from windbreak.forecast.providers.http_cassettes import HttpTransport

#: The Anthropic Messages endpoint every request is POSTed to by default.
ANTHROPIC_MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"

#: The default response token cap requested from the model.
_MAX_TOKENS = 1024

#: Response content-block keys, and the sole block ``type`` this adapter reads.
_CONTENT_KEY = "content"
_TYPE_KEY = "type"
_TEXT_KEY = "text"
_TEXT_TYPE = "text"


def _extract_completion_text(payload: dict[str, object], fingerprint: str) -> str:
    """Extract ``content[0]["text"]`` from an Anthropic Messages envelope.

    Args:
        payload: The parsed response object.
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        The first text content block's text, verbatim.

    Raises:
        ProviderResponseRejectedError: If ``content`` is missing/empty/not an
            array, its first element is not an object, that object's ``type`` is
            not ``"text"``, or its ``text`` is missing or not a string.
    """
    block = require_object(
        require_first_element(payload, _CONTENT_KEY, fingerprint), fingerprint
    )
    if block.get(_TYPE_KEY) != _TEXT_TYPE:
        reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    return require_text(block.get(_TEXT_KEY), fingerprint)


class AnthropicMessagesTransport:
    """An :class:`LlmTransport` over the Anthropic Messages HTTP API."""

    def __init__(
        self,
        http_transport: HttpTransport,
        *,
        endpoint_url: str = ANTHROPIC_MESSAGES_ENDPOINT,
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        """Bind the HTTP transport, endpoint, and response token cap.

        Args:
            http_transport: The HTTP transport (fake, recording, replay, or
                forbidden-live) each request is sent through.
            endpoint_url: The Messages endpoint to POST to.
            max_tokens: The response token cap to request.
        """
        self._http = http_transport
        self._endpoint_url = endpoint_url
        self._max_tokens = max_tokens

    def complete(self, request: LlmRequest) -> str:
        """Send one completion request and return the model's text response.

        Args:
            request: The completion request whose prompt/model drive the call.

        Returns:
            The extracted ``content[0]["text"]`` completion text, verbatim.

        Raises:
            ProviderResponseRejectedError: On a non-2xx status, a malformed
                body, a non-finite JSON constant, or an unexpected envelope
                shape; the error carries the failure code and fingerprint only.
        """
        body = build_chat_request_body(request, max_tokens=self._max_tokens)
        envelope = fetch_envelope(
            self._http, endpoint_url=self._endpoint_url, body=body
        )
        return _extract_completion_text(envelope.payload, envelope.fingerprint)
