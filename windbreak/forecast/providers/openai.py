"""An :class:`LlmTransport` over the OpenAI chat-completions API (issue #191).

:class:`OpenAiChatTransport` adapts the OpenAI chat-completions endpoint to the
forecast engine's :class:`~windbreak.forecast.cassettes.LlmTransport` seam (SPEC
S8.9), mirroring
:class:`~windbreak.forecast.providers.anthropic.AnthropicMessagesTransport`
exactly except for the response envelope's shape. Its request body is the same
deterministic, canonical JSON object naming exactly ``max_tokens``, ``messages``
(a single ``user`` turn carrying the prompt verbatim), ``model``, and an
*integer* ``temperature`` of ``0`` -- shared with the Anthropic adapter through
:mod:`windbreak.forecast.providers._llm_http`.

The response is fetched, HTTP-status-screened, and JSON-parsed by that shared
plumbing (float-free, fail-closed on any non-finite constant); this module adds
only the OpenAI-specific extraction of ``choices[0]["message"]["content"]``. Any
other envelope shape is rejected as
:data:`~windbreak.forecast.sanitize.RESPONSE_FAILURE_MALFORMED_VOTE_JSON`,
carrying only a fingerprint of the untrusted text, never the raw bytes.

The module is stdlib-only and float-free, never imports ``windbreak.config``
(SPEC S8.3), never imports ``requests`` (the transport is injected), and never
reads the process environment or names an API key -- the live recorder in
``scripts/record_vote_cassettes.py`` owns all of that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from windbreak.forecast.cassettes import Completion
from windbreak.forecast.providers._llm_http import (
    build_chat_request_body,
    extract_usage,
    fetch_envelope,
    require_first_element,
    require_object,
    require_text,
)

if TYPE_CHECKING:
    from windbreak.forecast.cassettes import LlmRequest
    from windbreak.forecast.providers.http_cassettes import HttpTransport

#: The OpenAI chat-completions endpoint every request is POSTed to by default.
OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

#: The default response token cap requested from the model.
_MAX_TOKENS = 1024

#: Response envelope keys.
_CHOICES_KEY = "choices"
_MESSAGE_KEY = "message"
_CONTENT_KEY = "content"

#: OpenAI's ``usage`` block keys for billed input and output tokens.
_PROMPT_TOKENS_KEY = "prompt_tokens"
_COMPLETION_TOKENS_KEY = "completion_tokens"


def _extract_completion_text(payload: dict[str, object], fingerprint: str) -> str:
    """Extract ``choices[0]["message"]["content"]`` from a chat-completion body.

    Args:
        payload: The parsed response object.
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        The first choice's message content, verbatim.

    Raises:
        ProviderMalformedResponseError: If ``choices`` is missing/empty/not an
            array, its first element or its ``message`` is not an object, or the
            message's ``content`` is missing or not a string.
    """
    choice = require_object(
        require_first_element(payload, _CHOICES_KEY, fingerprint), fingerprint
    )
    message = require_object(choice.get(_MESSAGE_KEY), fingerprint)
    return require_text(message.get(_CONTENT_KEY), fingerprint)


class OpenAiChatTransport:
    """An :class:`LlmTransport` over the OpenAI chat-completions HTTP API."""

    def __init__(
        self,
        http_transport: HttpTransport,
        *,
        endpoint_url: str = OPENAI_CHAT_ENDPOINT,
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        """Bind the HTTP transport, endpoint, and response token cap.

        Args:
            http_transport: The HTTP transport (fake, recording, replay, or
                forbidden-live) each request is sent through.
            endpoint_url: The chat-completions endpoint to POST to.
            max_tokens: The response token cap to request.
        """
        self._http = http_transport
        self._endpoint_url = endpoint_url
        self._max_tokens = max_tokens

    def complete(self, request: LlmRequest) -> Completion:
        """Send one completion request and return the model's response.

        Args:
            request: The completion request whose prompt/model drive the call.

        Returns:
            The extracted ``choices[0]["message"]["content"]`` text, verbatim,
            paired with the envelope's reported ``usage`` token counts -- or
            ``None`` usage when the envelope reported none readably, which
            costs the fail-closed unmetered figure rather than nothing
            (issue #451).

        Raises:
            ProviderHTTPError: On a non-2xx status, carrying that status code.
            ProviderMalformedResponseError: On a malformed
                body, a non-finite JSON constant, or an unexpected envelope
                shape; the error carries the failure code and fingerprint only.
        """
        body = build_chat_request_body(request, max_tokens=self._max_tokens)
        envelope = fetch_envelope(
            self._http, endpoint_url=self._endpoint_url, body=body
        )
        return Completion(
            text=_extract_completion_text(envelope.payload, envelope.fingerprint),
            usage=extract_usage(
                envelope.payload,
                input_key=_PROMPT_TOKENS_KEY,
                output_key=_COMPLETION_TOKENS_KEY,
            ),
        )
