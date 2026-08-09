"""Shared HTTP plumbing for the chat-completion LLM transports (issue #191).

:class:`~windbreak.forecast.providers.anthropic.AnthropicMessagesTransport` and
:class:`~windbreak.forecast.providers.openai.OpenAiChatTransport` speak two
different response-envelope shapes but share an identical request body and an
identical fetch/screen/parse pipeline. That common, float-sensitive plumbing
lives here once -- built canonical-JSON request body, HTTP-status fast-reject,
and a float-free JSON parse (``parse_float=Decimal``, ``parse_constant`` banning
``Infinity``/``NaN``) -- so neither adapter re-implements it. Each adapter keeps
only its own envelope-extraction step, mirroring
:mod:`windbreak.forecast.providers.futuresearch`'s idioms.

The module is stdlib-only and float-free -- it sits on the probability/money
path guarded by ``scripts/lint_no_floats.py`` -- and never imports
``windbreak.config`` (SPEC S8.3) or ``requests`` (the transport is injected).
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

from windbreak.forecast.providers.base import (
    ProviderResponseRejectedError,
    fingerprint_response,
)
from windbreak.forecast.providers.http_cassettes import HttpRequest
from windbreak.forecast.sanitize import (
    RESPONSE_FAILURE_HTTP_STATUS,
    RESPONSE_FAILURE_MALFORMED_VOTE_JSON,
)

if TYPE_CHECKING:
    from windbreak.forecast.cassettes import LlmRequest
    from windbreak.forecast.providers.http_cassettes import HttpTransport

#: The HTTP method every chat-completion request uses.
_REQUEST_METHOD = "POST"

#: Inclusive lower bound of the HTTP success (2xx) status range: a response
#: below it is rejected fast, before any body parsing.
_HTTP_SUCCESS_MIN: Final = 200

#: Exclusive upper bound of the HTTP success (2xx) status range: a response
#: reaching it (300+) is rejected fast, before any body parsing.
_HTTP_SUCCESS_MAX_EXCLUSIVE: Final = 300

#: Request-body top-level keys (a closed, deterministic set).
_MAX_TOKENS_KEY = "max_tokens"
_MESSAGES_KEY = "messages"
_MODEL_KEY = "model"
_TEMPERATURE_KEY = "temperature"

#: Per-message keys, and the single user turn's role.
_CONTENT_KEY = "content"
_ROLE_KEY = "role"
_USER_ROLE = "user"

#: The fixed sampling temperature stamped on every request: an *integer* ``0``,
#: never a float ``0.0``, so the money/probability float ban holds even in a
#: request body that never itself carries a probability.
_TEMPERATURE = 0


class ChatEnvelope(NamedTuple):
    """A fetched, HTTP-screened, JSON-parsed chat-completion response envelope.

    Attributes:
        payload: The parsed top-level JSON object.
        fingerprint: The sha256 fingerprint of the raw response body, for any
            downstream rejection -- fingerprint-only, never the raw text.
    """

    payload: dict[str, object]
    fingerprint: str


def reject(failure_code: str, fingerprint: str) -> NoReturn:
    """Raise a fingerprint-only rejection, never leaking the raw response text.

    Args:
        failure_code: The ``RESPONSE_FAILURE_*`` code describing the failure.
        fingerprint: The rejected response's sha256 fingerprint.

    Raises:
        ProviderResponseRejectedError: Always.
    """
    raise ProviderResponseRejectedError(failure_code, fingerprint)


def reject_constant(token: str) -> NoReturn:
    """Reject a non-finite JSON constant token (``Infinity``/``NaN``).

    Installed as ``json.loads(..., parse_constant=...)`` so a non-standard
    constant -- which ``json.loads`` would otherwise materialize as a real
    Python ``float`` -- fails the parse instead of smuggling a float onto the
    probability path.

    Args:
        token: The non-finite constant token the parser encountered.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"non-finite JSON constant is banned, got {token!r}")


def build_chat_request_body(request: LlmRequest, *, max_tokens: int) -> str:
    """Serialize a request into a deterministic, canonical chat-completion body.

    The body names exactly ``max_tokens``, ``messages`` (a single ``user`` turn
    carrying the prompt verbatim), ``model`` (the request's pinned version), and
    an integer ``temperature`` of ``0``. Keys are sorted and separators are
    space-free, so two equal requests build byte-identical bodies.

    Args:
        request: The completion request whose prompt/model the body carries.
        max_tokens: The response token cap to request.

    Returns:
        The canonical JSON request-body text.
    """
    return json.dumps(
        {
            _MAX_TOKENS_KEY: max_tokens,
            _MESSAGES_KEY: [{_CONTENT_KEY: request.prompt, _ROLE_KEY: _USER_ROLE}],
            _MODEL_KEY: request.model_version,
            _TEMPERATURE_KEY: _TEMPERATURE,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def fetch_envelope(
    transport: HttpTransport, *, endpoint_url: str, body: str
) -> ChatEnvelope:
    """POST ``body``, fast-reject a non-2xx status, and JSON-parse the response.

    The raw body is fingerprinted before any parsing, so a rejection carries
    only that fingerprint. A non-2xx status is rejected before the body is
    parsed at all; a malformed body, a non-object payload, or a non-finite JSON
    constant anywhere in the envelope is rejected as malformed.

    Args:
        transport: The HTTP transport to send the request through.
        endpoint_url: The endpoint URL to POST to.
        body: The canonical request body text.

    Returns:
        The parsed envelope paired with its response fingerprint.

    Raises:
        ProviderResponseRejectedError: On a non-2xx status, a malformed body, a
            non-object payload, or a non-finite JSON constant.
    """
    response = transport.send(
        HttpRequest(method=_REQUEST_METHOD, url=endpoint_url, body=body)
    )
    fingerprint = fingerprint_response(response.body)
    if not (_HTTP_SUCCESS_MIN <= response.status_code < _HTTP_SUCCESS_MAX_EXCLUSIVE):
        reject(RESPONSE_FAILURE_HTTP_STATUS, fingerprint)
    try:
        payload = json.loads(
            response.body, parse_float=Decimal, parse_constant=reject_constant
        )
    except ValueError:
        reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    if not isinstance(payload, dict):
        reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    return ChatEnvelope(payload=payload, fingerprint=fingerprint)


def require_first_element(
    payload: dict[str, object], key: str, fingerprint: str
) -> object:
    """Return the first element of a non-empty JSON array at ``payload[key]``.

    Args:
        payload: The parsed response object.
        key: The array-valued key to index.
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        The array's first element.

    Raises:
        ProviderResponseRejectedError: If the value is absent, not a JSON array,
            or an empty array.
    """
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    return value[0]


def require_object(value: object, fingerprint: str) -> dict[str, object]:
    """Return ``value`` as a mapping, or reject it as malformed.

    Args:
        value: The candidate value.
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        ``value`` when it is a JSON object.

    Raises:
        ProviderResponseRejectedError: If ``value`` is not a JSON object.
    """
    if not isinstance(value, dict):
        reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    return value


def require_text(value: object, fingerprint: str) -> str:
    """Return ``value`` as a string, or reject it as malformed.

    Args:
        value: The candidate value.
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        ``value`` when it is a string.

    Raises:
        ProviderResponseRejectedError: If ``value`` is not a string.
    """
    if not isinstance(value, str):
        reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    return value
