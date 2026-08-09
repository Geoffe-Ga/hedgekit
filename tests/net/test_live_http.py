"""The live, allowlisted HTTP seam the forecast providers dial through (#344).

Every forecast provider adapter in ``windbreak/forecast/providers`` takes its
``HttpTransport`` by injection and, before this module, the *only* concrete
networked implementation lived privately inside four ``scripts/record_*.py``
recorders. That is why no live provider was reachable from ``windbreak run``.
:class:`~windbreak.net.live_http.LiveHttpTransport` promotes that proven
recorder shape into the package, at the network boundary where the outbound
allowlist already lives.

These tests pin the four properties that make it safe to point at a real host:

* **Screened before dialed.** An off-allowlist URL raises without the session
  ever being touched -- the check is not merely *before the response is used*,
  it is before any byte leaves the process (SPEC S15).
* **Redirect-free.** An on-path responder must not be able to steer a dial to
  another host, so redirects are refused rather than followed and re-screened.
* **Secrets stay off the request record.** The API key rides in send-time
  headers held by the transport; the header-free ``HttpRequest`` that adapters
  build (and that recording cassettes persist) can never carry it.
* **A dial failure is a discardable vote failure.** A timeout or connection
  fault surfaces as a :class:`ProviderTimeoutError`, which the pipeline
  discards per-vote into the quorum-abstention path -- never as a bare
  ``requests`` exception that would crash the whole tick.

Nothing here reaches a network: every test drives a stub session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import requests

from windbreak.forecast.cassettes import LlmRequest
from windbreak.forecast.providers import HttpRequest, ProviderTimeoutError
from windbreak.net.allowlist import EgressDeniedError, OutboundAllowlist
from windbreak.net.live_http import (
    LiveHttpTransport,
    RoutingLlmTransport,
    monotonic_ms,
    sleep_ms,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The single host every allowlisted test below permits.
_ALLOWED_HOST = "api.anthropic.com"
_ALLOWED_URL = f"https://{_ALLOWED_HOST}/v1/messages"

#: A host deliberately absent from the allowlist under test.
_DENIED_URL = "https://evil.example.com/v1/messages"

#: A distinctive marker standing in for a live credential in the send-time
#: headers. Named for what it is -- a canary the assertions look for -- rather
#: than as a credential, so the repo's secret scanner is never asked to tell a
#: fixture apart from the real thing.
_HEADER_CANARY = "live-http-header-canary-0001"
_HEADERS = {"x-api-key": _HEADER_CANARY, "content-type": "application/json"}

#: The integer dial timeout every transport below is constructed with.
_TIMEOUT_SECONDS = 30


class _StubResponse:
    """A minimal stand-in for a ``requests.Response``."""

    def __init__(
        self, status_code: int, text: str, headers: Mapping[str, str] | None = None
    ) -> None:
        """Store the fixed response fields.

        Args:
            status_code: The status code to report.
            text: The decoded body text to report.
            headers: The response headers, or ``None`` for none.
        """
        self.status_code = status_code
        self.text = text
        self.headers = dict(headers or {})


class _StubSession:
    """A ``requests.Session`` double recording each call's keyword arguments."""

    def __init__(self, response: _StubResponse | None = None) -> None:
        """Store the fixed response and prepare the call log.

        Args:
            response: The response every ``request`` returns, or ``None`` for a
                plain ``200``.
        """
        self._response = response or _StubResponse(200, "{}")
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _StubResponse:
        """Record one call and return the fixed response.

        Args:
            method: The HTTP method.
            url: The target URL.
            **kwargs: The remaining request keyword arguments.

        Returns:
            The fixed stub response.
        """
        self.calls.append({"method": method, "url": url, **kwargs})
        return self._response


class _RaisingSession:
    """A session double whose every ``request`` raises a fixed exception."""

    def __init__(self, error: Exception) -> None:
        """Store the exception every call raises.

        Args:
            error: The exception to raise.
        """
        self._error = error
        self.calls = 0

    def request(self, method: str, url: str, **kwargs: Any) -> _StubResponse:
        """Raise the fixed exception, recording the attempt.

        Args:
            method: The (unused) HTTP method.
            url: The (unused) target URL.
            **kwargs: The (unused) remaining request keyword arguments.

        Raises:
            Exception: The stored exception, always.
        """
        del method, url, kwargs
        self.calls += 1
        raise self._error


def _allowlist() -> OutboundAllowlist:
    """Build the single-host allowlist every test screens against.

    Returns:
        An allowlist permitting only ``_ALLOWED_HOST``.
    """
    return OutboundAllowlist(frozenset({_ALLOWED_HOST}))


def _transport(session: object) -> LiveHttpTransport:
    """Build the transport under test over ``session``.

    Args:
        session: The session double to dial through.

    Returns:
        The configured transport.
    """
    return LiveHttpTransport(
        session=session,
        allowlist=_allowlist(),
        headers=_HEADERS,
        timeout_seconds=_TIMEOUT_SECONDS,
    )


def _request(url: str = _ALLOWED_URL) -> HttpRequest:
    """Build an adapter-shaped request for ``url``.

    Args:
        url: The target URL.

    Returns:
        The header-free `HttpRequest` an adapter would build.
    """
    return HttpRequest(method="POST", url=url, body='{"model":"pinned"}')


# --- Screened before dialed -------------------------------------------------------


def test_an_off_allowlist_url_is_refused() -> None:
    """A host outside the allowlist raises rather than dialing."""
    transport = _transport(_StubSession())

    with pytest.raises(EgressDeniedError):
        transport.send(_request(_DENIED_URL))


def test_an_off_allowlist_url_never_reaches_the_session() -> None:
    """The refusal precedes the dial: no byte leaves the process (SPEC S15)."""
    session = _StubSession()
    transport = _transport(session)

    with pytest.raises(EgressDeniedError):
        transport.send(_request(_DENIED_URL))

    assert session.calls == []


def test_an_allowlisted_url_is_dialed() -> None:
    """The permitted host is reached, with method, URL, and body relayed."""
    session = _StubSession()

    _transport(session).send(_request())

    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["url"] == _ALLOWED_URL
    assert session.calls[0]["data"] == b'{"model":"pinned"}'


# --- Redirect-free, bounded ------------------------------------------------------


def test_redirects_are_refused_not_followed() -> None:
    """An on-path responder must not be able to steer the dial to another host."""
    session = _StubSession()

    _transport(session).send(_request())

    assert session.calls[0]["allow_redirects"] is False


def test_the_configured_integer_timeout_bounds_the_dial() -> None:
    """The dial is bounded by the configured whole-second timeout."""
    session = _StubSession()

    _transport(session).send(_request())

    assert session.calls[0]["timeout"] == _TIMEOUT_SECONDS
    assert isinstance(session.calls[0]["timeout"], int)


# --- Secret containment -----------------------------------------------------------


def test_the_api_key_is_injected_as_a_send_time_header() -> None:
    """The key reaches the wire through headers the transport holds."""
    session = _StubSession()

    _transport(session).send(_request())

    assert session.calls[0]["headers"]["x-api-key"] == _HEADER_CANARY


def test_the_api_key_never_touches_the_http_request_record() -> None:
    """`HttpRequest` is header-free, so a recording cassette cannot persist it."""
    request = _request()

    assert _HEADER_CANARY not in repr(request)
    assert not hasattr(request, "headers")


# --- Response relay ---------------------------------------------------------------


def test_the_status_code_and_body_are_relayed_verbatim() -> None:
    """The adapter sees exactly what the endpoint returned."""
    session = _StubSession(_StubResponse(503, "upstream unavailable"))

    response = _transport(session).send(_request())

    assert response.status_code == 503
    assert response.body == "upstream unavailable"


def test_the_response_content_type_is_relayed() -> None:
    """`LiveFetchTransport` screens on content type, so it must survive the hop."""
    session = _StubSession(
        _StubResponse(200, "<html></html>", {"content-type": "text/html"})
    )

    response = _transport(session).send(_request())

    assert response.content_type == "text/html"


def test_a_missing_content_type_header_relays_as_empty() -> None:
    """An absent header is reported as absent, never guessed at."""
    session = _StubSession(_StubResponse(200, "{}"))

    response = _transport(session).send(_request())

    assert response.content_type == ""


# --- Dial failures degrade to a discardable vote failure --------------------------


def test_a_timeout_becomes_a_discardable_provider_timeout() -> None:
    """A dial timeout is a `ProviderVoteError`, so one vote is discarded."""
    session = _RaisingSession(requests.Timeout("timed out"))

    with pytest.raises(ProviderTimeoutError):
        _transport(session).send(_request())


def test_a_connection_failure_becomes_a_discardable_provider_timeout() -> None:
    """A connection fault yields no response either, so it discards the same way.

    Left as a bare ``requests`` exception it would escape ``run_pipeline`` and
    crash the whole tick instead of degrading to quorum abstention.
    """
    session = _RaisingSession(requests.ConnectionError("no route to host"))

    with pytest.raises(ProviderTimeoutError):
        _transport(session).send(_request())


def test_a_non_requests_exception_propagates_untouched() -> None:
    """A real bug must not be dressed up as a benign provider timeout."""
    session = _RaisingSession(ZeroDivisionError("a genuine bug"))

    with pytest.raises(ZeroDivisionError):
        _transport(session).send(_request())


# --- Provider routing --------------------------------------------------------------


class _StubLlmTransport:
    """An `LlmTransport` double returning a fixed completion text."""

    def __init__(self, text: str) -> None:
        """Store the fixed completion text.

        Args:
            text: The text every `complete` returns.
        """
        self._text = text

    def complete(self, request: LlmRequest) -> str:
        """Return the fixed completion text.

        Args:
            request: The (unused) completion request.

        Returns:
            The fixed text.
        """
        del request
        return self._text


def _llm_request(provider: str) -> LlmRequest:
    """Build a completion request naming ``provider``.

    Args:
        provider: The provider identifier to route on.

    Returns:
        The completion request.
    """
    return LlmRequest(provider=provider, model_version="pinned", prompt="p")


def test_each_request_routes_to_its_own_providers_adapter() -> None:
    """One ensemble, several providers: each vote reaches its own adapter."""
    router = RoutingLlmTransport(
        {"anthropic": _StubLlmTransport("A"), "openai": _StubLlmTransport("O")}
    )

    assert router.complete(_llm_request("anthropic")) == "A"
    assert router.complete(_llm_request("openai")) == "O"


def test_an_unrouted_provider_fails_closed() -> None:
    """A provider with no live adapter must not silently reach another's."""
    router = RoutingLlmTransport({"anthropic": _StubLlmTransport("A")})

    with pytest.raises(KeyError):
        router.complete(_llm_request("openai"))


# --- The real-world time seam ------------------------------------------------------


def test_monotonic_ms_returns_whole_integer_milliseconds() -> None:
    """The retry schedule is integer-millisecond, so its clock must be too."""
    reading = monotonic_ms()

    assert isinstance(reading, int)
    assert not isinstance(reading, bool)


def test_monotonic_ms_does_not_run_backwards() -> None:
    """A monotonic clock is what makes a deadline meaningful."""
    assert monotonic_ms() <= monotonic_ms()


def test_sleep_ms_converts_milliseconds_to_the_underlying_seconds_sleep() -> None:
    """The ms-based retry API is honored exactly by the real sleep."""
    slept: list[float] = []

    sleep_ms(250, sleep=slept.append)

    assert slept == [0.25]


def test_sleep_ms_of_zero_still_calls_through() -> None:
    """A zero wait is a real (immediate) wait, not a skipped one."""
    slept: list[float] = []

    sleep_ms(0, sleep=slept.append)

    assert slept == [0.0]
