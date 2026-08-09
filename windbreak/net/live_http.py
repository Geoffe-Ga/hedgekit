"""The live, allowlisted network and wall-clock seams (issue #344).

Every forecast provider adapter in :mod:`windbreak.forecast.providers` takes its
:class:`~windbreak.forecast.providers.http_cassettes.HttpTransport` by injection
and never imports ``requests`` itself -- deliberately, so the forecast engine
stays network-library-free and offline-testable (SPEC S8.3). Until issue #344
the only concrete *networked* implementation of that seam lived privately inside
four ``scripts/record_*.py`` recorders, which is precisely why no live provider
was reachable from ``windbreak run``: the composition root had nothing to wire.

This module is that missing piece, placed at the network boundary where the
outbound allowlist already lives rather than inside the forecast package:

* :class:`LiveHttpTransport` -- a ``requests``-backed transport that screens
  every URL against an :class:`~windbreak.net.allowlist.OutboundAllowlist`
  *before* any byte leaves the process (SPEC S15), refuses redirects so an
  on-path responder cannot steer a dial to another host, bounds the dial with a
  whole-second timeout, and injects credentials as *send-time* headers so a key
  can never reach the header-free ``HttpRequest`` a recording cassette persists.
* :class:`RoutingLlmTransport` -- dispatches each vote to the adapter for its
  own provider, so a mixed-provider ensemble cannot send one vendor's prompt
  (or key) to another.
* :func:`monotonic_ms` / :func:`sleep_ms` -- the real-world time seam
  :class:`~windbreak.forecast.providers.retry.RetryingProvider` is injected
  with. They live here, outside the four float-banned money-path packages,
  because converting milliseconds to a real ``time.sleep`` requires a true
  division that ``scripts/lint_no_floats.py`` rightly forbids there. The retry
  layer's own schedule stays integer-millisecond throughout; only this final
  hop into the operating system is fractional.

A dial that never produced a response (a timeout, a refused connection, a DNS
failure) is translated into
:class:`~windbreak.forecast.providers.base.ProviderTimeoutError` -- a
``ProviderVoteError``, so the pipeline discards that one vote into the
quorum-abstention path. Left as a bare ``requests`` exception it would escape
``run_pipeline`` and crash the whole tick instead of degrading. A
non-``requests`` exception propagates untouched: a real bug must never be
dressed up as a benign provider timeout.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

import requests

from windbreak.forecast.providers.base import ProviderTimeoutError
from windbreak.forecast.providers.http_cassettes import HttpResponse

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from windbreak.forecast.cassettes import LlmRequest, LlmTransport
    from windbreak.forecast.providers.http_cassettes import HttpRequest
    from windbreak.net.allowlist import OutboundAllowlist

#: The response header naming a body's media type, lowercased for lookup.
_CONTENT_TYPE_HEADER = "content-type"

#: Milliseconds per second, converting the retry layer's integer-millisecond
#: schedule into the fractional seconds ``time.sleep`` takes.
_MS_PER_SECOND = 1000

#: Nanoseconds per millisecond, converting ``time.monotonic_ns`` to whole ms.
_NS_PER_MS = 1_000_000


class ResponseLike(Protocol):
    """The narrow slice of ``requests.Response`` this module reads.

    Attributes:
        status_code: The HTTP status code.
        text: The decoded response body.
        headers: The response headers.
    """

    @property
    def status_code(self) -> int:
        """Return the HTTP status code.

        Returns:
            The status code.
        """
        ...

    @property
    def text(self) -> str:
        """Return the decoded response body.

        Returns:
            The body text.
        """
        ...

    @property
    def headers(self) -> Mapping[str, str]:
        """Return the response headers.

        Returns:
            The headers.
        """
        ...


class SessionLike(Protocol):
    """The narrow slice of ``requests.Session`` this transport dials through.

    Declared structurally so a test can drive the transport with a double, and
    so the redirect-free session wrapper the connector already uses satisfies it
    without inheritance.
    """

    def request(
        self,
        method: str,
        url: str,
        *,
        data: bytes,
        headers: Mapping[str, str],
        timeout: int,
        allow_redirects: bool,
    ) -> Any:
        """Send one HTTP request.

        The keyword set is spelled out exactly as :meth:`LiveHttpTransport.send`
        passes it, rather than as ``**kwargs``: a protocol declaring ``**kwargs``
        would demand a session accepting *arbitrary* keywords, which the real
        ``requests.Session.request`` -- with its fixed, named option list -- does
        not satisfy. The return stays ``Any`` because it is narrowed immediately
        by :func:`_to_http_response`, and pinning it to
        :class:`ResponseLike` here would reject the genuine ``Response`` for no
        gain.

        Args:
            method: The HTTP method.
            url: The target URL.
            data: The encoded request body (keyword-only).
            headers: The send-time headers, including any credential
                (keyword-only).
            timeout: The dial timeout in whole seconds (keyword-only).
            allow_redirects: Whether redirects may be followed; always ``False``
                here (keyword-only).

        Returns:
            The response object.
        """
        ...


class LiveHttpTransport:
    """A live ``requests``-backed transport: allowlisted, redirect-free, keyed.

    The API key is held here inside the header map and injected on each send, so
    it never touches the header-free
    :class:`~windbreak.forecast.providers.http_cassettes.HttpRequest` and can
    never be written into a recorded cassette.
    """

    def __init__(
        self,
        *,
        session: SessionLike,
        allowlist: OutboundAllowlist,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> None:
        """Store the session, egress allowlist, send-time headers, and timeout.

        Args:
            session: The session to dial through (keyword-only).
            allowlist: The egress allowlist screening every outbound URL
                (keyword-only).
            headers: The headers -- including any API key -- injected on each
                send (keyword-only).
            timeout_seconds: The per-request dial timeout, in whole seconds
                (keyword-only).
        """
        self._session = session
        self._allowlist = allowlist
        self._headers = dict(headers)
        self._timeout_seconds = timeout_seconds

    def send(self, request: HttpRequest) -> HttpResponse:
        """Screen the URL, dial the endpoint once, and return its response.

        Args:
            request: The request to send.

        Returns:
            The endpoint's response, with its status, body, and declared
            content type relayed verbatim.

        Raises:
            EgressDeniedError: If the URL is off the allowlist -- raised before
                the session is touched at all.
            ProviderTimeoutError: If the dial produced no response (a timeout, a
                refused connection, a DNS failure), so the pipeline discards
                this one vote rather than crashing the tick.
        """
        self._allowlist.require(request.url)
        try:
            response = self._session.request(
                request.method,
                request.url,
                data=request.body.encode("utf-8"),
                headers=self._headers,
                timeout=self._timeout_seconds,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            raise ProviderTimeoutError from error
        return _to_http_response(response)


def _content_type_of(headers: Mapping[str, str]) -> str:
    """Return the declared media type, matching the header name case-insensitively.

    Args:
        headers: The response headers.

    Returns:
        The ``content-type`` header's value, or ``""`` when absent -- reported
        as absent rather than guessed at, since
        :class:`~windbreak.forecast.providers.fetch_live.LiveFetchTransport`
        screens on it.
    """
    for name, value in headers.items():
        if name.lower() == _CONTENT_TYPE_HEADER:
            return value
    return ""


def _to_http_response(response: ResponseLike) -> HttpResponse:
    """Adapt a ``requests``-shaped response into the provider seam's own type.

    Args:
        response: The response object returned by the session.

    Returns:
        The equivalent
        :class:`~windbreak.forecast.providers.http_cassettes.HttpResponse`.
    """
    return HttpResponse(
        status_code=response.status_code,
        body=response.text,
        content_type=_content_type_of(response.headers),
    )


class RoutingLlmTransport:
    """An :class:`LlmTransport` dispatching each request to its provider adapter.

    A vote ensemble mixes providers, but the pipeline drives every member
    through the single transport the composition root supplied. This router is
    what makes that single seam multi-provider: it forwards each
    :class:`~windbreak.forecast.cassettes.LlmRequest` to the adapter keyed by
    its ``provider`` field, so one vendor's prompt -- and one vendor's key,
    which lives inside its own transport -- can never reach another's endpoint.
    """

    def __init__(self, adapters: Mapping[str, LlmTransport]) -> None:
        """Store the per-provider completion adapters.

        Args:
            adapters: A mapping of provider identifier to its live adapter.
        """
        self._adapters = dict(adapters)

    def complete(self, request: LlmRequest) -> str:
        """Forward ``request`` to the adapter for its provider.

        Args:
            request: The completion request to route.

        Returns:
            The routed adapter's completion text.

        Raises:
            KeyError: If no adapter is registered for the request's provider --
                failing closed rather than silently borrowing another
                provider's transport (and credential).
        """
        return self._adapters[request.provider].complete(request)


def monotonic_ms() -> int:
    """Return a monotonic clock reading in whole milliseconds.

    Derived from :func:`time.monotonic_ns` by integer division, so the reading
    is a true ``int`` and never a rounded float.

    Returns:
        The monotonic reading, in whole milliseconds.
    """
    return time.monotonic_ns() // _NS_PER_MS


def sleep_ms(milliseconds: int, *, sleep: Callable[[float], None] = time.sleep) -> None:
    """Sleep for ``milliseconds``, honoring the retry layer's integer schedule.

    This is the one hop where the integer-millisecond retry schedule becomes a
    fractional real-world wait, which is why this seam lives outside the
    float-banned money-path packages.

    Args:
        milliseconds: The wait, in whole milliseconds.
        sleep: The underlying seconds-based sleep (keyword-only), injected so a
            test can observe the conversion without really waiting.
    """
    sleep(milliseconds / _MS_PER_SECOND)
