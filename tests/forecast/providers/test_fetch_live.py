"""Tests for windbreak.forecast.providers.fetch_live (issue #192).

Pins the live-fetch `FetchTransport` implementation's full contract:

* `fetch` builds `HttpRequest("GET", url, "")` and sends it through the
  injected `HttpTransport` -- never a `requests` import, never a live dial on
  the replay path (proven by driving the transport over a
  `ReplayHttpCassette`, whose miss/hit behavior is itself fail-closed).
* A non-2xx response status raises `UnreachableUrlError`.
* A response media type outside `config.allowed_content_types` -- matched
  case-insensitively and with any `;`-delimited parameters (e.g. a trailing
  `; charset=utf-8`) stripped first -- raises `ContentTypeRejectedError`.
* A response body whose UTF-8-encoded byte length exceeds
  `config.max_body_bytes` raises `BodyTooLargeError`.
* All three exceptions are `OSError` subclasses, so
  `windbreak.forecast.pipeline.bounded_web_research`'s existing
  `except OSError: continue` skips (and still counts) a live-fetch failure
  exactly like today's `ConnectionError`, with no pipeline-side change needed.
* Otherwise `fetch` returns `response.body` verbatim (the pipeline sanitizes
  and extracts a publication date from it downstream, not this transport).

`windbreak/forecast/providers/fetch_live.py` does not exist yet, so importing
it fails collection with `ModuleNotFoundError` -- the expected Gate 1 RED
state for issue #192.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.cassettes import CassetteMissError
from windbreak.forecast.providers.fetch_live import (
    BodyTooLargeError,
    ContentTypeRejectedError,
    LiveFetchConfig,
    LiveFetchTransport,
    UnreachableUrlError,
)
from windbreak.forecast.providers.http_cassettes import (
    ForbiddenLiveHttpTransport,
    HttpRequest,
    HttpResponse,
    RecordingHttpCassette,
    ReplayHttpCassette,
)

if TYPE_CHECKING:
    from pathlib import Path

#: The URL every test below fetches, unless a test overrides it.
_URL = "https://research.local/some-article"


def _config(
    *,
    max_body_bytes: int = 1_000_000,
    allowed_content_types: tuple[str, ...] = ("text/html", "text/plain"),
) -> LiveFetchConfig:
    """Build a `LiveFetchConfig` test double.

    Args:
        max_body_bytes: The maximum accepted response body size, in bytes.
        allowed_content_types: The accepted (lowercased, param-stripped)
            response media types.

    Returns:
        The constructed `LiveFetchConfig`.
    """
    return LiveFetchConfig(
        max_body_bytes=max_body_bytes, allowed_content_types=allowed_content_types
    )


class _StubHttpTransport:
    """A minimal `HttpTransport` double returning one fixed response verbatim."""

    def __init__(
        self, body: str, *, status_code: int = 200, content_type: str = "text/html"
    ) -> None:
        """Store the fixed response every `send` call returns.

        Args:
            body: The fixed raw response body text to return.
            status_code: The fixed HTTP status code to return.
            content_type: The fixed response media-type string to return.
        """
        self._body = body
        self._status_code = status_code
        self._content_type = content_type
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        """Record one call and return the fixed response, ignoring `request`.

        Args:
            request: The (recorded, otherwise unused) HTTP request.

        Returns:
            `HttpResponse(status_code, body, content_type)`, verbatim.
        """
        self.calls.append(request)
        return HttpResponse(self._status_code, self._body, self._content_type)


# --- Happy path: GET request shape, verbatim body passthrough --------------------


def test_fetch_sends_a_get_request_with_an_empty_body() -> None:
    """`fetch` builds `HttpRequest("GET", url, "")` -- a fetch never carries a
    request body, unlike the POST-shaped search transport.
    """
    transport = _StubHttpTransport("<html>hello</html>")
    live_fetch = LiveFetchTransport(transport, _config())

    live_fetch.fetch(_URL)

    assert len(transport.calls) == 1
    assert transport.calls[0] == HttpRequest("GET", _URL, "")


def test_fetch_returns_the_response_body_verbatim() -> None:
    """A clean 2xx, allowlisted-content-type, within-budget response returns
    `response.body` unchanged -- `fetch` never sanitizes or transforms it.
    """
    transport = _StubHttpTransport("<html><body>content</body></html>")
    live_fetch = LiveFetchTransport(transport, _config())

    result = live_fetch.fetch(_URL)

    assert result == "<html><body>content</body></html>"


# --- Non-2xx status: UnreachableUrlError ------------------------------------------


@pytest.mark.parametrize("status_code", [404, 410, 500, 503])
def test_fetch_non_2xx_status_raises_unreachable_url_error(status_code: int) -> None:
    """Any non-2xx status raises `UnreachableUrlError`, before any content-type
    or body-size check.
    """
    transport = _StubHttpTransport("error page", status_code=status_code)
    live_fetch = LiveFetchTransport(transport, _config())

    with pytest.raises(UnreachableUrlError):
        live_fetch.fetch(_URL)


def test_fetch_status_299_is_accepted_as_the_2xx_upper_boundary() -> None:
    """`299` is still inside the 2xx success range -- the inclusive/exclusive
    boundary mutant this pins.
    """
    transport = _StubHttpTransport("<html>ok</html>", status_code=299)
    live_fetch = LiveFetchTransport(transport, _config())

    result = live_fetch.fetch(_URL)

    assert result == "<html>ok</html>"


def test_fetch_status_300_raises_unreachable_url_error() -> None:
    """`300` is just outside the 2xx success range -- the other side of the
    same boundary.
    """
    transport = _StubHttpTransport("redirect", status_code=300)
    live_fetch = LiveFetchTransport(transport, _config())

    with pytest.raises(UnreachableUrlError):
        live_fetch.fetch(_URL)


def test_unreachable_url_error_is_an_os_error_subclass() -> None:
    """`UnreachableUrlError` is an `OSError` subclass, so
    `bounded_web_research`'s existing `except OSError: continue` skips it (and
    still counts the page) with no pipeline-side change.
    """
    assert issubclass(UnreachableUrlError, OSError)


# --- Content-type allowlist: case-insensitive, param-stripped --------------------


def test_fetch_content_type_off_the_allowlist_raises_content_type_rejected_error() -> (
    None
):
    """A response media type outside `allowed_content_types` is rejected."""
    transport = _StubHttpTransport("{}", content_type="application/json")
    live_fetch = LiveFetchTransport(
        transport, _config(allowed_content_types=("text/html",))
    )

    with pytest.raises(ContentTypeRejectedError):
        live_fetch.fetch(_URL)


def test_content_type_rejected_error_is_an_os_error_subclass() -> None:
    """`ContentTypeRejectedError` is an `OSError` subclass."""
    assert issubclass(ContentTypeRejectedError, OSError)


def test_fetch_content_type_with_charset_parameter_is_stripped_before_matching() -> (
    None
):
    """`"text/html; charset=utf-8"` matches an allowlisted `"text/html"` --
    everything from the first `;` onward is stripped before comparison.
    """
    transport = _StubHttpTransport(
        "<html>ok</html>", content_type="text/html; charset=utf-8"
    )
    live_fetch = LiveFetchTransport(
        transport, _config(allowed_content_types=("text/html",))
    )

    result = live_fetch.fetch(_URL)

    assert result == "<html>ok</html>"


def test_fetch_content_type_matching_is_case_insensitive() -> None:
    """`"Text/HTML"` matches an allowlisted lowercase `"text/html"`."""
    transport = _StubHttpTransport("<html>ok</html>", content_type="Text/HTML")
    live_fetch = LiveFetchTransport(
        transport, _config(allowed_content_types=("text/html",))
    )

    result = live_fetch.fetch(_URL)

    assert result == "<html>ok</html>"


def test_fetch_empty_content_type_is_rejected_when_not_itself_allowlisted() -> None:
    """A response reporting no content type at all (`""`) is rejected unless
    `""` itself is on the allowlist -- never silently trusted.
    """
    transport = _StubHttpTransport("mystery bytes", content_type="")
    live_fetch = LiveFetchTransport(
        transport, _config(allowed_content_types=("text/html",))
    )

    with pytest.raises(ContentTypeRejectedError):
        live_fetch.fetch(_URL)


# --- Body-size ceiling: exact-byte boundary, multi-byte UTF-8 --------------------


def test_fetch_body_exceeding_max_bytes_raises_body_too_large_error() -> None:
    """A response body whose UTF-8 byte length exceeds `max_body_bytes` is
    rejected.
    """
    transport = _StubHttpTransport("x" * 101)
    live_fetch = LiveFetchTransport(transport, _config(max_body_bytes=100))

    with pytest.raises(BodyTooLargeError):
        live_fetch.fetch(_URL)


def test_fetch_body_at_exactly_max_bytes_is_accepted() -> None:
    """A response body whose UTF-8 byte length exactly equals `max_body_bytes`
    is accepted (a strict `>` ceiling, never `>=`) -- the mutation-critical
    boundary edge.
    """
    body = "x" * 100
    transport = _StubHttpTransport(body)
    live_fetch = LiveFetchTransport(transport, _config(max_body_bytes=100))

    result = live_fetch.fetch(_URL)

    assert result == body


def test_body_too_large_error_is_an_os_error_subclass() -> None:
    """`BodyTooLargeError` is an `OSError` subclass."""
    assert issubclass(BodyTooLargeError, OSError)


def test_fetch_body_size_is_measured_in_utf8_encoded_bytes_not_characters() -> None:
    """A body of multi-byte characters is measured by its UTF-8 *byte* length,
    not its character count -- e.g. 40 three-byte characters is 120 bytes,
    over a 100-byte ceiling, even though `len(body) == 40`.
    """
    body = "☃" * 40  # SNOWMAN, 3 UTF-8 bytes each: 120 bytes, 40 chars.
    transport = _StubHttpTransport(body)
    live_fetch = LiveFetchTransport(transport, _config(max_body_bytes=100))

    with pytest.raises(BodyTooLargeError):
        live_fetch.fetch(_URL)


# --- Fixed check order: status before content-type before body size --------------


def test_non_2xx_status_is_rejected_before_content_type_is_ever_checked() -> None:
    """A non-2xx status raises `UnreachableUrlError` even when the content
    type is *also* off the allowlist -- status is checked first.
    """
    transport = _StubHttpTransport(
        "error", status_code=500, content_type="application/json"
    )
    live_fetch = LiveFetchTransport(
        transport, _config(allowed_content_types=("text/html",))
    )

    with pytest.raises(UnreachableUrlError):
        live_fetch.fetch(_URL)


def test_content_type_is_rejected_before_body_size_is_ever_checked() -> None:
    """An off-allowlist content type raises `ContentTypeRejectedError` even
    when the body is *also* over the size ceiling -- content type is checked
    before body size.
    """
    transport = _StubHttpTransport("x" * 1000, content_type="application/json")
    live_fetch = LiveFetchTransport(
        transport,
        _config(max_body_bytes=10, allowed_content_types=("text/html",)),
    )

    with pytest.raises(ContentTypeRejectedError):
        live_fetch.fetch(_URL)


# --- Record/replay: never dials ForbiddenLiveHttpTransport on the replay path ----


def test_fetch_over_replay_http_cassette_round_trips_and_never_dials_live(
    tmp_path: Path,
) -> None:
    """Recording a `LiveFetchTransport` run, then replaying the persisted
    cassette, reproduces the identical fetched body -- and the replay path
    never touches `ForbiddenLiveHttpTransport`, proving the live-shaped
    transport composes cleanly with the offline record/replay harness.
    """
    cassette_path = tmp_path / "fetch_cassette.json"
    recorder = RecordingHttpCassette(
        transport=_StubHttpTransport("<html>recorded content</html>"),
        path=cassette_path,
    )
    recorded = LiveFetchTransport(recorder, _config()).fetch(_URL)

    replay_fetch = LiveFetchTransport(
        ReplayHttpCassette.from_path(cassette_path), _config()
    )
    replayed = replay_fetch.fetch(_URL)

    assert replayed == recorded == "<html>recorded content</html>"


def test_fetch_over_empty_replay_cassette_raises_cassette_miss_error() -> None:
    """An unrecorded request fails closed via `CassetteMissError` -- never a
    live fallback -- proving the fetch transport really does call
    `transport.send` rather than short-circuiting.
    """
    live_fetch = LiveFetchTransport(ReplayHttpCassette({}), _config())

    with pytest.raises(CassetteMissError):
        live_fetch.fetch(_URL)


def test_fetch_over_forbidden_live_transport_fails_closed() -> None:
    """Driving `LiveFetchTransport` directly over `ForbiddenLiveHttpTransport`
    fails closed, confirming the transport is genuinely invoked (not bypassed).
    """
    from windbreak.forecast.cassettes import LiveCallForbiddenError

    live_fetch = LiveFetchTransport(ForbiddenLiveHttpTransport(), _config())

    with pytest.raises(LiveCallForbiddenError):
        live_fetch.fetch(_URL)
