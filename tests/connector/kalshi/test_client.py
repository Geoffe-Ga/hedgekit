"""Tests for windbreak.connector.kalshi.client (issue #17): thin HTTPS JSON GET.

`KalshiClient.get(*segments, params=...)` treats each positional argument as
one path segment, percent-encoding it individually (`urllib.parse.quote(seg,
safe="")`) before joining with `/` -- so a ticker segment containing `/` or a
space is escaped rather than corrupting the request path. `session.get` is an
injected seam (never real HTTP; SPEC S7.1: CI runs offline).
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, tzinfo
from email.utils import format_datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from windbreak.connector.kalshi import client as client_module
from windbreak.connector.kalshi.client import (
    KALSHI_API_BASE,
    KalshiApiError,
    KalshiClient,
    KalshiResponse,
    _parse_server_date,
    _RedirectFreeSession,
)
from windbreak.net.allowlist import OutboundAllowlist

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import requests

#: Allowlist admitting the non-canonical fake host every construction below
#: uses, now that ``KalshiClient`` enforces its base URL host at construction
#: (issue #57).
_EXAMPLE_ALLOWLIST = OutboundAllowlist(frozenset({"example.kalshi.test"}))


class _FakeResponse:
    """A minimal stand-in for a `requests.Response`."""

    def __init__(
        self,
        status_code: int,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize a scripted fake response.

        Args:
            status_code: The HTTP status code to report.
            payload: The value `.json()` returns.
            headers: The response headers; empty when omitted.
        """
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> Any:
        """Return the scripted JSON payload."""
        return self._payload


class _RecordingSession:
    """Captures every `.get(...)` call and returns one scripted response."""

    def __init__(self, response: _FakeResponse) -> None:
        """Initialize with the single response every call will return.

        Args:
            response: The fake response returned by every `.get()` call.
        """
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        timeout: int | None = None,
    ) -> _FakeResponse:
        """Record the call's arguments and return the scripted response.

        Args:
            url: The request URL.
            params: The forwarded query parameters.
            timeout: The forwarded request timeout.

        Returns:
            The scripted fake response.
        """
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.response


def test_get_joins_quoted_segments_onto_base_url() -> None:
    """A segment containing `/` or a space is percent-encoded, not split."""
    # A schema-clean order book: the on-by-default validator inspects every
    # parsed payload, so a synthetic ``{"ok": True}`` would now (correctly)
    # raise ``SchemaAnomalyHaltError`` -- this test only cares about URL building.
    clean_orderbook = {"orderbook": {"yes": [], "no": []}}
    session = _RecordingSession(_FakeResponse(200, clean_orderbook))
    client = KalshiClient(
        base_url="https://example.kalshi.test",
        allowlist=_EXAMPLE_ALLOWLIST,
        timeout=7,
        session=session,
    )

    client.get("markets", "AB/CD E", "orderbook")

    assert (
        session.calls[0]["url"]
        == "https://example.kalshi.test/markets/AB%2FCD%20E/orderbook"
    )


def test_get_forwards_params_and_int_timeout() -> None:
    """`params` and the constructor's `timeout` reach `session.get` unchanged."""
    # A schema-clean ``/markets`` page: the on-by-default validator would reject
    # a synthetic ``{"ok": True}`` payload; this test only checks param/timeout
    # forwarding, so a valid empty page keeps its intent intact.
    session = _RecordingSession(_FakeResponse(200, {"markets": [], "cursor": ""}))
    client = KalshiClient(
        base_url="https://example.kalshi.test",
        allowlist=_EXAMPLE_ALLOWLIST,
        timeout=9,
        session=session,
    )

    client.get("markets", params={"cursor": "abc"})

    call = session.calls[0]
    assert call["params"] == {"cursor": "abc"}
    assert call["timeout"] == 9
    assert isinstance(call["timeout"], int)


@pytest.mark.parametrize("status_code", [300, 400, 404, 500, 503])
def test_non_2xx_status_raises_kalshi_api_error_naming_the_status(
    status_code: int,
) -> None:
    """A non-2xx response raises `KalshiApiError` naming the status code.

    ``300`` pins the upper boundary of the accepted range: a redirect status is
    outside 2xx and must fail closed, guarding the ``status_code <= 299`` bound.

    Resilience is disabled (`resilience=None`) so this exercises the *raw*
    transport status handling: a retryable ``5xx`` surfaces on the first attempt
    with no backoff sleep. The retry/breaker behavior around ``5xx`` is covered
    end-to-end in `test_client_resilience.py`.
    """
    session = _RecordingSession(_FakeResponse(status_code, {"error": "nope"}))
    client = KalshiClient(
        base_url="https://example.kalshi.test",
        allowlist=_EXAMPLE_ALLOWLIST,
        session=session,
        resilience=None,
    )

    with pytest.raises(KalshiApiError, match=str(status_code)):
        client.get("markets")


def test_status_299_is_accepted_as_success() -> None:
    """``299`` is the inclusive upper bound of the 2xx success range.

    Pairs with the ``300`` case above to pin both sides of the accept boundary,
    so a ``<=``-to-``<`` mutation on the upper bound is caught.
    """
    session = _RecordingSession(_FakeResponse(299, {"markets": []}, headers={}))
    client = KalshiClient(
        base_url="https://example.kalshi.test",
        allowlist=_EXAMPLE_ALLOWLIST,
        session=session,
    )

    assert client.get("markets").payload == {"markets": []}


def test_2xx_returns_kalshi_response_with_payload_and_parsed_date() -> None:
    """A 2xx response yields the parsed JSON payload and UTC server date."""
    fixed = datetime(2024, 12, 1, 12, 30, 0, tzinfo=UTC)
    session = _RecordingSession(
        _FakeResponse(200, {"markets": []}, headers={"Date": format_datetime(fixed)})
    )
    client = KalshiClient(
        base_url="https://example.kalshi.test",
        allowlist=_EXAMPLE_ALLOWLIST,
        session=session,
    )

    response = client.get("markets")

    assert isinstance(response, KalshiResponse)
    assert response.payload == {"markets": []}
    assert response.server_date == fixed


def test_2xx_without_date_header_has_none_server_date() -> None:
    """A missing `Date` header yields `server_date is None`, not an error."""
    session = _RecordingSession(_FakeResponse(200, {"markets": []}, headers={}))
    client = KalshiClient(
        base_url="https://example.kalshi.test",
        allowlist=_EXAMPLE_ALLOWLIST,
        session=session,
    )

    response = client.get("markets")

    assert response.server_date is None


class _OffsetlessTimezone(tzinfo):
    """A `tzinfo` whose `utcoffset` is `None`, so its datetimes are still naive.

    Python defines a datetime as naive when `tzinfo is None` *or* when
    `tzinfo.utcoffset(dt)` returns `None`. `parsedate_to_datetime` never
    produces that second shape, so the double below stands in for the
    third-party or deserializing parser that could (issue #346). Only
    `utcoffset` is overridden -- it is the single method both the naive-ness
    test and `astimezone` consult.
    """

    def utcoffset(self, dt: datetime | None) -> None:
        """Return `None` -- this zone declines to define an offset.

        Args:
            dt: The datetime being interrogated; unused, since the answer is
                unconditional.

        Returns:
            `None`, always.
        """
        return None


@pytest.fixture
def local_timezone_utc_minus_5() -> Iterator[None]:
    """Pin the process's local timezone to a fixed UTC-05:00 for one test.

    `datetime.astimezone()` silently interprets a *naive* datetime as **local**
    time. A test distinguishing "stamped UTC" from "reinterpreted as local" is
    therefore a false green on a UTC host -- which most CI runners are -- since
    both paths then yield the same instant. Pinning a fixed-offset zone (`EST5`
    carries no DST rule, so it is UTC-05:00 on every date) makes the correct
    and buggy results differ by five hours on every host, in CI and locally
    alike.

    Yields:
        None, with `TZ` pinned. The previous `TZ` and the interpreter's cached
        zone are both restored on exit, whether or not the test raises.
    """
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "EST5"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = previous
        time.tzset()


@pytest.mark.usefixtures("local_timezone_utc_minus_5")
def test_offsetless_tzinfo_date_header_is_stamped_utc_not_localized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parsed `Date` whose `tzinfo.utcoffset()` is `None` is naive, so it
    takes the same "HTTP dates are GMT" branch a `tzinfo`-less parse takes
    (issue #346).

    Routing it to `astimezone` instead does *not* raise: it silently reads the
    wall clock as the **host's** local time, so the recorded exchange instant
    would skew by whatever offset the machine happens to run at -- a wrong
    answer that differs between a developer's laptop and CI. The fixture pins
    UTC-05:00, so the buggy path would return 17:30 and this assertion pins
    12:30. The stdlib parser cannot emit this shape, so it is monkeypatched:
    the guard exists for the deserialized or third-party-parsed datetime.
    """
    offsetless = datetime(2024, 12, 1, 12, 30, tzinfo=_OffsetlessTimezone())
    monkeypatch.setattr(client_module, "parsedate_to_datetime", lambda _raw: offsetless)

    assert _parse_server_date({"Date": "Sun, 01 Dec 2024 12:30:00 -0000"}) == datetime(
        2024, 12, 1, 12, 30, tzinfo=UTC
    )


def test_construction_rejects_non_https_base_url() -> None:
    """A non-`https://` base URL is rejected at construction, not at request time."""
    with pytest.raises(ValueError, match="https"):
        KalshiClient(base_url="http://example.kalshi.test")


class _RedirectRecordingSession:
    """Captures the `allow_redirects` a `_RedirectFreeSession` forwards."""

    def __init__(self) -> None:
        """Initialize with no recorded `allow_redirects` yet."""
        self.allow_redirects: object = "unset"

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        """Record `allow_redirects` and return a scripted 200 response.

        Args:
            url: The request URL (unused).
            **kwargs: The forwarded request kwargs.

        Returns:
            A scripted fake response.
        """
        self.allow_redirects = kwargs.get("allow_redirects")
        return _FakeResponse(200, {"ok": True})


def test_real_transport_forbids_redirect_following() -> None:
    """The real-transport seam pins `allow_redirects=False` (SSRF posture).

    A fixed-endpoint public client must never chase a `3xx` `Location` to an
    arbitrary host, so every GET through the real transport disables redirects.
    """
    recorder = _RedirectRecordingSession()
    session = _RedirectFreeSession(cast("requests.Session", recorder))

    session.get("https://example.kalshi.test/markets", params=None, timeout=5)

    assert recorder.allow_redirects is False


def test_client_without_injected_session_uses_redirect_free_transport() -> None:
    """Constructing without a session wraps `requests` in the redirect-free seam."""
    client = KalshiClient(
        base_url="https://example.kalshi.test", allowlist=_EXAMPLE_ALLOWLIST
    )

    assert isinstance(client._session, _RedirectFreeSession)


# --- issue #57: outbound-network allowlist enforced at construction ------------
#
# `KalshiClient.__init__` does not yet accept an `allowlist` keyword, so any
# construction call below that passes one currently fails with
# `TypeError: __init__() got an unexpected keyword argument 'allowlist'`.
# `test_construction_with_a_non_canonical_base_url_and_no_allowlist_raises`
# fails for a different, independent reason: today's `KalshiClient` performs
# no host-allowlist check at all, so a non-canonical `base_url` currently
# constructs successfully, and the test's `pytest.raises(ValueError)` context
# reports `DID NOT RAISE` -- both are the expected Gate 1 RED state for issue
# #57's construction-boundary enforcement.
#
# NOTE (flagged for implementation to confirm): pre-issue-#57, every other
# test in this file freely constructs `KalshiClient(base_url="https://
# example.kalshi.test", ...)` -- a *non-canonical* host -- with no
# `allowlist` at all. Once construction-time allowlist enforcement lands as
# specified below, every one of those pre-existing calls will also need an
# explicit `allowlist` (or a canonical `base_url`) to keep constructing; this
# file does not attempt that migration itself (out of scope for a
# test-authorship pass), but the implementation specialist should expect a
# wide blast radius across this file's existing (currently-green) tests.


class _NeverCalledSession:
    """A `Session` double whose `.get` fails the test if ever invoked.

    Used to prove `KalshiClient` never reaches the network when construction
    itself is rejected by the outbound allowlist.
    """

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None,
        timeout: int,
    ) -> _FakeResponse:
        """Fail the test: construction must reject before any network call.

        Args:
            url: The request URL (unused; this must never be reached).
            params: Query parameters (unused).
            timeout: The request timeout (unused).
        """
        del url, params, timeout
        raise AssertionError(
            "KalshiClient must reject a disallowed base_url at construction, "
            "before ever calling session.get"
        )


def test_construction_with_a_non_canonical_base_url_and_no_allowlist_raises() -> None:
    """A `base_url` other than the canonical `KALSHI_API_BASE`, with no
    explicit `allowlist` supplied, is rejected at construction -- before any
    network call -- rather than silently trusting an arbitrary host.
    """
    with pytest.raises(ValueError):
        KalshiClient(
            base_url="https://not-the-real-kalshi.example.com/trade-api/v2",
            session=_NeverCalledSession(),
        )


def test_construction_with_an_explicit_allowlist_omitting_the_host_raises() -> None:
    """An explicit `allowlist` that does not include the requested
    `base_url`'s host also rejects construction, and the session's `.get` is
    never called.
    """
    from windbreak.net.allowlist import OutboundAllowlist

    allowlist = OutboundAllowlist(frozenset({"some-other-host.example.com"}))

    with pytest.raises(ValueError):
        KalshiClient(
            base_url="https://not-the-real-kalshi.example.com/trade-api/v2",
            session=_NeverCalledSession(),
            allowlist=allowlist,
        )


def test_stock_construction_against_the_canonical_base_url_still_works() -> None:
    """`KalshiClient(session=<spy>)` against the default `KALSHI_API_BASE`
    still constructs with no explicit `allowlist` -- the canonical host is
    always implicitly permitted.
    """
    client = KalshiClient(session=_NeverCalledSession())

    assert client._base_url == KALSHI_API_BASE
