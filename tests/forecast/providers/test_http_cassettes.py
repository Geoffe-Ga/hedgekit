"""Tests for windbreak.forecast.providers.http_cassettes (issue #189, #192).

Pins `HttpRequest.request_hash()` determinism (a stable sha256 hex digest over
canonical JSON of `{method, url, body}`, independent of any environment
variable, and with no `headers` field to ever carry secret material),
`ReplayHttpCassette` hit/miss behavior, `RecordingHttpCassette` ->
`ReplayHttpCassette` round-tripping, `ForbiddenLiveHttpTransport` as the
structural proof a stage never reaches a live network, and the two committed
`tests/fixtures/forecast/futuresearch_cassette*.json` fixtures used only for
hash-independent structural checks -- mirroring
`tests/forecast/test_cassettes.py`'s LLM-side test suite over the HTTP-shaped
seam this module adds for the hosted FutureSearch provider.

Issue #192 adds a single non-secret `HttpResponse.content_type` media-type
string leaf (never a headers map -- there is still nowhere on either dataclass
for API-key material to live): `RecordingHttpCassette` persists it, and
`ReplayHttpCassette.from_path` loads it with a back-compat default of `""`
when an older, pre-#192 cassette file's recorded response omits the key
entirely -- so no previously-committed cassette fixture is invalidated by this
field's addition.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.cassettes import CassetteMissError, LiveCallForbiddenError
from windbreak.forecast.providers.http_cassettes import (
    ForbiddenLiveHttpTransport,
    HttpRequest,
    HttpResponse,
    RecordingHttpCassette,
    ReplayHttpCassette,
)

if TYPE_CHECKING:
    from pathlib import Path


def _request(**overrides: object) -> HttpRequest:
    """Build an `HttpRequest`, overriding any field via keyword arguments.

    Args:
        **overrides: Field values to override; unspecified fields fall back
            to a fixed, valid default.

    Returns:
        The constructed `HttpRequest`.
    """
    fields: dict[str, object] = {
        "method": "POST",
        "url": "https://futuresearch.example/v1/forecast",
        "body": '{"ticker": "EXAMPLE-TICKER"}',
    }
    fields.update(overrides)
    return HttpRequest(**fields)


class _FakeHttpTransport:
    """A minimal deterministic `HttpTransport` returning one fixed response."""

    def __init__(
        self, body: str, *, status_code: int = 200, content_type: str = ""
    ) -> None:
        """Store the fixed response every `send` call returns.

        Args:
            body: The fixed raw response body text.
            status_code: The fixed HTTP status code.
            content_type: The fixed response media-type string.
        """
        self._body = body
        self._status_code = status_code
        self._content_type = content_type
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        """Record the call and return the fixed canned response.

        Args:
            request: The HTTP request to record.

        Returns:
            `HttpResponse(self._status_code, self._body, self._content_type)`,
            verbatim.
        """
        self.calls.append(request)
        return HttpResponse(self._status_code, self._body, self._content_type)


# --- HttpRequest: no headers field, hashable, hash determinism -------------------


def test_http_request_has_no_headers_field() -> None:
    """`HttpRequest` declares exactly `{method, url, body}` -- never `headers`,
    so no seam exists for an API key to be hashed or persisted.
    """
    field_names = {field.name for field in dataclasses.fields(HttpRequest)}

    assert field_names == {"method", "url", "body"}


def test_http_request_is_frozen() -> None:
    """Mutating a constructed `HttpRequest` raises."""
    request = _request()

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        request.body = "mutated"  # type: ignore[misc]


def test_request_hash_is_sha256_hex_digest() -> None:
    """`request_hash()` is a lowercase, 64-character sha256 hex digest."""
    digest = _request().request_hash()

    assert isinstance(digest, str)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_request_hash_is_deterministic_for_identical_fields() -> None:
    """Two requests built from identical fields hash identically."""
    assert _request().request_hash() == _request().request_hash()


def test_request_hash_is_independent_of_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hash never changes with an unrelated environment variable, since
    `HttpRequest` carries no field a live transport's API key could ever
    reach.
    """
    request = _request()
    monkeypatch.delenv("FUTURESEARCH_API_KEY", raising=False)
    hash_without_env = request.request_hash()

    monkeypatch.setenv("FUTURESEARCH_API_KEY", "sk-unrelated-secret")
    hash_with_env = request.request_hash()

    assert hash_without_env == hash_with_env


def test_request_hash_differs_for_different_body() -> None:
    """Changing only the body changes the hash."""
    baseline_hash = _request().request_hash()
    changed_hash = _request(body='{"ticker": "DIFFERENT-TICKER"}').request_hash()

    assert baseline_hash != changed_hash


def test_request_hash_differs_for_different_method() -> None:
    """Changing only the method changes the hash."""
    baseline_hash = _request().request_hash()
    changed_hash = _request(method="GET").request_hash()

    assert baseline_hash != changed_hash


def test_request_hash_differs_for_different_url() -> None:
    """Changing only the url changes the hash."""
    baseline_hash = _request().request_hash()
    changed_url = "https://futuresearch.example/v2/forecast"
    changed_hash = _request(url=changed_url).request_hash()

    assert baseline_hash != changed_hash


# --- ForbiddenLiveHttpTransport: the structural no-network proof -----------------


def test_forbidden_live_http_transport_always_raises() -> None:
    """`ForbiddenLiveHttpTransport.send` always raises, never reaches a network."""
    transport = ForbiddenLiveHttpTransport()

    with pytest.raises(LiveCallForbiddenError):
        transport.send(_request())


# --- HttpResponse: content_type field (issue #192) --------------------------------


def test_http_response_content_type_defaults_to_empty_string() -> None:
    """`HttpResponse.content_type` defaults to `""` when not supplied, so every
    pre-#192 call site constructing `HttpResponse(status_code, body)`
    positionally keeps working unchanged.
    """
    response = HttpResponse(200, "some body")

    assert response.content_type == ""


def test_http_response_content_type_can_be_set() -> None:
    """`HttpResponse.content_type` carries the caller-supplied media-type
    string verbatim.
    """
    response = HttpResponse(200, "<html></html>", "text/html; charset=utf-8")

    assert response.content_type == "text/html; charset=utf-8"


def test_http_response_is_frozen() -> None:
    """Mutating a constructed `HttpResponse` raises."""
    response = HttpResponse(200, "some body", "text/html")

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        response.content_type = "application/json"  # type: ignore[misc]


def test_http_response_content_type_field_declared_on_the_dataclass() -> None:
    """`content_type` is a real dataclass field (not e.g. a headers mapping):
    the field set is exactly `{status_code, body, content_type}`, so there is
    still nowhere on `HttpResponse` for a headers/API-key structure to live.
    """
    field_names = {field.name for field in dataclasses.fields(HttpResponse)}

    assert field_names == {"status_code", "body", "content_type"}


# --- RecordingHttpCassette persists content_type; replay round-trips it ----------


def test_recording_http_cassette_persists_content_type_to_disk(tmp_path: Path) -> None:
    """The persisted cassette file's response entry carries the recorded
    `content_type` string verbatim.
    """
    cassette_path = tmp_path / "cassette.json"
    cassette = RecordingHttpCassette(
        transport=_FakeHttpTransport("body-1", content_type="text/html"),
        path=cassette_path,
    )

    cassette.send(_request())

    raw = json.loads(cassette_path.read_text(encoding="utf-8"))
    entry = next(iter(raw.values()))
    assert entry["response"]["content_type"] == "text/html"


def test_recording_then_replay_round_trip_preserves_content_type(
    tmp_path: Path,
) -> None:
    """A recorded `content_type` survives a full record -> replay round trip."""
    cassette_path = tmp_path / "cassette.json"
    recorder = RecordingHttpCassette(
        transport=_FakeHttpTransport("body-2", content_type="application/json"),
        path=cassette_path,
    )
    request = _request(body='{"ticker": "CONTENT-TYPE-ROUND-TRIP"}')
    recorder.send(request)

    replay = ReplayHttpCassette.from_path(cassette_path)

    assert replay.send(request) == HttpResponse(200, "body-2", "application/json")


def test_from_path_backcompat_missing_content_type_defaults_to_empty_string(
    tmp_path: Path,
) -> None:
    """A pre-#192 cassette file whose recorded response omits `content_type`
    entirely still loads, defaulting the field to `""` rather than raising --
    so no previously-committed cassette fixture is invalidated by this field's
    addition.
    """
    request = _request()
    legacy_entry = {
        request.request_hash(): {
            "request": {
                "method": request.method,
                "url": request.url,
                "body": request.body,
            },
            "response": {"status_code": 200, "body": "legacy body"},
        }
    }
    cassette_path = tmp_path / "legacy_cassette.json"
    cassette_path.write_text(json.dumps(legacy_entry), encoding="utf-8")

    replay = ReplayHttpCassette.from_path(cassette_path)

    assert replay.send(request) == HttpResponse(200, "legacy body", "")


# --- RecordingHttpCassette -> ReplayHttpCassette round-trip ----------------------


def test_recording_http_cassette_delegates_to_transport_and_returns_response(
    tmp_path: Path,
) -> None:
    """`RecordingHttpCassette.send` returns exactly what the transport returns."""
    transport = _FakeHttpTransport("recorded-response-1")
    cassette = RecordingHttpCassette(
        transport=transport, path=tmp_path / "cassette.json"
    )

    result = cassette.send(_request())

    assert result == HttpResponse(200, "recorded-response-1")
    assert len(transport.calls) == 1


def test_recording_http_cassette_persists_to_disk(tmp_path: Path) -> None:
    """After `send`, the cassette path exists and holds valid JSON."""
    cassette_path = tmp_path / "cassette.json"
    cassette = RecordingHttpCassette(
        transport=_FakeHttpTransport("r1"), path=cassette_path
    )

    cassette.send(_request())

    assert cassette_path.exists()
    json.loads(cassette_path.read_text(encoding="utf-8"))


def test_recording_http_cassette_round_trips_through_replay_http_cassette(
    tmp_path: Path,
) -> None:
    """A recorded request replays to the exact same response it recorded."""
    cassette_path = tmp_path / "cassette.json"
    recorder = RecordingHttpCassette(
        transport=_FakeHttpTransport("recorded-response-2"), path=cassette_path
    )
    request = _request(body='{"ticker": "A-SPECIFIC-RECORDED-TICKER"}')
    recorder.send(request)

    replay = ReplayHttpCassette.from_path(cassette_path)

    assert replay.send(request) == HttpResponse(200, "recorded-response-2")


def test_replay_http_cassette_miss_raises_cassette_miss_error(
    tmp_path: Path,
) -> None:
    """Querying a request that was never recorded raises `CassetteMissError`."""
    cassette_path = tmp_path / "cassette.json"
    recorder = RecordingHttpCassette(
        transport=_FakeHttpTransport("r1"), path=cassette_path
    )
    recorder.send(_request(body='{"ticker": "RECORDED-TICKER"}'))
    replay = ReplayHttpCassette.from_path(cassette_path)

    with pytest.raises(CassetteMissError):
        replay.send(_request(body='{"ticker": "A-COMPLETELY-DIFFERENT-TICKER"}'))


def test_replay_http_cassette_from_empty_mapping_misses_any_request() -> None:
    """`ReplayHttpCassette({})` (an empty mapping) is a guaranteed miss."""
    replay = ReplayHttpCassette({})

    with pytest.raises(CassetteMissError):
        replay.send(_request())


def test_replay_http_cassette_from_empty_mapping_file_misses_any_request(
    tmp_path: Path,
) -> None:
    """An empty cassette file (`{}`) is a guaranteed miss for any request."""
    empty_path = tmp_path / "empty.json"
    empty_path.write_text("{}", encoding="utf-8")
    replay = ReplayHttpCassette.from_path(empty_path)

    with pytest.raises(CassetteMissError):
        replay.send(_request())


# --- Static, committed fixtures: shape/miss + float-leaf rejection ---------------


def test_from_path_loads_committed_fixture_without_error(fixture_dir: Path) -> None:
    """`from_path` parses the committed fixture file without raising."""
    ReplayHttpCassette.from_path(fixture_dir / "futuresearch_cassette.json")


def test_from_path_committed_fixture_misses_on_unrecorded_request(
    fixture_dir: Path,
) -> None:
    """The committed fixture's key is a human-readable placeholder, never a
    real 64-char hex `request_hash()`, so querying it with any real
    `HttpRequest` is a guaranteed miss.
    """
    replay = ReplayHttpCassette.from_path(fixture_dir / "futuresearch_cassette.json")

    with pytest.raises(CassetteMissError):
        replay.send(_request())


def test_from_path_rejects_float_leaf(fixture_dir: Path) -> None:
    """A cassette file containing a float leaf in its envelope structure (e.g.
    a stray `latency_seconds: 0.42`) is rejected by the loader.

    The exact error message is an implementation detail of the "raising
    `parse_float` hook" design contract, so only the exception type is pinned
    here -- not a specific message string.
    """
    with pytest.raises(ValueError, match="float leaf"):
        ReplayHttpCassette.from_path(
            fixture_dir / "futuresearch_cassette_with_float.json"
        )
