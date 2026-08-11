"""Tests for windbreak.forecast.cassettes (issue #22): the offline LLM harness.

Pins `LlmRequest.request_hash()` determinism (a stable sha256 hex digest over
canonical JSON of its fields), `ReplayCassette` hit/miss behavior,
`RecordingCassette` -> `ReplayCassette` round-tripping, and
`ForbiddenLiveTransport` as the structural proof that a stage never reaches a
live network. `windbreak/forecast/` does not exist yet, so importing
`windbreak.forecast.cassettes` fails collection with `ModuleNotFoundError: No
module named 'windbreak.forecast'` -- the expected Gate 1 RED state for
issue #22.

See `tests/forecast/conftest.py`'s "Cassette-fixture choice" docstring note for
why the two committed fixtures below (`cassettes.json`,
`cassettes_with_float.json`) are used only for hash-independent structural
checks -- never to assert a specific, hand-computed request hash.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from windbreak.forecast.budget import TokenUsage
from windbreak.forecast.cassettes import (
    CassetteMissError,
    Completion,
    ForbiddenLiveTransport,
    LiveCallForbiddenError,
    LlmRequest,
    RecordingCassette,
    ReplayCassette,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "forecast"


def _request(**overrides: object) -> LlmRequest:
    fields: dict[str, object] = {
        "provider": "openai",
        "model_version": "gpt-5-forecast",
        "prompt": "What is the probability the Fed raises rates?",
    }
    fields.update(overrides)
    return LlmRequest(**fields)


class _FakeTransport:
    """A minimal deterministic transport returning one fixed canned response."""

    def __init__(self, response: str, *, usage: TokenUsage | None = None) -> None:
        self._response = response
        self._usage = usage
        self.calls: list[LlmRequest] = []

    def complete(self, request: LlmRequest) -> Completion:
        """Record the call and return the fixed canned completion."""
        self.calls.append(request)
        return Completion(text=self._response, usage=self._usage)


# --- LlmRequest.request_hash(): determinism --------------------------------------


def test_request_hash_is_sha256_hex_digest() -> None:
    """`request_hash()` is a lowercase, 64-character sha256 hex digest."""
    digest = _request().request_hash()

    assert isinstance(digest, str)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_request_hash_is_deterministic_for_identical_fields() -> None:
    """Two requests built from identical fields hash identically."""
    assert _request().request_hash() == _request().request_hash()


def test_request_hash_differs_for_different_prompt() -> None:
    """Changing only the prompt changes the hash."""
    baseline_hash = _request().request_hash()
    changed_hash = _request(prompt="a different prompt entirely").request_hash()

    assert baseline_hash != changed_hash


def test_request_hash_differs_for_different_provider() -> None:
    """Changing only the provider changes the hash."""
    baseline_hash = _request().request_hash()
    changed_hash = _request(provider="anthropic").request_hash()

    assert baseline_hash != changed_hash


def test_request_hash_differs_for_different_model_version() -> None:
    """Changing only the model_version changes the hash."""
    baseline_hash = _request().request_hash()
    changed_hash = _request(model_version="gpt-4-forecast").request_hash()

    assert baseline_hash != changed_hash


# --- ForbiddenLiveTransport: the structural no-network proof ---------------------


def test_forbidden_live_transport_always_raises() -> None:
    """`ForbiddenLiveTransport.complete` always raises, never reaches a network."""
    transport = ForbiddenLiveTransport()

    with pytest.raises(LiveCallForbiddenError):
        transport.complete(_request())


# --- RecordingCassette -> ReplayCassette round-trip ------------------------------


def test_recording_cassette_delegates_to_transport_and_returns_response(
    tmp_path: Path,
) -> None:
    """`RecordingCassette.complete` returns exactly what the transport returns."""
    transport = _FakeTransport("recorded-response-1")
    cassette = RecordingCassette(transport=transport, path=tmp_path / "cassette.json")

    result = cassette.complete(_request())

    assert result == Completion(text="recorded-response-1")
    assert len(transport.calls) == 1


def test_recording_cassette_persists_to_disk(tmp_path: Path) -> None:
    """After `complete`, the cassette path exists and holds valid JSON."""
    cassette_path = tmp_path / "cassette.json"
    cassette = RecordingCassette(transport=_FakeTransport("r1"), path=cassette_path)

    cassette.complete(_request())

    assert cassette_path.exists()
    json.loads(cassette_path.read_text(encoding="utf-8"))


def test_recording_cassette_persists_the_reported_token_usage(
    tmp_path: Path,
) -> None:
    """The recorded entry carries the exact token counts the transport reported.

    The cassette is what makes a replayed run's *cost* reproducible (issue
    #451); a recording that dropped the counts would replay every vote as
    unknown-cost.
    """
    cassette_path = tmp_path / "cassette.json"
    usage = TokenUsage(input_tokens=4_000, output_tokens=500)
    cassette = RecordingCassette(
        transport=_FakeTransport("r1", usage=usage), path=cassette_path
    )

    cassette.complete(_request())

    entry = next(iter(json.loads(cassette_path.read_text(encoding="utf-8")).values()))
    assert entry["usage"] == {"input_tokens": 4_000, "output_tokens": 500}


def test_recording_cassette_omits_usage_when_none_was_reported(
    tmp_path: Path,
) -> None:
    """A transport that reported no usage records no usage block at all.

    Writing ``{"input_tokens": 0, "output_tokens": 0}`` instead would turn "we
    do not know what this cost" into "this cost nothing", which is the exact
    fail-open issue #451 closes.
    """
    cassette_path = tmp_path / "cassette.json"
    cassette = RecordingCassette(transport=_FakeTransport("r1"), path=cassette_path)

    cassette.complete(_request())

    entry = next(iter(json.loads(cassette_path.read_text(encoding="utf-8")).values()))
    assert "usage" not in entry


def test_replay_cassette_serves_the_recorded_token_usage(tmp_path: Path) -> None:
    """A replayed completion carries the usage the recording captured."""
    cassette_path = tmp_path / "cassette.json"
    usage = TokenUsage(input_tokens=4_000, output_tokens=500)
    request = _request(prompt="a specific, recorded prompt")
    RecordingCassette(
        transport=_FakeTransport("r2", usage=usage), path=cassette_path
    ).complete(request)

    replayed = ReplayCassette.from_path(cassette_path).complete(request)

    assert replayed == Completion(text="r2", usage=usage)


def test_recording_cassette_round_trips_through_replay_cassette(
    tmp_path: Path,
) -> None:
    """A recorded request replays to the exact same response it recorded."""
    cassette_path = tmp_path / "cassette.json"
    recorder = RecordingCassette(
        transport=_FakeTransport("recorded-response-2"), path=cassette_path
    )
    request = _request(prompt="a specific, recorded prompt")
    recorder.complete(request)

    replay = ReplayCassette.from_path(cassette_path)

    assert replay.complete(request) == Completion(text="recorded-response-2")


def test_replay_cassette_miss_raises_cassette_miss_error(tmp_path: Path) -> None:
    """Querying a request that was never recorded raises `CassetteMissError`."""
    cassette_path = tmp_path / "cassette.json"
    recorder = RecordingCassette(transport=_FakeTransport("r1"), path=cassette_path)
    recorder.complete(_request(prompt="recorded prompt"))
    replay = ReplayCassette.from_path(cassette_path)

    with pytest.raises(CassetteMissError):
        replay.complete(_request(prompt="a completely different, unrecorded prompt"))


def test_replay_cassette_from_empty_mapping_file_misses_any_request(
    tmp_path: Path,
) -> None:
    """An empty cassette file (`{}`) is a guaranteed miss for any request."""
    empty_path = tmp_path / "empty.json"
    empty_path.write_text("{}", encoding="utf-8")
    replay = ReplayCassette.from_path(empty_path)

    with pytest.raises(CassetteMissError):
        replay.complete(_request())


# --- Static, committed fixtures: shape/miss + float-leaf rejection ---------------


def test_from_path_loads_committed_fixture_without_error() -> None:
    """`from_path` parses the committed fixture file without raising."""
    ReplayCassette.from_path(FIXTURE_DIR / "cassettes.json")


def test_from_path_committed_fixture_misses_on_unrecorded_request() -> None:
    """The committed fixture's keys are human-readable placeholders, never a
    real 64-char hex `request_hash()`, so querying it with any real
    `LlmRequest` is a guaranteed miss.
    """
    replay = ReplayCassette.from_path(FIXTURE_DIR / "cassettes.json")

    with pytest.raises(CassetteMissError):
        replay.complete(_request())


def test_from_path_rejects_float_leaf() -> None:
    """A cassette file containing a float leaf is rejected by the loader.

    The exact error message is an implementation detail of the "raising
    `parse_float` hook" the design contract specifies, so only the exception
    type is pinned here -- not a specific message string.
    """
    with pytest.raises(ValueError):
        ReplayCassette.from_path(FIXTURE_DIR / "cassettes_with_float.json")
