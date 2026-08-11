"""Offline record/replay harness for the forecast engine's LLM calls (S8.9).

The forecast pipeline must run fully offline and deterministically in CI, so
every LLM completion flows through an :class:`LlmTransport` seam -- a
dependency-injection point modeled on
:class:`windbreak.connector.snapshot.EventLedgerWriter`. Three transports back
the three modes the tests exercise:

* :class:`RecordingCassette` wraps a real (or fake) transport, persisting each
  request/response pair to disk keyed by a stable request hash.
* :class:`ReplayCassette` serves recorded responses purely from disk and
  *fails closed* (:class:`CassetteMissError`) on any unrecorded request --
  never a live fallback.
* :class:`ForbiddenLiveTransport` always raises
  :class:`LiveCallForbiddenError`, a structural proof that a given run never
  reaches a live network.

Request hashing uses the ledger's canonical JSON form (sorted keys, no-space
separators) over sha256, re-implemented here with only the standard library
so this module stays dependency-free and float-free.

THE SEAM CARRIES TOKEN ACCOUNTING (issue #451)

:meth:`LlmTransport.complete` returns a :class:`Completion` -- the response text
*and* the provider's reported :class:`~windbreak.forecast.budget.TokenUsage` --
rather than bare text. It returned bare text until issue #451, and that was the
structural reason a live vote could only ever be charged a flat per-attempt list
price: no layer above the transport had a token count to price from, so the
budget's ceilings bounded a count of attempts rather than spend.

Usage rides through record/replay for the same reason the response text does.
:class:`RecordingCassette` writes the reported counts into each cassette entry
and :class:`ReplayCassette` serves them back, so a replayed run reproduces the
recorded run's cost exactly instead of the meter being special-cased offline.
An entry recorded before this field existed (or one whose provider reported no
usage) replays with ``usage=None``, which the rate table charges its fail-closed
unmetered figure -- never zero.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, Protocol

from windbreak.forecast.budget import TokenUsage

if TYPE_CHECKING:
    from pathlib import Path

#: Cassette-entry keys. ``response`` is the pre-#451 text leaf, kept verbatim so
#: an older cassette loads unchanged; ``usage`` is the optional sibling issue
#: #451 adds.
_RESPONSE_KEY = "response"
_USAGE_KEY = "usage"
_INPUT_TOKENS_KEY = "input_tokens"
_OUTPUT_TOKENS_KEY = "output_tokens"


class CassetteMissError(Exception):
    """Raised when a replayed request has no recorded response (fail-closed)."""


class LiveCallForbiddenError(Exception):
    """Raised when a run attempts a forbidden live LLM call."""


def _canonical_json(obj: dict[str, str]) -> str:
    """Serialize a mapping to deterministic, whitespace-free JSON.

    Mirrors :func:`windbreak.ledger.events.canonical_json`: keys are sorted and
    separators carry no spaces, so the output is a byte-stable function of the
    mapping's contents alone.

    Args:
        obj: The mapping to serialize.

    Returns:
        The canonical JSON encoding of ``obj``.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class LlmRequest:
    """A single, hashable LLM completion request.

    Attributes:
        provider: The LLM provider identifier.
        model_version: The pinned model version string.
        prompt: The full prompt text.
    """

    provider: str
    model_version: str
    prompt: str

    def request_hash(self) -> str:
        """Return a stable sha256 hex digest of this request's fields.

        The digest is taken over the canonical JSON of ``{provider,
        model_version, prompt}``, so it is deterministic across processes and
        changes if and only if a field changes.

        Returns:
            A lowercase, 64-character sha256 hex digest.
        """
        canonical = _canonical_json(
            {
                "provider": self.provider,
                "model_version": self.model_version,
                "prompt": self.prompt,
            }
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Completion:
    """One LLM completion: its text plus what the provider says it cost.

    Attributes:
        text: The completion response text, verbatim.
        usage: The provider's reported token accounting, or ``None`` when the
            response carried none, carried none in a readable form, or came
            from a transport that has no billing to report (a replayed
            cassette recorded before issue #451). ``None`` means *unknown*, and
            :meth:`windbreak.forecast.budget.ModelRateTable.micros_for` charges
            it the fail-closed unmetered figure rather than zero.
    """

    text: str
    usage: TokenUsage | None = None


class LlmTransport(Protocol):
    """The seam through which a single LLM completion is obtained."""

    def complete(self, request: LlmRequest) -> Completion:
        """Return the completion for ``request``.

        Args:
            request: The completion request.

        Returns:
            The completion text paired with any reported token usage.
        """
        ...


class ForbiddenLiveTransport:
    """An :class:`LlmTransport` that structurally forbids any live call."""

    def complete(self, request: LlmRequest) -> NoReturn:
        """Refuse the call, proving no stage reached a live network.

        Args:
            request: The (rejected) completion request.

        Raises:
            LiveCallForbiddenError: Always.
        """
        raise LiveCallForbiddenError(
            f"live LLM call forbidden for {request.provider}:{request.model_version}"
        )


class RecordingCassette:
    """An :class:`LlmTransport` that records each call to disk as it delegates.

    Delegates every completion to an underlying transport, accumulates the
    request/response pairs keyed by :meth:`LlmRequest.request_hash`, and
    rewrites the full mapping to ``path`` after each call so a replay cassette
    can be reloaded from it deterministically.
    """

    def __init__(self, *, transport: LlmTransport, path: Path) -> None:
        """Initialize the recorder.

        Args:
            transport: The underlying transport to delegate to.
            path: The file path the recorded mapping is written to.
        """
        self._transport = transport
        self._path = path
        self._entries: dict[str, dict[str, object]] = {}

    def complete(self, request: LlmRequest) -> Completion:
        """Delegate to the transport, record the pair, and persist to disk.

        The reported token usage is recorded alongside the response text, and
        omitted entirely when the transport reported none -- so a replayed run
        reproduces the recorded run's metered cost, and an unmetered recording
        stays honestly unmetered rather than being written down as zero tokens
        (issue #451).

        Args:
            request: The completion request.

        Returns:
            The completion returned by the underlying transport.
        """
        completion = self._transport.complete(request)
        entry: dict[str, object] = {
            "request": {
                "provider": request.provider,
                "model_version": request.model_version,
                "prompt": request.prompt,
            },
            _RESPONSE_KEY: completion.text,
        }
        if completion.usage is not None:
            entry[_USAGE_KEY] = {
                _INPUT_TOKENS_KEY: completion.usage.input_tokens,
                _OUTPUT_TOKENS_KEY: completion.usage.output_tokens,
            }
        self._entries[request.request_hash()] = entry
        self._path.write_text(
            json.dumps(self._entries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return completion


def _reject_float(raw: str) -> NoReturn:
    """Reject any float leaf encountered while loading a cassette.

    Installed as ``json.loads(..., parse_float=...)`` so a cassette containing
    a float (e.g. ``temperature: 0.7``) fails loudly rather than smuggling a
    float onto the probability path.

    Args:
        raw: The raw float token text from the JSON parser.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"float leaf is banned in cassettes, got {raw!r}")


def _recorded_usage(entry: dict[str, object], key: str) -> TokenUsage | None:
    """Read one cassette entry's recorded token usage, if it has any.

    A missing ``usage`` block is not an error: cassettes recorded before issue
    #451, and recordings of providers that report no accounting, legitimately
    have none, and ``None`` is the honest answer for both -- it charges the
    fail-closed unmetered figure downstream rather than zero. A *present* block
    that is not two non-negative integers is a different thing entirely: it is a
    corrupt recording, and it fails the load loudly instead of silently
    degrading a real recorded cost into "unknown".

    Args:
        entry: One parsed cassette entry.
        key: The entry's request-hash key, named in any error.

    Returns:
        The recorded usage, or ``None`` when the entry recorded none.

    Raises:
        ValueError: If the entry carries a ``usage`` block that is not an
            object of two integer token counts.
    """
    raw = entry.get(_USAGE_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        msg = f"cassette entry {key!r} has a non-object {_USAGE_KEY} block"
        raise ValueError(msg)
    counts: list[int] = []
    for token_key in (_INPUT_TOKENS_KEY, _OUTPUT_TOKENS_KEY):
        value = raw.get(token_key)
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"cassette entry {key!r} has a non-integer {_USAGE_KEY}.{token_key}"
            raise ValueError(msg)
        counts.append(value)
    return TokenUsage(input_tokens=counts[0], output_tokens=counts[1])


class ReplayCassette:
    """An :class:`LlmTransport` that serves recorded responses, fail-closed."""

    def __init__(self, entries: dict[str, Completion]) -> None:
        """Initialize the replayer.

        Args:
            entries: A mapping of request hash to recorded completion.
        """
        self._entries = entries

    @classmethod
    def from_path(cls, path: Path) -> ReplayCassette:
        """Load a recorded cassette file into a replayer.

        The file is parsed with a float-rejecting hook, so any float leaf
        raises :class:`ValueError`. Each top-level key is used verbatim as the
        replay lookup key, paired with its entry's ``response`` text and any
        recorded ``usage`` block (issue #451).

        Args:
            path: The cassette file to load.

        Returns:
            A replayer serving the file's recorded completions.

        Raises:
            ValueError: If the cassette contains a float leaf, or an entry
                carries a malformed ``usage`` block.
        """
        raw = json.loads(path.read_text(encoding="utf-8"), parse_float=_reject_float)
        entries = {
            key: Completion(
                text=entry[_RESPONSE_KEY], usage=_recorded_usage(entry, key)
            )
            for key, entry in raw.items()
        }
        return cls(entries)

    def complete(self, request: LlmRequest) -> Completion:
        """Return the recorded completion for ``request`` or fail closed.

        Args:
            request: The completion request.

        Returns:
            The recorded completion, token usage included when recorded.

        Raises:
            CassetteMissError: If ``request`` has no recorded response.
        """
        key = request.request_hash()
        if key not in self._entries:
            raise CassetteMissError(f"no recorded response for request {key}")
        return self._entries[key]
