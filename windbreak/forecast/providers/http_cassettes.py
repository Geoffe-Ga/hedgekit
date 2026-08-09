"""Offline record/replay harness for the forecast engine's HTTP calls (S8.9).

A second, HTTP-shaped record/replay harness alongside
:mod:`windbreak.forecast.cassettes`'s LLM-completion one, so a hosted
research-forecaster provider (:mod:`windbreak.forecast.providers.futuresearch`)
can run fully offline and deterministically in CI. It mirrors that module's
record-then-fail-closed-replay contract exactly, over an
:class:`HttpTransport` seam instead of the ``LlmTransport`` one:

* :class:`RecordingHttpCassette` wraps a real (or fake) transport, persisting
  each request/response pair to disk keyed by a stable request hash.
* :class:`ReplayHttpCassette` serves recorded responses purely from disk and
  *fails closed* (:class:`~windbreak.forecast.cassettes.CassetteMissError`, the
  reused sibling exception) on any unrecorded request -- never a live fallback.
* :class:`ForbiddenLiveHttpTransport` always raises the reused
  :class:`~windbreak.forecast.cassettes.LiveCallForbiddenError`, a structural
  proof that a given run never reaches a live network.

:class:`HttpRequest` deliberately carries no ``headers`` field: there is
nowhere on the dataclass to put API-key material, so a secret can never be
hashed into a request key or written to a recorded cassette file. A response
``body`` is an opaque string leaf holding the provider's raw JSON response
*text*, so decimal-looking numbers inside it are never parsed as floats by the
envelope's float-rejecting loader -- only a float leaf in the cassette's own
envelope *structure* is rejected. Request hashing uses the ledger's canonical
JSON form (sorted keys, no-space separators) over sha256, restated here with
only the standard library so this module stays dependency-free and float-free.

The record/replay seam is intentionally extended on the *response* side only:
:class:`HttpResponse` carries a single ``content_type`` media-type string
(e.g. ``text/html``) so a live-fetch transport
(:mod:`windbreak.forecast.providers.fetch_live`) can enforce a content-type
allowlist over a replayed response. This stays secret-free because it is a
lone, non-secret media type -- *not* a full headers map, which is where an
``Authorization``/API-key header would otherwise live -- and the request side
remains header-free, so the hash and the persisted request are unchanged. Older
cassettes recorded before this field existed replay unchanged: the loader
back-fills a missing ``content_type`` with the empty string.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, Protocol

from windbreak.forecast.cassettes import CassetteMissError, LiveCallForbiddenError

if TYPE_CHECKING:
    from pathlib import Path


def _canonical_json(obj: dict[str, str]) -> str:
    """Serialize a mapping to deterministic, whitespace-free JSON.

    Mirrors :func:`windbreak.ledger.events.canonical_json` (and the private
    sibling in :mod:`windbreak.forecast.cassettes`, restated here rather than
    imported so this module stays independently stdlib-only): keys are sorted
    and separators carry no spaces, so the output is a byte-stable function of
    the mapping's contents alone.

    Args:
        obj: The mapping to serialize.

    Returns:
        The canonical JSON encoding of ``obj``.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class HttpRequest:
    """A single, hashable HTTP request -- carrying no header material.

    The three fields are the entire request identity: there is deliberately no
    ``headers`` field, so an API key a live transport injects at send time can
    never be hashed into a request key or persisted to a cassette file.

    Attributes:
        method: The HTTP method (e.g. ``POST``).
        url: The fully qualified request URL.
        body: The raw request body text.
    """

    method: str
    url: str
    body: str

    def request_hash(self) -> str:
        """Return a stable sha256 hex digest of this request's fields.

        The digest is taken over the canonical JSON of ``{method, url, body}``,
        so it is deterministic across processes -- and independent of any
        environment variable -- changing if and only if a field changes.

        Returns:
            A lowercase, 64-character sha256 hex digest.
        """
        canonical = _canonical_json(
            {"method": self.method, "url": self.url, "body": self.body}
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """A single HTTP response -- its body an opaque raw-JSON text leaf.

    Attributes:
        status_code: The HTTP status code.
        body: The raw response body text (the provider's un-parsed JSON).
        content_type: The response media-type string (e.g. ``text/html``), a
            single non-secret value -- never a full headers map, so no API-key
            header can ever be persisted here. Defaults to the empty string so
            a response (or an older cassette) that reports none replays cleanly.
    """

    status_code: int
    body: str
    content_type: str = ""


class HttpTransport(Protocol):
    """The seam through which a single HTTP request/response is exchanged."""

    def send(self, request: HttpRequest) -> HttpResponse:
        """Return the response for ``request``.

        Args:
            request: The request to send.

        Returns:
            The HTTP response.
        """
        ...


class ForbiddenLiveHttpTransport:
    """An :class:`HttpTransport` that structurally forbids any live call."""

    def send(self, request: HttpRequest) -> NoReturn:
        """Refuse the call, proving no stage reached a live network.

        Args:
            request: The (rejected) HTTP request.

        Raises:
            LiveCallForbiddenError: Always.
        """
        raise LiveCallForbiddenError(
            f"live HTTP call forbidden for {request.method} {request.url}"
        )


class RecordingHttpCassette:
    """An :class:`HttpTransport` that records each call to disk as it delegates.

    Delegates every request to an underlying transport, accumulates the
    request/response pairs keyed by :meth:`HttpRequest.request_hash`, and
    rewrites the full mapping to ``path`` after each call so a replay cassette
    can be reloaded from it deterministically -- mirroring
    :class:`windbreak.forecast.cassettes.RecordingCassette`.
    """

    def __init__(self, *, transport: HttpTransport, path: Path) -> None:
        """Initialize the recorder.

        Args:
            transport: The underlying transport to delegate to.
            path: The file path the recorded mapping is written to.
        """
        self._transport = transport
        self._path = path
        self._entries: dict[str, dict[str, object]] = {}

    def send(self, request: HttpRequest) -> HttpResponse:
        """Delegate to the transport, record the pair, and persist to disk.

        Args:
            request: The HTTP request.

        Returns:
            The response returned by the underlying transport.
        """
        response = self._transport.send(request)
        entry: dict[str, object] = {
            "request": {
                "method": request.method,
                "url": request.url,
                "body": request.body,
            },
            "response": {
                "status_code": response.status_code,
                "body": response.body,
                "content_type": response.content_type,
            },
        }
        self._entries[request.request_hash()] = entry
        self._path.write_text(
            json.dumps(self._entries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return response


def _reject_float(raw: str) -> NoReturn:
    """Reject any float leaf encountered while loading a cassette.

    Installed as ``json.loads(..., parse_float=...)`` so a cassette whose own
    *envelope* structure carries a float (e.g. a stray ``latency_seconds:
    0.42`` beside ``status_code``/``body``) fails loudly. A decimal-looking
    number *inside* a response ``body`` string never reaches this hook: the
    ``body`` is an opaque text leaf, never parsed as a JSON number.

    Args:
        raw: The raw float token text from the JSON parser.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"float leaf is banned in cassettes, got {raw!r}")


class ReplayHttpCassette:
    """An :class:`HttpTransport` that serves recorded responses, fail-closed."""

    def __init__(self, entries: dict[str, HttpResponse]) -> None:
        """Initialize the replayer.

        Args:
            entries: A mapping of request hash to recorded :class:`HttpResponse`.
        """
        self._entries = entries

    @classmethod
    def from_path(cls, path: Path) -> ReplayHttpCassette:
        """Load a recorded cassette file into a replayer.

        The file is parsed with a float-rejecting hook, so any float leaf in
        the cassette's envelope structure raises :class:`ValueError`. Each
        top-level key is used verbatim as the replay lookup key, paired with an
        :class:`HttpResponse` rebuilt from its recorded
        ``status_code``/``body``/``content_type``. A cassette recorded before
        the ``content_type`` field existed omits it, so it is back-filled with
        the empty string for backward compatibility.

        Args:
            path: The cassette file to load.

        Returns:
            A replayer serving the file's recorded responses.

        Raises:
            ValueError: If the cassette contains a float leaf in its envelope.
        """
        raw = json.loads(path.read_text(encoding="utf-8"), parse_float=_reject_float)
        entries = {
            key: HttpResponse(
                status_code=entry["response"]["status_code"],
                body=entry["response"]["body"],
                content_type=entry["response"].get("content_type", ""),
            )
            for key, entry in raw.items()
        }
        return cls(entries)

    def send(self, request: HttpRequest) -> HttpResponse:
        """Return the recorded response for ``request`` or fail closed.

        Args:
            request: The HTTP request.

        Returns:
            The recorded :class:`HttpResponse`.

        Raises:
            CassetteMissError: If ``request`` has no recorded response.
        """
        key = request.request_hash()
        if key not in self._entries:
            raise CassetteMissError(f"no recorded response for request {key}")
        return self._entries[key]
