"""A committed research/vote corpus, replayed offline (issue #510).

The shipped PAPER loop could not reach a non-abstaining forecast from any entry
point an operator can type. Two seams were reachable only by a test injecting
into :func:`windbreak.scheduler.loop.build_paper_deps`:

* **Research.** ``offline_research_tools``'s transports find nothing by
  construction, so :func:`~windbreak.forecast.pipeline.run_pipeline` abstains on
  ``no_verified_citations`` before a single vote is cast.
* **Votes.** The committed prompt-hash cassette holds three human-readable
  placeholders and could not answer a real vote prompt.

This module is the replayed half of the fix: a *corpus* -- a small directory of
recorded search results, recorded pages, and recorded per-member votes -- and
the two transports that serve it. :mod:`windbreak.scheduler.provider_wiring`
selects it from ``forecast.replay_corpus`` in configuration.

WHY A CORPUS AND NOT A BETTER CASSETTE
--------------------------------------

A cassette key is :meth:`~windbreak.forecast.cassettes.LlmRequest.request_hash`
over a prompt that interpolates ``market.close_time.isoformat()``, while the
SPEC §16 horizon filter measures that same close against the run's clock. So a
market that *keeps* clearing the screen must carry a close that moves with the
clock, and a key that moves with the clock cannot be committed. The two
requirements contradict, and that contradiction is proved rather than asserted
by ``tests/scheduler/test_paper_intent_barriers.py::\
test_static_vote_cassette_and_horizon_filter_are_mutually_exclusive``.

A corpus keys on what does not move:

* research, on the **subquestion text** -- built by
  :func:`~windbreak.forecast.pipeline.decompose_subquestions` from a fixed
  prefix and ``market.title`` alone, with no clock and no close time in it;
* votes, on the **``provider:model_version`` pair** -- the ensemble member's own
  identity, which is configuration rather than a function of the market.

FAIL CLOSED ON THE CAPABILITY, NEVER ON THE PROCESS
---------------------------------------------------

Every runtime miss degrades rather than raising out of the tick:

* an unrecorded subquestion returns **no candidate URLs**, so the forecast
  gathers no citation and abstains -- the same honest outcome the offline
  default produces, reached because the corpus genuinely holds nothing on the
  question;
* an unrecorded URL raises :class:`OSError`, which both
  :func:`~windbreak.forecast.pipeline.bounded_web_research` and
  :func:`~windbreak.forecast.citations.verify_citation` already treat as an
  unreachable page;
* an unrecorded ensemble member raises :class:`ProviderCorpusMissError`, a
  :class:`~windbreak.forecast.providers.base.ProviderVoteError`, so the pipeline
  *discards that one vote* and the remaining members still form a quorum.

A malformed corpus is different in kind and refuses at load
(:class:`CorpusFormatError`): a directory an operator pointed at and got wrong
must not read as a corpus that legitimately knows nothing.

THIS IS NOT A LIVE CAPABILITY, AND NOT A FORECAST
--------------------------------------------------

Neither transport holds an endpoint, a credential, or a session; both read
committed files and nothing here can leave the host, so there is nothing for
the outbound allowlist to screen. And a replayed vote is recorded material, not
a measurement: a run in this mode demonstrates that the stack composes end to
end, and says nothing whatever about edge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from windbreak.forecast.cassettes import load_recorded_completions
from windbreak.forecast.providers.base import (
    NO_RESPONSE_FINGERPRINT,
    ProviderVoteError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from windbreak.forecast.cassettes import Completion, LlmRequest

#: The recorded search results and pages, inside a corpus directory.
CORPUS_RESEARCH_FILENAME: Final = "research.json"

#: The recorded per-ensemble-member votes, inside a corpus directory.
CORPUS_VOTES_FILENAME: Final = "votes.json"

#: The wire-format failure code a discarded vote carries when the corpus holds
#: no recording for the ensemble member that was asked.
PROVIDER_FAILURE_CORPUS_MISS: Final = "provider_corpus_miss"

#: ``research.json``'s two top-level keys.
_DOCUMENTS_KEY: Final = "documents"
_RESULTS_KEY: Final = "results"

#: A recorded document's two leaves.
_URL_KEY: Final = "url"
_BODY_KEY: Final = "body"

#: The only URL schemes a recorded document may be keyed by. Nothing here ever
#: dials, but the sandbox's own allowlist is derived from these hosts, so a
#: scheme it would refuse must never reach it as an allowed host.
_ALLOWED_SCHEMES: Final = frozenset({"http", "https"})


class CorpusFormatError(ValueError):
    """Raised when a replay-corpus directory is missing or malformed."""


class ProviderCorpusMissError(ProviderVoteError):
    """Raised when the corpus holds no recorded vote for an ensemble member.

    A :class:`~windbreak.forecast.providers.base.ProviderVoteError`, so the
    pipeline discards this one vote through the same per-vote path as any
    transport or screen failure rather than crashing the run: two of three
    recorded members still form a quorum, and zero recorded members abstain on
    the vote shortfall.

    Deliberately **not** transport-class. No provider was unreachable; the
    deployment's own corpus does not cover a member its ensemble names, which is
    a configuration fact. Classifying it as transport-class would stamp a
    zero-survivor run ``provider_unavailable`` and ledger a rationale claiming
    no provider could be reached, which would be false.

    Attributes:
        failure_code: :data:`PROVIDER_FAILURE_CORPUS_MISS`.
        response_fingerprint: :data:`NO_RESPONSE_FINGERPRINT` -- no response
            body ever existed.
    """

    def __init__(self, key: str) -> None:
        """Name the uncovered ensemble member.

        Args:
            key: The ``provider:model_version`` key with no recorded vote.
        """
        super().__init__(
            f"replay corpus holds no recorded vote for ensemble member {key!r}",
            failure_code=PROVIDER_FAILURE_CORPUS_MISS,
            response_fingerprint=NO_RESPONSE_FINGERPRINT,
            cost_micros=0,
        )


def vote_key(provider: str, model_version: str) -> str:
    """Return the corpus key one ensemble member's recorded vote is filed under.

    Derived in one place so the recorder, the loader, and the transport can
    never disagree about what identifies a member.

    Args:
        provider: The member's provider identifier.
        model_version: The member's pinned model version.

    Returns:
        The ``provider:model_version`` key.
    """
    return f"{provider}:{model_version}"


@dataclass(frozen=True, slots=True)
class ReplayCorpus:
    """One loaded corpus: recorded pages, recorded results, recorded votes.

    Attributes:
        documents: Every recorded page body, keyed by its URL.
        results: The recorded candidate URLs for each recorded subquestion, in
            recorded order.
        votes: Each ensemble member's recorded completion, keyed by
            :func:`vote_key`.
    """

    documents: Mapping[str, str]
    results: Mapping[str, tuple[str, ...]]
    votes: Mapping[str, Completion]

    def hosts(self) -> frozenset[str]:
        """Return every host this corpus holds a document for, lowercased.

        The research sandbox's egress allowlist is *derived* from this rather
        than configured beside it, so the set of hosts a corpus run may reach is
        exactly the set it has recordings for -- one fact, stated once. A
        transcribed second copy could grant a host the corpus cannot serve.

        Returns:
            The recorded hosts.
        """
        return frozenset(_host_of(url) for url in self.documents)


def _host_of(url: str) -> str:
    """Return a recorded URL's lowercased host, refusing an unusable one.

    Args:
        url: The recorded document URL.

    Returns:
        The lowercased hostname.

    Raises:
        CorpusFormatError: If the scheme is not http(s) or there is no host.
    """
    split = urlsplit(url)
    if split.scheme not in _ALLOWED_SCHEMES or not split.hostname:
        msg = (
            f"recorded document url {url!r} must be an http(s) URL with a host; "
            f"allowed schemes are {sorted(_ALLOWED_SCHEMES)}"
        )
        raise CorpusFormatError(msg)
    return split.hostname.lower()


def _read_json_object(path: Path) -> Mapping[str, object]:
    """Read one corpus file as a JSON object, refusing anything else.

    Args:
        path: The corpus file to read.

    Returns:
        The parsed mapping.

    Raises:
        CorpusFormatError: If the file is absent, unparseable, or not an object.
    """
    if not path.is_file():
        msg = f"replay corpus is missing its {path.name!r} file at {path}"
        raise CorpusFormatError(msg)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        msg = f"replay corpus file {path} is not readable JSON: {exc}"
        raise CorpusFormatError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"replay corpus file {path} is not a JSON object"
        raise CorpusFormatError(msg)
    return raw


def _load_documents(raw: Mapping[str, object], path: Path) -> dict[str, str]:
    """Read the recorded pages out of a parsed ``research.json``.

    Args:
        raw: The parsed research document.
        path: The file it came from, named in any error.

    Returns:
        Each recorded page body keyed by its URL.

    Raises:
        CorpusFormatError: If the block is missing, empty, or holds an entry
            that is not a ``{"url": str, "body": str}`` object.
    """
    entries = raw.get(_DOCUMENTS_KEY)
    if not isinstance(entries, list) or not entries:
        msg = f"{path} needs a non-empty {_DOCUMENTS_KEY!r} list"
        raise CorpusFormatError(msg)
    documents: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            msg = f"{path}: every {_DOCUMENTS_KEY!r} entry must be an object"
            raise CorpusFormatError(msg)
        url = entry.get(_URL_KEY)
        body = entry.get(_BODY_KEY)
        if not isinstance(url, str) or not isinstance(body, str):
            msg = (
                f"{path}: every {_DOCUMENTS_KEY!r} entry needs string "
                f"{_URL_KEY!r} and {_BODY_KEY!r} leaves"
            )
            raise CorpusFormatError(msg)
        _host_of(url)
        documents[url] = body
    return documents


def _load_results(
    raw: Mapping[str, object], documents: Mapping[str, str], path: Path
) -> dict[str, tuple[str, ...]]:
    """Read the recorded search results out of a parsed ``research.json``.

    Every referenced URL must be one this corpus actually holds. A result
    promising a page the corpus cannot serve would degrade at *fetch* time into
    an unreachable citation -- a run that abstains for a reason that is really a
    typo in a fixture -- so it is refused at load instead.

    Args:
        raw: The parsed research document.
        documents: The already-loaded pages, for the cross-reference check.
        path: The file it came from, named in any error.

    Returns:
        Each recorded subquestion's candidate URLs, in recorded order.

    Raises:
        CorpusFormatError: If the block is missing, empty, malformed, or names a
            URL the corpus holds no document for.
    """
    results = raw.get(_RESULTS_KEY)
    if not isinstance(results, dict) or not results:
        msg = f"{path} needs a non-empty {_RESULTS_KEY!r} object"
        raise CorpusFormatError(msg)
    loaded: dict[str, tuple[str, ...]] = {}
    for query, urls in results.items():
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            msg = f"{path}: {_RESULTS_KEY}[{query!r}] must be a list of URL strings"
            raise CorpusFormatError(msg)
        unknown = [url for url in urls if url not in documents]
        if unknown:
            msg = (
                f"{path}: {_RESULTS_KEY}[{query!r}] names {unknown[0]!r}, which "
                f"this corpus holds no recorded document for"
            )
            raise CorpusFormatError(msg)
        loaded[query] = tuple(urls)
    return loaded


def load_replay_corpus(directory: Path) -> ReplayCorpus:
    """Load a committed corpus directory, refusing a malformed one.

    Args:
        directory: The directory holding :data:`CORPUS_RESEARCH_FILENAME` and
            :data:`CORPUS_VOTES_FILENAME`.

    Returns:
        The loaded corpus.

    Raises:
        CorpusFormatError: If either file is absent or malformed, a recorded URL
            is not an http(s) URL with a host, a recorded result names a
            document the corpus does not hold, or the vote document is empty.
        ValueError: If the vote document holds a float leaf or a malformed
            ``usage`` block; see
            :func:`~windbreak.forecast.cassettes.load_recorded_completions`.
    """
    research_path = directory / CORPUS_RESEARCH_FILENAME
    votes_path = directory / CORPUS_VOTES_FILENAME
    raw = _read_json_object(research_path)
    documents = _load_documents(raw, research_path)
    results = _load_results(raw, documents, research_path)
    if not votes_path.is_file():
        msg = f"replay corpus is missing its {CORPUS_VOTES_FILENAME!r} file"
        raise CorpusFormatError(msg)
    votes = load_recorded_completions(votes_path)
    if not votes:
        msg = f"{votes_path} records no vote for any ensemble member"
        raise CorpusFormatError(msg)
    return ReplayCorpus(documents=documents, results=results, votes=votes)


class CorpusResearchTransport:
    """A search/fetch transport serving a committed corpus, never a network."""

    def __init__(self, corpus: ReplayCorpus) -> None:
        """Bind the transport to one loaded corpus.

        Args:
            corpus: The loaded corpus this transport replays.
        """
        self._corpus = corpus

    def search(self, query: str) -> tuple[str, ...]:
        """Return the candidate URLs recorded for ``query``, or none.

        A subquestion the corpus never recorded returns an empty tuple rather
        than raising, which is the honest answer -- this corpus holds nothing on
        that question -- and lets the pipeline abstain on zero verified
        citations instead of killing the tick.

        Args:
            query: The subquestion text, verbatim.

        Returns:
            The recorded candidate URLs, in recorded order; empty when the
            corpus recorded nothing for ``query``.
        """
        return self._corpus.results.get(query, ())

    def fetch(self, url: str) -> str:
        """Return the recorded page body at ``url``.

        Args:
            url: The URL to serve from the corpus.

        Returns:
            The recorded body, byte-identical on every call -- which is what
            lets :func:`~windbreak.forecast.citations.verify_citation`'s
            content-hash re-check pass over a replay.

        Raises:
            OSError: If the corpus holds no document for ``url``. Both the
                research gather and the citation re-check already read that as
                an unreachable page, so a gap in a corpus costs a citation
                rather than the run.
        """
        try:
            return self._corpus.documents[url]
        except KeyError as exc:
            msg = f"replay corpus holds no recorded document for {url!r}"
            raise OSError(msg) from exc


class CorpusVoteTransport:
    """An ``LlmTransport`` serving one recorded vote per ensemble member."""

    def __init__(self, corpus: ReplayCorpus) -> None:
        """Bind the transport to one loaded corpus.

        Args:
            corpus: The loaded corpus this transport replays.
        """
        self._corpus = corpus

    def complete(self, request: LlmRequest) -> Completion:
        """Return the recorded vote for ``request``'s ensemble member.

        The prompt is deliberately not part of the lookup: it carries the
        market's close time, which an anchored replay moves on every run, so a
        prompt-keyed recording could never be committed. What identifies the
        recording is the member that asked for it.

        Args:
            request: The completion request, read for its provider and pinned
                model version only.

        Returns:
            The recorded completion, token usage included when recorded.

        Raises:
            ProviderCorpusMissError: If the corpus records no vote for that
                member; the pipeline discards the one vote and carries on.
        """
        key = vote_key(request.provider, request.model_version)
        try:
            return self._corpus.votes[key]
        except KeyError as exc:
            raise ProviderCorpusMissError(key) from exc
