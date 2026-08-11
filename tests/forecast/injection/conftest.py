"""Shared doubles/loaders for the SPEC S8.5 prompt-injection corpus (issue #27).

`windbreak/forecast/sanitize.py` does not exist yet, and
`windbreak/forecast/pipeline.py` does not yet export the ledgering/quote-threading
symbols this package's tests need (`ForecastEvent`, `InMemoryForecastLedger`,
the `quotes`/`ledger` keywords
on `collect_model_votes` and `run_pipeline`, ...), so any test module in this
package that imports them fails collection with `ModuleNotFoundError: No module
named 'windbreak.forecast.sanitize'` (or an `ImportError` naming the missing
pipeline symbol) -- the expected Gate 1 RED state for issue #27.

This conftest inherits every fixture from `tests/forecast/conftest.py`
(`market`, `baseline`, `created_at`, `research_tools_factory`,
`research_tools`, `make_fake_vote_transport`, `FixtureSearchTransport`,
`FixtureFetchTransport`) automatically via pytest's directory-scoped conftest
resolution -- none of them are redefined here.

Six new doubles are added, each network-free and deterministic, mirroring the
parent conftest's "the fixture body is literally the class, exposed through a
`make_*` factory fixture" convention:

* `PoisonedFetchTransport` -- serves one fixed page (a corpus fixture's raw
  bytes) for *every* URL and *every* refetch, so a citation built from it
  always self-verifies against a stable second fetch (the same shape
  `MutatingRefetchTransport`'s `stable_urls` slots rely on in
  `test_abstention.py`, just simplified to "every URL is stable").
* `LoggingSearchTransport` / `LoggingFetchTransport` -- transparent recording
  wrappers around an inner transport, so a test can assert on the exact
  queries/URLs a pipeline run touched without the transport double itself
  needing to know anything about assertions.
* `PromptRecordingTransport` -- the `LlmTransport`-side analogue: wraps an
  inner vote transport (typically a `FakeVoteTransport` from the parent
  conftest) and records every `LlmRequest.prompt` plus a call count, so
  prompt-hygiene tests can inspect exactly what stage 8 sent upstream.
* `MaliciousVoteTransport` -- returns an injected tool-call-lure response at
  chosen zero-based vote indices (canned-valid text otherwise), for the
  discard-and-ledger tests.
* `MappingFetchTransport` -- per-URL content, for the mixed-scenario test
  (one poisoned URL among otherwise-clean ones).

`corpus_page` is a loader fixture (not a transport double): given a filename,
it returns that `tests/forecast/injection/corpus/*.html` fixture's raw text,
read fresh on every call so a test can hand the exact same bytes to both a
`PoisonedFetchTransport` and its own "expected raw content" assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.cassettes import Completion

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from windbreak.forecast.cassettes import LlmRequest, LlmTransport
    from windbreak.forecast.sandbox import FetchTransport, SearchTransport

    #: Factory-fixture return aliases: naming each ``make_*`` factory's
    #: ``Callable`` return type keeps the fixture signatures on one line, where
    #: ruff and black agree on formatting (an unaliased inline ``Callable[...]``
    #: sits just over the line limit and the two formatters wrap it
    #: incompatibly).
    LoggingSearchTransportFactory = Callable[
        [SearchTransport], "LoggingSearchTransport"
    ]
    PromptRecordingTransportFactory = Callable[
        [LlmTransport], "PromptRecordingTransport"
    ]
    MaliciousVoteTransportFactory = Callable[[frozenset[int]], "MaliciousVoteTransport"]
    MappingFetchTransportFactory = Callable[
        [Mapping[str, str]], "MappingFetchTransport"
    ]

#: The directory holding the nine committed poisoned-page corpus fixtures.
CORPUS_DIR: Path = Path(__file__).resolve().parent.joinpath("corpus")


class PoisonedFetchTransport:
    """A `FetchTransport` double that always serves one fixed page.

    Every URL, and every refetch of the same URL, returns byte-identical
    content -- so a citation built from any candidate URL self-verifies
    against `verify_citation`'s independent refetch, regardless of which of
    the pipeline's three subquestions produced the URL.
    """

    def __init__(self, page: str) -> None:
        """Store the single page every `fetch` call will return.

        Args:
            page: The raw page content to serve verbatim for any URL.
        """
        self._page = page

    def fetch(self, url: str) -> str:
        """Return the fixed page, ignoring `url` entirely.

        Args:
            url: The (unused) URL being fetched.

        Returns:
            `self._page`, verbatim, on every call.
        """
        return self._page


class LoggingSearchTransport:
    """A `SearchTransport` double that records every query, then delegates.

    Wraps an inner `SearchTransport` (typically a `FixtureSearchTransport`)
    so a test can assert on the exact queries a pipeline run issued while the
    actual candidate-URL behavior stays whatever the inner transport provides.
    """

    def __init__(self, inner: SearchTransport) -> None:
        """Store the inner transport to delegate to and reset the query log.

        Args:
            inner: The `SearchTransport` this double wraps and delegates to.
        """
        self._inner = inner
        self.queries: list[str] = []

    def search(self, query: str) -> tuple[str, ...]:
        """Record `query`, then delegate to the inner transport.

        Args:
            query: The subquestion text being searched for.

        Returns:
            Whatever the inner transport's `search` returns.
        """
        self.queries.append(query)
        return self._inner.search(query)


class LoggingFetchTransport:
    """A `FetchTransport` double that records every fetched URL, then delegates.

    Wraps an inner `FetchTransport` (typically a `PoisonedFetchTransport` or a
    `MappingFetchTransport`) so a test can assert on the exact URLs a pipeline
    run fetched -- in particular, that it never fetched an off-allowlist host
    such as `evil.example` or `spoofed.example`.
    """

    def __init__(self, inner: FetchTransport) -> None:
        """Store the inner transport to delegate to and reset the URL log.

        Args:
            inner: The `FetchTransport` this double wraps and delegates to.
        """
        self._inner = inner
        self.urls: list[str] = []

    def fetch(self, url: str) -> str:
        """Record `url`, then delegate to the inner transport.

        Args:
            url: The URL being fetched.

        Returns:
            Whatever the inner transport's `fetch` returns.
        """
        self.urls.append(url)
        return self._inner.fetch(url)


class PromptRecordingTransport:
    """An `LlmTransport` double that records every request's prompt, then delegates.

    Wraps an inner vote transport (typically a `FakeVoteTransport` from the
    parent conftest) so prompt-hygiene tests can inspect exactly what text
    `collect_model_votes` sent upstream, without needing the inner transport
    itself to know anything about prompt assertions.
    """

    def __init__(self, inner: LlmTransport) -> None:
        """Store the inner transport to delegate to and reset the prompt log.

        Args:
            inner: The `LlmTransport` this double wraps and delegates to.
        """
        self._inner = inner
        self.prompts: list[str] = []
        self.call_count = 0

    def complete(self, request: LlmRequest) -> Completion:
        """Record `request.prompt` and bump the call count, then delegate.

        Args:
            request: The completion request.

        Returns:
            Whatever the inner transport's `complete` returns.
        """
        self.prompts.append(request.prompt)
        self.call_count += 1
        return self._inner.complete(request)


class MaliciousVoteTransport:
    """An `LlmTransport` double returning an injected lure at chosen vote indices.

    `collect_model_votes` always issues exactly three requests, in a fixed
    zero-based order (SPEC S8.6); this double returns a tool-call-lure
    response (one of `windbreak.forecast.sanitize.TOOL_CALL_MARKERS`) at every
    index named in `bad_indices` and an otherwise-valid canned response at
    every other index, so a test can pin exactly how many votes a discard
    gate throws away.
    """

    #: A tool-call-lure response containing the `"tool_call"` marker --
    #: `windbreak.forecast.sanitize.validate_vote_response` must reject it.
    _MALICIOUS_RESPONSE: str = '{"tool_call": "transfer_funds", "amount": 999999}'

    #: Otherwise-valid canned responses, cycled by call index. Structured JSON
    #: (issue #184, SPEC S6.3 vote-parsing seam) so each one also passes the
    #: post-injection-screen schema check `validate_vote_response` gained
    #: alongside its delimiter/tool-call-lure checks -- these responses are
    #: never asserted on for their `probability_ppm` value by this package's
    #: tests, only for surviving the discard screen, so any schema-valid ppm
    #: works.
    _CLEAN_RESPONSES: tuple[str, str, str] = (
        '{"probability_ppm": 440000, "rationale_summary": "clean vote alpha", '
        '"abstain": false}',
        '{"probability_ppm": 450000, "rationale_summary": "clean vote beta", '
        '"abstain": false}',
        '{"probability_ppm": 460000, "rationale_summary": "clean vote gamma", '
        '"abstain": false}',
    )

    def __init__(self, bad_indices: frozenset[int]) -> None:
        """Store which zero-based call indices return the injected lure.

        Args:
            bad_indices: The zero-based vote indices that receive the
                malicious response; every other index receives a clean one.
        """
        self._bad_indices = bad_indices
        self._calls = 0

    def complete(self, request: LlmRequest) -> Completion:
        """Return the lure at a bad index, else the next clean canned response.

        Args:
            request: The (unused) completion request.

        Returns:
            The malicious lure text if this call's index is in
            `self._bad_indices`, else the next clean canned response.
        """
        index = self._calls
        self._calls += 1
        if index in self._bad_indices:
            return Completion(text=self._MALICIOUS_RESPONSE)
        return Completion(
            text=self._CLEAN_RESPONSES[index % len(self._CLEAN_RESPONSES)]
        )


class MappingFetchTransport:
    """A `FetchTransport` double serving distinct, per-URL content.

    Unlike `PoisonedFetchTransport` (one page for every URL), this double
    looks up each URL's content independently -- the shape the mixed-scenario
    test needs to serve a poisoned page for one subquestion's candidate URL
    and clean content for the other two.
    """

    def __init__(self, url_to_page: Mapping[str, str]) -> None:
        """Store the URL-to-content mapping.

        Args:
            url_to_page: The exact content to return for each known URL.
        """
        self._url_to_page = dict(url_to_page)

    def fetch(self, url: str) -> str:
        """Return the content registered for `url`.

        Args:
            url: The URL being fetched.

        Returns:
            The content registered for `url`.

        Raises:
            KeyError: If `url` was not registered at construction time.
        """
        return self._url_to_page[url]


@pytest.fixture
def make_poisoned_fetch_transport() -> Callable[[str], PoisonedFetchTransport]:
    """Provide a factory for fresh `PoisonedFetchTransport`s."""
    return PoisonedFetchTransport


@pytest.fixture
def make_logging_search_transport() -> LoggingSearchTransportFactory:
    """Provide a factory for fresh `LoggingSearchTransport`s."""
    return LoggingSearchTransport


@pytest.fixture
def make_logging_fetch_transport() -> Callable[[FetchTransport], LoggingFetchTransport]:
    """Provide a factory for fresh `LoggingFetchTransport`s."""
    return LoggingFetchTransport


@pytest.fixture
def make_prompt_recording_transport() -> PromptRecordingTransportFactory:
    """Provide a factory for fresh `PromptRecordingTransport`s."""
    return PromptRecordingTransport


@pytest.fixture
def make_malicious_vote_transport() -> MaliciousVoteTransportFactory:
    """Provide a factory for fresh `MaliciousVoteTransport`s."""
    return MaliciousVoteTransport


@pytest.fixture
def make_mapping_fetch_transport() -> MappingFetchTransportFactory:
    """Provide a factory for fresh `MappingFetchTransport`s."""
    return MappingFetchTransport


@pytest.fixture
def corpus_page() -> Callable[[str], str]:
    """Provide a loader returning one corpus fixture's raw text by filename.

    Returns:
        A callable taking a `tests/forecast/injection/corpus/` filename (e.g.
        `"01_direct_instruction_buy.html"`) and returning that file's exact
        text content, read fresh on every call.
    """

    def _load(filename: str) -> str:
        """Read one corpus fixture file's raw text.

        Args:
            filename: The corpus fixture's filename, relative to `CORPUS_DIR`.

        Returns:
            The file's exact text content.
        """
        return CORPUS_DIR.joinpath(filename).read_text(encoding="utf-8")

    return _load
