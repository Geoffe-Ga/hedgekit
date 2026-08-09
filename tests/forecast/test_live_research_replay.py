"""End-to-end live-research replay tests (issue #192).

Builds a `ResearchTools` over the new
`windbreak.forecast.providers.search_live.LiveSearchTransport` /
`windbreak.forecast.providers.fetch_live.LiveFetchTransport` pair, each backed
by the offline `ReplayHttpCassette` harness, and drives
`windbreak.forecast.pipeline.bounded_web_research` +
`windbreak.forecast.citations.verify_citations` end-to-end: a clean page
yields verified citations carrying real, timezone-aware publication dates
extracted by `windbreak.forecast.pubdate.extract_publication_date`, raw
fetched snapshots land in the sandboxed `ResearchCache`, and no recorded
cassette ever carries key-like material (`HttpRequest` has no `headers` field
at all, so there is structurally nowhere for one to go). A companion test
routes one poisoned-page corpus fixture (issue #27's delimiter-forgery case)
through the same live-shaped transports to prove the SPEC S8.5 injection
defense holds unchanged on this new transport surface: the citation fails to
verify and the run abstains before ever touching the vote transport.

Fixture-generation note
    The architect's design calls for a *static, pre-committed*
    `tests/fixtures/forecast/research_cassette.json` whose recorded request
    hashes line up byte-for-byte with what `LiveSearchTransport`/
    `LiveFetchTransport` build at test time. That file is committed (see
    `tests/fixtures/forecast/research_cassette.json`) but, like
    `futuresearch_cassette.json` before it, its key is a human-readable
    placeholder rather than a real 64-char `request_hash()` -- computing a
    real sha256 request hash by hand (rather than by running the recorder)
    is not something this authoring pass can do reliably, so it backs only
    the hash-independent structural checks below, not a byte-precise replay
    hit. The actual end-to-end proof instead uses this package's established
    "record against a fake transport into `tmp_path`, then reload with
    `ReplayHttpCassette.from_path`" pattern (see
    `tests/forecast/providers/test_futuresearch.py`'s
    `test_record_then_replay_round_trip_yields_identical_provider_forecast`),
    which is self-consistent by construction regardless of the exact request
    hash involved. A follow-up with the ability to run the recorder script
    could replace this file with a byte-precise static one if a literally
    static committed replay (rather than a record-then-replay round trip) is
    still wanted.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from windbreak.forecast.citations import verify_citations
from windbreak.forecast.pipeline import (
    ABSTENTION_NO_VERIFIED_CITATIONS,
    bounded_web_research,
    decompose_subquestions,
    run_pipeline,
)
from windbreak.forecast.providers.fetch_live import (
    LiveFetchConfig,
    LiveFetchTransport,
)
from windbreak.forecast.providers.http_cassettes import (
    HttpRequest,
    HttpResponse,
    RecordingHttpCassette,
    ReplayHttpCassette,
)
from windbreak.forecast.providers.search_live import (
    LiveSearchConfig,
    LiveSearchTransport,
)
from windbreak.forecast.sandbox import build_research_tools

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime as _datetime

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.records import BaselineQuoteSnapshot

#: The single research host every test below allowlists.
_RESEARCH_HOST = "research.local"

#: The one candidate URL every search response below resolves to.
_ARTICLE_URL = f"https://{_RESEARCH_HOST}/live-research-article"

#: The search endpoint every `LiveSearchConfig` below POSTs to.
_SEARCH_ENDPOINT = "https://search.example/v1/search"

#: A clean article page carrying a genuine JSON-LD `datePublished`, well
#: before the fixed `created_at` instant every test below verifies against.
_CLEAN_ARTICLE_PAGE = (
    '<html><head><script type="application/ld+json">'
    '{"@context": "https://schema.org", "@type": "NewsArticle", '
    '"datePublished": "2024-11-15T08:00:00Z"}</script></head>'
    "<body><p>Independent reporting on the underlying question, with no "
    "unusual claims or disruptions noted by observers this period.</p></body>"
    "</html>"
)

#: The corpus's issue #27 delimiter-forgery poisoned page (plain text, no
#: markup): a forged closing `<<<END-UNTRUSTED-DATA>>>` delimiter inside the
#: first `MAX_QUOTE_WORDS` words breaks the sanitized quote's raw-substring
#: property, so independent re-verification must fail it.
_POISONED_PAGE_PATH = (
    Path(__file__).resolve().parent
    / "injection"
    / "corpus"
    / "09_delimiter_forgery.html"
)


class _RoutingHttpTransport:
    """A minimal `HttpTransport` double routing by request method.

    Every `POST` (a `LiveSearchTransport` search call) returns a fixed
    single-URL `results` array; every `GET` (a `LiveFetchTransport` fetch
    call) returns a fixed page body, regardless of the exact URL -- mirroring
    `tests/forecast/injection/conftest.py`'s `PoisonedFetchTransport` "one
    fixed page for every URL" shape, extended to also serve the search side.
    """

    def __init__(self, *, results: list[str], page: str) -> None:
        """Store the fixed search results and fetch page this double serves.

        Args:
            results: The URL array every search (`POST`) response reports.
            page: The content every fetch (`GET`) response reports.
        """
        self._results = results
        self._page = page
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        """Route `request` by method to a canned search or fetch response.

        Args:
            request: The HTTP request to route.

        Returns:
            A search-results `HttpResponse` for `POST`, or the fixed page
            `HttpResponse` for `GET`.
        """
        self.calls.append(request)
        if request.method == "POST":
            body = json.dumps({"results": self._results})
            return HttpResponse(200, body, "application/json")
        return HttpResponse(200, self._page, "text/html; charset=utf-8")


def _build_tools_over_recorded_cassette(
    *,
    cassette_path: Path,
    cache_dir: Path,
    transport: _RoutingHttpTransport,
    subquestions: tuple[str, ...],
) -> tuple[object, Path]:
    """Record one live-research run's traffic, then rebuild tools over replay.

    Args:
        cassette_path: Where the combined search+fetch cassette is persisted.
        cache_dir: The sandboxed research cache's jail root.
        transport: The fake transport search and fetch calls are recorded
            against.
        subquestions: The exact subquestion texts the real run will later
            search for -- the priming pass must search these *same* texts
            (not placeholders), since `LiveSearchTransport`'s canonical
            request body is keyed on the query text, and a mismatched priming
            query would leave the real run's search request unrecorded.

    Returns:
        A `(tools, cassette_path)` pair: a `ResearchTools` built over
        `ReplayHttpCassette`-backed `LiveSearchTransport`/`LiveFetchTransport`
        instances sharing the one persisted cassette file, and that file's
        path (for the "no key material" / cache assertions).
    """
    recorder = RecordingHttpCassette(transport=transport, path=cassette_path)
    # Prime the cassette by running one full record pass first, over the same
    # shared recorder, so both the search and fetch traffic land in the one
    # file -- mirroring `tests/forecast/providers/test_futuresearch.py`'s
    # `test_record_then_replay_round_trip_yields_identical_provider_forecast`.
    priming_tools = build_research_tools(
        allowed_hosts=frozenset({_RESEARCH_HOST}),
        cache_dir=cache_dir / "priming",
        search_transport=LiveSearchTransport(
            recorder, LiveSearchConfig(endpoint_url=_SEARCH_ENDPOINT, max_results=5)
        ),
        fetch_transport=LiveFetchTransport(
            recorder,
            LiveFetchConfig(
                max_body_bytes=1_000_000,
                allowed_content_types=("text/html", "text/html; charset=utf-8"),
            ),
        ),
    )
    for subquestion in subquestions:
        for url in priming_tools.search(subquestion):
            priming_tools.fetch(url)

    replay = ReplayHttpCassette.from_path(cassette_path)
    tools = build_research_tools(
        allowed_hosts=frozenset({_RESEARCH_HOST}),
        cache_dir=cache_dir,
        search_transport=LiveSearchTransport(
            replay, LiveSearchConfig(endpoint_url=_SEARCH_ENDPOINT, max_results=5)
        ),
        fetch_transport=LiveFetchTransport(
            replay,
            LiveFetchConfig(
                max_body_bytes=1_000_000,
                allowed_content_types=("text/html", "text/html; charset=utf-8"),
            ),
        ),
    )
    return tools, cassette_path


# --- End-to-end: verified citations with real publication dates ------------------


def test_replayed_live_research_yields_verified_citations_with_real_dates(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: _datetime,
    tmp_path: Path,
) -> None:
    """A clean article, fetched and searched purely over the offline replay
    harness, yields citations that independently verify and carry the real
    timezone-aware publication date `windbreak.forecast.pubdate.
    extract_publication_date` pulled from the raw page.
    """
    transport = _RoutingHttpTransport(results=[_ARTICLE_URL], page=_CLEAN_ARTICLE_PAGE)
    subquestions = decompose_subquestions(market)
    tools, _cassette_path = _build_tools_over_recorded_cassette(
        cassette_path=tmp_path / "research_cassette.json",
        cache_dir=tmp_path / "cache",
        transport=transport,
        subquestions=subquestions,
    )

    citations = bounded_web_research(subquestions, tools=tools)
    verdicts = verify_citations(tools, citations, as_of=created_at)

    assert citations
    assert all(verdict.verified for verdict in verdicts)
    assert all(
        citation.publication_date == datetime(2024, 11, 15, 8, 0, 0, tzinfo=UTC)
        for citation in citations
    )
    assert all(citation.publication_date.tzinfo is not None for citation in citations)


def test_replayed_live_research_full_pipeline_run_is_live_eligible(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: _datetime,
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., object],
) -> None:
    """Driving the full `run_pipeline` over the replayed live-research tools
    (search + fetch both live-shaped) yields a normal, live-eligible "full"
    record -- the new transports compose end-to-end with the rest of the
    pipeline.
    """
    transport = _RoutingHttpTransport(results=[_ARTICLE_URL], page=_CLEAN_ARTICLE_PAGE)
    tools, _cassette_path = _build_tools_over_recorded_cassette(
        cassette_path=tmp_path / "research_cassette.json",
        cache_dir=tmp_path / "cache",
        transport=transport,
        subquestions=decompose_subquestions(market),
    )

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=tools,
    )

    assert record.triage_stage == "full"
    assert record.abstention_reason is None
    assert record.eligible_for_live is True
    assert len(record.citations) == 3
    assert all(citation.publication_date is not None for citation in record.citations)


def test_replayed_live_research_caches_raw_snapshots_under_the_sandbox_root(
    market: NormalizedMarket,
    tmp_path: Path,
) -> None:
    """Every fetch through the live-shaped transport still lands its raw
    content under the sandboxed `ResearchCache` root -- the live transport
    swap changes nothing about the cache-write discipline.
    """
    transport = _RoutingHttpTransport(results=[_ARTICLE_URL], page=_CLEAN_ARTICLE_PAGE)
    cache_dir = tmp_path / "cache"
    subquestions = decompose_subquestions(market)
    tools, _cassette_path = _build_tools_over_recorded_cassette(
        cassette_path=tmp_path / "research_cassette.json",
        cache_dir=cache_dir,
        transport=transport,
        subquestions=subquestions,
    )

    bounded_web_research(subquestions, tools=tools)

    cached_files = list(cache_dir.glob("*.txt"))
    assert cached_files
    for path in cached_files:
        assert path.read_text(encoding="utf-8") == _CLEAN_ARTICLE_PAGE


def test_replayed_live_research_cassette_carries_no_key_like_material(
    market: NormalizedMarket, tmp_path: Path
) -> None:
    """The persisted cassette file never carries anything resembling API-key
    material -- structurally guaranteed by `HttpRequest` having no `headers`
    field at all for a live transport's key to ever be hashed or written
    into.
    """
    transport = _RoutingHttpTransport(results=[_ARTICLE_URL], page=_CLEAN_ARTICLE_PAGE)
    cassette_path = tmp_path / "research_cassette.json"
    _tools, _path = _build_tools_over_recorded_cassette(
        cassette_path=cassette_path,
        cache_dir=tmp_path / "cache",
        transport=transport,
        subquestions=decompose_subquestions(market),
    )

    text = cassette_path.read_text(encoding="utf-8")
    for marker in ("api_key", "API_KEY", "Authorization", "Bearer ", "sk-"):
        assert marker not in text


# --- Committed fixture: hash-independent structural checks -----------------------


def test_committed_research_cassette_fixture_loads_without_error(
    fixture_dir: Path,
) -> None:
    """The committed `research_cassette.json` fixture parses without error --
    mirroring `test_from_path_loads_committed_fixture_without_error` in
    `tests/forecast/providers/test_http_cassettes.py`.
    """
    ReplayHttpCassette.from_path(fixture_dir / "research_cassette.json")


def test_committed_research_cassette_fixture_carries_no_key_like_material(
    fixture_dir: Path,
) -> None:
    """The committed fixture file itself never carries key-like material."""
    text = (fixture_dir / "research_cassette.json").read_text(encoding="utf-8")

    for marker in ("api_key", "API_KEY", "Authorization", "Bearer ", "sk-"):
        assert marker not in text


# --- Tracer-code invariant: a poisoned page through the live transports ---------


def test_poisoned_page_through_live_transports_fails_to_verify_and_abstains(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: _datetime,
    tmp_path: Path,
    make_fake_vote_transport: Callable[..., object],
) -> None:
    """The issue #27 delimiter-forgery poisoned page, served through the new
    live-shaped search/fetch transports instead of the fixture doubles,
    still fails independent re-verification (the forged closing delimiter
    inside the quote window breaks the sanitized quote's raw-substring
    property) and the run abstains with `ABSTENTION_NO_VERIFIED_CITATIONS`
    before the vote transport is ever touched -- zero prompt effect, proving
    the SPEC S8.5 injection defense holds unchanged on this new transport
    surface.
    """
    poisoned_page = _POISONED_PAGE_PATH.read_text(encoding="utf-8")
    transport = _RoutingHttpTransport(results=[_ARTICLE_URL], page=poisoned_page)
    tools, _cassette_path = _build_tools_over_recorded_cassette(
        cassette_path=tmp_path / "poisoned_research_cassette.json",
        cache_dir=tmp_path / "cache",
        transport=transport,
        subquestions=decompose_subquestions(market),
    )
    vote_transport = make_fake_vote_transport()

    record = run_pipeline(
        market,
        baseline,
        transport=vote_transport,
        created_at=created_at,
        research_tools=tools,
    )

    assert record.abstention_reason == ABSTENTION_NO_VERIFIED_CITATIONS
    assert record.eligible_for_live is False
    assert record.model_votes == ()
    baseline_ppm = baseline.price_pips * 100
    assert record.probability_ppm == baseline_ppm
