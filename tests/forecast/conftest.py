"""Shared fixtures for windbreak.forecast tests (issue #22, SPEC S8.2 / S6.3).

`windbreak/forecast/` does not exist yet, so any test module in this directory
that imports from it fails collection with `ModuleNotFoundError: No module
named 'windbreak.forecast'` -- the expected Gate 1 RED state for issue #22.

Two deliberate fixture-design choices, both explained here because they shape
every test file in this package:

Fixture-construction choice (market / baseline)
    `NormalizedMarket` and `BaselineQuoteSnapshot` are constructed directly in
    Python (mirroring `tests/connector/test_models.py`'s `_VALID_KWARGS`
    pattern) rather than loaded from a JSON fixture file. Unlike
    `FakeExchange.from_fixture_dir`, `windbreak.forecast` has no existing
    "load a normalized model from a fixture directory" helper -- inventing one
    here would test fixture-loading plumbing that has nothing to do with
    issue #22's actual contract (the pipeline stages, the record schema, the
    cassette harness).

Cassette-fixture choice (populated replay cassette)
    The *populated* cassette used to prove "replay never touches the network"
    is deliberately NOT built from a static, pre-committed JSON file with
    hardcoded SHA-256 request hashes. `LlmRequest.request_hash()` is specified
    as "sha256 hex of canonical JSON of its fields", but the exact request
    *prompts* the not-yet-written pipeline stages construct are an
    implementation detail this test suite cannot see in advance -- a
    hand-computed hash baked into a committed fixture would be silently
    brittle to any wording change in the real implementation's prompts, for
    zero added confidence. Instead, `make_fake_vote_transport` below provides
    a factory for a deterministic, network-free transport double
    (`FakeVoteTransport`) that pipeline tests wrap in a `RecordingCassette`
    pointed at `tmp_path`, then reload with `ReplayCassette.from_path` -- so
    every hash is self-consistent by construction, regardless of what the real
    implementation's prompt text turns out to be. The two committed
    `tests/fixtures/forecast/*.json` files (`cassettes.json`,
    `cassettes_with_float.json`) are used only for narrow, hash-independent
    structural tests in `test_cassettes.py`: a successful load, a *guaranteed*
    miss (their keys are human-readable placeholders, never a real 64-char
    hex digest), and float-leaf rejection.

Sandbox-transport fixture choice (issue #24)
    `FixtureSearchTransport` / `FixtureFetchTransport` below are the
    `windbreak.forecast.sandbox` analogue of `FakeVoteTransport`: deterministic,
    network-free doubles for the `SearchTransport` / `FetchTransport`
    injection seams. `research_tools_factory` defers its
    `from windbreak.forecast.sandbox import build_research_tools` to call time
    (inside the returned closure, not at module import time) so this conftest
    module keeps collecting cleanly for every other `tests/forecast/*` module
    while `windbreak/forecast/sandbox.py` does not yet exist -- only a test that
    actually calls the factory (or the `research_tools` fixture) hits the
    `ModuleNotFoundError`, which is the expected Gate 1 RED state for
    issue #24.

Citation-verification fixture choice (issue #26)
    `RaisingFetchTransport` and `MutatingRefetchTransport` below are the
    `windbreak.forecast.citations` analogue of the sandbox doubles above:
    deterministic `FetchTransport` doubles shared by `test_citations.py`
    (unit-level `verify_citation` checks) and `test_abstention.py`
    (end-to-end `run_pipeline`/`run_triaged_pipeline` abstention checks), so
    both test modules pin the exact same fetch-failure and fetch-mutation
    shapes rather than each inventing a slightly different one. Both are
    exposed only through factory fixtures (`make_raising_fetch_transport`,
    `make_mutating_refetch_transport`) mirroring `make_fake_vote_transport`'s
    "the fixture body is literally the class" convention -- a fresh,
    independently-stateful double per test, never one shared instance.

Divergence-cassette fixture choice (issue #191)
    `DIVERGENT_VOTE_RESPONSES` and the `diverse_markets` fixture below back
    `test_vote_cassette_divergence.py`'s cross-market record/replay coverage:
    three distinct (`NormalizedMarket`, `BaselineQuoteSnapshot`) pairs -- a
    Fed-rate market, a weather market, and a presidential-election market --
    each paired with three hand-authored, schema-valid #184 vote JSON
    responses whose `probability_ppm` values diverge from each other and,
    for at least one member, from that market's own `baseline_ppm`
    (`price_pips * 100`) by more than 20_000 ppm. Unlike
    `CANNED_VOTE_RESPONSES` (chosen to reproduce one fixed baseline-derived
    sequence for a single market), these responses are deliberately varied
    per market so the divergence tests exercise the record/replay cassette
    harness under genuinely different vote content, not just a different
    market shell wrapped around identical votes. Exactly one response in the
    whole mapping carries `"abstain": true`, exercising that schema branch
    end-to-end through the cassette harness.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.connector.models import NormalizedMarket
from windbreak.forecast.records import BaselineQuoteSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

#: Canned completion text for the three deterministic `collect_model_votes`
#: calls a pipeline run is expected to make. Each entry is a schema-valid
#: *structured* vote response (issue #184, SPEC S6.3 vote-parsing seam): a
#: compact JSON object carrying an integer `probability_ppm`, a
#: `rationale_summary`, and an `abstain` flag -- the shape
#: `windbreak.forecast.sanitize.parse_vote_response` requires once vote
#: probabilities are parsed from the response instead of derived from
#: `baseline ± offset`. The three `probability_ppm` values (440000 / 450000 /
#: 460000) are chosen to reproduce, byte-for-byte, what the pre-#184 pipeline
#: derived from `baseline ± 10_000 ppm` for this package's `baseline` fixture
#: (4500 pips -> 450_000 ppm baseline), so every pre-existing test asserting
#: on vote counts, dispersion, or aggregated medians keeps passing unchanged
#: once #184 wires vote probabilities from the response instead of the
#: baseline. No test asserts anything about `rationale_summary`'s wording --
#: only that each resulting vote carries a non-empty `response_fingerprint`.
CANNED_VOTE_RESPONSES: tuple[str, str, str] = (
    '{"probability_ppm": 440000, "rationale_summary": "steady evidence alpha", '
    '"abstain": false}',
    '{"probability_ppm": 450000, "rationale_summary": "steady evidence beta", '
    '"abstain": false}',
    '{"probability_ppm": 460000, "rationale_summary": "steady evidence gamma", '
    '"abstain": false}',
)


class FakeVoteTransport:
    """Deterministic, network-free stand-in `LlmTransport` for RED-state tests.

    Returns `CANNED_VOTE_RESPONSES` in call order, cycling if called more than
    three times, so a recording session is fully reproducible. Unlike
    `ForbiddenLiveTransport`, this double never raises -- it is the transport
    every non-cassette pipeline test wires in.
    """

    def __init__(self, responses: tuple[str, ...] = CANNED_VOTE_RESPONSES) -> None:
        """Store the canned response sequence and reset the call counter.

        Args:
            responses: The canned completions to cycle through, in call order.
        """
        self._responses = responses
        self._calls = 0

    def complete(self, request: object) -> str:
        """Return the next canned response, ignoring `request`'s contents.

        Args:
            request: The (unused) `LlmRequest`-shaped call.

        Returns:
            The next canned response in `self._responses`, cycling by index.
        """
        response = self._responses[self._calls % len(self._responses)]
        self._calls += 1
        return response


@pytest.fixture
def make_fake_vote_transport() -> Callable[[], FakeVoteTransport]:
    """Provide a factory for fresh, independently-stateful `FakeVoteTransport`s.

    A factory (not one shared instance) so a determinism test can build two
    completely independent transports that still produce the identical canned
    sequence.
    """
    return FakeVoteTransport


@pytest.fixture
def fixture_dir() -> Path:
    """Return the path to windbreak.forecast's own committed JSON fixtures."""
    return Path(__file__).resolve().parents[1] / "fixtures" / "forecast"


@pytest.fixture
def created_at() -> datetime:
    """Provide a fixed, timezone-aware UTC instant for deterministic records."""
    return datetime(2024, 12, 10, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def market() -> NormalizedMarket:
    """Provide a valid `NormalizedMarket`, constructed directly.

    See the module docstring's "Fixture-construction choice" for why this is
    not loaded from a JSON fixture file.
    """
    return NormalizedMarket(
        exchange="fake-exchange",
        ticker="KXFED-24DEC",
        event_ticker="KXFED-24",
        title="Fed raises rates in December 2024?",
        resolution_criteria="Resolves YES if the FOMC raises rates.",
        category="economics",
        close_time=datetime(2024, 12, 18, 19, tzinfo=UTC),
        expected_resolution_time=None,
        market_type="fully_collateralized_binary",
        price_tick_pips=100,
        min_order_contract_centis=100,
        fractional_trading_enabled=False,
        mutually_exclusive_group_id=None,
        jurisdiction_status="eligible",
        raw_exchange_payload_hash="sha256:abc123",
        volume_24h_micros=0,
    )


@pytest.fixture
def baseline(created_at: datetime) -> BaselineQuoteSnapshot:
    """Provide a valid `BaselineQuoteSnapshot` fixture."""
    return BaselineQuoteSnapshot(
        snapshot_id="snap-0001",
        price_pips=4500,
        fetched_at=created_at,
    )


#: The single host every sandbox fixture below allowlists by default.
_DEFAULT_ALLOWED_HOST = "research.local"


class FixtureSearchTransport:
    """Deterministic, network-free stand-in `SearchTransport` for RED-state tests.

    Returns exactly one candidate URL per query, on a fixed host, so
    `bounded_web_research` always has a single, reproducible candidate to
    fetch -- no live network, no branching on query content.
    """

    def __init__(self, host: str = _DEFAULT_ALLOWED_HOST) -> None:
        """Store the host embedded in every candidate URL this double returns.

        Args:
            host: The hostname every returned candidate URL resolves under.
        """
        self._host = host

    def search(self, query: str) -> tuple[str, ...]:
        """Return one deterministic candidate URL derived from `query`.

        Args:
            query: The subquestion text being searched for.

        Returns:
            A one-element tuple holding a URL on `self._host`, whose path is a
            short sha256 digest of `query` (so distinct queries get distinct,
            still-fully-deterministic, URLs).
        """
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        return (f"https://{self._host}/{digest}",)


class FixtureFetchTransport:
    """Deterministic, network-free stand-in `FetchTransport` for RED-state tests.

    Returns fixed content derived only from the URL itself, so fetching the
    same URL twice is always byte-identical across runs.
    """

    def fetch(self, url: str) -> str:
        """Return deterministic canned content for `url`.

        Args:
            url: The URL being fetched.

        Returns:
            A deterministic content string derived from `url`.
        """
        return f"fixture content for {url}"


@pytest.fixture
def research_tools_factory() -> Callable[..., object]:
    """Provide a factory building a sandboxed `ResearchTools` over fixture doubles.

    A factory (not one shared fixture) so tests that need an off-allowlist or
    otherwise non-default transport can override just the piece they care
    about while keeping the deterministic fixture doubles for everything
    else. The `windbreak.forecast.sandbox` import is deferred to the returned
    closure's call time -- see the module docstring's "Sandbox-transport
    fixture choice" note for why.
    """

    def _build(
        *,
        cache_dir: Path,
        allowed_hosts: frozenset[str] = frozenset({_DEFAULT_ALLOWED_HOST}),
        search_transport: object | None = None,
        fetch_transport: object | None = None,
    ) -> object:
        """Build one `ResearchTools`, defaulting transports to fixture doubles.

        Args:
            cache_dir: The research-cache root this instance persists under.
            allowed_hosts: The egress allowlist; defaults to
                `{"research.local"}`.
            search_transport: The `SearchTransport` to inject; defaults to a
                fresh `FixtureSearchTransport`.
            fetch_transport: The `FetchTransport` to inject; defaults to a
                fresh `FixtureFetchTransport`.

        Returns:
            A `ResearchTools` built by `build_research_tools`.
        """
        from windbreak.forecast.sandbox import build_research_tools

        return build_research_tools(
            allowed_hosts=allowed_hosts,
            cache_dir=cache_dir,
            search_transport=search_transport or FixtureSearchTransport(),
            fetch_transport=fetch_transport or FixtureFetchTransport(),
        )

    return _build


@pytest.fixture
def research_tools(
    tmp_path: Path, research_tools_factory: Callable[..., object]
) -> object:
    """Provide a `ResearchTools` sandboxed to the `research.local` allowlist."""
    return research_tools_factory(cache_dir=tmp_path / "research-cache")


class RaisingFetchTransport:
    """A `FetchTransport` modeling a dead/unreachable URL (issue #26).

    `fetch` always raises `ConnectionError` -- a subclass of `OSError` -- so
    any citation verified through a `ResearchTools` built over this transport
    always yields `windbreak.forecast.citations.verify_citation`'s
    `FAILURE_UNREACHABLE` verdict. The transport never distinguishes by URL:
    the RED-state contract under test is "the transport itself is down", not
    "one specific URL is down", so every call raises unconditionally.
    """

    def fetch(self, url: str) -> str:
        """Always raise, modeling a fully unreachable fetch transport.

        Args:
            url: The (unused) URL being fetched.

        Raises:
            ConnectionError: Unconditionally, on every call.
        """
        raise ConnectionError(f"connection refused for {url}")


class MutatingRefetchTransport:
    """A `FetchTransport` that mutates a URL's content on its *second* fetch.

    Issue #26's `bounded_web_research` (pipeline stage 5) fetches each
    subquestion's candidate URL exactly once to build its `Citation` -- the
    content hash and quoted text baked into that citation come from that one
    fetch. `windbreak.forecast.citations.verify_citation` then *refetches* the
    same URL through `tools.fetch` to independently recompute the content
    hash, so a citation whose backing URL returns *different* content on that
    second call is guaranteed to fail with a content-hash mismatch. This
    double exploits that exact double-fetch shape (one fetch to build, one
    refetch to verify) so an *end-to-end pipeline run* -- not just a
    unit-level `verify_citation` call -- yields a fully deterministic,
    predetermined count of verified citations.

    Distinct URLs are remembered in first-seen order (the order
    `bounded_web_research` discovers them in -- one per subquestion, via
    `FixtureSearchTransport`), tracked in an explicit list rather than relying
    on any dict/set iteration order. The first `stable_urls` distinct URLs
    behave exactly like `FixtureFetchTransport`: *every* fetch of one of
    them -- first, second, or any later one -- returns the same deterministic
    `f"fixture content for {url}"` string, so their citations verify cleanly
    forever. Every other (later-discovered) distinct URL returns that same
    stable content on its *first* fetch only (so the citation stage 5 builds
    from that fetch is itself internally self-consistent: its stored content
    hash matches its own first-fetch content), but returns different,
    fetch-count-tagged content on every subsequent fetch of that same URL --
    so `verify_citation`'s refetch-and-rehash always mismatches for it.

    Net effect, given the package's fixed three-subquestion decomposition: a
    full pipeline run wired to `MutatingRefetchTransport(stable_urls=N)`
    yields exactly `N` verified citations (for any `0 <= N <= 3`), every time.
    """

    def __init__(self, stable_urls: int) -> None:
        """Store the stable-URL count and reset the per-URL fetch bookkeeping.

        Args:
            stable_urls: How many distinct URLs (in first-seen order) always
                return stable content, however many times they are fetched.
        """
        self._stable_urls = stable_urls
        self._seen_order: list[str] = []
        self._fetch_counts: dict[str, int] = {}

    def fetch(self, url: str) -> str:
        """Return stable or mutated content for `url`, per this URL's slot.

        Args:
            url: The URL being fetched.

        Returns:
            Deterministic stable content (`f"fixture content for {url}"`) for
            one of the first `stable_urls` distinct URLs seen (on any fetch of
            it), or for any other URL's very first fetch; mutated,
            fetch-count-tagged content for that other URL's second-or-later
            fetch.
        """
        if url not in self._seen_order:
            self._seen_order.append(url)
        self._fetch_counts[url] = self._fetch_counts.get(url, 0) + 1
        stable_content = f"fixture content for {url}"
        slot = self._seen_order.index(url)
        if slot < self._stable_urls or self._fetch_counts[url] == 1:
            return stable_content
        return f"mutated content for {url} (fetch #{self._fetch_counts[url]})"


@pytest.fixture
def make_raising_fetch_transport() -> Callable[[], RaisingFetchTransport]:
    """Provide a factory for fresh `RaisingFetchTransport` instances.

    A factory (mirroring `make_fake_vote_transport`) so citation-verification
    tests can build a fresh, independently-stateless double per test.
    """
    return RaisingFetchTransport


@pytest.fixture
def make_mutating_refetch_transport() -> Callable[..., MutatingRefetchTransport]:
    """Provide a factory for fresh `MutatingRefetchTransport` instances.

    Callable with `stable_urls` (positional or keyword), exactly like
    constructing `MutatingRefetchTransport` directly -- mirrors
    `make_fake_vote_transport`'s "the fixture body is literally the class"
    convention.
    """
    return MutatingRefetchTransport


# --- Divergence-cassette fixtures (issue #191) ------------------------------------
#
# See the module docstring's "Divergence-cassette fixture choice" note.

#: Ticker for the Fed-rate divergence-test market.
_FED_TICKER = "KXFED-25MAR"

#: Ticker for the weather divergence-test market.
_WEATHER_TICKER = "KXWEATHER-25JUL-NYC90"

#: Ticker for the presidential-election divergence-test market.
_ELECTIONS_TICKER = "KXPRES-28-DEM"


def _diverse_market(
    *,
    ticker: str,
    event_ticker: str,
    title: str,
    resolution_criteria: str,
    category: str,
    close_time: datetime,
) -> NormalizedMarket:
    """Build one diverse `NormalizedMarket` for the divergence tests (#191).

    Every field not named as a parameter is fixed to the same shared default
    the `market` fixture above uses, so only the caller-supplied fields vary
    between the three divergence-test markets.

    Args:
        ticker: The market's ticker.
        event_ticker: The market's parent event ticker.
        title: The market's question text.
        resolution_criteria: The market's resolution-criteria prose.
        category: The market's category.
        close_time: The market's close instant.

    Returns:
        A valid `NormalizedMarket`.
    """
    return NormalizedMarket(
        exchange="fake-exchange",
        ticker=ticker,
        event_ticker=event_ticker,
        title=title,
        resolution_criteria=resolution_criteria,
        category=category,
        close_time=close_time,
        expected_resolution_time=None,
        market_type="fully_collateralized_binary",
        price_tick_pips=100,
        min_order_contract_centis=100,
        fractional_trading_enabled=False,
        mutually_exclusive_group_id=None,
        jurisdiction_status="eligible",
        raw_exchange_payload_hash="sha256:abc123",
        volume_24h_micros=0,
    )


def _diverse_baseline(
    *, snapshot_id: str, price_pips: int, fetched_at: datetime
) -> BaselineQuoteSnapshot:
    """Build one diverse `BaselineQuoteSnapshot` for the divergence tests (#191).

    Args:
        snapshot_id: The baseline snapshot's unique identifier.
        price_pips: The baseline executable price, in pips.
        fetched_at: When the baseline snapshot was taken.

    Returns:
        A valid `BaselineQuoteSnapshot`.
    """
    return BaselineQuoteSnapshot(
        snapshot_id=snapshot_id, price_pips=price_pips, fetched_at=fetched_at
    )


@pytest.fixture
def diverse_markets(
    created_at: datetime,
) -> tuple[tuple[NormalizedMarket, BaselineQuoteSnapshot], ...]:
    """Provide three diverse (market, baseline) pairs for divergence tests (#191).

    A Fed-rate market, a weather market, and a presidential-election market,
    each with a distinct baseline price so `baseline_ppm` (`price_pips *
    100`) differs across all three -- see the module docstring's
    "Divergence-cassette fixture choice" note.
    """
    fed = _diverse_market(
        ticker=_FED_TICKER,
        event_ticker="KXFED-25",
        title="Fed cuts rates in March 2025?",
        resolution_criteria=(
            "Resolves YES if the FOMC cuts the federal funds rate at its "
            "March 2025 meeting."
        ),
        category="economics",
        close_time=datetime(2025, 3, 19, 19, tzinfo=UTC),
    )
    weather = _diverse_market(
        ticker=_WEATHER_TICKER,
        event_ticker="KXWEATHER-25JUL",
        title="Will NYC hit 90F on July 15, 2025?",
        resolution_criteria=(
            "Resolves YES if the NWS-reported high temperature at Central "
            "Park exceeds 90F on 2025-07-15."
        ),
        category="weather",
        close_time=datetime(2025, 7, 15, 23, tzinfo=UTC),
    )
    elections = _diverse_market(
        ticker=_ELECTIONS_TICKER,
        event_ticker="KXPRES-28",
        title="Will the Democratic nominee win the 2028 presidential election?",
        resolution_criteria=(
            "Resolves YES if the Democratic Party's nominee wins a majority "
            "of the Electoral College in the 2028 U.S. presidential election."
        ),
        category="elections",
        close_time=datetime(2028, 11, 7, 23, tzinfo=UTC),
    )
    return (
        (
            fed,
            _diverse_baseline(
                snapshot_id="snap-fed-0001", price_pips=4500, fetched_at=created_at
            ),
        ),
        (
            weather,
            _diverse_baseline(
                snapshot_id="snap-weather-0001", price_pips=3000, fetched_at=created_at
            ),
        ),
        (
            elections,
            _diverse_baseline(
                snapshot_id="snap-elections-0001",
                price_pips=6000,
                fetched_at=created_at,
            ),
        ),
    )


#: Hand-authored, schema-valid #184 vote JSON responses per divergence-test
#: market (issue #191), keyed by ticker. Each market's three responses carry
#: mutually distinct `probability_ppm` values, and at least one *non-abstaining*
#: (hence surviving) response diverges from that market's own `baseline_ppm`
#: (`price_pips * 100`) by more than 20_000 ppm -- so the divergence survives
#: the #241 abstention exclusion. Exactly one response across the whole mapping
#: (the elections market's third vote) carries `"abstain": true`, exercising
#: that schema branch end-to-end through the cassette harness; the elections
#: market's divergent reading therefore sits on a surviving vote, not the
#: abstaining one.
DIVERGENT_VOTE_RESPONSES: dict[str, tuple[str, str, str]] = {
    _FED_TICKER: (
        '{"probability_ppm": 440000, "rationale_summary": '
        '"base rate holds steady", "abstain": false}',
        '{"probability_ppm": 455000, "rationale_summary": '
        '"recent dot plot shift", "abstain": false}',
        '{"probability_ppm": 490000, "rationale_summary": '
        '"market pricing in a cut", "abstain": false}',
    ),
    _WEATHER_TICKER: (
        '{"probability_ppm": 310000, "rationale_summary": '
        '"climatological base rate", "abstain": false}',
        '{"probability_ppm": 325000, "rationale_summary": '
        '"forecast model ensemble mean", "abstain": false}',
        '{"probability_ppm": 350000, "rationale_summary": '
        '"heat dome signal building", "abstain": false}',
    ),
    _ELECTIONS_TICKER: (
        '{"probability_ppm": 590000, "rationale_summary": '
        '"polling average steady", "abstain": false}',
        '{"probability_ppm": 635000, "rationale_summary": '
        '"incumbent-party tailwind", "abstain": false}',
        '{"probability_ppm": 560000, "rationale_summary": '
        '"insufficient signal this far out", "abstain": true}',
    ),
}
