"""Live provider transport selection at the PAPER composition root (#344, #269).

Before this, ``build_paper_deps`` hardwired ``transport=ReplayCassette
.from_path(cassette_path)``: epic #183 shipped real providers -- the
research-forecaster adapter, the live pinned-LLM transports, live search/fetch
-- and *none* of them was reachable from ``windbreak run``, because the
composition root had no seam to choose one. The running product replayed
recorded LLM responses forever, so it could not produce a novel forecast, so it
could not disagree with the market.

These tests pin the seam and the two guarantees that make it safe to add:

**The cassette stays the CI default.** ``WindbreakConfig()`` selects the
recorded replay transport and the offline no-network research tools. CI builds
exactly that config, so the default path acquires no network dependency by
omission. Going live is an explicit written act in configuration *and* an
explicitly supplied live HTTP seam -- and a half-configuration of either kind
refuses to start rather than degrading, exactly as the ``market_data`` /
``live_ticker`` pair already does for live books (issue #343).

**The retry/pricing hardening is active wherever it can matter.** Issue #269's
``RetryingProvider``, its config-sourced ``RetryPolicy``, and its fail-closed
``ProviderPriceTable`` wrap every live vote. The cassette path is deliberately
*not* wrapped: no money is spent replaying a recording, and charging a list
price for a replayed call would corrupt the very cost accounting the price
table exists to keep honest.

Everything here runs offline against stub HTTP transports; no test reaches a
network. (Byte-level cassette replay itself is covered by
``tests/forecast/providers/test_http_cassettes.py``.)
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from typing import TYPE_CHECKING

import pytest

from tests.integration.conftest import (
    FIXED_NOW_EPOCH_S,
    FIXTURE_SCREENER_CONFIG,
    ledger_path_for,
    read_event_type_payload_pairs,
)
from windbreak.config.schema import (
    DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    PROVIDER_TRANSPORT_CASSETTE,
    PROVIDER_TRANSPORT_LIVE,
    CapitalConfig,
    ForecastConfig,
    ProviderRetryConfig,
    ProviderTransportConfig,
    RiskConfig,
    WindbreakConfig,
)
from windbreak.forecast.providers import (
    FixtureVoteProvider,
    HttpResponse,
    ProviderHTTPError,
    RetryingProvider,
)

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.forecast.providers import HttpRequest

#: The probability the canned live provider responses vote, in ppm. Chosen to
#: be nothing the offline path could produce by accident, so seeing it in the
#: record proves the probability came from a *provider response*.
_LIVE_VOTE_PPM = 723_456

#: The two providers the default vote ensemble draws from.
_ANTHROPIC = "anthropic"
_OPENAI = "openai"

#: The ledger event a recorded per-provider vote appends.
_VOTE_EVENT = "ProviderVoteRecorded"

#: The ledger event a produced forecast appends.
_FORECAST_EVENT = "ForecastCreated"

#: The sole market the shared `deep_walk` books fixture offers, and so the one
#: market a tick here screens in. Named locally because since issue #345 the
#: deps bundle screens a universe instead of carrying a ticker of its own.
_TICKER = "MKT-DEEP"


def _fixed_clock() -> int:
    """Return the suite's fixed epoch second, for determinism.

    Returns:
        `FIXED_NOW_EPOCH_S`.
    """
    return FIXED_NOW_EPOCH_S


def _vote_json() -> str:
    """Build a schema-valid #184 vote response body.

    Returns:
        The vote JSON a provider would return.
    """
    return json.dumps(
        {
            "probability_ppm": _LIVE_VOTE_PPM,
            "rationale_summary": "canned live-path response for wiring proof",
            "abstain": False,
        }
    )


def _anthropic_envelope() -> str:
    """Wrap the vote JSON in an Anthropic Messages envelope.

    Returns:
        The raw response body text.
    """
    return json.dumps({"content": [{"type": "text", "text": _vote_json()}]})


def _openai_envelope() -> str:
    """Wrap the vote JSON in an OpenAI chat-completion envelope.

    Returns:
        The raw response body text.
    """
    return json.dumps({"choices": [{"message": {"content": _vote_json()}}]})


class _CannedHttpTransport:
    """An `HttpTransport` returning one fixed response to every request."""

    def __init__(self, body: str, *, status_code: int = 200) -> None:
        """Store the fixed response.

        Args:
            body: The raw response body text.
            status_code: The HTTP status code to report.
        """
        self._body = body
        self._status_code = status_code
        self.calls = 0

    def send(self, request: HttpRequest) -> HttpResponse:
        """Count the call and return the fixed response.

        Args:
            request: The (unused) HTTP request.

        Returns:
            The fixed `HttpResponse`.
        """
        del request
        self.calls += 1
        return HttpResponse(self._status_code, self._body)


class _FixtureSearchTransport:
    """Deterministic, network-free `SearchTransport` yielding one candidate URL.

    Mirrors `tests/integration/test_provider_vote_costing.py`'s local double:
    the shared `research_tools_factory` finds nothing, so a pipeline driven by
    it abstains on zero verified citations *before* the vote stage and never
    touches the transport at all (SPEC S8.8). Reaching the live vote stage
    therefore requires research that genuinely gathers and verifies.
    """

    def search(self, query: str) -> tuple[str, ...]:
        """Return one deterministic candidate URL derived from `query`.

        Args:
            query: The subquestion text being searched for.

        Returns:
            A one-element tuple holding a URL on `research.local`.
        """
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        return (f"https://research.local/{digest}",)


class _FixtureFetchTransport:
    """Deterministic `FetchTransport` whose content depends only on the URL.

    Byte-identical across the citation-building fetch and the verification
    refetch, so every gathered citation verifies cleanly.
    """

    def fetch(self, url: str) -> str:
        """Return deterministic canned content for `url`.

        Args:
            url: The URL being fetched.

        Returns:
            A deterministic content string derived from `url`.
        """
        return f"fixture content for {url}"


def _citation_producing_research_tools(tmp_path: Path) -> object:
    """Build research tools that gather and verify real citations.

    Args:
        tmp_path: The pytest scratch directory the research cache lives under.

    Returns:
        A `ResearchTools` bundle that never touches the network.
    """
    from windbreak.forecast.sandbox import build_research_tools

    return build_research_tools(
        allowed_hosts=frozenset({"research.local"}),
        cache_dir=tmp_path / "research-cache",
        search_transport=_FixtureSearchTransport(),
        fetch_transport=_FixtureFetchTransport(),
        max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    )


class _NeverCalledHttpTransport:
    """An `HttpTransport` proving the offline path never dials it."""

    def send(self, request: HttpRequest) -> HttpResponse:
        """Fail loudly: reaching this is itself the bug under test.

        Args:
            request: The (rejected) HTTP request.

        Raises:
            AssertionError: Always.
        """
        msg = f"offline path unexpectedly dialed {request.url!r}"
        raise AssertionError(msg)


def _live_http(
    *, anthropic: object | None = None, openai: object | None = None
) -> object:
    """Build the live HTTP seam bundle the composition root wires from.

    Args:
        anthropic: The Anthropic HTTP transport, or `None` for a canned one.
        openai: The OpenAI HTTP transport, or `None` for a canned one.

    Returns:
        A `LiveProviderHttp` bundle.
    """
    from windbreak.scheduler.provider_wiring import LiveProviderHttp

    return LiveProviderHttp(
        llm={
            _ANTHROPIC: anthropic or _CannedHttpTransport(_anthropic_envelope()),
            _OPENAI: openai or _CannedHttpTransport(_openai_envelope()),
        },
        search=_NeverCalledHttpTransport(),
        fetch=_NeverCalledHttpTransport(),
    )


def _config(
    *, mode: str = PROVIDER_TRANSPORT_CASSETTE, **retry: int
) -> WindbreakConfig:
    """Build a PAPER-ceilinged config selecting a provider transport mode.

    Carries `FIXTURE_SCREENER_CONFIG` for the reason set out in
    `tests/integration/conftest.py`: since issue #345 a tick forecasts only
    markets that pass the screen, and `deep_walk` fails the production depth and
    horizon defaults -- so without it the end-to-end scenarios below would
    forecast nothing and prove nothing about the provider transport.

    Args:
        mode: The provider transport mode to select.
        **retry: `ProviderRetryConfig` field overrides.

    Returns:
        The configuration under test.
    """
    return WindbreakConfig(
        mode_ceiling="paper",
        capital=CapitalConfig(floor_micros=0),
        risk=RiskConfig(),
        screener=FIXTURE_SCREENER_CONFIG,
        forecast=ForecastConfig(
            # No shrink-to-market, so an aggregated probability that equals the
            # canned vote proves it came off the provider response rather than
            # being pulled back toward the book.
            shrink_to_market_lambda_ppm=0,
            provider_transport=ProviderTransportConfig(
                mode=mode, retry=ProviderRetryConfig(**retry)
            ),
        ),
    )


def _build_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    config: WindbreakConfig,
    research_tools: object = None,
    provider_http: object = None,
) -> object:
    """Build one `PaperTickDeps` over the shared offline fixtures.

    Args:
        books_dir: The books-fixture directory.
        cassette_path: The recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        tmp_path: The pytest scratch directory.
        config: The configuration under test.
        research_tools: Explicit research tools, or `None` for the resolved
            default.
        provider_http: The live HTTP seam bundle, or `None`.

    Returns:
        A fully wired `PaperTickDeps`.
    """
    from windbreak.scheduler.loop import build_paper_deps

    return build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=config,
        research_tools=research_tools,
        clock=_fixed_clock,
        provider_http=provider_http,
    )


# --- The cassette is the CI default ----------------------------------------------


def test_the_default_config_wires_the_replay_cassette(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """An omitted transport section leaves the offline replay path in place."""
    from windbreak.forecast.cassettes import ReplayCassette

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(),
        research_tools=research_tools_factory(),
    )

    assert isinstance(deps.transport, ReplayCassette)


def test_the_default_config_needs_no_live_http_seam(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """CI builds the default config and supplies nothing; that must just work."""
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(),
        research_tools=research_tools_factory(),
    )

    assert deps.config.forecast.provider_transport.mode == PROVIDER_TRANSPORT_CASSETTE


def test_the_cassette_path_builds_an_unwrapped_fixture_provider(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """Replaying a recording spends no money, so it is not priced or retried.

    Wrapping it would charge a list price for a call that never happened,
    corrupting the very cost accounting the price table exists to keep honest.
    """
    from windbreak.forecast.providers import DEFAULT_VOTE_ENSEMBLE

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(),
        research_tools=research_tools_factory(),
    )

    provider = deps.provider_factory(deps.transport, DEFAULT_VOTE_ENSEMBLE[0])

    assert isinstance(provider, FixtureVoteProvider)
    assert not isinstance(provider, RetryingProvider)


# --- Fail closed on a half-configuration -----------------------------------------


def test_live_mode_without_a_live_http_seam_refuses_to_start(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """Configured live but wired offline must refuse, never quietly replay.

    An operator who configured live providers and silently got a recording
    would be reading a paper tape while believing it was a novel forecast.
    """
    with pytest.raises(ValueError, match="provider_http"):
        _build_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            report_dir=report_dir,
            tmp_path=tmp_path,
            config=_config(mode=PROVIDER_TRANSPORT_LIVE),
            research_tools=research_tools_factory(),
        )


def test_a_live_http_seam_without_live_mode_refuses_to_start(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """The other half-configuration refuses too: supplied but unselected."""
    with pytest.raises(ValueError, match="cassette"):
        _build_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            report_dir=report_dir,
            tmp_path=tmp_path,
            config=_config(),
            research_tools=research_tools_factory(),
            provider_http=_live_http(),
        )


def test_an_unrecognized_transport_mode_refuses_to_start(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """An unknown mode is an operator error, not a reason to pick a default."""
    with pytest.raises(ValueError, match="unknown"):
        _build_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            report_dir=report_dir,
            tmp_path=tmp_path,
            config=_config(mode="semi-live"),
            research_tools=research_tools_factory(),
        )


# --- The live path ----------------------------------------------------------------


def test_live_mode_wires_a_transport_that_reaches_the_live_adapters(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """The bundle's transport routes a completion to the live provider adapter."""
    from windbreak.forecast.cassettes import LlmRequest

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(mode=PROVIDER_TRANSPORT_LIVE),
        research_tools=research_tools_factory(),
        provider_http=_live_http(),
    )

    completion = deps.transport.complete(
        LlmRequest(provider=_ANTHROPIC, model_version="pinned", prompt="p")
    )

    assert json.loads(completion.text)["probability_ppm"] == _LIVE_VOTE_PPM


def test_each_provider_routes_to_its_own_live_transport(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """One vendor's prompt -- and key -- must never reach another's endpoint."""
    from windbreak.forecast.cassettes import LlmRequest

    anthropic = _CannedHttpTransport(_anthropic_envelope())
    openai = _CannedHttpTransport(_openai_envelope())
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(mode=PROVIDER_TRANSPORT_LIVE),
        research_tools=research_tools_factory(),
        provider_http=_live_http(anthropic=anthropic, openai=openai),
    )

    deps.transport.complete(
        LlmRequest(provider=_OPENAI, model_version="pinned", prompt="p")
    )

    assert openai.calls == 1
    assert anthropic.calls == 0


def test_live_mode_wraps_every_vote_in_the_retrying_provider(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """Issue #269's hardening is active on the path where money is actually spent."""
    from windbreak.forecast.providers import DEFAULT_VOTE_ENSEMBLE

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(mode=PROVIDER_TRANSPORT_LIVE),
        research_tools=research_tools_factory(),
        provider_http=_live_http(),
    )

    provider = deps.provider_factory(deps.transport, DEFAULT_VOTE_ENSEMBLE[0])

    assert isinstance(provider, RetryingProvider)


def test_the_configured_retry_policy_reaches_the_wrapped_provider(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """An operator-raised attempt count is the count that is actually attempted.

    `max_attempts=1` means no retry at all, so a canned 503 is dialed exactly
    once -- proving the *config* value drove the policy rather than the
    engine's own default of three.
    """
    from windbreak.forecast.providers import DEFAULT_VOTE_ENSEMBLE

    failing = _CannedHttpTransport("upstream unavailable", status_code=503)
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(mode=PROVIDER_TRANSPORT_LIVE, max_attempts=1),
        research_tools=research_tools_factory(),
        provider_http=_live_http(anthropic=failing),
    )
    member = next(m for m in DEFAULT_VOTE_ENSEMBLE if m.provider == _ANTHROPIC)

    with pytest.raises(ProviderHTTPError):
        deps.provider_factory(deps.transport, member).forecast(
            deps.exchange.get_market(_TICKER),
            _baseline(deps),
            0,
            (),
        )

    assert failing.calls == 1


def _baseline(deps: object) -> object:
    """Build a baseline quote snapshot for a direct provider call.

    Args:
        deps: The dependency bundle whose exchange the `_TICKER` book is read
            from.

    Returns:
        A `BaselineQuoteSnapshot`.
    """
    from windbreak.forecast.records import BaselineQuoteSnapshot

    book = deps.exchange.get_order_book(_TICKER)
    return BaselineQuoteSnapshot(
        snapshot_id=f"{_TICKER}-baseline",
        price_pips=50_000,
        fetched_at=book.fetched_at,
    )


def test_a_retryable_fault_is_retried_up_to_the_configured_attempts(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """A transient 503 really is retried -- the #269 hardening is not inert."""
    from windbreak.forecast.providers import DEFAULT_VOTE_ENSEMBLE

    failing = _CannedHttpTransport("upstream unavailable", status_code=503)
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(
            mode=PROVIDER_TRANSPORT_LIVE,
            max_attempts=3,
            backoff_base_ms=1,
            total_deadline_ms=60_000,
            max_cost_micros=10_000_000,
        ),
        research_tools=research_tools_factory(),
        provider_http=_live_http(anthropic=failing),
    )
    member = next(m for m in DEFAULT_VOTE_ENSEMBLE if m.provider == _ANTHROPIC)

    with pytest.raises(ProviderHTTPError):
        deps.provider_factory(deps.transport, member).forecast(
            deps.exchange.get_market(_TICKER), _baseline(deps), 0, ()
        )

    assert failing.calls == 3


# --- Structural guarantees --------------------------------------------------------


def test_there_is_no_provider_factory_injection_door() -> None:
    """Config plus the live seam are the only sources, as with budget/gate.

    With no `provider_factory` parameter there is no door through which an
    unpriced, unretried live provider could arrive.
    """
    from windbreak.scheduler.loop import build_paper_deps

    assert "provider_factory" not in inspect.signature(build_paper_deps).parameters


def test_the_bundle_always_carries_a_provider_factory(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """The field is non-optional, so an unwired vote stage is unrepresentable."""
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(),
        research_tools=research_tools_factory(),
    )

    assert deps.provider_factory is not None


# --- End to end: a live tick's probability comes from provider responses ----------


def test_a_live_tick_ledgers_a_forecast_built_from_provider_responses(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """The whole point of #344: a tick whose probability came off the wire."""
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(mode=PROVIDER_TRANSPORT_LIVE),
        research_tools=_citation_producing_research_tools(tmp_path),
        provider_http=_live_http(),
    )

    run_single_tick(deps, beat=1)

    pairs = read_event_type_payload_pairs(deps.store.read_all())
    votes = [payload for event_type, payload in pairs if event_type == _VOTE_EVENT]
    forecasts = [
        payload for event_type, payload in pairs if event_type == _FORECAST_EVENT
    ]

    assert votes != []
    assert all(vote["outcome"] == "voted" for vote in votes)
    assert [f["probability_ppm"] for f in forecasts] == [_LIVE_VOTE_PPM]


def test_an_unroutable_provider_abstains_rather_than_crashing_the_tick(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A vote with no live route must not take the whole tick down.

    The regression this exists for: ``RoutingLlmTransport.complete`` raised a
    bare ``KeyError``, which neither ``RetryingProvider.forecast`` nor
    ``pipeline._collect_provider_forecasts`` catches -- both catch
    ``ProviderVoteError`` -- so it escaped ``run_pipeline`` and crashed the
    tick. Transport-level tests pinned that the ``KeyError`` was *raised*; only
    driving a whole tick shows that nothing *handles* it.

    Startup validation now refuses this configuration outright, so the bundle is
    built routable and its transport swapped for an empty router afterwards --
    exercising the defence-in-depth layer on a path that should be unreachable.
    """
    from windbreak.net.live_http import RoutingLlmTransport
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(mode=PROVIDER_TRANSPORT_LIVE),
        research_tools=_citation_producing_research_tools(tmp_path),
        provider_http=_live_http(),
    )
    unrouted = dataclasses.replace(deps, transport=RoutingLlmTransport({}))

    run_single_tick(unrouted, beat=1)

    forecasts = [
        payload
        for event_type, payload in read_event_type_payload_pairs(
            unrouted.store.read_all()
        )
        if event_type == _FORECAST_EVENT
    ]
    assert forecasts != []
    assert all(not forecast["eligible_for_live"] for forecast in forecasts)
    assert all(forecast["abstention_reason"] for forecast in forecasts)


def test_a_live_ensemble_naming_an_unroutable_provider_refuses_to_start(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """The misconfiguration is refused at composition, not discovered mid-tick.

    Reachable by an operator adding ``futuresearch`` to a live ensemble, or
    simply typo'ing a provider name.
    """
    from windbreak.config.schema import EnsembleMemberConfig

    base = _config(mode=PROVIDER_TRANSPORT_LIVE)
    forecast = dataclasses.replace(
        base.forecast,
        vote_ensemble=(
            EnsembleMemberConfig("anthropic", "claude-pinned", "2025-07-31"),
            EnsembleMemberConfig("futuresearch", "fs-pinned", "2025-07-31"),
        ),
    )

    with pytest.raises(ValueError, match="futuresearch"):
        _build_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            report_dir=report_dir,
            tmp_path=tmp_path,
            config=dataclasses.replace(base, forecast=forecast),
            research_tools=research_tools_factory(),
            provider_http=_live_http(),
        )


def test_a_total_provider_outage_abstains_rather_than_inventing_a_value(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    research_tools_factory,
) -> None:
    """A transport fault degrades to quorum abstention, never a silent number."""
    from windbreak.scheduler.loop import run_single_tick

    down = _CannedHttpTransport("upstream unavailable", status_code=503)
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(mode=PROVIDER_TRANSPORT_LIVE, backoff_base_ms=1),
        research_tools=_citation_producing_research_tools(tmp_path),
        provider_http=_live_http(anthropic=down, openai=down),
    )

    run_single_tick(deps, beat=1)

    forecasts = [
        payload
        for event_type, payload in read_event_type_payload_pairs(deps.store.read_all())
        if event_type == _FORECAST_EVENT
    ]
    assert forecasts != []
    assert all(not forecast["eligible_for_live"] for forecast in forecasts)
    assert all(forecast["abstention_reason"] for forecast in forecasts)
