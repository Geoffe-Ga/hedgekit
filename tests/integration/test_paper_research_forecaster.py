"""The hosted research forecaster, reached from the shipped CLI (issue #555).

``FutureSearchProvider`` (SPEC S8.9, ADR-0005 family (b)) was fully built and
fully tested and **reachable from nowhere**: a live ensemble naming
``futuresearch`` refused at startup, ``ForecastConfig.futuresearch`` was read by
nothing in ``windbreak/``, and the engine priced a call the composition root
could not make. This module is the proof that all three are closed, and it is
deliberately written to fail if any of them reopens.

Every test here drives the **real** composition root -- a committed YAML
configuration through ``load_config``, then ``build_paper_deps`` and
``run_single_tick`` -- rather than constructing a provider by hand. #343 is the
named precedent for why: an adapter with thorough unit coverage and no
composition test is an adapter nobody can run, and its unit tests say nothing
about that. The only thing this suite fabricates is the *transport*, exactly the
seam ``windbreak.main`` fills with a credentialed live one; the provider itself
is built by the code under test or not at all.

Nothing here reaches a network: the research-forecaster responses come from a
committed HTTP cassette keyed by the real request hashes the composition root
produces.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.integration.conftest import (
    FIXED_NOW_EPOCH_S,
    ledger_path_for,
    read_event_type_payload_pairs,
)
from windbreak.config.loader import load_config
from windbreak.forecast.providers import (
    FixtureVoteProvider,
    FutureSearchProvider,
    HttpResponse,
    ReplayHttpCassette,
    RetryingProvider,
)

if TYPE_CHECKING:
    from windbreak.config.schema import WindbreakConfig
    from windbreak.forecast.providers import HttpRequest

#: The committed operator configuration under test -- the one a human could
#: write, loaded through the real loader rather than assembled in Python.
_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "config"
    / ("research-forecaster.yaml")
)

#: The committed recorded responses, keyed by the real ``HttpRequest``
#: hashes the composition root produces for vote 0 and vote 1.
_CASSETTE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "forecast"

#: The recorded run in which both responses report their own ``cost_usd``.
_COSTED_CASSETTE = _CASSETTE_DIR / "futuresearch_tick_cassette.json"

#: The recorded run in which neither response reports a ``cost_usd`` at all.
_UNCOSTED_CASSETTE = _CASSETTE_DIR / "futuresearch_tick_cassette_uncosted.json"

#: The probability both recorded responses report, in ppm. Nothing the offline
#: path could produce by accident, so seeing it in the ledger proves it came off
#: a research-forecaster response.
_RESEARCH_VOTE_PPM = 723_456

#: The two recorded ``cost_usd`` figures, converted to micros. Deliberately
#: *unequal to each other*, so a single canned response served to both votes
#: could not satisfy the assertion, and deliberately unlike both the configured
#: per-call ceiling and the schema default.
_VOTE_0_COST_MICROS = 370_000
_VOTE_1_COST_MICROS = 110_000

#: The ceiling the committed config pins, charged when a response declines to
#: report its cost. Unequal to the schema's own 2_000_000 default and to both
#: recorded costs, so an assertion on it cannot be satisfied by a coincidence.
_CONFIGURED_CEILING_MICROS = 1_250_000

#: The provider identifier a research-forecaster vote is stamped with.
_FUTURESEARCH = "futuresearch"

#: The ledger events a recorded per-provider vote and a produced forecast append.
_VOTE_EVENT = "ProviderVoteRecorded"
_FORECAST_EVENT = "ForecastCreated"


def _fixed_clock() -> int:
    """Return the suite's fixed epoch second, for determinism.

    Returns:
        ``FIXED_NOW_EPOCH_S``.
    """
    return FIXED_NOW_EPOCH_S


def _config() -> WindbreakConfig:
    """Load the committed research-forecaster configuration.

    Returns:
        The configuration an operator could have written, parsed by the real
        loader.
    """
    return load_config(_CONFIG_PATH)


class _NeverCalledHttpTransport:
    """An `HttpTransport` proving a seam is never dialed."""

    def send(self, request: HttpRequest) -> HttpResponse:
        """Fail loudly: reaching this is itself the bug under test.

        Args:
            request: The (rejected) HTTP request.

        Raises:
            AssertionError: Always.
        """
        msg = f"unexpectedly dialed {request.url!r}"
        raise AssertionError(msg)


class _FixtureSearchTransport:
    """Deterministic, network-free `SearchTransport` yielding one candidate URL.

    Mirrors ``tests/integration/test_paper_live_providers.py``'s double: the
    shared offline research tools find nothing, so a pipeline driven by them
    abstains on zero verified citations *before* the vote stage and the provider
    is never reached at all (SPEC S8.8). Reaching a research-forecaster vote
    therefore requires research that genuinely gathers and verifies.
    """

    def search(self, query: str) -> tuple[str, ...]:
        """Return one deterministic candidate URL derived from ``query``.

        Args:
            query: The subquestion text being searched for.

        Returns:
            A one-element tuple holding a URL on ``research.local``.
        """
        import hashlib

        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        return (f"https://research.local/{digest}",)


class _FixtureFetchTransport:
    """Deterministic `FetchTransport` whose content depends only on the URL."""

    def fetch(self, url: str) -> str:
        """Return deterministic canned content for ``url``.

        Args:
            url: The URL being fetched.

        Returns:
            A deterministic content string derived from ``url``.
        """
        return f"fixture content for {url}"


def _citation_producing_research_tools(tmp_path: Path) -> object:
    """Build research tools that gather and verify real citations.

    Args:
        tmp_path: The pytest scratch directory the research cache lives under.

    Returns:
        A `ResearchTools` bundle that never touches the network.
    """
    from windbreak.config.schema import DEFAULT_RESEARCH_CACHE_MAX_BYTES
    from windbreak.forecast.sandbox import build_research_tools

    return build_research_tools(
        allowed_hosts=frozenset({"research.local"}),
        cache_dir=tmp_path / "research-cache",
        search_transport=_FixtureSearchTransport(),
        fetch_transport=_FixtureFetchTransport(),
        max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    )


def _live_http(futuresearch: object) -> object:
    """Build the live HTTP seam bundle carrying a research-forecaster transport.

    Args:
        futuresearch: The HTTP transport the research forecaster is dialed
            through.

    Returns:
        A `LiveProviderHttp` bundle whose LLM seams are empty -- the committed
        ensemble names no completion-transport provider, so building one would
        demand a credential nothing in this configuration needs.
    """
    from windbreak.scheduler.provider_wiring import LiveProviderHttp

    return LiveProviderHttp(
        llm={},
        search=_NeverCalledHttpTransport(),
        fetch=_NeverCalledHttpTransport(),
        futuresearch=futuresearch,
    )


def _build_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    config: WindbreakConfig,
    provider_http: object,
) -> object:
    """Build one `PaperTickDeps` over the shared offline fixtures.

    Args:
        books_dir: The books-fixture directory.
        cassette_path: The recorded LLM-cassette path (unused on this path).
        report_dir: Where weekly-report stubs would be written.
        tmp_path: The pytest scratch directory.
        config: The configuration under test.
        provider_http: The live HTTP seam bundle.

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
        research_tools=_citation_producing_research_tools(tmp_path),
        clock=_fixed_clock,
        provider_http=provider_http,
    )


def _run_tick(
    *,
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
    cassette: Path,
    config: WindbreakConfig | None = None,
) -> list[tuple[str, dict[str, object]]]:
    """Run one whole tick against a committed research-forecaster cassette.

    Args:
        books_dir: The books-fixture directory.
        cassette_path: The recorded LLM-cassette path.
        report_dir: Where weekly-report stubs would be written.
        tmp_path: The pytest scratch directory.
        cassette: The committed HTTP cassette the research forecaster replays.
        config: The configuration under test, or ``None`` for the committed one.

    Returns:
        The tick's ``(event_type, payload)`` ledger rows.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=config if config is not None else _config(),
        provider_http=_live_http(ReplayHttpCassette.from_path(cassette)),
    )
    run_single_tick(deps, beat=1)
    return read_event_type_payload_pairs(deps.store.read_all())


def _vote_rows(
    pairs: list[tuple[str, dict[str, object]]],
) -> list[dict[str, object]]:
    """Return the per-provider vote rows from a tick's ledger.

    Args:
        pairs: The tick's ``(event_type, payload)`` rows.

    Returns:
        Every ``ProviderVoteRecorded`` payload, in ledger order.
    """
    return [payload for event_type, payload in pairs if event_type == _VOTE_EVENT]


# --- AC1: the shipped composition root really reaches the provider ----------------


def test_a_live_tick_ledgers_research_forecaster_votes_from_recorded_responses(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A committed configuration reaches `FutureSearchProvider` end to end.

    The issue this closes in one assertion: before it, no configuration reached
    this provider through ``windbreak run`` at all. Both votes are asserted, and
    their *distinct* recorded costs prove each vote replayed its own recorded
    response rather than one canned body served twice.
    """
    pairs = _run_tick(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        cassette=_COSTED_CASSETTE,
    )
    votes = _vote_rows(pairs)

    assert [vote["provider"] for vote in votes] == [_FUTURESEARCH, _FUTURESEARCH]
    assert all(vote["outcome"] == "voted" for vote in votes)
    assert sorted(vote["vote_index"] for vote in votes) == [0, 1]


def test_the_ledgered_probability_comes_off_the_research_forecaster_response(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """The forecast's probability is the one the recorded response reported."""
    pairs = _run_tick(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        cassette=_COSTED_CASSETTE,
    )
    forecasts = [
        payload for event_type, payload in pairs if event_type == _FORECAST_EVENT
    ]

    assert [f["probability_ppm"] for f in forecasts] == [_RESEARCH_VOTE_PPM]


def test_the_composition_root_builds_a_research_forecaster_not_a_vote_provider(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """The factory the root returns builds the research forecaster itself.

    Not the `FixtureVoteProvider` the completion seam builds for an LLM member,
    and not a `RetryingProvider` around one: a research forecaster reports its
    own cost, so wrapping it in the metering layer would add a fabricated
    charge on top of a measured one.
    """
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        config=_config(),
        provider_http=_live_http(_NeverCalledHttpTransport()),
    )
    member = deps.config.forecast.vote_ensemble[0]

    provider = deps.provider_factory(deps.transport, member)

    assert isinstance(provider, FutureSearchProvider)
    assert not isinstance(provider, FixtureVoteProvider | RetryingProvider)


# --- AC5: the reported cost, and the fail-closed fallback -------------------------


def test_each_vote_books_the_cost_its_own_recorded_response_reported(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """The booked money is the *reported* ``cost_usd``, converted to micros.

    This is the whole reason the research forecaster is worth reaching: it is
    the only forecast path whose cost is a measurement rather than a constant.
    The two recorded figures are unequal, so a per-vote assertion cannot be
    satisfied by one shared value.
    """
    pairs = _run_tick(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        cassette=_COSTED_CASSETTE,
    )

    costs = sorted(vote["cost_micros"] for vote in _vote_rows(pairs))

    assert costs == [_VOTE_1_COST_MICROS, _VOTE_0_COST_MICROS]


def test_a_response_reporting_no_cost_books_the_configured_ceiling_not_zero(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A silent response charges ``per_call_ceiling_micros``, never zero.

    The configured ceiling is unlike the schema default *and* unlike either
    recorded cost, so passing this proves the operator's own figure reached the
    provider -- not a constant that happens to agree.
    """
    pairs = _run_tick(
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
        cassette=_UNCOSTED_CASSETTE,
    )

    costs = [vote["cost_micros"] for vote in _vote_rows(pairs)]

    assert costs == [_CONFIGURED_CEILING_MICROS, _CONFIGURED_CEILING_MICROS]
    assert 0 not in costs


# --- AC4: a misconfigured research forecaster refuses, naming the leaf ------------


def _misconfigured(**futuresearch: object) -> WindbreakConfig:
    """Return the committed config with `forecast.futuresearch` fields replaced.

    Args:
        **futuresearch: The `FutureSearchProviderSettings` fields to override.

    Returns:
        The mutated configuration.
    """
    base = _config()
    settings = dataclasses.replace(base.forecast.futuresearch, **futuresearch)
    return dataclasses.replace(
        base, forecast=dataclasses.replace(base.forecast, futuresearch=settings)
    )


def _build_refused(
    config: WindbreakConfig,
    *,
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> str:
    """Build deps expecting a startup refusal, returning its message.

    Args:
        config: The misconfiguration under test.
        books_dir: The books-fixture directory.
        cassette_path: The recorded LLM-cassette path.
        report_dir: Where weekly-report stubs would be written.
        tmp_path: The pytest scratch directory.

    Returns:
        The refusal message.
    """
    with pytest.raises(ValueError) as excinfo:
        _build_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            report_dir=report_dir,
            tmp_path=tmp_path,
            config=config,
            provider_http=_live_http(_NeverCalledHttpTransport()),
        )
    return str(excinfo.value)


def test_an_unconfigured_endpoint_refuses_to_start_naming_the_leaf(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A live research forecaster left on the operator placeholder refuses.

    Naming the leaf, not the provider: an operator who selected the member and
    forgot the endpoint needs to be told which line to fill in.
    """
    from windbreak.config.schema import UNCONFIGURED_PLACEHOLDER

    message = _build_refused(
        _misconfigured(endpoint_url=UNCONFIGURED_PLACEHOLDER),
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
    )

    assert "forecast.futuresearch.endpoint_url" in message


def test_a_missing_key_variable_name_refuses_to_start_naming_the_leaf(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A blank ``api_key_env`` refuses rather than dialing unauthenticated.

    Configuration names the variable a key is read from; a blank name is not a
    credential-free deployment, it is an unfinished one.
    """
    message = _build_refused(
        _misconfigured(api_key_env=""),
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
    )

    assert "forecast.futuresearch.api_key_env" in message


def test_a_member_version_off_the_pinned_set_refuses_to_start(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A drifted forecaster version refuses at startup, not per vote.

    ``ProviderVoteRecorded`` stamps the *member's* ``model_version`` while
    ``ModelVote`` stamps the version the *response* reported. If the ensemble
    pins one version and ``pinned_forecaster_versions`` admits another, those
    two rows describe different models for the same forecast -- an incoherent
    audit trail. Refusing here is what keeps them the same string.
    """
    base = _config()
    settings = dataclasses.replace(
        base.forecast.futuresearch, pinned_forecaster_versions=("fs-2099-01-01",)
    )
    config = dataclasses.replace(
        base, forecast=dataclasses.replace(base.forecast, futuresearch=settings)
    )

    message = _build_refused(
        config,
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
    )

    assert "fs-2026-06-01" in message
    assert "forecast.futuresearch.pinned_forecaster_versions" in message


def test_a_zero_per_call_ceiling_refuses_to_start(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A zero fail-closed fallback would book a silent response as free."""
    message = _build_refused(
        _misconfigured(per_call_ceiling_micros=0),
        books_dir=books_dir,
        cassette_path=cassette_path,
        report_dir=report_dir,
        tmp_path=tmp_path,
    )

    assert "forecast.futuresearch.per_call_ceiling_micros" in message


def test_a_live_research_forecaster_without_its_http_seam_refuses_to_start(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """Selecting the member without building its transport must not degrade.

    The other half-configuration: configuration said live research forecaster,
    the composition root supplied no seam for it. Silently falling back to a
    completion-transport provider would vote through a seam the operator never
    selected.
    """
    from windbreak.scheduler.provider_wiring import LiveProviderHttp

    with pytest.raises(ValueError) as excinfo:
        _build_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            report_dir=report_dir,
            tmp_path=tmp_path,
            config=_config(),
            provider_http=LiveProviderHttp(
                llm={},
                search=_NeverCalledHttpTransport(),
                fetch=_NeverCalledHttpTransport(),
                futuresearch=None,
            ),
        )

    assert "futuresearch" in str(excinfo.value)
