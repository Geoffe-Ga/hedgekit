"""End-to-end tests for the scheduler's per-provider vote-cost ledgering
(issue #281, RED): `windbreak.scheduler.loop._forecast_stage` must append one
`ProviderVoteRecorded` event per ensemble member driven, each carrying the
tick's own `forecast_id`, immediately after its `ForecastCreated`.

`windbreak.ledger.events.ProviderVoteRecorded` does not exist yet, so the
positive-vote-reaching test below fails collection with `ImportError: cannot
import name 'ProviderVoteRecorded' from 'windbreak.ledger.events'` -- the
expected Gate 1 RED state for issue #281. The zero-events (abstain-before-
vote-stage) scenario collects fine today (it asserts an ABSENCE, not an
import) but fails with a genuine `AssertionError` once
`ProviderVoteRecorded` exists and something starts emitting it incorrectly
-- today it is a vacuous "still zero, as always" pass, which is fine: this
suite's positive test is what actually pins the missing behavior.

The same `ProviderVoteRecorded` rows are also this suite's observable for the
composition root's *ensemble* wiring (issue #294, ADR-0006): each row carries
the driving member's `provider`/`model_version`, so the rows spell out which
ensemble `_forecast_stage` actually drove -- the configured
`config.forecast.vote_ensemble` or the engine's built-in
`DEFAULT_VOTE_ENSEMBLE`.

Four scenarios:

1. `test_zero_verified_citations_path_ledgers_zero_provider_vote_recorded`
   -- the shared `tests/integration/conftest.py` offline defaults
   (`NullSearchTransport`, zero citations) abstain BEFORE the vote stage
   (`ABSTENTION_NO_VERIFIED_CITATIONS`), so `collect_model_votes` -- and
   therefore the vote-cost ledgering it will carry -- is never reached: zero
   `ProviderVoteRecorded` events, matching the pre-existing
   `FORECAST_OUTPUT_DISCARDED`-is-also-absent contract this same offline
   fixture already proves elsewhere.
2. `test_one_paper_tick_ledgers_one_provider_vote_recorded_per_ensemble_member`
   -- a real, citation-producing research double plus a real, vote-producing
   LLM transport double drive `_forecast_stage` all the way through
   aggregation: exactly `len(DEFAULT_VOTE_ENSEMBLE)` (three)
   `ProviderVoteRecorded` events, each stamped `component="scheduler"` and
   `forecast_id` equal to the tick's own `ForecastCreated.forecast_id`. This
   scenario also pins the *default*-config path: an untouched
   `ForecastConfig.vote_ensemble` must stay byte-identical in provenance to
   `DEFAULT_VOTE_ENSEMBLE` (issue #294's cassette-determinism constraint).
3. `test_forecast_stage_drives_the_configured_non_default_vote_ensemble`
   (issue #294) -- a config whose `forecast.vote_ensemble` names a two-member
   ensemble sharing no `model_version` with `DEFAULT_VOTE_ENSEMBLE` must be
   the ensemble actually driven: the ledgered
   `(provider, model_version)` pairs equal the configured members, in order.
4. `test_forecast_stage_empty_configured_vote_ensemble_abstains_fail_closed`
   (issue #294) -- an operator who empties `forecast.vote_ensemble` gets zero
   votes and an abstaining, live-ineligible record, never a silent fallback to
   the built-in default triple. An ensemble configured away must not read as a
   healthy one.

Local-doubles choice
    This module defines its own `_FixtureSearchTransport` /
    `_FixtureFetchTransport` / `_FakeVoteTransport` doubles rather than
    importing `tests/forecast/conftest.py`'s near-identical ones --
    cross-test-package imports are not supported under this project's
    rootdir-relative pytest import mode (see `tests/forecast/conftest.py`'s
    and `tests/forecast/test_triage.py`'s docstrings for the same convention
    applied elsewhere); this module also redefines a narrow `_build_deps`
    helper rather than importing `test_paper_loop.py`'s private one, for the
    same reason.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests.integration.conftest import FIXED_NOW_EPOCH_S, ledger_path_for
from windbreak.config.schema import EnsembleMemberConfig

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig

#: The always-present per-tick stage events regardless of forecast outcome
#: (mirrors `test_paper_loop.py::_ALWAYS_PRESENT_EVENT_TYPES`, narrowed to
#: what this module's tests actually need).
_FORECAST_CREATED = "ForecastCreated"
_PROVIDER_VOTE_RECORDED = "ProviderVoteRecorded"

#: A deliberately non-default two-member vote ensemble (issue #294): it shares
#: no `model_version` with `DEFAULT_VOTE_ENSEMBLE` and is a *different size*, so
#: a run driving the built-in default instead of this configured tuple cannot
#: accidentally match. Two members still clear the pipeline's default
#: `min_ensemble_votes` quorum (2), so the run reaches full aggregation and the
#: assertion is about *which* members voted, never about an abstention artifact.
_CONFIGURED_VOTE_ENSEMBLE: tuple[EnsembleMemberConfig, ...] = (
    EnsembleMemberConfig("anthropic", "claude-operator-pinned-a", "2025-01-31"),
    EnsembleMemberConfig("openai", "gpt-operator-pinned-b", "2024-12-31"),
)


def _fixed_clock() -> int:
    """Return the fixed, non-advancing epoch second this module builds
    `PaperTickDeps` against, for cross-run determinism.
    """
    return FIXED_NOW_EPOCH_S


class _FixtureSearchTransport:
    """Deterministic, network-free stand-in `SearchTransport`.

    Returns exactly one candidate URL per subquestion, on a fixed host, so
    `bounded_web_research` always has a reproducible candidate to fetch --
    mirrors `tests/forecast/conftest.py::FixtureSearchTransport` (redefined
    locally; see the module docstring's "Local-doubles choice" note).
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
    """Deterministic, network-free stand-in `FetchTransport`.

    Returns fixed content derived only from the URL, so every fetch of a URL
    (the citation-building fetch AND the verification refetch) is
    byte-identical -- every gathered citation therefore verifies cleanly.
    """

    def fetch(self, url: str) -> str:
        """Return deterministic canned content for `url`.

        Args:
            url: The URL being fetched.

        Returns:
            A deterministic content string derived from `url`.
        """
        return f"fixture content for {url}"


#: Three schema-valid, non-abstaining #184 vote JSON responses (SPEC S6.3),
#: cycled by `_FakeVoteTransport` in call order -- mirrors
#: `tests/forecast/conftest.py::CANNED_VOTE_RESPONSES` (redefined locally).
_CANNED_VOTE_RESPONSES: tuple[str, str, str] = (
    '{"probability_ppm": 440000, "rationale_summary": "steady evidence alpha", '
    '"abstain": false}',
    '{"probability_ppm": 450000, "rationale_summary": "steady evidence beta", '
    '"abstain": false}',
    '{"probability_ppm": 460000, "rationale_summary": "steady evidence gamma", '
    '"abstain": false}',
)


class _FakeVoteTransport:
    """Deterministic, network-free stand-in `LlmTransport`.

    Returns `_CANNED_VOTE_RESPONSES` in call order, cycling if called more
    than three times -- mirrors `tests/forecast/conftest.py::FakeVoteTransport`
    (redefined locally; see the module docstring's "Local-doubles choice"
    note).
    """

    def __init__(self) -> None:
        """Reset the call counter."""
        self._calls = 0

    def complete(self, request: object) -> str:
        """Return the next canned response, ignoring `request`'s contents.

        Args:
            request: The (unused) `LlmRequest`-shaped call.

        Returns:
            The next canned response, cycling by call index.
        """
        del request
        response = _CANNED_VOTE_RESPONSES[self._calls % len(_CANNED_VOTE_RESPONSES)]
        self._calls += 1
        return response


def _build_deps_with_real_citations(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    tmp_path: Path,
):
    """Build one `PaperTickDeps` wired to reach the vote-collection stage.

    Unlike `tests/integration/conftest.py`'s shared `research_tools_factory`
    (an offline `NullSearchTransport` producing zero citations, deliberately
    abstaining before the vote stage), this builds a research double that
    genuinely gathers and verifies citations, so a subsequent
    `_forecast_stage` call actually drives `collect_model_votes`.

    Args:
        books_dir: The `deep_walk` books-fixture directory.
        cassette_path: The (unused once `transport` is replaced) recorded-
            cassette path `build_paper_deps` requires.
        ledger_path: Where the tick's `SqliteLedgerStore` is created.
        report_dir: Where the weekly-report stub is written.
        config: The PAPER-ceilinged configuration.
        tmp_path: The test's isolated tmp directory, for the research cache.

    Returns:
        A `PaperTickDeps` whose `transport` is a fresh `_FakeVoteTransport`
        and whose `research_tools` gathers real, verifying citations.
    """
    from windbreak.forecast.sandbox import build_research_tools
    from windbreak.scheduler.loop import build_paper_deps

    research_tools = build_research_tools(
        allowed_hosts=frozenset({"research.local"}),
        cache_dir=tmp_path / "research-cache",
        search_transport=_FixtureSearchTransport(),
        fetch_transport=_FixtureFetchTransport(),
    )
    deps = build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
        config=config,
        research_tools=research_tools,
        clock=_fixed_clock,
    )
    return dataclasses.replace(deps, transport=_FakeVoteTransport())


def _config_with_vote_ensemble(
    config: WindbreakConfig, members: tuple[EnsembleMemberConfig, ...]
) -> WindbreakConfig:
    """Return `config` with only its `forecast.vote_ensemble` replaced.

    Every other forecast knob (budget ceilings, `min_verified_citations`, the
    legacy `ensemble`) is carried through untouched, so a scenario built on this
    helper isolates the vote-ensemble seam rather than perturbing the whole
    forecast section.

    Args:
        config: The base PAPER-ceilinged configuration.
        members: The vote-ensemble members to configure (possibly empty).

    Returns:
        A new `WindbreakConfig` whose `forecast.vote_ensemble` is `members`.
    """
    forecast = dataclasses.replace(config.forecast, vote_ensemble=members)
    return dataclasses.replace(config, forecast=forecast)


def _ledgered_member_pairs(records) -> tuple[tuple[str, str], ...]:
    """Extract the driven ensemble members from a tick's ledger, in row order.

    Each `ProviderVoteRecorded` row names the member that produced it, so the
    ordered `(provider, model_version)` pairs are the ledger's own testimony
    about which ensemble the vote stage actually drove -- the observable issue
    #294's wiring is pinned against.

    Args:
        records: The ledger rows read back from the tick's store.

    Returns:
        One `(provider, model_version)` pair per `ProviderVoteRecorded` row.
    """
    payloads = (
        json.loads(record.payload_json)["data"]
        for record in records
        if record.event_type == _PROVIDER_VOTE_RECORDED
    )
    return tuple(
        (payload["provider"], payload["model_version"]) for payload in payloads
    )


def test_zero_verified_citations_path_ledgers_zero_provider_vote_recorded(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """The shared offline fixture's `NullSearchTransport` gathers zero
    citations, so the pipeline abstains with `ABSTENTION_NO_VERIFIED_
    CITATIONS` BEFORE the vote stage: `collect_model_votes` is never called,
    so zero `ProviderVoteRecorded` events are ever appended to the tick's
    ledger, exactly like the pre-existing `FORECAST_OUTPUT_DISCARDED`
    absence this same offline fixture already proves for `test_paper_loop.py`.
    """
    from windbreak.scheduler.loop import build_paper_deps, run_single_tick

    deps = build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools=research_tools_factory(),
        clock=_fixed_clock,
    )

    run_single_tick(deps, beat=1)

    records = deps.store.read_all()
    forecast_record = next(
        record for record in records if record.event_type == _FORECAST_CREATED
    )
    forecast_payload = json.loads(forecast_record.payload_json)["data"]
    assert forecast_payload["abstention_reason"] == "no_verified_citations"
    provider_vote_events = [
        record for record in records if record.event_type == _PROVIDER_VOTE_RECORDED
    ]
    assert provider_vote_events == []


def test_one_paper_tick_ledgers_one_provider_vote_recorded_per_ensemble_member(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    tmp_path: Path,
) -> None:
    """A tick that genuinely reaches the vote-collection stage (real,
    verifying citations plus a real, voting LLM transport double) appends
    exactly `len(DEFAULT_VOTE_ENSEMBLE)` (three) `ProviderVoteRecorded`
    events, each `component="scheduler"` and each carrying the SAME
    `forecast_id` as the tick's own `ForecastCreated` row.
    """
    from windbreak.forecast.providers import DEFAULT_VOTE_ENSEMBLE
    from windbreak.scheduler.loop import _forecast_stage

    deps = _build_deps_with_real_citations(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        tmp_path=tmp_path,
    )
    order_book = deps.exchange.get_order_book(deps.ticker)
    created_at = datetime.fromtimestamp(FIXED_NOW_EPOCH_S, tz=UTC)

    forecast = _forecast_stage(deps, order_book, created_at)

    records = deps.store.read_all()
    forecast_record = next(
        record for record in records if record.event_type == _FORECAST_CREATED
    )
    assert (
        json.loads(forecast_record.payload_json)["data"]["forecast_id"]
        == forecast.forecast_id
    )

    provider_vote_records = [
        record for record in records if record.event_type == _PROVIDER_VOTE_RECORDED
    ]
    assert len(provider_vote_records) == len(DEFAULT_VOTE_ENSEMBLE)
    for record in provider_vote_records:
        assert record.component == "scheduler"
        payload = json.loads(record.payload_json)["data"]
        assert payload["forecast_id"] == forecast.forecast_id

    # ForecastCreated must precede every ProviderVoteRecorded row (issue
    # #281's own ordering directive: "after its ForecastCreated").
    event_types = [record.event_type for record in records]
    forecast_position = event_types.index(_FORECAST_CREATED)
    vote_positions = [
        index
        for index, event_type in enumerate(event_types)
        if event_type == _PROVIDER_VOTE_RECORDED
    ]
    assert vote_positions, "expected at least one ProviderVoteRecorded row"
    assert all(position > forecast_position for position in vote_positions)

    # Issue #294's determinism constraint: an untouched config drives exactly
    # the engine's built-in default provenance, in order -- so wiring the
    # config through the composition root changed no default-path byte.
    assert _ledgered_member_pairs(records) == tuple(
        (member.provider, member.model_version) for member in DEFAULT_VOTE_ENSEMBLE
    )


# --- issue #294: config.forecast.vote_ensemble reaches run_pipeline ---------------


def test_forecast_stage_drives_the_configured_non_default_vote_ensemble(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    tmp_path: Path,
) -> None:
    """A config naming a non-default `forecast.vote_ensemble` is the ensemble
    `_forecast_stage` actually drives: the ledgered `(provider, model_version)`
    pairs equal the configured two members, in configured order, and not the
    engine's built-in three-member default (issue #294, ADR-0006).
    """
    from windbreak.scheduler.loop import _forecast_stage

    deps = _build_deps_with_real_citations(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config_with_vote_ensemble(paper_config, _CONFIGURED_VOTE_ENSEMBLE),
        tmp_path=tmp_path,
    )
    order_book = deps.exchange.get_order_book(deps.ticker)
    created_at = datetime.fromtimestamp(FIXED_NOW_EPOCH_S, tz=UTC)

    _forecast_stage(deps, order_book, created_at)

    assert _ledgered_member_pairs(deps.store.read_all()) == tuple(
        (member.provider, member.model_version) for member in _CONFIGURED_VOTE_ENSEMBLE
    )


def test_forecast_stage_empty_configured_vote_ensemble_abstains_fail_closed(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    tmp_path: Path,
) -> None:
    """An operator who empties `forecast.vote_ensemble` gets no votes at all --
    zero `ProviderVoteRecorded` rows and an abstaining, live-ineligible record
    -- never a silent fallback to the built-in default triple (issue #294). An
    ensemble the operator configured away must not read as a healthy one.

    The reason stamped is the forecast package's zero-survivor classification
    (`ABSTENTION_ALL_VOTES_DISCARDED`; `_vote_shortfall_reason` reserves
    `ABSTENTION_PROVIDER_UNAVAILABLE` for an all-transport-fault wipeout, which
    needs at least one actual discard). What this test pins is the composition
    root's behaviour -- no members driven, no fallback, not live-eligible -- so
    it asserts the constant rather than a hand-copied string.
    """
    from windbreak.forecast import ABSTENTION_ALL_VOTES_DISCARDED
    from windbreak.scheduler.loop import _forecast_stage

    deps = _build_deps_with_real_citations(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config_with_vote_ensemble(paper_config, ()),
        tmp_path=tmp_path,
    )
    order_book = deps.exchange.get_order_book(deps.ticker)
    created_at = datetime.fromtimestamp(FIXED_NOW_EPOCH_S, tz=UTC)

    forecast = _forecast_stage(deps, order_book, created_at)

    assert forecast is not None
    assert forecast.abstention_reason == ABSTENTION_ALL_VOTES_DISCARDED
    assert forecast.eligible_for_live is False
    assert _ledgered_member_pairs(deps.store.read_all()) == ()
