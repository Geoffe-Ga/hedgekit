"""What actually stands between the shipped PAPER loop and one order intent (#438).

Issue #438 states that the shipped cassette-mode PAPER loop abstains on
``no_verified_citations`` before any vote, forever, because
:func:`~windbreak.scheduler.loop.build_paper_deps` resolves research to
:func:`~windbreak.scheduler.provider_wiring.offline_research_tools`, whose
transports find nothing by construction. That is true, and it is **neither the
first thing that stops the loop nor the last**. This module drives the real
:func:`~windbreak.scheduler.loop.run_single_tick` over the real shipped
composition and pins the whole chain, in the order one tick meets it, with the
exact values each stage records.

Six barriers stand between ``docker compose up`` and one order intent:

1. **The screen.** The shipped ``deep_walk`` fixture rests 300 contract-centis
   against a 10 000 floor and closes on a frozen 2025 literal that no longer
   sits in the ``[2, 120]``-day window, so the venue's only market is refused
   before any research money is spent. Forecasting is never entered -- which is
   why #438's abstention is not the operative cause on the shipped
   configuration at all (:func:`test_shipped_configuration_never_forecasts`).
2. **Research** -- #438's own barrier, reached only once the screen is cleared
   (:func:`test_clearing_the_screen_reaches_the_citation_abstention`).
3. **The correlation declaration.** The selector refuses an unbucketable market
   outright, ahead of any edge arithmetic, and the shipped stack ships no
   configuration declaring a bucket
   (:func:`test_unbucketed_market_refuses_before_any_edge_arithmetic`).
4. **The vote cassette.** The committed ``cassettes.json`` is keyed on
   human-readable placeholders, so the first real vote raises
   :class:`~windbreak.forecast.cassettes.CassetteMissError` out of the tick
   (:func:`test_shipped_vote_cassette_cannot_serve_the_pipelines_own_prompt`)
   -- and no *committed* cassette can ever serve a market that passes barrier
   1, because the two requirements contradict each other
   (:func:`test_static_vote_cassette_and_horizon_filter_are_mutually_exclusive`).
5. **The provider track record.** Absent the M6 artifact every vote provider is
   unproven, so no forecast is live-eligible.
6. **The research charge at the entry probe.** This one is decisive and
   unconditional. Every full-pipeline forecast books a flat
   :data:`~windbreak.forecast.budget.FULL_PIPELINE_RESEARCH_COST_MICROS`
   (3 000 000 micros, $3.00), and :func:`windbreak.selector.select` gates its
   twelve entry conditions on a fixed **1.00-contract** probe fill, amortizing
   that whole charge over that single contract. A one-contract gross edge
   cannot exceed 1 000 000 ppm ($1.00), so ``net_edge_min`` is unreachable for
   *any* market, at *any* price, with *any* capital
   (:func:`test_flat_research_charge_makes_net_edge_unreachable`).

Barrier 6 is why this module pins an impossibility rather than an emitted
intent on the shipped composition: no fixture, cassette, or configuration
change can clear it.
:func:`test_one_intent_is_emitted_once_the_research_charge_is_amortizable`
keeps that claim falsifiable by isolating the single term -- with it reduced
and nothing else changed, the real tick emits exactly one intent and the ledger
carries every stage. The parametrized negatives in
:func:`test_restoring_any_single_barrier_removes_the_intent` then put each
condition back one at a time and watch the intent disappear, so every barrier
is shown independently load-bearing rather than jointly asserted.

Nothing here relaxes a threshold. ``min_depth_contract_centis``,
``horizon_days``, ``min_net_edge_ppm`` and the price bands are read at their
production defaults throughout; where a barrier is cleared it is cleared by
handing the loop an input that genuinely satisfies the gate.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.config.schema import (
    CorrelationConfig,
    CorrelationTagConfig,
    WindbreakConfig,
)
from windbreak.connector.paper import PaperExchange
from windbreak.forecast import pipeline as forecast_pipeline
from windbreak.forecast.budget import FULL_PIPELINE_RESEARCH_COST_MICROS
from windbreak.forecast.cassettes import CassetteMissError, LlmRequest
from windbreak.forecast.providers.base import build_vote_prompt
from windbreak.forecast.records import BaselineQuoteSnapshot
from windbreak.forecast.sandbox import build_research_tools
from windbreak.scheduler.loop import build_paper_deps, run_single_tick

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.forecast.sandbox import ResearchTools
    from windbreak.scheduler.loop import PaperTickDeps

#: The repository root, resolved from this file rather than the process's cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed books fixture `deploy/docker-compose.yml` and
#: `deploy/systemd/windbreak-pipeline.service` both name on their command line.
SHIPPED_BOOKS = REPO_ROOT / "tests" / "fixtures" / "books" / "deep_walk"

#: The committed vote cassette those same two shipped command lines name.
SHIPPED_CASSETTE = REPO_ROOT / "tests" / "fixtures" / "forecast" / "cassettes.json"

#: The shipped fixture's sole market.
TICKER = "MKT-DEEP"

#: The single host the research doubles below are allowlisted for.
RESEARCH_HOST = "research.local"

#: The one article URL every search double below resolves to.
ARTICLE_URL = f"https://{RESEARCH_HOST}/article"

#: A clean article carrying a genuine JSON-LD `datePublished` well before any
#: instant these tests verify against, so `verify_citations` really verifies it
#: rather than failing it for an unreadable or future publication date.
ARTICLE_PAGE = (
    '<html><head><script type="application/ld+json">'
    '{"@context": "https://schema.org", "@type": "NewsArticle", '
    '"datePublished": "2024-11-15T08:00:00Z"}</script></head>'
    "<body><p>Independent reporting on the underlying question, with no "
    "unusual claims or disruptions noted by observers this period.</p></body>"
    "</html>"
)

#: A near-certain vote body. Deliberately extreme: barrier 6 is a claim about
#: the *best case*, so the vote is the most favourable one the SPEC S6.3 vote
#: schema admits rather than a plausible one.
NEAR_CERTAIN_VOTE = json.dumps(
    {
        "probability_ppm": 990000,
        "rationale_summary": "Evidence supports a near-certain resolution.",
        "abstain": False,
    }
)

#: The near-certain vote's probability, in ppm, as the ledger reports it.
NEAR_CERTAIN_PROBABILITY_PPM = 990000

#: A fixed epoch every clock-injecting test below reads, so nothing in this
#: module depends on the host's wall clock or timezone (2027-01-15T08:00:00Z).
FIXED_NOW_EPOCH_S = 1_800_000_000

#: Whole days from the injected clock to the re-dated fixture close: inside
#: `HorizonDays()`'s production [2, 120]-day window, which is therefore
#: satisfied rather than widened.
IN_WINDOW_HORIZON_DAYS = 30

#: The ask price, in pips, the barrier-6 tests quote: the lowest price inside
#: `price_within_bands`' [500, 9500] open band, and therefore the price that
#: maximizes the gross edge an in-band market can offer.
BEST_CASE_ASK_PIPS = 600

#: The best-case bid, one tick below the best-case ask.
BEST_CASE_BID_PIPS = 590

#: A resting size, in contract-centis, far above `min_depth_contract_centis`.
DEEP_QUANTITY_CENTIS = 100_000

#: The shipped fixture's own resting size, far below that floor.
THIN_QUANTITY_CENTIS = 300

#: An opening balance, in micros, well above `CapitalConfig().floor_micros`.
FUNDED_CASH_MICROS = 10_000_000_000

#: The research charge, in micros, the positive control substitutes for the
#: shipped flat charge: small enough that its per-probe-contract amortization
#: leaves the near-certain edge above `min_net_edge_ppm`.
AMORTIZABLE_RESEARCH_COST_MICROS = 3_000

#: The name of the module-private constant the positive control substitutes.
#: `monkeypatch.setattr` raises when it is absent, so a rename fails this suite
#: loudly instead of silently leaving the substitution unapplied.
RESEARCH_COST_ATTR = "_RESEARCH_COST_MICROS"

#: The twelve SPEC S9.3 entry conditions, in the order the selector renders
#: them, as they read when every one passes.
ALL_ENTRY_CONDITIONS_PASSING = [
    "pass:net_edge_min",
    "pass:annualized_hurdle",
    "pass:ci_straddles_executable_price",
    "pass:quote_snapshot_fresh",
    "pass:forecast_fresh",
    "pass:fee_model_current",
    "pass:market_coherent",
    "pass:citation_support",
    "pass:jurisdiction_eligible",
    "pass:category_eligible",
    "pass:price_within_bands",
    "pass:forecast_live_eligible",
]


class _FindingResearchTransport:
    """A search/fetch double that always finds one clean, verifiable article."""

    def search(self, query: str) -> tuple[str, ...]:
        """Return the single candidate article URL.

        Args:
            query: The (unused) subquestion text.

        Returns:
            A one-element tuple naming :data:`ARTICLE_URL`.
        """
        del query
        return (ARTICLE_URL,)

    def fetch(self, url: str) -> str:
        """Return the clean article page for any URL.

        Args:
            url: The (unused) URL being fetched.

        Returns:
            :data:`ARTICLE_PAGE`.
        """
        del url
        return ARTICLE_PAGE


class _NearCertainVoteTransport:
    """An ``LlmTransport`` answering every prompt with the near-certain vote."""

    def complete(self, request: LlmRequest) -> str:
        """Return :data:`NEAR_CERTAIN_VOTE` for any request.

        Args:
            request: The (unused) completion request.

        Returns:
            The canned vote JSON.
        """
        del request
        return NEAR_CERTAIN_VOTE


def _finding_research_tools(cache_dir: Path) -> ResearchTools:
    """Build sandboxed research tools that gather one verifiable citation.

    Args:
        cache_dir: The root the sandbox jails its fetch cache to.

    Returns:
        A capability-closed bundle over :class:`_FindingResearchTransport`.
    """
    transport = _FindingResearchTransport()
    return build_research_tools(
        allowed_hosts=frozenset({RESEARCH_HOST}),
        cache_dir=cache_dir,
        search_transport=transport,
        fetch_transport=transport,
    )


def _fixed_now() -> datetime:
    """Return the module's fixed instant as a timezone-aware UTC datetime.

    Returns:
        :data:`FIXED_NOW_EPOCH_S` in UTC.
    """
    return datetime.fromtimestamp(FIXED_NOW_EPOCH_S, UTC)


def _fixed_clock() -> int:
    """Return the module's fixed epoch reading.

    Returns:
        :data:`FIXED_NOW_EPOCH_S`.
    """
    return FIXED_NOW_EPOCH_S


def _set_close_time(books: Path, *, closes_at: datetime) -> None:
    """Re-date the fixture market's close time.

    Args:
        books: The books directory to rewrite.
        closes_at: The instant trading closes.
    """
    path = books / "markets.json"
    markets = json.loads(path.read_text(encoding="utf-8"))
    markets[0]["close_time"] = closes_at.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    path.write_text(json.dumps(markets, indent=2), encoding="utf-8")


def _set_book(books: Path, *, bid_pips: int, ask_pips: int, quantity: int) -> None:
    """Replace every recorded book with one resting level per side.

    Args:
        books: The books directory to rewrite.
        bid_pips: The resting YES bid price, in pips.
        ask_pips: The resting YES ask price, in pips.
        quantity: The resting quantity on each side, in contract-centis.
    """
    path = books / "sessions.json"
    sessions = json.loads(path.read_text(encoding="utf-8"))
    for steps in sessions.values():
        for step in steps:
            step["book"]["yes_bids"] = [{"price": bid_pips, "quantity": quantity}]
            step["book"]["yes_asks"] = [{"price": ask_pips, "quantity": quantity}]
    path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def _set_cash(books: Path, *, micros: int) -> None:
    """Set the fixture account's opening balance.

    Args:
        books: The books directory to rewrite.
        micros: The opening total and available balance, in micros.
    """
    path = books / "balances.json"
    balances = json.loads(path.read_text(encoding="utf-8"))
    balances["total"] = micros
    balances["available"] = micros
    path.write_text(json.dumps(balances, indent=2), encoding="utf-8")


#: The M6 artifact's filename beside the loop's other evaluation artifacts.
TRACK_RECORD_FILENAME = "provider-track-records.json"

#: A resolved-forecast count comfortably above `provider_gate.min_resolved`.
PROVEN_RESOLVED_COUNT = 200

#: A resolved-forecast count below that bar, and above zero, so it is the bar
#: itself -- not the record's absence -- that leaves the provider unproven.
UNDER_RESOLVED_COUNT = 1

#: A Brier skill comfortably above `provider_gate.min_brier_skill_ppm`.
PROVEN_BRIER_SKILL_PPM = 50000

#: A Brier skill below that bar, for the same reason as
#: :data:`UNDER_RESOLVED_COUNT`.
UNDER_SKILL_BRIER_PPM = 1


def _write_track_records(
    report_dir: Path,
    *,
    resolved_count: int = PROVEN_RESOLVED_COUNT,
    brier_skill_ppm: int = PROVEN_BRIER_SKILL_PPM,
) -> None:
    """Write an M6 artifact for both shipped vote providers.

    Args:
        report_dir: The evaluation-artifact directory the gate reads.
        resolved_count: The resolved-forecast count recorded for each provider.
        brier_skill_ppm: The Brier skill recorded for each provider, in ppm.
    """
    entry = {"resolved_count": resolved_count, "brier_skill_ppm": brier_skill_ppm}
    (report_dir / TRACK_RECORD_FILENAME).write_text(
        json.dumps({"openai": dict(entry), "anthropic": dict(entry)}),
        encoding="utf-8",
    )


def _bucketed_config() -> WindbreakConfig:
    """Return the shipped config plus one declared correlation bucket.

    Returns:
        A configuration identical to the shipped defaults except that
        :data:`TICKER` carries an operator-declared bucket, which is what
        :func:`windbreak.scheduler.exposure.project_exposure` requires before
        it will project any exposure at all.
    """
    return WindbreakConfig(
        correlation=CorrelationConfig(
            tags=(
                CorrelationTagConfig(
                    ticker=TICKER,
                    bucket_ids=("fed-policy",),
                    tagged_at="2025-01-01T00:00:00+00:00",
                ),
            )
        )
    )


def _rows(deps: PaperTickDeps) -> list[tuple[str, dict[str, object]]]:
    """Return every ledgered row as an ``(event_type, data)`` pair.

    Args:
        deps: The wired tick dependencies whose store is read.

    Returns:
        The rows in sequence order.
    """
    return [
        (record.event_type, json.loads(record.payload_json)["data"])
        for record in deps.store.read_all()
    ]


def _only(rows: list[tuple[str, dict[str, object]]], event: str) -> dict[str, object]:
    """Return the payload of the single row of ``event``.

    Args:
        rows: The ledgered rows.
        event: The event type wanted.

    Returns:
        That row's ``data`` payload.
    """
    matches = [data for event_type, data in rows if event_type == event]
    assert len(matches) == 1, f"expected exactly one {event}, got {len(matches)}"
    return matches[0]


def _reasons(rows: list[tuple[str, dict[str, object]]]) -> list[str]:
    """Return the selector decision's rendered reasons.

    Args:
        rows: The ledgered rows.

    Returns:
        The reason strings, in the order the selector rendered them.
    """
    raw = _only(rows, "SelectorDecisionRecorded")["reasons"]
    assert isinstance(raw, list)
    return [str(reason) for reason in raw]


def _tradeable_books(books: Path) -> None:
    """Make the shipped fixture satisfy every screen and capital precondition.

    Args:
        books: The books directory to rewrite in place.
    """
    _set_close_time(
        books, closes_at=_fixed_now() + timedelta(days=IN_WINDOW_HORIZON_DAYS)
    )
    _set_book(
        books,
        bid_pips=BEST_CASE_BID_PIPS,
        ask_pips=BEST_CASE_ASK_PIPS,
        quantity=DEEP_QUANTITY_CENTIS,
    )
    _set_cash(books, micros=FUNDED_CASH_MICROS)


def _build_deps(
    *,
    books: Path,
    tmp_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    research: ResearchTools | None = None,
    votes: object | None = None,
) -> PaperTickDeps:
    """Wire one PAPER tick over the shipped composition.

    Args:
        books: The books directory the exchange replays.
        tmp_path: The root the ledger and research cache live under.
        report_dir: The evaluation-artifact directory.
        config: The active configuration.
        research: Injected research tools, or ``None`` for the shipped offline
            default.
        votes: An injected vote transport, or ``None`` to keep the shipped
            replay cassette.

    Returns:
        The wired dependencies.
    """
    deps = build_paper_deps(
        books_dir=books,
        cassette_path=SHIPPED_CASSETTE,
        ledger_path=tmp_path / "ledger.db",
        report_dir=report_dir,
        config=config,
        research_tools=research,
        clock=_fixed_clock,
    )
    if votes is None:
        return deps
    return dataclasses.replace(deps, transport=votes)


@pytest.fixture
def shipped_books(tmp_path: Path) -> Path:
    """Provide a private copy of the shipped books fixture.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The copied books directory.
    """
    books = tmp_path / "books"
    shutil.copytree(SHIPPED_BOOKS, books)
    return books


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    """Provide an empty evaluation-artifact directory.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The created directory.
    """
    reports = tmp_path / "reports"
    reports.mkdir()
    return reports


def test_shipped_configuration_never_forecasts(
    tmp_path: Path, report_dir: Path
) -> None:
    """The shipped stack screens its only market out before any research.

    Drives the composition ``deploy/docker-compose.yml`` and
    ``deploy/systemd/windbreak-pipeline.service`` both invoke -- default
    ``WindbreakConfig()``, the committed ``deep_walk`` books, the committed
    vote cassette, the real wall clock -- and pins that the tick never reaches
    forecasting at all. This is the correction to #438: the abstention it
    describes is real, but it sits behind a barrier the shipped configuration
    never gets past.

    The wall clock is deliberately *not* injected here, because the shipped
    deployment does not inject one. The horizon verdict is nonetheless stable:
    the fixture's frozen 2025 close recedes further into the past with every
    passing day, so it can never re-enter the ``[2, 120]``-day window.
    """
    deps = build_paper_deps(
        books_dir=SHIPPED_BOOKS,
        cassette_path=SHIPPED_CASSETTE,
        ledger_path=tmp_path / "ledger.db",
        report_dir=report_dir,
        config=WindbreakConfig(),
    )

    outcome = run_single_tick(deps, beat=1)

    rows = _rows(deps)
    assert outcome.candidate_tickers == ()
    assert outcome.forecast_ids == ()
    assert outcome.intent_count == 0
    screen = _only(rows, "ScreenDecisionRecorded")
    assert screen["ticker"] == TICKER
    assert screen["eligible"] is False
    assert screen["blocked_by"] == ["min_depth_contract_centis", "horizon_days"]
    assert [event for event, _ in rows if event == "ForecastCreated"] == []
    assert [event for event, _ in rows if event == "SelectorDecisionRecorded"] == []


def test_clearing_the_screen_reaches_the_citation_abstention(
    shipped_books: Path, tmp_path: Path, report_dir: Path
) -> None:
    """Behind the screen sits #438's barrier: offline research finds nothing.

    Only the two inputs the screen measured are changed -- the resting depth
    and the close time -- and both are changed to values that genuinely satisfy
    the production thresholds. The thresholds themselves stay at their shipped
    defaults, so what this pins is the *next* refusal, never a relaxed one.

    The abstained forecast's probability equals the market's own baseline ask
    (4 600 pips = 460 000 ppm), which is the observable that separates "the
    pipeline abstained" from "the pipeline forecast and happened to agree".
    """
    _set_close_time(
        shipped_books, closes_at=_fixed_now() + timedelta(days=IN_WINDOW_HORIZON_DAYS)
    )
    _set_book(
        shipped_books, bid_pips=4500, ask_pips=4600, quantity=DEEP_QUANTITY_CENTIS
    )
    deps = _build_deps(
        books=shipped_books,
        tmp_path=tmp_path,
        report_dir=report_dir,
        config=_bucketed_config(),
    )

    outcome = run_single_tick(deps, beat=1)

    rows = _rows(deps)
    screen = _only(rows, "ScreenDecisionRecorded")
    assert screen["eligible"] is True
    assert screen["blocked_by"] == []
    forecast = _only(rows, "ForecastCreated")
    assert forecast["abstention_reason"] == "no_verified_citations"
    assert forecast["eligible_for_live"] is False
    assert forecast["probability_ppm"] == 460000
    assert forecast["market_price_baseline_pips"] == 4600
    assert outcome.intent_count == 0


def test_unbucketed_market_refuses_before_any_edge_arithmetic(
    shipped_books: Path, tmp_path: Path, report_dir: Path
) -> None:
    """An undeclared correlation bucket refuses the market ahead of the edge.

    The shipped stack ships no ``--config``, so ``CorrelationConfig()`` is
    empty and every market is unbucketable. The refusal is the *entire* reason
    set, which is what proves it happens before any entry condition is
    evaluated: a tick that reached the twelve conditions would carry twelve
    reasons here instead of one.
    """
    _set_close_time(
        shipped_books, closes_at=_fixed_now() + timedelta(days=IN_WINDOW_HORIZON_DAYS)
    )
    _set_book(
        shipped_books, bid_pips=4500, ask_pips=4600, quantity=DEEP_QUANTITY_CENTIS
    )
    deps = _build_deps(
        books=shipped_books,
        tmp_path=tmp_path,
        report_dir=report_dir,
        config=WindbreakConfig(),
    )

    run_single_tick(deps, beat=1)

    rows = _rows(deps)
    assert _only(rows, "SelectorDecisionRecorded")["intent_count"] == 0
    assert _reasons(rows) == [
        f"unprovable_exposure: no correlation bucket or holding evidence for {TICKER}"
    ]


def test_shipped_vote_cassette_cannot_serve_the_pipelines_own_prompt(
    shipped_books: Path, tmp_path: Path, report_dir: Path
) -> None:
    """With citations in hand the committed cassette misses and the tick raises.

    ``tests/fixtures/forecast/cassettes.json`` is keyed on human-readable
    placeholders (``placeholder-hash-vote-1``), never on a real
    :meth:`~windbreak.forecast.cassettes.LlmRequest.request_hash`, so the first
    genuine vote fails closed. The failure is a raise out of
    :func:`~windbreak.scheduler.loop.run_single_tick`, not an abstention --
    which is the honest posture, and worth pinning so a later change cannot
    quietly downgrade it into a silently skipped vote.
    """
    _set_close_time(
        shipped_books, closes_at=_fixed_now() + timedelta(days=IN_WINDOW_HORIZON_DAYS)
    )
    _set_book(
        shipped_books, bid_pips=4500, ask_pips=4600, quantity=DEEP_QUANTITY_CENTIS
    )
    deps = _build_deps(
        books=shipped_books,
        tmp_path=tmp_path,
        report_dir=report_dir,
        config=_bucketed_config(),
        research=_finding_research_tools(tmp_path / "cache"),
    )

    with pytest.raises(CassetteMissError) as caught:
        run_single_tick(deps, beat=1)

    assert type(caught.value) is CassetteMissError
    prefix = "no recorded response for request "
    message = str(caught.value)
    assert message.startswith(prefix)
    digest = message[len(prefix) :]
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_static_vote_cassette_and_horizon_filter_are_mutually_exclusive() -> None:
    """No committed cassette can serve a market that clears the horizon filter.

    :func:`~windbreak.forecast.providers.base.build_vote_prompt` interpolates
    ``market.close_time.isoformat()`` into the prompt, and
    :meth:`~windbreak.forecast.cassettes.LlmRequest.request_hash` digests that
    prompt -- so the replay key is a function of the close time.
    :func:`~windbreak.screener.filters.horizon_filter` measures the close time
    against the run's clock, so a market that keeps clearing it must carry a
    close that moves with the clock, and a cassette key that moves with the
    clock cannot be committed. The two requirements contradict.

    Derived, not restated: both prompts come from the real builder and both
    keys from the real hash, and the differing line is *located* rather than
    assumed, so a change to either surface stays covered.
    """
    market = PaperExchange.from_fixture_dir(SHIPPED_BOOKS).get_market(TICKER)
    later = dataclasses.replace(
        market, close_time=market.close_time + timedelta(days=1)
    )
    baseline = BaselineQuoteSnapshot(
        snapshot_id="snapshot-1",
        price_pips=BEST_CASE_ASK_PIPS,
        fetched_at=_fixed_now(),
    )

    prompt = build_vote_prompt(market, baseline, 0)
    later_prompt = build_vote_prompt(later, baseline, 0)

    differing = [
        (left, right)
        for left, right in zip(
            prompt.splitlines(), later_prompt.splitlines(), strict=True
        )
        if left != right
    ]
    assert differing != []
    assert differing == [
        (
            f"Market closes at: {market.close_time.isoformat()}",
            f"Market closes at: {later.close_time.isoformat()}",
        )
    ]
    key = LlmRequest(provider="openai", model_version="m", prompt=prompt)
    later_key = LlmRequest(provider="openai", model_version="m", prompt=later_prompt)
    assert key.request_hash() != later_key.request_hash()


def test_flat_research_charge_makes_net_edge_unreachable(
    shipped_books: Path, tmp_path: Path, report_dir: Path
) -> None:
    """The flat $3.00 research charge alone puts ``net_edge_min`` out of reach.

    Every barrier above is cleared -- deep two-sided book, in-window close,
    verifiable citations, near-certain votes, a declared bucket, proven
    providers, capital far above the floor -- and the market is quoted at the
    lowest price its open band admits, which is the price that maximizes the
    gross edge an in-band fill can offer. The tick still emits no intent,
    because the entry conditions amortize the whole flat charge over a fixed
    1.00-contract probe.

    The bound is arithmetic rather than empirical: a one-contract gross edge
    cannot exceed 1 000 000 ppm, and the charge is worth strictly more than
    that per probe contract. Both the measured refusal and the bound that
    explains it are asserted, so this test states *why* no other market would
    do better rather than merely reporting that this one did not.
    """
    _tradeable_books(shipped_books)
    _write_track_records(report_dir)
    deps = _build_deps(
        books=shipped_books,
        tmp_path=tmp_path,
        report_dir=report_dir,
        config=_bucketed_config(),
        research=_finding_research_tools(tmp_path / "cache"),
        votes=_NearCertainVoteTransport(),
    )

    outcome = run_single_tick(deps, beat=1)

    rows = _rows(deps)
    forecast = _only(rows, "ForecastCreated")
    assert forecast["abstention_reason"] is None
    assert forecast["probability_ppm"] == NEAR_CERTAIN_PROBABILITY_PPM
    assert forecast["eligible_for_live"] is True
    assert forecast["research_cost_micros"] == FULL_PIPELINE_RESEARCH_COST_MICROS
    assert outcome.intent_count == 0
    assert _only(rows, "SelectorDecisionRecorded")["intent_count"] == 0
    reasons = _reasons(rows)
    assert [reason for reason in reasons if reason.startswith("fail:")] == [
        "fail:net_edge_min: net_edge_ppm=-2080000 min_net_edge_ppm=30000",
        "fail:annualized_hurdle: annualized_ppm=-421777778 hurdle_ppm=240000",
    ]
    assert FULL_PIPELINE_RESEARCH_COST_MICROS > 1_000_000


def test_one_intent_is_emitted_once_the_research_charge_is_amortizable(
    shipped_books: Path,
    tmp_path: Path,
    report_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With that one term reduced and nothing else changed, one intent appears.

    The substitution isolates the single quantity barrier 6 names: the flat
    per-forecast research charge the entry probe amortizes. Nothing else about
    the tick changes, so an intent appearing here proves the other five
    barriers are genuinely cleared above, and that this term alone accounts for
    the remaining refusal -- the positive control the impossibility claim needs
    in order to be falsifiable rather than merely asserted.

    It asserts nothing about what the charge *should* be. Choosing that is a
    risk decision inside ``windbreak/forecast`` and ``windbreak/selector``,
    neither of which this change touches.
    """
    monkeypatch.setattr(
        forecast_pipeline, RESEARCH_COST_ATTR, AMORTIZABLE_RESEARCH_COST_MICROS
    )
    _tradeable_books(shipped_books)
    _write_track_records(report_dir)
    deps = _build_deps(
        books=shipped_books,
        tmp_path=tmp_path,
        report_dir=report_dir,
        config=_bucketed_config(),
        research=_finding_research_tools(tmp_path / "cache"),
        votes=_NearCertainVoteTransport(),
    )

    outcome = run_single_tick(deps, beat=1)

    rows = _rows(deps)
    assert outcome.intent_count == 1
    assert outcome.candidate_tickers == (TICKER,)
    entered = [event for event, _ in rows]
    assert entered.count("ScreenDecisionRecorded") == 1
    assert entered.count("MarketSnapshotRecorded") == 1
    assert entered.count("ForecastCreated") == 1
    assert entered.count("ProviderVoteRecorded") == 3
    assert entered.count("SelectorDecisionRecorded") == 1
    assert _only(rows, "SelectorDecisionRecorded")["intent_count"] == 1
    reasons = _reasons(rows)
    assert reasons[:12] == ALL_ENTRY_CONDITIONS_PASSING
    assert [reason for reason in reasons if reason.startswith("fail:")] == []


def _restore_shipped_research_charge(
    monkeypatch: pytest.MonkeyPatch, books: Path, report_dir: Path
) -> None:
    """Restore the shipped flat research charge (barrier 6).

    Args:
        monkeypatch: The active patcher.
        books: The (unused) books directory.
        report_dir: The (unused) artifact directory.
    """
    del books, report_dir
    monkeypatch.setattr(
        forecast_pipeline, RESEARCH_COST_ATTR, FULL_PIPELINE_RESEARCH_COST_MICROS
    )


def _restore_thin_book(
    monkeypatch: pytest.MonkeyPatch, books: Path, report_dir: Path
) -> None:
    """Restore a book below the shipped depth floor (barrier 1a).

    Args:
        monkeypatch: The (unused) active patcher.
        books: The books directory to thin.
        report_dir: The (unused) artifact directory.
    """
    del monkeypatch, report_dir
    _set_book(
        books,
        bid_pips=BEST_CASE_BID_PIPS,
        ask_pips=BEST_CASE_ASK_PIPS,
        quantity=THIN_QUANTITY_CENTIS,
    )


def _restore_frozen_close(
    monkeypatch: pytest.MonkeyPatch, books: Path, report_dir: Path
) -> None:
    """Restore a close time outside the shipped horizon window (barrier 1b).

    Args:
        monkeypatch: The (unused) active patcher.
        books: The books directory to re-date.
        report_dir: The (unused) artifact directory.
    """
    del monkeypatch, report_dir
    _set_close_time(books, closes_at=_fixed_now() - timedelta(days=400))


def _no_file_restoration(
    monkeypatch: pytest.MonkeyPatch, books: Path, report_dir: Path
) -> None:
    """Change nothing; the case restores its barrier through the flags instead.

    Args:
        monkeypatch: The (unused) active patcher.
        books: The (unused) books directory.
        report_dir: The (unused) artifact directory.
    """
    del monkeypatch, books, report_dir


def _restore_missing_track_record(
    monkeypatch: pytest.MonkeyPatch, books: Path, report_dir: Path
) -> None:
    """Remove the M6 artifact so every provider is unproven again (barrier 5).

    Args:
        monkeypatch: The (unused) active patcher.
        books: The (unused) books directory.
        report_dir: The artifact directory to empty.
    """
    del monkeypatch, books
    (report_dir / TRACK_RECORD_FILENAME).unlink()


def _restore_under_resolved_track_record(
    monkeypatch: pytest.MonkeyPatch, books: Path, report_dir: Path
) -> None:
    """Rewrite the M6 artifact below ``min_resolved`` (barrier 5).

    Distinct from :func:`_restore_missing_track_record` on purpose: a provider
    with *no* record is unproven whatever the bar is, so deleting the artifact
    leaves ``min_resolved`` itself untested. Recording a real but insufficient
    count is what makes that threshold the thing under test.

    Args:
        monkeypatch: The (unused) active patcher.
        books: The (unused) books directory.
        report_dir: The artifact directory to rewrite.
    """
    del monkeypatch, books
    _write_track_records(report_dir, resolved_count=UNDER_RESOLVED_COUNT)


def _restore_under_skill_track_record(
    monkeypatch: pytest.MonkeyPatch, books: Path, report_dir: Path
) -> None:
    """Rewrite the M6 artifact below ``min_brier_skill_ppm`` (barrier 5).

    The skill half of the same argument made in
    :func:`_restore_under_resolved_track_record`.

    Args:
        monkeypatch: The (unused) active patcher.
        books: The (unused) books directory.
        report_dir: The artifact directory to rewrite.
    """
    del monkeypatch, books
    _write_track_records(report_dir, brier_skill_ppm=UNDER_SKILL_BRIER_PPM)


def _assert_screen_blocked(
    rows: list[tuple[str, dict[str, object]]], filter_name: str
) -> None:
    """Assert the screen refused the market on exactly ``filter_name``.

    Args:
        rows: The ledgered rows.
        filter_name: The single filter expected to have blocked.
    """
    screen = _only(rows, "ScreenDecisionRecorded")
    assert screen["eligible"] is False
    assert screen["blocked_by"] == [filter_name]
    assert [event for event, _ in rows if event == "ForecastCreated"] == []


def _assert_abstained(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert the forecast abstained for want of verified citations.

    Args:
        rows: The ledgered rows.
    """
    assert _only(rows, "ForecastCreated")["abstention_reason"] == (
        "no_verified_citations"
    )


def _assert_reason(rows: list[tuple[str, dict[str, object]]], reason: str) -> None:
    """Assert the selector rendered ``reason`` among its decision's reasons.

    Args:
        rows: The ledgered rows.
        reason: The reason string that must appear verbatim.
    """
    assert reason in _reasons(rows)


def _evidence_research_charge(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert the shipped research charge is what refused the intent.

    Args:
        rows: The ledgered rows.
    """
    _assert_reason(
        rows, "fail:net_edge_min: net_edge_ppm=-2080000 min_net_edge_ppm=30000"
    )


def _evidence_thin_book(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert the depth floor is what refused the intent.

    Args:
        rows: The ledgered rows.
    """
    _assert_screen_blocked(rows, "min_depth_contract_centis")


def _evidence_frozen_close(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert the horizon window is what refused the intent.

    Args:
        rows: The ledgered rows.
    """
    _assert_screen_blocked(rows, "horizon_days")


def _evidence_missing_track_record(
    rows: list[tuple[str, dict[str, object]]],
) -> None:
    """Assert the unproven providers are what refused the intent.

    Args:
        rows: The ledgered rows.
    """
    held = _only(rows, "ProviderGateHeld")
    assert held["unproven_providers"] == "anthropic,openai"
    assert _only(rows, "ForecastCreated")["eligible_for_live"] is False
    _assert_reason(rows, "fail:forecast_live_eligible: eligible_for_live=False")


def _evidence_offline_research(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert the offline research default is what refused the intent.

    Args:
        rows: The ledgered rows.
    """
    _assert_abstained(rows)
    _assert_reason(rows, "fail:citation_support: citation_count=0")


def _evidence_undeclared_bucket(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert the missing correlation declaration is what refused the intent.

    Args:
        rows: The ledgered rows.
    """
    assert _reasons(rows) == [
        f"unprovable_exposure: no correlation bucket or holding evidence for {TICKER}"
    ]


@pytest.mark.parametrize(
    ("restore", "bucketed", "researched", "evidence"),
    [
        pytest.param(
            _restore_shipped_research_charge,
            True,
            True,
            _evidence_research_charge,
            id="research_charge",
        ),
        pytest.param(
            _restore_thin_book, True, True, _evidence_thin_book, id="thin_book"
        ),
        pytest.param(
            _restore_frozen_close, True, True, _evidence_frozen_close, id="frozen_close"
        ),
        pytest.param(
            _restore_missing_track_record,
            True,
            True,
            _evidence_missing_track_record,
            id="missing_track_record",
        ),
        pytest.param(
            _restore_under_resolved_track_record,
            True,
            True,
            _evidence_missing_track_record,
            id="under_resolved_track_record",
        ),
        pytest.param(
            _restore_under_skill_track_record,
            True,
            True,
            _evidence_missing_track_record,
            id="under_skill_track_record",
        ),
        pytest.param(
            _no_file_restoration,
            True,
            False,
            _evidence_offline_research,
            id="offline_research",
        ),
        pytest.param(
            _no_file_restoration,
            False,
            True,
            _evidence_undeclared_bucket,
            id="undeclared_bucket",
        ),
    ],
)
def test_restoring_any_single_barrier_removes_the_intent(
    shipped_books: Path,
    tmp_path: Path,
    report_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore: Callable[[pytest.MonkeyPatch, Path, Path], None],
    bucketed: bool,
    researched: bool,
    evidence: Callable[[list[tuple[str, dict[str, object]]]], None],
) -> None:
    """Each barrier alone is enough to take the intent away again.

    Starts from the one configuration that emits an intent, puts back exactly
    one blocking condition, and pins both that the intent disappears *and* that
    the restored condition is what removed it. Asserting each separately is
    what keeps the six barriers independently load-bearing: a single
    all-at-once negative would still pass if five of them had quietly stopped
    mattering, and an ``intent_count == 0`` with no accompanying evidence would
    pass for any reason at all, including a crash upstream.

    Args:
        shipped_books: A private copy of the shipped books fixture.
        tmp_path: pytest's per-test temporary directory.
        report_dir: The evaluation-artifact directory.
        monkeypatch: The active patcher.
        restore: The restoration applied to the otherwise-emitting setup.
        bucketed: Whether the correlation bucket stays declared.
        researched: Whether citation-finding research tools stay injected.
        evidence: The per-case assertion naming why the intent went away.
    """
    monkeypatch.setattr(
        forecast_pipeline, RESEARCH_COST_ATTR, AMORTIZABLE_RESEARCH_COST_MICROS
    )
    _tradeable_books(shipped_books)
    _write_track_records(report_dir)
    restore(monkeypatch, shipped_books, report_dir)
    deps = _build_deps(
        books=shipped_books,
        tmp_path=tmp_path,
        report_dir=report_dir,
        config=_bucketed_config() if bucketed else WindbreakConfig(),
        research=_finding_research_tools(tmp_path / "cache") if researched else None,
        votes=_NearCertainVoteTransport(),
    )

    outcome = run_single_tick(deps, beat=1)

    assert outcome.intent_count == 0
    evidence(_rows(deps))
