"""What the *shipped entry point* forecasts, driven through `windbreak run` (#510).

Issue #510 states that the shipped CLI cannot reach a non-abstention forecast,
so no intent, approval or fill can cross a real process boundary. Every existing
test of that claim stops one composition short of proving it: they call
:func:`~windbreak.scheduler.loop.build_paper_deps` directly, which accepts
``research_tools``, ``clock`` and (via :func:`dataclasses.replace`) ``transport``
seams that ``windbreak run`` has no flag for. ``tests/scheduler/``'s barrier map
and ``tests/integration/test_paper_real_approval_fill.py`` both reach an intent
*through those seams*. Nothing drove the argument vector an operator actually
types and read the ledger it actually wrote.

That gap is the composition trap in its exact form: the seam below the CLI is
green, and the end state the CLI reaches is a different one. This module closes
it by calling :func:`windbreak.main.main` with the four PAPER flags
``deploy/docker-compose.yml`` and ``deploy/systemd/windbreak-pipeline.service``
name, over a books directory copied from the committed fixture, and asserting
the exact rows the run appends to its own hash-chained ledger.

The books copy is prepared, and what that does and does not mean is the whole
honesty of the exercise. Three *inputs* are moved so the run gets past the
screen at all -- a close time inside the production ``[2, 120]``-day window, a
book resting above the production depth floor, an opening balance above the
production capital floor. **No threshold is moved**, and each is asserted
against its own production default in
:func:`test_the_prepared_books_satisfy_production_thresholds_rather_than_moving_them`
so "prepared, not relaxed" is measured rather than promised. What remains after
that preparation is what #510 is about, and it is not a fixture problem:

* :func:`test_the_shipped_cli_ledgers_an_abstaining_forecast_and_no_intent` is
  #510's reproduction, pinned at the entry point. The screen clears, and the
  forecast still abstains on ``no_verified_citations`` -- before a single vote,
  so the run ledgers **zero** ``ProviderVoteRecorded`` rows -- and the selector
  records ``intent_count == 0``. The process exits 0 and reports healthy while
  doing it, which is the separate finding #485 carries.
* :func:`test_the_cli_composition_can_resolve_no_citation_finding_research`
  names the cause structurally rather than by narrating the run: the
  composition ``windbreak run`` builds resolves the research bundle that finds
  nothing *by construction*, and there is no configuration that changes that in
  cassette mode. A positive control proves the assertion can fail.

Why a better fixture cannot close this
--------------------------------------

The obvious repair -- record a real vote cassette and commit it -- is not
available, and that is a proof rather than an opinion.
:func:`~windbreak.forecast.providers.base.build_vote_prompt` interpolates
``market.close_time.isoformat()`` into the prompt, and
:meth:`~windbreak.forecast.cassettes.LlmRequest.request_hash` digests that
prompt, so a cassette key is a function of the close time; while
:func:`~windbreak.screener.filters.horizon_filter` measures that same close
against the run's clock, so a market that *keeps* clearing the screen must carry
a close that moves with the clock. A committed literal satisfies the cassette
and permanently fails the horizon; a clock-relative close satisfies the horizon
and misses the cassette on every run. That contradiction is pinned by
``tests/scheduler/test_paper_intent_barriers.py::test_static_vote_cassette_and_horizon_filter_are_mutually_exclusive``
and is not restated here.

So #510's fix has to be a *composed* transport selectable from configuration --
its own proposal -- rather than a better committed fixture. This module is the
falsifiable statement of the end state that fix has to change: when a hermetic
research transport becomes selectable from committed configuration, the second
test below stops being true, and it is meant to.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from windbreak.config.schema import (
    DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    CapitalConfig,
    HorizonDays,
    ScreenerConfig,
    WindbreakConfig,
)
from windbreak.forecast.sandbox import build_research_tools
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.main import _resolve_provider_http, main
from windbreak.scheduler.loop import build_paper_deps

#: The repository root, resolved from this file rather than the process's cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed books fixture the two shipped command lines both name.
SHIPPED_BOOKS = REPO_ROOT / "tests" / "fixtures" / "books" / "deep_walk"

#: The committed vote cassette those same two command lines name.
SHIPPED_CASSETTE = REPO_ROOT / "tests" / "fixtures" / "forecast" / "cassettes.json"

#: The shipped fixture's sole market.
TICKER = "MKT-DEEP"

#: Whole days from the run's own clock to the re-dated fixture close. Inside
#: `HorizonDays()`'s production [2, 120]-day window, which is therefore
#: satisfied rather than widened.
IN_WINDOW_HORIZON_DAYS = 30

#: The resting size, in contract-centis, written onto every level of the
#: prepared book -- above `ScreenerConfig().min_depth_contract_centis`.
DEEP_QUANTITY_CENTIS = 100_000

#: The prepared book's resting YES bid, in pips. The shipped fixture's own.
PREPARED_BID_PIPS = 4500

#: The prepared book's resting YES ask, in pips. The shipped fixture's own, and
#: therefore the baseline price the abstaining forecast echoes back.
PREPARED_ASK_PIPS = 4600

#: An opening balance, in micros, above `CapitalConfig().floor_micros`.
FUNDED_CASH_MICROS = 10_000_000_000

#: Pips-to-ppm: a price in pips is a probability in ppm scaled by this.
PPM_PER_PIP = 100

#: The abstention an offline research bundle forces, before any vote is cast.
NO_CITATIONS_ABSTENTION = "no_verified_citations"

#: The one host the positive-control research bundle below is allowlisted for.
CONTROL_RESEARCH_HOST = "research.local"

#: The single URL that bundle's search resolves to.
CONTROL_ARTICLE_URL = f"https://{CONTROL_RESEARCH_HOST}/article"


class _FindingSearchTransport:
    """A search/fetch transport that always finds one article (the control)."""

    def search(self, query: str) -> tuple[str, ...]:
        """Return the single candidate article URL.

        Args:
            query: The (unused) subquestion text.

        Returns:
            A one-element tuple naming :data:`CONTROL_ARTICLE_URL`.
        """
        del query
        return (CONTROL_ARTICLE_URL,)

    def fetch(self, url: str) -> str:
        """Return an empty page for any URL; never reached by these tests.

        Args:
            url: The (unused) URL being fetched.

        Returns:
            The empty string.
        """
        del url
        return ""


def _prepare_books(tmp_path: Path) -> Path:
    """Copy the shipped books fixture and move its three screening *inputs*.

    The close time, the resting depth and the opening cash are the three
    fixture values that keep the shipped venue's only market off the screen.
    Each is moved to a value that genuinely satisfies the production threshold
    it faces; no threshold is read, lowered, or overridden anywhere.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The prepared books directory.
    """
    books = tmp_path / "books"
    shutil.copytree(SHIPPED_BOOKS, books)

    closes_at = datetime.now(UTC) + timedelta(days=IN_WINDOW_HORIZON_DAYS)
    markets_path = books / "markets.json"
    markets = json.loads(markets_path.read_text(encoding="utf-8"))
    markets[0]["close_time"] = closes_at.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    markets_path.write_text(json.dumps(markets, indent=2), encoding="utf-8")

    sessions_path = books / "sessions.json"
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    for steps in sessions.values():
        for step in steps:
            step["book"]["yes_bids"] = [
                {"price": PREPARED_BID_PIPS, "quantity": DEEP_QUANTITY_CENTIS}
            ]
            step["book"]["yes_asks"] = [
                {"price": PREPARED_ASK_PIPS, "quantity": DEEP_QUANTITY_CENTIS}
            ]
    sessions_path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")

    balances_path = books / "balances.json"
    balances = json.loads(balances_path.read_text(encoding="utf-8"))
    balances["total"] = FUNDED_CASH_MICROS
    balances["available"] = FUNDED_CASH_MICROS
    balances_path.write_text(json.dumps(balances, indent=2), encoding="utf-8")

    return books


def _run_shipped_cli(tmp_path: Path, books: Path) -> tuple[int, Path]:
    """Run one beat of the shipped PAPER entry point over ``books``.

    The argument vector is the one ``deploy/docker-compose.yml`` and
    ``deploy/systemd/windbreak-pipeline.service`` invoke, narrowed to a single
    beat: the four PAPER flags plus the process selector. Nothing is injected,
    because there is nothing to inject through -- that is the finding.

    Args:
        tmp_path: pytest's per-test temporary directory.
        books: The prepared books directory.

    Returns:
        The process exit code and the ledger path it wrote.
    """
    ledger_path = tmp_path / "ledger.db"
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    exit_code = main(
        [
            "run",
            "--process",
            "pipeline",
            "--heartbeat-interval",
            "0",
            "--max-beats",
            "1",
            "--ledger-path",
            str(ledger_path),
            "--paper-books-dir",
            str(books),
            "--cassette-path",
            str(SHIPPED_CASSETTE),
            "--report-dir",
            str(report_dir),
        ]
    )
    return exit_code, ledger_path


def _ledgered(ledger_path: Path) -> list[tuple[str, dict[str, object]]]:
    """Read every row of a ledger written by a finished run, chain first.

    Args:
        ledger_path: The hash-chained ledger the run wrote.

    Returns:
        One ``(event_type, data)`` pair per row, in ledger order.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    return [
        (record.event_type, dict(json.loads(record.payload_json)["data"]))
        for record in records
    ]


def _only(rows: list[tuple[str, dict[str, object]]], event: str) -> dict[str, object]:
    """Return the payload of the single row of ``event``, asserting uniqueness.

    Args:
        rows: Every ledgered ``(event_type, data)`` pair.
        event: The event type to select.

    Returns:
        That event's one payload.
    """
    matched = [data for name, data in rows if name == event]
    assert len(matched) == 1
    return matched[0]


def test_the_prepared_books_satisfy_production_thresholds_rather_than_moving_them(
    tmp_path: Path,
) -> None:
    """Every prepared input clears the production default it faces, unmodified.

    The reproduction below is only evidence about the *shipped* configuration
    if its preparation moved inputs and nothing else. Each of the three values
    is therefore compared against the production constant it has to satisfy,
    read from the config classes rather than transcribed, so a future change
    that reached the forecast by lowering a floor fails here instead of quietly
    strengthening the claim next door.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    books = _prepare_books(tmp_path)
    markets = json.loads((books / "markets.json").read_text(encoding="utf-8"))
    sessions = json.loads((books / "sessions.json").read_text(encoding="utf-8"))
    balances = json.loads((books / "balances.json").read_text(encoding="utf-8"))

    closes_at = datetime.strptime(
        markets[0]["close_time"], "%Y-%m-%dT%H:%M:%S.%fZ"
    ).replace(tzinfo=UTC)
    horizon_days = (closes_at - datetime.now(UTC)).days
    window = HorizonDays()

    assert window.min <= horizon_days <= window.max
    assert horizon_days == IN_WINDOW_HORIZON_DAYS - 1
    assert ScreenerConfig().min_depth_contract_centis < DEEP_QUANTITY_CENTIS
    assert CapitalConfig().floor_micros < FUNDED_CASH_MICROS
    resting = [
        level["quantity"]
        for steps in sessions.values()
        for step in steps
        for side in ("yes_bids", "yes_asks")
        for level in step["book"][side]
    ]
    assert resting != []
    assert set(resting) == {DEEP_QUANTITY_CENTIS}
    assert balances["total"] == FUNDED_CASH_MICROS


def test_the_shipped_cli_ledgers_an_abstaining_forecast_and_no_intent(
    tmp_path: Path,
) -> None:
    """`windbreak run` clears the screen, abstains, and emits no intent (#510).

    This is #510's reproduction driven through :func:`windbreak.main.main`
    rather than through :func:`~windbreak.scheduler.loop.build_paper_deps`, so
    it measures the composition an operator actually gets. With the screen
    cleared, the run reaches forecasting and stops there:
    ``offline_research_tools`` finds nothing, so ``run_pipeline`` abstains on
    ``no_verified_citations`` **before any vote is cast** -- asserted as an
    exact count of zero ``ProviderVoteRecorded`` rows, not as an absence of a
    row this test forgot to look for.

    The abstaining forecast's ``probability_ppm`` is the market's own ask
    echoed back, which is asserted as the derived identity rather than as a
    bare literal: a "forecast" that is exactly the price carries no information,
    and that is the whole reason an abstaining loop produces no track record.

    The exit code is pinned too. The run does all of this and reports success,
    which is the separate finding issue #485 carries; recording it here keeps
    "the loop is healthy" and "the loop can never trade" visible as the same
    run.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    books = _prepare_books(tmp_path)

    exit_code, ledger_path = _run_shipped_cli(tmp_path, books)

    assert exit_code == 0
    rows = _ledgered(ledger_path)
    assert rows != []
    assert _only(rows, "ScreenDecisionRecorded") == {
        "blocked_by": [],
        "eligible": True,
        "ticker": TICKER,
    }
    forecast = _only(rows, "ForecastCreated")
    assert forecast["abstention_reason"] == NO_CITATIONS_ABSTENTION
    assert forecast["market_ticker"] == TICKER
    assert forecast["market_price_baseline_pips"] == PREPARED_ASK_PIPS
    assert forecast["probability_ppm"] == PREPARED_ASK_PIPS * PPM_PER_PIP
    assert forecast["eligible_for_live"] is False
    assert [name for name, _ in rows if name == "ProviderVoteRecorded"] == []
    assert _only(rows, "SelectorDecisionRecorded")["intent_count"] == 0


def test_the_cli_composition_can_resolve_no_citation_finding_research(
    tmp_path: Path,
) -> None:
    """The composition `windbreak run` builds finds nothing, by construction.

    The narration in the reproduction above says the run abstained because the
    research bundle found nothing; this asserts it against the bundle itself,
    resolved through the very call ``_build_paper_on_beat`` makes -- default
    :class:`~windbreak.config.schema.WindbreakConfig` and the ``provider_http``
    that same configuration resolves to, which in cassette mode is ``None``.
    Asserting the resolved capability rather than the run's outcome is what
    separates "no citation was found today" from "no citation can be found".

    The positive control is the point of the test, not decoration: a bundle
    built over a transport that *does* find an article returns it through the
    same :meth:`~windbreak.forecast.sandbox.ResearchTools.search` call, so the
    empty tuple above is a measurement rather than a method that cannot return
    anything.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    config = WindbreakConfig()
    deps = build_paper_deps(
        books_dir=_prepare_books(tmp_path),
        cassette_path=SHIPPED_CASSETTE,
        ledger_path=tmp_path / "ledger.db",
        report_dir=tmp_path / "report",
        config=config,
        provider_http=_resolve_provider_http(config),
    )
    try:
        found = deps.research_tools.search("will this market resolve yes")
    finally:
        deps.store.close()

    control = build_research_tools(
        allowed_hosts=frozenset({CONTROL_RESEARCH_HOST}),
        cache_dir=tmp_path / "control-cache",
        search_transport=_FindingSearchTransport(),
        fetch_transport=_FindingSearchTransport(),
        max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    )

    assert found == ()
    assert control.search("will this market resolve yes") == (CONTROL_ARTICLE_URL,)
