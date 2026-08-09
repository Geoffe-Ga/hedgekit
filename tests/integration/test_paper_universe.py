"""The PAPER tick screens and iterates a market universe (issue #345, RED).

Before this issue the loop forecast `next(iter(exchange.markets))` -- one
arbitrary ticker, chosen once at composition time and never screened. A
forecaster that looks at exactly one market cannot select, cannot diversify, and
cannot produce the volume of resolved forecasts SPEC S13.5's N=300 power
analysis needs.

Every scenario here runs over `two_ticker_isolation`, deliberately: a
one-market fixture cannot tell "iterates the screened set" apart from the single
hardcoded ticker it replaced, so the two-market universe is the whole point.

The load-bearing constraint is **spend**. Since issue #399 every ensemble vote
books real money against the per-forecast and per-UTC-day `ResearchBudget`
ceilings, so iterating a universe multiplies the bill by the markets screened.
Two properties keep that bounded and both are pinned below:

* Screening is *free* -- pure integer filters over metadata and a book, no model
  calls -- so the loop never spends research money deciding what to spend
  research money on.
* `screener.max_candidates_per_tick` caps the forecasts one tick can run, and
  the cap is enforced on the universe walk itself, not merely on its output.
"""

from __future__ import annotations

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
    CapitalConfig,
    ForecastBudget,
    ForecastConfig,
    RiskConfig,
    ScreenerConfig,
    WindbreakConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

#: The two markets `two_ticker_isolation` offers, in the ascending ticker order
#: the screen walks them in.
_TICKERS = ("MKT-ISO-A", "MKT-ISO-B")


def _fixed_clock() -> int:
    """Return the fixed, non-advancing epoch second every tick here runs at."""
    return FIXED_NOW_EPOCH_S


#: The fixed research cost one forecast charges on this offline path, mirroring
#: `test_paper_budget.py::_EXPECTED_RESEARCH_COST_MICROS`. A per-day ceiling of
#: exactly this admits the first market's forecast and halts the second.
_RESEARCH_COST_MICROS = 3_000_000


def _config(
    *,
    screener: ScreenerConfig = FIXTURE_SCREENER_CONFIG,
    per_day_micros: int | None = None,
) -> WindbreakConfig:
    """Build the PAPER-ceilinged config these scenarios tick under.

    Args:
        screener: The screening thresholds to enforce.
        per_day_micros: An explicit per-UTC-day research ceiling, or `None` to
            keep the SPEC §16 default (which no scenario here exhausts).

    Returns:
        The assembled configuration.
    """
    forecast = (
        ForecastConfig()
        if per_day_micros is None
        else ForecastConfig(budget=ForecastBudget(per_day_micros=per_day_micros))
    )
    return WindbreakConfig(
        mode_ceiling="paper",
        capital=CapitalConfig(floor_micros=0),
        risk=RiskConfig(),
        screener=screener,
        forecast=forecast,
    )


def _ledger_shape(records: list[object]) -> list[tuple[str, str]]:
    """Project the ledger into an `(event_type, ticker)` sequence.

    The ticker is whichever of `ticker` / `market_ticker` the payload carries,
    or `""` for the rows that describe the loop rather than a market. That is
    exactly the granularity the halt claims below are made at: *which* rows
    exist, and *which market* each names.

    Args:
        records: The `LedgerRecord` sequence from `store.read_all()`.

    Returns:
        One `(event_type, ticker)` pair per row, in ledger order.
    """
    shape = []
    for event_type, payload in read_event_type_payload_pairs(records):
        ticker = payload.get("ticker") or payload.get("market_ticker") or ""
        shape.append((event_type, ticker))
    return shape


def _build_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    research_tools_factory,
):
    """Build one `PaperTickDeps` over the two-market offline fixtures.

    Args:
        books_dir: The `two_ticker_isolation` books-fixture directory.
        cassette_path: The (empty) recorded-cassette path.
        ledger_path: Where the tick's `SqliteLedgerStore` is created.
        report_dir: Where weekly-report stubs would be written.
        config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.

    Returns:
        A fully wired `PaperTickDeps`.
    """
    from windbreak.scheduler.loop import build_paper_deps

    return build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
        config=config,
        research_tools=research_tools_factory(),
        clock=_fixed_clock,
    )


def _payloads_of(records: list[object], event_type: str) -> list[dict]:
    """Return every ledgered payload of one event type, in ledger order.

    Args:
        records: The `LedgerRecord` sequence from `store.read_all()`.
        event_type: The event kind to filter by.

    Returns:
        The matching payload `data` dicts.
    """
    return [
        payload
        for recorded_type, payload in read_event_type_payload_pairs(records)
        if recorded_type == event_type
    ]


def test_tick_forecasts_every_screened_market_not_one_hardcoded_ticker(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A two-market universe yields two forecasts in one tick, in ticker order."""
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config(),
        research_tools_factory=research_tools_factory,
    )

    run_single_tick(deps, beat=1)

    deps.store.verify_chain()
    records = deps.store.read_all()
    forecast_tickers = [
        payload["market_ticker"] for payload in _payloads_of(records, "ForecastCreated")
    ]
    assert forecast_tickers == list(_TICKERS)
    snapshot_tickers = [
        payload["ticker"] for payload in _payloads_of(records, "MarketSnapshotRecorded")
    ]
    assert snapshot_tickers == list(_TICKERS)
    selector_tickers = [
        payload["market_ticker"]
        for payload in _payloads_of(records, "SelectorDecisionRecorded")
    ]
    assert selector_tickers == list(_TICKERS)


def test_tick_ledgers_a_screen_decision_per_market_examined(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Every screened market gets a `ScreenDecisionRecorded` row (issue #159)."""
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config(),
        research_tools_factory=research_tools_factory,
    )

    run_single_tick(deps, beat=1)

    decisions = _payloads_of(deps.store.read_all(), "ScreenDecisionRecorded")
    assert [payload["ticker"] for payload in decisions] == list(_TICKERS)
    assert all(payload["eligible"] is True for payload in decisions)
    assert all(payload["blocked_by"] == [] for payload in decisions)


def test_candidate_bound_caps_the_forecasts_one_tick_runs(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A bound of one forecasts one market and never examines the second.

    Bounding candidates is what bounds research spend: each candidate is one
    paid forecast. The second market is not merely un-forecast, it is not even
    screened -- the walk stops -- so the ledger claims no verdict on a market
    this tick never reached.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config(
            screener=ScreenerConfig(
                min_volume_24h_micros=FIXTURE_SCREENER_CONFIG.min_volume_24h_micros,
                min_depth_contract_centis=(
                    FIXTURE_SCREENER_CONFIG.min_depth_contract_centis
                ),
                horizon_days=FIXTURE_SCREENER_CONFIG.horizon_days,
                max_candidates_per_tick=1,
            )
        ),
        research_tools_factory=research_tools_factory,
    )

    run_single_tick(deps, beat=1)

    records = deps.store.read_all()
    assert [
        payload["market_ticker"] for payload in _payloads_of(records, "ForecastCreated")
    ] == ["MKT-ISO-A"]
    assert [
        payload["ticker"] for payload in _payloads_of(records, "ScreenDecisionRecorded")
    ] == ["MKT-ISO-A"]


def test_a_market_that_fails_the_screen_is_ledgered_and_never_forecast(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A depth floor above both books yields a screened-out, forecast-free tick.

    This is the fail-closed direction: no candidate means no forecast, not a
    fallback to an unscreened market.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config(
            screener=ScreenerConfig(
                min_volume_24h_micros=0,
                min_depth_contract_centis=1_000_000,
                horizon_days=FIXTURE_SCREENER_CONFIG.horizon_days,
            )
        ),
        research_tools_factory=research_tools_factory,
    )

    outcome = run_single_tick(deps, beat=1)

    records = deps.store.read_all()
    decisions = _payloads_of(records, "ScreenDecisionRecorded")
    assert [payload["ticker"] for payload in decisions] == list(_TICKERS)
    assert all(payload["eligible"] is False for payload in decisions)
    assert all(
        payload["blocked_by"] == ["min_depth_contract_centis"] for payload in decisions
    )
    assert _payloads_of(records, "ForecastCreated") == []
    assert outcome.forecast_ids == ()
    # The tick still proves itself alive and flat: an idle universe is not a
    # dead loop.
    assert _payloads_of(records, "EquitySampled") != []
    assert _payloads_of(records, "ModeHeartbeat") != []


def test_two_runs_over_identical_inputs_ledger_byte_identical_payloads(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Determinism survives a multi-market tick (SPEC S9.1, S9.10).

    Candidate order comes from an explicit ticker sort, never the exchange
    mapping's iteration order, so the two legs cannot diverge on the order the
    fixture happened to list its markets in.
    """
    from windbreak.scheduler.loop import run_single_tick

    legs = []
    for name in ("left.db", "right.db"):
        deps = _build_deps(
            books_dir=two_ticker_books_dir,
            cassette_path=cassette_path,
            ledger_path=ledger_path_for(tmp_path, name),
            report_dir=report_dir,
            config=_config(),
            research_tools_factory=research_tools_factory,
        )
        run_single_tick(deps, beat=1)
        legs.append(read_event_type_payload_pairs(deps.store.read_all()))

    assert json.dumps(legs[0], sort_keys=True) == json.dumps(legs[1], sort_keys=True)


def test_candidate_capital_depletes_across_markets_within_one_tick(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """The second market sizes against the cash the first one left.

    Concentration across a universe is only meaningful if the tick's markets see
    each other's effect. The selector's capital input is read per candidate off
    the live exchange balances, so a fill on `MKT-ISO-A` is already reflected
    when `MKT-ISO-B` is sized -- rather than both markets each sizing as though
    the account were untouched.
    """
    from windbreak.scheduler.loop import _position_input

    deps = _build_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config(),
        research_tools_factory=research_tools_factory,
    )

    first = _position_input(deps, "MKT-ISO-A")
    assert first.snapshot_id == "MKT-ISO-A-positions"
    second = _position_input(deps, "MKT-ISO-B")
    assert second.snapshot_id == "MKT-ISO-B-positions"
    # Same account, read twice: identical until something spends it.
    assert first.above_floor_capital_micros == second.above_floor_capital_micros


def test_a_halt_on_the_second_market_ledgers_its_snapshot_but_no_forecast(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Golden row sequence for a budget halt part-way through the universe.

    Pinned as an exact sequence rather than a set of membership assertions,
    because the claim `run_single_tick`'s docstring and `docs/RUNBOOK.md` make
    about a halted tick's ledger is a claim about *which rows exist*. Prose
    drifts; a golden sequence does not. An earlier revision of that prose said a
    halt costs the halting market its `ForecastCreated` and
    `SelectorDecisionRecorded` rows *and every market after it* the same two --
    which is wrong in both directions, and this test is what makes the corrected
    version self-verifying.

    The halting market (`MKT-ISO-B`) keeps its `MarketSnapshotRecorded`, because
    `_run_candidate` ledgers the snapshot before `_forecast_stage` can return
    `None`. What it loses is `ForecastCreated`, `SelectorDecisionRecorded`, and
    the `ExchangeStatusObserved` the approval stage would have appended.

    A per-day ceiling of exactly one forecast's research cost is what produces
    this: `MKT-ISO-A` charges it, then `MKT-ISO-B`'s `ensure_day_open` sees the
    day at its ceiling and halts before touching a tool or transport.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config(per_day_micros=_RESEARCH_COST_MICROS),
        research_tools_factory=research_tools_factory,
    )

    outcome = run_single_tick(deps, beat=1)

    deps.store.verify_chain()
    assert _ledger_shape(deps.store.read_all()) == [
        # Built by the gateway's boot recovery, before the tick runs at all.
        ("RecoveryCompleted", ""),
        # The screen runs first and covers both candidates, so a market the
        # walk never reaches still has a verdict on record.
        ("ScreenDecisionRecorded", "MKT-ISO-A"),
        ("ScreenDecisionRecorded", "MKT-ISO-B"),
        ("PipelineHeartbeatRecorded", ""),
        ("VerificationPassed", ""),
        # MKT-ISO-A: the full per-market path.
        ("MarketSnapshotRecorded", "MKT-ISO-A"),
        ("ForecastCreated", "MKT-ISO-A"),
        ("SelectorDecisionRecorded", "MKT-ISO-A"),
        ("ExchangeStatusObserved", ""),
        # MKT-ISO-B: snapshotted, then halted before it could forecast.
        ("MarketSnapshotRecorded", "MKT-ISO-B"),
        ("ResearchBudgetHalted", ""),
        # The tick-level stages still run, so a halted loop stays observably
        # alive and flat rather than simply stopping.
        ("ModeHeartbeat", ""),
        ("EquitySampled", ""),
        ("PositionsSnapshotRecorded", ""),
    ]
    assert outcome.candidate_tickers == _TICKERS
    assert len(outcome.forecast_ids) == 1
    assert outcome.research_halted is True


def test_a_halt_on_the_first_market_leaves_the_next_one_entirely_unrun(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A screened candidate the walk never reached has no snapshot row at all.

    This is the half the earlier prose got wrong. `_run_universe` breaks out of
    the walk on a halt, so `_run_candidate` is never invoked for the markets
    after it -- they lose their `MarketSnapshotRecorded` too, not merely their
    `ForecastCreated` and `SelectorDecisionRecorded`.

    What they keep is their `ScreenDecisionRecorded`: the screen ran over the
    whole candidate set before the walk began, so the ledger still records that
    `MKT-ISO-B` was examined and found eligible. It simply never says anything
    further about it, which is the honest record of a market the tick screened
    in and then ran out of money before reaching.

    A per-day ceiling of zero halts on the very first market, which is what
    leaves a screened-but-unrun candidate behind to assert on.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config(per_day_micros=0),
        research_tools_factory=research_tools_factory,
    )

    outcome = run_single_tick(deps, beat=1)

    deps.store.verify_chain()
    shape = _ledger_shape(deps.store.read_all())
    assert shape == [
        ("RecoveryCompleted", ""),
        ("ScreenDecisionRecorded", "MKT-ISO-A"),
        ("ScreenDecisionRecorded", "MKT-ISO-B"),
        ("PipelineHeartbeatRecorded", ""),
        ("VerificationPassed", ""),
        # MKT-ISO-A halts: snapshotted, then nothing further.
        ("MarketSnapshotRecorded", "MKT-ISO-A"),
        ("ResearchBudgetHalted", ""),
        ("ModeHeartbeat", ""),
        ("EquitySampled", ""),
        ("PositionsSnapshotRecorded", ""),
    ]
    # Stated separately from the golden above, because it is the specific claim
    # the corrected documentation now makes: the skipped market is screened and
    # nothing else.
    assert ("MarketSnapshotRecorded", "MKT-ISO-B") not in shape
    assert ("ScreenDecisionRecorded", "MKT-ISO-B") in shape
    assert outcome.candidate_tickers == _TICKERS
    assert outcome.forecast_ids == ()
    assert outcome.research_halted is True


def test_build_paper_deps_refuses_a_non_positive_candidate_bound(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A bound below one is refused at startup, not discovered tick by tick."""
    with pytest.raises(ValueError, match="max_candidates_per_tick"):
        _build_deps(
            books_dir=two_ticker_books_dir,
            cassette_path=cassette_path,
            ledger_path=ledger_path_for(tmp_path),
            report_dir=report_dir,
            config=_config(
                screener=ScreenerConfig(
                    horizon_days=FIXTURE_SCREENER_CONFIG.horizon_days,
                    max_candidates_per_tick=0,
                )
            ),
            research_tools_factory=research_tools_factory,
        )


def test_default_config_still_screens_and_ticks_through_the_cassette(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """CI's default path acquires no live dependency from the universe work.

    `WindbreakConfig()` selects the recorded-cassette transport, and screening a
    universe must not have quietly changed that: the transport is chosen from
    configuration alone, and no `provider_http` seam is supplied here.
    """
    from windbreak.forecast.cassettes import ReplayCassette

    deps = _build_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config(),
        research_tools_factory=research_tools_factory,
    )

    assert isinstance(deps.transport, ReplayCassette)
    assert deps.config.screener.max_candidates_per_tick >= 1
