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
    CorrelationConfig,
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

#: The `two_ticker_isolation` fixture's opening available cash, in micros.
_OPENING_CAPITAL_MICROS = 100_000_000

#: A limit deliberately far through `MKT-ISO-A`'s resting ask, so the order
#: fills outright against the book rather than resting as a remainder.
_CROSSING_LIMIT_PIPS = 9_900

#: The price `MKT-ISO-A`'s resting ask actually sits at, and so the price the
#: crossing order above fills at -- a taker pays the book, not its own limit.
_RESTING_ASK_PIPS = 4_400

#: The quantity filled, in contract-centis. Small enough to be taken by the
#: fixture's single 1_000-centi resting ask level without walking it.
_FILL_SIZE_CENTIS = 50

#: The fill's notional, in micros: a pip is 1e-4 $ and a centi 1e-2 contracts,
#: so `centis * pips` is exactly micros (50 * 4_400).
_FILL_NOTIONAL_MICROS = _FILL_SIZE_CENTIS * _RESTING_ASK_PIPS

#: The fee the fixture's fee model charges on that fill, in micros.
_FILL_FEE_MICROS = 10_000

#: What remains deployable after the fill: opening cash less notional and fee.
_CAPITAL_AFTER_FILL_MICROS = (
    _OPENING_CAPITAL_MICROS - _FILL_NOTIONAL_MICROS - _FILL_FEE_MICROS
)


def _config(
    *,
    screener: ScreenerConfig = FIXTURE_SCREENER_CONFIG,
    per_day_micros: int | None = None,
    correlation: CorrelationConfig | None = None,
) -> WindbreakConfig:
    """Build the PAPER-ceilinged config these scenarios tick under.

    Args:
        screener: The screening thresholds to enforce.
        per_day_micros: An explicit per-UTC-day research ceiling, or `None` to
            keep the SPEC §16 default (which no scenario here exhausts).
        correlation: The operator's declared bucket assignments, or `None` for
            the empty default. Empty is not permissive (issue #407): a market
            with no declared bucket has unprovable bucket exposure, so the
            selector declines it rather than sizing against an empty bucket.
            Only the scenario that asserts on bucket exposure declares any.

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
        correlation=correlation if correlation is not None else CorrelationConfig(),
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


def test_a_fill_on_the_first_market_debits_what_the_second_can_deploy(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """The second market sizes against the cash the first one's fill debited.

    Concentration across a universe is only meaningful if the tick's markets see
    each other's effect. `_position_input` is read *per candidate* off the live
    exchange balances, so a fill on `MKT-ISO-A` is already reflected when
    `MKT-ISO-B` is sized -- rather than both markets sizing as though the
    account were untouched.

    The fill is placed directly on the loop's own exchange rather than routed
    through a tick, for the same reason
    `test_paper_fill_reconciliation.py::_fill_the_account` does it: **a PAPER
    tick does not currently mint a token at all** -- the concentration and
    participation checks still veto every intent (see
    `tests/integration/test_paper_verification.py`). So what this pins is the
    wiring, which is the honest claim: *if* a candidate fills, the next
    candidate's deployable capital is smaller by exactly what that fill cost.
    It does not claim that a stock tick produces such a fill today.

    Every figure is pinned exactly rather than by inequality, so a change that
    merely perturbs the balance cannot pass this by accident:

    * Opening balance: 100_000_000 micros (the fixture's own `balances.json`).
    * The fill: 50 contract-centis taking the resting ask at 4_400 pips. A pip
      is 1e-4 dollars and a centi 1e-2 contracts, so the notional is exactly
      50 * 4_400 = 220_000 micros, plus a 10_000-micro fee.
    * Remaining: 100_000_000 - 230_000 = 99_770_000 micros.
    """
    from tests.scheduler.conftest import proven_flat_exposure
    from windbreak.connector.paper import PaperOrderIntent
    from windbreak.numeric.types import ContractCentis, PricePips
    from windbreak.scheduler.loop import _position_input

    deps = _build_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config(),
        research_tools_factory=research_tools_factory,
    )

    before = _position_input(deps, "MKT-ISO-A", proven_flat_exposure("MKT-ISO-A"))
    assert before.snapshot_id == "MKT-ISO-A-positions"
    assert before.above_floor_capital_micros.value == _OPENING_CAPITAL_MICROS

    placement = deps.exchange.place_order(
        PaperOrderIntent(
            ticker="MKT-ISO-A",
            side="yes",
            price=PricePips(_CROSSING_LIMIT_PIPS),
            quantity=ContractCentis(_FILL_SIZE_CENTIS),
        ),
        None,
    )
    assert placement.resting_order is None, "no remainder may rest"
    (fill,) = placement.fills
    assert fill.price.value == _RESTING_ASK_PIPS
    assert fill.quantity.value == _FILL_SIZE_CENTIS

    after = _position_input(deps, "MKT-ISO-B", proven_flat_exposure("MKT-ISO-B"))

    assert after.snapshot_id == "MKT-ISO-B-positions"
    # The load-bearing assertion: strictly less, by exactly the fill's cost.
    assert after.above_floor_capital_micros.value == _CAPITAL_AFTER_FILL_MICROS
    assert (
        before.above_floor_capital_micros.value - after.above_floor_capital_micros.value
        == _FILL_NOTIONAL_MICROS + _FILL_FEE_MICROS
    )
    # The deploy cap tracks it too, so the second market cannot size against
    # capital the first one already spent.
    assert after.total_deploy_cap_micros.value == _CAPITAL_AFTER_FILL_MICROS


def test_a_fill_on_the_first_market_binds_the_seconds_correlation_bucket(
    two_ticker_books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """The second market's bucket exposure already carries the first's fill.

    This is issue #345's open acceptance criterion, closed by issue #407: a
    single tick over two markets sharing a correlation bucket has the second
    market's sizing bound by the first market's *exposure*, not merely by the
    cash it spent.

    The sibling test above pins the capital half, which was the only
    cross-market bound in force before #407. This pins the exposure half. They
    are genuinely different bounds: capital falls by what a fill *cost*
    (notional plus fee), while bucket exposure rises by what the resulting
    position is *worth* -- and a cap on one cannot substitute for a cap on the
    other.

    `read_candidate_exposure` is read per candidate off the venue's live
    positions, so `MKT-ISO-A`'s fill is a holding the venue reports by the time
    `MKT-ISO-B` is projected. Both markets are declared into the weather bucket,
    which is what lets A's exposure reach B at all: without the declaration
    `effective_buckets` would be empty and the projection would refuse outright.

    Every figure is pinned exactly:

    * The fill: 50 contract-centis at 4_400 pips, exactly as the sibling test.
    * The resulting position is worth `quantity * average_price` == 50 * 4_400
      == 220_000 micros, which is the notional and excludes the 10_000-micro
      fee -- a position's mark value is not its acquisition cost.
    * `MKT-ISO-B` holds nothing itself, so its `market_exposure` stays 0 while
      its `bucket_exposure` is 220_000. That gap is the whole mechanism: a
      per-market cap alone would still read this account as flat in B.
    """
    from tests.scheduler.conftest import weather_bucket_correlation
    from windbreak.connector.paper import PaperOrderIntent
    from windbreak.numeric.types import ContractCentis, PricePips
    from windbreak.scheduler.loop import read_candidate_exposure
    from windbreak.scheduler.screening import MarketCandidate

    deps = _build_deps(
        books_dir=two_ticker_books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_config(
            correlation=weather_bucket_correlation("MKT-ISO-A", "MKT-ISO-B")
        ),
        research_tools_factory=research_tools_factory,
    )

    def _candidate_for(ticker: str) -> MarketCandidate:
        """Build the screened candidate for one fixture market.

        Args:
            ticker: The market to build a candidate for.

        Returns:
            The assembled candidate, carrying the venue's own book.
        """
        return MarketCandidate(
            market=deps.exchange.get_market(ticker),
            order_book=deps.exchange.get_order_book(ticker),
        )

    before = read_candidate_exposure(deps, _candidate_for("MKT-ISO-B"))
    assert before is not None
    assert before.bucket_exposure.value == 0

    placement = deps.exchange.place_order(
        PaperOrderIntent(
            ticker="MKT-ISO-A",
            side="yes",
            price=PricePips(_CROSSING_LIMIT_PIPS),
            quantity=ContractCentis(_FILL_SIZE_CENTIS),
        ),
        None,
    )
    assert placement.resting_order is None, "no remainder may rest"

    after = read_candidate_exposure(deps, _candidate_for("MKT-ISO-B"))
    assert after is not None
    # The load-bearing assertion: A's fill is exposure B is now bound by.
    assert after.bucket_exposure.value == _FILL_NOTIONAL_MICROS
    # B holds nothing of its own, so a per-market cap would still see it flat.
    assert after.market_exposure.value == 0
    assert after.total_exposure.value == _FILL_NOTIONAL_MICROS
    # And the peer that carries it is A, not B.
    (peer,) = after.bucket_peers
    assert peer.market_ticker == "MKT-ISO-A"


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
    # `candidate_tickers` is the screened-in set, not a record of what ran:
    # `MKT-ISO-B` is listed here having never reached `_run_candidate` at all.
    # Comparing it against `forecast_ids` is what says how far the tick got.
    assert outcome.candidate_tickers == _TICKERS
    assert "MKT-ISO-B" in outcome.candidate_tickers
    assert outcome.forecast_ids == ()
    assert len(outcome.forecast_ids) < len(outcome.candidate_tickers)
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
