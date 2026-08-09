"""Failing-first tests for the PAPER read-only verification cycle (issue #353).

Issue #342 wired the two liveness feeds; the three SPEC S10.3 reconciliation
checks were the last thing keeping a PAPER tick from ever minting a token,
because `windbreak.scheduler.loop` supplied `verification=None` and no
verification cycle ran in PAPER at all.

Tests 1-3 are the failing-first trio: each was written and run against the
pre-#353 loop and observed RED (no `VerificationPassed` row, no
`VerificationMismatch` row, no `verification_view` attribute). Test 4 is a
*characterization pin* added after they went green, to stop the honest
remainder from being misread as a fill:

1. `test_paper_tick_runs_a_read_only_verification_cycle_and_clears_the_vetoes`
   -- a tick must run one read-only cycle against the same `PaperExchange` the
   loop trades on, ledger its evidence (`VerificationPassed`), and leave the
   real `KernelApproval` seam able to mint a token: none of the three
   reconciliation veto reasons may survive. Today the loop wires no verifier,
   so `RiskKernel._latest_verification` stays `None`, `_stamp_verification`
   stamps `verification=None`, and all three reconciliation checks veto.
2. `test_paper_verification_mismatch_halts_the_kernel_and_ledgers_the_breach`
   -- a venue that has moved away from the baseline the kernel reconciles
   against must HALT the kernel per issue #32 (a `VerificationMismatch` plus a
   `VerificationMismatchHalt` row, and `Mode.HALT`), never merely veto.
3. `test_paper_verification_holds_no_trade_scope_write_surface` -- the object
   the verification path holds must expose no `place_order`/`cancel_order`
   (SPEC S1.1 invariant 3). `PaperExchange` itself exposes both, so handing the
   exchange straight to the verifier hands over the write surface.
4. `test_loop_production_context_vetoes_carry_no_verification_reason` -- under
   the context `windbreak.scheduler.loop._approve_stage` actually composes, the
   surviving veto reasons are exactly `_SURVIVING_VETO_REASONS`. #353 removes
   verification as a blocker; it does not make a PAPER tick fill, and this
   test is the guard against that overclaim quietly becoming true-by-assertion.
5. `test_loop_supplies_its_own_ledgered_equity_baseline_and_book_depth` (issue
   #364, RED -- today `build_evaluation_context` takes neither figure) -- the
   start-of-day equity must be read back from the `EquitySampled` row the tick
   itself ledgered, and the visible depth from the book the tick itself
   snapshotted, with `None` (and therefore a veto) before the UTC day has a
   sample at all.
6. `test_a_slow_forecast_stage_ages_its_own_forecast_out_of_the_approval`
   (issue #380, RED -- today `build_evaluation_context` takes no forecast
   epoch and stamps `MarketView.forecast_epoch_s` with its own `now_epoch_s`)
   -- the approval must carry the forecast's own `created_at`, so a forecast
   that took longer than its ttl to produce vetoes on `forecast_freshness`
   instead of reading as zero seconds old.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.integration.conftest import ledger_path_for
from tests.riskkernel.conftest import DEFAULT_NOW_EPOCH_S, make_context, make_intent
from windbreak.numeric.types import ContractCentis, PricePips
from windbreak.riskkernel.modes import Mode

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig

#: The single ticker in the shared `deep_walk` books fixture.
_TICKER = "MKT-DEEP"

#: The three issue-#32 reconciliation veto reasons a PAPER evaluation produced
#: while no verification cycle ran. None of them may survive once one does.
_RECONCILIATION_VETO_REASONS = frozenset(
    {
        "balance verification stale or missing",
        "position verification stale or missing",
        "open-order verification stale or missing",
    }
)

#: The `deep_walk` fixture book's shallower visible side, in contract-centis:
#: 300 resting on its single bid level, against 1200 across its two ask levels.
#: That is the depth `visible_depth_centis` bounds participation against, since
#: the context is composed before any intent's side is known.
_SHALLOW_SIDE_CENTIS = 300

#: What still vetoes `tests.riskkernel.conftest.make_intent` under the loop's
#: own composed context, reason by reason, in SPEC S10.3 check order:
#:
#: * `concentration limit exceeded` -- the intent's 5_000_000-micro cost is over
#:   the 2% (`max_pos_market_pct_ppm`) share of the fixture's 100_000_000-micro
#:   verified cash. Real evidence, real limit; unchanged by issue #364.
#: * `participation cap exceeded` -- the intent's 1000 centis is over the 25%
#:   (`max_participation_ppm`) share of the fixture book's 300-centi shallower
#:   side. This *replaces* `visible depth unknown`: the check now measures a
#:   real book instead of failing closed on absent evidence, and this
#:   particular test intent is simply too big for that book.
#:
#: `daily loss limit reached` is gone: issue #364 supplies the genuine
#: start-of-day equity the loop ledgers every tick, and today's zero realized
#: loss is below its 2% threshold.
#:
#: Issue #373 re-examined this tuple reason by reason and left it unchanged, on
#: purpose rather than by omission: threading the real `open_position` cannot
#: move either survivor, because `make_intent`'s default action is `buy` -- an
#: *opening* action, which `reduce_only_provable` approves outright and which
#: `_is_derisking_close` can never exempt from `concentration_limits`. What
#: changed is proven directly instead, in
#: `test_loop_production_context_carries_the_venues_own_open_position` and in
#: `tests/scheduler/test_loop.py`'s issue-#373 section.
#:
#: Compared with exact equality on purpose, mirroring
#: `tests/scheduler/test_loop.py::_EXPECTED_VETO_REASONS`. Never soften it to a
#: membership check: it is the repo's designated measurement of what actually
#: gates a PAPER fill, and a membership assertion would silently tolerate a new
#: veto -- or a silently loosened check.
_SURVIVING_VETO_REASONS = (
    "concentration limit exceeded",
    "participation cap exceeded",
)

#: What `_SURVIVING_VETO_REASONS` becomes once the run's clock has advanced
#: past `quote_ttl_seconds` while the replay stayed where it was, in SPEC S10.3
#: check order:
#:
#: * `quote is stale or missing` (position 14) -- the point of issue #369. It
#:   lands *between* the two original survivors (`concentration_limits` at 10,
#:   `participation_cap_compliance` at 17) rather than after them.
#: * `clock skew exceeds limit` (position 21) -- issue #377's consequence, and
#:   a genuine one rather than a nuisance. A replay's venue clock does not
#:   advance with wall time, so eleven seconds of wall clock against a
#:   two-second `clock_skew_max_seconds` is real divergence between our clock
#:   and the venue's. Before #377 this reason was unreachable at any drift.
#:
#: Both arrive from the same single act of moving the clock, which is precisely
#: why the tuple is compared with exact equality: a membership check would not
#: prove either veto arrived in the right place, nor that nothing *else*
#: changed when the clock moved.
_STALE_QUOTE_VETO_REASONS = (
    "concentration limit exceeded",
    "quote is stale or missing",
    "participation cap exceeded",
    "clock skew exceeds limit",
)


def _production_context(deps, *, now_epoch_s: int = DEFAULT_NOW_EPOCH_S):
    """Rebuild the exact `EvaluationContext` `_approve_stage` composes.

    Every argument mirrors that stage's own call, so what this test pins is the
    loop's production wiring rather than a permissive test fixture. In
    particular `quote_snapshot_epoch_s` comes from the book's own `fetched_at`
    (issue #369), never from `now_epoch_s`: a quote stamped with the evaluation
    instant is zero seconds old by construction and `quote_freshness` could
    never veto.

    `forecast_epoch_s` is `DEFAULT_NOW_EPOCH_S` for the same reason and by the
    same rule (issue #380): the tick under test ran on `_fixed_clock`, so the
    forecast it produced carries exactly that `created_at`. It deliberately
    does *not* track `now_epoch_s` -- a stamp that moved with the evaluation
    instant would be the substitution this pin exists to forbid.

    Args:
        deps: The wired `PaperTickDeps` whose ledger, exchange, and kernel the
            context is composed from.
        now_epoch_s: The evaluation instant, defaulting to the fixed clock every
            builder here agrees on. A caller passes a later instant to age the
            replayed book against the run's own clock.

    Returns:
        The composed `EvaluationContext`.
    """
    from windbreak.scheduler.eligibility import project_exchange_status
    from windbreak.scheduler.loop import (
        build_evaluation_context,
        read_exchange_clock_epoch_s,
        read_open_position_centis,
        start_of_day_equity_micros,
        visible_depth_centis,
    )

    order_book = deps.exchange.get_order_book(_TICKER)
    return build_evaluation_context(
        deps.config,
        now_epoch_s=now_epoch_s,
        verification=deps.kernel.latest_verification,
        instrument_whitelist=frozenset({_TICKER}),
        market=deps.exchange.get_market(_TICKER),
        exchange_status=project_exchange_status(
            deps.exchange.get_exchange_status().status
        ),
        exchange_status_epoch_s=DEFAULT_NOW_EPOCH_S,
        pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S,
        quote_snapshot_epoch_s=int(order_book.fetched_at.timestamp()),
        exchange_clock_epoch_s=read_exchange_clock_epoch_s(deps.exchange),
        forecast_epoch_s=DEFAULT_NOW_EPOCH_S,
        open_position=read_open_position_centis(deps.exchange, ticker=_TICKER),
        equity_start_of_day=start_of_day_equity_micros(
            deps.store.read_all(), now_epoch_s=now_epoch_s
        ),
        visible_depth=visible_depth_centis(order_book),
    )


def _fixed_clock() -> int:
    """Return the fixed epoch second every deps-builder call here agrees on.

    Deliberately `tests.riskkernel.conftest.DEFAULT_NOW_EPOCH_S`, the instant
    `make_context`'s permissive quote/forecast/status/heartbeat stamps are all
    struck at: the loop's injected clock is what the verification snapshot is
    stamped with, so agreeing on one instant keeps every *freshness* check out
    of the way and leaves the reconciliation checks as the only thing under
    test here.

    Returns:
        The fixed epoch second.
    """
    return DEFAULT_NOW_EPOCH_S


def _build_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    research_tools_factory,
):
    """Build one `PaperTickDeps` over the shared offline fixtures.

    Args:
        books_dir: The `deep_walk` books-fixture directory.
        cassette_path: The (empty) recorded-cassette path.
        ledger_path: Where the tick's `SqliteLedgerStore` is created.
        report_dir: Where weekly-report stubs would be written.
        config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.

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


def test_paper_tick_runs_a_read_only_verification_cycle_and_clears_the_vetoes(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A tick verifies the venue, ledgers the pass, and mints on the real seam.

    The token mint is the point: it is decided by the *real* `KernelApproval`
    (real `RiskKernel.evaluate_intent` composed with the real
    `ApprovalPipeline.approve`) with nothing patched out, so it can only
    succeed if all three reconciliation checks genuinely passed on a real
    snapshot the loop's own cycle produced.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )

    run_single_tick(deps, beat=1)

    event_types = [record.event_type for record in deps.store.read_all()]
    assert "VerificationPassed" in event_types
    snapshot = deps.kernel.latest_verification
    assert snapshot is not None
    # The snapshot the loop itself threads into the approval context -- not a
    # permissive test fixture -- so both halves of the seam judge the very
    # observation the tick's own cycle produced.
    outcome = deps.approval.decide(
        make_intent(market_ticker=_TICKER),
        make_context(
            mode=Mode.PAPER,
            verification=snapshot,
            now_epoch_s=DEFAULT_NOW_EPOCH_S,
            instrument_whitelist=frozenset({_TICKER}),
        ),
    )
    assert _RECONCILIATION_VETO_REASONS.isdisjoint(outcome.decision.reasons)
    assert outcome.decision.vetoed is False
    assert outcome.token is not None
    deps.store.verify_chain()


def test_loop_production_context_vetoes_carry_no_verification_reason(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Pin what still vetoes under the loop's *own* composed context.

    The test above proves the reconciliation checks pass on a permissive
    context. This one refuses to let that stand in for the production wiring:
    it rebuilds the exact context `_approve_stage` composes and pins the exact
    remaining veto reasons (see `_SURVIVING_VETO_REASONS` for why each one
    survives). Since issue #364 every survivor is a real limit measured against
    real evidence -- an oversized test intent against genuine equity and a
    genuine book -- rather than a check failing closed on an absent datum.

    The tuple is compared with exact equality on purpose, mirroring
    `tests/scheduler/test_loop.py::_EXPECTED_VETO_REASONS`: a membership check
    would silently tolerate a verification reason creeping back in.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)

    context = _production_context(deps)
    outcome = deps.approval.decide(make_intent(market_ticker=_TICKER), context)

    assert outcome.decision.reasons == _SURVIVING_VETO_REASONS
    assert _RECONCILIATION_VETO_REASONS.isdisjoint(outcome.decision.reasons)
    assert context.account.exchange_verified_available_cash.value == 100_000_000


def test_loop_supplies_its_own_ledgered_equity_baseline_and_book_depth(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Both issue-#364 figures come from evidence this loop itself produced.

    The baseline is not merely *a* number: it must equal the `equity_micros` of
    the very `EquitySampled` row this tick ledgered, and the depth must equal
    the shallower side of the book this tick snapshotted. Before the tick runs
    there is no sample for the UTC day at all, and the honest answer is `None`
    -- proven here rather than assumed, because a baseline that quietly
    defaulted to a number would raise the daily-loss threshold on nothing.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.numeric import ContractCentis
    from windbreak.scheduler.loop import (
        run_single_tick,
        start_of_day_equity_micros,
        visible_depth_centis,
    )

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )

    before = start_of_day_equity_micros(
        deps.store.read_all(), now_epoch_s=DEFAULT_NOW_EPOCH_S
    )
    outcome = run_single_tick(deps, beat=1)
    after = start_of_day_equity_micros(
        deps.store.read_all(), now_epoch_s=DEFAULT_NOW_EPOCH_S
    )

    assert before is None
    assert after is not None
    assert after.value == outcome.equity_micros
    context = _production_context(deps)
    assert context.account.equity_start_of_day == after
    assert context.market.visible_depth == ContractCentis(_SHALLOW_SIDE_CENTIS)
    assert context.market.visible_depth == visible_depth_centis(
        deps.exchange.get_order_book(_TICKER)
    )


def test_paper_verification_mismatch_halts_the_kernel_and_ledgers_the_breach(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A venue that moved off the reconciled baseline HALTs the kernel (#32).

    The divergence is created by trading directly on the exchange, behind the
    loop's back -- exactly the unattributed venue movement reconciliation
    exists to catch. Halting, not vetoing, is the required response.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.connector.paper import PaperOrderIntent
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)

    deps.exchange.place_order(
        PaperOrderIntent(
            ticker=_TICKER,
            side="yes",
            price=PricePips(4600),
            quantity=ContractCentis(100),
        ),
        None,
    )
    outcome = run_single_tick(deps, beat=2)

    event_types = [record.event_type for record in deps.store.read_all()]
    assert "VerificationMismatch" in event_types
    assert "VerificationMismatchHalt" in event_types
    assert deps.kernel.mode is Mode.HALT
    assert outcome.kernel_halted is True
    deps.store.verify_chain()


def test_paper_verification_holds_no_trade_scope_write_surface(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """The verification path holds a read-only view, never the exchange itself.

    SPEC S1.1 invariant 3: the verification path never holds trade-scope
    credentials. `PaperExchange` exposes `place_order`/`cancel_order`, so the
    narrow view the loop hands the verifier must not.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.
        tmp_path: The pytest scratch directory.
    """
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )

    view = deps.verification_view

    assert not hasattr(view, "place_order")
    assert not hasattr(view, "cancel_order")
    assert view.get_balances().available == deps.exchange.get_balances().available


def test_loop_context_vetoes_a_book_aged_past_its_quote_ttl(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A replayed book that ages past its ttl vetoes; a fresh one does not (#369).

    The whole point of the issue: `build_evaluation_context` stamped
    `quote_snapshot_epoch_s` with the tick's own `now_epoch_s`, so
    `_is_stale(now, now, ttl)` was `False` for every ttl and `quote_freshness`
    could not veto against any book, however old.

    Both halves are asserted, so the fix cannot be a mere inversion:

    * at the anchor instant the replayed book is zero seconds old and the
      surviving reasons are exactly `_SURVIVING_VETO_REASONS` -- no quote
      reason;
    * one second past `quote_ttl_seconds` the very same book vetoes, and the
      reason lands at its SPEC S10.3 position (`_STALE_QUOTE_VETO_REASONS`) --
      alongside `clock skew exceeds limit`, since the same clock move also
      drifts the run away from the replay's own venue clock (issue #377).

    Only the clock moves between the two evaluations. The exchange, the ledger,
    the verification snapshot, and the book itself are identical, so the
    difference is attributable to the book's age and nothing else.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.scheduler.loop import build_paper_deps, run_single_tick

    epoch = [DEFAULT_NOW_EPOCH_S]
    deps = build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools=research_tools_factory(),
        clock=lambda: epoch[0],
    )
    run_single_tick(deps, beat=1)

    fresh = deps.approval.decide(
        make_intent(market_ticker=_TICKER),
        _production_context(deps, now_epoch_s=epoch[0]),
    )

    epoch[0] = DEFAULT_NOW_EPOCH_S + paper_config.risk.quote_ttl_seconds + 1
    stale = deps.approval.decide(
        make_intent(market_ticker=_TICKER),
        _production_context(deps, now_epoch_s=epoch[0]),
    )

    assert fresh.decision.reasons == _SURVIVING_VETO_REASONS
    assert stale.decision.reasons == _STALE_QUOTE_VETO_REASONS


def test_loop_production_context_carries_the_venues_own_open_position(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """`open_position` is the connector's answer, never a hardcoded `None` (#373).

    Before this issue the loop passed `open_position=None` outright, so
    `reduce_only_provable` vetoed every close forever. The fixture account has
    never filled, so the venue's honest answer here is `ContractCentis(0)` --
    an *observed* flat, distinguishable from the `None` that means "could not
    be determined". Both still veto a close; only one of them is evidence.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)

    context = _production_context(deps)

    assert deps.exchange.get_positions() == ()
    assert context.market.open_position == ContractCentis(0)


#: The loop's `_DEFAULT_FORECAST_TTL_SECONDS`, pinned as a literal rather than
#: read back off the composed limits: a test deriving its own boundary from the
#: value under test would pass against any ttl, including a widened one.
_FORECAST_TTL_SECONDS = 3_600


def _forecast_freshness_check():
    """Return the real SPEC S10.3 `forecast_freshness` check.

    Returns:
        The production check callable, taken from `DEFAULT_CHECKS` rather than
        reconstructed, so the assertion runs the very check the kernel runs.
    """
    from windbreak.riskkernel.checks import DEFAULT_CHECKS

    return next(check for check in DEFAULT_CHECKS if check.name == "forecast_freshness")


def test_a_slow_forecast_stage_ages_its_own_forecast_out_of_the_approval(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`_approve_stage` stamps the forecast's `created_at`, not its own clock.

    The forecast stage is wrapped -- the *real* one still runs and still
    produces the real record -- so that the injected clock advances past
    `_DEFAULT_FORECAST_TTL_SECONDS` while it works. That is precisely the
    aging `_approve_stage`'s deliberate second clock read exists to expose:
    "a slow forecast stage can legitimately age the heartbeat out rather than
    being masked by a reading taken before it ran."

    While `MarketView.forecast_epoch_s` was fed that second reading, the
    forecast was zero seconds old however long the stage had taken, so this
    veto was unreachable for any forecast at any age (issue #380). The
    composed context is captured at the `build_evaluation_context` seam rather
    than through the approval seam, so the proof does not depend on the
    selector happening to emit an intent this tick.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.
        tmp_path: The pytest scratch directory.
        monkeypatch: Wraps the two `windbreak.scheduler.loop` module globals.
    """
    from windbreak.scheduler import loop as loop_module
    from windbreak.scheduler.loop import _forecast_stage as real_forecast_stage
    from windbreak.scheduler.loop import (
        build_evaluation_context as real_build_context,
    )
    from windbreak.scheduler.loop import build_paper_deps, run_single_tick

    epoch = [DEFAULT_NOW_EPOCH_S]
    deps = build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools=research_tools_factory(),
        clock=lambda: epoch[0],
    )
    forecasts = []
    contexts = []

    def _slow_forecast_stage(*args, **kwargs):
        """Run the real forecast stage, then advance the clock past the ttl.

        Args:
            *args: Forwarded verbatim to the real stage.
            **kwargs: Forwarded verbatim to the real stage.

        Returns:
            The real stage's own `ForecastRecord`.
        """
        forecast = real_forecast_stage(*args, **kwargs)
        epoch[0] = DEFAULT_NOW_EPOCH_S + _FORECAST_TTL_SECONDS + 1
        forecasts.append(forecast)
        return forecast

    def _recording_build_context(*args, **kwargs):
        """Compose the real context and keep it for inspection.

        Args:
            *args: Forwarded verbatim to the real builder.
            **kwargs: Forwarded verbatim to the real builder.

        Returns:
            The real builder's own `EvaluationContext`.
        """
        context = real_build_context(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(loop_module, "_forecast_stage", _slow_forecast_stage)
    monkeypatch.setattr(
        loop_module, "build_evaluation_context", _recording_build_context
    )
    run_single_tick(deps, beat=1)

    assert len(forecasts) == 1
    assert len(contexts) == 1
    context = contexts[0]
    assert context.market.forecast_epoch_s == int(forecasts[0].created_at.timestamp())
    assert context.market.forecast_epoch_s == DEFAULT_NOW_EPOCH_S
    assert context.now_epoch_s == DEFAULT_NOW_EPOCH_S + _FORECAST_TTL_SECONDS + 1
    assert _forecast_freshness_check()(make_intent(), context).vetoed is True
