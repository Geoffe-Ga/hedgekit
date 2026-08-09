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
   surviving veto reasons are exactly `concentration limit exceeded`,
   `daily loss limit reached`, and `visible depth unknown`. #353 removes
   verification as a blocker; it does not make a PAPER tick fill, and this
   test is the guard against that overclaim quietly becoming true-by-assertion.
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
    remaining veto reasons. Both survivors are honest zero/`None` feeds in this
    module's own account/market view -- `equity_start_of_day=0` floors the
    daily-loss threshold at zero, and `visible_depth` is `None` -- and neither
    is a verification concern.

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
    from windbreak.scheduler.eligibility import project_exchange_status
    from windbreak.scheduler.loop import build_evaluation_context, run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)

    context = build_evaluation_context(
        deps.config,
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
        verification=deps.kernel.latest_verification,
        instrument_whitelist=frozenset({deps.ticker}),
        market=deps.exchange.get_market(deps.ticker),
        exchange_status=project_exchange_status(
            deps.exchange.get_exchange_status().status
        ),
        exchange_status_epoch_s=DEFAULT_NOW_EPOCH_S,
        pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S,
    )
    outcome = deps.approval.decide(make_intent(market_ticker=deps.ticker), context)

    assert outcome.decision.reasons == (
        "concentration limit exceeded",
        "daily loss limit reached",
        "visible depth unknown",
    )
    assert _RECONCILIATION_VETO_REASONS.isdisjoint(outcome.decision.reasons)
    assert context.account.exchange_verified_available_cash.value == 100_000_000


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
