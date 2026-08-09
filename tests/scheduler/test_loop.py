"""Per-stage failing-first tests for `windbreak.scheduler.loop` (issue #48, RED).

`windbreak/scheduler/` does not exist yet -- only `windbreak/__init__.py` and its
sibling packages do -- so every import below of `windbreak.scheduler.loop` fails
collection with `ModuleNotFoundError: No module named 'windbreak.scheduler'`,
the expected Gate 1 RED state for issue #48.

This module pins the per-stage composition contract the ralph-chief-architect
specified, plus a handful of small, invented supporting names this test suite
needs and documents inline (the architect fixed `PaperTickDeps`,
`build_paper_deps`, `run_single_tick`, `TickOutcome`, `ApprovalSeam`, and
`KernelApproval`; everything else below -- `build_evaluation_context`,
`risk_limits_from_config`, `compute_equity_micros`, `is_quote_fresh`,
`market_snapshot_event_to_record` -- is this test's own minimal, documented
invention for the per-stage seams, kept small on purpose).

The single most load-bearing fact this module proves (issue #48's own
"Load-bearing constraint"): composing the *real*, unmodified
`RiskKernel.evaluate_intent` with the *real* `ApprovalPipeline.approve` via
`KernelApproval` cannot mint a token on a context carrying no verification
evidence -- the three reconciliation checks fail closed on a `None` snapshot.
`test_kernel_approval_vetoes_before_minting_any_token` pins the *exact* veto
reasons this yields, mirroring
`tests/riskkernel/test_checks.py::test_default_checks_over_permissive_context_leaves_only_stubs_vetoing`
but with `verification=None` instead of that test's permissive CLEAN snapshot.

Note what `verification=None` means here since issue #353: it is no longer
"the honest PAPER-loop wiring". The loop now runs a real read-only
verification cycle every tick and threads its snapshot into the context
(`windbreak/scheduler/loop.py::_verification_stage`), so `None` is now the
*absent-evidence* case this suite deliberately constructs, not the loop's
steady state. What still gates a real PAPER fill is pinned end to end in
`tests/integration/test_paper_verification.py`; keep this module's assertions
about the fail-closed contract, and read that one for the live answer.
Issue #110 promoted `exchange_status_ok` / `pipeline_heartbeat_ok` from stub
to real logic; both pass here because `tests.riskkernel.conftest.make_context`'s
defaults (a fresh, `OPEN` `exchange_status` and a fresh
`pipeline_heartbeat_epoch_s`) are deliberately permissive, exactly like every
other real check's default -- it is `build_evaluation_context`'s *own*,
separately-pinned fail-closed PAPER wiring
(`test_build_evaluation_context_fails_closed_on_exchange_status_and_heartbeat`)
that supplies `None` for both in production.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from tests.riskkernel.conftest import DEFAULT_MARKET_TICKER, make_context, make_intent
from tests.scheduler.conftest import (
    DEFAULT_NOW_EPOCH_S,
    build_kernel_approval_components,
    proven_flat_exposure,
    proven_untraded_day,
)
from windbreak.numeric.types import MoneyMicros
from windbreak.riskkernel.modes import Mode

if TYPE_CHECKING:
    from pathlib import Path

#: The three veto reasons a PAPER-mode evaluation with `verification=None` must
#: produce, in the exact SPEC S10.3 check order
#: (`windbreak/riskkernel/checks.py::_SPEC_10_3_CHECK_NAMES`): the three
#: reconciliation checks (positions 5-7), each failing closed on the missing
#: verification snapshot. Issue #340 removed a fourth reason -- the
#: `jurisdiction_product_eligibility` stub at position 2 -- by making that check
#: real; it passes here because `tests.riskkernel.conftest.make_context`'s
#: defaults supply an eligible jurisdiction and a tradable product.
#:
#: This tuple is compared with exact equality on purpose. Never soften it to a
#: membership check: it is the repo's designated proof of what an *absent*
#: verification snapshot costs, and a membership assertion would silently
#: tolerate a new veto. Since issue #353 it is no longer the list of reasons
#: gating a real PAPER fill -- the loop supplies a real snapshot now; see
#: `tests/integration/test_paper_verification.py` for that list.
_EXPECTED_VETO_REASONS = (
    "balance verification stale or missing",
    "position verification stale or missing",
    "open-order verification stale or missing",
)


# --- ApprovalSeam / KernelApproval composition (the load-bearing constraint) ----


def test_kernel_approval_vetoes_before_minting_any_token() -> None:
    """`KernelApproval.decide` vetoes and mints no token (issue #48, #110).

    Composes the real `RiskKernel.evaluate_intent` (for the ledgered audit
    event) with the real `ApprovalPipeline.approve` (for the reserve-and-issue
    path), over an otherwise fully-permissive context (every one of the 24
    real SPEC S10.3 checks passes -- `tests.riskkernel.conftest.make_context`'s
    documented guarantee) except `verification=None`. Exactly the three known
    reasons veto; the pipeline's `approve` is never reached far enough to
    reserve capital or issue a token.
    """
    from windbreak.scheduler.loop import ApprovalOutcome, KernelApproval

    kernel, pipeline, _writer = build_kernel_approval_components()
    approval = KernelApproval(kernel, pipeline)
    intent = make_intent()
    context = make_context(
        mode=Mode.PAPER,
        verification=None,
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
    )

    outcome = approval.decide(intent, context)

    assert isinstance(outcome, ApprovalOutcome)
    assert outcome.token is None
    assert outcome.decision.vetoed is True
    assert outcome.decision.reasons == _EXPECTED_VETO_REASONS


def test_kernel_approval_ledgers_exactly_one_intent_vetoed_event() -> None:
    """The kernel's own ledgered audit trail carries exactly one veto event.

    `KernelApproval` must not double-record: `RiskKernel.evaluate_intent`
    ledgers the audit `IntentVetoed` event once, and a vetoed decision must
    never reach `ApprovalPipeline.approve`'s reservation-ledger writes.
    """
    from windbreak.scheduler.loop import KernelApproval

    kernel, pipeline, writer = build_kernel_approval_components()
    approval = KernelApproval(kernel, pipeline)
    intent = make_intent()
    context = make_context(mode=Mode.PAPER, verification=None)

    approval.decide(intent, context)

    vetoed_events = [
        event for event in writer.events if event.event_type == "IntentVetoed"
    ]
    reservation_events = [
        event for event in writer.events if event.event_type == "ReservationCreated"
    ]
    approval_events = [
        event for event in writer.events if event.event_type == "ApprovalTokenIssued"
    ]
    assert len(vetoed_events) == 1
    assert reservation_events == []
    assert approval_events == []


def test_kernel_approval_mints_a_token_when_every_check_passes() -> None:
    """Given a context where every one of the 24 checks passes, a token mints.

    Proves `KernelApproval` is not *structurally* incapable of approving --
    only today's stub/verification wiring blocks it -- by excluding the one
    remaining hard-veto stub (plus, defensively, the two former #110 stubs,
    now real checks that already pass given `make_context`'s permissive
    defaults) and supplying a permissive `VerificationSnapshot` (mirroring
    `tests.riskkernel.conftest`'s own default), so this test does not silently
    pass for the wrong reason (e.g. a `KernelApproval` that always vetoes).
    """
    import dataclasses

    from tests.riskkernel.conftest import make_verification_snapshot
    from windbreak.riskkernel import checks as checks_module
    from windbreak.scheduler.loop import KernelApproval

    kernel, pipeline, _writer = build_kernel_approval_components()
    approval = KernelApproval(kernel, pipeline)
    intent = make_intent()
    context = make_context(
        mode=Mode.PAPER,
        verification=make_verification_snapshot(),
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
    )
    real_only_checks = tuple(
        check
        for check in checks_module.DEFAULT_CHECKS
        if check.name not in {"exchange_status_ok", "pipeline_heartbeat_ok"}
        and check.name != "jurisdiction_product_eligibility"
    )
    # Neither `RiskKernel.evaluate_intent` nor `ApprovalPipeline.approve`
    # exposes a seam to override `DEFAULT_CHECKS`; both call
    # `checks.evaluate_intent(intent, effective)` via a module-attribute
    # lookup (`from windbreak.riskkernel import checks`), so patching the
    # attribute on the shared `windbreak.riskkernel.checks` module object
    # affects both call sites identically -- proving the composition end to
    # end (kernel evaluates and ledgers `IntentApproved`, then the pipeline
    # re-evaluates, reserves, and mints) rather than just one half of it.
    original_evaluate_intent = checks_module.evaluate_intent

    def _patched_evaluate_intent(
        intent_arg: object, context_arg: object, checks: object = real_only_checks
    ) -> object:
        return original_evaluate_intent(intent_arg, context_arg, checks)  # type: ignore[arg-type]

    checks_module.evaluate_intent = _patched_evaluate_intent  # type: ignore[assignment]
    try:
        outcome = approval.decide(intent, context)
    finally:
        checks_module.evaluate_intent = original_evaluate_intent  # type: ignore[assignment]

    assert outcome.token is not None
    assert outcome.token.claims.intent_id == intent.intent_id
    assert dataclasses.is_dataclass(outcome.token.claims)


# --- config -> RiskLimits/AccountState mapping ---------------------------------


def test_build_evaluation_context_maps_capital_floor_from_config() -> None:
    """`build_evaluation_context` maps `config.capital.floor_micros` to
    `RiskLimits.floor`, so the composed PAPER context honors the operator's
    configured equity floor rather than some hardcoded value.
    """
    from windbreak.config.schema import CapitalConfig, WindbreakConfig
    from windbreak.scheduler.loop import build_evaluation_context

    config = WindbreakConfig(capital=CapitalConfig(floor_micros=42_000_000))

    context = build_evaluation_context(
        config,
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=None,
        exchange_status_epoch_s=None,
        pipeline_heartbeat_epoch_s=None,
        quote_snapshot_epoch_s=None,
        exchange_clock_epoch_s=None,
        forecast_epoch_s=None,
        open_position=None,
        equity_start_of_day=None,
        visible_depth=None,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )

    assert context.limits.floor == MoneyMicros(42_000_000)


def test_build_evaluation_context_maps_risk_thresholds_from_config() -> None:
    """`build_evaluation_context` maps every `config.risk` ttl/threshold field
    it has a `RiskLimits` counterpart for, not just the floor.
    """
    from windbreak.config.schema import RiskConfig, WindbreakConfig
    from windbreak.scheduler.loop import build_evaluation_context

    config = WindbreakConfig(
        risk=RiskConfig(quote_ttl_seconds=17, clock_skew_max_seconds=3)
    )

    context = build_evaluation_context(
        config,
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=None,
        exchange_status_epoch_s=None,
        pipeline_heartbeat_epoch_s=None,
        quote_snapshot_epoch_s=None,
        exchange_clock_epoch_s=None,
        forecast_epoch_s=None,
        open_position=None,
        equity_start_of_day=None,
        visible_depth=None,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )

    assert context.limits.quote_ttl_seconds == 17
    assert context.limits.clock_skew_max_seconds == 3


def test_build_evaluation_context_fails_closed_on_verification_none() -> None:
    """`verification=None` flows straight through -- the fail-closed default.

    No production default is threaded in its place: a forgotten wiring
    reaching the real checks must fail closed via the three reconciliation
    checks (mirrors `windbreak.riskkernel.context.EvaluationContext`'s own
    documented "no production default" contract for this field).
    """
    from windbreak.config.schema import WindbreakConfig
    from windbreak.scheduler.loop import build_evaluation_context

    context = build_evaluation_context(
        WindbreakConfig(),
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=None,
        exchange_status_epoch_s=None,
        pipeline_heartbeat_epoch_s=None,
        quote_snapshot_epoch_s=None,
        exchange_clock_epoch_s=None,
        forecast_epoch_s=None,
        open_position=None,
        equity_start_of_day=None,
        visible_depth=None,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )

    assert context.verification is None


def test_build_evaluation_context_fails_closed_on_exchange_status_and_heartbeat() -> (
    None
):
    """Absent status/heartbeat evidence lands as `None` and fails closed.

    Issue #342 made these three values caller-supplied rather than hardcoded
    `None`, so the loop can pass real evidence. This test keeps the original
    fail-closed intent: when the caller genuinely has nothing to supply, the
    values must land on the context as `None` and be vetoed by
    `exchange_status_ok` / `pipeline_heartbeat_ok` -- never quietly defaulted
    to something permissive. It is the negative half of the pair; the positive
    half lives in the checks-pass tests below.
    """
    from windbreak.config.schema import WindbreakConfig
    from windbreak.scheduler.loop import build_evaluation_context

    context = build_evaluation_context(
        WindbreakConfig(),
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=None,
        exchange_status_epoch_s=None,
        pipeline_heartbeat_epoch_s=None,
        quote_snapshot_epoch_s=None,
        exchange_clock_epoch_s=None,
        forecast_epoch_s=None,
        open_position=None,
        equity_start_of_day=None,
        visible_depth=None,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )

    assert context.market.exchange_status is None
    assert context.market.exchange_status_epoch_s is None
    assert context.pipeline_heartbeat_epoch_s is None


def test_build_evaluation_context_stamps_now_epoch_s_verbatim() -> None:
    """The supplied `now_epoch_s` is stamped verbatim -- never `time.time()`."""
    from windbreak.config.schema import WindbreakConfig
    from windbreak.scheduler.loop import build_evaluation_context

    context = build_evaluation_context(
        WindbreakConfig(),
        now_epoch_s=1_234_567,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=None,
        exchange_status_epoch_s=None,
        pipeline_heartbeat_epoch_s=None,
        quote_snapshot_epoch_s=None,
        exchange_clock_epoch_s=None,
        forecast_epoch_s=None,
        open_position=None,
        equity_start_of_day=None,
        visible_depth=None,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )

    assert context.now_epoch_s == 1_234_567


@pytest.mark.parametrize(
    ("quote_epoch_s", "expected"),
    [(None, None), (1_234_000, 1_234_000)],
    ids=["absent-book", "real-book"],
)
def test_build_evaluation_context_never_stamps_the_quote_with_its_own_clock(
    quote_epoch_s: int | None, expected: int | None
) -> None:
    """The quote epoch is the caller's, never `now_epoch_s` (issue #369).

    Both arms are asserted against a `now_epoch_s` deliberately *unequal* to
    the supplied quote epoch, so the substitution this issue removes would show
    up as `1_234_567` in either arm: an absent book must land as `None` and
    keep `quote_freshness` vetoing, and a real book's `fetched_at` must land
    verbatim -- with its genuine age, not a zero one.

    Args:
        quote_epoch_s: The caller's quote epoch, absent or real.
        expected: What must land on `MarketView.quote_snapshot_epoch_s`.
    """
    from windbreak.config.schema import WindbreakConfig
    from windbreak.scheduler.loop import build_evaluation_context

    context = build_evaluation_context(
        WindbreakConfig(),
        now_epoch_s=1_234_567,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=None,
        exchange_status_epoch_s=None,
        pipeline_heartbeat_epoch_s=None,
        quote_snapshot_epoch_s=quote_epoch_s,
        exchange_clock_epoch_s=None,
        forecast_epoch_s=None,
        open_position=None,
        equity_start_of_day=None,
        visible_depth=None,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )

    assert context.market.quote_snapshot_epoch_s == expected


# --- equity math (scaled ints only, no float) ----------------------------------


def test_compute_equity_micros_sums_cash_and_positions_value_exactly() -> None:
    """Equity is the exact integer sum of available cash and positions value."""
    from windbreak.scheduler.loop import compute_equity_micros

    equity = compute_equity_micros(
        available_cash=MoneyMicros(100_000_000),
        positions_value=MoneyMicros(25_000_000),
    )

    assert equity == MoneyMicros(125_000_000)


def test_compute_equity_micros_rejects_a_float_argument() -> None:
    """A float can never enter the equity path (SPEC S6.1): passing one raises.

    `MoneyMicros.__post_init__` already rejects a non-int `.value`, so
    smuggling a float in via a raw (non-`MoneyMicros`) argument must raise
    rather than silently truncate or coerce.
    """
    from windbreak.scheduler.loop import compute_equity_micros

    with pytest.raises((TypeError, AttributeError)):
        compute_equity_micros(available_cash=1_000_000.5, positions_value=0)  # type: ignore[arg-type]


# --- stale-quote skip via ensure_fresh ------------------------------------------


def test_is_quote_fresh_true_within_ttl() -> None:
    """A quote exactly at the ttl boundary is fresh (inclusive), per
    `windbreak.connector.freshness.is_fresh`'s own documented boundary.
    """
    from windbreak.connector.models import OrderBookSnapshot
    from windbreak.scheduler.loop import is_quote_fresh

    fetched_at = datetime(2026, 1, 1, tzinfo=UTC)
    book = OrderBookSnapshot(
        ticker=DEFAULT_MARKET_TICKER, yes_bids=(), yes_asks=(), fetched_at=fetched_at
    )

    fresh = is_quote_fresh(
        book, ttl_seconds=10, now=datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC)
    )

    assert fresh is True


def test_is_quote_fresh_false_past_ttl() -> None:
    """A quote one second past its ttl is stale, never silently accepted."""
    from windbreak.connector.models import OrderBookSnapshot
    from windbreak.scheduler.loop import is_quote_fresh

    fetched_at = datetime(2026, 1, 1, tzinfo=UTC)
    book = OrderBookSnapshot(
        ticker=DEFAULT_MARKET_TICKER, yes_bids=(), yes_asks=(), fetched_at=fetched_at
    )

    fresh = is_quote_fresh(
        book, ttl_seconds=10, now=datetime(2026, 1, 1, 0, 0, 11, tzinfo=UTC)
    )

    assert fresh is False


# --- connector-event -> ledger adapter ------------------------------------------


def test_market_snapshot_event_to_record_carries_best_bid_and_ask() -> None:
    """The adapter projects a market + book into a `MarketSnapshotRecorded`
    carrying the top-of-book best bid/ask, in pips (never a float).
    """
    from windbreak.connector.models import OrderBookLevel, OrderBookSnapshot
    from windbreak.ledger.events import MarketSnapshotRecorded
    from windbreak.numeric import ContractCentis, PricePips
    from windbreak.scheduler.loop import market_snapshot_event_to_record

    fetched_at = datetime(2026, 1, 1, tzinfo=UTC)
    book = OrderBookSnapshot(
        ticker=DEFAULT_MARKET_TICKER,
        yes_bids=(OrderBookLevel(PricePips(4500), ContractCentis(300)),),
        yes_asks=(OrderBookLevel(PricePips(4600), ContractCentis(200)),),
        fetched_at=fetched_at,
    )

    event = market_snapshot_event_to_record(
        ticker=DEFAULT_MARKET_TICKER, order_book=book, component="scheduler"
    )

    assert isinstance(event, MarketSnapshotRecorded)
    assert event.ticker == DEFAULT_MARKET_TICKER
    assert event.best_bid_pips == 4500
    assert event.best_ask_pips == 4600


def test_market_snapshot_event_to_record_handles_an_empty_book_side() -> None:
    """A one-sided (or empty) book projects `None` for the missing side, never
    a crash or a fabricated zero price.
    """
    from windbreak.connector.models import OrderBookSnapshot
    from windbreak.ledger.events import MarketSnapshotRecorded
    from windbreak.scheduler.loop import market_snapshot_event_to_record

    fetched_at = datetime(2026, 1, 1, tzinfo=UTC)
    book = OrderBookSnapshot(
        ticker=DEFAULT_MARKET_TICKER, yes_bids=(), yes_asks=(), fetched_at=fetched_at
    )

    event = market_snapshot_event_to_record(
        ticker=DEFAULT_MARKET_TICKER, order_book=book, component="scheduler"
    )

    assert isinstance(event, MarketSnapshotRecorded)
    assert event.best_bid_pips is None
    assert event.best_ask_pips is None


# --- Issue #339: the budget-event -> ledger-event translation table ----------


def test_sqlite_budget_ledger_writer_maps_a_day_exhausted_event(
    tmp_path: Path,
) -> None:
    """A per-day exhaustion becomes a `per_day` halt row with an empty ticker.

    Exact values are asserted rather than mere presence: the per-day payload's
    spend field is named `spent_micros`, and a mutant that read `budget_micros`
    into `spent_micros` (or vice versa) would survive any "a row exists" check.
    """
    from windbreak.forecast.budget import BUDGET_DAY_EXHAUSTED_EVENT, BudgetEvent
    from windbreak.ledger.store import SqliteLedgerStore
    from windbreak.scheduler.loop import _SqliteBudgetLedgerWriter

    store = SqliteLedgerStore(tmp_path / "day.db")
    writer = _SqliteBudgetLedgerWriter(store)

    writer.record(
        BudgetEvent(
            BUDGET_DAY_EXHAUSTED_EVENT,
            {
                "utc_day": "2024-12-24",
                "spent_micros": 6_000_000,
                "budget_micros": 6_000_000,
            },
            "2024-12-24T00:00:00.000000Z",
        )
    )

    records = store.read_all()
    assert [record.event_type for record in records] == ["ResearchBudgetHalted"]
    assert json.loads(records[0].payload_json)["data"] == {
        "market_ticker": "",
        "halt_kind": "per_day",
        "utc_day": "2024-12-24",
        "spent_micros": 6_000_000,
        "budget_micros": 6_000_000,
    }


def test_sqlite_budget_ledger_writer_maps_a_forecast_exceeded_event(
    tmp_path: Path,
) -> None:
    """A per-forecast breach becomes a `per_forecast` halt row carrying the ticker.

    Pins the field-name asymmetry between the two kinds: the per-forecast
    payload names the spend `cost_micros`, not `spent_micros`.
    """
    from windbreak.forecast.budget import BUDGET_FORECAST_EXCEEDED_EVENT, BudgetEvent
    from windbreak.ledger.store import SqliteLedgerStore
    from windbreak.scheduler.loop import _SqliteBudgetLedgerWriter

    store = SqliteLedgerStore(tmp_path / "forecast.db")
    writer = _SqliteBudgetLedgerWriter(store)

    writer.record(
        BudgetEvent(
            BUDGET_FORECAST_EXCEEDED_EVENT,
            {
                "cost_micros": 3_000_000,
                "budget_micros": 2_999_999,
                "market_ticker": "MKT-DEEP",
                "utc_day": "2024-12-24",
            },
            "2024-12-24T00:00:00.000000Z",
        )
    )

    records = store.read_all()
    assert json.loads(records[0].payload_json)["data"] == {
        "market_ticker": "MKT-DEEP",
        "halt_kind": "per_forecast",
        "utc_day": "2024-12-24",
        "spent_micros": 3_000_000,
        "budget_micros": 2_999_999,
    }


def test_sqlite_budget_ledger_writer_rejects_an_unhandled_budget_event_type(
    tmp_path: Path,
) -> None:
    """An unrecognized budget event raises rather than dropping an audit row.

    `COST_REPORT` is a real budget event kind this writer deliberately does not
    translate. Failing loudly is the fail-closed choice: a silently swallowed
    event would be an audit row that simply never appears.
    """
    from windbreak.forecast.budget import COST_REPORT_EVENT, BudgetEvent
    from windbreak.ledger.store import SqliteLedgerStore
    from windbreak.scheduler.loop import _SqliteBudgetLedgerWriter

    store = SqliteLedgerStore(tmp_path / "unhandled.db")
    writer = _SqliteBudgetLedgerWriter(store)

    with pytest.raises(ValueError, match="COST_REPORT"):
        writer.record(BudgetEvent(COST_REPORT_EVENT, {}, "2024-12-24T00:00:00.000000Z"))

    assert store.read_all() == []


# --- Issue #342: real status and heartbeat evidence -------------------------


def _check_named(name: str):
    """Return the real SPEC S10.3 check registered under `name`.

    Args:
        name: The pinned check name.

    Returns:
        The check callable from the production `DEFAULT_CHECKS` sequence.
    """
    from windbreak.riskkernel.checks import DEFAULT_CHECKS

    return next(check for check in DEFAULT_CHECKS if check.name == name)


def _context_with(*, status, status_epoch_s: int | None, heartbeat_epoch_s: int | None):
    """Build a PAPER context carrying the given liveness evidence.

    Args:
        status: The projected `ExchangeTradingStatus`, or `None`.
        status_epoch_s: Epoch second the status was observed, or `None`.
        heartbeat_epoch_s: Epoch second the pipeline was seen alive, or `None`.

    Returns:
        The composed `EvaluationContext`.
    """
    from windbreak.config.schema import WindbreakConfig
    from windbreak.scheduler.loop import build_evaluation_context

    return build_evaluation_context(
        WindbreakConfig(),
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=status,
        exchange_status_epoch_s=status_epoch_s,
        pipeline_heartbeat_epoch_s=heartbeat_epoch_s,
        quote_snapshot_epoch_s=None,
        exchange_clock_epoch_s=None,
        forecast_epoch_s=None,
        open_position=None,
        equity_start_of_day=None,
        visible_depth=None,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )


def test_real_status_and_heartbeat_evidence_clears_both_liveness_checks() -> None:
    """Healthy, fresh evidence makes both liveness checks approve.

    This is the positive half of issue #342: before it, no value a caller could
    supply would let these two checks pass, because the loop hardcoded `None`.
    """
    from windbreak.riskkernel.context import ExchangeTradingStatus

    context = _context_with(
        status=ExchangeTradingStatus.OPEN,
        status_epoch_s=DEFAULT_NOW_EPOCH_S,
        heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S,
    )
    intent = make_intent()

    assert _check_named("exchange_status_ok")(intent, context).vetoed is False
    assert _check_named("pipeline_heartbeat_ok")(intent, context).vetoed is False


@pytest.mark.parametrize(
    ("age_seconds", "expected_vetoed"),
    [(60, False), (61, True)],
    ids=["fresh", "stale"],
)
def test_pipeline_heartbeat_vetoes_exactly_past_its_ttl(
    age_seconds: int, expected_vetoed: bool
) -> None:
    """The heartbeat is fresh at exactly its ttl and stale one second later.

    Pinned against the production ttl (60s), not a permissive test fixture. The
    `fresh` case is the one that carries real signal: the `stale` case would
    pass vacuously before issue #342, since `None` also vetoed.

    Args:
        age_seconds: How old the heartbeat is at evaluation time.
        expected_vetoed: Whether `pipeline_heartbeat_ok` must veto.
    """
    from windbreak.riskkernel.context import ExchangeTradingStatus

    context = _context_with(
        status=ExchangeTradingStatus.OPEN,
        status_epoch_s=DEFAULT_NOW_EPOCH_S,
        heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S - age_seconds,
    )

    result = _check_named("pipeline_heartbeat_ok")(make_intent(), context)

    assert result.vetoed is expected_vetoed


@pytest.mark.parametrize("status_name", ["PAUSED", "CLOSED"], ids=["paused", "closed"])
def test_a_non_open_exchange_status_vetoes_for_not_open(status_name: str) -> None:
    """A genuinely observed non-open status vetoes, distinguishably from absence.

    The reason must be `not open for trading`, not `stale or missing`: an
    operator has to be able to tell "the exchange is shut" from "we have no
    idea". That distinction is why the projection keeps PAUSED and CLOSED as
    real members instead of collapsing them to `None`.

    Args:
        status_name: The non-tradable `ExchangeTradingStatus` member name.
    """
    from windbreak.riskkernel.context import ExchangeTradingStatus

    context = _context_with(
        status=getattr(ExchangeTradingStatus, status_name),
        status_epoch_s=DEFAULT_NOW_EPOCH_S,
        heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S,
    )

    result = _check_named("exchange_status_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "exchange not open for trading"


def test_absent_status_vetoes_as_stale_or_missing_not_as_closed() -> None:
    """Absent evidence vetoes with the missing-evidence reason, not a verdict."""
    context = _context_with(status=None, status_epoch_s=None, heartbeat_epoch_s=None)

    result = _check_named("exchange_status_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "exchange status stale or missing"


# --- Issue #364: real start-of-day equity and real visible depth -------------


#: One whole UTC day, in seconds -- used to place a ledgered equity sample on
#: the calendar day *before* the one under test.
_ONE_DAY_SECONDS = 86_400

#: The fixture book's shallower visible side, in contract-centis: 300 resting
#: on the single bid level against 1200 across the two ask levels.
_SHALLOW_SIDE_CENTIS = 300

#: The largest order the SPEC S16 default `max_participation_ppm` (250000 ppm =
#: 25%) admits against `_SHALLOW_SIDE_CENTIS` of visible depth: 75 centis.
_CAP_AT_SHALLOW_SIDE_CENTIS = 75


def _book_with(*, bid_centis: int | None, ask_centis: int | None):
    """Build a one-level-per-side book carrying the given visible quantities.

    Args:
        bid_centis: Resting quantity on the single YES bid level, or `None`
            for a book with no bids at all.
        ask_centis: Resting quantity on the single YES ask level, or `None`
            for a book with no asks at all.

    Returns:
        The composed `OrderBookSnapshot`.
    """
    from windbreak.connector.models import OrderBookLevel, OrderBookSnapshot
    from windbreak.numeric import ContractCentis, PricePips

    bids = (
        ()
        if bid_centis is None
        else (OrderBookLevel(PricePips(4500), ContractCentis(bid_centis)),)
    )
    asks = (
        ()
        if ask_centis is None
        else (OrderBookLevel(PricePips(4600), ContractCentis(ask_centis)),)
    )
    return OrderBookSnapshot(
        ticker=DEFAULT_MARKET_TICKER,
        yes_bids=bids,
        yes_asks=asks,
        fetched_at=datetime.fromtimestamp(DEFAULT_NOW_EPOCH_S, UTC),
    )


def _ledger_with_equity_samples(path: Path, samples: tuple[tuple[int, int], ...]):
    """Append `(epoch_s, equity_micros)` samples to a fresh ledger, in order.

    Uses a real `SqliteLedgerStore` and the real `EquitySampled` event rather
    than a hand-rolled record double, so the fold under test reads the very
    envelope shape `windbreak.scheduler.loop._equity_and_positions_stage`
    writes.

    Args:
        path: Where the ledger database is created.
        samples: The `(epoch_s, equity_micros)` pairs to append, in order.

    Returns:
        The opened `SqliteLedgerStore`.
    """
    from windbreak.ledger.events import EquitySampled
    from windbreak.ledger.store import SqliteLedgerStore

    store = SqliteLedgerStore(path)
    for epoch_s, equity_micros in samples:
        store.append(
            EquitySampled(
                component="scheduler",
                equity_micros=equity_micros,
                floor_micros=0,
                epoch_s=epoch_s,
            )
        )
    return store


def test_start_of_day_equity_micros_reads_the_utc_days_first_sample(
    tmp_path: Path,
) -> None:
    """The baseline is the *earliest* sample of the current UTC day, not the last.

    A later, larger sample must not become the baseline: `daily_loss_limit`
    measures today's loss against where the day *started*, so reading the most
    recent sample would silently raise the loss threshold every time equity
    grew intraday.
    """
    from windbreak.numeric import MoneyMicros
    from windbreak.scheduler.loop import start_of_day_equity_micros

    store = _ledger_with_equity_samples(
        tmp_path / "ledger.db",
        (
            (DEFAULT_NOW_EPOCH_S - 3_600, 90_000_000),
            (DEFAULT_NOW_EPOCH_S, 140_000_000),
        ),
    )

    baseline = start_of_day_equity_micros(
        store.read_all(), now_epoch_s=DEFAULT_NOW_EPOCH_S
    )

    assert baseline == MoneyMicros(90_000_000)


def test_start_of_day_equity_micros_ignores_a_previous_utc_days_samples(
    tmp_path: Path,
) -> None:
    """Yesterday's samples are not today's baseline -- the day boundary is real.

    Yesterday's equity is a genuine figure, but it is the wrong one: carrying it
    forward would let a day that has not yet sampled anything trade against a
    stale baseline instead of failing closed.
    """
    from windbreak.scheduler.loop import start_of_day_equity_micros

    store = _ledger_with_equity_samples(
        tmp_path / "ledger.db",
        ((DEFAULT_NOW_EPOCH_S - _ONE_DAY_SECONDS, 90_000_000),),
    )

    baseline = start_of_day_equity_micros(
        store.read_all(), now_epoch_s=DEFAULT_NOW_EPOCH_S
    )

    assert baseline is None


def test_start_of_day_equity_micros_is_none_on_an_empty_ledger(
    tmp_path: Path,
) -> None:
    """No sample at all means no baseline -- never a permissive zero-by-default."""
    from windbreak.scheduler.loop import start_of_day_equity_micros

    store = _ledger_with_equity_samples(tmp_path / "ledger.db", ())

    baseline = start_of_day_equity_micros(
        store.read_all(), now_epoch_s=DEFAULT_NOW_EPOCH_S
    )

    assert baseline is None


def test_visible_depth_centis_takes_the_shallower_visible_side() -> None:
    """Depth is the shallower of the two sides, summed across every level.

    The context is composed once per tick, before any intent's side is known,
    so the bound has to hold for either side: taking the deeper side would
    admit an order larger than the other side could absorb.
    """
    from windbreak.numeric import ContractCentis
    from windbreak.scheduler.loop import visible_depth_centis

    depth = visible_depth_centis(_book_with(bid_centis=300, ask_centis=1200))

    assert depth == ContractCentis(300)


def test_visible_depth_centis_is_zero_for_a_one_sided_book() -> None:
    """An empty side is an observed zero, which vetoes -- never an unknown.

    Zero depth admits no positive-size order at all (`participation cap
    exceeded`), so a one-sided book fails closed on the cap rather than on
    absent evidence.
    """
    from windbreak.numeric import ContractCentis
    from windbreak.scheduler.loop import visible_depth_centis

    depth = visible_depth_centis(_book_with(bid_centis=None, ask_centis=1200))

    assert depth == ContractCentis(0)


def _exposure_context(*, equity_start_of_day, visible_depth):
    """Build a PAPER context carrying the given exposure evidence.

    Every other feed is the loop's own production wiring; only the two figures
    issue #364 supplies vary, so each assertion below isolates them.

    Args:
        equity_start_of_day: The `MoneyMicros` start-of-day baseline, or `None`
            when no sample exists for the current UTC day.
        visible_depth: The `ContractCentis` visible book depth, or `None` when
            the book could not be read.

    Returns:
        The composed `EvaluationContext`.
    """
    from windbreak.config.schema import WindbreakConfig
    from windbreak.riskkernel.context import ExchangeTradingStatus
    from windbreak.scheduler.loop import build_evaluation_context

    return build_evaluation_context(
        WindbreakConfig(),
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=ExchangeTradingStatus.OPEN,
        exchange_status_epoch_s=DEFAULT_NOW_EPOCH_S,
        pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S,
        quote_snapshot_epoch_s=None,
        exchange_clock_epoch_s=None,
        forecast_epoch_s=None,
        open_position=None,
        equity_start_of_day=equity_start_of_day,
        visible_depth=visible_depth,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )


def test_absent_equity_baseline_and_depth_still_veto_both_exposure_checks() -> None:
    """With neither figure available, both checks keep failing closed.

    This is the half of issue #364 that must never regress: an absent baseline
    or an unreadable book loosens nothing, because a fabricated value for
    either would raise the daily-loss threshold and the participation ceiling
    on invented evidence.
    """
    context = _exposure_context(equity_start_of_day=None, visible_depth=None)
    intent = make_intent()

    loss = _check_named("daily_loss_limit")(intent, context)
    participation = _check_named("participation_cap_compliance")(intent, context)

    assert loss.vetoed is True
    assert loss.reason == "daily loss limit reached"
    assert participation.vetoed is True
    assert participation.reason == "visible depth unknown"


def test_a_genuinely_zero_equity_baseline_still_vetoes() -> None:
    """A real zero baseline vetoes exactly as an absent one does.

    `AccountState.equity_start_of_day` has no `None`, so absence is carried as
    zero -- and zero floors the daily-loss threshold at zero, which today's
    zero realized loss already reaches. Pinned so the mapping of absence onto
    zero can never be mistaken for a permissive default.
    """
    from windbreak.numeric import MoneyMicros

    context = _exposure_context(equity_start_of_day=MoneyMicros(0), visible_depth=None)

    result = _check_named("daily_loss_limit")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "daily loss limit reached"


def test_a_real_equity_baseline_clears_the_daily_loss_veto() -> None:
    """A genuine start-of-day baseline lets today's zero loss pass the limit.

    The positive half of issue #364: before it, no value a caller could supply
    would let this check pass, because the loop hardcoded a zero baseline.
    """
    from windbreak.numeric import ContractCentis, MoneyMicros

    context = _exposure_context(
        equity_start_of_day=MoneyMicros(100_000_000),
        visible_depth=ContractCentis(_SHALLOW_SIDE_CENTIS),
    )

    result = _check_named("daily_loss_limit")(make_intent(), context)

    assert result.vetoed is False


@pytest.mark.parametrize(
    ("size_centis", "expected_vetoed"),
    [
        (_CAP_AT_SHALLOW_SIDE_CENTIS, False),
        (_CAP_AT_SHALLOW_SIDE_CENTIS + 1, True),
    ],
    ids=["at-cap", "one-over-cap"],
)
def test_participation_cap_vetoes_exactly_past_its_share_of_real_depth(
    size_centis: int, expected_vetoed: bool
) -> None:
    """The cap binds at 25% of the real depth, and one centi past it vetoes.

    Pinned against the SPEC S16 default `max_participation_ppm`, not a
    hand-tuned fixture: the `at-cap` case is the one carrying signal, since the
    over-cap case vetoed vacuously while depth was `None`.

    Args:
        size_centis: The intent size, in contract-centis.
        expected_vetoed: Whether `participation_cap_compliance` must veto.
    """
    from windbreak.numeric import ContractCentis

    context = _exposure_context(
        equity_start_of_day=None,
        visible_depth=ContractCentis(_SHALLOW_SIDE_CENTIS),
    )
    intent = make_intent(size=ContractCentis(size_centis))

    result = _check_named("participation_cap_compliance")(intent, context)

    assert result.vetoed is expected_vetoed


# --- Issue #361: the loop reads the connector's positions, never its own fold --


#: The sole ticker in the `deep_walk` replay fixture -- deliberately not
#: `tests.riskkernel.conftest.DEFAULT_MARKET_TICKER`, which names a market the
#: book fixtures do not carry.
_DEEP_WALK_TICKER = "MKT-DEEP"

#: The NO-side taker fill the `deep_walk` book produces for a 100-centi order
#: limited at 5500 NO pips: 75 centis at 5500, with 25 centis left resting.
_NO_FILL_CENTIS = 75
_NO_FILL_PRICE_PIPS = 5500

#: The same holding in the YES frame, as `PaperExchange.get_positions` reports
#: it: a long NO is a short YES, priced at the complement of the NO average.
_NO_HOLDING_YES_FRAME_CENTIS = -_NO_FILL_CENTIS
_NO_HOLDING_YES_FRAME_PIPS = 10_000 - _NO_FILL_PRICE_PIPS

#: What that holding is worth, in micros: the cash it actually cost, priced in
#: its own NO frame (75 centis * 5500 pips). Marking it at the *YES*-frame
#: product would report a negative -337_500, which is not a mark at all.
_NO_HOLDING_VALUE_MICROS = _NO_FILL_CENTIS * _NO_FILL_PRICE_PIPS


def _books_fixture_dir() -> Path:
    """Return the shared `tests/fixtures/books` replay-fixture directory."""
    from pathlib import Path as _Path

    return _Path(__file__).resolve().parents[1] / "fixtures" / "books"


def _paper_exchange_holding_no():
    """Build a `PaperExchange` whose only holding is a NO-side position.

    Returns:
        The `PaperExchange`, already filled on the NO side of `MKT-DEEP`.
    """
    from windbreak.connector import paper
    from windbreak.numeric import ContractCentis, PricePips

    exchange = paper.PaperExchange.from_fixture_dir(_books_fixture_dir() / "deep_walk")
    exchange.place_order(
        paper.PaperOrderIntent(
            _DEEP_WALK_TICKER,
            "no",
            PricePips(_NO_FILL_PRICE_PIPS),
            ContractCentis(100),
        ),
        approval_token=object(),
    )
    return exchange


def _stage_deps(exchange, store):
    """Bundle the three `_equity_and_positions_stage` reads into a stand-in.

    The stage touches only `deps.exchange`, `deps.store`, and the configured
    capital floor, so a full `PaperTickDeps` (which would drag in a transport,
    a gateway, and a signing key) buys nothing here. The exchange and the
    ledger store are both the *real* production types, so the rows asserted on
    are the rows production writes.

    Args:
        exchange: The `PaperExchange` the stage observes.
        store: The ledger store the stage appends to.

    Returns:
        A `PaperTickDeps`-shaped stand-in carrying exactly those three reads.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        exchange=exchange,
        store=store,
        config=SimpleNamespace(capital=SimpleNamespace(floor_micros=0)),
    )


def _ledgered(store, event_type: str):
    """Return every payload of `event_type` on `store`, oldest first.

    Args:
        store: The ledger store to read.
        event_type: The event type to select.

    Returns:
        The matching payloads' `data` envelopes, in append order.
    """
    return [
        json.loads(record.payload_json)["data"]
        for record in store.read_all()
        if record.event_type == event_type
    ]


def test_positions_stage_reports_a_no_side_holding_exactly_as_the_connector_does(
    tmp_path: Path,
) -> None:
    """A NO-side holding is ledgered as the short-YES row the connector reports.

    The loop used to sum YES-side fills only, so an account long NO snapshotted
    as flat -- and `concentration_limits` / `reduce_only_provable`, whose whole
    job is to bound exposure, then evaluated against a position of zero. The
    row must now match `PaperExchange.get_positions()` exactly, because two
    independent definitions of "position" is the defect itself.
    """
    from windbreak.ledger.store import SqliteLedgerStore
    from windbreak.scheduler.loop import _equity_and_positions_stage

    exchange = _paper_exchange_holding_no()
    store = SqliteLedgerStore(tmp_path / "ledger.db")

    _equity_and_positions_stage(_stage_deps(exchange, store), DEFAULT_NOW_EPOCH_S)

    snapshots = _ledgered(store, "PositionsSnapshotRecorded")
    assert len(snapshots) == 1
    assert snapshots[-1]["positions"] == [
        {
            "ticker": position.ticker,
            "quantity_centis": position.quantity.value,
            "average_price_pips": position.average_price.value,
        }
        for position in exchange.get_positions()
    ]
    assert snapshots[-1]["positions"] == [
        {
            "ticker": _DEEP_WALK_TICKER,
            "quantity_centis": _NO_HOLDING_YES_FRAME_CENTIS,
            "average_price_pips": _NO_HOLDING_YES_FRAME_PIPS,
        }
    ]


def test_positions_stage_marks_a_long_no_holding_at_what_it_cost(
    tmp_path: Path,
) -> None:
    """Equity counts the NO leg at its own side's price, not the YES complement.

    A short-YES row's mark is not `quantity * average_price`: that product is
    negative, and a long NO is economically a short YES *plus* a full $1 of
    collateral per contract. Valuing it in its own frame is the same figure the
    exchange debited from cash, so buying NO moves equity by the fee alone.
    """
    from windbreak.ledger.store import SqliteLedgerStore
    from windbreak.scheduler.loop import _equity_and_positions_stage

    exchange = _paper_exchange_holding_no()
    store = SqliteLedgerStore(tmp_path / "ledger.db")

    equity = _equity_and_positions_stage(
        _stage_deps(exchange, store), DEFAULT_NOW_EPOCH_S
    )

    expected = exchange.get_balances().available.value + _NO_HOLDING_VALUE_MICROS
    assert equity == expected
    assert _ledgered(store, "EquitySampled")[-1]["equity_micros"] == expected


def test_positions_stage_records_no_snapshot_for_a_two_sided_ticker(
    tmp_path: Path,
) -> None:
    """A ticker held on both sides ledgers no position row, and never a flat one.

    `PaperExchange.get_positions` refuses to net a YES leg against a NO leg,
    because the netted answer reports flat while both legs and their collateral
    are live. The loop must inherit that refusal rather than quietly writing
    the YES leg alone (an understated holding) or an empty row list (a
    fabricated healthy zero). The tick still finishes -- an exception that kills
    the loop is not failing closed -- and still samples equity, understated by
    the unpriceable holding, which can only tighten `daily_loss_limit`.
    """
    from windbreak.connector import paper
    from windbreak.ledger.store import SqliteLedgerStore
    from windbreak.numeric import ContractCentis, PricePips
    from windbreak.scheduler.loop import _equity_and_positions_stage

    exchange = _paper_exchange_holding_no()
    exchange.place_order(
        paper.PaperOrderIntent(
            _DEEP_WALK_TICKER, "yes", PricePips(4700), ContractCentis(100)
        ),
        approval_token=object(),
    )
    with pytest.raises(paper.TwoSidedPositionError):
        exchange.get_positions()
    store = SqliteLedgerStore(tmp_path / "ledger.db")

    equity = _equity_and_positions_stage(
        _stage_deps(exchange, store), DEFAULT_NOW_EPOCH_S
    )

    assert _ledgered(store, "PositionsSnapshotRecorded") == []
    assert equity == exchange.get_balances().available.value
    assert _ledgered(store, "EquitySampled")[-1]["equity_micros"] == equity


def test_the_loop_no_longer_carries_its_own_position_fold() -> None:
    """`_positions_from_fills` is gone: one definition of "position", not two.

    The helper summed YES-side fills only and disagreed with
    `PaperExchange.get_positions()`; keeping a second fold alive is what let
    them disagree in the first place.
    """
    from windbreak.scheduler import loop

    assert not hasattr(loop, "_positions_from_fills")


# --- Issue #373: the real open position reaches `reduce_only_provable` --------


def _paper_exchange_holding_yes():
    """Build a `PaperExchange` whose only holding is a YES-side position.

    Returns:
        The `PaperExchange`, already filled on the YES side of `MKT-DEEP`.
    """
    from windbreak.connector import paper
    from windbreak.numeric import ContractCentis, PricePips

    exchange = paper.PaperExchange.from_fixture_dir(_books_fixture_dir() / "deep_walk")
    exchange.place_order(
        paper.PaperOrderIntent(
            _DEEP_WALK_TICKER, "yes", PricePips(4700), ContractCentis(100)
        ),
        approval_token=object(),
    )
    return exchange


def _paper_exchange_holding_both_sides():
    """Build a `PaperExchange` holding `MKT-DEEP` on both YES and NO.

    `get_positions` refuses to net the two legs (`TwoSidedPositionError`),
    because a netted YES-plus-NO reports flat while both legs and their
    collateral are live.

    Returns:
        The `PaperExchange`, filled on both sides of one ticker.
    """
    from windbreak.connector import paper
    from windbreak.numeric import ContractCentis, PricePips

    exchange = _paper_exchange_holding_yes()
    exchange.place_order(
        paper.PaperOrderIntent(
            _DEEP_WALK_TICKER,
            "no",
            PricePips(_NO_FILL_PRICE_PIPS),
            ContractCentis(100),
        ),
        approval_token=object(),
    )
    return exchange


def _flat_paper_exchange():
    """Build a `PaperExchange` that has never filled, so it is genuinely flat.

    Returns:
        The unfilled `PaperExchange`.
    """
    from windbreak.connector import paper

    return paper.PaperExchange.from_fixture_dir(_books_fixture_dir() / "deep_walk")


def _position_context(open_position):
    """Build a PAPER context carrying the given open position.

    Every other feed is the loop's own production wiring; only the figure issue
    #373 supplies varies, so each assertion below isolates it.

    Args:
        open_position: The `ContractCentis` holding, or `None` when it could
            not be determined.

    Returns:
        The composed `EvaluationContext`.
    """
    from windbreak.config.schema import WindbreakConfig
    from windbreak.riskkernel.context import ExchangeTradingStatus
    from windbreak.scheduler.loop import build_evaluation_context

    return build_evaluation_context(
        WindbreakConfig(),
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=ExchangeTradingStatus.OPEN,
        exchange_status_epoch_s=DEFAULT_NOW_EPOCH_S,
        pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S,
        quote_snapshot_epoch_s=DEFAULT_NOW_EPOCH_S,
        exchange_clock_epoch_s=DEFAULT_NOW_EPOCH_S,
        forecast_epoch_s=DEFAULT_NOW_EPOCH_S,
        open_position=open_position,
        equity_start_of_day=None,
        visible_depth=None,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )


def test_a_real_yes_holding_lets_a_close_within_it_prove_reduce_only() -> None:
    """The connector's own YES holding reaches the check, and a close passes.

    This is the positive half of issue #373: before it, `open_position` was a
    hardcoded `None`, so `reduce_only_provable` vetoed *every* close on every
    tick -- correctly, but permanently, because the kernel had never been told
    what the account holds. The position is read straight off
    `PaperExchange.get_positions()`, the process's one definition of
    "position" (issue #361), so the check and the ledgered snapshot can never
    disagree.
    """
    from windbreak.numeric import ContractCentis
    from windbreak.scheduler.loop import read_open_position_centis

    exchange = _paper_exchange_holding_yes()
    held = exchange.get_positions()[0].quantity

    position = read_open_position_centis(exchange, ticker=_DEEP_WALK_TICKER)
    context = _position_context(position)
    within = make_intent(action="sell_to_close", size=ContractCentis(held.value))
    beyond = make_intent(action="sell_to_close", size=ContractCentis(held.value + 1))

    assert position == held
    assert held.value > 0
    assert context.market.open_position == held
    assert _check_named("reduce_only_provable")(within, context).vetoed is False
    assert _check_named("reduce_only_provable")(beyond, context).vetoed is True


def test_a_two_sided_holding_stays_none_and_keeps_vetoing() -> None:
    """An indeterminate position fails closed -- never a fabricated zero.

    `PaperExchange.get_positions` refuses to net a YES leg against a NO leg,
    because the netted answer reports flat while both legs and their collateral
    are live. The loop must carry that refusal through as `None`: substituting
    zero would be a *claim* that the account is flat, which is exactly the
    fabricated healthy zero the connector declined to invent.
    """
    from windbreak.numeric import ContractCentis
    from windbreak.scheduler.loop import read_open_position_centis

    position = read_open_position_centis(
        _paper_exchange_holding_both_sides(), ticker=_DEEP_WALK_TICKER
    )
    context = _position_context(position)
    close = make_intent(action="sell_to_close", size=ContractCentis(1))

    assert position is None
    assert context.market.open_position is None
    assert _check_named("reduce_only_provable")(close, context).vetoed is True


def test_a_genuinely_flat_account_reports_zero_and_still_vetoes_every_close() -> None:
    """An observed absence of fills is zero, and zero still admits no close.

    Zero and `None` are not interchangeable in general -- `None` is absent
    evidence, zero is a claim -- but here the claim is one the connector
    actually makes: `get_positions` answered, and it reported no row for this
    ticker, which its own contract defines as genuinely flat. The distinction
    costs nothing in permissiveness: a close of any positive size exceeds zero,
    so the check vetoes exactly as it does on `None`.
    """
    from windbreak.numeric import ContractCentis
    from windbreak.scheduler.loop import read_open_position_centis

    exchange = _flat_paper_exchange()

    position = read_open_position_centis(exchange, ticker=_DEEP_WALK_TICKER)
    context = _position_context(position)
    close = make_intent(action="sell_to_close", size=ContractCentis(1))

    assert exchange.get_positions() == ()
    assert position == ContractCentis(0)
    assert _check_named("reduce_only_provable")(close, context).vetoed is True


def test_a_long_no_holding_lands_signed_and_never_proves_a_yes_close() -> None:
    """The YES-frame sign survives, so a NO holding cannot license a YES close.

    `get_positions` reports a long NO as a *negative* YES-frame quantity. That
    sign must reach the check verbatim: taking its magnitude would claim a
    YES-side close reduces a NO-side holding, which is false -- it opens a
    second position. `intent.size <= open_position` is then unsatisfiable for
    any positive size, which is the fail-closed answer.
    """
    from windbreak.numeric import ContractCentis
    from windbreak.scheduler.loop import read_open_position_centis

    position = read_open_position_centis(
        _paper_exchange_holding_no(), ticker=_DEEP_WALK_TICKER
    )
    context = _position_context(position)
    close = make_intent(action="sell_to_close", size=ContractCentis(1))

    assert position == ContractCentis(_NO_HOLDING_YES_FRAME_CENTIS)
    assert position.value < 0
    assert _check_named("reduce_only_provable")(close, context).vetoed is True


def test_an_unknown_ticker_reports_flat_rather_than_another_markets_holding() -> None:
    """The position is selected by ticker, never taken from whichever row exists.

    A single-market loop makes this easy to get wrong by reading `positions[0]`.
    The `two_ticker_isolation` fixture holds two markets, so a positional read
    would hand the kernel the wrong market's exposure.
    """
    from windbreak.numeric import ContractCentis
    from windbreak.scheduler.loop import read_open_position_centis

    exchange = _paper_exchange_holding_yes()

    position = read_open_position_centis(exchange, ticker="MKT-ABSENT")

    assert exchange.get_positions()[0].ticker == _DEEP_WALK_TICKER
    assert position == ContractCentis(0)


# --- Issue #377: the venue's clock, not our own, drives `clock_skew_limit` ----


def _skew_context(exchange_clock_epoch_s):
    """Build a PAPER context carrying the given venue-clock reading.

    Every other feed is the loop's own production wiring; only the figure issue
    #377 supplies varies, so each assertion below isolates it.

    Args:
        exchange_clock_epoch_s: The venue's own epoch second, or `None` when it
            could not be read.

    Returns:
        The composed `EvaluationContext`.
    """
    from windbreak.config.schema import WindbreakConfig
    from windbreak.riskkernel.context import ExchangeTradingStatus
    from windbreak.scheduler.loop import build_evaluation_context

    return build_evaluation_context(
        WindbreakConfig(),
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=ExchangeTradingStatus.OPEN,
        exchange_status_epoch_s=DEFAULT_NOW_EPOCH_S,
        pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S,
        quote_snapshot_epoch_s=DEFAULT_NOW_EPOCH_S,
        exchange_clock_epoch_s=exchange_clock_epoch_s,
        forecast_epoch_s=DEFAULT_NOW_EPOCH_S,
        open_position=None,
        equity_start_of_day=None,
        visible_depth=None,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )


#: The SPEC S16 default `clock_skew_max_seconds`, pinned rather than read back
#: from the config: a test that derived its own boundary from the value under
#: test would pass against any tolerance, including a widened one.
_CLOCK_SKEW_MAX_SECONDS = 2


@pytest.mark.parametrize(
    ("skew_seconds", "expected_vetoed"),
    [(-3, True), (-2, False), (0, False), (2, False), (3, True)],
    ids=[
        "behind-past-limit",
        "behind-at-limit",
        "agreed",
        "ahead-at-limit",
        "ahead-past-limit",
    ],
)
def test_clock_skew_vetoes_exactly_past_its_limit_in_either_direction(
    skew_seconds: int, expected_vetoed: bool
) -> None:
    """The check binds at the tolerance and one second past it, both ways.

    The `at-limit` and `agreed` cases are the ones carrying signal: while
    `exchange_clock_epoch_s` was fed the tick's own `now_epoch_s`, the skew was
    identically zero and *every* case approved, so the two vetoing cases here
    could not have been produced by any venue at any drift.

    Args:
        skew_seconds: How far the venue's clock sits from the tick's, signed.
        expected_vetoed: Whether `clock_skew_limit` must veto.
    """
    context = _skew_context(DEFAULT_NOW_EPOCH_S + skew_seconds)

    result = _check_named("clock_skew_limit")(make_intent(), context)

    assert context.limits.clock_skew_max_seconds == _CLOCK_SKEW_MAX_SECONDS
    assert result.vetoed is expected_vetoed


def test_an_unreadable_venue_clock_stays_none_and_still_vetoes() -> None:
    """A venue whose clock cannot be read fails closed, never to the local one.

    Defaulting to `now_epoch_s` is the defect this issue removes: it reads as
    perfect agreement, which is the single most reassuring answer the check can
    give and the one least supported by evidence.
    """
    context = _skew_context(None)

    result = _check_named("clock_skew_limit")(make_intent(), context)

    assert context.market.exchange_clock_epoch_s is None
    assert result.vetoed is True
    assert result.reason == "exchange clock unknown"


def test_the_loop_reads_the_venue_clock_rather_than_restamping_its_own() -> None:
    """The context's venue clock is the exchange's answer, not the tick's clock.

    Asserted against a `now_epoch_s` deliberately unequal to the anchored venue
    clock, so the removed substitution would show up as a skew of exactly zero.
    """
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime

    from windbreak.connector import paper
    from windbreak.scheduler.loop import read_exchange_clock_epoch_s

    anchor = _datetime.fromtimestamp(DEFAULT_NOW_EPOCH_S - 3_600, _UTC)
    exchange = paper.PaperExchange.from_fixture_dir(
        _books_fixture_dir() / "deep_walk",
        clock=lambda: anchor,
        replay_anchor=anchor,
    )

    context = _skew_context(read_exchange_clock_epoch_s(exchange))

    assert context.market.exchange_clock_epoch_s == DEFAULT_NOW_EPOCH_S - 3_600
    assert context.market.exchange_clock_epoch_s != context.now_epoch_s
    assert _check_named("clock_skew_limit")(make_intent(), context).vetoed is True


# --- Issue #382: an exhausted replay reports no venue clock, not a stale one -


def test_an_exhausted_replay_vetoes_for_a_stated_reason_not_as_drift() -> None:
    """A run that outlives its recording reads as unknown, never as skew.

    The anchored venue clock does not advance with wall time (#377), so before
    this issue a re-enactment that outran its recorded span kept reporting a
    literal the recording no longer substantiates and `clock_skew_limit` vetoed
    on accumulated drift -- an incidental veto that says nothing about the
    venue. Now the replay refuses, the reading is absent, and the check vetoes
    with the reason an operator can act on: the recording ran out.
    """
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime
    from datetime import timedelta as _timedelta

    from windbreak.connector import paper
    from windbreak.scheduler.loop import read_exchange_clock_epoch_s

    anchor = _datetime.fromtimestamp(DEFAULT_NOW_EPOCH_S, _UTC)
    exchange = paper.PaperExchange.from_fixture_dir(
        _books_fixture_dir() / "deep_walk",
        clock=lambda: anchor + _timedelta(hours=1),
        replay_anchor=anchor,
    )

    reading = read_exchange_clock_epoch_s(exchange)
    result = _check_named("clock_skew_limit")(make_intent(), _skew_context(reading))

    assert reading is None
    assert result.vetoed is True
    assert result.reason == "exchange clock unknown"


def test_a_replay_inside_its_recorded_span_still_reports_the_venue_clock() -> None:
    """The refusal is bounded: inside the span the reading is still served.

    Without this the change would have "fixed" the drift veto by making the
    venue clock permanently unreadable, which fails closed but also makes
    `clock_skew_limit` unable to approve for any replay -- the mirror image of
    the unfalsifiable check #377 removed.
    """
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime
    from datetime import timedelta as _timedelta

    from windbreak.connector import paper
    from windbreak.scheduler.loop import read_exchange_clock_epoch_s

    anchor = _datetime.fromtimestamp(DEFAULT_NOW_EPOCH_S, _UTC)
    exchange = paper.PaperExchange.from_fixture_dir(
        _books_fixture_dir() / "deep_walk",
        clock=lambda: anchor + _timedelta(minutes=1),
        replay_anchor=anchor,
    )

    reading = read_exchange_clock_epoch_s(exchange)
    result = _check_named("clock_skew_limit")(make_intent(), _skew_context(reading))

    assert reading == DEFAULT_NOW_EPOCH_S
    assert result.vetoed is False


# --- Issue #380: the forecast's own `created_at`, not the tick's clock -------


def _forecast_context(forecast_epoch_s):
    """Build a PAPER context carrying the given forecast-creation instant.

    Every other feed is the loop's own production wiring; only the figure issue
    #380 supplies varies, so each assertion below isolates it.

    Args:
        forecast_epoch_s: The forecast's own `created_at` in epoch seconds, or
            `None` when the tick produced no forecast at all.

    Returns:
        The composed `EvaluationContext`.
    """
    from windbreak.config.schema import WindbreakConfig
    from windbreak.riskkernel.context import ExchangeTradingStatus
    from windbreak.scheduler.loop import build_evaluation_context

    return build_evaluation_context(
        WindbreakConfig(),
        now_epoch_s=DEFAULT_NOW_EPOCH_S,
        verification=None,
        instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
        market=None,
        exchange_status=ExchangeTradingStatus.OPEN,
        exchange_status_epoch_s=DEFAULT_NOW_EPOCH_S,
        pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S,
        quote_snapshot_epoch_s=DEFAULT_NOW_EPOCH_S,
        exchange_clock_epoch_s=DEFAULT_NOW_EPOCH_S,
        forecast_epoch_s=forecast_epoch_s,
        open_position=None,
        equity_start_of_day=None,
        visible_depth=None,
        exposure=proven_flat_exposure(),
        notional_today=proven_untraded_day(),
    )


#: The loop's `_DEFAULT_FORECAST_TTL_SECONDS`, pinned as a literal rather than
#: read back off the composed limits: a test that derived its own boundary from
#: the value under test would pass against any ttl, including a widened one.
_FORECAST_TTL_SECONDS = 3_600


@pytest.mark.parametrize(
    ("age_seconds", "expected_vetoed"),
    [
        (-1, True),
        (0, False),
        (_FORECAST_TTL_SECONDS, False),
        (_FORECAST_TTL_SECONDS + 1, True),
    ],
    ids=["future-dated", "just-created", "at-ttl", "past-ttl"],
)
def test_forecast_freshness_vetoes_exactly_past_its_ttl(
    age_seconds: int, expected_vetoed: bool
) -> None:
    """The check binds at the ttl and one second past it, and on a future stamp.

    Both vetoing cases carry the signal: while `forecast_epoch_s` was fed the
    tick's own `now_epoch_s`, the age was identically zero and *every* case
    approved, so `forecast_freshness` could not have vetoed any forecast at any
    age.

    Args:
        age_seconds: How old the forecast is at evaluation time, signed --
            negative means the stamp sits in the future.
        expected_vetoed: Whether `forecast_freshness` must veto.
    """
    context = _forecast_context(DEFAULT_NOW_EPOCH_S - age_seconds)

    result = _check_named("forecast_freshness")(make_intent(), context)

    assert context.limits.forecast_ttl_seconds == _FORECAST_TTL_SECONDS
    assert result.vetoed is expected_vetoed


def test_a_tick_that_produced_no_forecast_stays_none_and_still_vetoes() -> None:
    """A research-halted tick has no forecast, and `None` keeps vetoing.

    The budget-halt path (issue #339) skips the forecast stage entirely, so
    there is no `created_at` to stamp. Substituting the tick's clock there
    would claim a forecast zero seconds old on a tick that provably produced
    none -- the most reassuring answer available and the least evidenced.
    """
    context = _forecast_context(None)

    result = _check_named("forecast_freshness")(make_intent(), context)

    assert context.market.forecast_epoch_s is None
    assert result.vetoed is True
    assert result.reason == "forecast is stale or missing"


def test_build_evaluation_context_never_stamps_the_forecast_with_its_own_clock() -> (
    None
):
    """The forecast epoch is the caller's, never `now_epoch_s` (issue #380).

    Asserted against a `now_epoch_s` deliberately unequal to the supplied
    stamp, so the removed substitution would show up as an age of exactly zero.
    """
    context = _forecast_context(DEFAULT_NOW_EPOCH_S - _FORECAST_TTL_SECONDS)

    assert (
        context.market.forecast_epoch_s == DEFAULT_NOW_EPOCH_S - _FORECAST_TTL_SECONDS
    )
    assert context.market.forecast_epoch_s != context.now_epoch_s
