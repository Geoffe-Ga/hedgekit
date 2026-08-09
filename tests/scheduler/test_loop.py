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
`KernelApproval` can never mint a token today, because the
`jurisdiction_product_eligibility` SPEC S10.3 check is still an
unconditional-veto stub (`windbreak/riskkernel/checks.py`) and the three
reconciliation checks fail closed on a `None` verification snapshot.
`test_kernel_approval_vetoes_before_minting_any_token` pins the *exact* four
veto reasons this yields, mirroring
`tests/riskkernel/test_checks.py::test_default_checks_over_permissive_context_leaves_only_stubs_vetoing`
but with `verification=None` (the honest PAPER-loop wiring: no live exchange
verification cycle runs yet) instead of that test's permissive CLEAN
snapshot, so the three reconciliation checks join the one remaining stub.
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
#: membership check: it is the repo's designated proof of which reasons gate a
#: PAPER fill, and a membership assertion would silently tolerate a new veto.
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
    )

    assert context.now_epoch_s == 1_234_567


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
