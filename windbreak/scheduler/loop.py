"""The single always-on PAPER-mode tick composition (issue #48, SPEC S5.3).

This module is the PAPER loop's one composition root. :func:`build_paper_deps`
wires the real, unmodified Market Connector (a `PaperExchange`), Forecast Engine,
Trade Selector, Risk Kernel, Order Gateway, and Reconciler over a single
hash-chained :class:`~windbreak.ledger.store.SqliteLedgerStore`, and
:func:`run_single_tick` drives one SPEC S5.3 SINGLE order-path tick through them:

    snapshot -> forecast -> select -> approve(seam) -> (only if a token minted)
    route -> PaperExchange fill -> reconcile

appending one audit event to the ledger at every stage, plus a per-tick
``ModeHeartbeat``, an ``EquitySampled``, and a ``PositionsSnapshotRecorded``.

The approval seam is the load-bearing safety boundary: :class:`KernelApproval`
composes the *real* ``RiskKernel.evaluate_intent`` with the *real*
``ApprovalPipeline.approve``. Every SPEC S10.3 check is now real (issue
#340 promoted the last one) and the loop now observes real exchange status
and stamps a real pipeline heartbeat each tick (issue #342), so
``exchange_status_ok`` and ``pipeline_heartbeat_ok`` evaluate genuine evidence
and can pass. The PAPER tick still never fills, for one remaining honest
reason: the three reconciliation checks fail closed on the ``verification=None``
this loop supplies, because no read-only verification cycle runs in PAPER
yet. The fill leg is proven separately by driving the gateway with a genuinely
minted token through a doubled seam.

Money and equity fields are scaled integers (micros/centis/pips), never floats
(SPEC S6.1); this package is on ``scripts/lint_no_floats.py``'s denylist.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

from windbreak.config import config_hash
from windbreak.connector.freshness import is_fresh
from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.paper import PaperExchange
from windbreak.forecast.budget import (
    BUDGET_DAY_EXHAUSTED_EVENT,
    BUDGET_FORECAST_EXCEEDED_EVENT,
    DailyBudgetExhaustedError,
    PerForecastBudgetExceededError,
    ResearchBudget,
)
from windbreak.forecast.cassettes import ReplayCassette
from windbreak.forecast.pipeline import (
    PROVIDER_VOTE_COSTED_EVENT,
    InMemoryForecastLedger,
    run_pipeline,
)
from windbreak.forecast.records import BaselineQuoteSnapshot
from windbreak.forecast.sandbox import build_research_tools
from windbreak.ledger.events import (
    EquitySampled,
    ExchangeStatusObserved,
    ForecastCreated,
    MarketSnapshotRecorded,
    ModeHeartbeat,
    PipelineHeartbeatRecorded,
    PositionsSnapshotRecorded,
    ProviderVoteRecorded,
    ResearchBudgetHalted,
    SelectorDecisionRecorded,
)
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.numeric import MoneyMicros, PricePips
from windbreak.order_gateway.gateway import OrderGateway, PaperSubmitter
from windbreak.order_gateway.ledger_writer import SqliteGatewayLedgerWriter
from windbreak.order_gateway.reconciler import Reconciler
from windbreak.order_gateway.wal import WriteAheadLog
from windbreak.reports.weekly import maybe_write_weekly
from windbreak.riskkernel.context import (
    AccountState,
    EvaluationContext,
    ExchangeTradingStatus,
    FeeBounds,
    MarketView,
    RiskLimits,
)
from windbreak.riskkernel.modes import Mode, ModeStateMachine
from windbreak.riskkernel.process import RiskKernel
from windbreak.riskkernel.reservations import (
    ApprovalOutcome,
    ApprovalPipeline,
    ReservationLedger,
)
from windbreak.riskkernel.tokens import TokenIssuer
from windbreak.scheduler.eligibility import (
    project_exchange_status,
    project_jurisdiction,
    project_product_type,
)
from windbreak.scheduler.weekly_data import weekly_report_body
from windbreak.selector import select
from windbreak.selector.types import (
    FeeModelInput,
    PositionReadModelInput,
    RiskConfigInput,
    SelectorInputs,
    SlippageModelInput,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig
    from windbreak.connector.models import NormalizedMarket, OrderBookSnapshot
    from windbreak.forecast.budget import BudgetEvent
    from windbreak.forecast.cassettes import LlmTransport
    from windbreak.forecast.records import ForecastRecord
    from windbreak.forecast.sandbox import ResearchTools
    from windbreak.ledger.events import Event
    from windbreak.riskkernel.checks import Decision, OrderIntent
    from windbreak.riskkernel.verification import VerificationSnapshot
    from windbreak.selector.types import SelectorDecision
    from windbreak.tokens.verify import SignedApprovalToken

#: The component label stamped on every scheduler-authored ledger event.
_COMPONENT = "scheduler"

#: The calibration-map version tag echoed into every selector decision.
_CALIBRATION_MAP_VERSION = "v0"

#: ``halt_kind`` stamped on a halt caused by the UTC day's budget running out.
_HALT_KIND_PER_DAY = "per_day"

#: ``halt_kind`` stamped on a halt caused by one forecast breaching its ceiling.
_HALT_KIND_PER_FORECAST = "per_forecast"

#: A full parts-per-million share (100%), used for the total-position ceiling the
#: SPEC S16 ``RiskConfig`` has no dedicated field for.
_FULL_PPM = 1_000_000

#: Default max admissible forecast age, in seconds, for the risk limits mapped
#: from config (``RiskConfig`` carries a quote ttl but no forecast ttl).
_DEFAULT_FORECAST_TTL_SECONDS = 3600

#: Default max admissible verification-snapshot age, in seconds. The PAPER loop
#: supplies ``verification=None`` (fail-closed), so this only bounds a future
#: live cycle; a conservative one-hour default suffices.
_DEFAULT_VERIFICATION_TTL_SECONDS = 3600

#: Default max admissible exchange-status age, in seconds (SPEC S7.3
#: approval/submission snapshot TTL range). The PAPER loop observes a real
#: status every tick (issue #342), so this genuinely bounds that evidence.
_DEFAULT_EXCHANGE_STATUS_TTL_SECONDS = 30

#: Default max admissible pipeline-heartbeat age, in seconds. The PAPER loop
#: stamps a real heartbeat every tick (issue #342), so this genuinely bounds
#: how long a stalled pipeline may go unnoticed.
_DEFAULT_PIPELINE_HEARTBEAT_TTL_SECONDS = 60

#: The slippage-model id stamped on the selector's per-contract buffer input.
_SLIPPAGE_MODEL_ID = "paper"

#: The single ephemeral signing-key length (bytes) the kernel and gateway share
#: per process (SPEC S10.6 symmetric approval tokens).
_SIGNING_KEY_BYTES = 32

#: The bounded maximum number of ``Reconciler.run_once`` cycles a tick runs to
#: fixpoint after routing a filled order -- never an unbounded loop.
_RECONCILE_MAX_CYCLES = 5

#: The exclusive, timezone-aware lower bound for reading every paper fill: the
#: paper exchange models fills (not balances), so positions are folded from the
#: full fill history each tick.
_EPOCH_START = datetime(1, 1, 1, tzinfo=UTC)

#: The research egress host allowlisted for the default offline research tools
#: built when a caller supplies none. The offline default never actually
#: searches, so nothing is ever fetched against it.
_DEFAULT_RESEARCH_HOST = "research.local"


# --- approval seam (the load-bearing constraint) --------------------------------


class ApprovalSeam(Protocol):
    """The seam an intent is run through to (maybe) mint an approval token.

    Implemented in production by :class:`KernelApproval` (the real kernel +
    pipeline), and doubled in tests by a fixed-token seam that proves the
    gateway/exchange fill leg without depending on the kernel's stubs.
    """

    def decide(
        self, intent: OrderIntent, context: EvaluationContext
    ) -> ApprovalOutcome:
        """Evaluate ``intent`` and return its approval outcome.

        Args:
            intent: The order intent to approve.
            context: The evaluation context the checks read.

        Returns:
            The :class:`~windbreak.riskkernel.reservations.ApprovalOutcome`; its
            ``token`` is ``None`` on a veto and a signed token on approval.
        """
        ...


class KernelApproval:
    """Composes the real Risk Kernel and approval pipeline into one seam.

    ``RiskKernel.evaluate_intent`` records the ledgered audit verdict
    (``IntentVetoed``/``IntentApproved``); only when it does *not* veto is
    ``ApprovalPipeline.approve`` reached to reserve capital and mint a single-use
    token. A vetoed decision therefore never reserves capital or issues a token
    (the pipeline is never called), so the audit trail carries exactly one veto
    event and no reservation events.
    """

    def __init__(self, kernel: RiskKernel, pipeline: ApprovalPipeline) -> None:
        """Bind the seam to a kernel and its approval pipeline.

        Args:
            kernel: The Risk Kernel whose ledgered evaluation gates approval.
            pipeline: The approval pipeline that reserves and mints on a pass.
        """
        self._kernel = kernel
        self._pipeline = pipeline

    def decide(
        self, intent: OrderIntent, context: EvaluationContext
    ) -> ApprovalOutcome:
        """Evaluate through the kernel, then the pipeline only if not vetoed.

        Args:
            intent: The order intent to approve.
            context: The evaluation context the checks read.

        Returns:
            An :class:`~windbreak.riskkernel.reservations.ApprovalOutcome` with a
            ``None`` token on a veto, else the pipeline's reserve-and-mint
            outcome.
        """
        decision: Decision = self._kernel.evaluate_intent(intent, context)
        if decision.vetoed:
            return ApprovalOutcome(decision=decision, token=None)
        return self._pipeline.approve(intent, context)


class _SqliteKernelLedgerWriter:
    """A kernel/pipeline ledger writer that appends to a `SqliteLedgerStore`.

    The persisting counterpart of ``InMemoryKernelLedgerWriter`` (mirrors
    :class:`~windbreak.order_gateway.ledger_writer.SqliteGatewayLedgerWriter`), so
    the kernel's veto/approve events join the same hash-chained ledger as every
    other stage of the tick.
    """

    def __init__(self, store: SqliteLedgerStore) -> None:
        """Bind the writer to a ledger store.

        Args:
            store: The append-only store every kernel event is persisted to.
        """
        self._store = store

    def record(self, event: Event) -> None:
        """Append a kernel/pipeline event to the ledger store.

        Args:
            event: The event to persist.
        """
        self._store.append(event)


class _SqliteBudgetLedgerWriter:
    """A research-budget ledger writer that appends to a `SqliteLedgerStore`.

    The persisting counterpart of
    :class:`~windbreak.forecast.budget.InMemoryBudgetLedger` (which that module
    documents as test-only), so a fail-closed research halt joins the same
    hash-chained ledger as every other stage of the tick.

    The budget engine's own :class:`~windbreak.forecast.budget.BudgetEvent` is a
    different shape from a ledger :class:`~windbreak.ledger.events.Event`, so
    this writer is the translation seam between them. Translation is *total*:
    each of the two breach kinds maps onto a typed ``ResearchBudgetHalted`` row,
    and any other kind raises rather than silently dropping an audit row.
    """

    def __init__(self, store: SqliteLedgerStore) -> None:
        """Bind the writer to a ledger store.

        Args:
            store: The append-only store every halt event is persisted to.
        """
        self._store = store

    def record(self, event: BudgetEvent) -> None:
        """Append a budget breach to the ledger as a typed halt event.

        Payload keys are read by subscript, never ``.get`` with a default, so a
        payload-shape drift surfaces as a loud ``KeyError`` rather than a
        silently zeroed audit row. Note the two kinds name the spent amount
        differently: the per-day payload carries ``spent_micros`` (the day's
        cumulative spend) while the per-forecast payload carries ``cost_micros``
        (the single breaching forecast's cost).

        Args:
            event: The budget event to persist.

        Raises:
            ValueError: If ``event`` is neither of the two breach kinds.
        """
        payload = event.payload
        if event.event_type == BUDGET_DAY_EXHAUSTED_EVENT:
            self._store.append(
                ResearchBudgetHalted(
                    component=_COMPONENT,
                    market_ticker="",
                    halt_kind=_HALT_KIND_PER_DAY,
                    utc_day=cast("str", payload["utc_day"]),
                    spent_micros=cast("int", payload["spent_micros"]),
                    budget_micros=cast("int", payload["budget_micros"]),
                )
            )
            return
        if event.event_type == BUDGET_FORECAST_EXCEEDED_EVENT:
            self._store.append(
                ResearchBudgetHalted(
                    component=_COMPONENT,
                    market_ticker=cast("str", payload["market_ticker"]),
                    halt_kind=_HALT_KIND_PER_FORECAST,
                    utc_day=cast("str", payload["utc_day"]),
                    spent_micros=cast("int", payload["cost_micros"]),
                    budget_micros=cast("int", payload["budget_micros"]),
                )
            )
            return
        raise ValueError(f"unhandled budget event type {event.event_type!r}")


# --- small, individually-tested composition seams -------------------------------


def compute_equity_micros(
    *, available_cash: MoneyMicros, positions_value: MoneyMicros
) -> MoneyMicros:
    """Return equity as the exact integer sum of cash and positions value.

    Reading ``.value`` off each argument means a smuggled-in ``float`` (a raw,
    non-:class:`~windbreak.numeric.MoneyMicros` argument) raises rather than
    silently coercing -- no float can ever enter the equity path (SPEC S6.1).

    Args:
        available_cash: Exchange-confirmed available cash, in micros.
        positions_value: The mark value of open positions, in micros.

    Returns:
        The summed equity, in micros.
    """
    return MoneyMicros(available_cash.value + positions_value.value)


def is_quote_fresh(
    order_book: OrderBookSnapshot, *, ttl_seconds: int, now: datetime
) -> bool:
    """Return whether a book snapshot is fresh for the caller's ttl.

    Delegates to :func:`windbreak.connector.freshness.is_fresh`, so the boundary
    is inclusive and fails closed on clock skew exactly as every other freshness
    consumer's does.

    Args:
        order_book: The book snapshot to age.
        ttl_seconds: The caller's freshness budget, in whole seconds.
        now: The reference instant to measure the snapshot's age against.

    Returns:
        ``True`` when the snapshot's age is within ``[0, ttl_seconds]``.
    """
    return is_fresh(order_book.fetched_at, ttl_seconds=ttl_seconds, now=now)


def _best_bid_pips(order_book: OrderBookSnapshot) -> int | None:
    """Return the top-of-book best YES bid in pips, or ``None`` for an empty side.

    Args:
        order_book: The book snapshot to read.

    Returns:
        The best bid price in pips, or ``None`` when there are no bids.
    """
    return order_book.yes_bids[0].price.value if order_book.yes_bids else None


def _best_ask_pips(order_book: OrderBookSnapshot) -> int | None:
    """Return the top-of-book best YES ask in pips, or ``None`` for an empty side.

    Args:
        order_book: The book snapshot to read.

    Returns:
        The best ask price in pips, or ``None`` when there are no asks.
    """
    return order_book.yes_asks[0].price.value if order_book.yes_asks else None


def market_snapshot_event_to_record(
    *, ticker: str, order_book: OrderBookSnapshot, component: str
) -> MarketSnapshotRecorded:
    """Project a book snapshot into a `MarketSnapshotRecorded` audit event.

    Carries the top-of-book best bid/ask in pips (never a float), each ``None``
    for a missing (empty) book side rather than a fabricated zero price.

    Args:
        ticker: The market the snapshot is for.
        order_book: The book snapshot to project.
        component: The component label stamped on the event.

    Returns:
        The assembled :class:`~windbreak.ledger.events.MarketSnapshotRecorded`.
    """
    return MarketSnapshotRecorded(
        component=component,
        ticker=ticker,
        best_bid_pips=_best_bid_pips(order_book),
        best_ask_pips=_best_ask_pips(order_book),
        fetched_at_epoch_s=int(order_book.fetched_at.timestamp()),
    )


def _human_ack_micros(config: WindbreakConfig) -> MoneyMicros | None:
    """Return the configured human-ack notional threshold, or ``None``.

    Args:
        config: The configuration whose risk section carries the threshold.

    Returns:
        The threshold as :class:`~windbreak.numeric.MoneyMicros`, or ``None`` when
        no threshold is configured (the permissive default).
    """
    raw = config.risk.require_human_ack_above_micros
    return MoneyMicros(raw) if raw is not None else None


def _build_limits(
    config: WindbreakConfig, instrument_whitelist: frozenset[str]
) -> RiskLimits:
    """Map a configuration into the risk limits the pre-trade checks read.

    Every field with a SPEC S16 counterpart is mapped from config; the few
    ``RiskLimits`` fields the schema has no dedicated field for take conservative
    named defaults (see the module constants).

    Args:
        config: The configuration to map.
        instrument_whitelist: The tradable-ticker set for this tick.

    Returns:
        The assembled :class:`~windbreak.riskkernel.context.RiskLimits`.
    """
    risk = config.risk
    return RiskLimits(
        floor=MoneyMicros(config.capital.floor_micros),
        instrument_whitelist=instrument_whitelist,
        micro_cap=MoneyMicros(config.capital.micro_cap_micros),
        min_open_price=PricePips(risk.min_open_price_pips),
        max_open_price=PricePips(risk.max_open_price_pips),
        max_participation_ppm=risk.max_participation_ppm,
        max_pos_market_pct_ppm=risk.max_pos_market_pct_ppm,
        max_pos_event_pct_ppm=risk.max_pos_event_pct_ppm,
        max_pos_bucket_pct_ppm=risk.max_pos_bucket_pct_ppm,
        max_pos_total_pct_ppm=_FULL_PPM,
        daily_loss_limit_pct_ppm=risk.daily_loss_limit_pct_ppm,
        max_drawdown_pct_ppm=risk.max_drawdown_pct_ppm,
        max_orders_per_hour=risk.max_orders_per_hour,
        max_notional_per_day=MoneyMicros(risk.max_notional_per_day_micros),
        quote_ttl_seconds=risk.quote_ttl_seconds,
        forecast_ttl_seconds=_DEFAULT_FORECAST_TTL_SECONDS,
        clock_skew_max_seconds=risk.clock_skew_max_seconds,
        rounding_buffer=MoneyMicros(0),
        verification_ttl_seconds=_DEFAULT_VERIFICATION_TTL_SECONDS,
        require_human_ack_above_micros=_human_ack_micros(config),
        exchange_status_ttl_seconds=_DEFAULT_EXCHANGE_STATUS_TTL_SECONDS,
        pipeline_heartbeat_ttl_seconds=_DEFAULT_PIPELINE_HEARTBEAT_TTL_SECONDS,
    )


def _zero_account() -> AccountState:
    """Return a flat, zero-valued account snapshot.

    The PAPER loop honestly supplies ``verification=None``, so the reconciliation
    checks fail closed regardless of account contents; a zeroed account keeps the
    composed context valid and deterministic without pretending to know figures a
    live verification cycle would supply.

    Returns:
        A zero-valued :class:`~windbreak.riskkernel.context.AccountState`.
    """
    zero = MoneyMicros(0)
    return AccountState(
        exchange_verified_available_cash=zero,
        guaranteed_terminal_value_of_positions=zero,
        pending_kernel_reservations=zero,
        unresolved_fee_upper_bounds=zero,
        reconciliation_uncertainty_buffer=zero,
        equity_start_of_day=zero,
        equity_high_water_mark=zero,
        realized_loss_today=zero,
        market_exposure=zero,
        event_exposure=zero,
        bucket_exposure=zero,
        total_exposure=zero,
        orders_last_hour=0,
        notional_today=zero,
    )


def build_evaluation_context(
    config: WindbreakConfig,
    *,
    now_epoch_s: int,
    verification: VerificationSnapshot | None,
    instrument_whitelist: frozenset[str],
    market: NormalizedMarket | None,
    exchange_status: ExchangeTradingStatus | None,
    exchange_status_epoch_s: int | None,
    pipeline_heartbeat_epoch_s: int | None,
) -> EvaluationContext:
    """Compose the evaluation context a PAPER-mode approval reads.

    Maps the operator's configured capital floor and risk thresholds onto the
    risk limits, stamps the supplied ``now_epoch_s`` verbatim (never
    ``time.time()``), and passes ``verification`` straight through -- there is no
    production default in its place, so a forgotten wiring must fail closed via
    the reconciliation checks rather than open (mirroring
    :class:`~windbreak.riskkernel.context.EvaluationContext`'s own contract).

    The exchange status and pipeline heartbeat are caller-supplied rather than
    hardcoded (issue #342), so the loop can pass genuine observations. They are
    still ``| None``, and a caller with nothing to report must pass ``None``
    rather than a placeholder: both checks fail closed on it. In particular the
    heartbeat must be stamped by an earlier stage and never set to this
    function's own ``now_epoch_s`` -- a heartbeat equal to ``now`` can never go
    stale, which would make its check unfalsifiable and strictly worse than the
    ``None`` it replaced.

    Args:
        config: The configuration whose capital/risk sections map to the limits.
        now_epoch_s: The kernel's current wall clock, in epoch seconds.
        verification: The latest verification snapshot, or ``None`` (fail-closed).
        instrument_whitelist: The tradable-ticker set for this tick.
        market: The market being evaluated, or ``None`` when it could not be
            resolved -- which fails closed exactly like ``verification=None``.
            Its connector eligibility metadata is projected onto the kernel's
            own enums here, so the connector's ``"unknown"`` becomes ``None``
            and can never masquerade as eligible (issue #340).
        exchange_status: The observed exchange trading status, or ``None`` when
            none could be observed -- which fails closed (issue #342).
        exchange_status_epoch_s: Epoch second the status was observed, or
            ``None``. Freshness is measured against this, never against the
            tick's own clock, so a failed read cannot read as fresh.
        pipeline_heartbeat_epoch_s: Epoch second the pipeline was last observed
            alive, or ``None``. Stamped by an earlier stage, never by this
            function -- a heartbeat equal to ``now`` could never go stale and
            would make its check unfalsifiable.

    Returns:
        The composed :class:`~windbreak.riskkernel.context.EvaluationContext`.
    """
    market_view = MarketView(
        quote_snapshot_epoch_s=now_epoch_s,
        forecast_epoch_s=now_epoch_s,
        visible_depth=None,
        exchange_clock_epoch_s=now_epoch_s,
        open_position=None,
        exchange_status=exchange_status,
        exchange_status_epoch_s=exchange_status_epoch_s,
        jurisdiction_status=project_jurisdiction(
            market.jurisdiction_status if market is not None else None
        ),
        product_type=project_product_type(
            market.market_type if market is not None else None
        ),
    )
    fees = FeeBounds(max_trading_fee=MoneyMicros(0), max_settlement_fee=MoneyMicros(0))
    return EvaluationContext(
        mode=Mode.PAPER,
        limits=_build_limits(config, instrument_whitelist),
        account=_zero_account(),
        market=market_view,
        fees=fees,
        now_epoch_s=now_epoch_s,
        used_intent_ids=frozenset(),
        used_idempotency_keys=frozenset(),
        verification=verification,
        acknowledged_intent_ids=frozenset(),
        pipeline_heartbeat_epoch_s=pipeline_heartbeat_epoch_s,
    )


# --- dependency bundle and its factory ------------------------------------------


@dataclass(frozen=True)
class PaperTickDeps:
    """The immutable dependency bundle one PAPER tick runs against.

    Frozen so a tick can never mutate its own wiring; the ``approval`` seam is
    intentionally swappable via :func:`dataclasses.replace` so a test can drive
    the gateway/exchange fill leg with a doubled, fixed-token seam while reusing
    every other real component.

    Attributes:
        config: The active PAPER-ceilinged configuration.
        ticker: The single market ticker this loop ticks.
        store: The hash-chained ledger every stage appends to.
        exchange: The replay-driven paper exchange orders fill against.
        gateway: The recovered Order Gateway submissions route through.
        reconciler: The bounded reconciler run to fixpoint after a fill.
        approval: The approval seam intents are decided through.
        verification_key: The ephemeral per-process signing key the kernel mints
            and the gateway verifies under (SPEC S10.6 symmetric tokens).
        transport: The offline LLM transport the forecast vote stage would use.
        research_tools: The sandboxed, offline research tools the forecast stage
            gathers citations through.
        report_dir: Where the weekly report stub is written each tick.
        clock: The injected zero-arg epoch-second clock, for determinism.
        budget: The research spend guard every tick's forecast runs under. This
            is the bundle's one deliberately *mutable* member: the frozen
            dataclass forbids rebinding the field, not mutating the object it
            holds, and that is load-bearing. The instance is built once per
            process in :func:`build_paper_deps`, so its per-UTC-day spend bucket
            accumulates across every ``run_single_tick`` call against this
            bundle -- which is the only thing making the per-day ceiling mean
            anything on an always-on loop. :func:`dataclasses.replace` shares
            the same instance by design, so swapping the ``approval`` seam
            cannot reset the day.
    """

    config: WindbreakConfig
    ticker: str
    store: SqliteLedgerStore
    exchange: PaperExchange
    gateway: OrderGateway
    reconciler: Reconciler
    approval: ApprovalSeam
    verification_key: bytes
    transport: LlmTransport
    research_tools: ResearchTools
    report_dir: Path
    clock: Callable[[], int]
    budget: ResearchBudget


def _default_clock() -> int:
    """Return the current wall clock as whole epoch seconds.

    Casts :func:`time.time` to an ``int`` so the loop's clock stays off the
    banned float path (SPEC S6.1).

    Returns:
        The current time, in whole epoch seconds.
    """
    return int(time.time())


def _resolve_research_tools(
    research_tools: ResearchTools | None, ledger_path: Path
) -> ResearchTools:
    """Return the supplied research tools, or an offline no-network default.

    The default never actually searches (its transports find nothing), so the
    forecast pipeline abstains on zero verified citations before any fetch --
    matching the offline PAPER contract without a live network.

    Args:
        research_tools: The caller-supplied tools, or ``None``.
        ledger_path: The tick's ledger path, whose parent roots the fetch cache.

    Returns:
        A sandboxed :class:`~windbreak.forecast.sandbox.ResearchTools`.
    """
    if research_tools is not None:
        return research_tools
    transport = _OfflineResearchTransport()
    return build_research_tools(
        allowed_hosts=frozenset({_DEFAULT_RESEARCH_HOST}),
        cache_dir=ledger_path.parent.joinpath("research-cache"),
        search_transport=transport,
        fetch_transport=transport,
    )


class _OfflineResearchTransport:
    """A search/fetch transport that finds nothing (the offline default)."""

    def search(self, query: str) -> tuple[str, ...]:
        """Return no candidate URLs, unconditionally.

        Args:
            query: The (unused) subquestion text.

        Returns:
            An empty tuple, always.
        """
        del query
        return ()

    def fetch(self, url: str) -> str:
        """Never reached (search finds nothing); raises defensively.

        Args:
            url: The (unused) URL that would have been fetched.

        Raises:
            RuntimeError: Always -- reaching this is itself a wiring bug.
        """
        raise RuntimeError(
            f"offline research transport fetch unexpectedly called: {url!r}"
        )


def _build_approval(
    store: SqliteLedgerStore, config: WindbreakConfig, key: bytes
) -> KernelApproval:
    """Wire the real kernel + approval pipeline into a `KernelApproval` seam.

    The kernel tracks PAPER mode (so its ledgered evaluation stamps PAPER) with
    ``kill_integration=None`` -- kill wiring is out of scope -- and shares the one
    ephemeral signing key with the gateway. The same hash-chained ``store`` is
    wired as the kernel's ``gate_plan_store`` (issue #185), so a PAPER ->
    LIVE_MICRO promotion reads its three thresholds from the pre-registered gate
    plan on the ledger, failing closed when none is registered.

    Args:
        store: The ledger both the kernel and the pipeline record through.
        config: The configuration whose hash is stamped into minted tokens.
        key: The ephemeral 32-byte signing key.

    Returns:
        The composed :class:`KernelApproval` seam.
    """
    writer = _SqliteKernelLedgerWriter(store)
    mode_machine = ModeStateMachine(
        mode_ceiling=Mode.from_config(config.mode_ceiling), mode=Mode.PAPER
    )
    kernel = RiskKernel(
        writer,
        mode_machine=mode_machine,
        gate_plan_store=store,
        kill_integration=None,
    )
    ledger = ReservationLedger(writer)
    issuer = TokenIssuer.from_key_material(key)
    pipeline = ApprovalPipeline(ledger, issuer, config_hash=config_hash(config))
    return KernelApproval(kernel, pipeline)


def _build_gateway(
    exchange: PaperExchange,
    store: SqliteLedgerStore,
    key: bytes,
    clock: Callable[[], int],
    ledger_path: Path,
) -> OrderGateway:
    """Wire and boot-recover the Order Gateway exactly as the chaos suite does.

    The gateway is constructed over the same durable ledger, a fresh write-ahead
    log beside it, and the paper exchange as both submitter and reconciliation
    source, then ``recover()`` runs once at boot.

    Args:
        exchange: The paper exchange orders are submitted to and reconciled with.
        store: The durable ledger the gateway reads and writes through.
        key: The ephemeral verification key (the same bytes the kernel mints
            under).
        clock: The injected epoch-second clock.
        ledger_path: The ledger path whose sibling ``.wal`` file backs the WAL.

    Returns:
        A recovered :class:`~windbreak.order_gateway.gateway.OrderGateway`.
    """
    wal_path = ledger_path.parent.joinpath(ledger_path.name + ".wal")
    gateway = OrderGateway(
        PaperSubmitter(exchange),
        verification_key=key,
        clock=clock,
        ledger_writer=SqliteGatewayLedgerWriter(store),
        wal=WriteAheadLog(wal_path),
        ledger_reader=store,
        reconciliation_source=exchange,
        status_source=exchange,
    )
    gateway.recover()
    return gateway


def _build_research_budget(
    store: SqliteLedgerStore, config: WindbreakConfig
) -> ResearchBudget:
    """Build the loop's one research spend guard from configuration.

    Config is the single source of the three ceilings, and there is deliberately
    no way to inject a budget from outside: that is what makes an unlimited or
    absent budget unrepresentable rather than merely discouraged. All three
    ceilings are already scaled integers on the config, so they pass through
    untouched -- no arithmetic, and therefore no float, enters this path.

    Args:
        store: The ledger store a fail-closed halt is recorded to.
        config: The active configuration supplying the three ceilings.

    Returns:
        The process-lived research budget.

    Raises:
        ValueError: If any configured ceiling is negative -- aborting startup
            rather than degrading to an unenforceable budget.
    """
    caps = config.forecast.budget
    return ResearchBudget(
        per_forecast_micros=caps.per_forecast_micros,
        per_day_micros=caps.per_day_micros,
        max_pages=caps.max_pages,
        ledger=_SqliteBudgetLedgerWriter(store),
    )


def build_paper_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    research_tools: ResearchTools | None = None,
    clock: Callable[[], int] | None = None,
) -> PaperTickDeps:
    """Assemble every real component one PAPER tick runs against.

    Loads a :class:`~windbreak.connector.paper.PaperExchange` from ``books_dir``,
    opens the hash-chained ledger at ``ledger_path``, mints one ephemeral 32-byte
    signing key shared by the kernel and gateway (SPEC S10.6), and wires the real
    approval seam, gateway (boot-recovered), and reconciler over them.

    Args:
        books_dir: The paper-exchange fixture directory (books/markets/fees).
        cassette_path: The recorded LLM cassette the offline replay transport
            serves from (never reached when the forecast abstains offline).
        ledger_path: Where the tick's ledger database (and sibling WAL) live.
        report_dir: Where the weekly report stub is written each tick.
        config: The PAPER-ceilinged configuration.
        research_tools: The sandboxed research tools, or ``None`` for an offline
            no-network default.
        clock: The injected epoch-second clock, or ``None`` for the wall clock.

    Returns:
        A fully wired :class:`PaperTickDeps`.
    """
    resolved_clock = clock if clock is not None else _default_clock
    # The exchange must observe on the same clock the tick reads, or its status
    # attestation drifts against `now_epoch_s` and `exchange_status_ok` judges
    # freshness against two unrelated timelines (issue #342).
    exchange = PaperExchange.from_fixture_dir(
        books_dir,
        clock=lambda: datetime.fromtimestamp(resolved_clock(), UTC),
    )
    ticker = next(iter(exchange.markets))
    store = SqliteLedgerStore(ledger_path)
    key = secrets.token_bytes(_SIGNING_KEY_BYTES)
    approval = _build_approval(store, config, key)
    gateway = _build_gateway(exchange, store, key, resolved_clock, ledger_path)
    reconciler = Reconciler(
        gateway,
        ledger_reader=store,
        reconciliation_source=exchange,
        ledger_writer=SqliteGatewayLedgerWriter(store),
    )
    return PaperTickDeps(
        config=config,
        ticker=ticker,
        store=store,
        exchange=exchange,
        gateway=gateway,
        reconciler=reconciler,
        approval=approval,
        verification_key=key,
        transport=ReplayCassette.from_path(cassette_path),
        research_tools=_resolve_research_tools(research_tools, ledger_path),
        report_dir=report_dir,
        clock=resolved_clock,
        budget=_build_research_budget(store, config),
    )


# --- the single tick ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TickOutcome:
    """The summary result of one :func:`run_single_tick` call.

    Attributes:
        beat: The 1-based tick sequence number.
        forecast_id: The forecast this tick produced.
        intent_count: How many normalized intents the selector emitted.
        filled_centis: The quantity filled through the gateway this tick, in
            contract-centis (``0`` whenever the real kernel vetoes, as it always
            does today).
        equity_micros: The sampled account equity this tick, in micros.
        research_halted: Whether this tick's research was halted fail-closed on
            a budget ceiling. When ``True`` no forecast exists, so
            ``forecast_id`` is ``""`` and no selector decision was made.
    """

    beat: int
    forecast_id: str
    intent_count: int
    filled_centis: int
    equity_micros: int
    research_halted: bool = False


def _snapshot_stage(deps: PaperTickDeps) -> OrderBookSnapshot:
    """Snapshot the market's book and ledger the snapshot event.

    Args:
        deps: The tick's dependency bundle.

    Returns:
        The current order-book snapshot.
    """
    order_book = deps.exchange.get_order_book(deps.ticker)
    deps.store.append(
        market_snapshot_event_to_record(
            ticker=deps.ticker, order_book=order_book, component=_COMPONENT
        )
    )
    return order_book


def _baseline_pips(order_book: OrderBookSnapshot) -> int:
    """Return a positive baseline price for the forecast, from the book.

    Prefers the best ask, then the best bid; falls back to a nominal single pip
    only when the book is entirely empty (a baseline must be strictly positive).

    Args:
        order_book: The book snapshot to derive the baseline from.

    Returns:
        The baseline executable price, in pips (always positive).
    """
    return _best_ask_pips(order_book) or _best_bid_pips(order_book) or 1


def _forecast_stage(
    deps: PaperTickDeps, order_book: OrderBookSnapshot, created_at: datetime
) -> ForecastRecord | None:
    """Run the forecast pipeline and ledger the forecast event.

    The ledgered ``ForecastCreated`` carries the forecast's ``research_cost_micros``
    and ``market_price_baseline_pips`` (issue #188), the two fields the weekly
    evaluation/cost-meter fold reads verbatim off the payload.

    Research runs under the bundle's budget (issue #339). On a budget breach the
    pipeline raises, and this stage answers ``None`` rather than fabricating a
    forecast: in a hash-chained audit ledger an honest gap beats a
    ``ForecastCreated`` row for a tick where the engine provably never ran. The
    ``except`` deliberately names exactly the two budget errors -- widening it
    would dress an unrelated pipeline failure up as a benign budget halt. It
    ledgers nothing itself, because the budget's own ledger writer has already
    appended the durable ``ResearchBudgetHalted`` row before raising.

    Args:
        deps: The tick's dependency bundle.
        order_book: The current book snapshot the baseline is struck against.
        created_at: The injected creation instant, for determinism.

    Returns:
        The produced forecast record, or ``None`` when research halted
        fail-closed on the budget -- in which case neither a ``ForecastCreated``
        nor any ``ProviderVoteRecorded`` row is appended for this tick.
    """
    market = deps.exchange.get_market(deps.ticker)
    baseline = BaselineQuoteSnapshot(
        snapshot_id=f"{deps.ticker}-{int(order_book.fetched_at.timestamp())}",
        price_pips=_baseline_pips(order_book),
        fetched_at=order_book.fetched_at,
    )
    vote_ledger = InMemoryForecastLedger()
    try:
        forecast = run_pipeline(
            market,
            baseline,
            transport=deps.transport,
            created_at=created_at,
            research_tools=deps.research_tools,
            ledger=vote_ledger,
            budget=deps.budget,
        )
    except (DailyBudgetExhaustedError, PerForecastBudgetExceededError):
        return None
    deps.store.append(
        ForecastCreated(
            component=_COMPONENT,
            forecast_id=forecast.forecast_id,
            market_ticker=forecast.market_ticker,
            probability_ppm=forecast.probability_ppm,
            eligible_for_live=forecast.eligible_for_live,
            abstention_reason=forecast.abstention_reason,
            research_cost_micros=forecast.research_cost_micros,
            market_price_baseline_pips=forecast.market_price_baseline_pips,
        )
    )
    _ledger_provider_votes(deps, forecast.forecast_id, vote_ledger)
    return forecast


def _ledger_provider_votes(
    deps: PaperTickDeps, forecast_id: str, vote_ledger: InMemoryForecastLedger
) -> None:
    """Fold buffered per-vote cost events into ``ProviderVoteRecorded`` rows.

    The pipeline buffers one :data:`PROVIDER_VOTE_COSTED_EVENT` per ensemble
    member driven into ``vote_ledger``; this composition-root fold stamps each
    with the tick's own ``forecast_id`` and appends it to the durable store, in
    emission order, immediately after the tick's ``ForecastCreated`` (issue
    #281). A run that abstains before the vote stage buffers zero events, so
    zero ``ProviderVoteRecorded`` rows are appended.

    Args:
        deps: The tick's dependency bundle (its ``store`` receives the rows).
        forecast_id: The tick's forecast id, stamped on every appended row.
        vote_ledger: The in-memory ledger the pipeline buffered vote costs into.
    """
    for event in vote_ledger.events_by_type(PROVIDER_VOTE_COSTED_EVENT):
        payload = event.payload
        deps.store.append(
            ProviderVoteRecorded(
                component=_COMPONENT,
                forecast_id=forecast_id,
                market_ticker=cast("str", payload["market_ticker"]),
                provider=cast("str", payload["provider"]),
                model_version=cast("str", payload["model_version"]),
                vote_index=cast("int", payload["vote_index"]),
                cost_micros=cast("int", payload["cost_micros"]),
                outcome=cast("str", payload["outcome"]),
                failure_code=cast("str", payload["failure_code"]),
            )
        )


def _position_input(deps: PaperTickDeps) -> PositionReadModelInput:
    """Build the selector's capital/exposure input from the paper balances.

    Args:
        deps: The tick's dependency bundle.

    Returns:
        The :class:`~windbreak.selector.types.PositionReadModelInput` the sizing
        stage reads.
    """
    available = deps.exchange.get_balances().available
    floor = MoneyMicros(deps.config.capital.floor_micros)
    above_floor = MoneyMicros(max(available.value - floor.value, 0))
    zero = MoneyMicros(0)
    return PositionReadModelInput(
        snapshot_id=f"{deps.ticker}-positions",
        equity_micros=available,
        above_floor_capital_micros=above_floor,
        total_deploy_cap_micros=above_floor,
        market_exposure=zero,
        event_exposure=zero,
        bucket_exposure=zero,
        total_exposure=zero,
        notional_today=zero,
    )


def _select_stage(
    deps: PaperTickDeps,
    order_book: OrderBookSnapshot,
    forecast: ForecastRecord,
    created_at: datetime,
) -> SelectorDecision:
    """Run the selector over the tick's inputs and ledger the decision event.

    Args:
        deps: The tick's dependency bundle.
        order_book: The current book snapshot.
        forecast: The forecast under evaluation.
        created_at: The fee schedule's freshness stamp for this tick.

    Returns:
        The selector's decision.
    """
    inputs = SelectorInputs(
        forecast=forecast,
        calibration_map_version=_CALIBRATION_MAP_VERSION,
        order_book=order_book,
        fee_model=FeeModelInput(
            model=deps.exchange.get_fee_model(deps.ticker), as_of=created_at
        ),
        slippage_model=SlippageModelInput(
            model_id=_SLIPPAGE_MODEL_ID, per_contract_buffer_ppm=0
        ),
        positions=_position_input(deps),
        risk_config=RiskConfigInput(
            config=deps.config.risk, config_hash=config_hash(deps.config)
        ),
        correlation_tags=(),
    )
    decision = select(inputs)
    deps.store.append(
        SelectorDecisionRecorded(
            component=_COMPONENT,
            forecast_id=decision.forecast_id,
            market_ticker=decision.market_ticker,
            intent_count=len(decision.intents),
            reasons=list(decision.reasons),
        )
    )
    return decision


def _reconcile_to_fixpoint(deps: PaperTickDeps) -> None:
    """Run the reconciler to a bounded fixpoint (never unbounded).

    Args:
        deps: The tick's dependency bundle.
    """
    previous = None
    for _ in range(_RECONCILE_MAX_CYCLES):
        if deps.gateway.halted:
            return
        outcome = deps.reconciler.run_once()
        if outcome.halted or outcome == previous:
            return
        previous = outcome


def _route_intent(
    deps: PaperTickDeps, intent: OrderIntent, token: SignedApprovalToken
) -> int:
    """Route an approved intent to the gateway, then reconcile; return the fill.

    Args:
        deps: The tick's dependency bundle.
        intent: The approved order intent.
        token: The genuinely minted approval token authorizing it.

    Returns:
        The quantity filled on submission, in contract-centis (``0`` when the
        gateway did not ack).
    """
    result = deps.gateway.process_intent(intent, token)
    _reconcile_to_fixpoint(deps)
    return result.ack.filled.value if result.ack is not None else 0


def _approve_stage(
    deps: PaperTickDeps, decision: SelectorDecision, heartbeat_epoch_s: int
) -> int:
    """Approve each emitted intent through the seam; route any minted token.

    Reads the exchange status once, here at decision time rather than at
    composition time, so its freshness is measured from a genuine observation
    (issue #342). The status *value* comes from the connector and is never
    synthesized, so a paused or closed exchange still vetoes.

    PAPER fills nonetheless stay gated: ``verification=None`` still fails the
    three reconciliation checks, so ``filled_centis`` remains ``0``.

    A market the exchange cannot resolve becomes ``None`` rather than an
    exception, so an unknown ticker vetoes the tick instead of aborting it.

    Args:
        deps: The tick's dependency bundle.
        decision: The selector's decision carrying any emitted intents.
        heartbeat_epoch_s: The instant an earlier stage observed the pipeline
            alive.

    Returns:
        The total quantity filled this tick, in contract-centis.
    """
    try:
        market: NormalizedMarket | None = deps.exchange.get_market(deps.ticker)
    except UnknownMarketError:
        market = None
    observed = deps.exchange.get_exchange_status()
    status_epoch_s = int(observed.fetched_at.timestamp())
    deps.store.append(
        ExchangeStatusObserved(
            component=_COMPONENT,
            status=observed.status,
            observed_at_epoch_s=status_epoch_s,
        )
    )
    # A second clock read, deliberately not the tick-start `now_epoch_s`:
    # freshness must be judged at evaluation time, so a slow forecast stage can
    # legitimately age the heartbeat out rather than being masked by a reading
    # taken before it ran.
    context = build_evaluation_context(
        deps.config,
        now_epoch_s=deps.clock(),
        verification=None,
        instrument_whitelist=frozenset({deps.ticker}),
        market=market,
        exchange_status=project_exchange_status(observed.status),
        exchange_status_epoch_s=status_epoch_s,
        pipeline_heartbeat_epoch_s=heartbeat_epoch_s,
    )
    filled = 0
    for intent in decision.intents:
        outcome = deps.approval.decide(intent, context)
        if outcome.token is not None:
            filled += _route_intent(deps, intent, outcome.token)
    return filled


def _positions_from_fills(exchange: PaperExchange) -> list[dict[str, object]]:
    """Fold the paper exchange's fill history into open-position rows.

    The paper exchange models fills but not balances/positions, so each tick
    derives positions by summing every YES-side fill per ticker (average price is
    the quantity-weighted mean, floor-divided). Every value is a scaled integer.

    Args:
        exchange: The paper exchange whose fills are folded.

    Returns:
        One ``{ticker, quantity_centis, average_price_pips}`` row per held
        ticker, sorted by ticker for determinism (empty when flat).
    """
    aggregates: dict[str, list[int]] = {}
    for fill in exchange.get_fills(_EPOCH_START):
        if fill.side != "yes":
            continue
        entry = aggregates.setdefault(fill.ticker, [0, 0])
        entry[0] += fill.quantity.value
        entry[1] += fill.price.value * fill.quantity.value
    rows: list[dict[str, object]] = []
    for ticker, (quantity, notional) in sorted(aggregates.items()):
        if quantity <= 0:
            continue
        rows.append(
            {
                "ticker": ticker,
                "quantity_centis": quantity,
                "average_price_pips": notional // quantity,
            }
        )
    return rows


def _positions_value_micros(positions: list[dict[str, object]]) -> int:
    """Return the mark value of open positions, in micros.

    A pip is ``1e-4`` $ and a centi ``1e-2`` contracts, so
    ``price_pips * quantity_centis`` is an exact micros product.

    Args:
        positions: The position rows produced by :func:`_positions_from_fills`.

    Returns:
        The summed positions value, in micros.
    """
    total = 0
    for position in positions:
        quantity = position["quantity_centis"]
        price = position["average_price_pips"]
        if isinstance(quantity, int) and isinstance(price, int):
            total += price * quantity
    return total


def _equity_and_positions_stage(deps: PaperTickDeps, now_epoch_s: int) -> int:
    """Sample equity and snapshot positions, ledgering both events.

    Args:
        deps: The tick's dependency bundle.
        now_epoch_s: The tick's epoch-second clock reading.

    Returns:
        The sampled equity, in micros.
    """
    positions = _positions_from_fills(deps.exchange)
    equity = compute_equity_micros(
        available_cash=deps.exchange.get_balances().available,
        positions_value=MoneyMicros(_positions_value_micros(positions)),
    )
    deps.store.append(
        EquitySampled(
            component=_COMPONENT,
            equity_micros=equity.value,
            floor_micros=deps.config.capital.floor_micros,
            epoch_s=now_epoch_s,
        )
    )
    deps.store.append(
        PositionsSnapshotRecorded(component=_COMPONENT, positions=positions)
    )
    return equity.value


def _heartbeat_stage(deps: PaperTickDeps, now_epoch_s: int) -> int:
    """Ledger a pipeline heartbeat and return the instant it attests to.

    Called after the snapshot stage, so the heartbeat is stamped only once the
    tick has proven the pipeline genuinely running -- an attestation rather than
    a constant. It is deliberately NOT stamped inside the approval context: a
    heartbeat equal to the approval's own ``now`` could never be stale, which
    would make ``pipeline_heartbeat_ok`` unfalsifiable and therefore worse than
    the ``None`` it replaces.

    Args:
        deps: The tick's dependency bundle.
        now_epoch_s: The tick's clock reading.

    Returns:
        The epoch second the pipeline was observed alive.
    """
    deps.store.append(
        PipelineHeartbeatRecorded(component=_COMPONENT, heartbeat_epoch_s=now_epoch_s)
    )
    return now_epoch_s


def _decide_and_approve(
    deps: PaperTickDeps,
    order_book: OrderBookSnapshot,
    forecast: ForecastRecord | None,
    created_at: datetime,
    heartbeat_epoch_s: int,
) -> tuple[str, int, int]:
    """Select and approve against a forecast, or short-circuit a halted tick.

    Narrows the optional forecast in one place so the select and approve stages
    keep their existing non-optional contracts.

    Args:
        deps: The tick's dependency bundle.
        order_book: The current book snapshot the selector reads.
        forecast: The tick's forecast, or ``None`` when research halted.
        created_at: The injected creation instant, for determinism.
        heartbeat_epoch_s: The instant an earlier stage observed the pipeline
            alive, threaded through to the approval context.

    Returns:
        A ``(forecast_id, intent_count, filled_centis)`` triple -- ``("", 0, 0)``
        when research halted, since no forecast exists to select against.
    """
    if forecast is None:
        return "", 0, 0
    decision = _select_stage(deps, order_book, forecast, created_at)
    filled = _approve_stage(deps, decision, heartbeat_epoch_s)
    return forecast.forecast_id, len(decision.intents), filled


def run_single_tick(deps: PaperTickDeps, *, beat: int) -> TickOutcome:
    """Drive one PAPER tick end to end, ledgering every stage (SPEC S5.3).

    The tick follows the SINGLE order path -- snapshot -> forecast -> select ->
    approve(seam) -> (only if a token minted) route -> fill -> reconcile -- then
    emits the per-tick heartbeat, equity sample, and positions snapshot, and
    writes this ISO-week's report -- folding the real ledger through
    :func:`windbreak.scheduler.weekly_data.weekly_report_body` so the report
    carries genuine evaluation and cost-meter data (issue #188), built lazily so
    the fold is paid for only on the genuine per-week write. Every stage appends
    an audit event to the shared hash-chained ledger. With the real kernel the
    approval still vetoes every intent -- on the honest ``None`` verification,
    exchange status, and pipeline heartbeat this loop supplies, not on a stub --
    so no order routes and ``filled_centis`` is ``0``.

    A tick whose per-forecast or per-UTC-day research budget is exhausted halts
    fail-closed (issue #339): it ledgers one ``ResearchBudgetHalted`` row, skips
    the forecast, select, and approve stages, and still emits its heartbeat,
    equity sample, positions snapshot, and weekly report -- so the loop stays
    observably alive and flat rather than dying on an uncaught budget error.
    Its ledger therefore differs from a normal tick's by exactly two absent
    rows: ``ForecastCreated`` and ``SelectorDecisionRecorded``.

    Args:
        deps: The fully wired dependency bundle.
        beat: The 1-based tick sequence number, stamped on the heartbeat.

    Returns:
        A :class:`TickOutcome` summarizing the tick.
    """
    now_epoch_s = deps.clock()
    created_at = datetime.fromtimestamp(now_epoch_s, UTC)
    order_book = _snapshot_stage(deps)
    heartbeat_epoch_s = _heartbeat_stage(deps, now_epoch_s)
    forecast = _forecast_stage(deps, order_book, created_at)
    forecast_id, intent_count, filled = _decide_and_approve(
        deps, order_book, forecast, created_at, heartbeat_epoch_s
    )
    deps.store.append(
        ModeHeartbeat(component=_COMPONENT, mode=Mode.PAPER.name, beat=beat)
    )
    equity = _equity_and_positions_stage(deps, now_epoch_s)
    report_date = created_at.date()
    maybe_write_weekly(
        deps.report_dir,
        today=report_date,
        body=lambda: weekly_report_body(deps.store.read_all(), today=report_date),
    )
    return TickOutcome(
        beat=beat,
        forecast_id=forecast_id,
        intent_count=intent_count,
        filled_centis=filled,
        equity_micros=equity,
        research_halted=forecast is None,
    )
