"""The single always-on PAPER-mode tick composition (issue #48, SPEC S5.3).

This module is the PAPER loop's one composition root. :func:`build_paper_deps`
wires the real, unmodified Market Connector (a `PaperExchange`), Forecast Engine,
Trade Selector, Risk Kernel, Order Gateway, and Reconciler over a single
hash-chained :class:`~windbreak.ledger.store.SqliteLedgerStore`, and
:func:`run_single_tick` screens the venue's market universe and then drives one
SPEC S5.3 SINGLE order-path pass through them *per screened market*:

    screen -> [ snapshot -> forecast -> select -> approve(seam) ->
    (only if a token minted) route -> PaperExchange fill -> reconcile ]

appending one audit event to the ledger at every stage, plus a per-tick
``ModeHeartbeat``, an ``EquitySampled``, and -- whenever the connector can
describe the account at all -- a ``PositionsSnapshotRecorded``
(:func:`_equity_and_positions_stage` explains the one case that omits it, and
why an omitted row is safer there than a written one).

Issue #345 supplied that leading screen. Before it the loop forecast
``next(iter(exchange.markets))``: one arbitrary market, chosen once at
composition time from a mapping's iteration order, and never screened -- so
nothing in the loop ever established that the market it traded was tradeable.
:mod:`windbreak.scheduler.screening` now walks the universe each tick, in an
explicit ticker sort so determinism does not depend on a fixture's key order,
and bounds the survivors at ``config.screener.max_candidates_per_tick``. The
bound is a *money* guard first and a wall-clock guard second: since issue #399
each ensemble vote books real spend, so candidates multiply the bill. Screening
itself is free of model calls, which is the only reason it can run over the
whole universe every tick.

The approval seam is the load-bearing safety boundary: :class:`KernelApproval`
composes the *real* ``RiskKernel.evaluate_intent`` with the *real*
``ApprovalPipeline.approve``. Every SPEC S10.3 check is now real (issue #340
promoted the last one), the loop observes real exchange status and stamps a
real pipeline heartbeat each tick (issue #342), and -- since issue #353 -- it
runs a real read-only verification cycle each tick and threads that snapshot
into the approval context, so the three SPEC S10.3 reconciliation checks
evaluate real evidence and pass on a clean cycle.

Issue #364 closed the last two unconditional vetoes the same way -- by
supplying evidence the loop already holds, never a convenient number:

* ``daily_loss_limit`` now measures against the current UTC day's *first*
  ledgered ``EquitySampled`` row (:func:`read_start_of_day_equity_micros`),
  which is a genuine start-of-day baseline. Until the day has a sample --
  including the day's first approval, since a tick samples equity only after
  approving -- the baseline is absent, lands on the account as zero, and the
  check keeps vetoing. That read is bounded to the samples taken since the day
  boundary (issue #370), because an always-on loop asks it every tick against a
  ledger that never stops growing.
* ``participation_cap_compliance`` now measures against the shallower visible
  side of the book the tick snapshotted (:func:`visible_depth_centis`). A book
  that cannot be read at all stays ``None`` and the check keeps vetoing.

Both figures loosen a real exposure limit when they are wrong -- a larger
baseline raises the loss threshold, a deeper book raises the participation
ceiling -- so absence is never rounded to a permissive default here. That is
the same discipline issues #340/#342/#353 applied: supply *genuine* evidence,
or keep failing closed.

What still vetoes a given intent is therefore a question about that intent and
the market it targets, not about missing feeds; the stock fixtures' surviving
reasons are measured, reason by reason, in
``tests/integration/test_paper_verification.py``.

Verification is also the loop's one HALT path. The baseline the cycle
reconciles against is frozen at startup from the venue's own opening state
(``LedgerExpectationSource``; see :func:`_build_verifier` for why freezing it
is what makes the comparison falsifiable at all), and the ledger carries no
fill amounts that could update it. So the first tick after a fill grades a
``BREACH``, the kernel transitions to ``HALT``, and every later approval vetoes
on the halted mode. That is the honest fail-closed reading of "our books cannot
account for the venue" -- and it is the reason an always-on PAPER deployment
must watch :attr:`TickOutcome.kernel_halted`.

Money and equity fields are scaled integers (micros/centis/pips), never floats
(SPEC S6.1); this package is on ``scripts/lint_no_floats.py``'s denylist.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

from windbreak.alerts.dispatch import AlertDispatcher, LoggingLedgerWriter
from windbreak.config import config_hash
from windbreak.connector.freshness import is_fresh
from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.paper import (
    COMPLEMENT_PIPS,
    LiveBookPaperExchange,
    PaperExchange,
    ReplayExhaustedError,
    TwoSidedPositionError,
)
from windbreak.connector.readonly import ReadOnlyConnectorView, ReadOnlyVenueView
from windbreak.evaluation.track_records import (
    PROVIDER_TRACK_RECORD_FILENAME as _PROVIDER_TRACK_RECORD_FILENAME,
)
from windbreak.forecast.budget import (
    BUDGET_DAY_EXHAUSTED_EVENT,
    BUDGET_FORECAST_EXCEEDED_EVENT,
    BUDGET_SPEND_RECORDED_EVENT,
    DailyBudgetExhaustedError,
    PerForecastBudgetExceededError,
    ResearchBudget,
)
from windbreak.forecast.cassettes import ReplayCassette
from windbreak.forecast.pipeline import (
    PROVIDER_GATE_HELD_EVENT,
    PROVIDER_VOTE_COSTED_EVENT,
    InMemoryForecastLedger,
    ledger_safe_ticker,
    run_pipeline,
)
from windbreak.forecast.providers.base import ProviderMarketMetadataRejectedError
from windbreak.forecast.providers.track_record import (
    InMemoryTrackRecordSource,
    ProviderTrackRecordGate,
    parse_track_records,
)
from windbreak.forecast.records import BaselineQuoteSnapshot
from windbreak.ledger.events import (
    EquitySampled,
    ExchangeStatusObserved,
    ForecastCreated,
    MarketSnapshotRecorded,
    ModeHeartbeat,
    PipelineHeartbeatRecorded,
    PositionsSnapshotRecorded,
    ProviderGateHeld,
    ProviderVoteRecorded,
    ResearchBudgetHalted,
    SelectorDecisionRecorded,
)
from windbreak.ledger.store import (
    LedgerStore,
    ReverseTypeScan,
    SqliteLedgerStore,
    events_from_records,
)
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips
from windbreak.order_gateway.cancel_all import VenueCancelAllSink
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
from windbreak.riskkernel.kill import (
    DirectiveSink,
    KillFileWatcher,
    KillIntegration,
    KillSwitch,
    ReconciliationMismatchMonitor,
)
from windbreak.riskkernel.modes import Mode, ModeStateMachine
from windbreak.riskkernel.process import RiskKernel
from windbreak.riskkernel.reservations import (
    ApprovalOutcome,
    ApprovalPipeline,
    ReservationLedger,
)
from windbreak.riskkernel.tokens import TokenIssuer
from windbreak.riskkernel.verification import (
    LedgerExpectationSource,
    ReadOnlyVerifier,
    VerificationTolerances,
)
from windbreak.scheduler.eligibility import (
    project_exchange_status,
    project_jurisdiction,
    project_product_type,
)
from windbreak.scheduler.exposure import (
    ExposureProjection,
    HeldMarket,
    project_exposure,
)
from windbreak.scheduler.fill_accounting import (
    LedgerFillAccountingFeed,
    LedgerFillBookkeeper,
)
from windbreak.scheduler.provider_wiring import (
    ProviderFactory,
    build_corpus_research_tools,
    build_corpus_vote_transport,
    build_live_llm_transport,
    build_live_research_tools,
    build_provider_factory,
    is_live_mode,
    load_corpus,
    offline_research_tools,
    replay_corpus_directory,
    replay_corpus_source,
)
from windbreak.scheduler.research_spend import (
    ResearchSpendRecorded,
    effective_per_day_micros,
    spend_by_day_from_records,
)
from windbreak.scheduler.screening import (
    ScreenLedgerWriter,
    require_candidate_bound,
    screen_universe,
)
from windbreak.scheduler.weekly_data import weekly_report_body
from windbreak.screener import Screener
from windbreak.selector import select
from windbreak.selector.types import (
    FeeModelInput,
    PositionReadModelInput,
    RiskConfigInput,
    SelectorDecision,
    SelectorInputs,
    SlippageModelInput,
)
from windbreak.timekeeping import require_aware

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from datetime import date

    from windbreak.config.schema import WindbreakConfig
    from windbreak.connector.live import MarketDataSource
    from windbreak.connector.models import (
        NormalizedMarket,
        OrderBookSnapshot,
        Position,
    )
    from windbreak.connector.snapshot import MarketScreener
    from windbreak.forecast.budget import BudgetEvent
    from windbreak.forecast.cassettes import LlmTransport
    from windbreak.forecast.corpus import ReplayCorpus
    from windbreak.forecast.records import ForecastRecord
    from windbreak.forecast.sandbox import ResearchTools
    from windbreak.ledger.events import Event
    from windbreak.ledger.store import LedgerRecord
    from windbreak.riskkernel.checks import Decision, OrderIntent
    from windbreak.riskkernel.verification import VerificationSnapshot
    from windbreak.scheduler.provider_wiring import LiveProviderHttp
    from windbreak.scheduler.screening import MarketCandidate
    from windbreak.tokens.verify import SignedApprovalToken

#: The scheduler's structured logger, installed by ``windbreak run``'s
#: :func:`windbreak.logging_setup.configure_logging`.
_LOGGER = logging.getLogger("windbreak.scheduler")

#: The component label stamped on every scheduler-authored ledger event.
_COMPONENT = "scheduler"

#: Evidence source token for a research bundle the caller handed in. Not
#: reachable from ``windbreak run`` -- ``build_paper_deps``'s ``research_tools``
#: parameter is a test seam -- and deliberately *not* classified: this guard
#: cannot know whether an injected bundle finds anything.
RESEARCH_EVIDENCE_INJECTED: Final = "injected"

#: Evidence source token for the committed replay corpus (issue #510), the one
#: source a shipped command line can select without a network.
RESEARCH_EVIDENCE_CORPUS: Final = "replay-corpus"

#: Evidence source token for the live search/fetch transports (issue #344).
RESEARCH_EVIDENCE_LIVE: Final = "live-search"

#: Evidence source token for the offline bundle whose search finds nothing by
#: construction. A run wired to it must abstain on ``no_verified_citations``
#: before a single vote, every tick, forever.
RESEARCH_EVIDENCE_NONE: Final = "none"

#: Every evidence source token, in the order :func:`_resolve_research_tools`
#: decides them. Enumerated here so a caller reasons over the set rather than
#: restating it.
RESEARCH_EVIDENCE_SOURCES: Final = (
    RESEARCH_EVIDENCE_INJECTED,
    RESEARCH_EVIDENCE_CORPUS,
    RESEARCH_EVIDENCE_LIVE,
    RESEARCH_EVIDENCE_NONE,
)

#: What an operator is told, once at startup, when this deployment composed no
#: source of research evidence at all (issue #485).
#:
#: It is a constant with **no interpolation site**: the guard reports
#: configuration, and every leaf that can select an evidence source sits beside
#: one naming a credential's environment variable, so a message assembled from
#: values would be one refactor away from carrying a secret into a log
#: aggregator and, via ``AlertEmitted``, into the append-only ledger.
EVIDENCE_STARVED_MESSAGE: Final = (
    "DEGRADED: this PAPER loop composed no research evidence source, so every "
    "forecast it makes must abstain on no_verified_citations before a single "
    "vote and it can never emit an order intent -- it will keep beating, and "
    "keep reporting healthy, for as long as it runs. "
    "REMEDY: either replay a committed corpus (set forecast.replay_corpus.mode "
    "to 'replay' and forecast.replay_corpus.corpus_dir to that directory), or "
    "configure live research (set forecast.provider_transport.mode to 'live' "
    "and forecast.research.search_endpoint_url to your search endpoint). "
    "WHAT THIS DOES NOT CLAIM: it reads the research wiring this process "
    "actually composed and nothing else. Its implication runs one way -- no "
    "evidence source means no intent, ever -- and it proves nothing about a "
    "deployment that has one, because the depth floor, the resolution horizon, "
    "the correlation declaration and the provider track record are judged per "
    "tick against the books and this guard reads none of them."
)

#: The per-UTC-day research ceiling a budget opens with when this ledger's
#: research rows cannot be folded. Zero, so every market halts on the budget
#: and no research money is spent, while the loop itself keeps beating. See
#: :func:`_research_ledger_state`.
_UNREADABLE_LEDGER_CEILING_MICROS: Final = 0

#: The calibration-map version tag echoed into every selector decision.
_CALIBRATION_MAP_VERSION = "v0"

#: ``halt_kind`` stamped on a halt caused by the UTC day's budget running out.
_HALT_KIND_PER_DAY = "per_day"

#: ``halt_kind`` stamped on a halt caused by one forecast breaching its ceiling.
_HALT_KIND_PER_FORECAST = "per_forecast"

#: Default max admissible forecast age, in seconds, for the risk limits mapped
#: from config (``RiskConfig`` carries a quote ttl but no forecast ttl).
_DEFAULT_FORECAST_TTL_SECONDS = 3600

#: Default max admissible verification-snapshot age, in seconds. The PAPER loop
#: runs a real cycle every tick (issue #353), so this genuinely bounds how long
#: a stalled reconciliation may go unnoticed; a conservative one-hour default
#: suffices for a loop that re-verifies every tick.
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

#: The ledger event type carrying one PAPER-loop equity sample. The tick
#: appends one every beat, and :func:`read_start_of_day_equity_micros` reads the
#: day's first one back as the ``daily_loss_limit`` baseline (issue #364). It is
#: also the single type the bounded reverse walk is filtered to (issue #370).
_EQUITY_SAMPLED_EVENT_TYPE = "EquitySampled"

#: Envelope key under which a ledgered event's typed payload is nested (mirrors
#: :data:`windbreak.scheduler.weekly_data._PAYLOAD_DATA_KEY` and
#: :func:`windbreak.ledger.rebuild._gateway_projection`'s own ``["data"]``).
_PAYLOAD_DATA_KEY = "data"

#: The ``EquitySampled`` payload key carrying the sampled equity, in micros.
_EQUITY_MICROS_KEY = "equity_micros"

#: The ``EquitySampled`` payload key carrying the sample instant, in epoch
#: seconds -- the field the UTC-day bucketing reads, never the row's own
#: ``created_at`` wall clock (which the injected clock does not control).
_SAMPLE_EPOCH_KEY = "epoch_s"

#: The ledger event type carrying one booked venue fill. The UTC-day fold behind
#: ``velocity_limits``' daily-notional cap reads these rows
#: (:func:`notional_today_micros`, issue #415), and it is the single type that
#: fold's reverse walk is filtered to.
_FILL_ACCOUNTED_EVENT_TYPE = "FillAccounted"

#: The ``FillAccounted`` payload key carrying the fill's signed available-cash
#: movement, in micros. Its *magnitude* is the notional that fill routed.
_CASH_DELTA_MICROS_KEY = "cash_delta_micros"

#: The ledger event type carrying one Order Gateway lifecycle transition. The
#: trailing-hour fold behind ``velocity_limits``' hourly order cap reads these
#: rows (:func:`orders_last_hour_count`, issue #491), and it is the single type
#: that fold's reverse walk is filtered to.
_ORDER_TRANSITION_EVENT_TYPE = "OrderTransitionLedgered"

#: The ``OrderTransitionLedgered`` payload key naming the lifecycle event that
#: drove the transition.
_TRANSITION_EVENT_KEY = "event"

#: The single lifecycle event that marks one order being routed at the venue.
#: :meth:`~windbreak.order_gateway.gateway.OrderGateway._submit_new` records it
#: immediately *before* it calls the submitter -- the write-before-next-action
#: discipline :func:`~windbreak.order_gateway.ledger_writer.apply_and_ledger`
#: enforces -- so every venue touch is preceded by exactly one such row, and a
#: submission the exchange then rejected still counts against the cap. Counting
#: the later ``SUBMIT`` instead would undercount precisely the runaway case:
#: orders flung at a venue that is failing to answer.
_REQUEST_SUBMISSION_EVENT = "REQUEST_SUBMISSION"

#: The width of ``velocity_limits``' hourly window, in whole seconds.
_TRAILING_HOUR_SECONDS = 3600

#: The :class:`~windbreak.ledger.store.LedgerRecord` field naming the recorded
#: instant, surfaced in the timezone-awareness refusal it is read through.
_CREATED_AT_FIELD = "created_at"

#: The M6 per-provider track-record artifact the loop's live-eligibility gate
#: reads, resolved inside the same ``report_dir`` every other evaluation
#: artifact is written to (issue #305). Re-exported from the module that
#: *writes* it (:mod:`windbreak.evaluation.track_records`, issue #440) rather
#: than declared here, so the producer and this reader can never name two
#: different files. Public because it is an operator- and test-facing
#: convention: the file ``windbreak evaluate-providers`` writes for a provider
#: to become live-eligible. Absent, the loop bootstraps fail-closed (every
#: provider unproven); malformed, it refuses to start.
PROVIDER_TRACK_RECORD_FILENAME: Final = _PROVIDER_TRACK_RECORD_FILENAME


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
    every successful charge maps onto a typed ``ResearchSpendRecorded`` row, and
    any other kind raises rather than silently dropping an audit row.

    Those ``ResearchSpendRecorded`` rows are what makes the daily ceiling
    survive a restart (issue #442): :func:`_build_research_budget` folds them
    back into the day counter the next process opens with. Before them the
    ledger recorded only breaches, so a restarted loop had nothing to read and
    began every day at zero however much the day had already cost.
    """

    def __init__(self, store: SqliteLedgerStore) -> None:
        """Bind the writer to a ledger store.

        Args:
            store: The append-only store every halt event is persisted to.
        """
        self._store = store

    def record(self, event: BudgetEvent) -> None:
        """Append a budget event to the ledger as a typed row.

        Payload keys are read by subscript, never ``.get`` with a default, so a
        payload-shape drift surfaces as a loud ``KeyError`` rather than a
        silently zeroed audit row. Note the three kinds name the spent amount
        differently: the per-day breach payload carries ``spent_micros`` (the
        day's cumulative spend) while the per-forecast breach and the successful
        charge both carry ``cost_micros`` (that forecast's, or that charge's,
        own cost).

        Args:
            event: The budget event to persist.

        Raises:
            ValueError: If ``event`` is none of the three known kinds.
        """
        payload = event.payload
        if event.event_type == BUDGET_SPEND_RECORDED_EVENT:
            self._store.append(
                ResearchSpendRecorded(
                    component=_COMPONENT,
                    utc_day=cast("str", payload["utc_day"]),
                    market_ticker=cast("str", payload["market_ticker"]),
                    cost_micros=cast("int", payload["cost_micros"]),
                )
            )
            return
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

    The ticker crosses through
    :func:`~windbreak.forecast.pipeline.ledger_safe_ticker` (issue #530).
    :func:`_snapshot_stage` appends this row for every screened candidate,
    *before* :func:`_forecast_stage` runs -- so the entry screen issue #525
    added to ``run_pipeline`` refuses the market too late to keep its bytes off
    an append-only chain. Substituting here, at the sole place this event is
    constructed, is what closes that window.

    Nothing reads this ticker back, so the digest costs no downstream truth;
    it is the same stable digest the market's ``ScreenDecisionRecorded`` row
    carries, so the two still correlate.

    Args:
        ticker: The market the snapshot is for. Ledgered as a digest when it
            fails the S8.5 screen, so the returned record's ``ticker`` and its
            payload agree and neither carries the raw bytes.
        order_book: The book snapshot to project.
        component: The component label stamped on the event.

    Returns:
        The assembled :class:`~windbreak.ledger.events.MarketSnapshotRecorded`.
    """
    return MarketSnapshotRecorded(
        component=component,
        ticker=ledger_safe_ticker(ticker),
        best_bid_pips=_best_bid_pips(order_book),
        best_ask_pips=_best_ask_pips(order_book),
        fetched_at_epoch_s=int(order_book.fetched_at.timestamp()),
    )


def _utc_day(epoch_s: int) -> date:
    """Return the UTC calendar day an epoch second falls on.

    Args:
        epoch_s: The instant to bucket, in whole epoch seconds.

    Returns:
        The UTC date containing ``epoch_s``.
    """
    return datetime.fromtimestamp(epoch_s, UTC).date()


@dataclass(frozen=True, slots=True)
class DayEquity:
    """The two points of the current UTC day's equity series the caps read.

    Folded together, in one walk, on purpose: ``daily_loss_limit`` measures the
    day's loss *from* the day's baseline and compares it against a ppm share
    *of that same baseline*, so a fix that folded the two separately could let
    a clock anomaly move one without the other and report a loss against an
    equity the account never opened at.

    Attributes:
        baseline: The day's earliest sampled equity -- the reference
            ``daily_loss_limit``'s threshold is a share of (issue #364).
        trough: The day's lowest sampled equity. The loss is measured to the
            trough rather than to the latest sample because SPEC S10.10 pauses
            a daily-loss breach *to the next UTC day*: a loss that un-booked
            itself the moment the mark recovered would let a volatile day cross
            the limit repeatedly and trade on every rebound (issue #513).
    """

    baseline: MoneyMicros
    trough: MoneyMicros

    @property
    def realized_loss(self) -> MoneyMicros:
        """Return the day's realized loss, in micros.

        Never negative: the trough is a minimum taken over a set that includes
        the baseline, so ``baseline >= trough`` holds by construction and no
        clamp is needed (a clamp here would be a branch production cannot
        reach).

        In this system the difference is *realized* rather than merely marked:
        :func:`_position_value_micros` marks every holding at its own average
        entry price, so opening a position moves cash and position value by
        equal and opposite amounts and leaves equity unchanged but for the fee.
        The equity curve therefore only steps down when something is actually
        booked -- a fee, a fill closing below cost, a settlement.

        Returns:
            ``baseline - trough``, in micros.
        """
        return MoneyMicros(self.baseline.value - self.trough.value)


@dataclass(frozen=True, slots=True)
class EquityCurve:
    """The account's peak and current equity, folded from the whole series.

    Both are points on the *same* series -- the ledger's own ``EquitySampled``
    rows -- which is what makes their difference a drawdown at all (issue
    #514). See :class:`~windbreak.riskkernel.context.AccountState` for why
    worst-case equity is not the comparand.

    Attributes:
        high_water_mark: The highest equity ever sampled on this ledger. The
            mark is all-time and ratchets: a "trailing" drawdown trails a peak
            that never resets, so a mark rebuilt from one day's samples would
            forgive every drawdown at UTC midnight.
        latest: The newest sampled equity, taken by ``sequence_number`` -- the
            row the loop appended last, not the highest stamp. Append order is
            the ledger's own chained order and no clock can reorder it.
    """

    high_water_mark: MoneyMicros
    latest: MoneyMicros


def _day_samples(records: Iterable[LedgerRecord]) -> Iterator[tuple[int, int]]:
    """Yield each ``EquitySampled`` row's ``(epoch_s, equity_micros)`` pair.

    Args:
        records: The ledger read, in any order. Rows of other types are
            ignored.

    Yields:
        One ``(epoch_s, equity_micros)`` pair per sample.
    """
    for record in records:
        if record.event_type == _EQUITY_SAMPLED_EVENT_TYPE:
            yield _equity_sample(record)


def _samples_back_to_the_day_boundary(
    store: ReverseTypeScan, today: date
) -> Iterator[tuple[int, int]]:
    """Yield today's samples from a newest-first walk, stopping at the boundary.

    Walks ``EquitySampled`` rows alone, newest first, and stops at the first
    one stamped on an *earlier* UTC day: everything beyond it is older still,
    so nothing there can belong to today. The cost is therefore O(samples taken
    today) -- bounded by ticks-per-day -- rather than O(ledger), with no new
    index and no change to what the answer means (issue #370).

    A sample stamped on a *later* day -- a forward clock blip -- is yielded
    rather than treated as the boundary; the fold discards it by day like any
    other non-today row. Stopping there would discard today's genuine baseline
    for as long as the blip sat at the head of the ledger.

    Args:
        store: The ledger, declaring the reverse-walk capability.
        today: The UTC calendar day the walk is bounded to.

    Yields:
        One ``(epoch_s, equity_micros)`` pair per row down to the boundary.
    """
    for record in store.iter_records_of_type_reversed(_EQUITY_SAMPLED_EVENT_TYPE):
        epoch_s, equity_micros = _equity_sample(record)
        if _utc_day(epoch_s) < today:
            return
        yield epoch_s, equity_micros


def _fold_day_equity(
    samples: Iterable[tuple[int, int]], today: date
) -> DayEquity | None:
    """Fold ``samples`` into the day's baseline and trough, or ``None``.

    The shared body of both read paths, so the bounded walk and the
    whole-ledger fold can never answer differently.

    The earliest sample is chosen by comparing stamped ``epoch_s`` values
    rather than by taking the first row the iteration surfaces, so neither
    append order nor a clock that steps backwards mid-day can promote a later
    sample to the baseline.

    Args:
        samples: The ``(epoch_s, equity_micros)`` pairs to fold, in any order.
        today: The UTC calendar day being measured.

    Returns:
        The day's :class:`DayEquity`, or ``None`` when the day carries no
        sample at all.
    """
    state: tuple[int, int, int] | None = None
    for epoch_s, equity_micros in samples:
        if _utc_day(epoch_s) != today:
            continue
        if state is None:
            state = (epoch_s, equity_micros, equity_micros)
            continue
        earliest_epoch_s, baseline, trough = state
        if epoch_s < earliest_epoch_s:
            earliest_epoch_s, baseline = epoch_s, equity_micros
        state = (earliest_epoch_s, baseline, min(trough, equity_micros))
    if state is None:
        return None
    return DayEquity(baseline=MoneyMicros(state[1]), trough=MoneyMicros(state[2]))


def day_equity_micros(
    records: Iterable[LedgerRecord], *, now_epoch_s: int
) -> DayEquity | None:
    """Fold the current UTC day's equity samples out of a whole ledger read.

    The reference definition of the answer both read paths must produce, and
    the fallback for any store without the bounded reverse walk.

    Samples are matched by their own ``epoch_s`` payload field rather than by
    the row's ``created_at`` wall clock, because only the payload is stamped
    from the loop's injected clock. That also means neither figure here has an
    offsetless-timestamp failure mode: unlike ``FillAccounted`` (issue #415)
    and ``OrderTransitionLedgered`` (issue #491), ``EquitySampled`` carries its
    own instant, so there is no host-dependent stamp to refuse.

    An absent answer is deliberately ``None`` and never a zero: the caller maps
    it onto the fail-closed account (see :func:`_account_from_verification`),
    and a numeric default here would be indistinguishable from a genuine
    reading.

    Args:
        records: The ledger read (``SqliteLedgerStore.read_all()``), in append
            order. Non-``EquitySampled`` rows are ignored.
        now_epoch_s: The instant whose UTC day is the "current" one.

    Returns:
        The day's :class:`DayEquity`, or ``None`` when the day carries no
        sample at all.

    Raises:
        KeyError: If an ``EquitySampled`` payload is missing either field this
            reads -- a loud shape drift, never a silently zeroed baseline.
    """
    return _fold_day_equity(_day_samples(records), _utc_day(now_epoch_s))


def read_day_equity(store: LedgerStore, *, now_epoch_s: int) -> DayEquity | None:
    """Read the current UTC day's baseline and trough equity from ``store``.

    The tick's entry point, and the reason it is not simply
    ``day_equity_micros(store.read_all(), ...)``: ``_approve_stage`` asks this
    on *every* tick of a loop built to run for weeks, while the ledger only
    grows and every tick appends to it, so a full fold would cost more each
    beat without bound (issue #370). A store declaring the optional
    :class:`~windbreak.ledger.store.ReverseTypeScan` capability answers it with
    a walk bounded at the day boundary instead; any other store, including
    every hand-rolled :class:`~windbreak.ledger.store.LedgerStore` double,
    falls back to the whole-ledger fold. The two paths return the same answer,
    so the dispatch is a pure optimization and never a behavioral fork.

    Args:
        store: The ledger to read the day's samples out of.
        now_epoch_s: The instant whose UTC day is the "current" one.

    Returns:
        The day's :class:`DayEquity`, or ``None`` when the day carries no
        sample at all -- which keeps ``daily_loss_limit`` vetoing.
    """
    if isinstance(store, ReverseTypeScan):
        today = _utc_day(now_epoch_s)
        return _fold_day_equity(_samples_back_to_the_day_boundary(store, today), today)
    return day_equity_micros(store.read_all(), now_epoch_s=now_epoch_s)


def _curve_samples(records: Iterable[LedgerRecord]) -> Iterator[tuple[int, int]]:
    """Yield each sample's ``(sequence_number, equity_micros)`` pair.

    Args:
        records: The ledger read, in any order. Rows of other types are
            ignored.

    Yields:
        One ``(sequence_number, equity_micros)`` pair per sample.
    """
    for record in records:
        if record.event_type == _EQUITY_SAMPLED_EVENT_TYPE:
            _, equity_micros = _equity_sample(record)
            yield record.sequence_number, equity_micros


def _fold_equity_curve(samples: Iterable[tuple[int, int]]) -> EquityCurve | None:
    """Fold ``samples`` into the all-time peak and the newest reading.

    Order-free: the peak is a maximum and the newest reading is selected by
    ``sequence_number``, so the newest-first walk and the oldest-first fold
    reach the same pair.

    Args:
        samples: The ``(sequence_number, equity_micros)`` pairs to fold, in any
            order.

    Returns:
        The :class:`EquityCurve`, or ``None`` when no sample exists at all.
    """
    state: tuple[int, int, int] | None = None
    for sequence_number, equity_micros in samples:
        if state is None:
            state = (equity_micros, sequence_number, equity_micros)
            continue
        peak, latest_sequence_number, latest = state
        if sequence_number > latest_sequence_number:
            latest_sequence_number, latest = sequence_number, equity_micros
        state = (max(peak, equity_micros), latest_sequence_number, latest)
    if state is None:
        return None
    return EquityCurve(
        high_water_mark=MoneyMicros(state[0]), latest=MoneyMicros(state[2])
    )


def equity_curve_micros(records: Iterable[LedgerRecord]) -> EquityCurve | None:
    """Fold the whole ledger's equity samples into a peak and a latest reading.

    The reference definition of the answer both read paths must produce, and
    the fallback for any store without the reverse walk.

    Args:
        records: The ledger read (``SqliteLedgerStore.read_all()``), in any
            order. Non-``EquitySampled`` rows are ignored.

    Returns:
        The :class:`EquityCurve`, or ``None`` when the ledger holds no sample
        at all -- which keeps ``trailing_drawdown_limit`` vetoing.

    Raises:
        KeyError: If an ``EquitySampled`` payload is missing the field this
            reads -- a loud shape drift, never a silently zeroed mark.
    """
    return _fold_equity_curve(_curve_samples(records))


def read_equity_curve(store: LedgerStore) -> EquityCurve | None:
    """Read the account's peak and current equity from ``store``.

    The tick's entry point for ``trailing_drawdown_limit``'s two terms, and the
    one fold in this module that deliberately does **not** stop at a boundary:
    a maximum over an all-time series has no recency predicate to stop on, and
    a mark that stopped at one would not be a high-water mark. A store
    declaring :class:`~windbreak.ledger.store.ReverseTypeScan` therefore pays
    one indexed pass over the ``EquitySampled`` rows alone -- the same
    complexity :func:`read_notional_today_micros` already pays over the booked
    fills, for the same reason (a sum, like a maximum, cannot be truncated
    safely) -- and every other store falls back to the whole-ledger fold.

    Re-folding from the ledger on each tick is also what makes the mark survive
    a process restart: it is never held in memory, so a loop that stops and
    starts recovers exactly the peak its own rows attest to, rather than
    resetting to whatever the first tick after the restart happens to sample.

    Args:
        store: The ledger to read the equity samples out of.

    Returns:
        The :class:`EquityCurve`, or ``None`` when the ledger holds no sample
        at all.
    """
    records: Iterable[LedgerRecord] = (
        store.iter_records_of_type_reversed(_EQUITY_SAMPLED_EVENT_TYPE)
        if isinstance(store, ReverseTypeScan)
        else store.read_all()
    )
    return _fold_equity_curve(_curve_samples(records))


def start_of_day_equity_micros(
    records: Iterable[LedgerRecord], *, now_epoch_s: int
) -> MoneyMicros | None:
    """Return the current UTC day's *first* ledgered equity sample, or ``None``.

    ``daily_loss_limit`` measures today's realized loss against where the day
    started, so the baseline has to be the earliest sample of the day and not
    the latest: reading the most recent one would quietly raise the loss
    threshold every time equity grew intraday, which is precisely the loosening
    a fabricated baseline would have caused (issue #364).

    Samples are matched by their own ``epoch_s`` payload field rather than by
    the row's ``created_at`` wall clock, because only the payload is stamped
    from the loop's injected clock; and the earliest is chosen by comparing
    those stamps rather than by taking the first matching row, so a clock that
    steps backwards mid-day cannot promote a later sample to the baseline.

    An absent answer is deliberately ``None`` and never a zero: the caller maps
    it onto the fail-closed account (see :func:`_account_from_verification`),
    and a numeric default here would be indistinguishable from a genuine
    reading. Before the day's first tick has ledgered its sample -- including
    the very first tick against a fresh ledger, since the sample is appended
    *after* the approval stage -- there simply is no baseline, and the check
    must keep vetoing.

    The whole record sequence is scanned rather than stopped at the first
    same-day hit, for the clock reason above: append order is not proof of
    chronological order.

    The baseline half of :func:`day_equity_micros`, which folds it in the same
    pass as the day's trough (issue #513) so the threshold and the loss
    measured against it can never come from different rows.

    Args:
        records: The ledger read (``SqliteLedgerStore.read_all()``), in append
            order. Non-``EquitySampled`` rows are ignored.
        now_epoch_s: The instant whose UTC day is the "current" one.

    Returns:
        The day's earliest sampled equity, in micros, or ``None`` when the day
        carries no sample at all.

    Raises:
        KeyError: If an ``EquitySampled`` payload is missing either field this
            reads -- a loud shape drift, never a silently zeroed baseline.
    """
    day = day_equity_micros(records, now_epoch_s=now_epoch_s)
    return day.baseline if day is not None else None


def realized_loss_today_micros(
    records: Iterable[LedgerRecord], *, now_epoch_s: int
) -> MoneyMicros | None:
    """Return the loss the current UTC day has realized, or ``None``.

    The realized-loss half of :func:`day_equity_micros`, folded in the same
    pass as the baseline it is measured from (issue #513), and the records-side
    sibling of :func:`read_realized_loss_today_micros`.

    Args:
        records: The ledger read (``SqliteLedgerStore.read_all()``), in append
            order. Non-``EquitySampled`` rows are ignored.
        now_epoch_s: The instant whose UTC day is the "current" one.

    Returns:
        The day's realized loss in micros, or ``None`` when the day carries no
        sample at all.

    Raises:
        KeyError: If an ``EquitySampled`` payload is missing either field this
            reads -- a loud shape drift, never a silently zeroed loss.
    """
    day = day_equity_micros(records, now_epoch_s=now_epoch_s)
    return day.realized_loss if day is not None else None


def _equity_sample(record: LedgerRecord) -> tuple[int, int]:
    """Return one ``EquitySampled`` row's ``(epoch_s, equity_micros)`` pair.

    Args:
        record: The ``EquitySampled`` record to read.

    Returns:
        The sample's stamped instant in epoch seconds and its equity in micros.

    Raises:
        KeyError: If the payload is missing either field -- a loud shape drift,
            never a silently zeroed baseline.
    """
    data = json.loads(record.payload_json)[_PAYLOAD_DATA_KEY]
    return int(data[_SAMPLE_EPOCH_KEY]), int(data[_EQUITY_MICROS_KEY])


def read_start_of_day_equity_micros(
    store: LedgerStore, *, now_epoch_s: int
) -> MoneyMicros | None:
    """Read the current UTC day's first ledgered equity sample from ``store``.

    The tick's baseline read, and the reason it is not simply
    ``start_of_day_equity_micros(store.read_all(), ...)``: ``_approve_stage``
    asks this on *every* tick of a loop built to run for weeks, while the ledger
    only grows and every tick appends to it, so a full fold would cost more each
    beat without bound (issue #370).

    A store declaring the optional
    :class:`~windbreak.ledger.store.ReverseTypeScan` capability answers it with a
    bounded walk instead -- see
    :func:`_samples_back_to_the_day_boundary` -- and any other store,
    including every hand-rolled :class:`~windbreak.ledger.store.LedgerStore`
    double, falls back to the original whole-ledger fold. The two paths return
    the same baseline, so the dispatch is a pure optimization and never a
    behavioral fork.

    Args:
        store: The ledger to read the day's samples out of.
        now_epoch_s: The instant whose UTC day is the "current" one.

    Returns:
        The day's earliest sampled equity, in micros, or ``None`` when the day
        carries no sample at all -- which keeps ``daily_loss_limit`` vetoing.
    """
    day = read_day_equity(store, now_epoch_s=now_epoch_s)
    return day.baseline if day is not None else None


def read_realized_loss_today_micros(
    store: LedgerStore, *, now_epoch_s: int
) -> MoneyMicros | None:
    """Read the loss the current UTC day has realized, from ``store``.

    The term ``daily_loss_limit`` was fed a hardcoded zero (issue #513), which
    made the cap unable to veto at any loss for any account whose day had a
    ledgered baseline -- the loop's normal steady state, since the check's
    threshold is a ppm share of that same baseline and ``0 >= threshold`` is
    false for every positive one.

    The loss is the distance from the day's baseline down to its trough, both
    read from the same walk (:func:`read_day_equity`), so the figure and the
    threshold it is measured against always describe one day's rows.

    Args:
        store: The ledger to read the day's samples out of.
        now_epoch_s: The instant whose UTC day is the "current" one.

    Returns:
        The day's realized loss in micros, or ``None`` when the day carries no
        sample at all -- the same unprovable case that leaves the baseline
        absent, and which keeps ``daily_loss_limit`` vetoing.
    """
    day = read_day_equity(store, now_epoch_s=now_epoch_s)
    return day.realized_loss if day is not None else None


def _fill_utc_day(record: LedgerRecord) -> date | None:
    """Return the UTC calendar day a booked fill was recorded on, or ``None``.

    Read from the row's own ``created_at`` rather than from any payload field,
    because ``FillAccounted`` carries no instant and the ``Event`` base carries
    none either -- the fact that made this cap look blocked. The column is not
    a soft one: :func:`~windbreak.ledger.store.compute_event_hash` folds
    ``created_at`` into the chain digest alongside the payload, so the stamp a
    risk cap buckets on carries exactly the tamper-evidence the payload does. A
    window folded over a rewritable timestamp would be worthless.

    An offsetless stamp yields ``None`` rather than a guessed day.
    :func:`~windbreak.timekeeping.require_aware` refuses rather than repairs,
    and the reason is this exact bucketing: read as the host's local time, a
    naive 20:00 stamp falls on the next calendar day west of UTC and on the
    current one at UTC, so "which day did this trade on" would answer
    differently per host and the cap would fail *open* west of UTC (PR #405).
    An unparseable stamp is refused on the same grounds.

    Args:
        record: The ``FillAccounted`` row to bucket.

    Returns:
        The UTC date the row was recorded on, or ``None`` when its stamp is
        unparseable or carries no UTC offset.
    """
    try:
        instant = datetime.fromisoformat(record.created_at)
        require_aware(instant, _CREATED_AT_FIELD)
    except ValueError:
        return None
    return instant.astimezone(UTC).date()


def _fill_notional_micros(record: LedgerRecord) -> int:
    """Return the notional one booked fill routed, in micros.

    The *magnitude* of the cash movement, never its sign: a buy consumes cash
    and a sell releases it, but both routed an order against the day's budget.
    Summing signed deltas would let a sale refund notional a purchase had
    spent, which is a cap that loosens the more the loop trades.

    Args:
        record: The ``FillAccounted`` row to read.

    Returns:
        The absolute cash movement the fill booked, in micros.

    Raises:
        KeyError: If the payload is missing the field this reads -- a loud
            shape drift, never a silently under-counted day.
    """
    data = json.loads(record.payload_json)[_PAYLOAD_DATA_KEY]
    return abs(int(data[_CASH_DELTA_MICROS_KEY]))


def _fold_notional_micros(
    fills: Iterable[LedgerRecord], today: date
) -> MoneyMicros | None:
    """Sum the notional booked on ``today`` across ``fills``.

    The shared body of both read paths below, so the indexed walk and the
    whole-ledger fold can never answer differently.

    A row whose day cannot be established abandons the whole answer rather than
    being skipped. A day is not partially provable: the unreadable row might be
    today's, so folding only the readable ones would report a total *smaller*
    than the evidence supports -- and under-reporting what a cap has already
    consumed is precisely the permissive direction.

    Args:
        fills: The ``FillAccounted`` rows to fold, in any order.
        today: The UTC calendar day being summed.

    Returns:
        The day's booked notional in micros, or ``None`` when any row's
        recorded instant could not be established.

    Raises:
        KeyError: If a payload is missing ``cash_delta_micros``.
    """
    total = 0
    for record in fills:
        day = _fill_utc_day(record)
        if day is None:
            return None
        if day == today:
            total += _fill_notional_micros(record)
    return MoneyMicros(total)


def notional_today_micros(
    records: Iterable[LedgerRecord], *, now_epoch_s: int
) -> MoneyMicros | None:
    """Return the notional booked so far on the current UTC day, or ``None``.

    ``velocity_limits`` caps the notional routed within the current UTC day, and
    the scheduler fed it a hardcoded zero -- permissive in exactly the way zero
    was permissive for ``concentration_limits``: the cap ran every tick,
    reported success, and could not bind however much the loop had traded
    (issue #415). This is the fold that feeds it.

    An empty day is a genuine ``MoneyMicros(0)`` and not ``None``, which is
    where this differs from :func:`start_of_day_equity_micros`. An unsampled day
    has no baseline to read, but a day with no booked fill has *provably* routed
    nothing: ``FillAccounted`` is written once at execution and never rewritten,
    so its absence is evidence rather than the lack of it. That makes the cap
    bind from the first tick against a fresh ledger instead of vetoing until
    something happens to be booked.

    ``None`` is reserved for the one case where the day genuinely cannot be
    established -- a row whose recorded instant carries no UTC offset (see
    :func:`_fill_utc_day`) -- and the caller must fail closed on it.

    Args:
        records: The ledger read (``SqliteLedgerStore.read_all()``). Rows of
            other types are ignored.
        now_epoch_s: The instant whose UTC day is the "current" one.

    Returns:
        The day's booked notional in micros, or ``None`` when it is unprovable.

    Raises:
        KeyError: If a ``FillAccounted`` payload is missing the field this
            reads -- a loud shape drift, never a silently under-counted day.
    """
    fills = (
        record for record in records if record.event_type == _FILL_ACCOUNTED_EVENT_TYPE
    )
    return _fold_notional_micros(fills, _utc_day(now_epoch_s))


def read_notional_today_micros(
    store: LedgerStore, *, now_epoch_s: int
) -> MoneyMicros | None:
    """Read the current UTC day's booked notional from ``store``.

    The tick's entry point, mirroring
    :func:`read_start_of_day_equity_micros`: a store declaring the optional
    :class:`~windbreak.ledger.store.ReverseTypeScan` capability is walked over
    its ``FillAccounted`` rows alone -- O(booked fills) rather than O(ledger),
    on the composite index that walk already has -- and every other store,
    including each hand-rolled double, falls back to the whole-ledger fold.

    Unlike the equity baseline's walk, this one deliberately does **not** stop
    at the day boundary. That early stop is safe when the answer is a single
    earliest row; it is not safe for a *sum*, because a clock that stepped
    backwards across midnight would put a previous-day stamp above a same-day
    one and truncate the total. A truncated total under-reports how much budget
    the day has spent, which fails open -- so the walk pays the full pass over
    the fills and keeps the two paths' answers identical.

    Args:
        store: The ledger to read the day's booked fills out of.
        now_epoch_s: The instant whose UTC day is the "current" one.

    Returns:
        The day's booked notional in micros, or ``None`` when it is unprovable
        -- which keeps the daily cap vetoing (see :func:`_build_limits`).
    """
    if isinstance(store, ReverseTypeScan):
        return _fold_notional_micros(
            store.iter_records_of_type_reversed(_FILL_ACCOUNTED_EVENT_TYPE),
            _utc_day(now_epoch_s),
        )
    return notional_today_micros(store.read_all(), now_epoch_s=now_epoch_s)


def _record_epoch_s(record: LedgerRecord) -> int | None:
    """Return the instant a row was recorded at, in epoch seconds, or ``None``.

    The sub-day sibling of :func:`_fill_utc_day`, and it reads the same column
    for the same reason: ``created_at`` is folded into the chain digest by
    :func:`~windbreak.ledger.store.compute_event_hash`, so a window measured on
    it carries the ledger's tamper-evidence rather than a rewritable field's.

    An offsetless stamp yields ``None`` rather than a guessed instant.
    :func:`~windbreak.timekeeping.require_aware` refuses rather than repairs,
    and here the stakes are the same shape as the day bucket's: read as the
    host's local time, a naive stamp lands five hours away from its true instant
    on a UTC-05:00 host, which slides rows into and out of the trailing hour
    per host. An unparseable stamp is refused on the same grounds.

    The cast to whole seconds truncates toward zero, which can only move a row
    *earlier* on the timeline. On the trailing hour's inclusive lower edge that
    is safe -- a true instant at or after the cutoff still truncates to at or
    after it -- so the truncation can never drop a genuinely in-window order out
    of the count.

    Args:
        record: The row to read.

    Returns:
        The epoch second the row was recorded at, or ``None`` when its stamp is
        unparseable or carries no UTC offset.
    """
    try:
        instant = datetime.fromisoformat(record.created_at)
        require_aware(instant, _CREATED_AT_FIELD)
    except ValueError:
        return None
    return int(instant.timestamp())


def _is_order_routing(record: LedgerRecord) -> bool:
    """Return whether a transition row marks one order being routed at a venue.

    Args:
        record: The ``OrderTransitionLedgered`` row to classify.

    Returns:
        ``True`` for the single ``REQUEST_SUBMISSION`` edge, ``False`` for every
        other lifecycle transition -- acks, fills, cancels and reconciliations
        all belong to orders already routed, and counting them would make the
        cap bind on an order's *progress* rather than on its placement.

    Raises:
        KeyError: If the payload is missing the field this reads -- a loud shape
            drift, never a silently uncounted order.
    """
    data = json.loads(record.payload_json)[_PAYLOAD_DATA_KEY]
    return bool(data[_TRANSITION_EVENT_KEY] == _REQUEST_SUBMISSION_EVENT)


def _fold_orders_last_hour(
    transitions: Iterable[LedgerRecord], cutoff_epoch_s: int
) -> int | None:
    """Count the orders routed at or after ``cutoff_epoch_s``.

    The shared body of both read paths below, so the indexed walk and the
    whole-ledger fold can never answer differently.

    The window is closed on its lower edge and *open* on its upper one: a row
    stamped ahead of the tick's own clock is counted rather than discarded. A
    forward clock blip is not evidence that the order was never routed, and
    skipping it would report a count smaller than the evidence supports --
    under-reporting what a cap has already consumed is precisely the permissive
    direction. The window is pure epoch arithmetic besides, so unlike the daily
    fold's calendar bucket it does not move with the host's timezone at all.

    A routing row whose instant cannot be established abandons the whole answer
    rather than being skipped, for the reason the day fold gives: the unreadable
    row might be this hour's, so counting only the readable ones would under-
    report the hour. A row of some *other* lifecycle edge is skipped whatever
    its stamp, because it could never have been counted.

    Args:
        transitions: The ``OrderTransitionLedgered`` rows to fold, in any order.
        cutoff_epoch_s: The oldest instant the trailing hour admits.

    Returns:
        The number of orders routed in the window, or ``None`` when any routing
        row's recorded instant could not be established.

    Raises:
        KeyError: If a payload is missing ``event``.
    """
    routed = 0
    for record in transitions:
        if not _is_order_routing(record):
            continue
        epoch_s = _record_epoch_s(record)
        if epoch_s is None:
            return None
        if epoch_s >= cutoff_epoch_s:
            routed += 1
    return routed


def orders_last_hour_count(
    records: Iterable[LedgerRecord], *, now_epoch_s: int
) -> int | None:
    """Return the number of orders routed in the trailing hour, or ``None``.

    ``velocity_limits`` caps the orders routed within the trailing hour, and the
    scheduler fed it a hardcoded zero. That made the gate evaluate ``0 + 1 >
    max_orders_per_hour`` -- false for every configured maximum of one or more
    -- so the runaway-order protection ran every tick, reported success, and
    could not veto however many orders the loop had just flung at the venue
    (issue #491). This is the fold that feeds it, and it is the last of the
    hardcoded-zero risk terms: the exposure quartet went in #407 and the day's
    notional in #415.

    An hour with no routed order is a genuine ``0`` and not ``None``, exactly as
    an untraded day is a genuine ``MoneyMicros(0)``: ``OrderTransitionLedgered``
    is written once per transition and never rewritten, so its absence is
    evidence rather than the lack of it. That makes the cap bind from the first
    tick against a fresh ledger instead of vetoing until something happens to be
    routed.

    ``None`` is reserved for the one case where the window genuinely cannot be
    established -- a routing row whose recorded instant carries no UTC offset
    (see :func:`_record_epoch_s`) -- and the caller must fail closed on it.

    Args:
        records: The ledger read (``SqliteLedgerStore.read_all()``). Rows of
            other types are ignored.
        now_epoch_s: The instant the trailing hour ends at.

    Returns:
        The number of orders routed in the trailing hour, or ``None`` when it is
        unprovable.

    Raises:
        KeyError: If an ``OrderTransitionLedgered`` payload is missing the field
            this reads -- a loud shape drift, never a silently uncounted order.
    """
    transitions = (
        record
        for record in records
        if record.event_type == _ORDER_TRANSITION_EVENT_TYPE
    )
    return _fold_orders_last_hour(transitions, now_epoch_s - _TRAILING_HOUR_SECONDS)


def read_orders_last_hour(store: LedgerStore, *, now_epoch_s: int) -> int | None:
    """Read the number of orders routed in the trailing hour from ``store``.

    The tick's entry point, mirroring :func:`read_notional_today_micros`: a
    store declaring the optional
    :class:`~windbreak.ledger.store.ReverseTypeScan` capability is walked over
    its ``OrderTransitionLedgered`` rows alone -- O(transitions) rather than
    O(ledger), on the composite index that walk already has -- and every other
    store, including each hand-rolled double, falls back to the whole-ledger
    fold.

    Like that fold's, and unlike the equity baseline's, this walk deliberately
    does **not** stop at the window boundary. An early stop is safe when the
    answer is a single earliest row; it is not safe for a *count*, because a
    clock that stepped backwards would put an out-of-window stamp above an
    in-window one and truncate the tally. A truncated tally under-reports how
    much of the hour's budget is spent, which fails open -- so the walk pays the
    full pass over the transitions and keeps the two paths' answers identical.

    Args:
        store: The ledger to read the hour's routed orders out of.
        now_epoch_s: The instant the trailing hour ends at.

    Returns:
        The number of orders routed in the trailing hour, or ``None`` when it is
        unprovable -- which keeps the hourly cap vetoing (see
        :func:`_build_limits`).
    """
    if isinstance(store, ReverseTypeScan):
        return _fold_orders_last_hour(
            store.iter_records_of_type_reversed(_ORDER_TRANSITION_EVENT_TYPE),
            now_epoch_s - _TRAILING_HOUR_SECONDS,
        )
    return orders_last_hour_count(store.read_all(), now_epoch_s=now_epoch_s)


def visible_depth_centis(order_book: OrderBookSnapshot) -> ContractCentis:
    """Return the visible depth ``participation_cap_compliance`` may bound against.

    The evaluation context is composed once per tick, before any intent's side
    is known, so the figure has to hold for either side of the book: it is the
    *shallower* of the two visible sides, summed across every level. Bounding a
    sale against the (possibly much deeper) ask side would admit an order
    larger than the bid side could absorb, which is the loosening issue #364
    exists to avoid.

    An empty side is a genuine observation of zero depth, not an unknown one,
    so it yields ``0`` -- which admits no positive-size order at all. Only a
    caller with no book to read passes ``None`` on to
    :func:`build_evaluation_context`, and that is the case the check answers
    with ``visible depth unknown``.

    Args:
        order_book: The book snapshot this tick took.

    Returns:
        The shallower visible side's total resting quantity, in
        contract-centis.
    """
    bid_depth = sum(level.quantity.value for level in order_book.yes_bids)
    ask_depth = sum(level.quantity.value for level in order_book.yes_asks)
    return ContractCentis(min(bid_depth, ask_depth))


def read_open_position_centis(
    exchange: PaperExchange, *, ticker: str
) -> ContractCentis | None:
    """Return what the venue says this account holds in ``ticker``, or ``None``.

    The one figure ``reduce_only_provable`` proves a close against, read from
    :meth:`~windbreak.connector.paper.PaperExchange.get_positions` -- the
    process's single definition of "position" (issue #361) -- so the check and
    the ledgered ``PositionsSnapshotRecorded`` row can never disagree about what
    is held.

    Three answers, and the distinction between the last two is the whole point
    of issue #373:

    * **The venue's own signed quantity** when a row exists for ``ticker``,
      passed through verbatim in the YES frame. The sign is *not* stripped: a
      long NO reports negative, and taking its magnitude would claim a YES-side
      close reduces a NO-side holding when in fact it opens a second position.
      Left signed, ``intent.size <= open_position`` is unsatisfiable for any
      positive size, which is the fail-closed answer.
    * **Zero** when the venue answered and reported no row for ``ticker``. That
      is an observed absence of fills, which the connector's own contract
      defines as genuinely flat -- evidence, not a gap, exactly as an empty book
      is ``0`` depth rather than unknown depth (issue #364). It costs nothing in
      permissiveness: every positive-size close exceeds zero.
    * **``None``** when the venue *refused* to describe itself --
      :class:`~windbreak.connector.paper.TwoSidedPositionError`, raised for a
      ticker filled on both sides because the only single-row answer is a netted
      one and a netted YES-plus-NO reports flat while both legs and their
      collateral are live. Substituting zero here would be a *claim* that the
      account is flat: precisely the fabricated healthy zero the connector just
      declined to invent. The check keeps vetoing, and the loud half of the
      response is :func:`_verification_stage`'s forced ``BREACH``.

    Args:
        exchange: The paper exchange whose fill log defines the holding.
        ticker: The market whose holding this tick evaluates against.

    Returns:
        The signed YES-frame holding in contract-centis, ``0`` when the venue
        reports the account flat there, or ``None`` when it cannot be
        determined.
    """
    try:
        positions = exchange.get_positions()
    except TwoSidedPositionError:
        return None
    for position in positions:
        if position.ticker == ticker:
            return position.quantity
    return ContractCentis(0)


def read_exchange_clock_epoch_s(exchange: PaperExchange) -> int | None:
    """Return what the venue says the time is, or ``None`` when it cannot say.

    The one reading ``clock_skew_limit`` measures our clock against (issue
    #377), taken from the venue rather than from this tick's own clock -- a
    check fed its own ``now_epoch_s`` reports a skew of exactly zero for any
    venue at any drift.

    Two answers, and the second is what issue #382 adds:

    * **The venue's own instant**, in whole epoch seconds, while the replay's
      recording still covers this run. Whether that agrees with our clock is
      exactly the question the check exists to ask, so the reading is passed
      through untouched however far it sits from ours.
    * **``None``** when the replay refuses
      (:class:`~windbreak.connector.paper.ReplayExhaustedError`): its cursor has
      consumed the recording, or an anchored run has outlived the recorded
      span. The recording holds no observation covering this instant, so there
      is no venue clock to report and ``clock_skew_limit`` vetoes with
      "exchange clock unknown". Substituting the anchored reading would state a
      venue time the recording never witnessed, and substituting our own would
      be the identically-zero skew #377 removed -- an exhausted replay is
      precisely the case where both substitutions look most reassuring and are
      least supported.

    Note that the cast is to whole seconds, so a sub-second venue instant floors
    onto the same integer timeline ``now_epoch_s`` uses; the tolerance is
    measured in whole seconds too (SPEC S6.1 keeps every epoch off the float
    path).

    Args:
        exchange: The paper exchange whose replay answers for the venue.

    Returns:
        The venue's clock in whole epoch seconds, or ``None`` when the replay
        can no longer substantiate one.
    """
    try:
        venue_now = exchange.get_exchange_time()
    except ReplayExhaustedError:
        return None
    return int(venue_now.timestamp())


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
    config: WindbreakConfig,
    instrument_whitelist: frozenset[str],
    *,
    exposure_provable: bool,
    notional_provable: bool,
    orders_provable: bool,
) -> RiskLimits:
    """Map a configuration into the risk limits the pre-trade checks read.

    Every field with a SPEC S16 counterpart is mapped from config; the few
    ``RiskLimits`` fields the schema has no dedicated field for take conservative
    named defaults (see the module constants).

    ``exposure_provable=False`` is the fail-closed reading for issue #407. When
    the tick could not establish what the account holds, the four concentration
    caps are set to ``0`` ppm rather than their configured shares. That is the
    only place the "unprovable" fact *can* be said: ``AccountState``'s exposure
    terms are :class:`~windbreak.numeric.MoneyMicros` with no ``None``, so an
    unprovable exposure would have to be carried as zero -- and zero is
    permissive, the exact defect this issue exists to remove. Saying it in the
    limits instead states a real, meaningful policy ("no exposure share is
    permitted right now") rather than inventing an exposure figure the account
    does not have, and ``concentration_limits`` then vetoes any positive cost.

    ``notional_provable=False`` is the same seam for issue #415's daily cap.
    ``AccountState.notional_today`` is likewise a
    :class:`~windbreak.numeric.MoneyMicros` with no ``None``, so a day whose
    booked notional could not be established would have to be carried as a zero
    that reads as a clean slate. Zeroing ``max_notional_per_day`` instead states
    the real policy -- "no notional may be routed while the day's spend is
    unknown" -- and ``velocity_limits`` then vetoes any positive cost. A
    provably de-risking close stays exempt, correctly: it can only reduce
    exposure, so an unknown budget must not block the exit.

    ``orders_provable=False`` is that same seam for issue #491's hourly cap, and
    it is the last of the three. ``AccountState.orders_last_hour`` is a plain
    ``int`` with no ``None``, so an hour whose routed-order count could not be
    established would have to be carried as a zero that reads as an idle hour --
    the very value that made this cap incapable of vetoing in the first place.
    Zeroing ``max_orders_per_hour`` instead states the real policy: "no order may
    be routed while the hour's routing history is unknown". ``velocity_limits``
    then vetoes on ``0 + 1 > 0`` before it consults anything else.

    That last one is deliberately stricter than the daily-notional half beside
    it, because the check is: the hourly gate runs *before* the de-risking-close
    exemption, on the stated grounds that a close can flood a venue just as an
    open can. So an unprovable hour blocks exits too. That is a real cost, and it
    is the one the check's own contract already chose -- widening the exemption
    to cover the hourly term would be loosening a risk gate under cover of a
    wiring fix, which belongs in its own issue with its own evidence.

    All three are deliberately *conditional*. A cap that vetoed unconditionally
    would not be failing closed, it would be broken; these caps veto only while
    the evidence is missing, and pass on a legitimate position once it is not.

    Args:
        config: The configuration to map.
        instrument_whitelist: The tradable-ticker set for this tick.
        exposure_provable: Whether this tick established the account's exposure.
            ``False`` zeroes the four concentration caps, as above.
        notional_provable: Whether this tick established the day's booked
            notional. ``False`` zeroes ``max_notional_per_day``, as above.
        orders_provable: Whether this tick established how many orders the
            trailing hour routed. ``False`` zeroes ``max_orders_per_hour``, as
            above.

    Returns:
        The assembled :class:`~windbreak.riskkernel.context.RiskLimits`.
    """
    risk = config.risk
    no_share_permitted = 0
    market_pct_ppm = (
        risk.max_pos_market_pct_ppm if exposure_provable else no_share_permitted
    )
    event_pct_ppm = (
        risk.max_pos_event_pct_ppm if exposure_provable else no_share_permitted
    )
    bucket_pct_ppm = (
        risk.max_pos_bucket_pct_ppm if exposure_provable else no_share_permitted
    )
    total_pct_ppm = (
        risk.max_pos_total_pct_ppm if exposure_provable else no_share_permitted
    )
    no_notional_permitted = 0
    notional_per_day = (
        risk.max_notional_per_day_micros if notional_provable else no_notional_permitted
    )
    no_order_permitted = 0
    orders_per_hour = (
        risk.max_orders_per_hour if orders_provable else no_order_permitted
    )
    return RiskLimits(
        floor=MoneyMicros(config.capital.floor_micros),
        instrument_whitelist=instrument_whitelist,
        micro_cap=MoneyMicros(config.capital.micro_cap_micros),
        min_open_price=PricePips(risk.min_open_price_pips),
        max_open_price=PricePips(risk.max_open_price_pips),
        max_participation_ppm=risk.max_participation_ppm,
        max_pos_market_pct_ppm=market_pct_ppm,
        max_pos_event_pct_ppm=event_pct_ppm,
        max_pos_bucket_pct_ppm=bucket_pct_ppm,
        max_pos_total_pct_ppm=total_pct_ppm,
        daily_loss_limit_pct_ppm=risk.daily_loss_limit_pct_ppm,
        max_drawdown_pct_ppm=risk.max_drawdown_pct_ppm,
        max_orders_per_hour=orders_per_hour,
        max_notional_per_day=MoneyMicros(notional_per_day),
        quote_ttl_seconds=risk.quote_ttl_seconds,
        forecast_ttl_seconds=_DEFAULT_FORECAST_TTL_SECONDS,
        clock_skew_max_seconds=risk.clock_skew_max_seconds,
        rounding_buffer=MoneyMicros(0),
        verification_ttl_seconds=_DEFAULT_VERIFICATION_TTL_SECONDS,
        require_human_ack_above_micros=_human_ack_micros(config),
        exchange_status_ttl_seconds=_DEFAULT_EXCHANGE_STATUS_TTL_SECONDS,
        pipeline_heartbeat_ttl_seconds=_DEFAULT_PIPELINE_HEARTBEAT_TTL_SECONDS,
    )


def _exposure_terms(
    exposure: ExposureProjection | None,
) -> tuple[MoneyMicros, MoneyMicros, MoneyMicros, MoneyMicros]:
    """Return the four concentration terms, or four fail-closed zeros.

    Extracted from :func:`_account_from_verification` so that function stays
    inside the ``xenon --max-absolute B`` ceiling as the last two hardcoded
    zeros are replaced (issues #513/#514, and the decomposition issue #492
    names). The four zeros are only safe paired with :func:`_build_limits`'
    zeroed concentration caps -- see that function.

    Args:
        exposure: The tick's projected exposure, or ``None`` when it could not
            be established.

    Returns:
        The market, event, bucket and total exposure terms, in that order.
    """
    if exposure is None:
        zero = MoneyMicros(0)
        return (zero, zero, zero, zero)
    return (
        exposure.market_exposure,
        exposure.event_exposure,
        exposure.bucket_exposure,
        exposure.total_exposure,
    )


def _equity_terms(
    equity_start_of_day: MoneyMicros | None,
    realized_loss_today: MoneyMicros | None,
    equity_curve: EquityCurve | None,
) -> tuple[MoneyMicros, MoneyMicros, MoneyMicros, MoneyMicros]:
    """Return the four equity-series terms, mapping absent evidence onto zero.

    All four are points on the ledger's own ``EquitySampled`` curve, and the
    zeros they fall back to are fail-*closed* rather than permissive -- which
    is a property of how they pair, not of the number itself:

    * A zero baseline floors ``daily_loss_limit``'s threshold at zero, which
      the zero loss beside it already reaches, so the check keeps vetoing on an
      unsampled day exactly as PR #481 pinned. That is a veto for the *unknown
      baseline*; issue #513's defect was the loss term, which was zero even
      when the day was fully sampled.
    * A zero mark floors ``trailing_drawdown_limit``'s threshold at zero, and
      the zero current reading beside it makes the drawdown zero, so
      ``0 >= 0`` vetoes. Zeroing only one side is what left that check inert:
      with a real worst-case equity opposite a zero mark the drawdown was
      *negative* and the cap could not bind at all (issue #514).

    Both pairs come from one fold each, so a caller cannot supply half a pair.

    Args:
        equity_start_of_day: The day's first ledgered equity sample, or
            ``None``.
        realized_loss_today: The day's realized loss, or ``None``.
        equity_curve: The all-time peak and newest sample, or ``None`` when the
            ledger holds no sample at all.

    Returns:
        The start-of-day equity, realized loss, high-water mark and newest
        sampled equity, in that order.
    """
    zero = MoneyMicros(0)
    mark = equity_curve.high_water_mark if equity_curve is not None else zero
    sampled = equity_curve.latest if equity_curve is not None else zero
    return (
        equity_start_of_day if equity_start_of_day is not None else zero,
        realized_loss_today if realized_loss_today is not None else zero,
        mark,
        sampled,
    )


def _velocity_terms(
    orders_last_hour: int | None, notional_today: MoneyMicros | None
) -> tuple[int, MoneyMicros]:
    """Return ``velocity_limits``' two terms, or their fail-closed zeros.

    Both halves of one check, resolved in one place so neither can be wired
    differently from the other: the hourly order count (issue #491) and the
    day's booked notional (issue #415). Each falls back to zero *only* because
    :func:`_build_limits` has simultaneously zeroed the cap that reads it --
    ``max_orders_per_hour`` and ``max_notional_per_day`` respectively -- so the
    zero is never the permissive reading it replaced. Neither half is correct
    alone; see :func:`_build_limits` for the other one.

    Args:
        orders_last_hour: The trailing hour's routed-order count, or ``None``
            when the hour's routing history could not be established.
        notional_today: The current UTC day's booked notional, or ``None`` when
            the day's spend could not be established.

    Returns:
        The routed-order count and the day's booked notional, in that order.
    """
    return (
        orders_last_hour if orders_last_hour is not None else 0,
        notional_today if notional_today is not None else MoneyMicros(0),
    )


def _account_from_verification(
    verification: VerificationSnapshot | None,
    equity_start_of_day: MoneyMicros | None,
    realized_loss_today: MoneyMicros | None,
    equity_curve: EquityCurve | None,
    exposure: ExposureProjection | None,
    notional_today: MoneyMicros | None,
    orders_last_hour: int | None,
) -> AccountState:
    """Return the account snapshot the tick's ledgered evidence supports.

    Two of the three populated terms come from the verification cycle:
    the venue-reported available cash and the observed drift (as the
    reconciliation-uncertainty buffer). That mirrors
    ``RiskKernel._stamp_verification`` exactly, and mirroring it is
    load-bearing rather than decorative: the kernel stamps those two terms onto
    its *own* copy of the context, but
    :meth:`~windbreak.riskkernel.reservations.ApprovalPipeline.approve`
    re-evaluates every check over the *caller's* context. If this function left
    verified cash at zero, the kernel would pass the floor invariant on the
    figures it stamped and the pipeline would then veto the same intent on the
    zeros the caller supplied -- a token that can never mint, for a reason
    invisible in the kernel's ledgered verdict.

    The third is the caller's ``equity_start_of_day``, folded out of the
    loop's own ledgered ``EquitySampled`` history by
    :func:`start_of_day_equity_micros` (issue #364). ``AccountState`` has no
    ``None`` for it, so an absent baseline is carried as zero -- and zero is
    the fail-closed reading, not a permissive one: it floors
    ``daily_loss_limit``'s threshold at zero, which today's zero realized loss
    already reaches, so the check keeps vetoing exactly as it did before any
    baseline existed.

    The four exposure terms come from ``exposure`` (issue #407). They were
    hardcoded to zero, and zero is *permissive* for ``concentration_limits``:
    ``0 + cost > share`` is false for any sane cost, so that check could not
    veto for one market or for many, however much the account already held.
    :func:`windbreak.scheduler.exposure.project_exposure` now derives all four
    from the venue's own reported holdings, priced by
    :func:`_position_value_micros`.

    ``exposure=None`` means the tick could not establish what the account
    holds, and the terms fall back to zero *only* because
    :func:`_build_limits` has simultaneously set the four concentration caps to
    ``0`` ppm, which makes ``concentration_limits`` veto any positive cost.
    ``AccountState`` has no ``None`` to carry "unprovable" with, so the fact is
    stated in the limits instead of fabricated into an exposure figure; the two
    halves are set together by :func:`build_evaluation_context` and neither is
    correct alone.

    ``notional_today`` comes from ``notional_today`` (issue #415). It was
    hardcoded to zero, and zero is *permissive* for ``velocity_limits``' daily
    cap: an account that looks untraded has its whole budget left however much
    the loop routed today. :func:`read_notional_today_micros` now folds it out
    of the ledger's own ``FillAccounted`` rows, bucketed by each row's
    hash-chained ``created_at``.

    ``notional_today=None`` means the day's spend could not be established, and
    the term falls back to zero *only* because :func:`_build_limits` has
    simultaneously zeroed ``max_notional_per_day`` -- the same paired reading
    ``exposure`` uses, and for the same reason: ``AccountState`` has no ``None``
    to carry "unprovable" with, so the fact is stated in the limits instead of
    fabricated into the account. Neither half is correct alone.

    ``orders_last_hour`` closes the same shape for ``velocity_limits``' other
    half (issue #491). It was hardcoded to zero, and zero is *permissive* in the
    strongest possible sense for the hourly cap: the gate reads ``0 + 1 >
    max_orders_per_hour``, which is false for every configured maximum of one or
    more, so the runaway-order protection could not veto at any order rate
    whatsoever. :func:`read_orders_last_hour` now folds it out of the ledger's
    own ``OrderTransitionLedgered`` rows, counting the one ``REQUEST_SUBMISSION``
    edge per routed order and bucketing on each row's hash-chained
    ``created_at``.

    ``orders_last_hour=None`` means the hour's routing history could not be
    established, and the term falls back to zero *only* because
    :func:`_build_limits` has simultaneously zeroed ``max_orders_per_hour`` --
    the same paired reading ``exposure`` and ``notional_today`` use, and for the
    same reason. Neither half is correct alone; here that is especially true,
    since the zero this term falls back to is the exact defect being removed.
    Both of ``velocity_limits``' terms are resolved together by
    :func:`_velocity_terms`, so neither can be wired differently from the other.

    ``realized_loss_today`` comes from ``realized_loss_today`` (issue #513) and
    ``equity_high_water_mark``/``sampled_equity`` from ``equity_curve`` (issue
    #514). Both were hardcoded zero, and both zeros were permissive:

    * ``0 >= _ppm_of(equity_start_of_day, daily_loss_limit_pct_ppm)`` is false
      for every positive baseline, so ``daily_loss_limit`` could not veto at
      any realized loss once the day had a ledgered sample.
    * ``0 - worst_case_equity >= _ppm_of(0, ...)`` is ``negative >= 0``, false
      for every solvent account, so ``trailing_drawdown_limit`` could not veto
      at any drawdown until equity had already reached zero.

    :func:`read_realized_loss_today_micros` and :func:`read_equity_curve` now
    fold all three out of the ledger's own ``EquitySampled`` rows.
    :func:`_equity_terms` documents why their ``None`` maps onto zero here
    rather than onto a zeroed limit the way ``exposure`` and ``notional_today``
    do: for these two caps *both* sides of the comparison come off the same
    fold, so the paired zeros already veto, and a zeroed cap would be a branch
    no input could reach.

    Args:
        verification: The tick's verification snapshot, or ``None`` when no
            cycle has produced one (the fail-closed reading).
        equity_start_of_day: The current UTC day's first ledgered equity
            sample, or ``None`` when the day has none yet (also fail-closed).
        realized_loss_today: The loss the current UTC day has realized, or
            ``None`` when the day carries no sample at all.
        equity_curve: The all-time high-water mark and the newest sampled
            equity, or ``None`` when the ledger holds no sample at all.
        exposure: The tick's projected exposure, or ``None`` when it could not
            be established -- in which case the caller must also have zeroed
            the concentration caps, as above.
        notional_today: The current UTC day's booked notional, or ``None`` when
            it could not be established -- in which case the caller must also
            have zeroed the daily notional cap, as above.
        orders_last_hour: The number of orders the trailing hour routed, or
            ``None`` when it could not be established -- in which case the
            caller must also have zeroed the hourly order cap, as above.

    Returns:
        The composed :class:`~windbreak.riskkernel.context.AccountState`.
    """
    zero = MoneyMicros(0)
    verified_cash = (
        verification.exchange_verified_available_cash
        if verification is not None
        else zero
    )
    drift = verification.cash_drift if verification is not None else zero
    baseline, loss, mark, sampled = _equity_terms(
        equity_start_of_day, realized_loss_today, equity_curve
    )
    market, event, bucket, total = _exposure_terms(exposure)
    routed, day_notional = _velocity_terms(orders_last_hour, notional_today)
    return AccountState(
        exchange_verified_available_cash=verified_cash,
        guaranteed_terminal_value_of_positions=zero,
        pending_kernel_reservations=zero,
        unresolved_fee_upper_bounds=zero,
        reconciliation_uncertainty_buffer=drift,
        equity_start_of_day=baseline,
        equity_high_water_mark=mark,
        sampled_equity=sampled,
        realized_loss_today=loss,
        market_exposure=market,
        event_exposure=event,
        bucket_exposure=bucket,
        total_exposure=total,
        orders_last_hour=routed,
        notional_today=day_notional,
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
    quote_snapshot_epoch_s: int | None,
    exchange_clock_epoch_s: int | None,
    forecast_epoch_s: int | None,
    open_position: ContractCentis | None,
    equity_start_of_day: MoneyMicros | None,
    realized_loss_today: MoneyMicros | None,
    equity_curve: EquityCurve | None,
    visible_depth: ContractCentis | None,
    exposure: ExposureProjection | None,
    notional_today: MoneyMicros | None,
    orders_last_hour: int | None,
) -> EvaluationContext:
    """Compose the evaluation context a PAPER-mode approval reads.

    Maps the operator's configured capital floor and risk thresholds onto the
    risk limits, stamps the supplied ``now_epoch_s`` verbatim (never
    ``time.time()``), and passes ``verification`` straight through -- there is no
    production default in its place, so a forgotten wiring must fail closed via
    the reconciliation checks rather than open (mirroring
    :class:`~windbreak.riskkernel.context.EvaluationContext`'s own contract).
    The account is derived from that same snapshot
    (:func:`_account_from_verification`), so the verified cash the floor
    invariant reads and the snapshot the reconciliation checks read describe
    one observation, never two.

    The exchange status and pipeline heartbeat are caller-supplied rather than
    hardcoded (issue #342), so the loop can pass genuine observations. They are
    still ``| None``, and a caller with nothing to report must pass ``None``
    rather than a placeholder: both checks fail closed on it. In particular the
    heartbeat must be stamped by an earlier stage and never set to this
    function's own ``now_epoch_s`` -- a heartbeat equal to ``now`` can never go
    stale, which would make its check unfalsifiable and strictly worse than the
    ``None`` it replaced.

    The quote snapshot epoch is caller-supplied for the same reason and with
    the same prohibition (issue #369). It must be the *book's own*
    ``fetched_at``, never this function's ``now_epoch_s``: ``_is_stale(now,
    now, ttl)`` is ``False`` for every ttl, so a quote stamped with the
    evaluation instant makes ``quote_freshness`` unfalsifiable -- and that is
    the SPEC S7.3 snapshot-TTL guarantee, the one check standing between the
    kernel and an order priced off a stale book. A caller holding no book
    passes ``None`` and the check keeps vetoing.

    The exchange clock is caller-supplied and subject to the same prohibition
    (issue #377): it is the *venue's* reading, from
    :meth:`~windbreak.connector.paper.PaperExchange.get_exchange_time`, never
    this function's ``now_epoch_s``. Fed the local clock it produced a skew of
    identically zero, so ``clock_skew_limit`` could not veto for any venue at
    any drift -- and that is the check guarding all the others, because quote
    freshness, exchange-status staleness, and the pipeline heartbeat are each
    measured as ``now - stamp`` and each mismeasure when ``now`` is wrong
    relative to the venue. A caller who cannot read the venue's clock passes
    ``None`` and the check vetoes with ``exchange clock unknown``; defaulting to
    the local clock instead would report perfect agreement, which is the most
    reassuring answer available and the least evidenced.

    The forecast epoch closes the same shape for the last time (issue #380). It
    is the *forecast's own* ``created_at``, threaded down from
    :func:`_forecast_stage`, never this function's ``now_epoch_s``. Fed the
    tick's clock, ``forecast_freshness`` measured now against now and could not
    veto a forecast of any age -- which is precisely the aging
    :func:`_approve_stage`'s deliberate second clock read exists to expose,
    since a slow forecast stage genuinely ages its own output between the two
    readings. A tick that produced no forecast at all (the issue-#339 research
    budget halt) passes ``None`` and the check keeps vetoing; the tick's clock
    would instead claim a forecast zero seconds old where provably none exists.

    The open position is caller-supplied too (issue #373), from
    :func:`read_open_position_centis`. It was a hardcoded ``None``, so
    ``reduce_only_provable`` vetoed every close on every tick -- correctly, but
    permanently, because the kernel had never been told what the account holds.
    ``None`` remains the answer whenever the holding genuinely cannot be
    determined, and it must never be softened to zero: zero is a *claim* that
    the account is flat, and an indeterminate position is not a flat one.

    The start-of-day equity and the visible depth are caller-supplied for the
    same reason (issue #364), and both loosen a real exposure limit when they
    are wrong: a larger baseline raises ``daily_loss_limit``'s threshold and a
    deeper book raises ``participation_cap_compliance``'s ceiling. So neither
    may be defaulted here. A caller that cannot prove either figure passes
    ``None`` and both checks keep failing closed -- ``daily loss limit
    reached`` on the zero baseline an absent sample maps to, and ``visible
    depth unknown`` on the absent book.

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
        quote_snapshot_epoch_s: Epoch second the order book this evaluation
            prices against was fetched, or ``None`` when no book could be read
            -- which fails closed (issue #369). Taken from the book's own
            ``fetched_at``, never from the tick's clock: a quote stamped
            ``now`` is zero seconds old by construction and ``quote_freshness``
            could never veto.
        exchange_clock_epoch_s: The venue's own clock, in epoch seconds, or
            ``None`` when it could not be read -- which fails closed (issue
            #377). Never the tick's clock: comparing the local clock with
            itself yields a skew of zero and makes ``clock_skew_limit``
            unfalsifiable.
        forecast_epoch_s: The instant this tick's forecast was created, in
            epoch seconds, taken from its own ``created_at``, or ``None`` when
            the tick produced no forecast -- which fails closed (issue #380).
            Never the tick's clock: a forecast stamped ``now`` is zero seconds
            old by construction, so ``forecast_freshness`` could never veto.
        open_position: The signed YES-frame holding in this tick's market, in
            contract-centis, or ``None`` when the venue could not describe it
            -- which fails closed (issue #373). A venue-reported flat account is
            ``0``, not ``None``: an observed absence of fills is evidence, and
            it proves no close either way.
        equity_start_of_day: The current UTC day's first ledgered equity
            sample, or ``None`` when the day has none yet -- which fails closed
            (issue #364).
        realized_loss_today: The loss the current UTC day has realized, from
            :func:`read_realized_loss_today_micros`, or ``None`` when the day
            carries no sample at all (issue #513). Caller-supplied for the same
            reason as the baseline: it was hardcoded to zero, and an account
            that looks to have lost nothing has its whole daily allowance left
            however much it has actually lost.
        equity_curve: The all-time high-water mark and the newest sampled
            equity, from :func:`read_equity_curve`, or ``None`` when the ledger
            holds no sample at all (issue #514). One argument for both terms
            because a drawdown is the distance between two readings of one
            series; supplying a mark without the reading it is measured
            against is what left that cap inert.
        visible_depth: The visible book depth, in contract-centis, or ``None``
            when no book could be read -- which fails closed (issue #364). A
            genuinely empty book is ``0``, not ``None``: an observed absence of
            liquidity is evidence, and it admits no order at all.
        exposure: The tick's projected exposure from
            :func:`windbreak.scheduler.exposure.project_exposure`, or ``None``
            when it could not be established (issue #407). This one argument
            sets *both* halves of the concentration caps: the four
            ``AccountState`` exposure terms and, when ``None``, the four
            ``RiskLimits`` caps zeroed to make them veto. They are set together
            here precisely because neither is the fail-closed answer alone -- a
            zeroed exposure under a configured cap is the permissive reading
            this issue removes.
        notional_today: The current UTC day's booked notional from
            :func:`read_notional_today_micros`, or ``None`` when it could not be
            established (issue #415). It sets both halves of the daily cap the
            same way ``exposure`` does: the ``AccountState`` term and, when
            ``None``, the zeroed ``max_notional_per_day`` that makes
            ``velocity_limits`` veto. A day with no booked fill is
            ``MoneyMicros(0)``, not ``None`` -- an untraded day provably routed
            nothing.
        orders_last_hour: The number of orders the trailing hour routed, from
            :func:`read_orders_last_hour`, or ``None`` when it could not be
            established (issue #491). It sets both halves of the hourly cap the
            way ``notional_today`` sets the daily one: the ``AccountState`` term
            and, when ``None``, the zeroed ``max_orders_per_hour`` that makes
            ``velocity_limits`` veto. An hour with no routed order is ``0``, not
            ``None`` -- a quiet hour provably routed nothing. Never this
            function's own ``now_epoch_s`` window read at composition time on a
            *different* clock than the one stamped here: the count and the
            instant it is a trailing hour of must be one reading, or the window
            is measured against a boundary the evaluation never used.

    Returns:
        The composed :class:`~windbreak.riskkernel.context.EvaluationContext`.
    """
    # Every `*_epoch_s` below is the caller's evidence; none is this function's
    # own `now_epoch_s`. That sweep is complete as of issue #380 -- a field fed
    # `now_epoch_s` makes its consumer measure now against now, which no
    # observation can falsify.
    market_view = MarketView(
        quote_snapshot_epoch_s=quote_snapshot_epoch_s,
        forecast_epoch_s=forecast_epoch_s,
        visible_depth=visible_depth,
        exchange_clock_epoch_s=exchange_clock_epoch_s,
        open_position=open_position,
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
        limits=_build_limits(
            config,
            instrument_whitelist,
            exposure_provable=exposure is not None,
            notional_provable=notional_today is not None,
            orders_provable=orders_last_hour is not None,
        ),
        account=_account_from_verification(
            verification,
            equity_start_of_day,
            realized_loss_today,
            equity_curve,
            exposure,
            notional_today,
            orders_last_hour,
        ),
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
        screener: The real §16 screener every tick puts the venue's market
            universe through before spending any research money (issue #345).
            It replaces the single ``ticker`` this bundle used to carry, which
            was ``next(iter(exchange.markets))`` -- one arbitrary market, fixed
            for the life of the process and never screened at all. Deliberately
            non-optional and with no injection parameter, for the same reason as
            ``budget`` and ``provider_gate``: a ``None`` here would be a loop
            that forecasts unscreened markets, so the type makes an unscreened
            loop unrepresentable rather than merely discouraged. Its decisions
            land on ``store`` as ``ScreenDecisionRecorded`` rows through
            :class:`~windbreak.scheduler.screening.ScreenLedgerWriter`.
        store: The hash-chained ledger every stage appends to.
        exchange: The replay-driven paper exchange orders fill against.
        verification_view: The narrow, read-only view of that same exchange the
            verification cycle observes through. It exposes only the five
            account/market reads -- no ``place_order``/``cancel_order`` -- so
            the verification path structurally cannot trade (SPEC S1.1
            invariant 3), even though it watches the very exchange this loop
            fills against.
        fill_bookkeeper: Books each of the exchange's executions into the ledger
            exactly once, so the verification cycle's expectation can advance
            from ledgered evidence instead of freezing at process start and
            halting on the first fill (issue #365). Paired by component label
            with the feed :func:`_build_verifier` wires.
        gateway: The recovered Order Gateway submissions route through.
        reconciler: The bounded reconciler run to fixpoint after a fill.
        approval: The approval seam intents are decided through.
        kernel: The very Risk Kernel inside ``approval``, exposed so the tick
            can drive its per-tick verification cycle, thread the resulting
            snapshot into the approval context, and stamp the kernel's *real*
            mode on the tick heartbeat. Held separately from ``approval``
            precisely because ``approval`` is swappable: a test that doubles
            the seam still ticks the real kernel's verification.
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
        provider_gate: The per-provider track-record live-eligibility gate every
            tick's forecast runs under (issue #305). Built once per process from
            the M6 track-record artifact and the configured thresholds, and
            deliberately non-optional: a ``None`` here would be a loop that
            grants live eligibility to providers with no measured edge, which
            SPEC S19 forbids outright, so the type makes an ungated loop
            unrepresentable rather than merely discouraged.
        provider_factory: Builds the vote provider for one ensemble member
            (issue #269). Non-optional for the same reason as ``budget`` and
            ``provider_gate``: on the live path it is what wraps each vote in
            the configured bounded-retry policy and fail-closed price table, so
            leaving it absent would be a loop calling paid providers with
            neither. Offline it is the bare fixture provider the pipeline would
            have built itself, keeping the cassette path byte-identical; see
            :mod:`windbreak.scheduler.provider_wiring` for why only the live
            path is wrapped.
    """

    config: WindbreakConfig
    screener: MarketScreener
    store: SqliteLedgerStore
    exchange: PaperExchange
    verification_view: ReadOnlyVenueView
    fill_bookkeeper: LedgerFillBookkeeper
    gateway: OrderGateway
    reconciler: Reconciler
    approval: ApprovalSeam
    kernel: RiskKernel
    verification_key: bytes
    transport: LlmTransport
    research_tools: ResearchTools
    report_dir: Path
    clock: Callable[[], int]
    budget: ResearchBudget
    provider_gate: ProviderTrackRecordGate
    provider_factory: ProviderFactory


def _default_clock() -> int:
    """Return the current wall clock as whole epoch seconds.

    Casts :func:`time.time` to an ``int`` so the loop's clock stays off the
    banned float path (SPEC S6.1).

    Returns:
        The current time, in whole epoch seconds.
    """
    return int(time.time())


def _log_replay_corpus(config: WindbreakConfig, corpus: ReplayCorpus | None) -> None:
    """Report the replay-corpus mode in force and the source that chose it.

    An operator has to be able to tell, from the log of a run that traded,
    whether that run's evidence was recorded. The line therefore names the
    *effective* mode, the directory, the source that won, and what the corpus
    actually holds -- a mode name alone would not distinguish a corpus that
    covers the ensemble from one that covers nothing and abstains.

    It is logged at ``WARNING`` when a corpus is in force, because a replaying
    run's forecasts are recorded material rather than measurements and that must
    not be discoverable only by reading configuration; and at ``INFO`` for the
    shipped default, which is the offline loop that cannot trade.

    Args:
        config: The active configuration.
        corpus: The loaded corpus, or ``None`` when none is selected.
    """
    settings = config.forecast.replay_corpus
    source = replay_corpus_source(config)
    if corpus is None:
        _LOGGER.info("forecast replay corpus mode=%s source=%s", settings.mode, source)
        return
    _LOGGER.warning(
        "forecast replay corpus mode=%s dir=%s source=%s documents=%d votes=%d; "
        "forecasts on this run replay recorded material and measure nothing",
        settings.mode,
        settings.corpus_dir,
        source,
        len(corpus.documents),
        len(corpus.votes),
    )


def _resolve_replay_corpus(
    config: WindbreakConfig, provider_http: LiveProviderHttp | None
) -> ReplayCorpus | None:
    """Load the committed replay corpus this run selects, or ``None`` (#510).

    Resolved once per process, before the ledger database or any exchange
    session exists, so a corpus an operator pointed at and got wrong aborts
    startup rather than leaving half-built durable state behind -- the same
    ordering :func:`_resolve_forecast_transport` already has.

    A corpus and the live transport are mutually exclusive, and the refusal is
    explicit rather than a precedence rule. They are two answers to the same
    question -- where does this run's evidence come from -- and a deployment
    that stated both has stated a contradiction. Silently preferring either
    would hand an operator recorded material while they believed they were
    reading the world, or the reverse.

    Args:
        config: The active configuration naming the corpus mode.
        provider_http: The live HTTP seams, or ``None`` in cassette mode.

    Returns:
        The loaded corpus, or ``None`` when no corpus is selected.

    Raises:
        ValueError: On an unknown mode, on a replay mode naming no directory,
            or on a corpus selected alongside the live transport.
        CorpusFormatError: If the named directory is not a well-formed corpus.
    """
    directory = replay_corpus_directory(config)
    if directory is None:
        return None
    if provider_http is not None:
        raise ValueError(
            "forecast.replay_corpus selects a recorded corpus while "
            "forecast.provider_transport.mode is 'live'; a run reads recorded "
            "material or the world, never both -- select one"
        )
    return load_corpus(directory)


def _resolve_forecast_transport(
    config: WindbreakConfig,
    cassette_path: Path,
    provider_http: LiveProviderHttp | None,
    corpus: ReplayCorpus | None = None,
) -> tuple[LlmTransport, bool]:
    """Select the recorded cassette or the live provider transport (issue #344).

    Configuration states the intent and the caller supplies the live seam, and
    the two must agree. A half-configuration in *either* direction refuses to
    start, mirroring the ``market_data``/``live_ticker`` pair (issue #343):

    * **Live selected, no seam supplied.** Degrading to the cassette would hand
      an operator who asked for novel forecasts a recorded paper tape while
      they believed they were reading the market.
    * **Seam supplied, live not selected.** The transports (and the credentials
      inside them) were built for nothing. Silently ignoring them hides a
      mistake in exactly the place a mistake is expensive.

    Args:
        config: The active configuration naming the transport mode.
        cassette_path: The recorded cassette the offline replay transport
            serves from.
        provider_http: The live HTTP seams, or ``None``.
        corpus: The loaded replay corpus (issue #510), or ``None``. When one is
            selected the votes come from it rather than from the prompt-hash
            cassette, because a cassette key digests the market's close time and
            an anchored replay moves that on every run.

    Returns:
        The selected transport paired with whether live mode is in force.

    Raises:
        ValueError: On an unknown mode, or on either half-configuration.
    """
    live = is_live_mode(config)
    if live and provider_http is None:
        raise ValueError(
            "forecast.provider_transport.mode is 'live' but no `provider_http` "
            "seam was supplied; supply the live transports or select 'cassette'"
        )
    if not live and provider_http is not None:
        raise ValueError(
            "a `provider_http` seam was supplied while "
            "forecast.provider_transport.mode is 'cassette'; select 'live' or "
            "omit the seam"
        )
    if corpus is not None:
        return build_corpus_vote_transport(corpus), False
    if provider_http is None:
        return ReplayCassette.from_path(cassette_path), False
    return build_live_llm_transport(config, provider_http), True


def _resolve_research_tools(
    research_tools: ResearchTools | None,
    ledger_path: Path,
    config: WindbreakConfig,
    provider_http: LiveProviderHttp | None,
    corpus: ReplayCorpus | None = None,
) -> tuple[str, ResearchTools]:
    """Return this run's evidence source token and the research tools for it.

    An explicitly supplied bundle always wins, so a test can drive counted or
    doubled transports through either mode. Otherwise the default follows the
    selected transport (issue #344): live mode gets the live search/fetch
    transports behind the sandbox's own host allowlist, and cassette mode gets
    the offline no-network default, which never actually searches (its
    transports find nothing) so the pipeline abstains on zero verified
    citations before any fetch.

    Args:
        research_tools: The caller-supplied tools, or ``None``.
        ledger_path: The tick's ledger path, whose parent roots the fetch cache.
        config: The active configuration supplying live research settings.
        provider_http: The live HTTP seams, or ``None`` in cassette mode.
        corpus: The loaded replay corpus (issue #510), or ``None``. A corpus
            outranks the offline default and is mutually exclusive with the
            live seams -- :func:`_resolve_replay_corpus` refuses that pairing
            before either can be built.

    Returns:
        The branch's :data:`RESEARCH_EVIDENCE_SOURCES` token paired with the
        sandboxed :class:`~windbreak.forecast.sandbox.ResearchTools` it built.
        The token is returned *from here* rather than re-derived from config by
        the guard that consumes it (issue #485), so the two cannot disagree:
        dropping ``corpus`` from this function's call site moves the reported
        source to :data:`RESEARCH_EVIDENCE_NONE` in the same edit. A guard that
        re-read configuration instead would keep reporting a corpus that the
        wiring no longer passed -- the composition trap, wearing a safety label.

        The live branch's token keys on the same ``search``/``fetch`` pair
        :func:`~windbreak.scheduler.provider_wiring.build_live_research_tools`
        keys its own offline fallback on, so a live deployment that pinned an
        LLM but never named a search endpoint is reported as having no evidence
        source -- which is exactly what it has.
    """
    cache_dir = ledger_path.parent.joinpath("research-cache")
    if research_tools is not None:
        return RESEARCH_EVIDENCE_INJECTED, research_tools
    if corpus is not None:
        return RESEARCH_EVIDENCE_CORPUS, build_corpus_research_tools(corpus, cache_dir)
    if provider_http is not None:
        tools = build_live_research_tools(config, provider_http, cache_dir)
        if provider_http.search is None or provider_http.fetch is None:
            return RESEARCH_EVIDENCE_NONE, tools
        return RESEARCH_EVIDENCE_LIVE, tools
    return RESEARCH_EVIDENCE_NONE, offline_research_tools(cache_dir)


def research_evidence_fold(source: str) -> str:
    """Return the startup fold line reporting the effective evidence source.

    PR #487 established the shape: an operator must be able to read the
    *effective* value out of the log rather than infer it from configuration
    they may not have written. One formatter, used by the emitter and by the
    tests that pin it, so the line an operator greps for is the line asserted.

    Args:
        source: One of :data:`RESEARCH_EVIDENCE_SOURCES`.

    Returns:
        The rendered fold line.
    """
    return f"research evidence source={source}"


def _log_research_evidence(source: str) -> None:
    """Fold this run's evidence source into the log, loudly when there is none.

    Issue #438 asked for a startup failure when the PAPER loop is activated in a
    composition that can only abstain; issue #485 carried that forward and the
    owner deferred it twice, because until PR #522 (#510) made a corpus
    selectable from a command line, *every* configuration was such a
    composition and this signal would have fired on every start. A signal that
    always fires teaches operators to ignore it, which is worse than none.

    It **warns and continues** rather than refusing, which is the deliberate
    deviation from #438's wording:

    * PR #487's standard is fail closed on the capability, never on the
      process. A starved loop already fails closed on the capability -- it
      emits no intent. Refusing the process would additionally cost the
      operator the kill file, the ledger and the verification cycle; a
      deployment that cannot produce evidence is not one that must not be
      stoppable.
    * The shipped default *is* the starved composition, on purpose, and #522
      proved it stays that way. Refusing it would turn ``docker compose up``
      into a hard failure -- a regression dressed as a fix.
    * Configuration *contradictions* do refuse, here and next door:
      :func:`_resolve_replay_corpus` rejects an unknown mode, a replay mode
      naming no directory, and a corpus selected alongside the live transport.
      An evidence-starved deployment has contradicted nothing; it has merely
      configured less than it needed, and the two must not read alike.

    An injected bundle (:data:`RESEARCH_EVIDENCE_INJECTED`) folds like any other
    source and is never warned on: a caller that handed in its own tools knows
    what they find, and this function does not.

    Args:
        source: The token :func:`_resolve_research_tools` returned.
    """
    if source == RESEARCH_EVIDENCE_NONE:
        _LOGGER.warning(EVIDENCE_STARVED_MESSAGE)
        return
    _LOGGER.info(research_evidence_fold(source))


def _resolved_dispatcher(dispatcher: AlertDispatcher | None) -> AlertDispatcher:
    """Return the injected alert root, or the documented log-only fallback.

    The single place :func:`build_paper_deps`'s optional ``dispatcher`` becomes
    concrete, so "no deliverable sink means log-only" is stated once rather than
    re-derived at each consumer. A deployment that declares no sink must still
    start and still log -- that is the shipped default (``config.alerts``'s SPEC
    S16 entry is a placeholder that builds nothing) -- so this fallback is a
    documented behaviour, not a silent degradation. Issue #444's defect was the
    converse: this was the *only* behaviour available, including to a deployment
    that had configured a real sink.

    Args:
        dispatcher: The composed alert root, or ``None``.

    Returns:
        ``dispatcher`` when one was injected, else a dispatcher over no sinks
        whose ``log-only`` fallback carries every alert.
    """
    if dispatcher is not None:
        return dispatcher
    return AlertDispatcher(sinks=[], ledger_writer=LoggingLedgerWriter())


def _build_verifier(
    store: SqliteLedgerStore,
    config: WindbreakConfig,
    view: ReadOnlyVenueView,
    writer: _SqliteKernelLedgerWriter,
    dispatcher: AlertDispatcher,
) -> ReadOnlyVerifier:
    """Wire the PAPER loop's read-only verification cycle (issue #353).

    Mirrors ``windbreak.main._build_verifier``'s live composition, over the same
    hash-chained ``store``: a :class:`~windbreak.riskkernel.verification.\
LedgerExpectationSource` folds the replayed history *once, here at startup*
    into one frozen baseline, and the verifier diffs the venue against it each
    cycle. That freeze is what makes the comparison falsifiable rather than a
    tautology. Every PAPER dimension falls back to this view's startup capture
    (nothing stamps ``component="riskkernel"`` position snapshots), so the
    baseline is the account *as it stood before this process traded*: flat,
    with the fixture's opening cash and no resting orders. Since issue #352 the
    exchange derives its balances and positions from its own fill log, so every
    later cycle compares a live, moving observation against that fixed
    baseline -- a comparison that can, and on any real fill does, fail.

    That baseline no longer stays frozen for the life of the process. Until
    issue #365 it did, and the consequence was that the first PAPER fill moved
    the venue away from the only baseline the ledger could justify -- fills were
    not ledgered with amounts, so no ledgered fact could update it -- the next
    cycle graded ``BREACH``, the kernel HALTed per issue #32, and only a restart
    cleared it. An always-on PAPER deployment could not survive its own first
    fill.

    A :class:`~windbreak.scheduler.fill_accounting.LedgerFillAccountingFeed`
    now advances it from *ledgered evidence*. The paired
    :class:`~windbreak.scheduler.fill_accounting.LedgerFillBookkeeper` books
    each execution once, durably, into this same hash chain, and -- since issue
    #390 -- each order that comes to rest, so the open-order dimension advances
    too. Before that, an order left resting was one the ledger had never heard
    of, and the loop breached on the next cycle the moment it routed anything
    other than an outright marketable order. Two composition-time decisions live
    here, both deliberate:

    * **Which component is trusted.** The feed accepts only bookings stamped
      ``_COMPONENT`` -- this loop's own. In the PAPER deployment the scheduler
      *is* the account's bookkeeper, so declaring it here is the honest form of
      that trust; the alternative, letting the kernel fold whatever
      ``FillAccounted`` rows a shared ``ledger`` volume happens to carry, is
      exactly the cross-process contamination
      ``_own_component_events`` was written to stop.
    * **Where the cursor starts.** At the chain head *as of this call*, which is
      the ledger position the baseline above was captured over. Entries booked
      by an earlier process are already reflected in that baseline -- a fresh
      ``PaperExchange`` opens flat -- so folding them again would advance the
      expectation past cash the venue never moved.

    This is not the issue #352 tautology returning. A booked entry is frozen at
    execution and describes one discrete movement; the observation is the
    venue's live *aggregate*. A venue that moves by anything the books cannot
    explain -- an unbooked fill, a settlement, a retired resting order -- still
    diverges and still halts. Only the explained part is absorbed. Re-reading
    the expectation off the same connector each cycle, by contrast, would make
    all three dimensions structurally incapable of failing.

    The tolerances come from ``config.risk`` (both default to ``0``: exact
    match). The ``dispatcher`` that fans mismatch and unknown-jurisdiction
    alerts out is *injected*, and that is the whole of issue #444: this function
    used to build its own ``AlertDispatcher(sinks=[], ...)``, so the only
    alerting surface the always-on loop has could reach nothing but the log-only
    fallback no matter what ``config.alerts`` declared -- while
    ``docs/RUNBOOK.md`` told the operator that declaring a sink was "a
    prerequisite for any unattended run". The scheduler must not read
    ``config.alerts`` itself: resolving a sink's ``*_env`` destination means
    reading the real environment, and
    :func:`windbreak.main._build_alert_dispatcher` is deliberately the one place
    that happens, so no second composition site can leak a destination into a
    log line or the append-only ledger. The seam is therefore a parameter, and
    :func:`build_paper_deps` documents the one door it arrives through.

    Args:
        store: The hash-chained ledger whose replayed history seeds the
            baseline.
        config: The configuration supplying the two drift tolerances.
        view: The narrow read-only venue view the cycle observes through -- it
            exposes no ``place_order``/``cancel_order`` (SPEC S1.1 invariant 3).
        writer: The kernel ledger writer each cycle's event is recorded through.
        dispatcher: The composed alert root every mismatch and
            unknown-jurisdiction alert is delivered through.

    Returns:
        The composed :class:`~windbreak.riskkernel.verification.ReadOnlyVerifier`.
    """
    history = events_from_records(store.read_all())
    head = store.head()
    feed = LedgerFillAccountingFeed(
        store,
        component=_COMPONENT,
        after_sequence=0 if head is None else head.sequence_number,
    )
    return ReadOnlyVerifier(
        connector=view,
        expectation_source=LedgerExpectationSource(history, view, fill_accounting=feed),
        tolerances=VerificationTolerances(
            balance_tolerance=MoneyMicros(
                config.risk.verification_balance_tolerance_micros
            ),
            position_tolerance=ContractCentis(
                config.risk.verification_position_tolerance_centis
            ),
        ),
        dispatcher=dispatcher,
        ledger_writer=writer,
    )


def _build_kill_integration(
    config: WindbreakConfig,
    history: tuple[Event, ...],
    mode_machine: ModeStateMachine,
    writer: _SqliteKernelLedgerWriter,
    clock: Callable[[], int],
    dispatcher: AlertDispatcher,
    reservations: ReservationLedger,
    directive_sink: DirectiveSink,
) -> KillIntegration:
    """Compose the always-on PAPER loop's kill switch and triggers (issue #441).

    The same composition ``windbreak.main._build_risk_kernel`` builds for
    ``windbreak run --process riskkernel``, over *this* loop's own seams: a
    :class:`~windbreak.riskkernel.kill.KillSwitch` replayed from the loop's
    hash chain, a :class:`~windbreak.riskkernel.kill.KillFileWatcher` over
    ``config.ops.state_dir``, and a
    :class:`~windbreak.riskkernel.kill.ReconciliationMismatchMonitor` at
    ``config.risk.kill_after_consecutive_mismatches``.

    Until this existed the PAPER loop -- the process that actually trades --
    passed ``kill_integration=None``, so ``windbreak kill --state-dir DIR``
    wrote a ``KILL`` file nothing polled and the configured auto-kill threshold
    bound nothing. #144 wired the kill switch into the CLI's *other* kernel and
    was closed as done; the RUNBOOK went on documenting a control the running
    loop did not honour. For a safety-critical kill switch an inert one is worse
    than none.

    Four composition decisions, each deliberate:

    * **Replayed, not fresh.** :meth:`KillSwitch.from_events` restores the
      monotonic kill sequence from ``history``, so a post-restart kill still
      increments rather than reissuing sequence 1 and letting a stale re-arm
      phrase unlock it. Driving the machine itself back to ``KILLED`` is
      :meth:`RiskKernel.from_events`'s job, so the two never race to transition
      the one machine.
    * **The state dir is read, never created.** A missing directory is exactly
      "the operator has asked for nothing", which
      :meth:`KillFileWatcher.poll_once` already reads correctly from a
      presence check. Creating it here would put a filesystem side effect in
      every ``build_paper_deps`` call for no control it buys: a mistyped
      ``state_dir`` is equally unreachable created or not. A kill from a
      non-file trigger still drops the ``KILL`` file, and
      :meth:`KillSwitch._write_kill_file` creates the directory then.
    * **The reservation ledger is wired.** A killed loop must hold no live
      capital reservation, so the switch is handed the very ledger the approval
      pipeline reserves against and releases every active one on kill.
    * **The directive sink is wired** (issue #480). Until it was, the one
      :class:`CancelAllDirective` a kill wrote was consumed by nothing here or
      in ``_build_risk_kernel``, so resting-order cancellation on kill was an
      audit record rather than an effect -- and an order resting on the book is
      not a held position, it is a live instruction that can still fill after
      the operator has killed the system and walked away. The seam looked
      blocked because the *gateway* is built after this point and exposes no
      ``submit(directive)``; the resolution is that it never needed to. What a
      cancel-all needs is the venue, and :func:`build_paper_deps` builds the
      exchange *before* the kernel, so a
      :class:`~windbreak.order_gateway.cancel_all.VenueCancelAllSink` over it
      can be handed in here with no late binding and no mutable holder. The
      directive still crosses the kernel's seam as data (SPEC S5): nothing in
      ``windbreak/riskkernel/`` names a connector type.

    Args:
        config: The configuration supplying ``ops.state_dir`` and the
            consecutive-mismatch auto-kill threshold.
        history: The replayed ledger history the kill sequence is restored from.
        mode_machine: The one LOCKED mode machine the switch drives to
            ``KILLED``.
        writer: The seam every kill-path event is recorded through.
        clock: The injected epoch-second clock ``KillEngaged.epoch`` is stamped
            at, so a kill is dated on the same timeline the tick reads.
        dispatcher: The run's one alert root the ``HALT_KILL`` page is delivered
            through (issue #444), so a kill reaches the sinks the operator
            configured rather than a second, log-only path.
        reservations: The capital ledger whose active reservations a kill
            releases.
        directive_sink: The order-gateway-side consumer that turns the kill's
            one cancel-all directive into venue cancellations (issue #480).

    Returns:
        The composed :class:`~windbreak.riskkernel.kill.KillIntegration`.
    """
    state_dir = Path(config.ops.state_dir).expanduser()
    switch = KillSwitch.from_events(
        history,
        mode_machine,
        writer,
        dispatcher,
        reservation_ledger=reservations,
        directive_sink=directive_sink,
        state_dir=state_dir,
        clock=clock,
    )
    return KillIntegration(
        switch=switch,
        watcher=KillFileWatcher(switch, state_dir),
        monitor=ReconciliationMismatchMonitor(
            switch, threshold=config.risk.kill_after_consecutive_mismatches
        ),
    )


def _build_approval(
    store: SqliteLedgerStore,
    config: WindbreakConfig,
    key: bytes,
    view: ReadOnlyVenueView,
    clock: Callable[[], int],
    dispatcher: AlertDispatcher,
    directive_sink: DirectiveSink,
) -> tuple[KernelApproval, RiskKernel]:
    """Wire the real kernel + approval pipeline into a `KernelApproval` seam.

    The kernel tracks PAPER mode (so its ledgered evaluation stamps PAPER) and
    shares the one ephemeral signing key with the gateway. The same hash-chained
    ``store`` is wired as the kernel's ``gate_plan_store`` (issue #185), so a
    PAPER -> LIVE_MICRO promotion reads its three thresholds from the
    pre-registered gate plan on the ledger, failing closed when none is
    registered.

    Issue #353 additionally wires a real read-only verifier and the tick's own
    injected ``clock``, so every cycle the kernel runs is stamped at the same
    instant the rest of the tick reads -- a snapshot aged against an unrelated
    wall clock could go stale against ``now_epoch_s`` for no real reason.

    Issue #441 gave it the kill wiring this docstring used to declare out of
    scope (see :func:`_build_kill_integration`), and rebuilt the kernel through
    :meth:`RiskKernel.from_events` rather than the bare constructor, so an
    engaged kill is recovered from the loop's own hash chain on restart --
    durable ledgered state, not a file that can be deleted. The polling that
    makes the wiring real is :func:`_kill_stage`, one call per tick.

    Args:
        store: The ledger both the kernel and the pipeline record through.
        config: The configuration whose hash is stamped into minted tokens.
        key: The ephemeral 32-byte signing key.
        view: The read-only venue view the verification cycle observes through.
        clock: The injected epoch-second clock the verification cycle stamps
            its snapshots at.
        dispatcher: The composed alert root the verification cycle's alerts are
            delivered through (issue #444).
        directive_sink: The order-gateway-side consumer a kill's cancel-all
            directive is delivered through (issue #480), forwarded to
            :func:`_build_kill_integration`.

    Returns:
        The composed :class:`KernelApproval` seam and the kernel inside it, so
        the tick can drive that kernel's verification cycle and read its mode.
    """
    writer = _SqliteKernelLedgerWriter(store)
    mode_machine = ModeStateMachine(
        mode_ceiling=Mode.from_config(config.mode_ceiling), mode=Mode.PAPER
    )
    ledger = ReservationLedger(writer)
    history = events_from_records(store.read_all())
    kernel = RiskKernel.from_events(
        history,
        writer,
        mode_machine=mode_machine,
        verifier=_build_verifier(store, config, view, writer, dispatcher),
        clock=clock,
        gate_plan_store=store,
        kill_integration=_build_kill_integration(
            config,
            history,
            mode_machine,
            writer,
            clock,
            dispatcher,
            ledger,
            directive_sink,
        ),
    )
    issuer = TokenIssuer.from_key_material(key)
    pipeline = ApprovalPipeline(ledger, issuer, config_hash=config_hash(config))
    return KernelApproval(kernel, pipeline), kernel


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


def _research_ledger_state(
    records: tuple[LedgerRecord, ...], *, configured_micros: int
) -> tuple[int, Mapping[str, int]]:
    """Fold this ledger's research rows, failing closed *on the spend* (#442).

    Both folds refuse a row they cannot read rather than skipping it, because a
    skipped charge would undercount the day and re-open a ceiling that should be
    shut. What that refusal must not do is take the process with it. A malformed
    row can arrive from a schema migration or an external tool, it cannot be
    removed from an append-only hash-chained ledger, and
    :func:`_build_research_budget` is called from :func:`build_paper_deps` -- so
    letting the ``ValueError`` escape would stop the loop from *composing*: no
    heartbeat, no equity sample, no reconciliation, and no kill handling, under
    a ``restart: on-failure`` supervisor that would retry it forever. A loop
    that cannot start cannot honour a kill file; a loop that runs with research
    disabled can.

    So the refusal is caught here and turned into the strictest budget there is:
    a :data:`_UNREADABLE_LEDGER_CEILING_MICROS` (zero) ceiling opening on an
    empty day counter. Every market then halts on the budget and ledgers a
    ``ResearchBudgetHalted`` row saying so, no research money is spent, and the
    rest of the tick keeps running. The cause is logged as a ``CRITICAL`` on
    every rebuild -- at startup and at the head of every tick -- because a loop
    that cannot read its own spend history is an incident, not a mode.

    Args:
        records: This ledger's rows, in append order.
        configured_micros: The ceiling from configuration, used when the ledger
            carries no cap row (keyword-only).

    Returns:
        The ``(per_day_micros, spend_by_day)`` pair a budget opens with.
    """
    try:
        return (
            effective_per_day_micros(records, configured_micros=configured_micros),
            spend_by_day_from_records(records),
        )
    except ValueError as exc:
        _LOGGER.critical(
            "research spend history unreadable, opening a zero ceiling: %s",
            exc,
            extra={"component": _COMPONENT},
        )
        return (_UNREADABLE_LEDGER_CEILING_MICROS, {})


def _build_research_budget(
    store: SqliteLedgerStore, config: WindbreakConfig
) -> ResearchBudget:
    """Build the loop's research spend guard from configuration *and the ledger*.

    Config supplies the per-forecast and page ceilings, and there is deliberately
    no way to inject a budget from outside: that is what makes an unlimited or
    absent budget unrepresentable rather than merely discouraged. All ceilings
    are already scaled integers, so they pass through untouched -- no arithmetic,
    and therefore no float, enters this path.

    Two things come off the **ledger** rather than the config, and both are
    issue #442 (see :mod:`windbreak.scheduler.research_spend`):

    * **The day's spend so far.** Folded from this ledger's
      ``ResearchSpendRecorded`` rows, so a process that restarts mid-day resumes
      on the day's real total. Without it the per-UTC-day ceiling was a
      per-process ceiling, and ``restart: on-failure`` makes the process count
      per day unbounded.
    * **The per-UTC-day ceiling itself**, when an operator has changed it at
      runtime with ``windbreak set-research-budget``. The configured value --
      itself overridable at startup by ``windbreak run
      --research-per-day-micros`` -- is the fallback, so a deployment that never
      runs the verb is unaffected.

    Called by :func:`build_paper_deps` at startup *and* by
    :func:`_refreshed_budget` at the head of every tick, so both facts are
    re-read while the loop runs rather than frozen at process start.

    An unreadable ``ResearchSpendRecorded`` / ``ResearchBudgetCapSet`` row does
    **not** propagate: :func:`_research_ledger_state` turns it into a zero
    ceiling on an empty counter, so the loop composes, beats, and refuses to
    spend rather than failing to start. A negative *configured* ceiling still
    aborts, below -- that is an operator's own YAML, correctable without
    touching an append-only ledger, and a loop that cannot determine its ceiling
    at all must not run.

    Args:
        store: The ledger store a charge or a fail-closed halt is recorded to,
            and the durable state both are folded back out of.
        config: The active configuration supplying the ceilings.

    Returns:
        A research budget opened on the ledger's own day counter and ceiling.

    Raises:
        ValueError: If any configured ceiling is negative -- aborting rather
            than degrading to an unenforceable budget.
    """
    caps = config.forecast.budget
    per_day_micros, opening_spend_by_day = _research_ledger_state(
        tuple(store.read_all()), configured_micros=caps.per_day_micros
    )
    return ResearchBudget(
        per_forecast_micros=caps.per_forecast_micros,
        per_day_micros=per_day_micros,
        max_pages=caps.max_pages,
        ledger=_SqliteBudgetLedgerWriter(store),
        opening_spend_by_day=opening_spend_by_day,
    )


def _build_provider_gate(
    report_dir: Path, config: WindbreakConfig
) -> ProviderTrackRecordGate:
    """Build the loop's per-provider track-record gate from artifact + config.

    The gate is a *read model* over M6's evaluation output (SPEC S13/S16, S19):
    it consumes each provider's resolved-forecast count and Brier skill from the
    :data:`PROVIDER_TRACK_RECORD_FILENAME` artifact beside the loop's other
    evaluation artifacts, and never recomputes a score itself.

    Two policies are deliberate and both fail *closed*:

    * **Bootstrap (artifact absent).** Before the first evaluation pass has
      written a record, every provider is unproven, so every full forecast is
      held back from live eligibility. That is the honest reading of "no
      measured edge yet", and it is the direction a mistake must point: a loop
      that granted live eligibility while no track record existed would let an
      entirely unmeasured provider back a live order.
    * **Malformed artifact.** :func:`parse_track_records` raises, and this
      function lets the raise out, aborting startup. Degrading to an empty
      source would *also* withhold eligibility, but it would do so while
      silently discarding an operator's evaluation output -- a broken pass and
      a genuinely empty one must not look the same from the outside.

    The thresholds come from ``config.forecast.provider_gate`` -- the dedicated
    per-provider knob whose defaults mirror ``config.evaluation``'s own
    promotion bars (``min_resolved_for_calibration`` /
    ``brier_skill_required_ppm``) -- and are never defaulted away to the gate's
    own module-level constants, so an operator who raises the bar gets the bar
    they raised.

    Args:
        report_dir: The evaluation-artifact directory the track-record document
            is resolved inside.
        config: The active configuration supplying the two thresholds.

    Returns:
        The process-lived per-provider live-eligibility gate.

    Raises:
        ValueError: If the artifact exists but is not a readable, strictly
            integer track-record document -- aborting startup rather than
            trading on a record the loop could not parse.
    """
    artifact = report_dir / PROVIDER_TRACK_RECORD_FILENAME
    records = (
        parse_track_records(artifact.read_text(encoding="utf-8"))
        if artifact.is_file()
        else {}
    )
    bars = config.forecast.provider_gate
    return ProviderTrackRecordGate(
        InMemoryTrackRecordSource(records.values()),
        min_resolved=bars.min_resolved,
        min_brier_skill_ppm=bars.min_brier_skill_ppm,
    )


def _build_paper_exchange(
    books_dir: Path,
    clock: Callable[[], int],
    *,
    market_data: MarketDataSource | None,
    live_ticker: str | None,
) -> PaperExchange:
    """Build the tick's exchange: replayed fixture books, or the venue's live ones.

    Both modes produce a :class:`~windbreak.connector.paper.PaperExchange`, so
    every consumer :func:`build_paper_deps` wires downstream -- gateway,
    reconciler, verification view -- takes the one object this returns and
    needs no knowledge of which mode it is in. That is what makes the live wire
    *total*: there is no second exchange for a consumer to be left holding.

    The two modes differ in exactly one further respect, and it is a
    fail-closed one. The fixture path anchors the recording to this run's clock
    (issue #369): a committed book's frozen literals sit permanently outside
    every ttl -- or, for a recording dated after the run's clock, permanently
    in the future -- so without an anchor ``quote_freshness`` could only ever
    veto. The anchor shifts every book and print by one offset, so the
    recording's internal timing survives intact and a book genuinely ages as
    the replay runs.

    A live book has no such problem: it already carries the instant the venue
    was actually observed, and shifting that would fabricate freshness the
    venue never claimed. Live mode therefore passes **no** ``replay_anchor``,
    and the venue's ``fetched_at`` reaches the freshness check untouched --
    which is what leaves that check able to genuinely veto a stale book.

    Args:
        books_dir: The fixture directory. In fixture mode it supplies books,
            markets, and fees; in live mode only its *account* fixtures
            (opening balances and balance semantics) are read, because the
            whole point of live mode is that the market data is the venue's.
        clock: The injected epoch-second clock, read for the exchange's own
            observation timeline.
        market_data: The read-only venue surface, or ``None`` for fixtures.
        live_ticker: The single market a live session binds to, or ``None`` for
            fixtures. It names the venue's *universe* for a live session, not
            the market the tick will trade: since issue #345 the loop screens
            whatever markets the exchange offers, so a named live market that
            fails the §16 screen is not traded. Binding one market is the
            venue-surface's own limit today (``LiveBookPaperExchange`` holds a
            single ticker), not a second selection knob -- which is why
            ``--paper-live-ticker`` stayed one flag.

    Returns:
        The wired exchange -- a
        :class:`~windbreak.connector.paper.LiveBookPaperExchange` in live mode.

    Raises:
        ValueError: If exactly one of ``market_data`` / ``live_ticker`` is
            supplied. Half a live configuration must not degrade to fixtures:
            an operator who named a live market and silently got recorded books
            would be reading a paper tape while believing it was the venue.
    """

    def observed_at() -> datetime:
        """Return this run's clock reading as a timezone-aware UTC instant.

        Returns:
            The injected clock's current reading, in UTC.
        """
        return datetime.fromtimestamp(clock(), UTC)

    if market_data is None:
        if live_ticker is not None:
            raise ValueError(
                f"live_ticker={live_ticker!r} was named without a `market_data` "
                "source to read it from; supply both or neither"
            )
        return PaperExchange.from_fixture_dir(
            books_dir, clock=observed_at, replay_anchor=observed_at()
        )
    if live_ticker is None:
        raise ValueError(
            "`market_data` was supplied without a `live_ticker` naming the market "
            "to trade; supply both or neither"
        )
    return LiveBookPaperExchange.from_account_dir(
        books_dir, market_data=market_data, ticker=live_ticker, clock=observed_at
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
    market_data: MarketDataSource | None = None,
    live_ticker: str | None = None,
    provider_http: LiveProviderHttp | None = None,
    dispatcher: AlertDispatcher | None = None,
) -> PaperTickDeps:
    """Assemble every real component one PAPER tick runs against.

    Loads a :class:`~windbreak.connector.paper.PaperExchange` from ``books_dir``,
    opens the hash-chained ledger at ``ledger_path``, mints one ephemeral 32-byte
    signing key shared by the kernel and gateway (SPEC S10.6), and wires the real
    approval seam, gateway (boot-recovered), and reconciler over them.

    Supplying ``market_data`` with ``live_ticker`` swaps the fixture books for a
    venue's live ones -- real prices in, paper money out (issue #343). The swap
    is *total* by construction: :func:`_build_paper_exchange` returns the single
    exchange object that the gateway's submitter, its status and reconciliation
    sources, the :class:`~windbreak.order_gateway.reconciler.Reconciler`, the
    read-only verification view, and :attr:`PaperTickDeps.exchange` are all
    built from, so no consumer can be left reading a different venue from the
    one the loop trades against. Omitting both leaves every existing caller
    byte-identical.

    The exchange is given ``clock`` as its observation clock, so its status
    attestation is stamped on the same timeline the tick judges freshness on
    (issue #342) -- and, in fixture mode only, as its ``replay_anchor`` (issue
    #369; see :func:`_build_paper_exchange` for why a live book must never be
    re-dated). Both readings come from the one injected clock, so the books, the
    status, and the evaluation can never be judged against three unrelated
    timelines.

    It builds the process's one real §16 :class:`~windbreak.screener.Screener`
    (issue #345), reading its thresholds from ``config.screener`` and writing its
    verdicts into the same hash-chained store every other stage appends to. That
    screener is what replaced this function's old
    ``ticker = next(iter(exchange.markets))``: the market a tick forecasts is now
    decided each tick, from evidence, rather than once at composition time from
    a mapping's iteration order. Like the budget and the provider gate it has no
    injection parameter, so there is no door an unscreened loop can arrive
    through.

    ``dispatcher`` is the loop's alert root, and it is the *only* door alerting
    arrives through (issue #444). Unlike the screener, the budget, and the
    provider gate -- each of which this function builds from config so no
    unguarded loop has an injection door -- an alert root cannot be built here:
    turning ``config.alerts`` into live channels means resolving each sink's
    ``*_env`` destination against the real environment, and
    :func:`windbreak.main._build_alert_dispatcher` is deliberately the single
    place that happens, so a destination (an ntfy topic, a webhook URL with a
    token in its path) has exactly one code path that can ever hold it. Omitting
    it composes the log-only fallback, which is the *documented* behaviour of a
    deployment that declares no deliverable sink -- not, as before, the only
    behaviour available to one that declares several.

    It also builds the process's one per-provider track-record gate from
    ``report_dir``'s M6 artifact and ``config.forecast.provider_gate`` (see
    :func:`_build_provider_gate`). Like the research budget there is deliberately
    no ``provider_gate`` parameter: config plus artifact are the only sources, so
    an ungated -- and therefore unmeasured-edge-granting -- loop has no injection
    door to arrive through.

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
        market_data: The read-only live venue surface, or ``None`` (the default)
            to read the fixture directory's recorded books.
        live_ticker: The single market a live session trades. Required with
            ``market_data`` and meaningless without it.
        provider_http: The live forecast-provider HTTP seams, or ``None`` (the
            default) to replay ``cassette_path``. Required by, and only by,
            ``forecast.provider_transport.mode == "live"``; see
            :func:`_resolve_forecast_transport`.
        dispatcher: The composed alert root the verification cycle's mismatch
            and unknown-jurisdiction alerts are delivered through, or ``None``
            (the default) for the log-only fallback a deployment with no
            deliverable sink documents.

    Returns:
        A fully wired :class:`PaperTickDeps`.

    Raises:
        ValueError: If exactly one of ``market_data``/``live_ticker`` is given,
            if the configured provider-transport mode is unknown or disagrees
            with whether ``provider_http`` was supplied, if a configured ceiling
            or list price is not positive, if
            ``config.screener.max_candidates_per_tick`` is below one, or if the
            track-record artifact exists but cannot be read as a strict integer
            document -- each way refusing to start rather than running
            unguarded.
    """
    resolved_clock = clock if clock is not None else _default_clock
    # Selected first, before the ledger database or any exchange session
    # exists, so a misconfigured transport aborts startup without leaving
    # half-built durable state behind.
    corpus = _resolve_replay_corpus(config, provider_http)
    _log_replay_corpus(config, corpus)
    transport, live = _resolve_forecast_transport(
        config, cassette_path, provider_http, corpus
    )
    # Resolved and folded here, beside the other two evidence decisions and
    # before the ledger database or any exchange session exists, so an operator
    # reads what this deployment can produce from the first lines of its log
    # rather than inferring it from a tick that never forecasts (issue #485).
    evidence_source, resolved_research_tools = _resolve_research_tools(
        research_tools, ledger_path, config, provider_http, corpus
    )
    _log_research_evidence(evidence_source)
    # The exchange must observe on the same clock the tick reads, or its status
    # attestation drifts against `now_epoch_s` and `exchange_status_ok` judges
    # freshness against two unrelated timelines (issue #342).
    # Validated before the ledger database exists, alongside the transport
    # check above, so a bound that can never forecast refuses to start rather
    # than leaving an always-idle loop behind a half-built durable state.
    require_candidate_bound(config.screener.max_candidates_per_tick)
    exchange = _build_paper_exchange(
        books_dir, resolved_clock, market_data=market_data, live_ticker=live_ticker
    )
    store = SqliteLedgerStore(ledger_path)
    key = secrets.token_bytes(_SIGNING_KEY_BYTES)
    # The verification path gets a view, never the exchange: `PaperExchange`
    # exposes `place_order`/`cancel_order` alongside its reads, and the
    # read-only cycle must not be able to reach them (SPEC S1.1 invariant 3).
    verification_view = ReadOnlyConnectorView(exchange)
    # Books each execution into the ledger exactly once, under this loop's own
    # component -- the label `_build_verifier`'s feed is told to trust. The
    # bookkeeper is built before the verifier so no execution can slip between
    # the baseline capture and the first booking (issue #365).
    fill_bookkeeper = LedgerFillBookkeeper(store, exchange, component=_COMPONENT)
    approval, kernel = _build_approval(
        store,
        config,
        key,
        verification_view,
        resolved_clock,
        _resolved_dispatcher(dispatcher),
        # The kill path gets the raw exchange, not `verification_view`: the
        # view exists to keep the read-only cycle away from `cancel_order`,
        # and cancelling resting orders is this seam's entire purpose (#480).
        VenueCancelAllSink(exchange),
    )
    gateway = _build_gateway(exchange, store, key, resolved_clock, ledger_path)
    reconciler = Reconciler(
        gateway,
        ledger_reader=store,
        reconciliation_source=exchange,
        ledger_writer=SqliteGatewayLedgerWriter(store),
    )
    return PaperTickDeps(
        config=config,
        screener=Screener(
            config.screener,
            ScreenLedgerWriter(store, component=_COMPONENT),
            # The screener measures a market's resolution horizon against
            # "now", so it reads the very clock the tick judges everything else
            # on -- never the wall clock, which would let an injected-clock run
            # screen against a different timeline than it trades on.
            clock=lambda: datetime.fromtimestamp(resolved_clock(), UTC),
        ),
        store=store,
        exchange=exchange,
        verification_view=verification_view,
        fill_bookkeeper=fill_bookkeeper,
        gateway=gateway,
        reconciler=reconciler,
        approval=approval,
        kernel=kernel,
        verification_key=key,
        transport=transport,
        research_tools=resolved_research_tools,
        report_dir=report_dir,
        clock=resolved_clock,
        budget=_build_research_budget(store, config),
        provider_gate=_build_provider_gate(report_dir, config),
        provider_factory=build_provider_factory(config, live=live),
    )


# --- the single tick ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TickOutcome:
    """The summary result of one :func:`run_single_tick` call.

    Since issue #345 a tick runs over a *set* of screened markets rather than
    one hardcoded ticker, so the per-market figures are reported per market
    (``forecast_ids``) or summed across them (``intent_count``,
    ``filled_centis``). ``candidate_tickers`` and ``forecast_ids`` are separate
    fields rather than one, because a tick can screen a market in and still
    produce no forecast for it -- the budget can halt mid-universe.

    Attributes:
        beat: The 1-based tick sequence number.
        candidate_tickers: The markets that passed the screen, in the ascending
            ticker order the walk would take them in. This is the *screened-in
            set*, not a record of what ran: a budget halt stops the walk, so
            markets after the halting one appear here having never reached
            :func:`_run_candidate` at all. Empty when the whole universe
            screened out, which is a tick that correctly forecast nothing.
        forecast_ids: One forecast id per market this tick actually forecast, in
            processing order. Shorter than ``candidate_tickers`` exactly when
            research halted part-way through the universe -- so comparing the
            two lengths, not reading ``candidate_tickers`` alone, is what tells
            an operator how far the tick got.
        intent_count: How many normalized intents the selector emitted, summed
            over every candidate.
        filled_centis: The quantity filled through the gateway this tick, in
            contract-centis, summed over every candidate (``0`` whenever the
            kernel vetoed every intent).
        equity_micros: The sampled account equity this tick, in micros.
        research_halted: Whether this tick's research was halted fail-closed on
            a budget ceiling. When ``True`` the tick stopped walking its
            candidates at that point, so ``forecast_ids`` covers only the
            markets reached before the ceiling bit.
        kernel_halted: Whether the Risk Kernel is in ``HALT`` at the end of this
            tick -- today only a verification ``BREACH`` puts it there (issue
            #32). A halted kernel vetoes every later intent, so an always-on
            driver must treat this as "stop and get a human", not as a
            transient. Reported per tick rather than raised, because the tick
            must still finish ledgering its heartbeat, equity, and positions:
            the halt is exactly when that audit trail matters most.
    """

    beat: int
    candidate_tickers: tuple[str, ...]
    forecast_ids: tuple[str, ...]
    intent_count: int
    filled_centis: int
    equity_micros: int
    research_halted: bool = False
    kernel_halted: bool = False


def _screen_stage(deps: PaperTickDeps) -> tuple[MarketCandidate, ...]:
    """Screen the venue's market universe into this tick's bounded candidates.

    Runs *first*, before any paid stage, which is the whole point: the four §16
    filters are pure integer comparisons over a market's own metadata and its
    book, so deciding what to research costs no research money. Since issue #399
    every ensemble vote books real spend, so a screen that ran after the
    forecast -- or a loop with no screen at all -- would pay for markets it was
    about to reject.

    Each examined market's verdict is ledgered by the screener itself, as a
    ``ScreenDecisionRecorded`` row (issue #159), eligible or not.

    Args:
        deps: The tick's dependency bundle.

    Returns:
        The screened candidates, in ascending ticker order, bounded by
        ``config.screener.max_candidates_per_tick``.
    """
    return screen_universe(
        deps.exchange,
        deps.screener,
        max_candidates=deps.config.screener.max_candidates_per_tick,
    )


def _snapshot_stage(deps: PaperTickDeps, candidate: MarketCandidate) -> None:
    """Ledger the snapshot event for the book this candidate was screened on.

    The book is the candidate's own rather than a fresh read. One market, one
    observation per tick: re-reading here would ledger a *later* book than the
    screen's depth floor was measured against, so the audit trail would claim a
    market was screened in on liquidity that is not the liquidity recorded --
    and on a live venue the two genuinely differ.

    Args:
        deps: The tick's dependency bundle.
        candidate: The screened market whose book is being recorded.
    """
    deps.store.append(
        market_snapshot_event_to_record(
            ticker=candidate.ticker,
            order_book=candidate.order_book,
            component=_COMPONENT,
        )
    )


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
    deps: PaperTickDeps, candidate: MarketCandidate, created_at: datetime
) -> ForecastRecord | None:
    """Run the forecast pipeline and ledger the forecast event.

    The ledgered ``ForecastCreated`` carries the forecast's ``research_cost_micros``
    and ``market_price_baseline_pips`` (issue #188), the two fields the weekly
    evaluation/cost-meter fold reads verbatim off the payload.

    The vote stage is driven from ``config.forecast.vote_ensemble`` -- the
    authoritative vote-stage ensemble (ADR-0006), threaded here so an operator's
    configured ensemble is the one the PAPER loop actually calls (issue #294).
    It is passed through verbatim, never defaulted away: the default config's
    ensemble is provenance-identical to the engine's own
    :data:`~windbreak.forecast.providers.DEFAULT_VOTE_ENSEMBLE`, so the default
    path is byte-identical, while an operator who empties the field gets zero
    votes and a fail-closed abstention rather than a silent fallback to a
    triple they configured away.

    Live eligibility additionally runs through the bundle's per-provider
    track-record gate (issue #305), so SPEC S13/S16's "earned, never granted"
    promotion bar is enforced by the PAPER loop itself rather than merely being
    available at the ``run_pipeline`` seam. A provider the M6 track record has
    not proven cannot back a live order: its votes still run and still cost
    (that is how a paper track record accrues at all), only the record's
    ``eligible_for_live`` is forced ``False``, and the reason is ledgered as a
    ``ProviderGateHeld`` row.

    Research runs under the bundle's budget (issue #339). On a budget breach the
    pipeline raises, and this stage answers ``None`` rather than fabricating a
    forecast: in a hash-chained audit ledger an honest gap beats a
    ``ForecastCreated`` row for a tick where the engine provably never ran. The
    ``except`` deliberately names exactly the two budget errors -- widening it
    would dress an unrelated pipeline failure up as a benign budget halt. It
    ledgers nothing itself, because the budget's own ledger writer has already
    appended the durable ``ResearchBudgetHalted`` row before raising.

    Both the market and the baseline book come off the ``candidate`` the screen
    produced (issue #345), never from a fresh exchange read: the forecast is
    struck against the very observation the market was screened in on.

    Args:
        deps: The tick's dependency bundle.
        candidate: The screened market being forecast, carrying the book the
            baseline is struck against.
        created_at: The injected creation instant, for determinism.

    Returns:
        The produced forecast record, or ``None`` when research halted
        fail-closed on the budget -- in which case neither a ``ForecastCreated``
        nor any ``ProviderVoteRecorded`` row is appended for this market.
    """
    market = candidate.market
    order_book = candidate.order_book
    baseline = BaselineQuoteSnapshot(
        snapshot_id=f"{candidate.ticker}-{int(order_book.fetched_at.timestamp())}",
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
            ensemble=deps.config.forecast.vote_ensemble,
            provider_gate=deps.provider_gate,
            # Bound to `deps.transport` at call time, never at composition
            # time, so a bundle whose transport was swapped via
            # `dataclasses.replace` really votes through the swapped one.
            provider_factory=lambda member: deps.provider_factory(
                deps.transport, member
            ),
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
    _ledger_provider_gate_holds(deps, forecast, vote_ledger)
    return forecast


def _ledger_provider_gate_holds(
    deps: PaperTickDeps, forecast: ForecastRecord, vote_ledger: InMemoryForecastLedger
) -> None:
    """Fold a buffered provider-gate hold into a durable ``ProviderGateHeld`` row.

    The pipeline buffers at most one :data:`PROVIDER_GATE_HELD_EVENT` per run
    (only a full run that reaches aggregation screens providers at all, and it
    screens them once); this composition-root fold stamps it with the tick's own
    ``forecast_id`` and ``market_ticker`` -- neither of which the forecast-side
    payload carries -- and appends it to the durable store after the tick's
    ``ForecastCreated`` and its ``ProviderVoteRecorded`` rows. Ordering the hold
    *after* the vote rows keeps the audit trail in causal order: the votes
    happened, then their providers were screened.

    A run that abstains before the vote stage buffers no such event, so a
    default offline tick's ledger is unchanged by this fold.

    Args:
        deps: The tick's dependency bundle (its ``store`` receives the row).
        forecast: The tick's forecast record, supplying the correlating
            ``forecast_id`` and ``market_ticker``.
        vote_ledger: The in-memory ledger the pipeline buffered the hold into.
    """
    for event in vote_ledger.events_by_type(PROVIDER_GATE_HELD_EVENT):
        payload = event.payload
        deps.store.append(
            ProviderGateHeld(
                component=_COMPONENT,
                forecast_id=forecast.forecast_id,
                market_ticker=forecast.market_ticker,
                unproven_providers=cast("str", payload["unproven_providers"]),
                unproven_count=cast("int", payload["unproven_count"]),
                min_resolved=cast("int", payload["min_resolved"]),
                min_brier_skill_ppm=cast("int", payload["min_brier_skill_ppm"]),
            )
        )


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


def _held_markets(deps: PaperTickDeps) -> tuple[HeldMarket, ...] | None:
    """Read and price what the account holds, or refuse (issue #407).

    One venue read for the positions, then one per held ticker for its parent
    event -- ``event_ticker`` is a factual venue field, so asking the venue for
    it claims nothing the venue did not say. Each holding is priced by
    :func:`_position_value_micros`, the same fold
    :func:`_equity_and_positions_stage` uses, so exposure and equity can never
    disagree about what a position is worth.

    Two refusals, both returning ``None`` rather than an empty tuple:

    * :class:`~windbreak.connector.paper.TwoSidedPositionError` -- the venue
      declined to describe a ticker filled on both sides, because the only
      single-row answer is a netted one and a netted YES-plus-NO reports flat
      while both legs and their collateral are live (issue #361).
    * :class:`~windbreak.connector.paper.UnknownMarketError` -- a held ticker
      the venue will not describe, so its event and value cannot be attributed.

    An empty tuple is *not* a refusal: the venue answered and reported no rows,
    which its contract defines as genuinely flat.

    Read per candidate, at the moment that candidate is sized, rather than once
    per tick -- the same discipline issue #345 established for the balance read,
    and for the same reason. It is what makes a tick's markets see each other in
    *exposure* as well as in capital: an earlier candidate's fill is a position
    the venue reports by the time the next candidate is sized, so the second
    market's bucket exposure already includes the first. A once-per-tick read
    would have handed every market an identical, pre-tick account.

    Args:
        deps: The tick's dependency bundle.

    Returns:
        The priced holdings, or ``None`` when the venue could not describe them.
    """
    try:
        positions = deps.exchange.get_positions()
    except TwoSidedPositionError:
        return None
    held: list[HeldMarket] = []
    for position in positions:
        try:
            market = deps.exchange.get_market(position.ticker)
        except UnknownMarketError:
            return None
        held.append(
            HeldMarket(
                ticker=position.ticker,
                event_ticker=market.event_ticker,
                value_micros=MoneyMicros(_position_value_micros(position)),
            )
        )
    return tuple(held)


def read_candidate_exposure(
    deps: PaperTickDeps, candidate: MarketCandidate
) -> ExposureProjection | None:
    """Project one candidate's exposure from the venue's holdings, or refuse.

    The single entry point both the selector's notional caps and the kernel's
    ``concentration_limits`` read (issue #407), so the two enforce SPEC S9.9's
    "defense in depth" over the *same* evidence rather than over two
    independent guesses.

    Per candidate, not per tick. This is what closes issue #345's open
    acceptance criterion: two markets in one tick sharing a correlation bucket
    now bind on each other, because the first market's fill is a holding the
    venue reports before the second is sized. The candidate's event ticker
    comes straight off its already-screened
    :class:`~windbreak.connector.models.NormalizedMarket`, so no second venue
    read can disagree with the one the screen judged.

    Args:
        deps: The tick's dependency bundle.
        candidate: The screened market being sized.

    Returns:
        The projection, or ``None`` when the venue could not describe the
        account's holdings, or when the operator has declared no correlation
        bucket for the candidate or for something held.
    """
    held = _held_markets(deps)
    if held is None:
        return None
    return project_exposure(
        held,
        target_ticker=candidate.ticker,
        target_event_ticker=candidate.market.event_ticker,
        correlation=deps.config.correlation,
    )


def _position_input(
    deps: PaperTickDeps, ticker: str, exposure: ExposureProjection
) -> PositionReadModelInput:
    """Build the selector's capital/exposure input from the paper balances.

    The four exposure terms were hardcoded to zero, which left all five of the
    selector's SPEC S9.6 notional caps computing headroom against an account
    that looked untouched (issue #407). They now come from ``exposure``, which
    is required rather than optional precisely so no zeroed path survives: a
    tick that cannot prove its exposure never reaches this function, because
    :func:`_select_stage` declines first.

    ``notional_today`` stays zero here. Issue #415 fed the *risk kernel's*
    daily cap from :func:`read_notional_today_micros`, which is where the
    binding veto lives; the selector's own SPEC S9.6 daily-notional headroom
    term reads this input instead, and wiring it needs the tick's instant,
    which this per-candidate function is not given. Left as a stated gap rather
    than silently fed a clock read at sizing time.

    The balances are read per candidate, at the moment that candidate is sized
    (issue #345), so an earlier candidate's fill has already debited them. Both
    cross-market bounds now come from the same instant: capital from these
    balances, exposure from ``exposure``'s holdings.

    Args:
        deps: The tick's dependency bundle.
        ticker: The candidate market being sized, which stamps the snapshot id.
        exposure: The candidate's projected exposure from
            :func:`read_candidate_exposure`.

    Returns:
        The :class:`~windbreak.selector.types.PositionReadModelInput` the sizing
        stage reads.
    """
    available = deps.exchange.get_balances().available
    floor = MoneyMicros(deps.config.capital.floor_micros)
    above_floor = MoneyMicros(max(available.value - floor.value, 0))
    return PositionReadModelInput(
        snapshot_id=f"{ticker}-positions",
        equity_micros=available,
        above_floor_capital_micros=above_floor,
        total_deploy_cap_micros=above_floor,
        market_exposure=exposure.market_exposure,
        event_exposure=exposure.event_exposure,
        bucket_exposure=exposure.bucket_exposure,
        total_exposure=exposure.total_exposure,
        notional_today=MoneyMicros(0),
    )


def _unbucketable_decision(
    candidate: MarketCandidate, forecast: ForecastRecord
) -> SelectorDecision:
    """Return the decline for a market whose exposure could not be proven.

    A refusal, not a zero. Before issue #407 this candidate would have sized
    against ``correlation_tags=()`` and four zeroed exposure terms, so the
    per-bucket cap aggregated an empty peer set and passed -- a cap reporting
    success on evidence nobody held. Declining with a stated reason honors
    :class:`~windbreak.selector.types.SelectorDecision`'s contract that a
    verdict emitting no intents must still say why.

    Declining one candidate does not stop the walk: a sibling the operator
    *has* bucketed still sizes. The refusal is as narrow as the missing
    evidence.

    Args:
        candidate: The screened market that could not be bucketed.
        forecast: The forecast that would have been sized.

    Returns:
        A no-intent :class:`~windbreak.selector.types.SelectorDecision`.
    """
    return SelectorDecision(
        intents=(),
        reasons=(
            f"unprovable_exposure: no correlation bucket or holding evidence "
            f"for {candidate.ticker}",
        ),
        forecast_id=forecast.forecast_id,
        market_ticker=candidate.ticker,
        calibration_map_version=_CALIBRATION_MAP_VERSION,
    )


def _select_stage(
    deps: PaperTickDeps,
    candidate: MarketCandidate,
    forecast: ForecastRecord,
    created_at: datetime,
    exposure: ExposureProjection | None,
) -> SelectorDecision:
    """Run the selector over one candidate's inputs and ledger the decision event.

    ``correlation_tags`` and ``bucket_peers`` were ``()`` and defaulted, so
    ``effective_buckets(()) == ()`` and ``aggregate_bucket_exposure`` returned
    ``(0, None)`` -- the per-bucket cap could not bind however many markets a
    tick screened (issue #407). Both now come from ``exposure``, whose tags are
    the operator's own declaration rather than a derivation.

    That resolves what issue #345 left open here. The previous note in this
    docstring was right to refuse to invent a ``source``: the tag's field
    admits only ``"llm"`` or ``"human"``, and a composition-root derivation is
    honestly neither. The answer was not a third source value but a different
    producer -- an operator declaring buckets in configuration *is* the human,
    so no provenance is invented and SPEC S9.9's "human-overridable, stored as
    data" is satisfied literally. Capital is no longer the only cross-market
    bound in force: exposure now binds alongside it, on the same per-candidate
    read.

    Args:
        deps: The tick's dependency bundle.
        candidate: The screened market being selected over, carrying the book it
            was screened on.
        forecast: The forecast under evaluation.
        created_at: The fee schedule's freshness stamp for this tick.
        exposure: The candidate's projected exposure from
            :func:`read_candidate_exposure`, or ``None`` when it could not be
            established -- which declines this candidate, not the whole tick.

    Returns:
        The selector's decision.
    """
    if exposure is None:
        decision = _unbucketable_decision(candidate, forecast)
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
    inputs = SelectorInputs(
        forecast=forecast,
        calibration_map_version=_CALIBRATION_MAP_VERSION,
        order_book=candidate.order_book,
        fee_model=FeeModelInput(
            model=deps.exchange.get_fee_model(candidate.ticker), as_of=created_at
        ),
        slippage_model=SlippageModelInput(
            model_id=_SLIPPAGE_MODEL_ID, per_contract_buffer_ppm=0
        ),
        positions=_position_input(deps, candidate.ticker, exposure),
        risk_config=RiskConfigInput(
            config=deps.config.risk, config_hash=config_hash(deps.config)
        ),
        correlation_tags=exposure.target_tags,
        bucket_peers=exposure.bucket_peers,
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
    deps: PaperTickDeps,
    candidate: MarketCandidate,
    decision: SelectorDecision,
    heartbeat_epoch_s: int,
    forecast: ForecastRecord,
    exposure: ExposureProjection | None,
) -> int:
    """Approve each emitted intent through the seam; route any minted token.

    Reads the exchange status once, here at decision time rather than at
    composition time, so its freshness is measured from a genuine observation
    (issue #342). The status *value* comes from the connector and is never
    synthesized, so a paused or closed exchange still vetoes.

    Threads the kernel's own latest verification snapshot onto the context
    (issue #353). This is not redundant with the kernel stamping it internally:
    ``RiskKernel.evaluate_intent`` stamps its snapshot on a private copy, but
    :meth:`~windbreak.riskkernel.reservations.ApprovalPipeline.approve`
    re-evaluates every check over the context handed *here*, so without this
    thread the three reconciliation checks would pass in the kernel and veto in
    the pipeline, and no token could ever mint. The snapshot is read straight
    off the kernel rather than re-derived, so both halves judge one identical
    observation. Before the first cycle it is ``None`` and everything still
    fails closed.

    Threads the two exposure figures issue #364 supplies, both out of evidence
    this tick itself produced: the start-of-day equity is read back from the
    ``EquitySampled`` rows already on the ledger -- bounded to those taken since
    the UTC day boundary rather than the whole log, since this runs every tick
    of an always-on loop (issue #370) -- and the visible depth from the book the
    snapshot stage just read. Both are read at the same clock
    reading the context is stamped with, so the UTC day the baseline is bucketed
    into is the day the evaluation happens on. On the day's first tick the
    ledger carries no sample yet -- the tick appends its own only after this
    stage -- so the baseline is genuinely ``None`` and ``daily_loss_limit``
    keeps vetoing rather than trading against an invented one.

    Threads the book's *own* ``fetched_at`` as the quote snapshot epoch (issue
    #369), taken from the very snapshot the participation cap is measured
    against, so ``quote_freshness`` compares the price's age with the instant
    it is being priced at. It used to receive this stage's ``now_epoch_s``,
    which made ``_is_stale(now, now, ttl)`` ``False`` for every ttl and the
    check incapable of ever vetoing -- worse than the absent datum it replaced,
    because it advertised the SPEC S7.3 guarantee without providing it. That
    reading only became honest once :func:`build_paper_deps` anchored the
    replay, so the recording's frozen literals age against this run's clock.

    Threads the venue's own clock (issue #377) rather than this stage's
    ``now_epoch_s``, so ``clock_skew_limit`` measures our clock against the
    venue's instead of against itself. ``PaperExchange.get_exchange_time``
    answers from the anchored replay timeline and deliberately does not renew
    itself per read -- a clock that did could never disagree. A replay that has
    run out of recording reports no clock at all rather than a stale one (issue
    #382; see :func:`read_exchange_clock_epoch_s`), so a run that outlives its
    recording vetoes for a stated reason instead of on accumulated drift.

    Threads the venue's own open position (issue #373), read here rather than
    taken from :func:`_equity_and_positions_stage`, which runs *after* this
    stage on purpose: the position an approval is proven against must be the
    one held before this tick's fills, not after them. A venue that refuses to
    describe its holding yields ``None`` and ``reduce_only_provable`` keeps
    vetoing -- see :func:`read_open_position_centis` for why zero would be a
    fabrication rather than a fallback.

    Threads the forecast's own ``created_at`` as the forecast epoch (issue
    #380) rather than this stage's ``now_epoch_s``, so ``forecast_freshness``
    compares the estimate's age with the instant it is being acted on. The two
    differ by exactly how long the forecast stage took, and that gap is the
    whole point of the second clock read documented below: a research run slow
    enough to outlive ``forecast_ttl_seconds`` must age its own output out.
    Stamped with this stage's clock instead, the forecast was zero seconds old
    however long it had taken, and the check could not veto at any age.

    Threads the trailing hour's routed-order count (issue #491), folded out of
    the ledger's own ``OrderTransitionLedgered`` rows at the same clock reading
    the context is stamped with, so the window the cap measures ends exactly
    where the evaluation happens. It was a hardcoded ``0``, which made
    ``velocity_limits`` evaluate ``0 + 1 > max_orders_per_hour`` on every tick --
    false for every configured maximum -- so the runaway-order gate could not
    veto at any order rate. It is read here rather than at tick start for the
    same reason the equity baseline is: a tick that has already routed for an
    earlier candidate must have that order counted against the next one.

    A market the exchange cannot resolve becomes ``None`` rather than an
    exception, so an unknown ticker vetoes the tick instead of aborting it.

    The instrument whitelist is this candidate's ticker alone, not the tick's
    whole candidate set (issue #345). Each approval is proven against exactly
    the one market it would trade, so widening the universe cannot widen what
    any single token authorizes.

    This stage runs once per candidate, so a multi-market tick observes the
    exchange status -- and ledgers an ``ExchangeStatusObserved`` row -- once per
    candidate rather than once per tick. That repetition is deliberate: the
    venue can pause part-way through a universe walk, and every approval must be
    proven against a status observed at *its own* evaluation time. Observing
    once at the top of the tick and reusing it would let the last candidate in a
    long walk trade on a reading taken before the pause, with
    ``exchange_status_ok`` unable to notice.

    Args:
        deps: The tick's dependency bundle.
        candidate: The screened market being approved for, carrying the book
            whose shallower visible side bounds the participation cap.
        decision: The selector's decision carrying any emitted intents.
        heartbeat_epoch_s: The instant an earlier stage observed the pipeline
            alive.
        forecast: The very forecast ``decision`` was selected against, whose
            ``created_at`` stamps the context. Non-optional on purpose: a tick
            with no forecast never reaches this stage at all
            (:func:`_decide_and_approve` short-circuits), so there is no
            approval here to fail closed -- the fail-closed ``None`` lives one
            seam down, on :func:`build_evaluation_context`.

    Returns:
        The total quantity filled this tick, in contract-centis.
    """
    order_book = candidate.order_book
    try:
        market: NormalizedMarket | None = deps.exchange.get_market(candidate.ticker)
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
    now_epoch_s = deps.clock()
    context = build_evaluation_context(
        deps.config,
        now_epoch_s=now_epoch_s,
        verification=deps.kernel.latest_verification,
        instrument_whitelist=frozenset({candidate.ticker}),
        market=market,
        exchange_status=project_exchange_status(observed.status),
        exchange_status_epoch_s=status_epoch_s,
        pipeline_heartbeat_epoch_s=heartbeat_epoch_s,
        quote_snapshot_epoch_s=int(order_book.fetched_at.timestamp()),
        exchange_clock_epoch_s=read_exchange_clock_epoch_s(deps.exchange),
        forecast_epoch_s=int(forecast.created_at.timestamp()),
        open_position=read_open_position_centis(deps.exchange, ticker=candidate.ticker),
        equity_start_of_day=read_start_of_day_equity_micros(
            deps.store, now_epoch_s=now_epoch_s
        ),
        realized_loss_today=read_realized_loss_today_micros(
            deps.store, now_epoch_s=now_epoch_s
        ),
        equity_curve=read_equity_curve(deps.store),
        visible_depth=visible_depth_centis(order_book),
        exposure=exposure,
        notional_today=read_notional_today_micros(deps.store, now_epoch_s=now_epoch_s),
        orders_last_hour=read_orders_last_hour(deps.store, now_epoch_s=now_epoch_s),
    )
    filled = 0
    for intent in decision.intents:
        outcome = deps.approval.decide(intent, context)
        if outcome.token is not None:
            filled += _route_intent(deps, intent, outcome.token)
    return filled


def _position_rows(positions: tuple[Position, ...]) -> list[dict[str, object]]:
    """Project the connector's positions into the ledger's row shape.

    A pure rename of fields -- the numbers are the connector's own, untouched --
    so ``PositionsSnapshotRecorded`` and
    :meth:`~windbreak.connector.paper.PaperExchange.get_positions` can never
    disagree about what this account holds (issue #361). ``quantity_centis`` is
    therefore signed and in the YES frame: a long NO reports negative, the way
    :func:`windbreak.connector.paper._position_row` states it, rather than
    vanishing the way this module's deleted YES-only fill fold used to make it.

    Args:
        positions: The connector's positions, already in ticker order.

    Returns:
        One ``{ticker, quantity_centis, average_price_pips}`` row per holding,
        in the order given (empty when flat).
    """
    return [
        {
            "ticker": position.ticker,
            "quantity_centis": position.quantity.value,
            "average_price_pips": position.average_price.value,
        }
        for position in positions
    ]


def _position_value_micros(position: Position) -> int:
    """Return one holding's mark value, in micros, priced in its own side's frame.

    A pip is ``1e-4`` $ and a centi ``1e-2`` contracts, so a
    ``price_pips * quantity_centis`` product is exactly micros.

    A *positive* (long YES) row marks at ``quantity * average_price`` directly.
    A *negative* row is the YES-frame projection of a long NO, and its YES-frame
    product is negative -- which is not a mark at all. A long NO is economically
    a short YES plus a full ``$1`` of collateral per contract, so the two extra
    terms cancel to exactly what the holding cost: its size at its own NO price,
    the complement of the reported YES-frame price. That is also precisely the
    cash :meth:`~windbreak.connector.paper.PaperExchange.get_balances` debited
    for it, so the NO leg contributes the same figure it removed from cash and
    equity moves by the fee alone.

    The complement round-trips exactly: :func:`windbreak.connector.paper._position_row`
    reports ``COMPLEMENT_PIPS - floor(no_average)``, and subtracting that from
    ``COMPLEMENT_PIPS`` recovers the floored NO average -- still the
    understating direction, so this can only mark a holding at less than it
    cost, never more.

    Args:
        position: One connector-reported holding, in the YES frame.

    Returns:
        That holding's value, in micros (never negative).
    """
    quantity = position.quantity.value
    if quantity >= 0:
        return quantity * position.average_price.value
    return -quantity * (COMPLEMENT_PIPS - position.average_price.value)


def _positions_value_micros(positions: tuple[Position, ...]) -> int:
    """Return the mark value of open positions, in micros.

    Args:
        positions: The connector's reported holdings.

    Returns:
        The summed positions value, in micros.
    """
    return sum(_position_value_micros(position) for position in positions)


def _equity_and_positions_stage(deps: PaperTickDeps, now_epoch_s: int) -> int:
    """Sample equity and snapshot positions, ledgering both events.

    Positions come from :meth:`~windbreak.connector.paper.PaperExchange.get_positions`
    -- the one definition of "position" in the process (issue #361). The loop
    used to keep its own fold that summed YES-side fills only, so an account
    long NO snapshotted as *flat* and the exposure-bounding checks
    (``concentration_limits``, ``reduce_only_provable``) read a position of zero.

    When the venue refuses to describe itself, this stage fails closed without
    dying. ``get_positions`` raises
    :class:`~windbreak.connector.paper.TwoSidedPositionError` for a ticker
    filled on both sides, because the only single-row answer is a
    netted one and a netted YES-plus-NO reports flat while both legs and their
    collateral are live. Letting that escape would abort the tick -- and a
    stage that kills the loop is not failing closed, it is failing silent (the
    same reasoning :func:`_forecast_stage` applies to a budget breach and
    :meth:`~windbreak.riskkernel.verification.ReadOnlyVerifier._unobservable_venue_breach`
    applies to an unobservable venue). So instead:

    * **No** ``PositionsSnapshotRecorded`` is appended. An empty row list would
      be the fabricated healthy zero the connector just refused to invent, and
      writing the YES leg alone would be the understated holding this issue
      exists to remove. An honest gap is the only truthful option.
    * The equity sample still lands, valuing the unpriceable holding at zero.
      That can only *understate* equity, which only lowers the
      ``daily_loss_limit`` baseline :func:`start_of_day_equity_micros` reads
      back -- tightening the check, never loosening it.

    The loud half of the response is the verification cycle's, not this stage's:
    :func:`_verification_stage` observes the same connector at the *top* of every
    tick, before anything can route, and grades a refusing venue a forced
    ``BREACH`` that HALTs the kernel and records ``VerificationMismatchHalt``.
    So a two-sided holding stops trading on the very next tick through the
    audited path, and this stage's duty is only to neither lie nor die.

    Args:
        deps: The tick's dependency bundle.
        now_epoch_s: The tick's epoch-second clock reading.

    Returns:
        The sampled equity, in micros.
    """
    try:
        holdings: tuple[Position, ...] | None = deps.exchange.get_positions()
    except TwoSidedPositionError:
        holdings = None
    positions_value = 0 if holdings is None else _positions_value_micros(holdings)
    equity = compute_equity_micros(
        available_cash=deps.exchange.get_balances().available,
        positions_value=MoneyMicros(positions_value),
    )
    deps.store.append(
        EquitySampled(
            component=_COMPONENT,
            equity_micros=equity.value,
            floor_micros=deps.config.capital.floor_micros,
            epoch_s=now_epoch_s,
        )
    )
    if holdings is not None:
        deps.store.append(
            PositionsSnapshotRecorded(
                component=_COMPONENT, positions=_position_rows(holdings)
            )
        )
    return equity.value


def _heartbeat_stage(deps: PaperTickDeps, now_epoch_s: int) -> int:
    """Ledger a pipeline heartbeat and return the instant it attests to.

    Called after the screen stage -- which reads the venue's markets and their
    books -- so the heartbeat is stamped only once the tick has proven the
    pipeline genuinely running, an attestation rather than a constant. Before
    issue #345 the same duty was served by the snapshot stage, which used to run
    first; the screen now does, and it exercises the same connector reads.

    Stamped once per tick rather than once per candidate: it attests that the
    *pipeline* is alive, which is not a per-market fact, and one heartbeat per
    market would let a slow universe walk keep refreshing its own freshness.
    Every candidate's approval is therefore judged against the one instant the
    tick actually observed. It is deliberately NOT stamped inside the approval
    context either: a
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


def _kill_stage(deps: PaperTickDeps) -> None:
    """Poll the operator's kill/re-arm files once, before the tick does anything.

    The one call that makes issue #441's wiring real rather than merely present:
    a :class:`~windbreak.riskkernel.kill.KillIntegration` that is composed but
    never polled is indistinguishable at runtime from the ``None`` it replaced.

    Runs *first*, ahead of the screen, the heartbeat, and the verification
    cycle, because that is the fail-safe direction: a ``KILL`` file already on
    disk when the beat starts must stop *this* tick rather than the next one.
    The poll is bounded -- one directory probe, never a wait -- so it cannot
    stall the beat, and it is also how a re-arm is consumed, since
    :meth:`~windbreak.riskkernel.kill.KillFileWatcher.poll_once` reads the
    ``REARM`` file only while the switch is ``KILLED``.

    It deliberately does not decide anything about the tick. Whether the
    universe is walked is read from the kernel's *mode* after the verification
    cycle (:func:`run_single_tick`), not from a flag returned here, so the
    file-driven kill and the reconciliation auto-kill -- which fires mid-cycle,
    after this stage has run -- stop the tick by exactly the same mechanism.

    Args:
        deps: The tick's dependency bundle.
    """
    deps.kernel.poll_kill_triggers()


def _verification_stage(deps: PaperTickDeps) -> None:
    """Run one read-only verification cycle, HALTing the kernel on a breach.

    Runs early in the tick -- before any order can route -- so an account that
    has already drifted away from the reconciled baseline is caught *before*
    this tick adds to the drift, not after. The cycle observes the exchange
    through ``deps.verification_view``, which carries no order-placing method,
    and it runs on every tick including one whose research halted: reconciling
    the venue is a liveness duty, not a consequence of trading.

    Everything the cycle can do is already the kernel's contract
    (:meth:`~windbreak.riskkernel.process.RiskKernel.run_verification_cycle`):
    it records exactly one ``VerificationPassed`` / ``VerificationDrift`` /
    ``VerificationMismatch`` event, retains the snapshot for this tick's
    approvals, and on a ``BREACH`` transitions the kernel to ``HALT`` and
    records a ``VerificationMismatchHalt`` (issue #32). A venue the view cannot
    even describe -- ``PaperExchange.get_positions`` refuses to net a two-sided
    holding -- is graded a forced breach there rather than escaping as an
    exception, so an unobservable venue halts instead of killing the tick.

    Every execution the venue has reported, and every order it has come to rest,
    is booked into the ledger first (issues #365 and #390), so the expectation
    the cycle diffs against has already absorbed what the ledger can explain.
    Booking here rather than beside the routing call catches activity from
    *every* source -- a taker walk on a placed order and a remainder resting
    behind it alike -- and does it at the one moment the answer is needed. A
    recorded trade-through fill is a third source in principle, but only for a
    harness that steps the replay itself: the loop's cursor is stationary (SPEC
    S7.5.1, issue #387), so no ``PaperExchange.advance`` fill arises in a run.
    Booking is idempotent on the venue's fill id and order id, so re-entering
    this stage never advances the expectation past cash the venue moved once or
    an order it rested once.

    The booking reads the venue's *discrete reports* -- executions, and orders
    arriving on the resting book; the cycle reads the venue's *aggregate*
    balances, positions, and live resting book. Those are different questions,
    which is why the comparison can still fail -- see
    :class:`~windbreak.riskkernel.verification.LedgerExpectationSource`.

    Args:
        deps: The tick's dependency bundle.
    """
    deps.fill_bookkeeper.book_new()
    deps.kernel.run_verification_cycle()


@dataclass(frozen=True, slots=True)
class _UniverseOutcome:
    """What one tick's walk over its screened candidates produced.

    Attributes:
        forecast_ids: One id per market actually forecast, in processing order.
        intent_count: Intents emitted, summed over every market processed.
        filled_centis: Quantity filled, in contract-centis, summed likewise.
        research_halted: Whether a budget ceiling stopped the walk early.
    """

    forecast_ids: tuple[str, ...]
    intent_count: int
    filled_centis: int
    research_halted: bool


def _run_candidate(
    deps: PaperTickDeps,
    candidate: MarketCandidate,
    created_at: datetime,
    heartbeat_epoch_s: int,
) -> tuple[str, int, int] | None:
    """Run one screened market through snapshot, forecast, select, and approve.

    Narrows the optional forecast in one place so the select and approve stages
    keep their non-optional contracts. That narrowing is why a budget-halted
    market needs no fail-closed forecast stamp of its own (issue #380): it never
    reaches an approval at all, which is strictly stronger than vetoing one.

    Args:
        deps: The tick's dependency bundle.
        candidate: The screened market to run.
        created_at: The injected creation instant, for determinism.
        heartbeat_epoch_s: The instant an earlier stage observed the pipeline
            alive, threaded through to the approval context.

    Returns:
        A ``(forecast_id, intent_count, filled_centis)`` triple, or ``None``
        when research halted fail-closed on a budget ceiling.
    """
    _snapshot_stage(deps, candidate)
    forecast = _forecast_stage(deps, candidate, created_at)
    if forecast is None:
        return None
    exposure = read_candidate_exposure(deps, candidate)
    decision = _select_stage(deps, candidate, forecast, created_at, exposure)
    filled = _approve_stage(
        deps, candidate, decision, heartbeat_epoch_s, forecast, exposure
    )
    return forecast.forecast_id, len(decision.intents), filled


def _run_universe(
    deps: PaperTickDeps,
    candidates: tuple[MarketCandidate, ...],
    created_at: datetime,
    heartbeat_epoch_s: int,
) -> _UniverseOutcome:
    """Walk the tick's screened candidates, stopping on a budget halt.

    The walk stops at the first market whose research halts on a ceiling rather
    than trying the rest (issue #345). The per-UTC-day ceiling is checked before
    any tool or transport is touched, so every remaining candidate would halt on
    the same exhausted bucket and append a ``ResearchBudgetHalted`` row saying
    nothing the first one did not: an exhausted day is a property of the day,
    not of the market that happened to discover it.

    A market **refused** on its own hostile free-text metadata is the opposite
    case and gets the opposite answer: the walk *continues* (issue #525). Being
    unforecastable is a property of that one market, so stopping the walk would
    let a single forged ticker in the venue's universe deny service to every
    market behind it. ``run_pipeline`` raises
    :class:`~windbreak.forecast.providers.base.ProviderMarketMetadataRejectedError`
    at entry for such a market, which is the only way that error can reach here:
    the vote-collection loop discards its own per-vote refusals internally. The
    market contributes no ``ForecastCreated``, no ``SelectorDecisionRecorded``
    and no approval row -- which is exactly the point, since its ticker would
    otherwise be written verbatim into an append-only chain.

    Args:
        deps: The tick's dependency bundle.
        candidates: The screened markets, in processing order.
        created_at: The injected creation instant, for determinism.
        heartbeat_epoch_s: The instant an earlier stage observed the pipeline
            alive, threaded through to each approval context.

    Returns:
        The summed :class:`_UniverseOutcome` for the markets actually reached.
    """
    forecast_ids: list[str] = []
    intent_count = 0
    filled_centis = 0
    halted = False
    for candidate in candidates:
        try:
            result = _run_candidate(deps, candidate, created_at, heartbeat_epoch_s)
        except ProviderMarketMetadataRejectedError:
            continue
        if result is None:
            halted = True
            break
        forecast_id, intents, filled = result
        forecast_ids.append(forecast_id)
        intent_count += intents
        filled_centis += filled
    return _UniverseOutcome(
        forecast_ids=tuple(forecast_ids),
        intent_count=intent_count,
        filled_centis=filled_centis,
        research_halted=halted,
    )


def _refreshed_budget(deps: PaperTickDeps) -> PaperTickDeps:
    """Re-read the research budget off the ledger before the tick spends (#442).

    The budget is rebuilt rather than mutated, from the same
    :func:`_build_research_budget` the composition root uses, so exactly one
    function decides what a budget opens with and the startup path and the
    per-tick path cannot drift apart.

    Rebuilding here -- rather than only at process start -- is what makes the
    ceiling both **durable** and **live**:

    * A sibling process, or this process before its last crash, has its spend
      folded back in, so the day's ceiling holds across the unbounded restart
      count ``restart: on-failure`` permits.
    * An operator's ``windbreak set-research-budget`` row is picked up on the
      *next tick*, with no restart, which is what "adjustable on the fly"
      requires.

    The ledger read is one full scan per tick, the same cost
    :func:`windbreak.scheduler.weekly_data.weekly_report_body` already pays on
    every tick, so it adds a constant factor rather than a new order of work.

    Args:
        deps: The wired dependency bundle whose budget is being refreshed.

    Returns:
        A bundle identical to ``deps`` but for a budget opened on the ledger's
        current day counter and ceiling.
    """
    return dataclasses.replace(
        deps, budget=_build_research_budget(deps.store, deps.config)
    )


def run_single_tick(deps: PaperTickDeps, *, beat: int) -> TickOutcome:
    """Drive one PAPER tick end to end, ledgering every stage (SPEC S5.3).

    The tick opens by screening the venue's market universe
    (:func:`_screen_stage`) into a bounded candidate set, then runs the SINGLE
    order path -- snapshot -> forecast -> select -> approve(seam) -> (only if a
    token minted) route -> fill -> reconcile -- once **per screened candidate**,
    then emits the per-tick heartbeat, equity sample, and positions snapshot, and
    writes this ISO-week's report -- folding the real ledger through
    :func:`windbreak.scheduler.weekly_data.weekly_report_body` so the report
    carries genuine evaluation and cost-meter data (issue #188), built lazily so
    the fold is paid for only on the genuine per-week write. Every stage appends
    an audit event to the shared hash-chained ledger.

    Issue #345 is what made that per-candidate. The loop used to forecast
    ``next(iter(exchange.markets))`` -- one arbitrary market, fixed for the life
    of the process and never screened at all. The single-market path was not
    merely narrow: nothing checked that the market it traded was tradeable.

    Iterating a universe multiplies research spend by the markets screened, so
    two things bound it and both are load-bearing. Screening is *free* -- the
    four §16 filters are pure integer comparisons over metadata and a book, no
    model calls -- so the loop never spends money deciding what to spend money
    on. And ``config.screener.max_candidates_per_tick`` caps the forecasts one
    tick may run, enforced on the universe walk itself, which caps the tick's
    worst-case bill at that many per-forecast ceilings. The per-UTC-day ceiling
    then bounds the *day* independently, and a tick that trips it stops walking
    (see :func:`_run_universe`).

    The tick-level stages -- heartbeat, verification cycle, mode heartbeat,
    equity sample, positions snapshot, weekly report -- still run exactly once
    per tick, not once per market. They describe the *loop*, not a market, and
    duplicating them per candidate would inflate the audit trail with rows that
    say the same thing N times. The verification cycle in particular runs before
    any candidate is touched, so a breach halts the kernel ahead of the whole
    universe rather than part-way through it.

    The tick does **not** step the replay cursor, and that is a decision rather
    than an omission (issue #387, SPEC S7.5.1). A fixture run therefore prices,
    fills, and allocates against the one recorded step it opened on for the life
    of the process; ``PaperExchange.advance`` is harness API with no caller
    here, and the always-on path whose market data genuinely advances is
    ``LiveBookPaperExchange``, which reads the venue's real books. Stepping the
    cursor per tick was considered and rejected -- S7.5.1 is the canonical
    account of why, and of the consequences the stationary reading accepts, so
    the argument is not restated here. Instead,
    ``tests/integration/test_paper_replay_cursor.py`` fails if the answer
    changes here.

    Every mention of "positions snapshot" below carries one standing exception,
    stated once here rather than repeated: a connector that refuses to describe
    the account (a ticker filled on both sides) yields no
    ``PositionsSnapshotRecorded`` row at all, because the only rows available to
    write would be fabricated or understated. See
    :func:`_equity_and_positions_stage`.

    Since issue #353 the tick also runs one read-only verification cycle
    (:func:`_verification_stage`) before deciding anything, and threads its
    snapshot into the approval context, so the three reconciliation checks now
    evaluate real evidence rather than failing closed on ``None``. Issue #364
    did the same for the two exposure feeds (see the module docstring), so no
    SPEC S10.3 check now vetoes for want of a datum the loop holds -- with one
    honest exception: the day's *first* tick approves before it has sampled the
    day's equity, so ``daily_loss_limit`` still fails closed on that one tick.

    Stage order is load-bearing for that reason. The equity sample is taken
    after the approval stage on purpose -- it reflects the tick's own fills --
    so moving it earlier to hand the day's first approval a baseline would make
    "start of day" mean "a moment ago", which is not the figure
    ``daily_loss_limit`` is defined against.

    A cycle that grades a ``BREACH`` halts the kernel instead (issue #32); the
    tick still completes and still ledgers its heartbeat, equity sample, and
    positions snapshot, but every later approval vetoes on the halted mode, and
    :attr:`TickOutcome.kernel_halted` says so.

    Since issue #441 the tick *opens* by polling the operator's kill/re-arm
    files (:func:`_kill_stage`) and walks **no** candidates at all while the
    kernel is ``KILLED``. Two triggers reach that state and both stop the same
    tick: the ``KILL`` file ``windbreak kill`` writes, read before anything else
    happens, and the ``AUTO_RECONCILIATION`` auto-kill that fires inside the
    verification cycle once ``risk.kill_after_consecutive_mismatches``
    consecutive breaches have accumulated. The mode is therefore read *after*
    that cycle, so neither trigger needs its own path.

    A killed tick is not a skipped tick. It still screens (the screen is free
    and its rows are the honest record of what was examined), still stamps its
    pipeline heartbeat, still reconciles the venue, still samples equity and
    positions, and still writes the week's report -- an always-on loop must stay
    observably alive and flat while dead, not fall silent. What it does not do
    is research or route: the walk is where money is spent, and a kernel that
    can approve nothing must not pay for forecasts it cannot act on.
    ``ModeHeartbeat`` carries ``KILLED``, so the ledger and the heartbeat line
    both say so. ``HALT`` deliberately does *not* skip the walk -- a halted
    kernel is expected to recover and its approvals veto individually -- so this
    is the kill switch's dead hand, not a general mode gate.

    A market whose per-forecast or per-UTC-day research budget is exhausted
    halts fail-closed (issue #339): it ledgers one ``ResearchBudgetHalted`` row,
    skips that market's forecast, select, and approve stages, and stops the
    universe walk there. The tick still emits its heartbeat, equity sample,
    positions snapshot, and weekly report -- so the loop stays observably alive
    and flat rather than dying on an uncaught budget error.

    The halting market and the markets behind it leave *different* ledger
    shapes, and the difference is worth stating precisely, because someone
    auditing a halt reads this paragraph to know which rows to expect:

    * **The halting market** keeps its ``MarketSnapshotRecorded``.
      :func:`_run_candidate` ledgers the snapshot before :func:`_forecast_stage`
      can return ``None``, so the book it was about to research is on record.
      What it loses is ``ForecastCreated``, ``SelectorDecisionRecorded``, and
      the ``ExchangeStatusObserved`` the approval stage would have appended.
    * **Every candidate after it** is never run at all -- :func:`_run_universe`
      breaks before reaching them -- so they lose their
      ``MarketSnapshotRecorded`` too, not merely the forecast and selector rows.
      What they keep is their ``ScreenDecisionRecorded``, because the screen ran
      over the whole candidate set before the walk began. The ledger therefore
      says such a market was examined and found eligible, and says nothing
      further about it: the honest record of a market the tick screened in and
      then ran out of money before reaching.

    Both shapes are pinned as exact golden row sequences in
    ``tests/integration/test_paper_universe.py``, so this description is
    checkable rather than prose that can drift away from the code.

    A market **refused** on hostile free-text metadata (issue #525) leaves a
    third shape, and the walk does not stop for it (see :func:`_run_universe`).
    It keeps its ``ScreenDecisionRecorded`` and its ``MarketSnapshotRecorded``,
    both appended before the forecast stage is reached, and loses everything
    from ``ForecastCreated`` onward -- no research is paid for, so it leaves no
    ``ResearchSpendRecorded`` either. Every candidate behind it still runs.
    Both surviving rows carry the market's ticker as a stable
    ``<rejected-ticker:sha256:...>`` digest rather than its bytes (issue #530),
    so the two rows still correlate with each other and with nothing attacker-
    chosen reaching the chain. A market the §16 screen turns *away* leaves only
    the first of those two rows, digest included, and never reaches the forecast
    stage at all -- a strictly wider population, since it need not screen in.
    ``tests/integration/test_paper_hostile_ticker.py`` pins every one of these
    shapes by reading the chain back from disk.

    Args:
        deps: The fully wired dependency bundle.
        beat: The 1-based tick sequence number, stamped on the heartbeat.

    Returns:
        A :class:`TickOutcome` summarizing the tick.
    """
    deps = _refreshed_budget(deps)
    now_epoch_s = deps.clock()
    created_at = datetime.fromtimestamp(now_epoch_s, UTC)
    _kill_stage(deps)
    candidates = _screen_stage(deps)
    heartbeat_epoch_s = _heartbeat_stage(deps, now_epoch_s)
    _verification_stage(deps)
    # Read *after* the verification cycle, so the reconciliation auto-kill that
    # fires inside it stops this tick by the same door the operator's KILL file
    # does. A killed kernel walks no candidates at all: its `evaluate_intent`
    # would hard-veto every intent anyway, but the walk is where the loop spends
    # research money, and a dead kernel paying for forecasts it can never act on
    # is the one failure a kill switch exists to prevent.
    tradeable = () if deps.kernel.mode is Mode.KILLED else candidates
    universe = _run_universe(deps, tradeable, created_at, heartbeat_epoch_s)
    # The kernel's *real* mode, never a hardcoded PAPER: a verification breach
    # drives it to HALT mid-tick, and a heartbeat still claiming PAPER would be
    # the loop's own audit trail lying about whether it is trading.
    mode = deps.kernel.mode
    deps.store.append(ModeHeartbeat(component=_COMPONENT, mode=mode.name, beat=beat))
    equity = _equity_and_positions_stage(deps, now_epoch_s)
    report_date = created_at.date()
    maybe_write_weekly(
        deps.report_dir,
        today=report_date,
        body=lambda: weekly_report_body(deps.store.read_all(), today=report_date),
    )
    return TickOutcome(
        beat=beat,
        candidate_tickers=tuple(candidate.ticker for candidate in candidates),
        forecast_ids=universe.forecast_ids,
        intent_count=universe.intent_count,
        filled_centis=universe.filled_centis,
        equity_micros=equity,
        research_halted=universe.research_halted,
        kernel_halted=mode is Mode.HALT,
    )
