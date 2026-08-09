"""The single always-on PAPER-mode tick composition (issue #48, SPEC S5.3).

This module is the PAPER loop's one composition root. :func:`build_paper_deps`
wires the real, unmodified Market Connector (a `PaperExchange`), Forecast Engine,
Trade Selector, Risk Kernel, Order Gateway, and Reconciler over a single
hash-chained :class:`~windbreak.ledger.store.SqliteLedgerStore`, and
:func:`run_single_tick` drives one SPEC S5.3 SINGLE order-path tick through them:

    snapshot -> forecast -> select -> approve(seam) -> (only if a token minted)
    route -> PaperExchange fill -> reconcile

appending one audit event to the ledger at every stage, plus a per-tick
``ModeHeartbeat``, an ``EquitySampled``, and -- whenever the connector can
describe the account at all -- a ``PositionsSnapshotRecorded``
(:func:`_equity_and_positions_stage` explains the one case that omits it, and
why an omitted row is safer there than a written one).

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

import json
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
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
from windbreak.forecast.budget import (
    BUDGET_DAY_EXHAUSTED_EVENT,
    BUDGET_FORECAST_EXCEEDED_EVENT,
    DailyBudgetExhaustedError,
    PerForecastBudgetExceededError,
    ResearchBudget,
)
from windbreak.forecast.cassettes import ReplayCassette
from windbreak.forecast.pipeline import (
    PROVIDER_GATE_HELD_EVENT,
    PROVIDER_VOTE_COSTED_EVENT,
    InMemoryForecastLedger,
    run_pipeline,
)
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
from windbreak.scheduler.fill_accounting import (
    LedgerFillAccountingFeed,
    LedgerFillBookkeeper,
)
from windbreak.scheduler.provider_wiring import (
    ProviderFactory,
    build_live_llm_transport,
    build_live_research_tools,
    build_provider_factory,
    is_live_mode,
    offline_research_tools,
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
    from collections.abc import Callable, Iterable
    from datetime import date
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig
    from windbreak.connector.live import MarketDataSource
    from windbreak.connector.models import (
        NormalizedMarket,
        OrderBookSnapshot,
        Position,
    )
    from windbreak.forecast.budget import BudgetEvent
    from windbreak.forecast.cassettes import LlmTransport
    from windbreak.forecast.records import ForecastRecord
    from windbreak.forecast.sandbox import ResearchTools
    from windbreak.ledger.events import Event
    from windbreak.ledger.store import LedgerRecord
    from windbreak.riskkernel.checks import Decision, OrderIntent
    from windbreak.riskkernel.verification import VerificationSnapshot
    from windbreak.scheduler.provider_wiring import LiveProviderHttp
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

#: The M6 per-provider track-record artifact the loop's live-eligibility gate
#: reads, resolved inside the same ``report_dir`` every other evaluation
#: artifact is written to (issue #305). Public because it is an operator- and
#: test-facing convention: the file an evaluation pass must write for a provider
#: to become live-eligible. Absent, the loop bootstraps fail-closed (every
#: provider unproven); malformed, it refuses to start.
PROVIDER_TRACK_RECORD_FILENAME: Final = "provider-track-records.json"


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


def _utc_day(epoch_s: int) -> date:
    """Return the UTC calendar day an epoch second falls on.

    Args:
        epoch_s: The instant to bucket, in whole epoch seconds.

    Returns:
        The UTC date containing ``epoch_s``.
    """
    return datetime.fromtimestamp(epoch_s, UTC).date()


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

    That makes this fold O(ledger), which is why the tick no longer reaches it
    directly: :func:`read_start_of_day_equity_micros` is the entry point, and it
    prefers a bounded reverse walk on any store offering one (issue #370). This
    fold remains the fallback for stores without that capability -- and the
    reference definition of the answer both paths must produce.

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
    today = _utc_day(now_epoch_s)
    earliest: tuple[int, int] | None = None
    for record in records:
        if record.event_type != _EQUITY_SAMPLED_EVENT_TYPE:
            continue
        epoch_s, equity_micros = _equity_sample(record)
        if _utc_day(epoch_s) != today:
            continue
        if earliest is None or epoch_s < earliest[0]:
            earliest = (epoch_s, equity_micros)
    return MoneyMicros(earliest[1]) if earliest is not None else None


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
    :func:`_bounded_start_of_day_equity_micros` -- and any other store,
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
    if isinstance(store, ReverseTypeScan):
        return _bounded_start_of_day_equity_micros(store, now_epoch_s=now_epoch_s)
    return start_of_day_equity_micros(store.read_all(), now_epoch_s=now_epoch_s)


def _bounded_start_of_day_equity_micros(
    store: ReverseTypeScan, *, now_epoch_s: int
) -> MoneyMicros | None:
    """Derive the baseline from a newest-first walk that stops at the day boundary.

    Walks ``EquitySampled`` rows alone, newest first, and stops at the first one
    stamped on an *earlier* UTC day: everything older is older still, so nothing
    beyond it can be today's earliest. The cost is therefore O(samples taken
    today) -- bounded by ticks-per-day -- rather than O(ledger), with no new
    index and no change to what the answer means.

    Two subtleties keep it faithful to the full fold it replaces:

    * The earliest of today's samples is chosen by comparing stamped
      ``epoch_s`` values, not by taking the first row the walk happens to
      surface, so a clock that steps backwards mid-day cannot promote a later
      sample to the baseline (the walk arrives newest-first, so seizing its
      first hit would be exactly the latest-of-day reading this must not do).
    * A sample stamped on a *later* day -- a forward clock blip -- is skipped
      rather than treated as the boundary. Only predating today ends the walk;
      stopping on a future stamp would discard today's genuine baseline for as
      long as the blip sat at the head of the ledger.

    The one case where this can diverge from the full fold is a clock that
    rewinds *across* midnight and then jumps forward again, burying a still
    earlier same-day sample beneath a previous-day one. That ordering makes the
    samples mutually contradictory anyway, and the walk keeps the invariant that
    matters here: it never reports a baseline that no sample carried.

    Args:
        store: The ledger, declaring the reverse-walk capability.
        now_epoch_s: The instant whose UTC day is the "current" one.

    Returns:
        The day's earliest sampled equity, in micros, or ``None`` when the walk
        reaches the day boundary (or the end of the ledger) without a sample.

    Raises:
        KeyError: If an ``EquitySampled`` payload is missing either field this
            reads -- a loud shape drift, never a silently zeroed baseline.
    """
    today = _utc_day(now_epoch_s)
    earliest: tuple[int, int] | None = None
    for record in store.iter_records_of_type_reversed(_EQUITY_SAMPLED_EVENT_TYPE):
        epoch_s, equity_micros = _equity_sample(record)
        sample_day = _utc_day(epoch_s)
        if sample_day < today:
            break
        if sample_day > today:
            continue
        if earliest is None or epoch_s < earliest[0]:
            earliest = (epoch_s, equity_micros)
    return MoneyMicros(earliest[1]) if earliest is not None else None


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


def _account_from_verification(
    verification: VerificationSnapshot | None,
    equity_start_of_day: MoneyMicros | None,
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

    Every other term stays zero: they are the ones the ledger cannot yet
    justify (high-water mark, exposures, velocity), and a fabricated figure
    there would loosen a limit rather than tighten it. With
    ``verification=None`` and no baseline the whole account is zero, exactly as
    before, so the fail-closed path is unchanged.

    Args:
        verification: The tick's verification snapshot, or ``None`` when no
            cycle has produced one (the fail-closed reading).
        equity_start_of_day: The current UTC day's first ledgered equity
            sample, or ``None`` when the day has none yet (also fail-closed).

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
    baseline = equity_start_of_day if equity_start_of_day is not None else zero
    return AccountState(
        exchange_verified_available_cash=verified_cash,
        guaranteed_terminal_value_of_positions=zero,
        pending_kernel_reservations=zero,
        unresolved_fee_upper_bounds=zero,
        reconciliation_uncertainty_buffer=drift,
        equity_start_of_day=baseline,
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
    quote_snapshot_epoch_s: int | None,
    exchange_clock_epoch_s: int | None,
    forecast_epoch_s: int | None,
    open_position: ContractCentis | None,
    equity_start_of_day: MoneyMicros | None,
    visible_depth: ContractCentis | None,
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
        visible_depth: The visible book depth, in contract-centis, or ``None``
            when no book could be read -- which fails closed (issue #364). A
            genuinely empty book is ``0``, not ``None``: an observed absence of
            liquidity is evidence, and it admits no order at all.

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
        limits=_build_limits(config, instrument_whitelist),
        account=_account_from_verification(verification, equity_start_of_day),
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
    ticker: str
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


def _resolve_forecast_transport(
    config: WindbreakConfig,
    cassette_path: Path,
    provider_http: LiveProviderHttp | None,
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
    if provider_http is None:
        return ReplayCassette.from_path(cassette_path), False
    return build_live_llm_transport(config, provider_http), True


def _resolve_research_tools(
    research_tools: ResearchTools | None,
    ledger_path: Path,
    config: WindbreakConfig,
    provider_http: LiveProviderHttp | None,
) -> ResearchTools:
    """Return the supplied research tools, or the mode's own default.

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

    Returns:
        A sandboxed :class:`~windbreak.forecast.sandbox.ResearchTools`.
    """
    cache_dir = ledger_path.parent.joinpath("research-cache")
    if research_tools is not None:
        return research_tools
    if provider_http is not None:
        return build_live_research_tools(config, provider_http, cache_dir)
    return offline_research_tools(cache_dir)


def _build_verifier(
    store: SqliteLedgerStore,
    config: WindbreakConfig,
    view: ReadOnlyVenueView,
    writer: _SqliteKernelLedgerWriter,
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
    each execution once, durably, into this same hash chain, and the feed hands
    those entries to the expectation. Two composition-time decisions live here,
    both deliberate:

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
    match). The dispatcher fans mismatch and unknown-jurisdiction alerts out
    through the log-only fallback, matching every other no-sink composition
    root in ``windbreak.main``.

    Args:
        store: The hash-chained ledger whose replayed history seeds the
            baseline.
        config: The configuration supplying the two drift tolerances.
        view: The narrow read-only venue view the cycle observes through -- it
            exposes no ``place_order``/``cancel_order`` (SPEC S1.1 invariant 3).
        writer: The kernel ledger writer each cycle's event is recorded through.

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
        dispatcher=AlertDispatcher(sinks=[], ledger_writer=LoggingLedgerWriter()),
        ledger_writer=writer,
    )


def _build_approval(
    store: SqliteLedgerStore,
    config: WindbreakConfig,
    key: bytes,
    view: ReadOnlyVenueView,
    clock: Callable[[], int],
) -> tuple[KernelApproval, RiskKernel]:
    """Wire the real kernel + approval pipeline into a `KernelApproval` seam.

    The kernel tracks PAPER mode (so its ledgered evaluation stamps PAPER) with
    ``kill_integration=None`` -- kill wiring is out of scope -- and shares the one
    ephemeral signing key with the gateway. The same hash-chained ``store`` is
    wired as the kernel's ``gate_plan_store`` (issue #185), so a PAPER ->
    LIVE_MICRO promotion reads its three thresholds from the pre-registered gate
    plan on the ledger, failing closed when none is registered.

    Issue #353 additionally wires a real read-only verifier and the tick's own
    injected ``clock``, so every cycle the kernel runs is stamped at the same
    instant the rest of the tick reads -- a snapshot aged against an unrelated
    wall clock could go stale against ``now_epoch_s`` for no real reason.

    Args:
        store: The ledger both the kernel and the pipeline record through.
        config: The configuration whose hash is stamped into minted tokens.
        key: The ephemeral 32-byte signing key.
        view: The read-only venue view the verification cycle observes through.
        clock: The injected epoch-second clock the verification cycle stamps
            its snapshots at.

    Returns:
        The composed :class:`KernelApproval` seam and the kernel inside it, so
        the tick can drive that kernel's verification cycle and read its mode.
    """
    writer = _SqliteKernelLedgerWriter(store)
    mode_machine = ModeStateMachine(
        mode_ceiling=Mode.from_config(config.mode_ceiling), mode=Mode.PAPER
    )
    kernel = RiskKernel(
        writer,
        mode_machine=mode_machine,
        verifier=_build_verifier(store, config, view, writer),
        clock=clock,
        gate_plan_store=store,
        kill_integration=None,
    )
    ledger = ReservationLedger(writer)
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
        live_ticker: The single market a live session trades, or ``None`` for
            fixtures. The market universe is a separate concern (issue #345).

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

    Returns:
        A fully wired :class:`PaperTickDeps`.

    Raises:
        ValueError: If exactly one of ``market_data``/``live_ticker`` is given,
            if the configured provider-transport mode is unknown or disagrees
            with whether ``provider_http`` was supplied, if a configured ceiling
            or list price is not positive, or if the track-record artifact
            exists but cannot be read as a strict integer document -- each way
            refusing to start rather than running unguarded.
    """
    resolved_clock = clock if clock is not None else _default_clock
    # Selected first, before the ledger database or any exchange session
    # exists, so a misconfigured transport aborts startup without leaving
    # half-built durable state behind.
    transport, live = _resolve_forecast_transport(config, cassette_path, provider_http)
    # The exchange must observe on the same clock the tick reads, or its status
    # attestation drifts against `now_epoch_s` and `exchange_status_ok` judges
    # freshness against two unrelated timelines (issue #342).
    exchange = _build_paper_exchange(
        books_dir, resolved_clock, market_data=market_data, live_ticker=live_ticker
    )
    # In live mode `markets` is the single bound ticker, so this one line
    # answers both modes: the fixture directory's first market, or the market
    # the operator named.
    ticker = next(iter(exchange.markets))
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
        store, config, key, verification_view, resolved_clock
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
        ticker=ticker,
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
        research_tools=_resolve_research_tools(
            research_tools, ledger_path, config, provider_http
        ),
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

    Attributes:
        beat: The 1-based tick sequence number.
        forecast_id: The forecast this tick produced.
        intent_count: How many normalized intents the selector emitted.
        filled_centis: The quantity filled through the gateway this tick, in
            contract-centis (``0`` whenever the kernel vetoed every intent).
        equity_micros: The sampled account equity this tick, in micros.
        research_halted: Whether this tick's research was halted fail-closed on
            a budget ceiling. When ``True`` no forecast exists, so
            ``forecast_id`` is ``""`` and no selector decision was made.
        kernel_halted: Whether the Risk Kernel is in ``HALT`` at the end of this
            tick -- today only a verification ``BREACH`` puts it there (issue
            #32). A halted kernel vetoes every later intent, so an always-on
            driver must treat this as "stop and get a human", not as a
            transient. Reported per tick rather than raised, because the tick
            must still finish ledgering its heartbeat, equity, and positions:
            the halt is exactly when that audit trail matters most.
    """

    beat: int
    forecast_id: str
    intent_count: int
    filled_centis: int
    equity_micros: int
    research_halted: bool = False
    kernel_halted: bool = False


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
    deps: PaperTickDeps,
    decision: SelectorDecision,
    heartbeat_epoch_s: int,
    order_book: OrderBookSnapshot,
    forecast: ForecastRecord,
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

    A market the exchange cannot resolve becomes ``None`` rather than an
    exception, so an unknown ticker vetoes the tick instead of aborting it.

    Args:
        deps: The tick's dependency bundle.
        decision: The selector's decision carrying any emitted intents.
        heartbeat_epoch_s: The instant an earlier stage observed the pipeline
            alive.
        order_book: The book snapshot this tick took, whose shallower visible
            side bounds the participation cap.
        forecast: The very forecast ``decision`` was selected against, whose
            ``created_at`` stamps the context. Non-optional on purpose: a tick
            with no forecast never reaches this stage at all
            (:func:`_decide_and_approve` short-circuits), so there is no
            approval here to fail closed -- the fail-closed ``None`` lives one
            seam down, on :func:`build_evaluation_context`.

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
    now_epoch_s = deps.clock()
    context = build_evaluation_context(
        deps.config,
        now_epoch_s=now_epoch_s,
        verification=deps.kernel.latest_verification,
        instrument_whitelist=frozenset({deps.ticker}),
        market=market,
        exchange_status=project_exchange_status(observed.status),
        exchange_status_epoch_s=status_epoch_s,
        pipeline_heartbeat_epoch_s=heartbeat_epoch_s,
        quote_snapshot_epoch_s=int(order_book.fetched_at.timestamp()),
        exchange_clock_epoch_s=read_exchange_clock_epoch_s(deps.exchange),
        forecast_epoch_s=int(forecast.created_at.timestamp()),
        open_position=read_open_position_centis(deps.exchange, ticker=deps.ticker),
        equity_start_of_day=read_start_of_day_equity_micros(
            deps.store, now_epoch_s=now_epoch_s
        ),
        visible_depth=visible_depth_centis(order_book),
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

    Every execution the venue has reported is booked into the ledger first
    (issue #365), so the expectation the cycle diffs against has already
    absorbed the fills the ledger can explain. Booking here rather than beside
    the routing call catches fills from *every* source -- a taker walk on a
    placed order and a resting order filled by ``PaperExchange.advance`` alike
    -- and does it at the one moment the answer is needed. Booking is
    idempotent on the venue's fill id, so re-entering this stage never advances
    the expectation past cash the venue moved once.

    The booking reads the venue's *execution reports*; the cycle reads the
    venue's *aggregate* balances and positions. Those are different questions,
    which is why the comparison can still fail -- see
    :class:`~windbreak.riskkernel.verification.LedgerExpectationSource`.

    Args:
        deps: The tick's dependency bundle.
    """
    deps.fill_bookkeeper.book_new()
    deps.kernel.run_verification_cycle()


def _decide_and_approve(
    deps: PaperTickDeps,
    order_book: OrderBookSnapshot,
    forecast: ForecastRecord | None,
    created_at: datetime,
    heartbeat_epoch_s: int,
) -> tuple[str, int, int]:
    """Select and approve against a forecast, or short-circuit a halted tick.

    Narrows the optional forecast in one place so the select and approve stages
    keep their non-optional contracts. That narrowing is why a halted tick
    needs no fail-closed forecast stamp of its own (issue #380): it never
    reaches an approval at all, which is strictly stronger than vetoing one.

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
    filled = _approve_stage(deps, decision, heartbeat_epoch_s, order_book, forecast)
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
    an audit event to the shared hash-chained ledger.

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
    _verification_stage(deps)
    forecast = _forecast_stage(deps, order_book, created_at)
    forecast_id, intent_count, filled = _decide_and_approve(
        deps, order_book, forecast, created_at, heartbeat_epoch_s
    )
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
        forecast_id=forecast_id,
        intent_count=intent_count,
        filled_centis=filled,
        equity_micros=equity,
        research_halted=forecast is None,
        kernel_halted=mode is Mode.HALT,
    )
