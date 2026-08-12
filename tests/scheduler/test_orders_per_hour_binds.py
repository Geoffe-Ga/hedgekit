"""`velocity_limits`' hourly-order cap binds on real routed orders (#491).

`tests/riskkernel/test_checks.py` already pins the cap's arithmetic to the
order. What it cannot prove is that anything ever *feeds* it: the scheduler
composed every account with `orders_last_hour=0`, so the gate evaluated
`0 + 1 > max_orders_per_hour` -- false for every `max >= 1` -- on every tick of
every run. It ran, it reported approval, and it could not veto however many
orders the loop had just flung at the venue. A fail-*open* hole in the risk
kernel, and the same shape #415 removed from the sibling `notional_today` term.

So these tests exercise the *wiring*, from hash-chained
`OrderTransitionLedgered` rows through the trailing-hour fold and
`build_evaluation_context` into the real `velocity_limits` instance -- and,
because a fold nobody calls is the same defect one layer up, through the very
`_approve_stage` call site that composes the context in production. A test that
built an `AccountState` by hand would pass identically before and after this
change.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import chain, repeat
from typing import TYPE_CHECKING, Any

from windbreak.config.schema import CapitalConfig, RiskConfig, WindbreakConfig
from windbreak.connector.models import (
    ExchangeStatus,
    OrderBookLevel,
    OrderBookSnapshot,
    Position,
)
from windbreak.ledger.events import OrderTransitionLedgered
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips, ProbabilityPpm
from windbreak.order_gateway.ledger_writer import (
    SqliteGatewayLedgerWriter,
    apply_and_ledger,
)
from windbreak.order_gateway.state_machine import OrderEvent, OrderState
from windbreak.riskkernel.checks import DEFAULT_CHECKS, OrderIntent
from windbreak.scheduler.loop import (
    _ORDER_TRANSITION_EVENT_TYPE,
    _REQUEST_SUBMISSION_EVENT,
    _TRAILING_HOUR_SECONDS,
    EquityCurveCursor,
    _approve_stage,
    orders_last_hour_count,
    read_orders_last_hour,
)

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.riskkernel.context import EvaluationContext

#: The tick's instant: 2026-03-01T05:30:00Z. Chosen so the trailing hour spans
#: local midnight on a UTC-05:00 host, where a window bucketed on anything but
#: raw epoch arithmetic would move.
_NOW_EPOCH_S = 1_772_343_000

#: The oldest instant the trailing hour admits: `_NOW_EPOCH_S - 3600`, i.e.
#: 2026-03-01T04:30:00Z.
_HOUR_AGO_EPOCH_S = 1_772_339_400

#: Three instants comfortably inside the trailing hour.
_INSIDE_EARLY = datetime(2026, 3, 1, 4, 45, tzinfo=UTC)
_INSIDE_MIDDLE = datetime(2026, 3, 1, 5, 0, tzinfo=UTC)
_INSIDE_LATE = datetime(2026, 3, 1, 5, 29, tzinfo=UTC)

#: An instant well outside it -- two hours before the tick.
_OUTSIDE = datetime(2026, 3, 1, 3, 30, tzinfo=UTC)

#: The three instants the window's lower edge is decided at, to the second: the
#: order exactly one hour old, the one a second newer, the one a second older.
_EXACTLY_AN_HOUR_OLD = datetime.fromtimestamp(_HOUR_AGO_EPOCH_S, tz=UTC)
_ONE_SECOND_INSIDE = datetime.fromtimestamp(_HOUR_AGO_EPOCH_S + 1, tz=UTC)
_ONE_SECOND_OUTSIDE = datetime.fromtimestamp(_HOUR_AGO_EPOCH_S - 1, tz=UTC)

#: A stamp carrying no offset at all, placed inside the trailing hour when read
#: as UTC. Read on a UTC-05:00 host it is five hours older and falls outside;
#: read on a UTC+05:00 host it is five hours newer and still falls outside the
#: other way. Nothing establishes which, so the fold must refuse it.
_NAIVE_INSIDE = _INSIDE_MIDDLE.replace(tzinfo=None)

#: The configured hourly cap. Three routed orders must refuse the fourth.
_MAX_ORDERS_PER_HOUR = 3

#: The market every fixture below trades.
_TICKER = "KXRAINNYC-26MAR01"

#: The component every fixture stamps on its rows.
_COMPONENT = "order_gateway"


def _submission_request(*, coid: str) -> OrderTransitionLedgered:
    """Build the transition the Gateway writes before touching the venue.

    `REQUEST_SUBMISSION` is recorded in exactly one place, immediately before
    `OrderGateway._place`, so it marks one -- and exactly one -- attempt to
    route an order at the exchange.

    Args:
        coid: The intent's content-addressed client-order-id.

    Returns:
        The assembled :class:`~windbreak.ledger.events.OrderTransitionLedgered`.
    """
    return OrderTransitionLedgered(
        component=_COMPONENT,
        client_order_id=coid,
        from_state="APPROVED",
        event="REQUEST_SUBMISSION",
        to_state="SUBMISSION_REQUESTED",
    )


def _store(tmp_path: Path, stamps: tuple[datetime, ...]) -> SqliteLedgerStore:
    """Open a ledger holding one routed order per stamp.

    The stamp is the store's injected clock reading at append time, so it lands
    in the row's ``created_at`` -- the field the chain hash covers and the fold
    reads. Readings past the fixture's own rows settle on the tick's instant,
    because `_approve_stage` appends an `ExchangeStatusObserved` of its own; that
    row is a live negative control, stamped squarely inside the trailing hour
    and of the wrong event type, so a fold that counted ledger rows rather than
    *routed orders* would over-count every case below.

    Args:
        tmp_path: The directory the database file is created in.
        stamps: The ``created_at`` readings to append rows under, in order.

    Returns:
        The opened store, holding exactly those rows.
    """
    readings = chain(stamps, repeat(datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC)))
    store = SqliteLedgerStore(tmp_path / "ledger.db", now=lambda: next(readings))
    for index in range(len(stamps)):
        store.append(_submission_request(coid=f"coid-{index}"))
    return store


def _config() -> WindbreakConfig:
    """Build a configuration whose hourly order cap is `_MAX_ORDERS_PER_HOUR`.

    Returns:
        The assembled :class:`~windbreak.config.schema.WindbreakConfig`.
    """
    return WindbreakConfig(
        capital=CapitalConfig(floor_micros=0),
        risk=RiskConfig(max_orders_per_hour=_MAX_ORDERS_PER_HOUR),
    )


def _intent() -> OrderIntent:
    """Build the order whose approval the cap decides.

    Returns:
        The assembled :class:`~windbreak.riskkernel.checks.OrderIntent`.
    """
    return OrderIntent(
        intent_id="intent-491",
        market_ticker=_TICKER,
        outcome="yes",
        action="buy",
        price=PricePips(5_000),
        size=ContractCentis(1_000),
        max_notional=MoneyMicros(1_000_000_000),
        implied_probability=ProbabilityPpm(500_000),
        idempotency_key="key-491",
    )


def _order_book() -> OrderBookSnapshot:
    """Build a deep two-sided book stamped at the tick's instant.

    Returns:
        The assembled :class:`~windbreak.connector.models.OrderBookSnapshot`.
    """
    level = OrderBookLevel(price=PricePips(5_000), quantity=ContractCentis(1_000_000))
    return OrderBookSnapshot(
        ticker=_TICKER,
        yes_bids=(level,),
        yes_asks=(level,),
        fetched_at=datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC),
    )


@dataclass
class _StubCandidate:
    """The two `MarketCandidate` attributes `_approve_stage` reads.

    Attributes:
        ticker: The market being approved for.
        order_book: The book the participation cap is measured against.
    """

    ticker: str
    order_book: OrderBookSnapshot


@dataclass
class _StubDecision:
    """The one `SelectorDecision` attribute `_approve_stage` reads.

    Attributes:
        intents: The intents this tick emitted.
    """

    intents: tuple[OrderIntent, ...]


@dataclass
class _StubForecast:
    """The one `ForecastRecord` attribute `_approve_stage` reads.

    Attributes:
        created_at: The instant the forecast was created.
    """

    created_at: datetime


@dataclass
class _StubExchange:
    """A venue answering only the four calls `_approve_stage` makes.

    Every answer is deliberately permissive, so no check other than the one
    under test can supply the veto.
    """

    def get_market(self, ticker: str) -> None:
        """Decline to describe the market, which no cap under test reads.

        Args:
            ticker: The market asked about.

        Returns:
            ``None`` -- the eligibility projections fail closed on it, and
            `velocity_limits` never consults them.
        """
        del ticker
        return None

    def get_exchange_status(self) -> ExchangeStatus:
        """Report the venue open, observed at the tick's instant.

        Returns:
            The assembled :class:`~windbreak.connector.models.ExchangeStatus`.
        """
        return ExchangeStatus(
            status="open", fetched_at=datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC)
        )

    def get_exchange_time(self) -> datetime:
        """Report the venue's clock in perfect agreement with ours.

        Returns:
            The tick's instant.
        """
        return datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC)

    def get_positions(self) -> tuple[Position, ...]:
        """Report a flat account.

        Returns:
            An empty tuple -- an observed absence of holdings.
        """
        return ()


@dataclass
class _StubKernel:
    """The one `RiskKernel` attribute `_approve_stage` reads.

    Attributes:
        latest_verification: The kernel's newest verification snapshot.
    """

    latest_verification: None = None


@dataclass
class _CapturingApproval:
    """An approval seam that records the context and mints nothing.

    Attributes:
        contexts: Every context `_approve_stage` composed, in order.
    """

    contexts: list[EvaluationContext] = field(default_factory=list)

    def decide(self, intent: OrderIntent, context: EvaluationContext) -> Any:
        """Record the composed context and veto.

        Args:
            intent: The intent under approval, unused.
            context: The context `_approve_stage` composed.

        Returns:
            An outcome carrying no token, so no order is ever routed.
        """
        del intent
        self.contexts.append(context)
        return _NO_TOKEN


@dataclass(frozen=True)
class _NoTokenOutcome:
    """The `ApprovalOutcome` field `_approve_stage` reads.

    Attributes:
        token: The minted approval token, always absent here.
    """

    token: None = None


#: The single vetoing outcome the capturing seam hands back.
_NO_TOKEN = _NoTokenOutcome()


@dataclass
class _StubDeps:
    """The `PaperTickDeps` attributes `_approve_stage` reads.

    Attributes:
        config: The configuration supplying the hourly cap.
        store: The real hash-chained ledger the fold reads.
        exchange: The venue double.
        approval: The capturing approval seam.
        kernel: The kernel double supplying the verification snapshot.
        clock: The injected epoch-second clock.
        equity_curve_cursor: The per-process memo the real `PaperTickDeps`
            carries and `_approve_stage` threads into `read_equity_curve`
            (issue #516). Nothing in this module measures it -- it is here
            because the stage genuinely reads it, and a double that omitted it
            would be a double the production call site cannot run against.
    """

    config: WindbreakConfig
    store: SqliteLedgerStore
    exchange: _StubExchange
    approval: _CapturingApproval
    kernel: _StubKernel
    equity_curve_cursor: EquityCurveCursor = field(default_factory=EquityCurveCursor)

    def clock(self) -> int:
        """Return the tick's fixed instant.

        Returns:
            `_NOW_EPOCH_S`.
        """
        return _NOW_EPOCH_S


def _composed_context(store: SqliteLedgerStore) -> EvaluationContext:
    """Drive the real `_approve_stage` and return the context it composed.

    This is the composition proof: nothing here hand-builds an `AccountState`
    or calls `build_evaluation_context` directly. The count reaches the check
    only if the production call site folds it out of `store`.

    Args:
        store: The ledger holding this hour's routed orders.

    Returns:
        The single :class:`~windbreak.riskkernel.context.EvaluationContext`
        `_approve_stage` composed for the tick's one intent.
    """
    approval = _CapturingApproval()
    deps: Any = _StubDeps(
        config=_config(),
        store=store,
        exchange=_StubExchange(),
        approval=approval,
        kernel=_StubKernel(),
    )
    candidate: Any = _StubCandidate(ticker=_TICKER, order_book=_order_book())
    decision: Any = _StubDecision(intents=(_intent(),))
    forecast: Any = _StubForecast(
        created_at=datetime.fromtimestamp(_NOW_EPOCH_S, tz=UTC)
    )
    _approve_stage(deps, candidate, decision, _NOW_EPOCH_S, forecast, None)
    assert len(approval.contexts) == 1
    return approval.contexts[0]


def _verdict(context: EvaluationContext) -> str | None:
    """Run the real `velocity_limits` over a composed context.

    Args:
        context: The context `_approve_stage` composed.

    Returns:
        The check's veto reason, or ``None`` when it approved.
    """
    fees = dataclasses.replace(
        context.fees,
        max_trading_fee=MoneyMicros(0),
        max_settlement_fee=MoneyMicros(0),
    )
    check = next(c for c in DEFAULT_CHECKS if c.name == "velocity_limits")
    result = check(_intent(), dataclasses.replace(context, fees=fees))
    return result.reason if result.vetoed else None


class TestTheLoopFeedsTheHourlyCap:
    """The count the production call site composes is the ledger's own."""

    def test_the_composed_account_carries_this_hours_routed_orders(
        self, tmp_path: Path
    ) -> None:
        """Three `REQUEST_SUBMISSION` rows in the hour compose as three."""
        store = _store(tmp_path, (_INSIDE_EARLY, _INSIDE_MIDDLE, _INSIDE_LATE))
        assert _composed_context(store).account.orders_last_hour == 3

    def test_orders_older_than_the_hour_are_not_carried(self, tmp_path: Path) -> None:
        """A row two hours old leaves the trailing-hour count at zero."""
        store = _store(tmp_path, (_OUTSIDE,))
        assert _composed_context(store).account.orders_last_hour == 0


class TestTheHourlyCapBindsBothWays:
    """The pair that proves the cap is alive rather than merely present."""

    def test_vetoes_when_the_hour_is_already_at_the_cap(self, tmp_path: Path) -> None:
        """3 routed + 1 == 4 > 3 configured: veto, by exact reason."""
        store = _store(tmp_path, (_INSIDE_EARLY, _INSIDE_MIDDLE, _INSIDE_LATE))
        assert _verdict(_composed_context(store)) == "hourly order cap exceeded"

    def test_approves_one_order_below_the_cap(self, tmp_path: Path) -> None:
        """2 routed + 1 == 3 == 3 configured: approve.

        The half that stops "fail closed" from degenerating into "veto
        everything". Identical to the case above but for one routed order.
        """
        store = _store(tmp_path, (_INSIDE_EARLY, _INSIDE_MIDDLE))
        assert _verdict(_composed_context(store)) is None

    def test_the_cases_rest_on_the_arithmetic_they_claim(self) -> None:
        """Pin the figures, so a fixture edit cannot silently align the pair."""
        assert _MAX_ORDERS_PER_HOUR + 1 > _MAX_ORDERS_PER_HOUR
        assert (_MAX_ORDERS_PER_HOUR - 1) + 1 == _MAX_ORDERS_PER_HOUR
        assert _NOW_EPOCH_S - _HOUR_AGO_EPOCH_S == _TRAILING_HOUR_SECONDS


class TestTheWindowsLowerEdgeIsDecidedToTheSecond:
    """Where a cap's off-by-one lives, driven through the real composition.

    Each case holds two orders unambiguously inside the hour and moves a third
    across the boundary by one second, so exactly one second of window width
    separates a veto from an approval.
    """

    def test_an_order_exactly_one_hour_old_is_still_inside(
        self, tmp_path: Path
    ) -> None:
        """The window is closed on its lower edge: `epoch >= now - 3600`."""
        store = _store(tmp_path, (_EXACTLY_AN_HOUR_OLD, _INSIDE_MIDDLE, _INSIDE_LATE))
        context = _composed_context(store)
        assert context.account.orders_last_hour == 3
        assert _verdict(context) == "hourly order cap exceeded"

    def test_one_second_inside_the_edge_is_counted(self, tmp_path: Path) -> None:
        """A second newer than the cutoff is unambiguously in the hour."""
        store = _store(tmp_path, (_ONE_SECOND_INSIDE, _INSIDE_MIDDLE, _INSIDE_LATE))
        context = _composed_context(store)
        assert context.account.orders_last_hour == 3
        assert _verdict(context) == "hourly order cap exceeded"

    def test_one_second_outside_the_edge_is_not_counted(self, tmp_path: Path) -> None:
        """A second older than the cutoff has aged out, and the order routes.

        The discriminating half of the boundary: identical to the case above
        but for two seconds of stamp, and it approves. Without it, a window an
        hour and a day wide would pass every veto case here.
        """
        store = _store(tmp_path, (_ONE_SECOND_OUTSIDE, _INSIDE_MIDDLE, _INSIDE_LATE))
        context = _composed_context(store)
        assert context.account.orders_last_hour == 2
        assert _verdict(context) is None


class TestTheWindowDoesNotMoveWithTheHostsTimezone:
    """CI runs UTC, where a locally-bucketed window and a correct one agree.

    The tick's instant sits at 00:30 local on a UTC-05:00 host, so its trailing
    hour spans local midnight -- the boundary a calendar-derived window would
    trip over, the way the sibling daily fold genuinely can. Both pinned hosts
    must return the identical count and the identical verdict.
    """

    def test_the_count_is_identical_west_of_utc(
        self, tmp_path: Path, local_timezone_utc_minus_5: None
    ) -> None:
        """UTC-05:00, where the tick's local wall clock reads 00:30."""
        del local_timezone_utc_minus_5
        store = _store(tmp_path, (_EXACTLY_AN_HOUR_OLD, _INSIDE_MIDDLE, _INSIDE_LATE))
        context = _composed_context(store)
        assert context.account.orders_last_hour == 3
        assert _verdict(context) == "hourly order cap exceeded"

    def test_the_count_is_identical_east_of_utc(
        self, tmp_path: Path, local_timezone_utc_plus_5: None
    ) -> None:
        """UTC+05:00, which skews a misread instant the opposite way."""
        del local_timezone_utc_plus_5
        store = _store(tmp_path, (_EXACTLY_AN_HOUR_OLD, _INSIDE_MIDDLE, _INSIDE_LATE))
        context = _composed_context(store)
        assert context.account.orders_last_hour == 3
        assert _verdict(context) == "hourly order cap exceeded"

    def test_the_boundary_still_releases_west_of_utc(
        self, tmp_path: Path, local_timezone_utc_minus_5: None
    ) -> None:
        """The approving direction holds under the pin too, to the second."""
        del local_timezone_utc_minus_5
        store = _store(tmp_path, (_ONE_SECOND_OUTSIDE, _INSIDE_MIDDLE, _INSIDE_LATE))
        context = _composed_context(store)
        assert context.account.orders_last_hour == 2
        assert _verdict(context) is None


class TestAnUnprovableHourFailsClosed:
    """An hour whose routing history cannot be established must not pass."""

    def test_a_naive_stamp_folds_to_none_rather_than_assumed_utc(
        self, tmp_path: Path
    ) -> None:
        """`require_aware` refuses rather than repairs, so the hour is unknown."""
        store = _store(tmp_path, (_NAIVE_INSIDE,))
        assert read_orders_last_hour(store, now_epoch_s=_NOW_EPOCH_S) is None

    def test_a_naive_stamp_is_refused_west_of_utc_too(
        self, tmp_path: Path, local_timezone_utc_minus_5: None
    ) -> None:
        """A fix must refuse the unprovable instant on either host.

        A guard that merely happens to be conservative on one host is not a
        guard; the refusal has to be the same fact everywhere.
        """
        del local_timezone_utc_minus_5
        store = _store(tmp_path, (_NAIVE_INSIDE,))
        assert read_orders_last_hour(store, now_epoch_s=_NOW_EPOCH_S) is None

    def test_one_naive_row_makes_the_whole_hour_unprovable(
        self, tmp_path: Path
    ) -> None:
        """An hour is not partially provable: the unreadable row may be in it.

        Counting only the readable rows would report a tally smaller than the
        evidence supports, and under-reporting a cap's consumption is exactly
        the permissive direction.
        """
        store = _store(tmp_path, (_INSIDE_EARLY, _NAIVE_INSIDE))
        assert read_orders_last_hour(store, now_epoch_s=_NOW_EPOCH_S) is None

    def test_an_unprovable_hour_zeroes_the_configured_cap(self, tmp_path: Path) -> None:
        """`None` states the fact in the limits, not as a fabricated zero.

        `AccountState.orders_last_hour` is a plain `int` with no `None`, so
        carrying "unprovable" there would mean writing the very zero this issue
        removes -- the seam #414 established for exposure and #415 for the day.
        """
        store = _store(tmp_path, (_NAIVE_INSIDE,))
        assert _composed_context(store).limits.max_orders_per_hour == 0

    def test_an_unprovable_hour_vetoes_the_next_order(self, tmp_path: Path) -> None:
        """The zeroed cap is what the check reads: `0 + 1 > 0` refuses."""
        store = _store(tmp_path, (_NAIVE_INSIDE,))
        assert _verdict(_composed_context(store)) == "hourly order cap exceeded"

    def test_a_provable_hour_keeps_the_configured_cap(self, tmp_path: Path) -> None:
        """The zeroing is conditional on absent evidence, not unconditional."""
        store = _store(tmp_path, (_INSIDE_EARLY, _INSIDE_MIDDLE))
        context = _composed_context(store)
        assert context.limits.max_orders_per_hour == _MAX_ORDERS_PER_HOUR
        assert context.account.orders_last_hour == 2


class TestOnlyRoutedOrdersCount:
    """The fold counts placements, not an order's progress through its states."""

    def test_the_rows_the_real_gateway_writes_are_the_rows_counted(
        self, tmp_path: Path
    ) -> None:
        """A positive control written by production's own ledger writer.

        `apply_and_ledger` is what `OrderGateway._submit_new` calls, through the
        same `SqliteGatewayLedgerWriter` `build_paper_deps` wires to this very
        store. Building the row by hand instead would prove only that the fold
        agrees with this module's fixture, which is the corpus-scan-over-zero-
        hits failure one layer along: the fold would keep passing after a
        payload rename that stopped production ever being counted.
        """
        store = SqliteLedgerStore(tmp_path / "ledger.db", now=lambda: _INSIDE_MIDDLE)
        apply_and_ledger(
            SqliteGatewayLedgerWriter(store),
            OrderState.APPROVED,
            OrderEvent.REQUEST_SUBMISSION,
            client_order_id="coid-real",
        )
        assert read_orders_last_hour(store, now_epoch_s=_NOW_EPOCH_S) == 1

    def test_the_later_lifecycle_edges_of_one_order_are_not_re_counted(
        self, tmp_path: Path
    ) -> None:
        """One placement is one order however many transitions follow it.

        The negative half of the control above, and the reason the fold keys on
        `REQUEST_SUBMISSION` rather than on the row type: a single order that
        submits, acks and fills writes four transitions, and a cap counting
        those would refuse the second order of an hour whose limit is three.
        """
        store = SqliteLedgerStore(tmp_path / "ledger.db", now=lambda: _INSIDE_MIDDLE)
        writer = SqliteGatewayLedgerWriter(store)
        state = apply_and_ledger(
            writer,
            OrderState.INTENT_CREATED,
            OrderEvent.APPROVE,
            client_order_id="coid-real",
        )
        state = apply_and_ledger(
            writer, state, OrderEvent.REQUEST_SUBMISSION, client_order_id="coid-real"
        )
        state = apply_and_ledger(
            writer, state, OrderEvent.SUBMIT, client_order_id="coid-real"
        )
        apply_and_ledger(writer, state, OrderEvent.ACK, client_order_id="coid-real")
        assert read_orders_last_hour(store, now_epoch_s=_NOW_EPOCH_S) == 1

    def test_rows_of_other_event_types_are_ignored(self, tmp_path: Path) -> None:
        """A ledger holding no transition at all reports a provable zero.

        The `ExchangeStatusObserved` row `_approve_stage` appends is stamped
        inside the trailing hour on every case in this module, so a fold that
        counted rows rather than routed orders would over-count everywhere.
        """
        store = _store(tmp_path, ())
        assert read_orders_last_hour(store, now_epoch_s=_NOW_EPOCH_S) == 0
        assert _composed_context(store).account.orders_last_hour == 0


class TestTheTwoReadPathsAgree:
    """The indexed walk is an optimization, never a behavioral fork."""

    def test_the_reverse_walk_and_the_whole_ledger_fold_agree(
        self, tmp_path: Path
    ) -> None:
        """Both paths answer identically on a mixed, boundary-straddling hour."""
        store = _store(
            tmp_path,
            (_OUTSIDE, _ONE_SECOND_OUTSIDE, _EXACTLY_AN_HOUR_OLD, _INSIDE_LATE),
        )
        assert read_orders_last_hour(store, now_epoch_s=_NOW_EPOCH_S) == 2
        assert read_orders_last_hour(store, now_epoch_s=_NOW_EPOCH_S) == (
            orders_last_hour_count(store.read_all(), now_epoch_s=_NOW_EPOCH_S)
        )

    def test_both_paths_refuse_the_same_unprovable_hour(self, tmp_path: Path) -> None:
        """The refusal is a property of the fold, not of one read path."""
        store = _store(tmp_path, (_INSIDE_EARLY, _NAIVE_INSIDE))
        assert read_orders_last_hour(store, now_epoch_s=_NOW_EPOCH_S) is None
        assert (
            orders_last_hour_count(store.read_all(), now_epoch_s=_NOW_EPOCH_S) is None
        )


class TestTheFoldsConstantsAreDerivedNotRestated:
    """A hand-copied literal is a fold that silently stops matching production."""

    def test_the_event_type_is_the_one_the_gateway_writes(self) -> None:
        """Renaming the event class must break this, not the risk cap."""
        written = OrderTransitionLedgered(
            component=_COMPONENT,
            client_order_id="coid-0",
            from_state="APPROVED",
            event="REQUEST_SUBMISSION",
            to_state="SUBMISSION_REQUESTED",
        )
        assert written.event_type == _ORDER_TRANSITION_EVENT_TYPE

    def test_the_routing_edge_is_the_state_machines_own_name(self) -> None:
        """`apply_and_ledger` records `OrderEvent.name`, so that is the key."""
        assert OrderEvent.REQUEST_SUBMISSION.name == _REQUEST_SUBMISSION_EVENT

    def test_the_window_is_an_hour(self) -> None:
        """The cap is `max_orders_per_hour`; the window must be an hour wide."""
        assert _TRAILING_HOUR_SECONDS == 60 * 60
