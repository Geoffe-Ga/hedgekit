"""SPEC S7.5 :class:`PaperExchange`: a replay-driven pessimistic paper trader.

:class:`PaperExchange` is a :class:`~windbreak.connector.interface.MarketConnector`
that simulates trading against *recorded* market data. A session is a per-ticker
sequence of steps, each pairing an order-book snapshot with the trade prints
that followed it. A crossing order fills immediately via the pessimistic taker
walk (:func:`windbreak.connector.fills.walk_taker_fill`); its unfilled remainder
rests, and resting orders fill on later steps only when the recorded market
*trades through* their limit (:func:`windbreak.connector.fills.resting_fill_quantity`).

The connector reuses :mod:`windbreak.connector.fake`'s JSON loader helpers for
the static market/balance/fee fixtures (DRY) and adds a ``sessions.json`` reader
for the book-and-print replay. The fixture balance is the account's *opening*
balance only: :meth:`PaperExchange.get_positions` and
:meth:`PaperExchange.get_balances` are both folded from this exchange's own
fill log, so what the exchange reports holding and what it reports owning are
consequences of what it actually traded (issue #352).

That independence is the point. ``LedgerExpectationSource`` seeds each
verification dimension from ``component == "riskkernel"`` events, and the PAPER
loop stamps ``"scheduler"`` on everything it writes, so every dimension falls
back to this connector. Had positions stayed a hardcoded ``()`` and balances a
frozen fixture, the ledger-derived *expectation* and the exchange-reported
*observation* would be the same constant and a reconciliation between them
could not fail for any reason -- checks that structurally cannot fail are worse
than no checks at all.

Consistency guard (issues #18, #362): the constructor rejects any fixture whose
:class:`~windbreak.connector.semantics.BalanceSemantics` claims a behavior this
simulator does not implement (see :data:`_SEMANTICS_REQUIREMENTS`). A taker walk
spans multiple book levels, each needing its own
:class:`~windbreak.connector.models.Fill` price, so partial fills must be
``PER_FILL_RECORDS``; and the three collateral answers are implemented exactly
one way each by :meth:`PaperExchange.get_balances` and
:meth:`PaperExchange.cancel_order`. A semantics record that disagrees with the
behavior is worse than one that admits ignorance: ``UNKNOWN`` makes a consumer
fail closed, a confidently wrong flag makes it fail open.

This package is float-denylisted by ``scripts/lint_no_floats.py``: no ``/`` true
division, no ``float`` literal, cast, or annotation appears here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from windbreak.connector.fake import (
    _book_from_dict,
    _load_balance_semantics,
    _load_balances,
    _load_exchange,
    _load_fee_models,
    _load_markets,
    _parse_dt,
    _read_json,
)
from windbreak.connector.fills import (
    DEFAULT_FEE_HAIRCUT_PPM,
    DEFAULT_MAX_PARTICIPATION_PPM,
    PAPER_FILL_MODEL_VERSION,
    RestingFillRequest,
    TradePrint,
    allocate_resting_fills,
    trade_through_fill_ts,
    walk_taker_fill,
)
from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.models import (
    BalanceSnapshot,
    Fill,
    OpenOrder,
    OrderBookLevel,
    Position,
)
from windbreak.connector.semantics import (
    CancelCollateralRelease,
    OrderCollateralInAvailable,
    OrderCollateralInTotal,
    PartialFillRepresentation,
)
from windbreak.numeric import (
    ContractCentis,
    MoneyMicros,
    PricePips,
    RoundingDirection,
    divide,
    money_from_price_and_count,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from enum import Enum
    from typing import Any, Literal

    from windbreak.connector.fees import FeeModel
    from windbreak.connector.models import (
        BalanceSemantics,
        ExchangeStatus,
        NormalizedMarket,
        OrderBookSnapshot,
    )

__all__ = [
    "COMPLEMENT_PIPS",
    "PAPER_FILL_MODEL_VERSION",
    "PaperExchange",
    "PaperOrderIntent",
    "TwoSidedPositionError",
]

#: Fee-schedule key used when a market has no ticker-specific schedule (mirrors
#: :data:`windbreak.connector.fake._DEFAULT_FEE_KEY`).
_DEFAULT_FEE_KEY: Final[str] = "default"

#: Pips in a full ``$1`` payout: a NO price is the complement ``10_000 - yes``.
#: Public because :func:`_position_row` projects a NO holding into the YES frame
#: through it, and any consumer that must mark such a row back in its own side's
#: frame -- the PAPER loop's equity sample does
#: (``windbreak/scheduler/loop.py::_position_value_micros``) -- has to invert the
#: exact same complement rather than restate the constant.
COMPLEMENT_PIPS: Final[int] = 10_000

#: Cash leaving the account always rounds *up*: an overstated debit is the safe
#: error, mirroring :mod:`windbreak.connector.fills`'s money rounding. Every
#: notional here is an exact ``price * quantity`` product, so this is inert in
#: practice -- it is stated at the call site because the money path requires the
#: direction to be visible (``scripts/lint_no_floats.py`` forbids bare ``//``).
_MONEY_ROUNDING: Final[RoundingDirection] = RoundingDirection.OVERSTATE_COST

#: An average entry price rounds *down*, so a reported holding is never marked
#: at more than it cost. This is now the *only* position fold in the codebase:
#: the PAPER loop's own YES-only ``notional // quantity`` projection was deleted
#: by issue #361 in favour of reading :meth:`PaperExchange.get_positions`, so
#: there is no second definition of "position" left to keep in step.
_PRICE_ROUNDING: Final[RoundingDirection] = RoundingDirection.UNDERSTATE_EQUITY


@dataclass(frozen=True, slots=True)
class _SemanticsRequirement:
    """One balance-semantics answer this simulator implements in exactly one way.

    Attributes:
        field: The :class:`~windbreak.connector.semantics.BalanceSemantics`
            field being constrained.
        required: The only member of that field's enum whose meaning this
            simulator's behavior actually matches.
        because: The behavior that pins it, quoted into the rejection message so
            a fixture author is told *why* the fixture cannot be simulated.
    """

    field: str
    required: Enum
    because: str


#: The answers :class:`PaperExchange` hard-codes by construction. A fixture that
#: claims a different one is refused rather than simulated wrong (issues #18,
#: #362): ``get_balance_semantics`` is the contract a consumer reads to decide
#: how much of ``available`` it may commit, so it must describe this class's
#: real behavior and not the venue a fixture author wishes were being replayed.
_SEMANTICS_REQUIREMENTS: Final[tuple[_SemanticsRequirement, ...]] = (
    _SemanticsRequirement(
        field="partial_fill_representation",
        required=PartialFillRepresentation.PER_FILL_RECORDS,
        because="the taker walk emits one Fill per consumed book level",
    ),
    _SemanticsRequirement(
        field="open_order_collateral_in_available",
        required=OrderCollateralInAvailable.DEDUCTED_FROM_AVAILABLE,
        because="get_balances withholds every resting order's collateral from "
        "available",
    ),
    _SemanticsRequirement(
        field="open_order_collateral_in_total",
        required=OrderCollateralInTotal.INCLUDED,
        because="get_balances debits total only when a fill executes, so a "
        "resting order's collateral is still inside total",
    ),
    _SemanticsRequirement(
        field="cancel_collateral_release",
        required=CancelCollateralRelease.IMMEDIATE,
        because="cancel_order drops the resting order outright, so the next "
        "get_balances already counts its collateral as free",
    ),
)


def _require_implemented_semantics(semantics: BalanceSemantics) -> None:
    """Reject a semantics record this simulator would misreport.

    Args:
        semantics: The fixture's balance-interpretation semantics.

    Raises:
        ValueError: If any field in :data:`_SEMANTICS_REQUIREMENTS` holds a
            member other than the one this class implements. The message names
            the field, the required member, and the behavior that pins it.
    """
    for requirement in _SEMANTICS_REQUIREMENTS:
        actual = getattr(semantics, requirement.field)
        if actual is not requirement.required:
            raise ValueError(
                f"{requirement.field} must be {requirement.required.name}, got "
                f"{actual.name}; {requirement.because}"
            )


class TwoSidedPositionError(RuntimeError):
    """Raised when one ticker's fill log carries both a YES and a NO leg.

    :class:`~windbreak.connector.models.Position` carries a single signed
    quantity per ticker, and
    :class:`~windbreak.riskkernel.verification.LedgerExpectationSource` keys its
    expectation by ticker, so a two-sided holding can only be reported by
    netting the legs -- and a netted one-YES-plus-one-NO reports as *flat* while
    both legs and their collateral are still live. That is precisely the
    fabricated "healthy zero" a reconciliation must never be handed, so the
    simulator refuses to report at all rather than report a state it cannot
    represent.

    Refusing is safe for the kernel's own cycle:
    :meth:`~windbreak.riskkernel.verification.ReadOnlyVerifier.run_cycle` grades
    a connector that raises as a ``BREACH`` and records it before alerting, so
    this error halts the kernel through the audited path instead of escaping the
    heartbeat loop.

    Attributes:
        ticker: The market held on both sides.
    """

    def __init__(self, ticker: str) -> None:
        """Build the error for a two-sided ``ticker``.

        Args:
            ticker: The market whose fill log carries both a YES and a NO leg.
        """
        self.ticker = ticker
        super().__init__(
            f"{ticker} is held on both the YES and the NO side; a single-row "
            "Position cannot represent that, and netting the legs would report "
            "a flat account while both legs are still open"
        )


@dataclass(slots=True)
class _SideTotals:
    """Running totals for one ``(ticker, side)`` slice of the fill log.

    Attributes:
        quantity_centis: The side's total filled size, in contract-centis.
        notional_micros: The side's total filled notional, in micros.
    """

    quantity_centis: int = 0
    notional_micros: int = 0


def _position_row(
    ticker: str, side: Literal["yes", "no"], totals: _SideTotals
) -> Position:
    """Project one side's fill totals into a YES-frame :class:`Position` row.

    Positions are reported in the YES frame -- positive is long YES, negative is
    long NO -- because a ticker gets exactly one row and a consumer reading only
    ``quantity`` must not mistake a NO holding for a YES one. A long NO is
    economically a short YES, so ``N`` NO contracts bought at ``p`` report as
    ``-N`` at the complement price ``10_000 - p``.

    :data:`_PRICE_ROUNDING` stays conservative through that complement:
    subtracting a *floored* NO average from ``10_000`` yields a *ceiling* on the
    reported YES-frame price, and marking a short position at the higher end of
    its price band is the direction that understates equity -- the same
    conservatism the YES row gets from flooring directly.

    Args:
        ticker: The market the holding is in.
        side: The side every folded fill was on.
        totals: That side's filled size and notional.

    Returns:
        The YES-frame position row.
    """
    average = divide(
        totals.notional_micros, totals.quantity_centis, rounding=_PRICE_ROUNDING
    )
    if side == "yes":
        return Position(
            ticker=ticker,
            quantity=ContractCentis(totals.quantity_centis),
            average_price=PricePips(average),
        )
    return Position(
        ticker=ticker,
        quantity=ContractCentis(-totals.quantity_centis),
        average_price=PricePips(COMPLEMENT_PIPS - average),
    )


@dataclass(frozen=True, slots=True)
class PaperOrderIntent:
    """A normalized order intent accepted by :meth:`PaperExchange.place_order`.

    Attributes:
        ticker: The market to trade.
        side: Whether the order buys the YES or the NO side.
        price: The order's limit price, in pips (a YES or NO price per ``side``).
        quantity: The requested size, in contract-centis.
    """

    ticker: str
    side: Literal["yes", "no"]
    price: PricePips
    quantity: ContractCentis


def _utc_now() -> datetime:
    """Return the current instant as a timezone-aware UTC datetime.

    Returns:
        The current UTC instant.
    """
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _SessionStep:
    """One replay step: a book snapshot and the prints that followed it.

    Attributes:
        book: The order-book snapshot recorded at this step.
        trades: The trade prints recorded during this step, in order.
    """

    book: OrderBookSnapshot
    trades: tuple[TradePrint, ...]


@dataclass(frozen=True, slots=True)
class PaperPlacement:
    """The receipt returned by :meth:`PaperExchange.place_order`.

    Attributes:
        fills: The fills emitted by the immediate taker walk (one per consumed
            book level), in walk order.
        resting_order: The order left resting for the unfilled remainder, or
            None when the order filled completely.
    """

    fills: tuple[Fill, ...]
    resting_order: OpenOrder | None


def _trade_print_from_dict(data: Mapping[str, Any]) -> TradePrint:
    """Build a :class:`TradePrint` from one raw ``trades`` fixture entry."""
    return TradePrint(
        price=PricePips(data["price"]),
        quantity=ContractCentis(data["quantity"]),
        ts=_parse_dt(data["ts"]),
    )


def _session_step_from_dict(ticker: str, data: Mapping[str, Any]) -> _SessionStep:
    """Build a :class:`_SessionStep` from one raw session-step fixture entry."""
    return _SessionStep(
        book=_book_from_dict(ticker, data["book"]),
        trades=tuple(_trade_print_from_dict(trade) for trade in data["trades"]),
    )


def _load_sessions(directory: Path) -> dict[str, tuple[_SessionStep, ...]]:
    """Load ``sessions.json`` into a ticker-keyed tuple of replay steps."""
    data = _read_json(directory.joinpath("sessions.json"))
    return {
        ticker: tuple(_session_step_from_dict(ticker, step) for step in steps)
        for ticker, steps in data.items()
    }


class PaperExchange:
    """A replay-driven, pessimistic paper-trading :class:`MarketConnector`.

    The connector replays a per-ticker session of recorded books and trade
    prints. Crossing orders fill immediately via the pessimistic taker walk;
    remainders rest and fill only on a later step's trade-through. Positions and
    balances are both folded from the resulting fill log (issue #352), so they
    move only when the simulator actually trades -- and ``available`` also
    withholds every resting order's collateral (issue #362), so the balance
    semantics this connector advertises is the one it implements.

    Attributes:
        markets: Ticker-keyed normalized markets.
        sessions: Ticker-keyed replay steps.
        exchange_status: The exchange's trading status.
        exchange_time: The exchange's server time.
        balances: The account's *opening* balances, before any fill is debited;
            :meth:`get_balances` reports the current ones.
        balance_semantics: The venue's balance-interpretation semantics.
        fee_models: Fee schedules keyed by ticker (plus a ``default``).
        haircut_ppm: The slippage haircut applied to modeled fees, in ppm.
        max_participation_ppm: The participation cap on recorded depth, in ppm.
    """

    def __init__(
        self,
        *,
        markets: Mapping[str, NormalizedMarket],
        sessions: Mapping[str, tuple[_SessionStep, ...]],
        exchange_status: ExchangeStatus,
        exchange_time: datetime,
        clock: Callable[[], datetime] = _utc_now,
        balances: BalanceSnapshot,
        balance_semantics: BalanceSemantics,
        fee_models: Mapping[str, FeeModel],
        haircut_ppm: int = DEFAULT_FEE_HAIRCUT_PPM,
        max_participation_ppm: int = DEFAULT_MAX_PARTICIPATION_PPM,
    ) -> None:
        """Initialize the paper exchange and validate fill-record consistency.

        Args:
            markets: Ticker-keyed normalized markets.
            sessions: Ticker-keyed replay steps.
            exchange_status: The exchange's trading status. Only its ``status``
                value is used; its ``fetched_at`` is restamped per observation.
            exchange_time: The exchange's server time.
            clock: Observation clock stamped onto each
                :meth:`get_exchange_status` reading (issue #342).
            balances: The account's opening balances, which every fill debits.
            balance_semantics: The venue's balance-interpretation semantics.
            fee_models: Fee schedules keyed by ticker (plus a ``default``).
            haircut_ppm: The slippage haircut on modeled fees, in ppm.
            max_participation_ppm: The participation cap on recorded depth, in ppm.

        Raises:
            ValueError: If ``balance_semantics`` answers any question in
                :data:`_SEMANTICS_REQUIREMENTS` differently from the way this
                class behaves -- aggregated partial fills would lose the taker
                walk's per-level prices (issue #18), and a collateral answer
                this simulator does not implement would make
                :meth:`get_balance_semantics` a lying flag (issue #362).
        """
        _require_implemented_semantics(balance_semantics)
        self.markets = markets
        self.sessions = sessions
        self.exchange_status = exchange_status
        self.exchange_time = exchange_time
        self._clock = clock
        self.balances = balances
        self.balance_semantics = balance_semantics
        self.fee_models = fee_models
        self.haircut_ppm = haircut_ppm
        self.max_participation_ppm = max_participation_ppm
        self._cursor: dict[str, int] = dict.fromkeys(sessions, 0)
        self._resting: list[OpenOrder] = []
        self._fills: list[Fill] = []
        self._order_seq = 0
        self._fill_seq = 0

    @classmethod
    def from_fixture_dir(
        cls,
        path: str | Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
        **overrides: int,
    ) -> PaperExchange:
        """Build a :class:`PaperExchange` from a fixture directory.

        Loads the :class:`~windbreak.connector.fake.FakeExchange`-shaped static
        fixtures plus a ``sessions.json`` replay. The fill log starts empty, so
        the account starts flat at the fixture's opening balance; the simulator
        produces its own fills -- and therefore its own positions and debits --
        as it runs.

        Args:
            path: The directory holding the JSON fixtures and ``sessions.json``.
            clock: Observation clock forwarded to the constructor (issue #342).
            **overrides: Optional ``haircut_ppm`` / ``max_participation_ppm``
                keyword overrides forwarded to the constructor.

        Returns:
            A fully loaded paper exchange positioned at each session's step 0.
        """
        directory = Path(path)
        status, exchange_time = _load_exchange(directory)
        return cls(
            markets=_load_markets(directory),
            sessions=_load_sessions(directory),
            exchange_status=status,
            exchange_time=exchange_time,
            clock=clock,
            balances=_load_balances(directory),
            balance_semantics=_load_balance_semantics(directory),
            fee_models=_load_fee_models(directory),
            **overrides,
        )

    def _steps_for(self, ticker: str) -> tuple[_SessionStep, ...]:
        """Return the replay steps for ``ticker`` or raise ``UnknownMarketError``."""
        try:
            return self.sessions[ticker]
        except KeyError as exc:
            raise UnknownMarketError(ticker) from exc

    def _current_step(self, ticker: str) -> _SessionStep:
        """Return the step at ``ticker``'s current cursor (clamped to the last)."""
        steps = self._steps_for(ticker)
        index = min(self._cursor[ticker], len(steps) - 1)
        return steps[index]

    def list_markets(self) -> tuple[NormalizedMarket, ...]:
        """Return every market the session offers."""
        return tuple(self.markets.values())

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Return the market for ``ticker`` or raise ``UnknownMarketError``."""
        try:
            return self.markets[ticker]
        except KeyError as exc:
            raise UnknownMarketError(ticker) from exc

    def get_order_book(self, ticker: str) -> OrderBookSnapshot:
        """Return the book at ``ticker``'s current replay cursor.

        Args:
            ticker: The market to look up.

        Returns:
            The order-book snapshot at the current step.

        Raises:
            UnknownMarketError: If ``ticker`` has no session.
        """
        return self._current_step(ticker).book

    def get_exchange_status(self) -> ExchangeStatus:
        """Return the fixture status, stamped with the observation instant.

        The ``status`` value comes verbatim from the fixture and is never
        synthesized, so a ``paused``/``closed`` fixture still reports as such.
        Only ``fetched_at`` is replaced: the committed fixtures carry a frozen
        literal far outside any freshness ttl, and this exchange answers
        synchronously in-process, so the observation genuinely happened now.
        Mirrors :meth:`KalshiConnector.get_exchange_status`, which likewise
        stamps its own clock rather than trusting the venue's timestamp.

        Returns:
            The fixture status observed at the injected clock's instant.
        """
        return replace(self.exchange_status, fetched_at=self._clock())

    def get_exchange_time(self) -> datetime:
        """Return the fixture exchange time."""
        return self.exchange_time

    def get_balance_semantics(self) -> BalanceSemantics:
        """Return the fixture balance semantics, every answer of which is implemented.

        The constructor refuses any fixture whose collateral or partial-fill
        answers this class does not actually behave like
        (:data:`_SEMANTICS_REQUIREMENTS`), so this record is a description of
        :meth:`get_balances` and :meth:`cancel_order`, not an aspiration about
        the venue a session was recorded from (issues #18, #362).

        Returns:
            The venue's balance-interpretation semantics.
        """
        return self.balance_semantics

    def get_balances(self) -> BalanceSnapshot:
        """Return the opening balance less what is spent and what is pledged.

        ``total`` is the opening total less every *executed* cash movement: each
        fill's book cost plus the fee schedule's charge on that fill. Buying a
        fully collateralized binary converts cash into contracts, and the
        contracts are reported separately by :meth:`get_positions`. Resting
        collateral is not taken out of ``total`` -- the money is pledged, not
        gone -- which is exactly the ``INCLUDED`` answer
        :meth:`get_balance_semantics` gives for
        ``open_order_collateral_in_total``.

        ``available`` is that same figure less every resting order's
        reservation, making the advertised ``DEDUCTED_FROM_AVAILABLE`` true
        rather than merely claimed (issue #362). Without the withholding a
        consumer reading ``available`` as spendable could authorize a second
        order against cash already committed to a live one -- a wrong flag makes
        callers fail open, which is worse than the ``UNKNOWN`` that makes them
        fail closed.

        A reservation is the resting order's book cost *plus* the same
        worst-case fee bound the fill will be charged. Reserving the fee is not
        a fee debit -- ``total`` still moves only at execution, honoring
        ``fee_debit_timing == AT_EXECUTION`` -- it keeps the reservation an
        upper bound on the cash the order will consume, so a caller that spends
        all of ``available`` cannot be left a fee short. Understating a
        reservation is the direction that lets cash be double-committed, and it
        makes a resting fill land as a continuous conversion: what was withheld
        is precisely what gets spent, so ``available`` does not move when a
        resting order fills at its own limit.

        The slippage haircut
        (:data:`~windbreak.connector.fills.DEFAULT_FEE_HAIRCUT_PPM`) is
        deliberately *not* charged anywhere here -- it is a pessimism allowance
        the sizing path reasons with, not money the venue takes.

        Neither figure is floored at zero. A paper account that spent or pledged
        more than it opened with reports a negative balance, because clamping
        would hide exactly the simulator-fidelity bug the number exists to
        expose.

        Returns:
            The current balances, stamped at the observation instant.
        """
        spent = self._cash_spent_micros()
        return BalanceSnapshot(
            total=MoneyMicros(self.balances.total.value - spent),
            available=MoneyMicros(
                self.balances.available.value - spent - self._reserved_micros()
            ),
            fetched_at=self._clock(),
        )

    def _cash_spent_micros(self) -> int:
        """Return the cash this exchange's own fills have spent, in micros.

        The fee is re-derived per fill from the same
        :meth:`~windbreak.connector.fees.FeeModel.max_trading_fee_micros` call
        the taker walk charged per consumed level -- and a taker walk emits
        exactly one :class:`~windbreak.connector.models.Fill` per consumed level
        (the ``PER_FILL_RECORDS`` invariant the constructor enforces) -- so this
        reconstruction reproduces the walk's fee exactly rather than
        approximating it. A resting trade-through fill is charged on the same
        schedule.

        Returns:
            The total book cost plus fees across every simulated fill, in
            micros.
        """
        return sum(
            self._order_cash_micros(fill.ticker, fill.price, fill.quantity)
            for fill in self._fills
        )

    def _reserved_micros(self) -> int:
        """Return the cash this exchange's resting orders have pledged, in micros.

        Each resting order reserves what it would cost to fill entirely at its
        own limit -- the same book-cost-plus-fee arithmetic
        :meth:`_cash_spent_micros` applies to a completed fill, which is what
        makes filling a resting order balance-neutral for ``available``. A
        cancelled order leaves :attr:`_resting` immediately, so its reservation
        disappears from the next reading (``CancelCollateralRelease.IMMEDIATE``).

        Returns:
            The total collateral withheld for currently resting orders, in
            micros.
        """
        return sum(
            self._order_cash_micros(order.ticker, order.price, order.quantity)
            for order in self._resting
        )

    def _order_cash_micros(
        self, ticker: str, price: PricePips, quantity: ContractCentis
    ) -> int:
        """Return the cash ``quantity`` at ``price`` moves on ``ticker``, in micros.

        Both sides are priced in their own frame: a NO order's notional is its
        NO price times its size, matching how :meth:`get_positions` folds the
        fill log, so a NO reservation is never taken at the YES complement.

        Args:
            ticker: The market the cash moves on (selects the fee schedule).
            price: The trade or limit price, in that side's own pips.
            quantity: The size, in contract-centis; must be positive, which
                every emitted fill and every resting remainder is.

        Returns:
            The book cost plus the venue's worst-case trading fee, in micros.
        """
        book_cost = money_from_price_and_count(
            price, quantity, rounding=_MONEY_ROUNDING
        ).value
        fee = self.get_fee_model(ticker).max_trading_fee_micros(
            price.value, quantity.value
        )
        return book_cost + fee

    def get_positions(self) -> tuple[Position, ...]:
        """Return the holdings this exchange's own fills have accumulated.

        One row per ticker that has filled, ordered by ticker for determinism,
        in the YES frame (see :func:`_position_row`). A ticker with no fills has
        no row: the account is genuinely flat there.

        Returns:
            The per-ticker positions, in ticker order.

        Raises:
            TwoSidedPositionError: If any one ticker has filled on both the YES
                and the NO side. See that class for why netting is refused.
        """
        positions: list[Position] = []
        seen: set[str] = set()
        for (ticker, side), totals in sorted(
            self._fill_totals().items(), key=lambda item: item[0]
        ):
            if ticker in seen:
                raise TwoSidedPositionError(ticker)
            seen.add(ticker)
            positions.append(_position_row(ticker, side, totals))
        return tuple(positions)

    def _fill_totals(self) -> dict[tuple[str, Literal["yes", "no"]], _SideTotals]:
        """Fold this exchange's fill log into per-``(ticker, side)`` totals.

        Returns:
            The filled size and notional for every ``(ticker, side)`` slice that
            has traded; a slice only exists once a fill landed on it, and every
            emitted fill carries a positive quantity, so no entry is ever empty.
        """
        totals: dict[tuple[str, Literal["yes", "no"]], _SideTotals] = {}
        for fill in self._fills:
            entry = totals.setdefault((fill.ticker, fill.side), _SideTotals())
            entry.quantity_centis += fill.quantity.value
            entry.notional_micros += money_from_price_and_count(
                fill.price, fill.quantity, rounding=_MONEY_ROUNDING
            ).value
        return totals

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Return the account's currently resting orders."""
        return tuple(self._resting)

    def get_fills(self, since: datetime) -> tuple[Fill, ...]:
        """Return simulated fills executed strictly after ``since``.

        Args:
            since: The exclusive lower bound; only fills with ``ts > since``
                are returned.

        Returns:
            The matching simulated fills, in emission order.
        """
        return tuple(fill for fill in self._fills if fill.ts > since)

    def get_fee_model(self, market_or_series: str) -> FeeModel:
        """Return the fee schedule for a ticker, falling back to ``default``.

        Args:
            market_or_series: The market ticker or series key to look up.

        Returns:
            The ticker-specific fee model when present, else the default one.
        """
        return self.fee_models.get(market_or_series, self.fee_models[_DEFAULT_FEE_KEY])

    def advance(self) -> bool:
        """Consume every session's current step, then step each cursor forward.

        Each ticker's current step is processed against its resting orders (a
        recorded trade-through fills, appending one :class:`Fill` per resting
        order that fills), and its cursor advances by one.

        Returns:
            True if any session still has an unconsumed step after advancing;
            False once every session's last step has been consumed.
        """
        any_next = False
        for ticker, steps in self.sessions.items():
            index = self._cursor[ticker]
            if index >= len(steps):
                continue
            self._process_resting_fills(ticker, steps[index])
            self._cursor[ticker] = index + 1
            any_next = any_next or index + 1 < len(steps)
        return any_next

    def _process_resting_fills(self, ticker: str, step: _SessionStep) -> None:
        """Fill this ticker's resting orders against ``step``'s trade prints.

        Resting orders on the same book side share one trade-through volume pool
        (:func:`~windbreak.connector.fills.allocate_resting_fills`): several
        orders can never jointly claim more volume than the step actually
        recorded, and the most aggressive limit is filled first. Orders on other
        tickers are carried over untouched.

        Args:
            ticker: The market whose resting orders this step fills.
            step: The replay step supplying the book depth and trade prints.
        """
        filled = self._allocate_resting_fills(ticker, step)
        survivors: list[OpenOrder] = []
        for order in self._resting:
            if order.ticker != ticker:
                survivors.append(order)
                continue
            survivor = self._apply_resting_fill(order, step, filled[order.id])
            if survivor is not None:
                survivors.append(survivor)
        self._resting = survivors

    def _allocate_resting_fills(
        self, ticker: str, step: _SessionStep
    ) -> dict[str, ContractCentis]:
        """Allocate ``step``'s trade volume across ``ticker``'s resting orders.

        Orders are grouped by side -- each side has its own buy-frame prints and
        shares one volume pool -- and allocated in price priority via
        :func:`~windbreak.connector.fills.allocate_resting_fills`.

        Args:
            ticker: The market whose resting orders are being filled.
            step: The replay step supplying the book depth and trade prints.

        Returns:
            A mapping from resting-order id to its realized fill this step.
        """
        filled: dict[str, ContractCentis] = {}
        for side in ("yes", "no"):
            side_orders = [
                order
                for order in self._resting
                if order.ticker == ticker and order.side == side
            ]
            if not side_orders:
                continue
            projected = [
                self._resting_fill_inputs(order, step) for order in side_orders
            ]
            requests = tuple(
                RestingFillRequest(limit, order.quantity, depth)
                for order, (limit, _, depth) in zip(side_orders, projected, strict=True)
            )
            allocated = allocate_resting_fills(
                requests,
                projected[0][1],
                max_participation_ppm=self.max_participation_ppm,
            )
            for order, fill in zip(side_orders, allocated, strict=True):
                filled[order.id] = fill
        return filled

    def _apply_resting_fill(
        self, order: OpenOrder, step: _SessionStep, filled: ContractCentis
    ) -> OpenOrder | None:
        """Emit ``order``'s allocated resting fill and return its survivor.

        Args:
            order: The resting order being filled.
            step: The replay step (supplies the triggering trade-through time).
            filled: The volume allocated to this order this step.

        Returns:
            The order with its remaining quantity reduced, or None if it filled
            completely and should be dropped.
        """
        if filled.value <= 0:
            return order
        limit, prints, _ = self._resting_fill_inputs(order, step)
        # A resting fill occurs when the market trades *through* the limit, so it
        # is stamped at the triggering print's time -- not the (earlier) book
        # snapshot -- keeping Fill.ts honest for get_fills() since-filtering.
        fill_ts = trade_through_fill_ts(limit, prints)
        self._emit_fill(order.ticker, order.side, order.price, filled, fill_ts)
        remaining = order.quantity.value - filled.value
        if remaining <= 0:
            return None
        return replace(order, quantity=ContractCentis(remaining))

    def _resting_fill_inputs(
        self, order: OpenOrder, step: _SessionStep
    ) -> tuple[PricePips, tuple[TradePrint, ...], ContractCentis]:
        """Project a resting order and step into buy-frame fill inputs.

        :func:`resting_fill_quantity` is written for a resting *buy* (a print
        strictly below the limit is a trade-through). A YES bid is already in
        that frame; a NO bid is complemented (``10_000 - price``) so the same
        buy-frame logic applies symmetrically.

        Args:
            order: The resting order to project.
            step: The replay step supplying the book and trade prints.

        Returns:
            The buy-frame limit, trade prints, and at-or-better depth.
        """
        if order.side == "yes":
            return self._yes_resting_inputs(order, step)
        return self._no_resting_inputs(order, step)

    def _yes_resting_inputs(
        self, order: OpenOrder, step: _SessionStep
    ) -> tuple[PricePips, tuple[TradePrint, ...], ContractCentis]:
        """Return buy-frame fill inputs for a resting YES bid (already in frame)."""
        depth = sum(
            level.quantity.value
            for level in step.book.yes_bids
            if level.price >= order.price
        )
        return order.price, step.trades, ContractCentis(depth)

    def _no_resting_inputs(
        self, order: OpenOrder, step: _SessionStep
    ) -> tuple[PricePips, tuple[TradePrint, ...], ContractCentis]:
        """Return buy-frame fill inputs for a resting NO bid (complemented)."""
        prints = tuple(
            replace(trade, price=PricePips(COMPLEMENT_PIPS - trade.price.value))
            for trade in step.trades
        )
        depth = sum(
            level.quantity.value
            for level in step.book.yes_asks
            if COMPLEMENT_PIPS - level.price.value >= order.price.value
        )
        return order.price, prints, ContractCentis(depth)

    def place_order(
        self, normalized_intent: object, approval_token: object
    ) -> PaperPlacement:
        """Place a :class:`PaperOrderIntent`, filling any crossing portion.

        The crossing portion fills immediately via the pessimistic taker walk
        (one :class:`Fill` per consumed level); any remainder rests at the
        order's own limit. The approval token is accepted and ignored -- paper
        trading has no real risk-kernel gate.

        Args:
            normalized_intent: The order intent; must be a :class:`PaperOrderIntent`.
            approval_token: Accepted and ignored.

        Returns:
            A :class:`PaperPlacement` receipt of the fills and any resting order.

        Raises:
            TypeError: If ``normalized_intent`` is not a :class:`PaperOrderIntent`.
            UnknownMarketError: If the intent's ticker has no session.
        """
        del approval_token
        if not isinstance(normalized_intent, PaperOrderIntent):
            raise TypeError(
                "normalized_intent must be a PaperOrderIntent, "
                f"got {type(normalized_intent).__name__}"
            )
        step = self._current_step(normalized_intent.ticker)
        result = walk_taker_fill(
            self._taker_levels(normalized_intent, step.book),
            normalized_intent.price,
            normalized_intent.quantity,
            self.get_fee_model(normalized_intent.ticker),
            haircut_ppm=self.haircut_ppm,
            max_participation_ppm=self.max_participation_ppm,
        )
        fills = self._emit_taker_fills(normalized_intent, result.consumed, step.book)
        remainder = normalized_intent.quantity.value - result.filled.value
        resting = self._rest_remainder(normalized_intent, remainder)
        return PaperPlacement(fills=fills, resting_order=resting)

    def _taker_levels(
        self, intent: PaperOrderIntent, book: OrderBookSnapshot
    ) -> tuple[OrderBookLevel, ...]:
        """Return the book side a crossing ``intent`` walks, best-first.

        A YES order crosses the recorded YES asks directly; a NO order crosses
        the complement of the YES bids (``10_000 - price``), which a best-first
        (descending) bid book turns into a best-first (ascending) NO-ask book.

        Args:
            intent: The order being placed.
            book: The current book snapshot.

        Returns:
            The eligible book side for the taker walk, best-first.
        """
        if intent.side == "yes":
            return book.yes_asks
        return tuple(
            OrderBookLevel(
                PricePips(COMPLEMENT_PIPS - level.price.value), level.quantity
            )
            for level in book.yes_bids
        )

    def _emit_taker_fills(
        self,
        intent: PaperOrderIntent,
        consumed: Sequence[OrderBookLevel],
        book: OrderBookSnapshot,
    ) -> tuple[Fill, ...]:
        """Emit one :class:`Fill` per consumed level of a taker walk.

        A taker order executes immediately, so each fill is stamped at the
        current book snapshot time.
        """
        fills: list[Fill] = []
        for level in consumed:
            fills.append(
                self._emit_fill(
                    intent.ticker,
                    intent.side,
                    level.price,
                    level.quantity,
                    book.fetched_at,
                )
            )
        return tuple(fills)

    def _rest_remainder(
        self, intent: PaperOrderIntent, remainder: int
    ) -> OpenOrder | None:
        """Rest an order for the unfilled ``remainder`` at the intent's limit."""
        if remainder <= 0:
            return None
        self._order_seq += 1
        order = OpenOrder(
            id=f"paper-order-{self._order_seq}",
            ticker=intent.ticker,
            side=intent.side,
            price=intent.price,
            quantity=ContractCentis(remainder),
        )
        self._resting.append(order)
        return order

    def _emit_fill(
        self,
        ticker: str,
        side: Literal["yes", "no"],
        price: PricePips,
        quantity: ContractCentis,
        ts: datetime,
    ) -> Fill:
        """Record and return one simulated fill stamped at ``ts``.

        Args:
            ticker: The market the fill is on.
            side: The filled side (``yes`` or ``no``).
            price: The fill price, in pips.
            quantity: The filled size, in contract-centis.
            ts: When the fill occurred -- the current book time for a taker fill
                (it executes now, against now's book), or the triggering trade
                print's time for a resting trade-through fill.

        Returns:
            The recorded :class:`Fill`.
        """
        self._fill_seq += 1
        fill = Fill(
            id=f"paper-fill-{self._fill_seq}",
            ticker=ticker,
            side=side,
            price=price,
            quantity=quantity,
            ts=ts,
        )
        self._fills.append(fill)
        return fill

    def cancel_order(self, order_id: str) -> None:
        """Remove the resting order with ``order_id`` (a no-op if absent).

        The order leaves the resting book here and nowhere else, so the next
        :meth:`get_balances` no longer withholds its collateral -- the
        ``CancelCollateralRelease.IMMEDIATE`` answer
        :meth:`get_balance_semantics` advertises (issue #362).

        Args:
            order_id: The identifier of the resting order to cancel.
        """
        self._resting = [order for order in self._resting if order.id != order_id]
