"""Read-only exchange verification for the Risk Kernel (SPEC S5.2 / S10.3).

Each cycle the kernel cross-checks a :class:`ReadOnlyVenueView`'s
exchange-verified balances, positions, and open orders against
ledger-derived :class:`LedgerExpectations`, classifies the result as
:class:`VerificationOutcome` ``CLEAN`` / ``DRIFT_WITHIN_TOLERANCE`` / ``BREACH``,
fires an operator alert for any held market whose jurisdiction is unknown, fires
a reconciliation-mismatch alert on a breach, and records exactly one bare
:class:`~windbreak.ledger.events.Event` per cycle. The resulting
:class:`VerificationSnapshot` feeds both the kernel's HALT-on-breach path and
the floor computation (its observed cash and drift rewrite the account's
verified-cash and reconciliation-buffer terms).

Balance and position drift are tolerance-graded: a nonzero diff within tolerance
is drift (still ``ok``), a diff beyond tolerance is a breach. Open orders are
discrete, so any set difference at all is a breach. A connector that raises
while being observed is itself graded a breach and recorded on the same audited
path (:meth:`ReadOnlyVerifier._unobservable_venue_breach`), never allowed to
escape the cycle -- an escaping exception would skip the record *and* the HALT
that snapshot drives. Every value on this path is
a :mod:`windbreak.numeric` scaled integer -- never a float (SPEC S6.1, enforced
by ``scripts/lint_no_floats.py``) -- and every recorded payload leaf is an
``int``, ``str``, or ``bool``.

The venue seam is typed as
:class:`~windbreak.connector.readonly.ReadOnlyVenueView`, not as the full
:class:`~windbreak.connector.interface.MarketConnector`, so SPEC S1.1
invariant 3 -- the verification path never holds trade-scope credentials -- is
enforced by the type rather than by convention: the five methods this module
calls are the only ones the seam exposes, and ``place_order``/``cancel_order``
are simply not reachable from here. Any ``MarketConnector`` still satisfies the
narrower protocol structurally, so a caller may pass one; a caller that wants
the guarantee passes a
:class:`~windbreak.connector.readonly.ReadOnlyConnectorView` instead.

:class:`KernelLedgerWriter` and :class:`ReadOnlyVenueView` are imported under
``TYPE_CHECKING`` only: both are structural protocols, so importing them for
typing alone keeps this module free of a runtime ``verification`` <-> ``process``
import cycle.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeGuard

from windbreak.alerts.registry import AlertType
from windbreak.ledger.events import Event
from windbreak.numeric.rounding import RoundingDirection, divide
from windbreak.numeric.types import ContractCentis, MoneyMicros

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from windbreak.alerts.dispatch import AlertDispatcher
    from windbreak.connector.models import OpenOrder, Position
    from windbreak.connector.readonly import ReadOnlyVenueView
    from windbreak.riskkernel.process import KernelLedgerWriter

#: Component label stamped on every verification event this module records --
#: and, because :func:`_own_component_events` filters the seeding path by it,
#: also this module's *seeding trust boundary*: the label written on the way out
#: is the label required on the way back in, so the baseline folds only facts
#: this kernel itself recorded.
_COMPONENT = "riskkernel"

#: Payload schema version stamped on every verification event.
_PAYLOAD_SCHEMA_VERSION = 1

#: The :class:`~windbreak.connector.models.NormalizedMarket.jurisdiction_status`
#: value that must fire a warning alert for any held market carrying it.
_UNKNOWN_JURISDICTION = "unknown"

#: The reconciliation-mismatch alert body dispatched on a breach.
_MISMATCH_MESSAGE = "exchange verification mismatch beyond tolerance"

#: The reconciliation-mismatch alert body dispatched when the venue could not be
#: observed at all. The raising call's message is appended, so the operator sees
#: *why* (e.g. a two-sided holding the connector refuses to net into one row).
_UNREADABLE_MESSAGE = "exchange verification could not observe the venue"


@dataclass(frozen=True, slots=True)
class LedgerExpectations:
    """The ledger-derived state one verification cycle checks the venue against.

    Attributes:
        expected_available_cash: The available cash the ledger expects the
            venue to report, in micros.
        expected_positions: The per-ticker net position the ledger expects, in
            contract-centis; a ticker absent from the mapping is expected flat.
        expected_open_order_ids: The exact set of resting-order ids the ledger
            expects the venue to report.
    """

    expected_available_cash: MoneyMicros
    expected_positions: Mapping[str, ContractCentis]
    expected_open_order_ids: frozenset[str]


class ExpectationSource(Protocol):
    """The seam a verifier reads its :class:`LedgerExpectations` from."""

    def get_expectations(self) -> LedgerExpectations:
        """Return the current ledger-derived expectations.

        Returns:
            The :class:`LedgerExpectations` to verify the venue against.
        """
        ...


@dataclass(frozen=True, slots=True)
class VerificationTolerances:
    """The admissible drift before a dimension is treated as a breach.

    Open orders are discrete and carry no tolerance: any set difference is a
    breach.

    Attributes:
        balance_tolerance: The maximum admissible available-cash drift, in
            micros (inclusive: a diff equal to it is still within tolerance).
        position_tolerance: The maximum admissible per-ticker position drift,
            in contract-centis (inclusive).
    """

    balance_tolerance: MoneyMicros
    position_tolerance: ContractCentis


class VerificationOutcome(enum.Enum):
    """The graded outcome of one verification cycle.

    Attributes:
        CLEAN: Every dimension matched its expectation exactly.
        DRIFT_WITHIN_TOLERANCE: A balance or position dimension drifted, but
            every drift stayed within its tolerance; not a breach.
        BREACH: At least one dimension drifted beyond tolerance, or the
            open-order sets differed at all.
    """

    CLEAN = enum.auto()
    DRIFT_WITHIN_TOLERANCE = enum.auto()
    BREACH = enum.auto()


@dataclass(frozen=True, slots=True)
class VerificationSnapshot:
    """The immutable result of one verification cycle.

    Attributes:
        outcome: The graded :class:`VerificationOutcome` of the cycle.
        balance_ok: Whether the available-cash drift was within tolerance.
        position_ok: Whether every per-ticker position drift was within
            tolerance.
        open_order_ok: Whether the observed and expected open-order id sets
            matched exactly.
        verified_at_epoch_s: The epoch second the cycle ran at.
        exchange_verified_available_cash: The venue-reported available cash the
            cycle observed, in micros.
        cash_drift: The absolute available-cash drift the cycle observed, in
            micros.
        semantics_fully_known: Whether every balance-semantics field the venue
            reported is a known (non-``UNKNOWN``) value.
    """

    outcome: VerificationOutcome
    balance_ok: bool
    position_ok: bool
    open_order_ok: bool
    verified_at_epoch_s: int
    exchange_verified_available_cash: MoneyMicros
    cash_drift: MoneyMicros
    semantics_fully_known: bool


#: Maps each outcome to the bare-``Event`` ``event_type`` recorded for it.
_EVENT_TYPE_BY_OUTCOME: dict[VerificationOutcome, str] = {
    VerificationOutcome.CLEAN: "VerificationPassed",
    VerificationOutcome.DRIFT_WITHIN_TOLERANCE: "VerificationDrift",
    VerificationOutcome.BREACH: "VerificationMismatch",
}

#: The verification ``event_type`` values whose recorded cash may seed the cash
#: baseline: a clean pass or a within-tolerance drift. ``"VerificationMismatch"``
#: is deliberately excluded, so a restart never re-baselines onto a cash figure
#: already graded a breach.
_CASH_SEED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        _EVENT_TYPE_BY_OUTCOME[VerificationOutcome.CLEAN],
        _EVENT_TYPE_BY_OUTCOME[VerificationOutcome.DRIFT_WITHIN_TOLERANCE],
    }
)

#: The ledger ``event_type`` whose latest rows seed the position baseline.
_POSITIONS_SNAPSHOT_EVENT_TYPE = "PositionsSnapshotRecorded"

#: The verification-event payload key the cash seed reads its micros off of.
_CASH_PAYLOAD_KEY = "exchange_verified_available_cash"

#: The ``PositionsSnapshotRecorded`` payload key holding its list of position
#: rows (see :class:`~windbreak.ledger.events.PositionsSnapshotRecorded`).
_POSITIONS_PAYLOAD_KEY = "positions"

#: The position-row keys the position seed reads: the market ticker and the
#: signed quantity in contract-centis.
_ROW_TICKER_KEY = "ticker"
_ROW_QUANTITY_CENTIS_KEY = "quantity_centis"

#: The ledger ``event_type`` carrying one fill's booked accounting entry (see
#: :class:`~windbreak.ledger.events.FillAccounted`). Anything else arriving on
#: the fill-accounting seam is treated as an unreconstructable entry.
_FILL_ACCOUNTED_EVENT_TYPE = "FillAccounted"

#: The ``FillAccounted`` payload keys the advance reads.
_FILL_CASH_DELTA_KEY = "cash_delta_micros"
_FILL_POSITION_DELTA_KEY = "position_delta_centis"

#: The ledger ``event_type`` booking one order's arrival on the venue's resting
#: book (see :class:`~windbreak.ledger.events.RestingOrderAccounted`). Together
#: with :data:`_FILL_ACCOUNTED_EVENT_TYPE` these are the only two types the
#: fill-accounting seam may deliver; anything else is an unreconstructable entry.
_RESTING_ORDER_ACCOUNTED_EVENT_TYPE = "RestingOrderAccounted"

#: The payload key naming the venue order an entry is about. It appears on a
#: ``RestingOrderAccounted`` (the order that came to rest) and, since issue #390,
#: on a ``FillAccounted`` (the resting order the execution consumed, or ``""``
#: for an outright taker walk that rested nothing).
_VENUE_ORDER_ID_KEY = "venue_order_id"

#: The ``RestingOrderAccounted`` payload key carrying the quantity that rested.
_RESTING_QUANTITY_KEY = "resting_quantity_centis"

#: The ``RestingOrderAccounted`` payload key carrying the cash the venue
#: withheld from ``available`` when the order came to rest (issue #423). Absent
#: on a schema-v1 row booked before reservations were recorded at all, which
#: reads as :data:`_NO_RESERVATION` -- the fail-closed direction, since an
#: uncredited reservation leaves the venue looking short and breaches.
_RESERVED_COLLATERAL_KEY = "reserved_collateral_micros"

#: The reservation attributed to an arrival that booked none, and to every
#: resting order captured from the connector at startup (whose collateral the
#: venue had already withheld from the cash baseline itself).
_NO_RESERVATION = 0


def _own_component_events(events: tuple[Event, ...]) -> Iterator[Event]:
    """Return an iterator over only the events this kernel itself recorded.

    This is the *seeding trust boundary*: the kernel seeds its safety-critical
    reconciliation baseline only from facts it recorded itself. ``--ledger-path``
    is a shared flag, and the documented compose topology mounts one ``ledger``
    volume across every process (``deploy/docker-compose.yml``), so one ledger
    routinely carries other components' events -- notably the PAPER loop's
    ``component="scheduler"`` snapshots, which are derived from a *simulated*
    ``PaperExchange`` (``windbreak/scheduler/loop.py``). Folding a foreign
    event into this kernel's baseline would describe an account it has never
    held, so every event-type match on the seeding path is scoped through here.

    Args:
        events: The startup event history, oldest first.

    Returns:
        A lazy iterator over the events stamped with :data:`_COMPONENT`, in
        history order.
    """
    return (event for event in events if event.component == _COMPONENT)


def _is_scaled_int(value: object) -> TypeGuard[int]:
    """Return whether ``value`` may be narrowed onto the scaled-int path.

    ``bool`` is a subclass of ``int``, so a bare ``isinstance(value, int)``
    admits ``True``/``False`` -- and the scaled-int constructors reject exactly
    that: :class:`~windbreak.numeric.types.MoneyMicros` and
    :class:`~windbreak.numeric.types.ContractCentis` raise ``TypeError`` on a
    ``bool``. Without this exclusion a single malformed ledger row does not take
    the defensive skip path it was meant to; it slips through the narrowing into
    a constructor that raises, turning a corrupt payload leaf into an unhandled
    crash while the kernel is projecting its startup baseline. Excluding ``bool``
    here keeps such a row on the skip path, where a malformed fact is simply not
    a fact.

    Args:
        value: A JSON-safe payload leaf, of unknown runtime type.

    Returns:
        Whether ``value`` is an ``int`` that is not a ``bool``.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _seed_cash(events: tuple[Event, ...], connector: ReadOnlyVenueView) -> MoneyMicros:
    """Return the cash baseline seeded from history, else from the connector.

    Scans only the events *this component recorded*
    (:func:`_own_component_events`) for the *last* clean-pass or
    within-tolerance-drift verification event and takes its recorded
    ``exchange_verified_available_cash`` (a breach event is ignored). Falls back
    to the connector's currently reported available cash when that filtered
    history carries no such event.

    The component scoping is latent hardening here, not an active fix: today
    :class:`ReadOnlyVerifier` is the only emitter of ``"VerificationPassed"`` /
    ``"VerificationDrift"`` / ``"VerificationMismatch"``, so no foreign row can
    currently reach this fold and the filter changes no observed behavior. It
    is applied anyway as defense in depth -- the cash baseline must never
    silently begin trusting a future foreign emitter of a same-named event.

    Args:
        events: The startup event history, oldest first.
        connector: The read-only connector supplying the fallback balance.

    Returns:
        The cash baseline, in micros.
    """
    seed: MoneyMicros | None = None
    for event in _own_component_events(events):
        if event.event_type in _CASH_SEED_EVENT_TYPES:
            cash = event.payload.get(_CASH_PAYLOAD_KEY)
            if _is_scaled_int(cash):
                seed = MoneyMicros(cash)
    if seed is not None:
        return seed
    return connector.get_balances().available


def _seed_positions(
    events: tuple[Event, ...], connector: ReadOnlyVenueView
) -> dict[str, ContractCentis]:
    """Return the position baseline seeded from history, else from the connector.

    Uses the rows of the *last* ``PositionsSnapshotRecorded`` event *this
    component recorded* (:func:`_own_component_events`), which override any
    earlier own snapshot's ticker set wholesale, mapped to ``ContractCentis``.
    Falls back to the connector's currently reported positions when that
    filtered history carries no snapshot at all -- so a *foreign* snapshot
    leaves the seed unset and the connector fallback fires, while an *own*
    snapshot recording a flat account (an empty row list) still wins over the
    connector.

    The component filter is the load-bearing one. ``windbreak/scheduler/loop.py``
    records a ``component="scheduler"`` ``PositionsSnapshotRecorded`` whose rows
    are derived from a *simulated* ``PaperExchange``, and the documented topology
    puts those rows in the very ledger a live kernel replays: one named
    ``ledger`` volume is mounted across every process
    (``deploy/docker-compose.yml``) and
    :meth:`~windbreak.ledger.store.SqliteLedgerStore.read_all` returns the whole
    chain unscoped -- which is why every row carries a ``component`` column at
    all (SPEC v3 S12). Unfiltered, a live kernel could seed its safety-critical
    reconciliation baseline from simulated positions, and then either auto-kill
    itself spuriously through ``AUTO_RECONCILIATION`` or mask real drift behind a
    coincidentally agreeable simulation.

    What that means today, stated plainly: *no* component records a
    ``riskkernel``-stamped ``PositionsSnapshotRecorded``, so the position
    dimension always falls back to the connector -- this is not a live ledger
    projection of positions and must not be read as one. The ledger-seed path is
    retained deliberately as the forward-compatible hook for when the kernel does
    record its own snapshots; until then the connector fallback is the
    fail-closed default.

    Args:
        events: The startup event history, oldest first.
        connector: The read-only connector supplying the fallback positions.

    Returns:
        The expected per-ticker position, in contract-centis.
    """
    rows: object | None = None
    for event in _own_component_events(events):
        if event.event_type == _POSITIONS_SNAPSHOT_EVENT_TYPE:
            rows = event.payload.get(_POSITIONS_PAYLOAD_KEY)
    if rows is None:
        return {
            position.ticker: position.quantity for position in connector.get_positions()
        }
    return _rows_to_positions(rows)


def _rows_to_positions(rows: object) -> dict[str, ContractCentis]:
    """Map ``PositionsSnapshotRecorded`` payload rows to a position mapping.

    Each well-formed row contributes ``{ticker: ContractCentis(quantity_centis)}``;
    the JSON-safe ``object`` payload leaves are narrowed defensively so a
    malformed row can never smuggle a non-int quantity onto the scaled-int path.
    Malformed rows are skipped individually, so one bad row never discards the
    good rows beside it -- and ``rows`` that is not a list at all yields an empty
    mapping rather than a connector fallback, because the snapshot *exists*: the
    kernel then expects a flat account and breaches loudly against any real
    holding, instead of quietly adopting a garbage baseline.

    Quantities are narrowed through :func:`_is_scaled_int`, not a bare
    ``isinstance(quantity, int)``, so a ``bool`` is skipped like any other
    malformed leaf instead of reaching ``ContractCentis`` and raising.

    Args:
        rows: The snapshot's ``positions`` payload value.

    Returns:
        The per-ticker position mapping, in contract-centis.
    """
    positions: dict[str, ContractCentis] = {}
    if not isinstance(rows, list):
        return positions
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = row.get(_ROW_TICKER_KEY)
        quantity = row.get(_ROW_QUANTITY_CENTIS_KEY)
        if isinstance(ticker, str) and _is_scaled_int(quantity):
            positions[ticker] = ContractCentis(quantity)
    return positions


@dataclass(slots=True)
class _RestingOrder:
    """One resting order the expectation is following, and its reservation.

    Mutable on purpose: an order is whittled away by several fills across
    several cycles, and both the quantity left and the collateral released so
    far have to survive between them.

    Attributes:
        total_centis: The quantity the order rested with, in contract-centis.
            The pro-rata denominator, so it never changes once set.
        consumed_centis: How much of ``total_centis`` booked fills have consumed
            so far, clamped to ``total_centis``: a fill larger than the order it
            names retires it and no more, so the surplus is never absorbed here.
        reserved_micros: The cash the venue withheld from ``available`` when the
            order came to rest, per the booked arrival. :data:`_NO_RESERVATION`
            for a venue that withholds nothing, and for an order captured from
            the connector at startup -- whose collateral the venue had already
            withheld from the cash baseline this source seeded itself with.
        released_micros: How much of ``reserved_micros`` has been released back
            into the cash expectation so far. Tracked as a running total, not
            per fill, so the release can be computed from *cumulative*
            consumption and can never drop a rounding remainder.
    """

    total_centis: int
    consumed_centis: int
    reserved_micros: int
    released_micros: int


def _seed_open_orders(
    events: tuple[Event, ...], connector: ReadOnlyVenueView
) -> dict[str, _RestingOrder]:
    """Return the resting-order baseline seeded from history, else the connector.

    Empty *only* while the history ends KILLED and unrearmed: a kill cancels
    every resting order (recording a ``CancelAllDirective`` alongside its
    ``KillEngaged``), so while that kill has no matching later ``KillReArmed``
    nothing is expected to remain resting -- the expectation is empty regardless
    of what the connector still reports. The killed-vs-rearmed decision reuses
    the kernel's one canonical, fail-closed kill fold
    (:func:`~windbreak.riskkernel.kill.kill_state_in`, whose ``.killed`` is the
    exact durable fact that drives the kernel to ``KILLED`` on replay), so the
    open-order expectation and the replayed mode can never disagree.

    Once that kill is re-armed (or the history never killed), the expectation
    falls back to the connector's currently reported resting orders. This is
    load-bearing: a *past* kill -- including a routine kill/re-arm drill -- must
    never permanently zero the expectation, or every legitimately-resting order
    after the re-arm would be a false-positive breach for the life of the
    ledger (and could spuriously auto-kill a correctly-operating kernel via the
    reconciliation-mismatch monitor).

    This one startup read is the *only* place the venue's aggregate open-order
    view reaches the expectation, and that is deliberate. Issue #390 gave the
    open-order dimension a way to *advance* -- from ledgered arrival and fill
    entries -- precisely so that no later cycle ever needs to re-read this view.
    A baseline re-derived from the same connector the cycle then compares
    against would agree by construction and could never fail (issue #352).

    Each order's resting quantity is carried alongside its id, not discarded.
    The comparison itself is still an id-set match, but the advance needs the
    quantity: a booked fill names the order it consumed and says how much, and
    the id may only leave the expectation once those consumptions exhaust it.

    A startup capture carries :data:`_NO_RESERVATION`, and that is not a
    shortcut. The cash baseline this source seeds beside it is either the
    connector's ``available`` -- which the venue has *already* withheld these
    orders' collateral from -- or a prior cycle's recorded cash, which was
    graded against exactly that figure. Their reservations are therefore inside
    the baseline already, and releasing them again as the orders fill would
    credit the same cash twice. What the venue does release as a startup-captured
    order fills is consequently unexplained movement and still breaches, which
    is where this dimension already stood before issue #423; only *booked*
    arrivals carry a reservation the expectation may follow.

    The :func:`~windbreak.riskkernel.kill.kill_state_in` import is deferred to
    call time because :mod:`windbreak.riskkernel.kill` imports this module at
    runtime (for :class:`VerificationOutcome`); a module-level import here would
    close that cycle.

    Args:
        events: The startup event history, oldest first.
        connector: The read-only connector supplying the fallback orders.

    Returns:
        The resting-order bookkeeping, per venue order id.
    """
    from windbreak.riskkernel.kill import kill_state_in

    if kill_state_in(events).killed:
        return {}
    return {
        order.id: _RestingOrder(
            total_centis=order.quantity.value,
            consumed_centis=0,
            reserved_micros=_NO_RESERVATION,
            released_micros=0,
        )
        for order in connector.get_open_orders()
    }


class FillAccountingFeed(Protocol):
    """The seam a live expectation reads *this account's* booked entries through.

    The contract is deliberately narrow and stateful: :meth:`drain` returns the
    :class:`~windbreak.ledger.events.FillAccounted` and
    :class:`~windbreak.ledger.events.RestingOrderAccounted` entries booked for
    this account that it has not returned before, oldest first, and never
    returns them again. Chain order across the two kinds is part of the
    contract: a fill draws down the resting order it names, so an arrival
    delivered after its own fill would present as a fill naming an order the
    expectation has never held -- an unexplained gap that latches the advance
    off. Everything the kernel must not decide for itself -- which ledger the
    entries live in, which component is trusted to book them, and where the
    baseline's cursor starts -- is settled by the composition root that builds
    the feed, exactly as the SPEC S5 trust boundary requires. This module sees
    only ``int``/``str`` payload leaves and never a connector type.
    """

    def drain(self) -> tuple[Event, ...]:
        """Return the booked entries not yet returned, oldest first.

        Returns:
            The newly booked
            :class:`~windbreak.ledger.events.RestingOrderAccounted` and
            :class:`~windbreak.ledger.events.FillAccounted` events, interleaved
            in chain order; empty when nothing has been booked since the last
            call.
        """
        ...


@dataclass(frozen=True, slots=True)
class _FillEntry:
    """One booked fill's reconstructed accounting, ready to apply.

    Attributes:
        ticker: The market that traded.
        cash_delta: The signed available-cash movement, in micros.
        position_delta: The signed YES-frame position movement, in centis. Its
            magnitude is the fill's own size, which is what the named resting
            order is drawn down by.
        venue_order_id: The resting order this fill executed against, or ``""``
            when it retired none -- an outright taker walk, or an entry booked
            before issue #390 recorded ids at all.
    """

    ticker: str
    cash_delta: MoneyMicros
    position_delta: ContractCentis
    venue_order_id: str


@dataclass(frozen=True, slots=True)
class _RestingEntry:
    """One booked order arrival, ready to apply.

    Attributes:
        venue_order_id: The venue's identifier for the order now resting.
        quantity_centis: The quantity that came to rest, in contract-centis;
            always strictly positive.
        reserved_micros: The cash the venue withheld from ``available`` when the
            order came to rest, in micros; never negative, and
            :data:`_NO_RESERVATION` for a venue that withholds nothing.
    """

    venue_order_id: str
    quantity_centis: int
    reserved_micros: int


def _resting_entry(event: Event) -> _RestingEntry | None:
    """Reconstruct one booked order arrival, or ``None`` if it cannot be.

    Same fail-closed narrowing as :func:`_fill_entry`. A non-positive quantity
    is rejected as malformed rather than treated as a retirement: an order that
    rested nothing never rested, and retirement is expressed only by a fill
    naming the order. Admitting it here would let a zero-sized arrival silently
    stand in for a retirement nobody executed.

    ``reserved_collateral_micros`` is the one leaf allowed to be absent, exactly
    as ``venue_order_id`` is on a fill: a ``RestingOrderAccounted`` booked before
    issue #423 (payload schema v1) carries no such key and said nothing about
    collateral at all. It defaults to :data:`_NO_RESERVATION`, which is the
    fail-closed reading -- an uncredited reservation leaves the venue looking
    short and breaches, whereas an invented one would absorb real divergence. A
    key that is *present* but not a scaled int, or one that is negative, is a
    gap: an order coming to rest can withhold cash or withhold nothing, never
    hand cash back.

    Args:
        event: The event drained from the fill-accounting seam, already known to
            carry the resting-order ``event_type``.

    Returns:
        The reconstructed :class:`_RestingEntry`, or ``None`` when this entry
        cannot be accounted for.
    """
    order_id = event.payload.get(_VENUE_ORDER_ID_KEY)
    quantity = event.payload.get(_RESTING_QUANTITY_KEY)
    reserved = event.payload.get(_RESERVED_COLLATERAL_KEY, _NO_RESERVATION)
    if not (isinstance(order_id, str) and order_id):
        return None
    if not (_is_scaled_int(quantity) and quantity > 0):
        return None
    if not (_is_scaled_int(reserved) and reserved >= _NO_RESERVATION):
        return None
    return _RestingEntry(
        venue_order_id=order_id,
        quantity_centis=quantity,
        reserved_micros=reserved,
    )


def _fill_entry(event: Event) -> _FillEntry | None:
    """Reconstruct one booked fill's accounting, or ``None`` if it cannot be.

    Fail-closed narrowing, mirroring :func:`_rows_to_positions`: the payload
    leaves arrive as JSON-safe ``object`` and every one of them must be exactly
    the type the scaled-integer constructors accept before it may advance a
    safety-critical baseline. ``bool`` is excluded via :func:`_is_scaled_int`,
    so a malformed leaf takes this ``None`` path instead of reaching a
    constructor that raises.

    ``venue_order_id`` is the one leaf allowed to be absent: a ``FillAccounted``
    booked before issue #390 (payload schema v1) carries no such key, and such a
    row genuinely retired nothing it could name. It defaults to ``""``, which
    the advance reads as "this fill consumed no resting order". A key that is
    *present* but not a ``str`` is still a gap.

    Args:
        event: The event drained from the fill-accounting seam, already known to
            carry the fill-accounting ``event_type``.

    Returns:
        The reconstructed :class:`_FillEntry`, or ``None`` when this event
        cannot be accounted for.
    """
    ticker = event.payload.get(_ROW_TICKER_KEY)
    cash = event.payload.get(_FILL_CASH_DELTA_KEY)
    position = event.payload.get(_FILL_POSITION_DELTA_KEY)
    order_id = event.payload.get(_VENUE_ORDER_ID_KEY, "")
    if not (isinstance(ticker, str) and _is_scaled_int(cash)):
        return None
    if not (_is_scaled_int(position) and isinstance(order_id, str)):
        return None
    return _FillEntry(
        ticker=ticker,
        cash_delta=MoneyMicros(cash),
        position_delta=ContractCentis(position),
        venue_order_id=order_id,
    )


def _advanced(
    expectations: LedgerExpectations,
    entry: _FillEntry,
    open_orders: Mapping[str, _RestingOrder],
    released_micros: int,
) -> LedgerExpectations:
    """Return ``expectations`` advanced by one booked fill's deltas.

    Cash moves by *two* terms: the fill's own signed delta, and the collateral
    the named resting order releases as this fill draws it down (issue #423).
    Both are movements the venue really made -- it spent the cash and stopped
    withholding the reservation in the same execution -- so absorbing only one
    of them would leave the other as permanent, unexplained drift. Positions
    move by the entry's own delta; the open-order dimension is rebuilt from
    ``open_orders``, which the caller has already drawn down by whatever this
    fill consumed. Splitting it that way keeps the resting-order bookkeeping --
    which spans entries, since an order can be whittled away by several fills --
    in the one place that owns it.

    Args:
        expectations: The expectation to advance.
        entry: The booked fill's reconstructed accounting.
        open_orders: The resting-order bookkeeping, after this fill.
        released_micros: The collateral the named order released to this fill,
            in micros; ``0`` when the fill named no order, when the venue
            withheld nothing, or when the order was captured at startup.

    Returns:
        A new :class:`LedgerExpectations` with every dimension moved.
    """
    positions = dict(expectations.expected_positions)
    held = positions.get(entry.ticker, ContractCentis(0))
    positions[entry.ticker] = ContractCentis(held.value + entry.position_delta.value)
    return LedgerExpectations(
        expected_available_cash=MoneyMicros(
            expectations.expected_available_cash.value
            + entry.cash_delta.value
            + released_micros
        ),
        expected_positions=positions,
        expected_open_order_ids=frozenset(open_orders),
    )


def _with_arrival(
    expectations: LedgerExpectations,
    open_orders: Mapping[str, _RestingOrder],
    reserved_micros: int,
) -> LedgerExpectations:
    """Return ``expectations`` advanced by one booked order arrival.

    An arrival moves no position -- an order holds nothing until it fills -- but
    it does move cash on a venue whose
    :class:`~windbreak.connector.semantics.OrderCollateralInAvailable` is
    ``DEDUCTED_FROM_AVAILABLE``: the reservation leaves ``available`` the moment
    the order rests. Booking that withdrawal is the whole of issue #423.
    ``reserved_micros`` is ``0`` on a venue that withholds nothing, and then the
    cash dimension is left exactly where it was rather than having a reservation
    invented for it.

    Args:
        expectations: The expectation to advance.
        open_orders: The resting-order bookkeeping, including this arrival.
        reserved_micros: The cash the venue withheld for the arriving order.

    Returns:
        A new :class:`LedgerExpectations` carrying the updated id set and cash.
    """
    return LedgerExpectations(
        expected_available_cash=MoneyMicros(
            expectations.expected_available_cash.value - reserved_micros
        ),
        expected_positions=expectations.expected_positions,
        expected_open_order_ids=frozenset(open_orders),
    )


def _released_by(order: _RestingOrder, consumed_centis: int) -> int:
    """Return the collateral released by consuming up to ``consumed_centis``.

    The release is pro-rata on quantity, and it is computed from *cumulative*
    consumption rather than per fill, which is the property the whole design
    turns on. Money is in micros and quantity in contract-centis, so a per-fill
    ``reserved * piece // total`` drops a remainder on every uneven piece and
    the released total never reconciles to the reserved total. Taking the
    difference of two cumulative floors instead makes the remainders cancel:
    once ``consumed_centis`` reaches ``total_centis`` the running total is
    ``reserved * total // total``, which is ``reserved`` exactly -- no drift, no
    leftover micro, however uneven the pieces were.

    The quotient rounds toward negative infinity
    (:data:`~windbreak.numeric.rounding.RoundingDirection.UNDERSTATE_EQUITY`),
    so an intermediate release is understated rather than overstated: cash is
    handed back to the expectation late, never early. Every intermediate
    understatement is recovered by the final step, which is exact.

    Pro-rata is a statement about the *reservation the arrival booked*, not a
    re-reading of the venue. A venue that re-prices its remaining reservation
    per partial fill -- book-cost-plus-fee on the surviving quantity, with its
    own per-call fee rounding -- can therefore sit a micro or two away from this
    figure while an order is *part* consumed, though the two agree exactly once
    it is exhausted, since both have then released the whole reservation. That
    intermediate window is unreachable in the shipped PAPER loop, whose replay
    cursor is stationary (SPEC S7.5.1), so a resting order is never drawn down
    across cycles there; re-reading the venue to close it is refused anyway,
    because an expectation sourced from the view the cycle compares against
    could never fail (issue #352).

    Args:
        order: The resting order being drawn down, carrying its reservation and
            everything released from it so far.
        consumed_centis: The cumulative consumption after this fill, already
            clamped to ``order.total_centis``.

    Returns:
        The collateral this fill releases, in micros; ``0`` when the order
        carries no reservation at all.
    """
    if order.reserved_micros == _NO_RESERVATION:
        return 0
    released_to_date = divide(
        order.reserved_micros * consumed_centis,
        order.total_centis,
        rounding=RoundingDirection.UNDERSTATE_EQUITY,
    )
    return released_to_date - order.released_micros


class LedgerExpectationSource:
    """An :class:`ExpectationSource` projecting a *scoped* startup baseline.

    Built once at kernel startup from the replayed ledger ``history`` and the
    :class:`~windbreak.connector.readonly.ReadOnlyVenueView`, this source folds
    that history exactly once, at construction, into a baseline
    :class:`LedgerExpectations` stored on ``self._expectations``. **The
    connector is read exactly there and nowhere else**: a connector mutated
    after construction never changes the result. That is the guarantee the
    verifier relies on, and it is what keeps the comparison falsifiable rather
    than the issue #352 tautology of grading the venue against itself.

    The baseline is not frozen forever, though. When a ``fill_accounting`` feed
    is supplied it advances -- from *ledgered evidence only* (issues #365 and
    #390). The feed delivers two kinds of durable, hash-chained entry, and
    between them they cover all three dimensions:

    * a :class:`~windbreak.ledger.events.FillAccounted`, booked once at
      execution, carrying that fill's signed cash and position deltas and the
      venue order id it executed against (``""`` when it rested nothing); and
    * a :class:`~windbreak.ledger.events.RestingOrderAccounted`, booked from the
      venue's placement receipt, saying that an order came to rest, with what
      size, and how much collateral the venue withheld from ``available`` for
      it.

    Each :meth:`get_expectations` call folds whatever the feed has newly drained
    onto the running expectation. Without the fill entries the baseline stayed at
    the account *as it stood before this process traded*, so the first real fill
    moved the venue off it permanently: the next cycle graded ``BREACH``, the
    kernel HALTed per issue #32, and only a restart cleared it -- an always-on
    PAPER deployment could not survive its own first fill. Without the arrival
    entries the same was true of the open-order dimension the moment the loop
    routed anything other than an outright marketable order: the remainder left
    resting was an order the ledger had never heard of.

    The advance is not a relaxation, because a booked entry and an observation
    remain independent. The entry is frozen at execution and describes one
    discrete movement; the observation is the venue's live *aggregate* balance,
    position, and resting book. A venue that moved by something other than what
    was booked -- an unbooked fill, a fee charged differently, a settlement, a
    phantom position, an order that rested unbidden, a resting order that
    vanished with no fill behind it -- still diverges and still halts. Only the
    *explained* part of the movement is absorbed.

    A resting order also withholds collateral from ``available`` on a venue
    whose :class:`~windbreak.connector.semantics.OrderCollateralInAvailable` is
    ``DEDUCTED_FROM_AVAILABLE``, and issue #423 gave that reservation a booked
    fact too: it rides on the arrival entry, as the venue's own figure, and is
    released back as booked fills draw the order down. The release is pro-rata
    on quantity and taken from *cumulative* consumption
    (:func:`_released_by`), so however unevenly an order fills the released
    total is the reserved total exactly -- no drift, no leftover micro. A venue
    that withholds nothing books ``0`` and has no reservation invented for it.
    Two residuals stay, both closed: an order captured from the *connector* at
    startup carries no reservation to release (its collateral is already inside
    the seeded cash baseline), and a cancel releases collateral no entry books
    at all -- in each case the venue moves further than the books explain and
    the cycle still breaches.

    Fail closed: an entry whose accounting cannot be reconstructed (a malformed
    payload leaf, or an event the seam should never have delivered) is never
    skipped past. The advance stops at it and latches off permanently, so the
    unexplained gap keeps showing up as divergence for the life of the process
    rather than being quietly re-baselined away. Entries booked *before* the gap
    are real facts and still apply; only advancing past it is refused.

    Omitting ``fill_accounting`` restores the pre-#365 behavior exactly: the
    baseline never moves, and :meth:`get_expectations` returns one identical
    object every call. That is the correct wiring wherever nothing books the
    evidence -- ``windbreak/main.py``'s LIVE composition still omits it
    deliberately, because a feed there would advance the expectation on evidence
    no process records, which is a fail-*open* regression.

    The projection is *scoped*, not a full venue reconstruction, because the
    ledger is not a complete record of intended venue state -- each dimension is
    ledger-seeded only where the ledger actually carries the fact, and otherwise
    falls back, independently, to the connector's own startup capture:

    * **cash** (ledger-seeded): the ``exchange_verified_available_cash`` of the
      last non-breach verification event *this component recorded*
      (``"VerificationPassed"`` / ``"VerificationDrift"``); a
      ``"VerificationMismatch"`` is excluded so a restart never re-baselines onto
      a cash figure already known to breach. Fallback: the connector's available
      cash. A full reconstruction is impossible here because incremental fills
      and available-cash movements are not ledgered with amounts.
    * **positions** (ledger-seeded): the rows of the last
      ``PositionsSnapshotRecorded`` event *this component recorded*. Fallback:
      the connector's positions.
    * **open orders** (startup-connector-captured, then ledger-advanced): empty
      only while the history ends KILLED and unrearmed (a ``KillEngaged`` --
      whose kill cancelled every resting order -- with no matching later
      ``KillReArmed``), otherwise the connector's resting orders, each carried
      with the quantity it rests for. The killed-vs-rearmed decision reuses the
      kernel's one canonical kill fold
      (:func:`~windbreak.riskkernel.kill.kill_state_in`), so it can never
      disagree with the mode the kernel replays to; scoping it to the *current*
      kill (not any historical one) is what stops a past kill or routine
      kill/re-arm drill from permanently zeroing the expectation and turning
      every later legitimately-resting order into a false-positive breach. Open
      orders still cannot be ledger-*seeded* across a restart -- a booked arrival
      says an order rested, not that it is still resting now -- so the startup
      capture remains the connector's. What issue #390 added is the *advance*:
      once captured, the set moves only by booked arrivals and by booked fills
      naming the order they consumed, never by re-reading the venue. This
      dimension is also deliberately *not* component-scoped, precisely because it
      delegates to that one kill fold -- the same fold ``KillSwitch.from_events``
      and ``RiskKernel.from_events`` run over the same unfiltered history -- so
      scoping it here would let the open-order expectation disagree with the kill
      mode the kernel replays to. A foreign kill event can only empty the
      expectation, i.e. make it strictly more conservative, so that asymmetry
      fails closed (follow-up: issue #327).

    Trust boundary: both ledger-seeded dimensions fold only events stamped
    :data:`_COMPONENT` (:func:`_own_component_events`), because one ledger is
    routinely shared across processes and the PAPER loop writes
    ``component="scheduler"`` position snapshots taken from a *simulated*
    exchange. Stated plainly: nothing records a ``riskkernel``-stamped
    ``PositionsSnapshotRecorded`` today, so **positions is always
    connector-fallback in practice** -- that ledger-seed path is a
    forward-compatible hook, not a live projection of venue positions. For cash
    the filter is latent hardening only, since this module is currently the sole
    emitter of the verification event types the cash seed reads.

    Bounded cross-restart residual: because the cash seed is the last non-breach
    verification cash, a within-tolerance drift persisted by a prior run's last
    clean/drift cycle becomes the next run's baseline, so tolerated drift can
    ratchet across restarts -- but only by at most one tolerance band per
    restart, since any move beyond tolerance is a breach and a breach event is
    never allowed to seed the baseline.

    :class:`~windbreak.connector.readonly.ReadOnlyVenueView` is annotation-only
    (:data:`TYPE_CHECKING`), as everywhere in this module, so this class adds no
    runtime ``verification`` <-> ``process`` import cycle.
    """

    def __init__(
        self,
        history: Iterable[Event],
        connector: ReadOnlyVenueView,
        *,
        fill_accounting: FillAccountingFeed | None = None,
    ) -> None:
        """Project ``history`` and ``connector`` into the baseline expectation.

        The history is materialized once (so a one-shot iterable is safe) and
        each dimension is folded independently; the connector is read only for
        the dimensions history does not seed. This all happens here, at
        construction: a later connector mutation never changes the result.

        Args:
            history: The replayed startup event history, oldest first.
            connector: The read-only market connector supplying the per-dimension
                fallbacks. It is read exactly here, at construction.
            fill_accounting: The seam newly booked fills and order arrivals are
                folded in through (issues #365 and #390). ``None`` -- the
                default -- leaves the baseline frozen for the life of the
                source, which is the correct wiring for a composition root where
                nothing books this accounting: an expectation that advanced on
                evidence nobody records would drift away from the venue
                silently.
        """
        events = tuple(history)
        self._fill_accounting = fill_accounting
        self._unaccountable = False
        self._open_orders = _seed_open_orders(events, connector)
        self._expectations = LedgerExpectations(
            expected_available_cash=_seed_cash(events, connector),
            expected_positions=_seed_positions(events, connector),
            expected_open_order_ids=frozenset(self._open_orders),
        )

    def get_expectations(self) -> LedgerExpectations:
        """Return the baseline, advanced by every entry booked since the last call.

        The connector is never touched here -- only the ledgered entries the
        fill-accounting seam drains. Once an unreconstructable entry has been
        seen the source latches and stops advancing for good, so an unexplained
        gap in the books can never be quietly re-baselined away.

        Returns:
            The current :class:`LedgerExpectations`. With no fill-accounting
            seam wired (or once latched) this is one identical object on every
            call, regardless of any later connector mutation.
        """
        if self._fill_accounting is None or self._unaccountable:
            return self._expectations
        for event in self._fill_accounting.drain():
            advanced = self._advance(event)
            if advanced is None:
                self._unaccountable = True
                break
            self._expectations = advanced
        return self._expectations

    def _advance(self, event: Event) -> LedgerExpectations | None:
        """Return the expectation advanced by one booked entry, or ``None``.

        The seam promises exactly two kinds of entry -- a booked fill and a
        booked order arrival -- so an event of any other type means it is not
        delivering what it promised, which is a gap in the books rather than a
        row to skip.

        Args:
            event: The event drained from the fill-accounting seam.

        Returns:
            The advanced :class:`LedgerExpectations`, or ``None`` when this
            entry cannot be accounted for.
        """
        if event.event_type == _FILL_ACCOUNTED_EVENT_TYPE:
            entry = _fill_entry(event)
            if entry is None:
                return None
            released = self._consume(entry)
            if released is None:
                return None
            return _advanced(self._expectations, entry, self._open_orders, released)
        if event.event_type == _RESTING_ORDER_ACCOUNTED_EVENT_TYPE:
            arrival = _resting_entry(event)
            if arrival is None or arrival.venue_order_id in self._open_orders:
                return None
            self._open_orders[arrival.venue_order_id] = _RestingOrder(
                total_centis=arrival.quantity_centis,
                consumed_centis=0,
                reserved_micros=arrival.reserved_micros,
                released_micros=0,
            )
            return _with_arrival(
                self._expectations, self._open_orders, arrival.reserved_micros
            )
        return None

    def _consume(self, entry: _FillEntry) -> int | None:
        """Draw the named resting order down, returning the collateral released.

        A fill that names no order consumes nothing, releases nothing, and
        trivially succeeds. A fill that names an order the expectation has never
        held fails instead of inventing one: applying it would conjure a resting
        order and discard it in the same step, quietly absorbing a retirement
        the books cannot explain. That is the exact fail-open shape this
        dimension exists to refuse, so it is reported as a gap and latches the
        advance.

        The drawdown is the fill's own size -- the magnitude of its signed
        position delta -- and the id leaves the expectation only once the order
        is exhausted. Consuming *more* than the order held still retires it and
        no more: the consumption is clamped to the order's own size, so the
        surplus releases no extra collateral and is not silently absorbed
        anywhere else, and a venue that moved further than the books explain
        still diverges.

        Args:
            entry: The booked fill's reconstructed accounting.

        Returns:
            The collateral this fill released, in micros, or ``None`` when the
            named order could not be drawn down at all. ``0`` and ``None`` are
            different answers: the caller must not read a zero release as a gap.
        """
        if not entry.venue_order_id:
            return 0
        order = self._open_orders.get(entry.venue_order_id)
        if order is None:
            return None
        consumed = min(
            order.consumed_centis + abs(entry.position_delta.value),
            order.total_centis,
        )
        released = _released_by(order, consumed)
        order.consumed_centis = consumed
        order.released_micros += released
        if consumed >= order.total_centis:
            del self._open_orders[entry.venue_order_id]
        return released


def _classify(
    *,
    balance_ok: bool,
    position_ok: bool,
    open_order_ok: bool,
    drift_present: bool,
) -> VerificationOutcome:
    """Grade a cycle from its per-dimension ok flags and drift presence.

    Args:
        balance_ok: Whether the balance drift was within tolerance.
        position_ok: Whether every position drift was within tolerance.
        open_order_ok: Whether the open-order sets matched exactly.
        drift_present: Whether any nonzero (but tolerable) drift was observed.

    Returns:
        ``BREACH`` if any dimension failed, else ``DRIFT_WITHIN_TOLERANCE`` if a
        tolerable drift was present, else ``CLEAN``.
    """
    if not (balance_ok and position_ok and open_order_ok):
        return VerificationOutcome.BREACH
    if drift_present:
        return VerificationOutcome.DRIFT_WITHIN_TOLERANCE
    return VerificationOutcome.CLEAN


class ReadOnlyVerifier:
    """Cross-checks a read-only venue against ledger expectations each cycle."""

    def __init__(
        self,
        connector: ReadOnlyVenueView,
        expectation_source: ExpectationSource,
        tolerances: VerificationTolerances,
        dispatcher: AlertDispatcher,
        ledger_writer: KernelLedgerWriter,
    ) -> None:
        """Initialize the verifier.

        Args:
            connector: The read-only venue view to observe the venue through.
                Narrower than a
                :class:`~windbreak.connector.interface.MarketConnector` on
                purpose: it carries no order-placing method, so this cycle
                structurally cannot trade (SPEC S1.1 invariant 3).
            expectation_source: The seam supplying ledger-derived expectations.
            tolerances: The per-dimension drift tolerances.
            dispatcher: The alert dispatcher jurisdiction and mismatch alerts
                fan out through.
            ledger_writer: The seam each cycle's single event is recorded
                through.
        """
        self._connector = connector
        self._expectation_source = expectation_source
        self._tolerances = tolerances
        self._dispatcher = dispatcher
        self._ledger_writer = ledger_writer

    def run_cycle(self, now_epoch_s: int) -> VerificationSnapshot:
        """Run one verification cycle and return its snapshot.

        Fetches the venue's balances, positions, open orders, and balance
        semantics; diffs each against the ledger expectations; grades the
        outcome; fires a jurisdiction-unknown alert for each held market whose
        jurisdiction is unknown and a reconciliation-mismatch alert on a breach;
        and records exactly one bare event.

        A connector that *raises* while being observed is graded a breach rather
        than allowed to propagate: see :meth:`_unobservable_venue_breach` for why
        an escaping exception here would defeat the very halt this cycle exists
        to drive.

        Args:
            now_epoch_s: The epoch second to stamp the snapshot at.

        Returns:
            The :class:`VerificationSnapshot` describing the cycle -- a forced
            ``BREACH`` snapshot when the venue could not be observed at all.
        """
        expectations = self._expectation_source.get_expectations()
        try:
            positions = self._connector.get_positions()
            open_orders = self._connector.get_open_orders()
            observed_cash = self._connector.get_balances().available
            semantics_known = self._connector.get_balance_semantics().is_fully_known()
        except Exception as exc:  # Fail-closed: an unobservable venue is a breach.
            return self._unobservable_venue_breach(now_epoch_s, expectations, exc)

        cash_drift = self._cash_drift(observed_cash, expectations)
        position_drift = self._position_drift(positions, expectations)
        open_order_ok = self._open_orders_match(open_orders, expectations)
        balance_ok = cash_drift.value <= self._tolerances.balance_tolerance.value
        position_ok = position_drift <= self._tolerances.position_tolerance.value

        outcome = _classify(
            balance_ok=balance_ok,
            position_ok=position_ok,
            open_order_ok=open_order_ok,
            drift_present=cash_drift.value > 0 or position_drift > 0,
        )
        snapshot = VerificationSnapshot(
            outcome=outcome,
            balance_ok=balance_ok,
            position_ok=position_ok,
            open_order_ok=open_order_ok,
            verified_at_epoch_s=now_epoch_s,
            exchange_verified_available_cash=observed_cash,
            cash_drift=cash_drift,
            semantics_fully_known=semantics_known,
        )
        # Record the audit event before any alerting: alert dispatch performs
        # per-ticker ``get_market`` lookups, and recording the cycle's outcome
        # first ensures a raise there can never lose the ledgered verification
        # event (nor, at the process level, the HALT this snapshot drives).
        self._record(snapshot)
        self._dispatch_alerts(positions, open_orders, outcome)
        return snapshot

    def _unobservable_venue_breach(
        self,
        now_epoch_s: int,
        expectations: LedgerExpectations,
        exc: Exception,
        /,
    ) -> VerificationSnapshot:
        """Record a forced ``BREACH`` for a venue that could not be observed.

        A connector read can raise -- ``PaperExchange.get_positions`` refuses to
        net a two-sided holding into one row, and a live adapter can fail its
        transport -- and those reads happen *before* :meth:`_record`. Letting
        such a raise propagate would take it straight out through
        ``RiskKernel.run_verification_cycle`` and the kernel's heartbeat loop,
        which hold no handler: the process would die with no ledgered
        verification event, no reconciliation alert, and -- worst -- no
        ``HALT``, since the halt is driven by the very snapshot that never got
        returned. A connector refusing to describe the account is the strongest
        possible reason to stop trading, so it is graded a breach here and
        recorded on the same audited path a drift-beyond-tolerance breach takes.

        Every dimension is reported not-ok because an unobservable venue has
        verified none of them, and the outcome is forced rather than derived:
        against an all-flat expectation the ordinary grading would score
        "observed nothing" as ``CLEAN``, which is exactly the fabricated healthy
        state fail-closed exists to prevent. The observed cash is reported as
        zero and the whole expected balance therefore becomes drift (via the
        same :meth:`_cash_drift` every cycle uses), so the snapshot can only
        tighten the equity floor it feeds, never loosen it.

        Args:
            now_epoch_s: The epoch second to stamp the snapshot at.
            expectations: The expectations the unobserved venue would have been
                diffed against; supplies the cash the drift is measured from.
            exc: The exception the connector raised, reported to the operator in
                the alert body.

        Returns:
            The forced ``BREACH`` snapshot, already recorded.
        """
        snapshot = VerificationSnapshot(
            outcome=VerificationOutcome.BREACH,
            balance_ok=False,
            position_ok=False,
            open_order_ok=False,
            verified_at_epoch_s=now_epoch_s,
            exchange_verified_available_cash=MoneyMicros(0),
            cash_drift=self._cash_drift(MoneyMicros(0), expectations),
            semantics_fully_known=False,
        )
        # Same ordering guarantee as the ordinary path: the audit event lands
        # before any alerting, so a raising dispatcher cannot lose the breach.
        self._record(snapshot)
        self._dispatcher.dispatch(
            AlertType.RECONCILIATION_MISMATCH, f"{_UNREADABLE_MESSAGE}: {exc}"
        )
        return snapshot

    def _cash_drift(
        self, observed_cash: MoneyMicros, expectations: LedgerExpectations
    ) -> MoneyMicros:
        """Return the absolute available-cash drift for this cycle.

        Args:
            observed_cash: The venue-reported available cash, in micros.
            expectations: The ledger expectations supplying the expected cash.

        Returns:
            The absolute cash drift, in micros.
        """
        drift = abs(observed_cash.value - expectations.expected_available_cash.value)
        return MoneyMicros(drift)

    def _position_drift(
        self,
        positions: tuple[Position, ...],
        expectations: LedgerExpectations,
    ) -> int:
        """Return the largest absolute per-ticker position drift, in centis.

        A ticker present on only one side is diffed against an implicit zero on
        the other, so an unexpected held position (or a missing expected one) is
        surfaced as its full size.

        Args:
            positions: The venue-reported open positions.
            expectations: The ledger expectations supplying expected positions.

        Returns:
            The maximum absolute position drift across every observed or
            expected ticker, in contract-centis; ``0`` when both sides are flat.
        """
        observed = {position.ticker: position.quantity.value for position in positions}
        expected = {
            ticker: quantity.value
            for ticker, quantity in expectations.expected_positions.items()
        }
        diffs = [
            abs(observed.get(ticker, 0) - expected.get(ticker, 0))
            for ticker in set(observed) | set(expected)
        ]
        return max(diffs, default=0)

    def _open_orders_match(
        self,
        open_orders: tuple[OpenOrder, ...],
        expectations: LedgerExpectations,
    ) -> bool:
        """Return whether the observed open-order ids match the expected set.

        Args:
            open_orders: The venue-reported resting orders.
            expectations: The ledger expectations supplying the expected id set.

        Returns:
            ``True`` iff the observed and expected id sets are equal.
        """
        observed_ids = frozenset(order.id for order in open_orders)
        return observed_ids == expectations.expected_open_order_ids

    def _dispatch_alerts(
        self,
        positions: tuple[Position, ...],
        open_orders: tuple[OpenOrder, ...],
        outcome: VerificationOutcome,
    ) -> None:
        """Fire jurisdiction-unknown and (on a breach) mismatch alerts.

        Args:
            positions: The venue-reported open positions.
            open_orders: The venue-reported resting orders.
            outcome: The graded outcome of the cycle.
        """
        self._alert_unknown_jurisdictions(positions, open_orders)
        if outcome is VerificationOutcome.BREACH:
            self._dispatcher.dispatch(
                AlertType.RECONCILIATION_MISMATCH, _MISMATCH_MESSAGE
            )

    def _alert_unknown_jurisdictions(
        self,
        positions: tuple[Position, ...],
        open_orders: tuple[OpenOrder, ...],
    ) -> None:
        """Fire one warning alert per held market of unknown jurisdiction.

        Args:
            positions: The venue-reported open positions.
            open_orders: The venue-reported resting orders.
        """
        held = {position.ticker for position in positions}
        held |= {order.ticker for order in open_orders}
        for ticker in sorted(held):
            market = self._connector.get_market(ticker)
            if market.jurisdiction_status == _UNKNOWN_JURISDICTION:
                self._dispatcher.dispatch(
                    AlertType.JURISDICTION_UNKNOWN,
                    f"held market {ticker} has unknown jurisdiction status",
                )

    def _record(self, snapshot: VerificationSnapshot) -> None:
        """Record exactly one bare event describing the cycle's snapshot.

        Args:
            snapshot: The snapshot to record.
        """
        self._ledger_writer.record(
            Event(
                event_type=_EVENT_TYPE_BY_OUTCOME[snapshot.outcome],
                component=_COMPONENT,
                payload_schema_version=_PAYLOAD_SCHEMA_VERSION,
                payload=_payload(snapshot),
            )
        )


def _payload(snapshot: VerificationSnapshot) -> dict[str, object]:
    """Project a snapshot into a JSON-safe, float-free event payload.

    Every leaf is an ``int``, ``str``, or ``bool`` (SPEC S6.1): the money terms
    are emitted as their scaled-integer ``.value`` and the outcome as its enum
    member name.

    Args:
        snapshot: The snapshot to project.

    Returns:
        The event payload mapping.
    """
    return {
        "outcome": snapshot.outcome.name,
        "balance_ok": snapshot.balance_ok,
        "position_ok": snapshot.position_ok,
        "open_order_ok": snapshot.open_order_ok,
        "verified_at_epoch_s": snapshot.verified_at_epoch_s,
        "exchange_verified_available_cash": (
            snapshot.exchange_verified_available_cash.value
        ),
        "cash_drift": snapshot.cash_drift.value,
        "semantics_fully_known": snapshot.semantics_fully_known,
    }
