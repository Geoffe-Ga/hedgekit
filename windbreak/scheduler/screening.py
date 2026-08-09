"""Turn an exchange's market universe into a bounded candidate set (issue #345).

The PAPER loop used to forecast ``next(iter(exchange.markets))`` -- one
arbitrary ticker, fixed for the life of the process, never screened. This module
is the seam that replaces it: each tick walks the venue's markets, puts each one
through the real §16 :class:`~windbreak.screener.Screener`, and hands back the
eligible ones as :class:`MarketCandidate` values the tick then forecasts over.

Three properties are load-bearing, and each is a separate design commitment.

**Deterministic order.** The walk is over ``sorted(..., key=ticker)``, never the
exchange mapping's own iteration order. SPEC S9.1/S9.10 require that two runs
over identical inputs produce identical ledgers, and "identical inputs" must not
quietly include the order a JSON file happened to list markets in.

**A hard bound, enforced on the walk itself.** Every candidate is one *paid*
forecast: since issue #399 each ensemble vote books real spend against the
per-forecast and per-UTC-day :class:`~windbreak.forecast.budget.ResearchBudget`
ceilings, so an unbounded universe is an unbounded bill that ends in a
fail-closed halt. :func:`screen_universe` therefore stops walking the moment it
holds ``max_candidates``, which bounds the tick's research spend at
``max_candidates`` forecasts *and* bounds the book reads that found them.

**Screening is free of model calls.** The four
:mod:`windbreak.screener.filters` filters are pure integer comparisons over a
market's own metadata and its book. Deciding what to spend research money on
therefore costs no research money -- which is the only reason screening a
universe every tick is affordable at all.

Nothing here is a screening *rule*: the thresholds, the blocklist, and the
horizon window all live in :class:`~windbreak.config.schema.ScreenerConfig` and
are enforced by the screener as built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from windbreak.connector.snapshot import SCREEN_DECISION_EVENT
from windbreak.ledger.events import ScreenDecisionRecorded

if TYPE_CHECKING:
    from windbreak.connector.models import NormalizedMarket, OrderBookSnapshot
    from windbreak.connector.snapshot import ConnectorEvent, MarketScreener
    from windbreak.ledger.store import SqliteLedgerStore


@dataclass(frozen=True, slots=True)
class MarketCandidate:
    """One screened market a tick may forecast, paired with the book it passed on.

    The book is carried rather than re-fetched because the screen and every
    later stage must judge *one* observation. Re-reading would give the snapshot
    stage a second, later book than the depth floor was measured against, and on
    a live venue the two could disagree -- leaving the ledger claiming a market
    was screened on liquidity it no longer had.

    Attributes:
        market: The screened market's normalized metadata.
        order_book: The very YES book the screen's depth floor was measured
            against, carrying the instant it was actually observed.
    """

    market: NormalizedMarket
    order_book: OrderBookSnapshot

    @property
    def ticker(self) -> str:
        """Return the candidate's market ticker."""
        return self.market.ticker


class MarketUniverse(Protocol):
    """The two reads :func:`screen_universe` needs from an exchange."""

    def list_markets(self) -> tuple[NormalizedMarket, ...]:
        """Return every market the venue offers.

        Returns:
            The venue's markets, in the venue's own order.
        """
        ...

    def get_order_book(self, ticker: str) -> OrderBookSnapshot:
        """Return one market's current YES book.

        Args:
            ticker: The market whose book is wanted.

        Returns:
            That market's order-book snapshot.
        """
        ...


class ScreenLedgerWriter:
    """Translate the screener's connector events into typed ledger rows.

    The :class:`~windbreak.screener.Screener` writes through the connector's
    :class:`~windbreak.connector.snapshot.EventLedgerWriter` seam, while the
    PAPER loop's audit trail is the hash-chained
    :class:`~windbreak.ledger.store.SqliteLedgerStore` of typed
    :class:`~windbreak.ledger.events.Event` rows. This adapter is the one-way
    translation between them, mirroring the kernel and budget writers
    :mod:`windbreak.scheduler.loop` already wires -- and it is what finally
    *emits* ``ScreenDecisionRecorded``, the event issue #159 found declared but
    dead.

    The translation table is closed. An event type it was not taught is refused
    rather than dropped: the screener's other event, ``LEGAL_RISK_ACK``, records
    an operator's explicit acceptance of legal risk, and losing that silently is
    exactly the class of gap a hash-chained ledger exists to make impossible.
    """

    def __init__(self, store: SqliteLedgerStore, *, component: str) -> None:
        """Initialize the adapter over a durable store.

        Args:
            store: The hash-chained ledger the translated rows are appended to.
            component: The component label stamped on every appended row
                (keyword-only).
        """
        self._store = store
        self._component = component

    def record(self, event: ConnectorEvent) -> None:
        """Append one screening decision to the durable ledger.

        The event's per-filter ``measured`` detail is deliberately not carried:
        :class:`~windbreak.ledger.events.ScreenDecisionRecorded`'s schema states
        the verdict and the filters that produced it, which is what the audit
        trail is asked for, and widening the schema is a migration rather than a
        wiring change.

        Args:
            event: The screener's connector event.

        Raises:
            ValueError: If the event is not a ``SCREEN_DECISION``. Refusing is
                the fail-closed answer: a dropped event is an audit gap that
                looks exactly like an event that never happened.
        """
        if event.event_type != SCREEN_DECISION_EVENT:
            raise ValueError(
                f"{type(self).__name__} was handed a {event.event_type!r} event "
                f"it cannot translate; it records only {SCREEN_DECISION_EVENT!r}"
            )
        payload = event.payload
        self._store.append(
            ScreenDecisionRecorded(
                component=self._component,
                ticker=cast("str", payload["ticker"]),
                eligible=cast("bool", payload["eligible"]),
                blocked_by=list(cast("list[str]", payload["blocked_by"])),
            )
        )


def require_candidate_bound(max_candidates: int) -> int:
    """Return the per-tick candidate bound, refusing a non-positive one.

    Called at composition time rather than per tick, so a misconfigured bound
    refuses to start instead of producing an always-idle loop that looks healthy
    from the outside -- the same fail-fast posture
    :func:`~windbreak.scheduler.loop.build_paper_deps` already takes on the
    budget ceilings.

    Args:
        max_candidates: The configured per-tick candidate ceiling.

    Returns:
        The bound, unchanged.

    Raises:
        ValueError: If the bound is below one, which would be a loop that can
            never forecast anything.
    """
    if max_candidates < 1:
        raise ValueError(
            f"screener.max_candidates_per_tick must be at least 1; got "
            f"{max_candidates}, which is a loop that can never forecast"
        )
    return max_candidates


def screen_universe(
    universe: MarketUniverse,
    screener: MarketScreener,
    *,
    max_candidates: int,
) -> tuple[MarketCandidate, ...]:
    """Screen the venue's markets into at most ``max_candidates`` candidates.

    Walks the universe in ascending ticker order, reads each market's book once,
    and puts the pair through the screener -- which ledgers exactly one decision
    per market examined, eligible or not. The walk stops as soon as
    ``max_candidates`` eligible markets are held, so both the tick's research
    spend and the book reads that decide it are bounded.

    Markets past the stopping point are simply not examined this tick, and no
    decision is ledgered for them. That is the honest record: the ledger says
    which markets were looked at, and does not claim a verdict on markets the
    tick never reached.

    Args:
        universe: The exchange surface supplying markets and books.
        screener: The screener each market is put through, which ledgers the
            decision itself.
        max_candidates: The per-tick candidate ceiling, already validated by
            :func:`require_candidate_bound` (keyword-only).

    Returns:
        The eligible candidates, in ascending ticker order, never more than
        ``max_candidates`` of them -- and empty when the whole universe screens
        out, which is a tick that forecasts nothing rather than one that falls
        back to an unscreened market.
    """
    candidates: list[MarketCandidate] = []
    for market in sorted(universe.list_markets(), key=lambda entry: entry.ticker):
        order_book = universe.get_order_book(market.ticker)
        if screener.screen_book(market, order_book).eligible:
            candidates.append(MarketCandidate(market=market, order_book=order_book))
            if len(candidates) >= max_candidates:
                break
    return tuple(candidates)
