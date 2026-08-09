"""The PAPER tick's fill-leg routing path (issue #162).

`windbreak.scheduler.loop`'s `_approve_stage` -> `_route_intent` ->
`_reconcile_to_fixpoint` chain is the only path in the loop that turns an
approved intent into a real order. Until today it was dormant: with the stock
fixtures the real kernel vetoed every intent the selector emitted, so
`_approve_stage`'s `if outcome.token is not None:` arm never ran and neither
routing function was ever entered. Issues #404/#407/#408/#415 made the tick
screen a universe, size against real exposure, book real vote costs, and bind
the daily-notional cap -- the path is now genuinely reachable, and untested.

`tests/integration/test_paper_loop.py`'s
`test_fill_leg_via_doubled_approval_seam_reaches_a_terminal_gateway_state`
looks like it covers this and does not: it calls `deps.gateway.process_intent`
*itself*, proving the Gateway -> PaperExchange -> Reconciler wiring while
stepping around the loop code that is supposed to drive it. Nothing there fails
if `_route_intent` routes the wrong leg, drops a leg, or reports a fill the
gateway never acked.

So every test below drives the *production* routing functions and asserts the
observable consequence -- the exact contract-centis reported, the exact holding
the venue ends up with, and the exact gateway transitions ledgered -- never
merely that a line ran. The approval seam is the one doubled component, exactly
as `PaperTickDeps`' own docstring anticipates ("the ``approval`` seam is
intentionally swappable ... so a test can drive the gateway/exchange fill leg
with a doubled, fixed-token seam while reusing every other real component"). The
double mints a *genuinely signed* token against `deps.verification_key`, so the
real Gateway still verifies the signature, consumes the token, and refuses
anything it should refuse; nothing here bypasses a production guard.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

from tests.integration.conftest import (
    FIXED_NOW_EPOCH_S,
    candidate_for,
    ledger_path_for,
)
from tests.order_gateway.conftest import issue_matching_token
from windbreak.numeric.types import (
    ContractCentis,
    MoneyMicros,
    PricePips,
    ProbabilityPpm,
)
from windbreak.order_gateway.reconciler import ReconcileOutcome, Reconciler
from windbreak.riskkernel.checks import Decision
from windbreak.riskkernel.reservations import ApprovalOutcome
from windbreak.selector.types import SelectorOrderIntent

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig
    from windbreak.riskkernel.checks import OrderIntent
    from windbreak.scheduler.loop import PaperTickDeps

#: The sole ticker in the shared `deep_walk` books fixture.
_TICKER = "MKT-DEEP"

#: The fill `deep_walk`'s sole 4600-pip/200-centis ask yields for a crossing
#: 200-centis buy: the venue's participation cap takes a quarter of the resting
#: level. Pinned as a literal (the figure every Gateway-suite test over this
#: fixture already asserts) rather than recomputed from the book, so a test
#: deriving its expectation from the code under test cannot pass against any
#: fill at all.
_FIRST_LEG_FILL_CENTIS = 50

#: The total two distinct crossing legs fill in one `_approve_stage` call. Each
#: takes the same participation slice off `deep_walk`'s 200-centis top ask -- the
#: fixture's resting level is not decremented between the two submissions -- so
#: the total is twice one leg. Pinned as its own literal rather than written
#: `2 * _FIRST_LEG_FILL_CENTIS`, so the expectation states an observed venue
#: outcome instead of restating the arithmetic it is meant to check.
_TWO_LEG_FILL_CENTIS = 100

#: `windbreak.scheduler.loop._RECONCILE_MAX_CYCLES`, pinned as a literal for the
#: same reason: read back off the module, the bound test below would pass
#: against any bound, including an unbounded one.
_RECONCILE_MAX_CYCLES = 5


def _fixed_clock() -> int:
    """Return the fixed epoch second every bundle in this module is built on."""
    return FIXED_NOW_EPOCH_S


def _build_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    research_tools_factory,
) -> PaperTickDeps:
    """Build one fully wired `PaperTickDeps` over the shared offline fixtures.

    Args:
        books_dir: The `deep_walk` books-fixture directory.
        cassette_path: The (empty) recorded-cassette path.
        ledger_path: Where the tick's `SqliteLedgerStore` is created.
        report_dir: Where weekly-report stubs would be written.
        config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.

    Returns:
        A fully wired `PaperTickDeps`, on this module's fixed clock.
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


def _crossing_intent(
    *, intent_id: str = "intent-0001", idempotency_key: str = "idem-0001"
) -> SelectorOrderIntent:
    """Build the selector-shaped intent that crosses `deep_walk`'s top ask.

    Mirrors `tests/order_gateway/conftest.py::make_intent`'s defaults exactly
    (4600 pips / 200 centis against `MKT-DEEP`), but as the
    `SelectorOrderIntent` a real `SelectorDecision` carries, so the decision fed
    to `_approve_stage` has the type the production selector would have emitted
    rather than a loosened stand-in.

    Args:
        intent_id: The intent's identifier.
        idempotency_key: The idempotency key the Gateway caches submissions
            under; two legs in one decision need distinct keys or the second is
            a replay of the first rather than a second order.

    Returns:
        The crossing `SelectorOrderIntent`.
    """
    return SelectorOrderIntent(
        intent_id=intent_id,
        market_ticker=_TICKER,
        outcome="yes",
        action="buy",
        price=PricePips(4600),
        size=ContractCentis(200),
        max_notional=MoneyMicros(50_000_000),
        implied_probability=ProbabilityPpm(520_000),
        idempotency_key=idempotency_key,
    )


class _MintingApprovalSeam:
    """An `ApprovalSeam` double minting a real token for whatever it is handed.

    Mints per call against the intent it actually receives, rather than
    returning one token fixed at construction time. That is what makes the
    multi-leg test meaningful: a seam handing back one pre-minted token would
    make every leg look identical to the Gateway, so routing the *wrong* leg
    would be indistinguishable from routing the right one.
    """

    __slots__ = ("_expires_at", "_key", "seen")

    def __init__(self, key_material: bytes, *, expires_at: int) -> None:
        """Record the signing key and expiry every minted token is issued under.

        Args:
            key_material: `deps.verification_key` -- the ephemeral key the real
                Gateway verifies under, so these tokens are genuine.
            expires_at: The epoch second the minted tokens expire at.
        """
        self._key = key_material
        self._expires_at = expires_at
        self.seen: list[OrderIntent] = []

    def decide(self, intent: OrderIntent, context: object) -> ApprovalOutcome:
        """Approve `intent`, carrying a token minted to match it exactly.

        Args:
            intent: The intent to approve; its own fields are what the token's
                claims are built from.
            context: The composed `EvaluationContext`; unused, since this double
                does not re-run the kernel's checks.

        Returns:
            A non-vetoing `ApprovalOutcome` carrying the matching token.
        """
        del context
        self.seen.append(intent)
        return ApprovalOutcome(
            decision=Decision(vetoed=False, reasons=()),
            token=issue_matching_token(
                intent, key_material=self._key, expires_at=self._expires_at
            ),
        )


class _VetoingApprovalSeam:
    """An `ApprovalSeam` double that vetoes, minting no token at all."""

    __slots__ = ()

    def decide(self, intent: OrderIntent, context: object) -> ApprovalOutcome:
        """Veto `intent` with no token.

        Args:
            intent: The intent under review; unused.
            context: The composed `EvaluationContext`; unused.

        Returns:
            A vetoing `ApprovalOutcome` whose `token` is `None`.
        """
        del intent, context
        return ApprovalOutcome(
            decision=Decision(vetoed=True, reasons=("test veto",)), token=None
        )


class _CountingReconciler(Reconciler):
    """The real `Reconciler`, wrapped so its cycles can be counted.

    Subclasses `Reconciler` (rather than standing in for it structurally) so
    `dataclasses.replace(deps, reconciler=...)` still type-checks against
    `PaperTickDeps.reconciler`, and delegates every cycle to the genuinely wired
    instance `build_paper_deps` produced, so what is counted is real
    reconciliation work rather than a script.
    """

    __slots__ = ("_inner", "cycles")

    def __init__(self, inner: Reconciler) -> None:
        """Wrap `inner`, starting the cycle count at zero.

        Args:
            inner: The real, fully wired `Reconciler` to delegate to.
        """
        self._inner = inner
        self.cycles = 0

    def run_once(self) -> ReconcileOutcome:
        """Delegate one reconciliation cycle, counting it.

        Returns:
            The real `Reconciler`'s own outcome, unaltered.
        """
        self.cycles += 1
        return self._inner.run_once()


class _ScriptedReconciler(Reconciler):
    """A `Reconciler` returning a fixed script of outcomes, counting its calls.

    `_reconcile_to_fixpoint`'s three termination arms are properties of the
    *loop*, not of any venue state, and two of them (a halted cycle, and the
    five-cycle ceiling) cannot be produced on demand by a real venue without
    fabricating a mismatch. Scripting the outcomes states each arm's precondition
    directly. The last scripted outcome repeats for any further call, so a loop
    that failed to terminate would run away rather than raise `IndexError` and
    look like a passing bound.
    """

    __slots__ = ("_outcomes", "calls")

    def __init__(self, outcomes: tuple[ReconcileOutcome, ...]) -> None:
        """Load the outcome script.

        Args:
            outcomes: The outcomes to return, in order.
        """
        self._outcomes = outcomes
        self.calls = 0

    def run_once(self) -> ReconcileOutcome:
        """Return the next scripted outcome, counting the call.

        Returns:
            The scripted `ReconcileOutcome` for this call.
        """
        self.calls += 1
        return self._outcomes[min(self.calls, len(self._outcomes)) - 1]


def _acked_client_order_ids(deps: PaperTickDeps) -> list[str]:
    """Return the client order id of every ACKED transition the ledger carries.

    Args:
        deps: The bundle whose store is read.

    Returns:
        One id per `OrderTransitionLedgered` row reaching `ACKED`, in ledger
        order.
    """
    ids = []
    for record in deps.store.read_all():
        if record.event_type != "OrderTransitionLedgered":
            continue
        data = json.loads(record.payload_json)["data"]
        if data["to_state"] == "ACKED":
            ids.append(data["client_order_id"])
    return ids


def _held_centis(deps: PaperTickDeps) -> int:
    """Return the venue's own reported holding in `_TICKER`, in contract-centis.

    Read from the exchange rather than from any loop-side tally, so the
    assertion is about what was actually traded, not about what the loop
    believes it traded.

    Args:
        deps: The bundle whose exchange is queried.

    Returns:
        The signed YES-frame quantity held, or `0` when flat.
    """
    for position in deps.exchange.get_positions():
        if position.ticker == _TICKER:
            return position.quantity.value
    return 0


def _approve_one(deps: PaperTickDeps, intents: tuple[SelectorOrderIntent, ...]) -> int:
    """Run the real `_approve_stage` over `intents` for the `deep_walk` market.

    Composes the stage's five other arguments the way `_run_candidate` does: the
    screened candidate over the exchange's current book, the tick's own clock
    reading as the pipeline heartbeat, and a real `ForecastRecord` produced by
    the real forecast stage. Nothing here is a stand-in for a stage the loop
    would have run.

    Args:
        deps: The wired bundle, with whatever approval seam the caller swapped
            in.
        intents: The intents the selector is to be treated as having emitted.

    Returns:
        `_approve_stage`'s own reported fill, in contract-centis.
    """
    from datetime import UTC, datetime

    from windbreak.scheduler.loop import (
        _approve_stage,
        _forecast_stage,
        _select_stage,
        read_candidate_exposure,
    )

    candidate = candidate_for(deps, _TICKER)
    created_at = datetime.fromtimestamp(deps.clock(), UTC)
    forecast = _forecast_stage(deps, candidate, created_at)
    assert forecast is not None, "the offline forecast stage must produce a record"
    exposure = read_candidate_exposure(deps, candidate)
    decision = dataclasses.replace(
        _select_stage(deps, candidate, forecast, created_at, exposure),
        intents=intents,
    )
    return _approve_stage(deps, candidate, decision, deps.clock(), forecast, exposure)


def test_approve_stage_routes_a_minted_intent_and_reports_the_gateway_s_fill(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A minted token routes to the real gateway and the venue really fills.

    The load-bearing assertions are the three independent observations of the
    *same* 50 centis: what `_approve_stage` returns, what the gateway ledgered,
    and what the venue reports holding. A `_route_intent` that reported a fill
    without submitting, or submitted without reporting, breaks two of the three.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
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
    reconciler = _CountingReconciler(deps.reconciler)
    seam = _MintingApprovalSeam(
        deps.verification_key, expires_at=FIXED_NOW_EPOCH_S + 60
    )
    routed = dataclasses.replace(deps, approval=seam, reconciler=reconciler)
    intent = _crossing_intent()

    filled = _approve_one(routed, (intent,))

    assert filled == _FIRST_LEG_FILL_CENTIS
    assert _held_centis(routed) == _FIRST_LEG_FILL_CENTIS
    assert len(_acked_client_order_ids(routed)) == 1
    # The seam saw the intent the decision carried -- not a substitute.
    assert [seen.intent_id for seen in seam.seen] == [intent.intent_id]
    # Routing reconciles to a fixpoint: a first cycle, then a second that
    # repeats it and terminates. One cycle would mean no fixpoint was sought.
    assert reconciler.cycles == 2
    routed.store.verify_chain()


def test_approve_stage_routes_every_minted_leg_and_sums_their_fills(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Two emitted intents produce two orders, and the reported fill is the sum.

    This is the assertion a dropped or duplicated leg fails. The legs are
    distinct orders under distinct idempotency keys, so routing only one lands
    on half the total and half the holding, and routing the *same* leg twice is
    a Gateway replay of the cached ack -- which ledgers no second ACKED
    transition, so the two-distinct-client-order-ids assertion catches it even
    though the arithmetic total would survive.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
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
    seam = _MintingApprovalSeam(
        deps.verification_key, expires_at=FIXED_NOW_EPOCH_S + 60
    )
    routed = dataclasses.replace(deps, approval=seam)
    first = _crossing_intent(intent_id="intent-0001", idempotency_key="idem-0001")
    second = _crossing_intent(intent_id="intent-0002", idempotency_key="idem-0002")

    filled = _approve_one(routed, (first, second))

    assert filled == _TWO_LEG_FILL_CENTIS
    assert _held_centis(routed) == _TWO_LEG_FILL_CENTIS
    acked = _acked_client_order_ids(routed)
    assert len(acked) == 2
    assert len(set(acked)) == 2, f"expected two distinct orders, got {acked}"
    assert [seen.intent_id for seen in seam.seen] == ["intent-0001", "intent-0002"]
    routed.store.verify_chain()


def test_approve_stage_routes_nothing_when_the_seam_mints_no_token(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A vetoed intent reaches neither the gateway nor the venue.

    The complement of the routing tests, and the one that makes them mean
    something: without it, a `_route_intent` call moved outside the
    `outcome.token is not None` guard would still pass every assertion above.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
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
    reconciler = _CountingReconciler(deps.reconciler)
    vetoed = dataclasses.replace(
        deps, approval=_VetoingApprovalSeam(), reconciler=reconciler
    )

    filled = _approve_one(vetoed, (_crossing_intent(),))

    assert filled == 0
    assert _held_centis(vetoed) == 0
    assert _acked_client_order_ids(vetoed) == []
    assert reconciler.cycles == 0
    vetoed.store.verify_chain()


def test_route_intent_reports_zero_when_the_gateway_refuses_the_token(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """An expired token routes, is refused, and reports a fill of exactly zero.

    `_route_intent`'s `result.ack is not None` arm is the loop's fail-closed
    reading of a refusal: no ack means no fill, never an assumed one. The token
    is genuinely minted and genuinely expired -- the real Gateway does the
    refusing, so this pins the loop's handling of a real refusal rather than of
    a doubled one.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.scheduler.loop import _route_intent

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    intent = _crossing_intent()
    expired = issue_matching_token(
        intent,
        key_material=deps.verification_key,
        expires_at=FIXED_NOW_EPOCH_S - 1,
    )

    filled = _route_intent(deps, intent, expired)

    assert filled == 0
    assert _held_centis(deps) == 0
    assert _acked_client_order_ids(deps) == []
    deps.store.verify_chain()


def test_reconcile_to_fixpoint_runs_no_cycle_against_a_halted_gateway(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A gateway already halted is reconciled zero times, not once.

    The check is on the *first* iteration for a reason: a halted gateway has
    latched fail-closed, and running a cycle against it would be work done on
    state no longer trusted.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.scheduler.loop import _reconcile_to_fixpoint

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    scripted = _ScriptedReconciler(
        (ReconcileOutcome(halted=False, healed=0, halt_reason=None),)
    )
    halted = dataclasses.replace(deps, reconciler=scripted)
    halted.gateway.mark_halted()

    _reconcile_to_fixpoint(halted)

    assert scripted.calls == 0


def test_reconcile_to_fixpoint_stops_on_the_first_halting_cycle(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A cycle that latches the gateway halted is the last cycle run.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.scheduler.loop import _reconcile_to_fixpoint

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    scripted = _ScriptedReconciler(
        (ReconcileOutcome(halted=True, healed=0, halt_reason="foreign_open_order"),)
    )

    _reconcile_to_fixpoint(dataclasses.replace(deps, reconciler=scripted))

    assert scripted.calls == 1


def test_reconcile_to_fixpoint_stops_when_a_cycle_repeats_its_predecessor(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Two identical outcomes are a fixpoint, and the loop stops at the second.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.scheduler.loop import _reconcile_to_fixpoint

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    healed = ReconcileOutcome(halted=False, healed=1, halt_reason=None)
    scripted = _ScriptedReconciler((healed, healed))

    _reconcile_to_fixpoint(dataclasses.replace(deps, reconciler=scripted))

    assert scripted.calls == 2


def test_reconcile_to_fixpoint_is_bounded_when_no_fixpoint_is_reached(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A never-repeating reconciler is cut off at the cycle ceiling, not looped.

    Every scripted outcome differs from its predecessor, so neither termination
    arm can fire and only the bound can stop the loop. Pinning the exact count
    is what makes this a bound rather than "it returned eventually".

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
        tmp_path: The pytest scratch directory.
    """
    from windbreak.scheduler.loop import _reconcile_to_fixpoint

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    scripted = _ScriptedReconciler(
        tuple(
            ReconcileOutcome(halted=False, healed=healed, halt_reason=None)
            for healed in range(1, _RECONCILE_MAX_CYCLES + 3)
        )
    )

    _reconcile_to_fixpoint(dataclasses.replace(deps, reconciler=scripted))

    assert scripted.calls == _RECONCILE_MAX_CYCLES


def test_run_single_tick_reports_the_routed_fill_in_its_outcome(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A whole tick carries the routed fill out through `TickOutcome`.

    The stage tests above prove the routing chain; this proves the tick actually
    reports what it routed, all the way to `TickOutcome.filled_centis` and the
    `PositionsSnapshotRecorded` row an operator reads. The selector stage is
    *wrapped*, not replaced -- the real one still runs and its real reasons are
    still ledgered -- with only its emitted intents supplied, because whether the
    stock offline fixtures organically clear `net_edge_min` is an open economics
    question orthogonal to whether a cleared intent gets routed.

    Args:
        books_dir: The shared books-fixture directory.
        cassette_path: The empty recorded-cassette path.
        report_dir: Where weekly-report stubs would be written.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research-tools double.
        tmp_path: The pytest scratch directory.
        monkeypatch: Wraps the loop module's own `_select_stage`.
    """
    from windbreak.scheduler import loop as loop_module
    from windbreak.scheduler.loop import _select_stage as real_select_stage
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    routed = dataclasses.replace(
        deps,
        approval=_MintingApprovalSeam(
            deps.verification_key, expires_at=FIXED_NOW_EPOCH_S + 60
        ),
    )

    def _emitting_select_stage(*args, **kwargs):
        """Run the real select stage, then attach one emitted intent.

        Args:
            *args: Forwarded verbatim to the real stage.
            **kwargs: Forwarded verbatim to the real stage.

        Returns:
            The real stage's own decision, carrying one crossing intent.
        """
        return dataclasses.replace(
            real_select_stage(*args, **kwargs), intents=(_crossing_intent(),)
        )

    monkeypatch.setattr(loop_module, "_select_stage", _emitting_select_stage)

    outcome = run_single_tick(routed, beat=1)

    assert outcome.intent_count == 1
    assert outcome.filled_centis == _FIRST_LEG_FILL_CENTIS
    assert _held_centis(routed) == _FIRST_LEG_FILL_CENTIS
    snapshots = [
        json.loads(record.payload_json)["data"]
        for record in routed.store.read_all()
        if record.event_type == "PositionsSnapshotRecorded"
    ]
    assert snapshots, "expected a PositionsSnapshotRecorded row"
    held = [row for row in snapshots[-1]["positions"] if row["ticker"] == _TICKER]
    assert [row["quantity_centis"] for row in held] == [_FIRST_LEG_FILL_CENTIS]
    routed.store.verify_chain()
