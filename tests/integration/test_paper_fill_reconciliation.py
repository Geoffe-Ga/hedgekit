"""Failing-first tests for a PAPER loop surviving its own fills (issue #365).

Before ledgered fill accounting, the PAPER loop's reconciliation baseline froze
at process start and nothing could advance it: the moment a real fill moved cash
and positions, the next cycle graded ``BREACH``, the kernel HALTed per issue
#32, and only a process restart cleared it. An always-on PAPER deployment could
not survive its own first fill.

These tests pin both halves of the fix, because either alone would be a
regression:

* a loop whose venue moved *exactly as the ledger booked* keeps ticking; and
* a loop whose venue moved by something the books cannot explain still halts.

The second is the guard against "fixing" the first by relaxing the check into
the issue #352 tautology. It is written as a movement the loop never booked --
cash leaving the account with no execution behind it -- which is precisely the
unattributed venue movement reconciliation exists to catch.

Issue #390 adds the same pair for the *open-order* dimension. A limit that only
partly crosses leaves a remainder resting, and before #390 no ledgered fact
could name that order: the remainder read as unexplained venue movement and
halted the loop the moment it routed anything other than an outright marketable
order. Those two tests assert on the cycle's `open_order_ok` flag specifically
rather than on the whole outcome, because when they were written the same tick
still breached on *cash*.

Issue #423 closes that last dimension. A resting order also withholds
collateral from `available`, and no entry accounted for the reservation --
1_200_000 micros of `cash_drift` on this very fixture. The arrival entry now
carries the venue's own reservation figure, so
`test_a_resting_order_reconciles_on_every_dimension_including_cash` can assert
what #390 could not: the whole cycle, every dimension, clean. Its non-vacuity
partner measures the withholding directly off the venue, so a reconciliation
that passed merely because nothing was ever withheld cannot pass here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests.integration.conftest import ledger_path_for
from tests.riskkernel.conftest import DEFAULT_NOW_EPOCH_S
from windbreak.numeric.types import ContractCentis, MoneyMicros, PricePips
from windbreak.riskkernel.modes import Mode

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig
    from windbreak.scheduler.loop import PaperTickDeps

#: The single ticker in the shared `deep_walk` books fixture.
_TICKER = "MKT-DEEP"

#: A limit that crosses every level of the fixture's ask book, so the order
#: fills outright and leaves no resting remainder. A remainder would rest, and
#: the open-order dimension -- which fill accounting deliberately does not
#: advance, since venue order ids are never ledgered -- would breach for a
#: reason that has nothing to do with what these tests pin.
_CROSSING_LIMIT = PricePips(9900)

#: Small enough to stay under the taker walk's participation cap, so the whole
#: order really does fill.
_SMALL_SIZE = ContractCentis(50)

#: A limit at the fixture's best ask, so the order crosses but the participation
#: cap stops the walk part-way and the rest has to rest (issue #390).
_RESTING_LIMIT = PricePips(4600)

#: Large enough that the participation-capped taker walk cannot consume it all,
#: so a genuine remainder is left resting at the venue.
_RESTING_SIZE = ContractCentis(300)

#: An instant no simulated fill can precede, for reading the venue's whole
#: execution log (mirrors `windbreak.scheduler.fill_accounting._EPOCH_FLOOR`).
_EPOCH_FLOOR = datetime(1970, 1, 1, tzinfo=UTC)


def _fixed_clock() -> int:
    """Return the one epoch second every deps-builder call here agrees on.

    Returns:
        `DEFAULT_NOW_EPOCH_S`, so no freshness check can interfere with the
        reconciliation dimensions under test.
    """
    return DEFAULT_NOW_EPOCH_S


def _build_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    research_tools_factory,
) -> PaperTickDeps:
    """Build one `PaperTickDeps` over the shared offline fixtures.

    Args:
        books_dir: The `deep_walk` books-fixture directory.
        cassette_path: The (empty) recorded-cassette path.
        ledger_path: Where the tick's `SqliteLedgerStore` is created.
        report_dir: Where weekly-report stubs would be written.
        config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.

    Returns:
        A fully wired `PaperTickDeps`.
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


def _fill_the_account(deps: PaperTickDeps) -> None:
    """Execute one outright fill on the loop's own exchange.

    Placed directly on the exchange rather than routed through the gateway
    because a PAPER tick does not currently mint a token (the concentration and
    participation checks still veto -- see
    `tests/integration/test_paper_verification.py`). What is under test is the
    *reconciliation* consequence of a fill, not the path that produced it: the
    venue moves, the bookkeeper books it from the venue's execution report, and
    the next cycle has to reconcile the two.

    Args:
        deps: The tick's dependency bundle.
    """
    from windbreak.connector.paper import PaperOrderIntent

    placement = deps.exchange.place_order(
        PaperOrderIntent(
            ticker=_TICKER, side="yes", price=_CROSSING_LIMIT, quantity=_SMALL_SIZE
        ),
        None,
    )
    assert placement.fills, "the fixture book must fill this order outright"
    assert placement.resting_order is None, "no remainder may rest"


def _partially_fill_the_account(deps: PaperTickDeps) -> str:
    """Execute one *partial* fill, leaving a remainder resting.

    The scenario issue #390 exists for. The limit crosses the fixture's best
    ask, the taker walk's participation cap stops it part-way, and the unfilled
    remainder rests as a brand-new venue order. Placed directly on the exchange
    for the same reason `_fill_the_account` is: what is under test is the
    reconciliation consequence, not the path that produced it.

    Args:
        deps: The tick's dependency bundle.

    Returns:
        The venue order id of the resting remainder.
    """
    from windbreak.connector.paper import PaperOrderIntent

    placement = deps.exchange.place_order(
        PaperOrderIntent(
            ticker=_TICKER, side="yes", price=_RESTING_LIMIT, quantity=_RESTING_SIZE
        ),
        None,
    )
    assert placement.fills, "the order must fill part-way"
    assert placement.resting_order is not None, "a remainder must rest"
    return placement.resting_order.id


def _open_order_verdicts(deps: PaperTickDeps) -> list[bool]:
    """Return each recorded verification cycle's open-order verdict, in order.

    The open-order dimension is graded independently of the other two and
    reported on its own payload flag, so a test can pin exactly the dimension
    issue #390 moved without riding on whichever other dimension happens to
    agree.

    Args:
        deps: The tick's dependency bundle.

    Returns:
        The `open_order_ok` flag of every verification event on the ledger.
    """
    import json

    verdicts: list[bool] = []
    for record in deps.store.read_all():
        if not record.event_type.startswith("Verification"):
            continue
        data = json.loads(record.payload_json)["data"]
        if "open_order_ok" in data:
            verdicts.append(bool(data["open_order_ok"]))
    return verdicts


def _cash_drifts(deps: PaperTickDeps) -> list[int]:
    """Return each recorded verification cycle's cash drift, in order.

    Args:
        deps: The tick's dependency bundle.

    Returns:
        The `cash_drift` micros of every verification event on the ledger.
    """
    import json

    drifts: list[int] = []
    for record in deps.store.read_all():
        if not record.event_type.startswith("Verification"):
            continue
        data = json.loads(record.payload_json)["data"]
        if "cash_drift" in data:
            drifts.append(int(data["cash_drift"]))
    return drifts


def test_a_partial_fill_leaves_the_open_order_dimension_reconciled(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """The whole of issue #390: a remainder that rests is explained evidence.

    Before it, venue order ids were never ledgered at all, so no booking could
    name the order the venue had come to rest. The surviving remainder read as
    unexplained open-order movement and breached -- latent only while the loop
    routed outright marketable orders, and live the moment it did not.

    Only the open-order dimension is asserted, because only it is what #390
    moved: when this test was written the same tick still breached on *cash*,
    over the collateral the resting remainder withholds from `available`. Issue
    #423 booked that reservation too, and
    `test_a_resting_order_reconciles_on_every_dimension_including_cash` is the
    whole-outcome assertion this one deliberately could not make. Both are kept:
    this one still fails if the open-order dimension alone regresses, which a
    whole-outcome assertion would blur into whichever dimension broke.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)

    _partially_fill_the_account(deps)
    run_single_tick(deps, beat=2)

    event_types = [record.event_type for record in deps.store.read_all()]
    assert "RestingOrderAccounted" in event_types
    assert "FillAccounted" in event_types
    assert _open_order_verdicts(deps) == [True, True]
    deps.store.verify_chain()


def test_a_resting_order_reconciles_on_every_dimension_including_cash(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Issue #423 through the real composition root: no dimension left over.

    #422 could only assert `open_order_ok`, because the same tick still breached
    on *cash*: the resting remainder withholds collateral from `available`
    (`OrderCollateralInAvailable.DEDUCTED_FROM_AVAILABLE`) and no ledgered entry
    accounted for it -- 1_200_000 micros of `cash_drift` on this exact fixture.
    The arrival entry now books that reservation, so the whole cycle grades
    clean and the loop keeps ticking.

    Asserted through `run_single_tick` rather than against the fold in
    isolation: the bookkeeper, the ledger row, the feed, and the expectation all
    have to agree, and a seam that is green on its own while the composition
    never reaches it is the failure mode this suite exists to catch.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)

    _partially_fill_the_account(deps)
    outcome = run_single_tick(deps, beat=2)

    event_types = [record.event_type for record in deps.store.read_all()]
    assert "RestingOrderAccounted" in event_types
    assert _cash_drifts(deps) == [0, 0]
    assert "VerificationMismatch" not in event_types
    assert "VerificationMismatchHalt" not in event_types
    assert deps.kernel.mode is not Mode.HALT
    assert outcome.kernel_halted is False
    deps.store.verify_chain()


def test_the_venue_really_did_withhold_the_collateral_that_now_reconciles(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Non-vacuity for the fixture itself: the reservation is not zero.

    A reconciliation that passes because nothing was ever withheld would pin
    nothing at all. This measures the withholding directly off the venue -- the
    1_200_000 micros #422 reported as `cash_drift` -- so the clean cycle above
    is known to be absorbing a real movement rather than an absent one.
    """
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    before = deps.exchange.get_balances().available.value

    _partially_fill_the_account(deps)

    (rested,) = deps.exchange.get_rested_orders()
    reserved = deps.exchange.resting_collateral_micros(rested).value
    spent = sum(
        deps.exchange.fill_cash_micros(fill).value
        for fill in deps.exchange.get_fills(_EPOCH_FLOOR)
    )
    assert reserved == 1_200_000
    assert before - deps.exchange.get_balances().available.value == spent + reserved


def test_a_resting_order_that_left_the_venue_unexplained_still_breaches(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Non-vacuity for the open-order dimension: an unexplained retirement fails.

    The order rests and is booked, so the expectation legitimately carries its
    id and the previous test's cycle graded that dimension clean. Then it leaves
    the venue with no fill behind it -- a cancel, and nothing in the books
    explains the disappearance. Were the expectation re-derived from the venue's
    own open-order view each cycle the two would agree by construction and this
    could never fail, which is the issue #352 tautology sixteen closed issues in
    this backlog exist to keep out. Because it advances only from booked arrival
    and fill entries, the unexplained retirement survives and the dimension
    breaches.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)
    order_id = _partially_fill_the_account(deps)
    run_single_tick(deps, beat=2)

    deps.exchange.cancel_order(order_id)
    run_single_tick(deps, beat=3)

    assert _open_order_verdicts(deps) == [True, True, False]
    assert deps.kernel.mode is Mode.HALT


def test_a_paper_loop_keeps_ticking_after_a_fill(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A fill the ledger booked is reconciled, not halted on.

    This is the whole of issue #365. Before it, the second tick here graded
    BREACH and HALTed the kernel, and only a restart cleared it.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)

    _fill_the_account(deps)
    outcome = run_single_tick(deps, beat=2)

    event_types = [record.event_type for record in deps.store.read_all()]
    assert "FillAccounted" in event_types
    assert "VerificationMismatch" not in event_types
    assert "VerificationMismatchHalt" not in event_types
    assert deps.kernel.mode is not Mode.HALT
    assert outcome.kernel_halted is False
    deps.store.verify_chain()


def test_a_paper_loop_keeps_ticking_across_several_fills(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """The advance compounds: three fills over three ticks, still no halt.

    A one-fill test would still pass if the expectation advanced once and then
    re-froze; this pins that every booked execution keeps advancing it.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)

    for beat in (2, 3, 4):
        _fill_the_account(deps)
        run_single_tick(deps, beat=beat)

    event_types = [record.event_type for record in deps.store.read_all()]
    assert event_types.count("FillAccounted") == 3
    assert "VerificationMismatchHalt" not in event_types
    assert deps.kernel.mode is not Mode.HALT


def test_a_venue_movement_nobody_booked_still_halts_the_loop(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Cash that left the account with no execution behind it still HALTs.

    The guard against relaxing the check into the issue #352 tautology: fill
    accounting absorbs only the movement the ledger can *explain*. An
    unexplained one is exactly what verification exists to catch, and the
    required response is a halt, never a veto.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)

    opening = deps.exchange.balances
    deps.exchange.balances = type(opening)(
        total=MoneyMicros(opening.total.value - 7_000_000),
        available=MoneyMicros(opening.available.value - 7_000_000),
        fetched_at=opening.fetched_at,
    )
    outcome = run_single_tick(deps, beat=2)

    event_types = [record.event_type for record in deps.store.read_all()]
    assert "FillAccounted" not in event_types
    assert "VerificationMismatch" in event_types
    assert "VerificationMismatchHalt" in event_types
    assert deps.kernel.mode is Mode.HALT
    assert outcome.kernel_halted is True


def test_a_fill_the_books_understate_still_halts_the_loop(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """The sharpest form of the guard: the ledger books a fill, and the venue
    moved *further* than that booking explains.

    Were the expectation re-derived from the connector each cycle the two would
    agree by construction and this could never fail. Because it advances only by
    the booked delta, the unexplained remainder survives and halts.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=paper_config,
        research_tools_factory=research_tools_factory,
    )
    run_single_tick(deps, beat=1)

    _fill_the_account(deps)
    opening = deps.exchange.balances
    deps.exchange.balances = type(opening)(
        total=MoneyMicros(opening.total.value - 3_000_000),
        available=MoneyMicros(opening.available.value - 3_000_000),
        fetched_at=opening.fetched_at,
    )
    outcome = run_single_tick(deps, beat=2)

    event_types = [record.event_type for record in deps.store.read_all()]
    assert "FillAccounted" in event_types
    assert "VerificationMismatchHalt" in event_types
    assert outcome.kernel_halted is True
