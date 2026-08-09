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
"""

from __future__ import annotations

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
