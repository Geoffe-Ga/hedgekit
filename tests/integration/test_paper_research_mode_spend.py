"""A `RESEARCH` loop keeps buying forecasts (issue #526, review follow-up).

Issue #526 stopped the tick walking its universe in a mode that may not trade.
The first fix asked `Mode.may_trade()`, which is the wrong question by exactly
one mode, and this module is the guard against that answer coming back.

THE TWO QUESTIONS
-----------------

`may_trade()` answers "may an order be routed". `may_research()` answers "may
research money be spent". They agree on six of the seven modes -- `PAPER`,
`LIVE_MICRO` and `LIVE` do both; `PAUSED`, `HALT` and `KILLED` do neither --
and disagree on `RESEARCH`, which may **not** trade and **must** research.

That is not a quirk. SPEC S5.1's bottom rung exists to produce forecasts
without routing them, so that promotion is *earned*:
`windbreak/riskkernel/promotion.py::_research_to_paper_gate` opens with

    GateCriterion(criterion_id="research_min_forecasts",
                  evidence_field="forecast_count", ...)

A `RESEARCH` loop that bought no forecasts could never produce the evidence
that promotes it out of `RESEARCH`. Gating the walk on `may_trade()` therefore
does not merely waste the rung -- it strands anything that lands on it.

WHAT IS AND IS NOT REACHABLE TODAY, STATED PLAINLY
--------------------------------------------------

Two things a reader should not have to take on trust, because the argument for
this module is weaker if either is overstated:

* **Nothing in production populates `GateEvidence.forecast_count`.**
  `GateEvidence` is *constructed* nowhere under `windbreak/`;
  `RiskKernel.request_promotion(evidence)` receives it from a caller and has no
  production caller. So the promotion gate is not wired to a live evidence
  source at all, and the deadlock above is latent rather than active.
* **Nothing in the shipped loop demotes into `RESEARCH`.**
  `RiskKernel.fire_demotion_trigger` implements it and
  `resolve_demotion(PAPER, DRAWDOWN_BREACH)` really does answer `RESEARCH`
  (`windbreak/riskkernel/demotion.py`), but neither `windbreak/main.py` nor
  `windbreak/scheduler/loop.py` calls it, and `main.py`'s `_paper_activated`
  separately refuses to start the loop under a `research` *ceiling*.

So this is a latent defect, and latency is not a defence -- it is the same
argument this backlog files issues about (see #541). The regression it guards
against is invisible precisely because no shipped path exercises the mode, and
a behaviour change no test can see is the composition trap. The test below
closes that by driving the real `fire_demotion_trigger` API rather than setting
a mode by hand, so it exercises the path a wiring change would light up.

WHY THE ASSERTIONS ARE SHAPED THIS WAY
--------------------------------------

The forecast count is read back through
`windbreak.ledger.rebuild.forecasts_read_model` -- the shipped ledger read
model over `ForecastCreated` rows -- rather than counted here. That is the
ledger-derived forecast population any future wiring of `forecast_count` would
have to read, so the assertion is about *the gate's evidence* and survives a
refactor of the mode sets entirely.

The `deep_walk` fixture mints **no intent** (`SelectorDecisionRecorded` carries
`intent_count: 0` on `unprovable_exposure`), so "and is then vetoed at the
approval seam" is not observable here in either mode, before or after the fix.
That is stated rather than faked: what is asserted is that a `RESEARCH` tick
reaches the approval stage at all (its `ExchangeStatusObserved` row) and that
the kernel would refuse it (`may_trade()` is False), which is the observable
half. `tests/riskkernel/test_checks.py` owns the veto itself.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tests.integration.conftest import FIXED_NOW_EPOCH_S, ledger_path_for
from windbreak.forecast.budget import FULL_PIPELINE_RESEARCH_COST_MICROS
from windbreak.ledger.rebuild import forecasts_read_model
from windbreak.riskkernel.demotion import DemotionTrigger
from windbreak.riskkernel.modes import Mode

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig
    from windbreak.scheduler.loop import PaperTickDeps

#: How many beats the multi-beat assertion runs. More than one on purpose: a
#: gate that let the *first* research beat through and then stopped would
#: satisfy any single-beat assertion.
_BEATS = 3

#: The row whose payload legitimately differs between a `PAPER` tick and a
#: `RESEARCH` tick -- it carries the mode by definition. Every *other* row of a
#: `RESEARCH` tick must match its `PAPER` counterpart exactly, which is what
#: "byte-identical research behaviour" means and is asserted below.
_MODE_ROW = "ModeHeartbeat"

#: The row the demotion itself writes. Present only in the demoted run, so it
#: is excluded from the row-sequence comparison rather than making it fail for
#: the one reason that is not about research at all.
_DEMOTION_ROW = "DemotionTriggerFired"


def _fixed_clock() -> int:
    """Return the suite's fixed epoch second, so a tick is deterministic.

    Returns:
        The shared fixture epoch second.
    """
    return FIXED_NOW_EPOCH_S


def _build_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    research_tools_factory: Any,
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


def _rows(deps: PaperTickDeps) -> list[tuple[str, dict[str, Any]]]:
    """Return every ledger row as an `(event_type, payload-data)` pair.

    Args:
        deps: The wired bundle whose ledger is read.

    Returns:
        One pair per row, in ledger order.
    """
    return [
        (str(record.event_type), dict(json.loads(record.payload_json)["data"]))
        for record in deps.store.read_all()
    ]


def _research_micros(deps: PaperTickDeps) -> int:
    """Return every research micro charged on this bundle's ledger.

    Args:
        deps: The wired bundle whose ledger is read.

    Returns:
        The sum of every `ResearchSpendRecorded.cost_micros`.
    """
    return sum(
        int(data["cost_micros"])
        for event_type, data in _rows(deps)
        if event_type == "ResearchSpendRecorded"
    )


def _ledger_forecast_count(deps: PaperTickDeps) -> int:
    """Return how many forecasts the *ledger* says were produced.

    Read through `windbreak.ledger.rebuild.forecasts_read_model`, the shipped
    read model over `ForecastCreated` rows, rather than counted inline: this is
    the ledger-derived forecast population that any wiring of
    `GateEvidence.forecast_count` would have to read, so the assertion is about
    the promotion gate's evidence rather than about a mode set.

    Args:
        deps: The wired bundle whose ledger is read.

    Returns:
        The number of forecasts on the chain.
    """
    return len(forecasts_read_model(list(deps.store.read_all())))


def test_a_demoted_research_loop_keeps_producing_the_evidence_that_promotes_it(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory: Any,
    tmp_path: Path,
) -> None:
    """`RESEARCH` beats keep buying forecasts, beat after beat (issue #526).

    The regression guard. If the walk gate is ever folded back onto
    `Mode.may_trade()` -- which answers a different question and excludes
    `RESEARCH` -- the forecast count stops rising and this fails.

    The mode is reached through the **real** kernel API,
    `fire_demotion_trigger(DRAWDOWN_BREACH)`, whose `DEMOTE_ONE_MODE` action
    steps `PAPER` down one ladder rung to `RESEARCH`. Setting the mode by hand
    would prove nothing about a path the system can actually take.

    The count is read off the ledger through the shipped `forecasts_read_model`
    and asserted to rise *per beat*, so a gate that let the first research beat
    through and then closed cannot pass. The spend is asserted in exact micros
    beside it, because "forecasts exist" and "forecasts were paid for" are
    different claims and the defect broke both.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        cassette_path: The empty offline cassette.
        report_dir: The weekly-report output directory.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.
        tmp_path: The per-test scratch directory.
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
    destination = deps.kernel.fire_demotion_trigger(DemotionTrigger.DRAWDOWN_BREACH)

    counts: list[int] = []
    spends: list[int] = []
    for beat in range(1, _BEATS + 1):
        run_single_tick(deps, beat=beat)
        counts.append(_ledger_forecast_count(deps))
        spends.append(_research_micros(deps))

    assert destination is Mode.RESEARCH
    assert deps.kernel.mode is Mode.RESEARCH
    assert deps.kernel.mode.may_trade() is False
    assert counts == [1, 2, 3]
    assert spends == [
        FULL_PIPELINE_RESEARCH_COST_MICROS,
        2 * FULL_PIPELINE_RESEARCH_COST_MICROS,
        3 * FULL_PIPELINE_RESEARCH_COST_MICROS,
    ]
    assert [mode for _, mode in _heartbeat_modes(deps)] == ["RESEARCH"] * _BEATS
    deps.store.verify_chain()


def _heartbeat_modes(deps: PaperTickDeps) -> list[tuple[int, str]]:
    """Return `(beat, mode)` for every `ModeHeartbeat` row, in ledger order.

    Args:
        deps: The wired bundle whose ledger is read.

    Returns:
        One pair per heartbeat row.
    """
    return [
        (int(data["beat"]), str(data["mode"]))
        for event_type, data in _rows(deps)
        if event_type == _MODE_ROW
    ]


def test_a_research_beat_is_byte_identical_to_a_paper_beat_but_for_the_mode(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    paper_config: WindbreakConfig,
    research_tools_factory: Any,
    tmp_path: Path,
) -> None:
    """A `RESEARCH` tick ledgers exactly what a `PAPER` tick does (issue #526).

    The "unchanged" half, asserted as **full equality on the payloads** rather
    than as counts or as the absence of an exception. Two bundles over the same
    fixture and the same fixed clock; one is demoted to `RESEARCH` first. Every
    row of the research tick must equal the paper tick's row -- same types, same
    order, same payload dicts -- except the two that cannot match by
    construction: the `ModeHeartbeat` (which carries the mode) and the
    `DemotionTriggerFired` row the demotion itself wrote.

    That comparison is what makes "byte-identical research behaviour" checkable.
    A gate that excluded `RESEARCH` fails it by omission: the research run loses
    its `MarketSnapshotRecorded`, `ResearchSpendRecorded`, `ForecastCreated`,
    `SelectorDecisionRecorded` and `ExchangeStatusObserved` rows, five row types
    at once.

    The coincidence trap is closed in the last two assertions. The comparison
    above would be satisfied just as well by two runs that both produced
    *nothing*, so the shared sequence is asserted non-empty and asserted to
    contain the money row; and the one payload that must differ is asserted to
    actually differ, so a fixture in which both modes answer identically cannot
    read as success.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        cassette_path: The empty offline cassette.
        report_dir: The weekly-report output directory.
        paper_config: The PAPER-ceilinged configuration.
        research_tools_factory: Builds the offline research tools double.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.scheduler.loop import run_single_tick

    bundles = {}
    for label in ("paper", "research"):
        deps = _build_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            ledger_path=ledger_path_for(tmp_path, f"{label}.db"),
            report_dir=report_dir,
            config=paper_config,
            research_tools_factory=research_tools_factory,
        )
        if label == "research":
            deps.kernel.fire_demotion_trigger(DemotionTrigger.DRAWDOWN_BREACH)
        run_single_tick(deps, beat=1)
        bundles[label] = deps

    def _comparable(deps: PaperTickDeps) -> list[tuple[str, dict[str, Any]]]:
        """Project the rows that must match across the two modes.

        Args:
            deps: The bundle whose ledger is projected.

        Returns:
            Every row but the mode heartbeat and the demotion record.
        """
        return [row for row in _rows(deps) if row[0] not in {_MODE_ROW, _DEMOTION_ROW}]

    paper_rows = _comparable(bundles["paper"])
    research_rows = _comparable(bundles["research"])

    assert research_rows == paper_rows
    assert paper_rows, "both runs ledgered nothing, so the equality is vacuous"
    assert (
        "ResearchSpendRecorded",
        {
            "cost_micros": FULL_PIPELINE_RESEARCH_COST_MICROS,
            "market_ticker": "MKT-DEEP",
            "utc_day": "2024-12-24",
        },
    ) in research_rows
    assert "ExchangeStatusObserved" in {event_type for event_type, _ in research_rows}
    assert _heartbeat_modes(bundles["research"]) != _heartbeat_modes(bundles["paper"])
    assert _heartbeat_modes(bundles["research"]) == [(1, "RESEARCH")]
    assert _heartbeat_modes(bundles["paper"]) == [(1, "PAPER")]
    for deps in bundles.values():
        deps.store.verify_chain()
        deps.store.close()
