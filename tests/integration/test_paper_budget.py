"""Research-budget enforcement in the always-on PAPER loop (issue #339, RED).

`windbreak.scheduler.loop._forecast_stage` calls `run_pipeline(...)` with six
arguments and no `budget=` (loop.py:812-819), and `run_pipeline`'s `budget`
parameter defaults to `None` -- documented as "a strict no-op"
(pipeline.py:1779). So the always-on loop enforces neither the per-forecast nor
the per-day research ceiling. This is masked today only because the loop's
transport is a `ReplayCassette` and its research tools are offline doubles; the
moment live providers are wired in, an unattended 24/7 loop calls paid LLM and
research APIs with no spend ceiling at all.

Every test here therefore fails before the fix, for one of two reasons:

* `AttributeError: 'PaperTickDeps' object has no attribute 'budget'` -- the
  bundle carries no budget (loop.py:515-526 lists twelve fields, none of them
  a budget); or
* an `AssertionError` on a `ForecastCreated` count, because an unbudgeted tick
  runs the pipeline unconditionally no matter what the config says.

The one exception is the byte-for-byte test, whose criterion is an assertion of
the *absence* of change and so cannot be made RED on its own terms; its RED is
honestly borrowed from a leading wiring assertion. See that test's docstring.

Design notes:

* Budget state is deliberately observed through the *ledger*, not through
  `ResearchBudget` internals: `_spend_by_day` is private (budget.py:334) and
  `max_pages` is the class's only public read surface (budget.py:336).
* `_CountingSearchTransport` makes "zero research ran after the halt" directly
  observable rather than inferred from the absence of a ledger row.
* Deferred `windbreak.scheduler.loop` imports inside each test body follow the
  idiom this suite already uses (`test_paper_loop.py:143`), so the module keeps
  collecting cleanly regardless of that package's state.
"""

from __future__ import annotations

import dataclasses
import inspect
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING

import pytest

from tests.integration.conftest import (
    FIXED_NOW_EPOCH_S,
    FIXTURE_SCREENER_CONFIG,
    NullSearchTransport,
    ledger_path_for,
    read_event_type_payload_pairs,
)
from windbreak.config.schema import (
    CapitalConfig,
    ForecastBudget,
    ForecastConfig,
    RiskConfig,
    WindbreakConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

#: The fixed per-forecast research charge the offline PAPER tick always incurs.
#: Mirrored locally rather than imported: `_RESEARCH_COST_MICROS` is private to
#: `windbreak.forecast.pipeline` (pipeline.py:103), and this suite's house rule
#: is to pin the expected value independently so a change to the constant
#: surfaces as a failing assertion rather than silently re-deriving itself.
_EXPECTED_RESEARCH_COST_MICROS = 3_000_000

#: The UTC calendar day `FIXED_NOW_EPOCH_S` (1_735_000_000) falls on. Verified:
#: `datetime.fromtimestamp(1_735_000_000, UTC).date().isoformat()`.
_FIXED_UTC_DAY = "2024-12-24"

#: A per-day ceiling admitting exactly two charged forecasts, so a third tick
#: on the same UTC day is the first to breach.
_TWO_FORECAST_DAY_CEILING_MICROS = 2 * _EXPECTED_RESEARCH_COST_MICROS

#: The `halt_kind` discriminator for a per-UTC-day exhaustion halt.
_HALT_KIND_PER_DAY = "per_day"

#: The `halt_kind` discriminator for a single-forecast ceiling breach.
_HALT_KIND_PER_FORECAST = "per_forecast"

#: The ledger event a fail-closed research halt appends.
_HALT_EVENT_TYPE = "ResearchBudgetHalted"

#: The sole market the shared `deep_walk` books fixture offers, and so the one
#: market every tick here screens in. Named locally because since issue #345 the
#: tick screens a universe rather than carrying a ticker of its own.
_TICKER = "MKT-DEEP"


class _CountingSearchTransport(NullSearchTransport):
    """A `NullSearchTransport` that counts how many searches were attempted.

    Finding nothing is inherited unchanged; the only addition is the counter,
    which makes "no research ran after the halt" a direct observation rather
    than something inferred from a missing ledger row.

    Attributes:
        search_count: How many times `search` has been called.
    """

    def __init__(self) -> None:
        """Initialize with a zeroed search counter."""
        self.search_count = 0

    def search(self, query: str) -> tuple[str, ...]:
        """Count the attempt, then find nothing exactly as the base class does.

        Args:
            query: The (unused) subquestion text.

        Returns:
            An empty tuple, always.
        """
        self.search_count += 1
        return super().search(query)


def _fixed_clock() -> int:
    """Return the fixed, non-advancing epoch second used across this module.

    Returns:
        `FIXED_NOW_EPOCH_S`, so every tick buckets to `_FIXED_UTC_DAY`.
    """
    return FIXED_NOW_EPOCH_S


def _budgeted_config(**ceilings: int) -> WindbreakConfig:
    """Build a PAPER-ceilinged config whose research budget is set explicitly.

    Carries `FIXTURE_SCREENER_CONFIG` for the same reason `paper_config` does
    (see its note in `tests/integration/conftest.py`): since issue #345 a tick
    forecasts only markets that pass the screen, and the `deep_walk` fixture
    fails the production depth and horizon defaults -- so a config built here
    without it would screen `MKT-DEEP` out and every budget scenario below would
    be measuring the screener rather than the research ceilings.

    Args:
        **ceilings: `ForecastBudget` field overrides (`per_forecast_micros`,
            `per_day_micros`, `max_pages`); unset fields keep SPEC §16 defaults.

    Returns:
        A `WindbreakConfig` identical to the suite's `paper_config` except for
        the named research-budget ceilings.
    """
    return WindbreakConfig(
        mode_ceiling="paper",
        capital=CapitalConfig(floor_micros=0),
        risk=RiskConfig(),
        forecast=ForecastConfig(budget=ForecastBudget(**ceilings)),
        screener=FIXTURE_SCREENER_CONFIG,
    )


def _build_deps(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
    config: WindbreakConfig,
    research_tools: object,
    clock: object = _fixed_clock,
) -> object:
    """Build one `PaperTickDeps` over the shared offline fixtures.

    Mirrors `test_paper_loop.py::_build_deps`, but takes an already-built
    `ResearchTools` (so a test can hold onto the counting transport behind it)
    and an overridable clock (so the day-rollover test can advance time).

    Args:
        books_dir: The `deep_walk` books-fixture directory.
        cassette_path: The (empty) recorded-cassette path.
        ledger_path: Where the tick's `SqliteLedgerStore` is created.
        report_dir: Where weekly-report stubs would be written.
        config: The PAPER-ceilinged configuration.
        research_tools: The offline research tools the forecast stage uses.
        clock: The zero-arg epoch-second clock injected for determinism.

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
        research_tools=research_tools,
        clock=clock,
    )


def _offline_research_tools(tmp_path: Path, transport: object) -> object:
    """Build offline `ResearchTools` over a caller-supplied transport double.

    Args:
        tmp_path: The pytest scratch directory the research cache lives under.
        transport: The search/fetch transport double to wire in.

    Returns:
        A `ResearchTools` bundle that never touches the network.
    """
    from windbreak.forecast.sandbox import build_research_tools

    return build_research_tools(
        allowed_hosts=frozenset({"research.local"}),
        cache_dir=tmp_path / "research-cache",
        search_transport=transport,
        fetch_transport=transport,
    )


def _event_types(deps: object) -> list[str]:
    """Return the ledger's event-type sequence, in append order.

    Args:
        deps: The `PaperTickDeps` whose store is read.

    Returns:
        One event-type string per ledger record.
    """
    pairs = read_event_type_payload_pairs(deps.store.read_all())
    return [event_type for event_type, _ in pairs]


def _payloads_of(deps: object, event_type: str) -> list[dict]:
    """Return every payload of a given event type, in append order.

    Args:
        deps: The `PaperTickDeps` whose store is read.
        event_type: The event type to filter for.

    Returns:
        The matching payload dicts.
    """
    pairs = read_event_type_payload_pairs(deps.store.read_all())
    return [payload for recorded, payload in pairs if recorded == event_type]


def test_build_paper_deps_builds_a_research_budget_from_config(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """`build_paper_deps` puts a config-derived `ResearchBudget` on the bundle.

    `max_pages` is asserted because it is `ResearchBudget`'s only public read
    surface (budget.py:336); choosing `7` -- which is neither the class default
    (20) nor the schema default (20) -- proves the *config* values reached the
    instance rather than the constructor defaults.

    The signature assertion is the structural fail-closed guard: with no
    `budget` parameter on `build_paper_deps`, there is no injection door through
    which an unlimited or `None` budget could ever arrive.
    """
    from windbreak.forecast.budget import ResearchBudget
    from windbreak.scheduler.loop import build_paper_deps

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_budgeted_config(
            per_forecast_micros=_EXPECTED_RESEARCH_COST_MICROS,
            per_day_micros=_TWO_FORECAST_DAY_CEILING_MICROS,
            max_pages=7,
        ),
        research_tools=research_tools_factory(),
    )

    assert isinstance(deps.budget, ResearchBudget)
    assert deps.budget.max_pages == 7
    assert "budget" not in inspect.signature(build_paper_deps).parameters


def test_forecast_stage_passes_the_deps_budget_to_run_pipeline(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A zero per-day ceiling suppresses the forecast, proving pass-through.

    Mock-free: a `per_day_micros=0` config is legal (`_require_non_negative`
    accepts zero, budget.py:327) and makes `ensure_day_open` raise on its very
    first call because `0 >= 0` (budget.py:359). The *only* way that config can
    suppress the tick's forecast is if the budget object genuinely reached
    `run_pipeline`'s `budget=` parameter.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_budgeted_config(per_day_micros=0),
        research_tools=research_tools_factory(),
    )

    outcome = run_single_tick(deps, beat=1)

    assert "ForecastCreated" not in _event_types(deps)
    assert "SelectorDecisionRecorded" not in _event_types(deps)
    assert _payloads_of(deps, _HALT_EVENT_TYPE) == [
        {
            "market_ticker": "",
            "halt_kind": _HALT_KIND_PER_DAY,
            "utc_day": _FIXED_UTC_DAY,
            "spent_micros": 0,
            "budget_micros": 0,
        }
    ]
    assert outcome.research_halted is True
    assert outcome.forecast_ids == ()
    deps.store.verify_chain()


def test_ticking_past_the_per_day_ceiling_halts_research_and_ledgers_the_overrun(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A third same-day tick halts fail-closed, ledgers it, and stays alive.

    The per-day ceiling admits exactly two charged forecasts, so tick 3 is the
    first to breach. That `run_single_tick` is *not* wrapped in `pytest.raises`
    is itself the fail-closed-not-crash assertion: today nothing catches a
    `DailyBudgetExhaustedError` (`run_single_tick` has no try/except and
    `run_loop` calls `on_beat(seq)` bare), so an uncaught raise would kill the
    entire always-on process rather than halting one tick's research.
    """
    from windbreak.scheduler.loop import run_single_tick

    transport = _CountingSearchTransport()
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_budgeted_config(per_day_micros=_TWO_FORECAST_DAY_CEILING_MICROS),
        research_tools=_offline_research_tools(tmp_path, transport),
    )

    outcomes = [run_single_tick(deps, beat=beat) for beat in (1, 2)]
    searches_after_two_ticks = transport.search_count
    outcomes.append(run_single_tick(deps, beat=3))

    assert len(_payloads_of(deps, "ForecastCreated")) == 2
    assert transport.search_count == searches_after_two_ticks
    assert _payloads_of(deps, _HALT_EVENT_TYPE) == [
        {
            "market_ticker": "",
            "halt_kind": _HALT_KIND_PER_DAY,
            "utc_day": _FIXED_UTC_DAY,
            "spent_micros": _TWO_FORECAST_DAY_CEILING_MICROS,
            "budget_micros": _TWO_FORECAST_DAY_CEILING_MICROS,
        }
    ]
    assert len(_payloads_of(deps, "EquitySampled")) == 3
    assert len(_payloads_of(deps, "MarketSnapshotRecorded")) == 3
    heartbeats = _payloads_of(deps, "ModeHeartbeat")
    assert [payload["beat"] for payload in heartbeats] == [1, 2, 3]
    assert outcomes[2].research_halted is True
    assert outcomes[2].intent_count == 0
    assert outcomes[2].filled_centis == 0
    deps.store.verify_chain()


def test_per_day_spend_accumulates_across_ticks_and_only_resets_at_utc_midnight(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Day spend survives ticks and rolls over only at UTC midnight.

    Phase 1 is the discriminating half. It fails not only against today's
    unbudgeted code but against the wrong-but-plausible fix of constructing a
    fresh `ResearchBudget` inside `_forecast_stage`/`run_single_tick`: that
    variant would present an empty day bucket every tick and so produce three
    forecasts instead of two.

    Phase 2 advances the injected clock to the next UTC day and asserts the
    bucket rolls with no reset code, proving it is keyed by the tick's instant
    rather than by a counter someone has to remember to clear.
    """
    from windbreak.scheduler.loop import run_single_tick

    epoch = [FIXED_NOW_EPOCH_S]
    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_budgeted_config(per_day_micros=_TWO_FORECAST_DAY_CEILING_MICROS),
        research_tools=research_tools_factory(),
        clock=lambda: epoch[0],
    )

    outcomes = [run_single_tick(deps, beat=beat) for beat in (1, 2, 3)]

    assert len(_payloads_of(deps, "ForecastCreated")) == 2
    assert [outcome.research_halted for outcome in outcomes] == [False, False, True]

    next_day = datetime.fromtimestamp(FIXED_NOW_EPOCH_S, UTC).date() + timedelta(days=1)
    epoch[0] = int(datetime.combine(next_day, time.min, tzinfo=UTC).timestamp())
    rolled_over = run_single_tick(deps, beat=4)

    assert rolled_over.research_halted is False
    assert len(_payloads_of(deps, "ForecastCreated")) == 3
    assert len(_payloads_of(deps, _HALT_EVENT_TYPE)) == 1
    assert dataclasses.replace(deps, approval=deps.approval).budget is deps.budget
    deps.store.verify_chain()


def test_unexhausted_budget_leaves_the_tick_ledger_payloads_byte_identical(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """Three wildly different unexhausted budgets produce identical ledgers.

    HONEST CAVEAT: this criterion asserts the *absence* of change, so the
    payload equalities already hold before the fix (the config is simply
    ignored). Its RED is supplied deliberately by the leading
    `isinstance(legs[0].budget, ResearchBudget)` wiring assertion, which fails
    with `AttributeError` for exactly the intended reason. Post-fix the
    comparison becomes a genuine regression guard, proving that threading a
    budget through -- which changes stage 5's `max_pages` from `None` to the
    config value -- changed no ledgered decision content.

    Leg (c) additionally pins two boundaries: `max_pages=3` exactly matches the
    three subquestion prefixes (inert offline), and a charge exactly equal to
    the per-forecast ceiling passes, because that ceiling is inclusive and only
    a strictly greater cost breaches (budget.py:395).
    """
    from windbreak.forecast.budget import ResearchBudget
    from windbreak.scheduler.loop import run_single_tick

    configs = (
        _budgeted_config(),
        _budgeted_config(
            per_forecast_micros=1_000_000_000,
            per_day_micros=1_000_000_000,
            max_pages=20,
        ),
        _budgeted_config(
            per_forecast_micros=_EXPECTED_RESEARCH_COST_MICROS,
            per_day_micros=_EXPECTED_RESEARCH_COST_MICROS,
            max_pages=3,
        ),
    )
    legs = [
        _build_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            ledger_path=ledger_path_for(tmp_path, f"ledger_{name}.db"),
            report_dir=report_dir,
            config=config,
            research_tools=research_tools_factory(),
        )
        for name, config in zip(("a", "b", "c"), configs, strict=True)
    ]

    assert isinstance(legs[0].budget, ResearchBudget)

    for deps in legs:
        run_single_tick(deps, beat=1)

    projections = [
        read_event_type_payload_pairs(deps.store.read_all()) for deps in legs
    ]
    assert projections[0] == projections[1] == projections[2]
    assert _event_types(legs[0]) == [
        "RecoveryCompleted",
        # Issue #345: the tick opens by screening the venue's market universe,
        # ledgering one verdict per market examined -- here exactly one, since
        # `deep_walk` offers the single market `MKT-DEEP`. It comes first
        # because screening is free and must precede every paid stage.
        "ScreenDecisionRecorded",
        "PipelineHeartbeatRecorded",
        # Issue #353: the tick now runs one read-only verification cycle before
        # it decides anything, and that cycle records exactly one event -- here
        # a clean pass, since nothing has traded yet. It runs ahead of the whole
        # universe (issue #345), so the snapshot of the one screened market now
        # follows it rather than opening the tick.
        "VerificationPassed",
        "MarketSnapshotRecorded",
        "ForecastCreated",
        "SelectorDecisionRecorded",
        "ExchangeStatusObserved",
        "ModeHeartbeat",
        "EquitySampled",
        "PositionsSnapshotRecorded",
    ]
    forecast_payload = _payloads_of(legs[0], "ForecastCreated")[0]
    assert forecast_payload["research_cost_micros"] == _EXPECTED_RESEARCH_COST_MICROS
    assert forecast_payload["abstention_reason"] == "no_verified_citations"
    assert forecast_payload["eligible_for_live"] is False
    for deps in legs:
        assert _payloads_of(deps, _HALT_EVENT_TYPE) == []
        deps.store.verify_chain()


def test_a_per_forecast_ceiling_breach_halts_the_tick_and_ledgers_per_forecast(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A per-forecast breach halts the tick and ledgers the per-forecast kind.

    Uses the repo's established ceiling trick: set the ceiling one micro below
    the fixed charge, so the offline tick's 3,000,000-micro cost strictly
    exceeds it. Unlike the per-day halt, `market_ticker` is populated here,
    because `BUDGET_FORECAST_EXCEEDED`'s payload carries it (budget.py:396-401).

    Without this test the `PerForecastBudgetExceededError` arm of the except
    tuple and the per-forecast branch of the ledger writer are unreachable from
    the other integration tests: the offline fixture providers report
    `cost_micros == 0`, so a stock-config tick charges only the fixed 3,000,000
    against a 6,060,000 ceiling and cannot breach it from the cost side. The
    ceiling knob is what makes the guard observable here; the cost side of the
    same boundary is pinned in `tests/forecast/test_budget_headroom.py`, which
    charges a worst-case run against the *stock* ceiling from both sides
    (issue #394).
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_budgeted_config(per_forecast_micros=_EXPECTED_RESEARCH_COST_MICROS - 1),
        research_tools=research_tools_factory(),
    )

    outcome = run_single_tick(deps, beat=1)

    assert "ForecastCreated" not in _event_types(deps)
    assert "SelectorDecisionRecorded" not in _event_types(deps)
    assert _payloads_of(deps, _HALT_EVENT_TYPE) == [
        {
            "market_ticker": _TICKER,
            "halt_kind": _HALT_KIND_PER_FORECAST,
            "utc_day": _FIXED_UTC_DAY,
            "spent_micros": _EXPECTED_RESEARCH_COST_MICROS,
            "budget_micros": _EXPECTED_RESEARCH_COST_MICROS - 1,
        }
    ]
    assert outcome.research_halted is True
    assert [payload["beat"] for payload in _payloads_of(deps, "ModeHeartbeat")] == [1]
    deps.store.verify_chain()


def test_zero_per_forecast_ceiling_is_legal_and_halts_immediately(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A zero per-forecast ceiling is accepted at build time and fails closed.

    Pins the documented fail-closed reading of zero: it is a legal ceiling
    meaning "immediately exhausted", not an accidental "unlimited". A budget
    that silently treated `0` as unset would produce a forecast here.
    """
    from windbreak.scheduler.loop import run_single_tick

    deps = _build_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
        config=_budgeted_config(per_forecast_micros=0),
        research_tools=research_tools_factory(),
    )

    outcome = run_single_tick(deps, beat=1)

    assert "ForecastCreated" not in _event_types(deps)
    assert outcome.research_halted is True
    halted = _payloads_of(deps, _HALT_EVENT_TYPE)[0]
    assert halted["halt_kind"] == _HALT_KIND_PER_FORECAST


def test_a_negative_ceiling_aborts_composition_rather_than_running_unbudgeted(
    books_dir: Path,
    cassette_path: Path,
    report_dir: Path,
    research_tools_factory,
    tmp_path: Path,
) -> None:
    """A negative ceiling raises at `build_paper_deps`, never degrading to unlimited.

    The YAML loader has no range checks, so a negative ceiling reaches the
    composition root intact. Building the budget there converts it into a loud
    startup abort -- the fail-closed outcome -- instead of a loop that runs
    forever with an unenforceable ceiling.
    """
    with pytest.raises(ValueError, match="per_day_micros"):
        _build_deps(
            books_dir=books_dir,
            cassette_path=cassette_path,
            ledger_path=ledger_path_for(tmp_path),
            report_dir=report_dir,
            config=_budgeted_config(per_day_micros=-1),
            research_tools=research_tools_factory(),
        )
