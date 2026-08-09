"""Failing-first tests for gate-plan-anchored `paper_window_days` (issue #243, RED).

The PAPER->LIVE_MICRO gate's `paper_window_days` criterion (SPEC S10.9, `GE 90`)
is meant to measure elapsed days *since the currently-registered gate plan's
`paper_clock_start`* -- so a `GatePlanChanged` reset (SPEC §13.6) always shortens
the effective window on the very next evidence snapshot. Nothing today derives
`GateEvidence.paper_window_days` that way: it is caller-supplied, untrusted, and
free to carry a stale value computed against a plan that has since been
superseded (the motivating bug this issue closes).

The new leaf module `windbreak.riskkernel.evidence` does not exist yet, so
every symbol imported from it below fails collection with `ModuleNotFoundError:
No module named 'windbreak.riskkernel.evidence'` -- the expected Gate 1 RED
state for issue #243. Its two functions:

- `anchored_paper_window_days(store, *, now)`: reads the ledger's *latest* gate
  plan registration and returns the floored whole-day count elapsed since its
  `paper_clock_start`, failing closed with `GatePlanUnavailableError` (imported
  from `windbreak.riskkernel.promotion`, already real) whenever no verified
  anchor is available -- no store, an empty ledger, a corrupt/tampered
  registration, or a clock that has gone backwards relative to the anchor.
- `anchor_gate_evidence(evidence, store, *, now)`: returns a new `GateEvidence`
  with `paper_window_days` unconditionally overwritten by the anchored value
  (`dataclasses.replace`, every other field byte-identical, the original
  object left untouched -- `GateEvidence` is frozen).

Symbols from already-existing modules (`windbreak.evaluation.preregistration`,
`windbreak.riskkernel.promotion`, `windbreak.ledger.store`,
`windbreak.config.schema`) are imported at module scope, since those modules
already exist and importing them cannot hide which new behavior a given test
covers -- matching this suite's established RED convention (see
`tests/evaluation/test_preregistration.py`'s module docstring).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from windbreak.config.schema import EvaluationConfig
from windbreak.evaluation.preregistration import build_gate_plan, register_gate_plan
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.riskkernel.promotion import (
    GateEvidence,
    GatePlanUnavailableError,
    evaluate_promotion,
    paper_gate_from_thresholds,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from windbreak.ledger.store import LedgerStore

#: The `paper_fill_model_version` every plan built in this file pins (SPEC
#: §17.4); its value is irrelevant to anchoring, so a single fixed string is
#: reused everywhere.
_PAPER_FILL_MODEL_VERSION = "evidence-anchoring-test-v1"

#: A fixed, arbitrary base epoch (2023-11-14T22:13:20Z) every registration in
#: this file anchors its arithmetic to.
_BASE_EPOCH = 1_700_000_000


def _ledger_store(tmp_path: Path) -> SqliteLedgerStore:
    """Build a tmp-path-backed `SqliteLedgerStore` for one test's ledger.

    Args:
        tmp_path: The pytest-provided temporary directory to root the
            database file in.

    Returns:
        A fresh `SqliteLedgerStore` at `tmp_path / "ledger.db"`.
    """
    return SqliteLedgerStore(tmp_path / "ledger.db")


def _sequence_clock(*epochs: int) -> Callable[[], int]:
    """Build a `now` callable returning each of `epochs`, once, in order.

    Args:
        epochs: The whole-epoch-second values to return, one per call.

    Returns:
        A callable that pops the next scripted epoch on every call.
    """
    iterator = iter(epochs)

    def _next_epoch() -> int:
        """Return the next scripted epoch."""
        return next(iterator)

    return _next_epoch


def _constant_clock(epoch: int) -> Callable[[], int]:
    """Build a `now` callable returning the same fixed `epoch` every call.

    Args:
        epoch: The whole-epoch-second value returned on every call.

    Returns:
        A callable that always returns `epoch`.
    """

    def _same_epoch() -> int:
        """Return the fixed epoch on every call."""
        return epoch

    return _same_epoch


def _register_plan(
    store: LedgerStore,
    evaluation: EvaluationConfig,
    *,
    now: Callable[[], int],
) -> None:
    """Build a `GatePlan` from `evaluation` and register it into `store`.

    Args:
        store: The ledger to register the plan into.
        evaluation: The config the plan's thresholds are snapshotted from.
        now: The clock supplying the registration's paper-clock epoch.
    """
    plan = build_gate_plan(
        evaluation, paper_fill_model_version=_PAPER_FILL_MODEL_VERSION
    )
    register_gate_plan(plan, store, now=now)


def _paper_evidence(*, paper_window_days: int) -> GateEvidence:
    """Build a `GateEvidence` carrying a caller-supplied `paper_window_days`.

    Every other field defaults to its failing (`0`/`False`) value; only
    `paper_window_days` and `resolved_realtime_forecast_count` (an arbitrary
    second field, chosen to prove `anchor_gate_evidence` leaves non-anchored
    fields untouched) are set to non-default values.

    Args:
        paper_window_days: The (untrusted, pre-anchoring) window-day value.

    Returns:
        The assembled `GateEvidence`.
    """
    return GateEvidence(
        paper_window_days=paper_window_days,
        resolved_realtime_forecast_count=7,
    )


# ---------------------------------------------------------------------------
# 1. Happy path: anchored_paper_window_days and anchor_gate_evidence agree,
#    every non-anchored field survives byte-identical, original untouched.
# ---------------------------------------------------------------------------


def test_anchored_paper_window_days_measures_whole_days_since_registration(
    tmp_path: Path,
) -> None:
    """`anchored_paper_window_days` returns the floored day count since the
    registered plan's `paper_clock_start`: 91 whole days after registration.
    """
    from windbreak.riskkernel.evidence import anchored_paper_window_days

    store = _ledger_store(tmp_path)
    try:
        _register_plan(store, EvaluationConfig(), now=_constant_clock(_BASE_EPOCH))

        result = anchored_paper_window_days(
            store, now=_constant_clock(_BASE_EPOCH + 91 * 86_400)
        )

        assert result == 91
    finally:
        store.close()


def test_anchor_gate_evidence_overwrites_paper_window_days_only(
    tmp_path: Path,
) -> None:
    """`anchor_gate_evidence` overwrites only `paper_window_days`, returning a
    new `GateEvidence` with every other field byte-identical to the input and
    leaving the original object's `paper_window_days` unchanged (frozen).
    """
    from windbreak.riskkernel.evidence import anchor_gate_evidence

    store = _ledger_store(tmp_path)
    try:
        _register_plan(store, EvaluationConfig(), now=_constant_clock(_BASE_EPOCH))
        original = _paper_evidence(paper_window_days=500)

        anchored = anchor_gate_evidence(
            original, store, now=_constant_clock(_BASE_EPOCH + 91 * 86_400)
        )

        assert anchored.paper_window_days == 91
        assert anchored is not original
        assert original.paper_window_days == 500

        expected_payload = dataclasses.replace(
            original, paper_window_days=91
        ).to_payload()
        assert anchored.to_payload() == expected_payload
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 2. The motivating bug: a GatePlanChanged reset shortens the anchored window,
#    flipping a stale-window pass into a real fail.
# ---------------------------------------------------------------------------


def test_gate_plan_change_resets_the_anchored_window_not_the_original_start(
    tmp_path: Path,
) -> None:
    """A `GatePlanChanged` registered ~100 days after plan A anchors the
    window to plan B's *own* `paper_clock_start`: 5 days later, the anchored
    value is 5 (measured from B), never ~105 (measured from A) -- the exact
    §13.6 reset semantics this issue closes the evidence-side gap on.
    """
    from windbreak.riskkernel.evidence import anchored_paper_window_days

    store = _ledger_store(tmp_path)
    try:
        plan_b_epoch = _BASE_EPOCH + 100 * 86_400
        clock = _sequence_clock(_BASE_EPOCH, plan_b_epoch)
        _register_plan(store, EvaluationConfig(), now=clock)
        changed_config = dataclasses.replace(
            EvaluationConfig(),
            brier_skill_required_ppm=(EvaluationConfig().brier_skill_required_ppm + 1),
        )
        _register_plan(store, changed_config, now=clock)

        result = anchored_paper_window_days(
            store, now=_constant_clock(plan_b_epoch + 5 * 86_400)
        )

        assert result == 5
    finally:
        store.close()


def test_gate_plan_change_flips_a_stale_passing_window_to_a_real_failure(
    tmp_path: Path,
) -> None:
    """Feeding the anchored 5-day window (post-reset) into the real PAPER gate
    fails `paper_window_days` (GE 90), whereas the stale ~105-day figure
    (measured against the superseded plan A) would have passed it -- proving
    the anchor, not just the arithmetic, is what the promotion gate consumes.
    """
    from windbreak.riskkernel.evidence import anchor_gate_evidence

    store = _ledger_store(tmp_path)
    try:
        plan_b_epoch = _BASE_EPOCH + 100 * 86_400
        clock = _sequence_clock(_BASE_EPOCH, plan_b_epoch)
        _register_plan(store, EvaluationConfig(), now=clock)
        changed_config = dataclasses.replace(
            EvaluationConfig(),
            brier_skill_required_ppm=(EvaluationConfig().brier_skill_required_ppm + 1),
        )
        _register_plan(store, changed_config, now=clock)
        stale_evidence = _paper_evidence(paper_window_days=105)

        anchored = anchor_gate_evidence(
            stale_evidence, store, now=_constant_clock(plan_b_epoch + 5 * 86_400)
        )

        gate = paper_gate_from_thresholds(
            promotion_min_resolved=0,
            promotion_min_independent_event_groups=0,
            brier_skill_required_ppm=0,
        )
        decision = evaluate_promotion(gate, anchored)
        results_by_id = {result.criterion_id: result for result in decision.results}

        assert results_by_id["paper_window_days"].observed == 5
        assert results_by_id["paper_window_days"].passed is False
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 3. Missing anchors fail closed: no store, empty (freshly-created) ledger.
# ---------------------------------------------------------------------------


def test_anchored_paper_window_days_raises_when_store_is_none() -> None:
    """`store=None` raises `GatePlanUnavailableError` -- there is no ledger to
    read an anchor from at all.
    """
    from windbreak.riskkernel.evidence import anchored_paper_window_days

    with pytest.raises(GatePlanUnavailableError):
        anchored_paper_window_days(None, now=_constant_clock(_BASE_EPOCH))


def test_anchor_gate_evidence_raises_when_store_is_none() -> None:
    """`store=None` raises `GatePlanUnavailableError` from `anchor_gate_evidence`
    too, propagated straight from the anchoring helper.
    """
    from windbreak.riskkernel.evidence import anchor_gate_evidence

    with pytest.raises(GatePlanUnavailableError):
        anchor_gate_evidence(
            _paper_evidence(paper_window_days=0),
            None,
            now=_constant_clock(_BASE_EPOCH),
        )


def test_anchored_paper_window_days_raises_on_an_empty_ledger(
    tmp_path: Path,
) -> None:
    """A wired but empty ledger (no `GatePlanRegistered` ever appended) raises
    `GatePlanUnavailableError` identically to no store at all.
    """
    from windbreak.riskkernel.evidence import anchored_paper_window_days

    store = _ledger_store(tmp_path)
    try:
        with pytest.raises(GatePlanUnavailableError):
            anchored_paper_window_days(store, now=_constant_clock(_BASE_EPOCH))
    finally:
        store.close()


def test_anchor_gate_evidence_raises_on_an_empty_ledger(tmp_path: Path) -> None:
    """`anchor_gate_evidence` also raises `GatePlanUnavailableError` on an
    empty ledger, matching `anchored_paper_window_days`.
    """
    from windbreak.riskkernel.evidence import anchor_gate_evidence

    store = _ledger_store(tmp_path)
    try:
        with pytest.raises(GatePlanUnavailableError):
            anchor_gate_evidence(
                _paper_evidence(paper_window_days=0),
                store,
                now=_constant_clock(_BASE_EPOCH),
            )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 4. Corrupt registration fails closed, chaining the original ValueError.
# ---------------------------------------------------------------------------


def test_anchored_paper_window_days_chains_the_hash_mismatch_value_error(
    tmp_path: Path,
) -> None:
    """A tampered `plan_hash` (so `latest_gate_plan_registration` raises
    `ValueError`) is wrapped in `GatePlanUnavailableError`, with the original
    `ValueError` preserved as `__cause__` -- mirroring
    `test_latest_gate_plan_registration_fails_closed_on_hash_mismatch`'s own
    tampering technique (`tests/evaluation/test_preregistration.py`).
    """
    from windbreak.evaluation.preregistration import (
        GatePlanRegistered,
        build_gate_plan,
    )
    from windbreak.riskkernel.evidence import anchored_paper_window_days

    store = _ledger_store(tmp_path)
    try:
        plan = build_gate_plan(
            EvaluationConfig(), paper_fill_model_version=_PAPER_FILL_MODEL_VERSION
        )
        store.append(
            GatePlanRegistered(
                component="evaluation",
                **plan.canonical_dict(),
                plan_hash="0" * 64,
                paper_clock_start=_BASE_EPOCH,
            )
        )

        with pytest.raises(GatePlanUnavailableError) as excinfo:
            anchored_paper_window_days(store, now=_constant_clock(_BASE_EPOCH + 86_400))

        assert isinstance(excinfo.value.__cause__, ValueError)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 5. Clock skew: now() before paper_clock_start fails closed, never negative.
# ---------------------------------------------------------------------------


def test_anchored_paper_window_days_raises_on_a_clock_behind_the_anchor(
    tmp_path: Path,
) -> None:
    """`now()` one second *before* the registered `paper_clock_start` raises
    `GatePlanUnavailableError` -- backwards clock skew is fail-closed, never
    clamped to `0` and never returned as a negative int.
    """
    from windbreak.riskkernel.evidence import anchored_paper_window_days

    store = _ledger_store(tmp_path)
    try:
        _register_plan(store, EvaluationConfig(), now=_constant_clock(_BASE_EPOCH))

        with pytest.raises(GatePlanUnavailableError):
            anchored_paper_window_days(store, now=_constant_clock(_BASE_EPOCH - 1))
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 6. Exact boundary math, both sides of the GE-90 comparator (mutation-
#    resistant): 90*86400 -> 90, 90*86400 - 1 -> 89, 0 -> 0.
# ---------------------------------------------------------------------------


def test_anchored_paper_window_days_at_exactly_ninety_days_is_ninety(
    tmp_path: Path,
) -> None:
    """Elapsed time of exactly `90 * 86_400` seconds anchors to `90` -- the
    exact GE-90 boundary the PAPER gate's `paper_window_days` criterion
    requires to *pass*.
    """
    from windbreak.riskkernel.evidence import anchored_paper_window_days

    store = _ledger_store(tmp_path)
    try:
        _register_plan(store, EvaluationConfig(), now=_constant_clock(_BASE_EPOCH))

        result = anchored_paper_window_days(
            store, now=_constant_clock(_BASE_EPOCH + 90 * 86_400)
        )

        assert result == 90
    finally:
        store.close()


def test_anchored_paper_window_days_one_second_short_of_ninety_days_is_eighty_nine(
    tmp_path: Path,
) -> None:
    """One second short of `90 * 86_400` seconds elapsed anchors to `89`, not
    `90` -- the exact GE-90 boundary the criterion requires to *fail*.
    """
    from windbreak.riskkernel.evidence import anchored_paper_window_days

    store = _ledger_store(tmp_path)
    try:
        _register_plan(store, EvaluationConfig(), now=_constant_clock(_BASE_EPOCH))

        result = anchored_paper_window_days(
            store, now=_constant_clock(_BASE_EPOCH + 90 * 86_400 - 1)
        )

        assert result == 89
    finally:
        store.close()


def test_anchored_paper_window_days_at_registration_instant_is_zero(
    tmp_path: Path,
) -> None:
    """`now() == paper_clock_start` (no time elapsed yet) anchors to exactly
    `0`, not a clock-skew failure -- the boundary between "just registered"
    and "clock went backwards".
    """
    from windbreak.riskkernel.evidence import anchored_paper_window_days

    store = _ledger_store(tmp_path)
    try:
        _register_plan(store, EvaluationConfig(), now=_constant_clock(_BASE_EPOCH))

        result = anchored_paper_window_days(store, now=_constant_clock(_BASE_EPOCH))

        assert result == 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 7. Float-free property: floor division by SECONDS_PER_DAY, always a
#    non-bool int, for arbitrary non-negative elapsed deltas.
# ---------------------------------------------------------------------------


@given(
    paper_clock_start=st.integers(min_value=1, max_value=2_000_000_000),
    delta=st.integers(min_value=0, max_value=500 * 86_400),
)
@settings(deadline=None, max_examples=100)
def test_anchored_paper_window_days_is_exact_floor_division_by_seconds_per_day(
    tmp_path_factory: pytest.TempPathFactory, paper_clock_start: int, delta: int
) -> None:
    """For arbitrary `paper_clock_start` and non-negative `delta`, anchoring
    `delta` seconds after registration yields exactly `delta // 86_400` -- a
    plain, float-free floor division -- and the result is always a genuine
    `int`, never a `bool` (SPEC §6.1).

    Args:
        tmp_path_factory: Pytest's per-example temp-directory factory (a
            plain fixture, not `tmp_path`, since Hypothesis reuses this test
            function across many examples and each needs its own ledger file).
        paper_clock_start: The registration's paper-clock epoch.
        delta: The non-negative number of elapsed seconds `now()` advances by.
    """
    from windbreak.riskkernel.evidence import anchored_paper_window_days

    store = _ledger_store(tmp_path_factory.mktemp("evidence-anchoring"))
    try:
        _register_plan(
            store, EvaluationConfig(), now=_constant_clock(paper_clock_start)
        )

        result = anchored_paper_window_days(
            store, now=_constant_clock(paper_clock_start + delta)
        )

        assert result == delta // 86_400
        assert type(result) is int
        assert not isinstance(result, bool)
    finally:
        store.close()
