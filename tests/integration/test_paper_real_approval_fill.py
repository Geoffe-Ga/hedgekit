"""The approve -> route -> fill leg, driven by a real kernel approval (#450).

Issue #450 states that ``_approve_stage`` -> ``_route_intent`` ->
``_reconcile_to_fixpoint`` had never been traversed by a **real**
:class:`~windbreak.scheduler.loop.KernelApproval` outcome: every test of that
leg doubled the approval seam, so nothing established that the real
:class:`~windbreak.riskkernel.process.RiskKernel` +
:class:`~windbreak.riskkernel.reservations.ApprovalPipeline` pair ever
*produces* a token on real inputs.

What has changed since #450 was filed
-------------------------------------

That premise is now **partly stale**, and saying so precisely is part of this
module's job. Issue #438's ``tests/scheduler/test_paper_intent_barriers.py``
(PR #481) already drives the real :func:`windbreak.scheduler.loop.run_single_tick`
over the real shipped composition to an emitted intent, and its
``test_the_intent_is_vetoed_on_beat_one_and_fills_on_beat_two`` already observes
that intent approved by the real seam and filled for 25 000 contract-centis on
the following beat. #450's acceptance criteria 1 and 2 -- a whole tick, an
``intent_count >= 1`` with ``filled_centis > 0``, no doubled approval, no
injected token -- were therefore satisfied by work that landed after the issue
was written.

Three of its six criteria were not, and this module is what closes them:

* **AC3** -- nothing documented *which* SPEC S10.3 checks pass and why, nor
  established that no threshold was moved to get there.
  :func:`test_every_spec_10_3_check_passes_under_the_context_the_fill_used`
  measures all 24 verdicts rather than describing them, and
  :func:`test_only_the_correlation_declaration_and_state_dir_leave_the_defaults`
  derives the config difference from :class:`~windbreak.config.schema.WindbreakConfig`
  rather than restating it.
* **AC4** -- the transition *subsequence* was pinned; the ledgered row sequence
  was not. :func:`test_a_real_kernel_approval_fills_and_ledgers_a_golden_sequence`
  pins every row of both beats as an exact golden sequence, the
  ``tests/integration/test_paper_universe.py`` precedent.
* **AC5** -- ``deps.store.verify_chain()`` was never called on that ledger. It
  is now, on the ledger the fill was written to.

Which SPEC S10.3 checks pass, and why (AC3)
-------------------------------------------

All 24. This list is the prose half; the measured half is
:func:`test_every_spec_10_3_check_passes_under_the_context_the_fill_used`,
which recomputes every verdict from the very context the approval was decided
under, so a claim here that stops being true fails there.

* ``instrument_whitelist`` -- ``_approve_stage`` sets the whitelist to the one
  candidate ticker it is approving for, so the intent's own market is on it.
* ``jurisdiction_product_eligibility`` -- the fixture market resolves through
  :meth:`~windbreak.connector.paper.PaperExchange.get_market` and carries a
  tradable product, so the check has a market to prove eligible.
* ``mode_permission_ceiling`` -- the kernel stamps its own ``Mode.PAPER``,
  which is a trading mode and is not the LIVE_MICRO-capped one.
* ``floor_invariant`` -- the fixture opens with 10 000 000 000 micros against
  :class:`~windbreak.config.schema.CapitalConfig`'s 1 000 000 000 floor, and the
  worst-case cost of a 25 000-centis order at 500 pips is 12 500 000.
* ``balance_reconciliation``, ``position_reconciliation``,
  ``open_order_reconciliation`` -- the tick's own read-only verification cycle
  ran first and ledgered ``VerificationPassed`` with ``outcome == "CLEAN"``;
  ``_approve_stage`` threads that same snapshot onto the context, so all three
  read a reconciled venue rather than the fail-closed ``None``.
* ``fee_upper_bound_present``, ``settlement_fee_upper_bound`` -- the paper
  venue's fee model supplies both bounds.
* ``concentration_limits`` -- the account is flat and the order's 12 500 000
  micros is a 0.125% share of equity, far inside every ppm cap.
* ``daily_loss_limit`` -- **one of the two that fail on beat 1**, on
  ``realized_loss_today (0) >= threshold (0)``, because the threshold is a ppm
  share of ``equity_start_of_day`` and no ``EquitySampled`` row exists yet on a
  fresh ledger. Beat 1 ledgers one, so beat 2 measures against a real baseline
  and against the day's own realized loss, and the check passes. That transient
  is issue #481's finding and is pinned, not fixed, here.
* ``trailing_drawdown_limit`` -- **the other one**, since issue #514. The same
  absent ``EquitySampled`` row leaves it with neither a high-water mark nor a
  current reading, so it refuses on ``0 >= 0`` for want of evidence and lifts
  on beat 2, where the mark and the reading are the same sample and nothing has
  drawn down. Before #514 it passed on beat 1 -- but so it would have at any
  drawdown whatsoever, because the mark it was fed was hardcoded to zero.
* ``velocity_limits`` -- one order in the hour, and the day's notional is the
  12 500 000 this order adds.
* ``quote_freshness`` -- the book's own ``fetched_at`` is the injected fixed
  clock reading, so the quote is zero seconds old at evaluation time.
* ``forecast_freshness`` -- the forecast was created by this same tick under the
  same injected clock.
* ``price_band_compliance`` -- 500 pips is exactly
  ``RiskConfig().min_open_price_pips``, which the band admits (issue #481's
  ``test_best_case_ask_is_the_open_band_floor`` is what makes that load-bearing
  rather than lucky).
* ``participation_cap_compliance`` -- the selector already sized to the
  participation cap: its own rendered sizing line reports
  ``binding_cap=participation final_centis=25000`` against a 100 000-centis
  resting level.
* ``human_ack_satisfied`` -- no human-ack ceiling is configured, so no order
  cost can exceed it.
* ``approval_token_uniqueness``, ``idempotency_key_uniqueness`` -- beat 1
  minted nothing (it vetoed), so beat 2's intent id and idempotency key are
  both unused on the ledger the pipeline consults.
* ``clock_skew_limit`` -- the venue's replay clock and the injected tick clock
  are anchored to the same instant.
* ``exchange_status_ok`` -- the venue reports ``open``, ledgered as
  ``ExchangeStatusObserved``.
* ``pipeline_heartbeat_ok`` -- the tick ledgered its own
  ``PipelineHeartbeatRecorded`` earlier in the same beat.
* ``reduce_only_provable`` -- the intent is an *open*, which the check approves
  without needing to prove a reduction.

No threshold was moved to reach any of that.
:func:`test_only_the_correlation_declaration_and_state_dir_leave_the_defaults`
derives that claim by comparing every field of the tick's config against
``WindbreakConfig()``: exactly two differ, and neither is a risk threshold.
``correlation`` carries the operator's bucket declaration, without which
:func:`windbreak.scheduler.exposure.project_exposure` refuses to project any
exposure at all -- a declaration, not a loosened limit. ``ops.state_dir`` points
the kill-file watcher at a per-test directory instead of the shipped
``~/.local/share/windbreak``, which is a filesystem location; leaving it at the
default would let a ``KILL`` file a developer created by hand silently kill
these ticks.

What this scenario cannot distinguish
-------------------------------------

Two of the twenty-four pass for reasons a single in-band, opening, outright
order cannot separate from their alternatives, and recording that is part of
being honest about *why* the fill is reached:

* ``reduce_only_provable`` -- an **equivalence**. The intent is an open, and
  ``_is_derisking_close`` returns ``False`` on an opening action before it ever
  reads ``context.market.open_position``, which is the only place any check
  reads it. So the open position ``_approve_stage`` threads (issue #373) cannot
  change any verdict on this path; replacing it with ``None`` leaves every
  assertion below green because it genuinely decides nothing here.
* ``quote_freshness`` -- a **gap, not an equivalence**. The book's own
  ``fetched_at`` and the approval stage's second clock read are the same instant
  under an injected fixed clock, so ``_approve_stage`` threading the book's
  epoch (issue #369) rather than its own ``now_epoch_s`` is unobservable here.
  ``_approve_stage``'s docstring calls that substitution the thing that made the
  check "incapable of ever vetoing", and nothing in this module falsifies it.
  Separating the two needs a clock that advances between the exchange's replay
  anchor and the approval stage, which is issue #369's scenario rather than
  #450's; it is reported rather than quietly absorbed.

What the golden sequence is allowed to say
------------------------------------------

``ACKED`` is where the gateway's submission chain stops for an order that
crosses outright: the gateway's ``_track_if_resting``
tracks only a *resting* remainder, and the ``FILL`` edge is the Reconciler's to
ledger for a tracked order that filled out-of-band. This order rests nothing, so
there is nothing left to reconcile and no further transition to record. That is
pinned in both directions -- the golden sequence carries exactly the four
transitions and no ``FILL``, and ``deps.gateway.tracked_orders()`` is empty --
so it is recorded as observed behaviour rather than asserted as desirable.
"""

from __future__ import annotations

import dataclasses
import shutil
from typing import TYPE_CHECKING

from tests.integration.conftest import read_event_type_payload_pairs
from tests.scheduler.test_paper_intent_barriers import (
    BEST_CASE_ASK_PIPS,
    FRESH_LEDGER_VETO_REASONS,
    SHIPPED_BOOKS,
    SHIPPED_CASSETTE,
    SIZED_FILL_CENTIS,
    TICKER,
    _bucketed_config,
    _finding_research_tools,
    _fixed_clock,
    _NearCertainVoteTransport,
    _only,
    _rows,
    _tradeable_books,
    _write_track_records,
)
from windbreak.config.schema import OpsConfig, WindbreakConfig
from windbreak.riskkernel.checks import DEFAULT_CHECKS
from windbreak.scheduler.loop import build_paper_deps, run_single_tick

if TYPE_CHECKING:
    from pathlib import Path

    from windbreak.riskkernel.checks import OrderIntent
    from windbreak.riskkernel.context import EvaluationContext
    from windbreak.riskkernel.reservations import ApprovalOutcome
    from windbreak.scheduler.loop import ApprovalSeam, PaperTickDeps

#: The SPEC S10.3 checks that veto on the first beat of a fresh ledger, and the
#: only ones that ever do in this scenario. Both refuse for want of the same
#: evidence -- a ledger with no `EquitySampled` row has neither a day baseline
#: (#481) nor a high-water mark (#514) -- and both lift on beat 2, once the
#: first beat's own sample exists. They are listed in `DEFAULT_CHECKS` order,
#: which is the order the kernel records their reasons in.
BEAT_ONE_VETOING_CHECKS = ("daily_loss_limit", "trailing_drawdown_limit")

#: The approval token's sequence number: the first (and only) token the
#: pipeline mints across both beats, since beat 1 vetoed before reserving.
FIRST_TOKEN_SEQUENCE_NUMBER = 1

#: The two config fields the emitting scenario moves off `WindbreakConfig()`.
#: Neither is a threshold -- see the module docstring's AC3 section.
NON_DEFAULT_CONFIG_FIELDS = ("correlation", "ops")

#: The ledgered gateway transitions an outright-crossing order produces, in
#: order. There is no `FILL`: nothing rests, so nothing reconciles.
SUBMISSION_TRANSITIONS = ("APPROVE", "REQUEST_SUBMISSION", "SUBMIT", "ACK")


class _RecordingApprovalSeam:
    """A pass-through observer around the real :class:`ApprovalSeam`.

    Deliberately **not** a double: every call is forwarded to the real
    :class:`~windbreak.scheduler.loop.KernelApproval`, and the real outcome --
    token and all -- is returned unmodified. Its only effect is to retain the
    ``(intent, context)`` pair each approval was decided under, which is
    otherwise built and discarded inside ``_approve_stage``.
    """

    def __init__(self, inner: ApprovalSeam) -> None:
        """Wrap the real seam.

        Args:
            inner: The real approval seam every call is forwarded to.
        """
        self._inner = inner
        self.calls: list[tuple[OrderIntent, EvaluationContext, ApprovalOutcome]] = []

    def decide(
        self, intent: OrderIntent, context: EvaluationContext
    ) -> ApprovalOutcome:
        """Forward to the real seam, retaining the inputs and the outcome.

        Args:
            intent: The order intent to approve.
            context: The evaluation context the checks read.

        Returns:
            The real seam's own
            :class:`~windbreak.riskkernel.reservations.ApprovalOutcome`,
            unmodified.
        """
        outcome = self._inner.decide(intent, context)
        self.calls.append((intent, context, outcome))
        return outcome


def _emitting_config(state_dir: Path) -> WindbreakConfig:
    """Return the emitting configuration with a per-test kill-state directory.

    Args:
        state_dir: The directory the kill-file watcher polls.

    Returns:
        The bucketed configuration with :attr:`OpsConfig.state_dir` redirected.
    """
    return dataclasses.replace(
        _bucketed_config(), ops=OpsConfig(state_dir=str(state_dir))
    )


def _emitting_deps(tmp_path: Path) -> PaperTickDeps:
    """Wire the one PAPER composition that reaches a real approval and a fill.

    Every barrier issue #438 enumerated is cleared by a genuine input -- a deep
    two-sided book, an in-window close, verifiable citations, near-certain
    votes, a declared correlation bucket, proven providers, capital far above
    the floor -- and no threshold is touched. The shipped books fixture is
    copied first, so the re-dating and re-pricing happen on a private copy and
    never on the committed one.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The wired dependencies, over a fresh ledger.
    """
    books = tmp_path / "books"
    shutil.copytree(SHIPPED_BOOKS, books)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    _tradeable_books(books)
    _write_track_records(report_dir)
    deps = build_paper_deps(
        books_dir=books,
        cassette_path=SHIPPED_CASSETTE,
        ledger_path=tmp_path / "ledger.db",
        report_dir=report_dir,
        config=_emitting_config(tmp_path / "state"),
        research_tools=_finding_research_tools(tmp_path / "cache"),
        clock=_fixed_clock,
    )
    return dataclasses.replace(deps, transport=_NearCertainVoteTransport())


def _row_shape(deps: PaperTickDeps) -> list[tuple[str, str]]:
    """Project the ledger into an ``(event_type, subject)`` sequence.

    ``subject`` is the gateway transition's own edge name for an
    ``OrderTransitionLedgered`` row, and otherwise whichever of
    ``ticker``/``market_ticker`` the payload carries, or ``""`` for the rows
    that describe the loop rather than a market.

    That is one field more than ``tests/integration/test_paper_universe.py``'s
    ``_ledger_shape`` projects, and the extra field is the point here: this
    suite's claim is about the *order* of the four submission transitions, and a
    ticker-only projection would render all four as the same indistinguishable
    pair -- a golden sequence that could not tell ``APPROVE, ACK, SUBMIT,
    REQUEST_SUBMISSION`` from the real chain.

    Args:
        deps: The wired tick dependencies whose store is read.

    Returns:
        One ``(event_type, subject)`` pair per row, in ledger order.
    """
    shape: list[tuple[str, str]] = []
    records: list[object] = list(deps.store.read_all())
    for event_type, payload in read_event_type_payload_pairs(records):
        subject = (
            payload.get("event")
            if event_type == "OrderTransitionLedgered"
            else payload.get("ticker") or payload.get("market_ticker")
        )
        shape.append((event_type, str(subject or "")))
    return shape


def _last(rows: list[tuple[str, dict[str, object]]], event: str) -> dict[str, object]:
    """Return the payload of the last row of ``event``.

    Args:
        rows: The ledgered rows.
        event: The event type wanted.

    Returns:
        That row's ``data`` payload.

    Raises:
        AssertionError: If no row of ``event`` is on the ledger at all.
    """
    matches = [data for event_type, data in rows if event_type == event]
    assert matches != [], f"expected at least one {event}, got none"
    return matches[-1]


def _check_verdicts(
    intent: OrderIntent, context: EvaluationContext
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Re-run all 24 SPEC S10.3 checks over one decided approval.

    Args:
        intent: The intent the approval was decided for.
        context: The context the approval was decided under.

    Returns:
        The passing check names and the vetoing check names, each in SPEC S10.3
        evaluation order.
    """
    passing: list[str] = []
    vetoing: list[str] = []
    for check in DEFAULT_CHECKS:
        target = vetoing if check(intent, context).vetoed else passing
        target.append(check.name)
    return tuple(passing), tuple(vetoing)


def test_a_real_kernel_approval_fills_and_ledgers_a_golden_sequence(
    tmp_path: Path,
) -> None:
    """Two real beats, every ledgered row pinned, and a verified hash chain.

    Issue #450's acceptance criteria 1, 4 and 5 in one run. The fill is reached
    through :func:`~windbreak.scheduler.loop.run_single_tick`, never a stage:
    nothing here calls ``_approve_stage``, ``_route_intent``,
    ``deps.approval.decide`` or ``deps.gateway.process_intent``, so the routing
    functions are entered by production code or not at all.

    The whole ledger is compared as an exact sequence rather than by membership,
    which is what makes it fail on a *dropped* leg as loudly as on a wrong one.
    Both beats are included because the pair is the claim: the same intent, the
    same books, the same votes, vetoed on beat 1 and filled on beat 2, so the
    fill cannot be read as something the fixture simply always produced.

    The approval's genuineness is asserted from the ledger rather than trusted.
    ``ReservationCreated`` and ``ApprovalTokenIssued`` are written by
    :class:`~windbreak.riskkernel.reservations.ApprovalPipeline` itself and by
    nothing else, exactly one of each exists across both beats, and the intent
    id on the minted token is the one the kernel's own ``IntentApproved`` row
    names -- so the token the gateway then verified is the one the real pipeline
    minted on this tick.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    deps = _emitting_deps(tmp_path)

    first = run_single_tick(deps, beat=1)
    second = run_single_tick(deps, beat=2)

    deps.store.verify_chain()
    assert _row_shape(deps) == [
        # Built by the gateway's boot recovery, before either beat runs.
        ("RecoveryCompleted", ""),
        # --- Beat 1: an intent the kernel vetoes on the day's zero baseline.
        ("ScreenDecisionRecorded", TICKER),
        ("PipelineHeartbeatRecorded", ""),
        ("VerificationPassed", ""),
        ("MarketSnapshotRecorded", TICKER),
        ("ResearchSpendRecorded", TICKER),
        ("ForecastCreated", TICKER),
        ("ProviderVoteRecorded", TICKER),
        ("ProviderVoteRecorded", TICKER),
        ("ProviderVoteRecorded", TICKER),
        ("SelectorDecisionRecorded", TICKER),
        ("ExchangeStatusObserved", ""),
        # No ReservationCreated and no ApprovalTokenIssued follow a veto: the
        # pipeline is never reached, so nothing is reserved and nothing minted.
        ("IntentVetoed", ""),
        ("ModeHeartbeat", ""),
        # The row that gives beat 2 the start-of-day equity beat 1 lacked.
        ("EquitySampled", ""),
        ("PositionsSnapshotRecorded", ""),
        # --- Beat 2: the same intent, now approved, routed, and filled.
        ("ScreenDecisionRecorded", TICKER),
        ("PipelineHeartbeatRecorded", ""),
        ("VerificationPassed", ""),
        ("MarketSnapshotRecorded", TICKER),
        ("ResearchSpendRecorded", TICKER),
        ("ForecastCreated", TICKER),
        ("ProviderVoteRecorded", TICKER),
        ("ProviderVoteRecorded", TICKER),
        ("ProviderVoteRecorded", TICKER),
        ("SelectorDecisionRecorded", TICKER),
        ("ExchangeStatusObserved", ""),
        ("IntentApproved", ""),
        # The two rows only ApprovalPipeline writes: capital reserved, then a
        # single-use token signed.
        ("ReservationCreated", ""),
        ("ApprovalTokenIssued", ""),
        ("OrderTransitionLedgered", "APPROVE"),
        ("OrderTransitionLedgered", "REQUEST_SUBMISSION"),
        ("OrderTransitionLedgered", "SUBMIT"),
        ("OrderTransitionLedgered", "ACK"),
        ("ModeHeartbeat", ""),
        ("EquitySampled", ""),
        ("PositionsSnapshotRecorded", ""),
    ]
    rows = _rows(deps)
    assert first.intent_count == 1
    assert first.filled_centis == 0
    assert second.intent_count == 1
    assert second.filled_centis == SIZED_FILL_CENTIS
    assert _only(rows, "IntentVetoed")["reasons"] == FRESH_LEDGER_VETO_REASONS
    approved = _only(rows, "IntentApproved")
    assert approved["reasons"] == []
    token = _only(rows, "ApprovalTokenIssued")
    assert token["intent_id"] == approved["intent_id"]
    assert token["sequence_number"] == FIRST_TOKEN_SEQUENCE_NUMBER
    assert _only(rows, "ReservationCreated")["intent_id"] == approved["intent_id"]
    # The venue really holds what the tick reported filling, so a fill the
    # gateway never acked cannot pass here.
    assert _last(rows, "PositionsSnapshotRecorded")["positions"] == [
        {
            "ticker": TICKER,
            "quantity_centis": SIZED_FILL_CENTIS,
            "average_price_pips": BEST_CASE_ASK_PIPS,
        }
    ]
    # Nothing rests, so `ACKED` is where this order's lifecycle ends and the
    # Reconciler has nothing to diff.
    assert deps.gateway.tracked_orders() == ()


def test_every_spec_10_3_check_passes_under_the_context_the_fill_used(
    tmp_path: Path,
) -> None:
    """All 24 checks pass on beat 2; exactly two veto on beat 1.

    Issue #450's acceptance criterion 3, measured rather than described. The
    approval seam is wrapped in a pass-through observer that forwards every call
    to the real :class:`~windbreak.scheduler.loop.KernelApproval` and returns its
    outcome unmodified, purely to retain the ``(intent, context)`` pair
    ``_approve_stage`` otherwise discards. That the wrapper changes nothing is
    itself asserted: beat 1's forwarded outcome carries no token and beat 2's
    does, which is the same pair of verdicts the unwrapped run above ledgers.

    The verdict lists are *derived* from
    :data:`~windbreak.riskkernel.checks.DEFAULT_CHECKS` -- every check the
    kernel actually runs, in its own order -- not transcribed, so a
    twenty-fifth check added later is covered here on the day it lands rather
    than silently omitted. They are cross-checked against the kernel's own
    ledgered verdict, which is the guard against the derivation quietly
    measuring something other than what the approval decided.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    deps = _emitting_deps(tmp_path)
    recorder = _RecordingApprovalSeam(deps.approval)
    observed = dataclasses.replace(deps, approval=recorder)

    run_single_tick(observed, beat=1)
    second = run_single_tick(observed, beat=2)

    check_names = tuple(check.name for check in DEFAULT_CHECKS)
    assert check_names != (), "no checks scanned: every verdict below is vacuous"
    assert set(BEAT_ONE_VETOING_CHECKS) <= set(check_names)
    assert len(recorder.calls) == 2
    (beat_one, beat_two) = recorder.calls
    assert beat_one[2].token is None
    assert beat_two[2].token is not None
    first_passing, first_vetoing = _check_verdicts(beat_one[0], beat_one[1])
    second_passing, second_vetoing = _check_verdicts(beat_two[0], beat_two[1])
    assert first_vetoing == BEAT_ONE_VETOING_CHECKS
    assert first_passing == tuple(
        name for name in check_names if name not in BEAT_ONE_VETOING_CHECKS
    )
    assert second_vetoing == ()
    assert second_passing == check_names
    # The kernel's own audit rows must agree with the recomputation above.
    rows = _rows(observed)
    assert _only(rows, "IntentVetoed")["reasons"] == FRESH_LEDGER_VETO_REASONS
    assert _only(rows, "IntentApproved")["reasons"] == []
    assert second.filled_centis == SIZED_FILL_CENTIS


def test_only_the_correlation_declaration_and_state_dir_leave_the_defaults(
    tmp_path: Path,
) -> None:
    """No risk, screener or capital threshold moved to reach the fill.

    Issue #450's acceptance criterion 3's second half: "loosening ``RiskConfig``
    until an intent squeaks through is a tautology". The claim is derived by
    walking every field of :class:`~windbreak.config.schema.WindbreakConfig` and
    comparing it against the shipped default, so a future edit that quietly
    relaxes a threshold shows up here as a third differing field rather than
    passing a hand-maintained list.

    The scan's non-vacuity is asserted first: ``risk``, ``capital`` and
    ``screener`` must genuinely be among the fields walked, since a comparison
    that never reached them would report "nothing moved" forever.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    config = _emitting_config(tmp_path / "state")
    defaults = WindbreakConfig()

    walked = tuple(field.name for field in dataclasses.fields(config))
    differing = tuple(
        name for name in walked if getattr(config, name) != getattr(defaults, name)
    )

    assert {"risk", "capital", "screener"} <= set(walked)
    assert differing == NON_DEFAULT_CONFIG_FIELDS
    assert config.risk == defaults.risk
    assert config.capital == defaults.capital
    assert config.screener == defaults.screener
    assert config.ops.state_dir == str(tmp_path / "state")
    assert dataclasses.replace(config.ops, state_dir=defaults.ops.state_dir) == (
        defaults.ops
    )


def test_the_submission_chain_ledgers_no_fill_edge_for_an_outright_cross(
    tmp_path: Path,
) -> None:
    """An order that rests nothing stops at ``ACKED``, and that is the whole set.

    Pinned separately from the golden sequence because it is a claim about what
    the ledger *cannot* contain, and a golden sequence read as "these rows
    appear" would not say it. The submission chain's four edges are the complete
    transition set for this order: no ``FILL``, no ``RECONCILE``, and nothing
    tracked for the Reconciler to later heal.

    Recorded as observed behaviour, not as a design endorsement. The gateway
    tracks only a resting remainder, so the ``FILL`` edge -- which exists in the
    state machine and is reachable via
    :class:`~windbreak.order_gateway.reconciler.Reconciler` -- is simply never
    reached by an outright cross.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    deps = _emitting_deps(tmp_path)

    run_single_tick(deps, beat=1)
    run_single_tick(deps, beat=2)

    transitions = tuple(
        str(data["event"])
        for event_type, data in _rows(deps)
        if event_type == "OrderTransitionLedgered"
    )
    coids = {
        str(data["client_order_id"])
        for event_type, data in _rows(deps)
        if event_type == "OrderTransitionLedgered"
    }
    assert transitions == SUBMISSION_TRANSITIONS
    assert len(coids) == 1
    assert deps.gateway.tracked_orders() == ()
    assert deps.gateway.halted is False
