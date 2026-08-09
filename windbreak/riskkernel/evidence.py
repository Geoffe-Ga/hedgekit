"""Gate-plan-anchored ``paper_window_days`` evidence producer (SPEC S13.6).

This leaf module is the anti-Goodhart anchoring seam for the PAPER -> LIVE_MICRO
promotion gate (issue #243, follow-up to #185). The gate's ``paper_window_days``
criterion (SPEC S10.9, ``GE 90``) is meant to measure elapsed whole days *since
the currently-registered gate plan's ``paper_clock_start``* -- so a
``GatePlanChanged`` reset (SPEC S13.6) always shortens the effective window on
the very next evidence snapshot. Left to callers, ``GateEvidence.paper_window_days``
is untrusted input that can carry a stale value computed against a plan that has
since been superseded; this module derives it authoritatively from the ledger
instead, failing closed whenever no verified anchor is available.

**Scope**: only ``paper_window_days`` is anchored to ``paper_clock_start``. The
other elapsed-time evidence fields -- ``live_micro_days`` and
``days_without_unhandled_errors`` -- measure against different clocks (the
live-micro start and the last unhandled-error reset, respectively) and are
deliberately out of scope for this producer.

The package boundary matters: this module lives in ``riskkernel`` (not
``evaluation``) because the only legal cross-package edge is ``riskkernel ->
evaluation`` (``process.py`` already imports ``evaluation.preregistration``). An
``evaluation``-side helper importing ``riskkernel`` would invert that edge.

The fail-closed error mapping here follows
:meth:`windbreak.riskkernel.process.RiskKernel._paper_gate_from_registered_plan`:
an absent store, an empty ledger, and an unreadable (corrupt/tampered)
registration each raise
:class:`~windbreak.riskkernel.promotion.GatePlanUnavailableError`, exactly as
there. This producer adds one further case that method has no equivalent for --
backwards clock skew (a ``now`` behind the anchor) -- which fails closed the
same way.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final

from windbreak.evaluation.preregistration import latest_gate_plan_registration
from windbreak.riskkernel.promotion import GateEvidence, GatePlanUnavailableError

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.ledger.store import LedgerStore

#: Whole epoch seconds in one paper-clock day. Anchoring is pure integer
#: epoch-second arithmetic (SPEC S6.1, float-free); this names the floor-division
#: divisor so the day-count math reads intent, not a magic literal. Defined
#: locally rather than reaching into ``windbreak.screener.filters``'s
#: module-private ``_SECONDS_PER_DAY``.
SECONDS_PER_DAY: Final = 86_400


def anchored_paper_window_days(
    store: LedgerStore | None, *, now: Callable[[], int]
) -> int:
    """Return whole days elapsed since the registered plan's ``paper_clock_start``.

    Reads the ledger's *latest* gate-plan registration and floors the elapsed
    epoch-second delta into complete days (SPEC S10.9 / S13.6). A
    ``GatePlanChanged`` reset re-anchors the window to the new plan's own
    ``paper_clock_start``, so a superseded plan can never keep a stale window
    alive (issues #185, #243). Fails closed -- raising rather than returning a
    negative or clamped value -- whenever no verified anchor is available.

    ``now`` is called exactly once and the returned epoch reused, so the reading
    is internally consistent even if the caller's clock is non-constant.

    Args:
        store: The wired ledger to read the anchor from, or ``None`` when no
            plan store is available.
        now: A zero-argument callable returning the current time in whole epoch
            seconds. Called exactly once.

    Returns:
        ``(now() - paper_clock_start) // SECONDS_PER_DAY`` -- the floored count
        of complete elapsed days as a plain, non-negative ``int``. Only whole
        elapsed days count; a partial trailing day is dropped.

    Raises:
        GatePlanUnavailableError: If ``store`` is ``None``, the ledger holds no
            registration, the latest registration is unreadable (the underlying
            ``ValueError``/``TypeError`` is chained as ``__cause__``), or the
            clock is behind the anchor (backwards skew is fail-closed).
    """
    if store is None:
        raise GatePlanUnavailableError(
            "no gate plan store wired; promotion evidence cannot be anchored "
            "(fail-closed)"
        )
    try:
        registration = latest_gate_plan_registration(store)
    except (ValueError, TypeError) as err:
        raise GatePlanUnavailableError(
            "registered gate plan is unreadable; promotion evidence cannot be "
            "anchored (fail-closed)"
        ) from err
    if registration is None:
        raise GatePlanUnavailableError(
            "no registered gate plan; promotion evidence cannot be anchored "
            "(fail-closed)"
        )
    elapsed = now() - registration.paper_clock_start
    if elapsed < 0:
        raise GatePlanUnavailableError(
            "clock is behind the registered paper_clock_start; promotion "
            "evidence cannot be anchored (fail-closed)"
        )
    return elapsed // SECONDS_PER_DAY


def anchor_gate_evidence(
    evidence: GateEvidence,
    store: LedgerStore | None,
    *,
    now: Callable[[], int],
) -> GateEvidence:
    """Return ``evidence`` with ``paper_window_days`` re-derived from the ledger.

    The caller-supplied ``paper_window_days`` is never trusted: it is
    unconditionally overwritten with the anchored value from
    :func:`anchored_paper_window_days` (SPEC S13.6, anti-Goodhart; issues #185,
    #243). Every other field is preserved byte-identically, and the input
    ``GateEvidence`` (frozen) is left untouched -- a new instance is returned via
    :func:`dataclasses.replace`.

    Args:
        evidence: The promotion-readiness snapshot whose ``paper_window_days``
            is to be anchored. Not mutated.
        store: The wired ledger to read the anchor from, or ``None``.
        now: A zero-argument callable returning the current time in whole epoch
            seconds.

    Returns:
        A new :class:`~windbreak.riskkernel.promotion.GateEvidence` identical to
        ``evidence`` except for its anchored ``paper_window_days``.

    Raises:
        GatePlanUnavailableError: Propagated from
            :func:`anchored_paper_window_days` when no verified anchor is
            available (no store, no registration, an unreadable registration, or
            backwards clock skew).
    """
    return dataclasses.replace(
        evidence, paper_window_days=anchored_paper_window_days(store, now=now)
    )
