"""Demotion-trigger machinery for the Risk Kernel (SPEC S5.1, S10.x).

Sixteen named :class:`DemotionTrigger`\\s each map to exactly one of four
:class:`DemotionAction`\\s via the pinned :data:`TRIGGER_ACTIONS` table, and
:func:`resolve_demotion` resolves a ``(current_mode, trigger)`` pair to the
destination :class:`~windbreak.riskkernel.modes.Mode` (or ``None`` for a no-op).

Resolution is table-driven, mirroring the ``_ALLOWED_TRANSITIONS`` style of
:mod:`windbreak.riskkernel.modes`, and reuses that module's ladder-rung math
(:func:`~windbreak.riskkernel.modes._prev_rung`) rather than re-deriving it:

* ``PAUSE``/``HALT``/``KILL`` target their safety mode, except an idempotent
  same-mode no-op, and ``KILLED`` is a dead end for every action.
* ``DEMOTE_ONE_MODE`` steps down one ladder rung, floors at ``PAUSED`` from
  ``RESEARCH`` (fail-safe), and is a no-op off the ladder (``PAUSED``/``HALT``/
  ``KILLED``).

The kernel-level :meth:`RiskKernel.fire_demotion_trigger` builds on this pure
resolver to ledger each firing. That method has no production caller, so while
this resolver is correct and exhaustively tested, no shipped path fires a
trigger through it (issue #542).
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from windbreak.riskkernel.modes import Mode, _prev_rung

if TYPE_CHECKING:
    from collections.abc import Mapping


class DemotionAction(enum.Enum):
    """The four demotion responses a trigger can demand."""

    PAUSE = enum.auto()
    DEMOTE_ONE_MODE = enum.auto()
    HALT = enum.auto()
    KILL = enum.auto()


class DemotionTrigger(enum.Enum):
    """The sixteen named conditions that can demand a demotion (SPEC S10.x)."""

    DAILY_LOSS_BREACH = enum.auto()
    DRAWDOWN_BREACH = enum.auto()
    BALANCE_POSITION_MISMATCH = enum.auto()
    FLOOR_CHECK_FAILURE = enum.auto()
    SCHEMA_ANOMALY = enum.auto()
    JURISDICTION_UNKNOWN = enum.auto()
    ROLLING_BRIER_DEGRADATION = enum.auto()
    LIVE_PAPER_SLIPPAGE_DIVERGENCE = enum.auto()
    CLOCK_SKEW = enum.auto()
    STALE_HEARTBEAT = enum.auto()
    FEE_MODEL_UNAVAILABLE = enum.auto()
    CANARY_DRIFT_UNACKNOWLEDGED = enum.auto()
    TOKEN_REPLAY_ATTEMPT = enum.auto()
    BACKUP_FAILURES_BEYOND_LIMIT = enum.auto()
    DISK_BELOW_THRESHOLD = enum.auto()
    MANUAL_KILL = enum.auto()


#: The pinned trigger -> action table (SPEC S10.x). Exhaustive over every
#: :class:`DemotionTrigger`; a single misassignment is a policy change.
TRIGGER_ACTIONS: Mapping[DemotionTrigger, DemotionAction] = {
    DemotionTrigger.DAILY_LOSS_BREACH: DemotionAction.PAUSE,
    DemotionTrigger.DRAWDOWN_BREACH: DemotionAction.DEMOTE_ONE_MODE,
    DemotionTrigger.ROLLING_BRIER_DEGRADATION: DemotionAction.DEMOTE_ONE_MODE,
    DemotionTrigger.LIVE_PAPER_SLIPPAGE_DIVERGENCE: DemotionAction.DEMOTE_ONE_MODE,
    DemotionTrigger.CANARY_DRIFT_UNACKNOWLEDGED: DemotionAction.DEMOTE_ONE_MODE,
    DemotionTrigger.BALANCE_POSITION_MISMATCH: DemotionAction.HALT,
    DemotionTrigger.FLOOR_CHECK_FAILURE: DemotionAction.HALT,
    DemotionTrigger.SCHEMA_ANOMALY: DemotionAction.HALT,
    DemotionTrigger.JURISDICTION_UNKNOWN: DemotionAction.HALT,
    DemotionTrigger.CLOCK_SKEW: DemotionAction.HALT,
    DemotionTrigger.STALE_HEARTBEAT: DemotionAction.HALT,
    DemotionTrigger.FEE_MODEL_UNAVAILABLE: DemotionAction.HALT,
    DemotionTrigger.TOKEN_REPLAY_ATTEMPT: DemotionAction.HALT,
    DemotionTrigger.BACKUP_FAILURES_BEYOND_LIMIT: DemotionAction.HALT,
    DemotionTrigger.DISK_BELOW_THRESHOLD: DemotionAction.HALT,
    DemotionTrigger.MANUAL_KILL: DemotionAction.KILL,
}

#: Each safety action's target mode. ``DEMOTE_ONE_MODE`` is resolved by ladder
#: math instead and is deliberately absent here.
_SAFETY_ACTION_TARGET: Mapping[DemotionAction, Mode] = {
    DemotionAction.PAUSE: Mode.PAUSED,
    DemotionAction.HALT: Mode.HALT,
    DemotionAction.KILL: Mode.KILLED,
}


def _resolve_safety_action(action: DemotionAction, current: Mode) -> Mode | None:
    """Resolve a PAUSE/HALT/KILL action to its destination, or ``None``.

    Args:
        action: One of ``PAUSE``, ``HALT``, or ``KILL``.
        current: The current operating mode.

    Returns:
        The target safety mode, or ``None`` when already there (idempotent
        no-op) or when ``current`` is ``KILLED`` (a dead end for every action).
    """
    if current is Mode.KILLED:
        return None
    target = _SAFETY_ACTION_TARGET[action]
    if current is target:
        return None
    return target


def _resolve_demote_one_mode(current: Mode) -> Mode | None:
    """Resolve a DEMOTE_ONE_MODE action to its destination, or ``None``.

    Args:
        current: The current operating mode.

    Returns:
        ``PAUSED`` as a fail-safe floor from ``RESEARCH``; the next ladder rung
        down from a higher ladder mode; or ``None`` when off the ladder
        (``PAUSED``/``HALT``/``KILLED``, which have no rung below).
    """
    if current is Mode.RESEARCH:
        return Mode.PAUSED
    return _prev_rung(current)


def resolve_demotion(current: Mode, trigger: DemotionTrigger) -> Mode | None:
    """Resolve a firing to its destination mode, or ``None`` for a no-op.

    Args:
        current: The current operating mode.
        trigger: The firing trigger.

    Returns:
        The destination :class:`~windbreak.riskkernel.modes.Mode`, or ``None``
        when the firing is a no-op from ``current`` (already at the target
        safety mode, off the ladder for a demotion, or in ``KILLED``).
    """
    action = TRIGGER_ACTIONS[trigger]
    if action is DemotionAction.DEMOTE_ONE_MODE:
        return _resolve_demote_one_mode(current)
    return _resolve_safety_action(action, current)
