"""Operating-mode state machine for the Risk Kernel (SPEC S5.1-S5.3).

The Risk Kernel advances through a seven-state ladder: the four promotable
runtime modes ``RESEARCH -> PAPER -> LIVE_MICRO -> LIVE`` (each reachable only
one rung at a time and only up to a configured ``mode_ceiling``), plus the
three always-reachable safety modes ``PAUSED``, ``HALT``, and ``KILLED``. Once
in ``KILLED`` the machine is a dead end: no ordinary transition escapes it, and
the only way back to ``RESEARCH`` is :meth:`ModeStateMachine.rearm` with an
exact typed-confirmation phrase.

The legal-transition rules are held in the module-level
:data:`_ALLOWED_TRANSITIONS` table (a mapping of each source mode to the set of
targets legal *ignoring the ceiling*), so :meth:`ModeStateMachine.transition`
is a small, table-driven lookup plus a single dynamic ceiling check rather than
a sprawling branch tree.
"""

from __future__ import annotations

import enum

#: The exact confirmation an operator must type to re-arm a KILLED kernel. It
#: deliberately contains cased characters, so a case-folded near-miss (compared
#: verbatim by :meth:`ModeStateMachine.rearm`) is rejected.
REARM_CONFIRMATION_PHRASE = "RE-ARM AFTER KILL: I ACCEPT FULL RESPONSIBILITY"


class IllegalModeTransitionError(Exception):
    """Raised when a requested mode transition is not permitted."""


class ModeCeilingExceededError(IllegalModeTransitionError):
    """Raised when a promotion would exceed the configured ``mode_ceiling``.

    A more specific :class:`IllegalModeTransitionError`: the transition would be
    legal on the ladder but is blocked solely because the target rung sits
    above the runtime's permitted ceiling.
    """


class KillReArmError(Exception):
    """Raised when a re-arm is attempted from the wrong state or phrase."""


class Mode(enum.Enum):
    """The seven Risk Kernel operating modes, in SPEC promotion order.

    The first four members form the promotable ladder (``RESEARCH`` up to
    ``LIVE``); the final three are the safety modes reachable from any
    non-``KILLED`` state.
    """

    RESEARCH = enum.auto()
    PAPER = enum.auto()
    LIVE_MICRO = enum.auto()
    LIVE = enum.auto()
    PAUSED = enum.auto()
    HALT = enum.auto()
    KILLED = enum.auto()

    @classmethod
    def from_config(cls, token: str) -> Mode:
        """Parse a SPEC S16 ``mode_ceiling`` token into a promotable mode.

        Only the four promotable ladder tokens (``research``, ``paper``,
        ``live_micro``, ``live``) are valid ceilings; safety-mode tokens,
        wrong case, empty, and unknown input are all rejected.

        Args:
            token: The lowercase ceiling token from configuration.

        Returns:
            The :class:`Mode` the token names.

        Raises:
            ValueError: If ``token`` is not one of the four ceiling tokens.
        """
        try:
            return _CONFIG_TOKENS[token]
        except KeyError:
            raise ValueError(f"not a valid mode_ceiling token: {token!r}") from None

    def may_trade(self) -> bool:
        """Return whether an **order may be routed** while the kernel is in this mode.

        The single definition of "this mode can trade", so no caller has to
        restate the set. Its one consumer is
        :class:`~windbreak.riskkernel.checks._ModePermissionCeiling`, which
        vetoes an intent minted in a non-trading mode.

        **Do not reuse this for spend decisions.** It answers "may an order be
        routed", not "is a forecast worth buying", and those two questions have
        different answers -- see :meth:`may_research`, which is the predicate
        the tick's walk gate asks.

        Returns:
            True for the three trading modes (``PAPER``, ``LIVE_MICRO``,
            ``LIVE``); False for ``RESEARCH`` and every safety mode.
        """
        return self in TRADING_MODES

    def may_research(self) -> bool:
        """Return whether **research money may be spent** in this mode (issue #526).

        The predicate :func:`~windbreak.scheduler.loop.run_single_tick`'s walk
        gate asks before it walks the screened universe -- the walk being where
        the loop pays for forecasts. Derived from :data:`_SAFETY_MODES` rather
        than listed, so it is exactly "this mode is on the promotable ladder":
        a safety mode is one the system entered in order to *stop* doing
        something, and a new safety mode added later is excluded automatically
        rather than by remembering to edit a set here.

        Two failures shaped this, one in each direction, and both are worth
        having in front of you before you touch this:

        * **Restating the trading set at the walk gate cost real money.** The
          gate named ``KILLED`` alone, so a re-armed loop sitting in ``PAUSED``
          bought a forecast per market per beat and then vetoed every intent
          those forecasts produced, drawing the whole day down against the
          durable per-UTC-day research ceiling (issues #442/#483) for a day in
          which nothing could be traded. That is issue #526.

        * **Answering it with :meth:`may_trade` is the opposite error, and the
          counterexample is ``RESEARCH``.** The two predicates agree on
          ``PAPER``, ``LIVE_MICRO``, ``LIVE``, ``PAUSED``, ``HALT`` and
          ``KILLED``, and disagree on exactly one mode -- ``RESEARCH``, which
          may **not** trade and **must** research. It is SPEC S5.1's bottom
          rung: its entire purpose is to produce forecasts without routing
          them, so that promotion is earned rather than granted, and
          ``_research_to_paper_gate``'s ``research_min_forecasts`` criterion
          reads that production. A ``RESEARCH`` loop that bought no forecasts
          could never satisfy the gate that promotes it out of ``RESEARCH``.
          ``tests/riskkernel/test_modes.py`` pins the divergence set to
          ``{RESEARCH}`` so this paragraph cannot quietly stop being true.

        The safety modes are the agreement: ``PAUSED``, ``HALT`` and ``KILLED``
        neither trade nor research, and none of them is a source for any
        promotion gate -- ``_ALLOWED_TRANSITIONS`` gives all three no path to a
        trading mode at all, so no forecast bought in one of them could ever be
        acted on or counted toward leaving it.

        Returns:
            True for the four promotable ladder modes (``RESEARCH``, ``PAPER``,
            ``LIVE_MICRO``, ``LIVE``); False for ``PAUSED``, ``HALT`` and
            ``KILLED``.
        """
        return self not in _SAFETY_MODES


#: The modes an order may be routed in. The one definition of that set; read it
#: through :meth:`Mode.may_trade` rather than by membership, so the question is
#: asked the same way everywhere (issue #526). This is deliberately *not* the
#: set of modes that may spend research money -- see :meth:`Mode.may_research`,
#: which differs at ``RESEARCH`` and is what a spend decision must ask.
TRADING_MODES: frozenset[Mode] = frozenset({Mode.PAPER, Mode.LIVE_MICRO, Mode.LIVE})

#: The promotable ladder, low to high. Position is the promotion rank used for
#: one-step-up checks and ceiling comparison; the safety modes are off-ladder.
_PROMOTION_LADDER: tuple[Mode, ...] = (
    Mode.RESEARCH,
    Mode.PAPER,
    Mode.LIVE_MICRO,
    Mode.LIVE,
)

#: Each ladder mode's promotion rank (0 = RESEARCH ... 3 = LIVE).
_LADDER_RANK: dict[Mode, int] = {
    mode: rank for rank, mode in enumerate(_PROMOTION_LADDER)
}

#: The three safety modes reachable from any non-KILLED mode.
_SAFETY_MODES: frozenset[Mode] = frozenset({Mode.PAUSED, Mode.HALT, Mode.KILLED})

#: Maps each SPEC S16 ceiling token to its promotable mode.
_CONFIG_TOKENS: dict[str, Mode] = {
    "research": Mode.RESEARCH,
    "paper": Mode.PAPER,
    "live_micro": Mode.LIVE_MICRO,
    "live": Mode.LIVE,
}


def _next_rung(mode: Mode) -> Mode | None:
    """Return the one-rung-up promotion target for a mode, if any.

    Args:
        mode: The candidate source mode.

    Returns:
        The next ladder mode above ``mode``, or ``None`` if ``mode`` is
        off-ladder or already at the top rung.
    """
    rank = _LADDER_RANK.get(mode)
    if rank is None or rank + 1 >= len(_PROMOTION_LADDER):
        return None
    return _PROMOTION_LADDER[rank + 1]


def _prev_rung(mode: Mode) -> Mode | None:
    """Return the one-rung-down demotion target for a mode, if any.

    Args:
        mode: The candidate source mode.

    Returns:
        The next ladder mode below ``mode``, or ``None`` if ``mode`` is
        off-ladder or already at the bottom rung (``RESEARCH``).
    """
    rank = _LADDER_RANK.get(mode)
    if rank is None or rank == 0:
        return None
    return _PROMOTION_LADDER[rank - 1]


def _build_allowed_transitions() -> dict[Mode, frozenset[Mode]]:
    """Build the source-to-legal-targets table, ignoring the ceiling.

    Every non-``KILLED`` mode may move to any safety mode other than itself,
    and each ladder mode may additionally promote one rung up. ``KILLED`` is a
    dead end with no legal ordinary transitions.

    Returns:
        A mapping of each :class:`Mode` to the frozenset of targets that are
        legal from it before the dynamic ceiling check is applied.
    """
    allowed: dict[Mode, frozenset[Mode]] = {}
    for mode in Mode:
        if mode is Mode.KILLED:
            allowed[mode] = frozenset()
            continue
        targets = set(_SAFETY_MODES - {mode})
        next_mode = _next_rung(mode)
        if next_mode is not None:
            targets.add(next_mode)
        allowed[mode] = frozenset(targets)
    return allowed


#: The static legal-transition table: source mode -> targets legal ignoring the
#: ceiling. The ceiling is enforced separately by :meth:`ModeStateMachine`.
_ALLOWED_TRANSITIONS: dict[Mode, frozenset[Mode]] = _build_allowed_transitions()


class ModeStateMachine:
    """A guarded state machine over the seven Risk Kernel modes.

    Transitions are validated against :data:`_ALLOWED_TRANSITIONS` and the
    machine's ``mode_ceiling``; an illegal request raises without mutating the
    current mode.

    Attributes:
        mode: The current operating mode (read-only).
    """

    def __init__(self, mode_ceiling: Mode, mode: Mode = Mode.RESEARCH) -> None:
        """Initialize the machine.

        Args:
            mode_ceiling: The highest ladder rung the runtime may ever promote
                to. Must be one of the four promotable ladder modes.
            mode: The starting mode. Defaults to ``RESEARCH``.

        Raises:
            ValueError: If ``mode_ceiling`` is a safety mode rather than one of
                the four promotable ladder modes -- the ceiling only bounds the
                ladder, so a safety ceiling is meaningless and would otherwise
                surface as a raw ``KeyError`` on the first promotion.
        """
        if mode_ceiling not in _LADDER_RANK:
            raise ValueError(
                "mode_ceiling must be a promotable ladder mode, got "
                f"{mode_ceiling.name}"
            )
        self._mode_ceiling = mode_ceiling
        self._mode = mode

    @property
    def mode(self) -> Mode:
        """Return the current operating mode."""
        return self._mode

    @property
    def mode_ceiling(self) -> Mode:
        """Return the configured highest promotable ladder rung."""
        return self._mode_ceiling

    def _exceeds_ceiling(self, target: Mode) -> bool:
        """Return whether promoting to ``target`` would exceed the ceiling.

        Off-ladder (safety) targets are never ceiling-bounded, since the
        ceiling only caps the promotable ladder.

        Args:
            target: The requested target mode.

        Returns:
            True if ``target`` is a ladder mode ranked above the ceiling.
        """
        target_rank = _LADDER_RANK.get(target)
        if target_rank is None:
            return False
        return target_rank > _LADDER_RANK[self._mode_ceiling]

    def transition(self, target: Mode) -> None:
        """Move to ``target`` if the transition is legal, else raise.

        Args:
            target: The requested next mode.

        Raises:
            ModeCeilingExceededError: If the (otherwise legal) promotion would rise
                above ``mode_ceiling``.
            IllegalModeTransitionError: If the transition is not in the legal table
                (a same-mode move, a skipped rung, a demotion off the ladder,
                or any move out of ``KILLED``).
        """
        if target not in _ALLOWED_TRANSITIONS[self._mode]:
            raise IllegalModeTransitionError(
                f"illegal transition {self._mode.name} -> {target.name}"
            )
        if self._exceeds_ceiling(target):
            raise ModeCeilingExceededError(
                f"promotion to {target.name} exceeds ceiling {self._mode_ceiling.name}"
            )
        self._mode = target

    def promote_one_rung(self, *, effective_ceiling: Mode) -> Mode:
        """Promote exactly one ladder rung, honoring both ceilings.

        A narrower, ceiling-aware convenience path layered on top of the static
        :data:`_ALLOWED_TRANSITIONS` table: it never widens the legal-transition
        matrix, only advances the machine one rung up the promotion ladder when
        both the static ``mode_ceiling`` and the caller-supplied
        ``effective_ceiling`` permit the target rung.

        Args:
            effective_ceiling: A second, dynamically-computed ceiling (e.g. a
                significance-override cap) the target rung must also satisfy.

        Returns:
            The new (higher) mode.

        Raises:
            IllegalModeTransitionError: If there is no rung above the current
                mode (already at ``LIVE``, or in a safety mode). The mode is
                left unchanged.
            ModeCeilingExceededError: If a next rung exists but sits above the
                static ``mode_ceiling`` *or* the ``effective_ceiling``. The mode
                is left unchanged.
        """
        target = _next_rung(self._mode)
        if target is None:
            raise IllegalModeTransitionError(
                f"no ladder rung above {self._mode.name} to promote to"
            )
        target_rank = _LADDER_RANK[target]
        if (
            target_rank > _LADDER_RANK[self._mode_ceiling]
            or target_rank > _LADDER_RANK[effective_ceiling]
        ):
            raise ModeCeilingExceededError(
                f"promotion to {target.name} exceeds an active ceiling"
            )
        self._mode = target
        return target

    def demote_one_rung(self) -> Mode:
        """Demote exactly one ladder rung downward.

        The low-level state-machine primitive stepping ``LIVE -> LIVE_MICRO ->
        PAPER -> RESEARCH`` one rung at a time. Unlike the demotion-trigger
        policy built on top of it, this raises rather than floors: there is no
        rung below ``RESEARCH`` and none below any safety mode.

        Returns:
            The new (lower) mode.

        Raises:
            IllegalModeTransitionError: If there is no rung below the current
                mode (already at ``RESEARCH``, or in a safety mode including
                ``KILLED``). The mode is left unchanged.
        """
        target = _prev_rung(self._mode)
        if target is None:
            raise IllegalModeTransitionError(
                f"no ladder rung below {self._mode.name} to demote to"
            )
        self._mode = target
        return target

    def rearm(self, confirmation: str) -> None:
        """Re-arm a KILLED machine back to RESEARCH on exact confirmation.

        Args:
            confirmation: The typed confirmation phrase; must equal
                :data:`REARM_CONFIRMATION_PHRASE` verbatim (no case folding).

        Raises:
            KillReArmError: If the machine is not in ``KILLED``, or the
                confirmation phrase does not match exactly. The mode is left
                unchanged.
        """
        if self._mode is not Mode.KILLED:
            raise KillReArmError("rearm is only valid from KILLED")
        if confirmation != REARM_CONFIRMATION_PHRASE:
            raise KillReArmError("rearm confirmation phrase does not match")
        self._mode = Mode.RESEARCH
