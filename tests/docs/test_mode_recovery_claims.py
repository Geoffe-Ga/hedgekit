"""The runbook's mode-recovery claims are replayed against the machine (#526).

`docs/RUNBOOK.md` said `windbreak kill`/`windbreak rearm` "do stop and re-arm
the PAPER loop". The first half is true. The second half is true only after a
**process restart**: `KillSwitch.rearm` exits `KILLED` into `PAUSED`, and
nothing shipped returns a running process to `PAPER`.

Correcting that in prose alone would have reproduced the phantom-gate class
this repository has closed five times (#359, #401, #351, #411, #449): a
document describing a control the code does not implement, believed for weeks
because it was written down, with nothing failing when the two disagreed. The
issue's own acceptance criterion says so -- "the runbook's claim is corrected
**and pinned by a test**".

So the claim is carried by a **derived** table rather than by a sentence. The
runbook's marker-delimited "in-process mode successors" region lists, per mode,
the modes a running process can still reach one transition at a time; this
module recomputes that mapping by driving :class:`ModeStateMachine` itself --
every mode as a source, every mode as a target, plus the one non-`transition`
door (`rearm`, out of `KILLED`) -- and compares the two in **both** directions.
A row the machine contradicts fails, and a successor the machine permits that
the table omits fails just as loudly. The day someone adds a `PAUSED -> PAPER`
transition, this file turns RED and the runbook's restart instruction has to be
rewritten on the same commit.

The consequence the table exists to support is asserted directly and
separately, so it does not depend on a reader drawing the inference:
:func:`test_no_trading_mode_is_reachable_from_paused_at_any_depth` closes the
successor relation transitively, which is strictly stronger than the table's
one-step rows -- a two-step escape (`PAUSED -> HALT -> PAPER`) would satisfy
every row above and still make the runbook wrong.

Three traps this module is shaped around:

* **A corpus scan over zero hits passes forever.** The table region is asserted
  non-empty and asserted to cover every member of :class:`Mode`, so a marker
  typo or a parser that silently returned nothing fails instead of certifying.
* **A guard comparing the wrong dimension.** The positive control
  (:func:`test_a_trading_mode_is_reachable_from_a_trading_mode`) proves the
  reachability closure can answer "yes": from `PAPER` under a `LIVE` ceiling it
  finds `LIVE_MICRO` and `LIVE`. Without it, a closure that always returned the
  empty set would make the `PAUSED` assertions vacuous.
* **A hand-restated list.** Nothing here transcribes the transition table.
  Every expectation is computed from `ModeStateMachine`'s own behaviour, by
  calling it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from windbreak.riskkernel.modes import (
    REARM_CONFIRMATION_PHRASE,
    IllegalModeTransitionError,
    KillReArmError,
    Mode,
    ModeStateMachine,
)

#: The runbook region the successor table lives in. The markers are HTML
#: comments so they are invisible to a reader and unmissable to this parser.
_BEGIN_MARKER = "<!-- BEGIN in-process mode successors -->"
_END_MARKER = "<!-- END in-process mode successors -->"

#: The runbook, relative to the repo root.
_RUNBOOK_RELPATH = "docs/RUNBOOK.md"

#: Every mode token inside a code span in one table cell.
_CODE_SPAN = re.compile(r"`([A-Z_]+)`")

#: The ceiling every machine below is built with. `LIVE` is the *permissive*
#: choice on purpose: a lower ceiling would suppress promotions and make the
#: table look narrower than the machine really is, so the successors recorded
#: here are the widest set any deployment can reach.
_CEILING = Mode.LIVE


def _repo_root() -> Path:
    """Return the repo root, resolved from this test file's location.

    Returns:
        The repo root (this file's great-grandparent directory).
    """
    return Path(__file__).resolve().parents[2]


def _runbook_text() -> str:
    """Return the whole runbook as text.

    Returns:
        The contents of ``docs/RUNBOOK.md``.
    """
    return (_repo_root() / _RUNBOOK_RELPATH).read_text(encoding="utf-8")


def _documented_successors() -> dict[Mode, frozenset[Mode]]:
    """Parse the runbook's marker-delimited successor table.

    Only the mode tokens inside code spans are read, so the prose in a cell
    (``by `windbreak rearm` only``) annotates a row without changing what it
    claims. A row naming a token that is not a :class:`Mode` raises rather than
    being skipped: a typo in this table must fail, never narrow the comparison.

    Returns:
        Each documented source mode mapped to its documented successor set.

    Raises:
        AssertionError: If the markers are missing or the region holds no rows.
    """
    text = _runbook_text()
    assert _BEGIN_MARKER in text, f"{_RUNBOOK_RELPATH} lost {_BEGIN_MARKER}"
    assert _END_MARKER in text, f"{_RUNBOOK_RELPATH} lost {_END_MARKER}"
    region = text.split(_BEGIN_MARKER, 1)[1].split(_END_MARKER, 1)[0]
    documented: dict[Mode, frozenset[Mode]] = {}
    for line in region.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 2:
            continue
        sources = _CODE_SPAN.findall(cells[0])
        if len(sources) != 1:
            continue
        documented[Mode[sources[0]]] = frozenset(
            Mode[token] for token in _CODE_SPAN.findall(cells[1])
        )
    assert documented, (
        f"the {_BEGIN_MARKER} region of {_RUNBOOK_RELPATH} parsed to no rows at "
        "all, so every comparison below would be vacuous"
    )
    return documented


def _successors(mode: Mode) -> frozenset[Mode]:
    """Return every mode a running process in ``mode`` can reach in one step.

    Computed by *driving* :class:`ModeStateMachine`, never by reading its
    tables: a fresh machine is built per candidate target, the transition is
    attempted, and the target counts as a successor exactly when the machine
    accepts it. `KillSwitch.rearm`'s door out of `KILLED` is exercised through
    the machine's own :meth:`~ModeStateMachine.rearm` for the same reason.

    `KILLED` is deliberately handled through both doors. Its ordinary
    transitions are all illegal, and its `rearm` lands in `RESEARCH`; the
    operator-visible exit the runbook documents is `PAUSED`, because
    `KillSwitch.rearm` composes that `rearm` with a transition to `PAUSED` in
    one operation. Recording the composed destination is what makes the table
    describe the shipped verb rather than a primitive no operator can call.

    Args:
        mode: The source mode.

    Returns:
        The modes reachable from ``mode`` in one operator-visible step.
    """
    reached: set[Mode] = set()
    for target in Mode:
        machine = ModeStateMachine(mode_ceiling=_CEILING, mode=mode)
        try:
            machine.transition(target)
        except IllegalModeTransitionError:
            continue
        reached.add(machine.mode)
    if mode is Mode.KILLED:
        machine = ModeStateMachine(mode_ceiling=_CEILING, mode=Mode.KILLED)
        machine.rearm(REARM_CONFIRMATION_PHRASE)
        machine.transition(Mode.PAUSED)
        reached.add(machine.mode)
    return frozenset(reached)


def _reachable_from(mode: Mode) -> frozenset[Mode]:
    """Return every mode reachable from ``mode`` at any depth.

    The transitive closure of :func:`_successors`, excluding the source itself
    unless a cycle genuinely returns to it. One step is not enough to support
    the runbook's claim: a `PAUSED -> HALT -> PAPER` escape would leave every
    documented row correct and the restart instruction wrong.

    Args:
        mode: The source mode.

    Returns:
        Every mode reachable from ``mode`` through one or more transitions.
    """
    frontier = set(_successors(mode))
    reached: set[Mode] = set()
    while frontier:
        current = frontier.pop()
        if current in reached:
            continue
        reached.add(current)
        frontier |= set(_successors(current)) - reached
    return frozenset(reached)


def test_the_runbook_table_covers_every_mode() -> None:
    """The documented table names all seven modes, so no row can be missing.

    The non-vacuity guard. A table listing only the modes that happen to agree
    with the machine would pass every comparison below while saying nothing
    about the ones that do not.
    """
    assert set(_documented_successors()) == set(Mode)


@pytest.mark.parametrize("mode", list(Mode), ids=lambda mode: mode.name)
def test_each_documented_row_is_the_machines_own_successor_set(mode: Mode) -> None:
    """Every runbook row equals what `ModeStateMachine` actually permits.

    Compared as whole sets, in both directions at once: a successor the runbook
    invents fails, and a successor the machine permits that the runbook omits
    fails identically. That second direction is the one that matters for issue
    #526 -- a `PAUSED -> PAPER` transition added later must redden this file
    rather than silently making the restart instruction obsolete.

    Args:
        mode: The source mode whose row is under test.
    """
    assert _documented_successors()[mode] == _successors(mode)


def test_no_trading_mode_is_reachable_from_paused_at_any_depth() -> None:
    """A `PAUSED` process cannot reach a trading mode, however many steps.

    The claim the runbook's restart instruction rests on, asserted directly
    rather than left to be inferred from the rows. `KillSwitch.rearm` exits
    `KILLED` into `PAUSED` deliberately, so this is what "only a restart
    returns the loop to `PAPER`" means mechanically.
    """
    reachable = _reachable_from(Mode.PAUSED)
    assert reachable == frozenset({Mode.HALT, Mode.KILLED, Mode.PAUSED})
    assert not any(mode.may_trade() for mode in reachable)


def test_no_trading_mode_is_reachable_from_halt_at_any_depth() -> None:
    """A halted process cannot reach a trading mode either.

    The same property for `HALT`, which is why issue #526 gave the two modes
    one gate rather than two: `loop.py` justified excluding `HALT` from the
    walk gate on the grounds that "a halted kernel is expected to recover", and
    the machine says it cannot.
    """
    assert not any(mode.may_trade() for mode in _reachable_from(Mode.HALT))


def test_a_trading_mode_is_reachable_from_a_trading_mode() -> None:
    """The positive control: the closure can answer "yes".

    Without this, a `_reachable_from` that always returned the empty set would
    make both assertions above pass while proving nothing at all.
    """
    reachable = _reachable_from(Mode.PAPER)
    assert Mode.LIVE_MICRO in reachable
    assert Mode.LIVE in reachable
    assert any(mode.may_trade() for mode in reachable)


def test_the_runbook_states_the_restart_requirement_in_prose_too() -> None:
    """The operator-facing sentence exists, next to the table that proves it.

    An operator reads the paragraph, not the transition table. Both are
    required: the table cannot drift because it is derived, and the sentence
    cannot go missing because it is asserted here.
    """
    text = _runbook_text()
    assert "### Re-arming, and why it needs a restart" in text
    assert "**A re-arm does not resume trading.**" in text
    assert "**To return the loop to `PAPER`, restart the process.**" in text


def test_the_machine_still_refuses_a_rearm_that_is_not_from_killed() -> None:
    """`rearm` is a `KILLED`-only door, so the table's one exception is bounded.

    :func:`_successors` grants `KILLED` an extra successor no ordinary
    transition provides. If `rearm` worked from any other mode that grant would
    be understating the machine, and every other row would be wrong.
    """
    machine = ModeStateMachine(mode_ceiling=_CEILING, mode=Mode.PAUSED)
    with pytest.raises(KillReArmError, match="only valid from KILLED"):
        machine.rearm(REARM_CONFIRMATION_PHRASE)
    assert machine.mode is Mode.PAUSED
