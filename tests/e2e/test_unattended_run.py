"""An accelerated unattended run: a restart, an induced fault, bounded spend (#473).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------

Issue #455's definition of done is a **seven-day unattended run**, and this
module does **not** replace it and cannot. Seven days of wall clock is not
something a merge gate can hold, and a compressed run says nothing about a slow
leak, a clock rollover, or a venue that changes under you. What this module is:
the standing regression guard that makes the seven-day run *likely to succeed*,
by crossing in seconds every **lifecycle boundary** the seven-day run crosses.
If one of these tests is red, the seven-day run will fail; if they are green, it
still might.

Epic #465's whole diagnosis is that no test in the suite crossed a process
boundary, so every property that is only true *across a restart* was
demonstrated in-process or not at all. Three fixes landed on exactly those
properties -- #442 (a durable per-UTC-day research ceiling), #441 (the kill
switch wired into the PAPER loop), #513/#514 (the day's realized loss and the
equity high-water mark folded into the risk context) -- and each was proved at
its own seam. This module is the outside view: the same properties, through the
shipped entry point, observed from a second process, on durable evidence alone.

WHY THIS IS ONLY NOW POSSIBLE
-----------------------------

Until PR #522 there was nothing non-trivial to survive a restart. The shipped
CLI could not reach a non-abstaining forecast in a hermetic environment, so a
restarted run resumed *nothing*: no spend, no fill, no equity, no approval.
#522's ``forecast.replay_corpus`` closed that, and :mod:`tests.hermetic_demo`
now names that composition once. Every run below is the real
``python -m windbreak run`` over committed fixtures and a configuration that
moves no threshold -- asserted, not promised, by
:func:`test_the_unattended_configuration_moves_no_threshold`.

ACCELERATED, NOT FAKED
----------------------

Time is compressed through two flags the product ships and an operator uses:
``--max-beats`` bounds a phase and ``--heartbeat-interval`` sets its cadence.
Nothing here injects a clock. ``windbreak run`` has no clock injection at all --
``_build_paper_on_beat`` never passes ``build_paper_deps``'s ``clock=`` -- and a
test that only passed under a clock nothing ships with would prove nothing about
the daemon an operator starts.

DETERMINISM
-----------

There is no unconditional sleep. Every wait is
:func:`~tests.e2e.harness.wait_until` over a real, durable condition -- a row on
the ledger, a process having exited -- with a stated ceiling, and every
assertion is on an outcome rather than on elapsed time. Two hazards get explicit
handling:

* **UTC midnight.** The research ceiling buckets per UTC *day*
  (``ResearchSpendRecorded.utc_day``), so a phase-one/phase-two pair straddling
  midnight would bucket into two days and the combined ceiling would not bind.
  :func:`_require_utc_day_headroom` refuses to start such a run, waiting the
  boundary out instead. This is a hazard about the **UTC** day and not about the
  local zone: nothing on this path reads local time, so ``tests/conftest.py``'s
  ``local_timezone_utc_minus_5`` fixture is not applicable and is deliberately
  unused.
* **A hard kill mid-write.** Phase one is ended with ``SIGKILL`` to its whole
  process group, not a graceful shutdown, so the restart is a crash recovery --
  which is the restart ``restart: on-failure`` and ``Restart=on-failure``
  actually perform. The chain is verified afterwards, which is what makes that
  safe to assert rather than hope for.

RUNTIME GATING
--------------

There is none, deliberately: nothing here needs a docker daemon or systemd, so
this module probes no runtime and therefore never reaches
:func:`~tests.e2e.harness.require_runtime`.
``tests/e2e/test_tier_selection_contract.py`` enforces the converse -- a module
that *does* probe a runtime must route the answer through that gate -- so the
absence here is checked rather than assumed.

WHAT RUNNING THIS FOUND
-----------------------

One product observation, filed as issue **#526** and since **repaired** -- this
paragraph and
:func:`test_a_kill_survives_a_restart_that_lost_its_state_directory` moved
together with it, as the paragraph they replace promised they would.

What running this found: a re-arm lands the kernel in ``PAUSED``, and no
shipped command returns it to ``PAPER`` -- only a process restart does. That
half is unchanged and is now *documented* rather than merely true: it is a
deliberate human checkpoint, the mode machine offers no ``PAUSED`` -> trading
transition at all (``_ALLOWED_TRANSITIONS[Mode.PAUSED] == {HALT, KILLED}``),
and ``docs/RUNBOOK.md`` says so at the ``kill``/``rearm`` bullet, in the
kill-switch drill, and in a machine-derived table
``tests/docs/test_mode_recovery_claims.py`` replays against the state machine
itself.

What was a defect, and is fixed: that loop also **forecast and spent research
money** on every one of those paused beats, then vetoed every intent it had
paid for on ``mode PAUSED may not trade`` -- against a per-UTC-day ceiling that
is durable since #442/#483, so the waste outlived the process. The tick's walk
gate now asks ``Mode.may_trade``, the same predicate the kernel's own approval
check vetoes on, so a beat that may not trade buys nothing. The test below
asserts the exact micros on both sides of the re-arm.

ONE PREMISE OF #473 HAS SHIFTED
-------------------------------

The issue's item 1 asks that a restarted run's **replay cursor advance**. It
cannot, and that is a decision rather than an omission: ``run_single_tick`` does
not step the cursor at all (``windbreak/scheduler/loop.py:4472``, issue #387,
SPEC S7.5.1), so there is no cursor state for a restart to carry, and
``tests/integration/test_paper_replay_cursor.py`` fails if that answer ever
changes. Nothing is asserted about a cursor here, and nothing was written to
make one appear.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from tests.e2e.harness import (
    ledger_payloads,
    read_ledger_records,
    run_windbreak,
    verify_ledger_chain,
    wait_until,
)
from tests.hermetic_demo import (
    DEMO_BOOKS,
    DEMO_CONFIG,
    demo_run_args,
    place_track_records,
    write_run_config,
)
from windbreak.config.loader import load_config
from windbreak.config.schema import WindbreakConfig
from windbreak.forecast.budget import FULL_PIPELINE_RESEARCH_COST_MICROS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.e2e.harness import ProcessLauncher, RunRoot, SpawnedProcess

pytestmark = pytest.mark.e2e

#: Beat budget for a phase ended by a signal rather than by exhausting it.
#: Bounded anyway, so a child leaked past teardown cannot outlive the suite.
UNBOUNDED_PHASE_BEATS = "100000"

#: Seconds between beats while a phase must stay running long enough for the
#: test to intervene. Fast enough that no wait below is slow, slow enough that
#: the tier is not a spin loop.
SLOW_INTERVAL = "0.2"

#: Seconds between beats for a phase that only has to run its budget out. Zero
#: is a value ``--heartbeat-interval`` accepts (``_non_negative_float``), not a
#: bypass of one.
NO_INTERVAL = "0"

ONE_BEAT = "1"
TWO_BEATS = "2"
FOUR_BEATS = "4"

#: Deadlines. Every wait in this module is bounded by one of these.
LEDGER_TIMEOUT_SECONDS = 60.0
EXIT_TIMEOUT_SECONDS = 60.0

#: The per-UTC-day research ceiling the spend tests run under: exactly three
#: full-pipeline forecasts, derived from the production charge rather than
#: transcribed, so a cheapened charge moves the ceiling with it instead of
#: silently buying a fourth forecast.
FORECASTS_PER_CAPPED_DAY = 3
CAPPED_DAY_MICROS = FULL_PIPELINE_RESEARCH_COST_MICROS * FORECASTS_PER_CAPPED_DAY

#: Beats each phase of the capped run gets. One before the restart, four after:
#: two of those four can still be afforded and two cannot, so the ceiling is
#: observed *binding* rather than merely not exceeded.
CAPPED_PHASE_ONE_FORECASTS = 1
CAPPED_PHASE_TWO_HALTS = 2

#: A per-UTC-day ceiling no bounded run in this module can reach, used by the
#: tests that assert nothing about spend. Without it they would be coupled to a
#: ceiling they do not test: a phase left running one second longer than
#: expected buys another forecast, and at the shipped ``per_day_micros`` a
#: handful of extra beats exhausts the day and the *next* phase halts on the
#: budget instead of doing the thing under test. Raising it here costs nothing,
#: because :func:`test_the_daily_research_ceiling_binds_across_a_restart` raises
#: nothing and is where the ceiling itself is proved to bind.
AMPLE_DAY_MICROS = FULL_PIPELINE_RESEARCH_COST_MICROS * 1000

#: Seconds of UTC day that must remain before a per-UTC-day run may start.
#: Comfortably longer than this whole module; see the module docstring.
UTC_DAY_HEADROOM_SECONDS = 120.0

#: The environment variable the configured ``webhook`` alert sink resolves its
#: destination from, and the loopback destination it resolves to. The host is
#: declared to ``alerts.allowed_hosts`` separately, which is what lets the
#: egress screen veto at all (issue #274). The scheme is ``http``, which the
#: shipped transport refuses *before dialling anything*, so this tier composes a
#: real, screened, configured sink and still opens no socket -- a sink that
#: could reach a destination would need either a network or a TLS certificate
#: this repository has no dependency to mint, and neither belongs in a required
#: check. What #444 broke was the wiring, and the wiring is what is asserted.
ALERT_URL_ENV_VAR = "WINDBREAK_E2E_ALERT_WEBHOOK_URL"
ALERT_HOST = "127.0.0.1"
ALERT_URL = f"http://{ALERT_HOST}/windbreak-alert"

#: The delivery evidence a configured sink leaves on the chain. The first entry
#: is the whole point: ``fallback: false`` means the escalation was offered to a
#: destination the operator declared. Issue #444's defect produced the second
#: entry **alone** -- the dispatcher reporting that nothing was configured.
CONFIGURED_SINK_DELIVERIES = [
    {"fallback": False, "outcome": "errored", "sink": "webhook"},
    {"fallback": True, "outcome": "delivered", "sink": "log-only"},
]

#: The severity :data:`windbreak.alerts.registry.AlertType.HALT_KILL` carries.
HALT_KILL_SEVERITY = "critical"

#: The two SPEC S10.3 checks that fail closed on the first tick of a *fresh*
#: ledger, because neither the day's opening equity nor the equity high-water
#: mark exists until that tick's own ``EquitySampled`` row does. A restarted run
#: reads both back off the ledger (#513, #514), so its first tick does not carry
#: these -- which is what makes them the discriminator this module uses.
FRESH_LEDGER_VETO_REASONS = [
    "daily loss limit reached",
    "trailing drawdown limit reached",
]

#: How many of the seven beats
#: :func:`test_a_kill_survives_a_restart_that_lost_its_state_directory` runs are
#: in a mode that may trade, and therefore how many forecasts that whole run may
#: buy: two before the kill and two after the restart. The three in between --
#: two ``KILLED`` and one ``PAUSED`` -- buy nothing since issue #526. The test
#: derives the same figure from its own ledger and compares the two, so this
#: constant cannot drift away from the sequence it describes.
PAPER_BEATS_IN_THIS_RUN = 4

#: The confirmation phrase template ``KillSwitch.expected_rearm_phrase`` builds.
#: The sequence it embeds is read off the ledger's own ``KillEngaged`` row, so a
#: test cannot re-arm a kill it did not observe.
REARM_PHRASE_TEMPLATE = "RE-ARM KILL {sequence}: I ACCEPT FULL RESPONSIBILITY"

MODE_HEARTBEAT_EVENT = "ModeHeartbeat"
EQUITY_SAMPLED_EVENT = "EquitySampled"
INTENT_APPROVED_EVENT = "IntentApproved"
INTENT_VETOED_EVENT = "IntentVetoed"
RESEARCH_SPEND_EVENT = "ResearchSpendRecorded"
RESEARCH_HALTED_EVENT = "ResearchBudgetHalted"
ALERT_EMITTED_EVENT = "AlertEmitted"
KILL_ENGAGED_EVENT = "KillEngaged"
KILL_REARMED_EVENT = "KillReArmed"

#: Mode tokens the loop stamps on its ``ModeHeartbeat`` rows.
MODE_PAPER = "PAPER"
MODE_KILLED = "KILLED"
MODE_PAUSED = "PAUSED"
MODE_TICK_FAILED = "TICK_FAILED"

#: The ``--process`` token every run below uses, and therefore the component
#: stamped on the rows :class:`windbreak.main.BeatSupervisor` writes.
PIPELINE_COMPONENT = "pipeline"


def _opening_equity_micros() -> int:
    """Return the committed fixture's opening account equity, in micros.

    Read out of the fixture rather than transcribed, so the high-water-mark
    assertion below is an identity against its source.

    Returns:
        The demonstration account's opening total, in micros.
    """
    balances = json.loads((DEMO_BOOKS / "balances.json").read_text(encoding="utf-8"))
    return int(balances["total"])


def _seconds_left_in_the_utc_day() -> float:
    """Return how much of the current UTC day remains.

    Returns:
        Seconds until the next UTC midnight.
    """
    now = datetime.now(UTC)
    midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return (midnight - now).total_seconds()


def _require_utc_day_headroom() -> None:
    """Refuse to start a per-UTC-day run that would straddle midnight.

    Starting anyway would give a test that is correct for all but a two-minute
    window a day and then reds somebody's unrelated pull request -- on a
    required check, a defect this module would be introducing. The wait is
    bounded and on a real condition (the date changing), never a fixed sleep,
    and it is reached only inside that window.
    """
    if _seconds_left_in_the_utc_day() >= UTC_DAY_HEADROOM_SECONDS:
        return
    today = datetime.now(UTC).date()
    wait_until(
        lambda: datetime.now(UTC).date() != today,
        timeout=UTC_DAY_HEADROOM_SECONDS + 10.0,
        description="the UTC day to roll over before a per-UTC-day run starts",
        interval=0.5,
    )


def _run_environment() -> dict[str, str]:
    """Return the child environment carrying the alert sink's destination.

    The destination reaches the child only through the environment, which is the
    one channel the product accepts it on: a URL in configuration would be
    hash-chained onto the ledger and could never be redacted (issue #274).

    Returns:
        This process's environment plus the alert destination variable.
    """
    return dict(os.environ, **{ALERT_URL_ENV_VAR: ALERT_URL})


def _prepared_run(run_root: RunRoot, *, alerts: bool = True) -> Path:
    """Lay out one run root's configuration, state and report directories.

    Args:
        run_root: The run root to prepare.
        alerts: Whether to declare a configured alert sink.

    Returns:
        The written configuration file.
    """
    place_track_records(run_root.report_dir)
    return write_run_config(
        run_root.root / "windbreak.yaml",
        state_dir=run_root.state_dir,
        alert_url_env=ALERT_URL_ENV_VAR if alerts else None,
        allowed_hosts=(ALERT_HOST,) if alerts else (),
    )


def _phase(
    run_root: RunRoot,
    config: Path,
    *,
    beats: str,
    interval: str = NO_INTERVAL,
    ledger_path: Path | None = None,
    per_day_micros: str | None = None,
) -> None:
    """Run one bounded phase of the loop to completion and require a clean exit.

    Args:
        run_root: The run root supplying the report directory and ledger.
        config: The configuration file the phase runs under.
        beats: The ``--max-beats`` budget.
        interval: The ``--heartbeat-interval`` cadence.
        ledger_path: The ledger to append to; defaults to the run root's.
        per_day_micros: An explicit ``--research-per-day-micros`` ceiling.

    Raises:
        AssertionError: If the phase exits non-zero.
    """
    completed = run_windbreak(
        *demo_run_args(
            config=config,
            ledger_path=run_root.ledger_path if ledger_path is None else ledger_path,
            report_dir=run_root.report_dir,
            max_beats=beats,
            heartbeat_interval=interval,
            research_per_day_micros=per_day_micros,
        ),
        env=_run_environment(),
        timeout=EXIT_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, (
        f"phase exited {completed.returncode}:\n{completed.stderr}"
    )


def _spawn_phase(
    launcher: ProcessLauncher,
    run_root: RunRoot,
    config: Path,
    *,
    name: str,
    report_dir: Path | None = None,
    per_day_micros: str | None = None,
) -> SpawnedProcess:
    """Start a phase as a tracked background child at the slow cadence.

    Args:
        launcher: The launcher that guarantees the child is reaped.
        run_root: The run root supplying the ledger and report directory.
        config: The configuration file the phase runs under.
        name: Short label used for the child's log filenames.
        report_dir: The ``--report-dir``; defaults to the run root's.
        per_day_micros: An explicit ``--research-per-day-micros`` ceiling.

    Returns:
        The started process.
    """
    return launcher.spawn(
        *demo_run_args(
            config=config,
            ledger_path=run_root.ledger_path,
            report_dir=run_root.report_dir if report_dir is None else report_dir,
            max_beats=UNBOUNDED_PHASE_BEATS,
            heartbeat_interval=SLOW_INTERVAL,
            research_per_day_micros=per_day_micros,
        ),
        name=name,
        env=_run_environment(),
    )


def _verb(*args: str, stdin: str | None = None) -> None:
    """Run one windbreak operator subcommand and require a clean exit.

    Args:
        *args: The subcommand and its arguments.
        stdin: Text fed to the child's stdin, or ``None`` to send EOF at once.

    Raises:
        AssertionError: If the subcommand exits non-zero.
    """
    completed = run_windbreak(*args, input_text=stdin, timeout=EXIT_TIMEOUT_SECONDS)
    assert completed.returncode == 0, (
        f"`windbreak {args[0]}` exited {completed.returncode}:\n{completed.stderr}"
    )


def _modes(ledger_path: Path) -> list[tuple[int, str]]:
    """Return every ledgered beat mode, in chain order.

    Args:
        ledger_path: The ledger to read.

    Returns:
        One ``(beat, mode)`` pair per ``ModeHeartbeat`` row.
    """
    return [
        (int(data["beat"]), str(data["mode"]))
        for data in ledger_payloads(ledger_path, MODE_HEARTBEAT_EVENT)
    ]


def _equity_samples(ledger_path: Path) -> list[int]:
    """Return every ledgered equity sample, in chain order.

    Args:
        ledger_path: The ledger to read.

    Returns:
        Each ``EquitySampled`` row's ``equity_micros``.
    """
    return [
        int(data["equity_micros"])
        for data in ledger_payloads(ledger_path, EQUITY_SAMPLED_EVENT)
    ]


def _veto_reasons(ledger_path: Path) -> list[list[str]]:
    """Return every ledgered veto's reason list, in chain order.

    Args:
        ledger_path: The ledger to read.

    Returns:
        One reason list per ``IntentVetoed`` row.
    """
    vetoes: list[list[str]] = []
    for data in ledger_payloads(ledger_path, INTENT_VETOED_EVENT):
        reasons = data["reasons"]
        assert isinstance(reasons, list)
        vetoes.append([str(reason) for reason in reasons])
    return vetoes


def _chain_hashes(ledger_path: Path) -> list[str]:
    """Return every row's chain hash, in chain order.

    Args:
        ledger_path: The ledger to read.

    Returns:
        Each record's ``event_hash``.
    """
    return [record.event_hash for record in read_ledger_records(ledger_path)]


def _await_condition(
    predicate: Callable[[], bool], what: str, *, spawned: SpawnedProcess
) -> None:
    """Block until a condition holds, failing at once if the writer has died.

    The ``spawned`` check is what separates "not yet" from "never": without it a
    child that exited would burn the whole deadline and report only a timeout,
    which is exactly the diagnosis a supervised-beat regression needs *not* to
    produce. A dead child fails immediately and carries its own log.

    Args:
        predicate: The condition, re-evaluated until true.
        what: What is being waited for, quoted in the failure message.
        spawned: The child expected to satisfy the condition.
    """

    def _still_possible() -> bool:
        """Report the condition, refusing to keep waiting on a dead writer.

        Returns:
            ``True`` once the condition holds.

        Raises:
            AssertionError: If the child exited before it held.
        """
        if predicate():
            return True
        if not spawned.is_running():
            message = (
                f"the {spawned.name} process exited before {what}. It was "
                "supposed to survive:\n"
                f"{spawned.stdout_text()}{spawned.stderr_text()}"
            )
            raise AssertionError(message)
        return False

    wait_until(_still_possible, timeout=LEDGER_TIMEOUT_SECONDS, description=what)


def _await_rows(
    ledger_path: Path,
    event: str,
    count: int,
    what: str,
    *,
    spawned: SpawnedProcess,
) -> None:
    """Block until a ledger another process is writing holds ``count`` rows.

    Args:
        ledger_path: The ledger to poll.
        event: The event type to count.
        count: How many rows must be present.
        what: What is being waited for, quoted in the failure message.
        spawned: The child writing the ledger.
    """
    _await_condition(
        lambda: len(ledger_payloads(ledger_path, event)) >= count,
        what,
        spawned=spawned,
    )


def _kill_group(spawned: SpawnedProcess) -> None:
    """SIGKILL a child's whole process group and wait for it to be reaped.

    ``SIGKILL`` rather than ``SIGTERM``: a graceful shutdown would let the loop
    finish its beat and close its ledger, and a restart after a *clean* stop is
    not the restart a supervisor performs.

    Args:
        spawned: The child to kill.
    """
    os.killpg(os.getpgid(spawned.pid), signal.SIGKILL)
    wait_until(
        lambda: not spawned.is_running(),
        timeout=EXIT_TIMEOUT_SECONDS,
        description=f"the SIGKILLed {spawned.name} process to be reaped",
    )


def _differing_leaves(left: object, right: object, prefix: str) -> list[str]:
    """Return the dotted paths of every leaf where two configurations differ.

    Walks the dataclass rather than comparing a hand-written list of fields, so
    a leaf added later is covered without anyone remembering to add it -- the
    difference between measuring "nothing else moved" and asserting it.

    Args:
        left: The left value.
        right: The right value.
        prefix: The dotted path reached so far.

    Returns:
        Every differing leaf's dotted path, deepest name last.
    """
    if dataclasses.is_dataclass(left) and dataclasses.is_dataclass(right):
        differing: list[str] = []
        for field in dataclasses.fields(left):
            differing += _differing_leaves(
                getattr(left, field.name),
                getattr(right, field.name),
                f"{prefix}.{field.name}" if prefix else field.name,
            )
        return differing
    return [] if left == right else [prefix]


def test_the_unattended_configuration_moves_no_threshold(tmp_path: Path) -> None:
    """The configuration these runs use differs from the defaults in six leaves.

    The evidence every test below produces is evidence about the *shipped*
    thresholds only if the configuration relaxes none of them. The differing set
    is therefore **derived**, by walking every dataclass field of
    :class:`~windbreak.config.schema.WindbreakConfig` exactly as
    ``tests/integration/test_shipped_cli_hermetic_forecast.py`` derives it for
    the committed file -- so a lowered floor, a widened window or a cheapened
    charge appears here as a new entry rather than passing unnoticed next door.

    Six leaves differ: the three the committed demonstration already carries,
    and the three this tier adds -- its own state directory and the two
    independent halves of a declared alert sink. None is a threshold, and the
    per-UTC-day research ceiling in particular is asserted **equal** to the
    shipped default, so the spend tests below cap it from the command line
    rather than from a quietly edited config.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    generated = write_run_config(
        tmp_path / "windbreak.yaml",
        state_dir=tmp_path / "state",
        alert_url_env=ALERT_URL_ENV_VAR,
        allowed_hosts=(ALERT_HOST,),
    )

    configured = load_config(generated)

    assert _differing_leaves(configured, WindbreakConfig(), "") == [
        "correlation.tags",
        "forecast.replay_corpus.mode",
        "forecast.replay_corpus.corpus_dir",
        "ops.state_dir",
        "alerts.sinks",
        "alerts.allowed_hosts",
    ]
    committed = load_config(DEMO_CONFIG)
    assert configured.correlation.tags == committed.correlation.tags
    assert configured.forecast.replay_corpus.mode == (
        committed.forecast.replay_corpus.mode
    )
    assert configured.alerts.allowed_hosts == (ALERT_HOST,)
    assert configured.forecast.budget.per_day_micros == (
        WindbreakConfig().forecast.budget.per_day_micros
    )


def test_a_restart_on_the_same_ledger_resumes_instead_of_starting_over(
    launcher: ProcessLauncher, run_root: RunRoot, tmp_path: Path
) -> None:
    """A hard-killed run restarts onto its own ledger and picks up where it was.

    Issue #473's items 1 and 5, and the outside view of #513/#514. Phase one is
    ended with ``SIGKILL`` to its whole process group -- a crash, not a
    shutdown -- and phase two is a *new process* on the same ledger path.

    **What proves resumption is a decision, not a row count.** On the first tick
    of a fresh ledger the day's opening equity and the equity high-water mark do
    not exist yet, so ``daily_loss_limit`` and ``trailing_drawdown`` both fail
    closed and that tick's one intent is vetoed for exactly those two reasons.
    Both terms are folded back out of the ledger's own ``EquitySampled`` rows, so
    a run restarting onto a populated ledger has them on its *first* tick and
    approves. The control run is what makes that comparison honest: the same
    command line, the same committed fixtures, one beat, a ledger of its own --
    and it vetoes. Without it, "the restart approved" would be
    indistinguishable from "any first beat approves".

    Those two veto reasons are exactly the checks #513 and #514 repaired --
    ``daily_loss_limit`` reads the realized-loss baseline, ``trailing_drawdown``
    reads the equity high-water mark -- so the comparison is a statement about
    both, made through the kernel's own decision rather than through a number
    this test recomputed.

    **What is deliberately not asserted, and why.** The equity *magnitude* is
    not durable and must not be treated as though it were: ``PaperExchange``
    reloads the committed ``balances.json`` at composition, so a restarted
    process opens on the fixture's balance again. Only the ledger-derived risk
    *terms* cross the boundary. An earlier draft compared the peak sample
    against phase two's samples and passed for several runs on nothing but a
    fee-sized coincidence -- the trap this suite has hit repeatedly -- so the
    claim is made where it is structural and dropped where it was not.

    The chain is asserted to have been **appended to and not rewritten**: every
    hash phase one wrote is still present, in order, and there are more. And the
    anchor taken over phase one's head -- immediately after the kill, before the
    restart appends anything -- still verifies against the live chain once phase
    two has, which is #473's item 5.

    Every run here is given :data:`AMPLE_DAY_MICROS`, because this test asserts
    nothing about spend and must not fail when a phase happens to run a beat
    longer than expected and exhausts a ceiling that is proved elsewhere.

    Args:
        launcher: The launcher that reaps every child.
        run_root: This test's isolated run root.
        tmp_path: pytest's per-test temporary directory.
    """
    config = _prepared_run(run_root)
    ample = str(AMPLE_DAY_MICROS)
    phase_one = _spawn_phase(
        launcher, run_root, config, name="phase-one", per_day_micros=ample
    )
    _await_rows(
        run_root.ledger_path,
        INTENT_APPROVED_EVENT,
        1,
        "phase one to approve an intent",
        spawned=phase_one,
    )
    _kill_group(phase_one)
    _verb(
        "anchor",
        "--ledger-path",
        str(run_root.ledger_path),
        "--anchor-path",
        str(run_root.anchor_path),
    )

    verify_ledger_chain(run_root.ledger_path)
    phase_one_hashes = _chain_hashes(run_root.ledger_path)
    phase_one_samples = _equity_samples(run_root.ledger_path)
    phase_one_vetoes = _veto_reasons(run_root.ledger_path)
    phase_one_approvals = len(
        ledger_payloads(run_root.ledger_path, INTENT_APPROVED_EVENT)
    )
    _phase(run_root, config, beats=ONE_BEAT, per_day_micros=ample)
    control_ledger = tmp_path / "control-ledger.db"
    _phase(
        run_root,
        config,
        beats=ONE_BEAT,
        ledger_path=control_ledger,
        per_day_micros=ample,
    )

    verify_ledger_chain(run_root.ledger_path)
    assert run_root.anchor_path.read_text(encoding="utf-8").splitlines() != []
    resumed_hashes = _chain_hashes(run_root.ledger_path)
    assert resumed_hashes[: len(phase_one_hashes)] == phase_one_hashes
    assert len(resumed_hashes) > len(phase_one_hashes)
    assert phase_one_vetoes == [FRESH_LEDGER_VETO_REASONS]
    approvals = ledger_payloads(run_root.ledger_path, INTENT_APPROVED_EVENT)
    assert len(approvals) == phase_one_approvals + 1
    assert approvals[-1]["reasons"] == []
    assert _veto_reasons(run_root.ledger_path) == phase_one_vetoes
    assert _modes(run_root.ledger_path)[-1] == (1, MODE_PAPER)
    assert ledger_payloads(control_ledger, INTENT_APPROVED_EVENT) == []
    assert _veto_reasons(control_ledger) == [FRESH_LEDGER_VETO_REASONS]
    assert _equity_samples(control_ledger) == [_opening_equity_micros()]
    assert _equity_samples(run_root.ledger_path)[: len(phase_one_samples)] == (
        phase_one_samples
    )
    _verb(
        "verify",
        "--ledger-path",
        str(run_root.ledger_path),
        "--anchor-path",
        str(run_root.anchor_path),
    )


def test_the_daily_research_ceiling_binds_across_a_restart(run_root: RunRoot) -> None:
    """The day's research spend is the *day's*, not the process's (#442, #483).

    Issue #473's item 2, in exact micros. The ceiling is set from the command
    line (``--research-per-day-micros``) to exactly three full-pipeline
    forecasts -- derived from
    :data:`~windbreak.forecast.budget.FULL_PIPELINE_RESEARCH_COST_MICROS`, never
    transcribed -- and the run is split so that no phase can satisfy the
    assertion on its own:

    * phase one buys **one** forecast and stops;
    * phase two, a new process on the same ledger, gets **four** beats: it can
      afford two more and is refused the last two.

    The three numbers are therefore all different -- one charge before the
    restart, two after, three in the day -- so an equality here cannot be a
    fixture coincidence. If the counter were still per-process, as it was before
    #442, phase two would open on zero, buy three, and the day would end on four
    charges and twelve million micros with the last two beats never halted. Both
    halves of that go red.

    The halts are asserted as whole payloads rather than as a count, because the
    interesting claim is *which* ceiling refused and what it thought had been
    spent: ``per_day``, at the ledgered total, against the ledgered ceiling.

    Args:
        run_root: This test's isolated run root.
    """
    _require_utc_day_headroom()
    config = _prepared_run(run_root, alerts=False)
    cap = str(CAPPED_DAY_MICROS)

    _phase(run_root, config, beats=ONE_BEAT, per_day_micros=cap)
    after_phase_one = ledger_payloads(run_root.ledger_path, RESEARCH_SPEND_EVENT)
    _phase(run_root, config, beats=FOUR_BEATS, per_day_micros=cap)

    verify_ledger_chain(run_root.ledger_path)
    spends = ledger_payloads(run_root.ledger_path, RESEARCH_SPEND_EVENT)
    assert len(after_phase_one) == CAPPED_PHASE_ONE_FORECASTS
    assert len(spends) == FORECASTS_PER_CAPPED_DAY
    assert [int(spend["cost_micros"]) for spend in spends] == (
        [FULL_PIPELINE_RESEARCH_COST_MICROS] * FORECASTS_PER_CAPPED_DAY
    )
    days = {str(spend["utc_day"]) for spend in spends}
    assert len(days) == 1, (
        f"the run straddled UTC midnight and bucketed into {sorted(days)}, so "
        "no combined per-day ceiling was ever under test"
    )
    assert sum(int(spend["cost_micros"]) for spend in spends) == CAPPED_DAY_MICROS
    halts = ledger_payloads(run_root.ledger_path, RESEARCH_HALTED_EVENT)
    assert (
        halts
        == [
            {
                "budget_micros": CAPPED_DAY_MICROS,
                "halt_kind": "per_day",
                "market_ticker": "",
                "spent_micros": CAPPED_DAY_MICROS,
                "utc_day": next(iter(days)),
            }
        ]
        * CAPPED_PHASE_TWO_HALTS
    )


def test_an_induced_report_volume_fault_is_survived_and_stays_loud(
    launcher: ProcessLauncher, run_root: RunRoot, tmp_path: Path
) -> None:
    """A tick that raises is escalated three ways and the daemon keeps beating.

    Issue #473's item 3, and the outside view of PR #460. The fault is induced
    from the filesystem, not from a patched object: the run's ``--report-dir``
    is replaced by a regular *file* mid-run, which is what a bind mount that
    came back wrong looks like. ``run_single_tick`` writes this ISO week's
    report on every tick, and ``write_weekly_stub``'s ``mkdir(exist_ok=True)``
    over a non-directory raises ``FileExistsError``. That is a genuine tick
    exception -- the class ``#443`` says must never be fatal -- reached without
    stubbing anything the product owns.

    A swallowed fault would be worse than a crash, so all three escalations are
    asserted, and none of them by scraping a log:

    * the beat's own ``ModeHeartbeat`` row now says ``TICK_FAILED``, appended
      **after** the healthy ``PAPER`` row the tick had already stamped, which is
      exactly the ordering issue #447 exists for;
    * an ``AlertEmitted`` row carries the escalation, with the full operator
      message asserted for equality -- and *derived*, by making the same call
      the child made on the same path, so it cannot drift with a platform's
      ``strerror`` wording;
    * that row's delivery evidence names a **configured** sink,
      ``fallback: false``. Issue #444's defect produced the log-only fallback
      entry alone.

    A wedged process is alive and useless, so liveness is asserted as *work*:
    the repair is applied and a strictly later beat is required to come back
    clean, on the same ledger, whose chain still verifies.

    Args:
        launcher: The launcher that reaps every child.
        run_root: This test's isolated run root.
        tmp_path: pytest's per-test temporary directory.
    """
    config = _prepared_run(run_root)
    report_dir = tmp_path / "reports"
    place_track_records(report_dir)
    running = _spawn_phase(
        launcher, run_root, config, name="fault", report_dir=report_dir
    )
    _await_rows(
        run_root.ledger_path,
        MODE_HEARTBEAT_EVENT,
        1,
        "the first clean beat",
        spawned=running,
    )
    clean_beats = _modes(run_root.ledger_path)

    shutil.rmtree(report_dir)
    report_dir.write_text("", encoding="utf-8")
    _await_condition(
        lambda: any(
            mode == MODE_TICK_FAILED for _, mode in _modes(run_root.ledger_path)
        ),
        "the loop to ledger a supervised beat failure",
        spawned=running,
    )
    alive_during_the_fault = running.is_running()
    raised = _induced_failure_text(report_dir)

    report_dir.unlink()
    place_track_records(report_dir)
    failed_beat = max(
        beat for beat, mode in _modes(run_root.ledger_path) if mode == MODE_TICK_FAILED
    )
    _await_condition(
        lambda: any(
            mode == MODE_PAPER and beat > failed_beat
            for beat, mode in _modes(run_root.ledger_path)
        ),
        "a clean beat after the report volume was repaired",
        spawned=running,
    )

    assert clean_beats != []
    assert {mode for _, mode in clean_beats} == {MODE_PAPER}
    assert alive_during_the_fault
    assert running.is_running()
    modes = _modes(run_root.ledger_path)
    assert modes.index((failed_beat, MODE_PAPER)) < modes.index(
        (failed_beat, MODE_TICK_FAILED)
    )
    alerts = ledger_payloads(
        run_root.ledger_path, ALERT_EMITTED_EVENT, component=PIPELINE_COMPONENT
    )
    assert alerts == [
        {
            "deliveries": CONFIGURED_SINK_DELIVERIES,
            "delivery_reported": True,
            "message": f"beat seq={failed_beat} failed: {raised}",
            "severity": HALT_KILL_SEVERITY,
        }
    ]
    verify_ledger_chain(run_root.ledger_path)


def _induced_failure_text(report_dir: Path) -> str:
    """Return the ``Type: message`` text the induced report-volume fault raises.

    Derived rather than transcribed: this makes the *same* call the tick made,
    on the *same* path, and renders it the way
    :meth:`windbreak.main.BeatSupervisor.observe` renders a raising beat. A
    platform whose ``strerror`` wording differs therefore moves the expectation
    with the behaviour instead of reddening a required check.

    Must be called while the fault is still induced, which is asserted rather
    than assumed: a call that *succeeded* would mean the fault was never in
    place and every assertion resting on it proves nothing.

    Args:
        report_dir: The report directory, currently a regular file.

    Returns:
        The exception's type name and message, as the supervisor renders them.

    Raises:
        AssertionError: If the induced fault is not actually in place.
    """
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:
        return f"{type(exc).__name__}: {exc}"
    message = (
        f"{report_dir} is a usable directory, so the fault this test induces "
        "was never in place and the assertions resting on it prove nothing"
    )
    raise AssertionError(message)


def test_a_kill_survives_a_restart_that_lost_its_state_directory(
    run_root: RunRoot,
) -> None:
    """The kill latch is carried by the ledger, not only by the ``KILL`` file.

    Issue #473's item 4, taken one step past what it asks. ``windbreak kill``
    drops a ``KILL`` file, so a restart that still has its state directory is
    kept dead by the file alone and proves nothing about durability. Here the
    state directory is **deleted** before the restart -- the shape of a
    container whose ledger volume is named and whose state directory is not --
    so the only thing that can keep the kernel dead is
    ``KillSwitch.from_events``' replay of the chain.

    That is asserted two ways, because "still killed" alone would also be true
    if the restart had simply killed itself again: the mode is ``KILLED``, and
    there is still exactly **one** ``KillEngaged`` row on the ledger.

    The sequence then runs to the end of #455's DoD item 7. Re-arming in place
    exits ``KILLED`` into ``PAUSED`` -- never back into ``PAPER`` -- so the very
    next beat still refuses to trade, and it takes a further restart before
    approvals resume. That half is deliberate and is pinned here.

    What changed with issue #526 is what that paused beat *costs*. It used to
    forecast, charge ``FULL_PIPELINE_RESEARCH_COST_MICROS`` against the durable
    per-UTC-day ceiling, and only then veto the intent it had paid for on
    ``mode PAUSED may not trade``. So the assertion that used to read "the
    paused beat leaves a veto row" now reads "the paused beat leaves **no**
    veto row and **no** spend row", and the spend is asserted as exact micros
    across the whole run: one charge per ``PAPER`` beat, none for any other
    mode. Those two figures -- four charges here, five before the fix -- are
    the whole difference, which is why they are counted rather than bounded.

    Both directions are load-bearing. A paused beat that bought nothing would
    also be produced by a loop that had stopped beating altogether, so the mode
    heartbeats, the surviving ``KillReArmed`` row, and the two approvals the
    restarted run goes on to make are all still asserted here.

    Args:
        run_root: This test's isolated run root.
    """
    config = _prepared_run(run_root, alerts=False)

    _phase(run_root, config, beats=TWO_BEATS)
    _verb("kill", "--state-dir", str(run_root.state_dir))
    _phase(run_root, config, beats=ONE_BEAT)
    killed_modes = _modes(run_root.ledger_path)
    shutil.rmtree(run_root.state_dir)
    _phase(run_root, config, beats=ONE_BEAT)
    survived_modes = _modes(run_root.ledger_path)
    engagements = ledger_payloads(run_root.ledger_path, KILL_ENGAGED_EVENT)
    _verb(
        "rearm",
        "--state-dir",
        str(run_root.state_dir),
        stdin=REARM_PHRASE_TEMPLATE.format(
            sequence=int(engagements[-1]["kill_sequence"])
        )
        + "\n",
    )
    _phase(run_root, config, beats=ONE_BEAT)
    rearmed_modes = _modes(run_root.ledger_path)
    approvals_before_the_restart = len(
        ledger_payloads(run_root.ledger_path, INTENT_APPROVED_EVENT)
    )
    _phase(run_root, config, beats=TWO_BEATS)

    assert killed_modes == [(1, MODE_PAPER), (2, MODE_PAPER), (1, MODE_KILLED)]
    assert survived_modes[3:] == [(1, MODE_KILLED)]
    assert len(engagements) == 1
    assert engagements[0]["trigger"] == "KILL_FILE"
    assert rearmed_modes[4:] == [(1, MODE_PAUSED)]
    assert [
        int(rearm["kill_sequence"])
        for rearm in ledger_payloads(run_root.ledger_path, KILL_REARMED_EVENT)
    ] == [int(engagements[0]["kill_sequence"])]
    assert _modes(run_root.ledger_path)[5:] == [(1, MODE_PAPER), (2, MODE_PAPER)]
    assert _veto_reasons(run_root.ledger_path) == [FRESH_LEDGER_VETO_REASONS]
    approvals = ledger_payloads(run_root.ledger_path, INTENT_APPROVED_EVENT)
    assert len(approvals) == approvals_before_the_restart + 2
    paper_beats = [mode for _, mode in _modes(run_root.ledger_path)].count(MODE_PAPER)
    spends = ledger_payloads(run_root.ledger_path, RESEARCH_SPEND_EVENT)
    assert paper_beats == PAPER_BEATS_IN_THIS_RUN
    assert [int(spend["cost_micros"]) for spend in spends] == (
        [FULL_PIPELINE_RESEARCH_COST_MICROS] * paper_beats
    )
    assert ledger_payloads(run_root.ledger_path, RESEARCH_HALTED_EVENT) == []
    verify_ledger_chain(run_root.ledger_path)
