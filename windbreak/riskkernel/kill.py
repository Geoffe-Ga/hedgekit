"""The Risk Kernel kill switch and its four trigger sources (SPEC S10.12, issue #35).

This module gives the Risk Kernel a single kill executor, :class:`KillSwitch`,
reachable from four independent triggers -- a CLI ``KILL`` file, an in-process
``KILL`` state-dir file, a dashboard challenge/confirm handshake, and
consecutive reconciliation-mismatch auto-detection -- plus the
typed-confirmation re-arm that is the *only* way back out of ``KILLED``.

Every trigger funnels into :meth:`KillSwitch.kill`, which composes the LOCKED
:mod:`windbreak.riskkernel.modes` primitives (it never widens the mode ladder):
it drives the machine to ``KILLED`` *first* (the fail-safe direction), then
records the closed kill-effect surface -- one :class:`KillEngaged`, one
:class:`CancelAllDirective`, and one ``ReservationReleased`` per active
reservation -- dispatches one ``HALT_KILL`` alert, and drops a ``KILL`` file.
The switch **holds positions by design**: it only cancels resting orders and
releases capital reservations, and no string it ever ledgers names a
sell/close/submit/dump action.

The trigger adapters (:class:`KillFileWatcher`,
:class:`ReconciliationMismatchMonitor`, :class:`DashboardKillStub`) are all
poll- or event-driven with no unbounded waits: the fleet drives them one beat
at a time. :class:`KillIntegration` bundles the switch and its adapters into
the single value the Risk Kernel process consumes.

**Wiring status (issue #35, composed in #144):** these adapters are now
composed into the live ``windbreak run`` process. ``windbreak/main.py``'s
``_build_risk_kernel`` builds a :class:`KillSwitch` + :class:`KillFileWatcher`
over ``ops.state_dir`` and a :class:`ReconciliationMismatchMonitor`, bundles
them into a :class:`KillIntegration`, and hands it to the
:class:`~windbreak.riskkernel.process.RiskKernel` that ``_run_riskkernel``
drives from ``windbreak run --process riskkernel`` -- the production
entrypoint per the ``systemd``/``docker-compose`` deploy configs. That
kernel's :meth:`KillFileWatcher.poll_once` runs every beat, so
``windbreak kill --state-dir DIR`` now halts a running kernel whenever
``DIR`` is that kernel's configured ``ops.state_dir``. The
:class:`ReconciliationMismatchMonitor` is composed from the configured
threshold and, since issue #236, is fed a live signal: when ``windbreak run
--process riskkernel`` is given a ``--snapshot-fixture-dir``,
``_build_risk_kernel`` wires a
:class:`~windbreak.riskkernel.verification.ReadOnlyVerifier`
over that read-only connector, so the kernel's per-beat verification cycle
grades a real reconciliation outcome and a sustained ``BREACH`` engages the
``AUTO_RECONCILIATION`` auto-kill -- no longer composed-but-dormant. The wiring
is live, but a static ``--snapshot-fixture-dir`` connector reports identical
data every cycle, so it can never drift from its own frozen startup baseline:
against a static fixture this exercises the reconciliation *plumbing*, not a
real breach kill. A drift only arises from a mutating or live connector -- the
ledger-derived expectation source seam tracked in issue #288. With no
``--snapshot-fixture-dir`` no verifier is wired and the operator ``KILL`` file
remains the live trigger. Two edges remain: ``windbreak/riskkernel/
process.py``'s ``main()`` (reachable only via ``python -m
windbreak.riskkernel``, a dev-only path) still runs without kill wiring; and
replaying an existing ``KILLED`` state from the ledger on startup is now wired
into that composition (issue #235) whenever ``windbreak run --process
riskkernel`` is given a ``--ledger-path``, with the on-disk ``KILL`` file below
kept as the belt-and-suspenders restart measure alongside it.

**Durable kill state (issue #123):** kill state need not live only in process
memory, and the ``KILL`` file need not be the sole interim restart measure. The
kill/re-arm ledger is authoritative: :func:`kill_state_in` folds a recorded
history into a :class:`ReplayedKillState`, and :meth:`KillSwitch.from_events`
(with :meth:`~windbreak.riskkernel.process.RiskKernel.from_events`) rebuilds a
switch already at its restored kill sequence and, on an unrearmed history, a
kernel already ``KILLED``. Ledger replay is thus the *primary* durable
mechanism, with the ``KILL`` file as belt-and-suspenders: ``_build_risk_kernel``
now calls ``from_events`` on startup over a supplied ledger (issue #235), so an
engaged kill survives a ``windbreak run --process riskkernel --ledger-path``
restart even after the operator's ``KILL`` file is deleted.

Everything on this path is float-free (SPEC S6.1): epoch seconds and kill
sequence numbers are ``int`` only.
"""

from __future__ import annotations

import enum
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from windbreak.alerts.registry import AlertType
from windbreak.ledger.events import CancelAllDirective, KillEngaged, KillReArmed
from windbreak.riskkernel import modes
from windbreak.riskkernel.modes import KillReArmError, Mode
from windbreak.riskkernel.verification import VerificationOutcome

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from windbreak.ledger.events import Event
    from windbreak.riskkernel.modes import ModeStateMachine
    from windbreak.riskkernel.process import KernelLedgerWriter
    from windbreak.riskkernel.reservations import ReservationLedger

#: Component label stamped on every event this module records.
_COMPONENT = "riskkernel"

#: The state-dir filenames that form the on-disk kill/re-arm protocol. The CLI
#: (``windbreak kill`` / ``windbreak rearm``) writes these same names, so they are
#: public for that one cross-module reuse rather than duplicated as literals.
KILL_FILENAME = "KILL"
REARM_FILENAME = "REARM"

#: The single cancel-all scope: resting *orders* only, never open positions
#: (position-hold invariant). Chosen to contain no forbidden action token.
_CANCEL_ALL_SCOPE = "all_open_orders"

#: The reason stamped on every kill-path reservation release. A hold-only
#: phrase with no sell/close/submit/dump substring, so the ledgered kill
#: surface never names a trading action.
_RELEASE_REASON = "kill_switch_engaged"

#: The ``HALT_KILL`` alert body. Also hold-only wording (no action token).
_HALT_KILL_MESSAGE = "kill switch engaged; trading halted, positions held"

#: Byte budget handed to :func:`secrets.token_urlsafe` for a dashboard nonce.
_CHALLENGE_TOKEN_BYTES = 32

#: The kill sequence a replay reports when the history records no kill at all --
#: also a fresh switch's pre-first-kill default, so a restored sequence of 0 is
#: indistinguishable from "never killed".
_NO_KILL_SEQUENCE = 0

#: The ``event_type`` discriminators the durable-state fold matches, kept as
#: literals so :func:`kill_state_in` recognizes both live event instances and
#: deserialized base ``Event`` records by type string alone -- mirroring
#: :meth:`FloorGovernance._absorb_replayed_event`'s string-keyed matching.
_KILL_ENGAGED_EVENT_TYPE = "KillEngaged"
_KILL_REARMED_EVENT_TYPE = "KillReArmed"


def _default_clock() -> int:
    """Return the current wall clock as whole epoch seconds.

    Casts :func:`time.time` to an ``int`` so the kill path stays off the banned
    float path (SPEC S6.1): a ``KillEngaged.epoch`` is always integral.

    Returns:
        The current time, in whole epoch seconds.
    """
    return int(time.time())


@dataclass(frozen=True, slots=True)
class ReplayedKillState:
    """The two durable kill facts recovered from a replayed ledger (issue #123).

    Attributes:
        last_kill_sequence: The ``kill_sequence`` of the most recent
            :class:`KillEngaged` in the history, or :data:`_NO_KILL_SEQUENCE`
            (0) when the history records no kill at all. It survives a matching
            re-arm, so the monotonic counter never resets across a restart.
        killed: ``True`` iff that latest kill has no matching later
            :class:`KillReArmed` -- i.e. the kernel was killed and not yet
            re-armed when the history ended.
    """

    last_kill_sequence: int
    killed: bool


def kill_state_in(events: Iterable[Event]) -> ReplayedKillState:
    """Fold a kill/re-arm history into its durable :class:`ReplayedKillState`.

    Fail-closed, because this drives a re-arm authorization control: a
    :class:`KillReArmed` clears the kill flag *only* when its ``kill_sequence``
    matches the latest kill's, so a stale or mismatched re-arm record can never
    un-kill a replay. Events are matched by ``event_type`` string alone
    (mirroring :meth:`FloorGovernance._absorb_replayed_event`), so the fold
    works on both live :class:`KillEngaged` / :class:`KillReArmed` instances and
    deserialized base :class:`~windbreak.ledger.events.Event` records.

    Args:
        events: The kill/re-arm event history to replay (e.g. on restart).

    Returns:
        The :class:`ReplayedKillState` the history folds to.
    """
    # The fold tracks the *last-in-iteration-order* kill, which equals the
    # highest kill_sequence only under the ledger's single-writer, append-only,
    # strictly-increasing invariant (KillSwitch.kill increments then records).
    # That invariant holds for every stream this replays; were a future
    # reordered/compacted ledger to break it, an older KillEngaged landing last
    # could let its matching (older) re-arm phrase clear the flag -- so the
    # ordering guarantee is load-bearing for this fail-closed control.
    last_kill_sequence = _NO_KILL_SEQUENCE
    killed = False
    for event in events:
        if event.event_type == _KILL_ENGAGED_EVENT_TYPE:
            last_kill_sequence = cast("int", event.payload["kill_sequence"])
            killed = True
        elif event.event_type == _KILL_REARMED_EVENT_TYPE and (
            cast("int", event.payload["kill_sequence"]) == last_kill_sequence
        ):
            killed = False
    return ReplayedKillState(last_kill_sequence=last_kill_sequence, killed=killed)


class KillTrigger(enum.Enum):
    """The four named sources that can engage the kill switch (issue #35).

    Each names *why* a kill fired; the resulting :class:`KillEngaged` event
    carries the trigger's ``name`` so the audit trail records the source.
    """

    CLI = enum.auto()
    KILL_FILE = enum.auto()
    DASHBOARD = enum.auto()
    AUTO_RECONCILIATION = enum.auto()


class AlertDispatcherProtocol(Protocol):
    """The narrow seam the kill switch dispatches its one alert through.

    Structural (mirroring
    :class:`~windbreak.riskkernel.process.KernelLedgerWriter`) so any object with
    a matching :meth:`dispatch` -- the real
    :class:`~windbreak.alerts.dispatch.AlertDispatcher` or a test double -- fits
    without inheritance.
    """

    def dispatch(self, alert_type: AlertType, message: str) -> object:
        """Dispatch one operator alert.

        Args:
            alert_type: The alert type to fire.
            message: The human-readable alert body.

        Returns:
            An implementation-defined value the kill switch ignores. Typed
            ``object`` so both the real
            :class:`~windbreak.alerts.dispatch.AlertDispatcher` (which returns
            an ``AlertEmitted`` record) and a ``None``-returning test double
            structurally satisfy this seam.
        """
        ...


class DirectiveSink(Protocol):
    """The seam a :class:`CancelAllDirective` is delivered to (order gateway).

    Structural, so the kill switch can hand its one cancel-all directive to the
    order-gateway-facing boundary without importing it.
    """

    def submit(self, directive: CancelAllDirective) -> None:
        """Submit a cancel-all directive for delivery.

        Args:
            directive: The directive to deliver downstream.
        """
        ...


class KillSwitch:
    """The Risk Kernel's single kill executor and re-arm authority (issue #35).

    Every trigger funnels into :meth:`kill`; the only way back out of
    ``KILLED`` is :meth:`rearm` with the exact typed confirmation phrase. The
    switch composes the LOCKED :mod:`~windbreak.riskkernel.modes` primitives and
    never widens the mode ladder.

    Kill state is durable, not merely process memory (issue #123): the
    monotonic kill sequence is recovered from the ledger via
    :meth:`from_events` (folding :func:`kill_state_in`), so a switch rebuilt
    after a restart re-arms the exact ledgered kill and its next kill still
    increments monotonically. The ``KILL`` file remains as belt-and-suspenders
    for a ledger-less restart.
    """

    def __init__(
        self,
        mode_machine: ModeStateMachine,
        ledger_writer: KernelLedgerWriter,
        alert_dispatcher: AlertDispatcherProtocol,
        *,
        reservation_ledger: ReservationLedger | None = None,
        directive_sink: DirectiveSink | None = None,
        state_dir: Path | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        """Wire the kill switch to the mode machine and its optional effect seams.

        Args:
            mode_machine: The LOCKED operating-mode state machine the switch
                drives to ``KILLED`` (and, on re-arm, back to ``PAUSED``).
            ledger_writer: The seam every kill-path event is recorded through.
            alert_dispatcher: The seam the one ``HALT_KILL`` alert fires through.
            reservation_ledger: The capital ledger whose active reservations are
                released on kill, or ``None`` to release nothing.
            directive_sink: The order-gateway seam the cancel-all directive is
                delivered to, or ``None`` to only ledger it.
            state_dir: The directory a ``KILL`` file is written into on kill, or
                ``None`` to write no file.
            clock: A zero-argument callable returning the current epoch second,
                injected so ``KillEngaged.epoch`` is deterministic under test.
                Defaults to :func:`_default_clock`.
        """
        self._mode_machine = mode_machine
        self._ledger_writer = ledger_writer
        self._alert_dispatcher = alert_dispatcher
        self._reservation_ledger = reservation_ledger
        self._directive_sink = directive_sink
        self._state_dir = state_dir
        self._clock = clock if clock is not None else _default_clock
        self._kill_sequence = _NO_KILL_SEQUENCE

    @classmethod
    def from_events(
        cls,
        events: Iterable[Event],
        mode_machine: ModeStateMachine,
        ledger_writer: KernelLedgerWriter,
        alert_dispatcher: AlertDispatcherProtocol,
        *,
        reservation_ledger: ReservationLedger | None = None,
        directive_sink: DirectiveSink | None = None,
        state_dir: Path | None = None,
        clock: Callable[[], int] | None = None,
    ) -> KillSwitch:
        """Rebuild a kill switch, restoring its kill sequence from the ledger.

        The kill sequence is durable, ledgered state, so a switch rebuilt over a
        recorded history restores :attr:`active_kill_sequence` from the latest
        :class:`KillEngaged` -- *unconditionally*, even when the net replayed
        state is not killed (a matching re-arm cleared it) -- so a post-restart
        :meth:`kill` still increments monotonically to ``N + 1`` rather than
        resetting to 1. It restores only the switch's own counter: it does
        **not** transition ``mode_machine`` to ``KILLED``. Driving the shared
        machine to ``KILLED`` on an unrearmed history is
        :meth:`~windbreak.riskkernel.process.RiskKernel.from_events`'s sole
        responsibility, so the two never race to transition the one machine.

        Args:
            events: The event history to restore the kill sequence from.
            mode_machine: The LOCKED operating-mode state machine (adopted, not
                transitioned here).
            ledger_writer: The seam the rebuilt switch records new events through.
            alert_dispatcher: The seam the rebuilt switch fires its one alert
                through.
            reservation_ledger: The capital ledger released on a later kill, or
                ``None``.
            directive_sink: The order-gateway seam for a later cancel-all, or
                ``None``.
            state_dir: The directory a later kill writes a ``KILL`` file into, or
                ``None``.
            clock: The rebuilt switch's injected epoch-second clock, or ``None``
                for :func:`_default_clock`.

        Returns:
            A :class:`KillSwitch` whose :attr:`active_kill_sequence` reflects
            ``events``.
        """
        switch = cls(
            mode_machine,
            ledger_writer,
            alert_dispatcher,
            reservation_ledger=reservation_ledger,
            directive_sink=directive_sink,
            state_dir=state_dir,
            clock=clock,
        )
        switch._kill_sequence = kill_state_in(events).last_kill_sequence
        return switch

    @property
    def mode(self) -> Mode:
        """Return the mode machine's current mode (read-only)."""
        return self._mode_machine.mode

    @property
    def active_kill_sequence(self) -> int:
        """Return the current kill's monotonic sequence number (valid in KILLED)."""
        return self._kill_sequence

    def kill(self, trigger: KillTrigger) -> None:
        """Engage the kill switch, driving the machine to ``KILLED`` fail-safe.

        Transitions to ``KILLED`` *first* (the fail-safe direction: even if a
        later effect raised, the machine is already halted), then records the
        closed kill-effect surface and fires the one alert. Idempotent: a kill
        while already ``KILLED`` is a pure no-op -- no event, no directive, no
        release, no alert, no raise -- so a persistent ``KILL`` file or a
        repeated trigger kills exactly once.

        Args:
            trigger: The source engaging the kill; its ``name`` is recorded on
                the :class:`KillEngaged` event.
        """
        if self._mode_machine.mode is Mode.KILLED:
            return
        self._mode_machine.transition(Mode.KILLED)
        self._kill_sequence += 1
        self._ledger_writer.record(
            KillEngaged(
                component=_COMPONENT,
                trigger=trigger.name,
                kill_sequence=self._kill_sequence,
                epoch=self._clock(),
            )
        )
        self._emit_cancel_all()
        self._release_reservations()
        self._alert_dispatcher.dispatch(AlertType.HALT_KILL, _HALT_KILL_MESSAGE)
        self._write_kill_file()

    def expected_rearm_phrase(self, kill_sequence: int) -> str:
        """Return the exact confirmation phrase required to re-arm a given kill.

        The phrase embeds ``kill_sequence`` so a stale confirmation typed for an
        earlier kill can never re-arm a later one, and carries cased characters
        so a case-folded near-miss is rejected by the verbatim compare in
        :meth:`rearm`.

        Args:
            kill_sequence: The kill sequence the phrase must confirm.

        Returns:
            The confirmation phrase for that kill sequence.
        """
        return f"RE-ARM KILL {kill_sequence}: I ACCEPT FULL RESPONSIBILITY"

    def rearm(self, confirmation: str) -> None:
        """Re-arm out of ``KILLED`` to ``PAUSED`` on the exact typed phrase.

        Valid only from ``KILLED`` and only for the verbatim (never case-folded)
        phrase :meth:`expected_rearm_phrase` returns for the active kill
        sequence. On success it composes the LOCKED mode primitives --
        ``rearm`` (``KILLED`` -> ``RESEARCH``) then ``transition`` to
        ``PAUSED`` -- and records one :class:`KillReArmed` event. A rejected
        re-arm ledgers nothing and leaves the mode unchanged.

        Args:
            confirmation: The typed confirmation phrase.

        Raises:
            KillReArmError: If not in ``KILLED``, or ``confirmation`` does not
                match the expected phrase exactly.
        """
        if self._mode_machine.mode is not Mode.KILLED:
            raise KillReArmError("rearm is only valid from KILLED")
        if confirmation != self.expected_rearm_phrase(self._kill_sequence):
            raise KillReArmError("rearm confirmation phrase does not match")
        self._mode_machine.rearm(modes.REARM_CONFIRMATION_PHRASE)
        self._mode_machine.transition(Mode.PAUSED)
        self._ledger_writer.record(
            KillReArmed(component=_COMPONENT, kill_sequence=self._kill_sequence)
        )

    def _emit_cancel_all(self) -> None:
        """Ledger the one cancel-all directive and hand it to any wired sink."""
        directive = CancelAllDirective(component=_COMPONENT, scope=_CANCEL_ALL_SCOPE)
        self._ledger_writer.record(directive)
        if self._directive_sink is not None:
            self._directive_sink.submit(directive)

    def _release_reservations(self) -> None:
        """Release every active reservation when a reservation ledger is wired."""
        if self._reservation_ledger is not None:
            self._reservation_ledger.release_all_active(reason=_RELEASE_REASON)

    def _write_kill_file(self) -> None:
        """Drop an empty ``KILL`` file when a state directory is wired.

        The file's mere presence -- never its content -- is a belt-and-
        suspenders kill signal a peer watcher or a ledger-less restart reads,
        backing the ledger replay that is now the primary durable mechanism
        (issue #123). The directory is
        created (parents included) first: a non-CLI trigger (dashboard,
        auto-reconciliation) can fire against a fresh ``state_dir`` before
        ``windbreak kill`` has ever run, and the fail-toward-dead file write must
        never be defeated by a missing directory.
        """
        if self._state_dir is not None:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._state_dir.joinpath(KILL_FILENAME).write_text("", encoding="utf-8")


class KillFileWatcher:
    """Polls a state dir for the ``KILL`` / ``REARM`` files (issue #35).

    One bounded :meth:`poll_once` per beat -- never a loop or sleep. A ``KILL``
    file kills on presence alone (its content is never read, so an unreadable
    file still kills: fail toward dead); a ``REARM`` file is always consumed
    whether or not its phrase re-arms, and a successful re-arm also clears the
    now-stale ``KILL`` file so a later poll cannot instantly re-kill.
    """

    def __init__(self, switch: KillSwitch, state_dir: Path) -> None:
        """Wire the watcher to a switch and the directory it polls.

        Args:
            switch: The kill switch this watcher engages and re-arms.
            state_dir: The directory scanned for ``KILL`` / ``REARM`` files.
        """
        self._switch = switch
        self._state_dir = state_dir

    def poll_once(self, now_epoch_s: int) -> None:
        """Scan the state dir once, killing or re-arming as its files dictate.

        Args:
            now_epoch_s: The current beat's epoch second. Unused -- the file
                protocol is presence-driven, not time-driven -- but accepted so
                the watcher shares the fleet's uniform per-beat call signature.
        """
        del now_epoch_s
        if self._switch.mode is not Mode.KILLED:
            if self._state_dir.joinpath(KILL_FILENAME).exists():
                self._switch.kill(KillTrigger.KILL_FILE)
            return
        self._consume_rearm_file()

    def _consume_rearm_file(self) -> None:
        """Attempt a re-arm from a ``REARM`` file, always consuming it.

        Delegates the read-and-re-arm to :meth:`_attempt_rearm` and *always*
        deletes the ``REARM`` file afterward -- in the ``finally``, so a phrase
        that is rejected, a file that is unreadable, or bytes that are not valid
        UTF-8 all still consume the file rather than wedge the beat. The stale
        ``KILL`` file is cleared only on a genuine re-arm, so any failure leaves
        the switch ``KILLED`` (fail toward dead) and a subsequent poll re-reads
        the still-present ``KILL`` file rather than coming back armed.
        """
        rearm_path = self._state_dir.joinpath(REARM_FILENAME)
        if not rearm_path.exists():
            return
        try:
            rearmed = self._attempt_rearm(rearm_path)
        finally:
            rearm_path.unlink(missing_ok=True)
        if rearmed:
            self._state_dir.joinpath(KILL_FILENAME).unlink(missing_ok=True)

    def _attempt_rearm(self, rearm_path: Path) -> bool:
        """Read the ``REARM`` phrase and try to re-arm, failing toward dead.

        Args:
            rearm_path: The ``REARM`` file to read the confirmation phrase from.

        Returns:
            ``True`` only when the switch actually re-armed. Any read failure
            (an unreadable file -- ``OSError`` -- or non-UTF-8 bytes --
            ``UnicodeError``) or a rejected phrase (``KillReArmError``) returns
            ``False`` without raising, so the caller keeps the kernel ``KILLED``
            and leaves the stale ``KILL`` file in place.
        """
        try:
            phrase = rearm_path.read_text(encoding="utf-8")
            self._switch.rearm(phrase)
        except (OSError, UnicodeError, KillReArmError):
            return False
        return True


class ReconciliationMismatchMonitor:
    """Auto-kills on consecutive reconciliation breaches (issue #35).

    Counts *consecutive* ``BREACH`` verification outcomes; any non-``BREACH``
    outcome resets the count, so only a sustained run of breaches -- not an
    isolated blip -- engages the kill switch.
    """

    def __init__(self, switch: KillSwitch, threshold: int) -> None:
        """Wire the monitor to a switch and its breach threshold.

        Args:
            switch: The kill switch engaged once the threshold is reached.
            threshold: The number of consecutive breaches that auto-kills
                (typically ``RiskConfig.kill_after_consecutive_mismatches``).
        """
        self._switch = switch
        self._threshold = threshold
        self._consecutive_breaches = 0

    def observe(self, outcome: VerificationOutcome) -> None:
        """Fold one verification outcome into the consecutive-breach count.

        Args:
            outcome: The latest verification outcome. A ``BREACH`` increments
                the run and may trip the threshold; anything else resets it.
        """
        if outcome is not VerificationOutcome.BREACH:
            self._consecutive_breaches = 0
            return
        self._consecutive_breaches += 1
        if self._consecutive_breaches >= self._threshold:
            self._switch.kill(KillTrigger.AUTO_RECONCILIATION)


class DashboardChallengeError(Exception):
    """Raised when a dashboard kill confirmation token is wrong or reused."""


class DashboardKillStub:
    """A one-time challenge/confirm kill handshake for the dashboard (issue #35).

    :meth:`request_challenge` mints a fresh single-use nonce; :meth:`confirm`
    kills only on that exact, still-unused token. A wrong or already-consumed
    token raises without killing, so an accidental or replayed dashboard click
    can never engage the kill switch.
    """

    def __init__(self, switch: KillSwitch) -> None:
        """Wire the handshake to the switch it engages.

        Args:
            switch: The kill switch a confirmed challenge engages.
        """
        self._switch = switch
        self._pending_token: str | None = None

    def request_challenge(self) -> str:
        """Mint and remember a fresh single-use confirmation nonce.

        Returns:
            A URL-safe random token the caller must echo back to
            :meth:`confirm`.
        """
        token = secrets.token_urlsafe(_CHALLENGE_TOKEN_BYTES)
        self._pending_token = token
        return token

    def confirm(self, token: str) -> None:
        """Kill the switch iff ``token`` is the exact, unused pending nonce.

        The compare is constant-time (:func:`secrets.compare_digest`) and the
        pending nonce is cleared before the kill, so a token confirms at most
        once.

        Args:
            token: The nonce to confirm against the pending challenge.

        Raises:
            DashboardChallengeError: If no challenge is pending, or ``token``
                does not match the pending nonce. The switch is not engaged.
        """
        pending = self._pending_token
        if pending is None or not secrets.compare_digest(token, pending):
            raise DashboardChallengeError(
                "dashboard kill confirmation token is invalid"
            )
        self._pending_token = None
        self._switch.kill(KillTrigger.DASHBOARD)


@dataclass(frozen=True, slots=True)
class KillIntegration:
    """Bundles the kill switch and its trigger adapters (issue #35).

    The single value the Risk Kernel process holds: it polls ``watcher`` each
    beat and feeds ``monitor`` each verification outcome, both optional so a
    kernel can wire in only the triggers it needs.

    Attributes:
        switch: The kill executor.
        watcher: The state-dir file watcher polled each beat, or ``None``.
        monitor: The reconciliation-mismatch monitor, or ``None``.
    """

    switch: KillSwitch
    watcher: KillFileWatcher | None = None
    monitor: ReconciliationMismatchMonitor | None = None
