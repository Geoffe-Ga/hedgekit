"""Command-line entry point and heartbeat loop for windbreak.

The ``windbreak run`` command starts the always-on RESEARCH-mode heartbeat
loop that later issues will grow into the full four-process pipeline. For now
it emits a periodic heartbeat line and shuts down cleanly on SIGINT, SIGTERM,
or an optional beat budget.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from windbreak.alerts import AlertDispatcher, AlertType, LoggingLedgerWriter, cli_token
from windbreak.alerts.factory import AlertSinkConfigError, build_sinks
from windbreak.config import (
    ConfigError,
    InMemoryConfigEventRecorder,
    LedgerConfigEventRecorder,
    config_hash,
    load_config,
    load_default_config,
)
from windbreak.drills.catalog import DRILL_NAMES
from windbreak.drills.context import bind_paper_context, bind_production_context
from windbreak.ledger import (
    ChainIntegrityError,
    SqliteLedgerStore,
    anchor_command,
    events_from_records,
    rebuild_command,
    verify_command,
)
from windbreak.logging_setup import configure_logging
from windbreak.riskkernel.ack_flow import ACKS_DIRNAME
from windbreak.riskkernel.kill import KILL_FILENAME, REARM_FILENAME

if TYPE_CHECKING:
    import http.server
    from collections.abc import Callable, Mapping, Sequence
    from types import FrameType

    from windbreak.config import ConfigLoadEvent, ScreenerConfig, WindbreakConfig
    from windbreak.connector.interface import MarketConnector
    from windbreak.connector.live import MarketDataSource
    from windbreak.dashboard.app import DashboardStatus
    from windbreak.ledger import Event, LedgerStore
    from windbreak.net.live_http import LiveHttpTransport
    from windbreak.riskkernel.kill import KillIntegration
    from windbreak.riskkernel.process import KernelLedgerWriter, RiskKernel
    from windbreak.riskkernel.verification import ReadOnlyVerifier
    from windbreak.scheduler.provider_wiring import LiveProviderHttp

#: Operating mode reported in every heartbeat line. Matches the RESEARCH state
#: of the SPEC mode machine; windbreak ships research-only for now.
MODE_RESEARCH = "RESEARCH"

#: The four SPEC processes ``windbreak run --process`` can represent, in SPEC
#: order. Each invocation stands in for exactly one; the chosen token is
#: stamped as the ``component`` on every heartbeat and shutdown log line. The
#: gateway token is underscore-separated (``order_gateway``) to match its
#: Python package name, even though its compose/systemd unit names are
#: hyphenated (``order-gateway``).
PROCESS_CHOICES = ("pipeline", "riskkernel", "order_gateway", "dashboard")

#: Default process represented when ``--process`` is omitted.
_DEFAULT_PROCESS = "pipeline"

#: Environment variable the loopback dashboard's bearer token is minted from
#: (issue #79). Never sourced from config: config is ledgered, so a secret held
#: there would leak into the hash chain; the token lives only in the process
#: environment.
DASHBOARD_AUTH_ENV_VAR = "WINDBREAK_DASHBOARD_TOKEN"

#: The ``component`` label stamped on the dashboard process's serve and shutdown
#: log lines, matching its ``--process dashboard`` token.
_DASHBOARD_COMPONENT = "dashboard"

#: Bounded join timeout, in seconds, for the dashboard's serving thread on
#: shutdown, so a wedged thread can never hang the process indefinitely.
_DASHBOARD_THREAD_JOIN_TIMEOUT_SECONDS = 5.0

#: Seconds between heartbeats when ``--heartbeat-interval`` is omitted.
_DEFAULT_HEARTBEAT_INTERVAL = 5.0

#: Default alert body dispatched by the ``alert-test`` subcommand.
_DEFAULT_ALERT_MESSAGE = "test alert"

#: The environment variable a leaked trade key would surface in; the preflight
#: leak check (SPEC S5.2) fails closed if it is visible to this process.
_TRADE_KEY_ENV_VAR = "WINDBREAK_TRADE_KEY"

#: Default fixture/state directories for the ``drill`` verb when unspecified.
_DEFAULT_DRILL_FIXTURE_DIR = Path("drills/fixtures")
_DEFAULT_DRILL_STATE_DIR = Path("drills/state")

#: Maps each alert's CLI token back to its :class:`AlertType` member.
_TOKEN_TO_ALERT_TYPE = {cli_token(alert_type): alert_type for alert_type in AlertType}

#: Shutdown reason logged when the loop exhausts its ``--max-beats`` budget.
_REASON_MAX_BEATS = "max_beats"

#: Shutdown reason logged when the loop is stopped via its stop event.
_REASON_SIGNAL = "signal"

#: Log-friendly source label for a configuration built from built-in defaults.
_DEFAULTS_SOURCE_LABEL = "<defaults>"

#: An approval id is exactly 32 lowercase hex characters -- the shape
#: ``HumanAckQueue`` mints via ``secrets.token_hex(16)`` -- so the ``ack`` verb
#: rejects any other token as a usage error before writing a bogus drop-box file.
_APPROVAL_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

_LOGGER = logging.getLogger("windbreak")


def _approval_id(raw: str) -> str:
    """Parse a 32-hex-character approval id for use as an argparse ``type``.

    Args:
        raw: The raw command-line token.

    Returns:
        The validated approval id, unchanged.

    Raises:
        argparse.ArgumentTypeError: If ``raw`` is not exactly 32 lowercase hex
            characters.
    """
    if _APPROVAL_ID_PATTERN.fullmatch(raw) is None:
        raise argparse.ArgumentTypeError(
            "approval id must be exactly 32 lowercase hex characters"
        )
    return raw


@dataclass
class ShutdownState:
    """Shared mutable state coordinating a graceful shutdown.

    Attributes:
        stop_event: Set to request the heartbeat loop stop.
        reason: Name of the signal that triggered shutdown, or None while
            the loop is still running.
    """

    stop_event: threading.Event = field(default_factory=threading.Event)
    reason: str | None = None


def _non_negative_float(raw: str) -> float:
    """Parse a non-negative float for use as an argparse ``type``.

    Args:
        raw: The raw command-line token.

    Returns:
        The parsed floating-point value.

    Raises:
        argparse.ArgumentTypeError: If ``raw`` is not a float or is negative.
    """
    value = float(raw)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("heartbeat interval must be finite")
    if value < 0:
        raise argparse.ArgumentTypeError("heartbeat interval must be non-negative")
    return value


def _non_negative_int(raw: str) -> int:
    """Parse a non-negative int for use as an argparse ``type``.

    Args:
        raw: The raw command-line token.

    Returns:
        The parsed integer value.

    Raises:
        argparse.ArgumentTypeError: If ``raw`` is not an int or is negative.
    """
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("max beats must be non-negative")
    return value


def _add_run_arguments(run_parser: argparse.ArgumentParser) -> None:
    """Register the ``run`` subcommand's options on its subparser.

    Args:
        run_parser: The ``run`` subparser to populate with options.
    """
    run_parser.add_argument(
        "--heartbeat-interval",
        type=_non_negative_float,
        default=_DEFAULT_HEARTBEAT_INTERVAL,
        help="Seconds between heartbeats (default: %(default)s).",
    )
    run_parser.add_argument(
        "--max-beats",
        type=_non_negative_int,
        default=None,
        help="Stop after this many heartbeats (default: run until signalled).",
    )
    run_parser.add_argument(
        "--process",
        choices=PROCESS_CHOICES,
        default=_DEFAULT_PROCESS,
        help="Which SPEC process this invocation represents (default: %(default)s).",
    )
    run_parser.add_argument(
        "--snapshot-fixture-dir",
        default=None,
        help=(
            "Directory of exchange JSON fixtures to snapshot each beat; for "
            "--process riskkernel it is the read-only exchange the kernel's "
            "verifier observes each beat (default: snapshotting is off)."
        ),
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a SPEC §16 YAML config (default: built-in §16 defaults).",
    )
    _add_paper_loop_arguments(run_parser)


def _add_paper_loop_arguments(run_parser: argparse.ArgumentParser) -> None:
    """Register the always-on PAPER-loop composition flags (issues #48, #343).

    PAPER activates only when the mode ceiling permits PAPER *and* all four
    composition flags are supplied; each defaults to ``None`` so omitting any one
    leaves the loop in its byte-identical RESEARCH-only behavior.
    ``--paper-live-ticker`` is a fifth, optional flag selecting live venue books
    for an already-activated loop.

    Args:
        run_parser: The ``run`` subparser to populate with the PAPER flags.
    """
    run_parser.add_argument(
        "--paper-books-dir",
        type=Path,
        default=None,
        help=(
            "Paper-exchange fixture directory (default: PAPER loop off). With "
            "--paper-live-ticker only its account fixtures (opening balances "
            "and balance semantics) are read; the market data is the venue's."
        ),
    )
    run_parser.add_argument(
        "--paper-live-ticker",
        default=None,
        help=(
            "Read this market off the exchange's LIVE order books, with fills "
            "still simulated (default: replay the fixture directory's recorded "
            "books). One market only; read-only market data, no credentials, "
            "and no order ever leaves the process. This names the session's "
            "market universe, not a guaranteed trade: the loop screens it every "
            "tick and will not forecast a market that fails the screen."
        ),
    )
    run_parser.add_argument(
        "--cassette-path",
        type=Path,
        default=None,
        help="Recorded LLM cassette for the offline forecast replay transport.",
    )
    run_parser.add_argument(
        "--ledger-path",
        type=Path,
        default=None,
        help=(
            "Path to the operational hash-chained ledger database. Every "
            "successful config load records a ConfigLoaded event here; the "
            "PAPER loop, when activated, appends its events to the same file."
        ),
    )
    run_parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory the weekly PAPER report stub is written into.",
    )


def _add_rebuild_arguments(rebuild_parser: argparse.ArgumentParser) -> None:
    """Register the ``rebuild`` subcommand's options on its subparser.

    Args:
        rebuild_parser: The ``rebuild`` subparser to populate with options.
    """
    rebuild_parser.add_argument(
        "--ledger-path",
        type=Path,
        required=True,
        help="Path to the SQLite ledger database.",
    )
    rebuild_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the read-model files into.",
    )


def _add_anchor_arguments(anchor_parser: argparse.ArgumentParser) -> None:
    """Register the ``anchor`` subcommand's options on its subparser.

    Args:
        anchor_parser: The ``anchor`` subparser to populate with options.
    """
    anchor_parser.add_argument(
        "--ledger-path",
        type=Path,
        required=True,
        help="Path to the SQLite ledger database.",
    )
    anchor_parser.add_argument(
        "--anchor-path",
        type=Path,
        required=True,
        help="Path to the append-only JSON-lines anchor file.",
    )


def _add_verify_arguments(verify_parser: argparse.ArgumentParser) -> None:
    """Register the ``verify`` subcommand's options on its subparser.

    Args:
        verify_parser: The ``verify`` subparser to populate with options.
    """
    verify_parser.add_argument(
        "--ledger-path",
        type=Path,
        required=True,
        help="Path to the SQLite ledger database.",
    )
    verify_parser.add_argument(
        "--anchor-path",
        type=Path,
        required=True,
        help="Path to the append-only JSON-lines anchor file.",
    )


def _add_kill_arguments(kill_parser: argparse.ArgumentParser) -> None:
    """Register the ``kill`` subcommand's options on its subparser.

    Args:
        kill_parser: The ``kill`` subparser to populate with options.
    """
    kill_parser.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="Directory to write the KILL file into.",
    )


def _add_ack_arguments(ack_parser: argparse.ArgumentParser) -> None:
    """Register the ``ack`` subcommand's options on its subparser.

    Args:
        ack_parser: The ``ack`` subparser to populate with options.
    """
    ack_parser.add_argument(
        "--approval-id",
        type=_approval_id,
        required=True,
        help="The 32-hex-character approval id to acknowledge.",
    )
    ack_parser.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="Directory whose acks/ drop-box the ack file is written into.",
    )


def _add_rearm_arguments(rearm_parser: argparse.ArgumentParser) -> None:
    """Register the ``rearm`` subcommand's options on its subparser.

    Args:
        rearm_parser: The ``rearm`` subparser to populate with options.
    """
    rearm_parser.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="Directory to write the REARM file into.",
    )


def _add_preflight_arguments(preflight_parser: argparse.ArgumentParser) -> None:
    """Register the ``preflight`` subcommand's options on its subparser.

    Args:
        preflight_parser: The ``preflight`` subparser to populate with options.
    """
    preflight_parser.add_argument(
        "--fixture-dir",
        type=Path,
        required=True,
        help="Directory of exchange JSON fixtures to run the checks against.",
    )
    preflight_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as a JSON document instead of a table.",
    )
    preflight_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a SPEC §16 YAML config (default: built-in §16 defaults).",
    )
    preflight_parser.add_argument(
        "--secrets-file",
        type=Path,
        action="append",
        default=None,
        help="A secrets file whose permissions to check (repeatable).",
    )


def _add_drill_arguments(drill_parser: argparse.ArgumentParser) -> None:
    """Register the ``drill`` subcommand's options on its subparser.

    Args:
        drill_parser: The ``drill`` subparser to populate with options.
    """
    drill_parser.add_argument(
        "name",
        choices=sorted(DRILL_NAMES),
        help="Which operational drill to run.",
    )
    drill_parser.add_argument(
        "--production",
        action="store_true",
        help=(
            "Rebind only the exchange adapter for a manual production run "
            "(requires a non-empty exchange credential in the environment; "
            "rebinds a fresh stub exchange until a live adapter lands). "
            "Default: paper."
        ),
    )
    drill_parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=_DEFAULT_DRILL_FIXTURE_DIR,
        help="Directory of drill fixtures (default: %(default)s).",
    )
    drill_parser.add_argument(
        "--state-dir",
        type=Path,
        default=_DEFAULT_DRILL_STATE_DIR,
        help="Directory for drill protocol/scratch files (default: %(default)s).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``windbreak`` command-line argument parser.

    Returns:
        A parser with a required ``run`` subcommand exposing
        ``--heartbeat-interval``, ``--max-beats``, ``--process``,
        ``--snapshot-fixture-dir``, and ``--config``; a ``rebuild`` subcommand
        exposing ``--ledger-path`` and ``--output-dir``; ``anchor`` and
        ``verify`` subcommands exposing ``--ledger-path`` and ``--anchor-path``
        (append the ledger head to, and check the live chain against, the
        anchor file); ``kill`` and ``rearm`` subcommands exposing
        ``--state-dir``; an ``ack`` subcommand exposing ``--approval-id`` and
        ``--state-dir``; and a developer-only ``alert-test`` subcommand hidden
        from ``--help``.
    """
    parser = argparse.ArgumentParser(
        prog="windbreak",
        description="windbreak always-on forecast trader CLI.",
    )
    # ``metavar`` keeps the auto-generated ``{run,alert-test}`` choice list --
    # which would otherwise leak the hidden ``alert-test`` command -- out of the
    # top-level usage line. The ``alert-test`` parser below is registered without
    # a ``help`` argument, so argparse creates no pseudo-action for it and it is
    # omitted from the detailed subcommand listing (a developer-only command).
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    _add_run_arguments(subparsers.add_parser("run", help="Start the heartbeat loop."))
    _add_rebuild_arguments(
        subparsers.add_parser(
            "rebuild", help="Rebuild derived read models from the ledger."
        )
    )
    _add_anchor_arguments(
        subparsers.add_parser(
            "anchor", help="Append the ledger's head hash to the anchor file."
        )
    )
    _add_verify_arguments(
        subparsers.add_parser(
            "verify", help="Verify the ledger's live chain against its anchors."
        )
    )
    _add_kill_arguments(
        subparsers.add_parser(
            "kill", help="Engage the kill switch (write a KILL file)."
        )
    )
    _add_ack_arguments(
        subparsers.add_parser(
            "ack",
            help="Grant a human acknowledgement (write an acks/<id> file).",
        )
    )
    _add_rearm_arguments(
        subparsers.add_parser(
            "rearm",
            help="Re-arm after a kill (write the typed phrase to a REARM file).",
        )
    )
    _add_preflight_arguments(
        subparsers.add_parser(
            "preflight",
            help="Run the production-readiness preflight checklist.",
        )
    )
    _add_drill_arguments(
        subparsers.add_parser(
            "drill",
            help="Run an operational drill (rehearse a safety mechanism).",
        )
    )
    alert_parser = subparsers.add_parser("alert-test")
    alert_parser.add_argument(
        "type",
        choices=[cli_token(alert_type) for alert_type in AlertType],
        help="Alert type (as a CLI token) to emit a test alert for.",
    )
    alert_parser.add_argument(
        "--message",
        default=_DEFAULT_ALERT_MESSAGE,
        help="Alert body to dispatch (default: %(default)s).",
    )
    alert_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a SPEC §16 YAML config whose alert sinks to dispatch "
        "through (default: built-in §16 defaults, i.e. log-only).",
    )
    return parser


def run_loop(
    interval_seconds: float,
    *,
    max_beats: int | None = None,
    stop_event: threading.Event | None = None,
    state: ShutdownState | None = None,
    component: str = _DEFAULT_PROCESS,
    on_beat: Callable[[int], None] | None = None,
) -> None:
    """Emit heartbeats until stopped by the stop event or a beat budget.

    Args:
        interval_seconds: Seconds to wait between heartbeats. Passed to
            ``stop_event.wait`` unmodified.
        max_beats: Optional maximum number of heartbeats before shutting down
            with reason ``max_beats``. None runs until the stop event is set.
        stop_event: Optional event used to request shutdown. Defaults to
            ``state.stop_event`` when ``state`` is given, else a fresh event.
        state: Optional shared shutdown state. When a signal handler has
            recorded a signal name on it, that name becomes the shutdown
            reason; otherwise the generic ``signal`` reason is used.
        component: Which SPEC process (one of :data:`PROCESS_CHOICES`) this
            loop represents. Stamped as the ``component`` extra on every
            heartbeat and shutdown log record; the rendered message text is
            unchanged.
        on_beat: Optional hook invoked once per beat with the 1-based sequence
            number, after that beat's heartbeat is logged. None (the default)
            leaves the heartbeat behavior unchanged.
    """
    if stop_event is None:
        stop_event = state.stop_event if state is not None else threading.Event()

    seq = 0
    reason = _REASON_SIGNAL
    while True:
        if stop_event.is_set():
            if state is not None and state.reason is not None:
                reason = state.reason
            break
        if max_beats is not None and seq >= max_beats:
            reason = _REASON_MAX_BEATS
            break
        seq += 1
        _LOGGER.info(
            "mode=%s heartbeat seq=%d",
            MODE_RESEARCH,
            seq,
            extra={"component": component},
        )
        if on_beat is not None:
            on_beat(seq)
        stop_event.wait(interval_seconds)

    _LOGGER.info("shutdown reason=%s", reason, extra={"component": component})


def _install_signal_handlers(state: ShutdownState) -> None:
    """Install SIGINT/SIGTERM handlers that request a graceful shutdown.

    The installed handler is directly invokable as ``handler(signum, frame)``.
    On delivery it records the signal name on ``state.reason`` and sets
    ``state.stop_event`` so an in-flight :func:`run_loop` unwinds cleanly.

    Args:
        state: Shared shutdown state mutated when a signal arrives.
    """

    def _handle(signum: int, _frame: FrameType | None) -> None:
        """Record the signal name and request shutdown."""
        state.reason = signal.Signals(signum).name
        state.stop_event.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _load_configured(
    args: argparse.Namespace, recorder: InMemoryConfigEventRecorder
) -> WindbreakConfig:
    """Load the config named by ``--config``, or the built-in defaults.

    Args:
        args: The parsed CLI arguments; ``args.config`` is a path or None.
        recorder: The recorder notified of the resulting hash and diff.

    Returns:
        The loaded configuration.

    Raises:
        ConfigError: If a ``--config`` path cannot be read or validated.
    """
    if args.config is not None:
        return load_config(args.config, recorder=recorder)
    return load_default_config(recorder=recorder)


def _build_alert_dispatcher(config: WindbreakConfig) -> AlertDispatcher:
    """Compose the alert dispatcher from the configured sinks (issue #274).

    The single place the CLI turns ``alerts.sinks`` into live delivery channels,
    screened against the deployment's own egress allowlist. Every process shares
    it so no composition site can quietly regress to the log-only fallback the
    way all three did before this existed. It is also the only place the real
    :data:`os.environ` is read for alert destinations: configuration names the
    variables, this seam resolves them, so no destination is ever a config leaf
    the ledger would persist in plaintext.

    Args:
        config: The loaded configuration whose alerts section supplies the
            sinks and whose whole shape supplies the egress allowlist.

    Returns:
        A dispatcher over every deliverable configured sink. With none
        configured the sink list is empty and the dispatcher's ``log-only``
        fallback carries the alert.

    Raises:
        AlertSinkConfigError: If a sink names an unknown type, names a
            destination environment variable that is unset or empty, targets a
            host outside ``alerts.allowed_hosts``, or cannot deliver as
            configured.
    """
    from windbreak.net.allowlist import allowlist_from_config

    return AlertDispatcher(
        sinks=build_sinks(
            config.alerts,
            allowlist=allowlist_from_config(config),
            environ=os.environ,
        ),
        ledger_writer=LoggingLedgerWriter(),
    )


def _run_preflight(args: argparse.Namespace) -> int:
    """Run the production-readiness checklist and print its report (SPEC S3.3).

    Builds the injected seams -- a fixture-backed read-only connector, an honest
    no-self-test scope prober (the real self-test client is issue #57), a
    trade-key environment-leak prober over :data:`os.environ`, an alert
    dispatcher over the *configured* sinks (issue #274; log-only when none are
    configured), and the configured secrets paths -- runs the seven checks, and
    prints the report as JSON or a table to stdout. The connector and preflight
    imports are local so the RESEARCH heartbeat path never imports them.

    Args:
        args: Parsed ``preflight`` arguments carrying ``fixture_dir``, ``json``,
            ``config``, and ``secrets_file``.

    Returns:
        The report's fail-closed exit code (0 on all-pass, 1 on any failure), or
        1 on a fatal ``--config`` error.
    """
    from windbreak.connector import FakeExchange
    from windbreak.preflight import (
        EnvTradeKeyLeakProber,
        KeyScopeProbe,
        render_table,
        report_to_json,
        run_preflight,
    )

    class _NullScopeProber:
        """A scope prober reporting no self-test support (real one: issue #57)."""

        def probe(self) -> KeyScopeProbe:
            """Return an all-unsupported probe so scope checks honestly SKIP."""
            return KeyScopeProbe(
                self_test_supported=False,
                scope_verified=False,
                withdrawal_capable=False,
            )

    recorder = InMemoryConfigEventRecorder()
    try:
        config = _load_configured(args, recorder)
    except ConfigError as exc:
        _LOGGER.critical("FATAL: %s", exc)
        return 1
    connector = FakeExchange.from_fixture_dir(args.fixture_dir)
    try:
        dispatcher = _build_alert_dispatcher(config)
    except AlertSinkConfigError as exc:
        _LOGGER.critical("FATAL: %s", exc)
        return 1
    report = run_preflight(
        connector=connector,
        scope_prober=_NullScopeProber(),
        leak_prober=EnvTradeKeyLeakProber(environ=os.environ, var=_TRADE_KEY_ENV_VAR),
        eligible_markets=connector.list_markets(),
        alert_dispatcher=dispatcher,
        secrets_paths=tuple(args.secrets_file or ()),
        config=config,
    )
    print(report_to_json(report) if args.json else render_table(report))
    return report.exit_code


def _epoch_now() -> int:
    """Return the current wall clock as whole epoch seconds (SPEC S6.1).

    Casts :func:`time.time` to an ``int`` so the drill clock is float-free; this
    is the CLI's one reading of the wall clock, injected into the drill context
    so the drills themselves never call :func:`time.time`.

    Returns:
        The current time, in whole epoch seconds.
    """
    return int(time.time())


def _run_drill(args: argparse.Namespace) -> int:
    """Run one operational drill and map its verdict to an exit code (issue #59).

    Builds the deterministic paper context from the injected wall clock and the
    real process environment (the CLI's one reading of each), rebinding only the
    exchange adapter when ``--production`` is set, then runs the named drill and
    ledgers exactly one ``DrillCompleted``. The heavy registry/framework imports
    are local so the RESEARCH heartbeat path never imports them.

    Args:
        args: Parsed ``drill`` arguments carrying ``name``, ``production``,
            ``fixture_dir``, and ``state_dir``.

    Returns:
        ``0`` iff the drill passed, else ``1``.
    """
    from windbreak.drills.framework import run_drill
    from windbreak.drills.registry import DRILLS
    from windbreak.riskkernel.process import LoggingKernelLedgerWriter

    writer = LoggingKernelLedgerWriter()
    paper_ctx = bind_paper_context(
        fixture_dir=args.fixture_dir,
        state_dir=args.state_dir,
        ledger_writer=writer,
        clock=_epoch_now,
        env=os.environ,
    )
    ctx = (
        bind_production_context(paper_ctx, env=os.environ)
        if args.production
        else paper_ctx
    )
    result = run_drill(DRILLS[args.name](), ctx, writer)
    return 0 if result.passed else 1


def _run_alert_test(args: argparse.Namespace) -> int:
    """Dispatch a single test alert through the configured sinks.

    The operator's end-to-end proof that alerting works: with ``--config`` the
    alert travels the real sinks that config declares, and with none (or none
    configured) the dispatcher's log-only fallback fires instead. Either way the
    ledger writer logs the resulting :class:`~windbreak.alerts.AlertEmitted`
    event, whose per-sink outcomes are observable as JSON on stderr -- so a sink
    that silently fails to deliver is visible here rather than during an
    incident.

    Args:
        args: Parsed ``alert-test`` arguments carrying ``type``, ``message``,
            and ``config``.

    Returns:
        The process exit code: 0 on dispatch, 1 on a fatal ``--config`` or
        sink-configuration error.
    """
    alert_type = _TOKEN_TO_ALERT_TYPE[args.type]
    recorder = InMemoryConfigEventRecorder()
    try:
        dispatcher = _build_alert_dispatcher(_load_configured(args, recorder))
    except (ConfigError, AlertSinkConfigError) as exc:
        _LOGGER.critical("FATAL: %s", exc)
        return 1
    dispatcher.dispatch(alert_type, args.message)
    return 0


def _run_kill(args: argparse.Namespace) -> int:
    """Engage the kill switch by dropping a ``KILL`` file into ``--state-dir``.

    The file's mere presence is the durable kill signal a running kernel's file
    watcher acts on; its content is never read, so an empty file suffices.

    Args:
        args: Parsed ``kill`` arguments carrying ``state_dir``.

    Returns:
        The process exit code (always 0).
    """
    args.state_dir.mkdir(parents=True, exist_ok=True)
    (args.state_dir / KILL_FILENAME).write_text("", encoding="utf-8")
    return 0


def _run_ack(args: argparse.Namespace) -> int:
    """Grant a human acknowledgement by dropping a file into ``acks/``.

    Writes an empty file at ``<state-dir>/acks/<approval-id>`` -- the
    presence-driven signal :class:`windbreak.riskkernel.ack_flow.AckFileWatcher`
    polls for, mirroring ``windbreak kill``'s ``KILL``-file convention. The
    approval id is already validated (32 hex chars) by the argparse ``type``, so
    a malformed id is rejected before this runs and no file is ever written. No
    network, no credentials.

    Args:
        args: Parsed ``ack`` arguments carrying ``approval_id`` and ``state_dir``.

    Returns:
        The process exit code (always 0).
    """
    acks_dir = args.state_dir / ACKS_DIRNAME
    acks_dir.mkdir(parents=True, exist_ok=True)
    (acks_dir / args.approval_id).write_text("", encoding="utf-8")
    return 0


def _run_rearm(args: argparse.Namespace) -> int:
    """Write the typed re-arm phrase *verbatim* to a ``REARM`` file.

    The phrase is written exactly as typed -- no stripping, no case change --
    because the kernel's re-arm compares it byte-for-byte against the expected
    confirmation, so any normalization here would silently break re-arm.

    Args:
        args: Parsed ``rearm`` arguments carrying ``state_dir``.

    Returns:
        The process exit code (always 0).
    """
    phrase = input()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    (args.state_dir / REARM_FILENAME).write_text(phrase, encoding="utf-8")
    return 0


def _build_snapshot_on_beat(
    fixture_dir: str, screener_config: ScreenerConfig
) -> Callable[[int], None]:
    """Build a per-beat hook that snapshots and screens a fixture-backed exchange.

    The connector/screener imports are local so the heartbeat path stays free of
    those packages unless snapshotting is actually requested. A single
    ``LoggingEventLedgerWriter`` is shared by the real :class:`Screener` (the
    single ``SCREEN_DECISION`` emitter) and the :class:`MarketSnapshotTask`, so
    both event kinds land in the same ledger.

    Args:
        fixture_dir: Directory of exchange JSON fixtures to snapshot.
        screener_config: The screening thresholds and blocklist to enforce.

    Returns:
        A callable that, given the beat sequence, runs one snapshot pass.
    """
    from datetime import UTC, datetime

    from windbreak.connector import (
        FakeExchange,
        LoggingEventLedgerWriter,
        MarketSnapshotTask,
    )
    from windbreak.screener import Screener

    writer = LoggingEventLedgerWriter()
    screener = Screener(screener_config, writer, clock=lambda: datetime.now(UTC))
    task = MarketSnapshotTask(
        FakeExchange.from_fixture_dir(fixture_dir),
        screener,
        writer,
    )

    def _on_beat(_seq: int) -> None:
        """Run one snapshot pass, ignoring the beat sequence number."""
        task.run_once()

    return _on_beat


def _paper_activated(config: WindbreakConfig, args: argparse.Namespace) -> bool:
    """Return whether the always-on PAPER loop should be wired this run (#48).

    PAPER activates only when the configured mode ceiling permits PAPER *and*
    every one of the four PAPER flags is supplied. A ``research`` ceiling -- even
    with all four flags -- never activates it (the tracer invariant), and neither
    do partial flags. The ceiling is parsed from the SPEC S16 token, whose four
    ladder values are the only valid ceilings; a non-``RESEARCH`` ceiling permits
    PAPER.

    Args:
        config: The loaded configuration whose ``mode_ceiling`` gates activation.
        args: The parsed ``run`` arguments carrying the four PAPER flags.

    Returns:
        ``True`` only when PAPER is permitted and all four flags are supplied.
    """
    from windbreak.riskkernel.modes import Mode

    flags = (
        args.paper_books_dir,
        args.cassette_path,
        args.ledger_path,
        args.report_dir,
    )
    if any(flag is None for flag in flags):
        return False
    return Mode.from_config(config.mode_ceiling) is not Mode.RESEARCH


def _resolve_paper_market_data(
    args: argparse.Namespace, config: WindbreakConfig
) -> MarketDataSource | None:
    """Resolve the PAPER loop's live venue surface, or ``None`` for fixtures.

    ``--paper-live-ticker`` is a single flag rather than a mode switch plus a
    market name, so the mode and the market cannot disagree: naming a market
    *is* selecting live books, and there is no combination that half-selects
    one. The loop then either gets both halves or neither (issue #343).

    Issue #345 gave the PAPER loop a real market universe, and it deliberately
    added no second flag. The venue surface this builds is a *one-market*
    universe -- :class:`~windbreak.connector.paper.LiveBookPaperExchange` binds
    exactly the named ticker -- so what the flag selects is unchanged; what
    changed is that the loop now *screens* that market every tick rather than
    assuming it. An operator who names a market with too little depth, too far
    a horizon, or a blocklisted category gets a ledgered
    ``ScreenDecisionRecorded`` saying so and no forecast, instead of a forecast
    on a market the §16 filters would have refused. Adding a second flag to
    widen the live universe would only reintroduce the two-flag combination
    issue #343 removed; widening it belongs to the venue surface, where the
    single-ticker limit actually lives.

    Nothing here holds a credential. Kalshi's market-data routes are public and
    :class:`~windbreak.connector.kalshi.client.KalshiClient` never attaches an
    auth header, so there is no ``*_api_key_env`` leaf to configure and no
    secret that could reach a log or the hash-chained ledger. The resolved base
    URL is screened against the deployment's own outbound allowlist before any
    session exists (SPEC S15), and the object handed back is a
    :class:`~windbreak.connector.live.MarketDataOnlyView` with no order surface
    at all (SPEC S1.1 invariant 3).

    Connector-side refusals and halts are recorded through the connector's
    logging ledger writer -- the same seam ``--snapshot-fixture-dir``'s wiring
    already uses -- rather than the hash-chained store, which
    :func:`build_paper_deps` opens and owns for the duration of the run.

    Args:
        args: The parsed ``run`` arguments carrying ``--paper-live-ticker``.
        config: The loaded configuration supplying the exchange environment and
            the outbound allowlist.

    Returns:
        The live market-data view, or ``None`` when the flag was omitted.

    Raises:
        ValueError: If the configured environment is unrecognized, or the
            venue host it resolves to is not on this deployment's allowlist --
            refusing to start rather than dialing a venue nobody declared.
    """
    if args.paper_live_ticker is None:
        return None
    from windbreak.connector.live import build_kalshi_market_data
    from windbreak.connector.snapshot import LoggingEventLedgerWriter
    from windbreak.net.allowlist import allowlist_from_config

    return build_kalshi_market_data(
        environment=config.exchange.environment,
        allowlist=allowlist_from_config(config),
        ledger_writer=LoggingEventLedgerWriter(),
    )


#: Each live provider's endpoint, the environment-variable *name* its key is
#: read from on the config, and the header shape that key is sent in. A closed
#: table, deliberately mirroring ``windbreak.net.allowlist``'s own closed
#: provider->host map: a provider absent from here has no live route at all,
#: rather than one assembled from guesses.
_LIVE_LLM_PROVIDERS: dict[str, tuple[str, str, Callable[[str], dict[str, str]]]] = {
    "anthropic": (
        "https://api.anthropic.com/v1/messages",
        "anthropic_api_key_env",
        lambda key: {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    ),
    "openai": (
        "https://api.openai.com/v1/chat/completions",
        "openai_api_key_env",
        lambda key: {
            "authorization": f"Bearer {key}",
            "content-type": "application/json",
        },
    ),
}


def _read_provider_key(env_var: str) -> str:
    """Read one provider API key from the environment, or refuse to start.

    Args:
        env_var: The environment variable the key is read from -- a *name* taken
            from configuration, never a value stored there.

    Returns:
        The key value, which is never logged, echoed, or placed on an exception.

    Raises:
        ValueError: If the variable is unset. The message names the variable
            only; a key that was never exported cannot be leaked, and a
            neighbouring provider's exported key must not be either.
    """
    import os

    key = os.environ.get(env_var)
    if not key:
        msg = (
            f"live forecast providers are selected but environment variable "
            f"{env_var} is unset; export it or select the 'cassette' transport"
        )
        raise ValueError(msg)
    return key


def _live_http_for(
    endpoint_url: str,
    headers: dict[str, str],
    config: WindbreakConfig,
    *,
    timeout_seconds: int,
) -> LiveHttpTransport:
    """Build one credentialed, single-host live transport for ``endpoint_url``.

    The endpoint is first screened against the *deployment's* own outbound
    allowlist, so a host nobody declared refuses to start rather than being
    dialed -- the same order ``_resolve_paper_market_data`` uses, and before any
    session exists (SPEC S15). The transport is then given a single-host
    allowlist of its own, so even holding this object an attacker-supplied
    endpoint cannot steer this provider's credential to another provider's host.

    Args:
        endpoint_url: The provider endpoint this transport may reach.
        headers: The send-time headers, including the API key.
        config: The configuration supplying the deployment allowlist.
        timeout_seconds: The whole-second dial timeout.

    Returns:
        The configured :class:`~windbreak.net.live_http.LiveHttpTransport`.

    Raises:
        ValueError: If ``endpoint_url``'s host is not on this deployment's
            outbound allowlist.
    """
    from urllib.parse import urlsplit

    import requests

    from windbreak.net.allowlist import (
        EgressDeniedError,
        OutboundAllowlist,
        allowlist_from_config,
    )
    from windbreak.net.live_http import LiveHttpTransport

    try:
        allowlist_from_config(config).require(endpoint_url)
    except EgressDeniedError as exc:
        host = urlsplit(endpoint_url).hostname or endpoint_url
        msg = (
            f"live provider host {host!r} is not on this deployment's outbound "
            f"allowlist; declare the provider in forecast.vote_ensemble or "
            f"select the 'cassette' transport"
        )
        raise ValueError(msg) from exc
    host = urlsplit(endpoint_url).hostname or ""
    return LiveHttpTransport(
        session=requests.Session(),
        allowlist=OutboundAllowlist(frozenset({host})),
        headers=headers,
        timeout_seconds=timeout_seconds,
    )


def _resolve_provider_http(config: WindbreakConfig) -> LiveProviderHttp | None:
    """Build the live forecast-provider HTTP seams, or ``None`` for the cassette.

    This is the one place on the PAPER path that reads the process environment.
    Configuration may only ever name the variable a provider key is read from
    (``*_api_key_env``), because every config leaf is flattened verbatim into
    the hash-chained ``ConfigLoaded`` ledger event and can never be redacted
    afterwards. So the name lives in config and the value is resolved here, at
    the composition root, and travels no further than the transport that sends
    it.

    Each provider gets its own transport, its own credential, and its own
    single-host allowlist -- never one shared transport, which would send the
    first provider's key to the second provider's endpoint.

    Transports are built for exactly the providers ``forecast.vote_ensemble``
    names, not for every provider this module knows how to route. That keeps the
    demand for credentials honest -- an Anthropic-only ensemble does not require
    an OpenAI key to exist -- and it is what makes an unroutable ensemble member
    structurally impossible to build a bundle for: the provider that has no
    routing entry is refused here, by name, rather than discovered mid-tick.

    Args:
        config: The loaded configuration naming the transport mode, the vote
            ensemble, the key environment variables, and the dial timeout.

    Returns:
        The live seams when ``forecast.provider_transport.mode`` is ``live``,
        else ``None``.

    Raises:
        ValueError: If the mode is unrecognized, the ensemble names a provider
            with no live transport, a named key variable is unset, or a provider
            host is not on this deployment's outbound allowlist -- each way
            refusing to start rather than dialing unauthorized, unauthenticated,
            or unroutable.
    """
    from windbreak.scheduler.provider_wiring import (
        LiveProviderHttp,
        is_live_mode,
        live_vote_providers,
        routable_live_providers,
    )

    if not is_live_mode(config):
        return None
    settings = config.forecast.provider_transport
    timeout_seconds = settings.request_timeout_seconds
    llm = {}
    for provider in live_vote_providers(config):
        routing = _LIVE_LLM_PROVIDERS.get(provider)
        if routing is None:
            routable = sorted(routable_live_providers())
            msg = (
                f"forecast.vote_ensemble names provider {provider!r}, which has "
                f"no live transport; live-routable providers are {routable}. "
                f"Remove it from the ensemble or select the 'cassette' transport"
            )
            raise ValueError(msg)
        endpoint_url, key_env_field, build_headers = routing
        llm[provider] = _live_http_for(
            endpoint_url,
            build_headers(_read_provider_key(getattr(settings, key_env_field))),
            config,
            timeout_seconds=timeout_seconds,
        )
    search, fetch = _live_research_http(config, timeout_seconds)
    return LiveProviderHttp(llm=llm, search=search, fetch=fetch)


def _live_research_http(
    config: WindbreakConfig, timeout_seconds: int
) -> tuple[LiveHttpTransport | None, LiveHttpTransport | None]:
    """Build the live search and fetch transports, or ``(None, None)`` (#344).

    Live *providers* and live *research* are configured independently, so
    selecting the live LLM transport must not force a deployment to invent a
    search endpoint it has not pinned. An unconfigured ``search_endpoint_url``
    (still the repo's operator placeholder) therefore resolves to no live
    research at all, and the loop falls back to the offline transport that finds
    nothing -- whereupon the pipeline abstains on zero verified citations before
    any vote. That is the honest reading of "no research configured", and it
    fails closed.

    The search endpoint carries the configured search API key; page fetches
    carry no credential at all, because a research fetch is an anonymous read of
    a third-party page -- inventing a key leaf for it would put a secret in the
    hash-chained ledger for nothing. The fetch transport's allowlist is the
    configured research host set, which itself defaults to empty.

    Args:
        config: The configuration supplying endpoint, hosts, and key variable.
        timeout_seconds: The whole-second dial timeout.

    Returns:
        The search transport paired with the fetch transport, or ``(None,
        None)`` when live research is unconfigured.

    Raises:
        ValueError: If a configured search endpoint's host is not on this
            deployment's outbound allowlist, or its key variable is unset.
    """
    import requests

    from windbreak.config.schema import UNCONFIGURED_PLACEHOLDER
    from windbreak.net.allowlist import OutboundAllowlist
    from windbreak.net.live_http import LiveHttpTransport

    research = config.forecast.research
    if research.search_endpoint_url == UNCONFIGURED_PLACEHOLDER:
        return None, None
    search_key = _read_provider_key(research.search_api_key_env)
    search = _live_http_for(
        research.search_endpoint_url,
        {
            "authorization": f"Bearer {search_key}",
            "content-type": "application/json",
        },
        config,
        timeout_seconds=timeout_seconds,
    )
    fetch = LiveHttpTransport(
        session=requests.Session(),
        allowlist=OutboundAllowlist(
            frozenset(host.lower() for host in research.allowed_research_hosts)
        ),
        headers={},
        timeout_seconds=timeout_seconds,
    )
    return search, fetch


def _build_paper_on_beat(
    args: argparse.Namespace, config: WindbreakConfig
) -> Callable[[int], None]:
    """Build a per-beat hook that runs one always-on PAPER tick (issue #48).

    The scheduler imports are local so the RESEARCH heartbeat path never imports
    ``windbreak.scheduler`` (nor, transitively, the paper order-submission client)
    unless PAPER is actually activated. The dependency bundle -- which opens the
    ledger database -- is built once here, so no ledger is ever created on a run
    that does not activate PAPER.

    ``market_data`` and ``live_ticker`` are passed as the pair they are: both
    derive from ``--paper-live-ticker``, so this call site cannot supply half a
    live configuration (issue #343).

    Args:
        args: The parsed ``run`` arguments carrying the four PAPER flags.
        config: The loaded PAPER-ceilinged configuration.

    Returns:
        A callable that, given the beat sequence, runs one PAPER tick.
    """
    from windbreak.scheduler.loop import build_paper_deps, run_single_tick

    deps = build_paper_deps(
        books_dir=args.paper_books_dir,
        cassette_path=args.cassette_path,
        ledger_path=args.ledger_path,
        report_dir=args.report_dir,
        config=config,
        market_data=_resolve_paper_market_data(args, config),
        live_ticker=args.paper_live_ticker,
        provider_http=_resolve_provider_http(config),
    )

    def _on_beat(seq: int) -> None:
        """Run one PAPER tick for the given beat sequence."""
        run_single_tick(deps, beat=seq)

    return _on_beat


def _resolve_on_beat(
    args: argparse.Namespace, config: WindbreakConfig
) -> Callable[[int], None] | None:
    """Resolve the per-beat hook: the PAPER tick, a snapshot pass, or none.

    PAPER activation (issue #48) takes precedence when permitted and fully
    flagged; otherwise the pre-existing snapshot hook is wired when a fixture
    directory is given; otherwise there is no hook and the loop is a bare
    RESEARCH heartbeat.

    Args:
        args: The parsed ``run`` arguments.
        config: The loaded configuration.

    Returns:
        The resolved per-beat hook, or ``None`` for a bare heartbeat.
    """
    if _paper_activated(config, args):
        return _build_paper_on_beat(args, config)
    if args.snapshot_fixture_dir is not None:
        return _build_snapshot_on_beat(args.snapshot_fixture_dir, config.screener)
    return None


def _ledger_config_loads(
    ledger_path: Path, events: list[ConfigLoadEvent], component: str
) -> None:
    """Append each captured config-load event to the hash-chained ledger.

    Opens the ledger at ``ledger_path`` only after the config has already
    loaded cleanly through the in-memory recorder, so a fatal ``--config``
    error stays fail-closed and never creates a database file. The store is
    closed before returning, so a later PAPER loop can reopen the same file.

    Args:
        ledger_path: Filesystem path to the hash-chained ledger database.
        events: The config-load events captured during this run, in order.
        component: The process label stamped on each ``ConfigLoaded`` event.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        recorder = LedgerConfigEventRecorder(store, component=component)
        for event in events:
            recorder.record_config_loaded(
                config_hash=event.config_hash, diff=event.diff, source=event.source
            )
    finally:
        store.close()


def _load_dashboard_token(environ: Mapping[str, str] | None = None) -> str:
    """Read the dashboard's bearer token from the environment, fail-closed.

    Reads :data:`DASHBOARD_AUTH_ENV_VAR` from ``environ`` (defaulting to
    :data:`os.environ`), mirroring
    :meth:`windbreak.riskkernel.signing.SigningKeyHandle.from_env`: a missing
    *or* blank value is rejected the same way -- a blank token can never be
    presented, so it is a misconfiguration, not a usable credential. The token
    is sourced only from the environment, never from the ledgered config.

    Args:
        environ: The environment mapping to read from. Defaults to
            :data:`os.environ`.

    Returns:
        The non-empty bearer token.

    Raises:
        ValueError: If the variable is absent or an empty string.
    """
    source = os.environ if environ is None else environ
    token = source.get(DASHBOARD_AUTH_ENV_VAR)
    if not token:
        raise ValueError(
            f"missing or blank environment variable {DASHBOARD_AUTH_ENV_VAR}"
        )
    return token


def _build_dashboard_status_source(
    ledger_path: Path | None,
) -> Callable[[], DashboardStatus]:
    """Build the dashboard's zero-arg status source (issue #79).

    With a ``ledger_path`` the returned source opens the ledger fresh on every
    call, verifies its hash chain (fail-closed, mirroring
    :func:`windbreak.dashboard.views.build_ledger_read_models_source`), folds the
    ``ModeHeartbeat`` rows via
    :func:`windbreak.ledger.rebuild.mode_history_read_model`, and reports the
    latest row's mode and timestamp -- or the default RESEARCH / no-heartbeat
    status when the history is empty. With ``None`` it always yields that
    default. The dashboard/ledger imports are local so the RESEARCH heartbeat
    path never imports them.

    Args:
        ledger_path: Path to the SQLite ledger database, or ``None`` to serve
            the static RESEARCH / no-heartbeat default.

    Returns:
        A zero-arg callable suitable for
        :func:`windbreak.dashboard.app.create_server`'s ``status_source``.
    """
    from windbreak.dashboard.app import DashboardStatus
    from windbreak.ledger.rebuild import mode_history_read_model
    from windbreak.ledger.store import SqliteLedgerStore

    def _default_source() -> DashboardStatus:
        """Report the static RESEARCH / no-heartbeat status."""
        return DashboardStatus(mode=MODE_RESEARCH, last_heartbeat=None)

    if ledger_path is None:
        return _default_source

    def _ledger_source() -> DashboardStatus:
        """Fold the verified ledger's mode history into the latest status."""
        store = SqliteLedgerStore(ledger_path)
        try:
            store.verify_chain()
            records = store.read_all()
        finally:
            store.close()
        rows = mode_history_read_model(records)
        if not rows:
            return _default_source()
        last_row = rows[-1]
        return DashboardStatus(
            mode=cast("str", last_row["mode"]),
            last_heartbeat=cast("str", last_row["created_at"]),
        )

    return _ledger_source


def _build_dashboard_server(
    args: argparse.Namespace,
    config: WindbreakConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> http.server.ThreadingHTTPServer:
    """Build the loopback dashboard server from parsed args and config (#79).

    Mints the bearer token from the environment, wires the ledger-backed status
    and read-model sources when ``--ledger-path`` is given (else the static
    defaults), and binds the configured ``dashboard.port`` on the hardcoded
    loopback host. The dashboard imports are local so the RESEARCH heartbeat
    path never imports them.

    Args:
        args: Parsed ``run`` arguments carrying ``ledger_path``.
        config: The loaded configuration whose ``dashboard.port`` is bound.
        environ: The environment mapping the token is read from. Defaults to
            :data:`os.environ`.

    Returns:
        A loopback-bound :class:`http.server.ThreadingHTTPServer` ready to serve.

    Raises:
        ValueError: If the token environment variable is absent or blank.
    """
    from windbreak.dashboard.app import create_server
    from windbreak.dashboard.views import build_ledger_read_models_source

    token = _load_dashboard_token(environ)
    read_models_source = (
        build_ledger_read_models_source(args.ledger_path)
        if args.ledger_path is not None
        else None
    )
    return create_server(
        token=token,
        status_source=_build_dashboard_status_source(args.ledger_path),
        port=config.dashboard.port,
        read_models_source=read_models_source,
    )


def _serve_until_shutdown(
    server: http.server.ThreadingHTTPServer, state: ShutdownState
) -> None:
    """Serve the dashboard on a daemon thread until a shutdown is requested.

    ``serve_forever`` runs on a daemon thread while this call waits on
    ``state.stop_event``; a signal handler (never the serving thread) sets that
    event, at which point the server is shut down, its listening socket closed,
    and the serving thread joined with a bounded timeout so a wedged thread can
    never hang the process. The shutdown reason is logged exactly like
    :func:`run_loop`'s shutdown line, stamped with the dashboard component.

    Args:
        server: The dashboard server to serve and then shut down.
        state: Shared shutdown state whose ``stop_event`` gates the serve and
            whose ``reason`` (a signal name, if any) is logged on shutdown.
    """
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.stop_event.wait()
    reason = state.reason if state.reason is not None else _REASON_SIGNAL
    server.shutdown()
    server.server_close()
    thread.join(timeout=_DASHBOARD_THREAD_JOIN_TIMEOUT_SECONDS)
    _LOGGER.info(
        "shutdown reason=%s", reason, extra={"component": _DASHBOARD_COMPONENT}
    )


def _load_and_ledger_config(args: argparse.Namespace) -> WindbreakConfig | None:
    """Load the requested config, log it, and ledger its load events.

    The config-load front half shared by :func:`_run_heartbeat` and
    :func:`_run_dashboard`. Structured logging is already installed by
    :func:`main`, so the diagnostics here are JSON records like every other log
    line. The config is first loaded through an in-memory recorder so a bad
    ``--config`` fails closed *before* any ledger file is opened; only after the
    successful load does a supplied ``--ledger-path`` get the captured
    ``ConfigLoaded`` event(s) persisted to the real hash-chained ledger (issue
    #74) -- as the first records, before any PAPER-loop events.

    Args:
        args: Parsed ``run`` arguments carrying ``config``, ``process``, and
            ``ledger_path``.

    Returns:
        The loaded configuration, or ``None`` (after logging a ``FATAL``
        critical) on a fatal ``--config`` error.
    """
    recorder = InMemoryConfigEventRecorder()
    try:
        config = _load_configured(args, recorder)
    except ConfigError as exc:
        _LOGGER.critical("FATAL: %s", exc)
        return None
    source = str(args.config) if args.config is not None else _DEFAULTS_SOURCE_LABEL
    _LOGGER.info(
        "config loaded source=%s mode_ceiling=%s hash=%s",
        source,
        config.mode_ceiling,
        config_hash(config),
    )
    if args.ledger_path is not None:
        _ledger_config_loads(args.ledger_path, recorder.events, args.process)
    return config


def _run_heartbeat(args: argparse.Namespace) -> int:
    """Load the requested config, log it, then drive the heartbeat loop.

    The ``--config`` loader (issue #11) and the JSON logging pipeline (issue
    #14) compose here via :func:`_load_and_ledger_config`: the config is loaded,
    logged, and (with ``--ledger-path``) ledgered, then the heartbeat loop runs
    stamped with the ``--process`` component. When ``--snapshot-fixture-dir`` is
    given, a per-beat snapshot hook is wired in alongside.

    Args:
        args: Parsed ``run`` arguments carrying ``config``, ``process``,
            ``heartbeat_interval``, ``max_beats``, ``ledger_path``, and
            ``snapshot_fixture_dir``.

    Returns:
        The process exit code (0 on success, 1 on a fatal config error).
    """
    config = _load_and_ledger_config(args)
    if config is None:
        return 1
    state = ShutdownState()
    _install_signal_handlers(state)
    on_beat = _resolve_on_beat(args, config)
    run_loop(
        args.heartbeat_interval,
        max_beats=args.max_beats,
        stop_event=state.stop_event,
        state=state,
        component=args.process,
        on_beat=on_beat,
    )
    return 0


def _build_risk_kernel(
    config: WindbreakConfig,
    *,
    ledger_store: LedgerStore | None = None,
    verification_connector: MarketConnector | None = None,
) -> tuple[RiskKernel, KillIntegration]:
    """Compose a live :class:`RiskKernel` wired to its :class:`KillIntegration`.

    The kill switch, its file watcher, its reconciliation-mismatch monitor, and
    the kernel are all built over **one shared**
    :class:`~windbreak.riskkernel.modes.ModeStateMachine`: the switch drives that
    machine to ``KILLED`` and the kernel reads its mode from the same object, so
    a kill from any trigger is immediately visible to the kernel's evaluation.
    Two independent machines would let the kernel keep approving after a kill --
    the exact defect this wiring closes -- so the single-machine invariant is
    load-bearing, not incidental.

    With a ``ledger_store`` the kernel persists its events to the real
    hash-chained ledger and replays durable kill state from it at startup (issue
    #235): the store's chain is verified fail-closed up front, its records are
    reconstructed into an event history, and both the switch and the kernel are
    rebuilt over that history through a single composition path (an empty history
    -- the no-``ledger_store`` case -- is equivalent to a fresh build). The
    replay *order* is load-bearing:
    :meth:`~windbreak.riskkernel.kill.KillSwitch.from_events` restores only the
    kill counter and never transitions, then
    :meth:`~windbreak.riskkernel.process.RiskKernel.from_events` drives the one
    shared machine to ``KILLED`` on an unrearmed history -- so the two never race
    to transition the single machine.

    That same ``ledger_store`` is also handed to the kernel as its
    ``gate_plan_store`` (issue #246), so a PAPER -> LIVE_MICRO promotion reads
    its three plan-sourced thresholds from the pre-registered, content-addressed
    gate plan already on that ledger. Without this the CLI-composed kernel would
    fail closed with ``GatePlanUnavailableError`` on *every* PAPER promotion even
    with a perfectly good registration on disk. It stays fail-closed where it
    should: with no ``ledger_store`` there is no plan source, and the promotion
    is refused rather than falling back to the live config.

    With a ``verification_connector`` the kernel's per-beat verification cycle
    becomes live (issue #288, superseding issue #236's frozen-startup snapshot):
    a :class:`LedgerExpectationSource` folds the replayed ``history`` once into a
    scoped baseline -- cash seeded from the last non-breach verification event
    *the kernel itself recorded*, positions from the last positions snapshot *it
    recorded* (a shared ledger also carries other components' rows, so only
    ``component="riskkernel"`` rows may seed this safety-critical baseline; none
    records such a snapshot today, so positions always falls through) -- falling
    back per dimension to that connector's own startup
    balances/positions/open-orders where the ledger carries no such fact, a
    :class:`VerificationTolerances` is composed from
    ``risk.verification_balance_tolerance_micros`` /
    ``risk.verification_position_tolerance_centis``, and a
    :class:`ReadOnlyVerifier` -- sharing **both** the kernel's ledger writer and
    the kill switch's one :class:`AlertDispatcher` -- is forwarded to
    :meth:`RiskKernel.from_events`. The shared writer means each cycle's
    verification *event* lands on the same hash-chained ledger the kernel
    persists to; the shared dispatcher means mismatch and jurisdiction *alerts*
    fan out through the same sink chain the kill switch uses -- since issue #274
    the sinks ``config.alerts`` declares, falling back to log-only only when the
    operator has configured none. This is what makes the
    ``AUTO_RECONCILIATION`` auto-kill trigger *live* rather than
    composed-but-dormant: a sustained reconciliation ``BREACH`` now feeds the
    shared monitor and engages the shared kill switch. With no connector the
    verifier is ``None`` and the composition is byte-identical to its
    pre-issue-#236 shape.

    The kernel imports are local (mirroring :func:`_run_drill` /
    :func:`_build_paper_on_beat`) so the RESEARCH heartbeat path never imports
    the kernel eagerly. ``ops.state_dir`` is created up front, fail-closed: a
    build that cannot create its state directory raises ``OSError`` before any
    kernel loop is entered, so a mis-provisioned deployment never runs blind.

    Args:
        config: The loaded configuration. ``ops.state_dir`` roots the kill/re-arm
            file protocol, ``mode_ceiling`` caps the shared machine,
            ``risk.kill_after_consecutive_mismatches`` sets the auto-kill
            threshold, and ``risk.verification_balance_tolerance_micros`` /
            ``risk.verification_position_tolerance_centis`` set the live
            verifier's per-dimension drift tolerances.
        ledger_store: The append-only ledger the kernel persists to, replays
            durable kill state from (issue #235), and reads its registered PAPER
            gate plan from (issue #246), or ``None`` to run against a log-only
            writer with no replay (an empty history) and no plan source.
        verification_connector: The read-only market connector the live verifier
            observes the venue through (issue #236), or ``None`` to compose no
            verifier -- leaving the per-beat verification cycle a no-op exactly
            as before.

    Returns:
        The composed kernel and the kill integration it shares, so a caller can
        drive both (e.g. observe the monitor) against the one machine.

    Raises:
        OSError: If ``ops.state_dir`` cannot be created (fail-closed startup).
        ChainIntegrityError: If a supplied ``ledger_store``'s hash chain fails
            verification (fail-closed replay).
        AlertSinkConfigError: If ``config.alerts`` declares a sink that cannot
            work -- the kernel refuses to start rather than run the money path
            believing it can page an operator when it cannot.
    """
    from windbreak.riskkernel.kill import (
        KillFileWatcher,
        KillIntegration,
        KillSwitch,
        ReconciliationMismatchMonitor,
    )
    from windbreak.riskkernel.modes import Mode, ModeStateMachine
    from windbreak.riskkernel.process import (
        LoggingKernelLedgerWriter,
        PersistingKernelLedgerWriter,
        RiskKernel,
    )

    state_dir = Path(config.ops.state_dir).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    machine = ModeStateMachine(mode_ceiling=Mode.from_config(config.mode_ceiling))
    if ledger_store is not None:
        ledger_store.verify_chain()
        history: tuple[Event, ...] = events_from_records(ledger_store.read_all())
        writer: KernelLedgerWriter = PersistingKernelLedgerWriter(ledger_store)
    else:
        writer = LoggingKernelLedgerWriter()
        history = ()
    # One dispatcher shared by the kill switch's HALT_KILL alerts and (when a
    # verification connector is wired) the verifier's mismatch/jurisdiction
    # alerts, so both fan out through the same configured sinks (issue #274).
    dispatcher = _build_alert_dispatcher(config)
    switch = KillSwitch.from_events(
        history, machine, writer, dispatcher, state_dir=state_dir
    )
    watcher = KillFileWatcher(switch, state_dir)
    monitor = ReconciliationMismatchMonitor(
        switch, threshold=config.risk.kill_after_consecutive_mismatches
    )
    integration = KillIntegration(switch=switch, watcher=watcher, monitor=monitor)
    verifier = _build_verifier(
        config, verification_connector, dispatcher, writer, history
    )
    kernel = RiskKernel.from_events(
        history,
        writer,
        mode_machine=machine,
        verifier=verifier,
        kill_integration=integration,
        gate_plan_store=ledger_store,
    )
    return kernel, integration


def _build_verifier(
    config: WindbreakConfig,
    verification_connector: MarketConnector | None,
    dispatcher: AlertDispatcher,
    writer: KernelLedgerWriter,
    history: tuple[Event, ...],
) -> ReadOnlyVerifier | None:
    """Compose the live read-only verifier, or ``None`` when unconfigured.

    Projects a scoped ledger-derived baseline the verifier reconciles the venue
    against each beat (issue #288, superseding the frozen-startup-snapshot seam
    of issue #236): a :class:`LedgerExpectationSource` folds ``history`` once,
    seeding cash from the last non-breach verification event *the kernel itself
    recorded* and positions from the last positions snapshot *it recorded*, and
    falling back -- per dimension, independently -- to
    ``verification_connector``'s own startup state where the ledger carries no
    such fact. Only ``component="riskkernel"`` rows may seed, because a shared
    ledger also carries e.g. the PAPER loop's simulated position snapshots; no
    component records a ``riskkernel``-stamped snapshot today, so positions is
    connector-fallback in practice (see :class:`LedgerExpectationSource`). The
    per-dimension tolerances come from ``config.risk``. The verifier shares
    ``dispatcher`` and ``writer`` with the kill switch so its alerts and events
    reach the same sinks and hash-chained ledger.

    Args:
        config: The loaded configuration supplying the balance/position drift
            tolerances.
        verification_connector: The read-only market connector to observe (and
            the per-dimension fallback source), or ``None`` to compose no
            verifier.
        dispatcher: The alert dispatcher shared with the kill switch.
        writer: The kernel ledger writer shared with the kill switch.
        history: The replayed startup event history the baseline is seeded from
            (empty when there is no ledger store to replay).

    Returns:
        The composed :class:`ReadOnlyVerifier`, or ``None`` when
        ``verification_connector`` is ``None``.
    """
    if verification_connector is None:
        return None
    from windbreak.numeric.types import ContractCentis, MoneyMicros
    from windbreak.riskkernel.verification import (
        LedgerExpectationSource,
        ReadOnlyVerifier,
        VerificationTolerances,
    )

    return ReadOnlyVerifier(
        connector=verification_connector,
        expectation_source=LedgerExpectationSource(history, verification_connector),
        tolerances=VerificationTolerances(
            balance_tolerance=MoneyMicros(
                config.risk.verification_balance_tolerance_micros
            ),
            position_tolerance=ContractCentis(
                config.risk.verification_position_tolerance_centis
            ),
        ),
        dispatcher=dispatcher,
        ledger_writer=writer,
    )


def _kernel_heartbeat_interval(args: argparse.Namespace) -> int:
    """Map the CLI's float ``--heartbeat-interval`` onto the kernel's int seconds.

    The shared ``run`` flag parses as a float, but the ``riskkernel`` package is
    float-free (SPEC S6.1) and :meth:`RiskKernel.run` takes whole seconds, so the
    fractional value is rounded up here -- the one float/int seam -- keeping every
    ``windbreak/riskkernel/`` call float-free.

    Args:
        args: Parsed ``run`` arguments carrying ``heartbeat_interval``.

    Returns:
        The heartbeat interval in whole seconds (ceiling of the float value).
    """
    interval_seconds: float = args.heartbeat_interval
    return math.ceil(interval_seconds)


def _run_riskkernel(args: argparse.Namespace) -> int:
    """Compose and drive the live Risk Kernel for ``--process riskkernel`` (#144).

    Reuses :func:`_load_and_ledger_config`'s config-load front half, then builds
    the kernel and its kill integration via :func:`_build_risk_kernel` -- catching
    an uncreatable state dir (``OSError``), a bad mode ceiling (``ValueError``),
    or a tampered ledger (``ChainIntegrityError``) as a fatal error that logs
    ``FATAL`` and returns 1 *before* the loop is entered, so a fail-closed startup
    emits no heartbeat. This is the routing divergence (issue #144) from the
    RESEARCH heartbeat loop the other ``--process`` choices run: it drives a real
    :class:`RiskKernel` whose file watcher polls ``ops.state_dir`` for a ``KILL``
    file each beat.

    With ``--ledger-path`` the kernel is built over a real
    :class:`~windbreak.ledger.store.SqliteLedgerStore` so its events persist and a
    durable ``KILLED`` state is replayed at startup (issue #235). The store is
    closed in a ``finally`` -- even when the build fails closed -- so a later
    reopen (the next run's replay) is never blocked by a lingering handle.

    Args:
        args: Parsed ``run`` arguments carrying ``config``, ``process`` (always
            ``riskkernel`` when routed here), ``heartbeat_interval``,
            ``max_beats``, and ``ledger_path``.

    Returns:
        The process exit code (0 on a clean shutdown, 1 on a fatal config or
        kernel-build error).
    """
    config = _load_and_ledger_config(args)
    if config is None:
        return 1
    store = (
        SqliteLedgerStore(args.ledger_path) if args.ledger_path is not None else None
    )
    try:
        return _drive_risk_kernel(args, config, ledger_store=store)
    finally:
        if store is not None:
            store.close()


def _snapshot_connector(args: argparse.Namespace) -> MarketConnector | None:
    """Build the read-only verification connector from ``--snapshot-fixture-dir``.

    Mirrors :func:`_run_preflight`'s local-import ``FakeExchange`` construction.
    Called inside :func:`_drive_risk_kernel`'s fail-closed ``try`` so a missing
    or malformed fixture directory raises there (issue #236).

    Args:
        args: Parsed ``run`` arguments carrying ``snapshot_fixture_dir``.

    Returns:
        A :class:`~windbreak.connector.fake.FakeExchange` over the fixture
        directory, or ``None`` when ``--snapshot-fixture-dir`` was not given.
    """
    if args.snapshot_fixture_dir is None:
        return None
    from windbreak.connector import FakeExchange

    return FakeExchange.from_fixture_dir(args.snapshot_fixture_dir)


def _drive_risk_kernel(
    args: argparse.Namespace,
    config: WindbreakConfig,
    *,
    ledger_store: LedgerStore | None,
) -> int:
    """Build the kernel over ``ledger_store`` and drive its bounded loop.

    The inner half of :func:`_run_riskkernel`, split out so the store's
    open/close lifecycle stays a single ``try``/``finally`` there while this
    function owns the fail-closed build and the run. A build that raises
    (uncreatable state dir, bad ceiling, a tampered ledger chain, or an alert
    sink that cannot work) logs ``FATAL`` and returns 1 before the loop is
    entered, emitting no heartbeat.

    ``AlertSinkConfigError`` is named alongside the rest because
    :func:`_build_risk_kernel` composes ``config.alerts`` into the kill
    switch's dispatcher (issue #274): a misconfigured sink must fail this
    process closed with the same ``FATAL`` + exit-1 contract ``preflight`` and
    ``alert-test`` honor, not escape as an unhandled traceback out of the one
    process that runs the kill switch.

    When ``--snapshot-fixture-dir`` is given, a read-only
    :class:`~windbreak.connector.fake.FakeExchange` is built over it *inside*
    this same fail-closed ``try`` and wired as the kernel's verification
    connector (issue #236): a missing or malformed fixture directory raises
    ``FileNotFoundError``/``JSONDecodeError`` -- subclasses of ``OSError`` /
    ``ValueError`` already caught here -- so a bad snapshot dir fails closed
    identically to an uncreatable state dir, never entering the kernel loop.

    Args:
        args: Parsed ``run`` arguments carrying ``process``, ``max_beats``,
            ``heartbeat_interval``, and ``snapshot_fixture_dir``.
        config: The loaded configuration the kernel is composed from.
        ledger_store: The ledger the kernel persists to and replays from, or
            ``None`` for the log-only, replay-free path.

    Returns:
        The process exit code (0 on a clean shutdown, 1 on a fatal build error).
    """
    try:
        verification_connector = _snapshot_connector(args)
        kernel, _integration = _build_risk_kernel(
            config,
            ledger_store=ledger_store,
            verification_connector=verification_connector,
        )
    except (OSError, ValueError, ChainIntegrityError, AlertSinkConfigError) as exc:
        _LOGGER.critical("FATAL: %s", exc)
        return 1
    state = ShutdownState()
    _install_signal_handlers(state)
    kernel.run(
        max_beats=args.max_beats,
        heartbeat_interval=_kernel_heartbeat_interval(args),
        stop_event=state.stop_event,
    )
    _LOGGER.info(
        "shutdown reason=%s",
        _shutdown_reason(state, args),
        extra={"component": args.process},
    )
    return 0


def _shutdown_reason(state: ShutdownState, args: argparse.Namespace) -> str:
    """Resolve the shutdown reason logged after a bounded process loop returns.

    Mirrors :func:`run_loop`'s reason selection so a diverging process (dashboard,
    riskkernel) logs the same parity line: a signal name recorded on ``state``
    wins; otherwise an exhausted ``--max-beats`` budget reports ``max_beats`` and
    a bare stop reports the generic ``signal`` reason.

    Args:
        state: Shared shutdown state whose ``reason`` (a signal name, if any) the
            handler recorded.
        args: Parsed ``run`` arguments carrying ``max_beats``.

    Returns:
        The shutdown reason string to log.
    """
    if state.reason is not None:
        return state.reason
    return _REASON_MAX_BEATS if args.max_beats is not None else _REASON_SIGNAL


def _run_dashboard(args: argparse.Namespace) -> int:
    """Serve the authenticated loopback dashboard until shutdown (issue #79).

    Reuses :func:`_load_and_ledger_config`'s config-load front half, then builds
    the dashboard server -- catching a missing/blank token or bad port as a
    fatal ``ValueError`` -- installs the graceful-shutdown signal handlers, logs
    the bound loopback address, and serves until SIGINT/SIGTERM. This is the
    routing divergence from the RESEARCH heartbeat loop that every other
    ``--process`` choice still runs (issue #15).

    Args:
        args: Parsed ``run`` arguments carrying ``config``, ``ledger_path``, and
            ``process`` (always ``dashboard`` when routed here).

    Returns:
        The process exit code (0 on a clean shutdown, 1 on a fatal config or
        token/port error).
    """
    config = _load_and_ledger_config(args)
    if config is None:
        return 1
    try:
        server = _build_dashboard_server(args, config)
    except ValueError as exc:
        _LOGGER.critical("FATAL: %s", exc)
        return 1
    state = ShutdownState()
    _install_signal_handlers(state)
    _LOGGER.info(
        "dashboard serving on 127.0.0.1:%d",
        server.server_address[1],
        extra={"component": _DASHBOARD_COMPONENT},
    )
    _serve_until_shutdown(server, state)
    return 0


def _run_run_command(args: argparse.Namespace) -> int:
    """Route the ``run`` command to its per-process handler.

    Two ``--process`` choices diverge from the shared RESEARCH heartbeat loop
    (issue #15): ``dashboard`` serves the loopback dashboard server (issue #79),
    and ``riskkernel`` composes and drives a live :class:`RiskKernel` with its
    kill-switch wiring (issue #144). Every other choice runs the heartbeat loop.

    Args:
        args: Parsed ``run`` arguments carrying ``process``.

    Returns:
        The selected handler's process exit code.
    """
    if args.process == "dashboard":
        return _run_dashboard(args)
    if args.process == "riskkernel":
        return _run_riskkernel(args)
    return _run_heartbeat(args)


#: Maps each subcommand token to the handler that runs it. ``run`` fans out to
#: its own router (dashboard vs. heartbeat); every handler has signature
#: ``(args) -> int`` and returns the process exit code.
_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "rebuild": rebuild_command,
    "anchor": anchor_command,
    "verify": verify_command,
    "kill": _run_kill,
    "ack": _run_ack,
    "rearm": _run_rearm,
    "alert-test": _run_alert_test,
    "preflight": _run_preflight,
    "drill": _run_drill,
    "run": _run_run_command,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the requested windbreak command.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code (0 on success, 1 on a fatal config error).
    """
    args = build_parser().parse_args(argv)
    configure_logging(level=logging.INFO)
    return _COMMAND_HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
