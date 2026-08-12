"""Command-line entry point and heartbeat loop for windbreak.

The ``windbreak run`` command starts the always-on RESEARCH-mode heartbeat
loop that later issues will grow into the full four-process pipeline. For now
it emits a periodic heartbeat line and shuts down cleanly on SIGINT, SIGTERM,
or an optional beat budget.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, cast

from windbreak import __version__ as _source_version
from windbreak.alerts import (
    AlertDispatcher,
    AlertType,
    LoggingLedgerWriter,
    cli_token,
    ledger_deliveries,
)
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
from windbreak.evaluation.ingest import (
    MarketResolved,
    SettlementReversed,
    describe_resolution_claim,
    fold_resolutions,
    ingested_resolution_of,
    parse_resolution_instant,
    resolutions_conflict,
)
from windbreak.evaluation.resolution import ResolutionOutcome
from windbreak.evaluation.track_records import (
    PROVIDER_TRACK_RECORD_FILENAME,
    provider_track_records,
    write_provider_track_records,
)
from windbreak.ledger import (
    AlertEmitted as LedgerAlertEmitted,
)
from windbreak.ledger import (
    ChainIntegrityError,
    ModeHeartbeat,
    SqliteLedgerStore,
    anchor_command,
    events_from_records,
    rebuild_command,
    verify_command,
)
from windbreak.logging_setup import configure_logging
from windbreak.riskkernel.ack_flow import ACKS_DIRNAME
from windbreak.riskkernel.kill import KILL_FILENAME, REARM_FILENAME
from windbreak.scheduler.research_spend import (
    RESEARCH_BUDGET_CAP_SET_EVENT_TYPE,
    ResearchBudgetCapSet,
    effective_per_day_micros,
    spend_by_day_from_records,
)

if TYPE_CHECKING:
    import http.server
    from collections.abc import Callable, Mapping, Sequence
    from types import FrameType

    from windbreak.alerts import AlertEmitted, LedgerWriter
    from windbreak.config import ConfigLoadEvent, ScreenerConfig, WindbreakConfig
    from windbreak.connector.interface import MarketConnector
    from windbreak.connector.live import MarketDataSource
    from windbreak.dashboard.app import DashboardStatus
    from windbreak.evaluation.ingest import IngestedResolution, ResolutionFold
    from windbreak.forecast.providers.track_record import ProviderTrackRecord
    from windbreak.ledger import Event, LedgerStore
    from windbreak.net.live_http import LiveHttpTransport
    from windbreak.riskkernel.kill import KillIntegration
    from windbreak.riskkernel.process import KernelLedgerWriter, RiskKernel
    from windbreak.riskkernel.verification import ReadOnlyVerifier
    from windbreak.scheduler.provider_wiring import LiveProviderHttp

#: Operating mode reported in a heartbeat line whose beat had no tick to report
#: a mode of. Matches the RESEARCH state of the SPEC mode machine, which is what
#: a hook-less (or snapshot-only) pass genuinely is. A beat that *does* run a
#: tick reports that tick's own mode instead (issue #447).
MODE_RESEARCH = "RESEARCH"

#: Heartbeat mode token for a beat whose hook raised (issue #443). Surviving a
#: tick failure must never make the loop quieter: the one line an operator
#: watches has to say the tick failed rather than keep claiming the RESEARCH
#: mode of a healthy pass.
MODE_TICK_FAILED = "TICK_FAILED"

#: Component label stamped on an operator-ingested ``MarketResolved`` event
#: (issue #439). Deliberately *not* one of the pipeline process labels: the
#: resolution is a human's claim about the outside world, and stamping it with a
#: component that never produced it would forge provenance in a hash-chained
#: audit trail.
_RESOLUTION_INGEST_COMPONENT = "operator"

#: Component label stamped on an operator-appended ``ResearchBudgetCapSet``
#: event (issues #442/#483). Deliberately the same human label the resolution
#: ingest uses and never a pipeline process name: a spend ceiling is a human's
#: instruction, and stamping it with a component that never issued it would
#: forge provenance in a hash-chained audit trail.
_RESEARCH_BUDGET_COMPONENT = "operator"

#: ``source=`` token logged when ``run --research-per-day-micros`` decided the
#: per-UTC-day research ceiling this process starts on.
_CEILING_SOURCE_FLAG = "--research-per-day-micros"

#: ``source=`` token logged when the configuration file (or its defaults)
#: decided the ceiling.
_CEILING_SOURCE_CONFIG = "configuration"

#: ``source=`` token logged when a ledgered ``ResearchBudgetCapSet`` row decided
#: the ceiling. It supersedes both of the above by
#: :func:`~windbreak.scheduler.research_spend.effective_per_day_micros`'s own
#: precedence rule, so it is the source whenever such a row is present -- which
#: is exactly the case the startup log used to misreport (it printed the flag's
#: value and named the flag while the ledger's ceiling was the one in force).
_CEILING_SOURCE_LEDGER = "set-research-budget"

#: ``source=`` token logged when this ledger's research rows cannot be folded at
#: all. The loop does not refuse to start over it -- see
#: :func:`windbreak.scheduler.loop._research_ledger_state` -- it opens a zero
#: ceiling and halts every market on the budget, so this is what the startup log
#: must report if it is to match what the loop will actually do.
_CEILING_SOURCE_UNREADABLE = "unreadable-ledger"

#: The ceiling reported (and enforced) when this ledger's research rows cannot
#: be folded. Mirrors
#: :data:`windbreak.scheduler.loop._UNREADABLE_LEDGER_CEILING_MICROS`, which is
#: private to that module; a test pins the two equal.
_UNREADABLE_LEDGER_CEILING_MICROS = 0

#: Mode reported where there is no evidence of any mode -- the dashboard's
#: no-ledger and no-heartbeat defaults (issue #447). Absent evidence is not
#: healthy evidence, so it is never rendered as a real mode.
MODE_UNKNOWN = "UNKNOWN"

#: Escalation kind recorded when a beat's hook raised. Alerts are deduplicated
#: per *kind*, each kind's active state tracked independently, so an ongoing
#: outage pages once but a halt arriving behind it still pages on its own --
#: and, crucially, neither condition's arrival re-pages the other.
_ESCALATION_BEAT_FAILED = "beat-failed"

#: Escalation kind recorded when a beat reported a halted Risk Kernel.
_ESCALATION_KERNEL_HALT = "kernel-halt"

#: How many consecutive beats must *observe* an escalated condition to be gone
#: before it re-arms and a recurrence pages again.
#:
#: One quiet beat is not evidence an outage ended. The failure #443 was filed
#: against is a near-full ledger volume, which by its nature refuses some
#: appends and accepts others; a single successful beat between two failing
#: ones is the outage continuing, not clearing. Re-arming on it turns an
#: intermittent fault into one page per recurrence -- at the issue's own
#: ``--heartbeat-interval 0.01`` that is a pager storm. Three is a confirmation
#: window (15 seconds at the default 5s interval): a condition that genuinely
#: cleared costs at most two beats of delay before its recurrence pages.
_ESCALATION_CLEAR_RUN_BEATS = 3

#: Cap on how many times the inter-beat wait is doubled while beats keep
#: escalating (issue #443's "continue on the next beat with backoff").
#:
#: A loop whose every beat fails or halts achieves nothing by retrying at full
#: cadence: it burns the venue's rate limit and floods the log -- possibly onto
#: the very volume that filled. The wait doubles per consecutive escalating
#: beat and resets on the first clean one, so recovery is noticed within one
#: backed-off wait. The cap keeps the heartbeat itself a usable liveness signal:
#: 16x the configured interval, so 80 seconds at the 5s default.
_MAX_ESCALATION_BACKOFF_DOUBLINGS = 4

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


def _non_negative_micros(raw: str) -> int:
    """Parse a non-negative micros amount for use as an argparse ``type``.

    Distinct from :func:`_non_negative_int`, whose message names ``max beats``;
    a money ceiling refused with that wording would send an operator looking at
    the wrong flag.

    Args:
        raw: The raw command-line token.

    Returns:
        The parsed micros value.

    Raises:
        argparse.ArgumentTypeError: If ``raw`` is not an int or is negative. A
            negative ceiling is unenforceable, so it is refused at parse time
            -- before anything is composed and before any ledger is opened.
    """
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("micros amount must be non-negative")
    return value


def _add_run_arguments(run_parser: argparse.ArgumentParser) -> None:
    """Register the ``run`` subcommand's options on its subparser.

    Args:
        run_parser: The ``run`` subparser to populate with options.
    """
    run_parser.add_argument(
        "--research-per-day-micros",
        type=_non_negative_micros,
        default=None,
        help=(
            "Per-UTC-day research spend ceiling, in micros, overriding the "
            "configured value for this run (default: the configuration's). "
            "This is the startup default only: a ledgered "
            "`windbreak set-research-budget` row supersedes it, and is picked "
            "up on the next tick without a restart."
        ),
    )
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


def _add_ingest_resolution_arguments(ingest_parser: argparse.ArgumentParser) -> None:
    """Register the ``ingest-resolution`` subcommand's options on its subparser.

    Every option is required. A resolution with a defaulted market, outcome,
    instant or provenance would be a fabricated fact in a hash-chained audit
    trail, so there is nothing to default to.

    Args:
        ingest_parser: The ``ingest-resolution`` subparser to populate.
    """
    ingest_parser.add_argument(
        "--ledger-path",
        type=Path,
        required=True,
        help="Path to the SQLite ledger database to append the resolution to.",
    )
    ingest_parser.add_argument(
        "--market-ticker",
        required=True,
        help="Ticker of the market that settled.",
    )
    ingest_parser.add_argument(
        "--outcome",
        required=True,
        choices=[member.value for member in ResolutionOutcome],
        help="The settled ground-truth outcome.",
    )
    ingest_parser.add_argument(
        "--resolved-at",
        required=True,
        help="ISO-8601 instant the market settled, WITH a UTC offset "
        "(e.g. 2026-03-01T12:00:00+00:00). An offsetless instant is refused.",
    )
    ingest_parser.add_argument(
        "--source",
        required=True,
        help="Where the settlement was read (the claim's provenance), "
        "recorded verbatim in the ledgered event.",
    )


def _add_correct_resolution_arguments(
    correct_parser: argparse.ArgumentParser,
) -> None:
    """Register the ``correct-resolution`` subcommand's options (issue #484).

    Every option is required, including ``--superseded-sequence-number``: a
    correction that defaulted to "whatever row is current" would be an
    unreviewable act in a hash-chained audit trail, and naming the row makes
    the operator confirm which claim they mean to overturn.

    Args:
        correct_parser: The ``correct-resolution`` subparser to populate.
    """
    correct_parser.add_argument(
        "--ledger-path",
        type=Path,
        required=True,
        help="Path to the SQLite ledger database to append the correction to.",
    )
    correct_parser.add_argument(
        "--market-ticker",
        required=True,
        help="Ticker of the market whose settlement was recorded wrongly.",
    )
    correct_parser.add_argument(
        "--superseded-sequence-number",
        type=int,
        required=True,
        help="Ledger sequence_number of the row this correction supersedes: "
        "the market's first ingest, or its most recent correction. A refusal "
        "names the right one.",
    )
    correct_parser.add_argument(
        "--outcome",
        required=True,
        choices=[member.value for member in ResolutionOutcome],
        help="The CORRECTED ground-truth outcome.",
    )
    correct_parser.add_argument(
        "--resolved-at",
        required=True,
        help="The CORRECTED ISO-8601 settlement instant, WITH a UTC offset "
        "(e.g. 2026-03-01T12:00:00+00:00). An offsetless instant is refused.",
    )
    correct_parser.add_argument(
        "--source",
        required=True,
        help="Where the correction was read (the correction's own provenance), "
        "recorded verbatim in the ledgered event.",
    )


def _add_set_research_budget_arguments(
    budget_parser: argparse.ArgumentParser,
) -> None:
    """Register the ``set-research-budget`` subcommand's options (#442, #483).

    Every option is required. A money ceiling with a defaulted value would be a
    guess written into a hash-chained audit trail, and a change nothing explains
    cannot be reviewed, so there is nothing to default to.

    Args:
        budget_parser: The ``set-research-budget`` subparser to populate.
    """
    budget_parser.add_argument(
        "--ledger-path",
        type=Path,
        required=True,
        help="Path to the SQLite ledger database to append the change to.",
    )
    budget_parser.add_argument(
        "--per-day-micros",
        type=_non_negative_micros,
        required=True,
        help=(
            "The new per-UTC-day research spend ceiling, in micros. 0 stops "
            "all further research spend."
        ),
    )
    budget_parser.add_argument(
        "--note",
        required=True,
        help=(
            "Why the ceiling is being changed, recorded verbatim in the "
            "ledgered event. Never compared against anything, so its wording "
            "can never make two changes disagree."
        ),
    )


def _add_evaluate_providers_arguments(
    evaluate_parser: argparse.ArgumentParser,
) -> None:
    """Register the ``evaluate-providers`` subcommand's options (issue #440).

    Both options are required and neither is created if missing. The verb's
    whole job is to publish the artifact a *particular* loop's gate reads, so a
    defaulted ledger or report directory would let it write a confident,
    exit-0 track record that the running loop never reads.

    Args:
        evaluate_parser: The ``evaluate-providers`` subparser to populate.
    """
    evaluate_parser.add_argument(
        "--ledger-path",
        type=Path,
        required=True,
        help="Path to the existing SQLite ledger database to score.",
    )
    evaluate_parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help=(
            "The existing evaluation-artifact directory the loop reads its "
            f"{PROVIDER_TRACK_RECORD_FILENAME} from (the loop's --report-dir)."
        ),
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


#: The name this CLI is installed under -- `[project] name` in `pyproject.toml`
#: and therefore the key `importlib.metadata` looks the installed build up by.
_DISTRIBUTION_NAME = "windbreak"


def _reported_version() -> str:
    """Report the version of the build this process is actually running.

    ``--version`` exists so that an operator holding a DEPLOYED ARTIFACT can
    ask it which build it is, so the installed distribution's own metadata --
    what `pip show` reports, stamped into the wheel's ``.dist-info`` from
    `pyproject.toml` at build time -- is authoritative whenever there is an
    installed distribution to ask.

    ``windbreak.__version__`` is the fallback, and it is not the same claim: it
    is a string in the source tree, which in an install whose checkout has since
    been edited can disagree with what was actually installed. It is what a
    checkout that was never installed has, though, and that is a real way to run
    this CLI -- including this repository's own test suite -- so falling back
    beats raising `PackageNotFoundError` in front of an operator.

    The two are pinned equal for a release by
    `tests/test_cli_version.py::test_the_package_dunder_version_matches_the_pyproject_declaration`,
    so neither answer can drift from `pyproject.toml` unnoticed. Nothing here
    restates a version literal: both values are read.

    Returns:
        The installed distribution's version, or the source tree's declared
        version when no distribution is installed.
    """
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return _source_version


def build_parser() -> argparse.ArgumentParser:
    """Build the ``windbreak`` command-line argument parser.

    Returns:
        A parser with a required ``run`` subcommand exposing
        ``--heartbeat-interval``, ``--max-beats``, ``--process``,
        ``--snapshot-fixture-dir``, and ``--config``; a ``rebuild`` subcommand
        exposing ``--ledger-path`` and ``--output-dir``; ``anchor`` and
        ``verify`` subcommands exposing ``--ledger-path`` and ``--anchor-path``
        (append the ledger head to, and check the live chain against, the
        anchor file); an ``ingest-resolution`` subcommand exposing
        ``--ledger-path``, ``--market-ticker``, ``--outcome``, ``--resolved-at``
        and ``--source`` (record that a market settled, so forecasts on it can
        be scored); a ``correct-resolution`` subcommand exposing those five
        plus ``--superseded-sequence-number`` (supersede a wrongly recorded
        resolution with the right one, issue #484); an ``evaluate-providers``
        subcommand exposing
        ``--ledger-path`` and ``--report-dir`` (score each provider's resolved
        forecasts into the live-eligibility gate's artifact, issue #440); a
        ``set-research-budget`` subcommand exposing
        ``--ledger-path``, ``--per-day-micros`` and ``--note`` (change the
        per-UTC-day research spend ceiling on a running loop, issues #442/#483);
        ``kill`` and ``rearm`` subcommands exposing
        ``--state-dir``; an ``ack`` subcommand exposing ``--approval-id`` and
        ``--state-dir``; and a developer-only ``alert-test`` subcommand hidden
        from ``--help``. The parser itself takes ``--version``, which prints
        :func:`_reported_version` and exits 0 *without* a subcommand (#507).
    """
    parser = argparse.ArgumentParser(
        prog="windbreak",
        description="windbreak always-on forecast trader CLI.",
    )
    # Registered BEFORE the required ``command`` positional, and as argparse's
    # ``version`` action rather than a flag the dispatch table would have to
    # notice. Both matter: a ``version`` action short-circuits during argv
    # consumption, so it exits 0 before argparse's end-of-parse check for the
    # required positional -- which is precisely what issue #507 hit, where the
    # missing positional errored (exit 2) without the typed flag even being
    # mentioned. The printed string is bare, with no ``%(prog)s`` prefix, so
    # `windbreak --version` is directly comparable to `pyproject.toml`'s
    # declaration and usable as `VERSION="$(windbreak --version)"`.
    parser.add_argument(
        "--version",
        action="version",
        version=_reported_version(),
        help="Print the version of the installed build and exit.",
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
    _add_ingest_resolution_arguments(
        subparsers.add_parser(
            "ingest-resolution",
            help="Record that a market settled, so forecasts on it can be scored.",
        )
    )
    _add_correct_resolution_arguments(
        subparsers.add_parser(
            "correct-resolution",
            help="Supersede a wrongly recorded resolution with the right one.",
        )
    )
    _add_set_research_budget_arguments(
        subparsers.add_parser(
            "set-research-budget",
            help="Change the per-UTC-day research spend ceiling, from now on.",
        )
    )
    _add_evaluate_providers_arguments(
        subparsers.add_parser(
            "evaluate-providers",
            help="Score each provider's resolved forecasts into the gate's artifact.",
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


@dataclass(frozen=True)
class BeatReport:
    """What one beat's hook reports back to the loop that ran it (issue #447).

    The PAPER hook's ``TickOutcome`` is a scheduler type the heartbeat loop must
    not depend on, so the hook projects the two facts the loop actually acts on
    into this narrow value: what mode the tick ran in, and whether it left the
    Risk Kernel halted. A hook with nothing to report (the snapshot pass) returns
    ``None`` instead, which is not the same thing as reporting health.

    Attributes:
        mode: The tick's own operating mode, from ``deps.kernel.mode`` -- never
            a hardcoded constant, which is exactly what #447 found in the
            heartbeat line.
        halted: Whether the Risk Kernel is in ``HALT`` after the tick
            (``TickOutcome.kernel_halted``). A halted kernel may not trade and
            has no transition back to a mode that may, so the loop escalates it
            rather than beating on quietly.
        research_halted: Whether the tick's research stopped on a budget ceiling
            (``TickOutcome.research_halted``). Deliberately *not* folded into
            ``halted``: a spent research budget is an expected, self-clearing
            daily event that leaves the kernel's mode untouched, so it is
            reported at WARNING rather than paged as a kernel halt.
    """

    mode: str
    halted: bool = False
    research_halted: bool = False


class BeatSupervisor:
    """Run one beat's hook so a failure or a halt is survivable *and* louder.

    Closes issues #443 and #447 at the same seam, and the interaction is the
    point. #443 asks that a tick exception -- a full ledger volume, a locked
    SQLite file, a transient venue error -- stop killing the daemon. Catching it
    and continuing would on its own have *deepened* #447: a loop that swallows
    every tick failure while still logging ``mode=RESEARCH heartbeat seq=N`` is
    an unobservable failure, which is worse than a crash an operator can see.

    So every condition this survives is made noisier than the crash it replaces:

    * the heartbeat's mode slot carries :data:`MODE_TICK_FAILED` (or the tick's
      real halted mode) instead of the RESEARCH constant;
    * a CRITICAL line -- carrying the raising beat's full traceback -- is logged
      on *every* affected beat, so an ongoing outage keeps saying so rather than
      falling silent after the first alert;
    * a failed beat's mode is appended to the hash-chained ledger, so the
      dashboard's ledger-backed status source cannot keep reading the healthy
      ``ModeHeartbeat`` the tick stamped before it failed;
    * an alert is dispatched once per *transition* -- through the dispatcher it
      was composed with, and thence to the ledger.

    Each escalation kind's active state is tracked **independently**, and a kind
    re-arms only after :data:`_ESCALATION_CLEAR_RUN_BEATS` consecutive beats
    that could observe it and found it gone. A single last-escalated slot pages
    every beat under the conjunction these two issues describe -- a persistent
    kernel HALT behind an intermittent tick failure -- because the kind flips
    every beat and every beat then looks like a transition. So does re-arming on
    one quiet beat, because a near-full volume refuses some appends and accepts
    others. Both are pager storms at the issues' own 0.01s interval.

    A beat that raised is also *not* evidence about the kernel's mode: it never
    got far enough to read one. So a failing beat leaves the kernel-halt state
    exactly as it found it rather than clearing it.

    ``KeyboardInterrupt`` and ``SystemExit`` are deliberately not caught: those
    are shutdown requests, not tick failures.
    """

    def __init__(
        self,
        *,
        component: str,
        dispatcher: AlertDispatcher,
        mode_writer: LedgerModeWriter | None = None,
    ) -> None:
        """Initialize the supervisor.

        Args:
            component: The SPEC process token stamped on every escalation log
                record, matching the loop's own ``component``.
            dispatcher: The alert dispatcher every escalation is delivered
                through. :func:`_build_beat_supervisor` composes it from the
                CLI's real alert root, so escalations reach the configured
                sinks rather than a log-only fallback.
            mode_writer: Where a failed beat's mode is ledgered, so the
                dashboard stops reading the healthy row the tick stamped before
                it raised. ``None`` (the default) ledgers nothing, which is what
                a run without ``--ledger-path`` has to do.
        """
        self._component = component
        self._dispatcher = dispatcher
        self._mode_writer = mode_writer
        #: Currently-escalated kinds, each mapped to how many consecutive beats
        #: have since observed it gone. Absence means re-armed.
        self._active: dict[str, int] = {}
        self._escalating_beats = 0

    def observe(self, seq: int, on_beat: Callable[[int], BeatReport | None]) -> str:
        """Run one beat's hook and return the mode its heartbeat must report.

        Args:
            seq: The 1-based beat sequence number handed to the hook.
            on_beat: The hook to run for this beat.

        Returns:
            The mode token for this beat's heartbeat line: the tick's own mode,
            :data:`MODE_TICK_FAILED` if the hook raised, or
            :data:`MODE_RESEARCH` when the hook reported no mode at all.
        """
        try:
            report = on_beat(seq)
        except Exception as exc:
            self._settle(
                {
                    _ESCALATION_BEAT_FAILED: (
                        f"beat seq={seq} failed: {type(exc).__name__}: {exc}"
                    )
                },
                exc=exc,
            )
            if self._mode_writer is not None:
                self._mode_writer.record(seq, MODE_TICK_FAILED)
            return MODE_TICK_FAILED
        if report is None:
            self._settle({_ESCALATION_BEAT_FAILED: None})
            return MODE_RESEARCH
        if report.research_halted:
            _LOGGER.warning(
                "research budget halted at beat seq=%d (mode=%s)",
                seq,
                report.mode,
                extra={"component": self._component},
            )
        halt_message = (
            f"risk kernel HALT at beat seq={seq} (mode={report.mode})"
            if report.halted
            else None
        )
        self._settle(
            {_ESCALATION_BEAT_FAILED: None, _ESCALATION_KERNEL_HALT: halt_message}
        )
        return report.mode

    def wait_seconds(self, interval_seconds: float) -> float:
        """Return how long to wait before the next beat, backing off (#443).

        Args:
            interval_seconds: The configured inter-beat interval.

        Returns:
            ``interval_seconds`` while beats are clean, doubled once per
            consecutive escalating beat and capped at
            ``2 ** _MAX_ESCALATION_BACKOFF_DOUBLINGS`` times it. A zero
            interval stays zero, so a test loop is never slowed.
        """
        doublings = min(self._escalating_beats, _MAX_ESCALATION_BACKOFF_DOUBLINGS)
        return interval_seconds * float(2**doublings)

    def _settle(
        self,
        observed: Mapping[str, str | None],
        *,
        exc: BaseException | None = None,
    ) -> None:
        """Fold one beat's observations into the per-kind escalation state.

        Args:
            observed: The escalation kinds this beat was *able* to observe,
                each mapped to its operator-facing message if the condition is
                present or ``None`` if it is absent. A kind the beat could not
                observe is simply omitted, leaving its state untouched.
            exc: The exception a raising beat produced, logged as the CRITICAL
                record's ``exc_info`` so the operator gets the traceback that
                names the failing call site.
        """
        self._escalating_beats = (
            self._escalating_beats + 1
            if any(message is not None for message in observed.values())
            else 0
        )
        for kind, message in observed.items():
            if message is None:
                self._clear(kind)
            else:
                self._escalate(kind, message, exc)

    def _clear(self, kind: str) -> None:
        """Count one beat that observed ``kind`` gone, re-arming after a run.

        Args:
            kind: The escalation kind this beat found absent.
        """
        streak = self._active.get(kind)
        if streak is None:
            return
        if streak + 1 >= _ESCALATION_CLEAR_RUN_BEATS:
            del self._active[kind]
        else:
            self._active[kind] = streak + 1

    def _escalate(self, kind: str, message: str, exc: BaseException | None) -> None:
        """Log this beat's escalation, alerting once per transition into it.

        Args:
            kind: The escalation kind, one of :data:`_ESCALATION_BEAT_FAILED`
                or :data:`_ESCALATION_KERNEL_HALT`. Alerts deduplicate on it,
                so a condition that persists pages once and a *different*
                condition arriving behind it still pages.
            message: The operator-facing line, logged and dispatched verbatim.
            exc: The exception to render as a traceback beside ``message``, or
                ``None`` for an escalation that had no exception (a halt).
        """
        _LOGGER.critical(
            "%s", message, exc_info=exc, extra={"component": self._component}
        )
        first_beat = kind not in self._active
        self._active[kind] = 0
        if first_beat:
            self._dispatcher.dispatch(AlertType.HALT_KILL, message)


class LedgerAlertWriter:
    """Persist each dispatched alert to the hash-chained ledger (issue #443).

    The :class:`~windbreak.alerts.LedgerWriter` the CLI wires in place of
    :class:`~windbreak.alerts.LoggingLedgerWriter` when ``--ledger-path`` is
    given, so "nothing is ledgered about the cause" stops being true of a beat
    that failed. The store is opened per record and closed again: escalations
    are once-per-transition events, so this costs nothing measurable and leaves
    no second long-lived connection contending with the PAPER tick's own.

    A ledger write that fails is logged at CRITICAL and swallowed, for the
    reason :func:`_append_swallowing` records.

    **Two rows, one dispatch, and no order of precedence (issue #488).** A kill
    appends *two* ``AlertEmitted`` rows to the one chain: this writer's, under
    the run's ``--process`` token, and
    :meth:`~windbreak.riskkernel.kill.KillSwitch.kill`'s own tamper-evident
    audit row under ``riskkernel``, appended only after every fail-safe effect
    has landed (issue #287). Neither is removable. Dropping this one would
    un-ledger every *other* alert the run dispatches, which is the whole of
    #443; dropping the kill switch's would surrender the ordering guarantee
    that its row cannot precede the ``KILL`` file.

    So both stay -- and the reader is given the only rule that needs no
    convention: **they carry identical payloads**. Both project the same
    dispatched :class:`~windbreak.alerts.AlertEmitted` through
    :func:`~windbreak.alerts.ledger_deliveries`, so either row is authoritative
    and ``component`` records *provenance*, never precedence. Until #488 that
    was false in the worst direction: this writer was handed the dispatch and
    kept only its severity and body, so its schema-2 row stamped
    ``delivery_reported: false`` -- an explicit denial of evidence its own
    parameter held, which an auditor scanning by ``event_type`` rather than by
    ``component`` reads as a false negative for a page that was delivered. A
    schema-1 row made no claim; this one made a wrong one, permanently.
    """

    def __init__(self, ledger_path: Path, *, component: str) -> None:
        """Initialize the writer.

        Args:
            ledger_path: The hash-chained ledger database to append to.
            component: The SPEC process token stamped on each appended event.
        """
        self._ledger_path = ledger_path
        self._component = component

    def record(self, event: AlertEmitted) -> None:
        """Append one emitted alert, and its delivery evidence, never raising.

        ``delivery_reported`` is unconditionally ``True``: this seam is reached
        only from :meth:`~windbreak.alerts.AlertDispatcher.dispatch`, which
        hands over the concrete ``AlertEmitted`` it just built, so the report
        is never absent here and a fail-closed "unknown" would be a second
        false claim in place of the first.

        The delivery rows come from
        :func:`~windbreak.alerts.ledger_deliveries` rather than from
        ``event.outcomes``, and that is the security-relevant half: it is the
        single producer of ledgered delivery evidence, emitting only a screened
        sink identity, an enumerated outcome and a fallback flag.
        ``SinkOutcome.detail`` is ``str(exc)`` from an arbitrary sink -- the
        shape that leaked whole token-bearing URLs in issue #274 -- and this
        chain can never be redacted, so it must not be reachable from here.

        Args:
            event: The dispatched alert to persist, carrying the per-sink
                outcomes of the dispatch that produced it.
        """
        _append_swallowing(
            self._ledger_path,
            LedgerAlertEmitted(
                component=self._component,
                severity=event.severity.value,
                message=event.message,
                deliveries=ledger_deliveries(event),
                delivery_reported=True,
            ),
            component=self._component,
            what="alert",
        )


class LedgerModeWriter:
    """Ledger a supervised beat's own outcome mode (issue #447).

    Without this the dashboard reports health straight through an outage.
    ``run_single_tick`` appends its ``ModeHeartbeat`` *before* the stages that
    can raise, so a beat that fails after that point has already stamped a
    healthy ``PAPER`` row -- and
    :func:`_build_dashboard_status_source`'s ledger-backed source reads the
    latest row and reports exactly that, re-stamped to now, every failing beat.
    A value re-stamped to now can never veto anything.

    So the supervisor appends its own row, *after* the tick's, naming the
    outcome the tick actually had. Only failed beats are written: a beat that
    completed already ledgered its real mode from ``deps.kernel.mode``, and
    duplicating that would be noise. The first clean beat after an outage
    therefore returns the dashboard to the tick's own mode by itself.

    Like :class:`LedgerAlertWriter` this opens the store per record and swallows
    a write failure at CRITICAL. When the outage *is* the ledger volume, this
    append fails too and the dashboard keeps its stale row -- the honest limit
    of a ledger-backed status source on a full disk, and the reason the failure
    is announced on the log line and the alert path as well as here.
    """

    def __init__(self, ledger_path: Path, *, component: str) -> None:
        """Initialize the writer.

        Args:
            ledger_path: The hash-chained ledger database to append to.
            component: The SPEC process token stamped on each appended event.
        """
        self._ledger_path = ledger_path
        self._component = component

    def record(self, seq: int, mode: str) -> None:
        """Append one beat's outcome mode to the ledger, never raising.

        Args:
            seq: The 1-based beat sequence number.
            mode: The mode token this beat's heartbeat reports.
        """
        _append_swallowing(
            self._ledger_path,
            ModeHeartbeat(component=self._component, mode=mode, beat=seq),
            component=self._component,
            what="beat mode",
        )


def _append_swallowing(
    ledger_path: Path, event: Event, *, component: str, what: str
) -> None:
    """Append one event to the ledger, logging and swallowing any failure.

    The deliberate order of two hazards, shared by :class:`LedgerAlertWriter`
    and :class:`LedgerModeWriter`. The failure these exist to record is a full
    ledger volume -- the very condition that makes appending the record fail
    too -- so letting it raise would put the daemon back exactly where #443
    found it. Losing the audit row is the lesser harm only because the loss is
    announced at CRITICAL rather than passed over in silence.

    Args:
        ledger_path: The hash-chained ledger database to append to.
        event: The event to persist.
        component: The SPEC process token stamped on the failure log record.
        what: What was being ledgered, named in the failure message.
    """
    try:
        store = SqliteLedgerStore(ledger_path)
        try:
            store.append(event)
        finally:
            store.close()
    except Exception as exc:
        _LOGGER.critical(
            "%s ledgering failed: %s: %s",
            what,
            type(exc).__name__,
            exc,
            exc_info=exc,
            extra={"component": component},
        )


def _beat_stop_reason(
    *,
    stop_event: threading.Event,
    state: ShutdownState | None,
    max_beats: int | None,
    beats_done: int,
) -> str | None:
    """Decide whether the heartbeat loop should stop before its next beat.

    Args:
        stop_event: The event a signal handler sets to request shutdown.
        state: The shared shutdown state, whose ``reason`` names the delivered
            signal when a handler recorded one.
        max_beats: The optional beat budget.
        beats_done: How many beats have already run.

    Returns:
        The shutdown reason to log, or ``None`` to run another beat.
    """
    if stop_event.is_set():
        if state is not None and state.reason is not None:
            return state.reason
        return _REASON_SIGNAL
    if max_beats is not None and beats_done >= max_beats:
        return _REASON_MAX_BEATS
    return None


def run_loop(
    interval_seconds: float,
    *,
    max_beats: int | None = None,
    stop_event: threading.Event | None = None,
    state: ShutdownState | None = None,
    component: str = _DEFAULT_PROCESS,
    on_beat: Callable[[int], BeatReport | None] | None = None,
    supervisor: BeatSupervisor | None = None,
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
            number, before that beat's heartbeat is logged. None (the default)
            leaves the heartbeat behavior unchanged. Its optional
            :class:`BeatReport` is what the heartbeat reports the mode from.
        supervisor: Optional :class:`BeatSupervisor` running the hook and
            setting the inter-beat backoff. Defaults to one dispatching through
            the log-only fallback, so survival never depends on a caller
            remembering to supply it; :func:`_build_beat_supervisor` provides
            the configured one.

    Raises:
        BaseException: Whatever ``on_beat`` raises that is not an
            :class:`Exception` -- ``KeyboardInterrupt`` and ``SystemExit`` are
            shutdown requests, not tick failures, so they still stop the loop.
    """
    if stop_event is None:
        stop_event = state.stop_event if state is not None else threading.Event()
    if supervisor is None:
        supervisor = BeatSupervisor(
            component=component,
            dispatcher=AlertDispatcher(sinks=[], ledger_writer=LoggingLedgerWriter()),
        )

    seq = 0
    reason = _REASON_SIGNAL
    while True:
        stop_reason = _beat_stop_reason(
            stop_event=stop_event, state=state, max_beats=max_beats, beats_done=seq
        )
        if stop_reason is not None:
            reason = stop_reason
            break
        seq += 1
        # The beat runs *before* its heartbeat is logged: the mode the line
        # reports is this beat's own, and a beat cannot report a mode it has
        # not yet run (issue #447). Reporting the previous beat's mode would be
        # a lie by one beat's lag, which on a halt is the whole defect.
        mode = MODE_RESEARCH if on_beat is None else supervisor.observe(seq, on_beat)
        _LOGGER.info(
            "mode=%s heartbeat seq=%d",
            mode,
            seq,
            extra={"component": component},
        )
        # Backoff, not full cadence, while beats keep escalating (issue #443's
        # "continue on the next beat with backoff"): a loop whose every beat
        # fails gains nothing by retrying instantly and floods a log that may
        # be on the volume that filled. The stop event still cuts the wait
        # short, so shutdown is never delayed by a backoff.
        stop_event.wait(supervisor.wait_seconds(interval_seconds))

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


def _build_alert_dispatcher(
    config: WindbreakConfig, *, ledger_writer: LedgerWriter | None = None
) -> AlertDispatcher:
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
        ledger_writer: Where each emitted alert is recorded. Defaults to the
            logging writer; :func:`_build_beat_supervisor` passes a
            :class:`LedgerAlertWriter` so a supervised beat failure reaches the
            hash chain as well as the sinks (issue #443).

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
        ledger_writer=(
            LoggingLedgerWriter() if ledger_writer is None else ledger_writer
        ),
    )


def _beat_alert_root(
    args: argparse.Namespace, config: WindbreakConfig
) -> Callable[[], AlertDispatcher]:
    """Return the compose-once factory for this run's single alert root (#444).

    Every per-beat seam of one ``run`` -- the :class:`BeatSupervisor` that
    escalates a failed beat or a kernel halt (issues #443/#447), and the PAPER
    tick's own read-only verification cycle (issue #444) -- delivers through the
    *same* :class:`AlertDispatcher` object, composed here at most once. One
    alert root means one set of live sinks, one allowlist screen, and one place
    that ever reads a destination out of the environment; two roots over the
    same configuration would be a second parallel alerting path that could drift
    from the first without anything failing.

    It is a factory rather than a dispatcher because composition must not happen
    at all on a run with no per-beat hook: a bare RESEARCH heartbeat has nothing
    to alert about, and making it refuse to start over a sink it will never use
    would be a new failure mode rather than a fixed one. Deferring also puts the
    composition *ahead* of :func:`_build_paper_on_beat`'s bundle, so an
    undeliverable sink refuses before a SQLite ledger has been created and
    abandoned.

    Args:
        args: The parsed ``run`` arguments carrying ``ledger_path`` and the
            ``process`` token stamped on ledgered alerts.
        config: The loaded configuration supplying the sinks and allowlist.

    Returns:
        A zero-argument callable yielding the one composed dispatcher. Calling
        it a second time returns the same object, never a second composition.
    """
    composed: list[AlertDispatcher] = []

    def _root() -> AlertDispatcher:
        """Compose the alert root on first call and return it on every later one.

        Returns:
            The single dispatcher every per-beat seam of this run shares.

        Raises:
            AlertSinkConfigError: If a configured sink cannot be composed.
        """
        if not composed:
            ledger_writer: LedgerWriter = LoggingLedgerWriter()
            if args.ledger_path is not None:
                ledger_writer = LedgerAlertWriter(
                    args.ledger_path, component=args.process
                )
            composed.append(
                _build_alert_dispatcher(config, ledger_writer=ledger_writer)
            )
        return composed[0]

    return _root


def _build_beat_supervisor(
    args: argparse.Namespace, *, dispatcher: AlertDispatcher
) -> BeatSupervisor:
    """Compose the beat supervisor over the CLI's real alert root (#443/#447).

    With ``--ledger-path`` the supervisor also gets a :class:`LedgerModeWriter`,
    so a failed beat's mode reaches the same ledger the dashboard's status
    source reads and the dashboard cannot keep reporting the healthy row the
    tick stamped before it raised (issue #447).

    The ``dispatcher`` is handed in rather than composed here, and it is the
    same object :func:`_build_paper_on_beat` threads into the PAPER loop's
    verification cycle (issue #444). Before that, the supervisor was composed
    from :func:`_build_alert_dispatcher` while the PAPER loop built its own
    ``AlertDispatcher(sinks=[])``, so one of the two alerting surfaces an
    unattended run has reached the operator and the other could not.

    Args:
        args: The parsed ``run`` arguments carrying ``ledger_path`` and the
            ``process`` token stamped on escalations.
        dispatcher: The run's single composed alert root, from
            :func:`_beat_alert_root`.

    Returns:
        The supervisor :func:`run_loop` runs each beat's hook through.
    """
    mode_writer: LedgerModeWriter | None = None
    if args.ledger_path is not None:
        mode_writer = LedgerModeWriter(args.ledger_path, component=args.process)
    return BeatSupervisor(
        component=args.process,
        dispatcher=dispatcher,
        mode_writer=mode_writer,
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


def _unfoldable_ledger_refusal(action: str, market_ticker: str, reason: str) -> str:
    """Phrase the refusal for a ledger whose existing rows cannot be folded.

    Args:
        action: The verb's own word for what it was about to do (``"ingest"``
            or ``"correct"``), so the operator reads back the command they ran.
        market_ticker: The market this call was about.
        reason: The fold's own message, quoted verbatim.

    Returns:
        The operator-facing refusal.
    """
    return (
        f"refusing to {action} market_ticker={market_ticker!r}: this ledger's "
        f"existing MarketResolved rows cannot be folded -- {reason} Until that "
        "is resolved the weekly report cannot be built at all, and appending "
        "another row would not help. Nothing was written."
    )


def _folded_or_refusal(
    store: SqliteLedgerStore, action: str, market_ticker: str
) -> tuple[ResolutionFold | None, str]:
    """Fold the ledger's ground truth, or explain why it cannot be folded.

    Both resolution verbs read the ledger back through the *same* fold the
    weekly tick uses, so "this verb refuses exactly what the tick would refuse
    to read" holds by construction rather than by a parallel reimplementation
    that can drift.

    Args:
        store: The open ledger store, read but not yet written.
        action: The verb's own word for what it was about to do.
        market_ticker: The market this call was about.

    Returns:
        The fold paired with ``""``, or ``None`` paired with the refusal.
    """
    try:
        return fold_resolutions(store.read_all()), ""
    except ValueError as exc:
        return None, _unfoldable_ledger_refusal(action, market_ticker, str(exc))


def _conflicting_ingest_refusal(store: SqliteLedgerStore, event: MarketResolved) -> str:
    """Explain why this resolution must not be appended, or return ``""``.

    The read-back that makes ``ingest-resolution`` safe (issue #439 review).
    The ledger is append-only and
    :func:`~windbreak.scheduler.weekly_data.weekly_report_body` folds it on
    *every* tick, so a contradicting row is not a bad record -- it is a
    permanent, unrecoverable stop on the always-on loop, reachable through a
    well-formed CLI call. Refusing here costs the operator one retyped command;
    refusing at the fold costs them the loop.

    The check reads through :func:`ingested_resolutions_from_records`, the same
    function the tick uses, so "this verb refuses exactly what the tick would
    refuse to read" holds by construction rather than by a parallel
    reimplementation that can drift.

    Args:
        store: The open ledger store, read but not yet written.
        event: The ``MarketResolved`` event this call would append.

    Returns:
        The operator-facing refusal, or ``""`` when the append is safe.
    """
    candidate = ingested_resolution_of(event)
    fold, refusal = _folded_or_refusal(store, "ingest", candidate.market_ticker)
    if fold is None:
        return refusal
    existing = fold.resolution_of(candidate.market_ticker)
    if existing is None or not resolutions_conflict(existing, candidate):
        return ""
    target = fold.claim_sequence_numbers[candidate.market_ticker]
    return (
        f"refusing to ingest market_ticker={candidate.market_ticker!r}: it "
        f"already resolved on this ledger with "
        f"{describe_resolution_claim(existing)}, and this call claims "
        f"{describe_resolution_claim(candidate)}. The ledger is append-only, so "
        "a contradicting row could never be un-written and every later weekly "
        "fold -- one per tick -- would refuse to read it. Nothing was written. "
        "Re-run with the values the ledger already carries; a genuinely wrong "
        "recorded outcome is corrected with `windbreak correct-resolution "
        f"--superseded-sequence-number {target}`, which supersedes that row "
        "rather than contradicting it, and never by ingesting again."
    )


def _run_ingest_resolution(args: argparse.Namespace) -> int:
    """Append one ``MarketResolved`` row recording that a market settled.

    This is the seam through which ground truth enters the running system
    (issue #439). Nothing else ever learns that a market resolved, so until an
    operator runs this every evaluation metric the weekly report prints reads
    ``UNDEFINED`` -- structurally, not for want of data. The next weekly fold
    (:func:`windbreak.scheduler.weekly_data.weekly_report_body`) reads the row
    back off the same ledger and scores the forecasts on that market against it.

    The resolution's own fields are validated *before* the ledger is opened, so
    a malformed ingest leaves the chain -- and, on a first run, the database
    file itself -- untouched. A half-ingested resolution would be worse than
    none: the report would then quote a number nothing supports.

    The resolution is then checked *against the ledger* before anything is
    appended (:func:`_conflicting_ingest_refusal`). That second check is what
    keeps a mistyped operator command from being terminal: the ledger is
    append-only and the weekly fold runs on every tick, so a contradicting row
    would stop the always-on loop permanently, with no correction possible.
    Refusing here writes nothing and leaves the loop beating.

    Args:
        args: Parsed ``ingest-resolution`` arguments carrying ``ledger_path``,
            ``market_ticker``, ``outcome``, ``resolved_at`` and ``source``.

    Returns:
        The process exit code: 0 once the row is appended, 1 on a refused
        resolution (with the reason logged as a ``FATAL`` critical).
    """
    try:
        event = MarketResolved(
            component=_RESOLUTION_INGEST_COMPONENT,
            market_ticker=args.market_ticker,
            outcome=ResolutionOutcome(args.outcome),
            resolved_at=parse_resolution_instant(args.resolved_at),
            source=args.source,
        )
    except ValueError as exc:
        _LOGGER.critical("FATAL: %s", exc)
        return 1
    store = SqliteLedgerStore(args.ledger_path)
    try:
        refusal = _conflicting_ingest_refusal(store, event)
        if refusal:
            _LOGGER.critical("FATAL: %s", refusal)
            return 1
        sequence_number = store.append(event)
    finally:
        store.close()
    _LOGGER.info(
        "resolution ingested ticker=%s outcome=%s resolved_at=%s source=%s sequence=%d",
        event.market_ticker,
        event.outcome.value,
        event.payload["resolved_at"],
        event.source,
        sequence_number,
    )
    return 0


def _no_resolution_refusal(market_ticker: str) -> str:
    """Phrase the refusal for correcting a market that never resolved.

    Args:
        market_ticker: The market this call named.

    Returns:
        The operator-facing refusal, naming the verb that *would* apply.
    """
    return (
        f"refusing to correct market_ticker={market_ticker!r}: this ledger "
        "carries no resolution for that market, so there is nothing to "
        "supersede. Record it with `windbreak ingest-resolution` instead. "
        "Nothing was written."
    )


def _wrong_target_refusal(
    market_ticker: str, named: int, target: int, current: IngestedResolution
) -> str:
    """Phrase the refusal for a correction naming the wrong ledger row.

    Args:
        market_ticker: The market this call named.
        named: The ``--superseded-sequence-number`` the operator typed.
        target: The position that actually carries the market's claim.
        current: The claim that row makes.

    Returns:
        The operator-facing refusal. It names the correct position rather than
        only rejecting the wrong one: a refusal with no way forward is the trap
        this verb exists to remove, and a stale target is exactly what an
        operator correcting an already-corrected market will type.
    """
    return (
        f"refusing to correct market_ticker={market_ticker!r}: "
        f"--superseded-sequence-number={named} does not name the row carrying "
        f"this market's current resolution, which is sequence_number={target} "
        f"({describe_resolution_claim(current)}). A correction supersedes "
        "exactly the row it names, so naming any other position would leave "
        "two rows claiming one market with no rule for which wins. Nothing "
        f"was written. Re-run with --superseded-sequence-number={target}."
    )


def _no_op_correction_refusal(
    market_ticker: str, target: int, claim: IngestedResolution
) -> str:
    """Phrase the refusal for a correction that changes nothing.

    Args:
        market_ticker: The market this call named.
        target: The position being superseded.
        claim: The claim both the ledger and this call make.

    Returns:
        The operator-facing refusal.
    """
    return (
        f"refusing to correct market_ticker={market_ticker!r}: this call "
        f"claims {describe_resolution_claim(claim)}, which is exactly what "
        f"sequence_number={target} already carries. A correction that changes "
        "nothing would record an unexplained reversal in an append-only audit "
        "trail without moving a single metric. Nothing was written."
    )


def _correction_refusal(store: SqliteLedgerStore, event: SettlementReversed) -> str:
    """Explain why this correction must not be appended, or return ``""``.

    The read-back that makes ``correct-resolution`` safe, on #482's principle:
    refuse at the verb, never at the fold. The ledger is append-only and
    :func:`~windbreak.scheduler.weekly_data.weekly_report_body` folds it on
    *every* tick, so a correction naming a row that does not carry the market's
    claim would not be a bad record -- it would be a permanent stop on the
    always-on loop, reachable through a well-formed CLI call. Every refusal
    here costs the operator one retyped command and names the row to retype.

    Args:
        store: The open ledger store, read but not yet written.
        event: The ``SettlementReversed`` event this call would append.

    Returns:
        The operator-facing refusal, or ``""`` when the append is safe.
    """
    ticker = event.market_ticker
    fold, refusal = _folded_or_refusal(store, "correct", ticker)
    if fold is None:
        return refusal
    current = fold.resolution_of(ticker)
    if current is None:
        return _no_resolution_refusal(ticker)
    target = fold.claim_sequence_numbers[ticker]
    if target != event.superseded_sequence_number:
        return _wrong_target_refusal(
            ticker, event.superseded_sequence_number, target, current
        )
    corrected = ingested_resolution_of(event)
    if not resolutions_conflict(current, corrected):
        return _no_op_correction_refusal(ticker, target, corrected)
    return ""


def _run_correct_resolution(args: argparse.Namespace) -> int:
    """Append one ``SettlementReversed`` row superseding a wrong resolution.

    The correction path issue #484 asked for. Before it, a *first*
    ``ingest-resolution`` call with a wrong ``--outcome`` exited 0, nothing
    contradicted it, and the append-only ledger carried that false settlement
    forever -- with every weekly metric folded from it wrong, permanently.

    Nothing is redacted, because nothing on a hash-chained ledger can be: the
    wrong row stays exactly where it is, and this row supersedes it by naming
    its ``sequence_number``. Both are visible afterwards, and the weekly
    report's ``## Resolution corrections`` section names the superseded claim,
    so a corrected outcome can never be mistaken for one that was right all
    along.

    The event's own fields are validated *before* the ledger is opened, then
    the correction is checked *against the ledger* (:func:`_correction_refusal`)
    before anything is appended.

    Args:
        args: Parsed ``correct-resolution`` arguments carrying ``ledger_path``,
            ``market_ticker``, ``superseded_sequence_number``, ``outcome``,
            ``resolved_at`` and ``source``.

    Returns:
        The process exit code: 0 once the row is appended, 1 on a refused
        correction (with the reason logged as a ``FATAL`` critical).
    """
    try:
        event = SettlementReversed(
            component=_RESOLUTION_INGEST_COMPONENT,
            market_ticker=args.market_ticker,
            superseded_sequence_number=args.superseded_sequence_number,
            outcome=ResolutionOutcome(args.outcome),
            resolved_at=parse_resolution_instant(args.resolved_at),
            source=args.source,
        )
    except ValueError as exc:
        _LOGGER.critical("FATAL: %s", exc)
        return 1
    store = SqliteLedgerStore(args.ledger_path)
    try:
        refusal = _correction_refusal(store, event)
        if refusal:
            _LOGGER.critical("FATAL: %s", refusal)
            return 1
        sequence_number = store.append(event)
    finally:
        store.close()
    _LOGGER.info(
        "resolution corrected ticker=%s superseded=%d outcome=%s resolved_at=%s "
        "source=%s sequence=%d",
        event.market_ticker,
        event.superseded_sequence_number,
        event.outcome.value,
        event.payload["resolved_at"],
        event.source,
        sequence_number,
    )
    return 0


def _evaluate_providers_refusal(ledger_path: Path, report_dir: Path) -> str | None:
    """Return why the provider-scoring pass cannot run, or ``None``.

    Both refusals guard the same fail-open shape, in opposite directions.

    An absent ``--ledger-path`` would be *created* empty by
    :class:`~windbreak.ledger.store.SqliteLedgerStore`, fold to zero providers,
    and exit 0 having written ``{}`` -- shutting a gate the operator believed
    they were opening, on evidence from a ledger that is not the loop's.

    An absent ``--report-dir`` would be a directory the loop never reads, so the
    operator would be told a provider was proven while the running loop's gate
    stayed in bootstrap forever. Neither is created here: creating one is
    exactly how a typo becomes a confident, invisible success.

    Args:
        ledger_path: The ``--ledger-path`` value to check.
        report_dir: The ``--report-dir`` value to check.

    Returns:
        The operator-facing refusal, or ``None`` when both paths are usable.
    """
    if ledger_path.is_dir():
        return f"--ledger-path is a directory, not a ledger database: {ledger_path}"
    if not ledger_path.exists():
        return (
            f"no ledger database at {ledger_path}: refusing to create one. An "
            "empty ledger scores no provider at all, which would publish an "
            "empty track record over whatever the loop's gate reads today."
        )
    if not report_dir.is_dir():
        return (
            f"no report directory at {report_dir}: refusing to create one. The "
            "gate reads its artifact from the directory the loop was started "
            "with, so a track record written anywhere else is never read."
        )
    return None


def _run_evaluate_providers(args: argparse.Namespace) -> int:
    """Score each provider's resolved forecasts into the provider gate's artifact.

    The evaluation pass ``docs/RUNBOOK.md`` and ``EVALUATION.md`` have always
    described and no code performed (issue #440). Until this verb existed,
    ``provider-track-records.json`` was written by nothing at all, so
    :func:`windbreak.scheduler.loop._build_provider_gate`'s documented
    "bootstrap" -- every provider unproven -- was terminal rather than
    transient, and no run of any length could ever lift the hold.

    It reads the same hash-chained ledger the loop writes, folds the
    ``MarketResolved`` rows an operator ingested through ``ingest-resolution``
    against the ``ForecastCreated`` rows their markets settled, attributes each
    forecast to the providers whose votes survived to back it, and writes the
    strict-integer document the gate parses. Nothing optimistic is ever
    written: a provider with no admitted resolved forecast, or an undefined
    skill, is omitted, and omitted reads as unproven.

    The gate is built once per process, so the loop picks the new artifact up
    on its **next start** -- see ``docs/RUNBOOK.md``, "Prove a provider before
    it can back a live order".

    Args:
        args: Parsed ``evaluate-providers`` arguments carrying ``ledger_path``
            and ``report_dir``.

    Returns:
        The process exit code: 0 once the artifact is published, 1 on a refused
        path or a ledger that cannot be folded (with the reason logged as a
        ``FATAL`` critical, and nothing written).
    """
    refusal = _evaluate_providers_refusal(args.ledger_path, args.report_dir)
    if refusal is not None:
        _LOGGER.critical("FATAL: %s", refusal)
        return 1
    store = SqliteLedgerStore(args.ledger_path)
    try:
        records = provider_track_records(store.read_all())
    except ValueError as exc:
        _LOGGER.critical(
            "FATAL: refusing to publish a track record from %s: %s Nothing was "
            "written.",
            args.ledger_path,
            exc,
        )
        return 1
    finally:
        store.close()
    artifact = write_provider_track_records(args.report_dir, records)
    _LOGGER.info(
        "provider track records published path=%s providers=%d summary=%s",
        artifact,
        len(records),
        _track_record_summary(records),
    )
    return 0


def _track_record_summary(records: tuple[ProviderTrackRecord, ...]) -> str:
    """Render the published records for the verb's own log line.

    Args:
        records: The published track records, in document order.

    Returns:
        A ``provider=resolved/skill`` fragment per record, or an explicit
        ``none`` -- never an empty string, which would read as a formatting
        slip rather than as "no provider earned a record".
    """
    if not records:
        return "none"
    return " ".join(
        f"{record.provider}={record.resolved_count}/{record.brier_skill_ppm}ppm"
        for record in records
    )


def _existing_ledger_refusal(ledger_path: Path) -> str | None:
    """Return why ``--ledger-path`` cannot be appended to, or ``None``.

    ``set-research-budget`` is the one verb whose whole purpose is to *stop*
    money, so a mistyped path must not look like success. Opening a
    :class:`~windbreak.ledger.store.SqliteLedgerStore` on a path that does not
    exist creates the database, appends at sequence 1, logs a cheerful "ceiling
    set to 0 micros sequence=1" and exits 0 -- while the running loop, reading a
    different file, never sees the cap at all. That is a fail-open hazard on a
    spend brake: the operator is told the brake is on and it is not.

    A directory is refused for the same reason with a message rather than an
    uncaught ``sqlite3.OperationalError`` traceback, which this verb's own
    docstring already promised was an exit-1 refusal.

    Args:
        ledger_path: The ``--ledger-path`` value to check.

    Returns:
        The refusal message, or ``None`` when the path is an existing file.
    """
    if ledger_path.is_dir():
        return f"--ledger-path is a directory, not a ledger database: {ledger_path}"
    if not ledger_path.exists():
        return (
            f"no ledger database at {ledger_path}: refusing to create one. A "
            "cap change appended to a new ledger is invisible to the running "
            "loop, which reads the ledger it was started with."
        )
    return None


def _run_set_research_budget(args: argparse.Namespace) -> int:
    """Append one ``ResearchBudgetCapSet`` row changing the daily research cap.

    The runtime half of the owner's 2026-08-10 decision (#483) and the reason
    #442's durable counter is usable rather than merely correct: an operator
    re-caps a *running* loop, which re-reads the ledger at the head of every
    tick, so the change takes effect on the next beat with no restart and no
    source edit.

    A later row supersedes an earlier one and two rows can therefore never
    contradict, which is why -- unlike ``ingest-resolution`` (issue #439) --
    there is no read-back conflict check here and nothing this verb writes can
    ever stop a tick. What it does refuse, it refuses *before* opening the
    ledger: the event's own ``__post_init__`` rejects a negative ceiling and a
    blank note, and :func:`_existing_ledger_refusal` rejects a ``--ledger-path``
    that is not an existing database -- so a refused call leaves the chain
    byte-for-byte untouched and creates no database file at all.

    That second refusal is the one an operator's fingers need. A typo'd path
    used to be silently *successful*: a fresh ledger was created, the row landed
    at sequence 1, the log said the ceiling was set, the exit code was 0, and
    the running loop -- reading the ledger it was started with -- never saw it.
    On the verb whose purpose is to stop money, that is a fail-open.

    Args:
        args: Parsed ``set-research-budget`` arguments carrying ``ledger_path``,
            ``per_day_micros`` and ``note``.

    Returns:
        The process exit code: 0 once the row is appended, 1 on a refused
        change (with the reason logged as a ``FATAL`` critical).
    """
    try:
        event = ResearchBudgetCapSet(
            component=_RESEARCH_BUDGET_COMPONENT,
            per_day_micros=args.per_day_micros,
            note=args.note,
        )
    except ValueError as exc:
        _LOGGER.critical("FATAL: %s", exc)
        return 1
    refusal = _existing_ledger_refusal(args.ledger_path)
    if refusal is not None:
        _LOGGER.critical("FATAL: %s", refusal)
        return 1
    store = SqliteLedgerStore(args.ledger_path)
    try:
        sequence_number = store.append(event)
    finally:
        store.close()
    _LOGGER.info(
        "research per-day ceiling set to %d micros sequence=%d note=%s",
        event.per_day_micros,
        sequence_number,
        event.note,
    )
    return 0


def research_capped_config(
    config: WindbreakConfig, *, per_day_micros: int | None
) -> WindbreakConfig:
    """Apply ``run --research-per-day-micros`` to a loaded configuration.

    The invocation-argument half of "settable without a source-code change".
    Exactly one field is replaced, through :func:`dataclasses.replace` at each
    level, so an override cannot silently reset the per-forecast ceiling or the
    page cap alongside the one it was asked to change.

    With ``per_day_micros`` ``None`` the *same object* is returned, not an equal
    copy: every deployment that does not pass the flag must be byte-identical to
    before, and returning the original is the only version of that claim a test
    can pin by identity.

    Public because ``tests/test_cli_set_research_budget.py`` exercises it
    directly: it is a pure function of a config and a flag, and testing it
    through a whole ``run`` invocation would prove less about the field-level
    behaviour that matters here.

    Args:
        config: The loaded configuration.
        per_day_micros: The overriding per-UTC-day ceiling in micros, or
            ``None`` to leave the configuration untouched (keyword-only).

    Returns:
        The configuration the run should use.
    """
    if per_day_micros is None:
        return config
    budget = dataclasses.replace(config.forecast.budget, per_day_micros=per_day_micros)
    forecast = dataclasses.replace(config.forecast, budget=budget)
    return dataclasses.replace(config, forecast=forecast)


def _startup_ceiling_source(args: argparse.Namespace) -> str:
    """Return which non-ledger source set the ceiling this run starts on.

    Args:
        args: Parsed ``run`` arguments carrying ``research_per_day_micros``.

    Returns:
        :data:`_CEILING_SOURCE_FLAG` when the flag was supplied, otherwise
        :data:`_CEILING_SOURCE_CONFIG`.
    """
    if args.research_per_day_micros is not None:
        return _CEILING_SOURCE_FLAG
    return _CEILING_SOURCE_CONFIG


def _effective_research_ceiling(
    config: WindbreakConfig, ledger_path: Path | None, *, args: argparse.Namespace
) -> tuple[int, str]:
    """Fold the ceiling actually in force, and name the source that won.

    The startup log has to report the **folded** ceiling, not the configured
    one. ``config.forecast.budget.per_day_micros`` is only the fallback: by
    :func:`~windbreak.scheduler.research_spend.effective_per_day_micros`'s
    precedence rule a ledgered ``ResearchBudgetCapSet`` row always wins when
    present, so an operator who had run ``set-research-budget --per-day-micros
    0`` and then restarted with ``--research-per-day-micros 40000000`` used to
    be told the ceiling was 40 000 000 while 0 was the ceiling in force. A log
    line that is wrong in the *permissive* direction on a spend brake is worse
    than no log line.

    Both folds run, not just the cap one: a malformed ``ResearchSpendRecorded``
    row leaves the ceiling readable but makes
    :func:`windbreak.scheduler.loop._build_research_budget` open a zero one, and
    reporting the configured ceiling in that case would be the same lie in the
    same direction.

    Without ``--ledger-path`` there is no ledger to fold and no loop that could
    read one, so the configured value stands and the source is the flag or the
    configuration. The store is opened read-only in the sense that matters --
    nothing is appended -- and closed in a ``finally`` so the PAPER loop can
    reopen the same file.

    Args:
        config: The already-``--research-per-day-micros``-capped configuration.
        ledger_path: The run's ``--ledger-path``, or ``None``.
        args: The parsed ``run`` arguments, for the non-ledger source label
            (keyword-only).

    Returns:
        The ``(per_day_micros, source)`` pair the startup log should report.
    """
    configured = config.forecast.budget.per_day_micros
    if ledger_path is None:
        return configured, _startup_ceiling_source(args)
    store = SqliteLedgerStore(ledger_path)
    try:
        records = tuple(store.read_all())
    finally:
        store.close()
    try:
        effective = effective_per_day_micros(records, configured_micros=configured)
        spend_by_day_from_records(records)
    except ValueError:
        return _UNREADABLE_LEDGER_CEILING_MICROS, _CEILING_SOURCE_UNREADABLE
    if any(
        record.event_type == RESEARCH_BUDGET_CAP_SET_EVENT_TYPE for record in records
    ):
        return effective, _CEILING_SOURCE_LEDGER
    return effective, _startup_ceiling_source(args)


def _research_capped_and_logged(
    loaded: WindbreakConfig, args: argparse.Namespace
) -> WindbreakConfig:
    """Apply ``--research-per-day-micros`` and log the ceiling really in force.

    The single seam between the parsed flag and every ``run`` route, so the
    three routes cannot drift apart in either what they apply or what they
    report. ``tests/test_cli.py`` drives a real ``main(["run", ...])`` through
    it and observes the flag's *effect* -- a market halted on the budget, and
    the ledgered ``ResearchBudgetHalted`` row saying so -- rather than only its
    parse, because "every seam is tested and the composition of them is not" is
    the defect issue #483 exists to close.

    Args:
        loaded: The freshly loaded configuration.
        args: The parsed ``run`` arguments.

    Returns:
        The configuration the run should use.
    """
    config = research_capped_config(loaded, per_day_micros=args.research_per_day_micros)
    per_day_micros, source = _effective_research_ceiling(
        config, args.ledger_path, args=args
    )
    _LOGGER.info(
        "research per-day ceiling %d micros source=%s",
        per_day_micros,
        source,
    )
    return config


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
    args: argparse.Namespace,
    config: WindbreakConfig,
    *,
    dispatcher: AlertDispatcher,
) -> Callable[[int], BeatReport]:
    """Build a per-beat hook that runs one always-on PAPER tick (issue #48).

    The scheduler imports are local so the RESEARCH heartbeat path never imports
    ``windbreak.scheduler`` (nor, transitively, the paper order-submission client)
    unless PAPER is actually activated. The dependency bundle -- which opens the
    ledger database -- is built once here, so no ledger is ever created on a run
    that does not activate PAPER.

    ``market_data`` and ``live_ticker`` are passed as the pair they are: both
    derive from ``--paper-live-ticker``, so this call site cannot supply half a
    live configuration (issue #343).

    The ``dispatcher`` closes issue #444. ``build_paper_deps`` threads it into
    the read-only verification cycle, whose mismatch and unknown-jurisdiction
    alerts are the only alerting surface the loop itself has; until it was
    passed, that cycle dispatched into a hardcoded ``sinks=[]`` and an operator
    who had followed ``docs/RUNBOOK.md`` and declared a sink still learned
    nothing at 3am. It is the same object :func:`_build_beat_supervisor` holds,
    so a halt escalated by the supervisor and the breach that caused it travel
    the same configured channels.

    Args:
        args: The parsed ``run`` arguments carrying the four PAPER flags.
        config: The loaded PAPER-ceilinged configuration.
        dispatcher: The run's single composed alert root, from
            :func:`_beat_alert_root`.

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
        dispatcher=dispatcher,
    )

    def _on_beat(seq: int) -> BeatReport:
        """Run one PAPER tick and report what it found (issue #447).

        Args:
            seq: The 1-based beat sequence number stamped on the tick.

        Returns:
            The tick's kernel mode and its two halt flags. The mode is read
            from ``deps.kernel.mode`` after the tick, so a verification breach
            that drove the kernel to ``HALT`` mid-tick is what the heartbeat
            reports -- not the ``RESEARCH`` constant it used to print.
        """
        outcome = run_single_tick(deps, beat=seq)
        return BeatReport(
            mode=deps.kernel.mode.name,
            halted=outcome.kernel_halted,
            research_halted=outcome.research_halted,
        )

    return _on_beat


def _resolve_on_beat(
    args: argparse.Namespace,
    config: WindbreakConfig,
    *,
    alert_root: Callable[[], AlertDispatcher],
) -> Callable[[int], BeatReport | None] | None:
    """Resolve the per-beat hook: the PAPER tick, a snapshot pass, or none.

    PAPER activation (issue #48) takes precedence when permitted and fully
    flagged; otherwise the pre-existing snapshot hook is wired when a fixture
    directory is given; otherwise there is no hook and the loop is a bare
    RESEARCH heartbeat.

    ``alert_root`` is called only on the PAPER branch, and *before*
    :func:`_build_paper_on_beat` opens the ledger database (issue #444): a sink
    that cannot be composed refuses the run while there is still nothing durable
    to abandon. The bare-heartbeat branch never calls it at all, so a run with no
    hook composes no sinks -- it has nothing to alert about, and refusing over an
    unusable channel it will never reach would be a new failure, not a fixed one.

    Args:
        args: The parsed ``run`` arguments.
        config: The loaded configuration.
        alert_root: The compose-once alert-root factory from
            :func:`_beat_alert_root`.

    Returns:
        The resolved per-beat hook, or ``None`` for a bare heartbeat.

    Raises:
        AlertSinkConfigError: If PAPER activates and a configured sink cannot be
            composed.
    """
    if _paper_activated(config, args):
        return _build_paper_on_beat(args, config, dispatcher=alert_root())
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
    latest row's mode and timestamp -- or the no-evidence
    :data:`MODE_UNKNOWN` / no-heartbeat status when the history is empty. With
    ``None`` it always yields that default. The dashboard/ledger imports are
    local so the RESEARCH heartbeat path never imports them.

    The latest row is only as honest as what writes it, which is why
    :class:`LedgerModeWriter` exists: ``run_single_tick`` stamps its
    ``ModeHeartbeat`` before the stages that can raise, so without the
    supervisor's own row a failing beat would keep re-stamping a healthy
    ``PAPER`` here and this source would report health through an outage.

    Args:
        ledger_path: Path to the SQLite ledger database, or ``None`` to serve
            the static :data:`MODE_UNKNOWN` / no-heartbeat default.

    Returns:
        A zero-arg callable suitable for
        :func:`windbreak.dashboard.app.create_server`'s ``status_source``.
    """
    from windbreak.dashboard.app import DashboardStatus
    from windbreak.ledger.rebuild import mode_history_read_model
    from windbreak.ledger.store import SqliteLedgerStore

    def _default_source() -> DashboardStatus:
        """Report the no-evidence status: an unknown mode, no heartbeat."""
        return DashboardStatus(mode=MODE_UNKNOWN, last_heartbeat=None)

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
        The process exit code (0 on success, 1 on a fatal config or alert-sink
        error).
    """
    loaded = _load_and_ledger_config(args)
    if loaded is None:
        return 1
    config = _research_capped_and_logged(loaded, args)
    # Handlers first, and before `_resolve_on_beat`: the PAPER hook's bundle is
    # built eagerly there -- it opens the SQLite ledger, loads the books, and
    # builds the connector -- and a SIGTERM arriving inside that window would
    # otherwise hit the default disposition and kill the process outright
    # instead of unwinding through `ShutdownState`.
    state = ShutdownState()
    _install_signal_handlers(state)
    # One alert root for the whole run, composed at most once and only if some
    # per-beat seam actually needs it (issue #444). A hook is the only thing
    # there is to supervise, and composing fails closed on a misconfigured sink
    # -- at startup, rather than at the first halt, when the alert is the whole
    # point.
    alert_root = _beat_alert_root(args, config)
    try:
        on_beat = _resolve_on_beat(args, config, alert_root=alert_root)
        supervisor = (
            None
            if on_beat is None
            else _build_beat_supervisor(args, dispatcher=alert_root())
        )
    except AlertSinkConfigError as exc:
        _LOGGER.critical("FATAL: %s", exc)
        return 1
    run_loop(
        args.heartbeat_interval,
        max_beats=args.max_beats,
        stop_event=state.stop_event,
        state=state,
        component=args.process,
        on_beat=on_beat,
        supervisor=supervisor,
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

    That same connector is, since issue #480, also the kill switch's
    ``directive_sink``, via a
    :class:`~windbreak.order_gateway.cancel_all.VenueCancelAllSink`. Until then
    the one :class:`~windbreak.ledger.events.CancelAllDirective` a kill writes
    was consumed by nothing on this path *or* on the PAPER loop's, so resting
    orders survived a kill as live instructions that could still fill after the
    operator had walked away. The sink is deliberately built over the raw
    connector rather than the read-only projection the verifier gets: cancelling
    is the capability being exercised, and the read-only views elsewhere exist
    to keep *other* stages away from precisely it. Note the asymmetry with the
    PAPER loop: this process holds no order gateway and never places an order,
    so its venue surface is whatever ``--snapshot-fixture-dir`` supplies. With
    no connector at all no sink is wired, and the kill's directive row records
    ``delivery_reported: false`` -- the fail-closed unknown, never a claimed
    cancellation.

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
    from windbreak.order_gateway.cancel_all import VenueCancelAllSink
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
    # The cancel-all directive is delivered to the one venue surface this
    # process has (issue #480). With no connector there is nothing to deliver
    # to, and the kill's `CancelAllDirective` row says so with
    # `delivery_reported: false` -- an honest unknown, never a claimed cancel.
    directive_sink = (
        None
        if verification_connector is None
        else VenueCancelAllSink(verification_connector)
    )
    switch = KillSwitch.from_events(
        history,
        machine,
        writer,
        dispatcher,
        directive_sink=directive_sink,
        state_dir=state_dir,
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

    No ``fill_accounting`` feed is wired here, and that omission is load-bearing
    rather than an oversight (issues #365, #390). The feed advances the baseline
    from booked fill and order-arrival entries; nothing on this LIVE path books
    either. A feed here would therefore advance the expectation on evidence no
    process records, quietly drifting it away from the venue -- a fail-*open*
    regression in the one check that exists to catch unexplained venue movement.
    The baseline stays frozen for the life of the process until something on
    this path actually books the evidence.

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
    loaded = _load_and_ledger_config(args)
    if loaded is None:
        return 1
    config = _research_capped_and_logged(loaded, args)
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
    loaded = _load_and_ledger_config(args)
    if loaded is None:
        return 1
    config = _research_capped_and_logged(loaded, args)
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
    kill-switch wiring (issue #144). Every other choice runs the heartbeat loop
    -- which, when PAPER is activated, drives a per-beat tick whose kernel
    carries kill wiring of its own (issue #441), so the kill switch is no longer
    something only the ``riskkernel`` branch has.

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
    "ingest-resolution": _run_ingest_resolution,
    "correct-resolution": _run_correct_resolution,
    "evaluate-providers": _run_evaluate_providers,
    "set-research-budget": _run_set_research_budget,
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
