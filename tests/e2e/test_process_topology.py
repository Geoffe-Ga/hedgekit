"""The four-process topology, run as four real operating-system processes (#469).

``ARCHITECTURE.md:10`` states the product's central structural claim -- "Process
isolation is mandatory: killing one process must never kill another" -- over
"four isolated processes sharing only the ledger volume and localhost sockets".
Issue #469's finding is that every test of that topology ran in **one** process:
``tests/integration/`` spawns nothing, and its ``conftest`` builds the object
graph by hand. An object graph a test file wired itself cannot distinguish four
processes from four objects, which is the whole claim.

This module drives the shipped entry point -- ``python -m windbreak run
--process <token>``, once per token in
:data:`windbreak.main.PROCESS_CHOICES` -- as four real children, and asserts
only on evidence an operator would have: process ids, the shared ledger file,
and loopback HTTP.

What the four processes share, and what they do not
---------------------------------------------------

Exactly one thing is shared: the ledger path. Every other input is
deliberately *different* per process -- its own ``--config`` file, its own
``ops.state_dir`` inside it, its own report directory, its own log files. That
is not decoration. If two processes wrote to one path because a fixture handed
them the same default, the test would prove the fixture uniform rather than the
volume shared. Because each config file names a different ``state_dir``, each
process hashes to a *different* ``ConfigLoaded`` row, so
:func:`test_the_four_process_tokens_run_as_four_distinct_operating_system_processes`
can assert four distinct config hashes landing on one ledger -- four genuinely
differently-configured processes, one volume.

Per-process ``ops.state_dir`` is also load-bearing on its own: the shipped
default is ``~/.local/share/windbreak``, and a ``KILL`` file a developer left
there would silently kill these runs (the hazard
``tests/integration/conftest.py``'s ``state_dir`` fixture documents).

What a "decision" is here, and the limit this tier used to hit
--------------------------------------------------------------

``tests/integration/test_paper_real_approval_fill.py`` (issue #450, PR #501)
defines a decision in-process as the chain ``ForecastCreated`` ->
``SelectorDecisionRecorded`` -> ``IntentApproved`` -> ``ApprovalTokenIssued`` ->
four ``OrderTransitionLedgered`` edges. When this module was written that chain
was **not reachable from the shipped CLI in a hermetic environment**, for three
reasons this docstring recorded and issue #510 carried: the offline research
tools found nothing, so ``run_pipeline`` abstained on
``no_verified_citations``; the committed cassette held placeholders; and no
committed fixture could hold a market horizon that stayed open, because
``PaperExchange`` re-dated its books onto the replay anchor and assigned its
markets verbatim.

**PR #522 closed all three**, and the gap was filed rather than papered over
exactly so that the fix could land in ``windbreak/`` instead of inside a test
tier. ``forecast.replay_corpus`` selects a committed research/vote corpus from
configuration and ``_anchor_markets`` re-dates the calendar with the books, so
the chain now crosses a real process boundary --
:func:`test_the_whole_intent_chain_crosses_a_process_boundary` is that claim,
and :mod:`tests.hermetic_demo` names the composition it needs.

The two runs in this module are therefore deliberately different, and both are
kept:

* over the **shipped default** command line -- ``deep_walk`` and the committed
  cassette, no ``--config`` -- the forecast still abstains and
  ``intent_count`` is still ``0``. That is what an operator who configures
  nothing gets, and
  :func:`test_a_decision_one_process_recorded_is_served_by_another_over_loopback`
  pins the decision it produces.
* over the **hermetic demonstration** composition the intent chain completes.

Either way the second process -- the dashboard, started separately, with its
own credentials and its own config -- reads the decision off the shared volume
and serves it over loopback HTTP, byte-for-byte, while the writing process is
already **dead and reaped**, so no shared-memory or same-interpreter
explanation survives.

Determinism
-----------

The CLI has no clock injection (``_build_paper_on_beat`` never passes
``build_paper_deps``'s ``clock=``), so the tick reads the wall clock and the
committed fixtures' 2025 close times are long past. The books are therefore
copied and re-dated to a close ``IN_WINDOW_HORIZON_DAYS`` ahead of *now*, the
same repair ``tests/scheduler/test_paper_intent_barriers.py::_tradeable_books``
makes on a private copy -- a statement about the fixture, never a moved
threshold: ``ScreenerConfig``'s production defaults are untouched and no
``screener`` key is written to any config file here. What varies run to run is a
timestamp; what is asserted is decision content, which does not.

What running the real topology found
------------------------------------

Two product defects, both reported rather than repaired here, since this is a
test-tier change:

* **#328** -- opening the shared ledger is itself a write, so a process started
  while a sibling held the write lock died on an unhandled
  ``sqlite3.OperationalError`` before it existed. This is how this module first
  went red. **Fixed** in ``windbreak/ledger/store.py``: the open now waits for
  the lock and, if the budget runs out, fails closed with a ``FATAL`` line
  instead of a bare traceback.
  :func:`test_a_process_opening_a_write_locked_ledger_waits_and_then_starts` is
  the same reproduction inverted, asserting the process survives the wait.
* **#510** -- the abstention gap above. **Fixed** by PR #522; the chain it
  unblocked is now asserted here rather than only described.

Waiting is bounded and always on a real condition (:func:`wait_until`); there is
no sleep in this module.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import signal
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pytest

from tests.e2e.harness import (
    free_port,
    ledger_payloads,
    pid_alive,
    port_is_serving,
    read_ledger_records,
    verify_ledger_chain,
    wait_until,
)
from tests.hermetic_demo import (
    TICKER as DEMO_TICKER,
)
from tests.hermetic_demo import (
    demo_run_args,
    place_track_records,
    write_run_config,
)
from tests.paper_books import set_close_time
from windbreak.main import PROCESS_CHOICES

if TYPE_CHECKING:
    from tests.e2e.harness import ProcessLauncher, RunRoot, SpawnedProcess

pytestmark = pytest.mark.e2e

#: The books fixture ``deploy/docker-compose.yml`` names on the pipeline's own
#: command line, and the market it holds.
SHIPPED_BOOKS = Path(__file__).resolve().parents[1] / "fixtures" / "books" / "deep_walk"
TICKER = "MKT-DEEP"

#: The committed vote cassette that same shipped command line names. Its three
#: entries are placeholders; see the module docstring.
SHIPPED_CASSETTE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "forecast" / "cassettes.json"
)

#: Whole days from *now* to the re-dated fixture close: inside ``HorizonDays()``'s
#: production ``[2, 120]``-day window, which is therefore satisfied rather than
#: widened.
IN_WINDOW_HORIZON_DAYS = 30

#: A resting size, in contract-centis, above the production
#: ``min_depth_contract_centis`` floor of 10 000, on both sides of the book.
DEEP_QUANTITY_CENTIS = 100_000

#: The resting quotes the re-dated book carries, in pips.
BOOK_ASK_PIPS = 500
BOOK_BID_PIPS = 490

#: An opening balance, in micros, well above ``CapitalConfig().floor_micros``.
FUNDED_CASH_MICROS = 10_000_000_000

#: Seconds between pipeline/gateway beats. Fast enough that liveness shows up
#: promptly, slow enough that the tier is not a spin loop.
PIPELINE_BEAT_INTERVAL = "1"
GATEWAY_BEAT_INTERVAL = "0.05"

#: The Risk Kernel is float-free, so ``_kernel_heartbeat_interval`` takes the
#: ceiling of this flag: one second is its floor, not a choice.
KERNEL_BEAT_INTERVAL = "1"

#: Beat budgets. The long-running processes are bounded so a leaked child cannot
#: outlive the suite even if teardown were skipped entirely.
LONG_RUN_BEATS = "100000"
KERNEL_RUN_BEATS = "600"
ONE_BEAT = "1"

#: Deadlines. Every wait in this module is bounded by one of these.
STARTUP_TIMEOUT_SECONDS = 60.0
LIVENESS_TIMEOUT_SECONDS = 30.0
EXIT_TIMEOUT_SECONDS = 60.0
HTTP_TIMEOUT_SECONDS = 10.0

#: Kernel heartbeats that must land on the shared ledger *after* the pipeline is
#: killed before the kernel counts as still working. Two, not one: a single row
#: could have been written between the kill and the baseline read.
KERNEL_HEARTBEATS_AFTER_KILL = 2

#: Gateway heartbeats that must appear in its log after the kill, for the same
#: reason.
GATEWAY_HEARTBEATS_AFTER_KILL = 2

#: The bearer token the dashboard child mints its ``Authorization`` gate from.
#: A literal is safe: it never leaves this process tree and never reaches the
#: ledger (config is hash-chained, so a secret held there could not be redacted).
DASHBOARD_TOKEN = "issue-469-topology-token"

#: The mode the kernel process reports on its own heartbeat rows. It is
#: ``RESEARCH``, not ``PAPER``: ``--process riskkernel`` composes a live kernel
#: rather than the PAPER tick, and only the pipeline's tick is PAPER-activated.
KERNEL_HEARTBEAT_MODE = "RESEARCH"

#: The selector's reason on the hermetic path: the forecast abstained for want of
#: verified citations, so the market has no bucket or holding evidence behind it.
SELECTOR_REASON = (
    f"unprovable_exposure: no correlation bucket or holding evidence for {TICKER}"
)

#: The decision row the pipeline process records and the dashboard process serves.
DECISION_EVENT = "SelectorDecisionRecorded"

#: The row every ``run`` invocation given ``--ledger-path`` appends at startup,
#: stamped with its own ``--process`` token.
CONFIG_LOADED_EVENT = "ConfigLoaded"

#: The line ``windbreak.main._load_and_ledger_config`` logs after the config has
#: loaded cleanly and immediately *before* it opens the ledger. Waiting on it is
#: how :func:`test_a_process_opening_a_write_locked_ledger_waits_and_then_starts`
#: knows the child has reached the contended open without sleeping on a guess.
CONFIG_LOADED_LOG_LINE = "config loaded source="

#: The kernel's per-beat liveness row.
MODE_HEARTBEAT_EVENT = "ModeHeartbeat"

#: The component the Risk Kernel process stamps its own rows with.
KERNEL_COMPONENT = "riskkernel"

#: The component the PAPER tick's own stages stamp. It is not a ``--process``
#: token: inside the pipeline process the scheduler writes under its own name.
SCHEDULER_COMPONENT = "scheduler"

#: The rest of the decision chain, now that PR #522 makes it reachable.
APPROVAL_EVENT = "IntentApproved"
TOKEN_EVENT = "ApprovalTokenIssued"
TRANSITION_EVENT = "OrderTransitionLedgered"

#: The four order-state edges one approved intent walks in a beat.
ORDER_EDGES = ["APPROVE", "REQUEST_SUBMISSION", "SUBMIT", "ACK"]

#: Beats and cadence for the hermetic demonstration run. Two beats because the
#: first beat of a fresh ledger fails ``daily_loss_limit`` closed; no interval
#: because nothing has to observe it while it runs.
HERMETIC_DEMO_BEATS = "2"
HERMETIC_DEMO_INTERVAL = "0"


def _decisions_page(reason: str) -> str:
    """Render the ``/decisions`` page a single abstaining decision produces.

    Built from the renderer's own skeleton
    (:mod:`windbreak.dashboard.views.decisions`) rather than pasted, so the
    expectation names which parts are ledger-derived.

    Args:
        reason: The single reason the selector recorded.

    Returns:
        The complete HTTP response body the dashboard must serve.
    """
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8"><title>windbreak decisions</title></head>\n'
        "<body>\n"
        "<section>\n"
        "<h2>Decisions</h2>\n"
        f"<article><h3>{DECISION_EVENT}: {TICKER}</h3>"
        f"<ul><li>{reason}</li></ul></article>\n"
        "</section>\n"
        "</body>\n"
        "</html>\n"
    )


#: What ``/decisions`` renders when the ledger it reads holds no decision at all
#: -- the negative control's whole expectation.
NO_DECISIONS_PAGE = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    '<head><meta charset="utf-8"><title>windbreak decisions</title></head>\n'
    "<body>\n"
    "<section>\n"
    "<h2>Decisions</h2>\n"
    "<p>No data yet.</p>\n"
    "</section>\n"
    "</body>\n"
    "</html>\n"
)


def _tradeable_books(destination: Path) -> Path:
    """Copy the shipped books fixture and repair it for a wall-clock run.

    Three edits, each fixing something about the *fixture* rather than relaxing
    something about the product: the close time is moved inside the production
    horizon window, the book is deepened past the production depth floor, and the
    account is funded above the production capital floor. The committed fixture
    is never touched -- everything happens on this private copy.

    Args:
        destination: Directory the copy is created at; must not exist.

    Returns:
        The path to the repaired copy.
    """
    shutil.copytree(SHIPPED_BOOKS, destination)
    set_close_time(destination, days_after_recording=IN_WINDOW_HORIZON_DAYS)
    sessions_path = destination / "sessions.json"
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    for steps in sessions.values():
        for step in steps:
            step["book"]["yes_bids"] = [
                {"price": BOOK_BID_PIPS, "quantity": DEEP_QUANTITY_CENTIS}
            ]
            step["book"]["yes_asks"] = [
                {"price": BOOK_ASK_PIPS, "quantity": DEEP_QUANTITY_CENTIS}
            ]
    sessions_path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    balances_path = destination / "balances.json"
    balances = json.loads(balances_path.read_text(encoding="utf-8"))
    balances["total"] = FUNDED_CASH_MICROS
    balances["available"] = FUNDED_CASH_MICROS
    balances_path.write_text(json.dumps(balances, indent=2), encoding="utf-8")
    return destination


def _process_config(run_root: RunRoot, name: str, *, port: int | None = None) -> Path:
    """Write one process's own SPEC §16 config file.

    Every process gets a *different* ``ops.state_dir``, which is what makes the
    four ``ConfigLoaded`` hashes differ and what keeps a stray ``KILL`` file in
    the shipped default directory from reaching these runs.

    Args:
        run_root: The run root the config and its state directory live under.
        name: The process token this config belongs to.
        port: The loopback port to publish, for the dashboard only.

    Returns:
        The path the config was written to.
    """
    lines = ["ops:", f"  state_dir: {run_root.state_dir / name}"]
    if port is not None:
        lines.extend(["dashboard:", f"  port: {port}"])
    path = run_root.root / f"{name}.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _spawn_pipeline(
    launcher: ProcessLauncher,
    run_root: RunRoot,
    books: Path,
    *,
    ledger_path: Path,
    max_beats: str,
) -> SpawnedProcess:
    """Start the pipeline process with all four PAPER composition flags.

    Args:
        launcher: The launcher that owns the child's lifetime.
        run_root: The run root supplying config and report directories.
        books: The repaired books fixture directory.
        ledger_path: The ledger this process writes to.
        max_beats: The beat budget, as the CLI spells it.

    Returns:
        The started process.
    """
    report_dir = run_root.report_dir / "pipeline"
    report_dir.mkdir(parents=True, exist_ok=True)
    return launcher.spawn(
        "run",
        "--process",
        "pipeline",
        "--config",
        str(_process_config(run_root, "pipeline")),
        "--ledger-path",
        str(ledger_path),
        "--paper-books-dir",
        str(books),
        "--cassette-path",
        str(SHIPPED_CASSETTE),
        "--report-dir",
        str(report_dir),
        "--max-beats",
        max_beats,
        "--heartbeat-interval",
        PIPELINE_BEAT_INTERVAL,
        name="pipeline",
    )


def _spawn_riskkernel(
    launcher: ProcessLauncher, run_root: RunRoot, *, ledger_path: Path
) -> SpawnedProcess:
    """Start the Risk Kernel process against the shared ledger.

    Args:
        launcher: The launcher that owns the child's lifetime.
        run_root: The run root supplying this process's config.
        ledger_path: The ledger this process writes its heartbeats to.

    Returns:
        The started process.
    """
    return launcher.spawn(
        "run",
        "--process",
        "riskkernel",
        "--config",
        str(_process_config(run_root, "riskkernel")),
        "--ledger-path",
        str(ledger_path),
        "--max-beats",
        KERNEL_RUN_BEATS,
        "--heartbeat-interval",
        KERNEL_BEAT_INTERVAL,
        name="riskkernel",
    )


def _spawn_order_gateway(
    launcher: ProcessLauncher, run_root: RunRoot, *, ledger_path: Path
) -> SpawnedProcess:
    """Start the Order Gateway process against the shared ledger.

    Args:
        launcher: The launcher that owns the child's lifetime.
        run_root: The run root supplying this process's config.
        ledger_path: The ledger this process records its config load on.

    Returns:
        The started process.
    """
    return launcher.spawn(
        "run",
        "--process",
        "order_gateway",
        "--config",
        str(_process_config(run_root, "order_gateway")),
        "--ledger-path",
        str(ledger_path),
        "--max-beats",
        LONG_RUN_BEATS,
        "--heartbeat-interval",
        GATEWAY_BEAT_INTERVAL,
        name="order_gateway",
    )


def _spawn_dashboard(
    launcher: ProcessLauncher, run_root: RunRoot, *, ledger_path: Path, port: int
) -> SpawnedProcess:
    """Start the dashboard process, reading a ledger and serving loopback HTTP.

    The bearer token reaches the child only through its environment, which is
    the one channel the product accepts it on (``DASHBOARD_AUTH_ENV_VAR``): a
    token in config would be hash-chained onto the ledger and unredactable.

    Args:
        launcher: The launcher that owns the child's lifetime.
        run_root: The run root supplying this process's config.
        ledger_path: The ledger this process reads its read models from.
        port: The loopback port its config publishes.

    Returns:
        The started process.
    """
    return launcher.spawn(
        "run",
        "--process",
        "dashboard",
        "--config",
        str(_process_config(run_root, "dashboard", port=port)),
        "--ledger-path",
        str(ledger_path),
        name="dashboard",
        env=dict(os.environ, WINDBREAK_DASHBOARD_TOKEN=DASHBOARD_TOKEN),
    )


def _process_log(spawned: SpawnedProcess) -> str:
    """Return everything a child has written to either captured stream.

    Args:
        spawned: The child whose streams are read.

    Returns:
        Its stdout followed by its stderr.
    """
    return spawned.stdout_text() + spawned.stderr_text()


def _get(port: int, path: str, *, token: str | None) -> tuple[int, str]:
    """Issue one bounded loopback GET and return its status and body.

    :mod:`http.client` rather than :mod:`urllib.request` on purpose: it consults
    no proxy environment, so a developer's ``http_proxy`` cannot silently divert
    a request that is supposed to prove a loopback socket works.

    Args:
        port: The loopback port to connect to.
        path: The request path.
        token: The bearer token to present, or ``None`` to present none.

    Returns:
        The response status code and its decoded body.
    """
    connection = http.client.HTTPConnection(
        "127.0.0.1", port, timeout=HTTP_TIMEOUT_SECONDS
    )
    try:
        headers = {} if token is None else {"Authorization": f"Bearer {token}"}
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


def _await_dashboard(spawned: SpawnedProcess, port: int) -> None:
    """Block until the dashboard child is accepting loopback connections.

    A child that dies while being waited for fails immediately, carrying its own
    log, rather than burning the whole deadline and reporting only a timeout.

    Args:
        spawned: The dashboard child.
        port: The loopback port it was configured to bind.

    Raises:
        AssertionError: If the child exits before it serves.
    """

    def _serving() -> bool:
        """Report whether the port answers, failing fast on a dead child.

        Returns:
            ``True`` once the port accepts a connection.

        Raises:
            AssertionError: If the child has already exited.
        """
        if not spawned.is_running():
            message = f"dashboard exited before serving:\n{_process_log(spawned)}"
            raise AssertionError(message)
        return port_is_serving(port)

    wait_until(
        _serving,
        timeout=STARTUP_TIMEOUT_SECONDS,
        description=f"the dashboard process to serve on 127.0.0.1:{port}",
    )


def _payloads(
    ledger_path: Path, event_type: str, component: str | None = None
) -> list[dict[str, object]]:
    """Read one component's rows of one event type off a ledger another wrote.

    A thin positional-argument spelling of
    :func:`~tests.e2e.harness.ledger_payloads`, which is where the WAL-aware
    read lives so every module in the tier gets the same one.

    Args:
        ledger_path: The ledger to read.
        event_type: The event type wanted.
        component: The component whose rows are wanted, or ``None`` for all.

    Returns:
        Each matching row's ``data`` payload, in chain order.
    """
    return ledger_payloads(ledger_path, event_type, component=component)


def _kernel_heartbeat_count(ledger_path: Path) -> int:
    """Count the Risk Kernel process's own heartbeat rows on a ledger.

    Args:
        ledger_path: The shared ledger to read.

    Returns:
        How many ``ModeHeartbeat`` rows the kernel component has written.
    """
    return len(_payloads(ledger_path, MODE_HEARTBEAT_EVENT, KERNEL_COMPONENT))


def _gateway_heartbeat_count(spawned: SpawnedProcess) -> int:
    """Count the heartbeats the Order Gateway process has logged.

    The gateway is not PAPER-activated, so it records nothing but its config load
    on the ledger; its log is the liveness signal it actually has.

    Args:
        spawned: The gateway child.

    Returns:
        How many heartbeat lines its captured streams hold.
    """
    return _process_log(spawned).count("heartbeat seq=")


class LiveProcess(Protocol):
    """The narrow view :func:`assert_four_distinct_live_processes` takes.

    A protocol rather than :class:`~tests.e2e.harness.SpawnedProcess` itself so
    the guard's own ``os.getpid()`` branch can be falsified without starting a
    process that could not possibly have the test runner's process id.

    Attributes:
        pid: The operating-system process id.
        name: The short label the process was started under.
    """

    @property
    def pid(self) -> int:
        """Return the operating-system process id."""

    @property
    def name(self) -> str:
        """Return the short label the process was started under."""

    def is_running(self) -> bool:
        """Report whether the process has not yet exited."""


@dataclass(frozen=True)
class _StandInProcess:
    """A process id and name with no process behind them.

    Used *only* to falsify :func:`assert_four_distinct_live_processes`'s
    "not the test runner" branch, which no real child could ever trip. It stands
    in for a guard input, never for a windbreak component: nothing in this module
    asserts product behaviour through it.

    Attributes:
        pid: The process id to report.
        name: The label to report.
    """

    pid: int
    name: str

    def is_running(self) -> bool:
        """Report the stand-in as running.

        Returns:
            ``True``, always.
        """
        return True


def assert_four_distinct_live_processes(spawned: tuple[LiveProcess, ...]) -> None:
    """Assert four *different* live operating-system processes were started.

    Factored out because it is the claim the whole module rests on, and because
    a claim that cannot fail is worth nothing:
    :func:`test_the_topology_check_rejects_a_collapsed_topology`
    hands this same function collapsed topologies and requires it to raise.

    Args:
        spawned: The started processes.

    Raises:
        AssertionError: If they are not four distinct, live, non-test processes.
    """
    assert len(spawned) == len(PROCESS_CHOICES)
    pids = {process.pid for process in spawned}
    assert len(pids) == len(PROCESS_CHOICES), (
        f"expected {len(PROCESS_CHOICES)} distinct process ids, got {sorted(pids)}"
    )
    assert os.getpid() not in pids, (
        "a process id equal to the test runner's is an object graph, not a process"
    )
    assert [process.pid for process in spawned if pid_alive(process.pid)] == [
        process.pid for process in spawned
    ]
    assert [process.name for process in spawned if process.is_running()] == [
        process.name for process in spawned
    ]


def _start_topology(
    launcher: ProcessLauncher, run_root: RunRoot, books: Path, port: int
) -> tuple[SpawnedProcess, ...]:
    """Start all four processes against one shared ledger, one at a time.

    **The start order and the waiting between are not style.** Opening the ledger
    is itself a write -- ``SqliteLedgerStore.__init__`` issues ``PRAGMA
    journal_mode=WAL`` and a ``CREATE TABLE`` -- and until issue #328 was fixed a
    process that opened the shared ledger while a sibling held the write lock
    died at once with an unhandled ``sqlite3.OperationalError``. That was a
    defect of the shipped product, not of this test, and it is now repaired in
    ``windbreak/ledger/store.py``: the open waits for the lock.

    The sequencing therefore no longer *has* to be here, and is kept in this
    change only because retiring it is a question about this tier rather than
    about the ledger -- it also makes each process's startup individually
    observable, which the assertions below rely on.
    :func:`test_a_process_opening_a_write_locked_ledger_waits_and_then_starts`
    is what now holds the ledger's side of the contract.

    Args:
        launcher: The launcher that reaps every child.
        run_root: The run root supplying the shared ledger and per-process config.
        books: The repaired books fixture the pipeline replays.
        port: The loopback port the dashboard publishes.

    Returns:
        The four processes, in :data:`windbreak.main.PROCESS_CHOICES` order.
    """
    ledger_path = run_root.ledger_path
    dashboard = _spawn_dashboard(launcher, run_root, ledger_path=ledger_path, port=port)
    _await_component_on_the_ledger(ledger_path, "dashboard")
    order_gateway = _spawn_order_gateway(launcher, run_root, ledger_path=ledger_path)
    _await_component_on_the_ledger(ledger_path, "order_gateway")
    riskkernel = _spawn_riskkernel(launcher, run_root, ledger_path=ledger_path)
    _await_component_on_the_ledger(ledger_path, "riskkernel")
    pipeline = _spawn_pipeline(
        launcher, run_root, books, ledger_path=ledger_path, max_beats=LONG_RUN_BEATS
    )
    _await_component_on_the_ledger(ledger_path, "pipeline")
    return (pipeline, riskkernel, order_gateway, dashboard)


def _await_component_on_the_ledger(ledger_path: Path, component: str) -> None:
    """Block until one process has recorded its own config load on the ledger.

    The startup handshake: a ``ConfigLoaded`` row stamped with this process's
    ``--process`` token is the first durable evidence the process exists, opened
    the shared volume, and got its own configuration.

    Args:
        ledger_path: The shared ledger.
        component: The ``--process`` token to wait for.
    """
    wait_until(
        lambda: len(_payloads(ledger_path, CONFIG_LOADED_EVENT, component)) == 1,
        timeout=STARTUP_TIMEOUT_SECONDS,
        description=f"the {component} process to record its own config load",
    )


def test_the_four_process_tokens_run_as_four_distinct_operating_system_processes(
    launcher: ProcessLauncher, run_root: RunRoot, tmp_path: Path
) -> None:
    """Four ``--process`` tokens, four pids, four configs, one verified ledger.

    The topology assertion is made against process ids, not collaborators, and
    the shared-volume assertion is made against four *different* config hashes:
    each process was handed a config naming its own ``ops.state_dir``, so four
    matching hashes would mean the fixture had made them identical rather than
    that they shared anything.

    This is also issue #469's concurrent-writer criterion: the chain is verified
    while all four are still running and writing.

    Args:
        launcher: The launcher that reaps every child.
        run_root: This test's isolated run root.
        tmp_path: pytest's per-test temporary directory.
    """
    books = _tradeable_books(tmp_path / "books")
    port = free_port()

    spawned = _start_topology(launcher, run_root, books, port)

    assert_four_distinct_live_processes(spawned)
    assert tuple(process.name for process in spawned) == PROCESS_CHOICES
    records = read_ledger_records(run_root.ledger_path)
    config_rows = [
        record for record in records if record.event_type == CONFIG_LOADED_EVENT
    ]
    assert {record.component for record in config_rows} == set(PROCESS_CHOICES)
    hashes = {
        json.loads(record.payload_json)["data"]["config_hash"] for record in config_rows
    }
    assert len(hashes) == len(PROCESS_CHOICES), (
        "four processes sharing one config hash were handed one configuration; "
        "this test would then prove the fixture uniform, not the volume shared"
    )
    verify_ledger_chain(run_root.ledger_path)


def test_the_topology_check_rejects_a_collapsed_topology(
    launcher: ProcessLauncher, run_root: RunRoot
) -> None:
    """Four names over one process, and four ids that are the test's, both fail.

    The collapsed topology as a standing test rather than something someone did
    once. Without it :func:`assert_four_distinct_live_processes` could not
    distinguish four processes from four references to one -- exactly the defect
    issue #469 describes in ``tests/integration/``, whose ``conftest`` builds one
    object graph and calls it a topology.

    The first collapse is real: one genuinely started child, listed four times.
    The second is the in-process case the guard exists for, which no started
    child could reproduce, since a child never has its parent's process id.

    Args:
        launcher: The launcher that reaps the single child.
        run_root: This test's isolated run root.
    """
    only = _spawn_order_gateway(launcher, run_root, ledger_path=run_root.ledger_path)

    with pytest.raises(AssertionError) as one_process:
        assert_four_distinct_live_processes((only, only, only, only))
    in_process = tuple(
        _StandInProcess(pid=os.getpid() + offset, name=name)
        for offset, name in enumerate(PROCESS_CHOICES)
    )
    with pytest.raises(AssertionError) as same_interpreter:
        assert_four_distinct_live_processes(in_process)

    assert f"expected {len(PROCESS_CHOICES)} distinct process ids" in str(
        one_process.value
    )
    assert str(same_interpreter.value).startswith(
        "a process id equal to the test runner's is an object graph, not a process"
    )
    assert only.is_running()


def test_a_decision_one_process_recorded_is_served_by_another_over_loopback(
    launcher: ProcessLauncher, run_root: RunRoot, tmp_path: Path
) -> None:
    """A decision outlives the process that made it and reaches another over HTTP.

    The pipeline runs exactly one beat and exits, so the decision set is exactly
    one row and the served page is pinnable byte-for-byte. By the time the page
    is fetched the writing process is gone -- asserted, not assumed -- so no
    shared object, shared interpreter or shared memory can explain what the
    dashboard is serving. The only thing the two processes ever shared is the
    ledger path.

    The expected page is built from the reason the *ledger* carries, so the HTML
    and the row cannot drift apart, and the reason itself is pinned as a literal
    so the two cannot drift together.

    Args:
        launcher: The launcher that reaps every child.
        run_root: This test's isolated run root.
        tmp_path: pytest's per-test temporary directory.
    """
    books = _tradeable_books(tmp_path / "books")
    port = free_port()
    pipeline = _spawn_pipeline(
        launcher, run_root, books, ledger_path=run_root.ledger_path, max_beats=ONE_BEAT
    )

    assert pipeline.wait(timeout=EXIT_TIMEOUT_SECONDS) == 0
    dashboard = _spawn_dashboard(
        launcher, run_root, ledger_path=run_root.ledger_path, port=port
    )
    _await_dashboard(dashboard, port)

    assert pipeline.pid != dashboard.pid
    assert not pipeline.is_running()
    assert dashboard.is_running()
    decisions = _payloads(run_root.ledger_path, DECISION_EVENT, SCHEDULER_COMPONENT)
    assert len(decisions) == 1
    assert decisions[0]["market_ticker"] == TICKER
    assert decisions[0]["intent_count"] == 0
    assert decisions[0]["reasons"] == [SELECTOR_REASON]
    status, body = _get(port, "/decisions", token=DASHBOARD_TOKEN)
    assert status == 200
    assert body == _decisions_page(SELECTOR_REASON)
    unauthorized_status, _ = _get(port, "/decisions", token=None)
    assert unauthorized_status == 401


def test_the_whole_intent_chain_crosses_a_process_boundary(
    launcher: ProcessLauncher, run_root: RunRoot
) -> None:
    """The chain #510 could not reach now completes in a child process (#522).

    Issue #522's own third acceptance criterion, deferred to #473's lane: this
    module's docstring described the intent path as unreachable from the shipped
    CLI, and that stopped being true when ``forecast.replay_corpus`` and
    ``_anchor_markets`` landed. Describing a limit that no longer exists is
    worse than describing none, so the claim is asserted here instead.

    Nothing about the topology changes. The pipeline is the same shipped entry
    point in its own operating-system process with its own configuration; the
    only difference from the run above is which committed fixtures it is pointed
    at. Two beats, because the risk kernel fails ``daily_loss_limit`` closed on
    the first beat of a fresh ledger -- ``equity_start_of_day`` does not exist
    until that beat's own ``EquitySampled`` row does -- so an approval is only
    observable on the second.

    The whole chain is read back off the shared volume **after the writing
    process is dead and reaped**, asserted rather than assumed, so no
    same-interpreter explanation survives: one ``IntentApproved`` with no
    dissenting reasons, its token, and the four order-state edges that carry it
    to ``ACK``. Each is asserted as an exact value or an exact list; an
    ``intent_count >= 1`` would pass for a run that emitted an intent and then
    lost it.

    Args:
        launcher: The launcher that reaps every child.
        run_root: This test's isolated run root.
    """
    report_dir = run_root.report_dir / "hermetic"
    place_track_records(report_dir)
    config = write_run_config(
        run_root.root / "hermetic.yaml", state_dir=run_root.state_dir / "hermetic"
    )
    pipeline = launcher.spawn(
        *demo_run_args(
            config=config,
            ledger_path=run_root.ledger_path,
            report_dir=report_dir,
            max_beats=HERMETIC_DEMO_BEATS,
            heartbeat_interval=HERMETIC_DEMO_INTERVAL,
        ),
        name="hermetic-pipeline",
    )

    assert pipeline.wait(timeout=EXIT_TIMEOUT_SECONDS) == 0

    assert not pipeline.is_running()
    assert not pid_alive(pipeline.pid)
    decisions = _payloads(run_root.ledger_path, DECISION_EVENT, SCHEDULER_COMPONENT)
    assert [decision["market_ticker"] for decision in decisions] == [DEMO_TICKER] * int(
        HERMETIC_DEMO_BEATS
    )
    assert [decision["intent_count"] for decision in decisions] == [1] * int(
        HERMETIC_DEMO_BEATS
    )
    approvals = _payloads(run_root.ledger_path, APPROVAL_EVENT)
    assert len(approvals) == 1
    assert approvals[0]["reasons"] == []
    assert [
        token["intent_id"] for token in _payloads(run_root.ledger_path, TOKEN_EVENT)
    ] == [approvals[0]["intent_id"]]
    assert [
        edge["event"] for edge in _payloads(run_root.ledger_path, TRANSITION_EVENT)
    ] == ORDER_EDGES
    verify_ledger_chain(run_root.ledger_path)


def test_a_dashboard_on_its_own_ledger_cannot_see_that_decision(
    launcher: ProcessLauncher, run_root: RunRoot, tmp_path: Path
) -> None:
    """The shared volume is what carries the decision, not the fixture's shape.

    The same two processes, started the same way, except the dashboard is pointed
    at a ledger of its own. The pipeline's decision is on its ledger exactly as
    before -- asserted, so this cannot pass by the pipeline having decided
    nothing -- and the dashboard serves the empty page. Without this, the test
    above could not tell "the volume is shared" from "both fixtures happened to
    name the same file".

    Args:
        launcher: The launcher that reaps every child.
        run_root: This test's isolated run root.
        tmp_path: pytest's per-test temporary directory.
    """
    books = _tradeable_books(tmp_path / "books")
    private_ledger = run_root.root / "dashboard-only.db"
    port = free_port()
    pipeline = _spawn_pipeline(
        launcher, run_root, books, ledger_path=run_root.ledger_path, max_beats=ONE_BEAT
    )

    assert pipeline.wait(timeout=EXIT_TIMEOUT_SECONDS) == 0
    dashboard = _spawn_dashboard(
        launcher, run_root, ledger_path=private_ledger, port=port
    )
    _await_dashboard(dashboard, port)

    assert private_ledger != run_root.ledger_path
    assert (
        len(_payloads(run_root.ledger_path, DECISION_EVENT, SCHEDULER_COMPONENT)) == 1
    )
    assert _payloads(private_ledger, DECISION_EVENT, SCHEDULER_COMPONENT) == []
    status, body = _get(port, "/decisions", token=DASHBOARD_TOKEN)
    assert status == 200
    assert body == NO_DECISIONS_PAGE


def test_killing_the_pipeline_leaves_the_other_three_alive_and_still_working(
    launcher: ProcessLauncher, run_root: RunRoot, tmp_path: Path
) -> None:
    """``ARCHITECTURE.md:10``'s claim, tested: one dies, three keep working.

    ``SIGKILL`` to the pipeline's whole process group, so nothing graceful can be
    mistaken for survival. The survivors are then required to do *work*, not
    merely hold a process id -- a wedged process is alive and useless:

    * the Risk Kernel appends two further heartbeat rows to the shared ledger,
      and the second one's ``beat`` counter is checked exactly;
    * the Order Gateway logs two further heartbeats of its own;
    * the dashboard answers a fresh loopback request, which re-opens the shared
      ledger and re-verifies its hash chain on the way.

    The chain is verified once more at the end, over a ledger three live
    processes are still writing to and a fourth died mid-run.

    Args:
        launcher: The launcher that reaps every child.
        run_root: This test's isolated run root.
        tmp_path: pytest's per-test temporary directory.
    """
    books = _tradeable_books(tmp_path / "books")
    port = free_port()
    pipeline, riskkernel, order_gateway, dashboard = _start_topology(
        launcher, run_root, books, port
    )
    _await_dashboard(dashboard, port)
    wait_until(
        lambda: _kernel_heartbeat_count(run_root.ledger_path) > 0,
        timeout=LIVENESS_TIMEOUT_SECONDS,
        description="the Risk Kernel process to record its first heartbeat",
    )

    kernel_before = _kernel_heartbeat_count(run_root.ledger_path)
    gateway_before = _gateway_heartbeat_count(order_gateway)
    os.killpg(os.getpgid(pipeline.pid), signal.SIGKILL)
    wait_until(
        lambda: not pipeline.is_running(),
        timeout=LIVENESS_TIMEOUT_SECONDS,
        description="the killed pipeline process to be reaped",
    )

    assert not pid_alive(pipeline.pid)
    assert kernel_before != 0, "no kernel heartbeat before the kill: nothing to compare"
    assert gateway_before != 0, (
        "no gateway heartbeat before the kill: nothing to compare"
    )
    kernel_target = kernel_before + KERNEL_HEARTBEATS_AFTER_KILL
    wait_until(
        lambda: _kernel_heartbeat_count(run_root.ledger_path) >= kernel_target,
        timeout=LIVENESS_TIMEOUT_SECONDS,
        description="the Risk Kernel process to keep beating after the kill",
    )
    wait_until(
        lambda: (
            _gateway_heartbeat_count(order_gateway)
            >= gateway_before + GATEWAY_HEARTBEATS_AFTER_KILL
        ),
        timeout=LIVENESS_TIMEOUT_SECONDS,
        description="the Order Gateway process to keep beating after the kill",
    )
    heartbeats = _payloads(run_root.ledger_path, MODE_HEARTBEAT_EVENT, KERNEL_COMPONENT)
    assert heartbeats[kernel_target - 1] == {
        "beat": kernel_target,
        "mode": KERNEL_HEARTBEAT_MODE,
    }
    assert riskkernel.is_running()
    assert order_gateway.is_running()
    assert dashboard.is_running()
    status, _ = _get(port, "/decisions", token=DASHBOARD_TOKEN)
    assert status == 200
    verify_ledger_chain(run_root.ledger_path)


def test_a_process_opening_a_write_locked_ledger_waits_and_then_starts(
    launcher: ProcessLauncher, run_root: RunRoot
) -> None:
    """The defect this tier found, now from the other side (#328).

    Opening the ledger is a *write*: ``SqliteLedgerStore.__init__`` issues
    ``PRAGMA journal_mode=WAL``, a ``CREATE TABLE`` and a ``CREATE INDEX``. This
    test used to pin what that cost -- a ``windbreak run`` given
    ``--ledger-path`` while any sibling held the write lock died on an unhandled
    ``sqlite3.OperationalError``, exit 1, with no ``FATAL`` line, before the
    process existed at all. It was pinned rather than worked around because the
    repair belonged in ``windbreak/ledger/store.py``, not in a test tier.

    It has been repaired, so the pin is inverted rather than removed: the same
    reproduction, asserting the behaviour that replaced it. The store now waits
    ``LEDGER_BUSY_TIMEOUT_MILLIS`` for the lock, so the started process is
    **still alive** while the lock is held -- asserted before the release, which
    is the assertion the old defect could not have survived -- and lands its own
    ``ConfigLoaded`` row once the lock frees.

    That the failure is now loud rather than silent is proved at the unit seam
    (``tests/ledger/test_ledger_open_contention.py``), where the budget can be
    taken to zero without a wall-clock wait. Here the point is the process: it
    does not die.

    It matters far beyond the tests: ``deploy/docker-compose.yml`` starts all
    four services at once under ``restart: on-failure``, so on a busy ledger the
    shipped stack used to crash-loop -- three of "four isolated processes
    sharing only the ledger volume" taken down by the fourth merely *writing*,
    which is the same ``ARCHITECTURE.md:10`` claim this module tests, failing
    from the other direction.

    The lock is held by an ordinary second connection -- exactly what a sibling
    process is -- so nothing about this reproduction is special to a test.

    Args:
        launcher: The launcher that reaps the child.
        run_root: This test's isolated run root.
    """
    ledger_path = run_root.ledger_path
    holder = sqlite3.connect(ledger_path, isolation_level=None)
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("BEGIN IMMEDIATE")
    try:
        gateway = _spawn_order_gateway(launcher, run_root, ledger_path=ledger_path)
        wait_until(
            lambda: CONFIG_LOADED_LOG_LINE in _process_log(gateway),
            timeout=STARTUP_TIMEOUT_SECONDS,
            description="the Order Gateway process to reach the ledger open",
        )
        # Nothing may read the ledger here: this process would take the same
        # contended open the child is sitting in, and wait out the same budget.
        assert gateway.is_running()
    finally:
        holder.rollback()
        holder.close()

    _await_component_on_the_ledger(ledger_path, "order_gateway")
    assert gateway.is_running()
    assert "FATAL" not in _process_log(gateway)
    verify_ledger_chain(ledger_path)
