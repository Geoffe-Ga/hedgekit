"""Failing-first tests binding the dashboard HTTP server into
`windbreak run --process dashboard` (issue #79, RED).

Today `run --process dashboard` just runs the idle RESEARCH heartbeat loop
and exits 0 -- identical to every other `--process` choice. This module pins
the corrected contract: the dashboard process boots
`windbreak.dashboard.app.create_server` on a configured loopback port,
authenticated by a bearer token minted *only* from the
`WINDBREAK_DASHBOARD_TOKEN` environment variable (never from config, since
config is ledgered and a secret there would leak).

None of the following exist yet:

- `windbreak.config.schema.DashboardConfig` (and the `dashboard` field on
  `WindbreakConfig` that carries it).
- `windbreak.main.DASHBOARD_AUTH_ENV_VAR`, `_load_dashboard_token`,
  `_build_dashboard_server`, and `_serve_until_shutdown`.

so the import block below fails the whole module at collection with
`ImportError` -- the expected Gate 1 RED state, mirroring
`tests/dashboard/test_app.py`'s identical whole-module-import-failure
precedent for issue #15. Once the implementation specialist adds those
symbols, the remaining tests pin (per test) a `ValueError`/`TypeError`,
exit-code, or HTTP-response assertion until the dashboard process is fully
wired through `main()`.

The five config-schema/loader assertions and the `mode_history_read_model`
projection test live alongside their existing siblings in
`tests/config/test_schema.py`, `tests/config/test_loader.py`, and
`tests/ledger/test_ledger_rebuild.py` (each via a *local* import, so only
that one new test fails at import time -- the rest of those modules keep
collecting and passing). `tests/test_process_flag.py`'s
`test_main_run_with_process_flag_stamps_matching_component_in_json` is also
narrowed to the three heartbeat processes, since `dashboard` no longer shares
that heartbeat/shutdown-log contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
from typing import TYPE_CHECKING

import pytest
import yaml

from tests.dashboard.test_app import TEST_TOKEN, _bearer, _get
from windbreak.config.schema import DashboardConfig, WindbreakConfig
from windbreak.dashboard.app import DashboardStatus, create_server
from windbreak.ledger.events import EquitySampled, ModeHeartbeat
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.main import (
    DASHBOARD_AUTH_ENV_VAR,
    ShutdownState,
    _build_dashboard_server,
    _load_dashboard_token,
    _serve_until_shutdown,
    main,
)

if TYPE_CHECKING:
    import http.server
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.timeout(20)

#: How the child subprocess's stderr is polled for its "serving on" line:
#: a fixed attempt budget plus a wall-clock deadline, never an unbounded
#: wait (house rule).
_STDERR_POLL_ATTEMPTS = 200
_STDERR_POLL_SELECT_TIMEOUT_SECONDS = 0.1
_STDERR_POLL_DEADLINE_SECONDS = 15.0

#: Matches the "dashboard serving on 127.0.0.1:<port>" log line `_run_dashboard`
#: is expected to emit, capturing the OS-assigned port.
_DASHBOARD_PORT_PATTERN = re.compile(r"dashboard serving on 127\.0\.0\.1:(\d+)")


def _free_tcp_port() -> int:
    """Return a currently-unused loopback TCP port.

    Binds an OS-assigned ephemeral port, reads it back, then releases the
    socket. A small, accepted TOCTOU race (another process could grab the
    port before the caller rebinds it) -- fine for this test's
    exact-port-plumbing assertion.

    Returns:
        A TCP port currently free on 127.0.0.1.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _await_dashboard_port(proc: subprocess.Popen[str]) -> int:
    """Poll a child process's stderr for its announced dashboard port.

    Bounded by both a fixed attempt budget and a wall-clock deadline -- never
    an unbounded wait -- so a hung or silent child fails this test loudly,
    with the captured stderr attached, instead of blocking the suite.

    Args:
        proc: The subprocess whose stderr announces its bound port via a
            `"dashboard serving on 127.0.0.1:<port>"` log line.

    Returns:
        The announced TCP port.

    Raises:
        AssertionError: If neither the attempt budget nor the deadline
            yields the port line.
    """
    assert proc.stderr is not None
    deadline = time.monotonic() + _STDERR_POLL_DEADLINE_SECONDS
    collected: list[str] = []
    for _attempt in range(_STDERR_POLL_ATTEMPTS):
        if time.monotonic() >= deadline:
            break
        ready, _write_ready, _err_ready = select.select(
            [proc.stderr], [], [], _STDERR_POLL_SELECT_TIMEOUT_SECONDS
        )
        if not ready:
            continue
        line = proc.stderr.readline()
        if not line:
            continue
        collected.append(line)
        match = _DASHBOARD_PORT_PATTERN.search(line)
        if match:
            return int(match.group(1))
    raise AssertionError(
        "dashboard did not report its serving port before the deadline; "
        f"captured stderr:\n{''.join(collected)}"
    )


# --- item 1: `run --process dashboard` routing (no/blank token) ------------


def test_main_run_process_dashboard_without_token_env_var_exits_1_with_fatal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--process dashboard` with no token env var fails closed at exit 1.

    Proves the routing change itself: every `--process` choice used to run
    the idle heartbeat loop and exit 0 (issue #15); `dashboard` must now
    diverge and refuse to serve without its bearer token, naming the missing
    variable in a `FATAL` log line.
    """
    monkeypatch.delenv(DASHBOARD_AUTH_ENV_VAR, raising=False)

    exit_code = main(
        [
            "run",
            "--process",
            "dashboard",
            "--heartbeat-interval",
            "0",
            "--max-beats",
            "1",
        ]
    )

    captured = capsys.readouterr()
    messages = [
        str(json.loads(line).get("msg", ""))
        for line in captured.err.splitlines()
        if line
    ]
    assert exit_code == 1
    assert any(
        "FATAL:" in message and DASHBOARD_AUTH_ENV_VAR in message
        for message in messages
    )


def test_main_run_process_dashboard_with_blank_token_env_var_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A blank (empty-string) token env var is rejected exactly like a missing one."""
    monkeypatch.setenv(DASHBOARD_AUTH_ENV_VAR, "")

    exit_code = main(
        [
            "run",
            "--process",
            "dashboard",
            "--heartbeat-interval",
            "0",
            "--max-beats",
            "1",
        ]
    )

    captured = capsys.readouterr()
    messages = [
        str(json.loads(line).get("msg", ""))
        for line in captured.err.splitlines()
        if line
    ]
    assert exit_code == 1
    assert any(
        "FATAL:" in message and DASHBOARD_AUTH_ENV_VAR in message
        for message in messages
    )


# --- item 2: `_load_dashboard_token` -----------------------------------------


def test_load_dashboard_token_returns_the_configured_token() -> None:
    """An injected mapping carrying the token returns it verbatim."""
    token = _load_dashboard_token({DASHBOARD_AUTH_ENV_VAR: "a-real-token"})

    assert token == "a-real-token"


def test_load_dashboard_token_missing_var_raises_value_error_naming_it() -> None:
    """A mapping without the var raises `ValueError` naming it."""
    with pytest.raises(ValueError, match=DASHBOARD_AUTH_ENV_VAR):
        _load_dashboard_token({})


def test_load_dashboard_token_blank_var_raises_value_error_naming_it() -> None:
    """A blank (empty-string) value is rejected exactly like a missing one."""
    with pytest.raises(ValueError, match=DASHBOARD_AUTH_ENV_VAR):
        _load_dashboard_token({DASHBOARD_AUTH_ENV_VAR: ""})


def test_load_dashboard_token_defaults_to_os_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no `environ` argument, the real `os.environ` is read."""
    monkeypatch.setenv(DASHBOARD_AUTH_ENV_VAR, "from-os-environ")

    assert _load_dashboard_token() == "from-os-environ"


# --- item 3: `_build_dashboard_server` --------------------------------------


@pytest.fixture
def dashboard_process_server() -> Iterator[
    tuple[http.server.ThreadingHTTPServer, tuple[str, int]]
]:
    """Build and serve a dashboard server via `_build_dashboard_server`.

    No `--ledger-path` is supplied, so the status source is the documented
    "no ledger" default (`DashboardStatus("UNKNOWN", None)` since issue #447 --
    with no ledger there is no evidence of any mode, and absent evidence must
    never render as the healthy RESEARCH loop it used to claim).

    Yields:
        The built server and its bound `(host, port)` address; shuts the
        server down and joins its thread on teardown so no test leaks a
        listening socket or background thread into the next test.
    """
    config = WindbreakConfig(dashboard=DashboardConfig(port=0))
    args = argparse.Namespace(ledger_path=None)

    server = _build_dashboard_server(
        args, config, environ={DASHBOARD_AUTH_ENV_VAR: TEST_TOKEN}
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, server.server_address
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_build_dashboard_server_binds_to_loopback_host_only(
    dashboard_process_server: tuple[http.server.ThreadingHTTPServer, tuple[str, int]],
) -> None:
    """The server `_build_dashboard_server` builds binds 127.0.0.1 only."""
    _server, address = dashboard_process_server

    assert address[0] == "127.0.0.1"


def test_build_dashboard_server_authenticated_root_shows_unknown_never(
    dashboard_process_server: tuple[http.server.ThreadingHTTPServer, tuple[str, int]],
) -> None:
    """With no `--ledger-path`, `/` reports an UNKNOWN mode and no heartbeat.

    Issue #447: the page has seen no ledger and therefore no heartbeat, so the
    only honest mode is "unknown". Rendering `RESEARCH` there made a dashboard
    with no evidence at all indistinguishable from one watching a healthy
    research-only loop -- the same lie the heartbeat's hardcoded constant told.
    """
    _server, address = dashboard_process_server

    status, _headers, body = _get(address, "/", headers=_bearer(TEST_TOKEN))

    assert status == 200
    assert "UNKNOWN" in body
    assert "RESEARCH" not in body
    assert "never" in body


def test_build_dashboard_server_unauthenticated_root_returns_401(
    dashboard_process_server: tuple[http.server.ThreadingHTTPServer, tuple[str, int]],
) -> None:
    """`/` still 401s without a bearer token, exactly like `create_server` alone."""
    _server, address = dashboard_process_server

    status, headers, _body = _get(address, "/")

    assert status == 401
    assert "WWW-Authenticate" in headers


def test_build_dashboard_server_binds_the_exact_configured_port() -> None:
    """A concrete configured port (not 0) is the exact port bound.

    Kills port-plumbing mutants that drop, scale, or ignore
    `config.dashboard.port` before it reaches `create_server`.
    """
    free_port = _free_tcp_port()
    config = WindbreakConfig(dashboard=DashboardConfig(port=free_port))
    args = argparse.Namespace(ledger_path=None)

    server = _build_dashboard_server(
        args, config, environ={DASHBOARD_AUTH_ENV_VAR: TEST_TOKEN}
    )
    try:
        assert server.server_address[1] == free_port
    finally:
        server.server_close()


# --- item 4: ledger-backed status + read models -----------------------------


def test_build_dashboard_server_with_ledger_path_shows_latest_mode_and_timestamp(
    tmp_path: Path,
) -> None:
    """`--ledger-path` surfaces the last `ModeHeartbeat`'s mode and `created_at`."""
    ledger_path = tmp_path / "ledger.db"
    store = SqliteLedgerStore(ledger_path)
    store.append(ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=1))
    store.append(ModeHeartbeat(component="pipeline", mode="PAPER", beat=2))
    last_record = store.read_all()[-1]
    store.close()
    config = WindbreakConfig(dashboard=DashboardConfig(port=0))
    args = argparse.Namespace(ledger_path=ledger_path)

    server = _build_dashboard_server(
        args, config, environ={DASHBOARD_AUTH_ENV_VAR: TEST_TOKEN}
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _headers, body = _get(
            server.server_address, "/", headers=_bearer(TEST_TOKEN)
        )
        assert status == 200
        assert "PAPER" in body
        assert last_record.created_at in body
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_build_dashboard_server_with_ledger_path_shows_equity_samples(
    tmp_path: Path,
) -> None:
    """`--ledger-path` also wires `read_models_source`; `/equity` reflects it."""
    ledger_path = tmp_path / "ledger.db"
    store = SqliteLedgerStore(ledger_path)
    store.append(
        EquitySampled(
            component="scheduler",
            equity_micros=1_234_000_000,
            floor_micros=1_000_000_000,
            epoch_s=1_700_000_000,
        )
    )
    store.close()
    config = WindbreakConfig(dashboard=DashboardConfig(port=0))
    args = argparse.Namespace(ledger_path=ledger_path)

    server = _build_dashboard_server(
        args, config, environ={DASHBOARD_AUTH_ENV_VAR: TEST_TOKEN}
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _headers, body = _get(
            server.server_address, "/equity", headers=_bearer(TEST_TOKEN)
        )
        assert status == 200
        assert "1234000000" in body
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


# --- item 5: `_serve_until_shutdown` ----------------------------------------


def test_serve_until_shutdown_with_preset_stop_event_closes_the_socket() -> None:
    """A pre-set `stop_event` shuts down promptly and fully closes the socket.

    A subsequent request raising `URLError` (connection refused) proves both
    `server.shutdown()` (stops `serve_forever`) and `server.server_close()`
    (releases the listening socket) ran -- either alone would leave the port
    either still serving or merely unresponsive rather than fully closed.
    """
    server = create_server(
        token=TEST_TOKEN,
        status_source=lambda: DashboardStatus(mode="RESEARCH", last_heartbeat=None),
        port=0,
    )
    address = server.server_address
    state = ShutdownState()
    state.stop_event.set()

    _serve_until_shutdown(server, state)

    with pytest.raises(urllib.error.URLError):
        _get(address, "/", headers=_bearer(TEST_TOKEN))


# --- item 6: in-process SIGINT via `main()` ---------------------------------


@pytest.mark.timeout(15)
def test_run_dashboard_in_process_sigint_exits_0_and_logs_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A SIGINT delivered mid-serve exits 0 and logs the serve + shutdown lines.

    Runs entirely on pytest's main thread: a `threading.Timer` delivers a
    real `SIGINT` to this process shortly after `main()` starts serving,
    bounded by this test's own timeout marker so a hang cannot wedge the
    suite.
    """
    monkeypatch.setenv(DASHBOARD_AUTH_ENV_VAR, TEST_TOKEN)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"dashboard": {"port": 0}}), encoding="utf-8")
    timer = threading.Timer(0.5, os.kill, args=(os.getpid(), signal.SIGINT))
    timer.daemon = True
    timer.start()
    try:
        exit_code = main(
            ["run", "--process", "dashboard", "--config", str(config_path)]
        )
    finally:
        timer.cancel()

    captured = capsys.readouterr()
    payloads = [json.loads(line) for line in captured.err.splitlines() if line]
    serve_payload = next(
        (
            payload
            for payload in payloads
            if "dashboard serving on 127.0.0.1:" in str(payload.get("msg", ""))
        ),
        None,
    )
    shutdown_payload = next(
        (
            payload
            for payload in payloads
            if str(payload.get("msg", "")).startswith("shutdown reason=")
        ),
        None,
    )
    assert exit_code == 0
    assert serve_payload is not None
    assert shutdown_payload is not None
    assert shutdown_payload["component"] == "dashboard"
    assert "SIGINT" in str(shutdown_payload["msg"])


# --- item 7: subprocess end-to-end (the issue's headline test) -------------


@pytest.mark.timeout(30)
def test_dashboard_process_subprocess_end_to_end(tmp_path: Path) -> None:
    """`python -m windbreak run --process dashboard` serves, authenticates,
    and shuts down cleanly on SIGTERM.

    The sole subprocess-level test in this module (mirrors
    `tests/test_process_flag.py`'s own single-subprocess-test convention);
    every other assertion above runs in-process for speed.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"dashboard": {"port": 0}}), encoding="utf-8")
    env = dict(os.environ)
    env[DASHBOARD_AUTH_ENV_VAR] = TEST_TOKEN

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "windbreak",
            "run",
            "--process",
            "dashboard",
            "--config",
            str(config_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        port = _await_dashboard_port(proc)
        address = ("127.0.0.1", port)

        status, _headers, _body = _get(address, "/", headers=_bearer(TEST_TOKEN))
        assert status == 200

        unauth_status, unauth_headers, _body = _get(address, "/")
        assert unauth_status == 401
        assert "WWW-Authenticate" in unauth_headers

        proc.send_signal(signal.SIGTERM)
        returncode = proc.wait(timeout=10)
        assert returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
