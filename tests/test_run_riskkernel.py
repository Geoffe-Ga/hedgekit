"""Failing-first tests wiring RiskKernel + KillIntegration into `windbreak run`
(issue #144, RED).

The kill switch (`KillSwitch`, `KillFileWatcher`, `ReconciliationMismatchMonitor`,
`KillIntegration` in `windbreak/riskkernel/kill.py`) is fully built and tested
(issue #35), but `windbreak/main.py` never composes it into the live
`windbreak run --process riskkernel` process -- that invocation still runs the
bare `run_loop` heartbeat, so a running deployment's `windbreak kill` writes a
`KILL` file no running kernel polls. `windbreak.main._build_risk_kernel` does
not exist yet, so every import below fails collection with
`ImportError: cannot import name '_build_risk_kernel' from 'windbreak.main'`
-- the expected Gate 1 RED state for issue #144.

Once `_build_risk_kernel` and its `_run_riskkernel` CLI routing land, this
file pins: a KILL file dropped into `ops.state_dir` actually halts a kernel
built by `_build_risk_kernel` (AC#2, no HTTP/dashboard involved); the real
`windbreak kill` CLI verb engages that same kernel end-to-end (AC#3); `windbreak
run --process riskkernel` routes to the real `RiskKernel` (proven by its
`heartbeat beat=N` log shape, distinct from `run_loop`'s `heartbeat seq=N`) and
its post-run shutdown line; a state directory that cannot be created fails
closed with a `FATAL` critical log and zero heartbeats (never entering the
kernel loop); `RiskConfig.kill_after_consecutive_mismatches` is threaded from
config into the wired `ReconciliationMismatchMonitor`, auto-killing the *shared*
mode machine both the kernel and the kill switch read; and the CLI's float
`--heartbeat-interval` is accepted and mapped onto the kernel's integer-seconds
`run(heartbeat_interval=...)` without raising.

Every test builds its own `WindbreakConfig` via `dataclasses.replace` over a
`tmp_path`-rooted `OpsConfig.state_dir`, and every CLI-level test that reads a
YAML `--config` file points `ops.state_dir` at a `tmp_path` too -- so no test
in this module ever touches the real `~/.local/share/windbreak`, where a KILL
file dropped by a previous manual run or a concurrent test process could
otherwise pollute this or a developer's environment.

Issue #235 (replay durable kill state on `windbreak run --process riskkernel`
startup) extends this module with the `--ledger-path` half of the story:
`_build_risk_kernel` gains a keyword-only `ledger_store` parameter that, when
given, persists kernel events to the real ledger (via a new
`PersistingKernelLedgerWriter`) and replays durable override/kill state from
it via `RiskKernel.from_events`/`KillSwitch.from_events` -- so an engaged kill
survives a `windbreak run --process riskkernel` restart even after its
belt-and-suspenders `KILL` file is deleted. None of this exists on the real,
not-yet-updated `windbreak/main.py` yet: `_build_risk_kernel(config,
ledger_store=...)` fails with `TypeError: _build_risk_kernel() got an
unexpected keyword argument 'ledger_store'`, and every persistence/replay
assertion below fails instead as a plain `AssertionError` against today's
`LoggingKernelLedgerWriter`-only, replay-free behavior -- both are the
expected Gate 1 RED state for issue #235, independent of (and in addition to)
whatever issue #144's tests above already pin.

Issue #236 (wire a `ReadOnlyVerifier` into the live composition, making the
`AUTO_RECONCILIATION` auto-kill trigger live rather than composed-but-dormant)
extends this module further: `_build_risk_kernel` gains a keyword-only
`verification_connector: MarketConnector | None = None` that, when supplied,
wires a `LedgerExpectationSource` (folding the replayed history, and -- with the
config-only ledger these tests use -- falling back on every dimension to that
same connector's own startup state) and a
`ReadOnlyVerifier` -- sharing the kernel's ledger writer and the kill switch's
`AlertDispatcher` -- through to `RiskKernel.from_events(verifier=...)`; and
`_drive_risk_kernel` builds a `FakeExchange.from_fixture_dir` connector from
`args.snapshot_fixture_dir` (when given) inside its existing fail-closed
`try`. None of this exists on the real, not-yet-updated `windbreak/main.py`
yet: `_build_risk_kernel(config, verification_connector=...)` fails with
`TypeError: _build_risk_kernel() got an unexpected keyword argument
'verification_connector'`, and the `--snapshot-fixture-dir` CLI-level
assertions fail instead as plain `AssertionError`s (zero `Verification*`
events recorded, or a fail-closed exit code of `0` instead of `1`) against
today's connector-less, snapshot-fixture-dir-ignoring composition -- both are
the expected Gate 1 RED state for issue #236, independent of (and in addition
to) issues #144 and #235's tests above.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from tests.riskkernel.conftest import make_context, make_intent
from windbreak.config import OpsConfig, RiskConfig, WindbreakConfig
from windbreak.connector.interface import UnknownMarketError
from windbreak.connector.models import (
    BalanceSemantics,
    BalanceSnapshot,
    ExchangeStatus,
    FeeModel,
    Fill,
    NormalizedMarket,
    OpenOrder,
    OrderBookSnapshot,
    Position,
)
from windbreak.connector.semantics import (
    CancelCollateralRelease,
    FeeDebitTiming,
    FeeRounding,
    HaltedMarketBehavior,
    OrderCollateralInAvailable,
    OrderCollateralInTotal,
    PartialFillRepresentation,
    UnsettledProceeds,
)
from windbreak.ledger.events import KillEngaged, KillReArmed
from windbreak.ledger.store import SqliteLedgerStore, events_from_records
from windbreak.main import _build_risk_kernel, main
from windbreak.numeric.types import MoneyMicros
from windbreak.riskkernel.kill import KILL_FILENAME, KillTrigger
from windbreak.riskkernel.modes import Mode
from windbreak.riskkernel.verification import VerificationOutcome

#: A fixed epoch for every seeded `KillEngaged` below, so this module never
#: depends on wall-clock time.
_FIXED_EPOCH_S = 1_700_000_000


def _config_with_state_dir(state_dir: Path, **overrides: object) -> WindbreakConfig:
    """Build a `WindbreakConfig` whose `ops.state_dir` is `state_dir`.

    Args:
        state_dir: The directory `ops.state_dir` is pointed at (a `tmp_path`
            in every caller, so no test ever touches the real state dir).
        **overrides: Additional top-level `WindbreakConfig` field overrides
            (e.g. `risk=RiskConfig(...)`).

    Returns:
        A `WindbreakConfig` built from defaults except `ops.state_dir` and
        any `**overrides`.
    """
    return dataclasses.replace(
        WindbreakConfig(), ops=OpsConfig(state_dir=str(state_dir)), **overrides
    )


def _write_state_dir_config(config_path: Path, state_dir: Path) -> None:
    """Write a minimal YAML config whose only key is `ops.state_dir`.

    Args:
        config_path: Where to write the YAML document.
        state_dir: The directory `ops.state_dir` is pointed at.
    """
    config_path.write_text(
        yaml.safe_dump({"ops": {"state_dir": str(state_dir)}}), encoding="utf-8"
    )


def _json_lines(stderr: str) -> list[dict[str, object]]:
    """Parse every non-blank line of captured stderr as a JSON log record.

    Args:
        stderr: The captured stderr text (one JSON object per line).

    Returns:
        The parsed JSON payloads, in emission order.
    """
    return [json.loads(line) for line in stderr.splitlines() if line]


# --- issue #236 fixtures: a mutable connector for the live-verifier wiring -----
#
# `_build_risk_kernel` will gain a keyword-only `verification_connector` that,
# when supplied, wires a `LedgerExpectationSource` (folding the replayed
# history; with the config-only ledger here it falls back on every dimension to
# the connector's own balances/positions/open-orders, captured exactly once, at
# construction) plus a `ReadOnlyVerifier` sharing the kernel's ledger writer
# and the kill switch's `AlertDispatcher`. `_DriftingBalanceConnector` reports
# one available-cash figure on its very first `get_balances` call -- the
# baseline-capture call -- and a second, caller-chosen figure on every call
# after that, so the venue "drifts" from its own captured startup baseline by
# exactly the chosen amount. Positions and open orders are held flat (always
# empty) throughout, so cash drift is the sole, deterministic breach cause and
# `_alert_unknown_jurisdictions`'s per-ticker `get_market` lookup is never
# reached.

#: A fixed UTC instant for every `BalanceSnapshot.fetched_at` the stub
#: connector below reports; its exact value is irrelevant to every assertion.
_FIXED_DATETIME = datetime(2024, 1, 1, tzinfo=UTC)

#: A `BalanceSemantics` with every field a known (non-`UNKNOWN`) member, so the
#: verifier's `semantics_fully_known` flag never confounds a test's assertions.
_FULLY_KNOWN_SEMANTICS = BalanceSemantics(
    open_order_collateral_in_total=OrderCollateralInTotal.EXCLUDED,
    open_order_collateral_in_available=OrderCollateralInAvailable.DEDUCTED_FROM_AVAILABLE,
    fee_debit_timing=FeeDebitTiming.AT_EXECUTION,
    fee_rounding=FeeRounding.EXACT,
    partial_fill_representation=PartialFillRepresentation.PER_FILL_RECORDS,
    cancel_collateral_release=CancelCollateralRelease.IMMEDIATE,
    unsettled_proceeds=UnsettledProceeds.INCLUDED_IMMEDIATELY,
    halted_market_behavior=HaltedMarketBehavior.NEW_ORDERS_REJECTED,
)

#: `tests/fixtures/verification/clean` -- a full `FakeExchange` fixture
#: directory whose own state trivially matches itself, so a
#: `LedgerExpectationSource` built over it (falling back to the connector for
#: the config-only ledger these tests use) always grades CLEAN: the
#: fixture used for the CLI's "verification wired and passing" happy path.
_CLEAN_SNAPSHOT_FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "verification" / "clean"
)


@dataclass
class _DriftingBalanceConnector:
    """A minimal, mutable `MarketConnector` whose available cash steps once.

    Reports `available_before_drift` on the very first `get_balances` call
    (the `LedgerExpectationSource` construction-time cash fallback, taken once
    for the config-only ledger these tests use) and `available_after_drift` on
    every call after that (every later verification cycle).

    Attributes:
        available_before_drift: The available cash reported on the first
            `get_balances` call only.
        available_after_drift: The available cash reported on every
            `get_balances` call after the first.
    """

    available_before_drift: MoneyMicros
    available_after_drift: MoneyMicros
    _calls: int = field(default=0, init=False)

    def get_balances(self) -> BalanceSnapshot:
        """Return the baseline balance once, then the drifted balance always."""
        self._calls += 1
        available = (
            self.available_before_drift
            if self._calls == 1
            else self.available_after_drift
        )
        return BalanceSnapshot(
            total=available, available=available, fetched_at=_FIXED_DATETIME
        )

    def get_positions(self) -> tuple[Position, ...]:
        """Return no positions, ever (cash is the sole breach dimension)."""
        return ()

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        """Return no open orders, ever (cash is the sole breach dimension)."""
        return ()

    def get_balance_semantics(self) -> BalanceSemantics:
        """Return a fully-known `BalanceSemantics` (irrelevant to this test)."""
        return _FULLY_KNOWN_SEMANTICS

    def list_markets(self) -> tuple[NormalizedMarket, ...]:
        """Return no markets; never called (no positions/open orders held)."""
        return ()

    def get_market(self, ticker: str) -> NormalizedMarket:
        """Raise; never called (no positions/open orders held)."""
        raise UnknownMarketError(ticker)

    def get_order_book(self, ticker: str) -> OrderBookSnapshot:
        """Raise; never called by the verification path."""
        raise NotImplementedError(ticker)

    def get_exchange_status(self) -> ExchangeStatus:
        """Raise; never called by the verification path."""
        raise NotImplementedError

    def get_exchange_time(self) -> datetime:
        """Raise; never called by the verification path."""
        raise NotImplementedError

    def get_fills(self, since: datetime) -> tuple[Fill, ...]:
        """Return no fills; never called by the verification path."""
        del since
        return ()

    def get_fee_model(self, market_or_series: str) -> FeeModel:
        """Raise; never called by the verification path."""
        raise NotImplementedError(market_or_series)

    def place_order(self, normalized_intent: object, approval_token: object) -> object:
        """Raise; the verifier never places orders."""
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> None:
        """Raise; the verifier never cancels orders."""
        raise NotImplementedError(order_id)


# --- _build_risk_kernel: shape of the wired kernel/integration ------------------


def test_build_risk_kernel_returns_research_mode_kernel_wired_with_kill_integration(
    tmp_path: Path,
) -> None:
    """`_build_risk_kernel` returns a fresh-RESEARCH kernel with a full
    `KillIntegration` (switch, watcher, monitor all present), and creates
    `ops.state_dir` fail-closed up front.
    """
    state_dir = tmp_path / "state"
    config = _config_with_state_dir(state_dir)

    kernel, integration = _build_risk_kernel(config)

    assert kernel.mode is Mode.RESEARCH
    assert integration.switch is not None
    assert integration.watcher is not None
    assert integration.monitor is not None
    assert state_dir.is_dir()


# --- AC#2: a KILL file halts the kernel with no HTTP/dashboard involved ---------


@pytest.mark.timeout(30)
def test_kill_file_works_without_http(tmp_path: Path) -> None:
    """A `KILL` file dropped into `ops.state_dir` halts a kernel built by
    `_build_risk_kernel`, with zero HTTP server or dashboard object ever
    constructed: one beat with no file leaves the kernel un-killed and
    approving evaluation as usual; dropping the file and running one more
    beat kills it, and `evaluate_intent` is thereafter hard-vetoed.
    """
    config = _config_with_state_dir(tmp_path)
    kernel, _integration = _build_risk_kernel(config)

    kernel.run(max_beats=1, heartbeat_interval=0)
    assert kernel.mode is not Mode.KILLED

    (tmp_path / KILL_FILENAME).write_text("", encoding="utf-8")
    kernel.run(max_beats=1, heartbeat_interval=0)

    assert kernel.mode is Mode.KILLED
    decision = kernel.evaluate_intent(make_intent(), make_context())
    assert decision.reasons == ("KILLED",)


# --- AC#3: the real `windbreak kill` CLI verb halts the wired kernel ------------


@pytest.mark.timeout(30)
def test_kill_cli_verb_engages_the_wired_kernel_end_to_end(tmp_path: Path) -> None:
    """`windbreak kill --state-dir DIR` (the real CLI verb, via `main`) writes
    the `KILL` file a kernel built by `_build_risk_kernel` over the same `DIR`
    then picks up on its next beat, hard-vetoing every subsequent intent.
    """
    config = _config_with_state_dir(tmp_path)
    kernel, _integration = _build_risk_kernel(config)

    exit_code = main(["kill", "--state-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / KILL_FILENAME).exists()

    kernel.run(max_beats=1, heartbeat_interval=0)

    assert kernel.mode is Mode.KILLED
    decision = kernel.evaluate_intent(make_intent(), make_context())
    assert decision.reasons == ("KILLED",)


# --- Routing: `run --process riskkernel` drives the real RiskKernel ------------


@pytest.mark.timeout(30)
def test_run_process_riskkernel_routes_to_the_real_risk_kernel_not_run_loop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`windbreak run --process riskkernel` exits 0 and logs a `RiskKernel`
    heartbeat (`heartbeat beat=1`, `component=riskkernel`) followed by a
    `shutdown reason=max_beats` line -- never the bare `run_loop`'s
    `heartbeat seq=` shape, proving the routing divergence actually composes
    a real kernel rather than falling through to the pre-#144 heartbeat.
    """
    config_path = tmp_path / "config.yaml"
    _write_state_dir_config(config_path, tmp_path)

    exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--max-beats",
            "1",
            "--heartbeat-interval",
            "0",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    payloads = _json_lines(capsys.readouterr().err)
    assert not any("seq=" in str(payload.get("msg", "")) for payload in payloads)
    heartbeat_payload = next(
        payload
        for payload in payloads
        if "heartbeat beat=1" in str(payload.get("msg", ""))
    )
    assert heartbeat_payload["component"] == "riskkernel"
    shutdown_payload = next(
        payload
        for payload in payloads
        if str(payload.get("msg", "")) == "shutdown reason=max_beats"
    )
    assert shutdown_payload["component"] == "riskkernel"


# --- Fail-closed startup: an uncreatable state dir never enters the loop -------


@pytest.mark.timeout(30)
def test_run_process_riskkernel_fails_closed_when_state_dir_cannot_be_created(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When `ops.state_dir` resolves under an existing regular file (so the
    builder's `mkdir` raises), `run --process riskkernel` returns 1, logs a
    `FATAL` critical line, and never enters the kernel loop -- zero heartbeat
    lines are emitted.
    """
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("this is a file, not a directory", encoding="utf-8")
    unusable_state_dir = blocking_file / "state"
    config_path = tmp_path / "config.yaml"
    _write_state_dir_config(config_path, unusable_state_dir)

    exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--max-beats",
            "1",
            "--heartbeat-interval",
            "0",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 1
    payloads = _json_lines(capsys.readouterr().err)
    assert any(
        payload.get("level") == "CRITICAL" and "FATAL" in str(payload.get("msg", ""))
        for payload in payloads
    )
    assert not any("heartbeat" in str(payload.get("msg", "")) for payload in payloads)


# --- Threshold wiring: RiskConfig.kill_after_consecutive_mismatches ------------


def test_reconciliation_threshold_is_threaded_from_config_onto_the_shared_machine(
    tmp_path: Path,
) -> None:
    """A kernel built from a config with
    `risk.kill_after_consecutive_mismatches=2` auto-kills on exactly the 2nd
    consecutive `BREACH` observation -- not the 1st -- and the kill is visible
    on both the kill switch and the kernel, proving they share one mode
    machine rather than two independently-tracked ones.
    """
    config = _config_with_state_dir(
        tmp_path, risk=RiskConfig(kill_after_consecutive_mismatches=2)
    )
    kernel, integration = _build_risk_kernel(config)

    integration.monitor.observe(VerificationOutcome.BREACH)
    assert integration.switch.mode is not Mode.KILLED
    assert kernel.mode is not Mode.KILLED

    integration.monitor.observe(VerificationOutcome.BREACH)
    assert integration.switch.mode is Mode.KILLED
    assert kernel.mode is Mode.KILLED


# --- Interval mapping: the CLI's float seconds reach the kernel's int seconds --


@pytest.mark.timeout(30)
def test_run_process_riskkernel_accepts_a_fractional_heartbeat_interval(
    tmp_path: Path,
) -> None:
    """`--heartbeat-interval 0.5` (a float, the shared `run` flag's type) is
    accepted for `--process riskkernel` and exits 0 -- the CLI's float seconds
    must be mapped onto the kernel's integer-seconds `run(heartbeat_interval=)`
    without ever raising a `TypeError`/`ValueError` from the float/int seam.
    """
    config_path = tmp_path / "config.yaml"
    _write_state_dir_config(config_path, tmp_path)

    exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--heartbeat-interval",
            "0.5",
            "--max-beats",
            "1",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0


# --- Persistence: kernel events reach the real ledger (issue #235) -------------


@pytest.mark.timeout(30)
def test_run_process_riskkernel_persists_kernel_events_to_the_real_ledger(
    tmp_path: Path,
) -> None:
    """`windbreak run --process riskkernel --ledger-path P` persists the
    kernel's own `ModeHeartbeat` events to the real ledger, landing after the
    `ConfigLoaded` row(s) `_load_and_ledger_config` writes first -- and the
    resulting chain verifies cleanly. This proves the running kernel now
    writes through a real `LedgerStore` rather than only the
    `LoggingKernelLedgerWriter` stand-in, whose events never reach disk.
    """
    ledger_path = tmp_path / "ledger.db"
    config_path = tmp_path / "config.yaml"
    _write_state_dir_config(config_path, tmp_path)

    exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--ledger-path",
            str(ledger_path),
            "--max-beats",
            "1",
            "--heartbeat-interval",
            "0",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    event_types = [record.event_type for record in records]
    assert "ConfigLoaded" in event_types
    assert "ModeHeartbeat" in event_types
    assert event_types.index("ModeHeartbeat") > event_types.index("ConfigLoaded")
    heartbeat_record = next(
        record for record in records if record.event_type == "ModeHeartbeat"
    )
    assert heartbeat_record.component == "riskkernel"


# --- Headline AC: a KILLED state survives a restart via ledger replay ----------


@pytest.mark.timeout(30)
def test_run_process_riskkernel_replays_killed_state_across_restart(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A kill engaged during run 1 (via a dropped `KILL` file) is replayed
    from the ledger at run 2's startup, even after the `KILL` file is deleted
    before run 2 begins: run 2's *first* heartbeat line already reports
    `mode=KILLED heartbeat beat=1`, proving ledger replay -- not the file
    watcher, which sees no `KILL` file on this run -- restored the kill.
    """
    ledger_path = tmp_path / "ledger.db"
    config_path = tmp_path / "config.yaml"
    _write_state_dir_config(config_path, tmp_path)
    (tmp_path / KILL_FILENAME).write_text("", encoding="utf-8")

    run_one_exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--ledger-path",
            str(ledger_path),
            "--max-beats",
            "1",
            "--heartbeat-interval",
            "0",
            "--config",
            str(config_path),
        ]
    )
    assert run_one_exit_code == 0
    capsys.readouterr()  # discard run 1's output; only run 2's is asserted below

    store = SqliteLedgerStore(ledger_path)
    try:
        run_one_event_types = [record.event_type for record in store.read_all()]
    finally:
        store.close()
    assert "KillEngaged" in run_one_event_types

    (tmp_path / KILL_FILENAME).unlink()

    run_two_exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--ledger-path",
            str(ledger_path),
            "--max-beats",
            "1",
            "--heartbeat-interval",
            "0",
            "--config",
            str(config_path),
        ]
    )

    assert run_two_exit_code == 0
    payloads = _json_lines(capsys.readouterr().err)
    heartbeat_payload = next(
        payload
        for payload in payloads
        if str(payload.get("msg", "")) == "mode=KILLED heartbeat beat=1"
    )
    assert heartbeat_payload["component"] == "riskkernel"


# --- Rearm pair: replayed as not-killed, but the sequence is preserved ---------


def test_build_risk_kernel_over_rearmed_history_stays_research_but_keeps_sequence(
    tmp_path: Path,
) -> None:
    """A ledger seeded with a matching `KillEngaged`/`KillReArmed` pair
    replays as *not killed*: `_build_risk_kernel(config, ledger_store=...)`
    rebuilds a fresh-`RESEARCH` kernel. But the kill switch's
    `active_kill_sequence` still reflects the ledgered kill (unconditionally
    restored, per `KillSwitch.from_events`), so a later kill increments
    monotonically rather than resetting back to 1.
    """
    ledger_path = tmp_path / "ledger.db"
    seed_store = SqliteLedgerStore(ledger_path)
    try:
        seed_store.append(
            KillEngaged(
                component="riskkernel",
                trigger="CLI",
                kill_sequence=1,
                epoch=_FIXED_EPOCH_S,
            )
        )
        seed_store.append(KillReArmed(component="riskkernel", kill_sequence=1))
    finally:
        seed_store.close()

    config = _config_with_state_dir(tmp_path)
    ledger_store = SqliteLedgerStore(ledger_path)
    try:
        kernel, integration = _build_risk_kernel(config, ledger_store=ledger_store)
    finally:
        ledger_store.close()

    assert kernel.mode is Mode.RESEARCH
    assert integration.switch.active_kill_sequence == 1


# --- Fail-closed: a tampered ledger never enters the loop -----------------------


@pytest.mark.timeout(30)
def test_run_process_riskkernel_fails_closed_on_a_tampered_ledger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ledger row tampered between two runs fails `verify_chain()` at
    kernel-build time: `run --process riskkernel --ledger-path P` returns 1,
    logs a `CRITICAL` `FATAL` line, and never enters the heartbeat loop --
    zero heartbeat lines are emitted, mirroring the uncreatable-state-dir
    fail-closed startup path above.
    """
    ledger_path = tmp_path / "ledger.db"
    config_path = tmp_path / "config.yaml"
    _write_state_dir_config(config_path, tmp_path)

    seed_exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--ledger-path",
            str(ledger_path),
            "--max-beats",
            "1",
            "--heartbeat-interval",
            "0",
            "--config",
            str(config_path),
        ]
    )
    assert seed_exit_code == 0
    capsys.readouterr()

    connection = sqlite3.connect(ledger_path)
    try:
        connection.execute(
            "UPDATE ledger SET event_hash = ? WHERE sequence_number = 1",
            ("0" * 64,),
        )
        connection.commit()
    finally:
        connection.close()

    exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--ledger-path",
            str(ledger_path),
            "--max-beats",
            "1",
            "--heartbeat-interval",
            "0",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 1
    payloads = _json_lines(capsys.readouterr().err)
    assert any(
        payload.get("level") == "CRITICAL" and "FATAL" in str(payload.get("msg", ""))
        for payload in payloads
    )
    assert not any("heartbeat" in str(payload.get("msg", "")) for payload in payloads)


# --- Issue #236: wiring a live ReadOnlyVerifier makes AUTO_RECONCILIATION live -


@pytest.mark.timeout(30)
def test_sustained_breach_through_build_risk_kernel_auto_kills_via_reconciliation(
    tmp_path: Path,
) -> None:
    """A `_build_risk_kernel(config, verification_connector=stub, ...)` kernel
    auto-kills on a sustained cash-drift breach: `kill_after_consecutive_mismatches=2`
    with default (zero) tolerances means the 2nd consecutive `BREACH` engages
    the shared kill switch via `KillTrigger.AUTO_RECONCILIATION`.

    `_build_risk_kernel` does not accept `verification_connector` yet, so this
    call fails with `TypeError: _build_risk_kernel() got an unexpected keyword
    argument 'verification_connector'` -- the expected Gate 1 RED state for
    issue #236.
    """
    ledger_path = tmp_path / "ledger.db"
    config = _config_with_state_dir(
        tmp_path, risk=RiskConfig(kill_after_consecutive_mismatches=2)
    )
    stub = _DriftingBalanceConnector(
        available_before_drift=MoneyMicros(10_000_000),
        available_after_drift=MoneyMicros(9_000_000),
    )
    store = SqliteLedgerStore(ledger_path)
    try:
        kernel, integration = _build_risk_kernel(
            config, verification_connector=stub, ledger_store=store
        )

        kernel.run(max_beats=3, heartbeat_interval=0)

        assert kernel.mode is Mode.KILLED
        assert integration.switch.mode is Mode.KILLED

        store.verify_chain()
        events = events_from_records(store.read_all())
    finally:
        store.close()

    event_types = [event.event_type for event in events]
    assert event_types.count("VerificationMismatch") == 3
    assert event_types.count("VerificationMismatchHalt") == 1
    kill_events = [event for event in events if event.event_type == "KillEngaged"]
    assert len(kill_events) == 1
    assert kill_events[0].payload["trigger"] == KillTrigger.AUTO_RECONCILIATION.name


def test_verification_balance_tolerance_threads_from_config_inclusive_boundary(
    tmp_path: Path,
) -> None:
    """`RiskConfig.verification_balance_tolerance_micros` reaches the composed
    `VerificationTolerances`: a drift exactly at the configured threshold is
    `DRIFT_WITHIN_TOLERANCE` and never halts (inclusive boundary, matching
    every other tolerance/ttl check in this codebase).

    `_build_risk_kernel` does not accept `verification_connector` yet, so this
    fails with the same `TypeError` as the breach test above.
    """
    tolerance_micros = 500
    baseline = MoneyMicros(10_000_000)
    at_boundary = MoneyMicros(baseline.value - tolerance_micros)
    config = _config_with_state_dir(
        tmp_path,
        risk=RiskConfig(verification_balance_tolerance_micros=tolerance_micros),
    )
    stub = _DriftingBalanceConnector(
        available_before_drift=baseline, available_after_drift=at_boundary
    )

    kernel, integration = _build_risk_kernel(config, verification_connector=stub)
    kernel.run(max_beats=1, heartbeat_interval=0)

    assert kernel.mode is Mode.RESEARCH
    assert integration.switch.mode is not Mode.KILLED


def test_verification_balance_tolerance_breaches_one_micro_past_the_boundary(
    tmp_path: Path,
) -> None:
    """One micro past `verification_balance_tolerance_micros` is a `BREACH`
    that halts the kernel -- proving the configured tolerance, not some
    hard-coded default, is what the composed `VerificationTolerances` carries.
    """
    tolerance_micros = 500
    baseline = MoneyMicros(10_000_000)
    past_boundary = MoneyMicros(baseline.value - tolerance_micros - 1)
    config = _config_with_state_dir(
        tmp_path,
        risk=RiskConfig(verification_balance_tolerance_micros=tolerance_micros),
    )
    stub = _DriftingBalanceConnector(
        available_before_drift=baseline, available_after_drift=past_boundary
    )

    kernel, _integration = _build_risk_kernel(config, verification_connector=stub)
    kernel.run(max_beats=1, heartbeat_interval=0)

    assert kernel.mode is Mode.HALT


def test_verification_default_zero_tolerance_breaches_on_any_drift(
    tmp_path: Path,
) -> None:
    """With no `verification_balance_tolerance_micros` configured (the
    fail-closed exact-match default of `0`), even a 1-micro cash drift is a
    `BREACH` that halts the kernel.
    """
    baseline = MoneyMicros(10_000_000)
    drifted = MoneyMicros(baseline.value - 1)
    config = _config_with_state_dir(tmp_path)
    stub = _DriftingBalanceConnector(
        available_before_drift=baseline, available_after_drift=drifted
    )

    kernel, _integration = _build_risk_kernel(config, verification_connector=stub)
    kernel.run(max_beats=1, heartbeat_interval=0)

    assert kernel.mode is Mode.HALT


def test_build_risk_kernel_accepts_an_explicit_none_verification_connector(
    tmp_path: Path,
) -> None:
    """`_build_risk_kernel(config, verification_connector=None)` -- the
    explicit opt-out spelling -- is accepted and records zero `Verification*`
    events across a beat: `verifier=None` reaches `RiskKernel.from_events`
    unchanged from pre-issue-#236 behavior.

    `_build_risk_kernel` has no `verification_connector` parameter yet, so
    this fails with `TypeError: _build_risk_kernel() got an unexpected keyword
    argument 'verification_connector'`.
    """
    ledger_path = tmp_path / "ledger.db"
    config = _config_with_state_dir(tmp_path)
    store = SqliteLedgerStore(ledger_path)
    try:
        kernel, _integration = _build_risk_kernel(
            config, verification_connector=None, ledger_store=store
        )
        kernel.run(max_beats=1, heartbeat_interval=0)
        event_types = [record.event_type for record in store.read_all()]
    finally:
        store.close()

    assert not any(event_type.startswith("Verification") for event_type in event_types)


@pytest.mark.timeout(30)
def test_run_process_riskkernel_without_snapshot_fixture_dir_records_no_verification(
    tmp_path: Path,
) -> None:
    """`windbreak run --process riskkernel` with no `--snapshot-fixture-dir`
    records zero `Verification*` events and exits 0 -- the CLI opt-out path
    stays byte-identical to pre-issue-#236 behavior. (This assertion already
    holds today, since no verifier is wired at all yet; it is included here as
    a regression guard alongside the RED tests above, not as a RED pin
    itself.)
    """
    ledger_path = tmp_path / "ledger.db"
    config_path = tmp_path / "config.yaml"
    _write_state_dir_config(config_path, tmp_path)

    exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--ledger-path",
            str(ledger_path),
            "--max-beats",
            "1",
            "--heartbeat-interval",
            "0",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    store = SqliteLedgerStore(ledger_path)
    try:
        event_types = [record.event_type for record in store.read_all()]
    finally:
        store.close()
    assert not any(event_type.startswith("Verification") for event_type in event_types)


@pytest.mark.timeout(30)
def test_run_process_riskkernel_with_snapshot_fixture_dir_records_verification_passed(
    tmp_path: Path,
) -> None:
    """`windbreak run --process riskkernel --snapshot-fixture-dir DIR` builds a
    `FakeExchange` over `DIR`, wires it as the verification connector, and
    records exactly one `VerificationPassed` event on the CLEAN fixture --
    proving `_drive_risk_kernel` now threads `--snapshot-fixture-dir` through
    to `_build_risk_kernel(verification_connector=...)`.

    Today `--snapshot-fixture-dir` is accepted by the CLI but never consulted
    for `--process riskkernel`, so no verifier is ever wired: this fails as a
    plain `AssertionError` (zero `VerificationPassed` events, not one) rather
    than a crash -- the expected Gate 1 RED state for issue #236.
    """
    ledger_path = tmp_path / "ledger.db"
    config_path = tmp_path / "config.yaml"
    _write_state_dir_config(config_path, tmp_path)

    exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--snapshot-fixture-dir",
            str(_CLEAN_SNAPSHOT_FIXTURE_DIR),
            "--ledger-path",
            str(ledger_path),
            "--max-beats",
            "1",
            "--heartbeat-interval",
            "0",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    passed_records = [
        record for record in records if record.event_type == "VerificationPassed"
    ]
    assert len(passed_records) == 1
    assert passed_records[0].component == "riskkernel"


@pytest.mark.timeout(30)
def test_run_process_riskkernel_fails_closed_on_a_missing_snapshot_fixture_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `--snapshot-fixture-dir` pointing at a nonexistent directory fails
    closed exactly like an uncreatable state dir or a tampered ledger: exit 1,
    a `CRITICAL` `FATAL` log line, and zero heartbeat lines -- the missing
    directory is discovered building the `FakeExchange` *inside*
    `_drive_risk_kernel`'s existing fail-closed `try`, so it never enters the
    kernel loop.

    Today `--snapshot-fixture-dir` is never consulted for `--process
    riskkernel`, so the bad path is silently ignored and the run exits 0: this
    fails as `assert 0 == 1`, the expected Gate 1 RED state for issue #236.
    """
    config_path = tmp_path / "config.yaml"
    _write_state_dir_config(config_path, tmp_path)
    missing_fixture_dir = tmp_path / "does-not-exist"

    exit_code = main(
        [
            "run",
            "--process",
            "riskkernel",
            "--snapshot-fixture-dir",
            str(missing_fixture_dir),
            "--max-beats",
            "1",
            "--heartbeat-interval",
            "0",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 1
    payloads = _json_lines(capsys.readouterr().err)
    assert any(
        payload.get("level") == "CRITICAL" and "FATAL" in str(payload.get("msg", ""))
        for payload in payloads
    )
    assert not any("heartbeat" in str(payload.get("msg", "")) for payload in payloads)
