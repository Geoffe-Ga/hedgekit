"""Failing-first tests for windbreak.riskkernel.process (issue #29, RED).

Issue #29 gives the Risk Kernel (Process B, SPEC S5.1-S5.3) its process
skeleton: a bounded heartbeat loop, an `evaluate_intent` entry point that
records every veto to a ledger writer, a `main()` CLI matching
`windbreak.main`'s bounded-loop conventions, and -- the load-bearing isolation
property SPEC S5.3 exists to protect -- the guarantee that *no other package*
can ever import the approval-token signing key handle.

None of `windbreak/riskkernel/{process,signing}.py` or
`windbreak/riskkernel/__main__.py` exist yet, so the imports below fail the
whole module at collection with
`ModuleNotFoundError: No module named 'windbreak.riskkernel.process'` -- the
expected Gate 1 RED state for issue #29. Once the modules exist, this file
pins: a self-contained AST scanner (mirroring
`tests/forecast/test_sandbox.py`'s import-boundary checker) proving zero
`windbreak.riskkernel.signing` imports anywhere outside the `riskkernel`
package; a matching `plans/architecture/.importlinter` contract; the
`KernelLedgerWriter` trio (`Logging`/`InMemory`, mirroring
`windbreak.connector.snapshot`'s `EventLedgerWriter` trio); `RiskKernel`'s
bounded heartbeat loop and ledgered `evaluate_intent`; a subprocess-level
"Process B survives Process A" isolation smoke test; and (issue #31) that
`SigningKeyHandle` signs with real HMAC-SHA256 while never exposing its key
material through any public attribute.

Issue #235 (replay durable kill state on `windbreak run --process riskkernel`
startup) adds a fourth `KernelLedgerWriter`, `PersistingKernelLedgerWriter`,
wrapping a real `LedgerStore` so kernel events are actually persisted rather
than only logged -- it does not exist on the real, not-yet-updated
`process.py` module yet, so importing it below fails collection with
`ImportError: cannot import name 'PersistingKernelLedgerWriter' from
'windbreak.riskkernel.process'`, the expected Gate 1 RED state for issue
#235. It also gives `RiskKernel.from_events` a new keyword-only
`kill_integration` passthrough, so a rebuilt kernel's kill wiring survives a
restart alongside its replayed mode; the not-yet-updated classmethod raises
`TypeError: from_events() got an unexpected keyword argument
'kill_integration'` when a test below supplies it, independently pinning
that half of issue #235's Gate 1 RED state.
"""

from __future__ import annotations

import ast
import configparser
import contextlib
import hashlib
import hmac
import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.riskkernel.conftest import make_context
from windbreak.alerts.dispatch import AlertDispatcher, LoggingLedgerWriter
from windbreak.ledger.events import Event
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.numeric.types import (
    ContractCentis,
    MoneyMicros,
    PricePips,
    ProbabilityPpm,
)
from windbreak.riskkernel.checks import OrderIntent
from windbreak.riskkernel.kill import (
    KILL_FILENAME,
    KillFileWatcher,
    KillIntegration,
    KillSwitch,
)
from windbreak.riskkernel.modes import Mode, ModeStateMachine
from windbreak.riskkernel.process import (
    InMemoryKernelLedgerWriter,
    LoggingKernelLedgerWriter,
    PersistingKernelLedgerWriter,
    RiskKernel,
)
from windbreak.riskkernel.process import main as riskkernel_main
from windbreak.riskkernel.signing import SigningKeyHandle

if TYPE_CHECKING:
    from windbreak.riskkernel.context import EvaluationContext
    from windbreak.riskkernel.process import KernelLedgerWriter

#: Repo root, derived from this test file's own location
#: (`<root>/tests/riskkernel/test_process_isolation.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]

_WINDBREAK_PACKAGE_DIR = _REPO_ROOT / "windbreak"
_RISKKERNEL_PACKAGE_DIR = _WINDBREAK_PACKAGE_DIR / "riskkernel"
_IMPORTLINTER_PATH = _REPO_ROOT / "plans" / "architecture" / ".importlinter"

#: The single module SPEC S5.3 reserves to the `riskkernel` package alone.
_FORBIDDEN_SIGNING_MODULE = "windbreak.riskkernel.signing"

#: The `.importlinter` contract section declaring the signing-key boundary.
_SIGNING_CONTRACT_SECTION = "importlinter:contract:signing-key-isolation"

#: Immutable scaled-int defaults for :func:`_make_intent`, held as module-level
#: singletons so they are not reconstructed in the function's argument defaults
#: (ruff B008); the wrapper types are frozen, so sharing one instance is safe.
_DEFAULT_PRICE = PricePips(5000)
_DEFAULT_SIZE = ContractCentis(1000)
_DEFAULT_MAX_NOTIONAL = MoneyMicros(50_000_000)
_DEFAULT_IMPLIED_PROBABILITY = ProbabilityPpm(520_000)


#: Default `OrderIntent.idempotency_key` for :func:`_make_intent` (issue #31).
_DEFAULT_IDEMPOTENCY_KEY = "idem-0001"


def _make_intent(
    *,
    intent_id: str = "intent-0001",
    market_ticker: str = "PRES-2028-DEM",
    outcome: str = "yes",
    action: str = "buy",
    price: PricePips = _DEFAULT_PRICE,
    size: ContractCentis = _DEFAULT_SIZE,
    max_notional: MoneyMicros = _DEFAULT_MAX_NOTIONAL,
    implied_probability: ProbabilityPpm = _DEFAULT_IMPLIED_PROBABILITY,
    idempotency_key: str = _DEFAULT_IDEMPOTENCY_KEY,
) -> OrderIntent:
    """Build a valid `OrderIntent` for kernel-evaluation tests.

    Args:
        intent_id: The intent's unique identifier.
        market_ticker: The exchange ticker the intent targets.
        outcome: The market outcome the intent trades (e.g. "yes"/"no").
        action: The trade action (e.g. "buy"/"sell").
        price: The limit price, in pips.
        size: The contract count, in centis.
        max_notional: The notional cap, in money-micros.
        implied_probability: The forecast-implied probability, in ppm.
        idempotency_key: The caller-supplied idempotency key (issue #31).

    Returns:
        A fully populated, valid `OrderIntent`.
    """
    return OrderIntent(
        intent_id=intent_id,
        market_ticker=market_ticker,
        outcome=outcome,
        action=action,
        price=price,
        size=size,
        max_notional=max_notional,
        implied_probability=implied_probability,
        idempotency_key=idempotency_key,
    )


# --- AST boundary: zero forbidden signing imports outside riskkernel -----------
#
# Self-contained (stdlib `ast` only), mirroring tests/forecast/test_sandbox.py's
# own import-boundary checker: `ast.walk` visits every node regardless of
# nesting, so an import inside `if TYPE_CHECKING:` is inspected exactly like a
# top-level one.


def _find_signing_imports(source: str) -> tuple[str, ...]:
    """Return every spelling of a `windbreak.riskkernel.signing` import found.

    A plain `import windbreak.riskkernel.signing`, an absolute
    `from windbreak.riskkernel import signing` (module-plus-symbol), and an
    absolute `from windbreak.riskkernel.signing import X` (module itself) are
    all flagged. Every *relative* import (`node.level > 0`) is also flagged,
    conservatively: a `..`-hop could reach the forbidden module, and the
    codebase's own convention is absolute imports throughout (verified: zero
    relative imports ship anywhere in `windbreak/` today).

    Args:
        source: Python source text to parse.

    Returns:
        The offending dotted names (or `.`-prefixed relative-import
        spellings) found, in AST-traversal order.
    """
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _FORBIDDEN_SIGNING_MODULE or alias.name.startswith(
                    _FORBIDDEN_SIGNING_MODULE + "."
                ):
                    found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                found.append("." * node.level + (node.module or ""))
                continue
            if node.module is None:
                continue
            for alias in node.names:
                combined = f"{node.module}.{alias.name}"
                if (
                    node.module == _FORBIDDEN_SIGNING_MODULE
                    or combined == _FORBIDDEN_SIGNING_MODULE
                ):
                    found.append(combined)
    return tuple(found)


def test_zero_forbidden_signing_imports_outside_riskkernel_package() -> None:
    """Every shipped `windbreak/**/*.py` module outside `riskkernel/` itself
    imports across the signing-key boundary cleanly -- zero hits.
    """
    violations: list[str] = []
    for path in sorted(_WINDBREAK_PACKAGE_DIR.rglob("*.py")):
        if _RISKKERNEL_PACKAGE_DIR in path.parents:
            continue
        source = path.read_text(encoding="utf-8")
        violations.extend(
            f"{path.relative_to(_REPO_ROOT)}:{name}"
            for name in _find_signing_imports(source)
        )

    assert violations == []


@pytest.mark.parametrize(
    "source",
    [
        "import windbreak.riskkernel.signing\n",
        "from windbreak.riskkernel import signing\n",
        "from windbreak.riskkernel.signing import SigningKeyHandle\n",
        "from . import signing\n",
        "from ..riskkernel import signing\n",
    ],
)
def test_ast_checker_flags_each_seeded_signing_import(source: str) -> None:
    """Each seeded synthetic source string is flagged as a forbidden import."""
    violations = _find_signing_imports(source)

    assert violations, f"expected a violation for: {source!r}"


def test_ast_checker_does_not_flag_an_unrelated_riskkernel_import() -> None:
    """Importing another `riskkernel` submodule (e.g. `checks`) is not itself
    a signing-key-boundary violation.
    """
    source = "from windbreak.riskkernel import checks\n"

    assert _find_signing_imports(source) == ()


# --- .importlinter: the config-file counterpart of the AST check ---------------


def test_importlinter_declares_signing_key_isolation_contract() -> None:
    """`.importlinter` names a contract forbidding `riskkernel.signing`
    imports, so the CI-enforced `import-linter` gate (wired via
    scripts/check-all.sh, issue #91) enforces the same boundary the AST
    checker above already enforces at the pytest layer.
    """
    parser = configparser.ConfigParser()
    read_files = parser.read(_IMPORTLINTER_PATH, encoding="utf-8")

    assert read_files, f"could not read {_IMPORTLINTER_PATH}"
    assert parser.has_section(_SIGNING_CONTRACT_SECTION)
    forbidden_modules = parser.get(
        _SIGNING_CONTRACT_SECTION, "forbidden_modules", fallback=""
    ).split()
    assert _FORBIDDEN_SIGNING_MODULE in forbidden_modules


# --- KernelLedgerWriter trio -----------------------------------------------------


def _accepts_kernel_ledger_writer(writer: KernelLedgerWriter) -> KernelLedgerWriter:
    """Identity helper whose signature pins `KernelLedgerWriter` as the type
    both concrete writers below must satisfy.

    Args:
        writer: Any object structurally satisfying `KernelLedgerWriter`.

    Returns:
        `writer`, unchanged.
    """
    return writer


def test_in_memory_and_logging_writers_satisfy_kernel_ledger_writer() -> None:
    """Both concrete writers are usable wherever `KernelLedgerWriter` is
    expected, and both expose a callable `.record`.
    """
    in_memory = _accepts_kernel_ledger_writer(InMemoryKernelLedgerWriter())
    logging_writer = _accepts_kernel_ledger_writer(LoggingKernelLedgerWriter())

    assert callable(in_memory.record)
    assert callable(logging_writer.record)


def test_in_memory_kernel_ledger_writer_records_events_in_order() -> None:
    """`InMemoryKernelLedgerWriter` retains every recorded event, in order,
    for direct test assertions.
    """
    writer = InMemoryKernelLedgerWriter()
    event_one = Event(
        event_type="Test",
        component="riskkernel",
        payload_schema_version=1,
        payload={"n": 1},
    )
    event_two = Event(
        event_type="Test",
        component="riskkernel",
        payload_schema_version=1,
        payload={"n": 2},
    )

    writer.record(event_one)
    writer.record(event_two)

    assert writer.events == [event_one, event_two]


def test_logging_kernel_ledger_writer_record_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`LoggingKernelLedgerWriter.record` logs the event rather than raising,
    and its emitted line names the recorded event type.
    """
    writer = LoggingKernelLedgerWriter()
    event = Event(
        event_type="Test",
        component="riskkernel",
        payload_schema_version=1,
        payload={},
    )
    caplog.set_level(logging.INFO)

    writer.record(event)

    assert any("Test" in record.message for record in caplog.records)


# --- PersistingKernelLedgerWriter + from_events kill wiring (issue #235) --------


def test_persisting_kernel_ledger_writer_appends_the_event_to_the_wrapped_store(
    tmp_path: Path,
) -> None:
    """`PersistingKernelLedgerWriter.record` is a thin `LedgerStore.append`
    passthrough: after recording one event, the wrapped store's `read_all()`
    shows exactly one row carrying that event's type and component -- proof
    the writer actually persists, unlike `LoggingKernelLedgerWriter`, which
    only logs.
    """
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        writer = PersistingKernelLedgerWriter(store)
        event = Event(
            event_type="Test",
            component="riskkernel",
            payload_schema_version=1,
            payload={"n": 1},
        )

        writer.record(event)

        records = store.read_all()
    finally:
        store.close()
    assert len(records) == 1
    assert records[0].event_type == "Test"
    assert records[0].component == "riskkernel"
    assert records[0].payload_schema_version == 1


def test_from_events_kill_integration_keyword_reaches_init(tmp_path: Path) -> None:
    """`RiskKernel.from_events`'s `kill_integration` keyword actually reaches
    `__init__` rather than being silently dropped: with a `KILL` file already
    present in the watcher's state dir, running one beat over a kernel
    rebuilt via `from_events` polls that watcher and drives the kernel
    `KILLED` -- proof the integration was truly wired, not merely accepted
    and discarded.
    """
    writer = InMemoryKernelLedgerWriter()
    machine = ModeStateMachine(mode_ceiling=Mode.LIVE)
    switch = KillSwitch(
        machine,
        writer,
        AlertDispatcher(sinks=[], ledger_writer=LoggingLedgerWriter()),
        state_dir=tmp_path,
    )
    watcher = KillFileWatcher(switch, tmp_path)
    integration = KillIntegration(switch=switch, watcher=watcher)
    (tmp_path / KILL_FILENAME).write_text("", encoding="utf-8")

    rebuilt = RiskKernel.from_events(
        [], writer, mode_machine=machine, kill_integration=integration
    )
    rebuilt.run(max_beats=1, heartbeat_interval=0)

    assert rebuilt.mode is Mode.KILLED


# --- RiskKernel: bounded heartbeat loop -----------------------------------------


@pytest.mark.timeout(30)
def test_risk_kernel_heartbeat_loop_emits_exactly_max_beats_events() -> None:
    """`RiskKernel.for_testing().run(max_beats=3, ...)` records exactly 3
    `ModeHeartbeat` events, monotonically numbered from 1, then returns
    (terminates) -- never an unbounded loop.
    """
    kernel = RiskKernel.for_testing()

    kernel.run(max_beats=3, heartbeat_interval=0)

    heartbeat_events = [
        event
        for event in kernel.ledger_writer.events
        if event.event_type == "ModeHeartbeat"
    ]
    assert len(heartbeat_events) == 3
    assert [event.payload["beat"] for event in heartbeat_events] == [1, 2, 3]
    assert all(event.component == "riskkernel" for event in heartbeat_events)
    assert all(
        event.payload["mode"] in {mode.name for mode in Mode}
        for event in heartbeat_events
    )


@pytest.mark.timeout(30)
def test_risk_kernel_heartbeat_loop_with_zero_max_beats_emits_nothing() -> None:
    """`max_beats=0` is the boundary case: the loop terminates immediately,
    recording no heartbeat events at all.
    """
    kernel = RiskKernel.for_testing()

    kernel.run(max_beats=0, heartbeat_interval=0)

    heartbeat_events = [
        event
        for event in kernel.ledger_writer.events
        if event.event_type == "ModeHeartbeat"
    ]
    assert heartbeat_events == []


# --- RiskKernel: ledgered evaluate_intent ---------------------------------------


@pytest.mark.timeout(30)
def test_risk_kernel_evaluate_intent_records_one_intent_vetoed_event() -> None:
    """`RiskKernel.evaluate_intent` records exactly one `IntentVetoed` event
    (component "riskkernel", schema version 1, payload carrying the intent id
    and reasons) and returns a `Decision` marked both vetoed and ledgered.

    The veto is sourced from a real dimension -- an unknown jurisdiction --
    because since issue #340 a fully-permissive context approves outright.
    """
    kernel = RiskKernel.for_testing()
    intent = _make_intent()
    context = make_context()

    decision = kernel.evaluate_intent(intent, context)

    assert decision.vetoed is True
    assert decision.ledgered is True

    vetoed_events = [
        event
        for event in kernel.ledger_writer.events
        if event.event_type == "IntentVetoed"
    ]
    assert len(vetoed_events) == 1
    event = vetoed_events[0]
    assert event.component == "riskkernel"
    assert event.payload_schema_version == 1
    assert event.payload["intent_id"] == intent.intent_id
    assert list(event.payload["reasons"]) == list(decision.reasons)


@pytest.mark.timeout(30)
def test_risk_kernel_evaluate_intent_records_intent_approved_when_not_vetoed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the check pipeline approves, the kernel ledgers one `IntentApproved`.

    Exactly one event, never a mislabeled `IntentVetoed`. Since issue #340
    promoted the last SPEC S10.3 check, a permissive context genuinely reaches
    this branch, so the approving path is exercised end-to-end rather than
    through a stubbed pipeline.
    """
    from windbreak.riskkernel import checks as checks_module

    approved = checks_module.Decision(vetoed=False, reasons=())
    monkeypatch.setattr(
        checks_module, "evaluate_intent", lambda intent, context: approved
    )

    kernel = RiskKernel.for_testing()
    intent = _make_intent()
    context = make_context()

    decision = kernel.evaluate_intent(intent, context)

    assert decision.vetoed is False
    assert decision.ledgered is True

    event_types = [event.event_type for event in kernel.ledger_writer.events]
    assert "IntentVetoed" not in event_types
    approved_events = [
        event
        for event in kernel.ledger_writer.events
        if event.event_type == "IntentApproved"
    ]
    assert len(approved_events) == 1
    event = approved_events[0]
    assert event.component == "riskkernel"
    assert event.payload_schema_version == 1
    assert event.payload["intent_id"] == intent.intent_id
    assert list(event.payload["reasons"]) == []


@pytest.mark.timeout(30)
def test_risk_kernel_evaluate_intent_stamps_its_own_mode_onto_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`RiskKernel.evaluate_intent` overrides `context.mode` with its own
    `ModeStateMachine.mode` (via `dataclasses.replace`) before evaluating --
    a caller-supplied `context.mode` is never trusted over the kernel's own
    tracked mode, and the caller's original context object is left untouched.
    """
    from windbreak.riskkernel import checks as checks_module

    captured_contexts: list[EvaluationContext] = []

    def _spy(intent: OrderIntent, context: EvaluationContext) -> checks_module.Decision:
        captured_contexts.append(context)
        return checks_module.Decision(vetoed=False, reasons=())

    monkeypatch.setattr(checks_module, "evaluate_intent", _spy)

    mode_machine = ModeStateMachine(mode_ceiling=Mode.LIVE, mode=Mode.LIVE)
    kernel = RiskKernel(InMemoryKernelLedgerWriter(), mode_machine=mode_machine)
    intent = _make_intent()
    caller_context = make_context(mode=Mode.RESEARCH)

    kernel.evaluate_intent(intent, caller_context)

    assert len(captured_contexts) == 1
    assert captured_contexts[0].mode == Mode.LIVE
    assert caller_context.mode == Mode.RESEARCH


# --- Process B survives Process A ------------------------------------------------


@pytest.mark.timeout(30)
def test_process_b_kernel_survives_process_a_pipeline_termination() -> None:
    """Killing the pipeline (Process A) subprocess does not affect an
    independently driven, in-process Risk Kernel (Process B) heartbeat loop --
    the two processes are isolated, per SPEC S5.1's process boundary.

    All waits below are bounded (hard `timeout=`), never an open-ended loop.
    """
    process_a = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "windbreak",
            "run",
            "--process",
            "pipeline",
            "--heartbeat-interval",
            "0.1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Expected: no --max-beats, so process A is still running after 1s.
        with contextlib.suppress(subprocess.TimeoutExpired):
            process_a.wait(timeout=1)
        process_a.terminate()
        process_a.wait(timeout=10)
    finally:
        if process_a.poll() is None:
            process_a.kill()
            process_a.wait(timeout=10)

    assert process_a.returncode is not None

    kernel = RiskKernel.for_testing()
    kernel.run(max_beats=3, heartbeat_interval=0)

    heartbeat_events = [
        event
        for event in kernel.ledger_writer.events
        if event.event_type == "ModeHeartbeat"
    ]
    assert len(heartbeat_events) == 3


# --- process.main(): bounded CLI, matching windbreak.main's conventions ----------


def test_process_main_returns_zero_for_a_bounded_run() -> None:
    """`process.main` with `--max-beats`/`--heartbeat-interval` exits 0."""
    exit_code = riskkernel_main(["--max-beats", "2", "--heartbeat-interval", "0"])

    assert exit_code == 0


def test_process_main_rejects_negative_max_beats() -> None:
    """A negative `--max-beats` is an argparse usage error (exit code 2),
    matching `windbreak.main`'s non-negative-int parsing convention.
    """
    with pytest.raises(SystemExit) as exc_info:
        riskkernel_main(["--max-beats", "-1"])

    assert exc_info.value.code == 2


def test_process_main_rejects_negative_heartbeat_interval() -> None:
    """A negative `--heartbeat-interval` is likewise an argparse usage error."""
    with pytest.raises(SystemExit) as exc_info:
        riskkernel_main(["--heartbeat-interval", "-1"])

    assert exc_info.value.code == 2


# --- __main__.py + subprocess smoke ----------------------------------------------


def test_riskkernel_dunder_main_module_imports_cleanly() -> None:
    """`python -m windbreak.riskkernel`'s entry module imports without error,
    for in-process coverage of the delegation to `process.main`.
    """
    module = importlib.import_module("windbreak.riskkernel.__main__")

    assert module is not None


@pytest.mark.timeout(30)
def test_riskkernel_module_invocation_smoke_via_subprocess() -> None:
    """`python -m windbreak.riskkernel --max-beats 2 --heartbeat-interval 0`
    exits 0 and logs at least one heartbeat line as JSON on stderr.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "windbreak.riskkernel",
            "--max-beats",
            "2",
            "--heartbeat-interval",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    lines = [line for line in result.stderr.splitlines() if line]
    payloads = [json.loads(line) for line in lines]
    assert any("heartbeat" in payload.get("msg", "").lower() for payload in payloads)


# --- SigningKeyHandle: real HMAC-SHA256 signing, no key material leakage -------


#: A fixed, valid (>=32-byte) key for the known-answer HMAC test below.
_KNOWN_KEY_MATERIAL = b"k" * 32


def test_signing_key_handle_sign_returns_32_byte_hmac_sha256_of_payload() -> None:
    """`SigningKeyHandle.sign` returns the exact 32-byte HMAC-SHA256 digest of
    the payload under the handle's key (issue #31) -- a known-answer test
    computed independently here, so a serialization- or algorithm-drift
    mutant changes the output and fails this test, not just "returns bytes".
    """
    handle = SigningKeyHandle(_KNOWN_KEY_MATERIAL)
    payload = b"payload"

    signature = handle.sign(payload)

    assert signature == hmac.new(_KNOWN_KEY_MATERIAL, payload, hashlib.sha256).digest()
    assert len(signature) == 32


def test_signing_key_handle_exposes_no_key_byte_attribute() -> None:
    """No public, non-callable attribute on a `SigningKeyHandle` instance
    ever holds raw key bytes -- the key is stored only as the private
    `_key`, never as a public attribute (issue #31).
    """
    handle = SigningKeyHandle(_KNOWN_KEY_MATERIAL)
    public_attribute_names = [name for name in dir(handle) if not name.startswith("_")]
    assert "sign" in public_attribute_names

    for name in public_attribute_names:
        value = getattr(handle, name)
        if callable(value):
            continue
        key_shaped = isinstance(value, (bytes, bytearray))
        assert not key_shaped, f"{name!r} exposes raw key-shaped bytes"
