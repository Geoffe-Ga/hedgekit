"""The RUNBOOK's operator commands are executed, not just parsed (#472, #465).

`tests/docs/test_documented_commands.py` proves every documented command is
*accepted* by the real CLI. That closes the #449 class -- a flag that does not
exist -- but it says nothing about whether running the command does what the
RUNBOOK claims. This module runs them.

Scope is the operator's durable-state path, which is what an operator reaches
for when something is wrong and is therefore the worst place for a documented
command to be subtly untrue: `anchor`, `verify`, `kill`, `rearm`, `ack` and
`rebuild`. Each is invoked exactly as `RUNBOOK.md` writes it, as a real child
process, against a ledger a real `windbreak run` produced.

TWO THINGS THIS PINS THAT NOTHING ELSE DOES

* **`verify` can fail.** A verifier that returns 0 on a tampered ledger is
  worse than no verifier, so the tamper case is asserted alongside the happy
  path. Without it, `test_anchor_then_verify` would pass against a `verify`
  that unconditionally exited 0.
* **`rearm` reads stdin.** `RUNBOOK.md:38` documents `windbreak rearm
  --state-dir <dir>` as a plain command. It is not: `windbreak/main.py`'s
  `_run_rearm` calls :func:`input`, so the documented invocation fails with
  `EOFError` in any non-interactive context -- a runbook automation, a systemd
  unit, a script. That gap is pinned here rather than silently worked around.

Deliberately NOT covered: `preflight` and `drill`, which need fixture
directories whose construction is a larger piece of work than this issue, and
the README's compose transcripts, which need a running stack (#468, gated on
#445). Both are named so the omission is visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.e2e.harness import (
    read_ledger_records,
    run_windbreak,
)

if TYPE_CHECKING:
    from tests.e2e.harness import RunRoot

pytestmark = pytest.mark.e2e

#: Filename `windbreak kill` writes into the state directory.
_KILL_FILENAME = "KILL"

#: Filename `windbreak rearm` writes the typed phrase into.
_REARM_FILENAME = "REARM"

#: Directory `windbreak ack` writes approval files into.
_ACKS_DIRNAME = "acks"

#: A syntactically valid approval id: exactly 32 lowercase hex characters.
_APPROVAL_ID = "a1b2c3d4e5f60718293a4b5c6d7e8f90"


@pytest.fixture
def seeded_ledger(run_root: RunRoot) -> RunRoot:
    """Produce a real ledger by running the pipeline process for one beat.

    Uses the shipped CLI rather than constructing a store in-process, so every
    assertion downstream is about a ledger the product actually wrote.

    Args:
        run_root: The isolated run root for this test.

    Returns:
        The same run root, with ``ledger_path`` now populated.
    """
    completed = run_windbreak(
        "run",
        "--process",
        "pipeline",
        "--max-beats",
        "1",
        "--heartbeat-interval",
        "0.1",
        "--ledger-path",
        str(run_root.ledger_path),
        "--report-dir",
        str(run_root.report_dir),
        timeout=120.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert read_ledger_records(run_root.ledger_path), "the seed run wrote no rows"
    return run_root


def test_documented_anchor_then_verify_round_trip(seeded_ledger: RunRoot) -> None:
    """`windbreak anchor` then `windbreak verify` both succeed, as documented.

    `RUNBOOK.md:159` and `RUNBOOK.md:165`. This is the operator's evidence path:
    if it does not work, nothing downstream of it can be trusted.

    Args:
        seeded_ledger: A run root whose ledger a real run produced.
    """
    anchored = run_windbreak(
        "anchor",
        "--ledger-path",
        str(seeded_ledger.ledger_path),
        "--anchor-path",
        str(seeded_ledger.anchor_path),
    )

    assert anchored.returncode == 0, anchored.stderr
    assert seeded_ledger.anchor_path.read_text(encoding="utf-8").strip()

    verified = run_windbreak(
        "verify",
        "--ledger-path",
        str(seeded_ledger.ledger_path),
        "--anchor-path",
        str(seeded_ledger.anchor_path),
    )

    assert verified.returncode == 0, verified.stderr


def test_documented_verify_rejects_a_tampered_ledger(seeded_ledger: RunRoot) -> None:
    """`windbreak verify` exits non-zero once the chain no longer matches.

    The half that makes the happy path mean something. A `verify` that always
    exited 0 would satisfy the test above and detect nothing -- the exact shape
    of guard epic #465 exists to eliminate.

    Args:
        seeded_ledger: A run root whose ledger a real run produced.
    """
    run_windbreak(
        "anchor",
        "--ledger-path",
        str(seeded_ledger.ledger_path),
        "--anchor-path",
        str(seeded_ledger.anchor_path),
    )
    seeded_ledger.anchor_path.write_text("0" * 64 + " 1 tampered\n", encoding="utf-8")

    verified = run_windbreak(
        "verify",
        "--ledger-path",
        str(seeded_ledger.ledger_path),
        "--anchor-path",
        str(seeded_ledger.anchor_path),
    )

    assert verified.returncode != 0


def test_documented_kill_writes_the_durable_kill_file(run_root: RunRoot) -> None:
    """`windbreak kill --state-dir <dir>` succeeds and leaves a KILL file.

    `RUNBOOK.md:31`.

    Args:
        run_root: The isolated run root for this test.
    """
    completed = run_windbreak("kill", "--state-dir", str(run_root.state_dir))

    assert completed.returncode == 0, completed.stderr
    assert (run_root.state_dir / _KILL_FILENAME).exists()


def test_documented_rearm_writes_the_typed_phrase_byte_for_byte(
    run_root: RunRoot,
) -> None:
    """`windbreak rearm` writes stdin verbatim -- no strip, no case change.

    `_run_rearm`'s docstring states this explicitly, because the kernel compares
    the phrase byte-for-byte and any normalisation here would silently break
    re-arm. A claim that specific deserves a test that would notice.

    Args:
        run_root: The isolated run root for this test.
    """
    phrase = "  RE-ARM KILL 7: I Accept Full Responsibility  "

    completed = run_windbreak(
        "rearm",
        "--state-dir",
        str(run_root.state_dir),
        input_text=f"{phrase}\n",
    )

    assert completed.returncode == 0, completed.stderr
    written = (run_root.state_dir / _REARM_FILENAME).read_text(encoding="utf-8")

    assert written == phrase


def test_documented_rearm_invocation_fails_without_an_interactive_phrase(
    run_root: RunRoot,
) -> None:
    """The RUNBOOK's `rearm` line cannot be run non-interactively.

    Pins a real documentation gap. `RUNBOOK.md:38` prints
    `windbreak rearm --state-dir <dir>` as though it were a complete command,
    but `_run_rearm` calls :func:`input`, so with no stdin it dies with
    `EOFError`. Anyone scripting the documented line -- a runbook automation, a
    systemd unit, a recovery script -- gets a traceback during an incident.

    Asserting the current behaviour rather than the fixed behaviour keeps
    `main` green while making the gap impossible to lose: this test fails the
    moment `rearm` grows a non-interactive path, which forces the RUNBOOK to be
    updated in the same change.

    Args:
        run_root: The isolated run root for this test.
    """
    completed = run_windbreak("rearm", "--state-dir", str(run_root.state_dir))

    assert completed.returncode != 0
    assert "EOFError" in completed.stderr
    assert not (run_root.state_dir / _REARM_FILENAME).exists()


def test_documented_ack_writes_the_approval_file(run_root: RunRoot) -> None:
    """`windbreak ack --approval-id <id> --state-dir <dir>` succeeds.

    `RUNBOOK.md:46`. The id is a real 32-hex value, since the CLI validates the
    shape and the documented placeholder deliberately does not satisfy it.

    Args:
        run_root: The isolated run root for this test.
    """
    completed = run_windbreak(
        "ack",
        "--approval-id",
        _APPROVAL_ID,
        "--state-dir",
        str(run_root.state_dir),
    )

    assert completed.returncode == 0, completed.stderr
    assert (run_root.state_dir / _ACKS_DIRNAME / _APPROVAL_ID).exists()


def test_documented_rebuild_reads_the_ledger_and_writes_read_models(
    seeded_ledger: RunRoot,
) -> None:
    """`windbreak rebuild` succeeds against a real ledger and writes output.

    `RUNBOOK.md:140`.

    Args:
        seeded_ledger: A run root whose ledger a real run produced.
    """
    output_dir = seeded_ledger.root / "read-models"

    completed = run_windbreak(
        "rebuild",
        "--ledger-path",
        str(seeded_ledger.ledger_path),
        "--output-dir",
        str(output_dir),
    )

    assert completed.returncode == 0, completed.stderr
    assert output_dir.is_dir()
    assert any(output_dir.iterdir()), "rebuild produced no read models"
