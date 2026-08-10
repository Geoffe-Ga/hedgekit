"""The deployment files are checked against the real CLI parser (#471, epic #465).

`deploy/` ships two ways to start the same four processes: four systemd units
and four compose services. Until now nothing type-checked either argv against
`windbreak.main.build_parser()`, which leaves two failure modes open.

**Flag drift.** A flag renamed or removed in the CLI leaves both deployment
paths silently wrong. This one fails *open*: a bare `windbreak run` is valid,
so the process starts happily and does the wrong thing. Nothing goes red.
Issue #446 is a live instance -- no deployment path passes the PAPER flags, so
the shipped deployment runs bare RESEARCH heartbeats.

**Path divergence.** systemd and compose can drift from *each other*, so the
documented deployment behaves differently depending on which one an operator
used. Neither had any guard at all.

Both are closed here by encoding the intended contract ONCE, in
:data:`_DEPLOYMENT_CONTRACT`, and checking both paths against it.

ON THE KNOWN GAP

:func:`test_no_deployment_path_yet_passes_the_paper_flags` pins #446's absence
rather than asserting the fixed state, because a test that is red on `main` is
not a guard -- it is noise that trains people to ignore the suite. It is
written to FAIL the moment #446 lands, which forces the contract above to be
updated in the same change that fixes the deployment. The gap cannot be closed
quietly and it cannot be forgotten.
"""

from __future__ import annotations

import contextlib
import io
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.e2e.harness import RUNTIME_PROBE_TIMEOUT_SECONDS, systemd_skip_reason
from windbreak.main import PROCESS_CHOICES, build_parser

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator

#: Repo root, derived from this file's own location
#: (`<root>/tests/deploy/test_deployment_cli_contract.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]

_SYSTEMD_DIR = _REPO_ROOT / "deploy" / "systemd"
_COMPOSE_PATH = _REPO_ROOT / "deploy" / "docker-compose.yml"

#: The console-script name both deployment paths invoke. Argv extraction keeps
#: everything after this token, so `/usr/bin/env windbreak run ...` and a bare
#: `["windbreak", "run", ...]` reduce to the same argument list.
_ENTRY_POINT = "windbreak"

#: THE DEPLOYMENT CONTRACT -- one source of truth, checked against BOTH paths.
#:
#: Maps each SPEC process token to the argv tail every deployment path must
#: pass. Compose service names are hyphenated by convention while the CLI token
#: is underscored (`order-gateway` vs `order_gateway`), which is exactly the
#: kind of near-miss this table exists to pin.
_DEPLOYMENT_CONTRACT: dict[str, tuple[str, ...]] = {
    "pipeline": ("run", "--process", "pipeline"),
    "riskkernel": ("run", "--process", "riskkernel"),
    "order_gateway": ("run", "--process", "order_gateway"),
    "dashboard": ("run", "--process", "dashboard"),
}

#: Maps each compose service name to its CLI process token.
_SERVICE_TO_TOKEN = {
    "pipeline": "pipeline",
    "riskkernel": "riskkernel",
    "order-gateway": "order_gateway",
    "dashboard": "dashboard",
}

#: Maps each systemd unit filename to its CLI process token.
_UNIT_TO_TOKEN = {
    "windbreak-pipeline.service": "pipeline",
    "windbreak-riskkernel.service": "riskkernel",
    "windbreak-order-gateway.service": "order_gateway",
    "windbreak-dashboard.service": "dashboard",
}

#: The flags the always-on PAPER loop needs. Issue #446 reports that no
#: deployment path passes any of them, so the shipped deployment runs bare
#: RESEARCH heartbeats. Named here so the gap is explicit and self-expiring.
_PAPER_FLAGS = ("--paper-books-dir", "--cassette-path", "--ledger-path", "--report-dir")


def _read_exec_start(unit_path: Path) -> str:
    """Extract a unit's single ``ExecStart=`` value.

    Args:
        unit_path: Path to the systemd unit file.

    Returns:
        The verbatim right-hand side of the ``ExecStart=`` assignment.

    Raises:
        AssertionError: If the unit has no ``ExecStart=`` line, or more than one.
    """
    lines = [
        line.strip()
        for line in unit_path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("ExecStart=")
    ]
    assert len(lines) == 1, (
        f"{unit_path.name} must declare exactly one ExecStart=, found {len(lines)}"
    )
    return lines[0].split("=", 1)[1].strip()


def _argv_tail(tokens: list[str], *, source: str) -> list[str]:
    """Reduce a full command to the arguments the windbreak CLI receives.

    Args:
        tokens: The full argv, possibly prefixed by ``/usr/bin/env`` or similar.
        source: Human-readable origin, quoted in the failure message.

    Returns:
        Every token after the ``windbreak`` entry point.

    Raises:
        AssertionError: If the command never invokes the entry point.
    """
    for index, token in enumerate(tokens):
        if Path(token).name == _ENTRY_POINT:
            return tokens[index + 1 :]
    message = f"{source} does not invoke the {_ENTRY_POINT!r} entry point: {tokens}"
    raise AssertionError(message)


def systemd_argv_tails() -> dict[str, list[str]]:
    """Read each systemd unit's windbreak arguments.

    Returns:
        Process token mapped to the argv tail its unit passes.
    """
    tails: dict[str, list[str]] = {}
    for unit_name, token in _UNIT_TO_TOKEN.items():
        exec_start = _read_exec_start(_SYSTEMD_DIR / unit_name)
        tails[token] = _argv_tail(shlex.split(exec_start), source=unit_name)
    return tails


def compose_argv_tails() -> dict[str, list[str]]:
    """Read each compose service's windbreak arguments.

    Returns:
        Process token mapped to the argv tail its service passes.

    Raises:
        AssertionError: If a service declares no ``command``.
    """
    document: dict[str, Any] = yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))
    services: dict[str, Any] = document["services"]
    tails: dict[str, list[str]] = {}
    for service_name, token in _SERVICE_TO_TOKEN.items():
        command = services[service_name].get("command")
        assert command is not None, (
            f"compose service {service_name!r} declares no command, so its "
            "process token cannot be checked against the CLI"
        )
        tokens = shlex.split(command) if isinstance(command, str) else list(command)
        tails[token] = _argv_tail(tokens, source=f"compose service {service_name!r}")
    return tails


def parse_with_real_cli(tail: list[str], *, source: str) -> argparse.Namespace:
    """Parse a deployment argv tail with the CLI's own parser.

    Using the real parser is the entire point: a hand-maintained list of valid
    flags would drift from the CLI exactly as the deployment files did.

    Args:
        tail: Arguments as the deployment path passes them.
        source: Human-readable origin, quoted in the failure message.

    Returns:
        The parsed namespace.

    Raises:
        AssertionError: If the real parser rejects the arguments.
    """
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            return build_parser().parse_args(tail)
    except SystemExit as exit_signal:
        message = (
            f"{source} passes arguments the windbreak CLI rejects: "
            f"{shlex.join(tail)}\n{stderr.getvalue().strip()}"
        )
        raise AssertionError(message) from exit_signal


@pytest.mark.parametrize("token", sorted(_DEPLOYMENT_CONTRACT))
def test_systemd_units_match_the_deployment_contract(token: str) -> None:
    """Each systemd unit passes exactly the contract's argv for its process.

    Args:
        token: The SPEC process token under test.
    """
    tail = systemd_argv_tails()[token]

    assert tuple(tail) == _DEPLOYMENT_CONTRACT[token]


@pytest.mark.parametrize("token", sorted(_DEPLOYMENT_CONTRACT))
def test_compose_services_match_the_deployment_contract(token: str) -> None:
    """Each compose service passes exactly the contract's argv for its process.

    Args:
        token: The SPEC process token under test.
    """
    tail = compose_argv_tails()[token]

    assert tuple(tail) == _DEPLOYMENT_CONTRACT[token]


@pytest.mark.parametrize("token", sorted(_DEPLOYMENT_CONTRACT))
def test_every_deployment_argv_parses_with_the_real_cli(token: str) -> None:
    """Both deployment paths pass arguments the real CLI accepts.

    This is the anti-drift guard. A flag renamed in `windbreak.main` without
    updating `deploy/` fails here, at the moment it is written, instead of at
    `systemctl start` on an operator's machine.

    Args:
        token: The SPEC process token under test.
    """
    for source, tails in (
        ("systemd unit", systemd_argv_tails()),
        ("compose service", compose_argv_tails()),
    ):
        namespace = parse_with_real_cli(tails[token], source=f"{source} for {token}")

        assert namespace.process == token


def test_the_two_deployment_paths_do_not_drift_from_each_other() -> None:
    """systemd and compose pass identical arguments for every process.

    Without this, the documented deployment behaves differently depending on
    which path an operator used -- a divergence neither file's own tests could
    ever see, because each was only ever read in isolation.
    """
    assert systemd_argv_tails() == compose_argv_tails()


def test_both_paths_cover_every_spec_process_exactly_once() -> None:
    """Each deployment path covers all four SPEC processes, once each.

    Catches a process silently dropped from a deployment path, and a token
    typo'd into a duplicate -- both of which leave a process simply not running.
    """
    expected = set(PROCESS_CHOICES)

    assert set(_DEPLOYMENT_CONTRACT) == expected
    assert set(systemd_argv_tails()) == expected
    assert set(compose_argv_tails()) == expected
    assert len(_UNIT_TO_TOKEN) == len(expected)
    assert len(_SERVICE_TO_TOKEN) == len(expected)


def test_no_deployment_path_yet_passes_the_paper_flags() -> None:
    """Pins issue #446's gap, and fails the moment #446 closes it.

    The shipped deployment starts bare RESEARCH heartbeats: no service or unit
    passes `--paper-books-dir`, `--cassette-path`, `--ledger-path` or
    `--report-dir`, so the always-on PAPER loop never runs.

    Asserting the *fixed* state instead would leave `main` red, which is not a
    guard -- it is noise. Asserting the *current* state makes the gap explicit
    and self-expiring: when #446 adds those flags this test fails, and the fix
    must update `_DEPLOYMENT_CONTRACT` in the same change. The gap cannot be
    closed quietly, and it cannot be forgotten.
    """
    all_arguments = {
        argument
        for tails in (systemd_argv_tails(), compose_argv_tails())
        for tail in tails.values()
        for argument in tail
    }

    present = sorted(flag for flag in _PAPER_FLAGS if flag in all_arguments)

    assert not present, (
        "A deployment path now passes PAPER flags "
        f"({present}), so issue #446 has been fixed. Update "
        "_DEPLOYMENT_CONTRACT to the new intended argv and delete this test."
    )


@pytest.fixture
def _systemd_runtime() -> Iterator[None]:
    """Skip the requesting test unless systemd is the running init system.

    Yields:
        ``None``, once the runtime has been confirmed present.
    """
    reason = systemd_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    yield


@pytest.mark.container
@pytest.mark.usefixtures("_systemd_runtime")
@pytest.mark.parametrize("unit_name", sorted(_UNIT_TO_TOKEN))
def test_systemd_analyze_accepts_each_unit(unit_name: str) -> None:
    """`systemd-analyze verify` accepts each shipped unit file.

    Text assertions cannot catch a misspelled directive, a bad ``Type=`` or an
    ``After=`` naming a unit that does not exist -- all of which pass a grep and
    fail at `systemctl start`. This runs the real validator.

    Args:
        unit_name: The unit file under test.
    """
    completed = subprocess.run(
        ["systemd-analyze", "verify", str(_SYSTEMD_DIR / unit_name)],
        capture_output=True,
        text=True,
        timeout=RUNTIME_PROBE_TIMEOUT_SECONDS,
        check=False,
    )

    assert completed.returncode == 0, (
        f"systemd-analyze rejected {unit_name}:\n{completed.stderr.strip()}"
    )
