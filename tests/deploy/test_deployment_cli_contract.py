"""The deployment files are checked against the real CLI parser (#471, #465).

`deploy/` ships two ways to start the same four processes: four systemd units
and four compose services. Until #471 nothing type-checked either argv against
`windbreak.main.build_parser()`, which left two failure modes open.

**Flag drift.** A flag renamed or removed in the CLI leaves both deployment
paths silently wrong. This one fails *open*: a bare `windbreak run` is valid, so
the process starts happily and does the wrong thing. Issue #446 was a live
instance -- no deployment path passed the PAPER flags, so the shipped
deployment ran bare RESEARCH heartbeats.

**Path divergence.** systemd and compose can drift from *each other*, so the
documented deployment behaves differently depending on which one an operator
used. Neither file's own tests could see it, because each was only ever read in
isolation.

Both are closed by encoding the intended argv ONCE, in
:data:`_DEPLOYMENT_CONTRACT`, and checking both paths against it and against
each other.

RELATIONSHIP TO `tests/deploy/test_deployment_launches_paper.py`

That module (from #445/#446) asserts the *deployment* properties: build
contexts, mounts, volumes, dockerignore, token handling, restart limits. This
one asserts the *CLI contract*: that the argv both paths pass is accepted by the
real parser, that the two paths agree, and that every SPEC process is covered
once. The parsing helpers are shared -- this module imports
`tests.deploy.artifacts` rather than keeping a second compose/unit parser, since
two parsers that must agree is the same defect shape this file exists to catch.

WHAT CHANGED WHEN #446 LANDED

`test_no_deployment_path_yet_passes_the_paper_flags` used to pin the gap and was
written to fail the moment it closed. It did exactly that: #446 added the four
activation arguments, the test went red, and the contract below was updated in
the same change. The test is gone because the thing it guarded is gone -- which
is the whole point of a self-expiring pin.
"""

from __future__ import annotations

import contextlib
import io
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.deploy import artifacts
from tests.e2e.harness import (
    RUNTIME_PROBE_TIMEOUT_SECONDS,
    require_runtime,
    systemd_skip_reason,
)
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

#: The four arguments that activate the always-on PAPER loop (#446). Only the
#: pipeline process runs that loop, so only its argv carries them; the other
#: three are deliberately bare. `test_exactly_one_process_activates_the_paper_loop`
#: pins that asymmetry, so PAPER activation cannot be copy-pasted onto a process
#: that does not run the loop.
_PAPER_ACTIVATION = (
    "--paper-books-dir",
    "tests/fixtures/books/deep_walk",
    "--cassette-path",
    "tests/fixtures/forecast/cassettes.json",
    "--ledger-path",
    "/var/lib/windbreak/ledger/windbreak.db",
    "--report-dir",
    "/var/lib/windbreak/reports",
)

#: THE DEPLOYMENT CONTRACT -- one source of truth, checked against BOTH paths.
#:
#: Maps each SPEC process token to the argv tail every deployment path must
#: pass. Compose service names are hyphenated by convention while the CLI token
#: is underscored (`order-gateway` vs `order_gateway`), which is exactly the
#: kind of near-miss this table exists to pin.
_DEPLOYMENT_CONTRACT: dict[str, tuple[str, ...]] = {
    "pipeline": ("run", "--process", "pipeline", *_PAPER_ACTIVATION),
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

    Parsing is delegated to :mod:`tests.deploy.artifacts` so this module and
    `test_deployment_launches_paper.py` read the units through one parser.

    Returns:
        Process token mapped to the argv tail its unit passes.
    """
    tails: dict[str, list[str]] = {}
    for unit_path in artifacts.unit_paths():
        token = _UNIT_TO_TOKEN[unit_path.name]
        tokens = artifacts.unit_exec_start_tokens(artifacts.parse_unit(unit_path))
        tails[token] = _argv_tail(tokens, source=unit_path.name)
    return tails


def compose_argv_tails() -> dict[str, list[str]]:
    """Read each compose service's windbreak arguments.

    Returns:
        Process token mapped to the argv tail its service passes.
    """
    tails: dict[str, list[str]] = {}
    for service_name, service in artifacts.compose_services().items():
        token = _SERVICE_TO_TOKEN[service_name]
        tokens = artifacts.command_tokens(service)
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


def test_exactly_one_process_activates_the_paper_loop() -> None:
    """Only the pipeline process carries PAPER activation, in both paths.

    Replaces the self-expiring pin that guarded #446's gap. That pin fired when
    #446 landed and was deleted; this is its successor, guarding the *fixed*
    state rather than the absence.

    The asymmetry is the point: `--paper-books-dir` and friends belong on the
    process that actually runs the loop. Copy-pasting them onto `riskkernel` or
    `dashboard` would look like thoroughness and would be wrong.
    """
    for source, tails in (
        ("systemd", systemd_argv_tails()),
        ("compose", compose_argv_tails()),
    ):
        activating = sorted(
            token
            for token, tail in tails.items()
            if any(flag in tail for flag in _PAPER_ACTIVATION[::2])
        )

        assert activating == ["pipeline"], (
            f"{source}: expected only 'pipeline' to activate the PAPER loop, "
            f"got {activating}"
        )


@pytest.fixture
def _systemd_runtime() -> Iterator[None]:
    """Gate the requesting test on systemd being the running init system.

    Routed through :func:`tests.e2e.harness.require_runtime` rather than
    calling `pytest.skip` directly, and the difference is the whole point.
    These tests carry the `container` marker, so their only CI home is the
    container job -- a REQUIRED status check, which sets
    ``WINDBREAK_E2E_REQUIRE_RUNTIME=1``. A required check that skips reports
    success. Calling `pytest.skip` here meant `systemd-analyze verify` over the
    four shipped units would vanish silently the day a runner stopped running
    systemd, and the gate would go on reporting green over four fewer
    assertions.

    That was latent, not theoretical: `harness.py` documents the fail-closed
    flag as making it "impossible" for any gate in the tier to skip, and this
    fixture was the counterexample that made the claim false. One call site,
    one universal quantifier restored.

    Yields:
        ``None``, once the runtime has been confirmed present.
    """
    require_runtime(systemd_skip_reason())
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
