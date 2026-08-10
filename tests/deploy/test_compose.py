"""Failing-first tests for the docker-compose skeleton (issue #15, RED).

`deploy/docker-compose.yml` does not exist yet, so every test in this module
fails at test-body time with `FileNotFoundError` when `_load_compose` tries
to open it. Once the deploy skeleton lands, these tests pin its structural
contract: exactly four services (one per SPEC process), each launching
`windbreak run --process <token>` with `restart: on-failure`, a shared named
`ledger` volume mounted read-write on `pipeline` and read-only on
`dashboard`, no top-level `networks` override, and every published port
bound to `127.0.0.1` only (never the implicit `0.0.0.0`).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from windbreak.main import DASHBOARD_AUTH_ENV_VAR, build_parser

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_PATH = _REPO_ROOT / "deploy" / "docker-compose.yml"

#: Maps each compose service name to its `--process` CLI token. Note
#: `order-gateway` (the compose/service naming convention, hyphenated) is
#: deliberately distinct from `order_gateway` (the CLI token, underscored).
_SERVICE_TO_CLI_TOKEN = {
    "pipeline": "pipeline",
    "riskkernel": "riskkernel",
    "order-gateway": "order_gateway",
    "dashboard": "dashboard",
}


def _load_compose() -> dict[str, Any]:
    """Parse `deploy/docker-compose.yml` with `yaml.safe_load`.

    Returns:
        The parsed top-level compose mapping.
    """
    with _COMPOSE_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _command_string(command: object) -> str:
    """Normalize a compose `command` (list or string form) to one string.

    Args:
        command: Either a shell-style string or a list of tokens.

    Returns:
        The command joined by single spaces if it was a list, else the
        string unchanged.
    """
    if isinstance(command, list):
        return " ".join(str(token) for token in command)
    return str(command)


def _ledger_volume_entries(service: dict[str, Any]) -> list[str]:
    """Return every volume mapping string on `service` that mounts `ledger`.

    Args:
        service: One service's parsed compose mapping.

    Returns:
        The subset of `service["volumes"]` entries mentioning the `ledger`
        named volume.
    """
    return [
        str(entry) for entry in service.get("volumes", []) if "ledger" in str(entry)
    ]


def test_services_are_exactly_the_four_spec_processes() -> None:
    """The compose file defines exactly pipeline/riskkernel/order-gateway/dashboard."""
    data = _load_compose()

    assert set(data["services"].keys()) == set(_SERVICE_TO_CLI_TOKEN)


@pytest.mark.parametrize(
    ("service_name", "cli_token"), sorted(_SERVICE_TO_CLI_TOKEN.items())
)
def test_service_command_runs_windbreak_with_matching_process_token(
    service_name: str, cli_token: str
) -> None:
    """Each service launches `windbreak run` for its own `--process` token.

    The command is parsed by the CLI's own parser rather than compared to a
    literal string, so the composition flags issue #446 added to `pipeline`
    (and any later flag) do not falsify the process-token claim this test is
    actually about. That the pipeline command still *activates* PAPER is
    tests/deploy/test_deployment_launches_paper.py's job.
    """
    data = _load_compose()

    tokens = shlex.split(_command_string(data["services"][service_name]["command"]))
    assert tokens[0] == "windbreak"
    assert tokens[1] == "run"
    assert build_parser().parse_args(tokens[1:]).process == cli_token


@pytest.mark.parametrize("service_name", sorted(_SERVICE_TO_CLI_TOKEN))
def test_service_restarts_on_failure(service_name: str) -> None:
    """Every service restarts `on-failure`, never `always` or unset."""
    data = _load_compose()

    service = data["services"][service_name]
    assert service["restart"] == "on-failure"


def test_top_level_volumes_declares_ledger() -> None:
    """The compose file declares a top-level named `ledger` volume."""
    data = _load_compose()

    assert "ledger" in data.get("volumes", {})


def test_pipeline_mounts_ledger_read_write() -> None:
    """`pipeline` needs write access to append to the ledger."""
    data = _load_compose()

    entries = _ledger_volume_entries(data["services"]["pipeline"])
    assert entries, "pipeline does not mount the ledger volume at all"
    assert not any(entry.endswith(":ro") for entry in entries)


def test_dashboard_mounts_ledger_read_only() -> None:
    """`dashboard` only ever reads the ledger, per SPEC S5.1's no-trade posture."""
    data = _load_compose()

    entries = _ledger_volume_entries(data["services"]["dashboard"])
    assert entries, "dashboard does not mount the ledger volume at all"
    assert all(entry.endswith(":ro") for entry in entries)


def test_no_top_level_networks_override() -> None:
    """No custom top-level `networks:` key -- services use the implicit default."""
    data = _load_compose()

    assert "networks" not in data


def test_every_published_port_binds_loopback_only() -> None:
    """Every `ports:` entry, across every service, is bound to `127.0.0.1`.

    A bare `"8080:8080"` mapping binds `0.0.0.0` by default, which would
    expose the dashboard beyond localhost; every entry must instead read
    `"127.0.0.1:<host-port>:<container-port>"`.
    """
    data = _load_compose()

    for service_name, service in data["services"].items():
        for port_entry in service.get("ports", []):
            assert str(port_entry).startswith("127.0.0.1:"), (
                f"{service_name} publishes a port not bound to 127.0.0.1: "
                f"{port_entry!r}"
            )


def _docker_compose_available() -> bool:
    """Probe whether `docker compose` (the v2 CLI plugin) is usable here.

    Returns:
        True if the `docker` binary exists and `docker compose version`
        exits zero; False otherwise (missing binary, or the plugin absent).
    """
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker CLI or the docker compose v2 plugin is unavailable here",
)
@pytest.mark.timeout(60)
def test_docker_compose_config_validates() -> None:
    """`docker compose config` accepts the file as syntactically valid compose.

    Secondary to the always-on YAML structural tests above -- this only
    confirms Compose itself agrees the file is well-formed, and only runs
    when Compose happens to be installed in the executing environment.

    The dashboard token variable is supplied because the file declares it in
    Compose's required-variable form; that an *unset* token is refused is
    asserted separately, in tests/deploy/test_deployment_launches_paper.py.
    """
    result = subprocess.run(
        ["docker", "compose", "-f", str(_COMPOSE_PATH), "config"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, DASHBOARD_AUTH_ENV_VAR: "compose-config-probe"},
    )

    assert result.returncode == 0, result.stderr
