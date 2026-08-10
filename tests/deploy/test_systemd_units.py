"""Failing-first tests for the systemd unit skeletons (issue #15, RED).

`deploy/systemd/` does not exist yet, so every parametrized case below fails
at test-body time with `FileNotFoundError` (surfaced by `configparser` as it
tries to read a nonexistent unit file). Once the four unit files land, these
tests pin their shared contract: each declares `[Unit]`, `[Service]`, and
`[Install]` sections, restarts `on-failure`, and its `ExecStart` invokes
`windbreak run --process <token>` for that unit's process -- the same
underscored `order_gateway` token used by the CLI and the compose skeleton.
"""

from __future__ import annotations

import configparser
import shlex
from pathlib import Path

import pytest

from windbreak.main import build_parser

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYSTEMD_DIR = _REPO_ROOT / "deploy" / "systemd"

#: Maps each expected unit filename to its `--process` CLI token.
_UNIT_TO_CLI_TOKEN = {
    "windbreak-pipeline.service": "pipeline",
    "windbreak-riskkernel.service": "riskkernel",
    "windbreak-order-gateway.service": "order_gateway",
    "windbreak-dashboard.service": "dashboard",
}


def _parse_unit(filename: str) -> configparser.ConfigParser:
    """Parse a systemd unit file, preserving its case-sensitive key names.

    Args:
        filename: The unit file's name under `deploy/systemd/`.

    Returns:
        A `ConfigParser` with `optionxform` disabled so `ExecStart` is not
        lowercased to `execstart` -- systemd unit keys are case-sensitive.

    Raises:
        FileNotFoundError: If the unit file does not exist under
            `deploy/systemd/` (the expected RED state before issue #15
            lands the deploy skeleton).
    """
    unit_path = _SYSTEMD_DIR / filename
    if not unit_path.is_file():
        raise FileNotFoundError(unit_path)
    parser = configparser.ConfigParser()
    parser.optionxform = str  # type: ignore[method-assign]
    parser.read(unit_path, encoding="utf-8")
    return parser


@pytest.mark.parametrize(("filename", "cli_token"), sorted(_UNIT_TO_CLI_TOKEN.items()))
def test_unit_declares_required_sections(filename: str, cli_token: str) -> None:
    """Every unit has the three sections systemd requires to be usable.

    `cli_token` is unused here (this test only checks section presence) but
    is threaded through so the parametrize id names the process, not just
    the filename.
    """
    del cli_token
    parser = _parse_unit(filename)

    assert parser.has_section("Unit")
    assert parser.has_section("Service")
    assert parser.has_section("Install")


@pytest.mark.parametrize(("filename", "cli_token"), sorted(_UNIT_TO_CLI_TOKEN.items()))
def test_unit_restarts_on_failure(filename: str, cli_token: str) -> None:
    """`Restart=on-failure` matches the compose skeleton's restart policy."""
    del cli_token
    parser = _parse_unit(filename)

    assert parser.get("Service", "Restart") == "on-failure"


@pytest.mark.parametrize(("filename", "cli_token"), sorted(_UNIT_TO_CLI_TOKEN.items()))
def test_unit_exec_start_runs_windbreak_with_matching_process_token(
    filename: str, cli_token: str
) -> None:
    """`ExecStart` launches `windbreak run` for this unit's `--process` token.

    The argument vector is parsed by the CLI's own parser rather than matched
    as a suffix, so the composition flags issue #446 added to the pipeline
    unit (and any later flag) do not falsify the process-token claim this test
    is actually about. That the pipeline unit still *activates* PAPER is
    tests/deploy/test_deployment_launches_paper.py's job.
    """
    parser = _parse_unit(filename)

    tokens = shlex.split(parser.get("Service", "ExecStart"))
    assert tokens[0] == "/usr/bin/env"
    assert tokens[1] == "windbreak"
    assert tokens[2] == "run"
    assert build_parser().parse_args(tokens[2:]).process == cli_token
