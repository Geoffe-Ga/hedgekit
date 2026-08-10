"""Fixtures for the end-to-end tier (issue #466, epic #465).

Thin by design: the behaviour lives in :mod:`tests.e2e.harness` so test modules
can import helpers directly instead of reaching into a ``conftest``. What is
here is exactly what needs pytest's lifecycle -- isolated run roots, guaranteed
process teardown, and the runtime gates that make a `container` test *skip*
rather than silently pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.e2e.harness import (
    ProcessLauncher,
    RunRoot,
    docker_skip_reason,
    systemd_skip_reason,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def run_root(tmp_path: Path) -> RunRoot:
    """Provide an isolated run root for one end-to-end run.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A :class:`RunRoot` whose directories already exist.
    """
    return RunRoot.create(tmp_path)


@pytest.fixture
def launcher(run_root: RunRoot) -> Iterator[ProcessLauncher]:
    """Provide a launcher that reaps every process it starts.

    The ``finally`` is the whole point: a test that fails mid-run, or that
    never reaches its own cleanup, still leaves no orphaned windbreak process.

    Args:
        run_root: The run root whose ``log_dir`` receives captured streams.

    Yields:
        A :class:`ProcessLauncher` bound to this test's run root.
    """
    process_launcher = ProcessLauncher(run_root.log_dir)
    try:
        yield process_launcher
    finally:
        process_launcher.reap_all()


@pytest.fixture
def requires_docker() -> None:
    """Skip the requesting test unless a docker daemon is reachable.

    Skips -- never passes -- when the runtime is absent, and names what is
    missing. A container assertion that quietly succeeds on a machine with no
    docker is precisely the kind of guard epic #465 exists to eliminate.
    """
    reason = docker_skip_reason()
    if reason is not None:
        pytest.skip(reason)


@pytest.fixture
def requires_systemd() -> None:
    """Skip the requesting test unless systemd is the running init system.

    Skips -- never passes -- when the runtime is absent, and names what is
    missing.
    """
    reason = systemd_skip_reason()
    if reason is not None:
        pytest.skip(reason)
