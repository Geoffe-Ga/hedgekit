"""Fixtures for the end-to-end tier (issue #466, epic #465).

Thin by design: the behaviour lives in :mod:`tests.e2e.harness` so test modules
can import helpers directly instead of reaching into a ``conftest``. What is
here is exactly what needs pytest's lifecycle: isolated run roots and
guaranteed process teardown.

Runtime gating is deliberately NOT here. `conftest` fixtures are visible only
within their own directory, and the `container` tests live in `tests/deploy/`,
so a `requires_docker` fixture defined here could never be consumed by them.
The reusable half is :func:`tests.e2e.harness.docker_skip_reason` and
:func:`~tests.e2e.harness.systemd_skip_reason`, which any suite can import.
Raised in review of PR #477, where the unused fixtures had been added anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.e2e.harness import ProcessLauncher, RunRoot

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
