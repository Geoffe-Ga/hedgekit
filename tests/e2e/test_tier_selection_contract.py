"""The `container` tier stays deselected wherever a script picks markers (#468).

`pyproject.toml` deselects the `container` tier through `addopts` with
``-m "not container"``, and every comment in this repository that mentions the
tier -- pyproject's own, and `ci.yml`'s container job -- relies on that holding
everywhere except the one job that opts in.

It does not hold by itself. **pytest takes the LAST `-m` it is given**, so any
script that passes its own marker expression REPLACES the deselection instead
of narrowing it. `scripts/test.sh --unit` did exactly that, and Gate 1 was
therefore one `container`-marked test away from running a `docker build` and
five `docker compose up` cycles on every developer machine, and three more
times across CI's `quality` matrix. Nothing noticed, because until issue #468
the tier held a single test that skipped for want of systemd -- a deselection
contract that could not fail was believed rather than enforced.

This module enforces it, by reading the script rather than restating what it
ought to say. It is deliberately unmarked: a guard that lived inside the tier
it guards would be deselected by the very bug it exists to catch.

The second half of the module is the same argument pointed at CI. The tier's
fail-closed behaviour (`tests.e2e.harness.require_runtime`) is entirely inert
unless the container job actually sets `WINDBREAK_E2E_REQUIRE_RUNTIME=1`; every
test of the mechanism would stay green with the wiring deleted, which is this
repository's most common trap -- every seam tested, the composition unreachable.
So the workflow file is read and the wiring asserted, from the same constants
the harness uses.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from tests.e2e.harness import (
    REQUIRE_RUNTIME_ENABLED_VALUE,
    REQUIRE_RUNTIME_ENV_VAR,
)

#: The repository root, two levels up from `tests/e2e/`.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The gate script whose marker selections are under test.
TEST_SCRIPT = REPO_ROOT / "scripts" / "test.sh"

#: The workflow whose container job is the tier's only CI entry point.
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: Key of the container-tier job inside that workflow's `jobs:` mapping.
CONTAINER_JOB_ID = "container"

#: Matches a `-m "<expression>"` argument in the script's pytest invocation.
_MARKER_ARGUMENT = re.compile(r'-m\s+"([^"]*)"')

#: The marker whose deselection every explicit selection must preserve.
GATED_MARKER = "container"


def _marker_expressions() -> list[str]:
    """Extract every `-m` marker expression `scripts/test.sh` passes to pytest.

    Returns:
        The quoted expressions, in the order they appear in the script.
    """
    return _MARKER_ARGUMENT.findall(TEST_SCRIPT.read_text(encoding="utf-8"))


def test_the_test_script_declares_at_least_one_marker_selection() -> None:
    """The extraction finds something, so the assertion below can fail.

    Trap #7 in this repository's own list: a corpus scan asserting over zero
    hits passes forever. If `scripts/test.sh` is ever restructured so this
    regex matches nothing, the guard underneath would go green while checking
    an empty set -- exactly the failure mode it exists to prevent.
    """
    expressions = _marker_expressions()

    assert expressions, (
        f'no `-m "..."` selections found in {TEST_SCRIPT}; the extraction is '
        "broken, so the deselection guard below is checking nothing"
    )
    assert len(expressions) >= 2, (
        f"expected the unit and integration selections, found {expressions}"
    )


def test_every_marker_selection_keeps_the_container_tier_deselected() -> None:
    """No `-m` in `scripts/test.sh` may re-select the `container` tier.

    pytest honours the last `-m` only, so an expression here that omits
    ``not container`` silently overrides `pyproject.toml`'s deselection and
    pulls a `docker build` into Gate 1 and into all three legs of CI's
    `quality` matrix.
    """
    offenders = [
        expression
        for expression in _marker_expressions()
        if f"not {GATED_MARKER}" not in expression
    ]

    assert offenders == [], (
        f"{TEST_SCRIPT} passes {offenders} to pytest. Each one REPLACES "
        f'pyproject.toml\'s `-m "not {GATED_MARKER}"` rather than narrowing '
        f"it, re-selecting the {GATED_MARKER} tier. Append "
        f"` and not {GATED_MARKER}` to every expression."
    )


def _container_job() -> dict[str, Any]:
    """Parse CI's container-tier job out of the workflow file.

    Returns:
        The job's parsed mapping.
    """
    with CI_WORKFLOW.open(encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)
    return dict(workflow["jobs"][CONTAINER_JOB_ID])


def test_the_container_job_exists_and_runs_the_container_selection() -> None:
    """CI still has a job that selects the tier, so the check below can fail.

    The positive control for the wiring assertion: if the job were renamed or
    removed, asserting over its (absent) steps would otherwise pass vacuously.
    """
    steps = _container_job()["steps"]
    run_lines = [str(step.get("run", "")) for step in steps]

    assert any(f"-m {GATED_MARKER}" in line for line in run_lines), (
        f"no step in the `{CONTAINER_JOB_ID}` job runs `pytest -m "
        f"{GATED_MARKER}`; the tier has no CI entry point at all"
    )


def test_the_container_job_arms_the_fail_closed_runtime_flag() -> None:
    """The job that selects the tier also forbids it to skip.

    `require_runtime` is a mechanism; this is its only caller in anger. With
    the `env:` block deleted, every test of the mechanism stays green and a
    required check quietly reports success on a runner that lost its docker
    daemon. Asserted against the harness's own constants so a rename of either
    cannot leave the two halves silently disagreeing.
    """
    selecting = [
        step
        for step in _container_job()["steps"]
        if f"-m {GATED_MARKER}" in str(step.get("run", ""))
    ]

    assert len(selecting) == 1, (
        f"expected exactly one step selecting `-m {GATED_MARKER}`, found "
        f"{len(selecting)}"
    )
    # `or {}` rather than a `get` default: YAML maps a keyless `env:` to None,
    # not to a missing key, and `{}.get` on None is an AttributeError. Deleting
    # the env block's contents is the likeliest way this wiring regresses, and
    # a mutation sweep confirmed it produced an unreadable traceback instead of
    # the message below -- red either way, but red for no stated reason.
    assert (selecting[0].get("env") or {}).get(REQUIRE_RUNTIME_ENV_VAR) == (
        REQUIRE_RUNTIME_ENABLED_VALUE
    ), (
        f"the `{CONTAINER_JOB_ID}` job's tier step does not set "
        f"{REQUIRE_RUNTIME_ENV_VAR}={REQUIRE_RUNTIME_ENABLED_VALUE}, so a "
        "missing docker daemon would SKIP and the required check would report "
        "success while verifying nothing"
    )


def test_the_container_job_builds_the_wheel_the_tier_installs() -> None:
    """The job produces `dist/` before pytest, or the wheel tier cannot run.

    `tests/e2e/test_installed_wheel.py` asserts on a built wheel and, with the
    flag above armed, FAILS rather than skips without one. That makes this step
    load-bearing: delete it and the container job goes red for a reason no
    deployment claim explains.
    """
    run_lines = [str(step.get("run", "")) for step in _container_job()["steps"]]

    assert any("python -m build" in line for line in run_lines), (
        f"no step in the `{CONTAINER_JOB_ID}` job runs `python -m build`, so "
        "dist/ is empty when the wheel tier runs"
    )
