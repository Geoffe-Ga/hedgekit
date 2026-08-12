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

The third half -- and it is the one that was missing when this module was first
written -- runs that argument out to its end. A job can exist, select the tier
and arm the flag, and still not matter: a workflow job that is not a REQUIRED
status check is advisory. It may go red and the pull request merges anyway.
Issue #467's AC1 ("gates the build") and #468's AC2 ("merge-gating") are that
one repository setting and nothing else, and no file in this tree can assert
it -- which is precisely why it went unasserted while three tests happily
checked the wiring underneath it. :func:`_required_status_check_contexts`
reads the live setting, so the claim is checked against GitHub rather than
against a comment restating it.

That last check is LOCAL-ONLY, and the rest of this module is not. Reading
branch protection needs an admin-scoped token, which CI does not have, so in
CI it skips while everything above it asserts on every run. The skip is loud
and says so; see the test's own docstring for why it is still worth having and
what would make it run in CI.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
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

#: Directories holding the tier's own test modules.
TIER_DIRECTORIES = (REPO_ROOT / "tests" / "e2e", REPO_ROOT / "tests" / "deploy")

#: Module that DEFINES the runtime gate, so it is not one of its callers.
RUNTIME_GATE_MODULE = "harness.py"

#: Probes whose answer must be handed to `require_runtime`, not to `pytest.skip`.
#:
#: Each returns ``None`` or a reason string. Deciding what a reason MEANS is
#: `require_runtime`'s job alone -- skip on a developer machine, fail in the
#: container job, which is a required status check where a skip would report
#: success over nothing. A module that probes and then skips on its own answer
#: opts out of that decision silently, which is what
#: `tests/deploy/test_deployment_cli_contract.py` did with systemd.
RUNTIME_PROBE_HELPERS = ("docker_skip_reason", "systemd_skip_reason")

#: Branch whose protection settings are the merge gate.
PROTECTED_BRANCH = "main"

#: Seconds allowed for the branch-protection query before it is killed.
PROTECTION_QUERY_TIMEOUT_SECONDS = 30.0

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


def _referenced_names(source: str) -> set[str]:
    """Collect the identifiers a module actually references in code.

    Parsed rather than grepped. `tests/e2e/conftest.py` *names* both probes in
    its module docstring, explaining where runtime gating lives; a substring
    scan reads that prose as a call and reports a module that gates nothing.
    An identifier bound or used in code is the only evidence that counts.

    Args:
        source: The module's source text.

    Returns:
        Every name imported, called or otherwise referenced by the module.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def _modules_probing_a_runtime() -> list[tuple[Path, set[str]]]:
    """Find every tier module that asks whether a runtime is present.

    Returns:
        ``(path, referenced_names)`` for each module referencing a probe from
        :data:`RUNTIME_PROBE_HELPERS`, excluding the module that defines them.
    """
    found = []
    for directory in TIER_DIRECTORIES:
        for path in sorted(directory.glob("*.py")):
            if path.name == RUNTIME_GATE_MODULE:
                continue
            names = _referenced_names(path.read_text(encoding="utf-8"))
            if names & set(RUNTIME_PROBE_HELPERS):
                found.append((path, names))
    return found


def test_the_scan_finds_modules_probing_a_runtime() -> None:
    """The probe scan matches something, so the guard below can fail.

    Trap #7 again, and the reason it is worth repeating: the guard underneath
    iterates over whatever this finds. Rename a helper, or move the tier's
    modules, and it would iterate over nothing and pass forever.
    """
    probing = _modules_probing_a_runtime()

    assert probing != [], (
        f"no module under {[str(d) for d in TIER_DIRECTORIES]} names any of "
        f"{RUNTIME_PROBE_HELPERS}. Either the helpers were renamed or the scan "
        "is broken; either way the guard below is checking an empty set."
    )


def test_every_module_probing_a_runtime_gates_on_require_runtime() -> None:
    """A module that probes a runtime must let `require_runtime` judge it.

    The probes return a reason; they do not decide what a reason means. That
    decision belongs to `require_runtime` alone, because it is the one place
    that knows the answer is environment-dependent: SKIP on a developer
    machine without a docker daemon, FAIL in the container job, where
    ``WINDBREAK_E2E_REQUIRE_RUNTIME=1`` is set because the job is a required
    status check and a required check that skips reports success.

    `tests/deploy/test_deployment_cli_contract.py` probed systemd and called
    `pytest.skip` on the answer itself, so its four `systemd-analyze verify`
    assertions over the shipped unit files would have vanished silently on a
    runner that lost systemd -- while `harness.py` documented the flag as
    making exactly that impossible. The claim was false for as long as one
    call site opted out, which is why this is asserted over every module
    rather than reviewed once.
    """
    ungated = [
        str(path.relative_to(REPO_ROOT))
        for path, names in _modules_probing_a_runtime()
        if "require_runtime" not in names
    ]

    assert ungated == [], (
        f"{ungated} probe for a runtime without importing `require_runtime`, "
        "so they decide for themselves what a missing runtime means. A "
        "`pytest.skip` there reports SUCCESS for a required check that "
        "verified nothing. Hand the probe's result to `require_runtime` "
        "instead."
    )


def _required_status_check_contexts() -> list[str]:
    """Read the status-check contexts branch protection requires on `main`.

    Queries GitHub rather than any file in this tree, because the setting is
    not in this tree. The read is deliberately allowed to *skip* -- never to
    pass -- when it cannot be performed. A silent pass here would be the same
    defect the caller exists to catch, one level up.

    WHERE THIS ACTUALLY RUNS: the endpoint needs an admin-scoped token. CI has
    none -- `ci.yml` sets no `GH_TOKEN` or `GITHUB_TOKEN`, and a runner's
    default `GITHUB_TOKEN` would not carry the scope anyway -- so on every CI
    run this takes the skip path. In practice the assertion is made by a
    maintainer running the suite locally with an authenticated `gh`. The skip
    reasons below say so in as many words, because a bare SKIP in a CI log
    reads like coverage.

    Returns:
        The required context names, exactly as GitHub records them.
    """
    if shutil.which("gh") is None:
        pytest.skip(
            "branch-protection guard is LOCAL-ONLY here: no `gh` CLI on PATH, "
            "so the required-status-check set cannot be read. It lives on the "
            "repository, not in this tree, so there is nothing to fall back "
            "to. This run asserted nothing about whether the container tier "
            "gates merge."
        )
    completed = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{{owner}}/{{repo}}/branches/{PROTECTED_BRANCH}/protection",
        ],
        capture_output=True,
        text=True,
        timeout=PROTECTION_QUERY_TIMEOUT_SECONDS,
        cwd=str(REPO_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(
            "branch-protection guard is LOCAL-ONLY here: `gh api "
            f"repos/.../branches/{PROTECTED_BRANCH}/protection` exited "
            f"{completed.returncode}. Reading protection needs an admin-scoped "
            "token; CI provides none (no GH_TOKEN/GITHUB_TOKEN is set in "
            "ci.yml, and the default runner token lacks the scope regardless), "
            "so this is the expected CI outcome and NOT evidence that the "
            "container tier gates merge. Run the suite locally with an "
            f"authenticated `gh` to assert it. stderr: {completed.stderr.strip()}"
        )
    protection = json.loads(completed.stdout)
    checks = protection.get("required_status_checks") or {}
    return [str(context) for context in checks.get("contexts", [])]


def test_the_container_job_is_a_required_status_check() -> None:
    """`main` does not merge unless the container tier reported success.

    The three tests above prove the job exists, selects the tier and arms the
    fail-closed flag. None of them proves the job *matters*: a workflow job
    that is not a required status check is advisory, free to go red while the
    pull request merges anyway. That is this module's own opening complaint --
    every seam tested, the composition unreachable -- aimed at the last seam,
    and it is the one that stayed unasserted longest.

    Issue #467's AC1 ("gates the build") and #468's AC2 ("merge-gating") ARE
    this setting. It is a property of the repository, so it can neither be
    changed by a pull request nor asserted from a file; the only honest check
    is a live read.

    WHAT THIS GUARD IS AND IS NOT. It asserts only where an admin-scoped token
    can read branch protection, which today means a maintainer running the
    suite locally with an authenticated `gh`. **In CI it skips, every time**:
    `ci.yml` sets no `GH_TOKEN` or `GITHUB_TOKEN`, and a runner's default token
    would not carry the scope in any case. So this test is not a merge gate on
    the setting, and nothing here should be read as claiming CI verifies it.

    It still earns its place twice over. It caught the `ci.yml` job-name
    truncation below on its very first run -- a live misconfiguration that
    would have blocked every pull request on this repository -- and it turns a
    silent repository-level misconfiguration into a visible local failure for
    anyone who runs the suite with credentials, which is the only place the
    check *can* be made at all.

    Making it run in CI is mechanical if the repo owner wants it: expose an
    admin-scoped token to the workflow as `GH_TOKEN`. That is a credentials
    policy decision, deliberately not taken here.

    The context GitHub matches on is the job's `name:`, not its key in `jobs:`,
    so the expected value is read out of the workflow. Rename the job without
    renaming the protection context and `main` blocks forever on a context
    nothing will ever report -- caught here rather than on the first pull
    request unlucky enough to hit it.

    THIS TEST FOUND THAT EXACT STATE ON ITS FIRST RUN. The job's name was an
    unquoted YAML scalar ending `(epic #465)`, so ` #465)` parsed as a comment
    and GitHub reported the check run as `End-to-End Container Tier (epic`
    while protection required the full string. Every pull request would have
    blocked on a context no job produces. The fix -- quoting the scalar -- is
    in `ci.yml` beside the name, and this assertion is what holds it.
    """
    expected = str(_container_job()["name"])
    contexts = _required_status_check_contexts()

    assert contexts != [], (
        f"branch protection on `{PROTECTED_BRANCH}` requires NO status checks "
        "at all. Every gate in this repository is advisory in that state, so "
        "the membership assertion below would pass over an empty set and "
        "report success while verifying nothing."
    )
    assert expected in contexts, (
        f"`{expected}` is not among the status checks branch protection "
        f"requires on `{PROTECTED_BRANCH}` ({contexts}). The container tier "
        "runs but does not gate merge, so a red tier merges anyway -- issue "
        "#467 AC1 and #468 AC2 are unmet in that state. Either the protection "
        "setting was removed, or the job's `name:` in "
        f"{CI_WORKFLOW.name} changed without the context being renamed to "
        "match."
    )
