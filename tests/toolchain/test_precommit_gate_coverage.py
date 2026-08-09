"""Tests pinning TOTAL pre-commit hook coverage for Gate 1 (issue #401).

Gate 1 (`./scripts/check-all.sh`) used to dispatch a hand-maintained SUBSET
of `.pre-commit-config.yaml`, hook by hook, by name. The list drifted, and
`check-all.sh` exit 0 -- which `CLAUDE.md` defines as "ready to commit" --
stopped being a superset of CI's `pre-commit run --all-files` job:

- `shellcheck` was in the config and dispatched by nothing. Two red-CI round
  trips (SC2153, SC2209) before issue #359 / PR #395 added it, for that one
  hook.
- `vulture` was in the config and dispatched by nothing. It failed CI on
  PR #398 (`windbreak/net/live_http.py:122: unused variable
  'allow_redirects'`) after a local Gate 1 run of 8/8 exit 0. `CLAUDE.md`
  had claimed for its whole life that `check-all.sh` ran "Dead code
  detection (vulture)"; it never did.
- Sixteen further hooks -- the file-hygiene set (`trailing-whitespace`,
  `end-of-file-fixer`, `check-yaml`/`toml`/`json`, `check-ast`,
  `debug-statements`, `check-docstring-first`, `detect-private-key`,
  `mixed-line-ending`, and the rest) -- had never been dispatched by any
  `scripts/*.sh` at all. Only 8 of 26 configured hooks were covered.

Adding each missing hook by name as it burns a CI cycle does not converge:
the gap reopens silently the next time a hook is added to the config. So
`scripts/precommit.sh` runs the WHOLE set through the pinned toolchain, and
this module is the part that keeps it whole.

The load-bearing test is
`test_every_precommit_hook_is_dispatched_by_gate1_or_explicitly_excluded`.
It enumerates every hook id in `.pre-commit-config.yaml` and requires each
one to be either dispatched by Gate 1 or present in `_GATE1_EXCLUDED_HOOKS`
with a stated reason. A hook escapes Gate 1 in exactly two ways -- being
named in the script's `SKIP` list, or declaring `stages:` that omit the
`pre-commit` stage -- and both are computed here from the real config and
the real script, then compared against the registry. Registering an
exclusion is therefore a recorded decision; omitting one fails the suite.

Its non-vacuity was demonstrated before this module was committed: adding a
dummy hook with `stages: [commit-msg]` to the config made it fail with that
hook named, and adding a hook id to `GATE1_SKIPPED_HOOKS` in
`scripts/precommit.sh` made it fail the same way. Note the shape that
correctly does NOT fail: a plain new hook with default stages is covered the
moment it is added, because Gate 1 runs the whole set. That is the point of
the structural fix, not a hole in the test.

These assertions began life as Gate 1 RED: `scripts/precommit.sh` did not
exist and `check-all.sh` dispatched no whole-set pre-commit run.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.toolchain.test_toolchain_pins import _load_precommit, _precommit_hook_ids

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRECOMMIT_SCRIPT_PATH = _REPO_ROOT / "scripts" / "precommit.sh"
_CHECK_ALL_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check-all.sh"
_CI_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: The pre-commit stage `pre-commit run --all-files` selects hooks for. A hook
#: declaring `stages:` without it is unreachable from Gate 1 *and* from CI's
#: identical invocation, so it must be registered as an exclusion.
_GATE1_STAGE = "pre-commit"

#: pre-commit's legacy stage names and their modern equivalents. Normalising
#: means a config written in either dialect is judged by what it RUNS rather
#: than by the spelling, which is the same mistake -- matching by name instead
#: of by behaviour -- that made `no-floats-money-paths` look uncovered.
_STAGE_ALIASES = {
    "commit": "pre-commit",
    "push": "pre-push",
    "merge-commit": "pre-merge-commit",
}

#: Hooks Gate 1 deliberately does NOT run, each with the reason on the record.
#: This is the "explicit exclusion" half of issue #401's acceptance criterion
#: 1: a hook may be left out of Gate 1, but only as a stated decision, never
#: as an omission nobody notices. Keys must match what the config and
#: `scripts/precommit.sh` actually put out of reach -- see
#: `test_every_precommit_hook_is_dispatched_by_gate1_or_explicitly_excluded`.
_GATE1_EXCLUDED_HOOKS: dict[str, str] = {
    "no-commit-to-branch": (
        "Asserts a property of the current BRANCH ('do not commit directly to "
        "main'), not of the tree's quality, so running it from a quality gate "
        "is a category error: a developer running check-all.sh while sitting "
        "on main would get a red Gate 1 that says nothing about the code. CI "
        "skips it by name for the same reason (issue #127). The policy is not "
        "weakened -- the hook still fires at commit time via `pre-commit "
        "install`, which is where a branch policy belongs. Named in "
        "GATE1_SKIPPED_HOOKS in scripts/precommit.sh."
    ),
    "conventional-pre-commit": (
        "Runs at the `commit-msg` stage: its input is a commit message, not a "
        "file set, so `pre-commit run --all-files` does not select it and "
        "there is nothing for Gate 1 to feed it. CI's identical invocation "
        "does not run it either, so excluding it costs no Gate-1/CI parity. "
        "It still fires on every real commit through the installed commit-msg "
        "hook."
    ),
}

#: The `GATE1_SKIPPED_HOOKS` assignment in `scripts/precommit.sh`: the set of
#: hook ids handed to pre-commit's `SKIP` environment variable.
_SKIP_LIST_ASSIGNMENT = re.compile(r'^GATE1_SKIPPED_HOOKS="([^"]*)"', re.MULTILINE)

#: pre-commit resolved through the pinned toolchain rather than by bare name
#: (issue #366), so which pre-commit runs -- and therefore which pinned hook
#: versions Gate 1 enforces -- is a property of this repo, not of the caller's
#: shell.
_PRE_COMMIT_RESOLUTION = re.compile(
    r'PRE_COMMIT="\$\(bash "\$TOOLCHAIN_ENV" --print-tool pre-commit\)"'
)

#: The whole-set invocation. The token immediately after `run` must be
#: `--all-files`: a hook id there would narrow the run back to the
#: subset-by-name arrangement issue #401 exists to remove.
_WHOLE_SET_ARGS = re.compile(r"PRE_COMMIT_ARGS=\(run --all-files[ )]")

#: The command that actually dispatches the hook set, with the named skip list
#: applied.
_HOOK_SET_INVOCATION = re.compile(
    r'SKIP="\$GATE1_SKIPPED_HOOKS" "\$PRE_COMMIT" "\$\{PRE_COMMIT_ARGS\[@\]\}"'
)

#: `check-all.sh`'s dispatch of the gate.
_CHECK_ALL_DISPATCH = re.compile(
    r'run_check "Pre-commit \(all hooks\)" "precommit\.sh"'
)

#: The false-green shape issue #366 catalogued: probe for the tool, treat its
#: absence as a pass.
_SKIP_GUARD = re.compile(r"command -v[ \t]+pre-commit")

#: A whole-line shell comment (including the shebang).
_COMMENT_LINE = re.compile(r"^[ \t]*#.*$", re.MULTILINE)


def _precommit_script_source() -> str:
    """Return the text of `scripts/precommit.sh`.

    Returns:
        The full source of the whole-hook-set gate script.
    """
    return _PRECOMMIT_SCRIPT_PATH.read_text(encoding="utf-8")


def _precommit_script_code() -> str:
    """Return `scripts/precommit.sh` with whole-line comments blanked out.

    The negative assertions below forbid *shapes of code* that the script's
    own comments describe in order to explain why they are wrong.
    Documenting a trap must not trip the test guarding against it. Lines are
    blanked rather than deleted so surviving code keeps its line structure.

    Returns:
        The gate script source with comment lines replaced by empty lines.
    """
    return _COMMENT_LINE.sub("", _precommit_script_source())


def _gate1_skip_list() -> set[str]:
    """Return the hook ids `scripts/precommit.sh` passes to pre-commit's SKIP.

    Returns:
        The hook ids named in `GATE1_SKIPPED_HOOKS`, empty entries dropped.

    Raises:
        AssertionError: If the script declares no `GATE1_SKIPPED_HOOKS`.
    """
    match = _SKIP_LIST_ASSIGNMENT.search(_precommit_script_code())
    assert match is not None, (
        "scripts/precommit.sh declares no GATE1_SKIPPED_HOOKS assignment -- "
        "the named exclusion list is where a Gate 1 omission becomes a "
        "recorded decision (issue #401)"
    )
    return {name.strip() for name in match.group(1).split(",") if name.strip()}


def _hooks_unreachable_by_stage() -> set[str]:
    """Return hook ids `pre-commit run --all-files` does not select.

    A hook whose `stages:` omit the `pre-commit` stage is never dispatched by
    a file-set run, locally or in CI, regardless of any SKIP list. Hooks that
    declare no stages inherit the config's `default_stages`, and with neither
    set they run at every stage.

    Returns:
        The hook ids that cannot run at the `pre-commit` stage.
    """
    config: dict[str, Any] = _load_precommit()
    default_stages = config.get("default_stages")

    unreachable: set[str] = set()
    for repo in config["repos"]:
        for hook in repo.get("hooks", []):
            stages = hook.get("stages", default_stages)
            if stages is None:
                continue
            normalised = {_STAGE_ALIASES.get(stage, stage) for stage in stages}
            if _GATE1_STAGE not in normalised:
                unreachable.add(hook["id"])
    return unreachable


def _ci_precommit_skip_list() -> set[str]:
    """Return the SKIP list CI applies to its whole-file pre-commit run.

    Returns:
        The hook ids named in the `SKIP` env of the CI step that runs
        `pre-commit run --all-files`.

    Raises:
        AssertionError: If no such CI step is found.
    """
    with _CI_WORKFLOW_PATH.open(encoding="utf-8") as handle:
        workflow: dict[str, Any] = yaml.safe_load(handle)

    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "pre-commit run --all-files" in str(step.get("run", "")):
                skip = str(step.get("env", {}).get("SKIP", ""))
                return {name.strip() for name in skip.split(",") if name.strip()}

    raise AssertionError(
        f"{_CI_WORKFLOW_PATH} has no step running `pre-commit run --all-files` "
        "-- Gate 1's hook set is defined to be a superset of that step's"
    )


def test_precommit_gate_script_exists_and_is_executable() -> None:
    """`scripts/precommit.sh` exists and can be run directly.

    `check-all.sh` dispatches it as `"$SCRIPT_DIR/precommit.sh"`, which needs
    the executable bit; a non-executable script would fail the gate with a
    permissions error rather than a quality verdict.
    """
    assert _PRECOMMIT_SCRIPT_PATH.is_file(), (
        f"{_PRECOMMIT_SCRIPT_PATH} does not exist -- issue #401 RED"
    )
    assert os.access(_PRECOMMIT_SCRIPT_PATH, os.X_OK), (
        f"{_PRECOMMIT_SCRIPT_PATH} exists but is not executable"
    )


def test_precommit_gate_resolves_pre_commit_from_the_pinned_toolchain() -> None:
    """The gate takes pre-commit from the pinned venv, not from PATH.

    Which pre-commit runs decides which pinned hook revisions are enforced.
    Resolving it by bare name would hand that choice to the caller's shell,
    reintroducing issue #366 in the one check whose job is matching CI.
    """
    assert _PRE_COMMIT_RESOLUTION.search(_precommit_script_source()), (
        "scripts/precommit.sh does not resolve pre-commit through "
        "toolchain-env.sh --print-tool pre-commit"
    )


def test_precommit_gate_runs_the_whole_hook_set_over_all_files() -> None:
    """Gate 1 runs every hook, not a named subset, across CI's file set.

    The token after `run` must be `--all-files`. A hook id in that position
    is exactly the hand-maintained-subset arrangement that let `shellcheck`
    and then `vulture` sit in the config, undispatched, until each cost a
    red-CI round trip.
    """
    source = _precommit_script_source()

    assert _WHOLE_SET_ARGS.search(source), (
        "scripts/precommit.sh does not run `pre-commit run --all-files` with "
        "no hook-id argument -- a narrowed run reopens the drift gap #401 closes"
    )
    assert _HOOK_SET_INVOCATION.search(source), (
        "scripts/precommit.sh does not dispatch the hook set with the named "
        "SKIP list applied"
    )


def test_precommit_gate_has_no_skip_guard() -> None:
    """A missing pre-commit must veto Gate 1, never be skipped green.

    Issue #366 catalogued four `command -v <tool> || skip` guards that
    reported SUCCESS for checks that never ran. A check that cannot fail is
    worse than one that vetoes, and this gate's entire value is that its
    verdict is trustworthy.
    """
    assert not _SKIP_GUARD.search(_precommit_script_code()), (
        "scripts/precommit.sh probes for pre-commit with `command -v`; the "
        "pinned resolver already vetoes when it is missing (issue #366)"
    )


def test_check_all_dispatches_the_precommit_gate() -> None:
    """Gate 1 actually runs the whole-hook-set check.

    The script existing is not coverage. `check-all.sh` is the gate CLAUDE.md
    defines as "ready to commit", so the dispatch is the thing that makes the
    hook set part of Gate 1 at all.
    """
    assert _CHECK_ALL_DISPATCH.search(
        _CHECK_ALL_SCRIPT_PATH.read_text(encoding="utf-8")
    ), (
        "scripts/check-all.sh does not dispatch precommit.sh -- the whole "
        "hook set would exist as a script nobody runs"
    )


def test_every_precommit_hook_is_dispatched_by_gate1_or_explicitly_excluded() -> None:
    """Every configured hook is run by Gate 1 or registered with a reason.

    This is the durable half of issue #401. A hook escapes Gate 1 in exactly
    two ways: being named in `GATE1_SKIPPED_HOOKS`, or declaring `stages:`
    that omit the `pre-commit` stage. Both are computed from the real config
    and the real script and compared against `_GATE1_EXCLUDED_HOOKS`, so
    putting a hook out of Gate 1's reach without recording why fails here --
    and so does registering an exclusion that no longer applies.

    A plain new hook needs no registration: the whole-set run covers it the
    moment it is added. That asymmetry is the fix working, not a gap.
    """
    all_hook_ids = _precommit_hook_ids()
    unreachable = _gate1_skip_list() | _hooks_unreachable_by_stage()
    registered = set(_GATE1_EXCLUDED_HOOKS)

    unregistered = unreachable - registered
    assert not unregistered, (
        f"hooks {sorted(unregistered)} are configured in "
        ".pre-commit-config.yaml but cannot run under Gate 1's "
        "`pre-commit run --all-files` (skipped by name, or declaring stages "
        "that omit 'pre-commit'). Either let Gate 1 run them, or add each to "
        "_GATE1_EXCLUDED_HOOKS with the reason -- an exclusion must be a "
        "decision on the record, not an omission (issue #401)."
    )

    stale = registered - unreachable
    assert not stale, (
        f"hooks {sorted(stale)} are registered as Gate 1 exclusions but Gate 1 "
        "now runs them (or they left the config). Drop the stale entries so "
        "the registry keeps meaning what it says."
    )

    unknown = registered - all_hook_ids
    assert not unknown, (
        f"_GATE1_EXCLUDED_HOOKS names {sorted(unknown)}, which is not a hook "
        "id in .pre-commit-config.yaml"
    )

    dispatched = all_hook_ids - unreachable
    assert dispatched | registered == all_hook_ids, (
        "every configured hook must be dispatched by Gate 1 or registered as "
        f"an exclusion; unaccounted: {sorted(all_hook_ids - dispatched - registered)}"
    )


@pytest.mark.parametrize("hook_id", sorted(_GATE1_EXCLUDED_HOOKS))
def test_every_gate1_exclusion_carries_a_stated_reason(hook_id: str) -> None:
    """An excluded hook must come with prose explaining the decision.

    "Excluded by name with a stated reason" is the acceptance criterion; a
    registry of bare names would be an omission list with extra steps, and
    the next reader would have no way to tell a deliberate call from an
    accident.

    Args:
        hook_id: The registered exclusion under test.
    """
    reason = _GATE1_EXCLUDED_HOOKS[hook_id]

    assert len(reason.split()) >= 20, (
        f"the Gate 1 exclusion for {hook_id!r} has no substantive reason: {reason!r}"
    )


def test_gate1_skip_list_matches_ci_skip_list() -> None:
    """Gate 1 and CI skip exactly the same hooks.

    Gate 1's contract is to be a superset of CI's pre-commit job. Both run
    `pre-commit run --all-files` against the same pinned config, so the only
    way they can disagree is via their SKIP lists -- pinning them equal makes
    "local green implies CI green" a construction guarantee for the hook set
    rather than something rediscovered on a red build.
    """
    assert _gate1_skip_list() == _ci_precommit_skip_list(), (
        f"Gate 1 skips {sorted(_gate1_skip_list())} but CI skips "
        f"{sorted(_ci_precommit_skip_list())}; a hook skipped locally and run "
        "in CI is exactly the divergence issue #401 closes"
    )


def test_vulture_hook_is_dispatched_by_gate1() -> None:
    """The hook that failed CI on PR #398 is now covered, by name.

    The general test above would catch a vulture regression too, but this one
    names the specific defect so a future failure reads as "the dead-code
    check left Gate 1 again" rather than as an abstract coverage arithmetic
    error.
    """
    assert "vulture" in _precommit_hook_ids(), (
        "no vulture hook in .pre-commit-config.yaml -- CLAUDE.md documents "
        "dead-code detection as part of Gate 1"
    )
    unreachable = _gate1_skip_list() | _hooks_unreachable_by_stage()
    assert "vulture" not in unreachable, (
        "vulture is configured but unreachable from Gate 1 -- this is the "
        "exact state that let `unused variable 'allow_redirects'` reach CI on "
        "PR #398 after a local 8/8 exit 0"
    )
