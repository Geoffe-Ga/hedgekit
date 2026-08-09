"""Failing-first tests pinning the pre-commit mypy-scope lockstep contract.

Issue #68 closes a content-debt gap: the pre-commit `mypy` hook is scoped
via `files: ^windbreak/`, which silently excludes `scripts/` from every
pre-commit run even though `scripts/typecheck.sh` -- the entry point
developers and CI both invoke directly -- also only type-checks
`windbreak/`. Nothing forces the two to agree, so a future change to one
scope can drift from the other without either check failing.

The target state: the mypy hook scopes type-checking via explicit
directory `args` (`windbreak/` and `scripts/`) plus `pass_filenames: false`
(so pre-commit never appends individually staged filenames on top of the
directory args), retaining `files:` only as a run-trigger that skips the
hook when no matching file changed. `scripts/typecheck.sh` runs the literal
command `mypy windbreak/ scripts/`. Because both sides are asserted against
each other's literal scope, any future edit that touches one is forced to
touch the other or this test breaks -- that is what "lockstep" means here.

Separately, issue #68 also requires the three whitespace-mangling
pre-commit hooks (`trailing-whitespace`, `end-of-file-fixer`,
`mixed-line-ending`) to `exclude` `scripts/ralph/state.json`, since that
file is machine-written orchestration state whose exact bytes the ralph
tooling depends on.

These assertions began life as Gate 1 RED (the mypy hook was scoped via
`files: ^windbreak/` with no `scripts/` arg and no `pass_filenames: false`,
`scripts/typecheck.sh` only ran `mypy windbreak/`, and none of the three
whitespace hooks declared an `exclude` key). They now pass and stand as the
regression guard that keeps the widened scope and the state-file excludes in
place -- any future edit that drifts one side of the lockstep breaks them.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from tests.toolchain.test_toolchain_pins import _find_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TYPECHECK_SCRIPT_PATH = _REPO_ROOT / "scripts" / "typecheck.sh"

#: `scripts/typecheck.sh`'s mypy invocation over the lockstep-pinned scope.
#: Accepts the bare command name or the toolchain-resolved absolute path
#: (`"$MYPY"`, issue #366); the scope after it is what this file guards.
_MYPY_SCRIPT_INVOCATION = re.compile(r'(?:\bmypy|"\$MYPY") windbreak/ scripts/')
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"

#: Repo URL substrings for the two pre-commit repos this module inspects,
#: reusing the same lookup convention as test_toolchain_pins._find_repo.
_MYPY_REPO_SUBSTRING = "mirrors-mypy"
_WHITESPACE_HOOKS_REPO_SUBSTRING = "pre-commit/pre-commit-hooks"

#: The path whose exact bytes the ralph orchestration tooling depends on,
#: and which the whitespace-mangling hooks must therefore never touch.
_RALPH_STATE_PATH = "scripts/ralph/state.json"


def _find_hook(repo_url_substring: str, hook_id: str) -> dict[str, Any]:
    """Find a specific hook mapping within a specific pre-commit repo.

    Reuses `test_toolchain_pins._find_repo` for the repo lookup rather than
    re-parsing `.pre-commit-config.yaml`, so both test modules agree on
    what "the mypy repo" or "the pre-commit-hooks repo" means.

    Args:
        repo_url_substring: A distinguishing fragment of the repo URL,
            e.g. "mirrors-mypy".
        hook_id: The hook `id` to locate within that repo's `hooks:` list.

    Returns:
        The matching hook mapping.

    Raises:
        AssertionError: If the repo, or the hook within it, is not
            configured.
    """
    repo = _find_repo(repo_url_substring)
    assert repo is not None, f"no pre-commit repo matches {repo_url_substring!r}"
    for hook in repo["hooks"]:
        if hook["id"] == hook_id:
            return dict(hook)
    raise AssertionError(f"repo {repo_url_substring!r} has no {hook_id!r} hook")


def test_mypy_hook_scope_and_typecheck_script_are_kept_in_lockstep() -> None:
    """mypy's pre-commit scope and scripts/typecheck.sh cannot drift apart.

    Asserts both halves of the lockstep contract in one test so a partial
    fix (only the hook, or only the script) still fails: the mypy hook's
    `args` must include both `windbreak/` and `scripts/` as directory
    arguments, and `scripts/typecheck.sh` must run mypy over exactly
    `windbreak/ scripts/`.

    Currently the hook is scoped via `files: ^windbreak/` with `args:
    [--strict]` (no directory args at all), and `scripts/typecheck.sh` runs
    only `mypy windbreak/` -- so both assertions fail against the present
    config.

    The script half is matched as a pattern, not a literal: issue #366
    replaced the bare `mypy` command name with an absolute path resolved
    from the pinned toolchain (`"$MYPY" windbreak/ scripts/`) so which
    binary runs no longer depends on the caller's PATH. The scope -- the
    only thing this lockstep guards -- is unchanged either way.
    """
    hook = _find_hook(_MYPY_REPO_SUBSTRING, "mypy")
    args = hook.get("args", [])

    assert "windbreak/" in args, f"mypy hook args {args!r} missing 'windbreak/'"
    assert "scripts/" in args, f"mypy hook args {args!r} missing 'scripts/'"

    typecheck_source = _TYPECHECK_SCRIPT_PATH.read_text(encoding="utf-8")
    assert _MYPY_SCRIPT_INVOCATION.search(typecheck_source), (
        "scripts/typecheck.sh does not run 'mypy windbreak/ scripts/'"
    )


def test_mypy_hook_has_pass_filenames_false() -> None:
    """The mypy hook must declare `pass_filenames: false`.

    Because the hook is scoped by explicit directory `args` (with `files:`
    kept only as a run-trigger, see the lockstep test above), pre-commit must
    not also append the individual staged filenames it would otherwise pass
    by default -- `pass_filenames: false` guarantees mypy always sees exactly
    the two directory args, regardless of which files are staged.
    """
    hook = _find_hook(_MYPY_REPO_SUBSTRING, "mypy")

    assert hook.get("pass_filenames") is False


def _mypy_config() -> dict[str, Any]:
    """Return pyproject.toml's `[tool.mypy]` table.

    Returns:
        The parsed `[tool.mypy]` mapping.

    Raises:
        AssertionError: If pyproject.toml has no `[tool.mypy]` table.
    """
    data = tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))
    mypy_config = data.get("tool", {}).get("mypy")
    assert isinstance(mypy_config, dict), "pyproject.toml has no [tool.mypy] table"
    return mypy_config


def test_mypy_config_strictness_is_in_lockstep_with_the_precommit_hook() -> None:
    """The `[tool.mypy]` config and the pre-commit hook enforce equal strictness.

    Issue #179: `scripts/typecheck.sh` (Gate 2 / check-all.sh) runs the
    config-driven command `mypy windbreak/ scripts/` with no explicit flags --
    it takes its strictness solely from pyproject.toml's `[tool.mypy]`. The
    pre-commit hook, by contrast, passes `--strict` on the command line. If the
    config enforced only a hand-picked *subset* of `--strict`'s checks (as it
    once did, omitting `no_implicit_reexport`), the local Gate-2 run would
    silently miss an entire error class that CI's `--strict` pre-commit step
    still caught -- the exact false-green recurrence #179 targets.

    This pins the two in lockstep: the hook must pass `--strict`, and the config
    must set `strict = true` (which enables the full `--strict` check set,
    `no_implicit_reexport` included). Any future edit that drops `strict` from
    the config, or `--strict` from the hook, breaks this test.
    """
    hook = _find_hook(_MYPY_REPO_SUBSTRING, "mypy")
    args = hook.get("args", [])

    assert "--strict" in args, (
        f"mypy pre-commit hook args {args!r} no longer pass '--strict' -- the "
        "config-driven typecheck.sh run would drift from the hook's strictness"
    )
    assert _mypy_config().get("strict") is True, (
        "pyproject.toml's [tool.mypy] must set `strict = true` so the "
        "config-driven `mypy windbreak/ scripts/` run in scripts/typecheck.sh "
        "enforces the same check set (including no_implicit_reexport) as the "
        "pre-commit hook's `--strict`"
    )


@pytest.mark.parametrize(
    "hook_id",
    ["trailing-whitespace", "end-of-file-fixer", "mixed-line-ending"],
)
def test_whitespace_class_hooks_exclude_ralph_state_json(hook_id: str) -> None:
    """Whitespace-mangling hooks must not rewrite scripts/ralph/state.json.

    `scripts/ralph/state.json` is machine-written orchestration state; the
    ralph tooling that consumes it depends on its exact byte content, so
    each of these three generic whitespace-fixing hooks must carry an
    `exclude` regex matching that path.
    """
    hook = _find_hook(_WHITESPACE_HOOKS_REPO_SUBSTRING, hook_id)

    assert "exclude" in hook, f"{hook_id} hook has no 'exclude' key"
    pattern = hook["exclude"]
    assert re.search(pattern, _RALPH_STATE_PATH), (
        f"{hook_id} hook's exclude pattern {pattern!r} does not match "
        f"{_RALPH_STATE_PATH!r}"
    )
