"""Tests pinning the Gate 1 shellcheck contract (issue #359).

CI's `Quality Checks` job runs `pre-commit run --all-files`, which includes
the `shellcheck-py` hook. Gate 1 (`./scripts/check-all.sh`) ran no shellcheck
at all, so its exit 0 was not a superset of CI for any change touching a
`.sh` file. The repository's stated contract -- `CLAUDE.md`: "exit code 0 =
ready to commit" -- was simply false for shell, and it failed silently:
nothing locally hinted that an unrun check existed. It cost two full red-CI
round trips (SC2153 in `scripts/mutation.sh`; SC2209 in
`scripts/ralph/test_assert_review_posted.sh`) before being filed.

The fix drives the PINNED pre-commit hook from `scripts/lint.sh`, which
`check-all.sh` already invokes. That choice is what these tests guard, and
the reason each assertion below exists:

- Running a `shellcheck` binary off `PATH` would close the "no check" gap
  while opening a version-skew one -- a local shellcheck of a different
  version than the `rev` pinned in `.pre-commit-config.yaml` disagrees about
  which findings exist. Driving the hook makes local and CI agree by
  construction, so there is no second version for a lockstep test to police.
- The guard must never be of the `command -v shellcheck || skip` shape that
  issue #366 catalogued across this repo: a check that reports success
  without running is strictly worse than one that vetoes.

These assertions began life as Gate 1 RED against `scripts/lint.sh` as it
stood before this change (it resolved only `ruff` and the float-lint
interpreter, and ran no shell lint whatsoever). They now pass and stand as
the regression guard: any future edit that drops the hook invocation, swaps
it for a PATH-resolved binary, or wraps it in a skip guard re-fails here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tests.toolchain.test_toolchain_pins import _find_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LINT_SCRIPT_PATH = _REPO_ROOT / "scripts" / "lint.sh"
_CHECK_ALL_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check-all.sh"

#: The pre-commit repo that supplies the version-pinned shellcheck binary.
_SHELLCHECK_REPO_SUBSTRING = "shellcheck-py"

#: `scripts/lint.sh` must resolve pre-commit through the pinned toolchain
#: (issue #366) rather than by bare name, so which pre-commit runs -- and
#: therefore which shellcheck it drives -- is a property of this repository
#: and not of the shell Gate 1 happened to be launched from.
_PRE_COMMIT_RESOLUTION = re.compile(
    r'PRE_COMMIT="\$\(bash "\$TOOLCHAIN_ENV" --print-tool pre-commit\)"'
)

#: The hook invocation itself: the `shellcheck` hook id over `--all-files`,
#: which is the file set CI's `pre-commit run --all-files` covers.
_HOOK_INVOCATION = re.compile(r'"\$PRE_COMMIT" run shellcheck --all-files')

#: A bare `shellcheck ...` command invocation, i.e. one resolved from PATH.
#: Anchored to a line start (allowing leading whitespace and an `if`/`then`
#: prefix) so it matches a command position rather than the hook-id argument
#: to pre-commit. Applied to comment-stripped source, see `_lint_script_code`.
_PATH_RESOLVED_SHELLCHECK = re.compile(
    r"^[ \t]*(?:if[ \t]+|then[ \t]+|!\s*)?shellcheck[ \t]", re.MULTILINE
)

#: The false-green shape issue #366 exists to prevent: probing for the tool
#: and treating its absence as a pass.
_SKIP_GUARD = re.compile(r"command -v[ \t]+shellcheck")

#: A whole-line shell comment (including the shebang).
_COMMENT_LINE = re.compile(r"^[ \t]*#.*$", re.MULTILINE)


def _lint_script_source() -> str:
    """Return the text of `scripts/lint.sh`.

    Returns:
        The full source of the lint entry point Gate 1 dispatches to.
    """
    return _LINT_SCRIPT_PATH.read_text(encoding="utf-8")


def _lint_script_code() -> str:
    """Return `scripts/lint.sh` with whole-line comments blanked out.

    The two negative assertions below forbid *shapes of code*: a
    PATH-resolved `shellcheck` call and a `command -v shellcheck` skip
    guard. Both shapes are also named in `lint.sh`'s own comments, which
    explain why they are wrong -- documenting the trap must not trip the
    test that guards against it. Lines are blanked rather than deleted so
    the surviving code keeps its line structure for the anchored patterns.

    Returns:
        The lint script source with comment lines replaced by empty lines.
    """
    return _COMMENT_LINE.sub("", _lint_script_source())


def test_precommit_config_defines_a_pinned_shellcheck_hook() -> None:
    """`.pre-commit-config.yaml` must define a version-pinned shellcheck hook.

    This is the version authority the local gate borrows. If the hook were
    removed or renamed, `pre-commit run shellcheck` in `scripts/lint.sh`
    would error out rather than silently pass -- but the failure would be
    reported as a pre-commit usage error, so pinning the hook's existence
    here names the real cause.
    """
    repo: dict[str, Any] | None = _find_repo(_SHELLCHECK_REPO_SUBSTRING)

    assert repo is not None, (
        "no shellcheck-py repo in .pre-commit-config.yaml -- Gate 1's shell "
        "lint (scripts/lint.sh) drives this hook, and CI runs it too"
    )
    assert repo.get("rev"), f"shellcheck-py repo {repo!r} carries no pinned rev"
    hook_ids = [hook["id"] for hook in repo["hooks"]]
    assert "shellcheck" in hook_ids, (
        f"shellcheck-py repo declares hooks {hook_ids!r}, none of them "
        "'shellcheck' -- scripts/lint.sh invokes that hook id by name"
    )


def test_lint_script_drives_the_pinned_precommit_shellcheck_hook() -> None:
    """Gate 1 runs shellcheck via the pinned hook, over CI's file set.

    Both halves are asserted together so a partial regression still fails:
    pre-commit must be resolved through `toolchain-env.sh` (not by bare
    name), and it must be told to run the `shellcheck` hook across
    `--all-files` -- the same file set CI's `pre-commit run --all-files`
    covers. Together these make the local and CI shellcheck verdicts equal
    by construction rather than by a version-comparison test that could rot.
    """
    source = _lint_script_source()

    assert _PRE_COMMIT_RESOLUTION.search(source), (
        "scripts/lint.sh does not resolve pre-commit from the pinned "
        "toolchain via toolchain-env.sh --print-tool pre-commit"
    )
    assert _HOOK_INVOCATION.search(source), (
        'scripts/lint.sh does not run \'"$PRE_COMMIT" run shellcheck '
        "--all-files' -- Gate 1 would not cover shell files that CI checks"
    )


def test_lint_script_does_not_invoke_a_path_resolved_shellcheck() -> None:
    """Gate 1 must not call a `shellcheck` binary off PATH.

    A PATH-resolved shellcheck closes the "no check at all" gap but reopens
    the same local/CI divergence as version skew: a different shellcheck
    version disagrees with the pinned hook about which findings exist. The
    hook is the only invocation that cannot drift.
    """
    source = _lint_script_code()

    assert not _PATH_RESOLVED_SHELLCHECK.search(source), (
        "scripts/lint.sh invokes shellcheck directly; drive the pinned "
        "pre-commit hook instead so the local version cannot drift from "
        ".pre-commit-config.yaml's rev (issue #359)"
    )


def test_lint_script_has_no_skip_guard_around_shell_lint() -> None:
    """A missing tool must veto Gate 1, never be skipped green.

    Issue #366 catalogued several `command -v <tool> || skip` guards in this
    repo that reported SUCCESS for a check that never ran. Resolution goes
    through `toolchain-env.sh`, which vetoes when the tool is absent; this
    test keeps a convenience guard from being added back on top of it.
    """
    source = _lint_script_code()

    assert not _SKIP_GUARD.search(source), (
        "scripts/lint.sh probes for shellcheck with `command -v` -- a check "
        "that skips green when its tool is missing cannot gate anything "
        "(issue #366). Let toolchain-env.sh's resolution veto instead."
    )


def test_check_all_dispatches_to_the_lint_script() -> None:
    """The Gate 1 entry point must actually reach the shell lint.

    `scripts/lint.sh` only gates anything because `check-all.sh` runs it;
    this pins the one link in that chain the other tests assume.
    """
    source = _CHECK_ALL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert re.search(r'run_check "Linting" "lint\.sh"', source), (
        "scripts/check-all.sh no longer dispatches to lint.sh -- Gate 1 "
        "would stop running Ruff, float-lint and shellcheck"
    )
