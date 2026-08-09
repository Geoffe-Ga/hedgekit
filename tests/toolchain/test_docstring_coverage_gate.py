"""Tests pinning the docstring-coverage gate to something that can fail (#351).

`CLAUDE.md`'s Quality Standards table listed a docstring threshold beside
line coverage, complexity and the rest -- and nothing measured it. Neither
`pydocstyle` nor `interrogate` was installed or dispatched by any script,
and `D` was absent from `[tool.ruff.lint].select`. The row was decoration:
a documented gate that could not fail, in a table whose whole purpose is to
say which gates can.

That is the third instance of this exact defect class found in the gate
machinery itself, after issue #359 (`shellcheck` in the hook config,
dispatched by no script) and issue #401 (18 of 26 hooks undispatched, and
`CLAUDE.md` claiming vulture ran when it never had). The recurring lesson
is the one these tests encode: a threshold is only real if something
executes it, and "something executes it" is a property worth asserting
rather than assuming.

WHAT IS ENFORCED, AND WHY IT IS NOT A PERCENTAGE

`D1` is pydocstyle's "missing docstring" family, D100-D107, reimplemented
natively by ruff -- the linter already pinned in `constraints-quality.txt`,
already dispatched by `scripts/lint.sh`, already run again by the
`ruff-check` pre-commit hook that Gate 1 executes in full since #401. It
enforces docstring PRESENCE per symbol, i.e. 100%, which is stricter than
the >=95% the docs used to claim.

A percentage is what you adopt when 100% is out of reach, and it buys that
headroom by letting a fixed fraction of symbols silently lose their
docstrings while the meter still reads green. This repository did not need
the headroom: when the gate was wired, `windbreak/`, `scripts/` and
`plans/` were already at 100% presence and all 129 findings were
undocumented test functions in eight files, all written rather than
ignored. So the docs were corrected to state the enforced rule instead of
an unmeasured percentage, and `test_claude_md_states_the_rule_that_is
_enforced` keeps the two from drifting apart again.

NON-VACUITY

The load-bearing tests here are the probe pair. It is not enough to assert
that `"D1"` appears in a config list -- that assertion passes just as
happily if ruff ignores the setting, if a `per-file-ignores` entry cancels
it, or if the family is spelled in a way ruff does not resolve. So the
probes actually RUN the pinned ruff, under this repository's real
`pyproject.toml`, over a synthetic module: one undocumented (must be
reported) and one documented (must be clean). Together they show the gate
discriminates, which is exactly what a gate that "cannot fail" did not.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
_CLAUDE_MD_PATH = _REPO_ROOT / "CLAUDE.md"
_LINT_SCRIPT_PATH = _REPO_ROOT / "scripts" / "lint.sh"
_CHECK_ALL_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check-all.sh"
_TOOLCHAIN_ENV_PATH = _REPO_ROOT / "scripts" / "toolchain-env.sh"

#: pydocstyle's "missing docstring" family, enumerated rather than expressed
#: as the prefix `D1` so the test states the OBLIGATION (these eight symbol
#: kinds must be documented) independently of the prefix spelling the config
#: happens to use. A future config that selected `D100` individually and
#: dropped `D102` would still satisfy a naive `"D1" in select` check; it
#: fails here, which is the point.
_DOCSTRING_COVERAGE_CODES = (
    "D100",  # module
    "D101",  # public class
    "D102",  # public method
    "D103",  # public function
    "D104",  # public package
    "D105",  # magic method
    "D106",  # public nested class
    "D107",  # __init__
)


def _pyproject() -> dict[str, object]:
    """Return the parsed `pyproject.toml` of this repository.

    Returns:
        The full parsed TOML document.
    """
    with _PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _ruff_lint_config() -> dict[str, object]:
    """Return the `[tool.ruff.lint]` table from `pyproject.toml`.

    Returns:
        The ruff lint configuration table.
    """
    tool = _pyproject()["tool"]
    assert isinstance(tool, dict)
    ruff = tool["ruff"]
    assert isinstance(ruff, dict)
    lint = ruff["lint"]
    assert isinstance(lint, dict)
    return lint


def _selected_prefixes() -> list[str]:
    """Return the rule prefixes in `[tool.ruff.lint].select`.

    Returns:
        The configured select entries, as written.
    """
    select = _ruff_lint_config()["select"]
    assert isinstance(select, list)
    return [str(entry) for entry in select]


def _is_enabled(code: str, prefixes: list[str]) -> bool:
    """Report whether `code` is selected by any of `prefixes`.

    Ruff resolves a rule code against a select entry by PREFIX, so `D`,
    `D1`, `D10` and `D103` all enable D103. This mirrors that resolution
    so the assertions below hold for any spelling of the family rather
    than for one hard-coded string.

    Args:
        code: A full ruff rule code, e.g. `"D103"`.
        prefixes: The configured select entries.

    Returns:
        True when at least one prefix selects the code.
    """
    return any(code.startswith(prefix) for prefix in prefixes)


def _resolve_ruff() -> str:
    """Return the path to the pinned ruff, failing loudly if absent.

    Deliberately NOT a `shutil.which` lookup and deliberately NOT wrapped
    in a skip: issue #366 catalogued four `command -v <tool> || skip`
    guards in this repo that reported SUCCESS for checks that never ran.
    Resolution goes through `scripts/toolchain-env.sh`, the same authority
    the gate scripts use, so this test judges the binary Gate 1 would
    actually run rather than whatever the caller's PATH offers.

    Returns:
        Absolute path to the pinned ruff executable.
    """
    result = subprocess.run(
        ["bash", str(_TOOLCHAIN_ENV_PATH), "--print-tool", "ruff"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        "could not resolve ruff through scripts/toolchain-env.sh -- the "
        "docstring-coverage gate cannot be verified without the pinned "
        f"linter, so this fails rather than skips (issue #366):\n{result.stderr}"
    )
    return result.stdout.strip()


def _ruff_findings(source: str, tmp_path: Path) -> list[str]:
    """Run the pinned ruff over `source` under this repo's real config.

    The probe module is written outside the repository tree and ruff is
    pointed at `pyproject.toml` explicitly with `--config`. Writing it
    *inside* the tree would let real config discovery apply, but a crashed
    run would then leave an undocumented function behind that reddens
    Gate 1 for everyone, and parallel xdist workers would collide over the
    path. Pointing `--config` at the real file keeps the rule set derived
    from this repository rather than restated here, with none of that risk.

    `--no-cache` because the probe paths are transient and a cached verdict
    keyed on a reused temp path would be worse than no verdict at all.

    Args:
        source: Python source text for the probe module.
        tmp_path: pytest-provided scratch directory.

    Returns:
        The rule codes ruff reported, in order.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")

    result = subprocess.run(
        [
            _resolve_ruff(),
            "check",
            "--no-cache",
            "--config",
            str(_PYPROJECT_PATH),
            "--output-format",
            "concise",
            str(probe),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    codes: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        # concise format: "<path>:<line>:<col>: <CODE> <message>"
        for index, token in enumerate(parts):
            if token.endswith(":") and index + 1 < len(parts):
                candidate = parts[index + 1]
                if candidate.startswith("D") and candidate[1:].isdigit():
                    codes.append(candidate)
                break
    return codes


# --- Configuration: the family is selected, and nothing cancels it ------------


@pytest.mark.parametrize("code", _DOCSTRING_COVERAGE_CODES)
def test_docstring_coverage_rule_is_selected(code: str) -> None:
    """Every "missing docstring" rule D100-D107 is enabled by ruff's select.

    Args:
        code: The ruff rule code under test.
    """
    prefixes = _selected_prefixes()

    assert _is_enabled(code, prefixes), (
        f"{code} is not selected by any entry in [tool.ruff.lint].select "
        f"({prefixes!r}). CLAUDE.md documents a docstring-coverage gate; "
        "with this rule off, that gate cannot fail for the symbol kind it "
        "covers -- the exact defect issue #351 fixed."
    )


def test_no_per_file_ignore_cancels_the_docstring_coverage_family() -> None:
    """No `per-file-ignores` entry re-disables a docstring-coverage rule.

    Selecting `D1` and then exempting `tests/**` from it would restore the
    unenforced status quo for the very files that supplied all 129 original
    findings, while leaving the config looking like a gate. Issue #351's
    acceptance criteria call that out by name: violations were to be fixed,
    not blanket-ignored.
    """
    per_file_ignores = _ruff_lint_config().get("per-file-ignores", {})
    assert isinstance(per_file_ignores, dict)

    offenders: dict[str, list[str]] = {}
    for pattern, ignored in per_file_ignores.items():
        assert isinstance(ignored, list)
        cancelled = [
            code
            for code in _DOCSTRING_COVERAGE_CODES
            if _is_enabled(code, [str(entry) for entry in ignored])
        ]
        if cancelled:
            offenders[str(pattern)] = cancelled

    assert not offenders, (
        f"per-file-ignores cancels docstring-coverage rules: {offenders!r}. "
        "Write the missing docstrings instead -- exempting a path from the "
        "gate reproduces issue #351 for that path."
    )


def test_global_ignore_does_not_cancel_the_docstring_coverage_family() -> None:
    """The top-level `ignore` list does not disable D100-D107.

    `select` and `ignore` are applied in that order, so an entry here would
    silently undo the selection asserted above without changing it.
    """
    ignored = _ruff_lint_config().get("ignore", [])
    assert isinstance(ignored, list)
    prefixes = [str(entry) for entry in ignored]

    cancelled = [
        code for code in _DOCSTRING_COVERAGE_CODES if _is_enabled(code, prefixes)
    ]
    assert not cancelled, (
        f"[tool.ruff.lint].ignore cancels {cancelled!r}, undoing the "
        "docstring-coverage selection"
    )


# --- Non-vacuity: the configured gate actually discriminates ------------------


def test_undocumented_public_function_is_reported(tmp_path: Path) -> None:
    """An undocumented public function is reported as D103 by the real config.

    THE non-vacuity test. Everything above inspects configuration; this one
    executes it. Without this, a config change that looked correct but
    resolved to nothing would pass the whole file.

    Args:
        tmp_path: pytest-provided scratch directory.
    """
    findings = _ruff_findings(
        '"""Probe module."""\n\n\ndef probe_function() -> int:\n    return 1\n',
        tmp_path,
    )

    assert "D103" in findings, (
        "the pinned ruff, run under this repository's pyproject.toml, did "
        f"NOT report D103 for an undocumented public function (got {findings!r}). "
        "The docstring-coverage gate is decorative again -- issue #351."
    )


def test_undocumented_public_method_and_class_are_reported(tmp_path: Path) -> None:
    """Undocumented classes and methods are reported, not just functions.

    D103 alone would leave the 51 undocumented methods that motivated this
    change unenforced, so the family is probed beyond its commonest member.

    Args:
        tmp_path: pytest-provided scratch directory.
    """
    findings = _ruff_findings(
        '"""Probe module."""\n\n\nclass Probe:\n    def probe(self) -> int:\n'
        "        return 1\n",
        tmp_path,
    )

    assert "D101" in findings, f"undocumented public class not reported: {findings!r}"
    assert "D102" in findings, f"undocumented public method not reported: {findings!r}"


def test_undocumented_module_is_reported(tmp_path: Path) -> None:
    """A module with no docstring is reported as D100 by the real config.

    Args:
        tmp_path: pytest-provided scratch directory.
    """
    findings = _ruff_findings("VALUE = 1\n", tmp_path)

    assert "D100" in findings, f"undocumented module not reported: {findings!r}"


def test_fully_documented_module_is_clean(tmp_path: Path) -> None:
    """A fully documented probe module yields no docstring findings.

    The other half of the non-vacuity argument. Without it, the probes above
    would still pass if the config reported D1xx on *everything* -- a gate
    that always fails is as uninformative as one that never does, and would
    mean the tests above prove nothing about discrimination.

    Args:
        tmp_path: pytest-provided scratch directory.
    """
    findings = _ruff_findings(
        '"""Probe module."""\n\n\n'
        "class Probe:\n"
        '    """A documented probe class."""\n\n'
        "    def probe(self) -> int:\n"
        '        """Return one."""\n'
        "        return 1\n\n\n"
        "def probe_function() -> int:\n"
        '    """Return one."""\n'
        "    return 1\n",
        tmp_path,
    )

    assert not findings, (
        "the docstring gate reports findings on a fully documented module "
        f"({findings!r}) -- it is firing indiscriminately, so the positive "
        "probes above prove nothing"
    )


# --- Reachability: Gate 1 actually runs the configured linter -----------------


def test_lint_script_runs_ruff_over_the_repository() -> None:
    """`scripts/lint.sh` runs the pinned ruff across the whole repository.

    A selected rule enforces nothing unless something dispatches the
    linter. This is the same reachability property `#401` had to establish
    for the pre-commit hooks, asserted here for the config route.
    """
    source = _LINT_SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"$RUFF" check .' in source, (
        "scripts/lint.sh no longer runs `ruff check .` over the repository "
        "root -- the docstring-coverage rules are configured but unreachable"
    )


def test_check_all_dispatches_the_lint_script() -> None:
    """Gate 1 (`check-all.sh`) dispatches `lint.sh`, so the gate is reachable.

    Completes the chain: `check-all.sh` -> `lint.sh` -> `ruff check .` ->
    the `D1` rules configured in `pyproject.toml`. Break any link and
    `check-all.sh` exit 0 stops meaning what CLAUDE.md says it means.
    """
    source = _CHECK_ALL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'run_check "Linting" "lint.sh"' in source, (
        "scripts/check-all.sh no longer dispatches lint.sh -- Gate 1 would "
        "stop enforcing the docstring-coverage gate"
    )


# --- Docs/implementation agreement --------------------------------------------


def test_claude_md_states_the_rule_that_is_enforced() -> None:
    """CLAUDE.md's docstring row names ruff `D1`, not an unmeasured percentage.

    Issue #351's acceptance criteria require the documented number and the
    enforced one to agree. The original row claimed ">=95%" via
    "pydocstyle / ruff D rules" while nothing ran at all; a row promising a
    percentage no tool computes is what created the issue, so re-introducing
    one must fail here.
    """
    text = _CLAUDE_MD_PATH.read_text(encoding="utf-8")
    row = next(
        (line for line in text.splitlines() if "**Docstring Coverage**" in line),
        None,
    )

    assert row is not None, "CLAUDE.md no longer documents a docstring row"
    assert "D1" in row, (
        f"the docstring row does not name the enforced rule family: {row!r}"
    )
    assert "95%" not in row, (
        f"the docstring row still promises a percentage no tool computes: {row!r}"
    )
    assert "pydocstyle" not in row, (
        "the docstring row still names pydocstyle as the tool; ruff "
        f"reimplements that rule family and is what actually runs: {row!r}"
    )
