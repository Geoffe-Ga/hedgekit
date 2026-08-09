"""Tests pinning every row of CLAUDE.md's Quality Standards table (issue #411).

The table is the first thing a contributor or agent reads about what this
repo enforces, and its rows were being believed. One of them was fiction:

```
| **Pylint Score** | >=9.0 | pylint |
```

`pylint` appeared in no `pyproject.toml`, no `requirements*.txt`, no
`constraints-quality.txt`, no `.pre-commit-config.yaml` and no `scripts/*.sh`;
it was not importable and had no pinned version anywhere. The row could not
fail, and it actively misinformed -- ">=9.0 pylint" was repeated as a hard
constraint in agent briefs written against this table.

That is the FOURTH instance of one defect class in the gate machinery itself:
#359 (a hook dispatched by no script), #401 (18 of 26 hooks undispatched),
#351 (a documented docstring threshold measured by nothing), and this. Fixing
the fourth instance by hand fixes the instance, not the class. So this module
makes the table SELF-VERIFYING: a row that names a tool the repo does not pin,
or a threshold the repo does not configure, fails the suite.

Two independent guards, because they catch different mistakes:

- `test_every_tool_named_in_the_table_is_pinned_in_constraints` is fully
  general and needs no registry edit. Every name in the Tool column must be a
  distribution pinned in `constraints-quality.txt` -- the file whose own
  header calls itself the "SINGLE version authority for the windbreak quality
  toolchain". A decorative `pylint` row fails here on the day it is written.
- `test_every_documented_metric_row_is_registered_as_enforced` requires each
  row to carry a recorded proof in `_ROW_ENFORCEMENT` that names the gate
  script dispatching it and verifies the documented number against the real
  configuration. Adding a row is therefore a decision someone records, never
  an assertion nobody checks.

Non-vacuity was demonstrated before this module was committed, by adding

```
| **Bogus Score** | >=9.0 | boguslint |
```

to the table: `test_every_tool_named_in_the_table_is_pinned_in_constraints`
failed with `boguslint` named, and
`test_every_documented_metric_row_is_registered_as_enforced` failed with
`Bogus Score` named. The row was then removed. Re-adding the real `pylint`
row fails identically, which is the assertion issue #411 asked for.

These assertions began life as Gate 1 RED: the pylint row was still present
and nothing read the table at all.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from tests.toolchain.test_toolchain_pins import _normalize, _parse_constraints

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_MD_PATH = _REPO_ROOT / "CLAUDE.md"
_DOCS_STANDARDS_PATH = _REPO_ROOT / ".claude" / "docs" / "quality-standards.md"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_CHECK_ALL_PATH = _SCRIPTS_DIR / "check-all.sh"

#: The heading that opens the table, and the one that closes the section.
#: Anchored to headings rather than to line numbers so re-ordering the
#: document does not silently narrow what this module reads.
_SECTION_START = "## 🎯 Quality Standards (Quick Reference)"
_SECTION_END = "## 📖 Project Overview"

#: A metric row: `| **Name** | threshold | tool, tool |`. The bold metric name
#: is what distinguishes a data row from the header and separator rows.
_TABLE_ROW = re.compile(r"^\|\s*\*\*(?P<metric>[^*]+)\*\*\s*\|(?P<rest>.*)\|\s*$")

#: `pylint` in any casing. Named explicitly so the tool this issue removed
#: cannot drift back into the docs as prose after the row itself is gone. The
#: REASONING for its removal deliberately lives in `pyproject.toml`, beside the
#: ruff `select` list it is a decision about, so these two normative documents
#: can stay free of the name entirely.
_PYLINT = re.compile(r"pylint", re.IGNORECASE)

#: The docs section whose every bullet is a tool name: `- **Name**: ...`.
#: Other sections bold things that are not tools ("Code Coverage", "Type
#: Hints"), so the general "is it pinned?" check is applied here, where the
#: bolded lead of a bullet is always a tool the reader is told to satisfy.
_LINTING_SECTION_START = "### Linting and Formatting"
_BULLET_TOOL = re.compile(r"^-\s+\*\*(?P<tool>[^*]+)\*\*", re.MULTILINE)


@dataclass(frozen=True)
class _RowEvidence:
    """What makes one documented metric row true.

    Attributes:
        tools: Names that must appear in the row's Tool column. The column
            states which tool a reader should look at to reproduce the
            number, so it must name the tool that ENFORCES it, not one that
            merely reports alongside it.
        gate: The `scripts/*.sh` that check-all.sh dispatches to enforce the
            row, or None when the row is deliberately not part of Gate 1.
        manual_reason: Why the row is out of Gate 1's reach. Non-empty
            exactly when `gate` is None, so "not automated" is always a
            recorded decision rather than an omission.
        prove: Raises AssertionError if the documented threshold is not the
            configured one. Given the row's full text so it can compare the
            published number against the real configuration.
    """

    tools: tuple[str, ...]
    gate: str | None
    manual_reason: str
    prove: Callable[[str], None]


def _claude_md() -> str:
    """Return the text of `CLAUDE.md`.

    Returns:
        The full source of the project context document.
    """
    return _CLAUDE_MD_PATH.read_text(encoding="utf-8")


def _quality_standards_section() -> str:
    """Return the Quality Standards section of `CLAUDE.md`.

    Returns:
        The text between the Quality Standards heading and the next section.

    Raises:
        AssertionError: If either boundary heading is missing, which would
            otherwise let this module read an empty section and pass
            vacuously.
    """
    text = _claude_md()
    start = text.find(_SECTION_START)
    assert start != -1, (
        f"CLAUDE.md no longer contains the heading {_SECTION_START!r}; the "
        "Quality Standards table cannot be located, so nothing below this "
        "point is actually being checked"
    )
    end = text.find(_SECTION_END, start)
    assert end != -1, (
        f"CLAUDE.md no longer contains the heading {_SECTION_END!r} after the "
        "Quality Standards section, so its end cannot be located"
    )
    return text[start:end]


def _table_rows() -> dict[str, str]:
    """Return the Quality Standards table keyed by metric name.

    Returns:
        A mapping of metric name (the bold first cell) to the row's full
        source line.

    Raises:
        AssertionError: If the section contains no metric rows.
    """
    rows: dict[str, str] = {}
    for line in _quality_standards_section().splitlines():
        match = _TABLE_ROW.match(line.strip())
        if match is not None:
            rows[match.group("metric").strip()] = line
    assert rows, (
        "no metric rows parsed out of CLAUDE.md's Quality Standards table -- "
        "either the table was removed or its shape changed, and every "
        "per-row assertion below would pass vacuously"
    )
    return rows


def _row_cells(row: str) -> list[str]:
    """Split one table row into its cells.

    Args:
        row: A full markdown table row, pipes included.

    Returns:
        The row's cells, stripped, without the empty leading and trailing
        fragments the outer pipes produce.
    """
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _row_tools(row: str) -> list[str]:
    """Return the tool names a table row names in its Tool column.

    Args:
        row: A full markdown table row.

    Returns:
        The comma-separated names in the final cell, with markdown emphasis
        and backticks removed.

    Raises:
        AssertionError: If the row does not have the expected three cells.
    """
    cells = _row_cells(row)
    assert len(cells) == 3, (
        f"expected a 3-cell Metric/Threshold/Tool row, got {len(cells)}: {row!r}"
    )
    return [
        name.strip().strip("*`")
        for name in cells[-1].split(",")
        if name.strip().strip("*`")
    ]


def _pyproject() -> dict[str, object]:
    """Return the parsed `pyproject.toml`.

    Returns:
        The decoded TOML document.
    """
    return tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))


def _script_source(name: str) -> str:
    """Return the text of a gate script.

    Args:
        name: The script's file name within `scripts/`.

    Returns:
        The script's full source.

    Raises:
        AssertionError: If the script does not exist.
    """
    path = _SCRIPTS_DIR / name
    assert path.is_file(), f"scripts/{name} does not exist"
    return path.read_text(encoding="utf-8")


def _coverage_report_config() -> dict[str, object]:
    """Return `[tool.coverage.report]` from `pyproject.toml`.

    Returns:
        The coverage report table, or an empty mapping when absent.
    """
    tool = _pyproject().get("tool", {})
    assert isinstance(tool, dict)
    coverage = tool.get("coverage", {})
    assert isinstance(coverage, dict)
    report = coverage.get("report", {})
    assert isinstance(report, dict)
    return report


def _coverage_run_config() -> dict[str, object]:
    """Return `[tool.coverage.run]` from `pyproject.toml`.

    Returns:
        The coverage run table, or an empty mapping when absent.
    """
    tool = _pyproject().get("tool", {})
    assert isinstance(tool, dict)
    coverage = tool.get("coverage", {})
    assert isinstance(coverage, dict)
    run = coverage.get("run", {})
    assert isinstance(run, dict)
    return run


def _prove_code_coverage(row: str) -> None:
    """Prove the documented line-coverage floor is the configured one.

    Args:
        row: The Code Coverage table row.

    Raises:
        AssertionError: If the documented 90% is not what pytest-cov
            enforces, in both the config default and the gate script's
            explicit flag.
    """
    assert "90%" in row, f"the Code Coverage row no longer states 90%: {row!r}"
    fail_under = _coverage_report_config().get("fail_under")
    assert fail_under == 90, (
        "CLAUDE.md documents a 90% coverage floor but "
        f"[tool.coverage.report] fail_under is {fail_under!r}"
    )
    assert "--cov-fail-under=90" in _script_source("coverage.sh"), (
        "scripts/coverage.sh no longer passes --cov-fail-under=90, so Gate 1 "
        "would stop enforcing the documented floor even though pyproject.toml "
        "still declares it"
    )


def _prove_branch_coverage(row: str) -> None:
    """Prove the branch-coverage row describes what is actually measured.

    Branch coverage is switched ON and folded into the SINGLE combined floor
    proved by `_prove_code_coverage`; coverage.py computes one percentage
    over statements and branch outcomes together. There is no second
    threshold. The row previously advertised ">=85%", a number nothing in
    the repo computes or compares against -- so a run whose branch coverage
    sat at 60% passed while the table said it could not.

    Correcting that number is not a lowered gate: no 85% check ever existed
    to lower, and requiring the COMBINED figure to clear 90% is strictly
    stronger than a bare line-coverage floor, because uncovered branches drag
    the combined figure down.

    Args:
        row: The Branch Coverage table row.

    Raises:
        AssertionError: If branch measurement is off, or if the row
            re-introduces a floor no tool computes.
    """
    assert _coverage_run_config().get("branch") is True, (
        "CLAUDE.md documents branch coverage but [tool.coverage.run] no "
        "longer sets branch = true, so branches are not measured at all"
    )
    assert "--cov-branch" in _script_source("coverage.sh"), (
        "scripts/coverage.sh no longer passes --cov-branch, so the combined "
        "figure the documented floor applies to would silently become "
        "line-only"
    )
    assert "85%" not in row, (
        "the Branch Coverage row promises an 85% floor again; nothing "
        f"computes or compares a separate branch percentage: {row!r}"
    )


def _prove_docstring_coverage(row: str) -> None:
    """Prove the docstring row names the rule family ruff actually enforces.

    Args:
        row: The Docstring Coverage table row.

    Raises:
        AssertionError: If ruff's select list does not enable the missing-
            docstring family, or the row reverts to an unmeasured
            percentage.
    """
    assert "D1" in row, f"the docstring row no longer names ruff D1: {row!r}"
    assert "95%" not in row, (
        f"the docstring row promises a percentage no tool computes: {row!r}"
    )
    tool = _pyproject().get("tool", {})
    assert isinstance(tool, dict)
    ruff = tool.get("ruff", {})
    assert isinstance(ruff, dict)
    lint = ruff.get("lint", {})
    assert isinstance(lint, dict)
    select = lint.get("select", [])
    assert isinstance(select, list)
    assert any(str(code).startswith("D1") or str(code) == "D" for code in select), (
        "CLAUDE.md documents ruff D1 docstring coverage but ruff's select "
        f"list does not enable that family: {select!r}"
    )


def _prove_mutation_score(row: str) -> None:
    """Prove the mutation row is a real script and deliberately not in Gate 1.

    Mutation testing is the MANUAL pre-v1.0.0 release gate by owner
    directive (issue #107). The row is honest only if it says so and if
    check-all.sh really does not dispatch it -- a row documented as manual
    that quietly ran in Gate 1 would be the same class of untruth as one
    documented as automated that ran nowhere.

    Args:
        row: The Mutation Score table row.

    Raises:
        AssertionError: If the row hides its manual status, if mutation.sh
            does not run mutmut, or if Gate 1 dispatches it after all.
    """
    assert "manual" in row.lower(), (
        "the Mutation Score row no longer states that it is a manual gate, "
        f"so a reader would expect check-all.sh to enforce it: {row!r}"
    )
    assert "mutmut" in _script_source("mutation.sh"), (
        "scripts/mutation.sh no longer runs mutmut"
    )
    assert "mutation.sh" not in _script_source("check-all.sh"), (
        "scripts/check-all.sh dispatches mutation.sh, but issue #107 makes "
        "the mutation gate manual and CLAUDE.md documents it as such"
    )


def _prove_cyclomatic_complexity(row: str) -> None:
    """Prove the complexity row credits the tool that can actually fail.

    `scripts/complexity.sh` runs radon with `|| true` -- radon REPORTS and
    cannot fail the gate. xenon is the enforcing half, and `--max-absolute B`
    is radon's B grade, i.e. cyclomatic complexity 6-10, which is exactly the
    documented "<=10 per function". Crediting radon in the Tool column sent a
    reader to the half of the script that cannot fail.

    Args:
        row: The Cyclomatic Complexity table row.

    Raises:
        AssertionError: If xenon no longer enforces the B ceiling, or if the
            row credits radon as the enforcing tool.
    """
    assert "10" in row, f"the complexity row no longer states <=10: {row!r}"
    source = _script_source("complexity.sh")
    assert "--max-absolute B" in source, (
        "scripts/complexity.sh no longer enforces xenon --max-absolute B, "
        "which is what makes the documented <=10 per function true"
    )
    assert "radon" not in [tool.lower() for tool in _row_tools(row)], (
        "the complexity row credits radon, which runs with `|| true` in "
        f"scripts/complexity.sh and cannot fail the gate: {row!r}"
    )


def _prove_security(row: str) -> None:
    """Prove the security row's tools are the ones the gate script runs.

    Args:
        row: The Security Vulnerabilities table row.

    Raises:
        AssertionError: If security.sh no longer runs both named tools.
    """
    source = _script_source("security.sh")
    assert "--print-tool bandit" in source, (
        "scripts/security.sh no longer resolves bandit through the pinned "
        "toolchain, so the documented bandit scan may not run"
    )
    assert "-m pip_audit" in source, (
        "scripts/security.sh no longer runs pip-audit as a module of the "
        "pinned interpreter"
    )
    assert "0" in row, f"the security row no longer states a zero-tolerance: {row!r}"


#: Every documented metric, with the recorded proof that something enforces
#: it. Keys must equal the metric names in CLAUDE.md's table exactly -- see
#: `test_every_documented_metric_row_is_registered_as_enforced`.
_ROW_ENFORCEMENT: dict[str, _RowEvidence] = {
    "Code Coverage": _RowEvidence(
        tools=("pytest-cov",),
        gate="coverage.sh",
        manual_reason="",
        prove=_prove_code_coverage,
    ),
    "Branch Coverage": _RowEvidence(
        tools=("pytest-cov",),
        gate="coverage.sh",
        manual_reason="",
        prove=_prove_branch_coverage,
    ),
    "Docstring Coverage": _RowEvidence(
        tools=("ruff",),
        gate="lint.sh",
        manual_reason="",
        prove=_prove_docstring_coverage,
    ),
    "Mutation Score": _RowEvidence(
        tools=("mutmut",),
        gate=None,
        manual_reason=(
            "The mutation gate is MANUAL by owner directive (issue #107): it "
            "is run before a v1.0.0 ship, locally or via mutation-gate.yml, "
            "never as an automated CI or pre-commit check. The row states "
            "that in its threshold cell, and _prove_mutation_score asserts "
            "check-all.sh really does not dispatch it."
        ),
        prove=_prove_mutation_score,
    ),
    "Cyclomatic Complexity": _RowEvidence(
        tools=("xenon",),
        gate="complexity.sh",
        manual_reason="",
        prove=_prove_cyclomatic_complexity,
    ),
    "Security Vulnerabilities": _RowEvidence(
        tools=("bandit", "pip-audit"),
        gate="security.sh",
        manual_reason="",
        prove=_prove_security,
    ),
}


def _registered_rows() -> list[str]:
    """Return the registered metric names, for parametrisation.

    Returns:
        The keys of `_ROW_ENFORCEMENT`, sorted for a stable test order.
    """
    return sorted(_ROW_ENFORCEMENT)


# --- The general guard: no row may name a tool the repo does not pin ---------


def test_every_tool_named_in_the_table_is_pinned_in_constraints() -> None:
    """Every Tool-column name is a distribution pinned in constraints.

    This is the guard that would have caught issue #411 the day the pylint
    row was written, with no registry entry to maintain: the Tool column
    names what a reader is told to run, and `constraints-quality.txt` calls
    itself the single version authority for the quality toolchain. A name in
    one and not the other means the table advertises a check the repo has no
    pinned way to perform.
    """
    pinned = {_normalize(name) for name in _parse_constraints()}
    unpinned: dict[str, str] = {}
    for metric, row in _table_rows().items():
        for tool in _row_tools(row):
            if _normalize(tool) not in pinned:
                unpinned[tool] = metric

    assert not unpinned, (
        "CLAUDE.md's Quality Standards table names tools that are not pinned "
        "in constraints-quality.txt, so the rows document checks the repo "
        "cannot run: "
        + ", ".join(f"{tool!r} (row {metric!r})" for tool, metric in unpinned.items())
    )


def test_every_documented_metric_row_is_registered_as_enforced() -> None:
    """The table's rows and the enforcement registry name the same metrics.

    Both directions matter. An unregistered row is a claim with no recorded
    proof -- the shape issue #411 removed. A registered row with no table
    entry is a stale proof that keeps passing after the claim it defended is
    gone, which would make the per-row assertions below quietly meaningless.
    """
    documented = set(_table_rows())
    registered = set(_ROW_ENFORCEMENT)

    unregistered = sorted(documented - registered)
    stale = sorted(registered - documented)
    assert documented == registered, (
        "CLAUDE.md's Quality Standards table and _ROW_ENFORCEMENT disagree. "
        f"Documented but unregistered (no recorded proof): {unregistered}. "
        f"Registered but undocumented (stale proof): {stale}."
    )


@pytest.mark.parametrize("metric", _registered_rows())
def test_documented_threshold_matches_the_enforced_configuration(metric: str) -> None:
    """Each row's published number is the one the repo really configures.

    Args:
        metric: The metric name being verified.
    """
    rows = _table_rows()
    assert metric in rows, f"CLAUDE.md no longer documents a {metric!r} row"
    _ROW_ENFORCEMENT[metric].prove(rows[metric])


@pytest.mark.parametrize("metric", _registered_rows())
def test_row_tool_column_names_the_enforcing_tool(metric: str) -> None:
    """Each row credits the tool that can fail, not one that only reports.

    Args:
        metric: The metric name being verified.
    """
    evidence = _ROW_ENFORCEMENT[metric]
    named = {tool.lower() for tool in _row_tools(_table_rows()[metric])}
    missing = {tool for tool in evidence.tools if tool.lower() not in named}

    assert not missing, (
        f"the {metric!r} row does not name its enforcing tool(s) "
        f"{sorted(missing)}; it names {sorted(named)}"
    )


@pytest.mark.parametrize("metric", _registered_rows())
def test_gate1_dispatches_each_automated_rows_script(metric: str) -> None:
    """Each automated row's gate script is reachable from check-all.sh.

    A row whose script exists but which Gate 1 never dispatches is the
    defect of issue #359 (a shellcheck hook no script ran) applied to the
    documentation table: the check is real, and still nothing runs it.

    Args:
        metric: The metric name being verified.
    """
    evidence = _ROW_ENFORCEMENT[metric]
    if evidence.gate is None:
        assert evidence.manual_reason, (
            f"the {metric!r} row is registered as outside Gate 1 with no "
            "reason recorded; 'not automated' must be a stated decision"
        )
        return

    assert not evidence.manual_reason, (
        f"the {metric!r} row registers both a Gate 1 script and a reason for "
        "being manual; exactly one of those can be true"
    )
    assert f'"{evidence.gate}"' in _script_source("check-all.sh"), (
        f"scripts/check-all.sh no longer dispatches {evidence.gate}, so Gate "
        f"1 would stop enforcing the documented {metric!r} row"
    )


# --- The removed row must stay removed ----------------------------------------


def test_pylint_is_absent_from_the_quality_standards_documentation() -> None:
    """No quality doc claims a pylint gate this repo does not have.

    Issue #411's row is gone, and the reasoning is on the record in the PR:
    ruff already reimplements pylint's rule set natively as its `PL` family
    and is the pinned, dispatched linter, so a second linter would buy a
    second opinion on the same rules. A ">=9.0 pylint score" additionally
    cannot be reproduced by anything except pylint -- it is a weighted
    aggregate over pylint's own message counts, and unlike every other row
    here it is a TOLERANCE for defects rather than a floor.

    Prose is checked as well as the table: the claim was echoed in
    `.claude/docs/quality-standards.md`, and re-introducing it there would
    misinform exactly as effectively.
    """
    offenders = {
        path.name: [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if _PYLINT.search(line)
        ]
        for path in (_CLAUDE_MD_PATH, _DOCS_STANDARDS_PATH)
    }
    found = {name: lines for name, lines in offenders.items() if lines}

    assert not found, (
        "pylint is claimed as a quality gate again, but it is not pinned in "
        "constraints-quality.txt, not installed, and dispatched by no "
        f"script (issue #411): {found}"
    )


def test_linting_docs_name_only_tools_the_repo_pins() -> None:
    """The docs' Linting and Formatting bullets name only pinned tools.

    The same defect the table carried had spread through this section as
    prose. Alongside pylint it required **Black** ("line length 88") and
    **isort** ("import sorting per configuration") -- both deleted from this
    repo by ADR-0002, which made ruff the single formatter authority, and
    both already asserted absent from the hooks and dev requirements by
    `test_toolchain_pins`. It also required **Safety** for dependency
    scanning, where the repo pins and runs pip-audit.

    Every bullet in this section leads with the tool it obliges a contributor
    to satisfy, so requiring each to be a pinned distribution catches the
    whole family at once rather than one name at a time.
    """
    text = _DOCS_STANDARDS_PATH.read_text(encoding="utf-8")
    start = text.find(_LINTING_SECTION_START)
    assert start != -1, (
        f"{_DOCS_STANDARDS_PATH.name} no longer has a "
        f"{_LINTING_SECTION_START!r} section, so this check would pass "
        "vacuously"
    )
    body = text[start + len(_LINTING_SECTION_START) :]
    end = body.find("\n### ")
    section = body if end == -1 else body[:end]

    named = [match.group("tool").strip() for match in _BULLET_TOOL.finditer(section)]
    assert named, "no tool bullets parsed out of the Linting and Formatting section"

    pinned = {_normalize(name) for name in _parse_constraints()}
    unpinned = [tool for tool in named if _normalize(tool) not in pinned]

    assert not unpinned, (
        "the Linting and Formatting standards require tools that are not "
        "pinned in constraints-quality.txt, so a contributor is told to "
        f"satisfy a check this repo cannot run: {unpinned}"
    )
