#!/usr/bin/env python3
"""Measure quality metrics for the dashboard collector (issue #122).

`.github/workflows/metrics.yml` runs `scripts/collect_metrics.py` in *script
mode*, which executes ``scripts/<gate>.sh --metrics`` and parses **pure JSON
from stdout**. Anything else -- a banner, a progress bar, a traceback -- is a
`json.JSONDecodeError` that degrades the metric to ``null``/``"unknown"``.
Before this module existed, no gate script implemented ``--metrics``, so all
ten quality tiles rendered N/A while the workflow stayed green.

This module is the measuring half of that contract. Each gate script resolves
its tools from the pinned toolchain (`scripts/toolchain-env.sh`) and hands the
absolute paths here, so *which* binary produced a number remains a property of
the repository rather than of the caller's PATH (issue #366). The gate scripts
stay thin wrappers; the parsing lives here where it is unit-testable against
synthetic tool output instead of only against whatever the real tools happen
to emit today.

TWO RULES PULL IN OPPOSITE DIRECTIONS, AND BOTH ARE LOAD-BEARING

*Measurement, not a gate.* ``--metrics`` always exits 0 and reports the number
even when the number is bad. Three lint violations means ``{"violations": 3}``
and exit 0; propagating the tool's exit status would just recreate the N/A
tiles from the other side.

*Fail closed.* A measurement that cannot be **proved** degrades to ``None``,
never to a healthy-looking zero. Zero security issues and an unmeasured
security scan are different facts, and `collect_metrics.py` renders a real 0
as a green tile but a `None` as a gray "unknown" one -- so `None` is the only
honest output when the evidence is missing. Every findings probe therefore
cross-checks the tool's exit status against the count it parsed
(`_corroborated_count`) and refuses the pair when they contradict each other.

Mutation score is deliberately absent: mutmut is the manual pre-v1.0.0 release
gate (owner directive, issue #107), its gray tile is correct, and running it
here would blow the collector's 300 s per-script timeout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: Root of the checkout that owns this file. Every tool runs with this as its
#: working directory so relative targets (``windbreak/``, ``tests/``) mean the
#: same thing regardless of where the collector was invoked from.
_REPO_ROOT: Final = Path(__file__).resolve().parent.parent

#: mypy's end-of-run summary. The exit status alone cannot distinguish "three
#: type errors" from "mypy could not start", so the count comes from here.
_MYPY_SUMMARY: Final = re.compile(r"^Found (\d+) errors? in ", re.MULTILINE)

#: Exit statuses a findings tool may legitimately use: 0 for a clean run, 1
#: for "ran fine, found something". Anything else is the tool failing, not the
#: code failing, and must not be read as a measurement.
_FINDINGS_STATUSES: Final = frozenset({0, 1})

#: How much of a tool's own stdout to echo for a human running a probe by
#: hand. It is machine payload -- radon's is half a megabyte -- so it is echoed
#: head-and-tail rather than in full: the head carries a parse error's first
#: line, the tail carries pytest's summary, and neither buries the terminal.
_ECHO_EDGE_CHARS: Final = 1000


@dataclass(frozen=True)
class _ToolRun:
    """One measured tool invocation.

    Attributes:
        returncode: The tool's exit status.
        stdout: Everything the tool wrote to stdout, captured for parsing.
    """

    returncode: int
    stdout: str


def _abridged(text: str) -> str:
    """Shorten a tool's stdout to its informative ends.

    Args:
        text: The tool's full stdout.

    Returns:
        The text unchanged when it is short, otherwise its head and tail with
        an explicit marker for the elided middle.
    """
    if len(text) <= 2 * _ECHO_EDGE_CHARS:
        return text
    elided = len(text) - 2 * _ECHO_EDGE_CHARS
    head = text[:_ECHO_EDGE_CHARS]
    tail = text[-_ECHO_EDGE_CHARS:]
    return f"{head}\n... [{elided} characters elided] ...\n{tail}"


def _run(command: list[str], *, env: dict[str, str] | None = None) -> _ToolRun:
    """Run a measurement tool, keeping its chatter off this process's stdout.

    Both streams the tool produced are forwarded to *stderr* so an operator
    running a probe by hand still sees the evidence, while stdout stays
    reserved for the single JSON object the collector parses. The tool's own
    stdout is abridged on the way -- it is machine payload, not a report.

    The directory holding the tool is prepended to ``PATH``, which is what
    ``. .venv/bin/activate`` does and what the gate scripts' callers normally
    have already done. Handing over an absolute path settles which binary runs
    but not which binaries *it* runs: pytest's suite shells out to
    `lint-imports` by bare name, pre-commit to its hook binaries. Without this,
    a probe launched from an unactivated shell reports the caller's missing
    tools as this repository's failures -- three phantom test failures on the
    Test Count tile, observed while wiring issue #122.

    Args:
        command: Argument vector; ``command[0]`` is an absolute tool path
            resolved by the calling gate script.
        env: Extra environment variables to layer over the inherited ones.

    Returns:
        The tool's exit status and captured stdout.
    """
    tool_bin = str(Path(command[0]).resolve().parent)
    merged = {
        **os.environ,
        "PATH": os.pathsep.join([tool_bin, os.environ.get("PATH", "")]),
        **(env or {}),
    }
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env=merged,
        check=False,
    )
    sys.stderr.write(_abridged(completed.stdout))
    sys.stderr.write(completed.stderr)
    return _ToolRun(completed.returncode, completed.stdout)


def _corroborated_count(returncode: int, count: int | None) -> int | None:
    """Return a finding count only when the tool's exit status agrees with it.

    A findings tool exits 0 with nothing to report and 1 when it has something.
    Two situations must never reach the dashboard as a number:

    * an exit status outside that pair, which means the tool itself failed and
      whatever landed on stdout does not describe this repository;
    * a status that contradicts the parse -- exit 1 with an empty report, or
      exit 0 with findings -- which means the parse read something other than
      the run that just happened.

    Args:
        returncode: The tool's exit status.
        count: The parsed number of findings, or None if unparseable.

    Returns:
        The count when it is corroborated, otherwise None.
    """
    if count is None or count < 0:
        return None
    if returncode not in _FINDINGS_STATUSES:
        return None
    if (returncode == 1) != (count > 0):
        return None
    return count


def _json_array_length(text: str) -> int | None:
    """Count the entries of a JSON array emitted by a linter.

    Args:
        text: Raw tool stdout.

    Returns:
        The array's length, or None when the text is not a JSON array.
    """
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, list):
        return None
    return len(payload)


def _non_empty_line_count(text: str) -> int:
    """Count the non-blank lines of a line-per-finding report.

    Args:
        text: Raw tool stdout.

    Returns:
        Number of lines carrying content.
    """
    return len([line for line in text.splitlines() if line.strip()])


def probe_lint(ruff: str, python: str, pre_commit: str) -> dict[str, int | None]:
    """Count everything `scripts/lint.sh` gates on.

    The gate has three halves and the tile must reflect all of them: ruff, the
    float-lint AST pass, and the pinned shellcheck hook. ruff and float-lint
    report countable findings. The shellcheck hook does not expose a
    machine-readable count, so a failing hook makes the total *unprovable* and
    the metric degrades to None -- reporting ruff's 0 while shell lint is red
    would paint a green tile over a failing gate.

    Args:
        ruff: Absolute path to the pinned ruff.
        python: Absolute path to the pinned interpreter (runs float-lint).
        pre_commit: Absolute path to the pinned pre-commit.

    Returns:
        ``{"violations": <int>}``, or ``{"violations": None}`` when any half
        could not be counted.
    """
    ruff_run = _run([ruff, "check", ".", "--output-format", "json"])
    ruff_count = _corroborated_count(
        ruff_run.returncode, _json_array_length(ruff_run.stdout)
    )

    float_run = _run([python, "scripts/lint_no_floats.py"])
    float_count = _corroborated_count(
        float_run.returncode, _non_empty_line_count(float_run.stdout)
    )

    shell_run = _run([pre_commit, "run", "shellcheck", "--all-files"])
    shell_count = 0 if shell_run.returncode == 0 else None

    if ruff_count is None or float_count is None or shell_count is None:
        return {"violations": None}
    return {"violations": ruff_count + float_count + shell_count}


def probe_typecheck(mypy: str) -> dict[str, int | None]:
    """Count mypy's type errors over the same targets the gate checks.

    The cache directory is a throwaway, matching `scripts/typecheck.sh`'s
    cold-cache enforcement (issue #179): a warm incremental cache skips
    re-validating cross-module re-exports, so a warm count is not the count CI
    would produce.

    Args:
        mypy: Absolute path to the pinned mypy.

    Returns:
        ``{"errors": <int>}``, or ``{"errors": None}`` when mypy neither
        succeeded nor printed a summary line to read the count from.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        run = _run([mypy, "windbreak/", "scripts/"], env={"MYPY_CACHE_DIR": cache_dir})
    if run.returncode == 0:
        return {"errors": 0}
    match = _MYPY_SUMMARY.search(run.stdout)
    if match is None:
        return {"errors": None}
    return {"errors": _corroborated_count(run.returncode, int(match.group(1)))}


def probe_security(bandit: str) -> dict[str, int | None]:
    """Count bandit's findings over the package the gate scans.

    Only bandit is measured here because ``bandit_issues`` is the key the
    collector reads. `scripts/security.sh`'s other two halves (pip-audit and
    the detect-secrets baseline hook) stay on the gate path; neither exposes a
    finding count that shares this tile's meaning.

    The report is written to a file rather than read off stdout: bandit
    prefixes its stdout with a ``Working...`` progress bar even when stdout is
    a pipe, so ``-f json`` alone yields output that is *not* JSON. Parsing that
    stream would have degraded the security tile to a permanent null.

    Args:
        bandit: Absolute path to the pinned bandit.

    Returns:
        ``{"bandit_issues": <int>}``, or None when the scan produced no usable
        report -- a security tile reading zero from a scan that never ran is
        the worst available failure mode.
    """
    with tempfile.TemporaryDirectory() as work_dir:
        report = Path(work_dir) / "bandit.json"
        run = _run(
            [
                bandit,
                "-c",
                "pyproject.toml",
                "-r",
                "windbreak/",
                "-f",
                "json",
                "-o",
                str(report),
            ]
        )
        payload = _load_json_file(report)
    count: int | None = None
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        count = len(payload["results"])
    return {"bandit_issues": _corroborated_count(run.returncode, count)}


def _artifact_dir_for(source_root: Path) -> Path:
    """Locate the shared pytest artifact directory for a checkout.

    Keyed by the checkout's path so parallel worktrees (the Ralph fleet runs
    several at once) never read each other's numbers.

    Args:
        source_root: Root of the checkout being measured.

    Returns:
        Directory holding this checkout's coverage and JUnit artifacts.
    """
    digest = hashlib.sha256(str(source_root).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"windbreak-metrics-{digest}"


def _newest_input_mtime(source_root: Path) -> float:
    """Return the newest modification time across the measured inputs.

    Args:
        source_root: Root of the checkout being measured.

    Returns:
        Epoch seconds of the most recently modified input, or 0.0 when none
        exist.
    """
    newest = 0.0
    for package in ("windbreak", "tests"):
        for path in (source_root / package).rglob("*.py"):
            newest = max(newest, path.stat().st_mtime)
    config = source_root / "pyproject.toml"
    if config.is_file():
        newest = max(newest, config.stat().st_mtime)
    return newest


def _artifacts_are_current(artifacts: tuple[Path, Path], source_root: Path) -> bool:
    """Report whether cached artifacts still describe the current tree.

    Coverage and test counts come from one pytest run, but the collector asks
    for them through two separate script invocations. Reusing the artifacts
    halves a two-and-a-half-minute collection -- but only while the tree they
    measured is unchanged, so any newer source or test file invalidates them.
    Staleness resolves toward re-running, never toward publishing an old
    number as a current one.

    Args:
        artifacts: The coverage-report and JUnit-report paths.
        source_root: Root of the checkout being measured.

    Returns:
        True when both artifacts exist and postdate every measured input.
    """
    if not all(path.is_file() for path in artifacts):
        return False
    oldest_artifact = min(path.stat().st_mtime for path in artifacts)
    return oldest_artifact > _newest_input_mtime(source_root)


def _pytest_artifacts(
    pytest_bin: str,
    artifact_dir: Path | None,
    source_root: Path | None,
) -> tuple[Path, Path]:
    """Produce (or reuse) one pytest run's coverage and JUnit reports.

    Args:
        pytest_bin: Absolute path to the pinned pytest.
        artifact_dir: Directory to hold the reports; defaults to a temporary
            directory keyed by the checkout.
        source_root: Root of the checkout being measured; defaults to the
            checkout that owns this file.

    Returns:
        The coverage-report path and the JUnit-report path, either of which
        may be absent if pytest failed to write it.
    """
    root = source_root or _REPO_ROOT
    directory = artifact_dir or _artifact_dir_for(root)
    coverage_json = directory / "coverage.json"
    junit_xml = directory / "junit.xml"
    if _artifacts_are_current((coverage_json, junit_xml), root):
        return coverage_json, junit_xml

    directory.mkdir(parents=True, exist_ok=True)
    for stale in (coverage_json, junit_xml):
        stale.unlink(missing_ok=True)
    _run(
        [
            pytest_bin,
            "-q",
            # The collector kills a script at 300 s and reads the corpse as an
            # N/A tile -- the exact silence issue #122 removes. Serially this
            # suite takes ~170 s on a developer machine, and CI's quality job
            # needs ~6 minutes for the same work, so the serial margin is not
            # one to bet four tiles on. pytest-xdist is already pinned in
            # requirements-dev.txt; measured over the whole suite it halves the
            # wall clock with an identical pass count and identical coverage.
            "-n",
            "auto",
            "--cov=windbreak",
            "--cov-branch",
            f"--cov-report=json:{coverage_json}",
            f"--junitxml={junit_xml}",
            "--cov-fail-under=0",
            "tests/",
        ]
    )
    return coverage_json, junit_xml


def _load_json_file(path: Path) -> Any:
    """Read a JSON document written by a tool, tolerating absence and garbage.

    Args:
        path: File to read.

    Returns:
        The decoded document, or None when it is missing or unparseable.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _percentage(totals: Any, key: str) -> float | None:
    """Extract one percentage from a coverage report's ``totals`` mapping.

    Args:
        totals: The report's ``totals`` value, of unverified shape.
        key: Percentage key to read.

    Returns:
        The percentage rounded to two decimals, or None when it is absent or
        outside the 0-100 range a percentage must occupy.
    """
    if not isinstance(totals, dict):
        return None
    value = totals.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not 0.0 <= float(value) <= 100.0:
        return None
    return round(float(value), 2)


def probe_coverage(
    pytest_bin: str,
    artifact_dir: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, float | None]:
    """Measure line and branch coverage from one pytest run.

    Args:
        pytest_bin: Absolute path to the pinned pytest.
        artifact_dir: Override for the shared artifact directory (tests).
        source_root: Override for the measured checkout (tests).

    Returns:
        ``{"coverage_pct": ..., "branch_coverage_pct": ...}``, each None when
        the report did not carry it.
    """
    coverage_json, _ = _pytest_artifacts(pytest_bin, artifact_dir, source_root)
    document = _load_json_file(coverage_json)
    totals = document.get("totals") if isinstance(document, dict) else None
    return {
        "coverage_pct": _percentage(totals, "percent_covered"),
        "branch_coverage_pct": _percentage(totals, "percent_branches_covered"),
    }


def _junit_totals(path: Path) -> dict[str, int] | None:
    """Read the counts off a JUnit report's ``testsuite`` element.

    Args:
        path: JUnit XML report written by pytest.

    Returns:
        Mapping of ``tests``/``failures``/``errors``/``skipped`` to counts, or
        None when the report is missing, malformed, or incomplete.
    """
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError):
        return None
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return None
    totals: dict[str, int] = {}
    for name in ("tests", "failures", "errors", "skipped"):
        raw = suite.get(name)
        if raw is None or not raw.lstrip("-").isdigit():
            return None
        totals[name] = int(raw)
    return totals


def probe_tests(
    pytest_bin: str,
    artifact_dir: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, int | None]:
    """Count collected, passing, failing and skipped tests from one run.

    Args:
        pytest_bin: Absolute path to the pinned pytest.
        artifact_dir: Override for the shared artifact directory (tests).
        source_root: Override for the measured checkout (tests).

    Returns:
        The four counts the collector reads, all None when the run produced no
        readable report.
    """
    _, junit_xml = _pytest_artifacts(pytest_bin, artifact_dir, source_root)
    totals = _junit_totals(junit_xml)
    if totals is None:
        return {
            "tests_total": None,
            "tests_passed": None,
            "tests_failed": None,
            "tests_skipped": None,
        }
    passed = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    return {
        "tests_total": totals["tests"],
        "tests_passed": max(passed, 0),
        "tests_failed": totals["failures"],
        "tests_skipped": totals["skipped"],
    }


def _mean(values: list[float]) -> float | None:
    """Average a list of measurements.

    Args:
        values: Measured values.

    Returns:
        The mean rounded to two decimals, or None for an empty list -- no
        blocks analysed is no measurement, and a complexity of 0.0 would sail
        under the ``<= 10`` threshold as the healthiest possible tile.
    """
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _radon_json(radon: str, subcommand: str) -> dict[str, Any] | None:
    """Run one radon subcommand and decode its JSON report.

    Args:
        radon: Absolute path to the pinned radon.
        subcommand: ``"cc"`` or ``"mi"``.

    Returns:
        The decoded report, or None when radon failed or emitted non-JSON.
    """
    run = _run([radon, subcommand, "-j", "windbreak/"])
    if run.returncode != 0:
        return None
    try:
        payload = json.loads(run.stdout)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def probe_complexity(radon: str) -> dict[str, float | None]:
    """Average cyclomatic complexity per block and maintainability per file.

    radon's ``cc -j`` already flattens classes, functions and methods into each
    file's top-level list, so averaging that list reproduces radon's own
    ``cc -a`` figure exactly. Files radon could not parse arrive as an ``error``
    object rather than a list and are skipped instead of counted as zero.

    Args:
        radon: Absolute path to the pinned radon.

    Returns:
        ``{"cyclomatic_avg": ..., "maintainability_avg": ...}``, each None when
        radon produced no usable report.
    """
    cc_report = _radon_json(radon, "cc")
    mi_report = _radon_json(radon, "mi")

    complexities: list[float] = []
    for blocks in (cc_report or {}).values():
        if not isinstance(blocks, list):
            continue
        complexities.extend(
            float(block["complexity"])
            for block in blocks
            if isinstance(block, dict) and isinstance(block.get("complexity"), int)
        )

    indices: list[float] = []
    for entry in (mi_report or {}).values():
        if isinstance(entry, dict) and isinstance(entry.get("mi"), (int, float)):
            indices.append(float(entry["mi"]))

    return {
        "cyclomatic_avg": _mean(complexities) if cc_report is not None else None,
        "maintainability_avg": _mean(indices) if mi_report is not None else None,
    }


def _is_documentable(name: str) -> bool:
    """Report whether ruff's D1 family checks a symbol with this name.

    pydocstyle -- which ruff reimplements natively as the ``D`` rules -- checks
    public symbols (D101/D102/D103/D106), magic methods (D105) and ``__init__``
    (D107), but not other underscore-prefixed names. Counting those private
    helpers would inflate the denominator, and a larger denominator can only
    push the reported percentage *up*, which is the wrong direction for a
    number an operator reads as "how documented is this".

    Args:
        name: Symbol name.

    Returns:
        True when ruff would require a docstring on the symbol.
    """
    if name.startswith("__") and name.endswith("__"):
        return True
    return not name.startswith("_")


def _count_documentable(tree: ast.AST) -> int:
    """Count the documentable definitions in one parsed module.

    The module itself counts (D100/D104). A definition counts only when its own
    name is documentable *and* every enclosing definition is too, matching
    pydocstyle's rule that members of a private scope are private.

    Args:
        tree: Parsed module.

    Returns:
        Number of definitions ruff's D1 family would check, module included.
    """
    total = 1
    stack: list[tuple[ast.AST, bool]] = [(tree, True)]
    while stack:
        node, visible = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                child_visible = visible and _is_documentable(child.name)
                total += 1 if child_visible else 0
                stack.append((child, child_visible))
            else:
                stack.append((child, visible))
    return total


def _documentable_symbols(listing: str) -> int | None:
    """Count documentable symbols across the files ruff says it will check.

    A file ruff can lint but this pass cannot parse voids the whole count: the
    two halves of the ratio would otherwise stop describing the same file set,
    and a denominator missing a file can only push the reported percentage up.

    "Nothing was listed" needs no special case here. It returns 0, and
    `probe_docs` already refuses a zero denominator -- one place to decide that
    an empty measurement is not full coverage, rather than two that must agree.

    Args:
        listing: ruff's ``--show-files`` output, one path per line.

    Returns:
        The total, or None when a listed file could not be parsed.
    """
    total = 0
    for line in listing.splitlines():
        path = Path(line.strip())
        if not line.strip() or path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            return None
        total += _count_documentable(tree)
    return total


def probe_docs(ruff: str) -> dict[str, float | None]:
    """Measure docstring presence over exactly the files ruff lints.

    The numerator comes from ruff itself (``--select D1``), so the tile agrees
    with the gate by construction. The denominator is counted from the same
    file set's ASTs, because ruff reports only what is *missing* and a
    percentage needs to know what was possible.

    Args:
        ruff: Absolute path to the pinned ruff.

    Returns:
        ``{"docs_coverage_pct": <float>}``, or None when either half of the
        ratio could not be established.
    """
    files_run = _run([ruff, "check", "--show-files", "."])
    if files_run.returncode != 0:
        return {"docs_coverage_pct": None}
    total = _documentable_symbols(files_run.stdout)

    findings_run = _run(
        [ruff, "check", ".", "--select", "D1", "--output-format", "json"]
    )
    missing = _corroborated_count(
        findings_run.returncode, _json_array_length(findings_run.stdout)
    )

    if total is None or missing is None or total == 0 or missing > total:
        return {"docs_coverage_pct": None}
    return {"docs_coverage_pct": round(100.0 * (total - missing) / total, 2)}


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI the gate scripts dispatch through.

    Every tool path is a required argument rather than a lookup, so this module
    can never resolve a binary the calling gate did not vouch for.

    Returns:
        Configured parser with one subcommand per dashboard metric.
    """
    parser = argparse.ArgumentParser(description="Emit one quality metric as JSON.")
    subparsers = parser.add_subparsers(dest="metric", required=True)

    lint = subparsers.add_parser("lint", help="lint violation count")
    lint.add_argument("--ruff", required=True)
    lint.add_argument("--python", required=True)
    lint.add_argument("--pre-commit", required=True)

    typecheck = subparsers.add_parser("typecheck", help="type error count")
    typecheck.add_argument("--mypy", required=True)

    security = subparsers.add_parser("security", help="bandit issue count")
    security.add_argument("--bandit", required=True)

    coverage = subparsers.add_parser("coverage", help="line and branch coverage")
    coverage.add_argument("--pytest", required=True)

    tests = subparsers.add_parser("tests", help="test counts")
    tests.add_argument("--pytest", required=True)

    complexity = subparsers.add_parser("complexity", help="complexity averages")
    complexity.add_argument("--radon", required=True)

    docs = subparsers.add_parser("docs", help="docstring coverage")
    docs.add_argument("--ruff", required=True)

    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    """Run the probe the CLI selected.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The metric payload for the chosen subcommand.

    Raises:
        ValueError: If the subcommand has no probe, which argparse's
            ``required=True`` choices should already have prevented.
    """
    if args.metric == "lint":
        return probe_lint(args.ruff, args.python, args.pre_commit)
    if args.metric == "typecheck":
        return probe_typecheck(args.mypy)
    if args.metric == "security":
        return probe_security(args.bandit)
    if args.metric == "coverage":
        return probe_coverage(args.pytest)
    if args.metric == "tests":
        return probe_tests(args.pytest)
    if args.metric == "complexity":
        return probe_complexity(args.radon)
    if args.metric == "docs":
        return probe_docs(args.ruff)
    raise ValueError(f"no probe for metric {args.metric!r}")


def main(argv: list[str] | None = None) -> int:
    """Emit one metric payload as a single JSON object on stdout.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Always 0. ``--metrics`` measures rather than gates, so a bad number is
        still a successful measurement.
    """
    args = _build_parser().parse_args(argv)
    sys.stdout.write(json.dumps(_dispatch(args)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
