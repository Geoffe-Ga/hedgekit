"""Failing-first tests for the dashboard's ``--metrics`` script contract (#122).

`.github/workflows/metrics.yml` collects every quality tile by running
``python scripts/collect_metrics.py --metrics-mode script --scripts-dir
scripts/``. In script mode `collect_from_script()` runs ``scripts/<name>.sh
--metrics`` and requires **pure JSON on stdout**; anything else is a
`json.JSONDecodeError`, which degrades the metric to `null`/``"unknown"``.

Before this change *no* hedgekit script implemented ``--metrics`` (they exited
2 on the unknown option) and ``scripts/metrics-docs.sh`` did not exist, so all
ten quality tiles rendered N/A while the workflow itself stayed green. A
dashboard that says N/A everywhere is indistinguishable, to a glancing
operator, from one that says all-healthy -- which is why the silent
degradation is the bug rather than a cosmetic gap.

Two properties are pinned here, and they pull in opposite directions:

1. **Measurement, not a gate.** ``--metrics`` must exit 0 and report the
   number even when the number is bad, so a red gate still produces a tile.
2. **Fail closed.** A measurement that could not be *proved* must degrade to
   `null` (a gray "unknown" tile), never to a healthy-looking zero. Every
   probe therefore cross-checks the tool's exit status against the count it
   parsed and refuses the pair when they contradict each other.

Both the shell wiring and the parsing are exercised for real: the shell tests
copy the genuine scripts into a throwaway checkout whose `.venv/bin` holds
stub tools (the `tests/toolchain/test_gate_tool_resolution.py` precedent), so
no test here runs ruff, mypy or the 5000-test suite.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest

if TYPE_CHECKING:
    import types

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR: Final = _REPO_ROOT / "scripts"
_PROBE_PATH: Final = _SCRIPTS_DIR / "metrics_probe.py"

#: The exact keys `scripts/collect_metrics.py` reads out of each script's
#: ``--metrics`` payload (`collect_lint_metrics` ... `collect_test_metrics`).
#: A key missing here is a `None` in `metrics.json` and an N/A tile.
_CONTRACT: Final[dict[str, tuple[str, ...]]] = {
    "lint.sh": ("violations",),
    "typecheck.sh": ("errors",),
    "security.sh": ("bandit_issues",),
    "coverage.sh": ("coverage_pct", "branch_coverage_pct"),
    "test.sh": ("tests_total", "tests_passed", "tests_failed", "tests_skipped"),
    "complexity.sh": ("cyclomatic_avg", "maintainability_avg"),
    "metrics-docs.sh": ("docs_coverage_pct",),
}

#: Tools each script resolves from the pinned toolchain, and therefore needs
#: present in a fake checkout's `.venv/bin` before it will run at all.
_SCRIPT_TOOLS: Final[dict[str, tuple[str, ...]]] = {
    "lint.sh": ("ruff", "pre-commit"),
    "typecheck.sh": ("mypy",),
    "security.sh": ("bandit", "pre-commit"),
    "coverage.sh": ("pytest",),
    "test.sh": ("pytest",),
    "complexity.sh": ("radon", "xenon"),
    "metrics-docs.sh": ("ruff",),
}

_COVERAGE_JSON: Final = json.dumps(
    {
        "totals": {
            "percent_covered": 94.80627071119042,
            "percent_branches_covered": 91.6052416052416,
        }
    }
)

_JUNIT_XML: Final = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<testsuites name="pytest tests">'
    '<testsuite name="pytest" errors="2" failures="3" skipped="4" tests="100"'
    ' time="70.7"></testsuite>'
    "</testsuites>"
)


def _load_probe() -> types.ModuleType:
    """Load ``scripts/metrics_probe.py`` as a module by file path.

    `scripts/` is not an importable package, so this mirrors
    `tests/scripts/test_run_canaries.py`'s precedent for exercising operator
    scripts directly.

    Returns:
        The loaded ``metrics_probe`` module.

    Raises:
        AssertionError: If the module cannot be loaded from disk.
    """
    spec = importlib.util.spec_from_file_location("metrics_probe", _PROBE_PATH)
    assert spec is not None and spec.loader is not None, (
        f"{_PROBE_PATH} is not importable -- issue #122 RED"
    )
    module = importlib.util.module_from_spec(spec)
    # `dataclasses` resolves a class's module through `sys.modules` while it
    # builds the generated methods, so a file-path import must be registered
    # before it is executed or `@dataclass` raises at import time.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="probe")
def probe_fixture() -> types.ModuleType:
    """Provide the loaded ``metrics_probe`` module.

    Returns:
        The loaded ``metrics_probe`` module.
    """
    return _load_probe()


def _write_stub(path: Path, body: str) -> Path:
    """Write an executable Python stub standing in for a measurement tool.

    Args:
        path: Where to write the stub; parent directories must exist.
        body: Python source executed when the stub runs.

    Returns:
        The stub's path.
    """
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _emitter(tmp_path: Path, name: str, stdout: str, returncode: int) -> str:
    """Create a stub tool that prints fixed stdout and exits with a fixed code.

    Args:
        tmp_path: Test-scoped temporary directory.
        name: File name for the stub.
        stdout: Text the stub writes to its stdout.
        returncode: Exit status the stub reports.

    Returns:
        Absolute path to the stub, as a string.
    """
    body = f"import sys\nsys.stdout.write({stdout!r})\nsys.exit({returncode})"
    return str(_write_stub(tmp_path / name, body))


def _bandit_stub(tmp_path: Path, report: str | None, returncode: int) -> str:
    """Create a stub `bandit` that writes its JSON report to the ``-o`` path.

    The real bandit prints a ``Working...`` progress bar to stdout before the
    report, so the probe reads the file rather than the stream. The stub does
    the same -- including the progress bar -- so a probe that went back to
    parsing stdout would fail here.

    Args:
        tmp_path: Test-scoped temporary directory.
        report: Text to write as the report, or None to write no file.
        returncode: Exit status the stub reports.

    Returns:
        Absolute path to the stub, as a string.
    """
    body = (
        "import sys, pathlib\n"
        "sys.stdout.write('Working... 100%\\n')\n"
        f"report = {report!r}\n"
        "if report is not None and '-o' in sys.argv:\n"
        "    pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_text(report)\n"
        f"sys.exit({returncode})"
    )
    return str(_write_stub(tmp_path / "bandit", body))


def _pytest_stub(
    tmp_path: Path,
    *,
    coverage_json: str | None = _COVERAGE_JSON,
    junit_xml: str | None = _JUNIT_XML,
    returncode: int = 0,
) -> str:
    """Create a stub `pytest` that writes the artifacts named in its argv.

    The real probe drives pytest with ``--cov-report=json:<path>`` and
    ``--junitxml=<path>``; the stub honours both so the artifact plumbing is
    exercised rather than mocked.

    Args:
        tmp_path: Test-scoped temporary directory.
        coverage_json: Text to write as the coverage report, or None to write
            no coverage report at all (the crashed-run case).
        junit_xml: Text to write as the JUnit report, or None to write none.
        returncode: Exit status the stub reports.

    Returns:
        Absolute path to the stub, as a string.
    """
    body = (
        "import sys, pathlib\n"
        f"cov = {coverage_json!r}\n"
        f"junit = {junit_xml!r}\n"
        "for arg in sys.argv[1:]:\n"
        "    if arg.startswith('--cov-report=json:') and cov is not None:\n"
        "        pathlib.Path(arg.split(':', 1)[1]).write_text(cov)\n"
        "    elif arg.startswith('--junitxml=') and junit is not None:\n"
        "        pathlib.Path(arg.split('=', 1)[1]).write_text(junit)\n"
        f"sys.exit({returncode})"
    )
    return str(_write_stub(tmp_path / "pytest", body))


# --------------------------------------------------------------------------
# 1. The premise: every script named by the collector implements --metrics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("script_name", sorted(_CONTRACT))
def test_every_collector_script_exists_and_is_executable(script_name: str) -> None:
    """Each script the collector runs in script mode exists and can be run.

    `collect_from_script()` returns None for a missing path, which is exactly
    how `metrics-docs.sh` silently produced an N/A Documentation Coverage tile.
    """
    script = _SCRIPTS_DIR / script_name
    assert script.is_file(), f"scripts/{script_name} does not exist -- issue #122 RED"
    assert os.access(script, os.X_OK), (
        f"scripts/{script_name} is not executable, so the collector's "
        "subprocess call cannot run it"
    )


@pytest.mark.parametrize("script_name", sorted(_CONTRACT))
def test_metrics_flag_is_a_documented_option(script_name: str) -> None:
    """``--metrics`` appears in each script's own ``--help`` text.

    The flag is a published contract with `collect_metrics.py`, so a reader of
    the script must be able to discover it without reading the collector.
    """
    result = subprocess.run(
        [str(_SCRIPTS_DIR / script_name), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr!r}"
    assert "--metrics" in result.stdout, (
        f"scripts/{script_name} --help does not document --metrics: {result.stdout!r}"
    )


def test_unknown_options_are_still_rejected() -> None:
    """Adding ``--metrics`` must not turn the parsers into anything-goes.

    Each gate script exits 2 on an unrecognised option. A parser that started
    ignoring unknown flags would silently accept a typo like ``--metric`` and
    run the *gate* inside the metrics workflow.
    """
    result = subprocess.run(
        [str(_SCRIPTS_DIR / "lint.sh"), "--metric"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 2, (
        f"lint.sh --metric did not exit 2: rc={result.returncode} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


# --------------------------------------------------------------------------
# 2. Probe unit tests: the numbers, and the fail-closed refusals
# --------------------------------------------------------------------------


def test_lint_probe_sums_ruff_and_float_lint_findings(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Lint violations are ruff's findings plus float-lint's, not either alone.

    The two counts are deliberately distinct (2 and 3): a probe that returned
    only one of them, or that multiplied instead of summing, would produce a
    different number than 5.
    """
    ruff = _emitter(
        tmp_path,
        "ruff",
        json.dumps([{"code": "E501"}, {"code": "F401"}]),
        1,
    )
    float_findings = "a.py:1:1 FLOAT-001 x\nb.py:2:1 FLOAT-002 y\nc.py:3:1 FLOAT-3 z\n"
    python = _emitter(tmp_path, "python", float_findings, 1)
    pre_commit = _emitter(tmp_path, "pre-commit", "", 0)

    assert probe.probe_lint(ruff, python, pre_commit) == {"violations": 5}


def test_lint_probe_reports_zero_when_every_sub_check_is_clean(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A genuinely clean tree reports 0, not None.

    Fail-closed must not collapse into fail-always: the healthy case has to
    survive, or the tile would be permanently gray.
    """
    ruff = _emitter(tmp_path, "ruff", "[]", 0)
    python = _emitter(tmp_path, "python", "", 0)
    pre_commit = _emitter(tmp_path, "pre-commit", "", 0)

    assert probe.probe_lint(ruff, python, pre_commit) == {"violations": 0}


def test_lint_probe_refuses_when_shellcheck_fails(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A failing shellcheck hook makes the count unprovable, so it is None.

    The pinned pre-commit hook emits no machine-readable finding count, so the
    only honest reading of a nonzero exit is "there are findings, and I cannot
    say how many". Reporting ruff's 0 alone would paint a green tile over a red
    shell lint.
    """
    ruff = _emitter(tmp_path, "ruff", "[]", 0)
    python = _emitter(tmp_path, "python", "", 0)
    pre_commit = _emitter(tmp_path, "pre-commit", "In x.sh line 3:\n", 1)

    assert probe.probe_lint(ruff, python, pre_commit) == {"violations": None}


def test_lint_probe_refuses_unparseable_ruff_output(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Non-JSON on ruff's stdout degrades to None rather than to zero."""
    ruff = _emitter(tmp_path, "ruff", "ruff failed: invalid config\n", 2)
    python = _emitter(tmp_path, "python", "", 0)
    pre_commit = _emitter(tmp_path, "pre-commit", "", 0)

    assert probe.probe_lint(ruff, python, pre_commit) == {"violations": None}


def test_lint_probe_refuses_a_findings_exit_with_an_empty_report(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Exit 1 with zero parsed findings is a contradiction, so it is refused.

    This is the near-miss the whole cross-check exists for: ruff says "I found
    something" while the parse says "nothing". Trusting the parse would report
    a healthy 0 for a failing linter.
    """
    ruff = _emitter(tmp_path, "ruff", "[]", 1)
    python = _emitter(tmp_path, "python", "", 0)
    pre_commit = _emitter(tmp_path, "pre-commit", "", 0)

    assert probe.probe_lint(ruff, python, pre_commit) == {"violations": None}


def test_lint_probe_refuses_a_clean_exit_with_a_non_empty_report(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Exit 0 alongside parsed findings is the same contradiction, mirrored."""
    ruff = _emitter(tmp_path, "ruff", json.dumps([{"code": "E501"}]), 0)
    python = _emitter(tmp_path, "python", "", 0)
    pre_commit = _emitter(tmp_path, "pre-commit", "", 0)

    assert probe.probe_lint(ruff, python, pre_commit) == {"violations": None}


def test_lint_probe_refuses_a_tool_crash_that_still_printed_findings(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A ruff exit status outside 0/1 is refused even when its report parses.

    ruff exits 2 when *ruff* failed -- a bad config, an unreadable file -- and
    whatever it managed to print before giving up does not describe the
    repository. The parse succeeding is not evidence that the run did.
    """
    ruff = _emitter(tmp_path, "ruff", json.dumps([{"code": "E501"}]), 2)
    python = _emitter(tmp_path, "python", "", 0)
    pre_commit = _emitter(tmp_path, "pre-commit", "", 0)

    assert probe.probe_lint(ruff, python, pre_commit) == {"violations": None}


def test_lint_probe_refuses_a_tool_crash_that_printed_an_empty_report(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A crash with an empty report is refused, not reported as zero findings.

    The mirrored crash test above is passed by the *contradiction* check alone
    (exit 2 disagrees with a non-empty parse), so it leaves the exit-status
    check unproved -- deleting that check kept the whole suite green until this
    case existed. Here the two agree: ruff died with nothing printed, and
    "nothing printed" reads as a clean repository unless the status is
    consulted. Zero findings from a linter that never finished is the exact
    fail-open the dashboard must not paint green.
    """
    ruff = _emitter(tmp_path, "ruff", "[]", 2)
    python = _emitter(tmp_path, "python", "", 0)
    pre_commit = _emitter(tmp_path, "pre-commit", "", 0)

    assert probe.probe_lint(ruff, python, pre_commit) == {"violations": None}


def test_security_probe_refuses_a_crashed_scan_that_reported_no_results(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Bandit dying with an empty ``results`` array is unknown, never clean.

    The same fail-open as the lint case, on the tile where it costs most: an
    empty findings list from a scan that crashed is indistinguishable, in the
    payload alone, from a repository with no security issues.
    """
    bandit = _bandit_stub(tmp_path, json.dumps({"results": []}), 2)

    assert probe.probe_security(bandit) == {"bandit_issues": None}


def test_lint_probe_refuses_a_float_lint_failure_with_no_lines(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """The cross-check covers float-lint too, not only ruff."""
    ruff = _emitter(tmp_path, "ruff", "[]", 0)
    python = _emitter(tmp_path, "python", "", 1)
    pre_commit = _emitter(tmp_path, "pre-commit", "", 0)

    assert probe.probe_lint(ruff, python, pre_commit) == {"violations": None}


def test_tool_stdout_is_echoed_head_and_tail_not_in_full(
    probe: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tool's machine payload is abridged before it reaches stderr.

    radon's JSON report is around half a megabyte; echoing it whole buries the
    terminal of anyone running a probe by hand. Both ends survive, because the
    head carries a parse error's first line and the tail carries pytest's
    summary.
    """
    payload = "HEAD" + ("x" * 5000) + "TAIL"
    ruff = _emitter(tmp_path, "ruff", payload, 2)

    probe.probe_docs(ruff)

    echoed = capsys.readouterr().err
    assert "HEAD" in echoed
    assert "TAIL" in echoed
    assert len(echoed) < len(payload)


def test_typecheck_probe_reads_mypys_error_count(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """The error count comes from mypy's summary line, not from its exit code."""
    mypy = _emitter(
        tmp_path,
        "mypy",
        "windbreak/a.py:1: error: bad\nFound 7 errors in 3 files (checked 9 files)\n",
        1,
    )

    assert probe.probe_typecheck(mypy) == {"errors": 7}


def test_typecheck_probe_reports_zero_on_success(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A clean mypy run reports 0 errors."""
    mypy = _emitter(tmp_path, "mypy", "Success: no issues found in 9 source files\n", 0)

    assert probe.probe_typecheck(mypy) == {"errors": 0}


def test_typecheck_probe_refuses_a_crash_with_no_summary(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """mypy crashing without a summary line yields None, never 0."""
    mypy = _emitter(tmp_path, "mypy", "mypy: error: Cannot find module\n", 2)

    assert probe.probe_typecheck(mypy) == {"errors": None}


def test_typecheck_probe_refuses_a_failure_summarised_as_zero(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """ "Found 0 errors" alongside a failing exit is refused as contradictory."""
    mypy = _emitter(tmp_path, "mypy", "Found 0 errors in 0 files\n", 1)

    assert probe.probe_typecheck(mypy) == {"errors": None}


def test_security_probe_counts_bandit_results(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """The issue count is the length of bandit's ``results`` array."""
    payload = json.dumps(
        {
            "results": [
                {"issue_severity": "HIGH"},
                {"issue_severity": "LOW"},
                {"issue_severity": "MEDIUM"},
            ],
            "metrics": {"_totals": {"loc": 10}},
        }
    )
    bandit = _bandit_stub(tmp_path, payload, 1)

    assert probe.probe_security(bandit) == {"bandit_issues": 3}


def test_security_probe_reports_zero_for_a_clean_scan(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A clean bandit scan reports 0, keeping the tile green rather than gray."""
    bandit = _bandit_stub(tmp_path, json.dumps({"results": []}), 0)

    assert probe.probe_security(bandit) == {"bandit_issues": 0}


def test_security_probe_refuses_output_without_a_results_array(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A payload with no ``results`` list is unusable, so the metric is None.

    A `.get("results", [])` default would have reported a reassuring 0 here --
    a security tile reading "0 issues" from a scan that never produced results
    is the worst possible false green.
    """
    bandit = _bandit_stub(tmp_path, json.dumps({"errors": ["boom"]}), 2)

    assert probe.probe_security(bandit) == {"bandit_issues": None}


def test_security_probe_refuses_a_clean_exit_with_no_results_array(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A report lacking ``results`` is None even when bandit exited cleanly.

    This is the shape a `.get("results", [])` default would launder into a
    green "0 security issues" tile: bandit says it finished, the report says
    nothing about findings, and the default invents the reassuring answer.
    Absent evidence must read as unknown, not as clean.
    """
    bandit = _bandit_stub(tmp_path, json.dumps({"metrics": {"_totals": {}}}), 0)

    assert probe.probe_security(bandit) == {"bandit_issues": None}


def test_security_probe_refuses_a_progress_bar_instead_of_a_report(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A scan that wrote no report file yields None, whatever stdout said.

    bandit's stdout carries a ``Working...`` progress bar, so a probe that read
    the stream instead of the ``-o`` file would see non-JSON and -- if it were
    less careful -- could report an empty findings list as zero issues.
    """
    bandit = _bandit_stub(tmp_path, None, 0)

    assert probe.probe_security(bandit) == {"bandit_issues": None}


def test_coverage_probe_reads_line_and_branch_percentages(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Both percentages come from `coverage.json`, and they are not swapped.

    The fixture's two values are deliberately far apart (94.81 vs 91.61) so a
    probe that assigned branch coverage to the line-coverage key would fail.
    """
    pytest_bin = _pytest_stub(tmp_path)

    payload = probe.probe_coverage(pytest_bin, artifact_dir=tmp_path / "artifacts")

    assert payload == {"coverage_pct": 94.81, "branch_coverage_pct": 91.61}


def test_coverage_probe_refuses_when_pytest_produced_no_report(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A pytest run that wrote no coverage report yields None, not 0.0.

    Zero coverage and unmeasured coverage are different facts; only one of them
    is a reason to stop the release.
    """
    pytest_bin = _pytest_stub(tmp_path, coverage_json=None, returncode=4)

    payload = probe.probe_coverage(pytest_bin, artifact_dir=tmp_path / "artifacts")

    assert payload == {"coverage_pct": None, "branch_coverage_pct": None}


def test_coverage_probe_refuses_a_report_missing_the_branch_key(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A report without branch data leaves the branch tile unknown."""
    pytest_bin = _pytest_stub(
        tmp_path, coverage_json=json.dumps({"totals": {"percent_covered": 94.5}})
    )

    payload = probe.probe_coverage(pytest_bin, artifact_dir=tmp_path / "artifacts")

    assert payload == {"coverage_pct": 94.5, "branch_coverage_pct": None}


def test_tests_probe_derives_passed_from_the_junit_totals(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Passed = tests - failures - errors - skipped, with distinct fixtures.

    The fixture uses 100/3/2/4 rather than repeated values so a probe that read
    ``errors`` where it meant ``failures`` cannot produce the same answer.
    """
    pytest_bin = _pytest_stub(tmp_path)

    payload = probe.probe_tests(pytest_bin, artifact_dir=tmp_path / "artifacts")

    assert payload == {
        "tests_total": 100,
        "tests_passed": 91,
        "tests_failed": 3,
        "tests_skipped": 4,
    }


def test_tests_probe_refuses_when_no_junit_report_was_written(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A collection error that wrote no report leaves every count None.

    `collect_test_metrics` renders a total of 0 as ``"unknown"``, but the raw
    counts must still be null so nothing downstream can read "0 failures".
    """
    pytest_bin = _pytest_stub(tmp_path, junit_xml=None, returncode=2)

    payload = probe.probe_tests(pytest_bin, artifact_dir=tmp_path / "artifacts")

    assert payload == {
        "tests_total": None,
        "tests_passed": None,
        "tests_failed": None,
        "tests_skipped": None,
    }


def test_tests_probe_refuses_a_malformed_junit_report(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Unparseable XML degrades to None instead of raising or reporting zero."""
    pytest_bin = _pytest_stub(tmp_path, junit_xml="<testsuites><oops")

    payload = probe.probe_tests(pytest_bin, artifact_dir=tmp_path / "artifacts")

    assert payload["tests_total"] is None


def test_pytest_artifacts_are_produced_once_and_reused(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """The coverage and test probes share one suite run, not two.

    The stub appends a line per invocation, so the counter proves the second
    probe reused the first probe's artifacts rather than re-running a
    seventy-second suite.
    """
    log = tmp_path / "runs.log"
    body = (
        "import sys, pathlib\n"
        f"pathlib.Path({str(log)!r}).open('a').write('run\\n')\n"
        f"cov = {_COVERAGE_JSON!r}\n"
        f"junit = {_JUNIT_XML!r}\n"
        "for arg in sys.argv[1:]:\n"
        "    if arg.startswith('--cov-report=json:'):\n"
        "        pathlib.Path(arg.split(':', 1)[1]).write_text(cov)\n"
        "    elif arg.startswith('--junitxml='):\n"
        "        pathlib.Path(arg.split('=', 1)[1]).write_text(junit)\n"
        "sys.exit(0)"
    )
    pytest_bin = str(_write_stub(tmp_path / "pytest", body))
    artifacts = tmp_path / "artifacts"

    probe.probe_coverage(pytest_bin, artifact_dir=artifacts)
    probe.probe_tests(pytest_bin, artifact_dir=artifacts)

    assert log.read_text(encoding="utf-8").count("run") == 1


def test_pytest_artifacts_are_regenerated_when_a_source_file_changes(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A source edit invalidates the shared artifacts.

    Reuse is only safe while the tree it measured is unchanged; a cache that
    outlived an edit would publish yesterday's coverage as today's.
    """
    log = tmp_path / "runs.log"
    body = (
        "import sys, pathlib\n"
        f"pathlib.Path({str(log)!r}).open('a').write('run\\n')\n"
        f"cov = {_COVERAGE_JSON!r}\n"
        f"junit = {_JUNIT_XML!r}\n"
        "for arg in sys.argv[1:]:\n"
        "    if arg.startswith('--cov-report=json:'):\n"
        "        pathlib.Path(arg.split(':', 1)[1]).write_text(cov)\n"
        "    elif arg.startswith('--junitxml='):\n"
        "        pathlib.Path(arg.split('=', 1)[1]).write_text(junit)\n"
        "sys.exit(0)"
    )
    pytest_bin = str(_write_stub(tmp_path / "pytest", body))
    artifacts = tmp_path / "artifacts"
    source_root = tmp_path / "src"
    (source_root / "windbreak").mkdir(parents=True)
    edited = source_root / "windbreak" / "thing.py"
    edited.write_text("x = 1\n", encoding="utf-8")

    probe.probe_coverage(pytest_bin, artifact_dir=artifacts, source_root=source_root)
    os.utime(edited, (_future(), _future()))
    probe.probe_tests(pytest_bin, artifact_dir=artifacts, source_root=source_root)

    assert log.read_text(encoding="utf-8").count("run") == 2


def test_artifacts_exactly_as_old_as_the_source_are_not_reused(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A tie between artifact and source mtime resolves toward re-running.

    Reuse is decided by comparing modification times, and mtime granularity is
    coarse -- one second on many filesystems. An edit landing in the same
    second as the run that measured it is therefore indistinguishable from an
    edit landing just before it, so a tie must count as stale. Treating it as
    current publishes a number measured from code that no longer exists, which
    is the reuse optimisation quietly turning into a lie.
    """
    log = tmp_path / "runs.log"
    pytest_bin = str(
        _write_stub(
            tmp_path / "pytest",
            "import sys, pathlib\n"
            f"pathlib.Path({str(log)!r}).open('a').write('run' + chr(10))\n"
            f"cov = {_COVERAGE_JSON!r}\n"
            "for arg in sys.argv[1:]:\n"
            "    if arg.startswith('--cov-report=json:'):\n"
            "        pathlib.Path(arg.split(':', 1)[1]).write_text(cov)\n"
            f"    elif arg.startswith('--junitxml='):\n"
            f"        pathlib.Path(arg.split('=', 1)[1]).write_text({_JUNIT_XML!r})\n"
            "sys.exit(0)",
        )
    )
    artifacts = tmp_path / "artifacts"
    source_root = tmp_path / "src"
    (source_root / "windbreak").mkdir(parents=True)
    edited = source_root / "windbreak" / "thing.py"
    edited.write_text("x = 1\n", encoding="utf-8")

    probe.probe_coverage(pytest_bin, artifact_dir=artifacts, source_root=source_root)
    tie = _future()
    for path in (edited, artifacts / "coverage.json", artifacts / "junit.xml"):
        os.utime(path, (tie, tie))
    probe.probe_tests(pytest_bin, artifact_dir=artifacts, source_root=source_root)

    assert log.read_text(encoding="utf-8").count("run") == 2, (
        "artifacts exactly as old as the source were reused, so an edit made "
        "within the same mtime tick would be measured as if it had not happened"
    )


def test_the_shared_suite_run_is_parallelised(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """The one suite run the two probes share is distributed across workers.

    `collect_from_script()` kills a script at 300 s and reads the corpse as an
    N/A tile -- which is the failure this whole issue is about, arriving by a
    different road. Serially this suite needs ~170 s on a developer machine and
    CI's quality job needs ~6 minutes for comparable work, so the four tiles
    fed by pytest would be riding a margin thinner than the measurement error.
    The stub records the argv it was actually invoked with, so this fails on
    the real command line rather than on a grep of the source.
    """
    argv_log = tmp_path / "argv.txt"
    pytest_bin = str(
        _write_stub(
            tmp_path / "pytest",
            "import sys, pathlib\n"
            f"pathlib.Path({str(argv_log)!r}).write_text(chr(10).join(sys.argv[1:]))\n"
            f"cov = {_COVERAGE_JSON!r}\n"
            "for arg in sys.argv[1:]:\n"
            "    if arg.startswith('--cov-report=json:'):\n"
            "        pathlib.Path(arg.split(':', 1)[1]).write_text(cov)\n"
            "sys.exit(0)",
        )
    )

    probe.probe_coverage(pytest_bin, artifact_dir=tmp_path / "artifacts")

    argv = argv_log.read_text(encoding="utf-8").splitlines()
    assert "-n" in argv, f"the suite run is not parallelised: {argv!r}"
    assert argv[argv.index("-n") + 1] == "auto", (
        f"-n was passed without a worker count: {argv!r}"
    )


def _future() -> float:
    """Return a timestamp comfortably ahead of any file just written.

    Returns:
        An epoch-seconds value one hour in the future.
    """
    return Path(__file__).stat().st_mtime + 3600.0


def test_complexity_probe_averages_every_analysed_block(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Cyclomatic average is the mean over blocks; maintainability over files.

    Three distinct complexities (1, 4, 7) across two files with distinct
    maintainability indices (60.0, 40.0) mean a probe that averaged per file,
    or that read the wrong key, lands on a different number than 4.0 / 50.0.
    """
    cc_payload = json.dumps(
        {
            "windbreak/a.py": [
                {"type": "function", "name": "f", "complexity": 1},
                {"type": "function", "name": "g", "complexity": 4},
            ],
            "windbreak/b.py": [{"type": "function", "name": "h", "complexity": 7}],
        }
    )
    mi_payload = json.dumps(
        {
            "windbreak/a.py": {"mi": 60.0, "rank": "A"},
            "windbreak/b.py": {"mi": 40.0, "rank": "A"},
        }
    )
    radon = _write_stub(
        tmp_path / "radon",
        "import sys\n"
        f"cc = {cc_payload!r}\n"
        f"mi = {mi_payload!r}\n"
        "sys.stdout.write(cc if 'cc' in sys.argv else mi)\n"
        "sys.exit(0)",
    )

    payload = probe.probe_complexity(str(radon))

    assert payload == {"cyclomatic_avg": 4.0, "maintainability_avg": 50.0}


def test_complexity_probe_skips_files_radon_could_not_parse(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A per-file ``error`` entry is skipped, not counted as complexity zero.

    radon reports unparseable files as ``{"error": "..."}``; treating that
    object as a block list would either crash or drag the average down.
    """
    cc_payload = json.dumps(
        {
            "windbreak/a.py": [{"type": "function", "name": "f", "complexity": 6}],
            "windbreak/broken.py": {"error": "invalid syntax"},
        }
    )
    mi_payload = json.dumps(
        {
            "windbreak/a.py": {"mi": 55.0, "rank": "A"},
            "windbreak/broken.py": {"error": "invalid syntax"},
        }
    )
    radon = _write_stub(
        tmp_path / "radon",
        "import sys\n"
        f"cc = {cc_payload!r}\n"
        f"mi = {mi_payload!r}\n"
        "sys.stdout.write(cc if 'cc' in sys.argv else mi)\n"
        "sys.exit(0)",
    )

    payload = probe.probe_complexity(str(radon))

    assert payload == {"cyclomatic_avg": 6.0, "maintainability_avg": 55.0}


def test_complexity_probe_refuses_unparseable_radon_output(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """radon failing outright leaves both complexity tiles unknown.

    `_compute_status` treats a complexity of 0 as *passing* the ``<= 10``
    threshold, so a zero here would render the healthiest possible tile from
    no measurement at all.
    """
    radon = _emitter(tmp_path, "radon", "radon: command failed\n", 1)

    payload = probe.probe_complexity(radon)

    assert payload == {"cyclomatic_avg": None, "maintainability_avg": None}


def _radon_stub(tmp_path: Path, cc_payload: str, mi_payload: str) -> str:
    """Create a stub `radon` answering ``cc -j`` and ``mi -j`` differently.

    Args:
        tmp_path: Test-scoped temporary directory.
        cc_payload: Text to emit for the cyclomatic-complexity report.
        mi_payload: Text to emit for the maintainability-index report.

    Returns:
        Absolute path to the stub, as a string.
    """
    return str(
        _write_stub(
            tmp_path / "radon",
            "import sys\n"
            f"cc = {cc_payload!r}\n"
            f"mi = {mi_payload!r}\n"
            "sys.stdout.write(cc if 'cc' in sys.argv else mi)\n"
            "sys.exit(0)",
        )
    )


def test_complexity_probe_refuses_a_report_with_no_analysed_blocks(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """radon succeeding over nothing is no measurement, so both tiles are None.

    An empty report is the shape radon produces when its target matched no
    files. Averaging nothing to 0.0 would render the *healthiest possible*
    complexity tile -- 0 sails under the ``<= 10`` threshold -- from a run
    that analysed not one line of this repository.
    """
    radon = _radon_stub(tmp_path, "{}", "{}")

    payload = probe.probe_complexity(radon)

    assert payload == {"cyclomatic_avg": None, "maintainability_avg": None}


def test_complexity_probe_survives_an_entry_that_is_not_a_block_list(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A non-iterable per-file entry degrades to None instead of crashing.

    ``--metrics`` must always exit 0 with a JSON object; an unhandled
    `TypeError` here would make the script exit nonzero with a traceback on
    stderr and nothing on stdout, which the collector reads as an N/A tile --
    the exact failure this issue removes.
    """
    radon = _radon_stub(tmp_path, '{"windbreak/a.py": null}', '{"windbreak/a.py": 7}')

    payload = probe.probe_complexity(radon)

    assert payload == {"cyclomatic_avg": None, "maintainability_avg": None}


def test_docs_probe_reports_full_coverage_when_ruff_finds_nothing(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Zero D1 findings over a documentable tree is 100 percent."""
    package = tmp_path / "pkg"
    package.mkdir()
    module = package / "m.py"
    module.write_text('"""Doc."""\n\n\ndef f():\n    """Doc."""\n', encoding="utf-8")
    ruff = _write_stub(
        tmp_path / "ruff",
        "import sys\n"
        f"sys.stdout.write({str(module)!r} + chr(10) if '--show-files' in sys.argv "
        "else '[]')\n"
        "sys.exit(0)",
    )

    assert probe.probe_docs(str(ruff)) == {"docs_coverage_pct": 100.0}


def test_docs_probe_divides_missing_docstrings_by_documentable_symbols(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """One missing docstring out of four symbols is 75 percent.

    The module counts as a documentable symbol alongside its public class,
    method and function, so the denominator is 4 rather than 3.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    module = package / "m.py"
    module.write_text(
        "class Thing:\n"
        "    def method(self):\n"
        "        return 1\n"
        "\n"
        "\n"
        "def func():\n"
        "    return 2\n",
        encoding="utf-8",
    )
    findings = json.dumps([{"code": "D100"}])
    ruff = _write_stub(
        tmp_path / "ruff",
        "import sys\n"
        f"files = {str(module)!r} + chr(10)\n"
        f"findings = {findings!r}\n"
        "sys.stdout.write(files if '--show-files' in sys.argv else findings)\n"
        "sys.exit(0 if '--show-files' in sys.argv else 1)",
    )

    assert probe.probe_docs(str(ruff)) == {"docs_coverage_pct": 75.0}


def test_docs_probe_ignores_private_symbols_ruff_does_not_check(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """Private helpers are outside ruff's D1 set, so they leave the denominator.

    Counting them would inflate the percentage: extra undocumented-but-unchecked
    symbols in the denominator can only push the ratio toward 100.

    The fixture reports one missing docstring on purpose. With none missing the
    ratio is 100% however the denominator is computed, so the numbers would
    coincide and this test would pass just as happily over a probe that counted
    every private helper -- which it did, until a mutation run noticed.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    module = package / "m.py"
    module.write_text(
        '"""Doc."""\n\n\ndef _helper():\n    return 1\n\n\n'
        "def public():\n    return 2\n",
        encoding="utf-8",
    )
    ruff = _write_stub(
        tmp_path / "ruff",
        "import sys\n"
        f"sys.stdout.write({str(module)!r} + chr(10) if '--show-files' in sys.argv "
        'else \'[{"code": "D103"}]\')\n'
        "sys.exit(0 if '--show-files' in sys.argv else 1)",
    )

    # Documentable: the module and `public` -- two, one of them missing a
    # docstring, so 50%. Counting `_helper` would make it two of three, 66.67%.
    assert probe.probe_docs(str(ruff)) == {"docs_coverage_pct": 50.0}


def test_docs_probe_refuses_a_file_it_cannot_parse(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """An unparseable file voids the denominator rather than shrinking it.

    ruff lists a file it can lint; if this probe's own AST pass cannot read it,
    the two halves of the ratio no longer describe the same file set. Skipping
    the file would silently shrink the denominator and push the reported
    percentage up -- a smaller measured world always looks better documented.

    The listing carries a *healthy* file beside the broken one, because that is
    the only shape where the two behaviours differ: with the broken file alone,
    skipping it leaves an empty denominator that `probe_docs` refuses anyway,
    and the test would agree with either implementation.
    """
    healthy = tmp_path / "healthy.py"
    healthy.write_text('"""Doc."""\n', encoding="utf-8")
    broken = tmp_path / "broken.py"
    broken.write_text("def (\n", encoding="utf-8")
    listing = f"{healthy}\n{broken}\n"
    ruff = _write_stub(
        tmp_path / "ruff",
        "import sys\n"
        f"sys.stdout.write({listing!r} if '--show-files' in sys.argv else '[]')\n"
        "sys.exit(0)",
    )

    # Skipping `broken.py` would leave `healthy.py` alone in the denominator
    # with nothing missing, and report a confident 100%.
    assert probe.probe_docs(str(ruff)) == {"docs_coverage_pct": None}


def test_docs_probe_refuses_an_empty_denominator_instead_of_dividing_by_zero(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A listing with no Python file is no measurement, and must not raise.

    This is where the two rules meet. *Fail closed* says nothing measured is
    `null`, never a confident 100%. *Measurement, not a gate* says the script
    still exits 0 with one JSON object -- and an unguarded ``0 / 0`` would
    instead exit nonzero with a traceback, which the collector reads as the
    same silent N/A by a noisier route.
    """
    listing = f"{tmp_path / 'pyproject.toml'}\n"
    ruff = _write_stub(
        tmp_path / "ruff",
        "import sys\n"
        f"sys.stdout.write({listing!r} if '--show-files' in sys.argv else '[]')\n"
        "sys.exit(0)",
    )

    assert probe.probe_docs(str(ruff)) == {"docs_coverage_pct": None}


def test_coverage_probe_refuses_a_percentage_outside_zero_to_one_hundred(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A percentage that is not a percentage is refused, not published.

    A coverage report claiming 150% did not measure this repository -- it is a
    truncated write, a different tool's schema, or a unit mix-up. Passing the
    number through would put an impossible figure on the dashboard above a
    green "passing" badge, since `_compute_status` only asks whether the value
    clears the threshold.
    """
    pytest_bin = _pytest_stub(
        tmp_path,
        coverage_json=json.dumps(
            {"totals": {"percent_covered": 150.0, "percent_branches_covered": -3.0}}
        ),
    )

    assert probe.probe_coverage(pytest_bin, artifact_dir=tmp_path / "artifacts") == {
        "coverage_pct": None,
        "branch_coverage_pct": None,
    }


def test_docs_probe_refuses_when_ruff_cannot_list_its_files(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """No file list means no denominator, so the tile stays unknown."""
    ruff = _emitter(tmp_path, "ruff", "", 2)

    assert probe.probe_docs(ruff) == {"docs_coverage_pct": None}


def test_docs_probe_refuses_more_findings_than_symbols(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """More missing docstrings than documentable symbols is incoherent.

    It means the finding count and the symbol count describe different file
    sets, which would otherwise yield a negative percentage.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    module = package / "m.py"
    module.write_text('"""Doc."""\n', encoding="utf-8")
    findings = json.dumps([{"code": "D100"}, {"code": "D103"}, {"code": "D101"}])
    ruff = _write_stub(
        tmp_path / "ruff",
        "import sys\n"
        f"files = {str(module)!r} + chr(10)\n"
        f"findings = {findings!r}\n"
        "sys.stdout.write(files if '--show-files' in sys.argv else findings)\n"
        "sys.exit(0 if '--show-files' in sys.argv else 1)",
    )

    assert probe.probe_docs(str(ruff)) == {"docs_coverage_pct": None}


# --------------------------------------------------------------------------
# 3. The shell wiring, exercised for real against a throwaway checkout
# --------------------------------------------------------------------------


def _fake_repo(tmp_path: Path, *script_names: str) -> Path:
    """Build a throwaway checkout able to run gate scripts' metrics modes.

    The gate scripts resolve their tools from the pinned `.venv` of *the
    checkout that owns them* (`scripts/toolchain-env.sh`), so copying the real
    scripts into a temp checkout with a stubbed `.venv/bin` exercises the
    genuine shell path without running ruff, mypy or the test suite. The
    interpreter is a symlink to the running one, because `metrics_probe.py`
    must actually execute.

    Args:
        tmp_path: Test-scoped temporary directory.
        script_names: Gate scripts to copy in, e.g. ``"lint.sh"``.

    Returns:
        Root of the fake checkout.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in (
        "provision-venv.sh",
        "toolchain-env.sh",
        "metrics_probe.py",
        "lint_no_floats.py",
        *script_names,
    ):
        shutil.copy(_SCRIPTS_DIR / name, repo / "scripts" / name)
    (repo / "sample.py").write_text(
        '"""Module docstring."""\n\n\ndef public():\n    """Doc."""\n    return 1\n',
        encoding="utf-8",
    )
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    installed: set[str] = set()
    for script_name in script_names:
        for tool in _SCRIPT_TOOLS[script_name]:
            if tool not in installed:
                _install_tool_stub(venv_bin / tool, tool, repo)
                installed.add(tool)
    return repo


def _install_tool_stub(path: Path, tool: str, repo: Path) -> None:
    """Install a stub for one measurement tool inside a fake `.venv/bin`.

    Each stub emits the smallest output its probe can parse, and reports the
    *failing* exit status its real counterpart would use when it has findings
    -- proving `--metrics` still exits 0 over a red gate.

    Args:
        path: Destination inside the fake `.venv/bin`.
        tool: Tool name being stubbed.
        repo: Root of the fake checkout the stub reports about.
    """
    sample = repo / "sample.py"
    bodies = {
        "ruff": (
            "import sys\n"
            "if '--show-files' in sys.argv:\n"
            f"    sys.stdout.write({str(sample)!r} + chr(10))\n"
            "    sys.exit(0)\n"
            'sys.stdout.write(\'[{"code": "E501"}]\')\n'
            "sys.exit(1)"
        ),
        "pre-commit": "import sys\nsys.exit(0)",
        "mypy": (
            "import sys\n"
            "sys.stdout.write('Found 2 errors in 1 file (checked 9 files)\\n')\n"
            "sys.exit(1)"
        ),
        "bandit": (
            "import sys, pathlib\n"
            "sys.stdout.write('Working... 100%\\n')\n"
            "if '-o' in sys.argv:\n"
            "    pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_text(\n"
            '        \'{"results": [{"a": 1}]}\'\n'
            "    )\n"
            "sys.exit(1)"
        ),
        "radon": (
            "import sys\n"
            'cc = \'{"a.py": [{"complexity": 3}]}\'\n'
            'mi = \'{"a.py": {"mi": 42.0}}\'\n'
            "sys.stdout.write(cc if 'cc' in sys.argv else mi)\n"
            "sys.exit(0)"
        ),
        "xenon": "import sys\nsys.exit(0)",
    }
    if tool == "pytest":
        _write_stub(
            path,
            "import sys, pathlib\n"
            f"cov = {_COVERAGE_JSON!r}\n"
            f"junit = {_JUNIT_XML!r}\n"
            "for arg in sys.argv[1:]:\n"
            "    if arg.startswith('--cov-report=json:'):\n"
            "        pathlib.Path(arg.split(':', 1)[1]).write_text(cov)\n"
            "    elif arg.startswith('--junitxml='):\n"
            "        pathlib.Path(arg.split('=', 1)[1]).write_text(junit)\n"
            "sys.exit(1)",
        )
        return
    _write_stub(path, bodies[tool])


def _run_metrics_mode(repo: Path, script_name: str) -> subprocess.CompletedProcess[str]:
    """Run one gate script's ``--metrics`` mode inside a fake checkout.

    Args:
        repo: Root of the fake checkout.
        script_name: Gate script to run.

    Returns:
        The completed process with stdout/stderr captured as text.
    """
    return subprocess.run(
        [str(repo / "scripts" / script_name), "--metrics"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.parametrize("script_name", sorted(_CONTRACT))
def test_metrics_mode_emits_exactly_one_json_object(
    tmp_path: Path, script_name: str
) -> None:
    """stdout carries one JSON object and nothing else.

    `collect_from_script()` does `json.loads(result.stdout.strip())`, so a
    single stray banner line on stdout is the whole difference between a value
    and an N/A tile. Every stub here reports findings, which is also the proof
    that tool chatter cannot leak onto stdout on the unhappy path.
    """
    repo = _fake_repo(tmp_path, script_name)

    result = _run_metrics_mode(repo, script_name)

    assert "\n" not in result.stdout.strip(), (
        f"scripts/{script_name} --metrics wrote more than one line to stdout: "
        f"{result.stdout!r}"
    )
    payload = json.loads(result.stdout.strip())
    assert isinstance(payload, dict), f"stdout is not a JSON object: {payload!r}"


@pytest.mark.parametrize("script_name", sorted(_CONTRACT))
def test_metrics_mode_exits_zero_even_when_the_gate_would_fail(
    tmp_path: Path, script_name: str
) -> None:
    """``--metrics`` measures; it never gates.

    Every stubbed tool exits nonzero (ruff has a finding, mypy has errors,
    bandit has an issue, pytest has failures). The collector treats *any*
    nonzero-with-unparseable-output as a dead metric, so a script that
    propagated its tool's exit status would still produce N/A tiles.
    """
    repo = _fake_repo(tmp_path, script_name)

    result = _run_metrics_mode(repo, script_name)

    assert result.returncode == 0, (
        f"scripts/{script_name} --metrics exited {result.returncode}: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


@pytest.mark.parametrize(
    ("script_name", "key"),
    [(script, key) for script, keys in sorted(_CONTRACT.items()) for key in keys],
)
def test_metrics_payload_carries_the_key_the_collector_reads(
    tmp_path: Path, script_name: str, key: str
) -> None:
    """Each payload key the collector reads is present and numeric.

    `collect_*_metrics` calls `data.get(<key>)`; a renamed or absent key is a
    silent `None`, which is exactly the failure this issue is about.
    """
    repo = _fake_repo(tmp_path, script_name)

    result = _run_metrics_mode(repo, script_name)
    payload: dict[str, Any] = json.loads(result.stdout.strip())

    assert key in payload, (
        f"scripts/{script_name} --metrics omitted {key!r}: {payload!r}"
    )
    assert isinstance(payload[key], (int, float)), (
        f"scripts/{script_name} --metrics reported a non-numeric {key!r}: "
        f"{payload[key]!r}"
    )


def test_an_unmeasurable_metric_is_still_a_successful_measurement(
    tmp_path: Path,
) -> None:
    """A refused number leaves stdout as ``null`` and the exit status at 0.

    The sibling test above proves a *bad* number exits 0. This proves an
    *absent* one does too, which is the harder half: the moment a probe cannot
    prove its number is exactly the moment it is most tempting to signal
    failure by exiting nonzero. Doing so would put a script that measures back
    in the business of gating -- and it must still print the JSON object, since
    a traceback on stdout is the decode error this whole issue is about.
    """
    repo = _fake_repo(tmp_path, "security.sh")
    _write_stub(repo / ".venv" / "bin" / "bandit", "import sys\nsys.exit(2)")

    result = _run_metrics_mode(repo, "security.sh")

    assert result.returncode == 0, (
        f"security.sh --metrics exited {result.returncode} because it could "
        f"not measure: stderr={result.stderr[-1000:]!r}"
    )
    assert json.loads(result.stdout.strip()) == {"bandit_issues": None}


def test_metrics_mode_skips_the_gate_it_normally_runs(tmp_path: Path) -> None:
    """``--metrics`` must not also run the enforcing half of a gate.

    `complexity.sh` gates on xenon; the metrics mode only measures with radon.
    Running xenon too would double the cost and, worse, let a gate failure
    escape through a collection path that is supposed to be side-effect free.
    """
    repo = _fake_repo(tmp_path, "complexity.sh")
    marker = tmp_path / "xenon-ran"
    _write_stub(
        repo / ".venv" / "bin" / "xenon",
        f"import pathlib\npathlib.Path({str(marker)!r}).write_text('ran')",
    )

    _run_metrics_mode(repo, "complexity.sh")

    assert not marker.exists(), "complexity.sh --metrics ran the xenon gate"


def test_default_mode_still_runs_the_gate(tmp_path: Path) -> None:
    """Without ``--metrics`` the gate behaves exactly as before.

    The counterpart to the test above: the enforcing tool must still run on the
    default path, so wiring in a measurement mode cannot quietly disarm Gate 1.
    """
    repo = _fake_repo(tmp_path, "complexity.sh")
    marker = tmp_path / "xenon-ran"
    _write_stub(
        repo / ".venv" / "bin" / "xenon",
        f"import pathlib\npathlib.Path({str(marker)!r}).write_text('ran')",
    )

    subprocess.run(
        [str(repo / "scripts" / "complexity.sh")],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert marker.exists(), "complexity.sh no longer runs the xenon gate by default"


# --------------------------------------------------------------------------
# 4. End to end: the collector fills every tile the dashboard renders
# --------------------------------------------------------------------------


def _run_collector(repo: Path, output: Path) -> subprocess.CompletedProcess[str]:
    """Run the real `collect_metrics.py` in script mode inside a fake checkout.

    Args:
        repo: Root of the fake checkout.
        output: Path the collector should write `metrics.json` to.

    Returns:
        The completed process with stdout/stderr captured as text.
    """
    return subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "collect_metrics.py"),
            "--project-name",
            "windbreak",
            "--output",
            str(output),
            "--metrics-mode",
            "script",
            "--scripts-dir",
            str(repo / "scripts"),
        ],
        cwd=str(repo),
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(repo)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_collector_script_mode_fills_every_quality_tile(tmp_path: Path) -> None:
    """The real collector, over the real scripts, leaves no quality key null.

    This is issue #122's acceptance criterion in miniature: ten tiles that all
    read N/A were indistinguishable from ten healthy ones. Mutation score is
    excluded on purpose -- mutmut is the manual pre-v1.0.0 gate (issue #107)
    and its gray tile is correct.
    """
    repo = _fake_repo(tmp_path, *sorted(_CONTRACT))
    shutil.copy(_SCRIPTS_DIR / "collect_metrics.py", repo / "scripts")
    output = repo / "metrics.json"

    completed = _run_collector(repo, output)
    assert completed.returncode == 0, (
        f"collector failed: stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )

    metrics = json.loads(output.read_text(encoding="utf-8"))["metrics"]
    still_na = [
        key
        for key in (
            "coverage",
            "branch_coverage",
            "complexity_avg",
            "maintainability_avg",
            "docs_coverage",
            "security_issues",
            "lint_violations",
            "type_errors",
            "tests_total",
        )
        if metrics.get(key) is None
    ]
    assert not still_na, f"these dashboard tiles are still N/A: {still_na}"


def test_collector_script_mode_leaves_mutation_score_unknown(tmp_path: Path) -> None:
    """Mutation score stays N/A by owner directive (issue #107).

    Nothing in the metrics path may run mutmut, so with no `.mutmut-cache` the
    tile must degrade to unknown rather than acquire a value.
    """
    repo = _fake_repo(tmp_path, *sorted(_CONTRACT))
    shutil.copy(_SCRIPTS_DIR / "collect_metrics.py", repo / "scripts")
    output = repo / "metrics.json"

    _run_collector(repo, output)

    metrics = json.loads(output.read_text(encoding="utf-8"))["metrics"]
    assert metrics["mutation_score"] is None
    assert metrics["mutation_status"] == "unknown"


def test_metrics_mode_never_runs_mutmut(tmp_path: Path) -> None:
    """`test.sh --metrics` must not reach for mutmut.

    Issue #107 makes mutation testing the manual pre-v1.0.0 gate. `test.sh`
    keeps its explicit ``--mutation`` flag for that, but a metrics run must
    never trigger it: the collector's 300 s timeout would kill the run and turn
    a deliberately gray tile into a mysterious one. The stub records any
    invocation, so this fails on behaviour rather than on a source grep.
    """
    repo = _fake_repo(tmp_path, "test.sh")
    marker = tmp_path / "mutmut-ran"
    _write_stub(
        repo / ".venv" / "bin" / "mutmut",
        f"import pathlib\npathlib.Path({str(marker)!r}).write_text('ran')",
    )

    result = _run_metrics_mode(repo, "test.sh")

    assert result.returncode == 0, f"test.sh --metrics failed: {result.stderr!r}"
    assert not marker.exists(), "test.sh --metrics invoked mutmut"


# --------------------------------------------------------------------------
# 5. The contract is derived from the collector, not asserted alongside it
# --------------------------------------------------------------------------


def _collector_contract() -> dict[str, tuple[str, ...]]:
    """Derive the script-mode contract from `scripts/collect_metrics.py` itself.

    Every test above is parametrised over `_CONTRACT`, a hand-written table.
    A hand-written table cannot notice a tile the collector *gained*: add a
    `collect_foo_metrics` that runs ``foo.sh --metrics`` and every parametrised
    test here keeps passing over the old six while the new tile renders N/A
    forever. So the truth is read out of the collector's AST: each method that
    calls ``collect_from_script("<script>", ...)`` is matched with the
    ``data.get("<key>")`` literals in the same method.

    Returns:
        Mapping of script filename to the payload keys the collector reads.
    """
    tree = ast.parse((_SCRIPTS_DIR / "collect_metrics.py").read_text(encoding="utf-8"))
    derived: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        script: str | None = None
        keys: list[str] = []
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not isinstance(
                call.func, ast.Attribute
            ):
                continue
            first = call.args[0] if call.args else None
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            if call.func.attr == "collect_from_script":
                script = first.value
            elif call.func.attr == "get":
                keys.append(first.value)
        if script is not None:
            derived[script] = tuple(sorted(keys))
    return derived


def test_the_contract_table_matches_what_the_collector_actually_reads() -> None:
    """`_CONTRACT` equals the script/key pairs found in the collector's source.

    Two ways for the dashboard to regress silently are closed here at once: a
    key renamed on the collector side (the tests would keep asserting the old
    one) and a *new* tile wired to a script that does not implement
    ``--metrics`` (nothing would notice). The derived mapping is asserted
    non-empty first, because a parser that suddenly matches nothing would
    otherwise make this test agree with any table at all.
    """
    derived = _collector_contract()

    assert derived, (
        "no collect_from_script(...) call was found in scripts/collect_metrics.py "
        "-- this test's AST parse has gone blind and can no longer detect drift"
    )
    assert derived == {
        script: tuple(sorted(keys)) for script, keys in _CONTRACT.items()
    }, (
        "the script/key table in this test no longer matches "
        f"scripts/collect_metrics.py: collector reads {derived!r}"
    )


# --------------------------------------------------------------------------
# 6. The real repository, measured by the real tools
# --------------------------------------------------------------------------

#: Scripts whose ``--metrics`` mode is run against the *real* checkout with the
#: *real* pinned tools. Everything above this point substitutes stub tools, so
#: it proves the wiring but would keep passing if ruff, radon, bandit or mypy
#: changed their output format and every probe started returning ``null``.
#: These four run in about five seconds in total.
_REAL_REPO_PROBES: Final = (
    "complexity.sh",
    "metrics-docs.sh",
    "security.sh",
    "typecheck.sh",
)

#: Scripts deliberately excluded from the real-repository run, and why. They
#: keep the stub-driven wiring coverage above; they gain no real-tool alarm.
_REAL_REPO_EXCLUSIONS: Final[dict[str, str]] = {
    "lint.sh": (
        "drives `pre-commit run shellcheck --all-files`, which would make the "
        "unit suite depend on pre-commit hook environments being installed"
    ),
    "coverage.sh": "runs the whole pytest suite; it cannot run inside itself",
    "test.sh": "shares coverage.sh's suite run; it cannot run inside itself",
}

#: Real-repository keys that are counts. Zero is a legitimate, healthy answer:
#: this repository really does have no bandit findings and no type errors.
_REAL_REPO_COUNTS: Final = frozenset({"bandit_issues", "errors"})

#: Real-repository keys that are averages or percentages, mapped to their
#: plausibility ceiling. Zero is *not* a healthy answer for these -- it is the
#: shape of a run that analysed nothing, and 0.0 renders the healthiest
#: possible complexity tile. The ceilings are sanity bounds, not thresholds:
#: the ``<= 10`` complexity gate belongs to xenon, and this file never gates.
_REAL_REPO_AVERAGE_CEILINGS: Final[dict[str, float]] = {
    "cyclomatic_avg": 1000.0,
    "maintainability_avg": 100.0,
    "docs_coverage_pct": 100.0,
}


def test_every_contract_script_is_either_measured_for_real_or_excused() -> None:
    """No script may drop out of real-tool coverage without a written reason.

    The exclusions are a real cost -- those tiles are only protected by stub
    wiring -- so they are enumerated rather than implied. Silently moving a
    script out of `_REAL_REPO_PROBES` fails here instead of quietly halving
    what this file proves.
    """
    assert _REAL_REPO_PROBES, "no script is measured against the real repository"
    assert set(_REAL_REPO_PROBES) | set(_REAL_REPO_EXCLUSIONS) == set(_CONTRACT), (
        "the real-repository split does not cover the contract: "
        f"measured={sorted(_REAL_REPO_PROBES)} excused={sorted(_REAL_REPO_EXCLUSIONS)}"
    )
    assert not set(_REAL_REPO_PROBES) & set(_REAL_REPO_EXCLUSIONS)
    measured_keys = {key for name in _REAL_REPO_PROBES for key in _CONTRACT[name]}
    assert measured_keys == _REAL_REPO_COUNTS | set(_REAL_REPO_AVERAGE_CEILINGS), (
        "every measured key must be classified as a count or an average, or "
        f"its plausibility check is skipped: {sorted(measured_keys)}"
    )


@pytest.mark.parametrize("script_name", sorted(_REAL_REPO_PROBES))
def test_real_scripts_measure_the_real_repository_without_going_null(
    script_name: str,
) -> None:
    """Against this checkout and the pinned tools, no contract key is ``null``.

    This is the alarm issue #122 asks for. A ``null`` here is precisely the
    silent degradation the dashboard suffered: the collector writes
    `null`/``"unknown"``, the tile renders a neutral gray N/A, and to a
    glancing operator that is indistinguishable from a healthy one. If radon,
    ruff or bandit ever change their output shape so a probe stops being able
    to prove its number, this fails rather than the dashboard going quiet.

    Args:
        script_name: Gate script whose ``--metrics`` mode is under test.
    """
    result = subprocess.run(
        [str(_SCRIPTS_DIR / script_name), "--metrics"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        f"scripts/{script_name} --metrics exited {result.returncode}: "
        f"stderr={result.stderr[-2000:]!r}"
    )
    payload = json.loads(result.stdout.strip())
    null_keys = [key for key in _CONTRACT[script_name] if payload.get(key) is None]
    assert not null_keys, (
        f"scripts/{script_name} --metrics could not measure {null_keys} against "
        f"the real repository, so those tiles would render N/A: {payload!r}"
    )
    for key in _CONTRACT[script_name]:
        value = payload[key]
        if key in _REAL_REPO_COUNTS:
            assert isinstance(value, int) and value >= 0, (
                f"{key} is not a non-negative count: {value!r}"
            )
        else:
            assert 0.0 < value <= _REAL_REPO_AVERAGE_CEILINGS[key], (
                f"{key} is outside its plausible range -- a zero average is "
                f"the shape of a run that analysed nothing: {value!r}"
            )


def test_probes_run_their_tools_with_the_pinned_toolchain_on_path(
    probe: types.ModuleType, tmp_path: Path
) -> None:
    """A measured tool can resolve its pinned siblings by bare name.

    `scripts/toolchain-env.sh` hands each probe an absolute tool path, but the
    tools themselves spawn helpers off `PATH`: the pytest suite shells out to
    `lint-imports`, pre-commit to its hook binaries. Without the pinned `bin/`
    in front of `PATH` those lookups miss, and the probe reports failures that
    belong to the caller's environment rather than to this repository -- three
    phantom test failures on the Test Count tile, measured live.

    The stub ruff here refuses to run unless it can resolve a sibling planted
    beside it, so this fails on behaviour rather than on an inspected
    environment variable.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub(bin_dir / "sibling-helper", "pass")
    sample = tmp_path / "sample.py"
    sample.write_text('"""Doc."""\n', encoding="utf-8")
    ruff = _write_stub(
        bin_dir / "ruff",
        "import shutil, sys\n"
        "if shutil.which('sibling-helper') is None:\n"
        "    sys.exit(2)\n"
        "if '--show-files' in sys.argv:\n"
        f"    sys.stdout.write({str(sample)!r} + chr(10))\n"
        "else:\n"
        "    sys.stdout.write('[]')\n"
        "sys.exit(0)",
    )

    assert probe.probe_docs(str(ruff)) == {"docs_coverage_pct": 100.0}, (
        "the pinned bin/ was not prepended to PATH, so the tool could not "
        "resolve its sibling and the probe degraded to null"
    )
