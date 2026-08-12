"""No governing document states a threshold this repository does not enforce (#547).

PR #548 (`test_verification_claims.py`) answers *"is the named tool real?"*. It
deliberately does not answer the neighbouring question, which is this module's:
**"is the named number the enforced one?"** A brief can name the right tool and
still send an agent at the wrong target, and that is harder to spot precisely
because nothing looks broken.

Three claims motivated the issue:

* `.claude/agents/ralph-implementation-specialist.md` and
  `.claude/agents/ralph-performance-specialist.md` both told an agent to keep
  functions at **xenon A-grade**. The gate is `xenon --max-absolute B`, i.e.
  radon's B band, i.e. cyclomatic complexity <=10 -- exactly what `CLAUDE.md`
  publishes. Stricter-than-real is not the harmless direction it looks: an agent
  believing A is required refactors code the gate accepts, or reports a file
  BLOCKED that passes Gate 1 cleanly, and the second is indistinguishable from a
  genuine failure to whoever reads the report.
* Both also required **radon MI >= B**. `scripts/complexity.sh` runs `radon mi`
  with `|| true`; xenon is the only enforcing half of that script. There is no
  maintainability floor to be under.
* `.claude/agents/ralph-test-specialist.md`, `scripts/ralph/PROMPT.md` and two
  documents under `prompts/` required **>=80% branch** coverage. That floor does
  not exist either. `[tool.coverage.run] branch = true` folds branch outcomes
  into a SINGLE figure measured against `fail_under = 90`; there is no second
  threshold. An agent targeting 80% on a dimension with no floor is aiming at
  nothing, and would read 82% as passing something.

This is the eighth instance of one defect class in this backlog drain -- a
documented gate that cannot fail, or that names a number the repo does not
enforce (#351, #359, #401, #411, #534/#537, #536, #543/#546). Every one was
believed *because it was written down*.

THE TWO DERIVATIONS, AND WHY THEY MUST NOT MEET

The hard part of this issue is that extracting an ENFORCED threshold and
extracting a CLAIMED one must read different sources, or the test proves
nothing. A module that read `CLAUDE.md`'s table for both would be measuring the
table against itself and would pass forever -- the coincidence trap, which has
hidden a real defect eight times in this session (most recently PR #529, where a
version-pinning test would have compared one value against itself had the lane
not added a positive control).

So the enforced side comes only from where enforcement actually lives, one named
authority per gate (:data:`_GATES`):

===========================  ===============================================
gate                         authority
===========================  ===============================================
``coverage``                 ``pyproject.toml`` ``[tool.coverage.report]``
``branch-coverage``          ``pyproject.toml`` (derived absence -- see below)
``docstring-coverage``       ``pyproject.toml`` ``[tool.ruff.lint].select``
``cyclomatic-complexity``    ``scripts/complexity.sh`` (xenon's real flag)
``maintainability-index``    ``scripts/complexity.sh`` (derived absence)
``mutation-score``           ``scripts/mutation.sh`` (``MIN_SCORE``)
===========================  ===============================================

and the claimed side comes only from the prose corpus PR #548 established.
:func:`test_no_threshold_authority_is_a_document_in_the_scanned_corpus` asserts
the two file sets are disjoint MECHANICALLY -- an authority is not markdown, and
is not one of the files the corpus roots contain whatever its suffix -- so
repointing an extractor at the prose it is supposed to check turns this suite
red rather than green.

THE ABSENCE CASE, WHICH IS NOT A NUMBER COMPARISON

`>=80% branch` is wrong not because the number differs but because no such
threshold exists. A test that only compares numbers where both sides have one
walks straight past it. So a gate's enforced floor is `Decimal | None`, `None`
means "this repository gates nothing on this dimension", and a bound claimed on
a `None` gate is its own failure with its own control
(:func:`test_a_bound_on_an_ungated_dimension_is_a_violation`).

Both absences are DERIVED, not asserted by hand. `branch-coverage` is `None`
because `[tool.coverage.report]` declares exactly one failure threshold and it
is not qualified by outcome type; add `fail_under_branch` and the extractor
returns it. `maintainability-index` is `None` because the `radon mi` line in
`scripts/complexity.sh` is suffixed `|| true`; drop the suffix and the extractor
finds a gating invocation and says so.

DISCUSSING AN UNGATED DIMENSION IS STILL ALLOWED

`.claude/docs/quality-standards.md` lists a maintainability index "aim >=20"
under a **Guidance, not gated** heading, and `CLAUDE.md`'s own table states the
>=90% figure branch outcomes fold into. Neither is a false claim; both are the
honest form of the sentence. A rule that could not tell them from
`radon MI >= B` would have to be excepted for the two documents that get this
RIGHT, which is how a rule stops being applied. So a bound on an ungated gate is
a violation *unless its block states that nothing enforces it* -- a small, fixed
disclaimer vocabulary (:data:`_DISCLAIMER`), matched over the whitespace-
normalised block. That is a stronger specification than "never write the
number": it permits the aspiration and forbids the silent claim.

WHAT COUNTS AS A CLAIMED BOUND, AND WHAT IS DELIBERATELY NOT ONE

A number in a document is usually a measurement, not a requirement, and a
detector that cannot tell the difference fires on everything and gets turned
off. Three filters, all mechanical:

* **A bound carries an operator or a requirement word.** `>=90%`,
  `90%+ required`, `80% minimum`. `Total coverage: 57.14%` -- quoted from a
  pytest failure in `.claude/skills/ci-debugging/SKILL.md` -- carries neither and
  is not read as a claim.
* **A bound stated as a property of specific code is a measurement.**
  `Each function has complexity <= 3` in the `max-quality-no-shortcuts`
  worked example reports what a refactor achieved; the verb (:data:`_MEASURED`)
  is what distinguishes it from `complexity <=10 per function`.
* **The percentage gates require a `%`.** `Max Branches: 12 per function` is a
  branch COUNT, not a branch-coverage floor, and only the `%` separates them.

Attribution is clause-scoped. A bound takes the gate keyword in its own clause;
failing that, the nearest clause that does not already carry a bound of its own
-- because a clause with a bound has already spoken for its keyword. That is
what keeps `>=90% line / >=80% branch` from collapsing into one dimension, and
what leaves `>=90% Jest frontend` unattributed rather than silently charged to
`branch`.

WHAT IS NOT VERIFIED. The gate registry is written down, because no derivation
can enumerate the dimensions a repository might be claimed to gate; every
VERDICT about a gate in it is derived. Frontend thresholds (`>=90% Jest`) are
outside it -- this repository ships no frontend and no Jest configuration, which
is a phantom-gate defect of a different shape, filed separately rather than
smuggled in here. A bound whose gate keyword sits in no clause the attribution
rule reaches is invisible, as is a bound written without an operator or a
requirement word. This module does not parse English: it checks the parts that
are mechanical, and every filter above is controlled in both directions below.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
from radon.complexity import cc_rank

if TYPE_CHECKING:
    from pathlib import Path

from tests.toolchain.test_verification_claims import (
    _CORPUS_ROOTS,
    _REPO_ROOT,
    _corpus_documents,
    _documents_under,
    _ruff_docstring_rule,
)

_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_COMPLEXITY_SCRIPT = _REPO_ROOT / "scripts" / "complexity.sh"
_MUTATION_SCRIPT = _REPO_ROOT / "scripts" / "mutation.sh"

_IMPLEMENTATION_SPECIALIST = (
    _REPO_ROOT / ".claude" / "agents" / "ralph-implementation-specialist.md"
)
_PERFORMANCE_SPECIALIST = (
    _REPO_ROOT / ".claude" / "agents" / "ralph-performance-specialist.md"
)
_TEST_SPECIALIST = _REPO_ROOT / ".claude" / "agents" / "ralph-test-specialist.md"
_WORKER_PROMPT = _REPO_ROOT / "scripts" / "ralph" / "PROMPT.md"


@dataclass(frozen=True)
class _Gate:
    """One dimension this repository is claimed, somewhere, to gate.

    Attributes:
        name: The gate's identifier, used in failure messages.
        authority: The file where enforcement of this gate actually lives. It is
            asserted to be outside the prose corpus, which is what keeps the
            enforced and claimed derivations independent.
        keywords: Regexes naming the gate in prose, most specific first.
        percentage: Whether a bound on this gate is written as a percentage. The
            percentage gates ignore bare counts, so `Max Branches: 12 per
            function` is not read as a branch-coverage floor.
    """

    name: str
    authority: Path
    keywords: tuple[str, ...]
    percentage: bool


#: The gates, in attribution priority order: a bound equidistant from two
#: keywords takes the earlier gate, so `branch coverage >=85%` is charged to
#: `branch-coverage` rather than to the generic `coverage` whose keyword ends at
#: the same offset.
_GATES: tuple[_Gate, ...] = (
    _Gate(
        name="branch-coverage",
        authority=_PYPROJECT,
        keywords=(r"branch coverage", r"branch"),
        percentage=True,
    ),
    _Gate(
        name="docstring-coverage",
        authority=_PYPROJECT,
        keywords=(r"docstring coverage", r"docstring"),
        percentage=True,
    ),
    _Gate(
        name="mutation-score",
        authority=_MUTATION_SCRIPT,
        keywords=(r"mutation score", r"mutation", r"mutmut"),
        percentage=True,
    ),
    _Gate(
        name="maintainability-index",
        authority=_COMPLEXITY_SCRIPT,
        keywords=(r"maintainability index", r"maintainability", r"\bMI\b"),
        percentage=False,
    ),
    _Gate(
        name="cyclomatic-complexity",
        authority=_COMPLEXITY_SCRIPT,
        keywords=(r"cyclomatic complexity", r"complexity", r"xenon"),
        percentage=False,
    ),
    _Gate(
        name="coverage",
        authority=_PYPROJECT,
        keywords=(r"line coverage", r"coverage", r"\bline\b"),
        percentage=True,
    ),
)

_GATES_BY_NAME = {gate.name: gate for gate in _GATES}

#: A number, optionally a percentage. Bare group 1 is the value.
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%?)")

#: A comparison operator, matched at the END of the text preceding a number.
_OPERATOR = re.compile(r"(?:>=|<=|=<|=>|[≥≤><])\s*$")

#: Words that turn an adjacent number into a requirement rather than a reading.
#: Deliberately short, and shorter than the obvious draft: `min` was removed
#: because `| Comment tests | 15% | 15 min |` in `troubleshooting.md`'s summary
#: table is minutes, and `threshold` because `(threshold 8)` in
#: `prompts/scans/complexity.md` is an eslint setting in a worked example. Both
#: made a frequency table read as a coverage claim.
_REQUIREMENT = re.compile(
    r"\b(?:minimum|at\s+least|require[ds]?|meet|fail[_-]under|cov-fail-under)\b",
    re.IGNORECASE,
)

#: A requirement word may sit this far before its number, provided no other digit
#: intervenes -- `Required test coverage of 90%` reaches, and the `57.14%` later
#: in that same quoted pytest message does not, because `90` is in between.
_REQUIREMENT_REACH = 35

#: A requirement word may also FOLLOW its number closely: `90%+ required`,
#: `80% minimum (mutmut)`.
_TRAILING_REQUIREMENT_REACH = 12

#: Verbs that make a bound a reading of particular code rather than a rule for
#: all of it. `Each function HAS complexity <= 3` reports a refactor's result.
_MEASURED = re.compile(
    r"\b(?:has|have|had|is|are|was|were|shows?|reports?|grades?|reached|reaches"
    r"|scored?|measured|came\s+in)\b",
    re.IGNORECASE,
)

#: Clause boundaries. A bound takes its meaning from the clause it sits in.
#: Conjunctions count, however they are emphasised: `>=80% branch backend gate
#: and the >=90% Jest frontend gate` is two claims sharing a sentence, and
#: without the split the second one is charged to the first one's keyword.
#: The dashes are spelled as escapes because ruff `RUF001` reads a literal en
#: dash as an ambiguous character; both dashes punctuate a clause in this corpus.
_CLAUSE_DELIMITER = re.compile(
    r"[,;/|()\[\]\u2014\u2013]|\.\s|\.$|:\s|\n|\W+(?:and|or)\W+"
)

#: A statement that nothing enforces the number beside it. Matched over the
#: whitespace-normalised BLOCK, so `Branch coverage has no\n  separate floor`
#: still reads as one phrase. Kept deliberately small -- "manual gate" and "not
#: automated" are absent on purpose: the mutation score IS enforced, by
#: `scripts/mutation.sh`, and must still match the number that script uses.
_DISCLAIMER = re.compile(
    r"not gated|nothing enforces|nothing measures|no tool fails"
    r"|not enforced|enforced by nothing|no separate floor|no second floor"
    r"|has no floor|report-only|reports it, nothing",
    re.IGNORECASE,
)

#: A block that quotes a number in order to FORBID it.
#: `.claude/docs/troubleshooting.md` shows `xenon --max-absolute C  # Relaxed
#: from B` under a `❌ **FORBIDDEN - Reducing standards:**` heading, which is the
#: document teaching the exact mistake this module detects. A rule that could not
#: tell a worked counterexample from a claim would have to be excepted for the
#: page most emphatically on its side -- the same reason
#: `test_verification_claims` strips suppression directives before reading a
#: tool name out of an anti-bypass rule. Case-sensitive on purpose: these
#: documents shout their counterexamples, and lowercase "wrong" is ordinary
#: prose.
_COUNTEREXAMPLE = re.compile(r"\bWRONG\b|\bFORBIDDEN\b|❌|anti-?pattern")

#: Blocks are separated by blank lines, exactly as in `test_verification_claims`.
_BLOCK_SPLIT = re.compile(r"\n\s*\n")

#: The xenon grade the complexity gate is set to, read from the flag itself.
_MAX_ABSOLUTE = re.compile(r"--max-absolute\s+([A-F])\b")

#: A xenon grade as prose writes it: ``xenon A``, ``xenon A-grade``,
#: ``xenon (`--max-absolute B`)``.
_XENON_GRADE = re.compile(r"xenon[\s(`]*(?:--max-absolute[\s`]*)?([A-F])\b")

#: A maintainability grade as prose writes it: ``radon MI >= B``.
_MI_GRADE = re.compile(
    r"\bMI\b\s*(?:>=|<=|[≥≤><])?\s*([A-F])\b",
)

#: mutmut's floor, as `scripts/mutation.sh` sets it.
_MIN_SCORE = re.compile(r"^MIN_SCORE=(\d+(?:\.\d+)?)", re.MULTILINE)

#: A `radon mi` invocation in `scripts/complexity.sh`, and whether it gates.
_RADON_MI = re.compile(r"^(?P<command>.*\bmi\b.*?)$", re.MULTILINE)

#: Lower bounds, in the shape PR #548 established. The corpus stated 23
#: threshold claims across 13 documents when this was written; floors well under
#: those catch a scan that has stopped scanning -- trap #5, a corpus walk over
#: zero hits passes forever, which is the very defect class this module belongs
#: to -- without failing every time a document is edited.
_CLAIM_FLOOR = 12
_CLAIMING_DOCUMENT_FLOOR = 6
_GATE_FLOOR = 3

#: Every corpus root must state at least this many thresholds of its own. A
#: global floor cannot see a root that has gone silent: `.claude/` alone clears
#: every total above, so `scripts/ralph/` -- two documents, the smaller of which
#: is the Ralph worker's whole operating contract, and one of the documents that
#: carried `>=80% branch` -- would be invisible in the aggregate.
_ROOT_CLAIM_FLOOR = 1


@dataclass(frozen=True)
class _Bound:
    """A number a document states as a requirement, and the gate it names.

    Attributes:
        document: The document it was read from.
        line: The 1-based line it sits on, so a failure names a place.
        gate: The gate it was attributed to.
        value: Its value, normalised to the unit the gate is enforced in --
            a percentage, or a cyclomatic-complexity ceiling.
        text: The clause it was read from, for the failure message.
        disclaimed: Whether its block states that nothing enforces it.
    """

    document: Path
    line: int
    gate: str
    value: Decimal
    text: str
    disclaimed: bool

    def where(self) -> str:
        """Locate this bound for a failure message.

        Returns:
            A `path:line` string relative to the repository root.
        """
        return f"{self.document.relative_to(_REPO_ROOT)}:{self.line}"


# --- The enforced side: derived from the enforcing configuration only ---------


def _coverage_report_config() -> dict[str, object]:
    """Read coverage.py's report configuration out of `pyproject.toml`.

    Returns:
        The `[tool.coverage.report]` table.
    """
    tool = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8")).get("tool", {})
    assert isinstance(tool, dict)
    coverage = tool.get("coverage", {})
    assert isinstance(coverage, dict)
    report = coverage.get("report", {})
    assert isinstance(report, dict)
    return report


def _branch_measurement_is_on() -> bool:
    """Report whether coverage.py is measuring branch outcomes at all.

    Returns:
        `[tool.coverage.run] branch`.
    """
    tool = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8")).get("tool", {})
    assert isinstance(tool, dict)
    coverage = tool.get("coverage", {})
    assert isinstance(coverage, dict)
    run = coverage.get("run", {})
    assert isinstance(run, dict)
    return bool(run.get("branch", False))


def _enforced_coverage_floor() -> Decimal:
    """Derive the coverage percentage this repository fails a build under.

    Returns:
        `[tool.coverage.report] fail_under`.

    Raises:
        AssertionError: If no failure threshold is configured, which would make
            every coverage claim in the corpus a claim about nothing.
    """
    report = _coverage_report_config()
    floor = report.get("fail_under")
    assert isinstance(floor, int | float | str), (
        "pyproject.toml's [tool.coverage.report] configures no fail_under, so "
        "the coverage gate every governing document cites does not exist."
    )
    return Decimal(str(floor))


def _enforced_branch_floor() -> Decimal | None:
    """Derive the branch-specific coverage floor, if this repository has one.

    Branch outcomes are MEASURED -- `[tool.coverage.run] branch = true` -- and
    folded into the single combined figure `fail_under` gates. A branch-specific
    floor would have to be a second, outcome-qualified threshold key; there is
    none, so there is nothing for `>=80% branch` to be a claim about.

    Returns:
        The branch-qualified failure threshold, or `None` when the only
        threshold declared is the combined one.

    Raises:
        AssertionError: If branch measurement is off, which would make even the
            combined figure something other than what the docs describe.
    """
    assert _branch_measurement_is_on(), (
        "pyproject.toml's [tool.coverage.run] no longer sets branch = true, so "
        "branch outcomes are not folded into the combined figure and the "
        "documented coverage rule describes a measurement that stopped."
    )
    report = _coverage_report_config()
    qualified = {
        key: value
        for key, value in report.items()
        if "fail_under" in key and key != "fail_under"
    }
    if not qualified:
        return None
    return Decimal(str(next(iter(qualified.values()))))


def _enforced_docstring_floor() -> Decimal | None:
    """Derive the docstring-coverage percentage this repository gates on.

    Ruff `D1` is a PRESENCE rule, applied per public symbol (issue #351). There
    is no fraction of symbols permitted to lose their docstrings, and therefore
    no percentage a document can correctly cite.

    Returns:
        `None`, once the D1 family is confirmed to be what ruff selects.
    """
    assert _ruff_docstring_rule().startswith("D1")
    return None


def _complexity_script_commands() -> list[str]:
    """List the commands `scripts/complexity.sh` runs, gating ones only.

    A line suffixed `|| true` cannot fail the script and therefore enforces
    nothing -- which is exactly the status of its `radon mi` call.

    Returns:
        The command lines that can fail the script.
    """
    return [
        line
        for line in _COMPLEXITY_SCRIPT.read_text(encoding="utf-8").splitlines()
        if ("$RADON" in line or "$XENON" in line) and "|| true" not in line
    ]


def _enforced_complexity_ceiling() -> Decimal:
    """Derive the per-function cyclomatic complexity ceiling Gate 1 enforces.

    The grade comes from xenon's real `--max-absolute` flag in
    `scripts/complexity.sh`; the number that grade stands for comes from radon's
    own `cc_rank`, so the band boundaries are the tool's and not a table copied
    into this file.

    Returns:
        The highest cyclomatic complexity the configured grade accepts.

    Raises:
        AssertionError: If no gating xenon invocation carries the flag.
    """
    grades = [
        match.group(1)
        for line in _complexity_script_commands()
        for match in [_MAX_ABSOLUTE.search(line)]
        if match
    ]
    assert grades, (
        "no gating xenon invocation in scripts/complexity.sh carries "
        "--max-absolute, so the complexity ceiling every brief cites is not "
        f"enforced there. Gating commands: {_complexity_script_commands()!r}"
    )
    return _grade_ceiling(grades[0])


def _enforced_maintainability_floor() -> Decimal | None:
    """Derive the maintainability-index floor, if this repository has one.

    Returns:
        `None` while `radon mi` runs only in the report-only half of
        `scripts/complexity.sh`.

    Raises:
        AssertionError: If the script no longer invokes `radon mi` at all, which
            would make this absence a fact about a missing command rather than
            about a non-enforcing one.
    """
    source = _COMPLEXITY_SCRIPT.read_text(encoding="utf-8")
    invocations = [
        line
        for line in source.splitlines()
        if "$RADON" in line and re.search(r"\bmi\b", line)
    ]
    assert invocations, (
        "scripts/complexity.sh no longer invokes `radon mi` anywhere, so "
        "'nothing enforces the maintainability index' would be true for the "
        "uninteresting reason that nothing computes it either."
    )
    gating = [line for line in invocations if line in _complexity_script_commands()]
    if not gating:
        return None
    return Decimal("20")


def _enforced_mutation_floor() -> Decimal:
    """Derive the mutation score `scripts/mutation.sh` fails under.

    Returns:
        The script's `MIN_SCORE` default.

    Raises:
        AssertionError: If the script sets no default.
    """
    match = _MIN_SCORE.search(_MUTATION_SCRIPT.read_text(encoding="utf-8"))
    assert match, (
        "scripts/mutation.sh sets no MIN_SCORE default, so the >=80% mutation "
        "score the docs publish is enforced by nothing this module can find."
    )
    return Decimal(match.group(1))


def _grade_ceiling(grade: str) -> Decimal:
    """Translate a radon complexity grade into the highest score it accepts.

    Args:
        grade: A radon rank letter, `A` through `F`.

    Returns:
        The largest cyclomatic complexity `radon.complexity.cc_rank` still ranks
        at that grade.

    Raises:
        AssertionError: If radon ranks nothing at the grade, or ranks an
            unbounded band at it -- `F` is open-ended and cannot be a ceiling.
    """
    scores = [score for score in range(1, 101) if cc_rank(score) == grade]
    assert scores, f"radon ranks no cyclomatic complexity at grade {grade!r}."
    assert max(scores) < 100, (
        f"radon's {grade!r} band is unbounded, so it states no ceiling a "
        "document could correctly cite."
    )
    return Decimal(max(scores))


def _enforced_floors() -> dict[str, Decimal | None]:
    """Derive every gate's enforced bound from its own authority.

    Returns:
        Gate name to the bound this repository enforces, or `None` where it
        enforces none.
    """
    return {
        "coverage": _enforced_coverage_floor(),
        "branch-coverage": _enforced_branch_floor(),
        "docstring-coverage": _enforced_docstring_floor(),
        "cyclomatic-complexity": _enforced_complexity_ceiling(),
        "maintainability-index": _enforced_maintainability_floor(),
        "mutation-score": _enforced_mutation_floor(),
    }


# --- The claimed side: derived from the prose corpus only ---------------------


def _clause_spans(line: str) -> list[tuple[int, int]]:
    """Split a line into clauses, keeping each one's offsets.

    Args:
        line: The line to split.

    Returns:
        `(start, end)` offsets of each non-empty clause, in order.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for delimiter in _CLAUSE_DELIMITER.finditer(line):
        spans.append((start, delimiter.start()))
        start = delimiter.end()
    spans.append((start, len(line)))
    return [(begin, end) for begin, end in spans if line[begin:end].strip()]


def _is_required(line: str, start: int, end: int) -> bool:
    """Decide whether a number at these offsets is stated as a requirement.

    Args:
        line: The line the number sits on.
        start: The number's start offset.
        end: The offset just past the number and any `%`.

    Returns:
        True when an operator immediately precedes it, or a requirement word
        sits close enough on either side.
    """
    if _OPERATOR.search(line[:start]):
        return True
    before = line[max(0, start - _REQUIREMENT_REACH) : start]
    for match in _REQUIREMENT.finditer(before):
        if not re.search(r"\d", before[match.end() :]):
            return True
    after = line[end : end + _TRAILING_REQUIREMENT_REACH]
    return bool(_REQUIREMENT.search(after))


def _attribute(line: str, spans: list[tuple[int, int]], index: int) -> _Gate | None:
    """Name the gate a bound in clause `index` is talking about.

    The bound takes a keyword from its own clause first. Failing that it reaches
    outward to the nearest clause that carries no bound of its own, because a
    clause with a bound has already spoken for whatever keyword it holds -- which
    is what stops `>=90% Jest frontend` being charged to the `branch` in the
    clause before it.

    Args:
        line: The line being read.
        spans: Its clause offsets.
        index: The clause the bound sits in.

    Returns:
        The gate, or `None` when no clause in reach names one.
    """
    occupied = {
        position
        for position, (begin, end) in enumerate(spans)
        if position != index and _bounds_in_clause(line, begin, end)
    }
    order = [index] + [
        position
        for position in sorted(
            (p for p in range(len(spans)) if p != index),
            key=lambda p: abs(p - index),
        )
        if position not in occupied
    ]
    for position in order:
        begin, end = spans[position]
        gate = _gate_named_in(line[begin:end])
        if gate is not None:
            return gate
    return None


def _bounds_in_clause(line: str, begin: int, end: int) -> bool:
    """Report whether a clause states a requirement number of its own.

    Args:
        line: The line being read.
        begin: The clause's start offset.
        end: The clause's end offset.

    Returns:
        True when the clause holds at least one required number.
    """
    return any(
        _is_required(line, begin + match.start(1), begin + match.end(2))
        for match in _NUMBER.finditer(line[begin:end])
    )


def _gate_named_in(clause: str) -> _Gate | None:
    """Name the gate a clause mentions, most specific first.

    Args:
        clause: The clause text.

    Returns:
        The first gate whose keyword the clause carries, or `None`.
    """
    for gate in _GATES:
        for keyword in gate.keywords:
            if re.search(keyword, clause, re.IGNORECASE):
                return gate
    return None


def _numeric_bounds_on_line(
    document: Path, number: int, line: str, disclaimed: bool
) -> list[_Bound]:
    """Read every numeric bound one line states.

    Args:
        document: The document, for reporting.
        number: The 1-based line number.
        line: The line's text.
        disclaimed: Whether the enclosing block says nothing enforces it.

    Returns:
        One `_Bound` per attributed requirement number.
    """
    spans = _clause_spans(line)
    bounds: list[_Bound] = []
    for index, (begin, end) in enumerate(spans):
        clause = line[begin:end]
        if _MEASURED.search(clause):
            continue
        for match in _NUMBER.finditer(clause):
            if not _is_required(line, begin + match.start(1), begin + match.end(2)):
                continue
            gate = _attribute(line, spans, index)
            if gate is None or (gate.percentage and not match.group(2)):
                continue
            bounds.append(
                _Bound(
                    document=document,
                    line=number,
                    gate=gate.name,
                    value=Decimal(match.group(1)),
                    text=clause.strip(),
                    disclaimed=disclaimed,
                )
            )
    return bounds


def _grade_bounds_on_line(
    document: Path, number: int, line: str, disclaimed: bool
) -> list[_Bound]:
    """Read every letter-grade bound one line states.

    Grades are anchored to the tool that could enforce them -- xenon for
    cyclomatic complexity, `MI` for the maintainability index -- so a scan
    heuristic like `radon CC grade >= C`, which names neither, is not read as a
    claim about this repository's gates.

    Args:
        document: The document, for reporting.
        number: The 1-based line number.
        line: The line's text.
        disclaimed: Whether the enclosing block says nothing enforces it.

    Returns:
        One `_Bound` per grade, valued in the gate's own unit.
    """
    bounds: list[_Bound] = []
    for match in _XENON_GRADE.finditer(line):
        bounds.append(
            _Bound(
                document=document,
                line=number,
                gate="cyclomatic-complexity",
                value=_grade_ceiling(match.group(1)),
                text=match.group(0),
                disclaimed=disclaimed,
            )
        )
    for match in _MI_GRADE.finditer(line):
        bounds.append(
            _Bound(
                document=document,
                line=number,
                gate="maintainability-index",
                value=_grade_ceiling(match.group(1)),
                text=match.group(0),
                disclaimed=disclaimed,
            )
        )
    return bounds


def _bounds_in(document: Path, text: str) -> list[_Bound]:
    """Read every threshold one document states.

    Args:
        document: The document, for reporting.
        text: Its full source.

    Returns:
        The bounds, in document order.
    """
    bounds: list[_Bound] = []
    offset = 0
    for block in _BLOCK_SPLIT.split(text):
        first = text[:offset].count("\n") + 1
        offset += len(block) + 2
        flattened = " ".join(block.split())
        if _COUNTEREXAMPLE.search(flattened):
            continue
        disclaimed = bool(_DISCLAIMER.search(flattened))
        for index, line in enumerate(block.splitlines()):
            number = first + index
            bounds.extend(_numeric_bounds_on_line(document, number, line, disclaimed))
            bounds.extend(_grade_bounds_on_line(document, number, line, disclaimed))
    return bounds


def _corpus_neighbourhood() -> frozenset[Path]:
    """List every file the prose corpus is made of, whatever its suffix.

    The claim reader takes only `*.md`, but the roots hold scripts and data
    beside them. An authority parked in that neighbourhood is one widening of
    the walk away from being read as prose by the very module that calls it the
    enforcing side, and its suffix would not give it away.

    Returns:
        Every regular file under a corpus root, plus `CLAUDE.md`.
    """
    return frozenset(
        {_CLAUDE_MD}
        | {path for root in _CORPUS_ROOTS for path in root.rglob("*") if path.is_file()}
    )


def _claimed_bounds() -> tuple[_Bound, ...]:
    """Read every threshold the governing corpus states.

    Returns:
        The bounds, in document order.
    """
    return tuple(
        bound
        for document in _corpus_documents()
        for bound in _bounds_in(document, document.read_text(encoding="utf-8"))
    )


# --- Controls: the detector fires where it must, and only there ---------------

#: The claim this issue removed from two agent definitions, verbatim.
_XENON_A_CLAIM = "keep functions xenon A-grade / radon MI >= B, satisfy mypy strict"

#: The claim four documents carried, verbatim, on a dimension with no floor.
_BRANCH_CLAIM = "covered (>=90% line / >=80% branch backend), whether assertions"

#: A plausible but wrong number on a gate that is real. Nothing about it looks
#: broken; it is five points off `fail_under`.
_WRONG_COVERAGE_CLAIM = "Tests must hold coverage >= 85% before review."

#: The same sentence with the enforced number in it.
_RIGHT_COVERAGE_CLAIM = "Tests must hold coverage >= 90% before review."

#: A reading of specific code, quoted from a pytest failure. Two numbers, both
#: measurements, neither a rule anyone is being asked to meet.
_MEASUREMENT = (
    '**Symptom**: "Required test coverage of 90% not reached. Total coverage: 57.14%"'
)

#: A result reported about a worked example, not a rule for all functions.
_REFACTOR_RESULT = (
    "**Result**: Each function has complexity <= 3. Clear, testable units."
)

#: A branch COUNT. Only the missing `%` separates it from a coverage floor.
_BRANCH_COUNT = "- **Max Branches**: 12 per function"

#: The worked counterexample `.claude/docs/troubleshooting.md` teaches with,
#: verbatim: the loosened flag, quoted so that a reader recognises it.
_FORBIDDEN_EXAMPLE = (
    "```bash\n"
    "# In scripts/complexity.sh - WRONG\n"
    "xenon --max-absolute C  # Relaxed from B because a function grew\n"
    "```"
)

#: The same two lines with the counterexample marker gone -- which is what a
#: document that had actually loosened the gate would look like.
_UNMARKED_EXAMPLE = (
    "```bash\n"
    "# In scripts/complexity.sh\n"
    "xenon --max-absolute C  # Relaxed from B because a function grew\n"
    "```"
)

#: An ungated dimension discussed honestly, as `quality-standards.md` discusses
#: it. The number is the same one `radon mi` prints; the sentence says so.
_DISCLAIMED_CLAIM = (
    "**Guidance, not gated** - write to these, but no tool fails a build over\n"
    "them.\n"
    "- **Maintainability Index**: aim >=20; `radon mi` reports it, nothing "
    "enforces it"
)

#: The same bound with the disclaimer removed. One sentence deleted; if this
#: stayed silent, the control above would be passing because the detector never
#: saw the number rather than because the disclaimer was honoured.
_UNDISCLAIMED_CLAIM = "- **Maintainability Index**: aim >=20"

#: A wrong number on a REAL gate, in a block that also disclaims an ungated one.
#: The disclaimer is true and belongs there; it must not travel to the sentence
#: beside it. This is the composition trap -- two rules that are each right and
#: whose interaction is not.
_DISCLAIMER_BESIDE_A_REAL_GATE = (
    "Nothing enforces the maintainability index, so treat it as guidance.\n"
    "Coverage, though, must be >= 85% before review."
)

#: Three claims in one sentence, the third about a gate this repository does not
#: have. `>=90% Jest frontend` sat immediately after `>=80% branch backend` in
#: four documents; a reader that lets a bound borrow the keyword of a clause
#: which already has a bound of its own charges 90 to `branch-coverage` and
#: reports a violation nobody can act on.
_UNATTRIBUTABLE_CLAIM = (
    "covered (>=90% line / >=80% branch backend, >=90% Jest frontend), whether"
)


def _control_bounds(text: str) -> list[_Bound]:
    """Run the claim reader over a control block.

    Args:
        text: The control's text.

    Returns:
        The bounds found in it.
    """
    return _bounds_in(_REPO_ROOT / "control.md", text)


def _violations(bounds: tuple[_Bound, ...] | list[_Bound]) -> list[str]:
    """Report which bounds disagree with what their gate's authority enforces.

    A bound on a gate with no enforced floor is a violation unless its block
    states that nothing enforces it. A bound on a gate that IS enforced must
    equal the enforced value, in either direction: stricter is not harmless.

    Args:
        bounds: The bounds to check.

    Returns:
        One human-readable line per violating bound, sorted.
    """
    enforced = _enforced_floors()
    reported: list[str] = []
    for bound in bounds:
        floor = enforced[bound.gate]
        if floor is None:
            if not bound.disclaimed:
                reported.append(
                    f"{bound.where()}: states {bound.value} for "
                    f"{bound.gate}, which this repository does not gate at all "
                    f"({_GATES_BY_NAME[bound.gate].authority.name} enforces no "
                    f"floor on it) -- {bound.text!r}"
                )
        elif bound.value != floor:
            reported.append(
                f"{bound.where()}: states {bound.value} for {bound.gate}; "
                f"{_GATES_BY_NAME[bound.gate].authority.name} enforces {floor} "
                f"-- {bound.text!r}"
            )
    return sorted(reported)


def test_the_reader_finds_a_xenon_grade_stricter_than_the_gate() -> None:
    """The positive control for the first claim this issue removed.

    `xenon A-grade` is a real grade of a real tool, five complexity points
    tighter than the flag `scripts/complexity.sh` passes. If the reader stops
    seeing it, the corpus assertions below run over a set that no longer
    contains the thing they were written for.
    """
    bounds = _control_bounds(_XENON_A_CLAIM)
    complexity = [b for b in bounds if b.gate == "cyclomatic-complexity"]

    assert len(complexity) == 1, (
        f"the reader found {len(complexity)} complexity bounds in "
        f"{_XENON_A_CLAIM!r}; it should find exactly the one grade."
    )
    assert complexity[0].value == Decimal(5), (
        f"the reader valued `xenon A-grade` at {complexity[0].value}; radon "
        "ranks cyclomatic complexity 1-5 at grade A."
    )
    assert _violations(bounds), (
        "`xenon A-grade` was not reported as a violation, yet "
        f"scripts/complexity.sh enforces {_enforced_complexity_ceiling()}."
    )


def test_the_reader_finds_a_branch_floor_beside_a_correct_coverage_floor() -> None:
    """The positive control for the second claim, and for clause attribution.

    `>=90% line / >=80% branch` states two numbers about two dimensions in one
    breath. A reader that collapses them charges 80 to `coverage` and reports a
    number mismatch -- right verdict, wrong reason, and the wrong fix. This is
    trap #4: a guard comparing the wrong dimension agrees with the right one
    most of the time.
    """
    bounds = _control_bounds(_BRANCH_CLAIM)
    by_gate = {bound.gate: bound.value for bound in bounds}

    assert by_gate == {
        "coverage": Decimal(90),
        "branch-coverage": Decimal(80),
    }, (
        f"the reader attributed {by_gate} to the two halves of "
        f"{_BRANCH_CLAIM!r}; 90 belongs to the combined coverage figure and 80 "
        "to the branch dimension."
    )
    assert [v for v in _violations(bounds) if "branch-coverage" in v], (
        "`>=80% branch` was not reported, though this repository declares no "
        "branch-qualified failure threshold at all."
    )
    assert not [v for v in _violations(bounds) if "for coverage" in v], (
        "`>=90% line` was reported as a violation; 90 is exactly the "
        "fail_under pyproject.toml configures, and a detector that fires on "
        "the correct half is a detector nobody keeps."
    )


def test_the_reader_catches_a_plausible_but_wrong_number_on_a_real_gate() -> None:
    """The control that matters most: a wrong number that looks entirely normal.

    The two claims above are the bugs this issue names, and a detector tuned to
    exactly them is not a detector. `>= 85%` coverage is the shape of drift that
    would arrive next: a real gate, a real unit, a number five points off. Both
    halves are asserted, because a rule that fires on 85 and also on 90 has not
    discriminated -- it has just failed.
    """
    wrong = _control_bounds(_WRONG_COVERAGE_CLAIM)
    right = _control_bounds(_RIGHT_COVERAGE_CLAIM)

    assert [bound.value for bound in wrong] == [Decimal(85)]
    assert [bound.value for bound in right] == [Decimal(90)]
    assert _violations(wrong), (
        f"a document claiming {_WRONG_COVERAGE_CLAIM!r} was not reported, "
        f"though pyproject.toml sets fail_under = {_enforced_coverage_floor()}."
    )
    assert _violations(right) == [], (
        f"a document claiming {_RIGHT_COVERAGE_CLAIM!r} was reported as a "
        "violation, so this rule fires on the enforced number too and "
        "distinguishes nothing."
    )


def test_a_bound_on_an_ungated_dimension_is_a_violation() -> None:
    """The absence case, which no number comparison would ever reach.

    `>=80% branch` and `radon MI >= B` are not wrong by five points. They are
    claims about dimensions this repository gates on nothing at all, and a rule
    that only compares values where both sides have one passes straight over
    them. So the enforced side is `Decimal | None` and `None` is its own verdict.
    """
    floors = _enforced_floors()

    assert floors["branch-coverage"] is None, (
        "a branch-qualified failure threshold has appeared in "
        "[tool.coverage.report], so `>=80% branch` may no longer be an absence "
        f"claim: {_coverage_report_config()!r}"
    )
    assert floors["maintainability-index"] is None, (
        "`radon mi` now runs in the gating half of scripts/complexity.sh, so "
        "the maintainability index has a floor and this control is stale."
    )
    assert (
        floors["coverage"] is not None and floors["cyclomatic-complexity"] is not None
    ), (
        "the ungated verdict is being produced for every dimension, so `None` "
        "means 'the extractor found nothing' rather than 'this repository "
        "enforces nothing'. Two behaviours have collapsed into one value."
    )
    assert _violations(_control_bounds("keep radon MI >= B")), (
        "`radon MI >= B` was not reported, though scripts/complexity.sh runs "
        "`radon mi` with `|| true` and enforces no maintainability floor."
    )


def test_an_ungated_dimension_may_be_discussed_when_the_prose_says_so() -> None:
    """Both directions of the disclaimer rule, one sentence apart.

    `.claude/docs/quality-standards.md` states the maintainability aim under a
    **Guidance, not gated** heading and says outright that nothing enforces it.
    That is the honest form of the sentence, and a rule that could not tell it
    from `radon MI >= B` would have to be excepted for the document that gets it
    right -- which is how a rule stops being applied.

    The pair differs by the disclaimer alone. Without the second half, the first
    control could be silent because the number was never read.
    """
    disclaimed = _control_bounds(_DISCLAIMED_CLAIM)
    undisclaimed = _control_bounds(_UNDISCLAIMED_CLAIM)

    assert disclaimed, (
        "the disclaimed control states no bound this reader can see, so its "
        "silence proves nothing about the disclaimer."
    )
    assert _violations(disclaimed) == [], (
        f"{_violations(disclaimed)} were reported for a block that says in so "
        "many words that nothing enforces the number it states."
    )
    assert _violations(undisclaimed), (
        "the same bound with the disclaimer deleted was still not reported, so "
        "the exemption above is not what is keeping it quiet."
    )


def test_a_disclaimer_does_not_excuse_a_wrong_number_on_a_gate_that_is_real() -> None:
    """The exemption reaches the ungated dimension and stops there.

    "Nothing enforces the maintainability index" is a true sentence, and a
    document is entitled to write it beside a coverage figure. If saying it
    quietened the coverage figure too, one honest clause would license every
    wrong number in the block -- the composition trap, where two rules that are
    each correct produce a hole between them. `>= 85%` is five points off
    `fail_under` and must still be reported.
    """
    bounds = _control_bounds(_DISCLAIMER_BESIDE_A_REAL_GATE)
    coverage = [bound for bound in bounds if bound.gate == "coverage"]

    assert [bound.value for bound in coverage] == [Decimal(85)], (
        f"the reader took {[(b.gate, b.value) for b in bounds]} out of a block "
        "whose second sentence states a coverage floor of 85%."
    )
    assert coverage[0].disclaimed, (
        "the block does not read as disclaimed at all, so its silence would "
        "prove nothing about whether the exemption is confined to ungated "
        "dimensions."
    )
    assert _violations(bounds), (
        "a coverage floor of 85% went unreported because the block also said, "
        "correctly, that nothing enforces the maintainability index. The "
        "exemption is leaking across dimensions."
    )


def test_a_bound_whose_clause_names_no_gate_is_not_charged_to_a_neighbour() -> None:
    """A bound may go unattributed; it may not be charged to someone else's gate.

    This is trap #4 -- a guard comparing the wrong dimension -- in its exact
    historical form. `>=90% Jest frontend` followed `>=80% branch backend` in
    four documents. A reader that reaches into the previous clause for a keyword
    finds `branch`, reports 90 against a dimension this repository does not
    gate, and sends whoever reads it to fix the wrong sentence. Frontend
    thresholds are out of this module's scope; being out of scope has to mean
    silence, not misattribution.
    """
    bounds = _control_bounds(_UNATTRIBUTABLE_CLAIM)

    assert [(bound.gate, bound.value) for bound in bounds] == [
        ("coverage", Decimal(90)),
        ("branch-coverage", Decimal(80)),
    ], (
        f"the reader took {[(b.gate, b.value) for b in bounds]} out of a "
        "sentence stating three figures. The third names Jest, which this "
        "repository does not run, and belongs to no gate here."
    )


def test_a_reported_measurement_is_not_a_claimed_threshold() -> None:
    """The negative control that decides whether this rule survives contact.

    A detector that reads every number as a requirement reports the pytest
    failure `.claude/skills/ci-debugging/SKILL.md` quotes, the complexity a
    worked refactor achieved, and a per-function branch COUNT. Three documents
    that are correct, and a rule with three exceptions is a rule nobody runs.
    """
    assert _control_bounds(_MEASUREMENT) == [] or all(
        bound.value == Decimal(90) for bound in _control_bounds(_MEASUREMENT)
    ), (
        f"{_control_bounds(_MEASUREMENT)} were read out of a quoted pytest "
        "failure; `Total coverage: 57.14%` is what the run measured."
    )
    assert _control_bounds(_REFACTOR_RESULT) == [], (
        f"{_control_bounds(_REFACTOR_RESULT)} were read out of a worked "
        "example's result line, which reports what a refactor achieved rather "
        "than a ceiling anyone must meet."
    )
    assert _control_bounds(_BRANCH_COUNT) == [], (
        f"{_control_bounds(_BRANCH_COUNT)} were read out of a per-function "
        "branch COUNT. Only the missing `%` separates it from a coverage "
        "floor, and reading it as one makes the guidance list unwritable."
    )


def test_a_forbidden_example_may_quote_the_loosened_flag_it_forbids() -> None:
    """Both directions of the counterexample rule, one comment apart.

    `.claude/docs/troubleshooting.md` teaches "never lower a threshold" by
    SHOWING `xenon --max-absolute C  # Relaxed from B`. That is the document
    most emphatically on this module's side, and a rule that reported it would
    be excepted for it -- which is how a rule stops being applied. It is the
    same accommodation `test_verification_claims` makes when it strips
    `# pylint: disable` before reading a tool name out of an anti-bypass rule.

    The pair differs by the word `WRONG` alone. Without the second half, the
    first could be quiet because grades in fenced blocks are never read at all,
    and a document that really had loosened the gate would sail through.
    """
    assert _control_bounds(_FORBIDDEN_EXAMPLE) == [], (
        f"{_control_bounds(_FORBIDDEN_EXAMPLE)} were read out of a block whose "
        "own comment calls the flag it shows WRONG."
    )
    assert _violations(_control_bounds(_UNMARKED_EXAMPLE)), (
        "the same block with `- WRONG` deleted from its comment was still not "
        "reported, so grades inside fenced blocks are invisible and the "
        "exemption above is not what kept the first half quiet."
    )


# --- The corpus is real -------------------------------------------------------


def test_no_threshold_authority_is_a_document_in_the_scanned_corpus() -> None:
    """The two derivations read disjoint files, and that is checked, not hoped.

    This is the assertion that makes every comparison below mean something. If
    an enforced value could be read out of the same prose the claimed value came
    from, the suite would be measuring a document against itself -- the
    coincidence trap, which has hidden a real defect eight times in this session.

    Two checks, deliberately not three: neither subsumes the other, and each
    fails on cases the other waves through.

    * **It is not markdown.** `CLAUDE.md` is exactly the file that would make
      this suite measure a table against itself, and it sits at the repository
      root, outside every corpus root -- so only the suffix catches it.
    * **It is not a file the corpus roots contain, whatever its suffix.**
      `.claude/skills/de-slopify/scripts/*.sh` is a shell script living inside
      the prose corpus; its suffix would wave it through, and one widening of
      the walk turns it into prose this module still calls the enforcing side.
      The neighbourhood is every FILE under a root plus `CLAUDE.md`, which is
      why a separate "lies under no corpus root" assertion would add nothing.
    """
    neighbourhood = _corpus_neighbourhood()

    for gate in _GATES:
        authority = gate.authority
        assert authority.is_file(), (
            f"{gate.name}'s authority {authority} does not exist, so its "
            "enforced value is derived from nothing."
        )
        assert authority.suffix != ".md", (
            f"{gate.name}'s authority {authority} is a markdown document. "
            "Enforcement lives in configuration and scripts; reading a "
            "threshold out of prose and comparing it to prose proves nothing."
        )
        assert authority not in neighbourhood, (
            f"{gate.name}'s authority {authority} is one of the "
            f"{len(neighbourhood)} files the prose corpus is made of. Its "
            "suffix alone would not have caught that: the corpus roots hold "
            "scripts too, and a widening of the walk would start reading it as "
            "prose while this module went on calling it the enforcing side."
        )


def test_the_floors_in_this_module_are_floors_and_not_decoration() -> None:
    """A floor of zero is not a floor.

    Every bound below is an anti-vacuity guard: it exists to notice a scan that
    has stopped scanning. Zeroing one leaves the assertion in place, green, and
    inert -- `len(bounds) >= 0` -- which is trap #8 and the same shape of
    unfailable check this whole module was written to remove. The guards are
    therefore asserted to be capable of failing at all.
    """
    floors = {
        "_CLAIM_FLOOR": _CLAIM_FLOOR,
        "_CLAIMING_DOCUMENT_FLOOR": _CLAIMING_DOCUMENT_FLOOR,
        "_GATE_FLOOR": _GATE_FLOOR,
        "_ROOT_CLAIM_FLOOR": _ROOT_CLAIM_FLOOR,
    }

    inert = sorted(name for name, value in floors.items() if value < 1)

    assert inert == [], (
        f"{inert} are set below 1, so the assertions that cite them cannot "
        "fail. An anti-vacuity guard that admits the empty set is the defect it "
        "was written to catch."
    )


def test_the_corpus_states_thresholds_across_several_documents_and_gates() -> None:
    """The anti-vacuity assertion: this scan has something to be about.

    Trap #5 -- a corpus scan over zero hits passes forever -- is the exact
    failure mode this whole issue is about, so it is asserted rather than
    assumed. Narrowing the reader until it matches nothing would turn "every
    stated threshold is the enforced one" into a statement about the empty set.
    """
    bounds = _claimed_bounds()
    documents = {bound.document for bound in bounds}
    gates = {bound.gate for bound in bounds}

    assert len(bounds) >= _CLAIM_FLOOR, (
        f"only {len(bounds)} thresholds were read across "
        f"{len(_corpus_documents())} documents, below the floor of "
        f"{_CLAIM_FLOOR}. The reader has stopped matching how this corpus "
        "writes a number."
    )
    assert len(documents) >= _CLAIMING_DOCUMENT_FLOOR, (
        f"the thresholds come from only {sorted(str(d) for d in documents)}, "
        f"below the floor of {_CLAIMING_DOCUMENT_FLOOR} documents."
    )
    assert len(gates) >= _GATE_FLOOR, (
        f"the thresholds cover only {sorted(gates)}, below the floor of "
        f"{_GATE_FLOOR} gates. A reader that sees one dimension cannot notice "
        "a claim about another."
    )


@pytest.mark.parametrize("root", _CORPUS_ROOTS, ids=lambda path: path.name)
def test_every_corpus_root_states_a_threshold_of_its_own(root: Path) -> None:
    """No root is silently contributing nothing.

    The totals above are dominated by `.claude/`, which alone clears every one
    of them, so a root that stopped matching would be invisible in the
    aggregate. `scripts/ralph/` is two documents and one of them, `PROMPT.md`,
    is the Ralph worker's whole operating contract -- and one of the four places
    `>=80% branch` was written. `prompts/` is where the other two were.

    Args:
        root: The corpus root being verified.
    """
    documents = _documents_under(root)
    bounds = [
        bound
        for document in documents
        for bound in _bounds_in(document, document.read_text(encoding="utf-8"))
    ]

    assert documents, (
        f"{root.relative_to(_REPO_ROOT)} is listed as a corpus root but the "
        "walk finds no markdown under it."
    )
    assert len(bounds) >= _ROOT_CLAIM_FLOOR, (
        f"{root.relative_to(_REPO_ROOT)} contributed {len(documents)} documents "
        f"and {len(bounds)} thresholds, below the floor of "
        f"{_ROOT_CLAIM_FLOOR}. Its documents are being read and no threshold is "
        "found in any of them, so this root is in the corpus without being "
        "covered by it."
    )


# --- The assertions this issue exists for -------------------------------------


def test_every_threshold_the_corpus_states_is_the_one_the_repository_enforces() -> None:
    """No governing document states a number this repository does not enforce.

    In EITHER direction. `xenon A-grade` is stricter than `--max-absolute B` and
    that is not the harmless direction: an agent believing it refactors code the
    gate accepts, or reports BLOCKED on a file that passes Gate 1 cleanly -- a
    report indistinguishable from a real failure to whoever reads it.

    Every value on the left of this comparison came from `pyproject.toml`,
    `scripts/complexity.sh` or `scripts/mutation.sh`; every value on the right
    came from markdown. `test_no_threshold_authority_is_a_document_in_the_
    scanned_corpus` is what keeps that true.
    """
    assert _violations(_claimed_bounds()) == [], (
        "the documents above state thresholds this repository does not "
        "enforce. Fix the prose -- never the threshold: a number moved to make "
        "a sentence true is the defect this module exists to catch, wearing a "
        "different hat."
    )


def test_the_briefs_name_the_complexity_ceiling_that_is_actually_enforced() -> None:
    """The replacement is present, not merely the falsehood absent.

    Deleting `xenon A-grade / radon MI >= B` would satisfy every assertion above
    and leave three of the highest-traffic briefs in the repository with no idea
    what the complexity gate is. An agent whose brief names no bound picks one.

    The expected text is DERIVED from `scripts/complexity.sh`, so if the gate
    were ever loosened to `--max-absolute C` this fails rather than pinning a
    number that had stopped being true.
    """
    ceiling = _enforced_complexity_ceiling()
    grade = _MAX_ABSOLUTE.search(_COMPLEXITY_SCRIPT.read_text(encoding="utf-8"))
    assert grade is not None
    stated = re.compile(rf"--max-absolute\s+{grade.group(1)}\b|\b{ceiling}\b")

    for brief in (
        _IMPLEMENTATION_SPECIALIST,
        _PERFORMANCE_SPECIALIST,
        _WORKER_PROMPT,
    ):
        assert stated.search(brief.read_text(encoding="utf-8")), (
            f"{brief.relative_to(_REPO_ROOT)} names neither "
            f"`--max-absolute {grade.group(1)}` nor the ceiling {ceiling} it "
            "stands for. It used to say `xenon A-grade`; removing a false "
            "bound without stating the true one leaves the agent to guess."
        )


def test_the_test_specialist_states_the_coverage_rule_that_is_enforced() -> None:
    """The same, for the brief that carried the phantom branch floor.

    `ralph-test-specialist.md` is the reviewer that decides whether new code is
    "genuinely covered". It asked for `>=90% line / >=80% branch`, and the
    second half is a target that does not exist. What has to be in its place is
    the combined figure `fail_under` gates, named as combined -- otherwise the
    next reader reconstructs the split.
    """
    floor = _enforced_coverage_floor()
    text = _TEST_SPECIALIST.read_text(encoding="utf-8")

    assert re.search(rf"\b{floor}\s*%", text), (
        f"{_TEST_SPECIALIST.relative_to(_REPO_ROOT)} no longer names the {floor}% "
        "coverage figure pyproject.toml enforces."
    )
    assert re.search(r"combined|folded", text, re.IGNORECASE), (
        f"{_TEST_SPECIALIST.relative_to(_REPO_ROOT)} states a coverage figure "
        "without saying it covers line AND branch outcomes together. A bare "
        "percentage beside the word 'line' is what invited a second, "
        "non-existent branch floor beside it in the first place."
    )
