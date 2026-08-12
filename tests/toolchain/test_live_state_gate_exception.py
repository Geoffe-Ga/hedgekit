"""The Gate-1 exception for live-repository-state checks is self-verifying (#534).

`.claude/docs/principles.md` #4 states Stay Green as absolute, and for every
check whose subject is code in this tree it is. It has no answer for a check
whose subject is **live GitHub repository state**: a setting a pull request
cannot change, so a rename of it cannot be atomic with the commit that requires
the new name. PR #533 shipped with Gate 1 red for exactly that reason -- one
failing test, `test_the_container_job_is_a_required_status_check`, correctly
reporting that a renamed job did not yet gate merge -- and the exception was
argued in that PR's description alone.

An exception that lives only in prose is the shape this repository has removed
four times already (#351, #359, #401, #411): a documented gate nothing measures,
believed because it is written down. So the exception is written in
`.claude/docs/workflow.md` in a form that can disagree with reality, and this
module is what notices when it does.

WHAT IS ACTUALLY VERIFIED HERE, and nothing beyond it:

* The set of live-repository-state checks the document ENUMERATES equals the set
  DERIVED from `tests/` by :func:`_live_state_check_ids`, in both directions. A
  name the document lists that no such check answers to fails; a check in the
  tree the document does not list fails.
* The condition tags the document's bullets carry are exactly the tags the
  worked examples in this module evaluate, so a sixth condition cannot be added
  to the prose without an example that exercises it.
* The combinator the document publishes (``LS1 AND LS2 AND ...``) is parsed, and
  every worked example's verdict is checked against it. Rewriting the document
  to read `OR` turns each expects-no-exception example red, because each of them
  satisfies at least one condition.
* Exactly one document in the governing corpus states the conditions, and every
  other document that states the Stay Green rule links to the canonical section
  by its derived anchor.

WHAT IS NOT VERIFIED. The detector reads *calls inside functions*: a marker
held in a module-level constant is attributed to no function, a value merely
named in a failure message is a description rather than a read (which is what
this module's own positive control was flagged for on its first run), and a
helper in one module called from a test in another is not followed across the
module boundary. Functions are keyed by bare name, so two sharing one are
merged rather than resolved -- deliberately over-attributing, since the
alternative silently drops one of them. The cross-module hole is made loud
rather than left silent --
:func:`test_no_live_state_read_is_unattributed` fails on any module holding a
live-state read that none of its own tests reach -- but a live-state read
performed through a module-level constant and a non-subprocess HTTP client
would evade both markers. It would also be the first such check in this
repository's history to do so; today every one goes through the `gh` CLI.

The prose in the document is prose. This module does not parse English, and no
assertion here should be read as checking that the wording means what it says.
It checks the parts that were made machine-readable precisely so they could be.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTS_ROOT = _REPO_ROOT / "tests"
_DOCS_DIR = _REPO_ROOT / ".claude" / "docs"
_WORKFLOW_DOC = _DOCS_DIR / "workflow.md"
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"

#: The heading opening the canonical section. Matched exactly: the anchor every
#: other document links to is DERIVED from this string, so renaming the heading
#: must break the pointers rather than silently orphan them.
_CANONICAL_HEADING = "### 2.4 Exception: checks whose subject is live repository state"

#: Lower bound on the modules the detector must parse. `tests/` held 343 `*.py`
#: files when this was written; a floor well under that catches a corpus walk
#: that has stopped walking (trap #5 -- a scan over zero hits passes forever)
#: without failing every time a module is deleted.
_MODULE_FLOOR = 250

#: The `gh api` path template naming THIS repository. `gh` expands `{owner}` and
#: `{repo}` from the checkout, so this literal is what a live read looks like in
#: source. Matched against string constants that appear in CODE, never against
#: docstrings: `tests/e2e/conftest.py` and this module's own docstring both
#: discuss such reads in prose, and prose is not a read.
_GITHUB_REPO_API_PATH = re.compile(r"repos/\{owner\}/\{repo\}")

#: Callables that start a subprocess. A `gh` string handed to one of these is a
#: live read; the same string in a documentation list -- `tests/docs/
#: test_documented_commands.py` has one -- is not, which is why the marker is a
#: CALL and not a substring.
_PROCESS_LAUNCHERS = frozenset(
    {"run", "which", "check_output", "check_call", "call", "Popen"}
)

#: The CLI whose invocation marks a live read.
_GH_CLI = "gh"

#: A pytest node id: `tests/<path>.py::<function>`.
_NODE_ID = re.compile(r"tests/[\w./-]+\.py::\w+")

#: A condition bullet in the canonical section: `- **(LS1) Subject.** ...`.
_CONDITION_BULLET = re.compile(r"^-\s+\*\*\((?P<tag>LS\d+)\)", re.MULTILINE)

#: The published combinator, e.g. `` `LS1 AND LS2 AND LS3` ``. Read rather than
#: assumed: the worked examples below are checked against whatever operator this
#: document publishes, so a document rewritten to say `OR` fails against them
#: instead of quietly meaning something else.
_COMBINATOR_FORMULA = re.compile(r"`(LS\d+(?:\s+(?:AND|OR)\s+LS\d+)+)`")

#: Splits that formula into its operands and operators.
_FORMULA_TOKEN = re.compile(r"LS\d+|AND|OR")

#: Marks a document as stating the Stay Green rule, and therefore as owing the
#: reader a pointer at the exception to it.
_STAY_GREEN = re.compile(r"Stay Green", re.IGNORECASE)

#: How the combinator each document may publish is applied to a set of answers.
_COMBINATORS = {"AND": all, "OR": any}


@dataclass(frozen=True)
class _ModuleAnalysis:
    """What one parsed module contributes to the live-state census.

    Attributes:
        seeds: Names of functions performing a live-repository-state read.
        live_state_tests: Names of `test_*` functions that reach one, directly
            or through another function in the same module.
    """

    seeds: frozenset[str]
    live_state_tests: frozenset[str]


@dataclass(frozen=True)
class _WorkedExample:
    """One situation the documented conditions are applied to.

    Attributes:
        name: How the situation reads in a PR description.
        conditions: Which documented conditions the situation satisfies.
        exception_applies: Whether the exception covers it. Checked against the
            combinator the document publishes, never assumed.
        why: The reason a reader would give, recorded so a failure here reads
            as a disagreement about policy rather than about arithmetic.
    """

    name: str
    conditions: Mapping[str, bool]
    exception_applies: bool
    why: str


#: The situations the documented conditions are evaluated over.
#:
#: Every example expecting NO exception satisfies at least one condition. That
#: is deliberate: it is what makes a disjunctive reading of the rule fail here
#: rather than pass, and it is how the "an ordinary red test cannot claim this"
#: requirement is demonstrated rather than asserted.
_WORKED_EXAMPLES = (
    _WorkedExample(
        name="the container-tier guard during the #509 required-context rename",
        conditions={"LS1": True, "LS2": True, "LS3": True, "LS4": True, "LS5": True},
        exception_applies=True,
        why=(
            "PR #533 quoted a job name YAML had been truncating, which changed "
            "the status-check context the job reports. The commit and the "
            "protection swap cannot land together, so Gate 1 read `1 failed, "
            "6337 passed` with the guard as the one failure. The swap ran "
            "after the merge and the guard then passed against `main`."
        ),
    ),
    _WorkedExample(
        name="an ordinary failing unit test",
        conditions={"LS1": False, "LS2": False, "LS3": False, "LS4": True, "LS5": True},
        exception_applies=False,
        why=(
            "The subject is code in this tree, which a pull request can change, "
            "so LS1 fails at the first hurdle and LS2 and LS3 have nothing to "
            "be true about. That the author left it red rather than skipping "
            "it (LS4) and that it is the only failure (LS5) are exactly the "
            "two conditions an ordinary red test CAN satisfy -- and they are "
            "not enough."
        ),
    ),
    _WorkedExample(
        name="the container-tier guard, xfail-marked so Gate 1 exits 0",
        conditions={"LS1": True, "LS2": True, "LS3": True, "LS4": False, "LS5": True},
        exception_applies=False,
        why=(
            "Marking it xfail makes the run green while the repository is "
            "still misconfigured. That is the false-green defect the exception "
            "exists to prevent, so it cannot be a way of satisfying it."
        ),
    ),
    _WorkedExample(
        name="the container-tier guard red alongside two failing unit tests",
        conditions={"LS1": True, "LS2": True, "LS3": True, "LS4": True, "LS5": False},
        exception_applies=False,
        why=(
            "The exception excuses the named check and nothing standing beside "
            "it. Two unrelated red tests are two ordinary Gate 1 failures, and "
            "the run blocks on them."
        ),
    ),
    _WorkedExample(
        name="the container-tier guard red because the context was dropped for good",
        conditions={"LS1": True, "LS2": True, "LS3": False, "LS4": True, "LS5": True},
        exception_applies=False,
        why=(
            "Nothing is mid-swap. The guard is reporting a settled state in "
            "which the tier no longer gates merge, which is a defect to fix, "
            "not a window to wait out."
        ),
    ),
    _WorkedExample(
        name="a live-state check rewritten to answer `[]` when `gh` is absent",
        conditions={"LS1": True, "LS2": False, "LS3": True, "LS4": True, "LS5": True},
        exception_applies=False,
        why=(
            "A check that answers `[]` where it cannot read the setting passes "
            "in CI over nothing. Its local red is then not evidence of a swap "
            "window; it is evidence the check reports success everywhere it "
            "matters."
        ),
    ),
)


def _callee_name(node: ast.expr) -> str:
    """Name the callee of a call as written.

    Args:
        node: The `func` of an :class:`ast.Call`.

    Returns:
        The attribute or bare name called, or the empty string for anything
        else (a call through a subscript or another call).
    """
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _docstring_constant(node: ast.AST) -> ast.Constant | None:
    """Return the docstring constant a scope opens with, if it has one.

    Args:
        node: Any AST node; only modules, classes and functions can carry one.

    Returns:
        The docstring's :class:`ast.Constant`, or None.
    """
    if not isinstance(
        node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ):
        return None
    first = node.body[0] if node.body else None
    if not isinstance(first, ast.Expr):
        return None
    value = first.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value
    return None


def _docstring_ids(tree: ast.AST) -> frozenset[int]:
    """Identify every string constant that is a docstring rather than code.

    Prose is not evidence. `tests/e2e/test_tier_selection_contract.py` names its
    own endpoint in prose, and this module names it in three places; a substring
    scan would read all of them as live reads.

    Args:
        tree: A parsed module.

    Returns:
        `id()` of each docstring constant in the tree.
    """
    found = (_docstring_constant(node) for node in ast.walk(tree))
    return frozenset(id(node) for node in found if node is not None)


def _code_strings(node: ast.AST, docstrings: frozenset[int]) -> set[str]:
    """Collect the string constants a scope uses in code.

    Args:
        node: The scope to read.
        docstrings: Ids of constants that are docstrings, from
            :func:`_docstring_ids`.

    Returns:
        Every non-docstring string constant in the scope.
    """
    return {
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant)
        and isinstance(sub.value, str)
        and id(sub) not in docstrings
    }


def _call_is_a_live_read(call: ast.Call, docstrings: frozenset[int]) -> bool:
    """Report whether one call reads live GitHub repository state.

    Two independent markers, because they fail differently: handing `gh` to a
    process launcher is what every such read in this tree does today, and the
    templated API path catches a read that reaches the endpoint by some other
    route. Both require the value to be PASSED somewhere. A test that merely
    quotes the endpoint in a failure message -- this module does, twice -- is
    describing a read, not performing one.

    Args:
        call: The call to judge.
        docstrings: Ids of docstring constants to disregard.

    Returns:
        True if either marker is present among the call's arguments.
    """
    passed = _code_strings(call, docstrings)
    if _callee_name(call.func) in _PROCESS_LAUNCHERS and _GH_CLI in passed:
        return True
    return any(_GITHUB_REPO_API_PATH.search(text) for text in passed)


def _reads_live_state(node: ast.AST, docstrings: frozenset[int]) -> bool:
    """Report whether a function reads live GitHub repository state.

    Args:
        node: The function to read.
        docstrings: Ids of docstring constants to disregard.

    Returns:
        True if any call it makes is a live read.
    """
    return any(
        _call_is_a_live_read(sub, docstrings)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
    )


def _reaches(start: str, calls: Mapping[str, set[str]], seeds: frozenset[str]) -> bool:
    """Decide whether a function reaches a live-state read through its module.

    Args:
        start: The function to start from.
        calls: Function name to the names it calls.
        seeds: Functions that perform a live-state read.

    Returns:
        True if `start` is a seed or transitively calls one.
    """
    seen: set[str] = set()
    pending = [start]
    while pending:
        name = pending.pop()
        if name in seeds:
            return True
        if name in seen:
            continue
        seen.add(name)
        pending.extend(calls.get(name, set()))
    return False


def _called_names(node: ast.AST) -> set[str]:
    """Name every function a scope calls, as written.

    Args:
        node: The scope to read.

    Returns:
        The callee names, unresolved -- a bare name is all a call site gives.
    """
    return {
        _callee_name(sub.func) for sub in ast.walk(node) if isinstance(sub, ast.Call)
    }


def _analyze(source: str) -> _ModuleAnalysis:
    """Census one module's live-repository-state reads.

    Functions are keyed by BARE NAME, because a bare name is all a call site
    offers. Two functions in one module can share one -- two test classes with
    a same-named helper, say -- so their entries are MERGED rather than
    overwritten: a name is a seed if any function bearing it reads live state,
    and it calls the union of what they all call. That resolves a collision by
    over-attributing rather than by losing one side, which is the only safe
    direction here; the opposite makes a live read invisible.

    Args:
        source: The module's source text.

    Returns:
        Its seeds and the `test_*` functions reaching them.
    """
    tree = ast.parse(source)
    docstrings = _docstring_ids(tree)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    seeds = frozenset(
        node.name for node in functions if _reads_live_state(node, docstrings)
    )
    calls: dict[str, set[str]] = {}
    for node in functions:
        calls.setdefault(node.name, set()).update(_called_names(node))
    tests = frozenset(
        name
        for name in calls
        if name.startswith("test_") and _reaches(name, calls, seeds)
    )
    return _ModuleAnalysis(seeds=seeds, live_state_tests=tests)


@cache
def _corpus() -> tuple[tuple[Path, _ModuleAnalysis], ...]:
    """Parse every module under `tests/`.

    Returns:
        `(path, analysis)` for each, in sorted path order.
    """
    return tuple(
        (path, _analyze(path.read_text(encoding="utf-8")))
        for path in sorted(_TESTS_ROOT.rglob("*.py"))
    )


def _live_state_check_ids() -> set[str]:
    """Derive the live-repository-state checks this tree actually contains.

    Returns:
        Node ids, `tests/<path>.py::<function>`.
    """
    return {
        f"{path.relative_to(_REPO_ROOT).as_posix()}::{name}"
        for path, analysis in _corpus()
        for name in analysis.live_state_tests
    }


def _section(document: Path, heading: str) -> str:
    """Slice a document from one heading to the next of the same depth or above.

    Args:
        document: The document to read.
        heading: The exact heading line opening the section.

    Returns:
        The section's text, empty if the heading is absent.
    """
    text = document.read_text(encoding="utf-8")
    if heading not in text:
        return ""
    after = text.split(heading, 1)[1]
    depth = heading.split(" ", 1)[0]
    boundary = re.compile(rf"^#{{1,{len(depth)}}} ", re.MULTILINE)
    stop = boundary.search(after)
    return after[: stop.start()] if stop else after


def _canonical_section() -> str:
    """Read the canonical exception section out of the workflow document.

    Returns:
        Its text, empty if the section does not exist.
    """
    return _section(_WORKFLOW_DOC, _CANONICAL_HEADING)


def _documented_check_ids() -> set[str]:
    """Read the checks the canonical section claims the exception covers.

    Returns:
        The node ids it names.
    """
    return set(_NODE_ID.findall(_canonical_section()))


def _documented_condition_tags() -> tuple[str, ...]:
    """Read the condition tags the canonical section's bullets carry.

    Returns:
        The tags, in document order.
    """
    return tuple(_CONDITION_BULLET.findall(_canonical_section()))


def _documented_formula() -> tuple[str, tuple[str, ...]]:
    """Read the combinator the canonical section publishes.

    Returns:
        The single operator joining the conditions, and its operands.

    Raises:
        AssertionError: If no formula is published, or it mixes operators.
    """
    match = _COMBINATOR_FORMULA.search(_canonical_section())
    assert match is not None, (
        f"{_WORKFLOW_DOC} publishes no `LS1 AND LS2 ...` formula under "
        f"'{_CANONICAL_HEADING}', so how the conditions combine is prose only "
        "and the worked examples below are checked against nothing."
    )
    tokens = _FORMULA_TOKEN.findall(match.group(1))
    operators = {token for token in tokens if token in _COMBINATORS}
    assert len(operators) == 1, (
        f"the published formula `{match.group(1)}` mixes {sorted(operators)}. "
        "A rule that combines its conditions two ways cannot be applied."
    )
    return operators.pop(), tuple(t for t in tokens if t not in _COMBINATORS)


def _anchor(heading: str) -> str:
    """Derive a document's link anchor from its heading, GitHub-style.

    Args:
        heading: The heading line.

    Returns:
        The `#slug` other documents must link to.
    """
    text = heading.lstrip("#").strip().lower()
    return "#" + re.sub(r"\s+", "-", re.sub(r"[^a-z0-9\s-]", "", text).strip())


def _governing_documents() -> tuple[Path, ...]:
    """List the documents that govern how work is done in this repository.

    Returns:
        `CLAUDE.md` and every document it navigates to under `.claude/docs/`.
    """
    return (_CLAUDE_MD, *sorted(_DOCS_DIR.glob("*.md")))


def _stay_green_documents() -> tuple[Path, ...]:
    """Find the governing documents that state the Stay Green rule.

    Returns:
        Those whose text names it.
    """
    return tuple(
        path
        for path in _governing_documents()
        if _STAY_GREEN.search(path.read_text(encoding="utf-8"))
    )


def _documents_stating_the_conditions() -> tuple[Path, ...]:
    """Find the governing documents that spell the conditions out.

    Returns:
        Those carrying a condition tag.
    """
    return tuple(
        path
        for path in _governing_documents()
        if _CONDITION_BULLET.search(path.read_text(encoding="utf-8"))
    )


#: A module doing what the real guard does, used as the detector's positive
#: control. If the detector stops firing on this, every assertion derived from
#: it is passing over an empty set.
_POSITIVE_CONTROL = '''
"""A module that reads live repository state."""

import subprocess


def _rulesets():
    """Read the repository's rulesets."""
    return subprocess.run(
        ["gh", "api", "repos/{owner}/{repo}/rulesets"], check=False
    ).stdout


def test_a_ruleset_gates_main():
    """Assert on live repository configuration."""
    assert _rulesets()
'''

#: Two colliding helper names, carrying the live read in OPPOSITE definition
#: orders: `_alpha`'s live half is written first, `_beta`'s second. A census
#: keyed by bare name that keeps one node per name therefore loses exactly one
#: of them whichever way it resolves -- last write wins loses `_alpha`, first
#: write wins loses `_beta` -- and only merging keeps both.
#: `test_a_name_collision_cannot_hide_a_live_read` pins that it does.
_COLLISION_CONTROL = '''
"""Two colliding helper names, one live read in each definition order."""

import subprocess


class TestAlphaLive:
    """Holds the live half of the `_alpha` collision, defined FIRST."""

    def _alpha(self):
        """Read live repository state."""
        return subprocess.run(
            ["gh", "api", "repos/{owner}/{repo}/rulesets"], check=False
        )

    def test_alpha_live(self):
        """Assert on the live read."""
        assert self._alpha()


class TestAlphaLocal:
    """Holds the local half of the `_alpha` collision, defined SECOND."""

    def _alpha(self):
        """Read something local."""
        return "pyproject.toml"

    def test_alpha_local(self):
        """Assert on the local read."""
        assert self._alpha()


class TestBetaLocal:
    """Holds the local half of the `_beta` collision, defined FIRST."""

    def _beta(self):
        """Read something local."""
        return "pyproject.toml"

    def test_beta_local(self):
        """Assert on the local read."""
        assert self._beta()


class TestBetaLive:
    """Holds the live half of the `_beta` collision, defined SECOND."""

    def _beta(self):
        """Read live repository state."""
        return subprocess.run(
            ["gh", "api", "repos/{owner}/{repo}/branches/main/protection"],
            check=False,
        )

    def test_beta_live(self):
        """Assert on the live read."""
        assert self._beta()
'''

#: A module that talks about live reads without performing one. The prose here
#: is denser than the real module's, on purpose: a substring detector reports
#: this as a live-state check, and an AST detector does not. The second test is
#: the case this module hit on its own first run -- a failure message quoting
#: the endpoint, which is a description of a read and not a read.
_NEGATIVE_CONTROL = '''
"""Discusses `gh api repos/{owner}/{repo}/branches/main/protection` at length."""

from pathlib import Path


def _config_text():
    """Read a file this pull request can change.

    A live read would run `gh api repos/{owner}/{repo}/rulesets` here. This
    does not; it reads `pyproject.toml`, which is code in the tree.
    """
    return Path("pyproject.toml").read_text(encoding="utf-8")


def test_the_config_declares_a_build_backend():
    """Assert on a file, not on a repository setting."""
    assert "build-backend" in _config_text()


def test_a_failure_message_may_quote_the_endpoint():
    """Name the endpoint in an assertion message without querying it."""
    assert _config_text(), (
        "a live check would read repos/{owner}/{repo}/rulesets; this one "
        "read a file"
    )
'''


def test_the_detector_fires_on_a_live_repository_state_read() -> None:
    """The positive control: the detector reports a read that is really there.

    Every derived assertion in this module iterates over whatever the detector
    finds. A detector that finds nothing makes all of them pass forever, which
    is trap #5 in this repository's own list.
    """
    analysis = _analyze(_POSITIVE_CONTROL)

    assert analysis.seeds == frozenset({"_rulesets"}), (
        f"the detector found seeds {sorted(analysis.seeds)} in a module whose "
        "`_rulesets` runs `gh api repos/{owner}/{repo}/rulesets`. It is not "
        "detecting live reads, so the census below is empty by construction."
    )
    assert analysis.live_state_tests == frozenset({"test_a_ruleset_gates_main"}), (
        "the detector did not attribute the live read to the test that calls "
        f"it; it reported {sorted(analysis.live_state_tests)}. Call-graph "
        "propagation is broken, so only tests reading state inline count."
    )


def test_a_name_collision_cannot_hide_a_live_read() -> None:
    """Two functions sharing a name merge, so neither side can be dropped.

    Functions are keyed by bare name, because a bare name is all a call site
    gives. A plain `{node.name: node}` census keeps one node per name, and the
    live read would vanish from the census while the suite went on performing
    it -- the silent gap this module exists to close, reappearing inside the
    module itself. The control carries the live half of one collision first and
    of the other second, so no single-winner rule can pass this: last write
    wins loses `_alpha`, first write wins loses `_beta`.
    """
    analysis = _analyze(_COLLISION_CONTROL)

    assert analysis.seeds == frozenset({"_alpha", "_beta"}), (
        f"the detector reported seeds {sorted(analysis.seeds)}; both `_alpha` "
        "and `_beta` name a function that reads live state. A missing one "
        "means the census keeps a single node per name and drops the rest, so "
        "which read is visible depends on definition order."
    )
    assert analysis.live_state_tests == frozenset(
        {"test_alpha_live", "test_alpha_local", "test_beta_live", "test_beta_local"}
    ), (
        "merging resolves a collision by over-attributing: all four tests call "
        "a name that some function bearing it reads live state through, and a "
        "bare name cannot tell the callees apart. Naming all four costs a "
        "documentation line; naming fewer means guessing, and a wrong guess "
        f"loses a read. The detector reported "
        f"{sorted(analysis.live_state_tests)}."
    )


def test_the_detector_is_silent_on_prose_about_live_reads() -> None:
    """The negative control: talking about a read is not performing one.

    A grep-based detector reports the negative control three times over: it
    names the endpoint in a module docstring, in a function docstring, and in
    an assertion message. Reporting it would force a module that reads
    `pyproject.toml` into the documented exception list, which would make the
    exception mean something it does not -- and the assertion-message case is
    not hypothetical, it is what this module's own positive control tripped on
    before the marker was narrowed to values a call is actually handed.
    """
    analysis = _analyze(_NEGATIVE_CONTROL)

    assert analysis.seeds == frozenset(), (
        f"the detector reported {sorted(analysis.seeds)} in a module that "
        "names the API path only in docstrings and a failure message. It is "
        "matching prose, so the documented list will grow entries for tests "
        "that read files."
    )
    assert analysis.live_state_tests == frozenset()


def test_the_scanned_corpus_is_non_empty_and_covers_the_known_guard() -> None:
    """The census walks a real corpus, so the agreement check can fail.

    A corpus walk that has stopped walking -- a moved `tests/` root, a glob
    that matches nothing -- would leave the derived set empty and the document
    trivially in agreement with it.
    """
    corpus = _corpus()
    scanned = {path.relative_to(_REPO_ROOT).as_posix() for path, _ in corpus}

    assert len(corpus) >= _MODULE_FLOOR, (
        f"only {len(corpus)} modules parsed under {_TESTS_ROOT}, below the "
        f"floor of {_MODULE_FLOOR}. The walk is not reaching the test tree, so "
        "the derived census is empty for a reason that has nothing to do with "
        "the documented list."
    )
    assert "tests/e2e/test_tier_selection_contract.py" in scanned, (
        "the module holding this repository's only live-repository-state check "
        "is not in the scanned corpus at all"
    )


def test_the_tree_contains_at_least_one_live_repository_state_check() -> None:
    """The derived set is non-empty, so agreeing with the document means something.

    If this repository ever genuinely has no such check, the documented list
    must be emptied with it -- and this test is where that decision gets made
    deliberately rather than by an assertion quietly passing over nothing.
    """
    derived = _live_state_check_ids()

    assert derived, (
        "no test under `tests/` reads live repository state, by either marker "
        f"({_GITHUB_REPO_API_PATH.pattern}, or `{_GH_CLI}` handed to "
        f"{sorted(_PROCESS_LAUNCHERS)}). Either the markers no longer match "
        "how such a read is written, or the last such check was removed and "
        f"{_WORKFLOW_DOC.name}'s list should be emptied to match."
    )


def test_no_live_state_read_is_unattributed() -> None:
    """A live read no test in its module reaches is invisible to the document.

    The census attributes a read to the tests that reach it within the same
    module. A read placed in `conftest.py`, or in a helper module imported
    elsewhere, would therefore be performed by the suite and named by nothing.
    That is the composition trap -- the wiring present, the claim about it
    unreachable -- so it fails here instead of being missed.
    """
    orphaned = [
        f"{path.relative_to(_REPO_ROOT).as_posix()}: {sorted(analysis.seeds)}"
        for path, analysis in _corpus()
        if analysis.seeds and not analysis.live_state_tests
    ]

    assert orphaned == [], (
        f"{orphaned} read live repository state but no `test_*` in the same "
        "module reaches the read, so the census cannot name it and "
        f"{_WORKFLOW_DOC.name} cannot document it. Move the read into the "
        "module whose tests depend on it."
    )


def test_the_documented_checks_are_exactly_the_ones_in_the_tree() -> None:
    """The document's list and the tree's census agree, in both directions.

    This is the assertion the whole module exists for. A name in the document
    that no live-state check answers to is a rule protecting a test that does
    not exist. A live-state check the document does not name inherits an
    exception nobody wrote down -- which is the state PR #533 shipped in.
    """
    documented = _documented_check_ids()
    derived = _live_state_check_ids()

    assert documented == derived, (
        f"{_WORKFLOW_DOC}'s '{_CANONICAL_HEADING}' section and the tree "
        f"disagree.\n  documented but not a live-state check: "
        f"{sorted(documented - derived)}\n  a live-state check but not "
        f"documented: {sorted(derived - documented)}\nThe exception covers "
        "exactly the checks the document names, so either list moving alone "
        "silently widens or narrows a rule labelled non-negotiable."
    )


def test_the_documented_conditions_are_the_ones_the_examples_evaluate() -> None:
    """Every documented condition is exercised, and no example invents one.

    Restating the conditions here would make this module agree with itself. The
    tags come out of the document, so a sixth condition added to the prose
    fails until a worked example says what satisfying it looks like.
    """
    documented = _documented_condition_tags()
    _, operands = _documented_formula()

    assert documented, (
        f"no `- **(LSn) ...**` condition bullets found under "
        f"'{_CANONICAL_HEADING}' in {_WORKFLOW_DOC}. The conditions are the "
        "whole content of the exception; without them it reads as permission."
    )
    assert set(documented) == set(operands), (
        f"the document's bullets {sorted(set(documented))} and its formula "
        f"{sorted(set(operands))} name different conditions, so one of them is "
        "not being applied."
    )
    for example in _WORKED_EXAMPLES:
        assert set(example.conditions) == set(documented), (
            f"worked example '{example.name}' answers "
            f"{sorted(example.conditions)} but the document lists "
            f"{sorted(documented)}. An unevaluated condition is decoration."
        )


def test_every_worked_example_agrees_with_the_documented_combinator() -> None:
    """The conditions combine the way the document says they combine.

    The verdicts below are not compared against a hard-coded `all`; they are
    compared against the operator the document publishes. Rewriting that
    operator to `OR` turns every expects-no-exception example red here, because
    each of them satisfies at least one condition -- which is the point of
    writing them that way.
    """
    operator, _ = _documented_formula()
    combine = _COMBINATORS[operator]

    for example in _WORKED_EXAMPLES:
        satisfied = sorted(tag for tag, held in example.conditions.items() if held)
        failed = sorted(tag for tag, held in example.conditions.items() if not held)
        assert example.exception_applies == combine(example.conditions.values()), (
            f"'{example.name}': the document combines its conditions with "
            f"{operator}, which makes the exception "
            f"{'apply' if combine(example.conditions.values()) else 'not apply'} "
            f"to a situation satisfying {satisfied} and failing {failed}. The "
            f"recorded verdict is that it does "
            f"{'' if example.exception_applies else 'not '}apply, because: "
            f"{example.why}"
        )


def test_an_ordinary_failing_test_can_never_claim_the_exception() -> None:
    """The wording cannot be quoted to excuse a normal red test.

    An ordinary red test genuinely satisfies two of the conditions: the author
    left it failing rather than skipping it, and it may be the only failure in
    the run. Under a disjunctive rule that would be enough. Under the published
    conjunctive one it is not, and this is where that difference is made to
    matter rather than trusted to careful reading.
    """
    operator, _ = _documented_formula()
    ordinary = next(
        example
        for example in _WORKED_EXAMPLES
        if example.name == "an ordinary failing unit test"
    )
    satisfied = [tag for tag, held in ordinary.conditions.items() if held]

    assert operator == "AND", (
        f"{_WORKFLOW_DOC} combines the conditions with {operator}. An ordinary "
        f"failing test satisfies {sorted(satisfied)} on its own, so any "
        "disjunctive reading exempts every red test in the suite."
    )
    assert satisfied, (
        "the ordinary-failing-test example satisfies no condition at all, so "
        "it demonstrates nothing about conjunction: a rule combined with OR "
        "would reject it too."
    )
    assert not _COMBINATORS[operator](ordinary.conditions.values())


def test_the_conditions_are_written_down_exactly_once() -> None:
    """One canonical statement, per this repository's DRY principle.

    Two copies of a rule labelled non-negotiable is how they come to differ.
    Derived rather than asserted about `workflow.md` alone: a copy pasted into
    `principles.md` or `CLAUDE.md` fails here, whichever document it lands in.
    """
    stating = _documents_stating_the_conditions()

    assert stating == (_WORKFLOW_DOC,), (
        "the exception's conditions are spelled out in "
        f"{[str(p.relative_to(_REPO_ROOT)) for p in stating]}. Exactly one "
        f"document may state them ({_WORKFLOW_DOC.relative_to(_REPO_ROOT)}); "
        "the rest must link to it."
    )


def test_every_stay_green_document_points_at_the_canonical_section() -> None:
    """A reader who meets the absolute rule is told where its one exception is.

    The failure this guards against is specific and has already happened once:
    an author reads "non-negotiable, without exception", sees Gate 1 red for a
    reason no commit can fix, and reaches for `pytest.skip` to comply. The
    anchor is derived from the heading, so renaming the section breaks these
    pointers rather than orphaning them.
    """
    documents = _stay_green_documents()
    anchor = _anchor(_CANONICAL_HEADING)

    assert len(documents) >= 2, (
        f"only {[str(p.relative_to(_REPO_ROOT)) for p in documents]} state the "
        "Stay Green rule. With fewer than two documents the cross-reference "
        "assertion below checks nothing."
    )
    assert _WORKFLOW_DOC in documents, (
        f"{_WORKFLOW_DOC} does not state the Stay Green rule it is supposed to "
        "hold the canonical exception to"
    )
    missing = [
        str(path.relative_to(_REPO_ROOT))
        for path in documents
        if path != _WORKFLOW_DOC
        and f"workflow.md{anchor}" not in path.read_text(encoding="utf-8")
    ]

    assert missing == [], (
        f"{missing} state the Stay Green rule without linking to "
        f"`workflow.md{anchor}`. Either the pointer was dropped or the "
        f"canonical heading was renamed without updating it."
    )


def test_the_canonical_section_exists_where_the_pointers_send_readers() -> None:
    """The section the anchors point at is really in the workflow document.

    :func:`_section` returns empty text for a missing heading, and every
    document-derived assertion above reads from it. Without this, renaming the
    heading would empty the documented list, the tags and the formula at once
    -- and the formula's own assertion would be the only thing to notice.
    """
    section = _canonical_section()

    assert section.strip(), (
        f"{_WORKFLOW_DOC} has no '{_CANONICAL_HEADING}' section, so the "
        "exception this module verifies is not written anywhere and the "
        "pointers to it lead nowhere."
    )


def test_the_documented_checks_carry_the_reason_they_cannot_be_green() -> None:
    """Each named check is a real, reachable test that really reads live state.

    Set equality with the census already implies existence; this fails with the
    file and function named instead of with two set differences, because
    "documented but not a live-state check" has two very different causes -- a
    typo, and a check that quietly stopped reading live state.
    """
    derived = _live_state_check_ids()
    unreal = [node_id for node_id in _documented_check_ids() if node_id not in derived]

    assert unreal == [], (
        f"{unreal} are named as live-repository-state checks by "
        f"{_WORKFLOW_DOC.name}, but the census does not find them. Either the "
        "node id is wrong, or the test no longer reads repository state and "
        "has no business claiming a Gate 1 exception."
    )


def _document_names(paths: Iterable[Path]) -> list[str]:
    """Render document paths for a failure message.

    Args:
        paths: The documents to name.

    Returns:
        Repo-relative paths, sorted.
    """
    return sorted(str(path.relative_to(_REPO_ROOT)) for path in paths)


def test_the_governing_document_corpus_is_non_empty() -> None:
    """The DRY and pointer scans read a real set of documents.

    Both scans above iterate over `_governing_documents()`. A glob that stops
    matching would make "exactly one document states the conditions" fail
    loudly, but "every other document points at it" pass over nothing.
    """
    documents = _governing_documents()

    assert len(documents) >= 5, (
        f"only {_document_names(documents)} found as governing documents; "
        f"expected CLAUDE.md and the files under {_DOCS_DIR}"
    )
    assert _CLAUDE_MD in documents
    assert _WORKFLOW_DOC in documents
