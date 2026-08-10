"""A guard against null wiring at the composition root (issue #470, epic #465).

Four of epic #455's launch blockers are one defect wearing four coats: a
component with thorough unit coverage, connected at the production composition
root to a null or empty collaborator.

    #441  RiskKernel(..., kill_integration=None)      `windbreak kill` cannot
                                                       stop the PAPER loop
    #444  AlertDispatcher(sinks=[])                    no configured sink ever
                                                       receives an alert
    #439  resolutions={}                               every metric is
                                                       permanently UNDEFINED
    #438  cassette mode as shipped                     the loop can only abstain

Every component underneath is correct in isolation, which is exactly why no
unit test sees any of this. The defect lives in the wiring, and until now the
wiring had no test.

WHAT THIS SCANNER IS, AND IS NOT

It is deliberately narrow in two dimensions, because a broad "no None kwargs
anywhere" rule would drown in legitimate absence-of-value and be switched off
within a week:

* **Scope** -- only the composition roots in :data:`_COMPOSITION_ROOTS`. Those
  are the modules that assemble the shipped process graph. A ``None`` inside a
  value object (``GatewayResult(ack=None)``, ``ForecastRecord(citations=())``)
  means "this datum is absent", which is a different and legitimate thing.
* **Parameter** -- only the names in :data:`_WATCHED_COLLABORATORS`, which are
  behaviour-providing dependencies. Null-wiring one silently disables a
  capability rather than recording an absence.

WHY A REGISTRY, AND WHY IT EXPIRES

Two of the three findings this scanner reports today are the open defects #441
and #444; the third is a deliberate, documented survival fallback. All three
are registered in :data:`_REGISTERED` with a reason and, where the entry is
debt, the issue that will remove it.

A registry that only ever grows is a rubber stamp, so
:func:`test_every_registration_still_matches_a_real_site` fails on a *stale*
entry. When #441 lands and the ``kill_integration=None`` disappears, this suite
goes red until its registration is deleted. The registry cannot outlive the
debt it documents.

WHAT THE EXPIRY DOES NOT CATCH

Registrations are keyed on ``(module, call, parameter)`` rather than line
number, because line-keyed entries go stale on any edit above them and train
reviewers to re-bless the registry without reading it. The cost is that a site
which *survives* a fix with inverted intent keeps matching, so its stated
reason can rot while the key stays live.

That is not hypothetical: #444 was fixed while this PR was open. The dispatcher
became an injected parameter, and ``AlertDispatcher(sinks=[])`` survived in
`scheduler/loop.py` as the fallback for when none is supplied -- the same shape
as `main.py`'s, and no longer debt. The key still matched, so nothing went red;
only reading the diff caught that the reason had inverted. Stated here so the
guard is trusted for what it does -- finding *unregistered* null wiring -- and
not for auditing the prose of entries that are already registered.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

#: Repo root, derived from this file's own location
#: (`<root>/tests/architecture/test_composition_root_wiring.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The modules that assemble the shipped process graph. `main.py` is the CLI's
#: composition root for all four processes; `scheduler/loop.py` is the PAPER
#: loop's. Adding a third composition root to the codebase should add it here.
_COMPOSITION_ROOTS = (
    "windbreak/main.py",
    "windbreak/scheduler/loop.py",
)

#: Parameter names that name a *collaborator* -- something that does work --
#: rather than a datum. Binding one of these to a null literal disables a
#: capability silently, which is the #441/#444/#439 defect shape.
#:
#: Every name here is a real parameter somewhere in `windbreak/`, enforced by
#: `test_every_watched_collaborator_is_a_real_parameter_name`, so this list
#: cannot quietly fill up with decoration that can never match.
_WATCHED_COLLABORATORS = frozenset(
    {
        "alert_dispatcher",
        "connector",
        "dispatcher",
        "exchange",
        "expectation_source",
        "gate_plan_store",
        "kill_integration",
        "ledger_writer",
        "on_beat",
        "read_models_source",
        "reconciliation_source",
        "resolutions",
        "sinks",
        "status_source",
        "submitter",
        "supervisor",
        "verifier",
        "wal",
    }
)


@dataclass(frozen=True)
class NullWiring:
    """One call site binding a watched collaborator to a null literal.

    Attributes:
        module: Repo-relative posix path of the module holding the call.
        line: 1-based line number of the call.
        call: The callee's name, e.g. ``RiskKernel`` or ``AlertDispatcher``.
        parameter: The bound keyword-argument name.
        literal: The null literal's source form: ``None``, ``[]``, ``{}``, ``()``.
    """

    module: str
    line: int
    call: str
    parameter: str
    literal: str

    @property
    def key(self) -> tuple[str, str, str]:
        """Return this site's registry key.

        Deliberately excludes the line number: registrations keyed by line go
        stale on any edit above them, which trains reviewers to re-bless the
        registry without reading it.

        Returns:
            The ``(module, call, parameter)`` triple.
        """
        return (self.module, self.call, self.parameter)

    def describe(self) -> str:
        """Render this site for a failure message.

        Returns:
            A ``file:line`` reference with the offending binding.
        """
        return f"{self.module}:{self.line} {self.call}({self.parameter}={self.literal})"


@dataclass(frozen=True)
class Registration:
    """Why a known null-wiring site is allowed to exist right now.

    Attributes:
        reason: Why this binding is acceptable, or what is broken because of it.
        issue: The issue that will remove it, or ``None`` when the binding is a
            deliberate design choice rather than debt.
    """

    reason: str
    issue: int | None


#: Every null-wiring site the composition roots contain today.
#:
#: Two are open debt and one is a design choice. Nothing else may exist without
#: being added here, and nothing may stay here after it stops existing.
_REGISTERED: dict[tuple[str, str, str], Registration] = {
    ("windbreak/main.py", "AlertDispatcher", "sinks"): Registration(
        reason=(
            "Deliberate log-only fallback, not debt. `run_loop` builds this "
            "dispatcher ONLY when no supervisor was supplied, so beat "
            "supervision never depends on a caller remembering to pass one "
            "(#443/#447). The configured root is `_build_alert_dispatcher`, "
            "which turns `alerts.sinks` into live channels."
        ),
        issue=None,
    ),
    ("windbreak/scheduler/loop.py", "AlertDispatcher", "sinks"): Registration(
        reason=(
            "Deliberate log-only fallback since #444 was fixed, not debt. The "
            "dispatcher is now INJECTED into the PAPER loop, and this "
            "construction is reached only when no dispatcher was supplied. "
            "#444's defect was the converse: this was the only behaviour "
            "available, so a deployment that had configured a real sink still "
            "reached nothing but the log-only fallback. The scheduler must not "
            "read `config.alerts` itself -- resolving a sink's `*_env` "
            "destination reads the real environment, and "
            "`windbreak.main._build_alert_dispatcher` is deliberately the one "
            "place that happens."
        ),
        issue=None,
    ),
    ("windbreak/scheduler/loop.py", "RiskKernel", "kill_integration"): Registration(
        reason=(
            "OPEN DEFECT. The always-on PAPER loop builds its kernel with no "
            "kill integration, so `windbreak kill` cannot stop it approving. "
            "Remove this registration when the wiring is fixed."
        ),
        issue=441,
    ),
}


def _null_literal(node: ast.expr) -> str | None:
    """Classify an expression as a null literal, if it is one.

    Args:
        node: The keyword argument's value expression.

    Returns:
        The literal's source form, or ``None`` if the expression is not one.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    if isinstance(node, ast.List) and not node.elts:
        return "[]"
    if isinstance(node, ast.Dict) and not node.keys:
        return "{}"
    if isinstance(node, ast.Tuple) and not node.elts:
        return "()"
    return _empty_constructor_call(node)


#: Builtin constructors whose no-argument call is an empty collection. Raised in
#: review of PR #477: without these, `AlertDispatcher(sinks=list())` disables the
#: same capability as `sinks=[]` while evading a literals-only scanner.
_EMPTY_CONSTRUCTORS = frozenset({"dict", "frozenset", "list", "set", "tuple"})


def _empty_constructor_call(node: ast.expr) -> str | None:
    """Classify a no-argument builtin collection call as a null literal.

    Args:
        node: The keyword argument's value expression.

    Returns:
        The call's source form, e.g. ``list()``, or ``None`` if it is not one.
    """
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return None
    if not isinstance(node.func, ast.Name) or node.func.id not in _EMPTY_CONSTRUCTORS:
        return None
    return f"{node.func.id}()"


def _callee_name(node: ast.Call) -> str:
    """Return a readable name for a call's callee.

    Args:
        node: The call node.

    Returns:
        The callee's simple name, or ``"<expr>"`` for a computed callee.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return "<expr>"


def scan_source(source: str, *, module: str) -> list[NullWiring]:
    """Find every watched collaborator bound to a null literal in ``source``.

    Args:
        source: Python source text to scan.
        module: Repo-relative path recorded on each finding.

    Returns:
        Findings in source order.
    """
    tree = ast.parse(source)
    findings: list[NullWiring] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        findings.extend(_scan_call(node, module=module))
    return sorted(findings, key=lambda finding: (finding.line, finding.parameter))


def _scan_call(node: ast.Call, *, module: str) -> list[NullWiring]:
    """Find watched null bindings on a single call node.

    Args:
        node: The call node to inspect.
        module: Repo-relative path recorded on each finding.

    Returns:
        Findings for this call, possibly empty.
    """
    findings: list[NullWiring] = []
    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg not in _WATCHED_COLLABORATORS:
            continue
        literal = _null_literal(keyword.value)
        if literal is None:
            continue
        findings.append(
            NullWiring(
                module=module,
                line=node.lineno,
                call=_callee_name(node),
                parameter=keyword.arg,
                literal=literal,
            )
        )
    return findings


def scan_composition_roots() -> list[NullWiring]:
    """Scan every composition root for watched null wiring.

    Returns:
        Every finding across :data:`_COMPOSITION_ROOTS`, module order preserved.
    """
    findings: list[NullWiring] = []
    for module in _COMPOSITION_ROOTS:
        source = (_REPO_ROOT / module).read_text(encoding="utf-8")
        findings.extend(scan_source(source, module=module))
    return findings


def _format_unregistered(findings: list[NullWiring]) -> str:
    """Render unregistered findings into an actionable failure message.

    Args:
        findings: The unregistered findings.

    Returns:
        A multi-line message naming each site and what to do about it.
    """
    lines = [
        "Unregistered null wiring at a composition root.",
        "",
        "A collaborator wired to a null literal is silently disabled: the",
        "component is still fully unit-tested, it is simply never reached.",
        "That is the #441 / #444 / #439 defect shape.",
        "",
    ]
    lines.extend(f"  {finding.describe()}" for finding in findings)
    lines.extend(
        [
            "",
            "Fix the wiring, or -- if the binding is genuinely correct --",
            "add it to _REGISTERED in this file with a reason a reviewer can",
            "check, and an issue number if it is debt.",
        ]
    )
    return "\n".join(lines)


def test_no_unregistered_null_wiring_in_the_composition_roots() -> None:
    """No composition root wires a watched collaborator to a null literal.

    The guard itself. Anything new fails here, naming the file, line and
    parameter, so the next lane fixes it in minutes rather than discovering it
    in production six weeks later.
    """
    unregistered = [
        finding
        for finding in scan_composition_roots()
        if finding.key not in _REGISTERED
    ]

    assert not unregistered, _format_unregistered(unregistered)


def test_every_registration_still_matches_a_real_site() -> None:
    """The registry expires: a stale entry fails rather than lingering.

    This is what stops :data:`_REGISTERED` becoming a rubber stamp. When #441's
    ``kill_integration=None`` is fixed, its registration no longer matches any
    site and this test goes red until the entry is deleted -- so the file can
    never claim debt that has already been paid.
    """
    live_keys = {finding.key for finding in scan_composition_roots()}

    stale = sorted(key for key in _REGISTERED if key not in live_keys)

    assert not stale, (
        "These registrations no longer match any site, so the debt they "
        f"document is gone. Delete them from _REGISTERED: {stale}"
    )


def test_the_known_open_defects_are_still_detected() -> None:
    """The scanner actually finds #441 and #444 in the real tree.

    Without this, a scanner that silently stopped matching -- a renamed
    parameter, a refactor into a helper, a bug in the AST walk -- would report
    a clean tree and be believed. It pins that the guard is looking at
    something real, not at nothing.
    """
    findings = {finding.key for finding in scan_composition_roots()}

    assert ("windbreak/scheduler/loop.py", "RiskKernel", "kill_integration") in findings
    assert ("windbreak/scheduler/loop.py", "AlertDispatcher", "sinks") in findings


def test_every_watched_collaborator_is_a_real_parameter_name() -> None:
    """Every watched name is a parameter that actually exists in the package.

    A watch list is forward-looking by nature, but an entry that can never
    match any signature is decoration -- the failure mode issue #411 deleted a
    documented gate row for. Requiring each name to exist somewhere in
    `windbreak/` keeps the list honest without pinning it to today's call sites.
    """
    parameter_names = _package_parameter_names()

    unknown = sorted(_WATCHED_COLLABORATORS - parameter_names)

    assert not unknown, (
        "These watched collaborator names are not parameters anywhere in "
        f"windbreak/, so they can never match: {unknown}"
    )


def test_the_watch_list_covers_the_defect_class_it_was_built_for() -> None:
    """The three parameters behind #441, #444 and #439 are watched.

    Pins the guard's purpose against a future edit that trims the watch list
    until it no longer covers the defects that motivated it.
    """
    for parameter in ("kill_integration", "sinks", "resolutions"):
        assert parameter in _WATCHED_COLLABORATORS


def _package_parameter_names() -> set[str]:
    """Collect every parameter name declared anywhere in the package.

    Returns:
        The set of parameter names across all of `windbreak/`.
    """
    names: set[str] = set()
    for path in sorted((_REPO_ROOT / "windbreak").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                arguments = node.args
                names.update(
                    argument.arg
                    for argument in (
                        *arguments.posonlyargs,
                        *arguments.args,
                        *arguments.kwonlyargs,
                    )
                )
    return names


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("RiskKernel(writer, kill_integration=None)", 1),
        ("AlertDispatcher(sinks=[], ledger_writer=LoggingLedgerWriter())", 1),
        ("Fold(resolutions={})", 1),
        ("Thing(reconciliation_source=())", 1),
        ("build_deps(kill_integration=None)", 1),
        ("AlertDispatcher(sinks=list())", 1),
        ("Fold(resolutions=dict())", 1),
        ("Thing(reconciliation_source=tuple())", 1),
    ],
)
def test_scanner_detects_seeded_null_wiring(source: str, expected: int) -> None:
    """Seeded null wirings are detected, including at factory-function calls.

    The last case matters: a composition root can hand a null collaborator to a
    lowercase factory just as easily as to a constructor, so the scanner keys
    on the parameter name rather than on the callee looking class-like.

    Args:
        source: Seeded source text.
        expected: How many findings it should produce.
    """
    assert len(scan_source(source, module="seeded.py")) == expected


@pytest.mark.parametrize(
    "source",
    [
        "GatewayResult(ack=None, verify_result=None)",
        "ForecastRecord(citations=(), model_votes=())",
        "AckGatedOutcome(token=None, pending_ack=None)",
        "LiveHttpTransport(headers={})",
        "DashboardStatus(last_heartbeat=None)",
        "SelectorDecision(intents=())",
        "AlertDispatcher(sinks=configured_sinks)",
        "RiskKernel(writer, kill_integration=integration)",
        "AlertDispatcher(sinks=list(configured))",
        "Fold(resolutions=dict(loaded))",
    ],
)
def test_scanner_ignores_absent_values_and_real_wiring(source: str) -> None:
    """Absent *data* and genuinely wired collaborators are both left alone.

    An over-eager guard is worse than none: it gets muted, and takes the real
    signal with it. These are the shapes that must never be flagged -- value
    objects recording absence, and collaborators bound to actual objects.

    Args:
        source: Seeded source text that must produce no findings.
    """
    assert scan_source(source, module="seeded.py") == []
