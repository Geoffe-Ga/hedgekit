"""Failing-first tests for scripts/lint_no_floats.py (issue #12, SPEC S17.3).

The AST float-lint is the enforcement mechanism behind SPEC S6.1's "no
floats in the money path" rule: `windbreak/numeric`, `windbreak/ledger`,
`windbreak/riskkernel`, `windbreak/connector`, `windbreak/screener`,
`windbreak/forecast`, `windbreak/tokens`, `windbreak/selector`,
`windbreak/scheduler`, `windbreak/evaluation`, and `windbreak/reports` must
never contain a float literal, a `float` annotation (including forward-ref
string annotations), a true-division operator, or a `float(...)` cast. This
module loads the script directly by path
(`importlib.util.spec_from_file_location`) because it lives outside the
`windbreak` package -- it is a repo-maintenance tool, not shipped code. The
script does not exist yet: that missing-file state *is* the RED milestone
for issue #12.

Issue #16 extends the money-path denylist with `windbreak/connector` (the
exchange-facing numeric types: prices, quantities, balances) and
`windbreak/screener` (jurisdiction/eligibility decisions derived from those
same values); issue #22 then appends `windbreak/forecast` (the pipeline's
probability/money-bearing record fields); issue #31 appends `windbreak/tokens`
(the shared approval-token package's money-bearing claims fields); issue #43
appends `windbreak/selector` (the pure Trade Selector's fixed-point
price/edge/sizing paths); issue #48 appends `windbreak/scheduler` (the always-on
PAPER loop's scaled-integer equity/floor sampling); issue #187 appends
`windbreak/evaluation` and `windbreak/reports` (the evaluation harness and
reporting surface, whose money/probability aggregates are scaled-integer).
`EXPECTED_DENYLISTED_PACKAGES` below is updated to the eleven entries the
implementations must append to the
script's own `DENYLISTED_PACKAGES`; until each append lands,
`test_denylisted_packages_constant` fails on a tuple mismatch -- the
expected Gate 1 RED state for each issue (layered on top of issue #12's own
missing-script RED state, which resolves independently).

Pinned violation codes (the implementer must emit exactly these):
    FLOAT-001  float literal            (e.g. `x = 1.5`)
    FLOAT-002  float annotation         (arg / return / AnnAssign / string)
    FLOAT-003  true division            (`/` or `/=`; `//` and divmod() are fine)
    FLOAT-004  float(...) cast          (including `builtins.float(...)`)
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import types

REPO_ROOT = Path(__file__).resolve().parents[2]
LINT_SCRIPT_PATH = REPO_ROOT / "scripts" / "lint_no_floats.py"

#: Mirrors the script's own DENYLISTED_PACKAGES constant, for a direct
#: cross-check once the module can be loaded. Issue #16 appends
#: `windbreak/connector` and `windbreak/screener` to the original three
#: money-path packages from issue #12. Issue #22 appends `windbreak/forecast`
#: (the pipeline's probability/money-bearing record fields), bringing the
#: total to six. Issue #31 appends `windbreak/tokens` (the shared approval-token
#: package, whose claims carry money-bearing scaled-integer fields), bringing
#: the total to seven. Issue #43 appends `windbreak/selector` (the pure Trade
#: Selector, whose price/edge/sizing paths are fixed-point per SPEC S9.1),
#: bringing the total to eight. Issue #48 appends `windbreak/scheduler` (the
#: always-on PAPER loop, whose equity/floor sampling is scaled-integer money),
#: bringing the total to nine. Issue #187 appends `windbreak/evaluation` and
#: `windbreak/reports` (the evaluation harness and reporting surface, whose
#: money/probability aggregates stay scaled-integer), bringing the total to
#: eleven.
EXPECTED_DENYLISTED_PACKAGES = (
    "windbreak/numeric",
    "windbreak/ledger",
    "windbreak/riskkernel",
    "windbreak/connector",
    "windbreak/screener",
    "windbreak/forecast",
    "windbreak/tokens",
    "windbreak/selector",
    "windbreak/scheduler",
    "windbreak/evaluation",
    "windbreak/reports",
)


def _load_lint_module() -> types.ModuleType:
    """Load scripts/lint_no_floats.py by file path and execute it.

    Raises FileNotFoundError today, since the script has not been
    written yet -- that is the expected RED failure for issue #12.
    """
    spec = importlib.util.spec_from_file_location("lint_no_floats", LINT_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lint_module() -> types.ModuleType:
    """Provide the freshly loaded lint script module for each test."""
    return _load_lint_module()


def test_denylisted_packages_constant(lint_module: types.ModuleType) -> None:
    """The script's denylist must cover exactly the eleven money-path packages."""
    assert lint_module.DENYLISTED_PACKAGES == EXPECTED_DENYLISTED_PACKAGES


# --- Detection: float literal -------------------------------------------------


def test_detects_float_literal(lint_module: types.ModuleType) -> None:
    """A float literal (`value = 1.5`) is reported once as FLOAT-001."""
    violations = lint_module.collect_violations("value = 1.5\n", "example.py")

    assert len(violations) == 1
    assert violations[0].line == 1
    assert violations[0].code == "FLOAT-001"


# --- Detection: float annotations ---------------------------------------------


def test_detects_float_argument_annotation(lint_module: types.ModuleType) -> None:
    """A `float` parameter annotation is reported as FLOAT-002."""
    source = "def f(x: float) -> None:\n    return None\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 1
    assert violations[0].code == "FLOAT-002"


def test_detects_float_return_annotation(lint_module: types.ModuleType) -> None:
    """A `-> float` return annotation is reported as FLOAT-002."""
    source = "def g() -> float:\n    return 0\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 1
    assert violations[0].code == "FLOAT-002"


def test_detects_float_ann_assign(lint_module: types.ModuleType) -> None:
    """A `y: float = 0` annotated assignment is reported as FLOAT-002."""
    source = "y: float = 0\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 1
    assert violations[0].code == "FLOAT-002"


def test_detects_string_literal_float_annotation(lint_module: types.ModuleType) -> None:
    """Forward-ref style `x: "float"` must be caught, not just bare `float`."""
    source = "def h(x: 'float') -> None:\n    return None\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 1
    assert violations[0].code == "FLOAT-002"


# --- Detection: true division --------------------------------------------------


def test_detects_true_division_but_not_floor_division(
    lint_module: types.ModuleType,
) -> None:
    """True division `/` is FLOAT-003 while floor division `//` is allowed."""
    source = "a = 1 // 2\nb = 4 / 2\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 2
    assert violations[0].code == "FLOAT-003"


def test_does_not_flag_divmod(lint_module: types.ModuleType) -> None:
    """`divmod(7, 2)` is integer-exact and yields no violation."""
    source = "q, r = divmod(7, 2)\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


def test_detects_augmented_true_division(lint_module: types.ModuleType) -> None:
    """Augmented true division `a /= 2` is reported as FLOAT-003."""
    source = "a = 4\na /= 2\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 2
    assert violations[0].code == "FLOAT-003"


def test_does_not_flag_augmented_floor_division(lint_module: types.ModuleType) -> None:
    """Augmented floor division `a //= 2` yields no violation."""
    source = "a = 4\na //= 2\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


# --- FLOAT-003 Path-join suppression --------------------------------------------
#
# `pathlib.Path.__truediv__` overloads `/` for path joins (`output_dir / "x"`);
# that is not "true division" in SPEC S17.3's sense and must not trip
# FLOAT-003. The suppression is conservative and scope-aware: it only silences
# a `/` (or `/=`) when at least one operand is *provably* Path-typed -- a
# direct `Path(...)` / `pathlib.Path(...)` call, a bare/qualified `Path`
# parameter annotation, or a name most recently assigned from one of those
# within the same scope. Everything else -- including a name whose Path-typing
# was overwritten by a later rebinding -- still fails closed as FLOAT-003
# (issue #81, PR #73 follow-up).


def test_does_not_flag_direct_path_call_join(lint_module: types.ModuleType) -> None:
    """`Path("a") / "b"` is a path join, not true division."""
    source = 'from pathlib import Path\nq = Path("a") / "b"\n'

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


def test_does_not_flag_qualified_pathlib_path_call_join(
    lint_module: types.ModuleType,
) -> None:
    """`pathlib.Path("a") / "b"` (qualified attribute call) is a path join."""
    source = 'import pathlib\nq = pathlib.Path("a") / "b"\n'

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


def test_does_not_flag_join_via_module_scope_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A name bound from `Path(...)` at module scope stays suppressed on join."""
    source = 'from pathlib import Path\np = Path("a")\nq = p / "b"\n'

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


def test_does_not_flag_join_via_function_scope_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A name bound from `Path(...)` inside a function body is suppressed too."""
    source = (
        "from pathlib import Path\n"
        "def f() -> None:\n"
        '    p = Path("a")\n'
        '    q = p / "b"\n'
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


def test_does_not_flag_join_via_bare_path_param_annotation(
    lint_module: types.ModuleType,
) -> None:
    """PR #73's acceptance pattern: a bare `Path`-annotated parameter joined."""
    source = (
        "from pathlib import Path\n"
        "def save(output_dir: Path) -> None:\n"
        '    target = output_dir / "name"\n'
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


def test_does_not_flag_join_via_qualified_path_param_annotation(
    lint_module: types.ModuleType,
) -> None:
    """The same acceptance pattern with a qualified `pathlib.Path` annotation."""
    source = (
        "import pathlib\n"
        "def save(output_dir: pathlib.Path) -> None:\n"
        '    target = output_dir / "name"\n'
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


def test_does_not_flag_chained_path_join(lint_module: types.ModuleType) -> None:
    """A chained join `output_dir / "a" / "b"` suppresses both nested Div nodes."""
    source = (
        "from pathlib import Path\n"
        "def save(output_dir: Path) -> None:\n"
        '    target = output_dir / "a" / "b"\n'
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


def test_does_not_flag_augmented_join_on_path_name(
    lint_module: types.ModuleType,
) -> None:
    """`p /= "sub"` on a Path-typed name is a join, not augmented division."""
    source = 'from pathlib import Path\np = Path("a")\np /= "sub"\n'

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


def test_does_not_flag_right_operand_path_join(lint_module: types.ModuleType) -> None:
    """`"a" / Path("b")` is a join via `Path.__rtruediv__`, not true division."""
    source = 'from pathlib import Path\nq = "a" / Path("b")\n'

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


def test_still_flags_numeric_true_division(lint_module: types.ModuleType) -> None:
    """Genuine numeric division (`4 / 2`) still raises exactly one FLOAT-003."""
    source = "b = 4 / 2\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 1
    assert violations[0].code == "FLOAT-003"


def test_still_flags_numeric_augmented_division(
    lint_module: types.ModuleType,
) -> None:
    """`a /= 2` on a plain int name still raises FLOAT-003."""
    source = "a = 4\na /= 2\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 2
    assert violations[0].code == "FLOAT-003"


def test_still_flags_unknown_names_with_no_path_signal(
    lint_module: types.ModuleType,
) -> None:
    """`x / y` with no Path provenance for either name still fails closed."""
    source = "x / y\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 1
    assert violations[0].code == "FLOAT-003"


def test_still_flags_after_rebinding_removes_path_typing(
    lint_module: types.ModuleType,
) -> None:
    """Rebinding a Path name to a non-Path value de-registers it for FLOAT-003."""
    source = 'from pathlib import Path\np = Path("a")\np = 5\nz = p / 2\n'

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 4
    assert violations[0].code == "FLOAT-003"


def test_suppression_is_per_expression_not_per_scope(
    lint_module: types.ModuleType,
) -> None:
    """A Path join and a genuine numeric division in the same function differ."""
    source = (
        "from pathlib import Path\n"
        "def f(d: Path) -> int:\n"
        '    x = d / "a"\n'
        "    return 6 / 2\n"
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 4
    assert violations[0].code == "FLOAT-003"


def test_float_001_unaffected_by_path_join_in_same_source(
    lint_module: types.ModuleType,
) -> None:
    """FLOAT-001 still fires on a float literal alongside a suppressed join."""
    source = (
        "from pathlib import Path\n"
        "def save(output_dir: Path) -> None:\n"
        '    target = output_dir / "n"\n'
        "x = 1.5\n"
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 4
    assert violations[0].code == "FLOAT-001"
    assert all(v.code != "FLOAT-003" for v in violations)


# --- FLOAT-003 fail-closed regressions (issue #81) ------------------------------
#
# Self-review of PR #73's Path-join suppression found it fails *open*: real
# numeric division escapes FLOAT-003 in several shadowing/rebinding shapes
# because (a) `_name_is_path` searches every enclosing scope regardless of
# shadowing, and (b) Path-typing is only de-registered on a single-Name `=`
# assignment, leaving stale Path-typing after other binding forms (for-loop
# targets, comprehension targets, tuple unpacking) or scope leaks (class
# bodies). Issue #81 requires fail-*closed*: when unsure, keep flagging. These
# tests pin that every one of these shapes still raises FLOAT-003.


def test_flags_division_when_int_param_shadows_module_path_name(
    lint_module: types.ModuleType,
) -> None:
    """An int parameter named after a module-scope Path name still divides."""
    source = (
        "from pathlib import Path\n"
        'root = Path("a")\n'
        "def scale(root: int) -> int:\n"
        "    return root / 2\n"
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 4
    assert violations[0].code == "FLOAT-003"


def test_flags_division_when_local_int_shadows_module_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A local int rebind of a module-scope Path name still divides."""
    source = (
        "from pathlib import Path\n"
        'p = Path("a")\n'
        "def f() -> int:\n"
        "    p = 6\n"
        "    return p / 2\n"
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 5
    assert violations[0].code == "FLOAT-003"


def test_flags_division_on_for_loop_target_rebinding_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A for-loop target that rebinds a Path name to a non-Path still divides."""
    source = (
        'from pathlib import Path\np = Path("a")\nfor p in range(3):\n    z = p / 2\n'
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 4
    assert violations[0].code == "FLOAT-003"


def test_flags_division_on_comprehension_target_shadowing_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A comprehension target shadowing an outer Path name still divides."""
    source = (
        'from pathlib import Path\np = Path("a")\nresult = [p / 2 for p in range(3)]\n'
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 3
    assert violations[0].code == "FLOAT-003"


def test_flags_division_after_tuple_unpack_rebinds_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A tuple-unpack rebind of a Path name still divides."""
    source = 'from pathlib import Path\np = Path("a")\np, q = 5, 6\nz = p / 2\n'

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 4
    assert violations[0].code == "FLOAT-003"


def test_flags_division_on_class_attr_path_not_leaking_to_module(
    lint_module: types.ModuleType,
) -> None:
    """A class-body Path attribute must not leak Path-typing to module scope."""
    source = 'from pathlib import Path\nclass C:\n    d = Path("a")\nz = d / 2\n'

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 4
    assert violations[0].code == "FLOAT-003"


def test_flags_division_when_lambda_param_shadows_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A lambda parameter shadowing an outer Path name still divides."""
    source = 'from pathlib import Path\nroot = Path("a")\nf = lambda root: root / 2\n'

    violations = lint_module.collect_violations(source, "example.py")

    floats = [v for v in violations if v.code == "FLOAT-003"]
    assert len(floats) == 1
    assert floats[0].line == 3


def test_flags_division_when_with_as_target_shadows_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A `with ... as` target rebinding a Path name still divides."""
    source = (
        'from pathlib import Path\np = Path("a")\nwith open("x") as p:\n    z = p / 2\n'
    )

    violations = lint_module.collect_violations(source, "example.py")

    floats = [v for v in violations if v.code == "FLOAT-003"]
    assert len(floats) == 1
    assert floats[0].line == 4


def test_flags_division_when_async_with_as_target_shadows_path_name(
    lint_module: types.ModuleType,
) -> None:
    """An `async with ... as` target rebinding a Path name still divides."""
    source = (
        "from pathlib import Path\n\n\n"
        "async def run() -> None:\n"
        '    p = Path("a")\n'
        "    async with ctx() as p:\n"
        "        z = p / 2\n"
    )

    violations = lint_module.collect_violations(source, "example.py")

    floats = [v for v in violations if v.code == "FLOAT-003"]
    assert len(floats) == 1
    assert floats[0].line == 7


def test_flags_division_when_walrus_rebinds_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A walrus assignment rebinding a Path name to an int still divides."""
    source = 'from pathlib import Path\np = Path("a")\nvalue = (p := 5)\nz = p / 2\n'

    violations = lint_module.collect_violations(source, "example.py")

    floats = [v for v in violations if v.code == "FLOAT-003"]
    assert len(floats) == 1
    assert floats[0].line == 4


def test_flags_division_when_except_as_target_shadows_path_name(
    lint_module: types.ModuleType,
) -> None:
    """An `except ... as` handler name rebinding a Path name still divides."""
    source = (
        'from pathlib import Path\np = Path("a")\ntry:\n    pass\n'
        "except Exception as p:\n    z = p / 2\n"
    )

    violations = lint_module.collect_violations(source, "example.py")

    floats = [v for v in violations if v.code == "FLOAT-003"]
    assert len(floats) == 1
    assert floats[0].line == 6


def test_flags_division_when_starred_target_rebinds_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A starred unpack target rebinding a Path name still divides."""
    source = (
        'from pathlib import Path\np = Path("a")\nfirst, *p = [1, 2, 3]\nz = p / 2\n'
    )

    violations = lint_module.collect_violations(source, "example.py")

    floats = [v for v in violations if v.code == "FLOAT-003"]
    assert len(floats) == 1
    assert floats[0].line == 4


def test_flags_division_when_match_capture_shadows_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A `match`/`case` capture pattern rebinding a Path name still divides."""
    source = (
        'from pathlib import Path\np = Path("a")\nmatch object():\n'
        "    case p:\n        z = p / 2\n"
    )

    violations = lint_module.collect_violations(source, "example.py")

    floats = [v for v in violations if v.code == "FLOAT-003"]
    assert len(floats) == 1
    assert floats[0].line == 5


def test_does_not_flag_join_via_walrus_bound_path_name(
    lint_module: types.ModuleType,
) -> None:
    """A walrus-bound Path name stays suppressed on join, then and later.

    Pins the intended (not-yet-implemented) behaviour: once the implementer
    registers walrus (`:=`) Path bindings, both the walrus expression itself
    and any later reference to the bound name join rather than divide.
    """
    source = 'from pathlib import Path\ny = (p := Path("a")) / "b"\nz = p / "c"\n'

    violations = lint_module.collect_violations(source, "example.py")

    assert not any(v.code == "FLOAT-003" for v in violations)


# --- Detection: float(...) casts -----------------------------------------------


def test_detects_float_cast(lint_module: types.ModuleType) -> None:
    """A bare `float(...)` cast is reported as FLOAT-004."""
    source = "n = float('1.5')\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 1
    assert violations[0].code == "FLOAT-004"


def test_detects_builtins_qualified_float_cast(lint_module: types.ModuleType) -> None:
    """A qualified `builtins.float(...)` cast is also reported as FLOAT-004."""
    source = "import builtins\nn = builtins.float(1)\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 1
    assert violations[0].line == 2
    assert violations[0].code == "FLOAT-004"


# --- Clean source ---------------------------------------------------------------


def test_clean_source_yields_no_violations(lint_module: types.ModuleType) -> None:
    """Int-annotated source using only floor division reports no violations."""
    source = (
        "import math\n"
        "\n"
        "\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def ratio(a: int, b: int) -> int:\n"
        "    return a // b\n"
    )

    violations = lint_module.collect_violations(source, "example.py")

    assert violations == []


# --- CLI: dual invocation modes ---------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the lint script as a subprocess, from the repo root.

    Asserts the script file exists first, so a not-yet-implemented
    script fails with an unambiguous "not yet implemented" message
    rather than an opaque non-zero exit code from the interpreter.
    """
    assert LINT_SCRIPT_PATH.is_file(), f"lint script missing: {LINT_SCRIPT_PATH}"
    return subprocess.run(
        [sys.executable, str(LINT_SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_exits_1_and_prints_violation_for_bad_file(tmp_path: Path) -> None:
    """The CLI exits 1 and prints `path:line` with FLOAT-001 for a bad file."""
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("x = 1.5\n")

    result = _run_cli(str(bad_file))

    assert result.returncode == 1
    assert "FLOAT-001" in result.stdout
    assert f"{bad_file}:1" in result.stdout


def test_cli_exits_0_for_clean_file(tmp_path: Path) -> None:
    """The CLI exits 0 with empty stdout for a float-free file."""
    clean_file = tmp_path / "clean.py"
    clean_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

    result = _run_cli(str(clean_file))

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_full_scan_mode_real_repo_denylisted_packages_are_clean() -> None:
    """No path args -> scan DENYLISTED_PACKAGES; today's stubs are float-free.

    This test remains meaningful after the RED->GREEN transition too: once
    types.py/rounding.py exist under windbreak/numeric/, this is the same
    check that guards them (and windbreak/ledger, windbreak/riskkernel) in CI.
    """
    result = _run_cli()

    assert result.returncode == 0, result.stdout


# --- In-process CLI / file-IO coverage ------------------------------------------
#
# The subprocess-based `_run_cli` tests above exercise the CLI end-to-end but
# run the script in a *separate* interpreter, so they contribute no coverage to
# this test suite's measurement. The tests below drive the same public callables
# (`main`, `_lint_file`, `_expand_targets`, `_denylisted_files`) in-process so
# CI's `--cov=.` scope (which includes `scripts/`) actually measures them, and
# so their behaviour is pinned with real assertions rather than smoke calls.


def test_main_returns_1_and_prints_each_violation(
    lint_module: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main` on a dirty file returns 1 and prints one line per violation."""
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("x = 1.5\ny = float(2)\n", encoding="utf-8")

    exit_code = lint_module.main([str(bad_file)])

    assert exit_code == 1
    out_lines = capsys.readouterr().out.splitlines()
    assert len(out_lines) == 2
    assert any("FLOAT-001" in line for line in out_lines)
    assert any("FLOAT-004" in line for line in out_lines)
    assert all(str(bad_file) in line for line in out_lines)


def test_main_returns_0_and_prints_nothing_for_clean_file(
    lint_module: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main` on a clean file returns 0 with empty stdout."""
    clean_file = tmp_path / "clean.py"
    clean_file.write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8"
    )

    exit_code = lint_module.main([str(clean_file)])

    assert exit_code == 0
    assert capsys.readouterr().out == ""


def test_main_no_args_scans_denylisted_packages(
    lint_module: types.ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main([])` falls back to scanning DENYLISTED_PACKAGES (money path clean)."""
    exit_code = lint_module.main([])

    assert exit_code == 0
    assert capsys.readouterr().out == ""


def test_main_expands_directory_argument(
    lint_module: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A directory argument is recursively expanded to its `*.py` files."""
    pkg = tmp_path / "pkg"
    nested = pkg / "nested"
    nested.mkdir(parents=True)
    (pkg / "clean.py").write_text("z: int = 3\n", encoding="utf-8")
    (nested / "dirty.py").write_text("w: float = 0\n", encoding="utf-8")

    exit_code = lint_module.main([str(pkg)])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FLOAT-002" in out
    assert "dirty.py" in out
    assert "clean.py" not in out


def test_lint_file_formats_path_line_col_code_message(
    lint_module: types.ModuleType,
    tmp_path: Path,
) -> None:
    """`_lint_file` renders `path:line:col CODE message` per violation."""
    target = tmp_path / "sample.py"
    target.write_text("value = 2.5\n", encoding="utf-8")

    lines = lint_module._lint_file(target)

    assert lines == [f"{target}:1:8 FLOAT-001 float literal is banned"]


def test_lint_file_returns_empty_for_clean_file(
    lint_module: types.ModuleType,
    tmp_path: Path,
) -> None:
    """`_lint_file` returns no lines when the file is float-free."""
    target = tmp_path / "ok.py"
    target.write_text("value: int = 2\n", encoding="utf-8")

    assert lint_module._lint_file(target) == []


def test_expand_targets_recurses_dirs_and_passes_files_through(
    lint_module: types.ModuleType,
    tmp_path: Path,
) -> None:
    """Directories expand to sorted `*.py`; explicit file paths pass through."""
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "notpython.txt").write_text("", encoding="utf-8")
    explicit = tmp_path / "explicit_file.py"
    explicit.write_text("", encoding="utf-8")

    expanded = lint_module._expand_targets([tmp_path, explicit])

    # Directory expansion is sorted and .py-only; the explicit path is appended.
    assert expanded == [
        tmp_path / "a.py",
        tmp_path / "b.py",
        explicit,
        explicit,
    ]


def test_denylisted_files_only_covers_existing_packages(
    lint_module: types.ModuleType,
) -> None:
    """Scan targets come only from denylisted packages that exist on disk."""
    files = lint_module._denylisted_files()

    repo_root = lint_module._REPO_ROOT
    numeric_dir = repo_root / "windbreak/numeric"
    # windbreak/numeric exists in this repo, so at least its files are returned.
    assert files, "expected at least the numeric package to be scanned"
    assert all(path.suffix == ".py" for path in files)
    assert any(numeric_dir in path.parents for path in files)
    # Every returned file must live under an existing denylisted package dir.
    existing_dirs = [
        repo_root / pkg
        for pkg in lint_module.DENYLISTED_PACKAGES
        if (repo_root / pkg).is_dir()
    ]
    assert all(
        any(directory in path.parents for directory in existing_dirs) for path in files
    )


def test_denylisted_files_skips_nonexistent_packages(
    lint_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A denylisted prefix with no directory on disk is silently skipped."""
    monkeypatch.setattr(
        lint_module,
        "DENYLISTED_PACKAGES",
        ("windbreak/definitely_not_a_real_package",),
    )

    assert lint_module._denylisted_files() == []


# --- Signature-annotation edge cases (branch coverage) --------------------------


def test_vararg_and_kwarg_float_annotations_are_flagged(
    lint_module: types.ModuleType,
) -> None:
    """`*args: float` and `**kwargs: float` annotations are both caught."""
    source = "def f(*args: float, **kwargs: float) -> None:\n    return None\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert len(violations) == 2
    assert {v.code for v in violations} == {"FLOAT-002"}


def test_unannotated_function_yields_no_violation(
    lint_module: types.ModuleType,
) -> None:
    """A function with no annotations at all produces nothing (skip branches)."""
    source = "def f(x, *args, **kwargs):\n    return x\n"

    assert lint_module.collect_violations(source, "example.py") == []


def test_float_named_via_subscript_call_is_not_a_cast(
    lint_module: types.ModuleType,
) -> None:
    """A call whose callee is neither Name nor Attribute is not a float cast.

    Exercises the final `return False` of `_names_float`: here the callee is a
    Subscript (`registry['float']()`), so no FLOAT-004 is emitted.
    """
    source = "registry = {}\nregistry['float']()\n"

    violations = lint_module.collect_violations(source, "example.py")

    assert all(v.code != "FLOAT-004" for v in violations)
