"""Failing-first tests pinning PATH-independent quality-gate tool resolution.

Issue #366: `./scripts/check-all.sh` reached a *different verdict depending on
the caller's shell*. Every gate sub-script invoked its tool by bare name
(`pip-audit`, `mypy`, `ruff`, ...), so which binary actually ran was decided by
whatever happened to be first on the inherited `PATH`. With an ambient shell
`PATH` on macOS, `pip-audit` resolved to `/opt/homebrew/bin/pip-audit`, whose
shebang is Homebrew's interpreter -- and pip-audit audits *the dependency set of
the interpreter that owns it*. The gate therefore reported `PYSEC-2026-2132`
(click 8.3.1) against Homebrew's site-packages, a dependency set this repo does
not control, while the project venv (click 8.4.2) was clean.

The noisy direction is the harmless one. The dangerous direction is the same
misresolution pointed the other way: an audit aimed at the wrong environment can
report **CLEAN** while the project's real dependency set is vulnerable. That is
a false negative in the one check whose entire job is catching those -- absent
or unprovable evidence reading as healthy. So this is a correctness bug in a
security gate, not a convenience fix.

Target state pinned here:

- `scripts/toolchain-env.sh` is the single tool-resolution authority. It
  resolves the shared pinned `.venv` (issue #133) and hands back ABSOLUTE tool
  paths. When a pinned venv exists, a tool is taken from it and *only* from it:
  a hostile `PATH` cannot redirect a gate, and a tool missing from the pinned
  venv is a hard veto rather than a silent fallback to an alien binary.
- With no pinned venv (how CI runs -- dependencies are installed straight into
  the runner's interpreter) resolution falls back to `PATH`, but says so out
  loud, naming the absolute path it chose, so a verdict is always attributable
  to a named environment. A tool absent everywhere still vetoes.
- `scripts/security.sh` runs pip-audit as `"<pinned python>" -m pip_audit`
  rather than as a bare console script. That makes the audited environment
  identical to the pinned interpreter *by construction* -- pip-audit audits its
  own interpreter's site-packages -- so no `PATH` arrangement can point the
  audit at some other environment.

These assertions began life as Gate 1 RED for issue #366:
`scripts/toolchain-env.sh` did not exist, and every gate script (`security.sh`
included) invoked its tools by bare name.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_TOOLCHAIN_ENV_SCRIPT = _SCRIPTS_DIR / "toolchain-env.sh"
_SECURITY_SCRIPT = _SCRIPTS_DIR / "security.sh"

#: The gate scripts `check-all.sh` dispatches through `run_check`, mapped to the
#: tools each one drives. `architecture.sh` is deliberately absent -- see
#: `test_architecture_scripts_inherited_path_is_documented` for the reason it is
#: the one documented still-inherited sub-check.
_GATE_SCRIPT_TOOLS: dict[str, tuple[str, ...]] = {
    "lint.sh": ("ruff", "python3"),
    "format.sh": ("ruff",),
    "typecheck.sh": ("mypy",),
    "security.sh": ("bandit", "pip-audit", "pre-commit"),
    "complexity.sh": ("radon", "xenon"),
    "test.sh": ("pytest", "mutmut"),
    "coverage.sh": ("pytest",),
}

#: Marker a decoy prints so a test can tell "the gate ran the PATH binary" from
#: "the gate ran the pinned binary" without depending on either being real.
_DECOY_MARKER = "DECOY"


def _bash() -> str:
    """Locate the bash interpreter used to run the shell helpers under test.

    Returns:
        Absolute path to `bash`, falling back to the conventional location.
    """
    return shutil.which("bash") or "/bin/bash"


def _make_stub(path: Path, marker: str) -> None:
    """Write an executable stub script that identifies itself when run.

    Args:
        path: Location to write the stub to; parent dirs must already exist.
        marker: Text the stub echoes, letting a caller prove which copy ran.
    """
    path.write_text(f'#!/bin/sh\necho "{marker}"\n', encoding="utf-8")
    path.chmod(0o755)


def _hostile_path(decoy_dir: Path) -> str:
    """Build a PATH whose FIRST entry is a directory full of decoy tools.

    This is the deliberately hostile caller environment issue #366 asks the
    gate to be immune to: the decoys shadow every real tool by bare name.

    Args:
        decoy_dir: Directory holding the decoy executables.

    Returns:
        A PATH string with `decoy_dir` ahead of the standard system dirs.
    """
    return os.pathsep.join([str(decoy_dir), "/usr/bin", "/bin"])


def _decoy_dir(tmp_path: Path, tools: tuple[str, ...]) -> Path:
    """Create a directory of decoy executables for the named tools.

    Args:
        tmp_path: Test-scoped temporary directory.
        tools: Tool names to shadow.

    Returns:
        The directory holding the decoys.
    """
    decoys = tmp_path / "hostile-bin"
    decoys.mkdir()
    for tool in tools:
        _make_stub(decoys / tool, f"{_DECOY_MARKER}:{tool}")
    return decoys


def _fake_repo(tmp_path: Path, *, venv_tools: tuple[str, ...] | None) -> Path:
    """Build a throwaway checkout carrying the real resolution scripts.

    The resolver anchors the shared venv to the checkout that owns the script
    (`provision-venv.sh`'s `PROJECT_ROOT`), so copying the two real scripts into
    a temp checkout exercises the genuine code path hermetically -- identically
    on a developer machine (which has a shared `.venv`) and on CI (which does
    not). Nothing here touches the real repo's `.venv`.

    Args:
        tmp_path: Test-scoped temporary directory.
        venv_tools: Tool names to place in the fake `.venv/bin`, or None to
            create no `.venv` at all (the ambient-environment branch).

    Returns:
        Root of the fake checkout.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("provision-venv.sh", "toolchain-env.sh"):
        source = _SCRIPTS_DIR / name
        if source.is_file():
            shutil.copy(source, repo / "scripts" / name)
    if venv_tools is not None:
        venv_bin = repo / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        _make_stub(venv_bin / "python", "PINNED:python")
        for tool in venv_tools:
            _make_stub(venv_bin / tool, f"PINNED:{tool}")
    return repo


def _run_resolver(
    repo: Path, args: list[str], path_value: str
) -> subprocess.CompletedProcess[str]:
    """Run the copied `toolchain-env.sh` CLI inside a fake checkout.

    Args:
        repo: Root of the fake checkout (from `_fake_repo`).
        args: CLI arguments to pass to the resolver.
        path_value: The PATH the resolver must survive.

    Returns:
        The completed process, with stdout/stderr captured as text.
    """
    return subprocess.run(
        [_bash(), str(repo / "scripts" / "toolchain-env.sh"), *args],
        cwd=str(repo),
        env={"PATH": path_value, "HOME": str(repo)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _read(script: Path) -> str:
    """Read a shell script's source text.

    Args:
        script: Path to the script.

    Returns:
        The script's contents, decoded as UTF-8.

    Raises:
        AssertionError: If the script does not exist.
    """
    assert script.is_file(), f"{script} does not exist"
    return script.read_text(encoding="utf-8")


def _bare_invocations(source: str, tool: str) -> list[str]:
    """Find lines that invoke `tool` as a bare command name.

    A line counts as a bare invocation when the tool name is the first word of
    the command (optionally behind `if`/`!`), i.e. the binary that runs is
    whatever `PATH` resolves -- the exact defect issue #366 removes. Comment
    lines and help-text lines never match, since they do not begin with the
    tool name.

    Args:
        source: Full script source.
        tool: Tool name to look for.

    Returns:
        The offending source lines, in file order.
    """
    pattern = re.compile(rf"^[ \t]*(?:if[ \t]+|![ \t]*)?{re.escape(tool)}(?=[ \t])")
    return [line for line in source.splitlines() if pattern.search(line)]


# --- 1. The resolver exists and is PATH-proof when a venv is pinned -------


def test_toolchain_env_helper_exists_and_is_executable() -> None:
    """`scripts/toolchain-env.sh` exists and can be executed as a CLI.

    The gate scripts source it, but it must also be runnable directly so an
    operator (and these tests) can ask which binary a gate *would* use without
    running the gate itself.
    """
    assert _TOOLCHAIN_ENV_SCRIPT.is_file(), (
        f"{_TOOLCHAIN_ENV_SCRIPT} does not exist -- issue #366 RED"
    )
    assert os.access(_TOOLCHAIN_ENV_SCRIPT, os.X_OK), (
        f"{_TOOLCHAIN_ENV_SCRIPT} exists but is not executable"
    )


def test_pinned_venv_wins_over_a_hostile_path_for_the_interpreter(
    tmp_path: Path,
) -> None:
    """With a pinned venv present, `--print-python` ignores a hostile PATH.

    The interpreter is the load-bearing one: `python -m pip_audit` audits the
    site-packages of whichever interpreter runs it, so if a decoy `python3` on
    PATH could win here, the security gate would audit the decoy's environment.
    """
    repo = _fake_repo(tmp_path, venv_tools=("pip-audit",))
    decoys = _decoy_dir(tmp_path, ("python", "python3", "pip-audit"))

    result = _run_resolver(repo, ["--print-python"], _hostile_path(decoys))

    assert result.returncode == 0, (
        f"--print-python failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stdout.strip() == str(repo / ".venv" / "bin" / "python"), (
        "a hostile PATH redirected the pinned interpreter: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_pinned_venv_wins_over_a_hostile_path_for_a_tool(tmp_path: Path) -> None:
    """With a pinned venv present, `--print-tool` ignores a hostile PATH.

    This is issue #366's headline symptom in miniature: a decoy `pip-audit`
    first on PATH must not be what the security gate runs.
    """
    repo = _fake_repo(tmp_path, venv_tools=("pip-audit", "bandit"))
    decoys = _decoy_dir(tmp_path, ("pip-audit", "bandit", "python3"))

    result = _run_resolver(repo, ["--print-tool", "pip-audit"], _hostile_path(decoys))

    assert result.returncode == 0, (
        f"--print-tool failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stdout.strip() == str(repo / ".venv" / "bin" / "pip-audit"), (
        "a hostile PATH redirected pip-audit away from the pinned venv: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_missing_tool_in_a_pinned_venv_vetoes_instead_of_falling_back(
    tmp_path: Path,
) -> None:
    """A tool absent from the pinned venv is a veto, never a PATH fallback.

    Falling back would reintroduce exactly the bug: the gate would quietly run
    an alien binary against an environment it knows nothing about, and a clean
    verdict from it would be unprovable. Fail closed instead.
    """
    repo = _fake_repo(tmp_path, venv_tools=("bandit",))
    decoys = _decoy_dir(tmp_path, ("pip-audit", "python3"))

    result = _run_resolver(repo, ["--print-tool", "pip-audit"], _hostile_path(decoys))

    assert result.returncode != 0, (
        "pip-audit missing from the pinned venv did not veto: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert str(decoys) not in result.stdout, (
        f"resolver fell back to the hostile PATH copy: stdout={result.stdout!r}"
    )
    assert ".venv" in result.stderr, (
        "the veto message does not name the pinned venv it looked in: "
        f"stderr={result.stderr!r}"
    )


# --- 2. The ambient (no pinned venv) branch stays attributable ------------


def test_ambient_resolution_is_announced_with_an_absolute_path(
    tmp_path: Path,
) -> None:
    """With no pinned venv, resolution falls back to PATH but says so.

    CI installs the toolchain straight into the runner's interpreter, so there
    is no venv to anchor to and PATH is the only signal available. That is
    acceptable *only* if the gate names the environment it used, so a verdict
    is always attributable rather than silently ambient.
    """
    repo = _fake_repo(tmp_path, venv_tools=None)
    decoys = _decoy_dir(tmp_path, ("pip-audit", "python3"))

    result = _run_resolver(repo, ["--print-tool", "pip-audit"], _hostile_path(decoys))

    assert result.returncode == 0, (
        f"--print-tool failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stdout.strip() == str(decoys / "pip-audit"), (
        f"unexpected ambient resolution: stdout={result.stdout!r}"
    )
    assert str(decoys / "pip-audit") in result.stderr, (
        "the ambient fallback was not announced with the absolute path it "
        f"chose: stderr={result.stderr!r}"
    )


def test_tool_absent_everywhere_vetoes(tmp_path: Path) -> None:
    """A tool present in neither the pinned venv nor PATH must veto.

    A gate that cannot run its tool must fail, never quietly report success --
    unprovable evidence must not read as healthy.
    """
    repo = _fake_repo(tmp_path, venv_tools=None)
    decoys = _decoy_dir(tmp_path, ("python3",))

    result = _run_resolver(repo, ["--print-tool", "pip-audit"], _hostile_path(decoys))

    assert result.returncode != 0, (
        "an unresolvable tool did not veto: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "pip-audit" in result.stderr, (
        f"the veto message does not name the missing tool: stderr={result.stderr!r}"
    )


# --- 3. The security gate audits the pinned interpreter's environment -----


def test_security_script_never_invokes_pip_audit_by_bare_name() -> None:
    """`security.sh` must not run `pip-audit` as a PATH-resolved console script.

    A bare `pip-audit` is the defect: the console script's shebang decides which
    interpreter -- and therefore which site-packages -- gets audited.
    """
    source = _read(_SECURITY_SCRIPT)

    offenders = _bare_invocations(source, "pip-audit")
    assert not offenders, (
        "scripts/security.sh still invokes pip-audit by bare name, so the "
        f"audited environment is chosen by the caller's PATH: {offenders}"
    )


def test_security_script_audits_via_the_pinned_interpreter() -> None:
    """`security.sh` must run pip-audit as a module of the pinned interpreter.

    `<python> -m pip_audit` audits that interpreter's own site-packages, so
    binding the audit to the pinned interpreter makes "which environment was
    audited" a construction guarantee rather than a PATH accident.
    """
    source = _read(_SECURITY_SCRIPT)

    assert "-m pip_audit" in source, (
        "scripts/security.sh does not run pip-audit as `<python> -m pip_audit`, "
        "so the audited environment is not provably the pinned one"
    )

    module_lines = [line for line in source.splitlines() if "-m pip_audit" in line]
    interpreters = {
        match.group(1)
        for line in module_lines
        for match in [
            re.match(r'^[ \t]*"\$([A-Za-z_][A-Za-z0-9_]*)"[ \t]+-m pip_audit', line)
        ]
        if match
    }
    assert interpreters, (
        "the `-m pip_audit` invocation does not run through a shell variable "
        f"holding a resolved interpreter path: {module_lines}"
    )

    for variable in interpreters:
        assigned = re.search(
            rf'^[ \t]*{variable}="\$\(bash "\$[A-Za-z_][A-Za-z0-9_]*" '
            r'--print-python\)"',
            source,
            re.MULTILINE,
        )
        assert assigned, (
            f"scripts/security.sh runs pip-audit through ${variable}, which is "
            "not assigned from `toolchain-env.sh --print-python` -- the audited "
            "environment is therefore not provably the pinned one"
        )


@pytest.mark.parametrize("script_name", sorted(_GATE_SCRIPT_TOOLS))
def test_gate_scripts_use_the_toolchain_resolver(script_name: str) -> None:
    """Every gate script resolves its tools through the shared resolver.

    Doing it in each script -- not only in `check-all.sh` -- is what makes a
    direct `./scripts/security.sh` behave identically to the same check run via
    Gate 1. The old arrangement anchored PATH only inside `check-all.sh`, so the
    documented standalone invocations silently used ambient tools.
    """
    source = _read(_SCRIPTS_DIR / script_name)

    assert "toolchain-env.sh" in source, (
        f"scripts/{script_name} does not resolve through scripts/"
        "toolchain-env.sh, so its tools still come from the caller's PATH"
    )


@pytest.mark.parametrize(
    ("script_name", "tool"),
    [
        (script_name, tool)
        for script_name, tools in sorted(_GATE_SCRIPT_TOOLS.items())
        for tool in tools
    ],
)
def test_gate_scripts_have_no_bare_name_tool_invocations(
    script_name: str, tool: str
) -> None:
    """No gate script may invoke its tool by bare name.

    Bare names hand the choice of binary to the caller's PATH. Resolved
    absolute paths (via the toolchain helper) make the gate's verdict a
    property of the repo's pinned toolchain instead.
    """
    source = _read(_SCRIPTS_DIR / script_name)

    offenders = _bare_invocations(source, tool)
    assert not offenders, (
        f"scripts/{script_name} invokes {tool!r} by bare name, so which binary "
        f"runs depends on the caller's PATH: {offenders}"
    )


def test_architecture_scripts_inherited_path_is_documented() -> None:
    """The one still-inherited sub-check must be named, with its reason.

    Issue #366's acceptance criteria allow a sub-check to keep inheriting PATH
    only if it is *explicitly listed with the reason*. `architecture.sh`
    delegates to `plans/architecture/run-check.sh`, whose bare `lint-imports`
    lookup is deliberately PATH-sensitive: `tests/architecture/
    test_import_linter_gate.py` pins a fail-loud-when-unreachable-on-PATH
    contract that anchoring would silently defeat. Pinning the documentation
    here stops that exemption from quietly becoming an unexplained hole.
    """
    source = _read(_TOOLCHAIN_ENV_SCRIPT)

    assert "architecture.sh" in source, (
        "scripts/toolchain-env.sh does not document architecture.sh as the "
        "still-inherited sub-check required by issue #366's acceptance criteria"
    )
    assert "lint-imports" in source, (
        "the architecture.sh exemption does not name the inherited tool "
        "(lint-imports), so the reason cannot be audited"
    )


# --- 4. End-to-end proof against the real shared venv ---------------------


def test_real_gate_audits_the_project_venv_under_a_hostile_path(
    tmp_path: Path,
) -> None:
    """With a hostile PATH, the real gate still audits the project venv.

    This is issue #366's headline acceptance criterion run against the actual
    repository. `pip_audit` audits the site-packages of the interpreter that
    imports it, so proving the resolved interpreter's `sys.prefix` is the
    shared venv proves the audited dependency set is the project's.

    Skipped where no shared venv exists (CI installs the toolchain into the
    runner interpreter instead); the hermetic tests above cover that branch on
    every platform.
    """
    venv_dir = subprocess.run(
        [_bash(), str(_SCRIPTS_DIR / "provision-venv.sh"), "--print-venv"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.strip()
    if not venv_dir or not (Path(venv_dir) / "bin" / "python").is_file():
        pytest.skip(f"no shared pinned venv at {venv_dir!r} to prove anchoring against")

    decoys = _decoy_dir(tmp_path, ("python", "python3", "pip-audit"))
    resolved = subprocess.run(
        [_bash(), str(_TOOLCHAIN_ENV_SCRIPT), "--print-python"],
        cwd=str(_REPO_ROOT),
        env={"PATH": _hostile_path(decoys), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert resolved.returncode == 0, (
        f"--print-python failed: stdout={resolved.stdout!r} stderr={resolved.stderr!r}"
    )

    interpreter = resolved.stdout.strip()
    assert interpreter == str(Path(venv_dir) / "bin" / "python"), (
        f"hostile PATH redirected the gate interpreter to {interpreter!r}"
    )

    prefix = subprocess.run(
        [interpreter, "-c", "import sys; print(sys.prefix)"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.strip()
    assert Path(prefix) == Path(venv_dir), (
        "the interpreter the security gate would audit reports "
        f"sys.prefix={prefix!r}, not the project venv {venv_dir!r}"
    )
