"""The built wheel is installed and executed, not just metadata-checked (#467).

CI's `build` job runs ``python -m build`` then ``twine check dist/*``. That
verifies the long description renders and the metadata is well-formed. It never
installs the wheel, never resolves the wheel's own declared dependencies, and
never runs the console script -- so ``[project.scripts] windbreak =
"windbreak.main:main"``, the product's entire user-facing surface, has never
been executed from an installed artifact. Every other CLI test in this
repository imports :mod:`windbreak.main` in-process, from a checkout whose root
is already on ``sys.path`` with the dev requirements present.

WHAT THAT HIDES, and what each test below therefore has to prove:

* a package dropped from ``[tool.setuptools.packages.find]`` -- caught by
  :func:`test_the_wheel_ships_every_subpackage_in_the_source_tree` and, from the
  other direction, :func:`test_every_shipped_subpackage_imports_from_the_wheel`;
* a runtime dependency satisfied only because it is *also* a dev dependency --
  caught because the venv installs the wheel and nothing else;
* a broken entry-point target -- caught by every test that runs the console
  script by name out of the venv's ``bin/``.

THE ISOLATION IS THE EXPERIMENT. Each assertion below is worthless if the
subject can reach the checkout, so
:func:`test_the_clean_venv_cannot_reach_the_repo_or_the_dev_toolchain` runs
first in file order and is the control: it pins that the repository root is not
on the venv's ``sys.path``, that ``windbreak`` resolves inside the venv, and
that a dev-only distribution (`pytest`) is absent. Without it a green run here
would be indistinguishable from importing the source tree.

WHY THE `container` MARKER, on a test that needs no container: the marker is
this repository's one "deselected by default, run by a dedicated CI job"
selector, and this module must not run in the default suite -- it needs a wheel
in ``dist/`` and a reachable package index, neither of which Gate 1 or an
offline developer machine can assume. `pyproject.toml`'s marker description
(line 71) still describes the tier as "needing a docker daemon or running
systemd"; that line now under-describes it and wants "or a built distribution
in `dist/`". It is not edited here because `pyproject.toml` is outside this
change's owned paths -- recorded rather than silently left wrong.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.e2e.harness import (
    DEFAULT_TIMEOUT_SECONDS,
    SpawnedProcess,
    free_port,
    ledger_event_types,
    port_is_serving,
    require_runtime,
    wait_until,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.e2e.harness import ProcessLauncher, RunRoot

pytestmark = pytest.mark.container

#: The repository root, two levels up from `tests/e2e/test_installed_wheel.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Directory `python -m build` writes the distribution into.
DIST_DIR = REPO_ROOT / "dist"

#: The importable package the wheel is built from, as a source directory.
PACKAGE_DIR = REPO_ROOT / "windbreak"

#: Distribution name, as `pyproject.toml` declares it.
DISTRIBUTION_NAME = "windbreak"

#: Console script `[project.scripts]` installs into the venv's `bin/`.
CONSOLE_SCRIPT = "windbreak"

#: A distribution that is dev-only. Its presence in the clean venv would mean
#: the venv is not clean and every isolation claim in this module is void.
DEV_ONLY_DISTRIBUTION = "pytest"

#: Seconds allowed for `pip install` of the wheel and its declared dependencies.
INSTALL_TIMEOUT_SECONDS = 600.0

#: Seconds allowed for the dashboard to bind its loopback port.
DASHBOARD_BIND_TIMEOUT_SECONDS = 60.0

#: Seconds allowed for the dashboard to exit after `SIGTERM`.
DASHBOARD_SHUTDOWN_TIMEOUT_SECONDS = 30.0

#: The `--process` tokens whose `run` terminates on its own `--max-beats`
#: budget, mapped to the exact ledger histogram one beat must produce.
#:
#: `dashboard` is deliberately absent: it does not honour `--max-beats` at all
#: (`windbreak/main.py:_run_dashboard` serves until signalled), so it gets its
#: own test below. Issue #467 assumed all four terminate; they do not.
TERMINATING_PROCESS_BEAT_ROWS: dict[str, dict[str, int]] = {
    "pipeline": {"ConfigLoaded": 1},
    "riskkernel": {"ConfigLoaded": 1, "ModeHeartbeat": 1},
    "order_gateway": {"ConfigLoaded": 1},
}


def _declared_version() -> str:
    """Read the distribution version `pyproject.toml` declares.

    Derived rather than restated so a release that bumps one and forgets the
    other cannot pass by matching a copy of itself.

    Returns:
        The `[project] version` string.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        parsed = tomllib.load(handle)
    return str(parsed["project"]["version"])


def _source_subpackages() -> list[str]:
    """Enumerate every importable package under the source tree.

    Walks the filesystem instead of naming packages by hand: a hand-restated
    list drifts from the artifact it mirrors, and the whole point here is to
    notice a package the wheel stopped shipping.

    Returns:
        Dotted package names, `windbreak` first, then its subpackages sorted.
    """
    found = [DISTRIBUTION_NAME]
    for init in sorted(PACKAGE_DIR.rglob("__init__.py")):
        relative = init.parent.relative_to(REPO_ROOT)
        if relative == Path(DISTRIBUTION_NAME):
            continue
        found.append(".".join(relative.parts))
    return found


def _wheel_skip_reason() -> str | None:
    """Explain why the wheel tier cannot run here, if it cannot.

    Returns:
        `None` when exactly one matching wheel is present, else a reason.
    """
    if not DIST_DIR.is_dir():
        return (
            f"no built distribution: {DIST_DIR} does not exist. Run "
            "`python -m build` first -- this tier asserts on the wheel, not "
            "the checkout."
        )
    candidates = sorted(DIST_DIR.glob(f"{DISTRIBUTION_NAME}-*.whl"))
    if not candidates:
        return (
            f"no built distribution: {DIST_DIR} holds no "
            f"{DISTRIBUTION_NAME}-*.whl. Run `python -m build` first."
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        return (
            f"ambiguous built distribution: {DIST_DIR} holds {len(candidates)} "
            f"wheels ({names}). Clear it and rebuild, so the artifact under "
            "test is unambiguous."
        )
    return None


@pytest.fixture(scope="session")
def built_wheel() -> Path:
    """Locate the single wheel CI's build step produced.

    Returns:
        Path to the wheel under test.
    """
    require_runtime(_wheel_skip_reason())
    return next(iter(sorted(DIST_DIR.glob(f"{DISTRIBUTION_NAME}-*.whl"))))


@pytest.fixture(scope="session")
def clean_venv(built_wheel: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a virtualenv holding the wheel and its declared dependencies only.

    No editable install, no dev requirements, no ``--system-site-packages``.
    The only thing pip is told to install is the wheel; anything else that ends
    up importable got there because the wheel asked for it, which is exactly
    the claim under test.

    PIP RUNS WITH ``PYTHONPATH`` STRIPPED, and that is not hygiene. pip treats
    an importable ``windbreak.egg-info`` on ``PYTHONPATH`` as the requirement
    already being satisfied: it then installs the *dependencies* only, exits 0,
    and leaves a venv with requests and pyyaml in it and no ``windbreak`` at
    all. ``PYTHONPATH=<repo>`` is a common CI convention, so this fixture --
    the one that builds the subject of every other test here -- was the single
    place in the module that could still see the checkout, while
    :func:`_run_in_venv` and :func:`_run_console_script` already stripped it.

    A zero exit therefore proves nothing on its own, and is not what is
    asserted: the version is read back out of the venv's own metadata and
    compared with the declaration, so "pip succeeded" and "the package is
    installed" stop being the same claim.

    Args:
        built_wheel: The wheel to install.
        tmp_path_factory: pytest's session-scoped temporary directory factory.

    Returns:
        The virtualenv's root directory.
    """
    venv_dir = tmp_path_factory.mktemp("wheel-venv") / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        env=_outside_repo_env(),
        check=True,
    )
    venv_python = venv_dir / "bin" / "python"
    installed = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            str(built_wheel),
        ],
        capture_output=True,
        text=True,
        timeout=INSTALL_TIMEOUT_SECONDS,
        env=_outside_repo_env(),
        check=False,
    )
    assert installed.returncode == 0, (
        f"installing {built_wheel.name} into a clean venv failed -- the wheel "
        "cannot satisfy its own declared dependencies:\n"
        f"{installed.stdout}\n{installed.stderr}"
    )
    resolved = _run_in_venv(
        venv_python,
        "import importlib.metadata as metadata\n"
        f"print(metadata.version({DISTRIBUTION_NAME!r}))",
        cwd=venv_dir,
    )

    assert resolved.returncode == 0, (
        f"pip exited 0 but {DISTRIBUTION_NAME} is not installed in the venv it "
        f"claims to have installed into:\n{resolved.stderr}"
    )
    assert resolved.stdout.strip() == _declared_version(), (
        f"the venv reports {DISTRIBUTION_NAME} "
        f"{resolved.stdout.strip()!r}, but pyproject.toml declares "
        f"{_declared_version()!r} -- the installed distribution is not the one "
        "built from this tree"
    )
    return venv_dir


@pytest.fixture(scope="session")
def venv_python(clean_venv: Path) -> Path:
    """Return the clean venv's interpreter.

    Args:
        clean_venv: The virtualenv holding the installed wheel.

    Returns:
        Path to the venv's ``bin/python``.
    """
    return clean_venv / "bin" / "python"


@pytest.fixture(scope="session")
def venv_console_script(clean_venv: Path) -> Path:
    """Return the console script the wheel's entry point installed.

    Args:
        clean_venv: The virtualenv holding the installed wheel.

    Returns:
        Path to the venv's ``bin/windbreak``.
    """
    return clean_venv / "bin" / CONSOLE_SCRIPT


def _outside_repo_env() -> dict[str, str]:
    """Build a child environment that cannot smuggle the checkout in.

    Returns:
        A copy of this environment with ``PYTHONPATH`` removed.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def _run_in_venv(
    venv_python: Path,
    source: str,
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Execute a snippet with the clean venv's interpreter.

    ``cwd`` is never the repository root: ``python -c`` puts the working
    directory on ``sys.path``, so running from the checkout would import the
    source tree and quietly answer a different question than the one asked.

    Args:
        venv_python: The venv's interpreter.
        source: Python source to execute with ``-c``.
        cwd: Working directory for the child, outside the checkout.

    Returns:
        The completed process, decoded as text.
    """
    return subprocess.run(
        [str(venv_python), "-c", source],
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        cwd=str(cwd),
        env=_outside_repo_env(),
        check=False,
    )


def _run_console_script(
    venv_console_script: Path,
    *args: str,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run the installed console script by name, as an operator would.

    Args:
        venv_console_script: Path to the venv's ``bin/windbreak``.
        *args: Arguments appended to the console script.
        cwd: Working directory for the child, outside the checkout.
        env: Complete environment for the child. ``None`` strips ``PYTHONPATH``
            from this one.
        timeout: Seconds to allow before the child is killed.

    Returns:
        The completed process, decoded as text.
    """
    return subprocess.run(
        [str(venv_console_script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd),
        env=_outside_repo_env() if env is None else dict(env),
        check=False,
    )


def test_the_clean_venv_cannot_reach_the_repo_or_the_dev_toolchain(
    venv_python: Path,
    run_root: RunRoot,
) -> None:
    """The venv resolves `windbreak` from the wheel and nothing from the repo.

    The control for this whole module. Three independent things are pinned
    together because any one of them failing alone would make every other test
    here pass for the wrong reason: the repository root is not on ``sys.path``,
    the imported ``windbreak`` lives under the venv, and a dev-only
    distribution is absent.

    Args:
        venv_python: The clean venv's interpreter.
        run_root: An isolated directory to run the child from.
    """
    probe = (
        "import importlib.util, json, sys, windbreak\n"
        "print(json.dumps({"
        '"sys_path": sys.path, '
        '"windbreak_file": windbreak.__file__, '
        f'"dev_only_found": importlib.util.find_spec("{DEV_ONLY_DISTRIBUTION}")'
        " is not None}))"
    )

    completed = _run_in_venv(venv_python, probe, cwd=run_root.root)

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    resolved = [Path(entry).resolve() for entry in observed["sys_path"] if entry]
    assert REPO_ROOT not in resolved, (
        f"the repository root is on the clean venv's sys.path: {resolved}"
    )
    assert (
        Path(observed["windbreak_file"])
        .resolve()
        .is_relative_to(venv_python.parent.parent)
    ), f"windbreak resolved outside the venv: {observed['windbreak_file']}"
    assert observed["dev_only_found"] is False, (
        f"{DEV_ONLY_DISTRIBUTION} is importable in the clean venv, so the venv "
        "is not clean and no isolation claim in this module holds"
    )


def test_the_wheel_installs_the_console_script_as_an_executable(
    venv_console_script: Path,
) -> None:
    """`[project.scripts]` puts an executable `windbreak` in the venv's bin.

    Args:
        venv_console_script: Path to the venv's ``bin/windbreak``.
    """
    assert venv_console_script.is_file(), (
        f"the wheel installed no {CONSOLE_SCRIPT} console script at "
        f"{venv_console_script}"
    )
    assert os.access(venv_console_script, os.X_OK)


def test_the_installed_distribution_version_matches_pyproject(
    venv_python: Path,
    run_root: RunRoot,
) -> None:
    """The wheel's metadata version and `windbreak.__version__` both match.

    Issue #467 asked for this through ``windbreak --version``. That flag does
    not exist -- see
    :func:`test_the_console_script_has_no_version_flag_yet` -- so the same
    guarantee is taken from the installed distribution's own metadata, which is
    strictly closer to what a release actually publishes. A bump that touches
    `pyproject.toml` but not `windbreak/__init__.py` (or the reverse) fails
    here.

    Args:
        venv_python: The clean venv's interpreter.
        run_root: An isolated directory to run the child from.
    """
    probe = (
        "import json, importlib.metadata, windbreak\n"
        "print(json.dumps({"
        f'"metadata": importlib.metadata.version("{DISTRIBUTION_NAME}"), '
        '"dunder": windbreak.__version__}))'
    )

    completed = _run_in_venv(venv_python, probe, cwd=run_root.root)

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    declared = _declared_version()
    assert observed["metadata"] == declared
    assert observed["dunder"] == declared


def test_the_console_script_has_no_version_flag_yet(
    venv_console_script: Path,
    run_root: RunRoot,
) -> None:
    """`windbreak --version` is not implemented; this pins the gap (#507).

    Issue #467's acceptance criteria assume the flag exists. It does not:
    `build_parser` registers no ``--version`` action, so argparse rejects the
    invocation and exits 2 with a usage message. Asserting the *current*
    behaviour keeps `main` green while making the gap impossible to lose --
    this test fails the moment the flag lands, which forces #467's original
    assertion to be written then. Same pattern as
    `test_documented_rearm_invocation_fails_without_an_interactive_phrase`.

    Args:
        venv_console_script: Path to the venv's ``bin/windbreak``.
        run_root: An isolated directory to run the child from.
    """
    completed = _run_console_script(venv_console_script, "--version", cwd=run_root.root)

    assert completed.returncode == 2
    assert _declared_version() not in completed.stdout
    assert "usage: windbreak" in completed.stderr


def test_the_console_script_help_exits_zero(
    venv_console_script: Path,
    run_root: RunRoot,
) -> None:
    """`windbreak --help` runs the installed entry point and exits 0.

    The narrowest possible proof that ``windbreak.main:main`` resolves and
    executes from the wheel: a broken entry-point target cannot get this far.

    Args:
        venv_console_script: Path to the venv's ``bin/windbreak``.
        run_root: An isolated directory to run the child from.
    """
    completed = _run_console_script(venv_console_script, "--help", cwd=run_root.root)

    assert completed.returncode == 0, completed.stderr
    assert "usage: windbreak" in completed.stdout


@pytest.mark.parametrize("process", sorted(TERMINATING_PROCESS_BEAT_ROWS))
def test_the_console_script_runs_one_beat_for_each_terminating_process(
    process: str,
    venv_console_script: Path,
    run_root: RunRoot,
) -> None:
    """One beat of each self-terminating process exits 0 and ledgers its rows.

    Exit 0 alone would pass against a binary that started and stopped without
    doing anything, so the ledger histogram is asserted exactly -- not
    ``>= 1``, and not merely "non-empty".

    Args:
        process: The ``--process`` token under test.
        venv_console_script: Path to the venv's ``bin/windbreak``.
        run_root: The isolated run root supplying ``--ledger-path``.
    """
    ledger_path = run_root.root / f"{process}.db"

    completed = _run_console_script(
        venv_console_script,
        "run",
        "--process",
        process,
        "--max-beats",
        "1",
        "--heartbeat-interval",
        "0.1",
        "--ledger-path",
        str(ledger_path),
        "--report-dir",
        str(run_root.report_dir),
        cwd=run_root.root,
    )

    assert completed.returncode == 0, completed.stderr
    histogram: dict[str, int] = {}
    for event_type in ledger_event_types(ledger_path):
        histogram[event_type] = histogram.get(event_type, 0) + 1
    assert histogram == TERMINATING_PROCESS_BEAT_ROWS[process]


def test_the_console_script_serves_the_dashboard_and_shuts_down_cleanly(
    venv_console_script: Path,
    run_root: RunRoot,
    launcher: ProcessLauncher,
) -> None:
    """The installed `dashboard` process binds loopback and exits 0 on SIGTERM.

    The fourth ``--process`` token, and the one issue #467 got wrong: it
    ignores ``--max-beats`` entirely and serves until signalled, so a
    ``--max-beats 1`` invocation would hang rather than exit 0. Running it for
    real is still worth doing -- it is the only process that imports
    :mod:`windbreak.dashboard`, so it is the only one that would notice that
    subpackage missing from the wheel at runtime.

    Args:
        venv_console_script: Path to the venv's ``bin/windbreak``.
        run_root: The isolated run root supplying paths.
        launcher: Guarantees the server is reaped however this test ends.
    """
    port = free_port()
    config_path = run_root.root / "dashboard.yaml"
    config_path.write_text(f"dashboard:\n  port: {port}\n", encoding="utf-8")
    env = _outside_repo_env()
    env["WINDBREAK_DASHBOARD_TOKEN"] = "e2e-wheel-token-not-a-secret"
    stdout_path = run_root.log_dir / "wheel-dashboard.stdout.log"
    stderr_path = run_root.log_dir / "wheel-dashboard.stderr.log"
    stdout_stream = stdout_path.open("wb")
    stderr_stream = stderr_path.open("wb")
    argv = (
        str(venv_console_script),
        "run",
        "--process",
        "dashboard",
        "--config",
        str(config_path),
        "--ledger-path",
        str(run_root.ledger_path),
    )
    spawned = launcher.track(
        SpawnedProcess(
            name="wheel-dashboard",
            argv=argv,
            process=subprocess.Popen(
                list(argv),
                stdout=stdout_stream,
                stderr=stderr_stream,
                cwd=str(run_root.root),
                env=env,
                start_new_session=True,
            ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_stream=stdout_stream,
            stderr_stream=stderr_stream,
        )
    )

    def _serving_or_died() -> bool:
        """Report whether the port is up, failing fast if the child died.

        Returns:
            ``True`` once something accepts connections on ``port``.

        Raises:
            AssertionError: If the child exited before binding.
        """
        if not spawned.is_running():
            message = (
                f"the installed console script exited "
                f"{spawned.process.returncode} before binding {port}:\n"
                f"{spawned.stderr_text()}"
            )
            raise AssertionError(message)
        return port_is_serving(port)

    wait_until(
        _serving_or_died,
        timeout=DASHBOARD_BIND_TIMEOUT_SECONDS,
        description=f"the installed console script to serve the dashboard on {port}",
    )
    spawned.process.terminate()
    exit_code = spawned.wait(timeout=DASHBOARD_SHUTDOWN_TIMEOUT_SECONDS)

    assert exit_code == 0, spawned.stderr_text()
    assert ledger_event_types(run_root.ledger_path) == ["ConfigLoaded"]


def test_the_wheel_ships_every_subpackage_in_the_source_tree(
    built_wheel: Path,
) -> None:
    """Every package under `windbreak/` is present inside the wheel archive.

    This is the assertion that goes red when a package is dropped from
    ``[tool.setuptools.packages.find]``, and it names the missing packages
    rather than failing on an `ImportError` three layers down.

    Args:
        built_wheel: The wheel under test.
    """
    expected = _source_subpackages()

    assert expected, "the package walk found nothing -- the walk itself is broken"
    assert f"{DISTRIBUTION_NAME}.ledger" in expected, (
        "the package walk missed a package that certainly exists, so a green "
        "result below would prove nothing"
    )
    with zipfile.ZipFile(built_wheel) as archive:
        shipped = set(archive.namelist())
    missing = [
        package
        for package in expected
        if f"{package.replace('.', '/')}/__init__.py" not in shipped
    ]

    assert not missing, (
        f"{built_wheel.name} is missing {len(missing)} package(s) present in "
        f"the source tree: {', '.join(missing)}. Check "
        "[tool.setuptools.packages.find] in pyproject.toml."
    )


def _declared_packages() -> list[str]:
    """Compute the package set `pyproject.toml` currently declares.

    Asks setuptools the same question the build backend asks it, with the
    same `[tool.setuptools.packages.find]` arguments, rather than restating
    the answer or re-implementing the glob semantics.

    THE IMPORT IS DEFERRED ON PURPOSE. `setuptools` is a build-time
    dependency -- `[build-system] requires` names it, `requirements.txt` and
    `requirements-dev.txt` do not. At module scope the import would become a
    *collection*-time dependency for every pytest run in the repository,
    including Gate 1 and all three `quality` legs, none of which reach this
    function: pytest imports a module to read its markers even when the whole
    module is about to be deselected. Deferring it scopes the dependency to
    the one job that genuinely has setuptools, the container job, which
    installs it explicitly. Pinning it in `requirements-dev.txt` would be the
    alternative, and is defensible -- but that file is outside this change's
    owned paths, and needing it there at all is an artefact of where the
    import sat.

    Returns:
        The declared package names, sorted.
    """
    from setuptools import find_namespace_packages, find_packages

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        parsed = tomllib.load(handle)
    config = parsed["tool"]["setuptools"]["packages"]["find"]
    discover = (
        find_namespace_packages if config.get("namespaces", True) else find_packages
    )
    declared: set[str] = set()
    for where in config.get("where", ["."]):
        declared |= set(
            discover(
                where=str(REPO_ROOT / where),
                include=config.get("include", ["*"]),
                exclude=config.get("exclude", []),
            )
        )
    return sorted(declared)


def test_the_wheel_ships_exactly_the_packages_pyproject_declares(
    built_wheel: Path,
) -> None:
    """The wheel's package set matches the CURRENT build configuration.

    THE ARTIFACT IS NOT TIED TO THE TREE BY ITS FILENAME.
    :func:`_wheel_skip_reason` asks only that ``dist/`` hold exactly one
    ``windbreak-*.whl``; nothing there notices that the wheel predates the
    `pyproject.toml` sitting beside it. Reproduced, not hypothesised: with a
    stale ``build/`` directory present, ``python -m build`` re-emitted a wheel
    still containing ``windbreak/dashboard/**`` after that package had been
    excluded from `[tool.setuptools.packages.find]`, and this tier reported 11
    passed against it. The proof-of-failure procedure this pull request
    mandates would have read as "the break did not go red".

    The other direction is already covered --
    :func:`test_the_wheel_ships_every_subpackage_in_the_source_tree` catches a
    package dropped from the wheel. It cannot catch a package the wheel keeps
    after the configuration stopped asking for it, because the source
    directory is still there. Only the declaration knows, so the declaration
    is what this compares against.

    Args:
        built_wheel: The wheel under test.
    """
    declared = _declared_packages()

    assert declared != [], (
        "setuptools discovered no packages from pyproject.toml's find "
        "configuration, so the comparison below would be between two empty "
        "sets and could not fail"
    )
    with zipfile.ZipFile(built_wheel) as archive:
        shipped = sorted(
            name.rsplit("/", 1)[0].replace("/", ".")
            for name in archive.namelist()
            if name.endswith("/__init__.py")
        )

    assert shipped == declared, (
        f"{built_wheel.name} does not match the build configuration it sits "
        f"beside. Only in the wheel: {sorted(set(shipped) - set(declared))}. "
        f"Only in pyproject.toml: {sorted(set(declared) - set(shipped))}. The "
        "wheel is stale -- rebuild it after clearing `build/`, which caches "
        "the previous configuration's file list."
    )


def test_every_module_in_the_wheel_matches_the_source_it_was_built_from(
    built_wheel: Path,
) -> None:
    """Each `.py` the wheel ships is byte-identical to the tree's copy.

    The package-set comparison above catches a wheel built under a different
    *configuration*. This catches one built from different *content*: same
    packages, same filenames, an edit made since. Both are the same defect --
    a tier certifying an artifact that is not the tree under review -- and
    neither is visible from the wheel's filename, which carries only a version
    that changes once a release.

    Args:
        built_wheel: The wheel under test.
    """
    with zipfile.ZipFile(built_wheel) as archive:
        shipped = sorted(
            name
            for name in archive.namelist()
            if name.startswith(f"{DISTRIBUTION_NAME}/") and name.endswith(".py")
        )
        source_files = sorted(
            str(path.relative_to(REPO_ROOT))
            for path in PACKAGE_DIR.rglob("*.py")
            if path.is_file()
        )

        assert shipped == source_files, (
            f"{built_wheel.name} ships a different set of modules than the "
            f"source tree holds. Only in the wheel: "
            f"{sorted(set(shipped) - set(source_files))}. Only in the tree: "
            f"{sorted(set(source_files) - set(shipped))}."
        )
        differing = [
            name
            for name in shipped
            if archive.read(name) != (REPO_ROOT / name).read_bytes()
        ]

    assert differing == [], (
        f"{len(differing)} module(s) in {built_wheel.name} differ from the "
        f"source they were supposedly built from: {differing}. The wheel is "
        "stale; every assertion in this tier is being made against code that "
        "is not under review. Clear `build/` and `dist/` and rebuild."
    )


def test_every_shipped_subpackage_imports_from_the_wheel(
    venv_python: Path,
    run_root: RunRoot,
) -> None:
    """Importing each subpackage in the clean venv raises no `ImportError`.

    The runtime half of the previous test. A package can be *in* the wheel and
    still fail to import there -- because it reaches for a module that only
    exists in the checkout, or for a dependency that is only ever installed
    because it is also a dev dependency.

    Args:
        venv_python: The clean venv's interpreter.
        run_root: An isolated directory to run the child from.
    """
    expected = _source_subpackages()

    assert expected, "the package walk found nothing -- the walk itself is broken"
    # `imported` collects each module's OWN `__name__`, read back off the
    # imported object. The obvious formulation -- reporting `len(packages)` --
    # cannot fail: the probe embeds the parent's list literally, so the count
    # is that list's length by construction, whatever importing did. Comparing
    # a count where an identity was meant is a assertion-shaped no-op, and this
    # was one until review of PR #508 named it.
    probe = (
        "import importlib, json\n"
        f"packages = {expected!r}\n"
        "failures = {}\n"
        "imported = []\n"
        "for name in packages:\n"
        "    try:\n"
        "        module = importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        failures[name] = f'{type(exc).__name__}: {exc}'\n"
        "    else:\n"
        "        imported.append(module.__name__)\n"
        "print(json.dumps({'imported': imported, 'failures': failures}))"
    )

    completed = _run_in_venv(venv_python, probe, cwd=run_root.root)

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["failures"] == {}, (
        f"{len(observed['failures'])} package(s) are in the wheel but do not "
        f"import from it: {observed['failures']}"
    )
    assert observed["imported"] == expected, (
        "the modules that imported are not the modules that were asked for. "
        f"Asked for {expected}, imported {observed['imported']} -- a name "
        "resolved to a module calling itself something else, which means the "
        "wheel's package layout does not match the source tree's."
    )
