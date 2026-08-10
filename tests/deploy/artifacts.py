"""Parsers for the shipped deployment artifacts (issues #445, #446).

Every helper here reads a **real** file — `deploy/docker-compose.yml`,
`deploy/systemd/*.service`, the repo-root `Dockerfile` and `.dockerignore` —
and returns its parsed contents. Nothing in this module restates what those
files say: the artifacts *are* the fixtures, so a test built on these helpers
fails when the shipped artifact changes rather than when a hand-kept copy of it
drifts (CLAUDE.md's derive-never-restate rule).

The service commands are parsed by :func:`windbreak.main.build_parser` — the
same parser the container invokes — so a flag that the CLI would reject, or a
flag whose ``dest`` the activation rule no longer reads, cannot pass unnoticed.
"""

from __future__ import annotations

import ast
import configparser
import fnmatch
import inspect
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from windbreak import main as windbreak_main

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

#: The repository root, three levels up from `tests/deploy/artifacts.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "deploy"
COMPOSE_PATH = DEPLOY_DIR / "docker-compose.yml"
SYSTEMD_DIR = DEPLOY_DIR / "systemd"
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"

#: systemd units invoke the CLI through this shim so they stay
#: install-prefix-agnostic; it is not part of the parsed argument vector.
ENV_SHIM = "/usr/bin/env"

#: The console-script name every shipped command must launch.
ENTRYPOINT = "windbreak"

#: Docker's default dockerfile name inside a build context.
DEFAULT_DOCKERFILE_NAME = "Dockerfile"

_SUBPROCESS_TIMEOUT_SECONDS = 60


def load_compose() -> dict[str, Any]:
    """Parse `deploy/docker-compose.yml` with `yaml.safe_load`.

    Returns:
        The parsed top-level compose mapping.
    """
    with COMPOSE_PATH.open(encoding="utf-8") as handle:
        parsed: dict[str, Any] = yaml.safe_load(handle)
    return parsed


def compose_services() -> dict[str, dict[str, Any]]:
    """Return every service defined by the shipped compose file.

    Returns:
        A mapping of service name to that service's parsed compose mapping.
    """
    services: dict[str, dict[str, Any]] = load_compose()["services"]
    return services


def command_tokens(service: dict[str, Any]) -> list[str]:
    """Return one compose service's `command` as an argument vector.

    Args:
        service: One service's parsed compose mapping.

    Returns:
        The command's tokens; a shell-string form is split with `shlex`.
    """
    command = service["command"]
    if isinstance(command, list):
        return [str(token) for token in command]
    return shlex.split(str(command))


def build_spec(service: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one service's `build` key to its long mapping form.

    Args:
        service: One service's parsed compose mapping.

    Returns:
        A mapping with at least a `context` key, or `None` when the service
        declares no build at all (i.e. it runs a pre-built `image`).
    """
    build = service.get("build")
    if build is None:
        return None
    if isinstance(build, str):
        return {"context": build}
    spec: dict[str, Any] = dict(build)
    return spec


def resolved_dockerfile(spec: dict[str, Any]) -> Path:
    """Resolve a build spec's dockerfile the way Compose resolves it.

    Compose resolves a relative `context` against the **compose file's own
    directory**, never the invoking shell's working directory — the rule that
    made `build: .` point at `deploy/` (issue #445).

    Args:
        spec: A normalized build spec from :func:`build_spec`.

    Returns:
        The absolute path Compose would read the dockerfile from.
    """
    context = (COMPOSE_PATH.parent / str(spec["context"])).resolve()
    return context / str(spec.get("dockerfile", DEFAULT_DOCKERFILE_NAME))


def parse_mount(entry: str) -> tuple[str, str, bool]:
    """Split one short-form compose volume entry into its three parts.

    Args:
        entry: A `source:target` or `source:target:mode` mapping string.

    Returns:
        A `(source, target, read_only)` triple; `read_only` is True only for
        an explicit `:ro` mode.
    """
    parts = entry.split(":")
    read_only = len(parts) > 2 and parts[2] == "ro"
    return parts[0], parts[1], read_only


def mounts(service: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """Return every volume mount a compose service declares.

    Args:
        service: One service's parsed compose mapping.

    Returns:
        One `(source, target, read_only)` triple per `volumes:` entry.
    """
    return [parse_mount(str(entry)) for entry in service.get("volumes", [])]


def unit_paths() -> list[Path]:
    """Return every shipped systemd unit file, sorted by name.

    Returns:
        The `deploy/systemd/*.service` paths.
    """
    return sorted(SYSTEMD_DIR.glob("*.service"))


def parse_unit(unit_path: Path) -> configparser.ConfigParser:
    """Parse a systemd unit file, preserving its case-sensitive key names.

    Args:
        unit_path: The unit file to read.

    Returns:
        A `ConfigParser` with `optionxform` disabled so `ExecStart` is not
        lowercased — systemd unit keys are case-sensitive.
    """
    parser = configparser.ConfigParser()
    parser.optionxform = str  # type: ignore[method-assign]
    parser.read(unit_path, encoding="utf-8")
    return parser


def unit_exec_start_tokens(parser: configparser.ConfigParser) -> list[str]:
    """Return a unit's `ExecStart` as an argument vector.

    Args:
        parser: A parsed unit from :func:`parse_unit`.

    Returns:
        The `ExecStart` line split with `shlex`.
    """
    return shlex.split(parser.get("Service", "ExecStart"))


def strip_env_shim(tokens: Sequence[str]) -> list[str]:
    """Drop a leading `/usr/bin/env` shim from an argument vector.

    Args:
        tokens: A command's tokens, possibly prefixed by the shim.

    Returns:
        The tokens with the shim removed, if it was there.
    """
    argv = list(tokens)
    if argv and argv[0] == ENV_SHIM:
        return argv[1:]
    return argv


def parse_run_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse a shipped command's arguments with windbreak's own parser.

    Args:
        argv: The tokens after the `windbreak` entrypoint, e.g.
            `["run", "--process", "pipeline", ...]`.

    Returns:
        The parsed namespace, exactly as the running process would see it.
    """
    return windbreak_main.build_parser().parse_args(list(argv))


def paper_activation_dests() -> frozenset[str]:
    """Derive the argument names :func:`_paper_activated` actually reads.

    Walks the AST of `windbreak.main._paper_activated` and collects every
    `args.<dest>` attribute access. Deriving the set means a rename, an
    addition or a removal in the activation rule reaches these tests as a
    changed set rather than as a stale hardcoded tuple (CLAUDE.md trap 8).

    Returns:
        The `argparse` destinations the activation rule gates on.
    """
    source = textwrap.dedent(inspect.getsource(windbreak_main._paper_activated))
    tree = ast.parse(source)
    return frozenset(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    )


def path_arguments(args: argparse.Namespace) -> dict[str, Path]:
    """Return every parsed argument the CLI itself typed as a filesystem path.

    Derived from the parser's own `type=Path` actions rather than from a list
    of flag names: whichever options the CLI declares as paths come back as
    `Path` instances, and everything else does not.

    Args:
        args: A namespace from :func:`parse_run_args`.

    Returns:
        A mapping of destination name to the supplied path.
    """
    return {
        dest: value for dest, value in vars(args).items() if isinstance(value, Path)
    }


def dockerfile_instructions() -> list[str]:
    """Return the Dockerfile's instructions, one per logical line.

    Backslash continuations are joined, so a multi-line `RUN a \\`/`&& b`
    reads as the single instruction Docker executes. Without that, a guard
    looking for `chown` in a `RUN` would miss it whenever the author wrapped
    the line — passing for a formatting reason rather than a real one.

    Returns:
        Each logical instruction, whitespace-collapsed, in file order.
    """
    text = DOCKERFILE_PATH.read_text(encoding="utf-8")
    instructions: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1].strip() + " "
            continue
        instructions.append(" ".join((pending + stripped).split()))
        pending = ""
    if pending:
        instructions.append(" ".join(pending.split()))
    return instructions


def dockerfile_instruction_lines(keyword: str) -> list[str]:
    """Return every Dockerfile instruction starting with `keyword`.

    Args:
        keyword: The Dockerfile instruction keyword to match (e.g. `USER`).

    Returns:
        Each matching logical instruction, in file order.
    """
    return [
        instruction
        for instruction in dockerfile_instructions()
        if instruction.split(maxsplit=1)[0].upper() == keyword
    ]


def dockerignore_patterns() -> list[str]:
    """Return the repo-root `.dockerignore`'s effective patterns.

    Returns:
        Each non-blank, non-comment line; an empty list when no
        `.dockerignore` exists.
    """
    if not DOCKERIGNORE_PATH.is_file():
        return []
    lines = DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
    return [
        stripped
        for stripped in (line.strip() for line in lines)
        if stripped and not stripped.startswith("#")
    ]


def dockerignore_exclusion(relative: Path) -> str | None:
    """Return the first `.dockerignore` pattern that would exclude `relative`.

    Approximates Docker's matcher: a pattern excludes a path when it fnmatches
    the path itself or any of its parent directories. Negation (`!`) patterns
    are ignored, so this can only ever **over**-report an exclusion — it fails
    closed, never open.

    Args:
        relative: A repo-relative path the build context must contain.

    Returns:
        The offending pattern, or `None` if nothing excludes the path.
    """
    candidates = [relative, *relative.parents]
    for pattern in dockerignore_patterns():
        if pattern.startswith("!"):
            continue
        needle = pattern.rstrip("/").lstrip("/")
        for candidate in candidates:
            if str(candidate) == "." or not fnmatch.fnmatch(str(candidate), needle):
                continue
            return pattern
    return None


def docker_compose_available() -> bool:
    """Probe whether `docker compose` (the v2 CLI plugin) is usable here.

    Returns:
        True if the `docker` binary exists and `docker compose version` exits
        zero; False otherwise (missing binary, or the plugin absent).
    """
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def run_docker_compose_config(
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `docker compose config` against the shipped compose file.

    Args:
        env: The complete environment to run under, or `None` to inherit.

    Returns:
        The completed process, with stdout/stderr captured as text.
    """
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_PATH), "config"],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
        env=env,
    )
