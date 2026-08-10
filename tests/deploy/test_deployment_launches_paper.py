"""The shipped deployment must be able to build and to activate PAPER.

Two P0 defects share one artifact surface (issues #445 and #446):

* **#445** — every compose service declared `build: .`. Compose resolves a
  relative context against the *compose file's* directory, so the context was
  `deploy/`, which holds no `Dockerfile`; `up -d` failed before creating a
  container.
* **#446** — no shipped command supplied the four PAPER composition flags, so
  `windbreak.main._resolve_on_beat` returned `None` and every service was a
  bare RESEARCH heartbeat writing nothing to the ledger volume it mounted.

Both are "a deployment file that is merely edited is indistinguishable from one
edited wrongly" defects, so every test below reads the **real** artifact
(`tests/deploy/artifacts.py` parses them) and derives its expectations from
production code — the activation rule's own AST, the CLI's own parser, the
Dockerfile's own `USER` — rather than restating a copy that can drift.

Nothing here claims the deployment *trades*. Bringing the stack up shows the
shipped `deep_walk` fixture's only market screened ineligible every tick
(`blocked_by: ["min_depth_contract_centis", "horizon_days"]`), so the pipeline
never reaches forecasting; behind that, cassette-mode research transports find
nothing by construction (issue #438). These tests pin activation and
durability, not profit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from tests.deploy import artifacts
from windbreak import main as windbreak_main
from windbreak.config import load_default_config

if TYPE_CHECKING:
    import argparse

#: The compose service and systemd unit that carry the PAPER loop.
_PIPELINE_SERVICE = "pipeline"
_PIPELINE_UNIT = "windbreak-pipeline.service"
_DASHBOARD_SERVICE = "dashboard"
_DASHBOARD_UNIT = "windbreak-dashboard.service"

#: A non-secret placeholder for the dashboard token, used only so Compose's
#: required-variable interpolation can succeed while a config is rendered.
_TOKEN_PLACEHOLDER = "compose-config-probe"


def _service_argv(service_name: str) -> list[str]:
    """Return one compose service's argument vector, entrypoint included.

    Args:
        service_name: The compose service to read.

    Returns:
        The service's `command` tokens.
    """
    return artifacts.command_tokens(artifacts.compose_services()[service_name])


def _unit_argv(unit_filename: str) -> list[str]:
    """Return one systemd unit's argument vector, entrypoint included.

    Args:
        unit_filename: The unit file's name under `deploy/systemd/`.

    Returns:
        The unit's `ExecStart` tokens with the `/usr/bin/env` shim removed.
    """
    parser = artifacts.parse_unit(artifacts.SYSTEMD_DIR / unit_filename)
    return artifacts.strip_env_shim(artifacts.unit_exec_start_tokens(parser))


def _parsed(argv: list[str]) -> argparse.Namespace:
    """Parse an argument vector whose first token is the console script.

    Args:
        argv: Tokens starting with the `windbreak` entrypoint.

    Returns:
        The namespace the running process would see.
    """
    assert argv[0] == artifacts.ENTRYPOINT, f"command does not launch windbreak: {argv}"
    return artifacts.parse_run_args(argv[1:])


def _activates_paper(argv: list[str]) -> bool:
    """Ask production code whether this exact command activates PAPER.

    Args:
        argv: Tokens starting with the `windbreak` entrypoint.

    Returns:
        `windbreak.main._paper_activated`'s verdict for the default config.
    """
    return windbreak_main._paper_activated(load_default_config(), _parsed(argv))


def _mount_containing(
    target_paths: list[tuple[str, str, bool]], path: Path
) -> tuple[str, str, bool] | None:
    """Return the mount whose target contains `path`, if any.

    Args:
        target_paths: `(source, target, read_only)` triples for one service.
        path: An absolute path the service's command names.

    Returns:
        The matching triple, or `None` when no mount covers the path.
    """
    for mount in target_paths:
        target = Path(mount[1])
        if path == target or target in path.parents:
            return mount
    return None


# --------------------------------------------------------------------------
# Issue #445: the build context must contain the Dockerfile.
# --------------------------------------------------------------------------


def test_every_service_build_context_contains_the_dockerfile() -> None:
    """Each service's build context resolves to a directory holding a Dockerfile.

    This is issue #445 stated as an invariant rather than as a fix: the
    resolution is performed here exactly as Compose performs it (relative to
    the compose file's own directory), so `build: .` — which resolves to
    `deploy/` — fails this test.
    """
    services = artifacts.compose_services()
    assert services, "the compose file defines no services at all"

    specs = {
        name: artifacts.build_spec(service)
        for name, service in services.items()
        if artifacts.build_spec(service) is not None
    }
    assert specs, "no service declares a build context to check"

    for name, spec in sorted(specs.items()):
        assert spec is not None
        dockerfile = artifacts.resolved_dockerfile(spec)
        assert dockerfile.is_file(), (
            f"service {name!r} builds from {spec['context']!r}, which resolves "
            f"to {dockerfile.parent} — no dockerfile at {dockerfile}"
        )


@pytest.mark.skipif(
    not artifacts.docker_compose_available(),
    reason="docker CLI or the docker compose v2 plugin is unavailable here",
)
@pytest.mark.timeout(90)
def test_docker_compose_config_resolves_context_to_a_dockerfile() -> None:
    """Compose itself agrees the resolved build context holds a Dockerfile.

    The structural test above computes the resolution; this one asks the real
    binary, so a change in Compose's own resolution rule is caught too.
    """
    env = {**os.environ, windbreak_main.DASHBOARD_AUTH_ENV_VAR: _TOKEN_PLACEHOLDER}
    result = artifacts.run_docker_compose_config(env=env)
    assert result.returncode == 0, result.stderr

    rendered = yaml.safe_load(result.stdout)
    builds = {
        name: service["build"]
        for name, service in rendered["services"].items()
        if "build" in service
    }
    assert builds, "docker compose config rendered no build sections"

    for name, build in sorted(builds.items()):
        dockerfile = Path(build["context"]) / build.get(
            "dockerfile", artifacts.DEFAULT_DOCKERFILE_NAME
        )
        assert dockerfile.is_file(), f"{name}: no dockerfile at {dockerfile}"


# --------------------------------------------------------------------------
# Issue #446: the shipped commands must activate the PAPER loop.
# --------------------------------------------------------------------------


def test_paper_activation_rule_names_only_real_parser_destinations() -> None:
    """The AST-derived activation argument set is non-empty and CLI-real.

    Every activation test below iterates this set, so an empty or bogus
    derivation would make them pass vacuously (CLAUDE.md trap 7). Asserting
    the set is non-empty *and* a subset of the parser's own destinations is
    what makes those iterations mean something.
    """
    dests = artifacts.paper_activation_dests()
    assert dests, "no args.<dest> access found in _paper_activated"

    parser_dests = set(vars(artifacts.parse_run_args(["run"])))
    assert dests <= parser_dests, f"not run-command arguments: {dests - parser_dests}"


def test_pipeline_compose_command_activates_the_paper_loop() -> None:
    """The `pipeline` service's command makes production code activate PAPER.

    Asks `_paper_activated` itself rather than counting flags, so the test
    tracks the activation rule instead of a snapshot of it.
    """
    assert _activates_paper(_service_argv(_PIPELINE_SERVICE)) is True


def test_pipeline_systemd_unit_activates_the_paper_loop() -> None:
    """The pipeline unit's `ExecStart` makes production code activate PAPER."""
    assert _activates_paper(_unit_argv(_PIPELINE_UNIT)) is True


@pytest.mark.parametrize("dest", sorted(artifacts.paper_activation_dests()))
def test_pipeline_compose_command_supplies_every_activation_argument(
    dest: str,
) -> None:
    """Each argument the activation rule reads is supplied by the compose command.

    Redundant with the activation test above by design: this one fails with
    the *name* of the missing argument, so a regression says which flag was
    dropped rather than only that PAPER went dark.
    """
    args = _parsed(_service_argv(_PIPELINE_SERVICE))

    assert getattr(args, dest) is not None, f"pipeline service supplies no {dest}"


@pytest.mark.parametrize("dest", sorted(artifacts.paper_activation_dests()))
def test_pipeline_systemd_unit_supplies_every_activation_argument(dest: str) -> None:
    """Each argument the activation rule reads is supplied by the pipeline unit."""
    args = _parsed(_unit_argv(_PIPELINE_UNIT))

    assert getattr(args, dest) is not None, f"pipeline unit supplies no {dest}"


# --------------------------------------------------------------------------
# The paths those flags name must actually exist where the service will look.
# --------------------------------------------------------------------------


def _relative_path_arguments() -> list[tuple[str, str, Path]]:
    """Collect every relative path argument across compose and systemd.

    Returns:
        `(artifact, dest, path)` triples, where a relative path is resolved
        against the image `WORKDIR` (compose) or `WorkingDirectory` (systemd)
        — in both cases a copy of this repository.
    """
    found: list[tuple[str, str, Path]] = []
    for name, service in sorted(artifacts.compose_services().items()):
        args = _parsed(artifacts.command_tokens(service))
        found.extend(
            (f"compose:{name}", dest, value)
            for dest, value in sorted(artifacts.path_arguments(args).items())
            if not value.is_absolute()
        )
    for unit_path in artifacts.unit_paths():
        args = _parsed(_unit_argv(unit_path.name))
        found.extend(
            (f"systemd:{unit_path.name}", dest, value)
            for dest, value in sorted(artifacts.path_arguments(args).items())
            if not value.is_absolute()
        )
    return found


def test_every_relative_path_argument_exists_in_the_repository() -> None:
    """Relative path flags name real repo paths, so the copied tree serves them.

    Both skeletons run out of a copy of this repository — the image's
    `WORKDIR` for compose, the unit's `WorkingDirectory` for systemd — so a
    relative flag value is a repo path, and a typo in one is a container that
    starts and then dies on a missing fixture.
    """
    relative = _relative_path_arguments()
    assert relative, "no shipped command passes a repo-relative path argument"

    for artifact, dest, value in relative:
        assert (artifacts.REPO_ROOT / value).exists(), (
            f"{artifact} passes {dest}={value}, which does not exist in the repo"
        )


def test_no_dockerignore_pattern_excludes_a_relative_path_argument() -> None:
    """`.dockerignore` never drops a path the compose commands depend on.

    The build context became the whole repo in issue #445's fix, which makes
    `.dockerignore` load-bearing: a pattern that excludes the paper fixtures
    would leave the repo-side check above green while the *image* lacks them.
    """
    relative = _relative_path_arguments()
    assert relative, "no shipped command passes a repo-relative path argument"

    for artifact, dest, value in relative:
        pattern = artifacts.dockerignore_exclusion(value)
        assert pattern is None, (
            f"{artifact} passes {dest}={value}, excluded by .dockerignore "
            f"pattern {pattern!r}"
        )


def _absolute_compose_path_arguments() -> list[tuple[str, str, Path]]:
    """Collect every absolute path argument the compose services pass.

    Returns:
        `(service, dest, path)` triples.
    """
    found: list[tuple[str, str, Path]] = []
    for name, service in sorted(artifacts.compose_services().items()):
        args = _parsed(artifacts.command_tokens(service))
        found.extend(
            (name, dest, value)
            for dest, value in sorted(artifacts.path_arguments(args).items())
            if value.is_absolute()
        )
    return found


def test_every_absolute_path_argument_lands_on_a_declared_volume() -> None:
    """State a service writes is under one of its own volume mounts.

    An absolute path with no volume behind it is written into the container's
    writable layer and vanishes on the next `restart:` — the exact "alive and
    useless" shape issue #446 is about, only quieter.
    """
    absolute = _absolute_compose_path_arguments()
    assert absolute, "no compose command passes an absolute path argument"

    services = artifacts.compose_services()
    declared = set(artifacts.load_compose().get("volumes", {}))
    for name, dest, value in absolute:
        mount = _mount_containing(artifacts.mounts(services[name]), value)
        assert mount is not None, (
            f"{name} passes {dest}={value} with no volume behind it"
        )
        assert mount[0] in declared, (
            f"{name} mounts {mount[0]!r} at {mount[1]}, which is not a declared "
            f"top-level volume"
        )


def test_no_path_argument_resolves_under_a_read_only_mount() -> None:
    """A path flag under a `:ro` mount is a crash, not a restriction.

    `SqliteLedgerStore.__init__` runs `PRAGMA journal_mode=WAL` and a
    `CREATE TABLE IF NOT EXISTS` on open, and `_load_and_ledger_config`
    (`windbreak/main.py`) opens the ledger for *any* process given
    `--ledger-path`. Pointing such a flag at a read-only mount raises
    `sqlite3.OperationalError: unable to open database file` before the
    process serves anything — under `restart: on-failure`, forever.
    """
    absolute = _absolute_compose_path_arguments()
    assert absolute, "no compose command passes an absolute path argument"

    services = artifacts.compose_services()
    checked = 0
    for name, dest, value in absolute:
        mount = _mount_containing(artifacts.mounts(services[name]), value)
        assert mount is not None
        checked += 1
        assert not mount[2], (
            f"{name} passes {dest}={value} under read-only mount {mount[1]}; "
            f"the ledger store opens read-write, so this cannot start"
        )
    assert checked == len(absolute)


def test_a_unit_passing_a_relative_path_declares_a_working_directory() -> None:
    """Relative flags in a unit only resolve if `WorkingDirectory` is set.

    systemd's default working directory is `/`, so a unit that passes a
    repo-relative fixture path without `WorkingDirectory=` looks correct and
    fails at startup.
    """
    units_with_relative_paths = []
    for unit_path in artifacts.unit_paths():
        args = _parsed(_unit_argv(unit_path.name))
        if any(
            not value.is_absolute() for value in artifacts.path_arguments(args).values()
        ):
            units_with_relative_paths.append(unit_path)
    assert units_with_relative_paths, "no unit passes a repo-relative path argument"

    for unit_path in units_with_relative_paths:
        parser = artifacts.parse_unit(unit_path)
        assert parser.has_option("Service", "WorkingDirectory"), (
            f"{unit_path.name} passes a relative path but sets no WorkingDirectory"
        )


def _unit_state_path_arguments() -> list[tuple[str, str, Path]]:
    """Collect every unit path argument that lives under systemd's state root.

    Returns:
        `(unit filename, dest, path)` triples, derived by parsing each real
        `ExecStart=` with the CLI's own parser — never from a restated list of
        paths, so a unit that starts writing somewhere new is checked too.
    """
    found: list[tuple[str, str, Path]] = []
    for unit_path in artifacts.unit_paths():
        args = _parsed(_unit_argv(unit_path.name))
        found.extend(
            (unit_path.name, dest, value)
            for dest, value in sorted(artifacts.path_arguments(args).items())
            if artifacts.STATE_DIRECTORY_ROOT in value.parents
        )
    return found


def test_every_unit_state_path_is_provisioned_by_the_unit_that_writes_it() -> None:
    """A unit that writes under `/var/lib` also creates the directory it needs.

    Nothing else does. `docs/RUNBOOK.md` is explicit that `--ledger-path`'s
    parent directory must exist — the ledger file and its `-wal` sibling are
    created on first use, the directory is not — and `SqliteLedgerStore`
    raises `sqlite3.OperationalError: unable to open database file` on the
    very first config load when it is absent. Under `Restart=on-failure` with
    `StartLimitIntervalSec=0`, that is not a unit that lands in `failed`; it
    is one that retries forever while writing nothing, which is issue #446's
    defect moved from the compose path onto the systemd path.

    The directory is accepted as provisioned when the unit names it or its
    immediate parent (`--report-dir X` names a directory, `--ledger-path
    X/y.db` names a file inside one), so the check needs no hardcoded notion
    of which flags are files.
    """
    state_paths = _unit_state_path_arguments()
    assert state_paths, (
        f"no shipped unit passes a path under {artifacts.STATE_DIRECTORY_ROOT}, "
        f"so this test would pass vacuously"
    )

    for unit_filename, dest, value in state_paths:
        provisioned = artifacts.unit_provisioned_directories(
            artifacts.SYSTEMD_DIR / unit_filename
        )
        assert value in provisioned or value.parent in provisioned, (
            f"{unit_filename} passes {dest}={value} but provisions only "
            f"{sorted(str(path) for path in provisioned)}; nothing creates the "
            f"directory, so the unit fails on every start"
        )


# --------------------------------------------------------------------------
# Issue #446 consequence 3: the dashboard cannot start without its token.
# --------------------------------------------------------------------------


def test_dashboard_service_declares_the_token_variable_without_a_literal() -> None:
    """The dashboard service declares its token variable and never inlines one.

    The variable name is read from `windbreak.main.DASHBOARD_AUTH_ENV_VAR`, so
    renaming the constant reaches this test. The value must be a Compose
    interpolation of the *same* variable: a literal here would be a secret in
    a committed file, and the ledger is a hash chain nothing can be redacted
    from once it is written.
    """
    variable = windbreak_main.DASHBOARD_AUTH_ENV_VAR
    environment = artifacts.compose_services()[_DASHBOARD_SERVICE].get("environment")
    assert environment, f"{_DASHBOARD_SERVICE} declares no environment at all"

    if isinstance(environment, list):
        entries = dict(str(item).split("=", 1) for item in environment)
    else:
        entries = {str(k): str(v) for k, v in environment.items()}

    assert variable in entries, f"{_DASHBOARD_SERVICE} does not declare {variable}"
    value = entries[variable]
    assert value.startswith(f"${{{variable}"), (
        f"{variable} must interpolate from the operator's environment, got {value!r}"
    )
    assert ":?" in value, (
        f"{variable} must use Compose's required-variable form (`:?`) so an "
        f"unset token fails before any container is created"
    )


@pytest.mark.skipif(
    not artifacts.docker_compose_available(),
    reason="docker CLI or the docker compose v2 plugin is unavailable here",
)
@pytest.mark.timeout(90)
def test_docker_compose_config_fails_closed_without_the_dashboard_token() -> None:
    """Compose refuses to render the stack when the token variable is unset.

    Fail-closed at the operator's shell: without this, `up -d` creates a
    dashboard that exits 1 on the missing token and then crash-loops under
    `restart: on-failure`, which reads as "deployed" in `ps -a`.
    """
    variable = windbreak_main.DASHBOARD_AUTH_ENV_VAR
    env = {k: v for k, v in os.environ.items() if k != variable}

    result = artifacts.run_docker_compose_config(env=env)

    assert result.returncode != 0, result.stdout
    assert variable in result.stderr


def test_dashboard_unit_names_the_token_variable_and_requires_its_file() -> None:
    """The dashboard unit loads a mandatory env file and names the variable.

    `EnvironmentFile=` without a leading `-` is systemd's fail-closed form:
    a missing file fails the unit instead of starting a dashboard with no
    token. The unit must also spell the assignment the file has to carry —
    `WINDBREAK_DASHBOARD_TOKEN=...` — because an `EnvironmentFile` names a
    path and nothing else, so the file's required contents are otherwise
    undiscoverable from the unit. Merely mentioning the variable in prose is
    not enough: the operator needs the line to write.
    """
    variable = windbreak_main.DASHBOARD_AUTH_ENV_VAR
    unit_path = artifacts.SYSTEMD_DIR / _DASHBOARD_UNIT
    parser = artifacts.parse_unit(unit_path)

    assert parser.has_option("Service", "EnvironmentFile"), (
        f"{_DASHBOARD_UNIT} declares no EnvironmentFile, so the dashboard has "
        f"no way to receive its token"
    )
    environment_file = parser.get("Service", "EnvironmentFile")
    assert not environment_file.startswith("-"), (
        f"{_DASHBOARD_UNIT} marks its EnvironmentFile optional; a missing file "
        f"would start a tokenless dashboard"
    )
    assignments = [
        line
        for line in unit_path.read_text(encoding="utf-8").splitlines()
        if f"{variable}=" in line
    ]
    assert assignments, (
        f"{_DASHBOARD_UNIT} never spells the `{variable}=<token>` line its "
        f"EnvironmentFile {environment_file} must contain"
    )


# --------------------------------------------------------------------------
# The image must let its non-root user write the volumes it is given.
# --------------------------------------------------------------------------


def _image_user() -> str:
    """Return the user the image finally switches to.

    Returns:
        The argument of the last `USER` directive in the Dockerfile.
    """
    user_lines = artifacts.dockerfile_instruction_lines("USER")
    assert user_lines, "Dockerfile has no USER directive"
    return user_lines[-1].split(maxsplit=1)[1].strip()


def _chown_covers(line: str, target: str) -> bool:
    """Report whether a `chown` line covers `target` or an ancestor of it.

    Args:
        line: One Dockerfile `RUN` line mentioning `chown`.
        target: An absolute container path a volume is mounted at.

    Returns:
        True when the line names the target or one of its parent directories.
    """
    candidates = [Path(target), *Path(target).parents]
    return any(str(candidate) in line for candidate in candidates)


def test_every_named_volume_target_is_precreated_for_the_image_user() -> None:
    """The image creates and owns each volume mount point before dropping root.

    Docker seeds a fresh named volume from the image's directory at that path.
    If the image has no such directory, Docker creates it `root`-owned, and the
    image's non-root `USER` cannot write it — so the ledger volume stays empty
    for a *second* reason beyond issue #446's missing flags.
    """
    services = artifacts.compose_services()
    declared = set(artifacts.load_compose().get("volumes", {}))
    targets = {
        mount[1]
        for service in services.values()
        for mount in artifacts.mounts(service)
        if mount[0] in declared
    }
    assert targets, "the compose file mounts no named volumes"

    user = _image_user()
    run_lines = artifacts.dockerfile_instruction_lines("RUN")
    mkdir_lines = [line for line in run_lines if "mkdir" in line]
    chown_lines = [line for line in run_lines if "chown" in line]
    assert mkdir_lines, "Dockerfile never creates a directory"
    assert chown_lines, "Dockerfile never chowns anything"

    for target in sorted(targets):
        assert any(target in line for line in mkdir_lines), (
            f"volume mount point {target} is never created in the image, so "
            f"Docker will seed it root-owned"
        )
        assert any(
            user in line and _chown_covers(line, target) for line in chown_lines
        ), f"volume mount point {target} is never chowned to {user!r}"


# --------------------------------------------------------------------------
# Issue #446: a fast-failing unit must be retried, not abandoned.
# --------------------------------------------------------------------------


def _restart_seconds(value: str) -> int:
    """Parse a systemd time value expressed in whole seconds.

    Args:
        value: A `RestartSec=` value such as `5` or `5s`.

    Returns:
        The delay in seconds.
    """
    return int(value.removesuffix("s"))


@pytest.mark.parametrize("unit_path", artifacts.unit_paths(), ids=lambda p: p.name)
def test_every_unit_retries_instead_of_tripping_the_start_rate_limit(
    unit_path: Path,
) -> None:
    """Units back off and are never rate-limited into a permanent `failed`.

    `Restart=on-failure` with no `RestartSec` retries immediately; systemd's
    default limit of 5 starts per 10s then gives up and leaves the unit
    `failed` — unattended, which is the whole point of the deployment.
    """
    parser = artifacts.parse_unit(unit_path)

    assert parser.has_option("Service", "RestartSec"), (
        f"{unit_path.name} restarts immediately, so five failures in ten "
        f"seconds leave it permanently failed"
    )
    assert _restart_seconds(parser.get("Service", "RestartSec")) > 0
    assert parser.get("Unit", "StartLimitIntervalSec") == "0", (
        f"{unit_path.name} keeps systemd's default start rate limit, which "
        f"abandons a fast-failing process"
    )
