"""The container artifacts are BUILT and RUN, not read as text (issue #468).

`tests/deploy/test_dockerfile.py` opens the Dockerfile as a string and greps for
instruction lines; `tests/deploy/test_compose.py` parses the YAML and, where
docker exists, runs `docker compose config` -- a syntax check. Both stay green
against a Dockerfile whose `pip install .` fails, whose base image no longer
exists, or whose `CMD` names a flag the CLI rejects, and `config` is exactly
what let #445 ship: valid YAML declaring a build context that did not exist.

This module runs `docker build`, `docker compose up`, and `docker compose exec`
against the artifacts as committed. Every deployment claim #455 recorded as
"unverified in either direction" gets its own independently-named test, so a
failure report says *which* claim broke:

* the image builds, and its process is not root (1, 2, 3);
* the four services reach `running` and stay there (4);
* killing one never kills another -- ARCHITECTURE.md:11 (5);
* the dashboard is the only publisher, and only on loopback -- SPEC S5.1 (6);
* the shared ledger volume is writable by `pipeline` and read-only for
  `dashboard`, proven by attempting the write (7);
* `restart: on-failure` is the policy the daemon applied, and that policy does
  restart a genuinely failing container while `no` does not (8a, 8b).

ONE FINDING FELL OUT OF WRITING THIS. Issue #468 asks claim 8 to be proven by
killing a service and watching it come back. It cannot be: the daemon suppresses
the restart policy for `docker kill`, treating it as an operator's deliberate
stop. Measured, the container sits at `exited` code 137 with `RestartCount` 0
indefinitely. Written the obvious way the test would have failed; written with a
`>= 1` tolerance and a generous timeout it would have been a slow, permanent
red. Claim 5 depends on that same suppression, which is why 5 and 8 are split.

EVERY STACK TEST GETS ITS OWN STACK. Sharing one would be faster and wrong:
tests 5 and 8 kill services, so a shared stack would make the verdicts depend
on collection order -- the shape of regression guard PR #477 shipped inert
because `reap_all` pops LIFO. Independent stacks cost about 20s each and buy an
independent failure report.

SKIPPING IS NOT PASSING. `require_runtime` skips on a developer machine with no
daemon and FAILS in CI, where `WINDBREAK_E2E_REQUIRE_RUNTIME=1` is set. This
job is a required check under branch protection; a required check that skips
reports success while verifying nothing.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from typing import TYPE_CHECKING, Any

import pytest

from tests.deploy.artifacts import COMPOSE_PATH, DOCKERFILE_PATH, REPO_ROOT
from tests.e2e.harness import (
    docker_skip_reason,
    read_ledger_records,
    require_runtime,
    wait_until,
)
from windbreak.main import DASHBOARD_AUTH_ENV_VAR

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.container

#: Tag the tier builds the repo-root Dockerfile under. Namespaced so it can
#: never collide with an operator's own `windbreak` image.
IMAGE_TAG = "windbreak-e2e-container-tier:under-test"

#: Compose project name. Isolates this tier's containers, networks and volumes
#: from any stack a developer already has up.
COMPOSE_PROJECT = "windbreak-e2e-container-tier"

#: Environment variable the dashboard reads its bearer token from, taken from
#: the CLI rather than restated, so a rename cannot leave this tier passing
#: against a variable nothing reads.
DASHBOARD_TOKEN_ENV_VAR = DASHBOARD_AUTH_ENV_VAR

#: Bearer token the compose file's `:?` guard requires. A test value, never a
#: real credential: compose interpolates it into the dashboard's environment
#: only, and nothing here writes it to the hash-chained ledger.
DASHBOARD_TOKEN = "container-tier-token-not-a-secret"

#: The four services the compose file declares, derived nowhere else because
#: this is the claim under test: the stack that comes up must be exactly these.
EXPECTED_SERVICES = frozenset({"pipeline", "riskkernel", "order-gateway", "dashboard"})

#: Path the shared ledger volume is mounted at inside every service.
LEDGER_MOUNT = "/var/lib/windbreak/ledger"

#: Loopback address SPEC S14 requires every published port to bind.
LOOPBACK_HOST = "127.0.0.1"

#: Seconds allowed for `docker build`, including the base-image pull.
BUILD_TIMEOUT_SECONDS = 1800.0

#: Seconds allowed for `docker compose up -d`.
UP_TIMEOUT_SECONDS = 900.0

#: Seconds allowed for a short docker/compose command.
COMMAND_TIMEOUT_SECONDS = 180.0

#: Seconds allowed for every service to reach `running`.
SETTLE_TIMEOUT_SECONDS = 120.0

#: Seconds the stack must hold `running` before test 4 believes it.
#:
#: Not a guess at how long startup takes -- `wait_until` already handled that.
#: This is a soak: `restart: on-failure` makes a crash-looping service pass
#: through `running` on every cycle, so a single sample cannot tell a healthy
#: stack from a broken one.
STABILITY_SOAK_SECONDS = 5.0

#: Cadence of the soak's samples.
SOAK_INTERVAL_SECONDS = 0.25


def _compose_plugin_skip_reason() -> str | None:
    """Explain why the compose tier cannot run here, if it cannot.

    Returns:
        ``None`` when docker and its v2 compose plugin are both usable, else a
        reason naming the missing runtime.
    """
    docker_reason = docker_skip_reason()
    if docker_reason is not None:
        return docker_reason
    completed = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        return (
            "docker runtime unavailable: the v2 `docker compose` plugin is "
            f"absent (`docker compose version` exited {completed.returncode})"
        )
    return None


def _run(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a docker command from the repository root and capture its output.

    The dashboard token is supplied here rather than left to the ambient
    environment. `deploy/docker-compose.yml` requires it through compose's
    ``:?`` form, so a test that inherited it would pass or fail depending on
    the shell it was launched from -- and would report an infrastructure
    problem as a deployment defect.

    Args:
        argv: The full argument vector.
        timeout: Seconds to allow before the child is killed.

    Returns:
        The completed process, decoded as text.
    """
    env = dict(os.environ)
    env[DASHBOARD_TOKEN_ENV_VAR] = DASHBOARD_TOKEN
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )


def _compose(
    *args: str, timeout: float = COMMAND_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run `docker compose` against the shipped file, as README documents it.

    Args:
        *args: Arguments appended after the project/file selection.
        timeout: Seconds to allow before the child is killed.

    Returns:
        The completed process, decoded as text.
    """
    return _run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_PATH),
            "-p",
            COMPOSE_PROJECT,
            *args,
        ],
        timeout=timeout,
    )


def _compose_ps() -> list[dict[str, Any]]:
    """Read the stack's container inventory from docker, not from the YAML.

    Returns:
        One mapping per container, as ``docker compose ps --format json``
        reports it. Compose emits either a JSON array or newline-delimited
        objects depending on version; both are accepted.
    """
    completed = _compose("ps", "--all", "--format", "json")
    assert completed.returncode == 0, completed.stderr
    payload = completed.stdout.strip()
    if not payload:
        return []
    if payload.startswith("["):
        return list(json.loads(payload))
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _service_states() -> dict[str, str]:
    """Map each running service name to docker's reported state.

    Returns:
        Service name to state (``running``, ``restarting``, ``exited``, ...).
    """
    return {entry["Service"]: entry["State"] for entry in _compose_ps()}


def _container_ids() -> dict[str, str]:
    """Map each service name to its current container id.

    Returns:
        Service name to container id.
    """
    return {entry["Service"]: entry["ID"] for entry in _compose_ps()}


def _inspect(container: str, template: str) -> str:
    """Ask the daemon about a container, through a Go template.

    Args:
        container: Container id or name.
        template: The ``--format`` template to render.

    Returns:
        The rendered output, stripped.
    """
    completed = _run(
        ["docker", "inspect", "-f", template, container],
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _restart_counts() -> dict[str, int]:
    """Read every service's docker restart counter.

    Returns:
        Service name to the daemon's ``RestartCount`` for its container.
    """
    return {
        service: int(_inspect(container_id, "{{.RestartCount}}"))
        for service, container_id in _container_ids().items()
    }


def _assert_stable_for(
    predicate: object,
    *,
    seconds: float,
    description: str,
) -> None:
    """Assert a condition holds continuously for a bounded window.

    Distinct from :func:`~tests.e2e.harness.wait_until`, which waits for a
    condition to *become* true. Here the condition is already true and the
    question is whether it stays that way -- which is the only way to tell a
    healthy service from one `restart: on-failure` keeps resurrecting.

    Args:
        predicate: Zero-argument condition, sampled repeatedly.
        seconds: How long the condition must hold.
        description: What is being soaked, quoted in the failure message.

    Raises:
        AssertionError: If the predicate is ever false during the window.
    """
    checker = predicate
    assert callable(checker)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        assert checker(), f"{description} stopped holding during the soak"
        time.sleep(SOAK_INTERVAL_SECONDS)


@pytest.fixture(scope="session")
def _container_runtime() -> None:
    """Skip -- or, under CI's fail-closed flag, fail -- with no docker runtime."""
    require_runtime(_compose_plugin_skip_reason())


@pytest.fixture(scope="session")
def built_image(_container_runtime: None) -> str:
    """Build the repo-root Dockerfile from the repo root, exactly as committed.

    Session-scoped: the build is the expensive step, and every test below wants
    the same image. Compose rebuilds from the same context and Dockerfile, so
    its build is a layer-cache hit rather than a second real build.

    Args:
        _container_runtime: Gates the whole tier on a usable docker runtime.

    Returns:
        The tag the image was built under.
    """
    completed = _run(
        [
            "docker",
            "build",
            "--tag",
            IMAGE_TAG,
            "--file",
            str(DOCKERFILE_PATH),
            str(REPO_ROOT),
        ],
        timeout=BUILD_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0, (
        "`docker build` on the committed Dockerfile failed -- the text "
        "assertions in tests/deploy/test_dockerfile.py cannot see this:\n"
        f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
    )
    return IMAGE_TAG


@pytest.fixture
def compose_stack(built_image: str) -> Iterator[None]:
    """Bring the four-service stack up, and guarantee it comes down.

    ``down -v`` runs in a ``finally`` so a failing test still removes the
    containers, the network and the named volumes -- including when ``up``
    itself fails halfway.

    ``--build`` IS LOAD-BEARING, and README.md's bare ``up -d`` is not enough
    here. Compose only builds when the image is missing, so on a machine that
    already holds one this tier passed against a compose file with #445's
    broken ``context: .`` restored -- it never re-resolved the context, and the
    guard for the exact defect the tier exists to catch was inert. Caught by
    running the required proof-of-failure rather than assuming it.

    Args:
        built_image: Ensures the image exists before compose is invoked.

    Yields:
        ``None``, once every service has a container.
    """
    assert built_image == IMAGE_TAG
    _compose("down", "--volumes", "--remove-orphans", timeout=UP_TIMEOUT_SECONDS)
    try:
        started = _compose("up", "--detach", "--build", timeout=UP_TIMEOUT_SECONDS)
        assert started.returncode == 0, (
            "`docker compose up -d` failed on the committed compose file -- "
            "`docker compose config` cannot see this (issue #445):\n"
            f"{started.stdout[-4000:]}\n{started.stderr[-4000:]}"
        )
        yield
    finally:
        _compose("down", "--volumes", "--remove-orphans", timeout=UP_TIMEOUT_SECONDS)


def _wait_for_all_running() -> None:
    """Block until every expected service reports ``running``.

    Raises:
        AssertionError: If they do not all get there in time.
    """
    wait_until(
        lambda: all(
            _service_states().get(service) == "running" for service in EXPECTED_SERVICES
        ),
        timeout=SETTLE_TIMEOUT_SECONDS,
        description=f"all of {sorted(EXPECTED_SERVICES)} to report running",
    )


def test_docker_build_produces_an_image_running_the_committed_cmd(
    built_image: str,
) -> None:
    """`docker build` succeeds and the image's `CMD` is the committed one.

    Claim 1. The Dockerfile is executed rather than grepped: a base image that
    no longer exists, or a `pip install .` that fails, stops here -- and cannot
    stop `test_cmd_invokes_windbreak_run`, which reads the same file as text.

    Args:
        built_image: The tag the session build produced.
    """
    completed = _run(
        ["docker", "inspect", "-f", "{{json .Config.Cmd}}", built_image],
        timeout=COMMAND_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == ["windbreak", "run"]


def test_the_images_default_cmd_runs_a_beat_and_writes_the_mounted_ledger(
    built_image: str,
    tmp_path: Path,
) -> None:
    """Claim 2: the image's own `CMD`, plus `--max-beats 1`, ledgers a real row.

    The command is read back off the built image rather than restated here, so
    a Dockerfile that changes `CMD` cannot leave this test exercising the old
    one. The ledger is copied out of the container and read from the host, so
    the evidence is not something the container told us about itself; the whole
    mount directory is copied because `PRAGMA journal_mode=WAL` leaves fresh
    rows in the `-wal` sidecar, and a `.db`-only read produced a false green in
    PR #474.

    Args:
        built_image: The tag the session build produced.
        tmp_path: Per-test temporary directory for the copied-out ledger.
    """
    inspected = _run(
        ["docker", "inspect", "-f", "{{json .Config.Cmd}}", built_image],
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    assert inspected.returncode == 0, inspected.stderr
    image_cmd = json.loads(inspected.stdout)
    container_name = f"{COMPOSE_PROJECT}-cmd-probe"
    volume_name = f"{COMPOSE_PROJECT}-cmd-probe-ledger"
    _run(["docker", "rm", "--force", container_name], timeout=COMMAND_TIMEOUT_SECONDS)
    _run(
        ["docker", "volume", "rm", "--force", volume_name],
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    try:
        completed = _run(
            [
                "docker",
                "run",
                "--name",
                container_name,
                "--volume",
                f"{volume_name}:{LEDGER_MOUNT}",
                built_image,
                *image_cmd,
                "--max-beats",
                "1",
                "--heartbeat-interval",
                "0.1",
                "--ledger-path",
                f"{LEDGER_MOUNT}/windbreak.db",
            ],
            timeout=UP_TIMEOUT_SECONDS,
        )

        assert completed.returncode == 0, (
            f"the image's own CMD {image_cmd} failed inside the container:\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
        )
        copied = _run(
            [
                "docker",
                "cp",
                f"{container_name}:{LEDGER_MOUNT}/.",
                str(tmp_path),
            ],
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        assert copied.returncode == 0, copied.stderr
    finally:
        _run(
            ["docker", "rm", "--force", container_name],
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        _run(
            ["docker", "volume", "rm", "--force", volume_name],
            timeout=COMMAND_TIMEOUT_SECONDS,
        )

    event_types = [
        record.event_type for record in read_ledger_records(tmp_path / "windbreak.db")
    ]

    assert event_types == ["ConfigLoaded"]


def test_the_containers_main_process_is_not_root(compose_stack: None) -> None:
    """Claim 3: PID 1 inside a service runs as a non-root uid.

    `test_declares_a_non_root_user` proves a `USER` *line exists*. This proves
    it *took effect* for the process that is actually running, by reading the
    kernel's own view of PID 1's real uid rather than the image config.

    Args:
        compose_stack: The running four-service stack.
    """
    _wait_for_all_running()

    status = _compose("exec", "-T", "pipeline", "cat", "/proc/1/status")
    identity = _compose("exec", "-T", "pipeline", "id", "-u")

    assert status.returncode == 0, status.stderr
    assert identity.returncode == 0, identity.stderr
    uid_lines = [line for line in status.stdout.splitlines() if line.startswith("Uid:")]
    assert len(uid_lines) == 1, f"no Uid: line in /proc/1/status:\n{status.stdout}"
    real_uid = int(uid_lines[0].split()[1])

    assert real_uid != 0
    assert real_uid == int(identity.stdout.strip())


def test_compose_up_brings_all_four_services_to_running(
    compose_stack: None,
) -> None:
    """Claim 4: every declared service reaches `running` and stays there.

    The soak is what makes this mean something. `restart: on-failure` cycles a
    crash-looping container back through `running`, so one sample would report
    a green stack that is in fact dying every few seconds.

    Args:
        compose_stack: The running four-service stack.
    """
    _wait_for_all_running()

    observed = _service_states()

    assert set(observed) == set(EXPECTED_SERVICES), (
        f"the stack came up as {sorted(observed)}, not {sorted(EXPECTED_SERVICES)}"
    )
    assert set(observed.values()) == {"running"}
    _assert_stable_for(
        lambda: set(_service_states().values()) == {"running"},
        seconds=STABILITY_SOAK_SECONDS,
        description="all four services running",
    )
    assert set(_restart_counts().values()) == {0}


def test_killing_one_service_leaves_the_other_three_running(
    compose_stack: None,
) -> None:
    """Claim 5: process isolation, the transcript README.md:150-156 prints.

    ARCHITECTURE.md:11 -- "killing one process must never kill another". The
    kill is confirmed to have landed -- the victim reaches ``exited`` with
    SIGKILL's 137 -- before the siblings are judged, so a kill that silently
    did nothing cannot produce a green.

    Args:
        compose_stack: The running four-service stack.
    """
    _wait_for_all_running()
    victim = "pipeline"
    survivors = sorted(EXPECTED_SERVICES - {victim})
    before_ids = _container_ids()

    killed = _compose("kill", "--signal", "SIGKILL", victim)
    assert killed.returncode == 0, killed.stderr
    wait_until(
        lambda: _service_states().get(victim) == "exited",
        timeout=SETTLE_TIMEOUT_SECONDS,
        description=f"{victim} to be gone after SIGKILL",
    )
    assert _inspect(before_ids[victim], "{{.State.ExitCode}}") == str(
        128 + int(signal.SIGKILL)
    )

    after_states = _service_states()
    after_ids = _container_ids()
    after_counts = _restart_counts()

    assert [after_states[service] for service in survivors] == ["running"] * len(
        survivors
    )
    assert [after_ids[service] for service in survivors] == [
        before_ids[service] for service in survivors
    ]
    assert [after_counts[service] for service in survivors] == [0] * len(survivors)


def test_the_dashboard_is_the_only_publisher_and_binds_loopback_only(
    compose_stack: None,
) -> None:
    """Claim 6: one published port, on 127.0.0.1, on the dashboard (SPEC S5.1).

    Read from the daemon's port bindings, not from the compose file. A compose
    file can declare `127.0.0.1:8080:8080` and still be wrong about what the
    running stack exposes; only the daemon knows.

    Args:
        compose_stack: The running four-service stack.
    """
    _wait_for_all_running()

    inventory = _compose_ps()

    assert inventory, "docker reported no containers for a stack that is up"
    published: dict[str, list[dict[str, Any]]] = {}
    for entry in inventory:
        bindings = [
            publisher
            for publisher in (entry.get("Publishers") or [])
            if publisher.get("PublishedPort")
        ]
        if bindings:
            published[entry["Service"]] = bindings

    assert sorted(published) == ["dashboard"], (
        f"expected only the dashboard to publish a port, got {sorted(published)}"
    )
    hosts = {publisher.get("URL") for publisher in published["dashboard"]}
    assert hosts == {LOOPBACK_HOST}, (
        f"the dashboard publishes on {sorted(hosts)}, not {LOOPBACK_HOST} only"
    )


def test_the_ledger_volume_is_writable_by_pipeline_and_read_only_for_dashboard(
    compose_stack: None,
) -> None:
    """Claim 7: prove the `:ro` mount by attempting the write, not by re-reading.

    The pipeline half is the positive control. Without it a green dashboard
    result could equally mean the mount path does not exist, that `touch` is
    missing from the image, or that `exec` failed for an unrelated reason --
    the read-only mount would be credited for someone else's error.

    Args:
        compose_stack: The running four-service stack.
    """
    _wait_for_all_running()

    writable = _compose(
        "exec", "-T", "pipeline", "touch", f"{LEDGER_MOUNT}/.write-probe"
    )
    read_only = _compose(
        "exec", "-T", "dashboard", "touch", f"{LEDGER_MOUNT}/.write-probe-ro"
    )

    assert writable.returncode == 0, (
        "the pipeline could not write the shared ledger volume, so the "
        f"read-only assertion below would prove nothing:\n{writable.stderr}"
    )
    assert read_only.returncode != 0
    assert "Read-only file system" in read_only.stderr


def test_every_service_carries_the_on_failure_restart_policy(
    compose_stack: None,
) -> None:
    """Claim 8a: the daemon applied `on-failure` to all four containers.

    Read back from `docker inspect`, which is the daemon's own record of the
    policy in force -- not from `deploy/docker-compose.yml`, which is the text
    this issue exists to stop trusting.

    Args:
        compose_stack: The running four-service stack.
    """
    _wait_for_all_running()

    policies = {
        service: _inspect(container_id, "{{.HostConfig.RestartPolicy.Name}}")
        for service, container_id in _container_ids().items()
    }

    assert set(policies) == set(EXPECTED_SERVICES)
    assert set(policies.values()) == {"on-failure"}


def test_the_on_failure_policy_restarts_a_non_zero_exit_and_no_policy_does_not(
    built_image: str,
    compose_stack: None,
) -> None:
    """Claim 8b: `on-failure` really restarts a failing container; `no` does not.

    WHY THIS IS NOT `docker compose kill`. The obvious test -- SIGKILL a
    service and watch it come back -- cannot work, and quietly proves nothing
    if written anyway. The daemon treats `docker kill` as an operator's
    deliberate stop and suppresses the restart policy entirely: measured here,
    the container reaches `exited` with code 137 and `RestartCount` stays 0
    forever. `test_killing_one_service_leaves_the_other_three_running` relies
    on exactly that, which is why the two claims are separate tests.

    So the policy is exercised on the image the stack runs, with the policy
    name read off a real stack container, against a genuinely failing command:
    an unknown `--process` token argparse rejects. Two containers run that
    identical argv and differ in exactly one variable, the restart policy, so
    the divergence in outcome is attributable to it -- without the control a
    climbing counter could be anything the daemon does to crashing containers.

    The failing command's exit code is asserted on the CONTROL, not the
    subject: the subject is being restarted, and a container the daemon has
    already brought back reports ``State.ExitCode`` 0 because that field
    describes the *current* incarnation. Sampling it on the subject read as a
    clean exit and would have inverted this test's meaning.

    Args:
        built_image: The tag the session build produced.
        compose_stack: Supplies a real service whose policy name is copied.
    """
    _wait_for_all_running()
    shipped_policy = _inspect(
        _container_ids()["riskkernel"], "{{.HostConfig.RestartPolicy.Name}}"
    )
    failing_argv = ["windbreak", "run", "--process", "not-a-real-process"]
    names = {
        shipped_policy: f"{COMPOSE_PROJECT}-restart-subject",
        "no": f"{COMPOSE_PROJECT}-restart-control",
    }
    try:
        for policy, name in names.items():
            _run(["docker", "rm", "--force", name], timeout=COMMAND_TIMEOUT_SECONDS)
            started = _run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--restart",
                    policy,
                    "--name",
                    name,
                    built_image,
                    *failing_argv,
                ],
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
            assert started.returncode == 0, started.stderr
        subject = names[shipped_policy]
        control = names["no"]
        wait_until(
            lambda: int(_inspect(subject, "{{.RestartCount}}")) >= 1,
            timeout=SETTLE_TIMEOUT_SECONDS,
            description=(
                f"the shipped `{shipped_policy}` policy to restart a container "
                "whose command exits non-zero"
            ),
        )

        wait_until(
            lambda: _inspect(control, "{{.State.Status}}") == "exited",
            timeout=SETTLE_TIMEOUT_SECONDS,
            description="the `no`-policy control container to exit",
        )

        assert _inspect(control, "{{.State.ExitCode}}") != "0"
        _assert_stable_for(
            lambda: (
                _inspect(control, "{{.State.Status}}") == "exited"
                and int(_inspect(control, "{{.RestartCount}}")) == 0
            ),
            seconds=STABILITY_SOAK_SECONDS,
            description="the `no`-policy control staying dead with 0 restarts",
        )
    finally:
        for name in names.values():
            _run(["docker", "rm", "--force", name], timeout=COMMAND_TIMEOUT_SECONDS)
