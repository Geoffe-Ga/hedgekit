"""The committed hermetic PAPER demonstration, named once (issues #510, #473).

PR #522 made ``windbreak run`` reach a real order intent without a network, out
of committed fixtures alone: a books directory whose calendar is re-enacted with
its books, a research/vote corpus selected from configuration, a correlation
declaration, and the committed M6 track-record artifact. Three test tiers now
need that same composition -- the integration tier that pins what the CLI
forecasts, the process-topology tier, and the unattended-run tier -- so the
paths and the argument vector live here rather than in whichever module happened
to need them first.

Nothing here relaxes a production threshold, and that is asserted rather than
promised:
``tests/integration/test_shipped_cli_hermetic_forecast.py::\
test_the_hermetic_configuration_moves_no_threshold`` diffs the committed
configuration against :class:`~windbreak.config.schema.WindbreakConfig` field by
field, and
``tests/e2e/test_unattended_run.py::\
test_the_unattended_configuration_moves_no_threshold`` does the same for the
configuration :func:`write_run_config` generates.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

#: The repository root, resolved from this file rather than the process's cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The committed books fixture the hermetic demonstration replays, verbatim.
DEMO_BOOKS = REPO_ROOT / "tests" / "fixtures" / "books" / "hermetic_demo"

#: The committed configuration that selects the corpus and declares the bucket.
DEMO_CONFIG = REPO_ROOT / "tests" / "fixtures" / "config" / "hermetic-demo.yaml"

#: The committed research/vote corpus that configuration names.
DEMO_CORPUS = REPO_ROOT / "tests" / "fixtures" / "forecast" / "hermetic_corpus"

#: The committed vote cassette the shipped command lines still name. A corpus
#: run never reads it; it is passed because ``--cassette-path`` is one of the
#: four flags that activate the PAPER loop at all.
SHIPPED_CASSETTE = REPO_ROOT / "tests" / "fixtures" / "forecast" / "cassettes.json"

#: The committed M6 evaluation artifact ``windbreak evaluate-providers`` writes
#: and the loop's live-eligibility gate reads out of ``--report-dir``.
DEMO_TRACK_RECORDS = (
    REPO_ROOT / "tests" / "fixtures" / "evaluation" / "provider-track-records.json"
)

#: That artifact's filename inside a run's ``--report-dir``.
TRACK_RECORD_FILENAME = "provider-track-records.json"

#: The demonstration fixture's sole market.
TICKER = "MKT-DEMO"


def place_track_records(report_dir: Path) -> Path:
    """Copy the committed M6 artifact into a run's report directory.

    The artifact is a *fixture*, and it clears the shipped provider-gate bars
    rather than lowering them --
    ``tests/integration/test_shipped_cli_hermetic_forecast.py::\
test_the_provider_gate_bars_are_the_shipped_ones`` reads both bars off
    :class:`~windbreak.config.schema.ProviderGateConfig` and compares.

    Args:
        report_dir: The run's ``--report-dir``; created when absent.

    Returns:
        The path the artifact was placed at.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    destination = report_dir / TRACK_RECORD_FILENAME
    shutil.copy(DEMO_TRACK_RECORDS, destination)
    return destination


def write_run_config(
    path: Path,
    *,
    state_dir: Path,
    alert_url_env: str | None = None,
    allowed_hosts: Iterable[str] = (),
) -> Path:
    """Write the committed demonstration's declarations, plus a run's own dirs.

    Three differences from the committed ``hermetic-demo.yaml``, none of them a
    threshold:

    * ``corpus_dir`` is written **absolute**. The committed file's value is
      repo-relative and resolves against the process's working directory, which
      a spawned child does not necessarily share with the test runner.
    * ``ops.state_dir`` names this run's own directory. The shipped default is
      ``~/.local/share/windbreak``, and a ``KILL`` file a developer left there
      would silently kill every run in this tier.
    * ``alerts`` is declared when a sink is asked for, so an escalation reaches
      a *configured* destination instead of the dispatcher's log-only fallback
      -- the exact distinction issue #444 turned on.

    Args:
        path: The file to write.
        state_dir: The value for ``ops.state_dir``.
        alert_url_env: The environment variable a ``webhook`` alert sink reads
            its destination from, or ``None`` to declare no ``alerts`` section.
        allowed_hosts: The hosts ``alerts.allowed_hosts`` declares. A sink whose
            resolved host is absent from this list refuses at composition, so it
            is stated separately from the destination on purpose (issue #274).

    Returns:
        The path that was written.
    """
    lines = [
        "forecast:",
        "  replay_corpus:",
        '    mode: "replay"',
        f'    corpus_dir: "{DEMO_CORPUS}"',
        "correlation:",
        "  tags:",
        f"    - ticker: {TICKER}",
        "      bucket_ids:",
        "        - fed-policy",
        '      tagged_at: "2025-01-01T00:00:00+00:00"',
        "ops:",
        f"  state_dir: {state_dir}",
    ]
    if alert_url_env is not None:
        lines += [
            "alerts:",
            "  sinks:",
            "    - type: webhook",
            f"      url_env: {alert_url_env}",
            "  allowed_hosts:",
        ]
        lines += [f"    - {host}" for host in allowed_hosts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def demo_run_args(
    *,
    config: Path,
    ledger_path: Path,
    report_dir: Path,
    max_beats: str,
    heartbeat_interval: str,
    research_per_day_micros: str | None = None,
) -> tuple[str, ...]:
    """Build the ``windbreak run`` arguments the demonstration is driven by.

    The argument vector ``deploy/docker-compose.yml`` and
    ``deploy/systemd/windbreak-pipeline.service`` invoke, narrowed by the two
    bounding flags the product already ships: ``--max-beats`` and
    ``--heartbeat-interval``. Neither is a test-only clock -- compressing a
    run through them is compressing it through the product.

    Args:
        config: The ``--config`` file, normally from :func:`write_run_config`.
        ledger_path: The ``--ledger-path`` this run appends to.
        report_dir: The ``--report-dir`` holding the M6 artifact.
        max_beats: The beat budget, as the CLI spells it.
        heartbeat_interval: Seconds between beats, as the CLI spells it.
        research_per_day_micros: An explicit ``--research-per-day-micros``
            startup ceiling, or ``None`` to leave the configured one in force.

    Returns:
        The arguments to pass after ``python -m windbreak``.
    """
    args = [
        "run",
        "--process",
        "pipeline",
        "--config",
        str(config),
        "--ledger-path",
        str(ledger_path),
        "--paper-books-dir",
        str(DEMO_BOOKS),
        "--cassette-path",
        str(SHIPPED_CASSETTE),
        "--report-dir",
        str(report_dir),
        "--max-beats",
        max_beats,
        "--heartbeat-interval",
        heartbeat_interval,
    ]
    if research_per_day_micros is not None:
        args += ["--research-per-day-micros", research_per_day_micros]
    return tuple(args)
