"""The startup guard on a PAPER loop that can only ever abstain (issue #485).

Issue #438 asked the loop to "fail startup loudly when the PAPER loop is
activated with a research configuration that can only abstain, instead of
running an abstention-forever loop that looks healthy". #485 split that out and
the owner deferred it **twice**, both times for the same reason: until PR #522
(#510) landed, *every* configuration an operator could type could only abstain,
so a guard on that predicate would have fired on every single startup, and a
signal that always fires teaches operators to ignore it.

PR #522 made a non-abstaining configuration expressible from the shipped CLI,
so the predicate finally discriminates. This module is the proof that it does,
driven through :func:`windbreak.main.main` -- the argument vector
``deploy/docker-compose.yml`` invokes -- rather than through the guard in
isolation.

The predicate, stated exactly
-----------------------------

The guard reads the research bundle this process *actually composed* and asks
one question: **does it have any source of evidence at all?**
:func:`~windbreak.scheduler.loop._resolve_research_tools` returns that answer
alongside the bundle it built, so the two cannot disagree -- deleting the
corpus from the wiring call moves the reported source to ``none`` in the same
edit, which is what keeps this from being the composition trap wearing a safety
label.

It claims a **one-way implication and no more**: no evidence source means every
forecast must abstain on ``no_verified_citations`` before a single vote, so no
intent can ever be emitted. It does *not* claim the converse. A deployment that
has an evidence source may still abstain or screen out forever, and the guard
says nothing about that -- the depth floor, the resolution horizon, the
correlation declaration and the provider track record are all judged per tick
against the books, and the guard reads none of them.

Warn, not refuse
----------------

The acceptance criteria offer both postures and this lane takes the second, for
three reasons that are in the review record rather than in taste:

* PR #487 established **fail closed on the capability, never on the process**.
  A starved loop is *already* failing closed on the capability: it emits no
  intent. Refusing the process would additionally take away the operator's
  ability to kill it, verify its chain, or read its ledger -- a deployment that
  cannot produce evidence is not the same as one that cannot be stopped.
* The shipped default *is* the starved configuration, deliberately (#522 kept
  it that way and proved it). Refusing would turn ``docker compose up`` into a
  hard failure, which is PR #482's defect relocated to launch.
* Configuration *contradictions* already refuse, and should
  (:func:`~windbreak.scheduler.loop._resolve_replay_corpus` refuses an unknown
  mode, a replay mode naming no directory, and a corpus selected alongside the
  live transport). A coherent-but-incapable configuration is not a
  contradiction, and must not be answered like one.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from windbreak.config.schema import WindbreakConfig
from windbreak.main import main
from windbreak.scheduler.loop import (
    EVIDENCE_STARVED_MESSAGE,
    RESEARCH_EVIDENCE_CORPUS,
    research_evidence_fold,
)

#: The repository root, resolved from this file rather than the process's cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed books fixture PR #522's hermetic demonstration replays. Both
#: directions below run over the *same* books, so the books are never what
#: distinguishes them -- only the research wiring is.
DEMO_BOOKS = REPO_ROOT / "tests" / "fixtures" / "books" / "hermetic_demo"

#: The committed configuration that selects the corpus (PR #522). This is the
#: only CLI-expressible configuration that is not evidence-starved, and it is
#: the reason this issue stopped being blocked.
DEMO_CONFIG = REPO_ROOT / "tests" / "fixtures" / "config" / "hermetic-demo.yaml"

#: The committed vote cassette both shipped command lines name. Never read on
#: the corpus path, and never reached on the starved path.
SHIPPED_CASSETTE = REPO_ROOT / "tests" / "fixtures" / "forecast" / "cassettes.json"

#: The committed M6 evaluation artifact the loop's live-eligibility gate reads.
DEMO_TRACK_RECORDS = (
    REPO_ROOT / "tests" / "fixtures" / "evaluation" / "provider-track-records.json"
)

#: That artifact's filename inside a run's ``--report-dir``.
TRACK_RECORD_FILENAME = "provider-track-records.json"

#: The ``component`` the startup fold and the starvation warning are logged
#: under -- the scheduler logger's own name, since neither line overrides it.
LOOP_LOGGER = "windbreak.scheduler"

#: The JSON ``level`` token of a warning line.
WARNING = "WARNING"

#: The JSON ``level`` token of an informational line.
INFO = "INFO"

#: The prefix every startup fold line carries, whatever source won.
FOLD_PREFIX = "research evidence source="

#: The four configuration leaves the warning's remedy names, as field paths to
#: be walked out of the live schema rather than trusted as strings.
REMEDY_LEAVES = [
    ("forecast", "replay_corpus", "mode"),
    ("forecast", "replay_corpus", "corpus_dir"),
    ("forecast", "provider_transport", "mode"),
    ("forecast", "research", "search_endpoint_url"),
]


def _run_cli(tmp_path: Path, *, config: Path | None) -> int:
    """Run one beat of the shipped PAPER entry point over the demo books.

    The argument vector is ``deploy/docker-compose.yml``'s, narrowed to a single
    beat -- the guard runs once, when
    :func:`~windbreak.scheduler.loop.build_paper_deps` composes the bundle,
    before any beat -- so nothing here depends on what a tick then does.

    Args:
        tmp_path: pytest's per-test temporary directory.
        config: The ``--config`` file, or ``None`` to run the shipped defaults.

    Returns:
        The process exit code.
    """
    report_dir = tmp_path / "report"
    report_dir.mkdir(exist_ok=True)
    shutil.copy(DEMO_TRACK_RECORDS, report_dir / TRACK_RECORD_FILENAME)
    argv = [
        "run",
        "--process",
        "pipeline",
        "--heartbeat-interval",
        "0",
        "--max-beats",
        "1",
        "--ledger-path",
        str(tmp_path / "ledger.db"),
        "--paper-books-dir",
        str(DEMO_BOOKS),
        "--cassette-path",
        str(SHIPPED_CASSETTE),
        "--report-dir",
        str(report_dir),
    ]
    if config is not None:
        argv += ["--config", str(config)]
    return main(argv)


def _loop_messages(stderr: str, level: str) -> list[str]:
    """Return every loop-logger message the run emitted at ``level``, in order.

    Read back out of the process's own JSON log stream rather than through
    ``caplog``, because :func:`~windbreak.logging_setup.configure_logging` calls
    ``logging.basicConfig(force=True, ...)`` inside :func:`windbreak.main.main`
    and evicts every handler pytest installed. Reading stderr is also the
    stronger assertion: these are the exact bytes an operator's log aggregator
    receives, after the process's own redaction filter has run over them.

    Args:
        stderr: The captured standard-error stream of a finished run.
        level: The exact JSON ``level`` token to select.

    Returns:
        The ``msg`` of each matching line, in emission order.
    """
    lines = [json.loads(line) for line in stderr.splitlines() if line.startswith("{")]
    return [
        str(line["msg"])
        for line in lines
        if line["component"] == LOOP_LOGGER and line["level"] == level
    ]


def _config_path(names: tuple[str, ...]) -> str:
    """Return the dotted config path for ``names``, proving each leaf exists.

    Walks a real :class:`~windbreak.config.schema.WindbreakConfig` instance
    field by field rather than restating a path as a string literal, so a leaf
    the remedy names but the schema no longer has fails here instead of being
    read by an operator who then cannot act on it.

    Args:
        names: The field names to walk, outermost first.

    Returns:
        The dotted path, e.g. ``forecast.replay_corpus.mode``.
    """
    node: object = WindbreakConfig()
    for name in names:
        assert dataclasses.is_dataclass(node)
        declared = {field.name for field in dataclasses.fields(node)}
        assert name in declared, f"{type(node).__name__} declares no {name!r}"
        node = getattr(node, name)
    return ".".join(names)


def test_the_shipped_default_cli_warns_that_it_can_never_produce_evidence(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The default composition starts, and says once that it cannot forecast.

    This is the ``docker compose up`` case issue #485 opened on: it starts
    (exit 0 -- the process is not what fails closed here), it does not stay
    quiet about it, and the warning it emits is the whole operator-facing
    message, compared for equality rather than matched on a substring.

    The starved run emits the warning *instead of* a fold line, not alongside
    one: an operator scanning ``INFO`` for the effective source must not find a
    reassuring row on the run that has none.

    Args:
        capsys: The pytest stream-capture fixture.
        tmp_path: pytest's per-test temporary directory.
    """
    exit_code = _run_cli(tmp_path, config=None)

    stderr = capsys.readouterr().err
    assert exit_code == 0
    assert _loop_messages(stderr, WARNING) == [EVIDENCE_STARVED_MESSAGE]
    folds = [
        message
        for message in _loop_messages(stderr, INFO)
        if message.startswith(FOLD_PREFIX)
    ]
    assert folds == []


def test_the_corpus_configured_cli_is_not_warned_and_names_its_source(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The one non-starved CLI configuration starts unwarned, and names it.

    The same books, the same cassette, the same beat count as the default case
    above -- only ``--config`` differs -- so the discrimination this asserts can
    only come from the research wiring. Without it the guard would fire on every
    startup and be worth less than nothing, which is exactly why the owner
    deferred this issue twice.

    Args:
        capsys: The pytest stream-capture fixture.
        tmp_path: pytest's per-test temporary directory.
    """
    exit_code = _run_cli(tmp_path, config=DEMO_CONFIG)

    stderr = capsys.readouterr().err
    assert exit_code == 0
    assert EVIDENCE_STARVED_MESSAGE not in _loop_messages(stderr, WARNING)
    folds = [
        message
        for message in _loop_messages(stderr, INFO)
        if message.startswith(FOLD_PREFIX)
    ]
    assert folds == [research_evidence_fold(RESEARCH_EVIDENCE_CORPUS)]


@pytest.mark.parametrize("leaf", REMEDY_LEAVES)
def test_the_starvation_warning_names_a_remedy_the_schema_still_has(
    leaf: tuple[str, ...],
) -> None:
    """Each configuration leaf the remedy names is walked out of the schema.

    A refusal that does not name the remedy is a worse experience than a warning
    that does -- and a remedy naming a leaf that was renamed away is worse
    still, because an operator follows it and nothing changes.

    Args:
        leaf: The field names of one remedy leaf, outermost first.
    """
    assert _config_path(leaf) in EVIDENCE_STARVED_MESSAGE
