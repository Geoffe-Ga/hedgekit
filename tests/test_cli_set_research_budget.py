"""The two ways an operator changes the daily research cap (#442, #483).

The owner's 2026-08-10 decision requires the per-UTC-day research ceiling to be
"changeable by the operator **without a source-code change** -- via invocation
argument and, where practical, adjustable on the fly rather than only at
startup". Both halves live here:

* ``windbreak run --research-per-day-micros N`` overrides the configured
  ceiling for a run, so a deployment can be re-capped without editing a YAML
  file or a Python constant.
* ``windbreak set-research-budget --ledger-path P --per-day-micros N --note ...``
  appends one ``ResearchBudgetCapSet`` row to the hash-chained ledger. The
  always-on loop re-reads the ledger at the head of every tick, so the change
  takes effect on the next beat of a *running* loop, and it is recorded
  tamper-evidently rather than living in a shell history.

THE REFUSALS, AND THE NON-REFUSALS

Refusing the wrong thing is the failure mode this verb was written against. PR
#482's ``ingest-resolution`` shipped a conflict check that compared a free-text
provenance label, so two ingests differing only in wording read as
contradicting each other -- and because the fold ran on every tick against an
append-only ledger, that mistake was terminal.

A cap is an instruction, not evidence, so two cap rows can never contradict:
the later one is simply in force. This verb therefore refuses only what is
genuinely unusable -- a negative ceiling, which is unenforceable, and a blank
note, which leaves a money-ceiling change unreviewable -- and it refuses both
*before* opening the ledger, so a refused call leaves the chain untouched (the
negative ceiling earlier still, at parse time, so no database file is even
opened). ``test_two_calls_differing_only_in_note_are_both_accepted`` pins the
non-refusal, which is the half a future conflict check would break.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pytest

from windbreak.config.schema import ForecastBudget, ForecastConfig, WindbreakConfig
from windbreak.ledger.events import ModeHeartbeat
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.main import build_parser, main, research_capped_config
from windbreak.scheduler.research_spend import (
    RESEARCH_BUDGET_CAP_SET_EVENT_TYPE,
    ResearchSpendRecorded,
    effective_per_day_micros,
)

if TYPE_CHECKING:
    from pathlib import Path

#: A well-formed replacement ceiling, in micros.
NEW_CAP_MICROS = 7_500_000

#: The operator's note recorded verbatim alongside it.
NOTE = "raising for the Fed-week backlog"


def _argv(
    ledger_path: Path, *, micros: int = NEW_CAP_MICROS, note: str = NOTE
) -> list[str]:
    """Build a complete ``set-research-budget`` argument vector.

    Args:
        ledger_path: The ledger database the verb appends into.
        micros: The per-UTC-day ceiling to set, in micros.
        note: The operator's note.

    Returns:
        The argument vector for :func:`windbreak.main.main`.
    """
    return [
        "set-research-budget",
        "--ledger-path",
        str(ledger_path),
        "--per-day-micros",
        str(micros),
        "--note",
        note,
    ]


def _rows(ledger_path: Path) -> list[tuple[str, dict[str, object]]]:
    """Read every persisted row straight out of SQLite.

    Args:
        ledger_path: The ledger database to read.

    Returns:
        One ``(event_type, data)`` pair per row, in sequence order.
    """
    conn = sqlite3.connect(ledger_path)
    try:
        return [
            (str(event_type), dict(json.loads(str(payload))["data"]))
            for event_type, payload in conn.execute(
                "SELECT event_type, payload_json FROM ledger ORDER BY sequence_number"
            )
        ]
    finally:
        conn.close()


def _seed(ledger_path: Path) -> None:
    """Put one unrelated row on the ledger so it is non-empty.

    A refusal that "wrote nothing" is only meaningful against a ledger that
    exists and has content; against an absent file, "unchanged" and "never
    created" are indistinguishable.

    Args:
        ledger_path: The ledger database to seed.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.append(ModeHeartbeat(component="scheduler", mode="PAPER", beat=1))
    finally:
        store.close()


@pytest.fixture
def seeded_ledger(tmp_path: Path) -> Path:
    """Provide an *existing* ledger database carrying one unrelated row.

    Every accepted call below appends to this rather than conjuring a database,
    because since the ``--ledger-path`` typo hazard was closed the verb refuses
    to create one: a cap change written to a brand-new ledger is invisible to
    the running loop, which reads the ledger it was started with.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The path of a ledger holding exactly one ``ModeHeartbeat`` row.
    """
    path = tmp_path / "ledger.db"
    _seed(path)
    return path


def test_the_verb_appends_exactly_one_cap_row(seeded_ledger: Path) -> None:
    """A well-formed call exits 0 and leaves one row carrying exact values.

    The payload is compared whole rather than key by key, so a field silently
    added, dropped, or renamed fails here. The whole row *sequence* is compared
    too, so the append lands after the ledger's existing content rather than
    replacing it.

    Args:
        seeded_ledger: An existing ledger carrying one ``ModeHeartbeat``.
    """
    exit_code = main(_argv(seeded_ledger))

    rows = _rows(seeded_ledger)
    assert exit_code == 0
    assert [event_type for event_type, _ in rows] == [
        "ModeHeartbeat",
        RESEARCH_BUDGET_CAP_SET_EVENT_TYPE,
    ]
    assert rows[1][1] == {"per_day_micros": NEW_CAP_MICROS, "note": NOTE}


def test_the_appended_row_is_what_the_loop_reads_back(seeded_ledger: Path) -> None:
    """The verb and the tick agree by construction, not by restatement.

    The fold the scheduler runs at the head of every tick is called here
    directly on the ledger the verb just wrote, so "the loop picks this up on
    the next beat" is checked rather than asserted. The configured fallback is
    deliberately a *different* number, so the assertion cannot pass by the two
    coinciding.

    Args:
        seeded_ledger: An existing ledger carrying one ``ModeHeartbeat``.
    """
    ledger = seeded_ledger
    configured = NEW_CAP_MICROS + 1

    assert main(_argv(ledger)) == 0

    store = SqliteLedgerStore(ledger)
    try:
        effective = effective_per_day_micros(
            store.read_all(), configured_micros=configured
        )
    finally:
        store.close()

    assert effective == NEW_CAP_MICROS
    assert effective != configured


def test_a_blank_note_exits_one_and_writes_nothing(tmp_path: Path) -> None:
    """A blank note is refused with exit 1 and the ledger left unchanged.

    The whole row list is compared before and after, so a refusal that appended
    anything -- even a row of a different type -- fails. The ledger is seeded
    first, so "unchanged" is a statement about real content rather than about
    a file that was never created.
    """
    ledger = tmp_path / "ledger.db"
    _seed(ledger)
    before = _rows(ledger)

    exit_code = main(_argv(ledger, note="   "))

    assert before
    assert exit_code == 1
    assert _rows(ledger) == before


def test_a_negative_ceiling_is_refused_at_parse_time(tmp_path: Path) -> None:
    """A negative ceiling never reaches the ledger: argparse refuses the token.

    Refusing at parse time is strictly stronger than refusing in the handler --
    no database file is opened, so on a first run none is created either. The
    exit status is argparse's own 2, deliberately distinguished from the
    handler's 1: an operator who scripts this verb can tell "you typed
    something impossible" from "the change itself was rejected".
    """
    ledger = tmp_path / "ledger.db"
    _seed(ledger)
    before = _rows(ledger)

    with pytest.raises(SystemExit) as caught:
        main(_argv(ledger, micros=-1))

    assert before
    assert caught.value.code == 2
    assert _rows(ledger) == before


def test_a_zero_ceiling_is_accepted(seeded_ledger: Path) -> None:
    """Zero is a legitimate instruction: stop all research spend.

    Refusing it would leave an operator with no way to halt spend on a running
    loop without a restart, which is the capability this verb exists to give
    them. Pinned explicitly because ``0`` sits one step from the ``-1`` the
    test above refuses, and a guard written ``<= 0`` would swallow it.

    Args:
        seeded_ledger: An existing ledger carrying one ``ModeHeartbeat``.
    """
    exit_code = main(_argv(seeded_ledger, micros=0, note="cost anomaly under review"))

    rows = _rows(seeded_ledger)
    assert exit_code == 0
    assert len(rows) == 2
    assert rows[1][1]["per_day_micros"] == 0


def test_two_calls_differing_only_in_note_are_both_accepted(
    seeded_ledger: Path,
) -> None:
    """A retyped note is not a contradiction, and never bricks the loop.

    This is PR #482's lesson made falsifiable. A conflict check that compared
    the free-text note would refuse the second call here; a conflict check at
    the *fold* would refuse every later tick, terminally. Both rows are
    accepted, both stay on the append-only ledger, and the fold reads the last
    one's ceiling.

    Args:
        seeded_ledger: An existing ledger carrying one ``ModeHeartbeat``.
    """
    ledger = seeded_ledger

    first = main(_argv(ledger, note="raising for Fed week"))
    second = main(_argv(ledger, note="raising for fed-week"))

    rows = _rows(ledger)
    store = SqliteLedgerStore(ledger)
    try:
        effective = effective_per_day_micros(store.read_all(), configured_micros=1)
    finally:
        store.close()

    assert (first, second) == (0, 0)
    assert len(rows) == 3
    assert [
        row["note"]
        for event_type, row in rows
        if event_type == RESEARCH_BUDGET_CAP_SET_EVENT_TYPE
    ] == [
        "raising for Fed week",
        "raising for fed-week",
    ]
    assert effective == NEW_CAP_MICROS


def test_a_later_call_supersedes_an_earlier_one_in_both_directions(
    seeded_ledger: Path,
) -> None:
    """Lowering after raising works, and raising after lowering works.

    Both directions, off one ledger, so "last row wins" is shown rather than
    inferred from a single ordering. A fold that took the minimum would pass
    the lowering half and fail the raising half.

    Args:
        seeded_ledger: An existing ledger carrying one ``ModeHeartbeat``.
    """
    ledger = seeded_ledger

    assert main(_argv(ledger, micros=9_000_000, note="raise")) == 0
    assert main(_argv(ledger, micros=2_000_000, note="lower")) == 0
    assert main(_argv(ledger, micros=4_000_000, note="raise again")) == 0

    store = SqliteLedgerStore(ledger)
    try:
        effective = effective_per_day_micros(store.read_all(), configured_micros=1)
    finally:
        store.close()

    assert effective == 4_000_000


def _ceiling_line(err: str) -> str:
    """Return the single startup line reporting the research ceiling.

    Reads the JSON log stream ``configure_logging`` installs rather than a
    ``caplog`` capture: ``main`` calls ``logging.basicConfig(force=True, ...)``,
    which removes pytest's own root handler, so stderr is where the record
    actually lands.

    Args:
        err: The captured stderr of a ``main(["run", ...])`` call.

    Returns:
        The ``msg`` of the one ``research per-day ceiling`` record.
    """
    lines = [
        str(json.loads(line)["msg"])
        for line in err.splitlines()
        if line.startswith("{")
    ]
    ceilings = [line for line in lines if line.startswith("research per-day ceiling")]
    assert len(ceilings) == 1, f"expected one ceiling line, got {ceilings}"
    return ceilings[0]


def _one_beat_run(ledger_path: Path, *, extra: list[str]) -> int:
    """Run one bare heartbeat beat against ``ledger_path``.

    No PAPER flags, so nothing is composed beyond the config load and the
    startup ceiling report -- which is the whole surface under test here.

    Args:
        ledger_path: The ledger the run reads (and ledgers ``ConfigLoaded`` to).
        extra: Additional flags appended verbatim (keyword-only).

    Returns:
        The process exit code.
    """
    return main(
        [
            "run",
            "--heartbeat-interval",
            "0",
            "--max-beats",
            "1",
            "--ledger-path",
            str(ledger_path),
            *extra,
        ]
    )


def test_the_startup_log_reports_the_ledgered_ceiling_that_supersedes_the_flag(
    seeded_ledger: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The log names the ceiling *in force*, not the one the flag asked for.

    The counterexample this closes, exactly as an operator meets it: they run
    ``set-research-budget --per-day-micros 0`` during a cost incident, then
    restart with ``--research-per-day-micros 40000000``. The startup log used
    to say ``ceiling 40000000 source=--research-per-day-micros`` -- while the
    ceiling actually in force was ``0``, because by this module's own
    precedence rule the ledgered row always wins when present. Wrong, and
    wrong in the *permissive* direction, on a spend brake.

    Both the figure and the source are asserted by full-string equality, so a
    fix that folded the ceiling but kept naming the flag would still fail.

    Args:
        seeded_ledger: An existing ledger carrying one ``ModeHeartbeat``.
        capsys: pytest's stdout/stderr capture.
    """
    assert main(_argv(seeded_ledger, micros=0, note="cost anomaly under review")) == 0
    capsys.readouterr()

    exit_code = _one_beat_run(
        seeded_ledger, extra=["--research-per-day-micros", "40000000"]
    )

    assert exit_code == 0
    assert _ceiling_line(capsys.readouterr().err) == (
        "research per-day ceiling 0 micros source=set-research-budget"
    )


def test_the_startup_log_names_the_flag_when_no_cap_row_exists(
    seeded_ledger: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no ledgered row the flag is genuinely in force, and is named.

    The other half of the test above: a fold that reported
    ``set-research-budget`` unconditionally, or reported ``0``
    unconditionally, would pass that test and fail this one. The figure is the
    flag's own value, so the override is observed rather than assumed.

    Args:
        seeded_ledger: An existing ledger carrying one ``ModeHeartbeat``.
        capsys: pytest's stdout/stderr capture.
    """
    exit_code = _one_beat_run(
        seeded_ledger, extra=["--research-per-day-micros", "40000000"]
    )

    assert exit_code == 0
    assert _ceiling_line(capsys.readouterr().err) == (
        "research per-day ceiling 40000000 micros source=--research-per-day-micros"
    )


def test_the_startup_log_names_the_configuration_when_no_flag_is_given(
    seeded_ledger: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shipped default is reported as coming from the configuration.

    Pins the third source token and the shipped 20 000 000-micro ceiling, so a
    change to either is a visible change to what operators are told.

    Args:
        seeded_ledger: An existing ledger carrying one ``ModeHeartbeat``.
        capsys: pytest's stdout/stderr capture.
    """
    exit_code = _one_beat_run(seeded_ledger, extra=[])

    assert exit_code == 0
    assert _ceiling_line(capsys.readouterr().err) == (
        "research per-day ceiling 20000000 micros source=configuration"
    )


def test_the_startup_log_reports_zero_when_the_research_rows_cannot_be_folded(
    seeded_ledger: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreadable ledger is reported as the zero ceiling the loop will use.

    The loop does not refuse to start over a malformed research row -- it opens
    a zero ceiling and halts every market on the budget -- so the startup log
    has to say zero too. Reporting the configured ceiling here would be the
    same lie in the same permissive direction as the flag case above.

    The row is corrupted through ``sqlite3`` because the typed event refuses to
    construct without the key, which is precisely why this shape can only reach
    a ledger from outside windbreak.

    Args:
        seeded_ledger: An existing ledger carrying one ``ModeHeartbeat``.
        capsys: pytest's stdout/stderr capture.
    """
    store = SqliteLedgerStore(seeded_ledger)
    try:
        store.append(
            ResearchSpendRecorded(
                component="scheduler",
                utc_day="2026-08-09",
                market_ticker="MKT-DEEP",
                cost_micros=1,
            )
        )
    finally:
        store.close()
    _drop_payload_key(seeded_ledger, "cost_micros")
    capsys.readouterr()

    exit_code = _one_beat_run(
        seeded_ledger, extra=["--research-per-day-micros", "40000000"]
    )

    assert exit_code == 0
    assert _ceiling_line(capsys.readouterr().err) == (
        "research per-day ceiling 0 micros source=unreadable-ledger"
    )


def _drop_payload_key(ledger_path: Path, key: str) -> None:
    """Delete one key from every row's typed payload, in place.

    Args:
        ledger_path: The ledger database to rewrite.
        key: The payload key to remove.
    """
    conn = sqlite3.connect(ledger_path)
    try:
        for sequence_number, payload_json in conn.execute(
            "SELECT sequence_number, payload_json FROM ledger"
        ).fetchall():
            envelope = json.loads(str(payload_json))
            envelope["data"].pop(key, None)
            conn.execute(
                "UPDATE ledger SET payload_json = ? WHERE sequence_number = ?",
                (json.dumps(envelope), sequence_number),
            )
        conn.commit()
    finally:
        conn.close()


def test_a_typoed_ledger_path_is_refused_and_creates_no_database(
    tmp_path: Path,
) -> None:
    """A path that is not an existing ledger exits 1 and writes nothing at all.

    The fail-open this closes: the verb used to *succeed* on a mistyped
    ``--ledger-path``. A fresh SQLite database was created, the row landed at
    sequence 1, the log said "research per-day ceiling set to 0 micros
    sequence=1", the exit code was 0 -- and the running loop, reading the
    ledger it was started with, never saw the cap. On the one verb whose whole
    purpose is to stop money, silence is the wrong answer.

    The absence of the file is asserted, not just the exit code: a refusal that
    still created an empty database would leave the operator's *next* attempt
    at the same typo silently succeeding.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    typo = tmp_path / "ledgre.db"

    exit_code = main(_argv(typo, micros=0, note="cost anomaly under review"))

    assert exit_code == 1
    assert not typo.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_directory_ledger_path_exits_one_instead_of_raising(
    tmp_path: Path,
) -> None:
    """A directory is a refused change, not an uncaught sqlite traceback.

    This verb's docstring already promised "exit 1 on a refused change"; a
    directory used to escape as a bare ``sqlite3.OperationalError`` stack
    trace instead, which is neither the documented contract nor something an
    operator's wrapper script can branch on.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    directory = tmp_path / "ledger.db"
    directory.mkdir()

    exit_code = main(_argv(directory))

    assert exit_code == 1
    assert list(directory.iterdir()) == []


def test_the_refusal_leaves_a_real_ledger_reachable_afterwards(
    seeded_ledger: Path, tmp_path: Path
) -> None:
    """The recovery path works: retype the path and the change lands.

    A refusal that were somehow sticky -- a half-open handle, a stray file --
    would turn an operator's typo into an outage on the verb they reach for
    during a cost incident. So the correction is exercised end to end rather
    than assumed.

    Args:
        seeded_ledger: An existing ledger carrying one ``ModeHeartbeat``.
        tmp_path: pytest's per-test temporary directory.
    """
    refused = main(_argv(tmp_path / "typo.db", micros=0, note="halt spend"))

    accepted = main(_argv(seeded_ledger, micros=0, note="halt spend"))

    rows = _rows(seeded_ledger)
    assert refused == 1
    assert accepted == 0
    assert [event_type for event_type, _ in rows] == [
        "ModeHeartbeat",
        RESEARCH_BUDGET_CAP_SET_EVENT_TYPE,
    ]
    assert rows[1][1] == {"per_day_micros": 0, "note": "halt spend"}


def test_the_run_flag_overrides_the_configured_ceiling() -> None:
    """``run --research-per-day-micros`` replaces exactly one config field.

    Every other budget field is asserted unchanged, so an override that
    rebuilt the section with defaults -- silently resetting the per-forecast
    ceiling or the page cap -- fails here rather than in production.
    """
    config = WindbreakConfig(
        forecast=ForecastConfig(
            budget=ForecastBudget(
                per_forecast_micros=6_060_000, per_day_micros=20_000_000, max_pages=20
            )
        )
    )

    overridden = research_capped_config(config, per_day_micros=3_000_000)

    assert overridden.forecast.budget.per_day_micros == 3_000_000
    assert overridden.forecast.budget.per_forecast_micros == 6_060_000
    assert overridden.forecast.budget.max_pages == 20
    assert config.forecast.budget.per_day_micros == 20_000_000


def test_omitting_the_run_flag_leaves_the_configuration_untouched() -> None:
    """With no override the very same config object comes back.

    Identity, not equality: a rebuilt-but-equal config would still be a change
    of behaviour for every deployment that never passes the flag, and identity
    is the only assertion that rules it out.
    """
    config = WindbreakConfig()

    assert research_capped_config(config, per_day_micros=None) is config


def test_the_run_parser_accepts_the_cap_flag() -> None:
    """``run`` exposes ``--research-per-day-micros`` and defaults it to ``None``.

    The default is what keeps every existing invocation byte-identical, so it
    is pinned rather than assumed.
    """
    parser = build_parser()

    default = parser.parse_args(["run"])
    supplied = parser.parse_args(["run", "--research-per-day-micros", "1234"])

    assert default.research_per_day_micros is None
    assert supplied.research_per_day_micros == 1234


def test_the_run_parser_refuses_a_negative_cap_flag() -> None:
    """A negative ceiling is rejected by argparse, before anything is composed.

    Refusing at parse time means the loop never starts on an unenforceable
    ceiling, rather than starting and failing later at the first charge.
    """
    parser = build_parser()

    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["run", "--research-per-day-micros", "-1"])

    assert caught.value.code == 2
