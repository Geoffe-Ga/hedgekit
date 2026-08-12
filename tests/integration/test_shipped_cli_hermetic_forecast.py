"""What the *shipped entry point* forecasts, driven through `windbreak run` (#510).

Issue #510 stated that the shipped CLI could not reach a non-abstention
forecast, so no intent, approval or fill could cross a real process boundary.
PR #520 pinned that end state here, through :func:`windbreak.main.main` rather
than through :func:`~windbreak.scheduler.loop.build_paper_deps`, and said in as
many words that its assertions were **meant to flip** when the fix landed. They
have. This module is that module, inverted: the same entry point, the same
argument vector an operator types, and the ledger it actually writes now
carrying a forecast that abstains on nothing and a selector decision holding one
order intent.

What made the fix a composition rather than a fixture
-----------------------------------------------------

Three doors were closed, and only the first two were named by the issue.

1. **Research.** ``offline_research_tools`` finds nothing by construction, so
   ``run_pipeline`` abstained on ``no_verified_citations`` before a single vote.
2. **Votes.** The committed ``cassettes.json`` holds three placeholders, and no
   *committed* cassette can ever serve a market that clears the horizon:
   :func:`~windbreak.forecast.providers.base.build_vote_prompt` interpolates
   ``close_time.isoformat()`` into the prompt the replay key digests, while the
   horizon filter measures that same close against the run's clock. That
   contradiction is proved by ``tests/scheduler/
   test_paper_intent_barriers.py::\
test_static_vote_cassette_and_horizon_filter_are_mutually_exclusive`` and is
   why the fix is a *corpus* keyed on what does not move -- the subquestion text
   and the ``provider:model_version`` pair -- selected from configuration.
3. **The market calendar.** ``PaperExchange`` re-dated its books, its trade
   prints and its venue clock onto the replay anchor and assigned its markets
   verbatim, so a committed fixture could hold a fresh *book* but never an open
   *horizon*: its close was a literal and the clock was not. That is why #520
   had to rewrite ``markets.json`` from the test before it could reach the
   screen at all. ``tests/integration/test_paper_replay_market_horizon.py``
   pins the repair; the consequence here is that **nothing in this module
   edits the committed books fixture** on the emitting path.

What is asserted, and what it is not
------------------------------------

* :func:`test_the_hermetic_configuration_moves_no_threshold` compares the
  committed YAML against :class:`~windbreak.config.schema.WindbreakConfig`
  field by field, *derived* by walking the dataclass rather than by a
  hand-written list, and pins that exactly two leaves differ. An intent reached
  by relaxing a floor would fail here rather than passing next door.
* :func:`test_the_shipped_cli_ledgers_a_forecast_and_an_approved_intent` is the
  end-to-end evidence: two beats of the real CLI over committed fixtures and
  committed configuration, asserting the whole chain #510's acceptance criteria
  name, with exact values.
* :func:`test_the_default_configuration_still_abstains_and_emits_no_intent` is
  the other half of the same run and the reason the fix is safe: the identical
  command line **without** ``--config`` still abstains and still emits nothing.
  The capability is opt-in; the default is the offline loop that cannot trade.
* The parametrized negatives put each barrier back one at a time and assert both
  that the intent disappears *and* the evidence that removed it -- a bare
  ``intent_count == 0`` passes for any reason, including a crash upstream.

A replayed forecast is recorded material and measures nothing. What these tests
prove is that the *stack* composes end to end from a command line, which is the
claim ``ARCHITECTURE.md`` rests on and the one that had no evidence.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from typing import TYPE_CHECKING

import pytest

from tests.hermetic_demo import (
    DEMO_BOOKS,
    DEMO_CONFIG,
    DEMO_CORPUS,
    DEMO_TRACK_RECORDS,
    REPO_ROOT,
    SHIPPED_CASSETTE,
    TICKER,
    place_track_records,
)
from tests.paper_books import recording_origin, set_close_time
from windbreak.config.loader import load_config
from windbreak.config.schema import (
    REPLAY_CORPUS_DISABLED,
    REPLAY_CORPUS_REPLAY,
    CapitalConfig,
    HorizonDays,
    ProviderGateConfig,
    ScreenerConfig,
    WindbreakConfig,
)
from windbreak.forecast.budget import FULL_PIPELINE_RESEARCH_COST_MICROS
from windbreak.forecast.corpus import (
    CORPUS_RESEARCH_FILENAME,
    CORPUS_VOTES_FILENAME,
    vote_key,
)
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.main import main

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: The resting size, in contract-centis, on every level of the committed book.
DEMO_QUANTITY_CENTIS = 100_000

#: A resting size below `ScreenerConfig().min_depth_contract_centis`, for the
#: depth negative.
THIN_QUANTITY_CENTIS = 300

#: The committed book's resting YES ask, in pips -- `RiskConfig`'s own
#: `min_open_price_pips`, the lowest price the open band admits.
DEMO_ASK_PIPS = 500

#: Pips-to-ppm: a price in pips is a probability in ppm scaled by this.
PPM_PER_PIP = 100

#: Whole days from the recording's leading book to the committed close, inside
#: `HorizonDays()`'s production [2, 120]-day window.
DEMO_HORIZON_DAYS = 30

#: A close before the recording, and so in the past on any beat that replays it.
OUT_OF_WINDOW_HORIZON_DAYS = -400

#: The abstention an empty research bundle forces, before any vote is cast.
NO_CITATIONS_ABSTENTION = "no_verified_citations"

#: The three ensemble members the shipped `vote_ensemble` pins, in order.
ENSEMBLE_MEMBERS = (
    ("openai", "gpt-5-2025-08-07"),
    ("anthropic", "claude-sonnet-4-5-20250929"),
    ("openai", "gpt-5-mini-2025-08-07"),
)

#: The four order-state edges one approved intent walks in a beat.
ORDER_EDGES = ["APPROVE", "REQUEST_SUBMISSION", "SUBMIT", "ACK"]


def _corpus_vote_probabilities(corpus: Path = DEMO_CORPUS) -> dict[str, int]:
    """Return each ensemble member's recorded vote probability, in ppm.

    Read out of the committed corpus rather than restated, so the assertions
    below are identities against the fixture instead of transcriptions of it.

    Args:
        corpus: The corpus directory to read.

    Returns:
        Each ``provider:model_version`` key paired with its recorded ppm.
    """
    recorded = json.loads((corpus / CORPUS_VOTES_FILENAME).read_text(encoding="utf-8"))
    return {
        key: int(json.loads(entry["response"])["probability_ppm"])
        for key, entry in recorded.items()
    }


def _run_cli(
    tmp_path: Path,
    *,
    books: Path,
    config: Path | None,
    beats: int = 2,
    track_records: bool = True,
) -> tuple[int, Path]:
    """Run ``beats`` beats of the shipped PAPER entry point.

    The argument vector is the one ``deploy/docker-compose.yml`` and
    ``deploy/systemd/windbreak-pipeline.service`` invoke, narrowed to a fixed
    beat count. Two beats by default because the risk kernel vetoes on the first
    beat of a fresh ledger -- ``equity_start_of_day`` is 0 until that beat's own
    ``EquitySampled`` row exists -- so an approval is only observable on the
    second.

    Args:
        tmp_path: pytest's per-test temporary directory.
        books: The books directory to replay.
        config: The ``--config`` file, or ``None`` to run the shipped defaults.
        beats: How many beats to run (keyword-only).
        track_records: Whether to place the committed M6 artifact in the report
            directory (keyword-only).

    Returns:
        The process exit code and the ledger path it wrote.
    """
    ledger_path = tmp_path / "ledger.db"
    report_dir = tmp_path / "report"
    report_dir.mkdir(exist_ok=True)
    if track_records:
        place_track_records(report_dir)
    argv = [
        "run",
        "--process",
        "pipeline",
        "--heartbeat-interval",
        "0",
        "--max-beats",
        str(beats),
        "--ledger-path",
        str(ledger_path),
        "--paper-books-dir",
        str(books),
        "--cassette-path",
        str(SHIPPED_CASSETTE),
        "--report-dir",
        str(report_dir),
    ]
    if config is not None:
        argv += ["--config", str(config)]
    return main(argv), ledger_path


def _ledgered(ledger_path: Path) -> list[tuple[str, dict[str, object]]]:
    """Read every row of a ledger written by a finished run, chain first.

    Args:
        ledger_path: The hash-chained ledger the run wrote.

    Returns:
        One ``(event_type, data)`` pair per row, in ledger order.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    return [
        (record.event_type, dict(json.loads(record.payload_json)["data"]))
        for record in records
    ]


def _rows_of(
    rows: list[tuple[str, dict[str, object]]], event: str
) -> list[dict[str, object]]:
    """Return every payload of ``event``, in ledger order.

    Args:
        rows: Every ledgered ``(event_type, data)`` pair.
        event: The event type to select.

    Returns:
        That event's payloads.
    """
    return [data for name, data in rows if name == event]


def _first(rows: list[tuple[str, dict[str, object]]], event: str) -> dict[str, object]:
    """Return the first payload of ``event``, asserting at least one exists.

    Args:
        rows: Every ledgered ``(event_type, data)`` pair.
        event: The event type to select.

    Returns:
        That event's first payload.
    """
    matched = _rows_of(rows, event)
    assert matched != []
    return matched[0]


def _demo_books_copy(tmp_path: Path) -> Path:
    """Copy the committed demonstration books so a negative can spoil one input.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The private copy.
    """
    books = tmp_path / "books"
    shutil.copytree(DEMO_BOOKS, books)
    return books


def _config_with(tmp_path: Path, **replacements: str) -> Path:
    """Write a variant of the committed configuration into ``tmp_path``.

    Only the named ``forecast.replay_corpus`` leaves move; the correlation
    declaration and every default are carried over verbatim from the committed
    file, so a negative differs from the emitting run in exactly one stated way.

    Args:
        tmp_path: pytest's per-test temporary directory.
        **replacements: ``mode`` and/or ``corpus_dir`` overrides.

    Returns:
        The written configuration file.
    """
    committed = load_config(DEMO_CONFIG)
    corpus = dataclasses.replace(
        committed.forecast.replay_corpus,
        **replacements,  # type: ignore[arg-type]
    )
    path = tmp_path / "config.yaml"
    tag = committed.correlation.tags[0]
    path.write_text(
        "forecast:\n"
        "  replay_corpus:\n"
        f'    mode: "{corpus.mode}"\n'
        f'    corpus_dir: "{corpus.corpus_dir}"\n'
        "correlation:\n"
        "  tags:\n"
        f"    - ticker: {tag.ticker}\n"
        "      bucket_ids:\n"
        f"        - {tag.bucket_ids[0]}\n"
        f'      tagged_at: "{tag.tagged_at}"\n',
        encoding="utf-8",
    )
    return path


def _differing_leaves(left: object, right: object, prefix: str) -> list[str]:
    """Return the dotted paths of every leaf where two configs differ.

    Walks the dataclass rather than comparing a hand-written list of fields, so
    a leaf added later is covered without anyone remembering to add it -- the
    difference between measuring "nothing else moved" and asserting it.

    Args:
        left: The left value.
        right: The right value.
        prefix: The dotted path reached so far.

    Returns:
        Every differing leaf's dotted path, deepest name last.
    """
    if dataclasses.is_dataclass(left) and dataclasses.is_dataclass(right):
        differing: list[str] = []
        for field in dataclasses.fields(left):
            differing += _differing_leaves(
                getattr(left, field.name),
                getattr(right, field.name),
                f"{prefix}.{field.name}" if prefix else field.name,
            )
        return differing
    return [] if left == right else [prefix]


def test_the_hermetic_configuration_moves_no_threshold() -> None:
    """The committed configuration differs from the defaults in two leaves only.

    #510's fourth acceptance criterion, measured rather than promised, in the
    shape ``test_only_the_correlation_declaration_and_state_dir_leave_the_\
defaults`` established: the differing set is *derived* by walking every
    dataclass field of :class:`~windbreak.config.schema.WindbreakConfig`, so a
    lowered floor, a widened window or a cheapened charge appears here as a new
    entry rather than hiding behind a hand-maintained list.

    The two leaves that do differ are the two the fix needs: the replay-corpus
    selection and the correlation declaration. Neither is a threshold.
    """
    configured = load_config(DEMO_CONFIG)

    differing = _differing_leaves(configured, WindbreakConfig(), "")

    assert differing == [
        "correlation.tags",
        "forecast.replay_corpus.mode",
        "forecast.replay_corpus.corpus_dir",
    ]
    assert configured.forecast.replay_corpus.mode == REPLAY_CORPUS_REPLAY
    assert WindbreakConfig().forecast.replay_corpus.mode == REPLAY_CORPUS_DISABLED
    assert configured.correlation.tags[0].ticker == TICKER


def test_the_committed_books_satisfy_production_thresholds() -> None:
    """Every committed input clears the production default it faces, unmodified.

    The evidence below is only evidence about the *shipped* thresholds if the
    fixture satisfies them rather than the configuration relaxing them. Each
    value is therefore compared against the production constant it has to
    clear, read from the config classes rather than transcribed.

    The horizon is the one that could not be stated before issue #510: it is
    asserted as a *distance from the recording's own leading book*, which is
    what an anchored replay preserves, so this fixture does not go stale.
    """
    markets = json.loads((DEMO_BOOKS / "markets.json").read_text(encoding="utf-8"))
    sessions = json.loads((DEMO_BOOKS / "sessions.json").read_text(encoding="utf-8"))
    balances = json.loads((DEMO_BOOKS / "balances.json").read_text(encoding="utf-8"))
    window = HorizonDays()

    from tests.paper_books import fixture_instant

    horizon = fixture_instant(markets[0]["close_time"]) - recording_origin(DEMO_BOOKS)
    resting = [
        level["quantity"]
        for steps in sessions.values()
        for step in steps
        for side in ("yes_bids", "yes_asks")
        for level in step["book"][side]
    ]

    assert horizon.days == DEMO_HORIZON_DAYS
    assert window.min <= horizon.days <= window.max
    assert resting != []
    assert set(resting) == {DEMO_QUANTITY_CENTIS}
    assert ScreenerConfig().min_depth_contract_centis < DEMO_QUANTITY_CENTIS
    assert CapitalConfig().floor_micros < balances["total"]
    assert markets[0]["ticker"] == TICKER


def test_the_committed_corpus_covers_every_shipped_ensemble_member() -> None:
    """The corpus records one vote per member of the shipped vote ensemble.

    The corpus keys votes by ``provider:model_version``, so "covers the
    ensemble" is a claim about two lists agreeing. Both are read from their
    sources -- the ensemble from
    :class:`~windbreak.config.schema.WindbreakConfig`, the keys from the
    committed file -- and compared for **equality**, so a member added to the
    ensemble later fails here rather than silently losing its vote to a discard
    at run time.

    The three recorded probabilities are asserted **distinct**, which is what
    keeps the emitting test's ``probability_ppm`` assertion discriminating: with
    three equal votes the ensemble's median, its CI bounds and every member's
    own vote are one number, and any wiring at all would reproduce it.
    """
    ensemble = WindbreakConfig().forecast.vote_ensemble
    recorded = _corpus_vote_probabilities()

    assert tuple((m.provider, m.model_version) for m in ensemble) == ENSEMBLE_MEMBERS
    assert sorted(recorded) == sorted(
        vote_key(provider, model) for provider, model in ENSEMBLE_MEMBERS
    )
    assert len(set(recorded.values())) == len(ENSEMBLE_MEMBERS)
    assert (DEMO_CORPUS / CORPUS_RESEARCH_FILENAME).is_file()


def test_the_shipped_cli_ledgers_a_forecast_and_an_approved_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`windbreak run` forecasts, emits one intent, and gets it approved (#510).

    Issue #510's first three acceptance criteria, through the entry point an
    operator types, over **committed** fixtures and **committed**
    configuration. Nothing is injected and nothing is edited: the books
    directory is the repository's own, unmodified, because an anchored replay
    now re-enacts its calendar with its books.

    Every value is pinned exactly, and the two that could be reached by
    accident are pinned as identities against their sources:

    * ``research_cost_micros`` is compared to the production constant, so this
      cannot be a run that reached an intent by cheapening the charge;
    * ``probability_ppm`` equals the **lowest** recorded corpus vote, read from
      the committed file. That is the pipeline's own arithmetic and not a
      coincidence: the ensemble median is shrunk toward the market's own
      50 000 ppm baseline and the result is then clamped into the vote CI, whose
      low bound is the minimum vote. It is asserted *unequal* to the baseline
      price in ppm as well -- the exact inverse of the identity #520 pinned,
      where the "forecast" was the price echoed back and therefore carried no
      information at all.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: The active patcher, used to run from the repository root so
            the committed configuration's repo-relative ``corpus_dir`` resolves
            the way ``docs/RUNBOOK.md`` says it does.
    """
    monkeypatch.chdir(REPO_ROOT)
    recorded = _corpus_vote_probabilities()

    exit_code, ledger_path = _run_cli(tmp_path, books=DEMO_BOOKS, config=DEMO_CONFIG)

    assert exit_code == 0
    rows = _ledgered(ledger_path)
    assert _first(rows, "ScreenDecisionRecorded") == {
        "blocked_by": [],
        "eligible": True,
        "ticker": TICKER,
    }
    forecast = _first(rows, "ForecastCreated")
    assert forecast["abstention_reason"] is None
    assert forecast["market_ticker"] == TICKER
    assert forecast["eligible_for_live"] is True
    assert forecast["market_price_baseline_pips"] == DEMO_ASK_PIPS
    assert forecast["research_cost_micros"] == FULL_PIPELINE_RESEARCH_COST_MICROS
    assert forecast["probability_ppm"] == min(recorded.values())
    assert forecast["probability_ppm"] != DEMO_ASK_PIPS * PPM_PER_PIP
    votes = _rows_of(rows, "ProviderVoteRecorded")
    assert [
        (vote["provider"], vote["model_version"]) for vote in votes[: len(recorded)]
    ] == list(ENSEMBLE_MEMBERS)
    assert {vote["outcome"] for vote in votes} == {"voted"}
    assert _first(rows, "SelectorDecisionRecorded")["intent_count"] == 1
    assert [reason for reason in _reasons(rows) if reason.startswith("fail:")] == []
    approved = _first(rows, "IntentApproved")
    assert approved["reasons"] == []
    intent_id = approved["intent_id"]
    assert _first(rows, "ApprovalTokenIssued")["intent_id"] == intent_id
    assert [
        edge["event"] for edge in _rows_of(rows, "OrderTransitionLedgered")
    ] == ORDER_EDGES
    assert _rows_of(rows, "ProviderGateHeld") == []


def _reasons(rows: list[tuple[str, dict[str, object]]]) -> list[str]:
    """Return the first selector decision's rendered reasons.

    Args:
        rows: Every ledgered ``(event_type, data)`` pair.

    Returns:
        The reason strings, in the order the selector rendered them.
    """
    raw = _first(rows, "SelectorDecisionRecorded")["reasons"]
    assert isinstance(raw, list)
    return [str(reason) for reason in raw]


def test_the_default_configuration_still_abstains_and_emits_no_intent(
    tmp_path: Path,
) -> None:
    """The identical command line without `--config` still cannot trade.

    The safety half of the fix, and the reason the capability is a
    configuration token rather than a behaviour change. Over the *same*
    committed books that emit an intent above -- so the screen is not what
    stops it -- the shipped defaults resolve the offline research bundle,
    gather nothing, and abstain before a single vote is cast. That is asserted
    as an exact count of **zero** ``ProviderVoteRecorded`` rows rather than as
    the absence of a row this test forgot to look for.

    The abstaining forecast's ``probability_ppm`` is the market's own ask
    echoed back, asserted as the derived identity #520 established: a
    "forecast" that is exactly the price carries no information, which is why
    an abstaining loop produces no track record.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    exit_code, ledger_path = _run_cli(tmp_path, books=DEMO_BOOKS, config=None)

    assert exit_code == 0
    rows = _ledgered(ledger_path)
    assert _first(rows, "ScreenDecisionRecorded")["eligible"] is True
    forecast = _first(rows, "ForecastCreated")
    assert forecast["abstention_reason"] == NO_CITATIONS_ABSTENTION
    assert forecast["eligible_for_live"] is False
    assert forecast["probability_ppm"] == DEMO_ASK_PIPS * PPM_PER_PIP
    assert _rows_of(rows, "ProviderVoteRecorded") == []
    assert {
        int(decision["intent_count"])  # type: ignore[call-overload]
        for decision in _rows_of(rows, "SelectorDecisionRecorded")
    } == {0}
    assert _rows_of(rows, "IntentApproved") == []
    assert _rows_of(rows, "OrderTransitionLedgered") == []


def test_the_default_configuration_selects_no_corpus() -> None:
    """`WindbreakConfig()` selects no corpus, so an omitted section cannot trade.

    The structural statement behind the run above: the abstention there is a
    consequence of the shipped default and not of anything the invocation did.
    Read off the dataclass so it stays true of a configuration nobody wrote.
    """
    shipped = WindbreakConfig().forecast.replay_corpus

    assert shipped.mode == REPLAY_CORPUS_DISABLED
    assert shipped.corpus_dir == "configured-by-operator"


def _thin_the_book(tmp_path: Path) -> tuple[Path, Path]:
    """Rest the committed book below the production depth floor (barrier 1a).

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The spoiled books directory and the committed configuration.
    """
    books = _demo_books_copy(tmp_path)
    path = books / "sessions.json"
    sessions = json.loads(path.read_text(encoding="utf-8"))
    for steps in sessions.values():
        for step in steps:
            for side in ("yes_bids", "yes_asks"):
                for level in step["book"][side]:
                    level["quantity"] = THIN_QUANTITY_CENTIS
    path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    return books, DEMO_CONFIG


def _stale_the_close(tmp_path: Path) -> tuple[Path, Path]:
    """Put the committed close outside the production window (barrier 1b).

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The spoiled books directory and the committed configuration.
    """
    books = _demo_books_copy(tmp_path)
    set_close_time(books, days_after_recording=OUT_OF_WINDOW_HORIZON_DAYS)
    return books, DEMO_CONFIG


def _switch_the_corpus_off(tmp_path: Path) -> tuple[Path, Path]:
    """Deselect the corpus, keeping every other declaration (barriers 2 and 4).

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The committed books and a configuration selecting no corpus.
    """
    return DEMO_BOOKS, _config_with(tmp_path, mode=REPLAY_CORPUS_DISABLED)


def _empty_the_corpus_research(tmp_path: Path) -> tuple[Path, Path]:
    """Point at a corpus whose recordings answer a different market (barrier 2).

    Distinct from switching the corpus off: the corpus is selected, loaded, and
    consulted, and it genuinely holds nothing on *this* market's subquestions.
    That is the branch that proves the search key is the subquestion text rather
    than something the transport ignores.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The committed books and a configuration naming the re-keyed corpus.
    """
    corpus = tmp_path / "other-corpus"
    shutil.copytree(DEMO_CORPUS, corpus)
    path = corpus / CORPUS_RESEARCH_FILENAME
    research = json.loads(path.read_text(encoding="utf-8"))
    research["results"] = {
        f"{query} (a different market entirely)": urls
        for query, urls in research["results"].items()
    }
    path.write_text(json.dumps(research, indent=2), encoding="utf-8")
    return DEMO_BOOKS, _config_with(tmp_path, corpus_dir=str(corpus))


def _drop_the_correlation_declaration(tmp_path: Path) -> tuple[Path, Path]:
    """Remove the correlation bucket, keeping the corpus (barrier 3).

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The committed books and a configuration declaring no bucket.
    """
    path = tmp_path / "config.yaml"
    path.write_text(
        "forecast:\n"
        "  replay_corpus:\n"
        f'    mode: "{REPLAY_CORPUS_REPLAY}"\n'
        f'    corpus_dir: "{DEMO_CORPUS}"\n',
        encoding="utf-8",
    )
    return DEMO_BOOKS, path


def _evidence_thin_book(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert the production depth floor is what refused the intent.

    Args:
        rows: Every ledgered ``(event_type, data)`` pair.
    """
    screen = _first(rows, "ScreenDecisionRecorded")
    assert screen["eligible"] is False
    assert screen["blocked_by"] == ["min_depth_contract_centis"]
    assert _rows_of(rows, "ForecastCreated") == []


def _evidence_stale_close(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert the production horizon window is what refused the intent.

    Args:
        rows: Every ledgered ``(event_type, data)`` pair.
    """
    screen = _first(rows, "ScreenDecisionRecorded")
    assert screen["eligible"] is False
    assert screen["blocked_by"] == ["horizon_days"]
    assert _rows_of(rows, "ForecastCreated") == []


def _evidence_no_citations(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert an empty research bundle is what refused the intent.

    Args:
        rows: Every ledgered ``(event_type, data)`` pair.
    """
    assert _first(rows, "ScreenDecisionRecorded")["eligible"] is True
    assert _first(rows, "ForecastCreated")["abstention_reason"] == (
        NO_CITATIONS_ABSTENTION
    )
    assert _rows_of(rows, "ProviderVoteRecorded") == []
    assert "fail:citation_support: citation_count=0" in _reasons(rows)


def _evidence_undeclared_bucket(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert the missing correlation declaration is what refused the intent.

    Args:
        rows: Every ledgered ``(event_type, data)`` pair.
    """
    assert _first(rows, "ForecastCreated")["abstention_reason"] is None
    assert _reasons(rows) == [
        f"unprovable_exposure: no correlation bucket or holding evidence for {TICKER}"
    ]


def _evidence_unproven_providers(rows: list[tuple[str, dict[str, object]]]) -> None:
    """Assert the absent M6 artifact is what refused the intent (barrier 5).

    Args:
        rows: Every ledgered ``(event_type, data)`` pair.
    """
    assert _first(rows, "ProviderGateHeld")["unproven_providers"] == "anthropic,openai"
    assert _first(rows, "ForecastCreated")["eligible_for_live"] is False
    assert "fail:forecast_live_eligible: eligible_for_live=False" in _reasons(rows)


@pytest.mark.parametrize(
    ("spoil", "track_records", "evidence"),
    [
        pytest.param(_thin_the_book, True, _evidence_thin_book, id="thin_book"),
        pytest.param(_stale_the_close, True, _evidence_stale_close, id="stale_close"),
        pytest.param(
            _switch_the_corpus_off, True, _evidence_no_citations, id="corpus_off"
        ),
        pytest.param(
            _empty_the_corpus_research,
            True,
            _evidence_no_citations,
            id="corpus_answers_another_market",
        ),
        pytest.param(
            _drop_the_correlation_declaration,
            True,
            _evidence_undeclared_bucket,
            id="undeclared_bucket",
        ),
        pytest.param(
            lambda _tmp: (DEMO_BOOKS, DEMO_CONFIG),
            False,
            _evidence_unproven_providers,
            id="missing_track_record",
        ),
    ],
)
def test_restoring_any_single_barrier_removes_the_intent_from_the_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spoil: Callable[[Path], tuple[Path, Path]],
    track_records: bool,
    evidence: Callable[[list[tuple[str, dict[str, object]]]], None],
) -> None:
    """Each barrier alone is enough to take the intent away again, from the CLI.

    Starts from the one command line that emits, puts back exactly one blocking
    condition, and pins both that no intent is recorded on any beat *and* the
    ledgered evidence naming what removed it. Asserting each separately is what
    keeps the barriers independently load-bearing: an all-at-once negative would
    still pass if all but one had quietly stopped mattering, and an
    ``intent_count == 0`` with no accompanying evidence passes for any reason at
    all, including an upstream crash.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: The active patcher, used to run from the repository root.
        spoil: The single condition restored, returning the books and config.
        track_records: Whether the M6 artifact is placed for this case.
        evidence: The per-case assertion naming why the intent went away.
    """
    monkeypatch.chdir(REPO_ROOT)
    books, config = spoil(tmp_path)

    exit_code, ledger_path = _run_cli(
        tmp_path, books=books, config=config, track_records=track_records
    )

    assert exit_code == 0
    rows = _ledgered(ledger_path)
    assert {
        int(decision["intent_count"])  # type: ignore[call-overload]
        for decision in _rows_of(rows, "SelectorDecisionRecorded")
    } <= {0}
    assert _rows_of(rows, "IntentApproved") == []
    assert _rows_of(rows, "OrderTransitionLedgered") == []
    evidence(rows)


def test_the_provider_gate_bars_are_the_shipped_ones() -> None:
    """The committed M6 artifact clears the shipped bars rather than lowering them.

    The artifact is a *fixture* placed in the report directory, so the barrier-5
    negative above is only honest if the artifact satisfies the production
    thresholds instead of the thresholds having been relaxed to fit it. Both
    bars are read from :class:`~windbreak.config.schema.ProviderGateConfig`.
    """
    bars = ProviderGateConfig()
    recorded = json.loads(DEMO_TRACK_RECORDS.read_text(encoding="utf-8"))

    assert sorted(recorded) == ["anthropic", "openai"]
    assert all(
        entry["resolved_count"] > bars.min_resolved for entry in recorded.values()
    )
    assert all(
        entry["brier_skill_ppm"] > bars.min_brier_skill_ppm
        for entry in recorded.values()
    )
    assert bars.min_resolved == 150
    assert bars.min_brier_skill_ppm == 10000
