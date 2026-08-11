"""The daily research ceiling is durable, day-scoped, and live (#442, #483).

Issue #442's reproduction, as a test, plus everything it takes to believe the
fix. ``ResearchBudget`` kept its per-UTC-day counter in an instance ``dict``
that ``_build_research_budget`` rebuilt empty on every process start, and only
*breaches* were ledgered, so nothing on disk said what the day had spent. With
``restart: on-failure`` in ``deploy/docker-compose.yml`` and ``Restart=on-
failure`` in every ``deploy/systemd/*.service``, the per-day ceiling was a
per-*process* ceiling and the process count per day is unbounded.

That mattered little while issue #483's per-trade research charge refused every
market and held spend near zero by accident. It matters entirely once that
charge is gone, which is why the two land together: the daily cap is now the
only governor of research spend, and a governor that resets on restart is not
one.

Four properties are pinned here, and each is pinned from *both* sides so a
guard that always fires reads as the failure it would be:

1. The day's spend survives a restart -- and the *next* day is still open, so
   rehydration is day-scoped rather than a blanket halt.
2. The day boundary is **UTC**. CI runs UTC and cannot see a timezone bug, so
   the two rollovers are driven under a pinned UTC-05:00 process timezone: a
   local midnight must not reset the day, and a UTC midnight must.
3. The cap binds when exhausted, and the refusal is ledgered.
4. The cap is adjustable at runtime through a ledgered
   ``ResearchBudgetCapSet`` row: raising it lets the identical run proceed,
   with no restart and no source change.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.conftest import pinned_local_timezone
from windbreak.config.schema import (
    ForecastBudget,
    ForecastConfig,
    WindbreakConfig,
)
from windbreak.forecast.budget import (
    DailyBudgetExhaustedError,
    PerForecastBudgetExceededError,
)
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.scheduler.loop import build_paper_deps
from windbreak.scheduler.research_spend import (
    RESEARCH_BUDGET_CAP_SET_EVENT_TYPE,
    RESEARCH_SPEND_RECORDED_EVENT_TYPE,
    ResearchBudgetCapSet,
    ResearchSpendRecorded,
    effective_per_day_micros,
    spend_by_day_from_records,
)

if TYPE_CHECKING:
    from windbreak.scheduler.loop import PaperTickDeps

#: The books fixture every bundle below replays. Its contents are irrelevant
#: here -- no tick is run -- but ``build_paper_deps`` needs a real directory.
BOOKS_DIR = Path("tests/fixtures/books/deep_walk")

#: The market ticker every charge in this module is attributed to.
TICKER = "MKT-DEEP"

#: The per-UTC-day research ceiling every bundle here is built with, in micros.
#: Small enough that one charge exhausts it exactly, so the boundary the tests
#: cross is the ceiling itself and not an accumulation artefact.
DAY_CAP_MICROS = 1_000_000

#: A raised ceiling, used to prove the cap binding above is the cap and not
#: some other refusal: with this in force the identical spend proceeds.
RAISED_DAY_CAP_MICROS = 5_000_000

#: Midday on the reference UTC day, comfortably inside both the UTC and the
#: UTC-05:00 calendar day so neither rollover test starts near a boundary.
MIDDAY = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)

#: 04:00Z on the reference day. Under a UTC-05:00 local zone this is 23:00 on
#: the *previous* local day, so it and :data:`AFTER_LOCAL_MIDNIGHT` sit on
#: opposite local days while sharing one UTC day.
BEFORE_LOCAL_MIDNIGHT = datetime(2026, 8, 9, 4, 0, tzinfo=UTC)

#: 06:00Z on the reference day: 01:00 local, the next local day, same UTC day.
AFTER_LOCAL_MIDNIGHT = datetime(2026, 8, 9, 6, 0, tzinfo=UTC)

#: 23:00Z on the reference day: 18:00 local. Paired with
#: :data:`AFTER_UTC_MIDNIGHT` these share one *local* day and straddle UTC
#: midnight -- the mirror image of the pair above.
BEFORE_UTC_MIDNIGHT = datetime(2026, 8, 9, 23, 0, tzinfo=UTC)

#: 01:00Z the next day: 20:00 local on the same local day as the instant above.
AFTER_UTC_MIDNIGHT = datetime(2026, 8, 10, 1, 0, tzinfo=UTC)

#: The UTC day key :data:`MIDDAY` buckets to.
MIDDAY_UTC_DAY = "2026-08-09"


def _config(*, per_day_micros: int = DAY_CAP_MICROS) -> WindbreakConfig:
    """Build a configuration whose only non-default is the per-day ceiling.

    Args:
        per_day_micros: The per-UTC-day research ceiling, in micros.

    Returns:
        The constructed configuration.
    """
    return WindbreakConfig(
        forecast=ForecastConfig(budget=ForecastBudget(per_day_micros=per_day_micros))
    )


def _deps(tmp_path: Path, *, per_day_micros: int = DAY_CAP_MICROS) -> PaperTickDeps:
    """Wire one dependency bundle over a *shared* ledger path.

    Every bundle this module builds points at the same ``ledger.db`` under
    ``tmp_path``, which is what makes "restart" mean what issue #442 means by
    it: a second process reading the first one's durable state.

    Args:
        tmp_path: pytest's per-test temporary directory.
        per_day_micros: The per-UTC-day research ceiling, in micros.

    Returns:
        The wired dependencies. The caller must close ``deps.store``.
    """
    cassette = tmp_path / "cassettes.json"
    if not cassette.exists():
        cassette.write_text("{}", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir(exist_ok=True)
    return build_paper_deps(
        books_dir=BOOKS_DIR,
        cassette_path=cassette,
        ledger_path=tmp_path / "ledger.db",
        report_dir=reports,
        config=_config(per_day_micros=per_day_micros),
    )


def _spend(deps: PaperTickDeps, micros: int, *, at: datetime) -> None:
    """Charge research against a bundle's budget.

    Args:
        deps: The bundle whose budget is charged.
        micros: The charge, in micros.
        at: The instant bucketing the charge to a UTC day.
    """
    deps.budget.charge_forecast(micros, market_ticker=TICKER, at=at)


def _rows(ledger_path: Path) -> list[tuple[str, dict[str, object]]]:
    """Read every persisted row straight out of SQLite.

    Reading through ``sqlite3`` rather than the open store proves the rows
    reached disk rather than an in-process buffer -- the whole question issue
    #442 turns on.

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


def test_the_days_spend_survives_a_process_restart(tmp_path: Path) -> None:
    """Issue #442's reproduction, inverted: the second process is refused.

    The first bundle spends the whole day cap -- in two unequal instalments,
    neither of which alone reaches it -- and closes its store, exactly as a
    crash-and-restart leaves things. The second bundle is built from scratch
    over the *same* ledger path and the *same* UTC day, and must refuse to open
    the day. Splitting the spend is what makes this test measure the *fold*
    rather than merely the presence of a row: a rehydration that read only the
    last charge would see 750 000 against a 1 000 000 ceiling and re-open.

    The raise is checked by exact type and full message equality rather than
    ``pytest.raises(..., match=...)``: a subclass, or a message naming
    different figures, would satisfy a looser check while proving nothing about
    what the second process actually knew.
    """
    first = _deps(tmp_path)
    _spend(first, DAY_CAP_MICROS // 4, at=MIDDAY)
    _spend(first, DAY_CAP_MICROS - DAY_CAP_MICROS // 4, at=MIDDAY)
    first.store.close()

    second = _deps(tmp_path)
    try:
        with pytest.raises(DailyBudgetExhaustedError) as caught:
            second.budget.ensure_day_open(at=MIDDAY)
    finally:
        second.store.close()

    assert type(caught.value) is DailyBudgetExhaustedError
    assert caught.value.spent_micros == DAY_CAP_MICROS
    assert caught.value.budget_micros == DAY_CAP_MICROS
    assert caught.value.utc_day == MIDDAY_UTC_DAY
    assert str(caught.value) == (
        f"UTC day {MIDDAY_UTC_DAY} budget exhausted: spent {DAY_CAP_MICROS} "
        f"micros of {DAY_CAP_MICROS} micros"
    )


def test_the_restarted_process_still_opens_the_next_utc_day(tmp_path: Path) -> None:
    """Rehydration is day-scoped: tomorrow is open even though today is shut.

    The negative control for the test above. A rehydration that halted every
    day, or that folded all recorded spend into one bucket, would pass that
    test and fail this one -- so the pair distinguishes a per-day counter from a
    blanket refusal.
    """
    first = _deps(tmp_path)
    _spend(first, DAY_CAP_MICROS, at=MIDDAY)
    first.store.close()

    second = _deps(tmp_path)
    try:
        with pytest.raises(DailyBudgetExhaustedError):
            second.budget.ensure_day_open(at=MIDDAY)
        second.budget.ensure_day_open(at=MIDDAY + timedelta(days=1))
    finally:
        second.store.close()


def test_a_local_midnight_rollover_does_not_reopen_the_day(tmp_path: Path) -> None:
    """Crossing *local* midnight leaves the UTC day's ceiling shut.

    Run under a pinned fixed-offset UTC-05:00 process timezone, because CI runs
    UTC and a day-keying bug that reads the host's local calendar is invisible
    there: on a UTC host both readings agree and the test passes either way.
    Under this pin, 04:00Z and 06:00Z on 2026-08-09 fall on *different* local
    days and the same UTC day, so a budget keyed on local time would re-open
    here and a budget keyed on UTC must not.
    """
    with pinned_local_timezone("EST5"):
        first = _deps(tmp_path)
        _spend(first, DAY_CAP_MICROS, at=BEFORE_LOCAL_MIDNIGHT)
        first.store.close()

        second = _deps(tmp_path)
        try:
            with pytest.raises(DailyBudgetExhaustedError) as caught:
                second.budget.ensure_day_open(at=AFTER_LOCAL_MIDNIGHT)
        finally:
            second.store.close()

    assert caught.value.utc_day == MIDDAY_UTC_DAY
    assert caught.value.spent_micros == DAY_CAP_MICROS


def test_a_utc_midnight_rollover_does_reopen_the_day(tmp_path: Path) -> None:
    """Crossing *UTC* midnight opens a fresh ceiling, local day unchanged.

    The mirror of the test above and the half that keeps it honest. 23:00Z and
    01:00Z straddle UTC midnight while sitting on one local day under the same
    UTC-05:00 pin, so a budget keyed on local time would stay shut here. A fix
    that simply never reset would pass the previous test; it fails this one.
    """
    with pinned_local_timezone("EST5"):
        first = _deps(tmp_path)
        _spend(first, DAY_CAP_MICROS, at=BEFORE_UTC_MIDNIGHT)
        first.store.close()

        second = _deps(tmp_path)
        try:
            with pytest.raises(DailyBudgetExhaustedError):
                second.budget.ensure_day_open(at=BEFORE_UTC_MIDNIGHT)
            second.budget.ensure_day_open(at=AFTER_UTC_MIDNIGHT)
        finally:
            second.store.close()


def test_every_successful_charge_reaches_the_ledger_and_the_fold_sums_them(
    tmp_path: Path,
) -> None:
    """Each charge is one row, and the day fold **adds** the rows up.

    Exact counts and exact payloads, not "at least one": two charges must leave
    two rows carrying their own costs, so a writer that collapsed them, dropped
    one, or wrote a running total fails here.

    The fold is then asserted on the same ledger, and the two charges are
    deliberately *unequal and both smaller than their total*, so a fold that
    took the last row (600 000), the first (400 000), or the maximum (600 000)
    is a different number from the sum (1 000 000). That separation is the
    whole point: with two equal charges, or with one, "sum" and "last" coincide
    and neither assertion would be measuring anything.
    """
    deps = _deps(tmp_path, per_day_micros=RAISED_DAY_CAP_MICROS)
    try:
        _spend(deps, 400_000, at=MIDDAY)
        _spend(deps, 600_000, at=MIDDAY)
        folded = spend_by_day_from_records(deps.store.read_all())
    finally:
        deps.store.close()

    spend_rows = [
        data
        for event_type, data in _rows(tmp_path / "ledger.db")
        if event_type == RESEARCH_SPEND_RECORDED_EVENT_TYPE
    ]

    assert len(spend_rows) == 2
    assert [row["cost_micros"] for row in spend_rows] == [400_000, 600_000]
    assert {row["utc_day"] for row in spend_rows} == {MIDDAY_UTC_DAY}
    assert {row["market_ticker"] for row in spend_rows} == {TICKER}
    assert folded == {MIDDAY_UTC_DAY: 1_000_000}
    assert folded[MIDDAY_UTC_DAY] not in (400_000, 600_000)


def test_a_breaching_forecast_still_leaves_its_spend_on_the_ledger(
    tmp_path: Path,
) -> None:
    """Money already committed is durable even when the charge goes on to raise.

    ``charge_forecast`` books the spend into the day bucket *before* it checks
    the per-forecast ceiling, so a breaching forecast still counts against the
    day. The ledger row has to follow the same order for the same reason: a
    breaching forecast has by definition spent the most, and recording it only
    on the way out would lose exactly the spend most worth remembering -- a
    restart after a breach would then resume the day undercounted.

    Both halves are asserted from one call. The raise happens, and the row is
    there anyway, and the restarted process sees the spend.
    """
    first = _deps(tmp_path, per_day_micros=RAISED_DAY_CAP_MICROS)
    over_ceiling = first.config.forecast.budget.per_forecast_micros + 1
    try:
        with pytest.raises(PerForecastBudgetExceededError):
            _spend(first, over_ceiling, at=MIDDAY)
    finally:
        first.store.close()

    spend_rows = [
        data
        for event_type, data in _rows(tmp_path / "ledger.db")
        if event_type == RESEARCH_SPEND_RECORDED_EVENT_TYPE
    ]
    second = _deps(tmp_path, per_day_micros=RAISED_DAY_CAP_MICROS)
    try:
        rehydrated = spend_by_day_from_records(second.store.read_all())
    finally:
        second.store.close()

    assert len(spend_rows) == 1
    assert spend_rows[0]["cost_micros"] == over_ceiling
    assert rehydrated == {MIDDAY_UTC_DAY: over_ceiling}


def test_the_exhausted_day_ledgers_its_refusal(tmp_path: Path) -> None:
    """A refused day writes exactly one ``ResearchBudgetHalted`` row.

    The refusal has to be auditable, not merely raised: an operator reading the
    ledger must be able to tell a spent budget from a quiet loop. The count is
    exact and the payload's figures are compared in full.
    """
    first = _deps(tmp_path)
    _spend(first, DAY_CAP_MICROS, at=MIDDAY)
    first.store.close()

    second = _deps(tmp_path)
    try:
        with pytest.raises(DailyBudgetExhaustedError):
            second.budget.ensure_day_open(at=MIDDAY)
    finally:
        second.store.close()

    halts = [
        data
        for event_type, data in _rows(tmp_path / "ledger.db")
        if event_type == "ResearchBudgetHalted"
    ]

    assert len(halts) == 1
    assert halts[0] == {
        "market_ticker": "",
        "halt_kind": "per_day",
        "utc_day": MIDDAY_UTC_DAY,
        "spent_micros": DAY_CAP_MICROS,
        "budget_micros": DAY_CAP_MICROS,
    }


def test_a_ledgered_cap_raise_lets_the_identical_spend_proceed(
    tmp_path: Path,
) -> None:
    """Raising the cap at runtime re-opens the same day, no restart, no source edit.

    The negative half of the binding test above, and issue #483's acceptance
    criterion 3 in full: the *same* ledger, the *same* UTC day and the *same*
    already-spent total, differing only by an operator-appended
    ``ResearchBudgetCapSet`` row, opens where it had refused. Both branches are
    asserted in one test so a cap that binds unconditionally cannot pass.

    The configuration handed to every bundle here still carries the original
    ``DAY_CAP_MICROS``, so the raise can only be coming from the ledger.
    """
    first = _deps(tmp_path)
    _spend(first, DAY_CAP_MICROS, at=MIDDAY)
    first.store.close()

    refused = _deps(tmp_path)
    try:
        with pytest.raises(DailyBudgetExhaustedError):
            refused.budget.ensure_day_open(at=MIDDAY)
    finally:
        refused.store.close()

    store = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        store.append(
            ResearchBudgetCapSet(
                component="operator",
                per_day_micros=RAISED_DAY_CAP_MICROS,
                note="investigating a quiet day; raising the ceiling",
            )
        )
    finally:
        store.close()

    reopened = _deps(tmp_path)
    try:
        reopened.budget.ensure_day_open(at=MIDDAY)
        _spend(reopened, 1, at=MIDDAY)
    finally:
        reopened.store.close()

    assert _config().forecast.budget.per_day_micros == DAY_CAP_MICROS


def test_the_latest_cap_row_wins_and_can_lower_as_well_as_raise(
    tmp_path: Path,
) -> None:
    """Two cap rows never conflict: the later one is simply in force.

    Pinned in both directions off one ledger, so "last row wins" is shown
    rather than asserted, and a fold that took the maximum, the minimum, or the
    first row fails. This is where the PR #482 lesson lands: a cap is an
    instruction, not evidence, so no pair of rows can ever be a contradiction
    and no fold of them can ever brick a tick.
    """
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        store.append(
            ResearchBudgetCapSet(
                component="operator", per_day_micros=9_000_000, note="raise"
            )
        )
        after_raise = effective_per_day_micros(
            store.read_all(), configured_micros=DAY_CAP_MICROS
        )
        store.append(
            ResearchBudgetCapSet(
                component="operator", per_day_micros=2_000_000, note="lower again"
            )
        )
        after_lower = effective_per_day_micros(
            store.read_all(), configured_micros=DAY_CAP_MICROS
        )
        rows = [
            event_type
            for event_type, _ in [
                (record.event_type, record) for record in store.read_all()
            ]
        ]
    finally:
        store.close()

    assert after_raise == 9_000_000
    assert after_lower == 2_000_000
    assert rows.count(RESEARCH_BUDGET_CAP_SET_EVENT_TYPE) == 2


def test_the_configured_ceiling_stands_when_the_ledger_carries_no_cap_row(
    tmp_path: Path,
) -> None:
    """A deployment that never runs the verb is unaffected by it.

    Asserted against a ledger that is genuinely non-empty -- it carries spend
    rows -- so the fallback is shown to be "no cap row", not "no rows at all".
    """
    deps = _deps(tmp_path)
    try:
        _spend(deps, 1, at=MIDDAY)
        records = tuple(deps.store.read_all())
        effective = effective_per_day_micros(records, configured_micros=DAY_CAP_MICROS)
    finally:
        deps.store.close()

    assert records
    assert effective == DAY_CAP_MICROS


@pytest.mark.parametrize("missing_key", ["cost_micros", "utc_day"])
def test_a_spend_row_missing_a_required_field_is_refused_not_skipped(
    tmp_path: Path, missing_key: str
) -> None:
    """An unreadable charge fails the fold closed rather than undercounting.

    Skipping a malformed row would silently lower the day's spend and re-open a
    ceiling that should be shut -- the exact runaway this machinery exists to
    prevent -- so the fold raises. Both required fields are swept, because they
    fail through different guards: a charge with no amount and a charge with no
    day are equally unfoldable, and only one of the two was covered before this
    was parametrized.

    The message is compared in full, so it is required to name the offending
    key *and* the row's sequence number -- a live ledger's bad row has to be
    locatable, not merely reported.

    ``windbreak set-research-budget`` and the budget writer both validate
    before appending, so this row is not one either of them can produce; the
    guard is the last line of defence for a ledger they did not write.

    Args:
        tmp_path: pytest's per-test temporary directory.
        missing_key: The payload key removed from the row.
    """
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        store.append(
            ResearchSpendRecorded(
                component="scheduler",
                utc_day=MIDDAY_UTC_DAY,
                market_ticker=TICKER,
                cost_micros=123,
            )
        )
    finally:
        store.close()
    _drop_payload_key(tmp_path / "ledger.db", missing_key)

    reader = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        with pytest.raises(ValueError) as caught:
            spend_by_day_from_records(reader.read_all())
    finally:
        reader.close()

    assert str(caught.value) == (
        f"ResearchSpendRecorded payload at sequence_number=1 is missing {missing_key!r}"
    )


def test_a_wrongly_typed_payload_field_is_refused_on_both_folds(
    tmp_path: Path,
) -> None:
    """A field of the wrong *type* fails closed as loudly as a missing one.

    Two distinct holes, closed separately. A ``cost_micros`` that arrives as a
    string would raise deep inside the summation with an unlocatable
    ``TypeError``, and a ``per_day_micros`` that arrives as a string would be
    handed to ``ResearchBudget`` as a ceiling nothing can compare against. Both
    are refused where the row is read, with the row's sequence number in the
    message.

    ``True`` is checked explicitly for ``per_day_micros`` because ``bool`` is a
    subclass of ``int`` in Python: a JSON ``true`` would otherwise pass an
    ``isinstance(value, int)`` guard and become a one-micro daily ceiling --
    silently halting all research on the next tick.
    """
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        store.append(
            ResearchSpendRecorded(
                component="scheduler",
                utc_day=MIDDAY_UTC_DAY,
                market_ticker=TICKER,
                cost_micros=123,
            )
        )
        store.append(
            ResearchBudgetCapSet(component="operator", per_day_micros=1, note="n")
        )
    finally:
        store.close()
    _retype_payload_key(tmp_path / "ledger.db", 1, "cost_micros", "123")
    _retype_payload_key(tmp_path / "ledger.db", 2, "per_day_micros", True)

    reader = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        with pytest.raises(ValueError) as spend_failure:
            spend_by_day_from_records(reader.read_all())
        with pytest.raises(ValueError) as cap_failure:
            effective_per_day_micros(
                reader.read_all(), configured_micros=DAY_CAP_MICROS
            )
    finally:
        reader.close()

    assert str(spend_failure.value) == (
        "ResearchSpendRecorded payload at sequence_number=1 carries a "
        "non-integer 'cost_micros': '123'"
    )
    assert str(cap_failure.value) == (
        "ResearchBudgetCapSet payload at sequence_number=2 carries a "
        "non-integer 'per_day_micros': True"
    )


def test_a_spend_row_with_a_non_string_day_is_refused(tmp_path: Path) -> None:
    """A ``utc_day`` that is not a string cannot key a bucket, so it is refused.

    Silently coercing it would bucket the charge under a key no other row uses,
    which reads as "that day spent nothing" -- the undercount this fold exists
    to make impossible.
    """
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        store.append(
            ResearchSpendRecorded(
                component="scheduler",
                utc_day=MIDDAY_UTC_DAY,
                market_ticker=TICKER,
                cost_micros=123,
            )
        )
    finally:
        store.close()
    _retype_payload_key(tmp_path / "ledger.db", 1, "utc_day", 20260809)

    reader = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        with pytest.raises(ValueError) as caught:
            spend_by_day_from_records(reader.read_all())
    finally:
        reader.close()

    assert str(caught.value) == (
        "ResearchSpendRecorded payload at sequence_number=1 carries a "
        "non-string 'utc_day': 20260809"
    )


def test_a_cap_row_missing_its_ceiling_is_refused(tmp_path: Path) -> None:
    """A cap row with no ``per_day_micros`` names the key and the row.

    Falling back to the configured ceiling here would be the dangerous
    direction: an operator who *lowered* the cap would find their instruction
    silently discarded and the higher configured ceiling back in force.
    """
    store = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        store.append(
            ResearchBudgetCapSet(component="operator", per_day_micros=7, note="n")
        )
    finally:
        store.close()
    _drop_payload_key(tmp_path / "ledger.db", "per_day_micros")

    reader = SqliteLedgerStore(tmp_path / "ledger.db")
    try:
        with pytest.raises(ValueError) as caught:
            effective_per_day_micros(
                reader.read_all(), configured_micros=DAY_CAP_MICROS
            )
    finally:
        reader.close()

    assert str(caught.value) == (
        "ResearchBudgetCapSet payload at sequence_number=1 is missing 'per_day_micros'"
    )


def test_a_recorded_spend_refuses_a_negative_cost() -> None:
    """A negative charge is refused at construction, never credited to a day.

    A negative cost would *reduce* the day's counter, so an event carrying one
    is not a bad record but a hole in the ceiling. Asserted by exact type and
    full message so a differently-caused ``ValueError`` cannot stand in.
    """
    with pytest.raises(ValueError) as caught:
        ResearchSpendRecorded(
            component="scheduler",
            utc_day=MIDDAY_UTC_DAY,
            market_ticker=TICKER,
            cost_micros=-1,
        )

    assert type(caught.value) is ValueError
    assert str(caught.value) == "cost_micros must be non-negative, got -1"


def test_a_cap_row_refuses_a_negative_ceiling_and_a_blank_note() -> None:
    """Both of a cap row's own invariants are refused at construction.

    A negative ceiling is unenforceable, and an unexplained money-ceiling change
    cannot be reviewed. A zero ceiling is *not* refused, and that is asserted
    alongside: stopping all research spend is a legitimate operator instruction,
    and refusing it would leave the operator no way to halt spend without a
    restart.
    """
    with pytest.raises(ValueError) as negative:
        ResearchBudgetCapSet(component="operator", per_day_micros=-1, note="n")
    with pytest.raises(ValueError) as blank:
        ResearchBudgetCapSet(component="operator", per_day_micros=1, note="   ")
    zero = ResearchBudgetCapSet(
        component="operator", per_day_micros=0, note="halt all research spend"
    )

    assert str(negative.value) == "per_day_micros must be non-negative, got -1"
    assert str(blank.value) == "note must be non-blank"
    assert zero.payload["per_day_micros"] == 0


def test_an_unknown_payload_schema_version_is_refused_on_a_spend_row(
    tmp_path: Path,
) -> None:
    """``payload_schema_version`` is *read*, so versioning it means something.

    It was written and never consulted: both folds went straight to
    ``envelope["data"]`` and subscripted version-1 keys whatever the row said
    it was. A field that is only ever written cannot make a shape change
    survivable -- a v2 payload would either be mis-read as a v1 one, or be
    reported as "missing 'cost_micros'" when the key had merely been renamed,
    and either way every ledger carrying v1 rows would be unusable the day a v2
    shipped.

    Reading it turns that into a refusal that names the version, which is what
    a future v2 reader dispatches on.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    ledger = tmp_path / "ledger.db"
    store = SqliteLedgerStore(ledger)
    try:
        store.append(
            ResearchSpendRecorded(
                component="scheduler",
                utc_day=MIDDAY_UTC_DAY,
                market_ticker=TICKER,
                cost_micros=123,
            )
        )
    finally:
        store.close()
    _reversion_row(ledger, 1, 2)

    reader = SqliteLedgerStore(ledger)
    try:
        with pytest.raises(ValueError) as caught:
            spend_by_day_from_records(reader.read_all())
    finally:
        reader.close()

    assert str(caught.value) == (
        "ResearchSpendRecorded payload at sequence_number=1 carries "
        "payload_schema_version=2, which this fold cannot read (it reads "
        "version 1)"
    )


def test_an_unknown_payload_schema_version_is_refused_on_a_cap_row(
    tmp_path: Path,
) -> None:
    """The cap fold reads the version too, so neither guard is decorative.

    Swept separately from the spend fold above: both folds write the field, and
    a guard added to only one of them would leave the other exactly as
    decorative as before.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    ledger = tmp_path / "ledger.db"
    store = SqliteLedgerStore(ledger)
    try:
        store.append(
            ResearchBudgetCapSet(component="operator", per_day_micros=9, note="n")
        )
    finally:
        store.close()
    _reversion_row(ledger, 1, 2)

    reader = SqliteLedgerStore(ledger)
    try:
        with pytest.raises(ValueError) as caught:
            effective_per_day_micros(
                reader.read_all(), configured_micros=DAY_CAP_MICROS
            )
    finally:
        reader.close()

    assert str(caught.value) == (
        "ResearchBudgetCapSet payload at sequence_number=1 carries "
        "payload_schema_version=2, which this fold cannot read (it reads "
        "version 1)"
    )


def test_a_row_stamped_with_the_readable_version_is_still_folded(
    tmp_path: Path,
) -> None:
    """The version guard refuses version 2 and *accepts* version 1.

    Without this half the guard above would pass just as happily if it refused
    every row unconditionally, which would brick the fold rather than version
    it.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    ledger = tmp_path / "ledger.db"
    store = SqliteLedgerStore(ledger)
    try:
        store.append(
            ResearchSpendRecorded(
                component="scheduler",
                utc_day=MIDDAY_UTC_DAY,
                market_ticker=TICKER,
                cost_micros=123,
            )
        )
        records = store.read_all()
        folded = spend_by_day_from_records(records)
    finally:
        store.close()

    assert [record.payload_schema_version for record in records] == [1]
    assert folded == {MIDDAY_UTC_DAY: 123}


def test_an_unreadable_spend_row_opens_a_zero_ceiling_instead_of_refusing_to_compose(
    tmp_path: Path,
) -> None:
    """A malformed row fails closed on the *budget*, never on the process.

    The blast radius this bounds: ``spend_by_day_from_records`` is called from
    ``_build_research_budget``, which ``build_paper_deps`` calls, so letting the
    refusal escape stopped the loop from *composing at all* -- no heartbeat, no
    equity sample, no positions snapshot, no verification cycle, and no kill
    handling -- under a ``restart: on-failure`` supervisor that retries forever,
    over a row that cannot be deleted from an append-only hash-chained ledger.

    A loop that cannot start cannot honour a kill file. A loop that runs with
    research disabled can. So the refusal becomes the strictest budget there
    is: a zero ceiling on an empty counter. The bundle composes, and the very
    first ``ensure_day_open`` refuses with a ceiling of zero -- asserted by full
    message equality, so a budget that quietly kept the configured 1 000 000
    ceiling could not pass.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    seeding = _deps(tmp_path)
    try:
        _spend(seeding, 1, at=MIDDAY)
    finally:
        seeding.store.close()
    _drop_payload_key(tmp_path / "ledger.db", "cost_micros")

    restarted = _deps(tmp_path)
    try:
        with pytest.raises(DailyBudgetExhaustedError) as caught:
            restarted.budget.ensure_day_open(at=MIDDAY)
    finally:
        restarted.store.close()

    halts = [
        data
        for event_type, data in _rows(tmp_path / "ledger.db")
        if event_type == "ResearchBudgetHalted"
    ]
    assert type(caught.value) is DailyBudgetExhaustedError
    assert str(caught.value) == (
        f"UTC day {MIDDAY_UTC_DAY} budget exhausted: spent 0 micros of 0 micros"
    )
    assert restarted.config.forecast.budget.per_day_micros == DAY_CAP_MICROS
    assert len(halts) == 1
    assert halts[0] == {
        "market_ticker": "",
        "halt_kind": "per_day",
        "utc_day": MIDDAY_UTC_DAY,
        "spent_micros": 0,
        "budget_micros": 0,
    }


def test_an_unreadable_cap_row_also_opens_a_zero_ceiling(tmp_path: Path) -> None:
    """The cap fold's refusal is caught on the same terms as the spend fold's.

    Both folds run behind one ``try``, so a fix that caught only the spend
    fold would leave a malformed ``ResearchBudgetCapSet`` row -- the rows an
    operator's own verb writes -- still terminal.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    ledger = tmp_path / "ledger.db"
    store = SqliteLedgerStore(ledger)
    try:
        store.append(
            ResearchBudgetCapSet(
                component="operator", per_day_micros=RAISED_DAY_CAP_MICROS, note="n"
            )
        )
    finally:
        store.close()
    _drop_payload_key(ledger, "per_day_micros")

    deps = _deps(tmp_path)
    try:
        with pytest.raises(DailyBudgetExhaustedError) as caught:
            deps.budget.ensure_day_open(at=MIDDAY)
    finally:
        deps.store.close()

    assert str(caught.value) == (
        f"UTC day {MIDDAY_UTC_DAY} budget exhausted: spent 0 micros of 0 micros"
    )


def _reversion_row(ledger_path: Path, sequence_number: int, version: int) -> None:
    """Rewrite one row's ``payload_schema_version`` column, in place.

    Fabricates the future-schema row the version guard exists for. Written
    straight through ``sqlite3`` because the typed events stamp the version
    themselves, which is the point.

    Args:
        ledger_path: The ledger database to rewrite.
        sequence_number: Which row to restamp.
        version: The schema version to write.
    """
    conn = sqlite3.connect(ledger_path)
    try:
        conn.execute(
            "UPDATE ledger SET payload_schema_version = ? WHERE sequence_number = ?",
            (version, sequence_number),
        )
        conn.commit()
    finally:
        conn.close()


def _drop_payload_key(ledger_path: Path, key: str) -> None:
    """Delete one key from every row's typed payload, in place.

    Fabricates the malformed row the fold's guard exists for. It is written
    straight through ``sqlite3`` because the typed event refuses to construct
    without the key, which is the point.

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


def _retype_payload_key(
    ledger_path: Path, sequence_number: int, key: str, value: object
) -> None:
    """Replace one payload field's value, in place, with a wrongly-typed one.

    Fabricates the malformed row the folds' type guards exist for. Written
    straight through ``sqlite3`` because the typed events refuse to construct
    with a wrongly-typed field, which is the point.

    Args:
        ledger_path: The ledger database to rewrite.
        sequence_number: Which row to rewrite.
        key: The payload key to retype.
        value: The value to write in its place.
    """
    conn = sqlite3.connect(ledger_path)
    try:
        (payload_json,) = conn.execute(
            "SELECT payload_json FROM ledger WHERE sequence_number = ?",
            (sequence_number,),
        ).fetchone()
        envelope = json.loads(str(payload_json))
        envelope["data"][key] = value
        conn.execute(
            "UPDATE ledger SET payload_json = ? WHERE sequence_number = ?",
            (json.dumps(envelope), sequence_number),
        )
        conn.commit()
    finally:
        conn.close()
