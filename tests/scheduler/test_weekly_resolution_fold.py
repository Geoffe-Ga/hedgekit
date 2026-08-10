"""The weekly fold adjudicates real resolutions against real forecasts (#439).

`weekly_report_body` used to hardcode `resolutions={}` and
`deployment_sequence=0`, so every metric in the always-on loop's only
evaluation consumer read `UNDEFINED` after seven days exactly as it did after
seven minutes -- not "not enough data yet" but structurally unreachable. These
tests pin the fold against a ledger that carries ground truth:

1. **A metric moves.** `brier` reads `UNDEFINED` over a ledger holding a
   forecast alone, and reads an exact integer over the same ledger once a
   `MarketResolved` row is appended -- computed through the real
   `weekly_report_body`, not a hand-built `EvaluationInputs`.
2. **The negative.** Fold the same records *without* the resolution row and the
   metric returns to `UNDEFINED`, so the number in (1) is attributable to the
   ingested resolution and to nothing else.
3. **Temporal integrity, both directions, with exact instants.** A forecast
   created before its market's resolution instant is admitted and scored; one
   created after it is refused `backdated` and named in the report's rejection
   ledger. Both resolutions are ingested *last*, so the gate demonstrably keys
   on when the market settled rather than on when the operator noticed.
4. **The deployment boundary is real.** A forecast that predates the ledger's
   `ConfigLoaded` marker is `pre_deployment`, and a fold carrying no marker at
   all refuses every forecast rather than admitting them on trust.

The expectation never comes from the thing being checked (#422): the Brier
value asserted here is computed by hand from the forecast's own
`probability_ppm` and the outcome the operator ingested, which enters from
outside the forecast record entirely.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from windbreak.evaluation.ingest import MarketResolved
from windbreak.evaluation.resolution import ResolutionOutcome
from windbreak.ledger.events import ConfigLoaded, Event, ForecastCreated
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.scheduler.weekly_data import weekly_report_body

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from windbreak.ledger.store import LedgerRecord

#: The component label stamped on every event this suite ledgers.
_COMPONENT = "scheduler"

#: The report date stamped into every rendered body here.
_TODAY = date(2026, 3, 8)

#: Base instant the per-row clock steps forward from.
_BASE = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)

#: The forecast probability the scored forecasts carry, in ppm.
_PROBABILITY_PPM = 250_000

#: The exact mean Brier of one `_PROBABILITY_PPM` forecast on a market that
#: settled ``NO``: ``(250_000 - 0)^2 / (1 * 1_000_000)`` ppm, computed here from
#: the forecast's probability and the ingested outcome rather than read back off
#: the report the assertion checks.
_BRIER_OF_ONE_NO_MARKET_PPM = (_PROBABILITY_PPM - 0) ** 2 // 1_000_000


def _stepping_clock(instants: list[datetime]) -> Iterator[datetime]:
    """Yield each instant in turn, so every appended row gets a chosen one.

    Args:
        instants: The instants to stamp, one per append, in append order.

    Yields:
        Each instant in `instants`, in order.
    """
    yield from instants


class _ScriptedStore:
    """A `SqliteLedgerStore` whose `created_at` stamps follow a fixed script.

    The temporal projection under test reads each row's `created_at`, so a test
    that cannot choose those stamps cannot distinguish "the resolution instant
    gated this" from "the append order happened to gate this".
    """

    def __init__(self, db_path: Path, instants: list[datetime]) -> None:
        """Open a store that stamps `instants` in order, one per append.

        Args:
            db_path: The SQLite database path to open or create.
            instants: The `created_at` instants to stamp, in append order.
        """
        self._clock = _stepping_clock(instants)
        self.store = SqliteLedgerStore(db_path, now=lambda: next(self._clock))

    def append(self, event: Event) -> int:
        """Append one event, stamping the script's next instant.

        Args:
            event: The event to persist.

        Returns:
            The sequence number assigned to the new record.
        """
        return self.store.append(event)

    def read_all(self) -> list[LedgerRecord]:
        """Return every persisted record in ascending sequence order.

        Returns:
            The full ledger read.
        """
        return self.store.read_all()


def _config_loaded() -> ConfigLoaded:
    """Build the deployment marker every real run writes as its first row.

    Returns:
        A `ConfigLoaded` event.
    """
    return ConfigLoaded(component=_COMPONENT, config_hash="deadbeef", diff={})


def _forecast(
    *, forecast_id: str, market_ticker: str, probability_ppm: int = _PROBABILITY_PPM
) -> ForecastCreated:
    """Build one v2-shaped `ForecastCreated` event.

    Args:
        forecast_id: The forecast's deterministic id.
        market_ticker: The market the forecast is for.
        probability_ppm: The forecast probability, in ppm.

    Returns:
        The assembled `ForecastCreated` event.
    """
    return ForecastCreated(
        component=_COMPONENT,
        forecast_id=forecast_id,
        market_ticker=market_ticker,
        probability_ppm=probability_ppm,
        eligible_for_live=False,
        abstention_reason=None,
        research_cost_micros=1_000_000,
        market_price_baseline_pips=4600,
    )


def _resolved(
    *, market_ticker: str, resolved_at: datetime, outcome: ResolutionOutcome
) -> MarketResolved:
    """Build one operator-ingested `MarketResolved` event.

    Args:
        market_ticker: The market that settled.
        resolved_at: The exact instant it settled.
        outcome: The settled outcome.

    Returns:
        The assembled `MarketResolved` event.
    """
    return MarketResolved(
        component=_COMPONENT,
        market_ticker=market_ticker,
        outcome=outcome,
        resolved_at=resolved_at,
        source="kalshi-settlement-notice",
    )


def _metric_line(body: str, name: str) -> str:
    """Return the single rendered line for one metric name.

    Args:
        body: The rendered weekly-report body.
        name: The metric's registry name.

    Returns:
        The one matching `name [window] = value` line.

    Raises:
        AssertionError: If the body does not carry exactly one such line.
    """
    matches = [line for line in body.splitlines() if line.startswith(f"{name} [")]
    assert len(matches) == 1, f"expected exactly one {name!r} line, got {matches}"
    return matches[0]


def _rejection_lines(body: str) -> list[str]:
    """Return every rendered temporal-integrity rejection line.

    Args:
        body: The rendered weekly-report body.

    Returns:
        The rejection ledger's lines, in render order.
    """
    return [
        line
        for line in body.splitlines()
        if line.startswith("EVALUATION_RECORD_REJECTED")
    ]


def _cost_meter_line(body: str, label: str) -> str:
    """Return the single Cost meter line carrying `label`.

    Args:
        body: The rendered weekly-report body.
        label: The cost-meter field label.

    Returns:
        The one matching `label: value` line.

    Raises:
        AssertionError: If the body does not carry exactly one such line.
    """
    section = body.split("## Cost meter", 1)[1]
    matches = [line for line in section.splitlines() if line.startswith(f"{label}: ")]
    assert len(matches) == 1, f"expected exactly one {label!r} line, got {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# 1./2. A metric moves off UNDEFINED on ingestion, and back on its removal.
# ---------------------------------------------------------------------------


def test_brier_moves_from_undefined_to_an_exact_value_on_ingestion(
    tmp_path: Path,
) -> None:
    """Ingesting one resolution makes `brier` computable through the real fold.

    Before the `MarketResolved` row exists the metric is structurally
    unreachable; after it, it reads the exact hand-computed Brier of the single
    forecast against the outcome the operator ingested.
    """
    store = _ScriptedStore(
        tmp_path / "ledger.db",
        [
            _BASE,
            _BASE + timedelta(hours=1),
            _BASE + timedelta(days=7),
        ],
    )
    store.append(_config_loaded())
    store.append(_forecast(forecast_id="fc-a", market_ticker="MKT-A"))

    before = weekly_report_body(store.read_all(), today=_TODAY)

    assert _metric_line(before, "brier") == "brier [latest_before_close] = UNDEFINED"
    assert _cost_meter_line(before, "resolved forecasts") == "resolved forecasts: 0"

    store.append(
        _resolved(
            market_ticker="MKT-A",
            resolved_at=_BASE + timedelta(hours=2),
            outcome=ResolutionOutcome.NO,
        )
    )

    after = weekly_report_body(store.read_all(), today=_TODAY)

    assert _metric_line(after, "brier") == (
        f"brier [latest_before_close] = {_BRIER_OF_ONE_NO_MARKET_PPM}"
    )
    assert _BRIER_OF_ONE_NO_MARKET_PPM == 62_500
    assert _cost_meter_line(after, "resolved forecasts") == "resolved forecasts: 1"
    assert _rejection_lines(after) == []


def test_dropping_the_ingested_resolution_returns_brier_to_undefined(
    tmp_path: Path,
) -> None:
    """The metric is attributable to the resolution row and to nothing else.

    Folding the identical records with the one `MarketResolved` row removed
    puts `brier` straight back to `UNDEFINED` -- so the value in the sibling
    test cannot be coming from the forecast rows, the deployment marker, or the
    fold's own defaults.
    """
    store = _ScriptedStore(
        tmp_path / "ledger.db",
        [_BASE, _BASE + timedelta(hours=1), _BASE + timedelta(days=7)],
    )
    store.append(_config_loaded())
    store.append(_forecast(forecast_id="fc-a", market_ticker="MKT-A"))
    store.append(
        _resolved(
            market_ticker="MKT-A",
            resolved_at=_BASE + timedelta(hours=2),
            outcome=ResolutionOutcome.NO,
        )
    )
    records = store.read_all()
    without_resolution = [
        record for record in records if record.event_type != "MarketResolved"
    ]

    assert len(without_resolution) == len(records) - 1

    with_resolution_body = weekly_report_body(records, today=_TODAY)
    without_resolution_body = weekly_report_body(without_resolution, today=_TODAY)

    assert _metric_line(with_resolution_body, "brier") == (
        f"brier [latest_before_close] = {_BRIER_OF_ONE_NO_MARKET_PPM}"
    )
    assert _metric_line(without_resolution_body, "brier") == (
        "brier [latest_before_close] = UNDEFINED"
    )
    assert (
        _cost_meter_line(without_resolution_body, "resolved forecasts")
        == "resolved forecasts: 0"
    )


# ---------------------------------------------------------------------------
# 3. Temporal integrity, both directions, driven by exact resolution instants.
# ---------------------------------------------------------------------------


def test_a_forecast_predating_its_resolution_is_scored_and_one_postdating_it_is_not(
    tmp_path: Path,
) -> None:
    """One forecast is admitted, the other refused `backdated`, on instants alone.

    Both markets settle at 02:00. `fc-early` was created at 01:00 and is
    scored; `fc-late` was created at 03:00 -- after its market's answer was
    knowable -- and is refused. Both resolutions are ingested a week later, so
    a gate keying on the ingesting row's own ledger position would wrongly
    admit both. The two forecasts carry *different* probabilities, so the
    Brier value distinguishes "only `fc-early` scored" from "both scored".
    """
    store = _ScriptedStore(
        tmp_path / "ledger.db",
        [
            _BASE,
            _BASE + timedelta(hours=1),
            _BASE + timedelta(hours=3),
            _BASE + timedelta(days=7),
            _BASE + timedelta(days=7, seconds=1),
        ],
    )
    store.append(_config_loaded())
    store.append(_forecast(forecast_id="fc-early", market_ticker="MKT-EARLY"))
    store.append(
        _forecast(
            forecast_id="fc-late", market_ticker="MKT-LATE", probability_ppm=900_000
        )
    )
    settled_at = _BASE + timedelta(hours=2)
    store.append(
        _resolved(
            market_ticker="MKT-EARLY",
            resolved_at=settled_at,
            outcome=ResolutionOutcome.NO,
        )
    )
    store.append(
        _resolved(
            market_ticker="MKT-LATE",
            resolved_at=settled_at,
            outcome=ResolutionOutcome.NO,
        )
    )

    body = weekly_report_body(store.read_all(), today=_TODAY)

    assert _rejection_lines(body) == [
        "EVALUATION_RECORD_REJECTED fc-late MKT-LATE backdated"
    ]
    assert _metric_line(body, "brier") == (
        f"brier [latest_before_close] = {_BRIER_OF_ONE_NO_MARKET_PPM}"
    )
    # Had `fc-late` also been scored the mean would be 436_250, not 62_500.
    both_scored_ppm = ((_PROBABILITY_PPM**2) + (900_000**2)) // 2_000_000
    assert both_scored_ppm != _BRIER_OF_ONE_NO_MARKET_PPM
    assert str(both_scored_ppm) not in _metric_line(body, "brier")


def test_a_resolution_after_every_ledger_row_refuses_nothing(tmp_path: Path) -> None:
    """A market settling after the last recorded row cannot have leaked into it.

    The mirror of the `backdated` direction: the same two forecasts, with the
    settlement instant moved past the end of the ledger, are both admitted and
    both scored.
    """
    store = _ScriptedStore(
        tmp_path / "ledger.db",
        [
            _BASE,
            _BASE + timedelta(hours=1),
            _BASE + timedelta(hours=3),
            _BASE + timedelta(hours=4),
            _BASE + timedelta(hours=5),
        ],
    )
    store.append(_config_loaded())
    store.append(_forecast(forecast_id="fc-early", market_ticker="MKT-EARLY"))
    store.append(
        _forecast(
            forecast_id="fc-late", market_ticker="MKT-LATE", probability_ppm=900_000
        )
    )
    settles_later = _BASE + timedelta(days=365)
    store.append(
        _resolved(
            market_ticker="MKT-EARLY",
            resolved_at=settles_later,
            outcome=ResolutionOutcome.NO,
        )
    )
    store.append(
        _resolved(
            market_ticker="MKT-LATE",
            resolved_at=settles_later,
            outcome=ResolutionOutcome.NO,
        )
    )

    body = weekly_report_body(store.read_all(), today=_TODAY)

    assert _rejection_lines(body) == []
    both_scored_ppm = ((_PROBABILITY_PPM**2) + (900_000**2)) // 2_000_000
    assert _metric_line(body, "brier") == (
        f"brier [latest_before_close] = {both_scored_ppm}"
    )
    assert _cost_meter_line(body, "resolved forecasts") == "resolved forecasts: 2"


# ---------------------------------------------------------------------------
# 4. The deployment boundary is a real ledger fact, and fails closed.
# ---------------------------------------------------------------------------


def test_a_forecast_predating_the_deployment_marker_is_refused(
    tmp_path: Path,
) -> None:
    """A forecast written before the ledger's `ConfigLoaded` row is pre-deployment.

    The marker sits at sequence 2, so the forecast at sequence 1 is refused
    even though its market resolved afterwards -- a forecast that predates the
    system's first recorded act cannot be an honest prediction by it.
    """
    store = _ScriptedStore(
        tmp_path / "ledger.db",
        [
            _BASE,
            _BASE + timedelta(hours=1),
            _BASE + timedelta(days=7),
        ],
    )
    store.append(_forecast(forecast_id="fc-seeded", market_ticker="MKT-A"))
    store.append(_config_loaded())
    store.append(
        _resolved(
            market_ticker="MKT-A",
            resolved_at=_BASE + timedelta(days=6),
            outcome=ResolutionOutcome.NO,
        )
    )

    body = weekly_report_body(store.read_all(), today=_TODAY)

    assert _rejection_lines(body) == [
        "EVALUATION_RECORD_REJECTED fc-seeded MKT-A pre_deployment"
    ]
    assert _metric_line(body, "brier") == "brier [latest_before_close] = UNDEFINED"


def test_deployment_is_the_first_config_marker_not_the_latest_restart(
    tmp_path: Path,
) -> None:
    """A restart writes another marker; deployment is still the first one.

    The ledger carries two `ConfigLoaded` rows -- the original run and a
    restart at sequence 3. Taking the *latest* marker would retroactively place
    the deployment boundary after a forecast the system had genuinely already
    made, refusing real evidence as `pre_deployment`. Deployment is the
    earliest marker, mirroring `deployment_sequence_from_fixture`'s minimum
    over `mode_transitions`.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    store = _ScriptedStore(
        tmp_path / "ledger.db",
        [
            _BASE,
            _BASE + timedelta(hours=1),
            _BASE + timedelta(hours=2),
            _BASE + timedelta(hours=3),
        ],
    )
    store.append(_config_loaded())
    store.append(_forecast(forecast_id="fc-a", market_ticker="MKT-A"))
    store.append(_config_loaded())
    store.append(
        _resolved(
            market_ticker="MKT-A",
            resolved_at=_BASE + timedelta(days=30),
            outcome=ResolutionOutcome.NO,
        )
    )

    body = weekly_report_body(store.read_all(), today=_TODAY)

    assert _rejection_lines(body) == []
    assert _metric_line(body, "brier") == (
        f"brier [latest_before_close] = {_BRIER_OF_ONE_NO_MARKET_PPM}"
    )


def test_a_fold_with_no_deployment_marker_refuses_every_forecast(
    tmp_path: Path,
) -> None:
    """No `ConfigLoaded` row means no provable deployment point, so nothing scores.

    Absent evidence must never read as healthy: a ledger slice that cannot
    show when the system deployed refuses its forecasts loudly in the
    rejection ledger rather than admitting them and reporting a number.
    """
    store = _ScriptedStore(
        tmp_path / "ledger.db",
        [_BASE + timedelta(hours=1), _BASE + timedelta(days=7)],
    )
    store.append(_forecast(forecast_id="fc-a", market_ticker="MKT-A"))
    store.append(
        _resolved(
            market_ticker="MKT-A",
            resolved_at=_BASE + timedelta(days=6),
            outcome=ResolutionOutcome.NO,
        )
    )

    body = weekly_report_body(store.read_all(), today=_TODAY)

    assert _rejection_lines(body) == [
        "EVALUATION_RECORD_REJECTED fc-a MKT-A pre_deployment"
    ]
    assert _metric_line(body, "brier") == "brier [latest_before_close] = UNDEFINED"


# ---------------------------------------------------------------------------
# 5. A malformed ledger timestamp is named, not silently skipped.
# ---------------------------------------------------------------------------


def test_an_unparseable_created_at_is_refused_naming_the_offending_row(
    tmp_path: Path,
) -> None:
    """A row whose `created_at` is not ISO-8601 raises, naming its sequence.

    The projection compares each row's creation instant against the settlement
    instant, so a row it cannot read is a hole in the ordering. Skipping such a
    row would quietly move the projected sequence and change which forecasts
    are scored, so the fold refuses instead -- and names the row so an operator
    can find it on a live ledger.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    store = _ScriptedStore(
        tmp_path / "ledger.db",
        [_BASE, _BASE + timedelta(hours=1), _BASE + timedelta(days=7)],
    )
    store.append(_config_loaded())
    store.append(_forecast(forecast_id="fc-a", market_ticker="MKT-A"))
    store.append(
        _resolved(
            market_ticker="MKT-A",
            resolved_at=_BASE + timedelta(days=6),
            outcome=ResolutionOutcome.NO,
        )
    )
    corrupted = [
        replace(record, created_at="whenever")
        if record.sequence_number == 2
        else record
        for record in store.read_all()
    ]

    with pytest.raises(ValueError) as exc_info:
        weekly_report_body(corrupted, today=_TODAY)

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == (
        "ledger row at sequence_number=2 carries an unparseable created_at: 'whenever'"
    )
