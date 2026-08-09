"""SPEC S8.4/S16 research budget enforcement and cost reporting.

The forecast engine spends real money on research; SPEC S16 caps that spend on
two axes -- a per-forecast micros ceiling and a per-UTC-day micros ceiling --
and bounds how many web pages a single forecast may fetch. :class:`ResearchBudget`
enforces all three fail-closed: :meth:`ResearchBudget.ensure_day_open` halts a
run *before any research* once the day bucket is exhausted, and
:meth:`ResearchBudget.charge_forecast` records a forecast's spend into the day
bucket *first* (so a breached forecast still counts against the day) before
raising on a per-forecast overrun. :func:`report_research_costs` summarizes
accumulated spend into per-resolved-forecast and per-profitable-trade unit
costs.

The three defaults deliberately mirror
:data:`windbreak.forecast.triage.PER_FORECAST_BUDGET_MICROS` and its siblings
rather than importing them: importing ``triage`` here would create a
``budget -> triage -> pipeline -> budget`` import cycle once ``pipeline`` grows a
budget seam, so the constants are restated locally.

Every decision is ledgered through the :class:`BudgetLedgerWriter` seam (modeled
verbatim on :class:`windbreak.forecast.triage.TriageLedgerWriter`) with
``int``/``str``/``bool`` payload leaves only -- never a float, per the
package-wide no-float convention ``scripts/lint_no_floats.py`` enforces. All
money math is integer-only, with the single per-unit division routed through
:func:`windbreak.numeric.rounding.divide` so its rounding direction is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Final, Protocol

from windbreak.numeric.rounding import RoundingDirection, divide

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

#: The per-forecast research budget, in micros (SPEC S16
#: ``budget.per_forecast_micros``). Restated (not imported from ``triage``) to
#: avoid a ``budget -> triage -> pipeline -> budget`` import cycle.
DEFAULT_PER_FORECAST_BUDGET_MICROS = 3_000_000

#: The per-UTC-day research budget, in micros (SPEC S16 ``budget.per_day_micros``).
DEFAULT_PER_DAY_BUDGET_MICROS = 20_000_000

#: The maximum web pages a single forecast may fetch (SPEC S16 ``budget.max_pages``).
DEFAULT_MAX_PAGES = 20

#: Event type recorded when a single forecast exceeds its per-forecast budget.
BUDGET_FORECAST_EXCEEDED_EVENT = "BUDGET_FORECAST_EXCEEDED"

#: Event type recorded when a UTC day's cumulative budget is exhausted.
BUDGET_DAY_EXHAUSTED_EVENT = "BUDGET_DAY_EXHAUSTED"

#: Event type recorded when a research-cost report is produced.
COST_REPORT_EVENT = "COST_REPORT"


def _require_non_negative(value: int, field_name: str) -> None:
    """Guard that a micros/count field is non-negative.

    Args:
        value: The candidate integer.
        field_name: The owning field's name, surfaced in the error message.

    Raises:
        ValueError: If ``value`` is negative. The message names ``field_name``.
    """
    if value < 0:
        msg = f"{field_name} must be non-negative, got {value}"
        raise ValueError(msg)


def _require_positive_micros(value: int, field_name: str) -> None:
    """Guard that a per-attempt price/ceiling field is strictly positive.

    Mirrors :func:`_require_non_negative`'s style but rejects ``0`` as well as
    negatives: a zero-priced entry on this fail-closed money path would let a
    runaway provider spend for free, so every price must be at least one micro.

    Args:
        value: The candidate micros integer.
        field_name: The owning field's name, surfaced in the error message.

    Raises:
        ValueError: If ``value`` is less than ``1``. The message names
            ``field_name``.
    """
    if value < 1:
        msg = f"{field_name} must be at least 1 micro, got {value}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ProviderPriceTable:
    """A fail-closed, never-zero per-provider per-attempt price map (SPEC S16).

    Maps each known provider to its list price for a single forecast attempt, in
    micros, and falls back to :attr:`unknown_provider_price_micros` for any
    provider absent from the map -- so an unrecognized provider is never
    silently treated as free. Every price (mapped or fallback) is validated
    strictly positive at construction, matching the fail-closed money-path
    convention every budget dataclass in this module uses.

    Attributes:
        prices_micros: The per-provider list price for one attempt, in micros.
        unknown_provider_price_micros: The fallback price charged for any
            provider absent from ``prices_micros``, in micros.
    """

    prices_micros: Mapping[str, int]
    unknown_provider_price_micros: int

    def __post_init__(self) -> None:
        """Validate the fallback ceiling and every mapped price are positive.

        Raises:
            ValueError: If the fallback ceiling or any mapped price is below
                ``1`` micro.
        """
        _require_positive_micros(
            self.unknown_provider_price_micros, "unknown_provider_price_micros"
        )
        for provider, price in self.prices_micros.items():
            _require_positive_micros(price, f"prices_micros[{provider!r}]")

    def price_micros(self, provider: str) -> int:
        """Return the per-attempt price for ``provider``, in micros.

        Args:
            provider: The provider identifier to price.

        Returns:
            The mapped per-attempt price, or
            :attr:`unknown_provider_price_micros` when ``provider`` is absent.
        """
        return self.prices_micros.get(provider, self.unknown_provider_price_micros)


#: The fallback per-attempt price charged for any provider absent from a price
#: table, in micros. A deliberately conservative ceiling: an unknown provider is
#: priced high, never free, so an unpriced provider cannot evade its budget.
DEFAULT_UNKNOWN_PROVIDER_PRICE_MICROS: Final = 1_000_000

#: Per-attempt list price for the OpenAI provider, in micros.
_OPENAI_PROVIDER_PRICE_MICROS: Final = 200_000

#: Per-attempt list price for the Anthropic provider, in micros.
_ANTHROPIC_PROVIDER_PRICE_MICROS: Final = 300_000

#: Per-attempt list price for the FutureSearch provider, in micros.
_FUTURESEARCH_PROVIDER_PRICE_MICROS: Final = 500_000

#: The pinned default per-attempt price table covering the three real hosted
#: providers. The network-free fixture provider is deliberately absent: its true
#: cost is ``0`` and rides on ``ProviderForecast.cost_micros`` directly rather
#: than through this fail-closed (never-zero) list-price table.
#:
#: NOTE: sourcing these list prices from ``windbreak.config`` is a follow-up; it
#: is deliberately NOT wired to ``config.schema`` here (SPEC S8.3 keeps the
#: forecast engine free of any config import).
DEFAULT_PROVIDER_PRICE_TABLE: Final[ProviderPriceTable] = ProviderPriceTable(
    prices_micros={
        "openai": _OPENAI_PROVIDER_PRICE_MICROS,
        "anthropic": _ANTHROPIC_PROVIDER_PRICE_MICROS,
        "futuresearch": _FUTURESEARCH_PROVIDER_PRICE_MICROS,
    },
    unknown_provider_price_micros=DEFAULT_UNKNOWN_PROVIDER_PRICE_MICROS,
)


def _utc_day_key(at: datetime) -> str:
    """Return the ISO ``YYYY-MM-DD`` UTC-day key for an instant.

    Args:
        at: The (timezone-aware) instant to bucket.

    Returns:
        The instant's UTC calendar date as an ISO-8601 date string.
    """
    return at.astimezone(UTC).date().isoformat()


def _iso_z(moment: datetime) -> str:
    """Render a datetime as ISO-8601 UTC with a trailing ``Z``.

    Follows the local-``_iso_z`` precedent in ``triage.py``/``pipeline.py``/
    ``records.py`` (each module defines its own) rather than sharing one.

    Args:
        moment: The (timezone-aware) datetime to render; normalized to UTC.

    Returns:
        A string like ``2024-12-10T12:00:00.000000Z``.
    """
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


@dataclass(frozen=True, slots=True)
class BudgetEvent:
    """One recorded budget decision (mirrors ``TriageEvent``).

    Attributes:
        event_type: The event kind (one of the ``BUDGET_*``/``COST_REPORT``
            constants).
        payload: The JSON-safe event body (int/str/bool leaves only).
        ts: ISO-8601 UTC timestamp of when the event was created.
    """

    event_type: str
    payload: Mapping[str, object]
    ts: str


class BudgetLedgerWriter(Protocol):
    """The seam through which a budget decision is persisted."""

    def record(self, event: BudgetEvent) -> None:
        """Persist a budget event.

        Args:
            event: The event to persist.
        """
        ...


class InMemoryBudgetLedger:
    """A :class:`BudgetLedgerWriter` that retains events in memory for tests."""

    def __init__(self) -> None:
        """Initialize with an empty event log."""
        self._events: list[BudgetEvent] = []

    def record(self, event: BudgetEvent) -> None:
        """Append a budget event to the in-memory log.

        Args:
            event: The event to retain.
        """
        self._events.append(event)

    def events_by_type(self, event_type: str) -> tuple[BudgetEvent, ...]:
        """Return every retained event of a given type, in record order.

        Args:
            event_type: The event kind to filter by.

        Returns:
            The matching events.
        """
        return tuple(event for event in self._events if event.event_type == event_type)


class PerForecastBudgetExceededError(Exception):
    """Raised when one forecast's research cost exceeds the per-forecast budget.

    Attributes:
        cost_micros: The forecast's research cost, in micros.
        budget_micros: The per-forecast budget ceiling, in micros.
    """

    def __init__(self, cost_micros: int, budget_micros: int) -> None:
        """Initialize with the offending cost and the breached ceiling.

        Args:
            cost_micros: The forecast's research cost, in micros.
            budget_micros: The per-forecast budget ceiling, in micros.
        """
        self.cost_micros = cost_micros
        self.budget_micros = budget_micros
        super().__init__(
            f"forecast cost {cost_micros} micros exceeds per-forecast budget "
            f"{budget_micros} micros"
        )


class DailyBudgetExhaustedError(Exception):
    """Raised when a UTC day's cumulative research budget is exhausted.

    Attributes:
        spent_micros: The day's cumulative spend at the halt, in micros.
        budget_micros: The per-day budget ceiling, in micros.
        utc_day: The exhausted UTC day, as an ISO ``YYYY-MM-DD`` string.
    """

    def __init__(self, spent_micros: int, budget_micros: int, utc_day: str) -> None:
        """Initialize with the day's spend, its ceiling, and the day key.

        Args:
            spent_micros: The day's cumulative spend at the halt, in micros.
            budget_micros: The per-day budget ceiling, in micros.
            utc_day: The exhausted UTC day, as an ISO ``YYYY-MM-DD`` string.
        """
        self.spent_micros = spent_micros
        self.budget_micros = budget_micros
        self.utc_day = utc_day
        super().__init__(
            f"UTC day {utc_day} budget exhausted: spent {spent_micros} micros of "
            f"{budget_micros} micros"
        )


class ResearchBudget:
    """A per-forecast, per-day, and per-page research spending guard (SPEC S16)."""

    def __init__(
        self,
        *,
        per_forecast_micros: int = DEFAULT_PER_FORECAST_BUDGET_MICROS,
        per_day_micros: int = DEFAULT_PER_DAY_BUDGET_MICROS,
        max_pages: int = DEFAULT_MAX_PAGES,
        ledger: BudgetLedgerWriter,
    ) -> None:
        """Initialize the budget with its three ceilings and an empty day ledger.

        Args:
            per_forecast_micros: The per-forecast spend ceiling, in micros.
            per_day_micros: The per-UTC-day spend ceiling, in micros.
            max_pages: The per-forecast web-page fetch ceiling.
            ledger: The budget-event ledger writer (keyword-only).

        Raises:
            ValueError: If any of the three ceilings is negative.
        """
        _require_non_negative(per_forecast_micros, "per_forecast_micros")
        _require_non_negative(per_day_micros, "per_day_micros")
        _require_non_negative(max_pages, "max_pages")
        self._per_forecast_micros = per_forecast_micros
        self._per_day_micros = per_day_micros
        self._max_pages = max_pages
        self._ledger = ledger
        self._spend_by_day: dict[str, int] = {}

    @property
    def max_pages(self) -> int:
        """Return the per-forecast web-page fetch ceiling."""
        return self._max_pages

    def ensure_day_open(self, *, at: datetime) -> None:
        """Halt before any research if the day's budget is already exhausted.

        Called first on every budgeted run, before any tool or transport is
        touched. The day bucket is keyed by ``at``'s UTC calendar date, so the
        counter resets (not decays) at each UTC midnight.

        Args:
            at: The run's creation instant, bucketing it to a UTC day
                (keyword-only).

        Raises:
            DailyBudgetExhaustedError: If cumulative spend for ``at``'s UTC day
                is at or above the per-day ceiling. A ``BUDGET_DAY_EXHAUSTED``
                event is ledgered before the raise.
        """
        day = _utc_day_key(at)
        spent = self._spend_by_day.get(day, 0)
        if spent >= self._per_day_micros:
            payload: dict[str, object] = {
                "utc_day": day,
                "spent_micros": spent,
                "budget_micros": self._per_day_micros,
            }
            self._ledger.record(
                BudgetEvent(BUDGET_DAY_EXHAUSTED_EVENT, payload, _iso_z(at))
            )
            raise DailyBudgetExhaustedError(spent, self._per_day_micros, day)

    def charge_forecast(
        self, cost_micros: int, *, market_ticker: str, at: datetime
    ) -> None:
        """Charge one forecast's research cost, fail-closed on a per-forecast overrun.

        The spend lands in the day bucket *first*, so a breached forecast still
        counts against the day. The per-forecast ceiling is inclusive: a cost
        exactly equal to it passes; only a strictly greater cost breaches.

        Args:
            cost_micros: The forecast's research cost, in micros.
            market_ticker: The forecasted market's ticker, for the audit trail
                (keyword-only).
            at: The run's creation instant, bucketing the spend to a UTC day
                (keyword-only).

        Raises:
            ValueError: If ``cost_micros`` is negative.
            PerForecastBudgetExceededError: If ``cost_micros`` strictly exceeds
                the per-forecast ceiling. A ``BUDGET_FORECAST_EXCEEDED`` event is
                ledgered before the raise.
        """
        _require_non_negative(cost_micros, "cost_micros")
        day = _utc_day_key(at)
        self._spend_by_day[day] = self._spend_by_day.get(day, 0) + cost_micros
        if cost_micros > self._per_forecast_micros:
            payload: dict[str, object] = {
                "cost_micros": cost_micros,
                "budget_micros": self._per_forecast_micros,
                "market_ticker": market_ticker,
                "utc_day": day,
            }
            self._ledger.record(
                BudgetEvent(BUDGET_FORECAST_EXCEEDED_EVENT, payload, _iso_z(at))
            )
            raise PerForecastBudgetExceededError(cost_micros, self._per_forecast_micros)


@dataclass(frozen=True, slots=True)
class CostReport:
    """A research-cost summary with per-unit figures (SPEC S16).

    Attributes:
        total_research_cost_micros: Total research cost summarized, in micros.
        resolved_forecast_count: How many forecasts resolved.
        profitable_trade_count: How many trades were profitable.
        cost_per_resolved_forecast_micros: Cost per resolved forecast, in micros,
            or ``None`` when no forecast resolved.
        cost_per_profitable_trade_micros: Cost per profitable trade, in micros,
            or ``None`` when no trade was profitable.
    """

    total_research_cost_micros: int
    resolved_forecast_count: int
    profitable_trade_count: int
    cost_per_resolved_forecast_micros: int | None
    cost_per_profitable_trade_micros: int | None


def _cost_per_unit(total_micros: int, count: int) -> int | None:
    """Divide a total cost by a unit count, rounding costs up (ceiling).

    Rounding is ``OVERSTATE_COST`` (ceiling) because this unit cost feeds the
    promotion-gate expectancy check: understating cost would flatter the edge,
    so a remainder is always dropped toward the more conservative figure.

    Args:
        total_micros: The total cost to spread, in micros.
        count: The number of units to spread it across.

    Returns:
        The per-unit cost in micros, or ``None`` when ``count`` is zero.
    """
    if count == 0:
        return None
    return divide(total_micros, count, rounding=RoundingDirection.OVERSTATE_COST)


def report_research_costs(
    *,
    total_research_cost_micros: int,
    resolved_forecast_count: int,
    profitable_trade_count: int,
    ledger: BudgetLedgerWriter,
    at: datetime,
) -> CostReport:
    """Summarize research spend into per-unit costs and ledger the report.

    Each per-unit figure is a ceiling division (see :func:`_cost_per_unit`); a
    zero denominator yields a ``None`` field whose key is omitted entirely from
    the ledgered ``COST_REPORT`` payload.

    Args:
        total_research_cost_micros: Total research cost summarized, in micros.
        resolved_forecast_count: How many forecasts resolved.
        profitable_trade_count: How many trades were profitable.
        ledger: The budget-event ledger writer (keyword-only).
        at: The report's creation instant (keyword-only).

    Returns:
        The produced :class:`CostReport`.

    Raises:
        ValueError: If any cost or count input is negative.
    """
    _require_non_negative(total_research_cost_micros, "total_research_cost_micros")
    _require_non_negative(resolved_forecast_count, "resolved_forecast_count")
    _require_non_negative(profitable_trade_count, "profitable_trade_count")
    per_resolved = _cost_per_unit(total_research_cost_micros, resolved_forecast_count)
    per_profitable = _cost_per_unit(total_research_cost_micros, profitable_trade_count)
    report = CostReport(
        total_research_cost_micros=total_research_cost_micros,
        resolved_forecast_count=resolved_forecast_count,
        profitable_trade_count=profitable_trade_count,
        cost_per_resolved_forecast_micros=per_resolved,
        cost_per_profitable_trade_micros=per_profitable,
    )
    payload: dict[str, object] = {
        "total_research_cost_micros": total_research_cost_micros,
        "resolved_forecast_count": resolved_forecast_count,
        "profitable_trade_count": profitable_trade_count,
    }
    if per_resolved is not None:
        payload["cost_per_resolved_forecast_micros"] = per_resolved
    if per_profitable is not None:
        payload["cost_per_profitable_trade_micros"] = per_profitable
    ledger.record(BudgetEvent(COST_REPORT_EVENT, payload, _iso_z(at)))
    return report
