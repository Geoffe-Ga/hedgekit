"""SPEC S8.4/S16 research budget enforcement and cost reporting.

The forecast engine spends real money on research; SPEC S16 caps that spend on
two axes -- a per-forecast micros ceiling and a per-UTC-day micros ceiling --
and bounds how many web pages a single forecast may fetch. :class:`ResearchBudget`
enforces all three fail-closed: :meth:`ResearchBudget.ensure_day_open` halts a
run *before any research* once the day bucket is exhausted, and
:meth:`ResearchBudget.charge_forecast` records a forecast's spend into the day
bucket *first* (so a breached forecast still counts against the day) before
raising on a per-forecast overrun. A forecast that spends in stages (SPEC
S8.4's cheap triage prior, then the full pipeline) charges each stage through
:meth:`ResearchBudget.charge_stage`, which keeps the per-forecast ceiling
checked against their *total* -- a ceiling applied per stage instead could
never trip on the only figure that matters. :func:`report_research_costs`
summarizes accumulated spend into per-resolved-forecast and
per-profitable-trade unit costs.

This module is the single definition site for the per-forecast ceiling *and*
for every term it is derived from (issue #394): ``pipeline`` and ``triage``
import their research-cost constants from here rather than restating them. The
direction matters. This module imports nothing from ``windbreak.forecast``, so
``pipeline -> budget`` and ``triage -> budget`` are both acyclic; importing
``triage`` *here* would close a ``budget -> triage -> pipeline -> budget``
cycle. Keeping the arrows pointing this way is what lets the ceiling and its
terms be one set of numbers instead of four copies that once silently
collided.

Every decision is ledgered through the :class:`BudgetLedgerWriter` seam (modeled
verbatim on :class:`windbreak.forecast.triage.TriageLedgerWriter`) with
``int``/``str``/``bool`` payload leaves only -- never a float, per the
package-wide no-float convention ``scripts/lint_no_floats.py`` enforces. All
money math is integer-only, with the single per-unit division routed through
:func:`windbreak.numeric.rounding.divide` so its rounding direction is explicit.

THE DAY COUNTER IS DURABLE FROM OUTSIDE (issue #442)

Two changes make the per-UTC-day ceiling a per-*day* ceiling rather than a
per-*process* one. :meth:`ResearchBudget.charge_forecast` ledgers a
``BUDGET_SPEND_RECORDED`` event on every successful charge, not only on a
breach, so what a day spent exists somewhere other than this object's memory.
And :meth:`ResearchBudget.__init__` accepts the day counter it opens with, so a
composition root holding durable state can rehydrate it -- the scheduler folds
those rows back through
:func:`windbreak.scheduler.research_spend.spend_by_day_from_records` before the
first tick of every process.

The direction of that arrow is the point: this module still imports nothing
from the ledger (SPEC S8.3 keeps the forecast engine free of it), so it names an
event and takes a mapping, and the scheduler owns both the persistence and the
replay. Before this, ``restart: on-failure`` plus an unbounded restart count
meant the day's ceiling reset as often as the process died.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Final, Protocol

from windbreak.numeric.rounding import RoundingDirection, divide
from windbreak.timekeeping import iso_z, require_aware

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

#: The fixed research charge one full-pipeline run incurs, in micros. The
#: definition site for :data:`windbreak.forecast.pipeline._RESEARCH_COST_MICROS`,
#: which imports it from here (SPEC S16).
FULL_PIPELINE_RESEARCH_COST_MICROS: Final = 3_000_000

#: Divisor turning the full-pipeline research charge into the Stage-0 prior's
#: cost: SPEC S8.4 caps the cheap prior at ~2% (1/50) of the full pipeline it
#: exists to avoid paying for. Deliberately a fraction of the *pipeline charge*
#: and not of the per-forecast ceiling -- the latter would be self-referential,
#: since the ceiling is itself derived from this cost (issue #394).
STAGE0_COST_DIVISOR: Final = 50

#: The Stage-0 triage prior's cost, in micros. The definition site for
#: ``windbreak.forecast.triage``'s Stage-0 charge, which imports it from here.
STAGE0_PRIOR_COST_MICROS: Final = (
    FULL_PIPELINE_RESEARCH_COST_MICROS // STAGE0_COST_DIVISOR
)

#: How many members the default vote ensemble runs. Restated rather than
#: imported from :data:`windbreak.forecast.providers.DEFAULT_VOTE_ENSEMBLE`,
#: which would close a ``budget -> providers -> budget`` cycle (``providers``
#: imports :class:`ProviderPriceTable` from here); pinned mirror-equal by test.
DEFAULT_VOTE_ENSEMBLE_SIZE: Final = 3

#: The most one ensemble member may accrue across *all* its retry attempts, in
#: micros. Restated from ``RetryPolicy.max_cost_micros`` (same cycle reason as
#: above), and pinned mirror-equal by test. This is a hard bound, not an
#: estimate: ``RetryingProvider`` checks ``accrued + price`` against it before
#: every attempt and re-checks the total on success, so a member's booked spend
#: can never exceed it however the price table is configured.
DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS: Final = 1_000_000

#: The per-forecast research budget, in micros (SPEC S16
#: ``budget.per_forecast_micros``).
#:
#: Derived, never invented: it is exactly what the most expensive *correct*
#: triaged PROCEED-path forecast can spend -- the Stage-0 prior, plus the full
#: pipeline's research charge, plus every ensemble member spending its entire
#: retry allowance. The intended headroom above that worst case is **zero**,
#: and that is a deliberate choice in both directions (issue #394):
#:
#: * Anything *lower* breaches on a healthy run. The old value equalled
#:   :data:`FULL_PIPELINE_RESEARCH_COST_MICROS` exactly, so research alone
#:   consumed the whole ceiling and a triaged run failed closed before it could
#:   produce a forecast at all.
#: * Anything *higher* is slack the guard cannot see into. A ceiling set
#:   comfortably above any reachable cost still exists, still gets ledgered,
#:   and can never fire -- the unfalsifiable-guard defect this codebase has
#:   fixed repeatedly.
#:
#: Raising any term therefore raises this ceiling, and a test pins every term
#: and the total as literals so an edit to one cannot silently outgrow it.
DEFAULT_PER_FORECAST_BUDGET_MICROS: Final = (
    STAGE0_PRIOR_COST_MICROS
    + FULL_PIPELINE_RESEARCH_COST_MICROS
    + DEFAULT_VOTE_ENSEMBLE_SIZE * DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS
)

#: The per-UTC-day research budget, in micros (SPEC S16 ``budget.per_day_micros``).
DEFAULT_PER_DAY_BUDGET_MICROS = 20_000_000

#: The maximum web pages a single forecast may fetch (SPEC S16 ``budget.max_pages``).
DEFAULT_MAX_PAGES = 20

#: Event type recorded when a single forecast exceeds its per-forecast budget.
BUDGET_FORECAST_EXCEEDED_EVENT = "BUDGET_FORECAST_EXCEEDED"

#: Event type recorded when a UTC day's cumulative budget is exhausted.
BUDGET_DAY_EXHAUSTED_EVENT = "BUDGET_DAY_EXHAUSTED"

#: Event type recorded on every *successful* research charge (issue #442).
#:
#: Until this existed, only breaches were ledgered, so nothing on disk said what
#: a UTC day had already spent. :meth:`ResearchBudget.__init__` takes the day
#: counter as an argument precisely so a composition root can fold these rows
#: back and re-open a process on the day's real total instead of zero; the
#: scheduler does that in :mod:`windbreak.scheduler.research_spend`. This module
#: stays free of any ledger import (SPEC S8.3), so it names the event and lets
#: the writer seam translate it.
BUDGET_SPEND_RECORDED_EVENT = "BUDGET_SPEND_RECORDED"

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


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """One response's provider-reported token accounting (issue #451).

    The measured quantity a metered charge is derived from, carried verbatim
    from the provider's own ``usage`` block. Both counts are whole tokens --
    every provider reports them as integers, so nothing on this path needs a
    float, and a fractional count would be evidence the envelope was
    misparsed rather than a quantity to round.

    Absence is modelled by the *absence of this object* (a ``None`` usage), not
    by a zero-filled one: a response whose usage cannot be read is unpriceable
    and must fail closed to :attr:`ModelRateTable.unmetered_micros`, whereas a
    ``TokenUsage(0, 0)`` would otherwise price it at nothing.

    Attributes:
        input_tokens: Prompt (input) tokens the provider reported billing.
        output_tokens: Completion (output) tokens the provider reported billing.
    """

    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        """Validate both counts are non-negative, non-``bool`` integers.

        Raises:
            TypeError: If either count is a ``bool`` or not an ``int``. A
                ``bool`` is an ``int`` subclass and must never masquerade as a
                token count.
            ValueError: If either count is negative.
        """
        for field_name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                msg = f"{field_name} must be a non-bool int, got {type(value).__name__}"
                raise TypeError(msg)
            _require_non_negative(value, field_name)

    @property
    def total_tokens(self) -> int:
        """Return the summed input and output token counts."""
        return self.input_tokens + self.output_tokens


#: The denominator every per-token rate is quoted over. Rates are expressed as
#: *micros per million tokens* rather than micros per token because that is the
#: unit vendors publish and, decisively, the only one that stays an exact
#: integer: OpenAI's ``$1.25 / 1M`` input rate is ``1.25`` micros per token --
#: unrepresentable on this fixed-point money path -- but exactly ``1_250_000``
#: micros per million tokens.
MICROS_PER_MILLION_TOKENS_DENOMINATOR: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelTokenRate:
    """One model's published input/output token rates (issue #451).

    Both rates are validated strictly positive for the same fail-closed reason
    :func:`_require_positive_micros` exists: a zero-rated model would bill
    nothing however many tokens it consumed, which is precisely the
    unbounded-spend hole a metered charge exists to close.

    Attributes:
        model_version: The pinned model version these rates price.
        input_micros_per_million_tokens: The input (prompt) token rate, in
            micros per million tokens.
        output_micros_per_million_tokens: The output (completion) token rate, in
            micros per million tokens.
    """

    model_version: str
    input_micros_per_million_tokens: int
    output_micros_per_million_tokens: int

    def __post_init__(self) -> None:
        """Validate both rates are strictly positive.

        Raises:
            ValueError: If either rate is below ``1`` micro per million tokens.
        """
        _require_positive_micros(
            self.input_micros_per_million_tokens,
            f"input_micros_per_million_tokens[{self.model_version!r}]",
        )
        _require_positive_micros(
            self.output_micros_per_million_tokens,
            f"output_micros_per_million_tokens[{self.model_version!r}]",
        )


@dataclass(frozen=True, slots=True)
class ModelRateTable:
    """A fail-closed per-model metered-cost table (issue #451).

    Turns a response's reported :class:`TokenUsage` into micros at the rates
    configured for the model that produced it. Three distinct situations are
    all "this response's cost is not knowable", and all three charge
    :attr:`unmetered_micros` rather than anything derived:

    * the response carried no usage block, or one that could not be parsed
      (``usage is None``);
    * the producing model has no configured rate;
    * the usage block parsed but reports **zero** total tokens. A completion
      that billed no tokens at all did not happen; treating that as a free
      call would hand any provider a one-field way to bill nothing.

    :attr:`unmetered_micros` is deliberately *not* the provider's list price
    (that would restore exactly the flat charge issue #451 removed) and
    deliberately never zero.

    Attributes:
        rates: The per-model-version token rates.
        unmetered_micros: The fail-closed charge for a response whose cost
            cannot be derived, in micros.
    """

    rates: Mapping[str, ModelTokenRate]
    unmetered_micros: int

    def __post_init__(self) -> None:
        """Validate the fail-closed charge is positive.

        Each :class:`ModelTokenRate` validates its own rates at construction,
        so this checks only the table-level fallback.

        Raises:
            ValueError: If ``unmetered_micros`` is below ``1`` micro.
        """
        _require_positive_micros(self.unmetered_micros, "unmetered_micros")

    def micros_for(self, *, model_version: str, usage: TokenUsage | None) -> int:
        """Return what one response cost, in micros.

        The single rounding decision on this path is a ceiling
        (:attr:`~windbreak.numeric.rounding.RoundingDirection.OVERSTATE_COST`)
        over the *summed* token value, so a partial micro is resolved against
        the operator and in favour of the guard: the budget can only ever
        believe a vote cost slightly more than it did, never less, and the
        ceiling therefore binds a hair early rather than a hair late.

        Args:
            model_version: The producing model's pinned version (keyword-only).
            usage: The response's reported token accounting, or ``None`` when
                the response carried none (keyword-only).

        Returns:
            The metered cost in micros, or :attr:`unmetered_micros` when the
            cost cannot be derived.
        """
        rate = self.rates.get(model_version)
        if usage is None or rate is None or usage.total_tokens == 0:
            return self.unmetered_micros
        return divide(
            usage.input_tokens * rate.input_micros_per_million_tokens
            + usage.output_tokens * rate.output_micros_per_million_tokens,
            MICROS_PER_MILLION_TOKENS_DENOMINATOR,
            rounding=RoundingDirection.OVERSTATE_COST,
        )


#: Per-attempt list price for the OpenAI provider, in micros.
_OPENAI_PROVIDER_PRICE_MICROS: Final = 200_000

#: Per-attempt list price for the Anthropic provider, in micros.
_ANTHROPIC_PROVIDER_PRICE_MICROS: Final = 300_000

#: The pinned default per-attempt price table covering the two providers whose
#: votes ride the routed completion seam.
#:
#: This table is the *pre-gate estimate* a
#: :class:`~windbreak.forecast.providers.retry.RetryingProvider` checks
#: affordability against before each attempt, and nothing else reads it. Two
#: providers are therefore deliberately absent, for one shared reason: their
#: true cost rides on ``ProviderForecast.cost_micros`` directly rather than
#: through this fail-closed (never-zero) list-price table.
#:
#: * The network-free fixture provider, whose true cost is ``0``.
#: * The hosted research forecaster (issue #555), which reports its own
#:   ``cost_usd`` and falls back fail-closed to its configured
#:   ``per_call_ceiling_micros`` -- never to zero. It is not wrapped in the
#:   retry layer (``windbreak.scheduler.provider_wiring.build_provider_factory``
#:   explains why), so a per-attempt list price for it would be a number nothing
#:   charges. It was priced here at ``500_000`` until #555; that entry meant the
#:   engine could price a call the composition root refused to make, and pricing
#:   it now would mean advertising a per-attempt charge that is never applied.
#:
#: NOTE: sourcing these list prices from ``windbreak.config`` is a follow-up; it
#: is deliberately NOT wired to ``config.schema`` here (SPEC S8.3 keeps the
#: forecast engine free of any config import). The configuration's own
#: ``_default_provider_prices()`` is pinned *mirror-equal* to this table by
#: test, so the two can never disagree about a provider or a price.
DEFAULT_PROVIDER_PRICE_TABLE: Final[ProviderPriceTable] = ProviderPriceTable(
    prices_micros={
        "openai": _OPENAI_PROVIDER_PRICE_MICROS,
        "anthropic": _ANTHROPIC_PROVIDER_PRICE_MICROS,
    },
    unknown_provider_price_micros=DEFAULT_UNKNOWN_PROVIDER_PRICE_MICROS,
)

#: The fail-closed charge for a response whose cost cannot be derived, in
#: micros. Derived rather than invented: it is exactly
#: :data:`DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS`, the whole retry allowance one
#: ensemble member may spend. If a response will not say what it cost, the only
#: honest assumption is that it spent everything the member was allowed to --
#: which is also what makes the failure *recoverable* rather than fatal: one
#: unmetered vote is affordable and books its full allowance, a second attempt
#: on the same member is refused by the affordability pre-gate, and configuring
#: the model's rates restores normal metering. It is deliberately unequal to any
#: per-attempt list price, so a fail-closed charge can never be mistaken for a
#: metered one.
DEFAULT_UNMETERED_RESPONSE_MICROS: Final = DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS

#: The pinned default per-model token rates, covering exactly the three models
#: :data:`windbreak.forecast.providers.DEFAULT_VOTE_ENSEMBLE` pins, at their
#: vendors' published list rates in micros per million tokens. A model absent
#: from this table is not free: it charges
#: :data:`DEFAULT_UNMETERED_RESPONSE_MICROS`.
#:
#: NOTE: sourcing these rates from ``windbreak.config`` is the composition
#: root's job (``windbreak.scheduler.provider_wiring.rate_table_from_config``);
#: it is deliberately NOT wired to ``config.schema`` here, exactly as with
#: :data:`DEFAULT_PROVIDER_PRICE_TABLE` (SPEC S8.3 keeps the forecast engine
#: free of any config import).
DEFAULT_MODEL_RATE_TABLE: Final[ModelRateTable] = ModelRateTable(
    rates={
        "gpt-5-2025-08-07": ModelTokenRate(
            model_version="gpt-5-2025-08-07",
            input_micros_per_million_tokens=1_250_000,
            output_micros_per_million_tokens=10_000_000,
        ),
        "gpt-5-mini-2025-08-07": ModelTokenRate(
            model_version="gpt-5-mini-2025-08-07",
            input_micros_per_million_tokens=250_000,
            output_micros_per_million_tokens=2_000_000,
        ),
        "claude-sonnet-4-5-20250929": ModelTokenRate(
            model_version="claude-sonnet-4-5-20250929",
            input_micros_per_million_tokens=3_000_000,
            output_micros_per_million_tokens=15_000_000,
        ),
    },
    unmetered_micros=DEFAULT_UNMETERED_RESPONSE_MICROS,
)


def _utc_day_key(at: datetime) -> str:
    """Return the ISO ``YYYY-MM-DD`` UTC-day key for an instant.

    The single choke point for the day bucket: :meth:`ResearchBudget.ensure_day_open`
    and :meth:`ResearchBudget.charge_forecast` both route through it (and
    :meth:`ResearchBudget.charge_stage` delegates to the latter), so guarding
    the awareness invariant here covers every spend path at once.

    Args:
        at: The instant to bucket, which **must** be timezone-aware.

    Returns:
        The instant's UTC calendar date as an ISO-8601 date string.

    Raises:
        ValueError: If ``at`` carries no UTC offset. A naive instant read as
            host-local time buckets spend to the wrong calendar day (issue
            #397), so the daily ceiling silently resets or double-counts across
            the UTC-midnight boundary.
    """
    require_aware(at, "at")
    return at.astimezone(UTC).date().isoformat()


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
        opening_spend_by_day: Mapping[str, int] | None = None,
    ) -> None:
        """Initialize the budget with its three ceilings and its day counter.

        ``opening_spend_by_day`` is what makes the per-UTC-day ceiling a per-day
        ceiling rather than a per-*process* one (issue #442). A composition root
        that can read durable state -- the scheduler folds the hash-chained
        ledger through
        :func:`windbreak.scheduler.research_spend.spend_by_day_from_records` --
        passes the day's real spend here, so a restarted process resumes the
        day where the last one left it. Defaulting to ``None`` (an empty
        counter) keeps every existing caller and every unit test unchanged, and
        is correct for a budget with no durable store behind it.

        Args:
            per_forecast_micros: The per-forecast spend ceiling, in micros.
            per_day_micros: The per-UTC-day spend ceiling, in micros.
            max_pages: The per-forecast web-page fetch ceiling.
            ledger: The budget-event ledger writer (keyword-only).
            opening_spend_by_day: The day counter to open with, keyed by ISO
                ``YYYY-MM-DD`` UTC day, or ``None`` to start empty. Copied, not
                aliased, so the caller's mapping is never mutated by a charge.

        Raises:
            ValueError: If any of the three ceilings, or any opening day total,
                is negative.
        """
        _require_non_negative(per_forecast_micros, "per_forecast_micros")
        _require_non_negative(per_day_micros, "per_day_micros")
        _require_non_negative(max_pages, "max_pages")
        self._per_forecast_micros = per_forecast_micros
        self._per_day_micros = per_day_micros
        self._max_pages = max_pages
        self._ledger = ledger
        opening = dict(opening_spend_by_day or {})
        for day, spent in opening.items():
            _require_non_negative(spent, f"opening_spend_by_day[{day!r}]")
        self._spend_by_day: dict[str, int] = opening
        self._already_charged_micros = 0

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
                BudgetEvent(BUDGET_DAY_EXHAUSTED_EVENT, payload, iso_z(at))
            )
            raise DailyBudgetExhaustedError(spent, self._per_day_micros, day)

    def charge_forecast(
        self, cost_micros: int, *, market_ticker: str, at: datetime
    ) -> None:
        """Charge one forecast's research cost, fail-closed on a per-forecast overrun.

        The spend lands in the day bucket *first*, so a breached forecast still
        counts against the day. The per-forecast ceiling is inclusive: a cost
        exactly equal to it passes; only a strictly greater cost breaches. On a
        budget handed back by :meth:`charge_stage`, the ceiling is checked
        against this cost *plus* what earlier stages of the same forecast
        already spent, and that aggregate is what the error and its ledgered
        event report.

        A ``BUDGET_SPEND_RECORDED`` event is ledgered for the charge in the same
        order and for the same reason: *before* the per-forecast check, so money
        already committed is durable even when this call goes on to raise
        (issue #442). Ledgering it only on the way out would lose exactly the
        spend a breaching forecast made, which is the spend most worth
        remembering.

        Args:
            cost_micros: The forecast's research cost, in micros.
            market_ticker: The forecasted market's ticker, for the audit trail
                (keyword-only).
            at: The run's creation instant, bucketing the spend to a UTC day
                (keyword-only).

        Raises:
            ValueError: If ``cost_micros`` is negative.
            PerForecastBudgetExceededError: If the forecast's total cost
                strictly exceeds the per-forecast ceiling. A
                ``BUDGET_FORECAST_EXCEEDED`` event is ledgered before the raise.
        """
        _require_non_negative(cost_micros, "cost_micros")
        day = _utc_day_key(at)
        self._spend_by_day[day] = self._spend_by_day.get(day, 0) + cost_micros
        spend_payload: dict[str, object] = {
            "utc_day": day,
            "market_ticker": market_ticker,
            "cost_micros": cost_micros,
        }
        self._ledger.record(
            BudgetEvent(BUDGET_SPEND_RECORDED_EVENT, spend_payload, iso_z(at))
        )
        forecast_cost_micros = self._already_charged_micros + cost_micros
        if forecast_cost_micros > self._per_forecast_micros:
            payload: dict[str, object] = {
                "cost_micros": forecast_cost_micros,
                "budget_micros": self._per_forecast_micros,
                "market_ticker": market_ticker,
                "utc_day": day,
            }
            self._ledger.record(
                BudgetEvent(BUDGET_FORECAST_EXCEEDED_EVENT, payload, iso_z(at))
            )
            raise PerForecastBudgetExceededError(
                forecast_cost_micros, self._per_forecast_micros
            )

    def charge_stage(
        self, cost_micros: int, *, market_ticker: str, at: datetime
    ) -> ResearchBudget:
        """Charge a finished stage and return the budget for the rest of that forecast.

        A forecast can spend in more than one stage -- SPEC S8.4's cheap Stage-0
        prior, then the full pipeline -- but it is still *one* forecast, so the
        per-forecast ceiling governs their total. Putting each stage through
        :meth:`charge_forecast` independently would clear both checks whenever
        neither stage alone breaches, leaving a ceiling that cannot trip on the
        only figure that matters. This charges the stage that has already run
        (so spent money is on the books immediately, even if a later stage
        raises) and hands back a view onto the *same* forecast: the same day
        bucket and ledger, remembering what has been spent so the next stage's
        charge is checked in aggregate.

        The returned view is deliberately a separate object rather than mutated
        state on ``self``: a :class:`ResearchBudget` outlives any one forecast,
        and the already-spent offset must not follow it into the next one.

        Args:
            cost_micros: The finished stage's research cost, in micros.
            market_ticker: The forecasted market's ticker, for the audit trail
                (keyword-only).
            at: The run's creation instant, bucketing the spend to a UTC day
                (keyword-only).

        Returns:
            A budget governing the remainder of the same forecast, sharing this
            budget's day bucket, ledger, and ceilings.

        Raises:
            ValueError: If ``cost_micros`` is negative.
            PerForecastBudgetExceededError: If the forecast's cost through this
                stage strictly exceeds the per-forecast ceiling -- so no
                remainder is handed back and no later stage can run on it.
        """
        self.charge_forecast(cost_micros, market_ticker=market_ticker, at=at)
        remainder = ResearchBudget(
            per_forecast_micros=self._per_forecast_micros,
            per_day_micros=self._per_day_micros,
            max_pages=self._max_pages,
            ledger=self._ledger,
        )
        remainder._spend_by_day = self._spend_by_day
        remainder._already_charged_micros = self._already_charged_micros + cost_micros
        return remainder


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
    ledger.record(BudgetEvent(COST_REPORT_EVENT, payload, iso_z(at)))
    return report
