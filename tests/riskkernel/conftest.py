"""Shared `OrderIntent`/`EvaluationContext` builders for `tests/riskkernel/*`
(issues #30/#31/#32/#34, RED).

`windbreak/riskkernel/context.py` does not exist yet, so importing
`EvaluationContext` and its four constituent dataclasses below fails
collection with `ModuleNotFoundError: No module named
'windbreak.riskkernel.context'` -- the expected Gate 1 RED state for issue #30.

Issue #32 (read-only exchange verification) adds a fifth required field,
`EvaluationContext.verification: VerificationSnapshot | None` (no default,
mirroring the `used_intent_ids` fail-loud precedent from issue #31), plus a
new `RiskLimits.verification_ttl_seconds: int`. `windbreak/riskkernel/verification.py`
does not exist yet either, so the `VerificationOutcome`/`VerificationSnapshot`
import below independently fails collection with `ModuleNotFoundError` --
also the expected Gate 1 RED state for issue #32.

Issue #34 (floor governance / human acknowledgement) adds a sixth required
`EvaluationContext` field, `acknowledged_intent_ids: frozenset[str]` (mirroring
`used_intent_ids`'s fail-loud-but-test-permissive treatment, defaulting to an
empty `frozenset()` here), plus a new
`RiskLimits.require_human_ack_above_micros: MoneyMicros | None` (defaulting to
`None`, i.e. "no threshold configured", so `human_ack_satisfied` approves by
default). Neither field exists on the real, not-yet-updated `context.py`
dataclasses yet, so constructing `_DEFAULT_LIMITS` below fails collection with
`TypeError: __init__() got an unexpected keyword argument
'require_human_ack_above_micros'` -- the expected Gate 1 RED state for issue
#34, independent of and in addition to the `ModuleNotFoundError`s above.

Issue #110 (promoting `exchange_status_ok` / `pipeline_heartbeat_ok` from
stub to real logic) adds a new `ExchangeTradingStatus` enum (`OPEN` / `PAUSED`
/ `CLOSED`; "unknown/missing" is modeled as `None`, never an enum member),
two new `MarketView` fields (`exchange_status: ExchangeTradingStatus | None`,
`exchange_status_epoch_s: int | None`), two new `RiskLimits` ttl fields
(`exchange_status_ttl_seconds`, `pipeline_heartbeat_ttl_seconds`), and a
seventh required `EvaluationContext` field,
`pipeline_heartbeat_epoch_s: int | None` (no default, mirroring the
`verification` fail-loud precedent above). None of these exist on the real,
not-yet-updated `context.py` dataclasses yet, so both the `ExchangeTradingStatus`
import and constructing `_DEFAULT_LIMITS` / `_DEFAULT_MARKET` below fail
collection with `ImportError` / `TypeError: __init__() got an unexpected
keyword argument 'exchange_status_ttl_seconds'` -- the expected Gate 1 RED
state for issue #110, independent of and in addition to the errors above.

Builder-placement choice: unlike the rest of this test suite (where each file
duplicates its own small `make_intent`, e.g.
`tests/riskkernel/test_process_isolation.py`'s `_make_intent`),
`make_context` assembles five nested dataclasses spanning ~40 total fields.
Duplicating that across `test_checks.py`, `test_floor_metamorphic.py`, and
`test_process_isolation.py` would violate the "never duplicate content"
principle for no offsetting clarity gain, so it is centralized here instead
and imported explicitly (`from tests.riskkernel.conftest import
make_context, ...`) by every file that needs it -- an ordinary intra-package
import, not pytest fixture injection, so it works unchanged inside
`@given`-decorated Hypothesis tests too.

`make_context`'s defaults are deliberately tuned so that, paired with
`make_intent()`'s defaults (action "buy", price 5000 pips, size 1000
centis), **every one of the 24 real SPEC S10.3 checks passes**: each
per-check test in `test_checks.py` therefore only needs to override the one
or two fields that check actually reads, proving the override -- not a
coincidence of some other field -- is what flips the verdict.

Implementation note: overrides are applied via `dataclasses.replace` against
one pre-built default instance per nested dataclass, rather than merging
plain `dict[str, object]`s and splatting them into each constructor. The
constructors themselves expect concrete per-field types (`MoneyMicros`,
`frozenset[str]`, `int`, ...), which a homogeneous `dict[str, object]` splat
cannot satisfy under `mypy --strict`; `dataclasses.replace`'s stub accepts
`**changes: Any`, so it is the one spot in this module allowed to be loosely
typed. The per-dataclass field-name sets used to route each override are
themselves derived from `dataclasses.fields(...)` -- reflected off the real
production classes, not retyped by hand -- so they can never drift from
whatever fields `context.py` actually declares.
"""

from __future__ import annotations

import dataclasses

from windbreak.numeric.types import (
    ContractCentis,
    MoneyMicros,
    PricePips,
    ProbabilityPpm,
)
from windbreak.riskkernel.checks import OrderIntent
from windbreak.riskkernel.context import (
    AccountState,
    EvaluationContext,
    ExchangeTradingStatus,
    FeeBounds,
    JurisdictionStatus,
    MarketView,
    ProductType,
    RiskLimits,
)
from windbreak.riskkernel.modes import Mode
from windbreak.riskkernel.verification import VerificationOutcome, VerificationSnapshot

#: The exchange ticker every default `OrderIntent`/`RiskLimits.instrument_whitelist`
#: agree on, so the default context passes `instrument_whitelist` out of the box.
DEFAULT_MARKET_TICKER = "PRES-2028-DEM"

#: Immutable scaled-int defaults for :func:`make_intent`, held as module-level
#: singletons so they are not reconstructed in the function's argument defaults
#: (ruff B008); the wrapper types are frozen, so sharing one instance is safe.
_DEFAULT_PRICE = PricePips(5000)
_DEFAULT_SIZE = ContractCentis(1000)
_DEFAULT_MAX_NOTIONAL = MoneyMicros(50_000_000)
_DEFAULT_IMPLIED_PROBABILITY = ProbabilityPpm(520_000)


#: Default `OrderIntent.idempotency_key` for :func:`make_intent` (issue #31):
#: distinct from `intent_id` on purpose, so a per-check test overriding only
#: one of the two uniqueness dimensions never accidentally exercises both.
_DEFAULT_IDEMPOTENCY_KEY = "idem-0001"


def make_intent(
    *,
    intent_id: str = "intent-0001",
    market_ticker: str = DEFAULT_MARKET_TICKER,
    outcome: str = "yes",
    action: str = "buy",
    price: PricePips = _DEFAULT_PRICE,
    size: ContractCentis = _DEFAULT_SIZE,
    max_notional: MoneyMicros = _DEFAULT_MAX_NOTIONAL,
    implied_probability: ProbabilityPpm = _DEFAULT_IMPLIED_PROBABILITY,
    idempotency_key: str = _DEFAULT_IDEMPOTENCY_KEY,
) -> OrderIntent:
    """Build a valid `OrderIntent`, with any field overridable by keyword.

    Args:
        intent_id: The intent's unique identifier.
        market_ticker: The exchange ticker the intent targets.
        outcome: The market outcome the intent trades (e.g. "yes"/"no").
        action: The trade action (e.g. "buy"/"sell_to_close").
        price: The limit price, in pips.
        size: The contract count, in centis.
        max_notional: The notional cap, in money-micros.
        implied_probability: The forecast-implied probability, in ppm.
        idempotency_key: The caller-supplied idempotency key (issue #31).

    Returns:
        A fully populated, valid `OrderIntent`.
    """
    return OrderIntent(
        intent_id=intent_id,
        market_ticker=market_ticker,
        outcome=outcome,
        action=action,
        price=price,
        size=size,
        max_notional=max_notional,
        implied_probability=implied_probability,
        idempotency_key=idempotency_key,
    )


#: The fixed "current instant" every default `MarketView` timestamp and
#: `EvaluationContext.now_epoch_s` agree on, so freshness/skew checks pass by
#: default (zero age, zero skew).
DEFAULT_NOW_EPOCH_S = 1_700_000_000

#: `AccountState.exchange_verified_available_cash` /
#: `.equity_start_of_day` / `.equity_high_water_mark` default: $1,000, in
#: micros. Large relative to the default intent's ~$0.50 notional
#: (5000 pips * 1000 centis == 5_000_000 micros) so every equity-relative
#: check (floor, concentration, drawdown, loss) passes by default.
_DEFAULT_EQUITY_MICROS = 1_000_000_000

#: `RiskLimits.verification_ttl_seconds` default (issue #32): one hour, matching
#: the other freshness ttls below, so `balance_reconciliation` /
#: `position_reconciliation` / `open_order_reconciliation` pass by default.
_DEFAULT_VERIFICATION_TTL_SECONDS = 3_600

#: The permissive default `RiskLimits`: 100% caps, a wide price band, and
#: generous ttls, so only the field a given test overrides can flip a real
#: check's verdict. `require_human_ack_above_micros=None` (issue #34) means
#: "no threshold configured" -- the permissive default, so `human_ack_satisfied`
#: approves by default (a `None` threshold never requires an ack) unless a test
#: deliberately configures one. `exchange_status_ttl_seconds` /
#: `pipeline_heartbeat_ttl_seconds` (issue #110) match the other freshness
#: ttls, so `exchange_status_ok` / `pipeline_heartbeat_ok` pass by default
#: given `_DEFAULT_MARKET`'s fresh `exchange_status_epoch_s` and
#: `make_context`'s fresh `pipeline_heartbeat_epoch_s` default.
_DEFAULT_LIMITS = RiskLimits(
    floor=MoneyMicros(0),
    instrument_whitelist=frozenset({DEFAULT_MARKET_TICKER}),
    micro_cap=MoneyMicros(1_000_000_000_000),
    min_open_price=PricePips(0),
    max_open_price=PricePips(10_000),
    max_participation_ppm=1_000_000,
    max_pos_market_pct_ppm=1_000_000,
    max_pos_event_pct_ppm=1_000_000,
    max_pos_bucket_pct_ppm=1_000_000,
    max_pos_total_pct_ppm=1_000_000,
    daily_loss_limit_pct_ppm=1_000_000,
    max_drawdown_pct_ppm=1_000_000,
    max_orders_per_hour=1_000,
    max_notional_per_day=MoneyMicros(1_000_000_000_000),
    quote_ttl_seconds=3_600,
    forecast_ttl_seconds=3_600,
    clock_skew_max_seconds=3_600,
    rounding_buffer=MoneyMicros(0),
    verification_ttl_seconds=_DEFAULT_VERIFICATION_TTL_SECONDS,
    require_human_ack_above_micros=None,
    exchange_status_ttl_seconds=3_600,
    pipeline_heartbeat_ttl_seconds=3_600,
)

#: The permissive default `AccountState`: flat $1,000 equity, zero
#: exposure/reservations/fees/loss/orders, so every equity- and
#: velocity-relative check passes by default.
_DEFAULT_ACCOUNT = AccountState(
    exchange_verified_available_cash=MoneyMicros(_DEFAULT_EQUITY_MICROS),
    guaranteed_terminal_value_of_positions=MoneyMicros(0),
    pending_kernel_reservations=MoneyMicros(0),
    unresolved_fee_upper_bounds=MoneyMicros(0),
    reconciliation_uncertainty_buffer=MoneyMicros(0),
    equity_start_of_day=MoneyMicros(_DEFAULT_EQUITY_MICROS),
    equity_high_water_mark=MoneyMicros(_DEFAULT_EQUITY_MICROS),
    realized_loss_today=MoneyMicros(0),
    market_exposure=MoneyMicros(0),
    event_exposure=MoneyMicros(0),
    bucket_exposure=MoneyMicros(0),
    total_exposure=MoneyMicros(0),
    orders_last_hour=0,
    notional_today=MoneyMicros(0),
)

#: The permissive default `MarketView`: quote/forecast/clock all stamped at
#: `DEFAULT_NOW_EPOCH_S` (zero age, zero skew) and ample visible depth, so
#: freshness/skew/participation checks pass by default. `exchange_status`
#: defaults to `ExchangeTradingStatus.OPEN` with a fresh
#: `exchange_status_epoch_s` (issue #110), so `exchange_status_ok` passes by
#: default too. `jurisdiction_status`/`product_type` default to the one
#: eligible/tradable pair (issue #340), so `jurisdiction_product_eligibility`
#: passes by default as well.
_DEFAULT_MARKET = MarketView(
    quote_snapshot_epoch_s=DEFAULT_NOW_EPOCH_S,
    forecast_epoch_s=DEFAULT_NOW_EPOCH_S,
    visible_depth=ContractCentis(10_000_000),
    exchange_clock_epoch_s=DEFAULT_NOW_EPOCH_S,
    open_position=None,
    exchange_status=ExchangeTradingStatus.OPEN,
    exchange_status_epoch_s=DEFAULT_NOW_EPOCH_S,
    jurisdiction_status=JurisdictionStatus.ELIGIBLE,
    product_type=ProductType.FULLY_COLLATERALIZED_BINARY,
)

#: The permissive default `FeeBounds`: both bounds present (zero), so
#: `fee_upper_bound_present`/`settlement_fee_upper_bound` pass by default.
_DEFAULT_FEES = FeeBounds(
    max_trading_fee=MoneyMicros(0),
    max_settlement_fee=MoneyMicros(0),
)

#: The permissive default `VerificationSnapshot` (issue #32): a fresh CLEAN
#: verification exactly at `DEFAULT_NOW_EPOCH_S` (zero age), every per-dimension
#: ok flag `True`, zero cash drift, and fully-known balance semantics -- so
#: `balance_reconciliation` / `position_reconciliation` / `open_order_reconciliation`
#: all pass by default (including the LIVE-mode semantics gate, since
#: `semantics_fully_known` is `True`), matching `make_context()`'s default
#: `mode=Mode.LIVE`.
_DEFAULT_VERIFICATION_SNAPSHOT = VerificationSnapshot(
    outcome=VerificationOutcome.CLEAN,
    balance_ok=True,
    position_ok=True,
    open_order_ok=True,
    verified_at_epoch_s=DEFAULT_NOW_EPOCH_S,
    exchange_verified_available_cash=MoneyMicros(_DEFAULT_EQUITY_MICROS),
    cash_drift=MoneyMicros(0),
    semantics_fully_known=True,
)


def make_verification_snapshot(**overrides: object) -> VerificationSnapshot:
    """Build a fully-permissive `VerificationSnapshot`, any field overridable.

    Mirrors `make_intent`'s single-dataclass-replace shape (issue #32): every
    field keeps `_DEFAULT_VERIFICATION_SNAPSHOT`'s permissive default (see
    above) unless explicitly overridden here, so a per-check test can flip
    exactly one dimension (e.g. `balance_ok=False`) and know only that input
    caused the verdict it observes.

    Args:
        **overrides: Field name to value, for any `VerificationSnapshot` field.

    Returns:
        A fully populated `VerificationSnapshot`.
    """
    return dataclasses.replace(_DEFAULT_VERIFICATION_SNAPSHOT, **overrides)


#: Each nested dataclass's field names, reflected off the real classes (never
#: retyped by hand) so override-routing can never drift from `context.py`.
_LIMITS_FIELDS = frozenset(f.name for f in dataclasses.fields(RiskLimits))
_ACCOUNT_FIELDS = frozenset(f.name for f in dataclasses.fields(AccountState))
_MARKET_FIELDS = frozenset(f.name for f in dataclasses.fields(MarketView))
_FEES_FIELDS = frozenset(f.name for f in dataclasses.fields(FeeBounds))
#: `EvaluationContext`'s own direct fields (not nested in one of the four
#: value objects): `mode`/`now_epoch_s` from issue #30, the two
#: ledger-uniqueness sets issue #31 adds (`used_intent_ids`,
#: `used_idempotency_keys`), each defaulting to an empty `frozenset()` so
#: every pre-issue-#31 test keeps passing `approval_token_uniqueness` /
#: `idempotency_key_uniqueness` unless it deliberately overrides one, the
#: issue #32 `verification` snapshot (defaulting to
#: `_DEFAULT_VERIFICATION_SNAPSHOT`, a permissive CLEAN snapshot, so every
#: pre-issue-#32 test keeps passing `balance_reconciliation` /
#: `position_reconciliation` / `open_order_reconciliation` unless it
#: deliberately overrides `verification`), the issue #34
#: `acknowledged_intent_ids` set (defaulting to an empty `frozenset()`, mirroring
#: `used_intent_ids`'s fail-loud-but-test-permissive treatment) for the
#: `human_ack_satisfied` check, and the issue #110
#: `pipeline_heartbeat_epoch_s` field (defaulting to `DEFAULT_NOW_EPOCH_S`, a
#: fresh reading, so `pipeline_heartbeat_ok` passes by default unless a test
#: deliberately overrides it) for that same check.
_CONTEXT_FIELDS = frozenset(
    {
        "mode",
        "now_epoch_s",
        "used_intent_ids",
        "used_idempotency_keys",
        "verification",
        "acknowledged_intent_ids",
        "pipeline_heartbeat_epoch_s",
    }
)


def make_context(**overrides: object) -> EvaluationContext:
    """Build a fully-permissive `EvaluationContext`, any field overridable.

    Every keyword is routed to whichever of the five nested dataclasses
    (`RiskLimits`, `AccountState`, `MarketView`, `FeeBounds`) declares a field
    of that name, or to `EvaluationContext` itself for `mode`/`now_epoch_s`;
    the five field-name sets are disjoint, so routing by name alone is
    unambiguous. Every field not overridden keeps its permissive default (see
    the module docstring), so a test can flip exactly one input and know only
    that input caused the verdict it observes.

    Args:
        **overrides: Field name to value, for any field of `RiskLimits`,
            `AccountState`, `MarketView`, `FeeBounds`, or `EvaluationContext`
            itself (`mode`, `now_epoch_s`, `used_intent_ids`,
            `used_idempotency_keys`, `verification`, `acknowledged_intent_ids`,
            `pipeline_heartbeat_epoch_s`).

    Returns:
        A fully populated `EvaluationContext`.

    Raises:
        ValueError: If a keyword does not name a field of any of the five
            dataclasses -- catches a typo'd override silently doing nothing.
    """
    known_fields = (
        _LIMITS_FIELDS
        | _ACCOUNT_FIELDS
        | _MARKET_FIELDS
        | _FEES_FIELDS
        | _CONTEXT_FIELDS
    )
    unknown = set(overrides) - known_fields
    if unknown:
        raise ValueError(f"make_context() got unknown override(s): {sorted(unknown)}")

    limits = dataclasses.replace(
        _DEFAULT_LIMITS,
        **{k: v for k, v in overrides.items() if k in _LIMITS_FIELDS},
    )
    account = dataclasses.replace(
        _DEFAULT_ACCOUNT,
        **{k: v for k, v in overrides.items() if k in _ACCOUNT_FIELDS},
    )
    market = dataclasses.replace(
        _DEFAULT_MARKET,
        **{k: v for k, v in overrides.items() if k in _MARKET_FIELDS},
    )
    fees = dataclasses.replace(
        _DEFAULT_FEES,
        **{k: v for k, v in overrides.items() if k in _FEES_FIELDS},
    )
    mode = overrides.get("mode", Mode.LIVE)
    now_epoch_s = overrides.get("now_epoch_s", DEFAULT_NOW_EPOCH_S)
    used_intent_ids = overrides.get("used_intent_ids", frozenset())
    used_idempotency_keys = overrides.get("used_idempotency_keys", frozenset())
    verification = overrides.get("verification", _DEFAULT_VERIFICATION_SNAPSHOT)
    acknowledged_intent_ids = overrides.get("acknowledged_intent_ids", frozenset())
    pipeline_heartbeat_epoch_s = overrides.get(
        "pipeline_heartbeat_epoch_s", DEFAULT_NOW_EPOCH_S
    )
    assert isinstance(mode, Mode)
    assert isinstance(now_epoch_s, int)
    assert isinstance(used_intent_ids, frozenset)
    assert isinstance(used_idempotency_keys, frozenset)
    assert verification is None or isinstance(verification, VerificationSnapshot)
    assert isinstance(acknowledged_intent_ids, frozenset)
    assert pipeline_heartbeat_epoch_s is None or isinstance(
        pipeline_heartbeat_epoch_s, int
    )
    return EvaluationContext(
        mode=mode,
        limits=limits,
        account=account,
        market=market,
        fees=fees,
        now_epoch_s=now_epoch_s,
        used_intent_ids=used_intent_ids,
        used_idempotency_keys=used_idempotency_keys,
        verification=verification,
        acknowledged_intent_ids=acknowledged_intent_ids,
        pipeline_heartbeat_epoch_s=pipeline_heartbeat_epoch_s,
    )
