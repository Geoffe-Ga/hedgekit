"""The selector: pure forecast-to-intent decision stage (SPEC S9.1-S9.6).

The selector turns a forecast plus market and account context into a
ledgerable :class:`SelectorDecision`. Per SPEC S9.1 it is *pure,
credentialless, no-I/O, and no-clock*: it never opens a socket, reads a
secret, or calls the wall clock -- freshness is judged by comparing timestamps
carried *inside* :class:`SelectorInputs`, never against ``datetime.now`` -- so
the same inputs always yield the same, byte-identically serializable decision.

Issue #44 lands the real fee-aware edge and entry logic (SPEC S9.2-S9.3):
:func:`select` prices a fixed-size probe fill via
:func:`~windbreak.selector.edge.compute_executable_edge`, renders every SPEC
S9.3 entry condition into ``reasons`` (never silently empty), and only proceeds
when every condition passes. Issue #45 lands sizing (SPEC S9.5/S9.6): when the
twelve conditions pass, :func:`select` computes the dispersion-scaled
fractional-Kelly stake (:func:`~windbreak.selector.sizing.kelly_size`), clips it
through the notional/participation caps
(:func:`~windbreak.selector.sizing.clip_to_caps`), re-prices the fill at that
final size, and emits exactly one normalized ``yes``/``buy`` intent sized to it
-- or declines (no intent) with a pinned sizing reason when the size clips to
zero.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import TYPE_CHECKING

from windbreak.ledger.events import canonical_json
from windbreak.numeric import ContractCentis, MoneyMicros, ProbabilityPpm
from windbreak.selector.correlation import (
    BucketExposureEntry,
    CorrelationTag,
    TagSource,
    aggregate_bucket_exposure,
    effective_buckets,
)
from windbreak.selector.edge import (
    EdgeFigures,
    InsufficientDepth,
    NonAnnualizable,
    _fee_micros,
    compute_executable_edge,
)
from windbreak.selector.entry import evaluate_entry_conditions
from windbreak.selector.execution_style import (
    ExecutionStyleDecision,
    decide_execution_style,
)
from windbreak.selector.exits import CloseTrigger, build_close_intent
from windbreak.selector.serialization import serialize_decision
from windbreak.selector.sizing import (
    clip_to_caps,
    dispersion_scale,
    kelly_size,
)
from windbreak.selector.types import (
    FeeModelInput,
    NormalizedOrderIntent,
    PositionReadModelInput,
    RiskConfigInput,
    SelectorDecision,
    SelectorInputs,
    SelectorOrderIntent,
    SlippageModelInput,
)

if TYPE_CHECKING:
    from windbreak.connector.models import OrderBookSnapshot
    from windbreak.numeric import PricePips
    from windbreak.selector.entry import EntryCheck
    from windbreak.selector.sizing import CapClipResult

#: The probe size that gates entry, before sizing runs (SPEC S9.2-S9.3): a fixed
#: 1.00-contract fill priced only to feed the twelve entry conditions their
#: executable-edge figures. It is *no longer the emitted size* -- once every
#: condition passes, the real dispersion-scaled Kelly size (issue #45)
#: determines the fill, which is then re-priced at that size.
_PROBE_SIZE_CENTIS = ContractCentis(100)

#: The single outcome/action the selector emits today: a YES-side opening buy.
#: Sells and NO-side intents belong to later execution/sizing work.
_OUTCOME_YES = "yes"
_ACTION_BUY = "buy"

#: The intent-id suffix marking a real, dispersion-scaled-Kelly-sized intent
#: (issue #45), distinguishing it from the pre-sizing ``:probe`` shape.
_SIZED_SUFFIX = "sized"

#: Ppm-of-$1 per pip: a pip is 1e-4 $ and a ppm is 1e-6 $, so one pip is 100 ppm.
#: Converts a level's pip price into the ppm-of-$1 the notional caps clip at.
_PPM_PER_PIP = 100

__all__ = [
    "BucketExposureEntry",
    "CloseTrigger",
    "CorrelationTag",
    "ExecutionStyleDecision",
    "FeeModelInput",
    "NormalizedOrderIntent",
    "PositionReadModelInput",
    "RiskConfigInput",
    "SelectorDecision",
    "SelectorInputs",
    "SelectorOrderIntent",
    "SlippageModelInput",
    "TagSource",
    "aggregate_bucket_exposure",
    "build_close_intent",
    "clip_to_caps",
    "decide_execution_style",
    "dispersion_scale",
    "effective_buckets",
    "kelly_size",
    "select",
    "serialize_decision",
]


def _render(check: EntryCheck) -> str:
    """Render one entry check into a ledger reason string.

    Args:
        check: The evaluated entry condition.

    Returns:
        ``"pass:<name>"`` when the condition passed, else
        ``"fail:<name>: <detail>"``.
    """
    if check.passed:
        return f"pass:{check.name}"
    return f"fail:{check.name}: {check.detail}"


def _idempotency_key(
    forecast_id: str, market_ticker: str, price_pips: int, size_centis: int
) -> str:
    """Derive the intent's deterministic idempotency key (SPEC S9.1).

    Hashes exactly the six identifying fields through the same
    ``sha256(canonical_json(...))`` primitive
    :func:`windbreak.order_gateway.client_order_id.client_order_id` uses, so the
    key is a byte-stable function of the intent's economic identity alone.

    Args:
        forecast_id: The originating forecast's id.
        market_ticker: The market the intent targets.
        price_pips: The intent's price, in pips.
        size_centis: The intent's size, in contract-centis.

    Returns:
        The 64-character, lowercase-hex SHA-256 idempotency key.
    """
    fields: dict[str, object] = {
        "forecast_id": forecast_id,
        "market_ticker": market_ticker,
        "outcome": _OUTCOME_YES,
        "action": _ACTION_BUY,
        "price": price_pips,
        "size": size_centis,
    }
    return hashlib.sha256(canonical_json(fields).encode("utf-8")).hexdigest()


def _price_and_notional(
    inputs: SelectorInputs,
    figures: EdgeFigures,
    size: ContractCentis,
    style_decision: ExecutionStyleDecision,
) -> tuple[PricePips, MoneyMicros]:
    """Return the emitted price and notional cap for the chosen style (S9.5/S9.7).

    A ``rest_inside_spread`` decision (recognized by its non-``None``
    ``resting_price_pips``) prices at that passive rest price and caps the
    notional at the rest cost plus its worst-case fee. A ``cross`` decision keeps
    the pre-issue-#46 behavior byte-for-byte: it prices at the marginal
    (deepest-walked) level and caps at the executable fill cost plus fee.

    Args:
        inputs: The selector inputs (for the fee model).
        figures: The executable-edge figures for the fill priced at ``size``.
        size: The final, cap-clipped size to emit, in contract-centis.
        style_decision: The execution-style decision selecting price/notional.

    Returns:
        The ``(price, max_notional)`` pair to stamp on the emitted intent.
    """
    rest_price = style_decision.resting_price_pips
    if rest_price is not None:
        rest_fee_micros = _fee_micros(inputs.fee_model, rest_price.value, size.value)
        rest_notional = MoneyMicros(rest_price.value * size.value + rest_fee_micros)
        return rest_price, rest_notional
    fee_micros = _fee_micros(
        inputs.fee_model, figures.executable_price_pips.value, size.value
    )
    cross_notional = MoneyMicros(figures.executable_cost_micros.value + fee_micros)
    return figures.marginal_price_pips, cross_notional


def _build_intent(
    inputs: SelectorInputs,
    figures: EdgeFigures,
    size: ContractCentis,
    style_decision: ExecutionStyleDecision,
) -> SelectorOrderIntent:
    """Build the single normalized intent for an all-pass, sized evaluation (S9.5).

    Prices and caps the notional per the chosen execution style
    (:func:`_price_and_notional`), then stamps the style and its resting
    parameters (issue #46) onto the intent. The ``:sized`` intent-id suffix and
    the size-hashed idempotency key reflect the real dispersion-scaled Kelly size
    (issue #45), not the pre-sizing probe.

    Args:
        inputs: The selector inputs the intent is derived from.
        figures: The executable-edge figures for the fill priced at ``size``.
        size: The final, cap-clipped size to emit, in contract-centis.
        style_decision: The execution-style decision (cross vs. rest) to stamp.

    Returns:
        The normalized :class:`~windbreak.selector.types.SelectorOrderIntent`.
    """
    forecast = inputs.forecast
    price, max_notional = _price_and_notional(inputs, figures, size, style_decision)
    intent_id = f"{forecast.forecast_id}:{_OUTCOME_YES}:{_ACTION_BUY}:{_SIZED_SUFFIX}"
    return SelectorOrderIntent(
        intent_id=intent_id,
        market_ticker=forecast.market_ticker,
        outcome=_OUTCOME_YES,
        action=_ACTION_BUY,
        price=price,
        size=size,
        max_notional=max_notional,
        implied_probability=ProbabilityPpm(forecast.probability_ppm),
        idempotency_key=_idempotency_key(
            forecast.forecast_id,
            forecast.market_ticker,
            price.value,
            size.value,
        ),
        execution_style=style_decision.style,
        resting_ttl_seconds=style_decision.resting_ttl_seconds,
        cancel_on_move_ticks=style_decision.cancel_on_move_ticks,
    )


def _decision(
    inputs: SelectorInputs,
    intents: tuple[NormalizedOrderIntent, ...],
    reasons: tuple[str, ...],
) -> SelectorDecision:
    """Assemble a :class:`SelectorDecision`, echoing the inputs' identity.

    Args:
        inputs: The evaluated inputs, supplying the forecast id/ticker and the
            calibration-map version echoed for ledger traceability.
        intents: The emitted normalized intents (possibly empty).
        reasons: The non-empty reasons explaining the verdict.

    Returns:
        The assembled decision.
    """
    return SelectorDecision(
        intents=intents,
        reasons=reasons,
        forecast_id=inputs.forecast.forecast_id,
        market_ticker=inputs.forecast.market_ticker,
        calibration_map_version=inputs.calibration_map_version,
    )


def _sizing_reason(raw: ContractCentis, g_ppm: int, clip: CapClipResult) -> str:
    """Render the pinned sizing reason naming the raw size, scale, and outcome.

    Args:
        raw: The raw fractional-Kelly size before capping, in contract-centis.
        g_ppm: The dispersion scale applied, in ppm.
        clip: The cap-clip result carrying the final size and binding cap.

    Returns:
        ``"sizing: raw_centis=<R> g_ppm=<G> binding_cap=<C> final_centis=<F>"``,
        with ``<C>`` the binding cap name or ``"none"``.
    """
    return (
        f"sizing: raw_centis={raw.value} g_ppm={g_ppm} "
        f"binding_cap={clip.binding_cap or 'none'} final_centis={clip.size.value}"
    )


def _cap_reference_price_ppm(order_book: OrderBookSnapshot) -> int:
    """Return the worst-case per-contract price the notional caps clip at (S9.6).

    The deepest resting ask's price (``yes_asks`` is best-first, so the last
    level), in ppm-of-$1. Any participation-limited fill's executable VWAP is at
    most this price, so a notional cap that holds at it also holds at the fill's
    own -- never higher -- executable price: the cap is measured against the
    dearest contract the fill could pay for, never the cheaper probe price (which
    would understate the fill's cost and over-size the position). Called only
    after the probe fill priced, so ``yes_asks`` is non-empty.

    Args:
        order_book: The market's order-book snapshot.

    Returns:
        The deepest ask's price, in ppm-of-$1.
    """
    return order_book.yes_asks[-1].price.value * _PPM_PER_PIP


def _size_and_emit(
    inputs: SelectorInputs, figures: EdgeFigures, reasons: tuple[str, ...]
) -> SelectorDecision:
    """Size an all-pass evaluation and emit the sized intent, or decline (S9.5/S9.6).

    Computes the dispersion scale and the fractional-Kelly stake against the
    probe-priced edge, clips it through the notional/participation caps -- the
    notional caps measured at the fill's worst-case per-contract price
    (:func:`_cap_reference_price_ppm`) so they hold at whatever price the
    participation-limited fill actually pays. The per-bucket cap's exposure is
    the real correlation-bucket exposure aggregated from the target's effective
    buckets and ``inputs.bucket_peers`` (SPEC S9.9), overriding the placeholder
    ``positions.bucket_exposure`` and naming the binding bucket in the pinned
    reason. When the size survives,
    re-prices the fill at that final size before emitting one intent. A size
    clipped to zero, an unfillable re-walk (defensive), or a net edge that no
    longer clears the floor at the final size each decline with a pinned reason
    appended after the twelve entry reasons.

    Args:
        inputs: The evaluated selector inputs.
        figures: The probe-priced executable-edge figures the entry checks used.
        reasons: The twelve rendered entry-condition reasons.

    Returns:
        The decision: one sized intent plus the sizing reason, or no intent plus
        a pinned decline reason.
    """
    risk = inputs.risk_config.config
    g_ppm = dispersion_scale(
        inputs.forecast.vote_dispersion_ppm, risk.dispersion_zero_ceiling_ppm
    )
    raw = kelly_size(
        net_edge_ppm=figures.net_edge_ppm,
        min_net_edge_ppm=risk.min_net_edge_ppm,
        executable_price_ppm=figures.executable_price_ppm,
        kelly_fraction_ppm=risk.kelly_fraction_ppm,
        dispersion_scale_ppm=g_ppm,
        above_floor_capital_micros=inputs.positions.above_floor_capital_micros,
    )
    target_buckets = effective_buckets(inputs.correlation_tags)
    bucket_used, binding_bucket = aggregate_bucket_exposure(
        target_buckets, inputs.bucket_peers
    )
    bucket_cap_name = (
        f"per_bucket:{binding_bucket}" if binding_bucket is not None else "per_bucket"
    )
    clip = clip_to_caps(
        raw,
        executable_price_ppm=_cap_reference_price_ppm(inputs.order_book),
        order_book=inputs.order_book,
        risk_config=risk,
        positions=dataclasses.replace(
            inputs.positions, bucket_exposure=MoneyMicros(bucket_used)
        ),
        bucket_cap_name=bucket_cap_name,
    )
    sizing_reason = _sizing_reason(raw, g_ppm, clip)
    if clip.size.value == 0:
        return _decision(inputs, (), (*reasons, sizing_reason))

    final = compute_executable_edge(
        order_book=inputs.order_book,
        size=clip.size,
        forecast=inputs.forecast,
        fee_model=inputs.fee_model,
        slippage_model=inputs.slippage_model,
    )
    if not isinstance(final, EdgeFigures):
        return _decision(inputs, (), (*reasons, final.reason))
    net = final.net_edge_ppm
    if net < risk.min_net_edge_ppm:
        detail = f"net_edge_ppm={net} min_net_edge_ppm={risk.min_net_edge_ppm}"
        reason = f"fail:net_edge_at_final_size: {detail}"
        return _decision(inputs, (), (*reasons, reason))
    style_decision = decide_execution_style(inputs, clip.size)
    intent = _build_intent(inputs, final, clip.size, style_decision)
    return _decision(inputs, (intent,), (*reasons, sizing_reason))


def select(inputs: SelectorInputs) -> SelectorDecision:
    """Evaluate the selector inputs into a decision (SPEC S9.1-S9.6).

    Prices a fixed-size probe fill (SPEC S9.2). If the book is too shallow to
    fill it, declines with the depth-shortfall reason and no intents; if the
    fill priced but its return cannot be annualized (a 0-pip price or a
    zero-hour forecast horizon), declines with the non-annualizable reason and
    no intents. Otherwise renders every SPEC S9.3 entry condition into
    ``reasons``; only when all twelve pass does it size the position (SPEC
    S9.5/S9.6) -- the dispersion-scaled fractional-Kelly stake, clipped through
    the notional and participation caps and re-priced at the final size -- and
    emit exactly one normalized ``yes``/``buy`` intent, or decline with a pinned
    sizing reason when the size clips to zero. Reads no clock and does no I/O, so
    the decision is a pure function of ``inputs``.

    Args:
        inputs: The complete, immutable input bundle to evaluate.

    Returns:
        A :class:`SelectorDecision` carrying the emitted intents (one when every
        entry condition passes and the size survives capping, none otherwise)
        and the non-empty reasons for the verdict, alongside the forecast id,
        market ticker, and calibration-map version carried over from ``inputs``.
    """
    figures = compute_executable_edge(
        order_book=inputs.order_book,
        size=_PROBE_SIZE_CENTIS,
        forecast=inputs.forecast,
        fee_model=inputs.fee_model,
        slippage_model=inputs.slippage_model,
    )
    if isinstance(figures, (InsufficientDepth, NonAnnualizable)):
        return _decision(inputs, (), (figures.reason,))

    checks = evaluate_entry_conditions(inputs, figures)
    reasons = tuple(_render(check) for check in checks)
    if not all(check.passed for check in checks):
        return _decision(inputs, (), reasons)
    return _size_and_emit(inputs, figures, reasons)
