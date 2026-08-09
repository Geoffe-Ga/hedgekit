"""Pure, pytest-free JSON-to-domain loader for selector test fixtures (#43/#44).

`load_inputs` parses one recorded "bundle" JSON file (see
`tests/selector/fixtures/bundle_a.json` / `bundle_b.json`) into a real
`windbreak.selector.SelectorInputs` -- built from the actual, post-init
validated `windbreak.forecast.records.ForecastRecord` and
`windbreak.connector.models.OrderBookSnapshot` domain types, with every
arithmetic-bearing value wrapped in its `windbreak.numeric` scaled-integer unit
type at load time (mirroring `windbreak.connector.fake.FakeExchange`'s
`_market_from_dict` / `_book_from_dict` fixture-loading convention). No
float ever appears in a bundle file or in the objects built from it.

Issue #44 enriches three placeholder seams into concrete carriers: a
bundle's `fee_model` object becomes a `FeeModelInput` wrapping a real,
post-init-validated `windbreak.connector.fees.FeeModel` plus its `as_of`
freshness timestamp; `slippage_model` becomes a `SlippageModelInput`;
`risk_config` becomes a `RiskConfigInput` wrapping a real
`windbreak.config.schema.RiskConfig` built from the bundle's override fields
over the schema's own defaults. Issue #45 enriches the fourth and last
placeholder: a bundle's `positions` object becomes a `PositionReadModelInput`
carrying the snapshot id plus the eight money-valued capital/exposure fields
the dispersion-scaled Kelly sizing (and its five notional caps) reads, each
wrapped in `windbreak.numeric.MoneyMicros` at load time.

This module is deliberately free of any `pytest` import: `test_determinism_
golden.py`'s fresh-interpreter check imports it from a bare `python -c`
subprocess with no test runner present, so it must be importable and usable
on its own.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from windbreak.config.schema import RiskConfig
from windbreak.connector.fees import FeeModel
from windbreak.connector.models import OrderBookLevel, OrderBookSnapshot
from windbreak.forecast.records import Citation, ForecastRecord, ModelVote
from windbreak.numeric import ContractCentis, MoneyMicros, PricePips
from windbreak.selector import SelectorInputs
from windbreak.selector.correlation import BucketExposureEntry, CorrelationTag
from windbreak.selector.types import (
    FeeModelInput,
    PositionReadModelInput,
    RiskConfigInput,
    SlippageModelInput,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _parse_dt(value: str, *, field: str) -> datetime:
    """Parse an offset-bearing ISO-8601 fixture timestamp into a UTC datetime.

    The offset is required, matching
    :func:`windbreak.connector.fake._parse_dt` and
    :func:`windbreak.connector.kalshi.normalize._parse_dt`. ``fromisoformat``
    returns a *naive* datetime for an offsetless string, and ``.astimezone(UTC)``
    does not raise on one -- it reads the wall clock as the **host's** local
    time, so the same checked-in bundle would mean different instants on a
    developer's laptop and in UTC-running CI (issue #392). Every bundle under
    ``tests/selector/fixtures/`` carries a ``Z`` today; this guard is what keeps
    that true, rather than leaving it a property nothing checks.

    Args:
        value: An ISO-8601 string carrying a UTC offset, e.g.
            ``"2024-12-10T12:00:00.000000Z"``.
        field: The bundle field name ``value`` came from, named in the error so
            the offending entry is locatable.

    Returns:
        The timezone-aware datetime, normalized to UTC.

    Raises:
        ValueError: If ``value`` is not parseable as ISO-8601, or parses to a
            naive (offsetless) datetime.
    """
    parsed = datetime.fromisoformat(value)
    # Naive means `tzinfo is None` *or* a `tzinfo` whose `utcoffset()` returns
    # None -- Python's own definition, reported by `utcoffset()` in one call.
    if parsed.utcoffset() is None:
        raise ValueError(
            f"Fixture timestamp field {field!r} carries no UTC offset: {value!r}. "
            "An offsetless timestamp would be read against the host's local "
            "timezone, so the bundle is refused rather than loaded against a "
            "guessed offset."
        )
    return parsed.astimezone(UTC)


def _parse_optional_dt(value: str | None, *, field: str) -> datetime | None:
    """Parse an optional offset-bearing ISO-8601 timestamp, preserving ``None``.

    Args:
        value: An ISO-8601 string carrying a UTC offset, or ``None``.
        field: The bundle field name ``value`` came from, named in the error.

    Returns:
        The parsed UTC datetime, or ``None`` when ``value`` is ``None``.

    Raises:
        ValueError: If ``value`` is present and offsetless or unparseable.
    """
    return None if value is None else _parse_dt(value, field=field)


def _model_vote_from_dict(data: Mapping[str, object]) -> ModelVote:
    """Build a :class:`ModelVote` from one raw ``model_votes`` entry.

    Args:
        data: One raw ``model_votes`` list entry from a bundle JSON file.

    Returns:
        The constructed, post-init-validated :class:`ModelVote`.
    """
    return ModelVote(
        provider=data["provider"],
        model_version=data["model_version"],
        declared_training_cutoff=data["declared_training_cutoff"],
        probability_ppm=data["probability_ppm"],
        response_fingerprint=data["response_fingerprint"],
    )


def _citation_from_dict(data: Mapping[str, object]) -> Citation:
    """Build a :class:`Citation` from one raw ``citations`` entry.

    Args:
        data: One raw ``citations`` list entry from a bundle JSON file.

    Returns:
        The constructed :class:`Citation`, with its ``publication_date``
        parsed from an ISO-8601 string (or preserved as ``None``).
    """
    return Citation(
        url=data["url"],
        content_hash=data["content_hash"],
        quoted_text=data["quoted_text"],
        publication_date=_parse_optional_dt(
            data["publication_date"], field="publication_date"
        ),
        source_type=data["source_type"],
    )


def _forecast_record_from_dict(data: Mapping[str, object]) -> ForecastRecord:
    """Build a :class:`ForecastRecord` from a bundle's ``forecast`` object.

    Args:
        data: The raw ``forecast`` mapping from a bundle JSON file.

    Returns:
        The constructed, post-init-validated :class:`ForecastRecord`.
    """
    return ForecastRecord(
        forecast_id=data["forecast_id"],
        market_ticker=data["market_ticker"],
        normalized_question_hash=data["normalized_question_hash"],
        probability_ppm=data["probability_ppm"],
        ci_low_ppm=data["ci_low_ppm"],
        ci_high_ppm=data["ci_high_ppm"],
        model_votes=tuple(_model_vote_from_dict(vote) for vote in data["model_votes"]),
        vote_dispersion_ppm=data["vote_dispersion_ppm"],
        rationale_markdown=data["rationale_markdown"],
        citations=tuple(
            _citation_from_dict(citation) for citation in data["citations"]
        ),
        source_quality_notes=tuple(data["source_quality_notes"]),
        research_cost_micros=data["research_cost_micros"],
        triage_stage=data["triage_stage"],
        created_at=_parse_dt(data["created_at"], field="created_at"),
        forecast_horizon_hours=data["forecast_horizon_hours"],
        market_price_baseline_pips=data["market_price_baseline_pips"],
        baseline_quote_snapshot_id=data["baseline_quote_snapshot_id"],
        coherence_group_sum_ppm=data["coherence_group_sum_ppm"],
        coherence_flag=data["coherence_flag"],
        abstention_reason=data["abstention_reason"],
        eligible_for_live=data["eligible_for_live"],
    )


def _order_book_level_from_dict(data: Mapping[str, object]) -> OrderBookLevel:
    """Build a unit-wrapped :class:`OrderBookLevel` from one raw level entry.

    Args:
        data: One raw ``yes_bids``/``yes_asks`` list entry.

    Returns:
        The constructed :class:`OrderBookLevel`, with ``price`` and
        ``quantity`` wrapped in their scaled-integer unit types.
    """
    return OrderBookLevel(
        price=PricePips(data["price"]),
        quantity=ContractCentis(data["quantity"]),
    )


def _order_book_from_dict(data: Mapping[str, object]) -> OrderBookSnapshot:
    """Build an :class:`OrderBookSnapshot` from a bundle's ``order_book`` object.

    Args:
        data: The raw ``order_book`` mapping from a bundle JSON file.

    Returns:
        The constructed :class:`OrderBookSnapshot`.
    """
    return OrderBookSnapshot(
        ticker=data["ticker"],
        yes_bids=tuple(
            _order_book_level_from_dict(level) for level in data["yes_bids"]
        ),
        yes_asks=tuple(
            _order_book_level_from_dict(level) for level in data["yes_asks"]
        ),
        fetched_at=_parse_dt(data["fetched_at"], field="fetched_at"),
    )


def _fee_model_input_from_dict(data: Mapping[str, object]) -> FeeModelInput:
    """Build a :class:`FeeModelInput` from a bundle's ``fee_model`` object.

    Args:
        data: The raw ``fee_model`` mapping from a bundle JSON file, carrying
            the schedule's rate fields plus an ``as_of`` freshness timestamp.

    Returns:
        The constructed :class:`FeeModelInput`, wrapping a real,
        post-init-validated :class:`~windbreak.connector.fees.FeeModel`.
    """
    model = FeeModel(
        schedule_id=data["schedule_id"],
        maker_fee_ppm=data["maker_fee_ppm"],
        taker_fee_ppm=data["taker_fee_ppm"],
        settlement_fee_ppm=data["settlement_fee_ppm"],
    )
    return FeeModelInput(model=model, as_of=_parse_dt(data["as_of"], field="as_of"))


def _slippage_input_from_dict(data: Mapping[str, object]) -> SlippageModelInput:
    """Build a :class:`SlippageModelInput` from a bundle's ``slippage_model`` object.

    Args:
        data: The raw ``slippage_model`` mapping from a bundle JSON file.

    Returns:
        The constructed :class:`SlippageModelInput`.
    """
    return SlippageModelInput(
        model_id=data["model_id"],
        per_contract_buffer_ppm=data["per_contract_buffer_ppm"],
    )


def _positions_input_from_dict(data: Mapping[str, object]) -> PositionReadModelInput:
    """Build a :class:`PositionReadModelInput` from a bundle's `positions` object.

    Args:
        data: The raw ``positions`` mapping from a bundle JSON file, carrying
            the snapshot id plus the eight money-valued capital/exposure
            fields the dispersion-scaled Kelly sizing (issue #45) reads.

    Returns:
        The constructed :class:`PositionReadModelInput`, with every money
        field wrapped in :class:`~windbreak.numeric.MoneyMicros`.
    """
    return PositionReadModelInput(
        snapshot_id=data["snapshot_id"],
        equity_micros=MoneyMicros(data["equity_micros"]),
        above_floor_capital_micros=MoneyMicros(data["above_floor_capital_micros"]),
        total_deploy_cap_micros=MoneyMicros(data["total_deploy_cap_micros"]),
        market_exposure=MoneyMicros(data["market_exposure"]),
        event_exposure=MoneyMicros(data["event_exposure"]),
        bucket_exposure=MoneyMicros(data["bucket_exposure"]),
        total_exposure=MoneyMicros(data["total_exposure"]),
        notional_today=MoneyMicros(data["notional_today"]),
    )


def _risk_config_input_from_dict(data: Mapping[str, object]) -> RiskConfigInput:
    """Build a :class:`RiskConfigInput` from a bundle's ``risk_config`` object.

    Every key besides ``config_hash`` is passed through verbatim as a
    :class:`~windbreak.config.schema.RiskConfig` field override, so a bundle
    that supplies none inherits every :class:`RiskConfig` default.

    Args:
        data: The raw ``risk_config`` mapping from a bundle JSON file.

    Returns:
        The constructed :class:`RiskConfigInput`.
    """
    overrides = {key: value for key, value in data.items() if key != "config_hash"}
    return RiskConfigInput(
        config=RiskConfig(**overrides), config_hash=data["config_hash"]
    )


def _correlation_tag_from_dict(data: Mapping[str, object]) -> CorrelationTag:
    """Build a :class:`CorrelationTag` from one raw ``correlation_tags`` entry.

    Args:
        data: One raw ``correlation_tags`` list entry, carrying the ``bucket_id``,
            ``source``, and ISO-8601 ``tagged_at`` provenance timestamp.

    Returns:
        The constructed, post-init-validated :class:`CorrelationTag`.
    """
    return CorrelationTag(
        bucket_id=data["bucket_id"],
        source=data["source"],
        tagged_at=_parse_dt(data["tagged_at"], field="tagged_at"),
    )


def _bucket_peer_from_dict(data: Mapping[str, object]) -> BucketExposureEntry:
    """Build a :class:`BucketExposureEntry` from one raw ``bucket_peers`` entry.

    Args:
        data: One raw ``bucket_peers`` list entry, carrying the peer's
            ``market_ticker``, its ``exposure_micros``, and its own
            ``correlation_tags``.

    Returns:
        The constructed :class:`BucketExposureEntry`, with ``exposure_micros``
        wrapped in :class:`~windbreak.numeric.MoneyMicros`.
    """
    return BucketExposureEntry(
        market_ticker=data["market_ticker"],
        exposure_micros=MoneyMicros(data["exposure_micros"]),
        tags=tuple(_correlation_tag_from_dict(tag) for tag in data["correlation_tags"]),
    )


def load_inputs(path: str | Path) -> SelectorInputs:
    """Parse one recorded bundle JSON file into a `SelectorInputs`.

    Args:
        path: Path to a bundle JSON file (e.g.
            ``tests/selector/fixtures/bundle_a.json``).

    Returns:
        A fully constructed `SelectorInputs`, with its `forecast` and
        `order_book` built from the real, post-init-validated domain types
        and every placeholder ref wrapped in its `windbreak.selector.types`
        dataclass.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return SelectorInputs(
        forecast=_forecast_record_from_dict(raw["forecast"]),
        calibration_map_version=raw["calibration_map_version"],
        order_book=_order_book_from_dict(raw["order_book"]),
        fee_model=_fee_model_input_from_dict(raw["fee_model"]),
        slippage_model=_slippage_input_from_dict(raw["slippage_model"]),
        positions=_positions_input_from_dict(raw["positions"]),
        risk_config=_risk_config_input_from_dict(raw["risk_config"]),
        correlation_tags=tuple(
            _correlation_tag_from_dict(tag) for tag in raw["correlation_tags"]
        ),
        bucket_peers=tuple(
            _bucket_peer_from_dict(peer) for peer in raw.get("bucket_peers", ())
        ),
    )
