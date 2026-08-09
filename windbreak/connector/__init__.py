"""Shared component: the read-only exchange connector surface.

Per SPEC S5.2, this package models only *public, read-only market access* --
it carries no trade credentials. It defines the :class:`MarketConnector`
protocol (SPEC S7.2), the normalized SPEC S6.2 models, a fixture-backed
:class:`FakeExchange`, and the :class:`MarketSnapshotTask` that records a
market snapshot and screening decision per market through an
:class:`EventLedgerWriter` seam. Issue #20 adds the fail-closed data-quality
surface: caller-scoped freshness checks (:mod:`windbreak.connector.freshness`),
schema-drift validation (:mod:`windbreak.connector.validation`), and rate
limiting / retry / circuit breaking (:mod:`windbreak.connector.resilience`).
:mod:`windbreak.connector.readonly` narrows the surface further still, to the
five account/market reads a verification cycle may hold (SPEC S1.1 invariant 3).
Everything on the price/money path uses :mod:`windbreak.numeric` scaled-integer
types -- never floats (enforced by ``scripts/lint_no_floats.py``).
"""

from windbreak.connector.fake import FakeExchange
from windbreak.connector.fees import FeeModel, UnknownFeeModelError
from windbreak.connector.fills import (
    DEFAULT_FEE_HAIRCUT_PPM,
    DEFAULT_MAX_PARTICIPATION_PPM,
    PAPER_FILL_MODEL_VERSION,
    TakerFillResult,
    TradePrint,
    participation_cap,
    resting_fill_quantity,
    walk_taker_fill,
)
from windbreak.connector.freshness import StaleSnapshotError, ensure_fresh, is_fresh
from windbreak.connector.interface import MarketConnector, UnknownMarketError
from windbreak.connector.models import (
    BalanceSnapshot,
    ExchangeStatus,
    Fill,
    NormalizedMarket,
    OpenOrder,
    OrderBookLevel,
    OrderBookSnapshot,
    Position,
    market_to_payload,
)
from windbreak.connector.paper import PaperExchange, PaperOrderIntent
from windbreak.connector.readonly import ReadOnlyConnectorView, ReadOnlyVenueView
from windbreak.connector.resilience import (
    CONNECTOR_HALT_EVENT,
    DEFAULT_RESILIENCE_POLICY,
    CircuitBreaker,
    CircuitState,
    ConnectorHaltError,
    MaintenanceHaltError,
    ResiliencePolicy,
    ResilientCaller,
    TokenBucket,
    build_default_resilient_caller,
)
from windbreak.connector.semantics import (
    BalanceSemantics,
    CancelCollateralRelease,
    FeeDebitTiming,
    FeeRounding,
    HaltedMarketBehavior,
    OrderCollateralInAvailable,
    OrderCollateralInTotal,
    PartialFillRepresentation,
    UnsettledProceeds,
)
from windbreak.connector.snapshot import (
    MARKET_SNAPSHOT_EVENT,
    SCREEN_DECISION_EVENT,
    ConnectorEvent,
    EventLedgerWriter,
    InMemoryEventLedgerWriter,
    LoggingEventLedgerWriter,
    MarketSnapshotTask,
)
from windbreak.connector.validation import (
    SCHEMA_ANOMALY_EVENT,
    ResponseSchema,
    SchemaAnomalyHaltError,
    SchemaRegistry,
    SchemaValidator,
    kalshi_default_schema_registry,
)

__all__ = [
    "CONNECTOR_HALT_EVENT",
    "DEFAULT_FEE_HAIRCUT_PPM",
    "DEFAULT_MAX_PARTICIPATION_PPM",
    "DEFAULT_RESILIENCE_POLICY",
    "MARKET_SNAPSHOT_EVENT",
    "PAPER_FILL_MODEL_VERSION",
    "SCHEMA_ANOMALY_EVENT",
    "SCREEN_DECISION_EVENT",
    "BalanceSemantics",
    "BalanceSnapshot",
    "CancelCollateralRelease",
    "CircuitBreaker",
    "CircuitState",
    "ConnectorEvent",
    "ConnectorHaltError",
    "EventLedgerWriter",
    "ExchangeStatus",
    "FakeExchange",
    "FeeDebitTiming",
    "FeeModel",
    "FeeRounding",
    "Fill",
    "HaltedMarketBehavior",
    "InMemoryEventLedgerWriter",
    "LoggingEventLedgerWriter",
    "MaintenanceHaltError",
    "MarketConnector",
    "MarketSnapshotTask",
    "NormalizedMarket",
    "OpenOrder",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "OrderCollateralInAvailable",
    "OrderCollateralInTotal",
    "PaperExchange",
    "PaperOrderIntent",
    "PartialFillRepresentation",
    "Position",
    "ReadOnlyConnectorView",
    "ReadOnlyVenueView",
    "ResiliencePolicy",
    "ResilientCaller",
    "ResponseSchema",
    "SchemaAnomalyHaltError",
    "SchemaRegistry",
    "SchemaValidator",
    "StaleSnapshotError",
    "TakerFillResult",
    "TokenBucket",
    "TradePrint",
    "UnknownFeeModelError",
    "UnknownMarketError",
    "UnsettledProceeds",
    "build_default_resilient_caller",
    "ensure_fresh",
    "is_fresh",
    "kalshi_default_schema_registry",
    "market_to_payload",
    "participation_cap",
    "resting_fill_quantity",
    "walk_taker_fill",
]
