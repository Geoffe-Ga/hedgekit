"""Tests for `windbreak.ledger.events` (issue #13).

Pins the event/envelope contract that `windbreak.ledger.store` hashes and
persists:

- `canonical_json` is a deterministic, whitespace-free serialization of a
  dict, independent of key insertion order.
- `utc_now_iso` returns a UTC ISO-8601 timestamp with microsecond
  precision.
- `Event` is a frozen base dataclass with `event_type`, `component`,
  `payload_schema_version`, and `payload` fields, plus an `envelope_json`
  property wrapping them as `{"component", "data", "schema_version"}`.
- The three M0 event subtypes (`ConfigLoaded`, `ModeHeartbeat`,
  `AlertEmitted`) are frozen dataclasses whose ergonomic constructors
  (e.g. `ConfigLoaded(component=..., config_hash=..., diff=...)`)
  auto-populate `event_type` (equal to the class name) and
  `payload_schema_version`, and whose `payload` property assembles the
  typed fields into the persisted payload dict.
- `EVENT_TYPES` maps each `event_type` string to its class, so a
  persisted envelope can be reconstructed as
  `EVENT_TYPES[event_type](component=..., **data)`.

Issue #40 moves the four Order Gateway event types (`OrderTransitionLedgered`,
`SubmissionRefused`, `ReduceOnlyRefused`, `ReduceOnlyViolation`) into this
module (still re-exported from `windbreak.order_gateway.ledger_writer` for
backward compatibility) and adds three crash-recovery event types
(`ReconciliationHalted`, `ReconciliationHealed`, `RecoveryCompleted`), growing
`EVENT_TYPES` to 16 entries. Each gets the same registry-round-trip coverage
as every other concrete event type.

Issue #41 (RED -- `MarketFreeze`/`ReturnToScreener` do not exist yet, so the
import below fails collection with `ImportError: cannot import name
'MarketFreeze' from 'windbreak.ledger.events'`) adds two more Order Gateway
event types for the adverse-selection sweeper, growing `EVENT_TYPES` to 18
entries:

    * `MarketFreeze` -- a strict beyond-N-ticks move on a resting order's
      side-matched top of book freezes the whole ticker. `event_type` is the
      literal class name `"MarketFreeze"`, never the issue sketch's
      shouty-snake-case `"MARKET_FREEZE"` (every concrete `Event` subtype
      derives `event_type` from `type(self).__name__`).
    * `ReturnToScreener` -- the companion event marking a frozen ticker's
      orders as returned to manual/algorithmic re-screening, `reason` always
      `"market_freeze"`.

Issue #180 (RED -- `GatePlanRegistered`/`GatePlanChanged`/
`GateComputationMismatch` do not exist in this module yet, so the import
below fails collection with `ImportError: cannot import name
'GatePlanRegistered' from 'windbreak.ledger.events'`) moves all three
evaluation-defined events here, growing `EVENT_TYPES` to 28 entries. The two
`GatePlan*` events are redesigned as flat, fully-typed frozen dataclasses
whose constructor fields ARE the flattened payload keys -- the thirteen
canonical plan keys (`metric_windows` plus the eight int thresholds plus the
four str identity fields, matching `GatePlan.canonical_dict()`) plus
`plan_hash`, `paper_clock_start` (and `previous_plan_hash` for
`GatePlanChanged`) -- so `EVENT_TYPES[t](component=..., **envelope["data"])`
round-trips by construction, with no separate `plan_dict` wrapper key ever
appearing in the persisted payload. `GateComputationMismatch` moves verbatim.

Issue #195 (RED -- `CanaryVerdictRecorded` does not exist in this module yet,
so the import below fails collection with `ImportError: cannot import name
'CanaryVerdictRecorded' from 'windbreak.ledger.events'`) adds the M0 event the
scheduler's `run_canaries` composition root appends one of per provider per
canary battery run (fleet observability, SPEC S8.4/S16 extended per-provider):
`provider`, `status` (`ProviderCanaryStatus.name`), `drift_kind` (`"answer"`,
`"version"`, or `""` for a clean `OK` verdict), `drift_score_ppm`,
`tolerance_ppm`, `reported_version`, and `pinned_versions` (a list of the
provider's pinned version strings -- plural, since a provider's pin set may
carry more than one accepted version). Growing `EVENT_TYPES` to 29 entries.

Issue #244 (RED -- `PromotionBlocked` does not exist in this module yet, so
the import below fails collection with `ImportError: cannot import name
'PromotionBlocked' from 'windbreak.ledger.events'`) adds the optional
fail-closed-promotion audit event the Risk Kernel may ledger when a PAPER ->
LIVE_MICRO promotion attempt is refused before any gate is even evaluated
(`GatePlanUnavailableError`): `source_mode`, `target_mode`, `reason`. It
mirrors `CanaryVerdictRecorded`'s shape exactly -- `event_type` is the literal
class name `"PromotionBlocked"`, derived via `_derive_typed_event`, and
`payload_schema_version` is the module-wide default (`1`). Growing
`EVENT_TYPES` to 30 entries.

Issue #281 (RED -- `ProviderVoteRecorded` does not exist in this module yet,
so the import below fails collection with `ImportError: cannot import name
'ProviderVoteRecorded' from 'windbreak.ledger.events'`) adds the per-provider
vote-cost audit event the scheduler's `_forecast_stage` composition root
appends once per ensemble member per paper tick (per-provider vote-cost
signal, issue #281): `forecast_id`, `market_ticker`, `provider`,
`model_version`, `vote_index`, `cost_micros`, `outcome` (`"voted"` /
`"abstained"` / `"discarded"`), and `failure_code` (`""` unless
`outcome == "discarded"`, matching this module's inapplicable-string
convention -- see `CanaryVerdictRecorded.drift_kind`). It mirrors
`CanaryVerdictRecorded`'s shape exactly: `event_type` is the literal class
name `"ProviderVoteRecorded"`, derived via `_derive_typed_event`, and
`payload_schema_version` is the module-wide default (`1`). Growing
`EVENT_TYPES` to 31 entries. Unlike every other event in this module,
`ProviderVoteRecorded` is also re-exported directly from `windbreak.ledger`
(the package `__init__.py`), so `test_provider_vote_recorded_is_reexported_
from_windbreak_ledger` below fails today with `AttributeError: module
'windbreak.ledger' has no attribute 'ProviderVoteRecorded'` once the
`ImportError` above is otherwise resolved.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import json
import re
from pathlib import Path

import pytest

from windbreak.ledger.events import (
    EVENT_TYPES,
    GENESIS_PREV_HASH,
    AlertEmitted,
    CanaryVerdictRecorded,
    CancelAllDirective,
    ConfigLoaded,
    DemotionTriggerFired,
    DrillCompleted,
    EquitySampled,
    Event,
    ExchangeStatusObserved,
    ForecastCreated,
    GateComputationMismatch,
    GatePlanChanged,
    GatePlanRegistered,
    KillEngaged,
    KillReArmed,
    MarketFreeze,
    MarketSnapshotRecorded,
    ModeHeartbeat,
    OrderTransitionLedgered,
    PipelineHeartbeatRecorded,
    PositionsSnapshotRecorded,
    PromotionBlocked,
    PromotionEvaluated,
    ProviderVoteRecorded,
    ReconciliationHalted,
    ReconciliationHealed,
    RecoveryCompleted,
    ReduceOnlyRefused,
    ReduceOnlyViolation,
    ResearchBudgetHalted,
    ReturnToScreener,
    ScreenDecisionRecorded,
    SelectorDecisionRecorded,
    SignificanceOverrideApplied,
    SubmissionRefused,
    canonical_json,
    utc_now_iso,
)

_ISO_UTC_MICROSECOND_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$"
)

#: The three key/value pairs canonical_json tests permute the order of.
_SAMPLE_ITEMS = [("b", 1), ("a", 2), ("c", 3)]

#: Repo root, derived from this test file's own location
#: (`<root>/tests/ledger/test_ledger_events.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The module under test itself, scanned for forbidden `windbreak.evaluation`
#: imports by the acyclicity guard below.
_LEDGER_EVENTS_PATH = _REPO_ROOT / "windbreak" / "ledger" / "events.py"

#: The thirteen canonical `GatePlan` keys a `GatePlanRegistered`/
#: `GatePlanChanged` payload flattens in, matching
#: `GatePlan.canonical_dict()` exactly (issue #180).
_SAMPLE_PLAN_FIELDS: dict[str, object] = {
    "metric_windows": [["brier", "all"]],
    "min_resolved_for_calibration": 150,
    "promotion_min_resolved": 300,
    "promotion_min_independent_event_groups": 100,
    "brier_skill_required_ppm": 10_000,
    "bootstrap_confidence_ppm": 950_000,
    "live_rolling_window_size": 100,
    "live_slippage_ratio_limit_ppm": 1_500_000,
    "live_brier_degradation_band_ppm": 50_000,
    "observation_window": "latest_before_close",
    "baseline_scheme": "executable_price_at_baseline_snapshot",
    "clustering_scheme": "event_correlation_group",
    "paper_fill_model_version": "pfm-v1",
}

#: A 64-hex-char sample plan hash for the round-trip/pin tests below.
_SAMPLE_PLAN_HASH = "a" * 64

#: A distinct 64-hex-char sample hash standing in for the plan a
#: `GatePlanChanged` replaced.
_SAMPLE_PREVIOUS_PLAN_HASH = "b" * 64


def test_genesis_prev_hash_is_sixty_four_zero_characters() -> None:
    """GENESIS_PREV_HASH is the documented all-zero SHA-256-width sentinel."""
    assert GENESIS_PREV_HASH == "0" * 64
    assert len(GENESIS_PREV_HASH) == 64


@pytest.mark.parametrize("ordered_items", list(itertools.permutations(_SAMPLE_ITEMS)))
def test_canonical_json_is_independent_of_dict_insertion_order(
    ordered_items: tuple[tuple[str, int], ...],
) -> None:
    """Every insertion order of the same key/value pairs serializes identically."""
    obj = dict(ordered_items)

    assert canonical_json(obj) == '{"a":2,"b":1,"c":3}'


def test_canonical_json_contains_no_whitespace() -> None:
    """Nested structures still serialize with zero whitespace characters."""
    result = canonical_json({"nested": {"z": 1, "a": 2}, "list": [3, 1, 2]})

    assert " " not in result
    assert "\n" not in result
    assert "\t" not in result


def test_canonical_json_matches_sorted_compact_json_dumps() -> None:
    """canonical_json agrees with the equivalent explicit json.dumps call."""
    obj = {"z": 1, "a": {"y": 2, "x": 1}}
    expected = json.dumps(obj, sort_keys=True, separators=(",", ":"))

    assert canonical_json(obj) == expected


def test_utc_now_iso_returns_utc_iso8601_with_microseconds() -> None:
    """utc_now_iso() returns a UTC-offset ISO-8601 string with microseconds."""
    timestamp = utc_now_iso()

    assert _ISO_UTC_MICROSECOND_RE.match(timestamp) is not None, timestamp
    assert timestamp.endswith("+00:00")


def test_event_base_class_exposes_all_four_fields() -> None:
    """The base Event dataclass carries event_type/component/schema/payload."""
    event = Event(
        event_type="ConfigLoaded",
        component="pipeline",
        payload_schema_version=1,
        payload={"config_hash": "abc", "diff": {}},
    )

    assert event.event_type == "ConfigLoaded"
    assert event.component == "pipeline"
    assert event.payload_schema_version == 1
    assert event.payload == {"config_hash": "abc", "diff": {}}


def test_event_envelope_json_has_component_data_schema_version_sorted_keys() -> None:
    """envelope_json wraps the four fields into the pinned envelope shape."""
    event = Event(
        event_type="ConfigLoaded",
        component="pipeline",
        payload_schema_version=1,
        payload={"config_hash": "abc", "diff": {}},
    )

    envelope = json.loads(event.envelope_json)

    assert envelope == {
        "component": "pipeline",
        "data": {"config_hash": "abc", "diff": {}},
        "schema_version": 1,
    }
    assert event.envelope_json == canonical_json(
        {
            "component": "pipeline",
            "data": {"config_hash": "abc", "diff": {}},
            "schema_version": 1,
        }
    )


def test_event_is_frozen() -> None:
    """Event instances cannot be mutated after construction."""
    event = Event(
        event_type="ConfigLoaded",
        component="pipeline",
        payload_schema_version=1,
        payload={},
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.component = "changed"  # type: ignore[misc]


def test_config_loaded_populates_event_type_schema_version_and_payload() -> None:
    """ConfigLoaded's ergonomic constructor derives the full Event contract."""
    event = ConfigLoaded(component="pipeline", config_hash="deadbeef", diff={"x": 1})

    assert event.event_type == "ConfigLoaded"
    assert event.component == "pipeline"
    assert event.payload_schema_version == 1
    assert event.payload == {"config_hash": "deadbeef", "diff": {"x": 1}}


def test_mode_heartbeat_populates_event_type_schema_version_and_payload() -> None:
    """ModeHeartbeat's ergonomic constructor derives the full Event contract."""
    event = ModeHeartbeat(component="pipeline", mode="RESEARCH", beat=7)

    assert event.event_type == "ModeHeartbeat"
    assert event.component == "pipeline"
    assert event.payload_schema_version == 1
    assert event.payload == {"mode": "RESEARCH", "beat": 7}


def test_alert_emitted_populates_event_type_schema_version_and_payload() -> None:
    """AlertEmitted's ergonomic constructor derives the full Event contract."""
    event = AlertEmitted(component="alerts", severity="high", message="disk full")

    assert event.event_type == "AlertEmitted"
    assert event.component == "alerts"
    assert event.payload_schema_version == 1
    assert event.payload == {"severity": "high", "message": "disk full"}


def test_config_loaded_envelope_json_matches_canonical_envelope() -> None:
    """envelope_json is the canonical envelope.

    The persisted object is exactly the {"component", "data",
    "schema_version"} shape.
    """
    event = ConfigLoaded(component="pipeline", config_hash="deadbeef", diff={"x": 1})

    envelope = json.loads(event.envelope_json)

    assert envelope == {
        "component": "pipeline",
        "data": {"config_hash": "deadbeef", "diff": {"x": 1}},
        "schema_version": 1,
    }


def test_config_loaded_is_frozen() -> None:
    """ConfigLoaded, like the base Event, is immutable after construction."""
    event = ConfigLoaded(component="pipeline", config_hash="abc", diff={})

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.config_hash = "changed"  # type: ignore[misc]


def test_event_types_registry_maps_type_name_to_class() -> None:
    """EVENT_TYPES lets a persisted event_type string recover its class."""
    assert {
        "ConfigLoaded": ConfigLoaded,
        "ModeHeartbeat": ModeHeartbeat,
        "AlertEmitted": AlertEmitted,
        "PromotionEvaluated": PromotionEvaluated,
        "SignificanceOverrideApplied": SignificanceOverrideApplied,
        "DemotionTriggerFired": DemotionTriggerFired,
        "KillEngaged": KillEngaged,
        "CancelAllDirective": CancelAllDirective,
        "KillReArmed": KillReArmed,
        "OrderTransitionLedgered": OrderTransitionLedgered,
        "SubmissionRefused": SubmissionRefused,
        "ReduceOnlyRefused": ReduceOnlyRefused,
        "ReduceOnlyViolation": ReduceOnlyViolation,
        "ReconciliationHalted": ReconciliationHalted,
        "ReconciliationHealed": ReconciliationHealed,
        "RecoveryCompleted": RecoveryCompleted,
        "MarketFreeze": MarketFreeze,
        "ReturnToScreener": ReturnToScreener,
        "MarketSnapshotRecorded": MarketSnapshotRecorded,
        "ScreenDecisionRecorded": ScreenDecisionRecorded,
        "ForecastCreated": ForecastCreated,
        "SelectorDecisionRecorded": SelectorDecisionRecorded,
        "EquitySampled": EquitySampled,
        "PositionsSnapshotRecorded": PositionsSnapshotRecorded,
        "DrillCompleted": DrillCompleted,
        "GatePlanRegistered": GatePlanRegistered,
        "GatePlanChanged": GatePlanChanged,
        "GateComputationMismatch": GateComputationMismatch,
        "CanaryVerdictRecorded": CanaryVerdictRecorded,
        "PromotionBlocked": PromotionBlocked,
        "ProviderVoteRecorded": ProviderVoteRecorded,
        "ResearchBudgetHalted": ResearchBudgetHalted,
        "ExchangeStatusObserved": ExchangeStatusObserved,
        "PipelineHeartbeatRecorded": PipelineHeartbeatRecorded,
    } == EVENT_TYPES


def test_event_types_registry_round_trips_from_payload_data() -> None:
    """A registry lookup plus the persisted `data` dict reconstructs the event."""
    original = ConfigLoaded(component="pipeline", config_hash="deadbeef", diff={"x": 1})
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


# --- Issue #40: registry round-trips for the four moved Gateway events and ----
# --- the three new crash-recovery events, mirroring the test above -----------


def test_event_types_registry_round_trips_order_transition_ledgered() -> None:
    """A registry lookup + persisted `data` reconstructs `OrderTransitionLedgered`."""
    original = OrderTransitionLedgered(
        component="order_gateway",
        client_order_id="coid-abc",
        from_state="INTENT_CREATED",
        event="APPROVE",
        to_state="APPROVED",
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_event_types_registry_round_trips_submission_refused() -> None:
    """A registry lookup plus persisted `data` reconstructs `SubmissionRefused`."""
    original = SubmissionRefused(
        component="order_gateway", client_order_id="coid-abc", reason="paused"
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_event_types_registry_round_trips_reduce_only_refused() -> None:
    """A registry lookup plus persisted `data` reconstructs `ReduceOnlyRefused`."""
    original = ReduceOnlyRefused(
        component="order_gateway",
        client_order_id="coid-abc",
        ticker="MKT-DEEP",
        held_centis=500,
        inflight_closing_centis=0,
        requested_close_centis=600,
        reason="reduce_only",
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_event_types_registry_round_trips_reduce_only_violation() -> None:
    """A registry lookup plus persisted `data` reconstructs `ReduceOnlyViolation`."""
    original = ReduceOnlyViolation(
        component="order_gateway",
        client_order_id="coid-abc",
        ticker="MKT-DEEP",
        held_centis=500,
        filled_centis=600,
        net_centis=-100,
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_event_types_registry_round_trips_reconciliation_halted() -> None:
    """A registry lookup plus persisted `data` reconstructs `ReconciliationHalted`."""
    original = ReconciliationHalted(
        component="order_gateway",
        reason="foreign_open_order",
        ticker="MKT-DEEP",
        venue_order_id="paper-order-9",
        client_order_id="",
        detail="untracked order discovered on the venue",
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_event_types_registry_round_trips_reconciliation_healed() -> None:
    """A registry lookup plus persisted `data` reconstructs `ReconciliationHealed`."""
    original = ReconciliationHealed(
        component="order_gateway",
        client_order_id="coid-abc",
        action="fill_confirmed",
        detail="matched an out-of-band fill",
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_event_types_registry_round_trips_recovery_completed() -> None:
    """A registry lookup plus persisted `data` reconstructs `RecoveryCompleted`."""
    original = RecoveryCompleted(
        component="order_gateway", orders_reconciled=3, halted=False
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


# --- Issue #41: registry round-trips for the two sweeper/freeze events -------


def test_market_freeze_event_type_is_the_literal_class_name() -> None:
    """`MarketFreeze.event_type` is `"MarketFreeze"`, never `"MARKET_FREEZE"`.

    The issue's own sketch spells the *concept* in shouty-snake-case, but
    every concrete `Event` subtype derives `event_type` from
    `type(self).__name__` via `_derive_typed_event` -- nothing asks
    `MarketFreeze` to special-case that.
    """
    event = MarketFreeze(
        component="order_gateway",
        ticker="KXFED-25SEP-CUT25",
        trigger="cancel_on_move",
        baseline_price_pips=4500,
        observed_price_pips=4800,
        threshold_ticks=2,
        price_tick_pips=100,
        epoch=1_700_000_005,
    )

    assert event.event_type == "MarketFreeze"


def test_event_types_registry_round_trips_market_freeze() -> None:
    """A registry lookup plus persisted `data` reconstructs `MarketFreeze`."""
    original = MarketFreeze(
        component="order_gateway",
        ticker="KXFED-25SEP-CUT25",
        trigger="cancel_on_move",
        baseline_price_pips=4500,
        observed_price_pips=4800,
        threshold_ticks=2,
        price_tick_pips=100,
        epoch=1_700_000_005,
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_event_types_registry_round_trips_return_to_screener() -> None:
    """A registry lookup plus persisted `data` reconstructs `ReturnToScreener`."""
    original = ReturnToScreener(
        component="order_gateway",
        ticker="KXFED-25SEP-CUT25",
        reason="market_freeze",
        epoch=1_700_000_005,
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


# --- Issue #180: registry round-trips for the three evaluation-defined -------
# --- events moved into this module (the two GatePlan events plus the --------
# --- crosscheck mismatch event) -----------------------------------------------


def test_event_types_registry_round_trips_gate_plan_registered() -> None:
    """A registry lookup plus persisted `data` reconstructs `GatePlanRegistered`.

    The constructor's fields ARE the flattened payload keys, so this
    round-trips by construction once the class moves into this module.
    """
    original = GatePlanRegistered(
        component="evaluation",
        **_SAMPLE_PLAN_FIELDS,
        plan_hash=_SAMPLE_PLAN_HASH,
        paper_clock_start=1_700_000_000,
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_event_types_registry_round_trips_gate_plan_changed() -> None:
    """A registry lookup plus persisted `data` reconstructs `GatePlanChanged`.

    Carries `previous_plan_hash` in addition to `GatePlanRegistered`'s fields.
    """
    original = GatePlanChanged(
        component="evaluation",
        **_SAMPLE_PLAN_FIELDS,
        plan_hash=_SAMPLE_PLAN_HASH,
        paper_clock_start=1_700_000_100,
        previous_plan_hash=_SAMPLE_PREVIOUS_PLAN_HASH,
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_event_types_registry_round_trips_gate_computation_mismatch() -> None:
    """Registry lookup plus persisted `data` reconstructs `GateComputationMismatch`."""
    original = GateComputationMismatch(
        component="evaluation",
        plan_hash=_SAMPLE_PLAN_HASH,
        tolerance=1,
        mismatches=[
            {
                "name": "brier",
                "window": "latest_before_close",
                "python_value": 54_000,
                "sql_value": 55_000,
            }
        ],
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_gate_plan_registered_payload_is_exactly_the_flattened_plan_dict() -> None:
    """`GatePlanRegistered`'s persisted `data` is exactly the thirteen plan
    keys plus `plan_hash`/`paper_clock_start` -- never a separate `plan_dict`
    wrapper key, and always stamped `schema_version == 1`.
    """
    event = GatePlanRegistered(
        component="evaluation",
        **_SAMPLE_PLAN_FIELDS,
        plan_hash=_SAMPLE_PLAN_HASH,
        paper_clock_start=1_700_000_000,
    )

    envelope = json.loads(event.envelope_json)

    assert envelope["data"] == {
        **_SAMPLE_PLAN_FIELDS,
        "plan_hash": _SAMPLE_PLAN_HASH,
        "paper_clock_start": 1_700_000_000,
    }
    assert "plan_dict" not in envelope["data"]
    assert envelope["schema_version"] == 1


def test_gate_plan_changed_payload_is_exactly_the_flattened_plan_dict() -> None:
    """`GatePlanChanged`'s persisted `data` is exactly the thirteen plan keys
    plus `plan_hash`/`paper_clock_start`/`previous_plan_hash` -- never a
    separate `plan_dict` wrapper key, and always stamped `schema_version == 1`.
    """
    event = GatePlanChanged(
        component="evaluation",
        **_SAMPLE_PLAN_FIELDS,
        plan_hash=_SAMPLE_PLAN_HASH,
        paper_clock_start=1_700_000_100,
        previous_plan_hash=_SAMPLE_PREVIOUS_PLAN_HASH,
    )

    envelope = json.loads(event.envelope_json)

    assert envelope["data"] == {
        **_SAMPLE_PLAN_FIELDS,
        "plan_hash": _SAMPLE_PLAN_HASH,
        "paper_clock_start": 1_700_000_100,
        "previous_plan_hash": _SAMPLE_PREVIOUS_PLAN_HASH,
    }
    assert "plan_dict" not in envelope["data"]
    assert envelope["schema_version"] == 1


# --- Issue #195: CanaryVerdictRecorded, the per-provider canary-verdict event -


def test_canary_verdict_recorded_populates_event_type_and_payload() -> None:
    """`CanaryVerdictRecorded`'s ergonomic constructor derives the full
    `Event` contract and assembles its payload from typed fields.
    """
    event = CanaryVerdictRecorded(
        component="scheduler",
        provider="futuresearch",
        status="ANSWER_DRIFT",
        drift_kind="answer",
        drift_score_ppm=90_000,
        tolerance_ppm=50_000,
        reported_version="fs-2.0",
        pinned_versions=["fs-2.0"],
    )

    assert event.event_type == "CanaryVerdictRecorded"
    assert event.component == "scheduler"
    assert event.payload_schema_version == 1
    assert event.payload == {
        "provider": "futuresearch",
        "status": "ANSWER_DRIFT",
        "drift_kind": "answer",
        "drift_score_ppm": 90_000,
        "tolerance_ppm": 50_000,
        "reported_version": "fs-2.0",
        "pinned_versions": ["fs-2.0"],
    }


def test_canary_verdict_recorded_accepts_a_clean_ok_verdict_with_empty_drift_kind() -> (
    None
):
    """A clean `OK` verdict is representable: `drift_kind=""` (never `None`),
    matching the payload-leaf convention every other event in this module
    uses for an "inapplicable" string field.
    """
    event = CanaryVerdictRecorded(
        component="scheduler",
        provider="anthropic",
        status="OK",
        drift_kind="",
        drift_score_ppm=0,
        tolerance_ppm=50_000,
        reported_version="claude-1",
        pinned_versions=["claude-1"],
    )

    assert event.payload["drift_kind"] == ""
    assert event.payload["status"] == "OK"


def test_event_types_registry_round_trips_canary_verdict_recorded() -> None:
    """A registry lookup plus persisted `data` reconstructs `CanaryVerdictRecorded`."""
    original = CanaryVerdictRecorded(
        component="scheduler",
        provider="futuresearch",
        status="VERSION_DRIFT",
        drift_kind="version",
        drift_score_ppm=0,
        tolerance_ppm=50_000,
        reported_version="fs-2.1",
        pinned_versions=["fs-2.0"],
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_canary_verdict_recorded_envelope_json_has_json_safe_leaves() -> None:
    """The persisted envelope's `data` carries only int/str/bool/list leaves,
    never a float -- the package-wide no-float convention.
    """
    event = CanaryVerdictRecorded(
        component="scheduler",
        provider="futuresearch",
        status="ANSWER_DRIFT",
        drift_kind="answer",
        drift_score_ppm=90_000,
        tolerance_ppm=50_000,
        reported_version="fs-2.0",
        pinned_versions=["fs-2.0", "fs-2.1"],
    )

    envelope = json.loads(event.envelope_json)
    data = envelope["data"]

    assert isinstance(data["provider"], str)
    assert isinstance(data["status"], str)
    assert isinstance(data["drift_kind"], str)
    assert type(data["drift_score_ppm"]) is int
    assert type(data["tolerance_ppm"]) is int
    assert isinstance(data["reported_version"], str)
    assert all(isinstance(version, str) for version in data["pinned_versions"])


def test_canary_verdict_recorded_is_frozen() -> None:
    """`CanaryVerdictRecorded`, like every other concrete event, is immutable."""
    event = CanaryVerdictRecorded(
        component="scheduler",
        provider="futuresearch",
        status="OK",
        drift_kind="",
        drift_score_ppm=0,
        tolerance_ppm=50_000,
        reported_version="fs-2.0",
        pinned_versions=["fs-2.0"],
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.provider = "changed"  # type: ignore[misc]


def _find_evaluation_imports(tree: ast.AST) -> tuple[str, ...]:
    """Return every `windbreak.evaluation*` import target found in `tree`.

    Uses `ast` (not a text/substring scan) so a `:class:` cross-reference to
    an evaluation symbol inside a docstring never produces a false positive --
    only real `import`/`from ... import ...` statements are inspected.

    Args:
        tree: A parsed module AST.

    Returns:
        The offending dotted module names found, in AST-traversal order.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith("windbreak.evaluation")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("windbreak.evaluation")
        ):
            found.append(node.module)
    return tuple(found)


def test_ledger_events_module_imports_no_evaluation_package() -> None:
    """`windbreak.ledger.events` never imports `windbreak.evaluation*`.

    Issue #180 moves the two `GatePlan*` events and `GateComputationMismatch`
    from `windbreak.evaluation.preregistration`/`windbreak.evaluation.crosscheck`
    into this module; the evaluation package must stay a one-way runtime
    consumer of `windbreak.ledger.events`, never the reverse, or the two
    packages would import-cycle. A pure-`ast` scan (never a text/substring
    scan, which would false-positive on a docstring's `:class:` reference to
    an evaluation symbol) proves the module's actual import statements are
    clean.
    """
    source = _LEDGER_EVENTS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert _find_evaluation_imports(tree) == ()


# --- Issue #244: PromotionBlocked, the fail-closed-promotion audit event -----


def test_promotion_blocked_populates_event_type_and_payload() -> None:
    """`PromotionBlocked`'s ergonomic constructor derives the full `Event`
    contract and assembles its payload from typed fields.
    """
    event = PromotionBlocked(
        component="riskkernel",
        source_mode="PAPER",
        target_mode="LIVE_MICRO",
        reason="no gate plan store wired; promotion blocked (fail-closed)",
    )

    assert event.event_type == "PromotionBlocked"
    assert event.component == "riskkernel"
    assert event.payload_schema_version == 1
    assert event.payload == {
        "source_mode": "PAPER",
        "target_mode": "LIVE_MICRO",
        "reason": "no gate plan store wired; promotion blocked (fail-closed)",
    }


def test_promotion_blocked_envelope_json_has_schema_version_one() -> None:
    """The persisted envelope stamps `schema_version == 1`, matching the
    module-wide default every other M0 event but `ForecastCreated` uses.
    """
    event = PromotionBlocked(
        component="riskkernel",
        source_mode="PAPER",
        target_mode="LIVE_MICRO",
        reason="no registered gate plan; promotion blocked (fail-closed)",
    )

    envelope = json.loads(event.envelope_json)

    assert envelope["schema_version"] == 1
    assert envelope["component"] == "riskkernel"
    assert envelope["data"] == {
        "source_mode": "PAPER",
        "target_mode": "LIVE_MICRO",
        "reason": "no registered gate plan; promotion blocked (fail-closed)",
    }


def test_promotion_blocked_is_frozen() -> None:
    """`PromotionBlocked`, like every other concrete event, is immutable."""
    event = PromotionBlocked(
        component="riskkernel",
        source_mode="PAPER",
        target_mode="LIVE_MICRO",
        reason="no gate plan store wired; promotion blocked (fail-closed)",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.reason = "changed"  # type: ignore[misc]


def test_event_types_registry_round_trips_promotion_blocked() -> None:
    """A registry lookup plus persisted `data` reconstructs `PromotionBlocked`."""
    original = PromotionBlocked(
        component="riskkernel",
        source_mode="PAPER",
        target_mode="LIVE_MICRO",
        reason="registered gate plan is unreadable; promotion blocked (fail-closed)",
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


# --- Issue #281: ProviderVoteRecorded, the per-provider vote-cost event ------


def test_provider_vote_recorded_envelope_json_is_canonical() -> None:
    """`envelope_json` is the canonical (sorted-key, whitespace-free) envelope
    -- mirrors `test_config_loaded_envelope_json_matches_canonical_envelope`'s
    identical check for the base `Event` contract.
    """
    event = ProviderVoteRecorded(
        component="scheduler",
        forecast_id="fc-0001",
        market_ticker="MKT-DEEP",
        provider="openai",
        model_version="gpt-5-2025-08-07",
        vote_index=0,
        cost_micros=100,
        outcome="voted",
        failure_code="",
    )

    assert event.envelope_json == canonical_json(
        {
            "component": event.component,
            "data": event.payload,
            "schema_version": event.payload_schema_version,
        }
    )
    assert " " not in event.envelope_json
    assert "\n" not in event.envelope_json


def test_provider_vote_recorded_populates_event_type_and_payload() -> None:
    """`ProviderVoteRecorded`'s ergonomic constructor derives the full
    `Event` contract and assembles its payload from typed fields.
    """
    event = ProviderVoteRecorded(
        component="scheduler",
        forecast_id="fc-0001",
        market_ticker="MKT-DEEP",
        provider="openai",
        model_version="gpt-5-2025-08-07",
        vote_index=0,
        cost_micros=100,
        outcome="voted",
        failure_code="",
    )

    assert event.event_type == "ProviderVoteRecorded"
    assert event.component == "scheduler"
    assert event.payload_schema_version == 1
    assert event.payload == {
        "forecast_id": "fc-0001",
        "market_ticker": "MKT-DEEP",
        "provider": "openai",
        "model_version": "gpt-5-2025-08-07",
        "vote_index": 0,
        "cost_micros": 100,
        "outcome": "voted",
        "failure_code": "",
    }


def test_provider_vote_recorded_accepts_a_discarded_outcome_with_a_failure_code() -> (
    None
):
    """A discarded vote's `failure_code` is a non-empty wire-format string --
    the only outcome for which `failure_code` is ever non-`""`, matching this
    module's inapplicable-string convention (see `CanaryVerdictRecorded.
    drift_kind`).
    """
    event = ProviderVoteRecorded(
        component="scheduler",
        forecast_id="fc-0002",
        market_ticker="MKT-DEEP",
        provider="anthropic",
        model_version="claude-sonnet-4-5-20250929",
        vote_index=1,
        cost_micros=0,
        outcome="discarded",
        failure_code="malformed_vote_json",
    )

    assert event.payload["outcome"] == "discarded"
    assert event.payload["failure_code"] == "malformed_vote_json"


def test_provider_vote_recorded_non_discard_outcomes_carry_empty_failure_code() -> None:
    """`"voted"` and `"abstained"` outcomes always carry `failure_code=""` --
    only a `"discarded"` outcome carries a real failure code.
    """
    voted = ProviderVoteRecorded(
        component="scheduler",
        forecast_id="fc-0003",
        market_ticker="MKT-DEEP",
        provider="openai",
        model_version="gpt-5-2025-08-07",
        vote_index=0,
        cost_micros=0,
        outcome="voted",
        failure_code="",
    )
    abstained = ProviderVoteRecorded(
        component="scheduler",
        forecast_id="fc-0003",
        market_ticker="MKT-DEEP",
        provider="anthropic",
        model_version="claude-sonnet-4-5-20250929",
        vote_index=1,
        cost_micros=250_000,
        outcome="abstained",
        failure_code="",
    )

    assert voted.payload["failure_code"] == ""
    assert abstained.payload["failure_code"] == ""


def test_provider_vote_recorded_envelope_json_has_json_safe_leaves() -> None:
    """The persisted envelope's `data` carries only int/str leaves, never a
    `float` or a `bool` masquerading as `cost_micros`/`vote_index` -- the
    package-wide no-float convention (SPEC S6.1).
    """
    event = ProviderVoteRecorded(
        component="scheduler",
        forecast_id="fc-0004",
        market_ticker="MKT-DEEP",
        provider="openai",
        model_version="gpt-5-mini-2025-08-07",
        vote_index=2,
        cost_micros=999_999,
        outcome="voted",
        failure_code="",
    )

    envelope = json.loads(event.envelope_json)
    data = envelope["data"]

    assert isinstance(data["forecast_id"], str)
    assert isinstance(data["market_ticker"], str)
    assert isinstance(data["provider"], str)
    assert isinstance(data["model_version"], str)
    assert type(data["vote_index"]) is int
    assert type(data["vote_index"]) is not bool
    assert type(data["cost_micros"]) is int
    assert type(data["cost_micros"]) is not bool
    assert isinstance(data["outcome"], str)
    assert isinstance(data["failure_code"], str)


def test_provider_vote_recorded_is_frozen() -> None:
    """`ProviderVoteRecorded`, like every other concrete event, is immutable."""
    event = ProviderVoteRecorded(
        component="scheduler",
        forecast_id="fc-0005",
        market_ticker="MKT-DEEP",
        provider="openai",
        model_version="gpt-5-2025-08-07",
        vote_index=0,
        cost_micros=0,
        outcome="voted",
        failure_code="",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.outcome = "changed"  # type: ignore[misc]


def test_event_types_registry_round_trips_provider_vote_recorded() -> None:
    """A registry lookup plus persisted `data` reconstructs `ProviderVoteRecorded`."""
    original = ProviderVoteRecorded(
        component="scheduler",
        forecast_id="fc-0006",
        market_ticker="MKT-DEEP",
        provider="futuresearch",
        model_version="fs-2.0",
        vote_index=2,
        cost_micros=500_000,
        outcome="discarded",
        failure_code="provider_timeout",
    )
    envelope = json.loads(original.envelope_json)

    rebuilt_cls = EVENT_TYPES[original.event_type]
    rebuilt = rebuilt_cls(component=envelope["component"], **envelope["data"])

    assert rebuilt == original


def test_provider_vote_recorded_is_reexported_from_windbreak_ledger() -> None:
    """`ProviderVoteRecorded` is importable from `windbreak.ledger` directly,
    not only from `windbreak.ledger.events` (issue #281's own directive,
    matching several -- but not every -- prior event type's re-export
    contract, e.g. `ForecastCreated`).
    """
    import windbreak.ledger as ledger_package

    assert hasattr(ledger_package, "ProviderVoteRecorded"), (
        "ProviderVoteRecorded not re-exported from windbreak.ledger"
    )
    assert ledger_package.ProviderVoteRecorded is ProviderVoteRecorded


# --- Issue #339: ResearchBudgetHalted, the fail-closed research-halt event ---


def test_research_budget_halted_populates_event_type_and_payload() -> None:
    """`ResearchBudgetHalted` derives the full `Event` contract from typed fields.

    Every payload leaf must be int/str -- never a float and never `None` -- so
    the halt stays queryable as typed data rather than as prose buried in a
    message string.
    """
    event = ResearchBudgetHalted(
        component="scheduler",
        market_ticker="",
        halt_kind="per_day",
        utc_day="2024-12-24",
        spent_micros=6_000_000,
        budget_micros=6_000_000,
    )

    assert event.event_type == "ResearchBudgetHalted"
    assert event.component == "scheduler"
    assert event.payload == {
        "market_ticker": "",
        "halt_kind": "per_day",
        "utc_day": "2024-12-24",
        "spent_micros": 6_000_000,
        "budget_micros": 6_000_000,
    }
    assert all(isinstance(leaf, int | str) for leaf in event.payload.values())
    assert not any(isinstance(leaf, float) for leaf in event.payload.values())


def test_research_budget_halted_round_trips_through_the_event_types_registry() -> None:
    """A persisted halt envelope reconstructs to an equal event.

    Without the `EVENT_TYPES` registration this fails only at replay time, in
    production, rather than here.
    """
    event = ResearchBudgetHalted(
        component="scheduler",
        market_ticker="MKT-DEEP",
        halt_kind="per_forecast",
        utc_day="2024-12-24",
        spent_micros=3_000_000,
        budget_micros=2_999_999,
    )

    assert EVENT_TYPES["ResearchBudgetHalted"] is ResearchBudgetHalted
    rebuilt = EVENT_TYPES[event.event_type](
        component=event.component,
        **event.payload,
    )
    assert rebuilt == event
    assert rebuilt.envelope_json == event.envelope_json


def test_research_budget_halted_is_reexported_from_windbreak_ledger() -> None:
    """`ResearchBudgetHalted` is importable from `windbreak.ledger` directly.

    The package's import block and `__all__` are separately hand-maintained
    lists; this pins that they agree for the new event.
    """
    import windbreak.ledger as ledger_package

    assert hasattr(ledger_package, "ResearchBudgetHalted"), (
        "ResearchBudgetHalted not re-exported from windbreak.ledger"
    )
    assert ledger_package.ResearchBudgetHalted is ResearchBudgetHalted


# --- Issue #342: the two liveness-evidence events ---------------------------


def test_exchange_status_observed_round_trips_through_the_registry() -> None:
    """A persisted status observation reconstructs to an equal event."""
    event = ExchangeStatusObserved(
        component="scheduler", status="open", observed_at_epoch_s=1_735_000_000
    )

    assert event.event_type == "ExchangeStatusObserved"
    assert event.payload == {"status": "open", "observed_at_epoch_s": 1_735_000_000}
    assert all(isinstance(leaf, int | str) for leaf in event.payload.values())
    rebuilt = EVENT_TYPES[event.event_type](component=event.component, **event.payload)
    assert rebuilt == event


def test_pipeline_heartbeat_recorded_round_trips_through_the_registry() -> None:
    """A persisted heartbeat reconstructs to an equal event."""
    event = PipelineHeartbeatRecorded(
        component="scheduler", heartbeat_epoch_s=1_735_000_000
    )

    assert event.event_type == "PipelineHeartbeatRecorded"
    assert event.payload == {"heartbeat_epoch_s": 1_735_000_000}
    rebuilt = EVENT_TYPES[event.event_type](component=event.component, **event.payload)
    assert rebuilt == event
