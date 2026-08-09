"""Tests for the per-provider extension of `CanaryGate` (issue #195, RED).

`windbreak.forecast.canary.CanaryGate` does not yet accept a keyword-only
`provider: str | None = None` on `apply_run`/`acknowledge`/`is_live_blocked`,
nor does it define `apply_version_drift` -- so every test below fails today
with `TypeError: apply_run() got an unexpected keyword argument 'provider'`
(or `AttributeError: 'CanaryGate' object has no attribute
'apply_version_drift'`) -- the expected Gate 1 RED state for issue #195.

Pins the SPEC S8.4/S16 fleet-observability extension: a per-provider drift
dimension layered onto the existing global (pinned-canary-model) dimension,
which must stay byte-identical when `provider=None` (the module docstring's
own backward-compat contract). Reuses `tests/forecast/test_canary.py`'s
`RecordingAlertEmitter` and `_assert_json_safe_leaves` doubles (DRY) rather
than inventing near-duplicates, per this package's own cross-module-import
convention (see e.g. `tests/dashboard/test_app_scheduler_routes.py` importing
from `tests/dashboard/test_app.py`).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.forecast.test_canary import RecordingAlertEmitter, _assert_json_safe_leaves
from windbreak.alerts import AlertType
from windbreak.forecast.canary import (
    CANARY_DRIFT_EVENT,
    CanaryGate,
    CanaryRunResult,
    InMemoryCanaryLedger,
)

#: The two providers most per-provider tests exercise: the issue's own worked
#: example pair (a hosted research forecaster and a direct LLM vote member).
_PROVIDER_A = "futuresearch"
_PROVIDER_B = "anthropic"

_NOW = datetime(2024, 12, 10, 12, 0, 0, tzinfo=UTC)


def _drift_result(*, drift_score_ppm: int, question_id: str = "q1") -> CanaryRunResult:
    """Build a single-question `CanaryRunResult` at a fixed drift score."""
    return CanaryRunResult(
        distances_ppm={question_id: drift_score_ppm},
        drift_score_ppm=drift_score_ppm,
        worst_question_id=question_id,
    )


# --- Backward compatibility: provider=None stays byte-identical --------------


def test_apply_run_global_breach_payload_has_no_provider_or_drift_kind_keys() -> None:
    """A `provider=None` (global) breach payload carries none of the new
    per-provider keys -- the module's own byte-identical backward-compat
    contract for the pinned-canary-model dimension.
    """
    gate = CanaryGate()
    alerts = RecordingAlertEmitter()
    ledger = InMemoryCanaryLedger()

    gate.apply_run(
        _drift_result(drift_score_ppm=100_000),
        checked_at=_NOW,
        alerts=alerts,
        ledger=ledger,
    )

    events = ledger.events_by_type(CANARY_DRIFT_EVENT)
    assert len(events) == 1
    payload = events[0].payload
    assert "provider" not in payload
    assert "drift_kind" not in payload


def test_is_live_blocked_default_provider_none_matches_pre_195_signature() -> None:
    """`is_live_blocked(created_at=...)` with no `provider` kwarg still works,
    unaffected by the new per-provider dimension.
    """
    gate = CanaryGate()
    gate.apply_run(
        _drift_result(drift_score_ppm=100_000),
        checked_at=_NOW,
        alerts=RecordingAlertEmitter(),
        ledger=InMemoryCanaryLedger(),
    )

    assert gate.is_live_blocked(created_at=_NOW) is True


# --- is_live_blocked(provider=...): ORs the provider window with the global --


def test_is_live_blocked_provider_query_blocked_by_its_own_window_only() -> None:
    """A provider-scoped query is blocked when that provider's own window
    blocks, while a clean sibling provider and the (clean) global stay open.
    """
    gate = CanaryGate()
    gate.apply_run(
        _drift_result(drift_score_ppm=100_000),
        checked_at=_NOW,
        alerts=RecordingAlertEmitter(),
        ledger=InMemoryCanaryLedger(),
        provider=_PROVIDER_A,
    )

    assert gate.is_live_blocked(provider=_PROVIDER_A, created_at=_NOW) is True
    assert gate.is_live_blocked(provider=_PROVIDER_B, created_at=_NOW) is False
    assert gate.is_live_blocked(created_at=_NOW) is False


def test_is_live_blocked_provider_query_blocked_by_global_window_fails_closed() -> None:
    """A global (pinned-canary-model) breach blocks EVERY provider query too --
    fail-closed-for-everyone, per the issue's own semantics.
    """
    gate = CanaryGate()
    gate.apply_run(
        _drift_result(drift_score_ppm=100_000),
        checked_at=_NOW,
        alerts=RecordingAlertEmitter(),
        ledger=InMemoryCanaryLedger(),
    )

    assert gate.is_live_blocked(provider=_PROVIDER_A, created_at=_NOW) is True
    assert gate.is_live_blocked(provider=_PROVIDER_B, created_at=_NOW) is True


# --- apply_run(provider=...): answer drift blocks only that provider ---------


def test_apply_run_per_provider_drift_blocks_only_that_provider() -> None:
    """A per-provider answer-drift breach blocks only that provider; a sibling
    provider run at the same instant, within tolerance, stays open.
    """
    gate = CanaryGate(drift_tolerance_ppm=50_000)
    ledger = InMemoryCanaryLedger()

    breached_a = gate.apply_run(
        _drift_result(drift_score_ppm=100_000),
        checked_at=_NOW,
        alerts=RecordingAlertEmitter(),
        ledger=ledger,
        provider=_PROVIDER_A,
    )
    breached_b = gate.apply_run(
        _drift_result(drift_score_ppm=10_000),
        checked_at=_NOW,
        alerts=RecordingAlertEmitter(),
        ledger=ledger,
        provider=_PROVIDER_B,
    )

    assert breached_a is True
    assert breached_b is False
    assert gate.is_live_blocked(provider=_PROVIDER_A, created_at=_NOW) is True
    assert gate.is_live_blocked(provider=_PROVIDER_B, created_at=_NOW) is False


def test_apply_run_per_provider_at_exact_tolerance_is_ok_no_alert_no_block() -> None:
    """The strict `>` tolerance boundary holds per-provider too: exactly at
    tolerance stays within band (mutation-critical, paired with the
    one-over test below).
    """
    gate = CanaryGate(drift_tolerance_ppm=50_000)
    alerts = RecordingAlertEmitter()
    ledger = InMemoryCanaryLedger()

    breached = gate.apply_run(
        _drift_result(drift_score_ppm=50_000),
        checked_at=_NOW,
        alerts=alerts,
        ledger=ledger,
        provider=_PROVIDER_A,
    )

    assert breached is False
    assert alerts.calls == []
    assert gate.is_live_blocked(provider=_PROVIDER_A, created_at=_NOW) is False


def test_apply_run_per_provider_one_ppm_over_tolerance_breaches() -> None:
    """One ppm over tolerance breaches per-provider too (the boundary's other
    side, paired with the exact-tolerance test above).
    """
    gate = CanaryGate(drift_tolerance_ppm=50_000)
    alerts = RecordingAlertEmitter()
    ledger = InMemoryCanaryLedger()

    breached = gate.apply_run(
        _drift_result(drift_score_ppm=50_001),
        checked_at=_NOW,
        alerts=alerts,
        ledger=ledger,
        provider=_PROVIDER_A,
    )

    assert breached is True
    assert len(alerts.calls) == 1
    assert gate.is_live_blocked(provider=_PROVIDER_A, created_at=_NOW) is True


def test_apply_run_per_provider_breach_payload_has_answer_kind_and_provider() -> None:
    """A per-provider answer-drift breach's `CANARY_DRIFT` payload carries
    `provider` and `drift_kind="answer"`, with json-safe leaves throughout.
    """
    gate = CanaryGate()
    ledger = InMemoryCanaryLedger()

    gate.apply_run(
        _drift_result(drift_score_ppm=100_000),
        checked_at=_NOW,
        alerts=RecordingAlertEmitter(),
        ledger=ledger,
        provider=_PROVIDER_A,
    )

    events = ledger.events_by_type(CANARY_DRIFT_EVENT)
    assert len(events) == 1
    payload = events[0].payload
    assert payload["provider"] == _PROVIDER_A
    assert payload["drift_kind"] == "answer"
    _assert_json_safe_leaves(payload)


# --- apply_version_drift: version-pin gate ------------------------------------


def test_apply_version_drift_worked_example_blocks_only_futuresearch() -> None:
    """The issue's own worked example, verbatim: a version-drift observation
    for `futuresearch` (reported `fs-2.1`, pinned `fs-2.0`) blocks
    futuresearch only; `anthropic` stays unblocked.
    """
    gate = CanaryGate()

    breached = gate.apply_version_drift(
        _PROVIDER_A,
        "fs-2.1",
        ("fs-2.0",),
        checked_at=_NOW,
        alerts=RecordingAlertEmitter(),
        ledger=InMemoryCanaryLedger(),
    )

    assert breached is True
    assert gate.is_live_blocked(provider=_PROVIDER_A, created_at=_NOW) is True
    assert gate.is_live_blocked(provider=_PROVIDER_B, created_at=_NOW) is False


def test_apply_version_drift_off_pin_alerts_once_naming_provider_and_kind() -> None:
    """An off-pin version dispatches exactly ONE `CANARY_DRIFT` alert naming
    both the provider and the drift kind.
    """
    gate = CanaryGate()
    alerts = RecordingAlertEmitter()

    gate.apply_version_drift(
        _PROVIDER_A,
        "fs-2.1",
        ("fs-2.0",),
        checked_at=_NOW,
        alerts=alerts,
        ledger=InMemoryCanaryLedger(),
    )

    assert len(alerts.calls) == 1
    alert_type, message = alerts.calls[0]
    assert alert_type is AlertType.CANARY_DRIFT
    assert _PROVIDER_A in message
    assert "version" in message.lower()


def test_apply_version_drift_off_pin_ledgers_one_event_with_provider_and_versions() -> (
    None
):
    """An off-pin version ledgers exactly ONE `CANARY_DRIFT` `CanaryEvent`
    whose payload carries `provider`, `drift_kind="version"`, and the
    reported/pinned version leaves -- all int/str/bool, never a float.
    """
    gate = CanaryGate()
    ledger = InMemoryCanaryLedger()

    gate.apply_version_drift(
        _PROVIDER_A,
        "fs-2.1",
        ("fs-2.0",),
        checked_at=_NOW,
        alerts=RecordingAlertEmitter(),
        ledger=ledger,
    )

    events = ledger.events_by_type(CANARY_DRIFT_EVENT)
    assert len(events) == 1
    payload = events[0].payload
    assert payload["provider"] == _PROVIDER_A
    assert payload["drift_kind"] == "version"
    assert payload["reported_version"] == "fs-2.1"
    assert list(payload["pinned_versions"]) == ["fs-2.0"]
    _assert_json_safe_leaves(payload)


def test_apply_version_drift_on_pin_version_is_a_no_op() -> None:
    """A reported version that IS in the pinned set is a no-op: no alert, no
    ledgered event, no block.
    """
    gate = CanaryGate()
    alerts = RecordingAlertEmitter()
    ledger = InMemoryCanaryLedger()

    breached = gate.apply_version_drift(
        _PROVIDER_A,
        "fs-2.0",
        ("fs-2.0", "fs-2.1"),
        checked_at=_NOW,
        alerts=alerts,
        ledger=ledger,
    )

    assert breached is False
    assert alerts.calls == []
    assert ledger.events_by_type(CANARY_DRIFT_EVENT) == ()
    assert gate.is_live_blocked(provider=_PROVIDER_A, created_at=_NOW) is False


# --- acknowledge(provider=...): restores only that provider's NEW records ----


def test_acknowledge_per_provider_restores_only_that_providers_new_records() -> None:
    """Acking one provider's drift restores eligibility only for that
    provider's records created at/after the ack; a still-blocked sibling
    provider is unaffected, and the acked provider's own earlier-blocked
    record stays blocked.
    """
    drift_at = datetime(2024, 12, 10, 0, 0, tzinfo=UTC)
    blocked_at = datetime(2024, 12, 10, 1, 0, tzinfo=UTC)
    ack_at = datetime(2024, 12, 10, 2, 0, tzinfo=UTC)
    restored_at = datetime(2024, 12, 10, 3, 0, tzinfo=UTC)
    gate = CanaryGate()
    ledger = InMemoryCanaryLedger()

    gate.apply_run(
        _drift_result(drift_score_ppm=100_000),
        checked_at=drift_at,
        alerts=RecordingAlertEmitter(),
        ledger=ledger,
        provider=_PROVIDER_A,
    )
    gate.apply_run(
        _drift_result(drift_score_ppm=100_000),
        checked_at=drift_at,
        alerts=RecordingAlertEmitter(),
        ledger=ledger,
        provider=_PROVIDER_B,
    )

    assert gate.is_live_blocked(provider=_PROVIDER_A, created_at=blocked_at) is True
    assert gate.is_live_blocked(provider=_PROVIDER_B, created_at=blocked_at) is True

    gate.acknowledge(provider=_PROVIDER_A, acked_at=ack_at, ledger=ledger)

    # The acked provider's earlier-blocked record stays blocked.
    assert gate.is_live_blocked(provider=_PROVIDER_A, created_at=blocked_at) is True
    # A new record for the acked provider, created after the ack, is restored.
    assert gate.is_live_blocked(provider=_PROVIDER_A, created_at=restored_at) is False
    # The still-unacknowledged sibling provider is entirely unaffected.
    assert gate.is_live_blocked(provider=_PROVIDER_B, created_at=restored_at) is True


def test_acknowledge_per_provider_does_not_touch_the_global_window() -> None:
    """Acking one provider's drift never touches the independent global
    (pinned-canary-model) window -- the two dimensions are orthogonal.
    """
    gate = CanaryGate()
    ledger = InMemoryCanaryLedger()

    gate.apply_run(
        _drift_result(drift_score_ppm=100_000),
        checked_at=_NOW,
        alerts=RecordingAlertEmitter(),
        ledger=ledger,
    )
    gate.apply_run(
        _drift_result(drift_score_ppm=100_000),
        checked_at=_NOW,
        alerts=RecordingAlertEmitter(),
        ledger=ledger,
        provider=_PROVIDER_A,
    )

    gate.acknowledge(provider=_PROVIDER_A, acked_at=_NOW, ledger=ledger)

    # The global window is still drifted -- acking one provider never
    # acknowledges the shared global dimension.
    assert gate.is_live_blocked(created_at=_NOW) is True


def test_acknowledge_per_provider_with_no_active_drift_for_that_provider_raises() -> (
    None
):
    """Acking a provider with no active drift is a usage error, exactly like
    the pre-existing global `acknowledge` contract.
    """
    gate = CanaryGate()

    with pytest.raises(ValueError):
        gate.acknowledge(
            provider=_PROVIDER_A, acked_at=_NOW, ledger=InMemoryCanaryLedger()
        )
