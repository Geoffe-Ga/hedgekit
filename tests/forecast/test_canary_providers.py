"""Tests for `windbreak.forecast.canary_providers` (issue #195, RED).

`windbreak/forecast/canary_providers.py` does not exist yet, so every import
below fails collection with `ModuleNotFoundError: No module named
'windbreak.forecast.canary_providers'` -- the expected Gate 1 RED state for
issue #195.

Pins the provider-canary battery runner that composes the per-provider
`CanaryGate` extension (`tests/forecast/test_canary_provider_gate.py`) with a
small, injected `ProviderCanaryObserver` per spec: `run_provider_canaries`
gathers one observation per provider (answer drift via the existing
`score_canary_run`/`apply_run` seam, version drift via `apply_version_drift`),
and reduces the battery to one `ProviderCanaryVerdict` per provider. Entirely
offline: two small local fake observers stand in for a real hosted-forecaster
or LLM-vote transport, reusing `tests/forecast/test_canary.py`'s
`RecordingAlertEmitter`/`_assert_json_safe_leaves` doubles (DRY) rather than
inventing near-duplicates.
"""

from __future__ import annotations

from datetime import UTC, datetime

from tests.forecast.test_canary import RecordingAlertEmitter, _assert_json_safe_leaves
from windbreak.alerts import AlertType
from windbreak.forecast.canary import (
    CANARY_DRIFT_EVENT,
    CANARY_OK_EVENT,
    CanaryGate,
    CanaryQuestion,
    InMemoryCanaryLedger,
)
from windbreak.forecast.providers.base import ProviderVersionDriftError

_NOW = datetime(2024, 12, 10, 12, 0, 0, tzinfo=UTC)

#: A 64-hex-char sentinel fingerprint for the raising observer's drifted
#: response -- never a real hash, mirrors `test_canary.py`'s own sample values.
_SENTINEL_FINGERPRINT = "f" * 64

_QUESTION = CanaryQuestion(question_id="q1", prompt="p1", reference_ppm=500_000)


class _FixedObserver:
    """A `ProviderCanaryObserver` double returning one fixed observation."""

    def __init__(self, observed_ppm: dict[str, int], reported_version: str) -> None:
        """Store the fixed observation this double always returns.

        Args:
            observed_ppm: The per-question observed ppm to return.
            reported_version: The forecaster version to report.
        """
        self._observed_ppm = observed_ppm
        self._reported_version = reported_version

    def observe(self, spec: object) -> object:
        """Return the fixed observation, ignoring `spec`'s contents.

        Args:
            spec: The (unused) `ProviderCanarySpec` being observed.

        Returns:
            The fixed `ProviderCanaryObservation`.
        """
        del spec
        from windbreak.forecast.canary_providers import ProviderCanaryObservation

        return ProviderCanaryObservation(
            observed_ppm=self._observed_ppm, reported_version=self._reported_version
        )


class _RaisingVersionDriftObserver:
    """A `ProviderCanaryObserver` double that always fails closed on drift."""

    def __init__(self, reported_version: str, pinned_versions: tuple[str, ...]) -> None:
        """Store the drifted version this double always raises for.

        Args:
            reported_version: The off-pin version to report.
            pinned_versions: The pinned set the report drifted from.
        """
        self._reported_version = reported_version
        self._pinned_versions = pinned_versions

    def observe(self, spec: object) -> object:
        """Always raise `ProviderVersionDriftError`, ignoring `spec`.

        Args:
            spec: The (unused) `ProviderCanarySpec` being observed.

        Raises:
            ProviderVersionDriftError: Unconditionally.
        """
        del spec
        raise ProviderVersionDriftError(
            self._reported_version, self._pinned_versions, _SENTINEL_FINGERPRINT
        )


def _make_spec(
    *,
    provider: str,
    pinned_versions: tuple[str, ...],
    observer: object,
    questions: tuple[CanaryQuestion, ...] = (_QUESTION,),
) -> object:
    """Build one `ProviderCanarySpec`, deferring the import to call time.

    Args:
        provider: The provider identifier.
        pinned_versions: The provider's pinned version set.
        observer: The `ProviderCanaryObserver` this spec is paired with.
        questions: The canary questions this provider is checked against.

    Returns:
        The constructed `ProviderCanarySpec`.
    """
    from windbreak.forecast.canary_providers import ProviderCanarySpec

    return ProviderCanarySpec(
        provider=provider,
        questions=questions,
        pinned_versions=pinned_versions,
        observer=observer,
    )


def test_module_imports_cleanly_offline() -> None:
    """`windbreak.forecast.canary_providers` imports with no network access."""
    import windbreak.forecast.canary_providers  # noqa: F401


# --- The issue's own worked example, verbatim ---------------------------------


def test_worked_example_version_drift_blocks_only_futuresearch() -> None:
    """A version-drift observation for `futuresearch` (reported `fs-2.1`,
    pinned `fs-2.0`) blocks futuresearch only; `anthropic` (clean) stays
    unblocked.
    """
    from windbreak.forecast.canary_providers import (
        ProviderCanaryStatus,
        run_provider_canaries,
    )

    futuresearch_spec = _make_spec(
        provider="futuresearch",
        pinned_versions=("fs-2.0",),
        observer=_RaisingVersionDriftObserver("fs-2.1", ("fs-2.0",)),
    )
    anthropic_spec = _make_spec(
        provider="anthropic",
        pinned_versions=("claude-1",),
        observer=_FixedObserver({"q1": 500_000}, "claude-1"),
    )
    gate = CanaryGate()

    verdicts = run_provider_canaries(
        (futuresearch_spec, anthropic_spec),
        gate=gate,
        alerts=RecordingAlertEmitter(),
        ledger=InMemoryCanaryLedger(),
        checked_at=_NOW,
    )

    assert gate.is_live_blocked(provider="futuresearch", created_at=_NOW) is True
    assert not gate.is_live_blocked(provider="anthropic", created_at=_NOW)
    by_provider = {verdict.provider: verdict for verdict in verdicts}
    assert by_provider["futuresearch"].status is ProviderCanaryStatus.VERSION_DRIFT
    assert by_provider["anthropic"].status is ProviderCanaryStatus.OK


# --- Answer drift: blocks only the drifted provider ---------------------------


def test_answer_drift_for_one_provider_blocks_only_that_provider() -> None:
    """Answer drift for provider A (drifted observed ppm) blocks A; provider B
    (on-reference) stays open; verdict statuses returned correctly.
    """
    from windbreak.forecast.canary_providers import (
        ProviderCanaryStatus,
        run_provider_canaries,
    )

    drifted_spec = _make_spec(
        provider="openai",
        pinned_versions=("v1",),
        observer=_FixedObserver({"q1": 600_000}, "v1"),
    )
    clean_spec = _make_spec(
        provider="anthropic",
        pinned_versions=("v1",),
        observer=_FixedObserver({"q1": 500_000}, "v1"),
    )
    gate = CanaryGate(drift_tolerance_ppm=50_000)

    verdicts = run_provider_canaries(
        (drifted_spec, clean_spec),
        gate=gate,
        alerts=RecordingAlertEmitter(),
        ledger=InMemoryCanaryLedger(),
        checked_at=_NOW,
    )

    by_provider = {verdict.provider: verdict for verdict in verdicts}
    assert by_provider["openai"].status is ProviderCanaryStatus.ANSWER_DRIFT
    assert by_provider["anthropic"].status is ProviderCanaryStatus.OK
    assert gate.is_live_blocked(provider="openai", created_at=_NOW) is True
    assert not gate.is_live_blocked(provider="anthropic", created_at=_NOW)


# --- A raising observer is treated as VERSION_DRIFT, fail-closed -------------


def test_observer_raising_version_drift_error_is_fail_closed_version_drift() -> None:
    """An observer raising `ProviderVersionDriftError` is treated as
    `VERSION_DRIFT`, blocking that provider without ever attempting to score
    answer drift over a (nonexistent) observation.
    """
    from windbreak.forecast.canary_providers import (
        ProviderCanaryStatus,
        run_provider_canaries,
    )

    spec = _make_spec(
        provider="futuresearch",
        pinned_versions=("fs-2.0",),
        observer=_RaisingVersionDriftObserver("fs-9.9", ("fs-2.0",)),
    )
    gate = CanaryGate()

    verdicts = run_provider_canaries(
        (spec,),
        gate=gate,
        alerts=RecordingAlertEmitter(),
        ledger=InMemoryCanaryLedger(),
        checked_at=_NOW,
    )

    assert len(verdicts) == 1
    assert verdicts[0].status is ProviderCanaryStatus.VERSION_DRIFT
    assert gate.is_live_blocked(provider="futuresearch", created_at=_NOW) is True


# --- All-OK: one CANARY_OK per provider, zero alerts --------------------------


def test_all_ok_run_ledgers_one_canary_ok_per_provider_and_dispatches_no_alerts() -> (
    None
):
    """An all-clean run over two providers ledgers a per-provider `CANARY_OK`
    for each and returns all-`OK` verdicts, with zero alerts dispatched.
    """
    from windbreak.forecast.canary_providers import (
        ProviderCanaryStatus,
        run_provider_canaries,
    )

    spec_a = _make_spec(
        provider="p1",
        pinned_versions=("v1",),
        observer=_FixedObserver({"q1": 500_000}, "v1"),
    )
    spec_b = _make_spec(
        provider="p2",
        pinned_versions=("v1",),
        observer=_FixedObserver({"q1": 500_000}, "v1"),
    )
    alerts = RecordingAlertEmitter()
    ledger = InMemoryCanaryLedger()

    verdicts = run_provider_canaries(
        (spec_a, spec_b),
        gate=CanaryGate(),
        alerts=alerts,
        ledger=ledger,
        checked_at=_NOW,
    )

    assert all(verdict.status is ProviderCanaryStatus.OK for verdict in verdicts)
    assert alerts.calls == []
    assert len(ledger.events_by_type(CANARY_OK_EVENT)) == 2


# --- Payload/alert shape: json-safe leaves, one alert per drifting provider ---


def test_drifting_providers_each_alert_once_naming_provider_and_kind() -> None:
    """Every drifting provider (one answer-drift, one version-drift) dispatches
    exactly one `CANARY_DRIFT` alert naming its provider and drift kind, and
    every ledgered `CANARY_DRIFT` payload's leaves are json-safe.
    """
    from windbreak.forecast.canary_providers import run_provider_canaries

    answer_drift_spec = _make_spec(
        provider="openai",
        pinned_versions=("v1",),
        observer=_FixedObserver({"q1": 999_000}, "v1"),
    )
    version_drift_spec = _make_spec(
        provider="futuresearch",
        pinned_versions=("fs-2.0",),
        observer=_RaisingVersionDriftObserver("fs-2.1", ("fs-2.0",)),
    )
    alerts = RecordingAlertEmitter()
    ledger = InMemoryCanaryLedger()

    run_provider_canaries(
        (answer_drift_spec, version_drift_spec),
        gate=CanaryGate(drift_tolerance_ppm=50_000),
        alerts=alerts,
        ledger=ledger,
        checked_at=_NOW,
    )

    assert len(alerts.calls) == 2
    for alert_type, _message in alerts.calls:
        assert alert_type is AlertType.CANARY_DRIFT
    joined_messages = " ".join(message for _type, message in alerts.calls)
    assert "openai" in joined_messages
    assert "futuresearch" in joined_messages

    drift_events = ledger.events_by_type(CANARY_DRIFT_EVENT)
    assert len(drift_events) == 2
    for event in drift_events:
        _assert_json_safe_leaves(event.payload)
        assert "provider" in event.payload
        assert "drift_kind" in event.payload
