"""Tests for `windbreak.scheduler.canaries` (issue #195, RED).

`windbreak/scheduler/canaries.py` does not exist yet, so every import below
fails collection with `ModuleNotFoundError: No module named
'windbreak.scheduler.canaries'` -- the expected Gate 1 RED state for issue
#195.

`run_canaries` is the composition root `scripts/run-canaries.sh` (the
operator entry point, non-zero exit on any drift) drives: it runs
`windbreak.forecast.canary_providers.run_provider_canaries` over the supplied
battery, appends one `CanaryVerdictRecorded` per verdict to a real
`SqliteLedgerStore`, prints one pinned operator line per verdict to the
supplied output stream, and returns an int exit code (0 all-OK, 1 on any
drift). Mirrors `windbreak.scheduler.loop`'s own composition-root shape (a
plain function over a real `SqliteLedgerStore`, no framework), and
`tests/ledger/conftest.py`'s tmp-path-backed `SqliteLedgerStore` pattern
(this package's own `tests/scheduler/conftest.py` has no ledger fixture, so
this module builds its store directly, mirroring
`tests/ledger/test_canary_rebuild.py`'s own direct-construction precedent).
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests.forecast.test_canary import RecordingAlertEmitter
from windbreak.forecast.providers.base import ProviderVersionDriftError

if TYPE_CHECKING:
    from pathlib import Path

_NOW = datetime(2024, 12, 10, 12, 0, 0, tzinfo=UTC)

_SENTINEL_FINGERPRINT = "f" * 64


class _FixedObserver:
    """A `ProviderCanaryObserver` double returning one fixed observation."""

    def __init__(self, observed_ppm: dict[str, int], reported_version: str) -> None:
        """Store the fixed observation this double always returns."""
        self._observed_ppm = observed_ppm
        self._reported_version = reported_version

    def observe(self, spec: object) -> object:
        """Return the fixed observation, ignoring `spec`'s contents."""
        del spec
        from windbreak.forecast.canary_providers import ProviderCanaryObservation

        return ProviderCanaryObservation(
            observed_ppm=self._observed_ppm, reported_version=self._reported_version
        )


class _RaisingVersionDriftObserver:
    """A `ProviderCanaryObserver` double that always fails closed on drift."""

    def __init__(self, reported_version: str, pinned_versions: tuple[str, ...]) -> None:
        """Store the drifted version this double always raises for."""
        self._reported_version = reported_version
        self._pinned_versions = pinned_versions

    def observe(self, spec: object) -> object:
        """Always raise `ProviderVersionDriftError`, ignoring `spec`."""
        del spec
        raise ProviderVersionDriftError(
            self._reported_version, self._pinned_versions, _SENTINEL_FINGERPRINT
        )


def _make_spec(*, provider: str, pinned_versions: tuple[str, ...], observer: object):
    """Build one `ProviderCanarySpec` over a single reference question."""
    from windbreak.forecast.canary import CanaryQuestion
    from windbreak.forecast.canary_providers import ProviderCanarySpec

    question = CanaryQuestion(question_id="q1", prompt="p1", reference_ppm=500_000)
    return ProviderCanarySpec(
        provider=provider,
        questions=(question,),
        pinned_versions=pinned_versions,
        observer=observer,
    )


def _all_ok_specs() -> tuple[object, ...]:
    """Build two clean (non-drifting) provider specs."""
    return (
        _make_spec(
            provider="openai",
            pinned_versions=("v1",),
            observer=_FixedObserver({"q1": 500_000}, "v1"),
        ),
        _make_spec(
            provider="anthropic",
            pinned_versions=("v1",),
            observer=_FixedObserver({"q1": 500_000}, "v1"),
        ),
    )


def _one_drifting_spec_pair() -> tuple[object, ...]:
    """Build one clean provider spec and one version-drifting provider spec."""
    return (
        _make_spec(
            provider="anthropic",
            pinned_versions=("v1",),
            observer=_FixedObserver({"q1": 500_000}, "v1"),
        ),
        _make_spec(
            provider="futuresearch",
            pinned_versions=("fs-2.0",),
            observer=_RaisingVersionDriftObserver("fs-2.1", ("fs-2.0",)),
        ),
    )


def test_run_canaries_exit_code_zero_when_all_ok(tmp_path: Path) -> None:
    """An all-clean battery run exits `0`."""
    from windbreak.scheduler.canaries import run_canaries

    exit_code = run_canaries(
        _all_ok_specs(),
        ledger_path=tmp_path / "ledger.db",
        alerts=RecordingAlertEmitter(),
        output=io.StringIO(),
        checked_at=_NOW,
    )

    assert exit_code == 0


def test_run_canaries_exit_code_one_when_any_provider_drifts(tmp_path: Path) -> None:
    """A battery run with at least one drifting provider exits `1`."""
    from windbreak.scheduler.canaries import run_canaries

    exit_code = run_canaries(
        _one_drifting_spec_pair(),
        ledger_path=tmp_path / "ledger.db",
        alerts=RecordingAlertEmitter(),
        output=io.StringIO(),
        checked_at=_NOW,
    )

    assert exit_code == 1


def test_run_canaries_appends_one_canary_verdict_recorded_per_provider(
    tmp_path: Path,
) -> None:
    """Exactly one `CanaryVerdictRecorded` is appended per provider, and the
    ledger's hash chain still verifies afterward.
    """
    from windbreak.ledger.store import SqliteLedgerStore
    from windbreak.scheduler.canaries import run_canaries

    ledger_path = tmp_path / "ledger.db"

    run_canaries(
        _all_ok_specs(),
        ledger_path=ledger_path,
        alerts=RecordingAlertEmitter(),
        output=io.StringIO(),
        checked_at=_NOW,
    )

    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()

    verdict_records = [r for r in records if r.event_type == "CanaryVerdictRecorded"]
    assert len(verdict_records) == 2
    providers = {
        json.loads(r.payload_json)["data"]["provider"] for r in verdict_records
    }
    assert providers == {"openai", "anthropic"}


def test_run_canaries_operator_line_format_is_pinned(tmp_path: Path) -> None:
    """Each verdict prints one `provider=<p> canary=<STATUS>
    drift_score_ppm=<n>` operator line to the output stream.
    """
    from windbreak.scheduler.canaries import run_canaries

    output = io.StringIO()

    run_canaries(
        _all_ok_specs(),
        ledger_path=tmp_path / "ledger.db",
        alerts=RecordingAlertEmitter(),
        output=output,
        checked_at=_NOW,
    )

    printed = output.getvalue()
    assert "provider=openai canary=OK drift_score_ppm=0" in printed
    assert "provider=anthropic canary=OK drift_score_ppm=0" in printed


def test_run_canaries_drifting_provider_reflected_in_canary_status_read_model(
    tmp_path: Path,
) -> None:
    """A drifting provider's verdict is visible in the subsequent
    `canary_status_read_model` fold over the same ledger.
    """
    from windbreak.ledger.rebuild import canary_status_read_model
    from windbreak.ledger.store import SqliteLedgerStore
    from windbreak.scheduler.canaries import run_canaries

    ledger_path = tmp_path / "ledger.db"

    run_canaries(
        _one_drifting_spec_pair(),
        ledger_path=ledger_path,
        alerts=RecordingAlertEmitter(),
        output=io.StringIO(),
        checked_at=_NOW,
    )

    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()

    rows = canary_status_read_model(records)
    by_provider = {row["data"]["provider"]: row["data"] for row in rows}
    assert by_provider["futuresearch"]["status"] == "VERSION_DRIFT"
    assert by_provider["anthropic"]["status"] == "OK"
