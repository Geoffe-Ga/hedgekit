"""The provider-canary battery composition root (fleet observability, #195).

This is the single place the forecast-layer provider canary battery
(:func:`windbreak.forecast.canary_providers.run_provider_canaries`) is bridged
to the durable, hash-chained operator ledger. As the composition root it
legally imports both the forecast package (which, per the SPEC S8.3 sandbox
boundary, may not import :mod:`windbreak.ledger`) and the ledger package: it
runs the battery, appends one :class:`~windbreak.ledger.events.CanaryVerdictRecorded`
per verdict to a real :class:`~windbreak.ledger.store.SqliteLedgerStore` (keeping
the chain verifiable), prints one pinned operator line per verdict, and returns
a shell exit code (``0`` all-OK, ``1`` on any drift) so
``scripts/run-canaries.sh`` exits non-zero the moment any provider drifts.

Every ledgered leaf is an int/str/bool/list, never a float (SPEC S6.1); this
module is on ``scripts/lint_no_floats.py``'s denylist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from windbreak.forecast.canary import (
    DEFAULT_CANARY_DRIFT_TOLERANCE_PPM,
    CanaryGate,
    InMemoryCanaryLedger,
)
from windbreak.forecast.canary_providers import (
    ProviderCanaryStatus,
    run_provider_canaries,
)
from windbreak.ledger.events import CanaryVerdictRecorded
from windbreak.ledger.store import SqliteLedgerStore

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path
    from typing import TextIO

    from windbreak.forecast.canary import CanaryAlertEmitter
    from windbreak.forecast.canary_providers import (
        ProviderCanarySpec,
        ProviderCanaryVerdict,
    )

#: The ledger ``component`` stamped on every verdict this composition root
#: appends.
_COMPONENT = "scheduler"

#: The ``drift_kind`` payload leaf each verdict status maps to (``""`` for a
#: clean ``OK``, matching :class:`~windbreak.ledger.events.CanaryVerdictRecorded`'s
#: inapplicable-string convention).
_DRIFT_KIND_BY_STATUS: dict[ProviderCanaryStatus, str] = {
    ProviderCanaryStatus.OK: "",
    ProviderCanaryStatus.ANSWER_DRIFT: "answer",
    ProviderCanaryStatus.VERSION_DRIFT: "version",
}

#: Shell exit code returned when every provider stays within band.
_EXIT_ALL_OK = 0

#: Shell exit code returned when at least one provider drifts.
_EXIT_DRIFT = 1


def _verdict_event(
    spec: ProviderCanarySpec, verdict: ProviderCanaryVerdict
) -> CanaryVerdictRecorded:
    """Build the ``CanaryVerdictRecorded`` event for one verdict.

    The pinned version set is read off the ``spec`` (a verdict carries only the
    single reported version), and the drift tolerance is the module default.

    Args:
        spec: The provider spec the verdict was produced from.
        verdict: The reduced provider canary verdict.

    Returns:
        The typed ledger event to append.
    """
    return CanaryVerdictRecorded(
        component=_COMPONENT,
        provider=verdict.provider,
        status=verdict.status.name,
        drift_kind=_DRIFT_KIND_BY_STATUS[verdict.status],
        drift_score_ppm=verdict.drift_score_ppm,
        tolerance_ppm=DEFAULT_CANARY_DRIFT_TOLERANCE_PPM,
        reported_version=verdict.reported_version,
        pinned_versions=list(spec.pinned_versions),
    )


def _operator_line(verdict: ProviderCanaryVerdict) -> str:
    """Render one verdict's pinned operator line.

    Args:
        verdict: The reduced provider canary verdict.

    Returns:
        A ``provider=<p> canary=<STATUS> drift_score_ppm=<n>`` line.
    """
    return (
        f"provider={verdict.provider} canary={verdict.status.name} "
        f"drift_score_ppm={verdict.drift_score_ppm}"
    )


def run_canaries(
    specs: tuple[ProviderCanarySpec, ...],
    *,
    ledger_path: Path,
    alerts: CanaryAlertEmitter,
    output: TextIO,
    checked_at: datetime,
    gate: CanaryGate | None = None,
) -> int:
    """Run a provider canary battery and durably ledger every verdict (#195).

    Runs :func:`~windbreak.forecast.canary_providers.run_provider_canaries` over
    ``specs``, appends one
    :class:`~windbreak.ledger.events.CanaryVerdictRecorded` per verdict to a real
    :class:`~windbreak.ledger.store.SqliteLedgerStore` at ``ledger_path``
    (re-verifying the chain), and prints one operator line per verdict to
    ``output``.

    Args:
        specs: The provider canary specs to run, one per provider.
        ledger_path: The SQLite ledger the verdicts are appended to
            (keyword-only).
        alerts: The alert emitter drift breaches dispatch through (keyword-only).
        output: The text stream operator lines are printed to (keyword-only).
        checked_at: When the battery was checked (keyword-only).
        gate: The canary gate to drive, or ``None`` to build a fresh one
            (keyword-only).

    Returns:
        ``0`` when every provider stayed within band, else ``1``.
    """
    active_gate = CanaryGate() if gate is None else gate
    verdicts = run_provider_canaries(
        specs,
        gate=active_gate,
        alerts=alerts,
        ledger=InMemoryCanaryLedger(),
        checked_at=checked_at,
    )
    store = SqliteLedgerStore(ledger_path)
    try:
        for spec, verdict in zip(specs, verdicts, strict=True):
            store.append(_verdict_event(spec, verdict))
        store.verify_chain()
    finally:
        store.close()
    for verdict in verdicts:
        print(_operator_line(verdict), file=output)
    if all(verdict.status is ProviderCanaryStatus.OK for verdict in verdicts):
        return _EXIT_ALL_OK
    return _EXIT_DRIFT
