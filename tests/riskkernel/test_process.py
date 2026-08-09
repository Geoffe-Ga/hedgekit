"""Failing-first test pinning `RiskKernel.from_events(..., verifier=...)`
(issue #236, RED).

`windbreak run --process riskkernel` composes its `KillIntegration`'s
`ReconciliationMismatchMonitor` from `RiskConfig.kill_after_consecutive_mismatches`,
but `windbreak/main.py::_build_risk_kernel` wires no `ReadOnlyVerifier` (see
`windbreak/riskkernel/kill.py`'s module docstring): the kernel's per-beat
`RiskKernel.run_verification_cycle` is thus a permanent no-op in production,
and the `AUTO_RECONCILIATION` auto-kill trigger is composed-but-dormant.
`RiskKernel.__init__` already accepts a keyword-only `verifier`
(`windbreak/riskkernel/process.py:280`), but `RiskKernel.from_events` -- the
entrypoint `_build_risk_kernel` actually calls to rebuild a kernel over
replayed history -- does not forward one, so there is today no way to hand a
live verifier through the real composition path at all.

`from_events` does not yet accept a `verifier` keyword, so the call below
fails with `TypeError: from_events() got an unexpected keyword argument
'verifier'` -- the expected Gate 1 RED state for issue #236.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from windbreak.numeric.types import MoneyMicros
from windbreak.riskkernel.process import InMemoryKernelLedgerWriter, RiskKernel
from windbreak.riskkernel.verification import VerificationOutcome, VerificationSnapshot


@dataclass
class _RecordingCleanVerifier:
    """A stub `ReadOnlyVerifier` that always returns a fixed CLEAN snapshot.

    Records every `run_cycle` call, so a test can assert `from_events`'s
    `verifier=` keyword was actually threaded through to the rebuilt kernel's
    per-beat verification cycle rather than silently discarded.

    Attributes:
        calls: The `now_epoch_s` argument of every `run_cycle` call, in order.
    """

    calls: list[int] = field(default_factory=list)

    def run_cycle(self, now_epoch_s: int) -> VerificationSnapshot:
        """Record the call and return a fixed, permissive CLEAN snapshot.

        Args:
            now_epoch_s: The epoch second the kernel's clock supplied.

        Returns:
            A `VerificationSnapshot` with `outcome=CLEAN` and zero drift.
        """
        self.calls.append(now_epoch_s)
        return VerificationSnapshot(
            outcome=VerificationOutcome.CLEAN,
            balance_ok=True,
            position_ok=True,
            open_order_ok=True,
            verified_at_epoch_s=now_epoch_s,
            exchange_verified_available_cash=MoneyMicros(0),
            cash_drift=MoneyMicros(0),
            semantics_fully_known=True,
        )


def test_from_events_forwards_verifier_to_the_rebuilt_kernels_beat_loop() -> None:
    """`RiskKernel.from_events(..., verifier=stub)` runs `stub` each beat.

    Today `from_events` has no `verifier` parameter at all, so this call
    fails with `TypeError: from_events() got an unexpected keyword argument
    'verifier'` -- proving the classmethod, unlike `__init__`, does not yet
    accept or forward one.
    """
    writer = InMemoryKernelLedgerWriter()
    stub = _RecordingCleanVerifier()

    kernel = RiskKernel.from_events((), writer, verifier=stub)
    kernel.run(max_beats=1, heartbeat_interval=0)

    assert len(stub.calls) == 1
