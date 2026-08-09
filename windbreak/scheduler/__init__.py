"""The always-on PAPER-mode composition root (Process orchestration, issue #48).

:mod:`windbreak.scheduler.loop` is the single place the real, unmodified Market
Connector, Forecast Engine, Trade Selector, Risk Kernel, Order Gateway, and
Reconciler are wired together into one PAPER tick following the SPEC S5.3 SINGLE
order path (snapshot -> forecast -> select -> approve -> route -> fill ->
reconcile), appending an audit event to the hash-chained ledger at every stage.

Since issue #345 that path runs once per *screened market* rather than once per
tick: :mod:`windbreak.scheduler.screening` puts the venue's whole market
universe through the real §16 :class:`~windbreak.screener.Screener` at the top
of every tick and hands the loop a bounded, deterministically ordered candidate
set. Screening costs no model calls, so it is affordable every tick; the
candidate bound is what keeps the research spend it authorizes finite.

This package is the *only* legitimate importer of
:mod:`windbreak.connector.paper` outside the Order Gateway: it constructs a
`PaperExchange` -- or, when the operator names a live market, the
`LiveBookPaperExchange` subclass that reads real venue books while still
simulating every fill -- inside `build_paper_deps`'s own
`_build_paper_exchange` helper, and nowhere else. The RESEARCH loop never
imports this package (``windbreak.main`` wires the PAPER tick via a local import
only when PAPER is actually activated), so the paper fake stays off the
RESEARCH/LIVE trading path.
"""

from __future__ import annotations

from windbreak.riskkernel.reservations import ApprovalOutcome
from windbreak.scheduler.loop import (
    ApprovalSeam,
    KernelApproval,
    PaperTickDeps,
    TickOutcome,
    build_evaluation_context,
    build_paper_deps,
    compute_equity_micros,
    is_quote_fresh,
    market_snapshot_event_to_record,
    run_single_tick,
)

__all__ = [
    "ApprovalOutcome",
    "ApprovalSeam",
    "KernelApproval",
    "PaperTickDeps",
    "TickOutcome",
    "build_evaluation_context",
    "build_paper_deps",
    "compute_equity_micros",
    "is_quote_fresh",
    "market_snapshot_event_to_record",
    "run_single_tick",
]
