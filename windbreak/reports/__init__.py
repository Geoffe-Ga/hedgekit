"""Weekly operator-report generation for the always-on PAPER loop (issue #48).

The PAPER loop writes a dated, per-ISO-week markdown report stub so an operator
always has a current, human-readable summary file on disk. The equity, position,
and decision section bodies are rendered by :mod:`windbreak.reports.sections`
(issue #255) from the ledger read models; a section falls back to a
``No data yet.`` placeholder only when its body is not supplied.
"""

from __future__ import annotations

from windbreak.reports.weekly import maybe_write_weekly, write_weekly_stub

__all__ = ["maybe_write_weekly", "write_weekly_stub"]
