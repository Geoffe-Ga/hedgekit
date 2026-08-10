"""A `store.append` failure mid-tick must not terminate the process (#443).

Issue #443's second acceptance criterion asks for an integration test, not a
unit double: a *real* `run_single_tick` over the shared offline fixtures, whose
ledger volume fills partway through the tick, driven by the *real*
`_build_paper_on_beat` hook through the *real* `run_loop`. Only the disk is
faked, by the one wrapper that turns `EquitySampled`'s append -- the exact call
in the issue's traceback (`loop.py:3038` -> `store.py:420`) -- into the
`sqlite3.OperationalError` a full volume raises.

Before the fix that raise unwound out of `run_loop` and killed the daemon after
the first beat. The assertions below are the whole survival contract: beat 2
still runs, the failure is alerted exactly once and ledgered, the heartbeat line
stops claiming RESEARCH, and the hash chain the tick had already written
survives -- a liveness failure, never a corruption one.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sqlite3
from typing import TYPE_CHECKING

from tests.integration.conftest import FIXED_NOW_EPOCH_S, ledger_path_for
from windbreak.alerts import AlertDispatcher
from windbreak.ledger import SqliteLedgerStore
from windbreak.main import (
    BeatSupervisor,
    LedgerAlertWriter,
    _build_paper_on_beat,
    run_loop,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

    from windbreak.config.schema import WindbreakConfig
    from windbreak.ledger import Event
    from windbreak.ledger.store import LedgerRecord, LedgerStore

#: The event whose append the fake full volume refuses. `_equity_and_positions_stage`
#: appends it after the tick's `ModeHeartbeat`, so the tick is genuinely partway
#: through when the disk gives out -- the shape of issue #443's traceback.
_FULL_AT_EVENT_TYPE = "EquitySampled"


class _FullVolumeStore:
    """A `LedgerStore` delegating to a real one until the volume "fills".

    Attributes:
        refused: Every event type whose append was refused, in order.
    """

    def __init__(self, inner: LedgerStore, *, refuse: str) -> None:
        """Wrap a real store, refusing one event type's appends.

        Args:
            inner: The real store every other append is delegated to.
            refuse: The `event_type` whose append raises instead.
        """
        self._inner = inner
        self._refuse = refuse
        self.refused: list[str] = []

    def append(self, event: Event) -> int:
        """Append via the real store, or raise as a full volume does.

        Args:
            event: The event to persist.

        Returns:
            The sequence number the real store assigned.

        Raises:
            sqlite3.OperationalError: When ``event`` is the refused type.
        """
        if event.event_type == self._refuse:
            self.refused.append(event.event_type)
            raise sqlite3.OperationalError("database or disk is full")
        return self._inner.append(event)

    def read_all(self) -> list[LedgerRecord]:
        """Read every record from the real store.

        Returns:
            The real store's records, in ledger order.
        """
        return self._inner.read_all()

    def verify_chain(self) -> None:
        """Verify the real store's hash chain."""
        self._inner.verify_chain()

    def close(self) -> None:
        """Close the real store."""
        self._inner.close()


def _paper_args(tmp_path: Path, ledger_path: Path) -> argparse.Namespace:
    """Build the `run` namespace `_build_paper_on_beat` reads its flags from.

    Args:
        tmp_path: The per-test scratch directory.
        ledger_path: The tick ledger's path.

    Returns:
        A namespace carrying the four PAPER flags and no live ticker.
    """
    return argparse.Namespace(
        paper_books_dir=tmp_path / "books",
        cassette_path=tmp_path / "cassette.json",
        ledger_path=ledger_path,
        report_dir=tmp_path / "reports",
        paper_live_ticker=None,
    )


def _fixed_clock() -> int:
    """Return the suite's fixed epoch second, so a tick is deterministic.

    Returns:
        The shared fixture epoch second.
    """
    return FIXED_NOW_EPOCH_S


def _alert_messages(ledger_path: Path) -> list[str]:
    """Read every ledgered alert message, verifying the chain first.

    Args:
        ledger_path: The alert ledger's path.

    Returns:
        One message per `AlertEmitted` row, in ledger order.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    return [
        str(json.loads(record.payload_json)["data"]["message"])
        for record in records
        if record.event_type == "AlertEmitted"
    ]


def test_a_full_volume_mid_tick_leaves_the_loop_beating_and_says_so(
    books_dir: Path,
    caplog: pytest.LogCaptureFixture,
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    research_tools_factory: Callable[[], object],
    tmp_path: Path,
) -> None:
    """A real tick whose ledger volume fills is survived, alerted, and ledgered.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        caplog: The pytest log capture fixture.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to hand the hook the disk-failing dependency bundle.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        research_tools_factory: Builds the offline research tools.
        tmp_path: The per-test scratch directory.
    """
    caplog.set_level(logging.INFO)
    from windbreak.scheduler import loop as loop_module

    tick_ledger_path = ledger_path_for(tmp_path, "tick.db")
    alert_ledger_path = ledger_path_for(tmp_path, "alerts.db")
    real_deps = loop_module.build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=tick_ledger_path,
        report_dir=report_dir,
        config=paper_config,
        research_tools=research_tools_factory(),
        clock=_fixed_clock,
    )
    store = _FullVolumeStore(real_deps.store, refuse=_FULL_AT_EVENT_TYPE)
    deps = dataclasses.replace(real_deps, store=store)
    monkeypatch.setattr(loop_module, "build_paper_deps", lambda **_kwargs: deps)
    supervisor = BeatSupervisor(
        component="pipeline",
        dispatcher=AlertDispatcher(
            sinks=[],
            ledger_writer=LedgerAlertWriter(alert_ledger_path, component="pipeline"),
        ),
    )

    run_loop(
        0,
        max_beats=2,
        on_beat=_build_paper_on_beat(
            _paper_args(tmp_path, tick_ledger_path), paper_config
        ),
        supervisor=supervisor,
    )

    assert store.refused == [_FULL_AT_EVENT_TYPE, _FULL_AT_EVENT_TYPE]
    assert [
        record.message
        for record in caplog.records
        if "heartbeat seq=" in record.message
    ] == [
        "mode=TICK_FAILED heartbeat seq=1",
        "mode=TICK_FAILED heartbeat seq=2",
    ]
    assert _alert_messages(alert_ledger_path) == [
        "beat seq=1 failed: OperationalError: database or disk is full"
    ]
    store.verify_chain()
    assert [record.event_type for record in store.read_all()].count(
        "ModeHeartbeat"
    ) == 2
    store.close()


def test_the_same_tick_without_a_full_volume_reports_its_real_mode(
    books_dir: Path,
    caplog: pytest.LogCaptureFixture,
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    research_tools_factory: Callable[[], object],
    tmp_path: Path,
) -> None:
    """The control: an unimpeded real tick heartbeats PAPER and pages nobody.

    Without this the survival test above could pass against a loop that reports
    `TICK_FAILED` unconditionally. It is also #447's acceptance criterion (a)
    against the real kernel: the mode in the log line is `deps.kernel.mode`,
    which for this tick is the same `PAPER` the `ModeHeartbeat` row carries.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        caplog: The pytest log capture fixture.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to hand the hook the real dependency bundle.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        research_tools_factory: Builds the offline research tools.
        tmp_path: The per-test scratch directory.
    """
    caplog.set_level(logging.INFO)
    from windbreak.scheduler import loop as loop_module

    tick_ledger_path = ledger_path_for(tmp_path, "tick.db")
    alert_ledger_path = ledger_path_for(tmp_path, "alerts.db")
    deps = loop_module.build_paper_deps(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=tick_ledger_path,
        report_dir=report_dir,
        config=paper_config,
        research_tools=research_tools_factory(),
        clock=_fixed_clock,
    )
    monkeypatch.setattr(loop_module, "build_paper_deps", lambda **_kwargs: deps)
    supervisor = BeatSupervisor(
        component="pipeline",
        dispatcher=AlertDispatcher(
            sinks=[],
            ledger_writer=LedgerAlertWriter(alert_ledger_path, component="pipeline"),
        ),
    )

    run_loop(
        0,
        max_beats=1,
        on_beat=_build_paper_on_beat(
            _paper_args(tmp_path, tick_ledger_path), paper_config
        ),
        supervisor=supervisor,
    )

    assert [
        record.message
        for record in caplog.records
        if "heartbeat seq=" in record.message
    ] == ["mode=PAPER heartbeat seq=1"]
    assert not alert_ledger_path.exists()
    deps.store.verify_chain()
    deps.store.close()
