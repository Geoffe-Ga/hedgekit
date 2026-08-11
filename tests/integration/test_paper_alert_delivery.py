"""The PAPER loop's alerts reach the sinks the operator configured (issue #444).

`docs/RUNBOOK.md` tells the operator that declaring a sink under `config.alerts`
is "a prerequisite for any unattended run". Before this suite that sentence was
false of the always-on PAPER loop and of nothing else: `windbreak.main` composes
`config.alerts` into live channels in `_build_alert_dispatcher`, but
`windbreak.scheduler.loop._build_verifier` built its own
`AlertDispatcher(sinks=[], ...)`, so the verification cycle's mismatch and
unknown-jurisdiction alerts -- the only alerting surface the loop has -- could
only ever reach the log-only fallback.

What these tests pin, and why each one is shaped the way it is:

1. `test_a_paper_verification_mismatch_is_delivered_to_the_configured_sink` --
   the delivery proof. Asserting `dispatcher.sinks != []` would prove almost
   nothing: it is the same shape as a guard on a hardcoded zero. So this drives
   a *real* verification breach (the venue moved behind the loop's back, exactly
   the unattributed movement reconciliation exists to catch) through the real
   `_resolve_on_beat` -> `_build_paper_on_beat` -> `build_paper_deps` ->
   `_build_verifier` chain and asserts the recording sink RECEIVED it, with
   exact type, exact severity, and full message equality. Observed RED against
   `origin/main`'s hardcoded `sinks=[]`: the sink received `[]`.
2. `test_a_paper_loop_with_no_configured_sink_still_beats_and_logs` -- the
   other direction. The shipped default declares one *placeholder* ntfy sink,
   which builds nothing; that deployment must still start and still log, since
   that is the documented default. Pinned with exact values, never `>= 1`.
3. `test_a_paper_run_with_an_undeliverable_sink_never_builds_its_bundle` -- a
   misconfigured sink refuses at startup rather than silently degrading to
   log-only, and now refuses *before* `build_paper_deps` opens the books, boots
   the gateway, and writes into a ledger it then abandons. Also RED before this
   change, for that second reason: the composition root used to run only after
   `_resolve_on_beat` had already built the whole bundle.
4. `test_a_run_composes_its_alert_root_once_and_only_when_a_beat_needs_it` --
   the sharing proof, as an exact count in both directions: exactly one
   composition on a PAPER run (so the supervisor and the verification cycle
   share one root rather than each building a parallel path), and exactly zero
   on a run with no per-beat hook at all.
5. `test_no_substring_of_a_token_bearing_sink_url_reaches_the_ledger` -- alert
   destinations are bearer capabilities and the hash chain is append-only, so
   nothing written into it can ever be redacted. This one builds a real webhook
   sink from a real token-bearing URL through the real `build_sinks`, fails its
   delivery, and asserts no fragment of the URL or its token appears anywhere
   in the ledger's bytes.

Every mutation of the seam these five cover was killed: unwiring
`_build_verifier`'s dispatcher, dropping the kwarg at any of the three call
sites between `_run_heartbeat` and `_build_verifier`, inverting
`_resolved_dispatcher`'s guard, un-memoizing the root, composing it on a
hookless run, and composing it after the bundle rather than before.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from tests.integration.conftest import FIXED_NOW_EPOCH_S, ledger_path_for
from windbreak.alerts.registry import AlertSeverity, AlertType
from windbreak.ledger.store import SqliteLedgerStore
from windbreak.numeric.types import MoneyMicros

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from windbreak.config.schema import WindbreakConfig
    from windbreak.scheduler.loop import PaperTickDeps

#: Cash removed from the paper account with no execution behind it -- the one
#: venue movement fill accounting genuinely cannot explain, and therefore the
#: only reliable way to drive a reconciliation breach.
_UNEXPLAINED_CASH_MICROS = 7_000_000

#: The exact body `windbreak.riskkernel.verification` dispatches on a breach.
#: Compared in full rather than with a `match=` substring, because a substring
#: assertion passes just as well against a message that grew a second, wrong
#: half.
_MISMATCH_MESSAGE = "exchange verification mismatch beyond tolerance"

#: A webhook destination whose *path* is a bearer token, of the shape issue
#: #274 found `_denied()` echoing whole. Every one of these fragments is
#: searched for individually in the ledger bytes, so a leak of any part of it
#: -- not merely the whole string -- fails the secrets test.
_TOKEN_URL = "https://hooks.example.test/services/T0PS3CR3T/B0T0K3N/xoxb-444-leak"

#: The fragments of `_TOKEN_URL` that must never appear in the hash chain. The
#: host is deliberately excluded: it is declared separately in
#: `alerts.allowed_hosts`, so it is not the secret. Everything below is.
_TOKEN_FRAGMENTS = ("T0PS3CR3T", "B0T0K3N", "xoxb-444-leak", _TOKEN_URL)


@dataclass
class _RecordingSink:
    """An `AlertSink` double recording every alert delivered through it.

    Attributes:
        name: The sink's identifier, as the `AlertSink` protocol requires.
        delivered: One `(type, severity, message)` triple per delivered alert,
            in delivery order.
    """

    name: str = "recording"
    delivered: list[tuple[AlertType, AlertSeverity, str]] = field(default_factory=list)

    def send(
        self, alert_type: AlertType, severity: AlertSeverity, message: str
    ) -> None:
        """Record one delivered alert.

        Args:
            alert_type: The dispatched alert type.
            severity: The alert's severity.
            message: The alert body.
        """
        self.delivered.append((alert_type, severity, message))


def _fixed_clock() -> int:
    """Return the suite's fixed epoch second, so a tick is deterministic.

    Returns:
        The shared fixture epoch second.
    """
    return FIXED_NOW_EPOCH_S


def _paper_args(
    *,
    books_dir: Path,
    cassette_path: Path,
    ledger_path: Path,
    report_dir: Path,
) -> argparse.Namespace:
    """Build the `run` namespace the PAPER composition root reads its flags from.

    Args:
        books_dir: The paper-exchange fixture directory.
        cassette_path: The empty offline cassette.
        ledger_path: The tick ledger's path.
        report_dir: Where weekly-report stubs are written.

    Returns:
        A namespace carrying the four PAPER flags, no live ticker, and the
        `--process` token the alert root stamps its ledgered rows with.
    """
    return argparse.Namespace(
        paper_books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
        paper_live_ticker=None,
        process="pipeline",
    )


@pytest.fixture
def captured_deps(
    monkeypatch: pytest.MonkeyPatch, research_tools_factory: Callable[[], Any]
) -> Iterator[list[PaperTickDeps]]:
    """Capture the `PaperTickDeps` the real composition root builds.

    The wrapper is a *pass-through*, not a stand-in: it forwards every keyword
    `windbreak.main` supplies -- `dispatcher` above all -- to the real
    `build_paper_deps`, and only adds the two offline seams (`clock`,
    `research_tools`) that `_build_paper_on_beat` leaves at their wall-clock
    and network defaults. A stand-in that dropped the caller's kwargs would
    make the delivery assertion unfalsifiable, which is exactly the trap this
    suite exists to avoid.

    Args:
        monkeypatch: Used to install the pass-through wrapper.
        research_tools_factory: Builds the offline research tools double.

    Yields:
        The list the built bundle is appended to, one entry per build.
    """
    from windbreak.scheduler import loop as loop_module

    built: list[PaperTickDeps] = []
    real_build = loop_module.build_paper_deps

    def _capture(**kwargs: Any) -> PaperTickDeps:
        """Build the real bundle offline, recording it.

        Args:
            **kwargs: Every keyword the production call site supplied,
                forwarded verbatim.

        Returns:
            The real `PaperTickDeps`.
        """
        deps = real_build(
            **kwargs, clock=_fixed_clock, research_tools=research_tools_factory()
        )
        built.append(deps)
        return deps

    monkeypatch.setattr(loop_module, "build_paper_deps", _capture)
    yield built
    for deps in built:
        deps.store.close()


def _ledgered_alerts(ledger_path: Path) -> list[tuple[str, str]]:
    """Read every ledgered alert as a `(severity, message)` pair, chain first.

    Args:
        ledger_path: The hash-chained ledger to read.

    Returns:
        One pair per `AlertEmitted` row, in ledger order.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        records = store.read_all()
    finally:
        store.close()
    return [
        (
            str(json.loads(record.payload_json)["data"]["severity"]),
            str(json.loads(record.payload_json)["data"]["message"]),
        )
        for record in records
        if record.event_type == "AlertEmitted"
    ]


def _ledger_bytes(ledger_path: Path) -> bytes:
    """Return every byte the ledger holds, sidecars included.

    `SqliteLedgerStore` sets `PRAGMA journal_mode=WAL`, so a row appended
    moments ago lives in the `-wal` sidecar until a checkpoint folds it into
    the main database file. Reading only `ledger_path` therefore searches a
    file the newest rows have not reached yet, and a secrets assertion over
    those bytes passes for the wrong reason -- absent evidence reading as
    healthy, which is the one thing it must never do. Concatenating every
    `ledger.db*` file makes the search independent of checkpoint timing.

    Args:
        ledger_path: The hash-chained ledger's main database path.

    Returns:
        The concatenated bytes of the database and its WAL/shm sidecars.
    """
    return b"".join(
        sidecar.read_bytes()
        for sidecar in sorted(ledger_path.parent.glob(f"{ledger_path.name}*"))
    )


def _move_the_venue_behind_the_loops_back(deps: PaperTickDeps) -> None:
    """Take cash off the exchange with no execution behind it.

    The same divergence `tests/integration/test_paper_verification.py` uses to
    drive a breach, and the only kind that still is one. *Trading* directly on
    the exchange is not: the bookkeeper reads the venue's own fill and arrival
    logs, so it books whatever landed there regardless of who placed it, and
    since issue #423 booked the resting-order collateral too, the whole
    movement reconciles. Cash that simply left the account explains nothing and
    still halts the loop -- and the change is to the opening balance every later
    reading derives from, so the divergence persists across cycles.

    Args:
        deps: The wired bundle whose exchange is moved.
    """
    opening = deps.exchange.balances
    deps.exchange.balances = type(opening)(
        total=MoneyMicros(opening.total.value - _UNEXPLAINED_CASH_MICROS),
        available=MoneyMicros(opening.available.value - _UNEXPLAINED_CASH_MICROS),
        fetched_at=opening.fetched_at,
    )


def test_a_paper_verification_mismatch_is_delivered_to_the_configured_sink(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A real breach in the wired PAPER loop reaches the operator's sink (#444).

    This is the acceptance criterion issue #444 names. The assertion is on what
    the sink *received* -- exact type, exact severity, full message equality,
    and an exact one-element list rather than a membership check -- because a
    dispatcher that merely holds a non-empty sink list proves nothing about
    delivery.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures the bundle the composition root builds.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to make `build_sinks` yield the recording sink.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.main import _beat_alert_root, _resolve_on_beat

    sink = _RecordingSink()
    monkeypatch.setattr("windbreak.main.build_sinks", lambda *_a, **_k: (sink,))
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path_for(tmp_path),
        report_dir=report_dir,
    )

    on_beat = _resolve_on_beat(
        args, paper_config, alert_root=_beat_alert_root(args, paper_config)
    )
    assert on_beat is not None
    on_beat(1)
    _move_the_venue_behind_the_loops_back(captured_deps[0])
    on_beat(2)

    assert sink.delivered == [
        (AlertType.RECONCILIATION_MISMATCH, AlertSeverity.CRITICAL, _MISMATCH_MESSAGE)
    ]


def test_a_paper_loop_with_no_configured_sink_still_beats_and_logs(
    books_dir: Path,
    caplog: pytest.LogCaptureFixture,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    paper_config: WindbreakConfig,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """With no deliverable sink the same breach still starts, runs, and logs.

    The shipped SPEC S16 default declares one placeholder ntfy sink, which
    builds nothing. That deployment is the documented default, so it must not be
    made to refuse: it beats, and the dispatcher's log-only fallback carries the
    alert. Both halves are pinned by exact value -- exactly one fallback log
    record and exactly one ledgered `AlertEmitted` row, each carrying the breach
    body in full -- so a change that silenced the fallback, or that fanned one
    alert out twice, fails here rather than reading as healthy.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        caplog: The pytest log capture fixture.
        captured_deps: Captures the bundle the composition root builds.
        cassette_path: The empty offline cassette.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.main import _beat_alert_root, _resolve_on_beat

    caplog.set_level(logging.INFO, logger="windbreak.alerts")
    ledger_path = ledger_path_for(tmp_path)
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
    )

    on_beat = _resolve_on_beat(
        args, paper_config, alert_root=_beat_alert_root(args, paper_config)
    )
    assert on_beat is not None
    on_beat(1)
    _move_the_venue_behind_the_loops_back(captured_deps[0])
    on_beat(2)

    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "windbreak.alerts" and record.levelno == logging.CRITICAL
    ] == [_MISMATCH_MESSAGE]
    assert _ledgered_alerts(ledger_path) == [("critical", _MISMATCH_MESSAGE)]


def test_a_paper_run_with_an_undeliverable_sink_never_builds_its_bundle(
    books_dir: Path,
    capsys: pytest.CaptureFixture[str],
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A misconfigured sink stops a PAPER run before it builds durable state.

    Fail closed, and fail closed *early*: an operator mistake that makes the
    alert path undeliverable must refuse at startup rather than degrade to
    log-only, and it must refuse before `build_paper_deps` has opened the books,
    booted the order gateway, and written into a ledger it then abandons.

    The falsifiable half is the ledger's *contents*, not its existence: issue
    #74 has a successful config load append `ConfigLoaded` unconditionally,
    before any hook is resolved, so the file is there either way. What is not
    there under the fixed ordering is `RecoveryCompleted` -- the row
    `build_paper_deps`'s boot-recovering gateway writes -- so an exact-equality
    assertion on the event types distinguishes "refused before the bundle" from
    "refused after it", which mere existence cannot.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        capsys: Captures the JSON log lines `main` writes to stderr.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to make sink composition raise.
        report_dir: The weekly-report output directory.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.alerts.factory import AlertSinkConfigError
    from windbreak.main import main

    def _raise(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        """Refuse to compose any sink.

        Returns:
            Never returns.

        Raises:
            AlertSinkConfigError: Always.
        """
        raise AlertSinkConfigError("ntfy sink names unset NTFY_TOPIC_ENV")

    monkeypatch.setattr("windbreak.main.build_sinks", _raise)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mode_ceiling: paper\n", encoding="utf-8")
    ledger_path = ledger_path_for(tmp_path, "tick.db")

    exit_code = main(
        [
            "run",
            "--heartbeat-interval",
            "0",
            "--max-beats",
            "1",
            "--config",
            str(config_path),
            "--paper-books-dir",
            str(books_dir),
            "--cassette-path",
            str(cassette_path),
            "--ledger-path",
            str(ledger_path),
            "--report-dir",
            str(report_dir),
        ]
    )
    messages = [
        str(json.loads(line)["msg"])
        for line in capsys.readouterr().err.splitlines()
        if line
    ]

    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        event_types = [record.event_type for record in store.read_all()]
    finally:
        store.close()

    assert exit_code == 1
    assert "FATAL: ntfy sink names unset NTFY_TOPIC_ENV" in messages
    assert [message for message in messages if "heartbeat" in message] == []
    assert event_types == ["ConfigLoaded"]


def _run_argv(
    *,
    books_dir: Path | None,
    cassette_path: Path,
    config_path: Path,
    ledger_path: Path,
    report_dir: Path,
) -> list[str]:
    """Build the `run` argv for a one-beat loop, with or without PAPER wired.

    Args:
        books_dir: The paper-exchange fixture directory, or `None` to leave the
            PAPER flags incomplete so no hook is wired at all.
        cassette_path: The empty offline cassette.
        config_path: The `--config` file to load.
        ledger_path: The ledger database path.
        report_dir: The weekly-report output directory.

    Returns:
        The argument vector `main` is invoked with.
    """
    argv = [
        "run",
        "--heartbeat-interval",
        "0",
        "--max-beats",
        "1",
        "--config",
        str(config_path),
        "--ledger-path",
        str(ledger_path),
    ]
    if books_dir is None:
        return argv
    return [
        *argv,
        "--paper-books-dir",
        str(books_dir),
        "--cassette-path",
        str(cassette_path),
        "--report-dir",
        str(report_dir),
    ]


@pytest.mark.parametrize(
    ("paper_wired", "expected_compositions"),
    [(True, 1), (False, 0)],
)
def test_a_run_composes_its_alert_root_once_and_only_when_a_beat_needs_it(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    expected_compositions: int,
    monkeypatch: pytest.MonkeyPatch,
    paper_wired: bool,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """One `run` composes one alert root -- and none at all with no beat hook.

    Two properties in one exact count, because both are about the *same* number.

    * With PAPER wired the count is exactly `1`, never `2`. The supervisor
      (issues #443/#447) and the PAPER tick's verification cycle (issue #444)
      are the two alerting surfaces an unattended run has, and they must share
      one composed root rather than build a parallel path each: two roots over
      one configuration means two sets of live sinks, two reads of the
      environment for each destination, and two things that can drift apart
      with nothing failing. Composed once plus delivered from both surfaces
      (`test_a_paper_verification_mismatch_is_delivered_to_the_configured_sink`
      here, and `test_the_composed_supervisor_dispatches_through_the_configured_sinks`
      in `tests/test_beat_supervision.py`) is what makes "shared" falsifiable.
    * With no beat hook the count is exactly `0`. A bare RESEARCH heartbeat has
      nothing to alert about, so composing sinks it will never reach -- and
      possibly refusing to start over one -- would be a new failure mode rather
      than a fixed one.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures (and closes) any bundle the run builds.
        cassette_path: The empty offline cassette.
        expected_compositions: The exact number of `build_sinks` calls expected.
        monkeypatch: Used to count sink compositions.
        paper_wired: Whether the PAPER flags are supplied.
        report_dir: The weekly-report output directory.
        tmp_path: The per-test scratch directory.
    """
    from windbreak.main import main

    compositions: list[object] = []

    def _counting_build_sinks(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        """Compose no sinks, recording that composition was attempted.

        Returns:
            An empty sink tuple, so the run behaves as a no-sink deployment.
        """
        compositions.append(object())
        return ()

    monkeypatch.setattr("windbreak.main.build_sinks", _counting_build_sinks)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mode_ceiling: paper\n", encoding="utf-8")

    exit_code = main(
        _run_argv(
            books_dir=books_dir if paper_wired else None,
            cassette_path=cassette_path,
            config_path=config_path,
            ledger_path=ledger_path_for(tmp_path, "tick.db"),
            report_dir=report_dir,
        )
    )

    assert exit_code == 0
    assert len(compositions) == expected_compositions


def test_no_substring_of_a_token_bearing_sink_url_reaches_the_ledger(
    books_dir: Path,
    captured_deps: list[PaperTickDeps],
    cassette_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_config: WindbreakConfig,
    report_dir: Path,
    tmp_path: Path,
) -> None:
    """A sink's bearer token never reaches the append-only hash chain (#274).

    The whole configured path is real here -- a real `AlertSink` config entry,
    the real `build_sinks`, the real allowlist screen, a real `WebhookSink`
    over a URL whose path is a token -- with only the HTTP transport replaced,
    and it is replaced by one that *fails*, because a failing sink is what puts
    a sink's own detail string on the `AlertEmitted` event. The ledger is then
    read as raw bytes and searched for every fragment: a hash chain is
    append-only, so a value that reaches it can never be redacted afterwards.

    Args:
        books_dir: The shared `deep_walk` books fixture.
        captured_deps: Captures the bundle the composition root builds.
        cassette_path: The empty offline cassette.
        monkeypatch: Used to inject the failing transport and the environment.
        paper_config: The PAPER-ceilinged configuration.
        report_dir: The weekly-report output directory.
        tmp_path: The per-test scratch directory.
    """
    import dataclasses

    from windbreak.alerts import factory as factory_module
    from windbreak.config.schema import AlertsConfig
    from windbreak.config.schema import AlertSink as AlertSinkSpec
    from windbreak.main import _beat_alert_root, _resolve_on_beat

    dialled: list[str] = []

    def _failing_transport(url: str, payload: bytes, headers: dict[str, str]) -> None:
        """Fail every delivery, quoting the whole URL in the failure.

        Args:
            url: The destination URL, echoed into the error on purpose.
            payload: The (unused) request body.
            headers: The (unused) request headers.

        Raises:
            RuntimeError: Always, naming ``url`` so the leak has a live source.
        """
        del payload, headers
        dialled.append(url)
        raise RuntimeError(f"POST {url} failed")

    real_build_sinks = factory_module.build_sinks

    def _build_sinks_over_the_failing_transport(
        alerts: object, **kwargs: Any
    ) -> tuple[object, ...]:
        """Build the real sinks, but never over a real socket.

        Args:
            alerts: The configuration's alerts section.
            **kwargs: The allowlist and environment the caller supplied.

        Returns:
            Whatever the real `build_sinks` composes.
        """
        return real_build_sinks(alerts, **kwargs, http_transport=_failing_transport)

    monkeypatch.setattr(
        "windbreak.main.build_sinks", _build_sinks_over_the_failing_transport
    )
    monkeypatch.setenv("WINDBREAK_TEST_WEBHOOK_URL", _TOKEN_URL)
    config = dataclasses.replace(
        paper_config,
        alerts=AlertsConfig(
            sinks=(
                AlertSinkSpec(type="webhook", url_env="WINDBREAK_TEST_WEBHOOK_URL"),
            ),
            allowed_hosts=("hooks.example.test",),
        ),
    )
    ledger_path = ledger_path_for(tmp_path)
    args = _paper_args(
        books_dir=books_dir,
        cassette_path=cassette_path,
        ledger_path=ledger_path,
        report_dir=report_dir,
    )

    on_beat = _resolve_on_beat(args, config, alert_root=_beat_alert_root(args, config))
    assert on_beat is not None
    on_beat(1)
    _move_the_venue_behind_the_loops_back(captured_deps[0])
    on_beat(2)

    store = SqliteLedgerStore(ledger_path)
    try:
        store.verify_chain()
        event_types = [record.event_type for record in store.read_all()]
    finally:
        store.close()

    # Asserted before the redaction claim, never after: a run in which the sink
    # was never built and never dialled would satisfy "no token in the ledger"
    # for the wrong reason, forever. These two pin that the secret really was
    # live in the process at the moment the alert was recorded.
    assert dialled == [_TOKEN_URL]
    assert "VerificationMismatch" in event_types
    assert [
        fragment
        for fragment in _TOKEN_FRAGMENTS
        if fragment.encode() in _ledger_bytes(ledger_path)
    ] == []
