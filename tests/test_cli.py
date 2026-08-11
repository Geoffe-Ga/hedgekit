"""Tests for the `windbreak run` CLI (issue #10).

Covers argument parsing (`build_parser`) in isolation, plus one
end-to-end smoke test through `main()` that pins the observable
contract: heartbeats are logged to stderr and the process returns 0
after a bounded number of beats.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from windbreak.ledger.store import SqliteLedgerStore
from windbreak.main import build_parser, main


def test_build_parser_parses_run_with_defaults() -> None:
    """`run` with no flags uses the documented defaults."""
    args = build_parser().parse_args(["run"])

    assert args.command == "run"
    assert args.heartbeat_interval == 5.0
    assert args.max_beats is None


def test_build_parser_parses_custom_heartbeat_interval() -> None:
    """`--heartbeat-interval 0.5` overrides the 5.0s default."""
    args = build_parser().parse_args(["run", "--heartbeat-interval", "0.5"])

    assert args.heartbeat_interval == 0.5


def test_build_parser_parses_max_beats() -> None:
    """`--max-beats` accepts an integer bound on the number of heartbeats."""
    args = build_parser().parse_args(["run", "--max-beats", "2"])

    assert args.max_beats == 2


def test_build_parser_rejects_negative_heartbeat_interval() -> None:
    """A negative interval is not a valid heartbeat cadence.

    argparse reports usage errors via SystemExit(2), not a Python
    exception the caller could accidentally swallow.
    """
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["run", "--heartbeat-interval", "-1"])

    assert exc_info.value.code == 2


def test_build_parser_rejects_negative_max_beats() -> None:
    """A negative beat budget is meaningless, so it is a usage error.

    Validated for parity with ``--heartbeat-interval``; a silently accepted
    negative would behave like ``0`` and stop before the first beat.
    """
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["run", "--max-beats", "-1"])

    assert exc_info.value.code == 2


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf"])
def test_build_parser_rejects_non_finite_heartbeat_interval(bad_value: str) -> None:
    """Non-finite intervals cannot define a cadence, so they are rejected.

    ``stop_event.wait(nan)`` is ill-defined, so ``nan``/``inf`` must fail at
    parse time via SystemExit(2) rather than reaching the loop.
    """
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["run", "--heartbeat-interval", bad_value])

    assert exc_info.value.code == 2


def test_build_parser_requires_a_command() -> None:
    """Invoking the CLI with no subcommand is a usage error, not a no-op."""
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([])

    assert exc_info.value.code == 2


def test_build_parser_rejects_unknown_command() -> None:
    """An unrecognized subcommand is a usage error."""
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["fly-to-the-moon"])

    assert exc_info.value.code == 2


def test_main_run_emits_heartbeats_and_max_beats_shutdown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: `main` drives the loop to completion and returns 0.

    Interval 0 keeps the test fast; max-beats 2 bounds it deterministically.
    """
    exit_code = main(["run", "--heartbeat-interval", "0", "--max-beats", "2"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "seq=1" in captured.err
    assert "seq=2" in captured.err
    assert "shutdown reason=max_beats" in captured.err


def test_main_run_emits_json_heartbeat_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With `configure_logging` installed, heartbeats are JSON, not plain text.

    Every line on stderr must parse as JSON with `level == "INFO"` and the
    heartbeat's `seq=1` marker inside `msg` -- pinning that `main()` routes
    logging through `windbreak.logging_setup.configure_logging` instead of
    `logging.basicConfig`.
    """
    exit_code = main(["run", "--heartbeat-interval", "0", "--max-beats", "1"])

    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line]
    payloads = [json.loads(line) for line in lines]

    assert exit_code == 0
    seq_one = next(payload for payload in payloads if "seq=1" in payload["msg"])
    assert seq_one["level"] == "INFO"


def test_alert_test_subcommand_dispatches_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`alert-test mode-change` dispatches via the log-only fallback and exits 0.

    With no `--config`, `_run_alert_test` builds its dispatcher from the SPEC
    §16 defaults, whose one ntfy sink is still a placeholder -- so no sink is
    built, the fallback (log-only) fires, and the ledger writer logs an
    `AlertEmitted` line, both observable as JSON on stderr.
    """
    exit_code = main(["alert-test", "mode-change"])

    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line]
    payloads = [json.loads(line) for line in lines]

    assert exit_code == 0
    assert any(payload.get("alert_type") == "mode change" for payload in payloads)
    assert any("mode change" in json.dumps(payload) for payload in payloads)


def test_alert_test_subcommand_rejects_unknown_type(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An alert type absent from the registry's cli tokens is a usage error."""
    with pytest.raises(SystemExit) as exc_info:
        main(["alert-test", "not-a-type"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "not-a-type" in captured.err


def test_alert_test_subcommand_is_valid_but_hidden_from_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`alert-test` is a real subcommand that parses, but is excluded from --help.

    `subparsers.add_parser("alert-test")` is registered *without* a `help`
    argument, so argparse creates no pseudo-action for it and omits it from the
    subcommand listing; combined with the subparsers' `metavar="command"` (which
    suppresses the auto-generated `{run,alert-test}` choice list), the command
    stays functional yet unadvertised in the top-level usage text.
    """
    parser = build_parser()

    args = parser.parse_args(["alert-test", "veto"])
    assert args.command == "alert-test"

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    captured = capsys.readouterr()
    assert "alert-test" not in captured.out


def test_alert_test_subcommand_accepts_all_registered_alert_types() -> None:
    """Every `AlertType`'s `cli_token` is a valid `alert-test` positional choice."""
    from windbreak.alerts.registry import AlertType, cli_token

    parser = build_parser()
    for alert_type in AlertType:
        args = parser.parse_args(["alert-test", cli_token(alert_type)])
        assert args.command == "alert-test"
        assert args.type == cli_token(alert_type)


def test_alert_test_subcommand_message_defaults_to_test_alert() -> None:
    """`--message` defaults to "test alert" when omitted."""
    args = build_parser().parse_args(["alert-test", "veto"])

    assert args.message == "test alert"


def test_alert_test_subcommand_message_can_be_overridden() -> None:
    """`--message` overrides the default alert body."""
    args = build_parser().parse_args(["alert-test", "veto", "--message", "custom body"])

    assert args.message == "custom body"


# --- issue #274: `alert-test` dispatches through the *configured* sinks -------
#
# Alert destinations are never config leaves (they would land verbatim in the
# hash-chained `ConfigLoaded` ledger event), so each sink below names the
# environment variable its destination is read from.

#: The variable the webhook CLI tests point at their endpoint.
_WEBHOOK_URL_VAR = "WINDBREAK_TEST_CLI_WEBHOOK_URL"

#: The variable the ntfy CLI test points at its bearer-capability topic.
_NTFY_TOPIC_VAR = "WINDBREAK_TEST_CLI_NTFY_TOPIC"


def test_alert_test_subcommand_accepts_a_config_path() -> None:
    """`--config` is how an operator proves their real sinks deliver."""
    args = build_parser().parse_args(["alert-test", "veto", "--config", "wb.yaml"])

    assert args.config == Path("wb.yaml")


def test_alert_test_subcommand_config_defaults_to_none() -> None:
    """Omitting `--config` falls back to the built-in §16 defaults."""
    assert build_parser().parse_args(["alert-test", "veto"]).config is None


def test_alert_test_subcommand_fails_closed_on_an_unreadable_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `--config` that cannot be loaded is fatal, not silently ignored."""
    exit_code = main(["alert-test", "veto", "--config", str(tmp_path / "absent.yaml")])

    assert exit_code == 1
    assert "FATAL" in capsys.readouterr().err


def test_alert_test_subcommand_fails_closed_on_an_off_allowlist_sink(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A webhook aimed off the declared allowlist stops the command (issue #274).

    The end-to-end fail-closed proof at the CLI boundary: a sink the deployment
    may not dial must never degrade to a healthy-looking log-only dispatch. No
    network is touched -- the URL is refused at composition, before any POST.
    """
    monkeypatch.setenv(_WEBHOOK_URL_VAR, "https://169.254.169.254/latest/meta-data")
    config_path = tmp_path / "windbreak.yaml"
    config_path.write_text(
        "alerts:\n"
        "  allowed_hosts: [hooks.example.com]\n"
        "  sinks:\n"
        "    - type: webhook\n"
        f"      url_env: {_WEBHOOK_URL_VAR}\n",
        encoding="utf-8",
    )

    exit_code = main(["alert-test", "veto", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FATAL" in captured.err
    assert "169.254.169.254" in captured.err


def test_alert_test_subcommand_fails_closed_on_an_unexported_destination_variable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sink naming an unset variable stops the command, naming the variable.

    Naming a variable and never exporting it is an operator mistake, not absent
    configuration: silently skipping it would leave a process that reports a
    configured alert channel and delivers nothing. The FATAL line reaches stderr
    verbatim, so it names only the variable -- there is no value to disclose.
    """
    monkeypatch.delenv(_WEBHOOK_URL_VAR, raising=False)
    config_path = tmp_path / "windbreak.yaml"
    config_path.write_text(
        "alerts:\n"
        "  allowed_hosts: [hooks.example.com]\n"
        "  sinks:\n"
        "    - type: webhook\n"
        f"      url_env: {_WEBHOOK_URL_VAR}\n",
        encoding="utf-8",
    )

    exit_code = main(["alert-test", "veto", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FATAL" in captured.err
    assert _WEBHOOK_URL_VAR in captured.err
    assert "url_env" in captured.err


def test_alert_test_subcommand_never_logs_a_configured_ntfy_topic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured sink is skipped loudly without leaking its topic.

    An ntfy topic is a bearer capability: anyone holding it can publish to (and
    subscribe to) the operator's alert channel, so it must never reach a log --
    not the configuration file it is deliberately kept out of, and not stderr.
    """
    monkeypatch.setenv(_NTFY_TOPIC_VAR, "s3cr3t-capability")
    config_path = tmp_path / "windbreak.yaml"
    config_path.write_text(
        f"alerts:\n  sinks:\n    - type: ntfy\n      topic_env: {_NTFY_TOPIC_VAR}\n",
        encoding="utf-8",
    )

    exit_code = main(["alert-test", "veto", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "s3cr3t-capability" not in captured.err
    assert "log-only" in captured.err


# --- issue #48: the four new `run` flags gating the always-on PAPER loop ------
#
# `build_parser()` does not yet register `--paper-books-dir`/`--cassette-path`/
# `--ledger-path`/`--report-dir`, so every `parse_args` call below that passes
# one currently fails with `SystemExit(2)` (argparse "unrecognized arguments"),
# the expected Gate 1 RED state for issue #48 -- a legitimate "absent behavior"
# failure, not a broken fixture.


def test_build_parser_parses_new_paper_loop_flags() -> None:
    """`run` accepts the four new PAPER-loop composition flags, as paths."""
    from pathlib import Path

    args = build_parser().parse_args(
        [
            "run",
            "--paper-books-dir",
            "/tmp/books",
            "--cassette-path",
            "/tmp/cassette.json",
            "--ledger-path",
            "/tmp/ledger.db",
            "--report-dir",
            "/tmp/reports",
        ]
    )

    assert args.paper_books_dir == Path("/tmp/books")
    assert args.cassette_path == Path("/tmp/cassette.json")
    assert args.ledger_path == Path("/tmp/ledger.db")
    assert args.report_dir == Path("/tmp/reports")


def test_build_parser_new_paper_loop_flags_default_to_none() -> None:
    """Omitting all four PAPER-loop flags leaves them `None` -- PAPER activates
    only when every one of them is supplied (issue #48).
    """
    args = build_parser().parse_args(["run"])

    assert args.paper_books_dir is None
    assert args.cassette_path is None
    assert args.ledger_path is None
    assert args.report_dir is None


# --- issue #343: `--paper-live-ticker` selects live venue books ---------------
#
# `build_parser()` does not yet register `--paper-live-ticker`, so the first
# test below fails with `SystemExit(2)` (argparse "unrecognized arguments") and
# the second with `AttributeError` -- the expected Gate 1 RED state for #343.


def test_build_parser_parses_the_paper_live_ticker_flag() -> None:
    """`run` accepts `--paper-live-ticker`, naming the live market to trade."""
    args = build_parser().parse_args(["run", "--paper-live-ticker", "KXFED-24DEC"])

    assert args.paper_live_ticker == "KXFED-24DEC"


def test_paper_live_ticker_defaults_to_none() -> None:
    """Omitting it leaves the PAPER loop on its fixture books, byte-identically."""
    args = build_parser().parse_args(["run"])

    assert args.paper_live_ticker is None


# --- issue #57: `windbreak ack --approval-id <32-hex>` -------------------------
#
# `build_parser()` does not yet register an `ack` subcommand, so every
# `parse_args`/`main` call below currently fails with `SystemExit(2)`
# ("invalid choice: 'ack'") -- the expected Gate 1 RED state for issue #57's
# ack CLI verb (the dashboard/watcher-facing counterpart to `windbreak kill`).
# Approval ids below are obviously-fake, low-entropy repeated hex pairs (never
# a realistic high-entropy secret), mirroring `windbreak/riskkernel/human_ack.py`'s
# own 32-hex-character `secrets.token_hex(16)` shape.


def test_build_parser_parses_ack_with_a_valid_approval_id_and_state_dir() -> None:
    """`ack --approval-id <32-hex> --state-dir DIR` parses cleanly."""
    approval_id = "ab" * 16
    args = build_parser().parse_args(
        ["ack", "--approval-id", approval_id, "--state-dir", "/tmp/state"]
    )

    assert args.command == "ack"
    assert args.approval_id == approval_id


def test_build_parser_rejects_a_malformed_approval_id() -> None:
    """An approval id that is not exactly 32 lowercase hex characters is a
    usage error (`SystemExit(2)`), not a silently-accepted value.
    """
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            ["ack", "--approval-id", "not-hex-at-all", "--state-dir", "/tmp/state"]
        )

    assert exc_info.value.code == 2


def test_main_ack_subcommand_writes_the_ack_file_and_exits_zero(
    tmp_path: Path,
) -> None:
    """`windbreak ack --approval-id ID --state-dir DIR` writes an empty file
    at `DIR/acks/ID` and exits 0 -- the presence-driven signal
    `windbreak.riskkernel.ack_flow.AckFileWatcher` polls for, mirroring
    `windbreak kill`'s `KILL`-file convention.
    """
    approval_id = "cd" * 16

    exit_code = main(
        ["ack", "--approval-id", approval_id, "--state-dir", str(tmp_path)]
    )

    assert exit_code == 0
    ack_file = tmp_path / "acks" / approval_id
    assert ack_file.exists()
    assert ack_file.read_text(encoding="utf-8") == ""


def test_main_ack_subcommand_with_a_malformed_id_writes_no_file(
    tmp_path: Path,
) -> None:
    """A malformed `--approval-id` is rejected before any file is ever
    written -- a usage error, not a silently-created bogus ack file.
    """
    with pytest.raises(SystemExit) as exc_info:
        main(["ack", "--approval-id", "ZZ-not-hex", "--state-dir", str(tmp_path)])

    assert exc_info.value.code == 2
    assert not (tmp_path / "acks").exists()


def test_main_run_with_only_ledger_path_flag_ledgers_config_but_no_paper(
    tmp_path: Path,
) -> None:
    """Supplying only `--ledger-path` (not the other three PAPER flags) never
    activates the PAPER loop, but issue #74 wires config loading to the real
    ledger regardless of PAPER activation: exactly one `ConfigLoaded` row is
    appended, and none of the PAPER-loop event types appear.

    Uses the built-in `WindbreakConfig` default (`mode_ceiling: "paper"`, SPEC
    §16) with no `--config` at all, so this still pins the REAL invariant that
    *flags alone* -- not just the mode ceiling -- gate PAPER activation, while
    reflecting the new config-ledgering behavior `--ledger-path` alone now has.
    """
    ledger_path = tmp_path / "ledger.db"

    exit_code = main(
        [
            "run",
            "--heartbeat-interval",
            "0",
            "--max-beats",
            "1",
            "--ledger-path",
            str(ledger_path),
        ]
    )

    assert exit_code == 0
    assert ledger_path.exists()
    store = SqliteLedgerStore(ledger_path)
    records = store.read_all()
    store.close()
    assert len(records) == 1
    assert records[0].event_type == "ConfigLoaded"
    assert {record.event_type for record in records}.isdisjoint(
        {
            "MarketSnapshotRecorded",
            "EquitySampled",
            "SelectorDecisionRecorded",
            "ForecastCreated",
        }
    )


#: The shipped books fixture the PAPER-wiring tests below replay. Copied into
#: `tmp_path` and re-dated, never mutated in place.
_SHIPPED_BOOKS = Path(__file__).resolve().parent / "fixtures" / "books" / "deep_walk"

#: A resolution horizon comfortably inside the *production* `[2, 120]`-day
#: window, applied relative to the wall clock the CLI actually runs on -- the
#: fixture's committed 2025 close time is outside every window a live run could
#: have. No screener threshold is touched: the fixture is what is re-dated.
_IN_WINDOW_HORIZON_DAYS = 30

#: A resting quantity far above the production `min_depth_contract_centis`
#: floor of 10 000, so the screen passes on real depth rather than a lowered
#: bar. The committed fixture rests 300, which is deliberately tiny so its
#: taker-walk arithmetic is readable by hand.
_DEEP_QUANTITY_CENTIS = 100_000

#: The fixed per-forecast research charge one CLI-driven PAPER tick incurs.
#: Mirrored rather than imported so a change to the production constant
#: surfaces here as a failing assertion instead of silently re-deriving itself.
_EXPECTED_RESEARCH_COST_MICROS = 3_000_000


def _paper_books(tmp_path: Path) -> Path:
    """Copy the shipped books fixture and make it clear the production screen.

    Only the two inputs the screen measures are changed -- the close time and
    the resting depth -- and both to values that genuinely satisfy the shipped
    thresholds. `ScreenerConfig()`'s defaults are untouched, so what these tests
    pin is the wiring behind the screen and never a relaxed one.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The prepared books directory.
    """
    books = tmp_path / "books"
    shutil.copytree(_SHIPPED_BOOKS, books)
    closes_at = datetime.now(UTC) + timedelta(days=_IN_WINDOW_HORIZON_DAYS)
    markets_path = books / "markets.json"
    markets = json.loads(markets_path.read_text(encoding="utf-8"))
    markets[0]["close_time"] = closes_at.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    markets_path.write_text(json.dumps(markets, indent=2), encoding="utf-8")
    sessions_path = books / "sessions.json"
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    for steps in sessions.values():
        for step in steps:
            step["book"]["yes_bids"] = [
                {"price": 4500, "quantity": _DEEP_QUANTITY_CENTIS}
            ]
            step["book"]["yes_asks"] = [
                {"price": 4600, "quantity": _DEEP_QUANTITY_CENTIS}
            ]
    sessions_path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    return books


def _paper_run_argv(
    tmp_path: Path, books: Path, ledger_path: Path, *, extra: list[str]
) -> list[str]:
    """Build a complete one-beat PAPER `run` argument vector.

    Args:
        tmp_path: pytest's per-test temporary directory.
        books: The prepared books directory.
        ledger_path: The ledger the run writes to.
        extra: Additional flags appended verbatim (keyword-only).

    Returns:
        The argument vector for `windbreak.main.main`.
    """
    config_path = tmp_path / f"{ledger_path.stem}-config.yaml"
    config_path.write_text("mode_ceiling: paper\n", encoding="utf-8")
    cassette_path = tmp_path / "cassette.json"
    cassette_path.write_text("{}", encoding="utf-8")
    return [
        "run",
        "--heartbeat-interval",
        "0",
        "--max-beats",
        "1",
        "--config",
        str(config_path),
        "--paper-books-dir",
        str(books),
        "--cassette-path",
        str(cassette_path),
        "--ledger-path",
        str(ledger_path),
        "--report-dir",
        str(tmp_path / f"reports-{ledger_path.stem}"),
        *extra,
    ]


def _event_types(ledger_path: Path) -> list[str]:
    """Return every ledgered row's event type, in sequence order.

    Args:
        ledger_path: The ledger database to read.

    Returns:
        The event-type discriminators.
    """
    store = SqliteLedgerStore(ledger_path)
    try:
        return [record.event_type for record in store.read_all()]
    finally:
        store.close()


def test_run_research_per_day_micros_reaches_the_budget_that_stops_the_tick(
    tmp_path: Path,
) -> None:
    """The flag's *effect* is observed, not merely its parse (#483).

    Issue #483 names its own root cause as "every seam is tested and passes;
    the composition of those seams is unreachable, and nothing tests the
    composition". The flag's parsing had a test and the pure override function
    had a test, and between them the line that actually applies the override
    was executed on every `run` case with the flag left `None` -- so deleting
    it made `--research-per-day-micros` a complete no-op with the whole suite
    still green.

    This drives two real `main(["run", ...])` invocations that differ in one
    token, over separate ledgers, and reads the consequence off disk. With the
    flag at `0` the tick reaches the forecast stage and is refused there: one
    `ResearchBudgetHalted` row, no forecast, no charge. Without it the shipped
    20 000 000-micro ceiling stands and the identical tick forecasts and books
    the charge. Deleting the override call turns the capped arm into the
    uncapped one, so it cannot pass with the wire cut.

    The whole event sequence is compared, not a membership check, so a tick
    that halted for some *other* reason -- a screen it failed, a book it could
    not read -- fails here rather than passing as a false positive.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    books = _paper_books(tmp_path)
    capped_ledger = tmp_path / "capped.db"
    uncapped_ledger = tmp_path / "uncapped.db"

    capped = main(
        _paper_run_argv(
            tmp_path,
            books,
            capped_ledger,
            extra=["--research-per-day-micros", "0"],
        )
    )
    uncapped = main(_paper_run_argv(tmp_path, books, uncapped_ledger, extra=[]))

    capped_types = _event_types(capped_ledger)
    uncapped_types = _event_types(uncapped_ledger)
    assert (capped, uncapped) == (0, 0)
    assert capped_types.count("ScreenDecisionRecorded") == 1
    assert capped_types.count("ResearchBudgetHalted") == 1
    assert capped_types.count("ForecastCreated") == 0
    assert capped_types.count("ResearchSpendRecorded") == 0
    assert uncapped_types.count("ScreenDecisionRecorded") == 1
    assert uncapped_types.count("ResearchBudgetHalted") == 0
    assert uncapped_types.count("ForecastCreated") == 1
    assert uncapped_types.count("ResearchSpendRecorded") == 1


def test_run_research_per_day_micros_books_the_charge_it_permits(
    tmp_path: Path,
) -> None:
    """The uncapped arm's charge is the shipped one, read off the ledger.

    The negative half of the test above needs to be a *real* forecast rather
    than merely the absence of a halt, or a wire-cut capped arm could match a
    wire-cut uncapped arm and both assertions would be describing one
    behaviour. The exact charge is pinned, so a tick that forecast for free
    would fail here.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    books = _paper_books(tmp_path)
    ledger_path = tmp_path / "ledger.db"

    exit_code = main(_paper_run_argv(tmp_path, books, ledger_path, extra=[]))

    store = SqliteLedgerStore(ledger_path)
    try:
        spends = [
            json.loads(record.payload_json)["data"]
            for record in store.read_all()
            if record.event_type == "ResearchSpendRecorded"
        ]
    finally:
        store.close()
    assert exit_code == 0
    assert len(spends) == 1
    assert spends[0]["cost_micros"] == _EXPECTED_RESEARCH_COST_MICROS
    assert spends[0]["market_ticker"] == "MKT-DEEP"


def test_run_survives_a_malformed_spend_row_instead_of_failing_to_compose(
    tmp_path: Path,
) -> None:
    """A row the fold cannot read halts research, never the whole loop.

    `spend_by_day_from_records` fails closed by design, and it is called from
    `build_paper_deps`, so the refusal used to stop the process from composing
    at all: no heartbeat, no equity sample, no positions snapshot, no
    verification cycle, and -- the one that matters -- no kill handling, under
    a `restart: on-failure` supervisor that would retry it forever over a row
    that cannot be deleted from an append-only hash-chained ledger.

    The row is fabricated exactly as a schema migration or an external tool
    would leave it: a genuine `ResearchSpendRecorded` row written by beat 1,
    with its `cost_micros` key renamed in place. Beat 2 then runs against it.
    The assertion is that the *loop* survived -- the liveness rows an operator
    and the kill switch depend on are all present -- while research itself is
    refused with a zero ceiling.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    books = _paper_books(tmp_path)
    ledger_path = tmp_path / "ledger.db"
    assert main(_paper_run_argv(tmp_path, books, ledger_path, extra=[])) == 0
    _rename_payload_key(ledger_path, "cost_micros", "cost_micros_v2")

    exit_code = main(_paper_run_argv(tmp_path, books, ledger_path, extra=[]))

    types = _event_types(ledger_path)
    assert exit_code == 0
    assert types.count("ResearchBudgetHalted") == 1
    assert types.count("ForecastCreated") == 1
    assert types.count("ModeHeartbeat") == 2
    assert types.count("EquitySampled") == 2
    assert types.count("PositionsSnapshotRecorded") == 2
    assert types.count("PipelineHeartbeatRecorded") == 2
    assert types.count("VerificationPassed") == 2


def _rename_payload_key(ledger_path: Path, key: str, replacement: str) -> None:
    """Rename one payload key in every row that carries it, in place.

    Written straight through `sqlite3` because the typed event refuses to
    construct without the key -- which is the point: this is the shape an
    external writer produces, not one windbreak can.

    Args:
        ledger_path: The ledger database to rewrite.
        key: The payload key to rename.
        replacement: The name to give it.
    """
    conn = sqlite3.connect(ledger_path)
    try:
        for sequence_number, payload_json in conn.execute(
            "SELECT sequence_number, payload_json FROM ledger"
        ).fetchall():
            envelope = json.loads(str(payload_json))
            if key not in envelope["data"]:
                continue
            envelope["data"][replacement] = envelope["data"].pop(key)
            conn.execute(
                "UPDATE ledger SET payload_json = ? WHERE sequence_number = ?",
                (json.dumps(envelope), sequence_number),
            )
        conn.commit()
    finally:
        conn.close()
