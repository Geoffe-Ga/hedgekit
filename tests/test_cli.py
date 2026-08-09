"""Tests for the `windbreak run` CLI (issue #10).

Covers argument parsing (`build_parser`) in isolation, plus one
end-to-end smoke test through `main()` that pins the observable
contract: heartbeats are logged to stderr and the process returns 0
after a bounded number of beats.
"""

from __future__ import annotations

import json
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
