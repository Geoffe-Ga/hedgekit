# Security

This document describes windbreak's shipped security posture: the per-process
credential boundaries SPEC §5.2 mandates, the seven preflight checks SPEC §15
and §3.3 require before any live deployment, the outbound network allowlist,
and the supply-chain tooling actually wired into this repository today. It
documents the current, running code — not the aspirational SPEC — and calls
out gaps honestly where SPEC §15 asks for something not yet built.

## Reporting a vulnerability

Please use GitHub private security advisories: open the "Report a
vulnerability" form under the Security tab of
[github.com/Geoffe-Ga/windbreak](https://github.com/Geoffe-Ga/windbreak). Do
not open a public issue for a suspected vulnerability.

## Per-process credential boundaries (SPEC §5.2)

windbreak's four processes hold strictly disjoint credential scopes; no
component outside the Order Gateway ever holds a trade-capable key.

| Process | Exchange credentials | Other secrets |
|---|---|---|
| Process A — main pipeline (connector, screener, forecast engine, selector) | none | LLM + search provider keys only |
| Process B — Risk Kernel | read-only | approval-token **signing** key |
| Process C — Order Gateway | trade-only | approval-token **verification** key |
| Process D — Dashboard | none | dashboard bearer-auth token |

The dashboard's bearer-auth token is sourced only from the
`WINDBREAK_DASHBOARD_TOKEN` environment variable — never from config (which is
ledgered) or the ledger itself; a missing or blank value fails the process
closed.

This boundary is enforced structurally, not just documented: an AST-based
architectural test scans the whole tree and fails the build if any package
outside `windbreak.riskkernel` imports the signing-key handle, or if any
package outside `windbreak.order_gateway`/`windbreak.connector` imports the
exchange order-submission client (`tests/riskkernel/test_process_isolation.py`,
`tests/architecture/test_import_boundaries.py`). Both run as part of the
ordinary test suite that `scripts/check-all.sh` gates on — a credential-scope
violation is a Gate 1 failure, not a code-review nice-to-have.

## Preflight: production-readiness checks (SPEC §3.3, §15)

`windbreak preflight` runs seven fail-closed checks before a live deployment
and reports a nonzero exit code if any fails:

```bash
windbreak preflight --fixture-dir drills/fixtures --json
```

| Check | What it verifies | SPEC ref |
|---|---|---|
| `exchange.reachable_readonly` | The venue answers read-only status and balance calls. | §7.2 |
| `credentials.no_withdrawal_scope` | The trading key cannot withdraw funds. | §1.1-3 |
| `credentials.scope_verifiable` | The key's scope was actually self-tested (not merely assumed). | §15 |
| `credentials.trade_key_not_leaked` | The trade-key environment variable is not visible to this process. | §5.2 |
| `jurisdiction.markets_eligible` | Every cached market is jurisdiction-eligible, never `unknown` or `ineligible`. | §6.2 |
| `secrets.files_not_world_readable` | No configured secrets file has group/other read permission. | §15 |
| `credentials.llm_budgets_configured` | Both `config.forecast.budget.per_forecast_micros` and `config.forecast.budget.per_day_micros` are positive. | §5.2 |

Every check that reads a raising-capable collaborator runs fail-closed: a
collaborator that raises is graded a FAIL naming the exception, never silently
treated as a PASS or skipped. A world-readable secrets file is reported by path
and octal mode (never by content); the trade-key-leak check's own FAIL detail
names only that the variable is visible, never its value. `--secrets-file`
(repeatable) names each secrets file whose permissions to check; run it
against a real config with `--config <path>`.

**Known limitation — preflight runs against fixtures only today.** The
production-readiness check above runs against a fixture-backed, read-only
connector and an honest "no self-test support" scope prober; the real
credential self-test client and a real-connector preflight mode are tracked in
issue #197, which also covers adding a dedicated preflight entry to the
runbook.

## Outbound network egress allowlist

`windbreak.net.allowlist.OutboundAllowlist` makes the SPEC §15/§5.2 outbound
allowlist structural rather than advisory: every outbound URL a connector
dials is screened for parse-differential SSRF bytes and matched by exact,
lowercased hostname before the dial is permitted; anything else raises
`EgressDeniedError` and — when a ledger recorder is wired — records exactly
one `EgressDenied` event (telemetry never gates the refusal; the raise always
happens first). `allowlist_from_config` derives the permitted host set from
`config.exchange.provider`, `config.exchange.environment`, and each
recognized provider in `config.forecast.ensemble`,
`config.forecast.triage_model`, and `config.forecast.vote_ensemble`; an
unrecognized provider contributes no host, so an unknown exchange or model
can never silently inherit network access.

## Supply chain

The following run as pre-commit hooks and/or `scripts/check-all.sh` gates
(see `.pre-commit-config.yaml`):

- `pip-audit` — dependency vulnerability scanning.
- `mypy --strict` on `windbreak/`.
- `bandit` (Python security linter) on `windbreak/`.
- `detect-secrets` against a checked-in baseline.
- A local `no-floats-money-paths` hook (`scripts/lint_no_floats.py`) that
  forbids floats on `windbreak/numeric/`, `windbreak/ledger/`, and
  `windbreak/riskkernel/` — the money/price/probability paths SPEC §6.1
  requires stay integer-only.
- The import-boundary architectural tests named above, run as part of the
  standard pytest suite.

## Known limitations (SPEC §15 items not yet built)

- **Encrypted secrets file.** SPEC §15 calls for an OS keyring or an
  age-encrypted `secrets.enc.yaml`; today secrets are supplied via the
  environment and plain files whose permissions preflight checks, with no
  built-in encryption-at-rest layer. Not yet tracked.
- **Container image scan.** SPEC §15's supply-chain list includes a container
  image scan; no such scan is wired into CI today. Not yet tracked.
