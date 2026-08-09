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
`config.exchange.provider`, `config.exchange.environment`, each recognized
provider in `config.forecast.ensemble`, `config.forecast.triage_model`, and
`config.forecast.vote_ensemble`, the `config.forecast.research` hosts, and
`config.alerts.allowed_hosts`; an unrecognized provider contributes no host, so
an unknown exchange or model can never silently inherit network access.

### Live forecast-provider egress

When `forecast.provider_transport.mode` is `live` (it defaults to `cassette`,
which reaches nothing), `windbreak.main` builds one
`windbreak.net.live_http.LiveHttpTransport` **per provider**. Each carries only
that provider's credential and is pinned to a single-host allowlist of its own,
so one vendor's key structurally cannot travel to another vendor's endpoint —
the failure a single shared transport would produce on the very first vote.

Each endpoint is screened against `allowlist_from_config` *before any session
object exists*, and a provider host only reaches that allowlist by being named
in `config.forecast.vote_ensemble`, so egress nobody declared refuses at
startup. Redirects are refused (`allow_redirects=False`), so an on-path
responder cannot steer a dial to another host, and an integer timeout bounds
every dial.

Credentials are held in send-time headers on the transport and are never placed
on the header-free `HttpRequest` the provider adapters build — the object a
recording cassette persists — so a key cannot be written into a cassette. As
everywhere else in this repo, configuration carries the *name* of the
environment variable (`*_api_key_env`) and never the value, because every config
leaf is flattened verbatim into the hash-chained `ConfigLoaded` event and cannot
be redacted afterwards. Research *page fetches* deliberately carry no credential
at all: an anonymous read of a third-party page needs none, and inventing a key
leaf for it would put a secret in the ledger for nothing.

### Alert-sink egress

Every alert sink that leaves the box — `NtfySink`, `WebhookSink`, and
`SmtpSink` — takes the allowlist as a **required** constructor argument and
screens its destination at construction, so an off-list host is refused before
a single packet is sent (`SmtpSink` has no URL, so its bare relay host goes
through `OutboundAllowlist.require_host`). `DesktopSink` and `LogOnlySink` are
exempt: neither leaves the machine.

The hosts an alert may reach come from `alerts.allowed_hosts` **only** — never
from the sink entries' own destinations. Deriving the
allowlist from the URLs it screens would make the check unfalsifiable: every
configured sink would admit itself and the veto could never fire. Requiring the
host in two independent fields means a single mistyped or tampered destination
cannot open egress on its own. `alerts.allowed_hosts` is empty by default, so a
deployment that has declared nothing reaches nothing.

A sink whose destination is off the allowlist, whose type is unrecognized, or
which cannot deliver as configured raises `AlertSinkConfigError` at composition
and stops the process — a broken alerting path is never degraded to a
healthy-looking log-only dispatch. A sink the operator simply has not filled in
yet is skipped with a WARNING naming its *type only*, and the alert still
surfaces through the dispatcher's log-only fallback.

### Alert destinations never live in configuration

An ntfy topic is a bearer capability and a webhook URL can embed a token in its
userinfo, path, or query. Neither may be a configuration leaf, because every
config leaf is flattened by `diff_configs`, persisted **verbatim — old value and
new value** into the hash-chained `ConfigLoaded` ledger event on every
`windbreak run --config … --ledger-path …`, and folded again into the plaintext
`config_versions.json` read model by `windbreak.ledger.rebuild`. All three are
append-only: a secret that reaches them cannot be redacted afterwards.

So `AlertSink` stores only the **name of an environment variable** —
`topic_env`, `base_url_env`, `url_env` — exactly as
`FutureSearchProviderSettings.api_key_env` and
`ResearchSettings.search_api_key_env` already do, and
`windbreak.alerts.factory.build_sinks` resolves it against `os.environ` at
composition time. A named variable that is unset or empty raises
`AlertSinkConfigError` and stops the process, naming the *variable* and the
config field — never a value, of which there is none. Skipping it instead would
leave a deployment whose configuration advertises an alert channel and whose
alerts go nowhere.

The `smtp` block is deliberately **not** indirected, and its leaves do reach the
ledger diff. None of them can carry a credential:

- `smtp.host` must also be declared in `alerts.allowed_hosts` for the sink to
  build at all, so the relay hostname is unavoidably in plaintext configuration
  already; hiding it in one field while publishing it in the other would be
  theatre. The same is true of every alert host — host-level information is
  non-secret by construction in this design.
- `smtp.sender`/`smtp.recipients` are mailbox addresses, not bearer
  capabilities: holding one grants no ability to send or read a windbreak alert.
  `SmtpSinkConfig` has no username/password field — the relay authenticates this
  process by network position, never by these values — and `recipients` is a
  list, which environment-variable indirection cannot carry without inventing a
  delimiter convention that would itself be a parsing hazard.

This knowingly deviates from SPEC §16's literal `topic:` key
(`plans/SPEC_v3.md` §16); the placeholder is carried as `topic_env` instead.

Alert destinations are treated as secrets once resolved, too: no log line or
error message emitted by
`windbreak.alerts.factory` ever contains more of a destination than its
hostname. Where a hostname cannot be *proven* — a URL whose netloc will not
parse (`https:///host/path?token=…`, a plausible one-slash-too-many typo), or a
bare-host field such as `smtp.host` holding something that is not a bare host —
the message names the sink type and the configuration field to correct and
withholds the destination entirely, rather than falling back to echoing it. The
decision is driven by proving every character of a value is hostname-legal, not
by stripping the URL separators a leak might use, so an unanticipated malformed
destination fails closed onto disclosing nothing. One consequence worth knowing
before you meet it: an IPv6 literal relay host contains `:`, which is not
hostname-legal, so an off-allowlist IPv6 `smtp.host` gets the fully-redacted
"malformed destination" message rather than being named. That is the fail-safe
working as designed, not a parsing bug.

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
