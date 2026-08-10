# windbreak

An open-source, locally hosted, always-on AI forecast trader for fully collateralized binary event markets (e.g., Kalshi).

**Status:** Pre-implementation scaffold. Building against [`plans/SPEC_v3.md`](plans/SPEC_v3.md).

**Quality metrics:** [📊 Live dashboard](https://geoffe-ga.github.io/windbreak/dashboard.html) — regenerated on every push to `main` by the [Quality Metrics Dashboard workflow](.github/workflows/metrics.yml).

**License target:** Apache-2.0

## Why "windbreak"

A windbreak is a barrier that blocks damaging wind so what's behind it can grow —
that's the design brief for this project. It's meant to be a windbreak against
the headwinds of capitalism ordinary people face building wealth: scarcity
mindset, debt, disadvantage, and the general asymmetry of who gets access to
sophisticated trading tools. Breaking those headwinds — and breaking the wind
of risk itself — is the point: AI-assisted trading infrastructure a normal
person can run, not just institutions, with the Risk Kernel, kill switch, and
floor invariant blunting the "wind" of catastrophic loss so entry into
prediction-market trading is safer and lower-risk than going in unprotected.
If it works, it's meant to be someone's windfall, their big break.

## What this is

`windbreak` is a local-first daemon that:

1. Screens prediction markets for questions where careful research can plausibly beat the crowd.
2. Uses an LLM "superforecaster" scaffold to produce calibrated probability estimates with verified citations.
3. Compares those estimates against live executable order books.
4. May create order intents — none of which can reach an exchange unless approved by an independent, veto-holding **Risk Kernel** and submitted through a token-verifying, credential-isolated **Order Gateway**.

The design descends from publicly documented AI-forecasting pipelines (screen by volume → exclude information-disadvantaged categories → deep LLM research → trade only where forecast and executable price disagree beyond fees). It deliberately does **not** assume the headline results around those pipelines are reproducible — widely cited figures come from a paper portfolio with no commissions/borrow costs/dividends, and a self-reported anecdote whose own author says the edge is already competed away.

## Important disclaimers

- **This is not investment advice.**
- Most operators should expect **no durable edge**. Discovering that and stopping at paper trading is a **success state** of this design, not a failure of the software.
- **Live trading is disabled by default.** Promotion from paper to live trading is gated by pre-registered, quantitative evidence — never by narrative or operator impatience.
- Only bounded-loss, fully collateralized binary event contracts are in scope. No margin, perps, options, leverage, or shorting-to-open.
- Legal eligibility to trade these products varies by jurisdiction and product; this software does not provide legal advice.
- The truest floor is money never deposited: fund the exchange only with risk capital, keep floor capital in an unlinked account, and grant only trade-scope API keys.

## Core invariants

1. **Floor Invariant** — worst-case equity must always be ≥ a configured floor, computed conservatively and enforced pre-trade by an independent process. Any unprovable input halts the system.
2. **Bounded-loss instruments only** — fully collateralized binary contracts with exactly known maximum loss at order time. Margin, perps, options, leverage, and equities/crypto spot/derivs are forbidden, not configurable.
3. **No trade credentials outside the Order Gateway** — research, forecasting, selection, and dashboard components never hold trade-capable credentials.
4. **Evidence-gated autonomy** — `RESEARCH → PAPER → LIVE_MICRO → LIVE`, with promotion by pre-registered quantitative gates only; demotion and halting are automatic.
5. **Research/execution firewall** — web content and model output can influence only the probability fields of a forecast; never config, credentials, routing, or control flow.
6. **Temporal integrity** — only real-time forecasts on then-unresolved questions count toward gates, guarding against LLM training-data leakage from backtests.
7. **Append-only auditability** — every snapshot, forecast, decision, veto, approval, order transition, and reconciliation is written to a hash-chained ledger.

## Architecture

Four isolated processes sharing only a ledger volume and localhost sockets:

- **Process A — Main pipeline** (no trade credentials): Market Connector → Screener → Forecast Engine → Trade Selector.
- **Process B — Risk Kernel** (read-only exchange credentials + signing key): independent veto authority over mode, floor enforcement, capital reservations, and approval tokens.
- **Process C — Order Gateway** (trade-scope credentials + verify key): the only component that can submit live orders; owns reconciliation.
- **Process D — Dashboard** (no exchange credentials, `127.0.0.1` only): visibility and a constrained set of allowed mutations (pause, kill, acknowledge, raise floor — never lower it).

Order flow has exactly one path: market snapshot → screen → (triage) → forecast → selector decision → order intent → Risk Kernel checks → capital reservation → signed approval token → Order Gateway verification → exchange submission → reconciliation → ledgered terminal state.

See [`plans/SPEC_v3.md`](plans/SPEC_v3.md) for the full specification: threat model, canonical data model, evaluation methodology, configuration reference, testing strategy, and milestone plan.

## Documentation

Operator-facing documentation lives at the repo root (SPEC §19):

- [`SECURITY.md`](SECURITY.md) — credential boundaries, the preflight checklist, egress allowlist, supply chain.
- [`RUNBOOK.md`](RUNBOOK.md) — numbered operator procedures: start/stop, kill/re-arm/ack, drills, preflight, rebuild, anchor/verify.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the four-process topology, order-flow path, and the import-boundary rule.
- [`ACCOUNTING.md`](ACCOUNTING.md) — the fixed-point unit types, conservative rounding, and the floor formula.
- [`EVALUATION.md`](EVALUATION.md) — the three evaluation tracks, baselines, bootstrap, and pre-registration.
- [`LEGAL_AND_COMPLIANCE.md`](LEGAL_AND_COMPLIANCE.md) — jurisdiction/product eligibility, out-of-scope categories, record export.
- [`OPERATOR_WARNINGS.md`](OPERATOR_WARNINGS.md) — the residual risks this software cannot remove.
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — the always-on PAPER loop's day-to-day mechanics (activation, dashboard views, weekly reports).

## Development

Scaffolded with [Start Green Stay Green](https://github.com/Geoffe-Ga/start_green_stay_green): quality gates, CI/CD, AI subagents, and the Ralph autonomous fleet loop are pre-configured.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

### Quality checks

```bash
pre-commit run --all-files   # all 32 hooks (recommended before commit)

./scripts/test.sh            # pytest with coverage
./scripts/lint.sh            # ruff
./scripts/format.sh --fix    # black + isort
./scripts/typecheck.sh       # mypy --strict
./scripts/security.sh        # bandit + pip-audit + detect-secrets (baseline)
./scripts/complexity.sh      # radon/xenon (≤10 cyclomatic)
./scripts/mutation.sh        # mutmut
./scripts/check-all.sh       # everything
```

### Quality standards

- **Test coverage:** ≥90% (spec requires 100% branch coverage + ≥90% mutation score on `riskkernel`, fixed-point accounting, and token verification — see SPEC §17.6)
- **Cyclomatic complexity:** ≤10 per function
- **Type hints:** 100%, `mypy --strict`
- **All linters:** zero violations

### Repository layout

```
windbreak/            # Main package
tests/               # Test suite
scripts/             # Quality-gate scripts + Ralph fleet mechanics (scripts/ralph/)
plans/               # SPEC_v3.md and planning documents
prompts/             # Maintenance-scan prompts
.github/workflows/   # CI, AI code review, maintenance scans, metrics dashboard
.claude/             # CLAUDE.md docs, skills, and subagent profiles
docs/                # Live metrics dashboard (GitHub Pages) + docs/RUNBOOK.md
                     # (operator docs proper live at repo root -- see Documentation above)
```

### Deployment

SPEC §5.1 mandates process isolation: the four processes run as **separate
services** sharing only the ledger volume and localhost sockets — killing one
must never kill another. (`pipeline` also mounts a `reports` volume, which no
other service touches, so it shares nothing.) `deploy/` ships two equivalent
skeletons for this at M0.

**docker compose**

```bash
export WINDBREAK_DASHBOARD_TOKEN=...   # required; see Dashboard below
docker compose -f deploy/docker-compose.yml up -d
```

Starts four services — `pipeline`, `riskkernel`, `order-gateway`, `dashboard`
— each built from the repo-root `Dockerfile` with `restart: on-failure`. The
build context is the repo root, set explicitly as `context: ..`: Compose
resolves a relative context against the *compose file's own* directory, so the
previous `build: .` meant `deploy/`, which holds no `Dockerfile`, and `up -d`
failed before creating a container (issue #445). Only `dashboard` publishes a
port, bound to `127.0.0.1:8080` (SPEC §14: no public inbound), and its ledger
mount is read-only since it holds no trade authority.

`pipeline` carries the four PAPER composition flags, so the stack runs the
PAPER loop rather than a bare RESEARCH heartbeat (issue #446) — it reads the
`deep_walk` paper-exchange fixture and the recorded forecast cassette from the
image's copy of this repo, and writes its hash-chained ledger to the shared
`ledger` volume and its weekly report stub to a `reports` volume.

The stack below was brought up from the file as committed; every transcript in
this section is its actual output.

```bash
$ docker compose -f deploy/docker-compose.yml logs --tail 2 pipeline
{"level": "INFO", "component": "pipeline", "msg": "mode=PAPER heartbeat seq=6"}
{"level": "INFO", "component": "pipeline", "msg": "mode=PAPER heartbeat seq=7"}

$ docker compose -f deploy/docker-compose.yml exec pipeline \
    ls /var/lib/windbreak/ledger /var/lib/windbreak/reports
/var/lib/windbreak/ledger:
windbreak.db  windbreak.db-shm  windbreak.db-wal

/var/lib/windbreak/reports:
weekly-2026-08-10.md

$ docker compose -f deploy/docker-compose.yml kill pipeline
$ docker compose -f deploy/docker-compose.yml ps -a --format '{{.Name}} {{.State}}'
deploy-dashboard-1      running
deploy-order-gateway-1  running
deploy-pipeline-1       exited
deploy-riskkernel-1     running
```

`mode=PAPER` rather than `mode=RESEARCH` is issue #446's fix stated in one
token, and the `.db` file is the `ledger` volume finally being written.

**Activating PAPER is not trading, and this stack places no order.** Two
independent reasons, both observed in that run. First, the shipped `deep_walk`
fixture's only market is screened **ineligible every tick** —
`ScreenDecisionRecorded {"eligible": false, "blocked_by":
["min_depth_contract_centis", "horizon_days"]}` — so the pipeline never reaches
forecasting at all. Second, behind that, cassette-mode research transports find
nothing by construction, so no forecast could clear verification anyway (issue
#438). What the ledger actually accumulates is screen decisions, risk-kernel
cash/position reconciliations (`VerificationPassed` is that, **not** citation
verification), equity samples, position snapshots and heartbeats — a live,
non-empty hash chain, and not a trade.

**systemd**

`deploy/systemd/` ships one unit per process —
`windbreak-{pipeline,riskkernel,order-gateway,dashboard}.service` — each with
`Restart=on-failure`, `RestartSec=5s` and `StartLimitIntervalSec=0` (without
the last two, five failures in ten seconds leave a unit permanently `failed`,
which is the one outcome an unattended deployment cannot afford). Units are
install-prefix-agnostic: `ExecStart=/usr/bin/env windbreak run --process
<name>` resolves `windbreak` from `PATH` rather than a hardcoded install path.

Unlike the compose skeleton the units have no image to carry the repo, so they
expect one installed at `/opt/windbreak` (the pipeline unit's
`WorkingDirectory=`), against which its repo-relative fixture paths resolve —
the same invocation [`docs/RUNBOOK.md`](docs/RUNBOOK.md) documents.

They also have no image to carry the *state* directories. The pipeline unit
declares `StateDirectory=windbreak/ledger windbreak/reports`, which is what
creates `/var/lib/windbreak/{ledger,reports}` before `ExecStart` — the same two
directories the `Dockerfile` pre-creates for the compose skeleton. Nothing else
would: `--ledger-path`'s parent directory must already exist (the `.db` and its
`-wal` sibling are created on first use, the directory is not), so without it
the unit dies with `sqlite3.OperationalError: unable to open database file` on
its first config load and then, with the start rate limit removed, retries
forever writing nothing. `StateDirectory=` is preferred to an
`ExecStartPre=/bin/mkdir`: systemd creates the directories itself and re-asserts
their ownership on each start, handing them to whatever `User=` the unit
declares — the units declare none, so they run as root and own them as root,
but an operator who adds one needs no accompanying `chown`. The
dashboard unit reads its bearer token from a mandatory
`EnvironmentFile=/etc/windbreak/dashboard.env` containing one
`WINDBREAK_DASHBOARD_TOKEN=<token>` line; the path is deliberately not
`-`-prefixed, so a missing file fails the unit rather than starting a
tokenless dashboard.

**Dashboard**

`windbreak run --process dashboard` boots the dashboard server
(`windbreak.dashboard.app`), which binds `127.0.0.1` only — never a public
interface, and not configurable — on `config.dashboard.port` (default `8080`,
matching the `127.0.0.1:8080` compose publish). Every request must present a
bearer token (`Authorization: Bearer <token>`) read from the
`WINDBREAK_DASHBOARD_TOKEN` environment variable — never from config, since
config is ledgered and a secret there would leak into the hash chain; a
missing or blank token exits the process with code 1 — so the compose file
declares the variable in Compose's required form (`${WINDBREAK_DASHBOARD_TOKEN:?...}`),
which refuses the whole invocation before any container exists rather than
letting the service crash-loop under `restart: on-failure`. Pass
`--ledger-path` to back the status line and read-model views (positions,
equity, decisions, ...) with a live ledger; without it, `/` reports
`RESEARCH` / `never` and every view renders its "no data yet" placeholder.

**Two things the shipped skeletons deliberately do not claim.** Neither
skeleton passes `--ledger-path` to the dashboard, so it renders those
placeholders: `_load_and_ledger_config` opens the ledger read-write for any
process given that flag, and `SqliteLedgerStore` runs `PRAGMA
journal_mode=WAL` plus a `CREATE TABLE` on open, so the flag is fatal against
the read-only ledger mount SPEC §5.1 asks for. And the dashboard binds the
*container's* `127.0.0.1`, which a published port cannot reach, so the
compose stack's `127.0.0.1:8080` publish does not currently make it
reachable from the host. Both need a change in `windbreak/`, not in
`deploy/`. Until then the dashboard is reachable only from inside its own
container or from a systemd host running the unit directly.

The dashboard is **read-only as
shipped**: it exposes no working mutation surface at all — pause, kill,
acknowledge and raise-floor all arrive with later epics, and the single write
route the handler defines is an unwired seam the CLI never supplies a granter
for. Use `windbreak ack` to grant an acknowledgement. The route table in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md#observing-via-the-dashboard) is the one
canonical, test-pinned statement of what each route answers — this README
deliberately restates none of it.

This is still an M0 skeleton. The tracer `windbreak run` (no flags) idles in
`RESEARCH` mode emitting heartbeats; the shipped `pipeline` service and unit
are what supply the flags that make it a PAPER loop instead.

### Ralph fleet loop

The repo includes the opt-in Ralph autonomous fleet loop (`.claude/commands/ralph-tick.md`, `scripts/ralph/`, maintenance-scan workflows). It assumes a GitHub-hosted issue/PR backlog and git worktrees, and requires manual secret/label setup — see `scripts/ralph/FLEET.md` and `scripts/ralph/PROMPT.md`.

## Attribution

Generated with [Start Green Stay Green](https://github.com/Geoffe-Ga/start_green_stay_green) — maximum-quality Python projects from day one.
