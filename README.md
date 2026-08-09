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
must never kill another. `deploy/` ships two equivalent skeletons for this at
M0.

**docker compose**

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Starts four services — `pipeline`, `riskkernel`, `order-gateway`, `dashboard`
— each built from the repo-root `Dockerfile` and running
`windbreak run --process <name>`, with `restart: on-failure`. Only `dashboard`
publishes a port, bound to `127.0.0.1:8080` (SPEC §14: no public inbound), and
its ledger mount is read-only since it holds no trade authority. Process
isolation in practice:

```bash
$ docker compose -f deploy/docker-compose.yml kill pipeline
$ docker compose -f deploy/docker-compose.yml ps --format '{{.Name}} {{.State}}'
windbreak-pipeline       exited
windbreak-riskkernel     running
windbreak-order-gateway  running
windbreak-dashboard      running
```

**systemd**

`deploy/systemd/` ships one unit per process —
`windbreak-{pipeline,riskkernel,order-gateway,dashboard}.service` — each with
`Restart=on-failure`. Units are install-prefix-agnostic:
`ExecStart=/usr/bin/env windbreak run --process <name>` resolves `windbreak`
from `PATH` rather than a hardcoded install path.

**Dashboard**

`windbreak run --process dashboard` boots the dashboard server
(`windbreak.dashboard.app`), which binds `127.0.0.1` only — never a public
interface, and not configurable — on `config.dashboard.port` (default `8080`,
matching the `127.0.0.1:8080` compose publish). Every request must present a
bearer token (`Authorization: Bearer <token>`) read from the
`WINDBREAK_DASHBOARD_TOKEN` environment variable — never from config, since
config is ledgered and a secret there would leak into the hash chain; a
missing or blank token exits the process with code 1. Pass `--ledger-path` to
back the status line and read-model views (positions, equity, decisions, ...)
with a live ledger; without it, `/` reports `RESEARCH` / `never` and every
view renders its "no data yet" placeholder. Mutations (pause, kill,
acknowledge, raise floor) beyond the existing `POST /ack` arrive with later
epics.

This is an M0 skeleton: the tracer `windbreak run` (no flags) still just idles
in `RESEARCH` mode, emitting heartbeats.

### Ralph fleet loop

The repo includes the opt-in Ralph autonomous fleet loop (`.claude/commands/ralph-tick.md`, `scripts/ralph/`, maintenance-scan workflows). It assumes a GitHub-hosted issue/PR backlog and git worktrees, and requires manual secret/label setup — see `scripts/ralph/FLEET.md` and `scripts/ralph/PROMPT.md`.

## Attribution

Generated with [Start Green Stay Green](https://github.com/Geoffe-Ga/start_green_stay_green) — maximum-quality Python projects from day one.
