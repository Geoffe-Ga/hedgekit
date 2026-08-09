# windbreak Runbook

Operational instructions for running and observing windbreak. This runbook
grows with the project; today it covers the always-on PAPER loop shipped in
issue #48.

## Running the PAPER loop

### Prerequisites / config

The PAPER loop is one per-beat hook inside `windbreak run`'s existing RESEARCH
heartbeat loop (`windbreak/main.py`). It activates only when **both** of these
hold, checked by `_paper_activated`:

1. The active configuration's `mode_ceiling` (SPEC S16) permits `PAPER` --
   i.e. `Mode.from_config(config.mode_ceiling) is not Mode.RESEARCH`. The
   built-in default configuration (`windbreak.config.load_default_config`,
   used whenever `--config` is omitted) already ships `mode_ceiling: "paper"`,
   so no custom config is required to satisfy this condition.
2. **All four** of the following `run` flags are supplied together:

   | Flag | Meaning |
   |------|---------|
   | `--paper-books-dir` | Paper-exchange fixture directory (books/markets/fees) loaded via `PaperExchange.from_fixture_dir`. |
   | `--cassette-path` | Recorded LLM cassette file for the offline forecast-replay transport (`ReplayCassette.from_path`). |
   | `--ledger-path` | Path to the PAPER loop's hash-chained SQLite ledger database (a sibling `<name>.wal` file backs its write-ahead log). |
   | `--report-dir` | Directory the weekly report stub is written into. |

If the ceiling forbids PAPER, or even one of the four flags is missing, none
of this is wired: `windbreak run` falls back to its plain RESEARCH heartbeat
(optionally with `--snapshot-fixture-dir` snapshotting, if given) -- **byte
identical to today's behavior**. This "all four flags or nothing" gate is the
tracer invariant: partially-flagged or ceiling-mismatched invocations can
never half-activate PAPER.

**RESEARCH is the safe default.** Omitting the four PAPER flags (or setting
`mode_ceiling: research`) is always a safe, side-effect-free way to run
`windbreak run` -- no ledger is created, no paper exchange is touched.

### Starting the loop

```bash
windbreak run \
  --paper-books-dir tests/fixtures/books/deep_walk \
  --cassette-path /path/to/cassette.json \
  --ledger-path /path/to/state/ledger.db \
  --report-dir /path/to/state/reports \
  --heartbeat-interval 5
```

- `--cassette-path` must point at an existing recorded-cassette JSON file. An
  empty cassette (`{}`) is a valid, offline-safe placeholder as long as the
  forecast pipeline's research stage never actually reaches the LLM transport
  (it abstains first on zero verified citations when no research tools are
  wired) -- see `tests/integration/conftest.py` for the pattern this mirrors.
- `--ledger-path`'s parent directory must exist; the ledger file and its
  `.wal` sibling are created on first use.
- Stop the loop with `Ctrl-C` (SIGINT) or SIGTERM; it shuts down cleanly and
  logs the shutdown reason. `--max-beats N` stops it automatically after `N`
  heartbeats (useful for a bounded smoke run).

### What one PAPER tick actually does

Each beat runs one `windbreak.scheduler.loop.run_single_tick` pass over the
*real* (unmodified) components, per SPEC S5.3's SINGLE order path:

```
snapshot -> forecast -> select -> approve (seam) -> [only if a token mints]
route -> PaperExchange fill -> reconcile
```

Every stage appends an audit event to the shared hash-chained ledger, plus a
per-tick `ModeHeartbeat`, `EquitySampled`, and `PositionsSnapshotRecorded`.
Since issue #353 each tick also runs one **read-only verification cycle**
before it decides anything, which records exactly one `VerificationPassed`,
`VerificationDrift`, or `VerificationMismatch` row. The weekly report stub
(below) is also (re-)written each tick.

**Known limitation -- today's tick still never fills, but no longer for any
verification reason.** The `approve` stage composes the real
`RiskKernel.evaluate_intent` with the real `ApprovalPipeline.approve`
(`KernelApproval` in `windbreak/scheduler/loop.py`). Three earlier causes have
been removed: issue #340 made `jurisdiction_product_eligibility` a real check,
issue #342 wired real exchange-status and pipeline-heartbeat evidence, and
issue #353 wired the read-only verification cycle so the three reconciliation
checks evaluate a real snapshot and pass on a clean one.

The two remaining causes are both honest zero/`None` feeds in the *account and
market view* `windbreak/scheduler/loop.py` composes, not in the kernel:

- `daily_loss_limit` vetoes because `equity_start_of_day` is `0`, which floors
  the loss threshold at `0`; a flat account's `realized_loss_today` of `0`
  already "reaches" it.
- `participation_cap_compliance` vetoes because `visible_depth` is `None`.

Both are left honest on purpose: inventing a start-of-day equity or a depth
figure would loosen two real exposure limits on fabricated evidence.
`tests/integration/test_paper_verification.py::test_loop_production_context_vetoes_carry_no_verification_reason`
pins this exact remaining reason set, so it cannot drift unnoticed.

Note the status **value** is read from the connector every tick and never
synthesized, so a `paused` or `closed` exchange still vetoes -- with the
distinct reason `exchange not open for trading` rather than `stale or missing`,
so an operator can tell "the venue is shut" from "we have no reading".

So a real PAPER tick ledgers a full decision trail (snapshot, forecast,
selector decision, verification outcome, and an `IntentVetoed`) but routes
nothing and fills nothing; `filled_centis` on every tick's outcome is `0`.
Don't be surprised to see nothing but vetoes in `/decisions` or
`selector_decisions.json` -- that is the expected, honestly-ledgered state of
the loop today.

**Operationally important -- a verification breach HALTs the kernel.** The
baseline the cycle reconciles against is frozen at process start from the
venue's own opening state, and the ledger records no fill amounts that could
update it. So the first tick after any real paper fill grades a `BREACH`: the
kernel transitions to `HALT` (issue #32), records `VerificationMismatch` and
`VerificationMismatchHalt`, the per-tick `ModeHeartbeat` starts reporting
`HALT` instead of `PAPER`, and `TickOutcome.kernel_halted` is `True`. Every
later approval then vetoes on the halted mode. **Watch `kernel_halted`**: a
halted loop keeps ticking and keeps ledgering, but it is no longer a trading
loop, and only a restart re-baselines it. That is the fail-closed reading of
"our books cannot account for the venue" -- the alternative, re-reading the
expectation off the same connector each cycle, would make all three
reconciliation dimensions structurally incapable of failing.

The verification path holds a `ReadOnlyConnectorView`
(`windbreak/connector/readonly.py`), not the `PaperExchange` itself, so it has
no `place_order`/`cancel_order` attribute at all (SPEC S1.1 invariant 3).

**Known limitation -- the kill switch does not stop the PAPER loop yet.**
`windbreak kill --state-dir <dir>` and `windbreak rearm --state-dir <dir>` write
and clear a `KILL`/`REARM` file, but the PAPER loop's `RiskKernel` is
constructed with `kill_integration=None` (`windbreak/scheduler/loop.py`), so no
kill-file watcher is polled. To stop the loop today, stop the process itself
(`Ctrl-C`/SIGINT or SIGTERM).

### Acknowledging a held order (LIVE_MICRO / LIVE)

In the live modes, an order whose worst-case cost exceeds
`risk.require_human_ack_above_micros` is **held** — not routed — until an
operator explicitly acknowledges it (SPEC S10.8). Each held order opens a
pending acknowledgement with a single-use, unguessable 32-hex `approval_id` and
a ttl; if nobody acknowledges it before the ttl, the approval lapses and its
capital reservation is released. Every request, grant, and lapse is ledgered.

Two operator paths grant an acknowledgement, both drop-box based (they work with
the dashboard HTTP surface down, mirroring `kill`/`rearm`):

```bash
windbreak ack --approval-id <32-hex-approval-id> --state-dir <dir>
```

writes `<dir>/acks/<approval_id>`, which the kernel's ack-file watcher grants on
its next beat and then removes. The `--approval-id` must be exactly 32 lowercase
hex characters (the shape the kernel mints); any other token is rejected as a
usage error before a file is written. Alternatively, `POST /ack` on the
dashboard (below) grants the same acknowledgement over the authenticated
loopback surface. As with the kill switch, the live loop that polls the ack
drop-box is not wired yet — this verb writes the durable grant signal a future
live loop consumes.

### Observing via the dashboard

`windbreak.dashboard.app` serves a read-only, loopback-only HTTP surface:

- Binds `127.0.0.1` only (never a public interface -- not configurable, per
  SPEC S14).
- Every route requires `Authorization: Bearer <token>` (timing-safe compared
  against the token `create_server(token=...)` was built with); a
  missing/wrong token gets a `401` with a `WWW-Authenticate: Bearer`
  challenge.

Routes:

| Path | Renders |
|------|---------|
| `/` | Current mode and last-heartbeat status. |
| `/positions` | The latest open-positions snapshot. |
| `/equity` | The equity curve vs. the configured capital floor. |
| `/decisions` | The interleaved selector decisions, including skip/veto reasons. |
| `/providers` | The fleet-observability provider panel: one summary per provider (id, canary status; resolved count and Brier skill from the #194 track-record fold; abstention rate and per-provider `cost_per_forecast` from the #281 per-provider vote-cost fold) plus a fleet-wide cost-per-forecast line. Any figure falls back to `n/a` only for a provider its respective fold does not (yet) cover. See [Provider operations](#provider-operations) below. |
| `GET /acks` | The pending human acknowledgements awaiting an operator (SPEC S10.8). |
| `POST /ack` | Grant a pending acknowledgement — JSON body `{"approval_id": "<32-hex>"}`. |

`POST /ack` is the dashboard's only write surface: it shares the same bearer
gate as every read route (an unauthenticated post gets a `401` and never
reaches the granter), 404s when `create_server` was built with no `ack_granter`
seam wired, and rejects a malformed, oversized, or non-32-hex body with a `400`
before invoking the granter. It is enabled only when both `ack_granter` and
`pending_acks_source` are passed to `create_server`; the default build exposes
neither route.

`windbreak run --process dashboard` is the primary operator path (issue #79).
The bearer token is read only from the `WINDBREAK_DASHBOARD_TOKEN` environment
variable -- never from config, since config is ledgered and a secret would
leak into the hash chain -- and a missing or blank value fails closed with a
`FATAL` log and exit code 1. The listen port comes from `config.dashboard.port`
(default `8080`); the host is always the loopback `127.0.0.1` and is not
configurable. Passing `--ledger-path` wires the status line and every
read-model view to that ledger (the same one `windbreak rebuild` projects);
omit it and `/` reports `RESEARCH` / `never` with every view rendering its "no
data yet" placeholder:

```bash
export WINDBREAK_DASHBOARD_TOKEN=replace-with-a-real-secret
windbreak run --process dashboard --ledger-path /path/to/state/ledger.db
```

Embedding the server directly in a library caller -- bypassing the CLI
entirely -- is also supported via `create_server`:

```python
from pathlib import Path

from windbreak.dashboard.app import create_server
from windbreak.dashboard.views import build_ledger_read_models_source

server = create_server(
    token="replace-with-a-real-secret",
    status_source=lambda: ...,  # wire to a real status source
    read_models_source=build_ledger_read_models_source(Path("/path/to/state/ledger.db")),
    port=8765,
)
server.serve_forever()
```

Until the loop has ledgered data, `/positions`, `/equity`, and `/decisions`
each render a plain "No data yet." placeholder rather than an empty table or
an error -- this is the documented behavior, not a bug. Passing no
`read_models_source` at all (the default) renders that same placeholder on
every view unconditionally.

### Observing via ledger read models

`windbreak rebuild` folds a verified ledger into a set of byte-stable JSON
read-model files -- the same projection functions the dashboard reads live:

```bash
windbreak rebuild --ledger-path /path/to/state/ledger.db --output-dir /path/to/state/read-models
```

This writes (or overwrites) eleven files into `--output-dir`:

- `config_versions.json` -- every `ConfigLoaded` event.
- `mode_history.json` -- every `ModeHeartbeat` event.
- `gateway_events.json` -- the chronological Order Gateway / crash-recovery
  event trail.
- `positions.json` -- the latest `PositionsSnapshotRecorded` snapshot (at
  most one entry).
- `equity_curve.json` -- every `EquitySampled` sample, in ledger order.
- `selector_decisions.json` -- the interleaved `SelectorDecisionRecorded` /
  `IntentApproved` / `IntentVetoed` trail, in ledger order.
- `execution_quality.json` -- every `ExecutionQualityRecorded` row, in ledger
  order (issue #58).
- `live_divergence.json` -- the interleaved `LiveDivergenceSampled` /
  `LiveDivergenceBreached` trail, in ledger order (issue #58).
- `canary_status.json` -- the LATEST `CanaryVerdictRecorded` per provider,
  keyed at that provider's first-seen list position (issue #195; see
  [Provider operations](#provider-operations) below).
- `forecasts.json` -- every `ForecastCreated` row, in ledger order (issue
  #195), feeding the fleet cost-per-forecast/abstention fold.
- `provider_vote_costs.json` -- the per-provider vote-cost aggregate folded
  from `ProviderVoteRecorded` events (issue #281), feeding the `/providers`
  panel's real per-provider `cost_per_forecast` and `abstain_rate`.

`rebuild` verifies the ledger's hash chain before projecting; a corrupted
chain fails closed with a nonzero exit code and the offending sequence number
on stderr, rather than silently emitting a plausible-but-wrong projection.

### Anchoring and verifying against tail-rewrite tampering (issue #75)

`verify_chain`/`rebuild` prove the ledger is *internally* consistent, but
cannot distinguish a legitimately short chain from one whose tail a writer
with raw database access truncated and re-chained -- both verify cleanly.
Head-hash anchoring closes that gap: `windbreak anchor` appends the chain's
current head `(sequence_number, event_hash)` to an append-only, JSON-lines
anchor file, and `windbreak verify` independently checks the live chain
against every anchor recorded so far.

```bash
windbreak anchor --ledger-path /path/to/state/ledger.db --anchor-path /path/to/anchors/ledger.anchors.jsonl
windbreak verify --ledger-path /path/to/state/ledger.db --anchor-path /path/to/anchors/ledger.anchors.jsonl
```

Both verify the hash chain first (a corrupted chain fails closed with the
offending sequence number, exactly like `rebuild`); `windbreak anchor` is a
silent no-op against an empty ledger, and never anchors a broken chain.
`windbreak verify` additionally fails closed if the anchor file is missing,
empty, or holds a malformed line, and reports the first anchored position
whose live hash no longer matches -- or has vanished entirely -- as a
tail-rewrite mismatch on stderr.

**Trust boundary.** The anchor file only relocates the trust root; it does
not eliminate it. The guarantee holds only while the anchor file is
protected from whoever can write to the ledger database -- a writer with
access to *both* can truncate the chain, re-chain a forged tail, and append a
fresh anchor pinning the forged head, and both commands would pass. Put the
anchor file on a separately-permissioned volume, an append-only/write-once
medium, or a remote/off-host sink the ledger writer cannot reach; anchoring
next to the ledger under the same principal only catches accidental
corruption, not a determined local attacker.

### Weekly reports

Each PAPER tick calls `windbreak.reports.weekly.maybe_write_weekly`, which
writes at most one `weekly-YYYY-MM-DD.md` file per ISO calendar week into
`--report-dir` (idempotent: repeated calls within the same ISO week return the
already-written file untouched). The stub carries markdown section headers
(`Equity vs floor`, `Positions`, `Decisions`) each with a `No data yet.` body
-- populating the real bodies from ledgered data is a later documentation
pass.

### Known limitations (summary)

- The real Risk Kernel currently vetoes every intent, but no longer because of
  a stub or of any missing evidence feed in the kernel: issue #340 made
  `jurisdiction_product_eligibility` a real check, issue #342 wired real
  exchange-status and pipeline-heartbeat evidence, and issue #353 wired the
  read-only verification cycle the three reconciliation checks consume. The two
  remaining causes are honest zeros in the loop's own account/market view --
  `equity_start_of_day=0` (so `daily_loss_limit` vetoes) and
  `visible_depth=None` (so `participation_cap_compliance` vetoes). So no PAPER
  tick fills yet: expect vetoes, not fills, in the ledger and dashboard.
- A verification `BREACH` HALTs the kernel and it stays halted for the life of
  the process (see "What one PAPER tick actually does"). Watch
  `TickOutcome.kernel_halted` and the `ModeHeartbeat` mode.
- On a **live Kalshi** path the jurisdiction check vetoes for an additional
  reason: Kalshi publishes no eligibility signal, so `normalize_market` stamps
  every market `jurisdiction_status="unknown"`, which fails closed by design
  (SPEC §20 Q3, unresolved). Only a market carrying real eligibility metadata --
  as the paper fixture books do -- can clear that check.
- `windbreak kill`/`windbreak rearm` do not stop or gate the PAPER loop today
  (`kill_integration=None`); use process signals to stop the loop.
- `windbreak run --process dashboard` boots the HTTP dashboard server directly
  (issue #79); its bearer token comes only from `WINDBREAK_DASHBOARD_TOKEN`
  and its port only from `config.dashboard.port` -- there is no `--port` or
  `--token` CLI flag.
- Weekly reports are structural stubs (`No data yet.` bodies); the real
  report content is a later pass.

## Operator alerts

Alerts reach you only through the sinks `config.alerts` declares. Until you
declare one, every alert falls back to the log-only sink: it appears on stderr
as a JSON `AlertEmitted` line and **nowhere else**. That is the shipped default
(`alerts.sinks` holds one ntfy entry whose `topic_env` is still the
`configured-by-operator` placeholder), so treat configuring a real sink as a
prerequisite for any unattended run.

### Configure a sink

Each entry needs `type` plus the fields that type uses, and every destination
host must **also** be declared in `alerts.allowed_hosts` — the outbound egress
allowlist is deliberately separate from the destination, so a mistyped URL
cannot open egress by itself.

**An ntfy topic and a webhook URL never go in the config file.** They are bearer
capabilities, and every config value is written verbatim into the append-only
ledger and into `config_versions.json`. Instead the config names the environment
variable each one is read from — `topic_env`, `base_url_env`, `url_env` — the
same indirection `research.search_api_key_env` already uses:

```yaml
alerts:
  allowed_hosts: [ntfy.sh, hooks.example.com, smtp.example.com]
  sinks:
    - type: ntfy
      base_url_env: WINDBREAK_NTFY_BASE_URL
      topic_env: WINDBREAK_NTFY_TOPIC
    - type: webhook
      url_env: WINDBREAK_WEBHOOK_URL
    - type: smtp
      smtp:
        host: smtp.example.com
        port: 587
        sender: windbreak@example.com
        recipients: [ops@example.com]
```

```bash
export WINDBREAK_NTFY_BASE_URL=https://ntfy.sh
export WINDBREAK_NTFY_TOPIC=your-private-topic     # a bearer capability
export WINDBREAK_WEBHOOK_URL=https://hooks.example.com/services/TOKEN
```

Naming a variable you never export is **fatal**, not a skip: startup exits 1
with a `FATAL:` line naming the variable (never a value). A sink you have not
wired at all — the field left at `configured-by-operator` — is the skip case.

The `smtp` block stays in the config file on purpose: `host` has to be repeated
in `allowed_hosts` anyway, and `sender`/`recipients` are mailbox addresses, not
credentials. See SECURITY.md, "Alert destinations never live in configuration".

`desktop` is also a valid type, but only for a process that supplies a desktop
notifier; the CLI does not, so a `desktop` entry makes it exit 1 rather than
pretend it can notify you.

### Verify delivery

```bash
windbreak alert-test mode-change --config /path/to/windbreak.yaml
```

Read the `AlertEmitted` line's per-sink outcomes on stderr. `"ntfy=ok:True"`
means it was delivered; a `log-only` outcome means no sink was built or every
sink failed. Startup **fails closed** (exit 1, `FATAL:` on stderr) when a sink
names an unknown type, names a destination environment variable that is unset or
empty, targets a host missing from `allowed_hosts`, or cannot deliver as
configured — a half-wired alerting path is never silently downgraded.
A sink you have not finished filling in is skipped with a WARNING naming its
type only; no topic or webhook URL is ever logged.

## Provider operations

Fleet-observability provider canaries (issue #195, SPEC S8.4/S16 extended
per-provider) run one small reference battery per forecast provider and
ledger every verdict, so silent per-provider answer drift or forecaster
version drift is caught before it poisons a live forecast. The battery is
driven entirely by the operator-run `scripts/run-canaries.sh` (a thin wrapper
over `scripts/run_canaries.py`, which owns every `requests`/environment
access on this path -- CI never dials a live forecaster) -- never by CI, and
never by the PAPER/live heartbeat loop itself (see the known limitation at
the end of this section).

A battery is described by a `--spec-file` JSON document: a `"providers"` list,
each entry carrying `provider`, `questions` (a list of
`{"question_id", "prompt", "reference_ppm"}` objects), `pinned_versions` (the
operator-accepted forecaster version strings for that provider), and either an
`"observation"` object (`{"observed_ppm": {...}, "reported_version": "..."}`,
replay mode) or an `"endpoint"`/`"host"` pair (record mode; the outbound URL
must resolve to `host` exactly, or the run fails closed with
`EgressDeniedError`).

### Rotate provider keys

In `--record` mode, each provider's live API key is read from its
`<PROVIDER>_API_KEY` environment variable (the provider identifier
upper-cased plus the `_API_KEY` suffix, e.g. `FUTURESEARCH_API_KEY` --
`scripts/run_canaries.py`'s `_API_KEY_ENV_SUFFIX` constant), injected as an
`x-api-key` send-time HTTP header, and never printed, logged, or written to
any cassette or ledger row.

1. Export the new key under that exact variable name -- never a literal in
   any command, script, or commit:

   ```bash
   export FUTURESEARCH_API_KEY=replace-with-a-real-key
   ```

2. Validate the rotation by re-running that provider's battery in record
   mode, which dials the live endpoint once with the new key:

   ```bash
   scripts/run-canaries.sh --record \
       --spec-file provider_canaries.record.json \
       --ledger-path var/ledger.db
   ```

3. Confirm the process exits `0` and prints `provider=<name> canary=OK
   drift_score_ppm=<n>` for the rotated provider (the exact
   `provider=<p> canary=<STATUS> drift_score_ppm=<n>` line every verdict
   prints). A wrong or expired key surfaces as a live HTTP failure from the
   provider's endpoint, not a silent pass; a genuine drift line (`ANSWER_DRIFT`
   / `VERSION_DRIFT`) means the key worked but the provider itself drifted --
   treat that as [drift](#respond-to-canary-drift--provider-version-drift), not
   a rotation failure.
4. Revoke the old key at the provider's own dashboard once the new key is
   confirmed working; this script cannot do that for you.

Never echo the key value in any of the above -- the script deliberately never
prints it either (`scripts/run-canaries.sh`'s own `--record` banner names the
required variable, not its value).

### Respond to canary drift / provider version drift

A drift breach dispatches one `AlertType.CANARY_DRIFT` alert and ledgers one
`CanaryVerdictRecorded` event (`status` one of `OK` / `ANSWER_DRIFT` /
`VERSION_DRIFT`). Running via `scripts/run-canaries.sh`, the alert prints to
stderr as `ALERT AlertType.CANARY_DRIFT: <message>` and the process exits `1`
the moment any provider drifts:

- An answer-drift message reads `Provider <p> answer-drift: Canary drift <n>
  ppm exceeded tolerance <t> ppm; worst question <id>`.
- A version-drift message reads `Provider <p> version-drift: reported
  forecaster version '<v>' is off the pinned set [...]`.

1. **Read the alert** to identify the provider and the drift kind
   (`answer` vs. `version`).
2. **Inspect durable state** -- either the ledger read models:

   ```bash
   windbreak rebuild --ledger-path var/ledger.db --output-dir var/read-models
   cat var/read-models/canary_status.json    # latest verdict per provider
   ```

   or the live dashboard's `/providers` route, started with
   `windbreak run --process dashboard --ledger-path var/ledger.db` and
   bearer-gated via `WINDBREAK_DASHBOARD_TOKEN` (see
   [Observing via the dashboard](#observing-via-the-dashboard) above), which
   folds the same `canary_status.json` / `forecasts.json` projections through
   `render_provider_panel`.
3. **For VERSION drift**, decide: accept the new version by adding it to that
   provider's `pinned_versions` list in the canary `--spec-file`, or treat it
   as a vendor regression and investigate upstream before accepting anything.
   (FutureSearch's *live* forecaster has its own, separate pin --
   `config.forecast.futuresearch.pinned_forecaster_versions` -- which gates
   real forecasts, not the canary battery; update it too if the version bump
   is legitimate for live forecasting, not only for canaries.)
4. **For ANSWER drift**, investigate the underlying prompt/response
   regression (a silent vendor model swap not reflected in the reported
   version, a prompt-template change, etc.).
5. **Re-run the battery until it exits `0`**:

   ```bash
   scripts/run-canaries.sh --spec-file provider_canaries.json --ledger-path var/ledger.db
   ```

   every printed line should read `canary=OK`.

Per SPEC S8.6, `CanaryGate.is_live_blocked` is fail-closed and never
auto-adapts: a drifting provider is blocked from live eligibility until an
operator acknowledges it (`CanaryGate.acknowledge`), and a breach on the
*global* (pinned-canary-model) dimension blocks every provider closed, not
just the one that drifted -- a provider query ORs its own window with the
global one.

**Known limitation -- no persistent, wired gate to acknowledge against yet.**
`scripts/run_canaries.py`'s CLI never passes a `gate=` argument to
`windbreak.scheduler.canaries.run_canaries`, so each invocation of
`scripts/run-canaries.sh` scores against a brand-new, in-memory `CanaryGate()`
-- there is no cross-run block state, and therefore nothing to acknowledge via
this script. `CanaryGate.acknowledge()` is a real, tested primitive (used
directly by the test suite) that a future, persistent composition root will
drive, but no `windbreak` CLI verb or dashboard route calls it today. The
practical operator loop today is exactly steps 3-5 above: fix the root cause
(or accept the version), then re-run the battery until every provider reads
`canary=OK`. Separately, `windbreak.forecast.pipeline.run_pipeline`'s own
`canary_gate` seam is real and unit-tested, but `windbreak/scheduler/loop.py`'s
PAPER-tick `_forecast_stage` calls `run_pipeline(...)` without a `canary_gate`
argument -- so canary drift does not yet block a live PAPER-loop tick
end-to-end; today the battery is a standalone, operator-run detector, not (yet)
an in-loop gate.

### Prove a provider before it can back a live order (issue #305)

Unlike the canary gate above, the *per-provider track-record* gate **is** wired
into the PAPER tick: `build_paper_deps` builds one `ProviderTrackRecordGate`
per process and `_forecast_stage` passes it into every `run_pipeline` call, so
a provider that has not earned live eligibility cannot produce a live-eligible
forecast in the loop. Its votes still run and still cost -- that is the only
way a paper track record ever accrues -- but the record's `eligible_for_live`
is forced `False`.

The gate reads one strict-JSON artifact, `provider-track-records.json`, from
the same `--report-dir` the loop writes its weekly reports to, and compares
each provider against `config.forecast.provider_gate.min_resolved` (default
150) and `config.forecast.provider_gate.min_brier_skill_ppm` (default 10000):

```json
{
  "anthropic": {"resolved_count": 210, "brier_skill_ppm": 14000},
  "openai":    {"resolved_count": 180, "brier_skill_ppm": 11500}
}
```

Both boundary cases fail closed. **No artifact yet** is the bootstrap case:
every provider is unproven and every full forecast is held, which is the
correct reading of "no measured edge yet". **A malformed artifact** aborts
startup with a `ValueError` naming the offending leaf -- a broken evaluation
pass must not be indistinguishable from a genuinely empty one. Values are
integers throughout: a fractional `brier_skill_ppm` is rejected, not truncated.

Each hold appends exactly one `ProviderGateHeld` row (component `scheduler`)
after that tick's `ProviderVoteRecorded` rows, carrying `forecast_id`,
`market_ticker`, the comma-joined `unproven_providers`, their `unproven_count`,
and the `min_resolved` / `min_brier_skill_ppm` bars actually in force. To see
why a healthy-looking forecast was not live-eligible:

```bash
python -c "
from pathlib import Path
from windbreak.ledger.store import SqliteLedgerStore
for r in SqliteLedgerStore(Path('var/ledger.db')).read_all():
    if r.event_type == 'ProviderGateHeld':
        print(r.sequence_number, r.payload_json)
"
```

If a provider you expect to be proven is named there, the artifact is stale (or
missing) rather than the provider being bad -- re-run the evaluation pass that
writes it. Raising or lowering the bars is a config edit to
`config.forecast.provider_gate`; there is no runtime lever, and the bars in
force at the time of each decision are on the row itself.

### Respond to budget exhaustion

`windbreak.forecast.budget.ResearchBudget` enforces SPEC S16's three research
spend ceilings, mirrored in config at `config.forecast.budget.per_forecast_micros`
(default 3,000,000 micros / $3), `config.forecast.budget.per_day_micros`
(default 20,000,000 micros / $20), and `config.forecast.budget.max_pages`
(default 20 pages per forecast). Two events name the two ways a run can be
halted:

- `BUDGET_DAY_EXHAUSTED` -- `ResearchBudget.ensure_day_open` halts a run
  **before any research is attempted** once the current UTC day's cumulative
  spend is at or above the per-day ceiling; raises `DailyBudgetExhaustedError`.
- `BUDGET_FORECAST_EXCEEDED` -- `ResearchBudget.charge_forecast` charges a
  single forecast's cost into the day bucket **first** (so a breaching
  forecast still counts against the day), then raises
  `PerForecastBudgetExceededError` only if that forecast's own cost *strictly
  exceeds* the per-forecast ceiling (an exactly-equal cost passes).

The day bucket is keyed by the run's UTC calendar date
(`datetime.astimezone(UTC).date().isoformat()`) -- it resets, not decays, at
each UTC midnight; there is no manual reset lever and no partial-day rollover.

When either error is raised: check which UTC day is exhausted (the error's
`utc_day` field) and how much was spent (`spent_micros`/`cost_micros` vs.
`budget_micros`); a day-exhaustion halt clears itself automatically at the
next UTC midnight, while a per-forecast breach is a signal to look at that
one forecast's research cost (an unusually expensive research stage, a
runaway page-fetch loop bounded by `max_pages`, etc.) rather than the whole
day's spend.

**How the loop enforces this (issue #339).** `build_paper_deps` constructs
**one** `ResearchBudget` per process from `config.forecast.budget` and carries
it on `PaperTickDeps`, so its per-UTC-day spend bucket accumulates across every
tick that bundle runs. There is deliberately no `budget` parameter on
`build_paper_deps`: config is the single source, so there is no injection door
through which an unlimited or absent budget could arrive.

A halted tick appends exactly one `ResearchBudgetHalted` ledger row (component
`scheduler`) carrying `halt_kind` (`per_day` or `per_forecast`), `utc_day`,
`spent_micros`, and `budget_micros`. It then **skips the forecast, select, and
approve stages** but still emits its heartbeat, equity sample, positions
snapshot, and weekly report -- so a budget-halted loop stays observably alive
and flat rather than dying. Its ledger differs from a normal tick's by exactly
two absent rows: `ForecastCreated` and `SelectorDecisionRecorded`. No
`ForecastCreated` row is fabricated for a tick where the forecast engine never
ran; in a hash-chained audit ledger an honest gap beats an invented record.

**Operator arithmetic.** With the SPEC defaults -- a 20,000,000-micro day
ceiling against the fixed 3,000,000-micro per-forecast research charge -- a UTC
day permits **7 charged forecasts** before research halts for the rest of that
day. Raising the ceiling is a config edit to
`config.forecast.budget.per_day_micros`; there is no runtime lever and no
partial-day rollover.

**Caveat -- the day counter is process-local.** Day spend lives in memory on the
budget instance and is never persisted, so restarting the loop (crash, deploy,
manual bounce) resets the day's spend to zero and the ceiling can be re-spent.
This is fail-*open* across restarts and is the one gap in the guarantee above.
Durable day-spend rehydration is deliberately out of scope here, because the
obvious source -- summing `ForecastCreated.research_cost_micros` -- records a
fixed constant rather than the true charged amount, so it would under-count and
still fail open, just less visibly.

**Caveat -- a negative ceiling now aborts startup.** The YAML loader applies no
range checks, so a negative `per_forecast_micros`/`per_day_micros`/`max_pages`
reaches the composition root intact and `ResearchBudget` rejects it with a
`ValueError` while `--paper` is starting. That is the intended fail-closed
outcome: a loop that cannot determine its ceiling must not run. A **zero**
ceiling is legal and means "immediately exhausted" -- it silently halts all
research, which is fail-closed but easy to mistake for a hung loop.

### Add / remove a provider

**Adding** a provider to the canary battery:

1. Add its entry to the canary `--spec-file` JSON's `"providers"` list:
   `provider`, `questions` (one `{"question_id", "prompt", "reference_ppm"}`
   object per reference question), and `pinned_versions`.
2. Record and validate its first live observation once, in record mode (see
   [Rotate provider keys](#rotate-provider-keys) above for the key-export
   step) -- this is the only "recording" step provider canaries have; it is
   distinct from, and does not use, the LLM vote-ensemble cassette recorders
   (`scripts/record-cassettes.sh`, `scripts/record-research-cassettes.sh`,
   issues #191/#192), which record a different surface (ensemble-member vote
   completions, not provider canary endpoints):

   ```bash
   scripts/run-canaries.sh --record \
       --spec-file provider_canaries.record.json \
       --ledger-path var/ledger.db
   ```

3. Run the full battery in (the default, offline) replay mode and confirm the
   new provider's line reads `canary=OK`:

   ```bash
   scripts/run-canaries.sh --spec-file provider_canaries.json --ledger-path var/ledger.db
   ```

4. Confirm it appears:

   ```bash
   windbreak rebuild --ledger-path var/ledger.db --output-dir var/read-models
   ```

   then check `var/read-models/canary_status.json` for the new `"provider"`
   entry, or hit the dashboard's `/providers` route, or check the weekly
   report's `## Providers` section (`windbreak.reports.providers`'s
   `provider=<p> resolved=<n> ...` line) once that section is wired to real
   data (see the known limitation below).

**Retiring** a provider:

1. Remove its entry from the `--spec-file` so future battery runs stop
   appending fresh verdicts for it.

**Known limitation -- retirement leaves a stale, not an absent, entry.** The
`canary_status.json` fold (`canary_status_read_model`) is append-only and
keeps the LATEST verdict per provider ever ledgered; there is no tombstone or
removal event. A retired provider's last verdict therefore stays visible in
`canary_status.json` and on the `/providers` dashboard panel indefinitely --
"confirm it is gone" is not literally achievable with today's tooling.
Instead, treat a `canary_status.json` / `GET /providers` entry whose
`created_at` predates the retirement date as the retirement signal, until a
future retirement/tombstone mechanism ships. Similarly, the weekly report's
`## Providers` section (`windbreak.reports.providers.render_provider_lines`)
is a real, unit-tested renderer, but no production composition root supplies
its `provider_lines` argument yet (`windbreak/scheduler/loop.py` writes the
PAPER-loop's weekly report via the plain `maybe_write_weekly` stub path, not
`windbreak.evaluation.report.generate_weekly_report`) -- so today the weekly
report's `## Providers` section always renders its `No data yet.` fallback,
regardless of what the ledger holds.
