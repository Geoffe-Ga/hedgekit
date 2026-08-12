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

   A fifth flag, `--paper-live-ticker`, is optional and does *not* gate
   activation; it swaps the fixture books for the venue's live ones. See
   [Running against live venue books](#running-against-live-venue-books-issue-343).

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

### Running against live venue books (issue #343)

Adding one optional fifth flag points the loop at the exchange's **real,
current order books** while every fill, position, and balance stays simulated
-- real prices in, paper money out:

```bash
windbreak run \
  --paper-books-dir tests/fixtures/books/deep_walk \
  --paper-live-ticker KXFED-25DEC \
  --cassette-path /path/to/cassette.json \
  --ledger-path /path/to/state/ledger.db \
  --report-dir /path/to/state/reports
```

- **One flag, not two.** `--paper-live-ticker`'s *presence* selects live books
  and its *value* names the market, so the mode and the market cannot
  disagree. Omit it and the loop replays `--paper-books-dir`'s recorded books
  exactly as before -- byte identical.
- **`--paper-books-dir` is still required**, but in live mode only its
  *account* fixtures (`balances.json`, `balance_semantics.json`) are read.
  Paper money has to start somewhere; the market data comes from the venue.
- **One market.** A live session trades exactly the ticker you name. The
  multi-market universe is separate work (issue #345).
- **No credentials, and no order can leave.** The venue is reached through a
  read-only market-data view built by
  `windbreak.connector.live.build_kalshi_market_data`: Kalshi's market-data
  routes are public, so no API key is configured or held, and the object the
  loop holds has no `place_order` method at all (SPEC S1.1 invariant 3).
- **Egress is still gated.** The venue base URL is resolved from
  `exchange.environment` (`demo` | `production`) and screened against the
  deployment's own outbound allowlist before any session is created (SPEC
  S15). An unrecognized environment, or a Kalshi host the config's allowlist
  does not admit, refuses at startup rather than dialing.
- **Stale data still vetoes.** A live book carries the venue's own
  observation instant, passed through untouched -- unlike the fixture path,
  which re-dates a recording to the run's clock (issue #369). That is
  deliberate: it is what leaves `quote_freshness` and the clock-skew check
  able to actually veto.
- **Every consumer shares one exchange.** The gateway (submitter, status
  source, reconciliation source), the `Reconciler`, and the read-only
  verification view all hold the *same* live-book session object, so the loop
  can never reason about one venue and fill against another.

### Running against live forecast providers (issue #344)

By default the loop's forecast vote stage replays a **recorded LLM cassette**:
`forecast.provider_transport.mode` is `cassette`, so a deployment that says
nothing about providers -- and CI, which builds the default configuration --
never acquires a network dependency. Going live is an explicit written act:

```yaml
forecast:
  provider_transport:
    mode: live                    # default: cassette
    request_timeout_seconds: 30
    anthropic_api_key_env: ANTHROPIC_API_KEY
    openai_api_key_env: OPENAI_API_KEY
    retry:
      max_attempts: 3
      total_deadline_ms: 30000
      backoff_base_ms: 1000
      max_cost_micros: 1000000
    prices:
      - {provider: openai, price_micros: 200000}
      - {provider: anthropic, price_micros: 300000}
    unknown_provider_price_micros: 1000000
```

Then export the keys the config *names* and start the loop as usual:

```bash
export ANTHROPIC_API_KEY=replace-with-a-real-key
export OPENAI_API_KEY=replace-with-a-real-key
windbreak run --paper-books-dir ... --cassette-path ... --ledger-path ... \
  --report-dir ...
```

- **Configuration names variables; it never holds keys.** Every config leaf is
  flattened verbatim into the hash-chained `ConfigLoaded` ledger event and into
  `config_versions.json`, neither of which can be redacted afterwards. So
  `*_api_key_env` carries the *name* of an environment variable, and only
  `windbreak.main` ever reads its value -- the same rule the alert sinks and the
  FutureSearch provider already follow.
- **A missing key refuses to start**, naming the variable and never any value.
- **Each provider gets its own transport, credential, and single-host
  allowlist.** One shared transport would send the first provider's key to the
  second provider's endpoint on the very first vote.
- **Egress is still gated twice.** Each provider endpoint is screened against
  the deployment's own outbound allowlist *before any session exists* (SPEC
  S15); a provider host only reaches that allowlist by being named in
  `forecast.vote_ensemble`. The transport is then additionally pinned to that
  one host. Redirects are refused, so an on-path responder cannot steer a dial
  elsewhere.
- **A half-configuration refuses to start**, in either direction -- live mode
  with no live seam, or a live seam without live mode. Silently replaying a
  recording for an operator who asked for novel forecasts would hand them a
  paper tape they believed was the market. An unrecognized `mode` refuses too.
- **Live providers and live research are independent, and unconfigured research
  produces no evidence at all.** Leaving
  `forecast.research.search_endpoint_url` unconfigured means research finds
  nothing, so the pipeline abstains on zero verified citations *before* any vote
  (SPEC S8.8). That is the correct fail-closed direction, but do not read it as
  a benign default: a loop left this way abstains on **every tick, forever**,
  and a week of uptime yields exactly the evidence that zero days would.
  `research.search_endpoint_url`, `allowed_research_hosts` and
  `research.search_api_key_env` are therefore **required** for any
  evidence-producing run, not optional.

  No concrete end-to-end working configuration can be named here yet, and the
  reason is not research. Configuring research clears one of six barriers
  between an activated PAPER loop and a single order intent; the decisive one
  is arithmetic and unconditional -- every full-pipeline forecast books a flat
  $3.00 research charge, and the selector amortizes the whole of it over a
  fixed 1.00-contract entry probe, so `net_edge_min` is unreachable for any
  market, at any price, with any capital (issue #483). Until that is resolved,
  an activated PAPER loop cannot emit an order intent regardless of how
  research is configured, and nothing in the loop refuses to start or says so
  (issue #485). `tests/scheduler/test_paper_intent_barriers.py` drives the real
  tick over the real shipped composition and pins each barrier with the exact
  values it records.
- **Every live vote is retried and priced.** `RetryingProvider` bounds attempts,
  the total deadline, backoff, and spend from the `retry` block above, and
  charges each failed attempt against the `prices` table. The cassette path is
  deliberately *not* wrapped: billing a list price for a replayed call would
  corrupt the cost accounting the table exists to keep honest.
- **Budget headroom.** `forecast.budget.per_forecast_micros` defaults to
  `6060000` -- the Stage-0 triage prior (`60000`) plus the fixed research charge
  (`3000000`) plus the default three-member ensemble's worst case, which
  `retry.max_cost_micros` bounds at `1000000` per member. That is the exact cost
  of the most expensive *correct* run, with zero headroom above it by design
  (SPEC §16.1). Raising `max_cost_micros` or the ensemble size without raising
  this ceiling will make ticks halt fail-closed on budget and produce no
  forecast at all. Raising either money ceiling also changes
  `screener.max_candidates_per_tick`, which is derived as
  `per_day_micros // per_forecast_micros`.

### Sizing the research fetch cache (issue #453)

Every live fetch writes its payload to an on-disk archive under
`<ledger directory>/research-cache`, one `<sha256-of-url>.txt` file per URL.
Before this bound existed the archive had no size cap, no age bound, no entry
limit and no sweep: it grew for the life of the run.

**That matters because of where it lives.** In the shipped
`deploy/docker-compose.yml`, `--ledger-path
/var/lib/windbreak/ledger/windbreak.db` puts the cache on the **same named
`ledger` volume** as the hash-chained ledger, which `riskkernel` and
`order-gateway` also mount. When that volume fills, the next ledger append
raises and takes the daemon down (issue #443) -- so an unbounded cache was a
slow path to losing the loop, with the audit trail as collateral damage.

**The bound.** `forecast.research.cache_max_bytes` is a ceiling on the
**total bytes** the cache's own entries may hold, defaulting to `268435456`
(256 MiB). After each write the cache deletes its **oldest entries first**
until the total is at or below the ceiling.

```yaml
forecast:
  research:
    # Total bytes the on-disk research archive may hold. Oldest entries are
    # evicted first once the total would exceed it.
    cache_max_bytes: 67108864   # 64 MiB
```

- **What to set it to.** Budget the `ledger` volume first -- the ledger is
  append-only and never evicts, so it must be able to grow for the whole
  retention window -- then give the cache a slice of what is left. A useful
  rule: no more than a quarter of the volume, and never less than
  `forecast.research.fetch_max_bytes` (2 MB by default), since a single
  payload larger than the whole cap can never be held.
- **It must be positive.** `0` or a negative value is refused at composition
  with `forecast.research.cache_max_bytes must be a positive byte count, got
  <N>`, the same way an unrecognized `provider_transport.mode` is refused: an
  operator error in configuration is reported, never reinterpreted.
- **Eviction cannot change a forecast.** Nothing reads the cache back. A fetch
  calls its transport on every call and archives the result *afterwards*, so
  the cache is an archive of what was fetched, not a hit/miss cache in front
  of the network. An evicted entry therefore costs **no re-fetch, no charge
  against the daily research ceiling, and no abstention** -- only the archived
  copy of a page. It is still announced rather than silent:

  ```
  research cache evicted 12 entries (4291719 bytes) to hold its 67108864-byte cap
  ```

  Two warnings are worth alerting on. The first means a single payload is
  bigger than the whole cap, so the cap cannot be honoured -- raise
  `cache_max_bytes` or lower `fetch_max_bytes`:

  ```
  research cache holds <N> bytes against its <M>-byte cap after evicting every
  removable entry; raise forecast.research.cache_max_bytes or shrink
  forecast.research.fetch_max_bytes
  ```

  The second means the cache directory has become unreadable or unwritable, so
  the archive is unbounded again until you fix the volume. **The loop keeps
  beating** -- a loop that cannot start cannot honour a kill file -- and the
  fetch itself is unaffected:

  ```
  research cache eviction failed (PermissionError); the cache stays unbounded
  until its directory is readable and writable
  ```

  A `research cache write failed (...)` warning is the same story one step
  earlier: the fetch succeeded and its payload was simply not archived. Before
  issue #453, that write failure propagated into `verify_citation` and
  `bounded_web_research`, both of which catch `OSError` and treat it as an
  unreachable source -- so a **full volume silently turned every forecast into
  an abstention**, indistinguishable from every source being dead. It no
  longer does, and it now says so.
- **Deleting the cache by hand is safe** while the loop is stopped: nothing
  reads it. The sweep itself will only ever delete a file directly under the
  cache root whose name is exactly a sha256 hex digest plus `.txt` and which
  still resolves inside that root, so foreign files, subdirectories and
  symlinks pointing outside the root are left untouched even if the cache is
  pointed at the wrong directory.
- **Why the cache still shares the ledger volume.** Moving it to a separate,
  disposable volume would decouple research growth from audit-trail durability
  outright, and that is the better end state. It is not done here because the
  cache root is derived from the ledger path at
  `windbreak/scheduler/loop.py:1687` (`ledger_path.parent.joinpath(
  "research-cache")`), which is outside this change's scope, and moving it also
  requires a new compose volume and a matching mount in every systemd unit.
  With the byte cap in place the co-location is bounded and safe to reason
  about: the cache's worst-case footprint on the shared volume is exactly
  `cache_max_bytes`.

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
Don't be surprised to see nothing but vetoes in the dashboard's decisions view
or `selector_decisions.json` -- that is the expected, honestly-ledgered state
of the loop today.

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

**Stopping the PAPER loop with the kill switch (issue #441).**
`windbreak kill --state-dir <dir>` writes a `KILL` file, and the PAPER loop
polls for it once per beat -- first thing, before it screens, researches, or
trades anything -- against the directory `config.ops.state_dir` names. Pass the
*same* directory the running loop is configured with, or the file lands where
nothing reads it.

On the next beat the loop:

- transitions its kernel to `KILLED` and stamps `KILLED` on the `ModeHeartbeat`
  row and the heartbeat log line, so a killed loop is never reported healthy;
- walks no markets at all -- no forecast is run and no research money is spent,
  which is stronger than vetoing the intents afterwards;
- **holds every position**, cancelling only resting orders (ledgered as one
  `CancelAllDirective`; see the caveat below) and releasing its capital
  reservations;
- pages `HALT_KILL` through the sinks `config.alerts` declares, and ledgers one
  `AlertEmitted` audit row for that page.

It stays killed until an operator re-arms. The loop keeps beating, keeps
reconciling the venue, and keeps sampling equity while killed -- observably
alive and flat, which is what an unattended deployment needs. The kill is
recorded on the hash chain, so a restart comes back `KILLED` even if the `KILL`
file has been deleted; the file is belt-and-suspenders, the ledger is
authoritative.

`config.risk.kill_after_consecutive_mismatches` (default 3) is the same switch's
automatic trigger: that many *consecutive* reconciliation `BREACH` outcomes
engage it with trigger `AUTO_RECONCILIATION`, and any non-breach cycle resets
the run. The first breach still drives the kernel to `HALT` (SPEC §32); a
sustained run of them kills.

To re-arm, type the phrase for the engaged kill's sequence number verbatim (see
the re-arm procedure below). Stopping the process with a signal remains
available, but it is not equivalent: a signal provides none of hold-positions,
durable state, or manual re-arm.

**Resting orders are cancelled, and the row says whether they were.** The
`CancelAllDirective` a kill emits is delivered to the venue, not merely
ledgered (issue #480): both compositions — the always-on PAPER loop and
`--process riskkernel` — wire a `directive_sink` over their venue surface, and
the kill cancels every resting order before it appends the audit row.

Read that row's `delivery` field, not just its presence:

| `delivery.outcome` | What it means |
| --- | --- |
| `delivered` | Every resting order was cancelled (`cancelled: 0` here means there were none). |
| `partial` | Some were cancelled, some the venue refused. Orders are still live. |
| `refused` | The venue refused every one. Orders are still live. |
| `errored` | The sink failed before it could count anything. Assume orders are still live. |

Anything but `delivered` also appends a clause to the `HALT_KILL` page naming
the outcome and the counts, so an operator who reads only the page still learns
that live instructions may be resting at the venue. A `delivery_reported:
false` row means no sink was wired at all — on `--process riskkernel` that is
what running without `--snapshot-fixture-dir` produces, since that process
holds no order gateway and has no venue to cancel at. It is an *unknown*, never
a successful cancellation. The counts never name which orders: a venue order id
is venue-supplied text and the chain is unredactable (issue #274).

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
usage error before a file is written. This CLI verb is the **only** way to
grant an acknowledgement today: the dashboard's ack route is an unwired seam
under `windbreak run --process dashboard`, as the route table
[below](#observing-via-the-dashboard) records row by row. As with the kill
switch, the live loop that polls the ack drop-box is not wired yet — this verb
writes the durable grant signal a future live loop consumes.

### Observing via the dashboard

`windbreak.dashboard.app` serves a read-only, loopback-only HTTP surface:

- Binds `127.0.0.1` only (never a public interface -- not configurable, per
  SPEC S14).
- Every route requires `Authorization: Bearer <token>` (timing-safe compared
  against the token `create_server(token=...)` was built with); a
  missing/wrong token gets a `401` with a `WWW-Authenticate: Bearer`
  challenge.

Routes. The **Status** column is what `windbreak run --process dashboard`
actually answers for that method and path with a valid bearer token — not what
a future build might. Every row is replayed against the running CLI-built
server by `tests/docs/test_operator_control_claims.py`, which also fails if the
server grows a route this table omits, so a row here can never quietly become
a claim about a control that is not wired (issue #449).

The table below is also the **only** place in this repository's documentation
that may name a route path in a code span. The same suite scans every root and
`docs/` markdown file — this runbook's own prose included, since that is where
issue #449's defect actually lived — and fails on a route name found anywhere
outside these markers. So prose names the view ("the decisions view", "the
provider panel") and the replayed table names the route; a second, unverified
sentence about what a route answers cannot exist.

<!-- dashboard-routes:begin -->

| Method | Path | Status | Renders |
|---|---|---|---|
| `GET` | `/` | `200` | Current mode and last-heartbeat status. |
| `GET` | `/positions` | `200` | The latest open-positions snapshot. |
| `GET` | `/equity` | `200` | The equity curve vs. the configured capital floor. |
| `GET` | `/decisions` | `200` | The interleaved selector decisions, including skip/veto reasons. |
| `GET` | `/execution` | `200` | Execution quality: each fill's slippage against its decision reference (issue #58). |
| `GET` | `/divergence` | `200` | Live-vs-paper divergence: each sampled or breached row's two series against their thresholds, plus the firing trigger (issue #58). |
| `GET` | `/providers` | `200` | The fleet-observability provider panel: one summary per provider (id, canary status; resolved count and Brier skill from the #194 track-record fold; abstention rate and per-provider `cost_per_forecast` from the #281 per-provider vote-cost fold) plus a fleet-wide cost-per-forecast line. Any figure falls back to `n/a` only for a provider its respective fold does not (yet) cover. See [Provider operations](#provider-operations) below. |
| `GET` | `/acks` | `200` | The pending human acknowledgements awaiting an operator (SPEC S10.8) — always the empty placeholder under the CLI-built server, which wires no `pending_acks_source`. |
| `POST` | `/ack` | `404` | Nothing. The route exists in the handler but the CLI wires no `ack_granter`, so **the shipped dashboard has no working mutation surface**; use `windbreak ack` instead. |

<!-- dashboard-routes:end -->

**The dashboard is read-only as shipped.** The handler's only write route is
the ack route in the table above, and it is an unwired seam:
`windbreak run --process dashboard` builds the server with no `ack_granter`,
so no acknowledgement can be granted over HTTP. The pending-acknowledgements
view does answer, but with no `pending_acks_source` wired it renders its empty
placeholder rather than real pending acknowledgements — an empty page there
means "nothing is wired", never "nothing is pending".

A library caller that passes both seams to `create_server` gets the full
behaviour the handler already implements: the ack route shares the same bearer
gate as every read route (an unauthenticated post gets a `401` and never
reaches the granter) and rejects a malformed, oversized, or non-32-hex body
with a `400` before invoking the granter. Wiring those seams changes the table
above, and the suite will say so.

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

Until the loop has ledgered data, the positions, equity, and decisions views
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
  from `ProviderVoteRecorded` events (issue #281), feeding the provider
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
offending sequence number, exactly like `rebuild`), and `windbreak anchor`
never anchors a broken chain.
`windbreak verify` additionally fails closed if the anchor file is missing,
empty, or holds a malformed line, and reports the first anchored position
whose live hash no longer matches -- or has vanished entirely -- as a
tail-rewrite mismatch on stderr.

**When `anchor` writes nothing (issue #217).** Anchoring is usually scheduled,
so the two ways it can end up writing no anchor are reported differently and
neither is silent -- a cron whose anchor never advances would otherwise only be
discovered at the next `verify`, when the window it should have covered is
already unrecoverable.

| `--ledger-path` names | exit | stderr |
| --- | --- | --- |
| no existing file | 1 | `ledger not found at <path>: anchor reads an existing ledger and will not create one. …` |
| a ledger with no records | 0 | `nothing anchored: the ledger at <path> holds no records, …` |
| a ledger with a head | 0 | *(silent; one anchor line appended)* |

A missing path is a refusal because `anchor` reads an existing ledger and will
not create one -- the same contract, and the same wording, as `rebuild`. Treat
it as a misconfiguration: check `--ledger-path` before assuming the ledger was
lost. An existing-but-empty ledger is *not* an error (a pipeline that has not
yet appended its first event has no head to pin), but it is announced, so an
operator reading cron mail can tell "nothing to do yet" from "anchored". Alert
on the exit code for the first case and on the absence of new anchor lines over
time for the second.

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

### Ingesting a resolved outcome (issue #439)

**Nothing in the running system learns that a market settled on its own.** There
is no venue settlement feed (issues #450/#451). Until an operator runs the verb
below, every evaluation metric in the weekly report reads `UNDEFINED` — not
"not enough data yet", but structurally unreachable — so the paper track record
can neither pass nor fail a promotion gate. **This is a manual step, and the
unattended PAPER loop does not become self-evaluating without it.**

```bash
windbreak ingest-resolution \
  --ledger-path /path/to/state/ledger.db \
  --market-ticker KXPRES-26-DJT \
  --outcome no \
  --resolved-at 2026-03-01T12:00:00+00:00 \
  --source "kalshi settlement notice, retrieved 2026-03-01"
```

Every flag is required; nothing defaults. A defaulted market, outcome, instant
or provenance would be a fabricated fact in a hash-chained audit trail.

| Flag | Meaning |
|------|---------|
| `--ledger-path` | The same ledger the loop runs against. The row is appended to it; the next weekly fold picks it up with no restart and no scheduler change. |
| `--market-ticker` | The settled market. Must match the ticker on the `ForecastCreated` rows exactly, or the forecasts stay `UNRESOLVED`. |
| `--outcome` | `yes` or `no`. Nothing else parses. |
| `--resolved-at` | The instant the market **actually settled**, ISO-8601 **with a UTC offset**. Not the time you are typing this. |
| `--source` | Free text recording where you read the settlement. Never blank. |

`--resolved-at` is the load-bearing field. It is projected onto the ledger's
sequence axis to decide which forecasts could already have known the answer, so
a forecast created after that instant is refused `backdated` and never scored.
That is what makes ingesting a week late safe. It also means **the instant is
taken on your word** — nothing cross-checks it against the venue — so an instant
typed later than the true settlement silently admits forecasts that could have
peeked. Read it off the settlement notice; do not estimate it.

The verb exits `0` and logs `resolution ingested … sequence=N` on success. It
exits `1` and writes **nothing** if the instant carries no UTC offset or is not
ISO-8601, if `--source` or `--market-ticker` is blank, or if this market already
resolved on this ledger with a different outcome or a different instant.

#### If you mistype a flag

- **You notice before the market has resolved on the ledger**: there is nothing
  to undo. Nothing was written unless the verb printed `sequence=N`.
- **You re-run the verb with the same outcome and instant** (e.g. to correct the
  `--source` wording, or having forgotten you already ran it): this is
  accepted and harmless. Provenance is an audit label, not a claim about what
  the market did, so two spellings of it are one resolution. A second row is
  appended and the fold reads one resolution back, keeping the first row's
  label.
- **You re-run with a different `--outcome` or `--resolved-at`**: the verb
  **refuses** — exit `1`, nothing written — and names both the value the ledger
  already carries and the one you just typed. Re-run with the values on the
  ledger. This refusal is deliberate and load-bearing: the ledger is
  append-only, so a contradicting row could never be un-written, and the weekly
  fold runs on *every* tick, which would leave the loop reporting
  `mode=TICK_FAILED` forever with no recovery.
- **You ingested a wrong outcome and it is the only row for that market**:
  nothing refuses this, because nothing contradicts it. Correct it with
  `windbreak correct-resolution` — see the next section. Do **not** try to
  correct it by ingesting again: that is refused, and it is the right refusal,
  because a second contradicting row could never be un-written.

To see what a ledger currently believes resolved, read the report the next tick
writes into `--report-dir`: the `## Cost meter` section's `resolved forecasts`
count and the `== rejections ==` ledger under `## Evaluation` are both derived
from these rows.

### Correcting a wrong resolution (issue #484)

A wrong `--outcome` or `--resolved-at` on the **first** ingest is not caught by
anything, because nothing contradicts it — and from that moment every weekly
report scores every forecast on that market against a false outcome. This verb
is how you take it back.

**Nothing is deleted.** The ledger is hash-chained and append-only, so the wrong
row stays exactly where it is, forever. What this verb appends is a *later row
that supersedes it*, naming the superseded row's `sequence_number` explicitly:

```bash
windbreak correct-resolution \
  --ledger-path /path/to/state/ledger.db \
  --market-ticker KXPRES-26-DJT \
  --superseded-sequence-number 41 \
  --outcome yes \
  --resolved-at 2026-03-01T12:00:00+00:00 \
  --source "kalshi settlement correction notice, retrieved 2026-03-05"
```

| Flag | Meaning |
|------|---------|
| `--superseded-sequence-number` | The ledger row carrying this market's **current** claim: its first ingest, or its most recent correction. Get it from the `sequence=N` the verb logged, or from the refusal below, which names it for you. |
| `--outcome` | The **corrected** outcome. |
| `--resolved-at` | The **corrected** settlement instant. A correction may move the instant as well as the answer, and the temporal gate re-adjudicates against the corrected one — so a forecast that was scored can become `backdated`, and one that was `backdated` can become scored. |
| `--source` | Where you read the *correction* — its own provenance, not the original row's. Overturning a settled outcome is the one act on this ledger that most needs attribution. |

The other flags mean exactly what they mean for `ingest-resolution`.

The verb exits `0` and logs `resolution corrected … sequence=N`. It exits `1`
and writes **nothing** when:

- `--superseded-sequence-number` does not name the row carrying that market's
  current claim — including a row you have **already** corrected. The refusal
  names the sequence number you should have typed, so re-running is one edit.
- the market has no resolution on this ledger at all (use `ingest-resolution`).
- the corrected `--outcome` **and** `--resolved-at` are exactly what the
  superseded row already carries. A correction that changes nothing would put
  an unexplained reversal in the audit trail without moving a metric.
- `--resolved-at` carries no UTC offset or is not ISO-8601, or `--source` /
  `--market-ticker` is blank.

**You can correct a correction.** Name the correction's own row the second time.
A path that could be walked exactly once would be the same permanent trap this
verb removes, reached one command later.

#### How a reader tells a correction from a first ingest

- **On the chain**: the rows carry different `event_type` values —
  `MarketResolved` for an ingest, `SettlementReversed` for a correction — and a
  correction additionally carries `superseded_sequence_number`. A corrected
  outcome can never be mistaken for one that was right all along.
- **In the weekly report**: a `## Resolution corrections` section appears at the
  top, above the metrics, with one `RESOLUTION_CORRECTED` line per correction
  naming both ledger rows, both outcomes and both instants. That section is
  **absent** from a report with no corrections — its presence is the signal, so
  it is never a heading you learn to skip.

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
- `windbreak kill`/`windbreak rearm` do stop and re-arm the PAPER loop (issue
  #441), provided `--state-dir` is the directory `config.ops.state_dir` names.
  The cancel-all a kill emits is delivered to the venue, so resting orders are
  actually cancelled (issue #480); check the `CancelAllDirective` row's
  `delivery.outcome` — anything but `delivered` means orders may still be live,
  and the `HALT_KILL` page says so too.
- `windbreak run --process dashboard` boots the HTTP dashboard server directly
  (issue #79); its bearer token comes only from `WINDBREAK_DASHBOARD_TOKEN`
  and its port only from `config.dashboard.port` -- there is no `--port` or
  `--token` CLI flag.
- Weekly reports are structural stubs (`No data yet.` bodies); the real
  report content is a later pass.
- Evaluation metrics only move if an operator runs `windbreak
  ingest-resolution` (see "Ingesting a resolved outcome"). There is no venue
  settlement feed, so an otherwise-unattended loop still needs this one manual
  step per settled market. A wrong outcome ingested that way is correctable
  since issue #484 — see "Correcting a wrong resolution" — but only by an
  operator who notices it, since nothing cross-checks the claim.

### Declaring correlation buckets (required to size anything)

Since issue #407 the loop **refuses to size a market you have not placed in a
correlation bucket**, and refuses again if it is holding any position it cannot
place either. With the built-in defaults nothing is declared, so a tick logs a
`SelectorDecisionRecorded` carrying `unprovable_exposure: ...` and emits no
intent. That is deliberate, not a regression: before #407 the per-bucket cap
aggregated an empty peer set and the kernel's `concentration_limits` compared
four hardcoded zeros, so both caps ran on every tick, reported success, and
could not veto however concentrated the account became.

Declare buckets in configuration:

```yaml
correlation:
  tags:
    - ticker: KXRAINNYC-26MAR01
      bucket_ids: [weather]
      tagged_at: "2026-03-01T00:00:00+00:00"
    - ticker: KXSNOWCHI-26MAR01
      bucket_ids: [weather]
      tagged_at: "2026-03-01T00:00:00+00:00"
```

- `bucket_ids` must name the SPEC S9.9 seed taxonomy -- `us-election`,
  `fed-policy`, `inflation`, `weather`, `ai-regulation`, `company-specific`,
  `legal-case` -- or `geopolitics-<region>` with a non-empty region. A typo is
  refused at load rather than silently creating a bucket of one.
- `tagged_at` must carry a UTC offset. An offsetless value is refused rather
  than read as host-local, so a tag's provenance cannot depend on which machine
  loaded the config.
- These tags are recorded with `source: human`, because you wrote them. There
  is deliberately no venue-derived source: a correlation bucket is a claim
  about which markets move *together*, and the exchange's free-form `category`
  string ("Politics" spanning US elections, foreign elections and legislation
  alike) cannot support it. Deriving one and labelling it with the venue's name
  would put a provenance in the audit trail that nothing actually holds.
- **Every ticker you hold must be declared, not just the one you are trading.**
  An unclassified holding could be in the target's bucket, so while it is held
  the target's bucket exposure is unprovable and the loop declines. If a tick
  stops sizing after a fill, look for a held ticker missing from this list.

`risk.max_pos_total_pct_ppm` is also declarable now (it was a hardcoded 100%).
It defaults to `1000000` -- 100% of worst-case equity -- which preserves the
previous ceiling rather than choosing a tighter appetite on your behalf. It is a
live cap either way now that total exposure is fed real figures.

## Demonstrating the whole stack offline (the replay corpus)

**What this is for.** Bringing up the shipped stack produces heartbeats and
screen decisions and never a trade, because two seams are closed by
construction: the offline research bundle finds nothing (so every forecast
abstains on `no_verified_citations` before a single vote is cast), and the
committed vote cassette holds placeholders. That is the correct default — an
offline loop must not invent evidence — but it means an operator has no way to
see an intent, an approval or a fill cross the stack before trusting it with a
live key. `forecast.replay_corpus` is that way (issue #510).

**What it is not.** A replayed forecast is recorded material, not a
measurement. A run in this mode demonstrates that the pipeline composes end to
end and tells you **nothing whatever about edge**. Never point a production
deployment at a corpus, and never read a replayed `ForecastCreated` as a
forecast.

### Turning it on

The committed example is `tests/fixtures/config/hermetic-demo.yaml`, and it is
this, in full:

```yaml
forecast:
  replay_corpus:
    mode: "replay"                                    # default: "disabled"
    corpus_dir: tests/fixtures/forecast/hermetic_corpus

correlation:
  tags:
    - ticker: MKT-DEMO
      bucket_ids: [fed-policy]
      tagged_at: "2025-01-01T00:00:00+00:00"
```

```bash
# From the repository root: `corpus_dir` is resolved against the working
# directory. The M6 track-record artifact has to be in --report-dir first, or
# every provider is unproven and no forecast is live-eligible.
mkdir -p /tmp/wb-demo/report
cp tests/fixtures/evaluation/provider-track-records.json /tmp/wb-demo/report/

python -m windbreak run --process pipeline --max-beats 2 \
  --heartbeat-interval 0 \
  --ledger-path /tmp/wb-demo/ledger.db \
  --paper-books-dir tests/fixtures/books/hermetic_demo \
  --cassette-path tests/fixtures/forecast/cassettes.json \
  --report-dir /tmp/wb-demo/report \
  --config tests/fixtures/config/hermetic-demo.yaml
```

The ledger then carries `ScreenDecisionRecorded` (eligible), a
`ForecastCreated` with **no** `abstention_reason`, three `ProviderVoteRecorded`
rows, a `SelectorDecisionRecorded` with `intent_count: 1`, and — on the second
beat — `IntentApproved`, `ApprovalTokenIssued` and the four
`OrderTransitionLedgered` edges `APPROVE → REQUEST_SUBMISSION → SUBMIT → ACK`.
Two beats, not one: on the first beat of a fresh ledger the risk kernel vetoes,
because `equity_start_of_day` is 0 until that beat's own `EquitySampled` row
exists.

### Reading the startup line

Every start logs the mode in force and the source that chose it:

```
WARNING forecast replay corpus mode=replay dir=... source=configuration
        documents=3 votes=3; forecasts on this run replay recorded material and
        measure nothing
INFO    forecast replay corpus mode=disabled source=default
```

`source=configuration` means a file selected it; `source=default` means nothing
did and you are on the shipped, offline, cannot-trade path. The corpus line is
a **WARNING** on purpose: a replaying run must not be discoverable only by
reading configuration.

### What a corpus is, and why it is not a cassette

A corpus directory holds two files:

- `research.json` — `documents` (a `url`/`body` pair per recorded page) and
  `results` (recorded candidate URLs per recorded subquestion).
- `votes.json` — one recorded completion per ensemble member, keyed
  `provider:model_version`, in the same entry shape as a cassette.

It is not keyed by prompt hash, and could not be. A cassette key digests the
vote prompt, which interpolates the market's close time; the screen measures
that same close against the run's clock. A market that keeps clearing the screen
must carry a close that moves, and a key that moves cannot be committed. A
corpus keys on what does not move: the subquestion text (built from the market
title alone) and the ensemble member's own identity.

### Failure modes

- **Both halves come from one token.** There is no way to replay research
  without replaying votes. Doing only the first turns a graceful abstention into
  a `CassetteMissError` raised out of the tick.
- **A malformed or missing corpus refuses to start**, naming the file and the
  problem — including a `results` entry pointing at a URL the corpus holds no
  document for, which would otherwise surface much later as an unexplained
  abstention.
- **Misses degrade, they do not crash.** A subquestion the corpus never recorded
  finds nothing (the forecast abstains); a URL it lacks reads as an unreachable
  page (that citation is lost); an ensemble member it lacks discards that one
  vote (the others still form the two-of-three quorum).
- **`mode: replay` alongside `forecast.provider_transport.mode: live` is
  refused.** A run reads recorded material or the world, never both.
- **Write `disabled`, not `off`.** YAML reads a bare `off` as the boolean
  `false`, and the loader will reject it as a non-string.
- **Nothing here dials out.** Both transports read committed files; they hold no
  endpoint, no credential and no session, so no key is required and none is
  read.

## Operator alerts

Alerts reach you only through the sinks `config.alerts` declares. Until you
declare one, every alert falls back to the log-only sink: it appears on stderr
as a JSON `AlertEmitted` line and **nowhere else** you can be paged from. (The
one exception is the kill switch's `HALT_KILL` alert, which since issue #287
*also* appends an `AlertEmitted` row to the hash-chained ledger. That row is
still an audit record and not a delivery channel — it cannot page you — but
since issue #413 it records *delivery*, not merely emission: it carries one
closed `{sink, outcome, fallback}` entry per attempted sink, so after the fact
you can tell a page every sink accepted from one every sink dropped. See
"Read a kill page off the ledger" below.) The log-only fallback is the shipped
default
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

### Read a kill page off the ledger

After a kill, the `AlertEmitted` row for `HALT_KILL` answers *was anyone told?*
— not just *did we try?* (issue #413). Its payload carries `deliveries`, one
entry per attempted sink in attempt order:

```json
{"deliveries": [{"sink": "webhook", "outcome": "refused", "fallback": false},
                {"sink": "log-only", "outcome": "delivered", "fallback": true}],
 "delivery_reported": true}
```

- `outcome` is one of exactly four values. `delivered` — the sink accepted it.
  `refused` — the destination answered and declined: a non-2xx HTTPS response,
  or a refused connection. It is up and saying no, so check its quota, auth and
  status page. `timed_out` — no answer inside the transport timeout.
  `errored` — anything else, the fail-closed default for a failure the code
  could not classify.
- `fallback: true` marks the log-only fallback, which fires only when no
  configured sink accepted. **A row whose only `delivered` entry has
  `fallback: true` means nobody was paged** — the alert reached a log file and
  stopped there. That is the case this row exists to make visible; before #413
  it was byte-identical to a fully delivered page.
- `delivery_reported: false` with an empty `deliveries` means the dispatcher on
  that path records no delivery evidence at all — *unknown*, never *delivered*.
  Read it as no evidence, not as good news.

There is deliberately no free-form field here: no exception text, no URL, no
sink-supplied string. A sink name the codebase does not define is recorded as
`unregistered`. The chain is append-only, so nothing written into it could ever
be redacted, and a bearer-token-bearing webhook URL in a failure message is
exactly the disclosure issue #274 found (see SECURITY.md). The unredacted detail
stays on the log line only.

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

   or the live dashboard's provider panel, started with
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

#### Writing the artifact (issue #440)

**Until issue #440, nothing wrote that file**, so the bootstrap above was
terminal rather than transient: every provider stayed unproven whatever the
loop did, and `min_resolved` could never be approached. `windbreak
evaluate-providers` is the evaluation pass this section always described:

```bash
windbreak evaluate-providers \
  --ledger-path /path/to/state/ledger.db \
  --report-dir /path/to/state/reports
```

Both flags are required and neither path is created. `--ledger-path` must be
the ledger the loop runs against and `--report-dir` the directory it was
started with; a mistyped path is refused with exit `1` and nothing written,
because scoring the wrong ledger, or publishing into a directory the gate never
reads, both look exactly like success.

It folds three row types off that one ledger: `ForecastCreated` (the
probability and the executable-price baseline), `ProviderVoteRecorded` (which
providers' votes actually backed each forecast -- an `abstained` or `discarded`
vote backs nothing), and the `MarketResolved` rows you ingested with
`windbreak ingest-resolution`. **So run `ingest-resolution` first**: with no
settled market, no provider has a resolved forecast and the artifact is written
as `{}`.

Two properties are worth knowing before you read the numbers:

- **`brier_skill_ppm` is the skill of the forecasts a provider backed**, scored
  on the aggregate probability those forecasts were published with. The ledger
  does not record each ensemble member's own probability, so this is not the
  provider's isolated calibration. Providers that vote on identical market sets
  earn identical skill; they diverge where their abstentions and discards do.
- **Nothing optimistic is ever written.** A provider with no resolved forecast
  the temporal gate admits -- or one whose skill has no defined denominator --
  is omitted from the document entirely, and omitted reads as unproven. A
  provider with real but insufficient evidence *is* written, with its exact
  values, so the bars themselves are what hold it.

The verb exits `0` and logs
`provider track records published path=... providers=N summary=...` (the
summary reads `none` when nothing qualified). It exits `1`, writing nothing, on
a refused path or a ledger whose `MarketResolved` rows cannot be folded.

**The loop reads the artifact once, at startup.** The gate is a process-lived
read model built in `build_paper_deps`, so unlike `windbreak
set-research-budget` this change does **not** take effect on the next tick:
restart the loop after publishing, then confirm with the `ProviderGateHeld`
query below that the provider is no longer named.

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
missing) rather than the provider being bad -- re-run `windbreak
evaluate-providers` and restart the loop. Raising or lowering the bars is a
config edit to
`config.forecast.provider_gate`; there is no runtime lever, and the bars in
force at the time of each decision are on the row itself.

### Respond to budget exhaustion

`windbreak.forecast.budget.ResearchBudget` enforces SPEC S16's three research
spend ceilings, mirrored in config at `config.forecast.budget.per_forecast_micros`
(default 6,060,000 micros / $6.06), `config.forecast.budget.per_day_micros`
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

Since issue #442 a third row is written on the *successful* path too:

- `ResearchSpendRecorded` -- one ledger row per charge, carrying `utc_day`,
  `market_ticker` and `cost_micros`. Nothing halts on it; it exists so the
  day's spend is durable (see "Changing the daily research ceiling" below).

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

A halted market appends exactly one `ResearchBudgetHalted` ledger row (component
`scheduler`) carrying `halt_kind` (`per_day` or `per_forecast`), `utc_day`,
`spent_micros`, and `budget_micros`. The tick then **skips that market's
forecast, select, and approve stages and stops walking its remaining
candidates**, but still emits its heartbeat, equity sample, positions snapshot,
and weekly report -- so a budget-halted loop stays observably alive and flat
rather than dying. No `ForecastCreated` row is fabricated for a market where the
forecast engine never ran; in a hash-chained audit ledger an honest gap beats an
invented record.

**What a halt actually leaves on the ledger.** The halting market and the
markets behind it leave *different* shapes, so read them separately:

| Market | Rows present | Rows absent |
| --- | --- | --- |
| Forecast before the halt | `ScreenDecisionRecorded`, `MarketSnapshotRecorded`, `ForecastCreated`, `SelectorDecisionRecorded`, `ExchangeStatusObserved` | — |
| **The halting market** | `ScreenDecisionRecorded`, `MarketSnapshotRecorded`, then the tick's `ResearchBudgetHalted` | `ForecastCreated`, `SelectorDecisionRecorded`, `ExchangeStatusObserved` |
| **Every candidate after it** | `ScreenDecisionRecorded` only | `MarketSnapshotRecorded`, `ForecastCreated`, `SelectorDecisionRecorded`, `ExchangeStatusObserved` |

The halting market keeps its snapshot because the tick ledgers the book before
the forecast stage can halt. The markets behind it are never run at all, so
they lose their snapshot too -- do **not** expect a `MarketSnapshotRecorded` for
them. What they keep is their screen verdict, because the screen covers the
whole candidate set before the walk begins: the ledger says such a market was
examined and found eligible, and says nothing further about it.

Both shapes are pinned as exact golden row sequences in
`tests/integration/test_paper_universe.py`, so this table is checkable rather
than prose that can drift.

**How many forecasts one tick can buy (issue #345).** A tick no longer forecasts
one hardcoded ticker: it screens the venue's market universe and forecasts up to
`config.screener.max_candidates_per_tick` of the markets that pass (default
`3`). The screen itself is free -- the four §16 filters are integer comparisons
over market metadata and a book, with no model calls -- so it is the *candidate
bound*, not the screen, that caps a tick's research bill at
`max_candidates_per_tick x per_forecast_micros`. The default `3` is
`per_day_micros // per_forecast_micros` at their own defaults, so a single tick
cannot plan to spend more than a whole worst-case day. Raising
`max_candidates_per_tick` without raising `per_day_micros` simply moves the
day's halt earlier in the day; a value below `1` is refused at startup by
`build_paper_deps`.

Markets the walk did not reach get **no** `ScreenDecisionRecorded` row. That is
deliberate: the ledger states which markets were examined and does not claim a
verdict on markets the tick never looked at.

**Operator arithmetic.** With the SPEC defaults -- a 20,000,000-micro day
ceiling against the fixed 3,000,000-micro per-forecast research charge -- a UTC
day permits **7 charged forecasts** before research halts for the rest of that
day. There is no partial-day rollover: the bucket resets, it does not decay.

**The daily ceiling is now the *only* governor of research spend (issue #483).**
The entry gate used to subtract a forecast's whole research cost from the net
edge of a fixed 1.00-contract probe. A binary contract's gross edge over one
contract cannot exceed 1,000,000 ppm and the charge is 3,000,000 micros, so
`net_edge_min` was arithmetically unreachable for every market at every price
-- spend was held near zero by an accident, not by a policy. The owner's
2026-08-10 decision removed that subtraction and moved governance here. Read
that as an operational fact: the number below is what stands between an
unattended loop and its LLM bill.

**Caveat -- a negative ceiling now aborts startup.** The YAML loader applies no
range checks, so a negative `per_forecast_micros`/`per_day_micros`/`max_pages`
reaches the composition root intact and `ResearchBudget` rejects it with a
`ValueError` while `--paper` is starting. That is the intended fail-closed
outcome: a loop that cannot determine its ceiling must not run. A **zero**
ceiling is legal and means "immediately exhausted" -- it silently halts all
research, which is fail-closed but easy to mistake for a hung loop.

### Changing the daily research ceiling (issues #442, #483)

There are three levers, in increasing order of immediacy. **The ledgered one
wins**, and it is the only one that works without a restart.

**1. Configuration (persistent default).** Edit
`config.forecast.budget.per_day_micros` in the YAML you pass to `--config`.
Takes effect on the next process start.

**2. Invocation argument (per-run default, no file edit).**

```bash
windbreak run --process pipeline \
  --research-per-day-micros 40000000 \
  --config /path/to/windbreak.yaml \
  ...
```

Overrides the configured value for that run and nothing else -- the
per-forecast ceiling and the page cap are untouched. A negative value is
refused by the argument parser before anything is composed.

**What startup logs, exactly.** With `--ledger-path` supplied, `windbreak run`
*folds the ledger* and reports the ceiling genuinely in force, together with the
source that won:

```text
research per-day ceiling <N> micros source=<source>
```

| `source=` | Means |
|---|---|
| `set-research-budget` | A ledgered `ResearchBudgetCapSet` row is present, and by the precedence rule below it wins. `<N>` is that row's ceiling. |
| `--research-per-day-micros` | No cap row on the ledger; the invocation argument is in force. |
| `configuration` | Neither; the YAML value (or its default) is in force. |
| `unreadable-ledger` | This ledger's research rows cannot be folded. `<N>` is `0` and the loop will halt every market on the budget -- see [the malformed-row section](#a-malformed-research-row-is-not-terminal) below. |

Do not infer the ceiling from the flag you typed. Before this was fixed the log
printed the flag's own value and named the flag even when a `set-research-budget
--per-day-micros 0` row was in force, which is wrong in the permissive
direction on a spend brake. Without `--ledger-path` there is no ledger to fold
and no loop that could read one, so the flag or the configuration is reported.

**3. The ledgered verb (runtime, no restart).**

```bash
windbreak set-research-budget \
  --ledger-path /var/lib/windbreak/ledger.db \
  --per-day-micros 40000000 \
  --note "Fed week: raising the ceiling through Friday"
```

Appends one `ResearchBudgetCapSet` row to the hash-chained ledger. The running
loop re-reads the ledger at the head of **every tick**, so the new ceiling is in
force on the next beat -- no restart, no source change, and the change is
recorded tamper-evidently rather than living in a shell history.

- **The latest row wins**, in both directions. Lowering after raising works, and
  vice versa. Two rows can never "conflict": a ceiling is an instruction, not
  evidence, so a later instruction simply supersedes an earlier one and nothing
  this verb writes can ever stop a tick. (Contrast `ingest-resolution`, which
  *does* refuse a contradicting append -- a settled outcome is evidence.)
- **A ledgered row beats `--research-per-day-micros`, which beats the config
  file.** The invocation argument is a startup default; the verb is a runtime
  instruction. To go back to the configured value, run the verb again with it.
- `--per-day-micros 0` is legal and means **stop all research spend**. Use it
  when a cost anomaly is under investigation: it halts spending on the next
  tick while the loop stays observably alive, screening, reconciling, and
  reporting. Undo it by running the verb again with a positive ceiling.
- `--note` is required, recorded verbatim, and **never compared against
  anything**. Rewording it can never make two changes disagree.
- A refused call (blank note, negative ceiling) writes nothing at all: exit 1
  for the former, argparse's exit 2 for the latter, and the ledger is left
  byte-for-byte unchanged.
- **`--ledger-path` must name an existing ledger.** The verb refuses to create
  one, with exit 1 and no file written. A typo'd path used to *succeed*: a fresh
  database was created, the row landed at sequence 1, the log read `research
  per-day ceiling set to 0 micros sequence=1`, the exit code was 0 -- and the
  running loop, reading the ledger it was started with, never saw the cap. On
  the verb whose purpose is to stop money that is a fail-open, so it is now an
  error. Retype the path and re-run; nothing needs undoing. A path naming a
  *directory* is refused the same way, with exit 1 and a message, rather than an
  `sqlite3.OperationalError` traceback.

**The day's spend now survives a restart (issue #442).** `_build_research_budget`
folds this ledger's `ResearchSpendRecorded` rows into the day counter the
process opens with, at startup *and* at the head of every tick. Before that, day
spend lived only in process memory: with `restart: on-failure` in
`deploy/docker-compose.yml` and `Restart=on-failure` in every
`deploy/systemd/*.service`, the per-day ceiling was really a **per-process**
ceiling and the process count per day is unbounded. It is now genuinely per-day,
across crashes, deploys and manual bounces, and across two processes sharing one
`--ledger-path`.

Two consequences worth knowing before an incident:

- **A malformed research row halts research, not the loop.** See the next
  section -- it is the one shape of ledger damage this machinery can meet, and
  what it does is worth reading before you meet it.
- **The boundary is UTC, not local.** A day rolls over at 00:00 UTC. A local
  midnight does not reset anything, and a UTC midnight does even when the local
  date has not changed. `tests/scheduler/test_research_spend_durability.py`
  pins both directions under a fixed UTC-05:00 process timezone, because CI runs
  UTC and would not see the difference.

### A malformed research row is not terminal

Both folds fail **closed**: a `ResearchSpendRecorded` or `ResearchBudgetCapSet`
row they cannot read is refused, never skipped, because a skipped charge would
undercount the day and re-open a ceiling that should be shut. Neither the budget
writer nor `set-research-budget` can produce such a row -- both validate before
appending -- so it can only arrive from a schema migration or a tool windbreak
did not write. A row also counts as unreadable if its `payload_schema_version`
is not one this build knows how to read.

**What that refusal does, and deliberately does not do.** It fails closed on the
*spend*, not on the process:

- The loop **composes and beats normally**. Heartbeats, equity samples,
  positions snapshots, the verification cycle, reconciliation, reporting and --
  the one that matters -- **kill-file handling all keep working**.
- The research budget opens with a **ceiling of `0`**, so every market halts on
  the budget and ledgers a `ResearchBudgetHalted` row (`halt_kind: per_day`,
  `spent_micros: 0`, `budget_micros: 0`). No research money is spent.
- The cause is logged as a `CRITICAL` on **every tick**, naming the offending
  key and the row's `sequence_number`: `research spend history unreadable,
  opening a zero ceiling: ...`.
- `windbreak run` reports `source=unreadable-ledger` at startup, so the ceiling
  the log states is the ceiling the loop will actually enforce.

This is a deliberate reversal. Letting the refusal escape stopped
`build_paper_deps` from composing at all, which under the `restart: on-failure`
of `deploy/docker-compose.yml` and the `Restart=on-failure` of every
`deploy/systemd/*.service` is an unbounded crash loop -- over a row that cannot
be removed from an append-only hash-chained ledger. **A loop that cannot start
cannot honour a kill file.** A loop that runs with research disabled can, and
can be watched, reconciled and killed while you fix the ledger. Disabling
research is the strictest possible answer to "I cannot tell what this day has
spent"; refusing to run is a strictly worse one.

**Recovery.** The row cannot be deleted: the chain is append-only and
hash-linked, and there is deliberately no verb that edits it. The recovery is to
start the loop against a **fresh `--ledger-path`**, accepting that the new
ledger's day counter starts at zero (so set the day's ceiling accordingly with
`set-research-budget` before resuming, if the old day had already spent). Retain
the damaged ledger for the audit trail. Until you do any of that, the loop is
safe, loud, killable, and spending nothing on research -- which is why this is a
decision you make on your own schedule rather than an outage.

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
   entry, or hit the dashboard's provider panel, or check the weekly
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
`canary_status.json` and on the dashboard's provider panel indefinitely --
"confirm it is gone" is not literally achievable with today's tooling.
Instead, treat a `canary_status.json` or provider-panel entry whose
`created_at` predates the retirement date as the retirement signal, until a
future retirement/tombstone mechanism ships. Similarly, the weekly report's
`## Providers` section (`windbreak.reports.providers.render_provider_lines`)
is a real, unit-tested renderer, but no production composition root supplies
its `provider_lines` argument yet (`windbreak/scheduler/loop.py` writes the
PAPER-loop's weekly report via the plain `maybe_write_weekly` stub path, not
`windbreak.evaluation.report.generate_weekly_report`) -- so today the weekly
report's `## Providers` section always renders its `No data yet.` fallback,
regardless of what the ledger holds.
