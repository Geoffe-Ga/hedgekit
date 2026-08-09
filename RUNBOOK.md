# Runbook

This is the operator procedure index SPEC §19 requires: numbered,
copy-pasteable procedures naming the exact CLI invocation and the ledger or
alert evidence to expect. It documents the CLI and drills as shipped today; for
the always-on PAPER loop's deeper day-to-day mechanics (activation, weekly
reports, dashboard views), see [`docs/RUNBOOK.md`](docs/RUNBOOK.md) rather than
duplicating that material here.

## 1. Start / stop the daemon

Start one process (the default token is `pipeline`; the four SPEC §5.1 tokens
are `pipeline`, `riskkernel`, `order_gateway`, `dashboard`):

```bash
windbreak run --process pipeline --heartbeat-interval 5
```

Stop it with `Ctrl-C` (SIGINT) or SIGTERM; it shuts down cleanly and logs the
shutdown reason. `--max-beats N` stops it automatically after `N` heartbeats,
useful for a bounded smoke run. See `docs/RUNBOOK.md` for the four
`--paper-books-dir`/`--cassette-path`/`--ledger-path`/`--report-dir` flags that
activate the always-on PAPER loop, and for what one PAPER tick actually does.

## 2. Kill switch, re-arm, and human acknowledgement (SPEC §10.11, §10.8)

Engage the kill switch by dropping an empty `KILL` file into the state
directory:

```bash
windbreak kill --state-dir <dir>
```

Re-arm afterward with the typed confirmation phrase read from stdin, written
verbatim to a `REARM` file:

```bash
windbreak rearm --state-dir <dir>
```

Grant a held order's pending human acknowledgement (SPEC §10.8: any order
whose worst-case cost exceeds `config.risk.require_human_ack_above_micros` is
held, not routed, until acknowledged) by its 32-hex approval id:

```bash
windbreak ack --approval-id <32-hex-approval-id> --state-dir <dir>
```

**Known limitation.** `windbreak kill`/`windbreak rearm` write and clear the
durable `KILL`/`REARM` files, but the always-on PAPER loop's `RiskKernel` is
not yet constructed with a kill-file watcher wired in (issue #144), so these
files do not yet stop a running loop. Until #144 lands, stop the process
itself with a signal (procedure 1).

## 3. Restore from backup

Rehearse a ledger restore with the `restore-from-backup` drill, which copies a
backup ledger, verifies its hash chain, and asserts the rebuilt read models are
byte-identical to the original's:

```bash
windbreak drill restore-from-backup --fixture-dir <dir> --state-dir <dir>
```

A passing drill ledgers exactly one `DrillCompleted` event with a `passed`
verdict; a tampered backup surfaces as a chain-integrity failure naming the
offending sequence number, turned into a failed `DrillCompleted` rather than a
silently-accepted corrupt restore.

**Known limitation — no encrypted-backup producer yet.** The drill above
proves *restoring from* a ledger backup is safe, but there is no scheduled
process that *produces* an encrypted backup of the live ledger today, so
"restore from backup" as an end-to-end operator procedure has no producer half
yet. Tracked in issue #201, which also covers the missing audit-bundle and
tax-record export CLI (see procedure 10).

## 4. Rotate keys

Rehearse a credential rotation with the `key-rotation` drill, which replaces
each credential environment variable with a freshly generated, same-shape
value and confirms the shipped preflight checklist still grades a clean exit
code afterward:

```bash
windbreak drill key-rotation --fixture-dir <dir> --state-dir <dir>
```

Its ledgered `DrillCompleted` evidence carries no key material — only variable
names, booleans, and the integer preflight exit code.

## 5. Reconciliation mismatch (SPEC §10.10, §10.11)

Rehearse the automatic-kill-on-mismatch path with the `reconciliation-mismatch`
drill, which drives a scripted reconciler through a vanished-order scenario and
confirms the kill switch engages and dispatches its alert:

```bash
windbreak drill reconciliation-mismatch --fixture-dir <dir> --state-dir <dir>
```

The number of consecutive `BREACH` reconciliation outcomes that auto-engages
the kill switch is `config.risk.kill_after_consecutive_mismatches`. A passing
drill ledgers one `DrillCompleted`; the kill engagement itself ledgers the
switch's own trigger event.

## 6. Ratchet / profit sweep (SPEC §10.7)

Rehearse the floor ratchet and the "funds cannot move" invariant with the
`ratchet-sweep` drill, which observes a fresh equity gain, verifies the floor
ratchets up by the configured fraction, fires the profit-sweep advisory alert,
and independently confirms both the network egress allowlist and the drill's
exchange double structurally refuse any fund-movement path:

```bash
windbreak drill ratchet-sweep --fixture-dir <dir> --state-dir <dir>
```

One `DrillCompleted` is ledgered on completion.

## 7. Preflight (production readiness)

Run the full production-readiness checklist before any live deployment:

```bash
windbreak preflight --fixture-dir <dir> --json
```

Exit code `0` means every check passed or honestly skipped; nonzero means at
least one failed. See [`SECURITY.md`](SECURITY.md) for the seven checks and
their SPEC references. **Known limitation:** preflight runs against
fixture-backed connectors only; real-connector preflight mode and a dedicated
preflight runbook entry are tracked in issue #197.

## 8. Rebuild read models from the ledger

Fold a verified, hash-chained ledger into byte-stable JSON read models (the
same projections the dashboard reads live):

```bash
windbreak rebuild --ledger-path <path> --output-dir <path>
```

`rebuild` verifies the ledger's hash chain first; a corrupted chain fails
closed with a nonzero exit code and the offending sequence number on stderr,
rather than silently emitting a plausible-but-wrong projection. See
`docs/RUNBOOK.md` for the ten files it writes, including the two
fleet-observability projections (`canary_status.json`, `forecasts.json`,
issue #195) `docs/RUNBOOK.md`'s "Provider operations" section reads from.

## 9. Anchor and verify the ledger against tail-rewrite tampering (issue #75)

`rebuild` (procedure 8) and the drills' own `verify_chain` calls prove a
ledger is *internally* consistent, but neither can tell a legitimately short
chain from one whose tail was truncated and re-chained by a writer with raw
database access — both verify cleanly. Anchoring closes that gap. Anchor the
current chain head to an append-only, JSON-lines anchor file:

```bash
windbreak anchor --ledger-path <path> --anchor-path <path>
```

Later, verify the live chain against every anchor recorded so far:

```bash
windbreak verify --ledger-path <path> --anchor-path <path>
```

Both verify the chain first, so a corrupted chain fails closed with a
nonzero exit code and the offending sequence number on stderr, same as
`rebuild`. `verify` additionally fails closed on a missing or malformed
anchor file, and reports the first anchored position whose live hash no
longer matches (or has vanished) as a tail-rewrite mismatch. `anchor` is a
silent no-op on an empty ledger — there is nothing to anchor yet.

**Operator note — the anchor file is a separate trust root, not a stronger
lock on the same one.** Anchoring only detects a tail rewrite if the anchor
file is protected from whoever can reach the ledger database; an attacker (or
process) with write access to **both** can forge a matching pair. Write the
anchor file to a separately-permissioned volume, an append-only/write-once
medium, or a remote/off-host sink the ledger writer cannot rewrite — never
alongside the ledger database under the same principal.

## 10. SPEC §19 procedures with no CLI yet

The following SPEC §19 runbook items are not yet automatable end-to-end.
Each is described honestly below rather than invented as a fake command.

- **Raise the floor / request a floor lowering (SPEC §10.7 governance).**
  Raising `config.capital.floor_micros` is a config change applied
  immediately; lowering requires a ledgered pending change, a 48-hour
  cool-off, a second confirmation with a challenge nonce, an alert, and
  automatic demotion to PAPER until the next full reconciliation passes.
  There is no dedicated CLI verb for either step today — an operator edits
  the config and restarts, and the cool-off/demotion machinery is enforced
  by the Risk Kernel's governance module, not by a CLI workflow.
- **Respond to a canary-drift or schema-anomaly halt (SPEC §10.10).** Both
  are automatic demotion/halt triggers today; there is no interactive
  "respond" command. An operator investigates via the dashboard and ledger
  read models (procedure 8) and, once satisfied, re-arms or restarts as
  appropriate.
- **Export an audit bundle / export tax records (SPEC §19).** Today the only
  record-export path is `windbreak rebuild`'s ten read-model JSON files
  (procedure 8); there is no `audit-bundle` or `tax-export` command. Tracked
  in issue #201.
- **Pause the daemon.** There is no standalone pause command; stopping the
  process (procedure 1) or engaging the kill switch (procedure 2, pending
  #144's loop-wiring) are the current operator levers.
