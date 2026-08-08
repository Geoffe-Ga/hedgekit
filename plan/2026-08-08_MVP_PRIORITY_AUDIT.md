# MVP Priority Audit — 2026-08-08

Scope: all 98 open issues, audited against one question — *what does windbreak
need to be a minimal but genuinely **viable** product, and does the backlog's
priority ordering reflect that?*

Verdict: **no.** The backlog is well-formed, well-specified, and pointed at the
wrong target. The components are built to a high standard; the *product* does
not run end-to-end, and almost nothing in the backlog says so.

---

## 1. Headline finding

**`windbreak run` cannot produce a paper trade, and by construction never
will.** The always-on PAPER loop — the thing the whole system exists to be —
is wired end-to-end to fixtures and can never reach a fill.

Four independent facts, each verified in the code on `main`:

| # | Fact | Evidence |
|---|------|----------|
| 1 | The Risk Kernel **vetoes 100% of intents**, always | `scheduler/loop.py:1106` — *"With the real kernel the approval always vetoes, so no order ever routes and `filled_centis` is `0`."* |
| 2 | Market data comes from a **fixture directory**, not an exchange | `build_paper_deps` → `PaperExchange.from_fixture_dir(books_dir)` (`loop.py:703`); `--paper-books-dir` is the only market flag on `run` |
| 3 | Forecasts come from a **recorded cassette**, not providers | `transport=ReplayCassette.from_path(cassette_path)` (`loop.py:724`) |
| 4 | The Kalshi connector is **imported nowhere** outside its own package | `grep -rl "from windbreak.connector.kalshi"` → only `connector/kalshi/{__init__,adapter}.py` |

The M2.5 epic (#183) shipped real FutureSearch and pinned-LLM providers. The M6
epic (#8) shipped the full evaluation and calibration stack. The M5 loop (#7)
shipped a composition root that uses **neither**. Three excellent subsystems,
no wire between them.

### 1a. Why the kernel can never approve

23 of 24 SPEC §10.3 checks are real. The loop still cannot pass, for four
separate reasons (`riskkernel/checks.py:1122-1135`, `loop.py:15-27`):

- `jurisdiction_product_eligibility` is an **unconditional-veto stub** — and it
  is the only remaining stub. It has **no tracking issue at all**; the code
  comment says it is *"awaiting NormalizedMarket metadata."*
- `exchange_status_ok` and `pipeline_heartbeat_ok` are real, and fail closed on
  the `None` status and heartbeat the loop honestly supplies.
- The three reconciliation checks fail closed on `verification=None` — no
  read-only verification cycle runs in PAPER.

So the single most load-bearing defect in the product is tracked by a source
comment, not by an issue. Nothing in the backlog will ever schedule it.

### 1b. The unbounded-spend hazard

`_forecast_stage` calls `run_pipeline(...)` passing **no `budget`**
(`loop.py:812-819`; `budget` defaults to `None` at `pipeline.py:1779`). Epic
#183's binding honesty stance says *"all spend is ledgered in micros against
per-forecast/per-day budgets; a budget overrun fails closed."* In the always-on
loop, it does not. Today the cassette transport masks this. The moment live
providers are wired — which is the very next thing anyone would do — an
unattended 24/7 loop calls paid APIs with no ceiling.

`ResearchBudget` is built and tested. It is simply not passed. This is a
one-line composition fix guarding real money, and it has no issue either.

---

## 2. The priority inversion, quantified

98 open issues. **One P1** (#329, a tz-awareness bug in a pubdate parser).
**One P0** (#152, a CI review-agent flake — infrastructure, not product).

Composition of the backlog:

| Category | Count | Share |
|---|---:|---:|
| `scan:*` machine-filed (coverage 19, mutation 10, perf 6, docs 3, deps 2, bugs 1) | 41 | 42% |
| `de-slop` machine-filed | 15 | 15% |
| PR-review follow-ups + features | ~35 | 36% |
| Epics (children all closed) | 7 | 7% |

**57% of the backlog was filed by automated scanners.** They are good findings.
They are findings about code the running product does not reach.

This matters mechanically, not just aesthetically. `scripts/ralph/pick-next.sh`
selects strictly by priority tier, oldest-first within tier (lines 141-144). So
the autonomous fleet's actual work order right now is:

1. #152 (CI flake)
2. #329 (pubdate tz bug)
3. …then **~50 P2 issues, oldest first** — #101, #109, #114, #121, #124, and on
   through the coverage and perf scans.

The fleet will spend weeks raising branch coverage on `evaluation/bootstrap.py`
and de-duplicating connector doubles before it ever touches the reason the
product doesn't work. The labels are a program, and the program is wrong.

A second-order effect: the scans file faster than the fleet drains. The backlog
went 71 → 84 → 98 across the last three grooms. Left alone, scan output will
keep diluting the tier the fleet actually reads.

---

## 3. Proposed MVP definition

Minimal, and *viable* — the second word doing real work:

> **An operator runs `windbreak run` unattended against real Kalshi market data
> in PAPER mode. It forecasts real markets with real providers under a hard
> spend cap, the Risk Kernel genuinely approves or genuinely vetoes each intent,
> paper fills land in the hash-chained ledger, the dashboard shows what
> happened, and after the pre-registered window the weekly report renders an
> honest verdict — including, bluntly, "no edge."**

Note what that deliberately excludes: any live-money path, LIVE_MICRO, the
preflight/drill surface (#197, #201, epic #9), promotion machinery beyond
*computing* gate inputs. Per the README, stopping at paper with an honest "no
edge" is a **success state**. That makes PAPER the MVP, not a waypoint — and it
means M7 work is out of MVP scope entirely.

It also excludes the ≥80% mutation gate (#338, #333, #312, #168-175). That is
the documented **pre-v1.0.0 manual gate** (issue #107, CLAUDE.md), not an MVP
gate. Ten open mutation issues are currently competing for fleet attention
against a product that cannot trade.

---

## 4. Recommended critical path

Ordered. Items marked **FILE** do not exist as issues today — that is itself
the audit's most actionable finding.

### Tier A — unblock the loop (nothing else matters until these land)

| Order | Work | Issue |
|---|---|---|
| A1 | Resolve `jurisdiction_product_eligibility`: give `NormalizedMarket` the jurisdiction/product metadata the check awaits, and make it a real check | **FILE** |
| A2 | Supply exchange status, pipeline heartbeat, and a PAPER-mode read-only verification cycle so the reconciliation trio can pass | **FILE** |
| A3 | Hard spend ceiling: pass `ResearchBudget` into `run_pipeline` from the composition root | **FILE** (safety — treat as P0) |
| A4 | Live market data path: let `run` drive `PaperExchange` off the real Kalshi connector (real books, paper fills) instead of a fixture dir | **FILE** |
| A5 | Real forecasts in the loop: transport selection (cassette vs. live) at the composition root | **FILE** |

A1+A2 together are the difference between a demo and a product. A3 must land
*with or before* A5 — never after.

### Tier B — make it a trader, not a single-market toy

| Order | Work | Issue |
|---|---|---|
| B1 | The loop forecasts exactly one ticker: `next(iter(exchange.markets))` (`loop.py:704`). Screen and iterate the real universe | **FILE** |
| B2 | Wire `config.forecast.vote_ensemble` into `run_pipeline` | **#294** (P2 → P1) |
| B3 | Wire `RetryingProvider` / price table / error taxonomy into the live composition root | **#269** (P2 → P1) |
| B4 | Wire the per-provider track-record gate into the PAPER loop | **#305** (P2 → P1) |
| B5 | Wire `gate_plan_store` at the composition root; drop the full-ledger scan per promotion attempt | **#246** (P2 → P1) |

B2-B5 are all the same defect wearing four hats: **M2.5 and M6 shipped
subsystems the composition root never picked up.** They read as routine
follow-ups; together they are the reason real forecasting is unreachable from
the CLI.

### Tier C — the operator can actually see and trust it

| Order | Work | Issue |
|---|---|---|
| C1 | Wire configured alert sinks into `AlertDispatcher` + e2e delivery test — **alerts are configured but never delivered**, including kill-switch and floor alerts | **#274** (P2/medium → P1) |
| C2 | Dashboard tables emit orphan `<tr>`/`<td>` with no `<table>` wrapper — verified in `dashboard/views/{execution,equity,divergence,positions}.py`; the operator's only window renders invalid HTML | **#275** (medium → P1) |
| C3 | All 10 dashboard quality tiles show `N/A` — scripts don't implement the `--metrics` contract | **#122** (P3 → P2) |
| C4 | `windbreak anchor` on an empty ledger is a silent no-op | **#217** (P3 → P2) |

C1 is the sharpest of these. A safety system whose alerts go nowhere is not a
safety system, and it is filed as `priority-medium` de-slop.

### Tier D — quality, scoped to code the MVP reaches

Keep, at P2, the coverage scans on MVP-critical paths: **#162** (PAPER-tick
fill-leg routing — literally the path Tier A unblocks), **#163** (injection
sanitizer — a §8.5 release blocker), **#167** (sizing participation caps),
**#249** (net-edge rejection), **#252** (WAL corruption guard), **#254**,
**#331**. These harden the order path and the injection defense — both
load-bearing for a system that will handle real money later.

Also keep **#329** (P1, tz bug in the FutureSearch pubdate parser — that is
live-provider code Tier A5 turns on) and **#210** (bandit scope over
`scripts/`).

---

## 5. Recommended demotions

Not "wrong" — *wrong now*. All should sit below Tier A-C.

| Issues | Why defer |
|---|---|
| #168-175, #312, #333, #338 (10 mutation issues) | Pre-v1.0.0 manual gate per issue #107, explicitly *not* an automated gate. Zero MVP value. → P3 |
| #289, #290, #291, #319, #320, #332 (6 perf issues) | Optimizing O(orders × fills) and 11× ledger scans while the loop processes **one fixture market per tick and never fills**. Real once Tier A/B create volume; premature until then. → P3 |
| #257, #258, #295, #306, #142 (doc drift) | README/docstring accuracy. Real debt, no product impact. → P3 |
| #136, #137, #139, #205, #207, #208, #277, #280 (comment/dead-code de-slop) | Mostly stale-comment corrections. → P3, batch them |
| #272, #276, #279 (dedup refactors) | Duplication in code the loop doesn't execute. → P3 |
| #197, #201, epic #9 (M7 live-micro) | Out of MVP scope by definition — PAPER is the ceiling |
| #36, #209 | #36 is `blocked`; #209 is a mypy-strict-on-tests epic. Neither gates MVP |

One caveat worth stating plainly: deferring perf is a **bet that Tier A/B land
first**. If they do, #319 and #332 get promoted immediately — a dashboard that
rebuilds 8 ledger scans per HTTP request will be the first thing to hurt once
real data flows. Deferred, not dismissed.

---

## 6. Mechanical actions

1. **File 6 new issues** (A1-A5, B1). A3 as **P0**; the rest **P1**.
2. **Promote to P1:** #294, #269, #305, #246, #274, #275.
3. **Promote to P2:** #122, #217.
4. **Demote to P3:** the mutation set (10), the perf set (6), the doc-drift set
   (5), the de-slop comment set (8), the dedup set (3) — 32 issues.
5. **Leave alone:** #152 (P0), #329 (P1), the seven MVP-path coverage scans,
   #210.
6. **Throttle the scanners.** They produce more than the fleet drains and they
   only look at code, never at whether the product runs. Suggest pausing
   `scan:mutation` and `scan:perf` until Tier A closes.

After this pass the fleet's work order becomes: A3 → A1 → A2 → A4 → A5 → B1 →
B2-B5 → C1-C4 — i.e. *make it run*, then *make it real*, then *make it
visible* — instead of grinding coverage on unreachable branches.

---

## 7. What this audit did not find

Worth saying, because it's the flip side. The engineering standard here is
genuinely high and none of the above is a quality complaint:

- 164 modules, 237 test modules; the quality gates are real and enforced.
- The safety architecture is sound: 23/24 kernel checks real, hash-chained
  ledger, signing-key isolation, egress allowlist, injection corpus.
- The stubs are **honest** — `loop.py` and `checks.py` document exactly what is
  and isn't wired, in detail, without overclaiming. That candor is why this
  audit was possible from the source alone.

The problem is not code quality. It is that the backlog optimizes the parts and
never the whole, and the priority labels — which drive an autonomous fleet —
encode that mistake.
