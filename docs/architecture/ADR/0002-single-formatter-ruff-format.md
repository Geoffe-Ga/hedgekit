# ADR-0002: ruff format is the single authoritative formatter

- **Status:** Accepted
- **Date:** 2026-07-05
- **Issue:** #104 (align linters/formatters/type-checkers: exact pins, single
  formatter authority, dedup)

## Context

The repo ran **three** formatters that could each claim the last word on a
file's style: `ruff format` (the CI gate), `black` + `isort` (invoked by the
local `scripts/format.sh`), and all three wired into pre-commit. On identical
input these did not always agree. Two constructs oscillated in practice — a
long `assert expr, (f"...")` failure message, and a long `Callable[...]`
fixture return annotation — each requiring a hand-crafted restructure to find
a form all three tools would leave alone. A file `scripts/format.sh` left
"clean" locally could still fail `ruff format --check` in CI.

This was compounded by version drift across contexts: CI installed
unpinned-latest versions, pre-commit pinned its own (often stale) hook
revisions — e.g. a ruff pre-commit rev as old as `v0.2.0` — and the local
dev install pulled in yet a third set of versions via `requirements-dev.txt`.
With three tools *and* three different version sets per tool, "formatted
clean on my machine" carried no guarantee about CI.

The root defect is not a version-pinning problem; it is a governance problem.
A formatter is a style tool with no semantic content of its own — its only
job is to be a single, deterministic source of truth for what "correctly
formatted" means. Running more than one formatter over the same code makes
that undefined by construction: there is no way to be simultaneously
compliant with two formatters whose algorithms disagree on any single
construct, no matter how carefully their versions are pinned.

## Decision

Adopt **`ruff format` as the sole authoritative formatter**, in every
context that touches formatting: `scripts/format.sh` (local), pre-commit, and
CI. Remove `black` and `isort` entirely from all three gates and from
`requirements-dev.txt` / `pyproject.toml`'s `dev` extra.

This decision also folds in the redundant-tool cleanup that naturally
follows from "one formatter, one linter, no duplicated enforcement":

- **Import ordering** is retained — it is not lost, only relocated. Ruff's
  `I` lint rule (already selected in `[tool.ruff.lint].select`) enforces
  import order as part of `ruff check`, so removing the standalone `isort`
  tool is deduplication of an already-enforced rule, not a reduction in
  strictness.
- **`autoflake`** (unused imports/variables) and **`pyupgrade`** (syntax
  modernization) are removed as separate tools for the same reason: ruff's
  `F` and `UP` rule categories already cover this ground, are already
  selected, and are already passing. Two tools enforcing the same rule is
  waste, not extra safety.
- **`tryceratops`** (`TRY`) and **`refurb`** (`FURB`) are removed as
  standalone tools, but *not* folded into an enforced gate yet. They were
  present but non-gating — `main`@HEAD was red against them and code had
  already merged in violation of their rules — so removing them loses no
  enforced strictness today. Turning ruff's `TRY`/`FURB` categories into a
  real gate is deliberately deferred: doing so blind would fail the build on
  106 pre-existing violations. That cleanup is out of scope for this issue
  and is left as documented future work (see Consequences).

## Alternatives considered

1. **Keep all three formatters (`black`, `isort`, `ruff format`), pin every
   version identically, and prove compatibility.** Rejected: this still
   leaves two masters governing the same files. Any future construct where
   black's and ruff-format's algorithms diverge reopens the exact oscillation
   this issue exists to close, and it triples the version-pinning surface
   (three tools' versions to keep in lockstep across three contexts) for no
   benefit over dropping the redundant tools outright.
2. **Make `black` the sole formatter and drop `ruff format`.** Rejected:
   `ruff format` was already the CI gate, and `ruff check` was already the
   linter. Keeping `ruff format` as the single formatter means format and
   lint share one engine and one dependency, minimizing both tool count and
   the number of things that need to agree with each other.

## Consequences

- **Positive:** The oscillation class is eliminated by construction — there
  is exactly one tool with an opinion on formatting, so "formatted" is no
  longer contested. `constraints-quality.txt` is now the single version-pin
  authority for every cross-context quality tool (ruff included); pre-commit
  hook `rev`s are hand-kept in lockstep with those pins (pre-commit cannot
  read a constraints file itself), and
  `tests/toolchain/test_toolchain_pins.py` enforces both that the pins exist
  and that pre-commit's `rev`s match them, plus that `black`/`isort` are
  absent from pre-commit, `requirements-dev.txt`, and the `pyproject.toml`
  `dev` extra.
- **Regression coverage:** `tests/toolchain/test_formatter_regression.py`
  pins the two constructs that historically triggered
  disagreement (the long `assert`-with-`f`-string message, and the long
  `Callable[...]` fixture return annotation) so a future formatter change
  cannot silently reopen them.
- **Residual risk (accepted):** pre-commit supports an independent
  `pre-commit autoupdate` (and hosted `ci:` autoupdate) flow that can bump a
  hook's `rev` without touching `constraints-quality.txt`, reintroducing
  drift between the two. `test_toolchain_pins.py`'s
  `test_precommit_rev_matches_constraints_pin` is the guard that catches
  this the next time the test suite runs — it is not prevented outright, but
  it cannot pass silently.
- **Deferred, not resolved** *(since resolved — see below)*: `pylint` and
  `interrogate` remain documented in project docs but are not wired into an
  enforced gate — this predates #104 and is unchanged by it.
  - **Resolved for `interrogate`** by #351 / PR #409: docstring coverage is
    enforced by ruff `D1`, which reimplements the same rule family, rather
    than by adding a second tool.
  - **Resolved for `pylint`** by #411: the documented `≥9.0` row was removed
    rather than wired. Ruff already ships pylint's rules as its `PL` family,
    and a score threshold is a tolerance for defects rather than the
    per-rule floor this repo gates on — the reasoning, with measured
    `--select PL` counts, is in `pyproject.toml` beside the `select` list.
    `tests/toolchain/test_quality_standards_table.py` now fails the suite if
    any quality-standards row names an unpinned tool, so this instinct is
    enforced rather than merely recorded.
  Likewise, adopting ruff's `TRY`/`FURB` rule
  categories as an enforced gate (rather than just removing the standalone
  `tryceratops`/`refurb` tools) is deferred behind a separate cleanup of the
  106 pre-existing violations; it should be filed and tracked as its own
  issue before those categories are added to `[tool.ruff.lint].select`.
