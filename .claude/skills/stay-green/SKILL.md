---
name: stay-green
description: >-
  TDD development workflow through this repository's three automated gates:
  Gate 1 is the local `./scripts/check-all.sh` run (which contains the
  Red-Green-Refactor loop), Gate 2 is CI, Gate 3 is code review. Mutation
  testing is a separate manual pre-v1.0.0 gate, not part of the run. Use when
  implementing features, fixing bugs, or doing any development work. Ensures
  code is never committed without passing tests and quality checks.
metadata:
  author: Geoff
  version: 2.0.0
---

# Stay Green

Write tests first, then code. Never declare work finished until every gate is
green.

## The gate sequence

The gates are defined once, in
[`.claude/docs/workflow.md` §2.1](../../docs/workflow.md#21-the-gates). This is
a summary of that section, not a second copy of it — when the two disagree, the
document is right and this file is stale.

1. **Gate 1: Local Pre-Commit** — `./scripts/check-all.sh` exits 0.
2. **Gate 2: CI Pipeline** — every CI job green (`gh pr checks --watch`).
3. **Gate 3: Code Review** — LGTM with no reservations.

Each is iterated until green before the next begins. A fix at Gate 2 or Gate 3
sends you back to Gate 1.

Mutation testing (≥80%) is **not** a fourth automated gate. It is the
[manual pre-v1.0.0 release gate](../../docs/workflow.md#21a-manual-pre-release-gate-mutation-testing-80)
— `./scripts/mutation.sh`, run before shipping v1.0.0 and never on
push, PR or pre-commit (owner directive, issue #107).

## Instructions

### The TDD loop (inside Gate 1)

1. **Red** — write a failing test describing the behavior you want
   ```bash
   ./scripts/test.sh --all  # Should fail
   ```

2. **Green** — write just enough code to make the test pass
   ```bash
   ./scripts/test.sh --all  # Should pass
   ```

3. **Refactor** — clean up while keeping tests green
   ```bash
   ./scripts/test.sh --all  # Should still pass
   ```

Repeat for each small piece of functionality. Write tests incrementally, not
all at once.

### Closing Gate 1

```bash
./scripts/check-all.sh
```

Run the project script, not the underlying tools directly — it dispatches the
whole pre-commit hook set plus the dedicated gates, so a green run here is a
superset of CI's checks rather than a hand-maintained subset of them.

When checks fail: read errors, fix issues, run again. Repeat until all green.

Quality checks include: the whole pre-commit hook set (file hygiene, vulture,
shellcheck), formatting (ruff format — the single formatter authority,
[ADR-0002](../../../docs/architecture/ADR/0002-single-formatter-ruff-format.md)),
linting and docstring presence (ruff), architecture boundaries (import-linter),
type checking (mypy), complexity ≤10 per function (xenon), security (bandit,
pip-audit, detect-secrets), and the test suite with coverage ≥90% (pytest). The
authoritative list is CLAUDE.md's `check-all.sh` section.

### Work is DONE when

1. Gate 1 is green: `./scripts/check-all.sh` exits 0
2. Gate 2 is green: every CI job passes
3. Gate 3 is green: code review says LGTM

There is exactly one exception, and it is narrower than it sounds: a check
whose subject is **live GitHub repository configuration** — something a pull
request cannot change — under five conditions that must *all* hold. They are
stated in
[`workflow.md` §2.4](../../docs/workflow.md#24-exception-checks-whose-subject-is-live-repository-state)
and deliberately nowhere else; read them there before claiming it, because an
ordinary red test satisfies two of the five and is still not covered. Anything
else red means the work is not done.

## Examples

### Example 1: Adding a New Function

```python
# Red: write the failing test
def test_position_size_scales_with_conviction():
    assert position_size(base_micros=1_000_000, conviction=3) == 3_000_000

# Green: make it pass
def position_size(base_micros: int, conviction: int) -> int:
    return base_micros * conviction

# Refactor: (already clean, move on)
# Gate 1: ./scripts/check-all.sh -> exit 0
```

### Example 2: Fixing a Formatting Failure

```bash
# Gate 1 fails on formatting
$ ./scripts/check-all.sh
ruff-format..............................................................Failed

# Auto-fix and re-run
$ ./scripts/format.sh --fix
$ ./scripts/check-all.sh
# All passed!
```

## Troubleshooting

### Error: Coverage below 90%
```bash
./scripts/test.sh --all --coverage  # See what's not covered
# Add tests for uncovered lines, then re-run ./scripts/check-all.sh
```

### Error: Complexity above 10
```bash
./scripts/complexity.sh  # Find complex functions
# Extract helper functions, simplify branching
# Then verify: ./scripts/check-all.sh
```

### Error: Type errors
```bash
./scripts/typecheck.sh  # See specific errors
# Add/fix type annotations
# Then verify: ./scripts/check-all.sh
```
