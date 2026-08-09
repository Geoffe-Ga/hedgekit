# Tool Usage & Code Standards

**Navigation**: [← Back to CLAUDE.md](../CLAUDE.md) | [← Testing](testing.md) | [Troubleshooting →](troubleshooting.md)

---

## 1. Tool Invocation Patterns

**CRITICAL:** Always use the provided quality control scripts instead of invoking tools directly. The scripts ensure:
- Correct configuration is used
- Tools run in the proper order
- Results are consistent with CI pipeline
- Project-specific settings are applied

### Quick Reference

| Task | ❌ NEVER DO THIS | ✅ ALWAYS DO THIS |
|------|------------------|-------------------|
| **Format code** | `ruff format .` | `./scripts/format.sh` |
| **Check formatting** | `ruff format --check .` | `./scripts/check-all.sh` |
| **Lint code** | `ruff check .` | `./scripts/lint.sh` |
| **Type check** | `mypy windbreak/ scripts/` | `./scripts/typecheck.sh` |
| **Run tests** | `pytest` | `./scripts/test.sh` |
| **Run unit tests** | `pytest tests/unit/` | `./scripts/test.sh --unit` |
| **Check coverage** | `pytest --cov` | `./scripts/test.sh` |
| **Security scan** | `bandit -r src/` | `./scripts/security.sh` |
| **Fix issues** | `ruff check --fix .` | `./scripts/fix-all.sh` |
| **All checks** | *(running each tool manually)* | `./scripts/check-all.sh` |

### Why Use Scripts?

**Direct tool invocation bypasses project configuration:**

❌ **BAD - Direct invocation:**
```bash
# Missing project-specific flags
ruff format tests/unit/test_module.py

# Wrong configuration
ruff check . --fix

# Incomplete coverage reporting
pytest tests/
```

**Issues with direct invocation:**
- May use different settings than CI
- Won't generate proper coverage reports
- Results differ from CI pipeline
- Wastes time debugging CI failures locally

✅ **GOOD - Use scripts:**
```bash
# Formats with ruff format (the single authoritative formatter, ADR-0002)
./scripts/format.sh

# Fixes formatting and linting issues automatically
./scripts/fix-all.sh

# Runs all checks exactly as CI does
./scripts/check-all.sh
```

**Benefits of using scripts:**
- ✅ Same configuration as CI pipeline
- ✅ Comprehensive coverage reporting
- ✅ Consistent results across developers
- ✅ Catches issues before CI runs

### Available Scripts

**`./scripts/check-all.sh`** - Run all quality checks (use before every commit)

Runs in order (mirrors the `run_check` calls in `scripts/check-all.sh`):
1. The whole pre-commit hook set (`scripts/precommit.sh`, issue #401)
2. Linting (ruff, ruff `D1`, float-lint, shellcheck)
3. Architecture boundaries (import-linter)
4. Formatting checks (ruff format — single formatter authority, ADR-0002)
5. Type checking (mypy)
6. Security scanning (bandit, pip-audit, detect-secrets)
7. Complexity analysis (radon reports; xenon enforces)
8. Unit tests
9. Coverage report validation (90% minimum, line + branch combined)

**Note**: Mutation testing is NOT included in check-all.sh (long runtime) and is
NOT run by any automated trigger. It is the manual pre-v1.0.0 release gate (owner
directive, issue #107) — run it on demand via `./scripts/mutation.sh` or
`gh workflow run mutation-gate.yml`.

```bash
# Before committing - REQUIRED
./scripts/check-all.sh

# Exit code 0 = all checks pass
# Exit code 1 = some checks failed
```

**`./scripts/mutation.sh`** - Run mutation tests with score validation

```bash
# Run with 80% minimum (MAXIMUM QUALITY standard)
./scripts/mutation.sh

# Run with custom threshold
./scripts/mutation.sh --min-score 70

# Show detailed output
./scripts/mutation.sh --verbose

# This can take several minutes - be patient!
```

**`./scripts/format.sh`** - Auto-format all code

**`./scripts/lint.sh`** - Run all linters and type checkers

**`./scripts/test.sh`** - Run test suite with coverage

**`./scripts/fix-all.sh`** - Auto-fix formatting and linting issues

**`./scripts/security.sh`** - Run security scanners

**`./scripts/complexity.sh`** - Analyze code complexity

**`./scripts/provision-venv.sh`** - Provision/refresh the shared pinned-toolchain `.venv`

**Purpose**: pin the local quality toolchain to `constraints-quality.txt` so
local Gate 1 (`check-all.sh`, especially `pip-audit`) runs against byte-identical
package versions to CI, ending fleet-wide toolchain drift (issue #133).

```bash
# Provision or refresh the shared .venv (idempotent — a clean, matching
# venv does nothing; only installs when fresh or drifted)
bash scripts/provision-venv.sh

# Print the resolved shared .venv absolute path
bash scripts/provision-venv.sh --print-venv

# Fast drift check only (never creates/installs): exits nonzero with a
# "run provision-venv.sh" hint when the installed toolchain diverges from
# constraints-quality.txt or the venv is missing
bash scripts/provision-venv.sh --check

# Help
bash scripts/provision-venv.sh --help
```

**Shared-venv / worktree design**: there is exactly **one** `.venv`, at the
**main repo root**, shared by every fleet worktree under
`.ralph/worktrees/issue-N/`. A call from inside a linked worktree resolves to
`<main-root>/.venv` via git's common `.git` dir, so the toolchain is
provisioned **once** and reused by all lanes — never provisioned per-worktree.

It installs `requirements.txt` + `requirements-dev.txt` under `-c
constraints-quality.txt` and upgrades `pip`/`setuptools`/`wheel` to match CI.
It deliberately does **not** install the project editable (`-e .`): each
worktree must import its **own** checkout's `windbreak/` package, so per-lane
test isolation is preserved even though the toolchain venv is shared.

`check-all.sh` auto-detects the shared `.venv` when present, runs a drift
check against it, and PATH-prepends it so any subprocess a tool spawns also
sees the pinned toolchain. The `.venv` directory is gitignored.

**`./scripts/toolchain-env.sh`** - Resolve a gate tool from the pinned toolchain

**Purpose**: make a gate's verdict a property of this repository rather than of
the shell it was launched from (issue #366). Each gate sub-script resolves its
tools through this helper by absolute path instead of by bare name, so
`./scripts/security.sh` run directly reaches the same verdict it reaches through
Gate 1. It **fails closed**: with the shared `.venv` present a tool comes from it
and only from it, and a tool missing there vetoes rather than silently falling
back to an alien binary. With no `.venv` (how CI runs) it falls back to `PATH`
but names the absolute path it chose, so a verdict is always attributable.

```bash
# Which pip-audit would Gate 1 actually run?
bash scripts/toolchain-env.sh --print-tool pip-audit

# Which environment does the dependency audit inspect?
bash scripts/toolchain-env.sh --print-python

# The resolved shared .venv (empty when absent)
bash scripts/toolchain-env.sh --print-venv
```

One documented exception: `architecture.sh` → `plans/architecture/run-check.sh`
still resolves `lint-imports` from `PATH`, because
`tests/architecture/test_import_linter_gate.py` pins a
fail-loud-when-unreachable-on-PATH contract that anchoring would defeat. The
reason is recorded in `scripts/toolchain-env.sh`'s header.

### Complete Workflow Example

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make code changes
vim src/my_module.py

# 3. Write tests
vim tests/unit/test_my_module.py

# 4. Format code
./scripts/format.sh

# 5. Run all checks
./scripts/check-all.sh

# 6. If checks fail, auto-fix what you can
./scripts/fix-all.sh

# 7. Run checks again
./scripts/check-all.sh

# 8. Manually fix remaining issues if needed
# (edit files to fix mypy errors, add tests for coverage, etc.)

# 9. Final check before commit
./scripts/check-all.sh

# 10. (Optional) Run mutation tests locally for significant changes
# This takes several minutes. Mutation testing is the MANUAL pre-v1.0.0 release
# gate — it is NOT run automatically in CI (issue #107).
./scripts/mutation.sh

# 11. Commit (only if all checks pass)
git add .
git commit -m "feat(module): add my feature (#123)"

# 12. Push
git push origin feature/my-feature

# 13. Create PR (all automated CI checks will pass; mutation testing is a
# manual pre-v1.0.0 gate, not part of PR CI)
gh pr create --fill
```

### When Direct Tool Invocation Is Acceptable

**Only these cases justify direct tool invocation:**

1. **Running a single test file during development:**
   ```bash
   # Acceptable for quick iteration
   pytest tests/unit/test_module.py -v

   # But still run ./scripts/test.sh before committing
   ```

2. **Checking a specific file's types:**
   ```bash
   # Acceptable for quick feedback
   mypy src/module.py

   # But still run ./scripts/lint.sh before committing
   ```

3. **Debugging a specific linting rule:**
   ```bash
   # Acceptable to understand a specific error
   ruff check src/ --select E501

   # But still run ./scripts/lint.sh before committing
   ```

**Golden Rule:** Direct tool invocation is ONLY acceptable during active development for quick feedback. **ALWAYS** run the appropriate script before committing.

## 2. Code Style

{{LANGUAGE_SPECIFIC_STYLE_GUIDE}}

## 3. Docstring Format

All public functions, classes, and modules must have docstrings:

```python
def calculate_total(
    items: list[dict[str, float]],
    *,
    apply_discount: bool = False,
) -> float:
    """Calculate total cost of items.

    Sums the cost of all items in the list, optionally
    applying a discount based on quantity.

    Args:
        items: List of items with 'cost' and 'quantity' keys.
        apply_discount: Whether to apply bulk discount.
            Defaults to False.

    Returns:
        Total cost as float.

    Raises:
        ValueError: If items list is empty.
        KeyError: If required keys missing from item dicts.

    Examples:
        >>> items = [{"cost": 10.0, "quantity": 2}]
        >>> total = calculate_total(items)
        >>> assert total == 20.0

    Note:
        Discount is 10% for orders over 100 items.

    See Also:
        - apply_bulk_discount: Discount calculation logic
    """
```

---

**Navigation**: [← Back to CLAUDE.md](../CLAUDE.md) | [← Testing](testing.md) | [Troubleshooting →](troubleshooting.md)
