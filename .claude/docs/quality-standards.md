# Quality Standards

**Navigation**: [← Back to CLAUDE.md](../CLAUDE.md) | [← Principles](principles.md) | [Workflow →](workflow.md)

---

## 1. Code Quality Requirements

All code must meet these standards before merging to main:

> Every threshold in this section is one a tool in the pinned toolchain
> actually computes. `tests/toolchain/test_quality_standards_table.py` holds
> the summary table in [CLAUDE.md](../../CLAUDE.md) to that rule and fails if a
> row names an unpinned tool or an unconfigured number. Do not add a
> requirement here that nothing measures — issue #411 removed one after it had
> been quoted as a hard constraint in fifteen agent briefs.

### Test Coverage
- **Code Coverage**: 90% minimum, computed over line *and* branch outcomes
  combined (`branch = true` plus `--cov-fail-under=90`). Branch coverage has no
  separate floor: it is folded into this single figure, which makes the 90%
  stricter than a line-only 90% because uncovered branches drag it down.
- **Docstring Coverage**: 100% of public symbols, enforced per symbol by ruff
  `D1` (issue #351). Not a percentage — there is no fraction of symbols
  permitted to lose their docstrings.
- **Mutation Score**: 80% minimum (mutmut) — **manual pre-v1.0.0 release gate,
  NOT an automated check**. Run on demand via `./scripts/mutation.sh` or the
  `mutation-gate.yml` workflow; enforced before shipping v1.0.0 (owner
  directive, issue #107), never on push/PR/pre-commit.
- **Test Types**: unit suites per package, plus `tests/integration/`,
  Hypothesis property tests where behaviour is law-like, and
  `tests/toolchain/` for the gates themselves

### Type Checking
- **MyPy**: Strict mode, no `# type: ignore` without justification
- **Type Hints**: All function parameters and return types required
- **Generic Types**: Use for collections (List, Dict, etc.)

### Code Complexity

**Gated** — `scripts/complexity.sh`, dispatched by Gate 1:
- **Cyclomatic Complexity**: max 10 per function, enforced by xenon
  `--max-absolute B` (radon's B grade is CC 6–10). xenon is the enforcing half
  of that script; the `radon cc` / `radon mi` calls above it run with `|| true`
  and only report.

**Guidance, not gated** — write to these, but no tool fails a build over them.
Listing them as requirements while nothing measured them is what issue #411
was filed about, so they are labelled rather than deleted:
- **Maintainability Index**: aim ≥20; `radon mi` reports it, nothing enforces it
- **Max Arguments**: 5 per function
- **Max Branches**: 12 per function
- **Max Lines per Function**: 50 lines

### Linting and Formatting
- **Ruff**: the single lint *and* format authority (ADR-0002). A curated
  `select` list, not "all rules" — `pyproject.toml` records which families are
  enabled, which are deliberately not, and why. No second linter is installed,
  pinned, or run; ruff already reimplements the rule sets the alternatives
  offer.
- **Mypy**: strict mode across `windbreak/` and `scripts/`
- **Bandit**: security scanning with zero exceptions
- **Pip-audit**: dependency vulnerability scanning, run as a module of the
  pinned interpreter

### Documentation Standards
- **Google-style Docstrings**: All public APIs
- **Type Hints in Docstrings**: Args, Returns, Raises sections
- **Code Examples**: For complex functions
- **Architecture Decision Records**: For significant decisions
- **README Sections**: Updated when adding new components

## 2. Forbidden Patterns

The following patterns are NEVER allowed without explicit justification and issue reference:

1. **Type Ignore**
   ```python
   # ❌ FORBIDDEN
   value = some_function()  # type: ignore

   # ✅ ALLOWED (with issue reference)
   value = some_function()  # type: ignore  # Issue #42: Third-party lib returns Any
   ```

2. **NoQA Comments**
   ```python
   # ❌ FORBIDDEN
   x = 1  # noqa: E741

   # ✅ ALLOWED (with issue reference)
   i = 0  # noqa: E741 (Issue #99: Loop convention in legacy code)
   ```

3. **TODO/FIXME Comments**
   ```python
   # ❌ FORBIDDEN
   # TODO: optimize this later
   def process_data():
       pass

   # ✅ ALLOWED (with issue reference)
   # TODO(Issue #123): Optimize query performance in production
   def process_data():
       pass
   ```

4. **Print Statements**
   ```python
   # ❌ FORBIDDEN
   print("Debug info:", value)

   # ✅ ALLOWED
   logger.debug("Processing value: %s", value)
   ```

5. **Bare Except Clauses**
   ```python
   # ❌ FORBIDDEN
   try:
       risky_operation()
   except:
       pass

   # ✅ ALLOWED
   try:
       risky_operation()
   except FileNotFoundError as e:
       logger.error("File missing: %s", e)
   except Exception as e:
       logger.critical("Unexpected error: %s", e)
       raise
   ```

6. **Skipped Tests**
   ```python
   # ❌ FORBIDDEN
   @pytest.mark.skip
   def test_important_feature():
       pass

   # ✅ ALLOWED (with issue reference)
   @pytest.mark.skip(reason="Issue #456: Waiting for API endpoint")
   def test_important_feature():
       pass
   ```

## 3. Security Guidelines

### API Key Handling

**NEVER** store API keys in:
- ❌ Environment variables (`.env` files - committed by accident)
- ❌ Configuration files (hardcoded strings)
- ❌ Code files (constants)

**ALWAYS** use OS keyring:

```python
import keyring

# Store API key (one-time setup)
keyring.set_password("windbreak", "api_key", api_key)

# Retrieve API key
api_key = keyring.get_password("windbreak", "api_key")
if not api_key:
    msg = "API key not found in OS keyring"
    raise ValueError(msg)
```

---

### Path Validation

**NEVER** trust user-provided paths without validation:

```python
# ❌ WRONG: Path traversal vulnerability
def create_file(user_path: str, content: str) -> None:
    with open(user_path, 'w') as f:
        f.write(content)  # User could pass "../../etc/passwd"
```

**ALWAYS** validate and resolve paths:

```python
# ✅ CORRECT: Validate path is within allowed directory
from pathlib import Path

def create_file(user_path: str, content: str, base_dir: Path) -> None:
    """Create file with path traversal protection.

    Args:
        user_path: User-provided path (relative to base_dir).
        content: File content to write.
        base_dir: Base directory (all files must be within this).

    Raises:
        ValueError: If path traversal detected.
    """
    target = (base_dir / user_path).resolve()

    # Ensure resolved path is within base directory
    if not target.is_relative_to(base_dir):
        msg = f"Path traversal detected: {user_path}"
        raise ValueError(msg)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
```

---

### Subprocess Safety

**NEVER** use `shell=True`:

```python
# ❌ WRONG: Shell injection vulnerability
import subprocess
subprocess.run(f"git clone {user_repo}", shell=True)
```

**ALWAYS** use list form without shell:

```python
# ✅ CORRECT: No shell injection possible
import subprocess
subprocess.run(["git", "clone", user_repo], check=True)
```

**Why**: `shell=True` allows shell metacharacters (`; | & $`) to execute arbitrary commands.

---

### Input Validation

**NEVER** assume user input is valid:

```python
# ❌ WRONG: No validation
def process_input(name: str) -> None:
    subprocess.run(["mkdir", name])  # Name could be malicious
```

**ALWAYS** validate input:

```python
# ✅ CORRECT: Validate input
import re

def validate_name(name: str) -> None:
    """Validate name is safe.

    Args:
        name: Name to validate.

    Raises:
        ValueError: If name is invalid.
    """
    # Only allow alphanumeric, hyphens, underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        msg = f"Invalid name: {name}"
        raise ValueError(msg)

    if len(name) > 100:
        msg = f"Name too long: {len(name)} chars (max 100)"
        raise ValueError(msg)

    if name.startswith(('-', '_')):
        msg = f"Name cannot start with {name[0]}"
        raise ValueError(msg)

def process_input(name: str) -> None:
    """Process input with validation."""
    validate_name(name)
    subprocess.run(["mkdir", name], check=True)
```

---

**These security guidelines are critical. Follow them rigorously.**

---

**Navigation**: [← Back to CLAUDE.md](../CLAUDE.md) | [← Principles](principles.md) | [Workflow →](workflow.md)
