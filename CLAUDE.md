# Claude Code Project Context: windbreak

**Version**: 2.0 (Modular)

---

## Quick Navigation

**Core Documentation** (in `.claude/docs/`):
- 📋 **[Critical Principles](.claude/docs/principles.md)** - Non-negotiable rules (READ FIRST)
- 🎯 **[Quality Standards](.claude/docs/quality-standards.md)** - Requirements and enforcement
- 🔄 **[Development Workflow](.claude/docs/workflow.md)** - Stay Green process & mindset
- 🧪 **[Testing Strategy](.claude/docs/testing.md)** - Test patterns and coverage
- 🛠️ **[Tool Usage](.claude/docs/tools.md)** - Scripts, patterns, and code standards
- 🚨 **[Troubleshooting](.claude/docs/troubleshooting.md)** - Common mistakes and fixes

**Additional Resources**:
- [Appendix A: AI Subagent Guidelines](#appendix-a-ai-subagent-guidelines)
- [Appendix B: Key Files](#appendix-b-key-files)
- [Appendix C: External References](#appendix-c-external-references)

---

## 📋 Critical Principles (Quick Reference)

**For detailed explanation, see [.claude/docs/principles.md](.claude/docs/principles.md)**

1. **Use project scripts, not direct tools** - Invoke `./scripts/*`, never raw tools
2. **Never duplicate content (DRY)** - Always reference the canonical source
3. **No shortcuts - fix root causes** - Never bypass quality checks
4. **Stay Green** - Never request review with failing checks (gated workflow: 3 automated gates + manual pre-v1.0.0 mutation gate)
5. **Quality First** - Meet MAXIMUM QUALITY standards (90% coverage, ≤10 complexity); mutation ≥80% is the manual pre-v1.0.0 release gate
6. **Operate from project root** - Use relative paths, never `cd`
7. **Verify before commit** - All checks must pass (`./scripts/check-all.sh` → exit 0)

**The Automated Gates**:
1. Gate 1: `./scripts/check-all.sh` passes (exit 0)
2. Gate 2: CI pipeline green (all jobs ✅)
3. Gate 3: Code review LGTM

**Manual pre-release gate** (run before a v1.0.0 ship — NOT an automated check):
- Mutation score ≥80% via `./scripts/mutation.sh` or the `mutation-gate.yml`
  workflow (`gh workflow run mutation-gate.yml`). Owner directive, issue #107.

---

## 🎯 Quality Standards (Quick Reference)

**For complete standards, see [.claude/docs/quality-standards.md](.claude/docs/quality-standards.md)**

| Metric | Threshold | Tool |
|--------|-----------|------|
| **Code Coverage** | ≥90% | pytest-cov |
| **Branch Coverage** | ≥85% | pytest-cov |
| **Docstring Coverage** | ≥95% | pydocstyle / ruff D rules |
| **Mutation Score** | ≥80% (manual pre-v1.0.0 gate, not automated) | mutmut |
| **Cyclomatic Complexity** | ≤10 per function | radon |
| **Pylint Score** | ≥9.0 | pylint |
| **Security Vulnerabilities** | 0 critical/high | bandit, pip-audit |

---

## 📖 Project Overview

windbreak is a Python project built with maximum quality engineering principles. Every line of code is held to the highest standards of testing, documentation, security, and maintainability.

**Purpose**: To deliver production-grade Python software with comprehensive quality enforcement, zero-tolerance for technical debt, and AI-optimized development workflows.

---

## 🏗️ Architecture

**Core Philosophy**:
- **Maximum Quality**: No shortcuts, comprehensive tooling, strict enforcement
- **Composable**: Modular components with clear interfaces
- **Testable**: Every component designed for easy testing
- **Maintainable**: Clear structure, excellent documentation
- **Reproducible**: Consistent behavior across environments

**Component Structure**:

```
windbreak/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Continuous Integration
│   │   ├── security.yml              # Security scanning
│   │   └── dependency-review.yml     # Dependency audits
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── CODEOWNERS
├── .claude/
│   ├── docs/
│   │   ├── principles.md
│   │   ├── quality-standards.md
│   │   ├── workflow.md
│   │   ├── testing.md
│   │   ├── tools.md
│   │   └── troubleshooting.md
│   └── skills/                       # Skill definitions
├── scripts/
│   ├── check-all.sh                  # Run all quality checks
│   ├── test.sh                       # Run test suite
│   ├── lint.sh                       # Linting and type checking
│   ├── format.sh                     # Auto-format code
│   ├── security.sh                   # Security scanning
│   └── mutation.sh                   # Mutation testing
├── src/
│   └── windbreak/                     # Main package
├── tests/
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   ├── property/                     # Property-based tests
│   └── fixtures/                     # Test fixtures
├── docs/
│   ├── architecture/
│   │   └── ADR/                      # Architecture Decision Records
│   └── api/                          # API documentation
├── pyproject.toml                    # Project configuration
├── requirements.txt                  # Runtime dependencies
├── requirements-dev.txt              # Development dependencies
└── README.md
```

**For workflow and architecture detail, see [.claude/docs/workflow.md](.claude/docs/workflow.md)**

---

## 🔄 Development Workflow (Quick Start)

**For the complete workflow, see [.claude/docs/workflow.md](.claude/docs/workflow.md)**

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes and write tests (TDD)
# Write failing test first
pytest tests/unit/test_my_feature.py

# Implement feature
vim src/windbreak/my_feature.py

# 3. Run ALL quality checks
./scripts/check-all.sh

# 4. Fix any issues and run again
./scripts/format.sh  # Auto-fix formatting
./scripts/check-all.sh

# 5. Commit (only when all checks pass)
git add .
git commit -m "feat(module): add my feature (#123)"

# 6. Push and create PR
git push origin feature/my-feature
gh pr create --fill

# 7. Wait for the 4 gates to pass, then merge
```

---

## 🛠️ Tool Usage (Quick Reference)

**For complete patterns, see [.claude/docs/tools.md](.claude/docs/tools.md)**

**Primary Commands**:

### check-all.sh
Run all quality checks before every commit. This is the master quality gate.

```bash
./scripts/check-all.sh
```

Executes:
- **The whole pre-commit hook set** — every hook in `.pre-commit-config.yaml`
  over `--all-files`, via `scripts/precommit.sh`. This is what makes Gate 1 a
  superset of CI's pre-commit job *by construction* rather than by a
  hand-maintained list that drifts (issue #401). It covers dead-code detection
  (vulture), shellcheck, and the file-hygiene hooks (trailing whitespace,
  end-of-file, YAML/TOML/JSON validity, mixed line endings, debug statements,
  private keys, and the rest) in addition to the tools listed below.
  - One hook is excluded, deliberately and by name: `no-commit-to-branch`,
    which asserts a branch policy rather than code quality and would redden
    Gate 1 for anyone running it on `main`. CI skips it for the same reason
    (issue #127). `conventional-pre-commit` runs at the `commit-msg` stage, so
    no file-set run selects it — locally or in CI.
  - `tests/toolchain/test_precommit_gate_coverage.py` fails the suite if a hook
    is ever put out of Gate 1's reach without a registered reason.
  - Several hooks are **fixers**, so this step can rewrite files. It reports
    failure whenever it does, so Gate 1 never passes on unfixed content —
    review the printed diff and re-run.
- Code formatting verification (ruff format — single formatter authority, ADR-0002)
- Linting (ruff, float-lint, shellcheck)
- Architecture boundary checks (import-linter)
- Type checking (mypy)
- Security scanning (bandit, pip-audit, detect-secrets)
- Complexity analysis (radon, xenon)
- Test suite with coverage (pytest)

Runtime: roughly 2m40s warm on a developer machine. The whole-hook-set step is
about 9s standalone and cost +4.7s (+3%) end-to-end when added, most of its
work overlapping the dedicated gates that follow it.

Exit code 0 = ready to commit. Non-zero = fix issues first.

### test.sh
Run test suite with coverage reporting.

```bash
./scripts/test.sh
```

Features:
- Unit, integration, and property-based tests
- Branch coverage ≥85%
- Line coverage ≥90%
- HTML and XML coverage reports
- Parallel execution with pytest-xdist

### lint.sh
Run all linting and type checking.

```bash
./scripts/lint.sh
```

Executes:
- ruff check (comprehensive Python linting)
- pylint (≥9.0 score required)
- mypy (strict type checking)
- pydocstyle (docstring coverage)

### format.sh
Auto-format all Python code.

```bash
./scripts/format.sh
```

Applies:
- ruff format (the single authoritative formatter — black and isort removed per ADR-0002)

Safe to run - only modifies formatting, not logic.

### security.sh
Run security vulnerability scanning.

```bash
./scripts/security.sh
```

Executes:
- bandit (Python security linter)
- pip-audit (dependency vulnerability scanner)
- detect-secrets (secret detection)

Zero tolerance for critical/high vulnerabilities.

### mutation.sh
Run mutation testing to verify test quality.

```bash
./scripts/mutation.sh
```

Uses mutmut to:
- Mutate source code systematically
- Verify tests catch mutations
- Require ≥80% mutation score

Long-running and MANUAL only — it is the pre-v1.0.0 release gate (owner directive,
issue #107), never an automated CI/pre-commit check. Run it locally or via the
`mutation-gate.yml` workflow (`gh workflow run mutation-gate.yml`) before shipping
v1.0.0.

---

## 🧪 Python-Specific Testing Patterns

**For comprehensive testing strategy, see [.claude/docs/testing.md](.claude/docs/testing.md)**

### Unit Test Example

```python
# tests/unit/test_calculator.py
import pytest
from hypothesis import given, strategies as st

from windbreak.calculator import Calculator


class TestCalculator:
    """Test suite for Calculator class."""

    @pytest.fixture
    def calculator(self):
        """Provide a Calculator instance."""
        return Calculator()

    def test_add_positive_numbers_returns_sum(self, calculator):
        """Adding two positive numbers returns their sum."""
        result = calculator.add(2, 3)
        assert result == 5

    def test_divide_by_zero_raises_value_error(self, calculator):
        """Dividing by zero raises ValueError with clear message."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calculator.divide(10, 0)

    @given(st.integers(), st.integers())
    def test_add_commutative_property(self, calculator, a, b):
        """Addition is commutative: a + b == b + a."""
        assert calculator.add(a, b) == calculator.add(b, a)
```

### Property-Based Testing with Hypothesis

```python
from hypothesis import given, strategies as st
from windbreak.serialization import serialize, deserialize


@given(st.dictionaries(st.text(), st.integers()))
def test_serialization_roundtrip_preserves_data(data):
    """Serialization followed by deserialization returns original data."""
    serialized = serialize(data)
    deserialized = deserialize(serialized)
    assert deserialized == data
```

### Fixture Organization

```python
# tests/conftest.py
import pytest
from hypothesis import settings, Verbosity


# Global Hypothesis configuration
settings.register_profile("ci", max_examples=1000, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=100)


@pytest.fixture(scope="session")
def database_connection():
    """Provide a test database connection."""
    conn = create_test_database()
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def reset_database(database_connection):
    """Reset database state before each test."""
    database_connection.execute("TRUNCATE TABLE users CASCADE")
```

---

## 📚 Python Code Standards

### Type Annotations (Required)

```python
from typing import Optional, List, Dict, TypeVar, Protocol
from collections.abc import Iterable

T = TypeVar('T')


def process_items(
    items: Iterable[T],
    *,
    max_count: Optional[int] = None,
    metadata: Dict[str, str] | None = None
) -> List[T]:
    """Process items with optional filtering.

    Args:
        items: Items to process.
        max_count: Maximum number of items to process. None means unlimited.
        metadata: Optional metadata dictionary.

    Returns:
        Processed items as a list.

    Raises:
        ValueError: If max_count is negative.
    """
    if max_count is not None and max_count < 0:
        raise ValueError("max_count must be non-negative")

    result: List[T] = []
    for item in items:
        if max_count is not None and len(result) >= max_count:
            break
        result.append(item)

    return result
```

### Docstring Style (Google Format)

```python
class RiskCalculator:
    """Calculate risk scores for financial instruments.

    This class provides methods to assess risk based on multiple factors
    including market volatility, historical performance, and credit ratings.

    Attributes:
        volatility_threshold: Maximum acceptable volatility (0.0 to 1.0).
        use_historical: Whether to include historical trend analysis.

    Example:
        >>> calc = RiskCalculator(volatility_threshold=0.25)
        >>> score = calc.calculate_risk(instrument_data)
        >>> print(f"Risk score: {score.value}")
    """

    def __init__(
        self,
        volatility_threshold: float,
        *,
        use_historical: bool = True
    ) -> None:
        """Initialize the risk calculator.

        Args:
            volatility_threshold: Maximum acceptable volatility (0.0 to 1.0).
            use_historical: Whether to include historical analysis.

        Raises:
            ValueError: If volatility_threshold is not between 0 and 1.
        """
        if not 0 <= volatility_threshold <= 1:
            raise ValueError("volatility_threshold must be between 0 and 1")

        self.volatility_threshold = volatility_threshold
        self.use_historical = use_historical
```

### Error Handling

```python
# Good: Specific exceptions with context
class InsufficientDataError(Exception):
    """Raised when insufficient data is available for calculation."""

    def __init__(self, required: int, available: int) -> None:
        self.required = required
        self.available = available
        super().__init__(
            f"Need {required} data points, only {available} available"
        )


def calculate_average(values: List[float]) -> float:
    """Calculate average of values.

    Args:
        values: List of numeric values.

    Returns:
        The arithmetic mean.

    Raises:
        InsufficientDataError: If fewer than 2 values provided.
        ValueError: If values contain NaN or infinity.
    """
    if len(values) < 2:
        raise InsufficientDataError(required=2, available=len(values))

    if any(not math.isfinite(v) for v in values):
        raise ValueError("Values must be finite numbers")

    return sum(values) / len(values)
```

### Context Managers

```python
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def temporary_config(config: Dict[str, Any]) -> Iterator[Config]:
    """Temporarily override configuration settings.

    Args:
        config: Configuration overrides.

    Yields:
        Config object with overrides applied.

    Example:
        >>> with temporary_config({"debug": True}) as cfg:
        ...     # Debug mode active here
        ...     process_data(cfg)
        >>> # Original config restored
    """
    original = get_current_config()
    try:
        merged = {**original, **config}
        set_current_config(merged)
        yield Config(merged)
    finally:
        set_current_config(original)
```

---

## 🚨 Common Mistakes (Quick Reference)

**For detailed examples, see [.claude/docs/troubleshooting.md](.claude/docs/troubleshooting.md)**

### 1. Skipping Local Quality Checks (35%)

**Bad:**
```bash
git add .
git commit -m "feat: add feature"
git push
# CI fails
```

**Good:**
```bash
./scripts/check-all.sh  # ALWAYS run first
git add .
git commit -m "feat: add feature"
git push
```

### 2. Lowering Quality Thresholds (25%)

**Bad:**
```toml
[tool.coverage.report]
fail_under = 70  # Lowered from 90
```
