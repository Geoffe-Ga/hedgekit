#!/usr/bin/env bash
# scripts/check-all.sh - Run all quality checks
# Usage: ./scripts/check-all.sh [--verbose] [--help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run all quality checks in sequence.

Runs:
  1. Linting (Ruff + float-lint + shellcheck)
  2. Architecture boundary checks (import-linter)
  3. Formatting (ruff format)
  4. Type checking (MyPy)
  5. Security checks (Bandit + pip-audit + detect-secrets)
  6. Complexity analysis (Radon)
  7. Unit tests
  8. Coverage report

OPTIONS:
    --verbose   Show detailed output
    --help      Display this help message

EXIT CODES:
    0           All checks passed
    1           One or more checks failed
    2           Error running checks

EXAMPLES:
    $(basename "$0")          # Run all checks
    $(basename "$0") --verbose # Show detailed output
EOF
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Shared pinned-toolchain venv (issue #133): resolve the single shared .venv at
# the main repo root. When it exists, fail loudly on drift and prepend it to
# PATH so any subprocess a tool spawns also sees the pinned toolchain. The venv
# is OPTIONAL -- CI installs the toolchain into the runner's interpreter instead,
# and toolchain-env.sh announces that fallback out loud rather than silently.
#
# The sub-scripts no longer *depend* on this PATH: since issue #366 each one
# resolves its own tools by absolute path through toolchain-env.sh, so a check
# run directly reaches the identical verdict it reaches through Gate 1.
VENV_DIR="$(bash "$SCRIPT_DIR/toolchain-env.sh" --print-venv)"
if [[ -n "$VENV_DIR" ]]; then
    bash "$SCRIPT_DIR/provision-venv.sh" --check
    export PATH="$VENV_DIR/bin:$PATH"  # .venv/bin first so the pinned toolchain wins
fi
echo ""

# Set verbosity
VERBOSE_FLAG=""
if $VERBOSE; then
    VERBOSE_FLAG="--verbose"
fi

echo "=== Running All Quality Checks ==="
echo ""

FAILED_CHECKS=()
PASSED_CHECKS=()

# Helper function to run a check
run_check() {
    local check_name=$1
    local script=$2
    shift 2
    local args=("$@")

    echo "Running: $check_name"
    if "$SCRIPT_DIR/$script" "${args[@]+"${args[@]}"}" $VERBOSE_FLAG; then
        PASSED_CHECKS+=("$check_name")
        echo "✓ $check_name passed"
    else
        FAILED_CHECKS+=("$check_name")
        echo "✗ $check_name failed" >&2
    fi
    echo ""
}

# Run all checks
run_check "Linting" "lint.sh" --check
run_check "Architecture (import-linter)" "architecture.sh"
run_check "Formatting" "format.sh" --check
run_check "Type checking" "typecheck.sh"
run_check "Security checks" "security.sh"
run_check "Complexity analysis" "complexity.sh"
run_check "Unit tests" "test.sh" --unit
run_check "Coverage report" "coverage.sh"

echo "=== Quality Checks Summary ==="
echo "Passed: ${#PASSED_CHECKS[@]}"
echo "Failed: ${#FAILED_CHECKS[@]}"

if [ ${#FAILED_CHECKS[@]} -gt 0 ]; then
    echo ""
    echo "Failed checks:"
    for check in "${FAILED_CHECKS[@]}"; do
        echo "  ✗ $check"
    done
    exit 1
else
    echo ""
    echo "✓ All quality checks passed!"
    exit 0
fi
