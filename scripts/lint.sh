#!/usr/bin/env bash
# scripts/lint.sh - Run linting checks with Ruff
# Usage: ./scripts/lint.sh [--fix] [--check] [--verbose] [--help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

FIX=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX=true
            shift
            ;;
        --check)
            # Check-only is the default behaviour; accept the flag as a
            # no-op so callers (e.g. check-all.sh) can pass it explicitly.
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run linting checks on the project using Ruff.

OPTIONS:
    --fix       Auto-fix linting issues where possible
    --check     Check only, fail if issues found (default mode)
    --verbose   Show detailed output
    --help      Display this help message

EXIT CODES:
    0           All checks passed
    1           Linting issues found
    2           Error running checks

EXAMPLES:
    $(basename "$0")              # Run checks in check mode
    $(basename "$0") --fix         # Auto-fix issues
    $(basename "$0") --verbose     # Show detailed output
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

# Resolve tools from the PINNED toolchain, not the caller's PATH (issue #366).
# A failed resolution exits nonzero here and, under `set -e`, aborts the check.
TOOLCHAIN_ENV="$SCRIPT_DIR/toolchain-env.sh"
RUFF="$(bash "$TOOLCHAIN_ENV" --print-tool ruff)"
FLOAT_LINT_PYTHON="$(bash "$TOOLCHAIN_ENV" --print-python)"

# Set verbosity
if $VERBOSE; then
    set -x
fi

echo "=== Linting (Ruff) ==="

if $FIX; then
    if $VERBOSE; then
        echo "Fixing linting issues..."
    fi
    "$RUFF" check . --fix
    EXIT_CODE=$?
else
    if $VERBOSE; then
        echo "Checking for linting issues..."
    fi
    "$RUFF" check .
    EXIT_CODE=$?
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo "✗ Linting checks failed" >&2
    exit 1
fi
echo "✓ Linting checks passed"

echo "=== Float lint (AST) ==="
# Enforce "no floats on the money path" (SPEC S6.1/S17.3). No autofix exists,
# so --fix and check modes run the identical scan of the denylisted packages.
if "$FLOAT_LINT_PYTHON" scripts/lint_no_floats.py; then
    echo "✓ Float-lint checks passed"
else
    echo "✗ Float-lint checks failed" >&2
    exit 1
fi

exit 0
