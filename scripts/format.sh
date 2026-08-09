#!/usr/bin/env bash
# scripts/format.sh - Format code with ruff format (single formatter authority — issue #104)
# Usage: ./scripts/format.sh [--fix] [--check] [--verbose] [--help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

FIX=true
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX=true
            shift
            ;;
        --check)
            FIX=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Format code using ruff format (single formatter authority — issue #104).
Import sorting is handled by ruff's I lint rule in lint.sh, not here.

OPTIONS:
    --fix       Apply formatting changes (default)
    --check     Check only, fail if changes needed
    --verbose   Show detailed output
    --help      Display this help message

EXIT CODES:
    0           Code is properly formatted
    1           Formatting issues found
    2           Error running checks

EXAMPLES:
    $(basename "$0") --fix         # Apply formatting
    $(basename "$0") --check       # Check only
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

# Resolve ruff from the PINNED toolchain, not the caller's PATH (issue #366):
# a different ruff build reformats differently, so a PATH-chosen formatter makes
# `--check` disagree with CI for reasons nobody can see in the diff.
RUFF="$(bash "$SCRIPT_DIR/toolchain-env.sh" --print-tool ruff)"

# Set verbosity
if $VERBOSE; then
    set -x
fi

echo "=== Formatting (ruff format) ==="

if $FIX; then
    if $VERBOSE; then
        echo "Running ruff format..."
    fi
    "$RUFF" format . || { echo "✗ ruff format failed" >&2; exit 1; }
    echo "✓ Code formatted successfully"
else
    if $VERBOSE; then
        echo "Running ruff format --check..."
    fi
    "$RUFF" format --check . || { echo "✗ ruff format check failed" >&2; exit 1; }
    echo "✓ Code formatting check passed"
fi
exit 0
