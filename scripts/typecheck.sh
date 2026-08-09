#!/usr/bin/env bash
# scripts/typecheck.sh - Run type checking with MyPy
# Usage: ./scripts/typecheck.sh [--verbose] [--help]

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

Run type checking on the project using MyPy.

OPTIONS:
    --verbose   Show detailed output
    --help      Display this help message

EXIT CODES:
    0           All type checks passed
    1           Type errors found
    2           Error running type checker

EXAMPLES:
    $(basename "$0")          # Run type checking
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

# Resolve mypy from the PINNED toolchain, not the caller's PATH (issue #366).
# This also replaces a `command -v mypy` guard that SKIPPED type checking when
# mypy was missing: a check that silently reports success when it did not run is
# exactly the false-green this gate exists to prevent, so a missing mypy now
# vetoes (toolchain_tool fails closed).
MYPY="$(bash "$SCRIPT_DIR/toolchain-env.sh" --print-tool mypy)"

# Set verbosity
if $VERBOSE; then
    set -x
fi

echo "=== Type Checking (MyPy) ==="

# Issue #179: local Gate-2 mypy must match CI's cold-cache run. mypy's
# incremental warm `.mypy_cache` skips re-validating cross-module
# re-export surfaces, so implicit re-exports (no-implicit-reexport) and
# attr-defined errors false-green locally yet fail on CI's fresh
# checkout. We force a genuinely empty cache every run by pointing
# MYPY_CACHE_DIR at a fresh per-run temp dir; this DELIBERATELY
# overrides any inherited MYPY_CACHE_DIR (enforcement, not opt-in).
# Measured tradeoff: cold ~1.16s vs warm ~0.14s -- negligible, so cold
# is the enforced default with no opt-out flag (the prior advisory
# mitigation was missed twice precisely because it was manual). We use
# MYPY_CACHE_DIR rather than `rm -rf .mypy_cache` because the latter can
# be sandbox-denied for fleet workers; the EXIT trap deletes only our
# own $TMPDIR dir, never the repo `.mypy_cache`.
MYPY_CACHE_DIR="$(mktemp -d)"
export MYPY_CACHE_DIR
trap 'rm -rf "$MYPY_CACHE_DIR"' EXIT
"$MYPY" windbreak/ scripts/ || {
    echo "✗ Type checking failed" >&2
    exit 1
}
echo "✓ Type checking passed"

exit 0
