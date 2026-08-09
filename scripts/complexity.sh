#!/usr/bin/env bash
# scripts/complexity.sh - Code complexity analysis
# Usage: ./scripts/complexity.sh [--verbose] [--help]

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

Analyze code complexity using Radon and Xenon.

Metrics:
  - Cyclomatic complexity (should be <= 10)
  - Maintainability index (should be >= 20)
  - Cognitive complexity

OPTIONS:
    --verbose   Show detailed output
    --help      Display this help message

EXIT CODES:
    0           Complexity acceptable
    1           Complexity exceeds thresholds
    2           Error during analysis

EXAMPLES:
    $(basename "$0")          # Analyze complexity
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

# Resolve radon/xenon from the PINNED toolchain, not the caller's PATH (issue
# #366). This also replaces `command -v` guards that SKIPPED the checks when the
# tools were missing -- xenon is the only enforcing half of this script, so a
# silent skip turned the whole gate into a check that could not fail. A missing
# tool now vetoes (toolchain_tool fails closed).
TOOLCHAIN_ENV="$SCRIPT_DIR/toolchain-env.sh"
RADON="$(bash "$TOOLCHAIN_ENV" --print-tool radon)"
XENON="$(bash "$TOOLCHAIN_ENV" --print-tool xenon)"

# Set verbosity
if $VERBOSE; then
    set -x
fi

echo "=== Code Complexity Analysis ==="

# Radon reports only -- xenon below is what enforces the thresholds.
echo ""
echo "Cyclomatic Complexity (should be <= 10):"
"$RADON" cc -a windbreak/ || true

echo ""
echo "Maintainability Index (should be >= 20):"
"$RADON" mi -a windbreak/ || true

# Check complexity with Xenon
if $VERBOSE; then
    echo "Running Xenon complexity check..."
fi
"$XENON" --max-absolute B --max-modules B --max-average B windbreak/ \
    || { echo "✗ Complexity exceeds thresholds" >&2; exit 1; }

echo "✓ Complexity analysis completed"
exit 0
