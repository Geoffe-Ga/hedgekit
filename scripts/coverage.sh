#!/usr/bin/env bash
# scripts/coverage.sh - Run tests with coverage report
# Usage: ./scripts/coverage.sh [--html] [--xml] [--metrics] [--verbose] [--help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

HTML_REPORT=false
METRICS=false
XML_REPORT=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metrics)
            METRICS=true
            shift
            ;;
        --html)
            HTML_REPORT=true
            shift
            ;;
        --xml)
            XML_REPORT=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run tests with coverage report.

OPTIONS:
    --html      Generate HTML coverage report
    --xml       Generate XML coverage report (for CI)
    --metrics   Print {"coverage_pct": <num>, "branch_coverage_pct": <num>}
                as JSON on stdout and exit 0, for the quality dashboard
                (issue #122). Measurement, not a gate: the suite runs without
                --cov-fail-under, so a coverage regression is reported rather
                than enforced. Shares its pytest run with test.sh --metrics.
    --verbose   Show detailed output
    --help      Display this help message

EXIT CODES:
    0           Coverage threshold met
    1           Coverage below threshold
    2           Error running coverage

EXAMPLES:
    $(basename "$0")          # Run coverage with terminal report
    $(basename "$0") --html   # Generate HTML report
    $(basename "$0") --xml    # Generate XML report for CI
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

# Resolve pytest from the PINNED toolchain, not the caller's PATH (issue #366):
# the coverage threshold must be measured against the pinned dependency set, not
# whichever pytest/pytest-cov pair happens to be first on PATH.
PYTEST="$(bash "$SCRIPT_DIR/toolchain-env.sh" --print-tool pytest)"

# Dashboard measurement mode (issue #122). Dispatched before the gate below so
# stdout carries exactly the one JSON object collect_metrics.py parses. The
# probe writes a coverage report and a JUnit report from ONE suite run and
# reuses them for test.sh --metrics while no source or test file is newer, so
# a full collection runs the suite once rather than twice.
if $METRICS; then
    METRICS_PYTHON="$(bash "$SCRIPT_DIR/toolchain-env.sh" --print-python)"
    exec "$METRICS_PYTHON" "$SCRIPT_DIR/metrics_probe.py" coverage --pytest "$PYTEST"
fi

# Set verbosity
if $VERBOSE; then
    set -x
fi

echo "=== Coverage Report ==="

# Build pytest arguments
PYTEST_ARGS=(
    -v
    --cov=windbreak
    --cov-branch
    --cov-report=term-missing
    --cov-fail-under=90
)

# Add HTML report if requested
if $HTML_REPORT; then
    PYTEST_ARGS+=(--cov-report=html)
    echo "HTML report will be generated in htmlcov/"
fi

# Add XML report if requested
if $XML_REPORT; then
    PYTEST_ARGS+=(--cov-report=xml)
    echo "XML report will be generated as coverage.xml"
fi

# Run tests with coverage
"$PYTEST" "${PYTEST_ARGS[@]}" tests/ || {
    echo "✗ Coverage below threshold" >&2
    exit 1
}

echo "✓ Coverage threshold met"

exit 0
