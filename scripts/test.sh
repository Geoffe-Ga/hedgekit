#!/usr/bin/env bash
# scripts/test.sh - Run tests with Pytest
# Usage: ./scripts/test.sh [--unit|--integration|--e2e|--all] [--coverage]
#                          [--mutation] [--metrics] [--verbose] [--help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

TEST_TYPE="unit"
COVERAGE=false
METRICS=false
MUTATION=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metrics)
            METRICS=true
            shift
            ;;
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --e2e)
            TEST_TYPE="e2e"
            shift
            ;;
        --all)
            TEST_TYPE="all"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --mutation)
            MUTATION=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run tests using Pytest.

OPTIONS:
    --unit          Run unit tests only (default)
    --integration   Run integration tests only
    --e2e           Run end-to-end tests only
    --all           Run all test types
    --coverage      Generate coverage report
    --mutation      Run mutation tests
    --metrics       Print {"tests_total": <int>, "tests_passed": <int>,
                    "tests_failed": <int>, "tests_skipped": <int>} as JSON on
                    stdout and exit 0, for the quality dashboard (issue #122).
                    Measurement, not a gate: failures are counted, never
                    enforced, and mutation testing is never triggered (it is
                    the manual pre-v1.0.0 gate, issue #107). Shares its pytest
                    run with coverage.sh --metrics.
    --verbose       Show detailed output
    --help          Display this help message

EXIT CODES:
    0               All tests passed
    1               Test failures
    2               Error running tests

EXAMPLES:
    $(basename "$0")                     # Run unit tests
    $(basename "$0") --all               # Run all tests
    $(basename "$0") --unit --coverage   # Unit tests with coverage
    $(basename "$0") --mutation          # Run mutation tests
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
# an ambient pytest runs against an ambient site-packages, so the suite would be
# exercising a different dependency set than the one this repo pins.
TOOLCHAIN_ENV="$SCRIPT_DIR/toolchain-env.sh"
PYTEST="$(bash "$TOOLCHAIN_ENV" --print-tool pytest)"

# Dashboard measurement mode (issue #122). Dispatched before the gate below so
# stdout carries exactly the one JSON object collect_metrics.py parses. It runs
# the WHOLE suite (not the default `--unit` selection) because the dashboard's
# Test Count tile counts the repository's tests, and it shares that run's
# artifacts with coverage.sh --metrics.
if $METRICS; then
    METRICS_PYTHON="$(bash "$TOOLCHAIN_ENV" --print-python)"
    exec "$METRICS_PYTHON" "$SCRIPT_DIR/metrics_probe.py" tests --pytest "$PYTEST"
fi

# Set verbosity
if $VERBOSE; then
    set -x
fi

# Build pytest arguments
PYTEST_ARGS=(-v)

# Every -m below must carry `and not container`, and this is not optional.
# pytest takes the LAST -m it is given, so any selection here REPLACES the
# `-m "not container"` in pyproject.toml's addopts rather than intersecting
# with it -- silently re-selecting the tier that exists precisely because it is
# too expensive for the default run (issues #466, #467, #468). Until issue #468
# there was exactly one container test and it skipped for want of systemd, so
# the leak cost nothing visible; it does now, because Gate 1's unit step would
# otherwise run a `docker build` and five `docker compose up` cycles, and CI's
# `quality` matrix would run all of it three more times. The `all` case adds no
# -m at all, so pyproject's deselection stands there unaided.
# tests/e2e/test_tier_selection_contract.py fails if this guard is dropped.
case "$TEST_TYPE" in
    unit)
        echo "=== Running Unit Tests ==="
        PYTEST_ARGS+=(-m "not integration and not e2e and not container")
        ;;
    integration)
        echo "=== Running Integration Tests ==="
        PYTEST_ARGS+=(-m "integration and not container")
        ;;
    e2e)
        echo "=== Running End-to-End Tests ==="
        # `and not container` is not redundant with `e2e`: a module may carry
        # both markers, and this script is the process-level tier only. The
        # container tier has its own entry point, scripts/e2e.sh --container.
        PYTEST_ARGS+=(-m "e2e and not container")
        ;;
    all)
        echo "=== Running All Tests ==="
        ;;
esac

# Add coverage if requested
if $COVERAGE; then
    echo "Coverage enabled"
    PYTEST_ARGS+=(
        --cov=windbreak
        --cov-branch
        --cov-report=term-missing
        --cov-report=html
        --cov-report=xml
        --cov-fail-under=90
    )
fi

# Run tests
if $VERBOSE; then
    echo "Running pytest with args: ${PYTEST_ARGS[*]}"
fi

"$PYTEST" "${PYTEST_ARGS[@]}" tests/ || { echo "✗ Tests failed" >&2; exit 1; }

echo "✓ Tests passed"

# Run mutation tests if requested
if $MUTATION; then
    echo "=== Running Mutation Tests ==="
    # Resolved (not `command -v`-guarded) so an absent mutmut vetoes instead of
    # reporting a mutation run that never happened.
    MUTMUT="$(bash "$TOOLCHAIN_ENV" --print-tool mutmut)"
    "$MUTMUT" run || { echo "✗ Mutation tests failed" >&2; exit 1; }
    echo "✓ Mutation tests passed"
fi

exit 0
