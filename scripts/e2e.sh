#!/usr/bin/env bash
# scripts/e2e.sh - Run the end-to-end verification tier
# Usage: ./scripts/e2e.sh [--container] [--all] [--verbose] [--help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SELECTION="e2e"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --container)
            SELECTION="container"
            shift
            ;;
        --all)
            SELECTION="e2e or container"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run the end-to-end verification tier (epic #465): tests that exercise the
SHIPPED ARTIFACT -- installed package, built image, compose stack, systemd
units, real OS processes -- rather than an in-process object graph a test
file assembled itself.

TIERS:
    e2e         Spawns real windbreak processes. Needs no container runtime,
                so it also runs inside the default suite and Gate 1.
    container   Needs a docker daemon or a running systemd. Deselected from
                the default suite; tests SKIP (never silently pass) when the
                runtime is absent.

OPTIONS:
    --container Run the container tier only
    --all       Run both tiers
    --verbose   Show detailed output
    --help      Display this help message

EXIT CODES:
    0           Selected tier passed (or skipped for a missing runtime)
    1           A tier test failed
    2           Error running the tier

EXAMPLES:
    $(basename "$0")             # Run the process-level e2e tier
    $(basename "$0") --container # Run the container tier
    $(basename "$0") --all       # Run both
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

# Resolve pytest from the PINNED toolchain, not the caller's PATH (issue #366).
# This tier spawns child processes with `sys.executable`, so the interpreter
# pytest runs under IS the interpreter under test -- resolving it ambiently
# would silently exercise a different installation than the one this repo pins.
TOOLCHAIN_ENV="$SCRIPT_DIR/toolchain-env.sh"
PYTEST="$(bash "$TOOLCHAIN_ENV" --print-tool pytest)"

# Set verbosity
if $VERBOSE; then
    set -x
fi

echo "=== End-to-End Tier (${SELECTION}) ==="

# --no-cov is required, not cosmetic, and is NOT a gate weakening. The pytest
# addopts in pyproject.toml carry --cov=windbreak --cov-fail-under=90, but this
# tier exercises windbreak in CHILD processes, whose lines the parent's coverage
# collector cannot see. Measured against a marker-filtered subset it would
# therefore always report near-zero and fail the 90% gate on a green run. The
# 90% floor is enforced once, by the full suite in scripts/test.sh and
# scripts/check-all.sh, where these tests also run and are counted. Selecting
# with -m (rather than --override-ini or -o addopts=) keeps --strict-markers
# and --strict-config active.
"$PYTEST" -m "$SELECTION" --no-cov -v tests/ \
    || { echo "✗ End-to-end tier failed" >&2; exit 1; }

echo "✓ End-to-end tier passed"
exit 0
