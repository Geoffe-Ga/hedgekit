#!/usr/bin/env bash
# scripts/metrics-docs.sh - Report docstring coverage for the quality dashboard
# Usage: ./scripts/metrics-docs.sh [--metrics] [--verbose] [--help]
#
# WHY THIS SCRIPT EXISTS (issue #122)
#
# scripts/collect_metrics.py's `collect_docs_metrics()` runs
# `scripts/metrics-docs.sh --metrics` for the dashboard's Documentation
# Coverage tile. The file did not exist, so `collect_from_script()` returned
# None and the tile rendered N/A -- indistinguishable, to a glancing operator,
# from a healthy one.
#
# THIS IS A REPORTER, NOT A GATE, AND THAT IS DELIBERATE
#
# Docstring presence is ENFORCED by ruff's `D1` family through
# `scripts/lint.sh` (issue #351), at 100% presence with no percentage to hide
# behind. Duplicating that enforcement here would create a second opinion to
# keep in lockstep. So this script measures with the same tool, over the same
# file set, and reports -- the numerator is ruff's own `D1` findings, so the
# tile and the gate cannot disagree about what is missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

METRICS=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metrics)
            METRICS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Measure docstring coverage using ruff's D1 (missing-docstring) rules over
exactly the files ruff lints. Enforcement lives in ./scripts/lint.sh; this
script only reports, so it never fails on a low number.

OPTIONS:
    --metrics   Print {"docs_coverage_pct": <num>} as JSON on stdout and exit
                0, for the quality dashboard (issue #122). The value is null
                when either half of the ratio could not be established.
    --verbose   Show detailed output
    --help      Display this help message

EXIT CODES:
    0           Measurement completed (whatever the number)
    2           Error running the measurement (e.g. ruff unresolvable)

EXAMPLES:
    $(basename "$0") --metrics   # {"docs_coverage_pct": 100.0}
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
# A failed resolution exits nonzero here and, under `set -e`, aborts the run:
# a measurement produced by an unknown environment is not evidence.
TOOLCHAIN_ENV="$SCRIPT_DIR/toolchain-env.sh"
RUFF="$(bash "$TOOLCHAIN_ENV" --print-tool ruff)"
DOCS_PYTHON="$(bash "$TOOLCHAIN_ENV" --print-python)"

if $METRICS; then
    exec "$DOCS_PYTHON" "$SCRIPT_DIR/metrics_probe.py" docs --ruff "$RUFF"
fi

# Set verbosity
if $VERBOSE; then
    set -x
fi

echo "=== Documentation Coverage (ruff D1) ==="
"$DOCS_PYTHON" "$SCRIPT_DIR/metrics_probe.py" docs --ruff "$RUFF"
echo "Enforcement lives in ./scripts/lint.sh (ruff D1, 100% presence)."

exit 0
