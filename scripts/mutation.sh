#!/usr/bin/env bash
# scripts/mutation.sh - Run mutation tests with score validation
# Usage: ./scripts/mutation.sh [--min-score SCORE] [--verbose] [--help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MIN_SCORE=80  # MAXIMUM QUALITY: 80% mutation score minimum
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min-score)
            MIN_SCORE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run mutation tests and validate minimum score threshold.

Mutation testing introduces small changes (mutations) to your code
to verify that your test suite catches them. A high mutation score
indicates effective tests.

OPTIONS:
    --min-score SCORE   Minimum mutation score (default: 80)
    --verbose           Show detailed output
    --help              Display this help message

EXIT CODES:
    0                   Mutation score meets or exceeds minimum
    1                   Mutation score below minimum threshold
    2                   Error running mutation tests

QUALITY STANDARDS:
    MAXIMUM QUALITY:    80% minimum mutation score
    Good:               70-79%
    Acceptable:         60-69%
    Poor:               <60%

EXAMPLES:
    $(basename "$0")                    # Run with 80% minimum
    $(basename "$0") --min-score 70     # Run with 70% minimum
    $(basename "$0") --verbose          # Show detailed output
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

# Set verbosity
if $VERBOSE; then
    set -x
fi

# Resolve mutmut from the PINNED toolchain, not the caller's PATH (issue #366).
# This is a release gate, so which mutmut runs -- and against which installed
# dependency set -- must be a property of the repo, not of the invoking shell.
# A failed resolution exits nonzero here and, under `set -e`, aborts the run.
MUTMUT="$(bash "$SCRIPT_DIR/toolchain-env.sh" --print-tool mutmut)"
STATS_PYTHON="$(bash "$SCRIPT_DIR/toolchain-env.sh" --print-python)"

echo "=== Running Mutation Tests ==="
echo "Minimum required score: ${MIN_SCORE}%"
echo ""

# Run mutation tests (allow failure, we check the score ourselves).
echo "Running mutmut (this may take a long time)..."
if "$MUTMUT" run 2>&1; then
    echo "✓ Mutmut run completed"
else
    # mutmut exits non-zero when mutants survive, which is not a script error.
    echo "Info: Mutmut run completed (some mutants may have survived)"
fi

echo ""
echo "=== Mutation Test Results ==="

# mutmut 3.x (issue #324) removed the `junitxml` subcommand and changed the
# human-readable `results` output, so the 2.x approach -- grep the prose for
# "Killed: N" -- now silently matches nothing. 3.x ships `export-cicd-stats`
# for exactly this purpose: it writes machine-readable counts to
# mutants/mutmut-cicd-stats.json so a pipeline can gate on the score. Parsing
# that JSON replaces the old prose scraping, which was brittle even in 2.x.
STATS_FILE="mutants/mutmut-cicd-stats.json"
rm -f "$STATS_FILE"

if ! "$MUTMUT" export-cicd-stats; then
    echo "Error: 'mutmut export-cicd-stats' failed" >&2
    exit 2
fi

# Fail CLOSED when there are no stats at all. The previous implementation
# exited 0 here ("Skipping mutation score validation"), which meant a
# misconfigured mutmut that produced zero mutants silently PASSED the release
# gate -- the exact failure mode a major-version config change provokes. A
# release gate that cannot measure must not report success.
if [ ! -f "$STATS_FILE" ]; then
    echo "Error: mutmut produced no stats at $STATS_FILE" >&2
    echo "This means no mutants were generated -- a mutmut configuration" >&2
    echo "problem, not a passing test suite. Check [tool.mutmut] in" >&2
    echo "pyproject.toml (source_paths must be a LIST of paths)." >&2
    exit 2
fi

cat "$STATS_FILE"
echo ""

# Reduce the stats to integers. `total` deliberately is NOT used as the
# denominator: it counts every mutant including not-yet-checked ones (mutmut
# prints progress as "total - not_checked / total"), and `not_checked` is not
# exported -- so scoring against `total` would understate an interrupted run.
# Instead the denominator is rebuilt from the explicit outcome categories, and
# any shortfall against `total` is treated as an incomplete run.
STATS_VARS=$("$STATS_PYTHON" - "$STATS_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)


def count(key: str) -> int:
    return int(data.get(key) or 0)


killed = count("killed")
survived = count("survived")
no_tests = count("no_tests")
skipped = count("skipped")
suspicious = count("suspicious")
timeout = count("timeout")
segfault = count("segfault")
total = count("total")
interrupted = 1 if data.get("check_was_interrupted_by_user") else 0

# Every category that represents a real mutant verdict. `skipped` is counted
# here so the completeness check below balances against `total`, then removed
# from the scored denominator -- a deliberately skipped mutant is neither a
# success nor a failure of the test suite.
accounted = killed + survived + no_tests + skipped + suspicious + timeout + segfault
scored = accounted - skipped

# One space-separated line, in the fixed order the `read` below expects.
print(
    " ".join(
        str(value)
        for value in (
            killed,
            survived,
            no_tests,
            skipped,
            suspicious,
            timeout,
            segfault,
            total,
            accounted,
            scored,
            interrupted,
        )
    )
)
PY
)

# Assigned via `read` rather than `eval` of NAME=VALUE pairs. eval would execute
# whatever the file expanded to, and it also hides the assignments from static
# analysis, which then reports every downstream use as SC2153 "may not be
# assigned". Naming the variables here keeps them statically visible and means
# nothing from the JSON is ever executed.
#
# (This comment deliberately avoids opening a line with the linter's own name:
# a comment starting with that word is parsed as a directive, SC1073.)
if [ -z "$STATS_VARS" ]; then
    echo "Error: could not parse $STATS_FILE" >&2
    exit 2
fi

read -r KILLED SURVIVED NO_TESTS SKIPPED SUSPICIOUS TIMEOUT SEGFAULT \
    TOTAL ACCOUNTED SCORED INTERRUPTED <<EOF
$STATS_VARS
EOF

if [ -z "$INTERRUPTED" ]; then
    echo "Error: incomplete stats parsed from $STATS_FILE" >&2
    exit 2
fi

if [ "$INTERRUPTED" -ne 0 ]; then
    echo "Error: the mutation run was interrupted before it finished" >&2
    echo "A partial run cannot be scored against the release gate." >&2
    exit 2
fi

if [ "$ACCOUNTED" -lt "$TOTAL" ]; then
    echo "Error: incomplete mutation run -- only $ACCOUNTED of $TOTAL" >&2
    echo "mutants have a recorded verdict. Re-run 'mutmut run' to" >&2
    echo "completion before scoring the release gate." >&2
    exit 2
fi

if [ "$SCORED" -le 0 ]; then
    echo "Error: no scorable mutants were generated" >&2
    echo "  - Is there code to mutate under source_paths?" >&2
    echo "  - Is [tool.mutmut] in pyproject.toml valid for mutmut 3.x?" >&2
    exit 2
fi

# Mutation score (killed / scorable * 100). Compared with 2.x this denominator
# additionally includes `no_tests` and `segfault`, both of which are mutants the
# suite did NOT kill. Counting them can only lower the measured score, so the
# gate becomes stricter, never weaker, across this migration.
SCORE=$(awk "BEGIN {printf \"%.1f\", ($KILLED / $SCORED) * 100}")

echo "=== Mutation Score ==="
echo "Killed:      $KILLED"
echo "Survived:    $SURVIVED"
echo "No tests:    $NO_TESTS"
echo "Suspicious:  $SUSPICIOUS"
echo "Timeout:     $TIMEOUT"
echo "Segfault:    $SEGFAULT"
echo "Skipped:     $SKIPPED (excluded from the score)"
echo "Scorable:    $SCORED"
echo ""
echo "Mutation Score: ${SCORE}%"
echo "Required:       ${MIN_SCORE}%"
echo ""

# Compare score to threshold
if awk "BEGIN {exit !($SCORE >= $MIN_SCORE)}"; then
    echo "✓ Mutation score meets minimum threshold"
    echo ""

    if [ "$SURVIVED" -gt 0 ]; then
        echo "Note: $SURVIVED mutants survived. To view them:"
        echo "  mutmut results          # list mutants by outcome"
        echo "  mutmut browse           # interactive browser"
        echo "  mutmut show <mutant>    # one mutant's diff"
    fi

    exit 0
else
    echo "✗ Mutation score below minimum threshold" >&2
    echo "" >&2
    echo "Your test suite killed ${SCORE}% of scorable mutants" >&2
    echo "Minimum required: ${MIN_SCORE}%" >&2
    echo "" >&2
    echo "To improve mutation score:" >&2
    echo "  1. List surviving mutants: mutmut results" >&2
    echo "  2. Inspect one: mutmut show <mutant>" >&2
    echo "  3. Add tests that kill it, then re-run" >&2
    echo "  4. Browse interactively: mutmut browse" >&2
    echo "" >&2

    if [ "$SURVIVED" -gt 0 ]; then
        echo "Surviving mutants:" >&2
        "$MUTMUT" results 2>&1 | head -20 || true
    fi

    exit 1
fi
