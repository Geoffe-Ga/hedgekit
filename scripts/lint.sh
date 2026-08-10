#!/usr/bin/env bash
# scripts/lint.sh - Run linting checks (Ruff, float-lint, shellcheck)
# Usage: ./scripts/lint.sh [--fix] [--check] [--metrics] [--verbose] [--help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

FIX=false
METRICS=false
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

Run linting checks on the project:
    - Ruff        (Python lint)
    - float-lint  (no floats on money/price/probability paths)
    - shellcheck  (shell lint, via the pinned pre-commit hook so the local
                  verdict matches CI's by construction -- issue #359)

OPTIONS:
    --fix       Auto-fix linting issues where possible (Ruff only; neither
                float-lint nor shellcheck has an autofix, so both run the
                identical scan in --fix and check modes)
    --check     Check only, fail if issues found (default mode)
    --metrics   Print {"violations": <int>} as JSON on stdout and exit 0, for
                the quality dashboard (issue #122). Measurement, not a gate:
                findings are reported, never enforced. The count is null when
                any half of the lint could not be counted.
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
# Shell linting runs through pre-commit, never a bare `shellcheck` off PATH --
# see the "Shell lint" section below for why. Resolving pre-commit here (rather
# than at the point of use) keeps every tool this script needs proven present
# before any check reports a verdict.
PRE_COMMIT="$(bash "$TOOLCHAIN_ENV" --print-tool pre-commit)"

# Dashboard measurement mode (issue #122). Dispatched here -- after tool
# resolution, before any gate output -- so stdout carries exactly the one JSON
# object scripts/collect_metrics.py parses, and the gate path below is left
# untouched. `exec` guarantees nothing else in this script can run afterwards.
if $METRICS; then
    exec "$FLOAT_LINT_PYTHON" "$SCRIPT_DIR/metrics_probe.py" lint \
        --ruff "$RUFF" \
        --python "$FLOAT_LINT_PYTHON" \
        --pre-commit "$PRE_COMMIT"
fi

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

echo "=== Shell lint (shellcheck) ==="
# WHY THIS INVOKES THE HOOK INSTEAD OF THE shellcheck BINARY (issue #359)
#
# CI's Quality Checks job runs `pre-commit run --all-files`, which includes
# the shellcheck-py hook. Gate 1 used to run no shell lint at all, so
# `check-all.sh` exit 0 was not a superset of CI for any .sh change -- twice
# this cost a full red-CI round trip (SC2153 in mutation.sh; SC2209 in
# scripts/ralph/test_assert_review_posted.sh, where `pr_write_status=write`
# reads as capturing the output of the Unix `write` command).
#
# Running `shellcheck` off PATH would close the "no check" gap while opening a
# version-skew one: a local shellcheck newer or older than the `rev` pinned in
# .pre-commit-config.yaml disagrees about which findings exist, which is the
# same local/CI divergence wearing a different hat. Driving the PINNED HOOK
# makes the two agree BY CONSTRUCTION -- there is no second version to keep in
# lockstep, so no lockstep test can rot. The cost is a one-time hook-env install
# on a cold pre-commit cache; warm, this is a couple of seconds.
#
# --all-files matches CI's file set exactly. It is also honest about the working
# tree: with nothing staged, pre-commit lints the files as they are on disk, and
# with a partially-staged file it lints the staged content -- i.e. what a commit
# would actually contain. Neither path can report green on unlinted shell.
#
# The one file set it does NOT cover is a brand-new, still-untracked script:
# `--all-files` means `git ls-files`, so an unadded .sh is invisible here, as it
# is to CI. It stops being invisible the moment it is staged -- which it must be
# to be committed -- and the commit-time pre-commit hook lints it then. So the
# gap cannot reach a push; do not "fix" it by widening past CI's file set, which
# would reintroduce a local/CI disagreement in the opposite direction.
#
# No `command -v` probe guarding this, deliberately (issue #366): pre-commit is
# resolved from the pinned toolchain above, which vetoes when it is missing.
# A check that cannot run must fail loudly, never pass green.
#
# There is no autofix for shell findings, so --fix and check modes run the same
# scan. Note for anyone editing the prose above: a comment line beginning with
# "# shellcheck" is parsed by shellcheck as a DIRECTIVE (SC1073), so keep the
# word off the start of a line -- this check caught exactly that here.
if "$PRE_COMMIT" run shellcheck --all-files; then
    echo "✓ Shell-lint checks passed"
else
    echo "✗ Shell-lint checks failed" >&2
    exit 1
fi

exit 0
