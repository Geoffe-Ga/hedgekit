#!/usr/bin/env bash
# scripts/precommit.sh - Run the WHOLE pre-commit hook set as a Gate 1 check
# Usage: ./scripts/precommit.sh [--verbose] [--help]

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

Run EVERY hook in .pre-commit-config.yaml over --all-files, through the
pinned toolchain, so Gate 1's hook coverage is a superset of CI's
"Pre-commit (all files)" job BY CONSTRUCTION rather than by a
hand-maintained list that drifts (issue #401).

One hook is excluded, deliberately and by name -- see GATE1_SKIPPED_HOOKS
below for the hook and the reason.

This check can MODIFY the working tree: several hooks are fixers
(trailing-whitespace, end-of-file-fixer, mixed-line-ending,
fix-byte-order-marker, ruff-check --fix, ruff-format). pre-commit reports
failure whenever a hook changes a file, so Gate 1 can never pass on
unfixed content; re-run to confirm green and review the diff (printed by
--show-diff-on-failure) before committing.

OPTIONS:
    --verbose   Show per-hook detail (passes --verbose through to pre-commit)
    --help      Display this help message

EXIT CODES:
    0           Every dispatched hook passed
    1           One or more hooks failed (or a fixer changed a file)
    2           Error running the check (e.g. pre-commit unresolvable)

EXAMPLES:
    $(basename "$0")           # run the full hook set
    $(basename "$0") --verbose # show each hook's output
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

# Resolve pre-commit from the PINNED toolchain, never by bare name (issue #366).
# A missing pre-commit vetoes here, loudly, under `set -e`; there is deliberately
# no `command -v pre-commit || skip` probe, because a check that reports success
# without running is strictly worse than one that refuses to run.
TOOLCHAIN_ENV="$SCRIPT_DIR/toolchain-env.sh"
PRE_COMMIT="$(bash "$TOOLCHAIN_ENV" --print-tool pre-commit)"

# THE ONE NAMED EXCLUSION, AND WHY (issue #401 acceptance criterion 1)
#
# no-commit-to-branch guards a LOCAL COMMIT-TIME branch policy: "do not commit
# directly to main". It asserts a property of the current branch, not of the
# tree's quality, so running it from a quality gate is a category error -- a
# developer legitimately running ./scripts/check-all.sh while sitting on main
# (to verify a merge, say) would get a red Gate 1 that says nothing about the
# code. CI's "Pre-commit (all files)" step skips it by name for the same reason
# (issue #127: push-event runs attach HEAD to main, so it fails on every merge).
#
# Matching CI's SKIP set exactly is what makes Gate 1 a superset of CI's
# pre-commit job rather than merely an overlapping set, and
# tests/toolchain/test_precommit_gate_coverage.py asserts that equality plus a
# stated reason for every excluded hook. Adding a name here without registering
# its reason there fails the suite.
#
# The hook is NOT weakened by this: it still runs at commit time via
# `pre-commit install`, which is where a branch policy belongs.
GATE1_SKIPPED_HOOKS="no-commit-to-branch"

PRE_COMMIT_ARGS=(run --all-files --show-diff-on-failure)
if $VERBOSE; then
    PRE_COMMIT_ARGS+=(--verbose)
fi

echo "=== Pre-commit (all hooks) ==="
# WHY GATE 1 RUNS THE WHOLE HOOK SET (issue #401)
#
# Gate 1 used to dispatch a hand-maintained SUBSET of the hooks, hook by hook,
# by name -- and the list drifted. shellcheck was missing until issue #359;
# vulture was missing until this change, and failed CI on PR #398 (unused
# variable in windbreak/net/live_http.py) after a local Gate 1 run of 8/8
# exit 0. Sixteen file-hygiene hooks had never been dispatched at all.
#
# Adding the next missing hook by name does not converge: the gap silently
# reopens the moment a hook is added to .pre-commit-config.yaml. Running the
# whole set removes the class of defect instead of its current instance --
# there is no per-hook list left to fall out of date.
#
# --all-files is exactly CI's file set (CI runs the same invocation). It is
# also honest about the working tree: with nothing staged pre-commit sees the
# files on disk, and with a partially-staged file it sees the staged content,
# i.e. what a commit would actually contain.
#
# Deliberate overlap: scripts/lint.sh and scripts/security.sh still drive their
# own hooks (shellcheck, detect-secrets), and ruff/mypy/bandit/pip-audit run
# both here and in their dedicated gate scripts. That is not dead duplication.
# Each gate script must stay a complete check for its own dimension when invoked
# standalone -- CLAUDE.md documents `./scripts/lint.sh` as a thing you run on
# its own -- and this step exists to make TOTAL coverage a construction
# guarantee. Measured cost of the whole warm run is ~10s against a ~155s Gate 1,
# so the overlap is worth the loss of a drift class.
if SKIP="$GATE1_SKIPPED_HOOKS" "$PRE_COMMIT" "${PRE_COMMIT_ARGS[@]}"; then
    echo "✓ Pre-commit hook set passed"
else
    echo "✗ Pre-commit hook set failed" >&2
    echo "  note: a fixer hook may have rewritten files -- review the diff" >&2
    echo "        above, then re-run to confirm the tree is clean." >&2
    exit 1
fi

exit 0
