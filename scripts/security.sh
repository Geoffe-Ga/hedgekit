#!/usr/bin/env bash
# scripts/security.sh - Run security checks with Bandit, pip-audit and detect-secrets
# Usage: ./scripts/security.sh [--verbose] [--help]

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

Run security checks using Bandit, pip-audit and detect-secrets.

Every tool is resolved from the pinned toolchain (scripts/toolchain-env.sh),
so this script audits the same environment whether it is run directly or via
./scripts/check-all.sh, regardless of the caller's PATH.

OPTIONS:
    --verbose   Show detailed output
    --help      Display this help message

EXIT CODES:
    0           No security issues found
    1           Security issues found
    2           Error running checks

EXAMPLES:
    $(basename "$0")             # Run basic security checks
    $(basename "$0") --verbose   # Show detailed output
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

# Resolve every tool from the PINNED toolchain rather than the caller's PATH
# (issue #366). Doing it here -- not only in check-all.sh -- is what makes a
# direct `./scripts/security.sh` audit the same environment Gate 1 audits.
TOOLCHAIN_ENV="$SCRIPT_DIR/toolchain-env.sh"
BANDIT="$(bash "$TOOLCHAIN_ENV" --print-tool bandit)"
# The interpreter is the load-bearing artifact for the dependency audit below:
# pip-audit audits the site-packages of whichever interpreter imports it.
AUDIT_PYTHON="$(bash "$TOOLCHAIN_ENV" --print-python)"

# Set verbosity
if $VERBOSE; then
    set -x
fi

echo "=== Security Checks (Bandit) ==="

# Run Bandit
if $VERBOSE; then
    echo "Running Bandit security scanner..."
fi
"$BANDIT" -c pyproject.toml -r windbreak/ || { echo "✗ Bandit found issues" >&2; exit 1; }

echo "=== Security Checks (pip-audit) ==="

# Run pip-audit for dependency vulnerability scanning
if $VERBOSE; then
    echo "Running pip-audit dependency checker..."
fi

# Build ignore flags for known transitive dependency vulnerabilities
# that cannot be fixed (no fix available or deprecated transitive deps).
# Each entry should have a corresponding tracking issue.
PIP_AUDIT_ARGS=()
if [ -f "$PROJECT_ROOT/.pip-audit-known-vulnerabilities" ]; then
    while IFS= read -r line; do
        # Strip inline comments and trim whitespace
        vuln_id="${line%%#*}"
        vuln_id="${vuln_id%"${vuln_id##*[![:space:]]}"}"
        # Skip empty lines
        [[ -z "$vuln_id" ]] && continue
        PIP_AUDIT_ARGS+=(--ignore-vuln "$vuln_id")
    done < "$PROJECT_ROOT/.pip-audit-known-vulnerabilities"
fi

# Run pip-audit as a MODULE of the pinned interpreter, never as a PATH-resolved
# console script. A console script's shebang decides which interpreter -- and
# therefore which site-packages -- gets audited, which is how this gate ended up
# auditing Homebrew's Python and reporting a CVE against dependencies the repo
# does not own (issue #366). Worse, the same misresolution can report CLEAN
# while the project's real dependency set is vulnerable. Binding the audit to
# "$AUDIT_PYTHON" makes the audited environment a construction guarantee.
echo "Auditing dependencies of: $AUDIT_PYTHON"
"$AUDIT_PYTHON" -m pip_audit "${PIP_AUDIT_ARGS[@]}" || { echo "✗ pip-audit found issues" >&2; exit 1; }

echo "=== Security Checks (detect-secrets baseline) ==="

# Enforce the same baseline-diffing detect-secrets hook that CI's
# "Pre-commit (all files)" step runs, so local Gate 1 == CI (issue #262).
# Fail loud if pre-commit is unavailable rather than silently skipping the
# check -- a silent skip is the exact enforcement gap this section closes.
if ! PRE_COMMIT="$(bash "$TOOLCHAIN_ENV" --print-tool pre-commit)"; then
    echo "✗ pre-commit is not installed" >&2
    echo "  why: pre-commit runs the baseline-enforcing detect-secrets check" >&2
    echo "       that CI runs; without it local Gate 1 cannot match CI." >&2
    echo "  next: run scripts/provision-venv.sh (the shared pinned venv" >&2
    echo "        provides pre-commit) or" >&2
    echo "        pip install -c constraints-quality.txt pre-commit" >&2
    exit 2
fi

if $VERBOSE; then
    echo "Running detect-secrets via pre-commit..."
fi
"$PRE_COMMIT" run detect-secrets --all-files || {
    echo "✗ detect-secrets found a potential secret not in .secrets.baseline (or the hook failed)" >&2
    echo "  why: this is the same check CI's 'Pre-commit (all files)' step runs, so CI would fail too." >&2
    echo "  next: audit the flagged finding above. If it is a real secret, remove it." >&2
    echo "        If it is a genuine false positive, fix it structurally (e.g. rename the" >&2
    echo "        fixture) as PRs #260/#282 did. Do NOT regenerate or weaken .secrets.baseline" >&2
    echo "        to silence it -- that would launder a real secret into the allowlist." >&2
    exit 1
}

echo "✓ Security checks passed"
exit 0
