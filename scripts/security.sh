#!/usr/bin/env bash
# scripts/security.sh - Run security checks with Bandit and Safety
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

Run security checks using Bandit and Safety.

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

# Set verbosity
if $VERBOSE; then
    set -x
fi

echo "=== Security Checks (Bandit) ==="

# Run Bandit
if $VERBOSE; then
    echo "Running Bandit security scanner..."
fi
bandit -c pyproject.toml -r windbreak/ || { echo "✗ Bandit found issues" >&2; exit 1; }

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

pip-audit "${PIP_AUDIT_ARGS[@]}" || { echo "✗ pip-audit found issues" >&2; exit 1; }

echo "=== Security Checks (detect-secrets baseline) ==="

# Enforce the same baseline-diffing detect-secrets hook that CI's
# "Pre-commit (all files)" step runs, so local Gate 1 == CI (issue #262).
# Fail loud if pre-commit is unavailable rather than silently skipping the
# check -- a silent skip is the exact enforcement gap this section closes.
if ! command -v pre-commit &> /dev/null; then
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
pre-commit run detect-secrets --all-files || {
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
