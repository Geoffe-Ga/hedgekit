#!/usr/bin/env bash
# scripts/run-canaries.sh - Operator-run provider canary battery driver (issue #195)
#
# WHAT THIS DOES
#   Runs the provider canary battery (fleet observability, SPEC S8.4/S16
#   extended per-provider), appending one CanaryVerdictRecorded per provider to
#   the ledger and exiting NON-ZERO the moment any provider drifts -- so an
#   operator (or a cron/CI hook) is alerted to silent forecaster drift. This is
#   a thin wrapper around scripts/run_canaries.py, which owns all `requests`/env
#   access; the always-offline replay mode is the default and CI-safe path.
#
# USAGE
#   # Offline replay (default) -- observations are read from the spec file:
#   scripts/run-canaries.sh \
#       --spec-file tests/fixtures/forecast/provider_canaries.json \
#       --ledger-path var/ledger.db
#
#   # Live record mode -- dials each provider endpoint once (keys required):
#   scripts/run-canaries.sh --record \
#       --spec-file provider_canaries.record.json \
#       --ledger-path var/ledger.db
#
#   All arguments are forwarded verbatim to run_canaries.py; run it with --help
#   for the authoritative argument list.
#
# API-KEY SETUP (record mode only, never echoed by this script)
#   In --record mode each provider's live API key is read from its
#   <PROVIDER>_API_KEY environment variable (e.g. FUTURESEARCH_API_KEY),
#   injected as a send-time HTTP header by the Python driver and NEVER printed.
#   Export the keys before recording; the Python driver fails loudly (naming the
#   missing variable, never its value) if any required key is unset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Detect record mode without consuming the argument (it is forwarded verbatim).
record_mode=false
for arg in "$@"; do
    if [[ "$arg" == "--record" ]]; then
        record_mode=true
        break
    fi
done

# Record mode reaches live provider endpoints: remind the operator that keys
# must be exported (the Python driver validates each per-provider key and fails
# loudly by name). Never echo any key value.
if [[ "$record_mode" == true ]]; then
    echo "record mode: each provider's <PROVIDER>_API_KEY must be exported" >&2
    echo "             (values are never printed; missing keys fail loudly)." >&2
fi

cd "$PROJECT_ROOT"
exec python3 scripts/run_canaries.py "$@"
