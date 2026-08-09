#!/usr/bin/env bash
# scripts/record-cassettes.sh - Operator-run live vote-cassette recorder (issue #191)
#
# WHAT THIS DOES
#   Records one live vote per pinned ensemble member (OpenAI + Anthropic) into a
#   replay cassette that CI thereafter replays fully offline. This is the ONLY
#   entry point that makes a live LLM call; CI never does. It is a thin wrapper
#   around scripts/record_vote_cassettes.py, which owns all `requests`/env access.
#
# API-KEY SETUP (required, never echoed by this script)
#   Export both provider keys before running -- this script fails loudly if
#   either is unset and never prints their values:
#       export ANTHROPIC_API_KEY=sk-ant-...      # from console.anthropic.com
#       export OPENAI_API_KEY=sk-...             # from platform.openai.com
#   Keys are injected as send-time HTTP headers by the Python recorder and are
#   NEVER written onto the header-free HttpRequest, so they can never land in a
#   cassette file. Prefer a per-shell `export` (or a secrets manager) over any
#   file that risks being committed.
#
# CASSETTE SCRUBBING / INSPECTION BEFORE COMMIT (a Definition-of-Done line item)
#   The recorder wraps the LLM-level RecordingCassette, so a recorded entry holds
#   only the prompt request fields and the *extracted completion text*
#   (Anthropic content[0].text / OpenAI choices[0].message.content) -- never the
#   full HTTP envelope, and never an HTTP header, so keys are structurally absent
#   by construction (HttpRequest has no headers field), not merely scrubbed. A
#   non-2xx / error response is rejected by the adapter before anything is
#   recorded, so error envelopes never land on disk. Even so, inspect before you
#   `git add`:
#     1. Inspect it: `jq . <cassette.json>` -- confirm it holds only request/
#        response pairs and NO key-like material (there should be none, but
#        verify defense-in-depth).
#     2. Scrub any incidental PII or account identifiers a completion may echo
#        back inside its own text before committing.
#     3. Re-run the offline replay suite to confirm the cassette drives CI green:
#        `.venv/bin/python -m pytest tests/forecast -q`.
#
# USAGE
#   scripts/record-cassettes.sh \
#       --ticker KXFED-25MAR \
#       --title "Fed cuts rates in March 2025?" \
#       --resolution-criteria "Resolves YES if the FOMC cuts the federal funds rate." \
#       --close-time 2025-03-19T19:00:00+00:00 \
#       --baseline-price-pips 4500 \
#       --out tests/fixtures/forecast/vote_cassette.json
#
#   All arguments are forwarded verbatim to record_vote_cassettes.py; run it with
#   --help for the authoritative argument list.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Fail loudly (and without ever echoing the value) if either key is unset.
missing=()
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    missing+=("ANTHROPIC_API_KEY")
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    missing+=("OPENAI_API_KEY")
fi
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "error: required environment variable(s) unset: ${missing[*]}" >&2
    echo "       export them before recording (see this script's header)." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
exec python3 scripts/record_vote_cassettes.py "$@"
