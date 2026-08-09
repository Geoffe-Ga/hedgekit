#!/usr/bin/env bash
# scripts/record-research-cassettes.sh - Operator-run live research-cassette
# recorder (issue #192)
#
# WHAT THIS DOES
#   Records one live web-research search plus each of its fetches into a replay
#   cassette that CI thereafter replays fully offline. This is the ONLY entry
#   point that makes a live search/fetch call; CI never does. It is a thin
#   wrapper around scripts/record_research_cassettes.py, which owns all
#   `requests`/env access.
#
# API-KEY SETUP (required, never echoed by this script)
#   Export the search API key before running -- this script fails loudly if it
#   is unset and never prints its value:
#       export RESEARCH_SEARCH_API_KEY=...   # from your search provider console
#   The key is injected as a send-time HTTP header by the Python recorder and is
#   NEVER written onto the header-free HttpRequest, so it can never land in a
#   cassette file. Prefer a per-shell `export` (or a secrets manager) over any
#   file that risks being committed.
#
# CASSETTE INSPECTION BEFORE COMMIT (a Definition-of-Done line item)
#   The recorder wraps the HTTP-level RecordingHttpCassette, so a recorded entry
#   holds only the request method/url/body and the response status/body/
#   content-type -- never an HTTP header, so keys are structurally absent by
#   construction (HttpRequest has no headers field), not merely scrubbed. Even
#   so, inspect before you `git add`:
#     1. Inspect it: `jq . <cassette.json>` -- confirm it holds only request/
#        response pairs and NO key-like material.
#     2. Scrub any incidental PII a fetched page may echo before committing.
#     3. Re-run the offline replay suite to confirm the cassette drives CI green:
#        `.venv/bin/python -m pytest tests/forecast -q`.
#
# USAGE
#   scripts/record-research-cassettes.sh \
#       --endpoint-url https://search.example/v1/search \
#       --query "Fed rate decision December 2024" \
#       --allowed-host research.example \
#       --allowed-host news.example \
#       --out tests/fixtures/forecast/research_cassette.json
#
#   All arguments are forwarded verbatim to record_research_cassettes.py; run it
#   with --help for the authoritative argument list.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Fail loudly (and without ever echoing the value) if the key is unset.
if [[ -z "${RESEARCH_SEARCH_API_KEY:-}" ]]; then
    echo "error: required environment variable unset: RESEARCH_SEARCH_API_KEY" >&2
    echo "       export it before recording (see this script's header)." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
exec python3 scripts/record_research_cassettes.py "$@"
