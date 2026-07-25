#!/usr/bin/env bash
# scripts/ralph/test_architect_model.sh
#
# Offline tests for architect-model.sh — the Fable/Opus capacity switch for the
# ralph-chief-architect subagent's `model:` frontmatter pin.
#
# Every scenario runs against a FIXTURE agent file in a scratch directory (via
# RALPH_ARCHITECT_AGENT_FILE), never the real `.claude/agents/ralph-chief-architect.md`,
# so the suite is safe to run on a dirty tree and can assert on byte-exact
# rewrites. One static check at the end does read the real file — to prove the
# script's allowlist and the checked-in pin agree.
#
# Run:  bash scripts/ralph/test_architect_model.sh
set -euo pipefail

RALPH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$RALPH_DIR/../.." && pwd)"
AM="$RALPH_DIR/architect-model.sh"
REAL_AGENT="$REPO_ROOT/.claude/agents/ralph-chief-architect.md"
PASS=0
FAIL=0

check() { # check <desc> <expected> <actual>
  if [[ "$2" == "$3" ]]; then
    PASS=$((PASS + 1))
    printf '  ok  - %s\n' "$1"
  else
    FAIL=$((FAIL + 1))
    printf 'FAIL  - %s (expected [%s], got [%s])\n' "$1" "$2" "$3"
  fi
}

check_contains() { # check_contains <desc> <needle> <haystack>
  if [[ "$3" == *"$2"* ]]; then
    PASS=$((PASS + 1))
    printf '  ok  - %s\n' "$1"
  else
    FAIL=$((FAIL + 1))
    printf 'FAIL  - %s (expected to contain [%s], got [%s])\n' "$1" "$2" "$3"
  fi
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# A fixture agent file whose BODY also mentions `model:` at the start of a line —
# the regression guard for "only the frontmatter pin is read and rewritten".
fixture() { # fixture <model>
  cat >"$WORK/agent.md" <<EOF
---
name: ralph-chief-architect
description: "Strategic brain of a Ralph tick."
tools: Read,Grep,Glob,Task
model: $1
receives_from: []
---
# Chief Architect

Body prose that happens to start a line with the frontmatter key:
model: this-is-prose-not-the-pin
EOF
}

run() { # run <args...>  -> stdout+stderr; sets RC
  set +e
  OUT="$(RALPH_ARCHITECT_AGENT_FILE="$WORK/agent.md" bash "$AM" "$@" 2>&1)"
  RC=$?
  set -e
}

echo "== architect-model.sh =="

# --- read mode ----------------------------------------------------------------
fixture fable
run
check "no argument prints the current pin" "fable" "$OUT"
check "no argument exits 0" "0" "$RC"

fixture opus
run
check "read mode reflects a fallback pin" "opus" "$OUT"

# --- write mode ---------------------------------------------------------------
fixture fable
run opus
check "fable -> opus exits 0" "0" "$RC"
check_contains "fable -> opus reports the transition" "fable -> opus" "$OUT"
run
check "fable -> opus persists" "opus" "$OUT"

run fable
check_contains "opus -> fable reports the transition" "opus -> fable" "$OUT"
run
check "opus -> fable restores the primary pin" "fable" "$OUT"

# Idempotence: re-pinning the current model is a successful no-op.
run fable
check "re-pinning the current model exits 0" "0" "$RC"
check_contains "re-pinning the current model says so" "already pinned" "$OUT"

# --- the body's `model:` line is neither read nor rewritten -------------------
fixture opus
RALPH_ARCHITECT_AGENT_FILE="$WORK/agent.md" bash "$AM" fable >/dev/null
body_line="$(grep -c '^model: this-is-prose-not-the-pin$' "$WORK/agent.md")"
check "body 'model:' line survives a rewrite" "1" "$body_line"
pins="$(grep -c '^model: fable$' "$WORK/agent.md")"
check "exactly one frontmatter pin is written" "1" "$pins"

# Nothing but the pin line changes.
fixture fable
cp "$WORK/agent.md" "$WORK/before.md"
RALPH_ARCHITECT_AGENT_FILE="$WORK/agent.md" bash "$AM" opus >/dev/null
diff_lines="$(diff "$WORK/before.md" "$WORK/agent.md" | grep -c '^[<>]' || true)"
check "a flip rewrites exactly one line" "2" "$diff_lines"

# --- error paths --------------------------------------------------------------
fixture fable
run sonnet
check "unknown model exits 2" "2" "$RC"
check_contains "unknown model explains the allowlist" "expected fable or opus" "$OUT"
run
check "unknown model leaves the pin untouched" "fable" "$OUT"

run fable opus
check "too many arguments exits 2" "2" "$RC"

set +e
OUT="$(RALPH_ARCHITECT_AGENT_FILE="$WORK/nope.md" bash "$AM" opus 2>&1)"
RC=$?
set -e
check "missing agent file exits 1" "1" "$RC"
check_contains "missing agent file is named" "nope.md" "$OUT"

# Frontmatter without a `model:` line: refuse rather than silently append one.
cat >"$WORK/agent.md" <<'EOF'
---
name: ralph-chief-architect
tools: Read
---
# Chief Architect
EOF
run opus
check "frontmatter with no pin exits 1" "1" "$RC"
check_contains "frontmatter with no pin says why" "no 'model:' line" "$OUT"

# --- static guard over the real agent file ------------------------------------
real_pin="$(RALPH_ARCHITECT_AGENT_FILE="$REAL_AGENT" bash "$AM")"
if [[ "$real_pin" == "fable" || "$real_pin" == "opus" ]]; then
  PASS=$((PASS + 1))
  printf '  ok  - checked-in architect pin is on the allowlist (%s)\n' "$real_pin"
else
  FAIL=$((FAIL + 1))
  printf 'FAIL  - checked-in architect pin [%s] is not fable or opus\n' "$real_pin"
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]]
