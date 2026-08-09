#!/usr/bin/env bash
# scripts/ralph/architect-model.sh
#
# Pin the ralph-chief-architect subagent to its PRIMARY model (`fable`) or its
# FALLBACK model (`opus`), by rewriting the `model:` line in the YAML frontmatter
# of `.claude/agents/ralph-chief-architect.md`.
#
# Why this exists: Claude Code resolves a subagent's model from (in order) the
# CLAUDE_CODE_SUBAGENT_MODEL env var, the per-invocation `model` parameter, then
# the definition's `model:` frontmatter. There is no per-agent fallback CHAIN —
# so when Fable credits run out, something has to change one of those three
# inputs. The in-session fix is the per-invocation override (the conductor just
# re-dispatches with `model: "opus"`); this script is the durable one, flipping
# the frontmatter so every later tick skips the failing Fable attempt entirely.
#
# The allowlist is deliberately exactly {fable, opus} — the two models the
# role-based policy sanctions for the architect seat (see
# .claude/agents/shared/README.md). It is a capacity switch, not a general-purpose
# frontmatter editor, so a typo can never silently pin the architect to a model
# nobody chose.
#
# Usage:
#   ./scripts/ralph/architect-model.sh          # print the current pin
#   ./scripts/ralph/architect-model.sh fable    # primary  (Fable 5)
#   ./scripts/ralph/architect-model.sh opus     # fallback (Opus 5)
#
# Exit codes:
#   0  pin printed, or pin applied (setting the already-pinned model is a no-op)
#   1  the agent file is missing, or has no `model:` line in its frontmatter
#   2  usage error (unknown model, too many arguments)
#
# Override the target file with RALPH_ARCHITECT_AGENT_FILE (used by the tests).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AGENT_FILE="${RALPH_ARCHITECT_AGENT_FILE:-$REPO_ROOT/.claude/agents/ralph-chief-architect.md}"

PRIMARY_MODEL="fable"
FALLBACK_MODEL="opus"

usage() {
  cat >&2 <<EOF
Usage: $(basename "$0") [$PRIMARY_MODEL|$FALLBACK_MODEL]

  (no argument)  print the architect's current model pin
  $PRIMARY_MODEL          pin to Fable 5 — the primary planning model
  $FALLBACK_MODEL           fall back to Opus 5 — use when Fable credits run out
EOF
}

if [[ $# -gt 1 ]]; then
  echo "error: expected at most one argument, got $#" >&2
  usage
  exit 2
fi

case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
esac

if [[ ! -f "$AGENT_FILE" ]]; then
  echo "error: agent file not found: $AGENT_FILE" >&2
  exit 1
fi

# The CURRENT pin: the first `model:` line inside the leading `---` frontmatter
# block. Scoped to the frontmatter on purpose — the agent's prose body discusses
# models, and a body line must never be mistaken for the pin (or rewritten).
current="$(
  awk '
    NR == 1 && $0 == "---" { in_fm = 1; next }
    in_fm && $0 == "---"   { exit }
    in_fm && /^model:[[:space:]]*[^[:space:]]/ {
      sub(/^model:[[:space:]]*/, "")
      sub(/[[:space:]]+$/, "")
      print
      exit
    }
  ' "$AGENT_FILE"
)"

if [[ -z "$current" ]]; then
  echo "error: no 'model:' line in the frontmatter of $AGENT_FILE" >&2
  exit 1
fi

# No argument: report and stop.
if [[ $# -eq 0 ]]; then
  echo "$current"
  exit 0
fi

target="$1"
if [[ "$target" != "$PRIMARY_MODEL" && "$target" != "$FALLBACK_MODEL" ]]; then
  echo "error: unknown model '$target' (expected $PRIMARY_MODEL or $FALLBACK_MODEL)" >&2
  usage
  exit 2
fi

if [[ "$target" == "$current" ]]; then
  echo "ralph-chief-architect already pinned to $target"
  exit 0
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

awk -v target="$target" '
  NR == 1 && $0 == "---" { in_fm = 1; print; next }
  in_fm && $0 == "---"   { in_fm = 0; print; next }
  in_fm && !done && /^model:[[:space:]]*[^[:space:]]/ {
    print "model: " target
    done = 1
    next
  }
  { print }
' "$AGENT_FILE" >"$tmp"

# Preserve the destination's mode/ownership: write through the existing file
# rather than mv'ing the mktemp (which carries 0600) over it.
cat "$tmp" >"$AGENT_FILE"

echo "ralph-chief-architect: $current -> $target"
