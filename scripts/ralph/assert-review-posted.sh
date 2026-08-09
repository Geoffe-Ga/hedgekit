#!/usr/bin/env bash
# scripts/ralph/assert-review-posted.sh
#
# The LOUD post-gate counterpart to pr-ready.sh's silent freshness guard. The
# orchestrator runs this AFTER dispatching claude-code-action's review step, to
# prove a verdict comment actually landed on the PR during THIS run.
#
# WHY THIS EXISTS:
#   #135 (silent-success stall): the review action can finish "successfully"
#     without ever posting a verdict — the workflow step goes green, pr-ready.sh
#     then reports `awaiting-review` forever (no fresh verdict), and the lane
#     stalls with nothing loud to point at. This script turns that silent hole
#     into a failing step with an explicit "rerun" message.
#   #140 (instant-error incident): the review action can fail INSTANTLY (e.g. an
#     expired OAuth token / credit-balance error) while the step STILL reports
#     success — starving the PR of a verdict with zero diagnostics. When handed
#     the action's --execution-file we read its own is_error flag and fail fast
#     with the agent's error text, independent of whatever is on the PR.
#
# Two invariants, in order:
#   STEP A (defensive, best-effort): if the execution file says is_error → fail
#     LOUD immediately with the agent's error text. Never crashes on a
#     missing/unreadable/malformed file or absent jq — it just falls through.
#   STEP B (authoritative): assert a verdict-bearing comment (same VERDICT_RE
#     pr-ready.sh selects on) exists with createdAt >= STARTED_AT, so a stale
#     verdict from a PREVIOUS run cannot paper over a broken current one. When
#     --head-sha is supplied, that comment must ALSO carry the #400 subject
#     binding for this exact PR and head commit (see below).
#
#   #400 (cross-posted review): a well-formed, correctly-parsed, perfectly fresh
#     verdict can still be a review of a DIFFERENT pull request. It happened: the
#     review agent produced PR #398's review and the poster landed it, correctly
#     addressed, on PR #396, where it became the latest verdict. Freshness cannot
#     see this — the wrong review is posted at the right time. Nor can the
#     timestamp check catch a superseded concurrent run, whose verdict describes
#     an obsolete commit yet is genuinely newer than STARTED_AT.
#     So code-review.yml now writes the subject INTO the comment:
#         Review subject: PR #396 @ 635ac9d1ab7378f705303fcf802801d8c3ae5824
#     and --head-sha makes this script require that exact line. Identity, not
#     timing, is what proves a review belongs to its PR.
#
# Usage:  assert-review-posted.sh <PR_NUMBER> <STARTED_AT> \
#           [--repo <owner/repo>] [--execution-file <path>] [--head-sha <sha>]
set -euo pipefail

# Shared verdict regex — the single source of truth also sourced by pr-ready.sh
# (the merge gate). Resolve relative to THIS script so the check is
# cwd-independent. Provides VERDICT_RE (the one we need here).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ralph/verdict-regex.sh
# shellcheck disable=SC1091  # sourced at runtime; not followed without -x
source "$SCRIPT_DIR/verdict-regex.sh"

die() { echo "assert-review-posted: $1" >&2; exit 2; }

pr=""
started_at=""
execution_file=""
head_sha=""
repo_args=()
positional_seen=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)           [[ $# -ge 2 ]] || die "--repo needs a value"; repo_args+=(--repo "$2"); shift 2 ;;
    --execution-file) [[ $# -ge 2 ]] || die "--execution-file needs a value"; execution_file="$2"; shift 2 ;;
    --head-sha)       [[ $# -ge 2 ]] || die "--head-sha needs a value"; head_sha="$2"; shift 2 ;;
    -*)               die "unknown option: $1" ;;
    *)
      if [[ "$positional_seen" -eq 0 ]]; then
        pr="$1"; positional_seen=1
      elif [[ "$positional_seen" -eq 1 ]]; then
        started_at="$1"; positional_seen=2
      else
        die "unexpected extra argument: $1"
      fi
      shift ;;
  esac
done
[[ "$pr" =~ ^[0-9]+$ ]] || die "usage: assert-review-posted.sh <PR_NUMBER> <STARTED_AT> [--repo <owner/repo>] [--execution-file <path>]"
[[ -n "$started_at" ]]  || die "usage: assert-review-posted.sh <PR_NUMBER> <STARTED_AT> [--repo <owner/repo>] [--execution-file <path>]"
# A supplied head SHA must be abbreviated-or-full hex. This is a correctness
# check first (a malformed SHA could never match a rendered subject line, so it
# would fail every run for the wrong reason) and an injection check second: `pr`
# is already digits-only and this keeps `head_sha` free of any quote or
# backslash before it is spliced into the jq string literal below.
[[ -z "$head_sha" || "$head_sha" =~ ^[0-9a-fA-F]{7,40}$ ]] \
  || die "--head-sha must be a 7-40 character hex commit SHA, got: $head_sha"

# `${arr[@]+"${arr[@]}"}` expands to nothing when the array is empty instead of
# tripping `set -u` on bash 3.2 (stock /bin/bash on macOS).
gh_args=("$pr" ${repo_args[@]+"${repo_args[@]}"})

# --- STEP A: execution-file is_error fast-fail (#140 diagnosability) ----------
# Best-effort ONLY. An empty --execution-file value means "not provided". If the
# file is missing/unreadable/malformed or jq is absent we say nothing and let
# Step B stay authoritative — this path can only ADD a loud failure, never
# suppress the comment check. Every jq call is guarded so a garbage file can
# never crash the script.
if [[ -n "$execution_file" && -r "$execution_file" ]] && command -v jq >/dev/null 2>&1; then
  is_err="$(jq -r '[.[]? | select(.type == "result")] | last | .is_error // empty' "$execution_file" 2>/dev/null || true)"
  if [[ "$is_err" == "true" ]]; then
    detail="$(jq -r '[.[]? | select(.type == "result")] | last | (.result // .subtype // "unknown error")' "$execution_file" 2>/dev/null || true)"
    [[ -n "$detail" ]] || detail="unknown error"
    echo "assert-review-posted: review agent errored: $detail" >&2
    exit 1
  fi
fi

# --- STEP B: authoritative fresh-verdict-comment assertion --------------------
# Count PR comments whose body matches the shared verdict regex AND whose
# createdAt is at-or-after STARTED_AT. RFC3339 UTC timestamps are fixed-width,
# so the `>=` compare is done as a LEXICAL string compare INSIDE jq (portable,
# no date arithmetic). The compare is `>=` (inclusive — a comment posted in the
# same second as the run start counts), deliberately LOOSER than pr-ready.sh's
# strict `>` against HEAD: different invariant ("posted during THIS run" vs
# "postdates HEAD"), same lexical-RFC3339 technique.
#
# STARTED_AT is spliced into the jq program as a quoted JSON string literal, the
# same way VERDICT_RE is interpolated in pr-ready.sh's verdict query: `gh --jq`
# runs its expression server-side and (unlike standalone jq) exposes NO `--arg`
# flag, so
# interpolation is the only channel. Safe here — STARTED_AT is a fixed-width
# RFC3339 UTC timestamp (`date -u +%Y-%m-%dT%H:%M:%SZ`) with no quote characters
# to break out of the literal.
fresh_sel="[.comments[] | select(.body != null and (.body | test(\"$VERDICT_RE\")) and (.createdAt != null) and (.createdAt >= \"$started_at\"))]"

# --- #400 subject binding -----------------------------------------------------
# With a head SHA supplied, "fresh" is no longer sufficient: the comment must
# also SAY which PR and which commit it reviewed, and both must be ours. The
# expected line is assembled as a LITERAL and matched with jq `contains`
# (substring) rather than a regex — `pr` is validated as digits and `head_sha` as
# hex, so neither can carry a quote or backslash out of the jq string literal,
# and a literal cannot be silently widened the way a regex can. Interpolation is
# again the only channel: `gh --jq` evaluates server-side and offers no --arg.
#
# Absent --head-sha this is a no-op and the pre-#400 behaviour is preserved
# exactly, which is why code-review.yml is statically pinned to keep passing it.
subject_line=""
subject_pred=""
if [[ -n "$head_sha" ]]; then
  subject_line="Review subject: PR #${pr} @ ${head_sha}"
  subject_pred=" | map(select(.body | contains(\"$subject_line\")))"
fi

count="$(gh pr view "${gh_args[@]}" \
  --json comments \
  --jq "${fresh_sel}${subject_pred} | length")"

if [[ "$count" =~ ^[0-9]+$ ]] && [[ "$count" -ge 1 ]]; then
  exit 0
fi

# --- STEP B failure path: #400 subject-binding mismatch -----------------------
# Separate "no verdict at all" from "a verdict landed, but about something else".
# The second IS the #400 incident, and the generic rerun message below would be
# actively misleading for it: the run did post — it posted a review of a
# different PR, or a superseded concurrent run's review of an older commit. Name
# the binding we required so the operator can see the mismatch immediately.
if [[ -n "$head_sha" ]]; then
  unbound="$(gh pr view "${gh_args[@]}" --json comments --jq "${fresh_sel} | length" 2>/dev/null || true)"
  if [[ "$unbound" =~ ^[0-9]+$ ]] && [[ "$unbound" -ge 1 ]]; then
    echo "assert-review-posted: a fresh verdict comment exists, but none carries the required subject binding: \"${subject_line}\"" >&2
    echo "assert-review-posted: that verdict reviewed a DIFFERENT pull request or an OLDER head commit (issue #400) — it does not count as a review of this PR; rerun the Code Review workflow" >&2
    exit 1
  fi
fi

# --- STEP B failure path: workflow-validation-guard detection -----------------
# claude-code-action@v1 SKIPS the review agent entirely (step exits success, no
# verdict comment, empty execution_file) on any PR whose diff modifies the
# workflow file that invokes it — its own "workflow-validation guard". On that
# path the generic "rerun the Code Review workflow" message below is actively
# misleading: no rerun can ever produce a verdict until the PR merges. Best-effort
# detect the case here (never crashing STEP A's precedence, which already
# exited above) and emit a guard-specific message instead.
REVIEW_WORKFLOW_FILE=".github/workflows/code-review.yml"
# `|| true` inside the substitution keeps a failing/absent gh from tripping
# `set -e`; changed_files just ends up empty and we fall through to the generic
# message. Reuses gh_args (<PR> + optional --repo).
changed_files="$(gh pr diff "${gh_args[@]}" --name-only 2>/dev/null || true)"
# Exact-line containment in pure bash (no pipeline: printf | grep can yield rc
# 141 under pipefail and misclassify). Bracketing both sides with newlines makes
# it an exact whole-line match, so docs/code-review.yml and
# .github/workflows/code-review.yml.bak cannot false-positive.
if [[ -n "$changed_files" && $'\n'"$changed_files"$'\n' == *$'\n'"$REVIEW_WORKFLOW_FILE"$'\n'* ]]; then
  echo "assert-review-posted: the review action was SKIPPED BY DESIGN by claude-code-action's workflow validation guard because this PR modifies ${REVIEW_WORKFLOW_FILE}." >&2
  echo "assert-review-posted: no rerun will ever produce a review until the PR is merged; this PR requires human review and admin merge." >&2
  exit 1
fi

echo "assert-review-posted: no verdict-bearing comment created at/after ${started_at} — the review agent posted no verdict; rerun the Code Review workflow" >&2
exit 1
