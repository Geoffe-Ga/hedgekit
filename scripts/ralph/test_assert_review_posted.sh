#!/usr/bin/env bash
# scripts/ralph/test_assert_review_posted.sh
#
# Offline tests for assert-review-posted.sh (issue #135, hardened per the #140
# incident) — the check the orchestrator runs AFTER dispatching
# claude-code-action's review step to confirm a verdict comment actually
# landed on the PR. #140 showed the review action can fail INSTANTLY
# (OAuth/credit-balance error) while the workflow step itself still reports
# "success" — silently starving the PR of a verdict forever. This script closes
# that hole two ways:
#
#   1. If handed the action's --execution-file, it inspects the run's own
#      is_error flag and fails LOUD with the agent's error text the moment a
#      hard failure is detected — independent of whatever is (or isn't) on the
#      PR already.
#   2. Regardless of that file's presence/validity, it queries the PR's
#      comments for a verdict-bearing comment (the same verdict-prefix regex
#      pr-ready.sh uses) posted at-or-after the review step's start time
#      (STARTED_AT) — so a stale verdict from a PREVIOUS run can't paper over
#      a broken current one.
#
# Usage:  assert-review-posted.sh <PR_NUMBER> <STARTED_AT> [--repo <owner/repo>] [--execution-file <path>]
#
# `assert-review-posted.sh` does not exist yet, so every functional case below
# fails RED (the direct absolute-path invocation exits 127, "no such file").
# The two static workflow guards at the bottom are RED too: code-review.yml has
# no verification step yet, and ralph-fleet-tests.yml doesn't trigger on it.
#
# Run:  bash scripts/ralph/test_assert_review_posted.sh
set -euo pipefail

RALPH_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$RALPH_DIR/../.." && pwd)"
SCRIPT="$RALPH_DIR/assert-review-posted.sh"
WORKFLOWS_DIR="$ROOT_DIR/.github/workflows"
CODE_REVIEW_YML="$WORKFLOWS_DIR/code-review.yml"
FLEET_TESTS_YML="$WORKFLOWS_DIR/ralph-fleet-tests.yml"

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  ok  - %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL  - %s\n' "$1"; }
check() { # check <desc> <expected> <actual>
  if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1 (expected '$2', got '$3')"; fi
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
BIN="$WORK/bin"
mkdir -p "$BIN"
STDERR_FILE="$WORK/stderr.txt"

# Arg-aware fake gh. Only `--json comments` is exercised by this script today.
# When COMMENTS_JSON is set, the stub extracts the caller's own `--jq`
# expression (if any) from "$@" and runs the REAL jq against it — the same
# passthrough trick test_pr_ready.sh uses — so the production verdict regex
# and freshness compare are genuinely exercised, never masked by a scalar
# stub. If the caller passed no `--jq`, the raw payload is emitted instead so
# a script that parses comments itself in bash still gets real data.
#
# `pr diff --name-only` branch (issue #135 follow-up, workflow-validation-guard
# detection): mirrors the same env-var-injection convention. DIFF_FILES set ->
# print it verbatim (tests inject multi-line lists via $'\n'); DIFF_RC set
# (nonzero) -> exit with that code, simulating `gh pr diff` failing/unavailable;
# neither set -> print nothing, so a script under test that hasn't been taught
# to call `gh pr diff` at all is unaffected and every existing case stays on
# the untouched default path.
cat > "$BIN/gh" <<'STUB'
#!/usr/bin/env bash
args="$*"
case "$args" in
  *"--json comments"*)
    if [[ -n "${COMMENTS_JSON:-}" ]]; then
      expr="" prev="" has_jq=no
      for a in "$@"; do
        if [[ "$prev" == "--jq" ]]; then expr="$a"; has_jq=yes; fi
        prev="$a"
      done
      if [[ "$has_jq" == yes ]]; then
        printf '%s' "$COMMENTS_JSON" | jq -rc "$expr"
      else
        printf '%s' "$COMMENTS_JSON"
      fi
    else
      printf '{"comments":[]}\n'
    fi
    ;;
  *"--name-only"*)
    if [[ -n "${DIFF_RC:-}" ]]; then
      exit "$DIFF_RC"
    fi
    if [[ -n "${DIFF_FILES:-}" ]]; then
      printf '%s\n' "$DIFF_FILES"
    fi
    ;;
  *) echo '' ;;
esac
STUB
chmod +x "$BIN/gh"

# run_capture <args...> — invokes the script under test with the fake gh on
# PATH, discards stdout, captures stderr to $STDERR_FILE, and prints the exit
# code. Uses the `rc=0; ... || rc=$?` idiom so `set -e` never trips on a
# nonzero (including a missing-script 127) exit.
run_capture() {
  local rc=0
  PATH="$BIN:$PATH" "$SCRIPT" "$@" >/dev/null 2>"$STDERR_FILE" || rc=$?
  printf '%s' "$rc"
}

# run_capture_sub <args...> — identical, but invoked from a subdirectory, to
# prove the script derives no cwd-dependent state.
run_capture_sub() {
  local rc=0
  (cd "$WORK/sub" && PATH="$BIN:$PATH" "$SCRIPT" "$@" >/dev/null 2>"$STDERR_FILE") || rc=$?
  printf '%s' "$rc"
}

PR=100
STARTED="2026-07-01T10:00:00Z"     # the review step's recorded start time
FRESH="2026-07-01T11:00:00Z"       # a verdict posted AFTER STARTED (valid)
SAME="$STARTED"                    # a verdict posted AT STARTED (boundary, inclusive)
STALE="2026-07-01T09:00:00Z"       # a verdict posted BEFORE STARTED (stale)

cj() { printf '{"comments":[%s]}' "$1"; }   # wrap comment object(s) as a payload

EMPTY_COMMENTS='{"comments":[]}'
FRESH_LGTM_COMMENTS="$(cj '{"createdAt":"'"$FRESH"'","body":"## Summary\ngood\n\n## Verdict: LGTM\n"}')"
SAME_LGTM_COMMENTS="$(cj '{"createdAt":"'"$SAME"'","body":"## Verdict: LGTM\n"}')"
STALE_LGTM_COMMENTS="$(cj '{"createdAt":"'"$STALE"'","body":"## Verdict: LGTM\n"}')"
CHATTER_COMMENTS="$(cj '{"createdAt":"'"$FRESH"'","body":"just a chat comment"}')"
CR_COMMENTS="$(cj '{"createdAt":"'"$FRESH"'","body":"**Verdict:** CHANGES_REQUESTED\n"}')"
HEADING_NEXTLINE_COMMENTS="$(cj '{"createdAt":"'"$FRESH"'","body":"## Summary\ngood\n\n## Verdict\n✅ LGTM\n"}')"
PROSE_ONLY_COMMENTS="$(cj '{"createdAt":"'"$FRESH"'","body":"This is not ready; no verdict has been reached yet, just chatting about the verdict process.\n"}')"

if command -v jq >/dev/null 2>&1; then

  # --- 1) fresh verdict + the >= boundary (same-second) ------------------------
  rc="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" run_capture "$PR" "$STARTED")"
  check "fresh verdict comment (createdAt > STARTED_AT) -> exit 0" "0" "$rc"

  rc="$(COMMENTS_JSON="$SAME_LGTM_COMMENTS" run_capture "$PR" "$STARTED")"
  check "same-second verdict (createdAt == STARTED_AT) -> exit 0 (inclusive >=)" "0" "$rc"

  # --- 2) no comments at all ----------------------------------------------------
  rc="$(COMMENTS_JSON="$EMPTY_COMMENTS" run_capture "$PR" "$STARTED")"
  check "no comments at all -> exit 1" "1" "$rc"
  check "no-comments stderr mentions STARTED_AT" "yes" \
    "$(grep -qF "$STARTED" "$STDERR_FILE" && echo yes || echo no)"
  check "no-comments stderr mentions rerun" "yes" \
    "$(grep -qi 'rerun' "$STDERR_FILE" && echo yes || echo no)"

  # --- 3) verdict present but STALE (before STARTED_AT) -------------------------
  rc="$(COMMENTS_JSON="$STALE_LGTM_COMMENTS" run_capture "$PR" "$STARTED")"
  check "stale verdict comment (createdAt < STARTED_AT) -> exit 1" "1" "$rc"

  # --- 4) only non-verdict chatter ----------------------------------------------
  rc="$(COMMENTS_JSON="$CHATTER_COMMENTS" run_capture "$PR" "$STARTED")"
  check "only non-verdict chatter comment -> exit 1" "1" "$rc"

  # --- 5) real-jq regex cases mirroring test_pr_ready.sh's body shapes ----------
  # This script only cares that A verdict was posted — LGTM-ness is pr-ready.sh's
  # concern, not this one's — so CHANGES_REQUESTED must count exactly like LGTM.
  rc="$(COMMENTS_JSON="$CR_COMMENTS" run_capture "$PR" "$STARTED")"
  check "real **Verdict:** CHANGES_REQUESTED (fresh) -> exit 0 (any verdict counts)" "0" "$rc"

  rc="$(COMMENTS_JSON="$HEADING_NEXTLINE_COMMENTS" run_capture "$PR" "$STARTED")"
  check "real ## Verdict\\n(emoji) LGTM heading-with-token-on-next-line (fresh) -> exit 0" "0" "$rc"

  rc="$(COMMENTS_JSON="$PROSE_ONLY_COMMENTS" run_capture "$PR" "$STARTED")"
  check "real prose-only 'verdict' mention (not a verdict line), fresh -> exit 1" "1" "$rc"

  # --- 6) execution-file: is_error true fails FAST, independent of comments ----
  ERROR_EXEC="$WORK/exec-error.json"
  cat > "$ERROR_EXEC" <<'JSON'
[{"type":"result","subtype":"error_during_execution","is_error":true,"result":"Credit balance is too low to access the Anthropic API. Please add credits or check your OAuth token."}]
JSON

  rc="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" run_capture "$PR" "$STARTED" --execution-file "$ERROR_EXEC")"
  check "execution-file is_error:true -> exit 1 even with a fresh verdict comment present (fast/independent path)" \
    "1" "$rc"
  check "execution-file is_error:true stderr mentions 'review agent errored'" "yes" \
    "$(grep -qiF 'review agent errored' "$STDERR_FILE" && echo yes || echo no)"
  check "execution-file is_error:true stderr contains the agent's error text" "yes" \
    "$(grep -qiF 'Credit balance' "$STDERR_FILE" && echo yes || echo no)"

  # is_error:true with NO result/subtype text still yields a usable message
  # (the `.result // .subtype // "unknown error"` fallback), so a terse error
  # payload never produces an empty, useless failure line.
  NOTEXT_EXEC="$WORK/exec-error-notext.json"
  cat > "$NOTEXT_EXEC" <<'JSON'
[{"type":"result","is_error":true}]
JSON

  rc="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" run_capture "$PR" "$STARTED" --execution-file "$NOTEXT_EXEC")"
  check "execution-file is_error:true with no result text -> exit 1 (fallback message)" "1" "$rc"
  check "execution-file is_error:true no-text stderr still names 'review agent errored'" "yes" \
    "$(grep -qiF 'review agent errored' "$STDERR_FILE" && echo yes || echo no)"
  check "execution-file is_error:true no-text stderr uses the 'unknown error' fallback" "yes" \
    "$(grep -qiF 'unknown error' "$STDERR_FILE" && echo yes || echo no)"

  # --- 7) execution-file: is_error false does not itself fail; comments still gate
  SUCCESS_EXEC="$WORK/exec-success.json"
  cat > "$SUCCESS_EXEC" <<'JSON'
[{"type":"system","subtype":"init"},{"type":"result","subtype":"success","is_error":false,"result":"ok"}]
JSON

  rc="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" run_capture "$PR" "$STARTED" --execution-file "$SUCCESS_EXEC")"
  check "execution-file is_error:false + fresh verdict comment -> exit 0" "0" "$rc"

  rc="$(COMMENTS_JSON="$EMPTY_COMMENTS" run_capture "$PR" "$STARTED" --execution-file "$SUCCESS_EXEC")"
  check "execution-file is_error:false + no comment -> exit 1 (comment check still authoritative)" "1" "$rc"

  # --- 8) execution-file absent/empty/malformed must fall through, never crash -
  rc="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" run_capture "$PR" "$STARTED" --execution-file "")"
  check "--execution-file '' (empty value) falls through to comment check -> exit 0, no crash" "0" "$rc"

  rc="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" run_capture "$PR" "$STARTED" \
        --execution-file "$WORK/does-not-exist.json")"
  check "--execution-file missing path falls through to comment check -> exit 0, no crash" "0" "$rc"

  MALFORMED_EXEC="$WORK/exec-malformed.json"
  printf 'not valid json {' > "$MALFORMED_EXEC"
  rc="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" run_capture "$PR" "$STARTED" --execution-file "$MALFORMED_EXEC")"
  check "--execution-file malformed JSON falls through to comment check -> exit 0, no crash" "0" "$rc"

  # --- 5b) workflow-validation guard detection ----------------------------------
  # claude-code-action@v1 SKIPS the review agent entirely (step exits success,
  # no verdict comment, empty execution_file) whenever the PR's diff touches the
  # workflow file that invokes it (.github/workflows/code-review.yml) — its own
  # "workflow-validation guard". Today STEP B's generic "no verdict-bearing
  # comment ... rerun the Code Review workflow" message is misleading here:
  # rerunning can never help until the PR merges. On the STEP B failure path
  # ONLY, the script should best-effort `gh pr diff <PR> --name-only`; if the
  # file list contains EXACTLY `.github/workflows/code-review.yml`, it must emit
  # a PRIMARY guard message instead of the generic one (still exit 1).
  WORKFLOW_TOUCHING_DIFF=$'scripts/ralph/assert-review-posted.sh\n.github/workflows/code-review.yml'

  # 1) Guard fires: no comments at all + diff touches the workflow file.
  rc="$(COMMENTS_JSON="$EMPTY_COMMENTS" DIFF_FILES="$WORKFLOW_TOUCHING_DIFF" \
        run_capture "$PR" "$STARTED")"
  check "workflow-validation guard: no comments + diff touches workflow -> exit 1" "1" "$rc"
  check "  ...stderr mentions 'workflow validation'" "yes" \
    "$(grep -qi 'workflow validation' "$STDERR_FILE" && echo yes || echo no)"
  check "  ...stderr mentions 'no rerun'" "yes" \
    "$(grep -qi 'no rerun' "$STDERR_FILE" && echo yes || echo no)"
  check "  ...stderr mentions 'human review'" "yes" \
    "$(grep -qi 'human review' "$STDERR_FILE" && echo yes || echo no)"
  check "  ...stderr mentions 'admin merge'" "yes" \
    "$(grep -qi 'admin merge' "$STDERR_FILE" && echo yes || echo no)"
  check "  ...stderr does NOT contain the generic 'rerun the Code Review workflow' text" "no" \
    "$(grep -qF 'rerun the Code Review workflow' "$STDERR_FILE" && echo yes || echo no)"

  # 2) Guard fires even with a STALE verdict present too — proves the trigger is
  #    "STEP B failed + diff touches the workflow file", not "zero comments".
  rc="$(COMMENTS_JSON="$STALE_LGTM_COMMENTS" DIFF_FILES="$WORKFLOW_TOUCHING_DIFF" \
        run_capture "$PR" "$STARTED")"
  check "workflow-validation guard: stale verdict + diff touches workflow -> exit 1" "1" "$rc"
  check "  ...stderr mentions 'workflow validation' (stale-verdict case)" "yes" \
    "$(grep -qi 'workflow validation' "$STDERR_FILE" && echo yes || echo no)"

  # 3) Generic preserved: STEP B fails but the diff does NOT touch the workflow.
  rc="$(COMMENTS_JSON="$EMPTY_COMMENTS" DIFF_FILES='scripts/ralph/pr-ready.sh' \
        run_capture "$PR" "$STARTED")"
  check "no guard: diff touches an unrelated file -> exit 1 (generic message)" "1" "$rc"
  check "  ...stderr keeps the generic 'rerun the Code Review workflow' text" "yes" \
    "$(grep -qF 'rerun the Code Review workflow' "$STDERR_FILE" && echo yes || echo no)"
  check "  ...stderr has no guard vocabulary" "no" \
    "$(grep -qi 'workflow validation' "$STDERR_FILE" && echo yes || echo no)"

  # 4) No substring false-positive: near-miss filenames must NOT trigger the guard.
  rc="$(COMMENTS_JSON="$EMPTY_COMMENTS" DIFF_FILES='docs/code-review.yml' \
        run_capture "$PR" "$STARTED")"
  check "no guard: 'docs/code-review.yml' is not the workflow path -> generic message" "no" \
    "$(grep -qi 'workflow validation' "$STDERR_FILE" && echo yes || echo no)"

  rc="$(COMMENTS_JSON="$EMPTY_COMMENTS" \
        DIFF_FILES='.github/workflows/code-review.yml.bak' \
        run_capture "$PR" "$STARTED")"
  check "no guard: '.github/workflows/code-review.yml.bak' is not an exact match -> generic message" "no" \
    "$(grep -qi 'workflow validation' "$STDERR_FILE" && echo yes || echo no)"

  # 5) Diff unset entirely + no comments -> generic message (implicit regression net).
  rc="$(COMMENTS_JSON="$EMPTY_COMMENTS" run_capture "$PR" "$STARTED")"
  check "no guard: DIFF_FILES unset -> exit 1 (generic message)" "1" "$rc"
  check "  ...stderr keeps the generic 'rerun the Code Review workflow' text" "yes" \
    "$(grep -qF 'rerun the Code Review workflow' "$STDERR_FILE" && echo yes || echo no)"
  check "  ...stderr has no guard vocabulary" "no" \
    "$(grep -qi 'workflow validation' "$STDERR_FILE" && echo yes || echo no)"

  # 6) Diff query hard-fails (gh unavailable/errors) -> must degrade to the
  #    generic message, never crash under `set -euo pipefail` (rc==1, not 2).
  rc="$(COMMENTS_JSON="$EMPTY_COMMENTS" DIFF_RC=1 run_capture "$PR" "$STARTED")"
  check "diff query hard-fails -> exit 1 (not 2, no crash)" "1" "$rc"
  check "  ...stderr keeps the generic 'rerun the Code Review workflow' text" "yes" \
    "$(grep -qF 'rerun the Code Review workflow' "$STDERR_FILE" && echo yes || echo no)"

  # 7) No interference on success: detection only ever runs on the STEP B
  #    FAILURE path — a fresh verdict comment still short-circuits to exit 0.
  rc="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" DIFF_FILES="$WORKFLOW_TOUCHING_DIFF" \
        run_capture "$PR" "$STARTED")"
  check "guard detection does not interfere with a successful verdict -> exit 0" "0" "$rc"

  # 8) STEP A precedence intact: an execution-file is_error still wins and its
  #    message is untouched by the guard, even when the diff touches the workflow.
  rc="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" DIFF_FILES="$WORKFLOW_TOUCHING_DIFF" \
        run_capture "$PR" "$STARTED" --execution-file "$ERROR_EXEC")"
  check "STEP A precedence: execution-file is_error still wins over guard detection -> exit 1" \
    "1" "$rc"
  check "  ...stderr still names 'review agent errored'" "yes" \
    "$(grep -qiF 'review agent errored' "$STDERR_FILE" && echo yes || echo no)"
  check "  ...stderr has no guard vocabulary (STEP A short-circuits before STEP B)" "no" \
    "$(grep -qi 'workflow validation' "$STDERR_FILE" && echo yes || echo no)"

  # --- 10) cwd-independence guard -----------------------------------------------
  mkdir -p "$WORK/sub"
  a="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" run_capture "$PR" "$STARTED")"
  b="$(COMMENTS_JSON="$FRESH_LGTM_COMMENTS" run_capture_sub "$PR" "$STARTED")"
  check "identical result from \$WORK vs a subdirectory (cwd-independence guard)" "$a" "$b"
else
  echo "  skip - real-jq comment/verdict cases (jq not installed)"
fi

# --- 9) missing required args --------------------------------------------------
rc="$(run_capture)"
check "no args at all -> exit 2 (usage)" "2" "$rc"

rc="$(run_capture "$PR")"
check "PR given but STARTED_AT missing -> exit 2 (usage)" "2" "$rc"

# --- 11) static guard over .github/workflows/code-review.yml ------------------
# These pin the #135/#140 fix: a verification step must exist, its
# REVIEW_STARTED_AT capture must happen BEFORE the review action runs (else it
# would time the wrong window), the sticky-comment mode must stay off (an
# edited-in-place comment defeats the "was a FRESH verdict posted" freshness
# check), and the token must flow through `env:`, never inlined in `run:`
# (inline `GH_TOKEN=...` in a run: body is both a secret-exposure smell and
# unnecessary once `env:` is available).
verify_step_hits=$(grep -cF -- 'assert-review-posted.sh' "$CODE_REVIEW_YML" 2>/dev/null) || true
if [[ "$verify_step_hits" -ge 1 ]]; then verify_step_present=yes; else verify_step_present=no; fi
check "code-review.yml invokes assert-review-posted.sh as a verification step" \
  "yes" "$verify_step_present"

started_at_line=$(grep -n -- 'REVIEW_STARTED_AT' "$CODE_REVIEW_YML" 2>/dev/null | head -1 | cut -d: -f1) || true
claude_review_line=$(grep -n -- 'id: claude-review' "$CODE_REVIEW_YML" 2>/dev/null | head -1 | cut -d: -f1) || true
if [[ -n "$started_at_line" && -n "$claude_review_line" ]] \
   && [[ "$started_at_line" -lt "$claude_review_line" ]]; then
  order_status=before
else
  order_status=not-before-or-missing
fi
check "REVIEW_STARTED_AT is captured BEFORE the id: claude-review step" \
  "before" "$order_status"

sticky_hits=$(grep -c -- 'use_sticky_comment' "$CODE_REVIEW_YML" 2>/dev/null) || true
check "code-review.yml does not set use_sticky_comment" "0" "$sticky_hits"

env_mapping_hits=$(grep -cE -- '^[[:space:]]*GH_TOKEN:' "$CODE_REVIEW_YML" 2>/dev/null) || true
inline_token_hits=$(grep -cE -- 'GH_TOKEN=' "$CODE_REVIEW_YML" 2>/dev/null) || true
if [[ "$env_mapping_hits" -ge 1 && "$inline_token_hits" -eq 0 ]]; then
  token_status=env-mapping
else
  token_status=missing-or-inline
fi
check "GH token reaches the verification step via an env: mapping, not inline in run:" \
  "env-mapping" "$token_status"

# --- 11b) static guard: workflow-validation-guard regression comment ----------
# Pins the required regression-comment update in code-review.yml (mirroring the
# section-11 grep-guard style): the workflow file must itself document the
# workflow-validation-guard behavior (claude-code-action@v1 silently skipping
# the review agent on PRs whose diff touches this very file) so future editors
# don't reintroduce the misleading generic "rerun" message on that path.
workflow_guard_hits=$(grep -ci -- 'workflow validation' "$CODE_REVIEW_YML" 2>/dev/null) || true
if [[ "$workflow_guard_hits" -ge 1 ]]; then workflow_guard_documented=yes; else workflow_guard_documented=no; fi
check "code-review.yml documents the workflow-validation guard" \
  "yes" "$workflow_guard_documented"

# --- 11c) static guard: the review agent holds NO write-capable tool ----------
# The #152 fix raises the job's `pull-requests` permission from read to WRITE so
# the deterministic "Post review" step can comment with the workflow's own
# token. That is only safe because the AGENT itself was simultaneously stripped
# of every write-capable tool: if a future edit re-adds one while the token is
# write, a prompt-injected PR diff could make the agent post or approve
# arbitrarily. This guard is the thing standing between those two facts.
#
# It is an ALLOWLIST, deliberately, not a denylist of known-bad verbs: a
# denylist silently permits whatever nobody thought to ban. Every entry in
# --allowed-tools must match one of the read-only forms below, or this fails.
allowed_tools_line=$(grep -oE -- '--allowed-tools "[^"]*"' "$CODE_REVIEW_YML" 2>/dev/null | head -1) || true
allowed_tools_value=${allowed_tools_line#--allowed-tools \"}
allowed_tools_value=${allowed_tools_value%\"}

if [[ -z "$allowed_tools_value" ]]; then
  agent_tools_status=missing
else
  agent_tools_status=read-only
  IFS=',' read -ra _tools <<<"$allowed_tools_value"
  for _tool in "${_tools[@]}"; do
    _tool="${_tool#"${_tool%%[![:space:]]*}"}"   # ltrim
    _tool="${_tool%"${_tool##*[![:space:]]}"}"   # rtrim
    [[ -z "$_tool" ]] && continue
    if ! [[ "$_tool" =~ ^Bash\(gh\ (issue\ view|issue\ list|search|pr\ diff|pr\ view|pr\ list):\*\)$ ]]; then
      agent_tools_status="write-capable:$_tool"
      break
    fi
  done
fi
check "review agent's --allowed-tools are all read-only (no write verbs)" \
  "read-only" "$agent_tools_status"

# The write token is only justified while the agent is toolless-for-writes, so
# pin the pairing itself: if someone drops the permission back to read the
# posting step breaks silently, and if someone raises it without the allowlist
# above the guard is the only thing left.
pr_write_hits=$(grep -cE -- '^[[:space:]]*pull-requests:[[:space:]]*write' "$CODE_REVIEW_YML" 2>/dev/null) || true
if [[ "$pr_write_hits" -ge 1 ]]; then pr_write_status=write; else pr_write_status=not-write; fi
check "code-review.yml grants pull-requests: write for the deterministic poster" \
  "write" "$pr_write_status"

# --- 11d) static guard: the posted verdict line matches the shared regex ------
# The verdict line is now rendered by a printf in workflow YAML rather than by
# the agent, so nothing statically ties its format to verdict-regex.sh -- the
# single source of truth that pr-ready.sh (merge gate) and assert-review-posted.sh
# (post gate) both parse. A format drift here is the #135 silent-stall class:
# a verdict the poster considers valid but the merger never sees.
# shellcheck source=scripts/ralph/verdict-regex.sh
source "$RALPH_DIR/verdict-regex.sh"
printf_fmt=$(grep -oE -- "printf '[^']*## Verdict: %s[^']*'" "$CODE_REVIEW_YML" 2>/dev/null | head -1) || true
if [[ -z "$printf_fmt" ]]; then
  verdict_format_status=missing
else
  # Render exactly what the workflow renders, then match it the SAME way the
  # real gates do -- jq's test(), not grep. The constants carry Oniguruma
  # inline flags ((?im)), which only jq honours; matching them any other way
  # here would test a different thing than the gates enforce.
  rendered=$(printf '\n\n## Verdict: %s\n' "LGTM")
  if [[ "$(jq -Rrn --arg b "$rendered" '$b | test("'"$VERDICT_RE"'")')" == "true" \
     && "$(jq -Rrn --arg b "$rendered" '$b | test("'"$VERDICT_LGTM_RE"'")')" == "true" ]]; then
    verdict_format_status=matches-shared-regex
  else
    verdict_format_status=drifted
  fi
fi
check "the posted '## Verdict: <X>' line matches verdict-regex.sh's shared patterns" \
  "matches-shared-regex" "$verdict_format_status"

# --- 12) static guard: self-wiring into ralph-fleet-tests.yml -----------------
run_list_hits=$(grep -cF -- 'test_assert_review_posted.sh' "$FLEET_TESTS_YML" 2>/dev/null) || true
if [[ "$run_list_hits" -ge 1 ]]; then run_list_wired=yes; else run_list_wired=no; fi
check "ralph-fleet-tests.yml run: list invokes test_assert_review_posted.sh" \
  "yes" "$run_list_wired"

paths_filter_hits=$(grep -cF -- 'code-review.yml' "$FLEET_TESTS_YML" 2>/dev/null) || true
if [[ "$paths_filter_hits" -ge 1 ]]; then paths_filter_wired=yes; else paths_filter_wired=no; fi
check "ralph-fleet-tests.yml paths filters include code-review.yml" \
  "yes" "$paths_filter_wired"

# --- 13) actionlint, when available, must pass over code-review.yml -----------
# Bounded/graceful: this repo's CI may not have actionlint on PATH, and that
# must never fail the suite — only its ABSENCE is skipped, never its findings.
if command -v actionlint >/dev/null 2>&1; then
  rc=0
  actionlint "$CODE_REVIEW_YML" > "$WORK/actionlint.out" 2>&1 || rc=$?
  if [[ "$rc" -eq 0 ]]; then al_status=clean; else al_status=dirty; fi
  check "actionlint passes over code-review.yml" "clean" "$al_status"
  if [[ "$rc" -ne 0 ]]; then
    echo "  actionlint findings:"
    sed 's/^/    /' "$WORK/actionlint.out"
  fi
else
  echo "  skip - actionlint not on PATH; static workflow lint skipped"
fi

# --- summary --------------------------------------------------------------
echo
echo "assert-review-posted tests: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
