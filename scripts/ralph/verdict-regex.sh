#!/usr/bin/env bash
# scripts/ralph/verdict-regex.sh
#
# Single source of truth for the review-verdict regex constants. Sourced by
# BOTH pr-ready.sh (the merge gate — "is there a FRESH LGTM?") and
# assert-review-posted.sh (the post gate — "did the review agent post a verdict
# at all during this run?"). Keeping the two checks byte-identical is the whole
# point: a drift between the posting check and the merging check is exactly the
# class of silent-stall bug #135 fixes — a verdict the poster considers valid
# but the merger silently ignores (or vice versa) starves the lane forever.
#
# NOTE: this is a sourced fragment, not a standalone script. It deliberately
# does NOT set `set -euo pipefail` — it is sourced into callers that already do,
# and re-arming shell options in a fragment would clobber the caller's state.
# It only declares constants.
#
# Idempotent: if these constants are already defined in this shell, return early.
# The `readonly` declarations below would otherwise abort (and, under a caller's
# `set -e`, kill the whole script) if this fragment were sourced twice into one
# process — e.g. a future harness that sources both pr-ready.sh and
# assert-review-posted.sh. Today each consumer is its own process, so this is
# pure future-proofing.
[[ -n "${VERDICT_PREFIX_RE:-}" ]] && return 0
#
# The canonical verdict line `claude-code-review.yml` posts is
# `## Verdict: <LGTM|CHANGES_REQUESTED|COMMENTS>` (also tolerated: `**Verdict:**`
# and a bare `Verdict:`), sitting at the END of a longer `## Summary …` body — so
# the match must be case-insensitive AND multiline (`m`, so `^` anchors to the
# verdict line — which sits at the END of a multi-line `## Summary …` body, not
# at string start), prefix-tolerant, and keyed to the verdict LINE (a stray
# "LGTM" in prose must not count). Reviewers are also seen posting `## Verdict`
# as a bare heading with an emoji-prefixed token on the NEXT line (e.g.
# `## Verdict\n✅ LGTM`), so the LGTM separator tolerates any non-alphanumeric
# decoration (emoji/whitespace/newline) between `verdict` and `lgtm`. That stays
# safe: a stray "LGTM" in prose never matches (it is keyed to the verdict line),
# and a non-LGTM token like COMMENTS/CHANGES_REQUESTED puts a letter right after
# the emoji, breaking the non-alphanumeric run before any later "LGTM".
# `VERDICT_COMMENTS_RE` is the exact mirror of `VERDICT_LGTM_RE` for the
# `COMMENTS` token: same emoji/newline-tolerant `[^a-zA-Z0-9]+` separator, so
# `## Verdict\n💬 COMMENTS`, `## Verdict: 💬 COMMENTS`, `## Verdict: COMMENTS`,
# and `**Verdict:** COMMENTS` all match, and the same prose-guard property holds
# — the run stops at the first alphanumeric char, so a stray "comments" word in
# later prose (after an LGTM/CHANGES_REQUESTED line) never matches. The two
# token matchers widen; `VERDICT_RE` (comment selection) keeps its strict
# `[:*\s]` class unchanged. Backslashes are doubled because this text is spliced
# into a jq string literal, where `\s` is an invalid escape and must reach the
# regex engine as `\\s` (the negated class `[^a-zA-Z0-9]` has no backslash, so
# it is spelled literally — the explicit form self-documents and sidesteps the
# Oniguruma subtlety of case-folding a negated class). The per-branch fragments
# are SINGLE-quoted (not folded into the surrounding double quotes) so their
# `\\s` survives verbatim: inside double quotes bash would collapse `\\s` → `\s`,
# which jq then rejects as an invalid escape — the class must stay `[:*\\s]`.
# VERDICT_RE, VERDICT_LGTM_RE, and VERDICT_COMMENTS_RE are consumed by the
# scripts that source this fragment (pr-ready.sh uses all three;
# assert-review-posted.sh uses VERDICT_RE), so a standalone shellcheck of this
# file cannot see their use — silence SC2034.
readonly VERDICT_PREFIX_RE='(?im)^\\s*(?:#{1,6}\\s+|\\*\\*)?verdict'
# shellcheck disable=SC2034
readonly VERDICT_RE="${VERDICT_PREFIX_RE}"'[:*\\s]'
# shellcheck disable=SC2034
readonly VERDICT_LGTM_RE="${VERDICT_PREFIX_RE}"'[^a-zA-Z0-9]+lgtm'
# shellcheck disable=SC2034
readonly VERDICT_COMMENTS_RE="${VERDICT_PREFIX_RE}"'[^a-zA-Z0-9]+comments'
#
# --- SUBJECT BINDING (issue #400) --------------------------------------------
# A correctly-parsed verdict line proves a verdict was posted. It does NOT prove
# the review is ABOUT this PR, or about its CURRENT head commit. Issue #400 is
# exactly that hole: the review agent produced PR #398's review and the poster
# landed it, correctly and explicitly, on PR #396 — where it became the LATEST
# verdict. Every consumer that greps `^## Verdict` got a well-formed answer
# belonging to a different pull request.
#
# So the posted comment now states its own subject in a fixed, machine-readable
# form, rendered by code-review.yml's "Post review" step:
#
#     Review subject: PR #396 @ 635ac9d1ab7378f705303fcf802801d8c3ae5824
#
# REVIEW_SUBJECT_RE matches the SHAPE of that line (any PR, any SHA) and exists
# so the format has one owner: code-review.yml renders it, the #400 static guard
# in test_assert_review_posted.sh proves the rendering still conforms, and
# assert-review-posted.sh enforces the concrete pr/sha pair. Checking a SPECIFIC
# subject is deliberately NOT done with this regex — assert-review-posted.sh
# builds the exact literal line and uses jq `contains`, which needs no escaping
# and cannot be widened by accident.
#
# Same escaping rules as above: `\\s` and `\\d` are doubled because this text is
# spliced into a jq string literal and must reach Oniguruma as `\s` / `\d`, and
# the fragment is SINGLE-quoted so bash does not collapse the doubling first.
# The SHA class is `[0-9a-f]{7,40}` (git's abbreviated through full form); the
# `(?im)` flags are honoured only by jq's test(), never by BSD grep -E.
# shellcheck disable=SC2034
readonly REVIEW_SUBJECT_RE='(?im)^\\s*review\\s+subject:\\s*pr\\s*#[0-9]+\\s*@\\s*[0-9a-f]{7,40}\\s*$'
