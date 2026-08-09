## Context

The Ralph multi-agent router (`.claude/agents/`) currently pins every code-writing
agent to `fable` and the `ralph-chief-architect` to `opus`. Owner directive
(2026-07-24) inverts that: planning is the highest-leverage seat in a tick, so the
architect should run on **Fable 5**, and every other write-agent on **Opus 5**.

Fable capacity is metered separately from Opus, and Claude Code has no per-agent
fallback *chain* — a subagent's model resolves from `CLAUDE_CODE_SUBAGENT_MODEL`,
then the per-invocation `model` parameter, then the definition's frontmatter. So a
tick that dispatches the architect must degrade to Opus rather than stall when
Fable credits run out.

## Goal

1. Swap every `model: fable` frontmatter pin to `model: opus` (`ralph-worker`,
   `ralph-test-specialist`, `ralph-implementation-specialist`,
   `ralph-performance-specialist`, `ralph-documentation-specialist`).
2. Pin `ralph-chief-architect` to `model: fable`.
3. Add a graceful, documented fallback for exhausted Fable credits.

## Acceptance

- [ ] `ralph-chief-architect` is the only agent on `fable`; all other write-agents
      are on `opus`; `sonnet` (review) and `haiku` (dep checks) are unchanged.
- [ ] In-session fallback documented: the conductor re-dispatches the architect
      with a per-invocation `model: "opus"` override.
- [ ] Durable fallback: `scripts/ralph/architect-model.sh [fable|opus]` flips the
      checked-in pin (no argument prints the current pin), with an allowlist so a
      typo cannot pin an unsanctioned model.
- [ ] Shell tests for the switch, wired into `ralph-fleet-tests.yml`.
- [ ] `.claude/agents/shared/README.md`, `scripts/ralph/PROMPT.md`, and
      `ralph-worker.md` reflect the new tiers and the fallback path.
