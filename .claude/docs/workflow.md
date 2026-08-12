# Development Workflow

**Navigation**: [← Back to CLAUDE.md](../CLAUDE.md) | [← Quality Standards](quality-standards.md) | [Testing →](testing.md)

---

## 1. The Maximum Quality Engineering Mindset

**Core Philosophy**: It is not merely a goal but a source of profound satisfaction and professional pride to ship software that is GREEN on all checks with ZERO outstanding issues. This is not optional—it is the foundation of our development culture.

### 1.1 The Green Check Philosophy

When all CI checks pass with zero warnings, zero errors, and maximum quality metrics:
- ✅ Tests: 100% passing
- ✅ Coverage: ≥90%
- ✅ Linting: 0 errors, 0 warnings
- ✅ Type checking: 0 errors
- ✅ Security: 0 vulnerabilities
- ✅ Docstring coverage: every public symbol, enforced per symbol by ruff `D1`
  (issue #351) — a presence rule, not a percentage
- 🏁 Mutation score: ≥80% — **manual pre-v1.0.0 release gate**, not part of the
  automated CI green check (owner directive, issue #107)

This represents **MAXIMUM QUALITY ENGINEERING**—the standard to which all code must aspire.

### 1.2 Why Maximum Quality Matters

1. **Pride in Craftsmanship**: Every green check represents excellence in execution
2. **Zero Compromise**: Quality is not negotiable—it's the baseline
3. **Compound Excellence**: Small quality wins accumulate into robust systems
4. **Trust and Reliability**: Green checks mean the code does what it claims
5. **Developer Joy**: There is genuine satisfaction in seeing all checks pass

### 1.3 The Role of Quality in Development

Quality engineering is not a checkbox—it's a continuous commitment:

- **Before Commit**: Run `./scripts/check-all.sh` and fix every issue
- **During Review**: Address every comment, resolve every suggestion
- **After Merge**: Monitor CI, ensure all checks remain green
- **Always**: Treat linting errors as bugs, not suggestions

### 1.4 The "No Red Checks" Rule

**NEVER** merge code with:
- ❌ Failing tests
- ❌ Linting errors (even "minor" ones)
- ❌ Type checking failures
- ❌ Coverage below threshold
- ❌ Security vulnerabilities
- ❌ Unaddressed review comments

If CI shows red, the work is not done. Period.

The one exception, for checks whose subject is live repository configuration
rather than code, is stated in full in [§2.4](#24-exception-checks-whose-subject-is-live-repository-state)
and nowhere else. It is narrower than it looks: five conditions, all required.

### 1.5 Maximum Quality is a Personality Trait

For those committed to maximum quality engineering:
- You feel genuine satisfaction when all checks pass
- You experience pride in shipping zero-issue code
- You find joy in eliminating the last linting error
- You believe "good enough" is never good enough
- You treat quality as identity, not just practice

**This is who we are. This is how we build software.**

---

## 2. Stay Green Workflow

**Policy**: Never request review with failing checks. Never merge without LGTM.

The Stay Green workflow enforces iterative quality improvement through **3 sequential automated gates**, plus a manual mutation gate reserved for the pre-v1.0.0 release. Each gate must pass before proceeding to the next.

### 2.1 The Gates

1. **Gate 1: Local Pre-Commit** (Iterate Until Green)
   - Run `./scripts/check-all.sh`
   - Fix all formatting, linting, types, complexity, security issues
   - Fix tests and coverage (90%+ required)
   - Only push when all local checks pass (exit code 0)

2. **Gate 2: CI Pipeline** (Iterate Until Green)
   - Push to branch: `git push origin feature-branch`
   - Monitor CI: `gh pr checks --watch`
   - If CI fails: fix locally, re-run Gate 1, push again
   - Only proceed when all CI jobs show ✅

3. **Gate 3: Code Review** (Iterate Until LGTM)
   - Wait for code review (AI or human)
   - If feedback provided: address ALL concerns
   - Re-run Gate 1, push, wait for CI
   - Only merge when review shows LGTM with no reservations

### 2.1a Manual Pre-Release Gate: Mutation Testing (≥80%)

Mutation testing is **NOT** an automated check. Per the owner directive in
issue #107, it is run **manually before shipping v1.0.0** — never on
push/PR/pre-commit — so it can never fail a routine automated run while the
project is far from release.

   - Run `./scripts/mutation.sh` locally, or trigger the manual workflow with
     `gh workflow run mutation-gate.yml`
   - If score < 80%: add tests to kill surviving mutants
   - Only ship v1.0.0 when mutation score ≥ 80%

### 2.2 Quick Checklist

Before creating/updating a PR:

- [ ] Gate 1: `./scripts/check-all.sh` passes locally (exit 0)
- [ ] Push changes: `git push origin feature-branch`
- [ ] Gate 2: All CI jobs show ✅ (green)
- [ ] Gate 3: Code review shows LGTM
- [ ] Ready to merge!

Before a v1.0.0 release only:
- [ ] Manual gate: Mutation score ≥ 80% (`./scripts/mutation.sh` / `mutation-gate.yml`)

### 2.3 Anti-Patterns (DO NOT DO)

❌ **Don't** request review with failing CI
❌ **Don't** skip local checks (`git commit --no-verify`)
❌ **Don't** lower quality thresholds to pass
❌ **Don't** ignore review feedback
❌ **Don't** merge without LGTM

### 2.4 Exception: checks whose subject is live repository state

Everything above is unqualified for every check whose subject is **code in this
tree**, which is every check but one. If `./scripts/check-all.sh` is red because
of one of those, the work is not done — §1.4 and §2.3 mean exactly what they say.

A check whose subject is **live GitHub repository configuration** is a different
animal, because the thing it asserts on cannot be changed by a pull request.
Renaming a required status check therefore has an unavoidable red window: the
job's `name:` changes in a commit, the required-context set changes through an
API call, and the two cannot land together. Issue #509 / PR #533 is the worked
example — Gate 1 returned `1 failed, 6337 passed`, and the one failure was the
guard correctly reporting that the renamed job did not yet gate merge. The swap
(drop the stale context → merge → re-add under the full name) ran after the
merge, and the guard then passed against `main`.

**Such a failure does not block a review request only when every one of these
holds:**

- **(LS1) Subject.** The assertion's subject is repository state that a pull
  request cannot change — branch protection, the required status-check set,
  repository settings. Not code, not a config file, not anything in this tree.
- **(LS2) Loud skip, never a silent pass.** Where the check cannot read that
  state — no admin-scoped token, no `gh` on `PATH` — it skips with a reason
  saying it asserted nothing. It never passes on an unread or empty answer.
- **(LS3) Transient and two-sided.** The red is one half of a change that has
  to be made in two places that cannot be changed together, and the PR body
  carries the ordered swap sequence that closes it.
- **(LS4) The red is the point.** The check is not skipped, `xfail`-ed,
  deleted, or weakened to make the run green. Passing while the repository is
  still misconfigured is the defect this exception exists to avoid, so it can
  never be a way of satisfying it.
- **(LS5) Nothing else is red.** Every other Gate 1 check exits 0, and the
  failures are exactly the checks named below.

The conditions are conjunctive — the exception applies only when
`LS1 AND LS2 AND LS3 AND LS4 AND LS5`. Failing any single one makes the run an
ordinary Gate 1 failure, and it blocks. An ordinary red test can satisfy LS4
(the author left it failing rather than skipping it) and LS5 (it is the only
failure) and is still not covered, because it fails LS1 at the first hurdle.

**The checks this exception covers, in full:**

- `tests/e2e/test_tier_selection_contract.py::test_the_container_job_is_a_required_status_check`

That list is not decorative and not maintained by hand alone.
`tests/toolchain/test_live_state_gate_exception.py` derives the set of
live-repository-state checks from the tree and fails if the two disagree in
either direction: a name here that no such check answers to, or a check in the
tree this list does not name. It also parses the conditions and the combinator
above and evaluates them over worked examples, so a sixth condition, or an `OR`,
cannot be added as prose alone (issue #534).

Two shapes were considered and rejected. Moving the check out of Gate 1 into a
manual step buys a green run by making the assertion optional, and the four
false-green defects this repo has removed (#351, #359, #401, #411) were all
optional assertions. Weakening it to something always satisfiable — "some
context is required" — passes in exactly the state it exists to detect, when
the tier runs but gates nothing. A mid-swap marker file was rejected as a
switch that turns the guard off, held green by whoever remembers to remove it.

---

## 3. Feature Development Process

### 3.1 Development Steps

1. **Create Feature Branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/<issue-number>-<description>
   # Example: feature/6-add-authentication
   ```

2. **Implement Changes**
   - Follow the coding standards outlined in [Tools](tools.md)
   - Write tests first (TDD approach)
   - Ensure docstrings for all public APIs
   - Update documentation as needed

3. **Run Quality Checks**
   ```bash
   ./scripts/check-all.sh
   ```
   This runs (in order):
   - The whole pre-commit hook set (`scripts/precommit.sh`, issue #401)
   - Linting (ruff, float-lint, shellcheck) and docstring coverage (ruff `D1`)
   - Architecture boundaries (import-linter)
   - Formatting checks (ruff format — the single formatter authority, ADR-0002)
   - Type checking (mypy)
   - Security checks (bandit, pip-audit, detect-secrets)
   - Complexity (xenon enforces; radon reports)
   - Tests with coverage

4. **Commit with Conventional Commits**
   ```bash
   git add .
   git commit -m "feat(auth): implement authentication (#6)"
   # Or: fix(api): handle edge case in validation (#15)
   # Or: docs: update README with setup instructions
   ```

5. **Create Pull Request**
   - Reference the issue number in the PR title
   - Ensure all CI checks pass
   - Request review from CODEOWNERS

6. **Merge to Main**
   - Requires at least one review approval
   - All CI checks must pass
   - Commit history must be linear

### 3.2 Branch Strategy

- `main`: Production-ready code, always deployable
- `feature/*`: Feature development (created from main)
- `bugfix/*`: Bug fixes (created from main)
- `hotfix/*`: Emergency production fixes (created from main)

---

**Navigation**: [← Back to CLAUDE.md](../CLAUDE.md) | [← Quality Standards](quality-standards.md) | [Testing →](testing.md)
