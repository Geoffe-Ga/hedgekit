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
- ✅ Docstring coverage: ≥95%
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
