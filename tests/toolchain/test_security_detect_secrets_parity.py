"""Failing-first tests pinning the detect-secrets enforcement-parity contract.

Issue #262 closes an enforcement-parity gap: `scripts/security.sh` --
the script `./scripts/check-all.sh` (Gate 1) actually invokes -- only runs
`detect-secrets scan .` inside an `if $FULL; then ... fi` block, guarded on
`command -v detect-secrets` being present, and the result is discarded with
`|| true`. `check-all.sh` calls `security.sh` with no `--full` flag (see
`run_check "Security checks" "security.sh"`), so on every local Gate 1 run
and in CI's `check-all.sh` invocation, detect-secrets never runs at all --
and even when it is run manually with `--full`, its exit code is thrown
away, so a leaked secret would never fail the check. Meanwhile the
`detect-secrets` pre-commit hook (`Yelp/detect-secrets`, `.pre-commit-
config.yaml`) *does* enforce a `.secrets.baseline` diff on every commit, so
today's "protection" only exists at the pre-commit layer -- a layer
developers can skip with `--no-verify` and CI's `check-all.sh` step never
re-checks.

The target state: `scripts/security.sh` runs `pre-commit run detect-secrets
--all-files` unconditionally (not gated behind `--full`, not silenced with
`|| true`, not soft-gated behind `command -v detect-secrets`), delegating to
the exact same Yelp/detect-secrets hook -- and therefore the exact same
`.secrets.baseline` -- that pre-commit already enforces. This makes
`check-all.sh` (Gate 1) and pre-commit enforce detect-secrets identically:
whatever passes/fails locally via pre-commit passes/fails identically via
`./scripts/check-all.sh`, with no `--full` opt-in required and no swallowed
exit code. A failure must also point the developer at `.secrets.baseline`
with remediation guidance that does not amount to "just regenerate the
baseline" (which would silently launder a real secret into the allowlist).

These assertions began life as Gate 1 RED for issue #262: `scripts/
security.sh`'s current source contains no `pre-commit run detect-secrets
--all-files` invocation, still guards its (non-enforcing) detect-secrets
call behind `if $FULL`, still runs the swallowed-exit-code `detect-secrets
scan . || true` behind a soft `command -v detect-secrets` gate, and carries
no user-facing `.secrets.baseline` remediation message at all. Test 1 below
is the sole exception: it pins the pre-commit hook's existing, already-
correct baseline-enforcing configuration as a lockstep anchor, and is green
today and expected to stay green -- it exists to prove the hook that
`security.sh` will delegate to is itself baseline-enforcing, not to pin a
gap.
"""

from __future__ import annotations

from pathlib import Path

from tests.toolchain.test_toolchain_pins import _find_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SECURITY_SCRIPT_PATH = _REPO_ROOT / "scripts" / "security.sh"

#: Repo URL substring identifying the Yelp/detect-secrets pre-commit repo,
#: reused with `test_toolchain_pins._find_repo` rather than re-parsing
#: `.pre-commit-config.yaml` independently. The constant is deliberately named
#: without a denylist keyword: detect-secrets' KeywordDetector matches any
#: identifier embedding ``secret``/``key``/``token`` adjacent to a quoted value
#: (``\w*secret\w*\s*=\s*"..."``), so a name like ``_DETECT_SECRETS_...`` would
#: flag itself as a "Secret Keyword" finding.
_DETECT_HOOK_REPO_SUBSTRING = "detect-secrets"

#: The exact command `scripts/security.sh` must run to enforce detect-secrets
#: identically to the pre-commit hook (same hook, same `.secrets.baseline`).
_ENFORCING_INVOCATION = "pre-commit run detect-secrets --all-files"

#: A remediation keyword that must accompany any `.secrets.baseline`
#: reference in a failure/guidance context, steering a developer away from
#: silently regenerating (and thereby laundering a real secret into) the
#: baseline. `security.sh` must use one of these -- or an equivalent
#: instruction not to blindly regenerate the baseline -- rather than a bare
#: `.secrets.baseline` mention with no next step.
_REMEDIATION_KEYWORDS = ("do not regenerate", "audit")


def _security_source() -> str:
    """Read `scripts/security.sh` as text.

    Returns:
        The full source of `scripts/security.sh`.

    Raises:
        FileNotFoundError: If the script does not exist.
    """
    return _SECURITY_SCRIPT_PATH.read_text(encoding="utf-8")


def test_detect_secrets_hook_is_baseline_enforcing() -> None:
    """The Yelp/detect-secrets pre-commit hook enforces `.secrets.baseline`.

    Lockstep anchor, not a gap pin: this passes today and must keep passing,
    because `scripts/security.sh`'s fix (delegating to `pre-commit run
    detect-secrets --all-files`) is only meaningful if the hook it delegates
    to is itself baseline-enforcing. Uses `_find_repo` (shared with
    `test_precommit_scope.py`) rather than re-parsing the pre-commit config,
    so both test modules agree on what "the detect-secrets repo" means.
    """
    repo = _find_repo(_DETECT_HOOK_REPO_SUBSTRING)
    assert repo is not None, (
        f"no pre-commit repo matches {_DETECT_HOOK_REPO_SUBSTRING!r}"
    )

    hooks = {hook["id"]: hook for hook in repo["hooks"]}
    assert "detect-secrets" in hooks, (
        "Yelp/detect-secrets repo has no 'detect-secrets' hook"
    )

    args = hooks["detect-secrets"].get("args", [])
    assert "--baseline" in args, (
        f"detect-secrets hook args {args!r} missing '--baseline'"
    )
    assert ".secrets.baseline" in args, (
        f"detect-secrets hook args {args!r} missing '.secrets.baseline'"
    )


def test_security_script_runs_enforcing_precommit_invocation() -> None:
    """`security.sh` must run `pre-commit run detect-secrets --all-files`.

    RED today: the current script never invokes `pre-commit run
    detect-secrets`; it instead runs the raw, non-enforcing `detect-secrets
    scan .` (see the two tests below). Delegating to the pre-commit
    invocation is what guarantees `security.sh` enforces the exact same
    hook -- and therefore the exact same `.secrets.baseline` -- that
    pre-commit already enforces on every commit.
    """
    source = _security_source()

    assert _ENFORCING_INVOCATION in source, (
        f"scripts/security.sh does not run {_ENFORCING_INVOCATION!r} -- "
        "detect-secrets enforcement does not match the pre-commit hook"
    )


def test_security_script_does_not_gate_detect_secrets_behind_full() -> None:
    """detect-secrets enforcement must run unconditionally, not behind `--full`.

    RED today: the existing detect-secrets block is nested inside `if
    $FULL; then ... fi`, and `check-all.sh` (Gate 1) invokes `security.sh`
    with no `--full` flag (`run_check "Security checks" "security.sh"`), so
    detect-secrets silently never runs during Gate 1 or CI. Asserting the
    literal `if $FULL` conditional is gone from the source proves the
    check has been moved onto the unconditional/default execution path.
    """
    source = _security_source()

    assert "if $FULL" not in source, (
        "scripts/security.sh still gates a check behind 'if $FULL' -- "
        "check-all.sh never passes --full, so anything inside this block "
        "never runs during Gate 1 or CI"
    )


def test_security_script_has_no_non_enforcing_detect_secrets_decoy() -> None:
    """The old non-enforcing `detect-secrets scan` path must be fully removed.

    RED today, three ways at once so a partial fix still fails:

    1. The raw `detect-secrets scan` invocation (which never delegates to
       `.secrets.baseline` the way the pre-commit hook does) must be gone.
    2. No source line may combine `detect-secrets` with `|| true`: a `||
       true` swallows the exit code, so even a run that finds a real
       secret would report success.
    3. The soft `command -v detect-secrets` existence gate must be gone:
       today, if `detect-secrets` isn't on PATH, the whole check is
       silently skipped rather than failing loudly.
    """
    source = _security_source()

    assert "detect-secrets scan" not in source, (
        "scripts/security.sh still runs the non-enforcing 'detect-secrets "
        "scan' invocation instead of delegating to the pre-commit hook"
    )

    offending_lines = [
        line
        for line in source.splitlines()
        if "detect-secrets" in line and "|| true" in line
    ]
    assert not offending_lines, (
        f"scripts/security.sh swallows detect-secrets' exit code: {offending_lines}"
    )

    assert "command -v detect-secrets" not in source, (
        "scripts/security.sh still soft-gates detect-secrets behind "
        "'command -v detect-secrets' -- a missing binary must fail loudly, "
        "not silently skip the check"
    )


def test_security_script_baseline_failure_message_has_remediation_guidance() -> None:
    """A detect-secrets failure must point at `.secrets.baseline` with guidance.

    RED today: the script has no user-facing `.secrets.baseline` message at
    all (the current `--full`-gated block never references the baseline
    file in any echo/error text). What/why/next contract this pins for the
    eventual fix:

    - *what*: the failure message names `.secrets.baseline` explicitly, so
      a developer immediately knows which file is implicated.
    - *why*: implicitly, a secret was detected that isn't already in the
      allowlisted baseline.
    - *next*: the message must carry a remediation cue steering the
      developer toward reviewing/auditing the finding (`_REMEDIATION_
      KEYWORDS`) rather than reflexively regenerating the baseline --
      blindly regenerating it would launder a genuine leaked secret into
      the allowlist instead of surfacing it.
    """
    source = _security_source().lower()

    assert ".secrets.baseline" in source, (
        "scripts/security.sh has no user-facing '.secrets.baseline' "
        "reference -- a detect-secrets failure gives no guidance about "
        "which file to consult"
    )

    assert any(keyword in source for keyword in _REMEDIATION_KEYWORDS), (
        "scripts/security.sh's '.secrets.baseline' reference carries no "
        f"remediation cue (expected one of {_REMEDIATION_KEYWORDS!r}) -- a "
        "developer must be steered toward reviewing the finding, not "
        "blindly regenerating the baseline"
    )
