#!/usr/bin/env bash
# scripts/provision-venv.sh - Provision/refresh the SHARED repo-local .venv
# Usage: ./scripts/provision-venv.sh [--print-venv | --check | --help]
#
# Provisions a SINGLE shared virtual environment at the MAIN repo root and
# reuses it from every fleet worktree, so check-all.sh runs against the pinned
# quality toolchain (constraints-quality.txt) regardless of the operator's
# global Python -- ending fleet-wide pip-audit drift noise (issue #133).
#
# The shared venv lives at "<main-worktree-root>/.venv". A call from inside a
# linked worktree (.ralph/worktrees/issue-N/) still resolves to the SAME main
# root .venv, so the toolchain is installed once and shared by all lanes.
#
# CONCURRENCY: because the .venv is shared, two fleet lanes that independently
# detect toolchain drift could otherwise run `python3 -m venv` / `pip install`
# into the SAME site-packages at once and corrupt it (interleaved writes), or
# thrash it back and forth when their branches pin different constraints. So the
# create+install section is serialized with an advisory lock:
#   - Mechanism: a mkdir-based lock directory ("<venv>.lock"). mkdir is atomic
#     on every POSIX filesystem, and unlike flock(1) it needs no extra binary --
#     macOS ships no flock(1) by default, while the Linux CI runner does, so a
#     mkdir lock is the portable common denominator that works on both.
#   - Liveness: the lock records its holder's PID; a lock whose holder PID is no
#     longer alive is treated as stale and reclaimed (atomically, so two lanes
#     can't both reclaim it), so a lane that crashes mid-provision -- or is
#     SIGINT/SIGTERM-killed before its EXIT trap can release the lock -- can't
#     wedge the fleet forever. Waiters honor PROVISION_LOCK_TIMEOUT (seconds,
#     default 300) and never steal a lock held by a live process.
#   - Convergence (anti-thrash): after acquiring the lock, provision() RE-CHECKS
#     drift and skips reinstalling when a sibling lane already provisioned to
#     matching constraints. The shared venv therefore converges to the pins of
#     whichever lane last ran, rather than being reinstalled once per lane.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="provision"

# Parse command line arguments (mirrors check-all.sh / lint.sh conventions).
while [[ $# -gt 0 ]]; do
    case $1 in
        --print-venv)
            MODE="print-venv"
            shift
            ;;
        --check)
            MODE="check"
            shift
            ;;
        --help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Provision/refresh the SINGLE shared repo-local .venv at the main repo root and
reuse it from every fleet worktree so quality checks run against the pinned
toolchain (constraints-quality.txt).

OPTIONS:
    --print-venv   Print the resolved shared .venv absolute path and exit.
    --check        Drift check only (never creates/installs): exit 0 when the
                   installed toolchain matches constraints-quality.txt, nonzero
                   (with a provision-venv.sh hint) when drifted or missing.
    --help         Display this help message.
    (no option)    Provision: create the shared .venv if absent, then install
                   the pinned toolchain only when fresh or drifted (idempotent).

EXIT CODES:
    0           Success (in sync, provisioned, or informational)
    1           Drift detected / provisioning failed
    2           Error running the script (e.g. unknown option)

EXAMPLES:
    $(basename "$0")                # Provision or refresh the shared .venv
    $(basename "$0") --print-venv   # Print the shared .venv path
    $(basename "$0") --check        # Fail loudly on toolchain drift
EOF
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

# Trim leading/trailing whitespace (and CR) from a string.
trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# PEP-503 normalize a distribution name: lowercase and collapse any run of
# -, _ or . into a single "-". So "Foo_Bar" and "foo-bar" compare equal.
normalize_name() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[._-]+/-/g'
}

# Resolve the shared venv path: the MAIN worktree root .venv, even from inside
# a linked worktree, via git's common .git dir; fall back to this script's own
# PROJECT_ROOT .venv when outside a git repository. Anchored to PROJECT_ROOT
# (the checkout that owns this script), NOT the ambient CWD, so a caller's
# working directory can never redirect the shared venv to the wrong repo.
resolve_venv() {
    local common_dir main_root
    if common_dir="$(git -C "$PROJECT_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
        main_root="${common_dir%/.git}"
        if [[ "$main_root" == "$common_dir" ]]; then
            main_root="$(dirname "$common_dir")"
        fi
        main_root="$(cd "$main_root" 2>/dev/null && pwd -P)" || main_root="$PROJECT_ROOT"
    else
        main_root="$PROJECT_ROOT"
    fi
    printf '%s/.venv\n' "$main_root"
}

VENV="$(resolve_venv)"

# Advisory lock guarding the venv create+install section. Sits right next to the
# venv (same main repo root) so every fleet lane -- whichever worktree it runs
# from -- contends on the SAME lock. See the CONCURRENCY note in the header.
VENV_LOCK="${VENV}.lock"
LOCK_HELD=false

# Drift check: compare the pins in constraints-quality.txt against what is
# installed in the shared venv. Prints offenders + a provision-venv.sh hint to
# stderr. Returns 0 when in sync, 1 when drifted or the venv is missing.
# NEVER creates or mutates the venv.
drift_check() {
    local venv_python="$VENV/bin/python"
    if [[ ! -x "$venv_python" ]]; then
        printf 'Shared virtual environment missing at %s\n' "$VENV" >&2
        printf 'Run provision-venv.sh to create it.\n' >&2
        return 1
    fi

    local constraints="$PROJECT_ROOT/constraints-quality.txt"
    [[ -f "$constraints" ]] || return 0

    # Snapshot the installed toolchain as normalized "name==version" lines.
    local freeze_raw freeze_norm=""
    freeze_raw="$("$venv_python" -m pip list --format=freeze 2>/dev/null || true)"
    local fline fname fver fnorm
    while IFS= read -r fline; do
        fline="$(trim "$fline")"
        [[ -z "$fline" ]] && continue
        # Skip freeze lines that are not exact name==version pins (warnings,
        # "pkg @ file://...", "-e ...").
        [[ "$fline" != *"=="* ]] && continue
        fname="$(trim "${fline%%==*}")"
        fver="$(trim "${fline#*==}")"
        [[ "$fname" =~ ^[A-Za-z0-9._-]+$ ]] || continue
        fnorm="$(normalize_name "$fname")"
        freeze_norm+="${fnorm}==${fver}"$'\n'
    done <<< "$freeze_raw"

    # Compare every pin against the installed set.
    local offenders="" rline pname pver pnorm installed frozen
    while IFS= read -r rline; do
        # Strip inline "# comment" (e.g. "pytest==9.0.3  # CVE-...").
        rline="${rline%%#*}"
        rline="$(trim "$rline")"
        [[ -z "$rline" ]] && continue
        [[ "$rline" != *"=="* ]] && continue
        pname="$(trim "${rline%%==*}")"
        pver="$(trim "${rline#*==}")"
        [[ "$pname" =~ ^[A-Za-z0-9._-]+$ ]] || continue
        pnorm="$(normalize_name "$pname")"
        # Literal, anchored lookup of "<pnorm>==<ver>" in the installed set.
        # pnorm is already PEP-503 normalized ([a-z0-9-], dots collapsed away),
        # so the "${pnorm}==*" glob is a plain prefix match with no regex/glob
        # metacharacters -- tidier and safer than interpolating into a regex.
        installed=""
        while IFS= read -r frozen; do
            if [[ "$frozen" == "${pnorm}=="* ]]; then
                installed="$frozen"
                break
            fi
        done <<< "$freeze_norm"
        if [[ -z "$installed" ]]; then
            offenders+="  - ${pname} (pinned ${pver}, not installed)"$'\n'
        else
            installed="${installed#*==}"
            if [[ "$installed" != "$pver" ]]; then
                offenders+="  - ${pname} (pinned ${pver}, installed ${installed})"$'\n'
            fi
        fi
    done < "$constraints"

    if [[ -n "$offenders" ]]; then
        printf 'Toolchain drift from constraints-quality.txt:\n%s' "$offenders" >&2
        printf 'Run provision-venv.sh to refresh the shared .venv.\n' >&2
        return 1
    fi
    return 0
}

# True when the PID recorded in the lock is no longer running (a stale lock left
# by a lane that died mid-provision). An empty PID means a live racer is still
# writing it, so we wait rather than reclaim; a non-numeric PID is treated as
# stale (garbage the current script never writes).
lock_is_stale() {
    local holder
    holder="$(cat "$VENV_LOCK/pid" 2>/dev/null || true)"
    [[ -z "$holder" ]] && return 1
    [[ "$holder" =~ ^[0-9]+$ ]] || return 0
    if kill -0 "$holder" 2>/dev/null; then
        return 1
    fi
    return 0
}

# Acquire the advisory lock, waiting up to PROVISION_LOCK_TIMEOUT seconds for a
# live holder to release it (reclaiming a stale one immediately). Returns 1 on
# timeout without ever stealing a live lock.
acquire_lock() {
    local timeout="${PROVISION_LOCK_TIMEOUT:-300}" waited_ds=0 timeout_ds
    timeout_ds=$((timeout * 10))
    while ! mkdir "$VENV_LOCK" 2>/dev/null; do
        if lock_is_stale; then
            # Reclaim atomically: rename the stale dir to a unique name and only
            # the rename WINNER removes it. A plain `rm -rf` here would be racy
            # -- two lanes could both read the dead PID, both remove, and both
            # then mkdir (one clobbering the other's fresh lock), so both would
            # believe they hold it and run concurrent installs. `mv <dir>
            # <unique>` is atomic: the loser's source is already gone, its mv
            # fails, and it simply loops to re-contend on mkdir.
            mv "$VENV_LOCK" "$VENV_LOCK.stale.$$" 2>/dev/null && rm -rf "$VENV_LOCK.stale.$$"
            continue
        fi
        if ((waited_ds >= timeout_ds)); then
            printf 'Timed out after %ss waiting for the venv lock at %s\n' \
                "$timeout" "$VENV_LOCK" >&2
            printf 'Another provisioning run holds it; remove it if it is stale.\n' >&2
            return 1
        fi
        sleep 0.2
        waited_ds=$((waited_ds + 2))
    done
    printf '%s\n' "$$" > "$VENV_LOCK/pid"
    LOCK_HELD=true
    return 0
}

# Release the lock if this process holds it (idempotent; safe as an EXIT trap).
release_lock() {
    if [[ "$LOCK_HELD" == true ]]; then
        rm -rf "$VENV_LOCK"
        LOCK_HELD=false
    fi
}

# The mutating create+install section, run only while holding the lock.
provision_locked() {
    local fresh=false drifted=false
    # Re-check under the lock: a sibling lane may have already provisioned to
    # matching constraints while we waited, so converge instead of thrashing.
    if [[ -x "$VENV/bin/python" ]] && drift_check 2>/dev/null; then
        return 0
    fi
    if [[ ! -x "$VENV/bin/python" ]]; then
        # The one intentionally-ambient interpreter in the toolchain (issue
        # #366): this is what CREATES the pinned venv, so there is nothing
        # pinned to resolve against yet. Everything installed into it below is
        # pinned by constraints-quality.txt, and every gate afterwards resolves
        # through scripts/toolchain-env.sh rather than PATH.
        python3 -m venv "$VENV"
        fresh=true
    fi
    if ! drift_check 2>/dev/null; then
        drifted=true
    fi
    if [[ "$fresh" == true || "$drifted" == true ]]; then
        # Install from the invoking checkout's project root so the constraints
        # and requirements are exactly those relative paths. NEVER install
        # editable (-e): the shared venv must not bind to one lane's checkout.
        (
            cd "$PROJECT_ROOT" || exit 1
            # Mirror ci.yml's `--upgrade pip setuptools wheel` so pip-audit sees
            # the same env as CI (venv-bundled pip can carry flagged CVEs).
            "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
            "$VENV/bin/python" -m pip install \
                -c constraints-quality.txt \
                -r requirements.txt \
                -r requirements-dev.txt
        )
    fi
}

# Provision the shared venv: create it when absent, then install the pinned
# toolchain only when the venv was just created or has drifted (idempotent --
# a clean, matching venv exits 0 without invoking pip install). The mutating
# work is serialized across fleet lanes by an advisory lock.
provision() {
    # Fast path (lock-free): a clean, matching venv needs no mutation, and the
    # check is read-only, so most invocations avoid the lock entirely.
    if [[ -x "$VENV/bin/python" ]] && drift_check 2>/dev/null; then
        return 0
    fi
    acquire_lock || return 1
    # Release the lock on ANY exit from here on, including a set -e abort inside
    # provision_locked (e.g. pip install failing), so a crash can't wedge it.
    trap release_lock EXIT
    provision_locked
    release_lock
    trap - EXIT
}

case "$MODE" in
    print-venv)
        printf '%s\n' "$VENV"
        exit 0
        ;;
    check)
        if drift_check; then
            exit 0
        fi
        exit 1
        ;;
    provision)
        provision
        exit 0
        ;;
esac
