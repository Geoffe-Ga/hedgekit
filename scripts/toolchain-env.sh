#!/usr/bin/env bash
# scripts/toolchain-env.sh - Single authority for resolving quality-gate tools
# Usage: ./scripts/toolchain-env.sh [--print-venv | --print-python | --print-tool NAME | --help]
#
# The gate scripts call this as a subprocess -- e.g.
#   RUFF="$(bash "$SCRIPT_DIR/toolchain-env.sh" --print-tool ruff)"
# -- rather than sourcing it. A failed resolution then exits nonzero and, under
# the gates' `set -e`, aborts the gate with the explanation on stderr: the fail
# closed behaviour comes for free instead of needing every caller to remember a
# guard. (It also keeps shellcheck able to analyse each script standalone, which
# a sourced library would defeat without a suppression directive.)
#
# WHY THIS EXISTS (issue #366)
#
# Every gate script used to invoke its tool by bare name (`pip-audit`, `mypy`,
# `ruff`, ...), so the binary that actually ran -- and therefore the gate's
# verdict -- was decided by whatever happened to be first on the caller's PATH.
# `./scripts/check-all.sh` consequently passed or failed depending on the shell
# it was launched from.
#
# For pip-audit that is not cosmetic. pip-audit audits the dependency set of the
# interpreter that owns it, so a PATH-resolved /opt/homebrew/bin/pip-audit
# audits Homebrew's site-packages, not this project's. The noisy direction of
# that mistake is a CVE reported against dependencies the repo does not control.
# The dangerous direction is the same mistake pointed the other way: an audit
# aimed at the wrong environment can report CLEAN while the project's real
# dependency set is vulnerable -- a false negative in the one check whose entire
# job is catching those.
#
# So resolution is explicit here, and it FAILS CLOSED: when the shared pinned
# .venv (issue #133) exists, a tool comes from it and only from it. A tool
# missing from the pinned venv vetoes rather than silently falling back to an
# alien binary, because a verdict produced by an unknown environment is not
# evidence.
#
# STILL-INHERITED SUB-CHECK (issue #366 acceptance criteria)
#
#   scripts/architecture.sh -> plans/architecture/run-check.sh -> `lint-imports`
#
# is deliberately left resolving by bare name. `tests/architecture/
# test_import_linter_gate.py` pins a contract that running architecture.sh with
# `lint-imports` unreachable on PATH must fail loudly; anchoring that lookup to
# the pinned venv would make the tool always reachable and silently defeat that
# test's premise. The exposure is bounded: import-linter's verdict is computed
# from this repo's own source and `plans/architecture/.importlinter`, not from
# the contents of an external environment, and Gate 1 still runs it with the
# pinned .venv first on PATH (check-all.sh calls toolchain_init before dispatch).
# Version drift of that binary is separately caught by provision-venv.sh --check.

# Absolute directory of THIS file, so the helper works when sourced from any
# working directory and resolves the venv against the checkout that owns it.
TOOLCHAIN_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolution results. Deliberately NOT exported: a gate must never inherit
# another process's idea of where its toolchain lives. Every script re-resolves
# from its own checkout, so a caller cannot redirect a gate by exporting a
# variable -- that would just recreate issue #366 wearing a different hat.
TOOLCHAIN_VENV=""
TOOLCHAIN_BIN=""
TOOLCHAIN_PYTHON=""
TOOLCHAIN_RESOLVED=false

# Resolve the pinned toolchain for this process (idempotent within one shell).
#
# Sets TOOLCHAIN_VENV (empty when there is no shared venv), TOOLCHAIN_BIN and
# TOOLCHAIN_PYTHON, and puts the pinned bin directory first on PATH so tools
# that spawn their own subprocesses (pre-commit hooks, pytest plugins) also see
# the pinned toolchain.
#
# Returns 0 on success, 2 when no interpreter can be found at all.
toolchain_init() {
    if [[ "${TOOLCHAIN_RESOLVED:-false}" == true ]]; then
        return 0
    fi

    local venv
    venv="$(bash "$TOOLCHAIN_ENV_DIR/provision-venv.sh" --print-venv 2>/dev/null || true)"

    if [[ -n "$venv" && -x "$venv/bin/python" ]]; then
        TOOLCHAIN_VENV="$venv"
        TOOLCHAIN_BIN="$venv/bin"
        TOOLCHAIN_PYTHON="$venv/bin/python"
        # Prepend once: check-all.sh initializes, then each sub-script does too,
        # and repeating the entry would just grow PATH for no benefit.
        if [[ "$PATH" != "$TOOLCHAIN_BIN" && "$PATH" != "$TOOLCHAIN_BIN":* ]]; then
            export PATH="$TOOLCHAIN_BIN:$PATH"
        fi
        TOOLCHAIN_RESOLVED=true
        return 0
    fi

    # No shared venv. This is how CI runs: the toolchain is installed straight
    # into the runner's interpreter, so PATH *is* the correct signal there. It
    # is allowed, but never silent -- an unattributable verdict is the failure
    # mode this whole file exists to prevent.
    local ambient
    ambient="$(command -v python3 2>/dev/null || true)"
    if [[ -z "$ambient" ]]; then
        echo "✗ cannot resolve a Python toolchain" >&2
        echo "  why: there is no shared .venv and no python3 on PATH, so no" >&2
        echo "       quality gate can name the environment it would check." >&2
        echo "  next: run scripts/provision-venv.sh to create the shared .venv." >&2
        return 2
    fi
    TOOLCHAIN_VENV=""
    TOOLCHAIN_PYTHON="$ambient"
    TOOLCHAIN_BIN="$(cd "$(dirname "$ambient")" && pwd)"
    echo "Note: no shared .venv; resolving from the ambient environment at" \
        "$TOOLCHAIN_PYTHON (run scripts/provision-venv.sh to pin the toolchain)." >&2
    TOOLCHAIN_RESOLVED=true
    return 0
}

# Print the absolute path of the pinned interpreter.
#
# Callers use this to run tools as modules (`"$(toolchain_python)" -m pip_audit`)
# when the interpreter's identity IS the thing being checked: pip-audit audits
# its own interpreter's site-packages, so binding it to this interpreter makes
# "which environment was audited" a construction guarantee, not a PATH accident.
toolchain_python() {
    toolchain_init || return $?
    printf '%s\n' "$TOOLCHAIN_PYTHON"
}

# Print the absolute path of a pinned tool, or veto.
#
# Args:
#   $1: tool name, e.g. "bandit".
#
# Returns 0 and prints the absolute path on stdout; returns 2 with an
# explanation on stderr when the tool cannot be resolved from the pinned
# environment. Never falls back to PATH while a pinned venv exists.
toolchain_tool() {
    local tool="${1:-}"
    if [[ -z "$tool" ]]; then
        echo "✗ toolchain_tool: no tool name given" >&2
        return 2
    fi

    toolchain_init || return $?

    if [[ -n "$TOOLCHAIN_VENV" ]]; then
        if [[ ! -x "$TOOLCHAIN_BIN/$tool" ]]; then
            echo "✗ $tool is not installed in the pinned toolchain" >&2
            echo "  where: $TOOLCHAIN_BIN" >&2
            echo "  why: a gate must run the PINNED tool. Falling back to whatever" >&2
            echo "       PATH offers would let an unrelated environment decide the" >&2
            echo "       verdict (issue #366), so this vetoes instead of guessing." >&2
            echo "  next: run scripts/provision-venv.sh to refresh the shared .venv." >&2
            return 2
        fi
        printf '%s\n' "$TOOLCHAIN_BIN/$tool"
        return 0
    fi

    local resolved
    resolved="$(command -v "$tool" 2>/dev/null || true)"
    if [[ -z "$resolved" ]]; then
        echo "✗ $tool not found in the ambient environment" >&2
        echo "  why: there is no shared .venv to take it from, and PATH does not" >&2
        echo "       provide it, so this check cannot be run -- and a check that" >&2
        echo "       cannot run must veto, never report success." >&2
        echo "  next: run scripts/provision-venv.sh, or install the pinned" >&2
        echo "        toolchain with pip install -c constraints-quality.txt" >&2
        echo "        -r requirements-dev.txt" >&2
        return 2
    fi
    echo "Note: $tool resolved from the ambient environment: $resolved" >&2
    printf '%s\n' "$resolved"
    return 0
}

# CLI entry point, used by operators to ask which binary a gate WOULD run
# without running the gate, and by tests/toolchain/test_gate_tool_resolution.py
# to prove a hostile PATH cannot redirect a gate.
toolchain_env_main() {
    set -euo pipefail

    case "${1:-}" in
        --print-venv)
            toolchain_init
            printf '%s\n' "$TOOLCHAIN_VENV"
            ;;
        --print-python)
            toolchain_python
            ;;
        --print-tool)
            if [[ $# -lt 2 ]]; then
                echo "Error: --print-tool requires a tool name" >&2
                exit 2
            fi
            toolchain_tool "$2"
            ;;
        --help)
            cat << EOF
Usage: $(basename "$0") [OPTIONS]

Resolve quality-gate tools from the PINNED toolchain instead of the caller's
PATH, so a gate's verdict is a property of this repository rather than of the
shell it was launched from (issue #366).

OPTIONS:
    --print-venv        Print the shared pinned .venv path (empty when absent).
    --print-python      Print the pinned interpreter's absolute path.
    --print-tool NAME   Print the absolute path the gates would run for NAME.
    --help              Display this help message.

EXIT CODES:
    0           Resolved successfully
    2           Could not resolve from the pinned toolchain (fail closed)

EXAMPLES:
    $(basename "$0") --print-tool pip-audit   # which pip-audit would Gate 1 run?
    $(basename "$0") --print-python           # which environment gets audited?
EOF
            ;;
        *)
            echo "Error: Unknown option: ${1:-}" >&2
            exit 2
            ;;
    esac
}

# Executed directly (not sourced) -> behave as the CLI above.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    toolchain_env_main "$@"
fi
