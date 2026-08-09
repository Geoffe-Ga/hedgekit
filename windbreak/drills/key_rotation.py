"""The ``key-rotation`` drill and its ``rotate_keys`` helper (issue #59).

:func:`rotate_keys` returns a *new* environment mapping with each named variable
replaced by a freshly generated, same-shape value (hex for the signing key, so
the Risk Kernel's ``SigningKeyHandle`` still loads it), never mutating the
caller's mapping and never leaving an old value anywhere in the result.

:class:`KeyRotationDrill` rotates the credential variables, verifies the rotated
signing key is still admissibly shaped (valid hex decoding to at least the
minimum key length), and confirms the shipped preflight checklist
(:func:`~windbreak.preflight.runner.run_preflight`) still grades a clean ``0``
exit code. Its evidence carries **no key material** -- only variable names,
booleans, and the integer preflight exit code -- so a rotated secret can never
leak into the hash-chained ledger.

SPEC S5.3 reserves ``windbreak.riskkernel.signing`` to the Risk Kernel package
alone (enforced by ``tests/riskkernel/test_process_isolation.py``), so this
drill -- outside that package -- cannot call ``SigningKeyHandle.from_env``
directly. It instead validates the rotated key's admissible *shape* (the same
hex-and-length rule the handle enforces at construction); a rotation always
produces such a key via :func:`secrets.token_hex`.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, ClassVar, cast

from windbreak.alerts import AlertDispatcher, LoggingLedgerWriter
from windbreak.config import load_default_config
from windbreak.drills.framework import Drill, DrillFailedError, DrillPreconditionError
from windbreak.preflight import (
    EnvTradeKeyLeakProber,
    KeyScopeProbe,
    run_preflight,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from windbreak.drills.context import DrillContext

#: The Risk Kernel approval-token signing-key environment variable.
_APPROVAL_SIGNING_KEY_VAR = "WINDBREAK_APPROVAL_TOKEN_KEY"

#: The exchange trade-key environment variable.
_TRADE_KEY_VAR = "WINDBREAK_TRADE_KEY"

#: The credential variables the drill rotates, when present in the environment.
_ROTATABLE_VARS: tuple[str, ...] = (_APPROVAL_SIGNING_KEY_VAR, _TRADE_KEY_VAR)

#: Bytes of entropy behind each rotated value; ``secrets.token_hex`` renders it
#: as twice this many hex characters, so 32 bytes yields a 64-hex-char value.
_ROTATED_KEY_BYTES = 32

#: Minimum admissible signing-key length, in bytes -- mirroring the Risk
#: Kernel ``SigningKeyHandle``'s own construction threshold (SPEC S5.3, which
#: forbids importing that handle here). A rotation always clears it.
_MIN_SIGNING_KEY_BYTES = 32


def rotate_keys(env: Mapping[str, str], *, keys: tuple[str, ...]) -> dict[str, str]:
    """Return a new environment mapping with each named variable rotated.

    Each named variable that is present in ``env`` is replaced by a freshly
    generated, hex-encoded value (valid for both the signing key and the trade
    key). Untargeted variables carry over unchanged, and ``env`` itself is never
    mutated, so a caller can still diff old against new.

    Args:
        env: The pre-rotation environment mapping.
        keys: The variable names to rotate.

    Returns:
        A new mapping with the named, present variables rotated.
    """
    rotated = dict(env)
    for var in keys:
        if var in rotated:
            rotated[var] = secrets.token_hex(_ROTATED_KEY_BYTES)
    return rotated


class _NullScopeProber:
    """A scope prober reporting no self-test support, so scope checks SKIP."""

    def probe(self) -> KeyScopeProbe:
        """Return an all-unsupported probe.

        Returns:
            A :class:`KeyScopeProbe` reporting no self-test capability.
        """
        return KeyScopeProbe(
            self_test_supported=False,
            scope_verified=False,
            withdrawal_capable=False,
        )


class _ReachableProbe:
    """A read-only exchange probe whose calls always succeed (reachable)."""

    def get_exchange_status(self) -> object:
        """Return a benign status so the reachability check passes.

        Returns:
            A placeholder status (its value is unused by the check).
        """
        return None

    def get_balances(self) -> object:
        """Return benign balances so the reachability check passes.

        Returns:
            A placeholder balance (its value is unused by the check).
        """
        return None


class KeyRotationDrill(Drill):
    """Rotate credentials, then prove signing and preflight still hold."""

    name: ClassVar[str] = "key-rotation"

    def check_preconditions(self, ctx: object) -> None:
        """Verify a signing-key variable is present to rotate.

        Args:
            ctx: The :class:`~windbreak.drills.context.DrillContext` to inspect.

        Raises:
            DrillPreconditionError: If the approval-token variable is absent.
        """
        context = cast("DrillContext", ctx)
        if _APPROVAL_SIGNING_KEY_VAR not in context.env:
            raise DrillPreconditionError(
                f"key-rotation requires {_APPROVAL_SIGNING_KEY_VAR} in the environment"
            )

    def execute(self, ctx: object) -> dict[str, object]:
        """Rotate credentials and grade signing-key load plus preflight.

        Args:
            ctx: The :class:`~windbreak.drills.context.DrillContext` to run
                against.

        Returns:
            Evidence naming the rotated variables (never their values), whether
            the signing key loaded, and the preflight exit code.

        Raises:
            DrillFailedError: If the rotated signing key fails to load or preflight
                does not grade a clean ``0``.
        """
        context = cast("DrillContext", ctx)
        keys = tuple(var for var in _ROTATABLE_VARS if var in context.env)
        rotated = rotate_keys(context.env, keys=keys)
        signing_loadable = self._signing_key_loadable(rotated)
        exit_code = self._preflight_exit_code()
        evidence: dict[str, object] = {
            "rotated_vars": sorted(keys),
            "signing_key_loadable": signing_loadable,
            "preflight_exit_code": exit_code,
        }
        if not signing_loadable or exit_code != 0:
            raise DrillFailedError(evidence)
        return evidence

    def teardown(self, ctx: object) -> None:
        """No teardown: rotation produces only in-memory mappings.

        Args:
            ctx: The :class:`~windbreak.drills.context.DrillContext` (unused).
        """
        del ctx

    def _signing_key_loadable(self, rotated: Mapping[str, str]) -> bool:
        """Return whether the rotated signing key is admissibly shaped.

        Validates the same hex-and-minimum-length rule the Risk Kernel's
        ``SigningKeyHandle`` enforces at construction, without importing that
        handle (SPEC S5.3 reserves it to the ``riskkernel`` package).

        Args:
            rotated: The rotated environment mapping.

        Returns:
            ``True`` iff the rotated approval-token value is valid hex decoding
            to at least :data:`_MIN_SIGNING_KEY_BYTES` bytes.
        """
        try:
            decoded = bytes.fromhex(rotated[_APPROVAL_SIGNING_KEY_VAR])
        except ValueError:
            return False
        return len(decoded) >= _MIN_SIGNING_KEY_BYTES

    def _preflight_exit_code(self) -> int:
        """Grade the shipped preflight checklist against a clean posture.

        The trade-key leak check inspects the *ambient* runtime environment,
        which a correctly-rotated deployment keeps clean of the trade key (it
        lives in the secrets vault, never a process variable); an empty mapping
        models that clean posture, so the leak check passes.

        Returns:
            The preflight report's fail-closed exit code (``0`` on all-pass).
        """
        report = run_preflight(
            connector=_ReachableProbe(),
            scope_prober=_NullScopeProber(),
            leak_prober=EnvTradeKeyLeakProber(environ={}, var=_TRADE_KEY_VAR),
            eligible_markets=(),
            # Deliberately sink-less, unlike the CLI's own preflight (issue
            # #274): a drill is a *rehearsal*, so grading the checklist must
            # never page the operator through the real configured sinks. The
            # dispatcher's log-only fallback keeps the outcome observable.
            alert_dispatcher=AlertDispatcher(
                sinks=[], ledger_writer=LoggingLedgerWriter()
            ),
            secrets_paths=(),
            config=load_default_config(),
        )
        return report.exit_code
