"""The provider package's public re-export surface (issue #269, follow-up 1).

PR #268 landed the :class:`~windbreak.forecast.providers.base.ProviderVoteError`
failure taxonomy, the deterministic
:class:`~windbreak.forecast.providers.retry.RetryingProvider`/
:class:`~windbreak.forecast.providers.retry.RetryPolicy` pair, and the
fail-closed :class:`~windbreak.forecast.budget.ProviderPriceTable` in their own
submodules, leaving every caller to deep-import a module path. These tests pin
the flat ``windbreak.forecast.providers`` surface instead: the whole hardening
API is importable from the package, and each re-export *is* the very object the
defining module holds (identity, not a same-named copy), so a caller catching
``providers.ProviderTimeoutError`` catches what ``retry`` actually raises.
"""

from __future__ import annotations

import windbreak.forecast.providers as providers_pkg
from windbreak.forecast import budget as budget_module
from windbreak.forecast.providers import base as base_module
from windbreak.forecast.providers import retry as retry_module

#: Every name issue #269 follow-up 1 requires on the flat package surface,
#: mapped to the module that defines it -- the identity each re-export must
#: preserve.
_REQUIRED_REEXPORTS: dict[str, object] = {
    "ProviderVoteError": base_module,
    "ProviderTimeoutError": base_module,
    "ProviderRateLimitedError": base_module,
    "ProviderHTTPError": base_module,
    "ProviderMalformedResponseError": base_module,
    "ProviderCostOverrunError": base_module,
    "RetryingProvider": retry_module,
    "RetryPolicy": retry_module,
    "is_retryable_status": retry_module,
    "ProviderPriceTable": budget_module,
    "DEFAULT_PROVIDER_PRICE_TABLE": budget_module,
}


def test_every_hardening_name_is_importable_from_the_flat_package() -> None:
    """Each taxonomy/retry/pricing name is reachable off the package itself."""
    missing = [name for name in _REQUIRED_REEXPORTS if not hasattr(providers_pkg, name)]
    assert missing == []


def test_every_reexport_is_the_defining_modules_own_object() -> None:
    """Re-export identity holds, so an ``except`` on either spelling matches."""
    for name, module in _REQUIRED_REEXPORTS.items():
        assert getattr(providers_pkg, name) is getattr(module, name), name


def test_every_hardening_name_is_declared_in_dunder_all() -> None:
    """``__all__`` names them, so ``from ... import *`` and linters agree."""
    undeclared = [
        name for name in _REQUIRED_REEXPORTS if name not in providers_pkg.__all__
    ]
    assert undeclared == []
