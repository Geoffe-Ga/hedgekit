"""Failing-first tests for windbreak.riskkernel.checks (issues #30, #31, #32,
#34, #110, RED).

Issue #30 gave 15 of the 24 SPEC S10.3 pre-trade checks their real logic
(instrument whitelist, mode/ceiling, the floor invariant, fee-bound presence,
concentration, daily loss, trailing drawdown, velocity, quote/forecast
freshness, price band, participation cap, clock skew, and reduce-only
provability), each reading a full `EvaluationContext` rather than the
`OrderIntent` alone. Issue #31 promotes two more from stub to real logic --
`approval_token_uniqueness` and `idempotency_key_uniqueness`, each reading a
new `EvaluationContext.used_intent_ids` / `.used_idempotency_keys` set. Issue
#32 promotes three more -- `balance_reconciliation`, `position_reconciliation`,
and `open_order_reconciliation` -- each reading a new
`EvaluationContext.verification: VerificationSnapshot | None` (fail-closed via
the existing `_is_stale` pattern on a missing/future/stale snapshot, plus,
for `balance_reconciliation` only, a live-mode veto when
`verification.semantics_fully_known` is `False`). Issue #34 promotes one
more -- `human_ack_satisfied` -- reading a new
`RiskLimits.require_human_ack_above_micros: MoneyMicros | None` threshold
against the order's own worst-case cost and a new
`EvaluationContext.acknowledged_intent_ids: frozenset[str]` set: a `None`
threshold (or a cost at or below a configured one) always approves; a cost
strictly above a configured threshold approves only if the intent id is
already acknowledged, and otherwise vetoes; an unprovable cost (missing fee
bound) vetoes fail-closed, exactly like every other cost-consuming check.
Issue #110 promotes the final two -- `exchange_status_ok` and
`pipeline_heartbeat_ok` -- reading a new `MarketView.exchange_status:
ExchangeTradingStatus | None` / `.exchange_status_epoch_s: int | None` pair and
a new `EvaluationContext.pipeline_heartbeat_epoch_s: int | None` (no default,
mirroring the `verification` fail-loud precedent): `exchange_status_ok` fails
closed (staleness-first) on a missing or stale status via the existing
`_is_stale` pattern, then vetoes a fresh but non-`OPEN` status (`PAUSED` /
`CLOSED`); `pipeline_heartbeat_ok` fails closed on a missing or stale
heartbeat the same way. Issue #340 promoted the last one,
`jurisdiction_product_eligibility`, so all 24 SPEC S10.3 checks are now real and
none is an unconditional veto.

`windbreak/riskkernel/context.py` does not yet declare `used_intent_ids` /
`used_idempotency_keys` / `verification` / `acknowledged_intent_ids` /
`require_human_ack_above_micros` / `ExchangeTradingStatus` /
`pipeline_heartbeat_epoch_s`, and `windbreak/riskkernel/verification.py` does
not exist at all yet (this file's `conftest` import alone triggers the
collection failure), so importing anything here fails collection with
`ModuleNotFoundError`/`ImportError`/`TypeError` -- the expected Gate 1 RED
state for issues #31, #32, #34, and #110. Once `context.py`, `verification.py`,
and the eight real checks exist, this file pins: the exact boundary of every
real check (the precise value that passes vs. the one unit past it that
vetoes); that every `None` optional input (fee bounds, timestamps, depth, open
position, verification snapshot, human-ack threshold, exchange status,
pipeline heartbeat) vetoes or approves as documented; that an unknown `action`
vetoes every check that branches on open/close; the unchanged 24-name SPEC
S10.3 order; that a fully-permissive context leaves *only* the 1 stub check
vetoing, naming the metadata it awaits; the fail-closed error-conversion
contract; and the frozen result types.

This file supersedes and deletes `tests/riskkernel/test_checks_stub.py`:
every behavior that file pinned (frozen `OrderIntent`/`CheckResult`/
`Decision`, the 24-name sequence, fail-closed error conversion, veto/pass
aggregation) is re-pinned here against the new `(intent, context)` call
signature, so the older, `OrderIntent`-only-signature file would otherwise
merely duplicate (and, for its 24-checks-all-veto assumption, contradict) it.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

from tests.riskkernel.conftest import (
    DEFAULT_MARKET_TICKER,
    DEFAULT_NOW_EPOCH_S,
    make_context,
    make_intent,
    make_verification_snapshot,
)
from windbreak.numeric.types import (
    ContractCentis,
    MoneyMicros,
    PricePips,
    ProbabilityPpm,
)
from windbreak.riskkernel import checks as checks_module
from windbreak.riskkernel.checks import (
    DEFAULT_CHECKS,
    CheckResult,
    evaluate_intent,
)
from windbreak.riskkernel.context import (
    ExchangeTradingStatus,
    JurisdictionStatus,
    ProductType,
)
from windbreak.riskkernel.modes import Mode

if TYPE_CHECKING:
    from windbreak.riskkernel.checks import Check

#: The exact SPEC S10.3 check-name sequence, in this exact order -- unchanged
#: by issue #30 (only each check's internals and arity change).
EXPECTED_CHECK_NAMES: tuple[str, ...] = (
    "instrument_whitelist",
    "jurisdiction_product_eligibility",
    "mode_permission_ceiling",
    "floor_invariant",
    "balance_reconciliation",
    "position_reconciliation",
    "open_order_reconciliation",
    "fee_upper_bound_present",
    "settlement_fee_upper_bound",
    "concentration_limits",
    "daily_loss_limit",
    "trailing_drawdown_limit",
    "velocity_limits",
    "quote_freshness",
    "forecast_freshness",
    "price_band_compliance",
    "participation_cap_compliance",
    "human_ack_satisfied",
    "approval_token_uniqueness",
    "idempotency_key_uniqueness",
    "clock_skew_limit",
    "exchange_status_ok",
    "pipeline_heartbeat_ok",
    "reduce_only_provable",
)

_EXPECTED_CHECK_COUNT = 24

#: All 24 checks, now real, in SPEC S10.3 order. Issue #340 promoted the last
#: one -- `jurisdiction_product_eligibility` -- so this tuple is now identical
#: to `EXPECTED_CHECK_NAMES`; the two are kept separate deliberately, because
#: `EXPECTED_CHECK_NAMES` is the guardrail pinning the *order* while this one
#: names what is backed by real logic.
REAL_CHECK_NAMES: tuple[str, ...] = (
    "instrument_whitelist",
    "jurisdiction_product_eligibility",
    "mode_permission_ceiling",
    "floor_invariant",
    "balance_reconciliation",
    "position_reconciliation",
    "open_order_reconciliation",
    "fee_upper_bound_present",
    "settlement_fee_upper_bound",
    "concentration_limits",
    "daily_loss_limit",
    "trailing_drawdown_limit",
    "velocity_limits",
    "quote_freshness",
    "forecast_freshness",
    "price_band_compliance",
    "participation_cap_compliance",
    "human_ack_satisfied",
    "approval_token_uniqueness",
    "idempotency_key_uniqueness",
    "clock_skew_limit",
    "exchange_status_ok",
    "pipeline_heartbeat_ok",
    "reduce_only_provable",
)

#: Every check callable, keyed by its own `.name` -- built from each check's
#: self-reported name (not by zipping against `EXPECTED_CHECK_NAMES`), so a
#: per-check test's result is independent of the separately-pinned ordering
#: test below.
_CHECK_BY_NAME: dict[str, Check] = {check.name: check for check in DEFAULT_CHECKS}


def _real_check(name: str) -> Check:
    """Look up one of the 24 checks (real or stub) by name, for direct
    invocation.

    Args:
        name: The check's SPEC S10.3 name.

    Returns:
        The check callable from `DEFAULT_CHECKS`.
    """
    return _CHECK_BY_NAME[name]


class _RaisingCheck:
    """A check double that raises instead of returning a `CheckResult`."""

    name = "raising_check"

    def __call__(self, intent: object, context: object) -> CheckResult:
        """Raise unconditionally, ignoring both arguments.

        Args:
            intent: Unused; accepted only to match the check-callable shape.
            context: Unused; accepted only to match the check-callable shape.

        Raises:
            RuntimeError: Always, with a fixed, recognizable message.
        """
        raise RuntimeError("boom")


class _PassingCheck:
    """A check double that approves (does not veto) the intent."""

    name = "passing_check"

    def __call__(self, intent: object, context: object) -> CheckResult:
        """Approve unconditionally, ignoring both arguments.

        Args:
            intent: Unused; accepted only to match the check-callable shape.
            context: Unused; accepted only to match the check-callable shape.

        Returns:
            A non-vetoing `CheckResult`.
        """
        return CheckResult(vetoed=False, reason="approved")


class _VetoingCheck:
    """A check double that always vetoes, for pipeline-level aggregation tests."""

    name = "vetoing_check"

    def __call__(self, intent: object, context: object) -> CheckResult:
        """Veto unconditionally.

        Args:
            intent: Unused; accepted only to match the check-callable shape.
            context: Unused; accepted only to match the check-callable shape.

        Returns:
            A vetoing :class:`CheckResult`.
        """
        del intent, context
        return CheckResult(vetoed=True, reason="vetoed")


# --- jurisdiction_product_eligibility (issue #340) ---------------------------


def test_jurisdiction_product_eligibility_approves_an_eligible_binary() -> None:
    """An eligible jurisdiction on a fully collateralized binary approves."""
    result = _real_check("jurisdiction_product_eligibility")(
        make_intent(),
        make_context(
            jurisdiction_status=JurisdictionStatus.ELIGIBLE,
            product_type=ProductType.FULLY_COLLATERALIZED_BINARY,
        ),
    )

    assert result.vetoed is False


def test_jurisdiction_product_eligibility_vetoes_an_ineligible_jurisdiction() -> None:
    """A market proven ineligible is vetoed, with a reason naming why."""
    result = _real_check("jurisdiction_product_eligibility")(
        make_intent(), make_context(jurisdiction_status=JurisdictionStatus.INELIGIBLE)
    )

    assert result.vetoed is True
    assert result.reason == "market not jurisdiction-eligible"


def test_jurisdiction_product_eligibility_fails_closed_on_unknown_jurisdiction() -> (
    None
):
    """An unknown or absent jurisdiction vetoes rather than being read as eligible.

    This is the fail-closed arm (SPEC S1.1 invariant 1) and the one that matters
    in production: a live Kalshi market carries no eligibility signal at all, so
    it arrives here as `None`.
    """
    result = _real_check("jurisdiction_product_eligibility")(
        make_intent(), make_context(jurisdiction_status=None)
    )

    assert result.vetoed is True
    assert result.reason == "jurisdiction eligibility unknown or missing"


def test_jurisdiction_product_eligibility_fails_closed_on_unprovable_product() -> None:
    """An absent product type vetoes even when the jurisdiction is eligible.

    Ordering matters: the jurisdiction arm is satisfied here, so this can only
    pass if the product arm is genuinely evaluated (SPEC S1.1 invariant 2).
    """
    result = _real_check("jurisdiction_product_eligibility")(
        make_intent(),
        make_context(
            jurisdiction_status=JurisdictionStatus.ELIGIBLE, product_type=None
        ),
    )

    assert result.vetoed is True
    assert result.reason == "product not provably a fully collateralized binary"


# --- Sanity on the fixtures themselves -------------------------------------------


def test_every_pinned_check_name_is_backed_by_a_real_check() -> None:
    """No check in the pinned SPEC S10.3 sequence is an unconditional veto.

    Issue #340 promoted the last stub, so the real-name set is now exactly the
    pinned set. The `hasattr` guards are deliberately attribute-based: they fail
    if the stub machinery is left behind as an emptied table rather than
    deleted, which would otherwise pass every other assertion here.
    """
    assert REAL_CHECK_NAMES == EXPECTED_CHECK_NAMES
    assert len(REAL_CHECK_NAMES) == _EXPECTED_CHECK_COUNT
    for dead in (
        "_ExplicitVetoStub",
        "_STUB_REASON_ITEMS",
        "_STUB_REASONS",
        "_STUB_CHECKS",
    ):
        assert not hasattr(checks_module, dead), f"{dead} survived the stub deletion"


def test_make_context_defaults_make_every_real_check_pass() -> None:
    """`make_context()` paired with `make_intent()` passes all 24 real
    checks -- the load-bearing fixture assumption every per-check boundary
    test below relies on to isolate its one overridden field.
    """
    intent = make_intent()
    context = make_context()

    for name in REAL_CHECK_NAMES:
        result = _real_check(name)(intent, context)
        assert result.vetoed is False, f"{name} unexpectedly vetoed: {result.reason}"


# --- instrument_whitelist ---------------------------------------------------------


def test_instrument_whitelist_passes_when_ticker_is_listed() -> None:
    """A ticker present in the whitelist passes."""
    result = _real_check("instrument_whitelist")(make_intent(), make_context())

    assert result.vetoed is False


def test_instrument_whitelist_vetoes_when_ticker_is_absent() -> None:
    """A ticker not in the whitelist vetoes."""
    context = make_context(instrument_whitelist=frozenset({"OTHER-TICKER"}))

    result = _real_check("instrument_whitelist")(make_intent(), context)

    assert result.vetoed is True


def test_instrument_whitelist_vetoes_everything_when_empty() -> None:
    """An empty whitelist vetoes every ticker -- the boundary an
    accidentally-permissive "empty means allow all" mutant would fail.
    """
    context = make_context(instrument_whitelist=frozenset())

    result = _real_check("instrument_whitelist")(make_intent(), context)

    assert result.vetoed is True


# --- mode_permission_ceiling -------------------------------------------------------


@pytest.mark.parametrize("mode", [Mode.PAPER, Mode.LIVE_MICRO, Mode.LIVE])
def test_mode_permission_ceiling_passes_for_each_permitted_mode(mode: Mode) -> None:
    """PAPER, LIVE_MICRO, and LIVE are each permitted to trade."""
    context = make_context(mode=mode, micro_cap=MoneyMicros(1_000_000_000_000))

    result = _real_check("mode_permission_ceiling")(make_intent(), context)

    assert result.vetoed is False


@pytest.mark.parametrize("mode", [Mode.RESEARCH, Mode.PAUSED, Mode.HALT, Mode.KILLED])
def test_mode_permission_ceiling_vetoes_for_each_unpermitted_mode(mode: Mode) -> None:
    """RESEARCH and every safety mode are not permitted to trade."""
    context = make_context(mode=mode)

    result = _real_check("mode_permission_ceiling")(make_intent(), context)

    assert result.vetoed is True


def test_mode_permission_ceiling_live_micro_passes_at_exact_micro_cap() -> None:
    """In LIVE_MICRO, `total_exposure + worst_case_cost == micro_cap` passes
    (the cap is inclusive): default intent cost is 5_000_000 micros.
    """
    context = make_context(
        mode=Mode.LIVE_MICRO,
        total_exposure=MoneyMicros(0),
        micro_cap=MoneyMicros(5_000_000),
    )

    result = _real_check("mode_permission_ceiling")(make_intent(), context)

    assert result.vetoed is False


def test_mode_permission_ceiling_live_micro_vetoes_one_micro_over_cap() -> None:
    """One micro above `micro_cap` vetoes in LIVE_MICRO."""
    context = make_context(
        mode=Mode.LIVE_MICRO,
        total_exposure=MoneyMicros(0),
        micro_cap=MoneyMicros(4_999_999),
    )

    result = _real_check("mode_permission_ceiling")(make_intent(), context)

    assert result.vetoed is True


@pytest.mark.parametrize(
    "fees_override",
    [
        {"max_trading_fee": None},
        {"max_settlement_fee": None},
    ],
    ids=["trading_fee_none", "settlement_fee_none"],
)
def test_mode_permission_ceiling_live_micro_vetoes_when_a_fee_bound_is_none(
    fees_override: dict[str, object],
) -> None:
    """In LIVE_MICRO, a missing (`None`) fee bound of either kind makes the
    exposure-ceiling cost unprovable, so the check vetoes as `"unprovable"`
    (fail-closed) rather than raising or approving -- the guard that replaced a
    `python -O`-strippable `assert`.
    """
    context = make_context(mode=Mode.LIVE_MICRO, **fees_override)

    result = _real_check("mode_permission_ceiling")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "unprovable"


def test_mode_permission_ceiling_live_micro_exempts_provable_derisking_close() -> None:
    """(#100) A provable de-risking `sell_to_close` is exempt from the
    LIVE_MICRO micro-cap term entirely: even with `total_exposure +
    pending_kernel_reservations` already at/over `micro_cap` -- before any
    cost from this order is even considered -- the close still approves,
    because it can only reduce exposure, never add to it.
    """
    context = make_context(
        mode=Mode.LIVE_MICRO,
        micro_cap=MoneyMicros(1_000_000),
        total_exposure=MoneyMicros(1_000_000),
        pending_kernel_reservations=MoneyMicros(500_000),
        open_position=ContractCentis(1000),
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("mode_permission_ceiling")(intent, context)

    assert result.vetoed is False


def test_mode_permission_ceiling_provable_close_approves_with_missing_fee_bound() -> (
    None
):
    """(#100) The LIVE_MICRO exemption short-circuits before `_order_cost` is
    ever called: a provable de-risking close approves even when
    `max_trading_fee` is `None` -- proof that cost is never computed for an
    exempt close, matching the `concentration_limits` and `velocity_limits`
    missing-fee-bound pins. (The pipeline still fails closed on the missing
    bound via the independent `fee_upper_bound_present` / `floor_invariant`
    checks; this pins only that the exemption itself never consumes cost.)
    """
    context = make_context(
        mode=Mode.LIVE_MICRO,
        micro_cap=MoneyMicros(1_000_000),
        total_exposure=MoneyMicros(1_000_000),
        pending_kernel_reservations=MoneyMicros(500_000),
        open_position=ContractCentis(1000),
        max_trading_fee=None,
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("mode_permission_ceiling")(intent, context)

    assert result.vetoed is False


def test_mode_permission_ceiling_vetoes_non_provable_close_over_micro_cap() -> None:
    """The exemption above is narrow: at the identical over-cap exposure, a
    close that is NOT provably reduce-only (no open position on record) gets
    no exemption, is still charged full worst-case notional, and still
    vetoes (#100).
    """
    context = make_context(
        mode=Mode.LIVE_MICRO,
        micro_cap=MoneyMicros(1_000_000),
        total_exposure=MoneyMicros(1_000_000),
        pending_kernel_reservations=MoneyMicros(500_000),
        open_position=None,
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("mode_permission_ceiling")(intent, context)

    assert result.vetoed is True
    assert result.reason == "live-micro exposure ceiling exceeded"


def test_mode_permission_ceiling_mode_gate_is_not_weakened_by_derisking_close() -> None:
    """Security pin (#100): the de-risking exemption applies only to the
    LIVE_MICRO cap arithmetic, never to the mode gate itself. A provable
    de-risking close submitted in a non-trading mode (HALT here) still
    vetoes with the unchanged "mode ... may not trade" reason -- the safety
    valve for exposure must never become a bypass for the trading-mode gate.
    """
    context = make_context(mode=Mode.HALT, open_position=ContractCentis(1000))
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("mode_permission_ceiling")(intent, context)

    assert result.vetoed is True
    assert result.reason == "mode HALT may not trade"


# --- floor_invariant ---------------------------------------------------------------


def test_floor_invariant_open_passes_at_exact_floor_equality() -> None:
    """An open whose `worst_case_equity - worst_case_cost` equals the floor
    exactly passes: default equity 1_000_000_000, default cost 5_000_000.
    """
    context = make_context(floor=MoneyMicros(995_000_000))

    result = _real_check("floor_invariant")(make_intent(action="buy"), context)

    assert result.vetoed is False


def test_floor_invariant_open_vetoes_one_micro_below_floor_equality() -> None:
    """A floor one micro above what the equity affords vetoes the open."""
    context = make_context(floor=MoneyMicros(995_000_001))

    result = _real_check("floor_invariant")(make_intent(action="buy"), context)

    assert result.vetoed is True


def test_floor_invariant_close_ignores_proceeds_and_passes_at_exact_floor() -> None:
    """A provable close's cost is fees + buffer only (price/size proceeds
    are ignored): equity 1_000_000_000 - (300_000 + 150_000 + 75_000) ==
    999_475_000 passes at that exact floor.
    """
    context = make_context(
        floor=MoneyMicros(999_475_000),
        max_trading_fee=MoneyMicros(300_000),
        max_settlement_fee=MoneyMicros(150_000),
        rounding_buffer=MoneyMicros(75_000),
    )
    intent = make_intent(action="sell_to_close", price=PricePips(9_999))

    result = _real_check("floor_invariant")(intent, context)

    assert result.vetoed is False


def test_floor_invariant_provable_close_vetoes_one_micro_below_floor_equality() -> None:
    """The close-side floor boundary is exact too: one micro over vetoes."""
    context = make_context(
        floor=MoneyMicros(999_475_001),
        max_trading_fee=MoneyMicros(300_000),
        max_settlement_fee=MoneyMicros(150_000),
        rounding_buffer=MoneyMicros(75_000),
    )
    intent = make_intent(action="sell_to_close")

    result = _real_check("floor_invariant")(intent, context)

    assert result.vetoed is True


def test_floor_invariant_unknown_action_vetoes_as_unprovable() -> None:
    """An action that is neither an open nor a provable close vetoes as
    `"unprovable"` -- the exact reason SPEC S10.3 names.
    """
    intent = make_intent(action="hold")

    result = _real_check("floor_invariant")(intent, make_context())

    assert result.vetoed is True
    assert result.reason == "unprovable"


@pytest.mark.parametrize(
    "fees_override",
    [
        {"max_trading_fee": None},
        {"max_settlement_fee": None},
    ],
    ids=["trading_fee_none", "settlement_fee_none"],
)
def test_floor_invariant_vetoes_as_unprovable_when_any_fee_bound_is_none(
    fees_override: dict[str, object],
) -> None:
    """A missing (`None`) fee bound of either kind vetoes as `"unprovable"`,
    regardless of action.
    """
    context = make_context(**fees_override)

    result = _real_check("floor_invariant")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "unprovable"


# --- balance_reconciliation / position_reconciliation / open_order_reconciliation
# --- (issue #32) ------------------------------------------------------------------
#
# All three checks share the same fail-closed staleness gate over
# `context.verification` (mirroring `_is_stale`'s None/future/past-ttl trio,
# exactly as `quote_freshness`/`forecast_freshness`/`clock_skew_limit` already
# pin it), plus a per-dimension `*_ok` flag read off the snapshot.
# `balance_reconciliation` alone additionally gates on
# `verification.semantics_fully_known` in live trading modes.

#: (check name, snapshot per-dimension "ok" field, its exact mismatch reason).
_RECONCILIATION_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("balance_reconciliation", "balance_ok", "balance reconciliation mismatch"),
    ("position_reconciliation", "position_ok", "position reconciliation mismatch"),
    (
        "open_order_reconciliation",
        "open_order_ok",
        "open-order reconciliation mismatch",
    ),
)

#: Each reconciliation check's exact stale/missing-snapshot veto reason.
_RECONCILIATION_STALE_REASONS: dict[str, str] = {
    "balance_reconciliation": "balance verification stale or missing",
    "position_reconciliation": "position verification stale or missing",
    "open_order_reconciliation": "open-order verification stale or missing",
}


@pytest.mark.parametrize(
    ("check_name", "ok_field", "_mismatch_reason"),
    _RECONCILIATION_CHECKS,
    ids=[triple[0] for triple in _RECONCILIATION_CHECKS],
)
def test_reconciliation_check_passes_with_a_fresh_ok_snapshot(
    check_name: str, ok_field: str, _mismatch_reason: str
) -> None:
    """Each reconciliation check passes given a fresh snapshot with its own
    dimension marked `ok`."""
    context = make_context(verification=make_verification_snapshot(**{ok_field: True}))

    result = _real_check(check_name)(make_intent(), context)

    assert result.vetoed is False


@pytest.mark.parametrize(
    ("check_name", "ok_field", "mismatch_reason"),
    _RECONCILIATION_CHECKS,
    ids=[triple[0] for triple in _RECONCILIATION_CHECKS],
)
def test_reconciliation_check_vetoes_when_its_dimension_is_not_ok(
    check_name: str, ok_field: str, mismatch_reason: str
) -> None:
    """Each reconciliation check vetoes, with its exact mismatch reason, when
    its own per-dimension `ok` flag is `False` -- and only that flag; the
    other two dimensions stay `True`, proving the check reads its own flag,
    not some other dimension's.
    """
    context = make_context(verification=make_verification_snapshot(**{ok_field: False}))

    result = _real_check(check_name)(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == mismatch_reason


@pytest.mark.parametrize("check_name", [triple[0] for triple in _RECONCILIATION_CHECKS])
def test_reconciliation_check_vetoes_when_verification_is_none(
    check_name: str,
) -> None:
    """A missing (`None`) verification snapshot vetoes every reconciliation
    check -- fail-closed, mirroring the `used_intent_ids` no-default
    precedent: a forgotten verification wiring must veto, never silently pass.
    """
    context = make_context(verification=None)

    result = _real_check(check_name)(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == _RECONCILIATION_STALE_REASONS[check_name]


@pytest.mark.parametrize("check_name", [triple[0] for triple in _RECONCILIATION_CHECKS])
def test_reconciliation_check_vetoes_a_future_dated_snapshot(
    check_name: str,
) -> None:
    """A snapshot timestamped after `now_epoch_s` vetoes, however fresh its
    age would otherwise appear -- matching `quote_freshness`'s future-dated
    boundary."""
    context = make_context(
        verification=make_verification_snapshot(
            verified_at_epoch_s=DEFAULT_NOW_EPOCH_S + 1
        )
    )

    result = _real_check(check_name)(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == _RECONCILIATION_STALE_REASONS[check_name]


@pytest.mark.parametrize("check_name", [triple[0] for triple in _RECONCILIATION_CHECKS])
def test_reconciliation_check_passes_at_exact_ttl_age(check_name: str) -> None:
    """A snapshot exactly `verification_ttl_seconds` old (age == ttl) is still
    fresh -- the same inclusive boundary `quote_freshness` pins."""
    context = make_context(
        verification_ttl_seconds=3_600,
        verification=make_verification_snapshot(
            verified_at_epoch_s=DEFAULT_NOW_EPOCH_S - 3_600
        ),
    )

    result = _real_check(check_name)(make_intent(), context)

    assert result.vetoed is False


@pytest.mark.parametrize("check_name", [triple[0] for triple in _RECONCILIATION_CHECKS])
def test_reconciliation_check_vetoes_one_second_past_ttl(check_name: str) -> None:
    """A snapshot one second past its ttl is stale."""
    context = make_context(
        verification_ttl_seconds=3_600,
        verification=make_verification_snapshot(
            verified_at_epoch_s=DEFAULT_NOW_EPOCH_S - 3_601
        ),
    )

    result = _real_check(check_name)(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == _RECONCILIATION_STALE_REASONS[check_name]


@pytest.mark.parametrize("mode", [Mode.LIVE_MICRO, Mode.LIVE])
def test_balance_reconciliation_vetoes_unknown_semantics_in_live_modes(
    mode: Mode,
) -> None:
    """`balance_reconciliation` alone additionally vetoes, with a distinct
    reason, when `verification.semantics_fully_known` is `False` and the
    kernel is trading in LIVE_MICRO or LIVE -- refusing live trading while any
    `BalanceSemantics` field is unknown (issue #32's live-mode gate).
    """
    context = make_context(
        mode=mode,
        verification=make_verification_snapshot(semantics_fully_known=False),
    )

    result = _real_check("balance_reconciliation")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "balance semantics not fully known in live mode"


@pytest.mark.parametrize("mode", [Mode.PAPER, Mode.RESEARCH])
def test_balance_reconciliation_proceeds_with_unknown_semantics_outside_live_modes(
    mode: Mode,
) -> None:
    """Unknown balance semantics do not veto `balance_reconciliation` in PAPER
    or RESEARCH -- the live-mode gate is scoped to live trading only.
    """
    context = make_context(
        mode=mode,
        verification=make_verification_snapshot(semantics_fully_known=False),
    )

    result = _real_check("balance_reconciliation")(make_intent(), context)

    assert result.vetoed is False


def test_position_reconciliation_ignores_semantics_fully_known() -> None:
    """`position_reconciliation` has no live-mode semantics gate: unknown
    semantics alone, with its own dimension `ok`, never vetoes it -- only
    `balance_reconciliation` reads `semantics_fully_known`.
    """
    context = make_context(
        mode=Mode.LIVE,
        verification=make_verification_snapshot(semantics_fully_known=False),
    )

    result = _real_check("position_reconciliation")(make_intent(), context)

    assert result.vetoed is False


def test_open_order_reconciliation_ignores_semantics_fully_known() -> None:
    """`open_order_reconciliation` has no live-mode semantics gate either."""
    context = make_context(
        mode=Mode.LIVE,
        verification=make_verification_snapshot(semantics_fully_known=False),
    )

    result = _real_check("open_order_reconciliation")(make_intent(), context)

    assert result.vetoed is False


# --- fee_upper_bound_present / settlement_fee_upper_bound -----------------------


def test_fee_upper_bound_present_passes_when_trading_fee_bound_present() -> None:
    """A present (non-`None`) trading fee bound passes, even if the
    settlement bound is `None` -- this check reads only the trading fee.
    """
    context = make_context(max_trading_fee=MoneyMicros(0), max_settlement_fee=None)

    result = _real_check("fee_upper_bound_present")(make_intent(), context)

    assert result.vetoed is False


def test_fee_upper_bound_present_vetoes_when_trading_fee_bound_is_none() -> None:
    """A `None` trading fee bound vetoes."""
    context = make_context(max_trading_fee=None)

    result = _real_check("fee_upper_bound_present")(make_intent(), context)

    assert result.vetoed is True


def test_settlement_fee_upper_bound_passes_when_settlement_fee_bound_present() -> None:
    """A present settlement fee bound passes, even if the trading bound is
    `None` -- this check reads only the settlement fee.
    """
    context = make_context(max_settlement_fee=MoneyMicros(0), max_trading_fee=None)

    result = _real_check("settlement_fee_upper_bound")(make_intent(), context)

    assert result.vetoed is False


def test_settlement_fee_upper_bound_vetoes_when_settlement_fee_bound_is_none() -> None:
    """A `None` settlement fee bound vetoes."""
    context = make_context(max_settlement_fee=None)

    result = _real_check("settlement_fee_upper_bound")(make_intent(), context)

    assert result.vetoed is True


# --- concentration_limits ----------------------------------------------------------

#: Each of the four concentration dimensions: (account exposure field, limits
#: cap-percentage field). At a 10% cap over the default $1,000 equity, the
#: threshold is exactly 100_000_000 micros; less the default 5_000_000-micro
#: cost, the boundary exposure is 95_000_000.
_CONCENTRATION_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("market_exposure", "max_pos_market_pct_ppm"),
    ("event_exposure", "max_pos_event_pct_ppm"),
    ("bucket_exposure", "max_pos_bucket_pct_ppm"),
    ("total_exposure", "max_pos_total_pct_ppm"),
)

_CONCENTRATION_CAP_PPM = 100_000
_CONCENTRATION_BOUNDARY_EXPOSURE_MICROS = 95_000_000


@pytest.mark.parametrize(
    ("exposure_field", "cap_field"),
    _CONCENTRATION_DIMENSIONS,
    ids=[pair[0] for pair in _CONCENTRATION_DIMENSIONS],
)
def test_concentration_limits_passes_at_exact_threshold(
    exposure_field: str, cap_field: str
) -> None:
    """Each dimension passes when `exposure + cost` equals the floored
    percentage-of-equity threshold exactly.
    """
    context = make_context(
        **{
            cap_field: _CONCENTRATION_CAP_PPM,
            exposure_field: MoneyMicros(_CONCENTRATION_BOUNDARY_EXPOSURE_MICROS),
        }
    )

    result = _real_check("concentration_limits")(make_intent(), context)

    assert result.vetoed is False


@pytest.mark.parametrize(
    ("exposure_field", "cap_field"),
    _CONCENTRATION_DIMENSIONS,
    ids=[pair[0] for pair in _CONCENTRATION_DIMENSIONS],
)
def test_concentration_limits_vetoes_one_micro_over_threshold(
    exposure_field: str, cap_field: str
) -> None:
    """Each dimension vetoes one micro past that same threshold."""
    context = make_context(
        **{
            cap_field: _CONCENTRATION_CAP_PPM,
            exposure_field: MoneyMicros(_CONCENTRATION_BOUNDARY_EXPOSURE_MICROS + 1),
        }
    )

    result = _real_check("concentration_limits")(make_intent(), context)

    assert result.vetoed is True


@pytest.mark.parametrize(
    "fees_override",
    [
        {"max_trading_fee": None},
        {"max_settlement_fee": None},
    ],
    ids=["trading_fee_none", "settlement_fee_none"],
)
def test_concentration_limits_vetoes_when_a_fee_bound_is_none(
    fees_override: dict[str, object],
) -> None:
    """A missing (`None`) fee bound of either kind makes the order cost
    unprovable, so `concentration_limits` vetoes as `"unprovable"` (fail-closed)
    instead of raising or approving.
    """
    context = make_context(**fees_override)

    result = _real_check("concentration_limits")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "unprovable"


def test_concentration_limits_exempts_a_provable_derisking_close() -> None:
    """(#100) Reverses the pre-#100 behavior this test used to pin (see git
    history: `test_concentration_limits_charges_a_close_its_full_notional`).
    At the identical one-micro-over-threshold exposure, a provable
    de-risking `sell_to_close` now APPROVES: `concentration_limits` fully
    exempts a provable close instead of charging it full worst-case
    notional, so the safety valve that reduces exposure is never vetoed by
    the cap it is reducing.
    """
    context = make_context(
        max_pos_market_pct_ppm=_CONCENTRATION_CAP_PPM,
        market_exposure=MoneyMicros(_CONCENTRATION_BOUNDARY_EXPOSURE_MICROS + 1),
        open_position=ContractCentis(1000),
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("concentration_limits")(intent, context)

    assert result.vetoed is False


def test_concentration_limits_still_vetoes_non_provable_close_over_threshold() -> None:
    """The exemption above is narrow: at the identical over-threshold
    exposure, a close that is NOT provably reduce-only (no open position on
    record) gets no exemption, is still charged full worst-case notional,
    and still vetoes (#100).
    """
    context = make_context(
        max_pos_market_pct_ppm=_CONCENTRATION_CAP_PPM,
        market_exposure=MoneyMicros(_CONCENTRATION_BOUNDARY_EXPOSURE_MICROS + 1),
        open_position=None,
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("concentration_limits")(intent, context)

    assert result.vetoed is True
    assert result.reason == "concentration limit exceeded"


def test_concentration_limits_exempts_provable_close_even_massively_over_cap() -> None:
    """(#100) The exemption is unconditional on exposure magnitude: even at
    market_exposure vastly (not just one micro) over its cap, a provable
    de-risking close still approves.
    """
    context = make_context(
        max_pos_market_pct_ppm=_CONCENTRATION_CAP_PPM,
        market_exposure=MoneyMicros(900_000_000_000),
        open_position=ContractCentis(1000),
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("concentration_limits")(intent, context)

    assert result.vetoed is False


def test_concentration_limits_exemption_boundary_matches_reduce_only_provable() -> None:
    """(#100) The exemption's boundary is exactly `size <= open_position`
    (the `_ReduceOnlyProvable` invariant): `size == open_position` is
    provable and exempt (approves despite the over-cap exposure); one
    contract past it is not provable, so full notional is charged and the
    (still over-cap) check vetoes. Pins the `<=` boundary against `<`/`==`
    mutants.
    """
    context = make_context(
        max_pos_market_pct_ppm=_CONCENTRATION_CAP_PPM,
        market_exposure=MoneyMicros(_CONCENTRATION_BOUNDARY_EXPOSURE_MICROS + 1),
        open_position=ContractCentis(1000),
    )

    at_boundary = _real_check("concentration_limits")(
        make_intent(action="sell_to_close", size=ContractCentis(1000)), context
    )
    over_boundary = _real_check("concentration_limits")(
        make_intent(action="sell_to_close", size=ContractCentis(1001)), context
    )

    assert at_boundary.vetoed is False
    assert over_boundary.vetoed is True
    assert over_boundary.reason == "concentration limit exceeded"


def test_concentration_limits_provable_close_approves_with_missing_fee_bound() -> None:
    """(#100) The exemption short-circuits before `_order_cost` is ever
    called: a provable de-risking close approves even when `max_trading_fee`
    is `None`, which would otherwise make the cost unprovable and veto as
    `"unprovable"` -- proof that cost is never computed for an exempt close.
    """
    context = make_context(
        max_pos_market_pct_ppm=_CONCENTRATION_CAP_PPM,
        market_exposure=MoneyMicros(_CONCENTRATION_BOUNDARY_EXPOSURE_MICROS + 1),
        open_position=ContractCentis(1000),
        max_trading_fee=None,
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("concentration_limits")(intent, context)

    assert result.vetoed is False


# --- daily_loss_limit ---------------------------------------------------------------


def test_daily_loss_limit_passes_one_micro_below_threshold() -> None:
    """10% of the default $1,000 `equity_start_of_day` is 100_000_000
    micros; one micro below that passes.
    """
    context = make_context(
        daily_loss_limit_pct_ppm=100_000,
        realized_loss_today=MoneyMicros(99_999_999),
    )

    result = _real_check("daily_loss_limit")(make_intent(), context)

    assert result.vetoed is False


def test_daily_loss_limit_vetoes_at_exact_threshold() -> None:
    """The threshold itself vetoes: `>=`, not `>` -- the boundary is the
    opposite sense from `floor_invariant`'s (there, exact equality passes).
    """
    context = make_context(
        daily_loss_limit_pct_ppm=100_000,
        realized_loss_today=MoneyMicros(100_000_000),
    )

    result = _real_check("daily_loss_limit")(make_intent(), context)

    assert result.vetoed is True


# --- trailing_drawdown_limit ---------------------------------------------------------


def test_trailing_drawdown_limit_passes_one_micro_below_threshold() -> None:
    """Sampled equity 99_999_999 below the default $1,000 high-water mark
    keeps drawdown one micro below the 10% threshold (100_000_000).

    Driven through `sampled_equity` rather than through cash: the drawdown is
    measured between two readings of the ledger's own equity series, because
    worst-case equity falls by a position's full cost the moment one is opened
    and would report deployment as drawdown (issue #514).
    """
    context = make_context(
        max_drawdown_pct_ppm=100_000,
        sampled_equity=MoneyMicros(900_000_001),
    )

    result = _real_check("trailing_drawdown_limit")(make_intent(), context)

    assert result.vetoed is False


def test_trailing_drawdown_limit_vetoes_at_exact_threshold() -> None:
    """Drawdown exactly at the threshold vetoes (`>=`, matching
    `daily_loss_limit`'s boundary sense)."""
    context = make_context(
        max_drawdown_pct_ppm=100_000,
        sampled_equity=MoneyMicros(900_000_000),
    )

    result = _real_check("trailing_drawdown_limit")(make_intent(), context)

    assert result.vetoed is True


def test_trailing_drawdown_limit_ignores_capital_merely_deployed() -> None:
    """Opening a position is not a drawdown, however much cash it consumes.

    Worst-case equity here is zero -- every open position is assumed to resolve
    against us -- while the sampled equity still stands at the high-water mark.
    Measured against worst-case equity the cap would veto at a full 100%
    "drawdown"; measured on the equity series it has nothing to bind on.
    """
    context = make_context(
        max_drawdown_pct_ppm=100_000,
        exchange_verified_available_cash=MoneyMicros(0),
    )

    result = _real_check("trailing_drawdown_limit")(make_intent(), context)

    assert result.vetoed is False


# --- velocity_limits ------------------------------------------------------------------


def test_velocity_limits_passes_at_exact_orders_per_hour_boundary() -> None:
    """`orders_last_hour + 1 == max_orders_per_hour` passes (not `>`)."""
    context = make_context(max_orders_per_hour=5, orders_last_hour=4)

    result = _real_check("velocity_limits")(make_intent(), context)

    assert result.vetoed is False


def test_velocity_limits_vetoes_one_order_over_the_hourly_cap() -> None:
    """`orders_last_hour + 1 > max_orders_per_hour` vetoes."""
    context = make_context(max_orders_per_hour=5, orders_last_hour=5)

    result = _real_check("velocity_limits")(make_intent(), context)

    assert result.vetoed is True


def test_velocity_limits_passes_at_exact_daily_notional_boundary() -> None:
    """`notional_today + worst_case_cost == max_notional_per_day` passes
    (default cost is 5_000_000 micros)."""
    context = make_context(
        notional_today=MoneyMicros(0), max_notional_per_day=MoneyMicros(5_000_000)
    )

    result = _real_check("velocity_limits")(make_intent(), context)

    assert result.vetoed is False


def test_velocity_limits_vetoes_one_micro_over_the_daily_notional_cap() -> None:
    """One micro past the daily notional cap vetoes."""
    context = make_context(
        notional_today=MoneyMicros(0), max_notional_per_day=MoneyMicros(4_999_999)
    )

    result = _real_check("velocity_limits")(make_intent(), context)

    assert result.vetoed is True


@pytest.mark.parametrize(
    "fees_override",
    [
        {"max_trading_fee": None},
        {"max_settlement_fee": None},
    ],
    ids=["trading_fee_none", "settlement_fee_none"],
)
def test_velocity_limits_vetoes_when_a_fee_bound_is_none(
    fees_override: dict[str, object],
) -> None:
    """A missing (`None`) fee bound of either kind makes the daily-notional
    cost unprovable, so `velocity_limits` vetoes as `"unprovable"` (fail-closed)
    instead of raising or approving.
    """
    context = make_context(**fees_override)

    result = _real_check("velocity_limits")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "unprovable"


def test_velocity_limits_exempts_provable_derisking_close_from_daily_notional() -> None:
    """(#100) The daily-notional term is exempt for a provable de-risking
    close: even with `notional_today` already at (not just approaching) the
    daily cap -- before any cost from this order is even considered -- the
    close still approves.
    """
    context = make_context(
        notional_today=MoneyMicros(5_000_000),
        max_notional_per_day=MoneyMicros(5_000_000),
        open_position=ContractCentis(1000),
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("velocity_limits")(intent, context)

    assert result.vetoed is False


def test_velocity_limits_hourly_cap_still_applies_to_a_derisking_close() -> None:
    """(#100) Unlike the daily-notional term, the hourly-order-count term is
    runaway-order protection and is NOT exempted by a de-risking close: a
    provable close that would breach the hourly cap still vetoes with the
    unchanged "hourly order cap exceeded" reason.
    """
    context = make_context(
        max_orders_per_hour=5,
        orders_last_hour=5,
        open_position=ContractCentis(1000),
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("velocity_limits")(intent, context)

    assert result.vetoed is True
    assert result.reason == "hourly order cap exceeded"


def test_velocity_limits_daily_cap_still_vetoes_a_non_provable_close() -> None:
    """The daily-notional exemption is narrow: at the identical exhausted
    daily budget, a close that is NOT provably reduce-only (no open position
    on record) gets no exemption, is still charged full worst-case notional,
    and still vetoes with the unchanged "daily notional cap exceeded" reason
    (#100).
    """
    context = make_context(
        notional_today=MoneyMicros(5_000_000),
        max_notional_per_day=MoneyMicros(5_000_000),
        open_position=None,
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("velocity_limits")(intent, context)

    assert result.vetoed is True
    assert result.reason == "daily notional cap exceeded"


def test_velocity_limits_provable_close_approves_with_missing_fee_bound() -> None:
    """(#100) The daily-notional exemption short-circuits before
    `_order_cost` is ever called: a provable de-risking close, with hourly
    headroom to spare, approves even when `max_trading_fee` is `None` --
    proof that cost is never computed for an exempt close.
    """
    context = make_context(
        max_orders_per_hour=1_000,
        orders_last_hour=0,
        open_position=ContractCentis(1000),
        max_trading_fee=None,
    )
    intent = make_intent(action="sell_to_close", size=ContractCentis(1000))

    result = _real_check("velocity_limits")(intent, context)

    assert result.vetoed is False


# --- de-risking close exemption: cross-cap boundary consistency (#100) -----------

#: Per-cap context overrides that put `concentration_limits`,
#: `mode_permission_ceiling` (LIVE_MICRO), and `velocity_limits` (daily-notional
#: term) each already over its own cap *before* any cost from the probe order is
#: considered -- so a non-exempt (fully-charged) close is guaranteed to veto
#: regardless of the couple-of-micros cost difference between `size ==
#: open_position` and `size == open_position + 1`, isolating what this test
#: actually probes: whether each cap's reduce-only-provable boundary is exactly
#: `size <= open_position`, not the cap arithmetic itself (separately pinned
#: per-check above).
_DERISKING_BOUNDARY_CASES: tuple[tuple[str, dict[str, object], str], ...] = (
    (
        "concentration_limits",
        {
            "max_pos_market_pct_ppm": _CONCENTRATION_CAP_PPM,
            "market_exposure": MoneyMicros(200_000_000),
        },
        "concentration limit exceeded",
    ),
    (
        "mode_permission_ceiling",
        {
            "mode": Mode.LIVE_MICRO,
            "micro_cap": MoneyMicros(1_000_000),
            "total_exposure": MoneyMicros(2_000_000),
        },
        "live-micro exposure ceiling exceeded",
    ),
    (
        "velocity_limits",
        {
            "notional_today": MoneyMicros(2_000_000),
            "max_notional_per_day": MoneyMicros(1_000_000),
        },
        "daily notional cap exceeded",
    ),
)


@pytest.mark.parametrize(
    ("check_name", "context_kwargs", "veto_reason"),
    _DERISKING_BOUNDARY_CASES,
    ids=[case[0] for case in _DERISKING_BOUNDARY_CASES],
)
def test_derisking_close_exemption_boundary_is_consistent_across_all_three_caps(
    check_name: str, context_kwargs: dict[str, object], veto_reason: str
) -> None:
    """(#100) The `size <= open_position` reduce-only boundary is identical
    across `concentration_limits`, `mode_permission_ceiling`, and
    `velocity_limits`: at `size == open_position` the close is provably
    reduce-only and exempt (approves despite the cap already being breached
    independent of this order); one contract past it is not provable, so
    full notional is charged and the still-breached cap vetoes with its own
    reason. A divergent predicate in any one cap fails this test under that
    cap's own parametrize id.
    """
    open_position = ContractCentis(1000)
    context = make_context(open_position=open_position, **context_kwargs)

    at_boundary = _real_check(check_name)(
        make_intent(action="sell_to_close", size=open_position), context
    )
    over_boundary = _real_check(check_name)(
        make_intent(action="sell_to_close", size=ContractCentis(1001)), context
    )

    assert at_boundary.vetoed is False
    assert over_boundary.vetoed is True
    assert over_boundary.reason == veto_reason


# --- quote_freshness / forecast_freshness ---------------------------------------


def test_quote_freshness_passes_at_exact_ttl_age() -> None:
    """A quote exactly `quote_ttl_seconds` old (age == ttl) is still fresh."""
    context = make_context(
        quote_snapshot_epoch_s=DEFAULT_NOW_EPOCH_S - 3_600, quote_ttl_seconds=3_600
    )

    result = _real_check("quote_freshness")(make_intent(), context)

    assert result.vetoed is False


def test_quote_freshness_vetoes_one_second_past_ttl() -> None:
    """A quote one second past its ttl is stale."""
    context = make_context(
        quote_snapshot_epoch_s=DEFAULT_NOW_EPOCH_S - 3_601, quote_ttl_seconds=3_600
    )

    result = _real_check("quote_freshness")(make_intent(), context)

    assert result.vetoed is True


def test_quote_freshness_vetoes_a_future_timestamp() -> None:
    """A quote timestamped after `now_epoch_s` vetoes, however fresh its
    age would otherwise appear."""
    context = make_context(quote_snapshot_epoch_s=DEFAULT_NOW_EPOCH_S + 1)

    result = _real_check("quote_freshness")(make_intent(), context)

    assert result.vetoed is True


def test_quote_freshness_vetoes_when_snapshot_epoch_is_none() -> None:
    """A missing quote timestamp vetoes."""
    context = make_context(quote_snapshot_epoch_s=None)

    result = _real_check("quote_freshness")(make_intent(), context)

    assert result.vetoed is True


def test_forecast_freshness_passes_at_exact_ttl_age() -> None:
    """A forecast exactly `forecast_ttl_seconds` old is still fresh."""
    context = make_context(
        forecast_epoch_s=DEFAULT_NOW_EPOCH_S - 3_600, forecast_ttl_seconds=3_600
    )

    result = _real_check("forecast_freshness")(make_intent(), context)

    assert result.vetoed is False


def test_forecast_freshness_vetoes_one_second_past_ttl() -> None:
    """A forecast one second past its ttl is stale."""
    context = make_context(
        forecast_epoch_s=DEFAULT_NOW_EPOCH_S - 3_601, forecast_ttl_seconds=3_600
    )

    result = _real_check("forecast_freshness")(make_intent(), context)

    assert result.vetoed is True


def test_forecast_freshness_vetoes_a_future_timestamp() -> None:
    """A forecast timestamped after `now_epoch_s` vetoes."""
    context = make_context(forecast_epoch_s=DEFAULT_NOW_EPOCH_S + 1)

    result = _real_check("forecast_freshness")(make_intent(), context)

    assert result.vetoed is True


def test_forecast_freshness_vetoes_when_forecast_epoch_is_none() -> None:
    """A missing forecast timestamp vetoes."""
    context = make_context(forecast_epoch_s=None)

    result = _real_check("forecast_freshness")(make_intent(), context)

    assert result.vetoed is True


# --- price_band_compliance --------------------------------------------------------


def test_price_band_compliance_passes_at_each_inclusive_edge() -> None:
    """An open priced at exactly `min_open_price` or `max_open_price` passes
    -- both edges are inclusive.
    """
    context = make_context(
        min_open_price=PricePips(4_000), max_open_price=PricePips(6_000)
    )

    low_edge = _real_check("price_band_compliance")(
        make_intent(price=PricePips(4_000)), context
    )
    high_edge = _real_check("price_band_compliance")(
        make_intent(price=PricePips(6_000)), context
    )

    assert low_edge.vetoed is False
    assert high_edge.vetoed is False


def test_price_band_compliance_vetoes_one_pip_outside_each_edge() -> None:
    """One pip below the minimum, or above the maximum, vetoes."""
    context = make_context(
        min_open_price=PricePips(4_000), max_open_price=PricePips(6_000)
    )

    below = _real_check("price_band_compliance")(
        make_intent(price=PricePips(3_999)), context
    )
    above = _real_check("price_band_compliance")(
        make_intent(price=PricePips(6_001)), context
    )

    assert below.vetoed is True
    assert above.vetoed is True


def test_price_band_compliance_provable_close_passes_regardless_of_price() -> None:
    """A provable close passes even priced outside the open band -- the band
    only constrains opens."""
    context = make_context(
        min_open_price=PricePips(4_000), max_open_price=PricePips(6_000)
    )
    intent = make_intent(action="sell_to_close", price=PricePips(1))

    result = _real_check("price_band_compliance")(intent, context)

    assert result.vetoed is False


def test_price_band_compliance_vetoes_an_unknown_action() -> None:
    """An action that is neither an open nor a close vetoes."""
    result = _real_check("price_band_compliance")(
        make_intent(action="hold"), make_context()
    )

    assert result.vetoed is True


# --- participation_cap_compliance -------------------------------------------------


def test_participation_cap_compliance_passes_at_exact_threshold() -> None:
    """10% of a 10_000-centi visible depth is exactly 1_000 centis, the
    default intent size -- passes at that exact boundary.
    """
    context = make_context(
        visible_depth=ContractCentis(10_000), max_participation_ppm=100_000
    )

    result = _real_check("participation_cap_compliance")(make_intent(), context)

    assert result.vetoed is False


def test_participation_cap_compliance_vetoes_one_centi_over_threshold() -> None:
    """One centi past that threshold vetoes."""
    context = make_context(
        visible_depth=ContractCentis(10_000), max_participation_ppm=100_000
    )
    intent = make_intent(size=ContractCentis(1_001))

    result = _real_check("participation_cap_compliance")(intent, context)

    assert result.vetoed is True


def test_participation_cap_compliance_vetoes_when_visible_depth_is_none() -> None:
    """A missing visible depth vetoes."""
    context = make_context(visible_depth=None)

    result = _real_check("participation_cap_compliance")(make_intent(), context)

    assert result.vetoed is True


# --- human_ack_satisfied (issue #34) -----------------------------------------------


def test_human_ack_satisfied_approves_in_non_live_modes_even_over_threshold() -> None:
    """Outside LIVE_MICRO/LIVE, a cost over threshold never requires an ack
    -- real capital is not at risk in RESEARCH or PAPER."""
    context = make_context(
        mode=Mode.PAPER, require_human_ack_above_micros=MoneyMicros(0)
    )

    result = _real_check("human_ack_satisfied")(make_intent(), context)

    assert result.vetoed is False


def test_human_ack_satisfied_approves_with_a_null_threshold_even_at_huge_cost() -> None:
    """A `None` threshold means "no human-ack gate configured": it approves
    regardless of cost."""
    context = make_context(
        require_human_ack_above_micros=None,
        max_trading_fee=MoneyMicros(10**12),
        max_settlement_fee=MoneyMicros(10**12),
    )

    result = _real_check("human_ack_satisfied")(make_intent(), context)

    assert result.vetoed is False


def test_human_ack_satisfied_passes_at_exact_threshold_equality() -> None:
    """`cost == threshold` passes -- the threshold is inclusive. Default
    intent cost is 5_000_000 micros."""
    context = make_context(require_human_ack_above_micros=MoneyMicros(5_000_000))

    result = _real_check("human_ack_satisfied")(make_intent(), context)

    assert result.vetoed is False


def test_human_ack_satisfied_vetoes_one_micro_over_threshold_without_an_ack() -> None:
    """One micro past the threshold, with no acknowledgement on record,
    vetoes with the exact SPEC reason."""
    context = make_context(require_human_ack_above_micros=MoneyMicros(4_999_999))

    result = _real_check("human_ack_satisfied")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "human acknowledgement required"


def test_human_ack_satisfied_approves_over_threshold_when_acknowledged() -> None:
    """The same over-threshold cost passes once the intent id is present in
    `acknowledged_intent_ids`."""
    intent = make_intent(intent_id="acked-intent")
    context = make_context(
        require_human_ack_above_micros=MoneyMicros(4_999_999),
        acknowledged_intent_ids=frozenset({"acked-intent"}),
    )

    result = _real_check("human_ack_satisfied")(intent, context)

    assert result.vetoed is False


@pytest.mark.parametrize(
    "fees_override",
    [
        {"max_trading_fee": None},
        {"max_settlement_fee": None},
    ],
    ids=["trading_fee_none", "settlement_fee_none"],
)
def test_human_ack_satisfied_vetoes_as_unprovable_when_a_fee_bound_is_none(
    fees_override: dict[str, object],
) -> None:
    """A missing fee bound makes the cost unprovable, so this check fails
    closed as `"unprovable"` instead of approving."""
    context = make_context(
        require_human_ack_above_micros=MoneyMicros(0), **fees_override
    )

    result = _real_check("human_ack_satisfied")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "unprovable"


# --- approval_token_uniqueness (issue #31) ----------------------------------------


def test_approval_token_uniqueness_passes_when_intent_id_is_unused() -> None:
    """An intent id absent from `context.used_intent_ids` passes."""
    context = make_context(used_intent_ids=frozenset())

    result = _real_check("approval_token_uniqueness")(make_intent(), context)

    assert result.vetoed is False


def test_approval_token_uniqueness_vetoes_when_intent_id_already_used() -> None:
    """An intent id already present in `context.used_intent_ids` vetoes --
    the kernel never issues a second approval token for the same intent.
    """
    intent = make_intent(intent_id="dup-intent")
    context = make_context(used_intent_ids=frozenset({"dup-intent"}))

    result = _real_check("approval_token_uniqueness")(intent, context)

    assert result.vetoed is True


def test_approval_token_uniqueness_passes_when_only_other_ids_are_used() -> None:
    """A `used_intent_ids` set containing *other* intent ids does not veto
    this unrelated intent -- pins membership, not "any ids used at all".
    """
    intent = make_intent(intent_id="fresh-intent")
    context = make_context(used_intent_ids=frozenset({"other-1", "other-2"}))

    result = _real_check("approval_token_uniqueness")(intent, context)

    assert result.vetoed is False


# --- idempotency_key_uniqueness (issue #31) ---------------------------------------


def test_idempotency_key_uniqueness_passes_when_key_is_unused() -> None:
    """An idempotency key absent from `context.used_idempotency_keys` passes."""
    context = make_context(used_idempotency_keys=frozenset())

    result = _real_check("idempotency_key_uniqueness")(make_intent(), context)

    assert result.vetoed is False


def test_idempotency_key_uniqueness_vetoes_when_key_already_used() -> None:
    """An idempotency key already present in `context.used_idempotency_keys`
    vetoes -- the kernel never double-reserves against the same key.
    """
    intent = make_intent(idempotency_key="dup-idem")
    context = make_context(used_idempotency_keys=frozenset({"dup-idem"}))

    result = _real_check("idempotency_key_uniqueness")(intent, context)

    assert result.vetoed is True


def test_idempotency_key_uniqueness_passes_when_only_other_keys_are_used() -> None:
    """A `used_idempotency_keys` set containing *other* keys does not veto
    this unrelated intent -- pins membership, not "any keys used at all".
    """
    intent = make_intent(idempotency_key="fresh-idem")
    context = make_context(used_idempotency_keys=frozenset({"other-1", "other-2"}))

    result = _real_check("idempotency_key_uniqueness")(intent, context)

    assert result.vetoed is False


# --- clock_skew_limit -------------------------------------------------------------


def test_clock_skew_limit_passes_at_exact_skew_boundary() -> None:
    """A skew exactly at `clock_skew_max_seconds` passes (equal passes)."""
    context = make_context(
        exchange_clock_epoch_s=DEFAULT_NOW_EPOCH_S - 3_600, clock_skew_max_seconds=3_600
    )

    result = _real_check("clock_skew_limit")(make_intent(), context)

    assert result.vetoed is False


@pytest.mark.parametrize("direction", [-1, 1], ids=["behind", "ahead"])
def test_clock_skew_limit_vetoes_one_second_past_the_boundary_either_direction(
    direction: int,
) -> None:
    """One second past the skew boundary vetoes, whether the exchange clock
    is behind or ahead of `now_epoch_s` -- `abs()` is symmetric.
    """
    skewed_epoch = DEFAULT_NOW_EPOCH_S + direction * 3_601
    context = make_context(
        exchange_clock_epoch_s=skewed_epoch, clock_skew_max_seconds=3_600
    )

    result = _real_check("clock_skew_limit")(make_intent(), context)

    assert result.vetoed is True


def test_clock_skew_limit_vetoes_when_exchange_clock_epoch_is_none() -> None:
    """A missing exchange clock timestamp vetoes."""
    context = make_context(exchange_clock_epoch_s=None)

    result = _real_check("clock_skew_limit")(make_intent(), context)

    assert result.vetoed is True


# --- exchange_status_ok (issue #110) -----------------------------------------------


def test_exchange_status_ok_passes_with_default_open_and_fresh_status() -> None:
    """The default context (`OPEN`, fresh) passes."""
    result = _real_check("exchange_status_ok")(make_intent(), make_context())

    assert result.vetoed is False


def test_exchange_status_ok_vetoes_when_status_is_none() -> None:
    """A missing (`None`) exchange status vetoes -- fail-closed on an unknown
    status, mirroring every other `None`-means-veto optional field.
    """
    context = make_context(exchange_status=None)

    result = _real_check("exchange_status_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "exchange status stale or missing"


def test_exchange_status_ok_vetoes_when_epoch_is_none() -> None:
    """A present status with a missing epoch vetoes -- staleness is checked
    first, independent of the status value itself.
    """
    context = make_context(
        exchange_status=ExchangeTradingStatus.OPEN, exchange_status_epoch_s=None
    )

    result = _real_check("exchange_status_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "exchange status stale or missing"


def test_exchange_status_ok_vetoes_a_future_timestamp() -> None:
    """A status timestamped after `now_epoch_s` vetoes, however fresh its age
    would otherwise appear -- matching `quote_freshness`'s future-dated
    boundary.
    """
    context = make_context(
        exchange_status=ExchangeTradingStatus.OPEN,
        exchange_status_epoch_s=DEFAULT_NOW_EPOCH_S + 1,
    )

    result = _real_check("exchange_status_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "exchange status stale or missing"


def test_exchange_status_ok_vetoes_one_second_past_ttl() -> None:
    """A status one second past its ttl is stale."""
    context = make_context(
        exchange_status_ttl_seconds=3_600,
        exchange_status_epoch_s=DEFAULT_NOW_EPOCH_S - 3_601,
    )

    result = _real_check("exchange_status_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "exchange status stale or missing"


def test_exchange_status_ok_passes_at_exact_ttl_age() -> None:
    """A status exactly `exchange_status_ttl_seconds` old (age == ttl) is
    still fresh -- the same inclusive boundary `quote_freshness` pins.
    """
    context = make_context(
        exchange_status_ttl_seconds=3_600,
        exchange_status_epoch_s=DEFAULT_NOW_EPOCH_S - 3_600,
    )

    result = _real_check("exchange_status_ok")(make_intent(), context)

    assert result.vetoed is False


def test_exchange_status_ok_vetoes_a_fresh_paused_status() -> None:
    """A fresh `PAUSED` status vetoes with the distinct "not open for
    trading" reason -- proving the tradable-status branch runs once staleness
    has already passed.
    """
    context = make_context(exchange_status=ExchangeTradingStatus.PAUSED)

    result = _real_check("exchange_status_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "exchange not open for trading"


def test_exchange_status_ok_vetoes_a_fresh_closed_status() -> None:
    """A fresh `CLOSED` status vetoes with the same "not open" reason."""
    context = make_context(exchange_status=ExchangeTradingStatus.CLOSED)

    result = _real_check("exchange_status_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "exchange not open for trading"


# --- pipeline_heartbeat_ok (issue #110) ---------------------------------------------


def test_pipeline_heartbeat_ok_passes_with_default_fresh_heartbeat() -> None:
    """The default context (a fresh heartbeat) passes."""
    result = _real_check("pipeline_heartbeat_ok")(make_intent(), make_context())

    assert result.vetoed is False


def test_pipeline_heartbeat_ok_vetoes_when_epoch_is_none() -> None:
    """A missing heartbeat timestamp vetoes."""
    context = make_context(pipeline_heartbeat_epoch_s=None)

    result = _real_check("pipeline_heartbeat_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "pipeline heartbeat stale or missing"


def test_pipeline_heartbeat_ok_vetoes_a_future_timestamp() -> None:
    """A heartbeat timestamped after `now_epoch_s` vetoes, however fresh its
    age would otherwise appear.
    """
    context = make_context(pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S + 1)

    result = _real_check("pipeline_heartbeat_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "pipeline heartbeat stale or missing"


def test_pipeline_heartbeat_ok_vetoes_one_second_past_ttl() -> None:
    """A heartbeat one second past its ttl is stale."""
    context = make_context(
        pipeline_heartbeat_ttl_seconds=3_600,
        pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S - 3_601,
    )

    result = _real_check("pipeline_heartbeat_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "pipeline heartbeat stale or missing"


def test_pipeline_heartbeat_ok_passes_at_exact_ttl_age() -> None:
    """A heartbeat exactly `pipeline_heartbeat_ttl_seconds` old (age == ttl)
    is still fresh -- the same inclusive boundary `quote_freshness` pins.
    """
    context = make_context(
        pipeline_heartbeat_ttl_seconds=3_600,
        pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S - 3_600,
    )

    result = _real_check("pipeline_heartbeat_ok")(make_intent(), context)

    assert result.vetoed is False


def test_pipeline_heartbeat_ok_reads_a_distinct_configured_ttl() -> None:
    """A distinct, tight ttl (10s) proves the check reads `limits`'s own
    field rather than some other check's ttl constant.
    """
    context = make_context(
        pipeline_heartbeat_ttl_seconds=10,
        pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S - 11,
    )

    result = _real_check("pipeline_heartbeat_ok")(make_intent(), context)

    assert result.vetoed is True
    assert result.reason == "pipeline heartbeat stale or missing"


# --- exchange_status_ok / pipeline_heartbeat_ok: DEFAULT_CHECKS routing (#110) ------


def test_default_checks_routes_the_real_exchange_status_ok_check() -> None:
    """`evaluate_intent` over `DEFAULT_CHECKS` surfaces the exchange-status
    veto reason when only `exchange_status` is missing -- proving
    `DEFAULT_CHECKS` wires the real check, not the retired stub.
    """
    intent = make_intent()
    context = make_context(exchange_status=None)

    decision = evaluate_intent(intent, context)

    assert "exchange status stale or missing" in decision.reasons


def test_default_checks_routes_the_real_pipeline_heartbeat_ok_check() -> None:
    """`evaluate_intent` over `DEFAULT_CHECKS` surfaces the heartbeat veto
    reason when only the heartbeat is stale -- proving `DEFAULT_CHECKS`
    wires the real check, not the retired stub.
    """
    intent = make_intent()
    context = make_context(
        pipeline_heartbeat_ttl_seconds=3_600,
        pipeline_heartbeat_epoch_s=DEFAULT_NOW_EPOCH_S - 3_601,
    )

    decision = evaluate_intent(intent, context)

    assert "pipeline heartbeat stale or missing" in decision.reasons


# --- reduce_only_provable ----------------------------------------------------------


def test_reduce_only_provable_passes_any_open_regardless_of_open_position() -> None:
    """Opens are "not applicable" to reduce-only provability and pass even
    with no recorded open position.
    """
    context = make_context(open_position=None)

    result = _real_check("reduce_only_provable")(make_intent(action="buy"), context)

    assert result.vetoed is False


def test_reduce_only_provable_close_passes_at_size_equal_to_open_position() -> None:
    """A close sized exactly equal to the open position passes."""
    context = make_context(open_position=ContractCentis(1_000))
    intent = make_intent(action="sell_to_close", size=ContractCentis(1_000))

    result = _real_check("reduce_only_provable")(intent, context)

    assert result.vetoed is False


def test_reduce_only_provable_close_vetoes_one_centi_over_the_open_position() -> None:
    """A close one centi larger than the open position vetoes -- it would
    flip to a net-new position, not reduce one.
    """
    context = make_context(open_position=ContractCentis(999))
    intent = make_intent(action="sell_to_close", size=ContractCentis(1_000))

    result = _real_check("reduce_only_provable")(intent, context)

    assert result.vetoed is True


def test_reduce_only_provable_close_vetoes_when_open_position_is_none() -> None:
    """A close with no recorded open position vetoes -- there is nothing on
    record to prove it reduces."""
    context = make_context(open_position=None)
    intent = make_intent(action="sell_to_close")

    result = _real_check("reduce_only_provable")(intent, context)

    assert result.vetoed is True


def test_reduce_only_provable_vetoes_an_unknown_action() -> None:
    """An action that is neither an open nor a close vetoes."""
    intent = make_intent(action="hold")

    result = _real_check("reduce_only_provable")(intent, make_context())

    assert result.vetoed is True


# --- Pipeline: order, stub survivors, fail-closed, frozen types -------------------


def test_default_checks_names_match_spec_10_3_exactly_in_order() -> None:
    """`DEFAULT_CHECKS`' `.name`s equal the pinned SPEC S10.3 sequence
    exactly, in order -- unchanged by issue #30."""
    assert len(DEFAULT_CHECKS) == _EXPECTED_CHECK_COUNT
    assert tuple(check.name for check in DEFAULT_CHECKS) == EXPECTED_CHECK_NAMES


def test_default_checks_over_a_permissive_context_approves() -> None:
    """A fully permissive context now clears all 24 checks with no veto.

    This is the direct negation of the pre-#340 state: while
    `jurisdiction_product_eligibility` vetoed unconditionally, no context of any
    kind could produce an approving decision, so the Risk Kernel could never
    approve an intent in any mode.
    """
    intent = make_intent()
    context = make_context()

    decision = evaluate_intent(intent, context)

    assert decision.vetoed is False
    assert decision.reasons == ()


def test_evaluate_intent_fail_closed_converts_a_raised_exception_to_a_veto() -> None:
    """A check that raises is converted to a veto reason -- the exception
    never escapes `evaluate_intent` -- and checks positioned after the
    raising one still run.
    """
    intent = make_intent()
    context = make_context()
    checks = (*DEFAULT_CHECKS[:2], _RaisingCheck(), *DEFAULT_CHECKS[2:])

    decision = evaluate_intent(intent, context, checks=checks)

    assert decision.vetoed is True
    assert "raising_check: error: boom" in decision.reasons


def test_evaluate_intent_omits_reasons_for_checks_that_pass() -> None:
    """A non-vetoing check contributes no reason; only vetoing checks do."""
    intent = make_intent()
    context = make_context()
    checks = (_PassingCheck(), _VetoingCheck(), _PassingCheck())

    decision = evaluate_intent(intent, context, checks=checks)

    assert decision.vetoed is True
    assert len(decision.reasons) == 1


def test_evaluate_intent_with_only_passing_checks_is_not_vetoed() -> None:
    """When every check passes, the aggregate decision does not veto."""
    decision = evaluate_intent(
        make_intent(), make_context(), checks=(_PassingCheck(), _PassingCheck())
    )

    assert decision.vetoed is False
    assert decision.reasons == ()


def test_evaluate_intent_with_an_empty_checks_sequence_is_not_vetoed() -> None:
    """With zero checks, there is nothing to veto on."""
    decision = evaluate_intent(make_intent(), make_context(), checks=())

    assert decision.vetoed is False
    assert decision.reasons == ()


# --- OrderIntent: frozen, scaled-int-only (migrated from test_checks_stub.py) --------


def test_order_intent_is_frozen() -> None:
    """Mutating any field of a constructed `OrderIntent` raises."""
    intent = make_intent()

    # `setattr` on a runtime-bound name is the sanctioned idiom: a literal
    # `intent.action = ...` on a frozen dataclass is a static mypy error, not
    # the runtime `FrozenInstanceError` this test pins; a literal `setattr`
    # name in turn trips ruff B010, so the field name is bound at runtime.
    frozen_field = "action"
    with pytest.raises(FrozenInstanceError):
        setattr(intent, frozen_field, "sell")


def test_order_intent_numeric_fields_are_scaled_int_types_never_floats() -> None:
    """Every money/price/size/probability field is a scaled-int wrapper type
    whose underlying `.value` is a true `int`, never a `float`.
    """
    intent = make_intent()

    assert isinstance(intent.price, PricePips)
    assert isinstance(intent.size, ContractCentis)
    assert isinstance(intent.max_notional, MoneyMicros)
    assert isinstance(intent.implied_probability, ProbabilityPpm)

    for scaled_value in (
        intent.price,
        intent.size,
        intent.max_notional,
        intent.implied_probability,
    ):
        assert type(scaled_value.value) is int
        assert not isinstance(scaled_value.value, bool)


def test_order_intent_string_fields_hold_the_constructed_values() -> None:
    """The four str identity fields round-trip exactly through construction."""
    intent = make_intent(
        intent_id="intent-xyz",
        market_ticker="TICKER-X",
        outcome="no",
        action="sell_to_close",
    )

    assert intent.intent_id == "intent-xyz"
    assert intent.market_ticker == "TICKER-X"
    assert intent.outcome == "no"
    assert intent.action == "sell_to_close"


def test_order_intent_rejects_a_new_attribute() -> None:
    """`OrderIntent` is a frozen dataclass: assigning any undeclared
    attribute raises `AttributeError` (via `FrozenInstanceError`).
    """
    intent = make_intent()
    # The attribute name is bound at runtime (not a literal) because assigning an
    # *undeclared* attribute cannot be written as `intent.not_a_declared_field =
    # ...` -- that is a static attr-defined error, not the runtime rejection this
    # test pins. `setattr` on a runtime name is the sanctioned idiom for it.
    undeclared_attribute = "not_a_declared_field"

    with pytest.raises(AttributeError):
        setattr(intent, undeclared_attribute, "nope")


def test_check_result_is_frozen() -> None:
    """`CheckResult` instances reject attribute mutation."""
    result = CheckResult(vetoed=True, reason="blocked by issue #32")

    frozen_field = "reason"
    with pytest.raises(FrozenInstanceError):
        setattr(result, frozen_field, "changed")


def test_decision_is_frozen() -> None:
    """`Decision` instances (as returned by `evaluate_intent`) reject
    attribute mutation.
    """
    decision = evaluate_intent(make_intent(), make_context())

    frozen_field = "vetoed"
    with pytest.raises(FrozenInstanceError):
        setattr(decision, frozen_field, False)


def test_default_market_ticker_fixture_matches_default_intent_ticker() -> None:
    """Sanity check on the shared fixtures: the conftest constant and
    `make_intent()`'s default ticker agree, so `instrument_whitelist` passes
    by default as every other per-check test assumes.
    """
    assert make_intent().market_ticker == DEFAULT_MARKET_TICKER
