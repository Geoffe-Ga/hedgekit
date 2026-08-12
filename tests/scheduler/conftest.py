"""Shared builders for `tests/scheduler/*` (issue #48, RED).

`windbreak.scheduler.loop` does not exist yet, so importing `PaperTickDeps`,
`build_paper_deps`, `run_single_tick`, `ApprovalSeam`, or `KernelApproval`
below fails collection with `ModuleNotFoundError: No module named
'windbreak.scheduler'` -- the expected Gate 1 RED state for issue #48.

This module itself imports only already-shipped machinery
(`windbreak.riskkernel.*`, `windbreak.ledger.*`, `windbreak.numeric`,
`tests.riskkernel.conftest`), so it collects cleanly on its own; the
`ModuleNotFoundError` surfaces only from `test_loop.py`'s own
`from windbreak.scheduler.loop import ...` line.

Builder-placement choice mirrors `tests/riskkernel/conftest.py` and
`tests/order_gateway/conftest.py`: plain, explicitly-imported functions
rather than pytest fixtures, so they compose cleanly inside
`@pytest.mark.parametrize`-driven tables and inside any Hypothesis-decorated
test that might be added later.
"""

from __future__ import annotations

from windbreak.config.schema import CorrelationConfig, CorrelationTagConfig
from windbreak.numeric import MoneyMicros
from windbreak.riskkernel.modes import Mode, ModeStateMachine
from windbreak.riskkernel.process import InMemoryKernelLedgerWriter, RiskKernel
from windbreak.riskkernel.reservations import ApprovalPipeline, ReservationLedger
from windbreak.riskkernel.signing import SigningKeyHandle
from windbreak.riskkernel.tokens import TokenIssuer
from windbreak.scheduler.exposure import ExposureProjection, project_exposure
from windbreak.selector.correlation import BUCKET_WEATHER

#: A fixed 32-byte HMAC key shared by every mint/verify pair in this package's
#: tests -- mirrors `tests/order_gateway/conftest.py::KEY_MATERIAL`. SPEC S10.6
#: approval tokens are symmetric, so the exact same bytes both mint (via
#: `TokenIssuer`) and would verify (Gateway side) for the fill-leg scenario.
KEY_MATERIAL = b"s" * 32

#: The single ticker every scheduler-unit-test intent targets: the sole ticker
#: in the shared `tests/fixtures/books/deep_walk` fixture, matching
#: `tests/order_gateway/conftest.py::DEFAULT_MARKET_TICKER`.
DEFAULT_MARKET_TICKER = "MKT-DEEP"

#: The fixed "current instant" (epoch seconds) every builder below agrees on.
DEFAULT_NOW_EPOCH_S = 1_700_000_000

#: A fixed, content-stable config-revision hash stamped on every issued token.
TEST_CONFIG_HASH = "scheduler-test-config-hash"


def weather_bucket_correlation(*tickers: str) -> CorrelationConfig:
    """Declare every named ticker into the weather correlation bucket.

    The operator-supplied declaration issue #407 made load-bearing: a market
    absent from this config is unbucketable, so the loop declines to size it
    rather than sizing it against an empty bucket. Tests that need exposure to
    actually aggregate must declare every ticker they hold *and* the one they
    are sizing.

    The bucket choice is arbitrary and the instant is fixed rather than read
    from a clock -- `CorrelationTag.tagged_at` is provenance, never arithmetic,
    so a wall-clock read would make the tag unpinnable for no gain.

    Args:
        tickers: The markets to declare into one shared bucket.

    Returns:
        The assembled `CorrelationConfig`.
    """
    return CorrelationConfig(
        tags=tuple(
            CorrelationTagConfig(
                ticker=ticker,
                bucket_ids=(BUCKET_WEATHER,),
                tagged_at="2026-03-01T00:00:00+00:00",
            )
            for ticker in tickers
        )
    )


def proven_flat_exposure(ticker: str = "TEST-TICKER") -> ExposureProjection:
    """Project a genuinely flat account's exposure, the way production does.

    This exists because `exposure=None` and a *proven* flat projection are two
    different facts and `build_evaluation_context` treats them differently.
    `None` means exposure could not be established at all, which zeroes the four
    concentration caps so they veto; this helper means the venue answered, the
    account holds nothing, and all four exposure terms are provably zero --
    exactly what the pre-existing tests below assumed implicitly back when the
    four terms were hardcoded to zero. The projection is produced by calling
    `project_exposure` on an empty holdings tuple rather than by hand-building
    the dataclass, so the zeros the tests rely on come from the real production
    path and stay honest if that path ever changes.

    Args:
        ticker: The market being sized, declared into the weather bucket so
            `project_exposure` can bucket it rather than fail closed.

    Returns:
        The flat `ExposureProjection` for `ticker`.
    """
    correlation = CorrelationConfig(
        tags=(
            CorrelationTagConfig(
                ticker=ticker,
                bucket_ids=(BUCKET_WEATHER,),
                tagged_at="2026-03-01T00:00:00+00:00",
            ),
        )
    )
    projection = project_exposure(
        (),
        target_ticker=ticker,
        target_event_ticker=f"{ticker}-EVENT",
        correlation=correlation,
    )
    assert projection is not None
    return projection


def proven_untraded_day() -> MoneyMicros:
    """Return the day's booked notional for a ledger holding no fills.

    The `notional_today` companion of `proven_flat_exposure`, and it exists for
    the same reason: `None` and a proven zero are different facts, and
    `build_evaluation_context` treats them differently. `None` means the day's
    spend could not be established, which zeroes `max_notional_per_day` so
    `velocity_limits` vetoes; this helper means the ledger was read and nothing
    was booked today, which is what every test below assumed implicitly back
    when the term was hardcoded to zero (issue #415).

    Returns:
        `MoneyMicros(0)` -- an untraded day, provably.
    """
    return MoneyMicros(0)


def proven_idle_hour() -> int:
    """Return the trailing hour's routed-order count for a ledger with none.

    The `orders_last_hour` companion of `proven_untraded_day`, and it exists for
    the same reason: `None` and a proven zero are different facts, and
    `build_evaluation_context` treats them differently. `None` means the hour's
    routing history could not be established, which zeroes `max_orders_per_hour`
    so `velocity_limits` vetoes; this helper means the ledger was read and no
    order was routed in the trailing hour, which is what every test below
    assumed implicitly back when the term was hardcoded (issue #491).

    Named rather than written as a bare `0` at each call site precisely because
    the bare `0` *was* the defect: a reader could not tell the hardcoded
    placeholder from a measured idle hour, and neither could a reviewer.

    Returns:
        `0` -- an hour that routed nothing, provably.
    """
    return 0


def build_kernel_approval_components(
    *, key_material: bytes = KEY_MATERIAL, config_hash: str = TEST_CONFIG_HASH
) -> tuple[RiskKernel, ApprovalPipeline, InMemoryKernelLedgerWriter]:
    """Build a real `RiskKernel` + `ApprovalPipeline` pair over one shared ledger.

    Mirrors the exact composition `windbreak.scheduler.loop.KernelApproval` is
    specified to wire: a `RiskKernel` (for the ledgered `IntentVetoed`/
    `IntentApproved` audit event) and an `ApprovalPipeline` (for the
    reserve-and-issue path), both writing through the same in-memory ledger so
    a test can inspect every event either component recorded, in order.

    Args:
        key_material: The signing key material the issued tokens are minted
            under.
        config_hash: The configuration-revision hash stamped into every
            issued token's claims.

    Returns:
        A `(kernel, pipeline, writer)` triple sharing one ledger writer.
    """
    writer = InMemoryKernelLedgerWriter()
    # The kernel's *own* tracked mode -- not the caller's context -- is what
    # `RiskKernel.evaluate_intent` stamps onto the effective context, so the
    # mode machine is started already in PAPER (matching the always-on PAPER
    # tick this package composes) rather than the default RESEARCH, which
    # would add a spurious `mode_permission_ceiling` veto reason on top of the
    # ones this suite is pinning.
    mode_machine = ModeStateMachine(mode_ceiling=Mode.PAPER, mode=Mode.PAPER)
    kernel = RiskKernel(writer, mode_machine=mode_machine)
    ledger = ReservationLedger(writer)
    issuer = TokenIssuer(SigningKeyHandle(key_material))
    pipeline = ApprovalPipeline(ledger, issuer, config_hash=config_hash)
    return kernel, pipeline, writer
