"""Tests for issue #281's per-provider vote-cost forecast-event emission (RED).

`windbreak.forecast.pipeline` does not yet define `PROVIDER_VOTE_COSTED_EVENT`
(nor `VOTE_OUTCOME_VOTED`/`VOTE_OUTCOME_ABSTAINED`/`VOTE_OUTCOME_DISCARDED`), so
every import below fails collection with `ImportError: cannot import name
'PROVIDER_VOTE_COSTED_EVENT' from 'windbreak.forecast.pipeline'` -- the
expected Gate 1 RED state for issue #281.

Pins the chief architect's contract: when `collect_model_votes`/`run_pipeline`
is wired with an `InMemoryForecastLedger`, the vote-collection stage
(`_collect_provider_forecasts`) emits exactly one `PROVIDER_VOTE_COSTED`
forecast-event per ensemble member it drives:

* a surviving, non-abstaining vote -> outcome `"voted"`,
  `cost_micros=forecast.cost_micros`, `failure_code=""`;
* a surviving, explicitly-abstaining vote -> outcome `"abstained"`,
  `cost_micros=forecast.cost_micros`, `failure_code=""`;
* a discarded vote (the provider raised `ProviderVoteError`) -> outcome
  `"discarded"`, `cost_micros=failed.cost_micros`,
  `failure_code=failed.failure_code`.

Every event's payload carries exactly
`{market_ticker, provider, model_version, vote_index, cost_micros, outcome,
failure_code}` -- `vote_index` matching the member's position in the driven
ensemble, regardless of which provider name backs it (the default ensemble
votes `"openai"` twice, at positions 0 and 2, issue #281's own headline edge
case). With no ledger wired (`ledger=None`, the default) the vote-cost seam
is never touched -- the exact same routing yields the exact same surviving
votes either way.

Local-doubles choice
    This module defines its own minimal `_SucceedingProvider` /
    `_FailingProvider` / `_routed_provider_factory` /
    `_MEMBER_A`/`_MEMBER_B`/`_MEMBER_C` doubles rather than importing them
    from a sibling test module -- cross-test-module imports are not
    supported under this project's rootdir-relative pytest import mode (see
    `tests/forecast/conftest.py`'s and `tests/forecast/test_provider_failures.py`'s
    docstrings for the same convention applied elsewhere).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from windbreak.forecast.cassettes import ForbiddenLiveTransport
from windbreak.forecast.pipeline import (
    PROVIDER_VOTE_COSTED_EVENT,
    VOTE_OUTCOME_ABSTAINED,
    VOTE_OUTCOME_DISCARDED,
    VOTE_OUTCOME_VOTED,
    InMemoryForecastLedger,
    collect_model_votes,
)
from windbreak.forecast.providers import DEFAULT_VOTE_ENSEMBLE, ProviderForecast
from windbreak.forecast.providers.base import ProviderVoteError

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.pipeline import ForecastEvent
    from windbreak.forecast.providers import EnsembleMemberLike, ForecastProvider
    from windbreak.forecast.records import BaselineQuoteSnapshot

#: The three pinned default ensemble members (SPEC S6.3), indexed rather than
#: unpacked from the variable-length `DEFAULT_VOTE_ENSEMBLE` tuple (mirrors
#: `tests/forecast/test_provider_failures.py`'s identical convention). `_MEMBER_A`
#: and `_MEMBER_C` are both `"openai"` -- distinct `model_version`s -- the exact
#: same-provider-twice shape this issue's rebuild fold pins separately.
_MEMBER_A = DEFAULT_VOTE_ENSEMBLE[0]
_MEMBER_B = DEFAULT_VOTE_ENSEMBLE[1]
_MEMBER_C = DEFAULT_VOTE_ENSEMBLE[2]


class _SucceedingProvider:
    """A `ForecastProvider` double returning one fixed forecast every call."""

    def __init__(self, forecast: ProviderForecast) -> None:
        """Store the forecast every `forecast` call returns.

        Args:
            forecast: The fixed `ProviderForecast` to return every time.
        """
        self._forecast = forecast

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[object, ...],
    ) -> ProviderForecast:
        """Return the stored forecast, ignoring every argument.

        Args:
            market: The (unused) market under forecast.
            baseline: The (unused) baseline quote snapshot.
            vote_index: The (unused) zero-based vote index.
            quotes: The (unused) sanitized web quotes.

        Returns:
            The stored `ProviderForecast`, verbatim.
        """
        return self._forecast


class _FailingProvider:
    """A `ForecastProvider` double that raises a fixed error on every call."""

    def __init__(self, error: BaseException) -> None:
        """Store the error every `forecast` call raises.

        Args:
            error: The exception instance to raise, unmodified, every call.
        """
        self._error = error

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[object, ...],
    ) -> ProviderForecast:
        """Raise the stored error, ignoring every argument.

        Args:
            market: The (unused) market under forecast.
            baseline: The (unused) baseline quote snapshot.
            vote_index: The (unused) zero-based vote index.
            quotes: The (unused) sanitized web quotes.

        Raises:
            BaseException: The stored `self._error`, unconditionally.
        """
        raise self._error


def _routed_provider_factory(
    routes: dict[str, ForecastProvider],
) -> Callable[[EnsembleMemberLike], ForecastProvider]:
    """Build a `provider_factory` routing each member by its `model_version`.

    Args:
        routes: A `{model_version: ForecastProvider}` mapping covering every
            member the driven ensemble will route through.

    Returns:
        A `provider_factory` closure looking `member.model_version` up in
        `routes`.
    """

    def _factory(member: EnsembleMemberLike) -> ForecastProvider:
        """Return the provider routed for `member`'s pinned model version.

        Args:
            member: The ensemble member being driven.

        Returns:
            The routed `ForecastProvider`.
        """
        return routes[member.model_version]

    return _factory


def _member_provider(
    member: EnsembleMemberLike,
    *,
    probability_ppm: int,
    abstain: bool,
    cost_micros: int = 0,
) -> ForecastProvider:
    """Build a `_SucceedingProvider` returning a clean, member-stamped forecast.

    Args:
        member: The ensemble member this provider's forecast is stamped with.
        probability_ppm: The forecast's probability estimate, in ppm.
        abstain: Whether the returned forecast abstains.
        cost_micros: The forecast's billed cost, in micros.

    Returns:
        A `_SucceedingProvider` returning a valid, member-stamped forecast.
    """
    return _SucceedingProvider(
        ProviderForecast(
            probability_ppm=probability_ppm,
            rationale_summary="steady evidence",
            citations=(),
            cost_micros=cost_micros,
            provider=member.provider,
            model_version=member.model_version,
            training_cutoff=member.training_cutoff,
            response_fingerprint="f" * 64,
            abstain=abstain,
        )
    )


def _all_voted_routes() -> dict[str, ForecastProvider]:
    """Route all three default members to a clean, non-abstaining vote.

    Returns:
        A `{model_version: ForecastProvider}` mapping for the three default
        ensemble members, each voting cleanly with a distinct cost.
    """
    return {
        _MEMBER_A.model_version: _member_provider(
            _MEMBER_A, probability_ppm=400_000, abstain=False, cost_micros=100
        ),
        _MEMBER_B.model_version: _member_provider(
            _MEMBER_B, probability_ppm=500_000, abstain=False, cost_micros=200
        ),
        _MEMBER_C.model_version: _member_provider(
            _MEMBER_C, probability_ppm=600_000, abstain=False, cost_micros=300
        ),
    }


def _events_by_provider(
    events: tuple[ForecastEvent, ...],
) -> dict[str, ForecastEvent]:
    """Index a tuple of `PROVIDER_VOTE_COSTED` events by their `model_version`.

    Args:
        events: The `PROVIDER_VOTE_COSTED` events to index.

    Returns:
        The events keyed by `payload["model_version"]` (unique per member,
        unlike `payload["provider"]`, which the default ensemble repeats).
    """
    return {event.payload["model_version"]: event for event in events}


# --- Module-level literal pin -----------------------------------------------


def test_provider_vote_costed_event_string_constants_are_pinned() -> None:
    """The event-type and outcome wire-format strings are pinned exactly, so
    a future rename cannot silently drift the ledgered `outcome` value.
    """
    assert PROVIDER_VOTE_COSTED_EVENT == "PROVIDER_VOTE_COSTED"
    assert VOTE_OUTCOME_VOTED == "voted"
    assert VOTE_OUTCOME_ABSTAINED == "abstained"
    assert VOTE_OUTCOME_DISCARDED == "discarded"


# --- All three members vote cleanly: three "voted" events ------------------


def test_collect_model_votes_ledgers_one_voted_event_per_member(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
) -> None:
    """Three clean, surviving votes ledger exactly three `PROVIDER_VOTE_COSTED`
    events, each `outcome="voted"`, matching each member's own `cost_micros`
    and `vote_index` (positional, `0`/`1`/`2` -- not derived from provider
    name, since `_MEMBER_A`/`_MEMBER_C` share the `"openai"` provider).
    """
    ledger = InMemoryForecastLedger()

    collect_model_votes(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        provider_factory=_routed_provider_factory(_all_voted_routes()),
    )

    events = ledger.events_by_type(PROVIDER_VOTE_COSTED_EVENT)
    assert len(events) == 3
    by_member = _events_by_provider(events)
    assert by_member[_MEMBER_A.model_version].payload == {
        "market_ticker": market.ticker,
        "provider": _MEMBER_A.provider,
        "model_version": _MEMBER_A.model_version,
        "vote_index": 0,
        "cost_micros": 100,
        "outcome": VOTE_OUTCOME_VOTED,
        "failure_code": "",
    }
    assert by_member[_MEMBER_B.model_version].payload == {
        "market_ticker": market.ticker,
        "provider": _MEMBER_B.provider,
        "model_version": _MEMBER_B.model_version,
        "vote_index": 1,
        "cost_micros": 200,
        "outcome": VOTE_OUTCOME_VOTED,
        "failure_code": "",
    }
    assert by_member[_MEMBER_C.model_version].payload == {
        "market_ticker": market.ticker,
        "provider": _MEMBER_C.provider,
        "model_version": _MEMBER_C.model_version,
        "vote_index": 2,
        "cost_micros": 300,
        "outcome": VOTE_OUTCOME_VOTED,
        "failure_code": "",
    }


def test_collect_model_votes_voted_event_payload_leaves_are_never_bool_or_float(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
) -> None:
    """Every numeric payload leaf is a true `int` -- never a `bool` (an `int`
    subclass) or a `float` (SPEC S6.1's package-wide float ban).
    """
    ledger = InMemoryForecastLedger()

    collect_model_votes(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        provider_factory=_routed_provider_factory(_all_voted_routes()),
    )

    events = ledger.events_by_type(PROVIDER_VOTE_COSTED_EVENT)
    for event in events:
        assert type(event.payload["vote_index"]) is int
        assert type(event.payload["cost_micros"]) is int
        assert isinstance(event.payload["outcome"], str)
        assert isinstance(event.payload["failure_code"], str)


# --- One member abstains: the abstainer still gets its own event -----------


def test_collect_model_votes_ledgers_abstained_outcome_for_an_abstaining_member(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
) -> None:
    """A member that explicitly abstains (SPEC S6.3) is excluded from the
    returned votes, but still gets its own `PROVIDER_VOTE_COSTED` event --
    `outcome="abstained"`, its own `cost_micros`, and an empty `failure_code`
    (it did answer; the answer was simply "no vote").
    """
    ledger = InMemoryForecastLedger()
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _member_provider(
            _MEMBER_A, probability_ppm=400_000, abstain=False
        ),
        _MEMBER_B.model_version: _member_provider(
            _MEMBER_B, probability_ppm=500_000, abstain=True, cost_micros=250_000
        ),
        _MEMBER_C.model_version: _member_provider(
            _MEMBER_C, probability_ppm=600_000, abstain=False
        ),
    }

    votes = collect_model_votes(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        provider_factory=_routed_provider_factory(routes),
    )

    assert len(votes) == 2

    events = ledger.events_by_type(PROVIDER_VOTE_COSTED_EVENT)
    assert len(events) == 3
    abstained_event = _events_by_provider(events)[_MEMBER_B.model_version]
    assert abstained_event.payload["outcome"] == VOTE_OUTCOME_ABSTAINED
    assert abstained_event.payload["cost_micros"] == 250_000
    assert abstained_event.payload["failure_code"] == ""
    assert abstained_event.payload["vote_index"] == 1


# --- One member is discarded: cost/failure_code come off the raised error --


def test_collect_model_votes_ledgers_discarded_outcome_with_error_cost_and_code(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
) -> None:
    """A discarded vote (a raised `ProviderVoteError`) gets its own event too
    -- `outcome="discarded"`, `cost_micros` and `failure_code` copied off the
    raised error, never off a fabricated default.
    """
    ledger = InMemoryForecastLedger()
    rejection = ProviderVoteError(
        "rejected",
        failure_code="malformed_vote_json",
        response_fingerprint="a" * 64,
        cost_micros=777,
    )
    routes: dict[str, ForecastProvider] = {
        _MEMBER_A.model_version: _member_provider(
            _MEMBER_A, probability_ppm=400_000, abstain=False
        ),
        _MEMBER_B.model_version: _FailingProvider(rejection),
        _MEMBER_C.model_version: _member_provider(
            _MEMBER_C, probability_ppm=600_000, abstain=False
        ),
    }

    votes = collect_model_votes(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        provider_factory=_routed_provider_factory(routes),
    )

    assert len(votes) == 2

    events = ledger.events_by_type(PROVIDER_VOTE_COSTED_EVENT)
    assert len(events) == 3
    discarded_event = _events_by_provider(events)[_MEMBER_B.model_version]
    assert discarded_event.payload["outcome"] == VOTE_OUTCOME_DISCARDED
    assert discarded_event.payload["cost_micros"] == 777
    assert discarded_event.payload["failure_code"] == "malformed_vote_json"
    assert discarded_event.payload["vote_index"] == 1


# --- No ledger wired: the vote-cost seam is never touched -------------------


def test_collect_model_votes_without_a_ledger_yields_identical_votes(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
) -> None:
    """`ledger=None` (the default) never changes which votes survive: the
    exact same provider routing produces the exact same votes whether or not
    a ledger is wired -- the new emission is pure additive instrumentation,
    never a behavior change (byte-identical to the pre-#281 no-ledger path).
    """
    routes = _all_voted_routes()

    votes_without_ledger = collect_model_votes(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        provider_factory=_routed_provider_factory(routes),
    )
    votes_with_ledger = collect_model_votes(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        ledger=InMemoryForecastLedger(),
        created_at=created_at,
        provider_factory=_routed_provider_factory(routes),
    )

    assert votes_without_ledger == votes_with_ledger
