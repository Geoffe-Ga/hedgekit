"""Tests for windbreak.forecast.triage (issue #23): two-stage triage + cost ledgering.

Pins the SPEC S8.4 two-stage triage contract: a cheap Stage-0 prior (`>=`
threshold, mutation-critical at the exact boundary), the STOP path that never
touches the expensive full pipeline, the PROCEED path where both stages'
costs accumulate, and a `TriageLedgerWriter` event trail that is
deterministic and carries exact-int payload leaves (never a float, per the
package-wide no-float convention `scripts/lint_no_floats.py` enforces).
`windbreak/forecast/triage.py` does not exist yet, so importing
`windbreak.forecast.triage` fails collection with `ModuleNotFoundError` --
the expected Gate 1 RED state for issue #23.

Three deliberate test-design choices, explained here because they shape every
test below:

Transport-reuse choice (`make_fake_vote_transport`)
    `tests/forecast/conftest.py`'s `make_fake_vote_transport` fixture is
    literally `FakeVoteTransport` itself (the fixture body is
    `return FakeVoteTransport`), so calling the injected factory with a
    custom `responses` tuple -- e.g. `make_fake_vote_transport(("520000",))`
    -- constructs a `FakeVoteTransport` seeded with that response, exactly as
    calling `FakeVoteTransport(("520000",))` directly would. This lets every
    triage test reuse the one conftest-provided double (a cheap-model Stage-0
    response is modeled as a bare integer ppm string) without inventing a new
    transport class or reaching into `tests.forecast.conftest` via a
    cross-module import that the project's rootdir-relative pytest import
    mode does not support.

Call-counting choice (`_CountingTransport`)
    `FakeVoteTransport` tracks its own call count on a private attribute, so
    a thin, local `_CountingTransport` wrapper (mirroring `test_cassettes.py`'s
    `_FakeTransport` pattern) is used wherever a test must assert Stage-0
    calls `transport.complete` *exactly once* -- a public, typed seam rather
    than reaching into the fake's internals.

Boundary-testing choice (the `>=` gate)
    `should_run_full_pipeline` gates on `abs(prior_ppm - baseline_ppm) >=
    triage_threshold_ppm`. A prior exactly `triage_threshold_ppm` away from
    baseline must proceed (not stop) -- the single ppm below that must not.
    Both edges are asserted explicitly below because a `>` vs `>=` mutant is
    otherwise invisible to any test that only exercises the interior of each
    region.

TriageModel injection choice (issue #192)
    The formerly-private `_TriageModel`/`_TRIAGE_MODEL` become a public
    `TriageModel` NamedTuple and a module-level `_DEFAULT_TRIAGE_MODEL`
    default, threaded as an optional `model=`/`triage_model=` keyword through
    `run_stage0_prior`/`run_triaged_pipeline`. The tests below pin: omitting
    the keyword yields a request byte-identical to today's pinned
    `("openai", "gpt-5-triage-mini")` model; an injected model changes the
    sent `LlmRequest`'s `provider`/`model_version`; and the STOP/PROCEED
    ledger payloads gain `triage_provider`/`triage_model_version` provenance
    fields. A final replay test drives Stage-0 under a `TriageModel` built
    from a `windbreak.config.schema.ModelRef`-shaped pair
    (`TriageModel(cfg.forecast.triage_model.provider,
    cfg.forecast.triage_model.model)`), proving the config -> forecast wiring
    shape without `windbreak.forecast.triage` itself ever importing
    `windbreak.config` (the SPEC S8.3 sandbox boundary).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.cassettes import (
    CassetteMissError,
    ForbiddenLiveTransport,
    RecordingCassette,
    ReplayCassette,
)
from windbreak.forecast.pipeline import (
    FORECAST_OUTPUT_DISCARDED_EVENT,
    InMemoryForecastLedger,
)
from windbreak.forecast.records import forecast_record_to_payload
from windbreak.forecast.triage import (
    PER_FORECAST_BUDGET_MICROS,
    TRIAGE_PROCEED_EVENT,
    TRIAGE_STOP_EVENT,
    TRIAGE_THRESHOLD_PPM,
    InMemoryTriageLedger,
    TriageModel,
    TriagePrior,
    run_stage0_prior,
    run_triaged_pipeline,
    should_run_full_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.cassettes import LlmRequest, LlmTransport
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

    #: See the module docstring's "Transport-reuse choice" note: the
    #: conftest-provided factory *is* `FakeVoteTransport`, callable with an
    #: optional custom `responses` tuple. Typed by its structural result
    #: (`LlmTransport`) rather than `object` so it composes with helpers here
    #: (e.g. `_CountingTransport`) that are themselves typed against the seam.
    FakeVoteTransportFactory = Callable[..., LlmTransport]

#: The `baseline` fixture's price (4500 pips) converted at the package's
#: fixed 100x pips-to-ppm factor (see `windbreak/forecast/pipeline.py`'s
#: `_baseline_probability_ppm`) -- named here so every gating assertion below
#: reads against the same, single source of truth.
_BASELINE_PPM = 450_000


class _CountingTransport:
    """A thin `LlmTransport` wrapper counting `complete` calls, for assertion.

    See the module docstring's "Call-counting choice" note for why this
    exists instead of reaching into `FakeVoteTransport`'s private counter.
    """

    def __init__(self, transport: LlmTransport) -> None:
        """Store the delegate transport and reset the call counter.

        Args:
            transport: The underlying transport to delegate every call to.
        """
        self._transport = transport
        self.call_count = 0

    def complete(self, request: LlmRequest) -> str:
        """Record one call, then delegate to the wrapped transport.

        Args:
            request: The completion request to forward unchanged.

        Returns:
            The wrapped transport's response.
        """
        self.call_count += 1
        return self._transport.complete(request)


class _RequestRecordingTransport:
    """A minimal `LlmTransport` double recording every full `LlmRequest`.

    Unlike `_CountingTransport` (which only counts calls), this double keeps
    every `LlmRequest` it was called with, so a test can assert on the exact
    `provider`/`model_version`/`prompt` a call sent -- pinning `TriageModel`
    injection (issue #192).
    """

    def __init__(self, response: str = "500000") -> None:
        """Store the fixed response every `complete` call returns.

        Args:
            response: The fixed Stage-0 response text (a bare integer ppm
                string) every call returns.
        """
        self._response = response
        self.calls: list[LlmRequest] = []

    def complete(self, request: LlmRequest) -> str:
        """Record `request`, then return the fixed canned response.

        Args:
            request: The completion request to record.

        Returns:
            The fixed response text.
        """
        self.calls.append(request)
        return self._response


# --- Constants: SPEC-mandated exact values ---------------------------------------


def test_triage_threshold_and_budget_constants_have_expected_values() -> None:
    """Pin the SPEC S8.4 Stage-0 threshold and per-forecast budget constants."""
    assert TRIAGE_THRESHOLD_PPM == 50_000
    assert PER_FORECAST_BUDGET_MICROS == 3_000_000


# --- Stage-0 prior: parsing, cost, and the single-call contract ------------------


def test_stage0_prior_parses_integer_response_into_prior(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    make_fake_vote_transport: FakeVoteTransportFactory,
) -> None:
    """A bare-integer Stage-0 response becomes an exact, cheap `TriagePrior`."""
    transport = _CountingTransport(make_fake_vote_transport(("520000",)))

    prior = run_stage0_prior(market, baseline, transport=transport)

    assert prior == TriagePrior(prior_ppm=520_000, cost_micros=60_000)
    assert 0 < prior.cost_micros <= PER_FORECAST_BUDGET_MICROS // 50
    assert transport.call_count == 1


@pytest.mark.parametrize("bad_response", ["0.52", "maybe"])
def test_stage0_prior_non_integer_response_raises_value_error(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    make_fake_vote_transport: FakeVoteTransportFactory,
    bad_response: str,
) -> None:
    """A non-integer Stage-0 response fails loudly rather than silently defaulting."""
    transport = make_fake_vote_transport((bad_response,))

    with pytest.raises(ValueError):
        run_stage0_prior(market, baseline, transport=transport)


@pytest.mark.parametrize("bad_response", ["1000001", "-1"])
def test_stage0_prior_out_of_range_response_raises_value_error(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    make_fake_vote_transport: FakeVoteTransportFactory,
    bad_response: str,
) -> None:
    """A Stage-0 response outside `[0, 1_000_000]` fails loudly."""
    transport = make_fake_vote_transport((bad_response,))

    with pytest.raises(ValueError):
        run_stage0_prior(market, baseline, transport=transport)


@pytest.mark.parametrize("response", ["0", "1000000"])
def test_stage0_prior_accepts_inclusive_range_edges(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    make_fake_vote_transport: FakeVoteTransportFactory,
    response: str,
) -> None:
    """The inclusive `[0, 1_000_000]` range edges both parse successfully.

    A `_MIN_PRIOR_PPM <= value` or `value <= _MAX_PRIOR_PPM` mutant that
    tightens either bound to `<` would reject one of these two edges.
    """
    transport = make_fake_vote_transport((response,))

    prior = run_stage0_prior(market, baseline, transport=transport)

    assert prior.prior_ppm == int(response)


# --- TriageModel: NamedTuple shape and default (issue #192) ----------------------


def test_triage_model_is_a_named_tuple_with_provider_and_model_version_fields() -> None:
    """`TriageModel` exposes exactly `{provider, model_version}` -- mirroring
    the formerly-private `_TriageModel` field names verbatim.
    """
    model = TriageModel("openai", "gpt-5-triage-mini")

    assert model.provider == "openai"
    assert model.model_version == "gpt-5-triage-mini"
    assert isinstance(model, tuple)
    assert tuple(model) == ("openai", "gpt-5-triage-mini")


def test_default_triage_model_module_constant_matches_the_pinned_stage0_model() -> None:
    """`_DEFAULT_TRIAGE_MODEL` is exactly `TriageModel("openai",
    "gpt-5-triage-mini")` -- the pre-#192 pinned Stage-0 model, unchanged.
    """
    import windbreak.forecast.triage as triage_module

    assert (
        TriageModel("openai", "gpt-5-triage-mini")
        == triage_module._DEFAULT_TRIAGE_MODEL
    )


# --- run_stage0_prior: model= injection (issue #192) ------------------------------


def test_run_stage0_prior_without_model_kwarg_uses_the_pinned_default(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Omitting `model=` entirely sends a request on the pinned default
    `("openai", "gpt-5-triage-mini")` model -- byte-identical to before
    `model=` existed as a keyword at all.
    """
    transport = _RequestRecordingTransport()

    run_stage0_prior(market, baseline, transport=transport)

    assert transport.calls[0].provider == "openai"
    assert transport.calls[0].model_version == "gpt-5-triage-mini"


def test_run_stage0_prior_explicit_default_model_is_byte_identical_to_omitted(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Passing `model=TriageModel("openai", "gpt-5-triage-mini")` explicitly
    sends the identical request as omitting `model=` altogether.
    """
    transport_omitted = _RequestRecordingTransport()
    transport_explicit = _RequestRecordingTransport()

    run_stage0_prior(market, baseline, transport=transport_omitted)
    run_stage0_prior(
        market,
        baseline,
        transport=transport_explicit,
        model=TriageModel("openai", "gpt-5-triage-mini"),
    )

    assert transport_omitted.calls[0] == transport_explicit.calls[0]
    assert (
        transport_omitted.calls[0].request_hash()
        == transport_explicit.calls[0].request_hash()
    )


def test_run_stage0_prior_injected_model_changes_the_sent_request(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """An injected `model=` changes the sent `LlmRequest`'s `provider` and
    `model_version` away from the pinned default.
    """
    transport = _RequestRecordingTransport()
    custom_model = TriageModel("anthropic", "claude-triage-custom")

    run_stage0_prior(market, baseline, transport=transport, model=custom_model)

    assert transport.calls[0].provider == "anthropic"
    assert transport.calls[0].model_version == "claude-triage-custom"


def test_run_stage0_prior_injected_model_yields_a_different_request_hash(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """An injected, non-default model produces a request hash distinct from
    the default model's request -- the model provenance really is hashed into
    the request.
    """
    transport_default = _RequestRecordingTransport()
    transport_custom = _RequestRecordingTransport()

    run_stage0_prior(market, baseline, transport=transport_default)
    run_stage0_prior(
        market,
        baseline,
        transport=transport_custom,
        model=TriageModel("anthropic", "claude-triage-custom"),
    )

    assert (
        transport_default.calls[0].request_hash()
        != transport_custom.calls[0].request_hash()
    )


def test_run_stage0_prior_still_parses_the_response_with_an_injected_model(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Injecting a custom model does not change Stage-0's response-parsing
    contract: the returned `TriagePrior` still carries the parsed ppm and the
    fixed Stage-0 cost.
    """
    transport = _RequestRecordingTransport("520000")
    custom_model = TriageModel("anthropic", "claude-triage-custom")

    prior = run_stage0_prior(market, baseline, transport=transport, model=custom_model)

    assert prior == TriagePrior(prior_ppm=520_000, cost_micros=60_000)


# --- run_triaged_pipeline: triage_model= threading (issue #192) ------------------


def test_run_triaged_pipeline_threads_triage_model_into_the_stage0_request(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """`run_triaged_pipeline`'s `triage_model=` reaches Stage-0's sent request,
    on both the STOP and PROCEED paths.
    """
    transport = _RequestRecordingTransport("460000")
    custom_model = TriageModel("anthropic", "claude-triage-custom")

    run_triaged_pipeline(
        market,
        baseline,
        triage_transport=transport,
        full_transport=ForbiddenLiveTransport(),
        ledger=InMemoryTriageLedger(),
        created_at=created_at,
        research_tools=research_tools,
        triage_model=custom_model,
    )

    assert transport.calls[0].provider == "anthropic"
    assert transport.calls[0].model_version == "claude-triage-custom"


def test_stop_event_payload_carries_the_default_triage_model_provenance(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """With `triage_model=` omitted, a STOP event's payload names the pinned
    default model's provider and version.
    """
    ledger = InMemoryTriageLedger()

    run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("460000",)),
        full_transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        research_tools=research_tools,
    )

    payload = ledger.events_by_type(TRIAGE_STOP_EVENT)[0].payload
    assert payload["triage_provider"] == "openai"
    assert payload["triage_model_version"] == "gpt-5-triage-mini"
    assert json.dumps(payload)
    assert all(isinstance(v, int | str | bool) for v in payload.values())


def test_stop_event_payload_carries_an_injected_triage_model_provenance(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """An injected `triage_model=` is reflected in the STOP event's payload."""
    ledger = InMemoryTriageLedger()
    custom_model = TriageModel("anthropic", "claude-triage-custom")

    run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("460000",)),
        full_transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        research_tools=research_tools,
        triage_model=custom_model,
    )

    payload = ledger.events_by_type(TRIAGE_STOP_EVENT)[0].payload
    assert payload["triage_provider"] == "anthropic"
    assert payload["triage_model_version"] == "claude-triage-custom"


def test_proceed_event_payload_carries_an_injected_triage_model_provenance(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """An injected `triage_model=` is also reflected in the PROCEED event's
    payload.
    """
    ledger = InMemoryTriageLedger()
    custom_model = TriageModel("anthropic", "claude-triage-custom")

    run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("600000",)),
        full_transport=make_fake_vote_transport(),
        ledger=ledger,
        created_at=created_at,
        research_tools=research_tools,
        triage_model=custom_model,
    )

    payload = ledger.events_by_type(TRIAGE_PROCEED_EVENT)[0].payload
    assert payload["triage_provider"] == "anthropic"
    assert payload["triage_model_version"] == "claude-triage-custom"
    assert json.dumps(payload)
    assert all(isinstance(v, int | str | bool) for v in payload.values())


# --- Config -> forecast wiring replay (issue #192) --------------------------------


def test_run_stage0_prior_replays_over_a_config_shaped_triage_model(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    tmp_path: Path,
) -> None:
    """Stage-0 runs, record-then-replay, under a `TriageModel` built from a
    `ModelRef`-shaped `(provider, model)` pair -- the exact structural
    translation an operator-facing config->forecast wiring layer would apply
    (`TriageModel(cfg.forecast.triage_model.provider,
    cfg.forecast.triage_model.model)`) -- proving the model is a plain
    structural NamedTuple `windbreak.forecast.triage` never needs
    `windbreak.config` to build.

    The recorded/replayed cassette is built dynamically against a fake
    transport (mirroring this module's own
    `test_proceed_path_cassette_replay_matches_recording`), rather than
    reloaded from the committed `tests/fixtures/forecast/
    triage_stage0_cassette.json` placeholder fixture: that file's key is a
    human-readable placeholder (never a real 64-char request hash, exactly
    like `futuresearch_cassette.json`), so it backs only the
    hash-independent structural checks in `test_cassettes.py`'s sibling
    suite, not a byte-precise replay hit here.
    """
    # A minimal, local stand-in for a `windbreak.config.schema.ModelRef` pair
    # (`provider`, `model`) -- imported nowhere from `windbreak.config`.
    config_triage_model_provider = "openai"
    config_triage_model_version = "gpt-5-triage-mini"
    model = TriageModel(config_triage_model_provider, config_triage_model_version)

    cassette_path = tmp_path / "triage_stage0.json"
    recorder = RecordingCassette(
        transport=_RequestRecordingTransport("510000"), path=cassette_path
    )
    recorded = run_stage0_prior(market, baseline, transport=recorder, model=model)

    replay = ReplayCassette.from_path(cassette_path)
    replayed = run_stage0_prior(market, baseline, transport=replay, model=model)

    assert replayed == recorded == TriagePrior(prior_ppm=510_000, cost_micros=60_000)


def test_triage_stage0_cassette_fixture_loads_without_error(fixture_dir: Path) -> None:
    """The committed `triage_stage0_cassette.json` fixture parses without
    error via `ReplayCassette.from_path` -- a structural smoke test mirroring
    `test_from_path_loads_committed_fixture_without_error` in
    `tests/forecast/providers/test_http_cassettes.py`.
    """
    ReplayCassette.from_path(fixture_dir / "triage_stage0_cassette.json")


def test_triage_stage0_cassette_fixture_misses_on_any_real_request(
    fixture_dir: Path,
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
) -> None:
    """The fixture's key is a human-readable placeholder, never a real
    64-char `request_hash()`, so any real Stage-0 request is a guaranteed
    miss against it -- proving the file cannot be mistaken for a live replay
    source.
    """
    replay = ReplayCassette.from_path(fixture_dir / "triage_stage0_cassette.json")

    with pytest.raises(CassetteMissError):
        run_stage0_prior(market, baseline, transport=replay)


# --- should_run_full_pipeline: gating boundaries (mutation-critical `>=`) --------


@pytest.mark.parametrize(
    ("prior_ppm", "operator_flagged", "refresh_triggered", "expected"),
    [
        (499_999, False, False, False),  # diff 49_999 < threshold: stop.
        (500_000, False, False, True),  # diff exactly 50_000: the `>=` edge.
        (400_000, False, False, True),  # symmetric high-side diff of 50_000.
        (460_000, True, False, True),  # below threshold, but operator-flagged.
        (460_000, False, True, True),  # below threshold, but refresh-triggered.
    ],
)
def test_should_run_full_pipeline_boundaries(
    prior_ppm: int,
    operator_flagged: bool,
    refresh_triggered: bool,
    expected: bool,
) -> None:
    """The pure gate matches `>=` threshold, operator-flag, and refresh cases."""
    result = should_run_full_pipeline(
        prior_ppm,
        _BASELINE_PPM,
        triage_threshold_ppm=TRIAGE_THRESHOLD_PPM,
        operator_flagged=operator_flagged,
        refresh_triggered=refresh_triggered,
    )

    assert result is expected


# --- STOP path: end-to-end, never touching the full pipeline --------------------


def test_stop_path_never_runs_full_pipeline_and_records_triage_only(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """A within-band prior stops before the full pipeline and ledgers a STOP.

    `full_transport=ForbiddenLiveTransport()` completing *without* raising
    `LiveCallForbiddenError` is itself the structural proof the expensive
    pipeline never ran.
    """
    ledger = InMemoryTriageLedger()

    record = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("460000",)),
        full_transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        research_tools=research_tools,
    )

    assert record.triage_stage == "triage_only"
    assert record.eligible_for_live is False
    assert record.research_cost_micros == 60_000
    assert record.model_votes == ()
    assert record.citations == ()
    assert record.probability_ppm == 460_000
    assert record.ci_low_ppm == 460_000
    assert record.ci_high_ppm == 460_000

    stop_events = ledger.events_by_type(TRIAGE_STOP_EVENT)
    assert len(stop_events) == 1
    assert ledger.events_by_type(TRIAGE_PROCEED_EVENT) == ()
    payload = stop_events[0].payload
    assert payload["market_ticker"] == market.ticker
    assert payload["prior_ppm"] == 460_000
    assert payload["baseline_ppm"] == 450_000
    assert payload["triage_threshold_ppm"] == 50_000
    assert payload["operator_flagged"] is False
    assert payload["refresh_triggered"] is False
    assert payload["triage_cost_micros"] == 60_000
    assert json.dumps(payload)
    assert all(isinstance(v, int | str | bool) for v in payload.values())


# --- PROCEED path: end-to-end, both stages' costs accumulate ---------------------


def test_proceed_path_runs_full_pipeline_and_accumulates_both_costs(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """A far-from-baseline prior proceeds to the full pipeline and ledgers a PROCEED."""
    ledger = InMemoryTriageLedger()

    record = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("600000",)),
        full_transport=make_fake_vote_transport(),
        ledger=ledger,
        created_at=created_at,
        research_tools=research_tools,
    )

    assert record.triage_stage == "full"
    assert record.eligible_for_live is True
    assert record.research_cost_micros == 3_000_000 + 60_000
    assert len(record.model_votes) == 3

    proceed_events = ledger.events_by_type(TRIAGE_PROCEED_EVENT)
    assert len(proceed_events) == 1
    assert ledger.events_by_type(TRIAGE_STOP_EVENT) == ()
    payload = proceed_events[0].payload
    assert payload["market_ticker"] == market.ticker
    assert payload["operator_flagged"] is False
    assert payload["refresh_triggered"] is False
    assert payload["triage_cost_micros"] == 60_000
    assert payload["full_cost_micros"] == 3_000_000
    assert payload["total_research_cost_micros"] == 3_060_000
    assert json.dumps(payload)
    assert all(isinstance(v, int | str | bool) for v in payload.values())


def test_operator_flagged_forces_full_pipeline_despite_below_threshold_prior(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """`operator_flagged=True` forces the full pipeline inside the triage band."""
    ledger = InMemoryTriageLedger()

    record = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("460000",)),
        full_transport=make_fake_vote_transport(),
        ledger=ledger,
        created_at=created_at,
        operator_flagged=True,
        research_tools=research_tools,
    )

    assert record.triage_stage == "full"
    assert record.eligible_for_live is True
    proceed_events = ledger.events_by_type(TRIAGE_PROCEED_EVENT)
    assert proceed_events[0].payload["operator_flagged"] is True


def test_refresh_triggered_forces_full_pipeline_despite_below_threshold_prior(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """`refresh_triggered=True` forces the full pipeline inside the triage band."""
    ledger = InMemoryTriageLedger()

    record = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("460000",)),
        full_transport=make_fake_vote_transport(),
        ledger=ledger,
        created_at=created_at,
        refresh_triggered=True,
        research_tools=research_tools,
    )

    assert record.triage_stage == "full"
    assert record.eligible_for_live is True
    proceed_events = ledger.events_by_type(TRIAGE_PROCEED_EVENT)
    assert proceed_events[0].payload["refresh_triggered"] is True


# --- Determinism: identical inputs, fresh fakes and ledgers ----------------------


def test_run_triaged_pipeline_is_byte_deterministic_for_identical_inputs(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """Two proceed-path runs with fresh fakes/ledgers produce identical output."""
    ledger_a = InMemoryTriageLedger()
    ledger_b = InMemoryTriageLedger()

    record_a = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("600000",)),
        full_transport=make_fake_vote_transport(),
        ledger=ledger_a,
        created_at=created_at,
        research_tools=research_tools,
    )
    record_b = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("600000",)),
        full_transport=make_fake_vote_transport(),
        ledger=ledger_b,
        created_at=created_at,
        research_tools=research_tools,
    )

    assert record_a == record_b
    payload_a = json.dumps(forecast_record_to_payload(record_a), sort_keys=True)
    payload_b = json.dumps(forecast_record_to_payload(record_b), sort_keys=True)
    assert payload_a == payload_b

    for event_type in (TRIAGE_STOP_EVENT, TRIAGE_PROCEED_EVENT):
        assert ledger_a.events_by_type(event_type) == ledger_b.events_by_type(
            event_type
        )


def test_run_triaged_pipeline_stop_path_is_byte_deterministic_for_identical_inputs(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """Two stop-path runs with fresh fakes/ledgers produce identical output.

    The PROCEED-path determinism test above exercises `run_pipeline`'s own id
    derivation; `_triage_forecast_id`'s sha256-over-canonical-JSON id
    derivation is unique to the STOP path and would otherwise never be
    determinism-checked.
    """
    ledger_a = InMemoryTriageLedger()
    ledger_b = InMemoryTriageLedger()

    record_a = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("460000",)),
        full_transport=ForbiddenLiveTransport(),
        ledger=ledger_a,
        created_at=created_at,
        research_tools=research_tools,
    )
    record_b = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("460000",)),
        full_transport=ForbiddenLiveTransport(),
        ledger=ledger_b,
        created_at=created_at,
        research_tools=research_tools,
    )

    assert record_a == record_b
    payload_a = json.dumps(forecast_record_to_payload(record_a), sort_keys=True)
    payload_b = json.dumps(forecast_record_to_payload(record_b), sort_keys=True)
    assert payload_a == payload_b
    assert ledger_a.events_by_type(TRIAGE_STOP_EVENT) == ledger_b.events_by_type(
        TRIAGE_STOP_EVENT
    )


# --- Tracer invariant: triage composes with the cassette contract ---------------


def test_proceed_path_cassette_replay_matches_recording(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
    tmp_path: Path,
) -> None:
    """Recording `full_transport`, then replaying it, reproduces the same record.

    Proves triage's Stage-0 short-circuit does not break the full pipeline's
    existing record/replay cassette contract on the proceed path.
    """
    cassette_path = tmp_path / "votes.json"
    recorder = RecordingCassette(
        transport=make_fake_vote_transport(), path=cassette_path
    )
    recorded = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("600000",)),
        full_transport=recorder,
        ledger=InMemoryTriageLedger(),
        created_at=created_at,
        research_tools=research_tools,
    )

    replay = ReplayCassette.from_path(cassette_path)
    replayed = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("600000",)),
        full_transport=replay,
        ledger=InMemoryTriageLedger(),
        created_at=created_at,
        research_tools=research_tools,
    )

    assert replayed == recorded


# --- Discard-ledger threading (issue #98): separate seam from `ledger` -----------


def test_stop_path_never_touches_discard_ledger(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """A within-band prior stops before the full pipeline, so the wired
    `discard_ledger` -- a seam that only the full pipeline's vote-discard
    bookkeeping ever writes to -- must stay empty; `full_transport=
    ForbiddenLiveTransport()` completing without raising is the structural
    proof the full pipeline (and therefore `collect_model_votes`) never ran.
    """
    ledger = InMemoryTriageLedger()
    discard_ledger = InMemoryForecastLedger()

    record = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("460000",)),
        full_transport=ForbiddenLiveTransport(),
        ledger=ledger,
        discard_ledger=discard_ledger,
        created_at=created_at,
        research_tools=research_tools,
    )

    assert record.triage_stage == "triage_only"
    assert discard_ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT) == ()


def test_proceed_path_with_clean_votes_records_no_discard_events(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools: ResearchTools,
) -> None:
    """A far-from-baseline prior proceeds to the full pipeline; with every
    full-pipeline vote clean, the wired `discard_ledger` records zero
    `FORECAST_OUTPUT_DISCARDED` events and the record is a normal, live-
    eligible `"full"`-stage record. Kills an "always record a discard event"
    mutant that a proceed-path test without a clean-vote case would miss.
    """
    ledger = InMemoryTriageLedger()
    discard_ledger = InMemoryForecastLedger()

    record = run_triaged_pipeline(
        market,
        baseline,
        triage_transport=make_fake_vote_transport(("600000",)),
        full_transport=make_fake_vote_transport(),
        ledger=ledger,
        discard_ledger=discard_ledger,
        created_at=created_at,
        research_tools=research_tools,
    )

    assert record.triage_stage == "full"
    assert record.eligible_for_live is True
    assert discard_ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT) == ()
