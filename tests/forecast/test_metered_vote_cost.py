"""Tests for issue #451: a live vote is charged for tokens, not per attempt.

Issue #399 made a successful live vote book *something* -- its provider's flat
per-attempt list price. That was the sanctioned stopgap, and it left the real
defect standing: the figure booked is a constant, so `per_forecast_micros` and
`per_day_micros` bound a *count of attempts* rather than spend. A vote over a
long research context can cost a multiple of its list price while the meter
records exactly the list price and no breach fires.

Every test here drives the real seams -- the real
`windbreak.forecast.providers.openai.OpenAiChatTransport` and
`windbreak.forecast.providers.anthropic.AnthropicMessagesTransport` over a
doubled `HttpTransport`, the real `FixtureVoteProvider`, the real
`RetryingProvider`, the real `run_pipeline`, the real record/replay cassettes
-- so a costing claim is proven through the composition that ships rather than
beside it.

**Fixture figures are deliberately pairwise distinct.** The list price, the
metered charge, the fail-closed unmetered charge and the unknown-provider
fallback are four different numbers in every test below. A costing test whose
fixtures collide cannot tell which rule produced the figure it asserts, and
this is precisely a change that swaps one rule for another.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from windbreak.forecast.budget import (
    BUDGET_FORECAST_EXCEEDED_EVENT,
    BUDGET_SPEND_RECORDED_EVENT,
    DEFAULT_MODEL_RATE_TABLE,
    DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS,
    DEFAULT_UNMETERED_RESPONSE_MICROS,
    InMemoryBudgetLedger,
    ModelRateTable,
    ModelTokenRate,
    PerForecastBudgetExceededError,
    ProviderPriceTable,
    ResearchBudget,
    TokenUsage,
)
from windbreak.forecast.cassettes import (
    Completion,
    ForbiddenLiveTransport,
    LlmRequest,
    RecordingCassette,
    ReplayCassette,
)
from windbreak.forecast.pipeline import run_pipeline
from windbreak.forecast.providers.anthropic import AnthropicMessagesTransport
from windbreak.forecast.providers.base import (
    EnsembleMember,
    ProviderCostOverrunError,
    build_vote_prompt,
)
from windbreak.forecast.providers.fixture import FixtureVoteProvider
from windbreak.forecast.providers.http_cassettes import HttpResponse
from windbreak.forecast.providers.openai import OpenAiChatTransport
from windbreak.forecast.providers.retry import RetryingProvider, RetryPolicy

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.providers.base import (
        EnsembleMemberLike,
        ForecastProvider,
    )
    from windbreak.forecast.providers.http_cassettes import HttpRequest
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

# --- Pinned figures ---------------------------------------------------------------

#: The ensemble member every unit-level test here votes as.
_MEMBER = EnsembleMember("openai", "a-metered-model", "2024-09-30")

#: A second member on a *different* model, so a per-model rate can be shown to
#: be per-model rather than per-provider.
_OTHER_MEMBER = EnsembleMember("openai", "a-cheaper-metered-model", "2024-05-31")

#: A schema-valid vote body, reused verbatim by every canned response.
_VOTE_JSON = (
    '{"probability_ppm": 500000, "rationale_summary": "steady", "abstain": false}'
)

#: The per-attempt list price -- since issue #451 the affordability *estimate*
#: the pre-gate runs on, never what a completed vote is charged.
_LIST_PRICE_MICROS = 200_000

#: The fallback estimate for a provider absent from the price table.
_UNKNOWN_PRICE_MICROS = 1_000_000

#: The fail-closed charge for a response whose cost cannot be derived.
_UNMETERED_MICROS = 777_000

#: `_MEMBER`'s configured rates, in micros per million tokens.
_INPUT_RATE = 1_250_000
_OUTPUT_RATE = 10_000_000

#: `_OTHER_MEMBER`'s configured rates -- five times cheaper on both legs.
_OTHER_INPUT_RATE = 250_000
_OTHER_OUTPUT_RATE = 2_000_000

#: The rate table every unit-level test here meters through.
_RATE_TABLE = ModelRateTable(
    rates={
        _MEMBER.model_version: ModelTokenRate(
            model_version=_MEMBER.model_version,
            input_micros_per_million_tokens=_INPUT_RATE,
            output_micros_per_million_tokens=_OUTPUT_RATE,
        ),
        _OTHER_MEMBER.model_version: ModelTokenRate(
            model_version=_OTHER_MEMBER.model_version,
            input_micros_per_million_tokens=_OTHER_INPUT_RATE,
            output_micros_per_million_tokens=_OTHER_OUTPUT_RATE,
        ),
    },
    unmetered_micros=_UNMETERED_MICROS,
)

#: A modest vote: 4_000 input tokens, 500 output tokens. At `_MEMBER`'s rates
#: that is ``4_000 * 1_250_000 + 500 * 10_000_000 == 10_000_000_000`` over a
#: million tokens-per-million, i.e. 10_000 micros -- an order of magnitude
#: *below* the list price, so a test asserting it cannot pass on the estimate.
_MODEST_USAGE = TokenUsage(input_tokens=4_000, output_tokens=500)
_MODEST_METERED_MICROS = 10_000

#: The same usage at `_OTHER_MEMBER`'s five-times-cheaper rates.
_MODEST_METERED_MICROS_OTHER = 2_000

#: A superforecaster scaffold that inlined a megabyte of fetched research
#: really does bill this much: 1_000_000 input tokens and 900 output tokens.
#: ``1_000_000 * 1_250_000 + 900 * 10_000_000 == 1_259_000_000_000`` over a
#: million, i.e. 1_259_000 micros -- more than six times the list price, and
#: the exact overage issue #451 says the old meter could not see.
_HUGE_USAGE = TokenUsage(input_tokens=1_000_000, output_tokens=900)
_HUGE_METERED_MICROS = 1_259_000

#: A retry policy generous in every dimension, so only the behaviour under test
#: can fire.
_MAX_ATTEMPTS = 3
_TOTAL_DEADLINE_MS = 30_000
_BACKOFF_BASE_MS = 1_000
_PERMISSIVE_MAX_COST_MICROS = 100_000_000

#: `windbreak.forecast.pipeline`'s private `_RESEARCH_COST_MICROS` stub cost for
#: a full run -- named locally (it is private), mirroring the convention in
#: `tests/forecast/test_live_vote_cost_booking.py`.
_FULL_RUN_RESEARCH_COST_MICROS = 3_000_000


class _CannedHttpTransport:
    """An `HttpTransport` double returning one fixed 200 response body."""

    def __init__(self, body: str) -> None:
        """Store the canned response body.

        Args:
            body: The raw response body text every send returns.
        """
        self._body = body
        self.sends = 0

    def send(self, request: HttpRequest) -> HttpResponse:
        """Return the canned 200 response, ignoring the request.

        Args:
            request: The (unused) outbound HTTP request.

        Returns:
            A 200 `HttpResponse` carrying the canned body.
        """
        del request
        self.sends += 1
        return HttpResponse(status_code=200, body=self._body)


class _CannedLlmTransport:
    """An `LlmTransport` double returning one fixed `Completion`."""

    def __init__(self, completion: Completion) -> None:
        """Store the completion every call returns.

        Args:
            completion: The fixed completion to return.
        """
        self._completion = completion

    def complete(self, request: object) -> Completion:
        """Return the fixed completion, ignoring the request.

        Args:
            request: The (unused) completion request.

        Returns:
            The stored `Completion`, verbatim.
        """
        del request
        return self._completion


def _openai_body(usage: object) -> str:
    """Build an OpenAI chat-completions envelope carrying ``usage`` verbatim.

    Args:
        usage: The value to place under the envelope's ``usage`` key. Passing a
            malformed value is how the fail-closed arm is exercised.

    Returns:
        The JSON response body text.
    """
    return json.dumps(
        {"choices": [{"message": {"content": _VOTE_JSON}}], "usage": usage}
    )


def _openai_body_without_usage() -> str:
    """Build an OpenAI envelope with no ``usage`` key at all.

    Returns:
        The JSON response body text.
    """
    return json.dumps({"choices": [{"message": {"content": _VOTE_JSON}}]})


def _anthropic_body(usage: object) -> str:
    """Build an Anthropic Messages envelope carrying ``usage`` verbatim.

    Args:
        usage: The value to place under the envelope's ``usage`` key.

    Returns:
        The JSON response body text.
    """
    return json.dumps(
        {"content": [{"type": "text", "text": _VOTE_JSON}], "usage": usage}
    )


def _clock() -> tuple[object, object]:
    """Return a monotonic-ms / sleep-ms pair that never really sleeps.

    Returns:
        The `(monotonic_ms, sleep_ms)` callables.
    """
    return (lambda: 0, lambda _ms: None)


def _live_wrapped(
    inner: ForecastProvider,
    *,
    provider_name: str = "openai",
    rate_table: ModelRateTable | None = None,
    max_cost_micros: int = _PERMISSIVE_MAX_COST_MICROS,
) -> RetryingProvider:
    """Wrap ``inner`` exactly as the live composition root wraps a vote provider.

    Args:
        inner: The provider to wrap.
        provider_name: The name the pre-gate estimates through the price table
            (keyword-only).
        rate_table: The rate table successes are metered through, defaulting to
            `_RATE_TABLE` (keyword-only).
        max_cost_micros: The per-member affordability ceiling (keyword-only).

    Returns:
        The live-shaped `RetryingProvider`.
    """
    monotonic_ms, sleep_ms = _clock()
    return RetryingProvider(
        inner,
        provider_name=provider_name,
        policy=RetryPolicy(
            max_attempts=_MAX_ATTEMPTS,
            total_deadline_ms=_TOTAL_DEADLINE_MS,
            backoff_base_ms=_BACKOFF_BASE_MS,
            max_cost_micros=max_cost_micros,
        ),
        price_table=ProviderPriceTable(
            prices_micros={"openai": _LIST_PRICE_MICROS},
            unknown_provider_price_micros=_UNKNOWN_PRICE_MICROS,
        ),
        rate_table=rate_table if rate_table is not None else _RATE_TABLE,
        monotonic_ms=monotonic_ms,
        sleep_ms=sleep_ms,
    )


def _openai_vote(body: str, *, member: EnsembleMember = _MEMBER) -> RetryingProvider:
    """Build the full live vote stack over a canned OpenAI response body.

    Args:
        body: The canned response body the HTTP double returns.
        member: The ensemble member the vote is cast as (keyword-only).

    Returns:
        The live-shaped provider stack.
    """
    return _live_wrapped(
        FixtureVoteProvider(OpenAiChatTransport(_CannedHttpTransport(body)), member)
    )


# --- Criterion 1: the charge is derived from reported usage -----------------------


def test_a_vote_reporting_a_million_input_tokens_is_charged_for_them(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """The whole of issue #451 in one assertion.

    The response says a million input tokens were billed. Before this issue the
    meter booked exactly the flat list price regardless; now it books what the
    tokens cost -- six times more. Both figures are named so the assertion can
    only pass on the right one.
    """
    provider = _openai_vote(_openai_body(_usage_block(_HUGE_USAGE)))

    result = provider.forecast(market, baseline, 0, ())

    assert result.cost_micros == _HUGE_METERED_MICROS
    assert result.cost_micros != _LIST_PRICE_MICROS


def test_a_modest_vote_is_charged_less_than_its_list_price(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Metering cuts both ways: a cheap vote books below the flat estimate.

    The companion to the test above, and the one that proves the charge is
    *derived* rather than clamped: a rule that merely took the larger of the
    two figures would pass there and fail here.
    """
    provider = _openai_vote(_openai_body(_usage_block(_MODEST_USAGE)))

    result = provider.forecast(market, baseline, 0, ())

    assert result.cost_micros == _MODEST_METERED_MICROS
    assert result.cost_micros < _LIST_PRICE_MICROS


def test_the_anthropic_adapter_meters_its_own_usage_key_names(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Anthropic reports `input_tokens`/`output_tokens`, and is read that way.

    The two vendors name the same two quantities differently. Reading the wrong
    pair of keys yields no usage at all, which fails closed to
    `_UNMETERED_MICROS` -- a figure this assertion names so it cannot be
    mistaken for a metered one.
    """
    provider = _live_wrapped(
        FixtureVoteProvider(
            AnthropicMessagesTransport(
                _CannedHttpTransport(
                    _anthropic_body(
                        {
                            "input_tokens": _MODEST_USAGE.input_tokens,
                            "output_tokens": _MODEST_USAGE.output_tokens,
                        }
                    )
                )
            ),
            _MEMBER,
        )
    )

    result = provider.forecast(market, baseline, 0, ())

    assert result.cost_micros == _MODEST_METERED_MICROS
    assert result.cost_micros != _UNMETERED_MICROS


def test_the_same_usage_costs_different_amounts_on_different_models(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Rates are per *model*, not per provider.

    Both members below name the provider ``openai``; only their model versions
    differ. A table keyed by provider would charge them identically.
    """
    body = _openai_body(_usage_block(_MODEST_USAGE))

    expensive = _openai_vote(body).forecast(market, baseline, 0, ())
    cheap = _openai_vote(body, member=_OTHER_MEMBER).forecast(market, baseline, 0, ())

    assert expensive.cost_micros == _MODEST_METERED_MICROS
    assert cheap.cost_micros == _MODEST_METERED_MICROS_OTHER


# --- Criterion 2: integer rates, explicit rounding --------------------------------


def test_a_partial_micro_is_rounded_up_against_the_operator() -> None:
    """A remainder is resolved by overstating the cost, never understating it.

    One input token at 1_250_000 micros per million tokens is 1.25 micros --
    unrepresentable. The guard must believe 2, not 1: a ceiling binds a hair
    early, a floor lets a hair of real spend past every single vote.
    """
    one_token = ModelRateTable(
        rates={
            _MEMBER.model_version: ModelTokenRate(
                model_version=_MEMBER.model_version,
                input_micros_per_million_tokens=_INPUT_RATE,
                output_micros_per_million_tokens=_OUTPUT_RATE,
            )
        },
        unmetered_micros=_UNMETERED_MICROS,
    )

    charged = one_token.micros_for(
        model_version=_MEMBER.model_version,
        usage=TokenUsage(input_tokens=1, output_tokens=0),
    )

    assert charged == 2


def test_the_two_legs_are_summed_before_the_single_rounding() -> None:
    """Input and output value are added, then rounded exactly once.

    At the rates below one input token is worth 1.25 micros and one output
    token 1.5. Rounding each leg on its own would charge ``2 + 2 == 4``; one
    ceiling over the summed 2.75 charges 3. The exact figure is pinned so the
    arithmetic cannot drift into a second, hidden rounding.
    """
    table = ModelRateTable(
        rates={
            _MEMBER.model_version: ModelTokenRate(
                model_version=_MEMBER.model_version,
                input_micros_per_million_tokens=1_250_000,
                output_micros_per_million_tokens=1_500_000,
            )
        },
        unmetered_micros=_UNMETERED_MICROS,
    )

    combined = table.micros_for(
        model_version=_MEMBER.model_version,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )

    assert combined == 3


def test_every_default_rate_is_a_positive_integer_micros_per_million() -> None:
    """The shipped default rates are whole micros per million tokens.

    Derived from the table rather than restated, so a new model cannot be added
    with a rate this test never looks at.
    """
    assert DEFAULT_MODEL_RATE_TABLE.rates != {}
    for model_version, rate in DEFAULT_MODEL_RATE_TABLE.rates.items():
        assert rate.model_version == model_version
        assert isinstance(rate.input_micros_per_million_tokens, int)
        assert isinstance(rate.output_micros_per_million_tokens, int)
        assert rate.input_micros_per_million_tokens > 0
        assert rate.output_micros_per_million_tokens > 0


def test_a_zero_rate_is_refused_at_construction() -> None:
    """A zero-rated model would bill nothing however many tokens it burned."""
    with pytest.raises(ValueError) as excinfo:
        ModelTokenRate(
            model_version="free-lunch",
            input_micros_per_million_tokens=0,
            output_micros_per_million_tokens=_OUTPUT_RATE,
        )

    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == (
        "input_micros_per_million_tokens['free-lunch'] must be at least 1 micro, got 0"
    )


def test_a_zero_unmetered_charge_is_refused_at_construction() -> None:
    """A zero fail-closed charge would make every unmeasurable vote free."""
    with pytest.raises(ValueError) as excinfo:
        ModelRateTable(rates={}, unmetered_micros=0)

    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == "unmetered_micros must be at least 1 micro, got 0"


# --- Criterion 3: an unmeasurable response fails closed ---------------------------

#: Every way a ``usage`` block can fail to be two whole token counts. Each must
#: charge the fail-closed figure -- none may become free research.
_UNREADABLE_USAGE_BLOCKS = (
    ("absent", None),
    ("not_an_object", "1000"),
    ("missing_output_key", {"prompt_tokens": 4_000}),
    ("fractional", {"prompt_tokens": 4000.5, "completion_tokens": 500}),
    ("string_valued", {"prompt_tokens": "4000", "completion_tokens": 500}),
    ("boolean_valued", {"prompt_tokens": True, "completion_tokens": 500}),
    ("negative", {"prompt_tokens": -4_000, "completion_tokens": 500}),
    ("zero_total", {"prompt_tokens": 0, "completion_tokens": 0}),
)


@pytest.mark.parametrize(
    "block",
    [block for _, block in _UNREADABLE_USAGE_BLOCKS],
    ids=[label for label, _ in _UNREADABLE_USAGE_BLOCKS],
)
def test_an_unreadable_usage_block_charges_the_fail_closed_bound(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    block: object,
) -> None:
    """No unparseable usage block becomes free research.

    The vote itself is still perfectly usable -- a missing invoice line is not
    a reason to throw a forecast away -- so what must hold is the *charge*: the
    fail-closed bound, never zero and never the flat list price. All three
    figures are named.
    """
    body = _openai_body_without_usage() if block is None else _openai_body(block)
    provider = _openai_vote(body)

    result = provider.forecast(market, baseline, 0, ())

    assert result.cost_micros == _UNMETERED_MICROS
    assert result.cost_micros != 0
    assert result.cost_micros != _LIST_PRICE_MICROS


def test_a_negative_reported_token_count_is_refused_at_construction() -> None:
    """A negative token count is not a small bill -- it is a broken one.

    `TokenUsage` is the only structural guard between a hostile or corrupt
    envelope and the money path: a ``-1_000_000``-token response would other-
    wise *subtract* a dollar from the day's spend, which is worse than free.
    The exact message is asserted so the guard cannot be satisfied by a
    same-typed error raised for a different reason.
    """
    with pytest.raises(ValueError) as excinfo:
        TokenUsage(input_tokens=-1, output_tokens=0)

    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == "input_tokens must be non-negative, got -1"


def test_a_negative_output_token_count_is_refused_too() -> None:
    """The output leg carries the same guard, named for its own field."""
    with pytest.raises(ValueError) as excinfo:
        TokenUsage(input_tokens=0, output_tokens=-7)

    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == "output_tokens must be non-negative, got -7"


def test_a_boolean_cannot_masquerade_as_a_token_count() -> None:
    """``bool`` is an ``int`` subclass and must never pass for a count."""
    with pytest.raises(TypeError) as excinfo:
        TokenUsage(input_tokens=True, output_tokens=0)

    assert type(excinfo.value) is TypeError
    assert str(excinfo.value) == "input_tokens must be a non-bool int, got bool"


def test_a_cassette_recording_a_negative_count_fails_the_load(
    tmp_path: Path,
) -> None:
    """The structural guard holds through the real replay path too.

    `_recorded_usage` deliberately does not restate the non-negativity check --
    it constructs a `TokenUsage`, and that type is where the rule lives. This
    proves the rule actually reaches a cassette load rather than being a claim
    only the constructor's own unit test can see.
    """
    cassette_path = tmp_path / "cassette.json"
    cassette_path.write_text(
        json.dumps(
            {
                "some-hash": {
                    "request": {
                        "provider": "openai",
                        "model_version": "m",
                        "prompt": "p",
                    },
                    "response": _VOTE_JSON,
                    "usage": {"input_tokens": -4_000, "output_tokens": 500},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        ReplayCassette.from_path(cassette_path)

    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == "input_tokens must be non-negative, got -4000"


def test_a_response_from_an_unrated_model_charges_the_fail_closed_bound(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A model the operator never rated is unmeasurable, not free.

    The sibling of `unknown_provider_price_micros`: an unpriced *provider*
    cannot evade the pre-gate, and an unrated *model* cannot evade the meter.
    """
    unrated = EnsembleMember("openai", "a-model-with-no-configured-rate", "2024-09-30")
    provider = _openai_vote(_openai_body(_usage_block(_HUGE_USAGE)), member=unrated)

    result = provider.forecast(market, baseline, 0, ())

    assert result.cost_micros == _UNMETERED_MICROS


def test_an_unpriced_provider_still_cannot_evade_its_budget(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """The pre-gate keeps charging an unmapped provider the high fallback.

    The property issue #399 established, restated against the metered path so
    it cannot be lost: a member whose whole allowance is one micro below
    `unknown_provider_price_micros` cannot make even one attempt, and no HTTP
    request is sent.
    """
    http = _CannedHttpTransport(_openai_body(_usage_block(_MODEST_USAGE)))
    provider = _live_wrapped(
        FixtureVoteProvider(OpenAiChatTransport(http), _MEMBER),
        provider_name="a-provider-with-no-price",
        max_cost_micros=_UNKNOWN_PRICE_MICROS - 1,
    )

    with pytest.raises(ProviderCostOverrunError) as excinfo:
        provider.forecast(market, baseline, 0, ())

    assert type(excinfo.value) is ProviderCostOverrunError
    assert excinfo.value.ceiling_micros == _UNKNOWN_PRICE_MICROS - 1
    assert http.sends == 0


def test_the_default_unmetered_charge_is_the_whole_member_allowance() -> None:
    """The shipped fail-closed figure is derived, not invented.

    If a response will not say what it cost, the only honest assumption is that
    it spent everything the member was allowed to -- and that keeps the failure
    recoverable, since exactly one such vote stays affordable.
    """
    assert DEFAULT_UNMETERED_RESPONSE_MICROS == DEFAULT_PER_MEMBER_VOTE_CEILING_MICROS
    assert DEFAULT_MODEL_RATE_TABLE.unmetered_micros == (
        DEFAULT_UNMETERED_RESPONSE_MICROS
    )


# --- Criterion 4: cassette replay reproduces the recorded cost --------------------


def test_a_recorded_cassette_replays_the_exact_cost_it_recorded(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot, tmp_path: Path
) -> None:
    """Usage rides through record and replay, so an offline cost is stable.

    The meter is never special-cased for replay: the recording writes the
    reported token counts into the cassette, and the replay serves them back,
    so the replayed vote costs exactly what the recorded one did.
    """
    cassette_path = tmp_path / "cassette.json"
    recorder = RecordingCassette(
        transport=_CannedLlmTransport(Completion(text=_VOTE_JSON, usage=_MODEST_USAGE)),
        path=cassette_path,
    )
    recorded = _live_wrapped(FixtureVoteProvider(recorder, _MEMBER)).forecast(
        market, baseline, 0, ()
    )

    replayed = _live_wrapped(
        FixtureVoteProvider(ReplayCassette.from_path(cassette_path), _MEMBER)
    ).forecast(market, baseline, 0, ())

    assert recorded.cost_micros == _MODEST_METERED_MICROS
    assert replayed.cost_micros == recorded.cost_micros


def test_two_replays_of_one_cassette_cost_identically(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot, tmp_path: Path
) -> None:
    """Replay cost is a function of the file, so the offline suite is stable."""
    cassette_path = tmp_path / "cassette.json"
    RecordingCassette(
        transport=_CannedLlmTransport(Completion(text=_VOTE_JSON, usage=_HUGE_USAGE)),
        path=cassette_path,
    ).complete(_a_request(market, baseline))

    costs = [
        _live_wrapped(
            FixtureVoteProvider(ReplayCassette.from_path(cassette_path), _MEMBER)
        )
        .forecast(market, baseline, 0, ())
        .cost_micros
        for _ in range(2)
    ]

    assert costs == [_HUGE_METERED_MICROS, _HUGE_METERED_MICROS]


def test_a_cassette_recorded_before_usage_existed_replays_fail_closed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot, tmp_path: Path
) -> None:
    """An entry with no `usage` key is unknown-cost, never zero-cost.

    Every cassette committed before issue #451 has exactly this shape, and the
    honest reading of it is "nobody recorded what this cost" -- which charges
    the fail-closed bound rather than reading as free research.
    """
    cassette_path = tmp_path / "cassette.json"
    recorder = RecordingCassette(
        transport=_CannedLlmTransport(Completion(text=_VOTE_JSON)),
        path=cassette_path,
    )
    recorder.complete(_a_request(market, baseline))
    entry = next(iter(json.loads(cassette_path.read_text(encoding="utf-8")).values()))

    replayed = _live_wrapped(
        FixtureVoteProvider(ReplayCassette.from_path(cassette_path), _MEMBER)
    ).forecast(market, baseline, 0, ())

    assert "usage" not in entry
    assert replayed.cost_micros == _UNMETERED_MICROS


def test_a_corrupt_usage_block_fails_the_cassette_load_loudly(
    tmp_path: Path,
) -> None:
    """A malformed *recorded* usage block is a corrupt file, not an unknown.

    Absent usage is a legitimate state and reads as unknown. A `usage` block
    that is present but not two integers means the recording is damaged, and
    silently degrading it to "unknown" would hide that behind a plausible
    charge.
    """
    cassette_path = tmp_path / "cassette.json"
    cassette_path.write_text(
        json.dumps(
            {
                "some-hash": {
                    "request": {
                        "provider": "openai",
                        "model_version": "m",
                        "prompt": "p",
                    },
                    "response": _VOTE_JSON,
                    "usage": {"input_tokens": "4000", "output_tokens": 500},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        ReplayCassette.from_path(cassette_path)

    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == (
        "cassette entry 'some-hash' has a non-integer usage.input_tokens"
    )


# --- Criterion 5: the ceiling now binds on spend ----------------------------------


def _metered_factory(
    usage: TokenUsage,
) -> Callable[[EnsembleMemberLike], ForecastProvider]:
    """Build a live-shaped `provider_factory` metering every member's vote.

    Every member is deliberately voted as `_MEMBER`, so all three votes are
    metered at one model's rates and the run's charge is three identical
    figures -- which is what lets a ceiling be placed exactly one micro below
    the total.

    Args:
        usage: The token accounting every member's response reports.

    Returns:
        A `provider_factory` closure `run_pipeline` drives.
    """

    def _factory(member: EnsembleMemberLike) -> ForecastProvider:
        """Return the live-wrapped, metered provider for ``member``.

        Args:
            member: The (unused) ensemble member being driven.

        Returns:
            The wrapped `ForecastProvider`.
        """
        del member
        return _live_wrapped(
            FixtureVoteProvider(
                _CannedLlmTransport(Completion(text=_VOTE_JSON, usage=usage)),
                _MEMBER,
            )
        )

    return _factory


def test_one_expensive_vote_now_trips_the_per_forecast_ceiling(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """A run whose metered spend exceeds the ceiling halts and is ledgered.

    Issue #451's mandatory test. The ceiling sits one micro below what the
    three metered votes plus the research stub actually cost -- and *above*
    what the same three votes cost at the flat list price, which is the run
    that passed before this change. Both figures are computed here, so the
    test would fail if the meter reverted to the constant.
    """
    metered_total = _FULL_RUN_RESEARCH_COST_MICROS + 3 * _HUGE_METERED_MICROS
    list_priced_total = _FULL_RUN_RESEARCH_COST_MICROS + 3 * _LIST_PRICE_MICROS
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(per_forecast_micros=metered_total - 1, ledger=ledger)

    with pytest.raises(PerForecastBudgetExceededError) as excinfo:
        run_pipeline(
            market,
            baseline,
            transport=ForbiddenLiveTransport(),
            created_at=created_at,
            research_tools=research_tools,
            budget=budget,
            provider_factory=_metered_factory(_HUGE_USAGE),
        )

    assert list_priced_total <= metered_total - 1
    assert excinfo.value.cost_micros == metered_total
    assert excinfo.value.budget_micros == metered_total - 1
    events = ledger.events_by_type(BUDGET_FORECAST_EXCEEDED_EVENT)
    assert len(events) == 1
    assert events[0].payload == {
        "cost_micros": metered_total,
        "budget_micros": metered_total - 1,
        "market_ticker": market.ticker,
        "utc_day": "2024-12-10",
    }


def test_the_identical_run_on_cheaper_votes_completes_under_that_ceiling(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """The ceiling now responds to *usage*, not to the number of votes.

    Same market, same ensemble size, same ceiling, same number of attempts --
    only the reported token counts differ, and the run completes. Under the old
    flat charge both runs cost the identical three list prices, so no ceiling
    could distinguish them.
    """
    metered_total = _FULL_RUN_RESEARCH_COST_MICROS + 3 * _HUGE_METERED_MICROS
    budget = ResearchBudget(
        per_forecast_micros=metered_total - 1, ledger=InMemoryBudgetLedger()
    )

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        budget=budget,
        provider_factory=_metered_factory(_MODEST_USAGE),
    )

    assert record.abstention_reason is None


def test_the_metered_day_bucket_survives_a_local_midnight(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
    local_timezone_utc_minus_5: None,
) -> None:
    """A metered charge buckets to the UTC day, not the host's calendar day.

    CI runs UTC and cannot see this on its own. Under a pinned UTC-05:00 zone
    the run's instant is still 2024-12-10 in UTC but 2024-12-09 locally; a
    charge bucketed locally would open a fresh day and let the ceiling reset.
    """
    del local_timezone_utc_minus_5
    ledger = InMemoryBudgetLedger()
    budget = ResearchBudget(
        per_forecast_micros=100_000_000, per_day_micros=100_000_000, ledger=ledger
    )

    run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        budget=budget,
        provider_factory=_metered_factory(_HUGE_USAGE),
    )

    spends = ledger.events_by_type(BUDGET_SPEND_RECORDED_EVENT)
    assert len(spends) == 1
    assert spends[0].payload == {
        "utc_day": "2024-12-10",
        "market_ticker": market.ticker,
        "cost_micros": _FULL_RUN_RESEARCH_COST_MICROS + 3 * _HUGE_METERED_MICROS,
    }


def _usage_block(usage: TokenUsage) -> dict[str, int]:
    """Render ``usage`` as an OpenAI-shaped envelope usage block.

    Args:
        usage: The token accounting to render.

    Returns:
        The ``{prompt_tokens, completion_tokens}`` mapping.
    """
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
    }


def _a_request(market: NormalizedMarket, baseline: BaselineQuoteSnapshot) -> object:
    """Build the exact `LlmRequest` a `FixtureVoteProvider` vote would send.

    Args:
        market: The market under forecast.
        baseline: The baseline quote snapshot.

    Returns:
        The `LlmRequest` for vote index zero.
    """
    return LlmRequest(
        provider=_MEMBER.provider,
        model_version=_MEMBER.model_version,
        prompt=build_vote_prompt(market, baseline, 0, ()),
    )
