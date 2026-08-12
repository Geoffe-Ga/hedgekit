"""Tests for windbreak.forecast.providers.futuresearch (issue #189).

Pins the hosted research-forecaster `FutureSearchProvider`'s full response
contract: the request body is a pure, sorted-key canonical-JSON function of
the market/baseline/vote-index question fields -- never the pipeline's
`quotes` (ADR-0005 S1(b)); the response is screened for injection (whole-body,
then per-field for a JSON-escaped artifact a whole-body scan would miss),
parsed with `decimal.Decimal` (never a binary float) into an integer-ppm
probability with `ROUND_HALF_EVEN` and no re-clamp, a bounded non-empty
rationale, provider-reported citations, a pinned-version drift check, and a
`ROUND_CEILING` cost-in-micros conversion that defaults fail-closed to the
per-call ceiling. Also pins the module's own HTTP record/replay
determinism, its refusal to ever persist a live-only secret, its import
boundary (no `requests`, no `windbreak.config`), and how its reported
citations thread into `windbreak.forecast.pipeline.run_pipeline` without
inflating the independently-verified citation count.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import windbreak.forecast.providers.futuresearch as futuresearch_module
from windbreak.forecast.cassettes import (
    CassetteMissError,
    ForbiddenLiveTransport,
    LiveCallForbiddenError,
)
from windbreak.forecast.pipeline import (
    FORECAST_OUTPUT_DISCARDED_EVENT,
    PROVIDER_REPORTED_SOURCE_TYPE,
    InMemoryForecastLedger,
    collect_model_votes,
    run_pipeline,
)
from windbreak.forecast.providers import (
    DEFAULT_VOTE_ENSEMBLE,
    EnsembleMember,
    EnsembleMemberLike,
    ForbiddenLiveHttpTransport,
    FutureSearchProvider,
    FutureSearchProviderConfig,
    HttpResponse,
    ProviderCitation,
    ProviderForecast,
    ProviderResponseRejectedError,
    ProviderVersionDriftError,
    RecordingHttpCassette,
    ReplayHttpCassette,
)
from windbreak.forecast.sanitize import (
    DATA_BLOCK_BEGIN,
    MAX_QUOTE_WORDS,
    MAX_RATIONALE_CHARS,
    RESPONSE_FAILURE_DELIMITER_FORGERY,
    RESPONSE_FAILURE_INVALID_RATIONALE,
    RESPONSE_FAILURE_MALFORMED_VOTE_JSON,
    RESPONSE_FAILURE_PROBABILITY_OUT_OF_RANGE,
    RESPONSE_FAILURE_TOOL_CALL_LURE,
    ResearchQuote,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.providers import HttpRequest
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sandbox import ResearchTools

#: The endpoint every test config below POSTs to.
_ENDPOINT_URL = "https://futuresearch.example/v1/forecast"

#: The forecaster version every default test config pins.
_PINNED_VERSION = "futuresearch-v1"

#: `_PINNED_VERSION`, pre-encoded as a JSON string token.
_PINNED_VERSION_JSON = json.dumps(_PINNED_VERSION)

#: A valid rationale text, and its pre-encoded JSON string token.
_VALID_RATIONALE = "steady evidence with corroborating detail"
_VALID_RATIONALE_JSON = json.dumps(_VALID_RATIONALE)


def _config(
    *,
    endpoint_url: str = _ENDPOINT_URL,
    pinned_forecaster_versions: tuple[str, ...] = (_PINNED_VERSION,),
    api_key_env: str = "FUTURESEARCH_API_KEY",
    per_call_ceiling_micros: int = 2_000_000,
    reject_on_version_drift: bool = True,
) -> FutureSearchProviderConfig:
    """Build a `FutureSearchProviderConfig` test double.

    Args:
        endpoint_url: The endpoint URL override.
        pinned_forecaster_versions: The pinned-version tuple override.
        api_key_env: The API-key environment-variable name override.
        per_call_ceiling_micros: The per-call cost ceiling override.
        reject_on_version_drift: The version-drift rejection policy override.

    Returns:
        A `FutureSearchProviderConfig` built from the given (or default)
        fields.
    """
    return FutureSearchProviderConfig(
        endpoint_url=endpoint_url,
        pinned_forecaster_versions=pinned_forecaster_versions,
        api_key_env=api_key_env,
        per_call_ceiling_micros=per_call_ceiling_micros,
        reject_on_version_drift=reject_on_version_drift,
    )


def _body(
    *,
    probability: str = "0.5",
    rationale: str = _VALID_RATIONALE_JSON,
    citations: str = "[]",
    forecaster_version: str = _PINNED_VERSION_JSON,
    cost_usd: str | None = None,
) -> str:
    """Build a raw FutureSearch response body from pre-encoded JSON tokens.

    Every parameter is the *already JSON-encoded* text for that field (not a
    Python value), so a caller can inject an out-of-schema literal a plain
    `json.dumps` call would refuse to produce (an unquoted number where a
    string is expected, a bare `Infinity` constant, ...).

    Args:
        probability: The raw JSON token for the `probability` field.
        rationale: The raw JSON token for the `rationale` field.
        citations: The raw JSON token for the `citations` field.
        forecaster_version: The raw JSON token for the `forecaster_version`
            field.
        cost_usd: The raw JSON token for the `cost_usd` field, or `None` to
            omit the key entirely.

    Returns:
        The composed raw response body text.
    """
    fields = [
        f'"probability": {probability}',
        f'"rationale": {rationale}',
        f'"citations": {citations}',
        f'"forecaster_version": {forecaster_version}',
    ]
    if cost_usd is not None:
        fields.append(f'"cost_usd": {cost_usd}')
    return "{" + ", ".join(fields) + "}"


class _StubHttpTransport:
    """A minimal `HttpTransport` double returning one fixed response verbatim."""

    def __init__(self, body: str, *, status_code: int = 200) -> None:
        """Store the fixed response every `send` call returns.

        Args:
            body: The fixed raw response body text to return.
            status_code: The fixed HTTP status code to return.
        """
        self._body = body
        self._status_code = status_code
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        """Record one call and return the fixed response, ignoring `request`.

        Args:
            request: The (recorded, otherwise unused) HTTP request.

        Returns:
            `HttpResponse(self._status_code, self._body)`, verbatim.
        """
        self.calls.append(request)
        return HttpResponse(self._status_code, self._body)


class _KeyReadingFakeTransport:
    """An `HttpTransport` reading a live-only secret from the environment.

    Models a real live transport injecting an API key at send time -- never
    part of `HttpRequest` itself -- so a recorded cassette can never capture
    it.
    """

    def __init__(self, body: str) -> None:
        """Store the fixed response body every call returns.

        Args:
            body: The fixed raw response body text to return.
        """
        self._body = body

    def send(self, request: HttpRequest) -> HttpResponse:
        """Read (and discard) a live-only secret, then return the fixed body.

        Args:
            request: The (unused) HTTP request.

        Returns:
            `HttpResponse(200, self._body)`.
        """
        del request
        _ = os.environ["FUTURESEARCH_API_KEY"]
        return HttpResponse(200, self._body)


def _rejected(
    body: str, market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> ProviderResponseRejectedError:
    """Drive one forecast over `body` and return the raised rejection error.

    Args:
        body: The raw response body the stub transport returns.
        market: The market under forecast.
        baseline: The baseline quote snapshot.

    Returns:
        The `ProviderResponseRejectedError` the call raised.
    """
    provider = FutureSearchProvider(_StubHttpTransport(body), _config())
    with pytest.raises(ProviderResponseRejectedError) as excinfo:
        provider.forecast(market, baseline, 0, ())
    return excinfo.value


# --- Happy path: full ProviderForecast contract -----------------------------------


def test_forecast_happy_path_returns_provider_forecast(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A clean, schema-valid response yields a fully-populated `ProviderForecast`."""
    body = _body(probability="0.62", cost_usd="0.30")
    provider = FutureSearchProvider(_StubHttpTransport(body), _config())

    result = provider.forecast(market, baseline, 0, ())

    assert isinstance(result, ProviderForecast)
    assert result.probability_ppm == 620_000
    assert result.rationale_summary == _VALID_RATIONALE
    assert result.citations == ()
    assert result.cost_micros == 300_000
    assert result.provider == "futuresearch"
    assert result.model_version == _PINNED_VERSION
    assert result.training_cutoff == "server-managed"
    assert (
        result.response_fingerprint == hashlib.sha256(body.encode("utf-8")).hexdigest()
    )


def test_request_body_is_canonical_json_of_question_fields(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """The request body is exactly the sorted-key, compact-separator canonical
    JSON of the five question fields -- never the quotes.
    """
    transport = _StubHttpTransport(_body())
    provider = FutureSearchProvider(transport, _config())

    provider.forecast(market, baseline, 2, ())

    expected_body = json.dumps(
        {
            "baseline_price_pips": baseline.price_pips,
            "resolution_criteria": market.resolution_criteria,
            "ticker": market.ticker,
            "title": market.title,
            "vote_index": 2,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert transport.calls[0].method == "POST"
    assert transport.calls[0].url == _ENDPOINT_URL
    assert transport.calls[0].body == expected_body


def test_forecast_ignores_pipeline_quotes_entirely(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Supplying non-empty `quotes` yields an identical `ProviderForecast` and
    an identical underlying request body/hash to supplying none at all --
    proving the research forecaster never threads pipeline quotes into its
    request (ADR-0005 S1(b)).
    """
    transport_no_quotes = _StubHttpTransport(_body())
    transport_with_quotes = _StubHttpTransport(_body())
    quotes = (ResearchQuote(url="https://research.local/a", text="some quote text"),)

    result_no_quotes = FutureSearchProvider(transport_no_quotes, _config()).forecast(
        market, baseline, 0, ()
    )
    result_with_quotes = FutureSearchProvider(
        transport_with_quotes, _config()
    ).forecast(market, baseline, 0, quotes)

    assert result_no_quotes == result_with_quotes
    assert (
        transport_no_quotes.calls[0].request_hash()
        == transport_with_quotes.calls[0].request_hash()
    )
    assert transport_no_quotes.calls[0].body == transport_with_quotes.calls[0].body


# --- Probability -> ppm: Decimal exactness, ROUND_HALF_EVEN, no re-clamp ---------


@pytest.mark.parametrize(
    ("probability_token", "expected_ppm"),
    [
        ("0.62", 620_000),
        ("0.1", 100_000),
        ("0", 0),
        ("1", 1_000_000),
        ("0.03", 30_000),
        ("0.97", 970_000),
        ("0.5", 500_000),
        ("0.1234565", 123_456),  # exactly 123456.5 ppm -> round to even 123456
        ("0.1234575", 123_458),  # exactly 123457.5 ppm -> round to even 123458
    ],
    ids=[
        "0.62-ppm",
        "0.1-exact-decimal",
        "json-int-zero",
        "json-int-one",
        "0.03-no-reclamp",
        "0.97-no-reclamp",
        "0.5-midpoint",
        "half-even-rounds-down-to-even",
        "half-even-rounds-up-to-even",
    ],
)
def test_forecast_converts_probability_to_exact_ppm(
    probability_token: str,
    expected_ppm: int,
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
) -> None:
    """Every reported probability converts to the exact expected integer ppm,
    via `Decimal` arithmetic and `ROUND_HALF_EVEN`, never re-clamped.
    """
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(probability=probability_token)), _config()
    )

    result = provider.forecast(market, baseline, 0, ())

    assert result.probability_ppm == expected_ppm


@pytest.mark.parametrize("probability_token", ["-0.1", "1.1", "2", "-1"])
def test_out_of_range_probability_is_rejected(
    probability_token: str, market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A numeric probability outside `[0, 1]` is rejected, never clamped."""
    error = _rejected(_body(probability=probability_token), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_PROBABILITY_OUT_OF_RANGE


# --- Malformed responses: wrong types, bad shapes, non-finite constants ----------


def test_probability_wrong_type_string_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A string `probability` value is rejected as malformed, not coerced."""
    error = _rejected(_body(probability='"not-a-number"'), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_probability_true_bool_is_rejected(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A JSON `true`/`false` `probability` value is rejected as malformed: a
    `bool` is an `int` subclass and must never masquerade as a numeric
    probability (pins the `isinstance(value, bool)` guard at
    `futuresearch.py:257`, #189).
    """
    error = _rejected(_body(probability="true"), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_probability_null_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `null` `probability` value is rejected as malformed."""
    error = _rejected(_body(probability="null"), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_missing_probability_key_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A response missing the `probability` key entirely is rejected as
    malformed.
    """
    body = (
        f'{{"rationale": {_VALID_RATIONALE_JSON}, "citations": [], '
        f'"forecaster_version": {_PINNED_VERSION_JSON}}}'
    )
    error = _rejected(body, market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_response_body_not_a_json_object_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A response body that is valid JSON but not an object (a bare array) is
    rejected as malformed.
    """
    error = _rejected("[]", market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_malformed_json_body_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A response body that is not valid JSON at all is rejected as malformed."""
    error = _rejected("not json at all", market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


@pytest.mark.parametrize("constant_token", ["Infinity", "-Infinity", "NaN"])
def test_non_finite_json_constant_is_rejected_as_malformed(
    constant_token: str, market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A non-finite JSON constant (`Infinity`/`-Infinity`/`NaN`) anywhere in the
    body is rejected as malformed, never materialized as a Python float.
    """
    error = _rejected(_body(probability=constant_token), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_non_string_forecaster_version_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A non-string `forecaster_version` (a bare JSON number) is rejected as
    malformed.
    """
    error = _rejected(_body(forecaster_version="123"), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_citation_entry_not_an_object_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A citation array element that is not a JSON object is rejected."""
    error = _rejected(_body(citations='["not-a-dict"]'), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_citation_missing_url_and_quoted_text_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A citation object missing both `url` and `quoted_text` is rejected."""
    error = _rejected(_body(citations='[{"bad": "shape"}]'), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_citation_non_string_url_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A citation object whose `url` is not a string is rejected."""
    citations = json.dumps([{"url": 123, "quoted_text": "some text"}])
    error = _rejected(_body(citations=citations), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_citation_non_string_quoted_text_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A citation object whose `quoted_text` is not a string is rejected."""
    citations = json.dumps([{"url": "https://example.test/a", "quoted_text": 123}])
    error = _rejected(_body(citations=citations), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_citation_unparseable_publication_date_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A citation object whose `publication_date` is not a parseable ISO-8601
    string is rejected.
    """
    citations = json.dumps(
        [
            {
                "url": "https://example.test/a",
                "quoted_text": "some text",
                "publication_date": "not-a-date",
            }
        ]
    )
    error = _rejected(_body(citations=citations), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_citation_non_string_publication_date_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A citation whose `publication_date` is neither `null` nor a string (here
    a JSON number) is rejected before it can reach date parsing.
    """
    citations = json.dumps(
        [
            {
                "url": "https://example.test/a",
                "quoted_text": "some text",
                "publication_date": 123,
            }
        ]
    )
    error = _rejected(_body(citations=citations), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


@pytest.mark.parametrize(
    "naive",
    [
        "2024-03-15T00:00:00",
        "2024-03-15T00:00:00.123456",
        "2024-03-15",
    ],
    ids=["seconds", "microseconds", "date-only"],
)
def test_citation_naive_publication_date_is_rejected_as_malformed(
    naive: str, market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `publication_date` carrying no UTC offset is rejected at the provider
    boundary, like every other malformed date shape.

    `datetime.fromisoformat` parses all three of these cleanly and returns a
    value with `tzinfo is None`, which contradicts this parser's documented
    "timezone-aware datetime" contract. Left unguarded the naive value flows
    through `_build_citation` into a `ProviderCitation`, and the problem only
    surfaces much later as an S8.8 `FAILURE_PUBLICATION_DATE_INVALID` in
    `citations.py` -- so the provider emits a structurally-invalid citation it
    should have refused outright.

    The `date-only` case matters independently of the other two: a bare date is
    the shape a real provider is most likely to emit, and it is the one a naive
    check written against the `T`-separated forms would miss.

    Rejecting rather than coercing is deliberate and matches the siblings:
    `windbreak/forecast/pubdate.py` degrades a naive value to `None` precisely
    so a timezone is never fabricated, and assuming UTC here would invent a
    fact the provider did not report.
    """
    citations = json.dumps(
        [
            {
                "url": "https://example.test/a",
                "quoted_text": "some text",
                "publication_date": naive,
            }
        ]
    )
    error = _rejected(_body(citations=citations), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


def test_citations_field_not_a_list_is_rejected_as_malformed(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `citations` value that is present but not a JSON array is rejected."""
    error = _rejected(_body(citations='{"not": "a list"}'), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_MALFORMED_VOTE_JSON


# --- Rationale: missing / empty / too long ----------------------------------------


def test_rationale_missing_key_is_rejected_as_invalid(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A response missing the `rationale` key entirely is rejected."""
    body = (
        f'{{"probability": 0.5, "citations": [], '
        f'"forecaster_version": {_PINNED_VERSION_JSON}}}'
    )
    error = _rejected(body, market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_INVALID_RATIONALE


def test_rationale_empty_string_is_rejected_as_invalid(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """An empty-string `rationale` is rejected."""
    error = _rejected(_body(rationale='""'), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_INVALID_RATIONALE


def test_rationale_too_long_is_rejected_as_invalid(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `rationale` longer than `MAX_RATIONALE_CHARS` is rejected."""
    too_long = json.dumps("x" * (MAX_RATIONALE_CHARS + 1))
    error = _rejected(_body(rationale=too_long), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_INVALID_RATIONALE


# --- Injection: whole-body screen, and per-field for a JSON-escaped artifact -----


def test_delimiter_forgery_anywhere_in_body_is_rejected(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A forged untrusted-data delimiter token anywhere in the raw body is
    caught by the whole-body screen, before any JSON parse is attempted.
    """
    body = _body() + f" {DATA_BLOCK_BEGIN}"
    error = _rejected(body, market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_DELIMITER_FORGERY


def test_tool_call_marker_anywhere_in_body_is_rejected(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A tool-call marker anywhere in the raw body is caught by the whole-body
    screen.
    """
    body = _body() + ' "tool"'
    error = _rejected(body, market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_TOOL_CALL_LURE


def test_tool_call_lure_hidden_in_rationale_via_json_escape_is_rejected(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A tool-call marker JSON-escaped inside `rationale` (so the whole-body
    raw-text screen misses the escaped `\\"tool\\"` bytes) is still caught by
    the per-field screen once the field is parsed.
    """
    rationale_text = 'evidence mentioning "tool" inline'
    rationale_token = json.dumps(rationale_text)
    assert '"tool"' not in _body(rationale=rationale_token)  # whole-body screen blind

    error = _rejected(_body(rationale=rationale_token), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_TOOL_CALL_LURE


def test_tool_call_lure_hidden_in_citation_quoted_text_via_json_escape_is_rejected(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A tool-call marker JSON-escaped inside a citation's `quoted_text` is
    caught by the per-field screen after the citation is parsed.
    """
    quoted_text = 'a source mentioning "tool" inline'
    citations = json.dumps(
        [{"url": "https://example.test/a", "quoted_text": quoted_text}]
    )
    assert '"tool"' not in _body(citations=citations)  # whole-body screen blind

    error = _rejected(_body(citations=citations), market, baseline)

    assert error.failure_code == RESPONSE_FAILURE_TOOL_CALL_LURE


def test_rejection_error_never_leaks_the_raw_response_text(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """The rejection's string form carries the failure code and fingerprint,
    never the raw (untrusted) response text.
    """
    error = _rejected(_body(probability="1.5"), market, baseline)

    assert "1.5" not in str(error)
    assert error.failure_code in str(error)
    assert error.response_fingerprint in str(error)


# --- Citations: threading, null publication_date, quote truncation --------------


def test_citations_are_threaded_with_reported_fields(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Each reported citation object becomes a `ProviderCitation` carrying its
    reported url, parsed publication date, and quoted text.
    """
    citations = json.dumps(
        [
            {
                "url": "https://provider.example/report-a",
                "publication_date": "2024-03-15T00:00:00Z",
                "quoted_text": "a corroborating excerpt",
            }
        ]
    )
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(citations=citations)), _config()
    )

    result = provider.forecast(market, baseline, 0, ())

    assert result.citations == (
        ProviderCitation(
            url="https://provider.example/report-a",
            publication_date=datetime(2024, 3, 15, tzinfo=UTC),
            quoted_text="a corroborating excerpt",
        ),
    )


def test_citation_publication_date_null_maps_to_none(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `publication_date: null` citation field maps to `None`."""
    citations = json.dumps(
        [
            {
                "url": "https://provider.example/report-b",
                "publication_date": None,
                "quoted_text": "some text",
            }
        ]
    )
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(citations=citations)), _config()
    )

    result = provider.forecast(market, baseline, 0, ())

    assert result.citations[0].publication_date is None


def test_citation_publication_date_accepts_plus_zero_offset_suffix(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `publication_date` spelled with an explicit `+00:00` offset (instead
    of `Z`) parses to the identical aware datetime.
    """
    citations = json.dumps(
        [
            {
                "url": "https://provider.example/report-c",
                "publication_date": "2024-03-15T00:00:00+00:00",
                "quoted_text": "some text",
            }
        ]
    )
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(citations=citations)), _config()
    )

    result = provider.forecast(market, baseline, 0, ())

    assert result.citations[0].publication_date == datetime(2024, 3, 15, tzinfo=UTC)


def test_citation_quoted_text_is_truncated_to_max_quote_words(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A reported `quoted_text` longer than `MAX_QUOTE_WORDS` is truncated to
    exactly that many words in the resulting `ProviderCitation`.
    """
    words = [f"word{index}" for index in range(MAX_QUOTE_WORDS + 5)]
    long_quote = " ".join(words)
    citations = json.dumps(
        [
            {
                "url": "https://provider.example/report-d",
                "publication_date": None,
                "quoted_text": long_quote,
            }
        ]
    )
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(citations=citations)), _config()
    )

    result = provider.forecast(market, baseline, 0, ())

    assert result.citations[0].quoted_text == " ".join(words[:MAX_QUOTE_WORDS])
    assert len(result.citations[0].quoted_text.split()) == MAX_QUOTE_WORDS


# --- Cost: ROUND_CEILING, fail-closed default to the per-call ceiling ------------


def test_cost_usd_converts_to_micros_with_round_ceiling(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """An exact `cost_usd` value converts to micros unchanged under ceiling
    rounding.
    """
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(cost_usd="0.30")), _config()
    )

    result = provider.forecast(market, baseline, 0, ())

    assert result.cost_micros == 300_000


def test_cost_usd_rounds_a_fractional_micro_value_up(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A `cost_usd` value landing on a fractional micro rounds up (ceiling),
    dollars never undercounted.
    """
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(cost_usd="0.1000005")), _config()
    )

    result = provider.forecast(market, baseline, 0, ())

    assert result.cost_micros == 100_001


@pytest.mark.parametrize(
    "cost_usd_token",
    [None, "null", '"free"', "-0.5"],
    ids=["missing", "null", "non-numeric", "negative"],
)
def test_cost_usd_bad_value_defaults_to_per_call_ceiling(
    cost_usd_token: str | None,
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
) -> None:
    """A missing, `null`, non-numeric, or negative `cost_usd` does not reject
    the response -- it only defaults the cost to the configured per-call
    ceiling.
    """
    config = _config(per_call_ceiling_micros=999_000)
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(cost_usd=cost_usd_token)), config
    )

    result = provider.forecast(market, baseline, 0, ())

    assert result.cost_micros == 999_000


def test_cost_usd_true_bool_defaults_to_ceiling(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A JSON `true`/`false` `cost_usd` value does not reject the response --
    it only defaults the cost to the per-call ceiling, since a `bool` is an
    `int` subclass that must never be treated as a numeric cost (pins the
    `isinstance(value, bool)` guard at `futuresearch.py:389`, #189).
    """
    config = _config(per_call_ceiling_micros=777_000)
    provider = FutureSearchProvider(_StubHttpTransport(_body(cost_usd="true")), config)

    result = provider.forecast(market, baseline, 0, ())

    assert result.cost_micros == 777_000


# --- Version drift: strict reject by default, permissive with a warning ---------


def test_unpinned_version_raises_provider_version_drift_error_by_default(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """An unpinned reported version raises `ProviderVersionDriftError` under
    the strict default, carrying the reported version and the pinned set.
    """
    unpinned_version_json = json.dumps("futuresearch-v2-unpinned")
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(forecaster_version=unpinned_version_json)),
        _config(),
    )

    with pytest.raises(ProviderVersionDriftError) as excinfo:
        provider.forecast(market, baseline, 0, ())

    assert excinfo.value.reported_version == "futuresearch-v2-unpinned"
    assert excinfo.value.pinned_versions == (_PINNED_VERSION,)


def test_unpinned_version_with_drift_permitted_proceeds_and_logs_warning(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With `reject_on_version_drift=False`, an unpinned version does not
    raise: the forecast proceeds and stamps the *reported* version, logging
    exactly one warning through the module's own logger.
    """
    caplog.set_level(
        logging.WARNING, logger="windbreak.forecast.providers.futuresearch"
    )
    unpinned_version_json = json.dumps("futuresearch-v2-unpinned")
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(forecaster_version=unpinned_version_json)),
        _config(reject_on_version_drift=False),
    )

    result = provider.forecast(market, baseline, 0, ())

    assert result.model_version == "futuresearch-v2-unpinned"
    warnings = [
        record
        for record in caplog.records
        if record.name == "windbreak.forecast.providers.futuresearch"
        and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1


def test_pinned_version_is_stamped_without_drift(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A reported version inside the pinned set is stamped as-is, never
    raising drift.
    """
    provider = FutureSearchProvider(_StubHttpTransport(_body()), _config())

    result = provider.forecast(market, baseline, 0, ())

    assert result.model_version == _PINNED_VERSION


def test_version_drift_error_carries_failure_code_and_fingerprint(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """An unpinned reported version raises `ProviderVersionDriftError` that
    also carries `.failure_code` (== `RESPONSE_FAILURE_VERSION_DRIFT`) and a
    non-empty sha256 `.response_fingerprint`, alongside the pre-existing
    `.reported_version`/`.pinned_versions` -- so a version-drift failure can
    be discarded and ledgered by the pipeline exactly like a
    `ProviderResponseRejectedError` (#189).
    """
    from windbreak.forecast.sanitize import RESPONSE_FAILURE_VERSION_DRIFT

    unpinned_version_json = json.dumps("futuresearch-v2-unpinned")
    body = _body(forecaster_version=unpinned_version_json)
    provider = FutureSearchProvider(_StubHttpTransport(body), _config())

    with pytest.raises(ProviderVersionDriftError) as excinfo:
        provider.forecast(market, baseline, 0, ())

    assert excinfo.value.failure_code == RESPONSE_FAILURE_VERSION_DRIFT
    assert len(excinfo.value.response_fingerprint) == 64
    assert (
        excinfo.value.response_fingerprint
        == hashlib.sha256(body.encode("utf-8")).hexdigest()
    )
    assert excinfo.value.reported_version == "futuresearch-v2-unpinned"
    assert excinfo.value.pinned_versions == (_PINNED_VERSION,)


# --- HTTP status: non-2xx rejected fast, before any body-schema parsing ---------


def test_non_2xx_status_is_rejected_fast(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A non-2xx HTTP status is rejected immediately, before any body-schema
    parsing is attempted -- proven by pairing the bad status with a body that
    is not even valid JSON, which a parsing-first implementation would
    instead surface as `RESPONSE_FAILURE_MALFORMED_VOTE_JSON` (#189). Since
    issue #269 the rejection is the specific `ProviderHTTPError`, carrying the
    status code the retry layer classifies on.
    """
    from windbreak.forecast.providers import ProviderHTTPError
    from windbreak.forecast.sanitize import RESPONSE_FAILURE_HTTP_STATUS

    provider = FutureSearchProvider(
        _StubHttpTransport("not even json", status_code=500), _config()
    )

    with pytest.raises(ProviderHTTPError) as excinfo:
        provider.forecast(market, baseline, 0, ())

    assert excinfo.value.status_code == 500
    assert excinfo.value.failure_code == RESPONSE_FAILURE_HTTP_STATUS


def test_2xx_status_is_accepted(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """A 200 status accepts and parses the body as usual -- the status check
    never rejects a normal successful response (#189).
    """
    provider = FutureSearchProvider(
        _StubHttpTransport(_body(), status_code=200), _config()
    )

    result = provider.forecast(market, baseline, 0, ())

    assert isinstance(result, ProviderForecast)


# --- HTTP record/replay determinism, fail-closed misses, and secret hygiene -----


def test_record_then_replay_round_trip_yields_identical_provider_forecast(
    tmp_path: Path, market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Recording a provider run through `RecordingHttpCassette`, then replaying
    the persisted cassette through `ReplayHttpCassette`, yields a
    byte-identical `ProviderForecast` every time -- the harness's
    self-consistent record-replay contract.
    """
    cassette_path = tmp_path / "futuresearch_cassette.json"
    recorder = RecordingHttpCassette(
        transport=_StubHttpTransport(_body()), path=cassette_path
    )
    recorded = FutureSearchProvider(recorder, _config()).forecast(
        market, baseline, 0, ()
    )

    replay_provider = FutureSearchProvider(
        ReplayHttpCassette.from_path(cassette_path), _config()
    )
    replayed_once = replay_provider.forecast(market, baseline, 0, ())
    replayed_twice = replay_provider.forecast(market, baseline, 0, ())

    assert replayed_once == recorded
    assert replayed_twice == recorded


def test_replay_over_empty_http_cassette_raises_cassette_miss_error(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """An empty `ReplayHttpCassette` fails closed, never a live fallback."""
    provider = FutureSearchProvider(ReplayHttpCassette({}), _config())

    with pytest.raises(CassetteMissError):
        provider.forecast(market, baseline, 0, ())


def test_forbidden_live_http_transport_raises_live_call_forbidden_error(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot
) -> None:
    """Driving the provider directly over `ForbiddenLiveHttpTransport` fails
    closed, proving the provider really does call `transport.send`.
    """
    provider = FutureSearchProvider(ForbiddenLiveHttpTransport(), _config())

    with pytest.raises(LiveCallForbiddenError):
        provider.forecast(market, baseline, 0, ())


def test_recorded_cassette_never_persists_the_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
) -> None:
    """A live-only secret a transport reads from the environment at send time
    never appears in the persisted cassette file -- `HttpRequest` has nowhere
    to carry it.
    """
    monkeypatch.setenv("FUTURESEARCH_API_KEY", "sk-SENTINEL-DO-NOT-PERSIST")
    cassette_path = tmp_path / "futuresearch_cassette.json"
    recorder = RecordingHttpCassette(
        transport=_KeyReadingFakeTransport(_body()), path=cassette_path
    )
    provider = FutureSearchProvider(recorder, _config())

    provider.forecast(market, baseline, 0, ())

    assert "sk-SENTINEL-DO-NOT-PERSIST" not in cassette_path.read_text(encoding="utf-8")


# --- Import boundary: no requests, no windbreak.config ---------------------------


def _imported_module_names(tree: ast.Module) -> frozenset[str]:
    """Collect every module name a parsed module actually imports.

    Walks the AST so only real `import x` / `from x import y` statements are
    considered -- a module name that merely appears inside a docstring or a
    comment (as `windbreak.config` does in the boundary prose) never counts.

    Args:
        tree: The parsed module AST.

    Returns:
        The set of fully-qualified module names the module imports.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return frozenset(names)


def test_futuresearch_module_never_imports_requests_or_windbreak_config() -> None:
    """Per the SPEC S8.3 sandbox boundary and the network-library-free CI
    contract, the module's own source never imports `requests` or
    `windbreak.config` (checked over real import statements via the AST, so a
    module name mentioned only in the boundary docstring never trips it).
    """
    module_file = futuresearch_module.__file__
    assert module_file is not None
    source = Path(module_file).read_text(encoding="utf-8")
    imported = _imported_module_names(ast.parse(source))

    assert not any(name.split(".")[0] == "requests" for name in imported)
    assert not any(
        name == "windbreak.config" or name.startswith("windbreak.config.")
        for name in imported
    )


# --- Pipeline citation threading: audit-only, never inflating verified count ----


def test_run_pipeline_default_path_has_no_provider_reported_citations(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
    make_fake_vote_transport: Callable[..., object],
) -> None:
    """With no `provider_factory` at all, the record's citations carry zero
    `provider_reported` entries -- the default fixture-vote path is
    unaffected.
    """
    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
    )

    assert not any(
        citation.source_type == PROVIDER_REPORTED_SOURCE_TYPE
        for citation in record.citations
    )


def test_run_pipeline_with_provider_factory_threads_reported_citations(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
) -> None:
    """A `provider_factory` building `FutureSearchProvider`s threads each
    surviving forecast's reported citations into the record, stamped
    `provider_reported`, carrying the reported url and publication date.
    """
    citations = json.dumps(
        [
            {
                "url": "https://provider.example/pipeline-report",
                "publication_date": None,
                "quoted_text": "provider-reported corroboration",
            }
        ]
    )
    transport = _StubHttpTransport(_body(citations=citations))
    config = _config()

    record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        provider_factory=lambda member: FutureSearchProvider(transport, config),
    )

    reported = [
        citation
        for citation in record.citations
        if citation.source_type == PROVIDER_REPORTED_SOURCE_TYPE
    ]
    assert reported
    assert all(
        citation.url == "https://provider.example/pipeline-report"
        for citation in reported
    )
    assert all(citation.publication_date is None for citation in reported)


def test_run_pipeline_provider_citations_do_not_inflate_verified_count_or_eligibility(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    research_tools: ResearchTools,
    make_fake_vote_transport: Callable[..., object],
) -> None:
    """A research-forecaster run's own reported citations never stand in for
    independently-verified ones, so it cannot reach live eligibility on them
    (SPEC S8.8).

    Since #556 the pipeline does not research at all for an ensemble that
    researches for itself, which makes this the sharper reading of the same
    invariant rather than a weaker one: the provider-driven run has *zero*
    independently-verified citations, reports one of its own, and is live-
    ineligible -- while the default-path control, which did research, verifies
    several and is eligible. A run that let a reported citation count would show
    up here as an eligible record with no verified citation behind it.
    """
    baseline_record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=research_tools,
    )
    citations = json.dumps(
        [
            {
                "url": "https://provider.example/pipeline-report-2",
                "publication_date": None,
                "quoted_text": "provider-reported corroboration",
            }
        ]
    )
    transport = _StubHttpTransport(_body(citations=citations))
    config = _config()

    provider_record = run_pipeline(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        created_at=created_at,
        research_tools=research_tools,
        provider_factory=lambda member: FutureSearchProvider(transport, config),
    )

    baseline_verified = [
        citation
        for citation in baseline_record.citations
        if citation.source_type != PROVIDER_REPORTED_SOURCE_TYPE
    ]
    provider_verified = [
        citation
        for citation in provider_record.citations
        if citation.source_type != PROVIDER_REPORTED_SOURCE_TYPE
    ]
    provider_reported = [
        citation
        for citation in provider_record.citations
        if citation.source_type == PROVIDER_REPORTED_SOURCE_TYPE
    ]
    assert baseline_verified != []
    assert baseline_record.eligible_for_live is True
    assert provider_verified == []
    assert len(provider_reported) == len(DEFAULT_VOTE_ENSEMBLE)
    assert provider_record.eligible_for_live is False


# --- Pipeline integration: sibling errors discard the vote, not the run (#189) --


def test_version_drift_through_pipeline_discards_vote_not_whole_run(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
) -> None:
    """A `ProviderVersionDriftError` raised by one ensemble member's real
    `FutureSearchProvider` is caught by `collect_model_votes` exactly like a
    `ProviderResponseRejectedError`: the drifted vote is discarded -- the run
    never crashes -- and ledgered as `FORECAST_OUTPUT_DISCARDED` with the
    version-drift failure code and the drifted response's fingerprint, while
    the other ensemble member's vote survives.

    Today `ProviderVersionDriftError` propagates uncaught out of
    `_collect_provider_forecasts` and crashes the whole run; this is the #189
    regression this test pins.
    """
    from windbreak.forecast.sanitize import RESPONSE_FAILURE_VERSION_DRIFT

    member_ok = EnsembleMember("openai", "futuresearch-ok-member", "2024-06-01")
    member_drift = EnsembleMember(
        "anthropic", "futuresearch-drift-member", "2024-06-01"
    )
    ensemble = (member_ok, member_drift)

    ok_body = _body()
    drifted_body = _body(forecaster_version=json.dumps("futuresearch-v2-unpinned"))
    config = _config()

    def provider_factory(member: EnsembleMemberLike) -> FutureSearchProvider:
        """Route the drifted member to the drifted body, others to a clean one."""
        if member.model_version == "futuresearch-drift-member":
            return FutureSearchProvider(_StubHttpTransport(drifted_body), config)
        return FutureSearchProvider(_StubHttpTransport(ok_body), config)

    ledger = InMemoryForecastLedger()

    votes = collect_model_votes(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        ensemble=ensemble,
        provider_factory=provider_factory,
    )

    # Exactly one vote survives (the clean member); the drifted one is dropped.
    # The survivor's `model_version` is the response-stamped `forecaster_version`
    # (`_PINNED_VERSION`), never the ensemble member's -- so which member was
    # discarded is asserted through the ledger event's `provider`/`model_version`
    # (recorded from the member), not the surviving vote.
    assert len(votes) == 1
    assert votes[0].model_version == _PINNED_VERSION

    events = ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)
    assert len(events) == 1
    event = events[0]
    assert event.payload["failure"] == RESPONSE_FAILURE_VERSION_DRIFT
    assert event.payload["provider"] == member_drift.provider
    assert event.payload["model_version"] == member_drift.model_version
    assert (
        event.payload["response_fingerprint"]
        == hashlib.sha256(drifted_body.encode("utf-8")).hexdigest()
    )


def test_response_rejection_through_pipeline_discards_and_ledgers(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
) -> None:
    """A `ProviderResponseRejectedError` raised by a real `FutureSearchProvider`
    wired in through `provider_factory` is caught end-to-end by
    `collect_model_votes`: the rejected vote is discarded -- the run never
    crashes -- and ledgered as `FORECAST_OUTPUT_DISCARDED` with the exact
    failure code and response fingerprint, while the other member's vote
    survives.

    Unlike `_rejected` (which drives a single-member unit call directly),
    this exercises the real `provider_factory` wiring through
    `collect_model_votes` end-to-end -- the #189 HIGH finding.
    """
    member_ok = EnsembleMember("openai", "futuresearch-ok-member-2", "2024-06-01")
    member_rejected = EnsembleMember(
        "anthropic", "futuresearch-rejected-member", "2024-06-01"
    )
    ensemble = (member_ok, member_rejected)

    ok_body = _body()
    rejected_body = _body(probability="true")  # bool: malformed, never coerced
    config = _config()

    def provider_factory(member: EnsembleMemberLike) -> FutureSearchProvider:
        """Route the rejected member to the malformed body, others to a clean one."""
        if member.model_version == "futuresearch-rejected-member":
            return FutureSearchProvider(_StubHttpTransport(rejected_body), config)
        return FutureSearchProvider(_StubHttpTransport(ok_body), config)

    ledger = InMemoryForecastLedger()

    votes = collect_model_votes(
        market,
        baseline,
        transport=ForbiddenLiveTransport(),
        ledger=ledger,
        created_at=created_at,
        ensemble=ensemble,
        provider_factory=provider_factory,
    )

    # Exactly one vote survives (the clean member); the rejected one is dropped.
    # As above, the survivor's `model_version` is the response-stamped
    # `forecaster_version`, so the discarded member is identified via the ledger
    # event payload (recorded from the member), not the surviving vote.
    assert len(votes) == 1
    assert votes[0].model_version == _PINNED_VERSION

    events = ledger.events_by_type(FORECAST_OUTPUT_DISCARDED_EVENT)
    assert len(events) == 1
    event = events[0]
    assert event.payload["failure"] == RESPONSE_FAILURE_MALFORMED_VOTE_JSON
    assert event.payload["provider"] == member_rejected.provider
    assert event.payload["model_version"] == member_rejected.model_version
    assert (
        event.payload["response_fingerprint"]
        == hashlib.sha256(rejected_body.encode("utf-8")).hexdigest()
    )
