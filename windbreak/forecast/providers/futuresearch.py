"""The hosted research-forecaster :class:`ForecastProvider` (SPEC S8.9, ADR-0005).

:class:`FutureSearchProvider` is the first "research-forecaster" ensemble member
(ADR-0005 family (b)): a second :class:`ForecastProvider` behind the identical
seam :class:`FixtureVoteProvider` satisfies, but talking to a FutureSearch-shaped
hosted research API through the :class:`HttpTransport` seam instead of the
LLM-completion ``LlmTransport`` one. Per ADR-0005 S1(b) a research forecaster
does its own web research server-side, so this provider *ignores* the
pipeline-supplied ``quotes`` entirely -- proven by the request body being a pure
function of the market/baseline question fields, never the quotes.

The response is parsed with :class:`decimal.Decimal` (never a Python ``float``),
so a fractional probability never touches a float, and the module is stdlib-only
and float-free -- it sits on the probability/money path guarded by
``scripts/lint_no_floats.py``. Per the SPEC S8.3 sandbox boundary it never
imports ``windbreak.config``; and it never imports ``requests`` -- the network
transport is dependency-injected, so CI stays network-library-free on this path
(the live recorder lives in ``scripts/record_futuresearch_cassette.py``).

Fixed response-processing order (mirroring ``validate_vote_response``'s
"injection screen before schema" precedent):

1. :func:`~windbreak.forecast.sanitize.screen_untrusted_text` over the *entire*
   raw response body first (delimiter forgery / tool-call lure).
2. JSON-parse with ``parse_float=Decimal`` and a ``parse_constant`` hook that
   rejects ``Infinity``/``-Infinity``/``NaN`` outright.
3. ``probability`` -> integer ppm (round-half-even, no re-clamp).
4. ``rationale`` -> a non-empty string of at most ``MAX_RATIONALE_CHARS``.
5. ``citations`` -> a tuple of
   :class:`~windbreak.forecast.providers.base.ProviderCitation`.
6. ``forecaster_version`` -> a pinned-version drift check.
7. ``cost_usd`` -> micros (round-ceiling), defaulting fail-closed to the
   per-call ceiling.

Any screening/parse/domain failure raises
:class:`~windbreak.forecast.providers.base.ProviderResponseRejectedError`,
carrying only a fingerprint of the untrusted text, never the raw bytes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING, Final, NoReturn

from windbreak.forecast.providers.base import (
    ProviderCitation,
    ProviderForecast,
    ProviderHTTPError,
    ProviderResponseRejectedError,
    ProviderVersionDriftError,
    fingerprint_response,
)
from windbreak.forecast.providers.http_cassettes import HttpRequest
from windbreak.forecast.sanitize import (
    MAX_RATIONALE_CHARS,
    RESPONSE_FAILURE_INVALID_RATIONALE,
    RESPONSE_FAILURE_MALFORMED_VOTE_JSON,
    RESPONSE_FAILURE_PROBABILITY_OUT_OF_RANGE,
    extract_quote,
    screen_untrusted_text,
)

if TYPE_CHECKING:
    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.providers.http_cassettes import HttpTransport
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sanitize import ResearchQuote

#: The logger the permissive version-drift path warns through. The M0
#: ``logging_setup`` layer redacts secrets; the API key is never in scope here.
_LOGGER = logging.getLogger(__name__)

#: The provider identifier stamped on every :class:`ProviderForecast`.
_PROVIDER_NAME = "futuresearch"

#: The declared training cutoff stamped on every forecast: a hosted research
#: forecaster manages its own model server-side and reports no cutoff, so this
#: is a fixed, honest "managed elsewhere" marker rather than an invented date.
_TRAINING_CUTOFF = "server-managed"

#: The HTTP method every FutureSearch request uses.
_REQUEST_METHOD = "POST"

#: Inclusive lower bound of the HTTP success (2xx) status range: a response
#: whose status falls below it is rejected fast, before any body parsing.
_HTTP_SUCCESS_MIN: Final = 200

#: Exclusive upper bound of the HTTP success (2xx) status range: a response
#: whose status reaches it (300+) is rejected fast, before any body parsing.
_HTTP_SUCCESS_MAX_EXCLUSIVE: Final = 300

#: ppm scale: one full probability (1.0) is 1_000_000 ppm. A ``Decimal`` so the
#: whole conversion stays on the exact-decimal path, never a float.
_PPM_SCALE = Decimal(1_000_000)

#: Micro-dollars per dollar, for the exact cost (dollars -> micros) conversion.
_MICROS_PER_DOLLAR = Decimal(1_000_000)

#: Inclusive probability domain bounds a reported value must fall within, as
#: ``Decimal`` so the range check stays on the exact-decimal path.
_MIN_PROBABILITY = Decimal(0)
_MAX_PROBABILITY = Decimal(1)

#: Response-body top-level keys.
_PROBABILITY_KEY = "probability"
_RATIONALE_KEY = "rationale"
_CITATIONS_KEY = "citations"
_FORECASTER_VERSION_KEY = "forecaster_version"
_COST_KEY = "cost_usd"

#: Citation-object keys.
_URL_KEY = "url"
_PUBLICATION_DATE_KEY = "publication_date"
_QUOTED_TEXT_KEY = "quoted_text"

#: Request-body question-field keys (a closed, deterministic set).
_TICKER_KEY = "ticker"
_TITLE_KEY = "title"
_RESOLUTION_CRITERIA_KEY = "resolution_criteria"
_BASELINE_PRICE_PIPS_KEY = "baseline_price_pips"
_VOTE_INDEX_KEY = "vote_index"

#: The UTC-offset spelling ``datetime.fromisoformat`` accepts in place of ``Z``.
_UTC_SUFFIX = "+00:00"
_ZULU_SUFFIX = "Z"


@dataclass(frozen=True, slots=True)
class FutureSearchProviderConfig:
    """The pinned configuration one :class:`FutureSearchProvider` runs under.

    Attributes:
        endpoint_url: The forecast endpoint the request is POSTed to.
        pinned_forecaster_versions: The operator-pinned forecaster versions a
            reported version must belong to (else drift).
        api_key_env: The environment variable a *live* transport reads the API
            key from; never touched here (the transport is injected).
        per_call_ceiling_micros: The reported-cost fallback, in micros, charged
            when a response reports no usable ``cost_usd``.
        reject_on_version_drift: Whether an unpinned reported version raises
            (strict, the default) or proceeds with a logged warning.
    """

    endpoint_url: str
    pinned_forecaster_versions: tuple[str, ...]
    api_key_env: str = "FUTURESEARCH_API_KEY"
    per_call_ceiling_micros: int = 2_000_000
    reject_on_version_drift: bool = True


def _reject(failure_code: str, fingerprint: str) -> NoReturn:
    """Raise a fingerprint-only rejection, never leaking the raw response text.

    Args:
        failure_code: The ``RESPONSE_FAILURE_*`` code describing the failure.
        fingerprint: The rejected response's sha256 fingerprint.

    Raises:
        ProviderResponseRejectedError: Always.
    """
    raise ProviderResponseRejectedError(failure_code, fingerprint)


def _reject_constant(token: str) -> NoReturn:
    """Reject a non-finite JSON constant token (``Infinity``/``NaN``).

    Installed as ``json.loads(..., parse_constant=...)`` so a non-standard
    constant -- which ``json.loads`` would otherwise materialize as a real
    Python ``float`` -- fails the parse instead of smuggling a float onto the
    probability path.

    Args:
        token: The non-finite constant token the parser encountered.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"non-finite JSON constant is banned, got {token!r}")


def _canonical_request_body(
    market: NormalizedMarket, baseline: BaselineQuoteSnapshot, vote_index: int
) -> str:
    """Serialize the question fields into a deterministic request body.

    The body is a pure function of the market/baseline question and the vote
    index -- never the pipeline ``quotes`` -- so two calls that differ only in
    their supplied quotes build byte-identical requests (ADR-0005 S1(b)). Keys
    are sorted and separators are space-free, matching the ledger's canonical
    JSON form so the body (and thus the request hash) is byte-stable.

    Args:
        market: The market under forecast.
        baseline: The baseline quote snapshot.
        vote_index: The zero-based index of this vote in the ensemble.

    Returns:
        The canonical JSON request-body text.
    """
    return json.dumps(
        {
            _TICKER_KEY: market.ticker,
            _TITLE_KEY: market.title,
            _RESOLUTION_CRITERIA_KEY: market.resolution_criteria,
            _BASELINE_PRICE_PIPS_KEY: baseline.price_pips,
            _VOTE_INDEX_KEY: vote_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_payload(raw_body: str, fingerprint: str) -> dict[str, object]:
    """Parse the raw response body into a mapping, float-free and fail-closed.

    Args:
        raw_body: The raw, already injection-screened response body text.
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        The parsed top-level JSON object.

    Raises:
        ProviderResponseRejectedError: If the body is not a JSON object, is
            malformed, or carries a non-finite numeric constant.
    """
    try:
        payload = json.loads(
            raw_body, parse_float=Decimal, parse_constant=_reject_constant
        )
    except ValueError:
        _reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    if not isinstance(payload, dict):
        _reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    return payload


def _extract_probability_ppm(payload: dict[str, object], fingerprint: str) -> int:
    """Convert the reported probability into an integer ppm value.

    A JSON integer ``0``/``1`` or a ``Decimal`` in ``[0, 1]`` is accepted; any
    other type, or a numeric value outside ``[0, 1]``, is rejected (never
    clamped). The conversion uses banker's rounding (``ROUND_HALF_EVEN``), so a
    value landing exactly halfway between two ppm integers rounds to the nearest
    even one -- directionally unbiased, never systematically sharpening the
    estimate -- and stays exact for values like ``0.1`` that mangle through a
    binary ``float``. No re-clamp is applied: the ``[0, 1_000_000]`` guard lives
    in :meth:`ProviderForecast.__post_init__`.

    Args:
        payload: The parsed response object.
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        The probability in parts-per-million.

    Raises:
        ProviderResponseRejectedError: If the probability is the wrong type or
            outside ``[0, 1]``.
    """
    value = payload.get(_PROBABILITY_KEY)
    if isinstance(value, bool) or not isinstance(value, int | Decimal):
        _reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    numeric = Decimal(value)
    if not _MIN_PROBABILITY <= numeric <= _MAX_PROBABILITY:
        _reject(RESPONSE_FAILURE_PROBABILITY_OUT_OF_RANGE, fingerprint)
    scaled = (numeric * _PPM_SCALE).to_integral_value(rounding=ROUND_HALF_EVEN)
    return int(scaled)


def _extract_rationale(payload: dict[str, object], fingerprint: str) -> str:
    """Return the reported rationale after length and injection screening.

    Args:
        payload: The parsed response object.
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        The validated rationale text.

    Raises:
        ProviderResponseRejectedError: If the rationale is missing, not a
            non-empty string of at most ``MAX_RATIONALE_CHARS``, or carries an
            injection artifact that a JSON-escape hid from the whole-body screen.
    """
    value = payload.get(_RATIONALE_KEY)
    if not isinstance(value, str) or not value or len(value) > MAX_RATIONALE_CHARS:
        _reject(RESPONSE_FAILURE_INVALID_RATIONALE, fingerprint)
    injection = screen_untrusted_text(value)
    if injection is not None:
        _reject(injection, fingerprint)
    return value


def _parse_publication_date(value: object, fingerprint: str) -> datetime | None:
    """Parse a citation's reported publication date, or ``None`` for ``null``.

    Args:
        value: The raw ``publication_date`` value (an ISO-8601 string or
            ``None``).
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        The parsed timezone-aware datetime, or ``None`` when reported ``null``.

    Raises:
        ProviderResponseRejectedError: If the value is neither ``None`` nor a
            parseable ISO-8601 string, or if it parses to a *naive* datetime
            (issue #329).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        _reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    try:
        parsed = datetime.fromisoformat(value.replace(_ZULU_SUFFIX, _UTC_SUFFIX))
    except ValueError:
        _reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    # A string with no UTC offset -- "2024-03-15T00:00:00", or a bare
    # "2024-03-15" -- parses cleanly and yields tzinfo None, silently breaking
    # the "timezone-aware datetime" contract above. Unguarded it reaches
    # `_build_citation` and becomes a structurally-invalid `ProviderCitation`,
    # caught only much later as an S8.8 `FAILURE_PUBLICATION_DATE_INVALID` in
    # `windbreak/forecast/citations.py` -- so reject it here, at the provider
    # boundary, exactly like every other malformed date shape.
    #
    # The offset is not assumed: `windbreak/forecast/pubdate.py` degrades a
    # naive value rather than fabricating a timezone, and inventing UTC here
    # would assert a fact the provider never reported.
    #
    # `utcoffset()` is consulted rather than testing `tzinfo is None` alone,
    # because a tzinfo whose `utcoffset()` returns None is still naive by
    # Python's own definition.
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        _reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    return parsed


def _build_citation(entry: object, fingerprint: str) -> ProviderCitation:
    """Build one :class:`ProviderCitation` from a reported citation object.

    The quoted text is injection-screened (catching a JSON-escaped artifact the
    whole-body screen could miss) and length-capped through
    :func:`~windbreak.forecast.sanitize.extract_quote`.

    Args:
        entry: The raw citation object.
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        The constructed provider-reported citation.

    Raises:
        ProviderResponseRejectedError: If the object is malformed or its quoted
            text carries an injection artifact.
    """
    if not isinstance(entry, dict):
        _reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    url = entry.get(_URL_KEY)
    quoted = entry.get(_QUOTED_TEXT_KEY)
    if not isinstance(url, str) or not isinstance(quoted, str):
        _reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    injection = screen_untrusted_text(quoted)
    if injection is not None:
        _reject(injection, fingerprint)
    publication_date = _parse_publication_date(
        entry.get(_PUBLICATION_DATE_KEY), fingerprint
    )
    return ProviderCitation(
        url=url, publication_date=publication_date, quoted_text=extract_quote(quoted)
    )


def _extract_citations(
    payload: dict[str, object], fingerprint: str
) -> tuple[ProviderCitation, ...]:
    """Convert the reported citations array into a tuple of citations.

    Args:
        payload: The parsed response object.
        fingerprint: The response fingerprint, for any rejection.

    Returns:
        The reported citations, in order (empty when the field is absent).

    Raises:
        ProviderResponseRejectedError: If ``citations`` is present but not a
            JSON array, or any element is malformed.
    """
    raw = payload.get(_CITATIONS_KEY, [])
    if not isinstance(raw, list):
        _reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
    return tuple(_build_citation(entry, fingerprint) for entry in raw)


def _extract_cost_micros(payload: dict[str, object], ceiling_micros: int) -> int:
    """Convert the reported cost (dollars) to micros, defaulting fail-closed.

    A present, numeric, non-negative ``cost_usd`` converts to micros with
    ``ROUND_CEILING`` (dollars are never undercounted). A missing, ``null``,
    non-numeric, or negative value does *not* reject the response -- it only
    defaults the reported cost to ``ceiling_micros``.

    Args:
        payload: The parsed response object.
        ceiling_micros: The per-call ceiling charged when no usable cost is
            reported.

    Returns:
        The reported cost in micros, or ``ceiling_micros`` on any bad value.
    """
    value = payload.get(_COST_KEY)
    if isinstance(value, bool) or not isinstance(value, int | Decimal) or value < 0:
        return ceiling_micros
    micros = (Decimal(value) * _MICROS_PER_DOLLAR).to_integral_value(
        rounding=ROUND_CEILING
    )
    return int(micros)


class FutureSearchProvider:
    """A hosted research-forecaster :class:`ForecastProvider` over HTTP."""

    #: This provider researches server-side and never reads the pipeline's
    #: quotes (ADR-0005 S1(b)) -- the same fact ``forecast`` states by deleting
    #: its ``quotes`` argument, declared here where the pipeline can read it
    #: *before* deciding whether to research at all (issue #556). The two must
    #: agree: a provider that consumed ``quotes`` while declaring this would
    #: vote on evidence the pipeline was told not to gather.
    performs_own_research = True

    def __init__(
        self, transport: HttpTransport, config: FutureSearchProviderConfig
    ) -> None:
        """Bind the HTTP transport and pinned configuration.

        Args:
            transport: The HTTP transport (fake, recording, replay, or
                forbidden-live) the request is sent through.
            config: The pinned endpoint/version/cost configuration.
        """
        self._transport = transport
        self._config = config

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[ResearchQuote, ...],
    ) -> ProviderForecast:
        """Obtain, screen, and parse one research-forecaster vote.

        Per ADR-0005 S1(b) a research forecaster does its own web research
        server-side, so ``quotes`` is accepted (the 4-arg protocol requires it)
        but deliberately never threaded into the request or the result. The
        fields are extracted in the SPEC's fixed order (probability, rationale,
        citations, version, cost).

        Args:
            market: The market under forecast.
            baseline: The baseline quote snapshot.
            vote_index: The zero-based index of this vote in the ensemble.
            quotes: The pipeline's sanitized web quotes -- ignored here.

        Returns:
            The structured forecast parsed from a clean, schema-valid response.

        Raises:
            ProviderHTTPError: If the transport returns a non-2xx status
                (rejected fast, before any body parsing), carrying that status
                code so the retry layer can classify a transient upstream
                failure (issue #269).
            ProviderResponseRejectedError: If the raw response fails the
                injection screen, JSON parse, or a field's domain check; the
                error carries the failure code and response fingerprint only.
            ProviderVersionDriftError: If the reported forecaster version is off
                the pinned set and the config rejects drift.
        """
        del quotes  # research forecaster does its own research (ADR-0005 S1(b))
        request = HttpRequest(
            method=_REQUEST_METHOD,
            url=self._config.endpoint_url,
            body=_canonical_request_body(market, baseline, vote_index),
        )
        response = self._transport.send(request)
        raw_body = response.body
        fingerprint = fingerprint_response(raw_body)
        if not (
            _HTTP_SUCCESS_MIN <= response.status_code < _HTTP_SUCCESS_MAX_EXCLUSIVE
        ):
            raise ProviderHTTPError(response.status_code, fingerprint)
        injection = screen_untrusted_text(raw_body)
        if injection is not None:
            _reject(injection, fingerprint)
        payload = _parse_payload(raw_body, fingerprint)
        probability_ppm = _extract_probability_ppm(payload, fingerprint)
        rationale = _extract_rationale(payload, fingerprint)
        citations = _extract_citations(payload, fingerprint)
        model_version = self._resolve_model_version(payload, fingerprint)
        cost_micros = _extract_cost_micros(
            payload, self._config.per_call_ceiling_micros
        )
        return ProviderForecast(
            probability_ppm=probability_ppm,
            rationale_summary=rationale,
            citations=citations,
            cost_micros=cost_micros,
            provider=_PROVIDER_NAME,
            model_version=model_version,
            training_cutoff=_TRAINING_CUTOFF,
            response_fingerprint=fingerprint,
        )

    def _resolve_model_version(
        self, payload: dict[str, object], fingerprint: str
    ) -> str:
        """Enforce the pinned-forecaster-version drift policy.

        A missing or non-string ``forecaster_version`` is rejected as malformed
        (:data:`RESPONSE_FAILURE_MALFORMED_VOTE_JSON`) in *both* strict and
        permissive modes -- it is a schema violation, not drift. Only a version
        that is present and a string but outside the pinned set is the drift
        case: it raises :class:`ProviderVersionDriftError` under the strict
        default, or -- when drift is permitted -- proceeds with one logged
        warning and the *reported* version stamped onto the forecast. A reported
        version inside the pinned set is returned as-is.

        Args:
            payload: The parsed response object.
            fingerprint: The response fingerprint, for any rejection or drift.

        Returns:
            The version string to stamp onto the forecast.

        Raises:
            ProviderResponseRejectedError: If ``forecaster_version`` is missing
                or not a string (malformed, in both modes).
            ProviderVersionDriftError: If the version is present, a string, but
                unpinned and the config rejects drift; carries the response
                fingerprint so the pipeline can discard and ledger it per-vote.
        """
        version = payload.get(_FORECASTER_VERSION_KEY)
        if not isinstance(version, str):
            _reject(RESPONSE_FAILURE_MALFORMED_VOTE_JSON, fingerprint)
        pinned = self._config.pinned_forecaster_versions
        if version in pinned:
            return version
        if self._config.reject_on_version_drift:
            raise ProviderVersionDriftError(version, pinned, fingerprint)
        _LOGGER.warning(
            "futuresearch reported forecaster version %r off the pinned set %r; "
            "proceeding permissively and stamping the reported version",
            version,
            pinned,
        )
        return version
