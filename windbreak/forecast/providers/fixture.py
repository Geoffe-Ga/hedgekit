"""The first, network-free :class:`ForecastProvider` implementation (SPEC S8.9).

:class:`FixtureVoteProvider` wraps an
:class:`~windbreak.forecast.cassettes.LlmTransport` (a fake, a recording
cassette, a replay cassette, or a forbidden-live transport) plus one ensemble
member. Its :meth:`~FixtureVoteProvider.forecast` builds the deterministic vote
request, obtains the raw completion, and screens it through
:func:`windbreak.forecast.sanitize.validate_vote_response`: a rejected response
raises :class:`ProviderResponseRejectedError` (fingerprint only, never the raw
text), while a clean one is parsed into a :class:`ProviderForecast` carrying the
parsed probability/rationale, the member's provenance, a zero fixture cost, no
citations, and a fingerprint of the raw response.

The module is stdlib-only and float-free, and -- per the SPEC S8.3 sandbox
boundary -- never imports ``windbreak.config``: its member is accepted
structurally via :class:`EnsembleMemberLike`.

The parsed vote's ``abstain`` flag is threaded onto the returned
:class:`ProviderForecast`; the aggregation path excludes an abstaining member
from the median (issue #241).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from windbreak.forecast.cassettes import LlmRequest
from windbreak.forecast.providers.base import (
    ProviderForecast,
    ProviderResponseRejectedError,
    build_vote_prompt,
    fingerprint_response,
)
from windbreak.forecast.sanitize import parse_vote_response, validate_vote_response

if TYPE_CHECKING:
    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.cassettes import LlmTransport
    from windbreak.forecast.providers.base import EnsembleMemberLike
    from windbreak.forecast.records import BaselineQuoteSnapshot
    from windbreak.forecast.sanitize import ResearchQuote

#: The billed cost of a network-free fixture vote, in micro-dollars: none.
_FIXTURE_COST_MICROS = 0


class FixtureVoteProvider:
    """A network-free :class:`ForecastProvider` over a transport + member."""

    def __init__(self, transport: LlmTransport, member: EnsembleMemberLike) -> None:
        """Bind the transport and ensemble member this provider votes through.

        Args:
            transport: The LLM transport (fake, recording, replay, or forbidden).
            member: The ensemble member whose provenance stamps each forecast.
        """
        self._transport = transport
        self._member = member

    def forecast(
        self,
        market: NormalizedMarket,
        baseline: BaselineQuoteSnapshot,
        vote_index: int,
        quotes: tuple[ResearchQuote, ...],
    ) -> ProviderForecast:
        """Obtain, screen, and parse one ensemble vote into a forecast.

        Args:
            market: The market under forecast.
            baseline: The baseline quote snapshot.
            vote_index: The zero-based index of this vote in the ensemble.
            quotes: The sanitized web quotes to thread into the vote prompt.

        Returns:
            The structured forecast parsed from a clean, schema-valid response.

        Raises:
            ProviderResponseRejectedError: If the raw response fails the vote
                screen (injection artifact or schema violation); the error
                carries the failure code and the response fingerprint only.
        """
        request = LlmRequest(
            provider=self._member.provider,
            model_version=self._member.model_version,
            prompt=build_vote_prompt(market, baseline, vote_index, quotes),
        )
        response = self._transport.complete(request)
        fingerprint = fingerprint_response(response)
        failure = validate_vote_response(response)
        if failure is not None:
            raise ProviderResponseRejectedError(failure, fingerprint)
        parsed = parse_vote_response(response)
        return ProviderForecast(
            probability_ppm=parsed.probability_ppm,
            rationale_summary=parsed.rationale_summary,
            citations=(),
            cost_micros=_FIXTURE_COST_MICROS,
            provider=self._member.provider,
            model_version=self._member.model_version,
            training_cutoff=self._member.training_cutoff,
            response_fingerprint=fingerprint,
            abstain=parsed.abstain,
        )
