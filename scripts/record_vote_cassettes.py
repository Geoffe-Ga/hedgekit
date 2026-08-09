#!/usr/bin/env python3
"""Operator-run recorder for pinned-LLM vote cassettes (issue #191).

CI never makes a live OpenAI or Anthropic call: the forecast pipeline drives the
:class:`~windbreak.forecast.providers.fixture.FixtureVoteProvider` through a
replay-only :class:`~windbreak.forecast.cassettes.ReplayCassette`, which fails
closed on any unrecorded request. This developer-run script is the *only* place
a live vote call is ever made, and the *only* place ``requests`` and the process
environment appear on this path -- deliberately kept out of the ``windbreak``
package so CI stays network-library-free and the SPEC S8.3 sandbox boundary is
never crossed.

Workflow:

1. An operator exports the provider API keys (``ANTHROPIC_API_KEY`` and
   ``OPENAI_API_KEY``) and runs this script (via ``scripts/record-cassettes.sh``)
   with the market fields the prompt is built from.
2. For each member of
   :data:`~windbreak.forecast.providers.DEFAULT_VOTE_ENSEMBLE`, a live,
   allowlisted, redirect-free transport dials that member's provider once,
   injecting the key as a *send-time* header -- never onto the header-free
   :class:`~windbreak.forecast.providers.http_cassettes.HttpRequest`, so the key
   can never be persisted into the cassette.
3. The in-package adapter is wrapped in a
   :class:`~windbreak.forecast.cassettes.RecordingCassette`, and each vote is
   driven end-to-end through :class:`FixtureVoteProvider`, so the operator gets
   immediate #184 schema validation (a malformed live response fails loudly here,
   before anything is committed).
4. The recorded cassette is scrubbed/inspected, committed, and thereafter
   replayed in CI, offline.

Each live transport is wrapped exactly like
``windbreak.connector.kalshi.client._RedirectFreeSession``: redirects are refused
(``allow_redirects=False``) so an on-path responder can never steer the recorder
to another host, an integer timeout bounds the dial, and every URL is screened
against an :class:`~windbreak.net.allowlist.OutboundAllowlist` built over the
pinned provider host before any byte leaves the process.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from windbreak.connector.models import NormalizedMarket
from windbreak.forecast.cassettes import RecordingCassette
from windbreak.forecast.providers import (
    ANTHROPIC_MESSAGES_ENDPOINT,
    DEFAULT_VOTE_ENSEMBLE,
    OPENAI_CHAT_ENDPOINT,
    AnthropicMessagesTransport,
    FixtureVoteProvider,
    HttpResponse,
    OpenAiChatTransport,
)
from windbreak.forecast.records import BaselineQuoteSnapshot
from windbreak.net.allowlist import OutboundAllowlist

if TYPE_CHECKING:
    from windbreak.forecast.cassettes import LlmRequest, LlmTransport
    from windbreak.forecast.providers.http_cassettes import HttpRequest

#: Per-request timeout, in whole seconds (an int -- no float on any path here).
_TIMEOUT_SECONDS = 30

#: Provider identifiers, matching each ensemble member's ``provider`` field.
_ANTHROPIC_PROVIDER = "anthropic"
_OPENAI_PROVIDER = "openai"

#: Pinned provider API hosts (the only hosts egress is ever permitted to).
_ANTHROPIC_HOST = "api.anthropic.com"
_OPENAI_HOST = "api.openai.com"

#: Environment variables each provider's live key is *read from* -- these are
#: variable names, never the secrets themselves, and the values are never logged.
_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
_OPENAI_KEY_ENV = "OPENAI_API_KEY"

#: Anthropic requires a pinned API version header on every Messages call.
_ANTHROPIC_VERSION_HEADER = "anthropic-version"
_ANTHROPIC_VERSION = "2023-06-01"

#: Shared request headers.
_CONTENT_TYPE_HEADER = "content-type"
_JSON_CONTENT_TYPE = "application/json"
_ANTHROPIC_KEY_HEADER = "x-api-key"
_OPENAI_AUTH_HEADER = "authorization"


class _LiveHttpTransport:
    """A live ``requests``-backed transport: allowlisted, redirect-free, keyed.

    The API key is held here inside the header map and injected on each send, so
    it never touches the header-free
    :class:`~windbreak.forecast.providers.http_cassettes.HttpRequest` and can
    never be written into a recorded cassette.
    """

    def __init__(
        self,
        *,
        headers: dict[str, str],
        allowlist: OutboundAllowlist,
        session: requests.Session,
    ) -> None:
        """Store the send-time headers, the egress allowlist, and the session.

        Args:
            headers: The headers (including the API key) injected on each send.
            allowlist: The egress allowlist screening every outbound URL.
            session: The live ``requests`` session to dial through.
        """
        self._headers = headers
        self._allowlist = allowlist
        self._session = session

    def send(self, request: HttpRequest) -> HttpResponse:
        """Screen the URL, dial the endpoint once, and return its response.

        Args:
            request: The request to send.

        Returns:
            The endpoint's response as an
            :class:`~windbreak.forecast.providers.http_cassettes.HttpResponse`.

        Raises:
            EgressDeniedError: If the URL is off the allowlist.
        """
        self._allowlist.require(request.url)
        response = self._session.request(
            request.method,
            request.url,
            data=request.body.encode("utf-8"),
            headers=self._headers,
            timeout=_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
        return HttpResponse(status_code=response.status_code, body=response.text)


class _RoutingLlmTransport:
    """An :class:`LlmTransport` dispatching each request to its provider adapter.

    A single :class:`~windbreak.forecast.cassettes.RecordingCassette` records the
    whole ensemble to one cassette file by wrapping this router, which forwards
    each :class:`~windbreak.forecast.cassettes.LlmRequest` to the adapter keyed by
    its ``provider`` field.
    """

    def __init__(self, adapters: dict[str, LlmTransport]) -> None:
        """Store the per-provider completion adapters.

        Args:
            adapters: A mapping of provider identifier to its live adapter.
        """
        self._adapters = adapters

    def complete(self, request: LlmRequest) -> str:
        """Forward ``request`` to the adapter for its provider.

        Args:
            request: The completion request to route.

        Returns:
            The routed adapter's completion text.

        Raises:
            KeyError: If no adapter is registered for the request's provider.
        """
        return self._adapters[request.provider].complete(request)


def _read_key(env_var: str) -> str:
    """Read a provider API key from the environment, or exit with a clear message.

    Args:
        env_var: The environment variable the key is read from.

    Returns:
        The API key value (never logged or printed).
    """
    try:
        return os.environ[env_var]
    except KeyError:
        sys.exit(f"error: environment variable {env_var} is not set")


def _build_live_transports() -> dict[str, LlmTransport]:
    """Build the per-provider live completion adapters over live HTTP transports.

    Returns:
        A mapping of provider identifier to its live
        :class:`~windbreak.forecast.cassettes.LlmTransport` adapter.
    """
    anthropic_http = _LiveHttpTransport(
        headers={
            _ANTHROPIC_KEY_HEADER: _read_key(_ANTHROPIC_KEY_ENV),
            _ANTHROPIC_VERSION_HEADER: _ANTHROPIC_VERSION,
            _CONTENT_TYPE_HEADER: _JSON_CONTENT_TYPE,
        },
        allowlist=OutboundAllowlist(frozenset({_ANTHROPIC_HOST})),
        session=requests.Session(),
    )
    openai_http = _LiveHttpTransport(
        headers={
            _OPENAI_AUTH_HEADER: f"Bearer {_read_key(_OPENAI_KEY_ENV)}",
            _CONTENT_TYPE_HEADER: _JSON_CONTENT_TYPE,
        },
        allowlist=OutboundAllowlist(frozenset({_OPENAI_HOST})),
        session=requests.Session(),
    )
    return {
        _ANTHROPIC_PROVIDER: AnthropicMessagesTransport(
            anthropic_http, endpoint_url=ANTHROPIC_MESSAGES_ENDPOINT
        ),
        _OPENAI_PROVIDER: OpenAiChatTransport(
            openai_http, endpoint_url=OPENAI_CHAT_ENDPOINT
        ),
    }


def _build_market(args: argparse.Namespace) -> NormalizedMarket:
    """Build the market whose question fields the vote prompt is built from.

    Only the ticker, title, resolution criteria, and close time reach the prompt;
    every other structural field is fixed to a recording-only placeholder.

    Args:
        args: The parsed CLI arguments.

    Returns:
        A :class:`~windbreak.connector.models.NormalizedMarket`.
    """
    return NormalizedMarket(
        exchange="operator-recording",
        ticker=args.ticker,
        event_ticker=args.ticker,
        title=args.title,
        resolution_criteria=args.resolution_criteria,
        category="recording",
        close_time=datetime.fromisoformat(args.close_time),
        expected_resolution_time=None,
        market_type="fully_collateralized_binary",
        price_tick_pips=100,
        min_order_contract_centis=100,
        fractional_trading_enabled=False,
        mutually_exclusive_group_id=None,
        jurisdiction_status="eligible",
        raw_exchange_payload_hash="sha256:operator-recording",
        volume_24h_micros=0,
    )


def record_vote_cassettes(
    *,
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    cassette_path: Path,
    transports: dict[str, LlmTransport],
) -> None:
    """Record one live vote per ensemble member into a single cassette file.

    Args:
        market: The market whose question fields build each prompt.
        baseline: The baseline quote snapshot threaded into each prompt.
        cassette_path: The cassette file every recorded vote is written into.
        transports: The per-provider live adapters to route each vote through.
    """
    recorder = RecordingCassette(
        transport=_RoutingLlmTransport(transports), path=cassette_path
    )
    for index, member in enumerate(DEFAULT_VOTE_ENSEMBLE):
        provider = FixtureVoteProvider(recorder, member)
        forecast = provider.forecast(market, baseline, index, ())
        print(
            f"recorded vote {index} for {member.provider}:{member.model_version} "
            f"-> {forecast.probability_ppm} ppm"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the recorder's command-line arguments.

    Args:
        argv: The argument vector, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="Market ticker.")
    parser.add_argument("--title", required=True, help="Market question text.")
    parser.add_argument(
        "--resolution-criteria", required=True, help="Verbatim resolution criteria."
    )
    parser.add_argument(
        "--close-time", required=True, help="Market close time (ISO 8601, tz-aware)."
    )
    parser.add_argument(
        "--baseline-price-pips",
        required=True,
        type=int,
        help="Baseline executable price, in pips.",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Cassette file to write."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the vote-cassette recorder CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` once every ensemble member's vote is recorded.
    """
    args = _parse_args(argv)
    market = _build_market(args)
    baseline = BaselineQuoteSnapshot(
        snapshot_id="operator-recording",
        price_pips=args.baseline_price_pips,
        fetched_at=datetime.now(tz=UTC),
    )
    record_vote_cassettes(
        market=market,
        baseline=baseline,
        cassette_path=args.out,
        transports=_build_live_transports(),
    )
    print(f"recorded {len(DEFAULT_VOTE_ENSEMBLE)} votes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
