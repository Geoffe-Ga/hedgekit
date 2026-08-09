#!/usr/bin/env python3
"""Operator-run recorder for the triage Stage-0 LLM cassette (issue #192).

CI never makes a live Stage-0 triage call: the triage path drives
:func:`~windbreak.forecast.triage.run_stage0_prior` through a replay-only
:class:`~windbreak.forecast.cassettes.ReplayCassette`, which fails closed on any
unrecorded request. This developer-run script is the *only* place a live Stage-0
call is ever made, and the *only* place ``requests`` and the process environment
appear on this path -- deliberately kept out of the ``windbreak`` package so CI
stays network-library-free and the SPEC S8.3 sandbox boundary is never crossed.

The pinned Stage-0 model (:data:`~windbreak.forecast.triage._DEFAULT_TRIAGE_MODEL`,
an OpenAI model) is driven through its live OpenAI Chat adapter, wrapped in a
:class:`~windbreak.forecast.cassettes.RecordingCassette`, so one live prior is
recorded end-to-end. The key is injected as a *send-time* header, never onto the
header-free
:class:`~windbreak.forecast.providers.http_cassettes.HttpRequest`, so it can
never be persisted. This module does no floating-point arithmetic.
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
    OPENAI_CHAT_ENDPOINT,
    HttpResponse,
    OpenAiChatTransport,
)
from windbreak.forecast.records import BaselineQuoteSnapshot
from windbreak.forecast.triage import run_stage0_prior
from windbreak.net.allowlist import OutboundAllowlist

if TYPE_CHECKING:
    from windbreak.forecast.providers.http_cassettes import HttpRequest

#: Per-request timeout, in whole seconds (an int -- no float on any path here).
_TIMEOUT_SECONDS = 30

#: The pinned Stage-0 provider host (the only host egress is permitted to).
_OPENAI_HOST = "api.openai.com"

#: Environment variable the live key is read from (a name, never the secret).
_OPENAI_KEY_ENV = "OPENAI_API_KEY"

#: Shared request headers.
_CONTENT_TYPE_HEADER = "content-type"
_JSON_CONTENT_TYPE = "application/json"
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


def _read_key(env_var: str) -> str:
    """Read a provider API key from the environment, or exit with a message.

    Args:
        env_var: The environment variable the key is read from.

    Returns:
        The API key value (never logged or printed).
    """
    try:
        return os.environ[env_var]
    except KeyError:
        sys.exit(f"error: environment variable {env_var} is not set")


def _build_openai_adapter() -> OpenAiChatTransport:
    """Build the live OpenAI Chat adapter the pinned Stage-0 model runs on.

    Returns:
        The live :class:`~windbreak.forecast.providers.openai.OpenAiChatTransport`.
    """
    live = _LiveHttpTransport(
        headers={
            _OPENAI_AUTH_HEADER: f"Bearer {_read_key(_OPENAI_KEY_ENV)}",
            _CONTENT_TYPE_HEADER: _JSON_CONTENT_TYPE,
        },
        allowlist=OutboundAllowlist(frozenset({_OPENAI_HOST})),
        session=requests.Session(),
    )
    return OpenAiChatTransport(live, endpoint_url=OPENAI_CHAT_ENDPOINT)


def _build_market(args: argparse.Namespace) -> NormalizedMarket:
    """Build the market whose question fields the Stage-0 prompt is built from.

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
    """Run the triage Stage-0 cassette recorder CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` once the Stage-0 prior is recorded.
    """
    args = _parse_args(argv)
    market = _build_market(args)
    baseline = BaselineQuoteSnapshot(
        snapshot_id="operator-recording",
        price_pips=args.baseline_price_pips,
        fetched_at=datetime.now(tz=UTC),
    )
    recorder = RecordingCassette(transport=_build_openai_adapter(), path=args.out)
    prior = run_stage0_prior(market, baseline, transport=recorder)
    print(f"recorded stage-0 prior {prior.prior_ppm} ppm to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
