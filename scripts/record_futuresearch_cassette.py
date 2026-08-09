#!/usr/bin/env python3
"""Operator-run recorder for FutureSearch provider HTTP cassettes (issue #189).

CI never makes a live FutureSearch call: the forecast pipeline drives
:class:`~windbreak.forecast.providers.futuresearch.FutureSearchProvider` through
a replay-only :class:`~windbreak.forecast.providers.http_cassettes.ReplayHttpCassette`,
which fails closed on any unrecorded request. This developer-run script is the
*only* place a live call is ever made, and the *only* place ``requests`` and the
process environment appear on this path -- deliberately kept out of the
``windbreak`` package so CI stays network-library-free and the SPEC S8.3 sandbox
boundary is never crossed.

Workflow:

1. An operator sets the API key in the environment
   (``export FUTURESEARCH_API_KEY=...``) and runs this script, pointing it at a
   pinned endpoint and a request body.
2. The live, allowlisted, redirect-free transport dials the endpoint once,
   injecting the key as an ``Authorization`` header *at send time* -- never onto
   the :class:`~windbreak.forecast.providers.http_cassettes.HttpRequest`, which
   has no header field, so the key can never be persisted into the cassette.
3. :class:`~windbreak.forecast.providers.http_cassettes.RecordingHttpCassette`
   writes the request/response pair to the cassette file.
4. The recorded cassette is committed and thereafter replayed in CI, offline.

The live transport is wrapped exactly like
``windbreak.connector.kalshi.client._RedirectFreeSession``: redirects are
refused (``allow_redirects=False``) so an on-path responder can never steer the
recorder to another host, an integer timeout bounds the dial, and every URL is
screened against an :class:`~windbreak.net.allowlist.OutboundAllowlist` built
over the pinned endpoint host before any byte leaves the process.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

import requests

from windbreak.forecast.providers.futuresearch import FutureSearchProviderConfig
from windbreak.forecast.providers.http_cassettes import (
    HttpRequest,
    HttpResponse,
    RecordingHttpCassette,
)
from windbreak.net.allowlist import OutboundAllowlist

#: Per-request timeout, in whole seconds (an int -- no float on any path here).
_TIMEOUT_SECONDS = 30

#: The header the API key is injected into at send time; never persisted.
_AUTH_HEADER = "Authorization"

#: The HTTP method every recorded FutureSearch request uses.
_REQUEST_METHOD = "POST"


class _RequestsHttpTransport:
    """A live ``requests``-backed transport: allowlisted, redirect-free, keyed.

    The API key is held here and injected as an ``Authorization`` header on each
    send, so it never touches the header-free
    :class:`~windbreak.forecast.providers.http_cassettes.HttpRequest` and can
    never be written into a recorded cassette.
    """

    def __init__(
        self,
        *,
        api_key: str,
        allowlist: OutboundAllowlist,
        session: requests.Session,
    ) -> None:
        """Store the key, the egress allowlist, and the live session.

        Args:
            api_key: The FutureSearch API key, injected as a bearer token.
            allowlist: The egress allowlist screening every outbound URL.
            session: The live ``requests`` session to dial through.
        """
        self._api_key = api_key
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
            headers={_AUTH_HEADER: f"Bearer {self._api_key}"},
            timeout=_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
        return HttpResponse(status_code=response.status_code, body=response.text)


def _read_api_key(config: FutureSearchProviderConfig) -> str:
    """Read the API key from the environment, or exit with a clear message.

    Args:
        config: The provider configuration naming the key's environment
            variable.

    Returns:
        The API key value.
    """
    try:
        return os.environ[config.api_key_env]
    except KeyError:
        sys.exit(f"error: environment variable {config.api_key_env} is not set")


def _endpoint_allowlist(config: FutureSearchProviderConfig) -> OutboundAllowlist:
    """Build an egress allowlist over the config's pinned endpoint host.

    Args:
        config: The provider configuration naming the endpoint URL.

    Returns:
        An allowlist permitting only the endpoint's host.
    """
    # The allowlist host is derived from the same endpoint URL being dialed, so
    # it cannot pre-vet an independently-trusted host -- it only constrains where
    # a *redirect* may land (redirects are refused outright) and screens control
    # characters. That is acceptable for this operator-run, never-CI-executed
    # recorder, where the operator supplies the endpoint deliberately.
    host = urlsplit(config.endpoint_url).hostname or ""
    return OutboundAllowlist(frozenset({host}))


def record_cassette(
    config: FutureSearchProviderConfig,
    *,
    request_body: str,
    cassette_path: Path,
    session: requests.Session | None = None,
) -> HttpResponse:
    """Record one live FutureSearch call into a cassette file.

    Args:
        config: The provider configuration (endpoint, api-key env).
        request_body: The raw request body to POST.
        cassette_path: The cassette file to write the pair into.
        session: An optional pre-built session (a fresh one is created when
            ``None``), so a caller can inject a configured transport.

    Returns:
        The recorded response.
    """
    live = _RequestsHttpTransport(
        api_key=_read_api_key(config),
        allowlist=_endpoint_allowlist(config),
        session=session if session is not None else requests.Session(),
    )
    recorder = RecordingHttpCassette(transport=live, path=cassette_path)
    request = HttpRequest(
        method=_REQUEST_METHOD, url=config.endpoint_url, body=request_body
    )
    return recorder.send(request)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the recorder's command-line arguments.

    Args:
        argv: The argument vector, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The parsed arguments (``endpoint_url``, ``body``, ``out``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-url", required=True, help="Pinned endpoint URL.")
    parser.add_argument("--body", required=True, help="Raw request body to POST.")
    parser.add_argument(
        "--out", required=True, type=Path, help="Cassette file to write."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the recorder CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on a recorded call.
    """
    args = _parse_args(argv)
    config = FutureSearchProviderConfig(
        endpoint_url=args.endpoint_url, pinned_forecaster_versions=()
    )
    response = record_cassette(config, request_body=args.body, cassette_path=args.out)
    print(f"recorded {response.status_code} response to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
