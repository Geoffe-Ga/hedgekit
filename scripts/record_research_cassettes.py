#!/usr/bin/env python3
"""Operator-run recorder for live web-research HTTP cassettes (issue #192).

CI never makes a live search or fetch call: the forecast pipeline drives the
:class:`~windbreak.forecast.providers.search_live.LiveSearchTransport` /
:class:`~windbreak.forecast.providers.fetch_live.LiveFetchTransport` pair through
a replay-only
:class:`~windbreak.forecast.providers.http_cassettes.ReplayHttpCassette`, which
fails closed on any unrecorded request. This developer-run script is the *only*
place a live research call is ever made, and the *only* place ``requests`` and
the process environment appear on this path -- deliberately kept out of the
``windbreak`` package so CI stays network-library-free and the SPEC S8.3 sandbox
boundary is never crossed.

Workflow:

1. An operator exports the search API key
   (``export RESEARCH_SEARCH_API_KEY=...``) and runs this script (via
   ``scripts/record-research-cassettes.sh``) with a pinned search endpoint, a
   query, and the research hosts fetches are permitted to reach.
2. One live search POST is made -- injecting the key as an ``Authorization``
   header *at send time*, never onto the header-free
   :class:`~windbreak.forecast.providers.http_cassettes.HttpRequest`, so the key
   can never be persisted -- and each returned candidate URL is fetched with a
   GET (no key). Every URL is screened against an
   :class:`~windbreak.net.allowlist.OutboundAllowlist` over the search host plus
   the configured research hosts before any byte leaves the process; redirects
   are refused (``allow_redirects=False``) so an on-path responder can never
   steer the recorder to another host; an integer timeout bounds each dial; and
   the fetched body is read under a raw max-bytes cap.
3. Both the search and its fetches are written to *one*
   :class:`~windbreak.forecast.providers.http_cassettes.RecordingHttpCassette`
   file, capturing each response's ``Content-Type`` into
   :class:`~windbreak.forecast.providers.http_cassettes.HttpResponse`.
4. The recorded cassette is inspected, committed, and thereafter replayed in CI,
   offline.

This module does no floating-point arithmetic: the timeout and size cap are
whole integers.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import requests

from windbreak.config.schema import ResearchSettings
from windbreak.forecast.providers.fetch_live import LiveFetchConfig, LiveFetchTransport
from windbreak.forecast.providers.http_cassettes import (
    HttpResponse,
    RecordingHttpCassette,
)
from windbreak.forecast.providers.search_live import (
    LiveSearchConfig,
    LiveSearchTransport,
)
from windbreak.net.allowlist import OutboundAllowlist

if TYPE_CHECKING:
    from windbreak.forecast.providers.http_cassettes import HttpRequest

#: The header the search API key is injected into at send time; never persisted.
_AUTH_HEADER = "Authorization"

#: The response header the media type is read from into ``HttpResponse``.
_CONTENT_TYPE_HEADER = "Content-Type"

#: The HTTP method a search request uses (the key-bearing side).
_SEARCH_METHOD = "POST"

#: Byte size of each streamed response chunk while enforcing the body cap.
_CHUNK_SIZE = 8192

#: Default requested search-result count when the CLI omits ``--max-results``.
_DEFAULT_MAX_RESULTS = 5


def _read_capped_body(response: requests.Response, max_bytes: int) -> str:
    """Read a response body under a raw byte cap, decoding to text.

    Args:
        response: The streamed live response.
        max_bytes: The maximum number of bytes to read before truncating.

    Returns:
        The (possibly truncated) body decoded as UTF-8, replacing any partial
        trailing multibyte sequence.
    """
    raw = bytearray()
    for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
        raw.extend(chunk)
        if len(raw) >= max_bytes:
            del raw[max_bytes:]
            break
    return raw.decode("utf-8", errors="replace")


class _RequestsResearchTransport:
    """A live ``requests``-backed transport: allowlisted, redirect-free, keyed.

    The search API key is held here and injected as an ``Authorization`` header
    on each search send, so it never touches the header-free
    :class:`~windbreak.forecast.providers.http_cassettes.HttpRequest` and can
    never be written into a recorded cassette. A GET fetch carries no key.
    :class:`requests.RequestException` is translated to :class:`OSError` so a
    dead link is skipped by the live-fetch transport exactly like in CI.
    """

    def __init__(
        self,
        *,
        api_key: str,
        allowlist: OutboundAllowlist,
        session: requests.Session,
        config: ResearchSettings,
    ) -> None:
        """Store the key, egress allowlist, session, and research configuration.

        Args:
            api_key: The search API key, injected as a bearer token on searches.
            allowlist: The egress allowlist screening every outbound URL.
            session: The live ``requests`` session to dial through.
            config: The research settings bounding the timeout and body size.
        """
        self._api_key = api_key
        self._allowlist = allowlist
        self._session = session
        self._config = config

    def send(self, request: HttpRequest) -> HttpResponse:
        """Screen the URL, dial once, and return the capped, typed response.

        Args:
            request: The request to send.

        Returns:
            The endpoint's response as an
            :class:`~windbreak.forecast.providers.http_cassettes.HttpResponse`,
            carrying its ``Content-Type``.

        Raises:
            EgressDeniedError: If the URL is off the allowlist.
            OSError: If the live request fails (a translated
                :class:`requests.RequestException`).
        """
        self._allowlist.require(request.url)
        headers: dict[str, str] = {}
        data: bytes | None = None
        if request.method == _SEARCH_METHOD:
            headers[_AUTH_HEADER] = f"Bearer {self._api_key}"
            data = request.body.encode("utf-8")
        try:
            response = self._session.request(
                request.method,
                request.url,
                data=data,
                headers=headers,
                timeout=self._config.fetch_timeout_seconds,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as exc:
            msg = f"live research request failed for {request.url!r}"
            raise OSError(msg) from exc
        body = _read_capped_body(response, self._config.fetch_max_bytes)
        content_type = response.headers.get(_CONTENT_TYPE_HEADER, "")
        return HttpResponse(
            status_code=response.status_code, body=body, content_type=content_type
        )


def _read_api_key(config: ResearchSettings) -> str:
    """Read the search API key from the environment, or exit with a message.

    Args:
        config: The research settings naming the key's environment variable.

    Returns:
        The API key value (never logged or printed).
    """
    try:
        return os.environ[config.search_api_key_env]
    except KeyError:
        sys.exit(f"error: environment variable {config.search_api_key_env} is not set")


def _research_allowlist(config: ResearchSettings) -> OutboundAllowlist:
    """Build an egress allowlist over the search host plus research hosts.

    Args:
        config: The research settings naming the endpoint and research hosts.

    Returns:
        An allowlist permitting only the search host and the configured
        research hosts.
    """
    hosts = set(config.allowed_research_hosts)
    endpoint_host = urlsplit(config.search_endpoint_url).hostname
    if endpoint_host:
        hosts.add(endpoint_host)
    return OutboundAllowlist(frozenset(hosts))


def _record_fetch(fetch: LiveFetchTransport, url: str) -> None:
    """Fetch one URL for recording, skipping (not aborting on) a dead link.

    Args:
        fetch: The live-fetch transport to record through.
        url: The candidate URL to fetch.
    """
    try:
        fetch.fetch(url)
    except OSError as exc:
        print(f"skipped {url}: {exc}")


def record_research_cassettes(
    config: ResearchSettings,
    *,
    query: str,
    max_results: int,
    cassette_path: Path,
    session: requests.Session | None = None,
) -> tuple[str, ...]:
    """Record one live search and each of its fetches into one cassette file.

    Args:
        config: The research settings (endpoint, key env, hosts, budgets).
        query: The search query to record.
        max_results: The requested search-result count.
        cassette_path: The cassette file every recorded pair is written into.
        session: An optional pre-built session (a fresh one is created when
            ``None``), so a caller can inject a configured transport.

    Returns:
        The candidate URLs the recorded search returned.
    """
    live = _RequestsResearchTransport(
        api_key=_read_api_key(config),
        allowlist=_research_allowlist(config),
        session=session if session is not None else requests.Session(),
        config=config,
    )
    recorder = RecordingHttpCassette(transport=live, path=cassette_path)
    search = LiveSearchTransport(
        recorder,
        LiveSearchConfig(
            endpoint_url=config.search_endpoint_url, max_results=max_results
        ),
    )
    fetch = LiveFetchTransport(
        recorder,
        LiveFetchConfig(
            max_body_bytes=config.fetch_max_bytes,
            allowed_content_types=config.allowed_content_types,
        ),
    )
    urls = search.search(query)
    for url in urls:
        _record_fetch(fetch, url)
    return urls


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the recorder's command-line arguments.

    Args:
        argv: The argument vector, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-url", required=True, help="Pinned search endpoint URL."
    )
    parser.add_argument("--query", required=True, help="Search query to record.")
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="A research host fetches may reach (repeatable).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=_DEFAULT_MAX_RESULTS,
        help="Requested search-result count.",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Cassette file to write."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the research-cassette recorder CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` once the search and its fetches are recorded.
    """
    args = _parse_args(argv)
    config = ResearchSettings(
        search_endpoint_url=args.endpoint_url,
        allowed_research_hosts=tuple(args.allowed_host),
    )
    urls = record_research_cassettes(
        config,
        query=args.query,
        max_results=args.max_results,
        cassette_path=args.out,
    )
    print(f"recorded search + {len(urls)} fetch(es) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
