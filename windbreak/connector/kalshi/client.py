"""A thin, read-only HTTPS JSON GET client for Kalshi's public v2 REST API.

:class:`KalshiClient` builds a request URL by percent-encoding each path
segment individually (:func:`urllib.parse.quote` with ``safe=""``) and joining
them onto an ``https://`` base, then performs a single GET through an injected
``requests``-like session seam -- so tests supply a fake session with no real
network (SPEC S17.1: the full pipeline runs offline and deterministically in
CI). It models only *public, read-only market access* (SPEC S5.2): it never
attaches auth headers or credentials. Non-2xx responses raise
:class:`KalshiApiError`; a 2xx response yields a :class:`KalshiResponse`
carrying the parsed JSON payload and the parsed ``Date`` header.

Transport and parsing run through an on-by-default
:class:`~windbreak.connector.resilience.ResilientCaller` (issue #20): every
client -- unless it opts out with ``resilience=None`` or supplies its own --
gets live token-bucket rate limiting, exponential backoff with jitter on
retryable ``5xx``/``429``/transport failures, and a circuit breaker that halts
after repeated failures. Schema validation runs *outside* that retry loop, so a
:class:`~windbreak.connector.validation.SchemaAnomalyHaltError` is never retried
and never counted against the breaker.

This module sits on the money path guarded by ``scripts/lint_no_floats.py``:
the request timeout is an ``int`` and no ``/`` or ``float`` appears anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Final, Protocol, cast
from urllib.parse import quote, urlsplit

import requests

from windbreak.connector.resilience import (
    DEFAULT_RESILIENCE_POLICY,
    build_default_resilient_caller,
)
from windbreak.connector.snapshot import LoggingEventLedgerWriter
from windbreak.connector.validation import (
    SchemaValidator,
    kalshi_default_schema_registry,
)
from windbreak.net.allowlist import EgressDeniedError, OutboundAllowlist

if TYPE_CHECKING:
    from collections.abc import Mapping

    from windbreak.connector.resilience import ResiliencePolicy, ResilientCaller

#: The current-generation Kalshi public API base (SPEC S7.1).
KALSHI_API_BASE: Final = "https://api.elections.kalshi.com/trade-api/v2"

#: The Kalshi *demo* API base, the other half of SPEC S16's
#: ``exchange.environment: demo | production`` pair. Its host mirrors
#: :mod:`windbreak.net.allowlist`'s ``_KALSHI_DEMO_HOST``, which is what a
#: ``demo`` deployment's derived egress allowlist admits. Reaching it still
#: demands an explicit ``allowlist`` argument: the implicit
#: :data:`_CANONICAL_ALLOWLIST_HOSTS` covers only the production host, so a demo
#: base can never be dialed by a client whose deployment did not declare it.
KALSHI_DEMO_API_BASE: Final = "https://demo-api.kalshi.co/trade-api/v2"

#: The single host implicitly permitted when no explicit ``allowlist`` is
#: supplied: exactly the canonical :data:`KALSHI_API_BASE` host, so the stock
#: constructor works while any other base URL demands an explicit allowlist.
_CANONICAL_ALLOWLIST_HOSTS: Final = frozenset(
    {urlsplit(KALSHI_API_BASE).hostname or ""}
)

#: Default per-request timeout, in whole seconds (int: this is the money path).
DEFAULT_TIMEOUT_SECONDS: Final = 10

#: The only URL scheme this client will dial; enforced at construction.
_HTTPS_PREFIX: Final = "https://"

#: Inclusive lower/upper bounds of the HTTP 2xx success range.
_MIN_OK_STATUS: Final = 200
_MAX_OK_STATUS: Final = 299


class _Unset(Enum):
    """A distinct sentinel type so ``resilience`` can tell "default" from None.

    ``resilience`` accepts a caller, an explicit ``None`` (disable resilience --
    raw single-attempt transport), or -- left unset -- this sentinel, which
    triggers building the on-by-default :class:`ResilientCaller`.
    """

    DEFAULT = auto()


#: The unset-``resilience`` sentinel: build the default resilient caller.
_DEFAULT_RESILIENCE: Final = _Unset.DEFAULT


class KalshiApiError(RuntimeError):
    """Raised when Kalshi returns a non-2xx HTTP status."""

    def __init__(self, status_code: int) -> None:
        """Initialize with the failing status code, named in the message.

        Args:
            status_code: The non-2xx HTTP status code Kalshi returned.
        """
        self.status_code = status_code
        super().__init__(f"Kalshi API request failed with HTTP status {status_code}")


@dataclass(frozen=True, slots=True)
class KalshiResponse:
    """A parsed successful Kalshi response.

    Attributes:
        payload: The parsed JSON body.
        server_date: The ``Date`` header parsed to a UTC datetime, or None when
            the header is absent or unparseable.
    """

    payload: object
    server_date: datetime | None


class _Response(Protocol):
    """The minimal response surface :class:`KalshiClient` reads.

    Attributes:
        status_code: The HTTP status code.
        headers: The response headers.
    """

    status_code: int
    headers: Mapping[str, str]

    def json(self) -> object:
        """Return the parsed JSON body."""
        ...


class Session(Protocol):
    """The minimal ``requests``-like session seam :class:`KalshiClient` calls."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None,
        timeout: int,
    ) -> _Response:
        """Perform a GET request.

        Args:
            url: The fully built request URL.
            params: Query parameters, or None.
            timeout: The request timeout in whole seconds.

        Returns:
            The HTTP response.
        """
        ...


def _parse_server_date(headers: Mapping[str, str]) -> datetime | None:
    """Parse a response ``Date`` header into a UTC datetime.

    Args:
        headers: The response headers.

    Returns:
        The parsed, UTC-normalized datetime, or None when the header is absent
        or cannot be parsed.
    """
    raw = headers.get("Date")
    if raw is None:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    # Naive means `tzinfo is None` *or* a `tzinfo` whose `utcoffset()` returns
    # None -- Python's own definition; `utcoffset()` reports both in one call.
    # Testing `tzinfo is None` alone (issue #346) sent the second shape to
    # `astimezone`, which does not raise on a naive value but silently reads
    # its wall clock as the *host's* local time -- so the exchange instant
    # would skew by whatever offset the machine runs at, differing between a
    # developer's laptop and CI. Both naive shapes belong on the same branch:
    # an HTTP-date is GMT (RFC 9110 S5.6.7), which is a fact about the
    # protocol, not about the machine that parsed it.
    if parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class _RedirectFreeSession:
    """Adapt a real ``requests`` session, refusing to follow redirects.

    This client dials one fixed, public JSON endpoint, so it must never chase a
    ``3xx`` ``Location`` to an arbitrary host: following a redirect would let an
    on-path or compromised responder steer a (credential-free) GET to an
    attacker's server and have its body parsed as Kalshi data. Pinning
    ``allow_redirects=False`` turns any redirect into a non-2xx that
    :class:`KalshiClient` surfaces as :class:`KalshiApiError` -- failing closed
    rather than silently switching hosts. Only the real transport is wrapped;
    injected test seams keep their own two-argument ``get``.
    """

    def __init__(self, session: requests.Session) -> None:
        """Store the real session this wrapper forwards to.

        Args:
            session: The live :class:`requests.Session` to dial through.
        """
        self._session = session

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None,
        timeout: int,
    ) -> _Response:
        """Forward a GET, forbidding redirect-following.

        Args:
            url: The fully built request URL.
            params: Query parameters, or None.
            timeout: The request timeout in whole seconds.

        Returns:
            The HTTP response.
        """
        return cast(
            "_Response",
            self._session.get(
                url, params=params, timeout=timeout, allow_redirects=False
            ),
        )


def _default_wall_clock() -> datetime:
    """Return the current UTC time, the default validator's event-stamp clock."""
    return datetime.now(UTC)


class KalshiClient:
    """A thin HTTPS JSON GET client over Kalshi's public v2 REST API."""

    def __init__(
        self,
        base_url: str = KALSHI_API_BASE,
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        session: Session | None = None,
        resilience: ResilientCaller | None | _Unset = _DEFAULT_RESILIENCE,
        resilience_policy: ResiliencePolicy | None = None,
        validator: SchemaValidator | None = None,
        allowlist: OutboundAllowlist | None = None,
    ) -> None:
        """Initialize the client, validating the base URL scheme and host.

        Args:
            base_url: The API base URL; must begin with ``https://`` and its host
                must be on ``allowlist`` (or, when ``allowlist`` is ``None``, be
                the canonical :data:`KALSHI_API_BASE` host).
            timeout: Per-request timeout, in whole seconds.
            session: An injected ``requests``-like session; a real
                :class:`requests.Session` is created lazily when None.
            resilience: The caller wrapping transport+parse in rate limiting,
                retries, and a circuit breaker. Left unset, an on-by-default
                caller is built via
                :func:`~windbreak.connector.resilience.build_default_resilient_caller`
                so every client gets live protection; pass an explicit
                :class:`~windbreak.connector.resilience.ResilientCaller` to
                supply your own, or an explicit ``None`` to disable resilience
                (raw single-attempt transport -- e.g. when composing resilience
                upstream, or in thin-transport tests).
            resilience_policy: Tunables for the on-by-default caller; ignored
                when an explicit ``resilience`` (or ``None``) is passed. Defaults
                to :data:`~windbreak.connector.resilience.DEFAULT_RESILIENCE_POLICY`.
            validator: The schema validator run on every parsed payload;
                defaults to an on-by-default validator built from
                :func:`kalshi_default_schema_registry`, so schema drift fails
                closed for every endpoint unless a caller opts into another.
            allowlist: The outbound-network allowlist ``base_url``'s host must be
                on. ``None`` (the default) builds an allowlist of exactly the
                canonical :data:`KALSHI_API_BASE` host, so the stock constructor
                works but any other base URL must supply an explicit allowlist.

        Raises:
            ValueError: If ``base_url`` does not begin with ``https://``, or its
                host is not permitted by ``allowlist`` -- rejected at
                construction, before any session or network call.
        """
        if not base_url.startswith(_HTTPS_PREFIX):
            raise ValueError(
                f"base_url must begin with {_HTTPS_PREFIX!r}, got {base_url!r}"
            )
        self._require_allowed_host(base_url, allowlist)
        self._base_url = base_url
        self._timeout = timeout
        self._session: Session = (
            session if session is not None else _RedirectFreeSession(requests.Session())
        )
        if isinstance(resilience, _Unset):
            resilience = build_default_resilient_caller(
                policy=resilience_policy or DEFAULT_RESILIENCE_POLICY
            )
        self._resilience: ResilientCaller | None = resilience
        if validator is not None:
            self._validator = validator
        else:
            self._validator = SchemaValidator(
                kalshi_default_schema_registry(),
                LoggingEventLedgerWriter(),
                wall_clock=_default_wall_clock,
            )

    @staticmethod
    def _require_allowed_host(
        base_url: str, allowlist: OutboundAllowlist | None
    ) -> None:
        """Reject a base URL whose host is off the outbound allowlist.

        Runs the structural egress check at construction, translating the
        allowlist's :class:`~windbreak.net.allowlist.EgressDeniedError` into the
        ``ValueError`` construction contract this client already raises for a
        bad scheme -- so a disallowed host fails closed *before* any session or
        network call.

        Args:
            base_url: The API base URL whose host is checked.
            allowlist: The explicit allowlist, or ``None`` to permit exactly the
                canonical :data:`KALSHI_API_BASE` host.

        Raises:
            ValueError: If ``base_url``'s host is not permitted.
        """
        effective = (
            allowlist
            if allowlist is not None
            else OutboundAllowlist(_CANONICAL_ALLOWLIST_HOSTS)
        )
        try:
            effective.require(base_url)
        except EgressDeniedError as exc:
            raise ValueError(
                f"base_url host is not on the outbound allowlist: {base_url!r}"
            ) from exc

    def _build_url(self, segments: tuple[str, ...]) -> str:
        """Join percent-encoded path segments onto the base URL.

        Args:
            segments: The path segments, each encoded individually so a ``/``
                or space inside a segment is escaped rather than splitting the
                path.

        Returns:
            The fully built request URL.
        """
        encoded = "/".join(quote(segment, safe="") for segment in segments)
        return f"{self._base_url}/{encoded}"

    def get(
        self, *segments: str, params: Mapping[str, str] | None = None
    ) -> KalshiResponse:
        """Perform a GET over the joined path segments and parse the response.

        Transport and parsing run through the injected
        :class:`~windbreak.connector.resilience.ResilientCaller` when one is
        wired (rate limiting, retries, breaker), or directly otherwise. The
        parsed payload is then always run through the schema validator, so a
        :class:`~windbreak.connector.validation.SchemaAnomalyHaltError` is raised
        *outside* any retry loop -- schema drift is never retried and never
        counted against the circuit breaker.

        Args:
            *segments: Path segments appended to the base URL, each
                percent-encoded individually.
            params: Optional query parameters, forwarded unchanged.

        Returns:
            The parsed successful response.

        Raises:
            KalshiApiError: If the response status is outside the 2xx range.
            SchemaAnomalyHaltError: If the parsed payload fails schema validation.
        """

        def _fetch() -> KalshiResponse:
            """Run one transport+parse attempt (retried by the resilient caller)."""
            return self._transport(segments, params)

        response = (
            self._resilience.call(_fetch) if self._resilience is not None else _fetch()
        )
        self._validator.validate(
            segments, cast("Mapping[str, object]", response.payload)
        )
        return response

    def _transport(
        self, segments: tuple[str, ...], params: Mapping[str, str] | None
    ) -> KalshiResponse:
        """Perform one GET and parse a 2xx response, raising on any non-2xx.

        Args:
            segments: The path segments to build the request URL from.
            params: Optional query parameters, forwarded unchanged.

        Returns:
            The parsed successful response.

        Raises:
            KalshiApiError: If the response status is outside the 2xx range.
        """
        response = self._session.get(
            self._build_url(segments), params=params, timeout=self._timeout
        )
        if not _MIN_OK_STATUS <= response.status_code <= _MAX_OK_STATUS:
            raise KalshiApiError(response.status_code)
        return KalshiResponse(
            payload=response.json(), server_date=_parse_server_date(response.headers)
        )
