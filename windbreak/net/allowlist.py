"""Structural outbound-network allowlist for the live deployment (issue #57).

SPEC S5.2/S7.1 require a live (LIVE_MICRO/LIVE) deployment to be able to reach
only an explicit, small set of hosts -- the exchange, the forecast providers,
and the configured alert sink -- and *nothing else*. This module makes that
boundary **structural rather than advisory**, mirroring
:meth:`windbreak.forecast.sandbox.ResearchTools.fetch`: every outbound URL is
screened for parse-differential SSRF bytes and matched by exact, lowercased
hostname against a fixed allowlist before a connector may dial it.

:class:`OutboundAllowlist` fails closed: :meth:`~OutboundAllowlist.require`
*always* raises :class:`EgressDeniedError` on a denial, and only *additionally*
records an ``EgressDenied`` ledger event when a recorder is wired -- telemetry
never gates the refusal. :func:`allowlist_from_config` derives the host set from
a :class:`~windbreak.config.schema.WindbreakConfig`: the exchange host (by
provider/environment) and each forecast provider host. An unrecognized provider
contributes no host, so an unknown exchange or model can never silently inherit
network access.

Alert-sink hosts are deliberately *not* derived here: the SPEC S16
:class:`~windbreak.config.schema.AlertSink` schema carries only ``type``/
``topic`` (no ``base_url``), and SPEC S16 is canonical with unknown keys fatal,
so there is no config field to derive one from. An alert sink's host is instead
supplied at :class:`~windbreak.alerts.sinks.NtfySink` construction time via its
own explicit allowlist.

This module sits on the network boundary but off the money path; it is
float-free by construction (it does no arithmetic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, Protocol
from urllib.parse import urlsplit

from windbreak.ledger.events import Event

if TYPE_CHECKING:
    from windbreak.config.schema import (
        ExchangeConfig,
        ForecastConfig,
        ResearchSettings,
        WindbreakConfig,
    )

#: The only URL schemes egress is ever permitted for: plain http(s), never
#: ``file://``, ``ftp://``, or any other privileged scheme.
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

#: Ordinal of the first printable ASCII character (space, ``0x20``); anything
#: below it is a C0 control character a well-formed URL would percent-encode.
_MIN_PRINTABLE_ORD = 0x20

#: Ordinal of ASCII DEL (``0x7f``), the one control codepoint above the
#: printable range, so it is checked by hand.
_DEL_ORD = 0x7F

#: Component label stamped on every event this module records.
_COMPONENT = "net"

#: Payload schema version stamped on every event this module records.
_PAYLOAD_SCHEMA_VERSION = 1

#: Event-type discriminator for the one event this module records.
_EGRESS_DENIED_EVENT = "EgressDenied"

#: The current-generation Kalshi public API host -- the hostname of
#: ``windbreak.connector.kalshi.client.KALSHI_API_BASE``. Duplicated here as a
#: literal (not imported) to avoid a ``net`` <-> ``connector`` import cycle, since
#: the connector imports :class:`OutboundAllowlist` from this module.
_KALSHI_PRODUCTION_HOST = "api.elections.kalshi.com"

#: The Kalshi demo API host, added to a ``demo``-environment allowlist alongside
#: the production host so a demo deployment can reach either.
_KALSHI_DEMO_HOST = "demo-api.kalshi.co"

#: The recognized exchange provider name.
_KALSHI_PROVIDER = "kalshi"

#: The exchange environment token that also admits the demo host.
_DEMO_ENVIRONMENT = "demo"

#: Maps a recognized forecast provider name to its API host. A provider absent
#: from this table contributes no host (fail closed on an unknown provider).
_FORECAST_PROVIDER_HOSTS: dict[str, str] = {
    "anthropic": "api.anthropic.com",
    "openai": "api.openai.com",
}


class EgressDeniedError(Exception):
    """Raised when an outbound URL targets a scheme or host off the allowlist.

    Deliberately distinct from
    :class:`windbreak.forecast.sandbox.EgressDeniedError`: the research-tool
    sandbox and this outbound-connector allowlist are separate bounded contexts.
    """


class EventRecorder(Protocol):
    """The narrow seam a denial records its ``EgressDenied`` event through.

    Structurally satisfied by any ledger writer with a matching
    :meth:`record` -- the same duck type
    :class:`~windbreak.riskkernel.reservations.ReservationLedger` and
    :class:`~windbreak.riskkernel.human_ack.HumanAckQueue` already accept -- so
    no ``riskkernel`` import is needed here.
    """

    def record(self, event: Event) -> None:
        """Record one event.

        Args:
            event: The event to record.
        """
        ...


def _has_unsafe_url_chars(url: str) -> bool:
    """Return whether ``url`` holds any control or whitespace character.

    :func:`urllib.parse.urlsplit` silently strips tab/newline/carriage-return
    (and tolerates leading control/space bytes, CVE-2023-24329) while computing
    the ``hostname`` this allowlist trusts, but the raw ``url`` is later handed
    verbatim to a transport. A byte one parser drops and another keeps is a
    parse-differential SSRF escape, so any such byte must fail closed *before*
    parsing -- mirroring :func:`windbreak.forecast.sandbox._has_unsafe_url_chars`.

    Args:
        url: The candidate URL to screen.

    Returns:
        ``True`` if any character is ASCII whitespace, a C0 control byte, or DEL.
    """
    return any(
        char.isspace() or ord(char) < _MIN_PRINTABLE_ORD or ord(char) == _DEL_ORD
        for char in url
    )


class OutboundAllowlist:
    """A fail-closed, exact-hostname egress allowlist for outbound connectors.

    The allowlisted hosts are normalized to lowercase at construction, so
    matching is case-insensitive. A denial always raises
    :class:`EgressDeniedError`; when a ``recorder`` is wired the denial *also*
    records one ``EgressDenied`` event, but the recorder can never suppress the
    raise (fail-closed first, telemetry second).
    """

    __slots__ = ("_hosts", "_recorder")

    def __init__(
        self, hosts: frozenset[str], *, recorder: EventRecorder | None = None
    ) -> None:
        """Initialize the allowlist over its permitted hosts.

        Args:
            hosts: The permitted hostnames; normalized to lowercase so matching
                is case-insensitive.
            recorder: The optional seam a denial records its ``EgressDenied``
                event through. ``None`` records nothing (but still raises).
        """
        self._hosts = frozenset(host.lower() for host in hosts)
        self._recorder = recorder

    def require(self, url: str) -> None:
        """Permit ``url`` if allowlisted, else record (if wired) and raise.

        The URL is first screened for control/whitespace bytes (run *before* any
        parsing), then parsed with :func:`urllib.parse.urlsplit`; a non-http(s)
        scheme, a missing host, or an off-allowlist host is denied. The match is
        host-only: any port on an allowlisted host is permitted.

        Args:
            url: The outbound URL to check.

        Raises:
            EgressDeniedError: If the URL contains a control or whitespace
                character, the scheme is not http(s), the host is missing, or
                the host is not on the allowlist.
        """
        if _has_unsafe_url_chars(url):
            self._deny(url, None)
        parts = urlsplit(url)
        if parts.scheme.lower() not in _ALLOWED_SCHEMES:
            self._deny(url, parts.hostname)
        hostname = parts.hostname
        if not hostname:
            self._deny(url, None)
        if hostname.lower() not in self._hosts:
            self._deny(url, hostname)

    def _deny(self, url: str, host: str | None) -> NoReturn:
        """Record an ``EgressDenied`` event (if wired), then always raise.

        Args:
            url: The denied URL, named in the raised error for diagnostics.
            host: The parsed host, or ``None`` when none could be parsed; a
                ``None`` is recorded and named as the empty string.

        Raises:
            EgressDeniedError: Always -- this method never returns.
        """
        parsed_host = host or ""
        if self._recorder is not None:
            self._recorder.record(
                Event(
                    event_type=_EGRESS_DENIED_EVENT,
                    component=_COMPONENT,
                    payload_schema_version=_PAYLOAD_SCHEMA_VERSION,
                    payload={"host": parsed_host},
                )
            )
        raise EgressDeniedError(
            f"egress denied: host {parsed_host!r} is not allowlisted (url {url!r})"
        )


def _exchange_hosts(exchange: ExchangeConfig) -> frozenset[str]:
    """Derive the exchange host set from the exchange configuration.

    Args:
        exchange: The exchange configuration section.

    Returns:
        The production Kalshi host for a ``kalshi`` provider (plus the demo host
        in a ``demo`` environment), or an empty set for any unrecognized
        provider (fail closed on an unknown exchange).
    """
    if exchange.provider != _KALSHI_PROVIDER:
        return frozenset()
    hosts = {_KALSHI_PRODUCTION_HOST}
    if exchange.environment == _DEMO_ENVIRONMENT:
        hosts.add(_KALSHI_DEMO_HOST)
    return frozenset(hosts)


def _forecast_hosts(forecast: ForecastConfig) -> frozenset[str]:
    """Derive the forecast-provider host set from the forecast configuration.

    Args:
        forecast: The forecast configuration section.

    Returns:
        One host per recognized provider across three sources -- the legacy
        ``ensemble`` triage/promotion ``ModelRef`` set, the ``triage_model``,
        and each ``vote_ensemble`` member (issue #240) -- unioned; an
        unrecognized provider name contributes no host (fail closed).
    """
    providers = (
        *(model.provider for model in (*forecast.ensemble, forecast.triage_model)),
        *(member.provider for member in forecast.vote_ensemble),
    )
    return frozenset(
        _FORECAST_PROVIDER_HOSTS[provider]
        for provider in providers
        if provider in _FORECAST_PROVIDER_HOSTS
    )


def _research_hosts(research: ResearchSettings) -> frozenset[str]:
    """Derive the live-research host set from the research configuration.

    Args:
        research: The forecast configuration's research section.

    Returns:
        The parsed host of ``search_endpoint_url`` (only when it parses to a
        real host -- the ``configured-by-operator`` placeholder yields none, so
        an unconfigured deployment fails closed) plus each
        ``allowed_research_hosts`` entry, all lowercased.
    """
    hosts: set[str] = set()
    endpoint_host = urlsplit(research.search_endpoint_url).hostname
    if endpoint_host:
        hosts.add(endpoint_host.lower())
    hosts.update(host.lower() for host in research.allowed_research_hosts)
    return frozenset(hosts)


def allowlist_from_config(
    config: WindbreakConfig, *, recorder: EventRecorder | None = None
) -> OutboundAllowlist:
    """Build an :class:`OutboundAllowlist` from a windbreak configuration.

    The host set is the union of the exchange host (:func:`_exchange_hosts`), the
    forecast-provider hosts (:func:`_forecast_hosts`, spanning the legacy
    ``ensemble``, the ``triage_model``, and each ``vote_ensemble`` member), and
    the live-research hosts (:func:`_research_hosts`). Alert-sink hosts are not
    derived here (see the module docstring). An unrecognized exchange or model
    provider, and an unconfigured research section, each contribute no host, so
    the resulting allowlist fails closed.

    Args:
        config: The windbreak configuration to derive hosts from.
        recorder: The optional recorder forwarded to the built allowlist, so
            every later denial records through it.

    Returns:
        An allowlist over the derived host set, wired to ``recorder``.
    """
    hosts = (
        _exchange_hosts(config.exchange)
        | _forecast_hosts(config.forecast)
        | _research_hosts(config.forecast.research)
    )
    return OutboundAllowlist(hosts, recorder=recorder)
