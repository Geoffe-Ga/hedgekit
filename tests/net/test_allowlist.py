"""Failing-first tests for a structural outbound-network allowlist (issue #57,
RED).

Issue #57's plan calls for the LIVE_MICRO deployment to be able to reach only
an explicit, small set of hosts -- the exchange, the two forecast providers,
and the configured alert sink -- and *nothing else*, mirroring
`windbreak.forecast.sandbox.ResearchTools.fetch`'s structural (not
prompt-based) egress gate: parse-differential SSRF screening, exact
lowercased hostname matching, and fail-closed on any parse ambiguity.

`windbreak/net/allowlist.py` has since shipped, so these tests are GREEN; the
"proposed shape" notes below are retained as the record of what was agreed.
Issue #274 added `OutboundAllowlist.require_host` (the non-URL screen an SMTP
relay host goes through) and the `alerts.allowed_hosts` derivation in
`allowlist_from_config`; both are covered near the bottom of this file.

Proposed public shape (the implementation specialist must build to this
exactly, or confirm/rename via the handoff):

* ``EgressDeniedError(Exception)`` -- raised by ``OutboundAllowlist.require``
  on any denial. A new class local to ``windbreak.net.allowlist``, distinct
  from (but semantically identical to)
  ``windbreak.forecast.sandbox.EgressDeniedError``, since the sandbox's
  research-tool boundary and this outbound-connector boundary are separate
  bounded contexts.
* ``OutboundAllowlist(hosts: frozenset[str], *, recorder: EgressRecorder |
  None = None)`` -- ``recorder`` is any object exposing
  ``.record(event: windbreak.ledger.events.Event) -> None`` (the same duck
  type ``ReservationLedger``/``HumanAckQueue``/``KillSwitch`` all take),
  structurally satisfied by the local ``_RecordingRecorder`` fake below.

  * ``.require(url: str) -> None`` -- raises ``EgressDeniedError`` for a
    non-http(s) scheme, a missing host, a control/whitespace character
    anywhere in the URL (mirroring
    ``windbreak.forecast.sandbox._has_unsafe_url_chars``'s parse-differential
    SSRF screen -- run *before* any URL parsing), or a host not on the
    (case-insensitively matched) allowlist. Recording an ``"EgressDenied"``
    event through ``recorder`` (when wired) happens *in addition to* raising,
    never instead of it: a missing recorder must never change whether the
    call raises.

* ``allowlist_from_config(config: windbreak.config.schema.WindbreakConfig, *,
  recorder: EgressRecorder | None = None) -> OutboundAllowlist`` -- derives
  hosts from:

  - ``config.exchange.provider`` -- ``"kalshi"`` contributes
    ``"api.elections.kalshi.com"``; any other (unrecognized) provider name
    contributes no host at all (fail closed on an unknown exchange).
  - every ``ModelRef.provider`` across ``config.forecast.ensemble`` and
    ``config.forecast.triage_model`` -- ``"anthropic"`` contributes
    ``"api.anthropic.com"``, ``"openai"`` contributes ``"api.openai.com"``;
    any other provider name (e.g. the default triage model's
    ``"cheapest-adequate"``) contributes no host.
  - ``config.alerts.allowed_hosts`` -- the operator's explicit declaration of
    which alert-destination hosts this deployment may dial (issue #274, which
    resolved the open question originally flagged here). Deliberately NOT the
    per-sink ``base_url``/``url``/``smtp.host`` destinations: an allowlist
    derived from the URLs it screens could never veto one of them. Empty by
    default, so an undeclared deployment reaches no alert host.

Issue #192 additionally derives hosts from ``config.forecast.research``
(``windbreak.config.schema.ResearchSettings``, itself new in #192): the parsed
host of ``research.search_endpoint_url`` and every entry of
``research.allowed_research_hosts``, both additive with the exchange- and
forecast-provider-host derivation above. The default, unconfigured
``ResearchSettings()`` (a placeholder endpoint URL, an empty
``allowed_research_hosts`` tuple) contributes zero hosts, mirroring every
other "operator must fill this in" default's fail-closed behavior elsewhere
in this module.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from windbreak.config.schema import (
    AlertsConfig,
    AlertSink,
    EnsembleMemberConfig,
    ForecastConfig,
    ModelRef,
    ResearchSettings,
    WindbreakConfig,
)
from windbreak.net.allowlist import (
    EgressDeniedError,
    OutboundAllowlist,
    _exchange_hosts,
    allowlist_from_config,
)

if TYPE_CHECKING:
    from windbreak.config.schema import ExchangeConfig
    from windbreak.ledger.events import Event

#: The current-generation Kalshi public API host (SPEC S7.1), matching
#: ``windbreak.connector.kalshi.client.KALSHI_API_BASE``'s hostname.
_KALSHI_HOST = "api.elections.kalshi.com"

#: The Kalshi *demo* API host, matching
#: ``windbreak.connector.kalshi.client.KALSHI_DEMO_API_BASE``'s hostname. Written
#: as a literal for the same reason ``_KALSHI_HOST`` is: a test that imported the
#: module's own constant could not notice that constant changing.
_KALSHI_DEMO_HOST = "demo-api.kalshi.co"
_ANTHROPIC_HOST = "api.anthropic.com"
_OPENAI_HOST = "api.openai.com"


class _RecordingRecorder:
    """A minimal ``EgressRecorder`` fake: records every ``Event`` it sees."""

    def __init__(self) -> None:
        """Initialize with an empty recorded-events log."""
        self.events: list[Event] = []

    def record(self, event: Event) -> None:
        """Append ``event`` to the recorded-events log.

        Args:
            event: The event to record.
        """
        self.events.append(event)


# --- OutboundAllowlist.require: allow / deny -----------------------------------


def test_require_allows_an_exact_case_insensitive_allowlisted_host() -> None:
    """A URL whose host matches the allowlist -- any letter case -- passes."""
    allowlist = OutboundAllowlist(frozenset({_KALSHI_HOST}))

    allowlist.require(f"https://{_KALSHI_HOST.upper()}/trade-api/v2/markets")


def test_require_denies_an_off_list_host() -> None:
    """A syntactically valid https URL to a host not on the list is denied."""
    allowlist = OutboundAllowlist(frozenset({_KALSHI_HOST}))

    with pytest.raises(EgressDeniedError):
        allowlist.require("https://evil.example.com/steal")


def test_require_denies_a_lookalike_host() -> None:
    """A host that merely *contains* or extends the real one is still denied
    -- guards against a naive substring/prefix/suffix match.
    """
    allowlist = OutboundAllowlist(frozenset({_KALSHI_HOST}))

    with pytest.raises(EgressDeniedError):
        allowlist.require(f"https://{_KALSHI_HOST}.evil.com/phish")
    with pytest.raises(EgressDeniedError):
        allowlist.require(f"https://not-{_KALSHI_HOST}/phish")


@pytest.mark.parametrize("scheme", ["ftp", "file", "gopher", ""])
def test_require_denies_a_non_http_scheme(scheme: str) -> None:
    """Only ``http``/``https`` are ever admissible schemes."""
    allowlist = OutboundAllowlist(frozenset({_KALSHI_HOST}))
    url = f"{scheme}://{_KALSHI_HOST}/x" if scheme else f"//{_KALSHI_HOST}/x"

    with pytest.raises(EgressDeniedError):
        allowlist.require(url)


def test_require_denies_a_url_with_no_host() -> None:
    """A schemed URL with no host at all is denied, not a crash."""
    allowlist = OutboundAllowlist(frozenset({_KALSHI_HOST}))

    with pytest.raises(EgressDeniedError):
        allowlist.require("https:///no-host-here")


@pytest.mark.parametrize("bad_char", ["\t", "\n", "\r", " ", "\x00", "\x7f"])
def test_require_denies_a_url_containing_a_control_or_whitespace_character(
    bad_char: str,
) -> None:
    """A control/whitespace byte anywhere in the URL is denied *before*
    parsing -- the exact parse-differential SSRF screen
    ``windbreak.forecast.sandbox._has_unsafe_url_chars`` already applies, so
    a byte one parser strips and another keeps can never smuggle a
    different real host past the gate.
    """
    allowlist = OutboundAllowlist(frozenset({_KALSHI_HOST}))
    url = f"https://{_KALSHI_HOST}{bad_char}.evil.com/x"

    with pytest.raises(EgressDeniedError):
        allowlist.require(url)


# --- OutboundAllowlist.require: recorder wiring --------------------------------


def test_require_denial_records_an_egress_denied_event_when_a_recorder_is_wired() -> (
    None
):
    """A denial with a recorder wired both raises *and* records exactly one
    ``EgressDenied`` event -- the recorder never suppresses the raise.
    """
    recorder = _RecordingRecorder()
    allowlist = OutboundAllowlist(frozenset({_KALSHI_HOST}), recorder=recorder)

    with pytest.raises(EgressDeniedError):
        allowlist.require("https://evil.example.com/steal")

    denied = [event for event in recorder.events if event.event_type == "EgressDenied"]
    assert len(denied) == 1


def test_require_denial_still_raises_fail_closed_with_no_recorder_wired() -> None:
    """A denial with no recorder wired at all still raises -- fail-closed
    first, telemetry second.
    """
    allowlist = OutboundAllowlist(frozenset({_KALSHI_HOST}))

    with pytest.raises(EgressDeniedError):
        allowlist.require("https://evil.example.com/steal")


def test_require_success_records_nothing() -> None:
    """An allowed request never touches the recorder."""
    recorder = _RecordingRecorder()
    allowlist = OutboundAllowlist(frozenset({_KALSHI_HOST}), recorder=recorder)

    allowlist.require(f"https://{_KALSHI_HOST}/x")

    assert recorder.events == []


# --- allowlist_from_config: exchange + forecast-provider derivation ------------


def test_allowlist_from_config_derives_the_kalshi_exchange_host_by_default() -> None:
    """`WindbreakConfig()`'s default ``exchange.provider == "kalshi"``
    contributes exactly the production Kalshi host.
    """
    allowlist = allowlist_from_config(WindbreakConfig())

    allowlist.require(f"https://{_KALSHI_HOST}/trade-api/v2/markets")


def test_allowlist_from_config_derives_both_default_forecast_provider_hosts() -> None:
    """`WindbreakConfig()`'s default two-model ensemble
    (``anthropic``/``openai``) contributes both provider hosts.
    """
    allowlist = allowlist_from_config(WindbreakConfig())

    allowlist.require(f"https://{_ANTHROPIC_HOST}/v1/messages")
    allowlist.require(f"https://{_OPENAI_HOST}/v1/responses")


def test_allowlist_from_config_unknown_forecast_provider_contributes_no_host() -> None:
    """The default triage model's provider,
    ``"cheapest-adequate"`` (not a real host-mapped provider name), never
    resolves to a usable host -- fail closed on an unrecognized provider.
    """
    config = WindbreakConfig()
    assert config.forecast.triage_model.provider == "cheapest-adequate"

    allowlist = allowlist_from_config(config)

    with pytest.raises(EgressDeniedError):
        allowlist.require("https://cheapest-adequate.example.com/v1/x")


def test_allowlist_from_config_unknown_exchange_provider_contributes_no_host() -> None:
    """An unrecognized ``exchange.provider`` contributes no host at all, so a
    later attempt to reach the real Kalshi host through this allowlist fails
    closed -- an unconfigured/unknown exchange must never silently inherit
    network access to a *different* exchange's host.
    """
    config = dataclasses.replace(
        WindbreakConfig(),
        exchange=dataclasses.replace(WindbreakConfig().exchange, provider="acme-dex"),
    )

    allowlist = allowlist_from_config(config)

    with pytest.raises(EgressDeniedError):
        allowlist.require(f"https://{_KALSHI_HOST}/trade-api/v2/markets")


# --- Exchange environment: demo vs production host derivation (issue #318) ----
#
# ``exchange.environment`` selects the venue a live deployment dials --
# ``windbreak.connector.live._ENVIRONMENT_API_BASES`` maps SPEC S16's
# ``demo | production`` to their API bases -- and this allowlist is the control
# that stops that deployment reaching any *other* venue's host. Only the ``demo``
# arm was ever exercised (the shipped default is ``environment = "demo"``), so
# the production-only arm -- the false branch of ``environment == "demo"``, which
# is what withholds demo egress from a production deployment -- had no test at
# all, and nothing pinned that the demo host stays *out* of a production host
# set.
#
# Both arms are asserted as exact host **sets**, not by membership: for an egress
# allowlist an over-broad set is the entire risk, and a membership assertion
# passes just as happily while the set silently grows a host.


def _exchange_with_environment(environment: str) -> ExchangeConfig:
    """Build the shipped default exchange config with ``environment`` replaced.

    Args:
        environment: The ``exchange.environment`` value to configure.

    Returns:
        The default ``ExchangeConfig`` (so ``provider`` stays ``"kalshi"``)
        carrying that environment.
    """
    return dataclasses.replace(WindbreakConfig().exchange, environment=environment)


def test_exchange_config_defaults_are_kalshi_in_the_demo_environment() -> None:
    """Fixture assumption: the shipped default exchange is ``kalshi``/``demo``.

    Every ``WindbreakConfig()``-based test above therefore exercises the *demo*
    arm; the production arm is only reachable by overriding this field, which is
    exactly why it went untested.
    """
    exchange = WindbreakConfig().exchange
    assert exchange.provider == "kalshi"
    assert exchange.environment == "demo"


def test_exchange_hosts_in_a_production_environment_exclude_the_demo_host() -> None:
    """A ``kalshi`` provider outside ``demo`` derives exactly the production host.

    The demo host must be absent: a production deployment that could also dial
    ``demo-api.kalshi.co`` has an allowlist one host wider than the venue it
    declared.
    """
    hosts = _exchange_hosts(_exchange_with_environment("production"))

    assert hosts == frozenset({_KALSHI_HOST})


def test_exchange_hosts_in_a_demo_environment_add_the_demo_host() -> None:
    """A ``demo`` environment derives exactly the production *and* demo hosts.

    The complementary arm, pinned as a set so the demo addition cannot silently
    disappear (which would leave a demo deployment unable to reach its own venue)
    and cannot silently grow a third host either.
    """
    hosts = _exchange_hosts(_exchange_with_environment("demo"))

    assert hosts == frozenset({_KALSHI_HOST, _KALSHI_DEMO_HOST})


@pytest.mark.parametrize(
    "environment",
    ["Demo", "DEMO", " demo", "demo ", "demo2", "predemo", "prod", "", "production"],
)
def test_only_the_exact_demo_token_admits_the_demo_host(environment: str) -> None:
    """Near-misses of ``"demo"`` derive the production host set, not the demo one.

    The match is an exact, case-sensitive equality, and these cases pin that
    dimension rather than merely "the two arms differ": a substring, prefix or
    case-folded comparison would admit several of these and still pass a test
    that only contrasted ``"demo"`` with ``"production"``. Failing closed here
    agrees with ``windbreak.connector.live``, which refuses to resolve an API
    base for any environment token outside its exact ``demo``/``production``
    keys rather than defaulting one to the real venue.

    Args:
        environment: A near-miss ``exchange.environment`` token.
    """
    hosts = _exchange_hosts(_exchange_with_environment(environment))

    assert hosts == frozenset({_KALSHI_HOST})


def test_allowlist_from_config_in_production_denies_the_demo_kalshi_host() -> None:
    """Through the public seam: production admits its own host and denies demo's.

    ``_exchange_hosts`` is private, so this drives the same branch through
    ``allowlist_from_config`` -> ``OutboundAllowlist.require`` and asserts the
    refusal by exact type and exact message -- ``EgressDeniedError`` has
    subclass-free ancestry here, but matching a substring would pass for any
    denial reason at all, including one that never reached the host check.
    """
    config = dataclasses.replace(
        WindbreakConfig(), exchange=_exchange_with_environment("production")
    )
    allowlist = allowlist_from_config(config)

    allowlist.require(f"https://{_KALSHI_HOST}/trade-api/v2/markets")

    with pytest.raises(EgressDeniedError) as denied:
        allowlist.require(f"https://{_KALSHI_DEMO_HOST}/trade-api/v2/markets")

    assert type(denied.value) is EgressDeniedError
    assert str(denied.value) == (
        "egress denied: host 'demo-api.kalshi.co' is not allowlisted "
        "(url 'https://demo-api.kalshi.co/trade-api/v2/markets')"
    )


def test_allowlist_from_config_in_a_demo_environment_admits_the_demo_host() -> None:
    """Through the public seam: a ``demo`` deployment may dial the demo host.

    The positive control for the test above -- without it, an allowlist that had
    stopped deriving the demo host entirely would still satisfy the production
    assertion.
    """
    config = dataclasses.replace(
        WindbreakConfig(), exchange=_exchange_with_environment("demo")
    )

    allowlist = allowlist_from_config(config)

    allowlist.require(f"https://{_KALSHI_DEMO_HOST}/trade-api/v2/markets")


def test_allowlist_from_config_forwards_the_recorder() -> None:
    """A ``recorder`` passed to ``allowlist_from_config`` is the one every
    later denial records through.
    """
    recorder = _RecordingRecorder()
    allowlist = allowlist_from_config(WindbreakConfig(), recorder=recorder)

    with pytest.raises(EgressDeniedError):
        allowlist.require("https://evil.example.com/steal")

    assert any(event.event_type == "EgressDenied" for event in recorder.events)


# --- allowlist_from_config: vote_ensemble provider derivation (issue #240) -----
#
# Issue #240 documents the split between the legacy triage/promotion
# ``ForecastConfig.ensemble`` (``ModelRef``) and the vote-stage per-member
# ``ForecastConfig.vote_ensemble`` (``EnsembleMemberConfig``, issues #184/#191)
# and repoints the egress allowlist to union in hosts for providers named by
# *either* field -- additive-only, so legacy ``ensemble`` providers still
# contribute hosts and the default config's derived host set is unchanged.


def test_vote_ensemble_openai_member_absent_from_ensemble_admits_host() -> None:
    """A ``vote_ensemble`` member naming a provider absent from the legacy
    ``ensemble`` still contributes that provider's host -- the allowlist must
    union both fields, not derive from ``ensemble`` alone.
    """
    default_forecast = WindbreakConfig().forecast
    config = dataclasses.replace(
        WindbreakConfig(),
        forecast=dataclasses.replace(
            default_forecast,
            ensemble=(ModelRef("anthropic", "pinned-by-operator"),),
            vote_ensemble=(
                EnsembleMemberConfig("openai", "gpt-5-2025-08-07", "2024-09-30"),
            ),
        ),
    )

    allowlist = allowlist_from_config(config)

    allowlist.require(f"https://{_OPENAI_HOST}/v1/responses")


def test_vote_ensemble_anthropic_member_absent_from_ensemble_admits_host() -> None:
    """Mirror of the openai case above, with the providers swapped -- guards
    against a fix that hardcodes one specific provider name rather than
    genuinely unioning ``vote_ensemble`` providers into the host set.
    """
    default_forecast = WindbreakConfig().forecast
    config = dataclasses.replace(
        WindbreakConfig(),
        forecast=dataclasses.replace(
            default_forecast,
            ensemble=(ModelRef("openai", "pinned-by-operator"),),
            vote_ensemble=(
                EnsembleMemberConfig(
                    "anthropic", "claude-sonnet-4-5-20250929", "2025-07-31"
                ),
            ),
        ),
    )

    allowlist = allowlist_from_config(config)

    allowlist.require(f"https://{_ANTHROPIC_HOST}/v1/messages")


def test_vote_ensemble_unrecognized_provider_contributes_no_host() -> None:
    """A ``vote_ensemble`` member naming a provider absent from
    ``_FORECAST_PROVIDER_HOSTS`` contributes no host -- fail closed on an
    unrecognized provider, exactly like the legacy ``ensemble``/``triage_model``
    derivation.

    Uses ``futuresearch`` with its section left at the shipped default, so the
    fail-closed direction issue #555 preserved is the one under test: naming the
    provider is not on its own enough to open egress.
    """
    default_forecast = WindbreakConfig().forecast
    config = dataclasses.replace(
        WindbreakConfig(),
        forecast=dataclasses.replace(
            default_forecast,
            vote_ensemble=(EnsembleMemberConfig("futuresearch", "x", "y"),),
        ),
    )

    allowlist = allowlist_from_config(config)

    with pytest.raises(EgressDeniedError):
        allowlist.require("https://futuresearch.example.com/v1/x")


# --- allowlist_from_config: the research forecaster endpoint (issue #555) ------


def _futuresearch_config(*, endpoint_url: str, selected: bool) -> WindbreakConfig:
    """Build a config with a research-forecaster endpoint, optionally selected.

    Args:
        endpoint_url: The ``forecast.futuresearch.endpoint_url`` to set.
        selected: Whether a ``vote_ensemble`` member names the provider.

    Returns:
        The configuration under test.
    """
    from windbreak.config.schema import FutureSearchProviderSettings

    default_forecast = WindbreakConfig().forecast
    ensemble = (
        (EnsembleMemberConfig("futuresearch", "fs-1", "server-managed"),)
        if selected
        else default_forecast.vote_ensemble
    )
    return dataclasses.replace(
        WindbreakConfig(),
        forecast=dataclasses.replace(
            default_forecast,
            vote_ensemble=ensemble,
            futuresearch=FutureSearchProviderSettings(endpoint_url=endpoint_url),
        ),
    )


def test_a_selected_research_forecaster_admits_its_configured_endpoint_host() -> None:
    """A configured, *selected* research forecaster reaches its own endpoint.

    Its host cannot come from the closed provider->host table the way
    ``anthropic``/``openai`` do: a hosted research forecaster is an
    operator-chosen deployment, so only configuration can name it.
    """
    allowlist = allowlist_from_config(
        _futuresearch_config(
            endpoint_url="https://research.futuresearch.example/v1/forecast",
            selected=True,
        )
    )

    allowlist.require("https://research.futuresearch.example/v1/forecast")


def test_an_unselected_research_forecaster_endpoint_opens_no_egress() -> None:
    """Writing an endpoint down does not on its own admit it.

    Two independent fields must agree -- a member must name the provider *and*
    the endpoint must parse -- so a mistyped or tampered ``endpoint_url`` in a
    deployment that never selected the research forecaster reaches nothing. This
    is the alert sinks' "declare it twice" property, without a second host list
    to transcribe.
    """
    allowlist = allowlist_from_config(
        _futuresearch_config(
            endpoint_url="https://research.futuresearch.example/v1/forecast",
            selected=False,
        )
    )

    with pytest.raises(EgressDeniedError):
        allowlist.require("https://research.futuresearch.example/v1/forecast")


def test_a_selected_research_forecaster_admits_only_its_own_host() -> None:
    """The derivation adds one host, not a blanket allowance."""
    allowlist = allowlist_from_config(
        _futuresearch_config(
            endpoint_url="https://research.futuresearch.example/v1/forecast",
            selected=True,
        )
    )

    with pytest.raises(EgressDeniedError):
        allowlist.require("https://evil.example.com/steal")


def test_allowlist_from_config_default_forecast_host_set_is_unchanged() -> None:
    """`WindbreakConfig()`'s default forecast host set is exactly the
    unchanged two-provider set -- proves the default configuration's derived
    allowlist is byte-identical before and after the ``vote_ensemble`` union,
    since the default ``vote_ensemble``'s providers (``openai``/``anthropic``)
    are already covered by the default ``ensemble``.
    """
    allowlist = allowlist_from_config(WindbreakConfig())

    allowlist.require(f"https://{_ANTHROPIC_HOST}/v1/messages")
    allowlist.require(f"https://{_OPENAI_HOST}/v1/responses")
    with pytest.raises(EgressDeniedError):
        allowlist.require("https://evil.example.com/steal")


def test_legacy_ensemble_still_admits_hosts_with_empty_vote_ensemble() -> None:
    """With ``vote_ensemble`` emptied to ``()``, the legacy ``ensemble``
    providers still contribute their hosts -- the union is additive, not a
    replacement of the legacy derivation.
    """
    default_forecast = WindbreakConfig().forecast
    config = dataclasses.replace(
        WindbreakConfig(),
        forecast=dataclasses.replace(
            default_forecast,
            ensemble=(ModelRef("anthropic", "pinned-by-operator"),),
            vote_ensemble=(),
        ),
    )

    allowlist = allowlist_from_config(config)

    allowlist.require(f"https://{_ANTHROPIC_HOST}/v1/messages")


# --- allowlist_from_config: live-research host derivation (issue #192) ---------


def test_allowlist_from_config_derives_the_research_search_endpoint_host() -> None:
    """``config.forecast.research.search_endpoint_url``'s host is admitted,
    exactly like the exchange and per-model forecast-provider hosts.
    """
    research = ResearchSettings(search_endpoint_url="https://search.example/v1/search")
    config = dataclasses.replace(
        WindbreakConfig(),
        forecast=dataclasses.replace(WindbreakConfig().forecast, research=research),
    )

    allowlist = allowlist_from_config(config)

    allowlist.require("https://search.example/v1/search")


def test_allowlist_from_config_derives_each_allowed_research_host() -> None:
    """Every host named in ``config.forecast.research.allowed_research_hosts``
    is admitted.
    """
    research = ResearchSettings(
        allowed_research_hosts=("news.example", "wire-service.example")
    )
    config = dataclasses.replace(
        WindbreakConfig(),
        forecast=dataclasses.replace(WindbreakConfig().forecast, research=research),
    )

    allowlist = allowlist_from_config(config)

    allowlist.require("https://news.example/article")
    allowlist.require("https://wire-service.example/article")


def test_allowlist_from_config_default_research_settings_contributes_no_host() -> None:
    """`WindbreakConfig()`'s default, unconfigured
    ``forecast.research`` (a placeholder endpoint URL and an empty
    ``allowed_research_hosts`` tuple) contributes zero hosts -- an
    unconfigured live-research deployment fails closed rather than silently
    admitting some plausible-looking default host.
    """
    allowlist = allowlist_from_config(WindbreakConfig())

    with pytest.raises(EgressDeniedError):
        allowlist.require("https://search.example/v1/search")
    with pytest.raises(EgressDeniedError):
        allowlist.require("https://configured-by-operator/x")


def test_allowlist_from_config_research_hosts_additive() -> None:
    """A configured research section adds to -- never replaces -- the
    existing exchange and forecast-provider host derivation.
    """
    research = ResearchSettings(allowed_research_hosts=("news.example",))
    config = dataclasses.replace(
        WindbreakConfig(),
        forecast=dataclasses.replace(WindbreakConfig().forecast, research=research),
    )

    allowlist = allowlist_from_config(config)

    allowlist.require(f"https://{_KALSHI_HOST}/trade-api/v2/markets")
    allowlist.require(f"https://{_ANTHROPIC_HOST}/v1/messages")
    allowlist.require("https://news.example/article")


def test_allowlist_from_config_research_settings_fixture_assumption() -> None:
    """Fixture assumption: ``ForecastConfig``'s default ``research`` field is
    a bare ``ResearchSettings()`` -- the host-derivation tests above build
    their overrides against that same default via ``dataclasses.replace``.
    """
    assert ForecastConfig().research == ResearchSettings()


# --- Alert-sink host derivation (issue #274 resolved the open question) --------
#
# The question flagged here for the architect -- where the configured alert
# sink's host comes from -- is answered by issue #274: it comes from
# ``alerts.allowed_hosts``, a dedicated operator-declared list, and NOT from
# the per-sink ``base_url``/``url``/``smtp.host`` destination fields the same
# section now also carries. Deriving the allowlist from the very URLs it
# screens would make the check unfalsifiable (every configured sink would
# admit itself), so the declaration is kept independent of the destination.
# The tests below pin both halves of that.


def test_allowlist_from_config_derives_declared_alert_hosts() -> None:
    """Each ``alerts.allowed_hosts`` entry joins the allowlist, case-insensitively."""
    config = dataclasses.replace(
        WindbreakConfig(),
        alerts=AlertsConfig(allowed_hosts=("Ntfy.Example", "hooks.example")),
    )

    allowlist = allowlist_from_config(config)

    allowlist.require("https://ntfy.example/topic")
    allowlist.require("https://hooks.example/incoming")


def test_allowlist_from_config_ignores_the_sink_destination_fields() -> None:
    """A sink destination alone never admits its own host.

    This is what keeps the egress check able to *veto* a configured sink: if the
    sink's own destination fed the allowlist, no configured sink could ever be
    denied and the check would be decorative. The sink's destination is not even
    in configuration -- `base_url_env` names the environment variable holding it
    -- so the allowlist could not derive it without reading the environment,
    which it deliberately never does.
    """
    config = dataclasses.replace(
        WindbreakConfig(),
        alerts=AlertsConfig(
            sinks=(
                AlertSink(
                    type="ntfy",
                    topic_env="WINDBREAK_NTFY_TOPIC",
                    base_url_env="WINDBREAK_NTFY_BASE_URL",
                ),
            ),
        ),
    )

    with pytest.raises(EgressDeniedError):
        allowlist_from_config(config).require("https://ntfy.example/ops")


def test_allowlist_from_config_derives_no_alert_host_by_default() -> None:
    """The shipped default declares no alert host, so alert egress is closed."""
    assert WindbreakConfig().alerts.allowed_hosts == ()

    with pytest.raises(EgressDeniedError):
        allowlist_from_config(WindbreakConfig()).require("https://ntfy.sh/topic")


class TestRequireHost:
    """Tests for `OutboundAllowlist.require_host`, the non-URL egress screen.

    SMTP (`windbreak.alerts.sinks.SmtpSink`) speaks to a bare host over a
    non-http scheme, so `require`'s URL shape cannot screen it; without
    `require_host` SMTP would be the one outbound path with no allowlist.
    """

    def test_allowlisted_host_is_permitted(self) -> None:
        """A declared host passes, case-insensitively."""
        OutboundAllowlist(frozenset({"smtp.example"})).require_host("SMTP.Example")

    def test_off_list_host_raises(self) -> None:
        """An undeclared host is denied."""
        with pytest.raises(EgressDeniedError):
            OutboundAllowlist(frozenset({"smtp.example"})).require_host(
                "relay.internal"
            )

    def test_empty_host_raises(self) -> None:
        """An empty host is denied rather than treated as a wildcard."""
        with pytest.raises(EgressDeniedError):
            OutboundAllowlist(frozenset({"smtp.example"})).require_host("")

    def test_control_character_in_host_raises(self) -> None:
        """A control/whitespace byte fails closed before any matching."""
        with pytest.raises(EgressDeniedError):
            OutboundAllowlist(frozenset({"smtp.example"})).require_host(
                "smtp.example\nrelay.internal"
            )

    def test_denial_records_an_egress_denied_event_and_still_raises(self) -> None:
        """Telemetry is additive: the recorder never suppresses the refusal."""
        recorder = _RecordingRecorder()
        allowlist = OutboundAllowlist(frozenset({"smtp.example"}), recorder=recorder)

        with pytest.raises(EgressDeniedError):
            allowlist.require_host("relay.internal")

        assert [event.event_type for event in recorder.events] == ["EgressDenied"]
        assert recorder.events[0].payload == {"host": "relay.internal"}


# --- ModelRef sanity (documents the fixture assumption above) ------------------


def test_default_ensemble_providers_are_anthropic_and_openai() -> None:
    """Fixture assumption: `WindbreakConfig()`'s default ensemble is exactly
    the two-model ``anthropic``/``openai`` pair the host-derivation tests
    above rely on.
    """
    ensemble = WindbreakConfig().forecast.ensemble
    providers = {model.provider for model in ensemble}
    assert providers == {"anthropic", "openai"}
    assert all(isinstance(model, ModelRef) for model in ensemble)
