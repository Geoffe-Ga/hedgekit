"""Failing-first tests for a structural outbound-network allowlist (issue #57,
RED).

Issue #57's plan calls for the LIVE_MICRO deployment to be able to reach only
an explicit, small set of hosts -- the exchange, the two forecast providers,
and the configured alert sink -- and *nothing else*, mirroring
`windbreak.forecast.sandbox.ResearchTools.fetch`'s structural (not
prompt-based) egress gate: parse-differential SSRF screening, exact
lowercased hostname matching, and fail-closed on any parse ambiguity.

`windbreak/net/allowlist.py` (and the `windbreak/net/` package itself) does
not exist yet, so every import below fails collection with
`ModuleNotFoundError: No module named 'windbreak.net'` -- the expected Gate 1
RED state for issue #57.

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
  - **Open question, NOT tested here (flagged for the architect/implementer
    to resolve):** the plan also calls for deriving a host from each
    configured alert sink, but today's ``windbreak.config.schema.AlertSink``
    has only ``type``/``topic`` -- no ``base_url`` -- so there is no
    schema-level field to derive that host from without either adding a
    banned ``type: ignore`` to force a not-yet-existing keyword argument
    through mypy, or a test-only file editing the production schema itself.
    See the note near the bottom of this file.

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
    EnsembleMemberConfig,
    ForecastConfig,
    ModelRef,
    ResearchSettings,
    WindbreakConfig,
)
from windbreak.net.allowlist import (
    EgressDeniedError,
    OutboundAllowlist,
    allowlist_from_config,
)

if TYPE_CHECKING:
    from windbreak.ledger.events import Event

#: The current-generation Kalshi public API host (SPEC S7.1), matching
#: ``windbreak.connector.kalshi.client.KALSHI_API_BASE``'s hostname.
_KALSHI_HOST = "api.elections.kalshi.com"
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


# --- ModelRef sanity (documents the fixture assumption above) ------------------
#
# NOTE (flagged for implementation/architect to confirm, not itself tested
# here): the architect's plan also calls for deriving an allowlist host from
# each configured alert sink's ``base_url``. Today's
# ``windbreak.config.schema.AlertSink`` has only ``type``/``topic`` -- no
# ``base_url`` -- so there is no schema-level field to derive that host from
# yet, and this file cannot pin that sub-behavior without either (a) adding a
# banned ``type: ignore`` to force a not-yet-existing keyword argument
# through mypy, or (b) itself editing the production schema (out of scope for
# a test-only file). Whether ``AlertSink`` should gain a ``base_url`` field,
# or the alert-sink host should instead be derived from the runtime
# ``windbreak.alerts.sinks.NtfySinkConfig`` the operator separately
# constructs, is an open question for the chief architect / implementation
# specialist to resolve; `allowlist_from_config`'s exchange- and
# forecast-provider-host derivation below does not depend on the answer.


def test_default_ensemble_providers_are_anthropic_and_openai() -> None:
    """Fixture assumption: `WindbreakConfig()`'s default ensemble is exactly
    the two-model ``anthropic``/``openai`` pair the host-derivation tests
    above rely on.
    """
    ensemble = WindbreakConfig().forecast.ensemble
    providers = {model.provider for model in ensemble}
    assert providers == {"anthropic", "openai"}
    assert all(isinstance(model, ModelRef) for model in ensemble)
