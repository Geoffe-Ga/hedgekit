"""Typed, immutable configuration schema for windbreak (SPEC §16).

Every configuration section is a frozen dataclass whose field defaults are
the SPEC §16 example verbatim, so ``WindbreakConfig()`` is a complete, valid,
production-shaped configuration on its own. All leaf values are integers,
strings, booleans, ``None``, or tuples of those, never floats (SPEC §6.1's
integer-units invariant). The single fractional value in SPEC §16,
``bootstrap_confidence``, is stored as an integer parts-per-million field and
converted by the loader; see :attr:`EvaluationConfig.bootstrap_confidence_ppm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Metadata tag naming the YAML key a field is read from when it differs from
#: the field's own name (e.g. ``bootstrap_confidence`` -> ``*_ppm``).
_YAML_KEY = "yaml_key"

#: Metadata tag naming the loader converter applied to a field's raw value.
_CONVERT = "convert"

#: Number of most-recent resolved LIVE forecasts / execution records the
#: live-divergence gates score over, keyed off ``created_sequence`` descending
#: (issue #58, SPEC §10.9/§10.10). A count, not a ppm.
_DEFAULT_LIVE_ROLLING_WINDOW_SIZE = 100

#: Ceiling on the live-vs-paper cost slippage ratio, in parts-per-million;
#: 1_500_000 ppm == 1.5x the modeled cost (issue #58's worked example).
_DEFAULT_LIVE_SLIPPAGE_RATIO_LIMIT_PPM = 1_500_000

#: Allowed LIVE-over-PAPER rolling Brier degradation band, in parts-per-million,
#: before the divergence monitor demotes (issue #58).
_DEFAULT_LIVE_BRIER_DEGRADATION_BAND_PPM = 50_000


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A pinned reference to a single forecasting model.

    Attributes:
        provider: The model provider (e.g. ``anthropic``, ``openai``).
        model: The provider-specific, operator-pinned model identifier.
    """

    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class AlertSink:
    """A destination for operator alerts.

    Attributes:
        type: The sink transport (e.g. ``ntfy``).
        topic: The operator-configured topic or channel to publish to.
    """

    type: str
    topic: str


@dataclass(frozen=True, slots=True)
class HorizonDays:
    """Inclusive bounds, in days, on tradeable market resolution horizons."""

    min: int = 2
    max: int = 120


@dataclass(frozen=True, slots=True)
class ExchangeConfig:
    """Exchange connectivity and product-eligibility policy."""

    provider: str = "kalshi"
    environment: str = "demo"
    product_allowlist: tuple[str, ...] = ("predictions",)
    product_blocklist: tuple[str, ...] = ("perps", "margin")
    require_jurisdiction_eligible: bool = True


@dataclass(frozen=True, slots=True)
class CapitalConfig:
    """Capital floor, ratchet, and deployment limits (micro-dollar units)."""

    floor_micros: int = 1000000000
    floor_ratchet_ppm_of_new_profits: int = 500000
    profit_sweep_threshold_micros: int = 250000000
    max_deploy_pct_above_floor_ppm: int = 500000
    micro_cap_micros: int = 100000000


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Risk-kernel thresholds and time-to-live limits.

    Attributes:
        require_human_ack_above_micros: Notional above which a trade needs
            human acknowledgement, or ``None`` to require none (paper mode).
        kill_after_consecutive_mismatches: Number of consecutive reconciliation
            ``BREACH`` outcomes that auto-engages the kill switch (issue #35).
        verification_balance_tolerance_micros: The admissible available-cash
            drift, in micros, before the live verifier grades a reconciliation
            ``BREACH`` (issue #236). Defaults to ``0`` -- a fail-closed
            exact-match: any drift at all is a breach until an operator loosens
            it.
        verification_position_tolerance_centis: The admissible per-ticker
            position drift, in contract-centis, before the live verifier grades
            a ``BREACH`` (issue #236). Defaults to ``0`` -- fail-closed
            exact-match, as above.
    """

    min_net_edge_ppm: int = 30000
    annualized_hurdle_ppm: int = 200000
    idle_cash_apr_ppm: int = 40000
    kelly_fraction_ppm: int = 100000
    dispersion_zero_ceiling_ppm: int = 200000
    min_open_price_pips: int = 500
    max_open_price_pips: int = 9500
    max_participation_ppm: int = 250000
    max_pos_market_pct_ppm: int = 20000
    max_pos_event_pct_ppm: int = 40000
    max_pos_bucket_pct_ppm: int = 100000
    daily_loss_limit_pct_ppm: int = 20000
    max_drawdown_pct_ppm: int = 100000
    max_orders_per_hour: int = 20
    max_notional_per_day_micros: int = 500000000
    quote_ttl_seconds: int = 10
    approval_ttl_seconds: int = 60
    resting_order_ttl_seconds: int = 900
    cancel_on_move_ticks: int = 2
    clock_skew_max_seconds: int = 2
    require_human_ack_above_micros: int | None = None
    kill_after_consecutive_mismatches: int = 3
    verification_balance_tolerance_micros: int = 0
    verification_position_tolerance_centis: int = 0


@dataclass(frozen=True, slots=True)
class ScreenerConfig:
    """Market-screening filters applied before forecasting."""

    category_blocklist: tuple[str, ...] = (
        "sports",
        "crypto_price",
        "celebrity",
        "insider_prone",
    )
    min_volume_24h_micros: int = 5000000000
    min_depth_contract_centis: int = 10000
    horizon_days: HorizonDays = field(default_factory=HorizonDays)


@dataclass(frozen=True, slots=True)
class ForecastBudget:
    """Per-forecast and per-day research spend caps (micro-dollar units)."""

    per_forecast_micros: int = 3000000
    per_day_micros: int = 20000000
    max_pages: int = 20


@dataclass(frozen=True, slots=True)
class CanaryConfig:
    """Cadence policy for canary forecasts that probe calibration drift."""

    enabled: bool = True
    cadence_days: int = 7


@dataclass(frozen=True, slots=True)
class EnsembleMemberConfig:
    """One vote-ensemble member's pinned provider provenance (SPEC S6.3).

    Backs :attr:`ForecastConfig.vote_ensemble`. A structural triple the forecast
    engine consumes without importing this package (SPEC S8.3 sandbox boundary):
    it matches the engine's own ``EnsembleMemberLike`` shape by exposing the
    same three read-only strings.

    Attributes:
        provider: The LLM provider identifier (e.g. ``openai``).
        model_version: The pinned, operator-chosen model version string.
        training_cutoff: The model's declared training cutoff date.
    """

    provider: str
    model_version: str
    training_cutoff: str


def _default_ensemble() -> tuple[ModelRef, ...]:
    """Return the SPEC §16 default two-model forecasting ensemble."""
    return (
        ModelRef("anthropic", "pinned-by-operator"),
        ModelRef("openai", "pinned-by-operator"),
    )


def _default_triage_model() -> ModelRef:
    """Return the SPEC §16 default triage model reference."""
    return ModelRef("cheapest-adequate", "pinned-by-operator")


def _default_vote_ensemble() -> tuple[EnsembleMemberConfig, ...]:
    """Return the default three-member vote ensemble (issue #191).

    Pinned to the real, operator-pinned live triple -- mirror-equal in
    provenance to the forecast engine's own ``DEFAULT_VOTE_ENSEMBLE`` -- so a
    config file omitting ``vote_ensemble`` and the forecast engine's built-in
    default drive the vote stage with identical ensemble provenance and ordering.
    """
    return (
        EnsembleMemberConfig("openai", "gpt-5-2025-08-07", "2024-09-30"),
        EnsembleMemberConfig("anthropic", "claude-sonnet-4-5-20250929", "2025-07-31"),
        EnsembleMemberConfig("openai", "gpt-5-mini-2025-08-07", "2024-05-31"),
    )


@dataclass(frozen=True, slots=True)
class FutureSearchProviderSettings:
    """The FutureSearch research-forecaster provider's config-schema section.

    The config-schema mirror of
    :class:`windbreak.forecast.providers.futuresearch.FutureSearchProviderConfig`,
    with SPEC-integer-units-only leaves. ``endpoint_url`` and
    ``pinned_forecaster_versions`` have no natural real-world default, so -- like
    :class:`AlertSink` and :class:`ModelRef` elsewhere in this schema -- they
    default to the repo's "operator must fill this in" placeholder idiom rather
    than an invented, plausible-looking endpoint/version.

    Attributes:
        endpoint_url: The forecast endpoint the provider POSTs to.
        pinned_forecaster_versions: The operator-pinned forecaster versions a
            reported version must belong to (else drift).
        api_key_env: The environment variable a live transport reads the API
            key from.
        per_call_ceiling_micros: The reported-cost fallback, in micros.
        reject_on_version_drift: Whether an unpinned reported version rejects
            (strict) or proceeds with a logged warning.
    """

    endpoint_url: str = "configured-by-operator"
    pinned_forecaster_versions: tuple[str, ...] = ("pinned-by-operator",)
    api_key_env: str = "FUTURESEARCH_API_KEY"
    per_call_ceiling_micros: int = 2000000
    reject_on_version_drift: bool = True


@dataclass(frozen=True, slots=True)
class ResearchSettings:
    """The live web-research config-schema section (issue #192).

    Backs the :class:`windbreak.forecast.providers.search_live.LiveSearchTransport`
    / :class:`windbreak.forecast.providers.fetch_live.LiveFetchTransport` pair
    and the outbound-allowlist host derivation in
    :func:`windbreak.net.allowlist.allowlist_from_config`. Per SPEC §6.1 every
    leaf is an integer, string, or tuple of strings -- never a float.

    ``search_endpoint_url`` and ``allowed_research_hosts`` have no safe
    real-world default, so -- like :class:`AlertSink`, :class:`ModelRef`, and
    :class:`FutureSearchProviderSettings` -- they default to the "operator must
    fill this in" placeholder idiom and fail *closed*: an unconfigured
    deployment's live-research egress allowlist contributes zero hosts rather
    than an invented, plausible-looking one.

    Attributes:
        search_endpoint_url: The search endpoint a live search POSTs to.
        search_api_key_env: The environment variable a live recorder reads the
            search API key from; never a secret itself, only the var's *name*.
        allowed_research_hosts: The hosts a live fetch may reach, added to the
            outbound allowlist.
        fetch_timeout_seconds: The per-fetch timeout, in whole seconds.
        fetch_max_bytes: The maximum accepted fetched-body size, in bytes.
        allowed_content_types: The response media types a live fetch accepts.
    """

    search_endpoint_url: str = "configured-by-operator"
    search_api_key_env: str = "RESEARCH_SEARCH_API_KEY"
    allowed_research_hosts: tuple[str, ...] = ()
    fetch_timeout_seconds: int = 30
    fetch_max_bytes: int = 2_000_000
    allowed_content_types: tuple[str, ...] = ("text/html",)


@dataclass(frozen=True, slots=True)
class ProviderGateConfig:
    """Per-provider live-eligibility promotion thresholds (issue #194, SPEC S13/S16).

    Backs :class:`windbreak.forecast.providers.track_record.ProviderTrackRecordGate`:
    a voting provider is proven (may back a live order) only once its historical
    track record clears both bars. The defaults deliberately equal
    :class:`EvaluationConfig`'s own promotion thresholds
    (``min_resolved_for_calibration`` / ``brier_skill_required_ppm``) -- the same
    statistical bar, applied per provider rather than to the ensemble.

    Attributes:
        min_resolved: Minimum resolved forecasts a provider needs to be proven.
        min_brier_skill_ppm: Minimum Brier skill over baseline, in ppm, a
            provider needs to be proven.
    """

    min_resolved: int = 150
    min_brier_skill_ppm: int = 10000


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """Ensemble, triage, budget, and calibration-canary forecasting policy.

    ``vote_ensemble`` (issue #184) is the authoritative vote-stage ensemble:
    each :class:`EnsembleMemberConfig` member carries its own provider
    provenance, and the vote stage drives providers from this field alone. The
    legacy ``ensemble`` field is the SPEC S16 triage/promotion ``ModelRef`` set,
    retained verbatim for YAML-key and config-hash stability and deprecated for
    vote purposes (ADR-0006, issue #240); it no longer sources vote-stage
    behaviour. Both fields still contribute to the outbound network allowlist.
    ``futuresearch`` (issue #189) configures the hosted research-forecaster
    provider. ``provider_gate`` (issue #194) sets the per-provider track-record
    thresholds a voting provider must clear to be live-eligible.
    """

    ensemble: tuple[ModelRef, ...] = field(default_factory=_default_ensemble)
    triage_model: ModelRef = field(default_factory=_default_triage_model)
    triage_threshold_ppm: int = 50000
    shrink_to_market_lambda_ppm: int = 250000
    budget: ForecastBudget = field(default_factory=ForecastBudget)
    min_verified_citations: int = 3
    canary: CanaryConfig = field(default_factory=CanaryConfig)
    vote_ensemble: tuple[EnsembleMemberConfig, ...] = field(
        default_factory=_default_vote_ensemble
    )
    futuresearch: FutureSearchProviderSettings = field(
        default_factory=FutureSearchProviderSettings
    )
    research: ResearchSettings = field(default_factory=ResearchSettings)
    provider_gate: ProviderGateConfig = field(default_factory=ProviderGateConfig)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Calibration and model-promotion evaluation thresholds.

    Attributes:
        bootstrap_confidence_ppm: The bootstrap confidence level expressed in
            parts-per-million (an integer), not a probability float; SPEC
            §16's ``bootstrap_confidence: 0.95`` maps here to ``950000`` via
            the loader's converter, preserving the integer-units invariant.
        live_rolling_window_size: Number of most-recent resolved LIVE forecasts /
            execution records the live-divergence gates score over (a count).
            Pinned to :data:`~windbreak.evaluation.registry.LIVE_ROLLING_WINDOW_SIZE`,
            the constant the Python reference path truncates to, and validated
            fail-closed at :func:`~windbreak.evaluation.preregistration.build_gate_plan`
            (a divergent value raises) so both dual-paths agree exactly. Changing
            it is a code change to that constant, which re-registers the gate plan
            (§13.6 clock reset), not a freely operator-tunable knob.
        live_slippage_ratio_limit_ppm: Ceiling on the live-vs-paper cost slippage
            ratio, in ppm (``1_500_000`` == 1.5x the modeled cost).
        live_brier_degradation_band_ppm: Allowed LIVE-over-PAPER rolling Brier
            degradation, in ppm, before the divergence monitor demotes.
    """

    min_resolved_for_calibration: int = 150
    promotion_min_resolved: int = 300
    promotion_min_independent_event_groups: int = 100
    brier_skill_required_ppm: int = 10000
    bootstrap_confidence_ppm: int = field(
        default=950000,
        metadata={_YAML_KEY: "bootstrap_confidence", _CONVERT: "confidence_to_ppm"},
    )
    observation_window: str = "latest_before_close"
    live_rolling_window_size: int = _DEFAULT_LIVE_ROLLING_WINDOW_SIZE
    live_slippage_ratio_limit_ppm: int = _DEFAULT_LIVE_SLIPPAGE_RATIO_LIMIT_PPM
    live_brier_degradation_band_ppm: int = _DEFAULT_LIVE_BRIER_DEGRADATION_BAND_PPM


@dataclass(frozen=True, slots=True)
class OpsConfig:
    """Operational filesystem, disk-headroom, and shutdown policy."""

    state_dir: str = "~/.local/share/windbreak"
    backup_dir: str = "~/windbreak-backups"
    min_free_disk_mb: int = 1000
    cancel_open_orders_on_shutdown: bool = True


def _default_alert_sinks() -> tuple[AlertSink, ...]:
    """Return the SPEC §16 default single ntfy alert sink."""
    return (AlertSink("ntfy", "configured-by-operator"),)


@dataclass(frozen=True, slots=True)
class AlertsConfig:
    """Operator alert-sink fan-out configuration."""

    sinks: tuple[AlertSink, ...] = field(default_factory=_default_alert_sinks)


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """The loopback dashboard surface's operator-tunable settings (issue #79).

    Backs ``windbreak run --process dashboard`` (SPEC §14). Only the TCP
    ``port`` is configurable: the bind host is *never* a knob -- the dashboard
    accepts no public inbound traffic and always binds ``127.0.0.1`` -- so a
    ``dashboard.host`` key is an unknown key and fatal, the structural guarantee
    that pins loopback-only binding.

    Attributes:
        port: The loopback TCP port the dashboard serves on. Defaults to
            ``8080``, matching the reserved ``127.0.0.1:8080`` compose publish.
    """

    port: int = 8080


@dataclass(frozen=True, slots=True)
class WindbreakConfig:
    """The complete, immutable windbreak configuration (SPEC §16 root).

    Attributes:
        mode_ceiling: The highest operating mode the runtime may ever reach.
    """

    mode_ceiling: str = "paper"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ops: OpsConfig = field(default_factory=OpsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
