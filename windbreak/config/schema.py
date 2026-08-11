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

#: The repo-wide "the operator must fill this in" placeholder. A field still
#: holding it is *unconfigured*, and every consumer treats it as such by failing
#: closed (contributing no allowlist host, building no sink) rather than by
#: inventing a plausible-looking real-world default.
UNCONFIGURED_PLACEHOLDER = "configured-by-operator"

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
class SmtpSinkSettings:
    """SMTP relay settings for an ``smtp``-type :class:`AlertSink` (issue #274).

    The config-schema mirror of
    :class:`windbreak.alerts.sinks.SmtpSinkConfig`. ``host``/``sender`` carry no
    safe real-world default, so they ship as :data:`UNCONFIGURED_PLACEHOLDER`
    and an untouched section builds no SMTP sink at all.

    Unlike :class:`AlertSink`'s ``*_env`` destination fields, every leaf here is
    held in configuration directly, and therefore reaches the ``ConfigLoaded``
    ledger event and ``config_versions.json``. That is deliberate: none of them
    can carry a credential.

    - ``host`` must *also* be declared in ``alerts.allowed_hosts`` for the sink
      to build at all, so the relay hostname is unavoidably in plaintext
      configuration already; hiding it in one field while publishing it in the
      other would be theatre, not protection.
    - ``sender``/``recipients`` are mailbox addresses, not bearer capabilities:
      holding one grants no ability to send or read a windbreak alert.
      :class:`~windbreak.alerts.sinks.SmtpSinkConfig` has no username/password
      field at all -- the relay authenticates this process by network position,
      never by these values -- and ``recipients`` is a list, which
      environment-variable indirection cannot carry without inventing a
      delimiter convention that would itself be a parsing hazard.

    Attributes:
        host: The SMTP relay hostname. Screened against the outbound allowlist
            before a sink is built, exactly like an ntfy/webhook host.
        port: The SMTP relay port; ``587`` is the STARTTLS submission port the
            sink's transport upgrades on.
        sender: The envelope/from address alerts are sent as.
        recipients: The operator addresses alerts are delivered to. Empty means
            unconfigured: an email with no recipient cannot be delivered.
    """

    host: str = UNCONFIGURED_PLACEHOLDER
    port: int = 587
    sender: str = UNCONFIGURED_PLACEHOLDER
    recipients: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AlertSink:
    """A destination for operator alerts.

    One flat entry describes every supported sink type, with only the fields
    that type needs filled in (issue #274 bridged SPEC S16's original
    ``type``/``topic`` pair to the concrete
    ``windbreak.alerts.sinks.*SinkConfig`` shapes). The alternative -- a
    separate schema class per sink type -- cannot be expressed in a single
    homogeneous ``tuple[AlertSink, ...]`` the loader can build, so the union is
    carried here and :func:`windbreak.alerts.factory.build_sinks` reads only the
    fields belonging to ``type``. Every field defaults to
    :data:`UNCONFIGURED_PLACEHOLDER`, so an entry the operator has not finished
    filling in builds no sink instead of dialing a made-up endpoint.

    **No credential-bearing destination is stored here.** Every field below
    names an *environment variable*, exactly as
    :attr:`FutureSearchProviderSettings.api_key_env` and
    :attr:`ResearchSettings.search_api_key_env` do, and
    :func:`windbreak.alerts.factory.build_sinks` resolves it against the process
    environment at composition time. The indirection is structural, not
    cosmetic: a configuration leaf is flattened by
    :func:`windbreak.config.versioning.diff_configs` and persisted verbatim --
    old value *and* new value -- into the hash-chained ``ConfigLoaded`` ledger
    event and then into the plaintext ``config_versions.json`` read model, all
    of which are append-only and cannot be redacted after the fact. An ntfy
    topic is a bearer capability and a webhook URL can embed a token in its
    userinfo, path, or query, so none of them may ever be a config leaf. This
    deviates knowingly from SPEC S16's literal ``topic:`` key
    (``plans/SPEC_v3.md``); the SPEC example's placeholder is preserved as
    ``topic_env``.

    Attributes:
        type: The sink transport: ``ntfy``, ``webhook``, ``smtp``, or
            ``desktop``. Anything else is fatal at sink-construction time. Not a
            secret: it names a code path, not a destination.
        topic_env: The environment variable holding the ntfy topic to publish
            to; never the topic itself, which is a bearer capability. Unused by
            the other types.
        base_url_env: The environment variable holding the ``https://`` ntfy
            server base URL. Held out of configuration because it is a URL, and
            a URL can embed credentials in its userinfo. Unused by other types.
        url_env: The environment variable holding the ``https://`` endpoint a
            ``webhook`` sink POSTs to -- typically the highest-value alert
            secret, since providers routinely encode the whole capability in the
            path. Unused by other types.
        smtp: The relay settings an ``smtp`` sink connects with. Held in
            configuration directly; see :class:`SmtpSinkSettings` for why none
            of its leaves can carry a credential.
    """

    type: str
    topic_env: str = UNCONFIGURED_PLACEHOLDER
    base_url_env: str = UNCONFIGURED_PLACEHOLDER
    url_env: str = UNCONFIGURED_PLACEHOLDER
    smtp: SmtpSinkSettings = field(default_factory=SmtpSinkSettings)


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
        max_pos_total_pct_ppm: The share of worst-case equity the account's
            *total* exposure plus one order's cost may reach, in ppm (issue
            #407). Previously a ``_FULL_PPM`` literal hardcoded into the
            scheduler's limits builder, which is why it appears here with the
            same 100% default rather than a tighter invented one: this change
            moves the policy from a constant no operator could see into a
            declared value they can tighten, and does not choose the firm's
            risk appetite on their behalf. It is a live ceiling either way now
            that ``total_exposure`` is fed real figures -- at 100% it refuses
            to deploy past total equity, where against the former hardcoded
            zero exposure it could never bind at all.
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
    max_pos_total_pct_ppm: int = 1000000


@dataclass(frozen=True, slots=True)
class CorrelationTagConfig:
    """One operator-declared correlation-bucket assignment (SPEC S9.9, #407).

    SPEC S9.9 describes bucket tagging as "LLM-assisted ... human-overridable,
    stored as data". This is the *stored as data* half, and it is the only
    honest producer of a :class:`~windbreak.selector.correlation.CorrelationTag`
    that exists today.

    Deriving a bucket from venue metadata was considered and rejected. A
    correlation bucket is a claim about which markets move *together*; the seed
    taxonomy (``us-election``, ``fed-policy``, ``inflation``, ...) is a risk
    taxonomy, not a venue one. ``NormalizedMarket.category`` is a free-form
    string passed through verbatim from the exchange, whose "Politics" spans US
    elections, foreign elections, and legislation alike. Any table mapping it
    onto a bucket would be this codebase's judgment wearing the venue's name,
    and stamping such a tag with a new ``source`` value would put a provenance
    in the audit trail that no component actually holds. An operator writing a
    bucket here, by contrast, genuinely is the human that
    :data:`~windbreak.selector.correlation.TagSource`'s ``"human"`` names.

    Attributes:
        ticker: The market this assignment applies to.
        bucket_ids: The correlation buckets the market belongs to. Each must be
            a seed-taxonomy id or a ``geopolitics-<region>`` id;
            :class:`~windbreak.selector.correlation.CorrelationTag` refuses
            anything else at construction, so a typo here fails loudly rather
            than silently placing a market in a bucket of its own.
        tagged_at: When the operator made this assignment, as an ISO-8601
            timestamp carrying a UTC offset. Required rather than defaulted to
            the tick's clock: a tag stamped "now" on every tick would claim a
            provenance instant that never happened. An offsetless value is
            refused (issue #397) rather than read as host-local.
    """

    ticker: str
    bucket_ids: tuple[str, ...]
    tagged_at: str


@dataclass(frozen=True, slots=True)
class CorrelationConfig:
    """The operator's declared correlation-bucket assignments (SPEC S9.9, #407).

    Empty by default, and an empty declaration is *not* permissive: a market
    absent from :attr:`tags` is unbucketable, and
    :func:`windbreak.scheduler.exposure.project_exposure` refuses to size it
    rather than letting the per-bucket cap read a fabricated zero exposure.

    Attributes:
        tags: The declared per-market bucket assignments.
    """

    tags: tuple[CorrelationTagConfig, ...] = ()


@dataclass(frozen=True, slots=True)
class ScreenerConfig:
    """Market-screening filters, and the bound on what survives them.

    The four filter leaves are SPEC §16's own. ``max_candidates_per_tick`` is
    windbreak's addition (issue #345): the screen now runs over the venue's whole
    market universe each tick, and this is the ceiling on how many of the markets
    that pass it are actually forecast.

    The bound exists because a candidate is not free. Since issue #399 every
    ensemble vote books real money against
    :class:`ForecastBudget`'s per-forecast and per-day ceilings, so an unbounded
    universe is an unbounded bill whose only stopping condition is a fail-closed
    halt. The default of ``3`` is derived rather than invented: it is
    ``per_day_micros // per_forecast_micros`` at their own defaults
    (``20_000_000 // 6_060_000``), i.e. the most forecasts a *worst-case* day can
    afford at all. A tick therefore cannot plan to spend more than a whole day's
    research budget, and raising either ceiling is what raises this bound --
    never the other way round.

    The derivation is documented here but *stored* as a literal, because a
    computed default would silently follow a ceiling change instead of forcing
    someone to look at it. A test pins the equality, so editing either money
    ceiling without revisiting this bound fails the suite (issue #394).

    Attributes:
        category_blocklist: Topical categories no market may be screened in from.
        min_volume_24h_micros: The trailing-24h volume floor, in micros.
        min_depth_contract_centis: The two-sided book-depth floor, in
            contract-centis.
        horizon_days: The inclusive whole-day resolution-horizon window.
        max_candidates_per_tick: The most screened markets one tick forecasts.
            Must be at least one;
            :func:`windbreak.scheduler.screening.require_candidate_bound` refuses
            a lower value at startup rather than leaving an always-idle loop
            that looks healthy from the outside.
    """

    category_blocklist: tuple[str, ...] = (
        "sports",
        "crypto_price",
        "celebrity",
        "insider_prone",
    )
    min_volume_24h_micros: int = 5000000000
    min_depth_contract_centis: int = 10000
    horizon_days: HorizonDays = field(default_factory=HorizonDays)
    max_candidates_per_tick: int = 3


@dataclass(frozen=True, slots=True)
class ForecastBudget:
    """Per-forecast and per-day research spend caps (micro-dollar units).

    ``per_forecast_micros`` mirrors
    :data:`windbreak.forecast.budget.DEFAULT_PER_FORECAST_BUDGET_MICROS` exactly
    (pinned by test), and that module documents the full rationale. In short: it
    is the cost of the most expensive *correct* triaged PROCEED-path forecast,
    and the intended headroom above that worst case is **zero**::

        per_forecast_micros = stage0_prior_cost                    #    60_000
                            + full_pipeline_research_cost          # 3_000_000
                            + ensemble_size * max_cost_micros      # 3 x 1_000_000
                            = 6_060_000

    It was ``3_000_000`` until issue #269's follow-up 4, which is to say exactly
    the fixed research charge -- leaving *zero* room for the votes themselves,
    so the first live tick with failing providers breached the ceiling and
    produced no forecast at all. That follow-up raised it to ``6_000_000``, but
    derived that figure from ``3_000_000 + 3 x 1_000_000`` and omitted the
    Stage-0 triage charge, which is levied against the same forecast. The
    default was therefore ``60_000`` micros short of its own stated worst case;
    issue #394 closes that gap and pins every term.

    Anything lower guarantees breaches on a healthy run. Anything higher is
    slack the guard cannot see into -- a ceiling above any reachable cost is an
    unfalsifiable guard, which is the defect the raise was meant to fix rather
    than relocate.

    ``per_day_micros`` is deliberately left at ``20_000_000``. Doubling a real
    money cap is an operator's decision, not a wiring change's side effect; the
    effect is simply that a day admits fewer worst-case forecasts. Note that
    since issue #399 *every* vote attempt is priced, successes included, so the
    per-forecast ceiling now binds on the healthy path and not only when
    providers are failing.

    Attributes:
        per_forecast_micros: The ceiling one forecast's research plus provider
            votes may spend, in micros.
        per_day_micros: The ceiling all forecasts in one UTC day may spend, in
            micros.
        max_pages: The maximum pages one forecast's bounded web research fetches.
    """

    per_forecast_micros: int = 6060000
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

    endpoint_url: str = UNCONFIGURED_PLACEHOLDER
    pinned_forecaster_versions: tuple[str, ...] = ("pinned-by-operator",)
    api_key_env: str = "FUTURESEARCH_API_KEY"
    per_call_ceiling_micros: int = 2000000
    reject_on_version_drift: bool = True


#: The shipped ceiling on the *total* bytes the research fetch cache may hold
#: (issue #453): 256 MiB. The cache is an archive of fetched research payloads
#: rooted, in the shipped compose stack, on the same named volume as the
#: hash-chained ledger -- so an unbounded cache is a path to a full ledger
#: volume, which takes the daemon down (#443). 256 MiB is ~128 payloads at the
#: 2 MB ``fetch_max_bytes`` ceiling and many thousands at realistic page
#: sizes, while leaving any sanely-provisioned volume to the audit trail. An
#: operator sizing a small volume should lower it; see docs/RUNBOOK.md.
DEFAULT_RESEARCH_CACHE_MAX_BYTES: int = 268_435_456


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
        cache_max_bytes: The ceiling on the *total* bytes the on-disk research
            fetch cache may hold (issue #453). Distinct from
            ``fetch_max_bytes``, which bounds one payload: without this, N
            bounded fetches per forecast still accumulate without limit for
            the life of the run. Must be a positive byte count; a
            non-positive value is refused at composition rather than
            reinterpreted.
    """

    search_endpoint_url: str = UNCONFIGURED_PLACEHOLDER
    search_api_key_env: str = "RESEARCH_SEARCH_API_KEY"
    allowed_research_hosts: tuple[str, ...] = ()
    fetch_timeout_seconds: int = 30
    fetch_max_bytes: int = 2_000_000
    allowed_content_types: tuple[str, ...] = ("text/html",)
    cache_max_bytes: int = DEFAULT_RESEARCH_CACHE_MAX_BYTES


#: Transport-selection token naming the recorded offline replay cassette. The
#: default (issue #344): CI builds ``WindbreakConfig()``, so an omitted section
#: must never be the one that acquires a live network dependency.
PROVIDER_TRANSPORT_CASSETTE = "cassette"

#: Transport-selection token naming the live pinned-LLM / research transports.
#: Opt-in only, and inert on its own -- the composition root additionally
#: requires the live HTTP seam to be supplied, and refuses to start when this
#: mode is selected without one (see
#: :func:`windbreak.scheduler.loop.build_paper_deps`).
PROVIDER_TRANSPORT_LIVE = "live"


@dataclass(frozen=True, slots=True)
class ProviderRetryConfig:
    """The config-schema mirror of the provider retry policy (issue #269).

    Backs :class:`windbreak.forecast.providers.retry.RetryPolicy`, whose four
    guards bound a hosted provider call: an attempt count, a total wall-clock
    deadline, an exponential-backoff base, and an affordability ceiling. Every
    leaf is a whole integer (milliseconds, micros) -- SPEC S6.1 admits no float
    on this path, and the retry schedule is integer-millisecond by construction
    so it stays deterministic and test-reproducible.

    The first three defaults deliberately equal the engine's own
    ``DEFAULT_MAX_ATTEMPTS`` / ``DEFAULT_TOTAL_DEADLINE_MS`` /
    ``DEFAULT_BACKOFF_BASE_MS``; this module is deliberately dependency-free
    (it imports nothing from ``windbreak``), so the equality is written out here
    and pinned by test rather than imported -- the same mirror idiom
    :func:`_default_vote_ensemble` and :class:`ProviderGateConfig` use.

    Attributes:
        max_attempts: The maximum provider attempts (one initial call plus
            retries).
        total_deadline_ms: The total wall-clock deadline across all attempts,
            in whole milliseconds.
        backoff_base_ms: The exponential-backoff base wait, in whole
            milliseconds; attempt ``n`` waits ``backoff_base_ms << (n - 1)``
            unless the provider supplies an explicit ``Retry-After`` hint.
        max_cost_micros: The affordability ceiling across all attempts of a
            single vote, in micros. Checked *before* every attempt, so an
            unaffordable retry never reaches the provider.
    """

    max_attempts: int = 3
    total_deadline_ms: int = 30_000
    backoff_base_ms: int = 1_000
    max_cost_micros: int = 1_000_000


@dataclass(frozen=True, slots=True)
class ProviderPrice:
    """One provider's per-attempt list price, in micros (issue #269).

    A structural pair backing one entry of
    :attr:`windbreak.forecast.budget.ProviderPriceTable.prices_micros`. Modeled
    as a tuple of records rather than a mapping because every config leaf must
    flatten deterministically into the hash-chained ledger.

    Attributes:
        provider: The provider identifier priced (e.g. ``openai``).
        price_micros: That provider's per-attempt list price, in micros.
    """

    provider: str
    price_micros: int


def _default_provider_prices() -> tuple[ProviderPrice, ...]:
    """Return the default per-attempt price table (issue #269).

    Covers exactly the providers the live composition root can *route* -- the
    two pinned-LLM completion transports. Where it overlaps the forecast
    engine's own ``DEFAULT_PROVIDER_PRICE_TABLE`` the prices are mirror-equal,
    pinned by test.

    ``futuresearch`` is deliberately absent even though the engine's table
    prices it. Its provider is a ``ForecastProvider`` that does its own
    research, not an ``LlmTransport``, so it does not ride the routed
    completion seam this configuration selects; listing it here would advertise
    a live provider the composition root would then refuse at startup. An
    operator who wires it through the ``run_pipeline`` seam directly still gets
    the engine's own list price. The network-free fixture provider is absent for
    a different reason: its true cost is zero and rides on
    ``ProviderForecast.cost_micros`` directly, never through this fail-closed
    (never-zero) list-price table.

    Returns:
        The default per-provider list prices.
    """
    return (
        ProviderPrice("openai", 200_000),
        ProviderPrice("anthropic", 300_000),
    )


@dataclass(frozen=True, slots=True)
class ModelTokenPrice:
    """One model's per-token rates, in micros per million tokens (issue #451).

    A structural triple backing one entry of
    :attr:`windbreak.forecast.budget.ModelRateTable.rates`. Modeled as a tuple
    of records rather than a mapping for the same reason
    :class:`ProviderPrice` is: every config leaf must flatten deterministically
    into the hash-chained ledger.

    Rates are quoted *per million tokens* because that is the only denominator
    that keeps them exact integers on the SPEC S6.1 fixed-point money path --
    ``$1.25 / 1M`` input tokens is ``1.25`` micros per token, which this
    codebase cannot represent, and exactly ``1_250_000`` micros per million.

    Attributes:
        model_version: The pinned model version these rates price.
        input_micros_per_million_tokens: The prompt-token rate, in micros per
            million tokens.
        output_micros_per_million_tokens: The completion-token rate, in micros
            per million tokens.
    """

    model_version: str
    input_micros_per_million_tokens: int
    output_micros_per_million_tokens: int


def _default_model_token_prices() -> tuple[ModelTokenPrice, ...]:
    """Return the default per-model token rates (issue #451).

    Covers exactly the three models :attr:`ForecastConfig.vote_ensemble` pins,
    at their vendors' published list rates. Mirror-equal to the forecast
    engine's own ``DEFAULT_MODEL_RATE_TABLE``, pinned by test -- the same
    arrangement ``prices`` has with ``DEFAULT_PROVIDER_PRICE_TABLE``.

    A model absent from this tuple is not free: it charges
    :attr:`ProviderTransportConfig.unmetered_response_micros`.

    Returns:
        The default per-model token rates.
    """
    return (
        ModelTokenPrice("gpt-5-2025-08-07", 1_250_000, 10_000_000),
        ModelTokenPrice("gpt-5-mini-2025-08-07", 250_000, 2_000_000),
        ModelTokenPrice("claude-sonnet-4-5-20250929", 3_000_000, 15_000_000),
    )


@dataclass(frozen=True, slots=True)
class ProviderTransportConfig:
    """Which forecast provider transport the composition root wires (issue #344).

    The PAPER loop's forecast vote stage runs against either the recorded
    offline replay cassette or the live pinned-LLM / research transports.
    :attr:`mode` chooses, and it defaults to
    :data:`PROVIDER_TRANSPORT_CASSETTE` so that CI -- which builds
    ``WindbreakConfig()`` -- stays offline and byte-deterministic by omission.
    Going live is an explicit written act, never a default.

    The two ``*_api_key_env`` leaves name *environment variables*, never keys.
    Every config leaf is flattened verbatim into the hash-chained ledger via
    ``diff_configs`` -> ``ConfigLoaded``, so a secret placed in configuration is
    a secret in the audit trail permanently; this follows
    :attr:`FutureSearchProviderSettings.api_key_env` and
    :attr:`ResearchSettings.search_api_key_env`. Reading the named variable is
    the CLI composition root's job -- neither this module nor the forecast
    engine ever touches the process environment.

    Attributes:
        mode: :data:`PROVIDER_TRANSPORT_CASSETTE` (the default) or
            :data:`PROVIDER_TRANSPORT_LIVE`.
        request_timeout_seconds: The per-request dial timeout, in whole seconds.
        anthropic_api_key_env: The environment variable the Anthropic key is
            read from -- the variable's *name*, never the key.
        openai_api_key_env: The environment variable the OpenAI key is read
            from -- the variable's *name*, never the key.
        retry: The bounded-retry policy each provider vote runs under.
        prices: The per-provider per-attempt list prices, in micros. Since issue
            #451 these are the *affordability estimate* the retry layer gates an
            attempt on before making it, not what a completed vote is charged --
            a call's real cost is unknowable until it returns.
        unknown_provider_price_micros: The fallback price charged for a provider
            absent from :attr:`prices`. Deliberately high, never zero, so an
            unpriced provider cannot evade its budget.
        token_prices: The per-model token rates a completed vote's reported
            usage is charged at (issue #451). This is what makes the budget's
            ceilings bound *spend* rather than a count of attempts multiplied
            by a constant.
        unmetered_response_micros: The fail-closed charge for a response whose
            cost cannot be derived -- it reported no readable usage, or came
            from a model absent from :attr:`token_prices`. Deliberately high,
            never zero, and deliberately unequal to any per-attempt list price,
            so an unmeasurable response can neither read as free nor be
            mistaken for a metered charge.
    """

    mode: str = PROVIDER_TRANSPORT_CASSETTE
    request_timeout_seconds: int = 30
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    openai_api_key_env: str = "OPENAI_API_KEY"
    retry: ProviderRetryConfig = field(default_factory=ProviderRetryConfig)
    prices: tuple[ProviderPrice, ...] = field(default_factory=_default_provider_prices)
    unknown_provider_price_micros: int = 1_000_000
    token_prices: tuple[ModelTokenPrice, ...] = field(
        default_factory=_default_model_token_prices
    )
    unmetered_response_micros: int = 1_000_000


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
    provider_transport: ProviderTransportConfig = field(
        default_factory=ProviderTransportConfig
    )


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
    """Return the SPEC §16 default single, deliberately unconfigured ntfy sink."""
    return (AlertSink("ntfy", UNCONFIGURED_PLACEHOLDER),)


@dataclass(frozen=True, slots=True)
class AlertsConfig:
    """Operator alert-sink fan-out configuration.

    Attributes:
        sinks: The destinations each alert fans out to. The SPEC §16 default is
            one placeholder ntfy sink, which builds nothing until an operator
            fills in its ``base_url_env``/``topic_env`` and exports the
            variables they name.
        allowed_hosts: The alert-destination hosts this deployment may dial,
            unioned into :func:`windbreak.net.allowlist.allowlist_from_config`'s
            egress allowlist. Deliberately *separate* from the per-sink
            destination fields (issue #274): if the allowlist were derived from
            the sink URLs themselves it could never veto one of them, and a
            check that cannot fail is worse than no check. Requiring the host to
            be declared twice, in two independent fields, means a single
            mistyped or tampered destination cannot open egress on its own.
            Empty by default, so an undeclared deployment reaches nothing.
    """

    sinks: tuple[AlertSink, ...] = field(default_factory=_default_alert_sinks)
    allowed_hosts: tuple[str, ...] = ()


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
        correlation: The operator's declared correlation-bucket assignments
            (SPEC S9.9, issue #407). Empty by default, which refuses to size
            any market rather than sizing every market against an empty bucket.
    """

    mode_ceiling: str = "paper"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ops: OpsConfig = field(default_factory=OpsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
