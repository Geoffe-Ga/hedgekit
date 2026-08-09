# Evaluation

This document describes windbreak's evaluation and calibration methodology as
specified in SPEC §13: three separate tracks that are never merged into one
vanity number, precommitted baselines and observation windows, a clustered
bootstrap, a pre-registered gate plan hashed at PAPER entry, and a documented
power analysis — all implemented in `windbreak/evaluation/`.

## Three separate tracks (SPEC §13.1)

Evaluation never merges these into a single headline number:

- **Forecast quality** — did probabilities beat baselines?
- **Selection quality** — did traded forecasts outperform skipped ones?
- **Execution quality** — did fills match modeled prices, fees, and slippage?

The default expectation is plain: **no edge is the default expectation.**
Discovering that and stopping at paper trading is a success state, not a
failure — measurements outrank narratives (SPEC §3.7): if the evaluation
harness says "no edge," the harness wins, not the operator's read of the
rationale text.

## Baselines (SPEC §13.2)

The primary baseline is the **executable market price at the forecast's
baseline snapshot** — if the forecast can't beat the crowd's own price at the
same instant, no edge exists by construction. Secondary baselines (midpoint,
uniform 50%, base-rate where known, previous forecast) are also computed for
context. `windbreak/evaluation/baselines.py` derives all of these from a
forecast's own referenced quote snapshot using only exact integer arithmetic
(a pip is exactly 100 ppm; no division, no rounding decision, no float).

## Precommitted observation windows (SPEC §13.4)

Multiple forecasts per market are handled by declared windows (first-per-market,
latest-before-close, daily snapshots, trade-triggering); mixing windows in one
metric is a test failure, not a stylistic choice. `config.evaluation.observation_window`
names the window the headline Brier metric uses
(`windbreak/evaluation/windows.py`).

## Clustered bootstrap (SPEC §13.5)

Confidence intervals are computed by a bootstrap **clustered by
event/correlation group**, so related markets never masquerade as independent
observations (`windbreak/evaluation/bootstrap.py`). The confidence level is
`config.evaluation.bootstrap_confidence_ppm`, expressed as an integer
parts-per-million (never a raw float) — the loader converts the SPEC §16 YAML
`bootstrap_confidence: 0.95` into this field at load time. A companion power
analysis (`windbreak/evaluation/power.py`) states the minimum detectable Brier
skill at N=300 with observed clustering, so an underpowered pass is never
mistaken for proof.

## Pre-registration (SPEC §13.6, anti-Goodhart)

At PAPER entry, the complete gate plan — every metric, window, threshold,
baseline, and clustering scheme — is canonically serialized, SHA-256 hashed,
and ledgered (`windbreak/evaluation/preregistration.py`). A byte-identical
re-registration is idempotent; any actual change to the plan resets the PAPER
evaluation clock. Promotion thresholds referenced by the plan include
`config.evaluation.promotion_min_resolved`,
`config.evaluation.promotion_min_independent_event_groups`, and
`config.evaluation.brier_skill_required_ppm`.

## Temporal integrity (SPEC §1.1-6)

Only forecasts made in real time, on then-unresolved questions, ever count
toward a gate (`windbreak/evaluation/temporal.py`): a forecast is rejected at
ingestion if it predates deployment, was created at or after its market's
resolution, or its market never resolved at all. This guards against an LLM
recalling a resolved outcome from its training data.

## Live-vs-paper divergence (SPEC §10.9, §10.10)

Once a run has both LIVE forecasts and LIVE execution records,
`windbreak.evaluation.live_divergence.monitor_live_divergence` scores two
series against the pre-registered plan's thresholds:
`config.evaluation.live_slippage_ratio_limit_ppm` (cost slippage of real fills
vs. the paper model) and `config.evaluation.live_brier_degradation_band_ppm`
(rolling LIVE-over-PAPER forecast-skill decay), each scored over the most
recent `config.evaluation.live_rolling_window_size` resolved records. A breach
fires a critical alert and an automatic one-rung demotion.

**Known limitation.** `monitor_live_divergence` exists and is fully tested, but
it is not yet wired into any scheduler — nothing calls it automatically on a
cadence today. Tracked in issue #200, alongside the rest of the live-divergence
fast-follow hardening.

## Provider track-record gate (SPEC §13/§16, §19, §9.6)

Live eligibility is *earned* per provider, never granted by default. A voting
provider's forecasts are forced live-ineligible (`eligible_for_live=False`)
until that provider itself has accumulated at least
`config.forecast.provider_gate.min_resolved` (default 150) resolved paper
forecasts with a Brier skill over the executable-market baseline of at least
`config.forecast.provider_gate.min_brier_skill_ppm` (default 10000 ppm) — the
same statistical bar `config.evaluation.min_resolved_for_calibration` and
`config.evaluation.brier_skill_required_ppm` set for calibration and ensemble
promotion, applied here per provider rather than to the ensemble as a whole
(`windbreak.forecast.providers.track_record`).

An unproven provider's votes still **run** and are **recorded** in paper —
that is how its track record accrues in the first place — only its live
eligibility is withheld; the decision is ledgered as a `PROVIDER_GATE_HELD`
forecast event naming every unproven provider. The gate is a **read model**
over M6's evaluation artifacts: it consumes each provider's resolved-count
and Brier-skill figures and never recomputes a score itself, so it is only
ever as fresh as the last evaluation pass. A provider with no track record at
all — or a record below either bar — is treated as unproven; the gate fails
*closed* by construction, in keeping with the plain expectation that no
unmeasured edge ever backs a live order (SPEC §19). Because the dispersion
that SPEC §9.6 sizes against is deliberately provider-family-agnostic (see
`windbreak.forecast.ensemble`), gating a provider's *live eligibility*
without excluding its *vote* from the ensemble keeps that honest
disagreement signal intact even while the provider itself remains unproven.

## Versioned calibration map (SPEC §8.2)

`windbreak.forecast.calibration` loads a versioned probability-calibration
map applied at the pipeline's calibration stage (stage 11 of SPEC §8.2): a
deterministic integer piecewise-linear correction from a raw aggregate
probability toward the frequency its historical forecasts actually resolved
at. Until M6 fits a real map from resolved paper forecasts, every run applies
the `"v0"` identity map, which corrects nothing — the byte-identical behavior
of every run before this mechanism existed. The applied map's `map_id` and
`version` are ledgered as a `CALIBRATION_MAP_APPLIED` forecast event
alongside the exact pre-/post-calibration ppm, so which map (if any) touched
a given forecast is always reconstructable from the audit trail.

Temporal integrity applies to a calibration map exactly as it does to a
forecast's own creation date (see above): loading a map whose version — an
ISO-8601 training date — postdates the forecast's own `created_at` raises
`TemporalIntegrityError` and fails the whole run closed. A map fitted *after*
a forecast was made could only have learned from outcomes that forecast could
not yet have known about; letting it calibrate that forecast anyway would let
the future leak into the past exactly as a training-cutoff violation would.
