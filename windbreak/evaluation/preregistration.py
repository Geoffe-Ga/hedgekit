"""Pre-registered evaluation gate plans (SPEC §13.6 / §17.4 / T15).

A :class:`GatePlan` is a frozen, content-addressed snapshot of the evaluation
gate's full configuration: the metric/window catalogue, the eight
promotion/calibration thresholds, the observation window, the two named schemes
(executable-price baseline, event-correlation clustering), and the paper
fill-model version. Every value is an ``int``, a ``str``, or a tuple of those --
never a float -- so a plan hashes to a stable :attr:`GatePlan.plan_hash`
(SHA-256 of its canonical JSON) that is independent of metric-window input order.

:func:`register_gate_plan` writes a plan into the append-only ledger, resetting
the paper clock only when the plan's identity actually changes:

- the first registration ledgers a :class:`GatePlanRegistered`;
- a byte-identical re-registration is idempotent (no new event, clock unchanged);
- any change ledgers a :class:`GatePlanChanged` carrying the ``previous_plan_hash``
  and a strictly-later ``paper_clock_start`` -- failing closed (raising, appending
  nothing) if the injected clock is not strictly monotonic.

:func:`latest_gate_plan_registration` reconstructs the most recent registration
from the ledger alone. Because the promotion path calls it repeatedly, it reads
through the optional :class:`~windbreak.ledger.store.LatestRecordLookup`
capability when the store declares one (an index-backed reverse read) and only
falls back to a full ledger scan otherwise (issue #246).

Dependency direction. This module is a runtime *leaf consumer*: it holds
one-way runtime edges to :mod:`windbreak.evaluation.registry` (for
:func:`~windbreak.evaluation.registry.registered_metrics`),
:mod:`windbreak.evaluation.windows` (via each metric spec's window), and
:mod:`windbreak.ledger` (events and store). It references
:class:`~windbreak.config.schema.EvaluationConfig` only under
:data:`typing.TYPE_CHECKING`, and :mod:`windbreak.evaluation.registry` never
imports this module, so the graph stays acyclic.

Event naming. :class:`GatePlanRegistered` and :class:`GatePlanChanged` derive
their ``event_type`` from the concrete class name (the house convention for
every ledger event -- never a shouty snake-case variant; the issue's uppercase
names are pseudocode). Both event classes now live in
:mod:`windbreak.ledger.events` and are registered in its central ``EVENT_TYPES``
reconstruction map (issue #180); this module imports them from there, keeping the
evaluation package a one-way, acyclic runtime consumer of the ledger. Local
reconstruction still round-trips through :meth:`GatePlan.from_canonical` (which
strips the registration-only keys and re-validates the plan) rather than that
map.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from windbreak.evaluation.registry import (
    LIVE_ROLLING_WINDOW_SIZE,
    registered_metrics,
)
from windbreak.ledger.events import (
    GatePlanChanged as GatePlanChanged,  # re-exported (issue #180)
)
from windbreak.ledger.events import (
    GatePlanRegistered as GatePlanRegistered,  # re-exported (issue #180)
)
from windbreak.ledger.events import (
    canonical_json,
)
from windbreak.ledger.store import LatestRecordLookup

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from windbreak.config.schema import EvaluationConfig
    from windbreak.ledger.store import LedgerRecord, LedgerStore

#: The named baseline scheme the headline skill metric measures against: the
#: executable price captured at the baseline snapshot (SPEC §13.6).
EXECUTABLE_PRICE_BASELINE_SCHEME = "executable_price_at_baseline_snapshot"

#: The named clustering scheme the bootstrap resamples over: independent
#: event-correlation groups (SPEC §13.6).
CORRELATION_GROUP_CLUSTERING_SCHEME = "event_correlation_group"

#: The canonical-dict key carrying the metric/window catalogue.
_METRIC_WINDOWS_KEY = "metric_windows"

#: The number of elements in one ``metric_windows`` entry: a ``(name, window)``
#: pair.
_METRIC_WINDOW_PAIR_LEN = 2

#: The eight integer threshold fields, in declaration order. Iterated by the
#: construction guard and the canonical round-trip so the field list is stated
#: once.
_INT_FIELD_NAMES: tuple[str, ...] = (
    "min_resolved_for_calibration",
    "promotion_min_resolved",
    "promotion_min_independent_event_groups",
    "brier_skill_required_ppm",
    "bootstrap_confidence_ppm",
    "live_rolling_window_size",
    "live_slippage_ratio_limit_ppm",
    "live_brier_degradation_band_ppm",
)

#: The three live-threshold field names issue #58 added, carved out of
#: :func:`GatePlan.from_canonical`'s strict key check so a pre-#58, 10-key
#: registration (which predates these keys) still reads back with documented
#: legacy defaults -- content addressing stays schema-independent because the
#: stored hash is verified against the *persisted* plan dict, not a
#: newly-serialized one. Genuinely unknown keys are still rejected.
_LIVE_THRESHOLD_FIELD_NAMES: tuple[str, ...] = (
    "live_rolling_window_size",
    "live_slippage_ratio_limit_ppm",
    "live_brier_degradation_band_ppm",
)

#: Documented legacy defaults for the three new live-threshold keys, applied
#: when reading a pre-#58 registration that recorded none of them. A system
#: with no live thresholds recorded before is assumed to want today's standard
#: defaults applied retroactively (matching
#: :class:`~windbreak.config.schema.EvaluationConfig`'s own defaults).
_LEGACY_LIVE_THRESHOLD_DEFAULTS: Mapping[str, int] = {
    "live_rolling_window_size": 100,
    "live_slippage_ratio_limit_ppm": 1_500_000,
    "live_brier_degradation_band_ppm": 50_000,
}

#: The four string identity fields, in declaration order.
_STR_FIELD_NAMES: tuple[str, ...] = (
    "observation_window",
    "baseline_scheme",
    "clustering_scheme",
    "paper_fill_model_version",
)

#: The exact set of keys a canonical plan dict carries -- the metric catalogue,
#: the eight thresholds, and the four identity strings.
_CANONICAL_PLAN_KEYS: frozenset[str] = frozenset(
    (_METRIC_WINDOWS_KEY, *_INT_FIELD_NAMES, *_STR_FIELD_NAMES)
)

#: The registration-only keys the events add on top of the canonical plan dict,
#: stripped back off when reconstructing a plan from the ledger.
_REGISTRATION_ONLY_KEYS: frozenset[str] = frozenset(
    {"plan_hash", "paper_clock_start", "previous_plan_hash"}
)

#: The ``event_type`` discriminators a gate-plan registration can be recorded
#: under.
_REGISTRATION_EVENT_TYPES: frozenset[str] = frozenset(
    {"GatePlanRegistered", "GatePlanChanged"}
)


def _require_non_bool_int(value: object, field_name: str) -> None:
    """Reject a value that is not a non-``bool`` ``int``.

    Args:
        value: The value to check.
        field_name: The field the value belongs to, named in the error.

    Raises:
        TypeError: If ``value`` is a ``bool`` (an ``int`` subclass that must not
            masquerade as a number) or is not an ``int`` at all.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} requires a non-bool int, got {type(value).__name__}"
        )


def _require_str(value: object, field_name: str) -> None:
    """Reject a value that is not a ``str``.

    Args:
        value: The value to check.
        field_name: The field the value belongs to, named in the error.

    Raises:
        TypeError: If ``value`` is not a ``str``.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} requires a str, got {type(value).__name__}")


def _normalize_metric_windows(
    metric_windows: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Reject duplicate metric names and return the pairs sorted by name.

    Sorting makes the plan's identity independent of metric-window input order,
    so two plans built from the same metric set in different orders hash
    identically.

    Args:
        metric_windows: The ``(name, window)`` pairs to normalize.

    Returns:
        The pairs as a tuple, sorted (metric names being unique) by name.

    Raises:
        ValueError: If two pairs name the same metric, which would make the
            plan's per-metric window ambiguous; the message names the metric.
    """
    seen: set[str] = set()
    for name, _window in metric_windows:
        if name in seen:
            raise ValueError(f"duplicate metric name in gate plan: {name!r}")
        seen.add(name)
    return tuple(sorted(metric_windows))


@dataclass(frozen=True, slots=True)
class GatePlan:
    """An immutable, content-addressed snapshot of the evaluation gate config.

    Attributes:
        metric_windows: The ``(metric_name, window_value)`` catalogue, normalized
            to name-sorted order so identity is input-order independent.
        min_resolved_for_calibration: Minimum resolved forecasts before
            calibration statistics are computed.
        promotion_min_resolved: Minimum resolved forecasts required to promote.
        promotion_min_independent_event_groups: Minimum independent event
            groups required to promote.
        brier_skill_required_ppm: Required Brier skill score, in ppm.
        bootstrap_confidence_ppm: Bootstrap confidence level, in ppm.
        observation_window: The headline observation window value.
        baseline_scheme: The named executable-price baseline scheme.
        clustering_scheme: The named event-correlation clustering scheme.
        paper_fill_model_version: The paper fill-model version pinned into the
            plan's identity (SPEC §17.4).
        live_rolling_window_size: Rolling-window size for the live-divergence
            gates (issue #58); defaulted so a pre-#58 direct construction stays
            valid.
        live_slippage_ratio_limit_ppm: Live-vs-paper slippage ratio ceiling, in
            ppm (issue #58).
        live_brier_degradation_band_ppm: Allowed LIVE-over-PAPER rolling Brier
            degradation, in ppm (issue #58).
    """

    metric_windows: tuple[tuple[str, str], ...]
    min_resolved_for_calibration: int
    promotion_min_resolved: int
    promotion_min_independent_event_groups: int
    brier_skill_required_ppm: int
    bootstrap_confidence_ppm: int
    observation_window: str
    baseline_scheme: str
    clustering_scheme: str
    paper_fill_model_version: str
    live_rolling_window_size: int = _LEGACY_LIVE_THRESHOLD_DEFAULTS[
        "live_rolling_window_size"
    ]
    live_slippage_ratio_limit_ppm: int = _LEGACY_LIVE_THRESHOLD_DEFAULTS[
        "live_slippage_ratio_limit_ppm"
    ]
    live_brier_degradation_band_ppm: int = _LEGACY_LIVE_THRESHOLD_DEFAULTS[
        "live_brier_degradation_band_ppm"
    ]

    def __post_init__(self) -> None:
        """Validate every field and normalize ``metric_windows`` in place.

        Raises:
            TypeError: If any threshold is a ``bool`` or non-``int``, or any
                identity string is not a ``str``; the message names the field.
            ValueError: If two ``metric_windows`` entries name the same metric.
        """
        for name in _INT_FIELD_NAMES:
            _require_non_bool_int(getattr(self, name), name)
        for name in _STR_FIELD_NAMES:
            _require_str(getattr(self, name), name)
        normalized = _normalize_metric_windows(self.metric_windows)
        object.__setattr__(self, _METRIC_WINDOWS_KEY, normalized)

    def canonical_dict(self) -> dict[str, object]:
        """Return the plan as a JSON-safe dict of exactly the thirteen plan keys.

        ``metric_windows`` is rendered as a list of two-element lists (JSON has
        no tuple), matching the persisted form; key order is irrelevant because
        :func:`~windbreak.ledger.events.canonical_json` sorts keys.

        Returns:
            A mapping carrying the metric catalogue, the eight thresholds, and
            the four identity strings.
        """
        windows = [[name, window] for name, window in self.metric_windows]
        return {
            _METRIC_WINDOWS_KEY: windows,
            "min_resolved_for_calibration": self.min_resolved_for_calibration,
            "promotion_min_resolved": self.promotion_min_resolved,
            "promotion_min_independent_event_groups": (
                self.promotion_min_independent_event_groups
            ),
            "brier_skill_required_ppm": self.brier_skill_required_ppm,
            "bootstrap_confidence_ppm": self.bootstrap_confidence_ppm,
            "observation_window": self.observation_window,
            "baseline_scheme": self.baseline_scheme,
            "clustering_scheme": self.clustering_scheme,
            "paper_fill_model_version": self.paper_fill_model_version,
            "live_rolling_window_size": self.live_rolling_window_size,
            "live_slippage_ratio_limit_ppm": self.live_slippage_ratio_limit_ppm,
            "live_brier_degradation_band_ppm": self.live_brier_degradation_band_ppm,
        }

    @property
    def canonical_json_str(self) -> str:
        """Return the plan's canonical, sorted-keys, whitespace-free JSON.

        Returns:
            The :func:`~windbreak.ledger.events.canonical_json` encoding of
            :meth:`canonical_dict`.
        """
        return canonical_json(self.canonical_dict())

    @property
    def plan_hash(self) -> str:
        """Return the plan's content hash.

        Returns:
            The 64-character lowercase hex SHA-256 digest of
            :attr:`canonical_json_str`.
        """
        return hashlib.sha256(self.canonical_json_str.encode("utf-8")).hexdigest()

    @classmethod
    def from_canonical(cls, mapping: Mapping[str, object]) -> GatePlan:
        """Reconstruct a :class:`GatePlan` from a canonical plan mapping.

        The mapping must carry the thirteen canonical plan keys; a pre-#58
        ten-key registration is also accepted, with the three live-threshold
        keys defaulting per :data:`_LEGACY_LIVE_THRESHOLD_DEFAULTS`. The
        reconstructed plan's ``__post_init__`` re-validates and re-normalizes.

        Args:
            mapping: A canonical plan dict, e.g. one read back from the ledger.

        Returns:
            The reconstructed plan.

        Raises:
            ValueError: If the mapping carries any key outside the thirteen
                canonical plan keys (the message names the offending key(s)), or
                is missing a required key.
            TypeError: If a value has the wrong JSON type for its field.
        """
        unknown = set(mapping) - _CANONICAL_PLAN_KEYS
        if unknown:
            raise ValueError(f"unknown gate plan key(s): {sorted(unknown)}")
        return cls(
            metric_windows=_parse_metric_windows(mapping[_METRIC_WINDOWS_KEY]),
            min_resolved_for_calibration=_require_mapping_int(
                mapping, "min_resolved_for_calibration"
            ),
            promotion_min_resolved=_require_mapping_int(
                mapping, "promotion_min_resolved"
            ),
            promotion_min_independent_event_groups=_require_mapping_int(
                mapping, "promotion_min_independent_event_groups"
            ),
            brier_skill_required_ppm=_require_mapping_int(
                mapping, "brier_skill_required_ppm"
            ),
            bootstrap_confidence_ppm=_require_mapping_int(
                mapping, "bootstrap_confidence_ppm"
            ),
            observation_window=_require_mapping_str(mapping, "observation_window"),
            baseline_scheme=_require_mapping_str(mapping, "baseline_scheme"),
            clustering_scheme=_require_mapping_str(mapping, "clustering_scheme"),
            paper_fill_model_version=_require_mapping_str(
                mapping, "paper_fill_model_version"
            ),
            live_rolling_window_size=_mapping_int_or_legacy_default(
                mapping, "live_rolling_window_size"
            ),
            live_slippage_ratio_limit_ppm=_mapping_int_or_legacy_default(
                mapping, "live_slippage_ratio_limit_ppm"
            ),
            live_brier_degradation_band_ppm=_mapping_int_or_legacy_default(
                mapping, "live_brier_degradation_band_ppm"
            ),
        )


def _parse_metric_window_entry(entry: object) -> tuple[str, str]:
    """Parse one JSON ``metric_windows`` entry into a ``(name, window)`` pair.

    Args:
        entry: A JSON value expected to be a two-element ``[name, window]`` list.

    Returns:
        The entry as a two-string tuple.

    Raises:
        TypeError: If the entry is not a two-element list of two strings.
    """
    if not isinstance(entry, list) or len(entry) != _METRIC_WINDOW_PAIR_LEN:
        raise TypeError(
            "metric_windows entry requires a 2-element list, "
            f"got {type(entry).__name__}"
        )
    name, window = entry
    if not isinstance(name, str) or not isinstance(window, str):
        raise TypeError("metric_windows entry requires two strings")
    return (name, window)


def _parse_metric_windows(raw: object) -> tuple[tuple[str, str], ...]:
    """Parse a JSON ``metric_windows`` value into a tuple of string pairs.

    Args:
        raw: A JSON value expected to be a list of ``[name, window]`` lists.

    Returns:
        The catalogue as a tuple of two-string tuples.

    Raises:
        TypeError: If ``raw`` is not a list, or any entry is malformed.
    """
    if not isinstance(raw, list):
        raise TypeError(f"metric_windows requires a list, got {type(raw).__name__}")
    return tuple(_parse_metric_window_entry(entry) for entry in raw)


def _require_present(mapping: Mapping[str, object], key: str) -> object:
    """Return ``mapping[key]``, failing closed with a clear error if absent.

    Args:
        mapping: The mapping to read from.
        key: The required key.

    Returns:
        The value stored at ``key``.

    Raises:
        ValueError: If ``key`` is missing -- a malformed (e.g. truncated or
            adversarial) payload, treated as a fatal read error rather than a
            bare ``KeyError``.
    """
    if key not in mapping:
        raise ValueError(f"required gate plan key is missing: {key!r}")
    return mapping[key]


def _require_mapping_int(mapping: Mapping[str, object], key: str) -> int:
    """Return ``mapping[key]`` narrowed to a non-``bool`` ``int``.

    Args:
        mapping: The mapping to read from.
        key: The key to read.

    Returns:
        The value at ``key`` as an ``int``.

    Raises:
        ValueError: If ``key`` is missing.
        TypeError: If the value is a ``bool`` or not an ``int``.
    """
    value = _require_present(mapping, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} requires a non-bool int, got {type(value).__name__}")
    return value


def _mapping_int_or_legacy_default(mapping: Mapping[str, object], key: str) -> int:
    """Return a live-threshold int, defaulting a pre-#58 record's absent key.

    The three live-threshold keys did not exist before issue #58, so a legacy
    registration carries none of them. When absent, the documented legacy
    default in :data:`_LEGACY_LIVE_THRESHOLD_DEFAULTS` is applied; when present,
    the value is read and int-validated exactly like every other threshold.

    Args:
        mapping: The canonical plan mapping to read from.
        key: One of the three live-threshold keys.

    Returns:
        The stored value, or the documented legacy default when the key is
        absent.

    Raises:
        TypeError: If the (present) value is a ``bool`` or not an ``int``.
    """
    if key not in mapping:
        return _LEGACY_LIVE_THRESHOLD_DEFAULTS[key]
    return _require_mapping_int(mapping, key)


def _require_mapping_str(mapping: Mapping[str, object], key: str) -> str:
    """Return ``mapping[key]`` narrowed to a ``str``.

    Args:
        mapping: The mapping to read from.
        key: The key to read.

    Returns:
        The value at ``key`` as a ``str``.

    Raises:
        ValueError: If ``key`` is missing.
        TypeError: If the value is not a ``str``.
    """
    value = _require_present(mapping, key)
    if not isinstance(value, str):
        raise TypeError(f"{key} requires a str, got {type(value).__name__}")
    return value


def build_gate_plan(
    evaluation: EvaluationConfig,
    *,
    paper_fill_model_version: str,
    baseline_scheme: str = EXECUTABLE_PRICE_BASELINE_SCHEME,
    clustering_scheme: str = CORRELATION_GROUP_CLUSTERING_SCHEME,
) -> GatePlan:
    """Build a :class:`GatePlan` from an evaluation config and the live registry.

    The metric catalogue is derived from
    :func:`~windbreak.evaluation.registry.registered_metrics` (name-sorted), and
    the eight thresholds and observation window are copied off ``evaluation``.

    The live rolling-window size is *pinned* to
    :data:`~windbreak.evaluation.registry.LIVE_ROLLING_WINDOW_SIZE`, the constant
    the Python reference path truncates its cohort to. The SQL path binds the
    plan's ``live_rolling_window_size`` into its ``LIMIT``, so if a config drifts
    away from the reference constant the two dual-paths would score different
    windows and the ±1 crosscheck would raise a spurious CRITICAL mismatch. Rather
    than let that drift through silently (and misreport the applied window in the
    ledgered snapshot), this boundary fails closed: a config window that differs
    from the reference constant raises. Changing the window is therefore a code
    change to that constant, which re-registers the pre-registered gate plan
    (§13.6 clock reset), keeping the plan's identity hash-committed.

    Args:
        evaluation: The evaluation configuration to snapshot thresholds from.
        paper_fill_model_version: The paper fill-model version to pin (SPEC §17.4).
        baseline_scheme: The named baseline scheme; defaults to the
            executable-price scheme.
        clustering_scheme: The named clustering scheme; defaults to the
            event-correlation scheme.

    Returns:
        The assembled gate plan.

    Raises:
        ValueError: If ``evaluation.live_rolling_window_size`` differs from
            :data:`~windbreak.evaluation.registry.LIVE_ROLLING_WINDOW_SIZE`; the
            window is pinned to the reference-path constant so both dual-paths
            agree exactly and the pre-registered gate definition stays
            hash-committed.
    """
    if evaluation.live_rolling_window_size != LIVE_ROLLING_WINDOW_SIZE:
        raise ValueError(
            "live_rolling_window_size is pinned to the reference-path constant "
            f"LIVE_ROLLING_WINDOW_SIZE ({LIVE_ROLLING_WINDOW_SIZE}); config "
            f"requested {evaluation.live_rolling_window_size}. The Python "
            "reference path truncates to the constant while the SQL path binds "
            "the plan's window into its LIMIT, so a divergent window would make "
            "the dual-paths disagree and the pre-registered gate plan hash "
            "drift. Change the window by editing the constant (a code change "
            "that re-registers the gate plan, §13.6 clock reset), not the config."
        )
    metric_windows = tuple(
        sorted((name, spec.window.value) for name, spec in registered_metrics().items())
    )
    return GatePlan(
        metric_windows=metric_windows,
        min_resolved_for_calibration=evaluation.min_resolved_for_calibration,
        promotion_min_resolved=evaluation.promotion_min_resolved,
        promotion_min_independent_event_groups=(
            evaluation.promotion_min_independent_event_groups
        ),
        brier_skill_required_ppm=evaluation.brier_skill_required_ppm,
        bootstrap_confidence_ppm=evaluation.bootstrap_confidence_ppm,
        observation_window=evaluation.observation_window,
        baseline_scheme=baseline_scheme,
        clustering_scheme=clustering_scheme,
        paper_fill_model_version=paper_fill_model_version,
        live_rolling_window_size=evaluation.live_rolling_window_size,
        live_slippage_ratio_limit_ppm=evaluation.live_slippage_ratio_limit_ppm,
        live_brier_degradation_band_ppm=evaluation.live_brier_degradation_band_ppm,
    )


@dataclass(frozen=True, slots=True)
class GatePlanRegistration:
    """A reconstructed view of one gate-plan registration.

    Attributes:
        plan: The registered plan.
        plan_hash: The registered plan's content hash.
        paper_clock_start: The whole-epoch-second paper-clock start for the plan.
        previous_plan_hash: The prior plan's hash on a change, or ``None`` for a
            first registration.
        event_type: The ledger event type the registration was recorded under.
    """

    plan: GatePlan
    plan_hash: str
    paper_clock_start: int
    previous_plan_hash: str | None
    event_type: str


def _epoch_now() -> int:
    """Return the current wall-clock time in whole epoch seconds.

    Returns:
        ``int(time.time())``.
    """
    return int(time.time())


class _FlatPlanFields(TypedDict):
    """The thirteen flat gate-plan fields both registration events carry.

    Mirrors :meth:`GatePlan.canonical_dict` as a precisely-typed mapping so it
    unpacks straight into the flat
    :class:`~windbreak.ledger.events.GatePlanRegistered` /
    :class:`~windbreak.ledger.events.GatePlanChanged` constructors under
    ``mypy --strict`` -- which rejects unpacking an untyped ``dict[str, object]``
    (as :meth:`GatePlan.canonical_dict` returns) into typed ``int``/``str``
    parameters.

    Attributes:
        metric_windows: The ``[name, window]`` metric/window catalogue, as the
            JSON list-of-two-element-lists form.
        min_resolved_for_calibration: Minimum resolved forecasts before
            calibration statistics are computed.
        promotion_min_resolved: Minimum resolved forecasts required to promote.
        promotion_min_independent_event_groups: Minimum independent event groups
            required to promote.
        brier_skill_required_ppm: Required Brier skill score, in ppm.
        bootstrap_confidence_ppm: Bootstrap confidence level, in ppm.
        live_rolling_window_size: Rolling-window size for the live-divergence
            gates.
        live_slippage_ratio_limit_ppm: Live-vs-paper slippage ratio ceiling, in
            ppm.
        live_brier_degradation_band_ppm: Allowed LIVE-over-PAPER rolling Brier
            degradation, in ppm.
        observation_window: The headline observation window value.
        baseline_scheme: The named executable-price baseline scheme.
        clustering_scheme: The named event-correlation clustering scheme.
        paper_fill_model_version: The pinned paper fill-model version (SPEC §17.4).
    """

    metric_windows: list[list[str]]
    min_resolved_for_calibration: int
    promotion_min_resolved: int
    promotion_min_independent_event_groups: int
    brier_skill_required_ppm: int
    bootstrap_confidence_ppm: int
    live_rolling_window_size: int
    live_slippage_ratio_limit_ppm: int
    live_brier_degradation_band_ppm: int
    observation_window: str
    baseline_scheme: str
    clustering_scheme: str
    paper_fill_model_version: str


def _flat_plan_fields(plan: GatePlan) -> _FlatPlanFields:
    """Flatten ``plan`` into the precisely-typed shared registration-event fields.

    The values match :meth:`GatePlan.canonical_dict` exactly (so the persisted
    payload stays byte-identical), but the return type is a
    :class:`_FlatPlanFields` ``TypedDict`` rather than ``dict[str, object]``, so
    it unpacks straight into the flat registration-event constructors under
    ``mypy --strict``.

    Args:
        plan: The gate plan to flatten.

    Returns:
        The plan's metric catalogue (as JSON list-of-lists), its eight
        thresholds, and its four identity strings, precisely typed.
    """
    return {
        "metric_windows": [[name, window] for name, window in plan.metric_windows],
        "min_resolved_for_calibration": plan.min_resolved_for_calibration,
        "promotion_min_resolved": plan.promotion_min_resolved,
        "promotion_min_independent_event_groups": (
            plan.promotion_min_independent_event_groups
        ),
        "brier_skill_required_ppm": plan.brier_skill_required_ppm,
        "bootstrap_confidence_ppm": plan.bootstrap_confidence_ppm,
        "live_rolling_window_size": plan.live_rolling_window_size,
        "live_slippage_ratio_limit_ppm": plan.live_slippage_ratio_limit_ppm,
        "live_brier_degradation_band_ppm": plan.live_brier_degradation_band_ppm,
        "observation_window": plan.observation_window,
        "baseline_scheme": plan.baseline_scheme,
        "clustering_scheme": plan.clustering_scheme,
        "paper_fill_model_version": plan.paper_fill_model_version,
    }


def _append_first_registration(
    plan: GatePlan, store: LedgerStore, component: str, now: Callable[[], int]
) -> GatePlanRegistration:
    """Append a first :class:`GatePlanRegistered` and return its registration.

    Args:
        plan: The plan to register.
        store: The ledger to append to.
        component: The producing component recorded on the event.
        now: The clock supplying the paper-clock start epoch.

    Returns:
        The resulting registration, with ``previous_plan_hash`` of ``None``.
    """
    epoch = now()
    store.append(
        GatePlanRegistered(
            component=component,
            **_flat_plan_fields(plan),
            plan_hash=plan.plan_hash,
            paper_clock_start=epoch,
        )
    )
    return GatePlanRegistration(
        plan=plan,
        plan_hash=plan.plan_hash,
        paper_clock_start=epoch,
        previous_plan_hash=None,
        event_type="GatePlanRegistered",
    )


def _append_changed_registration(
    plan: GatePlan,
    store: LedgerStore,
    component: str,
    now: Callable[[], int],
    existing: GatePlanRegistration,
) -> GatePlanRegistration:
    """Append a :class:`GatePlanChanged` and return its registration.

    Args:
        plan: The new plan replacing the currently-registered one.
        store: The ledger to append to.
        component: The producing component recorded on the event.
        now: The clock supplying the new paper-clock start epoch.
        existing: The currently-registered plan's registration.

    Returns:
        The resulting registration, linking back to ``existing`` via
        ``previous_plan_hash``.

    Raises:
        ValueError: If the new epoch is not strictly later than the existing
            registration's ``paper_clock_start`` -- the fail-closed guard against
            a non-monotonic clock. Nothing is appended in that case.
    """
    new_epoch = now()
    if new_epoch <= existing.paper_clock_start:
        raise ValueError(
            "paper clock must advance strictly on a gate plan change: "
            f"new epoch {new_epoch} <= existing {existing.paper_clock_start}"
        )
    store.append(
        GatePlanChanged(
            component=component,
            **_flat_plan_fields(plan),
            plan_hash=plan.plan_hash,
            paper_clock_start=new_epoch,
            previous_plan_hash=existing.plan_hash,
        )
    )
    return GatePlanRegistration(
        plan=plan,
        plan_hash=plan.plan_hash,
        paper_clock_start=new_epoch,
        previous_plan_hash=existing.plan_hash,
        event_type="GatePlanChanged",
    )


def register_gate_plan(
    plan: GatePlan,
    store: LedgerStore,
    *,
    component: str = "evaluation",
    now: Callable[[], int] = _epoch_now,
) -> GatePlanRegistration:
    """Register a gate plan, resetting the paper clock only on a real change.

    Behavior:

    - No prior registration: append a :class:`GatePlanRegistered`.
    - A byte-identical plan already registered: idempotent -- append nothing and
      return the existing registration unchanged (the clock is not reset).
    - A different plan: append a :class:`GatePlanChanged` with a strictly-later
      ``paper_clock_start`` and the ``previous_plan_hash`` link, failing closed
      (appending nothing) if the clock is not strictly monotonic.

    Args:
        plan: The plan to register.
        store: The ledger to read the current registration from and append to.
        component: The producing component recorded on any appended event.
        now: A clock returning whole epoch seconds; injectable for tests.

    Returns:
        The resulting (or existing, on a no-op) registration.

    Raises:
        ValueError: On a change whose ``now()`` epoch is not strictly later than
            the current registration's ``paper_clock_start``.
    """
    existing = latest_gate_plan_registration(store)
    if existing is None:
        return _append_first_registration(plan, store, component, now)
    if existing.plan_hash == plan.plan_hash:
        return existing
    return _append_changed_registration(plan, store, component, now, existing)


def _require_dict(value: object, label: str) -> dict[str, object]:
    """Return ``value`` narrowed to a JSON-object dict.

    Args:
        value: The value to check, e.g. a parsed envelope section.
        label: The section name, named in the error.

    Returns:
        ``value`` as a ``dict[str, object]``.

    Raises:
        TypeError: If ``value`` is not a ``dict``.
    """
    if not isinstance(value, dict):
        raise TypeError(f"{label} requires a JSON object, got {type(value).__name__}")
    return value


def _optional_str(value: object) -> str | None:
    """Return ``value`` as a ``str`` or ``None``.

    Args:
        value: The value to check.

    Returns:
        ``None`` if ``value`` is ``None``, otherwise the value as a ``str``.

    Raises:
        TypeError: If ``value`` is neither ``None`` nor a ``str``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"previous_plan_hash requires a str or null, got {type(value).__name__}"
        )
    return value


def _registration_from_record(record: LedgerRecord) -> GatePlanRegistration:
    """Reconstruct a registration from one ledger record, verifying its hash.

    The content hash is **recomputed from the persisted, stripped plan dict**
    (the exact keys the record stored, whatever the schema at write time) and
    checked against the stored ``plan_hash``. Verifying the *persisted* content --
    rather than a freshly-serialized plan -- keeps the check content-addressed
    and schema-independent: a pre-#58, 10-key record still verifies against its
    own 10-key hash even though :meth:`GatePlan.from_canonical` reconstructs a
    13-key plan (applying documented legacy defaults for the three keys the
    record predates). The recomputed hash is authoritative -- the stored copy is
    a verified redundancy, never a trusted input -- so a tampered record fails
    closed rather than silently skipping the PAPER-clock reset (SPEC §13.6
    anti-Goodhart). The returned registration carries the *persisted* hash, so a
    legacy record's identity remains its legacy hash.

    Args:
        record: A ledger record whose ``event_type`` is a registration event.

    Returns:
        The registration reconstructed from the record's envelope.

    Raises:
        TypeError: If the persisted envelope is malformed for reconstruction.
        ValueError: If a required key is missing, the stripped plan mapping is
            not a valid canonical plan, or the recomputed content hash does not
            match the stored ``plan_hash``.
    """
    envelope = _require_dict(json.loads(record.payload_json), "envelope")
    data = _require_dict(_require_present(envelope, "data"), "data")
    plan_fields: dict[str, object] = {
        key: value for key, value in data.items() if key not in _REGISTRATION_ONLY_KEYS
    }
    stored_hash = _require_mapping_str(data, "plan_hash")
    content_hash = hashlib.sha256(
        canonical_json(plan_fields).encode("utf-8")
    ).hexdigest()
    if content_hash != stored_hash:
        raise ValueError(
            "gate plan hash mismatch on ledger read: "
            f"recomputed {content_hash} != stored {stored_hash}"
        )
    plan = GatePlan.from_canonical(plan_fields)
    return GatePlanRegistration(
        plan=plan,
        plan_hash=stored_hash,
        paper_clock_start=_require_mapping_int(data, "paper_clock_start"),
        previous_plan_hash=_optional_str(data.get("previous_plan_hash")),
        event_type=record.event_type,
    )


def latest_gate_plan_registration(store: LedgerStore) -> GatePlanRegistration | None:
    """Return the most recent gate-plan registration in the ledger, or ``None``.

    Locates the newest registration record (see
    :func:`_latest_registration_record` for the two lookup paths) and
    reconstructs its registration from the envelope.

    Args:
        store: The ledger to read.

    Returns:
        The latest registration, or ``None`` if the ledger has none.

    Raises:
        TypeError: If the latest registration record's envelope is malformed.
        ValueError: If that record is missing a required key or its recomputed
            plan hash does not match the stored one (fail-closed on a corrupt or
            tampered payload).
    """
    latest = _latest_registration_record(store)
    if latest is None:
        return None
    return _registration_from_record(latest)


def _latest_registration_record(store: LedgerStore) -> LedgerRecord | None:
    """Find the newest ``GatePlanRegistered``/``GatePlanChanged`` row.

    ``RiskKernel.request_promotion`` would drive this at least twice per PAPER
    promotion attempt, so the cost was made to matter (issue #246) -- though no
    production code calls ``request_promotion`` today, so that path is currently
    exercised only by tests and the optimization is latent (issue #542): a store
    declaring the optional :class:`~windbreak.ledger.store.LatestRecordLookup`
    capability answers it with a single index-backed reverse read, while any
    other store --
    including every hand-rolled :class:`~windbreak.ledger.store.LedgerStore`
    double, which is exactly why that capability is a separate protocol rather
    than a widening of ``LedgerStore`` -- falls back to the original full scan.
    Both paths return the same record, so the dispatch is a pure optimization
    and never a behavioral fork.

    Args:
        store: The ledger to read.

    Returns:
        The highest-sequence registration record, or ``None`` if the ledger
        holds none.
    """
    if isinstance(store, LatestRecordLookup):
        return store.latest_record_of_types(_REGISTRATION_EVENT_TYPES)
    latest: LedgerRecord | None = None
    for record in store.read_all():
        if record.event_type in _REGISTRATION_EVENT_TYPES:
            latest = record
    return latest
