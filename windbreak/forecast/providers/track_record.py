"""SPEC S13/S16 + S19 per-provider track-record live-eligibility gate (issue #194).

Governing sections: the promotion thresholds come from the S13/S16 ``evaluation``
block (``min_resolved_for_calibration`` / ``brier_skill_required_ppm``), applied
per provider; the honest-edge mandate is S19 (no unmeasured-edge claims).

A provider earns live eligibility only once its historical forecasts *prove* it:
enough resolved forecasts and a Brier skill at or above the promotion bar. This
module reads those M6 track-record artifacts (it never recomputes a score) and
turns them into a fail-closed gate -- a provider with no record, too few resolved
forecasts, or sub-threshold skill is "unproven" and its votes cannot back a live
order.

The promotion thresholds mirror :class:`windbreak.config.schema.EvaluationConfig`
(``min_resolved_for_calibration`` / ``brier_skill_required_ppm``) as local
constants rather than importing them, honoring the SPEC S8.3 sandbox boundary
(the ``canary.py`` convention). Every quantity is an integer parts-per-million or
a count -- never a float -- and :func:`parse_track_records` fails *closed* on a
float leaf, a ``bool`` where a count is expected, or an unknown key, so a
malformed artifact can never silently promote a provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import NoReturn

#: Minimum resolved forecasts a provider needs to be proven; mirrors
#: :attr:`windbreak.config.schema.EvaluationConfig.min_resolved_for_calibration`.
DEFAULT_MIN_RESOLVED: Final = 150

#: Minimum Brier skill, in ppm, a provider needs to be proven; mirrors
#: :attr:`windbreak.config.schema.EvaluationConfig.brier_skill_required_ppm`.
DEFAULT_MIN_BRIER_SKILL_PPM: Final = 10_000

#: The count field name every track-record entry must carry.
_RESOLVED_COUNT_KEY: Final = "resolved_count"

#: The Brier-skill field name every track-record entry must carry.
_BRIER_SKILL_PPM_KEY: Final = "brier_skill_ppm"

#: The optional provider field an entry may restate; tolerated but not
#: authoritative (the outer JSON key is always the provider identity).
_PROVIDER_KEY: Final = "provider"

#: Every key a single track-record entry may contain; any other is fatal. The
#: two measurement keys are additionally required (enforced per-key at read).
_ALLOWED_ENTRY_KEYS: Final = frozenset(
    {_PROVIDER_KEY, _RESOLVED_COUNT_KEY, _BRIER_SKILL_PPM_KEY}
)


def _reject_json_float(raw: str) -> NoReturn:
    """Fail closed on any JSON float leaf during a strict track-record parse.

    Wired as :func:`json.loads`'s ``parse_float`` so a fractional value (which a
    Brier skill or resolved count can never legitimately be) raises rather than
    being silently truncated to an int.

    Args:
        raw: The float literal's raw source text, as passed by the JSON scanner.

    Raises:
        ValueError: Always -- a float leaf is never a valid track-record value.
    """
    msg = f"track-record values must be integers, got float {raw!r}"
    raise ValueError(msg)


def _require_non_empty(value: str, field_name: str) -> None:
    """Reject an empty string identifier.

    Args:
        value: The identifier to check.
        field_name: The field name, for the error message.

    Raises:
        ValueError: If ``value`` is empty.
    """
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _require_int(value: int, field_name: str) -> None:
    """Guard that a field is a true (non-``bool``) integer.

    Args:
        value: The candidate value.
        field_name: The field name, for the error message.

    Raises:
        TypeError: If ``value`` is a ``bool`` or not an ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be a non-bool int, got {type(value).__name__}"
        )


def _require_non_negative_count(value: int, field_name: str) -> None:
    """Guard that a count field is a true, non-negative integer.

    Args:
        value: The candidate count.
        field_name: The field name, for the error message.

    Raises:
        TypeError: If ``value`` is a ``bool`` or not an ``int``.
        ValueError: If ``value`` is negative.
    """
    _require_int(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")


@dataclass(frozen=True, slots=True)
class ProviderTrackRecord:
    """One provider's resolved-forecast track record (read from M6 artifacts).

    Attributes:
        provider: The non-empty provider identifier (e.g. ``openai``).
        resolved_count: How many of the provider's forecasts have resolved; a
            non-negative count.
        brier_skill_ppm: The provider's Brier skill over baseline, in ppm. A
            negative value (worse than baseline) is valid but unproven.
    """

    provider: str
    resolved_count: int
    brier_skill_ppm: int

    def __post_init__(self) -> None:
        """Validate the provider identity and the two integer measurements.

        Mirrors :mod:`windbreak.forecast.records`'s bool/int convention: a stray
        ``bool`` (an ``int`` subclass) is rejected where a count/ppm is expected.

        Raises:
            TypeError: If either measurement is a ``bool`` or non-``int``.
            ValueError: If ``provider`` is empty or ``resolved_count`` is
                negative. Each message names the field.
        """
        _require_non_empty(self.provider, _PROVIDER_KEY)
        _require_non_negative_count(self.resolved_count, _RESOLVED_COUNT_KEY)
        _require_int(self.brier_skill_ppm, _BRIER_SKILL_PPM_KEY)


class TrackRecordSource(Protocol):
    """The seam through which one provider's track record is looked up."""

    def track_record_for(self, provider: str) -> ProviderTrackRecord | None:
        """Return the provider's track record, or ``None`` if it has none.

        Args:
            provider: The provider identifier to look up.

        Returns:
            The provider's :class:`ProviderTrackRecord`, or ``None``.
        """
        ...


class InMemoryTrackRecordSource:
    """A :class:`TrackRecordSource` backed by an in-memory provider map."""

    def __init__(self, records: Iterable[ProviderTrackRecord]) -> None:
        """Index an iterable of records by provider.

        Args:
            records: The track records to serve; a later record for a provider
                supersedes an earlier one.
        """
        self._by_provider: dict[str, ProviderTrackRecord] = {
            record.provider: record for record in records
        }

    def track_record_for(self, provider: str) -> ProviderTrackRecord | None:
        """Return the indexed record for ``provider``, or ``None`` if absent.

        Args:
            provider: The provider identifier to look up.

        Returns:
            The provider's record, or ``None`` when it was never supplied.
        """
        return self._by_provider.get(provider)


def _entry_int(entry: dict[str, object], key: str, provider: str) -> int:
    """Extract a required integer measurement from one parsed entry, fail-closed.

    Args:
        entry: The provider's parsed entry mapping.
        key: The measurement key to read.
        provider: The provider whose entry this is, for the error message.

    Returns:
        The integer value at ``key``.

    Raises:
        ValueError: If ``key`` is missing or its value is not a true integer
            (a ``bool`` is rejected, since it is an ``int`` subclass).
    """
    if key not in entry:
        raise ValueError(f"track record for {provider!r} is missing {key!r}")
    value = entry[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{provider}.{key} must be an integer, got {type(value).__name__}"
        )
    return value


def _parse_entry(provider: str, entry: object) -> ProviderTrackRecord:
    """Parse one provider's raw JSON entry into a validated track record.

    The provider identity is always the outer JSON key; an optional restated
    ``provider`` leaf inside the entry is tolerated (it is a permitted key) but
    never authoritative.

    Args:
        provider: The provider identifier (the outer JSON key).
        entry: The raw parsed entry value; must be a mapping.

    Returns:
        The validated :class:`ProviderTrackRecord`.

    Raises:
        ValueError: If ``entry`` is not a mapping, carries an unknown key, or
            omits a required measurement.
    """
    if not isinstance(entry, dict):
        raise ValueError(
            f"track record for {provider!r} must be a mapping, "
            f"got {type(entry).__name__}"
        )
    unknown = sorted(set(entry) - _ALLOWED_ENTRY_KEYS)
    if unknown:
        raise ValueError(f"unknown track-record key(s) for {provider!r}: {unknown}")
    return ProviderTrackRecord(
        provider=provider,
        resolved_count=_entry_int(entry, _RESOLVED_COUNT_KEY, provider),
        brier_skill_ppm=_entry_int(entry, _BRIER_SKILL_PPM_KEY, provider),
    )


def parse_track_records(text: str) -> dict[str, ProviderTrackRecord]:
    """Parse a strict-JSON track-record document into a provider->record map.

    The read model fails *closed*: a float leaf, a ``bool`` where an integer is
    expected, an unknown per-entry key, or a missing measurement all raise
    rather than silently coercing, so a malformed M6 artifact can never promote a
    provider it should not.

    Args:
        text: The JSON document mapping each provider to its ``{resolved_count,
            brier_skill_ppm}`` entry.

    Returns:
        The parsed records keyed by provider.

    Raises:
        ValueError: If the JSON is not a provider mapping, or any entry is
            malformed (float leaf, bool-as-int, unknown key, missing field).
    """
    raw = json.loads(text, parse_float=_reject_json_float)
    if not isinstance(raw, dict):
        raise ValueError(
            f"track-record document must be a JSON object, got {type(raw).__name__}"
        )
    return {provider: _parse_entry(provider, entry) for provider, entry in raw.items()}


class ProviderTrackRecordGate:
    """Fail-closed per-provider live-eligibility gate (SPEC S13/S16).

    A provider is *proven* only when it has a track record with at least
    ``min_resolved`` resolved forecasts and a Brier skill of at least
    ``min_brier_skill_ppm`` (both bounds inclusive). A missing record, too few
    resolved forecasts, or sub-threshold skill leaves it unproven, so its votes
    cannot back a live order.
    """

    def __init__(
        self,
        source: TrackRecordSource,
        *,
        min_resolved: int = DEFAULT_MIN_RESOLVED,
        min_brier_skill_ppm: int = DEFAULT_MIN_BRIER_SKILL_PPM,
    ) -> None:
        """Initialize the gate over a track-record source and its thresholds.

        Args:
            source: The track-record lookup seam.
            min_resolved: The minimum resolved-forecast count to be proven
                (keyword-only).
            min_brier_skill_ppm: The minimum Brier skill, in ppm, to be proven
                (keyword-only).
        """
        self._source = source
        self._min_resolved = min_resolved
        self._min_brier_skill_ppm = min_brier_skill_ppm

    @property
    def min_resolved(self) -> int:
        """Return the minimum resolved-forecast count a provider must meet."""
        return self._min_resolved

    @property
    def min_brier_skill_ppm(self) -> int:
        """Return the minimum Brier skill, in ppm, a provider must meet."""
        return self._min_brier_skill_ppm

    def is_provider_proven(self, provider: str) -> bool:
        """Return whether ``provider``'s track record clears both thresholds.

        Args:
            provider: The provider identifier to check.

        Returns:
            ``True`` when a record exists and meets both the resolved-count and
            Brier-skill bars (``>=`` on each); ``False`` otherwise, including a
            missing record (fail-closed).
        """
        record = self._source.track_record_for(provider)
        return (
            record is not None
            and record.resolved_count >= self._min_resolved
            and record.brier_skill_ppm >= self._min_brier_skill_ppm
        )

    def unproven_providers(self, providers: Iterable[str]) -> tuple[str, ...]:
        """Return the sorted, deduped subset of ``providers`` that are unproven.

        Args:
            providers: The provider identifiers to screen (duplicates allowed).

        Returns:
            The unproven providers, sorted and deduplicated.
        """
        unproven: set[str] = set()
        for provider in providers:
            if not self.is_provider_proven(provider):
                unproven.add(provider)
        return tuple(sorted(unproven))
