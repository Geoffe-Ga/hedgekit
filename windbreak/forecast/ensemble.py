"""SPEC S8.6 ensemble vote aggregation (integer-only, deterministic).

Aggregates the ensemble's :class:`~windbreak.forecast.records.ModelVote`s into a
single :class:`VoteAggregate`: the integer median probability, an
exclusive-median (Moore-McCabe) inter-quartile dispersion, and the min/max
confidence bounds (SPEC S6.3). Every quantity is a bare parts-per-million
integer and every rounding decision is explicit -- the median and Q1 round
*down*
(:attr:`~windbreak.numeric.rounding.RoundingDirection.UNDERSTATE_EQUITY`) while
Q3 rounds *up*
(:attr:`~windbreak.numeric.rounding.RoundingDirection.OVERSTATE_COST`), a
deliberately risk-widening convention -- so no float ever enters the
probability path guarded by ``scripts/lint_no_floats.py``.

Dispersion is deliberately *provider-family-agnostic*: aggregation reads only
each vote's ``probability_ppm``, never its ``provider`` identity, so every
surviving vote contributes symmetrically to the exclusive-median IQR regardless
of member family (a research-forecaster ``futuresearch`` vote weighs exactly the
same as an ``openai``/``anthropic`` LLM vote). The spread is the ensemble's
honest disagreement signal that SPEC S9.6 sizes against -- it is never damped,
weighted, or re-scaled per provider family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from windbreak.numeric.rounding import RoundingDirection, divide

if TYPE_CHECKING:
    from collections.abc import Sequence

    from windbreak.forecast.records import ModelVote

#: The provenance fields every vote must pin before it may be aggregated; an
#: empty value in any of them is an unpinned-version error (SPEC S8.6 / T14).
_PROVENANCE_FIELDS: tuple[str, ...] = (
    "provider",
    "model_version",
    "declared_training_cutoff",
)

#: The divisor for the mean of an even-count median's two middle elements.
_MIDDLE_PAIR = 2


@dataclass(frozen=True, slots=True)
class VoteAggregate:
    """The aggregated view of an ensemble's votes (SPEC S8.6).

    Attributes:
        probability_ppm: The integer median vote probability, in ppm.
        vote_dispersion_ppm: The exclusive-median IQR spread, in ppm. Derived
            purely from the votes' probabilities, never their provider identity,
            so it is provider-family-agnostic: every vote contributes
            symmetrically regardless of member family (SPEC S9.6's honest
            disagreement signal, never damped or weighted per family).
        ci_low_ppm: The lowest vote probability (min), in ppm.
        ci_high_ppm: The highest vote probability (max), in ppm.
    """

    probability_ppm: int
    vote_dispersion_ppm: int
    ci_low_ppm: int
    ci_high_ppm: int


def _require_pinned_provenance(votes: Sequence[ModelVote]) -> None:
    """Reject any vote whose provenance strings are not fully pinned.

    Args:
        votes: The votes to check.

    Raises:
        ValueError: If any vote has an empty ``provider``, ``model_version``,
            or ``declared_training_cutoff``. The message names the field.
    """
    for vote in votes:
        for field_name in _PROVENANCE_FIELDS:
            if not getattr(vote, field_name):
                raise ValueError(
                    f"{field_name} must be pinned (non-empty) to aggregate a vote"
                )


def _integer_median(values: Sequence[int], *, rounding: RoundingDirection) -> int:
    """Return the integer median of a non-empty ascending sequence.

    An odd count yields the middle element exactly; an even count yields the
    mean of the two middle elements, rounded in the requested direction.

    Args:
        values: A non-empty, ascending sequence of ppm integers.
        rounding: The direction to round an even-count middle-pair mean.

    Returns:
        The integer median, in ppm.
    """
    count = len(values)
    mid = count // 2
    if count % 2 == 1:
        return values[mid]
    return divide(values[mid - 1] + values[mid], _MIDDLE_PAIR, rounding=rounding)


def _quartiles(values: Sequence[int]) -> tuple[int, int]:
    """Return ``(Q1, Q3)`` under the Moore-McCabe exclusive-median convention.

    The median element(s) are excluded from both halves; Q1 is the lower
    half's median rounded *down* and Q3 the upper half's median rounded *up*,
    widening the reported spread. A single value has no spread, so ``Q1 == Q3``.

    Args:
        values: A non-empty, ascending sequence of ppm integers.

    Returns:
        The lower and upper quartiles, in ppm.
    """
    count = len(values)
    lower = values[: count // 2]
    upper = values[(count + 1) // 2 :]
    if not lower:
        return values[0], values[0]
    q1 = _integer_median(lower, rounding=RoundingDirection.UNDERSTATE_EQUITY)
    q3 = _integer_median(upper, rounding=RoundingDirection.OVERSTATE_COST)
    return q1, q3


def aggregate_votes(votes: Sequence[ModelVote]) -> VoteAggregate:
    """Aggregate ensemble votes into a median, dispersion, and CI bounds.

    Aggregation reads only each vote's ``probability_ppm``, never its
    ``provider`` identity, so the result -- including ``vote_dispersion_ppm`` --
    is provider-family-agnostic: relabeling or reordering which member family
    produced which vote leaves every quantity unchanged. A research-forecaster
    vote contributes exactly like an LLM vote to the honest disagreement signal
    (SPEC S9.6); it is never damped or weighted per family.

    Args:
        votes: The ensemble votes to aggregate; must be non-empty and each
            fully version-pinned.

    Returns:
        The :class:`VoteAggregate` for ``votes``.

    Raises:
        ValueError: If ``votes`` is empty, or any vote is not fully pinned
            (empty ``provider`` / ``model_version`` / ``declared_training_cutoff``).
    """
    if not votes:
        raise ValueError("votes must be a non-empty sequence")
    _require_pinned_provenance(votes)
    probabilities = sorted(vote.probability_ppm for vote in votes)
    median = _integer_median(
        probabilities, rounding=RoundingDirection.UNDERSTATE_EQUITY
    )
    q1, q3 = _quartiles(probabilities)
    return VoteAggregate(
        probability_ppm=median,
        vote_dispersion_ppm=q3 - q1,
        ci_low_ppm=probabilities[0],
        ci_high_ppm=probabilities[-1],
    )
