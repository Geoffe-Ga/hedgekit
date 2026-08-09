"""SPEC S8.4/S8.6/S16 weekly canary set and live-eligibility drift gate.

A *canary set* is a small, fixed battery of reference questions whose "correct"
resolution probabilities are known in advance. Re-running it on a pinned model
and comparing each observed probability against its reference surfaces silent
model drift before it can poison live forecasts. :func:`run_canary_set` gathers
one deterministic observation per question (mirroring
:func:`windbreak.forecast.triage.run_stage0_prior`'s single-call shape),
:func:`score_canary_run` reduces the observations to a pure drift score (the
worst per-question distance), and :class:`CanaryGate` turns a breach of that
score past a tolerance into a live-eligibility block plus an operator alert and
a ledgered audit trail.

Every decision is ledgered through the :class:`CanaryLedgerWriter` seam (modeled
verbatim on :class:`windbreak.forecast.triage.TriageLedgerWriter`): a
``CANARY_DRIFT``, ``CANARY_OK``, or ``CANARY_ACK`` :class:`CanaryEvent` whose
payload leaves are exact ``int``/``str``/``bool`` values -- never a float, per
the package-wide no-float convention ``scripts/lint_no_floats.py`` enforces. All
arithmetic here is integer-only for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol

from windbreak.alerts import AlertType
from windbreak.forecast.cassettes import LlmRequest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from windbreak.forecast.cassettes import LlmTransport

#: The canary drift tolerance, in ppm (provisional per SPEC Open Question #7).
#: It mirrors :data:`windbreak.forecast.triage.TRIAGE_THRESHOLD_PPM`'s 5-point
#: (50_000 ppm) scale; config plumbing of an operator-tunable value is deferred.
DEFAULT_CANARY_DRIFT_TOLERANCE_PPM: Final = 50_000

#: Lowest legal ppm value (0.0 probability), for references and observations.
_MIN_PPM = 0

#: Highest legal ppm value (1.0 probability), for references and observations.
_MAX_PPM = 1_000_000

#: Event type recorded when a canary run drifts past its tolerance.
CANARY_DRIFT_EVENT = "CANARY_DRIFT"

#: Event type recorded when a canary run stays within its tolerance.
CANARY_OK_EVENT = "CANARY_OK"

#: Event type recorded when an operator acknowledges an active drift.
CANARY_ACK_EVENT = "CANARY_ACK"


class _CanaryModel(NamedTuple):
    """The pinned canary model's provenance strings.

    Mirrors :class:`windbreak.forecast.triage.TriageModel`: pinning the
    provider/version keeps each canary request byte-stable across runs.

    Attributes:
        provider: The LLM provider identifier.
        model_version: The pinned model version string.
    """

    provider: str
    model_version: str


#: The single pinned model that produces every canary observation.
_CANARY_MODEL = _CanaryModel("openai", "gpt-5-canary-mini")


@dataclass(frozen=True, slots=True)
class CanaryQuestion:
    """One reference question in the weekly canary set (SPEC S8.4).

    Attributes:
        question_id: The stable identifier of this canary question.
        prompt: The question text posed to the pinned canary model.
        reference_ppm: The known-correct resolution probability, in ppm.
    """

    question_id: str
    prompt: str
    reference_ppm: int

    def __post_init__(self) -> None:
        """Validate the identifier non-emptiness and reference-ppm range.

        Raises:
            ValueError: If ``question_id`` is empty, or ``reference_ppm`` is
                outside ``[0, 1_000_000]``. Each message names the field.
        """
        if not self.question_id:
            msg = "question_id must be non-empty"
            raise ValueError(msg)
        if not _MIN_PPM <= self.reference_ppm <= _MAX_PPM:
            msg = (
                f"reference_ppm must be within [{_MIN_PPM}, {_MAX_PPM}], "
                f"got {self.reference_ppm}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CanaryRunResult:
    """The pure scoring of one canary run (SPEC S8.6).

    Attributes:
        distances_ppm: Per-question absolute distance from reference, in ppm.
        drift_score_ppm: The worst (maximum) per-question distance, in ppm.
        worst_question_id: The id of the question with the worst distance.
    """

    distances_ppm: Mapping[str, int]
    drift_score_ppm: int
    worst_question_id: str


@dataclass(frozen=True, slots=True)
class CanaryEvent:
    """One recorded canary decision (mirrors ``TriageEvent``).

    Attributes:
        event_type: The event kind (``CANARY_DRIFT``/``CANARY_OK``/``CANARY_ACK``).
        payload: The JSON-safe event body (int/str/bool leaves only).
        ts: ISO-8601 UTC timestamp of when the event was created.
    """

    event_type: str
    payload: Mapping[str, object]
    ts: str


class CanaryLedgerWriter(Protocol):
    """The seam through which a canary decision is persisted."""

    def record(self, event: CanaryEvent) -> None:
        """Persist a canary event.

        Args:
            event: The event to persist.
        """
        ...


class InMemoryCanaryLedger:
    """A :class:`CanaryLedgerWriter` that retains events in memory for tests."""

    def __init__(self) -> None:
        """Initialize with an empty event log."""
        self._events: list[CanaryEvent] = []

    def record(self, event: CanaryEvent) -> None:
        """Append a canary event to the in-memory log.

        Args:
            event: The event to retain.
        """
        self._events.append(event)

    def events_by_type(self, event_type: str) -> tuple[CanaryEvent, ...]:
        """Return every retained event of a given type, in record order.

        Args:
            event_type: The event kind to filter by.

        Returns:
            The matching events.
        """
        return tuple(event for event in self._events if event.event_type == event_type)


class CanaryAlertEmitter(Protocol):
    """The seam through which a canary drift alert is dispatched.

    A real :class:`windbreak.alerts.AlertDispatcher` satisfies this structurally.
    """

    def dispatch(self, alert_type: AlertType, message: str) -> object:
        """Dispatch a canary alert.

        Args:
            alert_type: The alert type to dispatch.
            message: The alert body.

        Returns:
            An opaque result; callers never inspect this seam's return value.
        """
        ...


def _iso_z(moment: datetime) -> str:
    """Render a datetime as ISO-8601 UTC with a trailing ``Z``.

    Follows the local-``_iso_z`` precedent in ``triage.py``/``pipeline.py``/
    ``records.py`` (each module defines its own) rather than sharing one.

    Args:
        moment: The (timezone-aware) datetime to render; normalized to UTC.

    Returns:
        A string like ``2024-12-10T12:00:00.000000Z``.
    """
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _canary_prompt(question: CanaryQuestion) -> str:
    """Build the deterministic canary prompt for one reference question.

    Args:
        question: The canary question being posed.

    Returns:
        A deterministic prompt string keyed on the question id and prompt text.
    """
    return (
        f"Canary check {question.question_id}: estimate the resolution "
        f"probability in ppm for: {question.prompt}"
    )


def parse_observed_ppm(response: str) -> int:
    """Parse a canary response into a validated ppm observation, fail-closed.

    Mirrors :func:`windbreak.forecast.triage._parse_prior_ppm`'s contract with a
    package-local implementation: the response must be a bare integer string
    within ``[0, 1_000_000]``; a non-integer (e.g. ``"0.5"`` or ``"maybe"``) or
    an out-of-range value fails loudly rather than silently defaulting.

    This is the single canonical observed-ppm contract shared across the module
    boundary: ``scripts/run_canaries.py`` reuses it so the operator-run replay
    and live-record paths validate observations identically to this in-package
    gate (issue #195 review parity fix), which is why it is public.

    Args:
        response: The raw canary completion text.

    Returns:
        The parsed ppm observation.

    Raises:
        ValueError: If ``response`` is not an integer or falls outside
            ``[0, 1_000_000]``.
    """
    try:
        value = int(response)
    except ValueError as exc:
        msg = f"canary observation must be an integer ppm string, got {response!r}"
        raise ValueError(msg) from exc
    if not _MIN_PPM <= value <= _MAX_PPM:
        msg = f"canary observation {response!r} is outside [{_MIN_PPM}, {_MAX_PPM}]"
        raise ValueError(msg)
    return value


def run_canary_set(
    questions: tuple[CanaryQuestion, ...],
    *,
    transport: LlmTransport,
) -> dict[str, int]:
    """Gather one deterministic observation per canary question (SPEC S8.4).

    Issues exactly one :class:`LlmRequest` per question on the pinned canary
    model, parsing each response into a validated ppm observation (fail-closed
    on a non-integer or out-of-range value).

    Args:
        questions: The canary questions to observe.
        transport: The LLM transport for the per-question calls (keyword-only).

    Returns:
        A mapping of each question id to its observed ppm.

    Raises:
        ValueError: If any response is not an integer ppm within
            ``[0, 1_000_000]``.
    """
    observed: dict[str, int] = {}
    for question in questions:
        request = LlmRequest(
            provider=_CANARY_MODEL.provider,
            model_version=_CANARY_MODEL.model_version,
            prompt=_canary_prompt(question),
        )
        response = transport.complete(request)
        observed[question.question_id] = parse_observed_ppm(response)
    return observed


def score_canary_run(
    questions: tuple[CanaryQuestion, ...],
    observed_ppm: Mapping[str, int],
) -> CanaryRunResult:
    """Score a canary run into its worst per-question drift (pure, SPEC S8.6).

    Each question's distance is the absolute difference between its observed and
    reference ppm; the drift score is the maximum such distance and the worst
    question id is the first question (in set order) attaining it.

    Args:
        questions: The canary questions that were observed.
        observed_ppm: The per-question observed ppm, keyed by question id.

    Returns:
        The scored :class:`CanaryRunResult`.

    Raises:
        ValueError: If ``observed_ppm``'s keys do not exactly match the question
            ids.
    """
    question_ids = {question.question_id for question in questions}
    if set(observed_ppm) != question_ids:
        msg = (
            f"observed ids {sorted(observed_ppm)} must exactly match question "
            f"ids {sorted(question_ids)}"
        )
        raise ValueError(msg)
    distances: dict[str, int] = {
        question.question_id: abs(
            observed_ppm[question.question_id] - question.reference_ppm
        )
        for question in questions
    }
    drift_score = max(distances.values())
    worst_question_id = next(
        question_id
        for question_id, distance in distances.items()
        if distance == drift_score
    )
    return CanaryRunResult(
        distances_ppm=distances,
        drift_score_ppm=drift_score,
        worst_question_id=worst_question_id,
    )


#: The ``drift_kind`` payload leaf stamped on a per-provider answer-drift breach
#: (an observed-probability distance past tolerance).
_DRIFT_KIND_ANSWER = "answer"

#: The ``drift_kind`` payload leaf stamped on a per-provider version-drift breach
#: (a reported forecaster version off its pinned set).
_DRIFT_KIND_VERSION = "version"


@dataclass(slots=True)
class _DriftWindow:
    """One drift dimension's mutable block window: its drift and ack instants.

    A single such window keys off either the global (pinned-canary-model)
    dimension or one provider. Its semantics are the pre-#195 gate's, lifted
    verbatim into a reusable helper so the keyed map cannot diverge from the
    global path.

    Attributes:
        drifted_at: The start of the current unacknowledged block window, or
            ``None`` when no drift is active.
        acked_at: The acknowledgement instant closing the window, or ``None``
            when the active drift is unacknowledged.
    """

    drifted_at: datetime | None = None
    acked_at: datetime | None = None

    def register_breach(self, checked_at: datetime) -> None:
        """Open or extend this window for a breach checked at ``checked_at``.

        A fresh breach at or after the last acknowledgement re-arms the window
        (a new unacknowledged drift opens); otherwise the earliest breach
        instant is kept, so a later, still-unacknowledged breach never pushes
        the window forward.

        Args:
            checked_at: When the breaching canary run was checked.
        """
        if self.acked_at is not None and checked_at >= self.acked_at:
            self.drifted_at = checked_at
            self.acked_at = None
        elif self.drifted_at is None or checked_at < self.drifted_at:
            self.drifted_at = checked_at

    def is_blocked(self, created_at: datetime) -> bool:
        """Return whether a record created at ``created_at`` is blocked here.

        Args:
            created_at: The record's creation instant.

        Returns:
            ``True`` when this window has drifted, the record was created at or
            after the drift instant, and the drift is either unacknowledged or
            the record predates the acknowledgement.
        """
        return (
            self.drifted_at is not None
            and created_at >= self.drifted_at
            and (self.acked_at is None or created_at < self.acked_at)
        )


class CanaryGate:
    """The live-eligibility gate driven by canary drift (SPEC S8.4/S16).

    A drift score strictly greater than the tolerance breaches the gate: an
    alert fires, the breach is ledgered, and every forecast created at or after
    the current unacknowledged drift instant is blocked from live eligibility
    until an operator acknowledges the drift. Acknowledging closes the window;
    a later breach at or after that acknowledgement re-arms the gate, opening a
    fresh block window at the new breach instant.

    Issue #195 layers a per-provider drift dimension onto the original global
    (pinned-canary-model) dimension. Each dimension is an independent
    :class:`_DriftWindow` keyed in :attr:`_windows`, where the key ``None`` is
    the global dimension and a provider string is that provider's dimension.
    The global path stays byte-identical: a ``provider=None`` call carries none
    of the new ``provider``/``drift_kind`` payload leaves and re-uses the exact
    global window logic. A provider query fails closed for everyone when the
    global window blocks (``is_live_blocked(provider=p)`` ORs the two windows).
    """

    def __init__(
        self, *, drift_tolerance_ppm: int = DEFAULT_CANARY_DRIFT_TOLERANCE_PPM
    ) -> None:
        """Initialize the gate with a drift tolerance and no active drift.

        Args:
            drift_tolerance_ppm: The maximum drift score, in ppm, that stays
                within band; a score strictly above it breaches.
        """
        self._drift_tolerance_ppm = drift_tolerance_ppm
        #: One block window per drift dimension: ``None`` is the global
        #: (pinned-canary-model) dimension, a provider string is that provider's.
        self._windows: dict[str | None, _DriftWindow] = {}

    def _window(self, key: str | None) -> _DriftWindow:
        """Return the drift window for ``key``, creating it on first use.

        Args:
            key: The drift dimension (``None`` for global, else a provider).

        Returns:
            The (possibly freshly created) window for the dimension.
        """
        window = self._windows.get(key)
        if window is None:
            window = _DriftWindow()
            self._windows[key] = window
        return window

    def _answer_breach_payload(
        self, result: CanaryRunResult, provider: str | None
    ) -> dict[str, object]:
        """Build the JSON-safe ``CANARY_DRIFT`` payload for an answer breach.

        The global (``provider=None``) payload is byte-identical to the pre-#195
        breach payload; a per-provider breach adds the ``provider`` and
        ``drift_kind="answer"`` leaves.

        Args:
            result: The scored canary run that breached.
            provider: The provider whose run breached, or ``None`` for global.

        Returns:
            A mapping of int/str leaves (never a float).
        """
        payload: dict[str, object] = {
            "drift_score_ppm": result.drift_score_ppm,
            "tolerance_ppm": self._drift_tolerance_ppm,
            "worst_question_id": result.worst_question_id,
            "question_count": len(result.distances_ppm),
        }
        if provider is not None:
            payload["provider"] = provider
            payload["drift_kind"] = _DRIFT_KIND_ANSWER
        return payload

    def _ok_payload(self, result: CanaryRunResult) -> dict[str, object]:
        """Build the JSON-safe ``CANARY_OK`` payload leaves.

        Args:
            result: The scored canary run that stayed within band.

        Returns:
            A mapping of int leaves (never a float).
        """
        return {
            "drift_score_ppm": result.drift_score_ppm,
            "tolerance_ppm": self._drift_tolerance_ppm,
            "question_count": len(result.distances_ppm),
        }

    def _answer_drift_message(
        self, result: CanaryRunResult, provider: str | None
    ) -> str:
        """Build the answer-drift alert message for global or a provider.

        The global message is byte-identical to the pre-#195 text; a
        per-provider message prefixes it with the provider and drift kind.

        Args:
            result: The scored canary run that breached.
            provider: The provider whose run breached, or ``None`` for global.

        Returns:
            The alert body.
        """
        base = (
            f"Canary drift {result.drift_score_ppm} ppm exceeded tolerance "
            f"{self._drift_tolerance_ppm} ppm; worst question "
            f"{result.worst_question_id}"
        )
        if provider is None:
            return base
        return f"Provider {provider} {_DRIFT_KIND_ANSWER}-drift: {base}"

    def apply_run(
        self,
        result: CanaryRunResult,
        *,
        checked_at: datetime,
        alerts: CanaryAlertEmitter,
        ledger: CanaryLedgerWriter,
        provider: str | None = None,
    ) -> bool:
        """Apply a scored canary run to the gate, returning whether it breached.

        The tolerance is STRICT: a drift score exactly at tolerance stays within
        band (``>`` decides a breach). A breach dispatches exactly one
        ``CANARY_DRIFT`` alert and ledgers one ``CANARY_DRIFT`` event; a
        within-band run ledgers one ``CANARY_OK`` event and touches nothing else.
        With ``provider`` supplied the breach blocks only that provider's window
        and stamps the ``provider``/``drift_kind="answer"`` payload leaves; with
        ``provider=None`` the global (pinned-canary-model) window is used and the
        payload stays byte-identical to the pre-#195 form.

        Args:
            result: The scored canary run to apply.
            checked_at: When the canary run was checked (keyword-only).
            alerts: The alert emitter to dispatch a breach through (keyword-only).
            ledger: The canary-event ledger writer (keyword-only).
            provider: The provider whose run this is, or ``None`` for the global
                dimension (keyword-only).

        Returns:
            ``True`` if the run breached tolerance, else ``False``.
        """
        if result.drift_score_ppm > self._drift_tolerance_ppm:
            alerts.dispatch(
                AlertType.CANARY_DRIFT,
                self._answer_drift_message(result, provider),
            )
            ledger.record(
                CanaryEvent(
                    CANARY_DRIFT_EVENT,
                    self._answer_breach_payload(result, provider),
                    _iso_z(checked_at),
                )
            )
            self._window(provider).register_breach(checked_at)
            return True
        ledger.record(
            CanaryEvent(CANARY_OK_EVENT, self._ok_payload(result), _iso_z(checked_at))
        )
        return False

    def apply_version_drift(
        self,
        provider: str,
        reported_version: str,
        pinned_versions: tuple[str, ...],
        *,
        checked_at: datetime,
        alerts: CanaryAlertEmitter,
        ledger: CanaryLedgerWriter,
    ) -> bool:
        """Gate a provider on its reported forecaster version (SPEC S8.4/T14).

        A reported version that IS in the pinned set is a no-op (no alert, no
        ledgered event, no block). An off-pin version blocks that provider,
        dispatches exactly one ``CANARY_DRIFT`` alert naming the provider and the
        version drift kind, and ledgers one ``CANARY_DRIFT`` event carrying the
        ``provider``, ``drift_kind="version"``, and reported/pinned version
        leaves (all int/str/bool, never a float).

        Args:
            provider: The provider whose reported version is being gated.
            reported_version: The forecaster version the provider reported.
            pinned_versions: The operator-pinned versions considered valid.
            checked_at: When the version was observed (keyword-only).
            alerts: The alert emitter to dispatch a breach through (keyword-only).
            ledger: The canary-event ledger writer (keyword-only).

        Returns:
            ``True`` if the version drifted off the pinned set, else ``False``.
        """
        if reported_version in pinned_versions:
            return False
        message = (
            f"Provider {provider} {_DRIFT_KIND_VERSION}-drift: reported "
            f"forecaster version {reported_version!r} is off the pinned set "
            f"{list(pinned_versions)}"
        )
        alerts.dispatch(AlertType.CANARY_DRIFT, message)
        payload: dict[str, object] = {
            "provider": provider,
            "drift_kind": _DRIFT_KIND_VERSION,
            "reported_version": reported_version,
            "pinned_versions": list(pinned_versions),
        }
        ledger.record(CanaryEvent(CANARY_DRIFT_EVENT, payload, _iso_z(checked_at)))
        self._window(provider).register_breach(checked_at)
        return True

    def acknowledge(
        self,
        *,
        acked_at: datetime,
        ledger: CanaryLedgerWriter,
        provider: str | None = None,
    ) -> None:
        """Acknowledge an active drift, restoring eligibility for new records.

        Acknowledging restores eligibility only for the named dimension's
        records created at or after ``acked_at``; the two dimensions are
        orthogonal, so acking one provider never touches the global window (or
        any sibling provider). The global (``provider=None``) ack payload stays
        byte-identical to the pre-#195 form.

        Args:
            acked_at: The acknowledgement instant; records created at or after
                it are no longer blocked (keyword-only).
            ledger: The canary-event ledger writer (keyword-only).
            provider: The provider whose drift is acknowledged, or ``None`` for
                the global dimension (keyword-only).

        Raises:
            ValueError: If the named dimension has no active drift.
        """
        window = self._windows.get(provider)
        if window is None or window.drifted_at is None:
            msg = "cannot acknowledge: no active canary drift"
            raise ValueError(msg)
        window.acked_at = acked_at
        payload: dict[str, object] = {
            "drifted_at": _iso_z(window.drifted_at),
            "acked_at": _iso_z(acked_at),
        }
        if provider is not None:
            payload["provider"] = provider
        ledger.record(CanaryEvent(CANARY_ACK_EVENT, payload, _iso_z(acked_at)))

    def _is_dimension_blocked(self, key: str | None, created_at: datetime) -> bool:
        """Return whether a single drift dimension blocks ``created_at``.

        Args:
            key: The drift dimension (``None`` for global, else a provider).
            created_at: The record's creation instant.

        Returns:
            ``True`` when that dimension's window exists and blocks the record.
        """
        window = self._windows.get(key)
        return window is not None and window.is_blocked(created_at)

    def is_live_blocked(
        self, *, created_at: datetime, provider: str | None = None
    ) -> bool:
        """Return whether a record created at ``created_at`` is drift-blocked.

        The global (pinned-canary-model) window blocks every record; a provider
        query additionally consults that provider's own window, so it fails
        closed whenever either the provider OR the global dimension has drifted.

        Args:
            created_at: The record's creation instant (keyword-only).
            provider: The provider the record would run under, or ``None`` to
                query the global dimension alone (keyword-only).

        Returns:
            ``True`` if the record is blocked from live eligibility.
        """
        if self._is_dimension_blocked(None, created_at):
            return True
        return provider is not None and self._is_dimension_blocked(provider, created_at)
