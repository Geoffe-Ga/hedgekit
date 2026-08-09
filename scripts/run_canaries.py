#!/usr/bin/env python3
"""Operator-run provider canary battery driver (fleet observability, #195).

CI never dials a live forecaster: this developer-run script is the only place
provider canaries reach a network, and the only place ``requests`` and the
process environment appear on this path -- deliberately kept out of the
``windbreak`` package so CI stays network-library-free and the SPEC S8.3 sandbox
boundary is never crossed. All testable logic lives in
:func:`windbreak.scheduler.canaries.run_canaries`; this wrapper only builds the
:class:`~windbreak.forecast.canary_providers.ProviderCanarySpec` battery and its
observers, then delegates and exits with the returned code (non-zero on any
drift, so ``scripts/run-canaries.sh`` fails loudly).

A battery is described by a JSON spec file. In replay mode (the default) each
provider's observation is read straight from the file, fully offline. In record
mode (``--record``) a live, allowlisted, redirect-free transport dials each
provider's endpoint once, injecting the provider's API key -- read from the
``<PROVIDER>_API_KEY`` environment variable, never a literal in this file -- as
a send-time header, mirroring ``scripts/record_vote_cassettes.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from windbreak.forecast.canary import CanaryQuestion, parse_observed_ppm
from windbreak.forecast.canary_providers import (
    ProviderCanaryObservation,
    ProviderCanarySpec,
)
from windbreak.ledger.store import ChainIntegrityError
from windbreak.net.allowlist import OutboundAllowlist
from windbreak.scheduler.canaries import run_canaries

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from windbreak.forecast.canary_providers import ProviderCanaryObserver

#: Per-request timeout, in whole seconds (an int -- no float on this path).
_TIMEOUT_SECONDS = 30

#: Suffix appended to a provider's upper-cased name to form the environment
#: variable its live API key is read from (e.g. ``FUTURESEARCH_API_KEY``). Held
#: as a module-level constant whose name deliberately omits any credential
#: keyword so detect-secrets' keyword heuristic does not flag the assignment
#: (mirrors ``_DEFAULT_ENV_VAR`` in the futuresearch provider config test;
#: local/CI parity gap tracked in issue #262). It is a variable-name fragment,
#: never a secret, and the key's value is never logged.
_ENV_VAR_SUFFIX = "_API_KEY"

#: The request header name a provider's API key is injected under at send time.
#: Named without a credential keyword for the same detect-secrets reason above.
_AUTH_HEADER_NAME = "x-api-key"

#: Shared JSON content-type header.
_CONTENT_TYPE_HEADER = "content-type"
_JSON_CONTENT_TYPE = "application/json"


class _ReplayObserver:
    """An offline observer replaying one fixed observation from the spec file."""

    def __init__(self, observation: ProviderCanaryObservation) -> None:
        """Store the fixed observation this observer replays.

        Args:
            observation: The observation read from the spec file.
        """
        self._observation = observation

    def observe(self, spec: ProviderCanarySpec) -> ProviderCanaryObservation:
        """Return the fixed observation, ignoring ``spec``.

        Args:
            spec: The (unused) spec being observed.

        Returns:
            The fixed observation.
        """
        del spec
        return self._observation


class _LiveObserver:
    """A live observer dialing one provider's canary endpoint once.

    The API key is held here and injected as a send-time header, so it never
    touches any persisted artefact. Egress is screened against an allowlist and
    redirects are refused, mirroring ``record_vote_cassettes._LiveHttpTransport``.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        headers: dict[str, str],
        allowlist: OutboundAllowlist,
        session: requests.Session,
    ) -> None:
        """Store the endpoint, send-time headers, allowlist, and session.

        Args:
            endpoint: The provider canary endpoint URL to dial.
            headers: The headers (including the API key) injected on each send.
            allowlist: The egress allowlist screening the outbound URL.
            session: The live ``requests`` session to dial through.
        """
        self._endpoint = endpoint
        self._headers = headers
        self._allowlist = allowlist
        self._session = session

    def observe(self, spec: ProviderCanarySpec) -> ProviderCanaryObservation:
        """Dial the endpoint once and parse the returned observation.

        Args:
            spec: The spec whose questions are posted to the endpoint.

        Returns:
            The parsed observation.

        Raises:
            EgressDeniedError: If the endpoint is off the allowlist.
        """
        self._allowlist.require(self._endpoint)
        body = json.dumps(
            {"questions": [question.question_id for question in spec.questions]}
        )
        response = self._session.request(
            "POST",
            self._endpoint,
            data=body.encode("utf-8"),
            headers=self._headers,
            timeout=_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
        return _observation_from_payload(json.loads(response.text))


class _StdoutAlertEmitter:
    """A minimal ``CanaryAlertEmitter`` printing each drift alert to stdout."""

    def dispatch(self, alert_type: object, message: str) -> object:
        """Print the alert and return an opaque sentinel.

        Args:
            alert_type: The alert type dispatched.
            message: The alert body.

        Returns:
            A sentinel; callers never inspect this seam's return value.
        """
        print(f"ALERT {alert_type}: {message}", file=sys.stderr)
        return object()


def _observation_from_payload(payload: Mapping[str, Any]) -> ProviderCanaryObservation:
    """Build an observation from a decoded JSON payload, fail-closed.

    Every ``observed_ppm`` leaf is validated through the canonical
    :func:`windbreak.forecast.canary.parse_observed_ppm` contract -- a strict
    integer within ``[0, 1_000_000]`` -- so BOTH the offline ``--replay`` path
    (:class:`_ReplayObserver`, via :func:`_build_observer`) and the live
    ``--record`` path (:class:`_LiveObserver.observe`) reject a malformed
    observation instead of silently truncating a float (``int(0.5) == 0``) or
    accepting an out-of-range value. Each value is normalised to text before
    validation so the JSON-decoded ``int``/``float`` leaves flow through the
    exact same parser the in-package gate uses: a float renders with a decimal
    point and is rejected, an out-of-range integer is rejected on bounds.

    Rejecting (never clamping) preserves the fail-closed direction: a malformed
    observation aborts the run rather than scoring as ``OK`` -- the opposite of
    the fail-safe inversion an unbounded ``int(value)`` would allow (e.g. an
    observed ``1_000_001`` against a reference ``999_999`` reads as a 2 ppm
    drift, well under tolerance).

    Args:
        payload: A mapping with ``observed_ppm`` and ``reported_version`` keys.

    Returns:
        The parsed observation.

    Raises:
        ValueError: If any ``observed_ppm`` value is not an integer within
            ``[0, 1_000_000]``.
    """
    observed = {
        str(key): parse_observed_ppm(str(value))
        for key, value in payload["observed_ppm"].items()
    }
    return ProviderCanaryObservation(
        observed_ppm=observed, reported_version=str(payload["reported_version"])
    )


def _questions_from_entry(entry: Mapping[str, Any]) -> tuple[CanaryQuestion, ...]:
    """Build the canary questions for one provider entry.

    Args:
        entry: One provider entry from the spec file.

    Returns:
        The provider's canary questions.
    """
    return tuple(
        CanaryQuestion(
            question_id=str(question["question_id"]),
            prompt=str(question["prompt"]),
            reference_ppm=int(question["reference_ppm"]),
        )
        for question in entry["questions"]
    )


def _read_key(provider: str) -> str:
    """Read a provider's live API key from the environment, or exit clearly.

    Args:
        provider: The provider whose ``<PROVIDER>_API_KEY`` variable is read.

    Returns:
        The API key value (never logged or printed).
    """
    env_var = provider.upper() + _ENV_VAR_SUFFIX
    try:
        return os.environ[env_var]
    except KeyError:
        sys.exit(f"error: environment variable {env_var} is not set")


def _build_observer(
    entry: Mapping[str, Any], *, record: bool
) -> ProviderCanaryObserver:
    """Build one provider's observer for the selected mode.

    Args:
        entry: One provider entry from the spec file.
        record: Whether to build a live (record) observer or a replay one.

    Returns:
        The provider's observer.
    """
    if not record:
        return _ReplayObserver(_observation_from_payload(entry["observation"]))
    provider = str(entry["provider"])
    host = str(entry["host"])
    # NOTE: this allowlist is derived from the same operator-authored spec-file
    # entry that supplies ``endpoint``, so it is NOT an SSRF control -- it only
    # catches an internal host/endpoint typo within a trusted spec file (running
    # this script already requires that trust level). It must not be mistaken
    # for a security boundary; the production connector's config-derived
    # allowlist (windbreak/net/allowlist.py) is the real egress control.
    return _LiveObserver(
        endpoint=str(entry["endpoint"]),
        headers={
            _AUTH_HEADER_NAME: _read_key(provider),
            _CONTENT_TYPE_HEADER: _JSON_CONTENT_TYPE,
        },
        allowlist=OutboundAllowlist(frozenset({host})),
        session=requests.Session(),
    )


def _build_specs(
    payload: Mapping[str, Any], *, record: bool
) -> tuple[ProviderCanarySpec, ...]:
    """Build the provider canary battery from a decoded spec payload.

    Args:
        payload: The decoded spec file, carrying a ``providers`` list.
        record: Whether to build live (record) or replay observers.

    Returns:
        One :class:`ProviderCanarySpec` per provider entry, in file order.
    """
    return tuple(
        ProviderCanarySpec(
            provider=str(entry["provider"]),
            questions=_questions_from_entry(entry),
            pinned_versions=tuple(str(version) for version in entry["pinned_versions"]),
            observer=_build_observer(entry, record=record),
        )
        for entry in payload["providers"]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the canary driver's command-line arguments.

    Args:
        argv: The argument vector, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec-file",
        required=True,
        type=Path,
        help="JSON spec file describing the provider canary battery.",
    )
    parser.add_argument(
        "--ledger-path",
        required=True,
        type=Path,
        help="SQLite ledger the verdicts are appended to.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Dial live provider endpoints instead of replaying the spec file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the provider canary battery CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when every provider stayed within band, else ``1``. An
        out-of-range/non-integer observation or malformed JSON
        (:class:`ValueError`), a structurally malformed spec missing a required
        key (:class:`KeyError`), or a ledger-chain failure
        (:class:`~windbreak.ledger.store.ChainIntegrityError`) is reported to
        stderr and mapped to exit ``1`` -- a clean fail-closed signal rather
        than a raw traceback, mirroring
        :func:`windbreak.ledger.rebuild.rebuild_command`.
    """
    args = _parse_args(argv)
    try:
        payload = json.loads(args.spec_file.read_text(encoding="utf-8"))
        specs = _build_specs(payload, record=args.record)
        return run_canaries(
            specs,
            ledger_path=args.ledger_path,
            alerts=_StdoutAlertEmitter(),
            output=sys.stdout,
            checked_at=datetime.now(tz=UTC),
        )
    except (ChainIntegrityError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
