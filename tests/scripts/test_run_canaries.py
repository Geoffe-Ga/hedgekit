"""Tests for ``scripts/run_canaries.py`` observation parsing (issue #195).

``scripts/run_canaries.py`` lives outside the ``windbreak`` package (it is the
only place provider canaries reach a live network, deliberately kept off the
CI-imported path), so this module loads it by file path via
``importlib.util.spec_from_file_location`` -- mirroring
``tests/connector/test_contract_matrix.py``'s and ``tests/numeric/
test_float_lint.py``'s precedent for exercising operator scripts.

The focus is ``_observation_from_payload``, the single parser shared by BOTH the
offline ``--replay`` path (``_ReplayObserver``) and the live ``--record`` path
(``_LiveObserver``). It must enforce the same strict-integer, ``[0, 1_000_000]``
fail-closed contract as ``windbreak.forecast.canary.parse_observed_ppm``:
reject (never clamp) an out-of-range value, and reject (never silently truncate)
a non-integer/float value, so a malformed observation aborts the run rather than
scoring as OK -- the opposite of the fail-safe direction a bug here would take.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

if TYPE_CHECKING:
    import types
    from collections.abc import Mapping

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_RUN_CANARIES_PATH: Final = _REPO_ROOT / "scripts" / "run_canaries.py"


def _load_run_canaries() -> types.ModuleType:
    """Load ``scripts/run_canaries.py`` as a module by file path.

    Returns:
        The loaded ``run_canaries`` module.
    """
    spec = importlib.util.spec_from_file_location("run_canaries", _RUN_CANARIES_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE: Final = _load_run_canaries()


def test_observation_from_payload_parses_in_range_integers() -> None:
    """A well-formed payload parses to a validated integer observation."""
    observation = _MODULE._observation_from_payload(
        {"observed_ppm": {"q1": 999_999, "q2": 0}, "reported_version": "v1"}
    )

    assert observation.observed_ppm == {"q1": 999_999, "q2": 0}
    assert observation.reported_version == "v1"


def test_observation_from_payload_rejects_out_of_range_value() -> None:
    """An out-of-range ppm is rejected, not clamped (fail-closed).

    The reviewer's concrete failure: observed ``1_000_001`` against a reference
    ``999_999`` yields a drift of only 2 ppm -- well under tolerance -- so a
    clamping/unbounded parser would report ``OK`` on malformed data, inverting
    the fail-safe direction. The parser must raise instead.
    """
    with pytest.raises(ValueError, match=r"1000001|1_000_001|outside"):
        _MODULE._observation_from_payload(
            {"observed_ppm": {"q1": 1_000_001}, "reported_version": "v1"}
        )


def test_observation_from_payload_rejects_negative_value() -> None:
    """A negative ppm is rejected rather than accepted."""
    with pytest.raises(ValueError, match=r"outside"):
        _MODULE._observation_from_payload(
            {"observed_ppm": {"q1": -1}, "reported_version": "v1"}
        )


def test_observation_from_payload_rejects_float_value() -> None:
    """A float ppm is rejected, not silently truncated by ``int()``.

    ``int(0.5)`` would truncate to 0 with no error, masking a genuine
    integration bug; the parser must reject non-integer input.
    """
    with pytest.raises(ValueError, match=r"must be an integer"):
        _MODULE._observation_from_payload(
            {"observed_ppm": {"q1": 0.5}, "reported_version": "v1"}
        )


def test_observation_from_payload_rejects_whole_number_float() -> None:
    """A whole-number float (``500000.0``) is still rejected as non-integer."""
    with pytest.raises(ValueError, match=r"must be an integer"):
        _MODULE._observation_from_payload(
            {"observed_ppm": {"q1": 500_000.0}, "reported_version": "v1"}
        )


def test_observation_from_payload_rejects_boolean_value() -> None:
    """A JSON boolean is rejected, never coerced via ``bool`` being an ``int``.

    ``bool`` is an ``int`` subclass (``int(True) == 1``), so passing the raw
    value to the parser would let ``true`` score as an in-range 1 ppm. The
    parser stringifies first (``str(True) == "True"``), so the boolean is
    rejected as a non-integer -- this test locks that ``str()`` guard so a
    future "simplification" that drops it cannot silently reopen the hole.
    """
    with pytest.raises(ValueError, match=r"must be an integer"):
        _MODULE._observation_from_payload(
            {"observed_ppm": {"q1": True}, "reported_version": "v1"}
        )


def test_observation_from_payload_rejects_non_numeric_string() -> None:
    """A non-integer string value fails closed rather than crashing opaquely."""
    with pytest.raises(ValueError, match=r"must be an integer"):
        _MODULE._observation_from_payload(
            {"observed_ppm": {"q1": "maybe"}, "reported_version": "v1"}
        )


def _replay_spec(*, observed_ppm: Mapping[str, object]) -> dict[str, object]:
    """Build a one-provider replay spec payload for ``main`` end-to-end tests.

    Args:
        observed_ppm: The observed-ppm leaves to embed in the provider's
            replayed observation.

    Returns:
        A decoded spec payload with a single provider entry.
    """
    return {
        "providers": [
            {
                "provider": "acme",
                "questions": [
                    {"question_id": "q1", "prompt": "p1", "reference_ppm": 500_000}
                ],
                "pinned_versions": ["v1"],
                "observation": {
                    "observed_ppm": observed_ppm,
                    "reported_version": "v1",
                },
            }
        ]
    }


def test_main_reports_malformed_observation_as_clean_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed replay observation exits 1 with a stderr message, not a trace.

    Exercises the fail-closed error handling in ``main``: an out-of-range
    ``observed_ppm`` is rejected by the shared parser and surfaced as a clean
    ``error: ...`` line on stderr with exit code 1, mirroring
    ``rebuild_command`` -- never a raw traceback.
    """
    spec_file = tmp_path / "battery.json"
    spec_file.write_text(
        json.dumps(_replay_spec(observed_ppm={"q1": 1_000_001})), encoding="utf-8"
    )
    ledger_path = tmp_path / "ledger.sqlite3"

    exit_code = _MODULE.main(
        ["--spec-file", str(spec_file), "--ledger-path", str(ledger_path)]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ")


def test_main_reports_missing_spec_key_as_clean_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A structurally malformed spec (missing key) exits 1 with a stderr line.

    A ``KeyError`` from a spec file missing a required key is mapped to the same
    clean ``error: ...`` + exit-1 signal as a bad observation, not a raw
    traceback.
    """
    spec_file = tmp_path / "battery.json"
    spec_file.write_text(json.dumps({"providers": [{}]}), encoding="utf-8")
    ledger_path = tmp_path / "ledger.sqlite3"

    exit_code = _MODULE.main(
        ["--spec-file", str(spec_file), "--ledger-path", str(ledger_path)]
    )

    assert exit_code == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_main_returns_zero_for_in_band_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A well-formed, in-band replay battery ledgers a verdict and exits 0."""
    spec_file = tmp_path / "battery.json"
    spec_file.write_text(
        json.dumps(_replay_spec(observed_ppm={"q1": 500_000})), encoding="utf-8"
    )
    ledger_path = tmp_path / "ledger.sqlite3"

    exit_code = _MODULE.main(
        ["--spec-file", str(spec_file), "--ledger-path", str(ledger_path)]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "provider=acme canary=OK" in captured.out
