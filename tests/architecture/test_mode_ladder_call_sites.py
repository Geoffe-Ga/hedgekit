"""The mode ladder's promotion/demotion seams have no production caller (#542).

Five seams that the operating-mode ladder is built out of -- the evidence
snapshot, the two kernel entrypoints that consume it, the anchoring producer
that is supposed to feed it, and the divergence monitor that is supposed to
demote off it -- are called from **nowhere** under ``windbreak/``. They are
constructed, exercised and asserted on exclusively by the test suite.

That is not obviously a bug. ``modes.py`` says "the ladder exists so that
trading is *earned*", and a system that promotes itself to ``LIVE`` on computed
evidence with no human in the loop is a materially different product from one
that computes evidence and presents it to an operator who acts. Issue #542 asks
the repository owner to decide which of those this is; **this module decides
nothing**. It pins the state so that the decision cannot be made by accident --
by a caller appearing in a diff whose reviewer never knew a decision was owed.

WHAT THIS SCANNER IS, AND IS NOT

It counts **call sites**, found by walking the shipped package's AST, and
deliberately nothing else. Three distinctions it draws that a ``grep`` cannot,
each of which would otherwise produce a false positive here:

* **Prose is not a call.** ``promotion.py`` and ``live_divergence.py`` name
  these seams in docstrings; a textual count would report those as callers.
* **A type annotation is not a call.** ``process.py:452`` annotates a parameter
  ``evidence: GateEvidence``; ``evidence.py`` both accepts and returns one.
* **An import is not a call.** ``riskkernel/__init__.py`` re-exports all five,
  and ``__all__`` lists them as strings.

It is a structural backstop, not a behavioural test. It cannot tell a
deliberately-unwired seam from a not-yet-wired one -- no scanner can, because
that distinction lives in the owner's intent and not in the syntax tree. What it
can do is make the transition from one to the other *loud*, which is the whole
of its job.

KNOWN BLIND SPOT

A ``GateEvidence`` can also be produced without naming the class, via
:func:`dataclasses.replace` -- and ``evidence.py:145`` does exactly that. That
site is covered transitively rather than directly: its only enclosing function,
``anchor_gate_evidence``, is itself pinned at zero callers below, so the replace
cannot run in production either. Wire ``anchor_gate_evidence`` and this test
fails, which is the intended alarm.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: The shipped package this scanner walks. Production only -- ``tests/`` calls
#: every one of these seams directly and must keep working.
_PACKAGE = Path(__file__).resolve().parents[2] / "windbreak"

#: The smallest number of production modules a real sweep of ``windbreak/`` can
#: plausibly visit (181 at the time of writing). A scan over zero files reports
#: "no callers" forever, including on the day the package moves and the glob
#: stops matching; this floor makes that failure mode loud instead of green.
_CORPUS_FLOOR = 120

#: The modules whose seams are pinned here. Each must actually be visited by the
#: sweep -- a walker that silently skipped ``process.py`` would report zero
#: callers of ``request_promotion`` for the most boring possible reason.
_REQUIRED_MODULES = (
    "windbreak/riskkernel/promotion.py",
    "windbreak/riskkernel/demotion.py",
    "windbreak/riskkernel/process.py",
    "windbreak/riskkernel/evidence.py",
    "windbreak/evaluation/live_divergence.py",
)

#: The five mode-ladder seams with no production caller, each mapped to the
#: module that defines it (for the failure message -- the reader who trips this
#: needs to know what they just connected, not go looking for it).
_UNWIRED_SEAMS: dict[str, str] = {
    "GateEvidence": "windbreak/riskkernel/promotion.py",
    "request_promotion": "windbreak/riskkernel/process.py",
    "fire_demotion_trigger": "windbreak/riskkernel/process.py",
    "anchor_gate_evidence": "windbreak/riskkernel/evidence.py",
    "monitor_live_divergence": "windbreak/evaluation/live_divergence.py",
}


def _called_name(call: ast.Call) -> str | None:
    """Return the bare name a call invokes, ignoring any receiver.

    ``kernel.request_promotion(e)`` and a bare ``request_promotion(e)`` both
    answer ``"request_promotion"``, so a seam reached through an attribute --
    which is how every one of these would be called in practice -- is caught.

    Args:
        call: The call node to name.

    Returns:
        The invoked name, or ``None`` for a call with no static name (a
        subscript or a call of a call).
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _scan(root: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Walk ``root`` for Python call sites, keyed by invoked name.

    Args:
        root: The package directory to sweep recursively.

    Returns:
        A ``(modules, sites)`` pair: every module visited as a POSIX path
        relative to ``root``'s parent, and a mapping of invoked name to the
        ``"<module>:<line>"`` label of each site calling it.
    """
    modules: list[str] = []
    sites: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).as_posix()
        modules.append(relative)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name is not None:
                sites.setdefault(name, []).append(f"{relative}:{node.lineno}")
    return modules, sites


def test_the_sweep_visits_a_real_corpus_including_every_pinned_module() -> None:
    """The scan is over real files, so "no callers" means something.

    Asserted before any seam count, because every assertion below is vacuously
    satisfiable by a sweep that visited nothing.
    """
    modules, _sites = _scan(_PACKAGE)

    assert len(modules) >= _CORPUS_FLOOR, (
        f"the production sweep visited only {len(modules)} modules, below the "
        f"floor of {_CORPUS_FLOOR}; the mode-ladder call-site pin (#542) is "
        f"reporting on an empty or truncated corpus, not on the package"
    )
    assert set(_REQUIRED_MODULES) <= set(modules)


def test_the_scanner_finds_a_real_call_when_one_exists(tmp_path: Path) -> None:
    """Positive control: the detector fires on a synthetic caller.

    Without this, "zero production call sites" is indistinguishable from a
    detector that cannot count -- a broken walker and a genuinely unwired seam
    produce byte-identical output. The synthetic module calls all five seams,
    two of them through an attribute, and surrounds them with the three
    non-call forms (import, annotation, docstring prose) the real modules
    contain, so this pins the discrimination and not merely the counting.
    """
    package = tmp_path / "windbreak"
    package.mkdir()
    (package / "synthetic.py").write_text(
        '"""Prose naming GateEvidence and request_promotion, which are not calls."""\n'
        "from windbreak.riskkernel.promotion import GateEvidence\n"
        "\n"
        "def caller(kernel, evidence: GateEvidence, store):\n"
        '    """Doc mentioning fire_demotion_trigger, also not a call."""\n'
        "    kernel.request_promotion(GateEvidence(forecast_count=50))\n"
        "    kernel.fire_demotion_trigger(None)\n"
        "    anchor_gate_evidence(evidence, store, now=int)\n"
        "    monitor_live_divergence(None)\n",
        encoding="utf-8",
    )

    modules, sites = _scan(package)

    assert modules == ["windbreak/synthetic.py"]
    found = {seam: len(sites.get(seam, [])) for seam in _UNWIRED_SEAMS}
    assert found == {
        "GateEvidence": 1,
        "request_promotion": 1,
        "fire_demotion_trigger": 1,
        "anchor_gate_evidence": 1,
        "monitor_live_divergence": 1,
    }


@pytest.mark.parametrize("seam", sorted(_UNWIRED_SEAMS))
def test_the_seam_still_has_no_production_caller(seam: str) -> None:
    """No shipped code calls this seam, so the ladder cannot move itself.

    Tripping this test is not a failure -- it means someone connected the
    ladder, which may well be right. It means the decision issue #542 asks for
    has been made, and needs recording rather than merging silently.
    """
    _modules, sites = _scan(_PACKAGE)
    callers = sites.get(seam, [])

    assert callers == [], (
        f"{seam} (defined in {_UNWIRED_SEAMS[seam]}) now has "
        f"{len(callers)} production call site(s): {', '.join(callers)}. "
        f"This seam had none, which is what made the promotion/demotion ladder "
        f"inert. Connecting it decides -- for real money -- whether this system "
        f"promotes itself or asks an operator first. That decision is issue "
        f"#542's AC1 and belongs to the repository owner: record it there (or "
        f"in an ADR), then update this pin in the same change."
    )
