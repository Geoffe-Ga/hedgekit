"""Golden determinism harness for windbreak.selector.select (issues #43/#44/#45).

The whole point of this module: `select` + `serialize_decision` over a fixed
input must produce byte-identical output every time it is called, in-process
or from a brand-new interpreter, regardless of `PYTHONHASHSEED`. Four layers
of proof, from weakest to strongest:

1. `test_select_is_byte_identical_on_recorded_inputs` -- the issue's own
   verbatim claim: two in-process calls over the same recorded bundle agree.
2. `test_bundle_a_decision_emits_exactly_one_named_yes_buy_intent` -- pins the
   *shape* of issue #44's real selection logic on bundle A: SPEC S9.1-S9.3
   evaluate every named condition into `reasons` (never silently empty) and,
   because bundle A hand-verifiably passes all twelve (see the per-condition
   values in `tests/selector/fixtures/bundle_a.json` against
   `tests/selector/test_entry_conditions.py`'s baseline scenario), emit
   exactly one `"yes"`/`"buy"` intent -- superseding the issue-#43 stub's
   "always zero intents" pin now that real selection logic exists (SPEC
   S9.2-S9.3, issue #44).
3. `test_serialized_output_matches_committed_golden` -- compares against a
   *committed* golden file, so a silent change to field order/formatting is
   caught even if it happens to still be self-consistent.
4. `test_fresh_interpreter_produces_identical_bytes` -- runs `select` in a
   brand-new `python -c` subprocess (a different `PYTHONHASHSEED` than this
   process) and diffs its stdout bytes against the in-process serialization,
   ruling out any accidental dependence on hash-seed-influenced iteration
   order (e.g. an un-sorted dict/set) that a same-process comparison could
   never expose.

Golden newline convention (hook-stability): ``serialize_decision`` returns
canonical JSON with **no** trailing newline, but this repo's own
``end-of-file-fixer`` pre-commit hook requires every committed text file to
end in exactly one ``"\n"``. To keep a freshly regenerated golden stable
under that hook (rather than exempting the fixtures, which would weaken the
hook), each ``fixtures/*.golden`` file stores the serialized bytes **plus a
single trailing ``"\n"``**. The comparison below re-appends that newline to
the live serialized output before diffing, and
`test_committed_golden_is_hook_stable` pins the invariant so a regenerated
golden is always exactly ``serialize_decision(...) + "\n"``.

Golden-file regeneration: the two `fixtures/*.golden` files committed
alongside this module hold real `select` output, regenerated from an actual
`select`/`serialize_decision` run (never hand-fabricated) -- bundle A at the
post-#45 sized shape (see the issue-#45 note below), bundle B at the #44 shape
(it declines before sizing). Bundle A hand-verifiably passes all twelve SPEC
S9.3 conditions (per
`test_bundle_a_decision_emits_exactly_one_named_yes_buy_intent` above) and so
emits one real, Kelly-sized intent; bundle B declines for four named
reasons -- `net_edge_
min` (gross edge is already negative: probability 310_000 ppm vs. the walked
executable price 320_000 ppm), `annualized_hurdle` (the resulting negative net
edge annualizes below the configured hurdle), `ci_straddles_executable_price`
(its CI [250_000, 380_000] straddles the 320_000 ppm executable price), and
`forecast_live_eligible` (`eligible_for_live=False`, forced by its
`abstention_reason`). Regenerate either golden the same way after any change
to `select`'s logic or these bundles' fixtures -- never hand-fabricate the
bytes, and always append the single trailing newline the convention requires:

    python -c "
    from tests.selector.fixture_loader import load_inputs
    from windbreak.selector import select, serialize_decision
    path = 'tests/selector/fixtures/bundle_a'
    bytes_ = serialize_decision(select(load_inputs(path + '.json')))
    open(path + '.golden', 'w').write(bytes_ + '\n')
    "

(substitute ``bundle_b`` for the second file).

Issue #45 (dispersion-scaled Kelly sizing): bundle A's ``positions`` object
(equity/capital huge, every exposure and ``notional_today`` zero -- see
``tests/selector/fixtures/bundle_a.json``) is deliberately generous enough
that none of the five notional caps or the participation cap ever bind, so
Kelly sizing alone determines bundle A's emitted size. Hand computation, off
the probe-size (100-centi) figures already pinned in
``test_entry_conditions.py``'s bundle-A cross-reference (probability
620_000 ppm, executable price 460_000 ppm, `net_edge_ppm` 135_000 ppm --
110_000 before issue #483 removed the research haircut that had been charging
bundle A's whole 25_000-micro research cost against a single probe contract):

    g = dispersion_scale(vote_dispersion_ppm=40_000, ceiling=200_000)
      = divide((200_000-40_000)*1_000_000, 200_000, floor) = 800_000
    stake_micros = divide(100_000_000*135_000*100_000*800_000,
                           540_000*10**12, floor)
                 = divide(1.08*10**24, 5.4*10**17, floor) = 2_000_000
    raw_size_centis = divide(2_000_000*100, 460_000, floor)
                    = divide(200_000_000, 460_000, floor) = 434
        (460_000*434=199_640_000; remainder=360_000 < 460_000)
    No cap binds (`binding_cap=None`) -> floor-to-100 quantization takes
    434 -> 400 -- bundle A's real emitted size is `ContractCentis(400)`,
    priced at the re-walked marginal 4_600 pips (still within the book's
    now-deepened first level; unchanged from the probe's own marginal price,
    since 400 <= 50_000).

`bundle_a.golden` has been regenerated (via the command above) to this
sized shape (`intent_id` suffix `:sized`, `size=400`, thirteen
reasons ending in the pinned ``sizing: raw_centis=434 g_ppm=800000
binding_cap=none final_centis=400`` line), so
`test_serialized_output_matches_committed_golden[bundle_a]` and
`test_fresh_interpreter_produces_identical_bytes[bundle_a]` pass against it.

Issue #46 (execution style) adds three serialized fields to bundle A's emitted
intent -- ``execution_style`` and the ``resting_ttl_seconds`` /
``cancel_on_move_ticks`` pair -- so the regenerated `bundle_a.golden` intent
object now also carries ``"execution_style":"cross"`` with both resting fields
serialized as JSON ``null`` (bundle A's recorded book has a 200-pip spread,
narrower than the 300-pip wide-spread floor, so it crosses rather than rests).
The intent's price (4600), size (300), idempotency key, and all thirteen
reasons are byte-unchanged; only the three sorted keys are new.

`bundle_b.golden` still emits no intent -- bundle B declines at the entry
conditions, before sizing runs -- but issue #483 did move two numbers inside
its reasons: with the research haircut gone its net edge reads -48_000 ppm
rather than -146_000, and the annualization of that edge -18_250_000 ppm rather
than -55_510_417. It still fails all four of the same named conditions, which
is the point: dropping the haircut moved the arithmetic, not the verdict, on a
market that genuinely does not clear the bar. Regenerate either golden the same
way (never hand-fabricate) after any change to `select`'s logic or these
bundles' fixtures.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.selector.fixture_loader import load_inputs
from windbreak.selector import select, serialize_decision

if TYPE_CHECKING:
    from windbreak.selector import SelectorInputs

#: This package's own committed bundle fixtures.
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: The repository root, computed from this file's location
#: (`tests/selector/test_determinism_golden.py` -> `tests/selector` ->
#: `tests` -> repo root), so the fresh-interpreter subprocess below can
#: import both `windbreak` and `tests.selector.fixture_loader` regardless of
#: its own working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Hard timeout for the fresh-interpreter subprocess (house rule: no
#: unbounded waits). A bare `python -c` importing a handful of modules and
#: running a stub selector comfortably finishes in well under this budget.
_SUBPROCESS_TIMEOUT_S = 30

#: Forced subprocess `PYTHONHASHSEED`, deliberately different from whatever
#: (random, by default) seed this parent test process is running under --
#: proving byte-identical output does not depend on hash-seed-influenced
#: iteration order.
_SUBPROCESS_HASHSEED = "12345"

#: All committed bundle names this module's parametrized tests iterate over.
_BUNDLE_NAMES: tuple[str, ...] = ("bundle_a", "bundle_b")


@pytest.mark.parametrize("bundle_name", _BUNDLE_NAMES)
def test_select_is_byte_identical_on_recorded_inputs(bundle_name: str) -> None:
    """`serialize_decision(select(bundle)) == serialize_decision(select(bundle))`.

    The issue's own verbatim determinism claim: calling `select` twice
    in-process over the identical recorded bundle must yield byte-identical
    serialized output.
    """
    inputs = load_inputs(_FIXTURES_DIR / f"{bundle_name}.json")

    assert serialize_decision(select(inputs)) == serialize_decision(select(inputs))


def test_bundle_a_decision_emits_exactly_one_named_yes_buy_intent(
    recorded_inputs_bundle_a: SelectorInputs,
) -> None:
    """Bundle A hand-verifiably passes all twelve SPEC S9.3 conditions (see
    this module's docstring), so `select` must emit exactly one intent -- on
    the `"yes"` outcome, a `"buy"` action -- and a non-empty, named set of
    reasons. Supersedes the issue-#43 stub's "always zero intents" pin now
    that real selection logic exists (SPEC S9.2-S9.3, issue #44).
    """
    decision = select(recorded_inputs_bundle_a)

    assert len(decision.intents) == 1
    intent = decision.intents[0]
    assert intent.outcome == "yes"
    assert intent.action == "buy"
    assert decision.reasons
    assert all(reason for reason in decision.reasons)


@pytest.mark.parametrize("bundle_name", _BUNDLE_NAMES)
def test_serialized_output_matches_committed_golden(bundle_name: str) -> None:
    """`serialize_decision(select(bundle)) + "\\n"` equals the committed golden.

    See this module's docstring for the newline convention and the exact
    regeneration command: the golden stores the serialized bytes plus one
    trailing newline (so it is stable under `end-of-file-fixer`), so the live
    serialized output is compared with that newline re-appended.
    """
    inputs = load_inputs(_FIXTURES_DIR / f"{bundle_name}.json")
    golden_path = _FIXTURES_DIR / f"{bundle_name}.golden"

    actual = serialize_decision(select(inputs))

    assert actual + "\n" == golden_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("bundle_name", _BUNDLE_NAMES)
def test_committed_golden_is_hook_stable(bundle_name: str) -> None:
    """Each committed golden ends in exactly one newline and no more.

    Pins the newline convention documented in this module's docstring: a
    golden is exactly ``serialize_decision(...) + "\\n"``. This guarantees a
    freshly regenerated golden survives this repo's `end-of-file-fixer`
    pre-commit hook unchanged -- ending in exactly one trailing ``"\\n"`` --
    so the fixtures never need a hook exemption and CI's "Pre-commit (all
    files)" step can never be tripped by a golden's missing/extra newline.
    """
    golden_text = (_FIXTURES_DIR / f"{bundle_name}.golden").read_text(encoding="utf-8")

    assert golden_text.endswith("\n")
    assert not golden_text.endswith("\n\n")
    assert golden_text.removesuffix("\n") == golden_text.rstrip("\n")


@pytest.mark.parametrize("bundle_name", _BUNDLE_NAMES)
def test_fresh_interpreter_produces_identical_bytes(bundle_name: str) -> None:
    """A brand-new interpreter (different `PYTHONHASHSEED`) reproduces the
    identical serialized bytes this process produces in-process.

    Rules out any accidental dependence on hash-seed-influenced iteration
    order (e.g. an un-sorted dict/set surviving into the serialized form).
    """
    bundle_path = _FIXTURES_DIR / f"{bundle_name}.json"
    in_process_bytes = serialize_decision(select(load_inputs(bundle_path))).encode(
        "utf-8"
    )

    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        "from tests.selector.fixture_loader import load_inputs\n"
        "from windbreak.selector import select, serialize_decision\n"
        f"inputs = load_inputs({str(bundle_path)!r})\n"
        "sys.stdout.write(serialize_decision(select(inputs)))\n"
    )
    subprocess_env = dict(os.environ)
    subprocess_env["PYTHONHASHSEED"] = _SUBPROCESS_HASHSEED

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        env=subprocess_env,
        capture_output=True,
        text=False,
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=True,
    )

    assert result.stdout == in_process_bytes
