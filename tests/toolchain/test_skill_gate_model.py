"""The skills corpus states the gate model this repository actually runs (#536).

A skill file is loaded into an agent's context *ahead of* the documents it
contradicts, so when `.claude/skills/stay-green/SKILL.md` and
`.claude/docs/workflow.md` answer "what are the gates?" differently, the skill's
answer is the one that gets followed. Before this module,
`.claude/skills/stay-green/SKILL.md` described a **two-gate** model (its own TDD
loop as "Gate 1", pre-commit as "Gate 2") and listed "formatting (Black +
isort)" as a quality check -- a gate count that stops one gate short of the
review gate, and two formatters ADR-0002 removed from every gate in this
repository.

That is the defect class this repository has now removed five times (#351,
#359, #401, #411, #534): policy text believed exactly as readily as policy
enforcement, because nothing measures it. `>=9.0 pylint` was quoted as a hard
constraint in agent briefs for weeks on the strength of one table row.

WHAT IS ACTUALLY VERIFIED HERE, and nothing beyond it:

* The automated gates the skill names are exactly the gates
  `.claude/docs/workflow.md` publishes under its `### 2.1 The Gates` heading --
  same ordinals, same names -- both sides DERIVED, so adding a fourth gate to
  the document turns the skill red rather than leaving it a gate behind.
* The skill links to that section and to the manual pre-v1.0.0 mutation gate by
  anchors derived from their headings, so renaming either heading breaks the
  pointer rather than silently orphaning it. Linking is what keeps the gate
  descriptions from being restated in a second file and diverging there.
* No document under `.claude/skills/` names a formatter ADR-0002 retired. The
  retired set is `tests.toolchain.test_toolchain_pins._BANNED_FORMATTERS`, the
  repository's one declaration of it, whose absence from pre-commit and
  `requirements-dev.txt` that module already asserts -- so "retired" here is a
  verified fact about the toolchain, not a claim this module makes.
* Every quality tool the stay-green skill names is one the repository installs.
  The installed side is derived from `constraints-quality.txt`,
  `requirements-dev.txt` and `.pre-commit-config.yaml`.

WHAT IS NOT VERIFIED. The lexicon of names that COUNT as a tool is written
down here (:data:`_UNPINNED_TOOL_NAMES`), because no derivation can tell a tool
name from an English word. It bounds only what is looked for; every verdict
about a name found is derived. A skill naming a tool that is in neither the
lexicon nor the toolchain is therefore invisible to this module -- a hole made
as small as the two known sources of such names allow (this repository's pins,
and the tools its own ADRs and issues record retiring). Two mentions are
deliberately not failures anywhere in this module: `# pylint: disable=` in
`max-quality-no-shortcuts`, which is an example of a suppression not to write,
and the cross-language tool list in `de-slopify`, which describes a scan and not
this repository's gates. The corpus-wide assertion is therefore about retired
FORMATTERS specifically -- ADR-0002 is a repository-wide decision with no
context in which naming black or isort as current practice is correct.

The prose in the skill is prose. This module does not parse English.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.toolchain.test_live_state_gate_exception import _anchor, _section
from tests.toolchain.test_toolchain_pins import (
    _BANNED_FORMATTERS,
    _normalize,
    _parse_constraints,
    _precommit_hook_ids,
    _requirement_names,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILLS_DIR = _REPO_ROOT / ".claude" / "skills"
_STAY_GREEN_SKILL = _SKILLS_DIR / "stay-green" / "SKILL.md"
_WORKFLOW_DOC = _REPO_ROOT / ".claude" / "docs" / "workflow.md"
_REQUIREMENTS_DEV = _REPO_ROOT / "requirements-dev.txt"

#: The heading opening the canonical gate list. Matched exactly, for the same
#: reason `test_live_state_gate_exception` matches its own: the anchor the skill
#: links to is DERIVED from this string, so renaming the heading must break the
#: pointer rather than orphan it.
_GATES_HEADING = "### 2.1 The Gates"

#: The heading opening the manual pre-v1.0.0 mutation gate. It is deliberately
#: NOT one of the automated gates -- owner directive, issue #107 -- and a skill
#: that folds it in with the other three misstates when it runs.
_MANUAL_GATE_HEADING = "### 2.1a Manual Pre-Release Gate: Mutation Testing (≥80%)"

#: A gate in the canonical list: `1. **Gate 1: Local Pre-Commit** (...)`.
_CANONICAL_GATE = re.compile(
    r"^\d+\.\s+\*\*Gate (?P<ordinal>\d+): (?P<name>[^*]+)\*\*", re.MULTILINE
)

#: Any claim about a numbered gate, wherever it appears in a document --
#: including a skill's YAML frontmatter, which is the part loaded into every
#: agent's context whether or not the body is ever read.
_GATE_MENTION = re.compile(r"Gate (?P<ordinal>\d+)")

#: Tool names this repository's own record says it does not run, so that naming
#: one is a claim about a gate that does not exist. `black`/`isort`/`autoflake`/
#: `pyupgrade`/`tryceratops`/`refurb` are ADR-0002's removals; `pylint` is
#: #411's; `interrogate` and `pydocstyle` are #351's. The rest are the common
#: Python formatters and linters a stale document reaches for. Membership here
#: is not what makes a name a failure -- being absent from the toolchain is.
_UNPINNED_TOOL_NAMES = frozenset(
    {
        "autoflake",
        "autopep8",
        "flake8",
        "interrogate",
        "pydocstyle",
        "pylint",
        "pyupgrade",
        "refurb",
        "tryceratops",
        "yapf",
    }
    | _BANNED_FORMATTERS
)

#: Lower bound on the markdown documents under `.claude/skills/`. The corpus
#: held 21 when this was written; a floor well under that catches a walk that
#: has stopped walking (trap #5 -- a scan over zero hits passes forever) without
#: failing every time a reference file is deleted.
_SKILL_DOC_FLOOR = 10

#: Lower bound on the pinned tools the stay-green skill must name. Its quality
#: checks list names ruff, mypy, bandit, xenon, pytest and coverage today; a
#: floor of three catches a detector that has stopped matching, which would make
#: "every tool it names is installed" pass over an empty set.
_NAMED_TOOL_FLOOR = 3

#: A skill stating the model this issue removed: its own TDD loop as Gate 1,
#: pre-commit as Gate 2, and no gate after them. Used as the positive control
#: for the gate detector, so the assertions below cannot all be passing because
#: the detector stopped finding gates at all.
_STALE_SKILL_CONTROL = """---
name: stay-green
description: >-
  2-gate TDD development workflow: Gate 1 is Red-Green-Refactor testing,
  Gate 2 is pre-commit quality checks.
---

### Gate 1: TDD (Red-Green-Refactor)
### Gate 2: Pre-Commit Quality Checks

Quality checks include: formatting (Black + isort), linting (Ruff).
"""

#: The same claim written against the current model. The negative control for
#: the retired-formatter detector: naming ruff format is not naming black.
_CURRENT_SKILL_CONTROL = """---
name: stay-green
description: Three automated gates plus a manual pre-v1.0.0 mutation gate.
---

### Gate 1: Local Pre-Commit
### Gate 2: CI Pipeline
### Gate 3: Code Review

Quality checks include: formatting (ruff format), linting (ruff), types (mypy).
"""


def _canonical_gates() -> dict[int, str]:
    """Read the automated gates the workflow document publishes.

    Returns:
        Ordinal to gate name, e.g. `{1: "Local Pre-Commit", ...}`.
    """
    section = _section(_WORKFLOW_DOC, _GATES_HEADING)
    return {
        int(match.group("ordinal")): match.group("name").strip()
        for match in _CANONICAL_GATE.finditer(section)
    }


def _gate_ordinals(text: str) -> set[int]:
    """Collect every numbered gate a document claims exists.

    Args:
        text: The document's text.

    Returns:
        The ordinals it names.
    """
    return {int(match.group("ordinal")) for match in _GATE_MENTION.finditer(text)}


def _installed_tools() -> frozenset[str]:
    """Derive the quality tools this repository actually installs and runs.

    Three sources, because a tool can be wired through any of them: the version
    authority (`constraints-quality.txt`), the dev-dependency name authority
    (`requirements-dev.txt`), and the hook set Gate 1 dispatches in full
    (`.pre-commit-config.yaml`, issue #401).

    Returns:
        Normalized tool names.
    """
    return frozenset(
        _normalize(name)
        for name in (
            *_parse_constraints(),
            *_requirement_names(_REQUIREMENTS_DEV.read_text(encoding="utf-8")),
            *_precommit_hook_ids(),
        )
    )


def _tool_lexicon() -> frozenset[str]:
    """List every name this module recognizes as a tool.

    Returns:
        The installed tools plus the names this repository records retiring.
    """
    return _installed_tools() | _UNPINNED_TOOL_NAMES


def _named_tools(text: str, lexicon: frozenset[str]) -> frozenset[str]:
    """Find the tools a document names.

    Matching is bounded by word characters AND hyphens on both sides, so
    `pytest-cov` is one name rather than a mention of `pytest`, and a phrase
    like `black-box testing` is not a mention of `black`.

    Args:
        text: The document's text.
        lexicon: The names to look for.

    Returns:
        The subset of `lexicon` the document names.
    """
    return frozenset(
        name
        for name in lexicon
        if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text, re.IGNORECASE)
    )


def _skill_documents() -> tuple[Path, ...]:
    """List every markdown document under `.claude/skills/`.

    Returns:
        The documents, in sorted path order.
    """
    return tuple(sorted(_SKILLS_DIR.rglob("*.md")))


def test_the_canonical_gate_list_is_parsed_and_contiguous() -> None:
    """The gates are read out of the workflow document, and there are some.

    Every assertion below compares the skill against this mapping. An empty one
    -- a renamed heading, a reformatted list -- would make "the skill names
    exactly these gates" a claim about nothing, which is the failure mode this
    whole issue is about.
    """
    gates = _canonical_gates()

    assert gates, (
        f"no `N. **Gate N: Name**` entries found under '{_GATES_HEADING}' in "
        f"{_WORKFLOW_DOC}. The gate model is the thing the skill is checked "
        "against; without it every comparison below passes over an empty set."
    )
    assert sorted(gates) == list(range(1, len(gates) + 1)), (
        f"the document's gates are numbered {sorted(gates)}, which is not a "
        "contiguous run from 1. A gap means a gate was removed without "
        "renumbering, and 'the skill names the same ordinals' would then "
        "enforce the gap."
    )


def test_the_manual_mutation_gate_is_not_one_of_the_automated_gates() -> None:
    """Mutation testing has its own section, deliberately outside the run.

    Per the owner directive in issue #107 it is a manual pre-v1.0.0 release
    gate, never an automated check. If it were ever folded into the numbered
    list, the skill would be right to count it -- so this reads the document
    rather than assuming.
    """
    manual_section = _section(_WORKFLOW_DOC, _MANUAL_GATE_HEADING)

    assert manual_section.strip(), (
        f"{_WORKFLOW_DOC} has no '{_MANUAL_GATE_HEADING}' section, so the "
        "anchor the skill links to leads nowhere."
    )
    assert not _CANONICAL_GATE.search(manual_section), (
        "the manual mutation gate section now carries a numbered `**Gate N:** "
        "entry, so it reads as one of the automated gates. Either it has been "
        "promoted into the run -- in which case say so in the numbered list -- "
        "or the entry is a formatting accident."
    )


def test_the_gate_detector_fires_on_the_two_gate_model() -> None:
    """The positive control: the detector reports the model this issue removed.

    `test_the_stay_green_skill_names_exactly_the_automated_gates` compares two
    derived sets. A detector that finds no gates in either would make them
    equal and the assertion vacuous, so it is exercised here against a skill
    that is known-wrong in the exact way the real one was.
    """
    stale = _gate_ordinals(_STALE_SKILL_CONTROL)

    assert stale == {1, 2}, (
        f"the detector read {sorted(stale)} from a document naming Gate 1 and "
        "Gate 2 four times over, including in its frontmatter description. It "
        "is not finding gate claims, so comparing it against the document's "
        "gates checks nothing."
    )
    assert stale != set(_canonical_gates()), (
        "the two-gate control matches the current gate model, so it is no "
        "longer a control for anything. If the repository really has moved to "
        "two gates, this module's premise needs rewriting, not this assertion."
    )


def test_the_retired_formatter_detector_reads_names_not_substrings() -> None:
    """The retired-formatter detector fires on a stale tool list, and only there.

    `formatting (Black + isort)` is the exact phrase this issue removed from
    the skill; `black-box` and `blacklist` are the words a substring scan would
    misread as it. Both directions are controlled, because a detector that
    fires on everything gets weakened until it fires on nothing.
    """
    lexicon = _tool_lexicon()

    assert _named_tools(_STALE_SKILL_CONTROL, lexicon) >= _BANNED_FORMATTERS, (
        "the detector did not find black and isort in a document whose quality "
        "checks read `formatting (Black + isort)`. Matching is "
        "case-insensitive precisely because that is how the stale line was "
        "written."
    )
    assert not (_BANNED_FORMATTERS & _named_tools(_CURRENT_SKILL_CONTROL, lexicon)), (
        "the detector reported a retired formatter in a document that names "
        "only `ruff format`, `ruff` and `mypy`."
    )
    assert not _named_tools("black-box testing, blacklisted, blackboard", lexicon), (
        "the detector read `black` out of `black-box`, `blacklisted` or "
        "`blackboard`. A hyphen or a letter on either side means the word is "
        "not the tool, and a detector that cannot tell will be turned off."
    )


def test_the_derived_toolchain_excludes_the_retired_formatters() -> None:
    """The installed side is real, and the retired formatters are really gone.

    "Retired" is not this module's opinion: black and isort are absent from the
    pins, the dev requirements and the hook set, which is what makes naming
    them in a skill a false claim rather than a style preference.
    """
    installed = _installed_tools()

    assert {"ruff", "mypy", "pytest"} <= installed, (
        f"the derived toolchain is {sorted(installed)}, which is missing one "
        "of ruff, mypy or pytest. The derivation has broken, so 'every tool "
        "the skill names is installed' would fail for the wrong reason."
    )
    assert not (_BANNED_FORMATTERS & installed), (
        f"{sorted(_BANNED_FORMATTERS & installed)} is pinned or hooked after "
        "all. ADR-0002 removed it from every gate; if it is back, this "
        "module's corpus-wide assertion is enforcing a decision that has been "
        "reversed."
    )


def test_the_skill_corpus_is_non_empty_and_holds_the_stay_green_skill() -> None:
    """The corpus-wide scan walks a real set of documents.

    A glob that stops matching would leave every corpus assertion below
    trivially satisfied.
    """
    documents = _skill_documents()

    assert len(documents) >= _SKILL_DOC_FLOOR, (
        f"only {len(documents)} markdown documents found under {_SKILLS_DIR}, "
        f"below the floor of {_SKILL_DOC_FLOOR}. The walk is not reaching the "
        "skills tree."
    )
    assert _STAY_GREEN_SKILL in documents, (
        f"{_STAY_GREEN_SKILL} is not in the scanned corpus at all, and it is "
        "the document this module exists for."
    )


def test_no_skill_names_a_formatter_this_repository_retired() -> None:
    """ADR-0002 left exactly one formatter, and the skills say so.

    A skill is loaded ahead of the ADR that contradicts it. `./scripts/
    format.sh` runs `ruff format` and nothing else, so a skill telling an agent
    to expect `black....Failed` is describing a run that cannot happen.
    """
    lexicon = _tool_lexicon()
    offenders = {
        str(path.relative_to(_REPO_ROOT)): sorted(
            _BANNED_FORMATTERS & _named_tools(path.read_text(encoding="utf-8"), lexicon)
        )
        for path in _skill_documents()
    }
    named = {path: tools for path, tools in offenders.items() if tools}

    assert named == {}, (
        f"{named} name formatters ADR-0002 removed from every gate in this "
        "repository. ruff format is the single formatter authority; the "
        "decision and its reasoning belong in "
        "docs/architecture/ADR/0002-single-formatter-ruff-format.md, not "
        "restated -- correctly or otherwise -- in a skill."
    )


def test_the_stay_green_skill_names_exactly_the_automated_gates() -> None:
    """The skill's gate model and the document's agree, in both directions.

    This is the assertion the module exists for. A gate the document publishes
    and the skill does not name is an agent declaring work done a gate early --
    which is what a two-gate skill did, stopping before code review. A gate the
    skill names and the document does not is a gate nobody has to pass.
    """
    documented = set(_canonical_gates())
    claimed = _gate_ordinals(_STAY_GREEN_SKILL.read_text(encoding="utf-8"))

    assert claimed == documented, (
        f"{_STAY_GREEN_SKILL.relative_to(_REPO_ROOT)} names gates "
        f"{sorted(claimed)}; {_WORKFLOW_DOC.relative_to(_REPO_ROOT)}'s "
        f"'{_GATES_HEADING}' publishes {sorted(documented)}.\n  named but not "
        f"a gate: {sorted(claimed - documented)}\n  a gate but not named: "
        f"{sorted(documented - claimed)}\nA skill is read before the document "
        "it disagrees with, so the skill's count is the one that gets acted on."
    )


def test_the_stay_green_skill_calls_each_gate_by_its_canonical_name() -> None:
    """Same ordinals is not enough; the gates must be the same gates.

    A skill can name Gate 1, Gate 2 and Gate 3 while meaning TDD, pre-commit
    and something else entirely -- which is close to what the two-gate version
    did, calling its TDD loop "Gate 1" while the document's Gate 1 is the local
    check-all run that contains it.
    """
    text = _STAY_GREEN_SKILL.read_text(encoding="utf-8")
    missing = [
        f"Gate {ordinal}: {name}"
        for ordinal, name in sorted(_canonical_gates().items())
        if f"Gate {ordinal}: {name}" not in text
    ]

    assert missing == [], (
        f"{missing} appear in {_WORKFLOW_DOC.relative_to(_REPO_ROOT)}'s gate "
        f"list but not in {_STAY_GREEN_SKILL.relative_to(_REPO_ROOT)}. The "
        "skill may summarize a gate, but it has to be naming the same one."
    )


def test_the_stay_green_skill_points_at_the_canonical_gate_sections() -> None:
    """The skill links to the gate list and the manual gate, and does not restate them.

    Anchors are derived from the headings, so renaming a section breaks these
    pointers rather than orphaning them -- the same mechanism
    `test_live_state_gate_exception` uses for the §2.4 exception, and for the
    same reason: two copies of a rule are how the two answers come to differ.
    """
    text = _STAY_GREEN_SKILL.read_text(encoding="utf-8")
    missing = [
        f"workflow.md{_anchor(heading)}"
        for heading in (_GATES_HEADING, _MANUAL_GATE_HEADING)
        if f"workflow.md{_anchor(heading)}" not in text
    ]

    assert missing == [], (
        f"{_STAY_GREEN_SKILL.relative_to(_REPO_ROOT)} does not link to "
        f"{missing}. Either the pointer was dropped, or the canonical heading "
        "was renamed without updating it. A skill that describes the gates "
        "without pointing at their definition is a second copy of the rule."
    )


def test_every_tool_the_stay_green_skill_names_is_one_the_repository_runs() -> None:
    """The skill's quality-check list describes this toolchain, not a generic one.

    The floor is the anti-vacuity half: a detector that has stopped matching
    would report no named tools, and "all of them are installed" would be
    true of the empty set.
    """
    lexicon = _tool_lexicon()
    installed = _installed_tools()
    named = _named_tools(_STAY_GREEN_SKILL.read_text(encoding="utf-8"), lexicon)

    assert len(named & installed) >= _NAMED_TOOL_FLOOR, (
        f"the skill names only {sorted(named & installed)} of this "
        f"repository's tools, below the floor of {_NAMED_TOOL_FLOOR}. A skill "
        "describing the quality checks names the tools that run them; finding "
        "none means the detector is not reading the file."
    )
    assert named <= installed, (
        f"{sorted(named - installed)} are named by "
        f"{_STAY_GREEN_SKILL.relative_to(_REPO_ROOT)} but are pinned in no "
        "constraints file, listed in no requirements file and wired into no "
        "pre-commit hook. An agent told to expect them will read their absence "
        "as a broken environment."
    )
