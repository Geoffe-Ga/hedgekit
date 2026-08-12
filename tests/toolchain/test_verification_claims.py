"""Nothing under `.claude/` verifies a gate with a tool this repo lacks (#543).

`.claude/agents/ralph-documentation-specialist.md` named `interrogate >=85%`
five times as the thing that verifies docstring coverage, and sent the reader to
`scripts/backend/check-all.sh` to run it. Three separate falsehoods, compounding:

* `interrogate` is installed nowhere and pinned nowhere. It was deliberately NOT
  adopted -- `pyproject.toml` argues the decision at length -- because ruff
  already reimplements pydocstyle's rule set natively (issue #351).
* `>=85%` is WEAKER than the rule this repository enforces. Ruff `D1` requires a
  docstring on every public module, class, method and function, with no
  percentage to fall below. An agent satisfying 85% satisfies something strictly
  looser than Gate 1, and would report DOCUMENTED with `D1` findings open.
* `scripts/backend/check-all.sh` does not exist. The script is
  `scripts/check-all.sh` (the `scripts/backend/` drift is issue #142).

That is the #411 shape with a twist: `pyproject.toml` records THIS FILE as one
of the sources that seeded #411's phantom Pylint row -- "the
ralph-documentation-specialist agent said `interrogate >=85%`". The row was
deleted from `CLAUDE.md`; the agent definition that helped seed it was left
behind, and an agent definition is a subagent's operating instructions, loaded
ahead of anything it might otherwise go and check.

THE CORPUS. `CLAUDE.md` plus every markdown document under `.claude/` --
`docs/`, `skills/`, `agents/` and `commands/`. PR #544 widened
`test_live_state_gate_exception`'s corpus from `CLAUDE.md` + `.claude/docs/` to
include `.claude/skills/`, on the argument that a governing corpus should cover
where a rule is READ and not only where it is written. The same argument reaches
`.claude/agents/` with more force, so the corpus here is simply everything: no
cherry-picking, no per-directory exception.

THE DISTINCTION THAT MAKES THAT AFFORDABLE, and it is the hard part of this
issue. A corpus-wide "every tool named must be installed" rule does not hold and
should not: `.claude/agents/shared/house-rules.md` names `# pylint: disable=` as
an example of a suppression NOT to write, and `.claude/skills/de-slopify/`
describes a cross-language toolbox for repositories that are not this one.
Neither is a false claim about this repository's gates. So what is looked for is
not a MENTION of a tool but a CLAIM that the tool VERIFIES something -- a name
carrying an obligation, in one of three machine-recognisable forms:

* a THRESHOLD the tool is said to report (`interrogate >=85%`);
* a GATE SCRIPT the reader is told to run it through (`via
  scripts/backend/check-all.sh`);
* an INVOCATION the reader is told to run (`` `interrogate src -v` ``).

A name with none of the three is a mention and passes. A name that occurs only
inside a suppression directive (`# pylint: disable`, `// eslint-disable`,
`# noqa`) is stripped before the scan, so an anti-bypass rule may quote the
directive it forbids without claiming the tool. Both directions are controlled
below, because a detector that fires on everything is turned off, and one that
fires on nothing is trap #5.

WHAT IS NOT VERIFIED. The lexicon of names that COUNT as a tool comes from
`test_skill_gate_model._tool_lexicon` -- this repository's pins plus the tools
its own ADRs and issues record retiring -- because no derivation can tell a tool
name from an English word. A document claiming a gate is verified by a tool in
neither set is invisible here. Invocations are read out of INLINE code spans
only; a command sitting in a fenced block, unaccompanied by a threshold or a
script path, would not be read as one. Blocks are separated by blank lines, so a
claim split across a paragraph break is two blocks and the marker in one does
not reach the name in the other. And this module does not parse English: it
checks the parts that are mechanical.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from tests.toolchain.test_skill_gate_model import (
    _installed_tools,
    _named_tools,
    _tool_lexicon,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_DIR = _REPO_ROOT / ".claude"
_AGENTS_DIR = _CLAUDE_DIR / "agents"
_SKILLS_DIR = _CLAUDE_DIR / "skills"
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_DOC_SPECIALIST = _AGENTS_DIR / "ralph-documentation-specialist.md"
_EVIDENCE_COLLECTOR = _SKILLS_DIR / "de-slopify" / "scripts" / "collect-evidence.sh"

#: A threshold a tool is claimed to report: an operator against a number, or a
#: bare percentage. `>=85%`, `>= 9.0`, `95%`. This is the marker that turns
#: "interrogate" from a name into an obligation, and it is what made the agent
#: definition's claim WEAKER than the rule Gate 1 enforces.
_THRESHOLD = re.compile(r"(?:[><]=?|[≥≤])\s*\d|\d+(?:\.\d+)?\s*%")

#: A gate script, with whatever directory prefix the document writes. The
#: lookbehind stops a longer path being truncated to its `scripts/...` tail, so
#: `.claude/skills/de-slopify/scripts/collect-evidence.sh` is matched whole
#: rather than reported as a missing `scripts/collect-evidence.sh`. A template
#: placeholder like `scripts/<side>/check-all.sh` does not match, and is meant
#: not to: `<side>` is not a path.
_GATE_SCRIPT = re.compile(r"(?<![\w./-])(?:[\w.-]+/)*scripts/[\w/-]+\.sh")

#: A suppression directive. The name inside one is the thing an anti-bypass rule
#: forbids, not a tool it claims to run -- `.claude/agents/shared/house-rules.md`
#: and `.claude/skills/max-quality-no-shortcuts/` both quote `# pylint: disable`
#: for exactly that purpose. Stripped before any name is read.
_SUPPRESSION = re.compile(
    r"(?:#|//)\s*[\w-]+\s*:\s*disable[\w=.-]*"
    r"|(?:#|//)\s*[\w-]+-disable[\w-]*"
    r"|#\s*noqa[^\s`]*"
    r"|@ts-(?:ignore|nocheck)"
)

#: An inline code span, which is how every document in this corpus writes a
#: command a reader is told to run.
_INLINE_CODE = re.compile(r"`([^`\n]+)`")

#: Runner prefixes to strip before reading a command's head, so `./scripts` and
#: `python -m pytest` name the same tool they would name without them.
_RUNNER_PREFIX = re.compile(
    r"^(?:\./|\$ |bash |sh |uv run |python -m |npx --no-install |npx )+"
)

#: Documents are split into claim blocks on blank lines.
_BLOCK_SPLIT = re.compile(r"\n\s*\n")

#: Lower bounds. The corpus held 40 documents making 60 verification claims over
#: 22 distinct tools and naming 26 distinct gate scripts when this was written.
#: Floors well under those catch a scan that has stopped scanning -- trap #5, a
#: corpus walk over zero hits passes forever, which is the exact failure mode
#: this whole issue is about -- without failing every time a file is deleted.
_DOCUMENT_FLOOR = 25
_CLAIM_FLOOR = 25
_CLAIMED_TOOL_FLOOR = 10
_GATE_SCRIPT_FLOOR = 10


@dataclass(frozen=True)
class _Claim:
    """One block of a governing document that claims a tool verifies something.

    Attributes:
        document: The document the block came from.
        line: The 1-based line the block starts on, so a failure names a place.
        text: The block, with suppression directives already stripped.
        tools: The lexicon tools the block names.
        markers: Which obligation markers the block carries.
    """

    document: Path
    line: int
    text: str
    tools: frozenset[str]
    markers: frozenset[str]


def _corpus_documents() -> tuple[Path, ...]:
    """List the documents that govern how an agent works in this repository.

    Returns:
        `CLAUDE.md` and every markdown document under `.claude/`, sorted.
    """
    return (_CLAUDE_MD, *sorted(_CLAUDE_DIR.rglob("*.md")))


def _strip_suppressions(text: str) -> str:
    """Remove suppression directives, so the names inside them are not read.

    Args:
        text: A block of document text.

    Returns:
        The block with each directive replaced by a space.
    """
    return _SUPPRESSION.sub(" ", text)


def _invoked_tools(text: str, lexicon: frozenset[str]) -> frozenset[str]:
    """Find the tools a block tells the reader to RUN.

    A code span holding a bare name is a name; one holding a name and at least
    one argument is a command. `` `ruff` `` is the former, `` `ruff check src` ``
    the latter.

    Args:
        text: A block of document text.
        lexicon: The names that count as tools.

    Returns:
        The tools invoked by an inline code span in the block.
    """
    invoked = set()
    for span in _INLINE_CODE.finditer(text):
        words = _RUNNER_PREFIX.sub("", span.group(1).strip()).split()
        if len(words) > 1 and words[0].lower() in lexicon:
            invoked.add(words[0].lower())
    return frozenset(invoked)


def _obligation_markers(text: str, lexicon: frozenset[str]) -> frozenset[str]:
    """Report which obligation markers a block carries.

    Args:
        text: A block of document text, suppressions already stripped.
        lexicon: The names that count as tools.

    Returns:
        Any of `threshold`, `gate-script`, `invocation`.
    """
    found = set()
    if _THRESHOLD.search(text):
        found.add("threshold")
    if _GATE_SCRIPT.search(text):
        found.add("gate-script")
    if _invoked_tools(text, lexicon):
        found.add("invocation")
    return frozenset(found)


def _claims_in(document: Path, text: str, lexicon: frozenset[str]) -> list[_Claim]:
    """Find the verification claims one document makes.

    Args:
        document: The document, for reporting.
        text: Its full source.
        lexicon: The names that count as tools.

    Returns:
        One `_Claim` per block that names a tool AND carries a marker.
    """
    claims: list[_Claim] = []
    offset = 0
    for block in _BLOCK_SPLIT.split(text):
        line = text[:offset].count("\n") + 1
        offset += len(block) + 2
        clean = _strip_suppressions(block)
        tools = _named_tools(clean, lexicon)
        markers = _obligation_markers(clean, lexicon)
        if tools and markers:
            claims.append(
                _Claim(
                    document=document,
                    line=line,
                    text=clean,
                    tools=tools,
                    markers=markers,
                )
            )
    return claims


def _verification_claims() -> tuple[_Claim, ...]:
    """Collect every verification claim the corpus makes.

    Returns:
        The claims, in document order.
    """
    lexicon = _tool_lexicon()
    return tuple(
        claim
        for document in _corpus_documents()
        for claim in _claims_in(document, document.read_text(encoding="utf-8"), lexicon)
    )


def _named_gate_scripts() -> dict[str, list[str]]:
    """Collect every gate-script path the corpus names, and who names it.

    Returns:
        Path as written, to the `document:line` sites naming it.
    """
    sites: dict[str, list[str]] = {}
    for document in _corpus_documents():
        for number, line in enumerate(
            document.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for path in _GATE_SCRIPT.findall(line):
                where = f"{document.relative_to(_REPO_ROOT)}:{number}"
                sites.setdefault(path, []).append(where)
    return sites


def _collector_invocations() -> frozenset[str]:
    """Read the binaries `collect-evidence.sh` runs through its `run` helper.

    Returns:
        The lowercased binary names, one per `run <outfile> <bin> ...` line.
    """
    source = _EVIDENCE_COLLECTOR.read_text(encoding="utf-8")
    return frozenset(
        match.group("bin").lower()
        for match in re.finditer(
            r"^\s{2}run\s+\S+\s+(?P<bin>[\w.-]+)", source, re.MULTILINE
        )
    )


def _ruff_docstring_rule() -> str:
    """Derive the docstring rule code ruff is configured to enforce.

    Returns:
        The selected rule code from the pydocstyle "missing docstring" family,
        e.g. `D1`.

    Raises:
        AssertionError: If ruff's select list enables no such family, which
            would make the agent definition's replacement text a claim about a
            gate that had itself been removed.
    """
    tool = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8")).get("tool", {})
    assert isinstance(tool, dict)
    ruff = tool.get("ruff", {})
    assert isinstance(ruff, dict)
    lint = ruff.get("lint", {})
    assert isinstance(lint, dict)
    select = lint.get("select", [])
    assert isinstance(select, list)
    codes = [str(code) for code in select if str(code).startswith("D1")]
    assert codes, (
        "ruff's select list in pyproject.toml no longer enables the D1 "
        f"missing-docstring family: {select!r}. The docstring gate issue #351 "
        "wired is gone, so the agent definition cannot correctly name it."
    )
    return codes[0]


def _describe(claims: tuple[_Claim, ...]) -> list[str]:
    """Render claims for a failure message.

    Args:
        claims: The claims to render.

    Returns:
        One `document:line` string per claim, sorted.
    """
    return sorted(
        f"{claim.document.relative_to(_REPO_ROOT)}:{claim.line}" for claim in claims
    )


# --- Controls: the detector fires where it must, and only there ---------------

#: A block making the claim this issue removed: a tool, and a percentage it is
#: said to report. Written as the agent definition wrote it.
_THRESHOLD_CLAIM = (
    "- **Owns**: Python docstrings (Google style consistent with the file;\n"
    "  interrogate ≥85%), TSDoc on exported APIs, README/module docs."
)

#: The same tool, pointed at a gate script instead of a number.
_GATE_SCRIPT_CLAIM = "Verify with: interrogate (via scripts/backend/check-all.sh)"

#: The same tool again, as a command the reader is told to run.
#: `coverage` is itself a pinned tool, so the real row's "docstring coverage
#: gaps" wording would put a second, installed name in the control and blunt the
#: exact-set assertion below.
_INVOCATION_CLAIM = "| **interrogate** | missing docstrings | `interrogate src -v` |"

#: The same tool with no obligation attached: a mention. `pyproject.toml`
#: discusses interrogate at length for exactly this reason and must stay able to.
_BARE_MENTION = (
    "Ruff reimplements pydocstyle's rule set natively, so neither pydocstyle nor\n"
    "interrogate was adopted."
)

#: An anti-bypass rule quoting the directive it forbids, ALONGSIDE a real
#: threshold. The threshold is there on purpose: without it the block would be
#: silent for the boring reason that it carries no marker, and the control would
#: prove nothing about suppression stripping. With it, the block is a claim
#: block, and `pylint` must still not be read out of it.
_SUPPRESSION_MENTION = (
    "> No bypasses. Do not add `# noqa`, `# type: ignore`, `# pylint: disable`,\n"
    "> `// eslint-disable`; do not lower the ≥90% coverage floor."
)

#: The same block with the directive turned into a claim -- one word changed,
#: `disable` to `≥9.0`. If stripping were removing the whole line rather than
#: the directive, this would go silent too, and the control above would be
#: passing for the wrong reason.
_SUPPRESSION_MENTION_TURNED_CLAIM = (
    "> No bypasses. Do not add `# noqa`, `# type: ignore`, pylint ≥9.0,\n"
    "> `// eslint-disable`; do not lower the ≥90% coverage floor."
)


def _control_claims(text: str) -> list[_Claim]:
    """Run the detector over a control block.

    Args:
        text: The control's text.

    Returns:
        The claims found in it.
    """
    return _claims_in(Path("control.md"), text, _tool_lexicon())


def test_the_detector_reads_a_threshold_as_an_obligation() -> None:
    """The positive control for the marker this issue is named after.

    `interrogate ≥85%` is the claim that sat in the agent definition's
    frontmatter, in its Scope section and in its worked example. If the detector
    stops reading it, every corpus assertion below passes over a smaller set.
    """
    claims = _control_claims(_THRESHOLD_CLAIM)

    assert len(claims) == 1, (
        f"the detector found {len(claims)} claims in a block reading "
        "`interrogate ≥85%`; it should find exactly one."
    )
    assert claims[0].tools == frozenset({"interrogate"}), (
        f"the detector named {sorted(claims[0].tools)} rather than interrogate."
    )
    assert claims[0].markers == frozenset({"threshold"}), (
        f"the detector read markers {sorted(claims[0].markers)}; a percentage "
        "after a tool name is a threshold and nothing else here."
    )


def test_the_detector_reads_a_gate_script_as_an_obligation() -> None:
    """The positive control for the marker the split-across-lines claim carries.

    Two of the agent definition's five claims carried no number at all: they
    pointed the reader at a script to run the tool through. A threshold-only
    detector reports three of five and calls the file clean.
    """
    claims = _control_claims(_GATE_SCRIPT_CLAIM)

    assert len(claims) == 1
    assert claims[0].tools == frozenset({"interrogate"})
    assert claims[0].markers == frozenset({"gate-script"}), (
        f"the detector read markers {sorted(claims[0].markers)} from a block "
        "naming a tool and the script it is said to run under."
    )


def test_the_detector_reads_an_invocation_as_an_obligation() -> None:
    """The positive control for a command the reader is told to run.

    `.claude/skills/de-slopify/references/detection-playbook.md` carried its
    stale claim in exactly this shape -- a table row whose third column is the
    invocation -- with no number and no script path anywhere near it.
    """
    claims = _control_claims(_INVOCATION_CLAIM)

    assert len(claims) == 1
    assert claims[0].tools == frozenset({"interrogate"})
    assert claims[0].markers == frozenset({"invocation"}), (
        f"the detector read markers {sorted(claims[0].markers)} from a table "
        "row whose code span is `interrogate src -v`."
    )


def test_a_tool_named_without_an_obligation_is_not_a_claim() -> None:
    """The negative control: naming a tool is not claiming it verifies anything.

    A detector that cannot tell these apart makes `pyproject.toml`'s own
    explanation of why interrogate was rejected into a violation, and gets
    turned off. This is the direction that kills the whole assertion.
    """
    assert _control_claims(_BARE_MENTION) == [], (
        "the detector reported a verification claim in a sentence that names "
        "interrogate only to say it was not adopted."
    )


def test_a_suppression_directive_is_a_mention_and_not_a_claim() -> None:
    """An anti-bypass rule may quote the directive it forbids.

    `.claude/agents/shared/house-rules.md` and the `max-quality-no-shortcuts`
    skill both name `# pylint: disable=` as an example of what NOT to write.
    That is the opposite of claiming pylint verifies a gate, and a rule that
    could not tell the difference would have to be excepted for both files --
    which is how a rule stops being applied.

    Both halves are asserted, and the pair differs by one word: strip the
    directive and pylint is gone; write `pylint ≥9.0` in its place and it is
    back. A control that only showed the silence could be silent because the
    stripper eats the whole line.
    """
    quoted = _control_claims(_SUPPRESSION_MENTION)
    claimed = _control_claims(_SUPPRESSION_MENTION_TURNED_CLAIM)

    assert quoted, (
        "the anti-bypass control is not a claim block at all, so its silence "
        "about pylint proves nothing about suppression stripping. It states a "
        "≥90% coverage floor precisely so that it is one."
    )
    assert all("pylint" not in claim.tools for claim in quoted), (
        f"pylint was read out of {[sorted(c.tools) for c in quoted]} in a block "
        "that names it only inside `# pylint: disable`, which is a suppression "
        "the rule forbids rather than a tool it claims to run."
    )
    assert any("pylint" in claim.tools for claim in claimed), (
        "the same block with `# pylint: disable` replaced by `pylint ≥9.0` is "
        "still not read as a pylint claim, so the stripper is removing more "
        "than the directive -- probably the whole line -- and the control above "
        "is silent for the wrong reason."
    )


# --- The corpus is real -------------------------------------------------------


def test_the_scanned_corpus_is_non_empty_and_reaches_the_agent_definitions() -> None:
    """The walk reaches every governing directory, agents included.

    PR #544 asserted the `.claude/skills/` half of its corpus separately for
    this reason: a widening that silently matches nothing is indistinguishable
    from never having widened. `.claude/agents/` is this issue's widening, and
    `ralph-documentation-specialist.md` is the document it exists for.
    """
    documents = _corpus_documents()
    agents = [path for path in documents if _AGENTS_DIR in path.parents]
    skills = [path for path in documents if _SKILLS_DIR in path.parents]

    assert len(documents) >= _DOCUMENT_FLOOR, (
        f"only {len(documents)} markdown documents found under {_CLAUDE_DIR}, "
        f"below the floor of {_DOCUMENT_FLOOR}. The walk is not reaching the "
        "corpus, so every assertion below is about a smaller set than it says."
    )
    assert _CLAUDE_MD in documents
    assert agents, (
        f"no document under {_AGENTS_DIR} reached the corpus. That directory is "
        "this issue's widening; without it the scan covers the documents the "
        "rules are written in and not the ones a subagent is instructed by."
    )
    assert skills, f"no document under {_SKILLS_DIR} reached the corpus (#536)."
    assert _DOC_SPECIALIST in documents, (
        f"{_DOC_SPECIALIST} is not in the scanned corpus at all, and it is the "
        "document this module exists for."
    )


def test_the_corpus_makes_verification_claims_including_in_the_agents() -> None:
    """The claim census is non-empty, so agreeing with the toolchain means something.

    This is the anti-vacuity assertion. Narrowing the detector until it matches
    nothing would turn "every claimed tool is installed" into a statement about
    the empty set -- passing forever, which is precisely the defect class this
    issue belongs to. The agents half is counted separately for the same reason
    the corpus walk is: a widening that finds nothing has not widened.
    """
    claims = _verification_claims()
    tools = {tool for claim in claims for tool in claim.tools}
    from_agents = [claim for claim in claims if _AGENTS_DIR in claim.document.parents]

    assert len(claims) >= _CLAIM_FLOOR, (
        f"only {len(claims)} verification claims found across "
        f"{len(_corpus_documents())} documents, below the floor of "
        f"{_CLAIM_FLOOR}. The markers have stopped matching how this corpus "
        "writes an obligation."
    )
    assert len(tools) >= _CLAIMED_TOOL_FLOOR, (
        f"the claims name only {sorted(tools)}, below the floor of "
        f"{_CLAIMED_TOOL_FLOOR} distinct tools."
    )
    assert from_agents, (
        f"no verification claim was found under {_AGENTS_DIR}. Either the agent "
        "definitions stopped naming the tools they tell a subagent to verify "
        "with, or the widening this issue made is inert."
    )


# --- The assertions this issue exists for -------------------------------------


def test_every_tool_claimed_to_verify_something_is_one_this_repository_runs() -> None:
    """A tool named as a verifier is a tool the repository installs.

    The assertion this module exists for. An agent told to verify with
    `interrogate` finds no such binary; the honest outcomes are that it reports
    BLOCKED, or -- far likelier, and what happened -- that it treats a gate it
    could not run as one it passed. Worse here than in a doc, because `>=85%` is
    LOOSER than the ruff `D1` presence rule Gate 1 actually enforces, so an
    agent could satisfy the written rule exactly and still leave Gate 1 red.
    """
    installed = _installed_tools()
    offenders = {
        f"{claim.document.relative_to(_REPO_ROOT)}:{claim.line}": sorted(
            claim.tools - installed
        )
        for claim in _verification_claims()
        if claim.tools - installed
    }

    assert offenders == {}, (
        f"{offenders} name a tool as the verifier of a gate -- with a "
        "threshold, a gate script, or an invocation -- that is pinned in no "
        "constraints file, listed in no requirements file and wired into no "
        "pre-commit hook. A mention of a tool is fine; telling an agent to "
        "verify with one that is not there is not."
    )


def test_every_gate_script_the_corpus_names_exists() -> None:
    """A verification path a reader is sent to is a path that is there.

    The third of the agent definition's three compounding falsehoods, and the
    one that survives fixing the tool: `scripts/backend/check-all.sh` has never
    existed in this repository (issue #142). An agent that runs it gets `No such
    file or directory` and has to decide what that means -- and a script that
    cannot run has never reported a failure.
    """
    named = _named_gate_scripts()
    missing = {
        path: sites
        for path, sites in named.items()
        if not (_REPO_ROOT / path).is_file()
    }

    assert len(named) >= _GATE_SCRIPT_FLOOR, (
        f"only {sorted(named)} parsed as gate-script paths across the corpus, "
        f"below the floor of {_GATE_SCRIPT_FLOOR}. The pattern has stopped "
        "matching how these documents write a path, and 'every one of them "
        "exists' is then a statement about almost nothing."
    )
    assert missing == {}, (
        f"{missing} are named as scripts to run but do not exist. Template "
        "placeholders like `scripts/<side>/check-all.sh` are deliberately not "
        "matched here; these are paths written as real ones."
    )


def test_the_documentation_specialist_names_the_docstring_rule_ruff_enforces() -> None:
    """The replacement is present, not merely the falsehood absent.

    Deleting the five `interrogate` claims would satisfy every assertion above
    and leave the documentation specialist with no idea what verifies its work.
    The rule code is derived from ruff's own select list, so if the D1 gate
    issue #351 wired were ever removed, this fails rather than pinning a code
    that had stopped meaning anything.
    """
    rule = _ruff_docstring_rule()
    text = _DOC_SPECIALIST.read_text(encoding="utf-8")

    assert re.search(rf"(?<![\w-]){rule}(?![\w-])", text), (
        f"{_DOC_SPECIALIST.relative_to(_REPO_ROOT)} does not name ruff `{rule}`, "
        "which is what pyproject.toml selects and therefore what actually "
        "verifies docstring presence here. An agent whose brief names no "
        "verifier will pick one."
    )


def test_the_evidence_collector_runs_only_tools_this_repository_installs() -> None:
    """`collect-evidence.sh` invokes no tool that can only ever report SKIPPED.

    The collector is honest about absence -- its `run` helper writes `SKIPPED:
    <bin> not installed` rather than pretending -- and that graceful degradation
    is correct and deliberately left alone. What is not correct is keeping a
    slot in the bundle that can never be anything but SKIPPED while the skill
    documents it as a check the collector performs. The docs are markdown and
    are covered by the corpus scan above; the script is not, so it is asserted
    here.
    """
    lexicon = _tool_lexicon()
    installed = _installed_tools()
    invoked = _collector_invocations()

    assert invoked, (
        f"no `run <outfile> <bin>` lines parsed out of {_EVIDENCE_COLLECTOR}, "
        "so this assertion is about the empty set."
    )
    assert (invoked & lexicon) <= installed, (
        f"{sorted((invoked & lexicon) - installed)} are invoked by "
        f"{_EVIDENCE_COLLECTOR.relative_to(_REPO_ROOT)} but installed nowhere "
        "in this repository, so their evidence slot can only ever read "
        "`SKIPPED`."
    )
