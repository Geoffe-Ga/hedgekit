"""No workflow `name:` or `if:` may be silently eaten by a `#` (issue #509).

In YAML a `#` preceded by whitespace opens a comment *inside an unquoted
scalar*. So this line:

    name: End-to-End Container Tier (epic #465)

does not declare the name its author wrote. It declares
`End-to-End Container Tier (epic` -- a string with a dangling open paren and
no closing one -- and discards ` #465)` as a comment. Nothing warns. The file
is valid YAML, the workflow runs, the log shows a plausible-looking name, and
the ONLY way to notice is to compare what the file says against what a YAML
parser makes of it.

WHY THIS IS NOT COSMETIC. A job's `name:` is the status-check context GitHub
reports, and branch protection matches required contexts against that string
exactly. On 2026-08-11 this repository sat in the resulting state: the
container job was registered as a required check under the *truncated* name,
because the truncated name is the only context the job will ever report. That
worked -- by coincidence, not by construction. Anyone tidying the quoting in
isolation would have renamed the check run out from under a protection entry
that no job then satisfies, and every pull request would have blocked forever
on a context nothing produces, with all checks green and nothing to point at.
That failure was observed live on PRs #503 and #508 before it was understood.

WHAT THIS MODULE ASSERTS, AND HOW IT AVOIDS BEING UNFALSIFIABLE

The detector needs BOTH sides. Grepping the source text for `#` in a `name:`
line is a lint on characters: it cannot tell a truncated scalar from a `#`
inside a quoted string or a `run: |` block, and it does not know what YAML
made of the line. Parsing the YAML alone is worse: the parser hands back the
*already-truncated* value with nothing to compare it to, so a truncated file
looks exactly like a correct one. :func:`truncations_in` therefore parses to a
node tree (which carries source marks and the scalar's quoting style) and then
reads the raw line back to see what sits past the end of the scalar. A plain
scalar followed by a `#` lost text; a quoted one did not; a block scalar's `#`
is literal content and is never flagged. Three of the tests below are controls
proving each of those three cases.

The corpus is derived from the directory listing, never restated, so a
workflow added tomorrow is covered without anyone remembering to add it -- and
it is asserted non-empty and named, because a corpus scan over zero files
passes forever and would be indistinguishable from a working guard. The same
trap is why :func:`truncations_in` takes a source string rather than only a
path: it lets the positive control feed the detector a known-truncated fixture
and assert it FIRES, which is the only evidence that a green run over the real
corpus means anything at all.

WHY ONLY `name:` AND `if:`

Twenty other scalars in `.github/workflows/` are followed by a `#`, and all of
them are deliberate: `uses: <owner>/<action>@<sha>  # v1.2.3`, the convention
this repo (and Dependabot) uses to record the human-readable version beside a
pinned SHA, and `permissions:` entries carrying a note about why a scope is
granted. In every one of those the VALUE IS COMPLETE before the `#` -- the SHA
is the SHA, `read` is `read` -- so nothing is lost and flagging them would
make this guard a nuisance to be suppressed rather than a rule to be kept.

`name:` and `if:` are different in kind, not degree. A `name:` is an identity
string that leaves the repository: it is the check-run context branch
protection matches, so losing its tail changes what the merge gate means. An
`if:` is gating logic, where a dropped tail silently changes which jobs run.
Neither has any legitimate use for a trailing comment; put the comment on the
line above. That is the rule, and it is deliberately absolute rather than
per-instance, so there is no exemption to reach for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The repository root, two levels up from `tests/toolchain/`.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Directory holding every GitHub Actions workflow in this repository.
WORKFLOW_DIRECTORY = REPO_ROOT / ".github" / "workflows"

#: Extensions GitHub Actions recognises for a workflow definition.
WORKFLOW_SUFFIXES = (".yml", ".yaml")

#: Mapping keys whose values must never be truncated by an inline comment.
#:
#: `name` is the check-run context branch protection matches on; `if` is job
#: and step gating logic. A trailing comment on either is never meaningful and
#: always destroys part of the value. See this module's docstring for why the
#: rule stops here rather than covering `uses:` and `permissions:`, where a
#: trailing comment is an established convention and loses nothing.
COMMENT_SENSITIVE_KEYS = ("name", "if")

#: Floor on the number of comment-sensitive scalars the corpus must yield.
#:
#: The guard below iterates whatever the scan finds. If a refactor broke the
#: traversal it would iterate over nothing and pass, so the count is asserted
#: to be substantial rather than merely non-zero. The corpus held 116 at the
#: time of writing (106 `name:`, 10 `if:`); the floor is set well below that
#: so ordinary deletions do not trip it.
MINIMUM_SENSITIVE_SCALARS = 40

#: The workflow whose container job is a required status check on `main`.
CI_WORKFLOW = WORKFLOW_DIRECTORY / "ci.yml"

#: Key of the container-tier job inside that workflow's `jobs:` mapping.
CONTAINER_JOB_ID = "container"

#: The epic reference that YAML was eating out of the container job's name.
#:
#: Asserted explicitly, and not only via the general rule above, because this
#: exact substring is the one whose loss silently reshaped a required status
#: check -- issue #509's acceptance criterion 1.
CONTAINER_JOB_ISSUE_REFERENCE = "#465"

#: A workflow fixture carrying the exact defect this module exists to catch.
#:
#: The positive control. Without it, a green run proves only that the detector
#: found nothing -- which a detector that can never fire also achieves.
TRUNCATED_FIXTURE = """
jobs:
  container:
    name: End-to-End Container Tier (epic #465)
    runs-on: ubuntu-latest
"""

#: The same fixture with the scalar quoted, i.e. the corrected form.
#:
#: The negative control. A detector that flags everything would satisfy the
#: positive control alone, so the corrected form must come back clean.
QUOTED_FIXTURE = """
jobs:
  container:
    name: "End-to-End Container Tier (epic #465)"
    runs-on: ubuntu-latest
"""

#: A fixture whose `#` characters are literal content, not comment markers.
#:
#: `run: |` opens a literal block scalar, in which `#` is data. A text-level
#: grep for `#` cannot tell this apart from a truncation; the parse-and-compare
#: detector can, and this proves it does. The `name:` here also carries a
#: quoted `#`, which likewise must not be flagged.
BLOCK_SCALAR_FIXTURE = """
jobs:
  demo:
    name: "Report on PR #123"
    steps:
      - name: Comment
        run: |
          # this is shell, not YAML
          echo "PR #123 has issue #456"
"""


@dataclass(frozen=True)
class Truncation:
    """One scalar whose text was cut short by an inline comment marker.

    Attributes:
        origin: Where the scalar came from, for the failure message.
        line: 1-based line number of the truncated scalar.
        key: The mapping key the scalar is the value of.
        parsed: What YAML actually produced -- the surviving prefix.
        dropped: The raw text discarded as a comment, `#` included.
    """

    origin: str
    line: int
    key: str
    parsed: str
    dropped: str

    def describe(self) -> str:
        """Render the finding with both sides of the comparison.

        Both values are shown because the whole defect is that they differ
        and nothing says so; a message naming only one of them would leave
        the reader to re-derive the other.

        Returns:
            A single-line description naming the file, key and both values.
        """
        return (
            f"{self.origin}:{self.line}: `{self.key}:` parses as "
            f"{self.parsed!r} -- YAML discarded {self.dropped!r} as a comment. "
            f"Quote the value to keep it."
        )


def _scalar_values(node: yaml.Node) -> Iterator[tuple[str, yaml.ScalarNode]]:
    """Walk a composed YAML tree yielding every keyed scalar value.

    Args:
        node: Root of a tree from :func:`yaml.compose`.

    Yields:
        ``(key, value_node)`` for each mapping entry whose value is a scalar,
        at any depth, including inside sequences.
    """
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and isinstance(
                value_node, yaml.ScalarNode
            ):
                yield str(key_node.value), value_node
            yield from _scalar_values(value_node)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            yield from _scalar_values(item)


def sensitive_scalars(source: str) -> list[tuple[str, yaml.ScalarNode]]:
    """Find every `name:`/`if:` scalar in a workflow source.

    Args:
        source: Raw text of a workflow file.

    Returns:
        ``(key, value_node)`` pairs for the comment-sensitive keys, in
        document order.
    """
    return [
        (key, value)
        for key, value in _scalar_values(yaml.compose(source))
        if key in COMMENT_SENSITIVE_KEYS
    ]


def truncations_in(source: str, origin: str) -> list[Truncation]:
    """Report comment-sensitive scalars whose text YAML cut short.

    Takes source text rather than a path so the controls below can hand it a
    fixture and prove the detector fires; a detector only ever pointed at a
    clean corpus is indistinguishable from one that cannot fire at all.

    A scalar is truncated when it is *plain* (unquoted -- `node.style` is
    ``None``; a quoted or block scalar cannot be cut short by a `#`) and the
    raw source picks up again with a `#` immediately past where the parser
    says the scalar ended.

    Args:
        source: Raw text of a workflow file.
        origin: Label for the source, used in the reported finding.

    Returns:
        One :class:`Truncation` per affected scalar, in document order.
    """
    lines = source.splitlines()
    findings = []
    for key, value in sensitive_scalars(source):
        if value.style is not None:
            continue
        end = value.end_mark
        if end.line >= len(lines):
            continue
        remainder = lines[end.line][end.column :]
        if not remainder.lstrip().startswith("#"):
            continue
        findings.append(
            Truncation(
                origin=origin,
                line=end.line + 1,
                key=key,
                parsed=str(value.value),
                dropped=remainder.strip(),
            )
        )
    return findings


def workflow_paths() -> list[Path]:
    """List every workflow file, derived from the directory listing.

    Derived rather than restated so a workflow added later is covered without
    anyone remembering to register it here.

    Returns:
        Sorted paths of every `.yml`/`.yaml` file under `.github/workflows/`.
    """
    return sorted(
        path
        for path in WORKFLOW_DIRECTORY.iterdir()
        if path.is_file() and path.suffix in WORKFLOW_SUFFIXES
    )


def _container_job_name() -> str:
    """Read the parsed `name:` of `ci.yml`'s container job.

    Returns:
        The name exactly as a YAML parser produces it -- which is the string
        GitHub reports the check run under.
    """
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    return str(workflow["jobs"][CONTAINER_JOB_ID]["name"])


def test_the_workflow_corpus_is_non_empty_and_covers_ci() -> None:
    """The scan finds files, so the guard over them can fail.

    A corpus scan over zero files passes forever and looks identical to a
    working guard. This repository has been bitten by that shape repeatedly,
    so the corpus is asserted to exist and to contain the workflow that
    carries the merge-gating job.
    """
    paths = workflow_paths()
    names = sorted(path.name for path in paths)

    assert paths, (
        f"no workflow files found under {WORKFLOW_DIRECTORY}. The truncation "
        "guard below iterates this list, so it would pass over an empty "
        "corpus while checking nothing."
    )
    assert CI_WORKFLOW.name in names, (
        f"{CI_WORKFLOW.name} is not among the scanned workflows {names}. It "
        "holds the container job whose name is a required status check -- the "
        "exact scalar this module exists to protect."
    )


def test_the_scan_reaches_a_substantial_number_of_sensitive_scalars() -> None:
    """The traversal visits real `name:`/`if:` values, not an empty set.

    The corpus can be non-empty while the tree walk finds nothing in it -- a
    refactor of :func:`_scalar_values` that stopped descending into sequences,
    for instance, would miss every step `name:` in the repository and still
    report a clean scan.
    """
    counts = {
        path.name: len(sensitive_scalars(path.read_text(encoding="utf-8")))
        for path in workflow_paths()
    }
    total = sum(counts.values())

    assert total >= MINIMUM_SENSITIVE_SCALARS, (
        f"the walk found only {total} `name:`/`if:` scalars across "
        f"{len(counts)} workflows ({counts}); expected at least "
        f"{MINIMUM_SENSITIVE_SCALARS}. The traversal is broken, so the guard "
        "below is checking almost nothing."
    )


def test_the_detector_fires_on_a_known_truncated_name() -> None:
    """POSITIVE CONTROL: the detector reports the historical defect.

    This is the string that actually broke the merge gate, fed to the same
    function the corpus guard uses. Without this, a green corpus run would be
    evidence of nothing: a detector that can never fire also produces one.
    """
    findings = truncations_in(TRUNCATED_FIXTURE, "<fixture>")

    assert len(findings) == 1, (
        f"expected exactly one truncation in the control fixture, got "
        f"{[finding.describe() for finding in findings]}. The detector cannot "
        "see the defect it exists to catch, so every green run below is "
        "meaningless."
    )
    finding = findings[0]
    assert finding.key == "name"
    assert finding.parsed == "End-to-End Container Tier (epic", (
        f"the control fixture parsed to {finding.parsed!r}; the whole point "
        "of the fixture is that YAML stops at the `#`"
    )
    assert finding.dropped == "#465)", (
        f"the detector reported {finding.dropped!r} as the discarded text; "
        "expected the '#465)' that YAML ate"
    )


def test_the_detector_stays_silent_when_the_name_is_quoted() -> None:
    """NEGATIVE CONTROL: the corrected form is not reported.

    A detector that flagged every `#` would satisfy the positive control and
    still be useless, because the fix it demands would not silence it. Quoting
    the scalar is the fix; it must come back clean.
    """
    findings = truncations_in(QUOTED_FIXTURE, "<fixture>")

    assert findings == [], (
        f"the quoted form was reported as truncated: "
        f"{[finding.describe() for finding in findings]}. The detector fires "
        "on correct YAML, so it cannot be used as a gate."
    )


def test_the_detector_ignores_comment_markers_that_are_data() -> None:
    """CONTROL: a `#` inside a block scalar or a quoted string is content.

    This is what separates parse-and-compare from grepping the source text. A
    `run: |` body is a literal block scalar in which `#` is shell, not YAML,
    and a quoted `name:` keeps its `#` too. A text-level guard would flag both
    and be disabled within the week.
    """
    findings = truncations_in(BLOCK_SCALAR_FIXTURE, "<fixture>")

    assert findings == [], (
        f"literal `#` content was reported as truncation: "
        f"{[finding.describe() for finding in findings]}. The detector is "
        "reading characters rather than comparing the parse against the source."
    )


def test_no_workflow_name_or_condition_is_truncated_by_a_comment() -> None:
    """THE GUARD: every `name:`/`if:` in the repo is what its author wrote.

    Reintroducing the unquoted form anywhere under `.github/workflows/` fails
    here, naming the file, the line, the value YAML kept and the text it threw
    away -- so the reader does not have to rediscover that a `#` opens a
    comment inside an unquoted scalar.
    """
    findings = [
        finding.describe()
        for path in workflow_paths()
        for finding in truncations_in(
            path.read_text(encoding="utf-8"),
            str(path.relative_to(REPO_ROOT)),
        )
    ]

    assert findings == [], (
        "workflow scalars are being silently truncated by an inline comment "
        "marker:\n  " + "\n  ".join(findings) + "\n"
        "In YAML a `#` after whitespace opens a comment inside an unquoted "
        "scalar. For a job `name:` that string is the status-check context "
        "branch protection matches on, so truncating it changes what the "
        "merge gate means (issue #509). Wrap the value in quotes, or move the "
        "comment to its own line above."
    )


def test_the_container_job_name_keeps_its_epic_reference() -> None:
    """`ci.yml`'s container job declares the full name, `#465` included.

    Issue #509's acceptance criterion 1, and a deliberately concrete anchor
    beside the general rule above: this is the one scalar whose truncation was
    load-bearing, so it is asserted by name rather than only as a member of a
    scanned set. The expected value is read out of the workflow rather than
    restated, so the test cannot agree with a rename it did not notice.

    NOTE FOR WHOEVER RENAMES THIS JOB: the parsed name IS the required status
    check context on `main`. Changing it without changing the protection entry
    blocks every pull request on a context nothing reports. The correspondence
    with the live setting is asserted by
    `tests/e2e/test_tier_selection_contract.py`.
    """
    name = _container_job_name()

    assert CONTAINER_JOB_ISSUE_REFERENCE in name, (
        f"the container job's name parses as {name!r}, without "
        f"{CONTAINER_JOB_ISSUE_REFERENCE}. The scalar is unquoted, so YAML "
        "read the ` #465)` as a comment and dropped it. GitHub reports the "
        "check run under the truncated string, and branch protection matches "
        "required contexts by exact string."
    )
    assert name.count("(") == name.count(")"), (
        f"the container job's name {name!r} has unbalanced parentheses, the "
        "signature of a scalar cut short mid-parenthetical"
    )
