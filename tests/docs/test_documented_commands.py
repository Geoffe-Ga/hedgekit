"""Documented operator commands are checked against the software (#472, #465).

The operator-facing docs are the product's real interface: an operator's first
contact is `README.md`'s deployment section and `RUNBOOK.md`. Both contained
commands and transcripts that nothing executed.

Issue #445 documents the sharp end. `README.md:150-156` prints a
`docker compose kill pipeline` transcript **against a running stack that could
not exist**, because the compose build context resolved to a directory with no
Dockerfile. The output in the README was never produced by running the command.
Issue #449 was the same failure in the CLI surface -- two documented operator
controls that did not exist.

`tests/docs/test_docs_consistency.py` already checks that the docs agree with
*each other*. This module checks that they agree with the *software*:

* every documented `windbreak ...` command parses against the real
  `windbreak.main.build_parser()`;
* every documented `docker compose -f ...` names a compose file that exists and
  services that file actually defines;
* every documented `./scripts/*.sh` exists and is executable.

ON PLACEHOLDERS

The RUNBOOK writes arguments as `<dir>`, `<path>`, `<32-hex-approval-id>`.
Those are substituted from :data:`_PLACEHOLDER_VALUES` before parsing, because
some of them hit real validators -- `--approval-id` rejects anything that is not
32 lowercase hex characters, and it is right to. Every placeholder appearing in
the docs must have an entry, enforced by
:func:`test_every_documented_placeholder_has_a_substitution`, so a new
placeholder shape cannot silently receive a substitution that happens to parse.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.deploy.test_deployment_cli_contract import parse_with_real_cli

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

#: Repo root, derived from this file's own location
#: (`<root>/tests/docs/test_documented_commands.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The operator-facing documents whose commands are checked.
_OPERATOR_DOCS = ("README.md", "RUNBOOK.md", "OPERATOR_WARNINGS.md")

#: Opening/closing fence, capturing the info string.
_FENCE = re.compile(r"^```([a-zA-Z0-9_-]*)\s*$")

#: Info strings whose blocks hold runnable shell. Blocks in any other language
#: (or with no info string, which in these docs means sample output) are not
#: commands and are not checked.
_SHELL_LANGUAGES = frozenset({"bash", "sh", "shell", "console"})

#: A `<...>` placeholder token as the docs write them.
_PLACEHOLDER = re.compile(r"<[^<>\s]+>")

#: Substitutions applied before parsing. Values are chosen to satisfy the real
#: validators rather than to dodge them -- `--approval-id` genuinely requires 32
#: lowercase hex characters, so the substitution supplies 32 of them.
_PLACEHOLDER_VALUES = {
    "<dir>": "/tmp/windbreak-doc-check",
    "<path>": "/tmp/windbreak-doc-check/file",
    "<32-hex-approval-id>": "0" * 32,
}

#: Compose subcommands whose trailing bare tokens all name services.
_SERVICE_TAKING_SUBCOMMANDS = frozenset(
    {"up", "down", "kill", "ps", "logs", "restart", "start", "stop", "rm"}
)

#: Compose subcommands that take exactly ONE service and then a command to run
#: inside it. `exec pipeline ls /var/lib/...` names one service, not three: the
#: rest is the command, and treating it as service names produced a confident,
#: completely wrong failure on a correct README line.
_SINGLE_SERVICE_SUBCOMMANDS = frozenset({"exec", "run"})

#: Compose flags that consume the following token as their value, so it is not
#: mistaken for a service name. `--tail 2 pipeline` is exactly the trap: without
#: `--tail` here, `2` was read as a service name and the check failed on a
#: perfectly correct README line.
#:
#: WHY THIS IS LITERAL, AND HOW IT IS KEPT HONEST. `docker compose` publishes no
#: machine-readable option table to derive from: `--help` needs Docker
#: installed, which this tier cannot assume, and Compose ships no completion
#: data here. A hand-kept flag list is the "restated instead of derived" trap,
#: so both sets are pinned from both ends and hold nothing speculative --
#: exactly the flags the operator docs use today, and no more:
#:
#: * :func:`test_every_documented_compose_flag_is_classified` fails, naming the
#:   flag and the document line, on anything in neither set. Unknown flags are
#:   never guessed at.
#: * :func:`test_every_classified_compose_flag_appears_in_the_docs` fails on an
#:   entry no document uses, so neither set can grow ahead of the docs and rot.
#:
#: A newly documented flag therefore costs one line and a named failure telling
#: you to write it -- which is the trade a list like this has to make to be
#: worth having.
_VALUE_TAKING_COMPOSE_FLAGS = frozenset({"--format", "--tail", "-f"})

#: Compose flags that take no value, so the token after them IS a service name.
#: Kept explicit so `test_every_documented_compose_flag_is_classified` can fail
#: on anything in neither set, rather than guessing and mis-parsing silently.
_BOOLEAN_COMPOSE_FLAGS = frozenset({"-a", "-d"})

#: `-f` is Compose's `--file` before the subcommand and `--follow` -- which takes
#: no value -- after it on `logs`. Both sets are position-blind, so the two
#: readings cannot coexist: classifying `-f` as value-taking (which the `-f
#: deploy/docker-compose.yml` every documented command opens with requires) would
#: make a documented `logs -f pipeline` swallow its service and check nothing.
#: :func:`test_no_documented_command_uses_the_follow_form_of_f` keeps that
#: collision impossible rather than latent.
_FOLLOW_AMBIGUOUS_FLAG = "-f"

#: Command prefixes that are environment setup rather than claims about this
#: software: a virtualenv, a package install, a pre-commit invocation. They are
#: deliberately not checked -- their correctness is pip's and pre-commit's, not
#: this repository's -- and the list is kept short enough to read.
_UNCHECKED_PREFIXES = (
    "python",
    "python3",
    "source",
    "pip",
    "pre-commit",
    "git",
    "gh",
    "cd",
    "export",
    "mkdir",
    "cat",
    "curl",
    "systemctl",
    "sudo",
    "ls",
    "cp",
    "mv",
    "echo",
)


@dataclass(frozen=True)
class DocumentedCommand:
    """One shell command extracted from an operator document.

    Attributes:
        document: Repo-relative document name.
        line: 1-based line number the command appears on.
        command: The command text, with any leading ``$`` prompt removed.
    """

    document: str
    line: int
    command: str

    def where(self) -> str:
        """Render this command's location for a failure message.

        Returns:
            A ``document:line`` reference with the command text.
        """
        return f"{self.document}:{self.line}: {self.command}"


def _strip_prompt(line: str) -> str:
    """Remove a leading shell prompt from a documented command line.

    Args:
        line: The raw line from inside a fenced block.

    Returns:
        The command text without a leading ``$`` prompt.
    """
    stripped = line.strip()
    if stripped.startswith("$ "):
        return stripped[2:].strip()
    return stripped


def _is_command_line(line: str) -> bool:
    """Report whether a line inside a shell block is a command to check.

    Blank lines, comments, and continuation fragments are not commands.

    Args:
        line: The already prompt-stripped line.

    Returns:
        ``True`` if the line should be treated as a command.
    """
    if not line or line.startswith("#"):
        return False
    return not line.startswith(("-", "|", ">"))


def _shell_blocks(text: str) -> Iterator[list[tuple[int, str]]]:
    """Group a document's lines into one list per fenced shell block.

    Blocks are yielded whole because a backslash continuation is joined *within*
    a block. Carrying that state across a closing fence would silently glue the
    tail of one block onto the first line of the next.

    Args:
        text: The document's full text.

    Yields:
        ``(line number, raw line)`` pairs for each shell block, in order.
    """
    block: list[tuple[int, str]] = []
    in_shell_block = False
    for number, raw in enumerate(text.splitlines(), start=1):
        fence = _FENCE.match(raw.rstrip())
        if fence is None and not raw.rstrip().startswith("```"):
            if in_shell_block:
                block.append((number, raw))
            continue
        if in_shell_block and block:
            yield block
        block = []
        in_shell_block = (
            fence is not None and fence.group(1).lower() in _SHELL_LANGUAGES
        )
    if in_shell_block and block:
        yield block


def _commands_in_block(
    document: str, block: list[tuple[int, str]]
) -> list[DocumentedCommand]:
    """Turn one shell block's lines into commands, joining continuations.

    A command wrapped with a trailing backslash is one command, reported at the
    line it starts on. Without the join its fragments are lexed separately, and
    the tail of `docker compose ... exec pipeline \\` is not valid shell alone,
    so the line falls out of every check in this module.

    Two properties matter beyond the join itself, because a dropped command is
    invisible to every other guard here -- including
    :func:`test_every_documented_command_can_be_lexed`, which can only see a
    command it was given:

    * A continuation never escapes its block; :func:`_shell_blocks` bounds it.
    * A dangling continuation at the end of a block is still emitted, rather
      than silently discarded with the state it was accumulating.

    Args:
        document: Repo-relative document name.
        block: ``(line number, raw line)`` pairs from one shell block.

    Returns:
        The block's commands, in order.
    """
    commands: list[DocumentedCommand] = []
    pending = ""
    start = 0
    for number, raw in block:
        text = raw.strip() if pending else _strip_prompt(raw)
        if not pending:
            if not _is_command_line(text):
                continue
            start = number
        if text.endswith("\\"):
            pending = f"{pending}{text[:-1].strip()} "
            continue
        commands.append(
            DocumentedCommand(document=document, line=start, command=f"{pending}{text}")
        )
        pending = ""
    if pending:
        commands.append(
            DocumentedCommand(document=document, line=start, command=pending.strip())
        )
    return commands


def extract_commands(document: str) -> list[DocumentedCommand]:
    """Extract every shell command from a document's fenced shell blocks.

    Args:
        document: Repo-relative document name.

    Returns:
        The commands found, in document order.
    """
    text = (_REPO_ROOT / document).read_text(encoding="utf-8")
    return [
        command
        for block in _shell_blocks(text)
        for command in _commands_in_block(document, block)
    ]


def all_documented_commands() -> list[DocumentedCommand]:
    """Extract commands from every operator document.

    Returns:
        Every documented command across :data:`_OPERATOR_DOCS`.
    """
    return [
        command for document in _OPERATOR_DOCS for command in extract_commands(document)
    ]


def _tokens_of(command: DocumentedCommand) -> list[str]:
    """Split a documented command into shell tokens.

    Args:
        command: The documented command.

    Returns:
        The command's tokens, or an empty list if it cannot be lexed.
    """
    try:
        return shlex.split(command.command, comments=True)
    except ValueError:
        return []


def _select(prefix: str) -> list[DocumentedCommand]:
    """Select documented commands whose first token equals ``prefix``.

    Args:
        prefix: The leading token to match, e.g. ``windbreak`` or ``docker``.

    Returns:
        The matching commands.
    """
    selected: list[DocumentedCommand] = []
    for command in all_documented_commands():
        tokens = _tokens_of(command)
        if tokens and tokens[0] == prefix:
            selected.append(command)
    return selected


def _substitute_placeholders(tokens: Iterable[str]) -> list[str]:
    """Replace documentation placeholders with values the real validators accept.

    Args:
        tokens: The command's tokens.

    Returns:
        The tokens with every known placeholder substituted.
    """
    substituted: list[str] = []
    for token in tokens:
        replaced = token
        for placeholder, value in _PLACEHOLDER_VALUES.items():
            replaced = replaced.replace(placeholder, value)
        substituted.append(replaced)
    return substituted


def _compose_document(compose_path: Path) -> dict[str, Any]:
    """Load a compose file.

    Args:
        compose_path: Path to the compose file.

    Returns:
        The parsed compose document.
    """
    return yaml.safe_load(compose_path.read_text(encoding="utf-8"))


def _subcommand_of(tokens: list[str]) -> str | None:
    """Return the compose subcommand, skipping flags and their values.

    Shared by :func:`_named_services` and its classification guard so the two
    cannot disagree about which token is the subcommand -- two parsers that must
    agree being the defect shape this module exists to catch.

    Args:
        tokens: The full `docker compose ...` token list.

    Returns:
        The subcommand, or ``None`` if the command names none.
    """
    index = 2
    while index < len(tokens):
        token = tokens[index]
        if token in _VALUE_TAKING_COMPOSE_FLAGS:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return None


def _named_services(tokens: list[str]) -> tuple[Path | None, list[str]]:
    """Extract the compose file and any service names a command references.

    Flags that consume a value are skipped so their arguments are never
    mistaken for service names -- `ps --format '{{.Name}}'` names no service.

    Args:
        tokens: The full `docker compose ...` token list.

    Returns:
        The referenced compose file (if any) and the service names named.
    """
    assert tokens[:2] == ["docker", "compose"], (
        "this extractor only understands `docker compose ...` invocations, and "
        f"the docs now contain another docker command: {tokens}. Teach "
        "_named_services about it rather than letting it be mis-parsed silently"
    )
    compose_file: Path | None = None
    services: list[str] = []
    subcommand: str | None = None
    index = 2
    while index < len(tokens):
        token = tokens[index]
        if token in _VALUE_TAKING_COMPOSE_FLAGS:
            if token in {"-f", "--file"} and index + 1 < len(tokens):
                compose_file = _REPO_ROOT / tokens[index + 1]
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        if subcommand is None:
            subcommand = token
        elif subcommand in _SINGLE_SERVICE_SUBCOMMANDS:
            services.append(token)
            break
        elif subcommand in _SERVICE_TAKING_SUBCOMMANDS:
            services.append(token)
        index += 1
    return compose_file, services


def test_every_documented_compose_subcommand_is_classified() -> None:
    """Every compose subcommand in the docs has known service-argument shape.

    Compose subcommands differ in how many trailing tokens are service names:
    `kill pipeline` names one-or-more, `exec pipeline ls ...` names exactly one
    and then a command. An unclassified subcommand defaults to "no services",
    which silently stops checking rather than failing -- so it is named here.
    """
    known = _SERVICE_TAKING_SUBCOMMANDS | _SINGLE_SERVICE_SUBCOMMANDS
    unclassified: list[str] = []
    for documented in _select("docker"):
        subcommand = _subcommand_of(_tokens_of(documented))
        if subcommand is not None and subcommand not in known:
            unclassified.append(f"{documented.where()} -> {subcommand}")

    assert not unclassified, (
        "These compose subcommands are classified in neither "
        "_SERVICE_TAKING_SUBCOMMANDS nor _SINGLE_SERVICE_SUBCOMMANDS, so their "
        f"service arguments go unchecked: {unclassified}"
    )


def test_every_documented_compose_flag_is_classified() -> None:
    """Every compose flag in the docs is known to take a value, or not to.

    `_named_services` decides what is a service by skipping flags and their
    values. An unclassified flag therefore either swallows a real service name
    or leaks its own argument in as one -- and both mis-parse silently. Adding
    a documented `logs --tail 2 pipeline` did exactly the latter, reading `2`
    as a service. This makes the next unknown flag a named failure instead.
    """
    unclassified: list[str] = []
    known = _VALUE_TAKING_COMPOSE_FLAGS | _BOOLEAN_COMPOSE_FLAGS
    for documented in _select("docker"):
        for token in _tokens_of(documented)[2:]:
            if token.startswith("-") and token not in known:
                unclassified.append(f"{documented.where()} -> {token}")

    assert not unclassified, (
        "These compose flags are in neither _VALUE_TAKING_COMPOSE_FLAGS nor "
        f"_BOOLEAN_COMPOSE_FLAGS, so _named_services mis-parses them: "
        f"{unclassified}"
    )


def test_every_classified_compose_flag_appears_in_the_docs() -> None:
    """The flag sets expire: an entry no documented command uses fails.

    The other half of keeping a hand-written table honest. The guard above stops
    the sets being too *small*; this one stops them being too *large*, which is
    the failure mode that actually happens -- a list seeded with every flag
    someone could imagine, most entries never exercised, and no way to tell the
    load-bearing ones from the decoration. An unused classification is not free:
    it is an unchecked claim about a CLI this repo does not own.
    """
    used: set[str] = set()
    for documented in _select("docker"):
        used.update(
            token for token in _tokens_of(documented)[2:] if token.startswith("-")
        )

    classified = _VALUE_TAKING_COMPOSE_FLAGS | _BOOLEAN_COMPOSE_FLAGS
    unused = sorted(classified - used)

    assert not unused, (
        f"These compose flags are classified but appear in no documented "
        f"command: {unused}. Delete them so the sets keep describing the docs "
        f"rather than the whole Compose CLI."
    )


def test_no_documented_command_uses_the_follow_form_of_f() -> None:
    """`-f` is only ever Compose's `--file`, never `logs --follow`.

    Both classification sets are position-blind, and `-f` genuinely means two
    things: `--file` before the subcommand, `--follow` after it on `logs`. It is
    classified value-taking because every documented command opens with
    `-f deploy/docker-compose.yml`. A documented `logs -f pipeline` would
    therefore have its service swallowed as a filename and be checked against
    nothing -- silently, since a command that names no service passes.

    Rather than leave that latent, the collision is made impossible: `-f` may
    only appear before the subcommand. A doc that needs `--follow` must spell it
    out, which the classification guard then requires be added.
    """
    misplaced: list[str] = []
    for documented in _select("docker"):
        tokens = _tokens_of(documented)
        subcommand = _subcommand_of(tokens)
        if subcommand is None or _FOLLOW_AMBIGUOUS_FLAG not in tokens:
            continue
        if tokens.index(_FOLLOW_AMBIGUOUS_FLAG) > tokens.index(subcommand):
            misplaced.append(documented.where())

    assert not misplaced, (
        f"These commands use {_FOLLOW_AMBIGUOUS_FLAG!r} after the subcommand, "
        f"where Compose reads it as `--follow` rather than `--file`: "
        f"{misplaced}. Spell it `--follow` and classify it as boolean."
    )


def test_every_documented_command_can_be_lexed() -> None:
    """No documented command is silently dropped because shlex cannot lex it.

    Raised in review of PR #477. `_tokens_of` swallows shlex's `ValueError`
    into an empty token list, so an unbalanced quote or a backslash
    line-continuation would remove a command from every check below without
    anyone noticing -- the "extractor quietly stops covering a doc form"
    failure this module's own anti-vacuity philosophy warns about. This makes
    that removal loud.
    """
    unlexable = [
        documented.where()
        for documented in all_documented_commands()
        if not _tokens_of(documented)
    ]

    assert not unlexable, (
        "These documented lines could not be lexed, so they are excluded from "
        f"every check in this module: {unlexable}"
    )


def test_no_project_command_hides_behind_a_wrapper() -> None:
    """A documented `sudo windbreak ...` would be silently unchecked.

    Also from review of PR #477. `_select` matches on `tokens[0]`, so a project
    command behind `sudo`, `env` or a `time` prefix falls out of every check
    while still looking covered. Rather than guess at wrapper semantics, this
    fails loudly and says what to do.
    """
    hidden: list[str] = []
    for documented in all_documented_commands():
        tokens = _tokens_of(documented)
        for checked in ("windbreak", "docker"):
            if checked in tokens[1:]:
                hidden.append(f"{documented.where()} (found {checked!r} at index > 0)")

    assert not hidden, (
        "These documented lines invoke a checked command behind a wrapper, so "
        f"_select skips them: {hidden}. Teach _select about the wrapper rather "
        "than leaving the line unchecked."
    )


def test_operator_docs_contain_commands_to_check() -> None:
    """The extractor finds commands, so a silent regression cannot pass.

    An extractor that quietly matched nothing would make every test below
    vacuously green -- the exact "guard that cannot fail" shape epic #465 was
    written to remove.
    """
    assert len(all_documented_commands()) >= 20
    assert len(_select("windbreak")) >= 10
    assert len(_select("docker")) >= 3


@pytest.mark.parametrize(
    "documented",
    _select("windbreak"),
    ids=lambda documented: f"{documented.document}:{documented.line}",
)
def test_every_documented_windbreak_command_parses(
    documented: DocumentedCommand,
) -> None:
    """Every documented `windbreak ...` command is accepted by the real CLI.

    This is the cheap half of the issue and it is the half that would have
    caught #449's phantom operator controls the day they were written.

    Args:
        documented: The documented command under test.
    """
    tail = _substitute_placeholders(_tokens_of(documented)[1:])

    parse_with_real_cli(tail, source=documented.where())


@pytest.mark.parametrize(
    "documented",
    _select("docker"),
    ids=lambda documented: f"{documented.document}:{documented.line}",
)
def test_every_documented_compose_command_names_real_files_and_services(
    documented: DocumentedCommand,
) -> None:
    """Documented compose commands reference a real file and real services.

    `README.md:150-156` prints a transcript of `docker compose kill pipeline`
    against a stack that could not start (#445). This pins the smaller, always
    checkable half of that claim: the file and the service exist.

    Args:
        documented: The documented command under test.
    """
    tokens = _tokens_of(documented)
    compose_file, services = _named_services(tokens)

    assert compose_file is not None, (
        f"{documented.where()} runs compose without naming a file with -f, so "
        "which stack it refers to depends on the reader's working directory"
    )
    assert compose_file.is_file(), (
        f"{documented.where()} names a compose file that does not exist: "
        f"{compose_file.relative_to(_REPO_ROOT)}"
    )

    defined = set(_compose_document(compose_file)["services"])
    unknown = sorted(service for service in services if service not in defined)

    assert not unknown, (
        f"{documented.where()} names services that {compose_file.name} does "
        f"not define: {unknown} (defined: {sorted(defined)})"
    )


@pytest.mark.parametrize(
    "documented",
    [
        documented
        for documented in all_documented_commands()
        if documented.command.startswith("./scripts/")
    ],
    ids=lambda documented: f"{documented.document}:{documented.line}",
)
def test_every_documented_script_exists_and_is_executable(
    documented: DocumentedCommand,
) -> None:
    """Documented `./scripts/*.sh` invocations name a real executable script.

    Args:
        documented: The documented command under test.
    """
    tokens = _tokens_of(documented)
    script = _REPO_ROOT / tokens[0]

    assert script.is_file(), f"{documented.where()} names a script that does not exist"
    assert script.stat().st_mode & 0o111, (
        f"{documented.where()} names a script that is not executable"
    )


def test_every_documented_placeholder_has_a_substitution() -> None:
    """Every `<placeholder>` in a documented command has a known substitution.

    Without this, a new placeholder shape would fall through unsubstituted and
    either parse by luck or fail for the wrong reason -- and in both cases the
    parse check would stop meaning what it claims to mean.
    """
    found: set[str] = set()
    for documented in _select("windbreak"):
        found.update(_PLACEHOLDER.findall(documented.command))

    unknown = sorted(found - set(_PLACEHOLDER_VALUES))

    assert not unknown, (
        "These documentation placeholders have no substitution in "
        f"_PLACEHOLDER_VALUES, so they are parsed verbatim: {unknown}"
    )


def test_every_substitution_is_still_used_by_the_docs() -> None:
    """The substitution table expires: an unused entry fails.

    Keeps :data:`_PLACEHOLDER_VALUES` honest in the same way the wiring
    registry in `tests/architecture/test_composition_root_wiring.py` is kept
    honest -- a table that only grows stops describing reality.
    """
    found: set[str] = set()
    for documented in _select("windbreak"):
        found.update(_PLACEHOLDER.findall(documented.command))

    unused = sorted(set(_PLACEHOLDER_VALUES) - found)

    assert not unused, (
        f"These substitutions match no documented placeholder: {unused}. "
        "Delete them so the table keeps describing the docs."
    )


def test_unchecked_prefixes_are_environment_setup_only() -> None:
    """The exemption list stays short, readable and free of project commands.

    An exemption list is where a guard goes to die. `windbreak`, `docker` and
    `./scripts/` must never appear here, since those are precisely the claims
    this module exists to check.
    """
    assert len(_UNCHECKED_PREFIXES) <= 25
    for forbidden in ("windbreak", "docker", "docker-compose", "./scripts"):
        assert forbidden not in _UNCHECKED_PREFIXES
