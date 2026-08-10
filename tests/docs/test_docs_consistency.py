"""Docs-consistency tests making doc/code drift structurally impossible (#60).

Three drift dimensions, each checked against the *real*, running code --
never a hardcoded allowlist:

Group 1 -- doc presence & scope.
    Issue #60 mandates seven new root-level docs (``SECURITY.md``,
    ``RUNBOOK.md``, ``ARCHITECTURE.md``, ``ACCOUNTING.md``, ``EVALUATION.md``,
    ``LEGAL_AND_COMPLIANCE.md``, ``OPERATOR_WARNINGS.md``). Each must exist at
    the repo root and its opening scope paragraph must cite a SPEC section
    (a bare ``§`` followed by a digit, e.g. ``SPEC §5.2`` or ``§19``). This is
    the group that is RED until the docs land.

Group 2 -- CLI invocation drift.
    Every ``windbreak <verb> ...`` invocation appearing in a fenced code
    block or an inline code span across the doc corpus (the seven root docs,
    plus ``README.md`` and ``docs/RUNBOOK.md``) must name a real verb and
    real options on the real :func:`windbreak.main.build_parser` parser, and
    (for ``drill``, whose positional has ``choices``) a real drill name.
    Placeholder positionals (``<name>``, ``/path/to/x``) are never validated.

Group 3 -- config dotted-key drift.
    **Documented convention:** doc authors reference a real config key with
    an inline code span of the exact form `` `config.<dotted.key>` `` --
    e.g. `` `config.capital.floor_micros` `` or
    `` `config.forecast.canary.enabled` ``. The dotted path (with the leading
    ``config.`` stripped) must resolve on the real
    :class:`windbreak.config.schema.WindbreakConfig` dataclass tree. A key
    that is unimplemented or planned for later MUST NOT use this
    ``config.``-prefixed code-span form -- describe it in prose instead, or
    this test will (correctly) fail as drift.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import shlex
import typing
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.config.schema import WindbreakConfig
from windbreak.drills.catalog import DRILL_NAMES
from windbreak.main import PROCESS_CHOICES, build_parser

if TYPE_CHECKING:
    from collections.abc import Mapping

# --- Group 1: doc-set presence & scope --------------------------------------

#: The seven new root-level docs issue #60 mandates.
_NEW_ROOT_DOCS: tuple[str, ...] = (
    "SECURITY.md",
    "RUNBOOK.md",
    "ARCHITECTURE.md",
    "ACCOUNTING.md",
    "EVALUATION.md",
    "LEGAL_AND_COMPLIANCE.md",
    "OPERATOR_WARNINGS.md",
)

#: A SPEC-section citation marker: a bare ``§`` immediately followed by a
#: digit, tolerant of both ``SPEC §5.2`` and a bare ``§19``.
_SPEC_SECTION_CITATION_PATTERN = re.compile(r"§\d")

#: The doc corpus Group 2/3 scan: the seven new docs plus the two existing
#: docs that already reference the CLI and config today.
_DOC_CORPUS_RELATIVE_PATHS: tuple[str, ...] = (
    *_NEW_ROOT_DOCS,
    "README.md",
    "docs/RUNBOOK.md",
)


def _repo_root() -> Path:
    """Return the repo root, resolved relative to this test file's location.

    Returns:
        The repo root (``tests/docs/test_docs_consistency.py``'s
        great-grandparent directory).
    """
    return Path(__file__).resolve().parents[2]


def _first_scope_paragraph(doc_text: str) -> str:
    """Return a markdown doc's opening scope paragraph.

    Skips leading blank lines and ATX headings (``# ...``), then returns the
    first contiguous block of non-blank lines that follows, whitespace-joined
    into one string.

    Args:
        doc_text: The full markdown document text.

    Returns:
        The scope paragraph, or ``""`` if the doc has no body after its
        leading headings.
    """
    paragraph: list[str] = []
    started = False
    for line in doc_text.splitlines():
        stripped = line.strip()
        if not started:
            if not stripped or stripped.startswith("#"):
                continue
            started = True
        if not stripped:
            if paragraph:
                break
            continue
        paragraph.append(stripped)
    return " ".join(paragraph)


@pytest.mark.parametrize("doc_name", _NEW_ROOT_DOCS)
def test_new_root_doc_exists_and_cites_a_spec_section(doc_name: str) -> None:
    """Each issue-#60-mandated root doc exists and opens with a SPEC citation."""
    doc_path = _repo_root() / doc_name
    assert doc_path.exists(), f"{doc_name} does not exist at repo root (issue #60)"

    scope = _first_scope_paragraph(doc_path.read_text(encoding="utf-8"))
    assert _SPEC_SECTION_CITATION_PATTERN.search(scope), (
        f"{doc_name}'s opening scope paragraph must cite a SPEC section "
        f"(a '§<digit>' marker); got: {scope!r}"
    )


def test_first_scope_paragraph_skips_heading_and_blank_lines() -> None:
    """The scope paragraph is the first body text after the leading heading."""
    doc_text = "# Title\n\nThis is the scope. It cites SPEC §5.2.\n\nMore.\n"

    assert _first_scope_paragraph(doc_text) == "This is the scope. It cites SPEC §5.2."


def test_first_scope_paragraph_is_empty_when_doc_has_only_a_heading() -> None:
    """A doc with no body after its heading has an empty scope paragraph."""
    assert _first_scope_paragraph("# Title\n") == ""


# --- Group 2: CLI invocation drift ------------------------------------------

_FENCE_PATTERN = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")


def _code_spans(doc_text: str) -> list[str]:
    """Return the text of every fenced code block and inline code span.

    Fenced blocks are extracted first and stripped from the remaining text,
    so the inline-code regex never mistakes a fence's own backtick
    delimiters for an inline code span.

    Args:
        doc_text: The full markdown document (or any snippet) to scan.

    Returns:
        Every fenced block's body, followed by every remaining inline code
        span's body, in source order.
    """
    fenced = _FENCE_PATTERN.findall(doc_text)
    remaining = _FENCE_PATTERN.sub("", doc_text)
    inline = _INLINE_CODE_PATTERN.findall(remaining)
    return [*fenced, *inline]


def _logical_lines(code: str) -> list[str]:
    """Join backslash line-continuations in `code` into single logical lines.

    Args:
        code: A fenced code block's body, or an inline code span's text.

    Returns:
        One entry per logical (continuation-joined) line, in source order.
    """
    logical: list[str] = []
    buffer = ""
    for raw_line in code.splitlines():
        line = buffer + " " + raw_line.strip() if buffer else raw_line.rstrip()
        buffer = ""
        if line.endswith("\\"):
            buffer = line[:-1].rstrip()
            continue
        logical.append(line)
    if buffer:
        logical.append(buffer)
    return logical


def _strip_prompt(line: str) -> str:
    """Strip a leading shell prompt (``$ ``) from one logical command line.

    Args:
        line: One logical (continuation-joined) line.

    Returns:
        The line with a leading ``$ `` prompt removed, or the stripped line
        unchanged if it has none.
    """
    stripped = line.strip()
    if stripped.startswith("$ "):
        return stripped[2:]
    return stripped


def _safe_shlex_split(command: str) -> list[str]:
    """Return `command`'s shlex tokens, or `[]` if it fails to tokenize.

    Args:
        command: A single logical shell command line.

    Returns:
        The tokens, or an empty list on an unterminated quote (a
        markdown-authoring bug outside this drift check's scope) rather than
        propagating the exception.
    """
    if not command:
        return []
    try:
        return shlex.split(command)
    except ValueError:
        return []


#: Prefixes that launch the *same* CLI as the ``windbreak`` console script, by
#: running its module entrypoint. Documented this way, an invocation is every
#: bit as copy-pasteable -- and every bit as capable of naming a flag or a path
#: that does not exist -- so it is folded to the console-script form rather
#: than left invisible to every check built on this extractor.
_MODULE_ENTRYPOINT_PREFIXES = (
    ("python", "-m", "windbreak"),
    ("python3", "-m", "windbreak"),
)


def _normalize_entrypoint(tokens: list[str]) -> tuple[str, ...]:
    """Return `tokens` with a ``python -m windbreak`` prefix folded to the verb.

    Only the exact module ``windbreak`` is folded: ``python -m
    windbreak.riskkernel`` launches a different entrypoint with a different
    argument surface, so it stays unrecognized rather than being validated
    against the console script's parser.

    Args:
        tokens: One logical command's shlex tokens.

    Returns:
        The tokens with any module-entrypoint prefix replaced by
        ``windbreak``, or the tokens unchanged.
    """
    for prefix in _MODULE_ENTRYPOINT_PREFIXES:
        if tuple(tokens[: len(prefix)]) == prefix:
            return ("windbreak", *tokens[len(prefix) :])
    return tuple(tokens)


def extract_windbreak_invocations(doc_text: str) -> tuple[tuple[str, ...], ...]:
    """Return every ``windbreak ...`` invocation's shlex tokens in `doc_text`.

    Only scans fenced code blocks and inline code spans -- never prose --
    joins backslash line-continuations into one logical command first, folds
    the ``python -m windbreak`` module entrypoint to the console-script form,
    and keeps only commands whose first token is then exactly ``windbreak``
    *and* that carry a verb after it. A bare ``windbreak`` code span (prose
    referring to the executable by name, e.g. "resolves ``windbreak`` from
    ``PATH``") is a name reference, not an invocation, so it is skipped --
    the parser has no verbless invocation to validate it against.

    Args:
        doc_text: The full markdown document (or any snippet) to scan.

    Returns:
        Each matching invocation's shlex tokens, ``windbreak`` first, in
        source order.
    """
    invocations: list[tuple[str, ...]] = []
    for span in _code_spans(doc_text):
        for logical_line in _logical_lines(span):
            command = _strip_prompt(logical_line)
            tokens = _normalize_entrypoint(_safe_shlex_split(command))
            if len(tokens) >= 2 and tokens[0] == "windbreak":
                invocations.append(tokens)
    return tuple(invocations)


def test_extract_windbreak_invocations_skips_a_bare_name_reference() -> None:
    """A verbless ``windbreak`` code span is a name reference, not a command."""
    doc_text = "Units resolve `windbreak` from `PATH` rather than a hard path."

    assert extract_windbreak_invocations(doc_text) == ()


def test_extract_windbreak_invocations_parses_a_fenced_command() -> None:
    """A simple fenced ``windbreak`` command is tokenized correctly."""
    doc_text = "```\nwindbreak run --max-beats 3\n```"

    assert extract_windbreak_invocations(doc_text) == (
        ("windbreak", "run", "--max-beats", "3"),
    )


def test_extract_windbreak_invocations_joins_backslash_continuations() -> None:
    """A backslash-continued multi-line command becomes one invocation."""
    doc_text = "```bash\nwindbreak run \\\n  --max-beats 3\n```"

    assert extract_windbreak_invocations(doc_text) == (
        ("windbreak", "run", "--max-beats", "3"),
    )


def test_extract_windbreak_invocations_strips_a_dollar_prompt() -> None:
    """A leading ``$ `` shell prompt is stripped before tokenizing."""
    doc_text = "```\n$ windbreak kill --state-dir /tmp/state\n```"

    assert extract_windbreak_invocations(doc_text) == (
        ("windbreak", "kill", "--state-dir", "/tmp/state"),
    )


def test_extract_windbreak_invocations_matches_inline_code_spans_too() -> None:
    """An inline `` `windbreak ...` `` code span (not just a fence) is scanned."""
    doc_text = "See `windbreak run` for details."

    assert extract_windbreak_invocations(doc_text) == (("windbreak", "run"),)


def test_extract_windbreak_invocations_ignores_non_windbreak_commands() -> None:
    """A code block whose command is not ``windbreak`` yields no invocations."""
    doc_text = "```\ndocker compose up -d\n```"

    assert extract_windbreak_invocations(doc_text) == ()


def test_extract_windbreak_invocations_ignores_prose_mentions() -> None:
    """A bare prose mention of ``windbreak run`` outside any code is ignored."""
    doc_text = "Running windbreak run is not inside code, so it is ignored."

    assert extract_windbreak_invocations(doc_text) == ()


def test_extract_windbreak_invocations_folds_the_module_entrypoint() -> None:
    """``python -m windbreak <verb>`` is the same CLI, so it is extracted.

    Issue #449 was reproduced with exactly this form; leaving it unrecognized
    would keep a whole documented invocation shape out of reach of every
    check built on this extractor.
    """
    doc_text = "```bash\npython -m windbreak preflight --fixture-dir x\n```"

    assert extract_windbreak_invocations(doc_text) == (
        ("windbreak", "preflight", "--fixture-dir", "x"),
    )


def test_extract_windbreak_invocations_folds_the_python3_module_entrypoint() -> None:
    """``python3 -m windbreak`` folds the same way ``python -m`` does."""
    doc_text = "```bash\npython3 -m windbreak run --max-beats 3\n```"

    assert extract_windbreak_invocations(doc_text) == (
        ("windbreak", "run", "--max-beats", "3"),
    )


def test_extract_windbreak_invocations_ignores_a_different_module() -> None:
    """A submodule entrypoint is a different CLI, so it is not folded in."""
    doc_text = "```bash\npython -m windbreak.riskkernel --beat 1\n```"

    assert extract_windbreak_invocations(doc_text) == ()


def _find_subparsers_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction[argparse.ArgumentParser]:
    """Return the parser's one ``_SubParsersAction``.

    Args:
        parser: The parser to search.

    Returns:
        The subparsers action, whose ``.choices`` maps each verb name to its
        subparser.

    Raises:
        AssertionError: If `parser` has no subparsers action at all -- would
            mean `build_parser()` itself is broken, not doc drift.
    """
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError(f"{parser.prog} has no subparsers action")


def _verb_option_map(parser: argparse.ArgumentParser) -> dict[str, frozenset[str]]:
    """Return each subcommand verb's registered option strings.

    Args:
        parser: The parser to introspect (normally
            :func:`windbreak.main.build_parser`'s return value).

    Returns:
        A mapping from verb name to the frozenset of option strings
        (``--foo``, ``-f``, ...) registered on that verb's subparser.
    """
    subparsers_action = _find_subparsers_action(parser)
    return {
        verb: frozenset(subparser._option_string_actions)
        for verb, subparser in subparsers_action.choices.items()
    }


def _first_positional_choices(
    subparser: argparse.ArgumentParser,
) -> tuple[str, ...] | None:
    """Return one subparser's first positional argument's ``choices``.

    Args:
        subparser: A single verb's subparser.

    Returns:
        The first positional action's ``choices`` as a tuple, or ``None`` if
        the subparser has no positional argument with ``choices`` set.
    """
    for action in subparser._actions:
        is_positional = not action.option_strings
        if is_positional and action.choices is not None:
            return tuple(action.choices)
    return None


def _verb_positional_choices(
    parser: argparse.ArgumentParser,
) -> dict[str, tuple[str, ...] | None]:
    """Return each subcommand verb's first positional ``choices``, or None.

    Args:
        parser: The parser to introspect (normally
            :func:`windbreak.main.build_parser`'s return value).

    Returns:
        A mapping from verb name to that verb's first positional argument's
        ``choices`` tuple (e.g. ``drill``'s five drill names), or ``None``
        when the verb has no choice-constrained positional argument.
    """
    subparsers_action = _find_subparsers_action(parser)
    return {
        verb: _first_positional_choices(subparser)
        for verb, subparser in subparsers_action.choices.items()
    }


def _looks_like_option(token: str) -> bool:
    """Return whether `token` is a CLI option token (starts with ``-``)."""
    return token.startswith("-")


def _option_flag_name(token: str) -> str:
    """Return `token`'s option-string portion, stripping a ``=value`` suffix."""
    return token.split("=", 1)[0]


def _looks_like_placeholder(token: str) -> bool:
    """Return whether `token` is an obviously-placeholder positional value.

    Placeholders are angle-bracketed (``<name>``) or path-shaped (contain a
    ``/``), e.g. ``/path/to/cassette.json`` or ``<32-hex>``; neither should
    be checked against a subcommand's positional ``choices``.

    Args:
        token: A single shlex token.

    Returns:
        Whether the token looks like a placeholder rather than a real value.
    """
    is_bracketed = token.startswith("<") and token.endswith(">")
    return is_bracketed or "/" in token


def _verb_option_choices(
    parser: argparse.ArgumentParser,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return each verb's choice-constrained option flags and their choices.

    Args:
        parser: The parser to introspect (normally
            :func:`windbreak.main.build_parser`'s return value).

    Returns:
        A mapping from verb name to a mapping from each choice-constrained
        option string (e.g. ``--process``) to its ``choices`` tuple. Flags
        without ``choices`` (store-true flags, free-form values) are absent.
    """
    subparsers_action = _find_subparsers_action(parser)
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for verb, subparser in subparsers_action.choices.items():
        flag_choices: dict[str, tuple[str, ...]] = {}
        for action in subparser._actions:
            if action.option_strings and action.choices is not None:
                for flag in action.option_strings:
                    flag_choices[flag] = tuple(action.choices)
        result[verb] = flag_choices
    return result


def _assert_value_in_choices(
    value: str,
    choices: tuple[str, ...],
    *,
    flag: str,
    command_text: str,
    doc_name: str,
) -> None:
    """Assert a choice-constrained option's value is one of its real choices.

    Placeholder values (``<...>`` or path-shaped) are skipped, mirroring the
    positional-argument handling.

    Args:
        value: The option's argument value.
        choices: The flag's registered ``choices``.
        flag: The option string the value was passed to, for the message.
        command_text: The full invocation text, for the message.
        doc_name: The doc the invocation came from, for the message.

    Raises:
        AssertionError: If `value` is a non-placeholder outside `choices`.
    """
    if _looks_like_placeholder(value):
        return
    assert value in choices, (
        f"{doc_name}: `{command_text}` passes {value!r} to {flag!r}; "
        f"valid choices are {choices}"
    )


def _assert_option_token(
    token: str,
    *,
    verb: str,
    options: frozenset[str],
    option_choices: Mapping[str, tuple[str, ...]],
    command_text: str,
    doc_name: str,
) -> str | None:
    """Assert one ``--option[=value]`` token is real; return a pending flag.

    Args:
        token: The option token (``--flag`` or ``--flag=value``).
        verb: The invocation's verb, for the message.
        options: The verb's registered option strings.
        option_choices: The verb's choice-constrained flags mapped to choices.
        command_text: The full invocation text, for the message.
        doc_name: The doc the invocation came from, for the message.

    Returns:
        The flag name when it is choice-constrained and its value is expected
        as the *next* token (the ``--flag value`` form); otherwise ``None``.

    Raises:
        AssertionError: If the flag is unregistered, or an inline ``=value``
            is outside the flag's choices.
    """
    flag = _option_flag_name(token)
    assert flag in options, (
        f"{doc_name}: `{command_text}` uses unknown option {flag!r} "
        f"for verb {verb!r}; registered options are {sorted(options)}"
    )
    if flag not in option_choices:
        return None
    if "=" in token:
        _assert_value_in_choices(
            token.split("=", 1)[1],
            option_choices[flag],
            flag=flag,
            command_text=command_text,
            doc_name=doc_name,
        )
        return None
    return flag


def _assert_positional_token(
    token: str,
    *,
    verb: str,
    positional_choices: tuple[str, ...] | None,
    command_text: str,
    doc_name: str,
) -> None:
    """Assert a first positional value is one of the verb's real choices.

    Args:
        token: The positional token.
        verb: The invocation's verb, for the message.
        positional_choices: The verb's first positional ``choices``, or None.
        command_text: The full invocation text, for the message.
        doc_name: The doc the invocation came from, for the message.

    Raises:
        AssertionError: If `token` is a non-placeholder outside the choices.
    """
    if positional_choices is None or _looks_like_placeholder(token):
        return
    assert token in positional_choices, (
        f"{doc_name}: `{command_text}` uses unknown {verb} name "
        f"{token!r}; valid choices are {positional_choices}"
    )


def _assert_invocation_matches_parser(
    tokens: tuple[str, ...],
    *,
    verb_options: Mapping[str, frozenset[str]],
    verb_positional_choices: Mapping[str, tuple[str, ...] | None],
    doc_name: str,
    verb_option_choices: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
) -> None:
    """Assert one ``windbreak ...`` invocation matches the real parser.

    Validates the verb, every option flag, each choice-constrained option's
    value (e.g. ``--process order_gateway``), and the first positional's
    ``choices`` (e.g. the ``drill`` name). Placeholder values are skipped.

    Args:
        tokens: The shlex-tokenized invocation; ``tokens[0] == "windbreak"``.
        verb_options: Each verb's registered option strings.
        verb_positional_choices: Each verb's first positional ``choices``,
            if any.
        doc_name: The doc the invocation came from, for the assertion
            message.
        verb_option_choices: Each verb's choice-constrained option flags mapped
            to their ``choices``; ``None`` disables option-value checking.

    Raises:
        AssertionError: If the verb, an option, an option value, or a
            non-placeholder first positional is not registered on the parser.
    """
    command_text = " ".join(tokens)
    assert len(tokens) >= 2, (
        f"{doc_name}: incomplete `windbreak` invocation: {command_text!r}"
    )
    verb = tokens[1]
    assert verb in verb_options, (
        f"{doc_name}: `{command_text}` names unknown verb {verb!r}; "
        f"real verbs are {sorted(verb_options)}"
    )

    options = verb_options[verb]
    positional_choices = verb_positional_choices.get(verb)
    option_choices = (verb_option_choices or {}).get(verb, {})
    first_positional_seen = False
    pending_flag: str | None = None
    for token in tokens[2:]:
        if pending_flag is not None:
            _assert_value_in_choices(
                token,
                option_choices[pending_flag],
                flag=pending_flag,
                command_text=command_text,
                doc_name=doc_name,
            )
            pending_flag = None
        elif _looks_like_option(token):
            pending_flag = _assert_option_token(
                token,
                verb=verb,
                options=options,
                option_choices=option_choices,
                command_text=command_text,
                doc_name=doc_name,
            )
        elif not first_positional_seen:
            first_positional_seen = True
            _assert_positional_token(
                token,
                verb=verb,
                positional_choices=positional_choices,
                command_text=command_text,
                doc_name=doc_name,
            )


def test_verb_option_map_includes_run_and_its_registered_flags() -> None:
    """`run`'s option map includes its documented `--max-beats`/interval flags."""
    options = _verb_option_map(build_parser())

    assert "run" in options
    assert "--max-beats" in options["run"]
    assert "--heartbeat-interval" in options["run"]


def test_verb_option_map_includes_the_hidden_alert_test_verb() -> None:
    """The hidden `alert-test` verb is still discovered by real introspection."""
    options = _verb_option_map(build_parser())

    assert "alert-test" in options


def test_verb_positional_choices_exposes_the_real_drill_names() -> None:
    """`drill`'s positional `choices` equal the real `DRILL_NAMES` catalog."""
    choices = _verb_positional_choices(build_parser())

    assert choices["drill"] == tuple(sorted(DRILL_NAMES))


def test_verb_positional_choices_is_none_for_a_verb_without_positionals() -> None:
    """A verb with no choice-constrained positional (e.g. `kill`) maps to None."""
    choices = _verb_positional_choices(build_parser())

    assert choices["kill"] is None


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("<name>", True),
        ("/path/to/cassette.json", True),
        ("<32-hex>", True),
        ("restore-from-backup", False),
        ("--flag", False),
        ("5", False),
    ],
)
def test_looks_like_placeholder(token: str, expected: bool) -> None:
    """Angle-bracketed and path-shaped tokens are recognized as placeholders."""
    assert _looks_like_placeholder(token) is expected


def test_assert_invocation_matches_parser_accepts_a_real_run_invocation() -> None:
    """A real `run` invocation with a real flag raises nothing."""
    parser = build_parser()

    _assert_invocation_matches_parser(
        ("windbreak", "run", "--max-beats", "3"),
        verb_options=_verb_option_map(parser),
        verb_positional_choices=_verb_positional_choices(parser),
        doc_name="<test>",
    )


def test_assert_invocation_matches_parser_rejects_an_unknown_verb() -> None:
    """An invented verb is reported as drift."""
    parser = build_parser()

    with pytest.raises(AssertionError, match="unknown verb"):
        _assert_invocation_matches_parser(
            ("windbreak", "fly-to-the-moon"),
            verb_options=_verb_option_map(parser),
            verb_positional_choices=_verb_positional_choices(parser),
            doc_name="<test>",
        )


def test_assert_invocation_matches_parser_rejects_an_unknown_option() -> None:
    """A flag that is not registered on the verb's subparser is drift."""
    parser = build_parser()

    with pytest.raises(AssertionError, match="unknown option"):
        _assert_invocation_matches_parser(
            ("windbreak", "run", "--not-a-real-flag"),
            verb_options=_verb_option_map(parser),
            verb_positional_choices=_verb_positional_choices(parser),
            doc_name="<test>",
        )


def test_assert_invocation_matches_parser_rejects_an_unknown_drill_name() -> None:
    """A drill name absent from the real `DRILL_NAMES` catalog is drift."""
    parser = build_parser()

    with pytest.raises(AssertionError, match="unknown drill name"):
        _assert_invocation_matches_parser(
            ("windbreak", "drill", "not-a-real-drill"),
            verb_options=_verb_option_map(parser),
            verb_positional_choices=_verb_positional_choices(parser),
            doc_name="<test>",
        )


def test_assert_invocation_matches_parser_skips_placeholder_positionals() -> None:
    """A placeholder drill name (`<name>`) is never checked against choices."""
    parser = build_parser()

    _assert_invocation_matches_parser(
        ("windbreak", "drill", "<name>"),
        verb_options=_verb_option_map(parser),
        verb_positional_choices=_verb_positional_choices(parser),
        doc_name="<test>",
    )


def test_verb_option_choices_exposes_run_process_choices() -> None:
    """`run --process` exposes the four real SPEC process tokens as choices."""
    option_choices = _verb_option_choices(build_parser())

    assert option_choices["run"]["--process"] == PROCESS_CHOICES


@pytest.mark.parametrize(
    "tokens",
    [
        ("windbreak", "run", "--process", "order_gateway"),
        ("windbreak", "run", "--process=order_gateway"),
        ("windbreak", "run", "--process", "<name>"),
    ],
)
def test_assert_invocation_accepts_valid_option_choice_values(
    tokens: tuple[str, ...],
) -> None:
    """A real (or placeholder) `--process` value in either form passes."""
    parser = build_parser()

    _assert_invocation_matches_parser(
        tokens,
        verb_options=_verb_option_map(parser),
        verb_positional_choices=_verb_positional_choices(parser),
        verb_option_choices=_verb_option_choices(parser),
        doc_name="<test>",
    )


@pytest.mark.parametrize(
    "tokens",
    [
        ("windbreak", "run", "--process", "gateway"),
        ("windbreak", "run", "--process=gateway"),
    ],
)
def test_assert_invocation_rejects_invalid_option_choice_values(
    tokens: tuple[str, ...],
) -> None:
    """A `--process` value outside the real choices is reported as drift."""
    parser = build_parser()

    with pytest.raises(AssertionError, match="valid choices are"):
        _assert_invocation_matches_parser(
            tokens,
            verb_options=_verb_option_map(parser),
            verb_positional_choices=_verb_positional_choices(parser),
            verb_option_choices=_verb_option_choices(parser),
            doc_name="<test>",
        )


def _existing_doc_corpus_paths() -> tuple[Path, ...]:
    """Return the doc-corpus paths that currently exist on disk.

    Docs mandated by issue #60 but not yet written are skipped here (their
    *existence* is separately pinned, and failed, by Group 1) so the Group
    2/3 drift checks run over whatever part of the corpus already exists
    rather than raising a bare `FileNotFoundError`.

    Returns:
        The existing doc corpus paths, in `_DOC_CORPUS_RELATIVE_PATHS` order.
    """
    root = _repo_root()
    return tuple(
        root / relative
        for relative in _DOC_CORPUS_RELATIVE_PATHS
        if (root / relative).exists()
    )


def test_cli_invocations_in_doc_corpus_match_the_real_parser() -> None:
    """Every `windbreak ...` invocation in the doc corpus matches the real CLI.

    Covers the seven new root docs (whichever already exist), `README.md`,
    and `docs/RUNBOOK.md` -- every verb, option, and (for `drill`) positional
    name must be real, or this test fails naming the offending doc/command.
    """
    parser = build_parser()
    verb_options = _verb_option_map(parser)
    verb_positional_choices = _verb_positional_choices(parser)
    verb_option_choices = _verb_option_choices(parser)

    for doc_path in _existing_doc_corpus_paths():
        doc_text = doc_path.read_text(encoding="utf-8")
        doc_name = str(doc_path.relative_to(_repo_root()))
        for tokens in extract_windbreak_invocations(doc_text):
            _assert_invocation_matches_parser(
                tokens,
                verb_options=verb_options,
                verb_positional_choices=verb_positional_choices,
                verb_option_choices=verb_option_choices,
                doc_name=doc_name,
            )


# --- Group 3: config dotted-key drift ---------------------------------------


def _dotted_config_paths(dataclass_type: type) -> frozenset[str]:
    """Return every valid dotted path reachable from `dataclass_type`'s fields.

    Recurses into nested dataclass-typed fields (resolved via
    `typing.get_type_hints`, which handles `from __future__ import
    annotations` string annotations) and stops at the first non-dataclass
    leaf (`int`, `str`, `bool`, `tuple[...]`, `X | None`, ...). Both the
    leaf's own dotted path and, for a nested dataclass, every path beneath it
    are included.

    Args:
        dataclass_type: A dataclass type to walk (typically
            :class:`~windbreak.config.schema.WindbreakConfig`).

    Returns:
        Every valid dotted path, e.g. ``"capital.floor_micros"`` or
        ``"forecast.canary.enabled"``.
    """
    hints = typing.get_type_hints(dataclass_type)
    paths: set[str] = set()
    for field in dataclasses.fields(dataclass_type):
        field_type = hints[field.name]
        paths.add(field.name)
        if dataclasses.is_dataclass(field_type):
            nested_type = typing.cast("type", field_type)
            paths.update(
                f"{field.name}.{nested}" for nested in _dotted_config_paths(nested_type)
            )
    return frozenset(paths)


def test_dotted_config_paths_contains_known_good_keys() -> None:
    """The schema walk yields the documented example paths verbatim."""
    paths = _dotted_config_paths(WindbreakConfig)

    assert "capital.floor_micros" in paths
    assert "exchange.product_allowlist" in paths
    assert "forecast.canary.enabled" in paths
    assert "evaluation.bootstrap_confidence_ppm" in paths
    assert "risk.kill_after_consecutive_mismatches" in paths


def test_dotted_config_paths_excludes_known_bad_keys() -> None:
    """A renamed or invented field never appears in the schema walk."""
    paths = _dotted_config_paths(WindbreakConfig)

    # Renamed to `bootstrap_confidence_ppm` -- the raw fractional YAML name
    # is never a real dataclass field.
    assert "evaluation.bootstrap_confidence" not in paths
    # `product_blocklist`, not `product_denylist`.
    assert "exchange.product_denylist" not in paths
    assert "capital.not_a_real_field" not in paths


_CONFIG_KEY_REFERENCE_PATTERN = re.compile(r"`config\.([A-Za-z0-9_.]+)`")


def extract_config_key_references(doc_text: str) -> tuple[str, ...]:
    """Return every dotted config key referenced via `` `config.<path>` ``.

    This is the one explicit, documented convention for citing a real config
    key in prose: an inline code span whose content begins with the literal
    ``config.`` prefix, e.g. `` `config.capital.floor_micros` ``. A key that
    is unimplemented or planned for later MUST NOT use this form -- describe
    it in prose instead -- or the corpus-wide drift test fails.

    Args:
        doc_text: The full markdown document (or any snippet) to scan.

    Returns:
        Each referenced key's dotted path, with the leading ``config.``
        prefix already stripped, in source order.
    """
    return tuple(_CONFIG_KEY_REFERENCE_PATTERN.findall(doc_text))


def test_extract_config_key_references_finds_a_dotted_key() -> None:
    """A `` `config.<path>` `` code span yields its stripped dotted path."""
    doc_text = "See `config.capital.floor_micros` for the floor."

    assert extract_config_key_references(doc_text) == ("capital.floor_micros",)


def test_extract_config_key_references_ignores_unrelated_code_spans() -> None:
    """A code span not starting with the literal `config.` prefix is ignored."""
    doc_text = "See `windbreak.config.load_default_config` and `--config`."

    assert extract_config_key_references(doc_text) == ()


def test_config_key_references_in_doc_corpus_are_real() -> None:
    """Every `` `config.<path>` `` reference in the doc corpus is a real key.

    Covers the seven new root docs (whichever already exist), `README.md`,
    and `docs/RUNBOOK.md` -- each referenced dotted path must resolve on the
    real `WindbreakConfig` schema, or this test fails naming the offending
    doc/key (see the module docstring's `config.<dotted.key>` convention).
    """
    valid_paths = _dotted_config_paths(WindbreakConfig)

    for doc_path in _existing_doc_corpus_paths():
        doc_text = doc_path.read_text(encoding="utf-8")
        doc_name = str(doc_path.relative_to(_repo_root()))
        for dotted_path in extract_config_key_references(doc_text):
            assert dotted_path in valid_paths, (
                f"{doc_name}: `config.{dotted_path}` is not a real WindbreakConfig path"
            )
