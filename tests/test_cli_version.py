"""`windbreak --version` answers "which build is this?" (issues #507, #467).

`build_parser` registered ``prog="windbreak"`` and a REQUIRED ``command``
positional and no ``--version`` action at all, so the flag an operator typed
was not merely unimplemented -- it was never mentioned. The missing positional
errored first, and the console script exited 2 with a usage line about
``command``. #467's acceptance criterion ("`--version` output matches the
version in `pyproject.toml`, pinned") could not be written against it.

WHICH SOURCE IS AUTHORITATIVE, AND WHY

Two candidates exist, and they are not interchangeable:

* ``importlib.metadata.version("windbreak")`` -- the INSTALLED DISTRIBUTION's
  metadata, stamped into the wheel's ``.dist-info`` at build time from
  `pyproject.toml`. This is what `pip show` reports and what a container image
  carries.
* ``windbreak.__version__`` -- a string in the SOURCE TREE that happens to be
  shipped inside the wheel.

They can genuinely disagree: in an install whose source tree has since been
edited, and in this repository's own test environment, where the checkout is on
``sys.path`` and no ``windbreak`` distribution is installed at all
(``importlib.metadata.version`` raises ``PackageNotFoundError`` there).

The flag exists so that an operator holding a DEPLOYED ARTIFACT can ask it
which build it is, so the installed distribution's metadata wins whenever there
is one, and the source declaration is the fallback for a checkout that was
never installed. Both branches are reachable in production -- the wheel and the
container image take the first, a checkout run takes the second -- and both are
exercised below with the two sources set to DIFFERENT values, so no assertion
here can pass by reading one source twice.

HOW THE PIN TO `pyproject.toml` IS CLOSED

Neither side is hand-restated:

* `test_the_package_dunder_version_matches_the_pyproject_declaration` pins
  ``windbreak.__version__`` (read from the module) to ``[project] version``
  (read from the file by :func:`tests.project_metadata.declared_project_version`),
  so a release that bumps one and forgets the other fails;
* `tests/e2e/test_installed_wheel.py` runs the INSTALLED console script out of
  a clean virtualenv and asserts its stdout equals that same declaration --
  the metadata branch, proven from the artifact rather than from the checkout.
"""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING

import pytest

import windbreak
from tests.docs.test_docs_consistency import _find_subparsers_action
from tests.project_metadata import declared_project_version
from windbreak.main import _COMMAND_HANDLERS, build_parser, main

if TYPE_CHECKING:
    from pathlib import Path

#: A stand-in for an installed distribution's metadata version. Deliberately
#: unlike :data:`_SOURCE_SENTINEL` and unlike the real declared version, so an
#: assertion cannot pass by reading the wrong source.
_METADATA_SENTINEL = "9.9.9+from-installed-metadata"

#: A stand-in for `windbreak.__version__`, likewise deliberately distinct.
_SOURCE_SENTINEL = "0.0.0+from-source-tree"

#: The exit status argparse uses for a usage error, which is what
#: `windbreak --version` used to produce. Asserting `SystemExit` alone cannot
#: tell it apart from the 0 the flag must now exit with.
_USAGE_ERROR_EXIT_CODE = 2


def _registered_subcommands() -> tuple[str, ...]:
    """Enumerate the subcommand names `build_parser` actually registers.

    Derived from the parser's own subparsers action rather than listed by
    hand: a hand-written list of verbs is exactly what a change to the CLI
    surface would leave stale and green.

    Returns:
        Every registered subcommand name, sorted.
    """
    return tuple(sorted(_find_subparsers_action(build_parser()).choices))


def _version_this_environment_reports() -> str:
    """State, independently of `windbreak.main`, what `--version` must print.

    The rule restated as a specification rather than imported from the code
    under test: the installed distribution's metadata when there is an
    installed distribution, else the source tree's declaration. Which branch
    applies depends on how the suite is being run -- this repository's own
    checkout-based run takes the second -- so it is resolved here rather than
    assumed. The PRECEDENCE between the two is pinned separately, by the two
    tests that force them to differ.

    Returns:
        The version string `--version` must emit in this environment.
    """
    try:
        return metadata.version("windbreak")
    except metadata.PackageNotFoundError:
        return windbreak.__version__


def test_version_flag_exits_zero_without_a_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`windbreak --version` prints a version and exits 0, with no subcommand.

    Driven through :func:`windbreak.main.main` -- the dispatch path the console
    script's entry point calls -- rather than through `build_parser`, because
    the defect was observable only as the process's exit status.

    The exit CODE is asserted, not merely that `SystemExit` was raised:
    argparse exits 0 for `--version` and 2 for the usage error this used to be,
    and both raise `SystemExit`.

    Args:
        capsys: pytest's stdout/stderr capture.
    """
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0, (
        f"`windbreak --version` exited {exit_info.value.code!r}; "
        f"{_USAGE_ERROR_EXIT_CODE} is argparse's usage error, which is the "
        "defect issue #507 reported"
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == _version_this_environment_reports()
    assert captured.err == "", (
        f"`--version` wrote to stderr: {captured.err!r} -- it is not an error"
    )


def test_version_prefers_the_installed_distribution_over_the_source_tree(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With a distribution installed, its metadata is what gets reported.

    The two candidate sources are forced to DIFFERENT values, which is the
    whole point: they read ``0.1.0`` alike in this repository today, so a test
    that left them equal would pass while reading either one -- or the same one
    twice.

    Args:
        monkeypatch: pytest's patching fixture.
        capsys: pytest's stdout/stderr capture.
    """
    asked_for: list[str] = []

    def _installed(distribution_name: str) -> str:
        asked_for.append(distribution_name)
        return _METADATA_SENTINEL

    monkeypatch.setattr("windbreak.main.metadata.version", _installed)
    monkeypatch.setattr("windbreak.main._source_version", _SOURCE_SENTINEL)

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == _METADATA_SENTINEL
    assert asked_for == ["windbreak"], (
        f"the version was looked up for {asked_for!r}, not for the "
        "distribution this CLI is shipped as"
    )


def test_version_falls_back_to_the_source_declaration_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A checkout with no installed distribution still reports its version.

    ``importlib.metadata.version`` raises `PackageNotFoundError` when the
    distribution is absent -- the state of this repository's own test
    environment, and of any clone run before `pip install`. Reporting nothing,
    or raising, would make the flag useless exactly where a developer reaches
    for it.

    Patching the source declaration to a sentinel rather than accepting the
    real ``0.1.0`` is deliberate: it is what distinguishes reading
    ``windbreak.__version__`` from restating its current value in `main.py`.

    Args:
        monkeypatch: pytest's patching fixture.
        capsys: pytest's stdout/stderr capture.
    """

    def _absent(distribution_name: str) -> str:
        raise metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr("windbreak.main.metadata.version", _absent)
    monkeypatch.setattr("windbreak.main._source_version", _SOURCE_SENTINEL)

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == _SOURCE_SENTINEL


def test_the_package_dunder_version_matches_the_pyproject_declaration() -> None:
    """The two declarations of the version agree (issue #507's drift trap).

    ``pyproject.toml``'s ``[project] version`` is what the wheel's metadata is
    built from; ``windbreak.__version__`` is what a source checkout reports.
    Nothing in the packaging makes them equal, so a release that bumps one and
    not the other must fail here rather than ship two answers to "which build
    is this?".

    Both sides are READ -- the file by
    :func:`tests.project_metadata.declared_project_version`, the module by
    attribute access -- and that reader's independence is proven by
    :func:`test_the_declared_version_reader_reads_the_file_it_is_given`.
    """
    assert windbreak.__version__ == declared_project_version(), (
        "windbreak/__init__.py and pyproject.toml declare different versions"
    )


def test_the_declared_version_reader_reads_the_file_it_is_given(
    tmp_path: Path,
) -> None:
    """The positive control for the pin above: the reader reports its input.

    Both declarations read ``0.1.0`` today, so the equality assertion in
    :func:`test_the_package_dunder_version_matches_the_pyproject_declaration`
    would pass even if the reader ignored the file and returned
    ``windbreak.__version__``. Pointing it at a synthetic `pyproject.toml`
    carrying a different version proves the two reads are independent, and so
    that drift between them is detectable rather than merely asserted.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    drifted_version = "0.0.0+drifted"
    drifted = tmp_path / "pyproject.toml"
    drifted.write_text(
        f'[project]\nname = "windbreak"\nversion = "{drifted_version}"\n',
        encoding="utf-8",
    )

    observed = declared_project_version(drifted)

    assert observed == drifted_version
    assert observed != windbreak.__version__, (
        "the control value coincides with the real declaration, so this test "
        "proves nothing -- pick a different one"
    )


def test_the_top_level_parser_gained_only_the_version_option() -> None:
    """`--version` is the only new option on the top-level parser.

    Pins the CLI surface change exactly: ``-h``/``--help`` and ``--version``,
    and nothing else, are the flags `windbreak` itself accepts. Subcommand
    options are unaffected and are covered by
    :func:`test_every_registered_subcommand_still_parses`.
    """
    parser = build_parser()

    assert set(parser._option_string_actions) == {"-h", "--help", "--version"}


def test_every_registered_subcommand_still_has_a_handler() -> None:
    """The registered verbs and the dispatch table are the same set.

    Both sides are derived -- the verbs from the parser's subparsers action,
    the handlers from :data:`windbreak.main._COMMAND_HANDLERS` -- so a verb
    that lost its registration (or its handler) fails here instead of raising
    `KeyError` in front of an operator.
    """
    registered = _registered_subcommands()

    assert registered, "build_parser registered no subcommands at all"
    assert set(registered) == set(_COMMAND_HANDLERS), (
        "the parser's subcommands and main's dispatch table disagree: "
        f"parser-only={sorted(set(registered) - set(_COMMAND_HANDLERS))}, "
        f"handler-only={sorted(set(_COMMAND_HANDLERS) - set(registered))}"
    )


@pytest.mark.parametrize("subcommand", _registered_subcommands())
def test_every_registered_subcommand_still_parses(subcommand: str) -> None:
    """Each verb the parser registers still parses under the top-level parser.

    Parametrized over the DERIVED verb list, so a subcommand added later is
    covered without editing this test. ``--help`` is the one argument every
    subparser accepts regardless of its own required arguments, and reaching it
    means the top-level parser routed the verb to a real subparser: exit 0, not
    the usage error a shadowed or unregistered verb would produce.

    Args:
        subcommand: One registered subcommand name.
    """
    parser = build_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args([subcommand, "--help"])

    assert exit_info.value.code == 0
    assert _find_subparsers_action(parser).choices[subcommand].prog == (
        f"windbreak {subcommand}"
    )
