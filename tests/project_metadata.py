"""One reader for the version `pyproject.toml` declares (issue #507).

The distribution version exists in two places that a release can bump
independently -- `pyproject.toml`'s ``[project] version`` and
`windbreak/__init__.py`'s ``__version__`` -- so the tests that pin them
together must READ both rather than restate either. A third hand-copied
literal in a test file would be the same drift trap in a new place.

This module holds the `pyproject.toml` side of that pin, once, so both the
default-suite CLI tests (`tests/test_cli_version.py`) and the installed-wheel
tier (`tests/e2e/test_installed_wheel.py`) compare against the same read of the
same file instead of two parsers that could disagree.

`pyproject_path` is a parameter rather than a constant baked into the body so
the reader itself is testable: `tests/test_cli_version.py` points it at a
synthetic file carrying a different version and asserts it reports THAT one.
Without that control, a reader that ignored its input and returned
``windbreak.__version__`` would make every equality assertion built on it pass
by comparing one value with itself.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

#: Repo root, derived from this file's own location (`<root>/tests/`).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The packaging declaration a release bumps, and the wheel's metadata source.
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def declared_project_version(pyproject_path: Path = PYPROJECT_PATH) -> str:
    """Read the distribution version a `pyproject.toml` declares.

    Args:
        pyproject_path: The `pyproject.toml` to read. Defaults to this
            repository's own.

    Returns:
        The ``[project] version`` string, exactly as declared.
    """
    with pyproject_path.open("rb") as handle:
        parsed = tomllib.load(handle)
    return str(parsed["project"]["version"])
