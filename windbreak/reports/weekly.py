"""Weekly PAPER-loop report stubs, idempotent per ISO calendar week (issue #48).

:func:`write_weekly_stub` writes a dated ``weekly-YYYY-MM-DD.md`` file with
markdown section headers and ``No data yet.`` bodies, creating the output
directory when absent. :func:`maybe_write_weekly` layers ISO-week idempotence on
top: at most one file is written per ISO calendar week, so the always-on loop
can call it every beat yet only produce one report per week. The real report
bodies are a later documentation pass; the stub exists so an operator always has
a current file on disk.

A report directory that is not a directory -- a volume that came back as a
regular file -- reaches the operator as exactly one diagnosis,
:class:`WeeklyReportDirectoryError`, whichever of the write's two syscalls
notices it (issue #551).
"""

from __future__ import annotations

import errno
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import date
    from pathlib import Path

    #: A report body: a ready ``str``, a zero-arg factory that builds one lazily
    #: (invoked only on the genuine write path, #188), or ``None`` for the stub.
    WeeklyBody = str | Callable[[], str] | None

#: The date format stamped into the ``weekly-YYYY-MM-DD.md`` filename.
_DATE_FORMAT = "%Y-%m-%d"

#: The glob matching every weekly report file, for the ISO-week idempotence scan.
_WEEKLY_GLOB = "weekly-*.md"

#: The report body: section headers each carrying a ``No data yet.`` placeholder
#: (the real bodies are a later documentation pass, issue #48).
_REPORT_BODY = (
    "# Weekly report {date}\n\n"
    "## Equity vs floor\n\n"
    "No data yet.\n\n"
    "## Positions\n\n"
    "No data yet.\n\n"
    "## Decisions\n\n"
    "No data yet.\n"
)


class WeeklyReportDirectoryError(OSError):
    """Raised when the report directory is not a usable directory (issue #551).

    One physical fault -- a report volume that came back as a regular file,
    which is what a bad bind mount looks like -- used to reach an operator as
    two different diagnoses, depending on which of :func:`write_weekly_stub`'s
    two syscalls noticed it first: ``FileExistsError`` naming the *directory*
    from ``mkdir``, or ``NotADirectoryError`` naming the *report file* from the
    write, when the fault landed in the window between them. The message is
    carried to an operator on an ``AlertEmitted`` row, so one recurring fault
    was filed under two headings and the second heading named a file rather
    than the directory that was actually wrong.

    Both are normalized to this one type, carrying one message that names the
    directory. An :class:`OSError` subclass, so a caller with an existing
    ``except OSError`` branch (the ``mkdir``/write failures both already were
    one) keeps handling it, and the raise that noticed the fault is preserved
    as ``__cause__`` for a traceback reader who wants the raw errno.
    """


#: The two errnos that mean "this path is not a usable directory", and the
#: only ones :func:`write_weekly_stub` re-diagnoses. ``EEXIST`` is what
#: ``mkdir(exist_ok=True)`` raises when the target exists and is *not* a
#: directory (it swallows the exist case); ``ENOTDIR`` is what either syscall
#: raises when the path, or a parent, is a file. Every other ``OSError`` --
#: ``ENOSPC`` on a full volume, ``EACCES`` on a read-only mount -- has a
#: different remedy and reaches the operator unchanged.
_UNUSABLE_DIR_ERRNOS = frozenset({errno.EEXIST, errno.ENOTDIR})


@contextmanager
def _one_diagnosis_for(output_dir: Path) -> Iterator[None]:
    """Re-raise an unusable-directory failure as one message naming ``output_dir``.

    Wraps a *single* syscall rather than the whole write, and both of
    :func:`write_weekly_stub`'s syscalls are wrapped separately: the body fold
    that runs between them reads the ledger, so an ``OSError`` it raises is
    about a different volume entirely and must keep its own diagnosis.

    Args:
        output_dir: The report directory the diagnosis names.

    Yields:
        ``None``; the caller's block runs inside the guard.

    Raises:
        WeeklyReportDirectoryError: If the wrapped call failed because the
            report path is not a usable directory.
    """
    try:
        yield
    except OSError as exc:
        if exc.errno in _UNUSABLE_DIR_ERRNOS:
            message = (
                f"the report directory {output_dir} is not usable: the path "
                "exists and is not a directory, which is what a report volume "
                "that came back as a file looks like"
            )
            raise WeeklyReportDirectoryError(message) from exc
        raise


def _iso_week_key(today: date) -> tuple[int, int]:
    """Return the ``(iso_year, iso_week)`` key identifying ``today``'s ISO week.

    Args:
        today: The date whose ISO calendar week is keyed.

    Returns:
        The ISO year and ISO week number, so two dates in the same Mon-Sun ISO
        week compare equal while a date in a later week does not.
    """
    iso = today.isocalendar()
    return iso.year, iso.week


def _resolve_body(body: WeeklyBody, stamp: str) -> str:
    """Resolve a body argument to the exact ``str`` to persist.

    Invoked only on the genuine write path, so a callable factory -- an
    expensive whole-ledger fold (#188) -- is built exactly once, immediately
    before the write, and never on an idempotent no-op tick.

    Args:
        body: The ready ``str``, the zero-arg factory to invoke, or ``None`` for
            the default stub.
        stamp: The ``YYYY-MM-DD`` date stamp for the default stub body.

    Returns:
        The stub body when ``body`` is ``None``, the factory's returned string
        when ``body`` is callable, else ``body`` verbatim.
    """
    if body is None:
        return _REPORT_BODY.format(date=stamp)
    if callable(body):
        return body()
    return body


def write_weekly_stub(
    output_dir: Path, *, today: date, body: WeeklyBody = None
) -> Path:
    """Write ``weekly-YYYY-MM-DD.md`` for ``today``, creating ``output_dir``.

    Unconditionally overwrites any existing file for the identical ``today`` (a
    plain write, never an error). When ``body`` is ``None`` the written body is
    the default stub -- markdown section headers each with a ``No data yet.``
    placeholder; a ``str`` is written verbatim; a zero-arg callable factory is
    invoked exactly once here (on this genuine write path) and its returned
    string is written, letting a caller defer an expensive report build to the
    moment it is actually persisted (issue #55, #188).

    Args:
        output_dir: The directory the report is written into; created (with
            parents) when absent.
        today: The report date, stamped into both the filename and the default
            body.
        body: The report body to write -- a ``str``, a zero-arg factory that
            returns one, or ``None`` to write the stub.

    Returns:
        The path of the written report file.

    Raises:
        WeeklyReportDirectoryError: If ``output_dir`` is not a usable
            directory, whichever of the two syscalls below noticed it. The
            fault is one physical thing -- a report volume that came back as a
            file -- and the operator-facing message is one string naming that
            directory, not two strings naming two different paths (#551).
    """
    with _one_diagnosis_for(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
    stamp = today.strftime(_DATE_FORMAT)
    path = output_dir.joinpath(f"weekly-{stamp}.md")
    # Resolved *between* the two guards, deliberately. The fold is expensive
    # (a whole-ledger walk, #188) and it is what makes the window between the
    # two syscalls 0.188ms wide rather than two adjacent instructions -- but it
    # reads the *ledger* volume, so an `OSError` out of it is not evidence
    # about the report directory and is never re-diagnosed as one.
    content = _resolve_body(body, stamp)
    with _one_diagnosis_for(output_dir):
        path.write_text(content, encoding="utf-8")
    return path


def maybe_write_weekly(
    output_dir: Path, *, today: date, body: WeeklyBody = None
) -> Path:
    """Write this ISO week's report at most once, returning the report path.

    Idempotent per ISO calendar week: if any ``weekly-*.md`` file already exists
    for a date in ``today``'s ISO week, that file is returned untouched --
    without touching ``body`` at all, so a callable ``body`` factory is never
    invoked on a no-op tick; otherwise a fresh report is written. The always-on
    loop can therefore call this every beat yet produce exactly one report per
    week, paying for an expensive body build only when it will be persisted.

    Args:
        output_dir: The directory the report is written into (created when
            absent).
        today: The report date, whose ISO week gates whether a new file is
            written.
        body: The report body to write when a new file is created -- a ``str``,
            a zero-arg factory that returns one, or ``None`` to write the default
            stub (issue #55, #188).

    Returns:
        The path of the freshly written, or already-existing, report file.

    Raises:
        WeeklyReportDirectoryError: If a report has to be written and
            ``output_dir`` is not a usable directory (#551). The
            already-exists short-circuit reads through
            :meth:`~pathlib.Path.is_dir`, which reports a non-directory as
            simply having no reports in it, so the fault surfaces from the
            write rather than from the scan.
    """
    target_key = _iso_week_key(today)
    if output_dir.is_dir():
        for existing in output_dir.glob(_WEEKLY_GLOB):
            existing_date = datetime.strptime(
                existing.stem.removeprefix("weekly-"), _DATE_FORMAT
            ).date()
            if _iso_week_key(existing_date) == target_key:
                return existing
    return write_weekly_stub(output_dir, today=today, body=body)
