"""Failing-first tests for `windbreak.reports.weekly` (issue #48, RED).

`windbreak/reports/` does not exist yet, so every import below fails
collection with `ModuleNotFoundError: No module named 'windbreak.reports'` --
the expected Gate 1 RED state for issue #48.

Pins:

- `write_weekly_stub(output_dir, *, today)` writes a dated
  `weekly-YYYY-MM-DD.md` file with section headers and "No data yet." bodies,
  creating `output_dir` if absent.
- `maybe_write_weekly(output_dir, *, today)` is idempotent per ISO week: two
  calls whose `today` falls in the same ISO calendar week write exactly one
  file; a `today` in a *different* ISO week writes a second, distinct file.

Issue #188 widens both functions' `body` parameter from `str | None` to
`str | Callable[[], str] | None`: a callable factory is invoked only on the
genuine write path (never when this ISO week's file already exists), so an
expensive body -- a whole-ledger evaluation/cost-meter fold -- is built only
when it will actually be persisted.
"""

from __future__ import annotations

import errno
import shutil
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.reports.weekly import WeeklyBody

#: A fixed Wednesday, so `+timedelta(days=n)` for small `n` stays in the same
#: ISO week (Mon-Sun) unless the test explicitly crosses a week boundary.
_A_WEDNESDAY = date(2026, 1, 7)

#: `_A_WEDNESDAY`'s filename stamp, and the date stamped into the stub body.
_A_WEDNESDAY_STAMP = "2026-01-07"

#: The exact bytes `write_weekly_stub` writes for `_A_WEDNESDAY` with no `body`.
#: Transcribed, not imported from the module under test: a test that read the
#: template it is pinning would pass under any edit to that template.
_A_WEDNESDAY_STUB = (
    "# Weekly report 2026-01-07\n\n"
    "## Equity vs floor\n\n"
    "No data yet.\n\n"
    "## Positions\n\n"
    "No data yet.\n\n"
    "## Decisions\n\n"
    "No data yet.\n"
)


def test_write_weekly_stub_creates_output_dir_if_absent(tmp_path: Path) -> None:
    """`write_weekly_stub` creates `output_dir` when it does not yet exist."""
    from windbreak.reports.weekly import write_weekly_stub

    output_dir = tmp_path / "reports"
    assert not output_dir.exists()

    write_weekly_stub(output_dir, today=_A_WEDNESDAY)

    assert output_dir.is_dir()


def test_write_weekly_stub_filename_is_dated_weekly_markdown(tmp_path: Path) -> None:
    """The written file is named `weekly-YYYY-MM-DD.md`, dated `today`."""
    from windbreak.reports.weekly import write_weekly_stub

    path = write_weekly_stub(tmp_path, today=_A_WEDNESDAY)

    assert path.name == "weekly-2026-01-07.md"
    assert path.parent == tmp_path
    assert path.exists()


def test_write_weekly_stub_body_has_section_headers_and_no_data_yet(
    tmp_path: Path,
) -> None:
    """The written body carries markdown section headers, each with a
    "No data yet." placeholder -- there is no real data to report yet.
    """
    from windbreak.reports.weekly import write_weekly_stub

    path = write_weekly_stub(tmp_path, today=_A_WEDNESDAY)

    body = path.read_text(encoding="utf-8")
    assert body.count("#") >= 1
    assert "No data yet." in body


def test_write_weekly_stub_overwrites_on_a_repeated_call_for_the_same_date(
    tmp_path: Path,
) -> None:
    """Calling `write_weekly_stub` twice for the identical `today` is a plain
    overwrite (unconditional -- unlike `maybe_write_weekly`'s idempotence),
    never an error or a second file.
    """
    from windbreak.reports.weekly import write_weekly_stub

    first = write_weekly_stub(tmp_path, today=_A_WEDNESDAY)
    second = write_weekly_stub(tmp_path, today=_A_WEDNESDAY)

    assert first == second
    assert len(list(tmp_path.glob("weekly-*.md"))) == 1


def test_maybe_write_weekly_writes_exactly_one_file_per_iso_week(
    tmp_path: Path,
) -> None:
    """Two calls whose `today` falls in the same ISO week write one file."""
    from datetime import timedelta

    from windbreak.reports.weekly import maybe_write_weekly

    maybe_write_weekly(tmp_path, today=_A_WEDNESDAY)
    maybe_write_weekly(
        tmp_path, today=_A_WEDNESDAY + timedelta(days=2)
    )  # same ISO week

    assert len(list(tmp_path.glob("weekly-*.md"))) == 1


def test_maybe_write_weekly_writes_a_second_file_for_a_later_iso_week(
    tmp_path: Path,
) -> None:
    """A `today` one ISO week later writes a second, distinct file."""
    from datetime import timedelta

    from windbreak.reports.weekly import maybe_write_weekly

    maybe_write_weekly(tmp_path, today=_A_WEDNESDAY)
    maybe_write_weekly(tmp_path, today=_A_WEDNESDAY + timedelta(days=7))

    assert len(list(tmp_path.glob("weekly-*.md"))) == 2


def test_maybe_write_weekly_returns_the_written_or_existing_path(
    tmp_path: Path,
) -> None:
    """`maybe_write_weekly` returns a real, existing path either way (freshly
    written, or the already-written-this-week file left untouched).
    """
    from windbreak.reports.weekly import maybe_write_weekly

    first_path = maybe_write_weekly(tmp_path, today=_A_WEDNESDAY)
    second_path = maybe_write_weekly(tmp_path, today=_A_WEDNESDAY)

    assert first_path.exists()
    assert second_path.exists()
    assert first_path == second_path


# ---------------------------------------------------------------------------
# Callable `body` factory (issue #188): built only on the genuine write path.
# ---------------------------------------------------------------------------


def test_write_weekly_stub_invokes_a_callable_body_exactly_once(tmp_path: Path) -> None:
    """`write_weekly_stub` (the unconditional-write primitive) accepts a
    zero-arg callable `body` and invokes it exactly once, writing its
    returned string -- never the callable object itself.
    """
    from windbreak.reports.weekly import write_weekly_stub

    calls: list[int] = []

    def _factory() -> str:
        calls.append(1)
        return "# stub body\n"

    path = write_weekly_stub(tmp_path, today=_A_WEDNESDAY, body=_factory)

    assert len(calls) == 1
    assert path.read_text(encoding="utf-8") == "# stub body\n"


def test_maybe_write_weekly_invokes_a_callable_body_exactly_once_on_a_genuine_write(
    tmp_path: Path,
) -> None:
    """A callable `body` factory is invoked exactly once when
    `maybe_write_weekly` actually writes this ISO week's first file.
    """
    from windbreak.reports.weekly import maybe_write_weekly

    calls: list[int] = []

    def _factory() -> str:
        calls.append(1)
        return "# built body\n"

    path = maybe_write_weekly(tmp_path, today=_A_WEDNESDAY, body=_factory)

    assert len(calls) == 1
    assert path.read_text(encoding="utf-8") == "# built body\n"


def test_maybe_write_weekly_never_invokes_a_callable_body_when_this_weeks_file_exists(
    tmp_path: Path,
) -> None:
    """When this ISO week's file already exists, a callable `body` factory is
    NOT invoked at all -- the entire point of deferring the body to a
    callable is to skip paying for an expensive fold on an idempotent no-op
    tick.

    The first call also passes a callable (not a plain `str`), so this test
    starts on the same genuine-write path as the test above and fails there
    today (`TypeError`) rather than passing trivially: today's
    `maybe_write_weekly` already skips touching `body` at all on its
    already-exists short-circuit, so a version of this test whose *first*
    call used a plain `str` body would pass before callable support exists
    at all -- exactly the "passes before the code exists" trap this suite
    must avoid.
    """
    from datetime import timedelta

    from windbreak.reports.weekly import maybe_write_weekly

    first_calls: list[int] = []

    def _first_factory() -> str:
        first_calls.append(1)
        return "# first body\n"

    first_path = maybe_write_weekly(tmp_path, today=_A_WEDNESDAY, body=_first_factory)
    assert len(first_calls) == 1
    assert first_path.read_text(encoding="utf-8") == "# first body\n"

    second_calls: list[int] = []

    def _second_factory() -> str:
        second_calls.append(1)
        return "# should never be built\n"

    second_path = maybe_write_weekly(
        tmp_path, today=_A_WEDNESDAY + timedelta(days=1), body=_second_factory
    )

    assert second_calls == []
    assert second_path == first_path
    assert second_path.read_text(encoding="utf-8") == "# first body\n"


# ---------------------------------------------------------------------------
# One physical fault, one operator-facing diagnosis (issue #551).
#
# A report volume that came back as a regular file -- what a bad bind mount
# looks like -- used to be reported two ways depending on which line noticed
# it: `FileExistsError` naming the *directory* from `mkdir`, or
# `NotADirectoryError` naming the *report file* from `write_text` when the
# fault landed in the window between them (0.188ms wide, because the body
# fold in between is a whole-ledger walk, #188). The message reaches an
# operator on an `AlertEmitted` row, so one fault filed under two headings is
# the defect; the assertions below are on that rendered text.
# ---------------------------------------------------------------------------


def _fault_the_volume(report_dir: Path) -> None:
    """Replace `report_dir` with a regular file, as a bad bind mount does.

    Args:
        report_dir: The report directory to replace with an empty file.
    """
    if report_dir.is_dir():
        shutil.rmtree(report_dir)
    report_dir.write_text("", encoding="utf-8")


def _repair_the_volume(report_dir: Path) -> None:
    """Restore `report_dir` to a real, empty directory.

    Args:
        report_dir: The report path currently occupied by a regular file.
    """
    report_dir.unlink()
    report_dir.mkdir(parents=True)


def _rendered_raise(call: Callable[[], object]) -> str:
    """Return the operator-facing text `call`'s raise reaches an alert as.

    Rendered exactly as `windbreak.main.BeatSupervisor.observe` renders a
    raising beat, which is the string that lands on the `AlertEmitted` row an
    operator greps -- so this compares what an operator actually reads, not
    what an exception happens to be typed as.

    Args:
        call: The zero-arg call expected to raise an `OSError`.

    Returns:
        The exception's type name and message, `Type: message`.

    Raises:
        AssertionError: If `call` did not raise, meaning the fault under test
            was never in place and any comparison of its result proves nothing.
    """
    try:
        call()
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    message = "the call succeeded, so the fault it was meant to hit was absent"
    raise AssertionError(message)


def test_a_report_volume_that_is_a_file_reads_the_same_whichever_line_notices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One physical fault produces one exception type and one message.

    The *same* fault on the *same* path is induced at both points it can land:
    before `mkdir`, and inside the window between `mkdir` and the write. The
    in-window injection is deterministic rather than a sleep race -- it is
    driven from a wrapper around the real `_resolve_body`, the whole-ledger
    fold that runs in that window, which delegates to the real function so the
    write path under test is the production one.

    Both rendered texts are asserted equal *and* asserted verbatim, so this
    cannot pass by both sides degenerating to the same wrong thing, and it
    cannot pass on the pre-fix code by an exception happening to subclass what
    is asserted: the pre-fix texts name two different type names and two
    different paths.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: Injects the in-window fault through `_resolve_body`.
    """
    from windbreak.reports import weekly

    report_dir = tmp_path / "reports"
    expected = (
        "WeeklyReportDirectoryError: the report directory "
        f"{report_dir} is not usable: the path exists and is not a directory, "
        "which is what a report volume that came back as a file looks like"
    )

    _fault_the_volume(report_dir)
    noticed_by_mkdir = _rendered_raise(
        lambda: weekly.write_weekly_stub(report_dir, today=_A_WEDNESDAY)
    )

    _repair_the_volume(report_dir)
    from windbreak.reports.weekly import _resolve_body as real_resolve_body

    def _fault_then_resolve(body: WeeklyBody, stamp: str) -> str:
        _fault_the_volume(report_dir)
        return real_resolve_body(body, stamp)

    monkeypatch.setattr(weekly, "_resolve_body", _fault_then_resolve)
    noticed_by_write = _rendered_raise(
        lambda: weekly.write_weekly_stub(report_dir, today=_A_WEDNESDAY)
    )

    assert noticed_by_mkdir == noticed_by_write
    assert noticed_by_mkdir == expected
    assert f"weekly-{_A_WEDNESDAY_STAMP}.md" not in noticed_by_mkdir


def test_the_unguarded_two_step_write_diagnosed_one_fault_two_ways(
    tmp_path: Path,
) -> None:
    """The negative: the pre-fix sequence really does split one fault in two.

    The two syscalls `write_weekly_stub` makes are replayed here directly --
    test-owned scaffolding, so the production code no longer has to contain
    the defect for it to be demonstrated -- with the same fault landing before
    the first and between the two. Their rendered texts differ in *both*
    dimensions an operator greps on: the exception type name, and the path the
    message names (the directory that is wrong, versus a report file that is
    merely downstream of it).

    Kept portable rather than transcribing `strerror`: the wording of "File
    exists" and "Not a directory" is the platform's, and pinning it would
    redden CI on a platform that words it differently -- which is the failure
    mode `_induced_failure_text` was already written to avoid.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    report_dir = tmp_path / "reports"
    report_file = report_dir / f"weekly-{_A_WEDNESDAY_STAMP}.md"

    _fault_the_volume(report_dir)
    noticed_by_mkdir = _rendered_raise(
        lambda: report_dir.mkdir(parents=True, exist_ok=True)
    )

    _repair_the_volume(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    _fault_the_volume(report_dir)
    noticed_by_write = _rendered_raise(
        lambda: report_file.write_text(_A_WEDNESDAY_STUB, encoding="utf-8")
    )

    assert noticed_by_mkdir != noticed_by_write
    assert noticed_by_mkdir.startswith("FileExistsError: ")
    assert noticed_by_write.startswith("NotADirectoryError: ")
    assert noticed_by_mkdir.endswith(f"'{report_dir}'")
    assert noticed_by_write.endswith(f"'{report_file}'")


def test_the_normal_write_path_is_byte_identical(tmp_path: Path) -> None:
    """The unified guard changes nothing an operator's report file contains.

    Full equality on the written bytes, the filename and the returned path --
    not the absence of an exception, which any half-written file would also
    satisfy.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    from windbreak.reports.weekly import write_weekly_stub

    report_dir = tmp_path / "reports"

    path = write_weekly_stub(report_dir, today=_A_WEDNESDAY)

    assert path == report_dir / f"weekly-{_A_WEDNESDAY_STAMP}.md"
    assert path.read_bytes() == _A_WEDNESDAY_STUB.encode("utf-8")
    assert sorted(entry.name for entry in report_dir.iterdir()) == [
        f"weekly-{_A_WEDNESDAY_STAMP}.md"
    ]


def test_a_body_factory_failure_is_never_diagnosed_as_a_directory_fault(
    tmp_path: Path,
) -> None:
    """A raising body fold keeps its own diagnosis, and leaves no report behind.

    The fold between the two syscalls reads the ledger, so it can raise an
    `OSError` about a *different* volume -- `ENOTDIR` on the ledger path, say.
    Laundering that into the report-directory diagnosis would be the same
    defect this issue fixes, pointing the other way, so the guard covers the
    two syscalls and not the fold between them. This test is what makes that
    placement load-bearing rather than incidental.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    from windbreak.reports import weekly

    report_dir = tmp_path / "reports"
    ledger_volume = str(tmp_path / "ledger" / "windbreak.db")

    def _factory() -> str:
        raise NotADirectoryError(errno.ENOTDIR, "Not a directory", ledger_volume)

    with pytest.raises(NotADirectoryError) as caught:
        weekly.write_weekly_stub(report_dir, today=_A_WEDNESDAY, body=_factory)

    assert not isinstance(caught.value, weekly.WeeklyReportDirectoryError)
    assert caught.value.filename == ledger_volume
    assert list(report_dir.iterdir()) == []


def test_a_write_failure_that_is_not_a_directory_fault_keeps_its_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full volume is still reported as a full volume, not as a bad mount.

    The guard is keyed on the two errnos that mean "this path is not a usable
    directory"; every other `OSError` the write raises reaches the operator
    unchanged. Without that key, an `ENOSPC` would be reported as a report
    volume that came back as a file, which is a different remedy entirely.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: Replaces the write with a full-volume failure.
    """
    from windbreak.reports import weekly

    report_dir = tmp_path / "reports"

    def _no_space(self: Path, *_args: object, **_kwargs: object) -> int:
        raise OSError(errno.ENOSPC, "No space left on device", str(self))

    monkeypatch.setattr(Path, "write_text", _no_space)

    with pytest.raises(OSError, match="No space left on device") as caught:
        weekly.write_weekly_stub(report_dir, today=_A_WEDNESDAY)

    assert caught.value.errno == errno.ENOSPC
    assert not isinstance(caught.value, weekly.WeeklyReportDirectoryError)


def test_maybe_write_weekly_reports_the_directory_fault_the_same_way(
    tmp_path: Path,
) -> None:
    """The loop's entry point carries the unified diagnosis, not a raw errno.

    `run_single_tick` calls `maybe_write_weekly`, never `write_weekly_stub`
    directly, so the property has to hold at the door the tick actually uses.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    from windbreak.reports import weekly

    report_dir = tmp_path / "reports"
    _fault_the_volume(report_dir)

    with pytest.raises(weekly.WeeklyReportDirectoryError) as caught:
        weekly.maybe_write_weekly(report_dir, today=_A_WEDNESDAY)

    assert str(caught.value) == (
        f"the report directory {report_dir} is not usable: the path exists and "
        "is not a directory, which is what a report volume that came back as a "
        "file looks like"
    )
