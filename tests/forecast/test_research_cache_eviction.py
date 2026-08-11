"""Tests for the research fetch cache's byte bound (issue #453).

`ResearchCache` was a write-only store with no size cap, no age bound, no
entry-count bound and no sweep, rooted -- in the shipped compose stack -- on
the same named volume as the hash-chained ledger. This module pins the bound
that stops it competing with the audit trail for that volume:

* **Bytes, not entries** -- the cap is a *total byte* ceiling on the cache's
  own entries. `test_eviction_holds_the_byte_cap_not_an_entry_count` fails any
  implementation that bounds the wrong dimension.
* **Oldest first** -- eviction orders by modification time. The fixture's four
  entries carry pairwise-distinct, non-dividing sizes deliberately *not*
  ordered with their ages, so an implementation that evicted largest-first,
  smallest-first or newest-first would leave a different survivor set.
* **The just-written entry is never the victim** -- the entry a fetch in the
  current forecast just produced survives even when it alone exceeds the cap
  (in which case the cap is announced as unhonourable rather than silently
  broken).
* **Nothing that is not provably a cache entry is ever deleted** -- foreign
  files, subdirectories, and a symlink named like an entry but resolving
  outside the jail all survive a sweep; the existing `_is_within_root` jail is
  re-checked on every delete path.
* **Fail closed on the capability, never on the process** -- an unreadable or
  unwritable cache directory degrades the *cache*, loudly, while the fetch it
  was archiving still returns its content. The loop keeps beating.
* **The bound is configured, not compiled in** -- the operator's
  `forecast.research.cache_max_bytes` is proved to reach the live cache
  through `build_live_research_tools`, the real composition, not the pure
  builder beside it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.config.schema import (
    DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    WindbreakConfig,
)
from windbreak.forecast.providers.http_cassettes import HttpResponse
from windbreak.forecast.sandbox import (
    ResearchCache,
    ResearchTools,
    SandboxPathViolationError,
    build_research_tools,
)
from windbreak.scheduler.provider_wiring import (
    LiveProviderHttp,
    build_live_research_tools,
)

if TYPE_CHECKING:
    from windbreak.forecast.providers.http_cassettes import HttpRequest

# --- Fixture corpus ------------------------------------------------------------
#
# Four entries with pairwise-distinct, non-dividing sizes whose ordering by
# size is deliberately unrelated to their ordering by age. Only an
# oldest-first *byte* eviction leaves {"gamma", "delta"}:
#
#   policy          evicts            survivors            total
#   oldest-first    alpha, beta       gamma, delta           518
#   largest-first   alpha, gamma      beta, delta            312
#   smallest-first  beta, delta, gamma  alpha                401
#   newest-first    delta, gamma      alpha, beta            502
#
# so a single assertion on the survivor set discriminates all four.
_SIZES: dict[str, int] = {"alpha": 401, "beta": 101, "gamma": 307, "delta": 211}

#: The seeds in age order, oldest first. Deliberately not alphabetical, so an
#: implementation that sorted by *filename* rather than by age would fail.
_AGE_ORDER: tuple[str, ...] = ("alpha", "beta", "gamma", "delta")

#: Total bytes of the whole fixture corpus.
_CORPUS_BYTES = 1020

#: A cap that admits only the two newest entries (307 + 211 = 518).
_CAP_ADMITTING_TWO_NEWEST = 600

#: Bytes surviving under :data:`_CAP_ADMITTING_TWO_NEWEST`.
_SURVIVING_BYTES = 518

#: One nanosecond-resolution step between two fixture entries' ages. Large
#: enough that no filesystem's timestamp granularity can collapse two entries
#: onto one age and make an age-ordered eviction indistinguishable from an
#: arbitrary one.
_AGE_STEP_NS = 1_000_000_000

#: An arbitrary, fixed base modification time (nanoseconds since the epoch).
#: Fixed rather than "now" so the corpus is byte-identical between runs.
_BASE_MTIME_NS = 1_700_000_000_000_000_000

#: The repository root, three levels up from `tests/forecast/<this file>`.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Bytes in one mebibyte, for checking the runbook's unit gloss.
_BYTES_PER_MIB = 1024 * 1024


def _entry_name(seed: str) -> str:
    """Return the cache-entry filename a payload for ``seed`` is stored under.

    Independently restates the production filename shape (a sha256 hex digest
    plus ``.txt``) so a change to that shape fails these tests rather than
    silently making the corpus invisible to the sweep.

    Args:
        seed: The distinguishing token the digest is taken over.

    Returns:
        A ``<sha256-hex>.txt`` filename.
    """
    return f"{hashlib.sha256(seed.encode('utf-8')).hexdigest()}.txt"


def _write_corpus(root: Path) -> dict[str, Path]:
    """Write the four-entry fixture corpus with distinct sizes and ages.

    Args:
        root: The cache root the entries are written directly into.

    Returns:
        The written path for each seed, keyed by seed.
    """
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for age_index, seed in enumerate(_AGE_ORDER):
        path = root.joinpath(_entry_name(seed))
        path.write_text("x" * _SIZES[seed], encoding="utf-8")
        mtime_ns = _BASE_MTIME_NS + age_index * _AGE_STEP_NS
        os.utime(path, ns=(mtime_ns, mtime_ns))
        written[seed] = path
    return written


def _surviving_seeds(root: Path) -> set[str]:
    """Return the seeds whose cache entries still exist under ``root``.

    Args:
        root: The cache root to inspect.

    Returns:
        The set of seeds still present.
    """
    return {seed for seed in _SIZES if root.joinpath(_entry_name(seed)).exists()}


def _entry_bytes(root: Path) -> int:
    """Return the total size of the fixture corpus entries still under ``root``.

    Args:
        root: The cache root to inspect.

    Returns:
        The summed byte size of the surviving fixture entries.
    """
    return sum(_SIZES[seed] for seed in _surviving_seeds(root))


# --- The byte cap ---------------------------------------------------------------


def test_a_corpus_under_the_cap_is_left_entirely_alone(tmp_path: Path) -> None:
    """A store that leaves the total at or below the cap evicts nothing.

    The positive control for every eviction assertion below: with the cap set
    to the corpus total plus the new entry, all five entries survive, so a
    sweep that deleted indiscriminately -- or one that never ran -- is
    distinguishable from one that ran and correctly found nothing to do.
    """
    root = tmp_path / "research-cache"
    _write_corpus(root)
    new_payload = "y" * 53
    cache = ResearchCache(root=root, max_bytes=_CORPUS_BYTES + len(new_payload))

    cache.store(_entry_name("epsilon"), new_payload)

    assert _surviving_seeds(root) == set(_SIZES)
    assert root.joinpath(_entry_name("epsilon")).exists()


def test_eviction_removes_oldest_entries_first_until_the_cap_holds(
    tmp_path: Path,
) -> None:
    """Storing past the cap evicts by age, oldest first, and stops at the cap.

    The corpus's sizes are ordered independently of its ages precisely so that
    this single survivor-set assertion rejects largest-first, smallest-first
    and newest-first eviction as well as an unordered one.
    """
    root = tmp_path / "research-cache"
    _write_corpus(root)
    cache = ResearchCache(root=root, max_bytes=_CAP_ADMITTING_TWO_NEWEST)
    # A zero-byte store: it adds an entry without adding bytes, so the
    # arithmetic under test is exactly the corpus's own.
    cache.store(_entry_name("trigger"), "")

    assert _surviving_seeds(root) == {"gamma", "delta"}
    assert _entry_bytes(root) == _SURVIVING_BYTES


def test_eviction_holds_the_byte_cap_not_an_entry_count(tmp_path: Path) -> None:
    """The cap bounds total bytes: one large entry evicts more than one small one.

    An implementation that bounded the *entry count* (or evicted a fixed
    number of entries per sweep) would leave a different survivor set here,
    because holding 350 bytes requires evicting the three oldest entries --
    809 bytes -- not a fixed count.
    """
    root = tmp_path / "research-cache"
    _write_corpus(root)
    cache = ResearchCache(root=root, max_bytes=350)

    cache.store(_entry_name("trigger"), "")

    assert _surviving_seeds(root) == {"delta"}
    assert _entry_bytes(root) == _SIZES["delta"]


def test_a_total_exactly_at_the_cap_is_not_evicted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The cap is inclusive: a total equal to it is within bounds, and silent.

    Pins the threshold comparison against an off-by-one in either of the two
    places it appears: an implementation evicting on ``total >= cap`` would
    remove the oldest entry here, and one *reporting* on ``total >= cap``
    would cry "cap unhonourable" at a cache that is exactly within its cap --
    a false alarm on every beat of a steady-state run. Nothing at all is
    logged when nothing happens, so a report that fires unconditionally is
    caught too.
    """
    root = tmp_path / "research-cache"
    _write_corpus(root)
    cache = ResearchCache(root=root, max_bytes=_CORPUS_BYTES)
    caplog.set_level(logging.INFO, logger="windbreak.forecast.sandbox")

    cache.store(_entry_name("trigger"), "")

    assert _surviving_seeds(root) == set(_SIZES)
    assert [record.getMessage() for record in caplog.records] == []


def test_a_total_one_byte_over_the_cap_evicts_exactly_the_oldest(
    tmp_path: Path,
) -> None:
    """One byte over the cap removes the single oldest entry and stops.

    The other side of the same off-by-one: an implementation evicting on
    ``total > cap`` but sweeping to strictly *below* the cap, or one that
    evicted every entry once triggered, would not leave exactly three.
    """
    root = tmp_path / "research-cache"
    _write_corpus(root)
    cache = ResearchCache(root=root, max_bytes=_CORPUS_BYTES - 1)

    cache.store(_entry_name("trigger"), "")

    assert _surviving_seeds(root) == {"beta", "gamma", "delta"}
    assert _entry_bytes(root) == _CORPUS_BYTES - _SIZES["alpha"]


def test_the_cache_is_still_usable_after_an_eviction(tmp_path: Path) -> None:
    """A cache that has evicted still stores, and still evicts, afterwards.

    Issue #453 acceptance criterion 4(c): eviction is a steady state, not a
    one-shot that leaves the cache broken. The second store pushes 518 + 90
    bytes past the same 600-byte cap, so the sweep must run a second time and
    take the next-oldest survivor.
    """
    root = tmp_path / "research-cache"
    _write_corpus(root)
    cache = ResearchCache(root=root, max_bytes=_CAP_ADMITTING_TWO_NEWEST)
    cache.store(_entry_name("trigger"), "")

    later_path = cache.store(_entry_name("later"), "z" * 90)

    assert later_path.read_text(encoding="utf-8") == "z" * 90
    assert _surviving_seeds(root) == {"delta"}


# --- The entry the current fetch depends on ------------------------------------


def test_eviction_never_removes_the_entry_just_written(tmp_path: Path) -> None:
    """The just-stored entry survives even when it alone exceeds the cap.

    Issue #453 acceptance criterion 3: eviction never removes an entry a fetch
    in the current forecast still depends on. The just-written entry is the
    only such entry (nothing ever reads an older one back), and it is the one
    `store` returns to its caller.
    """
    root = tmp_path / "research-cache"
    _write_corpus(root)
    oversized = "w" * 900
    cache = ResearchCache(root=root, max_bytes=500)

    stored_path = cache.store(_entry_name("oversized"), oversized)

    assert stored_path.read_text(encoding="utf-8") == oversized
    assert _surviving_seeds(root) == set()


def test_an_unhonourable_cap_is_announced_rather_than_silently_broken(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cap smaller than one entry logs exactly one warning naming the overshoot.

    Captured from ``INFO`` up, so the assertion also pins that *no* eviction
    was reported: nothing was removable, and a sweep that announced an empty
    eviction would be claiming work it did not do.
    """
    root = tmp_path / "research-cache"
    root.mkdir()
    cache = ResearchCache(root=root, max_bytes=500)
    caplog.set_level(logging.INFO, logger="windbreak.forecast.sandbox")

    cache.store(_entry_name("oversized"), "w" * 900)

    assert [record.getMessage() for record in caplog.records] == [
        "research cache holds 900 bytes against its 500-byte cap after "
        "evicting every removable entry; raise forecast.research.cache_max_bytes "
        "or shrink forecast.research.fetch_max_bytes"
    ]


def test_a_routine_eviction_is_reported_with_its_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An eviction logs exactly one record naming the entries and bytes reclaimed.

    Eviction discards an archived research payload. Nothing reads the cache
    back, so no forecast changes -- but the loss is still announced, and the
    record carries no path, URL or digest, only counts.
    """
    root = tmp_path / "research-cache"
    _write_corpus(root)
    cache = ResearchCache(root=root, max_bytes=_CAP_ADMITTING_TWO_NEWEST)
    caplog.set_level(logging.INFO, logger="windbreak.forecast.sandbox")

    cache.store(_entry_name("trigger"), "")

    assert [record.getMessage() for record in caplog.records] == [
        "research cache evicted 2 entries (502 bytes) to hold its 600-byte cap"
    ]


# --- Nothing that is not provably a cache entry is ever deleted -----------------


def test_eviction_refuses_to_delete_through_a_symlink_escaping_the_root(
    tmp_path: Path,
) -> None:
    """An entry-shaped symlink resolving outside the jail is refused, not followed.

    Issue #453 acceptance criterion 3: the `_is_within_root` jail must hold on
    every delete path. A symlink named exactly like a cache entry, pointing at
    a file outside the root, is the traversal attempt: unlinking it would be
    harmless, but *counting and unlinking it as an entry* is the road to
    deleting the thing it points at, so it is excluded from the sweep
    entirely. The genuine entries around it are still evicted, which is the
    positive control proving the sweep ran at all.

    The link's target is backdated to be **older than every genuine entry**, so
    an implementation missing the jail re-check would reach it *first* rather
    than never. A guard the eviction order can never arrive at is an inert
    guard, and this test would not notice.
    """
    root = tmp_path / "research-cache"
    _write_corpus(root)
    outside = tmp_path / "precious.txt"
    outside.write_text("do not delete me", encoding="utf-8")
    oldest_ns = _BASE_MTIME_NS - _AGE_STEP_NS
    os.utime(outside, ns=(oldest_ns, oldest_ns))
    escape_link = root.joinpath(_entry_name("escape"))
    escape_link.symlink_to(outside)
    cache = ResearchCache(root=root, max_bytes=350)

    cache.store(_entry_name("trigger"), "")

    assert outside.read_text(encoding="utf-8") == "do not delete me"
    assert escape_link.is_symlink()
    assert _surviving_seeds(root) == {"delta"}


def test_eviction_ignores_files_that_are_not_cache_entries(tmp_path: Path) -> None:
    """Foreign files and directories are neither counted nor deleted.

    An eviction routine pointed at a misconfigured directory is a data-loss
    hazard, so the sweep operates only on names it can prove it wrote: a
    sha256-digest stem with the cache suffix, directly under the root. The
    foreign bytes here (2000 of them, nearly twice the corpus) are not counted
    against the cap either -- an implementation that counted what it cannot
    evict would evict the whole corpus to chase a total it can never reach.

    Two of the foreign files are an operator's backups of real entries, whose
    names *contain* a digest without *being* one. They are the reason the
    entry pattern is anchored at both ends rather than merely searched for,
    and they are backdated below the whole corpus so an unanchored
    implementation would take them first.
    """
    root = tmp_path / "research-cache"
    _write_corpus(root)
    foreign_file = root / "operator-notes.md"
    foreign_file.write_text("k" * 2000, encoding="utf-8")
    prefixed_backup = root / f"backup-{_entry_name('alpha')}"
    prefixed_backup.write_text("b" * 137, encoding="utf-8")
    suffixed_backup = root.joinpath(f"{_entry_name('beta')}.bak")
    suffixed_backup.write_text("s" * 173, encoding="utf-8")
    oldest_ns = _BASE_MTIME_NS - _AGE_STEP_NS
    for backup in (prefixed_backup, suffixed_backup):
        os.utime(backup, ns=(oldest_ns, oldest_ns))
    entry_named_directory = root.joinpath(_entry_name("directory"))
    entry_named_directory.mkdir()
    nested = root / "subdir"
    nested.mkdir()
    nested_entry = nested.joinpath(_entry_name("nested"))
    nested_entry.write_text("n" * 700, encoding="utf-8")
    cache = ResearchCache(root=root, max_bytes=_CORPUS_BYTES)

    cache.store(_entry_name("trigger"), "")

    assert _surviving_seeds(root) == set(_SIZES)
    assert foreign_file.read_text(encoding="utf-8") == "k" * 2000
    assert prefixed_backup.read_text(encoding="utf-8") == "b" * 137
    assert suffixed_backup.read_text(encoding="utf-8") == "s" * 173
    assert entry_named_directory.is_dir()
    assert nested_entry.read_text(encoding="utf-8") == "n" * 700


def test_store_still_refuses_a_traversing_name(tmp_path: Path) -> None:
    """The pre-existing write jail is untouched by the eviction work."""
    cache = ResearchCache(root=tmp_path, max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES)

    with pytest.raises(SandboxPathViolationError):
        cache.store("../escape.txt", "content")


# --- Fail closed on the capability, never on the process ------------------------


def test_an_unreadable_cache_directory_degrades_the_cache_not_the_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A sweep that cannot read the cache root warns and returns, and the store stands.

    A loop that cannot start cannot honour a kill file, so an unreadable or
    unwritable cache directory must not propagate out of the cache.
    """
    root = tmp_path / "research-cache"
    root.mkdir()
    real_iterdir = Path.iterdir

    def refusing_iterdir(self: Path) -> object:
        """Raise for the cache root, delegating for every other path.

        Args:
            self: The path being listed.

        Returns:
            The real iterator, for any path other than the cache root.

        Raises:
            PermissionError: When the cache root itself is listed.
        """
        if self == root:
            raise PermissionError(13, "Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", refusing_iterdir)
    cache = ResearchCache(root=root, max_bytes=10)
    caplog.set_level(logging.WARNING, logger="windbreak.forecast.sandbox")

    stored_path = cache.store(_entry_name("payload"), "p" * 40)

    assert stored_path.read_text(encoding="utf-8") == "p" * 40
    assert [record.getMessage() for record in caplog.records] == [
        "research cache eviction failed (PermissionError); the cache stays "
        "unbounded until its directory is readable and writable"
    ]


def test_a_failed_cache_write_does_not_fail_the_fetch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A fetch whose archival write raises still returns its content, loudly.

    Before this change, a full volume made every `ResearchCache.store` raise
    `OSError`, which `verify_citation` and `bounded_web_research` both catch
    and convert into "unreachable source" -- so a full disk silently turned
    every forecast into an abstention, indistinguishable from a dead link.
    The fetch is now insulated from its own archival, and the degradation is
    announced.
    """
    tools = build_research_tools(
        allowed_hosts=frozenset({"research.local"}),
        # A path whose parent is a *file* cannot be created, so `store`'s
        # mkdir raises `NotADirectoryError` -- an `OSError`, exactly as a full
        # or read-only volume would.
        cache_dir=_file_blocked_cache_dir(tmp_path),
        search_transport=_StaticSearch(),
        fetch_transport=_StaticFetch("fetched-content"),
        max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    )
    caplog.set_level(logging.WARNING, logger="windbreak.forecast.sandbox")

    content = tools.fetch("https://research.local/page")

    assert content == "fetched-content"
    assert [record.getMessage() for record in caplog.records] == [
        "research cache write failed (NotADirectoryError); the fetch succeeded "
        "and its payload was not archived"
    ]


def test_a_jail_violation_still_escapes_the_fetch(tmp_path: Path) -> None:
    """A sandbox path violation is not swallowed by the write-failure guard.

    `SandboxPathViolationError` is deliberately not an `OSError`, so insulating
    the fetch from a full disk must not also insulate it from an escape
    attempt. Proven by pointing the cache at a root a store can never land
    inside.
    """
    cache = _EscapingCache(root=tmp_path, max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES)
    tools = _tools_over_cache(cache)

    with pytest.raises(SandboxPathViolationError):
        tools.fetch("https://research.local/page")


# --- The bound is configured, not compiled in ----------------------------------


def test_the_configured_cap_reaches_the_live_cache(tmp_path: Path) -> None:
    """`forecast.research.cache_max_bytes` bounds a live-composed research fetch.

    The composition test, not the pure-builder test beside it: deleting the
    one line that threads the configured cap from `ResearchSettings` into
    `build_research_tools` leaves every unit test above green, because they
    pass `max_bytes` themselves. This one goes through
    `build_live_research_tools` -- the function the scheduler actually calls --
    so it is the assertion that dies.
    """
    cache_dir = tmp_path / "research-cache"
    config = _config_with_cache_cap(120)
    tools = build_live_research_tools(config, _live_provider_http("a" * 100), cache_dir)

    tools.fetch("https://research.local/first")
    tools.fetch("https://research.local/second")

    entries = sorted(path.name for path in cache_dir.iterdir())
    assert entries == [_entry_name("https://research.local/second")]


def test_the_shipped_cap_default_is_a_positive_integer() -> None:
    """The shipped default is an integer byte count, never a float (SPEC §6.1)."""
    assert isinstance(DEFAULT_RESEARCH_CACHE_MAX_BYTES, int)
    assert DEFAULT_RESEARCH_CACHE_MAX_BYTES > 0
    assert WindbreakConfig().forecast.research.cache_max_bytes == (
        DEFAULT_RESEARCH_CACHE_MAX_BYTES
    )


def test_the_runbook_documents_the_shipped_cap_default_exactly() -> None:
    """docs/RUNBOOK.md's stated default is the code's default, in both units.

    An operator sizes the `ledger` volume from the runbook, not from the
    source. A default that drifted from its documentation would have them
    provision against a number the loop does not use, so the byte figure and
    its MiB gloss are both derived from the file rather than restated here.
    """
    runbook = _REPO_ROOT.joinpath("docs", "RUNBOOK.md").read_text(encoding="utf-8")

    matches = re.findall(r"defaulting to `(\d+)`\s+\((\d+) MiB\)", runbook)

    assert len(matches) == 1
    documented_bytes, documented_mib = matches[0]
    assert int(documented_bytes) == DEFAULT_RESEARCH_CACHE_MAX_BYTES
    assert int(documented_mib) * _BYTES_PER_MIB == DEFAULT_RESEARCH_CACHE_MAX_BYTES


@pytest.mark.parametrize("bad_cap", [0, -1])
def test_a_non_positive_cap_is_refused_at_construction(bad_cap: int) -> None:
    """A zero or negative cap raises `ValueError` naming the offending value.

    A cap of zero would evict every entry the moment it was written -- a cache
    that cannot cache. Like `is_live_mode`'s unknown-mode refusal, an operator
    error in configuration is reported, never quietly reinterpreted.
    """
    with pytest.raises(ValueError) as excinfo:
        ResearchCache(root=Path("/unused"), max_bytes=bad_cap)

    assert str(excinfo.value) == (
        f"forecast.research.cache_max_bytes must be a positive byte count, "
        f"got {bad_cap}"
    )


# --- Helpers -------------------------------------------------------------------


def _file_blocked_cache_dir(tmp_path: Path) -> Path:
    """Return a cache root whose parent is a regular file, so `mkdir` must fail.

    Args:
        tmp_path: The pytest scratch directory.

    Returns:
        A path under a regular file, unusable as a directory.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    return blocker / "research-cache"


class _StaticSearch:
    """A search transport returning one fixed URL."""

    def search(self, query: str) -> tuple[str, ...]:
        """Return the single canned URL.

        Args:
            query: The (ignored) subquestion text.

        Returns:
            A one-element tuple of the canned URL.
        """
        del query
        return ("https://research.local/page",)


class _StaticFetch:
    """A fetch transport returning fixed content."""

    def __init__(self, content: str) -> None:
        """Store the content every fetch returns.

        Args:
            content: The canned page content.
        """
        self._content = content

    def fetch(self, url: str) -> str:
        """Return the canned content.

        Args:
            url: The (ignored) URL.

        Returns:
            The canned content.
        """
        del url
        return self._content


class _EscapingCache(ResearchCache):
    """A cache whose every store attempt escapes the jail."""

    def store(self, name: str, content: str) -> Path:
        """Raise as the real jail would for an escaping name.

        Args:
            name: The (ignored) entry name.
            content: The (ignored) payload.

        Raises:
            SandboxPathViolationError: Always.
        """
        del name, content
        raise SandboxPathViolationError("cache path escapes the sandbox root")


def _tools_over_cache(cache: ResearchCache) -> ResearchTools:
    """Return research tools whose cache is ``cache``.

    Args:
        cache: The cache the tools persist fetched payloads through.

    Returns:
        A `ResearchTools` over static transports and ``cache``.
    """
    return ResearchTools(
        allowed_hosts=frozenset({"research.local"}),
        search_transport=_StaticSearch(),
        fetch_transport=_StaticFetch("fetched-content"),
        cache=cache,
    )


class _CannedHttp:
    """An HTTP transport answering every request with one canned body."""

    def __init__(self, body: str) -> None:
        """Store the canned response body.

        Args:
            body: The body every response carries.
        """
        self._body = body

    def send(self, request: HttpRequest) -> HttpResponse:
        """Return the canned 200 response.

        Args:
            request: The (ignored) request.

        Returns:
            A 200 `HttpResponse` carrying the canned body as ``text/html``.
        """
        del request
        return HttpResponse(status_code=200, body=self._body, content_type="text/html")


def _live_provider_http(body: str) -> LiveProviderHttp:
    """Return live seams whose search and fetch both answer from memory.

    Args:
        body: The body the fetch seam returns.

    Returns:
        A `LiveProviderHttp` with no LLM routes and canned research seams.
    """
    return LiveProviderHttp(llm={}, search=_CannedHttp(body), fetch=_CannedHttp(body))


def _config_with_cache_cap(cap: int) -> WindbreakConfig:
    """Return the default config with the research cache cap replaced.

    Args:
        cap: The `forecast.research.cache_max_bytes` value to install.

    Returns:
        A `WindbreakConfig` whose research section carries ``cap`` and reaches
        ``research.local``.
    """
    base = WindbreakConfig()
    research = dataclasses.replace(
        base.forecast.research,
        allowed_research_hosts=("research.local",),
        search_endpoint_url="https://research.local/search",
        cache_max_bytes=cap,
    )
    forecast = dataclasses.replace(base.forecast, research=research)
    return dataclasses.replace(base, forecast=forecast)
