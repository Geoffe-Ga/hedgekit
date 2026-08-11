"""SPEC S8.3 bounded research sandbox: the forecast engine's egress firewall.

SPEC S8.3 requires the forecast engine's web-research stage to be able to reach
a small, explicit allowlist of hosts and *nothing else* -- no ledger, no order
gateway, no config, no arbitrary network. This module makes that boundary
**structural rather than prompt-based**: instead of instructing a model "please
only fetch from these hosts" (a request a prompt-injected tool call can ignore),
the only capabilities a research step is ever handed are the two methods on a
slots-closed :class:`ResearchTools` object, and :meth:`ResearchTools.fetch`
allowlists by exact, parsed hostname before it will call its transport. A caller
(or a hijacked tool-use turn) cannot smuggle in a third capability, cannot
widen the allowlist, and cannot escape the on-disk cache jail, because those
constraints live in code the model never gets to rewrite.

Three seams enforce the boundary:

* **Egress allowlist** -- :meth:`ResearchTools.fetch` parses the URL with
  :func:`urllib.parse.urlsplit`, takes the real ``hostname`` (which already
  excludes any ``user@`` userinfo prefix), lowercases it, and refuses any
  non-http(s) scheme, missing host, or off-allowlist host with
  :class:`EgressDeniedError`. This mirrors the scheme/redirect-host refusals in
  :mod:`windbreak.connector.kalshi.client`.
* **Path jail** -- :class:`ResearchCache` resolves every candidate write path
  and refuses (:class:`SandboxPathViolationError`) anything that lands outside its
  root, defeating absolute names, ``..`` traversal, and symlink escapes.
* **Final capability surface** -- :func:`tool_registry` exposes exactly
  ``{"search", "fetch"}`` (the two egress primitives) as a read-only mapping.
  Citation verification is not a sandbox capability: it is performed
  pipeline-side by :func:`windbreak.forecast.citations.verify_citation` as a
  composition over the egress-guarded ``fetch`` primitive, keeping SPEC S8.3
  citation-verification traceability while the sandbox surface stays minimal.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

_LOGGER = logging.getLogger(__name__)

#: The only URL schemes egress is ever permitted for (SPEC S8.3): plain
#: http(s), never ``file://``, ``ftp://``, or any other privileged scheme.
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

#: Ordinal of the first printable ASCII character (space, ``0x20``). Anything
#: below it is a C0 control character; a well-formed URL percent-encodes such
#: bytes, so their raw presence is a parser-differential red flag.
_MIN_PRINTABLE_ORD = 0x20

#: Ordinal of the ASCII DEL control character (``0x7f``), the one control
#: codepoint that sorts *above* the printable range and so is checked by hand.
_DEL_ORD = 0x7F

#: Suffix stamped on every cached fetch payload; the stem is a sha256 digest of
#: the source URL, so distinct URLs cache to distinct, collision-free files.
_CACHE_FILE_SUFFIX = ".txt"

#: Length of a sha256 hex digest, derived from the hash rather than restated,
#: so the eviction sweep's notion of "a file this cache wrote" cannot drift
#: from :func:`_cache_filename`'s notion of what it writes.
_DIGEST_HEX_LENGTH = hashlib.sha256().digest_size * 2

#: The exact shape of a name this cache is willing to delete: a full sha256 hex
#: digest plus the cache suffix, and nothing else. An eviction routine pointed
#: at a misconfigured directory is a data-loss hazard, so the sweep never
#: touches a name it cannot prove it wrote itself.
_CACHE_ENTRY_PATTERN = re.compile(
    f"^[0-9a-f]{{{_DIGEST_HEX_LENGTH}}}{re.escape(_CACHE_FILE_SUFFIX)}$"
)


class EgressDeniedError(Exception):
    """Raised when a fetch targets a scheme or host outside the allowlist."""


class SandboxPathViolationError(Exception):
    """Raised when a cache write would escape the sandbox's on-disk root."""


class SearchTransport(Protocol):
    """The seam through which candidate URLs are obtained for a query."""

    def search(self, query: str) -> tuple[str, ...]:
        """Return candidate URLs for ``query``.

        Args:
            query: The subquestion text to search for.

        Returns:
            The candidate URLs, most relevant first.
        """
        ...


class FetchTransport(Protocol):
    """The seam through which a single URL's content is retrieved."""

    def fetch(self, url: str) -> str:
        """Return the textual content at ``url``.

        Args:
            url: The URL to retrieve.

        Returns:
            The retrieved content.
        """
        ...


def _has_unsafe_url_chars(url: str) -> bool:
    """Return whether ``url`` holds any control or whitespace character.

    :func:`urllib.parse.urlsplit` silently *strips* tab/newline/carriage-return
    (and tolerates leading control/space bytes, CVE-2023-24329) while computing
    the ``hostname`` this sandbox's allowlist gate trusts -- but the raw ``url``
    string is later handed verbatim to the fetch transport. A byte the gate's
    parser drops and the transport's parser keeps is a classic
    parse-differential SSRF escape, so any such byte must fail closed *before*
    parsing rather than let the two parsers disagree on the real host.

    Args:
        url: The candidate URL to screen.

    Returns:
        ``True`` if any character is ASCII whitespace, a C0 control byte, or
        DEL -- none of which belong, unencoded, in a well-formed URL.
    """
    return any(
        char.isspace() or ord(char) < _MIN_PRINTABLE_ORD or ord(char) == _DEL_ORD
        for char in url
    )


def _cache_filename(url: str) -> str:
    """Derive a collision-free cache filename from a source URL.

    Args:
        url: The URL whose fetched content is being cached.

    Returns:
        A ``<sha256-hex>.txt`` filename, a deterministic function of ``url``.
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"{digest}{_CACHE_FILE_SUFFIX}"


class _CacheEntry(NamedTuple):
    """One evictable cache entry, with the two keys the sweep needs.

    Attributes:
        mtime_ns: Last-modification time in integer nanoseconds since the
            epoch -- the eviction ordering key. Nanoseconds since the epoch
            are an absolute instant, so this ordering is identical in every
            process timezone; and being an ``int``, it keeps this module clear
            of the no-float money-path lint that ``datetime.timestamp()``
            would trip.
        name: The entry's filename, the deterministic tiebreak when two
            entries share a modification time.
        size: The entry's size in bytes.
        path: The entry's full path, already proven inside the jail.
    """

    mtime_ns: int
    name: str
    size: int
    path: Path


class ResearchCache:
    """A byte-bounded, write-jailed cache for fetched research payloads.

    Every write is confined to ``root``: names are resolved (following any
    symlinks) and rejected with :class:`SandboxPathViolationError` unless the
    resolved candidate is inside the resolved root, so absolute names, ``..``
    traversal, and symlink escapes all fail closed.

    Every write is also *bounded* (issue #453). The cache's own entries may
    hold at most ``max_bytes`` in total; each store sweeps oldest-first until
    that holds again. Before the bound existed the cache was a write-only
    store with no cap, no sweep and no ceiling of any kind, sharing the
    shipped compose stack's ``ledger`` volume with the hash-chained ledger --
    so it grew until the volume filled and the next ledger append took the
    daemon down (#443).

    Eviction cannot change a forecast. Nothing ever reads an entry back:
    :meth:`ResearchTools.fetch` calls its transport on every call and stores
    the result afterwards, so the cache is an archive of what was fetched, not
    a hit/miss cache in front of the network. An evicted entry therefore costs
    no re-fetch, no research budget, and no abstention -- only the archived
    copy of a payload -- and that loss is logged rather than silent.
    """

    __slots__ = ("_max_bytes", "_root")

    def __init__(self, root: Path, *, max_bytes: int) -> None:
        """Initialize the cache over its jail root and byte ceiling.

        Args:
            root: The directory every stored file must live under.
            max_bytes: The ceiling on the total bytes the cache's own entries
                may hold. Keyword-only and *required*: a defaulted bound would
                let a wiring change silently drop the operator's configured
                value while every test stayed green.

        Raises:
            ValueError: If ``max_bytes`` is not a positive byte count. A cap of
                zero is a cache that evicts everything it writes; a negative
                cap is meaningless. Either is an operator error in
                configuration, reported the way
                :func:`windbreak.scheduler.provider_wiring.is_live_mode`
                reports an unrecognized transport mode rather than quietly
                reinterpreted.
        """
        if max_bytes < 1:
            raise ValueError(
                "forecast.research.cache_max_bytes must be a positive byte "
                f"count, got {max_bytes}"
            )
        self._root = root
        self._max_bytes = max_bytes

    def _is_within_root(self, candidate: Path) -> bool:
        """Return whether ``candidate`` resolves to a path *strictly* inside the root.

        Both the candidate and the root are fully resolved (following
        symlinks) before the containment test, and containment is checked with
        :meth:`~pathlib.PurePath.is_relative_to` on those resolved paths -- not
        a string prefix -- so sibling roots like ``/cache`` and ``/cache2`` can
        never be confused. The resolved candidate must also differ from the
        resolved root itself: an empty or ``.`` name resolves *onto* the root
        directory, and letting that through would clobber the jail dir (writing
        a file where the cache root should be), so it is refused as out of jail.

        Args:
            candidate: The prospective write path, relative to or under root.

        Returns:
            ``True`` if the resolved candidate lies strictly within the resolved
            root (never equal to it).
        """
        resolved_root = self._root.resolve()
        resolved_candidate = candidate.resolve()
        return (
            resolved_candidate != resolved_root
            and resolved_candidate.is_relative_to(resolved_root)
        )

    def store(self, name: str, content: str) -> Path:
        """Write ``content`` to ``name`` under the root, or fail closed.

        Args:
            name: The relative path (under the root) to write to.
            content: The text to persist.

        Returns:
            The path the content was written to.

        Raises:
            SandboxPathViolationError: If ``name`` is absolute, traverses out of the
                root, resolves (via a symlink) outside it, or resolves onto the
                root directory itself. The message names the offending path.
        """
        # ``Path.joinpath`` (not the ``/`` operator) keeps this module clear of
        # the no-float lint (``scripts/lint_no_floats.py``), which reads a bare
        # ``/`` as banned true division on the probability/money path.
        candidate = self._root.joinpath(name)
        if Path(name).is_absolute() or not self._is_within_root(candidate):
            raise SandboxPathViolationError(
                f"cache path {name!r} escapes the sandbox root {self._root}"
            )
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(content, encoding="utf-8")
        self._evict_to_cap(keep=candidate)
        return candidate

    def _evictable_entries(self) -> list[_CacheEntry]:
        """Return every entry this cache may delete, unordered.

        An entry qualifies only if all four of these hold, which together mean
        "a file this cache itself wrote, still inside its jail":

        1. it sits *directly* under the root (a nested tree is not this
           cache's, and is left whole);
        2. its name is exactly a sha256 hex digest plus the cache suffix, the
           shape :func:`_cache_filename` produces;
        3. it is a regular file (a *directory* wearing an entry's name is not
           an entry, and unlinking it would raise); and
        4. it still resolves strictly inside the resolved root -- the same
           :meth:`_is_within_root` jail the write path uses. This is what
           refuses an entry-shaped **symlink** pointing outside the cache: such
           a link is neither counted nor unlinked, because treating it as an
           entry is the first step toward destroying whatever it aims at.

        Everything else -- foreign files, subdirectories, escaping links -- is
        invisible to the sweep: not deleted, and not counted against the cap
        either. Counting bytes that cannot be reclaimed would make the sweep
        evict the whole cache chasing a total it can never reach.

        Returns:
            The evictable entries.
        """
        return [
            _CacheEntry(
                mtime_ns=child.stat().st_mtime_ns,
                name=child.name,
                size=child.stat().st_size,
                path=child,
            )
            for child in self._root.iterdir()
            if _CACHE_ENTRY_PATTERN.match(child.name)
            and child.is_file()
            and self._is_within_root(child)
        ]

    def _evict_to_cap(self, *, keep: Path) -> None:
        """Delete entries oldest-first until the cache holds its byte cap.

        Fails closed on the *capability*, never on the process: any
        :class:`OSError` from listing, stating or unlinking (an unreadable,
        unwritable or full cache directory) is caught, announced, and
        swallowed. A loop that cannot start cannot honour a kill file, so an
        undeletable cache degrades the archive loudly rather than stopping the
        beat.

        Args:
            keep: The entry the current fetch just wrote. It is never the
                victim -- it is the one entry a caller in the current forecast
                still holds a path to -- so a payload larger than the whole cap
                leaves the cap unhonourable rather than deleting itself.
        """
        try:
            entries = self._evictable_entries()
            total = sum(entry.size for entry in entries)
            evicted = 0
            reclaimed = 0
            for entry in sorted(entries):
                if total <= self._max_bytes:
                    break
                if entry.path == keep:
                    continue
                entry.path.unlink()
                total -= entry.size
                evicted += 1
                reclaimed += entry.size
        except OSError as error:
            _LOGGER.warning(
                "research cache eviction failed (%s); the cache stays "
                "unbounded until its directory is readable and writable",
                type(error).__name__,
            )
            return
        self._report_sweep(evicted=evicted, reclaimed=reclaimed, total=total)

    def _report_sweep(self, *, evicted: int, reclaimed: int, total: int) -> None:
        """Announce what a completed sweep discarded, and any unheld cap.

        Neither record carries a path, a URL or a digest: a cache path or a
        fetched URL can carry credentials, and the log stream is not a place to
        find out. Counts and byte totals are enough to size the volume.

        Args:
            evicted: How many entries the sweep deleted.
            reclaimed: How many bytes those entries held.
            total: The cache's total bytes once the sweep finished.
        """
        if evicted:
            _LOGGER.info(
                "research cache evicted %d entries (%d bytes) to hold its %d-byte cap",
                evicted,
                reclaimed,
                self._max_bytes,
            )
        if total > self._max_bytes:
            _LOGGER.warning(
                "research cache holds %d bytes against its %d-byte cap after "
                "evicting every removable entry; raise "
                "forecast.research.cache_max_bytes or shrink "
                "forecast.research.fetch_max_bytes",
                total,
                self._max_bytes,
            )


class ResearchTools:
    """The final, capability-closed set of tools a research step may use.

    The only public methods are :meth:`search` and :meth:`fetch`; a flat
    ``__slots__`` blocks any third attribute from being attached. The allowlist
    and both transports are private, so a
    caller cannot widen the egress policy or swap a transport after
    construction -- the boundary is structural, not advisory.
    """

    __slots__ = ("_allowed_hosts", "_cache", "_fetch_transport", "_search_transport")

    def __init__(
        self,
        *,
        allowed_hosts: frozenset[str],
        search_transport: SearchTransport,
        fetch_transport: FetchTransport,
        cache: ResearchCache,
    ) -> None:
        """Initialize the capability object from its private collaborators.

        Args:
            allowed_hosts: The lowercased egress allowlist.
            search_transport: The seam returning candidate URLs for a query.
            fetch_transport: The seam returning content for an allowed URL.
            cache: The write-jailed cache fetched content is persisted to.
        """
        self._allowed_hosts = allowed_hosts
        self._search_transport = search_transport
        self._fetch_transport = fetch_transport
        self._cache = cache

    def search(self, query: str) -> tuple[str, ...]:
        """Return candidate URLs for ``query`` via the search transport.

        Args:
            query: The subquestion text to search for.

        Returns:
            The candidate URLs the search transport returned.
        """
        return self._search_transport.search(query)

    def fetch(self, url: str) -> str:
        """Fetch ``url`` if allowlisted, persist the content, and return it.

        The URL is first screened for control/whitespace bytes (which would let
        this gate's parser and the transport's parser disagree on the real
        host), then parsed with :func:`urllib.parse.urlsplit`; its ``hostname``
        (already stripped of any ``user@`` userinfo) is lowercased and checked
        against the allowlist. Only then is the fetch transport called and the
        content cached. The match is host-only: a port is intentionally
        unconstrained (SPEC S8.3 allowlists by host), so any port on an
        allowlisted host is permitted.

        Args:
            url: The URL to fetch.

        Returns:
            The fetched content, verbatim.

        Raises:
            EgressDeniedError: If the URL contains a control or whitespace
                character, the scheme is not http(s), the host is missing, or
                the host is not on the allowlist. The message names the host.
        """
        if _has_unsafe_url_chars(url):
            raise EgressDeniedError(
                f"egress denied: control or whitespace character in URL {url!r}"
            )
        parts = urlsplit(url)
        if parts.scheme.lower() not in _ALLOWED_SCHEMES:
            raise EgressDeniedError(
                f"egress denied: scheme {parts.scheme!r} in {url!r}"
            )
        hostname = parts.hostname
        if not hostname:
            raise EgressDeniedError(f"egress denied: no host in {url!r}")
        host = hostname.lower()
        if host not in self._allowed_hosts:
            raise EgressDeniedError(f"egress denied: host {host!r} is not allowlisted")
        content = self._fetch_transport.fetch(url)
        self._archive(url, content)
        return content

    def _archive(self, url: str, content: str) -> None:
        """Persist ``content`` to the cache, tolerating an unwritable cache.

        Archiving is a *capability*, not the fetch. A full or read-only volume
        used to make :meth:`ResearchCache.store` raise :class:`OSError` out of
        :meth:`fetch` -- and both callers of ``fetch``
        (:func:`windbreak.forecast.citations.verify_citation` and
        :func:`windbreak.forecast.pipeline.bounded_web_research`) catch
        ``OSError`` and treat it as an unreachable source. A full disk
        therefore turned every forecast into an abstention on zero verified
        citations, indistinguishable from every source being dead. The fetch is
        now insulated from its own archival, and the degradation is announced
        instead of inferred.

        :class:`SandboxPathViolationError` is deliberately *not* an
        ``OSError`` and so is deliberately *not* caught here: a jail escape is
        a security boundary, not a degraded volume, and must still reach the
        caller.

        Args:
            url: The URL whose payload is being archived.
            content: The fetched payload.
        """
        try:
            self._cache.store(_cache_filename(url), content)
        except OSError as error:
            _LOGGER.warning(
                "research cache write failed (%s); the fetch succeeded and "
                "its payload was not archived",
                type(error).__name__,
            )


def tool_registry(tools: ResearchTools) -> Mapping[str, Callable[..., object]]:
    """Return the read-only tool registry backed by ``tools``.

    The mapping surface is exactly ``{"search", "fetch"}`` and is wrapped in
    :class:`types.MappingProxyType`, so a caller can neither add a third
    capability nor rebind an existing one.

    Args:
        tools: The capability object whose bound methods back the registry.

    Returns:
        A read-only mapping of tool name to the bound method implementing it.
    """
    tools_by_name: dict[str, Callable[..., object]] = {
        "search": tools.search,
        "fetch": tools.fetch,
    }
    return MappingProxyType(tools_by_name)


def build_research_tools(
    *,
    allowed_hosts: Iterable[str],
    cache_dir: Path,
    search_transport: SearchTransport,
    fetch_transport: FetchTransport,
    max_bytes: int,
) -> ResearchTools:
    """Assemble a sandboxed :class:`ResearchTools` from its collaborators.

    Args:
        allowed_hosts: The hosts egress is permitted for; normalized to a
            lowercased frozenset so matching is case-insensitive.
        cache_dir: The root the fetch cache is jailed to.
        search_transport: The seam returning candidate URLs for a query.
        fetch_transport: The seam returning content for an allowed URL.
        max_bytes: The ceiling on the total bytes the fetch cache may hold
            (issue #453). Required, not defaulted: the bound belongs to the
            operator's ``forecast.research.cache_max_bytes``, and a module
            default here would let that plumbing be deleted without a single
            test noticing.

    Returns:
        A capability-closed :class:`ResearchTools` over the given seams.

    Raises:
        ValueError: If ``max_bytes`` is not a positive byte count; see
            :class:`ResearchCache`.
    """
    normalized_hosts = frozenset(host.lower() for host in allowed_hosts)
    cache = ResearchCache(root=cache_dir, max_bytes=max_bytes)
    return ResearchTools(
        allowed_hosts=normalized_hosts,
        search_transport=search_transport,
        fetch_transport=fetch_transport,
        cache=cache,
    )
