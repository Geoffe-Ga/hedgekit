"""Tests for windbreak.forecast.sandbox (issue #24): the bounded research sandbox.

`windbreak/forecast/sandbox.py` does not exist yet, so importing it below fails
collection with `ModuleNotFoundError: No module named 'windbreak.forecast.sandbox'`
-- the expected Gate 1 RED state for issue #24. This module pins the structural
tool boundary the sandbox must enforce:

* **Registry / instance surface** -- `tool_registry(tools)` and a bare
  `ResearchTools` instance expose *exactly* `{"search", "fetch"}` and nothing
  else; both are read-only / slots-closed so a caller (or a future
  prompt-injected tool call) cannot smuggle in a third capability.
* **Egress** -- `ResearchTools.fetch` allowlists by exact, lowercased hostname
  only, structurally defeating the classic `user@evil.example` userinfo trick,
  non-http(s) schemes (e.g. `file://`), and missing hosts.
* **Path jail** -- `ResearchCache.store` resolves every candidate path and
  refuses to write outside its root, including through a symlink escape.
* **No privileged handle** -- nothing reachable from a `ResearchTools`
  instance, nor any parameter of `build_research_tools`, ever touches
  `windbreak.ledger`, `windbreak.config`, or `windbreak.connector`.
* **Import boundary** -- a self-contained, stdlib-`ast`-based checker (the "CI
  teeth" for the previous bullet, at the source-file level rather than one
  object graph) scans every `windbreak/forecast/*.py` file for the same
  forbidden prefixes -- with one narrow, source-level allowance the instance
  check above does not need: `windbreak.connector.models`, the read-only
  `NormalizedMarket` data shape `bounded_web_research` (stage 5) is handed by
  the rest of the pipeline.
* **Pipeline integration** -- `run_pipeline`'s new, required `research_tools`
  keyword actually sits on the path `bounded_web_research` walks: a
  sandboxed, fixture-backed run stays schema-valid and byte-deterministic, and
  a `research_tools` wired to return an off-allowlist URL makes the *pipeline
  call itself* raise `EgressDeniedError` -- proving the guard isn't only reachable
  in isolation.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from windbreak.config.schema import DEFAULT_RESEARCH_CACHE_MAX_BYTES
from windbreak.forecast.pipeline import run_pipeline
from windbreak.forecast.records import forecast_record_to_payload
from windbreak.forecast.sandbox import (
    EgressDeniedError,
    ResearchCache,
    ResearchTools,
    SandboxPathViolationError,
    build_research_tools,
    tool_registry,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from windbreak.connector.models import NormalizedMarket
    from windbreak.forecast.records import BaselineQuoteSnapshot

    #: `make_fake_vote_transport` / `research_tools_factory` (see
    #: tests/forecast/conftest.py) are factories for fixture doubles defined in
    #: the conftest module (not part of the `windbreak` package under test), so
    #: they are typed structurally here rather than imported by name.
    FakeVoteTransportFactory = Callable[[], object]
    ResearchToolsFactory = Callable[..., ResearchTools]

#: Repo root, derived from this test file's own location
#: (`<root>/tests/forecast/test_sandbox.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every real, shipped module under the package this test suite guards.
_FORECAST_PACKAGE_DIR = _REPO_ROOT / "windbreak" / "forecast"


class _StaticSearchTransport:
    """A `SearchTransport` double returning one fixed URL for any query."""

    def __init__(self, url: str) -> None:
        """Store the single URL every `search` call will return.

        Args:
            url: The candidate URL to return for any query.
        """
        self._url = url

    def search(self, query: str) -> tuple[str, ...]:
        """Return a one-element tuple holding the fixed URL, ignoring `query`.

        Args:
            query: The (unused) subquestion text.

        Returns:
            A one-element tuple containing `self._url`.
        """
        return (self._url,)


class _StaticFetchTransport:
    """A `FetchTransport` double returning one fixed content string for any URL."""

    def __init__(self, content: str = "fetched-content") -> None:
        """Store the fixed content every `fetch` call will return.

        Args:
            content: The content string to return for any URL.
        """
        self._content = content

    def fetch(self, url: str) -> str:
        """Return the fixed content, ignoring `url`.

        Args:
            url: The (unused) URL being fetched.

        Returns:
            `self._content`, verbatim.
        """
        return self._content


class _EmptySearchTransport:
    """A `SearchTransport` double that finds no candidate URL for any query."""

    def search(self, query: str) -> tuple[str, ...]:
        """Return an empty tuple, ignoring `query`.

        Args:
            query: The (unused) subquestion text.

        Returns:
            An empty tuple -- the "no candidate found" case.
        """
        return ()


def _build_tools(
    tmp_path: Path,
    *,
    allowed_hosts: frozenset[str] = frozenset({"research.local"}),
    search_url: str = "https://research.local/x",
    fetch_content: str = "fetched-content",
) -> ResearchTools:
    """Build a `ResearchTools` directly over the two static doubles above.

    Args:
        tmp_path: The pytest-provided temporary directory to cache under.
        allowed_hosts: The egress allowlist.
        search_url: The single URL `search` returns for any query.
        fetch_content: The content `fetch` returns for any URL.

    Returns:
        A `ResearchTools` wired over `_StaticSearchTransport` /
        `_StaticFetchTransport`.
    """
    return build_research_tools(
        allowed_hosts=allowed_hosts,
        cache_dir=tmp_path,
        search_transport=_StaticSearchTransport(search_url),
        fetch_transport=_StaticFetchTransport(fetch_content),
        max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    )


# --- Registry surface --------------------------------------------------------------


def test_tool_registry_keys_are_exactly_search_fetch(
    research_tools: ResearchTools,
) -> None:
    """`tool_registry` exposes exactly `{search, fetch}`."""
    registry = tool_registry(research_tools)

    assert set(registry.keys()) == {"search", "fetch"}


def test_tool_registry_is_read_only(research_tools: ResearchTools) -> None:
    """Attempting to set a key on the registry mapping raises `TypeError`."""
    registry = tool_registry(research_tools)

    def _replacement_search(*_args: object) -> tuple[str, ...]:
        return ()

    with pytest.raises(TypeError):
        registry["search"] = _replacement_search


# --- Instance surface ----------------------------------------------------------


def test_research_tools_public_surface_is_exactly_two_methods(
    research_tools: ResearchTools,
) -> None:
    """A `ResearchTools` instance's non-underscore attributes are exactly two."""
    public_names = {name for name in dir(research_tools) if not name.startswith("_")}

    assert public_names == {"search", "fetch"}


def test_research_tools_is_slots_based_and_rejects_new_attribute(
    research_tools: ResearchTools,
) -> None:
    """Setting an attribute outside the declared slots raises `AttributeError`."""
    # The attribute name is held in a variable (not a literal) so ``setattr``
    # is the genuine dynamic-assignment under test rather than a constant-name
    # access ruff's B010 would rewrite to plain attribute assignment.
    forbidden_attribute = "new_forbidden_attribute"
    with pytest.raises(AttributeError):
        setattr(research_tools, forbidden_attribute, "nope")


# --- Egress: hostname allowlisting -----------------------------------------------


def test_fetch_denies_non_allowlisted_host(tmp_path: Path) -> None:
    """A host absent from the allowlist raises `EgressDeniedError` naming the host."""
    tools = _build_tools(
        tmp_path,
        allowed_hosts=frozenset({"research.local"}),
        search_url="https://evil.example/path",
    )

    with pytest.raises(EgressDeniedError, match=r"evil\.example"):
        tools.fetch("https://evil.example/path")


def test_fetch_denies_userinfo_trick_hostname(tmp_path: Path) -> None:
    """`https://research.local@evil.example/x` denies on `evil.example`, not the
    userinfo-embedded allowlisted-looking prefix -- proving the check parses
    the real hostname rather than substring-matching the raw URL.
    """
    tools = _build_tools(tmp_path, allowed_hosts=frozenset({"research.local"}))

    with pytest.raises(EgressDeniedError, match=r"evil\.example"):
        tools.fetch("https://research.local@evil.example/x")


def test_fetch_denies_file_scheme_with_no_host(tmp_path: Path) -> None:
    """A `file://` URL (non-http(s) scheme, no real host) raises `EgressDeniedError`."""
    tools = _build_tools(tmp_path)

    with pytest.raises(EgressDeniedError):
        tools.fetch("file:///etc/passwd")


def test_fetch_denies_missing_host(tmp_path: Path) -> None:
    """An http(s) URL with an empty/missing host raises `EgressDeniedError`."""
    tools = _build_tools(tmp_path)

    with pytest.raises(EgressDeniedError):
        tools.fetch("http:///path")


@pytest.mark.parametrize(
    "hostile_url",
    [
        "https://research.local\n/x",  # urlsplit strips \n -> host looks allowed
        "https://research.local\t/x",  # urlsplit strips \t
        "https://research.local\r/x",  # urlsplit strips \r
        " https://research.local/x",  # leading space (CVE-2023-24329 shape)
    ],
)
def test_fetch_denies_control_or_whitespace_url(
    tmp_path: Path, hostile_url: str
) -> None:
    """A URL carrying a raw control/whitespace byte is refused before parsing.

    `urllib.parse.urlsplit` silently strips tab/newline/carriage-return (and
    tolerates a leading space) when it computes the `hostname` the allowlist
    trusts, but the raw URL is what the transport would connect with -- a
    parse-differential SSRF escape. `fetch` must fail closed on any such byte
    even though the *stripped* host (`research.local`) is on the allowlist.
    """
    tools = _build_tools(tmp_path, allowed_hosts=frozenset({"research.local"}))

    with pytest.raises(EgressDeniedError):
        tools.fetch(hostile_url)


def test_fetch_allows_case_insensitive_allowlisted_host(tmp_path: Path) -> None:
    """An uppercased allowlisted host still fetches successfully."""
    tools = _build_tools(
        tmp_path,
        allowed_hosts=frozenset({"research.local"}),
        fetch_content="case-content",
    )

    result = tools.fetch("https://RESEARCH.LOCAL/x")

    assert result == "case-content"


def test_fetch_returns_transport_content_for_allowlisted_host(tmp_path: Path) -> None:
    """A fetch of an allowlisted host returns exactly what the transport returns."""
    tools = _build_tools(tmp_path, fetch_content="hello-world")

    assert tools.fetch("https://research.local/x") == "hello-world"


# --- Path jail: ResearchCache -----------------------------------------------------


def test_cache_store_rejects_parent_traversal(tmp_path: Path) -> None:
    """A `..`-traversing candidate name raises `SandboxPathViolationError`."""
    cache = ResearchCache(root=tmp_path, max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES)

    with pytest.raises(SandboxPathViolationError):
        cache.store("../escape.txt", "content")


def test_cache_store_rejects_absolute_name(tmp_path: Path) -> None:
    """An absolute candidate name raises `SandboxPathViolationError`."""
    cache = ResearchCache(root=tmp_path, max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES)
    absolute_name = str(tmp_path.parent / "escape.txt")

    with pytest.raises(SandboxPathViolationError):
        cache.store(absolute_name, "content")


def test_cache_store_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlink inside the cache root pointing outside it raises on resolve."""
    root = tmp_path / "cache-root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    escape_link = root / "escape-link"
    escape_link.symlink_to(outside, target_is_directory=True)
    cache = ResearchCache(root=root, max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES)

    with pytest.raises(SandboxPathViolationError):
        cache.store("escape-link/payload.txt", "content")


@pytest.mark.parametrize("onto_root_name", ["", "."])
def test_cache_store_rejects_name_resolving_onto_root(
    tmp_path: Path, onto_root_name: str
) -> None:
    """An empty or `.` name resolves onto the jail root itself and is refused.

    Such a name is *inside* the root by the `is_relative_to` test (a path is
    relative to itself), yet writing to it would clobber the cache directory
    (turning the jail root into a file). The store must fail closed with
    `SandboxPathViolationError` rather than an incidental `IsADirectoryError`
    or silently overwriting the root.
    """
    cache = ResearchCache(root=tmp_path, max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES)

    with pytest.raises(SandboxPathViolationError):
        cache.store(onto_root_name, "content")


def test_cache_store_writes_plain_relative_name_under_root(tmp_path: Path) -> None:
    """A plain relative name lands under the cache root and the file is written."""
    cache = ResearchCache(root=tmp_path, max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES)

    result_path = cache.store("note.txt", "hello")

    assert result_path == tmp_path / "note.txt"
    assert result_path.read_text(encoding="utf-8") == "hello"


# --- Fetch persistence ---------------------------------------------------------


def test_allowed_fetch_persists_content_under_cache_dir(tmp_path: Path) -> None:
    """After an allowed fetch, exactly one file exists under `cache_dir` and
    nothing was written outside it.
    """
    cache_dir = tmp_path / "cache"
    tools = build_research_tools(
        allowed_hosts=frozenset({"research.local"}),
        cache_dir=cache_dir,
        search_transport=_StaticSearchTransport("https://research.local/x"),
        fetch_transport=_StaticFetchTransport("persisted-content"),
        max_bytes=DEFAULT_RESEARCH_CACHE_MAX_BYTES,
    )

    tools.fetch("https://research.local/x")

    persisted_files = [path for path in cache_dir.rglob("*") if path.is_file()]
    assert len(persisted_files) == 1
    assert persisted_files[0].is_relative_to(cache_dir)
    assert persisted_files[0].read_text(encoding="utf-8") == "persisted-content"


# --- Retired capability: verify_citation is gone -----------------------------


def test_verify_citation_is_absent_from_surface(
    research_tools: ResearchTools,
) -> None:
    """The reserved `verify_citation` slot has been retired from the sandbox.

    Issue #26 moved citation verification to the pipeline-side
    `verify_citation` function in `windbreak/forecast/citations.py`, a
    composition over `fetch`. The model-facing sandbox never needed its own
    stub, so this follow-up (#93) removes it: `verify_citation` must be
    absent from both the tool registry and the `ResearchTools` instance
    itself, pinning the surface to exactly `{"search", "fetch"}` and failing
    closed if the retired slot ever reappears.
    """
    registry = tool_registry(research_tools)

    assert "verify_citation" not in registry
    assert not hasattr(research_tools, "verify_citation")


# --- No privileged handle ---------------------------------------------------------


def test_research_tools_instance_holds_no_privileged_module_handles(
    research_tools: ResearchTools,
) -> None:
    """No slot value's type lives in `windbreak.ledger`, `windbreak.config`, or
    `windbreak.connector` -- a `ResearchTools` instance holds only its cache,
    allowlist, and the two injected transports, never a privileged handle.
    """
    forbidden_module_prefixes = (
        "windbreak.ledger",
        "windbreak.config",
        "windbreak.connector",
    )
    for slot_name in type(research_tools).__slots__:
        value = getattr(research_tools, slot_name, None)
        module_name = type(value).__module__
        assert not module_name.startswith(forbidden_module_prefixes)


def test_build_research_tools_signature_has_no_privileged_parameters() -> None:
    """`build_research_tools` accepts only allowlist/cache/transport parameters.

    `max_bytes` (issue #453) is the cache's byte ceiling. It arrives as a bare
    `int`, never as a config object: the sandbox may not import
    `windbreak.config`, so the operator's `forecast.research.cache_max_bytes`
    is unwrapped by the scheduler's wiring and handed across as a plain
    number, exactly as `allowed_hosts` is.
    """
    parameter_names = set(inspect.signature(build_research_tools).parameters)

    assert parameter_names == {
        "allowed_hosts",
        "cache_dir",
        "search_transport",
        "fetch_transport",
        "max_bytes",
    }


# --- Import boundary AST check (the CI teeth) -------------------------------------
#
# Self-contained (stdlib `ast` only, no import of scripts/lint_no_floats.py) so
# this test module's enforcement of the sandbox's import boundary does not
# depend on any other tool's implementation staying stable. `ast.walk` visits
# every node in the tree regardless of nesting, so imports inside an
# `if TYPE_CHECKING:` block are inspected exactly like top-level imports --
# the boundary applies whether or not an import is "just for typing".

_FORBIDDEN_IMPORT_PREFIXES: frozenset[str] = frozenset(
    {
        "windbreak.ledger",
        "windbreak.order_gateway",
        "windbreak.riskkernel",
        "windbreak.config",
        "windbreak.connector",
    }
)

#: The single, narrow allowance carved out of the broader `windbreak.connector`
#: prohibition: the read-only, dataclass-only `NormalizedMarket` shapes.
_ALLOWED_IMPORT_NAMES: frozenset[str] = frozenset({"windbreak.connector.models"})


def _matches_prefix(candidate: str, prefixes: frozenset[str]) -> bool:
    """Return whether `candidate` equals, or is dot-nested under, any prefix.

    Args:
        candidate: A dotted module (or module-plus-symbol) name.
        prefixes: Dotted prefixes to test `candidate` against.

    Returns:
        `True` if `candidate` exactly equals a prefix, or starts with a prefix
        followed by a dot.
    """
    return any(
        candidate == prefix or candidate.startswith(prefix + ".") for prefix in prefixes
    )


def _find_forbidden_imports(
    source: str,
    forbidden_prefixes: frozenset[str],
    allowed_names: frozenset[str],
) -> tuple[str, ...]:
    """Return every forbidden dotted import name found in `source`.

    For a plain `import a.b.c`, `alias.name` (`"a.b.c"`) is tested directly.
    For an *absolute* `from a.b import c`, both `"a.b"` (the module) and
    `"a.b.c"` (the module plus the imported symbol) are candidates -- so `from
    windbreak.connector import MarketConnector` is caught even though
    `windbreak.connector.MarketConnector` is not a real submodule path. The
    `allowed_names` allowance is consulted *before* the forbidden-prefix test,
    and against both candidates, so `from windbreak.connector import models`
    (module-plus-symbol == the allowed `windbreak.connector.models`) and `from
    windbreak.connector.models import NormalizedMarket` (module itself ==
    the allowance) are both left unflagged.

    A *relative* `from`-import (any `node.level > 0`, including a bare
    `from . import config` whose `node.module` is `None`) is **always** flagged.
    Two reasons: a relative import can name a forbidden module
    (`from ..ledger import store` resolves to `windbreak.ledger`) yet its
    `node.module` alone (`"ledger"`) matches no absolute prefix, so resolving it
    correctly would require the importer's package path; and the forecast
    package convention is absolute imports throughout (verified: zero relative
    imports ship today), so a `.`-prefixed import is itself the anomaly the
    firewall must fail closed on. Flagging every relative import keeps the check
    sound without depending on package-path resolution -- and makes it strictly
    stronger than an import-linter `forbidden` contract, which would silently
    permit a benign-looking `from . import <forbidden>` re-export.

    Violations are flagged regardless of `TYPE_CHECKING` nesting (`ast.walk`
    does not distinguish).

    Args:
        source: Python source text to parse.
        forbidden_prefixes: Dotted module prefixes that may never be imported.
        allowed_names: Dotted names explicitly exempted from the prefix ban.

    Returns:
        The offending dotted names, in AST-traversal order. May contain
        duplicates if the same forbidden name is imported more than once.
        Relative imports are reported by their `.`-prefixed spelling.
    """
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in allowed_names:
                    continue
                if _matches_prefix(alias.name, forbidden_prefixes):
                    found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                found.append("." * node.level + (node.module or ""))
                continue
            if node.module is None:
                continue
            for alias in node.names:
                combined = f"{node.module}.{alias.name}"
                if node.module in allowed_names or combined in allowed_names:
                    continue
                if _matches_prefix(node.module, forbidden_prefixes):
                    found.append(node.module)
                elif _matches_prefix(combined, forbidden_prefixes):
                    found.append(combined)
    return tuple(found)


def test_windbreak_forecast_package_has_zero_forbidden_imports() -> None:
    """Every shipped `windbreak/forecast/*.py` module imports across the boundary
    cleanly -- zero forbidden-prefix hits anywhere in the package.
    """
    violations: list[str] = []
    for path in sorted(_FORECAST_PACKAGE_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        violations.extend(
            f"{path.name}:{name}"
            for name in _find_forbidden_imports(
                source, _FORBIDDEN_IMPORT_PREFIXES, _ALLOWED_IMPORT_NAMES
            )
        )

    assert violations == []


@pytest.mark.parametrize(
    "source",
    [
        "import windbreak.ledger.store\n",
        "import windbreak.order_gateway\n",
        "import windbreak.riskkernel\n",
        "import windbreak.config\n",
        "import windbreak.connector.interface\n",
        "from windbreak.connector import MarketConnector\n",
        # Relative imports are always flagged: a `..`-hop can reach a forbidden
        # module, and a bare `from . import x` hides its target from the check.
        "from ..ledger import store\n",
        "from ..config import schema\n",
        "from . import config\n",
    ],
)
def test_ast_checker_flags_each_seeded_forbidden_import(source: str) -> None:
    """Each seeded synthetic source string is flagged as a forbidden import."""
    violations = _find_forbidden_imports(
        source, _FORBIDDEN_IMPORT_PREFIXES, _ALLOWED_IMPORT_NAMES
    )

    assert violations, f"expected a violation for: {source!r}"


@pytest.mark.parametrize(
    "source",
    [
        "from windbreak.connector.models import NormalizedMarket\n",
        "from windbreak.connector import models\n",
    ],
)
def test_ast_checker_does_not_flag_the_connector_models_allowance(
    source: str,
) -> None:
    """The `windbreak.connector.models` allowance is not flagged, either as a
    direct import or via `from windbreak.connector import models`.
    """
    violations = _find_forbidden_imports(
        source, _FORBIDDEN_IMPORT_PREFIXES, _ALLOWED_IMPORT_NAMES
    )

    assert violations == ()


# --- Pipeline integration ----------------------------------------------------------


def test_run_pipeline_with_sandboxed_tools_is_schema_valid_and_byte_deterministic(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """Two fresh-instance runs over sandboxed tools yield `==` records and
    byte-identical JSON payloads, proving `research_tools` threads through
    stage 5 without breaking the pipeline's existing determinism guarantee.
    """
    tools_a = research_tools_factory(cache_dir=tmp_path / "cache-a")
    tools_b = research_tools_factory(cache_dir=tmp_path / "cache-b")

    record_a = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=tools_a,
    )
    record_b = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=tools_b,
    )

    assert record_a == record_b
    assert record_a.triage_stage == "full"
    payload_a = json.dumps(forecast_record_to_payload(record_a), sort_keys=True)
    payload_b = json.dumps(forecast_record_to_payload(record_b), sort_keys=True)
    assert payload_a == payload_b


def test_run_pipeline_raises_egress_denied_for_off_allowlist_search_result(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """A search result off the allowlist makes `run_pipeline` itself raise
    `EgressDeniedError` -- the structural proof the guard sits on the pipeline path
    (stage 5's `tools.fetch` call), not only reachable via a direct unit test.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path / "cache",
        search_transport=_StaticSearchTransport("https://evil.example/x"),
    )

    with pytest.raises(EgressDeniedError):
        run_pipeline(
            market,
            baseline,
            transport=make_fake_vote_transport(),
            created_at=created_at,
            research_tools=tools,
        )


def test_run_pipeline_yields_no_citations_when_search_finds_nothing(
    market: NormalizedMarket,
    baseline: BaselineQuoteSnapshot,
    created_at: datetime,
    make_fake_vote_transport: FakeVoteTransportFactory,
    research_tools_factory: ResearchToolsFactory,
    tmp_path: Path,
) -> None:
    """A subquestion whose search finds no candidate URL is skipped (never
    fetched), so the run still produces a schema-valid record -- here with an
    empty citation tuple -- rather than erroring. This pins stage 5's
    `if not urls: continue` no-candidate branch.
    """
    tools = research_tools_factory(
        cache_dir=tmp_path / "cache",
        search_transport=_EmptySearchTransport(),
    )

    record = run_pipeline(
        market,
        baseline,
        transport=make_fake_vote_transport(),
        created_at=created_at,
        research_tools=tools,
    )

    assert record.citations == ()
    assert record.triage_stage == "full"
