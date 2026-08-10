"""Tests pinning the two operator-facing control claims of issue #449.

Both defects in #449 are the documentation half of the phantom-gate class
this repo closed four times in the gate machinery itself (#359, #401, #351,
#411): a document describing a control the code does not implement, where
nothing fails when the two disagree.

Control 1 -- the dashboard's HTTP surface.
    ``README.md`` said mutations arrive later "beyond the existing
    ``POST /ack``", and ``docs/RUNBOOK.md`` said ``POST /ack`` "grants the
    same acknowledgement" as ``windbreak ack``. The server the CLI builds
    passes no ``ack_granter``, so that route 404s. The same table also
    claimed "the default build exposes neither route" while ``GET /acks``
    answers ``200`` with its empty placeholder, and omitted ``/execution``
    and ``/divergence`` entirely -- two routes the server has served since
    issue #58 that appeared in no document at all.

    So the route table in ``docs/RUNBOOK.md`` is made self-verifying: every
    row is replayed against the server
    :func:`windbreak.main._build_dashboard_server` actually builds, and the
    documented status code must be the status code that comes back. The
    check runs both ways -- a route the server serves that the table omits
    fails just as loudly as a table row the server does not honour -- so
    wiring ``ack_granter`` later turns the ``POST /ack`` row RED on the day
    it is wired, and a new view cannot ship undocumented.

    That table is also the *only* place any document may name a route: every
    root and ``docs/`` markdown file is scanned for route names in code
    spans, ``docs/RUNBOOK.md`` itself included, exempting only the
    marker-delimited table region. Exempting the whole runbook -- the file
    the defect actually lived in -- would have left its likeliest recurrence
    site uncovered.

Control 2 -- ``SECURITY.md``'s preflight invocation.
    ``windbreak preflight --fixture-dir drills/fixtures --json`` names a
    directory that has never existed in this repo, and ``--fixture-dir`` is
    required, so the one documented invocation of the gate SECURITY.md puts
    before every live deployment ends in an argparse usage error.
    ``tests/docs/test_docs_consistency.py`` did not catch it because that
    module validates that documented flags *parse*, never that a documented
    path *resolves*.

    So this module adds the missing dimension: for every option the CLI
    reads an existing path from, a documented non-placeholder value must
    exist in the tree; and SECURITY.md's preflight command specifically is
    executed in-process, with its exit code compared against the exit code
    SECURITY.md documents for it.

Non-vacuity, demonstrated before the documents were corrected -- the route
table was first transcribed *as the docs then claimed* (``POST /ack`` -> 200,
no ``/execution`` or ``/divergence`` row) and SECURITY.md was left untouched,
which produced exactly five RED tests and no others:

- ``test_documented_dashboard_route_answers_its_documented_status[POST /ack
  -> 200]``: the CLI-built server answered ``404``.
- ``test_every_route_the_dashboard_serves_is_documented``: extra items
  ``/divergence``, ``/execution``.
- ``test_every_documented_input_path_option_value_exists``:
  ``SECURITY.md: --fixture-dir drills/fixtures``.
- ``test_security_md_preflight_fixture_dir_exists_and_loads`` and
  ``test_security_md_preflight_command_is_runnable_as_written``:
  ``FileNotFoundError: drills/fixtures/exchange.json``.

``test_security_md_preflight_command_parses_on_the_real_cli`` passed
throughout, which is the point: parsing was never the broken dimension, and a
guard that stopped there would have certified the unrunnable command.

The helper-level tests below (synthetic markdown, near-miss requests) exist
so the guards cannot pass by comparing the wrong dimension: a table parser
that silently returned no rows, or a path scan that skipped every option,
would make the corpus-level assertions vacuously green.
"""

from __future__ import annotations

import argparse
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

from tests.dashboard.test_app import TEST_TOKEN, _bearer, _get, _post
from tests.docs.test_docs_consistency import extract_windbreak_invocations
from windbreak.config.schema import DashboardConfig, WindbreakConfig
from windbreak.connector.fake import FakeExchange
from windbreak.dashboard.app import (
    _ACK_PATH,
    _ACKS_PATH,
    _ROOT_PATH,
    _VIEWS,
    DashboardStatus,
    create_server,
)
from windbreak.main import (
    DASHBOARD_AUTH_ENV_VAR,
    _build_dashboard_server,
    build_parser,
    main,
)

if TYPE_CHECKING:
    import http.server
    from collections.abc import Iterator

# --- shared helpers ---------------------------------------------------------


def _repo_root() -> Path:
    """Return the repo root, resolved from this test file's location.

    Returns:
        The repo root (this file's great-grandparent directory).
    """
    return Path(__file__).resolve().parents[2]


def _read_doc(relative_path: str) -> str:
    """Return the text of a repo-relative markdown document.

    Args:
        relative_path: The document's path relative to the repo root.

    Returns:
        The document's full text.
    """
    return (_repo_root() / relative_path).read_text(encoding="utf-8")


# --- control 1: the dashboard route table -----------------------------------

#: The document carrying the pinned dashboard route table.
DASHBOARD_ROUTES_DOC = "docs/RUNBOOK.md"

#: The HTML-comment markers delimiting the pinned table. Markers (rather than
#: "the first table after a heading") mean an editor can add prose or another
#: table nearby without silently emptying this guard.
_ROUTES_BEGIN_MARKER = "<!-- dashboard-routes:begin"
_ROUTES_END_MARKER = "<!-- dashboard-routes:end -->"

#: The two HTTP methods the handler implements (``do_GET``/``do_POST``); a
#: documented row naming any other method could never be replayed.
_SUPPORTED_METHODS = frozenset({"GET", "POST"})

#: The heading whose section carries the pinned table, and the GitHub anchor
#: other documents link to instead of restating what a route answers.
_DASHBOARD_SECTION_HEADING = "### Observing via the dashboard"
_DASHBOARD_SECTION_ANCHOR = "observing-via-the-dashboard"

#: One inline code span's text (the backticks themselves excluded).
_INLINE_CODE_SPAN_PATTERN = re.compile(r"`([^`\n]+)`")


class DocumentedRoute(NamedTuple):
    """One documented route row from the pinned table.

    Attributes:
        method: The row's HTTP method (``GET`` or ``POST``).
        path: The row's request path (e.g. ``/acks``).
        status: The HTTP status code the row documents.
    """

    method: str
    path: str
    status: int


def _route_id(route: DocumentedRoute) -> str:
    """Return a readable parametrize id for one documented route.

    Args:
        route: The documented route to label.

    Returns:
        A ``"<METHOD> <path> -> <status>"`` label.
    """
    return f"{route.method} {route.path} -> {route.status}"


def _unfence(cell: str) -> str:
    """Strip surrounding backticks and whitespace from one markdown cell.

    Args:
        cell: The raw cell text between two table pipes.

    Returns:
        The cell's text with backticks and surrounding whitespace removed.
    """
    return cell.strip().strip("`").strip()


def split_route_table_region(doc_text: str) -> tuple[str, str]:
    """Split `doc_text` into its pinned route-table region and everything else.

    The marker-delimited region is the one span of the runbook whose route
    claims a mechanism reads: every row in it is replayed against the running
    server. Everything outside it is prose, so the two halves are handled
    differently -- the table is parsed, the prose is scanned for route names
    it must not restate.

    Args:
        doc_text: The full markdown document to split.

    Returns:
        The ``(table_region, surrounding_prose)`` pair. The table region
        starts at the begin marker and stops at the end marker; the prose is
        the document with that region (and the end marker) removed.

    Raises:
        AssertionError: If either marker is missing or they are misordered --
            which would otherwise silently reduce both halves of this guard
            to checking nothing.
    """
    begin = doc_text.find(_ROUTES_BEGIN_MARKER)
    end = doc_text.find(_ROUTES_END_MARKER)
    assert begin != -1, f"{_ROUTES_BEGIN_MARKER} marker is missing"
    assert end > begin, f"{_ROUTES_END_MARKER} marker is missing or misordered"

    prose = doc_text[:begin] + doc_text[end + len(_ROUTES_END_MARKER) :]
    return doc_text[begin:end], prose


def parse_route_table(doc_text: str) -> tuple[DocumentedRoute, ...]:
    """Return every route row inside `doc_text`'s pinned route-table markers.

    Rows are the pipe-delimited lines between the markers, minus the header
    row and the ``|---|`` separator. The first three columns are read as
    method, path, and status; any further columns (the prose description)
    are ignored.

    Args:
        doc_text: The full markdown document to scan.

    Returns:
        Every documented route, in source order.

    Raises:
        AssertionError: If the markers are missing, or a row's status column
            is not an integer -- both of which would otherwise silently
            reduce this guard to checking nothing.
    """
    table_region, _prose = split_route_table_region(doc_text)

    routes: list[DocumentedRoute] = []
    for line in table_region.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [_unfence(cell) for cell in stripped.strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= {"-", ":"} or cells[0] == "Method":
            continue
        method, path, status = cells[0], cells[1], cells[2]
        assert status.isdigit(), (
            f"route row {method} {path} documents a non-numeric status {status!r}"
        )
        routes.append(DocumentedRoute(method, path, int(status)))
    return tuple(routes)


def _documented_routes() -> tuple[DocumentedRoute, ...]:
    """Return the pinned route table's rows from the shipped document.

    Returns:
        Every documented route, in source order.
    """
    return parse_route_table(_read_doc(DASHBOARD_ROUTES_DOC))


def _served_paths() -> frozenset[str]:
    """Return every request path the dashboard handler recognizes.

    Read from :mod:`windbreak.dashboard.app`'s own routing constants rather
    than from a list kept here, so a route added to the server -- and to no
    document -- is discovered by this test instead of being invisible to it.

    Returns:
        The status path, every read-model view path, and both ack paths.
    """
    return frozenset({_ROOT_PATH, _ACKS_PATH, _ACK_PATH, *_VIEWS})


@pytest.fixture(scope="module")
def cli_dashboard_address() -> Iterator[tuple[str, int]]:
    """Serve the dashboard server the CLI builds, and yield its address.

    Built exactly as ``windbreak run --process dashboard`` builds it (no
    ``--ledger-path``), which is the configuration whose behaviour the docs
    describe to operators.

    Yields:
        The bound ``(host, port)`` of the serving CLI-built dashboard.
    """
    config = WindbreakConfig(dashboard=DashboardConfig(port=0))
    args = argparse.Namespace(ledger_path=None)
    server: http.server.ThreadingHTTPServer = _build_dashboard_server(
        args, config, environ={DASHBOARD_AUTH_ENV_VAR: TEST_TOKEN}
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _request(
    address: tuple[str, int], method: str, path: str
) -> tuple[int, dict[str, str], str]:
    """Perform one authenticated request of `method` against `path`.

    Args:
        address: The serving dashboard's ``(host, port)``.
        method: ``GET`` or ``POST``.
        path: The request path.

    Returns:
        The ``(status, headers, body)`` triple of the response.

    Raises:
        ValueError: If `method` is not one the handler implements.
    """
    headers = _bearer(TEST_TOKEN)
    if method == "GET":
        return _get(address, path, headers=headers)
    if method == "POST":
        return _post(address, path, json_body={}, headers=headers)
    raise ValueError(f"unsupported documented method {method!r}")


def test_route_table_documents_at_least_every_view_route() -> None:
    """The parsed table is non-empty and covers more than a single row.

    Guards against a parser or marker change that silently yields no rows,
    which would make every table-driven assertion below vacuously green.
    """
    routes = _documented_routes()

    assert len(routes) >= len(_VIEWS) + 1


def test_parse_route_table_reads_method_path_and_status() -> None:
    """A synthetic marked table parses into its ``(method, path, status)``."""
    doc_text = (
        f"{_ROUTES_BEGIN_MARKER} -->\n\n"
        "| Method | Path | Status | Renders |\n"
        "|---|---|---|---|\n"
        "| `GET` | `/` | `200` | Status. |\n"
        "| `POST` | `/ack` | `404` | Unwired. |\n\n"
        f"{_ROUTES_END_MARKER}\n"
    )

    assert parse_route_table(doc_text) == (
        DocumentedRoute("GET", "/", 200),
        DocumentedRoute("POST", "/ack", 404),
    )


def test_parse_route_table_ignores_rows_outside_the_markers() -> None:
    """A table before the begin marker contributes no rows."""
    doc_text = (
        "| Method | Path | Status |\n"
        "|---|---|---|\n"
        "| `GET` | `/elsewhere` | `200` |\n\n"
        f"{_ROUTES_BEGIN_MARKER} -->\n\n"
        "| Method | Path | Status |\n"
        "|---|---|---|\n"
        "| `GET` | `/` | `200` |\n\n"
        f"{_ROUTES_END_MARKER}\n"
    )

    assert parse_route_table(doc_text) == (DocumentedRoute("GET", "/", 200),)


def test_parse_route_table_rejects_a_non_numeric_status() -> None:
    """A status column that is not an integer fails loudly, never silently."""
    doc_text = (
        f"{_ROUTES_BEGIN_MARKER} -->\n\n"
        "| Method | Path | Status |\n"
        "|---|---|---|\n"
        "| `GET` | `/` | `works` |\n\n"
        f"{_ROUTES_END_MARKER}\n"
    )

    with pytest.raises(AssertionError, match="non-numeric status"):
        parse_route_table(doc_text)


def test_parse_route_table_requires_the_begin_marker() -> None:
    """A document with no begin marker fails rather than parsing zero rows."""
    with pytest.raises(AssertionError, match="begin marker is missing"):
        parse_route_table("| `GET` | `/` | `200` |\n")


def test_split_route_table_region_separates_the_table_from_its_prose() -> None:
    """Route names inside the markers stay in; prose around them stays out."""
    doc_text = (
        "The `POST /ack` route grants it.\n\n"
        f"{_ROUTES_BEGIN_MARKER} -->\n\n"
        "| `POST` | `/ack` | `404` |\n\n"
        f"{_ROUTES_END_MARKER}\n\n"
        "And `GET /acks` renders them.\n"
    )

    table_region, prose = split_route_table_region(doc_text)

    assert route_code_spans(table_region) == ("/ack",)
    assert route_code_spans(prose) == ("POST /ack", "GET /acks")


@pytest.mark.parametrize("route", _documented_routes(), ids=_route_id)
def test_documented_dashboard_route_answers_its_documented_status(
    route: DocumentedRoute, cli_dashboard_address: tuple[str, int]
) -> None:
    """Every documented route answers, authenticated, its documented status.

    This is the assertion issue #449 asked for: the status an operator reads
    in the runbook is the status the shipped `windbreak run --process
    dashboard` server returns for that exact method and path.
    """
    assert route.method in _SUPPORTED_METHODS, (
        f"{route.method} is not a method the dashboard handler implements"
    )

    status, _headers, _body = _request(cli_dashboard_address, route.method, route.path)

    assert status == route.status


def test_every_route_the_dashboard_serves_is_documented() -> None:
    """Every routing path in `windbreak.dashboard.app` appears in the table."""
    documented = {route.path for route in _documented_routes()}

    assert _served_paths() - documented == set()


def test_every_documented_route_path_is_a_real_server_route() -> None:
    """No table row names a path the dashboard handler does not route."""
    documented = {route.path for route in _documented_routes()}

    assert documented - _served_paths() == set()


def route_code_spans(doc_text: str) -> tuple[str, ...]:
    """Return every inline code span in `doc_text` that names a served route.

    Naming a route in an inline code span (`` `POST /ack` ``, `` `/acks` ``)
    is this corpus's convention for referring to the dashboard's HTTP
    surface, the same way `` `config.<dotted.key>` `` is its convention for a
    config key. The match is word-bounded so ``/execution`` inside the prose
    phrase "research/execution firewall" is not a route reference, and the
    root path ``/`` is excluded as too generic to identify one.

    Args:
        doc_text: The full markdown document to scan.

    Returns:
        Each matching code span's text, in source order.
    """
    patterns = [
        re.compile(rf"(?<![\w/-]){re.escape(path)}(?![\w/-])")
        for path in sorted(_served_paths() - {_ROOT_PATH})
    ]
    return tuple(
        span
        for span in _INLINE_CODE_SPAN_PATTERN.findall(doc_text)
        if any(pattern.search(span) for pattern in patterns)
    )


def test_route_code_spans_finds_a_route_reference() -> None:
    """An inline code span naming a route is reported."""
    assert route_code_spans("the existing `POST /ack` grants it") == ("POST /ack",)


def test_route_code_spans_ignores_a_prose_slash_word() -> None:
    """``research/execution`` in prose is not a reference to ``/execution``."""
    assert route_code_spans("the `research/execution` firewall") == ()


def _unverified_text(doc: Path, root: Path) -> str:
    """Return the part of `doc` whose route claims no mechanism replays.

    For every document but the one carrying the pinned table that is the
    whole file. For the runbook it is everything outside the markers: the
    table's own rows are replayed against the running server row by row, so
    they are the one place a route may be named, and the prose around them --
    including the paragraphs above the table, where issue #449's defect
    actually lived -- is scanned like any other document.

    Args:
        doc: The document being scanned.
        root: The repo root, used to recognize the pinned document.

    Returns:
        The document text whose route references must be delegated away.
    """
    doc_text = doc.read_text(encoding="utf-8")
    if doc != root / DASHBOARD_ROUTES_DOC:
        return doc_text
    _table_region, prose = split_route_table_region(doc_text)
    return prose


def test_the_pinned_table_names_every_route_it_exempts() -> None:
    """The exempted region is where the route names actually are.

    Non-vacuity for the stray scan below: that scan asserts an empty result,
    so a `route_code_spans` that matched nothing anywhere -- a renamed
    routing constant, a broken word-boundary -- would leave it green while
    checking nothing. Requiring the exempted region to name every served
    route makes the same failure RED here first.
    """
    table_region, _prose = split_route_table_region(_read_doc(DASHBOARD_ROUTES_DOC))

    assert set(route_code_spans(table_region)) == _served_paths() - {_ROOT_PATH}


def test_every_dashboard_route_reference_lives_in_the_pinned_table() -> None:
    """Only the pinned route table names dashboard routes -- in any document.

    Exactly one *region* of one document describes the dashboard's HTTP
    surface; every other document, and the runbook's own surrounding prose,
    links to it. Without this rule a second sentence can restate a route's
    behaviour in prose no mechanism reads -- which is precisely how "the
    existing ``POST /ack``" survived in README.md while the route 404ed, and
    how ``POST /ack`` "grants the same acknowledgement" survived three
    paragraphs above the table that contradicts it.
    """
    root = _repo_root()
    strays = [
        f"{doc.relative_to(root)}: `{span}`"
        for doc in _doc_corpus()
        for span in route_code_spans(_unverified_text(doc, root))
    ]

    assert strays == []


def test_readme_delegates_dashboard_route_claims_to_the_pinned_table() -> None:
    """README points at the pinned table instead of restating route behaviour.

    The claim issue #449 filed ("the existing ``POST /ack``") lived in README
    prose, which no mechanism can verify. The structural fix is that exactly
    one document states what a route answers, and README links to it; this
    pins the link (and the heading anchor it targets) so the delegation cannot
    be quietly dropped and replaced by a second, unchecked claim.
    """
    link_target = f"{DASHBOARD_ROUTES_DOC}#{_DASHBOARD_SECTION_ANCHOR}"

    assert link_target in _read_doc("README.md")
    assert _DASHBOARD_SECTION_HEADING in _read_doc(DASHBOARD_ROUTES_DOC)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/ack"),
        ("POST", "/"),
        ("POST", "/acks"),
        ("GET", "/acks/"),
        ("GET", "/positions/"),
        ("GET", "/provider"),
    ],
    ids=repr,
)
def test_a_near_miss_of_a_documented_route_is_a_404(
    method: str, path: str, cli_dashboard_address: tuple[str, int]
) -> None:
    """Near-misses 404, so the documented 200s are matches, not a catch-all.

    Each case differs from a documented row in exactly one dimension: the
    method (`GET /ack`, `POST /`, `POST /acks`), a trailing slash
    (`/acks/`, `/positions/`), or one character (`/provider`). Without these
    the table could be "verified" by a server that answered 200 to
    everything.
    """
    status, _headers, _body = _request(cli_dashboard_address, method, path)

    assert status == 404


@pytest.fixture
def granter_wired_dashboard() -> Iterator[tuple[tuple[str, int], list[str]]]:
    """Serve a dashboard built with the ``ack_granter`` seam wired.

    This is the "library caller that passes both seams" build the runbook
    describes next to the pinned table -- the only configuration in which a
    write actually happens.

    Yields:
        The bound ``(host, port)`` and the list every granted approval id is
        appended to.
    """
    granted: list[str] = []
    server = create_server(
        token=TEST_TOKEN,
        status_source=lambda: DashboardStatus(mode="RESEARCH", last_heartbeat=None),
        port=0,
        ack_granter=granted.append,
        pending_acks_source=tuple,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address, granted
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("path", ["/", "/acks", "/ack/", "/positions"], ids=repr)
def test_a_wired_granter_is_reachable_only_at_the_documented_ack_path(
    path: str, granter_wired_dashboard: tuple[tuple[str, int], list[str]]
) -> None:
    """With the seam wired, only ``POST /ack`` reaches the granter.

    The pinned table names one write path. If ``POST`` routing were ever
    widened to a catch-all, the CLI-built server would still answer `404`
    everywhere (its granter is `None`), so the table alone could not see the
    change -- while a seam-wired deployment would silently grant
    acknowledgements at any path. This pins the path dimension where it is
    observable.
    """
    address, granted = granter_wired_dashboard
    approval_id = "0" * 32

    status, _headers, _body = _post(
        address,
        path,
        json_body={"approval_id": approval_id},
        headers=_bearer(TEST_TOKEN),
    )

    assert status == 404
    assert granted == []


def test_a_wired_granter_grants_at_the_documented_ack_path(
    granter_wired_dashboard: tuple[tuple[str, int], list[str]],
) -> None:
    """The documented write path does reach the granter when it is wired.

    The negative test above would pass against a server that granted nothing
    anywhere; this is its paired positive case.
    """
    address, granted = granter_wired_dashboard
    approval_id = "a1" * 16

    status, _headers, _body = _post(
        address,
        _ACK_PATH,
        json_body={"approval_id": approval_id},
        headers=_bearer(TEST_TOKEN),
    )

    assert status == 200
    assert granted == [approval_id]


def test_an_unauthenticated_request_to_a_documented_route_is_401(
    cli_dashboard_address: tuple[str, int],
) -> None:
    """The documented statuses are the authenticated ones; anonymous gets 401.

    Pins that the table is read as "with a valid bearer token" -- the
    documented `200`s are never reachable without one.
    """
    status, _headers, _body = _get(cli_dashboard_address, "/")

    assert status == 401


# --- control 2: documented paths that must resolve --------------------------


#: The doc corpus scanned for CLI invocations: every root-level markdown doc
#: plus everything under ``docs/``. A glob (not a curated list) so a new
#: operator document is covered the day it lands.
def _doc_corpus() -> tuple[Path, ...]:
    """Return every markdown document whose CLI invocations are scanned.

    Returns:
        Root-level ``*.md`` files followed by every ``docs/**/*.md``, sorted.
    """
    root = _repo_root()
    return (*sorted(root.glob("*.md")), *sorted((root / "docs").rglob("*.md")))


#: Options whose value the CLI *reads*: a documented literal value must
#: therefore resolve in this tree, or the documented command cannot run.
READS_EXISTING_PATH_OPTIONS = frozenset(
    {
        "--cassette-path",
        "--config",
        "--fixture-dir",
        "--paper-books-dir",
        "--secrets-file",
        "--snapshot-fixture-dir",
    }
)

#: Options naming state the operator's own run creates or appends to. A
#: documented value here (e.g. ``var/ledger.db``) is an example destination,
#: not a repo file, so its existence is deliberately not asserted.
OPERATOR_CREATED_PATH_OPTIONS = frozenset(
    {
        "--anchor-path",
        "--ledger-path",
        "--output-dir",
        "--report-dir",
        "--state-dir",
    }
)

#: Option-name shapes treated as carrying a filesystem path.
_PATH_OPTION_SUFFIXES = ("-dir", "-path", "-file")

#: Path-carrying options whose name does not match `_PATH_OPTION_SUFFIXES`.
_EXTRA_PATH_OPTIONS = frozenset({"--config"})


def _is_path_option(option: str) -> bool:
    """Return whether `option` names a filesystem path on the CLI.

    Args:
        option: A long option string (e.g. ``--fixture-dir``).

    Returns:
        True for the ``-dir``/``-path``/``-file`` shapes and ``--config``.
    """
    return option.endswith(_PATH_OPTION_SUFFIXES) or option in _EXTRA_PATH_OPTIONS


def _parser_path_options() -> frozenset[str]:
    """Return every path-carrying option on the real CLI parser.

    Walks the real :func:`windbreak.main.build_parser` tree, including every
    subparser, so a new path option is discovered here rather than assumed.

    Returns:
        Every path-carrying long option string across all subcommands.
    """
    options: set[str] = set()

    def walk(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for subparser in action.choices.values():
                    walk(subparser)
                continue
            options.update(
                option for option in action.option_strings if _is_path_option(option)
            )

    walk(build_parser())
    return frozenset(options)


def _is_placeholder(value: str) -> bool:
    """Return whether a documented option value is a placeholder, not a path.

    Three placeholder forms are recognized: angle-bracket metavariables
    (``<dir>``), elisions (``...`` / ``…``, used where a doc shows a command's
    shape rather than its arguments), and absolute paths, which name a
    location in the operator's own filesystem and can never be checked
    against this repo.

    Args:
        value: The documented option value.

    Returns:
        True when the value is a placeholder rather than a repo-relative path.
    """
    return (
        "<" in value
        or ">" in value
        or "…" in value
        or value.strip(".") == ""
        or value.startswith("/")
    )


def input_path_claims(doc_text: str) -> tuple[tuple[str, str], ...]:
    """Return every ``(option, value)`` naming a repo path the CLI must read.

    Scans the document's ``windbreak ...`` invocations, keeps option/value
    pairs whose option is in :data:`READS_EXISTING_PATH_OPTIONS`, and drops
    placeholder values.

    Args:
        doc_text: The full markdown document to scan.

    Returns:
        Each surviving ``(option, value)`` pair, in source order.
    """
    claims: list[tuple[str, str]] = []
    for tokens in extract_windbreak_invocations(doc_text):
        for index, token in enumerate(tokens):
            option, _, inline_value = token.partition("=")
            if option not in READS_EXISTING_PATH_OPTIONS:
                continue
            value = inline_value or (
                tokens[index + 1] if index + 1 < len(tokens) else ""
            )
            if value and not value.startswith("-") and not _is_placeholder(value):
                claims.append((option, value))
    return tuple(claims)


def test_every_path_option_on_the_cli_is_classified() -> None:
    """Every path-carrying CLI option is classified as read-or-created.

    A new path option forces an explicit decision about whether documented
    values of it must exist, rather than defaulting to unchecked.
    """
    classified = READS_EXISTING_PATH_OPTIONS | OPERATOR_CREATED_PATH_OPTIONS

    assert _parser_path_options() - classified == set()


def test_no_path_option_is_classified_both_ways() -> None:
    """The read-existing and operator-created classifications are disjoint."""
    assert not READS_EXISTING_PATH_OPTIONS & OPERATOR_CREATED_PATH_OPTIONS


def test_input_path_claims_finds_a_relative_fixture_dir() -> None:
    """A documented relative ``--fixture-dir`` value is reported as a claim."""
    doc_text = "```bash\nwindbreak preflight --fixture-dir drills/fixtures --json\n```"

    assert input_path_claims(doc_text) == (("--fixture-dir", "drills/fixtures"),)


def test_input_path_claims_reads_a_module_entrypoint_invocation() -> None:
    """``python -m windbreak ...`` is scanned like the console script is.

    This is the exact form issue #449 used to reproduce its defect, so a
    documented path behind it must be checked, not skipped.
    """
    doc_text = (
        "```bash\npython -m windbreak preflight --fixture-dir drills/fixtures\n```"
    )

    assert input_path_claims(doc_text) == (("--fixture-dir", "drills/fixtures"),)


def test_input_path_claims_reads_an_equals_joined_value() -> None:
    """``--fixture-dir=<value>`` is read the same as a space-separated value."""
    doc_text = "```bash\nwindbreak preflight --fixture-dir=drills/fixtures\n```"

    assert input_path_claims(doc_text) == (("--fixture-dir", "drills/fixtures"),)


def test_input_path_claims_skips_placeholders_and_absolute_paths() -> None:
    """Metavariable, elided, and absolute values are not repo-path claims."""
    doc_text = (
        "```bash\nwindbreak drill key-rotation --fixture-dir <dir>\n"
        "windbreak run --config /path/to/windbreak.yaml\n"
        "windbreak run --paper-books-dir ... --cassette-path …\n```"
    )

    assert input_path_claims(doc_text) == ()


def test_input_path_claims_keeps_a_dotted_relative_path() -> None:
    """A real dotted path (``./x``) is a claim; only bare elisions are skipped."""
    doc_text = "```bash\nwindbreak preflight --fixture-dir ./drills/fixtures\n```"

    assert input_path_claims(doc_text) == (("--fixture-dir", "./drills/fixtures"),)


def test_input_path_claims_skips_operator_created_paths() -> None:
    """A relative ``--output-dir`` is operator state, not a repo path claim."""
    doc_text = (
        "```bash\nwindbreak rebuild --ledger-path var/ledger.db "
        "--output-dir var/read-models\n```"
    )

    assert input_path_claims(doc_text) == ()


def test_every_documented_input_path_option_value_exists() -> None:
    """Every documented path the CLI must read resolves in this tree.

    ``tests/docs/test_docs_consistency.py`` already proves documented flags
    parse; this proves the values behind them are real, which is the exact
    gap that let ``--fixture-dir drills/fixtures`` sit in SECURITY.md.
    """
    root = _repo_root()
    missing = [
        f"{doc.relative_to(root)}: {option} {value}"
        for doc in _doc_corpus()
        for option, value in input_path_claims(doc.read_text(encoding="utf-8"))
        if not (root / value).exists()
    ]

    assert missing == []


# --- control 2: SECURITY.md's preflight invocation --------------------------

#: The document whose preflight invocation is executed by this module.
PREFLIGHT_DOC = "SECURITY.md"

#: The documented exit code, written in SECURITY.md prose as ``exits `N` ``.
_DOCUMENTED_EXIT_CODE_PATTERN = re.compile(r"exits `(\d+)`")

#: The heading opening the section that documents the preflight command. The
#: exit-code claim is read from inside this section only: SECURITY.md
#: documents several other commands, and an ``exits `N` `` sentence added to
#: any of them would otherwise silently repoint the assertion below at prose
#: about a different command.
_PREFLIGHT_SECTION_HEADING = "## Preflight: production-readiness checks"


def _preflight_section(doc_text: str) -> str:
    """Return the text of `doc_text`'s preflight section.

    Args:
        doc_text: The full SECURITY.md text.

    Returns:
        The section text, from its heading to the next ``##`` heading (or the
        end of the document).

    Raises:
        AssertionError: If the section heading is absent, which would leave
            the scope of the exit-code claim below undefined.
    """
    start = doc_text.find(_PREFLIGHT_SECTION_HEADING)
    assert start != -1, (
        f"{PREFLIGHT_DOC} must carry a {_PREFLIGHT_SECTION_HEADING!r} section"
    )
    end = doc_text.find("\n## ", start + len(_PREFLIGHT_SECTION_HEADING))
    return doc_text[start:] if end == -1 else doc_text[start:end]


def _documented_preflight_invocation() -> tuple[str, ...]:
    """Return SECURITY.md's single documented ``windbreak preflight`` command.

    A bare `` `windbreak preflight` `` code span is prose naming the verb, not
    a command an operator copies, so only invocations carrying arguments are
    considered.

    Returns:
        The invocation's shlex tokens, ``windbreak`` included.

    Raises:
        AssertionError: If SECURITY.md documents zero or several argument-
            carrying preflight invocations -- either would make "the
            documented command" an ambiguous claim this guard could not pin.
    """
    invocations = [
        tokens
        for tokens in extract_windbreak_invocations(_read_doc(PREFLIGHT_DOC))
        if tokens[1:2] == ("preflight",) and len(tokens) > 2
    ]
    assert len(invocations) == 1, (
        f"{PREFLIGHT_DOC} must document exactly one preflight invocation; "
        f"found {len(invocations)}"
    )
    return invocations[0]


def documented_preflight_exit_code(doc_text: str) -> int:
    """Return the exit code `doc_text`'s preflight section claims.

    Scoped to the preflight section rather than the whole document: taking
    the first ``exits `N` `` anywhere in SECURITY.md would let an unrelated
    sentence added above it repoint this assertion at a different command's
    prose, and the suite would go on passing against the wrong claim.

    Args:
        doc_text: The full SECURITY.md text.

    Returns:
        The integer inside the section's ``exits `N` `` claim.

    Raises:
        AssertionError: If the section states zero or several such exit
            codes -- either leaves "the documented exit code" ambiguous.
    """
    claims = _DOCUMENTED_EXIT_CODE_PATTERN.findall(_preflight_section(doc_text))
    assert len(claims) == 1, (
        f"{PREFLIGHT_DOC}'s preflight section must state the command's exit "
        f"code exactly once as ``exits `N```; found {len(claims)}"
    )
    return int(claims[0])


def test_documented_preflight_exit_code_ignores_another_section_claim() -> None:
    """An ``exits `N` `` sentence above the preflight section is not read.

    The regression this pins: an unrelated claim added earlier in SECURITY.md
    used to become "the documented exit code" simply by being first.
    """
    doc_text = (
        "## Credential boundaries\n\n"
        "A key with withdrawal scope exits `9` at startup.\n\n"
        f"{_PREFLIGHT_SECTION_HEADING} (SPEC §15)\n\n"
        "Against the shipped fixtures this command exits `1`.\n"
    )

    assert documented_preflight_exit_code(doc_text) == 1


def test_documented_preflight_exit_code_stops_at_the_next_section() -> None:
    """A claim in the section *after* preflight is out of scope too."""
    doc_text = (
        f"{_PREFLIGHT_SECTION_HEADING}\n\nThis command exits `1`.\n\n"
        "## Outbound network egress allowlist\n\n"
        "A blocked host exits `7`.\n"
    )

    assert documented_preflight_exit_code(doc_text) == 1


def test_documented_preflight_exit_code_rejects_two_claims_in_section() -> None:
    """Two exit-code claims in the section are ambiguous, so they fail."""
    doc_text = (
        f"{_PREFLIGHT_SECTION_HEADING}\n\nIt exits `1` here and exits `0` there.\n"
    )

    with pytest.raises(AssertionError, match="exactly once"):
        documented_preflight_exit_code(doc_text)


def test_documented_preflight_exit_code_requires_the_section() -> None:
    """A document with no preflight section fails rather than scanning it all."""
    with pytest.raises(AssertionError, match="section"):
        documented_preflight_exit_code("This command exits `1`.\n")


def test_security_md_preflight_command_parses_on_the_real_cli() -> None:
    """SECURITY.md's preflight command parses without an argparse usage error.

    Before the fix this passed while the command was unrunnable, which is
    exactly why the two assertions below exist.
    """
    tokens = _documented_preflight_invocation()

    args = build_parser().parse_args(tokens[1:])

    assert args.command == "preflight"


def test_security_md_preflight_fixture_dir_exists_and_loads() -> None:
    """The documented ``--fixture-dir`` exists and the connector can load it.

    Existence alone would pass for any directory in the tree; loading it
    through the same ``FakeExchange.from_fixture_dir`` the preflight command
    uses proves the operator's copy-pasted command reaches the checks.
    """
    tokens = _documented_preflight_invocation()
    args = build_parser().parse_args(tokens[1:])
    fixture_dir = _repo_root() / args.fixture_dir

    assert fixture_dir.is_dir(), f"{fixture_dir} does not exist"
    assert FakeExchange.from_fixture_dir(fixture_dir).list_markets() != ()


def test_security_md_preflight_command_is_runnable_as_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running SECURITY.md's command yields the exit code SECURITY.md claims.

    The command is executed in-process from the repo root, exactly as an
    operator would copy it, so a documented gate that cannot run -- or that
    runs to a different verdict than the doc claims -- is a suite failure.
    """
    monkeypatch.chdir(_repo_root())
    tokens = _documented_preflight_invocation()

    exit_code = main(list(tokens[1:]))

    assert exit_code == documented_preflight_exit_code(_read_doc(PREFLIGHT_DOC))
