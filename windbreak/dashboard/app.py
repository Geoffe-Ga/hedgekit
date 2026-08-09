"""Stub localhost dashboard HTTP surface for Process D (SPEC S5.1, S14).

Serves a single authenticated status page on the loopback interface only.
Per SPEC S14 the dashboard accepts **no public inbound traffic**, so the bind
host is the hardcoded :data:`_BIND_HOST` (``127.0.0.1``) and is deliberately
*not* configurable -- there is no code path that binds any other interface.

Two dependency-injection seams are wired by successor issues: the dashboard
auth ``token`` is minted from configuration (issue #11), and the
``status_source`` callable will be backed by the read-only ledger view (issue
#13). Until then callers pass both explicitly, so this module has no ambient
dependency on either.
"""

from __future__ import annotations

import hmac
import html
import http.server
import json
import logging
import re
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

from windbreak.dashboard.views import (
    render_decisions,
    render_equity_vs_floor,
    render_execution_quality,
    render_live_divergence,
    render_positions,
    render_provider_panel,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from windbreak.dashboard.views import DashboardReadModels
    from windbreak.riskkernel.human_ack import PendingHumanAck

_LOGGER = logging.getLogger("windbreak.dashboard")

#: The only interface the dashboard ever binds. Not configurable: SPEC S14
#: forbids public inbound, so there is no code path to any other host.
_BIND_HOST = "127.0.0.1"

#: The status page's routable path; every non-routed path is a 404.
_ROOT_PATH = "/"

#: The PAPER-loop read-model view paths (issue #48), each gated behind the same
#: bearer auth as ``/`` and rendered from the injected read-models source.
_POSITIONS_PATH = "/positions"
_EQUITY_PATH = "/equity"
_DECISIONS_PATH = "/decisions"

#: The live execution-quality / divergence view paths (issue #58), each gated
#: behind the same bearer auth as ``/`` and rendered from the read-models source.
_EXECUTION_PATH = "/execution"
_DIVERGENCE_PATH = "/divergence"

#: The fleet-observability provider-panel view path (issue #195), gated behind
#: the same bearer auth and rendered from the read-models source.
_PROVIDERS_PATH = "/providers"

#: The human-acknowledgement surface paths (issue #57): ``POST /ack`` grants a
#: named pending acknowledgement, ``GET /acks`` renders the pending ones. Both
#: sit behind the same bearer gate as every other route.
_ACK_PATH = "/ack"
_ACKS_PATH = "/acks"

#: The Authorization scheme the bearer token must be presented under.
_BEARER_PREFIX = "Bearer "

#: Rendered in place of ``last_heartbeat`` before the first heartbeat arrives.
_NO_HEARTBEAT = "never"

#: Plain-text body returned with a 401 challenge.
_UNAUTHORIZED_BODY = "401 Unauthorized: a valid bearer token is required.\n"

#: Plain-text body returned for any path other than the root.
_NOT_FOUND_BODY = "404 Not Found.\n"

#: Plain-text body returned for a successful ``POST /ack``.
_ACK_GRANTED_BODY = "200 OK: acknowledgement granted.\n"

#: Plain-text body returned for a malformed or ill-shaped ``POST /ack`` body.
_BAD_REQUEST_BODY = "400 Bad Request: a 32-hex approval_id is required.\n"

#: An approval id is exactly 32 lowercase hex characters -- the shape
#: ``HumanAckQueue`` mints via ``secrets.token_hex(16)``. The POST handler
#: validates the posted id against this before ever invoking the granter, so a
#: traversal-shaped or otherwise bogus value can never reach a drop-box writer
#: (defense in depth: mirrors ``main._approval_id``'s CLI-side guard).
_APPROVAL_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

#: The largest ``POST /ack`` body accepted, in bytes. A valid body is a tiny
#: JSON object naming a 32-char id, so this cap fails an oversized (or absent
#: ``Content-Length``) body closed rather than reading it into memory.
_MAX_ACK_BODY_BYTES = 256

#: Rendered inside the ``/acks`` page when no acknowledgement is pending.
_NO_PENDING_ACKS = "<p>no pending acknowledgements</p>\n"

#: HTML skeleton for the authenticated status page. ``mode`` and ``heartbeat``
#: are HTML-escaped before substitution to prevent injection from a future
#: untrusted status string.
_STATUS_TEMPLATE = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    '<head><meta charset="utf-8"><title>windbreak dashboard</title></head>\n'
    "<body>\n"
    "<h1>windbreak dashboard</h1>\n"
    "<p>mode: {mode}</p>\n"
    "<p>last heartbeat: {heartbeat}</p>\n"
    "</body>\n"
    "</html>\n"
)

#: HTML skeleton for a PAPER-loop read-model view page. ``title`` is a trusted
#: literal and ``body`` is already fully escaped by the view renderer.
_VIEW_TEMPLATE = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    '<head><meta charset="utf-8"><title>windbreak {title}</title></head>\n'
    "<body>\n"
    "{body}"
    "</body>\n"
    "</html>\n"
)

#: HTML skeleton for the pending-acknowledgements page. ``rows`` is already
#: fully escaped by :func:`_render_pending_acks`.
_ACKS_TEMPLATE = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    '<head><meta charset="utf-8"><title>windbreak acks</title></head>\n'
    "<body>\n"
    "<h1>pending acknowledgements</h1>\n"
    "{rows}"
    "</body>\n"
    "</html>\n"
)


def _render_pending_acks(pending: tuple[PendingHumanAck, ...]) -> str:
    """Render pending acknowledgements as an escaped HTML list.

    Args:
        pending: The pending acknowledgements to render (possibly empty).

    Returns:
        An HTML fragment: a ``<ul>`` of one ``<li>`` per pending acknowledgement
        (approval id, intent id, worst-case cost, and expiry, all HTML-escaped),
        or a "no pending acknowledgements" placeholder when empty.
    """
    if not pending:
        return _NO_PENDING_ACKS
    items = "".join(
        f"<li>approval {html.escape(ack.approval_id)}: "
        f"intent {html.escape(ack.intent_id)}, "
        f"worst-case {ack.worst_case_cost.value} micros, "
        f"expires {ack.expires_at}</li>\n"
        for ack in pending
    )
    return f"<ul>\n{items}</ul>\n"


@dataclass(frozen=True)
class _ViewSpec:
    """One PAPER-loop read-model view's title, read model, and renderer.

    Attributes:
        title: The trusted page-title fragment (never ledger-derived).
        attr: The :class:`~windbreak.dashboard.views.DashboardReadModels`
            attribute holding this view's rows.
        render: The pure renderer projecting those rows into escaped HTML.
    """

    title: str
    attr: str
    render: Callable[[list[dict[str, object]]], str]


#: The read-model views, keyed by their route path: the three PAPER-loop views
#: (issue #48) plus the two live-divergence views (issue #58).
_VIEWS: dict[str, _ViewSpec] = {
    _POSITIONS_PATH: _ViewSpec("positions", "positions", render_positions),
    _EQUITY_PATH: _ViewSpec("equity", "equity_curve", render_equity_vs_floor),
    _DECISIONS_PATH: _ViewSpec("decisions", "decisions", render_decisions),
    _EXECUTION_PATH: _ViewSpec(
        "execution", "execution_quality", render_execution_quality
    ),
    _DIVERGENCE_PATH: _ViewSpec(
        "divergence", "live_divergence", render_live_divergence
    ),
    _PROVIDERS_PATH: _ViewSpec("providers", "provider_panel", render_provider_panel),
}


@dataclass(frozen=True)
class DashboardStatus:
    """Immutable snapshot of the operational status the dashboard renders.

    Attributes:
        mode: The current SPEC operating mode (e.g. ``RESEARCH``, ``PAPER``).
        last_heartbeat: ISO-8601 timestamp of the most recent heartbeat, or
            ``None`` before any heartbeat has been observed.
    """

    mode: str
    last_heartbeat: str | None


class _DashboardServer(http.server.ThreadingHTTPServer):
    """Threading HTTP server carrying the dashboard's injected dependencies.

    Holds the auth ``token`` and ``status_source`` so the request handler --
    which the stdlib instantiates per connection -- can reach them through
    ``self.server`` without any module-level global state.
    """

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[http.server.BaseHTTPRequestHandler],
        *,
        token: str,
        status_source: Callable[[], DashboardStatus],
        read_models_source: Callable[[], DashboardReadModels] | None = None,
        ack_granter: Callable[[str], None] | None = None,
        pending_acks_source: Callable[[], tuple[PendingHumanAck, ...]] | None = None,
    ) -> None:
        """Bind the server and stash its auth token and data sources.

        Args:
            server_address: The ``(host, port)`` to bind.
            handler_class: The request handler class to instantiate per request.
            token: The expected bearer token for every authenticated request.
            status_source: Zero-arg callable returning the current status,
                invoked fresh on each authenticated request.
            read_models_source: Zero-arg callable returning the current PAPER-loop
                read models, invoked fresh on each authenticated view request, or
                ``None`` (the default) to render every view's "no data yet"
                placeholder.
            ack_granter: One-arg callable granting the posted approval id on a
                ``POST /ack``, or ``None`` (the default) so that route 404s as
                an unwired seam.
            pending_acks_source: Zero-arg callable returning the current pending
                acknowledgements for ``GET /acks``, or ``None`` (the default) to
                render the empty placeholder.
        """
        super().__init__(server_address, handler_class)
        self.token = token
        self.status_source = status_source
        self.read_models_source = read_models_source
        self.ack_granter = ack_granter
        self.pending_acks_source = pending_acks_source


class _DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Handle ``GET /`` with a timing-safe bearer-token gate.

    Reads its auth token and status source from the owning
    :class:`_DashboardServer`, so a single handler class serves any server.
    """

    @property
    def _dashboard_server(self) -> _DashboardServer:
        """Return the owning server narrowed to :class:`_DashboardServer`."""
        return cast("_DashboardServer", self.server)

    def do_GET(self) -> None:
        """Route ``GET`` requests: 404 off-route, 401 unauthenticated, else 200.

        The status page (``/``) and every read-model view in :data:`_VIEWS`
        (the PAPER-loop ``/positions``/``/equity``/``/decisions``, issue #48,
        plus the live-divergence ``/execution``/``/divergence``, issue #58) share
        the same timing-safe bearer gate; every other path is a 404 regardless of
        auth.

        Named ``do_GET`` because :class:`http.server.BaseHTTPRequestHandler`
        dispatches by ``"do_" + command``; the name is fixed by that contract,
        not a style choice (see the ``ignore-names`` ruff config).
        """
        if self.path == _ROOT_PATH:
            self._authorized_or(self._send_status)
            return
        view = _VIEWS.get(self.path)
        if view is not None:
            self._authorized_or(lambda: self._send_view(view))
            return
        if self.path == _ACKS_PATH:
            self._authorized_or(self._send_acks)
            return
        self._send(HTTPStatus.NOT_FOUND, _NOT_FOUND_BODY, "text/plain")

    def do_POST(self) -> None:
        """Route ``POST`` requests: ``/ack`` grants; every other path is a 404.

        ``POST /ack`` shares the same timing-safe bearer gate as every other
        route, so an unauthenticated post is a 401 (the ``ack_granter`` is never
        reached). With no ``ack_granter`` wired the route 404s as an unwired
        seam. Named ``do_POST`` because
        :class:`http.server.BaseHTTPRequestHandler` dispatches by
        ``"do_" + command``.
        """
        if self.path != _ACK_PATH:
            self._send(HTTPStatus.NOT_FOUND, _NOT_FOUND_BODY, "text/plain")
            return
        self._authorized_or(self._grant_ack)

    def _authorized_or(self, send_ok: Callable[[], None]) -> None:
        """Run ``send_ok`` when the request is authorized, else send a 401.

        Args:
            send_ok: The zero-arg responder to invoke on a valid bearer token.
        """
        if not self._is_authorized():
            self._send_unauthorized()
            return
        send_ok()

    def _is_authorized(self) -> bool:
        """Return whether the request carries the exact bearer token.

        Uses :func:`hmac.compare_digest` for a timing-safe, exact match so a
        near-miss or prefix token cannot be inferred by response timing. Both
        sides are compared as UTF-8 bytes so a non-ASCII presented token is
        rejected cleanly (a 401) instead of raising ``TypeError`` from
        ``compare_digest``'s ASCII-only string path.

        Returns:
            True only for an ``Authorization: Bearer <token>`` header whose
            token matches exactly; False for missing, blank, wrong-scheme, or
            wrong-value headers.
        """
        header = self.headers.get("Authorization", "")
        if not header.startswith(_BEARER_PREFIX):
            return False
        presented = header.removeprefix(_BEARER_PREFIX)
        return hmac.compare_digest(
            presented.encode("utf-8"), self._dashboard_server.token.encode("utf-8")
        )

    def _send_unauthorized(self) -> None:
        """Send a 401 with a ``WWW-Authenticate: Bearer`` challenge."""
        self._send(
            HTTPStatus.UNAUTHORIZED,
            _UNAUTHORIZED_BODY,
            "text/plain",
            extra_headers={"WWW-Authenticate": "Bearer"},
        )

    def _send_status(self) -> None:
        """Render the current status (read fresh) as an authenticated 200."""
        status = self._dashboard_server.status_source()
        raw_heartbeat = status.last_heartbeat
        heartbeat = _NO_HEARTBEAT if raw_heartbeat is None else raw_heartbeat
        body = _STATUS_TEMPLATE.format(
            mode=html.escape(status.mode),
            heartbeat=html.escape(heartbeat),
        )
        self._send(HTTPStatus.OK, body, "text/html")

    def _send_view(self, view: _ViewSpec) -> None:
        """Render one PAPER-loop read-model view (read fresh) as a 200.

        With no ``read_models_source`` wired, the view renders over an empty row
        list -- the documented "no data yet" placeholder -- rather than 404ing or
        500ing, mirroring the ``last_heartbeat=None`` -> ``"never"`` precedent.

        Args:
            view: The view spec selecting the read model and its renderer.
        """
        source = self._dashboard_server.read_models_source
        rows: list[dict[str, object]] = (
            [] if source is None else getattr(source(), view.attr)
        )
        body = _VIEW_TEMPLATE.format(title=view.title, body=view.render(rows))
        self._send(HTTPStatus.OK, body, "text/html")

    def _grant_ack(self) -> None:
        """Grant the approval id from a ``POST /ack`` body, or 404 if unwired.

        With no ``ack_granter`` seam wired, the route 404s (granting nothing is
        not a meaningful default the way an empty read-model list is). Otherwise
        the JSON body's ``approval_id`` is handed to the granter and a 200 is
        returned; the granter is the seam that actually drops the ack file or
        calls the queue, so this handler stays free of kernel imports.
        """
        granter = self._dashboard_server.ack_granter
        if granter is None:
            self._send(HTTPStatus.NOT_FOUND, _NOT_FOUND_BODY, "text/plain")
            return
        approval_id = self._read_ack_approval_id()
        if approval_id is None:
            self._send(HTTPStatus.BAD_REQUEST, _BAD_REQUEST_BODY, "text/plain")
            return
        granter(approval_id)
        self._send(HTTPStatus.OK, _ACK_GRANTED_BODY, "text/plain")

    def _read_ack_approval_id(self) -> str | None:
        """Read and validate the ``approval_id`` from the JSON request body.

        Every failure mode -- a missing/non-numeric/zero/oversized
        ``Content-Length``, an undecodable or non-JSON body, a non-object
        payload, a missing or non-string ``approval_id``, or an id that is not
        exactly 32 lowercase hex characters -- returns ``None`` so the caller
        fails closed with a 400 and never hands a bogus value to the granter.
        The size cap bounds the read so an authenticated client cannot force an
        unbounded body into memory.

        Returns:
            The validated 32-hex ``approval_id``, or ``None`` when the body is
            malformed, oversized, or ill-shaped.
        """
        raw_length = self.headers.get("Content-Length", "")
        if not raw_length.isdigit():
            return None
        length = int(raw_length)
        if not 0 < length <= _MAX_ACK_BODY_BYTES:
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        approval_id = payload.get("approval_id")
        if not isinstance(approval_id, str) or (
            _APPROVAL_ID_PATTERN.fullmatch(approval_id) is None
        ):
            return None
        return approval_id

    def _send_acks(self) -> None:
        """Render the current pending acknowledgements (read fresh) as a 200.

        With no ``pending_acks_source`` wired the page renders the empty
        placeholder, mirroring the read-model views' "no data yet" precedent.
        """
        source = self._dashboard_server.pending_acks_source
        pending: tuple[PendingHumanAck, ...] = () if source is None else source()
        body = _ACKS_TEMPLATE.format(rows=_render_pending_acks(pending))
        self._send(HTTPStatus.OK, body, "text/html")

    def _send(
        self,
        status: HTTPStatus,
        body: str,
        media_type: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """Write a UTF-8 response with the given status, body, and headers.

        Args:
            status: The HTTP status code to send.
            body: The response body text (UTF-8 encoded before sending).
            media_type: The ``Content-Type`` media type, e.g. ``text/html``
                or ``text/plain`` (``; charset=utf-8`` is appended).
            extra_headers: Optional additional response headers to emit.
        """
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{media_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        """Route request logging through the module logger, sans credentials.

        The stdlib default writes to ``stderr`` and can surface request
        details; this override sends the request line through the structured
        module logger instead. The Authorization header is never included, so
        bearer tokens stay out of the log stream. The parameter name ``format``
        is fixed by the supertype's signature (renaming it would break Liskov
        substitution under mypy).

        Args:
            format: The stdlib ``%``-style format string (name mandated by
                :class:`http.server.BaseHTTPRequestHandler`).
            args: The positional arguments for ``format``.
        """
        _LOGGER.info("dashboard request %s", format % args)


def create_server(
    *,
    token: str,
    status_source: Callable[[], DashboardStatus],
    port: int,
    read_models_source: Callable[[], DashboardReadModels] | None = None,
    ack_granter: Callable[[str], None] | None = None,
    pending_acks_source: Callable[[], tuple[PendingHumanAck, ...]] | None = None,
) -> http.server.ThreadingHTTPServer:
    """Build a loopback-bound dashboard server guarded by a bearer token.

    Args:
        token: The bearer token every authenticated request must present.
            Must be non-empty -- a blank token can never be presented, so it
            is rejected as a misconfiguration rather than binding a server no
            client could ever reach.
        status_source: Zero-arg callable returning the current
            :class:`DashboardStatus`, invoked fresh on each authenticated
            request so responses always reflect live state.
        port: The loopback TCP port to bind. ``0`` binds an OS-assigned port.
        read_models_source: Zero-arg callable returning the current PAPER-loop
            :class:`~windbreak.dashboard.views.DashboardReadModels`, invoked fresh
            on each authenticated view request. ``None`` (the default) renders
            every view's "no data yet" placeholder, so the three view routes
            still 200 rather than 404 before any PAPER data exists (issue #48).
        ack_granter: One-arg callable granting the approval id posted to
            ``POST /ack``. ``None`` (the default) 404s that route as an unwired
            seam (issue #57).
        pending_acks_source: Zero-arg callable returning the current pending
            acknowledgements rendered by ``GET /acks``, invoked fresh on each
            authenticated request. ``None`` (the default) renders the empty
            placeholder (issue #57).

    Returns:
        A :class:`http.server.ThreadingHTTPServer` bound to
        ``(127.0.0.1, port)`` and ready for ``serve_forever``.

    Raises:
        ValueError: If ``token`` is the empty string.
    """
    if not token:
        raise ValueError("token must be a non-empty bearer token")
    return _DashboardServer(
        (_BIND_HOST, port),
        _DashboardHandler,
        token=token,
        status_source=status_source,
        read_models_source=read_models_source,
        ack_granter=ack_granter,
        pending_acks_source=pending_acks_source,
    )
