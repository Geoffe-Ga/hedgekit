"""A live-shaped :class:`FetchTransport` over the HTTP record/replay seam (S8.9).

:class:`LiveFetchTransport` retrieves one URL's content through the injected
:class:`~windbreak.forecast.providers.http_cassettes.HttpTransport` seam --
fake, recording, replay, or forbidden-live -- so a live web fetch composes with
the offline record/replay harness exactly like
:class:`~windbreak.forecast.providers.futuresearch.FutureSearchProvider`. The
transport is dependency-injected, so this module never imports ``requests`` (the
live recorder lives in ``scripts/record_research_cassettes.py``) and, per the
SPEC S8.3 sandbox boundary, never imports ``windbreak.config``. It is stdlib-only
and float-free.

Every fetched response is validated in a fixed order before its body is
returned -- status, then content type, then size:

1. A non-2xx status raises :class:`UnreachableUrlError` (a dead link), before
   any content-type or size check.
2. A response media type -- the part of ``content_type`` before the first
   ``;``, lowercased -- outside ``config.allowed_content_types`` raises
   :class:`ContentTypeRejectedError`.
3. A response body whose UTF-8-encoded byte length exceeds
   ``config.max_body_bytes`` (a strict ``>`` ceiling) raises
   :class:`BodyTooLargeError`.

All three failures subclass :class:`OSError`, so
:func:`windbreak.forecast.pipeline.bounded_web_research`'s existing
``except OSError: continue`` skips a live-fetch failure (and still counts the
page against its budget) exactly like today's ``ConnectionError``, with no
pipeline-side change. A clean response's body is returned verbatim: this
transport never sanitizes or transforms it (the pipeline sanitizes and extracts
a publication date downstream).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from windbreak.forecast.providers.http_cassettes import HttpRequest

if TYPE_CHECKING:
    from windbreak.forecast.providers.http_cassettes import HttpTransport

#: The HTTP method every fetch uses: a GET carries no request body.
_FETCH_METHOD = "GET"

#: A fetch never carries a request body, unlike the POST-shaped search transport.
_EMPTY_BODY = ""

#: The separator whose first occurrence bounds a media type's parameters
#: (e.g. the ``;`` before ``charset=utf-8``), stripped before allowlist matching.
_CONTENT_TYPE_PARAM_SEP = ";"

#: Inclusive lower bound of the HTTP success (2xx) status range.
_HTTP_SUCCESS_MIN: Final = 200

#: Exclusive upper bound of the HTTP success (2xx) status range: 300+ is a
#: dead link for fetch purposes (a redirect a live transport does not follow).
_HTTP_SUCCESS_MAX_EXCLUSIVE: Final = 300


def _media_type(content_type: str) -> str:
    """Reduce a raw ``Content-Type`` value to a bare, lowercased media type.

    Everything from the first ``;`` onward (e.g. ``; charset=utf-8``) is
    stripped, and the result lowercased, so allowlist matching is
    case-insensitive and parameter-insensitive.

    Args:
        content_type: The raw response ``content_type`` string.

    Returns:
        The lowercased media type with any parameters removed.
    """
    return content_type.split(_CONTENT_TYPE_PARAM_SEP)[0].strip().lower()


class UnreachableUrlError(OSError):
    """Raised when a fetch returns a non-2xx status (a dead link).

    An :class:`OSError` subclass so
    :func:`windbreak.forecast.pipeline.bounded_web_research` skips (and still
    counts) the page through its existing ``except OSError`` branch.
    """


class ContentTypeRejectedError(OSError):
    """Raised when a fetched response's media type is off the allowlist.

    An :class:`OSError` subclass, for the same fail-open-to-skip reason as
    :class:`UnreachableUrlError`.
    """


class BodyTooLargeError(OSError):
    """Raised when a fetched body exceeds the configured byte ceiling.

    An :class:`OSError` subclass, for the same fail-open-to-skip reason as
    :class:`UnreachableUrlError`.
    """


@dataclass(frozen=True, slots=True)
class LiveFetchConfig:
    """The pinned budget one :class:`LiveFetchTransport` fetches under.

    Attributes:
        max_body_bytes: The maximum accepted response body size, in
            UTF-8-encoded bytes (a strict ``>`` ceiling).
        allowed_content_types: The accepted response media types, each an
            already-lowercased, parameter-free string (e.g. ``text/html``).
    """

    max_body_bytes: int
    allowed_content_types: tuple[str, ...]


class LiveFetchTransport:
    """A :class:`FetchTransport` retrieving one URL over the HTTP seam."""

    def __init__(self, transport: HttpTransport, config: LiveFetchConfig) -> None:
        """Bind the HTTP transport and pinned fetch configuration.

        Args:
            transport: The HTTP transport (fake, recording, replay, or
                forbidden-live) each fetch is sent through.
            config: The pinned content-type/size budget.
        """
        self._transport = transport
        self._config = config

    def fetch(self, url: str) -> str:
        """Fetch ``url`` and return its body, validating status/type/size.

        The checks run in a fixed order -- status, then content type, then body
        size -- so a non-2xx status is reported before the content type is ever
        inspected, and an off-allowlist content type before the body size.

        Args:
            url: The URL to fetch.

        Returns:
            The response body verbatim, for a clean 2xx, allowlisted-media-type,
            within-budget response.

        Raises:
            UnreachableUrlError: If the response status is not 2xx.
            ContentTypeRejectedError: If the response media type is off the
                allowlist.
            BodyTooLargeError: If the response body exceeds ``max_body_bytes``.
        """
        request = HttpRequest(method=_FETCH_METHOD, url=url, body=_EMPTY_BODY)
        response = self._transport.send(request)
        if not (
            _HTTP_SUCCESS_MIN <= response.status_code < _HTTP_SUCCESS_MAX_EXCLUSIVE
        ):
            msg = f"fetch returned non-2xx status {response.status_code} for {url!r}"
            raise UnreachableUrlError(msg)
        media_type = _media_type(response.content_type)
        if media_type not in self._config.allowed_content_types:
            msg = f"fetch content type {media_type!r} is not allowlisted for {url!r}"
            raise ContentTypeRejectedError(msg)
        if len(response.body.encode("utf-8")) > self._config.max_body_bytes:
            msg = f"fetch body exceeds {self._config.max_body_bytes} bytes for {url!r}"
            raise BodyTooLargeError(msg)
        return response.body
