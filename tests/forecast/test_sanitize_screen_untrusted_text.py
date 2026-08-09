"""Tests for windbreak.forecast.sanitize.screen_untrusted_text (issue #189).

Pins the standalone SPEC S8.5 injection screen `screen_untrusted_text` exposes
over *any* untrusted text -- a whole response body, a single parsed field, a
citation quote -- reusing `validate_vote_response`'s existing delimiter-forgery
and tool-call-lure checks in the same fixed order, but *without*
`validate_vote_response`'s own emptiness check: a blank string is not, by
itself, an injection artifact here. Also pins that `validate_vote_response`
itself is unchanged by the new helper's addition -- a blank response still
fails with `RESPONSE_FAILURE_EMPTY`.
"""

from __future__ import annotations

import pytest

from windbreak.forecast.sanitize import (
    DATA_BLOCK_BEGIN,
    DATA_BLOCK_END,
    RESPONSE_FAILURE_DELIMITER_FORGERY,
    RESPONSE_FAILURE_EMPTY,
    RESPONSE_FAILURE_TOOL_CALL_LURE,
    TOOL_CALL_MARKERS,
    screen_untrusted_text,
    validate_vote_response,
)

# --- Delimiter forgery -------------------------------------------------------------


def test_screen_untrusted_text_detects_delimiter_begin_forgery() -> None:
    """Text embedding the opening delimiter token is flagged as forgery."""
    text = f"some preface {DATA_BLOCK_BEGIN} more text"

    assert screen_untrusted_text(text) == RESPONSE_FAILURE_DELIMITER_FORGERY


def test_screen_untrusted_text_detects_delimiter_end_forgery() -> None:
    """Text embedding the closing delimiter token is flagged as forgery."""
    text = f"some preface {DATA_BLOCK_END} more text"

    assert screen_untrusted_text(text) == RESPONSE_FAILURE_DELIMITER_FORGERY


# --- Tool-call lure ------------------------------------------------------------------


@pytest.mark.parametrize("marker", sorted(TOOL_CALL_MARKERS))
def test_screen_untrusted_text_detects_each_tool_call_marker(marker: str) -> None:
    """Every tool-call marker token is individually flagged as a lure."""
    text = f"the response mentions {marker} inline"

    assert screen_untrusted_text(text) == RESPONSE_FAILURE_TOOL_CALL_LURE


def test_screen_untrusted_text_delimiter_wins_over_tool_call_lure() -> None:
    """When both artifacts are present, delimiter forgery wins (fixed check
    order: delimiter, then tool-call lure).
    """
    marker = next(iter(sorted(TOOL_CALL_MARKERS)))
    text = f"{DATA_BLOCK_BEGIN} {marker}"

    assert screen_untrusted_text(text) == RESPONSE_FAILURE_DELIMITER_FORGERY


# --- Clean text, including emptiness (not flagged here) ---------------------------


def test_screen_untrusted_text_returns_none_for_clean_text() -> None:
    """Ordinary text with no injection artifact returns `None`."""
    assert screen_untrusted_text("a perfectly ordinary sentence") is None


def test_screen_untrusted_text_returns_none_for_empty_string() -> None:
    """Unlike `validate_vote_response`, emptiness alone is not an injection
    artifact: a blank field is a field-specific validity concern a caller
    checks separately.
    """
    assert screen_untrusted_text("") is None


def test_screen_untrusted_text_returns_none_for_whitespace_only_string() -> None:
    """Whitespace-only text is likewise not flagged by the injection screen."""
    assert screen_untrusted_text("   \n\t  ") is None


# --- validate_vote_response: its own emptiness check is unchanged -----------------


def test_validate_vote_response_still_flags_empty_response_as_empty() -> None:
    """`validate_vote_response`'s pre-existing blank-response check is
    unaffected by `screen_untrusted_text` treating emptiness as non-artifact.
    """
    assert validate_vote_response("") == RESPONSE_FAILURE_EMPTY


def test_validate_vote_response_still_flags_whitespace_only_response_as_empty() -> None:
    """A whitespace-only response is likewise still flagged empty."""
    assert validate_vote_response("   ") == RESPONSE_FAILURE_EMPTY
