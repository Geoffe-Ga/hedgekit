"""Tests for the visible-text parser in windbreak.forecast.sanitize (issue #163).

`sanitize_content` is the SPEC S8.5 response-side firewall between attacker-
controlled fetched bytes and an LLM prompt, so every unexercised arm of its
`_VisibleTextParser` is an *unknown behavior on hostile input*. This module pins
the arms the coverage scan flagged, each by the byte-level effect it has on the
extracted visible text rather than by "the parser ran":

* `_style_hides`'s `value is None` guard -- a valueless `style` attribute
  (`<div style>`), which HTMLParser reports as `("style", None)`.
* `_attrs_hidden`'s loop *continuing* past a non-hiding `style`, so a decoy
  `style="display:block"` cannot mask a later `hidden` attribute on the same tag.
* `handle_starttag`'s void-element skip, so a void tag can never leave an
  unpoppable frame on the stack (and `<br hidden>` cannot blank the rest of the
  page).
* `handle_endtag`'s tag-soup arms: a stray end tag matching *nothing* is
  ignored, and the search walks *past* non-matching frames innermost-first, so
  a forged or mismatched end tag cannot pop an enclosing hidden/suppressed
  subtree back into view.
* `handle_entityref` / `handle_charref` while hidden: an entity or numeric
  charref inside a hidden or suppressed subtree is *dropped*, never re-emitted --
  which is what keeps an entity-encoded delimiter forgery buried in a
  `display:none` div out of the quote entirely.

Also pins `_schema_failure`'s non-dict arm: a JSON document that parses but is
not an object (a *list of votes*, `null`, a bare string) is rejected as
malformed rather than indexed into.

Assertions are exact-equality on the whole sanitized string: an `in`/`not in`
check would pass for a parser that dropped or leaked neighbouring text too.
"""

from __future__ import annotations

import pytest

from windbreak.forecast.sanitize import (
    DATA_BLOCK_BEGIN,
    RESPONSE_FAILURE_MALFORMED_VOTE_JSON,
    sanitize_content,
    validate_vote_response,
)

# --- `_style_hides`: the valueless-`style` guard -------------------------------------


def test_valueless_style_attribute_does_not_hide_its_element() -> None:
    """A bare `style` attribute (no `=value`) leaves its element visible.

    HTMLParser reports `<div style>` as `("style", None)`; `_style_hides` must
    return `False` for that `None` rather than lowercasing it (an unguarded
    `value.lower()` raises `AttributeError` and takes the whole fetch down).
    """
    assert sanitize_content("<div style>keep-this</div>") == "keep-this"


def test_empty_style_attribute_does_not_hide_its_element() -> None:
    """A present-but-empty `style=""` is a distinct input from a valueless one
    and likewise leaves its element visible.
    """
    assert sanitize_content('<div style="">keep-this</div>') == "keep-this"


@pytest.mark.parametrize(
    "style_value",
    ["DISPLAY: NONE", "Visibility: Hidden", "color:red;  font-size: 0"],
)
def test_hiding_style_is_matched_case_and_space_insensitively(
    style_value: str,
) -> None:
    """Uppercase and space-padded hiding styles still hide.

    An attacker who writes `style="DISPLAY: NONE"` must not evade the matcher;
    `_style_hides` lowercases *and* strips spaces before testing its tokens.
    """
    page = f'before <div style="{style_value}">SECRET</div> after'

    assert sanitize_content(page) == "before after"


def test_non_hiding_display_style_leaves_its_element_visible() -> None:
    """`display:block` is a near miss for `display:none` and must not hide.

    Pins that the token test is a real substring test over the hidden-token set
    rather than a `display:`-prefix test that would swallow every layout style.
    """
    page = 'before <div style="display:block">keep-this</div> after'

    assert sanitize_content(page) == "before keep-this after"


# --- `_attrs_hidden`: continuing past a non-hiding attribute -------------------------


def test_decoy_style_before_hidden_attribute_still_hides_the_subtree() -> None:
    """A non-hiding `style` must not short-circuit the attribute scan.

    `<div style="display:block" hidden>` is the evasion this arm defends: if
    `_attrs_hidden` returned `_style_hides(value)` instead of *continuing* the
    loop, the decoy style would answer `False` for the whole element and the
    later `hidden` attribute would never be reached -- leaking the payload.
    """
    page = 'before <div style="display:block" hidden>SECRET</div> after'

    assert sanitize_content(page) == "before after"


def test_decoy_style_before_aria_hidden_still_hides_the_subtree() -> None:
    """The same evasion via `aria-hidden` rather than the bare `hidden`
    attribute is likewise defeated by continuing the attribute scan.
    """
    page = 'before <div style="color:red" aria-hidden="true">SECRET</div> after'

    assert sanitize_content(page) == "before after"


def test_aria_hidden_false_leaves_its_element_visible() -> None:
    """`aria-hidden="false"` is the near miss for `aria-hidden="true"` and must
    not hide, pinning that the value is compared rather than the name alone.
    """
    page = 'before <span aria-hidden="false">keep-this</span> after'

    assert sanitize_content(page) == "before keep-this after"


# --- `handle_starttag`: the void-element skip ----------------------------------------


def test_void_tag_does_not_push_a_suppressible_frame() -> None:
    """A void element never opens a frame, so text after it survives.

    A `<br>` has no end tag; had it been pushed, its frame would stay on the
    stack for the rest of the document.
    """
    assert sanitize_content("x<br>y") == "xy"


def test_hidden_attribute_on_a_void_tag_cannot_blank_the_rest_of_the_page() -> None:
    """`<br hidden>` must not suppress everything that follows it.

    A void element carries no text subtree, so its own `hidden` attribute has
    nothing to hide. If the void skip were removed, the pushed hidden frame
    would never be popped and every later data run would be dropped.
    """
    assert sanitize_content("before<br hidden>after") == "beforeafter"


def test_void_tag_inside_a_hidden_subtree_leaves_the_subtree_hidden() -> None:
    """A void tag nested in a hidden element does not disturb the stack, so the
    enclosing element's suppression still ends exactly at its own end tag.
    """
    page = "<div hidden>SECRET<br>ALSO-SECRET</div>after"

    assert sanitize_content(page) == "after"


# --- `handle_endtag`: the tag-soup arms ----------------------------------------------


def test_stray_end_tag_with_no_open_element_is_ignored() -> None:
    """An unmatched end tag is tolerated and the surrounding text is kept."""
    assert sanitize_content("a</div>b") == "ab"


def test_stray_end_tag_cannot_reveal_an_enclosing_hidden_subtree() -> None:
    """A forged end tag matching nothing on the stack must not pop the hidden
    frame that encloses it.

    This is the injection shape: the payload emits `</span>` from inside a
    `display:none` div hoping the parser unwinds to a visible state and starts
    collecting again. The whole hidden subtree must stay dropped.
    """
    page = '<div style="display:none">SECRET</span>ALSO-SECRET</div>after'

    assert sanitize_content(page) == "after"


def test_end_tag_search_walks_past_a_non_matching_inner_frame() -> None:
    """`</div>` closing over an unclosed inner `<span>` unwinds both frames.

    The search must *continue* past the non-matching innermost frame; a parser
    that only inspected the top of the stack would find no match, leave the
    hidden `<div>` open forever, and swallow every later data run.
    """
    page = "<div hidden><span>SECRET</div>after"

    assert sanitize_content(page) == "after"


def test_end_tag_discards_frames_still_open_inside_the_matched_element() -> None:
    """Closing an element also pops every frame left open inside it.

    An unclosed hidden `<span>` must not outlive its parent's `</div>`: a
    parser that popped only the matched frame would leave the hidden span on
    the stack and swallow the entire rest of the document. This pins the
    `del self._stack[index:]` *slice* on a hidden inner frame, complementing
    the suppressed-tag case already covered for `<script>`.
    """
    page = "<div><span hidden>SECRET</div>after"

    assert sanitize_content(page) == "after"


def test_end_tag_matches_the_innermost_frame_not_the_outermost() -> None:
    """With two frames of the same tag, an end tag closes the *inner* one.

    An outermost-first search would pop the hidden outer `<div>` on the inner
    `</div>` and leak `LEAKED`. The three markers are distinct strings so an
    innermost match, an outermost match, and a no-op each produce a different
    result rather than coinciding.
    """
    page = "<div hidden>SECRET<div>NESTED</div>LEAKED</div>visible"

    assert sanitize_content(page) == "visible"


def test_end_tag_inside_a_visible_element_pops_only_that_element() -> None:
    """The innermost-match rule on *visible* nesting keeps later text visible,
    so the previous test's empty-ish outcome is not merely "everything after a
    nested close is dropped".
    """
    page = "<div>a<div>b</div>c</div>d"

    assert sanitize_content(page) == "abcd"


# --- `handle_entityref` / `handle_charref` while hidden -------------------------------


def test_named_entity_inside_a_hidden_element_is_dropped() -> None:
    """A named entity in a hidden subtree is not re-emitted."""
    assert sanitize_content("<span hidden>keep&amp;out</span>visible") == "visible"


def test_named_entity_inside_a_suppressed_script_is_dropped() -> None:
    """A named entity inside `<script>` is dropped along with its subtree."""
    assert sanitize_content("<script>a&amp;b</script>visible") == "visible"


def test_entity_encoded_delimiter_forgery_inside_a_hidden_element_is_dropped() -> None:
    """An entity-encoded delimiter forgery buried in a `display:none` div never
    reaches the quote at all.

    Dropping beats neutralizing here: the forgery is removed before
    `_neutralize_delimiters` ever sees it, and nothing of it survives.
    """
    page = '<div style="display:none">&lt;&lt;&lt;UNTRUSTED-DATA</div>visible'

    result = sanitize_content(page)

    assert result == "visible"
    assert DATA_BLOCK_BEGIN not in result


def test_numeric_charref_inside_a_hidden_element_is_dropped() -> None:
    """A decimal numeric character reference in a hidden subtree is dropped."""
    page = '<div style="display:none">&#8217;</div>visible'

    assert sanitize_content(page) == "visible"


def test_hex_charref_inside_a_suppressed_subtree_is_dropped() -> None:
    """A hexadecimal character reference inside `<style>` is dropped too."""
    assert sanitize_content("<style>&#x2019;</style>visible") == "visible"


def test_entity_and_charref_outside_a_hidden_element_survive_verbatim() -> None:
    """The visible counterpart of the two drops above: outside a hidden
    subtree both reference forms are re-emitted byte-for-byte, so the raw-hash
    contract still holds. Without this, "always drop" would pass equally.
    """
    page = "<span hidden>SECRET</span>S&amp;P&#8217;s"

    assert sanitize_content(page) == "S&amp;P&#8217;s"


# --- `_schema_failure`: a JSON document that is not an object -------------------------


@pytest.mark.parametrize(
    "response",
    [
        '[{"probability_ppm": 1, "rationale_summary": "r", "abstain": false}]',
        "null",
        '"probability_ppm"',
        "42",
    ],
    ids=["list-of-votes", "null", "bare-string", "bare-number"],
)
def test_non_object_json_response_is_malformed(response: str) -> None:
    """Valid JSON that is not an object is rejected before any key lookup.

    A *list* wrapping an otherwise-valid vote is the interesting one: the
    payload is schema-shaped but the container is not, and indexing it would
    raise rather than fail closed with a code.
    """
    assert validate_vote_response(response) == RESPONSE_FAILURE_MALFORMED_VOTE_JSON
