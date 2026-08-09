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
not an object (a *list of votes*, a bare `null` or number, a *list of the key
names*) is rejected as malformed rather than indexed into.

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


def test_void_tag_neither_emits_text_nor_opens_a_frame() -> None:
    """A void element contributes no text of its own and leaves no frame behind.

    `x<br>y` pins the output half: a void tag emits nothing between its
    neighbours, not a space and not a newline. `<img aria-hidden="true">` pins
    the stack half, and is the half that discriminates -- a void element has no
    end tag, so a frame pushed for it could never be popped. Remove the skip
    and the decorative image's `aria-hidden` frame stays open for the rest of
    the document, dropping every later data run (`"before"`, not
    `"beforeafter"`).
    """
    assert sanitize_content("x<br>y") == "xy"
    assert sanitize_content('before<img aria-hidden="true">after') == "beforeafter"


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


def test_stray_end_tag_is_ignored_and_pops_nothing() -> None:
    """A stray end tag is tolerated, and tolerating it means popping *nothing*.

    Two shapes for the same arm. At top level (`a</div>b`) the stack is empty,
    so the search must fall straight through without an `IndexError` and the
    surrounding text must survive. Inside an open `<template>` the *pops
    nothing* half becomes observable: a parser that cleared the stack whenever
    the search matched nothing would end the suppressed subtree at the forged
    `</em>` and emit `SECRET`.
    """
    assert sanitize_content("a</div>b") == "ab"
    assert sanitize_content("<template>a</em>SECRET</template>after") == "after"


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


def test_visible_nesting_control_for_the_innermost_match_test() -> None:
    """Control for `test_end_tag_matches_the_innermost_frame_not_the_outermost`;
    it verifies nothing about `handle_endtag` on its own.

    Every frame in this input is visible, so no mutation of `handle_endtag`
    changes the result -- making the whole method a no-op still yields
    `"abcd"`. Its only job is to rule out the alternative reading of the
    sibling's near-empty output, that a nested close simply drops everything
    after it. Read the pair together or not at all.
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
    forgery = DATA_BLOCK_BEGIN.replace("<", "&lt;")
    page = f'<div style="display:none">{forgery}</div>visible'

    assert sanitize_content(page) == "visible"


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
        '["probability_ppm", "rationale_summary", "abstain"]',
    ],
    ids=["list-of-votes", "null", "bare-string", "bare-number", "list-of-key-names"],
)
def test_non_object_json_response_is_malformed(response: str) -> None:
    """Valid JSON that is not an object fails closed with the malformed code.

    The `isinstance` guard matters because every check below it assumes a
    mapping. Four of these shapes prove that. `list-of-votes`, `null` and
    `bare-number` all make the required-keys `issubset` raise (an unhashable
    `dict` element, and two non-iterables). `list-of-key-names` is the sharpest:
    it *satisfies* both key checks -- the required names are all present and
    nothing is extra -- and then indexes a list by string, so without the guard
    it raises instead of returning a code.

    `bare-string` is breadth, not a witness for the guard: a `str` is an
    iterable of one-character strings, so `issubset` independently answers
    "no" and reaches the same return whether the guard is there or not. It is
    kept because a bare JSON scalar is a shape a model really does emit, not
    because it discriminates.
    """
    assert validate_vote_response(response) == RESPONSE_FAILURE_MALFORMED_VOTE_JSON
