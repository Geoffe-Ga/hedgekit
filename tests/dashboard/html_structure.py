"""Structural HTML assertions for the dashboard view tests (issue #275).

A substring assertion (``assert "MKT-DEEP" in html``) is exactly what let the
orphan-``<tr>`` defect ship: cell *text* appears in the output whether or not a
browser can build a table out of the surrounding markup, so the assertion is
blind to the only thing that was broken. These helpers parse the emitted markup
into an element tree so a test can assert *where* a cell sits, not merely that
its text occurs somewhere in the string.

The parser here deliberately builds the tree the author *wrote* -- literal
nesting, no HTML5 foster-parenting, no implied ``<tbody>`` insertion. That is the
point: a real browser silently relocates an orphan ``<tr>`` out of its parent and
renders the cells as run-together text, which would make a spec-faithful parser
report a plausible-looking tree for markup that is in fact broken. Refusing to
repair the markup is what makes :func:`assert_data_table` fail closed on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Elements that never have an end tag, so they must not open a nesting scope.
#: Only the ones the dashboard actually emits (``<meta>`` in the page skeleton)
#: plus the common HTML void elements, so a stray one can never swallow the rest
#: of the document into itself.
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta"}
)


@dataclass
class Element:
    """One parsed HTML element and the subtree written inside it.

    Attributes:
        tag: The lowercased tag name; ``"#document"`` for the synthetic root.
        children: Child elements, in document order.
        text: Character data written directly inside this element (not inside a
            descendant). Character references are already decoded, so a
            correctly escaped ``&lt;script&gt;`` arrives here as text rather
            than as a child element -- which is how escaping is distinguishable
            from injection structurally, not by substring.
    """

    tag: str
    children: list[Element] = field(default_factory=list)
    text: str = ""


class _TreeBuilder(HTMLParser):
    """Build an :class:`Element` tree mirroring the literal markup nesting."""

    def __init__(self) -> None:
        """Start with an empty synthetic root as the open-element stack base."""
        super().__init__(convert_charrefs=True)
        self.root = Element(tag="#document")
        self._stack: list[Element] = [self.root]

    def handle_starttag(self, tag: str, attrs: object) -> None:
        """Append a child for ``tag`` and descend into it unless it is void.

        Args:
            tag: The lowercased start-tag name.
            attrs: The parsed attributes; unused -- these assertions are about
                structure, not presentation.
        """
        del attrs
        node = Element(tag=tag)
        self._stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: object) -> None:
        """Append a self-closing ``<tag/>`` without descending into it.

        Args:
            tag: The lowercased tag name.
            attrs: The parsed attributes; unused.
        """
        del attrs
        self._stack[-1].children.append(Element(tag=tag))

    def handle_endtag(self, tag: str) -> None:
        """Close the innermost open ``tag``, discarding anything still open.

        An end tag with no matching open element is ignored rather than raising,
        so a malformed fragment still yields a tree the assertions can describe.

        Args:
            tag: The lowercased end-tag name.
        """
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        """Attach character data to the innermost open element.

        Args:
            data: The decoded character data.
        """
        self._stack[-1].text += data


def parse_fragment(markup: str) -> Element:
    """Parse an HTML fragment or full page into its literal element tree.

    Args:
        markup: The rendered HTML to parse.

    Returns:
        The synthetic ``#document`` root whose children are ``markup``'s
        top-level elements.
    """
    builder = _TreeBuilder()
    builder.feed(markup)
    builder.close()
    return builder.root


def find_all(element: Element, tag: str) -> list[Element]:
    """Collect every descendant of ``element`` with tag name ``tag``.

    Args:
        element: The subtree root to search (excluded from the result itself).
        tag: The lowercased tag name to match.

    Returns:
        The matching descendants in document order.
    """
    found: list[Element] = []
    for child in element.children:
        if child.tag == tag:
            found.append(child)
        found.extend(find_all(child, tag))
    return found


def text_of(element: Element) -> str:
    """Return ``element``'s whitespace-stripped text, descendants included.

    Args:
        element: The element whose rendered text is wanted.

    Returns:
        The concatenated character data of ``element`` and its descendants,
        stripped of surrounding whitespace.
    """
    parts = [element.text] + [text_of(child) for child in element.children]
    return "".join(parts).strip()


def assert_data_table(
    markup: str,
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    """Assert ``markup`` contains exactly one well-formed data table.

    Checks the three things a substring assertion cannot: that a ``<table>``
    exists at all, that no row or cell is stranded outside it (the orphan-``<tr>``
    defect of issue #275), and that the header row labels the columns the data
    rows actually carry.

    Args:
        markup: The rendered HTML fragment or page to inspect.
        headers: The expected ``<thead>`` column labels, in order.
        rows: The expected ``<tbody>`` rows, each a sequence of cell texts in
            order.

    Raises:
        AssertionError: If no single ``<table>`` wraps every row and cell, or if
            the header or body cell texts differ from the expectation.
    """
    root = parse_fragment(markup)
    tables = find_all(root, "table")
    assert len(tables) == 1, f"expected exactly one <table>, found {len(tables)}"
    table = tables[0]

    for tag in ("tr", "th", "td"):
        inside = {id(element) for element in find_all(table, tag)}
        stranded = [e for e in find_all(root, tag) if id(e) not in inside]
        assert not stranded, f"{len(stranded)} <{tag}> element(s) outside the <table>"

    heads = find_all(table, "thead")
    assert len(heads) == 1, f"expected exactly one <thead>, found {len(heads)}"
    assert [text_of(th) for th in find_all(heads[0], "th")] == list(headers)

    bodies = find_all(table, "tbody")
    assert len(bodies) == 1, f"expected exactly one <tbody>, found {len(bodies)}"
    rendered = [
        [text_of(cell) for cell in find_all(row, "td")]
        for row in find_all(bodies[0], "tr")
    ]
    assert rendered == [list(row) for row in rows]
