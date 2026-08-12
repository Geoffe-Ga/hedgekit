"""Every ledger row carrying a market ticker is built at a guarded site (#530).

Issue #530's guards are two calls to
:func:`windbreak.forecast.pipeline.ledger_safe_ticker`, each placed at the sole
production site that constructs a ticker-carrying typed ledger event. That is
coverage by *convention* the moment a third construction site appears: the new
site would compile, ship, and quietly append attacker-chosen text to an
append-only hash chain, and every existing test would stay green -- exactly the
"deleting a wiring call left 5780 tests green" failure this repo has already
been bitten by.

The registry below turns that convention into an assertion. It is **derived**:
the sites are found by walking the shipped package's AST for calls to the event
classes by name, not by reading a list someone remembered to update. A fourth
site, or a site that stops calling the guard, fails here.

WHAT THIS SCANNER IS, AND IS NOT

It checks two things about each construction site and deliberately nothing else:

* **Where it is** -- the enclosing module and function must be a registered
  entry, so a new site anywhere in ``windbreak`` is reported.
* **That the guard is applied to the ticker** -- the ``ticker`` keyword
  argument's expression must contain a call to ``ledger_safe_ticker``. A site
  that computes the safe value and then passes the raw one (a guard whose result
  is dropped, which is inert while looking correct) fails.

It cannot check that the guard is *correct*; that is what
``tests/forecast/test_market_metadata_seam.py`` and
``tests/integration/test_paper_hostile_ticker.py`` are for. It is a structural
backstop under those behavioural tests, not a replacement for them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: The shipped package this scanner walks.
_PACKAGE = Path(__file__).resolve().parents[2] / "windbreak"

#: The guard every ticker bound for one of these rows must cross.
_GUARD = "ledger_safe_ticker"

#: The typed ledger events whose payload carries a raw market ticker under a
#: ``ticker`` keyword argument, mapped to the single ``module::function``
#: production site allowed to construct each.
#:
#: ``ForecastCreated`` is deliberately absent: its ``market_ticker`` cannot be
#: unscreened at all, because ``ForecastRecord.__post_init__`` refuses to
#: construct a record carrying one (issue #525). An invariant on the record type
#: needs no site registry; these two events are ledger rows on the tick's
#: unconditional path, where refusing would crash the tick, so they are guarded
#: at construction instead and that is what this registry pins.
_REGISTERED: dict[str, str] = {
    "ScreenDecisionRecorded": "windbreak/scheduler/screening.py::record",
    "MarketSnapshotRecorded": (
        "windbreak/scheduler/loop.py::market_snapshot_event_to_record"
    ),
}


def _construction_sites(event_name: str) -> list[tuple[str, ast.Call]]:
    """Find every call to ``event_name`` in the shipped package.

    Args:
        event_name: The event class whose construction sites are wanted.

    Returns:
        One ``("<relative path>::<enclosing function>", call)`` pair per site,
        sorted by site label.
    """
    found: list[tuple[str, ast.Call]] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == event_name
                ):
                    relative = path.relative_to(_PACKAGE.parent).as_posix()
                    found.append((f"{relative}::{node.name}", inner))
    return sorted(found, key=lambda entry: entry[0])


@pytest.mark.parametrize("event_name", sorted(_REGISTERED))
def test_the_event_is_constructed_only_at_its_registered_site(
    event_name: str,
) -> None:
    """No unregistered production code builds a ticker-carrying ledger row.

    Asserted non-empty first: a scanner that found nothing would otherwise
    report "no unregistered sites" forever, including on the day someone renames
    the class and the sweep stops matching anything at all.
    """
    sites = _construction_sites(event_name)

    assert len(sites) == 1
    assert [label for label, _call in sites] == [_REGISTERED[event_name]]


@pytest.mark.parametrize("event_name", sorted(_REGISTERED))
def test_the_registered_site_puts_its_ticker_through_the_guard(
    event_name: str,
) -> None:
    """The ``ticker`` argument's expression really calls the guard.

    Not "the guard is called somewhere in the function": the guarded value must
    be the one handed to ``ticker``, so a site that calls
    ``ledger_safe_ticker(...)`` and then passes the raw ticker anyway -- an
    inert guard, which reads as correct and ledgers the bytes -- fails here.
    """
    ((_label, call),) = _construction_sites(event_name)

    ticker_arguments = [
        keyword.value for keyword in call.keywords if keyword.arg == "ticker"
    ]

    assert len(ticker_arguments) == 1
    guard_calls = [
        node
        for node in ast.walk(ticker_arguments[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _GUARD
    ]
    assert len(guard_calls) == 1
