"""`windbreak run`'s live-market-data resolution (issue #343, RED).

The composition root's other half. `windbreak/scheduler/loop.py` learns to
build a live-book session; this module pins how `windbreak.main` decides
*whether* to hand it one, from a single new flag.

There is one flag, `--paper-live-ticker`, and not the two PR #384 sketched
(`--paper-live-books` plus a ticker). Two flags would admit two invalid
combinations -- live books with no market named, and a market named that
nothing reads -- each of which then has to be diagnosed at runtime. One flag
whose *presence* selects live mode and whose *value* names the market admits
none: the market and the mode cannot disagree because they are the same
token.

Nothing here reaches the network: `build_kalshi_market_data` resolves a base
URL, screens it against the deployment's own allowlist, and constructs a
client, all before any request is made, and no venue read happens until the
scheduler builds a session (SPEC S17.1: CI is offline).

RED: `windbreak.main` has no `_resolve_paper_market_data`, so every test below
fails with `ImportError`/`AttributeError`.
"""

from __future__ import annotations

import argparse

import pytest

from windbreak.config.schema import ExchangeConfig, WindbreakConfig
from windbreak.connector.live import MarketDataOnlyView
from windbreak.main import _resolve_paper_market_data


def _args(live_ticker: str | None) -> argparse.Namespace:
    """Build the parsed-arguments stand-in the resolver reads.

    Args:
        live_ticker: The value of ``--paper-live-ticker``, or ``None`` when the
            flag was omitted.

    Returns:
        A namespace carrying just the flag under test.
    """
    return argparse.Namespace(paper_live_ticker=live_ticker)


def test_omitting_the_flag_resolves_no_market_data() -> None:
    """No flag, no venue: the loop stays on its fixture path, unchanged."""
    assert _resolve_paper_market_data(_args(None), WindbreakConfig()) is None


def test_the_flag_resolves_a_credential_free_market_data_view() -> None:
    """The flag builds the narrowed venue view, which holds no write surface.

    SPEC S1.1 invariant 3, structurally: the object `windbreak run` hands the
    PAPER loop has no `place_order` attribute to reach for and no account read
    that could be mistaken for the simulator's own.
    """
    market_data = _resolve_paper_market_data(_args("KXFED-24DEC"), WindbreakConfig())

    assert isinstance(market_data, MarketDataOnlyView)
    assert not hasattr(market_data, "place_order")
    assert not hasattr(market_data, "get_balances")


def test_an_unrecognized_exchange_provider_refuses_to_build() -> None:
    """A non-Kalshi deployment refuses rather than silently dialing Kalshi.

    `allowlist_from_config` contributes no host for an unrecognized provider
    (fail closed), so the Kalshi base URL this factory would dial is off the
    deployment's own allowlist and construction is refused before any session
    exists.
    """
    config = WindbreakConfig(exchange=ExchangeConfig(provider="somewhere-else"))

    with pytest.raises(ValueError, match="allowlist"):
        _resolve_paper_market_data(_args("KXFED-24DEC"), config)


def test_an_unrecognized_environment_refuses_to_build() -> None:
    """A misspelled environment fails closed instead of defaulting to production."""
    config = WindbreakConfig(exchange=ExchangeConfig(environment="staging"))

    with pytest.raises(ValueError, match="unknown exchange environment"):
        _resolve_paper_market_data(_args("KXFED-24DEC"), config)
