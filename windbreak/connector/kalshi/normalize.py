"""Pure normalization of raw Kalshi payloads into windbreak's SPEC S6.2 models.

Every function here is a total, side-effect-free transform from a raw Kalshi
JSON mapping into a normalized model from :mod:`windbreak.connector.models`. All
prices and sizes convert to windbreak's scaled-integer unit types
(:class:`~windbreak.numeric.PricePips`, :class:`~windbreak.numeric.ContractCentis`)
with zero float intermediaries: this module sits on the money path guarded by
``scripts/lint_no_floats.py`` (SPEC S6.1), so only ``*``, ``+``, ``-`` and
``//`` integer arithmetic appears -- never ``/`` or ``float(...)``. Every
cents-to-pips and contract-to-centis conversion here is an exact multiplication
by :data:`PRICE_PIPS_PER_CENT` / :data:`CENTIS_PER_CONTRACT` (1 cent = 100
pips, 1 contract = 100 centis): none loses precision, so SPEC S6.1's
conservative-rounding direction never has to be chosen in this module. The 24h
volume likewise converts by exact multiplication -- traded contract count times
:data:`MICROS_PER_CONTRACT_NOTIONAL` -- into settlement-notional micro-dollars,
a windbreak extension to the literal SPEC S6.2 market schema.

The product gate is an *allowlist*: :func:`gate_product` normalizes only
``"binary"`` markets, refusing every other (or absent) ``market_type`` so a
newly-invented Kalshi product -- including any margin, perp, or other
derivative surface Kalshi might expose -- is refused by default rather than
silently mis-normalized (SPEC S1.1 invariant 2; SPEC S7.1).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

from windbreak.connector.models import (
    ExchangeStatus,
    NormalizedMarket,
    OrderBookLevel,
    OrderBookSnapshot,
)
from windbreak.ledger import canonical_json
from windbreak.numeric import ContractCentis, PricePips

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

#: Pips per US cent: a one-cent price increment is exactly 100 pips.
PRICE_PIPS_PER_CENT: Final = 100

#: Contract-centis per whole contract: one contract is exactly 100 centis.
CENTIS_PER_CONTRACT: Final = 100

#: Default price tick, in pips, when a market omits ``tick_size`` (a 1c tick).
DEFAULT_PRICE_TICK_PIPS: Final = 100

#: Minimum order size, in contract-centis, applied to every Kalshi market.
MIN_ORDER_CONTRACT_CENTIS: Final = 100

#: Micro-dollars of settlement notional per traded contract. A fully
#: collateralized binary settles at exactly $1 = 1,000,000 micros, so a traded
#: contract count times this notional is an exact, price-independent 24h volume
#: in micro-dollars -- no float and no price ever enters the conversion.
MICROS_PER_CONTRACT_NOTIONAL: Final = 1_000_000

#: The full YES+NO price of a binary contract, in cents (a resolved dollar):
#: a NO price of ``c`` cents implies a YES price of ``100 - c`` cents.
_FULL_PRICE_CENTS: Final = 100

#: The exchange identifier stamped on every market this connector normalizes.
KALSHI_EXCHANGE: Final = "kalshi"

#: The only raw Kalshi ``market_type`` the allowlist admits.
_BINARY: Final = "binary"

#: The normalized market-type Kalshi binaries map to (SPEC S6.2).
_NORMALIZED_MARKET_TYPE: Final = "fully_collateralized_binary"

#: Kalshi exposes no eligibility signal, so jurisdiction is always unknown
#: (SPEC S20 Q3) -- never inferred as ``"eligible"``.
_JURISDICTION_UNKNOWN: Final = "unknown"

#: Ledger event type recorded when a non-binary product is refused.
PRODUCT_REFUSED_EVENT: Final = "PRODUCT_REFUSED"

#: Ledger event type recorded when an allowed binary market fails to normalize
#: (a required field is missing or a leaf has the wrong type). Emitting this and
#: skipping the market keeps one malformed payload from aborting a whole scan,
#: while still never silently dropping it (SPEC S1.1 invariant 2).
MARKET_MALFORMED_EVENT: Final = "MARKET_MALFORMED"

#: The three narrowed exchange-status literals (fake.py ``_STATUS_BY_NAME`` idiom).
_STATUS_OPEN: Final = "open"
_STATUS_PAUSED: Final = "paused"
_STATUS_CLOSED: Final = "closed"


def _parse_dt(value: str, *, field: str) -> datetime:
    """Parse an offset-bearing ISO-8601 timestamp into a UTC datetime.

    The offset is required, not defaulted. ``datetime.fromisoformat`` returns a
    *naive* datetime for an offsetless string, and ``.astimezone(UTC)`` does not
    raise on a naive value -- it reads the wall clock as the **host's** local
    time, so the same payload would yield different instants on a developer's
    laptop and in UTC-running CI (issue #392). For a market-lifecycle field such
    as ``close_time`` that misreading can shift the instant across a date
    boundary and make a closed market read as open.

    Assuming UTC instead would only trade a loud wrong answer for a quiet one:
    an exchange timestamp with no offset is unprovable evidence about when the
    market closes, and unprovable evidence must never read as healthy (SPEC S1.1
    invariant 2). So this refuses, and the caller vetoes: ``normalize_market``'s
    ``ValueError`` degrades to a ledgered ``MARKET_MALFORMED`` event in
    :meth:`~windbreak.connector.kalshi.adapter.KalshiConnector.list_markets`,
    skipping that one market rather than aborting the scan.

    Args:
        value: An ISO-8601 string carrying a UTC offset, e.g.
            ``2024-12-18T19:00:00Z`` or ``2024-12-18T14:00:00-05:00``.
        field: The raw Kalshi field name ``value`` came from, named in the error
            so an operator reading the ledgered event can locate it.

    Returns:
        The timezone-aware datetime, normalized to UTC.

    Raises:
        ValueError: If ``value`` is not parseable as ISO-8601, or parses to a
            naive (offsetless) datetime.
    """
    parsed = datetime.fromisoformat(value)
    # Naive means `tzinfo is None` *or* a `tzinfo` whose `utcoffset()` returns
    # None -- Python's own definition, which `utcoffset()` reports in one call
    # (issue #346's finding, applied here at parse time).
    if parsed.utcoffset() is None:
        raise ValueError(
            f"Kalshi timestamp field {field!r} carries no UTC offset: {value!r}. "
            "An offsetless exchange timestamp is unprovable evidence -- reading "
            "its wall clock as the host's local time would silently shift the "
            "instant by the host's offset -- so the market is refused rather "
            "than normalized against a guessed timezone."
        )
    return parsed.astimezone(UTC)


def _parse_optional_dt(value: str | None, *, field: str) -> datetime | None:
    """Parse an optional offset-bearing ISO-8601 timestamp, preserving ``None``.

    ``None`` means the venue stated no such time, which is a fact about the
    market rather than an unprovable instant, so it passes through untouched.
    A *present* value is held to :func:`_parse_dt`'s offset requirement.

    Args:
        value: An ISO-8601 string carrying a UTC offset, or None.
        field: The raw Kalshi field name ``value`` came from, named in the error.

    Returns:
        The parsed UTC datetime, or None when ``value`` is None.

    Raises:
        ValueError: If ``value`` is present and offsetless or unparseable.
    """
    return None if value is None else _parse_dt(value, field=field)


def payload_hash(raw: Mapping[str, object]) -> str:
    """Return a stable SHA-256 hex digest of a raw exchange payload.

    The digest is taken over the canonical (key-sorted, whitespace-free) JSON
    encoding, so it depends only on the payload's contents -- never on dict
    insertion order -- giving a reproducible provenance fingerprint.

    Args:
        raw: The raw exchange payload to fingerprint.

    Returns:
        The 64-character lowercase-hex SHA-256 digest.
    """
    return hashlib.sha256(canonical_json(dict(raw)).encode("utf-8")).hexdigest()


def gate_product(raw_market: Mapping[str, object]) -> str | None:
    """Return why a market is refused, or None when it is an allowed binary.

    This is an allowlist, not a denylist: only ``market_type == "binary"``
    passes. Any other type -- or a payload missing ``market_type`` entirely --
    is refused so a newly-invented product is never silently normalized.

    Args:
        raw_market: The raw Kalshi market payload to gate.

    Returns:
        None when the market is an allowed binary; otherwise a non-empty reason
        string naming the offending type (containing ``"missing"`` when the
        ``market_type`` key is absent).
    """
    if "market_type" not in raw_market:
        return "market_type is missing"
    market_type = raw_market["market_type"]
    if market_type == _BINARY:
        return None
    return f"refused non-binary market_type: {market_type!r}"


def _as_mapping(raw: Mapping[str, object]) -> Mapping[str, Any]:
    """Re-view a raw payload as a ``str``-keyed ``Any`` mapping.

    JSON payloads are dynamically typed, so this narrows a raw ``object``-valued
    mapping to one whose values read as ``Any`` -- letting field access stay
    ergonomic while a bad leaf still fails loudly at unit-wrapping time.

    Args:
        raw: The raw payload to re-view.

    Returns:
        The same mapping, typed for ergonomic field access.
    """
    return cast("Mapping[str, Any]", raw)


def _is_grouped(raw_event: Mapping[str, object] | None) -> bool:
    """Return whether a market's event marks it mutually exclusive.

    Feeds :attr:`NormalizedMarket.mutually_exclusive_group_id`, which downstream
    coherence checks use to sum probabilities across an event's outcomes
    (SPEC S8.7).

    Args:
        raw_event: The market's parent event payload, or None when standalone.

    Returns:
        True only when an event is supplied and its ``mutually_exclusive`` flag
        is truthy.
    """
    return raw_event is not None and bool(raw_event.get("mutually_exclusive"))


def _tick_pips(market: Mapping[str, Any]) -> int:
    """Return a market's price tick in pips, defaulting a missing tick to 1c.

    Args:
        market: The raw market payload.

    Returns:
        The price tick in pips: :data:`DEFAULT_PRICE_TICK_PIPS` when
        ``tick_size`` is absent, else the cents tick multiplied into pips.
    """
    raw_tick = market.get("tick_size")
    if raw_tick is None:
        return DEFAULT_PRICE_TICK_PIPS
    return int(raw_tick) * PRICE_PIPS_PER_CENT


def normalize_market(
    raw_market: Mapping[str, object], raw_event: Mapping[str, object] | None
) -> NormalizedMarket:
    """Normalize one raw Kalshi binary market into a :class:`NormalizedMarket`.

    Cents-denominated fields convert to pips by exact multiplication; the
    jurisdiction status is always ``"unknown"`` (SPEC S20 Q3, since Kalshi
    exposes no per-market eligibility signal); and the group id is the event
    ticker exactly when the parent event is mutually exclusive, feeding
    downstream coherence checks across the event's outcomes (SPEC S8.7).
    :meth:`NormalizedMarket.__post_init__` remains the loud backstop for any
    invariant this function does not pre-check.

    The ``volume_24h`` count (whole traded contracts) becomes
    ``volume_24h_micros`` -- settlement-notional micro-dollars -- by exact
    multiplication by :data:`MICROS_PER_CONTRACT_NOTIONAL`; this is a windbreak
    extension beyond the literal SPEC S6.2 schema. It is indexed directly (no
    ``.get`` default), so a payload missing ``volume_24h`` raises and degrades
    via the adapter's ``MARKET_MALFORMED`` path rather than silently reading as
    zero volume.

    ``close_time`` and ``expected_expiration_time`` must carry a UTC offset; an
    offsetless timestamp is refused by :func:`_parse_dt` and degrades via that
    same ``MARKET_MALFORMED`` path, rather than being read against whatever
    timezone the host happens to run in (issue #392).

    Args:
        raw_market: The raw Kalshi market payload (a ``/markets`` list entry).
        raw_event: The market's parent event payload, or None when standalone.

    Returns:
        The normalized market.

    Raises:
        ValueError: If a timestamp field carries no UTC offset.
    """
    market = _as_mapping(raw_market)
    group_id = market["event_ticker"] if _is_grouped(raw_event) else None
    return NormalizedMarket(
        exchange=KALSHI_EXCHANGE,
        ticker=market["ticker"],
        event_ticker=market["event_ticker"],
        title=market["title"],
        resolution_criteria=market["rules_primary"],
        category=market["category"],
        close_time=_parse_dt(market["close_time"], field="close_time"),
        expected_resolution_time=_parse_optional_dt(
            market.get("expected_expiration_time"), field="expected_expiration_time"
        ),
        market_type=_NORMALIZED_MARKET_TYPE,
        price_tick_pips=_tick_pips(market),
        min_order_contract_centis=MIN_ORDER_CONTRACT_CENTIS,
        fractional_trading_enabled=False,
        mutually_exclusive_group_id=group_id,
        jurisdiction_status=_JURISDICTION_UNKNOWN,
        raw_exchange_payload_hash=payload_hash(raw_market),
        volume_24h_micros=int(market["volume_24h"]) * MICROS_PER_CONTRACT_NOTIONAL,
    )


def _price_value(level: OrderBookLevel) -> int:
    """Return a level's price in pips, the key both book sides sort on."""
    return level.price.value


def _yes_bid_level(pair: Sequence[int]) -> OrderBookLevel:
    """Build a YES bid level from a raw ``[cents, count]`` pair.

    Args:
        pair: A raw ``[price_cents, contract_count]`` YES entry. A non-int
            leaf (e.g. a float) fails loudly when wrapped in a unit type.

    Returns:
        The YES bid level, priced in pips and sized in contract-centis.
    """
    cents, count = pair[0], pair[1]
    return OrderBookLevel(
        price=PricePips(cents * PRICE_PIPS_PER_CENT),
        quantity=ContractCentis(count * CENTIS_PER_CONTRACT),
    )


def _no_bid_as_yes_ask(pair: Sequence[int]) -> OrderBookLevel:
    """Invert a raw NO bid ``[cents, count]`` pair into a YES ask level.

    A resting NO bid at ``c`` cents is, in YES terms, an offer to sell YES at
    ``100 - c`` cents, so it becomes a YES ask at that inverted price.

    Args:
        pair: A raw ``[price_cents, contract_count]`` NO entry.

    Returns:
        The equivalent YES ask level, priced in pips and sized in
        contract-centis.
    """
    cents, count = pair[0], pair[1]
    yes_cents = _FULL_PRICE_CENTS - cents
    return OrderBookLevel(
        price=PricePips(yes_cents * PRICE_PIPS_PER_CENT),
        quantity=ContractCentis(count * CENTIS_PER_CONTRACT),
    )


def _order_book_map(raw: Mapping[str, object]) -> Mapping[str, object]:
    """Return the raw ``orderbook`` sub-mapping, or an empty one when absent."""
    inner = raw.get("orderbook")
    return inner if isinstance(inner, Mapping) else {}


def _raw_side(book: Mapping[str, object], key: str) -> list[Sequence[int]]:
    """Return one raw book side as a list of ``[cents, count]`` pairs.

    Args:
        book: The raw ``orderbook`` sub-mapping.
        key: The side to extract, ``"yes"`` or ``"no"``.

    Returns:
        The raw price/count pairs for that side; an empty list when the side is
        absent, None, or not a list.
    """
    side = book.get(key)
    if not isinstance(side, list):
        return []
    return cast("list[Sequence[int]]", side)


def normalize_order_book(
    ticker: str, raw: Mapping[str, object], fetched_at: datetime
) -> OrderBookSnapshot:
    """Normalize a raw Kalshi order book into an :class:`OrderBookSnapshot`.

    YES levels become YES bids (best-first: descending price); NO levels invert
    into YES asks (best-first: ascending price). Absent, None, or empty sides
    yield empty tuples rather than missing fields.

    Args:
        ticker: The market the book belongs to.
        raw: The raw payload shaped ``{"orderbook": {"yes": ..., "no": ...}}``.
        fetched_at: When the snapshot was taken.

    Returns:
        The normalized order-book snapshot.
    """
    book = _order_book_map(raw)
    yes_bids = tuple(
        sorted(
            (_yes_bid_level(pair) for pair in _raw_side(book, "yes")),
            key=_price_value,
            reverse=True,
        )
    )
    yes_asks = tuple(
        sorted(
            (_no_bid_as_yes_ask(pair) for pair in _raw_side(book, "no")),
            key=_price_value,
        )
    )
    return OrderBookSnapshot(
        ticker=ticker, yes_bids=yes_bids, yes_asks=yes_asks, fetched_at=fetched_at
    )


def _resolve_status(raw: Mapping[str, object]) -> Literal["open", "paused", "closed"]:
    """Map raw active flags to the exchange-status literal domain.

    Args:
        raw: The raw payload with ``exchange_active`` / ``trading_active`` flags.

    Returns:
        ``"open"`` when both flags are set, ``"paused"`` when the exchange is up
        but trading is halted, and ``"closed"`` when the exchange is down.
    """
    if not bool(raw.get("exchange_active")):
        return _STATUS_CLOSED
    if bool(raw.get("trading_active")):
        return _STATUS_OPEN
    return _STATUS_PAUSED


def normalize_exchange_status(
    raw: Mapping[str, object], fetched_at: datetime
) -> ExchangeStatus:
    """Normalize raw exchange active-flags into an :class:`ExchangeStatus`.

    Args:
        raw: The raw ``{"exchange_active": ..., "trading_active": ...}`` payload.
        fetched_at: When the status was observed.

    Returns:
        The normalized exchange status.
    """
    return ExchangeStatus(status=_resolve_status(raw), fetched_at=fetched_at)
