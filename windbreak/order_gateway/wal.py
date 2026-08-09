"""Durable write-ahead intent log for crash recovery (issue #40, SPEC S11.4).

Before the Order Gateway takes any externally-visible action for an intent -- and
the instant a placement's ack comes back -- it journals a durable record here, so
a crash at any point along ``APPROVE -> REQUEST_SUBMISSION -> (place) -> SUBMIT
-> ACK`` leaves a fresh Gateway's :meth:`~windbreak.order_gateway.gateway.
OrderGateway.recover` enough durable truth to reconstruct what happened without
ever double-submitting.

The log is an append-only JSONL file: one :func:`~windbreak.ledger.events.
canonical_json` line per record, ``flush``ed and ``os.fsync``ed on every append
so a record survives a crash the moment the append returns. Two record kinds
share one :class:`WalRecord` shape:

    * an *intent* record -- the full nine :class:`~windbreak.riskkernel.checks.
      OrderIntent` fields (the four scaled-int money-path fields as their bare
      ``.value`` ints, SPEC S6.1), written *before* the ``REQUEST_SUBMISSION``
      transition. The signed token and any key material are **never** written.
    * an *ack* record -- the venue-order-id / ``client_order_id`` correlation and
      the immediately-filled quantity, written the instant ``place`` returns.

:meth:`WriteAheadLog.read_all` reconstructs each ``OrderIntent`` from ints only
and re-derives its :func:`~windbreak.order_gateway.client_order_id.client_order_id`,
failing loudly if the re-derived id disagrees with the recorded one (a tampered
or corrupt journal must never silently mis-attribute an order).

**The read path refuses anything it cannot fully validate** (issue #427). Every
line must decode to a JSON object; every record must declare a recognised
``kind``; and a record -- like an intent record's payload -- must carry
*exactly* the fields that kind requires, no more and no fewer. Each refusal is
an explicit ``ValueError`` naming the offending line and value. That exactness
is what closes the discriminator: ``"ack"`` is a legal ``kind``, so relabelling
a journalled *intent* record as an ack would otherwise route it past the id
re-derivation above -- but the relabelled record still carries an ``intent``
field and neither of the two an ack requires, so it is refused. An extra field
written by some newer build is refused for the same reason: this reader cannot
attest to content it does not understand, and fail-closed means refusing rather
than reading the subset it recognises.

**Blank and torn lines are refused, not skipped** (issue #427). A crash during
``_append`` can leave a partial final line, and the tempting rule -- tolerate a
torn *tail*, since the record it describes was never acted on -- does not
survive contact with an append-only file: a torn line has no terminating
newline, so the next append fuses onto it, taking a sound, durably written
record down with the damaged one. There is no way to tell a crash-torn line
from any other corruption after the fact, so the journal takes the one rule it
can prove: every line must be a complete record, and any line that is not stops
the read. Recovering from a silently shortened WAL is precisely the
resubmit/orphan hazard the log exists to prevent; truncating a damaged journal
is an operator's deliberate act, never the reader's guess.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from windbreak.ledger.events import canonical_json
from windbreak.numeric.types import (
    ContractCentis,
    MoneyMicros,
    PricePips,
    ProbabilityPpm,
)
from windbreak.order_gateway.client_order_id import client_order_id
from windbreak.riskkernel.checks import OrderIntent

if TYPE_CHECKING:
    from pathlib import Path

#: Discriminator value marking a journalled intent record.
_KIND_INTENT = "intent"

#: Discriminator value marking a journalled ack record.
_KIND_ACK = "ack"

#: The exact field set an intent record carries -- what :meth:`WriteAheadLog.
#: append_intent` writes, and therefore the only shape the read path accepts.
_INTENT_RECORD_FIELDS = frozenset({"kind", "client_order_id", "intent"})

#: The exact field set an ack record carries (:meth:`WriteAheadLog.append_ack`).
_ACK_RECORD_FIELDS = frozenset({"kind", "client_order_id", "order_id", "filled"})

#: The exact nine fields an intent record's payload carries (SPEC S6.1) -- what
#: :func:`_intent_to_payload` writes, and what ``client_order_id`` hashes.
_INTENT_PAYLOAD_FIELDS = frozenset(
    {
        "intent_id",
        "market_ticker",
        "outcome",
        "action",
        "price",
        "size",
        "max_notional",
        "implied_probability",
        "idempotency_key",
    }
)


@dataclass(frozen=True, slots=True)
class WalRecord:
    """One journalled write-ahead record (an intent or an ack).

    A single shape carries both kinds; only the fields relevant to ``kind`` are
    populated (the rest carry inert sentinels), so a caller filters on ``kind``
    before reading ``intent`` (intent records) or ``order_id``/``filled`` (ack
    records).

    Attributes:
        kind: Either ``"intent"`` or ``"ack"``.
        client_order_id: The content-addressed id the record belongs to.
        intent: The journalled :class:`~windbreak.riskkernel.checks.OrderIntent`
            on an intent record, else ``None``.
        order_id: The venue's resting-order id on an ack record (``None`` when
            the placement left nothing resting); always ``None`` on an intent
            record.
        filled: The quantity filled immediately on an ack record, in
            contract-centis; ``ContractCentis(0)`` on an intent record.
    """

    kind: str
    client_order_id: str
    intent: OrderIntent | None
    order_id: str | None
    filled: ContractCentis


class WriteAheadLogProtocol(Protocol):
    """The structural seam the Gateway durably journals intents and acks through.

    Mirrors :class:`~windbreak.order_gateway.ledger_writer.GatewayLedgerWriter`'s
    protocol-first design so a crash-simulating test wrapper (or any alternative
    durable log) can stand in for the real :class:`WriteAheadLog` while staying
    ``mypy --strict`` clean. Parameters are positional-only so an implementer may
    name them freely.
    """

    def append_intent(self, intent: OrderIntent, client_order_id: str, /) -> None:
        """Durably journal ``intent`` before the Gateway acts on it.

        Args:
            intent: The order intent to journal.
            client_order_id: The intent's content-addressed id.
        """
        ...

    def append_ack(
        self,
        client_order_id: str,
        order_id: str | None,
        filled: ContractCentis,
        /,
    ) -> None:
        """Durably journal a placement's ack the instant ``place`` returns.

        Args:
            client_order_id: The intent's content-addressed id.
            order_id: The venue's resting-order id, or ``None`` when nothing
                rested.
            filled: The quantity filled immediately, in contract-centis.
        """
        ...

    def read_all(self) -> tuple[WalRecord, ...]:
        """Return every journalled record, in append order.

        Returns:
            The journalled records, oldest first.
        """
        ...


class WriteAheadLog:
    """An append-only, ``fsync``-durable JSONL write-ahead log (issue #40)."""

    def __init__(self, path: Path) -> None:
        """Bind the log to its JSONL file (created lazily on first append).

        Args:
            path: Filesystem path to the append-only JSONL journal.
        """
        self._path = path

    def append_intent(self, intent: OrderIntent, client_order_id_: str) -> None:
        """Durably journal an intent record.

        Args:
            intent: The order intent to journal, serialized as ints and strings
                only (never the token or any key material).
            client_order_id_: The intent's content-addressed id.
        """
        self._append(
            {
                "kind": _KIND_INTENT,
                "client_order_id": client_order_id_,
                "intent": _intent_to_payload(intent),
            }
        )

    def append_ack(
        self, client_order_id_: str, order_id: str | None, filled: ContractCentis
    ) -> None:
        """Durably journal an ack record.

        Args:
            client_order_id_: The intent's content-addressed id.
            order_id: The venue's resting-order id, or ``None`` when nothing
                rested.
            filled: The quantity filled immediately, in contract-centis.
        """
        self._append(
            {
                "kind": _KIND_ACK,
                "client_order_id": client_order_id_,
                "order_id": order_id,
                "filled": filled.value,
            }
        )

    def _append(self, obj: dict[str, object]) -> None:
        """Append one canonical-JSON line, flushing and ``fsync``ing it durable.

        On the append that first creates the journal file, the parent directory
        is ``fsync``ed too, so the new directory entry survives a crash in the
        window between file creation and the OS flushing that entry -- otherwise
        a just-created log could vanish despite its data being ``fsync``ed.

        Args:
            obj: The record mapping to serialize as a single JSONL line.
        """
        line = canonical_json(obj)
        is_new_file = not self._path.exists()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if is_new_file:
            self._fsync_parent_dir()

    def _fsync_parent_dir(self) -> None:
        """``fsync`` the journal's parent directory to persist a new file entry."""
        dir_fd = os.open(self._path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def read_all(self) -> tuple[WalRecord, ...]:
        """Reconstruct every journalled record, verifying each intent's id.

        Returns:
            The journalled records, oldest first; an empty tuple when the
            journal file does not yet exist.

        Raises:
            ValueError: If any line is not a complete JSON object, declares an
                absent or unrecognised ``kind``, does not carry exactly the
                fields its kind requires, or is an intent whose re-derived
                :func:`~windbreak.order_gateway.client_order_id.client_order_id`
                disagrees with the id it was recorded under. Every one of those
                means a corrupt journal, and none is recoverable by guessing.
        """
        if not self._path.exists():
            return ()
        text = self._path.read_text(encoding="utf-8")
        records = [
            self._record_from_line(line, lineno)
            for lineno, line in enumerate(text.splitlines(), start=1)
        ]
        return tuple(records)

    def _record_from_line(self, line: str, lineno: int) -> WalRecord:
        """Parse one JSONL line into a fully validated :class:`WalRecord`.

        Args:
            line: One canonical-JSON journal line.
            lineno: The line's 1-based position, for error messages.

        Returns:
            The reconstructed record.

        Raises:
            ValueError: If the line is not a complete JSON object, carries no
                ``kind`` or an unrecognised one, does not carry exactly the
                fields its kind requires, or is an intent whose re-derived id
                disagrees with its recorded ``client_order_id``.
        """
        obj = _require_object(_decode_line(line, lineno), f"line {lineno}")
        if "kind" not in obj:
            raise ValueError(
                f"write-ahead log corrupt: the record on line {lineno} is missing "
                f"its 'kind' field"
            )
        kind = obj["kind"]
        if kind == _KIND_INTENT:
            _require_fields(
                obj, _INTENT_RECORD_FIELDS, f"the intent record on line {lineno}"
            )
            return self._intent_record(
                cast("str", obj["client_order_id"]), obj["intent"], lineno
            )
        if kind == _KIND_ACK:
            _require_fields(obj, _ACK_RECORD_FIELDS, f"the ack record on line {lineno}")
            return WalRecord(
                kind=_KIND_ACK,
                client_order_id=cast("str", obj["client_order_id"]),
                intent=None,
                order_id=cast("str | None", obj["order_id"]),
                filled=ContractCentis(cast("int", obj["filled"])),
            )
        raise ValueError(
            f"write-ahead log corrupt: the record on line {lineno} has unrecognised "
            f"kind {kind!r} (expected {_KIND_INTENT!r} or {_KIND_ACK!r})"
        )

    def _intent_record(self, coid: str, raw_payload: object, lineno: int) -> WalRecord:
        """Rebuild an intent record and verify its content-addressed id.

        Args:
            coid: The id the intent was recorded under.
            raw_payload: The journalled ``intent`` value, not yet known to be a
                mapping of exactly the nine intent fields.
            lineno: The record's 1-based line, for error messages.

        Returns:
            The reconstructed intent :class:`WalRecord`.

        Raises:
            ValueError: If the payload is not an object carrying exactly the
                nine intent fields, or if ``client_order_id(intent)`` disagrees
                with ``coid``.
        """
        where = f"the intent payload on line {lineno}"
        payload = _require_object(raw_payload, where)
        _require_fields(payload, _INTENT_PAYLOAD_FIELDS, where)
        intent = _intent_from_payload(payload)
        rederived = client_order_id(intent)
        if rederived != coid:
            raise ValueError(
                f"write-ahead log corrupt: intent re-derives client_order_id "
                f"{rederived!r} but was journalled under {coid!r}"
            )
        return WalRecord(
            kind=_KIND_INTENT,
            client_order_id=coid,
            intent=intent,
            order_id=None,
            filled=ContractCentis(0),
        )


def _decode_line(line: str, lineno: int) -> object:
    """Decode one journal line, refusing anything that will not parse.

    The raised error is a plain ``ValueError``, never the ``JSONDecodeError``
    ``json`` raises. That subclasses ``ValueError``, so letting it escape would
    look like a refusal without any code having refused anything -- and a caller
    catching ``ValueError`` could not tell the two apart.

    Args:
        line: One raw line of the journal (blank lines included).
        lineno: The line's 1-based position, for the error message.

    Returns:
        The decoded JSON value, not yet known to be an object.

    Raises:
        ValueError: If the line is blank or is not well-formed JSON -- the
            shape a crash mid-append leaves behind.
    """
    try:
        decoded: object = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"write-ahead log corrupt: line {lineno} is not a well-formed JSON record"
        ) from exc
    return decoded


def _require_object(value: object, where: str) -> dict[str, object]:
    """Narrow a decoded JSON value to a mapping, refusing anything else.

    Args:
        value: The decoded JSON value.
        where: Which part of which line it came from, for the error message.

    Returns:
        ``value`` as a mapping.

    Raises:
        ValueError: If ``value`` is not a JSON object.
    """
    if not isinstance(value, dict):
        raise ValueError(f"write-ahead log corrupt: {where} is not a JSON object")
    return cast("dict[str, object]", value)


def _require_fields(
    obj: dict[str, object], expected: frozenset[str], where: str
) -> None:
    """Refuse a mapping that does not carry exactly ``expected``'s fields.

    Exactly, in both directions: a missing field means the record cannot be
    fully read, and an unexpected one means it was written by something this
    build does not understand. Either way the reader cannot validate what it
    has, so it refuses rather than reading the part it recognises.

    Args:
        obj: The mapping to check.
        expected: The fields it must carry, and only those.
        where: Which part of which line it came from, for the error message.

    Raises:
        ValueError: If ``obj``'s field set differs from ``expected``. Both sets
            are reported sorted, so the message does not depend on key order.
    """
    if set(obj) != expected:
        raise ValueError(
            f"write-ahead log corrupt: {where} carries fields {sorted(obj)} "
            f"(expected {sorted(expected)})"
        )


def _intent_to_payload(intent: OrderIntent) -> dict[str, object]:
    """Project an ``OrderIntent`` into its JSON-safe, float-free journal payload.

    Args:
        intent: The order intent to serialize.

    Returns:
        The nine intent fields, with every scaled-int money-path field rendered
        as its bare ``.value`` integer (SPEC S6.1).
    """
    return {
        "intent_id": intent.intent_id,
        "market_ticker": intent.market_ticker,
        "outcome": intent.outcome,
        "action": intent.action,
        "price": intent.price.value,
        "size": intent.size.value,
        "max_notional": intent.max_notional.value,
        "implied_probability": intent.implied_probability.value,
        "idempotency_key": intent.idempotency_key,
    }


def _intent_from_payload(payload: dict[str, object]) -> OrderIntent:
    """Rebuild an ``OrderIntent`` from a journal payload, ints only.

    Args:
        payload: The journalled intent payload.

    Returns:
        The reconstructed :class:`~windbreak.riskkernel.checks.OrderIntent`, its
        scaled-int fields rewrapped from their integer ``.value`` (never a
        float).
    """
    return OrderIntent(
        intent_id=cast("str", payload["intent_id"]),
        market_ticker=cast("str", payload["market_ticker"]),
        outcome=cast("str", payload["outcome"]),
        action=cast("str", payload["action"]),
        price=PricePips(cast("int", payload["price"])),
        size=ContractCentis(cast("int", payload["size"])),
        max_notional=MoneyMicros(cast("int", payload["max_notional"])),
        implied_probability=ProbabilityPpm(cast("int", payload["implied_probability"])),
        idempotency_key=cast("str", payload["idempotency_key"]),
    )
