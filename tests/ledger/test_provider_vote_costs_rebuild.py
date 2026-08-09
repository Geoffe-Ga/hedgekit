"""Tests for the per-provider vote-cost read-model projection (issue #281, RED).

`windbreak.ledger.events` does not yet define `ProviderVoteRecorded`, and
`windbreak.ledger.rebuild` does not yet define `provider_vote_costs_read_model`
(nor does `rebuild()` write `provider_vote_costs.json`) -- so every test below
fails with either `ImportError` (the new event type or the new projection
function) or a missing output file -- the expected Gate 1 RED state for
issue #281.

Read-model shape pinned here (mirrors `tests/ledger/test_canary_rebuild.py`'s
own issue-scoped invention, but this fold's row shape is a per-provider
AGGREGATE, not the usual `{seq, created_at, event_type, data}` passthrough
every other projection in this package uses):

    {
        "provider": str,
        "cost_micros_total": int,   # summed across every outcome (charged spend)
        "vote_count": int,          # every ProviderVoteRecorded row for this provider
        "abstain_count": int,       # rows with outcome == "abstained"
        "forecast_count": int,      # DISTINCT forecast_ids for this provider
        "cost_per_forecast_micros": int,  # cost_micros_total // forecast_count
        "abstain_rate_ppm": int,    # abstain_count * 1_000_000 // vote_count
    }

First-seen provider order, one row per provider that has ever been ledgered
(zero events for a provider means no row at all, never a zero-denominator
row) -- `[]` when no `ProviderVoteRecorded` event has ever been ledgered.

Load-bearing edge case (the issue's own headline example): the pinned
`DEFAULT_VOTE_ENSEMBLE` votes `"openai"` TWICE per forecast (two distinct
model versions), so `vote_count` and `forecast_count` genuinely diverge for a
real fleet -- `test_provider_vote_costs_read_model_same_provider_twice_per_
forecast_diverges_vote_and_forecast_count` pins that divergence directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from hypothesis import given
from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from windbreak.ledger.events import Event
    from windbreak.ledger.store import LedgerRecord

#: The three outcome strings `ProviderVoteRecorded.outcome` accepts.
_OUTCOMES = ("voted", "abstained", "discarded")


def _vote_kwargs(
    *,
    forecast_id: str = "fc-0001",
    market_ticker: str = "MKT-DEEP",
    provider: str = "openai",
    model_version: str = "gpt-5-2025-08-07",
    vote_index: int = 0,
    cost_micros: int = 0,
    outcome: str = "voted",
    failure_code: str = "",
) -> dict[str, object]:
    """Build a minimal, valid `ProviderVoteRecorded` constructor kwargs dict.

    Args:
        forecast_id: The forecast this vote belongs to.
        market_ticker: The forecast's market ticker.
        provider: The provider identifier.
        model_version: The provider's pinned model version.
        vote_index: The zero-based index of this vote in the ensemble.
        cost_micros: The vote's billed cost, in micros.
        outcome: The vote outcome (`"voted"`/`"abstained"`/`"discarded"`).
        failure_code: The discard failure code, or `""` for a non-discard.

    Returns:
        The kwargs (sans `component`) `ProviderVoteRecorded` accepts.
    """
    return {
        "forecast_id": forecast_id,
        "market_ticker": market_ticker,
        "provider": provider,
        "model_version": model_version,
        "vote_index": vote_index,
        "cost_micros": cost_micros,
        "outcome": outcome,
        "failure_code": failure_code,
    }


def _fake_record(event: Event, *, seq: int) -> LedgerRecord:
    """Wrap an already-constructed `Event` in a minimal, valid `LedgerRecord`.

    Builds the record directly (no `SqliteLedgerStore` round trip needed --
    mirrors `tests/ledger/test_ledger_store.py`'s own direct-`LedgerRecord`-
    construction convention), since `provider_vote_costs_read_model` only
    reads `record.event_type` and `record.payload_json`.

    Args:
        event: The already-constructed `ProviderVoteRecorded` event.
        seq: The record's fabricated 1-based sequence number.

    Returns:
        A `LedgerRecord` wrapping `event`'s envelope, with placeholder hash-
        chain fields (never verified by this fold).
    """
    from windbreak.ledger.store import LedgerRecord

    return LedgerRecord(
        sequence_number=seq,
        event_type=event.event_type,
        created_at="2024-01-01T00:00:00.000000+00:00",
        component=event.component,
        payload_json=event.envelope_json,
        payload_schema_version=event.payload_schema_version,
        prev_hash="0" * 64,
        event_hash="0" * 64,
    )


# --- provider_vote_costs_read_model: pure fold ---------------------------------


def test_provider_vote_costs_read_model_empty_input_returns_empty_list() -> None:
    """No records at all yields an empty list, not an error."""
    from windbreak.ledger.rebuild import provider_vote_costs_read_model

    assert provider_vote_costs_read_model([]) == []


def test_provider_vote_costs_read_model_ignores_unrelated_event_types() -> None:
    """An unrelated ledgered event type (e.g. `ModeHeartbeat`) contributes no
    row and does not crash the fold.
    """
    from windbreak.ledger.events import ModeHeartbeat
    from windbreak.ledger.rebuild import provider_vote_costs_read_model

    heartbeat = ModeHeartbeat(component="scheduler", mode="PAPER", beat=1)
    record = _fake_record(heartbeat, seq=1)

    assert provider_vote_costs_read_model([record]) == []


def test_provider_vote_costs_read_model_single_voted_event_populates_one_row() -> None:
    """One clean `"voted"` event for one provider folds into exactly one row
    with `vote_count=1`, `abstain_count=0`, `forecast_count=1`, and both
    derived ratios equal to the single event's own `cost_micros`/`0`.
    """
    from windbreak.ledger.events import ProviderVoteRecorded
    from windbreak.ledger.rebuild import provider_vote_costs_read_model

    record = _fake_record(
        ProviderVoteRecorded(
            component="scheduler",
            **_vote_kwargs(provider="openai", cost_micros=1_000, outcome="voted"),
        ),
        seq=1,
    )

    rows = provider_vote_costs_read_model([record])

    assert rows == [
        {
            "provider": "openai",
            "cost_micros_total": 1_000,
            "vote_count": 1,
            "abstain_count": 0,
            "forecast_count": 1,
            "cost_per_forecast_micros": 1_000,
            "abstain_rate_ppm": 0,
        }
    ]


def test_provider_vote_costs_read_model_same_provider_twice_diverges_counts() -> None:
    """The issue's own headline edge case: the pinned `DEFAULT_VOTE_ENSEMBLE`
    votes `"openai"` TWICE per forecast (two distinct model versions), so
    `vote_count` (every row) and `forecast_count` (distinct `forecast_id`s)
    genuinely diverge -- `cost_per_forecast_micros` divides by
    `forecast_count`, never `vote_count`.
    """
    from windbreak.ledger.events import ProviderVoteRecorded
    from windbreak.ledger.rebuild import provider_vote_costs_read_model

    records = [
        _fake_record(
            ProviderVoteRecorded(
                component="scheduler",
                **_vote_kwargs(
                    forecast_id="fc-0001",
                    provider="openai",
                    model_version="gpt-5-2025-08-07",
                    vote_index=0,
                    cost_micros=600,
                    outcome="voted",
                ),
            ),
            seq=1,
        ),
        _fake_record(
            ProviderVoteRecorded(
                component="scheduler",
                **_vote_kwargs(
                    forecast_id="fc-0001",
                    provider="openai",
                    model_version="gpt-5-mini-2025-08-07",
                    vote_index=2,
                    cost_micros=400,
                    outcome="voted",
                ),
            ),
            seq=2,
        ),
    ]

    rows = provider_vote_costs_read_model(records)

    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "openai"
    assert row["vote_count"] == 2
    assert row["forecast_count"] == 1
    assert row["cost_micros_total"] == 1_000
    assert row["cost_per_forecast_micros"] == 1_000  # 1_000 // 1, NOT 1_000 // 2
    assert row["abstain_rate_ppm"] == 0


def test_provider_vote_costs_read_model_abstain_rate_is_exact_integer_floor() -> None:
    """One abstained vote out of two for a provider folds to
    `abstain_rate_ppm == 500_000` (exact integer floor division, never a
    float): `1 * 1_000_000 // 2`.
    """
    from windbreak.ledger.events import ProviderVoteRecorded
    from windbreak.ledger.rebuild import provider_vote_costs_read_model

    records = [
        _fake_record(
            ProviderVoteRecorded(
                component="scheduler",
                **_vote_kwargs(
                    forecast_id="fc-0001",
                    provider="anthropic",
                    vote_index=0,
                    cost_micros=100,
                    outcome="voted",
                ),
            ),
            seq=1,
        ),
        _fake_record(
            ProviderVoteRecorded(
                component="scheduler",
                **_vote_kwargs(
                    forecast_id="fc-0002",
                    provider="anthropic",
                    vote_index=1,
                    cost_micros=200,
                    outcome="abstained",
                ),
            ),
            seq=2,
        ),
    ]

    rows = provider_vote_costs_read_model(records)

    assert len(rows) == 1
    row = rows[0]
    assert row["vote_count"] == 2
    assert row["abstain_count"] == 1
    assert row["abstain_rate_ppm"] == 500_000
    assert type(row["abstain_rate_ppm"]) is int


def test_provider_vote_costs_read_model_charges_discarded_cost_into_the_total() -> None:
    """A `"discarded"` outcome's `cost_micros` is charged into
    `cost_micros_total` (charged spend across every outcome) AND counted in
    `vote_count` -- a discarded vote still cost real money, and it is still
    one recorded `ProviderVoteRecorded` row, even though the run never used
    its vote.
    """
    from windbreak.ledger.events import ProviderVoteRecorded
    from windbreak.ledger.rebuild import provider_vote_costs_read_model

    records = [
        _fake_record(
            ProviderVoteRecorded(
                component="scheduler",
                **_vote_kwargs(
                    forecast_id="fc-0001",
                    provider="futuresearch",
                    vote_index=0,
                    cost_micros=300,
                    outcome="voted",
                ),
            ),
            seq=1,
        ),
        _fake_record(
            ProviderVoteRecorded(
                component="scheduler",
                **_vote_kwargs(
                    forecast_id="fc-0001",
                    provider="futuresearch",
                    vote_index=1,
                    cost_micros=700,
                    outcome="discarded",
                    failure_code="provider_timeout",
                ),
            ),
            seq=2,
        ),
    ]

    rows = provider_vote_costs_read_model(records)

    assert len(rows) == 1
    row = rows[0]
    assert row["vote_count"] == 2
    assert row["abstain_count"] == 0
    assert row["forecast_count"] == 1
    assert row["cost_micros_total"] == 1_000
    assert row["cost_per_forecast_micros"] == 1_000


def test_provider_vote_costs_read_model_multiple_providers_first_seen_order() -> None:
    """Two providers each fold into their own row, in first-seen order --
    never alphabetical, never ledger-reverse."""
    from windbreak.ledger.events import ProviderVoteRecorded
    from windbreak.ledger.rebuild import provider_vote_costs_read_model

    records = [
        _fake_record(
            ProviderVoteRecorded(
                component="scheduler",
                **_vote_kwargs(
                    forecast_id="fc-0001",
                    provider="futuresearch",
                    vote_index=0,
                    cost_micros=100,
                    outcome="voted",
                ),
            ),
            seq=1,
        ),
        _fake_record(
            ProviderVoteRecorded(
                component="scheduler",
                **_vote_kwargs(
                    forecast_id="fc-0001",
                    provider="anthropic",
                    vote_index=1,
                    cost_micros=200,
                    outcome="voted",
                ),
            ),
            seq=2,
        ),
        _fake_record(
            ProviderVoteRecorded(
                component="scheduler",
                **_vote_kwargs(
                    forecast_id="fc-0002",
                    provider="futuresearch",
                    vote_index=0,
                    cost_micros=300,
                    outcome="voted",
                ),
            ),
            seq=3,
        ),
    ]

    rows = provider_vote_costs_read_model(records)

    assert [row["provider"] for row in rows] == ["futuresearch", "anthropic"]
    futuresearch_row = rows[0]
    assert futuresearch_row["vote_count"] == 2
    assert futuresearch_row["forecast_count"] == 2
    assert futuresearch_row["cost_micros_total"] == 400
    # forecast_count > 1: 400 // 2 == 200 (the floor-division case `// 1` cannot pin).
    assert futuresearch_row["cost_per_forecast_micros"] == 200


# --- Property test: fold invariants over arbitrary integer vote streams --------

_vote_dict_strategy = st.fixed_dictionaries(
    {
        "forecast_id": st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=6,
        ),
        "provider": st.sampled_from(["openai", "anthropic", "futuresearch"]),
        "model_version": st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=6,
        ),
        "vote_index": st.integers(min_value=0, max_value=5),
        "cost_micros": st.integers(min_value=0, max_value=10_000_000),
        "outcome": st.sampled_from(_OUTCOMES),
    }
)


@given(st.lists(_vote_dict_strategy, min_size=0, max_size=25))
def test_provider_vote_costs_read_model_fold_invariants_hold(
    votes: list[dict[str, object]],
) -> None:
    """Integer-only fold invariants hold for ANY stream of vote records:

    * every `abstain_rate_ppm` sits in the closed `[0, 1_000_000]` range;
    * the sum of every row's `vote_count` equals the total event count;
    * each row's `cost_micros_total` equals the exact sum of that provider's
      own event costs (never leaking another provider's spend);
    * every output value is a true `int`, never a `bool` or a `float`.
    """
    from windbreak.ledger.events import ProviderVoteRecorded
    from windbreak.ledger.rebuild import provider_vote_costs_read_model

    records = [
        _fake_record(
            ProviderVoteRecorded(
                component="scheduler",
                **_vote_kwargs(
                    forecast_id=cast("str", vote["forecast_id"]),
                    provider=cast("str", vote["provider"]),
                    model_version=cast("str", vote["model_version"]),
                    vote_index=cast("int", vote["vote_index"]),
                    cost_micros=cast("int", vote["cost_micros"]),
                    outcome=cast("str", vote["outcome"]),
                    failure_code=(
                        "some_failure" if vote["outcome"] == "discarded" else ""
                    ),
                ),
            ),
            seq=index,
        )
        for index, vote in enumerate(votes, start=1)
    ]

    rows = provider_vote_costs_read_model(records)

    total_vote_count = sum(cast("int", row["vote_count"]) for row in rows)
    assert total_vote_count == len(votes)
    for row in rows:
        provider = row["provider"]
        provider_votes = [vote for vote in votes if vote["provider"] == provider]
        assert row["vote_count"] == len(provider_votes)
        assert row["cost_micros_total"] == sum(
            cast("int", vote["cost_micros"]) for vote in provider_votes
        )
        assert row["abstain_count"] == sum(
            1 for vote in provider_votes if vote["outcome"] == "abstained"
        )
        abstain_rate_ppm = cast("int", row["abstain_rate_ppm"])
        assert 0 <= abstain_rate_ppm <= 1_000_000
        for value in row.values():
            assert type(value) in (int, str), f"unexpected leaf type: {value!r}"
            assert type(value) is not bool, f"bool leaked as a fold value: {value!r}"
            assert type(value) is not float, f"float leaked as a fold value: {value!r}"


# --- rebuild(): the projection is wired in, writing provider_vote_costs.json ---


def test_rebuild_writes_provider_vote_costs_json_unconditionally(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`rebuild()` writes `provider_vote_costs.json` unconditionally -- present
    (and empty, `[]`) even when no `ProviderVoteRecorded` event has ever been
    ledgered, mirroring every other projection's unconditional-write contract.
    """
    import json

    from windbreak.ledger.events import ModeHeartbeat
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(ModeHeartbeat(component="scheduler", mode="PAPER", beat=1))
    store.close()

    rebuild(db_path, output_dir)

    provider_vote_costs = json.loads(
        (output_dir / "provider_vote_costs.json").read_text()
    )
    assert provider_vote_costs == []


def test_rebuild_writes_provider_vote_costs_json_with_real_fold_data(
    tmp_path: Path, deterministic_clock: Callable[[], datetime]
) -> None:
    """`rebuild()`'s `provider_vote_costs.json` holds the rows
    `provider_vote_costs_read_model` would itself produce over the same
    verified ledger.
    """
    import json

    from windbreak.ledger.events import ProviderVoteRecorded
    from windbreak.ledger.rebuild import rebuild
    from windbreak.ledger.store import SqliteLedgerStore

    db_path = tmp_path / "ledger.db"
    output_dir = tmp_path / "out"
    store = SqliteLedgerStore(db_path, now=deterministic_clock)
    store.append(
        ProviderVoteRecorded(
            component="scheduler",
            **_vote_kwargs(
                forecast_id="fc-0001",
                provider="openai",
                cost_micros=1_500,
                outcome="voted",
            ),
        )
    )
    store.close()

    rebuild(db_path, output_dir)

    provider_vote_costs = json.loads(
        (output_dir / "provider_vote_costs.json").read_text()
    )
    assert provider_vote_costs == [
        {
            "provider": "openai",
            "cost_micros_total": 1_500,
            "vote_count": 1,
            "abstain_count": 0,
            "forecast_count": 1,
            "cost_per_forecast_micros": 1_500,
            "abstain_rate_ppm": 0,
        }
    ]
