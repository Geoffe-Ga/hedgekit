# Kalshi fixture evidence: `BalanceSemantics` (issue #18)

SPEC §7.3 ("Balance-semantics contract") calls the `BalanceSemantics` record a
**blocker for live trading**: the Risk Kernel refuses to trade live while any
one of its eight fields is `unknown`. This directory holds the fixtures that
back Kalshi's recorded answers -- one fixture (or fixture pair) per field --
so each field's value in
`windbreak.connector.kalshi.adapter.KALSHI_BALANCE_SEMANTICS` is proven by a
concrete before/after payload rather than asserted from memory. The tests in
`tests/connector/kalshi/test_balance_semantics.py` pin the record against
this evidence.

## Evidence table

| Field | Resolved value | Fixture(s) | Justification |
|---|---|---|---|
| `fee_debit_timing` | `FeeDebitTiming.AT_EXECUTION` | `fills.json` | Each fill record carries its own non-null `fee`, recorded at the fill's `created_time` -- i.e. at execution, not deferred to settlement. |
| `fee_rounding` | `FeeRounding.UP_TO_NEXT_CENT` | `series_KXFED.json` | Kalshi's documented fee formula (`fee = ceil(rate * C * P * (1-P))`) rounds every fee up to the next whole cent; see `windbreak.connector.fees.FeeModel.max_trading_fee_micros`, which mirrors this rounding as its conservative (overstate-cost) bound. |
| `partial_fill_representation` | `PartialFillRepresentation.PER_FILL_RECORDS` | `fills.json` | `trade-1` and `trade-2` share one `order_id` (`"order-abc123"`) but are two distinct fill records, each with its own `count` and `fee`, rather than one aggregated fill. |
| `unsettled_proceeds` | `UnsettledProceeds.EXCLUDED_UNTIL_CREDITED` | `settlements.json` | `balance_before_settlement` (89,000,000 micros) only becomes `balance_after_settlement` (189,000,000 micros -- a jump of exactly the `revenue` field: 10,000 cents = 100,000,000 micros) strictly after `settled_time`; proceeds are never visible earlier. |
| `open_order_collateral_in_available` | `OrderCollateralInAvailable.DEDUCTED_FROM_AVAILABLE` | `resting_order_balance.json` | `balance_before_order` vs. `balance_after_order` shows `available_balance` drop by exactly the resting order's max cost (50 contracts x 45c = 2,250 cents = 22,500,000 micros: 100,000,000 -> 77,500,000), while `balance` (total) is unchanged. |
| `open_order_collateral_in_total` | `OrderCollateralInTotal.UNKNOWN` | -- | No fixture evidence exists either way; see "Deliberately unknown" below. |
| `cancel_collateral_release` | `CancelCollateralRelease.UNKNOWN` | -- | No fixture evidence exists either way; see "Deliberately unknown" below. |
| `halted_market_behavior` | `HaltedMarketBehavior.UNKNOWN` | -- | No fixture evidence exists either way; see "Deliberately unknown" below. |

## Deliberately unknown, not guessed

Three fields stay `UNKNOWN`:

* `open_order_collateral_in_total` -- whether a resting order's posted
  collateral is folded into the reported account *total* (as opposed to
  *available*) is not documented in Kalshi's public API, and no
  before/after total-balance pair around placing an order was recorded.
* `cancel_collateral_release` -- whether collateral frees immediately on
  cancel-acknowledgment or only after some later step is not committed to
  in Kalshi's public docs, and no before/after balance pair around a
  cancellation was recorded.
* `halted_market_behavior` -- whether new orders are rejected or accepted
  against a halted market is not committed to in Kalshi's public docs, and
  no fixture captures order-submission behavior during a halt.

SPEC §7.3 requires these questions be answered "via fixture tests and, where
available, demo-environment tests" -- it does not permit inferring a
convenient default when no such evidence exists. These three fields are
therefore left at their enum's explicit `UNKNOWN` member on purpose: an
undocumented behavior is *recorded as unknown*, never silently guessed. Each
would be promoted to a known member only by future evidence of the same
shape as the five known rows above -- e.g. a demo-environment before/after
balance pair straddling a cancel, a total-balance snapshot pair straddling a
resting order, or an observed order-submission response against a halted
market.

## Fee-schedule fixtures are an invented shape

* `series_KXFED.json` -- a recorded `/series/{ticker}` fee-schedule document.
  **The shape is invented**: Kalshi's real `/series/{ticker}` fee-field
  layout was not confirmed from this offline environment. It carries a
  `fee_type` discriminator that acts as a sanctioned-schedule allowlist --
  `"quadratic"` is the only value the adapter accepts; any other value fails
  closed with `UnknownFeeModelError` rather than being misread -- plus
  `maker_fee_bps` / `taker_fee_bps` / `settlement_fee_bps` in basis points
  (1 bp = 100 ppm, so the adapter's `_bps_to_ppm` multiplies by 100), and an
  explicit `fee_schedule_id` copied verbatim into `FeeModel.schedule_id`.
  This fixture normalizes to `maker_fee_ppm=0`, `taker_fee_ppm=70_000`,
  `settlement_fee_ppm=0` -- the rate behind the golden trading-fee example
  proven in `tests/connector/kalshi/test_fees.py`: **$1.75 (1,750,000
  micros) for 100 contracts at 50c**, the issue's worked example for
  `FeeModel.max_trading_fee_micros`.
* `series_KXBAD.json` -- the same invented shape with an unrecognized
  `fee_type` (`"mystery_v9"`), backing the malformed-schedule fail-closed
  test.

Per SPEC §20 Q1, fee schedules must be data pulled from the exchange's live
fee schedule, never hardcoded from a secondary source, with golden tests
checked against exchange-documented examples. Confirming the real
`/series/{ticker}` schema against Kalshi's live API -- and re-recording these
two fixtures against it if the real shape differs -- is a follow-up, not
something this offline environment can verify.

## The published fee schedule itself (issue #452)

* `fee_schedule_published.json` -- **not an API payload**. It is Kalshi's own
  fee-schedule PDF transcribed into JSON: the four published formulas with
  their quoted text and basis-point rates, and both printed fee tables (the
  general one and the S&P500 / Nasdaq-100 one), each 21 rows of
  `price_cents` / `fee_1_contract_cents` / `fee_100_contracts_cents`. Source
  and effective date (`2026-02-05`) are carried as fields so no reader has to
  re-derive the provenance. The live PDF URL sits behind a bot checkpoint that
  refuses this environment; the copy transcribed was the Internet Archive's
  2026-02-18 capture, also recorded in the file.

  `tests/connector/kalshi/test_fee_schedule.py` replays every row against
  `FeeModel`, so the record is load-bearing: a change to the rate, the form,
  the scale, or the rounding direction breaks a test that cites a printed
  number. It also asserts the finding that closes issue #452 -- **every
  published trading fee is the same quadratic form**, so
  `max(maker_fee_ppm, taker_fee_ppm)` is the higher *fee*, not merely the
  higher rate. See `ACCOUNTING.md`'s "The venue's fee schedule, as published"
  for the reasoning and for the two guards that fail closed if that ever stops
  being true.

  Note what this fixture does *not* settle: which series carry a maker fee.
  The schedule defers that to a separate live page, so `series_KXFED.json`
  keeps `maker_fee_bps: 0` rather than asserting a maker rate for KXFED that
  no artifact here supports.

## Fixture inventory

* `series_KXFED.json` -- see above.
* `series_KXBAD.json` -- see above.
* `fee_schedule_published.json` -- see above.
* `fills.json` -- two fill records sharing one `order_id`, each with its own
  per-fill `fee`: evidence for `partial_fill_representation` and
  `fee_debit_timing`.
* `settlements.json` -- a settlement record plus a balance snapshot pair
  taken immediately before and after `settled_time`: evidence for
  `unsettled_proceeds`.
* `resting_order_balance.json` -- a balance snapshot pair taken immediately
  before and after placing one resting order: evidence for
  `open_order_collateral_in_available`.

These fixtures are evidence artifacts, not (yet) wired into `KalshiConnector`
methods beyond `get_fee_model` (`series_KXFED.json` / `series_KXBAD.json`).
`fills.json` (top-level list, keyed by `id`/`price`/`quantity`) already backs
`FakeExchange.get_fills`; this directory's `fills.json` is the
Kalshi-raw-shape sibling used only as semantics evidence, not loaded by
`FakeExchange`.

## `faults/`: schema-drift fixtures (issue #20)

Two hand-authored variants of the recorded `orderbook_KXFED-24DEC.json`
payload back `windbreak.connector.validation`'s fail-closed schema-drift
tests (`tests/connector/test_validation.py`,
`tests/connector/kalshi/test_client_resilience.py`):

* `orderbook_drift_money_fee.json` -- the same `yes`/`no` book, plus an
  unrecognized `fee` key nested *inside* the `orderbook` object (sibling to
  `yes`/`no`). Neither recognized nor cosmetic in
  `kalshi_default_schema_registry()`'s orderbook schema, so validating this
  payload must ledger a `SCHEMA_ANOMALY` event and raise
  `SchemaAnomalyHaltError` -- an unrecognized *money/risk* field is exactly the
  drift this validator exists to catch, fail closed rather than silently
  misread.
* `orderbook_drift_cosmetic.json` -- the same book, plus a top-level
  `display_label` key. `display_label` is deliberately listed in the
  orderbook schema's `cosmetic` allowlist, so validating this payload must
  only log a warning -- no ledger event, no raise.

The 5xx / 429 / malformed-JSON (and truncated-body) fault cases are
session-scripted (see `ScriptedFaultSession` / `QueuedFaultResponse` in
`tests/connector/kalshi/conftest.py`) rather than fixture files, since they are
HTTP-transport-level faults, not payload-shape drift. Each is exercised
end-to-end through a real `KalshiClient` in
`tests/connector/kalshi/test_client_resilience.py`:

* **5xx** — `test_get_recovers_after_two_5xx_then_a_200` (transparent retry to
  recovery) and `test_persistent_5xx_exhausts_retries_and_eventually_trips_the_breaker`
  (retry exhaustion → circuit-breaker halt).
* **429** — `test_get_recovers_after_a_429_then_a_200` (rate-limit backpressure
  retried like a 5xx).
* **Malformed / truncated body** — `QueuedFaultResponse(json_raises=...)` makes
  the transport's real `response.json()` raise;
  `test_get_recovers_after_a_malformed_body_then_a_200` proves recovery and
  `test_persistent_malformed_body_exhausts_retries_and_surfaces` proves it fails
  closed (the parse error surfaces; no partial data is returned).
