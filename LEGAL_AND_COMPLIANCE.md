# Legal and Compliance

This document describes windbreak's jurisdiction/product eligibility controls
(SPEC §6.2), the out-of-scope exclusions SPEC §1.2 mandates (including the
sports block), and the current state of operator record export. It is not
investment advice and it is not legal advice — read both statements below in
full before relying on anything else in this repository.

## No investment advice, no legal advice

**This is not investment advice.** Nothing in this software, its
documentation, or its output constitutes a recommendation to buy or sell any
financial instrument. **This is not legal advice.** Legal eligibility to trade
prediction-market products varies by jurisdiction and by product; this
software does not determine, and cannot determine, whether trading these
instruments is lawful for any particular operator in any particular place.
Consult your own qualified counsel.

## Jurisdiction and product eligibility (SPEC §6.2)

Every market the connector normalizes carries a `jurisdiction_status` field
that is exactly one of `eligible`, `ineligible`, or `unknown`
(`windbreak/connector/models.py`). Per SPEC §6.2:
`jurisdiction_status != "eligible"` means no live order is ever placed on that
market; `unknown` additionally raises an alert (the
`jurisdiction.markets_eligible` preflight check, see
[`SECURITY.md`](SECURITY.md), fails closed on any cached market that is
`unknown` or `ineligible`). Related configuration:

- `config.exchange.require_jurisdiction_eligible` — whether jurisdiction
  eligibility is enforced at all (defaults to enforced).
- `config.exchange.product_allowlist` / `config.exchange.product_blocklist` —
  the product categories permitted or forbidden regardless of jurisdiction
  status.

### How this is enforced at order time (issue #340)

The SPEC §10.3 pre-trade check `jurisdiction_product_eligibility` enforces the
§6.2 rule per order inside the Risk Kernel. It approves only a market proven
`eligible` **and** proven to be a fully collateralized binary; an `unknown`,
absent, or unrecognized value fails closed (SPEC §1.1 invariant 1).

Three things are worth stating plainly, because they are decisions rather than
deductions:

- **The kernel arm is unconditional.** `config.exchange.require_jurisdiction_eligible`
  governs the preflight and screener layers only; it does **not** relax the
  kernel check. A SPEC §10.3 check whose contract is "all must pass" should not
  be operator-flippable, or a single YAML edit would silently reopen the gate for
  `unknown` markets. Whether that knob should ever relax the kernel arm is an
  open owner decision.
- **`product_allowlist` / `product_blocklist` remain unenforced.** No
  `NormalizedMarket` field can hold the tokens those lists are written in terms
  of, so wiring them would require a human-supplied product vocabulary. They have
  no production readers today.
- **Kalshi supplies no eligibility signal.** `normalize_market` stamps every
  Kalshi market `jurisdiction_status="unknown"`, so on a live Kalshi path this
  check vetoes every order by design. Only a market carrying real eligibility
  metadata can clear it.

## Out-of-scope products and categories (SPEC §1.2)

v1 trades only fully collateralized binary event contracts; margin, perps,
options, leverage, and shorting-to-open are forbidden outright, not
configurable. Within the eligible product surface, an entire category can
still be screened out before it ever reaches a forecast. The default
`config.screener.category_blocklist` includes `sports` (blocked by default per
SPEC §1.2 — unblocking it requires an explicit config change plus a ledgered
legal-risk acknowledgement, which is not implemented today), along with
`crypto_price`, `celebrity`, and `insider_prone` markets, whose information
asymmetry or reversal risk makes them a poor fit for this design regardless of
jurisdiction.

Also out of scope per SPEC §1.2: hosted/multi-user operation; automatic
withdrawals or transfers; market making, HFT, or latency races;
celebrity/insider-dependent markets; strategy-driven early exits; portfolio
optimization across non-whitelisted instruments; and tax logic beyond record
export (below).

## Record export (current reality)

Today the only record-export path is `windbreak rebuild`, which folds a
verified, hash-chained ledger into six JSON read-model files (config versions,
mode history, gateway events, positions, equity curve, and selector
decisions) — see [`RUNBOOK.md`](RUNBOOK.md) procedure 8. There is no dedicated
audit-bundle or tax-record export command; an operator today reconstructs
those from the ledger and its read models by hand. Tracked in issue #201.
Tax *logic* (computing liability) is explicitly out of scope per SPEC §1.2 —
only record export is ever in scope.
