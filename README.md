# PSXHelper

Beginner-oriented PSX research demo. Implementation is incremental; this is
**research dashboard and reduced cash simulation implemented**, not the finished
challenge application. Real local AI remains unimplemented.

## Run from this repository

Requires Python 3.12 and `uv`. The existing provider utility lives outside the
installed package, so this checkpoint requires a source checkout and commands
run from the repository root (standalone wheel installation is not supported yet).

```bash
uv sync --locked
uv run psxhelper
```

Open <http://127.0.0.1:8000/docs> for interactive API documentation. Server binds
only to loopback. `/` provides ticker lookup; `/stocks/DCR` opens the research page.

```bash
curl http://127.0.0.1:8000/api/v1/stocks/DCR
uv run --no-sync python -m unittest discover -s tests -v
```

## Implemented

- `/health`, `/api/v1/tickers`, `/api/v1/stocks/{symbol}`, its `/ohlcv` view,
  and `/metrics` for price/volume, annual/TTM P/E and observed-volume averages.
- One adapter reusing `utils/sarmaaya_util.py`; fundamentals use isolated sessions
  to avoid the observed sticky-cookie 401 failure.
- Provider-validated free-text ticker lookup; DCR/AHCL are examples, not an allowlist.
- Quote and daily close/volume normalization, decimal-string JSON, source dates,
  provenance, coverage, manifest, and content hash excluding retrieval timestamps.
- Process-memory cache. No background refresh or disk storage. Restarting while
  offline cannot load cached data yet. Internal failed reload retains original data.
- Unknown symbols return 404; provider/validation failures return 503.
- Provider payout history and separate `dividend_summaries.dps` / `.yield`
  year-keyed facts in the aggregate response. Summaries preserve decimal strings,
  units, source and original provider values; not calculated trailing metrics.
- All four annual fundamentals use tidy `_df` methods. Income, point-in-time
  balance sheets and cash flow are statement rows; financial ratios are facts.
- Sarmaaya-reference dashboard structure: ticker header, compact snapshot and
  chart, section navigation, date-column financial/ratio tables and dividends.
  Evidence and coverage remain accessible in expandable sections.
- Dated provider `marketCap` values in `ohlcv` and its API view, with independent
  `market_cap_coverage`; units/scale unverified, no share-count reconstruction.
- Price/market-cap chart switch, hover/touch exact-value readout and keyboard
  observation slider. Missing market-cap observations leave gaps in the line.
- `POST /api/v1/allocations/preview`: single-stock, assumed one-unit lot,
  zero-fee PKR cash simulation with whole quantities and exact residual cash.

Live aggregate smoke checks returned 200 for both DCR and AHCL, with 249 bars each
and market date 2026-10-02. After restart, both aggregate responses also contain
five periods per statement category and 269 ratio facts each. Dividends and cash
preview also passed earlier live smoke checks. Last approved unit execution:
**92/92 passed**, before the later dashboard restyle. Synthetic Playwright chart
checks passed before final mobile wrapping and dashboard changes; those changes
remain unverified. Latest market-cap/dashboard live checks need user restart.
Provider availability may change; successful tests/fetching do not establish data
rights or accuracy.

## Observed limitations

Initial 401 observations are resolved: latest source probes returned 200 for
annual income and 201 for company snapshots for both tickers. The adapter now
fetches quote, 365-day history, company snapshot, four annual fundamentals, dividend history
and dated announcement references. Optional failures leave quote/history usable
with explicit coverage gaps.

Historical responses contain `price`, not full OHLC. Open/high/low remain null.
Adjustment basis, quote delay, and session completeness remain unverified.
Provider share counts are ingested, but effective dates and trading-unit
compatibility remain unverified. Annual income monetary rows use user-confirmed
PKR currency and a labeled user-assumed million-PKR scale; normalized values are
PKR while numeric-coerced provider values remain in facts. Original fundamentals
strings/precision are not retained by the client. EPS, percentages and unknown metrics are
not multiplied by a million. Period starts derive from the preceding adjacent
comparable annual end + 1 day; missing predecessors, gaps and changed fiscal
year-ends leave starts null. Derived dates are labeled, not provider-reported.
Accounting-basis verification is deferred at the user's request; basis remains
unknown. Known monetary income metrics have canonical keys; unknown metrics
remain unmapped. Repeated cross-metric values receive warnings. These standardized
rows are not presented as a verified full issuer income statement.

Business description, verified EPS and index membership remain missing/unknown.
Balance/cash-flow rows are available with unverified units/scale and unknown
accounting basis; they are not silently scaled by a million. Dividend payout records are available with unknown
ex-dates and lifecycle statuses; conflicting per-share amounts are withheld.
Posting/announcement dates do not substitute for ex-dates. Payouts are excluded
from trailing DPS/yield; provider-reported yearly summaries can display separately.
Do not infer outstanding shares from provider market capitalization. Financial
metadata uses the declared user-approved assumptions and date-derivation rule,
not verified provider metadata. Deferred accounting basis is not inferred from
ANN/LTM or ticker. Period-start, dividend and market-cap mapping tests pass.

`data_mode=remote_unverified` is an intentional temporary extension to the plan's
three release modes: accessible provider observations have not established
authorized acquisition, retention, public demonstration or redistribution rights.
No provider fixtures are committed. Live-fetch **release gate remains unverified**.
This application provides educational information, not recommendations or trading.

## Local model

Runtime binaries exist. Initial cache inspection found
`gemma-3-270m-Q8_0.gguf`; instruction tuning, chat template, compatibility and
answer quality are **not yet verified**. No model was downloaded or started.
The local assistant is not implemented; `/health` says `not connected`.

## Next checkpoints

Track acceptance and blockers in [docs/implementation-progress.md](docs/implementation-progress.md).
Full scope: [demo plan](docs/challenge-demo-plan.md). Final dashboard verification,
remaining provider-backed expansion, local AI, permitted offline snapshots,
recording and submission remain.
