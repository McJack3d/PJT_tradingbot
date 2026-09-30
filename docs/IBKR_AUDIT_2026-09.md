# IBKR Stocks-Only Audit & Roadmap (2026-09)

Scope: refocus TRAD_BOT on **US/EU equities traded through Interactive
Brokers only**. This document audits the current repository, lists the
defects found in the IBKR sentiment bot, and proposes a new strategy
stack plus the engineering work to support it.

State at audit time: `main` @ `2d8ccea`, 464 unit tests passing
(64 of them IBKR); the IBKR bot is ~4.1k of ~15.4k lines in `src/`.

---

## 0. Critical — do this first

**`.env.save` is committed to git** (added in `2d8ccea`). It contains a
64-char `BINANCE_API_KEY` and `BINANCE_API_SECRET` with
`SIMPLE_BOT_LIVE=true` / `BINANCE_TESTNET=false`, i.e. a live key.
`.gitignore` covers `.env` but not `.env.save`.

1. Revoke that key in Binance *now* (deleting the file does not remove
   it from history, forks, or clones).
2. `git rm --cached .env.save`, add `.env*` + `!.env.example` to
   `.gitignore`.
3. Enable the already-configured `detect-secrets` (`.secrets.baseline`
   exists) as a pre-commit hook **and** a CI step so this cannot recur.
4. Optionally purge history with `git filter-repo` (only matters once
   the key is revoked).

---

## 1. Repository structure audit

| Area | Finding | Recommendation |
|---|---|---|
| Scope | Two unrelated bots + 5 crypto strategies (funding arb, carry, regime-switch perp, BTC SMA, BB squeeze). ~73% of `src/` is crypto. `pyproject` description still says "Funding-rate arbitrage bot for Binance". | Move crypto code to a `legacy-crypto` branch/tag, then delete from `main`. Promote `src/ibkr_sentiment/` to top-level package `ibkr_bot/`. |
| Coupling | IBKR bot imports `src.logging_setup`, whose sibling `src.config` requires Binance settings; `ccxt`, `websockets` are hard deps. | IBKR package should have zero crypto imports; move `ccxt`, `python-telegram-bot` etc. into extras. |
| Repo hygiene | Tracked: `.agents/` (agent orchestration scratch), `ORIGINAL_REQUEST.md`, `sandbox_test.py`, `.env.save`. | Remove or move to `docs/history/`. |
| Naming | Bot is called "sentiment" but the proposed core is not sentiment-driven. | Rename to strategy-agnostic `ibkr_bot` with pluggable `strategies/`. |
| Deps | `ib_insync` has been unmaintained since 2024; the community fork is **`ib_async`** (same API). | Switch to `ib_async`. |
| Backtesting | **No backtester exists for the IBKR bot.** Every Binance strategy has one; the equities bot went straight to paper/live wiring. Acceptance gate #2 ("4-week paper Sharpe > 0.5") is statistically meaningless (≈20 daily observations). | Build an equities backtester before any further live work (§4). |
| Deploy | `deploy/ibkr_gateway/docker-compose.yml` brings up Redis + TimescaleDB + Qdrant — heavy infra for a 10-name universe. | SQLite/DuckDB + one IB Gateway container (with IBC for auto-login/restart) is enough. |
| CI | Runs ruff/pytest; no secret scan, no mypy gate. | Add `detect-secrets scan` + mypy on the IBKR package. |

---

## 2. IBKR bot defects (verified by reading code; #1 reproduced)

Severity: **H** = can lose money / block risk controls, **M** = wrong
numbers or silent degradation, **L** = hygiene.

| # | Sev | Location | Defect |
|---|---|---|---|
| 1 | H ✅ fixed | `risk/overlay.py` `check_target` | Gross/net caps are evaluated on the *whole basket* and the same verdict is applied to every delta — so one breach rejects **all** orders, including **closing** orders. Reproduced: 5 longs + closing an AAPL long → net 25% > 20% → AAPL close rejected. Risk-reducing orders must always pass. Also uses *target* notional on deltas instead of delta notional, and closes carry `notional=0`. |
| 2 | H ✅ fixed (daily stop; `trailing_stop_pct` / `max_orders_per_minute` still unused) | `risk/overlay.py`, `bot.py` | `daily_pnl` is never passed to `check_account` and `daily_anchor` is never set → **daily loss stop never fires**. `trailing_stop_pct` and `max_orders_per_minute` are config-only, unused. |
| 3 | H ✅ fixed | `execution/engine.py` | Account halt returns an error but **does not flatten or cancel open orders**; next tick repeats. `emergency_flatten` is never called by anything. |
| 4 | H ✅ fixed | `broker/ibkr.py` `positions()` | `mark_price = avgCost` → gross/net exposure, equity snapshots and orphan checks all use cost, not market. Use `ib.portfolio()` (has `marketPrice`, `unrealizedPNL`). |
| 5 | H ✅ fixed | `broker/ibkr.py` `place_order` | Returns immediately after submit; no fill tracking, no partial-fill handling, no timeout/cancel. `record_trade` persists the *submitted* snapshot as a trade. No reconciliation of open orders/executions on restart → duplicate orders after a crash. |
| 6 | H ✅ fixed | `execution/engine.py` | Only `MARKET` orders, no market-hours check. Loop ticks every 60s 24/7: orders queue overnight and fire at the open auction gap. |
| 7 | H ✅ fixed (no borrow-fee data in the TWS API; fee not checked) | shorting | No shortability / borrow-fee check (`genericTickList="236"` / `shortableShares`), no hard-to-borrow handling, no SSR (Rule 201) awareness. |
| 8 | M | `bot.py` tick | If there are **no signals**, the tick returns before execution, so positions are held indefinitely; if there are *some* signals, every position without a fresh signal is closed ("close — no signal in current cycle"). Holding period is therefore an accident of news flow, and a 4h rolling window forces heavy churn. |
| 9 | M ✅ fixed | `broker/ibkr.py` `quote` | `reqMktData` never cancelled → leaks market-data lines until the 100-line cap is hit. Contracts are never qualified (`qualifyContractsAsync`) — ambiguous symbols can route wrong. |
| 10 | M ✅ fixed | `bot.py` | Pulls 120 daily bars per signalled symbol **every tick** → burns historical-data pacing. Daily bars should be cached once per day. |
| 11 | M ✅ fixed | `dollar_neutral.py` | "Dollar-neutral" ≠ market-neutral: long NVDA/TSLA (β≈1.8) vs short XOM/UNH (β≈0.6) is net long beta. Sector trim uses `quantize(Decimal("1"))` (banker's rounding → can round *up* past cap). |
| 12 | M ✅ fixed | config | Reuters RSS feeds were discontinued in 2020 (silent zero items). SEC EDGAR needs a real contact UA. LLM model id is stale. |
| 13 | L ✅ fixed | `ExecutionEngine` | `DRY_RUN` still requires a full IB connection with trading permissions; use `readonly=True` for dry-run. |
| 14 | L ✅ fixed (FX: equity and loss stops are measured in USD, so for a non-USD base account EURUSD moves count toward the stops; USD bought on margin accrues interest) | general | No pre-trade margin check (`whatIfOrderAsync`), no PDT awareness, no FX handling for non-USD base accounts. |

**Conclusion:** the sentiment bot is a well-structured prototype but is
not safe for `live`, and its core edge is unvalidated (the repo's own
`SENTIMENT_VERDICT.md` found sentiment hurt the crypto trend bot, and
there is no historical news archive to backtest the LLM funnel).

---

## 3. Proposed strategy stack (IBKR stocks only)

Design principles, given a retail IBKR account:
- **Backtestable on free/cheap daily data** before a dollar goes live.
- **Low turnover** — IBKR commissions are cheap but spread + slippage
  are not; daily/weekly decisions, orders at the close or via
  Adaptive/MidPrice algos.
- **Long-biased or long/flat** — avoid dependence on shorting (borrow
  costs, recalls, margin) until the account justifies it.
- **Uncorrelated sleeves** combined by volatility targeting.

### Sleeve A — Trend-filtered cross-sectional momentum (core, ~60%)
- Universe: liquid large caps (S&P 500 or Nasdaq-100 members, price >
  $10, 20-day ADV > $20M). **Point-in-time constituents** to avoid
  survivorship bias.
- Signal: 12-1 month total return (skip last month), optionally
  risk-adjusted (return / 126-day vol).
- Portfolio: top 10–20 names, inverse-volatility weights, per-name cap
  10%, sector cap 30%.
- Regime filter: if the index (SPY or a UCITS equivalent) is below its
  200-day SMA, or 10-month SMA, move the sleeve to cash/T-bills.
  (Directly re-uses the SMA-filter tooling already in the repo.)
- Rebalance monthly, with a buffer rule (only replace a holding when it
  drops out of the top 2N) to cut turnover.
- Evidence: Jegadeesh–Titman (1993), Asness/Moskowitz/Pedersen (2013);
  the trend filter mainly reduces momentum-crash drawdowns (2009, 2020).

### Sleeve B — Short-term mean reversion in uptrends (satellite, ~25%)
- Universe: same large caps, only names above their 200-day SMA.
- Entry: RSI(2) < 10 or Internal Bar Strength < 0.2, entering with a
  MOC/LOC order at the close.
- Exit: close > 5-day SMA, or 5-day time stop; hard stop at 3×ATR.
- Max 5 concurrent positions; skip names with earnings in the next 3
  days.
- Low correlation to Sleeve A; historically strongest in volatile
  markets where momentum struggles. Edge has decayed since the 2010s,
  so treat it as unproven until the walk-forward (§4) confirms net of
  costs.

### Sleeve C — Post-earnings announcement drift (event, ~15%, phase 2)
- This is where the existing sentiment/LLM funnel earns its keep,
  **in a form that can be backtested**: earnings-surprise (SUE) plus
  gap-on-volume after the report is a documented anomaly with
  historical data.
- Long names with a positive surprise + gap up that holds day 1; hold
  20–40 trading days. The LLM classifies the call/8-K tone as a
  *confirming filter* only, forward-logged for ≥ 3 months before it
  may change sizing.

### Portfolio & risk layer (shared)
- Target portfolio volatility 10–12% annualised; scale gross exposure
  down when realised vol rises; gross ≤ 100% (no margin) at launch.
- Account kill switches: daily −2%, rolling-peak drawdown −15% →
  flatten + halt + alert; manual re-arm only.
- Per-order: max 1% of 20-day ADV, price-band check vs last close,
  `whatIfOrderAsync` margin check.
- Risk-reducing orders **always** allowed through (fixes defect #1).

### Keep the L/S sentiment bot?
Keep it as **research only** (forward-logging signals to the DB, no
orders) until it has ≥ 6 months of out-of-sample logs showing a
positive information coefficient. If it proves out, re-enter as a
beta-neutral (not dollar-neutral) overlay.

---

## 4. Engineering roadmap

**Phase 0 — Safety (1–2 days)**
- Revoke/scrub secret (§0); secret scan in CI.
- ✅ Fix defects #1, #2, #3 with regression tests (done: `check_basket`
  replaces `check_target`; daily anchor persisted via equity snapshots;
  latched halt cancels orders and flattens once).
- Set IBKR `mode: paper`
  guard that refuses `live` unless an acceptance-gate file is present.

**Phase 1 — Restructure (≈1 week)**
- Tag `legacy-crypto`, remove crypto code from `main`.
- New layout:
  ```
  ibkr_bot/
    broker/      ib_async adapter, paper broker, fill tracker, reconciler
    data/        bar cache (Parquet/DuckDB), calendars, universe (PIT)
    strategies/  base.py (target-weights interface), momentum.py,
                 mean_reversion.py, pead.py, sentiment_research.py
    portfolio/   combiner, vol targeting, constraints
    risk/        pre-trade checks, kill switches
    execution/   order planner (MOC/LOC/Adaptive), scheduler (market hours)
    backtest/    engine, IBKR cost model, walk-forward, reports
    state/  monitoring/  config.py  main.py
  ```
- Single strategy interface: `target_weights(asof, data) -> dict[str, float]`,
  used identically by backtester and live runner (no divergent code
  paths).

**Phase 2 — Backtester (≈1–2 weeks)**
- Daily vectorised engine on adjusted OHLCV + dividends.
- IBKR Tiered cost model: $0.0035/share (min $0.35, max 1% of value) +
  exchange/regulatory fees + modelled slippage (e.g. 5 bps + ½ spread);
  borrow fees for any shorts.
- Walk-forward (e.g. 5y train / 1y test, rolling), deflated Sharpe,
  turnover, capacity; results stored and diffed in CI.
- Acceptance gates to replace the current ones: OOS Sharpe > 0.7 net,
  max DD < 25%, profitable in ≥ 2 of 3 regimes (2008–09, 2020, 2022),
  parameter-stability heatmap without a lone peak.

**Phase 3 — Live plumbing (≈1–2 weeks)**
- `ib_async` adapter: contract qualification, `portfolio()` marks, event-
  driven fill tracking, open-order/execution reconciliation on start-up,
  cancelled market-data subscriptions, delayed data fallback.
- Scheduler on `exchange_calendars` (NYSE): compute signals after the
  close from cached bars → submit MOC/LOC or next-open Adaptive orders;
  nothing runs when the market is shut.
- IB Gateway via `gnzsnz/ib-gateway` Docker image (bundles IBC for
  auto-login and the daily restart); health check + Telegram alerts
  (reuse `src/monitoring/telegram_bot.py`).

**Phase 4 — Paper → live**
- ≥ 3 months IBKR paper trading with daily reconciliation of backtest-
  predicted vs actual fills (slippage tracking).
- Go live at 25% of target capital, scale up after 3 clean months.

---

## 5. Account & jurisdiction checks (confirm for your account)

- **EU retail & PRIIPs:** if the account is with an EU IBKR entity and
  classed as retail, US-domiciled ETFs (SPY, SGOV, XLK…) cannot be
  bought. Use UCITS equivalents (e.g. CSPX/VUAA for the regime filter,
  an ultra-short treasury UCITS for cash) — individual US stocks are
  unaffected.
- **Pattern Day Trader rule** (US margin accounts < $25k): Sleeve A and
  C are fine; Sleeve B's 1–5 day holds can hit the limit. Check the
  current FINRA rule status for your account before relying on it.
- **Tax (if French resident):** declare the foreign IBKR account on form
  3916; file a W-8BEN so US dividend withholding is 15%; high turnover
  (Sleeve B) generates many taxable events — the existing
  `scripts/tax_export.py` should be ported to IBKR Flex Query exports.
- **Market data:** live streaming US quotes need IBKR market data
  subscriptions; the proposed design only needs end-of-day bars, which
  keeps data cost near zero.
