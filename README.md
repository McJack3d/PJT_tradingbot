# trad-bot

Independent trading bots in one repository:

1. **Binance funding-rate arbitrage bot** — delta-neutral cash-and-carry
   on Binance: long spot + short perp in equal notional, harvest funding
   every 8h. Specified in `docs/trading_bot_spec_v1.md`.
2. **IBKR sentiment bot** — long/short US equities on Interactive
   Brokers driven by a two-stage sentiment funnel (FinBERT → LLM
   gatekeeper) with a dollar-neutral overlay. Specified in
   `docs/IBKR_SENTIMENT.md`.
3. **Small-cap crypto sentiment bot** — long/flat low-cap Binance spot
   pairs driven by the same FinBERT → LLM funnel applied to crypto news.
   Paper-default research scaffold. Specified in `docs/CRYPTO_SENTIMENT.md`.

The two bots share no runtime state and can be deployed independently.
This README covers the Binance bot; the IBKR sentiment bot is
documented in `docs/IBKR_SENTIMENT.md`.

## Status

v1 — pre-deployment. The strategy, risk, execution, and reconciliation
modules are implemented. Backtester is event-driven and runs against
local Parquet history. Telegram and email monitoring wired up. Live
trading **only after** the §15 acceptance gates in the spec are met.

## Layout

```
src/
  adapters/        Exchange adapter base + Binance (ccxt) + FakeExchange
  data/            Live MarketData (Binance WS + REST fallback) + history
  strategy/        Funding-arb signal engine
  risk/            Pre-trade checks + continuous monitoring + kill switch
  execution/       Two-leg order coordination with idempotency
  state/           SQLite schema and DAOs (async via SQLAlchemy)
  reconciliation/  Internal-vs-exchange state diff loop
  funding/         Funding-payment poller (records to DB)
  monitoring/      Telegram bot, email, daily/weekly digest, dashboard
  backtest/        Event-driven backtester + walk-forward + metrics
  config.py        Pydantic-validated settings (env + YAML)
  killswitch.py    /var/lib/bot/KILL file watcher
  main.py          Entry point (asyncio)
  ibkr_sentiment/  SECOND BOT — see docs/IBKR_SENTIMENT.md
config/            live.yaml / paper.yaml / backtest.yaml / ibkr_sentiment.yaml
scripts/           Download history, backtest, dry-run, tax export,
                   run_ibkr_sentiment
tests/             Unit (incl. adversarial risk-manager, e2e paper,
                   IBKR sentiment funnel/risk/basket/e2e)
deploy/            systemd unit + Ubuntu 24.04 setup script;
                   ibkr_gateway/ docker-compose for IB Gateway + Redis
                   + TimescaleDB + Qdrant
.github/           CI workflow
```

## Modes

- `backtest` — replays Parquet history through the strategy/risk engines.
- `paper`    — full pipeline against `FakeExchange` (no real orders).
- `dry_run`  — production VPS deployment; engine intercepts every order
               submission and logs instead of sending.
- `live`     — real money. Only enable after the §15 acceptance gates pass.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in
```

## Run modes

### Daemon — funding-rate arbitrage (24/7 on a VPS)

```bash
# Backtest
python -m scripts.run_backtest --config config/backtest.yaml

# Dry-run (full logic, order submission intercepted and logged)
python -m scripts.run_dry_run --config config/paper.yaml

# Live (only after acceptance gates pass)
BOT_ENV=live python -m src.main --config config/live.yaml
```

### Laptop bot — unified TradBot menu (run when laptop is on)

```bash
# Build the double-clickable TradBot.app launcher (macOS):
tradbot install-app
# Double-click ~/Applications/TradBot.app → top-level picker:
#   1. Binance trend bot   2. IBKR sentiment bot   0. Quit
```

The same menu is reachable from the terminal:

```bash
tradbot menu
```

Or drive either bot directly with subcommands:

```bash
# Binance trend bot (BTC SMA)
tradbot status                       # current state
tradbot evaluate                     # tick once
tradbot watch                        # live status loop

# IBKR sentiment bot
tradbot ibsent-status
tradbot ibsent-tick
tradbot ibsent-watch
```

Streamlit dashboard for the BTC trend bot (paper mode, no keys needed):

```bash
streamlit run src/app/streamlit_app.py
# → opens http://localhost:8501
# Big buttons: Start trading / Stop trading / Evaluate now / Flatten to USDT
```

Live mode (real Binance, real money — only with a no-withdrawal IP-whitelisted key):

```bash
SIMPLE_BOT_LIVE=true \
BINANCE_API_KEY=... \
BINANCE_API_SECRET=... \
streamlit run src/app/streamlit_app.py
```

Strategy: hold BTC when daily close > 50-day SMA, else hold USDT. Spot
only, no leverage. Closing the terminal stops the bot. Existing
positions stay on Binance until you flatten or restart and let the
signal decide.

### IBKR sentiment bot (separate)

Quickstart in paper mode (no IB connection, no LLM key needed):

```bash
python -m scripts.run_ibkr_sentiment --config config/ibkr_sentiment.yaml
```

For dry-run / live, start IB Gateway via docker compose and add the
appropriate `[ibkr,llm,redis]` extras. See `docs/IBKR_SENTIMENT.md`.

## Tests

```bash
pytest                          # unit tests (Binance + IBKR sentiment)
pytest -m integration           # integration (needs testnet keys)
pytest tests/unit/test_ibkr_sentiment_*.py   # IBKR sentiment only
```

## Safety

- No-withdrawal API keys. IP-whitelist on Binance side.
- Risk Manager has unilateral authority to flatten and halt.
- Three kill paths: Telegram `/kill`, VPS file `/var/lib/bot/KILL`,
  revoke the API key on Binance.
- Reconciliation every 60s; halt if stale > 5min.

## Deploy

See `deploy/setup.sh` and `deploy/systemd/bot.service`. Target host is
Hetzner CAX11 (ARM, Singapore) or DO Singapore. Ubuntu 24.04.

Works under construction, MIT license.
