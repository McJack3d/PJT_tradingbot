# IBKR Sentiment Bot

A second, fully independent bot that lives in this repository alongside
the Binance funding-arb daemon. It trades US equities through
Interactive Brokers (IB Gateway + ib_insync) driven by a multi-stage
sentiment funnel.

The Binance bot is untouched — both bots can run side by side, share
no code, and have no shared runtime state.

## Architecture

```
                    raw text (RSS, SEC, transcripts)
                                 │
                                 ▼
                ┌──────────────────────────────┐
                │  Stage 1: FinBERT (fast)     │  filters noise,
                │  discriminative polarity     │  surfaces high-conviction
                └──────────────┬───────────────┘  outliers only
                               │
                  high-conviction items only
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Stage 2: LLM gatekeeper     │  chain-of-thought,
                │  (Anthropic / OpenAI /       │  evaluates credibility,
                │   FinGPT)                    │  temporal impact,
                └──────────────┬───────────────┘  structural vs. transient
                               │
                  structured per-symbol score
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Signal engine               │  composite weighted
                │  (rolling window aggregator) │  by conviction,
                │                              │  credibility, source
                │                              │  accuracy, recency
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Technical confirmation      │  SMA + RSI guards
                │  (per-symbol)                │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Dollar-neutral basket       │  long high-score names,
                │  builder (long/short equity) │  short low-score names,
                │                              │  sector + gross + net
                │                              │  caps enforced
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Risk overlay (pre-trade +   │
                │  continuous drawdown stops)  │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  IBKR adapter (ib_insync)    │  Redis-backed
                │  IB Gateway, headless docker │  rate limiter
                └──────────────────────────────┘
```

## Code layout

```
src/ibkr_sentiment/
  config.py              Pydantic settings tree (env + YAML)
  bot.py                 IbkrSentimentBot — top-level orchestrator
  main.py                CLI entry point
  sentiment/
    models.py            dataclasses (NewsItem, FinBertScore, LLMVerdict,
                                       StructuredSignal)
    ingestion.py         RSS / Atom polling + dedup + symbol tagging
    finbert.py           FinBertScorer (real) + StubFinBertScorer (tests)
    llm_gatekeeper.py    Anthropic / OpenAI / FinGPT backends + Stub
    pipeline.py          two-stage funnel + rolling aggregator
    vector_store.py      InMemory / Qdrant / Chroma (optional RAG)
  broker/
    base.py              abstract Broker + dataclasses
    paper.py             in-memory paper broker (tests, paper mode)
    ibkr.py              ib_insync wrapper (lazy import)
    rate_limiter.py      token bucket + Redis-backed limiter
  signal_engine/
    technical.py         SMA + Wilder RSI
    mapping.py           sentiment → LONG/SHORT/FLAT decisions
    dollar_neutral.py    basket builder + diff_targets
  execution/
    engine.py            ExecutionEngine (risk-gated order placement)
  risk/
    overlay.py           pre-trade and account-level checks
  state/
    models.py            SQLAlchemy models (works on Postgres OR SQLite)
    db.py                async DAOs

config/ibkr_sentiment.yaml   default paper config
scripts/run_ibkr_sentiment.py CLI shim
deploy/ibkr_gateway/         docker-compose for IB Gateway + Redis +
                             TimescaleDB + Qdrant
tests/unit/test_ibkr_sentiment_*.py   unit tests
```

## Modes

| mode | broker | LLM | use case |
| --- | --- | --- | --- |
| `paper` (default) | in-memory `PaperBroker` | stub (deterministic) | quickstart, CI, full-pipeline smoke test |
| `dry_run` | live IB Gateway connection | configured | full read path against IB, orders intercepted |
| `live` | live IB Gateway connection | configured | real money, only after acceptance gates pass |

## Install

```bash
# core (gives you paper mode with stubs)
pip install -e ".[dev]"

# add what you need for real running:
pip install -e ".[ibkr]"        # ib_insync
pip install -e ".[sentiment]"   # FinBERT (transformers + torch)
pip install -e ".[llm]"         # anthropic + openai SDKs
pip install -e ".[redis]"       # shared rate-limit bucket
pip install -e ".[vector]"      # qdrant-client + chromadb
pip install -e ".[postgres]"    # asyncpg

# or everything:
pip install -e ".[dev,ibkr,sentiment,llm,redis,vector,postgres]"
```

## Run

Three equivalent entry points — pick whichever fits your workflow.

```bash
# 1. As a long-running daemon (recommended for VPS deploys)
python -m scripts.run_ibkr_sentiment --config config/ibkr_sentiment.yaml
# or, with the installed script:
ibkr-sentiment-bot --config config/ibkr_sentiment.yaml

# 2. Via the unified `tradbot` CLI (handy on a laptop / from the menu)
tradbot ibsent-status        # account + open positions
tradbot ibsent-tick          # one full evaluation cycle
tradbot ibsent-watch         # continuous loop, Ctrl+C to stop
tradbot ibsent-news --title "AAPL beats record growth" --symbol AAPL
tradbot ibsent-signals       # last 20 structured signals
tradbot ibsent-trades        # last 20 trades
tradbot ibsent-equity        # equity history
tradbot ibsent-flatten --yes # emergency close all
tradbot ibsent-config        # show resolved config
# Set the mode without editing YAML:
IBSENT_MODE=dry_run tradbot ibsent-tick

# 3. From the macOS .app launcher
tradbot install-app          # creates ~/Applications/TradBot.app
# Double-click TradBot.app → opens Terminal → top-level bot picker:
#   1. Binance trend bot
#   2. IBKR sentiment bot
```

The `.app` is a thin launcher (no Python is bundled) — it always runs
the current code, so `git pull` is enough to update it.

For dry-run / live, start IB Gateway first:

```bash
cd deploy/ibkr_gateway
cp .env.example .env             # fill in TWS_USERID/PASSWORD
docker compose up -d ib-gateway redis postgres
```

Then in `config/ibkr_sentiment.yaml`:

```yaml
mode: dry_run                    # or live, after acceptance
llm:
  provider: anthropic            # or openai
rate_limit:
  redis_url: redis://127.0.0.1:6379/0
db_url: postgresql+asyncpg://ibsent:PASSWORD@127.0.0.1:5432/ibsent
```

## Secrets

The bot reads from `.env` (see `.env.example`):

| Variable | Used by |
| --- | --- |
| `IBKR_ACCOUNT` | IbkrBroker (optional — only needed for multi-account logins) |
| `ANTHROPIC_API_KEY` | LLM gatekeeper (provider=anthropic) |
| `OPENAI_API_KEY` | LLM gatekeeper (provider=openai/fingpt) |
| `SEC_USER_AGENT` | SEC EDGAR 8-K feed: `"<name> <contact email>"` (required by the SEC; overrides `ingestion.sec_user_agent`) |
| `REDIS_URL` | rate limiter |
| `POSTGRES_URL` | optional override for `db_url` |
| `QDRANT_URL` / `QDRANT_API_KEY` | vector store |

## Tests

```bash
pytest tests/unit/test_ibkr_sentiment_*.py
```

The suite runs without any of the optional extras — the stub FinBERT
scorer, stub LLM gatekeeper, paper broker, in-memory rate limiter,
and SQLite database stand in for the real services.

## Acceptance gates (read before live)

The Binance bot has §15 acceptance gates in `docs/trading_bot_spec_v1.md`.
The sentiment bot inherits the same discipline:

1. Stage 1 stub-vs-real score correlation > 0.7 on a labelled
   evaluation set.
2. End-to-end paper run for ≥ 4 weeks with the configured universe,
   verified Sharpe > 0.5 net of fees + slippage.
3. Source accuracy tracker has ≥ 30 closed trades per source before
   that source's weight diverges from the 0.5 prior.
4. Live IB connection survives 24h with no rate-limit violations
   observed in the logs.
5. Daily and cumulative drawdown stops have been verified to fire in a
   simulated drawdown test.
6. Dollar-neutral overlay keeps `|net| / equity ≤ 0.20` over the
   four-week paper run.

Only after all six pass does `mode: live` become a sensible choice.

## How the funnel saves money

A naive design routes every article through the LLM. At 50k articles a
day and even a cheap LLM that's $200+/day in API spend. The funnel:

* drops ~95% of items in Stage 1 (FinBERT) for free
* batches the remaining items to the LLM with concurrency cap
* re-uses verdicts over a rolling window so the same item doesn't get
  re-analyzed every tick

In production, Stage 2 typically processes 1-2% of ingested items.

## How the dollar-neutral overlay isolates idiosyncratic risk

When the LLM produces a basket of bullish names AND a basket of
bearish names, the basket builder sizes each leg to ~50% of the
configured gross budget. This means a broad market drop hurts the
long leg but symmetrically helps the short leg. What's left is the
*idiosyncratic* component — the part that's actually about the news,
not the market. The `max_net_exposure_pct` cap (default 20% of NLV)
prevents the bot from drifting into a directional bet when one side
of the LLM output is thin.
