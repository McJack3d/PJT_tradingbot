# Small-cap crypto sentiment bot

A research bot (`src/crypto_sentiment/`) that screens low-cap Binance
spot pairs for news/social sentiment and goes **long the names with a
strong, near-term, credible bullish read** — reusing the two-stage
sentiment funnel built for the IBKR bot (FinBERT screen → LLM gatekeeper
→ per-symbol aggregation).

## Honest status — read first

This is **research scaffolding, defaulted to paper**. It is *not* a
proven money-maker, and three things make small-cap sentiment day
trading hard:

1. **Edge is unproven and probably small.** Sentiment on majors is
   priced-in; small-caps are the only place it *might* carry independent
   signal — and that's exactly where the data is thinnest and most
   **manipulated** (pump groups, paid shills, bots). An LLM faithfully
   scoring shill content is garbage-in-garbage-out. The gatekeeper rates
   source credibility and flags structural-vs-transient, which helps but
   doesn't make the signal clean.
2. **Spot only.** The bot is long or flat, never short. It can avoid
   bad-news names but can't profit from them.
3. **Fees + spreads.** Binance spot taker is 0.1%/side; small-cap
   spreads are often 1–2%. Round-trip cost can be 1.2–2%+, so the
   strategy needs a real per-trade edge above that just to break even.
   On tiny capital ($20–50) this is a "does the pipeline fire" test, not
   a strategy. Validate in paper, then fund properly.

The decisive question — *does sentiment add anything?* — is answered by
running it in **paper** and comparing realized behaviour, the same way
`src/backtest/sentiment_ab.py` does for the trend bot.

## How it works

```
universe (24h-volume small-cap filter)
   │  CcxtMarketProvider
   ▼
crypto news  ── CryptoPanic (per-coin tags) + RSS ──┐
   │  CryptoNewsGatherer                            │
   ▼                                                │
SentimentPipeline  (FinBERT screen → LLM gatekeeper)│
   │  per-symbol StructuredSignal(score, conviction,│
   ▼   horizon, structural)                         │
CryptoSentimentBot.tick()                           │
   ├─ EXIT held names: reversal / take-profit / stop-loss / time-stop
   └─ ENTER best bullish candidates, gated by:
        conviction · allowed horizon · spread · cooloff ·
        max concurrent · quote balance · daily-loss stop
```

Positions are tracked in a small JSON store (`PositionStore`) — entry
price/time, per-asset cooloff, and a daily realized-PnL tally that halts
new entries past the loss stop. State survives restarts.

## Run it

### Paper (default — no keys, no orders, real prices)

```bash
cd ~/TRAD_BOT && source .venv/bin/activate

# one-shot: what does sentiment say right now? (read-only)
python -m scripts.run_crypto_sentiment --once

# run the loop (paper) — logs what it WOULD trade
python -m scripts.run_crypto_sentiment
```

Add a CryptoPanic free token for far better small-cap coverage (its
posts are tagged per-coin, unlike scraping tickers from RSS text):

```bash
export CRYPTOPANIC_TOKEN=...        # free tier at cryptopanic.com/developers
export CRYPTO_SENTIMENT_LLM=anthropic   # use Claude for the gatekeeper (costs $)
export ANTHROPIC_API_KEY=...
python -m scripts.run_crypto_sentiment
```

With `llm=stub` (default) the gatekeeper is a deterministic keyword
mirror — fine for plumbing, useless as a real signal. Use `anthropic`
or `openai` for an actual reasoning gate.

### Live (deliberate opt-in — real money)

Only after paper shows an edge and the account is funded. Requires a
Binance key with **spot trading enabled, withdrawals OFF, IP-whitelisted**.

```bash
export CRYPTO_SENTIMENT_LIVE=true
export CRYPTO_SENTIMENT_CAPITAL_USD=200     # split across slots, floored at $5/name
export CRYPTO_SENTIMENT_MAX_POS=2
export CRYPTO_SENTIMENT_LLM=anthropic
# BINANCE_API_KEY / BINANCE_API_SECRET / ANTHROPIC_API_KEY / CRYPTOPANIC_TOKEN in .env
python -m scripts.run_crypto_sentiment
```

## Config

All knobs live in `CryptoSentimentConfig` (`src/crypto_sentiment/config.py`):
universe volume band, entry/exit score, min conviction, allowed horizons,
per-position size, max concurrent, spread guard, take-profit/stop-loss,
time-stop, daily-loss stop, cooloff, poll interval, feeds, LLM provider.

## Tests

```bash
pytest tests/unit/test_crypto_sentiment.py -q
```

Deterministic and network-free: stub FinBERT + stub LLM gatekeeper,
canned CryptoPanic JSON through an injected fetcher, `FakeExchange` for
fills. They pin the bot's *decisions* (universe filter, entry, exits,
spread/cooloff/max-concurrent/daily-stop guards), not the model's skill.

## What's deliberately not here (v1)

- No backtest of the crypto sentiment signal (no clean historical
  small-cap news corpus; paper-forward is the honest test).
- RSS ticker detection is noisy for short symbols — CryptoPanic's
  structured per-coin tags are the primary source.
- No shorting (spot only). No leverage.
