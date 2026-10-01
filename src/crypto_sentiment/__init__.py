"""Small-cap crypto sentiment day-trading bot.

A research bot that screens low-cap Binance spot pairs for news/social
sentiment and goes long the names with a strong, near-term, credible
bullish read — reusing the two-stage funnel (FinBERT screen → LLM
gatekeeper → per-symbol aggregation) already built for the IBKR
sentiment bot.

Honest priors, stated up front (consistent with src/sentiment/*):
  * Sentiment on the majors is heavily priced-in. The only place it
    plausibly carries *independent* edge is low-cap alts — which is
    also where the data is thinnest, the spreads widest, and the social
    signal most manipulated (pump groups, paid shills, bots).
  * Spot-only: the bot can be long or flat, never short. It cannot
    profit from the bad-news names, only avoid them.
  * Fees + small-cap spreads (often 1-2%) mean this is only viable once
    a real edge is demonstrated in PAPER mode and the account is funded
    properly. The default mode is therefore PAPER.

This package writes no orders to a real exchange unless explicitly run
in `mode="live"` against a live adapter.
"""

from __future__ import annotations

from src.crypto_sentiment.bot import CryptoSentimentBot, TickReport
from src.crypto_sentiment.config import CryptoSentimentConfig
from src.crypto_sentiment.universe import MarketInfo, build_universe

__all__ = [
    "CryptoSentimentBot",
    "TickReport",
    "CryptoSentimentConfig",
    "MarketInfo",
    "build_universe",
]
