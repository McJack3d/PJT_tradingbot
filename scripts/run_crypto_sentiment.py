"""Run the small-cap crypto sentiment bot.

Paper mode (default) needs no keys and places no real orders — it pulls
real Binance prices through a paper adapter and logs what it *would* do:

    python -m scripts.run_crypto_sentiment

A single read-only evaluation (news → signals, no trading):

    python -m scripts.run_crypto_sentiment --once

Live mode is a deliberate opt-in and requires real Binance keys
(spot trading enabled, no withdrawal, IP-whitelisted) plus, ideally, an
LLM key and a CryptoPanic token in the environment:

    CRYPTO_SENTIMENT_LIVE=true python -m scripts.run_crypto_sentiment

Env knobs (all optional):
    CRYPTO_SENTIMENT_LIVE          true → real Binance spot (default false)
    CRYPTO_SENTIMENT_CAPITAL_USD   total capital to spread across slots
    CRYPTO_SENTIMENT_MAX_POS       max concurrent positions (default 2)
    CRYPTO_SENTIMENT_LLM           stub | anthropic | openai (default stub)
    CRYPTOPANIC_TOKEN              CryptoPanic free auth token
    ANTHROPIC_API_KEY / OPENAI_API_KEY
    BINANCE_API_KEY / BINANCE_API_SECRET   (live only)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from src.crypto_sentiment.config import CryptoSentimentConfig
from src.crypto_sentiment.factory import build_pipeline, per_position_for_capital
from src.logging_setup import configure_logging


def _cfg_from_env() -> CryptoSentimentConfig:
    cfg = CryptoSentimentConfig()
    cfg.mode = "live" if os.environ.get("CRYPTO_SENTIMENT_LIVE", "false").lower() == "true" else "paper"
    cfg.llm_provider = os.environ.get("CRYPTO_SENTIMENT_LLM", "stub")
    cfg.max_concurrent_positions = int(os.environ.get("CRYPTO_SENTIMENT_MAX_POS", "2"))
    capital = os.environ.get("CRYPTO_SENTIMENT_CAPITAL_USD")
    if capital:
        cfg.per_position_usd = per_position_for_capital(
            Decimal(capital), cfg.max_concurrent_positions
        )
    cfg.cryptopanic_enabled = bool(os.environ.get("CRYPTOPANIC_TOKEN"))
    return cfg


async def _build(cfg: CryptoSentimentConfig):
    """Construct the bot for the chosen mode."""
    store_path = os.environ.get("CRYPTO_SENTIMENT_STORE", "data/crypto_sentiment.json")
    cryptopanic = os.environ.get("CRYPTOPANIC_TOKEN", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if cfg.mode == "live":
        from src.crypto_sentiment.factory import build_live_bot

        key = os.environ.get("BINANCE_API_KEY", "")
        secret = os.environ.get("BINANCE_API_SECRET", "")
        if not key or not secret:
            raise SystemExit("CRYPTO_SENTIMENT_LIVE=true but BINANCE_API_KEY/SECRET missing.")
        return await build_live_bot(
            cfg, binance_api_key=key, binance_api_secret=secret,
            anthropic_key=anthropic_key, openai_key=openai_key,
            cryptopanic_token=cryptopanic, store_path=store_path,
            testnet=os.environ.get("BINANCE_TESTNET", "false").lower() == "true",
        )

    # Paper: real public prices, fake balances, no orders sent.
    import httpx

    from src.adapters.paper_binance import PaperBinanceAdapter
    from src.crypto_sentiment.bot import CryptoSentimentBot
    from src.crypto_sentiment.feeds import CryptoNewsGatherer
    from src.crypto_sentiment.positions import PositionStore
    from src.crypto_sentiment.universe import CcxtMarketProvider

    exchange = PaperBinanceAdapter(starting_usdt=Decimal("1000"), quote_asset="USDT")
    await exchange.connect()

    async def _fetch(url: str) -> str:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": "trad-bot/1.0"})
            resp.raise_for_status()
            return resp.text

    return CryptoSentimentBot(
        exchange=exchange,
        pipeline=build_pipeline(cfg, anthropic_key=anthropic_key, openai_key=openai_key),
        market_provider=CcxtMarketProvider(exchange.spot),
        news=CryptoNewsGatherer(list(cfg.rss_feeds), fetcher=_fetch,
                                cryptopanic_token=cryptopanic),
        store=PositionStore(store_path),
        cfg=cfg,
    )


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Small-cap crypto sentiment bot")
    parser.add_argument("--once", action="store_true",
                        help="Run a single read-only evaluation and exit.")
    args = parser.parse_args()

    configure_logging()
    cfg = _cfg_from_env()
    bot = await _build(cfg)

    if args.once:
        signals = await bot.evaluate()
        if not signals:
            print("No in-universe sentiment signals right now.")
        for s in sorted(signals, key=lambda x: x.score * x.conviction, reverse=True):
            print(f"{s.symbol:>8}  score={s.score:+.2f}  conv={s.conviction:.2f}  "
                  f"horizon={s.temporal_impact}  sources={','.join(s.sources)}")
        return 0

    print(f"crypto-sentiment bot starting in {cfg.mode.upper()} mode "
          f"(per-position ${cfg.per_position_usd}, max {cfg.max_concurrent_positions}). "
          "Ctrl+C to stop.")
    try:
        await bot.run_loop()
    except KeyboardInterrupt:
        bot.stop()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
