"""Wiring helpers: build the funnel pipeline and a ready-to-run bot.

Kept separate from `bot.py` so the bot itself has no opinion about which
FinBERT/LLM/exchange backends exist — it just takes the pieces. This is
also the only place that decides paper-vs-live, so that decision is
auditable in one spot.
"""

from __future__ import annotations

from decimal import Decimal

from src.crypto_sentiment.bot import CryptoSentimentBot
from src.crypto_sentiment.config import CryptoSentimentConfig
from src.crypto_sentiment.feeds import CryptoNewsGatherer
from src.crypto_sentiment.positions import PositionStore
from src.crypto_sentiment.universe import CcxtMarketProvider
from src.ibkr_sentiment.sentiment.llm_gatekeeper import build_gatekeeper
from src.ibkr_sentiment.sentiment.pipeline import PipelineConfig, SentimentPipeline


def build_pipeline(
    cfg: CryptoSentimentConfig,
    *,
    anthropic_key: str = "",
    openai_key: str = "",
) -> SentimentPipeline:
    """FinBERT screen + LLM gatekeeper, wired from config.

    Uses the stub FinBERT scorer unless the real `transformers` model is
    importable; the gatekeeper backend follows `cfg.llm_provider`.
    """
    try:
        from src.ibkr_sentiment.sentiment.finbert import FinBertScorer
        scorer = FinBertScorer()
    except Exception:  # noqa: BLE001 - transformers/torch not installed
        from src.ibkr_sentiment.sentiment.finbert import StubFinBertScorer
        scorer = StubFinBertScorer()

    gatekeeper = build_gatekeeper(
        cfg.llm_provider, anthropic_key=anthropic_key, openai_key=openai_key
    )
    pcfg = PipelineConfig(
        min_conviction=cfg.min_conviction,
        signal_window=cfg.signal_window,
    )
    return SentimentPipeline(scorer=scorer, gatekeeper=gatekeeper, cfg=pcfg)


async def build_live_bot(
    cfg: CryptoSentimentConfig,
    *,
    binance_api_key: str,
    binance_api_secret: str,
    anthropic_key: str = "",
    openai_key: str = "",
    cryptopanic_token: str = "",
    store_path: str = "data/crypto_sentiment.json",
    testnet: bool = False,
) -> CryptoSentimentBot:
    """Construct a bot against the REAL Binance spot adapter.

    Integration-only path; the unit suite builds the bot directly with a
    FakeExchange. Importing the live adapter is deferred so paper runs
    and tests don't require ccxt credentials.
    """
    import httpx

    from src.adapters.binance import BinanceAdapter

    exchange = BinanceAdapter(
        api_key=binance_api_key, api_secret=binance_api_secret, testnet=testnet
    )
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
        news=CryptoNewsGatherer(
            list(cfg.rss_feeds), fetcher=_fetch, cryptopanic_token=cryptopanic_token
        ),
        store=PositionStore(store_path),
        cfg=cfg,
    )


def per_position_for_capital(total_usd: Decimal, max_concurrent: int) -> Decimal:
    """Convenience: even split of capital across concurrent slots,
    floored at Binance's ~$5 min notional."""
    if max_concurrent <= 0:
        return Decimal("0")
    return max(Decimal("5"), total_usd / max_concurrent)
