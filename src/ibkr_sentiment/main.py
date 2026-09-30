"""CLI entry point for the IBKR sentiment bot.

Spins up the bot in the requested mode and drives the main tick loop.
PAPER mode uses an in-process paper broker; DRY_RUN and LIVE both
connect to IB Gateway through ib_insync, with DRY_RUN intercepting
order placement.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import timedelta
from pathlib import Path

from src.ibkr_sentiment.bot import IbkrSentimentBot, build_default_bot
from src.ibkr_sentiment.broker.base import Broker
from src.ibkr_sentiment.broker.paper import PaperBroker
from src.ibkr_sentiment.config import (
    IbkrMode,
    IbkrSecrets,
    IbkrSentimentConfig,
)
from src.ibkr_sentiment.execution.engine import ExecutionEngine
from src.ibkr_sentiment.risk.overlay import RiskOverlay
from src.ibkr_sentiment.sentiment.finbert import (
    FinBertScorer,
    Scorer,
    StubFinBertScorer,
)
from src.ibkr_sentiment.sentiment.ingestion import IngestionService
from src.ibkr_sentiment.sentiment.llm_gatekeeper import build_gatekeeper
from src.ibkr_sentiment.sentiment.pipeline import PipelineConfig, SentimentPipeline
from src.logging_setup import configure_logging, log


def _build_scorer(cfg: IbkrSentimentConfig) -> Scorer:
    """Use real FinBERT when the transformers extra is installed,
    otherwise fall back to the deterministic stub. Production deploys
    should install the `sentiment` extra explicitly so the real model
    is loaded."""
    try:
        return FinBertScorer(
            model_name=cfg.finbert.model_name,
            device=cfg.finbert.device,
            max_input_chars=cfg.finbert.max_input_chars,
            batch_size=cfg.finbert.batch_size,
        )
    except ImportError:
        log.warning(
            "ibkr_sentiment.finbert.using_stub",
            reason="transformers not installed; using StubFinBertScorer",
        )
        return StubFinBertScorer(max_input_chars=cfg.finbert.max_input_chars)


async def _build_broker(cfg: IbkrSentimentConfig) -> Broker:
    if cfg.mode == IbkrMode.PAPER:
        broker = PaperBroker(starting_cash=cfg.risk.starting_equity_usd)
        await broker.connect()
        return broker
    from src.ibkr_sentiment.broker.ibkr import IbkrBroker

    secrets = IbkrSecrets()
    return IbkrBroker(
        host=cfg.ibkr.host,
        port=cfg.ibkr.port,
        client_id=cfg.ibkr.client_id,
        account=cfg.ibkr.account or secrets.ibkr_account or None,
        readonly=cfg.ibkr.readonly,
        connect_timeout_s=cfg.ibkr.connect_timeout_s,
        redis_url=cfg.rate_limit.redis_url or secrets.redis_url or None,
        orders_per_minute=cfg.rate_limit.orders_per_minute,
        historical_requests_per_10min=cfg.rate_limit.historical_requests_per_10min,
        market_data_lines=cfg.rate_limit.market_data_lines,
    )


def _build_bot(cfg: IbkrSentimentConfig, broker: Broker) -> IbkrSentimentBot:
    secrets = IbkrSecrets()
    scorer = _build_scorer(cfg)
    gatekeeper = build_gatekeeper(
        cfg.llm.provider,
        anthropic_key=secrets.anthropic_api_key,
        openai_key=secrets.openai_api_key,
        model=cfg.llm.model,
        max_concurrent=cfg.llm.max_concurrent,
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
        request_timeout_s=cfg.llm.request_timeout_s,
    )
    pipeline = SentimentPipeline(
        scorer=scorer,
        gatekeeper=gatekeeper,
        cfg=PipelineConfig(
            polarity_threshold=cfg.finbert.polarity_threshold,
            confidence_threshold=cfg.finbert.confidence_threshold,
            min_conviction=cfg.llm.min_conviction,
            default_source_weight=cfg.llm.default_source_weight,
            signal_window=timedelta(minutes=cfg.signal.rolling_window_minutes),
        ),
    )
    overlay = RiskOverlay(
        cfg=cfg.risk, starting_equity=cfg.risk.starting_equity_usd
    )
    bot = build_default_bot(
        cfg, broker, pipeline=pipeline, overlay=overlay
    )
    bot.execution = ExecutionEngine(
        broker=broker,
        overlay=overlay,
        dry_run=cfg.mode == IbkrMode.DRY_RUN,
        fill_timeout_s=cfg.fill_timeout_s,
        order_style=cfg.execution.order_style,
        limit_offset_bps=cfg.execution.limit_offset_bps,
        short_policy=cfg.execution.short_policy(),
    )
    universe_symbols = [u.symbol for u in cfg.universe]
    bot.ingestion = IngestionService(
        feeds=cfg.ingestion.rss_feeds,
        universe=universe_symbols,
        poll_interval_s=cfg.ingestion.poll_interval_s,
        max_items_per_poll=cfg.ingestion.max_items_per_poll,
        dedup_window_minutes=cfg.ingestion.dedup_window_minutes,
    )
    return bot


async def run(config_path: str) -> None:
    cfg = IbkrSentimentConfig.from_yaml(config_path)
    secrets = IbkrSecrets()
    configure_logging(secrets.bot_log_level)
    log.info("ibkr_sentiment.starting", mode=cfg.mode.value, config=config_path)

    broker = await _build_broker(cfg)
    bot = _build_bot(cfg, broker)
    await bot.start()

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        log.info("ibkr_sentiment.signal.received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows

    async def tick_loop() -> None:
        while not stop_event.is_set():
            try:
                report = await bot.tick()
                log.info(
                    "ibkr_sentiment.tick",
                    decisions=len(report.decisions),
                    targets=len(report.targets),
                    placed=len(report.execution.placed) if report.execution else 0,
                    rejected=len(report.execution.rejected_by_risk) if report.execution else 0,
                    notes=report.notes,
                )
            except Exception as e:
                log.exception("ibkr_sentiment.tick.error", error=str(e))
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=cfg.tick_seconds
                )
            except TimeoutError:
                pass

    tick_task = asyncio.create_task(tick_loop())
    await stop_event.wait()
    log.info("ibkr_sentiment.shutdown.start")
    tick_task.cancel()
    try:
        await tick_task
    except asyncio.CancelledError:
        pass
    await bot.stop()
    log.info("ibkr_sentiment.shutdown.done")


def cli() -> None:
    parser = argparse.ArgumentParser(description="ibkr-sentiment-bot")
    parser.add_argument("--config", default="config/ibkr_sentiment.yaml")
    args = parser.parse_args()
    if not Path(args.config).exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        sys.exit(2)
    try:
        import uvloop  # type: ignore[import-not-found]

        uvloop.install()
    except ImportError:
        pass
    asyncio.run(run(args.config))


if __name__ == "__main__":
    cli()
