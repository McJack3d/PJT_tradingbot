"""Top-level orchestrator for the IBKR sentiment bot.

Single class — `IbkrSentimentBot` — that wires the funnel, the broker,
the signal engine, the dollar-neutral basket, the risk overlay, and
the database together. `tick()` runs one full evaluation cycle.

The bot is push-driven for news (ingestion calls `on_item`) and
pull-driven for trade decisions (`tick()` on a timer). Keeping those
two paths decoupled is what lets a slow LLM call never stall a market
data update.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.ibkr_sentiment.broker.base import AccountSummary, Broker
from src.ibkr_sentiment.config import IbkrSentimentConfig
from src.ibkr_sentiment.execution.engine import ExecutionEngine, RunResult
from src.ibkr_sentiment.market_hours import MarketCalendar
from src.ibkr_sentiment.risk.overlay import NEW_YORK, RiskOverlay, trading_day
from src.ibkr_sentiment.sentiment.ingestion import IngestionService
from src.ibkr_sentiment.sentiment.models import LLMVerdict, NewsItem, StructuredSignal
from src.ibkr_sentiment.sentiment.pipeline import (
    PipelineConfig,
    SentimentPipeline,
    aggregate_signals,
)
from src.ibkr_sentiment.signal_engine.dollar_neutral import (
    TargetPosition,
    build_dollar_neutral_basket,
)
from src.ibkr_sentiment.signal_engine.mapping import SymbolDecision, decide
from src.ibkr_sentiment.signal_engine.technical import (
    TechnicalSnapshot,
    evaluate_technicals,
)
from src.ibkr_sentiment.state.db import IbkrSentimentDB


@dataclass
class TickReport:
    decisions: list[SymbolDecision] = field(default_factory=list)
    targets: list[TargetPosition] = field(default_factory=list)
    execution: RunResult | None = None
    fresh_signals: list[StructuredSignal] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class IbkrSentimentBot:
    cfg: IbkrSentimentConfig
    broker: Broker
    pipeline: SentimentPipeline
    overlay: RiskOverlay
    execution: ExecutionEngine
    db: IbkrSentimentDB
    ingestion: IngestionService | None = None
    startup_reconciliation: RunResult | None = None
    # None disables the market-hours gate (tests, or 24h instruments).
    calendar: MarketCalendar | None = None

    # Items that arrived since the last tick. Flushed at the top of
    # `tick()`. Keeping a small buffer rather than firing inference on
    # every single article smooths over bursty news feeds and lets the
    # LLM stage batch.
    _buffer: list[NewsItem] = field(default_factory=list)
    _buffer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Rolling stash of accepted signals: aggregator works over the
    # window so late-arriving items can still contribute.
    _recent_verdicts: list[tuple[NewsItem, LLMVerdict]] = field(default_factory=list)

    async def start(self) -> None:
        await self.db.init()
        await self._restore_daily_anchor()
        if not await self.broker.is_connected():
            await self.broker.connect()
        await self._reconcile_on_start()
        if self.ingestion is not None:
            await self.ingestion.start(self._on_item)

    async def stop(self) -> None:
        if self.ingestion is not None:
            await self.ingestion.stop()
        if await self.broker.is_connected():
            await self.broker.disconnect()
        await self.db.close()

    async def _reconcile_on_start(self) -> RunResult:
        """Bring DB and broker back in line after a restart.

        1. Trades the DB last saw as working get their final state from
           the broker (IBKR replays today's orders on connect).
        2. Any of our orders still working at the broker are cancelled,
           so the first tick plans from a quiet book instead of
           re-sending orders that are already live.
        """
        report = RunResult()
        for row in await self.db.open_trades():
            try:
                latest = await self.broker.order_status(row.client_order_id)
            except Exception as e:
                report.errors.append((row.symbol, f"order_status {type(e).__name__}: {e}"))
                continue
            if latest is not None:
                await self.db.update_trade(latest)
            else:
                report.errors.append((row.symbol, f"unknown to broker: {row.client_order_id}"))
        await self.execution.cancel_stale_orders(report)
        # Stale orders may have filled before the cancel landed.
        for cid in report.stale_cancelled:
            latest = await self.broker.order_status(cid)
            if latest is not None:
                await self.db.update_trade(latest)
        self.startup_reconciliation = report
        return report

    async def _restore_daily_anchor(self, now: datetime | None = None) -> None:
        """Seed the daily loss-stop anchor from today's first equity
        snapshot so a restart mid-session can't reset the daily stop."""
        now = now or datetime.now(UTC)
        day = trading_day(now)
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=NEW_YORK)
        row = await self.db.first_equity_since(day_start)
        if row is not None:
            self.overlay.daily_anchor = row.net_liquidation
            self.overlay.anchor_day = day

    async def _record_equity(self, account: AccountSummary) -> None:
        positions = await self.broker.positions()
        await self.db.record_equity(
            net_liquidation=account.net_liquidation,
            gross_exposure=sum(
                (abs(p.qty * p.mark_price) for p in positions), Decimal("0")
            ),
            net_exposure=sum((p.qty * p.mark_price for p in positions), Decimal("0")),
            open_positions=len(positions),
        )

    # ---- ingestion side ---------------------------------------------

    async def _on_item(self, item: NewsItem) -> None:
        async with self._buffer_lock:
            self._buffer.append(item)
        try:
            await self.db.record_news(item)
        except Exception:
            pass

    async def submit_item(self, item: NewsItem) -> None:
        """Manually push a NewsItem into the pipeline. Used by tests
        and one-off backfill scripts."""
        await self._on_item(item)

    # ---- main tick ---------------------------------------------------

    async def tick(self, *, now: datetime | None = None) -> TickReport:
        report = TickReport()
        now = now or datetime.now(UTC)

        # 1. Drain buffer + run the funnel on whatever arrived.
        async with self._buffer_lock:
            items = self._buffer
            self._buffer = []
        source_weights = await self.db.source_weights()
        if items:
            decisions, verdicts, fresh_signals = await self.pipeline.run(
                items, source_weights=source_weights, now=now
            )
            await self.db.record_finbert(decisions)
            await self.db.record_verdicts(verdicts)
            # Save (item, verdict) pairs in the rolling window for the
            # next aggregation pass.
            item_by_id = {i.id: i for i in items}
            for v in verdicts:
                if v.item_id in item_by_id:
                    self._recent_verdicts.append((item_by_id[v.item_id], v))
        else:
            fresh_signals = []
            report.notes.append("no_news_buffered")

        # 1b. Account-level gate. Runs every tick — before any early
        #     return — so a drawdown halts and flattens the book even
        #     when no news arrives.
        account = await self.broker.account_summary()
        prev_day = self.overlay.anchor_day
        self.overlay.roll_daily_anchor(account.net_liquidation, now)
        if self.overlay.anchor_day != prev_day:
            await self._record_equity(account)
        halt_result = RunResult()
        if not (await self.execution.enforce_account(account, halt_result)).ok:
            report.execution = halt_result
            report.fresh_signals = fresh_signals
            report.notes.append("account_halt")
            return report

        # 2. Trim the rolling verdict window.
        cutoff = now - self.pipeline.cfg.signal_window
        self._recent_verdicts = [
            (i, v) for (i, v) in self._recent_verdicts
            if i.published_at >= cutoff
        ]

        # 3. Re-aggregate signals over the rolling window (lets late
        #    news for the same symbol combine with earlier verdicts).
        signals = aggregate_signals(
            self._recent_verdicts,
            source_weights=source_weights,
            cfg=self.pipeline.cfg,
            now=now,
        )

        if not signals:
            report.notes.append("no_signals")
            report.fresh_signals = fresh_signals
            return report

        # 4. Pull bars + compute technicals for each symbol with a signal.
        snapshots: dict[str, TechnicalSnapshot] = {}
        for sig in signals:
            try:
                bars = await self.broker.historical_bars(
                    sig.symbol, duration="120 D", bar_size="1 day"
                )
            except Exception:
                bars = []
            closes = [float(b.close) for b in bars] if bars else []
            if not closes:
                # Fall back to a flat synthetic so we don't divide by zero
                # downstream — the technical guard will reject this anyway.
                continue
            snapshots[sig.symbol] = evaluate_technicals(
                sig.symbol,
                closes,
                sma_window=self.cfg.signal.sma_window,
                rsi_window=self.cfg.signal.rsi_window,
            )

        # 5. Map sentiment + technicals into LONG / SHORT / FLAT decisions.
        symbol_decisions = decide(
            signals,
            snapshots,
            long_threshold=self.cfg.signal.long_threshold,
            short_threshold=self.cfg.signal.short_threshold,
            sma_confirm_pct=self.cfg.signal.sma_confirm_pct,
            rsi_long_min=self.cfg.signal.rsi_long_min,
            rsi_short_max=self.cfg.signal.rsi_short_max,
            technical_confirm_required=self.cfg.signal.technical_confirm_required,
        )

        # 6. Annotate the structured signals with the technical verdict so
        #    they get stored with provenance, and persist them.
        decisions_by_sym = {d.symbol: d for d in symbol_decisions}
        for sig in signals:
            d = decisions_by_sym.get(sig.symbol)
            if d is not None:
                sig.technical_ok = (not d.rejected) and d.side != d.side.FLAT
                sig.technical_reason = d.technical_reason
        await self.db.record_signals(signals)

        # 6b. Market-hours gate. Signals are recorded above either way;
        #     orders only go out inside the NYSE trading window.
        if self.calendar is not None:
            window = self.calendar.window(now)
            if not window.open:
                report.decisions = symbol_decisions
                report.fresh_signals = fresh_signals
                report.notes.append(f"market_closed: {window.reason}")
                return report

        # 7. Build the dollar-neutral basket.
        min_qty = {u.symbol: u.min_qty for u in self.cfg.universe}
        sector_of = {
            u.symbol: u.sector_etf for u in self.cfg.universe if u.sector_etf
        }
        targets = build_dollar_neutral_basket(
            symbol_decisions,
            nlv=account.net_liquidation,
            max_gross_pct=self.cfg.risk.max_gross_exposure_pct,
            max_position_pct=self.cfg.risk.max_position_pct,
            min_qty=min_qty,
            sector_of=sector_of or None,
            max_sector_pct=self.cfg.risk.max_sector_pct,
        )

        # 8. Execute. Cancel our own leftover working orders first so
        #    positions are read from a quiet book.
        pre = RunResult()
        await self.execution.cancel_stale_orders(pre)
        positions = await self.broker.positions()
        current_positions = {p.symbol: p.qty for p in positions}
        result = await self.execution.execute_basket(
            targets,
            account=account,
            current_positions=current_positions,
            marks={p.symbol: p.mark_price for p in positions},
        )
        result.stale_cancelled = pre.stale_cancelled
        result.foreign_open_orders = pre.foreign_open_orders
        result.errors = pre.errors + result.errors

        # 9. Persist the resulting trades.
        for delta, placed in result.placed:
            await self.db.record_trade(delta, placed)

        # 10. Equity snapshot.
        await self._record_equity(account)

        report.decisions = symbol_decisions
        report.targets = targets
        report.execution = result
        report.fresh_signals = fresh_signals
        return report


def build_default_bot(
    cfg: IbkrSentimentConfig,
    broker: Broker,
    *,
    db_url: str | None = None,
    pipeline: SentimentPipeline | None = None,
    overlay: RiskOverlay | None = None,
    ingestion: IngestionService | None = None,
) -> IbkrSentimentBot:
    """Wire a bot up with reasonable defaults. Callers can override
    every component for tests (the broker is almost always a
    `PaperBroker` in tests)."""
    db = IbkrSentimentDB(db_url or cfg.db_url)
    if pipeline is None:
        from src.ibkr_sentiment.sentiment.finbert import StubFinBertScorer
        from src.ibkr_sentiment.sentiment.llm_gatekeeper import StubLLMGatekeeper

        pipeline = SentimentPipeline(
            scorer=StubFinBertScorer(max_input_chars=cfg.finbert.max_input_chars),
            gatekeeper=StubLLMGatekeeper(),
            cfg=PipelineConfig(
                polarity_threshold=cfg.finbert.polarity_threshold,
                confidence_threshold=cfg.finbert.confidence_threshold,
                min_conviction=cfg.llm.min_conviction,
                default_source_weight=cfg.llm.default_source_weight,
                signal_window=timedelta(
                    minutes=cfg.signal.rolling_window_minutes
                ),
            ),
        )
    if overlay is None:
        overlay = RiskOverlay(
            cfg=cfg.risk, starting_equity=cfg.risk.starting_equity_usd
        )
    execution = ExecutionEngine(
        broker=broker,
        overlay=overlay,
        fill_timeout_s=cfg.fill_timeout_s,
        order_style=cfg.execution.order_style,
        limit_offset_bps=cfg.execution.limit_offset_bps,
    )
    calendar = (
        MarketCalendar(
            open_buffer=timedelta(minutes=cfg.execution.open_buffer_minutes),
            close_buffer=timedelta(minutes=cfg.execution.close_buffer_minutes),
        )
        if cfg.execution.enforce_market_hours
        else None
    )
    return IbkrSentimentBot(
        cfg=cfg,
        broker=broker,
        pipeline=pipeline,
        overlay=overlay,
        execution=execution,
        db=db,
        ingestion=ingestion,
        calendar=calendar,
    )
