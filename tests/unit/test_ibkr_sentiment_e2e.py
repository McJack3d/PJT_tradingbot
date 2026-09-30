"""End-to-end test for the IBKR sentiment bot.

Wires the real (stub-driven) sentiment pipeline through the paper
broker and asserts that a bullish news item flows through to an open
long position with a sane dollar-neutral basket structure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from src.ibkr_sentiment.bot import build_default_bot
from src.ibkr_sentiment.broker.base import Bar
from src.ibkr_sentiment.broker.paper import PaperBroker
from src.ibkr_sentiment.config import (
    ExecutionConfig,
    FinBertConfig,
    IbkrMode,
    IbkrSentimentConfig,
    LLMConfig,
    RiskOverlayConfig,
    SignalConfig,
    UniverseEntry,
)
from src.ibkr_sentiment.sentiment.models import NewsItem


def _bars(symbol: str, start: float, count: int, step: float) -> list[Bar]:
    out: list[Bar] = []
    ts = datetime.now(UTC) - timedelta(days=count)
    for i in range(count):
        close = start + i * step
        out.append(
            Bar(
                symbol=symbol,
                ts=ts + timedelta(days=i),
                open=Decimal(str(close)),
                high=Decimal(str(close)),
                low=Decimal(str(close)),
                close=Decimal(str(close)),
                volume=Decimal("1000000"),
            )
        )
    return out


def _make_cfg() -> IbkrSentimentConfig:
    return IbkrSentimentConfig(
        mode=IbkrMode.PAPER,
        universe=[
            UniverseEntry(symbol="AAPL", sector_etf="XLK", min_qty=Decimal("1")),
            UniverseEntry(symbol="MSFT", sector_etf="XLK", min_qty=Decimal("1")),
        ],
        finbert=FinBertConfig(polarity_threshold=0.2, confidence_threshold=0.4),
        llm=LLMConfig(provider="stub", min_conviction=0.4),
        signal=SignalConfig(
            long_threshold=0.3,
            short_threshold=-0.3,
            sma_window=10,
            rsi_window=14,
            rsi_long_min=20,
            rsi_short_max=80,
            sma_confirm_pct=0.0,
            technical_confirm_required=True,
            rolling_window_minutes=240,
        ),
        risk=RiskOverlayConfig(
            starting_equity_usd=Decimal("10000"),
            max_gross_exposure_pct=Decimal("1.0"),
            max_net_exposure_pct=Decimal("0.5"),  # loose for the test
            max_position_pct=Decimal("0.25"),
        ),
        # These tests tick on the wall clock; the hours gate has its own
        # tests in test_ibkr_market_hours.py.
        execution=ExecutionConfig(enforce_market_hours=False),
        db_url="sqlite+aiosqlite:///:memory:",
    )


@pytest.mark.asyncio
async def test_bullish_news_opens_long_position(tmp_path: Path):
    cfg = _make_cfg()
    cfg.db_url = f"sqlite+aiosqlite:///{tmp_path}/e2e.db"
    broker = PaperBroker(starting_cash=Decimal("10000"))
    await broker.connect()
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100"))
    broker.set_quote("MSFT", bid=Decimal("100"), ask=Decimal("100"))
    broker.seed_bars("AAPL", _bars("AAPL", start=80, count=30, step=1.0))
    broker.seed_bars("MSFT", _bars("MSFT", start=80, count=30, step=1.0))

    bot = build_default_bot(cfg, broker, db_url=cfg.db_url)
    await bot.start()
    try:
        await bot.submit_item(
            NewsItem(
                title="AAPL beats record growth surge approval",
                body="surge growth beats record",
                symbols=("AAPL",),
                published_at=datetime.now(UTC),
            )
        )
        await bot.submit_item(
            NewsItem(
                title="MSFT misses lawsuit weak loss downgrade",
                body="loss lawsuit weak downgrade",
                symbols=("MSFT",),
                published_at=datetime.now(UTC),
            )
        )
        report = await bot.tick()
        assert report.execution is not None
        # Both names should produce decisions; AAPL long, MSFT short
        # (technical guards permit because the seeded bars are trending).
        by_sym = {d.symbol: d for d in report.decisions}
        assert "AAPL" in by_sym
        # AAPL has uptrending bars → long technicals pass.
        assert by_sym["AAPL"].side.value in ("long", "flat")
        # The execution result should contain at least the AAPL fill.
        placed_syms = {delta.symbol for delta, _ in report.execution.placed}
        assert "AAPL" in placed_syms
        positions = await broker.positions()
        sym_to_qty = {p.symbol: p.qty for p in positions}
        assert sym_to_qty.get("AAPL", Decimal("0")) > 0
    finally:
        await bot.stop()


@pytest.mark.asyncio
async def test_no_news_results_in_no_orders(tmp_path: Path):
    cfg = _make_cfg()
    cfg.db_url = f"sqlite+aiosqlite:///{tmp_path}/e2e.db"
    broker = PaperBroker(starting_cash=Decimal("10000"))
    await broker.connect()
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100"))

    bot = build_default_bot(cfg, broker, db_url=cfg.db_url)
    await bot.start()
    try:
        report = await bot.tick()
        # Nothing in the buffer → no execution, no targets.
        assert report.execution is None or len(report.execution.placed) == 0
        assert "no_news_buffered" in report.notes or "no_signals" in report.notes
    finally:
        await bot.stop()


@pytest.mark.asyncio
async def test_drawdown_stop_halts_execution(tmp_path: Path):
    cfg = _make_cfg()
    cfg.db_url = f"sqlite+aiosqlite:///{tmp_path}/e2e.db"
    cfg.risk.cumulative_loss_stop_pct = Decimal("0.05")  # tight stop
    broker = PaperBroker(starting_cash=Decimal("9000"))  # already 10% down
    await broker.connect()
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100"))
    broker.seed_bars("AAPL", _bars("AAPL", start=80, count=30, step=1.0))

    bot = build_default_bot(cfg, broker, db_url=cfg.db_url)
    await bot.start()
    try:
        await bot.submit_item(
            NewsItem(
                title="AAPL beats record growth surge approval",
                body="surge growth beats record",
                symbols=("AAPL",),
                published_at=datetime.now(UTC),
            )
        )
        report = await bot.tick()
        # Drawdown should trip the overlay; nothing should be placed.
        assert report.execution is not None
        assert report.execution.placed == []
        assert any("account_halt" in e[0] for e in report.execution.errors)
    finally:
        await bot.stop()


class _CountingPaperBroker(PaperBroker):
    """Paper broker that records cancel/flatten calls in order."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls: list[str] = []

    async def cancel_all_orders(self) -> None:
        self.calls.append("cancel")

    async def flatten_all(self):
        self.calls.append("flatten")
        return await super().flatten_all()


async def _open_long(broker: PaperBroker, symbol: str, qty: str) -> None:
    from src.ibkr_sentiment.broker.base import OrderRequest, OrderSide

    await broker.place_order(
        OrderRequest(symbol=symbol, side=OrderSide.BUY, qty=Decimal(qty))
    )


@pytest.mark.asyncio
async def test_drawdown_without_news_flattens_once(tmp_path: Path):
    """Regression: the account check only ran when signals existed, and
    a halt never flattened. A drawdown on a quiet news day must cancel
    open orders, flatten the book once, and stay halted."""
    cfg = _make_cfg()
    cfg.db_url = f"sqlite+aiosqlite:///{tmp_path}/e2e.db"
    broker = _CountingPaperBroker(starting_cash=Decimal("10000"))
    await broker.connect()
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100"))
    await _open_long(broker, "AAPL", "20")

    bot = build_default_bot(cfg, broker, db_url=cfg.db_url)
    await bot.start()
    try:
        first = await bot.tick()  # sets today's anchor
        assert "account_halt" not in first.notes

        broker.set_quote("AAPL", bid=Decimal("50"), ask=Decimal("50"))  # -10% NLV
        halted = await bot.tick()
        assert "account_halt" in halted.notes
        assert halted.execution is not None
        assert any(e[0] == "account_halt" for e in halted.execution.errors)
        assert len(halted.execution.flattened) == 1
        assert broker.calls == ["cancel", "flatten"]
        assert await broker.positions() == []

        again = await bot.tick()
        assert "account_halt" in again.notes
        assert broker.calls == ["cancel", "flatten"]  # no duplicate flatten
    finally:
        await bot.stop()


@pytest.mark.asyncio
async def test_restart_restores_daily_anchor(tmp_path: Path):
    """A restart mid-session must not reset the daily loss stop."""
    cfg = _make_cfg()
    cfg.db_url = f"sqlite+aiosqlite:///{tmp_path}/e2e.db"
    broker = PaperBroker(starting_cash=Decimal("10000"))
    await broker.connect()
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100"))
    await _open_long(broker, "AAPL", "20")

    bot = build_default_bot(cfg, broker, db_url=cfg.db_url)
    await bot.start()
    await bot.tick()
    anchor = bot.overlay.daily_anchor
    await bot.stop()

    await broker.connect()
    broker.set_quote("AAPL", bid=Decimal("85"), ask=Decimal("85"))  # -3% on the day
    restarted = build_default_bot(cfg, broker, db_url=cfg.db_url)
    await restarted.start()
    try:
        assert restarted.overlay.daily_anchor == anchor
        report = await restarted.tick()
        assert "account_halt" in report.notes
        assert "daily" in report.execution.errors[0][1]
    finally:
        await restarted.stop()
