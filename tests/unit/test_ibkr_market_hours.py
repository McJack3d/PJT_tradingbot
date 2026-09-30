"""Market-hours gate and order styles (audit defect #6).

Before: the bot ticked 24/7 and sent MARKET orders at any hour, so
overnight decisions queued up and filled into the opening auction gap.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ibkr_sentiment.bot import build_default_bot
from src.ibkr_sentiment.broker.base import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
)
from src.ibkr_sentiment.broker.ibkr import IbkrBroker
from src.ibkr_sentiment.broker.paper import PaperBroker
from src.ibkr_sentiment.config import ExecutionConfig, RiskOverlayConfig
from src.ibkr_sentiment.execution.engine import ExecutionEngine
from src.ibkr_sentiment.market_hours import MarketCalendar
from src.ibkr_sentiment.risk.overlay import RiskOverlay
from src.ibkr_sentiment.sentiment.models import NewsItem
from src.ibkr_sentiment.signal_engine.dollar_neutral import TargetPosition
from src.ibkr_sentiment.signal_engine.mapping import Side
from tests.unit.test_ibkr_sentiment_e2e import _bars, _make_cfg


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# ---- calendar -------------------------------------------------------


@pytest.mark.parametrize(
    ("ts", "is_open", "why"),
    [
        ("2026-09-30T13:34", False, "before"),  # 09:34 ET, inside open buffer
        ("2026-09-30T13:35", True, "within"),  # 09:35 ET
        ("2026-09-30T19:49", True, "within"),  # 15:49 ET
        ("2026-09-30T19:50", False, "after"),  # 15:50 ET, inside close buffer
        ("2026-10-03T15:00", False, "not a trading day"),  # Saturday
        ("2026-11-26T15:00", False, "not a trading day"),  # Thanksgiving
        ("2026-11-27T17:49", True, "within"),  # early close 13:00 ET
        ("2026-11-27T17:50", False, "after"),
        ("2026-12-01T14:34", False, "before"),  # EST: open is 14:30 UTC
        ("2026-12-01T14:35", True, "within"),
    ],
)
def test_calendar_window(ts, is_open, why):
    v = MarketCalendar().window(_utc(ts))
    assert v.open is is_open
    assert why in v.reason


def test_calendar_rejects_naive_datetime():
    with pytest.raises(ValueError):
        MarketCalendar().window(datetime(2026, 9, 30, 15, 0))


def test_calendar_rebuilds_near_its_horizon():
    cal = MarketCalendar()
    cal.window(_utc("2026-09-30T15:00"))
    far = _utc("2028-06-01T15:00")  # beyond the first build's ~1y horizon
    assert cal.window(far).open is True


# ---- bot gate -------------------------------------------------------


async def _bot_with_bullish_news(tmp_path: Path, now: datetime):
    cfg = _make_cfg()
    cfg.execution = ExecutionConfig()  # gate ON
    cfg.db_url = f"sqlite+aiosqlite:///{tmp_path}/h.db"
    broker = PaperBroker(starting_cash=Decimal("10000"))
    await broker.connect()
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100"))
    broker.seed_bars("AAPL", _bars("AAPL", start=80, count=30, step=1.0))
    bot = build_default_bot(cfg, broker, db_url=cfg.db_url)
    await bot.start()
    await bot.submit_item(
        NewsItem(
            title="AAPL beats record growth surge approval",
            body="surge growth beats record",
            symbols=("AAPL",),
            published_at=now,
        )
    )
    return bot, broker


@pytest.mark.asyncio
async def test_no_orders_outside_trading_window(tmp_path: Path):
    now = _utc("2026-10-03T15:00")  # Saturday
    bot, broker = await _bot_with_bullish_news(tmp_path, now)
    try:
        report = await bot.tick(now=now)
        assert report.execution is None
        assert any(n.startswith("market_closed") for n in report.notes)
        assert report.decisions  # signals still scored and recorded
        assert await broker.positions() == []
    finally:
        await bot.stop()


@pytest.mark.asyncio
async def test_orders_sent_inside_trading_window(tmp_path: Path):
    now = _utc("2026-09-30T15:00")  # 11:00 ET, Wednesday
    bot, broker = await _bot_with_bullish_news(tmp_path, now)
    try:
        report = await bot.tick(now=now)
        assert report.execution is not None and report.execution.placed
        [(_, res)] = report.execution.placed
        assert res.status == OrderStatus.FILLED
        assert [p.symbol for p in await broker.positions()] == ["AAPL"]
    finally:
        await bot.stop()


# ---- order styles ---------------------------------------------------


def _engine(broker, style: str = "limit") -> ExecutionEngine:
    cfg = RiskOverlayConfig(max_position_pct=Decimal("0.5"), max_net_exposure_pct=Decimal("1"))
    return ExecutionEngine(
        broker=broker,
        overlay=RiskOverlay(cfg=cfg, starting_equity=Decimal("100000")),
        order_style=style,
        limit_offset_bps=Decimal("10"),
        fill_timeout_s=0.0,
        cancel_confirm_timeout_s=0.0,
    )


def _delta(qty: str) -> TargetPosition:
    q = Decimal(qty)
    return TargetPosition("AAPL", Side.LONG if q > 0 else Side.SHORT, q, abs(q) * 100, "")


@pytest.mark.asyncio
async def test_limit_price_is_touch_plus_offset_rounded_inward():
    broker = PaperBroker()
    broker.set_quote("AAPL", bid=Decimal("99.97"), ask=Decimal("100.03"))
    engine = _engine(broker)
    buy = await engine._to_order_request(_delta("10"))
    sell = await engine._to_order_request(_delta("-10"))
    assert buy.order_type == OrderType.LIMIT
    assert buy.limit_price == Decimal("100.13")  # 100.03 * 1.001 = 100.13003 -> floor
    assert sell.limit_price == Decimal("99.88")  # 99.97 * 0.999 = 99.87003 -> ceil


@pytest.mark.asyncio
async def test_sub_dollar_limit_uses_fine_tick():
    broker = PaperBroker()
    broker.set_quote("AAPL", bid=Decimal("0.5"), ask=Decimal("0.5"))
    req = await _engine(broker)._to_order_request(_delta("10"))
    assert req.limit_price == Decimal("0.5005")


@pytest.mark.asyncio
async def test_no_quote_skips_order_with_error():
    class _NoQuoteBroker(PaperBroker):
        async def quote(self, symbol):
            return Quote(symbol, Decimal(0), Decimal(0), Decimal(0))

    broker = _NoQuoteBroker()
    res = await _engine(broker).execute_basket(
        [_delta("10")],
        account=await broker.account_summary(),
        current_positions={},
        marks={"AAPL": Decimal("100")},
    )
    assert res.placed == []
    assert any("no quote" in e[1] for e in res.errors)


@pytest.mark.asyncio
async def test_adaptive_and_market_styles():
    broker = PaperBroker()
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100"))
    adaptive = await _engine(broker, "adaptive")._to_order_request(_delta("1"))
    market = await _engine(broker, "market")._to_order_request(_delta("1"))
    assert (adaptive.order_type, adaptive.algo) == (OrderType.MARKET, "Adaptive")
    assert (market.order_type, market.algo) == (OrderType.MARKET, None)


# ---- paper broker limit semantics ------------------------------------


@pytest.mark.asyncio
async def test_paper_marketable_limit_never_fills_through_limit():
    broker = PaperBroker(slippage_bps=Decimal("50"))  # mid+50bps = 100.50
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100"))
    res = await broker.place_order(
        OrderRequest("AAPL", OrderSide.BUY, Decimal("1"), OrderType.LIMIT, Decimal("100.10"))
    )
    assert res.status == OrderStatus.FILLED
    assert res.avg_fill_price == Decimal("100.10")


@pytest.mark.asyncio
async def test_paper_non_marketable_limit_rests_then_cancels():
    broker = PaperBroker()
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100.10"))
    res = await broker.place_order(
        OrderRequest(
            "AAPL", OrderSide.BUY, Decimal("1"), OrderType.LIMIT, Decimal("99"), client_order_id="c1"
        )
    )
    assert res.status == OrderStatus.SUBMITTED
    [o] = await broker.open_orders()
    assert o.client_order_id == "c1"
    await broker.cancel_order(res.broker_order_id)
    assert (await broker.order_status("c1")).status == OrderStatus.CANCELED
    assert await broker.open_orders() == []
    assert await broker.positions() == []


@pytest.mark.asyncio
async def test_engine_cancels_unfilled_limit_in_paper():
    broker = PaperBroker()
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100"))
    engine = _engine(broker)
    engine.limit_offset_bps = Decimal("-100")  # force a non-marketable price
    res = await engine.execute_basket(
        [_delta("10")], account=await broker.account_summary(), current_positions={}
    )
    [(_, final)] = res.placed
    assert final.status == OrderStatus.CANCELED
    assert await broker.open_orders() == []


# ---- IbkrBroker Adaptive order --------------------------------------


@pytest.mark.asyncio
async def test_ibkr_adaptive_sets_algo_fields(monkeypatch):
    class _Order(SimpleNamespace):
        pass

    fake = SimpleNamespace(
        MarketOrder=lambda action, qty: _Order(action=action, totalQuantity=qty),
        TagValue=lambda k, v: (k, v),
    )
    placed = []

    class _IB:
        def placeOrder(self, contract, order):
            order.orderId = 1
            placed.append(order)
            return SimpleNamespace(
                order=order,
                orderStatus=SimpleNamespace(status="Submitted", filled=0, avgFillPrice=0),
            )

    b = IbkrBroker()
    b._ib = _IB()
    b._contract_cache["AAPL|SMART|USD"] = object()
    monkeypatch.setattr(IbkrBroker, "_ib_insync", staticmethod(lambda: fake))
    await b.place_order(OrderRequest("AAPL", OrderSide.BUY, Decimal("5"), algo="Adaptive"))
    [o] = placed
    assert o.algoStrategy == "Adaptive"
    assert o.algoParams == [("adaptivePriority", "Normal")]
