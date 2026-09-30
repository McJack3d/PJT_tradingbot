"""Fill tracking + reconciliation (audit defect #5).

Before: place_order returned the submission snapshot, the bot recorded
that as the trade, partial fills were never followed up, and a restart
could re-send orders that were still working at the broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src.ibkr_sentiment.broker.base import (
    OpenOrderView,
    OrderRequest,
    OrderResult,
    OrderStatus,
)
from src.ibkr_sentiment.broker.ibkr import IbkrBroker
from src.ibkr_sentiment.broker.paper import PaperBroker
from src.ibkr_sentiment.config import RiskOverlayConfig
from src.ibkr_sentiment.execution.engine import ExecutionEngine, RunResult
from src.ibkr_sentiment.risk.overlay import RiskOverlay
from src.ibkr_sentiment.signal_engine.dollar_neutral import TargetPosition
from src.ibkr_sentiment.signal_engine.mapping import Side
from src.ibkr_sentiment.state.db import IbkrSentimentDB

# ---- scripted broker ------------------------------------------------


class ScriptedBroker(PaperBroker):
    """Paper broker whose orders rest until the test says otherwise.

    `fill_script[symbol]` is the list of snapshots successive
    wait_for_fill calls return for that symbol's order."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.fill_script: dict[str, list[OrderResult]] = {}
        self.cancelled: list[str] = []
        self.working: list[OpenOrderView] = []
        self.known: dict[str, OrderResult] = {}

    async def place_order(self, req: OrderRequest) -> OrderResult:
        res = OrderResult(
            client_order_id=req.client_order_id,
            broker_order_id=f"B-{req.symbol}",
            status=OrderStatus.SUBMITTED,
            filled_qty=Decimal("0"),
            avg_fill_price=Decimal("0"),
        )
        self.known[req.client_order_id] = res
        return res

    async def wait_for_fill(self, order, timeout_s):
        script = self.fill_script.get(order.broker_order_id.removeprefix("B-"), [])
        if not script:
            return order
        snap = script.pop(0)
        snap.client_order_id = order.client_order_id
        snap.broker_order_id = order.broker_order_id
        return snap

    async def cancel_order(self, broker_order_id: str) -> None:
        self.cancelled.append(broker_order_id)

    async def open_orders(self):
        return list(self.working)

    async def order_status(self, client_order_id: str):
        return self.known.get(client_order_id)


def _snap(status: OrderStatus, filled: str, px: str = "100") -> OrderResult:
    return OrderResult(
        client_order_id="",
        broker_order_id=None,
        status=status,
        filled_qty=Decimal(filled),
        avg_fill_price=Decimal(px),
    )


def _engine(broker, **kw) -> ExecutionEngine:
    cfg = RiskOverlayConfig(
        starting_equity_usd=Decimal("100000"),
        max_position_pct=Decimal("0.5"),
        max_net_exposure_pct=Decimal("1"),
        max_gross_exposure_pct=Decimal("2"),
    )
    return ExecutionEngine(
        broker=broker,
        overlay=RiskOverlay(cfg=cfg, starting_equity=Decimal("100000")),
        fill_timeout_s=0.01,
        cancel_confirm_timeout_s=0.01,
        **kw,
    )


def _target(sym: str, qty: str) -> TargetPosition:
    return TargetPosition(
        symbol=sym,
        side=Side.LONG,
        target_qty=Decimal(qty),
        notional=Decimal(qty) * 100,
        reason="",
    )


async def _run(engine, broker, targets):
    return await engine.execute_basket(
        targets,
        account=await broker.account_summary(),
        current_positions={},
    )


# ---- engine ---------------------------------------------------------


@pytest.mark.asyncio
async def test_placed_carries_final_fill_not_submission_snapshot():
    broker = ScriptedBroker()
    broker.fill_script["AAPL"] = [_snap(OrderStatus.FILLED, "10", "101.5")]
    engine = _engine(broker)
    res = await _run(engine, broker, [_target("AAPL", "10")])
    [(_, final)] = res.placed
    assert final.status == OrderStatus.FILLED
    assert final.filled_qty == Decimal("10")
    assert final.avg_fill_price == Decimal("101.5")
    assert broker.cancelled == []


@pytest.mark.asyncio
async def test_partial_fill_at_timeout_cancels_remainder():
    broker = ScriptedBroker()
    broker.fill_script["AAPL"] = [
        _snap(OrderStatus.PARTIALLY_FILLED, "4"),  # at fill timeout
        _snap(OrderStatus.CANCELED, "4"),  # after cancel confirms
    ]
    engine = _engine(broker)
    res = await _run(engine, broker, [_target("AAPL", "10")])
    [(_, final)] = res.placed
    assert broker.cancelled == ["B-AAPL"]
    assert final.status == OrderStatus.CANCELED
    assert final.filled_qty == Decimal("4")
    assert res.errors == []


@pytest.mark.asyncio
async def test_unconfirmed_cancel_is_reported():
    broker = ScriptedBroker()
    broker.fill_script["AAPL"] = [
        _snap(OrderStatus.SUBMITTED, "0"),
        _snap(OrderStatus.SUBMITTED, "0"),
    ]
    res = await _run(_engine(broker), broker, [_target("AAPL", "10")])
    assert any("cancel unconfirmed" in e[1] for e in res.errors)


@pytest.mark.asyncio
async def test_orders_are_all_placed_before_waiting():
    """Fills are awaited together, not one order at a time."""
    broker = ScriptedBroker()
    order_of_events: list[str] = []
    orig_place = broker.place_order

    async def place(req):
        order_of_events.append(f"place {req.symbol}")
        return await orig_place(req)

    async def wait(order, t):
        order_of_events.append(f"wait {order.broker_order_id}")
        return _snap(OrderStatus.FILLED, "1")

    broker.place_order, broker.wait_for_fill = place, wait
    await _run(_engine(broker), broker, [_target("AAPL", "1"), _target("MSFT", "1")])
    assert order_of_events[:2] == ["place AAPL", "place MSFT"]


@pytest.mark.asyncio
async def test_cancel_stale_orders_only_touches_our_prefix():
    broker = ScriptedBroker()
    broker.working = [
        OpenOrderView("ibsent-abc", "1", "AAPL", Decimal("5")),
        OpenOrderView("", "2", "MSFT", Decimal("-3")),  # placed by hand in TWS
    ]
    report = RunResult()
    await _engine(broker).cancel_stale_orders(report)
    assert broker.cancelled == ["1"]
    assert report.stale_cancelled == ["ibsent-abc"]
    assert report.foreign_open_orders == ["2"]


@pytest.mark.asyncio
async def test_cancel_stale_orders_is_noop_in_dry_run():
    broker = ScriptedBroker()
    broker.working = [OpenOrderView("ibsent-abc", "1", "AAPL", Decimal("5"))]
    await _engine(broker, dry_run=True).cancel_stale_orders(RunResult())
    assert broker.cancelled == []


# ---- IbkrBroker against a fake ib_insync client ---------------------


@dataclass
class _Order:
    orderId: int
    action: str = "BUY"
    orderRef: str = ""


@dataclass
class _Status:
    status: str = "Submitted"
    filled: float = 0.0
    remaining: float = 0.0
    avgFillPrice: float = 0.0


@dataclass
class _Contract:
    symbol: str


@dataclass
class _Trade:
    order: _Order
    orderStatus: _Status
    contract: _Contract = field(default_factory=lambda: _Contract("AAPL"))


@dataclass
class _FakeIB:
    trades_: list[_Trade] = field(default_factory=list)

    def trades(self):
        return self.trades_

    def openTrades(self):
        return [t for t in self.trades_ if t.orderStatus.status not in ("Filled", "Cancelled")]


def _ibkr(ib) -> IbkrBroker:
    b = IbkrBroker()
    b._ib = ib
    b._fill_poll_s = 0.0
    return b


@pytest.mark.asyncio
async def test_ibkr_submitted_with_fills_maps_to_partially_filled():
    t = _Trade(_Order(7, orderRef="ibsent-x"), _Status("Submitted", 4, 6, 100.25))
    b = _ibkr(_FakeIB([t]))
    res = await b.wait_for_fill(
        OrderResult("ibsent-x", "7", OrderStatus.SUBMITTED, Decimal(0), Decimal(0)),
        timeout_s=0.0,
    )
    assert res.status == OrderStatus.PARTIALLY_FILLED
    assert res.filled_qty == Decimal("4")
    assert res.avg_fill_price == Decimal("100.25")


@pytest.mark.asyncio
async def test_ibkr_wait_for_fill_returns_once_filled():
    t = _Trade(_Order(7), _Status("Filled", 10, 0, 99.0))
    res = await _ibkr(_FakeIB([t])).wait_for_fill(
        OrderResult("c", "7", OrderStatus.SUBMITTED, Decimal(0), Decimal(0)),
        timeout_s=5.0,
    )
    assert res.status == OrderStatus.FILLED
    assert res.filled_qty == Decimal("10")


@pytest.mark.asyncio
async def test_ibkr_order_status_by_order_ref_and_open_orders_signed():
    ib = _FakeIB(
        [
            _Trade(_Order(1, "SELL", "ibsent-a"), _Status("Submitted", 0, 5)),
            _Trade(_Order(2, "BUY", "ibsent-b"), _Status("Filled", 3, 0, 50.0)),
        ]
    )
    b = _ibkr(ib)
    assert (await b.order_status("ibsent-b")).status == OrderStatus.FILLED
    assert await b.order_status("nope") is None
    [o] = await b.open_orders()
    assert (o.client_order_id, o.remaining_qty) == ("ibsent-a", Decimal("-5"))


# ---- DB -------------------------------------------------------------


@pytest.mark.asyncio
async def test_db_migrates_old_trades_table(tmp_path: Path):
    path = tmp_path / "old.db"
    eng = create_engine(f"sqlite:///{path}")
    with eng.begin() as c:
        c.execute(
            text(
                "CREATE TABLE ibsent_trades (id INTEGER PRIMARY KEY, client_order_id VARCHAR(64),"
                " broker_order_id VARCHAR(64), symbol VARCHAR(32), side VARCHAR(5),"
                " qty DECIMAL(28,8), avg_fill_price DECIMAL(28,8), status VARCHAR(32),"
                " placed_at TIMESTAMP)"
            )
        )
        c.execute(
            text(
                "INSERT INTO ibsent_trades VALUES (1,'ibsent-old','9','AAPL','LONG',"
                "10,0,'submitted','2026-09-29 14:00:00')"
            )
        )
    eng.dispose()

    db = IbkrSentimentDB(f"sqlite+aiosqlite:///{path}")
    await db.init()
    await db.init()  # idempotent
    [row] = await db.open_trades()
    assert row.filled_qty == 0
    await db.update_trade(
        OrderResult("ibsent-old", "9", OrderStatus.FILLED, Decimal("10"), Decimal("101"))
    )
    assert await db.open_trades() == []
    await db.close()


# ---- bot startup reconciliation -------------------------------------


@pytest.mark.asyncio
async def test_startup_reconciles_db_and_cancels_stale_orders(tmp_path: Path):
    from src.ibkr_sentiment.bot import build_default_bot
    from src.ibkr_sentiment.config import IbkrSentimentConfig, UniverseEntry

    cfg = IbkrSentimentConfig(
        universe=[UniverseEntry(symbol="AAPL")],
        db_url=f"sqlite+aiosqlite:///{tmp_path}/r.db",
    )
    db = IbkrSentimentDB(cfg.db_url)
    await db.init()
    await db.record_trade(
        _target("AAPL", "10"),
        OrderResult("ibsent-crashed", "B-AAPL", OrderStatus.SUBMITTED, Decimal(0), Decimal(0)),
    )
    await db.close()

    broker = ScriptedBroker()
    # While we were down, the order filled at the broker...
    broker.known["ibsent-crashed"] = _snap(OrderStatus.FILLED, "10", "100.5")
    broker.known["ibsent-crashed"].client_order_id = "ibsent-crashed"
    # ...and another of ours is still working.
    broker.working = [OpenOrderView("ibsent-live", "B-MSFT", "MSFT", Decimal("5"))]

    bot = build_default_bot(cfg, broker)
    await bot.start()
    try:
        assert await bot.db.open_trades() == []
        assert broker.cancelled == ["B-MSFT"]
        assert bot.startup_reconciliation.stale_cancelled == ["ibsent-live"]
    finally:
        await bot.stop()
