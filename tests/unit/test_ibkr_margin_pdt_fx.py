"""Margin what-if, PDT budget and FX (audit defect #14).

Before: no pre-trade margin check, no pattern-day-trader awareness
(the 4h signal window makes same-day round trips routine), and a
non-USD base account's NetLiquidation (e.g. EUR) was used as if it
were USD for position sizing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ibkr_sentiment.broker.base import (
    AccountSummary,
    MarginImpact,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
)
from src.ibkr_sentiment.broker.ibkr import FxRateUnavailableError, IbkrBroker
from src.ibkr_sentiment.broker.paper import PaperBroker
from src.ibkr_sentiment.config import RiskOverlayConfig
from src.ibkr_sentiment.execution.engine import ExecutionEngine
from src.ibkr_sentiment.risk.overlay import RiskOverlay
from src.ibkr_sentiment.signal_engine.dollar_neutral import TargetPosition
from src.ibkr_sentiment.signal_engine.mapping import Side
from src.ibkr_sentiment.state.db import IbkrSentimentDB

D = Decimal


class _MarginBroker(PaperBroker):
    """Each risk-increasing order adds 40 of initial margin; equity 100."""

    def __init__(self, impact="scripted"):
        super().__init__()
        self.impact = impact
        self.what_if_calls: list[str] = []
        for s in ("A", "B", "C"):
            self.set_quote(s, bid=D("100"), ask=D("100"))

    async def what_if(self, req):
        self.what_if_calls.append(req.symbol)
        if self.impact == "boom":
            raise RuntimeError("what-if timed out")
        if self.impact is None:
            return None
        return MarginImpact(D("40"), D("40"), D("100"))


def _engine(broker, **kw) -> ExecutionEngine:
    cfg = RiskOverlayConfig(
        max_position_pct=D("1"), max_gross_exposure_pct=D("5"), max_net_exposure_pct=D("5")
    )
    return ExecutionEngine(
        broker=broker,
        overlay=RiskOverlay(cfg=cfg, starting_equity=D("100000")),
        order_style="market",
        **kw,
    )


def _t(sym: str, qty: str) -> TargetPosition:
    q = D(qty)
    return TargetPosition(sym, Side.LONG if q > 0 else Side.SHORT, q, abs(q) * 100, "")


def _acct(dtr=None) -> AccountSummary:
    return AccountSummary(D("100000"), D("100000"), D("0"), day_trades_remaining=dtr)


# ---- margin ---------------------------------------------------------


@pytest.mark.asyncio
async def test_margin_is_cumulative_across_the_batch():
    broker = _MarginBroker()
    res = await _engine(broker).execute_basket(
        [_t("A", "1"), _t("B", "1"), _t("C", "1")], account=_acct(), current_positions={}
    )
    assert [d.symbol for d, _ in res.placed] == ["A", "B"]  # 40, 80 <= 90; 120 > 90
    [(d, why)] = res.rejected_by_risk
    assert d.symbol == "C" and "initial margin 120" in why and "90%" in why


@pytest.mark.asyncio
@pytest.mark.parametrize(("impact", "why"), [(None, "unavailable"), ("boom", "failed")])
async def test_missing_margin_check_blocks_the_order(impact, why):
    broker = _MarginBroker(impact)
    res = await _engine(broker).execute_basket([_t("A", "1")], account=_acct(), current_positions={})
    assert res.placed == []
    assert why in res.rejected_by_risk[0][1]


@pytest.mark.asyncio
async def test_risk_reducing_orders_and_dry_run_skip_what_if():
    broker = _MarginBroker(None)  # would block anything it checked
    res = await _engine(broker).execute_basket(
        [_t("A", "0")], account=_acct(), current_positions={"A": D("5")}
    )
    assert [d.symbol for d, _ in res.placed] == ["A"]
    dry = await _engine(broker, dry_run=True).execute_basket(
        [_t("B", "1")], account=_acct(), current_positions={}
    )
    assert [d.symbol for d, _ in dry.skipped_dry_run] == ["B"]
    assert broker.what_if_calls == []


@pytest.mark.asyncio
async def test_paper_what_if_uses_reg_t_initial_margin():
    broker = PaperBroker(starting_cash=D("10000"))
    broker.set_quote("A", bid=D("100"), ask=D("100"))
    impact = await broker.what_if(OrderRequest("A", OrderSide.BUY, D("30")))
    assert impact.init_margin_change == D("1500.0")
    assert impact.equity_with_loan_after == D("10000")


# ---- PDT ------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dtr", "opened_today", "current", "target", "held"),
    [
        (0, {"A"}, "5", "0", True),  # closing a same-day position: held
        (0, {"A"}, "5", "-3", True),  # flip also closes it
        (0, set(), "5", "0", False),  # position from a prior day: fine
        (1, {"A"}, "5", "0", False),  # a day trade left: fine
        (None, {"A"}, "5", "0", False),  # unlimited / cash account
        (0, {"A"}, "5", "8", False),  # adding, not closing
        (0, {"A"}, "0", "5", False),  # opening
    ],
)
async def test_pdt_holds_same_day_round_trips_at_zero(dtr, opened_today, current, target, held):
    broker = _MarginBroker()
    cur = {"A": D(current)} if D(current) else {}
    res = await _engine(broker, margin_check=False).execute_basket(
        [_t("A", target)], account=_acct(dtr), current_positions=cur, opened_today=opened_today
    )
    blocked = [why for _, why in res.rejected_by_risk if "PDT" in why]
    assert bool(blocked) is held


# ---- FX + IBKR parsing ----------------------------------------------


def _ibkr(rows, values=()):
    class _IB:
        async def accountSummaryAsync(self, account=""):
            return [SimpleNamespace(tag=t, value=v, currency=c) for t, v, c in rows]

        def accountValues(self, account=""):
            return [SimpleNamespace(tag=t, value=v, currency=c) for t, v, c in values]

    b = IbkrBroker()
    b._ib = _IB()
    return b


@pytest.mark.asyncio
async def test_eur_base_account_is_converted_to_usd():
    b = _ibkr(
        [
            ("NetLiquidation", "90000", "EUR"),
            ("AvailableFunds", "45000", "EUR"),
            ("GrossPositionValue", "18000", "EUR"),
            ("DayTradesRemaining", "2", ""),
        ],
        [("ExchangeRate", "0.9", "USD"), ("ExchangeRate", "1", "EUR")],
    )
    a = await b.account_summary()
    assert (a.net_liquidation, a.available_funds, a.gross_position_value) == (
        D("100000"),
        D("50000"),
        D("20000"),
    )
    assert (a.currency, a.base_currency, a.fx_rate, a.day_trades_remaining) == (
        "USD",
        "EUR",
        D("0.9"),
        2,
    )


@pytest.mark.asyncio
async def test_missing_fx_rate_refuses_to_size():
    b = _ibkr([("NetLiquidation", "90000", "EUR")])
    with pytest.raises(FxRateUnavailableError):
        await b.account_summary()


@pytest.mark.asyncio
async def test_usd_account_unconverted_and_unlimited_day_trades():
    b = _ibkr([("NetLiquidation", "30000", "USD"), ("DayTradesRemaining", "-1", "")])
    a = await b.account_summary()
    assert a.net_liquidation == D("30000") and a.fx_rate == 1
    assert a.day_trades_remaining is None


@pytest.mark.asyncio
async def test_ibkr_what_if_parses_order_state(monkeypatch):
    fake = SimpleNamespace(MarketOrder=lambda a, q: SimpleNamespace(action=a, totalQuantity=q))
    monkeypatch.setattr(IbkrBroker, "_ib_insync", staticmethod(lambda: fake))
    states = [
        SimpleNamespace(
            initMarginChange="-1250.5",
            initMarginAfter="8000",
            equityWithLoanAfter="50000",
            warningText="",
        ),
        SimpleNamespace(
            initMarginChange="1.7976931348623157E308",
            initMarginAfter="",
            equityWithLoanAfter="50000",
            warningText="",
        ),
    ]

    class _IB:
        async def whatIfOrderAsync(self, contract, order):
            return states.pop(0)

    b = IbkrBroker()
    b._ib = _IB()
    b._contract_cache["A|SMART|USD"] = object()
    req = OrderRequest("A", OrderSide.BUY, D("1"))
    ok = await b.what_if(req)
    assert (ok.init_margin_change, ok.init_margin_after, ok.equity_with_loan_after) == (
        D("-1250.5"),
        D("8000"),
        D("50000"),
    )
    assert await b.what_if(req) is None  # unset values -> unknown


# ---- DB -------------------------------------------------------------


@pytest.mark.asyncio
async def test_symbols_filled_since(tmp_path: Path):
    db = IbkrSentimentDB(f"sqlite+aiosqlite:///{tmp_path}/p.db")
    await db.init()
    now = datetime.now(UTC)

    async def trade(sym, filled, when):
        await db.record_trade(
            _t(sym, "1"),
            OrderResult(f"c-{sym}", "b", OrderStatus.FILLED, D(filled), D("1"), submitted_at=when),
        )

    await trade("A", "1", now)
    await trade("B", "0", now)  # never filled
    await trade("C", "1", now - timedelta(days=2))
    assert await db.symbols_filled_since(now - timedelta(hours=1)) == {"A"}
    await db.close()
