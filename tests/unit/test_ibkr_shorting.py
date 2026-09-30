"""Short-sale checks (audit defect #7).

Before: shorts were sent with no borrow check, no hard-to-borrow
handling and no Rule 201 (SSR) awareness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.ibkr_sentiment.broker.base import Bar, ShortInfo
from src.ibkr_sentiment.broker.ibkr import IbkrBroker
from src.ibkr_sentiment.broker.paper import PaperBroker
from src.ibkr_sentiment.config import ExecutionConfig, RiskOverlayConfig
from src.ibkr_sentiment.execution.engine import ExecutionEngine
from src.ibkr_sentiment.risk.overlay import RiskOverlay, trading_day
from src.ibkr_sentiment.risk.shorting import (
    ShortPolicy,
    evaluate_short,
    short_sale_qty,
    ssr_triggered,
)
from src.ibkr_sentiment.signal_engine.dollar_neutral import TargetPosition
from src.ibkr_sentiment.signal_engine.mapping import Side

D = Decimal


def _info(shares="1000000", level="3", ssr=False, sym="TSLA") -> ShortInfo:
    return ShortInfo(
        symbol=sym,
        shortable_shares=None if shares is None else D(shares),
        shortable_level=None if level is None else D(level),
        ssr_active=ssr,
    )


# ---- pure rules -----------------------------------------------------


@pytest.mark.parametrize(
    ("cur", "delta", "short"),
    [
        ("0", "-10", "10"),  # open a short
        ("5", "-15", "10"),  # flip long 5 -> short 10
        ("-5", "-5", "5"),  # add to a short
        ("10", "-4", "0"),  # trim a long
        ("-10", "4", "0"),  # cover part of a short
        ("0", "10", "0"),  # buy
    ],
)
def test_short_sale_qty(cur, delta, short):
    assert short_sale_qty(D(cur), D(delta)) == D(short)


def test_ssr_triggers_at_ten_percent_drop():
    assert ssr_triggered(prev_close=D("100"), day_low=D("90"), last=D("95")) is True
    assert ssr_triggered(prev_close=D("100"), day_low=D("90.01"), last=D("95")) is False
    assert ssr_triggered(prev_close=D("100"), day_low=None, last=D("89")) is True
    assert ssr_triggered(prev_close=None, day_low=D("1"), last=None) is False
    assert ssr_triggered(prev_close=None, day_low=None, last=None, carried_over=True) is True


@pytest.mark.parametrize(
    ("info", "policy", "ok", "why"),
    [
        (None, ShortPolicy(), False, "unknown"),
        (_info(ssr=True), ShortPolicy(), False, "Rule 201"),
        (_info(ssr=True), ShortPolicy(block_under_ssr=False), True, "allowed"),
        (_info(level="1.0"), ShortPolicy(), False, "not available"),
        (_info(level="2.0"), ShortPolicy(), False, "hard to borrow"),
        (_info(level="2.0"), ShortPolicy(allow_hard_to_borrow=True), True, "allowed"),
        (_info(shares="150"), ShortPolicy(), False, "< 200"),  # need 2 x 100
        (_info(shares="200"), ShortPolicy(), True, "allowed"),
        (_info(shares=None), ShortPolicy(), False, "unknown"),
        (_info(shares="200", level=None), ShortPolicy(), True, "allowed"),
        (None, ShortPolicy(enabled=False), True, "no short"),
    ],
)
def test_evaluate_short(info, policy, ok, why):
    v = evaluate_short(info, D("100"), policy)
    assert v.ok is ok
    assert why in v.reason


# ---- engine ---------------------------------------------------------


def _engine(broker, **risk) -> ExecutionEngine:
    cfg = RiskOverlayConfig(
        max_position_pct=D("0.5"),
        max_gross_exposure_pct=D("2"),
        max_net_exposure_pct=D(risk.get("net", "1")),
    )
    return ExecutionEngine(
        broker=broker,
        overlay=RiskOverlay(cfg=cfg, starting_equity=D("100000")),
        order_style="market",
        short_policy=ExecutionConfig().short_policy(),
    )


def _t(sym: str, qty: str) -> TargetPosition:
    q = D(qty)
    side = Side.LONG if q > 0 else Side.SHORT if q < 0 else Side.FLAT
    return TargetPosition(sym, side, q, abs(q) * 100, "")


async def _run(engine, broker, targets, current=None):
    for sym in ("AAPL", "TSLA"):
        broker.set_quote(sym, bid=D("100"), ask=D("100"))
    return await engine.execute_basket(
        targets,
        account=await broker.account_summary(),
        current_positions=current or {},
        marks={"AAPL": D("100"), "TSLA": D("100")},
    )


@pytest.mark.asyncio
async def test_blocked_short_is_not_sent():
    broker = PaperBroker()
    broker.set_short_info(_info(level="1.0"))
    res = await _run(_engine(broker), broker, [_t("AAPL", "10"), _t("TSLA", "-10")])
    placed = {d.symbol for d, _ in res.placed}
    assert placed == {"AAPL"}
    [(d, why)] = res.rejected_by_risk
    assert d.symbol == "TSLA" and "not available" in why


@pytest.mark.asyncio
async def test_blocked_short_rebalances_rest_of_basket():
    """With the short gone the long leg alone breaches the net cap, so
    the net check trims it — shorts are filtered before gross/net."""
    broker = PaperBroker()
    broker.set_short_info(None, symbol="TSLA")  # unknown -> blocked
    res = await _run(
        _engine(broker, net="0.005"), broker, [_t("AAPL", "10"), _t("TSLA", "-10")]
    )
    assert res.placed == []
    reasons = sorted(why for _, why in res.rejected_by_risk)
    assert any("net" in r for r in reasons)
    assert any("unknown" in r for r in reasons)


@pytest.mark.asyncio
async def test_blocked_flip_still_closes_the_long():
    broker = PaperBroker()
    broker.set_short_info(_info(ssr=True))
    await broker.connect()
    broker.set_quote("TSLA", bid=D("100"), ask=D("100"))
    from src.ibkr_sentiment.broker.base import OrderRequest, OrderSide

    await broker.place_order(OrderRequest("TSLA", OrderSide.BUY, D("5")))
    res = await _run(_engine(broker), broker, [_t("TSLA", "-10")], current={"TSLA": D("5")})
    [(d, final)] = res.placed
    assert d.target_qty == D("-5")  # sell to flat only
    assert "Rule 201" in d.reason
    assert await broker.positions() == []


@pytest.mark.asyncio
async def test_short_availability_error_fails_closed():
    class _Boom(PaperBroker):
        async def short_availability(self, symbol):
            raise RuntimeError("no data")

    broker = _Boom()
    res = await _run(_engine(broker), broker, [_t("TSLA", "-10")])
    assert res.placed == []
    assert any("short availability" in e[1] for e in res.errors)


@pytest.mark.asyncio
async def test_easy_to_borrow_short_is_sent():
    broker = PaperBroker()  # default: easy to borrow, no SSR
    res = await _run(_engine(broker), broker, [_t("TSLA", "-10")])
    assert [d.symbol for d, _ in res.placed] == ["TSLA"]


# ---- IbkrBroker -----------------------------------------------------


@dataclass
class _Ticker:
    shortableShares: float = float("nan")
    close: float = float("nan")
    low: float = float("nan")
    last: float = float("nan")
    ticks: list = field(default_factory=list)


@dataclass
class _FakeIB:
    ticker: _Ticker
    generic: list[str] = field(default_factory=list)
    cancelled: int = 0

    def reqMktData(self, contract, generic="", *a):
        self.generic.append(generic)
        return self.ticker

    def cancelMktData(self, contract):
        self.cancelled += 1


def _broker(ticker: _Ticker, bars: list[Bar] | None = None) -> tuple[IbkrBroker, list]:
    b = IbkrBroker()
    b._ib = _FakeIB(ticker)
    b._quote_poll_s = 0.0
    b._contract_cache["TSLA|SMART|USD"] = object()
    calls: list = []

    async def fake_bars(symbol, duration="60 D", bar_size="1 day"):
        calls.append(symbol)
        return bars or []

    b.historical_bars = fake_bars  # type: ignore[method-assign]
    return b, calls


def _bar(days_ago: int, close: str, low: str) -> Bar:
    ts = trading_day(datetime.now(UTC)) - timedelta(days=days_ago)
    return Bar("TSLA", ts, D(close), D(close), D(low), D(close), D("1"))


@pytest.mark.asyncio
async def test_ibkr_reads_tick_236_and_level_from_ticks():
    t = _Ticker(
        shortableShares=5_000_000.0,
        close=100.0,
        low=99.0,
        last=99.5,
        ticks=[SimpleNamespace(tickType=46, price=3.0)],
    )
    b, _ = _broker(t)
    info = await b.short_availability("TSLA")
    assert info == ShortInfo("TSLA", D("5000000.0"), D("3.0"), False)
    assert b._ib.generic == ["236"]
    assert b._ib.cancelled == 1


@pytest.mark.asyncio
async def test_ibkr_ssr_today_from_ticker():
    t = _Ticker(shortableShares=1e6, close=100.0, low=89.0, last=91.0)
    info = await _broker(t)[0].short_availability("TSLA")
    assert info.ssr_active is True


@pytest.mark.asyncio
async def test_ibkr_ssr_carried_over_from_yesterday_and_cached():
    bars = [_bar(3, "100", "99"), _bar(2, "100", "98"), _bar(1, "91", "89")]  # -11% low
    t = _Ticker(shortableShares=1e6, close=91.0, low=90.5, last=91.0)
    b, calls = _broker(t, bars)
    assert (await b.short_availability("TSLA")).ssr_active is True
    assert (await b.short_availability("TSLA")).ssr_active is True
    assert calls == ["TSLA"]  # one bars request per symbol per day


@pytest.mark.asyncio
async def test_ibkr_no_borrow_data_returns_none():
    info = await _broker(_Ticker())[0].short_availability("TSLA")
    assert info is None
