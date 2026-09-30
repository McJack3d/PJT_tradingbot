"""Contract qualification (audit defect #9, second half).

Before: every call built an unqualified Stock(symbol, "SMART", "USD")
and let IBKR guess the listing; option/future positions also leaked
into positions(), where flatten_all() would have sold the underlying
STOCK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.ibkr_sentiment.broker.base import OrderRequest, OrderSide, OrderType
from src.ibkr_sentiment.broker.ibkr import ContractError, ContractSpec, IbkrBroker
from src.ibkr_sentiment.broker.paper import PaperBroker
from src.ibkr_sentiment.config import RiskOverlayConfig
from src.ibkr_sentiment.execution.engine import ExecutionEngine
from src.ibkr_sentiment.risk.overlay import RiskOverlay
from src.ibkr_sentiment.signal_engine.dollar_neutral import TargetPosition
from src.ibkr_sentiment.signal_engine.mapping import Side


@dataclass
class _Contract:
    symbol: str = ""
    exchange: str = ""
    currency: str = ""
    primaryExchange: str = ""
    conId: int = 0
    secType: str = ""


def _stock(symbol, exchange, currency, primaryExchange=""):
    return _Contract(symbol, exchange, currency, primaryExchange, secType="STK")


FAKE_IB_INSYNC = SimpleNamespace(
    Stock=_stock,
    Contract=lambda conId, exchange: _Contract(conId=conId, exchange=exchange),
    MarketOrder=lambda action, qty: SimpleNamespace(action=action, totalQuantity=qty),
)


@dataclass
class _FakeIB:
    # What IBKR "knows": resolve(wanted) -> list of matches
    resolve: object = None
    qualify_calls: list = field(default_factory=list)
    positions_: list = field(default_factory=list)

    async def qualifyContractsAsync(self, wanted):
        self.qualify_calls.append(wanted)
        return self.resolve(wanted)

    def reqMktData(self, contract, *a):
        return SimpleNamespace(bid=100.0, ask=100.2, last=100.1, close=99.0)

    def cancelMktData(self, contract):
        pass

    def positions(self, account=""):
        return self.positions_

    def portfolio(self, account=""):
        return []


def _aapl(wanted):
    c = _Contract("AAPL", "SMART", "USD", "NASDAQ", conId=265598, secType="STK")
    return [c] if wanted.symbol in ("AAPL", "") else []


def _broker(ib, monkeypatch, specs=None) -> IbkrBroker:
    monkeypatch.setattr(IbkrBroker, "_ib_insync", staticmethod(lambda: FAKE_IB_INSYNC))
    b = IbkrBroker(contract_specs=specs)
    b._ib = ib
    b._quote_poll_s = 0.0
    return b


@pytest.mark.asyncio
async def test_qualifies_once_then_caches(monkeypatch):
    ib = _FakeIB(resolve=_aapl)
    b = _broker(ib, monkeypatch, {"AAPL": ContractSpec(primary_exchange="NASDAQ")})
    await b.quote("AAPL")
    await b.quote("AAPL")
    [wanted] = ib.qualify_calls
    assert (wanted.symbol, wanted.exchange, wanted.primaryExchange) == ("AAPL", "SMART", "NASDAQ")
    assert b._contract_cache["AAPL|SMART|USD"].conId == 265598


@pytest.mark.asyncio
async def test_con_id_spec_qualifies_by_id(monkeypatch):
    ib = _FakeIB(resolve=_aapl)
    b = _broker(ib, monkeypatch, {"AAPL": ContractSpec(con_id=265598)})
    await b.quote("AAPL")
    [wanted] = ib.qualify_calls
    assert (wanted.conId, wanted.symbol) == (265598, "")


@pytest.mark.asyncio
async def test_ambiguous_symbol_raises_with_hint(monkeypatch):
    b = _broker(_FakeIB(resolve=lambda w: []), monkeypatch)
    with pytest.raises(ContractError, match="primary_exchange or con_id"):
        await b.quote("XYZ")


@pytest.mark.asyncio
async def test_wrong_instrument_is_rejected(monkeypatch):
    def as_future(w):
        return [_Contract("AAPL", "SMART", "USD", conId=1, secType="FUT")]

    b = _broker(_FakeIB(resolve=as_future), monkeypatch)
    with pytest.raises(ContractError, match="expected a USD stock"):
        await b.quote("AAPL")


@pytest.mark.asyncio
async def test_unqualifiable_order_is_reported_not_sent(monkeypatch):
    b = _broker(_FakeIB(resolve=lambda w: []), monkeypatch)
    with pytest.raises(ContractError):
        await b.place_order(OrderRequest("XYZ", OrderSide.BUY, Decimal("1"), OrderType.MARKET))


@pytest.mark.asyncio
async def test_positions_ignore_non_stock_instruments(monkeypatch):
    stock = SimpleNamespace(contract=_Contract("AAPL", secType="STK", conId=1), position=10, avgCost=100)
    option = SimpleNamespace(contract=_Contract("AAPL", secType="OPT", conId=2), position=5, avgCost=3)
    ib = _FakeIB(resolve=_aapl, positions_=[stock, option])
    b = _broker(ib, monkeypatch)
    b._contract_cache["AAPL|SMART|USD"] = _Contract("AAPL", conId=1, secType="STK")
    out = await b.positions()
    assert [(p.symbol, p.qty) for p in out] == [("AAPL", Decimal("10"))]


@pytest.mark.asyncio
async def test_engine_turns_quote_failure_into_skipped_order():
    class _Broken(PaperBroker):
        async def quote(self, symbol):
            raise ContractError(f"{symbol}: unknown or ambiguous IBKR contract")

    broker = _Broken()
    engine = ExecutionEngine(
        broker=broker,
        overlay=RiskOverlay(
            cfg=RiskOverlayConfig(max_position_pct=Decimal("0.5"), max_net_exposure_pct=Decimal("1")),
            starting_equity=Decimal("100000"),
        ),
    )
    res = await engine.execute_basket(
        [TargetPosition("XYZ", Side.LONG, Decimal("1"), Decimal("100"), "")],
        account=await broker.account_summary(),
        current_positions={},
    )
    assert res.placed == []
    assert any("quote failed" in e[1] and "ambiguous" in e[1] for e in res.errors)
