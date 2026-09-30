"""IbkrBroker mark-to-market tests against a fake ib_insync client.

Regression for audit defect #4: positions were marked at average cost,
so exposure, equity snapshots and risk checks never saw price moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import pytest

from src.ibkr_sentiment.broker.ibkr import IbkrBroker

NAN = float("nan")


@dataclass
class _Contract:
    symbol: str
    conId: int


@dataclass
class _Position:
    contract: _Contract
    position: float
    avgCost: float


@dataclass
class _PortfolioItem:
    contract: _Contract
    marketPrice: float


@dataclass
class _Ticker:
    bid: float = NAN
    ask: float = NAN
    last: float = NAN
    close: float = NAN


@dataclass
class _FakeIB:
    positions_: list[_Position] = field(default_factory=list)
    portfolio_: list[_PortfolioItem] = field(default_factory=list)
    tickers: dict[str, _Ticker] = field(default_factory=dict)
    cancelled: list[str] = field(default_factory=list)
    accounts_seen: list[str] = field(default_factory=list)

    def positions(self, account: str = ""):
        self.accounts_seen.append(account)
        return self.positions_

    def portfolio(self, account: str = ""):
        return self.portfolio_

    def reqMktData(self, contract, *args):
        return self.tickers.get(contract.symbol, _Ticker())

    def cancelMktData(self, contract):
        self.cancelled.append(contract.symbol)


AAPL = _Contract("AAPL", 265598)
MSFT = _Contract("MSFT", 272093)


def _broker(ib: _FakeIB, account: str | None = None) -> IbkrBroker:
    b = IbkrBroker(account=account)
    b._ib = ib
    b._quote_poll_s = 0.0
    # Pre-seed contracts so tests don't need ib_insync installed.
    for c in (AAPL, MSFT):
        b._contract_cache[f"{c.symbol}|SMART|USD"] = c
    return b


@pytest.mark.asyncio
async def test_positions_use_portfolio_mark_not_cost():
    ib = _FakeIB(
        positions_=[_Position(AAPL, 10, 150.0)],
        portfolio_=[_PortfolioItem(AAPL, 180.0)],
    )
    [p] = await _broker(ib).positions()
    assert p.mark_price == Decimal("180.0")
    assert p.mark_source == "portfolio"
    assert p.unrealized_pnl == Decimal("300.0")
    assert ib.cancelled == []  # no quote needed


@pytest.mark.asyncio
async def test_short_position_pnl_sign():
    ib = _FakeIB(
        positions_=[_Position(AAPL, -10, 150.0)],
        portfolio_=[_PortfolioItem(AAPL, 180.0)],
    )
    [p] = await _broker(ib).positions()
    assert p.unrealized_pnl == Decimal("-300.0")


@pytest.mark.asyncio
async def test_nan_portfolio_mark_falls_back_to_quote_mid():
    ib = _FakeIB(
        positions_=[_Position(AAPL, 10, 150.0)],
        portfolio_=[_PortfolioItem(AAPL, NAN)],
        tickers={"AAPL": _Ticker(bid=179.0, ask=181.0)},
    )
    [p] = await _broker(ib).positions()
    assert p.mark_price == Decimal("180")
    assert p.mark_source == "quote"
    assert ib.cancelled == ["AAPL"]  # subscription released


@pytest.mark.asyncio
async def test_no_market_price_falls_back_to_cost_and_flags_it():
    ib = _FakeIB(
        positions_=[_Position(AAPL, 10, 150.0)],
        portfolio_=[_PortfolioItem(AAPL, -1.0)],  # IB's "no price"
    )
    [p] = await _broker(ib).positions()
    assert p.mark_price == Decimal("150.0")
    assert p.mark_source == "cost"
    assert p.unrealized_pnl == 0


@pytest.mark.asyncio
async def test_zero_qty_positions_are_skipped_and_account_is_passed():
    ib = _FakeIB(
        positions_=[_Position(AAPL, 0, 150.0), _Position(MSFT, 5, 400.0)],
        portfolio_=[_PortfolioItem(MSFT, 410.0)],
    )
    out = await _broker(ib, account="U123").positions()
    assert [p.symbol for p in out] == ["MSFT"]
    assert ib.accounts_seen == ["U123"]


@pytest.mark.asyncio
async def test_quote_never_returns_nan():
    """Regression: NaN is truthy, so the old wait loop exited at once and
    returned Decimal('NaN') prices."""
    ib = _FakeIB(tickers={"AAPL": _Ticker()})
    q = await _broker(ib).quote("AAPL")
    assert (q.bid, q.ask, q.last) == (0, 0, 0)
    assert not any(v.is_nan() for v in (q.bid, q.ask, q.last))
    assert ib.cancelled == ["AAPL"]


@pytest.mark.asyncio
async def test_quote_falls_back_to_close_when_no_last():
    ib = _FakeIB(tickers={"AAPL": _Ticker(bid=99.0, ask=101.0, close=98.0)})
    q = await _broker(ib).quote("AAPL")
    assert q.last == Decimal("98.0")
    assert q.bid == Decimal("99.0")
