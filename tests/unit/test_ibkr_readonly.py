"""Dry-run connects read-only (audit defect #13).

Before: dry_run opened a full read-write IB session and relied only on
the engine's dry_run flag to hold orders back.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.ibkr_sentiment.broker.base import OrderRequest, OrderSide
from src.ibkr_sentiment.broker.ibkr import IbkrBroker, ReadOnlyBrokerError
from src.ibkr_sentiment.config import (
    IbkrConnectionConfig,
    IbkrMode,
    IbkrSentimentConfig,
    UniverseEntry,
)
from src.ibkr_sentiment.main import _build_broker


def _cfg(mode: IbkrMode, readonly: bool = False) -> IbkrSentimentConfig:
    return IbkrSentimentConfig(
        mode=mode,
        universe=[UniverseEntry(symbol="AAPL")],
        ibkr=IbkrConnectionConfig(readonly=readonly),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "readonly"), [(IbkrMode.DRY_RUN, True), (IbkrMode.LIVE, False)])
async def test_dry_run_broker_is_read_only(mode, readonly):
    broker = await _build_broker(_cfg(mode))  # IbkrBroker doesn't connect until start()
    assert isinstance(broker, IbkrBroker)
    assert broker.readonly is readonly


def test_live_mode_rejects_read_only_config():
    with pytest.raises(ValueError, match="live cannot run"):
        _cfg(IbkrMode.LIVE, readonly=True)
    assert _cfg(IbkrMode.DRY_RUN, readonly=True).ibkr.readonly


class _NoCallsIB:
    def __getattr__(self, name):
        raise AssertionError(f"read-only broker touched ib.{name}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.place_order(OrderRequest("AAPL", OrderSide.BUY, Decimal("1"))),
        lambda b: b.cancel_order("1"),
        lambda b: b.cancel_all_orders(),
    ],
)
async def test_read_only_broker_refuses_order_changes(call):
    b = IbkrBroker(readonly=True)
    b._ib = _NoCallsIB()
    with pytest.raises(ReadOnlyBrokerError):
        await call(b)


@pytest.mark.asyncio
async def test_read_only_broker_refuses_flatten():
    class _IB:
        def positions(self, account=""):
            from types import SimpleNamespace

            return [
                SimpleNamespace(
                    contract=SimpleNamespace(symbol="AAPL", secType="STK", conId=1),
                    position=10,
                    avgCost=100,
                )
            ]

        def portfolio(self, account=""):
            from types import SimpleNamespace

            return [SimpleNamespace(contract=SimpleNamespace(conId=1), marketPrice=101.0)]

    b = IbkrBroker(readonly=True)
    b._ib = _IB()
    with pytest.raises(ReadOnlyBrokerError):
        await b.flatten_all()
