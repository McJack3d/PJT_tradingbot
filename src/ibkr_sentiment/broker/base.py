"""Abstract broker interface for the IBKR sentiment bot.

A thin shape on top of which we hang the real `IbkrBroker` (ib_insync)
and the `PaperBroker` used for tests and paper-mode runs. Strategy /
execution / risk modules only ever talk to this surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    TRAIL = "TRAIL"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


@dataclass(slots=True)
class Quote:
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class Bar:
    symbol: str
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(slots=True)
class PositionView:
    symbol: str
    qty: Decimal  # signed: positive = long, negative = short
    avg_cost: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    # Where mark_price came from: "portfolio" (broker's own mark),
    # "quote" (live market data), or "cost" (no market price available —
    # exposure figures for this position are stale).
    mark_source: str = "portfolio"


@dataclass(slots=True)
class AccountSummary:
    net_liquidation: Decimal
    available_funds: Decimal
    gross_position_value: Decimal
    currency: str = "USD"


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    qty: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    trail_percent: Decimal | None = None
    client_order_id: str = ""
    tif: str = "DAY"
    exchange: str = "SMART"
    currency: str = "USD"


@dataclass(slots=True)
class OrderResult:
    client_order_id: str
    broker_order_id: str | None
    status: OrderStatus
    filled_qty: Decimal
    avg_fill_price: Decimal
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class OpenOrderView:
    client_order_id: str  # IBKR orderRef; "" for orders placed outside the bot
    broker_order_id: str
    symbol: str
    remaining_qty: Decimal  # signed: + buy, - sell


class Broker(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def is_connected(self) -> bool: ...

    @abstractmethod
    async def account_summary(self) -> AccountSummary: ...

    @abstractmethod
    async def positions(self) -> list[PositionView]: ...

    @abstractmethod
    async def quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    async def historical_bars(
        self, symbol: str, duration: str = "60 D", bar_size: str = "1 day"
    ) -> list[Bar]: ...

    @abstractmethod
    async def place_order(self, req: OrderRequest) -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> None: ...

    @abstractmethod
    async def wait_for_fill(self, order: OrderResult, timeout_s: float) -> OrderResult:
        """Wait until `order` reaches a terminal status or `timeout_s`
        elapses; return the latest snapshot either way."""

    @abstractmethod
    async def order_status(self, client_order_id: str) -> OrderResult | None:
        """Look up an order the broker still knows about by our
        client order id. None if the broker has no record of it."""

    @abstractmethod
    async def open_orders(self) -> list[OpenOrderView]:
        """Every working order on the account, ours or not."""

    @abstractmethod
    async def cancel_all_orders(self) -> None:
        """Cancel every open order, including ones placed before a
        restart."""

    async def flatten_all(self) -> list[OrderResult]:
        """Close every open position with market orders. Default
        implementation works for any broker that exposes `positions()`
        and `place_order()`."""
        results: list[OrderResult] = []
        for pos in await self.positions():
            if pos.qty == 0:
                continue
            side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
            req = OrderRequest(
                symbol=pos.symbol,
                side=side,
                qty=abs(pos.qty),
                order_type=OrderType.MARKET,
                client_order_id=f"flatten-{pos.symbol}",
            )
            results.append(await self.place_order(req))
        return results

    async def batch_place(
        self, requests: Iterable[OrderRequest]
    ) -> list[OrderResult]:
        out: list[OrderResult] = []
        for req in requests:
            out.append(await self.place_order(req))
        return out
