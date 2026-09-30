"""IBKR adapter built on `ib_insync`.

ib_insync wraps the native TWS / IB Gateway API in an asyncio event
loop, so our bot can keep WebSocket-style market data streams,
trailing stops, and order status callbacks fully decoupled from the
sentiment pipeline.

Lazy import: this module is safe to import without the `ibkr` extra
installed; the dependency is only required when you actually call
`connect()`. That keeps the unit-test environment minimal.

All client-facing methods go through a shared `_Limiter` (see
`rate_limiter.py`) so we never exceed IBKR's documented pacing.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from src.ibkr_sentiment.broker.base import (
    AccountSummary,
    Bar,
    Broker,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    PositionView,
    Quote,
)
from src.ibkr_sentiment.broker.rate_limiter import (
    BucketSpec,
    InMemoryRateLimiter,
    build_rate_limiter,
    per_minute,
    per_window,
)


def default_bucket_specs(
    orders_per_minute: int = 30,
    historical_requests_per_10min: int = 50,
    market_data_lines: int = 100,
) -> dict[str, BucketSpec]:
    return {
        "orders": per_minute(orders_per_minute),
        "historical": per_window(historical_requests_per_10min, 600.0),
        "market_data": per_minute(market_data_lines),
        "generic": per_minute(60),
    }


def _to_decimal(x: Any, default: str = "0") -> Decimal:
    if x is None or x == "":
        return Decimal(default)
    return Decimal(str(x))


def _status_from_ib(status: str) -> OrderStatus:
    s = (status or "").lower()
    if s in ("filled",):
        return OrderStatus.FILLED
    if s in ("submitted", "presubmitted"):
        return OrderStatus.SUBMITTED
    if s in ("partiallyfilled", "partially_filled"):
        return OrderStatus.PARTIALLY_FILLED
    if s in ("cancelled", "canceled"):
        return OrderStatus.CANCELED
    if s in ("apipending", "pendingsubmit", "pendingcancel"):
        return OrderStatus.PENDING
    if s in ("inactive", "apicancelled"):
        return OrderStatus.CANCELED
    if s == "rejected":
        return OrderStatus.REJECTED
    return OrderStatus.PENDING


class IbkrBroker(Broker):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 17,
        account: str | None = None,
        readonly: bool = False,
        connect_timeout_s: float = 15.0,
        rate_limiter: InMemoryRateLimiter | None = None,
        redis_url: str | None = None,
        orders_per_minute: int = 30,
        historical_requests_per_10min: int = 50,
        market_data_lines: int = 100,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account
        self.readonly = readonly
        self.connect_timeout_s = connect_timeout_s
        self._ib = None  # set on connect()
        self._limiter = rate_limiter or build_rate_limiter(
            redis_url,
            default_bucket_specs(
                orders_per_minute, historical_requests_per_10min, market_data_lines
            ),
        )
        self._contract_cache: dict[str, Any] = {}

    # ---- lazy SDK import --------------------------------------------

    @staticmethod
    def _ib_insync():
        try:
            import ib_insync  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "IbkrBroker requires the 'ibkr' extra. "
                "Install with: pip install -e '.[ibkr]'"
            ) from e
        return ib_insync

    # ---- connection -------------------------------------------------

    async def connect(self) -> None:
        ib_insync = self._ib_insync()
        self._ib = ib_insync.IB()
        await asyncio.wait_for(
            self._ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                readonly=self.readonly,
            ),
            timeout=self.connect_timeout_s,
        )

    async def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
        await self._limiter.close()

    async def is_connected(self) -> bool:
        return self._ib is not None and bool(self._ib.isConnected())

    # ---- helpers ----------------------------------------------------

    def _contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD"):
        key = f"{symbol}|{exchange}|{currency}"
        if key in self._contract_cache:
            return self._contract_cache[key]
        ib_insync = self._ib_insync()
        c = ib_insync.Stock(symbol, exchange, currency)
        self._contract_cache[key] = c
        return c

    # ---- account / data ---------------------------------------------

    async def account_summary(self) -> AccountSummary:
        assert self._ib is not None, "broker not connected"
        await self._limiter.acquire("generic")
        rows = await self._ib.accountSummaryAsync(self.account or "")
        by_tag: dict[str, str] = {}
        currency = "USD"
        for row in rows:
            by_tag[row.tag] = row.value
            currency = row.currency or currency
        return AccountSummary(
            net_liquidation=_to_decimal(by_tag.get("NetLiquidation")),
            available_funds=_to_decimal(by_tag.get("AvailableFunds")),
            gross_position_value=_to_decimal(by_tag.get("GrossPositionValue")),
            currency=currency,
        )

    async def positions(self) -> list[PositionView]:
        assert self._ib is not None
        await self._limiter.acquire("generic")
        positions = self._ib.positions(self.account or "")
        out: list[PositionView] = []
        for p in positions:
            sym = p.contract.symbol
            qty = _to_decimal(p.position)
            avg = _to_decimal(p.avgCost)
            mark = avg  # IBKR computes unrealized server-side; we keep
            # the cost as a placeholder and let the strategy mark to
            # whatever quote source it uses.
            out.append(
                PositionView(
                    symbol=sym,
                    qty=qty,
                    avg_cost=avg,
                    mark_price=mark,
                    unrealized_pnl=Decimal("0"),
                )
            )
        return out

    async def quote(self, symbol: str) -> Quote:
        assert self._ib is not None
        await self._limiter.acquire("market_data")
        contract = self._contract(symbol)
        ticker = self._ib.reqMktData(contract, "", False, False)
        # Wait for a tick; if the market is closed and no last is
        # populated, fall back to bid/ask.
        for _ in range(50):
            if ticker.last or ticker.bid or ticker.ask:
                break
            await asyncio.sleep(0.05)
        return Quote(
            symbol=symbol,
            bid=_to_decimal(ticker.bid),
            ask=_to_decimal(ticker.ask),
            last=_to_decimal(ticker.last or ticker.close),
        )

    async def historical_bars(
        self, symbol: str, duration: str = "60 D", bar_size: str = "1 day"
    ) -> list[Bar]:
        assert self._ib is not None
        await self._limiter.acquire("historical")
        contract = self._contract(symbol)
        bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        out: list[Bar] = []
        for b in bars or []:
            out.append(
                Bar(
                    symbol=symbol,
                    ts=b.date if hasattr(b.date, "tzinfo") else b.date,  # type: ignore[arg-type]
                    open=_to_decimal(b.open),
                    high=_to_decimal(b.high),
                    low=_to_decimal(b.low),
                    close=_to_decimal(b.close),
                    volume=_to_decimal(b.volume),
                )
            )
        return out

    # ---- trading ----------------------------------------------------

    async def place_order(self, req: OrderRequest) -> OrderResult:
        assert self._ib is not None
        await self._limiter.acquire("orders")
        ib_insync = self._ib_insync()
        contract = self._contract(req.symbol, req.exchange, req.currency)

        if req.order_type == OrderType.MARKET:
            order = ib_insync.MarketOrder(req.side.value, float(req.qty))
        elif req.order_type == OrderType.LIMIT:
            if req.limit_price is None:
                raise ValueError("LIMIT order requires limit_price")
            order = ib_insync.LimitOrder(
                req.side.value, float(req.qty), float(req.limit_price)
            )
        elif req.order_type == OrderType.TRAIL:
            if req.trail_percent is None:
                raise ValueError("TRAIL order requires trail_percent")
            order = ib_insync.Order(
                action=req.side.value,
                totalQuantity=float(req.qty),
                orderType="TRAIL",
                trailingPercent=float(req.trail_percent),
            )
        else:
            raise ValueError(f"unsupported order type: {req.order_type}")

        order.tif = req.tif
        if req.client_order_id:
            # IBKR doesn't take an arbitrary client-order-id like Binance,
            # but we set `orderRef` so the bot's own logs can join back.
            order.orderRef = req.client_order_id
        trade = self._ib.placeOrder(contract, order)
        # Don't block until fill — return current snapshot. The
        # execution engine has its own follow-up loop.
        await asyncio.sleep(0)
        status = _status_from_ib(trade.orderStatus.status)
        return OrderResult(
            client_order_id=req.client_order_id or str(trade.order.orderId),
            broker_order_id=str(trade.order.orderId),
            status=status,
            filled_qty=_to_decimal(trade.orderStatus.filled),
            avg_fill_price=_to_decimal(trade.orderStatus.avgFillPrice),
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        assert self._ib is not None
        await self._limiter.acquire("generic")
        for trade in list(self._ib.openTrades()):
            if str(trade.order.orderId) == str(broker_order_id):
                self._ib.cancelOrder(trade.order)
                return

    async def cancel_all_orders(self) -> None:
        assert self._ib is not None
        await self._limiter.acquire("generic")
        # Global cancel covers orders from every client id, including
        # ones placed before a restart that this session never saw.
        self._ib.reqGlobalCancel()
