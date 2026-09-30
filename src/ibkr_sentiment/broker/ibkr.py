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
import math
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from src.ibkr_sentiment.bar_cache import bar_day
from src.ibkr_sentiment.broker.base import (
    AccountSummary,
    Bar,
    Broker,
    OpenOrderView,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    PositionView,
    Quote,
    ShortInfo,
)
from src.ibkr_sentiment.broker.rate_limiter import (
    BucketSpec,
    InMemoryRateLimiter,
    build_rate_limiter,
    per_minute,
    per_window,
)
from src.ibkr_sentiment.risk.overlay import trading_day
from src.ibkr_sentiment.risk.shorting import ssr_triggered
from src.logging_setup import log


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


def _price(x: Any) -> Decimal | None:
    """A usable price, or None. IB reports missing prices as NaN (which
    is truthy) or as -1, so plain truthiness checks are wrong."""
    if x is None or x == "":
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or f <= 0:
        return None
    return Decimal(str(x))


def _result_from_trade(trade: Any, *, fallback_client_id: str = "") -> OrderResult:
    filled = _to_decimal(trade.orderStatus.filled)
    status = _status_from_ib(trade.orderStatus.status)
    # IB reports a partial fill as "Submitted" with filled > 0.
    if filled > 0 and not status.is_terminal:
        status = OrderStatus.PARTIALLY_FILLED
    return OrderResult(
        client_order_id=fallback_client_id or str(trade.order.orderId),
        broker_order_id=str(trade.order.orderId),
        status=status,
        filled_qty=filled,
        avg_fill_price=_price(trade.orderStatus.avgFillPrice) or Decimal("0"),
    )


def _shortable_level(ticker: Any) -> Decimal | None:
    """IBKR's shortable indicator (generic tick 46). Newer clients expose
    it as `ticker.shortable`; ib_insync 0.9.x only records it in
    `ticker.ticks`."""
    value = getattr(ticker, "shortable", None)
    if value is None:
        for t in reversed(getattr(ticker, "ticks", None) or []):
            if getattr(t, "tickType", None) == 46:
                value = t.price
                break
    if value is None or math.isnan(float(value)) or float(value) <= 0:
        return None
    return Decimal(str(value))


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
    # Quote polling: up to 50 polls of 50ms each (overridable in tests).
    _quote_poll_s: float = 0.05
    _fill_poll_s: float = 0.25
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
        self._trades: dict[str, Any] = {}  # broker order id -> ib Trade
        # (symbol, NY trading day) -> Rule 201 carried over from yesterday
        self._ssr_carry: dict[tuple[str, date], bool] = {}

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
                # Subscribes account updates for this account, which is
                # what populates `ib.portfolio()` marks on multi-account
                # logins.
                account=self.account or "",
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
        """Open positions marked to market.

        Quantities come from `ib.positions()` (updated on every fill).
        Marks come from `ib.portfolio()` — IBKR's own mark, the same one
        behind NetLiquidation — falling back to a live quote, and only
        then to average cost (flagged via `mark_source="cost"`).
        """
        assert self._ib is not None
        await self._limiter.acquire("generic")
        account = self.account or ""
        marks: dict[int, Decimal] = {}
        for item in self._ib.portfolio(account):
            mark = _price(item.marketPrice)
            if mark is not None:
                marks[item.contract.conId] = mark
        out: list[PositionView] = []
        for p in self._ib.positions(account):
            qty = _to_decimal(p.position)
            if qty == 0:
                continue
            sym = p.contract.symbol
            avg = _to_decimal(p.avgCost)
            mark = marks.get(p.contract.conId)
            source = "portfolio"
            if mark is None:
                mark = await self._quote_mark(sym)
                source = "quote"
            if mark is None:
                mark, source = avg, "cost"
                log.warning("ibkr.positions.mark_fallback_to_cost", symbol=sym)
            out.append(
                PositionView(
                    symbol=sym,
                    qty=qty,
                    avg_cost=avg,
                    mark_price=mark,
                    unrealized_pnl=(mark - avg) * qty,
                    mark_source=source,
                )
            )
        return out

    async def _quote_mark(self, symbol: str) -> Decimal | None:
        try:
            q = await self.quote(symbol)
        except Exception as e:
            log.warning("ibkr.quote.error", symbol=symbol, error=str(e))
            return None
        if q.bid > 0 and q.ask > 0:
            return (q.bid + q.ask) / 2
        return q.last if q.last > 0 else None

    async def quote(self, symbol: str) -> Quote:
        assert self._ib is not None
        await self._limiter.acquire("market_data")
        contract = self._contract(symbol)
        ticker = self._ib.reqMktData(contract, "", False, False)
        try:
            # Wait up to ~2.5s for a tick. Outside market hours `last`
            # may never arrive, so bid/ask or the prior close will do.
            for _ in range(50):
                if any(
                    _price(v) is not None
                    for v in (ticker.last, ticker.bid, ticker.ask)
                ):
                    break
                await asyncio.sleep(self._quote_poll_s)
        finally:
            # Always release the line: each open subscription counts
            # against the account's market-data line limit.
            self._ib.cancelMktData(contract)
        zero = Decimal("0")
        return Quote(
            symbol=symbol,
            bid=_price(ticker.bid) or zero,
            ask=_price(ticker.ask) or zero,
            last=_price(ticker.last) or _price(ticker.close) or zero,
        )

    async def short_availability(self, symbol: str) -> ShortInfo | None:
        """Borrow data via generic tick 236 (shortable shares = tick 89,
        shortable level = tick 46) plus Rule 201 state from the same
        ticker's prior close / day low and yesterday's daily bar."""
        assert self._ib is not None
        await self._limiter.acquire("market_data")
        contract = self._contract(symbol)
        ticker = self._ib.reqMktData(contract, "236", False, False)
        try:
            for _ in range(50):
                if _price(ticker.shortableShares) is not None or _shortable_level(ticker):
                    break
                await asyncio.sleep(self._quote_poll_s)
        finally:
            self._ib.cancelMktData(contract)
        shares = ticker.shortableShares
        shares_d = None if shares is None or math.isnan(float(shares)) else Decimal(str(shares))
        level = _shortable_level(ticker)
        if shares_d is None and level is None:
            return None
        ssr = ssr_triggered(
            prev_close=_price(ticker.close),
            day_low=_price(ticker.low),
            last=_price(ticker.last),
            carried_over=await self._ssr_carried_over(symbol),
        )
        return ShortInfo(
            symbol=symbol, shortable_shares=shares_d, shortable_level=level, ssr_active=ssr
        )

    async def _ssr_carried_over(self, symbol: str) -> bool:
        """True if Rule 201 triggered on the previous session (it stays
        in force for the whole next day). Cached per symbol per day."""
        today = trading_day(datetime.now(UTC))
        key = (symbol, today)
        if key not in self._ssr_carry:
            try:
                bars = await self.historical_bars(symbol, duration="5 D", bar_size="1 day")
            except Exception as e:
                log.warning("ibkr.ssr.bars_error", symbol=symbol, error=str(e))
                return False  # don't cache a failure
            done = [b for b in bars if bar_day(b.ts) < today]
            carried = len(done) >= 2 and ssr_triggered(
                prev_close=done[-2].close, day_low=done[-1].low, last=None
            )
            self._ssr_carry[key] = carried
        return self._ssr_carry[key]

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
        if req.algo:
            order.algoStrategy = req.algo
            order.algoParams = [ib_insync.TagValue("adaptivePriority", "Normal")]
        if req.client_order_id:
            # IBKR doesn't take an arbitrary client-order-id like Binance,
            # but we set `orderRef` so the bot's own logs can join back.
            order.orderRef = req.client_order_id
        trade = self._ib.placeOrder(contract, order)
        self._trades[str(trade.order.orderId)] = trade
        # Don't block until fill — return current snapshot. The
        # execution engine follows up with `wait_for_fill`.
        await asyncio.sleep(0)
        return _result_from_trade(trade, fallback_client_id=req.client_order_id)

    def _find_trade(self, broker_order_id: str | None):
        assert self._ib is not None
        if broker_order_id and broker_order_id in self._trades:
            return self._trades[broker_order_id]
        for trade in self._ib.trades():
            if str(trade.order.orderId) == str(broker_order_id):
                return trade
        return None

    async def wait_for_fill(self, order: OrderResult, timeout_s: float) -> OrderResult:
        trade = self._find_trade(order.broker_order_id)
        if trade is None:
            return order
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            result = _result_from_trade(trade, fallback_client_id=order.client_order_id)
            if result.status.is_terminal or loop.time() >= deadline:
                result.submitted_at = order.submitted_at
                return result
            await asyncio.sleep(self._fill_poll_s)

    async def order_status(self, client_order_id: str) -> OrderResult | None:
        """Search this session's trades by orderRef. IBKR replays open
        orders and today's completed orders on connect, so this covers a
        restart within the same trading day."""
        assert self._ib is not None
        for trade in self._ib.trades():
            if getattr(trade.order, "orderRef", "") == client_order_id:
                return _result_from_trade(trade, fallback_client_id=client_order_id)
        return None

    async def open_orders(self) -> list[OpenOrderView]:
        assert self._ib is not None
        await self._limiter.acquire("generic")
        out: list[OpenOrderView] = []
        for trade in self._ib.openTrades():
            remaining = _to_decimal(trade.orderStatus.remaining)
            if trade.order.action == "SELL":
                remaining = -remaining
            out.append(
                OpenOrderView(
                    client_order_id=getattr(trade.order, "orderRef", "") or "",
                    broker_order_id=str(trade.order.orderId),
                    symbol=trade.contract.symbol,
                    remaining_qty=remaining,
                )
            )
        return out

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
