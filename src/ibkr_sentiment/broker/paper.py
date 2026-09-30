"""In-memory paper broker.

Used in tests and in paper mode (no IB connection). It models the
asynchronous order lifecycle, applies trivial price slippage, and
tracks positions and PnL so the rest of the bot can exercise the full
order path without touching the network.

Quotes default to a flat synthetic price; callers can either
`set_quote()` directly, or `seed_bars()` with a list of bars to drive
the historical-data API.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from src.ibkr_sentiment.broker.base import (
    AccountSummary,
    Bar,
    Broker,
    OpenOrderView,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionView,
    Quote,
    ShortInfo,
)


class PaperBroker(Broker):
    def __init__(
        self,
        starting_cash: Decimal = Decimal("100000"),
        currency: str = "USD",
        slippage_bps: Decimal = Decimal("2"),
    ):
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.currency = currency
        self.slippage_bps = slippage_bps
        self._connected = False
        self._quotes: dict[str, Quote] = {}
        self._bars: dict[str, list[Bar]] = defaultdict(list)
        # symbol -> (qty, avg_cost)
        self._positions: dict[str, tuple[Decimal, Decimal]] = {}
        self._orders: dict[str, OrderResult] = {}
        # Non-marketable LIMIT orders: broker id -> (request, live result)
        self._resting: dict[str, tuple[OrderRequest, OrderResult]] = {}
        self._short_info: dict[str, ShortInfo | None] = {}
        self._lock = asyncio.Lock()

    # ---- lifecycle ---------------------------------------------------

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def is_connected(self) -> bool:
        return self._connected

    # ---- account / data ---------------------------------------------

    async def account_summary(self) -> AccountSummary:
        gross = Decimal("0")
        for sym, (qty, _) in self._positions.items():
            quote = self._quotes.get(sym)
            mark = quote.last if quote else Decimal("0")
            gross += abs(qty) * mark
        nlv = self.cash + sum(
            qty * (self._quotes[sym].last if sym in self._quotes else Decimal("0"))
            for sym, (qty, _) in self._positions.items()
        )
        return AccountSummary(
            net_liquidation=nlv,
            available_funds=self.cash,
            gross_position_value=gross,
            currency=self.currency,
        )

    async def positions(self) -> list[PositionView]:
        out: list[PositionView] = []
        for sym, (qty, avg_cost) in self._positions.items():
            if qty == 0:
                continue
            quoted = sym in self._quotes
            mark = self._quotes[sym].last if quoted else avg_cost
            out.append(
                PositionView(
                    symbol=sym,
                    qty=qty,
                    avg_cost=avg_cost,
                    mark_price=mark,
                    unrealized_pnl=(mark - avg_cost) * qty,
                    mark_source="quote" if quoted else "cost",
                )
            )
        return out

    async def quote(self, symbol: str) -> Quote:
        q = self._quotes.get(symbol)
        if q is not None:
            return q
        # No quote? Return a flat-but-non-zero placeholder so callers
        # don't crash on missing data during tests.
        return Quote(
            symbol=symbol,
            bid=Decimal("100"),
            ask=Decimal("100"),
            last=Decimal("100"),
        )

    async def historical_bars(
        self, symbol: str, duration: str = "60 D", bar_size: str = "1 day"
    ) -> list[Bar]:
        return list(self._bars.get(symbol, []))

    # ---- trading -----------------------------------------------------

    async def place_order(self, req: OrderRequest) -> OrderResult:
        async with self._lock:
            client_id = req.client_order_id or uuid4().hex
            broker_id = f"P-{uuid4().hex[:10]}"
            quote = await self.quote(req.symbol)
            # Fill at mid + slippage in the trade direction. A LIMIT
            # order fills only if marketable (buy limit >= ask, sell
            # limit <= bid), never through its limit; otherwise it rests
            # until cancelled — like a real book, without partials.
            mid = (quote.bid + quote.ask) / 2 if quote.ask else quote.last
            bps = self.slippage_bps / Decimal("10000")
            buy = req.side == OrderSide.BUY
            fill_price = mid * (Decimal("1") + bps if buy else Decimal("1") - bps)
            if req.order_type == OrderType.LIMIT and req.limit_price is not None:
                touch = (quote.ask or quote.last) if buy else (quote.bid or quote.last)
                marketable = req.limit_price >= touch if buy else req.limit_price <= touch
                if not marketable:
                    resting = OrderResult(
                        client_order_id=client_id,
                        broker_order_id=broker_id,
                        status=OrderStatus.SUBMITTED,
                        filled_qty=Decimal("0"),
                        avg_fill_price=Decimal("0"),
                        submitted_at=datetime.now(UTC),
                    )
                    self._orders[client_id] = resting
                    self._resting[broker_id] = (req, resting)
                    return resting
                fill_price = min(fill_price, req.limit_price) if buy else max(
                    fill_price, req.limit_price
                )
            qty = req.qty if req.side == OrderSide.BUY else -req.qty
            self._apply_fill(req.symbol, qty, fill_price)
            result = OrderResult(
                client_order_id=client_id,
                broker_order_id=broker_id,
                status=OrderStatus.FILLED,
                filled_qty=req.qty,
                avg_fill_price=fill_price,
                submitted_at=datetime.now(UTC),
            )
            self._orders[client_id] = result
            return result

    async def cancel_order(self, broker_order_id: str) -> None:
        entry = self._resting.pop(broker_order_id, None)
        if entry is not None:
            entry[1].status = OrderStatus.CANCELED

    async def cancel_all_orders(self) -> None:
        for broker_id in list(self._resting):
            await self.cancel_order(broker_id)

    async def wait_for_fill(self, order: OrderResult, timeout_s: float) -> OrderResult:
        return self._orders.get(order.client_order_id, order)

    async def order_status(self, client_order_id: str) -> OrderResult | None:
        return self._orders.get(client_order_id)

    async def short_availability(self, symbol: str) -> ShortInfo | None:
        # Paper default: easy to borrow, no restriction. Tests override
        # per symbol with set_short_info().
        return self._short_info.get(
            symbol,
            ShortInfo(
                symbol=symbol,
                shortable_shares=Decimal("1000000"),
                shortable_level=Decimal("3"),
                ssr_active=False,
            ),
        )

    def set_short_info(self, info: ShortInfo | None, symbol: str | None = None) -> None:
        """Test helper: pass `info=None, symbol=...` to simulate unknown data."""
        self._short_info[symbol or info.symbol] = info  # type: ignore[union-attr]

    async def open_orders(self) -> list[OpenOrderView]:
        return [
            OpenOrderView(
                client_order_id=res.client_order_id,
                broker_order_id=broker_id,
                symbol=req.symbol,
                remaining_qty=req.qty if req.side == OrderSide.BUY else -req.qty,
            )
            for broker_id, (req, res) in self._resting.items()
        ]

    def _apply_fill(self, symbol: str, signed_qty: Decimal, price: Decimal) -> None:
        cost = signed_qty * price
        self.cash -= cost
        prev_qty, prev_cost = self._positions.get(
            symbol, (Decimal("0"), Decimal("0"))
        )
        new_qty = prev_qty + signed_qty
        if new_qty == 0:
            self._positions.pop(symbol, None)
            return
        # Same-side scaling: weighted average cost. Sign flips reset
        # the cost basis to the new fill price.
        if prev_qty == 0 or (prev_qty > 0) != (new_qty > 0):
            new_cost = price
        else:
            total_qty = abs(prev_qty) + abs(signed_qty)
            new_cost = (
                abs(prev_qty) * prev_cost + abs(signed_qty) * price
            ) / total_qty
        self._positions[symbol] = (new_qty, new_cost)

    # ---- test helpers ------------------------------------------------

    def set_quote(
        self, symbol: str, bid: Decimal, ask: Decimal, last: Decimal | None = None
    ) -> None:
        self._quotes[symbol] = Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=last if last is not None else (bid + ask) / 2,
        )

    def seed_bars(self, symbol: str, bars: list[Bar]) -> None:
        self._bars[symbol] = list(bars)
