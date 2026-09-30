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
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionView,
    Quote,
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
            # Market order: fill at mid + slippage in the trade direction.
            if req.order_type == OrderType.MARKET:
                mid = (quote.bid + quote.ask) / 2 if quote.ask else quote.last
                bps = self.slippage_bps / Decimal("10000")
                if req.side == OrderSide.BUY:
                    fill_price = mid * (Decimal("1") + bps)
                else:
                    fill_price = mid * (Decimal("1") - bps)
            else:
                fill_price = (
                    req.limit_price if req.limit_price is not None else quote.last
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
        # Paper broker fills synchronously, so cancellation is a no-op.
        return None

    async def cancel_all_orders(self) -> None:
        # Paper broker fills synchronously, so nothing is ever resting.
        pass

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
