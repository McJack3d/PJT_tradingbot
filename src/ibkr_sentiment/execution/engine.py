"""Execution engine for the IBKR sentiment bot.

Takes a list of `TargetPosition` deltas (from the dollar-neutral
basket builder) and routes them to the broker. Respects:

  * Risk overlay verdicts — vetoed targets are skipped, not silently
    truncated. Risk-reducing deltas are always allowed through.
  * Account halts — the first time the overlay trips, open orders are
    cancelled and every position is flattened; while the halt is
    latched nothing new is placed.
  * Fill tracking — after placing a batch, waits (concurrently) for
    every order to reach a terminal state; anything still working at
    `fill_timeout_s` has its remainder cancelled. `RunResult.placed`
    carries the FINAL state (filled qty / avg price), never the
    submission snapshot.
  * Stale-order reconciliation — `cancel_stale_orders()` cancels any
    working order carrying our prefix (e.g. left over from a crash)
    so positions are read from a quiet book and nothing is re-sent.
  * Order style — marketable LIMIT orders by default, priced at the
    touch +/- `limit_offset_bps` from a fresh quote so no fill can be
    worse than that; `adaptive` (IBKR Adaptive algo) and `market` are
    opt-in. Emergency flattening always uses market orders.
  * Short-sale checks — any delta that sells below zero needs borrow
    (IBKR shortable shares / level) and no Rule 201 restriction. A
    blocked short is dropped BEFORE the gross/net checks so the rest of
    the basket is re-balanced around it; a blocked long->short flip is
    trimmed to just closing the long.
  * `dry_run` mode — orders are logged, never sent.
  * IBKR pacing — the broker's own rate limiter is what we rely on; the
    engine does not double-count.

Returns a `RunResult` summarising what was placed, what was rejected,
and which positions were closed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from uuid import uuid4

from src.ibkr_sentiment.broker.base import (
    AccountSummary,
    Broker,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.ibkr_sentiment.risk.overlay import RiskOverlay, RiskVerdict
from src.ibkr_sentiment.risk.shorting import ShortPolicy, evaluate_short, short_sale_qty
from src.ibkr_sentiment.signal_engine.dollar_neutral import (
    TargetPosition,
    diff_targets,
)


class NoQuoteError(RuntimeError):
    pass


@dataclass(slots=True)
class PlannedOrder:
    target: TargetPosition
    request: OrderRequest


@dataclass
class RunResult:
    placed: list[tuple[TargetPosition, OrderResult]] = field(default_factory=list)
    rejected_by_risk: list[tuple[TargetPosition, str]] = field(default_factory=list)
    skipped_dry_run: list[tuple[TargetPosition, OrderRequest]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    flattened: list[OrderResult] = field(default_factory=list)
    stale_cancelled: list[str] = field(default_factory=list)  # client order ids
    foreign_open_orders: list[str] = field(default_factory=list)  # broker order ids


@dataclass
class ExecutionEngine:
    broker: Broker
    overlay: RiskOverlay
    dry_run: bool = False
    order_prefix: str = "ibsent"
    fill_timeout_s: float = 30.0
    cancel_confirm_timeout_s: float = 5.0
    order_style: str = "limit"  # limit | adaptive | market
    limit_offset_bps: Decimal = Decimal("10")
    short_policy: ShortPolicy = field(default_factory=ShortPolicy)

    async def enforce_account(
        self, account: AccountSummary, result: RunResult
    ) -> RiskVerdict:
        """Run the account-level gate. On the tick the halt first trips,
        cancel open orders and flatten the book; later ticks only report
        the latched halt so we never stack duplicate flatten orders."""
        was_halted = self.overlay.halted
        verdict = self.overlay.check_account(account)
        if verdict.ok:
            return verdict
        result.errors.append(("account_halt", verdict.reason))
        if not was_halted:
            try:
                result.flattened = await self.emergency_flatten()
            except Exception as e:
                result.errors.append(("flatten_failed", f"{type(e).__name__}: {e}"))
        return verdict

    async def execute_basket(
        self,
        targets: list[TargetPosition],
        *,
        account: AccountSummary,
        current_positions: dict[str, Decimal],
        marks: dict[str, Decimal] | None = None,
        betas: dict[str, Decimal] | None = None,
    ) -> RunResult:
        """`marks` are current market prices for held symbols; prices
        for symbols in `targets` are derived from the targets
        themselves."""
        result = RunResult()
        if not (await self.enforce_account(account, result)).ok:
            return result

        prices = dict(marks or {})
        for t in targets:
            if t.target_qty != 0 and t.notional > 0:
                prices[t.symbol] = t.notional / abs(t.target_qty)
        deltas = diff_targets(current_positions, targets)
        deltas = await self._filter_short_sales(deltas, current_positions, result)
        approved: list[TargetPosition] = []
        for d, verdict in self.overlay.check_basket(
            deltas,
            nlv=account.net_liquidation,
            current_positions=current_positions,
            prices=prices,
            betas=betas,
        ):
            if not verdict.ok:
                result.rejected_by_risk.append((d, verdict.reason))
                continue
            approved.append(d)

        submitted: list[tuple[TargetPosition, OrderResult]] = []
        for d in approved:
            try:
                req = await self._to_order_request(d)
            except NoQuoteError as e:
                result.errors.append((d.symbol, str(e)))
                continue
            if self.dry_run:
                result.skipped_dry_run.append((d, req))
                continue
            try:
                submitted.append((d, await self.broker.place_order(req)))
            except Exception as e:
                result.errors.append((d.symbol, f"{type(e).__name__}: {e}"))

        finals = await asyncio.gather(
            *(self._await_final(d, res, result) for d, res in submitted)
        )
        result.placed.extend(zip((d for d, _ in submitted), finals, strict=True))
        return result

    async def _filter_short_sales(
        self,
        deltas: list[TargetPosition],
        current_positions: dict[str, Decimal],
        result: RunResult,
    ) -> list[TargetPosition]:
        kept: list[TargetPosition] = []
        for d in deltas:
            cur = current_positions.get(d.symbol, Decimal("0"))
            qty = short_sale_qty(cur, d.target_qty)
            if qty <= 0:
                kept.append(d)
                continue
            try:
                info = await self.broker.short_availability(d.symbol)
            except Exception as e:
                result.errors.append((d.symbol, f"short availability {type(e).__name__}: {e}"))
                info = None
            verdict = evaluate_short(info, qty, self.short_policy)
            if verdict.ok:
                kept.append(d)
                continue
            result.rejected_by_risk.append((d, verdict.reason))
            if cur > 0:
                # Still close the long; only the short leg is blocked.
                kept.append(
                    TargetPosition(
                        symbol=d.symbol,
                        side=d.side,
                        target_qty=-cur,
                        notional=Decimal("0"),
                        reason=f"{d.reason}; short leg blocked: {verdict.reason}",
                    )
                )
        return kept

    async def _await_final(
        self, delta: TargetPosition, order: OrderResult, result: RunResult
    ) -> OrderResult:
        """Wait for a terminal state; on timeout cancel the remainder and
        wait briefly for the cancel to confirm."""
        try:
            final = await self.broker.wait_for_fill(order, self.fill_timeout_s)
            if final.status.is_terminal:
                return final
            if final.broker_order_id:
                await self.broker.cancel_order(final.broker_order_id)
            final = await self.broker.wait_for_fill(final, self.cancel_confirm_timeout_s)
            if not final.status.is_terminal:
                # Left for cancel_stale_orders() on the next tick.
                result.errors.append(
                    (delta.symbol, f"cancel unconfirmed for {final.client_order_id}")
                )
            return final
        except Exception as e:
            result.errors.append((delta.symbol, f"fill tracking {type(e).__name__}: {e}"))
            return order

    async def cancel_stale_orders(self, result: RunResult) -> None:
        """Cancel our own working orders before reading positions.

        Normally nothing survives a tick (`_await_final` cancels
        remainders), so anything found here is left over from a crash or
        an unconfirmed cancel. Orders placed outside the bot are
        reported but never touched."""
        if self.dry_run:
            return
        prefix = f"{self.order_prefix}-"
        stale: list[OrderResult] = []
        for o in await self.broker.open_orders():
            if not o.client_order_id.startswith(prefix):
                result.foreign_open_orders.append(o.broker_order_id)
                continue
            try:
                await self.broker.cancel_order(o.broker_order_id)
            except Exception as e:
                result.errors.append((o.symbol, f"stale cancel {type(e).__name__}: {e}"))
                continue
            result.stale_cancelled.append(o.client_order_id)
            stale.append(
                OrderResult(
                    client_order_id=o.client_order_id,
                    broker_order_id=o.broker_order_id,
                    status=OrderStatus.SUBMITTED,
                    filled_qty=Decimal("0"),
                    avg_fill_price=Decimal("0"),
                )
            )
        await asyncio.gather(
            *(self.broker.wait_for_fill(o, self.cancel_confirm_timeout_s) for o in stale)
        )

    async def _to_order_request(self, delta: TargetPosition) -> OrderRequest:
        side = OrderSide.BUY if delta.target_qty > 0 else OrderSide.SELL
        req = OrderRequest(
            symbol=delta.symbol,
            side=side,
            qty=abs(delta.target_qty),
            order_type=OrderType.MARKET,
            client_order_id=f"{self.order_prefix}-{uuid4().hex[:10]}",
        )
        if self.order_style == "adaptive":
            req.algo = "Adaptive"
        elif self.order_style == "limit":
            req.order_type = OrderType.LIMIT
            req.limit_price = await self._limit_price(delta.symbol, side)
        return req

    async def _limit_price(self, symbol: str, side: OrderSide) -> Decimal:
        """Marketable limit: the far touch (ask to buy, bid to sell) moved
        `limit_offset_bps` further, rounded to the tick inward so the
        cap is never exceeded."""
        q = await self.broker.quote(symbol)
        ref = q.ask if side == OrderSide.BUY else q.bid
        if ref <= 0:
            ref = q.last
        if ref <= 0:
            raise NoQuoteError(f"{symbol}: no quote, limit order not sent")
        offset = self.limit_offset_bps / Decimal("10000")
        tick = Decimal("0.01") if ref >= 1 else Decimal("0.0001")
        if side == OrderSide.BUY:
            raw, rounding = ref * (1 + offset), ROUND_FLOOR
        else:
            raw, rounding = ref * (1 - offset), ROUND_CEILING
        return (raw / tick).quantize(Decimal("1"), rounding=rounding) * tick

    async def emergency_flatten(self) -> list[OrderResult]:
        """Cancel open orders, then close all positions ignoring the
        risk overlay. Cancelling first stops a resting order from
        re-opening exposure after the flatten fills."""
        if self.dry_run:
            return []
        await self.broker.cancel_all_orders()
        return await self.broker.flatten_all()


def placed_qty(result: RunResult, symbol: str) -> Decimal:
    """Convenience for tests: sum filled qty for a given symbol."""
    total = Decimal("0")
    for delta, r in result.placed:
        if delta.symbol == symbol:
            total += r.filled_qty
    return total
