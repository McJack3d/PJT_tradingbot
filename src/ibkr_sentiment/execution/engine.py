"""Execution engine for the IBKR sentiment bot.

Takes a list of `TargetPosition` deltas (from the dollar-neutral
basket builder) and routes them to the broker. Respects:

  * Risk overlay verdicts — vetoed targets are skipped, not silently
    truncated. Risk-reducing deltas are always allowed through.
  * Account halts — the first time the overlay trips, open orders are
    cancelled and every position is flattened; while the halt is
    latched nothing new is placed.
  * `dry_run` mode — orders are logged, never sent.
  * IBKR pacing — the broker's own rate limiter is what we rely on; the
    engine does not double-count.

Returns a `RunResult` summarising what was placed, what was rejected,
and which positions were closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from uuid import uuid4

from src.ibkr_sentiment.broker.base import (
    AccountSummary,
    Broker,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
)
from src.ibkr_sentiment.risk.overlay import RiskOverlay, RiskVerdict
from src.ibkr_sentiment.signal_engine.dollar_neutral import (
    TargetPosition,
    diff_targets,
)


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


@dataclass
class ExecutionEngine:
    broker: Broker
    overlay: RiskOverlay
    dry_run: bool = False
    order_prefix: str = "ibsent"

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
        approved: list[TargetPosition] = []
        for d, verdict in self.overlay.check_basket(
            deltas,
            nlv=account.net_liquidation,
            current_positions=current_positions,
            prices=prices,
        ):
            if not verdict.ok:
                result.rejected_by_risk.append((d, verdict.reason))
                continue
            approved.append(d)

        for d in approved:
            req = self._to_order_request(d)
            if self.dry_run:
                result.skipped_dry_run.append((d, req))
                continue
            try:
                res = await self.broker.place_order(req)
                result.placed.append((d, res))
            except Exception as e:
                result.errors.append((d.symbol, f"{type(e).__name__}: {e}"))
        return result

    def _to_order_request(self, delta: TargetPosition) -> OrderRequest:
        side = OrderSide.BUY if delta.target_qty > 0 else OrderSide.SELL
        return OrderRequest(
            symbol=delta.symbol,
            side=side,
            qty=abs(delta.target_qty),
            order_type=OrderType.MARKET,
            client_order_id=f"{self.order_prefix}-{uuid4().hex[:10]}",
        )

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
