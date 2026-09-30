"""Pre-trade checks for short sales.

A sell that takes a position below zero is a short sale and needs
shares to borrow. Before placing one the execution engine asks the
broker for a `ShortInfo` snapshot and runs `evaluate_short()`:

  * borrow availability — IBKR's shortable-shares count must cover the
    order with headroom (availability is shared and moves intraday);
  * borrow difficulty — IBKR's shortable indicator: > 2.5 easy to
    borrow, 1.5-2.5 hard to borrow (locate needed), < 1.5 unavailable;
  * SEC Rule 201 (short-sale restriction, "SSR") — after a 10% drop
    from the prior close, short sales are only allowed above the
    national best bid for the rest of that day and all of the next.
    Our marketable limits sell AT or BELOW the bid, so new shorts are
    blocked while SSR is active.

Unknown data fails closed: no borrow data means no new short.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.ibkr_sentiment.broker.base import ShortInfo
from src.ibkr_sentiment.risk.overlay import RiskVerdict

SSR_DROP = Decimal("0.10")
EASY_TO_BORROW = Decimal("2.5")
BORROWABLE = Decimal("1.5")


@dataclass(slots=True)
class ShortPolicy:
    enabled: bool = True
    allow_hard_to_borrow: bool = False
    # Require this multiple of our order size to be available to borrow.
    min_shortable_multiple: Decimal = Decimal("2")
    block_under_ssr: bool = True


def short_sale_qty(current: Decimal, delta: Decimal) -> Decimal:
    """Shares of `delta` that are sold short, i.e. that take the position
    (further) below zero. Sells that only reduce a long are not shorts."""
    zero = Decimal("0")
    new = current + delta
    return max(zero, max(zero, -new) - max(zero, -current))


def ssr_triggered(
    *,
    prev_close: Decimal | None,
    day_low: Decimal | None,
    last: Decimal | None,
    carried_over: bool = False,
) -> bool:
    """Rule 201: triggered today if the price has traded 10% or more
    below the prior close, or carried over from a trigger yesterday."""
    if carried_over:
        return True
    if prev_close is None or prev_close <= 0:
        return False
    threshold = prev_close * (1 - SSR_DROP)
    return any(p is not None and 0 < p <= threshold for p in (day_low, last))


def evaluate_short(
    info: ShortInfo | None, qty: Decimal, policy: ShortPolicy
) -> RiskVerdict:
    """Verdict for selling `qty` shares short, given `info`."""
    if not policy.enabled or qty <= 0:
        return RiskVerdict(True, "no short sale")
    if info is None:
        return RiskVerdict(False, "short availability unknown")
    if policy.block_under_ssr and info.ssr_active:
        return RiskVerdict(False, f"{info.symbol} under short-sale restriction (Rule 201)")
    if info.shortable_level is not None:
        if info.shortable_level < BORROWABLE:
            return RiskVerdict(False, f"{info.symbol} not available to borrow")
        if info.shortable_level <= EASY_TO_BORROW and not policy.allow_hard_to_borrow:
            return RiskVerdict(False, f"{info.symbol} is hard to borrow")
    if info.shortable_shares is None:
        return RiskVerdict(False, f"{info.symbol} shortable shares unknown")
    need = qty * policy.min_shortable_multiple
    if info.shortable_shares < need:
        return RiskVerdict(
            False,
            f"{info.symbol} shortable shares {info.shortable_shares} < {need} needed",
        )
    return RiskVerdict(True, "short sale allowed")
