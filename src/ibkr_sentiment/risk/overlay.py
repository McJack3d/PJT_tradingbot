"""Risk overlay for the IBKR sentiment bot.

Two checks the rest of the bot defers to:

  * `RiskOverlay.check_basket()` — pre-trade gate. Walks the proposed
    order deltas against the CURRENT book and vetoes the ones that
    would breach the per-name, gross-exposure, or net-exposure caps.
    Orders that only shrink an existing position are always approved:
    a cap breach must never block the trade that reduces risk.

  * `RiskOverlay.check_account()` — continuous gate. Compare current
    NLV against the starting equity and the daily anchor (NLV at the
    start of the New York trading day) and trip the cumulative or
    daily loss stop if breached. A trip LATCHES: the overlay stays
    halted until an operator calls `rearm()`.

The overlay never sends orders itself — it returns a verdict and a
human-readable reason; the execution engine is responsible for acting
on it. That separation is what makes the risk module straightforward
to test in isolation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.ibkr_sentiment.broker.base import AccountSummary, PositionView
from src.ibkr_sentiment.config import RiskOverlayConfig
from src.ibkr_sentiment.signal_engine.dollar_neutral import TargetPosition

NEW_YORK = ZoneInfo("America/New_York")


def trading_day(ts: datetime) -> date:
    """The US-equities trading day a timestamp belongs to."""
    return ts.astimezone(NEW_YORK).date()


@dataclass(slots=True)
class RiskVerdict:
    ok: bool
    reason: str


def _is_risk_reducing(current: Decimal, delta: Decimal) -> bool:
    """True when `delta` moves the position toward zero without
    flipping it through to the other side."""
    if current == 0 or delta == 0:
        return False
    new = current + delta
    return abs(new) < abs(current) and (new == 0 or (new > 0) == (current > 0))


@dataclass
class RiskOverlay:
    cfg: RiskOverlayConfig
    starting_equity: Decimal
    daily_anchor: Decimal | None = None  # NLV at the start of the trading day
    anchor_day: date | None = None
    halted_reason: str | None = field(default=None)
    _halt_is_daily: bool = field(default=False, repr=False)

    @property
    def halted(self) -> bool:
        return self.halted_reason is not None

    def rearm(self) -> None:
        """Operator action: clear a latched halt."""
        self.halted_reason = None
        self._halt_is_daily = False

    def roll_daily_anchor(self, nlv: Decimal, now: datetime) -> None:
        """Set the daily anchor to `nlv` the first time we see a new
        trading day. Callers restoring state after a restart should set
        `daily_anchor` / `anchor_day` directly from persisted equity."""
        day = trading_day(now)
        if self.anchor_day != day or self.daily_anchor is None:
            self.anchor_day = day
            self.daily_anchor = nlv
            if self._halt_is_daily:
                self.rearm()

    def check_account(
        self, account: AccountSummary, *, daily_pnl: Decimal | None = None
    ) -> RiskVerdict:
        """Trip (and latch) if daily or cumulative drawdown limits are
        breached. `daily_pnl` defaults to NLV minus the daily anchor."""
        if self.halted_reason is not None:
            return RiskVerdict(False, f"halted: {self.halted_reason}")
        verdict = self._evaluate_account(account, daily_pnl)
        if not verdict.ok:
            self.halted_reason = verdict.reason
            self._halt_is_daily = verdict.reason.startswith("daily")
        return verdict

    def _evaluate_account(
        self, account: AccountSummary, daily_pnl: Decimal | None
    ) -> RiskVerdict:
        nlv = account.net_liquidation
        cum_loss = self.starting_equity - nlv
        cum_pct = (
            cum_loss / self.starting_equity if self.starting_equity > 0 else Decimal("0")
        )
        if cum_pct >= self.cfg.cumulative_loss_stop_pct:
            return RiskVerdict(
                False,
                f"cumulative drawdown {float(cum_pct):.2%} >= cap "
                f"{float(self.cfg.cumulative_loss_stop_pct):.2%}",
            )
        if daily_pnl is None and self.daily_anchor is not None:
            daily_pnl = nlv - self.daily_anchor
        base = self.daily_anchor if self.daily_anchor else self.starting_equity
        if daily_pnl is not None and base > 0:
            daily_pct = (-daily_pnl) / base
            if daily_pct >= self.cfg.daily_loss_stop_pct:
                return RiskVerdict(
                    False,
                    f"daily drawdown {float(daily_pct):.2%} >= cap "
                    f"{float(self.cfg.daily_loss_stop_pct):.2%}",
                )
        return RiskVerdict(True, "account within risk limits")

    def check_basket(
        self,
        deltas: Iterable[TargetPosition],
        *,
        nlv: Decimal,
        current_positions: dict[str, Decimal],
        prices: dict[str, Decimal],
    ) -> list[tuple[TargetPosition, RiskVerdict]]:
        """Approve or veto each order delta against the current book.

        * Risk-reducing deltas are always approved.
        * A risk-increasing delta is vetoed if the resulting position
          would exceed the per-name cap, or if it has no usable price.
        * If the post-trade book breaches the gross or net cap, the
          largest risk-increasing delta driving the breach is vetoed,
          repeatedly, until the book fits (or only risk-reducing
          deltas remain). One breach never vetoes the whole basket.

        Returns one (delta, verdict) pair per input delta, in order.
        """
        deltas = list(deltas)
        verdicts: dict[int, RiskVerdict] = {}
        if nlv <= 0:
            return [
                (d, RiskVerdict(True, "risk-reducing"))
                if _is_risk_reducing(current_positions.get(d.symbol, Decimal("0")), d.target_qty)
                else (d, RiskVerdict(False, "non-positive NLV"))
                for d in deltas
            ]
        per_name_cap = nlv * self.cfg.max_position_pct
        gross_cap = nlv * self.cfg.max_gross_exposure_pct
        net_cap = nlv * self.cfg.max_net_exposure_pct

        increasing: list[int] = []
        for i, d in enumerate(deltas):
            cur = current_positions.get(d.symbol, Decimal("0"))
            if _is_risk_reducing(cur, d.target_qty):
                verdicts[i] = RiskVerdict(True, "risk-reducing")
                continue
            price = prices.get(d.symbol, Decimal("0"))
            if price <= 0:
                verdicts[i] = RiskVerdict(False, f"{d.symbol} has no price")
                continue
            new_notional = abs((cur + d.target_qty) * price)
            if new_notional > per_name_cap:
                verdicts[i] = RiskVerdict(
                    False,
                    f"{d.symbol} notional {new_notional} > per-name cap {per_name_cap}",
                )
                continue
            increasing.append(i)

        def book(accepted: list[int]) -> dict[str, Decimal]:
            qty = dict(current_positions)
            for i, d in enumerate(deltas):
                if (verdicts.get(i) is not None and verdicts[i].ok) or i in accepted:
                    qty[d.symbol] = qty.get(d.symbol, Decimal("0")) + d.target_qty
            return qty

        while True:
            qty = book(increasing)
            # Symbols without a price contribute nothing; they can only
            # appear here via current positions we are not trading.
            gross = sum(
                (abs(q * prices.get(s, Decimal("0"))) for s, q in qty.items()),
                Decimal("0"),
            )
            net = sum(
                (q * prices.get(s, Decimal("0")) for s, q in qty.items()), Decimal("0")
            )
            # Deltas pushing net further from zero are removed first, so
            # trimming for either cap keeps the book balanced.
            sign = 1 if net > 0 else -1
            same_side = [i for i in increasing if deltas[i].target_qty * sign > 0]
            if gross > gross_cap:
                breach = f"proposed gross {gross} > gross cap {gross_cap}"
                candidates = same_side if net != 0 and same_side else increasing
            elif abs(net) > net_cap:
                breach = f"proposed net {net} > net cap ±{net_cap}"
                candidates = same_side
            else:
                break
            if not candidates:
                break
            worst = max(
                candidates,
                key=lambda i: abs(deltas[i].target_qty * prices[deltas[i].symbol]),
            )
            verdicts[worst] = RiskVerdict(False, breach)
            increasing.remove(worst)

        for i in increasing:
            verdicts[i] = RiskVerdict(True, "target within risk limits")
        return [(d, verdicts[i]) for i, d in enumerate(deltas)]

    def reconcile_positions(
        self,
        positions: list[PositionView],
        targets: list[TargetPosition],
    ) -> list[RiskVerdict]:
        """Sanity-check that no existing position is itself over-cap.

        Returns one verdict per offending position; an empty list means
        everything is within limits.
        """
        bad: list[RiskVerdict] = []
        target_by_sym = {t.symbol for t in targets}
        for p in positions:
            if p.symbol in target_by_sym:
                continue
            notional = abs(p.qty * p.mark_price)
            cap = self.starting_equity * self.cfg.max_position_pct
            if notional > cap:
                bad.append(
                    RiskVerdict(
                        False,
                        f"orphan position {p.symbol} notional {notional} > cap {cap}",
                    )
                )
        return bad
