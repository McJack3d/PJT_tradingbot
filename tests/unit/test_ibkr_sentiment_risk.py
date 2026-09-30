"""Tests for the IBKR sentiment bot's risk overlay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.ibkr_sentiment.broker.base import AccountSummary
from src.ibkr_sentiment.config import RiskOverlayConfig
from src.ibkr_sentiment.risk.overlay import RiskOverlay
from src.ibkr_sentiment.signal_engine.dollar_neutral import TargetPosition
from src.ibkr_sentiment.signal_engine.mapping import Side


def _overlay(equity: Decimal = Decimal("10000")) -> RiskOverlay:
    return RiskOverlay(
        cfg=RiskOverlayConfig(
            starting_equity_usd=equity,
            max_gross_exposure_pct=Decimal("1.0"),
            max_net_exposure_pct=Decimal("0.2"),
            max_position_pct=Decimal("0.1"),
            daily_loss_stop_pct=Decimal("0.02"),
            cumulative_loss_stop_pct=Decimal("0.1"),
        ),
        starting_equity=equity,
    )


def _acct(nlv: Decimal) -> AccountSummary:
    return AccountSummary(
        net_liquidation=nlv,
        available_funds=nlv,
        gross_position_value=Decimal("0"),
    )


def test_account_check_trips_on_cumulative_drawdown():
    o = _overlay(Decimal("10000"))
    v = o.check_account(_acct(Decimal("8500")))  # 15% drawdown
    assert v.ok is False
    assert "cumulative" in v.reason


def test_account_check_passes_within_limits():
    o = _overlay(Decimal("10000"))
    v = o.check_account(_acct(Decimal("9800")))
    assert v.ok is True


def test_account_check_trips_on_daily_drawdown():
    o = _overlay(Decimal("10000"))
    v = o.check_account(_acct(Decimal("9900")), daily_pnl=Decimal("-250"))
    assert v.ok is False
    assert "daily" in v.reason


def _delta(sym: str, qty: str, price: str = "100") -> TargetPosition:
    q = Decimal(qty)
    return TargetPosition(
        symbol=sym,
        side=Side.LONG if q > 0 else Side.SHORT,
        target_qty=q,
        notional=abs(q) * Decimal(price),
        reason="",
    )


def _check(o, deltas, current=None, prices=None, nlv="10000"):
    if prices is None:
        prices = {d.symbol: Decimal("100") for d in deltas}
    return o.check_basket(
        deltas,
        nlv=Decimal(nlv),
        current_positions=current or {},
        prices=prices,
    )


def test_basket_rejects_oversized_position():
    o = _overlay(Decimal("10000"))
    # 20 * $100 = $2000 = 20% of equity — over the 10% per-name cap.
    [(_, v)] = _check(o, [_delta("AAPL", "20")])
    assert v.ok is False
    assert "per-name" in v.reason


def test_basket_per_name_cap_uses_resulting_position_not_delta():
    o = _overlay(Decimal("10000"))
    # Holding 8 ($800); adding 5 makes $1300 > $1000 cap even though
    # the delta alone ($500) is under it.
    [(_, v)] = _check(o, [_delta("AAPL", "5")], current={"AAPL": Decimal("8")})
    assert v.ok is False
    assert "per-name" in v.reason


def test_basket_trims_gross_breach_without_rejecting_everything():
    o = _overlay(Decimal("10000"))
    # 8 longs + 8 shorts at $900 = $14.4k gross vs $10k cap, net 0.
    deltas = [_delta(f"L{i}", "9") for i in range(8)] + [
        _delta(f"S{i}", "-9") for i in range(8)
    ]
    out = _check(o, deltas)
    approved = [d for d, v in out if v.ok]
    rejected = [v for _, v in out if not v.ok]
    assert approved and rejected
    assert all("gross" in v.reason or "net" in v.reason for v in rejected)
    gross = sum(abs(d.target_qty) * 100 for d in approved)
    net = sum(d.target_qty * 100 for d in approved)
    assert gross <= 10000
    assert abs(net) <= 2000


def test_basket_trims_net_breach_by_dropping_largest_offender():
    o = _overlay(Decimal("10000"))
    # 5 longs * $500 = $2500 net (25% > 20% cap). Dropping one long fixes it.
    deltas = [_delta(f"S{i}", "5") for i in range(4)] + [_delta("BIG", "6")]
    out = dict((d.symbol, v) for d, v in _check(o, deltas))
    assert out["BIG"].ok is False
    assert "net" in out["BIG"].reason
    assert all(out[f"S{i}"].ok for i in range(4))


def test_basket_passes_balanced_dollar_neutral_basket():
    o = _overlay(Decimal("10000"))
    out = _check(o, [_delta("AAPL", "5"), _delta("MSFT", "-5")])
    assert all(v.ok for _, v in out)


def test_basket_always_allows_closing_order_during_net_breach():
    """Regression: one net-cap breach used to reject every delta,
    including the order closing an existing long."""
    o = _overlay(Decimal("10000"))
    deltas = [_delta(f"L{i}", "6") for i in range(5)]  # $3000 new longs
    deltas.append(_delta("AAPL", "-8"))  # close the held long
    out = dict(
        (d.symbol, v)
        for d, v in _check(
            o,
            deltas,
            current={"AAPL": Decimal("8")},
            prices={**{f"L{i}": Decimal("100") for i in range(5)}, "AAPL": Decimal("100")},
        )
    )
    assert out["AAPL"].ok is True
    assert out["AAPL"].reason == "risk-reducing"
    assert not all(v.ok for v in out.values())  # the breach is still trimmed


def test_basket_allows_reducing_an_over_cap_position():
    o = _overlay(Decimal("10000"))
    # Holding $3000 of AAPL (over the $1000 cap); trimming to $2000 must pass.
    [(_, v)] = _check(o, [_delta("AAPL", "-10")], current={"AAPL": Decimal("30")})
    assert v.ok is True


def test_basket_treats_flip_through_zero_as_risk_increasing():
    o = _overlay(Decimal("10000"))
    # Long 5 -> short 25 ($2500 short) breaches per-name cap.
    [(_, v)] = _check(o, [_delta("AAPL", "-30")], current={"AAPL": Decimal("5")})
    assert v.ok is False


def test_basket_rejects_increase_without_price():
    o = _overlay(Decimal("10000"))
    [(_, v)] = _check(o, [_delta("AAPL", "1")], prices={})
    assert v.ok is False
    assert "no price" in v.reason


# ---- daily anchor + latched halt -----------------------------------


def test_daily_stop_fires_from_anchor_without_explicit_pnl():
    """Regression: daily_pnl was never supplied, so the daily stop
    could not fire in production."""
    o = _overlay(Decimal("10000"))
    now = datetime(2026, 9, 30, 14, 0, tzinfo=UTC)
    o.roll_daily_anchor(Decimal("10000"), now)
    v = o.check_account(_acct(Decimal("9750")))  # -2.5% on the day
    assert v.ok is False
    assert "daily" in v.reason


def test_daily_anchor_only_rolls_on_new_trading_day():
    o = _overlay(Decimal("10000"))
    morning = datetime(2026, 9, 30, 14, 0, tzinfo=UTC)  # 10:00 New York
    o.roll_daily_anchor(Decimal("10000"), morning)
    o.roll_daily_anchor(Decimal("9900"), morning + timedelta(hours=5))
    assert o.daily_anchor == Decimal("10000")
    # 01:00 UTC on Oct 1 is still Sep 30 in New York.
    o.roll_daily_anchor(Decimal("9800"), datetime(2026, 10, 1, 1, 0, tzinfo=UTC))
    assert o.daily_anchor == Decimal("10000")
    o.roll_daily_anchor(Decimal("9800"), datetime(2026, 10, 1, 14, 0, tzinfo=UTC))
    assert o.daily_anchor == Decimal("9800")


def test_daily_halt_latches_until_next_trading_day():
    o = _overlay(Decimal("10000"))
    day1 = datetime(2026, 9, 30, 14, 0, tzinfo=UTC)
    o.roll_daily_anchor(Decimal("10000"), day1)
    assert o.check_account(_acct(Decimal("9750"))).ok is False
    # Recovering intraday does not clear the halt.
    assert o.check_account(_acct(Decimal("10000"))).ok is False
    o.roll_daily_anchor(Decimal("9750"), day1 + timedelta(days=1))
    assert o.halted is False
    assert o.check_account(_acct(Decimal("9750"))).ok is True


def test_cumulative_halt_survives_day_roll_until_rearm():
    o = _overlay(Decimal("10000"))
    day1 = datetime(2026, 9, 30, 14, 0, tzinfo=UTC)
    o.roll_daily_anchor(Decimal("8900"), day1)
    assert "cumulative" in o.check_account(_acct(Decimal("8900"))).reason
    o.roll_daily_anchor(Decimal("9500"), day1 + timedelta(days=1))
    assert o.check_account(_acct(Decimal("9500"))).ok is False
    o.rearm()
    assert o.check_account(_acct(Decimal("9500"))).ok is True
