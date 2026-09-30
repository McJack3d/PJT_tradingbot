"""Beta-neutral hedging (audit defect #11).

Before: legs were balanced in dollars, so long high-beta vs short
low-beta was still net long the market; the sector trim could also
round a quantity UP past its cap.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from src.ibkr_sentiment.bot import build_default_bot
from src.ibkr_sentiment.broker.base import Bar
from src.ibkr_sentiment.broker.paper import PaperBroker
from src.ibkr_sentiment.config import RiskOverlayConfig
from src.ibkr_sentiment.risk.overlay import RiskOverlay
from src.ibkr_sentiment.sentiment.models import NewsItem
from src.ibkr_sentiment.signal_engine.beta import estimate_beta
from src.ibkr_sentiment.signal_engine.dollar_neutral import (
    TargetPosition,
    beta_exposure,
    build_dollar_neutral_basket,
)
from src.ibkr_sentiment.signal_engine.mapping import Side, SymbolDecision
from tests.unit.test_ibkr_sentiment_e2e import _make_cfg

D = Decimal
START = date(2026, 5, 1)


def _returns(n: int) -> list[float]:
    # Deterministic, non-degenerate market returns.
    return [0.01 * math.sin(i * 1.7) + 0.004 * math.cos(i * 0.9) for i in range(n)]


def _series(sym: str, rets: list[float], start: float = 100.0, skip=()) -> list[Bar]:
    out, px = [], start
    days = [START + timedelta(days=i) for i in range(len(rets) + 1)]
    for i, d in enumerate(days):
        if i > 0:
            px *= 1 + rets[i - 1]
        if i in skip:
            continue
        c = D(str(round(px, 6)))
        out.append(Bar(sym, d, c, c, c, c, D("1")))
    return out


# ---- estimator ------------------------------------------------------


def test_beta_of_levered_asset_is_blume_adjusted():
    m = _returns(80)
    bench = _series("SPY", m)
    assert estimate_beta(_series("X", m), bench) == D("1.0")
    two_x = estimate_beta(_series("X", [2 * r for r in m]), bench)
    assert two_x == D("1.67")  # 0.67 * 2 + 0.33
    raw = estimate_beta(_series("X", [2 * r for r in m]), bench, blume=False)
    assert raw == D("2.0")


def test_beta_aligns_on_common_days():
    m = _returns(80)
    beta = estimate_beta(_series("X", m, skip={10, 11, 40}), _series("SPY", m))
    assert beta == D("1.0")


def test_beta_needs_twenty_observations_and_is_clamped():
    m = _returns(15)
    assert estimate_beta(_series("X", m), _series("SPY", m)) is None
    m = _returns(80)
    huge = estimate_beta(_series("X", [10 * r for r in m]), _series("SPY", m))
    assert huge == D("3.0")


# ---- basket ---------------------------------------------------------


def _dec(sym: str, side: Side, price: str = "100") -> SymbolDecision:
    return SymbolDecision(sym, side, 0.9 if side == Side.LONG else -0.9, 1.0, D(price), "ok")


def _basket(betas=None, **kw):
    return build_dollar_neutral_basket(
        [_dec("NVDA", Side.LONG), _dec("XOM", Side.SHORT)],
        nlv=D("100000"),
        max_gross_pct=D("1.0"),
        max_position_pct=D("0.25"),
        betas=betas,
        **kw,
    )


def test_dollar_mode_is_unchanged_without_betas():
    targets = {t.symbol: t for t in _basket()}
    assert targets["NVDA"].notional == targets["XOM"].notional == D("25000")


def test_beta_mode_shrinks_the_high_beta_leg():
    betas = {"NVDA": D("1.8"), "XOM": D("0.6")}
    targets = {t.symbol: t for t in _basket(betas)}
    assert targets["XOM"].notional == D("25000")
    # 25000 * 0.6 / 1.8 = 8333.33 -> floored to 83 shares
    assert targets["NVDA"].target_qty == D("83")
    assert abs(beta_exposure(targets.values(), betas)) < D("100")
    assert "beta-balance" in targets["NVDA"].reason


def test_beta_mode_leaves_one_sided_book_alone():
    targets = build_dollar_neutral_basket(
        [_dec("NVDA", Side.LONG)],
        nlv=D("100000"),
        max_gross_pct=D("1.0"),
        max_position_pct=D("0.05"),
        betas={"NVDA": D("1.8")},
    )
    assert targets[0].target_qty == D("50")


def test_sector_trim_never_rounds_up_past_cap():
    """Regression: quantize(Decimal("1")) rounded 6.5 shares up to 7."""
    targets = build_dollar_neutral_basket(
        [_dec("A", Side.LONG, "1000"), _dec("B", Side.LONG, "1000")],
        nlv=D("100000"),
        max_gross_pct=D("2.0"),
        max_position_pct=D("0.07"),
        sector_of={"A": "XLK", "B": "XLK"},
        max_sector_pct=D("0.13"),
    )
    assert all(t.target_qty == D("6") for t in targets)
    assert sum(t.notional for t in targets) <= D("13000")


# ---- overlay --------------------------------------------------------


def _t(sym: str, qty: str) -> TargetPosition:
    q = D(qty)
    return TargetPosition(sym, Side.LONG if q > 0 else Side.SHORT, q, abs(q) * 100, "")


def test_net_cap_uses_beta_weighted_exposure_when_given_betas():
    o = RiskOverlay(
        cfg=RiskOverlayConfig(max_net_exposure_pct=D("0.02"), max_position_pct=D("0.5")),
        starting_equity=D("100000"),
    )
    deltas = [_t("XOM", "100"), _t("NVDA", "-50")]  # $5k dollar-net long
    prices = {"XOM": D("100"), "NVDA": D("100")}
    kw = dict(nlv=D("100000"), current_positions={}, prices=prices)
    dollar = o.check_basket(deltas, **kw)
    assert not all(v.ok for _, v in dollar)
    beta = o.check_basket(deltas, betas={"XOM": D("0.5"), "NVDA": D("1.0")}, **kw)
    assert all(v.ok for _, v in beta)  # beta-weighted net = 0


# ---- bot ------------------------------------------------------------


@pytest.mark.asyncio
async def test_bot_builds_beta_balanced_basket(tmp_path: Path):
    cfg = _make_cfg()
    cfg.db_url = f"sqlite+aiosqlite:///{tmp_path}/b.db"
    cfg.risk.starting_equity_usd = D("100000")  # fine share granularity
    cfg.risk.max_net_exposure_pct = D("0.02")
    broker = PaperBroker(starting_cash=D("100000"))
    await broker.connect()
    m = _returns(80)
    # Uptrend + wiggle so the long's SMA/RSI technicals pass, and a
    # downtrend for the short.
    up = [r + 0.004 for r in m]
    broker.seed_bars("SPY", _series("SPY", up))
    aapl = _series("AAPL", [2 * r for r in up])
    msft = _series("MSFT", [r - 0.012 for r in up])
    broker.seed_bars("AAPL", aapl)
    broker.seed_bars("MSFT", msft)
    for sym, bars in (("AAPL", aapl), ("MSFT", msft)):
        px = bars[-1].close
        broker.set_quote(sym, bid=px, ask=px)

    bot = build_default_bot(cfg, broker, db_url=cfg.db_url)
    await bot.start()
    try:
        now = datetime.now(UTC)
        await bot.submit_item(
            NewsItem(
                title="AAPL beats record growth surge approval",
                body="surge growth beats record",
                symbols=("AAPL",),
                published_at=now,
            )
        )
        await bot.submit_item(
            NewsItem(
                title="MSFT misses lawsuit weak loss downgrade",
                body="loss lawsuit weak downgrade",
                symbols=("MSFT",),
                published_at=now,
            )
        )
        report = await bot.tick()
        sides = {t.symbol: t.target_qty for t in report.targets}
        assert sides["AAPL"] > 0 > sides["MSFT"]
        assert "beta-balance" in next(t.reason for t in report.targets if t.symbol == "AAPL")
        placed = {d.symbol for d, _ in report.execution.placed}
        assert placed == {"AAPL", "MSFT"}  # beta-net within a 2% cap
        betas = {"AAPL": D("1.67"), "MSFT": D("1.0")}
        assert abs(beta_exposure(report.targets, betas)) <= D("2000")
    finally:
        await bot.stop()
