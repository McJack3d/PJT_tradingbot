"""Daily-bar cache (audit defect #10).

Before: every tick requested 120 days of daily bars per signalled
symbol, burning IBKR's historical-data pacing for data that changes
once a day.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from src.ibkr_sentiment.bar_cache import BarCache
from src.ibkr_sentiment.bot import build_default_bot
from src.ibkr_sentiment.broker.base import Bar
from src.ibkr_sentiment.broker.paper import PaperBroker
from src.ibkr_sentiment.sentiment.models import NewsItem
from tests.unit.test_ibkr_sentiment_e2e import _bars, _make_cfg

T0 = datetime(2026, 9, 30, 15, 0, tzinfo=UTC)  # Wed 11:00 New York


def _bar(d: date, close: str) -> Bar:
    c = Decimal(close)
    return Bar("AAPL", d, c, c, c, c, Decimal("1"))


class _Broker(PaperBroker):
    def __init__(self):
        super().__init__()
        self.calls: list[str] = []
        self.responses: dict[str, list[Bar] | Exception] = {}

    async def historical_bars(self, symbol, duration="60 D", bar_size="1 day"):
        self.calls.append(duration)
        r = self.responses[duration]
        if isinstance(r, Exception):
            raise r
        return list(r)


def _history() -> list[Bar]:
    days = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30)]
    return [_bar(d, c) for d, c in zip(days, ("100", "101", "102"), strict=True)]


@pytest.mark.asyncio
async def test_serves_from_cache_within_refresh_interval():
    b = _Broker()
    b.responses["120 D"] = _history()
    cache = BarCache(b)
    first = await cache.get("AAPL", T0)
    again = await cache.get("AAPL", T0 + timedelta(minutes=59))
    assert again == first
    assert b.calls == ["120 D"]


@pytest.mark.asyncio
async def test_refresh_is_incremental_and_updates_todays_partial_bar():
    b = _Broker()
    b.responses["120 D"] = _history()
    b.responses["2 D"] = [_bar(date(2026, 9, 29), "101"), _bar(date(2026, 9, 30), "105")]
    cache = BarCache(b)
    await cache.get("AAPL", T0)
    bars = await cache.get("AAPL", T0 + timedelta(minutes=61))
    assert b.calls == ["120 D", "2 D"]
    assert [str(x.close) for x in bars] == ["100", "101", "105"]


@pytest.mark.asyncio
async def test_new_trading_day_refreshes_and_keeps_window_length():
    b = _Broker()
    b.responses["120 D"] = _history()
    b.responses["2 D"] = [_bar(date(2026, 9, 30), "102"), _bar(date(2026, 10, 1), "103")]
    cache = BarCache(b, refresh=timedelta(days=7))  # only the day change forces it
    await cache.get("AAPL", T0)
    bars = await cache.get("AAPL", T0 + timedelta(days=1))
    assert b.calls == ["120 D", "2 D"]
    assert [x.ts for x in bars] == [date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1)]


@pytest.mark.asyncio
async def test_failed_refresh_serves_stale_bars():
    b = _Broker()
    b.responses["120 D"] = _history()
    b.responses["2 D"] = RuntimeError("pacing violation")
    cache = BarCache(b)
    first = await cache.get("AAPL", T0)
    assert await cache.get("AAPL", T0 + timedelta(hours=2)) == first


@pytest.mark.asyncio
async def test_empty_first_fetch_is_not_cached():
    b = _Broker()
    b.responses["120 D"] = []
    cache = BarCache(b)
    assert await cache.get("AAPL", T0) == []
    b.responses["120 D"] = _history()
    assert len(await cache.get("AAPL", T0)) == 3
    assert b.calls == ["120 D", "120 D"]


@pytest.mark.asyncio
async def test_bot_ticks_reuse_cached_bars(tmp_path: Path):
    cfg = _make_cfg()
    cfg.db_url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    broker = PaperBroker(starting_cash=Decimal("10000"))
    await broker.connect()
    broker.set_quote("AAPL", bid=Decimal("100"), ask=Decimal("100"))
    broker.seed_bars("AAPL", _bars("AAPL", start=80, count=30, step=1.0))
    calls: list[str] = []
    orig = broker.historical_bars

    async def counting(symbol, duration="60 D", bar_size="1 day"):
        calls.append(symbol)
        return await orig(symbol, duration, bar_size)

    broker.historical_bars = counting  # type: ignore[method-assign]
    bot = build_default_bot(cfg, broker, db_url=cfg.db_url)
    await bot.start()
    try:
        for _ in range(3):
            await bot.submit_item(
                NewsItem(
                    title="AAPL beats record growth surge approval",
                    body="surge growth beats record",
                    symbols=("AAPL",),
                    published_at=datetime.now(UTC),
                )
            )
            await bot.tick()
        assert calls == ["AAPL"]  # one fetch across three ticks
    finally:
        await bot.stop()
