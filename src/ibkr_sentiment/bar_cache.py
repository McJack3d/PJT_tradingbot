"""Per-symbol daily-bar cache for the IBKR bot.

The tick used to request 120 days of daily bars for every signalled
symbol on every tick — up to 10 historical requests a minute against
IBKR's pacing limit of ~60 per 10 minutes, for data that changes once a
day. The cache fetches the full history once, then refreshes with a
small incremental request ("2 D": yesterday + today's partial bar) at
most every `refresh`, and always on a new trading day. A failed refresh
serves the last good bars rather than dropping the symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from src.ibkr_sentiment.broker.base import Bar, Broker
from src.ibkr_sentiment.risk.overlay import trading_day
from src.logging_setup import log


def bar_day(ts: datetime | date) -> date:
    if isinstance(ts, datetime):
        return trading_day(ts if ts.tzinfo else ts.replace(tzinfo=UTC))
    return ts


@dataclass
class _Entry:
    bars: list[Bar]
    fetched_at: datetime
    day: date


@dataclass
class BarCache:
    broker: Broker
    full_duration: str = "120 D"
    refresh_duration: str = "2 D"
    refresh: timedelta = timedelta(minutes=60)
    requests: int = 0  # historical requests issued (diagnostics / tests)
    _entries: dict[str, _Entry] = field(default_factory=dict)

    async def get(self, symbol: str, now: datetime) -> list[Bar]:
        entry = self._entries.get(symbol)
        day = trading_day(now)
        if entry is None:
            bars = await self._fetch(symbol, self.full_duration)
            if bars:
                self._entries[symbol] = _Entry(bars, now, day)
            return bars
        if day == entry.day and now - entry.fetched_at < self.refresh:
            return entry.bars
        try:
            fresh = await self._fetch(symbol, self.refresh_duration)
        except Exception as e:
            log.warning("bar_cache.refresh_failed", symbol=symbol, error=str(e))
            return entry.bars
        entry.bars = _merge(entry.bars, fresh)
        entry.fetched_at, entry.day = now, day
        return entry.bars

    async def _fetch(self, symbol: str, duration: str) -> list[Bar]:
        self.requests += 1
        return await self.broker.historical_bars(symbol, duration=duration, bar_size="1 day")

    def invalidate(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._entries.clear()
        else:
            self._entries.pop(symbol, None)


def _merge(old: list[Bar], new: list[Bar]) -> list[Bar]:
    """Replace/append `new` bars by trading day (today's partial bar is
    overwritten by its fresher version), keeping history length."""
    by_day = {bar_day(b.ts): b for b in old}
    for b in new:
        by_day[bar_day(b.ts)] = b
    merged = [by_day[d] for d in sorted(by_day)]
    return merged[-max(len(old), len(new)) :] if old else merged
