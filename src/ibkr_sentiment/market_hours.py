"""NYSE trading-window gate for the IBKR bot.

Wraps `exchange_calendars`' XNYS calendar (holidays, early closes,
special closures) and answers one question: may the bot send new
orders right now?

The window is the regular session minus a buffer after the open (the
opening auction gap and the widest spreads of the day) and before the
close (orders left working into the closing auction). Outside the
window the bot still ingests news and still runs the account-level
risk gate — it just doesn't trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import exchange_calendars as xcals
import pandas as pd


@dataclass(slots=True)
class WindowVerdict:
    open: bool
    reason: str


@dataclass
class MarketCalendar:
    open_buffer: timedelta = timedelta(minutes=5)
    close_buffer: timedelta = timedelta(minutes=10)
    exchange: str = "XNYS"
    _cal: xcals.ExchangeCalendar | None = field(default=None, init=False, repr=False)

    def _calendar(self, ts: pd.Timestamp) -> xcals.ExchangeCalendar:
        # exchange_calendars only builds sessions ~1 year ahead of the
        # day it is instantiated; rebuild when a long-running bot gets
        # close to the edge.
        day = ts.tz_localize(None).normalize()  # calendar sessions are tz-naive
        if self._cal is None or day >= self._cal.last_session - pd.Timedelta(days=7):
            self._cal = xcals.get_calendar(self.exchange, end=day + pd.Timedelta(days=366))
        return self._cal

    def window(self, now: datetime) -> WindowVerdict:
        ts = pd.Timestamp(now)
        if ts.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        ts = ts.tz_convert("UTC")
        cal = self._calendar(ts)
        day = ts.tz_convert(cal.tz).normalize().tz_localize(None)
        if not cal.is_session(day):
            return WindowVerdict(False, f"market closed ({day.date()} is not a trading day)")
        open_ = cal.session_open(day)
        close = cal.session_close(day)
        start = open_ + pd.Timedelta(self.open_buffer)
        end = close - pd.Timedelta(self.close_buffer)
        if ts < start:
            return WindowVerdict(False, f"before trading window (opens {start.isoformat()})")
        if ts >= end:
            return WindowVerdict(False, f"after trading window (closed {end.isoformat()})")
        return WindowVerdict(True, "within trading window")
