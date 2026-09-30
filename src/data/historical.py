"""Historical data downloader and loader wrapper.

Exposes custom DataFrame-based loaders and historical downloader classes
expected by the cash-and-carry (funding arb) backtester and scripts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.adapters.binance import BinanceAdapter
from src.data.history import (
    _cache_path,
    _download_funding,
    _download_ohlcv,
    _ohlcv_to_df,
)


def load_funding(symbol: str, data_dir: str = "data/history") -> pd.DataFrame:
    """Load funding parquet partitions into a single DataFrame as expected by backtest engine."""
    folder = Path(data_dir) / "funding" / symbol.replace("/", "_")
    if not folder.exists():
        # Fallback: check if there's a single parquet file in data_dir (regime-switch style)
        safe = symbol.replace("/", "").replace(":", "")
        fallback_path = Path(data_dir) / f"funding_{safe}_8h.parquet"
        if fallback_path.exists():
            try:
                df = pd.read_parquet(fallback_path)
                if isinstance(df, pd.Series):
                    df = df.to_frame("funding_rate")
                df = df.reset_index()
                # Find the timestamp column
                ts_col = None
                for col in df.columns:
                    if col != "funding_rate":
                        ts_col = col
                        break
                if ts_col:
                    df = df.rename(columns={ts_col: "ts"})
                else:
                    df["ts"] = df.index
                df["symbol"] = symbol
                if "mark_price" not in df.columns:
                    df["mark_price"] = 0.0
                return df[["ts", "symbol", "funding_rate", "mark_price"]]
            except Exception:
                pass
        return pd.DataFrame(columns=["ts", "symbol", "funding_rate", "mark_price"])

    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["ts", "symbol", "funding_rate", "mark_price"])

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Normalize column names / index
            if "ts" not in df.columns:
                df = df.reset_index().rename(columns={"index": "ts"})
            if "symbol" not in df.columns:
                df["symbol"] = symbol
            if "mark_price" not in df.columns:
                df["mark_price"] = 0.0
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["ts", "symbol", "funding_rate", "mark_price"])
    return pd.concat(frames, ignore_index=True)


def load_ohlcv(symbol: str, timeframe: str = "1h", data_dir: str = "data/history") -> pd.DataFrame:
    """Load cached OHLCV data into a DataFrame."""
    safe = symbol.replace("/", "").replace(":", "")
    path = Path(data_dir) / f"ohlcv_{safe}_{timeframe}.parquet"
    if path.exists():
        try:
            df = pd.read_parquet(path)
            if "ts" not in df.columns:
                df = df.reset_index().rename(columns={"index": "ts"})
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])


class HistoricalDownloader:
    def __init__(self, exchange: BinanceAdapter, data_dir: str = "data/history"):
        self.exchange = exchange
        self.data_dir = data_dir
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    async def download_funding(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.Series:
        since_ms = int(start.timestamp() * 1000)
        until_ms = int(end.timestamp() * 1000)
        rows = await _download_funding(symbol, since_ms, until_ms)
        if not rows:
            return pd.Series(dtype=float)
        idx = pd.to_datetime([ts for ts, _ in rows], unit="ms", utc=True)
        s = pd.Series([r for _, r in rows], index=idx, name="funding_rate").sort_index()
        path = _cache_path(self.data_dir, symbol, "8h", "funding")
        pd.DataFrame({"funding_rate": s}).to_parquet(path)
        return s

    async def download_ohlcv(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1h"
    ) -> pd.DataFrame:
        since_ms = int(start.timestamp() * 1000)
        until_ms = int(end.timestamp() * 1000)
        rows = await _download_ohlcv(symbol, timeframe, since_ms, until_ms)
        df = _ohlcv_to_df(rows)
        path = _cache_path(self.data_dir, symbol, timeframe, "ohlcv")
        if not df.empty:
            df.to_parquet(path)
        return df
