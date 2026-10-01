"""Small-cap universe construction.

The universe is the set of Binance spot pairs the bot is willing to
trade on a given day. It's rebuilt periodically from 24h ticker stats
so it tracks which low-caps are currently liquid enough to touch.

The filtering is a pure function (`build_universe`) so it's trivially
testable; fetching the raw market data is behind a `MarketProvider`
Protocol so the live ccxt path and a test fake are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.crypto_sentiment.config import CryptoSentimentConfig


@dataclass(slots=True)
class MarketInfo:
    symbol: str            # ccxt unified symbol, e.g. "FOO/USDT"
    base: str              # "FOO"
    quote: str             # "USDT"
    active: bool
    quote_volume_24h: Decimal
    last: Decimal
    bid: Decimal
    ask: Decimal

    @property
    def spread_pct(self) -> Decimal:
        if self.bid <= 0 or self.ask <= 0:
            return Decimal("1")  # treat unknown spread as "wide" → skip
        mid = (self.bid + self.ask) / 2
        return (self.ask - self.bid) / mid


class MarketProvider(Protocol):
    async def fetch_markets(self) -> list[MarketInfo]: ...


def build_universe(
    markets: list[MarketInfo], cfg: CryptoSentimentConfig
) -> list[MarketInfo]:
    """Filter raw markets down to the tradeable small-cap set.

    Keeps active, quote-asset-matching pairs whose 24h quote volume sits
    inside [min, max] and whose base isn't an excluded major/stablecoin.
    Sorted by volume descending (most liquid first) and capped.
    """
    excluded = {b.upper() for b in cfg.exclude_bases}
    quotes = {q.upper() for q in cfg.quote_assets}
    out: list[MarketInfo] = []
    for m in markets:
        if not m.active:
            continue
        if m.quote.upper() not in quotes:
            continue
        if m.base.upper() in excluded:
            continue
        if m.quote_volume_24h < cfg.min_quote_volume_24h:
            continue
        if m.quote_volume_24h > cfg.max_quote_volume_24h:
            continue
        out.append(m)
    out.sort(key=lambda m: m.quote_volume_24h, reverse=True)
    return out[: cfg.max_universe]


class CcxtMarketProvider:
    """Live provider: pulls 24h tickers + market metadata from a ccxt
    spot client (the same one `BinanceAdapter` holds).

    Integration-only — not exercised by the unit suite, which uses a
    fake provider. Kept small and obvious so it's easy to eyeball.
    """

    def __init__(self, spot_client) -> None:  # noqa: ANN001 - ccxt client
        self.spot = spot_client

    async def fetch_markets(self) -> list[MarketInfo]:
        markets = await self.spot.load_markets()
        tickers = await self.spot.fetch_tickers()
        out: list[MarketInfo] = []
        for symbol, t in tickers.items():
            m = markets.get(symbol)
            if m is None or m.get("type") != "spot":
                continue
            info = t.get("info", {}) or {}
            quote_vol = t.get("quoteVolume")
            if quote_vol is None:
                quote_vol = info.get("quoteVolume", 0)
            out.append(
                MarketInfo(
                    symbol=symbol,
                    base=str(m.get("base", "")),
                    quote=str(m.get("quote", "")),
                    active=bool(m.get("active", True)),
                    quote_volume_24h=_dec(quote_vol),
                    last=_dec(t.get("last")),
                    bid=_dec(t.get("bid")),
                    ask=_dec(t.get("ask")),
                )
            )
        return out


def _dec(v) -> Decimal:  # noqa: ANN001
    if v is None:
        return Decimal("0")
    try:
        return Decimal(str(v))
    except (ValueError, ArithmeticError):
        return Decimal("0")
