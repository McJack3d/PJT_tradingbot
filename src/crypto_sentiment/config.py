"""Configuration for the small-cap crypto sentiment bot.

Money quantities are `Decimal`; sentiment scores are plain floats in
[-1, +1] (that's what the funnel produces). Defaults are deliberately
conservative and paper-oriented — they exist to be overridden from
YAML/env once you've seen the bot behave.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal


@dataclass
class CryptoSentimentConfig:
    # ---- mode ---------------------------------------------------------
    # "paper" routes every order through a FakeExchange (no real money).
    # "live" is a deliberate opt-in handled by the runner, never here.
    mode: str = "paper"
    quote_assets: tuple[str, ...] = ("USDT", "USDC")

    # ---- universe: what counts as a tradeable "small cap" -------------
    # Defined by 24h quote-volume: liquid enough to enter/exit without
    # catastrophic slippage, illiquid enough that sentiment *might* not
    # be fully priced in. Both bounds are in quote units (USDT/USDC).
    min_quote_volume_24h: Decimal = Decimal("250000")
    max_quote_volume_24h: Decimal = Decimal("50000000")
    # Majors and stablecoins are excluded — sentiment there is priced-in
    # and they aren't "small caps" by any definition.
    exclude_bases: tuple[str, ...] = (
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX", "TON",
        "AVAX", "DOT", "LINK", "MATIC", "LTC", "BCH",
        "USDT", "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "WBTC", "WETH", "WBETH",
    )
    max_universe: int = 60
    universe_refresh_minutes: int = 60

    # ---- sentiment decision thresholds --------------------------------
    entry_score: float = 0.45       # composite sentiment >= this → long candidate
    exit_score: float = 0.10        # held name whose score falls to/below this → exit
    min_conviction: float = 0.60    # LLM stated confidence floor
    # Day trading: only act on near-term news. Long-horizon theses are
    # not what a sentiment day-trader should be chasing.
    allowed_horizons: tuple[str, ...] = ("intraday", "short_term")

    # ---- execution / sizing -------------------------------------------
    per_position_usd: Decimal = Decimal("10")     # notional per name
    max_concurrent_positions: int = 2
    max_spread_pct: Decimal = Decimal("0.015")    # skip entry if spread > 1.5%
    dust_usd: Decimal = Decimal("1")              # holdings below this = "flat"
    qty_step: Decimal = Decimal("0.0001")         # fallback lot step if unknown

    # ---- per-position risk --------------------------------------------
    take_profit_pct: Decimal = Decimal("0.05")    # +5% → take profit
    stop_loss_pct: Decimal = Decimal("0.03")      # -3% → cut
    max_hold_hours: float = 12.0                  # time-stop for "intraday"

    # ---- account risk -------------------------------------------------
    daily_loss_stop_usd: Decimal = Decimal("5")   # halt new entries for the day
    asset_cooloff_minutes: int = 60               # no re-entry right after an exit

    # ---- loop / feeds / llm -------------------------------------------
    poll_interval_s: int = 300
    rss_feeds: tuple[str, ...] = (
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
    )
    cryptopanic_enabled: bool = False             # needs a free auth token
    llm_provider: str = "stub"                    # "stub" | "anthropic" | "openai"
    signal_window: timedelta = timedelta(hours=4)
