"""Typed configuration for the IBKR sentiment bot.

Loaded from YAML + .env, validated with Pydantic. Mirrors the layout
of `src.config` (the Binance bot) so operators don't have to learn a
second config style.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from src.ibkr_sentiment.risk.shorting import ShortPolicy


class IbkrMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"  # full pipeline, paper broker, no IB connection
    DRY_RUN = "dry_run"  # full IB connection, intercept orders
    LIVE = "live"


class UniverseEntry(BaseModel):
    """One tradable instrument plus its sector ETF hedge candidate."""

    symbol: str  # e.g. "AAPL"
    exchange: str = "SMART"
    currency: str = "USD"
    sector_etf: str | None = None  # e.g. "XLK"; used by the dollar-neutral overlay
    min_qty: Decimal = Decimal("1")
    tick_size: Decimal = Decimal("0.01")


class FinBertConfig(BaseModel):
    """Stage 1 — discriminative screen."""

    model_name: str = "ProsusAI/finbert"
    device: str = "cpu"  # "cpu" / "cuda" / "mps"
    polarity_threshold: float = 0.55  # |score| above this passes to Stage 2
    confidence_threshold: float = 0.70  # softmax max prob required
    max_input_chars: int = 2_000
    batch_size: int = 16


class LLMConfig(BaseModel):
    """Stage 2 — generative gatekeeper.

    `provider` selects which client is wired in. The pipeline runs
    fine with `provider="stub"` (no network, deterministic) for tests
    and for paper-mode dry runs where you only care about the plumbing.
    """

    provider: str = "stub"  # "stub" | "anthropic" | "openai" | "fingpt"
    model: str = "claude-opus-4-7"
    temperature: float = 0.0
    max_tokens: int = 800
    max_concurrent: int = 4
    request_timeout_s: float = 30.0
    # Conviction floor: LLM verdicts below this are discarded.
    min_conviction: float = 0.55
    # Sources whose historic accuracy is unknown get this prior weight.
    default_source_weight: float = 0.5


class IngestionConfig(BaseModel):
    """Stage 0 — raw text feeds."""

    rss_feeds: list[str] = Field(default_factory=list)
    sec_filings_enabled: bool = False
    sec_user_agent: str = "trad-bot research contact@example.com"
    poll_interval_s: int = 60
    max_items_per_poll: int = 50
    # Dedup window — items with the same (source, url) seen in the last
    # N minutes are skipped.
    dedup_window_minutes: int = 240


class VectorStoreConfig(BaseModel):
    provider: str = "memory"  # "memory" | "qdrant" | "chroma"
    url: str | None = None
    collection: str = "filings"
    embedding_dim: int = 384
    top_k: int = 4


class RateLimitConfig(BaseModel):
    """Redis-backed token bucket for IBKR API calls."""

    redis_url: str | None = None  # falls back to in-memory bucket if None
    orders_per_minute: int = 30
    historical_requests_per_10min: int = 50
    market_data_lines: int = 100


class IbkrConnectionConfig(BaseModel):
    """IB Gateway connection (headless Docker recommended)."""

    host: str = "127.0.0.1"
    port: int = 4002  # 4001 live, 4002 paper for IB Gateway
    client_id: int = 17
    account: str | None = None  # leave blank for the only account on the login
    readonly: bool = False
    connect_timeout_s: float = 15.0


class SignalConfig(BaseModel):
    """Mapping from LLM verdicts to the executable signal."""

    rolling_window_minutes: int = 240
    long_threshold: float = 0.55  # composite score above this → long candidate
    short_threshold: float = -0.45  # below this → short candidate
    sma_window: int = 50
    sma_confirm_pct: float = 0.0  # 0 = price must just be above/below SMA
    rsi_window: int = 14
    rsi_long_min: float = 35.0  # ignore long if RSI < this (oversold collapse)
    rsi_short_max: float = 65.0  # ignore short if RSI > this (squeeze risk)
    technical_confirm_required: bool = True
    # Daily bars for the technicals are cached: full history once, then a
    # small incremental refresh at most this often (and on each new day).
    bars_refresh_minutes: int = 60


class RiskOverlayConfig(BaseModel):
    starting_equity_usd: Decimal = Decimal("100000")
    max_gross_exposure_pct: Decimal = Decimal("1.50")  # 150% gross
    max_net_exposure_pct: Decimal = Decimal("0.20")  # 20% net (close to dollar-neutral)
    max_position_pct: Decimal = Decimal("0.05")  # 5% of equity per name
    # beta: balance legs and cap net on beta-weighted exposure vs the
    #       benchmark (true market neutrality). dollar: legacy behaviour.
    neutrality: Literal["beta", "dollar"] = "beta"
    beta_benchmark: str = "SPY"
    beta_lookback_days: int = 60
    max_sector_pct: Decimal = Decimal("0.30")
    daily_loss_stop_pct: Decimal = Decimal("0.02")
    cumulative_loss_stop_pct: Decimal = Decimal("0.10")
    trailing_stop_pct: Decimal = Decimal("0.05")
    # Conservative default that respects IBKR pacing.
    max_orders_per_minute: int = 20


class ExecutionConfig(BaseModel):
    # Only send new orders inside the NYSE regular session, minus the
    # buffers below (holidays and early closes come from the calendar).
    enforce_market_hours: bool = True
    open_buffer_minutes: int = 5
    close_buffer_minutes: int = 10
    # limit:    marketable limit at the touch +/- limit_offset_bps
    #           (default; caps slippage, unfilled remainder is cancelled
    #           after fill_timeout_s).
    # adaptive: IBKR Adaptive algo market order (PaperBroker: market).
    # market:   plain market order.
    order_style: Literal["limit", "adaptive", "market"] = "limit"
    limit_offset_bps: Decimal = Decimal("10")
    # Short sales: require borrow (IBKR shortable shares >= multiple x
    # order size, easy-to-borrow unless allowed) and no Rule 201 SSR.
    # Missing borrow data blocks the short (fails closed).
    short_check_enabled: bool = True
    allow_hard_to_borrow: bool = False
    min_shortable_multiple: Decimal = Decimal("2")
    block_shorts_under_ssr: bool = True

    def short_policy(self) -> ShortPolicy:
        # Local import: risk.shorting -> risk.overlay -> config.
        from src.ibkr_sentiment.risk.shorting import ShortPolicy

        return ShortPolicy(
            enabled=self.short_check_enabled,
            allow_hard_to_borrow=self.allow_hard_to_borrow,
            min_shortable_multiple=self.min_shortable_multiple,
            block_under_ssr=self.block_shorts_under_ssr,
        )


class IbkrSentimentConfig(BaseModel):
    """Top-level config tree."""

    mode: IbkrMode = IbkrMode.PAPER
    universe: list[UniverseEntry]
    ibkr: IbkrConnectionConfig = Field(default_factory=IbkrConnectionConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    finbert: FinBertConfig = Field(default_factory=FinBertConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    risk: RiskOverlayConfig = Field(default_factory=RiskOverlayConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    tick_seconds: int = 60  # main loop cadence
    # Max wait for an order to fill before its remainder is cancelled.
    fill_timeout_s: float = 30.0
    db_url: str = "sqlite+aiosqlite:///data/ibkr_sentiment.db"

    @field_validator("universe")
    @classmethod
    def _non_empty_universe(cls, v: list[UniverseEntry]) -> list[UniverseEntry]:
        if not v:
            raise ValueError("universe must contain at least one symbol")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> IbkrSentimentConfig:
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls.model_validate(data)


class IbkrSecrets(BaseSettings):
    """Loaded from environment / .env. Optional — only providers you use
    need keys."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    ibkr_account: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    fingpt_api_key: str = ""
    redis_url: str = ""
    postgres_url: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    bot_log_level: str = "INFO"
