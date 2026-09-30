"""SQLAlchemy models for the IBKR sentiment bot.

Designed so the same schema works on:
  * Postgres / TimescaleDB (production) — tick / sentiment history is
    naturally time-series, an extension hypertable on `ts` columns
    fits without any model changes.
  * SQLite + aiosqlite (default; testing and small deploys).

We use plain SQLAlchemy 2 typed mappings — no Postgres-specific column
types — so the SQLite fallback works out of the box.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import DECIMAL, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class FunnelStage(str, Enum):
    INGESTED = "ingested"
    SCORED = "scored"
    GATED = "gated"
    SIGNAL = "signal"


class TradeSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class NewsItemRow(Base):
    __tablename__ = "ibsent_news"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    url: Mapped[str] = mapped_column(String(1024), default="")
    title: Mapped[str] = mapped_column(String(512), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    symbols: Mapped[str] = mapped_column(String(256), default="")  # comma-joined
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class FinBertScoreRow(Base):
    __tablename__ = "ibsent_finbert"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[str] = mapped_column(String(64), index=True)
    polarity: Mapped[str] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    forwarded: Mapped[bool] = mapped_column(Boolean, default=False)
    forwarded_reason: Mapped[str] = mapped_column(String(128), default="")
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class LLMVerdictRow(Base):
    __tablename__ = "ibsent_llm_verdicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[str] = mapped_column(String(64), index=True)
    verdict: Mapped[str] = mapped_column(String(16))
    conviction: Mapped[float] = mapped_column(Float)
    temporal_impact: Mapped[str] = mapped_column(String(32))
    structural: Mapped[bool] = mapped_column(Boolean, default=False)
    source_credibility: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text, default="")
    asset_score_json: Mapped[str] = mapped_column(Text, default="{}")
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class StructuredSignalRow(Base):
    __tablename__ = "ibsent_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    score: Mapped[float] = mapped_column(Float)
    conviction: Mapped[float] = mapped_column(Float)
    temporal_impact: Mapped[str] = mapped_column(String(32))
    structural: Mapped[bool] = mapped_column(Boolean, default=False)
    sources: Mapped[str] = mapped_column(String(256), default="")  # comma-joined
    item_ids: Mapped[str] = mapped_column(Text, default="")  # comma-joined
    technical_ok: Mapped[bool] = mapped_column(Boolean, default=False)
    technical_reason: Mapped[str] = mapped_column(String(256), default="")
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class TradeRow(Base):
    __tablename__ = "ibsent_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_order_id: Mapped[str] = mapped_column(String(64), index=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[TradeSide] = mapped_column(SAEnum(TradeSide))
    qty: Mapped[Decimal] = mapped_column(DECIMAL(28, 8))  # requested
    filled_qty: Mapped[Decimal] = mapped_column(DECIMAL(28, 8), default=Decimal("0"))
    avg_fill_price: Mapped[Decimal] = mapped_column(DECIMAL(28, 8))
    status: Mapped[str] = mapped_column(String(32))
    placed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class EquitySnapshotRow(Base):
    __tablename__ = "ibsent_equity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    net_liquidation: Mapped[Decimal] = mapped_column(DECIMAL(28, 8))
    gross_exposure: Mapped[Decimal] = mapped_column(DECIMAL(28, 8))
    net_exposure: Mapped[Decimal] = mapped_column(DECIMAL(28, 8))
    open_positions: Mapped[int] = mapped_column(Integer, default=0)


class SourceAccuracyRow(Base):
    """Per-source running accuracy used to weight sentiment.

    `hits` / `total` are updated whenever a closed trade tagged with
    this source's `item_id` produces a known-sign PnL.
    """

    __tablename__ = "ibsent_source_accuracy"

    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    last_update: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    @property
    def weight(self) -> float:
        if self.total <= 0:
            return 0.5
        # Smoothed accuracy: shrinks toward 0.5 when the sample is small
        # so a one-off lucky source doesn't dominate.
        return (self.hits + 5) / (self.total + 10)
