"""Async SQLAlchemy DAOs for the IBKR sentiment bot.

Picks the engine driver from the URL:

  * `postgresql+asyncpg://...`   — production (TimescaleDB extension
    is just an `ALTER TABLE` away; the schema is plain Postgres).
  * `sqlite+aiosqlite:///...`    — default fallback so the bot is
    runnable with no external services.

The DAOs intentionally do not return ORM objects to the rest of the
bot — they return either nothing (writes) or plain Python types — so
calling code doesn't pin a session open.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.ibkr_sentiment.broker.base import OrderResult
from src.ibkr_sentiment.sentiment.models import (
    LLMVerdict,
    NewsItem,
    StructuredSignal,
)
from src.ibkr_sentiment.sentiment.pipeline import FunnelDecision
from src.ibkr_sentiment.signal_engine.dollar_neutral import TargetPosition
from src.ibkr_sentiment.signal_engine.mapping import Side
from src.ibkr_sentiment.state.models import (
    Base,
    EquitySnapshotRow,
    FinBertScoreRow,
    LLMVerdictRow,
    NewsItemRow,
    SourceAccuracyRow,
    StructuredSignalRow,
    TradeRow,
    TradeSide,
)


def _ensure_sqlite_dir(url: str) -> None:
    """Make sure the SQLite parent directory exists; no-op for non-sqlite."""
    if url.startswith("sqlite"):
        # url like sqlite+aiosqlite:///path/to/file.db OR memory
        path_part = url.split("///", 1)[1] if "///" in url else ""
        if path_part and path_part != ":memory:":
            Path(path_part).parent.mkdir(parents=True, exist_ok=True)


class IbkrSentimentDB:
    def __init__(self, url: str = "sqlite+aiosqlite:///data/ibkr_sentiment.db"):
        _ensure_sqlite_dir(url)
        self.url = url
        self.engine = create_async_engine(url, echo=False, future=True)
        self._session = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._session() as s:
            yield s

    # ---- news / scores / verdicts -----------------------------------

    async def record_news(self, item: NewsItem) -> None:
        async with self._session() as s:
            row = NewsItemRow(
                id=item.id,
                source=item.source,
                url=item.url,
                title=item.title,
                body=item.body[:8000],
                symbols=",".join(item.symbols),
                published_at=item.published_at,
                ingested_at=item.ingested_at,
            )
            s.add(row)
            try:
                await s.commit()
            except Exception:
                await s.rollback()

    async def record_finbert(self, decisions: list[FunnelDecision]) -> None:
        if not decisions:
            return
        async with self._session() as s:
            for d in decisions:
                s.add(
                    FinBertScoreRow(
                        item_id=d.score.item_id,
                        polarity=d.score.polarity,
                        score=float(d.score.score),
                        confidence=float(d.score.confidence),
                        forwarded=d.forwarded,
                        forwarded_reason=d.reason,
                        scored_at=d.score.scored_at,
                    )
                )
            await s.commit()

    async def record_verdicts(self, verdicts: list[LLMVerdict]) -> None:
        if not verdicts:
            return
        async with self._session() as s:
            for v in verdicts:
                s.add(
                    LLMVerdictRow(
                        item_id=v.item_id,
                        verdict=v.verdict,
                        conviction=float(v.conviction),
                        temporal_impact=v.temporal_impact,
                        structural=bool(v.structural),
                        source_credibility=float(v.source_credibility),
                        rationale=v.rationale,
                        asset_score_json=json.dumps(v.asset_score),
                        decided_at=v.decided_at,
                    )
                )
            await s.commit()

    async def record_signals(self, signals: list[StructuredSignal]) -> None:
        if not signals:
            return
        async with self._session() as s:
            for sig in signals:
                s.add(
                    StructuredSignalRow(
                        symbol=sig.symbol,
                        score=float(sig.score),
                        conviction=float(sig.conviction),
                        temporal_impact=sig.temporal_impact,
                        structural=bool(sig.structural),
                        sources=",".join(sig.sources),
                        item_ids=",".join(sig.item_ids),
                        technical_ok=bool(sig.technical_ok or False),
                        technical_reason=sig.technical_reason,
                        generated_at=sig.generated_at,
                    )
                )
            await s.commit()

    # ---- trades -----------------------------------------------------

    async def record_trade(
        self, target: TargetPosition, result: OrderResult
    ) -> None:
        side_map = {Side.LONG: TradeSide.LONG, Side.SHORT: TradeSide.SHORT, Side.FLAT: TradeSide.FLAT}
        async with self._session() as s:
            s.add(
                TradeRow(
                    client_order_id=result.client_order_id,
                    broker_order_id=result.broker_order_id,
                    symbol=target.symbol,
                    side=side_map[target.side],
                    qty=abs(target.target_qty),
                    avg_fill_price=result.avg_fill_price,
                    status=result.status.value,
                    placed_at=result.submitted_at,
                )
            )
            await s.commit()

    async def record_equity(
        self,
        net_liquidation: Decimal,
        gross_exposure: Decimal,
        net_exposure: Decimal,
        open_positions: int,
    ) -> None:
        async with self._session() as s:
            s.add(
                EquitySnapshotRow(
                    net_liquidation=net_liquidation,
                    gross_exposure=gross_exposure,
                    net_exposure=net_exposure,
                    open_positions=open_positions,
                )
            )
            await s.commit()

    # ---- source accuracy --------------------------------------------

    async def source_weights(self, min_total: int = 0) -> dict[str, float]:
        """Return source → weight map. Sources below `min_total` use the
        default (0.5) prior."""
        async with self._session() as s:
            rows = (await s.execute(select(SourceAccuracyRow))).scalars().all()
        return {
            r.source: r.weight
            for r in rows
            if r.total >= min_total
        }

    async def update_source_accuracy(
        self, source: str, *, hit: bool
    ) -> None:
        async with self._session() as s:
            row = await s.get(SourceAccuracyRow, source)
            if row is None:
                row = SourceAccuracyRow(source=source, hits=0, total=0)
                s.add(row)
            row.total += 1
            if hit:
                row.hits += 1
            row.last_update = datetime.now(UTC)
            await s.commit()

    # ---- diagnostics ------------------------------------------------

    async def signals_since(
        self, since: datetime
    ) -> list[StructuredSignalRow]:
        async with self._session() as s:
            res = await s.execute(
                select(StructuredSignalRow).where(StructuredSignalRow.generated_at >= since)
            )
            return list(res.scalars().all())

    async def trade_count_last(self, window: timedelta) -> int:
        since = datetime.now(UTC) - window
        async with self._session() as s:
            res = await s.execute(
                select(func.count(TradeRow.id)).where(TradeRow.placed_at >= since)
            )
            return int(res.scalar() or 0)

    async def latest_equity(self) -> EquitySnapshotRow | None:
        async with self._session() as s:
            res = await s.execute(
                select(EquitySnapshotRow).order_by(EquitySnapshotRow.ts.desc()).limit(1)
            )
            return res.scalar_one_or_none()

    async def first_equity_since(self, since: datetime) -> EquitySnapshotRow | None:
        """Earliest equity snapshot at or after `since` — used to restore
        the daily loss-stop anchor after a restart."""
        async with self._session() as s:
            res = await s.execute(
                select(EquitySnapshotRow)
                .where(EquitySnapshotRow.ts >= since)
                .order_by(EquitySnapshotRow.ts.asc())
                .limit(1)
            )
            return res.scalar_one_or_none()
