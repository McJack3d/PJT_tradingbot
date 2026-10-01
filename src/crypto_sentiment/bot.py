"""The small-cap crypto sentiment day-trading bot.

One `tick()` does:
  1. Refresh the small-cap universe (cached for `universe_refresh_minutes`).
  2. Gather fresh news for the universe and run it through the funnel
     → per-symbol `StructuredSignal`s.
  3. EXIT held names whose signal reversed, hit take-profit/stop-loss,
     or aged past the time-stop.
  4. ENTER long the best bullish candidates, subject to: conviction,
     allowed horizon, spread guard, per-asset cooloff, max concurrent
     positions, available quote balance, and the account daily-loss stop.

Spot only — every position is long or flat. The bot never shorts and
never uses leverage. In paper mode the `exchange` is a FakeExchange, so
no real order is ever sent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from src.adapters.exchange_base import ExchangeAdapter
from src.crypto_sentiment.config import CryptoSentimentConfig
from src.crypto_sentiment.feeds import CryptoNewsGatherer
from src.crypto_sentiment.positions import OpenPosition, PositionStore
from src.crypto_sentiment.universe import MarketInfo, MarketProvider, build_universe
from src.execution.order import round_qty
from src.ibkr_sentiment.sentiment.models import StructuredSignal
from src.ibkr_sentiment.sentiment.pipeline import SentimentPipeline
from src.logging_setup import log


@dataclass
class TickReport:
    """What one tick decided — returned for logging/tests, never trusted
    as state (the exchange + store are the source of truth)."""

    signals: list[StructuredSignal] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)
    closed: list[tuple[str, str]] = field(default_factory=list)  # (base, reason)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (base, reason)
    halted: bool = False


class CryptoSentimentBot:
    def __init__(
        self,
        *,
        exchange: ExchangeAdapter,
        pipeline: SentimentPipeline,
        market_provider: MarketProvider,
        news: CryptoNewsGatherer,
        store: PositionStore,
        cfg: CryptoSentimentConfig,
    ) -> None:
        self.exchange = exchange
        self.pipeline = pipeline
        self.market_provider = market_provider
        self.news = news
        self.store = store
        self.cfg = cfg
        self._universe: list[MarketInfo] = []
        self._universe_at: datetime | None = None
        self._stopping = asyncio.Event()

    # ---- universe -----------------------------------------------------

    async def refresh_universe(self, now: datetime | None = None) -> list[MarketInfo]:
        now = now or datetime.now(UTC)
        stale = (
            self._universe_at is None
            or (now - self._universe_at).total_seconds()
            >= self.cfg.universe_refresh_minutes * 60
        )
        if stale:
            markets = await self.market_provider.fetch_markets()
            self._universe = build_universe(markets, self.cfg)
            self._universe_at = now
            log.info("crypto_sentiment.universe.refreshed", size=len(self._universe))
        return self._universe

    # ---- read-only evaluation ----------------------------------------

    async def evaluate(self, now: datetime | None = None) -> list[StructuredSignal]:
        """News → signals, no trading. Safe to call any time."""
        now = now or datetime.now(UTC)
        universe = await self.refresh_universe(now)
        bases = {m.base.upper() for m in universe}
        items = await self.news.gather(bases)
        _, _, signals = await self.pipeline.run(items, now=now)
        return signals

    # ---- full tick ----------------------------------------------------

    async def tick(self, now: datetime | None = None) -> TickReport:
        now = now or datetime.now(UTC)
        report = TickReport()
        universe = await self.refresh_universe(now)
        market_by_base = {m.base.upper(): m for m in universe}

        signals = await self.evaluate(now)
        report.signals = signals
        signal_by_base = {s.symbol.upper(): s for s in signals}

        # 1. EXITS — walk currently-open positions first so freed slots
        #    can be reused by entries in the same tick.
        for pos in self.store.open_positions():
            market = market_by_base.get(pos.base) or await self._market_for(pos)
            reason = self._exit_reason(pos, signal_by_base.get(pos.base), market, now)
            if reason:
                await self._close(pos, market, reason, now)
                report.closed.append((pos.base, reason))

        # 2. ENTRIES — account daily-loss stop gates all new risk.
        if self.store.realized_today(now) <= -self.cfg.daily_loss_stop_usd:
            report.halted = True
            log.warning("crypto_sentiment.daily_stop.halt_entries",
                        realized=str(self.store.realized_today(now)))
            return report

        candidates = [
            s for s in signals
            if s.score >= self.cfg.entry_score
            and s.conviction >= self.cfg.min_conviction
            and s.temporal_impact in self.cfg.allowed_horizons
            and s.symbol.upper() in market_by_base
            and not self.store.is_open(s.symbol.upper())
        ]
        candidates.sort(key=lambda s: s.score * s.conviction, reverse=True)

        for sig in candidates:
            base = sig.symbol.upper()
            if self.store.open_count() >= self.cfg.max_concurrent_positions:
                report.skipped.append((base, "max_concurrent"))
                continue
            if self.store.in_cooloff(base, self.cfg.asset_cooloff_minutes, now):
                report.skipped.append((base, "cooloff"))
                continue
            market = market_by_base[base]
            if market.spread_pct > self.cfg.max_spread_pct:
                report.skipped.append((base, "spread"))
                continue
            if not await self._has_quote(market.quote):
                report.skipped.append((base, "no_quote_balance"))
                continue
            ok = await self._open(market, sig, now)
            if ok:
                report.opened.append(base)
            else:
                report.skipped.append((base, "open_failed"))
        return report

    # ---- exit logic ---------------------------------------------------

    def _exit_reason(
        self,
        pos: OpenPosition,
        sig: StructuredSignal | None,
        market: MarketInfo | None,
        now: datetime,
    ) -> str:
        price = market.last if market and market.last > 0 else pos.entry_price
        if pos.entry_price > 0:
            change = (price - pos.entry_price) / pos.entry_price
            if change >= self.cfg.take_profit_pct:
                return "take_profit"
            if change <= -self.cfg.stop_loss_pct:
                return "stop_loss"
        age_h = (now - pos.opened_at).total_seconds() / 3600.0
        if age_h >= self.cfg.max_hold_hours:
            return "max_hold"
        if sig is not None and sig.score <= self.cfg.exit_score:
            return "sentiment_reversal"
        return ""

    # ---- order helpers ------------------------------------------------

    async def _open(
        self, market: MarketInfo, sig: StructuredSignal, now: datetime
    ) -> bool:
        ticker = await self.exchange.fetch_ticker(market.symbol, "spot")
        ask = ticker.ask if ticker.ask and ticker.ask > 0 else ticker.last
        if ask <= 0:
            return False
        qty = round_qty(self.cfg.per_position_usd / ask, self.cfg.qty_step)
        if qty <= 0:
            return False
        try:
            order = await self.exchange.submit_order(
                symbol=market.symbol, leg="spot", side="buy", qty=qty,
                client_order_id=f"cs-{market.base}-{int(now.timestamp())}",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("crypto_sentiment.open.failed", symbol=market.symbol, error=str(e))
            return False
        fill_qty = order.filled_qty if order and order.filled_qty > 0 else qty
        fill_px = order.avg_price if order and order.avg_price > 0 else ask
        self.store.record_entry(
            OpenPosition(
                base=market.base.upper(), symbol=market.symbol, qty=fill_qty,
                entry_price=fill_px, quote=market.quote.upper(), opened_at=now,
            )
        )
        log.info("crypto_sentiment.open", symbol=market.symbol, qty=str(fill_qty),
                 price=str(fill_px), score=round(sig.score, 3),
                 conviction=round(sig.conviction, 3))
        return True

    async def _close(
        self, pos: OpenPosition, market: MarketInfo | None, reason: str, now: datetime
    ) -> None:
        symbol = market.symbol if market else f"{pos.base}/{pos.quote}"
        try:
            order = await self.exchange.submit_order(
                symbol=symbol, leg="spot", side="sell", qty=pos.qty,
                client_order_id=f"cs-{pos.base}-exit-{int(now.timestamp())}",
            )
            exit_px = order.avg_price if order and order.avg_price > 0 else pos.entry_price
            fees = order.fee_paid if order else Decimal("0")
        except Exception as e:  # noqa: BLE001
            # Couldn't sell — keep the position recorded so we retry next
            # tick rather than silently forgetting we hold it.
            log.warning("crypto_sentiment.close.failed", symbol=symbol, error=str(e))
            return
        pnl = self.store.record_exit(pos.base, exit_px, fees=fees, now=now)
        log.info("crypto_sentiment.close", symbol=symbol, reason=reason,
                 exit_price=str(exit_px), pnl=str(pnl))

    async def _has_quote(self, quote: str) -> bool:
        balances = await self.exchange.fetch_balances()
        bal = balances.get(f"spot:{quote}")
        return bal is not None and bal.free >= self.cfg.per_position_usd

    async def _market_for(self, pos: OpenPosition) -> MarketInfo | None:
        """Build a minimal MarketInfo from a live ticker for a held name
        that's no longer in the (volume-filtered) universe — we still
        need its price to manage the exit."""
        try:
            t = await self.exchange.fetch_ticker(pos.symbol, "spot")
        except Exception:  # noqa: BLE001
            return None
        return MarketInfo(
            symbol=pos.symbol, base=pos.base, quote=pos.quote, active=True,
            quote_volume_24h=Decimal("0"), last=t.last, bid=t.bid, ask=t.ask,
        )

    # ---- daemon loop --------------------------------------------------

    async def run_loop(self) -> None:
        self._stopping.clear()
        log.info("crypto_sentiment.loop.start", mode=self.cfg.mode,
                 poll_s=self.cfg.poll_interval_s)
        while not self._stopping.is_set():
            try:
                report = await self.tick()
                log.info("crypto_sentiment.tick.done", opened=report.opened,
                         closed=[c[0] for c in report.closed], halted=report.halted)
            except Exception as e:  # noqa: BLE001 - a bad tick must not kill the loop
                log.exception("crypto_sentiment.tick.error", error=str(e))
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=self.cfg.poll_interval_s
                )
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stopping.set()
