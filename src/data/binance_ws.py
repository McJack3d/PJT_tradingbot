"""Binance WebSocket client for live price and funding updates.

Subscribes to spot bookTicker and perp markPrice + bookTicker streams,
updating the shared MarketData snapshots in real-time.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal

import websockets

from src.data.market_data import MarketData
from src.logging_setup import log


class BinanceWebSocket:
    def __init__(
        self,
        market_data: MarketData,
        symbols: list[str],
        perp_symbols: list[str],
        testnet: bool = True,
    ):
        self.market_data = market_data
        self.symbols = symbols
        self.perp_symbols = perp_symbols
        self.testnet = testnet

        # Maps normalized Binance symbols (e.g. "BTCUSDT") to spot symbols (e.g. "BTC/USDT")
        self.spot_map = {s.replace("/", "").upper(): s for s in symbols}
        self.perp_map = {}
        for perp in perp_symbols:
            # perp is e.g. "BTC/USDT:USDT" -> base/quote -> "btcusdt"
            parts = perp.split(":")
            base_quote = parts[0]
            normalized = base_quote.replace("/", "").upper()
            self.perp_map[normalized] = base_quote

        self._spot_task: asyncio.Task | None = None
        self._perp_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background WebSocket connection tasks."""
        if self._spot_task is None or self._spot_task.done():
            self._spot_task = asyncio.create_task(self._spot_loop())
        if self._perp_task is None or self._perp_task.done():
            self._perp_task = asyncio.create_task(self._perp_loop())
        log.info("ws.started")

    async def stop(self) -> None:
        """Cancel and clean up the WebSocket connection tasks."""
        if self._spot_task:
            self._spot_task.cancel()
            try:
                await self._spot_task
            except asyncio.CancelledError:
                pass
            self._spot_task = None
        if self._perp_task:
            self._perp_task.cancel()
            try:
                await self._perp_task
            except asyncio.CancelledError:
                pass
            self._perp_task = None
        log.info("ws.stopped")

    async def _spot_loop(self) -> None:
        if not self.symbols:
            log.warning("ws.spot.no_symbols")
            return

        base_url = (
            "wss://testnet.binance.vision/stream"
            if self.testnet
            else "wss://stream.binance.com:9443/stream"
        )
        streams = [f"{s.replace('/', '').lower()}@bookTicker" for s in self.symbols]
        url = f"{base_url}?streams={'/'.join(streams)}"

        backoff = 1.0
        while True:
            try:
                log.info("ws.spot.connecting", url=url)
                async with websockets.connect(url) as ws:
                    log.info("ws.spot.connected")
                    backoff = 1.0
                    async for message in ws:
                        try:
                            msg = json.loads(message)
                            self._handle_spot(msg)
                        except Exception as e:
                            log.exception("ws.spot.message_error", error=str(e))
            except asyncio.CancelledError:
                log.info("ws.spot.loop.cancelled")
                raise
            except Exception as e:
                log.warning("ws.spot.connection_lost", error=str(e), retry_in=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _perp_loop(self) -> None:
        if not self.perp_symbols:
            log.warning("ws.perp.no_symbols")
            return

        base_url = (
            "wss://stream.binancefuture.com/stream"
            if self.testnet
            else "wss://fstream.binance.com/stream"
        )
        streams = []
        for p in self.perp_symbols:
            sym = p.split(":")[0].replace("/", "").lower()
            streams.append(f"{sym}@markPrice@1s")
            streams.append(f"{sym}@bookTicker")
        url = f"{base_url}?streams={'/'.join(streams)}"

        backoff = 1.0
        while True:
            try:
                log.info("ws.perp.connecting", url=url)
                async with websockets.connect(url) as ws:
                    log.info("ws.perp.connected")
                    backoff = 1.0
                    async for message in ws:
                        try:
                            msg = json.loads(message)
                            self._handle_perp(msg)
                        except Exception as e:
                            log.exception("ws.perp.message_error", error=str(e))
            except asyncio.CancelledError:
                log.info("ws.perp.loop.cancelled")
                raise
            except Exception as e:
                log.warning("ws.perp.connection_lost", error=str(e), retry_in=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def _handle_spot(self, msg: dict) -> None:
        """Process incoming spot WebSocket message."""
        data = msg.get("data")
        if not data:
            return
        s = data.get("s")
        if not s:
            return
        spot_symbol = self.spot_map.get(s.upper())
        if not spot_symbol:
            return
        snap = self.market_data.get(spot_symbol)

        # Spot book ticker updates:
        # e.g., {"s": "BTCUSDT", "b": "29995.0", "B": "1", "a": "30005.0", "A": "1"}
        if "b" in data:
            snap.spot_bid = Decimal(str(data["b"]))
        if "a" in data:
            snap.spot_ask = Decimal(str(data["a"]))

    def _handle_perp(self, msg: dict) -> None:
        """Process incoming perp WebSocket message."""
        data = msg.get("data")
        if not data:
            return
        s = data.get("s")
        if not s:
            return
        spot_symbol = self.perp_map.get(s.upper())
        if not spot_symbol:
            return
        snap = self.market_data.get(spot_symbol)

        # Perp mark price updates:
        # e.g., {"e": "markPriceUpdate", "s": "BTCUSDT", "p": "30100.5", "r": "0.0003", "T": 1716200000000}
        # Perp book ticker updates:
        # e.g., {"s": "BTCUSDT", "b": "30001.0", "B": "1", "a": "30009.0", "A": "1"}
        if data.get("e") == "markPriceUpdate":
            if "p" in data:
                snap.mark_price = Decimal(str(data["p"]))
            if "r" in data:
                snap.funding_rate = Decimal(str(data["r"]))
            if "T" in data:
                snap.next_funding_time = datetime.fromtimestamp(
                    data["T"] / 1000, tz=UTC
                )
        else:
            # Perp book ticker
            if "b" in data:
                snap.perp_bid = Decimal(str(data["b"]))
            if "a" in data:
                snap.perp_ask = Decimal(str(data["a"]))
