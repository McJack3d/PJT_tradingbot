"""Lightweight JSON-backed position + daily-PnL store.

The crypto bot trades spot long/flat across many small-caps, so the
delta-neutral `Position` table in `src/state` doesn't fit. This store
keeps just what the day-trading loop needs — entry price/time per base
asset, a per-asset cooloff clock, and a daily realized-PnL tally that
resets at UTC midnight — persisted to a small JSON file so state
survives a restart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path


@dataclass
class OpenPosition:
    base: str
    symbol: str
    qty: Decimal
    entry_price: Decimal
    quote: str
    opened_at: datetime


class PositionStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._open: dict[str, OpenPosition] = {}
        self._last_exit: dict[str, datetime] = {}
        self._realized_today: Decimal = Decimal("0")
        self._today: str = date.today().isoformat()
        self._load()

    # ---- persistence --------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (ValueError, OSError):
            return
        for b, p in (raw.get("open") or {}).items():
            self._open[b] = OpenPosition(
                base=p["base"],
                symbol=p["symbol"],
                qty=Decimal(p["qty"]),
                entry_price=Decimal(p["entry_price"]),
                quote=p["quote"],
                opened_at=datetime.fromisoformat(p["opened_at"]),
            )
        self._last_exit = {
            b: datetime.fromisoformat(ts) for b, ts in (raw.get("last_exit") or {}).items()
        }
        self._realized_today = Decimal(str(raw.get("realized_today", "0")))
        self._today = str(raw.get("today", self._today))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "open": {
                b: {**asdict(p), "qty": str(p.qty), "entry_price": str(p.entry_price),
                    "opened_at": p.opened_at.isoformat()}
                for b, p in self._open.items()
            },
            "last_exit": {b: ts.isoformat() for b, ts in self._last_exit.items()},
            "realized_today": str(self._realized_today),
            "today": self._today,
        }
        self.path.write_text(json.dumps(payload, indent=2))

    # ---- daily roll ---------------------------------------------------

    def _roll_day(self, now: datetime) -> None:
        d = now.astimezone(UTC).date().isoformat()
        if d != self._today:
            self._today = d
            self._realized_today = Decimal("0")

    # ---- queries ------------------------------------------------------

    def is_open(self, base: str) -> bool:
        return base in self._open

    def get(self, base: str) -> OpenPosition | None:
        return self._open.get(base)

    def open_positions(self) -> list[OpenPosition]:
        return list(self._open.values())

    def open_count(self) -> int:
        return len(self._open)

    def realized_today(self, now: datetime | None = None) -> Decimal:
        self._roll_day(now or datetime.now(UTC))
        return self._realized_today

    def in_cooloff(self, base: str, minutes: int, now: datetime | None = None) -> bool:
        ts = self._last_exit.get(base)
        if ts is None:
            return False
        now = now or datetime.now(UTC)
        return (now - ts).total_seconds() < minutes * 60

    # ---- mutations ----------------------------------------------------

    def record_entry(self, pos: OpenPosition) -> None:
        self._open[pos.base] = pos
        self._save()

    def record_exit(
        self, base: str, exit_price: Decimal, fees: Decimal = Decimal("0"),
        now: datetime | None = None,
    ) -> Decimal:
        """Close the position, bank realized PnL, start the cooloff clock.
        Returns the realized PnL for this trade (price PnL minus fees)."""
        now = now or datetime.now(UTC)
        self._roll_day(now)
        pos = self._open.pop(base, None)
        pnl = Decimal("0")
        if pos is not None:
            pnl = (exit_price - pos.entry_price) * pos.qty - fees
            self._realized_today += pnl
        self._last_exit[base] = now
        self._save()
        return pnl
