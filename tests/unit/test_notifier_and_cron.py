"""Tests for the notifier hook and launchd plist generation."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from src.adapters.fake import FakeExchange
from src.adapters.paper_binance import PaperBinanceAdapter
from src.simple_bot import SimpleBot
from src.state.db import Database
from src.strategy.sma_trend import TrendState

# ---- notifier hook --------------------------------------------------


async def _setup_with_notifier(tmp_path: Path, notifier):
    db = Database(str(tmp_path / "bot.db"))
    await db.init(starting_equity=Decimal("1000"))
    ex = FakeExchange(starting_usdt=Decimal("1000"))
    ex.set_ticker("BTC/USDT", "spot", Decimal("60000"))
    bot = SimpleBot(
        exchange=ex,
        db=db,
        symbol="BTC/USDT",
        sma_window=50,
        notifier=notifier,
    )
    return bot, ex, db


async def test_notifier_fires_on_entry(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    bot, ex, db = await _setup_with_notifier(tmp_path, lambda t, m: calls.append((t, m)))
    await bot.enable()
    bot.closes_override = [50000.0] * 49 + [70000.0]
    await bot.tick()
    assert any("→ IN" in t for t, _ in calls), f"no IN notification, got: {calls}"
    await db.close()


async def test_notifier_fires_on_exit(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    bot, ex, db = await _setup_with_notifier(tmp_path, lambda t, m: calls.append((t, m)))
    await bot.enable()
    bot.closes_override = [50000.0] * 49 + [70000.0]
    await bot.tick()
    calls.clear()
    bot.closes_override = [70000.0] * 49 + [50000.0]
    await bot.tick()
    assert any("→ OUT" in t for t, _ in calls), f"no OUT notification, got: {calls}"
    await db.close()


async def test_notifier_does_not_fire_on_no_change(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    bot, ex, db = await _setup_with_notifier(tmp_path, lambda t, m: calls.append((t, m)))
    await bot.enable()
    bot.closes_override = [50000.0] * 50  # close == SMA → OUT, already OUT
    await bot.tick()
    assert calls == [], f"unexpected notification on no-change tick: {calls}"
    await db.close()


async def test_notifier_exception_does_not_crash_tick(tmp_path: Path) -> None:
    """A buggy notifier mustn't take down a real trade."""
    def boom(title: str, message: str) -> None:
        raise RuntimeError("notifier blew up")

    bot, ex, db = await _setup_with_notifier(tmp_path, boom)
    await bot.enable()
    bot.closes_override = [50000.0] * 49 + [70000.0]
    sig = await bot.tick()
    assert sig is not None and sig.state == TrendState.IN
    assert await bot.current_state() == TrendState.IN
    await db.close()


# ---- launchd plist --------------------------------------------------


def test_plist_contains_expected_fields(tmp_path: Path) -> None:
    from src.scheduler import build_plist, paths

    p = paths(tmp_path)
    body = build_plist(p, hour_utc=0, minute_utc=5)
    assert "<key>Label</key>" in body
    assert "com.tradbot.daily" in body
    assert "scripts.tradbot" in body
    assert "<key>StartCalendarInterval</key>" in body
    assert "<key>Hour</key>" in body
    assert "<key>Minute</key>" in body
    # Should reference the project venv python.
    assert "/.venv/bin/python" in body


def test_status_when_not_installed(tmp_path: Path, monkeypatch) -> None:
    import sys

    import src.scheduler
    from src.scheduler import SchedulerPaths, status

    def mock_paths(project_root: Path) -> SchedulerPaths:
        return SchedulerPaths(
            plist=tmp_path / "Library" / "LaunchAgents" / "com.tradbot.daily.plist",
            stdout_log=tmp_path / "Library" / "Logs" / "tradbot" / "evaluate.log",
            stderr_log=tmp_path / "Library" / "Logs" / "tradbot" / "evaluate.err",
            project_root=project_root,
            venv_python=project_root / ".venv" / "bin" / "python",
        )

    monkeypatch.setattr(src.scheduler, "paths", mock_paths)

    # On non-macOS or when no plist exists, status reports not installed.
    s = status(tmp_path)
    if sys.platform != "darwin":
        assert s["installed"] is False
    else:
        # macOS: should report not installed since this is a fake path.
        assert s["installed"] is False



def test_paper_adapter_with_notifier_smoke(tmp_path: Path) -> None:
    """PaperBinanceAdapter should construct cleanly when bot has a notifier."""
    ex = PaperBinanceAdapter(starting_usdt=Decimal("100"), spot_only=True)
    db = Database(str(tmp_path / "n.db"))
    bot = SimpleBot(
        exchange=ex, db=db, symbol="BTC/USDT",
        notifier=lambda t, m: None,
    )
    assert callable(bot.notifier)
