"""Regression tests for the three critical bugs identified by the Challenger agent.

Bug 1: Consecutive losses sorting — bar_index priority caused cross-asset misordering.
Bug 2: Cool-off bypass — stale last_loss_exit_bar propagated to all closed trades.
Bug 3: Metadata destruction — db.set_status(HALTED) wiped strategy metadata from halt_reason.

Each test reproduces the exact scenario from the bug report and verifies the fix.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.risk.perp_guards import check_asset_cooloff, check_consecutive_losses
from src.state.db import Database
from src.state.models import SystemStatusEnum

# =========================================================================
# Bug 1: Cross-asset consecutive-losses sorting
# =========================================================================
# The old code sorted by bar_index with priority (0, bar_idx), so a BTC
# trade at bar 9990 would sort AFTER an ETH trade at bar 495 regardless
# of actual UTC time.  The fix inverts the priority: (0, ts) > (1, bar_idx).


class TestBug1_CrossAssetSorting:
    """Verify that consecutive losses sorting uses UTC timestamps, not bar indices."""

    def test_cross_asset_sorted_by_timestamp_not_bar_index(self) -> None:
        """BTC closed at bar 9990 (22:00 UTC), ETH closed at bar 495 (23:00 UTC).

        Old behavior: sorted by bar_idx → [..., ETH@495, BTC@9990] → BTC appears
        last → false consecutive-loss trigger.
        Fixed behavior: sorted by timestamp → [..., BTC@22:00, ETH@23:00] → ETH
        (profit) appears last → no breach.
        """
        t_btc = datetime(2026, 6, 10, 22, 0, tzinfo=UTC)
        t_eth = datetime(2026, 6, 10, 23, 0, tzinfo=UTC)

        trades = [
            # BTC loss at bar 9990, but at 22:00 UTC
            {"symbol": "BTC/USDT", "net_pnl": Decimal("-50"), "exit_bar_index": 9990, "exit_ts": t_btc},
            # ETH loss at bar 493, at 21:00 UTC (earlier)
            {"symbol": "ETH/USDT", "net_pnl": Decimal("-30"), "exit_bar_index": 493, "exit_ts": t_btc - timedelta(hours=1)},
            # ETH loss at bar 494, at 21:30 UTC
            {"symbol": "ETH/USDT", "net_pnl": Decimal("-20"), "exit_bar_index": 494, "exit_ts": t_btc - timedelta(minutes=30)},
            # ETH PROFIT at bar 495, at 23:00 UTC (most recent by time)
            {"symbol": "ETH/USDT", "net_pnl": Decimal("100"), "exit_bar_index": 495, "exit_ts": t_eth},
        ]

        # With correct timestamp sorting, last trade is ETH profit → no breach
        result = check_consecutive_losses(trades, max_consecutive_losses=3)
        assert result.ok, (
            "Should NOT trigger consecutive-loss breaker: most recent trade by UTC "
            "is a profit (ETH at 23:00), even though BTC has a higher bar_index (9990)"
        )

    def test_cross_asset_true_consecutive_losses(self) -> None:
        """When the most recent N trades by UTC time ARE all losses, the breaker fires."""
        base = datetime(2026, 6, 10, 20, 0, tzinfo=UTC)
        trades = [
            {"symbol": "BTC/USDT", "net_pnl": Decimal("100"), "exit_ts": base},
            {"symbol": "ETH/USDT", "net_pnl": Decimal("-10"), "exit_ts": base + timedelta(hours=1)},
            {"symbol": "BTC/USDT", "net_pnl": Decimal("-20"), "exit_ts": base + timedelta(hours=2)},
            {"symbol": "ETH/USDT", "net_pnl": Decimal("-5"), "exit_ts": base + timedelta(hours=3)},
        ]
        result = check_consecutive_losses(trades, max_consecutive_losses=3)
        assert not result.ok, "Last 3 trades by UTC time are all losses → breaker should fire"

    def test_timestamp_priority_over_bar_index(self) -> None:
        """When both bar_index and timestamp are present, timestamp takes priority."""
        # Trades with conflicting orderings: bar_index says A < B, timestamp says B < A
        t1 = datetime(2026, 6, 10, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 6, 10, 11, 0, tzinfo=UTC)

        trades = [
            # Trade A: high bar_index but EARLIER timestamp → should sort first
            {"net_pnl": Decimal("-10"), "exit_bar_index": 500, "exit_ts": t1},
            # Trade B: low bar_index but LATER timestamp → should sort second
            {"net_pnl": Decimal("50"), "exit_bar_index": 100, "exit_ts": t2},
        ]
        # If sorted by timestamp: A (loss) then B (profit) → last = profit → pass
        # If sorted by bar_index: B (profit) then A (loss) → last = loss → would affect cooloff
        result = check_consecutive_losses(trades, max_consecutive_losses=1)
        assert result.ok, "Timestamp priority means most recent trade is B (profit)"


# =========================================================================
# Bug 2: Cool-off bypass via stale last_loss_exit_bar metadata
# =========================================================================
# The old code applied `last_loss_exit_bar` from metadata to EVERY closed
# trade for the symbol, overriding their actual close bar indices.
# This is tested at the RegimeLiveBot level (integration), but we can also
# verify the underlying guard logic directly.


class TestBug2_CooloffBypass:
    """Verify that cool-off correctly uses each trade's own exit bar, not a stale value."""

    def test_cooloff_respects_individual_exit_bars(self) -> None:
        """Each trade should be sorted by its own exit_bar_index.

        If a stale exit_bar=100 were applied to all trades (old bug), the last
        trade would appear at bar 100 instead of bar 110, and cool-off at bar
        112 would pass (elapsed=12 > 6) even though the real loss is 2 bars ago.
        """
        trades = [
            # Older trade: profit at bar 90
            {"symbol": "BTC/USDT", "net_pnl": Decimal("50"), "exit_bar_index": 90, "exit_ts": datetime(2026, 6, 10, 10, 0, tzinfo=UTC)},
            # Recent trade: loss at bar 110
            {"symbol": "BTC/USDT", "net_pnl": Decimal("-20"), "exit_bar_index": 110, "exit_ts": datetime(2026, 6, 10, 12, 0, tzinfo=UTC)},
        ]

        # At bar 112, only 2 bars since the loss → should be in cool-off (6 bar window)
        result = check_asset_cooloff("BTC/USDT", trades, current_bar_index=112, cooloff_bars=6)
        assert not result.ok, "Should be in cool-off: only 2 bars since loss at bar 110"

    def test_cooloff_clears_after_enough_bars(self) -> None:
        """After the cool-off period expires, entries are allowed again."""
        trades = [
            {"symbol": "BTC/USDT", "net_pnl": Decimal("-20"), "exit_bar_index": 100, "exit_ts": datetime(2026, 6, 10, 10, 0, tzinfo=UTC)},
        ]
        # At bar 106, exactly 6 bars elapsed → cool-off should be clear
        result = check_asset_cooloff("BTC/USDT", trades, current_bar_index=106, cooloff_bars=6)
        assert result.ok, "Cool-off should clear at exactly cooloff_bars elapsed"

    def test_cooloff_only_looks_at_last_trade(self) -> None:
        """Even if older trades had losses, only the most recent trade matters."""
        trades = [
            # Older loss at bar 80
            {"symbol": "BTC/USDT", "net_pnl": Decimal("-50"), "exit_bar_index": 80, "exit_ts": datetime(2026, 6, 10, 8, 0, tzinfo=UTC)},
            # Most recent trade: profit at bar 100
            {"symbol": "BTC/USDT", "net_pnl": Decimal("30"), "exit_bar_index": 100, "exit_ts": datetime(2026, 6, 10, 10, 0, tzinfo=UTC)},
        ]
        # Even at bar 82 (close to older loss), the most recent trade is a profit
        result = check_asset_cooloff("BTC/USDT", trades, current_bar_index=102, cooloff_bars=6)
        assert result.ok, "Last trade is a profit → no cool-off regardless of older losses"


# =========================================================================
# Bug 3: Metadata destruction on halt
# =========================================================================
# The old code stored strategy metadata (stop prices, entry legs) in
# SystemStatus.halt_reason.  When set_status(HALTED, reason=...) was called,
# it overwrote halt_reason with a plain-text reason, destroying all metadata.
# The fix adds a dedicated `strategy_meta` column.


class TestBug3_MetadataDestruction:
    """Verify that halting the bot does NOT destroy strategy metadata."""

    @pytest.mark.asyncio
    async def test_set_status_halted_preserves_strategy_meta(self, db: Database) -> None:
        """set_status(HALTED) should NOT touch strategy_meta."""
        # 1. Write strategy metadata
        await db.set_strategy_meta("BTC/USDT_stop_price:65000|BTC/USDT_entry_leg:trend|enabled:true")

        # 2. Halt the bot with a reason (this used to destroy metadata)
        await db.set_status(SystemStatusEnum.HALTED, reason="Consecutive losses limit reached")

        # 3. Verify metadata survived
        meta = await db.get_strategy_meta()
        assert meta is not None, "strategy_meta should survive a halt"
        assert "BTC/USDT_stop_price:65000" in meta
        assert "BTC/USDT_entry_leg:trend" in meta
        assert "enabled:true" in meta

        # 4. Verify halt_reason is the actual reason text, not metadata
        status = await db.get_status()
        assert status.status == SystemStatusEnum.HALTED
        assert status.halt_reason == "Consecutive losses limit reached"

    @pytest.mark.asyncio
    async def test_strategy_meta_independent_of_halt_reason(self, db: Database) -> None:
        """strategy_meta and halt_reason are fully independent columns."""
        await db.set_strategy_meta("key1:val1|key2:val2")
        await db.set_status(SystemStatusEnum.ACTIVE, reason=None)

        meta = await db.get_strategy_meta()
        assert meta == "key1:val1|key2:val2"

        status = await db.get_status()
        assert status.halt_reason is None
        assert status.strategy_meta == "key1:val1|key2:val2"

    @pytest.mark.asyncio
    async def test_strategy_meta_roundtrip(self, db: Database) -> None:
        """Write then read strategy_meta should return the same value."""
        test_data = "sym_stop:100.5|sym_leg:range|sym_atr:2.3|enabled:true"
        await db.set_strategy_meta(test_data)
        assert await db.get_strategy_meta() == test_data

    @pytest.mark.asyncio
    async def test_strategy_meta_overwrite(self, db: Database) -> None:
        """Writing strategy_meta again fully replaces the old value."""
        await db.set_strategy_meta("old:data")
        await db.set_strategy_meta("new:data")
        assert await db.get_strategy_meta() == "new:data"

    @pytest.mark.asyncio
    async def test_strategy_meta_none_by_default(self, db: Database) -> None:
        """On a fresh DB, strategy_meta should be None."""
        meta = await db.get_strategy_meta()
        assert meta is None
