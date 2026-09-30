"""IBKR sentiment bot commands for the `tradbot` CLI.

Mirrors the structure of the Binance trend-bot commands in
`scripts/tradbot.py`: each `cmd_ibsent_*` function is a coroutine that
takes (args, console) and returns an exit code. They're imported and
registered by `scripts/tradbot.py`.

PAPER mode by default — the bot runs end-to-end against the paper
broker with stub sentiment, no IB or LLM keys needed. To go further:

  IBSENT_MODE=dry_run   ANTHROPIC_API_KEY=...   ibkr-sentiment-bot
  IBSENT_MODE=live      ANTHROPIC_API_KEY=...   ibkr-sentiment-bot
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import desc, select

# Project root on sys.path so `src.*` imports work no matter where the
# user runs this from.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Sensible defaults — overridable via env so users can switch behaviour
# without editing YAML.
IBSENT_CONFIG = os.environ.get(
    "IBSENT_CONFIG", str(_PROJECT_ROOT / "config" / "ibkr_sentiment.yaml")
)
IBSENT_MODE = os.environ.get("IBSENT_MODE", "").strip().lower()  # paper|dry_run|live


def _load_cfg():
    from src.ibkr_sentiment.config import IbkrMode, IbkrSentimentConfig

    cfg = IbkrSentimentConfig.from_yaml(IBSENT_CONFIG)
    if IBSENT_MODE:
        try:
            cfg.mode = IbkrMode(IBSENT_MODE)
        except ValueError:
            pass
    return cfg


async def _make_bot(*, with_ingestion: bool = False):
    """Build a fully-wired IbkrSentimentBot for the current mode.

    `with_ingestion=False` (the default) drops the RSS poller so
    one-shot CLI commands return quickly. The `watch` command sets it
    to True so news flows in continuously.
    """
    from src.ibkr_sentiment.main import _build_bot, _build_broker

    cfg = _load_cfg()
    broker = await _build_broker(cfg)
    bot = _build_bot(cfg, broker)
    if not with_ingestion:
        bot.ingestion = None
    await bot.start()
    return bot, cfg


def _mode_banner(console: Console, mode_value: str) -> None:
    if mode_value == "live":
        console.print(
            Panel(
                "[bold red]🚨 IBKR LIVE MODE — real US equities on Interactive Brokers 🚨[/]",
                border_style="red",
                expand=False,
            )
        )
    elif mode_value == "dry_run":
        console.print(
            Panel(
                "[bold yellow]Dry-run[/] — IB Gateway connected, orders intercepted.",
                border_style="yellow",
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                "[bold green]Paper mode[/] — in-process paper broker, stub sentiment. "
                "Safe to run as long as you want.",
                border_style="green",
                expand=False,
            )
        )


# ---- subcommands -----------------------------------------------------


async def cmd_ibsent_status(args, console: Console) -> int:
    bot, cfg = await _make_bot()
    try:
        _mode_banner(console, cfg.mode.value)
        try:
            account = await bot.broker.account_summary()
            positions = await bot.broker.positions()
        except Exception as e:
            console.print(f"[red]✗[/] Couldn't fetch account: {e}")
            return 1

        table = Table(show_header=False, expand=False, border_style="dim")
        table.add_column(style="dim")
        table.add_column(style="bold")
        table.add_row("Config", IBSENT_CONFIG)
        table.add_row("Mode", cfg.mode.value)
        table.add_row("Universe", f"{len(cfg.universe)} symbols")
        table.add_row("LLM provider", cfg.llm.provider)
        table.add_row("Net liquidation", f"${account.net_liquidation:,.2f}")
        table.add_row("Available funds", f"${account.available_funds:,.2f}")
        table.add_row("Gross position value", f"${account.gross_position_value:,.2f}")
        table.add_row("Open positions", str(len(positions)))
        console.print(Panel(table, title="IBKR sentiment bot · status", expand=False))

        if positions:
            ptable = Table(title="Open positions", expand=False)
            ptable.add_column("symbol")
            ptable.add_column("qty", justify="right")
            ptable.add_column("avg cost", justify="right")
            ptable.add_column("mark", justify="right")
            ptable.add_column("unrealized PnL", justify="right")
            for p in positions:
                colour = "green" if p.qty > 0 else "red"
                ptable.add_row(
                    p.symbol,
                    f"[{colour}]{p.qty}[/]",
                    f"${p.avg_cost:,.2f}",
                    f"${p.mark_price:,.2f}"
                    + (" [yellow](cost — no market price)[/]" if p.mark_source == "cost" else ""),
                    f"${p.unrealized_pnl:,.2f}",
                )
            console.print(ptable)
    finally:
        await bot.stop()
    return 0


async def cmd_ibsent_tick(args, console: Console) -> int:
    """Run one tick of the sentiment bot, print the report."""
    bot, cfg = await _make_bot()
    try:
        _mode_banner(console, cfg.mode.value)
        with console.status("Running funnel + technical confirm + execution..."):
            try:
                report = await bot.tick()
            except Exception as e:
                console.print(f"[red]✗[/] Tick failed: {e}")
                return 1

        notes = ", ".join(report.notes) if report.notes else "(none)"
        console.print(f"[dim]Notes:[/] {notes}")
        console.print(f"[dim]Fresh signals this tick:[/] {len(report.fresh_signals)}")
        if not report.decisions:
            console.print("[dim]No symbol decisions made (no news in rolling window).[/]")
            return 0

        dtable = Table(title="Decisions", expand=False)
        dtable.add_column("symbol")
        dtable.add_column("side")
        dtable.add_column("score", justify="right")
        dtable.add_column("conviction", justify="right")
        dtable.add_column("technical")
        for d in report.decisions:
            colour = (
                "green" if d.side.value == "long"
                else "red" if d.side.value == "short"
                else "dim"
            )
            dtable.add_row(
                d.symbol,
                f"[{colour}]{d.side.value.upper()}[/]",
                f"{d.composite_score:+.2f}",
                f"{d.conviction:.2f}",
                d.technical_reason,
            )
        console.print(dtable)

        if report.execution and report.execution.placed:
            ttable = Table(title="Orders placed", expand=False)
            ttable.add_column("symbol")
            ttable.add_column("side")
            ttable.add_column("qty", justify="right")
            ttable.add_column("avg fill", justify="right")
            ttable.add_column("status")
            for delta, result in report.execution.placed:
                ttable.add_row(
                    delta.symbol,
                    delta.side.value,
                    f"{result.filled_qty}",
                    f"${result.avg_fill_price:,.2f}",
                    result.status.value,
                )
            console.print(ttable)
        if report.execution and report.execution.rejected_by_risk:
            console.print(
                "[yellow]Rejected by risk overlay:[/] "
                f"{len(report.execution.rejected_by_risk)} target(s)"
            )
            for delta, reason in report.execution.rejected_by_risk:
                console.print(f"  · {delta.symbol}: {reason}")
        if report.execution and report.execution.errors:
            for symbol, err in report.execution.errors:
                console.print(f"[red]✗[/] {symbol}: {err}")
    finally:
        await bot.stop()
    return 0


async def cmd_ibsent_watch(args, console: Console) -> int:
    """Run the bot in a persistent loop; Ctrl+C to stop."""
    bot, cfg = await _make_bot(with_ingestion=True)
    _mode_banner(console, cfg.mode.value)
    console.print(
        f"[dim]Tick every {cfg.tick_seconds}s. Ctrl+C to stop and flatten gracefully.[/]"
    )
    try:
        while True:
            try:
                report = await bot.tick()
                now = datetime.now(UTC).strftime("%H:%M:%S")
                placed = len(report.execution.placed) if report.execution else 0
                rejected = (
                    len(report.execution.rejected_by_risk) if report.execution else 0
                )
                console.print(
                    f"[dim]{now}[/]  decisions={len(report.decisions)}  "
                    f"targets={len(report.targets)}  placed={placed}  "
                    f"rejected={rejected}  notes={report.notes or '[]'}"
                )
            except Exception as e:
                console.print(f"[red]tick error[/]: {e}")
            try:
                await asyncio.sleep(cfg.tick_seconds)
            except asyncio.CancelledError:
                break
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching.[/]")
    finally:
        await bot.stop()
    return 0


async def cmd_ibsent_signals(args, console: Console) -> int:
    """Show the most recent structured signals from the DB."""
    bot, cfg = await _make_bot()
    try:
        from src.ibkr_sentiment.state.models import StructuredSignalRow

        async with bot.db.session() as s:
            rows = (
                await s.execute(
                    select(StructuredSignalRow)
                    .order_by(desc(StructuredSignalRow.generated_at))
                    .limit(args.limit)
                )
            ).scalars().all()
        if not rows:
            console.print(
                "[dim]No signals yet. Submit some news (`tradbot ibsent-tick`) first.[/]"
            )
            return 0
        table = Table(title=f"Last {len(rows)} signals", expand=False)
        table.add_column("ts")
        table.add_column("symbol")
        table.add_column("score", justify="right")
        table.add_column("conviction", justify="right")
        table.add_column("horizon")
        table.add_column("structural")
        table.add_column("technical")
        for r in rows:
            colour = "green" if r.score > 0 else "red" if r.score < 0 else "dim"
            table.add_row(
                r.generated_at.strftime("%Y-%m-%d %H:%M"),
                r.symbol,
                f"[{colour}]{r.score:+.2f}[/]",
                f"{r.conviction:.2f}",
                r.temporal_impact,
                "yes" if r.structural else "no",
                "[green]ok[/]" if r.technical_ok else "[yellow]blocked[/]",
            )
        console.print(table)
    finally:
        await bot.stop()
    return 0


async def cmd_ibsent_trades(args, console: Console) -> int:
    bot, cfg = await _make_bot()
    try:
        from src.ibkr_sentiment.state.models import TradeRow

        async with bot.db.session() as s:
            rows = (
                await s.execute(
                    select(TradeRow)
                    .order_by(desc(TradeRow.placed_at))
                    .limit(args.limit)
                )
            ).scalars().all()
        if not rows:
            console.print("[dim]No trades recorded yet.[/]")
            return 0
        table = Table(title=f"Last {len(rows)} trades", expand=False)
        table.add_column("ts")
        table.add_column("symbol")
        table.add_column("side")
        table.add_column("qty", justify="right")
        table.add_column("fill", justify="right")
        table.add_column("status")
        for r in rows:
            side_colour = (
                "green" if r.side.value == "long"
                else "red" if r.side.value == "short"
                else "dim"
            )
            table.add_row(
                r.placed_at.strftime("%Y-%m-%d %H:%M"),
                r.symbol,
                f"[{side_colour}]{r.side.value}[/]",
                f"{r.qty}",
                f"${r.avg_fill_price:,.2f}",
                r.status,
            )
        console.print(table)
    finally:
        await bot.stop()
    return 0


async def cmd_ibsent_equity(args, console: Console) -> int:
    bot, cfg = await _make_bot()
    try:
        from src.ibkr_sentiment.state.models import EquitySnapshotRow

        async with bot.db.session() as s:
            rows = (
                await s.execute(
                    select(EquitySnapshotRow)
                    .order_by(desc(EquitySnapshotRow.ts))
                    .limit(args.limit)
                )
            ).scalars().all()
        if not rows:
            console.print("[dim]No equity snapshots yet.[/]")
            return 0
        chronological = list(reversed(rows))
        baseline = chronological[0].net_liquidation
        table = Table(title=f"Last {len(rows)} equity snapshots", expand=False)
        table.add_column("ts")
        table.add_column("NLV", justify="right")
        table.add_column("gross", justify="right")
        table.add_column("net", justify="right")
        table.add_column("open", justify="right")
        table.add_column("return", justify="right")
        for snap in chronological:
            ret_pct = (
                float(snap.net_liquidation / baseline - 1) * 100 if baseline > 0 else 0.0
            )
            ret_cell = (
                f"[green]{ret_pct:+.2f}%[/]" if ret_pct > 0
                else f"[red]{ret_pct:+.2f}%[/]" if ret_pct < 0
                else "0.00%"
            )
            table.add_row(
                snap.ts.strftime("%Y-%m-%d %H:%M"),
                f"${snap.net_liquidation:,.2f}",
                f"${snap.gross_exposure:,.2f}",
                f"${snap.net_exposure:+,.2f}",
                str(snap.open_positions),
                ret_cell,
            )
        console.print(table)
    finally:
        await bot.stop()
    return 0


async def cmd_ibsent_news(args, console: Console) -> int:
    """Manually inject a news item and immediately run a tick.

    The bot keeps its news buffer in memory, so queueing in one CLI
    process and ticking in another wouldn't see each other; this
    command does both on the same bot instance.
    """
    from src.ibkr_sentiment.sentiment.models import NewsItem

    if not args.title:
        console.print("[red]✗[/] --title is required")
        return 1
    universe = [u.symbol for u in _load_cfg().universe]
    if args.symbol and args.symbol not in universe:
        console.print(
            f"[yellow]⚠[/] {args.symbol} is not in the configured universe "
            f"({', '.join(universe[:5])}…). Funnel will still run, but no "
            "trade will fire."
        )
    item = NewsItem(
        source=args.source or "manual",
        title=args.title,
        body=args.body or "",
        symbols=(args.symbol.upper(),) if args.symbol else (),
        published_at=datetime.now(UTC),
    )
    bot, cfg = await _make_bot()
    try:
        await bot.submit_item(item)
        console.print(f"[green]✓[/] Item queued (id={item.id[:8]}…). Running tick…")
        report = await bot.tick()
        notes = ", ".join(report.notes) if report.notes else "(none)"
        console.print(f"[dim]Notes:[/] {notes}")
        console.print(f"[dim]Decisions:[/] {len(report.decisions)}")
        if report.execution:
            console.print(
                f"[dim]Placed:[/] {len(report.execution.placed)}  "
                f"[dim]Rejected:[/] {len(report.execution.rejected_by_risk)}"
            )
        for d in report.decisions:
            colour = (
                "green" if d.side.value == "long"
                else "red" if d.side.value == "short"
                else "dim"
            )
            console.print(
                f"  · {d.symbol}: [{colour}]{d.side.value.upper()}[/]  "
                f"score={d.composite_score:+.2f}  conv={d.conviction:.2f}  "
                f"({d.technical_reason})"
            )
    finally:
        await bot.stop()
    return 0


async def cmd_ibsent_flatten(args, console: Console) -> int:
    bot, cfg = await _make_bot()
    try:
        _mode_banner(console, cfg.mode.value)
        positions = await bot.broker.positions()
        if not positions:
            console.print("[dim]Nothing to flatten.[/]")
            return 0
        gross = sum(abs(p.qty * p.mark_price) for p in positions)
        if not args.yes:
            console.print(
                f"About to flatten [bold]{len(positions)}[/] positions "
                f"(gross ≈ ${gross:,.2f}). Re-run with --yes."
            )
            return 1
        with console.status("Flattening..."):
            results = await bot.execution.emergency_flatten()
        console.print(f"[green]✓[/] Flattened {len(results)} positions.")
    finally:
        await bot.stop()
    return 0


async def cmd_ibsent_config(args, console: Console) -> int:
    cfg = _load_cfg()
    table = Table(title="IBKR sentiment bot config", expand=False)
    table.add_column("setting")
    table.add_column("value")
    table.add_row("Config file", IBSENT_CONFIG)
    table.add_row("Mode", cfg.mode.value)
    table.add_row("Universe size", str(len(cfg.universe)))
    table.add_row("Universe", ", ".join(u.symbol for u in cfg.universe[:8]) + ("…" if len(cfg.universe) > 8 else ""))
    table.add_row("LLM provider", cfg.llm.provider)
    table.add_row("LLM model", cfg.llm.model)
    table.add_row("FinBERT model", cfg.finbert.model_name)
    table.add_row("RSS feeds", str(len(cfg.ingestion.rss_feeds)))
    table.add_row("Long threshold", f"{cfg.signal.long_threshold:+.2f}")
    table.add_row("Short threshold", f"{cfg.signal.short_threshold:+.2f}")
    table.add_row("SMA window", str(cfg.signal.sma_window))
    table.add_row("Max gross", f"{float(cfg.risk.max_gross_exposure_pct):.0%}")
    table.add_row("Max net", f"{float(cfg.risk.max_net_exposure_pct):.0%}")
    table.add_row("Max per-name", f"{float(cfg.risk.max_position_pct):.0%}")
    table.add_row("Daily stop", f"{float(cfg.risk.daily_loss_stop_pct):.1%}")
    table.add_row("Cumulative stop", f"{float(cfg.risk.cumulative_loss_stop_pct):.1%}")
    table.add_row("Tick seconds", str(cfg.tick_seconds))
    table.add_row("DB URL", cfg.db_url)
    console.print(table)
    return 0


# ---- argparse + menu registration -----------------------------------


def register_subparsers(sub) -> None:
    """Add ibsent-* subcommands to an argparse subparsers object."""
    sub.add_parser("ibsent-status", help="IBKR sentiment bot · account + positions.")
    sub.add_parser("ibsent-tick", help="IBKR sentiment bot · run one full tick.")
    sub.add_parser("ibsent-watch", help="IBKR sentiment bot · loop until Ctrl+C.")
    p_sigs = sub.add_parser("ibsent-signals", help="IBKR sentiment bot · recent structured signals.")
    p_sigs.add_argument("--limit", type=int, default=20)
    p_trd = sub.add_parser("ibsent-trades", help="IBKR sentiment bot · recent trades.")
    p_trd.add_argument("--limit", type=int, default=20)
    p_eq = sub.add_parser("ibsent-equity", help="IBKR sentiment bot · equity history.")
    p_eq.add_argument("--limit", type=int, default=20)
    p_news = sub.add_parser("ibsent-news", help="IBKR sentiment bot · inject a NewsItem manually.")
    p_news.add_argument("--title", required=True)
    p_news.add_argument("--body", default="")
    p_news.add_argument("--symbol", default="")
    p_news.add_argument("--source", default="manual")
    p_flat = sub.add_parser("ibsent-flatten", help="IBKR sentiment bot · emergency flatten.")
    p_flat.add_argument("--yes", action="store_true")
    sub.add_parser("ibsent-config", help="IBKR sentiment bot · print resolved config.")


HANDLERS = {
    "ibsent-status": cmd_ibsent_status,
    "ibsent-tick": cmd_ibsent_tick,
    "ibsent-watch": cmd_ibsent_watch,
    "ibsent-signals": cmd_ibsent_signals,
    "ibsent-trades": cmd_ibsent_trades,
    "ibsent-equity": cmd_ibsent_equity,
    "ibsent-news": cmd_ibsent_news,
    "ibsent-flatten": cmd_ibsent_flatten,
    "ibsent-config": cmd_ibsent_config,
}


def menu_items():
    """Return the list of (key, label, fn, default_namespace) tuples for
    the interactive submenu."""
    import argparse as _ap

    ns_default = _ap.Namespace(limit=20, yes=False, title="", body="", symbol="", source="manual")
    flatten_ns = _ap.Namespace(yes=False)
    news_ns = _ap.Namespace(title="(menu)", body="", symbol="", source="manual")
    return [
        ("1", "Status (account + positions)", cmd_ibsent_status, ns_default),
        ("2", "Run one tick (funnel + execute)", cmd_ibsent_tick, ns_default),
        ("3", "Watch (loop, Ctrl+C to stop)", cmd_ibsent_watch, ns_default),
        ("4", "Recent signals", cmd_ibsent_signals, ns_default),
        ("5", "Recent trades", cmd_ibsent_trades, ns_default),
        ("6", "Equity history", cmd_ibsent_equity, ns_default),
        ("7", "Inject test news item", _cmd_ibsent_news_interactive, news_ns),
        ("8", "Flatten all positions", cmd_ibsent_flatten, flatten_ns),
        ("9", "Config", cmd_ibsent_config, ns_default),
    ]


async def _cmd_ibsent_news_interactive(args, console: Console) -> int:
    """Menu wrapper for cmd_ibsent_news that prompts for the fields."""
    title = console.input("News title: ").strip()
    if not title:
        console.print("[yellow]cancelled[/]")
        return 0
    symbol = console.input("Symbol (e.g. AAPL): ").strip().upper()
    body = console.input("Body (optional, press Enter to skip): ").strip()
    args.title = title
    args.symbol = symbol
    args.body = body
    args.source = "manual"
    return await cmd_ibsent_news(args, console)
