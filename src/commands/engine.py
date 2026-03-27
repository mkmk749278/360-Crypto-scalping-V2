"""Engine monitoring commands (admin) — /status, /dashboard, /scan, /pairs, /logs."""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional

import psutil

from src.commands.registry import CommandContext, CommandRegistry
from src.logger import get_recent_logs

registry = CommandRegistry()

_TELEGRAM_LOG_MAX_CHARS: int = 3_500


@registry.command(
    "/status",
    aliases=["/engine_status"],
    admin=True,
    group="engine",
    help_text="Engine status + resource usage",
)
async def handle_status(args: List[str], ctx: CommandContext) -> None:
    uptime_s = time.monotonic() - ctx.boot_time
    hours, rem = divmod(int(uptime_s), 3600)
    minutes, secs = divmod(rem, 60)
    ws_healthy = (
        (ctx.ws_spot.is_healthy if ctx.ws_spot else True)
        and (ctx.ws_futures.is_healthy if ctx.ws_futures else True)
    )
    proc = psutil.Process()
    mem_info = proc.memory_info()
    cpu_pct = proc.cpu_percent(interval=0.1)
    lines = [
        "🔧 Engine Status",
        f"Uptime: {hours}h {minutes}m {secs}s",
        f"Running tasks: {sum(1 for t in ctx.tasks if not t.done())}",
        f"Queue size: {await ctx.signal_queue.qsize()}",
        f"Pairs: {len(ctx.pair_mgr.pairs)}",
        f"Active signals: {len(ctx.router.active_signals)}",
        f"WS healthy: {'✅' if ws_healthy else '❌'}",
        "",
        "🧠 Resources",
        f"RSS: {mem_info.rss / 1024 / 1024:.1f} MB",
        f"CPU: {cpu_pct:.1f}%",
    ]
    await ctx.reply("\n".join(lines))


@registry.command(
    "/dashboard",
    aliases=["/view_dashboard"],
    admin=True,
    group="engine",
    help_text="Telemetry dashboard",
)
async def handle_dashboard(args: List[str], ctx: CommandContext) -> None:
    await ctx.reply(ctx.telemetry.dashboard_text())


@registry.command(
    "/scan",
    aliases=["/force_scan"],
    admin=True,
    group="engine",
    help_text="Force an immediate scan",
)
async def handle_scan(args: List[str], ctx: CommandContext) -> None:
    if ctx.scanner is not None:
        ctx.scanner.force_scan = True
    await ctx.reply("⚡ Force scan triggered.")


@registry.command(
    "/pairs",
    aliases=["/view_pairs"],
    admin=True,
    group="engine",
    help_text="View pairs: /pairs [spot|futures]",
)
async def handle_pairs(args: List[str], ctx: CommandContext) -> None:
    market_filter: Optional[str] = args[0].lower() if args else None
    all_pairs = list(ctx.pair_mgr.pairs.values())
    if market_filter in ("spot", "futures"):
        all_pairs = [p for p in all_pairs if p.market == market_filter]
    sorted_pairs = sorted(all_pairs, key=lambda p: p.volume_24h_usd, reverse=True)
    top = sorted_pairs[:10]
    label = market_filter.capitalize() if market_filter else "All"
    lines = [f"📊 {label} Pairs: {len(all_pairs)} active\n\nTop 10 by volume:"]
    for i, p in enumerate(top, 1):
        lines.append(f"{i}. {p.symbol} ({p.market}) — ${p.volume_24h_usd:,.0f}")
    await ctx.reply("\n".join(lines))


@registry.command(
    "/logs",
    aliases=["/view_logs"],
    admin=True,
    group="engine",
    help_text="View recent logs: /logs [n]",
)
async def handle_logs(args: List[str], ctx: CommandContext) -> None:
    n_lines = 50
    if args:
        try:
            n_lines = int(args[0])
        except ValueError:
            pass
    n_lines = min(max(n_lines, 1), 200)
    logs = get_recent_logs(n_lines)
    if not logs:
        await ctx.reply("No log file found.")
    else:
        excerpt = logs[-_TELEGRAM_LOG_MAX_CHARS:]
        await ctx.reply(f"```\n{excerpt}\n```")


@registry.command(
    "/update_pairs",
    admin=True,
    group="engine",
    help_text="Refresh pair list: /update_pairs [spot|futures] [n]",
)
async def handle_update_pairs(args: List[str], ctx: CommandContext) -> None:
    market: Optional[str] = args[0].lower() if args else None
    count: Optional[int] = None
    if len(args) >= 2:
        try:
            count = int(args[1])
        except ValueError:
            pass
    await ctx.pair_mgr.refresh_pairs(market=market, count=count)
    await ctx.reply(f"✅ Pairs refreshed: {len(ctx.pair_mgr.pairs)} active")


@registry.command(
    "/subscribe_alerts",
    admin=True,
    group="engine",
    help_text="Subscribe to admin alerts",
)
async def handle_subscribe_alerts(args: List[str], ctx: CommandContext) -> None:
    ctx.alert_subscribers.add(ctx.chat_id)
    await ctx.reply("✅ You are subscribed to admin alerts.")


@registry.command(
    "/view_active_signals",
    admin=True,
    group="engine",
    help_text="All active signals (admin)",
)
async def handle_view_active_signals(args: List[str], ctx: CommandContext) -> None:
    sigs = list(ctx.router.active_signals.values())
    if not sigs:
        await ctx.reply("No active signals.")
        return
    lines = [f"📡 Active Signals ({len(sigs)}):"]
    for s in sigs:
        lines.append(
            f"• [{s.signal_id}] {s.symbol} {s.direction.value} | "
            f"Entry {s.entry:.4f} | SL {s.stop_loss:.4f} | "
            f"Conf {s.confidence:.0f}% | {s.status}"
        )
    await ctx.reply("\n".join(lines))


@registry.command(
    "/force_update_ai",
    admin=True,
    group="engine",
    help_text="Refresh AI/sentiment cache",
)
async def handle_force_update_ai(args: List[str], ctx: CommandContext) -> None:
    try:
        count = 0
        symbols = list(ctx.symbols_fn())[:5] if ctx.symbols_fn else []
        for sym in symbols:
            try:
                if ctx.ai_insight_fn:
                    await asyncio.wait_for(ctx.ai_insight_fn(sym), timeout=3)
                count += 1
            except Exception:
                pass
        await ctx.reply(f"✅ AI/sentiment cache refreshed for {count} symbols.")
    except Exception as exc:
        await ctx.reply(f"❌ AI refresh error: {exc}")


@registry.command(
    "/set_free_channel_limit",
    admin=True,
    group="engine",
    help_text="Set free channel daily signal limit",
)
async def handle_set_free_channel_limit(args: List[str], ctx: CommandContext) -> None:
    if not args:
        await ctx.reply("Usage: /set\\_free\\_channel\\_limit <n>")
        return
    try:
        limit = int(args[0])
        ctx.free_channel_limit = max(0, limit)
        ctx.router.set_free_limit(ctx.free_channel_limit)
        await ctx.reply(f"✅ Free channel daily signal limit set to {ctx.free_channel_limit}")
    except ValueError:
        await ctx.reply("❌ Value must be an integer.")


@registry.command(
    "/suppressed",
    aliases=["/suppression"],
    admin=True,
    group="engine",
    help_text="Show suppressed signal digest for the last 4h",
)
async def handle_suppressed(args: List[str], ctx: CommandContext) -> None:
    """Send a rolling-window suppression digest to the admin chat."""
    tracker = getattr(ctx.scanner, "suppression_tracker", None)
    if tracker is None:
        await ctx.reply("⚠️ Suppression tracker not available.")
        return
    digest = tracker.format_telegram_digest()
    await ctx.reply(digest)
