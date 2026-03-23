"""Signal viewing commands — /signals, /history, /info."""

from __future__ import annotations

from typing import List

from src.commands.registry import CommandContext, CommandRegistry

registry = CommandRegistry()


@registry.command("/signals", group="signals", help_text="Active signals (last 5)")
async def handle_signals(args: List[str], ctx: CommandContext) -> None:
    sigs = list(ctx.router.active_signals.values())[:5]
    if not sigs:
        await ctx.reply("No active signals.")
        return
    lines = ["📡 Active Signals (last 5):"]
    for s in sigs:
        lines.append(
            f"• {s.symbol} {s.direction.value} | "
            f"Entry {s.entry:.4f} | Conf {s.confidence:.0f}% | {s.status}"
        )
    await ctx.reply("\n".join(lines))


@registry.command(
    "/history",
    aliases=["/signal_history"],
    group="signals",
    help_text="Last 10 completed signals",
)
async def handle_history(args: List[str], ctx: CommandContext) -> None:
    recent = ctx.signal_history[-10:]
    if not recent:
        await ctx.reply("No completed signals yet.")
        return
    lines = ["📜 Signal History (last 10):"]
    for s in reversed(recent):
        lines.append(
            f"• {s.symbol} {s.direction.value} | "
            f"{s.status} | PnL {s.pnl_pct:+.2f}%"
        )
    await ctx.reply("\n".join(lines))


@registry.command(
    "/info",
    aliases=["/signal_info"],
    group="signals",
    help_text="Detailed signal info: /info <id>",
)
async def handle_info(args: List[str], ctx: CommandContext) -> None:
    if not args:
        await ctx.reply("Usage: /info <signal_id>")
        return
    sid = args[0]
    sig = ctx.router.active_signals.get(sid)
    if sig is None:
        sig = next((s for s in ctx.signal_history if s.signal_id == sid), None)
    if sig is None:
        await ctx.reply(f"❌ Signal `{sid}` not found.")
        return
    lines = [
        f"📋 Signal {sig.signal_id}",
        f"Channel: {sig.channel}",
        f"Symbol: {sig.symbol}",
        f"Direction: {sig.direction.value}",
        f"Entry: {sig.entry:.4f}",
        f"SL: {sig.stop_loss:.4f}",
        f"TP1: {sig.tp1:.4f} | TP2: {sig.tp2:.4f}"
        + (f" | TP3: {sig.tp3:.4f}" if sig.tp3 else ""),
        f"Confidence: {sig.confidence:.0f}%",
        f"Status: {sig.status}",
        f"PnL: {sig.pnl_pct:+.2f}%",
        f"AI: {sig.ai_sentiment_label}",
    ]
    await ctx.reply("\n".join(lines))


@registry.command("/signal_stats", group="signals", help_text="Signal quality stats")
async def handle_signal_stats(args: List[str], ctx: CommandContext) -> None:
    if ctx.performance_tracker is None:
        await ctx.reply("ℹ️ Performance tracker is not enabled.")
        return
    channel_arg = args[0] if args else None
    msg = ctx.performance_tracker.format_signal_quality_stats_message(channel=channel_arg)
    await ctx.reply(msg)


@registry.command("/tp_stats", group="signals", help_text="TP hit statistics")
async def handle_tp_stats(args: List[str], ctx: CommandContext) -> None:
    if ctx.performance_tracker is None:
        await ctx.reply("ℹ️ Performance tracker is not enabled.")
        return
    channel_arg = args[0] if args else None
    msg = ctx.performance_tracker.format_tp_stats_message(channel=channel_arg)
    await ctx.reply(msg)


@registry.command("/free_signals", group="signals", help_text="Today's free signals")
async def handle_free_signals(args: List[str], ctx: CommandContext) -> None:
    sigs = [s for s in ctx.router.active_signals.values() if s.channel == "free"]
    if not sigs:
        await ctx.reply("No free signals today.")
        return
    lines = ["🆓 Today's Free Picks:"]
    for s in sigs:
        lines.append(
            f"• {s.symbol} {s.direction.value} | "
            f"Entry {s.entry:.4f} | Conf {s.confidence:.0f}%"
        )
    await ctx.reply("\n".join(lines))


@registry.command("/last_update", group="signals", help_text="Last scan latency and stats")
async def handle_last_update(args: List[str], ctx: CommandContext) -> None:
    scan_ms = ctx.telemetry._scan_latency_ms
    pairs = len(ctx.pair_mgr.pairs)
    active = len(ctx.router.active_signals)
    await ctx.reply(
        f"🕐 Last scan latency: {scan_ms:.0f}ms\n"
        f"Pairs: {pairs} | Active signals: {active}"
    )


@registry.command("/subscribe", group="signals", help_text="Subscribe to premium signals")
async def handle_subscribe(args: List[str], ctx: CommandContext) -> None:
    await ctx.reply("✅ Subscribed to premium signals.")


@registry.command("/unsubscribe", group="signals", help_text="Unsubscribe from signals")
async def handle_unsubscribe(args: List[str], ctx: CommandContext) -> None:
    await ctx.reply("✅ Unsubscribed.")
