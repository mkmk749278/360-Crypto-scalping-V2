"""Channel & safety commands (admin) — /pause, /resume, /confidence, /breaker, /stats, /gem."""

from __future__ import annotations

from typing import List

from src.commands.registry import CommandContext, CommandRegistry

registry = CommandRegistry()


@registry.command(
    "/pause",
    aliases=["/pause_channel"],
    admin=True,
    group="channels",
    help_text="Pause a channel: /pause <name>",
)
async def handle_pause(args: List[str], ctx: CommandContext) -> None:
    if not args:
        await ctx.reply("Usage: /pause <channel_name>")
        return
    name = args[0]
    ctx.paused_channels.add(name)
    await ctx.reply(f"⏸ Channel `{name}` paused.")


@registry.command(
    "/resume",
    aliases=["/resume_channel"],
    admin=True,
    group="channels",
    help_text="Resume a channel: /resume <name>",
)
async def handle_resume(args: List[str], ctx: CommandContext) -> None:
    if not args:
        await ctx.reply("Usage: /resume <channel_name>")
        return
    name = args[0]
    ctx.paused_channels.discard(name)
    await ctx.reply(f"▶️ Channel `{name}` resumed.")


@registry.command(
    "/confidence",
    aliases=["/set_confidence_threshold"],
    admin=True,
    group="channels",
    help_text="Set confidence threshold: /confidence <channel> <value>",
)
async def handle_confidence(args: List[str], ctx: CommandContext) -> None:
    if len(args) < 2:
        await ctx.reply("Usage: /confidence <channel> <value>")
        return
    channel = args[0]
    try:
        value = float(args[1])
    except ValueError:
        await ctx.reply("❌ Value must be a number.")
        return
    ctx.confidence_overrides[channel] = value
    await ctx.reply(f"✅ Confidence threshold for `{channel}` set to {value:.2f}")


@registry.command(
    "/breaker",
    admin=True,
    group="channels",
    help_text="Circuit breaker status or reset: /breaker [reset]",
)
async def handle_breaker(args: List[str], ctx: CommandContext) -> None:
    if ctx.circuit_breaker is None:
        await ctx.reply("ℹ️ Circuit breaker is not enabled.")
        return
    if args and args[0].lower() == "reset":
        ctx.circuit_breaker.reset()
        await ctx.reply(
            "✅ Circuit breaker reset. Rolling breaker history cleared and signal generation resumed."
        )
    else:
        await ctx.reply(ctx.circuit_breaker.status_text())


@registry.command(
    "/stats",
    aliases=["/real_stats"],
    admin=True,
    group="channels",
    help_text="Performance stats: /stats [channel]",
)
async def handle_stats(args: List[str], ctx: CommandContext) -> None:
    if ctx.performance_tracker is None:
        await ctx.reply("ℹ️ Performance tracker is not enabled.")
        return
    channel_arg = args[0] if args else None
    msg = ctx.performance_tracker.format_stats_message(channel=channel_arg)
    await ctx.reply(msg)


@registry.command(
    "/reset_stats",
    admin=True,
    group="channels",
    help_text="Reset performance stats: /reset_stats [channel]",
)
async def handle_reset_stats(args: List[str], ctx: CommandContext) -> None:
    if ctx.performance_tracker is None:
        await ctx.reply("ℹ️ Performance tracker is not enabled.")
        return
    channel_arg = args[0] if args else None
    cleared = ctx.performance_tracker.reset_stats(channel=channel_arg)
    label = channel_arg or "all channels"
    await ctx.reply(f"🗑 Performance stats reset: {cleared} records cleared for {label}.")


@registry.command(
    "/gem",
    aliases=["/gem_mode"],
    admin=True,
    group="channels",
    help_text="Gem scanner control: /gem [on|off|status]",
)
async def handle_gem(args: List[str], ctx: CommandContext) -> None:
    if ctx.gem_scanner is None:
        await ctx.reply("❌ Gem scanner is not initialized.")
        return
    sub = args[0].lower() if args else "status"
    if sub == "on":
        ctx.gem_scanner.enable()
        await ctx.reply(
            "💎 Gem scanner ON — macro reversal signals will publish to 360\\_GEM channel"
        )
    elif sub == "off":
        ctx.gem_scanner.disable()
        await ctx.reply("🔘 Gem scanner OFF — 360\\_GEM channel paused")
    else:
        await ctx.reply(ctx.gem_scanner.status_text())


@registry.command(
    "/gem_config",
    admin=True,
    group="channels",
    help_text="Configure gem scanner: /gem_config <key> <value>",
)
async def handle_gem_config(args: List[str], ctx: CommandContext) -> None:
    if ctx.gem_scanner is None:
        await ctx.reply("❌ Gem scanner is not initialized.")
        return
    if len(args) < 2:
        await ctx.reply("Usage: /gem\\_config <key> <value>")
        return
    _success, msg = ctx.gem_scanner.update_config(args[0], args[1])
    await ctx.reply(msg)


@registry.command(
    "/report",
    aliases=["/performance_report"],
    admin=True,
    group="channels",
    help_text="Generate HTML performance dashboard: /report",
)
async def handle_report(args: List[str], ctx: CommandContext) -> None:
    """Generate and optionally send an HTML performance report."""
    if ctx.performance_tracker is None:
        await ctx.reply("ℹ️ Performance tracker is not enabled.")
        return
    try:
        from src.performance_report import generate_html_report
        path = generate_html_report(ctx.performance_tracker)
        await ctx.reply(f"✅ Performance report generated: `{path}`")
        # Send the HTML file as a Telegram document if the API supports it
        try:
            report_bytes = open(path, "rb").read()
            await ctx.telegram.send_document(
                ctx.chat_id,
                document=report_bytes,
                filename="performance_report.html",
                caption="📊 360 Crypto — Performance Dashboard",
            )
        except Exception:
            # send_document may not be implemented in all TelegramBot versions;
            # the text confirmation above is sufficient.
            pass
    except Exception as exc:
        await ctx.reply(f"❌ Report generation failed: {exc}")


@registry.command(
    "/digest",
    aliases=["/ai_report", "/ai_digest"],
    admin=True,
    group="channels",
    help_text="On-demand AI trade digest: /digest [hours]",
)
async def handle_digest(args: List[str], ctx: CommandContext) -> None:
    """Trigger an on-demand AI trade observer digest."""
    if ctx.trade_observer is None:
        await ctx.reply("ℹ️ Trade Observer is not enabled.")
        return

    lookback_hours = None
    if args:
        try:
            lookback_hours = int(args[0])
            if lookback_hours < 1:
                lookback_hours = 1
            elif lookback_hours > 168:  # cap at 7 days
                lookback_hours = 168
        except ValueError:
            await ctx.reply("Usage: /digest [hours]  (e.g. /digest 12)")
            return

    await ctx.reply("⏳ Generating AI trade digest…")
    try:
        message = await ctx.trade_observer.run_digest_on_demand(lookback_hours)
        await ctx.reply(message)
    except Exception as exc:
        await ctx.reply(f"❌ Digest generation failed: {exc}")


@registry.command(
    "/statstats",
    admin=True,
    group="channels",
    help_text="Win-rate stats per (channel, pair, regime): /statstats",
)
async def handle_statstats(args: List[str], ctx: CommandContext) -> None:
    """Show rolling win-rate statistics for all tracked signal combinations."""
    if ctx.stat_filter is None:
        await ctx.reply("ℹ️ Statistical filter is not enabled.")
        return
    await ctx.reply(ctx.stat_filter.format_statstats())


