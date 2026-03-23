"""Portfolio commands — /portfolio, /trades, /leaderboard, /leverage, /risk."""

from __future__ import annotations

from typing import List

from src.commands.registry import CommandContext, CommandRegistry

registry = CommandRegistry()


@registry.command(
    "/portfolio",
    aliases=[],
    group="portfolio",
    help_text="Portfolio summary: /portfolio [channel]",
)
async def handle_portfolio(args: List[str], ctx: CommandContext) -> None:
    if ctx.paper_portfolio is None:
        await ctx.reply("ℹ️ Paper portfolio is not enabled.")
        return
    if args:
        channel_arg = args[0].upper()
        if not channel_arg.startswith("360_"):
            channel_arg = f"360_{channel_arg}"
        msg = ctx.paper_portfolio.get_channel_detail(ctx.chat_id, channel_arg)
    else:
        msg = ctx.paper_portfolio.get_portfolio_summary(ctx.chat_id)
    await ctx.reply(msg)


@registry.command("/reset_portfolio", group="portfolio", help_text="Reset paper portfolio")
async def handle_reset_portfolio(args: List[str], ctx: CommandContext) -> None:
    if ctx.paper_portfolio is None:
        await ctx.reply("ℹ️ Paper portfolio is not enabled.")
        return
    channel_arg = args[0].upper() if args else None
    if channel_arg and not channel_arg.startswith("360_"):
        channel_arg = f"360_{channel_arg}"
    msg = ctx.paper_portfolio.reset_portfolio(ctx.chat_id, channel_arg)
    await ctx.reply(msg)


@registry.command(
    "/trades",
    aliases=["/trade_history"],
    group="portfolio",
    help_text="Trade history: /trades [channel]",
)
async def handle_trades(args: List[str], ctx: CommandContext) -> None:
    if ctx.paper_portfolio is None:
        await ctx.reply("ℹ️ Paper portfolio is not enabled.")
        return
    channel_arg = args[0].upper() if args else None
    if channel_arg and not channel_arg.startswith("360_"):
        channel_arg = f"360_{channel_arg}"
    msg = ctx.paper_portfolio.get_trade_history(ctx.chat_id, channel_arg)
    await ctx.reply(msg)


@registry.command(
    "/leaderboard",
    group="portfolio",
    help_text="Top performers: /leaderboard [pnl|roi]",
)
async def handle_leaderboard(args: List[str], ctx: CommandContext) -> None:
    if ctx.paper_portfolio is None:
        await ctx.reply("ℹ️ Paper portfolio is not enabled.")
        return
    sort_by = "roi" if args and args[0].lower() == "roi" else "pnl"
    msg = ctx.paper_portfolio.get_leaderboard(sort_by=sort_by)
    await ctx.reply(msg)


@registry.command(
    "/leverage",
    aliases=["/set_leverage"],
    group="portfolio",
    help_text="Set leverage: /leverage <channel> <1-20>",
)
async def handle_leverage(args: List[str], ctx: CommandContext) -> None:
    if ctx.paper_portfolio is None:
        await ctx.reply("ℹ️ Paper portfolio is not enabled.")
        return
    if len(args) < 2:
        await ctx.reply("Usage: /leverage <channel> <1-20>")
        return
    channel_arg = args[0].upper()
    if not channel_arg.startswith("360_"):
        channel_arg = f"360_{channel_arg}"
    try:
        lev = int(args[1])
        msg = ctx.paper_portfolio.set_leverage(ctx.chat_id, channel_arg, lev)
    except ValueError:
        msg = "❌ Leverage must be a number."
    await ctx.reply(msg)


@registry.command(
    "/risk",
    aliases=["/set_risk"],
    group="portfolio",
    help_text="Set risk: /risk <channel> <0.5-10>",
)
async def handle_risk(args: List[str], ctx: CommandContext) -> None:
    if ctx.paper_portfolio is None:
        await ctx.reply("ℹ️ Paper portfolio is not enabled.")
        return
    if len(args) < 2:
        await ctx.reply("Usage: /risk <channel> <0.5-10>")
        return
    channel_arg = args[0].upper()
    if not channel_arg.startswith("360_"):
        channel_arg = f"360_{channel_arg}"
    try:
        risk = float(args[1])
        msg = ctx.paper_portfolio.set_risk(ctx.chat_id, channel_arg, risk)
    except ValueError:
        msg = "❌ Risk must be a number."
    await ctx.reply(msg)
