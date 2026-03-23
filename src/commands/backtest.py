"""Backtesting commands (admin) — /bt, /bt_all, /bt_config."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from src.backtester import Backtester
from src.commands.registry import CommandContext, CommandRegistry, split_message
from src.utils import get_logger

log = get_logger("commands.backtest")
registry = CommandRegistry()

_CHANNEL_EMOJIS: Dict[str, str] = {
    "360_SCALP": "⚡",
    "360_SWING": "🏛️",
    "360_SPOT": "📈",
    "360_GEM": "💎",
}


@registry.command(
    "/bt",
    aliases=["/backtest"],
    admin=True,
    group="backtest",
    help_text="Run backtest: /bt <symbol> [channel] [lookahead]",
)
async def handle_backtest(args: List[str], ctx: CommandContext) -> None:
    if not args:
        await ctx.reply("Usage: /bt <symbol> [channel] [lookahead]")
        return
    symbol = args[0].upper()
    channel_filter: Optional[str] = args[1] if len(args) >= 2 else None
    lookahead = ctx.bt_lookahead
    if len(args) >= 3:
        try:
            lookahead = int(args[2])
        except ValueError:
            pass
    candles_by_tf = ctx.data_store.candles.get(symbol, {})
    if not candles_by_tf:
        await ctx.reply(
            f"❌ No candle data found for `{symbol}`. "
            f"Make sure the symbol is tracked and data has been seeded."
        )
        return
    await ctx.reply("⏳ Running backtest…")
    try:
        bt = Backtester(
            lookahead_candles=lookahead,
            min_window=ctx.bt_min_window,
            fee_pct=ctx.bt_fee_pct,
            slippage_pct=ctx.bt_slippage_pct,
        )
        results = await asyncio.to_thread(bt.run, candles_by_tf, symbol, channel_filter)
    except Exception as exc:
        log.error("Backtest error for %s: %s", symbol, exc)
        await ctx.reply(f"❌ Backtest failed: {exc}")
        return
    lines = [f"📊 Backtest Results — {symbol}\n"]
    for r in results:
        emoji = _CHANNEL_EMOJIS.get(r.channel, "📈")
        lines.append(f"{emoji} {r.channel}")
        lines.append(r.summary().replace(f"Backtest: {r.channel}\n", ""))
        lines.append("")
    msg = "\n".join(lines).strip()
    for chunk in split_message(msg):
        await ctx.reply(chunk)


@registry.command(
    "/bt_all",
    aliases=["/backtest_all"],
    admin=True,
    group="backtest",
    help_text="Backtest all symbols: /bt_all [channel] [lookahead]",
)
async def handle_backtest_all(args: List[str], ctx: CommandContext) -> None:
    channel_filter: Optional[str] = args[0] if args else None
    lookahead = ctx.bt_lookahead
    if len(args) >= 2:
        try:
            lookahead = int(args[1])
        except ValueError:
            pass
    all_symbols = list(ctx.data_store.candles.keys())
    if not all_symbols:
        await ctx.reply("❌ No candle data available. Wait for the data store to be seeded.")
        return
    all_symbols = sorted(
        all_symbols,
        key=lambda s: len(ctx.data_store.candles.get(s, {})),
        reverse=True,
    )[:10]
    await ctx.reply(f"⏳ Running backtest across {len(all_symbols)} tracked symbol(s)…")
    bt_all = Backtester(
        lookahead_candles=lookahead,
        min_window=ctx.bt_min_window,
        fee_pct=ctx.bt_fee_pct,
        slippage_pct=ctx.bt_slippage_pct,
    )
    agg: Dict[str, Dict] = {}
    errors: List[str] = []
    for sym in all_symbols:
        ctf = ctx.data_store.candles.get(sym, {})
        if not ctf:
            continue
        try:
            sym_results = await asyncio.to_thread(bt_all.run, ctf, sym, channel_filter)
        except Exception as exc:
            log.error("Backtest error for %s: %s", sym, exc)
            errors.append(sym)
            continue
        for r in sym_results:
            if r.channel not in agg:
                agg[r.channel] = {
                    "total_signals": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "max_drawdown": 0.0,
                }
            agg[r.channel]["total_signals"] += r.total_signals
            agg[r.channel]["wins"] += r.wins
            agg[r.channel]["losses"] += r.losses
            agg[r.channel]["total_pnl"] += r.total_pnl_pct
            agg[r.channel]["max_drawdown"] = max(agg[r.channel]["max_drawdown"], r.max_drawdown)
    lines_all = [f"📊 Backtest Summary — {len(all_symbols)} symbol(s)\n"]
    for ch, data in agg.items():
        emoji = _CHANNEL_EMOJIS.get(ch, "📈")
        total = data["total_signals"]
        wins = data["wins"]
        losses = data["losses"]
        wr = (wins / total * 100) if total > 0 else 0.0
        lines_all.append(f"{emoji} {ch}")
        lines_all.append(f"Signals: {total} | Wins: {wins} | Losses: {losses}")
        lines_all.append(f"Win Rate: {wr:.1f}%")
        lines_all.append(f"Total PnL: {data['total_pnl']:+.2f}%")
        lines_all.append(f"Max Drawdown: {data['max_drawdown']:.2f}%")
        lines_all.append("")
    if errors:
        lines_all.append(f"⚠️ Failed symbols: {', '.join(errors)}")
    if not agg:
        lines_all.append("ℹ️ No results generated.")
    msg_all = "\n".join(lines_all).strip()
    for chunk in split_message(msg_all):
        await ctx.reply(chunk)


@registry.command(
    "/bt_config",
    aliases=["/backtest_config"],
    admin=True,
    group="backtest",
    help_text="Backtest config: /bt_config [key] [value]",
)
async def handle_backtest_config(args: List[str], ctx: CommandContext) -> None:
    if not args:
        config_msg = (
            "🔧 Backtest Configuration\n"
            f"Fee: {ctx.bt_fee_pct:.2f}%\n"
            f"Slippage: {ctx.bt_slippage_pct:.2f}%\n"
            f"Lookahead: {ctx.bt_lookahead} candles\n"
            f"Min Window: {ctx.bt_min_window} candles"
        )
        await ctx.reply(config_msg)
        return
    if len(args) < 2:
        await ctx.reply(
            "Usage: /bt\\_config [key] [value]\n"
            "Keys: fee, slippage, lookahead, min\\_window"
        )
        return
    key = args[0].lower()
    val_str = args[1]
    _valid_keys = {"fee", "slippage", "lookahead", "min_window"}
    if key not in _valid_keys:
        valid_keys_str = ", ".join(sorted(_valid_keys)).replace("_", "\\_")
        await ctx.reply(f"❌ Unknown config key `{key}`. Valid keys: {valid_keys_str}")
        return
    try:
        if key in ("lookahead", "min_window"):
            parsed: float = int(val_str)
            if parsed < 1:
                raise ValueError("must be >= 1")
        else:
            parsed = float(val_str)
            if parsed < 0:
                raise ValueError("must be >= 0")
    except ValueError as exc:
        await ctx.reply(f"❌ Invalid value: {exc}")
        return
    if key == "fee":
        ctx.bt_fee_pct = float(parsed)
        await ctx.reply(f"✅ Backtest fee updated to {ctx.bt_fee_pct:.2f}%")
    elif key == "slippage":
        ctx.bt_slippage_pct = float(parsed)
        await ctx.reply(f"✅ Backtest slippage updated to {ctx.bt_slippage_pct:.2f}%")
    elif key == "lookahead":
        ctx.bt_lookahead = int(parsed)
        await ctx.reply(f"✅ Backtest lookahead updated to {ctx.bt_lookahead} candles")
    elif key == "min_window":
        ctx.bt_min_window = int(parsed)
        await ctx.reply(f"✅ Backtest min\\_window updated to {ctx.bt_min_window} candles")
