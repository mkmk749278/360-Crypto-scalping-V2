"""Command registry – decorator-based command dispatch.

Replaces the monolithic 1,000-line elif chain with a decorator pattern so that
new commands can be added by dropping a single decorated function into the
relevant module without touching a central dispatch table.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

_TELEGRAM_MAX_MSG_CHARS: int = 4_096


def split_message(text: str, limit: int = _TELEGRAM_MAX_MSG_CHARS) -> List[str]:
    """Split *text* into chunks that fit within Telegram's message size limit."""
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


@dataclass
class CommandEntry:
    name: str
    handler: Callable
    admin: bool
    aliases: List[str]
    help_text: str
    group: str  # "signals", "engine", "channels", "deploy", "backtest"


@dataclass
class CommandContext:
    """Single context object replacing the 22-parameter __init__."""

    chat_id: str
    is_admin: bool
    telegram: Any
    router: Any
    scanner: Any
    pair_mgr: Any
    data_store: Any
    signal_queue: Any
    telemetry: Any
    signal_history: List[Any]
    paused_channels: Set[str]
    confidence_overrides: Dict[str, float]
    tasks: List[asyncio.Task]
    boot_time: float
    # Optional components
    performance_tracker: Optional[Any] = None
    circuit_breaker: Optional[Any] = None
    gem_scanner: Optional[Any] = None
    ws_spot: Optional[Any] = None
    ws_futures: Optional[Any] = None
    restart_callback: Optional[Callable] = None
    ai_insight_fn: Optional[Callable] = None
    symbols_fn: Optional[Callable] = None
    free_channel_limit: int = 5
    trade_observer: Optional[Any] = None
    alert_subscribers: Set[str] = field(default_factory=set)
    stat_filter: Optional[Any] = None
    # Backtest config (mutable, shared via context)
    bt_fee_pct: float = 0.08
    bt_slippage_pct: float = 0.02
    bt_lookahead: int = 20
    bt_min_window: int = 50

    async def reply(self, text: str) -> None:
        """Send a (potentially chunked) Telegram reply."""
        for chunk in split_message(text):
            await self.telegram.send_message(self.chat_id, chunk)


class CommandRegistry:
    """Decorator-based command dispatch registry."""

    def __init__(self) -> None:
        self._commands: Dict[str, CommandEntry] = {}

    def command(
        self,
        name: str,
        *,
        admin: bool = False,
        aliases: Optional[List[str]] = None,
        help_text: str = "",
        group: str = "general",
    ) -> Callable:
        """Register a command handler.

        Usage::

            @registry.command("/status", admin=True, group="engine")
            async def handle_status(args, ctx): ...
        """

        def decorator(fn: Callable) -> Callable:
            entry = CommandEntry(
                name=name,
                handler=fn,
                admin=admin,
                aliases=aliases or [],
                help_text=help_text,
                group=group,
            )
            self._commands[name] = entry
            for alias in aliases or []:
                self._commands[alias] = entry
            return fn

        return decorator

    async def dispatch(self, cmd: str, args: List[str], ctx: CommandContext) -> None:
        """Dispatch *cmd* to the registered handler, enforcing admin guard."""
        entry = self._commands.get(cmd)
        if entry is None:
            await ctx.reply(self._help_text(ctx.is_admin))
            return
        if entry.admin and not ctx.is_admin:
            await ctx.reply("⛔ This command is restricted to administrators.")
            return
        await entry.handler(args, ctx)

    def _help_text(self, is_admin: bool) -> str:
        """Auto-generate grouped help from registered commands."""
        groups: Dict[str, List[CommandEntry]] = {}
        seen: set[int] = set()
        for entry in self._commands.values():
            if id(entry) in seen:
                continue
            seen.add(id(entry))
            if entry.admin and not is_admin:
                continue
            groups.setdefault(entry.group, []).append(entry)

        _GROUP_LABELS = {
            "signals": "📡 Signals",
            "engine": "🔧 Engine (Admin)",
            "channels": "📢 Channels (Admin)",
            "deploy": "🚀 Deploy (Admin)",
            "backtest": "📊 Backtest (Admin)",
            "general": "ℹ️ General",
        }

        lines = ["*Available Commands:*\n"]
        for group_key in ["general", "signals", "engine", "channels", "deploy", "backtest"]:
            entries = groups.get(group_key, [])
            if not entries:
                continue
            label = _GROUP_LABELS.get(group_key, group_key.capitalize())
            lines.append(f"*{label}*")
            for e in sorted(entries, key=lambda x: x.name):
                cmd_str = e.name.replace("_", "\\_")
                lines.append(f"{cmd_str} — {e.help_text}" if e.help_text else cmd_str)
            lines.append("")
        return "\n".join(lines)
