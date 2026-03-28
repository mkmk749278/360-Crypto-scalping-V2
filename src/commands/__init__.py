"""360-Crypto-Eye-Scalping – command dispatch package.

This package replaces the former monolithic ``src/commands.py`` (1 048 lines,
35+ elif branches) with a decorator-based command registry pattern.

**Backward-compatible surface area**: ``CommandHandler`` is retained with the
same constructor signature and ``_handle_command`` entry-point so that the
engine and existing test suite require zero changes.

Internally every command is a thin async function decorated with
``@registry.command(...)`` and stored in one of the sub-modules:

- ``signals``   — /signals, /history, /info, ...
- ``engine``    — /status, /dashboard, /scan, /pairs, /logs
- ``channels``  — /pause, /resume, /confidence, /breaker, /stats, /gem
- ``deploy``    — /deploy, /restart, /rollback
- ``backtest``  — /bt, /bt_all, /bt_config
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Set

from config import TELEGRAM_ADMIN_CHAT_ID
from src.commands.registry import CommandContext, CommandRegistry
from src.utils import get_logger

# Import sub-modules so their @registry.command decorators execute
from src.commands import backtest as _bt_mod
from src.commands import channels as _ch_mod
from src.commands import deploy as _deploy_mod
from src.commands import engine as _engine_mod
from src.commands import signals as _signals_mod

log = get_logger("commands")

_TELEGRAM_MAX_MSG_CHARS: int = 4_096

_WELCOME_MESSAGE: str = (
    "🔮 *Welcome to 360 Crypto Eye* 🔮\n\n"
    "The Ultimate Institutional AI Signal Engine\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "🧠 *What We Do*\n"
    "We run a 24/7 AI-powered engine that detects Smart Money Concepts (SMC) "
    "— liquidity sweeps, market structure shifts, fair value gaps — across "
    "50–100 crypto pairs on Binance.\n\n"
    "Every signal is scored 0–100 by our multi-layer confidence system "
    "combining technical analysis, AI sentiment, and whale flow data.\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "📡 *Our Premium Channels*\n\n"
    "⚡ *SCALP* — M1/M5 high-frequency precision entries\n"
    "🏛️ *SWING* — H1/H4 institutional swing trades\n"
    "📈 *SPOT* — H4/D1 spot accumulation entries\n"
    "💎 *GEM* — Macro reversal signals for discounted altcoins\n"
    "🆓 *Free Channel* — Daily proof-of-results highlights\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "🎯 *What You Get*\n"
    "✅ Real-time AI-scored signals with entry, SL, TP1–TP3\n"
    "✅ Live trade updates & trailing stop adjustments\n"
    "✅ AI sentiment analysis (news + social + whale)\n"
    "✅ Confidence-based risk management\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "🤖 *Bot Commands*\n"
    "/history — Recent trade history\n"
    "/signals — View active signals\n"
    "/help — Show this message\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "💎 *Start trading smarter, not harder.*"
)


def _build_global_registry() -> CommandRegistry:
    """Merge all sub-module registries into a single dispatch table."""
    global_registry = CommandRegistry()
    for sub_registry in (
        _signals_mod.registry,
        _engine_mod.registry,
        _ch_mod.registry,
        _deploy_mod.registry,
        _bt_mod.registry,
    ):
        for name, entry in sub_registry._commands.items():
            global_registry._commands[name] = entry
    return global_registry


_GLOBAL_REGISTRY: CommandRegistry = _build_global_registry()


class CommandHandler:
    """Handles all Telegram commands on behalf of the engine.

    Maintains full backward compatibility with the original 22-parameter
    constructor so that ``src/main.py`` and existing tests need no changes.
    The dispatch logic delegates to the :class:`CommandRegistry`.
    """

    def __init__(
        self,
        telegram: Any,
        telemetry: Any,
        pair_mgr: Any,
        router: Any,
        data_store: Any,
        signal_queue: Any,
        signal_history: List[Any],
        paused_channels: Set[str],
        confidence_overrides: Dict[str, float],
        scanner: Any,
        ws_spot: Optional[Any],
        ws_futures: Optional[Any],
        tasks: List[asyncio.Task],
        boot_time: float,
        free_channel_limit: int,
        alert_subscribers: Set[str],
        restart_callback: Optional[Callable] = None,
        ai_insight_fn: Optional[Callable] = None,
        symbols_fn: Optional[Callable] = None,
        performance_tracker: Optional[Any] = None,
        circuit_breaker: Optional[Any] = None,
        gem_scanner: Optional[Any] = None,
        trade_observer: Optional[Any] = None,
        stat_filter: Optional[Any] = None,
    ) -> None:
        self._telegram = telegram
        self._telemetry = telemetry
        self._pair_mgr = pair_mgr
        self._router = router
        self._data_store = data_store
        self._signal_queue = signal_queue
        self._signal_history = signal_history
        self._paused_channels = paused_channels
        self._confidence_overrides = confidence_overrides
        self._scanner = scanner
        self.ws_spot = ws_spot
        self.ws_futures = ws_futures
        self._tasks = tasks
        self.boot_time = boot_time
        self.free_channel_limit = free_channel_limit
        self._alert_subscribers = alert_subscribers
        self._restart_callback = restart_callback
        self._ai_insight_fn = ai_insight_fn
        self._symbols_fn = symbols_fn
        self._performance_tracker = performance_tracker
        self._circuit_breaker = circuit_breaker
        self._gem_scanner = gem_scanner
        self._trade_observer = trade_observer
        self._stat_filter = stat_filter
        # Mutable backtest config shared via CommandContext
        self._bt_fee_pct: float = 0.08
        self._bt_slippage_pct: float = 0.02
        self._bt_lookahead: int = 20
        self._bt_min_window: int = 50

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_welcome_message(self) -> str:
        return _WELCOME_MESSAGE

    async def _handle_command(self, text: str, chat_id: str) -> None:
        """Route an incoming Telegram command to the registered handler."""
        parts = text.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        # Legacy alias kept for backward compatibility
        _legacy_aliases: Dict[str, str] = {
            "/status": "/engine_status",
            "/signal_history": "/history",
            "/signal_info": "/info",
            "/trade_history": "/trades",
            "/pause_channel": "/pause",
            "/resume_channel": "/resume",
            "/set_confidence_threshold": "/confidence",
            "/circuit_breaker_status": "/breaker",
            "/reset_circuit_breaker": "/breaker",
            "/view_dashboard": "/dashboard",
            "/force_scan": "/scan",
            "/view_pairs": "/pairs",
            "/view_logs": "/logs",
            "/update_code": "/deploy",
            "/restart_engine": "/restart",
            "/rollback_code": "/rollback",
            "/backtest": "/bt",
            "/backtest_all": "/bt_all",
            "/backtest_config": "/bt_config",
            "/real_stats": "/stats",
            "/gem_mode": "/gem",
            "/set_leverage": "/leverage",
            "/set_risk": "/risk",
        }
        cmd = _legacy_aliases.get(cmd, cmd)

        # /start and /help are handled here (not in registry) for simplicity
        if cmd in ("/start", "/help"):
            await self._telegram.send_message(chat_id, _WELCOME_MESSAGE)
            return

        # Handle /reset_circuit_breaker as /breaker reset
        if cmd == "/breaker" and not args:
            # Check if original was reset_circuit_breaker
            original_cmd = parts[0].lower()
            if original_cmd == "/reset_circuit_breaker":
                args = ["reset"]

        is_admin = bool(TELEGRAM_ADMIN_CHAT_ID and chat_id == TELEGRAM_ADMIN_CHAT_ID)

        ctx = self._make_context(chat_id, is_admin)
        await _GLOBAL_REGISTRY.dispatch(cmd, args, ctx)
        # Propagate any mutable state back
        self._bt_fee_pct = ctx.bt_fee_pct
        self._bt_slippage_pct = ctx.bt_slippage_pct
        self._bt_lookahead = ctx.bt_lookahead
        self._bt_min_window = ctx.bt_min_window
        self.free_channel_limit = ctx.free_channel_limit

    def _make_context(self, chat_id: str, is_admin: bool) -> CommandContext:
        return CommandContext(
            chat_id=chat_id,
            is_admin=is_admin,
            telegram=self._telegram,
            router=self._router,
            scanner=self._scanner,
            pair_mgr=self._pair_mgr,
            data_store=self._data_store,
            signal_queue=self._signal_queue,
            telemetry=self._telemetry,
            signal_history=self._signal_history,
            paused_channels=self._paused_channels,
            confidence_overrides=self._confidence_overrides,
            tasks=self._tasks,
            boot_time=self.boot_time,
            performance_tracker=self._performance_tracker,
            circuit_breaker=self._circuit_breaker,
            gem_scanner=self._gem_scanner,
            ws_spot=self.ws_spot,
            ws_futures=self.ws_futures,
            restart_callback=self._restart_callback,
            ai_insight_fn=self._ai_insight_fn,
            symbols_fn=self._symbols_fn,
            free_channel_limit=self.free_channel_limit,
            trade_observer=self._trade_observer,
            alert_subscribers=self._alert_subscribers,
            stat_filter=self._stat_filter,
            bt_fee_pct=self._bt_fee_pct,
            bt_slippage_pct=self._bt_slippage_pct,
            bt_lookahead=self._bt_lookahead,
            bt_min_window=self._bt_min_window,
        )
