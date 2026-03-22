"""Command Handler – Telegram admin and user command routing.

Extracted from :class:`src.main.CryptoSignalEngine` for modularity.
The engine delegates all incoming Telegram text commands to
:meth:`CommandHandler._handle_command`.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import psutil

from config import TELEGRAM_ADMIN_CHAT_ID
from src.backtester import Backtester
from src.logger import get_recent_logs
from src.utils import get_logger

log = get_logger("commands")

_TELEGRAM_LOG_MAX_CHARS: int = 3_500
_REPO_ROOT: Path = Path(__file__).parent.parent
_TELEGRAM_MAX_MSG_CHARS: int = 4_096

_CHANNEL_EMOJIS: Dict[str, str] = {
    "360_SCALP": "⚡",
    "360_SWING": "🏛️",
    "360_RANGE": "⚖️",
    "360_THE_TAPE": "🐋",
}

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
    "⚖️ *RANGE* — M15 mean-reversion with DCA\n"
    "🐋 *THE TAPE* — Real-time whale flow tracking\n"
    "🆓 *Free Channel* — Daily proof-of-results highlights\n"
    "🌟 *SELECT* — Curated best-of-best signals\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "🎯 *What You Get*\n"
    "✅ Real-time AI-scored signals with entry, SL, TP1–TP3\n"
    "✅ Live trade updates & trailing stop adjustments\n"
    "✅ AI sentiment analysis (news + social + whale)\n"
    "✅ Paper trading portfolio to track performance\n"
    "✅ Confidence-based risk management\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "🤖 *Bot Commands*\n"
    "/portfolio — View your paper trading portfolio\n"
    "/history — Recent trade history\n"
    "/leaderboard — Top performers\n"
    "/signals — View active signals\n"
    "/help — Show this message\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "💎 *Start trading smarter, not harder.*\n"
    "Type /portfolio to begin your paper trading journey!"
)


def _split_message(text: str, limit: int = _TELEGRAM_MAX_MSG_CHARS) -> List[str]:
    """Split *text* into chunks that fit within Telegram's message size limit."""
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline boundary
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


class CommandHandler:
    """Handles all Telegram commands on behalf of the engine.

    Parameters
    ----------
    telegram:
        :class:`src.telegram_bot.TelegramBot` instance.
    telemetry:
        :class:`src.telemetry.TelemetryCollector` instance.
    pair_mgr:
        :class:`src.pair_manager.PairManager` instance.
    router:
        :class:`src.signal_router.SignalRouter` instance.
    data_store:
        :class:`src.historical_data.HistoricalDataStore` instance.
    signal_queue:
        :class:`src.signal_queue.SignalQueue` instance.
    signal_history:
        Mutable list of completed :class:`src.channels.base.Signal` objects.
    paused_channels:
        Mutable set of paused channel names (shared with Scanner).
    confidence_overrides:
        Mutable dict of per-channel confidence thresholds (shared with Scanner).
    scanner:
        :class:`src.scanner.Scanner` instance (used for force-scan and
        circuit-breaker access).
    ws_spot:
        Optional spot :class:`src.websocket_manager.WebSocketManager`.
    ws_futures:
        Optional futures :class:`src.websocket_manager.WebSocketManager`.
    tasks:
        Mutable list of running :class:`asyncio.Task` objects.
    boot_time:
        Monotonic timestamp of engine boot (set by Bootstrap).
    free_channel_limit:
        Daily free-signal limit.
    alert_subscribers:
        Mutable set of chat IDs subscribed to admin alerts.
    restart_callback:
        Async callable that restarts all engine tasks.
    ai_insight_fn:
        ``get_ai_insight`` function reference.
    symbols_fn:
        Callable returning the current set of tracked symbols.
    performance_tracker:
        Optional :class:`src.performance_tracker.PerformanceTracker`.
    circuit_breaker:
        Optional :class:`src.circuit_breaker.CircuitBreaker`.
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
        paper_portfolio: Optional[Any] = None,
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
        self._paper_portfolio = paper_portfolio
        # Backtest configuration defaults
        self._bt_fee_pct: float = 0.08
        self._bt_slippage_pct: float = 0.02
        self._bt_lookahead: int = 20
        self._bt_min_window: int = 50

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_welcome_message(self) -> str:
        """Return the branded welcome message text."""
        return _WELCOME_MESSAGE

    async def _handle_command(self, text: str, chat_id: str) -> None:
        """Route an incoming Telegram command to the appropriate handler."""
        parts = text.strip().split()
        cmd = parts[0].lower()

        # Command aliases
        _aliases = {"/status": "/engine_status"}
        cmd = _aliases.get(cmd, cmd)

        # --- User-facing commands (no admin gate) ---
        if cmd in ("/start", "/help"):
            if self._paper_portfolio is not None:
                try:
                    self._paper_portfolio.ensure_user(chat_id)
                except Exception as exc:
                    log.debug("Failed to register user %s for paper portfolio: %s", chat_id, exc)
            await self._telegram.send_message(chat_id, _WELCOME_MESSAGE)
            return

        is_admin = bool(TELEGRAM_ADMIN_CHAT_ID and chat_id == TELEGRAM_ADMIN_CHAT_ID)

        # --- Admin-only guard ---
        admin_cmds = {
            "/view_dashboard", "/update_pairs", "/subscribe_alerts",
            "/view_pairs", "/force_scan", "/pause_channel", "/resume_channel",
            "/set_confidence_threshold", "/engine_status", "/memory_usage",
            "/set_free_channel_limit", "/force_update_ai", "/view_active_signals",
            "/view_logs", "/update_code", "/restart_engine", "/rollback_code",
            "/circuit_breaker_status", "/reset_circuit_breaker",
            "/gem_mode", "/gem_config", "/reset_stats",
            "/real_stats", "/stats",
            "/backtest", "/backtest_all", "/backtest_config",
        }
        if cmd in admin_cmds and not is_admin:
            await self._telegram.send_message(
                chat_id,
                "⛔ This command is restricted to administrators.",
            )
            return

        if cmd == "/view_dashboard":
            await self._telegram.send_message(chat_id, self._telemetry.dashboard_text())

        elif cmd == "/update_pairs":
            market: Optional[str] = parts[1].lower() if len(parts) >= 2 else None
            count: Optional[int] = None
            if len(parts) >= 3:
                try:
                    count = int(parts[2])
                except ValueError:
                    pass
            await self._pair_mgr.refresh_pairs(market=market, count=count)
            await self._telegram.send_message(
                chat_id, f"✅ Pairs refreshed: {len(self._pair_mgr.pairs)} active"
            )

        elif cmd == "/subscribe_alerts":
            self._alert_subscribers.add(chat_id)
            await self._telegram.send_message(
                chat_id, "✅ You are subscribed to admin alerts."
            )

        elif cmd == "/view_pairs":
            market_filter: Optional[str] = parts[1].lower() if len(parts) >= 2 else None
            all_pairs = list(self._pair_mgr.pairs.values())
            if market_filter in ("spot", "futures"):
                all_pairs = [p for p in all_pairs if p.market == market_filter]
            sorted_pairs = sorted(all_pairs, key=lambda p: p.volume_24h_usd, reverse=True)
            top = sorted_pairs[:10]
            label = market_filter.capitalize() if market_filter else "All"
            lines = [f"📊 {label} Pairs: {len(all_pairs)} active\n\nTop 10 by volume:"]
            for i, p in enumerate(top, 1):
                lines.append(f"{i}. {p.symbol} ({p.market}) — ${p.volume_24h_usd:,.0f}")
            await self._telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/force_scan":
            if self._scanner is not None:
                self._scanner.force_scan = True
            await self._telegram.send_message(chat_id, "⚡ Force scan triggered.")

        elif cmd == "/pause_channel":
            if len(parts) < 2:
                await self._telegram.send_message(chat_id, "Usage: /pause\\_channel <name>")
            else:
                name = parts[1]
                self._paused_channels.add(name)
                await self._telegram.send_message(chat_id, f"⏸ Channel `{name}` paused.")

        elif cmd == "/resume_channel":
            if len(parts) < 2:
                await self._telegram.send_message(chat_id, "Usage: /resume\\_channel <name>")
            else:
                name = parts[1]
                self._paused_channels.discard(name)
                await self._telegram.send_message(chat_id, f"▶️ Channel `{name}` resumed.")

        elif cmd == "/set_confidence_threshold":
            if len(parts) < 3:
                await self._telegram.send_message(
                    chat_id, "Usage: /set\\_confidence\\_threshold <channel> <value>"
                )
            else:
                channel = parts[1]
                try:
                    value = float(parts[2])
                except ValueError:
                    await self._telegram.send_message(chat_id, "❌ Value must be a number.")
                    return
                self._confidence_overrides[channel] = value
                await self._telegram.send_message(
                    chat_id,
                    f"✅ Confidence threshold for `{channel}` set to {value:.2f}",
                )

        elif cmd == "/engine_status":
            uptime_s = time.monotonic() - self.boot_time
            hours, rem = divmod(int(uptime_s), 3600)
            minutes, secs = divmod(rem, 60)
            ws_healthy = (
                (self.ws_spot.is_healthy if self.ws_spot else True)
                and (self.ws_futures.is_healthy if self.ws_futures else True)
            )
            lines = [
                "🔧 Engine Status",
                f"Uptime: {hours}h {minutes}m {secs}s",
                f"Running tasks: {sum(1 for t in self._tasks if not t.done())}",
                f"Queue size: {await self._signal_queue.qsize()}",
                f"Pairs: {len(self._pair_mgr.pairs)}",
                f"Active signals: {len(self._router.active_signals)}",
                f"WS healthy: {'✅' if ws_healthy else '❌'}",
            ]
            await self._telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/memory_usage":
            proc = psutil.Process()
            mem_info = proc.memory_info()
            cpu_pct = proc.cpu_percent(interval=0.1)
            children = proc.children(recursive=True)
            child_rss = sum(c.memory_info().rss for c in children if c.is_running())
            lines = [
                "🧠 Memory & CPU Usage",
                f"RSS: {mem_info.rss / 1024 / 1024:.1f} MB",
                f"VMS: {mem_info.vms / 1024 / 1024:.1f} MB",
                f"CPU: {cpu_pct:.1f}%",
                f"Child processes RSS: {child_rss / 1024 / 1024:.1f} MB",
            ]
            await self._telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/set_free_channel_limit":
            if len(parts) < 2:
                await self._telegram.send_message(
                    chat_id, "Usage: /set\\_free\\_channel\\_limit <n>"
                )
            else:
                try:
                    limit = int(parts[1])
                    self.free_channel_limit = max(0, limit)
                    self._router.set_free_limit(self.free_channel_limit)
                    await self._telegram.send_message(
                        chat_id,
                        f"✅ Free channel daily signal limit set to {self.free_channel_limit}",
                    )
                except ValueError:
                    await self._telegram.send_message(chat_id, "❌ Value must be an integer.")

        elif cmd == "/force_update_ai":
            try:
                count = 0
                symbols = (
                    list(self._symbols_fn())[:5] if self._symbols_fn else []
                )
                for sym in symbols:
                    try:
                        if self._ai_insight_fn:
                            await asyncio.wait_for(
                                self._ai_insight_fn(sym), timeout=3
                            )
                        count += 1
                    except Exception:
                        pass
                await self._telegram.send_message(
                    chat_id,
                    f"✅ AI/sentiment cache refreshed for {count} symbols.",
                )
            except Exception as exc:
                await self._telegram.send_message(
                    chat_id, f"❌ AI refresh error: {exc}"
                )

        elif cmd == "/view_active_signals":
            sigs = list(self._router.active_signals.values())
            if not sigs:
                await self._telegram.send_message(chat_id, "No active signals.")
            else:
                lines = [f"📡 Active Signals ({len(sigs)}):"]
                for s in sigs:
                    lines.append(
                        f"• [{s.signal_id}] {s.symbol} {s.direction.value} | "
                        f"Entry {s.entry:.4f} | SL {s.stop_loss:.4f} | "
                        f"Conf {s.confidence:.0f}% | {s.status}"
                    )
                await self._telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/view_logs":
            n_lines = 50
            if len(parts) >= 2:
                try:
                    n_lines = int(parts[1])
                except ValueError:
                    pass
            n_lines = min(max(n_lines, 1), 200)
            logs = get_recent_logs(n_lines)
            if not logs:
                await self._telegram.send_message(chat_id, "No log file found.")
            else:
                excerpt = logs[-_TELEGRAM_LOG_MAX_CHARS:]
                await self._telegram.send_message(chat_id, f"```\n{excerpt}\n```")

        elif cmd == "/update_code":
            await self._telegram.send_message(chat_id, "⏳ Running git pull …")
            try:
                result = subprocess.run(
                    ["git", "pull"],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(_REPO_ROOT),
                )
                output = (result.stdout + result.stderr).strip() or "No output."
                await self._telegram.send_message(
                    chat_id, f"✅ git pull result:\n```\n{output}\n```"
                )
            except subprocess.TimeoutExpired:
                await self._telegram.send_message(chat_id, "❌ git pull timed out.")
            except Exception as exc:
                await self._telegram.send_message(
                    chat_id, f"❌ git pull error: {exc}"
                )

        elif cmd == "/restart_engine":
            await self._telegram.send_message(chat_id, "🔄 Restarting engine tasks …")
            try:
                if self._restart_callback:
                    await self._restart_callback(chat_id)
                else:
                    await self._telegram.send_message(
                        chat_id, "❌ Restart callback not configured."
                    )
            except Exception as exc:
                log.error("Restart error: %s", exc)
                await self._telegram.send_message(
                    chat_id, f"❌ Restart error: {exc}"
                )

        elif cmd == "/rollback_code":
            if len(parts) < 2:
                await self._telegram.send_message(
                    chat_id, "Usage: /rollback\\_code <commit>"
                )
            else:
                commit = parts[1]
                if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9.\-]{0,79}$', commit):
                    await self._telegram.send_message(
                        chat_id, "❌ Invalid commit reference."
                    )
                else:
                    await self._telegram.send_message(
                        chat_id, f"⏳ Running git checkout {commit} …"
                    )
                    try:
                        result = subprocess.run(
                            ["git", "checkout", commit],
                            capture_output=True, text=True, timeout=30,
                            cwd=str(_REPO_ROOT),
                        )
                        output = (result.stdout + result.stderr).strip() or "Done."
                        await self._telegram.send_message(
                            chat_id,
                            f"✅ Rollback result:\n```\n{output}\n```",
                        )
                    except subprocess.TimeoutExpired:
                        await self._telegram.send_message(
                            chat_id, "❌ git checkout timed out."
                        )
                    except Exception as exc:
                        await self._telegram.send_message(
                            chat_id, f"❌ Rollback error: {exc}"
                        )

        elif cmd == "/circuit_breaker_status":
            if self._circuit_breaker:
                await self._telegram.send_message(
                    chat_id, self._circuit_breaker.status_text()
                )
            else:
                await self._telegram.send_message(
                    chat_id, "ℹ️ Circuit breaker is not enabled."
                )

        elif cmd == "/reset_circuit_breaker":
            if self._circuit_breaker:
                self._circuit_breaker.reset()
                await self._telegram.send_message(
                    chat_id,
                    "✅ Circuit breaker reset. Rolling breaker history cleared and signal generation resumed.",
                )
            else:
                await self._telegram.send_message(
                    chat_id, "ℹ️ Circuit breaker is not enabled."
                )

        elif cmd == "/stats":
            if self._performance_tracker is None:
                await self._telegram.send_message(
                    chat_id, "ℹ️ Performance tracker is not enabled."
                )
            else:
                channel_arg = parts[1] if len(parts) >= 2 else None
                msg = self._performance_tracker.format_stats_message(
                    channel=channel_arg
                )
                await self._telegram.send_message(chat_id, msg)

        elif cmd == "/real_stats":
            if self._performance_tracker is None:
                await self._telegram.send_message(
                    chat_id, "ℹ️ Performance tracker is not enabled."
                )
            else:
                channel_arg = parts[1] if len(parts) >= 2 else None
                msg = self._performance_tracker.format_stats_message(
                    channel=channel_arg
                )
                await self._telegram.send_message(chat_id, msg)

        elif cmd == "/reset_stats":
            if self._performance_tracker is None:
                await self._telegram.send_message(
                    chat_id, "ℹ️ Performance tracker is not enabled."
                )
            else:
                channel_arg = parts[1] if len(parts) >= 2 else None
                cleared = self._performance_tracker.reset_stats(channel=channel_arg)
                label = channel_arg or "all channels"
                await self._telegram.send_message(
                    chat_id,
                    f"🗑 Performance stats reset: {cleared} records cleared for {label}.",
                )

        # --- User commands ---

        elif cmd == "/signals":
            sigs = list(self._router.active_signals.values())[:5]
            if not sigs:
                await self._telegram.send_message(chat_id, "No active signals.")
            else:
                lines = ["📡 Active Signals (last 5):"]
                for s in sigs:
                    lines.append(
                        f"• {s.symbol} {s.direction.value} | "
                        f"Entry {s.entry:.4f} | Conf {s.confidence:.0f}% | {s.status}"
                    )
                await self._telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/free_signals":
            sigs = [
                s for s in self._router.active_signals.values()
                if s.channel == "free"
            ]
            if not sigs:
                await self._telegram.send_message(chat_id, "No free signals today.")
            else:
                lines = ["🆓 Today's Free Picks:"]
                for s in sigs:
                    lines.append(
                        f"• {s.symbol} {s.direction.value} | "
                        f"Entry {s.entry:.4f} | Conf {s.confidence:.0f}%"
                    )
                await self._telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/signal_info":
            if len(parts) < 2:
                await self._telegram.send_message(
                    chat_id, "Usage: /signal\\_info <id>"
                )
            else:
                sid = parts[1]
                sig = self._router.active_signals.get(sid)
                if sig is None:
                    sig = next(
                        (s for s in self._signal_history if s.signal_id == sid),
                        None,
                    )
                if sig is None:
                    await self._telegram.send_message(
                        chat_id, f"❌ Signal `{sid}` not found."
                    )
                else:
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
                    await self._telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/last_update":
            scan_ms = self._telemetry._scan_latency_ms
            pairs = len(self._pair_mgr.pairs)
            active = len(self._router.active_signals)
            await self._telegram.send_message(
                chat_id,
                f"🕐 Last scan latency: {scan_ms:.0f}ms\n"
                f"Pairs: {pairs} | Active signals: {active}",
            )

        elif cmd == "/subscribe":
            await self._telegram.send_message(
                chat_id, "✅ Subscribed to premium signals."
            )

        elif cmd == "/unsubscribe":
            await self._telegram.send_message(chat_id, "✅ Unsubscribed.")

        elif cmd == "/signal_history":
            recent = self._signal_history[-10:]
            if not recent:
                await self._telegram.send_message(
                    chat_id, "No completed signals yet."
                )
            else:
                lines = ["📜 Signal History (last 10):"]
                for s in reversed(recent):
                    lines.append(
                        f"• {s.symbol} {s.direction.value} | "
                        f"{s.status} | PnL {s.pnl_pct:+.2f}%"
                    )
                await self._telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/gem_mode":
            if self._gem_scanner is None:
                await self._telegram.send_message(
                    chat_id, "❌ Gem scanner is not initialized."
                )
                return
            sub = parts[1].lower() if len(parts) >= 2 else "status"
            if sub == "on":
                self._gem_scanner.enable()
                await self._telegram.send_message(
                    chat_id,
                    "💎 Gem scanner ON — macro reversal signals will publish to 360\\_GEM channel",
                )
            elif sub == "off":
                self._gem_scanner.disable()
                await self._telegram.send_message(
                    chat_id,
                    "🔘 Gem scanner OFF — 360\\_GEM channel paused",
                )
            else:
                await self._telegram.send_message(
                    chat_id, self._gem_scanner.status_text()
                )

        elif cmd == "/gem_config":
            if self._gem_scanner is None:
                await self._telegram.send_message(
                    chat_id, "❌ Gem scanner is not initialized."
                )
                return
            if len(parts) < 3:
                await self._telegram.send_message(
                    chat_id, "Usage: /gem\\_config <key> <value>"
                )
            else:
                key = parts[1]
                cfg_value = parts[2]
                success, msg = self._gem_scanner.update_config(key, cfg_value)
                await self._telegram.send_message(chat_id, msg)

        elif cmd == "/signal_stats":
            if self._performance_tracker is None:
                await self._telegram.send_message(
                    chat_id, "ℹ️ Performance tracker is not enabled."
                )
            else:
                channel_arg = parts[1] if len(parts) >= 2 else None
                msg = self._performance_tracker.format_signal_quality_stats_message(
                    channel=channel_arg
                )
                await self._telegram.send_message(chat_id, msg)

        elif cmd == "/tp_stats":
            if self._performance_tracker is None:
                await self._telegram.send_message(
                    chat_id, "ℹ️ Performance tracker is not enabled."
                )
            else:
                channel_arg = parts[1] if len(parts) >= 2 else None
                msg = self._performance_tracker.format_tp_stats_message(
                    channel=channel_arg
                )
                await self._telegram.send_message(chat_id, msg)

        elif cmd == "/backtest":
            if len(parts) < 2:
                await self._telegram.send_message(
                    chat_id, "Usage: /backtest <symbol> [channel] [lookahead]"
                )
                return
            symbol = parts[1].upper()
            channel_filter: Optional[str] = parts[2] if len(parts) >= 3 else None
            lookahead = self._bt_lookahead
            if len(parts) >= 4:
                try:
                    lookahead = int(parts[3])
                except ValueError:
                    pass
            candles_by_tf = self._data_store.candles.get(symbol, {})
            if not candles_by_tf:
                await self._telegram.send_message(
                    chat_id,
                    f"❌ No candle data found for `{symbol}`. Make sure the symbol is tracked and data has been seeded.",
                )
                return
            await self._telegram.send_message(chat_id, "⏳ Running backtest…")
            try:
                bt = Backtester(
                    lookahead_candles=lookahead,
                    min_window=self._bt_min_window,
                    fee_pct=self._bt_fee_pct,
                    slippage_pct=self._bt_slippage_pct,
                )
                results = await asyncio.to_thread(
                    bt.run, candles_by_tf, symbol, channel_filter
                )
            except Exception as exc:
                log.error("Backtest error for %s: %s", symbol, exc)
                await self._telegram.send_message(
                    chat_id, f"❌ Backtest failed: {exc}"
                )
                return
            lines = [f"📊 Backtest Results — {symbol}\n"]
            for r in results:
                emoji = _CHANNEL_EMOJIS.get(r.channel, "📈")
                lines.append(f"{emoji} {r.channel}")
                lines.append(r.summary().replace(f"Backtest: {r.channel}\n", ""))
                lines.append("")
            msg = "\n".join(lines).strip()
            # Split messages longer than 4096 chars
            for chunk in _split_message(msg):
                await self._telegram.send_message(chat_id, chunk)

        elif cmd == "/backtest_all":
            channel_filter_all: Optional[str] = parts[1] if len(parts) >= 2 else None
            lookahead_all = self._bt_lookahead
            if len(parts) >= 3:
                try:
                    lookahead_all = int(parts[2])
                except ValueError:
                    pass
            all_symbols = list(self._data_store.candles.keys())
            if not all_symbols:
                await self._telegram.send_message(
                    chat_id, "❌ No candle data available. Wait for the data store to be seeded."
                )
                return
            # Limit to top 10 symbols by number of timeframes available
            all_symbols = sorted(
                all_symbols,
                key=lambda s: len(self._data_store.candles.get(s, {})),
                reverse=True,
            )[:10]
            await self._telegram.send_message(
                chat_id,
                f"⏳ Running backtest across {len(all_symbols)} tracked symbol(s)…",
            )
            bt_all = Backtester(
                lookahead_candles=lookahead_all,
                min_window=self._bt_min_window,
                fee_pct=self._bt_fee_pct,
                slippage_pct=self._bt_slippage_pct,
            )
            # Aggregate results per channel
            agg: Dict[str, Dict] = {}
            errors: List[str] = []
            for sym in all_symbols:
                ctf = self._data_store.candles.get(sym, {})
                if not ctf:
                    continue
                try:
                    sym_results = await asyncio.to_thread(
                        bt_all.run, ctf, sym, channel_filter_all
                    )
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
                    agg[r.channel]["max_drawdown"] = max(
                        agg[r.channel]["max_drawdown"], r.max_drawdown
                    )
            _CHANNEL_EMOJIS_ALL = {
                "360_SCALP": "⚡",
                "360_SWING": "🏛️",
                "360_RANGE": "⚖️",
                "360_THE_TAPE": "🐋",
            }
            lines_all = [
                f"📊 Backtest Summary — {len(all_symbols)} symbol(s)\n"
            ]
            for ch, data in agg.items():
                emoji = _CHANNEL_EMOJIS.get(ch, "📈")
                total = data["total_signals"]
                wins = data["wins"]
                losses = data["losses"]
                wr = (wins / total * 100) if total > 0 else 0.0
                lines_all.append(f"{emoji} {ch}")
                lines_all.append(
                    f"Signals: {total} | Wins: {wins} | Losses: {losses}"
                )
                lines_all.append(f"Win Rate: {wr:.1f}%")
                lines_all.append(f"Total PnL: {data['total_pnl']:+.2f}%")
                lines_all.append(
                    f"Max Drawdown: {data['max_drawdown']:.2f}%"
                )
                lines_all.append("")
            if errors:
                lines_all.append(f"⚠️ Failed symbols: {', '.join(errors)}")
            if not agg:
                lines_all.append("ℹ️ No results generated.")
            msg_all = "\n".join(lines_all).strip()
            for chunk in _split_message(msg_all):
                await self._telegram.send_message(chat_id, chunk)

        elif cmd == "/backtest_config":
            if len(parts) == 1:
                # Show current config
                config_msg = (
                    "🔧 Backtest Configuration\n"
                    f"Fee: {self._bt_fee_pct:.2f}%\n"
                    f"Slippage: {self._bt_slippage_pct:.2f}%\n"
                    f"Lookahead: {self._bt_lookahead} candles\n"
                    f"Min Window: {self._bt_min_window} candles"
                )
                await self._telegram.send_message(chat_id, config_msg)
            elif len(parts) >= 3:
                key = parts[1].lower()
                val_str = parts[2]
                _valid_keys = {"fee", "slippage", "lookahead", "min_window"}
                if key not in _valid_keys:
                    valid_keys_str = ", ".join(sorted(_valid_keys)).replace("_", "\\_")
                    await self._telegram.send_message(
                        chat_id,
                        f"❌ Unknown config key `{key}`. Valid keys: {valid_keys_str}",
                    )
                    return
                try:
                    parsed: Union[int, float]
                    if key in ("lookahead", "min_window"):
                        parsed = int(val_str)
                        if parsed < 1:
                            raise ValueError("must be >= 1")
                    else:
                        parsed = float(val_str)
                        if parsed < 0:
                            raise ValueError("must be >= 0")
                except ValueError as exc:
                    await self._telegram.send_message(
                        chat_id, f"❌ Invalid value: {exc}"
                    )
                    return
                if key == "fee":
                    self._bt_fee_pct = parsed
                    await self._telegram.send_message(
                        chat_id, f"✅ Backtest fee updated to {self._bt_fee_pct:.2f}%"
                    )
                elif key == "slippage":
                    self._bt_slippage_pct = parsed
                    await self._telegram.send_message(
                        chat_id, f"✅ Backtest slippage updated to {self._bt_slippage_pct:.2f}%"
                    )
                elif key == "lookahead":
                    self._bt_lookahead = int(parsed)
                    await self._telegram.send_message(
                        chat_id, f"✅ Backtest lookahead updated to {self._bt_lookahead} candles"
                    )
                elif key == "min_window":
                    self._bt_min_window = int(parsed)
                    await self._telegram.send_message(
                        chat_id, f"✅ Backtest min\\_window updated to {self._bt_min_window} candles"
                    )
            else:
                await self._telegram.send_message(
                    chat_id,
                    "Usage: /backtest\\_config [key] [value]\n"
                    "Keys: fee, slippage, lookahead, min\\_window",
                )

        elif cmd == "/portfolio":
            if self._paper_portfolio is None:
                await self._telegram.send_message(chat_id, "ℹ️ Paper portfolio is not enabled.")
            else:
                if len(parts) >= 2:
                    channel_arg = parts[1].upper()
                    if not channel_arg.startswith("360_"):
                        channel_arg = f"360_{channel_arg}"
                    msg = self._paper_portfolio.get_channel_detail(chat_id, channel_arg)
                else:
                    msg = self._paper_portfolio.get_portfolio_summary(chat_id)
                await self._telegram.send_message(chat_id, msg)

        elif cmd == "/reset_portfolio":
            if self._paper_portfolio is None:
                await self._telegram.send_message(chat_id, "ℹ️ Paper portfolio is not enabled.")
            else:
                channel_arg = parts[1].upper() if len(parts) >= 2 else None
                if channel_arg and not channel_arg.startswith("360_"):
                    channel_arg = f"360_{channel_arg}"
                msg = self._paper_portfolio.reset_portfolio(chat_id, channel_arg)
                await self._telegram.send_message(chat_id, msg)

        elif cmd == "/set_leverage":
            if self._paper_portfolio is None:
                await self._telegram.send_message(chat_id, "ℹ️ Paper portfolio is not enabled.")
            elif len(parts) < 3:
                await self._telegram.send_message(
                    chat_id, "Usage: /set\\_leverage <channel> <1-20>"
                )
            else:
                channel_arg = parts[1].upper()
                if not channel_arg.startswith("360_"):
                    channel_arg = f"360_{channel_arg}"
                try:
                    lev = int(parts[2])
                    msg = self._paper_portfolio.set_leverage(chat_id, channel_arg, lev)
                except ValueError:
                    msg = "❌ Leverage must be a number."
                await self._telegram.send_message(chat_id, msg)

        elif cmd == "/set_risk":
            if self._paper_portfolio is None:
                await self._telegram.send_message(chat_id, "ℹ️ Paper portfolio is not enabled.")
            elif len(parts) < 3:
                await self._telegram.send_message(
                    chat_id, "Usage: /set\\_risk <channel> <0.5-10>"
                )
            else:
                channel_arg = parts[1].upper()
                if not channel_arg.startswith("360_"):
                    channel_arg = f"360_{channel_arg}"
                try:
                    risk = float(parts[2])
                    msg = self._paper_portfolio.set_risk(chat_id, channel_arg, risk)
                except ValueError:
                    msg = "❌ Risk must be a number."
                await self._telegram.send_message(chat_id, msg)

        elif cmd == "/trade_history":
            if self._paper_portfolio is None:
                await self._telegram.send_message(chat_id, "ℹ️ Paper portfolio is not enabled.")
            else:
                channel_arg = parts[1].upper() if len(parts) >= 2 else None
                if channel_arg and not channel_arg.startswith("360_"):
                    channel_arg = f"360_{channel_arg}"
                msg = self._paper_portfolio.get_trade_history(chat_id, channel_arg)
                await self._telegram.send_message(chat_id, msg)

        elif cmd == "/leaderboard":
            if self._paper_portfolio is None:
                await self._telegram.send_message(chat_id, "ℹ️ Paper portfolio is not enabled.")
            else:
                sort_by = "roi" if len(parts) >= 2 and parts[1].lower() == "roi" else "pnl"
                msg = self._paper_portfolio.get_leaderboard(sort_by=sort_by)
                await self._telegram.send_message(chat_id, msg)

        else:
            await self._telegram.send_message(
                chat_id,
                "Available commands:\n"
                "*Admin:*\n"
                "/view\\_dashboard\n"
                "/update\\_pairs [spot/futures] [n]\n"
                "/subscribe\\_alerts\n"
                "/view\\_pairs [spot/futures]\n"
                "/force\\_scan\n"
                "/pause\\_channel <name>\n"
                "/resume\\_channel <name>\n"
                "/set\\_confidence\\_threshold <channel> <value>\n"
                "/set\\_free\\_channel\\_limit <n>\n"
                "/force\\_update\\_ai\n"
                "/view\\_active\\_signals\n"
                "/view\\_logs [lines]\n"
                "/engine\\_status\n"
                "/memory\\_usage\n"
                "/update\\_code\n"
                "/restart\\_engine\n"
                "/rollback\\_code <commit>\n"
                "/circuit\\_breaker\\_status\n"
                "/reset\\_circuit\\_breaker\n"
                "/stats [channel]\n"
                "/real\\_stats [channel]\n"
                "/reset\\_stats [channel]\n"
                "/select\\_mode [on|off|status]\n"
                "/select\\_config <key> <value>\n"
                "/backtest <symbol> [channel] [lookahead]\n"
                "/backtest\\_all [channel] [lookahead]\n"
                "/backtest\\_config [key] [value]\n\n"
                "*User:*\n"
                "/signals\n"
                "/free\\_signals\n"
                "/signal\\_info <id>\n"
                "/last\\_update\n"
                "/subscribe\n"
                "/unsubscribe\n"
                "/signal\\_history\n"
                "/signal\\_stats [channel]\n"
                "/tp\\_stats [channel]\n"
                "/portfolio [channel]\n"
                "/reset\\_portfolio [channel]\n"
                "/set\\_leverage <channel> <1-20>\n"
                "/set\\_risk <channel> <0.5-10>\n"
                "/trade\\_history [channel]\n"
                "/leaderboard [pnl|roi]",
            )
