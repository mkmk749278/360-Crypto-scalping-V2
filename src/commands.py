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
from typing import Any, Callable, Dict, List, Optional, Set

import psutil

from config import TELEGRAM_ADMIN_CHAT_ID
from src.logger import get_recent_logs
from src.utils import get_logger

log = get_logger("commands")

_TELEGRAM_LOG_MAX_CHARS: int = 3_500
_REPO_ROOT: Path = Path(__file__).parent.parent


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
        select_mode_filter: Optional[Any] = None,
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
        self._select_mode = select_mode_filter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _handle_command(self, text: str, chat_id: str) -> None:
        """Route an incoming Telegram command to the appropriate handler."""
        parts = text.strip().split()
        cmd = parts[0].lower()

        # Command aliases
        _aliases = {"/status": "/engine_status"}
        cmd = _aliases.get(cmd, cmd)

        is_admin = bool(TELEGRAM_ADMIN_CHAT_ID and chat_id == TELEGRAM_ADMIN_CHAT_ID)

        # --- Admin-only guard ---
        admin_cmds = {
            "/view_dashboard", "/update_pairs", "/subscribe_alerts",
            "/view_pairs", "/force_scan", "/pause_channel", "/resume_channel",
            "/set_confidence_threshold", "/engine_status", "/memory_usage",
            "/set_free_channel_limit", "/force_update_ai", "/view_active_signals",
            "/view_logs", "/update_code", "/restart_engine", "/rollback_code",
            "/circuit_breaker_status", "/reset_circuit_breaker",
            "/select_mode", "/select_config", "/reset_stats",
            "/real_stats", "/stats",
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

        elif cmd == "/select_mode":
            if self._select_mode is None:
                await self._telegram.send_message(
                    chat_id, "❌ Select mode filter is not initialized."
                )
                return
            sub = parts[1].lower() if len(parts) >= 2 else "status"
            if sub == "on":
                self._select_mode.enable()
                await self._telegram.send_message(
                    chat_id,
                    "🌹 Select mode ON — signals will also publish to 360\\_SELECT channel",
                )
            elif sub == "off":
                self._select_mode.disable()
                await self._telegram.send_message(
                    chat_id,
                    "🔘 Select mode OFF — 360\\_SELECT channel paused",
                )
            else:
                await self._telegram.send_message(
                    chat_id, self._select_mode.status_text()
                )

        elif cmd == "/select_config":
            if self._select_mode is None:
                await self._telegram.send_message(
                    chat_id, "❌ Select mode filter is not initialized."
                )
                return
            if len(parts) < 3:
                await self._telegram.send_message(
                    chat_id, "Usage: /select\\_config <key> <value>"
                )
            else:
                key = parts[1]
                cfg_value = parts[2]
                success, msg = self._select_mode.update_config(key, cfg_value)
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
                "/select\\_config <key> <value>\n\n"
                "*User:*\n"
                "/signals\n"
                "/free\\_signals\n"
                "/signal\\_info <id>\n"
                "/last\\_update\n"
                "/subscribe\n"
                "/unsubscribe\n"
                "/signal\\_history\n"
                "/signal\\_stats [channel]\n"
                "/tp\\_stats [channel]",
            )
