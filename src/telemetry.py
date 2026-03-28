"""Telemetry – CPU, memory, WebSocket health, scan latency, API usage."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import psutil

from config import TELEMETRY_INTERVAL, NO_SIGNAL_ALERT_THRESHOLD_SECONDS, NO_SIGNAL_ALERT_COOLDOWN_SECONDS
from src.utils import get_logger

log = get_logger("telemetry")


@dataclass
class TelemetrySnapshot:
    cpu_pct: float = 0.0
    mem_mb: float = 0.0
    ws_connections: int = 0
    ws_healthy: bool = True
    active_signals: int = 0
    scan_latency_ms: float = 0.0
    api_calls_last_min: int = 0
    pairs_monitored: int = 0
    redis_connected: bool = False
    queue_size: int = 0
    signal_latency_ms: float = 0.0       # signal creation → Telegram delivery
    api_weight_used: int = 0             # Binance API weight used (last minute)
    ws_message_lag_ms: float = 0.0       # WebSocket kline message lag (ms)


class TelemetryCollector:
    """Periodically collects and logs system telemetry."""

    def __init__(self) -> None:
        self._running = False
        self._api_call_count: int = 0
        self._last_reset: float = time.monotonic()
        self.latest: TelemetrySnapshot = TelemetrySnapshot()
        self._ws_healthy: bool = True
        self._ws_connections: int = 0
        self._active_signals: int = 0
        self._pairs_monitored: int = 0
        self._scan_latency_ms: float = 0.0
        self._queue_size: int = 0
        self._redis_client: Optional[Any] = None
        # Enhanced metrics
        self._signal_latency_ms: float = 0.0
        self._api_weight_used: int = 0
        self._ws_message_lag_ms: float = 0.0
        # No-signal watchdog: track time of last dispatched signal and the
        # time of the last fired watchdog alert (to enforce cooldown).
        self._last_new_signal_time: float = time.monotonic()
        self._last_no_signal_alert_time: float = 0.0
        # Optional async callback for admin alerts (e.g. TelegramBot.send_admin_alert).
        self._admin_alert: Optional[Any] = None
        # Top-50 pairs filter (PR5): when non-empty, only log activity for
        # these pairs; all other pair-level log events are suppressed.
        self._top50_pairs: set = set()
        # When True, only active-trade events are logged (reduced verbosity).
        self._active_trades_only: bool = False

    def set_top50_pairs(self, pairs: List[Any]) -> None:
        """Set the top-50 futures pairs for reduced telemetry verbosity (PR5).

        When this list is non-empty and ``active_trades_only`` is ``True``,
        pair-level log output is restricted to pairs in this set.  Pass an
        empty list to clear the filter and restore full logging.

        Parameters
        ----------
        pairs:
            Iterable of symbol strings (e.g. ``["BTCUSDT", "ETHUSDT"]``).
        """
        self._top50_pairs = {s.upper() for s in pairs} if pairs else set()
        log.debug("Telemetry top-50 pairs set: %d symbols", len(self._top50_pairs))

    def set_active_trades_only(self, enabled: bool) -> None:
        """Toggle active-trades-only logging mode (PR5).

        When ``True``, verbose pair-level telemetry is suppressed for pairs
        that are not in the top-50 list and have no active signals.  CPU,
        memory, WS health, and scan-latency telemetry is always emitted.

        Parameters
        ----------
        enabled:
            ``True`` to reduce logging to active trades only.
        """
        self._active_trades_only = enabled
        log.info("Telemetry active_trades_only mode: %s", enabled)

    def is_top50_pair(self, symbol: str) -> bool:
        """Return True when *symbol* is in the configured top-50 list.

        Always returns ``True`` when no top-50 filter has been set (i.e.
        full-universe mode is active).
        """
        if not self._top50_pairs:
            return True
        return symbol.upper() in self._top50_pairs

    def record_api_call(self) -> None:
        self._api_call_count += 1

    def record_new_signal(self) -> None:
        """Record the time of the most recently dispatched signal.

        Call this whenever a signal is successfully enqueued so the no-signal
        watchdog can detect prolonged drought conditions.
        """
        self._last_new_signal_time = time.monotonic()

    def set_admin_alert_callback(self, callback: Any) -> None:
        """Register an async callable for admin alert delivery.

        The callback signature must be ``async def send(message: str) -> bool``.
        Typically wired to ``TelegramBot.send_admin_alert`` from ``main.py``.
        """
        self._admin_alert = callback

    def get_admin_alert_callback(self) -> Optional[Any]:
        """Return the registered admin alert callback, or ``None`` if unset."""
        return self._admin_alert

    @property
    def scan_latency_ms(self) -> float:
        """Most recently recorded scan cycle latency in milliseconds."""
        return self._scan_latency_ms

    def record_signal_latency(self, latency_ms: float) -> None:
        """Record time from signal creation to Telegram delivery (ms)."""
        self._signal_latency_ms = latency_ms

    def record_api_weight(self, weight_used: int) -> None:
        """Record the Binance API weight consumed in the last interval."""
        self._api_weight_used = weight_used

    def record_ws_message_lag(self, lag_ms: float) -> None:
        """Record the WebSocket kline message lag (expected vs received close time)."""
        self._ws_message_lag_ms = lag_ms

    def set_ws_health(self, healthy: bool, connections: int) -> None:
        self._ws_healthy = healthy
        self._ws_connections = connections

    def set_active_signals(self, count: int) -> None:
        self._active_signals = count

    def set_pairs_monitored(self, count: int) -> None:
        self._pairs_monitored = count

    def set_scan_latency(self, ms: float) -> None:
        self._scan_latency_ms = ms

    def set_queue_size(self, size: int) -> None:
        if isinstance(size, int):
            self._queue_size = size
        else:
            log.warning(
                "set_queue_size received non-int value (%s: %r) — ignoring",
                type(size).__name__,
                size,
            )

    def set_redis_client(self, client: Any) -> None:
        """Register the RedisClient instance for health reporting."""
        self._redis_client = client

    async def start(self) -> None:
        self._running = True
        log.info("Telemetry collector started (interval=%.0fs)", TELEMETRY_INTERVAL)
        while self._running:
            try:
                self._collect()
                log.info(
                    "CPU=%.1f%% | MEM=%.0fMB | WS=%d(ok=%s) | Signals=%d | "
                    "Pairs=%d | ScanLat=%.0fms | API/min=%d | Redis=%s",
                    self.latest.cpu_pct,
                    self.latest.mem_mb,
                    self.latest.ws_connections,
                    self.latest.ws_healthy,
                    self.latest.active_signals,
                    self.latest.pairs_monitored,
                    self.latest.scan_latency_ms,
                    self.latest.api_calls_last_min,
                    self.latest.redis_connected,
                )
                await self._check_no_signal_watchdog()
            except Exception as exc:
                log.debug("Telemetry error: %s", exc)
            await asyncio.sleep(TELEMETRY_INTERVAL)

    async def stop(self) -> None:
        self._running = False

    async def _check_no_signal_watchdog(self) -> None:
        """Fire an admin alert when no new signals have been seen for too long.

        The alert only fires when *both* conditions hold:
        1. No signal has been recorded for ``NO_SIGNAL_ALERT_THRESHOLD_SECONDS``.
        2. The WebSocket is currently unhealthy (``ws_healthy=False``).

        A cooldown of ``NO_SIGNAL_ALERT_COOLDOWN_SECONDS`` prevents repeated
        alerts for the same incident.  Runs inside the telemetry loop so it
        is subject to the same fail-open exception handling.
        """
        if self._admin_alert is None:
            return
        try:
            now = time.monotonic()
            drought_s = now - self._last_new_signal_time
            if drought_s < NO_SIGNAL_ALERT_THRESHOLD_SECONDS:
                return
            if self._ws_healthy:
                return
            if now - self._last_no_signal_alert_time < NO_SIGNAL_ALERT_COOLDOWN_SECONDS:
                return
            self._last_no_signal_alert_time = now
            minutes = int(drought_s / 60)
            msg = (
                f"⚠️ No new signals for {minutes} minutes. "
                "WebSocket unhealthy. Consider /restart."
            )
            log.warning("No-signal watchdog: %s", msg)
            await self._admin_alert(msg)
        except Exception as exc:
            log.debug("No-signal watchdog error: %s", exc)

    def _collect(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_reset
        api_rate = int(self._api_call_count / max(elapsed / 60, 0.01))
        self._api_call_count = 0
        self._last_reset = now

        proc = psutil.Process()
        redis_ok = bool(self._redis_client and self._redis_client.available)
        self.latest = TelemetrySnapshot(
            cpu_pct=proc.cpu_percent(interval=0),
            mem_mb=proc.memory_info().rss / (1024 * 1024),
            ws_connections=self._ws_connections,
            ws_healthy=self._ws_healthy,
            active_signals=self._active_signals,
            scan_latency_ms=self._scan_latency_ms,
            api_calls_last_min=api_rate,
            pairs_monitored=self._pairs_monitored,
            redis_connected=redis_ok,
            queue_size=self._queue_size,
            signal_latency_ms=self._signal_latency_ms,
            api_weight_used=self._api_weight_used,
            ws_message_lag_ms=self._ws_message_lag_ms,
        )

    def dashboard_text(self) -> str:
        s = self.latest
        return (
            "📊 *360-Crypto Dashboard*\n"
            f"CPU: {s.cpu_pct:.1f}% | RAM: {s.mem_mb:.0f} MB\n"
            f"WebSockets: {s.ws_connections} ({'✅' if s.ws_healthy else '❌'})\n"
            f"Active signals: {s.active_signals}\n"
            f"Pairs monitored: {s.pairs_monitored}\n"
            f"Scan latency: {s.scan_latency_ms:.0f} ms\n"
            f"Signal latency: {s.signal_latency_ms:.0f} ms\n"
            f"Queue size: {s.queue_size}\n"
            f"API calls/min: {s.api_calls_last_min}\n"
            f"API weight used: {s.api_weight_used}\n"
            f"WS message lag: {s.ws_message_lag_ms:.0f} ms\n"
            f"Redis: {'✅ connected' if s.redis_connected else '⚠️ in-memory'}"
        )
