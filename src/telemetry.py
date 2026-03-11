"""Telemetry – CPU, memory, WebSocket health, scan latency, API usage."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import psutil

from config import TELEMETRY_INTERVAL
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

    def record_api_call(self) -> None:
        self._api_call_count += 1

    def set_ws_health(self, healthy: bool, connections: int) -> None:
        self._ws_healthy = healthy
        self._ws_connections = connections

    def set_active_signals(self, count: int) -> None:
        self._active_signals = count

    def set_pairs_monitored(self, count: int) -> None:
        self._pairs_monitored = count

    def set_scan_latency(self, ms: float) -> None:
        self._scan_latency_ms = ms

    async def start(self) -> None:
        self._running = True
        log.info("Telemetry collector started (interval=%.0fs)", TELEMETRY_INTERVAL)
        while self._running:
            try:
                self._collect()
                log.info(
                    "CPU=%.1f%% | MEM=%.0fMB | WS=%d(ok=%s) | Signals=%d | "
                    "Pairs=%d | ScanLat=%.0fms | API/min=%d",
                    self.latest.cpu_pct,
                    self.latest.mem_mb,
                    self.latest.ws_connections,
                    self.latest.ws_healthy,
                    self.latest.active_signals,
                    self.latest.pairs_monitored,
                    self.latest.scan_latency_ms,
                    self.latest.api_calls_last_min,
                )
            except Exception as exc:
                log.debug("Telemetry error: %s", exc)
            await asyncio.sleep(TELEMETRY_INTERVAL)

    async def stop(self) -> None:
        self._running = False

    def _collect(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_reset
        api_rate = int(self._api_call_count / max(elapsed / 60, 0.01))
        self._api_call_count = 0
        self._last_reset = now

        proc = psutil.Process()
        self.latest = TelemetrySnapshot(
            cpu_pct=proc.cpu_percent(interval=0),
            mem_mb=proc.memory_info().rss / (1024 * 1024),
            ws_connections=self._ws_connections,
            ws_healthy=self._ws_healthy,
            active_signals=self._active_signals,
            scan_latency_ms=self._scan_latency_ms,
            api_calls_last_min=api_rate,
            pairs_monitored=self._pairs_monitored,
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
            f"API calls/min: {s.api_calls_last_min}"
        )
