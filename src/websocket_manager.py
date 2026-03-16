"""WebSocket manager – multi-connection, heartbeat, auto-reconnect.

Supports up to ``WS_MAX_STREAMS_PER_CONN`` streams per connection,
exponential-backoff reconnect, auto-resubscribe, REST fallback,
and admin Telegram alerts.

Message buffering during reconnection gaps is handled by the upstream
:class:`src.signal_queue.SignalQueue` layer (Redis + asyncio.Queue fallback),
not at the WebSocket manager level.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional, Set

import aiohttp

from config import (
    BINANCE_FUTURES_REST_BASE,
    BINANCE_FUTURES_WS_BASE,
    BINANCE_REST_BASE,
    BINANCE_WS_BASE,
    WS_HEARTBEAT_INTERVAL,
    WS_MAX_STREAMS_PER_CONN,
    WS_RECONNECT_BASE_DELAY,
    WS_RECONNECT_MAX_DELAY,
)
from src.utils import get_logger

log = get_logger("ws_manager")

MessageHandler = Callable[[dict], Coroutine[Any, Any, None]]


@dataclass
class WSConnection:
    """Tracks one WebSocket connection and its streams."""
    ws: Optional[aiohttp.ClientWebSocketResponse] = None
    streams: List[str] = field(default_factory=list)
    last_pong: float = 0.0
    reconnect_attempts: int = 0
    task: Optional[asyncio.Task] = None
    degraded: bool = False


class WebSocketManager:
    """Manages multiple Binance WebSocket connections with resilience."""

    def __init__(self, on_message: MessageHandler, market: str = "spot", admin_alert_callback=None) -> None:
        self._on_message = on_message
        self._market = market
        self._base_url = BINANCE_WS_BASE if market == "spot" else BINANCE_FUTURES_WS_BASE
        self._rest_base_url = BINANCE_REST_BASE if market == "spot" else BINANCE_FUTURES_REST_BASE
        self._connections: List[WSConnection] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._subscribed_streams: Set[str] = set()
        self._rest_fallback_active: bool = False
        self._critical_pairs: Set[str] = set()
        self._fallback_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._admin_alert = admin_alert_callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, streams: List[str]) -> None:
        """Subscribe to *streams* distributed across connections."""
        self._running = True
        self._connections = []
        self._subscribed_streams = set()
        self._rest_fallback_active = False
        self._session = aiohttp.ClientSession()

        # Chunk streams across connections
        for i in range(0, len(streams), WS_MAX_STREAMS_PER_CONN):
            chunk = streams[i: i + WS_MAX_STREAMS_PER_CONN]
            conn = WSConnection(streams=chunk)
            self._connections.append(conn)
            conn.task = asyncio.create_task(self._run_connection(conn))
        log.info(
            "WS manager started: {} streams across {} connections ({})",
            len(streams), len(self._connections), self._market,
        )
        self._watchdog_task = asyncio.create_task(self._health_watchdog())

    async def stop(self) -> None:
        self._running = False
        self._rest_fallback_active = False
        tasks: List[asyncio.Task] = []
        if self._fallback_task and not self._fallback_task.done():
            self._fallback_task.cancel()
            tasks.append(self._fallback_task)
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            tasks.append(self._watchdog_task)
        for conn in self._connections:
            conn.degraded = False
            if conn.task and not conn.task.done():
                conn.task.cancel()
                tasks.append(conn.task)
            if conn.ws and not conn.ws.closed:
                await conn.ws.close()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._fallback_task = None
        self._watchdog_task = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._connections = []
        log.info("WS manager stopped ({})", self._market)

    # ------------------------------------------------------------------
    # REST fallback for critical pairs
    # ------------------------------------------------------------------

    def set_critical_pairs(self, pairs: List[str]) -> None:
        """Define which symbols receive REST fallback during WS outages."""
        self._critical_pairs = set(pairs)
        log.info("Critical pairs set ({}): {}", len(self._critical_pairs), pairs)

    async def _rest_fallback_loop(self) -> None:
        """Poll REST klines for critical pairs while WS is down."""
        assert self._session is not None
        if self._market == "futures":
            url_tpl = f"{self._rest_base_url}/fapi/v1/klines?symbol={{symbol}}&interval=1m&limit=1"
        else:
            url_tpl = f"{self._rest_base_url}/api/v3/klines?symbol={{symbol}}&interval=1m&limit=1"

        log.info("REST fallback loop started for {} critical pairs", len(self._critical_pairs))
        try:
            while self._running and self._rest_fallback_active:
                for symbol in list(self._critical_pairs):
                    try:
                        url = url_tpl.format(symbol=symbol)
                        async with self._session.get(
                            url, timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status != 200:
                                log.debug("REST fallback {} status {}", symbol, resp.status)
                                continue
                            raw = await resp.json()
                        if not raw:
                            continue
                        k = raw[0]
                        msg: dict = {
                            "e": "kline",
                            "s": symbol,
                            "k": {
                                "i": "1m",
                                "o": str(k[1]),
                                "h": str(k[2]),
                                "l": str(k[3]),
                                "c": str(k[4]),
                                "v": str(k[5]),
                                "x": True,
                            },
                        }
                        await self._on_message(msg)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        log.debug("REST fallback error for {}: {}", symbol, exc)
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        log.info("REST fallback loop stopped")

    def _start_rest_fallback(self) -> None:
        """Activate REST fallback if critical pairs are configured."""
        if not self._critical_pairs:
            return
        if self._rest_fallback_active:
            return
        self._rest_fallback_active = True
        self._fallback_task = asyncio.create_task(self._rest_fallback_loop())
        if self._admin_alert:
            asyncio.create_task(
                self._admin_alert(
                    f"⚠️ REST fallback activated for {self._market} critical pairs."
                )
            )

    def _stop_rest_fallback(self) -> None:
        """Deactivate REST fallback once WS reconnects."""
        if not self._rest_fallback_active:
            return
        self._rest_fallback_active = False
        if self._fallback_task and not self._fallback_task.done():
            self._fallback_task.cancel()
        self._fallback_task = None

    def _connection_uses_fallback(self, conn: WSConnection) -> bool:
        return any(
            stream.split("@", 1)[0].upper() in self._critical_pairs
            for stream in conn.streams
        )

    def _sync_rest_fallback_state(self) -> None:
        should_run = any(
            conn.degraded and self._connection_uses_fallback(conn)
            for conn in self._connections
        )
        if should_run:
            self._start_rest_fallback()
        else:
            self._stop_rest_fallback()

    def _set_connection_degraded(self, conn: WSConnection, degraded: bool) -> None:
        if conn.degraded == degraded:
            return
        conn.degraded = degraded
        self._sync_rest_fallback_state()

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def _run_connection(self, conn: WSConnection) -> None:
        while self._running:
            try:
                await self._connect(conn)
                self._set_connection_degraded(conn, False)
                await self._listen(conn)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning("WS connection error: {}", exc)
            if self._running:
                self._set_connection_degraded(conn, True)
                if self._admin_alert:
                    asyncio.create_task(
                        self._admin_alert(
                            f"⚠️ WebSocket connection lost ({self._market}). Reconnecting…"
                        )
                    )
                delay = min(
                    WS_RECONNECT_BASE_DELAY * (2 ** conn.reconnect_attempts),
                    WS_RECONNECT_MAX_DELAY,
                )
                conn.reconnect_attempts += 1
                log.info(
                    "Reconnecting in {:.1f}s (attempt {}) …",
                    delay,
                    conn.reconnect_attempts,
                )
                await asyncio.sleep(delay)

    async def _connect(self, conn: WSConnection) -> None:
        assert self._session is not None
        stream_path = "/".join(conn.streams)
        url = f"{self._base_url}/{stream_path}"
        conn.ws = await self._session.ws_connect(url, heartbeat=WS_HEARTBEAT_INTERVAL)
        conn.last_pong = time.monotonic()
        conn.reconnect_attempts = 0
        conn.degraded = False
        self._subscribed_streams.update(conn.streams)
        log.info("Connected WS: {} streams", len(conn.streams))

    async def _listen(self, conn: WSConnection) -> None:
        assert conn.ws is not None
        async for msg in conn.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                # Any incoming data message proves the connection is alive;
                # update last_pong so is_healthy reflects real liveness.
                conn.last_pong = time.monotonic()
                try:
                    data = json.loads(msg.data)
                    await self._on_message(data)
                except Exception as exc:
                    log.debug("Message parse error: {}", exc)
            elif msg.type == aiohttp.WSMsgType.PONG:
                conn.last_pong = time.monotonic()
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                log.warning("WS closed/error, will reconnect")
                break

    # ------------------------------------------------------------------
    # Health watchdog
    # ------------------------------------------------------------------

    async def _health_watchdog(self) -> None:
        """Periodically force-close stale connections so _run_connection reconnects."""
        try:
            while self._running:
                await asyncio.sleep(WS_HEARTBEAT_INTERVAL)
                now = time.monotonic()
                for conn in self._connections:
                    if conn.ws and not conn.ws.closed:
                        if (now - conn.last_pong) >= WS_HEARTBEAT_INTERVAL * 3:
                            log.warning(
                                "Watchdog: stale WS connection ({:.0f}s since last data) — force-closing to trigger reconnect",
                                now - conn.last_pong,
                            )
                            await conn.ws.close()
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Dynamic subscription helpers
    # ------------------------------------------------------------------

    def build_kline_stream(self, symbol: str, interval: str) -> str:
        return f"{symbol.lower()}@kline_{interval}"

    def build_trade_stream(self, symbol: str) -> str:
        return f"{symbol.lower()}@trade"

    def build_depth_stream(self, symbol: str, level: int = 5) -> str:
        return f"{symbol.lower()}@depth{level}@100ms"

    @property
    def stream_count(self) -> int:
        return sum(len(c.streams) for c in self._connections)

    @property
    def is_healthy(self) -> bool:
        now = time.monotonic()
        open_connections = [
            c for c in self._connections if c.ws is not None and not c.ws.closed
        ]
        if not open_connections or len(open_connections) != len(self._connections):
            return False
        return all(
            (now - c.last_pong) < WS_HEARTBEAT_INTERVAL * 3
            for c in open_connections
        )
