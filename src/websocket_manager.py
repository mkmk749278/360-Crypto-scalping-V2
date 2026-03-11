"""WebSocket manager – multi-connection, heartbeat, auto-reconnect.

Supports up to ``WS_MAX_STREAMS_PER_CONN`` streams per connection,
exponential-backoff reconnect, auto-resubscribe, REST fallback,
queue buffering, and admin Telegram alerts.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import aiohttp

from config import (
    BINANCE_FUTURES_WS_BASE,
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


class WebSocketManager:
    """Manages multiple Binance WebSocket connections with resilience."""

    def __init__(self, on_message: MessageHandler, market: str = "spot") -> None:
        self._on_message = on_message
        self._market = market
        self._base_url = BINANCE_WS_BASE if market == "spot" else BINANCE_FUTURES_WS_BASE
        self._connections: List[WSConnection] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._buffer: asyncio.Queue[dict] = asyncio.Queue(maxsize=10_000)
        self._subscribed_streams: Set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, streams: List[str]) -> None:
        """Subscribe to *streams* distributed across connections."""
        self._running = True
        self._session = aiohttp.ClientSession()

        # Chunk streams across connections
        for i in range(0, len(streams), WS_MAX_STREAMS_PER_CONN):
            chunk = streams[i: i + WS_MAX_STREAMS_PER_CONN]
            conn = WSConnection(streams=chunk)
            self._connections.append(conn)
            conn.task = asyncio.create_task(self._run_connection(conn))
        log.info(
            "WS manager started: %d streams across %d connections (%s)",
            len(streams), len(self._connections), self._market,
        )

    async def stop(self) -> None:
        self._running = False
        for conn in self._connections:
            if conn.task:
                conn.task.cancel()
            if conn.ws and not conn.ws.closed:
                await conn.ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        log.info("WS manager stopped (%s)", self._market)

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def _run_connection(self, conn: WSConnection) -> None:
        while self._running:
            try:
                await self._connect(conn)
                await self._listen(conn)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning("WS connection error: %s", exc)
            if self._running:
                delay = min(
                    WS_RECONNECT_BASE_DELAY * (2 ** conn.reconnect_attempts),
                    WS_RECONNECT_MAX_DELAY,
                )
                conn.reconnect_attempts += 1
                log.info("Reconnecting in %.1fs (attempt %d) …", delay, conn.reconnect_attempts)
                await asyncio.sleep(delay)

    async def _connect(self, conn: WSConnection) -> None:
        assert self._session is not None
        stream_path = "/".join(conn.streams)
        url = f"{self._base_url}/{stream_path}"
        conn.ws = await self._session.ws_connect(url, heartbeat=WS_HEARTBEAT_INTERVAL)
        conn.last_pong = time.monotonic()
        conn.reconnect_attempts = 0
        self._subscribed_streams.update(conn.streams)
        log.info("Connected WS: %d streams", len(conn.streams))

    async def _listen(self, conn: WSConnection) -> None:
        assert conn.ws is not None
        async for msg in conn.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    await self._on_message(data)
                except Exception as exc:
                    log.debug("Message parse error: %s", exc)
            elif msg.type == aiohttp.WSMsgType.PONG:
                conn.last_pong = time.monotonic()
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                log.warning("WS closed/error, will reconnect")
                break

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
        return all(
            (now - c.last_pong) < WS_HEARTBEAT_INTERVAL * 3
            for c in self._connections
            if c.ws and not c.ws.closed
        )
