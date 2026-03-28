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
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import aiohttp

from config import (
    BINANCE_FUTURES_REST_BASE,
    BINANCE_FUTURES_WS_BASE,
    BINANCE_REST_BASE,
    BINANCE_WS_BASE,
    WS_ALERT_COOLDOWN,
    WS_FALLBACK_BULK_LIMIT,
    WS_FALLBACK_POLL_INTERVALS,
    WS_FALLBACK_TIMEFRAMES,
    WS_HEALTH_CHECK_INTERVAL,
    WS_HEARTBEAT_INTERVAL,
    WS_HEARTBEAT_INTERVAL_FUTURES,
    WS_MAX_STREAMS_PER_CONN,
    WS_MIN_MESSAGE_RATE,
    WS_PING_TIMEOUT_MS,
    WS_RECONNECT_BASE_DELAY,
    WS_RECONNECT_FAIL_ALERT_THRESHOLD,
    WS_RECONNECT_MAX_DELAY,
    WS_SESSION_RECYCLE_ATTEMPTS,
    WS_STALENESS_MULTIPLIER,
    WS_STALENESS_MULTIPLIER_FUTURES,
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
    # Ping/pong latency tracking
    last_ping_time: float = 0.0   # monotonic time when last manual ping was sent
    ping_latency_ms: float = 0.0  # RTT (ms) of the most recent completed ping/pong
    # Health monitoring fields (used by _health_check_loop)
    health_check_ts: float = 0.0   # last time a health check snapshot was taken
    health_msg_count: int = 0      # messages received since last health check


class WebSocketManager:
    """Manages multiple Binance WebSocket connections with resilience."""

    def __init__(self, on_message: MessageHandler, market: str = "spot", admin_alert_callback=None, data_store=None, label: str | None = None, staleness_multiplier: float | None = None) -> None:
        self._on_message = on_message
        self._market = market
        self._label = label or market
        self._base_url = BINANCE_WS_BASE if market == "spot" else BINANCE_FUTURES_WS_BASE
        self._rest_base_url = BINANCE_REST_BASE if market == "spot" else BINANCE_FUTURES_REST_BASE
        self._heartbeat_interval = WS_HEARTBEAT_INTERVAL_FUTURES if market == "futures" else WS_HEARTBEAT_INTERVAL
        default_multiplier = WS_STALENESS_MULTIPLIER_FUTURES if market == "futures" else WS_STALENESS_MULTIPLIER
        self._staleness_multiplier = staleness_multiplier if staleness_multiplier is not None else default_multiplier
        self._connections: List[WSConnection] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._subscribed_streams: Set[str] = set()
        self._rest_fallback_active: bool = False
        self._critical_pairs: Set[str] = set()
        self._fallback_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._admin_alert = admin_alert_callback
        self._last_alert_time: float = 0.0
        self._total_drops: int = 0
        self._data_store = data_store
        self._ws_rest_fallback_count: int = 0
        self._ws_reconnection_count: int = 0
        self._connection_message_rates: Dict[int, float] = {}  # conn_index -> msgs/min

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, streams: List[str]) -> None:
        """Subscribe to *streams* distributed across connections."""
        self._running = True
        self._connections = []
        self._subscribed_streams = set()
        self._rest_fallback_active = False
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(keepalive_timeout=30)
        )
        for i in range(0, len(streams), WS_MAX_STREAMS_PER_CONN):
            chunk = streams[i: i + WS_MAX_STREAMS_PER_CONN]
            conn = WSConnection(streams=chunk)
            self._connections.append(conn)
            conn.task = asyncio.create_task(self._run_connection(conn))
        log.info(
            "WS manager started: {} streams across {} connections ({})",
            len(streams), len(self._connections), self._label,
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
        log.info("WS manager stopped ({})", self._label)

    # ------------------------------------------------------------------
    # REST fallback for critical pairs
    # ------------------------------------------------------------------

    def set_critical_pairs(self, pairs: List[str]) -> None:
        """Define which symbols receive REST fallback during WS outages."""
        self._critical_pairs = set(pairs)
        log.info("Critical pairs set ({}): {}", len(self._critical_pairs), pairs)

    def auto_populate_critical_pairs(self, symbols: List[str]) -> None:
        """Populate *_critical_pairs* from the provided symbol list.

        Call this after :meth:`start` with the top-N pairs by volume so that
        REST fallback activates automatically when WS is fully degraded even
        if :meth:`set_critical_pairs` was never called explicitly.

        Existing critical pairs are *replaced* only when the incoming list is
        non-empty; an empty list is a no-op so bootstrap ordering is flexible.
        """
        if not symbols:
            return
        self._critical_pairs = set(s.upper() for s in symbols)
        log.info(
            "Critical pairs auto-populated ({}): {}", len(self._critical_pairs), list(self._critical_pairs)[:20]
        )

    async def update_streams_for_top50(
        self,
        symbols: List[str],
        intervals: Optional[List[str]] = None,
    ) -> None:
        """Dynamically update WS streams to cover *symbols* (top-50) only.

        Computes the symmetric difference between the current subscribed kline
        streams and the desired set, then restarts the manager with the new
        stream list.  This is a full restart (stop → start) to keep the
        implementation simple; the caller should avoid calling this too
        frequently.

        This method is the entry-point for PR4 dynamic top-50 WS management:
        whenever the top-50 futures list changes, the main engine calls this
        method to shed streams for pairs that dropped out of the top-50 and
        open new streams for pairs that entered.

        Parameters
        ----------
        symbols:
            Current top-50 futures symbols (upper-case, e.g. ``"BTCUSDT"``).
        intervals:
            Kline intervals to stream for each symbol.  Defaults to
            ``["1m", "5m", "15m"]``.
        """
        if not self._running:
            log.debug("update_streams_for_top50: manager not running — skipping")
            return

        intervals = intervals or ["1m", "5m", "15m"]
        desired_streams: Set[str] = set()
        for sym in symbols:
            for tf in intervals:
                desired_streams.add(self.build_kline_stream(sym.lower(), tf))

        current_streams = set(self._subscribed_streams)
        if desired_streams == current_streams:
            log.debug("update_streams_for_top50: no stream changes required")
            return

        added = desired_streams - current_streams
        removed = current_streams - desired_streams
        log.info(
            "update_streams_for_top50: +%d / -%d streams for %d symbols (%s)",
            len(added), len(removed), len(symbols), self._label,
        )

        # Restart with the new stream list.  auto_populate_critical_pairs is
        # re-applied so the REST fallback remains aligned to the new top-50.
        await self.stop()
        await self.start(sorted(desired_streams))
        self.auto_populate_critical_pairs(symbols)

    async def _rest_fallback_loop(self) -> None:
        """Poll REST klines for critical pairs while WS is down."""
        assert self._session is not None
        if self._market == "futures":
            url_tpl = f"{self._rest_base_url}/fapi/v1/klines?symbol={{symbol}}&interval={{interval}}&limit=1"
        else:
            url_tpl = f"{self._rest_base_url}/api/v3/klines?symbol={{symbol}}&interval={{interval}}&limit=1"

        log.info("REST fallback loop started for {} critical pairs", len(self._critical_pairs))

        # One-time bulk backfill to warm indicator pipelines (200 candles per
        # symbol × timeframe) so scanners can produce signals immediately after
        # a WS outage rather than waiting for candle-by-candle accumulation.
        if self._data_store is not None:
            for symbol in list(self._critical_pairs):
                for interval in WS_FALLBACK_TIMEFRAMES:
                    try:
                        await self._data_store.fetch_and_store_fallback(
                            symbol, interval=interval, limit=WS_FALLBACK_BULK_LIMIT, market=self._market
                        )
                        log.info(
                            "REST fallback: bulk-seeded {} {} candles for {}",
                            WS_FALLBACK_BULK_LIMIT, interval, symbol,
                        )
                    except Exception as exc:
                        log.warning(
                            "REST fallback bulk seed failed for {} {}: {}", symbol, interval, exc
                        )
                    await asyncio.sleep(0.2)  # ~5 req/s well within Binance's 1200 req/min weight limit

        try:
            while self._running and self._rest_fallback_active:
                for symbol in list(self._critical_pairs):
                    for interval in WS_FALLBACK_POLL_INTERVALS:
                        try:
                            url = url_tpl.format(symbol=symbol, interval=interval)
                            async with self._session.get(
                                url, timeout=aiohttp.ClientTimeout(total=10),
                            ) as resp:
                                if resp.status != 200:
                                    log.debug("REST fallback {} {} status {}", symbol, interval, resp.status)
                                    continue
                                raw = await resp.json()
                            if not raw:
                                continue
                            k = raw[0]
                            msg: dict = {
                                "e": "kline",
                                "s": symbol,
                                "k": {
                                    "i": interval,
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
                            log.debug("REST fallback error for {} {}: {}", symbol, interval, exc)
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
                    f"⚠️ REST fallback activated for {self._label} critical pairs."
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
        # If all connections are degraded but _critical_pairs was never
        # configured, auto-extract symbols from subscribed kline streams so
        # that the REST fallback activates rather than remaining a no-op.
        all_degraded = self._connections and all(c.degraded for c in self._connections)
        if all_degraded and not self._critical_pairs and self._subscribed_streams:
            extracted: List[str] = []
            for stream in self._subscribed_streams:
                if "@kline_" in stream:
                    symbol = stream.split("@", 1)[0].upper()
                    extracted.append(symbol)
            if extracted:
                # Deduplicate while preserving insertion order for logging
                seen: set = set()
                unique: List[str] = []
                for s in extracted:
                    if s not in seen:
                        seen.add(s)
                        unique.append(s)
                self._critical_pairs = set(unique[:20])
                log.warning(
                    "WS fully degraded with empty critical_pairs — "
                    "auto-extracted {} symbols from subscribed streams",
                    len(self._critical_pairs),
                )

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
                self._total_drops += 1
                if self._admin_alert:
                    now = time.monotonic()
                    if now - self._last_alert_time > WS_ALERT_COOLDOWN:
                        # Update timestamp *before* creating the task so that
                        # other connections resuming in the same event-loop tick
                        # see the updated value and skip their own alert.
                        self._last_alert_time = now
                        asyncio.create_task(
                            self._admin_alert(
                                f"⚠️ WebSocket connection lost ({self._label}, "
                                f"attempt {conn.reconnect_attempts + 1}, "
                                f"total drops: {self._total_drops}). Reconnecting…"
                            )
                        )
                delay = min(
                    WS_RECONNECT_BASE_DELAY * (2 ** conn.reconnect_attempts),
                    WS_RECONNECT_MAX_DELAY,
                )
                # Add ±25% jitter to prevent thundering-herd reconnects when
                # multiple connections drop simultaneously.
                jitter = delay * random.uniform(-0.25, 0.25)
                actual_delay = max(0.5, delay + jitter)
                conn.reconnect_attempts += 1
                log.info(
                    "Reconnecting in {:.1f}s (attempt {}) …",
                    actual_delay,
                    conn.reconnect_attempts,
                )
                # Recycle the aiohttp session periodically to clear stale TCP
                # connection pools and DNS caches after extended outages.
                if conn.reconnect_attempts % WS_SESSION_RECYCLE_ATTEMPTS == 0:
                    log.warning(
                        "Recycling HTTP session after {} consecutive failures ({})",
                        conn.reconnect_attempts, self._label,
                    )
                    await self._recreate_session(force_dns_reresolution=True)
                # Escalation alert: after many consecutive failures the issue
                # may require manual intervention (e.g. IP ban, firewall change).
                if (
                    conn.reconnect_attempts == WS_RECONNECT_FAIL_ALERT_THRESHOLD
                    and self._admin_alert
                ):
                    asyncio.create_task(
                        self._admin_alert(
                            f"🚨 WebSocket unable to reconnect after "
                            f"{WS_RECONNECT_FAIL_ALERT_THRESHOLD} attempts "
                            f"({self._label}) — manual intervention may be needed."
                        )
                    )
                await asyncio.sleep(actual_delay)

    async def _connect(self, conn: WSConnection) -> None:
        assert self._session is not None
        stream_path = "/".join(conn.streams)
        url = f"{self._base_url}/{stream_path}"
        conn.ws = await self._session.ws_connect(url, heartbeat=self._heartbeat_interval)
        conn.last_pong = time.monotonic()
        conn.reconnect_attempts = 0
        conn.degraded = False
        conn.last_ping_time = 0.0
        conn.ping_latency_ms = 0.0
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
            elif msg.type in (aiohttp.WSMsgType.PING, aiohttp.WSMsgType.PONG):
                # Binance sends PING frames every ~3 minutes; aiohttp auto-replies
                # with PONG.  Treat both as proof that the connection is alive.
                now_mono = time.monotonic()
                conn.last_pong = now_mono
                # When a PONG arrives, measure RTT against the last manual ping
                # sent by the watchdog and reset last_ping_time so the watchdog
                # does not report a false timeout on the next iteration.
                if msg.type == aiohttp.WSMsgType.PONG and conn.last_ping_time > 0:
                    conn.ping_latency_ms = (now_mono - conn.last_ping_time) * 1000
                    conn.last_ping_time = 0.0
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                log.warning("WS closed/error, will reconnect")
                break

    # ------------------------------------------------------------------
    # Health watchdog
    # ------------------------------------------------------------------

    async def _health_watchdog(self) -> None:
        """Periodically force-close stale or lagging connections so _run_connection reconnects.

        Three checks run on every shard at each tick:

        1. **Staleness** - no data at all for ``heartbeat_interval x staleness_multiplier``
           seconds -> dead connection, force-close.
        2. **Ping timeout** - a manual ping was sent on the previous tick and no PONG has
           come back within ``WS_PING_TIMEOUT_MS`` ms -> connection is lagging, force-close.
        3. **High latency** - the most recently completed ping/pong RTT exceeds
           ``WS_PING_TIMEOUT_MS`` ms -> connection is degraded, force-close.

        After the checks a fresh manual ping is sent so the next tick can measure
        latency.  aiohttp's built-in ``heartbeat=`` keepalive continues to run
        alongside; these manual pings are additional probes for latency measurement.
        """
        try:
            while self._running:
                await asyncio.sleep(self._heartbeat_interval)
                now = time.monotonic()
                for conn in self._connections:
                    if not (conn.ws and not conn.ws.closed):
                        continue

                    # 1. Staleness check (unchanged from original behaviour)
                    # Use max(1.0, ...) to guarantee a minimum 1-second threshold
                    # regardless of the configured heartbeat interval.
                    stale_threshold = max(1.0, self._heartbeat_interval * self._staleness_multiplier)
                    if (now - conn.last_pong) >= stale_threshold:
                        log.warning(
                            "Watchdog: stale WS connection ({:.0f}s since last data) — force-closing to trigger reconnect",
                            now - conn.last_pong,
                        )
                        conn.last_ping_time = 0.0
                        await conn.ws.close()
                        continue

                    # 2. Ping timeout: pong not received within WS_PING_TIMEOUT_MS
                    if conn.last_ping_time > 0:
                        overdue_ms = (now - conn.last_ping_time) * 1000
                        if overdue_ms > WS_PING_TIMEOUT_MS:
                            log.warning(
                                "Watchdog: ping timeout ({:.0f}ms > {}ms) — "
                                "force-closing shard to trigger reconnect",
                                overdue_ms,
                                WS_PING_TIMEOUT_MS,
                            )
                            conn.last_ping_time = 0.0
                            conn.ping_latency_ms = 0.0
                            await conn.ws.close()
                            continue

                    # 3. High latency: previous ping RTT exceeded threshold
                    if conn.ping_latency_ms > WS_PING_TIMEOUT_MS:
                        log.warning(
                            "Watchdog: high ping latency ({:.0f}ms > {}ms) — "
                            "force-closing shard to trigger reconnect",
                            conn.ping_latency_ms,
                            WS_PING_TIMEOUT_MS,
                        )
                        conn.ping_latency_ms = 0.0
                        conn.last_ping_time = 0.0
                        await conn.ws.close()
                        continue

                    # Send a fresh manual ping to probe latency on the next tick
                    try:
                        conn.last_ping_time = time.monotonic()
                        await conn.ws.ping()
                    except Exception as exc:
                        log.debug("Watchdog ping send error ({}): {}", self._label, exc)
                        conn.last_ping_time = 0.0
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Dynamic subscription helpers
    # ------------------------------------------------------------------

    async def _recreate_session(self, force_dns_reresolution: bool = False) -> None:
        """Close and recreate the aiohttp session to clear stale connections.

        Parameters
        ----------
        force_dns_reresolution:
            When ``True``, create the TCP connector with ``force_close=True``
            so that no connections are reused from the pool.  This forces the
            OS to resolve DNS again, which helps when the Binance endpoint IP
            changes or when a previous outage was caused by a stale ARP/DNS
            cache.
        """
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception as exc:
                log.debug("Error closing stale session ({}): {}", self._label, exc)
        connector = aiohttp.TCPConnector(
            keepalive_timeout=30,
            force_close=force_dns_reresolution,
        )
        self._session = aiohttp.ClientSession(connector=connector)

    def build_kline_stream(self, symbol: str, interval: str) -> str:
        return f"{symbol.lower()}@kline_{interval}"

    def build_trade_stream(self, symbol: str) -> str:
        return f"{symbol.lower()}@trade"

    def build_depth_stream(self, symbol: str, level: int = 5) -> str:
        return f"{symbol.lower()}@depth{level}@100ms"

    def build_force_order_stream(self, symbol: str) -> str:
        """Return the Binance Futures liquidation stream name for *symbol*.

        Subscribing to this stream delivers ``forceOrder`` events whenever a
        position in *symbol* is force-liquidated.  The event payload includes
        the order side (``"BUY"`` = short liq'd, ``"SELL"`` = long liq'd),
        quantity, and average fill price.
        """
        return f"{symbol.lower()}@forceOrder"

    @property
    def stream_count(self) -> int:
        return sum(len(c.streams) for c in self._connections)

    @property
    def health_ratio(self) -> float:
        """Fraction of connections that are currently open and non-stale (0.0–1.0).

        Unlike :attr:`is_healthy` which requires *all* connections to be
        healthy, this property returns a continuous value so callers can
        make proportional decisions (e.g. reduce REST fallback intensity when
        only a minority of connections are down).

        Returns ``1.0`` when no connections have been started yet so callers
        treat the manager as fully healthy before :meth:`start` is called.
        """
        if not self._connections:
            return 1.0
        now = time.monotonic()
        healthy_count = sum(
            1 for c in self._connections
            if c.ws is not None
            and not c.ws.closed
            and (now - c.last_pong) < self._heartbeat_interval * self._staleness_multiplier
        )
        return healthy_count / len(self._connections)

    @property
    def is_healthy(self) -> bool:
        now = time.monotonic()
        open_connections = [
            c for c in self._connections if c.ws is not None and not c.ws.closed
        ]
        if not open_connections or len(open_connections) != len(self._connections):
            return False
        return all(
            (now - c.last_pong) < self._heartbeat_interval * self._staleness_multiplier
            for c in open_connections
        )

    @property
    def ws_rest_fallback_count(self) -> int:
        return self._ws_rest_fallback_count

    @property
    def ws_reconnection_count(self) -> int:
        return self._ws_reconnection_count

    def get_healthy_connection_ratio(self) -> float:
        """Return ratio of healthy connections (0.0–1.0)."""
        if not self._connections:
            return 1.0
        healthy = sum(
            1 for c in self._connections
            if c.ws is not None and not c.ws.closed
        )
        return healthy / len(self._connections)

    async def _health_check_loop(self) -> None:
        """Periodically assess connection message rates and flag stale connections."""
        while self._running:
            await asyncio.sleep(WS_HEALTH_CHECK_INTERVAL)
            if not self._connections:
                continue
            now = time.monotonic()
            for idx, conn in enumerate(self._connections):
                elapsed = now - conn.health_check_ts if conn.health_check_ts > 0 else now
                msg_count = conn.health_msg_count
                if elapsed > 0:
                    rate = (msg_count / elapsed) * 60.0  # msgs/min
                else:
                    rate = 0.0
                self._connection_message_rates[idx] = rate
                if rate < WS_MIN_MESSAGE_RATE and conn.ws is not None and not conn.ws.closed:
                    log.info(
                        "WS connection {} ({}) has low message rate {:.2f} msgs/min — flagging unhealthy",
                        idx, self._label, rate,
                    )
                # Reset counters for next interval
                conn.health_check_ts = now
                conn.health_msg_count = 0
