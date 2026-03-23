"""Bootstrap – engine boot, shutdown, and WebSocket initialisation.

Extracted from :class:`src.main.CryptoSignalEngine` for modularity.
The :class:`Bootstrap` class handles the engine startup sequence,
WebSocket connection setup, pre-flight checks, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, List

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_SCALP_CHANNEL_ID,
)
from src.ai_engine import close_shared_session
from src.binance import BinanceClient
from src.rate_limiter import futures_rate_limiter, spot_rate_limiter
from src.utils import get_logger
from src.websocket_manager import WebSocketManager

log = get_logger("bootstrap")

# Higher weight budget during boot — no competing scan traffic yet, so we
# can safely use more of Binance's 6,000/min Spot allowance for fast seeding.
_BOOT_BUDGET: int = 5_500
# Normal steady-state Spot budget — leaves ~1,000 headroom for WS reconnects.
_STEADY_BUDGET: int = 5_000

# Futures budgets — Binance Futures hard cap is 2,400/min.
_BOOT_BUDGET_FUTURES: int = 2_200
_STEADY_BUDGET_FUTURES: int = 2_000


class Bootstrap:
    """Manages the engine lifecycle: boot, shutdown, and WebSocket setup.

    Parameters
    ----------
    engine:
        The :class:`src.main.CryptoSignalEngine` instance.  All state
        (pair_mgr, data_store, etc.) is accessed via this reference so
        that Bootstrap remains a thin coordinator and avoids circular
        import issues.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def preflight_check(self) -> bool:
        """Run pre-flight checks and return True if all critical checks pass."""
        engine = self._engine
        ok = True

        if not TELEGRAM_BOT_TOKEN:
            log.warning("Pre-flight: TELEGRAM_BOT_TOKEN is not set")
            ok = False

        if not TELEGRAM_SCALP_CHANNEL_ID:
            log.warning("Pre-flight: No Telegram channel IDs configured")

        if not engine.pair_mgr.pairs:
            log.warning("Pre-flight: pair_mgr has no pairs loaded")
            ok = False

        if not engine.data_store.has_data():
            log.warning("Pre-flight: data_store has no seeded data")
            ok = False

        ws_healthy = (
            (engine._ws_spot.is_healthy if engine._ws_spot else True)
            and (engine._ws_futures.is_healthy if engine._ws_futures else True)
        )
        if not ws_healthy:
            log.warning("Pre-flight: WebSocket managers are not all healthy")

        if not engine._redis_client.available:
            log.warning(
                "Pre-flight: Redis not available – using in-memory fallback"
            )

        try:
            _ping_client = BinanceClient("spot")
            ping_resp = await asyncio.wait_for(
                _ping_client._get("/api/v3/ping", weight=1), timeout=5
            )
            await _ping_client.close()
            if ping_resp is None:
                log.warning("Pre-flight: Binance REST ping returned no data")
            else:
                log.info("Pre-flight: Binance REST ping OK")
        except Exception as exc:
            log.warning("Pre-flight: Binance REST ping failed: {}", exc)

        if ok:
            log.info("Pre-flight checks passed")
        return ok

    async def boot(self) -> None:
        """Execute the full engine boot sequence."""
        engine = self._engine
        log.info("=== 360-Crypto-Eye-Scalping Engine BOOTING ===")
        engine._boot_time = time.monotonic()

        # 0. Connect to Redis (graceful fallback if unavailable)
        await engine._redis_client.connect()
        engine.telemetry.set_redis_client(engine._redis_client)

        # Wire API call tracking
        BinanceClient.on_api_call = engine.telemetry.record_api_call

        # 1. Fetch pairs
        await engine.pair_mgr.refresh_pairs()

        # 2. Smart seed — temporarily raise the rate-limit budget since there
        #    is no competing scan traffic during boot.  Spot and Futures use
        #    separate budgets matching Binance's independent per-market caps.
        spot_rate_limiter.set_budget(_BOOT_BUDGET)
        futures_rate_limiter.set_budget(_BOOT_BUDGET_FUTURES)
        cached = engine.data_store.load_snapshot()
        if cached:
            log.info("Disk cache loaded — gap-filling missing data only")
            await engine.data_store.gap_fill(engine.pair_mgr)
        else:
            log.info("No disk cache found — performing full historical seed")
            await engine.data_store.seed_all(engine.pair_mgr)
        # Restore steady-state budgets now that seeding is complete.
        spot_rate_limiter.set_budget(_STEADY_BUDGET)
        futures_rate_limiter.set_budget(_STEADY_BUDGET_FUTURES)

        # 3. Load predictive model
        await engine.predictive.load_model()

        # 4. Start WebSockets
        await self.start_websockets()

        # 4.5 Pre-flight checks
        if not await self.preflight_check():
            log.warning(
                "Pre-flight checks had warnings — engine will start but may be degraded"
            )

        # 5. Launch async tasks
        engine._tasks = self.launch_runtime_tasks()

        await engine.telegram.send_admin_alert("✅ Engine booted successfully")
        log.info("=== Engine RUNNING ===")

    def launch_runtime_tasks(self) -> list[asyncio.Task]:
        """Create the standard long-running tasks used after boot or restart.

        This helper is shared by the initial boot path and the admin-triggered
        restart flow so both launch the same runtime loops after one-time setup
        such as pair loading, historical seeding, and WebSocket startup.

        Returns
        -------
        list[asyncio.Task]
            The running task objects for the engine's background loops.
        """
        engine = self._engine
        tasks = [
            asyncio.create_task(engine.router.start()),
            asyncio.create_task(engine.monitor.start()),
            asyncio.create_task(engine.telemetry.start()),
            asyncio.create_task(engine._pair_refresh_loop()),
            asyncio.create_task(engine._scanner.scan_loop()),
            asyncio.create_task(engine.telegram.poll_commands(
                engine._handle_command,
                on_new_member=engine._welcome_new_member,
            )),
            asyncio.create_task(engine._free_channel_loop()),
            asyncio.create_task(engine._snapshot_loop()),
            asyncio.create_task(engine._macro_watchdog.start()),
            asyncio.create_task(engine._liquidation_flush_loop()),
        ]

        # OI poller – background REST polling for Binance Futures Open Interest
        if getattr(engine, "_oi_poller", None) is not None:
            tasks.append(asyncio.create_task(engine._oi_poller.start()))

        return tasks

    async def shutdown(self) -> None:
        """Gracefully shut down all engine components."""
        engine = self._engine
        log.info("Shutting down …")
        tasks = list(engine._tasks)
        for t in tasks:
            t.cancel()
        await engine.router.stop()
        await engine.monitor.stop()
        await engine.telemetry.stop()
        if engine._ws_spot:
            await engine._ws_spot.stop()
        if engine._ws_futures:
            await engine._ws_futures.stop()
        if getattr(engine, "_ws_futures_liq", None):
            await engine._ws_futures_liq.stop()
        try:
            await engine.data_store.save_snapshot()
        except Exception as exc:
            log.error("Failed to save snapshot on shutdown: {}", exc)
        await engine.data_store.close()
        await engine.pair_mgr.close()
        await engine._exchange_mgr.close()
        if engine._scanner.spot_client:
            await engine._scanner.spot_client.close()
        try:
            await close_shared_session()
        except Exception as exc:
            log.warning("Failed to close AI engine shared session: {}", exc)
        if getattr(engine, "_openai_evaluator", None) is not None:
            try:
                await engine._openai_evaluator.close()
            except Exception as exc:
                log.warning("Failed to close OpenAI evaluator session: {}", exc)
        if getattr(engine, "_macro_watchdog", None) is not None:
            try:
                await engine._macro_watchdog.stop()
            except Exception as exc:
                log.warning("Failed to stop MacroWatchdog: {}", exc)
        if getattr(engine, "_oi_poller", None) is not None:
            try:
                await engine._oi_poller.stop()
            except Exception as exc:
                log.warning("Failed to stop OIPoller: {}", exc)
        if getattr(engine, "_onchain_client", None) is not None:
            try:
                await engine._onchain_client.close()
            except Exception as exc:
                log.warning("Failed to close on-chain client session: {}", exc)
        await engine._redis_client.close()
        await engine.telegram.stop()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        engine._tasks = []
        log.info("Shutdown complete.")

    async def start_websockets(self) -> None:
        """Subscribe to WebSocket streams for all tracked pairs."""
        engine = self._engine
        spot_streams: List[str] = []
        futures_kline_streams: List[str] = []
        futures_liq_streams: List[str] = []

        for sym in engine.pair_mgr.spot_symbols:
            s = sym.lower()
            spot_streams.append(f"{s}@kline_1m")
            spot_streams.append(f"{s}@kline_5m")
            spot_streams.append(f"{s}@kline_1h")
            spot_streams.append(f"{s}@kline_4h")
            spot_streams.append(f"{s}@trade")

        futures_syms = engine.pair_mgr.futures_symbols
        for sym in futures_syms:
            s = sym.lower()
            futures_kline_streams.append(f"{s}@kline_1m")
            futures_kline_streams.append(f"{s}@kline_5m")
            futures_kline_streams.append(f"{s}@kline_1h")
            futures_kline_streams.append(f"{s}@kline_4h")
            # Separate @forceOrder (liquidation) streams into their own pool
            # to prevent liquidation cascades from starving kline connections.
            # During Extreme Fear events, the flood of forceOrder events across
            # 50 symbols creates event-loop pressure that delays last_pong
            # updates on kline connections, causing false staleness detections.
            futures_liq_streams.append(f"{s}@forceOrder")

        engine._ws_spot = WebSocketManager(
            engine._on_ws_message,
            market="spot",
            admin_alert_callback=engine.telegram.send_admin_alert,
            data_store=engine.data_store,
        )
        engine._ws_futures = WebSocketManager(
            engine._on_ws_message,
            market="futures",
            admin_alert_callback=engine.telegram.send_admin_alert,
            data_store=engine.data_store,
        )
        # Dedicated liquidation WebSocket manager — uses a separate connection
        # pool so that forceOrder event floods cannot stall kline connections.
        # admin_alert_callback is intentionally None: drops on this manager are
        # expected during Extreme Fear liquidation cascades and should not spam
        # the admin with alerts.
        engine._ws_futures_liq = WebSocketManager(
            engine._on_ws_message,
            market="futures",
            admin_alert_callback=None,
            data_store=engine.data_store,
            label="futures_liq",
        )

        if spot_streams:
            await engine._ws_spot.start(spot_streams)
        if futures_kline_streams:
            await engine._ws_futures.start(futures_kline_streams)
        if futures_liq_streams:
            await engine._ws_futures_liq.start(futures_liq_streams)

        # Set critical pairs for REST fallback during WS outages
        top_spot = engine.pair_mgr.spot_symbols[:10]
        top_futures = engine.pair_mgr.futures_symbols[:10]
        if engine._ws_spot and top_spot:
            engine._ws_spot.set_critical_pairs(top_spot)
        if engine._ws_futures and top_futures:
            engine._ws_futures.set_critical_pairs(top_futures)

        # Wire WS managers into the scanner
        engine._scanner.ws_spot = engine._ws_spot
        engine._scanner.ws_futures = engine._ws_futures

        # Register futures symbols with the OI poller so it knows what to poll
        if getattr(engine, "_oi_poller", None) is not None:
            engine._oi_poller.set_symbols(list(futures_syms))
