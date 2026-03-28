"""360-Crypto-Eye-Scalping – main orchestrator.

Boots the engine:
  1. Fetch top pairs from Binance
  2. Seed historical OHLCV + tick data
  3. Open WebSocket connections
  4. Run scanner → queue → router → Telegram pipeline
  5. Start trade monitor, telemetry, command handler

Usage:
    python -m src.main
"""

from __future__ import annotations

import asyncio
import collections
import os
import signal
import time
from typing import Dict, Deque, List, Optional, Set

from config import (
    PAIR_FETCH_INTERVAL_HOURS,
)
from src.ai_engine import get_ai_insight
from src.bootstrap import Bootstrap
from src.macro_watchdog import MacroWatchdog
from src.channels.base import Signal
from src.channels.scalp import ScalpChannel
from src.channels.swing import SwingChannel
from src.channels.spot import SpotChannel
from src.channels.scalp_fvg import ScalpFVGChannel
from src.channels.scalp_cvd import ScalpCVDChannel
from src.channels.scalp_vwap import ScalpVWAPChannel
from src.channels.scalp_obi import ScalpOBIChannel
from src.circuit_breaker import CircuitBreaker

from src.commands import CommandHandler
from src.detector import SMCDetector
from src.exchange import ExchangeManager
from src.historical_data import HistoricalDataStore
from src.onchain import OnChainClient
from src.openai_evaluator import OpenAIEvaluator
from src.order_flow import LiquidationEvent, OrderFlowStore, OIPoller
from src.pair_manager import PairManager, PairTier

from src.performance_tracker import PerformanceTracker
from src.predictive_ai import PredictiveEngine
from src.regime import MarketRegimeDetector
from src.scanner import Scanner
from src.gem_scanner import GemScanner
from src.signal_lifecycle import SignalLifecycleMonitor
from src.signal_router import SignalRouter
from src.telegram_bot import TelegramBot
from src.telemetry import TelemetryCollector
from src.trade_monitor import TradeMonitor
from src.trade_observer import TradeObserver
from src.exchange_client import CCXTClient
from src.order_manager import OrderManager
from src.utils import get_logger
from src.websocket_manager import WebSocketManager
from src.redis_client import RedisClient
from src.signal_queue import SignalQueue
from src.state_cache import StateCache
from config import (
    CIRCUIT_BREAKER_MAX_CONSECUTIVE_SL,
    CIRCUIT_BREAKER_MAX_HOURLY_SL,
    CIRCUIT_BREAKER_MAX_DAILY_DRAWDOWN_PCT,
    CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    CHANNEL_TELEGRAM_MAP,
    ONCHAIN_API_KEY,
    PERFORMANCE_TRACKER_PATH,
    AUTO_EXECUTION_ENABLED,
    EXCHANGE_ID,
    EXCHANGE_API_KEY,
    EXCHANGE_API_SECRET,
    EXCHANGE_SANDBOX,
    POSITION_SIZE_PCT,
    MAX_POSITION_USD,
)

log = get_logger("main")

# Interval between automatic disk snapshots of historical data (seconds)
_SNAPSHOT_INTERVAL_SECONDS: int = 300  # 5 minutes
_WS_SYMBOL_LIMIT: int = 50


class CryptoSignalEngine:
    """Top-level orchestrator for the signal engine.

    Wires together all sub-components and delegates to:
    - :class:`src.bootstrap.Bootstrap` for boot/shutdown/WebSocket setup
    - :class:`src.scanner.Scanner` for the periodic scan loop
    - :class:`src.commands.CommandHandler` for Telegram command routing
    """

    def __init__(self) -> None:
        self.pair_mgr = PairManager()
        self.data_store = HistoricalDataStore()
        self.telegram = TelegramBot()
        self.telemetry = TelemetryCollector()

        self._redis_client = RedisClient()
        self._signal_queue = SignalQueue(
            self._redis_client,
            alert_callback=self.telegram.send_admin_alert,
        )
        self._state_cache = StateCache(self._redis_client)
        self.router = SignalRouter(
            queue=self._signal_queue,
            send_telegram=self.telegram.send_message,
            format_signal=TelegramBot.format_signal,
            redis_client=self._redis_client,
        )

        # Portfolio enrichment: narrative builder and sector comparator
        from src.narrative import NarrativeBuilder
        from src.sector import SectorComparator
        self.router.narrative_builder = NarrativeBuilder()
        self.router.sector_comparator = SectorComparator(
            data_store=self.data_store,
            pair_mgr=self.pair_mgr,
        )

        # Order execution client and manager (feature 3 — off by default)
        _exchange_client: Optional[CCXTClient] = None
        if AUTO_EXECUTION_ENABLED:
            _exchange_client = CCXTClient(
                exchange_id=EXCHANGE_ID,
                api_key=EXCHANGE_API_KEY,
                secret=EXCHANGE_API_SECRET,
                sandbox=EXCHANGE_SANDBOX,
            )
        self._order_manager = OrderManager(
            auto_execution_enabled=AUTO_EXECUTION_ENABLED,
            exchange_client=_exchange_client,
            position_size_pct=POSITION_SIZE_PCT,
            max_position_usd=MAX_POSITION_USD,
        )

        # Circuit breaker (must be created before TradeMonitor)
        self._circuit_breaker = CircuitBreaker(
            max_consecutive_sl=CIRCUIT_BREAKER_MAX_CONSECUTIVE_SL,
            max_hourly_sl=CIRCUIT_BREAKER_MAX_HOURLY_SL,
            max_daily_drawdown_pct=CIRCUIT_BREAKER_MAX_DAILY_DRAWDOWN_PCT,
            cooldown_seconds=CIRCUIT_BREAKER_COOLDOWN_SECONDS,
            alert_callback=self.telegram.send_admin_alert,
        )

        # Performance tracker (must be created before TradeMonitor)
        self._performance_tracker = PerformanceTracker(
            storage_path=PERFORMANCE_TRACKER_PATH
        )

        self.monitor = TradeMonitor(
            data_store=self.data_store,
            send_telegram=self.telegram.send_message,
            get_active_signals=lambda: self.router.active_signals,
            remove_signal=self._remove_and_archive,
            update_signal=self.router.update_signal,
            performance_tracker=self._performance_tracker,
            circuit_breaker=self._circuit_breaker,
            order_manager=self._order_manager,
        )

        # Channel strategies
        self._channels = [
            ScalpChannel(),
            SwingChannel(),
            SpotChannel(),
            ScalpFVGChannel(),
            ScalpCVDChannel(),
            ScalpVWAPChannel(),
            ScalpOBIChannel(),
        ]

        # SMC detector and market regime classifier
        self._smc_detector = SMCDetector()
        self._regime_detector = MarketRegimeDetector()

        # Wire regime detector into trade monitor for signal invalidation checks
        self.monitor._regime_detector = self._regime_detector

        # Signal Lifecycle Monitor — actively monitors open SPOT/GEM/SWING
        # signals and posts periodic human-readable updates to subscribers.
        # Instantiated here; started as an async task in bootstrap.
        self._lifecycle_monitor = SignalLifecycleMonitor(
            router=self.router,
            data_store=self.data_store,
            regime_detector=self._regime_detector,
            send_telegram=self.telegram.send_message,
            exchange_mgr=None,  # wired after _exchange_mgr is created below
            send_photo=self.telegram.send_photo,  # chart images in lifecycle updates (feature 4)
        )

        # Predictive AI engine
        self.predictive = PredictiveEngine()

        # OpenAI GPT-4 macro-event evaluator (repurposed – no longer scores trade signals)
        self._openai_evaluator = OpenAIEvaluator()

        # Macro Watchdog – async background task for global market-event alerts
        # Polls news, Fear & Greed index, and uses OpenAI to detect significant
        # macro events (FOMC, wars, token listings) and sends alerts to Telegram.
        self._macro_watchdog = MacroWatchdog(
            send_alert=self.telegram.send_admin_alert,
            openai_evaluator=self._openai_evaluator,
        )

        # AI Trade Observer – captures full trade lifecycle data and generates
        # periodic AI-powered digests for the admin channel.
        self._trade_observer = TradeObserver(
            send_alert=self.telegram.send_admin_alert,
            data_store=self.data_store,
            regime_detector=self._regime_detector,
        )
        # Wire observer into router and monitor so they can call the hooks
        self.router.observer = self._trade_observer
        self.monitor.observer = self._trade_observer

        # On-chain intelligence client (optional — no-op if key is absent)
        self._onchain_client = OnChainClient(api_key=ONCHAIN_API_KEY)

        # Order flow analytics: OI tracking, liquidations, CVD divergence
        self._order_flow_store = OrderFlowStore()
        self._oi_poller = OIPoller(
            store=self._order_flow_store,
            futures_rest_base=os.getenv("BINANCE_FUTURES_REST_BASE", "https://fapi.binance.com"),
        )

        # Multi-exchange verification
        self._exchange_mgr = ExchangeManager(
            second_exchange_url=os.getenv("SECOND_EXCHANGE_URL")
        )
        # Wire exchange manager into lifecycle monitor now that it exists
        self._lifecycle_monitor._exchange_mgr = self._exchange_mgr

        # WebSocket managers (set during boot)
        self._ws_spot: Optional[WebSocketManager] = None
        self._ws_futures: Optional[WebSocketManager] = None
        # Dedicated liquidation WebSocket manager — forceOrder streams run in
        # their own connection pool so that liquidation floods during Extreme
        # Fear events cannot stall the kline WebSocket connections.
        self._ws_futures_liq: Optional[WebSocketManager] = None
        # Buffer for incoming forceOrder events — drained at the top of each
        # scan cycle so that processing is never inline on the WS message loop.
        self._pending_liquidations: Deque[LiquidationEvent] = collections.deque()
        self._tasks: List[asyncio.Task] = []
        self._shutdown_started: bool = False
        self._restart_lock = asyncio.Lock()

        # Command handler state
        self._paused_channels: Set[str] = set()
        self._confidence_overrides: Dict[str, float] = {}
        self._signal_history: List[Signal] = []  # capped at 500 entries
        self._boot_time: float = 0.0
        self._free_channel_limit: int = 2  # max free signals published per day
        self._alert_subscribers: Set[str] = set()  # admin IDs subscribed to alerts

        # Scanner (dependency-injected)
        self._scanner = Scanner(
            pair_mgr=self.pair_mgr,
            data_store=self.data_store,
            channels=self._channels,
            smc_detector=self._smc_detector,
            regime_detector=self._regime_detector,
            predictive=self.predictive,
            exchange_mgr=self._exchange_mgr,
            spot_client=None,
            telemetry=self.telemetry,
            signal_queue=self._signal_queue,
            router=self.router,
            openai_evaluator=self._openai_evaluator,
            onchain_client=self._onchain_client,
            order_flow_store=self._order_flow_store,
        )
        # Share mutable state with scanner
        self._scanner.paused_channels = self._paused_channels
        self._scanner.confidence_overrides = self._confidence_overrides
        self._scanner.circuit_breaker = self._circuit_breaker

        # Wire the free-channel highlight callback so the monitor posts winning
        # trades (TP2+) to the free channel in real-time.
        self.monitor.on_highlight_callback = lambda sig, tp, pnl: asyncio.ensure_future(
            self.router.publish_highlight(sig, tp, pnl)
        )

        # Gem scanner (OFF by default — admin must run /gem_mode on)
        self._gem_scanner = GemScanner()
        self._scanner.gem_scanner = self._gem_scanner

        # Command handler (delegates all Telegram commands)
        self._command_handler = CommandHandler(
            telegram=self.telegram,
            telemetry=self.telemetry,
            pair_mgr=self.pair_mgr,
            router=self.router,
            data_store=self.data_store,
            signal_queue=self._signal_queue,
            signal_history=self._signal_history,
            paused_channels=self._paused_channels,
            confidence_overrides=self._confidence_overrides,
            scanner=self._scanner,
            ws_spot=None,
            ws_futures=None,
            tasks=self._tasks,
            boot_time=self._boot_time,
            free_channel_limit=self._free_channel_limit,
            alert_subscribers=self._alert_subscribers,
            restart_callback=self._restart_tasks,
            ai_insight_fn=get_ai_insight,
            symbols_fn=lambda: self.pair_mgr.symbols,
            performance_tracker=self._performance_tracker,
            circuit_breaker=self._circuit_breaker,
            gem_scanner=self._gem_scanner,
            trade_observer=self._trade_observer,
        )

        # Bootstrap coordinates the boot/shutdown/WS sequence
        self._bootstrap = Bootstrap(self)

    def _remove_and_archive(self, signal_id: str) -> None:
        """Remove a signal from active tracking and archive it in history."""
        sig = self.router.active_signals.get(signal_id)
        if sig is not None:
            self._signal_history.append(sig)
            self._signal_history = self._signal_history[-500:]
        self.router.remove_signal(signal_id)

    # ------------------------------------------------------------------
    # Pre-flight checks (delegated to Bootstrap)
    # ------------------------------------------------------------------

    async def _preflight_check(self) -> bool:
        """Run pre-flight checks (delegated to Bootstrap)."""
        return await self._bootstrap.preflight_check()

    # ------------------------------------------------------------------
    # Boot / shutdown (delegated to Bootstrap)
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        # Warn operators about misconfigured Telegram channel IDs so that
        # signals are not silently dropped by the signal router.
        for chan_name, chan_id in CHANNEL_TELEGRAM_MAP.items():
            if not chan_id:
                log.warning(
                    "⚠️  STARTUP: Telegram channel ID for '%s' is not configured "
                    "(CHANNEL_TELEGRAM_MAP[%s] is empty). Signals for this channel "
                    "will be silently dropped. Set the corresponding env variable "
                    "in .env before starting the engine.",
                    chan_name, chan_name,
                )
        await self._bootstrap.boot()
        # Sync boot_time to command handler after boot sets it
        self._command_handler.boot_time = self._boot_time
        # Sync WS managers to command handler after boot starts them
        self._command_handler.ws_spot = self._ws_spot
        self._command_handler.ws_futures = self._ws_futures

    async def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        await self._bootstrap.shutdown()

    # ------------------------------------------------------------------
    # WebSocket setup (delegated to Bootstrap)
    # ------------------------------------------------------------------

    async def _start_websockets(self) -> None:
        await self._bootstrap.start_websockets()

    # ------------------------------------------------------------------
    # WebSocket message handler
    # ------------------------------------------------------------------

    async def _on_ws_message(self, data: dict) -> None:
        """Handle a raw WebSocket message (kline, trade, or forceOrder)."""
        event = data.get("e")
        symbol = data.get("s", "").upper()

        if event == "kline":
            k = data.get("k", {})
            interval = k.get("i", "")
            candle = {
                "open": float(k.get("o", 0)),
                "high": float(k.get("h", 0)),
                "low": float(k.get("l", 0)),
                "close": float(k.get("c", 0)),
                "volume": float(k.get("v", 0)),
            }
            if k.get("x"):  # candle closed
                self.data_store.update_candle(symbol, interval, candle)
                # Snapshot CVD at candle close to align with OHLCV for divergence detection
                self._order_flow_store.snapshot_cvd_at_candle_close(symbol)

        elif event == "trade":
            tick = {
                "price": float(data.get("p", 0)),
                "qty": float(data.get("q", 0)),
                "isBuyerMaker": data.get("m", False),
                "time": data.get("T", 0),
            }
            self.data_store.append_tick(symbol, tick)
            # Update running CVD from this tick
            price = tick["price"]
            qty = tick["qty"]
            vol_usd = price * qty
            if tick["isBuyerMaker"]:
                self._order_flow_store.update_cvd_from_tick(symbol, 0.0, vol_usd)
            else:
                self._order_flow_store.update_cvd_from_tick(symbol, vol_usd, 0.0)

        elif event == "forceOrder":
            # Buffer the liquidation event for deferred processing so that a
            # flood of forceOrder messages during a liquidation cascade (e.g.
            # Extreme Fear) does not block the WebSocket message loop and delay
            # PONG updates.  The buffer is drained at the start of each scan
            # cycle via _flush_pending_liquidations().
            order = data.get("o", {})
            liq_sym = order.get("s", "").upper()
            side = order.get("S", "")
            qty = float(order.get("q", 0))
            avg_price = float(order.get("ap") or order.get("p") or 0)
            if liq_sym and side and qty > 0 and avg_price > 0:
                self._pending_liquidations.append(
                    LiquidationEvent(
                        timestamp=time.monotonic(),
                        symbol=liq_sym,
                        side=side,
                        qty=qty,
                        price=avg_price,
                    )
                )

    # ------------------------------------------------------------------
    # Scanner loop (delegated to Scanner)
    # ------------------------------------------------------------------

    def _flush_pending_liquidations(self) -> None:
        """Drain the forceOrder buffer into the OrderFlowStore.

        Called periodically from ``_liquidation_flush_loop`` so liquidation
        events are processed in micro-batches rather than inline on the WS
        message loop.  This prevents event-loop blocking during liquidation
        cascades (e.g. Extreme Fear conditions) that would otherwise delay
        PONG updates and trigger false staleness detections.
        """
        while self._pending_liquidations:
            event = self._pending_liquidations.popleft()
            try:
                self._order_flow_store.add_liquidation(event)
            except Exception as exc:
                log.debug("Failed to add liquidation event: {}", exc)

    async def _liquidation_flush_loop(self) -> None:
        """Flush buffered forceOrder events every 100 ms."""
        while True:
            await asyncio.sleep(0.1)
            self._flush_pending_liquidations()

    async def _scan_loop(self) -> None:
        """Periodic scan over all pairs / channels (delegated to Scanner)."""
        await self._scanner.scan_loop()

    # ------------------------------------------------------------------
    # Free-channel, pair-refresh, snapshot loops
    # ------------------------------------------------------------------

    async def _free_channel_loop(self) -> None:
        """Publish daily performance recap every 24 hours."""
        while True:
            await asyncio.sleep(86_400)
            try:
                await self.router.publish_daily_recap(self._performance_tracker)
            except Exception as exc:
                log.error("Free channel publish error: %s", exc)

    async def _weekly_scoreboard_loop(self) -> None:
        """Publish weekly scoreboard every Sunday at ~00:00 UTC."""
        import datetime
        while True:
            now = datetime.datetime.now(datetime.timezone.utc)
            # Compute seconds until next Sunday 00:00 UTC (weekday 6 = Sunday)
            days_until_sunday = (6 - now.weekday()) % 7
            next_sunday = (now + datetime.timedelta(days=days_until_sunday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            wait_secs = (next_sunday - now).total_seconds()
            # If we are already past or very close to the target time (<60 s), push to
            # the following Sunday to avoid posting multiple times in the same window.
            if wait_secs < 60:
                next_sunday += datetime.timedelta(days=7)
                wait_secs = (next_sunday - now).total_seconds()
            await asyncio.sleep(max(wait_secs, 1))
            try:
                await self.router.publish_scoreboard(self._performance_tracker)
            except Exception as exc:
                log.error("Weekly scoreboard publish error: %s", exc)

    async def _daily_performance_report_loop(self) -> None:
        """Auto-generate an HTML performance report every 24 hours (feature 5)."""
        _REPORT_INTERVAL_SECONDS = 86400  # 24 hours
        while True:
            await asyncio.sleep(_REPORT_INTERVAL_SECONDS)
            try:
                from src.performance_report import generate_html_report
                path = generate_html_report(self._performance_tracker)
                log.info("Daily performance report generated: %s", path)
            except Exception as exc:
                log.error("Daily performance report generation failed: %s", exc)

    def _current_ws_symbol_sets(self) -> tuple[set[str], set[str]]:
        ws_limit = _WS_SYMBOL_LIMIT
        return (
            set(self.pair_mgr.spot_symbols[:ws_limit]),
            set(self.pair_mgr.futures_symbols[:ws_limit]),
        )

    async def _restart_websockets_if_pair_universe_changed(
        self,
        old_spot: set[str],
        old_futures: set[str],
    ) -> None:
        new_spot, new_futures = self._current_ws_symbol_sets()
        if old_spot == new_spot and old_futures == new_futures:
            return

        log.info("Tracked pair universe changed; restarting WebSocket subscriptions")
        if self._ws_spot:
            await self._ws_spot.stop()
            self._ws_spot = None
        if self._ws_futures:
            await self._ws_futures.stop()
            self._ws_futures = None
        if self._ws_futures_liq:
            await self._ws_futures_liq.stop()
            self._ws_futures_liq = None

        await self._bootstrap.start_websockets()
        self._command_handler.ws_spot = self._ws_spot
        self._command_handler.ws_futures = self._ws_futures

    async def _pair_refresh_loop(self) -> None:
        """Periodically refresh pairs, seed new ones, and prune removed ones."""
        while True:
            await asyncio.sleep(PAIR_FETCH_INTERVAL_HOURS * 3600)
            try:
                old_spot, old_futures = self._current_ws_symbol_sets()
                new_symbols, removed_symbols = await self.pair_mgr.refresh_pairs()

                # Handle removed (delisted / dropped) pairs
                if removed_symbols:
                    log.info(
                        "Pair pruning: removed %d pairs from universe",
                        len(removed_symbols),
                    )
                    for sym in removed_symbols:
                        self.data_store.candles.pop(sym, None)
                    await self.telegram.send_admin_alert(
                        f"📉 Pair universe pruned: {len(removed_symbols)} pairs removed "
                        f"(e.g. {', '.join(removed_symbols[:5])})"
                    )

                # Seed new pairs — all Tier 1 and Tier 2 pairs (skip Tier 3)
                if new_symbols:
                    log.info(
                        "Discovered %d new pairs — seeding historical data",
                        len(new_symbols),
                    )
                for sym in new_symbols:
                    info = self.pair_mgr.pairs.get(sym)
                    if info is None:
                        continue
                    # Tier 3 pairs receive only lightweight scans — no full seed
                    if info.tier == PairTier.TIER3:
                        continue
                    try:
                        await self.data_store.seed_symbol(sym, info.market)
                        for tf_name, data in self.data_store.candles.get(sym, {}).items():
                            self.pair_mgr.record_candles(
                                sym, tf_name, len(data.get("close", []))
                            )
                        log.info("Seeded new pair %s (%s, %s)", sym, info.market, info.tier)
                    except Exception as exc:
                        log.error("Failed to seed new pair %s: %s", sym, exc)
                await self._restart_websockets_if_pair_universe_changed(
                    old_spot, old_futures
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Pair refresh loop error: %s", exc)

    async def _snapshot_loop(self) -> None:
        """Periodically save historical data to disk for fast restarts."""
        while True:
            await asyncio.sleep(_SNAPSHOT_INTERVAL_SECONDS)
            try:
                await self.data_store.save_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Snapshot save error: %s", exc)

    # ------------------------------------------------------------------
    # Admin command handler (delegated to CommandHandler)
    # ------------------------------------------------------------------

    async def _handle_command(self, text: str, chat_id: str) -> None:
        """Route Telegram commands to CommandHandler."""
        await self._command_handler._handle_command(text, chat_id)

    async def _welcome_new_member(self, user_id: str) -> None:
        """Send a welcome DM when a user joins one of the bot's channels."""
        await self.telegram.send_message(
            user_id, self._command_handler.get_welcome_message()
        )

    async def _restart_tasks(self, chat_id: str) -> None:
        """Cancel and restart all async tasks (called by CommandHandler)."""
        async with self._restart_lock:
            old_tasks = list(self._tasks)
            for t in old_tasks:
                t.cancel()
            await asyncio.gather(*old_tasks, return_exceptions=True)
            self._tasks = []
            await self.router.stop()
            await self.monitor.stop()
            await self.telemetry.stop()
            await self.telegram.stop()
            if self._ws_spot:
                await self._ws_spot.stop()
                self._ws_spot = None
            if self._ws_futures:
                await self._ws_futures.stop()
                self._ws_futures = None
            await self._bootstrap.start_websockets()
            self._command_handler.ws_spot = self._ws_spot
            self._command_handler.ws_futures = self._ws_futures
            self._tasks = self._bootstrap.launch_runtime_tasks()
            # Re-sync tasks list into command handler
            self._command_handler._tasks = self._tasks
            await self.telegram.send_message(
                chat_id, "✅ Engine loops and WebSocket subscriptions restarted."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _run() -> None:
    engine = CryptoSignalEngine()
    loop = asyncio.get_running_loop()

    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, lambda: asyncio.create_task(engine.shutdown()))

    await engine.boot()
    # Keep running until cancelled
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await engine.shutdown()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
