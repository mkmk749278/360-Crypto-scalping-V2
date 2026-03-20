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
import os
import signal
import time
from typing import Dict, List, Optional, Set

from config import (
    PAIR_FETCH_INTERVAL_HOURS,
)
from src.ai_engine import get_ai_insight
from src.bootstrap import Bootstrap
from src.macro_watchdog import MacroWatchdog
from src.channels.base import Signal
from src.channels.scalp import ScalpChannel
from src.channels.swing import SwingChannel
from src.channels.range_channel import RangeChannel
from src.channels.tape import TapeChannel
from src.circuit_breaker import CircuitBreaker
from src.commands import CommandHandler
from src.detector import SMCDetector
from src.exchange import ExchangeManager
from src.historical_data import HistoricalDataStore
from src.onchain import OnChainClient
from src.openai_evaluator import OpenAIEvaluator
from src.order_flow import LiquidationEvent, OrderFlowStore, OIPoller
from src.pair_manager import PairManager
from src.paper_portfolio import PaperPortfolioManager
from src.performance_tracker import PerformanceTracker
from src.predictive_ai import PredictiveEngine
from src.regime import MarketRegimeDetector
from src.scanner import Scanner
from src.select_mode import SelectModeFilter
from src.signal_router import SignalRouter
from src.telegram_bot import TelegramBot
from src.telemetry import TelemetryCollector
from src.trade_monitor import TradeMonitor
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
    ONCHAIN_API_KEY,
    PERFORMANCE_TRACKER_PATH,
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

        # Paper trading portfolio simulator (virtual $1,000 per channel per user)
        self._paper_portfolio = PaperPortfolioManager()

        self.monitor = TradeMonitor(
            data_store=self.data_store,
            send_telegram=self.telegram.send_message,
            get_active_signals=lambda: self.router.active_signals,
            remove_signal=self._remove_and_archive,
            update_signal=self.router.update_signal,
            performance_tracker=self._performance_tracker,
            circuit_breaker=self._circuit_breaker,
            paper_portfolio=self._paper_portfolio,
        )

        # Channel strategies
        self._channels = [ScalpChannel(), SwingChannel(), RangeChannel(), TapeChannel()]

        # SMC detector and market regime classifier
        self._smc_detector = SMCDetector()
        self._regime_detector = MarketRegimeDetector()

        # Wire regime detector into trade monitor for signal invalidation checks
        self.monitor._regime_detector = self._regime_detector

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

        # WebSocket managers (set during boot)
        self._ws_spot: Optional[WebSocketManager] = None
        self._ws_futures: Optional[WebSocketManager] = None
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

        # Wire the per-symbol SL cooldown callback so the monitor triggers a
        # short cross-channel cooldown on the scanner after any stop-loss.
        self.monitor.on_sl_callback = self._scanner.set_symbol_sl_cooldown

        # Wire the post-invalidation cooldown callback so the monitor suppresses
        # rapid re-fire of the same (symbol, channel, direction) thesis after invalidation.
        self.monitor.on_invalidation_callback = self._scanner.set_invalidation_cooldown

        # Wire the thesis-based SL cooldown callback so the monitor suppresses
        # the same failing setup for a longer period after an SL hit.
        self.monitor.on_thesis_sl_callback = self._scanner.notify_sl_hit

        # Wire the free-channel highlight callback so the monitor posts winning
        # trades (TP2+) to the free channel in real-time.
        self.monitor.on_highlight_callback = lambda sig, tp, pnl: asyncio.ensure_future(
            self.router.publish_highlight(sig, tp, pnl)
        )

        # Select mode filter (OFF by default – admin must run /select_mode on)
        self._select_mode = SelectModeFilter()
        self._scanner.select_mode_filter = self._select_mode

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
            select_mode_filter=self._select_mode,
            paper_portfolio=self._paper_portfolio,
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
            # Binance Futures liquidation event
            order = data.get("o", {})
            liq_sym = order.get("s", "").upper()
            side = order.get("S", "")
            qty = float(order.get("q", 0))
            avg_price = float(order.get("ap") or order.get("p") or 0)
            if liq_sym and side and qty > 0 and avg_price > 0:
                self._order_flow_store.add_liquidation(
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

        await self._bootstrap.start_websockets()
        self._command_handler.ws_spot = self._ws_spot
        self._command_handler.ws_futures = self._ws_futures

    async def _pair_refresh_loop(self) -> None:
        """Periodically refresh pairs and seed any newly discovered symbols."""
        while True:
            await asyncio.sleep(PAIR_FETCH_INTERVAL_HOURS * 3600)
            try:
                old_spot, old_futures = self._current_ws_symbol_sets()
                new_symbols = await self.pair_mgr.refresh_pairs()
                for sym in new_symbols:
                    info = self.pair_mgr.pairs.get(sym)
                    if info is None:
                        continue
                    try:
                        await self.data_store.seed_symbol(sym, info.market)
                        for tf_name, data in self.data_store.candles.get(sym, {}).items():
                            self.pair_mgr.record_candles(
                                sym, tf_name, len(data.get("close", []))
                            )
                        log.info("Seeded new pair %s (%s)", sym, info.market)
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
