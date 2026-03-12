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
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import psutil

from config import (
    ALL_CHANNELS,
    CHANNEL_SCALP,
    CHANNEL_SWING,
    CHANNEL_RANGE,
    CHANNEL_TAPE,
    PAIR_FETCH_INTERVAL_HOURS,
    SEED_TIMEFRAMES,
    TELEGRAM_ADMIN_CHAT_ID,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_SCALP_CHANNEL_ID,
)
from src.ai_engine import get_ai_insight
from src.binance import BinanceClient
from src.channels.base import Signal
from src.channels.scalp import ScalpChannel
from src.channels.swing import SwingChannel
from src.channels.range_channel import RangeChannel
from src.channels.tape import TapeChannel
from src.confidence import (
    ConfidenceInput,
    compute_confidence,
    score_ai_sentiment,
    score_data_sufficiency,
    score_liquidity,
    score_multi_exchange,
    score_smc,
    score_spread,
    score_trend,
)
from src.detector import SMCDetector
from src.exchange import ExchangeManager
from src.historical_data import HistoricalDataStore
from src.indicators import adx, atr, bollinger_bands, ema, momentum, rsi, sma
from src.logger import get_recent_logs
from src.pair_manager import PairManager
from src.predictive_ai import PredictiveEngine
from src.regime import MarketRegimeDetector
from src.signal_router import SignalRouter
from src.telegram_bot import TelegramBot
from src.telemetry import TelemetryCollector
from src.trade_monitor import TradeMonitor
from src.utils import get_logger
from src.websocket_manager import WebSocketManager
from src.redis_client import RedisClient
from src.signal_queue import SignalQueue
from src.state_cache import StateCache

log = get_logger("main")

# Maximum characters to include in a Telegram /view_logs response
# (stays safely below Telegram's ~4096-char message limit)
_TELEGRAM_LOG_MAX_CHARS: int = 3_500

# Repo root for subprocess git commands
_REPO_ROOT: Path = Path(__file__).parent.parent

# Order book spread cache TTL and per-cycle fetch cap.
# A 30-second TTL avoids hammering Binance REST every scan cycle, and
# the per-cycle cap prevents rate-limit bursts when many pairs are uncached.
_SPREAD_CACHE_TTL: float = 30.0
_MAX_ORDER_BOOK_FETCHES_PER_CYCLE: int = 5


class CryptoSignalEngine:
    """Top-level orchestrator for the signal engine."""

    def __init__(self) -> None:
        self.pair_mgr = PairManager()
        self.data_store = HistoricalDataStore()
        self.telegram = TelegramBot()
        self.telemetry = TelemetryCollector()

        self._redis_client = RedisClient()
        self._signal_queue = SignalQueue(self._redis_client)
        self._state_cache = StateCache(self._redis_client)
        self.router = SignalRouter(
            queue=self._signal_queue,
            send_telegram=self.telegram.send_message,
            format_signal=TelegramBot.format_signal,
        )
        self.monitor = TradeMonitor(
            data_store=self.data_store,
            send_telegram=self.telegram.send_message,
            get_active_signals=lambda: self.router.active_signals,
            remove_signal=self._remove_and_archive,
            update_signal=self.router.update_signal,
        )

        # Channel strategies
        self._channels = [ScalpChannel(), SwingChannel(), RangeChannel(), TapeChannel()]

        # SMC detector and market regime classifier
        self._smc_detector = SMCDetector()
        self._regime_detector = MarketRegimeDetector()

        # Predictive AI engine
        self.predictive = PredictiveEngine()

        # Multi-exchange verification
        self._exchange_mgr = ExchangeManager(
            second_exchange_url=os.getenv("SECOND_EXCHANGE_URL")
        )

        # BinanceClient for real order book spread (shared, lazily opened)
        self._spot_client: Optional[BinanceClient] = None
        # Cache: symbol → (spread_pct, timestamp)
        self._order_book_cache: Dict[str, Tuple[float, float]] = {}
        # Tracks order book REST fetches made in the current scan cycle
        self._order_book_fetches_this_cycle: int = 0

        # WebSocket managers
        self._ws_spot: Optional[WebSocketManager] = None
        self._ws_futures: Optional[WebSocketManager] = None
        self._tasks: List[asyncio.Task] = []

        # Command handler state
        self._paused_channels: Set[str] = set()
        self._confidence_overrides: Dict[str, float] = {}
        self._force_scan: bool = False
        self._signal_history: List[Signal] = []  # capped at 500 entries
        self._boot_time: float = 0.0
        self._free_channel_limit: int = 2  # max free signals published per day
        self._alert_subscribers: Set[str] = set()  # admin IDs subscribed to alerts

    def _remove_and_archive(self, signal_id: str) -> None:
        """Remove a signal from active tracking and archive it in history."""
        sig = self.router.active_signals.get(signal_id)
        if sig is not None:
            self._signal_history.append(sig)
            self._signal_history = self._signal_history[-500:]
        self.router.remove_signal(signal_id)

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    async def _preflight_check(self) -> bool:
        """Run pre-flight checks and return True if all critical checks pass."""
        ok = True

        if not TELEGRAM_BOT_TOKEN:
            log.warning("Pre-flight: TELEGRAM_BOT_TOKEN is not set")
            ok = False

        if not TELEGRAM_SCALP_CHANNEL_ID:
            log.warning("Pre-flight: No Telegram channel IDs configured")

        if not self.pair_mgr.pairs:
            log.warning("Pre-flight: pair_mgr has no pairs loaded")
            ok = False

        if not self.data_store.has_data():
            log.warning("Pre-flight: data_store has no seeded data")
            ok = False

        ws_healthy = (
            (self._ws_spot.is_healthy if self._ws_spot else True)
            and (self._ws_futures.is_healthy if self._ws_futures else True)
        )
        if not ws_healthy:
            log.warning("Pre-flight: WebSocket managers are not all healthy")

        # Non-fatal: check Redis connectivity
        if not self._redis_client.available:
            log.warning("Pre-flight: Redis not available – using in-memory fallback")

        # Non-fatal: basic Binance REST ping
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
            log.warning("Pre-flight: Binance REST ping failed: %s", exc)

        if ok:
            log.info("Pre-flight checks passed")
        return ok

    # ------------------------------------------------------------------
    # Boot sequence
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        log.info("=== 360-Crypto-Eye-Scalping Engine BOOTING ===")
        self._boot_time = time.monotonic()

        # 0. Connect to Redis (graceful fallback if unavailable)
        await self._redis_client.connect()
        self.telemetry.set_redis_client(self._redis_client)

        # Wire API call tracking so every Binance REST call increments telemetry
        BinanceClient.on_api_call = self.telemetry.record_api_call

        # 1. Fetch pairs
        await self.pair_mgr.refresh_pairs()

        # 2. Seed historical data
        await self.data_store.seed_all(self.pair_mgr)

        # 3. Load predictive model
        await self.predictive.load_model()

        # 4. Start WebSockets
        await self._start_websockets()

        # 4.5 Pre-flight checks
        if not await self._preflight_check():
            log.warning("Pre-flight checks had warnings — engine will start but may be degraded")

        # 5. Launch async tasks
        self._tasks = [
            asyncio.create_task(self.router.start()),
            asyncio.create_task(self.monitor.start()),
            asyncio.create_task(self.telemetry.start()),
            asyncio.create_task(self._pair_refresh_loop()),
            asyncio.create_task(self._scan_loop()),
            asyncio.create_task(self.telegram.poll_commands(self._handle_command)),
            asyncio.create_task(self._free_channel_loop()),
        ]

        await self.telegram.send_admin_alert("✅ Engine booted successfully")
        log.info("=== Engine RUNNING ===")

    async def shutdown(self) -> None:
        log.info("Shutting down …")
        for t in self._tasks:
            t.cancel()
        await self.router.stop()
        await self.monitor.stop()
        await self.telemetry.stop()
        if self._ws_spot:
            await self._ws_spot.stop()
        if self._ws_futures:
            await self._ws_futures.stop()
        await self.data_store.close()
        await self.pair_mgr.close()
        await self._exchange_mgr.close()
        if self._spot_client:
            await self._spot_client.close()
        await self._redis_client.close()
        await self.telegram.stop()
        log.info("Shutdown complete.")

    # ------------------------------------------------------------------
    # WebSocket setup
    # ------------------------------------------------------------------

    async def _start_websockets(self) -> None:
        spot_streams: List[str] = []
        futures_streams: List[str] = []

        for sym in self.pair_mgr.spot_symbols[:50]:
            s = sym.lower()
            spot_streams.append(f"{s}@kline_1m")
            spot_streams.append(f"{s}@kline_5m")
            spot_streams.append(f"{s}@trade")

        for sym in self.pair_mgr.futures_symbols[:50]:
            s = sym.lower()
            futures_streams.append(f"{s}@kline_1m")
            futures_streams.append(f"{s}@kline_5m")
            futures_streams.append(f"{s}@trade")

        self._ws_spot = WebSocketManager(
            self._on_ws_message,
            market="spot",
            admin_alert_callback=self.telegram.send_admin_alert,
        )
        self._ws_futures = WebSocketManager(
            self._on_ws_message,
            market="futures",
            admin_alert_callback=self.telegram.send_admin_alert,
        )

        if spot_streams:
            await self._ws_spot.start(spot_streams)
        if futures_streams:
            await self._ws_futures.start(futures_streams)

        # Set critical pairs for REST fallback during WS outages
        top_spot = self.pair_mgr.spot_symbols[:10]
        top_futures = self.pair_mgr.futures_symbols[:10]
        if self._ws_spot and top_spot:
            self._ws_spot.set_critical_pairs(top_spot)
        if self._ws_futures and top_futures:
            self._ws_futures.set_critical_pairs(top_futures)

    async def _on_ws_message(self, data: dict) -> None:
        """Handle a raw WebSocket message (kline or trade)."""
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

        elif event == "trade":
            tick = {
                "price": float(data.get("p", 0)),
                "qty": float(data.get("q", 0)),
                "isBuyerMaker": data.get("m", False),
                "time": data.get("T", 0),
            }
            self.data_store.append_tick(symbol, tick)

    # ------------------------------------------------------------------
    # Scanner loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        """Periodic scan over all pairs / channels."""
        log.info("Scanner loop started")
        while True:
            t0 = time.monotonic()
            self._order_book_fetches_this_cycle = 0
            try:
                for sym, info in list(self.pair_mgr.pairs.items()):
                    await self._scan_symbol(sym, info.volume_24h_usd)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Scan loop error: %s", exc)

            elapsed_ms = (time.monotonic() - t0) * 1000
            self.telemetry.set_scan_latency(elapsed_ms)
            self.telemetry.set_pairs_monitored(len(self.pair_mgr.pairs))
            self.telemetry.set_active_signals(len(self.router.active_signals))
            try:
                qsize = await self._signal_queue.qsize()
            except Exception as exc:
                log.warning("Failed to read signal queue size: %s", exc)
                qsize = 0
            self.telemetry.set_queue_size(qsize)
            ws_conns = (
                (self._ws_spot.stream_count if self._ws_spot else 0)
                + (self._ws_futures.stream_count if self._ws_futures else 0)
            )
            ws_ok = (
                (self._ws_spot.is_healthy if self._ws_spot else True)
                and (self._ws_futures.is_healthy if self._ws_futures else True)
            )
            self.telemetry.set_ws_health(ws_ok, ws_conns)

            if not self._force_scan:
                await asyncio.sleep(1)  # scan cadence
            self._force_scan = False

    async def _scan_symbol(self, symbol: str, volume_24h: float) -> None:
        """Run all channel evaluations for one symbol."""
        candles: Dict[str, dict] = {}
        for tf in SEED_TIMEFRAMES:
            c = self.data_store.get_candles(symbol, tf.interval)
            if c:
                candles[tf.interval] = c

        if not candles:
            return

        # Compute indicators per timeframe
        indicators: Dict[str, dict] = {}
        for tf_key, cd in candles.items():
            h, l, c, v = cd["high"], cd["low"], cd["close"], cd.get("volume", np.array([]))
            ind: dict = {}
            if len(c) >= 21:
                ind["ema9_last"] = float(ema(c, 9)[-1])
                ind["ema21_last"] = float(ema(c, 21)[-1])
            if len(c) >= 200:
                ind["ema200_last"] = float(ema(c, 200)[-1])
            if len(c) >= 30:
                a = adx(h, l, c, 14)
                valid = a[~np.isnan(a)]
                ind["adx_last"] = float(valid[-1]) if len(valid) else None
            if len(c) >= 15:
                a = atr(h, l, c, 14)
                valid = a[~np.isnan(a)]
                ind["atr_last"] = float(valid[-1]) if len(valid) else None
            if len(c) >= 15:
                r = rsi(c, 14)
                valid = r[~np.isnan(r)]
                ind["rsi_last"] = float(valid[-1]) if len(valid) else None
            if len(c) >= 20:
                u, m, lo = bollinger_bands(c, 20)
                ind["bb_upper_last"] = float(u[-1]) if not np.isnan(u[-1]) else None
                ind["bb_mid_last"] = float(m[-1]) if not np.isnan(m[-1]) else None
                ind["bb_lower_last"] = float(lo[-1]) if not np.isnan(lo[-1]) else None
            if len(c) >= 4:
                mom = momentum(c, 3)
                ind["momentum_last"] = float(mom[-1]) if not np.isnan(mom[-1]) else None
            indicators[tf_key] = ind

        # SMC detection via dedicated detector
        ticks = self.data_store.ticks.get(symbol, [])
        smc_result = self._smc_detector.detect(symbol, candles, ticks)
        smc_data = smc_result.as_dict()

        # Market regime classification (primary timeframe: 5m with 1m fallback)
        regime_ind = indicators.get("5m", indicators.get("1m", {}))
        regime_candles = candles.get("5m", candles.get("1m"))
        regime_result = self._regime_detector.classify(regime_ind, regime_candles)
        log.debug("%s regime: %s", symbol, regime_result.regime.value)

        # AI insight (lightweight, no blocking)
        ai = {"label": "Neutral", "summary": "", "score": 0.0}
        try:
            insight = await asyncio.wait_for(get_ai_insight(symbol), timeout=2)
            ai = {"label": insight.label, "summary": insight.summary, "score": insight.score}
        except Exception:
            pass

        # Real order book spread with TTL cache and per-cycle fetch cap
        spread_pct = 0.01  # fallback
        now = time.monotonic()
        cached = self._order_book_cache.get(symbol)
        if cached and (now - cached[1]) < _SPREAD_CACHE_TTL:
            spread_pct = cached[0]
        elif self._order_book_fetches_this_cycle < _MAX_ORDER_BOOK_FETCHES_PER_CYCLE:
            try:
                self._order_book_fetches_this_cycle += 1
                if self._spot_client is None:
                    self._spot_client = BinanceClient("spot")
                book = await self._spot_client.fetch_order_book(symbol, limit=5)
                if book and book.get("bids") and book.get("asks"):
                    best_bid = float(book["bids"][0][0])
                    best_ask = float(book["asks"][0][0])
                    mid = (best_bid + best_ask) / 2.0
                    if mid > 0:
                        spread_pct = (best_ask - best_bid) / mid * 100.0
                self._order_book_cache[symbol] = (spread_pct, now)
            except Exception:
                pass  # keep fallback

        # Evaluate each channel
        for chan in self._channels:
            if chan.config.name in self._paused_channels:
                continue
            # Scanner-level dedup: skip if there is already an active signal
            # for this exact (symbol, channel) combination
            if any(
                s.symbol == symbol and s.channel == chan.config.name
                for s in self.router.active_signals.values()
            ):
                log.debug(
                    "Skipping %s %s – active signal already exists",
                    symbol, chan.config.name,
                )
                continue
            try:
                sig = chan.evaluate(
                    symbol=symbol,
                    candles=candles,
                    indicators=indicators,
                    smc_data=smc_data,
                    ai_insight=ai,
                    spread_pct=spread_pct,
                    volume_24h_usd=volume_24h,
                )
            except Exception as exc:
                log.debug("Channel %s eval error for %s: %s", chan.config.name, symbol, exc)
                continue

            if sig is None:
                continue

            # Predictive AI: adjust TP/SL and confidence
            try:
                ind_for_predict = indicators.get("5m", indicators.get("1m", {}))
                prediction = await self.predictive.predict(symbol, candles, ind_for_predict)
                self.predictive.adjust_tp_sl(sig, prediction)
                self.predictive.update_confidence(sig, prediction)
            except Exception as exc:
                log.debug("Predictive AI error for %s: %s", symbol, exc)

            # Confidence scoring
            has_sweep = bool(smc_data["sweeps"])
            has_mss = smc_data["mss"] is not None
            has_fvg = bool(smc_data["fvg"])

            ind_5m = indicators.get("5m", indicators.get("1m", {}))
            ema_aligned = (
                ind_5m.get("ema9_last") is not None
                and ind_5m.get("ema21_last") is not None
                and (
                    (ind_5m["ema9_last"] > ind_5m["ema21_last"])
                    if sig.direction.value == "LONG"
                    else (ind_5m["ema9_last"] < ind_5m["ema21_last"])
                )
            )
            adx_ok = (ind_5m.get("adx_last") or 0) >= 20
            mom_positive = (ind_5m.get("momentum_last") or 0) > 0 if sig.direction.value == "LONG" else (ind_5m.get("momentum_last") or 0) < 0

            candle_total = sum(
                len(cd.get("close", []))
                for cd in candles.values()
            )

            # Cross-exchange verification
            cross_verified: Optional[bool] = None
            try:
                cross_verified = await asyncio.wait_for(
                    self._exchange_mgr.verify_signal_cross_exchange(
                        symbol, sig.direction.value, sig.entry
                    ),
                    timeout=3,
                )
            except asyncio.TimeoutError:
                log.debug("Cross-exchange verification timed out for %s", symbol)
            except Exception as exc:
                log.debug("Cross-exchange verification error for %s: %s", symbol, exc)

            cinp = ConfidenceInput(
                smc_score=score_smc(has_sweep, has_mss, has_fvg),
                trend_score=score_trend(ema_aligned, adx_ok, mom_positive),
                ai_sentiment_score=score_ai_sentiment(ai.get("score", 0)),
                liquidity_score=score_liquidity(volume_24h),
                spread_score=score_spread(spread_pct),
                data_sufficiency=score_data_sufficiency(candle_total),
                multi_exchange=score_multi_exchange(verified=cross_verified),
                has_enough_history=self.pair_mgr.has_enough_history(symbol),
                opposing_position_open=False,
            )
            result = compute_confidence(cinp)
            if result.blocked:
                continue

            sig.confidence = result.total

            # Apply channel confidence override if set
            min_conf = self._confidence_overrides.get(
                chan.config.name, chan.config.min_confidence
            )
            if sig.confidence < min_conf:
                continue

            # Attach regime info
            sig.market_phase = regime_result.regime.value

            # Populate liquidity info from SMC data
            liq_parts = []
            if smc_result.sweeps:
                sweep = smc_result.sweeps[0]
                liq_parts.append(f"Sweep {sweep.direction.value} at {sweep.sweep_level:.4f}")
            if smc_result.fvg:
                fvg = smc_result.fvg[0]
                liq_parts.append(f"FVG {fvg.gap_high:.4f}-{fvg.gap_low:.4f}")
            if liq_parts:
                sig.liquidity_info = " | ".join(liq_parts)

            # Attach market context for risk manager
            sig.spread_pct = spread_pct
            sig.volume_24h_usd = volume_24h

            try:
                self._signal_queue.put_nowait(sig)
            except asyncio.QueueFull:
                log.warning("Signal queue full – dropping %s", sig.signal_id)
    # ------------------------------------------------------------------
    # Free-channel daily publication
    # ------------------------------------------------------------------

    async def _free_channel_loop(self) -> None:
        """Publish top free signals every 24 hours."""
        while True:
            await asyncio.sleep(86_400)
            try:
                await self.router.publish_free_signals()
            except Exception as exc:
                log.error("Free channel publish error: %s", exc)

    async def _pair_refresh_loop(self) -> None:
        """Periodically refresh pairs and seed any newly discovered symbols."""
        while True:
            await asyncio.sleep(PAIR_FETCH_INTERVAL_HOURS * 3600)
            try:
                new_symbols = await self.pair_mgr.refresh_pairs()
                for sym in new_symbols:
                    info = self.pair_mgr.pairs.get(sym)
                    if info is None:
                        continue
                    try:
                        await self.data_store.seed_symbol(sym, info.market)
                        for tf_name, data in self.data_store.candles.get(sym, {}).items():
                            self.pair_mgr.record_candles(sym, tf_name, len(data.get("close", [])))
                        log.info("Seeded new pair %s (%s)", sym, info.market)
                    except Exception as exc:
                        log.error("Failed to seed new pair %s: %s", sym, exc)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Pair refresh loop error: %s", exc)

    # ------------------------------------------------------------------
    # Admin command handler
    # ------------------------------------------------------------------

    async def _handle_command(self, text: str, chat_id: str) -> None:
        parts = text.strip().split()
        cmd = parts[0].lower()

        # Command aliases
        _aliases = {"/status": "/engine_status"}
        cmd = _aliases.get(cmd, cmd)

        is_admin = bool(TELEGRAM_ADMIN_CHAT_ID and chat_id == TELEGRAM_ADMIN_CHAT_ID)

        # --- Admin-only commands ---

        if cmd in (
            "/view_dashboard", "/update_pairs", "/subscribe_alerts",
            "/view_pairs", "/force_scan", "/pause_channel", "/resume_channel",
            "/set_confidence_threshold", "/engine_status", "/memory_usage",
            "/set_free_channel_limit", "/force_update_ai", "/view_active_signals",
            "/view_logs", "/update_code", "/restart_engine", "/rollback_code",
        ) and not is_admin:
            await self.telegram.send_message(
                chat_id,
                "⛔ This command is restricted to administrators.",
            )
            return

        if cmd == "/view_dashboard":
            await self.telegram.send_message(chat_id, self.telemetry.dashboard_text())

        elif cmd == "/update_pairs":
            # Optional: /update_pairs spot/futures <count>
            market: Optional[str] = parts[1].lower() if len(parts) >= 2 else None
            count: Optional[int] = None
            if len(parts) >= 3:
                try:
                    count = int(parts[2])
                except ValueError:
                    pass
            await self.pair_mgr.refresh_pairs(market=market, count=count)
            await self.telegram.send_message(
                chat_id, f"✅ Pairs refreshed: {len(self.pair_mgr.pairs)} active"
            )

        elif cmd == "/subscribe_alerts":
            self._alert_subscribers.add(chat_id)
            await self.telegram.send_message(chat_id, "✅ You are subscribed to admin alerts.")

        elif cmd == "/view_pairs":
            # Optional: /view_pairs spot   or   /view_pairs futures
            market_filter: Optional[str] = parts[1].lower() if len(parts) >= 2 else None
            all_pairs = list(self.pair_mgr.pairs.values())
            if market_filter in ("spot", "futures"):
                all_pairs = [p for p in all_pairs if p.market == market_filter]
            sorted_pairs = sorted(all_pairs, key=lambda p: p.volume_24h_usd, reverse=True)
            top = sorted_pairs[:10]
            label = market_filter.capitalize() if market_filter else "All"
            lines = [f"📊 {label} Pairs: {len(all_pairs)} active\n\nTop 10 by volume:"]
            for i, p in enumerate(top, 1):
                lines.append(f"{i}. {p.symbol} ({p.market}) — ${p.volume_24h_usd:,.0f}")
            await self.telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/force_scan":
            self._force_scan = True
            await self.telegram.send_message(chat_id, "⚡ Force scan triggered.")

        elif cmd == "/pause_channel":
            if len(parts) < 2:
                await self.telegram.send_message(chat_id, "Usage: /pause\\_channel <name>")
            else:
                name = parts[1]
                self._paused_channels.add(name)
                await self.telegram.send_message(chat_id, f"⏸ Channel `{name}` paused.")

        elif cmd == "/resume_channel":
            if len(parts) < 2:
                await self.telegram.send_message(chat_id, "Usage: /resume\\_channel <name>")
            else:
                name = parts[1]
                self._paused_channels.discard(name)
                await self.telegram.send_message(chat_id, f"▶️ Channel `{name}` resumed.")

        elif cmd == "/set_confidence_threshold":
            if len(parts) < 3:
                await self.telegram.send_message(
                    chat_id, "Usage: /set\\_confidence\\_threshold <channel> <value>"
                )
            else:
                channel = parts[1]
                try:
                    value = float(parts[2])
                except ValueError:
                    await self.telegram.send_message(chat_id, "❌ Value must be a number.")
                    return
                self._confidence_overrides[channel] = value
                await self.telegram.send_message(
                    chat_id, f"✅ Confidence threshold for `{channel}` set to {value:.2f}"
                )

        elif cmd == "/engine_status":
            uptime_s = time.monotonic() - self._boot_time
            hours, rem = divmod(int(uptime_s), 3600)
            minutes, secs = divmod(rem, 60)
            ws_healthy = (
                (self._ws_spot.is_healthy if self._ws_spot else True)
                and (self._ws_futures.is_healthy if self._ws_futures else True)
            )
            lines = [
                "🔧 Engine Status",
                f"Uptime: {hours}h {minutes}m {secs}s",
                f"Running tasks: {sum(1 for t in self._tasks if not t.done())}",
                f"Queue size: {await self._signal_queue.qsize()}",
                f"Pairs: {len(self.pair_mgr.pairs)}",
                f"Active signals: {len(self.router.active_signals)}",
                f"WS healthy: {'✅' if ws_healthy else '❌'}",
            ]
            await self.telegram.send_message(chat_id, "\n".join(lines))

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
            await self.telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/set_free_channel_limit":
            if len(parts) < 2:
                await self.telegram.send_message(
                    chat_id, "Usage: /set\\_free\\_channel\\_limit <n>"
                )
            else:
                try:
                    limit = int(parts[1])
                    self._free_channel_limit = max(0, limit)
                    self.router.set_free_limit(self._free_channel_limit)
                    await self.telegram.send_message(
                        chat_id,
                        f"✅ Free channel daily signal limit set to {self._free_channel_limit}",
                    )
                except ValueError:
                    await self.telegram.send_message(chat_id, "❌ Value must be an integer.")

        elif cmd == "/force_update_ai":
            try:
                # Invalidate the AI/sentiment cache by forcing a fresh fetch for known pairs
                count = 0
                for sym in list(self.pair_mgr.symbols)[:5]:
                    try:
                        await asyncio.wait_for(get_ai_insight(sym), timeout=3)
                        count += 1
                    except Exception:
                        pass
                await self.telegram.send_message(
                    chat_id, f"✅ AI/sentiment cache refreshed for {count} symbols."
                )
            except Exception as exc:
                await self.telegram.send_message(chat_id, f"❌ AI refresh error: {exc}")

        elif cmd == "/view_active_signals":
            sigs = list(self.router.active_signals.values())
            if not sigs:
                await self.telegram.send_message(chat_id, "No active signals.")
            else:
                lines = [f"📡 Active Signals ({len(sigs)}):"]
                for s in sigs:
                    lines.append(
                        f"• [{s.signal_id}] {s.symbol} {s.direction.value} | "
                        f"Entry {s.entry:.4f} | SL {s.stop_loss:.4f} | "
                        f"Conf {s.confidence:.0f}% | {s.status}"
                    )
                await self.telegram.send_message(chat_id, "\n".join(lines))

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
                await self.telegram.send_message(chat_id, "No log file found.")
            else:
                excerpt = logs[-_TELEGRAM_LOG_MAX_CHARS:]
                await self.telegram.send_message(chat_id, f"```\n{excerpt}\n```")

        elif cmd == "/update_code":
            await self.telegram.send_message(chat_id, "⏳ Running git pull …")
            try:
                result = subprocess.run(
                    ["git", "pull"],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(_REPO_ROOT),
                )
                output = (result.stdout + result.stderr).strip() or "No output."
                await self.telegram.send_message(chat_id, f"✅ git pull result:\n```\n{output}\n```")
            except subprocess.TimeoutExpired:
                await self.telegram.send_message(chat_id, "❌ git pull timed out.")
            except Exception as exc:
                await self.telegram.send_message(chat_id, f"❌ git pull error: {exc}")

        elif cmd == "/restart_engine":
            await self.telegram.send_message(chat_id, "🔄 Restarting engine tasks …")
            try:
                # Cancel running tasks (scan, router, monitor, telemetry, polling)
                old_tasks = list(self._tasks)
                for t in old_tasks:
                    t.cancel()
                # Wait briefly for tasks to finish
                await asyncio.gather(*old_tasks, return_exceptions=True)
                self._tasks = []
                # Stop subsystems gracefully (also resets telegram polling state)
                await self.router.stop()
                await self.monitor.stop()
                await self.telemetry.stop()
                await self.telegram.stop()
                # Re-launch tasks (skip pair fetch, data seed, WS — they're still alive)
                self._tasks = [
                    asyncio.create_task(self.router.start()),
                    asyncio.create_task(self.monitor.start()),
                    asyncio.create_task(self.telemetry.start()),
                    asyncio.create_task(self._pair_refresh_loop()),
                    asyncio.create_task(self._scan_loop()),
                    asyncio.create_task(self.telegram.poll_commands(self._handle_command)),
                    asyncio.create_task(self._free_channel_loop()),
                ]
                await self.telegram.send_message(chat_id, "✅ Engine tasks restarted.")
            except Exception as exc:
                log.error("Restart error: %s", exc)
                await self.telegram.send_message(chat_id, f"❌ Restart error: {exc}")

        elif cmd == "/rollback_code":
            if len(parts) < 2:
                await self.telegram.send_message(
                    chat_id, "Usage: /rollback\\_code <commit>"
                )
            else:
                commit = parts[1]
                # Restrict to safe commit references: hex SHAs, branch names, tags
                # (alphanumeric, hyphens only – no path traversal characters)
                if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9.\-]{0,79}$', commit):
                    await self.telegram.send_message(chat_id, "❌ Invalid commit reference.")
                else:
                    await self.telegram.send_message(chat_id, f"⏳ Running git checkout {commit} …")
                    try:
                        result = subprocess.run(
                            ["git", "checkout", commit],
                            capture_output=True, text=True, timeout=30,
                            cwd=str(_REPO_ROOT),
                        )
                        output = (result.stdout + result.stderr).strip() or "Done."
                        await self.telegram.send_message(
                            chat_id, f"✅ Rollback result:\n```\n{output}\n```"
                        )
                    except subprocess.TimeoutExpired:
                        await self.telegram.send_message(chat_id, "❌ git checkout timed out.")
                    except Exception as exc:
                        await self.telegram.send_message(chat_id, f"❌ Rollback error: {exc}")

        # --- User commands ---

        elif cmd == "/signals":
            sigs = list(self.router.active_signals.values())[:5]
            if not sigs:
                await self.telegram.send_message(chat_id, "No active signals.")
            else:
                lines = ["📡 Active Signals (last 5):"]
                for s in sigs:
                    lines.append(
                        f"• {s.symbol} {s.direction.value} | "
                        f"Entry {s.entry:.4f} | Conf {s.confidence:.0f}% | {s.status}"
                    )
                await self.telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/free_signals":
            sigs = [
                s for s in self.router.active_signals.values()
                if s.channel == "free"
            ]
            if not sigs:
                await self.telegram.send_message(chat_id, "No free signals today.")
            else:
                lines = ["🆓 Today's Free Picks:"]
                for s in sigs:
                    lines.append(
                        f"• {s.symbol} {s.direction.value} | "
                        f"Entry {s.entry:.4f} | Conf {s.confidence:.0f}%"
                    )
                await self.telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/signal_info":
            if len(parts) < 2:
                await self.telegram.send_message(chat_id, "Usage: /signal\\_info <id>")
            else:
                sid = parts[1]
                sig = self.router.active_signals.get(sid)
                if sig is None:
                    sig = next((s for s in self._signal_history if s.signal_id == sid), None)
                if sig is None:
                    await self.telegram.send_message(chat_id, f"❌ Signal `{sid}` not found.")
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
                    await self.telegram.send_message(chat_id, "\n".join(lines))

        elif cmd == "/last_update":
            scan_ms = self.telemetry._scan_latency_ms
            pairs = len(self.pair_mgr.pairs)
            active = len(self.router.active_signals)
            await self.telegram.send_message(
                chat_id,
                f"🕐 Last scan latency: {scan_ms:.0f}ms\n"
                f"Pairs: {pairs} | Active signals: {active}",
            )

        elif cmd == "/subscribe":
            await self.telegram.send_message(chat_id, "✅ Subscribed to premium signals.")

        elif cmd == "/unsubscribe":
            await self.telegram.send_message(chat_id, "✅ Unsubscribed.")

        elif cmd == "/signal_history":
            recent = self._signal_history[-10:]
            if not recent:
                await self.telegram.send_message(chat_id, "No completed signals yet.")
            else:
                lines = ["📜 Signal History (last 10):"]
                for s in reversed(recent):
                    lines.append(
                        f"• {s.symbol} {s.direction.value} | "
                        f"{s.status} | PnL {s.pnl_pct:+.2f}%"
                    )
                await self.telegram.send_message(chat_id, "\n".join(lines))

        else:
            await self.telegram.send_message(
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
                "/rollback\\_code <commit>\n\n"
                "*User:*\n"
                "/signals\n"
                "/free\\_signals\n"
                "/signal\\_info <id>\n"
                "/last\\_update\n"
                "/subscribe\n"
                "/unsubscribe\n"
                "/signal\\_history",
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
