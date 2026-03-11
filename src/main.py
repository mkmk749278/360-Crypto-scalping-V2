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
import signal
import time
from typing import Dict, List, Optional

import numpy as np

from config import (
    ALL_CHANNELS,
    CHANNEL_SCALP,
    CHANNEL_SWING,
    CHANNEL_RANGE,
    CHANNEL_TAPE,
    SEED_TIMEFRAMES,
)
from src.ai_engine import detect_whale_trade, detect_volume_delta_spike, get_ai_insight
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
from src.historical_data import HistoricalDataStore
from src.indicators import adx, atr, bollinger_bands, ema, momentum, rsi, sma
from src.pair_manager import PairManager
from src.signal_router import SignalRouter
from src.smc import detect_fvg, detect_liquidity_sweeps, detect_mss
from src.telegram_bot import TelegramBot
from src.telemetry import TelemetryCollector
from src.trade_monitor import TradeMonitor
from src.utils import get_logger
from src.websocket_manager import WebSocketManager

log = get_logger("main")


class CryptoSignalEngine:
    """Top-level orchestrator for the signal engine."""

    def __init__(self) -> None:
        self.pair_mgr = PairManager()
        self.data_store = HistoricalDataStore()
        self.telegram = TelegramBot()
        self.telemetry = TelemetryCollector()

        self._signal_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.router = SignalRouter(
            queue=self._signal_queue,
            send_telegram=self.telegram.send_message,
            format_signal=TelegramBot.format_signal,
        )
        self.monitor = TradeMonitor(
            data_store=self.data_store,
            send_telegram=self.telegram.send_message,
            get_active_signals=lambda: self.router.active_signals,
            remove_signal=self.router.remove_signal,
            update_signal=self.router.update_signal,
        )

        # Channel strategies
        self._channels = [ScalpChannel(), SwingChannel(), RangeChannel(), TapeChannel()]

        # WebSocket managers
        self._ws_spot: Optional[WebSocketManager] = None
        self._ws_futures: Optional[WebSocketManager] = None
        self._tasks: List[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Boot sequence
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        log.info("=== 360-Crypto-Eye-Scalping Engine BOOTING ===")

        # 1. Fetch pairs
        await self.pair_mgr.refresh_pairs()

        # 2. Seed historical data
        await self.data_store.seed_all(self.pair_mgr)

        # 3. Start WebSockets
        await self._start_websockets()

        # 4. Launch async tasks
        self._tasks = [
            asyncio.create_task(self.router.start()),
            asyncio.create_task(self.monitor.start()),
            asyncio.create_task(self.telemetry.start()),
            asyncio.create_task(self.pair_mgr.run_periodic_refresh()),
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

        self._ws_spot = WebSocketManager(self._on_ws_message, market="spot")
        self._ws_futures = WebSocketManager(self._on_ws_message, market="futures")

        if spot_streams:
            await self._ws_spot.start(spot_streams)
        if futures_streams:
            await self._ws_futures.start(futures_streams)

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
            ws_conns = (
                (self._ws_spot.stream_count if self._ws_spot else 0)
                + (self._ws_futures.stream_count if self._ws_futures else 0)
            )
            ws_ok = (
                (self._ws_spot.is_healthy if self._ws_spot else True)
                and (self._ws_futures.is_healthy if self._ws_futures else True)
            )
            self.telemetry.set_ws_health(ws_ok, ws_conns)

            await asyncio.sleep(1)  # scan cadence

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

        # SMC detection on 5m (scalp) and 4h (swing)
        smc_data: dict = {"sweeps": [], "mss": None, "fvg": []}
        for tf_key in ("5m", "4h", "15m", "1m"):
            cd = candles.get(tf_key)
            if cd is None or len(cd["close"]) < 51:
                continue
            sweeps = detect_liquidity_sweeps(cd["high"], cd["low"], cd["close"])
            if sweeps:
                smc_data["sweeps"] = sweeps
                # Check MSS on lower TF
                ltf = {"4h": "1h", "1h": "15m", "15m": "5m", "5m": "1m"}.get(tf_key, "1m")
                ltf_cd = candles.get(ltf)
                if ltf_cd and len(ltf_cd["close"]) > 1:
                    mss_sig = detect_mss(sweeps[0], ltf_cd["close"])
                    smc_data["mss"] = mss_sig
                fvg_zones = detect_fvg(cd["high"], cd["low"], cd["close"])
                smc_data["fvg"] = fvg_zones
                break  # use first TF with a sweep

        # Whale / tape data
        ticks = self.data_store.ticks.get(symbol, [])
        whale_alert = None
        if ticks:
            latest_tick = ticks[-1]
            whale_alert = detect_whale_trade(latest_tick["price"], latest_tick["qty"])
            smc_data["whale_alert"] = whale_alert
            smc_data["recent_ticks"] = ticks[-100:]
            buy_v = sum(t["qty"] * t["price"] for t in ticks[-100:] if not t.get("isBuyerMaker"))
            sell_v = sum(t["qty"] * t["price"] for t in ticks[-100:] if t.get("isBuyerMaker"))
            avg_delta = (buy_v + sell_v) / 2.0 if (buy_v + sell_v) > 0 else 0
            smc_data["volume_delta_spike"] = detect_volume_delta_spike(buy_v - sell_v, avg_delta)

        # AI insight (lightweight, no blocking)
        ai = {"label": "Neutral", "summary": "", "score": 0.0}
        try:
            insight = await asyncio.wait_for(get_ai_insight(symbol), timeout=2)
            ai = {"label": insight.label, "summary": insight.summary, "score": insight.score}
        except Exception:
            pass

        spread_pct = 0.01  # placeholder – real spread from order book

        # Evaluate each channel
        for chan in self._channels:
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

            cinp = ConfidenceInput(
                smc_score=score_smc(has_sweep, has_mss, has_fvg),
                trend_score=score_trend(ema_aligned, adx_ok, mom_positive),
                ai_sentiment_score=score_ai_sentiment(ai.get("score", 0)),
                liquidity_score=score_liquidity(volume_24h),
                spread_score=score_spread(spread_pct),
                data_sufficiency=score_data_sufficiency(candle_total),
                multi_exchange=score_multi_exchange(False),
                has_enough_history=self.pair_mgr.has_enough_history(symbol),
                opposing_position_open=False,
            )
            result = compute_confidence(cinp)
            if result.blocked:
                continue

            sig.confidence = result.total

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

    # ------------------------------------------------------------------
    # Admin command handler
    # ------------------------------------------------------------------

    async def _handle_command(self, text: str, chat_id: str) -> None:
        parts = text.strip().split()
        cmd = parts[0].lower()

        if cmd == "/view_dashboard":
            await self.telegram.send_message(chat_id, self.telemetry.dashboard_text())

        elif cmd == "/update_pairs":
            await self.pair_mgr.refresh_pairs()
            await self.telegram.send_message(
                chat_id, f"✅ Pairs refreshed: {len(self.pair_mgr.pairs)} active"
            )

        elif cmd == "/subscribe_alerts":
            await self.telegram.send_message(chat_id, "✅ You are subscribed to admin alerts.")

        else:
            await self.telegram.send_message(
                chat_id,
                "Available commands:\n/view\\_dashboard\n/update\\_pairs\n/subscribe\\_alerts",
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
