"""Scanner – periodic evaluation of all pairs across channel strategies.

Extracted from :class:`src.main.CryptoSignalEngine` for modularity.
Supports signal cooldown de-duplication, market-regime-aware gating,
and optional circuit-breaker integration.
"""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from config import SEED_TIMEFRAMES, SIGNAL_SCAN_COOLDOWN_SECONDS
from src.ai_engine import get_ai_insight
from src.binance import BinanceClient
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
from src.indicators import adx, atr, bollinger_bands, ema, momentum, rsi
from src.regime import MarketRegime
from src.utils import get_logger

log = get_logger("scanner")

# Order book spread cache TTL and per-cycle fetch cap
_SPREAD_CACHE_TTL: float = 30.0
_MAX_ORDER_BOOK_FETCHES_PER_CYCLE: int = 5

# ADX threshold below which SCALP signals are suppressed during RANGING regime
_RANGING_ADX_SUPPRESS_THRESHOLD: float = 15.0

# Confidence boost applied to RANGE channel when regime is RANGING
_RANGING_RANGE_CONF_BOOST: float = 5.0


class Scanner:
    """Scans all pairs across channel strategies on every cycle.

    Parameters
    ----------
    pair_mgr:
        :class:`src.pair_manager.PairManager` instance.
    data_store:
        :class:`src.historical_data.HistoricalDataStore` instance.
    channels:
        List of channel strategy objects.
    smc_detector:
        :class:`src.detector.SMCDetector` instance.
    regime_detector:
        :class:`src.regime.MarketRegimeDetector` instance.
    predictive:
        :class:`src.predictive_ai.PredictiveEngine` instance.
    exchange_mgr:
        :class:`src.exchange.ExchangeManager` instance.
    spot_client:
        Optional :class:`src.binance.BinanceClient` for order book fetches.
    telemetry:
        :class:`src.telemetry.TelemetryCollector` instance.
    signal_queue:
        :class:`src.signal_queue.SignalQueue` instance.
    router:
        :class:`src.signal_router.SignalRouter` instance.
    """

    def __init__(
        self,
        pair_mgr: Any,
        data_store: Any,
        channels: List[Any],
        smc_detector: Any,
        regime_detector: Any,
        predictive: Any,
        exchange_mgr: Any,
        spot_client: Optional[Any],
        telemetry: Any,
        signal_queue: Any,
        router: Any,
    ) -> None:
        self.pair_mgr = pair_mgr
        self.data_store = data_store
        self.channels = channels
        self.smc_detector = smc_detector
        self.regime_detector = regime_detector
        self.predictive = predictive
        self.exchange_mgr = exchange_mgr
        self.spot_client: Optional[Any] = spot_client
        self.telemetry = telemetry
        self.signal_queue = signal_queue
        self.router = router

        # Mutable state shared with the engine / command handler
        self.paused_channels: Set[str] = set()
        self.confidence_overrides: Dict[str, float] = {}
        self.force_scan: bool = False

        # WebSocket managers (set after boot)
        self.ws_spot: Optional[Any] = None
        self.ws_futures: Optional[Any] = None

        # Optional circuit breaker (set after construction)
        self.circuit_breaker: Optional[Any] = None

        # Optional select-mode filter (set after construction)
        self.select_mode_filter: Optional[Any] = None

        # Order book spread cache: symbol → (spread_pct, timestamp)
        self._order_book_cache: Dict[str, Tuple[float, float]] = {}
        self._order_book_fetches_this_cycle: int = 0

        # Cooldown tracking: (symbol, channel_name) → monotonic expiry time
        self._cooldown_until: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def scan_loop(self) -> None:
        """Periodic scan over all pairs / channels."""
        log.info("Scanner loop started")
        while True:
            t0 = time.monotonic()
            self._order_book_fetches_this_cycle = 0

            # Skip scanning when circuit breaker is tripped
            if self.circuit_breaker and self.circuit_breaker.is_tripped():
                log.warning("Circuit breaker tripped — skipping scan cycle")
                await asyncio.sleep(5)
                continue

            try:
                # Prioritise high-volume pairs for order book fetches
                sorted_pairs = sorted(
                    self.pair_mgr.pairs.items(),
                    key=lambda kv: kv[1].volume_24h_usd,
                    reverse=True,
                )
                for sym, info in sorted_pairs:
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
                qsize = await self.signal_queue.qsize()
            except Exception as exc:
                log.warning("Failed to read signal queue size: %s", exc)
                qsize = 0
            self.telemetry.set_queue_size(qsize)
            ws_conns = (
                (self.ws_spot.stream_count if self.ws_spot else 0)
                + (self.ws_futures.stream_count if self.ws_futures else 0)
            )
            ws_ok = (
                (self.ws_spot.is_healthy if self.ws_spot else True)
                and (self.ws_futures.is_healthy if self.ws_futures else True)
            )
            self.telemetry.set_ws_health(ws_ok, ws_conns)

            if not self.force_scan:
                await asyncio.sleep(1)
            self.force_scan = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_in_cooldown(self, symbol: str, channel_name: str) -> bool:
        """Return True if the (symbol, channel) pair is currently in cooldown."""
        key = (symbol, channel_name)
        expiry = self._cooldown_until.get(key)
        if expiry is None:
            return False
        if time.monotonic() < expiry:
            return True
        # Expired – clean up
        del self._cooldown_until[key]
        return False

    def _set_cooldown(self, symbol: str, channel_name: str) -> None:
        """Start the cooldown timer for (symbol, channel)."""
        cooldown_s = SIGNAL_SCAN_COOLDOWN_SECONDS.get(channel_name, 300)
        self._cooldown_until[(symbol, channel_name)] = (
            time.monotonic() + cooldown_s
        )
        log.debug(
            "Cooldown set for %s %s (%.0fs)", symbol, channel_name, cooldown_s
        )

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
            h, lo, c, _ = (
                cd["high"],
                cd["low"],
                cd["close"],
                cd.get("volume", np.array([])),
            )
            ind: dict = {}
            if len(c) >= 21:
                ind["ema9_last"] = float(ema(c, 9)[-1])
                ind["ema21_last"] = float(ema(c, 21)[-1])
            if len(c) >= 200:
                ind["ema200_last"] = float(ema(c, 200)[-1])
            if len(c) >= 30:
                a = adx(h, lo, c, 14)
                valid = a[~np.isnan(a)]
                ind["adx_last"] = float(valid[-1]) if len(valid) else None
            if len(c) >= 15:
                a = atr(h, lo, c, 14)
                valid = a[~np.isnan(a)]
                ind["atr_last"] = float(valid[-1]) if len(valid) else None
            if len(c) >= 15:
                r = rsi(c, 14)
                valid = r[~np.isnan(r)]
                ind["rsi_last"] = float(valid[-1]) if len(valid) else None
            if len(c) >= 20:
                u, m, lo_b = bollinger_bands(c, 20)
                ind["bb_upper_last"] = float(u[-1]) if not np.isnan(u[-1]) else None
                ind["bb_mid_last"] = float(m[-1]) if not np.isnan(m[-1]) else None
                ind["bb_lower_last"] = (
                    float(lo_b[-1]) if not np.isnan(lo_b[-1]) else None
                )
            if len(c) >= 4:
                mom = momentum(c, 3)
                ind["momentum_last"] = (
                    float(mom[-1]) if not np.isnan(mom[-1]) else None
                )
            indicators[tf_key] = ind

        # SMC detection
        ticks = self.data_store.ticks.get(symbol, [])
        smc_result = self.smc_detector.detect(symbol, candles, ticks)
        smc_data = smc_result.as_dict()

        # Market regime classification (primary timeframe: 5m with 1m fallback)
        regime_ind = indicators.get("5m", indicators.get("1m", {}))
        regime_candles = candles.get("5m", candles.get("1m"))
        regime_result = self.regime_detector.classify(regime_ind, regime_candles)
        log.debug("%s regime: %s", symbol, regime_result.regime.value)

        # AI insight (lightweight, no blocking)
        ai: Dict = {"label": "Neutral", "summary": "", "score": 0.0}
        try:
            insight = await asyncio.wait_for(get_ai_insight(symbol), timeout=2)
            ai = {
                "label": insight.label,
                "summary": insight.summary,
                "score": insight.score,
            }
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
                if self.spot_client is None:
                    self.spot_client = BinanceClient("spot")
                book = await self.spot_client.fetch_order_book(symbol, limit=5)
                if book and book.get("bids") and book.get("asks"):
                    best_bid = float(book["bids"][0][0])
                    best_ask = float(book["asks"][0][0])
                    mid = (best_bid + best_ask) / 2.0
                    if mid > 0:
                        spread_pct = (best_ask - best_bid) / mid * 100.0
                self._order_book_cache[symbol] = (spread_pct, now)
            except Exception:
                pass  # keep fallback

        # TODO: migrate order book spread fetching to @bookTicker WebSocket stream
        # for real-time spread data without REST API rate-limit cost.
        # See: https://binance-docs.github.io/apidocs/spot/en/#individual-symbol-book-ticker-streams

        is_ranging = regime_result.regime == MarketRegime.RANGING
        adx_val = regime_ind.get("adx_last") or 0

        # Evaluate each channel
        for chan in self.channels:
            chan_name = chan.config.name
            if chan_name in self.paused_channels:
                continue

            # Cooldown check: skip if a signal was recently fired for this pair/channel
            if self._is_in_cooldown(symbol, chan_name):
                log.debug(
                    "Cooldown active: skipping %s %s", symbol, chan_name
                )
                continue

            # Scanner-level dedup: skip if there is already an active signal
            if any(
                s.symbol == symbol and s.channel == chan_name
                for s in self.router.active_signals.values()
            ):
                log.debug(
                    "Skipping %s %s – active signal already exists", symbol, chan_name
                )
                continue

            # Market regime-aware gating
            # Suppress SCALP signals during ranging markets with low ADX
            if (
                chan_name == "360_SCALP"
                and is_ranging
                and adx_val < _RANGING_ADX_SUPPRESS_THRESHOLD
            ):
                log.debug(
                    "Suppressing SCALP signal for %s (RANGING, ADX=%.1f)",
                    symbol,
                    adx_val,
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
                log.debug(
                    "Channel %s eval error for %s: %s", chan_name, symbol, exc
                )
                continue

            if sig is None:
                continue

            # Boost RANGE confidence during ranging regime
            if chan_name == "360_RANGE" and is_ranging:
                sig.confidence = min(sig.confidence + _RANGING_RANGE_CONF_BOOST, 100.0)
                log.debug(
                    "RANGE confidence boosted for %s (RANGING): %.1f",
                    symbol,
                    sig.confidence,
                )

            # Predictive AI: adjust TP/SL and confidence
            try:
                ind_for_predict = indicators.get("5m", indicators.get("1m", {}))
                prediction = await self.predictive.predict(
                    symbol, candles, ind_for_predict
                )
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
            mom_positive = (
                (ind_5m.get("momentum_last") or 0) > 0
                if sig.direction.value == "LONG"
                else (ind_5m.get("momentum_last") or 0) < 0
            )

            candle_total = sum(
                len(cd.get("close", [])) for cd in candles.values()
            )

            # Cross-exchange verification
            cross_verified: Optional[bool] = None
            try:
                cross_verified = await asyncio.wait_for(
                    self.exchange_mgr.verify_signal_cross_exchange(
                        symbol, sig.direction.value, sig.entry
                    ),
                    timeout=3,
                )
            except asyncio.TimeoutError:
                log.debug(
                    "Cross-exchange verification timed out for %s", symbol
                )
            except Exception as exc:
                log.debug(
                    "Cross-exchange verification error for %s: %s", symbol, exc
                )

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
            min_conf = self.confidence_overrides.get(
                chan_name, chan.config.min_confidence
            )
            if sig.confidence < min_conf:
                continue

            # Attach regime info
            sig.market_phase = regime_result.regime.value

            # Populate liquidity info from SMC data
            liq_parts = []
            if smc_result.sweeps:
                sweep = smc_result.sweeps[0]
                liq_parts.append(
                    f"Sweep {sweep.direction.value} at {sweep.sweep_level:.4f}"
                )
            if smc_result.fvg:
                fvg = smc_result.fvg[0]
                liq_parts.append(
                    f"FVG {fvg.gap_high:.4f}-{fvg.gap_low:.4f}"
                )
            if liq_parts:
                sig.liquidity_info = " | ".join(liq_parts)

            # Attach market context for risk manager
            sig.spread_pct = spread_pct
            sig.volume_24h_usd = volume_24h

            # Start cooldown timer for this (symbol, channel) pair
            self._set_cooldown(symbol, chan_name)

            try:
                self.signal_queue.put_nowait(sig)
            except asyncio.QueueFull:
                log.warning("Signal queue full – dropping %s", sig.signal_id)

            # Select-mode: if enabled and signal passes stricter filters,
            # also enqueue a copy to 360_SELECT channel.
            # The original signal is always published to its regular channel.
            if (
                self.select_mode_filter is not None
                and self.select_mode_filter.enabled
            ):
                allowed, reason = self.select_mode_filter.should_publish(
                    signal=sig,
                    confidence=sig.confidence,
                    indicators=indicators,
                    smc_data=smc_data,
                    ai_sentiment=ai,
                    cross_exchange_verified=cross_verified,
                    volume_24h=volume_24h,
                    spread_pct=spread_pct,
                )
                if allowed:
                    select_sig = copy.deepcopy(sig)
                    select_sig.channel = "360_SELECT"
                    select_sig.signal_id = f"SELECT-{sig.signal_id}"
                    try:
                        self.signal_queue.put_nowait(select_sig)
                        log.info(
                            "SELECT copy enqueued for %s (%s)",
                            sig.symbol,
                            select_sig.signal_id,
                        )
                    except asyncio.QueueFull:
                        log.warning(
                            "Signal queue full – dropping SELECT copy %s",
                            select_sig.signal_id,
                        )
                else:
                    log.debug(
                        "SELECT filter rejected %s %s: %s",
                        sig.symbol,
                        chan_name,
                        reason,
                    )
