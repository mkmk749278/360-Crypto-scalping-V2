"""Scanner – periodic evaluation of all pairs across channel strategies.

Extracted from :class:`src.main.CryptoSignalEngine` for modularity.
Supports signal cooldown de-duplication, market-regime-aware gating,
and optional circuit-breaker integration.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
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
from src.onchain import score_onchain
from src.regime import MarketRegime
from src.signal_quality import (
    ExecutionAssessment,
    MarketState,
    PairQualityAssessment,
    RiskAssessment,
    SetupAssessment,
    assess_pair_quality,
    build_risk_plan,
    classify_market_state,
    classify_setup,
    execution_quality_check,
    score_signal_components,
)
from src.utils import get_logger, price_decimal_fmt

log = get_logger("scanner")

# Order book spread cache TTL and per-cycle fetch cap
_SPREAD_CACHE_TTL: float = 30.0
# Longer TTL for symbols that fail (e.g. futures-only on spot) to avoid
# hammering the endpoint every cycle.
_SPREAD_FAIL_CACHE_TTL: float = 300.0
_MAX_ORDER_BOOK_FETCHES_PER_CYCLE: int = 5

# ADX threshold below which SCALP signals are suppressed during RANGING regime
_RANGING_ADX_SUPPRESS_THRESHOLD: float = 15.0

# Confidence boost applied to RANGE channel when regime is RANGING
_RANGING_RANGE_CONF_BOOST: float = 5.0

# Confidence penalty applied to RANGE channel when ADX is in the borderline zone (20-25)
_RANGE_BORDERLINE_ADX_PENALTY: float = 10.0
_RANGE_BORDERLINE_ADX_LOW: float = 20.0
_RANGE_BORDERLINE_ADX_HIGH: float = 25.0

# Maximum number of symbols scanned concurrently
_MAX_CONCURRENT_SCANS: int = 10

# Regime-channel compatibility matrix.
# Maps channel name → list of regimes where that channel is blocked.
# SCALP needs movement: block in QUIET (nothing moves).
# SWING needs sustained trend: block in VOLATILE (chaotic, stops get swept).
_REGIME_CHANNEL_INCOMPATIBLE: Dict[str, List[str]] = {
    "360_SCALP": ["QUIET"],
    "360_SWING": ["VOLATILE", "DIRTY_RANGE"],
}


@dataclass
class ScanContext:
    candles: Dict[str, dict]
    indicators: Dict[str, dict]
    smc_result: Any
    smc_data: dict
    regime_result: Any
    ai: Dict[str, Any]
    spread_pct: float
    ind_for_predict: Dict[str, Any]
    is_ranging: bool
    adx_val: float
    onchain_data: Any
    candle_total: int
    pair_quality: PairQualityAssessment
    market_state: MarketState


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
        openai_evaluator: Optional[Any] = None,
        onchain_client: Optional[Any] = None,
    ) -> None:
        self.pair_mgr = pair_mgr
        self.data_store = data_store
        self.channels = channels
        self.smc_detector = smc_detector
        self.regime_detector = regime_detector
        self.predictive = predictive
        self.exchange_mgr = exchange_mgr
        self.spot_client: Optional[Any] = spot_client
        self.futures_client: Optional[Any] = None
        self.telemetry = telemetry
        self.signal_queue = signal_queue
        self.router = router
        self.openai_evaluator: Optional[Any] = openai_evaluator
        self.onchain_client: Optional[Any] = onchain_client

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

        # Order book spread cache: symbol → (spread_pct, expiry_monotonic_time)
        # expiry_monotonic_time is an absolute time.monotonic() value; the entry
        # is valid while time.monotonic() < expiry_monotonic_time.
        self._order_book_cache: Dict[str, Tuple[float, float]] = {}
        self._order_book_fetches_this_cycle: int = 0

        # Cooldown tracking: (symbol, channel_name) → monotonic expiry time
        self._cooldown_until: Dict[Tuple[str, str], float] = {}

        # Per-symbol cooldown after a stop-loss: prevents any channel from
        # firing on the same symbol for a short window after an SL event.
        self._symbol_sl_cooldown_until: Dict[str, float] = {}

        # Post-invalidation cooldown: (symbol, channel, direction) → monotonic expiry
        # Prevents rapid re-fire of the same thesis after invalidation.
        self._invalidation_cooldown_until: Dict[Tuple[str, str, str], float] = {}

        # Semaphore to limit concurrent symbol scans
        self._scan_semaphore: asyncio.Semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SCANS)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def scan_loop(self) -> None:
        """Periodic scan over all pairs / channels."""
        log.info("Scanner loop started")
        while True:
            t0 = time.monotonic()
            self._order_book_fetches_this_cycle = 0

            # Always clean up expired signals first (safety net for stuck slots)
            expired_count = self.router.cleanup_expired()
            if expired_count > 0:
                log.info("Cleaned up {} expired signals at start of scan cycle", expired_count)

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
                sem = self._scan_semaphore
                tasks = [
                    self._scan_symbol_bounded(sem, sym, info.volume_24h_usd)
                    for sym, info in sorted_pairs
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for sym_info, result in zip(sorted_pairs, results):
                    if isinstance(result, Exception):
                        log.warning(
                            "Scan error for {} ({}): {}",
                            sym_info[0], type(result).__name__, result,
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Scan loop error: {}", exc)

            elapsed_ms = (time.monotonic() - t0) * 1000
            self.telemetry.set_scan_latency(elapsed_ms)
            self.telemetry.set_pairs_monitored(len(self.pair_mgr.pairs))
            self.telemetry.set_active_signals(len(self.router.active_signals))
            try:
                qsize = await self.signal_queue.qsize()
            except Exception as exc:
                log.warning("Failed to read signal queue size: {}", exc)
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
            "Cooldown set for {} {} ({:.0f}s)", symbol, channel_name, cooldown_s
        )

    def set_symbol_sl_cooldown(self, symbol: str, duration_s: float = 60.0) -> None:
        """Apply a short cross-channel cooldown for *symbol* after a stop-loss.

        Called by the TradeMonitor when any signal for *symbol* hits its stop
        loss.  Prevents every other channel from immediately firing a new signal
        on the same symbol before the market has had time to stabilise.
        """
        self._symbol_sl_cooldown_until[symbol] = time.monotonic() + duration_s
        log.debug(
            "Per-symbol SL cooldown set for {} ({:.0f}s)", symbol, duration_s
        )

    # Post-invalidation cooldown durations per channel (seconds)
    _INVALIDATION_COOLDOWN_SECONDS: Dict[str, int] = {
        "360_THE_TAPE": 600,
        "360_SCALP": 300,
        "360_RANGE": 300,
        "360_SWING": 600,
        "360_SELECT": 600,
    }

    def set_invalidation_cooldown(
        self,
        symbol: str,
        channel: str,
        direction: str,
    ) -> None:
        """Apply a cooldown for (symbol, channel, direction) after invalidation.

        Prevents the same thesis from re-firing immediately after a signal is
        invalidated (e.g., EMA crossover kills BNBUSDT SHORT, then it fires again
        within minutes with the same parameters).
        """
        duration_s = self._INVALIDATION_COOLDOWN_SECONDS.get(channel, 300)
        key = (symbol, channel, direction)
        self._invalidation_cooldown_until[key] = time.monotonic() + duration_s
        log.debug(
            "Post-invalidation cooldown set for {} {} {} ({:.0f}s)",
            symbol, channel, direction, duration_s,
        )

    async def _scan_symbol_bounded(self, sem: asyncio.Semaphore, symbol: str, volume_24h: float) -> None:
        """Acquire *sem* then delegate to :meth:`_scan_symbol`."""
        async with sem:
            await self._scan_symbol(symbol, volume_24h)

    def _load_candles(self, symbol: str) -> Dict[str, dict]:
        candles: Dict[str, dict] = {}
        for tf in SEED_TIMEFRAMES:
            c = self.data_store.get_candles(symbol, tf.interval)
            if c:
                candles[tf.interval] = c
        return candles

    def _compute_indicators(self, candles: Dict[str, dict]) -> Dict[str, dict]:
        indicators: Dict[str, dict] = {}
        for tf_key, cd in candles.items():
            h = np.asarray(cd["high"], dtype=np.float64).ravel()
            lo = np.asarray(cd["low"], dtype=np.float64).ravel()
            c = np.asarray(cd["close"], dtype=np.float64).ravel()
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
        return indicators

    async def _fetch_ai_context(self, symbol: str) -> Dict[str, Any]:
        ai: Dict[str, Any] = {"label": "Neutral", "summary": "", "score": 0.0, "fear_greed_value": 50}
        try:
            insight = await asyncio.wait_for(get_ai_insight(symbol), timeout=2)
            ai = {
                "label": insight.label,
                "summary": insight.summary,
                "score": insight.score,
                "fear_greed_value": insight.fear_greed_value,
            }
        except Exception:
            pass
        return ai

    async def _get_spread_pct(self, symbol: str, market: str = "spot") -> float:
        spread_pct = 0.01  # fallback
        now = time.monotonic()
        cached = self._order_book_cache.get(symbol)
        if cached and now < cached[1]:
            return cached[0]
        if self._order_book_fetches_this_cycle >= _MAX_ORDER_BOOK_FETCHES_PER_CYCLE:
            return spread_pct
        try:
            self._order_book_fetches_this_cycle += 1
            if market == "futures":
                if self.futures_client is None:
                    self.futures_client = BinanceClient("futures")
                client = self.futures_client
            else:
                if self.spot_client is None:
                    self.spot_client = BinanceClient("spot")
                client = self.spot_client
            book = await client.fetch_order_book(symbol, limit=5)
            if book and book.get("bids") and book.get("asks"):
                best_bid = float(book["bids"][0][0])
                best_ask = float(book["asks"][0][0])
                mid = (best_bid + best_ask) / 2.0
                if mid > 0:
                    spread_pct = (best_ask - best_bid) / mid * 100.0
                # Successful fetch: cache with normal TTL
                self._order_book_cache[symbol] = (spread_pct, now + _SPREAD_CACHE_TTL)
            else:
                # Failed fetch (e.g. futures-only symbol on spot endpoint):
                # cache with a longer TTL to avoid hammering the endpoint every cycle.
                self._order_book_cache[symbol] = (spread_pct, now + _SPREAD_FAIL_CACHE_TTL)
        except Exception:
            self._order_book_cache[symbol] = (spread_pct, now + _SPREAD_FAIL_CACHE_TTL)
        return spread_pct

    async def _fetch_onchain_data(self, symbol: str) -> Any:
        try:
            if self.onchain_client is not None:
                return await asyncio.wait_for(
                    self.onchain_client.get_exchange_flow(symbol),
                    timeout=3,
                )
        except Exception as exc:
            log.debug("On-chain fetch error for {}: {}", symbol, exc)
        return None

    async def _verify_cross_exchange(
        self, symbol: str, direction: str, entry: float
    ) -> Optional[bool]:
        try:
            return await asyncio.wait_for(
                self.exchange_mgr.verify_signal_cross_exchange(
                    symbol, direction, entry
                ),
                timeout=3,
            )
        except asyncio.TimeoutError:
            log.debug("Cross-exchange verification timed out for {}", symbol)
        except Exception as exc:
            log.debug("Cross-exchange verification error for {}: {}", symbol, exc)
        return None

    def _build_smc_summary(self, smc_result: Any) -> str:
        smc_parts = []
        if smc_result.sweeps:
            sweep = smc_result.sweeps[0]
            fmt = price_decimal_fmt(sweep.sweep_level)
            smc_parts.append(
                f"Sweep {sweep.direction.value} at {sweep.sweep_level:{fmt}}"
            )
        if smc_result.fvg:
            fvg = smc_result.fvg[0]
            fmt = price_decimal_fmt(max(fvg.gap_high, fvg.gap_low))
            smc_parts.append(f"FVG {fvg.gap_high:{fmt}}-{fvg.gap_low:{fmt}}")
        return " | ".join(smc_parts) if smc_parts else "None detected"

    async def _build_scan_context(self, symbol: str, volume_24h: float) -> Optional[ScanContext]:
        candles = self._load_candles(symbol)
        if not candles:
            return None
        indicators = self._compute_indicators(candles)
        ticks = self.data_store.ticks.get(symbol, [])
        smc_result = self.smc_detector.detect(symbol, candles, ticks)
        smc_data = smc_result.as_dict()

        regime_ind = indicators.get("5m", indicators.get("1m", {}))
        regime_candles = candles.get("5m", candles.get("1m"))
        regime_result = self.regime_detector.classify(regime_ind, regime_candles)
        log.debug("{} regime: {}", symbol, regime_result.regime.value)

        ind_for_predict = indicators.get("5m", indicators.get("1m", {}))
        candle_total = sum(len(cd.get("close", [])) for cd in candles.values())
        market = (
            self.pair_mgr.pairs[symbol].market
            if symbol in self.pair_mgr.pairs
            else "spot"
        )
        ai, spread_pct, onchain_data = await asyncio.gather(
            self._fetch_ai_context(symbol),
            self._get_spread_pct(symbol, market=market),
            self._fetch_onchain_data(symbol),
        )
        pair_quality = assess_pair_quality(
            volume_24h=volume_24h,
            spread_pct=spread_pct,
            indicators=regime_ind,
            candles=regime_candles,
        )
        market_state = classify_market_state(
            regime_result=regime_result,
            indicators=regime_ind,
            candles=regime_candles,
            spread_pct=spread_pct,
        )
        return ScanContext(
            candles=candles,
            indicators=indicators,
            smc_result=smc_result,
            smc_data=smc_data,
            regime_result=regime_result,
            ai=ai,
            spread_pct=spread_pct,
            ind_for_predict=ind_for_predict,
            is_ranging=regime_result.regime == MarketRegime.RANGING,
            adx_val=regime_ind.get("adx_last") or 0,
            onchain_data=onchain_data,
            candle_total=candle_total,
            pair_quality=pair_quality,
            market_state=market_state,
        )

    def _should_skip_channel(self, symbol: str, chan_name: str, ctx: ScanContext) -> bool:
        if not ctx.pair_quality.passed:
            log.debug(
                "Skipping {} {} – pair quality gate failed: {}",
                symbol,
                chan_name,
                ctx.pair_quality.reason,
            )
            return True
        if ctx.market_state == MarketState.VOLATILE_UNSUITABLE:
            log.debug(
                "Skipping {} {} – volatile/unsuitable market state",
                symbol,
                chan_name,
            )
            return True
        if chan_name in self.paused_channels:
            return True
        if self._is_in_cooldown(symbol, chan_name):
            log.debug("Cooldown active: skipping {} {}", symbol, chan_name)
            return True
        # Per-symbol SL cooldown: suppress all channels briefly after an SL event
        sl_expiry = self._symbol_sl_cooldown_until.get(symbol)
        if sl_expiry is not None:
            if time.monotonic() < sl_expiry:
                log.debug(
                    "Per-symbol SL cooldown active: skipping {} {}", symbol, chan_name
                )
                return True
            del self._symbol_sl_cooldown_until[symbol]
        if any(
            s.symbol == symbol and s.channel == chan_name
            for s in self.router.active_signals.values()
        ):
            log.debug("Skipping {} {} – active signal already exists", symbol, chan_name)
            return True
        if (
            chan_name == "360_SCALP"
            and ctx.is_ranging
            and ctx.adx_val < _RANGING_ADX_SUPPRESS_THRESHOLD
        ):
            log.debug(
                "Suppressing SCALP signal for {} (RANGING, ADX={:.1f})",
                symbol,
                ctx.adx_val,
            )
            return True
        # Regime-channel compatibility matrix
        current_regime = ctx.regime_result.regime.value
        incompatible_regimes = _REGIME_CHANNEL_INCOMPATIBLE.get(chan_name, [])
        if current_regime in incompatible_regimes:
            log.debug(
                "Suppressing {} signal for {} (regime {} incompatible with channel)",
                chan_name,
                symbol,
                current_regime,
            )
            return True
        return False

    def _evaluate_setup(
        self,
        chan_name: str,
        sig: Any,
        ctx: ScanContext,
    ) -> SetupAssessment:
        return classify_setup(
            channel_name=chan_name,
            signal=sig,
            indicators=ctx.indicators,
            smc_data=ctx.smc_data,
            market_state=ctx.market_state,
        )

    def _evaluate_execution(
        self,
        sig: Any,
        ctx: ScanContext,
        setup: SetupAssessment,
    ) -> ExecutionAssessment:
        return execution_quality_check(
            signal=sig,
            indicators=ctx.indicators,
            smc_data=ctx.smc_data,
            setup=setup.setup_class,
            market_state=ctx.market_state,
        )

    def _evaluate_risk(
        self,
        sig: Any,
        ctx: ScanContext,
        setup: SetupAssessment,
    ) -> RiskAssessment:
        return build_risk_plan(
            signal=sig,
            indicators=ctx.indicators,
            candles=ctx.candles,
            smc_data=ctx.smc_data,
            setup=setup.setup_class,
            spread_pct=ctx.spread_pct,
        )

    def _apply_risk_plan_to_signal(
        self,
        sig: Any,
        risk: RiskAssessment,
    ) -> None:
        sig.stop_loss = risk.stop_loss
        sig.tp1 = risk.tp1
        sig.tp2 = risk.tp2
        sig.tp3 = risk.tp3
        sig.invalidation_summary = risk.invalidation_summary

    def _compute_base_confidence(
        self,
        symbol: str,
        volume_24h: float,
        sig: Any,
        ctx: ScanContext,
        cross_verified: Optional[bool],
    ) -> Optional[float]:
        has_sweep = bool(ctx.smc_data["sweeps"])
        has_mss = ctx.smc_data["mss"] is not None
        has_fvg = bool(ctx.smc_data["fvg"])
        ema_aligned = (
            ctx.ind_for_predict.get("ema9_last") is not None
            and ctx.ind_for_predict.get("ema21_last") is not None
            and (
                (ctx.ind_for_predict["ema9_last"] > ctx.ind_for_predict["ema21_last"])
                if sig.direction.value == "LONG"
                else (ctx.ind_for_predict["ema9_last"] < ctx.ind_for_predict["ema21_last"])
            )
        )
        adx_ok = (ctx.ind_for_predict.get("adx_last") or 0) >= 20
        mom_positive = (
            (ctx.ind_for_predict.get("momentum_last") or 0) > 0
            if sig.direction.value == "LONG"
            else (ctx.ind_for_predict.get("momentum_last") or 0) < 0
        )

        # Compute sweep depth percentage for gradient SMC scoring
        sweep_depth_pct = 0.0
        if ctx.smc_data["sweeps"]:
            sweep = ctx.smc_data["sweeps"][0]
            if hasattr(sweep, "sweep_level") and hasattr(sweep, "close_price"):
                ref_price = sweep.close_price if sweep.close_price > 0 else max(sig.entry, 1e-8)
                sweep_depth_pct = abs(sweep.sweep_level - sweep.close_price) / ref_price * 100.0

        # Compute FVG size relative to ATR for gradient SMC scoring
        fvg_atr_ratio = 0.0
        if ctx.smc_data["fvg"]:
            fvg = ctx.smc_data["fvg"][0]
            if hasattr(fvg, "gap_high") and hasattr(fvg, "gap_low"):
                fvg_size = abs(fvg.gap_high - fvg.gap_low)
                atr_val = ctx.ind_for_predict.get("atr_last")
                if atr_val and atr_val > 0:
                    fvg_atr_ratio = fvg_size / atr_val

        cinp = ConfidenceInput(
            smc_score=score_smc(
                has_sweep, has_mss, has_fvg,
                sweep_depth_pct=sweep_depth_pct,
                fvg_atr_ratio=fvg_atr_ratio,
            ),
            trend_score=score_trend(
                ema_aligned, adx_ok, mom_positive,
                adx_value=ctx.ind_for_predict.get("adx_last") or 0.0,
                momentum_strength=ctx.ind_for_predict.get("momentum_last") or 0.0,
            ),
            ai_sentiment_score=score_ai_sentiment(
                ctx.ai.get("score", 0),
                sig.direction.value,
                fear_greed_value=int(ctx.ai.get("fear_greed_value", 50)),
            ),
            liquidity_score=score_liquidity(volume_24h),
            spread_score=score_spread(ctx.spread_pct),
            data_sufficiency=score_data_sufficiency(ctx.candle_total),
            multi_exchange=score_multi_exchange(verified=cross_verified),
            onchain_score=score_onchain(ctx.onchain_data),
            has_enough_history=self.pair_mgr.has_enough_history(symbol),
            opposing_position_open=any(
                s.symbol == symbol and s.direction.value != sig.direction.value
                for s in self.router.active_signals.values()
            ),
        )
        result = compute_confidence(cinp)
        if result.blocked:
            return None
        return result.total

    def _apply_regime_channel_adjustments(
        self,
        symbol: str,
        chan_name: str,
        sig: Any,
        ctx: ScanContext,
    ) -> None:
        if chan_name == "360_RANGE" and ctx.is_ranging:
            sig.confidence += _RANGING_RANGE_CONF_BOOST
            log.debug(
                "RANGE confidence boosted for {} (RANGING): {:.1f}",
                symbol,
                sig.confidence,
            )

    async def _apply_predictive_adjustments(
        self,
        symbol: str,
        sig: Any,
        ctx: ScanContext,
    ) -> None:
        try:
            prediction = await self.predictive.predict(
                symbol, ctx.candles, ctx.ind_for_predict
            )
            self.predictive.adjust_tp_sl(sig, prediction)
            self.predictive.update_confidence(sig, prediction)
        except Exception as exc:
            log.debug("Predictive AI error for {}: {}", symbol, exc)

    async def _apply_openai_adjustments(
        self,
        symbol: str,
        chan_name: str,
        sig: Any,
        ctx: ScanContext,
    ) -> bool:
        if not (self.openai_evaluator and self.openai_evaluator.enabled):
            return True
        try:
            openai_eval = await asyncio.wait_for(
                self.openai_evaluator.evaluate(
                    symbol=symbol,
                    direction=sig.direction.value,
                    channel=chan_name,
                    entry_price=sig.entry,
                    stop_loss=sig.stop_loss,
                    tp1=sig.tp1,
                    tp2=sig.tp2,
                    indicators=ctx.ind_for_predict,
                    smc_summary=self._build_smc_summary(ctx.smc_result),
                    ai_sentiment_summary=ctx.ai.get("summary", ""),
                    market_phase=ctx.regime_result.regime.value,
                    confidence_before=sig.confidence,
                ),
                timeout=6,
            )
            if openai_eval and not openai_eval.recommended:
                log.info(
                    "OpenAI recommends SKIP for {} {}: {}",
                    symbol,
                    chan_name,
                    openai_eval.reasoning,
                )
                return False
            if openai_eval and openai_eval.adjustment != 0.0:
                sig.confidence += openai_eval.adjustment
                log.debug(
                    "OpenAI adjusted confidence for {} {} by {:+.1f} → {:.1f} ({})",
                    symbol,
                    chan_name,
                    openai_eval.adjustment,
                    sig.confidence,
                    openai_eval.reasoning,
                )
        except Exception as exc:
            log.debug("OpenAI evaluation error for {}: {}", symbol, exc)
        return True

    @staticmethod
    def _clamp_confidence(value: float) -> float:
        return max(0.0, min(100.0, round(value, 2)))

    def _populate_signal_context(self, sig: Any, volume_24h: float, ctx: ScanContext) -> None:
        sig.market_phase = ctx.market_state.value
        liq_parts = []
        if ctx.smc_result.sweeps:
            sweep = ctx.smc_result.sweeps[0]
            fmt = price_decimal_fmt(sweep.sweep_level)
            liq_parts.append(
                f"Sweep {sweep.direction.value} at {sweep.sweep_level:{fmt}}"
            )
        if ctx.smc_result.fvg:
            fvg = ctx.smc_result.fvg[0]
            fmt = price_decimal_fmt(max(fvg.gap_high, fvg.gap_low))
            liq_parts.append(f"FVG {fvg.gap_high:{fmt}}-{fvg.gap_low:{fmt}}")
        if liq_parts:
            sig.liquidity_info = " | ".join(liq_parts)
        sig.spread_pct = ctx.spread_pct
        sig.volume_24h_usd = volume_24h
        sig.pair_quality_score = ctx.pair_quality.score
        sig.pair_quality_label = ctx.pair_quality.label

    @staticmethod
    def _has_higher_timeframe_alignment(sig: Any, indicators: Dict[str, Dict[str, Any]]) -> bool:
        for tf in ("15m", "1h", "4h"):
            ind = indicators.get(tf, {})
            ema9 = ind.get("ema9_last")
            ema21 = ind.get("ema21_last")
            if ema9 is None or ema21 is None:
                continue
            if sig.direction.value == "LONG" and ema9 < ema21:
                return False
            if sig.direction.value == "SHORT" and ema9 > ema21:
                return False
        return True

    async def _enqueue_signal(self, sig: Any) -> bool:
        return await self.signal_queue.put(sig)

    async def _prepare_signal(
        self,
        symbol: str,
        volume_24h: float,
        chan: Any,
        ctx: ScanContext,
    ) -> Tuple[Optional[Any], Optional[bool]]:
        chan_name = chan.config.name
        try:
            sig = chan.evaluate(
                symbol=symbol,
                candles=ctx.candles,
                indicators=ctx.indicators,
                smc_data=ctx.smc_data,
                ai_insight=ctx.ai,
                spread_pct=ctx.spread_pct,
                volume_24h_usd=volume_24h,
            )
        except Exception as exc:
            log.debug("Channel {} eval error for {}: {}", chan_name, symbol, exc)
            return None, None
        if sig is None:
            return None, None

        # Post-invalidation cooldown: suppress same (symbol, channel, direction) thesis
        inv_key = (symbol, chan_name, sig.direction.value)
        inv_expiry = self._invalidation_cooldown_until.get(inv_key)
        if inv_expiry is not None:
            if time.monotonic() < inv_expiry:
                log.debug(
                    "Post-invalidation cooldown: skipping {} {} {}",
                    symbol, chan_name, sig.direction.value,
                )
                return None, None
            del self._invalidation_cooldown_until[inv_key]

        setup = self._evaluate_setup(chan_name, sig, ctx)
        if not setup.channel_compatible or not setup.regime_compatible:
            log.debug("Rejected {} {} setup: {}", symbol, chan_name, setup.reason)
            return None, None

        execution = self._evaluate_execution(sig, ctx, setup)
        if not execution.passed:
            log.debug("Rejected {} {} execution: {}", symbol, chan_name, execution.reason)
            return None, None

        risk = self._evaluate_risk(sig, ctx, setup)
        if not risk.passed:
            log.debug("Rejected {} {} risk: {}", symbol, chan_name, risk.reason)
            return None, None
        self._apply_risk_plan_to_signal(sig, risk)

        cross_verified = await self._verify_cross_exchange(
            symbol, sig.direction.value, sig.entry
        )
        legacy_confidence = self._compute_base_confidence(
            symbol,
            volume_24h,
            sig,
            ctx,
            cross_verified,
        )
        if legacy_confidence is None:
            return None, cross_verified
        sig.confidence = legacy_confidence
        self._apply_regime_channel_adjustments(symbol, chan_name, sig, ctx)
        await self._apply_predictive_adjustments(symbol, sig, ctx)
        setup_score = score_signal_components(
            pair_quality=ctx.pair_quality,
            setup=setup,
            execution=execution,
            risk=risk,
            legacy_confidence=sig.confidence,
            cross_verified=cross_verified,
        )
        sig.setup_class = setup.setup_class.value
        sig.analyst_reason = setup.thesis
        sig.execution_note = execution.execution_note
        sig.entry_zone = execution.entry_zone
        sig.component_scores = setup_score.components
        sig.quality_tier = setup_score.quality_tier.value
        sig.pre_ai_confidence = setup_score.total
        sig.confidence = setup_score.total
        # Apply confidence penalty for RANGE signals in the borderline ADX zone (20-25).
        # The channel now allows ADX up to 25, but a trending-leaning environment
        # warrants a penalty to reflect reduced signal quality.
        # Intentionally asymmetric: ADX <= 20 is clean range (no penalty); only
        # ADX strictly above 20 up to and including 25 triggers the penalty.
        if (
            chan_name == "360_RANGE"
            and ctx.adx_val is not None
            and _RANGE_BORDERLINE_ADX_LOW < ctx.adx_val <= _RANGE_BORDERLINE_ADX_HIGH
        ):
            sig.confidence -= _RANGE_BORDERLINE_ADX_PENALTY
            log.debug(
                "RANGE borderline ADX penalty for {} (ADX={:.1f}): {:.1f}",
                symbol,
                ctx.adx_val,
                sig.confidence,
            )
        if not await self._apply_openai_adjustments(symbol, chan_name, sig, ctx):
            return None, cross_verified
        sig.confidence = self._clamp_confidence(sig.confidence)
        sig.post_ai_confidence = sig.confidence
        min_conf = self.confidence_overrides.get(chan_name, chan.config.min_confidence)
        if (
            sig.confidence < min_conf
            or sig.component_scores.get("market", 0.0) < 12.0
            or sig.component_scores.get("execution", 0.0) < 10.0
            or sig.component_scores.get("risk", 0.0) < 10.0
        ):
            return None, cross_verified
        self._populate_signal_context(sig, volume_24h, ctx)
        return sig, cross_verified

    async def _scan_symbol(self, symbol: str, volume_24h: float) -> None:
        """Run all channel evaluations for one symbol."""
        ctx = await self._build_scan_context(symbol, volume_24h)
        if ctx is None:
            return
        for chan in self.channels:
            chan_name = chan.config.name
            if self._should_skip_channel(symbol, chan_name, ctx):
                continue
            sig, cross_verified = await self._prepare_signal(symbol, volume_24h, chan, ctx)
            if sig is None:
                continue
            # Only start scan cooldown after the signal has been accepted by the
            # queue; rejected/dropped signals must not suppress later scans.
            if not await self._enqueue_signal(sig):
                continue
            self._set_cooldown(symbol, chan_name)

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
                    indicators=ctx.indicators,
                    smc_data=ctx.smc_data,
                    ai_sentiment=ctx.ai,
                    cross_exchange_verified=cross_verified,
                    volume_24h=volume_24h,
                    spread_pct=ctx.spread_pct,
                    setup_class=sig.setup_class,
                    market_state=sig.market_phase,
                    quality_tier=sig.quality_tier,
                    component_scores=sig.component_scores,
                    pair_quality_score=sig.pair_quality_score,
                    r_multiple=sig.r_multiple,
                    higher_timeframe_aligned=self._has_higher_timeframe_alignment(sig, ctx.indicators),
                )
                if allowed:
                    select_sig = copy.deepcopy(sig)
                    select_sig.channel = "360_SELECT"
                    select_sig.signal_id = f"SELECT-{sig.signal_id}"
                    if await self._enqueue_signal(select_sig):
                        log.info(
                            "SELECT copy enqueued for {} ({})",
                            sig.symbol,
                            select_sig.signal_id,
                        )
                else:
                    log.debug(
                        "SELECT filter rejected {} {}: {}",
                        sig.symbol,
                        chan_name,
                        reason,
                    )
