"""Scanner – periodic evaluation of all pairs across channel strategies.

Extracted from :class:`src.main.CryptoSignalEngine` for modularity.
Supports signal cooldown de-duplication, market-regime-aware gating,
and optional circuit-breaker integration.
"""

from __future__ import annotations

import asyncio
import dataclasses as _dc
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import uuid

from config import (
    CHANNEL_GEM,
    MAX_CORRELATED_SCALP_SIGNALS,
    MTF_HARD_BLOCK,
    QUIET_SCALP_MIN_CONFIDENCE,
    SCAN_LATENCY_ALERT_CONSECUTIVE,
    SCAN_LATENCY_ALERT_MS,
    SCAN_LATENCY_REDUCE_MS,
    SCAN_LATENCY_WARN_MS,
    SCAN_MIN_VOLUME_USD,
    SEED_TIMEFRAMES,
    SIGNAL_SCAN_COOLDOWN_SECONDS,
    SIGNAL_VALID_FOR_MINUTES,
    SMC_SCALP_LOOKBACK,
    SMC_SCALP_TOLERANCE_PCT,
    TIER2_SCAN_EVERY_N_CYCLES,
    TIER3_SCAN_EVERY_N_CYCLES,
    TIER3_SCAN_INTERVAL_MINUTES,
    WS_DEGRADED_CYCLES_ALERT,
    WS_DEGRADED_MAX_PAIRS,
    WS_PARTIAL_HEALTH_THRESHOLD,
)
from src.binance import BinanceClient
from src.channels.base import Signal as _Signal
from src.smc import Direction
from src.rate_limiter import rate_limiter, futures_rate_limiter
from src.confidence import (
    ConfidenceInput,
    compute_confidence,
    score_data_sufficiency,
    score_liquidity,
    score_multi_exchange,
    score_order_flow,
    score_smc,
    score_spread,
    score_trend,
)
from src.indicators import adx, atr, bollinger_bands, ema, macd, momentum, rsi
from src.onchain import score_onchain
from src.regime import MarketRegime
from src.signal_quality import (
    ExecutionAssessment,
    MarketState,
    PairQualityAssessment,
    RiskAssessment,
    SetupAssessment,
    SignalScoringEngine,
    ScoringInput,
    assess_pair_quality,
    assess_pair_quality_for_channel,
    build_risk_plan,
    classify_market_state,
    classify_setup,
    execution_quality_check,
    score_signal_components,
)
from src.cluster_suppression import ClusterSuppressor
from src.confidence_decay import apply_confidence_decay
from src.cross_asset import AssetState, check_cross_asset_gate
from src.feedback_loop import FeedbackLoop
from src.kill_zone import check_kill_zone_gate
from src.mtf import check_mtf_gate, compute_mtf_confluence, _TF_WEIGHTS as _MTF_TF_WEIGHTS
from src.oi_filter import analyse_oi, check_oi_gate
from src.pair_manager import PairTier, classify_pair_tier
from src.spoof_detect import check_spoof_gate
from src.tier_manager import TierManager
from src.utils import get_logger, price_decimal_fmt, utcnow
from src.volume_divergence import check_volume_divergence_gate
from src.vwap import check_vwap_extension, compute_vwap
from src.ai_engine import get_ai_insight
from src.chart_patterns import detect_patterns, pattern_confidence_bonus, detect_all_patterns
from src.stat_filter import StatisticalFilter
from src.suppression_telemetry import (
    SuppressionTracker,
    SuppressionEvent,
    REASON_QUIET_REGIME,
    REASON_SPREAD_GATE,
    REASON_VOLUME_GATE,
    REASON_OI_INVALIDATION,
    REASON_CLUSTER,
    REASON_STAT_FILTER,
    REASON_LIFESPAN,
    REASON_CONFIDENCE,
)

log = get_logger("scanner")

# Composite signal scoring engine — instantiated once at module level.
_scoring_engine = SignalScoringEngine()

# Statistical filter — tracks rolling win rates per (channel, pair, regime)
# and suppresses or penalises signals from historically poor combinations.
_stat_filter = StatisticalFilter()

# Order book spread cache TTL and per-cycle fetch cap
_SPREAD_CACHE_TTL: float = 30.0
# Longer TTL for symbols that fail (e.g. futures-only on spot) to avoid
# hammering the endpoint every cycle.
_SPREAD_FAIL_CACHE_TTL: float = 300.0
# Upper bound on order book fetches per scan cycle.  The rate limiter in
# BinanceClient throttles requests globally, so this acts only as a final
# safety net — not the primary budget enforcement mechanism.
_MAX_ORDER_BOOK_FETCHES_PER_CYCLE: int = 50
# Minimum remaining rate-limiter weight required before issuing another order
# book fetch.  Each order book call costs 1 weight; we reserve 100 units as a
# buffer for kline fetches, exchange-info calls, and WebSocket reconnects that
# may occur concurrently — roughly 8% of the 1 200-weight Binance limit.
_MIN_WEIGHT_FOR_ORDER_BOOK: int = 100
# When WS is partially degraded, tighten the order book limits to avoid
# burning REST budget on depth fetches across hundreds of pairs.
_MAX_ORDER_BOOK_FETCHES_WS_DEGRADED: int = 10
_MIN_WEIGHT_FOR_ORDER_BOOK_WS_DEGRADED: int = 400

# Timeout for the global bookTicker pre-fetch issued when WS is degraded.
_BOOK_TICKER_PREFETCH_TIMEOUT_S: float = 8.0

# TTL for spread entries seeded from the global bookTicker endpoint.
# bookTicker returns only bid/ask (no depth), so we keep the TTL shorter
# than the standard depth-based entry to encourage fresher polling.
_BOOK_TICKER_CACHE_TTL: float = 20.0

# ADX threshold below which SCALP signals receive a soft penalty during RANGING regime
_RANGING_ADX_SUPPRESS_THRESHOLD: float = float(
    os.getenv("RANGING_ADX_SUPPRESS_THRESHOLD", "12.0")
)
_RANGING_LOW_ADX_CONF_PENALTY: float = float(
    os.getenv("REGIME_RANGING_PENALTY", "5.0")
)

# Confidence boost applied to SCALP RANGE_FADE setup class when regime is RANGING
_RANGING_RANGE_FADE_CONF_BOOST: float = 5.0

# Chart pattern direction sets (used by scanner for chart_pattern_names population)
_CHART_BULLISH_PATTERNS: frozenset = frozenset({"DOUBLE_BOTTOM", "ASCENDING_TRIANGLE"})
_CHART_BEARISH_PATTERNS: frozenset = frozenset({"DOUBLE_TOP", "DESCENDING_TRIANGLE"})

# SCALP channel names — used for fast-path logic (skip cross-exchange verification).
_SCALP_CHANNELS: frozenset = frozenset({
    "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD",
    "360_SCALP_VWAP", "360_SCALP_OBI",
})

# Maximum number of symbols scanned concurrently
_MAX_CONCURRENT_SCANS: int = 15

# Regime-channel compatibility matrix.
# Maps channel name → list of regimes where that channel is blocked.
# SCALP channels (except VWAP) are no longer hard-blocked in QUIET — they
# instead receive a higher soft-gate penalty (_SCALP_QUIET_REGIME_PENALTY)
# and must meet a minimum confidence threshold (QUIET_SCALP_MIN_CONFIDENCE).
# VWAP remains blocked in QUIET because VWAP signals are meaningless without
# sufficient trading volume to anchor the indicator.
# SWING needs sustained trend: block in VOLATILE (chaotic, stops get swept).
_REGIME_CHANNEL_INCOMPATIBLE: Dict[str, List[str]] = {
    "360_SCALP_VWAP": ["QUIET"],
    "360_SWING":      ["VOLATILE", "DIRTY_RANGE"],
    "360_SPOT":       ["DIRTY_RANGE"],
}

# Penalty multiplier applied to scalp-channel soft gates when the regime is
# QUIET.  Higher than the default QUIET multiplier (0.8) to ensure only
# top-tier signals — genuine mean-reversion setups — pass through.
_SCALP_QUIET_REGIME_PENALTY: float = 1.8

# Penalty multiplier applied to soft-gate base penalties depending on live market regime.
# Trending markets → lenient (clear direction, fewer false signals).
# Volatile markets → strict (high chaos, amplify quality gates).
_REGIME_PENALTY_MULTIPLIER: Dict[str, float] = {
    "TRENDING_UP":   0.6,   # Strong trend = clear direction, lenient penalties
    "TRENDING_DOWN": 0.6,   # Same — sustained trend, gates matter less
    "RANGING":       1.0,   # Mean-reversion market, all quality gates at full weight
    "VOLATILE":      1.5,   # High chaos = more false signals, amplify penalties
    "QUIET":         0.8,   # Low volume but stable, gates fire often on thin data
}

# Regime-specific MTF confluence configuration.
# min_score   — the minimum passing score for the hard MTF gate.
# higher_tf_weight — multiplier applied to 4h/1d weights (trend confirmation).
# lower_tf_weight  — multiplier applied to 1m/5m weights (entry precision).
# In TRENDING regimes the higher-TF alignment is critical; in RANGING the
# lower-TF precision matters more; in VOLATILE, MTF is relaxed because
# timeframes often diverge during volatile markets.
_MTF_REGIME_CONFIG: Dict[str, Dict[str, float]] = {
    "TRENDING_UP":   {"min_score": 0.6, "higher_tf_weight": 1.5, "lower_tf_weight": 0.8},
    "TRENDING_DOWN": {"min_score": 0.6, "higher_tf_weight": 1.5, "lower_tf_weight": 0.8},
    "RANGING":       {"min_score": 0.3, "higher_tf_weight": 0.7, "lower_tf_weight": 1.4},
    "VOLATILE":      {"min_score": 0.2, "higher_tf_weight": 1.0, "lower_tf_weight": 1.0},
    "QUIET":         {"min_score": 0.4, "higher_tf_weight": 0.8, "lower_tf_weight": 1.2},
}

# Per-channel SMC timeframe preference order.
# SCALP → low-TF sweeps are valid entry triggers.
# SWING → high-TF institutional sweeps only.
# SPOT  → institutional only (4h+).
# Channels not listed here use the detector's default order.
_CHANNEL_SMC_TIMEFRAMES: Dict[str, tuple[str, ...]] = {
    "360_SCALP":      ("1m", "5m", "15m"),
    "360_SCALP_FVG":  ("5m", "15m"),
    "360_SCALP_CVD":  ("5m", "15m"),
    "360_SCALP_VWAP": ("5m", "15m"),
    "360_SCALP_OBI":  ("5m", "15m"),
    "360_SWING":      ("4h", "1h", "15m"),
    "360_SPOT":       ("4h", "1h"),
}

# Which gates are active per channel family.
# True = gate runs normally, False = gate is skipped entirely.
# Channels not listed default to all-True (fail-safe).
_CHANNEL_GATE_PROFILE: Dict[str, Dict[str, bool]] = {
    # SCALP channels: ALL gates active — microstructure matters at 1m/5m
    "360_SCALP":      {"mtf": True,  "vwap": True,  "kill_zone": True,  "oi": True,  "cross_asset": True,  "spoof": True,  "volume_div": True,  "cluster": True},
    "360_SCALP_FVG":  {"mtf": True,  "vwap": True,  "kill_zone": True,  "oi": True,  "cross_asset": True,  "spoof": True,  "volume_div": True,  "cluster": True},
    "360_SCALP_CVD":  {"mtf": True,  "vwap": True,  "kill_zone": True,  "oi": True,  "cross_asset": True,  "spoof": True,  "volume_div": True,  "cluster": True},
    "360_SCALP_VWAP": {"mtf": True,  "vwap": True,  "kill_zone": True,  "oi": True,  "cross_asset": True,  "spoof": True,  "volume_div": True,  "cluster": True},
    "360_SCALP_OBI":  {"mtf": True,  "vwap": True,  "kill_zone": True,  "oi": True,  "cross_asset": True,  "spoof": True,  "volume_div": True,  "cluster": True},
    # SWING: trend gates matter, microstructure does not
    "360_SWING":      {"mtf": True,  "vwap": True,  "kill_zone": False, "oi": True,  "cross_asset": True,  "spoof": False, "volume_div": True,  "cluster": False},
    # SPOT: macro gates only — intraday noise is irrelevant at 4h/1d
    "360_SPOT":       {"mtf": True,  "vwap": False, "kill_zone": False, "oi": False, "cross_asset": True,  "spoof": False, "volume_div": False, "cluster": False},
    # GEM: almost no gates — macro reversal scanner
    "360_GEM":        {"mtf": False, "vwap": False, "kill_zone": False, "oi": False, "cross_asset": False, "spoof": False, "volume_div": False, "cluster": False},
}

# Per-channel soft penalty base weights.
# These override the hard-coded defaults in _prepare_signal().
# Gates not listed use the original fallback values.
_CHANNEL_PENALTY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "360_SCALP":      {"vwap": 15.0, "kill_zone": 10.0, "oi": 8.0,  "volume_div": 12.0, "cluster": 10.0, "spoof": 12.0},
    "360_SCALP_FVG":  {"vwap": 15.0, "kill_zone": 10.0, "oi": 8.0,  "volume_div": 12.0, "cluster": 10.0, "spoof": 12.0},
    "360_SCALP_CVD":  {"vwap": 12.0, "kill_zone": 8.0,  "oi": 10.0, "volume_div": 10.0, "cluster": 10.0, "spoof": 10.0},
    "360_SCALP_VWAP": {"vwap": 18.0, "kill_zone": 8.0,  "oi": 6.0,  "volume_div": 10.0, "cluster": 10.0, "spoof": 10.0},
    "360_SCALP_OBI":  {"vwap": 12.0, "kill_zone": 8.0,  "oi": 6.0,  "volume_div": 10.0, "cluster": 10.0, "spoof": 15.0},
    "360_SWING":      {"vwap": 10.0, "kill_zone": 0.0,  "oi": 10.0, "volume_div": 8.0,  "cluster": 0.0,  "spoof": 0.0},
    "360_SPOT":       {"vwap": 0.0,  "kill_zone": 0.0,  "oi": 0.0,  "volume_div": 0.0,  "cluster": 0.0,  "spoof": 0.0},
    "360_GEM":        {"vwap": 0.0,  "kill_zone": 0.0,  "oi": 0.0,  "volume_div": 0.0,  "cluster": 0.0,  "spoof": 0.0},
}


def _normalize_candle_dict(cd: dict) -> dict:
    """Ensure all array-like values in a candle dict are flat 1-D Python lists.

    Candle data can occasionally arrive from the data store as 2-D numpy arrays
    of shape ``(n, 1)`` instead of the expected 1-D shape ``(n,)``.  This causes
    ``ValueError: The truth value of an array with more than one element is
    ambiguous`` in any downstream code that uses the array in a boolean context
    (e.g. ``if closes:``).  Converting everything to a plain Python list at the
    data-store boundary protects all downstream consumers.
    """
    normalized: dict = {}
    for key, val in cd.items():
        if isinstance(val, np.ndarray):
            normalized[key] = np.asarray(val, dtype=np.float64).ravel().tolist()
        elif isinstance(val, list):
            try:
                normalized[key] = np.asarray(val, dtype=np.float64).ravel().tolist()
            except (ValueError, TypeError) as exc:
                log.debug("_normalize_candle_dict: could not flatten list for key '{}': {}", key, exc)
                normalized[key] = val
        else:
            normalized[key] = val
    return normalized


def classify_signal_tier(confidence: float) -> str:
    """Classify a signal into a quality tier based on its confidence score.

    Parameters
    ----------
    confidence:
        Signal confidence (0–100 scale).

    Returns
    -------
    One of: ``"A+"`` (sniper, 80-100), ``"B"`` (setup, 65-79),
    ``"WATCHLIST"`` (50-64), ``"FILTERED"`` (< 50).
    """
    if confidence >= 80:
        return "A+"
    elif confidence >= 65:
        return "B"
    elif confidence >= 50:
        return "WATCHLIST"
    return "FILTERED"


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
    regime_context: Any = None  # RegimeContext from regime detector
    confidence_adjustments: Dict[str, float] = _dc.field(default_factory=dict)


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
        order_flow_store: Optional[Any] = None,
        tier_manager: Optional[TierManager] = None,
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
        self.order_flow_store: Optional[Any] = order_flow_store

        # Optional dynamic tier manager (PR 2 — Market Watchdog & Dynamic Tiering).
        # When present, get_symbol_tier() delegates to TierManager.get_tier() which
        # returns a live, volume+volatility-ranked PairTier refreshed every ~5 min.
        # When absent, the scanner falls back to the PairManager's static tier
        # assignment (rank-by-volume-only, updated on pair refresh).
        self.tier_manager: Optional[TierManager] = tier_manager

        # Stateful signal-quality enhancement modules
        self.feedback_loop: FeedbackLoop = FeedbackLoop()
        self.cluster_suppressor: ClusterSuppressor = ClusterSuppressor()

        # Mutable state shared with the engine / command handler
        self.paused_channels: Set[str] = set()
        self.confidence_overrides: Dict[str, float] = {}
        self.force_scan: bool = False

        # WebSocket managers (set after boot)
        self.ws_spot: Optional[Any] = None
        self.ws_futures: Optional[Any] = None

        # Optional circuit breaker (set after construction)
        self.circuit_breaker: Optional[Any] = None

        # Optional gem scanner (set after construction)
        self.gem_scanner: Optional[Any] = None

        # Order book spread cache: symbol → (spread_pct, expiry_monotonic_time)
        # expiry_monotonic_time is an absolute time.monotonic() value; the entry
        # is valid while time.monotonic() < expiry_monotonic_time.
        self._order_book_cache: Dict[str, Tuple[float, float]] = {}
        self._order_book_fetches_this_cycle: int = 0

        # Order book depth cache: symbol → (bids, asks, expiry_monotonic_time)
        # Stores the raw top-level bids/asks for Order Book Imbalance filtering.
        self._order_book_depth_cache: Dict[str, Tuple[list, list, float]] = {}

        # Cooldown tracking: (symbol, channel_name) → monotonic expiry time
        self._cooldown_until: Dict[Tuple[str, str], float] = {}

        # Regime history: symbol → list of (monotonic_time, regime_value) tuples
        # Used to detect oscillating / unstable regimes (too many flips in window).
        self._regime_history: Dict[str, List[Tuple[float, str]]] = {}
        # Most recent overall-market regime value (updated per scan cycle; used by
        # gem scanner for adaptive threshold adjustment — feature 7).
        self._last_market_regime: str = "RANGING"

        # Semaphore to limit concurrent symbol scans
        self._scan_semaphore: asyncio.Semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SCANS)

        # Tiered scanning counters
        self._scan_cycle_count: int = 0
        self._last_tier3_scan_time: float = 0.0

        # WS health-aware scan gating: counts consecutive cycles where both
        # WS managers are unhealthy, used to trigger an admin alert.
        self._consecutive_ws_degraded_cycles: int = 0

        # Scan-latency circuit breaker: counts consecutive cycles where
        # elapsed_ms exceeded SCAN_LATENCY_ALERT_MS.
        self._consecutive_high_latency_cycles: int = 0

        # Per-cycle WS degradation flag: True when either WS manager is
        # partially degraded.  Set at the start of each scan cycle and used
        # by _get_spread_pct to apply tighter REST fetch limits.
        self._ws_any_degraded_this_cycle: bool = False

        # Suppression telemetry: counters per suppression reason, accumulated
        # over each scan cycle and logged as a summary at cycle end.
        self._suppression_counters: Dict[str, int] = defaultdict(int)

        # Suppression tracker — records structured suppression events for
        # Telegram digest and data-driven threshold tuning.
        self.suppression_tracker: SuppressionTracker = SuppressionTracker()

    # ------------------------------------------------------------------
    # Dynamic tier query helper
    # ------------------------------------------------------------------

    def get_symbol_tier(self, symbol: str) -> PairTier:
        """Return the current :class:`~src.pair_manager.PairTier` for *symbol*.

        Resolution order
        ----------------
        1. If a :class:`~src.tier_manager.TierManager` is attached, delegate to
           its live volume+volatility-ranked tier assignment (refreshed every
           ~5 minutes by the background polling loop).
        2. Otherwise fall back to the :class:`~src.pair_manager.PairManager`
           static tier — the volume-rank assignment from the last pair refresh.
        3. If the symbol is not found in either source, return
           :attr:`~src.pair_manager.PairTier.TIER3` as a safe default.
        """
        if self.tier_manager is not None:
            return self.tier_manager.get_tier(symbol)
        info = self.pair_mgr.pairs.get(symbol)
        if info is not None:
            return info.tier
        return PairTier.TIER3

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def scan_loop(self) -> None:
        """Periodic scan over all pairs / channels."""
        log.info("Scanner loop started")
        while True:
            t0 = time.monotonic()
            self._order_book_fetches_this_cycle = 0
            self._scan_cycle_count += 1

            # Always clean up expired signals first (safety net for stuck slots)
            expired_count = self.router.cleanup_expired()
            if expired_count > 0:
                log.info("Cleaned up {} expired signals at start of scan cycle", expired_count)

            # Skip scanning when circuit breaker is tripped
            if self.circuit_breaker and self.circuit_breaker.is_tripped():
                log.warning("Circuit breaker tripped — skipping scan cycle")
                await asyncio.sleep(5)
                continue

            # WS health-aware scan gating: when both WS managers are unhealthy
            # (or not set) there is no live kline data, so a full scan over
            # 796 pairs burns API weight on stale candles and produces no
            # signals.  Skip the full scan and track degraded-cycle count.
            ws_spot_ok = self.ws_spot.is_healthy if self.ws_spot else True
            ws_futures_ok = self.ws_futures.is_healthy if self.ws_futures else True
            ws_both_unhealthy = not ws_spot_ok and not ws_futures_ok
            # Partial degradation: either manager has below-threshold health.
            # Used to tighten REST fetch limits for the remainder of the cycle.
            ws_spot_ratio = self.ws_spot.health_ratio if self.ws_spot else 1.0
            ws_futures_ratio = self.ws_futures.health_ratio if self.ws_futures else 1.0
            self._ws_any_degraded_this_cycle = (
                ws_spot_ratio < WS_PARTIAL_HEALTH_THRESHOLD
                or ws_futures_ratio < WS_PARTIAL_HEALTH_THRESHOLD
            )
            if ws_both_unhealthy:
                self._consecutive_ws_degraded_cycles += 1
                log.warning(
                    "WS health degraded (spot={}, futures={}) — skipping full scan "
                    "(degraded cycle #{})",
                    ws_spot_ok, ws_futures_ok, self._consecutive_ws_degraded_cycles,
                )
                if self._consecutive_ws_degraded_cycles == WS_DEGRADED_CYCLES_ALERT:
                    try:
                        _alert_fn = self.telemetry.get_admin_alert_callback()
                        if _alert_fn is not None:
                            await _alert_fn(
                                f"⚠️ WebSocket unhealthy for "
                                f"{self._consecutive_ws_degraded_cycles} consecutive scan cycles. "
                                "Scan is paused until WS recovers. Consider /restart."
                            )
                    except Exception:
                        pass
                elapsed_ms = (time.monotonic() - t0) * 1000
                self.telemetry.set_scan_latency(elapsed_ms)
                await asyncio.sleep(5)
                continue
            else:
                if self._consecutive_ws_degraded_cycles > 0:
                    log.info(
                        "WS health restored after {} degraded cycles",
                        self._consecutive_ws_degraded_cycles,
                    )
                self._consecutive_ws_degraded_cycles = 0

            try:
                # Prioritise high-volume pairs for order book fetches
                sorted_pairs = sorted(
                    self.pair_mgr.pairs.items(),
                    key=lambda kv: kv[1].volume_24h_usd,
                    reverse=True,
                )

                # Tiered scanning:
                #   Tier 1 → every cycle (full scan, all channels)
                #   Tier 2 → every TIER2_SCAN_EVERY_N_CYCLES cycles (SWING+SPOT only)
                #   Tier 3 → every TIER3_SCAN_EVERY_N_CYCLES cycles (cycle-based)
                #            OR on the time-based interval (whichever fires first)
                scan_tier2 = (self._scan_cycle_count % TIER2_SCAN_EVERY_N_CYCLES == 0)
                scan_tier3 = (self._scan_cycle_count % TIER3_SCAN_EVERY_N_CYCLES == 0)
                # If scan latency has been extremely high, temporarily skip
                # Tier 2 pairs to reduce cycle time.
                skip_tier2_for_latency = (
                    self._consecutive_high_latency_cycles > 0
                    and self.telemetry.scan_latency_ms > SCAN_LATENCY_REDUCE_MS
                )
                if skip_tier2_for_latency:
                    log.warning(
                        "Scan latency circuit breaker: skipping Tier 2 pairs "
                        "(last latency={:.0f}ms)", self.telemetry.scan_latency_ms
                    )
                # PR 4 — Preemptive rate-limit throttling: pause Tier 2 scans
                # when either limiter reports >85% budget usage so the engine
                # never reaches the hard Binance cap and 42 s lockout.
                _rl_tier2_paused = (
                    rate_limiter.is_tier2_paused or futures_rate_limiter.is_tier2_paused
                )
                if _rl_tier2_paused:
                    log.warning(
                        "Rate limit >85%% — preemptively pausing Tier 2 pair scans "
                        "(spot=%d/%d, futures=%d/%d)",
                        rate_limiter.used, rate_limiter.budget,
                        futures_rate_limiter.used, futures_rate_limiter.budget,
                    )
                pairs_this_cycle = [
                    (sym, info) for sym, info in sorted_pairs
                    if info.tier == PairTier.TIER1
                    or (info.tier == PairTier.TIER2 and scan_tier2
                        and not skip_tier2_for_latency and not _rl_tier2_paused)
                    or (info.tier == PairTier.TIER3 and scan_tier3
                        and not skip_tier2_for_latency and not _rl_tier2_paused)
                ]

                # Apply cheap in-memory pre-filters to reduce the number of
                # symbols that reach expensive API calls (order book, klines).
                # This keeps Binance weight consumption ~400/min for 200+ pairs.
                filtered_pairs = self._prefilter_pairs(pairs_this_cycle)

                # When WS is partially degraded, cap the scan set to top-N
                # pairs by volume.  This prevents querying REST /depth for
                # hundreds of pairs that lack live kline updates, which was
                # the primary cause of the 100% rate-limit exhaustion observed
                # when the futures WS dropped (WS=300, ok=False).
                if self._ws_any_degraded_this_cycle and len(filtered_pairs) > WS_DEGRADED_MAX_PAIRS:
                    filtered_pairs = filtered_pairs[:WS_DEGRADED_MAX_PAIRS]
                    log.warning(
                        "WS partially degraded (spot_ratio={:.0%}, futures_ratio={:.0%}) "
                        "— limiting scan to top {} pairs to protect REST rate limit",
                        ws_spot_ratio, ws_futures_ratio, WS_DEGRADED_MAX_PAIRS,
                    )

                # PR 3 — Tier-aware REST fallback: when WS is degraded, issue
                # a single weight-efficient global bookTicker call (Weight 2)
                # to pre-populate the spread cache for all Tier 2 and Tier 3
                # pairs.  This replaces per-symbol /depth calls for those tiers
                # and reserves the heavier /depth endpoint strictly for Tier 1
                # (Hot) pairs.  The concurrent symbol scans below will then
                # find valid cache entries for Tier 2/3 symbols and skip the
                # per-symbol fetch entirely.
                if self._ws_any_degraded_this_cycle:
                    await self._fetch_global_book_tickers(market="futures")

                sem = self._scan_semaphore
                tasks = [
                    self._scan_symbol_bounded(sem, sym, info.volume_24h_usd)
                    for sym, info in filtered_pairs
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for sym_info, result in zip(filtered_pairs, results):
                    if isinstance(result, Exception):
                        log.warning(
                            "Scan error for {} ({}): {}",
                            sym_info[0], type(result).__name__, result,
                        )

                # Tier 3 lightweight scan (time-gated, independent of cycle count)
                _now = time.monotonic()
                # PR 4 — Skip Tier 3 scan when rate limit >70% to preserve budget.
                _rl_tier3_paused = (
                    rate_limiter.is_tier3_paused or futures_rate_limiter.is_tier3_paused
                )
                if _rl_tier3_paused:
                    log.warning(
                        "Rate limit >70%% — preemptively pausing Tier 3 lightweight scan "
                        "(spot=%d/%d, futures=%d/%d)",
                        rate_limiter.used, rate_limiter.budget,
                        futures_rate_limiter.used, futures_rate_limiter.budget,
                    )
                if (not _rl_tier3_paused
                        and _now - self._last_tier3_scan_time >= TIER3_SCAN_INTERVAL_MINUTES * 60):
                    self._last_tier3_scan_time = _now
                    await self._lightweight_tier3_scan()

                # Gem scanner (interval-gated by GEM_SCAN_INTERVAL_HOURS)
                if self.gem_scanner is not None and self.gem_scanner.enabled:
                    _wall_now = time.time()
                    _gem_interval_s = self.gem_scanner.scan_interval_hours * 3600
                    if _wall_now - self.gem_scanner._last_scan >= _gem_interval_s:
                        self.gem_scanner.update_last_scan()
                        for _sym, _info in sorted_pairs:
                            try:
                                await self._scan_gem_symbol(_sym)
                            except Exception as _exc:
                                log.warning(
                                    "Gem scan error for {} ({}): {}", _sym, type(_exc).__name__, _exc
                                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Scan loop error: {}", exc)

            elapsed_ms = (time.monotonic() - t0) * 1000
            self.telemetry.set_scan_latency(elapsed_ms)

            # Scan-latency circuit breaker checks
            if elapsed_ms > SCAN_LATENCY_WARN_MS:
                log.warning("Scan latency high: {:.0f}ms (threshold {:.0f}ms)", elapsed_ms, SCAN_LATENCY_WARN_MS)
            if elapsed_ms > SCAN_LATENCY_ALERT_MS:
                self._consecutive_high_latency_cycles += 1
                if self._consecutive_high_latency_cycles >= SCAN_LATENCY_ALERT_CONSECUTIVE:
                    try:
                        _alert_fn = self.telemetry.get_admin_alert_callback()
                        if _alert_fn is not None:
                            await _alert_fn(
                                f"⚠️ Scan latency critical: {elapsed_ms:.0f}ms for "
                                f"{self._consecutive_high_latency_cycles} consecutive cycles. "
                                "WS or depth API may be degraded."
                            )
                    except Exception:
                        pass
            else:
                self._consecutive_high_latency_cycles = 0

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

            # Log suppression telemetry summary for this cycle, then reset.
            if self._suppression_counters:
                log.info(
                    "Scan cycle suppression summary: {}",
                    dict(self._suppression_counters),
                )
                self._suppression_counters.clear()

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
        cooldown_s = SIGNAL_SCAN_COOLDOWN_SECONDS.get(channel_name, 60)
        self._cooldown_until[(symbol, channel_name)] = (
            time.monotonic() + cooldown_s
        )
        log.debug(
            "Cooldown set for {} {} ({:.0f}s)", symbol, channel_name, cooldown_s
        )

    def _count_regime_flips(self, symbol: str, window_minutes: int = 30) -> int:
        """Return the number of regime transitions within the rolling window."""
        history = self._regime_history.get(symbol, [])
        if len(history) < 2:
            return 0
        cutoff = time.monotonic() - window_minutes * 60
        recent = [r for t, r in history if t >= cutoff]
        if len(recent) < 2:
            return 0
        return sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])

    def _is_regime_unstable(
        self,
        symbol: str,
        window_minutes: int = 30,
        max_flips: int = 2,
    ) -> bool:
        """Return True if the regime for *symbol* has flipped more than *max_flips*
        times within the last *window_minutes* minutes.

        A regime "flip" is any transition between distinct regime values in the
        recorded history.  When the count of flips exceeds the threshold the
        symbol is considered too noisy for TAPE-style signals.
        """
        return self._count_regime_flips(symbol, window_minutes) > max_flips

    def _prefilter_pairs(
        self, pairs: List[Tuple[str, Any]]
    ) -> List[Tuple[str, Any]]:
        """Return a cheaply-filtered subset of pairs for expensive API scans.

        Applies three in-memory checks (zero API calls) before the main scan
        loop creates tasks and acquires the concurrency semaphore:

        1. **Volume filter** – skip symbols whose 24h USD volume is below
           ``SCAN_MIN_VOLUME_USD``.  Thin markets rarely produce valid signals
           and every API call they trigger wastes Binance weight budget.
        2. **All-channel active-signal filter** – skip symbols that already
           have an active signal on *every* channel we scan.  The per-channel
           check inside :meth:`_should_skip_channel` would catch each one
           individually, but pre-filtering avoids even building the scan context.
        3. **All-channel cooldown filter** – skip symbols where every channel
           is currently in cooldown, for the same reason as above.

        Typically reduces 200+ symbols down to ~60-80 before any order-book or
        kline fetches are triggered, keeping weight consumption ~400/min.
        """
        channel_names = [c.config.name for c in self.channels]
        active_symbols_channels = {
            (s.symbol, s.channel)
            for s in self.router.active_signals.values()
        }
        result: List[Tuple[str, Any]] = []
        skipped_volume = skipped_all_active = skipped_all_cooldown = 0

        for sym, info in pairs:
            # 1. Volume pre-filter
            if info.volume_24h_usd < SCAN_MIN_VOLUME_USD:
                skipped_volume += 1
                continue
            # 2. All channels already have an active signal for this symbol
            if channel_names and all(
                (sym, ch) in active_symbols_channels for ch in channel_names
            ):
                skipped_all_active += 1
                continue
            # 3. All channels are in cooldown for this symbol
            if channel_names and all(
                self._is_in_cooldown(sym, ch) for ch in channel_names
            ):
                skipped_all_cooldown += 1
                continue
            result.append((sym, info))

        if skipped_volume or skipped_all_active or skipped_all_cooldown:
            log.debug(
                "Pre-filter: %d/%d symbols kept "
                "(skipped %d low-volume, %d all-active, %d all-cooldown)",
                len(result), len(pairs),
                skipped_volume, skipped_all_active, skipped_all_cooldown,
            )
        return result

    async def _scan_symbol_bounded(self, sem: asyncio.Semaphore, symbol: str, volume_24h: float) -> None:
        """Acquire *sem* then delegate to :meth:`_scan_symbol`."""
        async with sem:
            await self._scan_symbol(symbol, volume_24h)

    def _load_candles(self, symbol: str) -> Dict[str, dict]:
        candles: Dict[str, dict] = {}
        for tf in SEED_TIMEFRAMES:
            c = self.data_store.get_candles(symbol, tf.interval)
            if c:
                candles[tf.interval] = _normalize_candle_dict(c)
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
            if len(c) >= 35:  # slow_period(26) + signal_period(9)
                ml, sl_line, hist = macd(c)
                ind["macd_histogram_last"] = (
                    float(hist[-1]) if not np.isnan(hist[-1]) else None
                )
                ind["macd_histogram_prev"] = (
                    float(hist[-2]) if len(hist) > 1 and not np.isnan(hist[-2]) else None
                )
            indicators[tf_key] = ind
        return indicators

    async def _fetch_global_book_tickers(self, market: str = "futures") -> None:
        """Pre-populate the spread cache for Tier 2 and Tier 3 pairs using a
        single weight-efficient ``bookTicker`` call.

        When WebSockets degrade, calling ``/fapi/v1/depth`` for every pair
        individually exhausts the Binance API weight budget and produces
        30-50 s scan latencies.  Instead, this method issues **one** global
        ``/fapi/v1/ticker/bookTicker`` request (Weight: 2) that returns the
        best bid/ask for *all* symbols, then seeds :attr:`_order_book_cache`
        for every Tier 2 and Tier 3 pair found in the response.

        Tier 1 (Hot) pairs are deliberately excluded from this pre-population
        so that their higher-priority ``/depth`` call (which also captures full
        order-book depth for OBI filtering) is not suppressed by the shorter
        bookTicker TTL.

        Parameters
        ----------
        market:
            ``"futures"`` (default) or ``"spot"``.  The appropriate
            :class:`~src.binance.BinanceClient` instance is lazily created if
            not already present.
        """
        try:
            if market == "futures":
                if self.futures_client is None:
                    self.futures_client = BinanceClient("futures")
                client = self.futures_client
            else:
                if self.spot_client is None:
                    self.spot_client = BinanceClient("spot")
                client = self.spot_client

            tickers = await asyncio.wait_for(
                client.fetch_all_book_tickers(),
                timeout=_BOOK_TICKER_PREFETCH_TIMEOUT_S,
            )
            if not tickers:
                log.warning(
                    "Global bookTicker pre-fetch returned no data (market={})", market
                )
                return

            now = time.monotonic()
            populated = 0
            for symbol, entry in tickers.items():
                # Only seed Tier 2/3; Tier 1 uses the fuller /depth endpoint.
                tier = self.get_symbol_tier(symbol)
                if tier == PairTier.TIER1:
                    continue
                # Skip if there is already a fresh cache entry for this symbol.
                existing = self._order_book_cache.get(symbol)
                if existing and now < existing[1]:
                    continue
                try:
                    best_bid = float(entry.get("bidPrice", 0))
                    best_ask = float(entry.get("askPrice", 0))
                except (TypeError, ValueError):
                    continue
                if best_bid <= 0 or best_ask <= 0:
                    continue
                mid = (best_bid + best_ask) / 2.0
                if mid <= 0:
                    continue
                spread_pct = (best_ask - best_bid) / mid * 100.0
                self._order_book_cache[symbol] = (spread_pct, now + _BOOK_TICKER_CACHE_TTL)
                populated += 1

            log.debug(
                "Global bookTicker pre-fetch populated {} Tier 2/3 spread cache entries",
                populated,
            )
        except asyncio.TimeoutError:
            log.warning("Global bookTicker pre-fetch timed out (market={})", market)
        except Exception as exc:
            log.warning("Global bookTicker pre-fetch error (market={}): {}", market, exc)

    async def _get_spread_pct(self, symbol: str, market: str = "spot") -> float:
        spread_pct = 0.01  # fallback
        now = time.monotonic()
        cached = self._order_book_cache.get(symbol)
        if cached and now < cached[1]:
            return cached[0]
        # Apply tighter limits when WS is partially degraded to prevent
        # burning REST rate-limit budget on depth fetches for all pairs.
        if self._ws_any_degraded_this_cycle:
            effective_max_fetches = _MAX_ORDER_BOOK_FETCHES_WS_DEGRADED
            effective_min_weight = _MIN_WEIGHT_FOR_ORDER_BOOK_WS_DEGRADED
            # PR 3 — Tier-aware REST fallback: when WS is degraded, skip the
            # heavy per-symbol /depth call for Tier 2 and Tier 3 pairs.
            # Their spread data was already pre-populated from the global
            # bookTicker call issued at the start of this scan cycle.
            # Reserve /depth strictly for Tier 1 (Hot) pairs.
            symbol_tier = self.get_symbol_tier(symbol)
            if symbol_tier != PairTier.TIER1:
                return spread_pct
        else:
            effective_max_fetches = _MAX_ORDER_BOOK_FETCHES_PER_CYCLE
            effective_min_weight = _MIN_WEIGHT_FOR_ORDER_BOOK
        # Guard 1: hard per-cycle cap (safety net; rate limiter is primary).
        if self._order_book_fetches_this_cycle >= effective_max_fetches:
            return spread_pct
        # Guard 2: rate-limiter budget check — skip fetch when budget is low
        # to preserve capacity for higher-priority calls (klines, exchange info).
        # Use the market-appropriate limiter so futures depth checks against the
        # futures budget rather than the spot budget.
        active_limiter = futures_rate_limiter if market == "futures" else rate_limiter
        if active_limiter.remaining < effective_min_weight:
            log.debug(
                "Skipping order book fetch for %s – %s rate limiter budget low (%d remaining)",
                symbol, market, active_limiter.remaining,
            )
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
            # Fetch 20 levels: top-of-book bids/asks give the spread while
            # the full depth is used for Order Book Imbalance (OBI) filtering.
            book = await client.fetch_order_book(symbol, limit=20)
            if book and book.get("bids") and book.get("asks"):
                best_bid = float(book["bids"][0][0])
                best_ask = float(book["asks"][0][0])
                mid = (best_bid + best_ask) / 2.0
                if mid > 0:
                    spread_pct = (best_ask - best_bid) / mid * 100.0
                # Cache spread with normal TTL
                self._order_book_cache[symbol] = (spread_pct, now + _SPREAD_CACHE_TTL)
                # Cache raw depth for OBI filtering (same TTL)
                self._order_book_depth_cache[symbol] = (
                    book["bids"],
                    book["asks"],
                    now + _SPREAD_CACHE_TTL,
                )
            else:
                # Failed fetch (e.g. futures-only symbol on spot endpoint):
                # cache with a longer TTL to avoid hammering the endpoint every cycle.
                self._order_book_cache[symbol] = (spread_pct, now + _SPREAD_FAIL_CACHE_TTL)
        except Exception:
            self._order_book_cache[symbol] = (spread_pct, now + _SPREAD_FAIL_CACHE_TTL)
        return spread_pct

    def _get_order_book_depth(self, symbol: str) -> Optional[dict]:
        """Return the most recent cached order book depth for *symbol*, or ``None``."""
        cached = self._order_book_depth_cache.get(symbol)
        if cached is None:
            return None
        bids, asks, expiry = cached
        if time.monotonic() >= expiry:
            return None
        return {"bids": bids, "asks": asks}

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
        # Use scalp-optimised sweep detection parameters: shorter lookback catches
        # recent S/R levels; wider tolerance catches institutional sweeps that
        # reclaim $100-200 past the level.  The full post-channel quality pipeline
        # (MTF, VWAP, KillZone, OI, CrossAsset, Spoof, VolDiv, Clustering filters
        # and confidence scoring) still runs on the detected sweeps.
        smc_result = self.smc_detector.detect(
            symbol, candles, ticks, self.order_flow_store,
            lookback=SMC_SCALP_LOOKBACK,
            tolerance_pct=SMC_SCALP_TOLERANCE_PCT,
        )
        smc_data = smc_result.as_dict()
        # Attach per-pair profile so channel evaluators can consume it via
        # smc_data.get("pair_profile") without any signature changes.
        smc_data["pair_profile"] = classify_pair_tier(symbol, volume_24h_usd=volume_24h)

        regime_tf = "5m" if "5m" in indicators else "1m"
        regime_ind = indicators.get("5m", indicators.get("1m", {}))
        regime_candles = candles.get("5m", candles.get("1m"))
        regime_result = self.regime_detector.classify(regime_ind, regime_candles, timeframe=regime_tf)
        log.debug("{} regime: {}", symbol, regime_result.regime.value)
        # Keep a rolling picture of the overall market regime using BTCUSDT as
        # the representative benchmark (feature 7 – gem adaptive thresholds).
        if "BTC" in symbol.upper():
            self._last_market_regime = regime_result.regime.value

        # Record regime history for oscillation / instability detection
        _now = time.monotonic()
        history = self._regime_history.setdefault(symbol, [])
        history.append((_now, regime_result.regime.value))
        # Prune entries older than 30 minutes
        _cutoff = _now - 30 * 60
        self._regime_history[symbol] = [(t, r) for t, r in history if t >= _cutoff]

        ind_for_predict = indicators.get("5m", indicators.get("1m", {}))
        candle_total = sum(len(cd.get("close", [])) for cd in candles.values())
        market = (
            self.pair_mgr.pairs[symbol].market
            if symbol in self.pair_mgr.pairs
            else "spot"
        )
        spread_pct, onchain_data = await asyncio.gather(
            self._get_spread_pct(symbol, market=market),
            self._fetch_onchain_data(symbol),
        )
        ai: Dict[str, Any] = {}
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
        # Build rich regime context for signal enrichment
        from src.vwap import compute_vwap  # noqa: PLC0415
        vwap_val = 0.0
        if regime_candles is not None:
            vwap_result = compute_vwap(
                regime_candles.get("high", []),
                regime_candles.get("low", []),
                regime_candles.get("close", []),
                regime_candles.get("volume", []),
            )
            if vwap_result is not None:
                vwap_val = vwap_result.vwap
        regime_context = self.regime_detector.build_regime_context(
            regime_result, regime_candles, regime_ind, vwap=vwap_val,
        )
        # Attach regime context so channel evaluators can access atr_percentile
        # via smc_data.get("regime_context") without any signature changes.
        smc_data["regime_context"] = regime_context
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
            regime_context=regime_context,
        )

    def _should_skip_channel(self, symbol: str, chan_name: str, ctx: ScanContext) -> bool:
        # Tier-based channel gating: Tier 2 pairs skip SCALP (REST-only, no
        # order book depth for tight scalp execution).
        pair_info = self.pair_mgr.pairs.get(symbol)
        if pair_info is not None and pair_info.tier == PairTier.TIER2 and chan_name == "360_SCALP":
            log.debug("Skipping {} {} – Tier 2 pair excluded from SCALP", symbol, chan_name)
            self._suppression_counters[f"tier2_scalp_excluded:{chan_name}"] += 1
            return True
        # Per-channel pair quality gate: the generic ctx.pair_quality uses a
        # universal 5% spread limit.  When it fails, we re-evaluate with
        # channel-specific thresholds — this allows wider-spread pairs on
        # SWING/SPOT/GEM channels while keeping SCALP at a tighter limit.
        if not ctx.pair_quality.passed:
            _regime_ind = ctx.indicators.get("5m", ctx.indicators.get("1m", {}))
            _regime_candles = ctx.candles.get("5m", ctx.candles.get("1m"))
            try:
                _vol = float(pair_info.volume_24h_usd) if pair_info is not None else 0.0
            except (TypeError, ValueError):
                _vol = 0.0
            # Only attempt the channel-specific re-check when we have valid data;
            # if volume is unavailable, fail open (don't double-penalise the pair).
            if _vol > 0.0:
                chan_quality = assess_pair_quality_for_channel(
                    volume_24h=_vol,
                    spread_pct=ctx.spread_pct,
                    indicators=_regime_ind,
                    candles=_regime_candles,
                    channel_name=chan_name,
                    current_regime=ctx.regime_result.regime.value,
                )
                if not chan_quality.passed:
                    log.info(
                        "Skipping {} {} – pair quality gate failed: {} (score={:.1f}, vol={:.0f}, spread={:.4f})",
                        symbol,
                        chan_name,
                        chan_quality.reason,
                        chan_quality.score,
                        _vol,
                        ctx.spread_pct,
                    )
                    _supp_reason = (
                        REASON_SPREAD_GATE if "spread" in chan_quality.reason
                        else REASON_VOLUME_GATE
                    )
                    self._suppression_counters[f"pair_quality:{chan_quality.reason}"] += 1
                    self.suppression_tracker.record(SuppressionEvent(
                        symbol=symbol,
                        channel=chan_name,
                        reason=_supp_reason,
                        regime=ctx.regime_result.regime.value,
                    ))
                    return True
                # Channel-specific re-check passed — allow through despite generic failure
                log.debug(
                    "{} {} passed channel-specific quality gate (generic failed)",
                    symbol, chan_name,
                )
            else:
                log.debug(
                    "Skipping {} {} – pair quality gate failed: {}",
                    symbol,
                    chan_name,
                    ctx.pair_quality.reason,
                )
                self._suppression_counters[f"pair_quality:{ctx.pair_quality.reason}"] += 1
                return True
        if ctx.market_state == MarketState.VOLATILE_UNSUITABLE:
            # Higher-timeframe strategies (SPOT/GEM operate on H4/D1/W1) can
            # tolerate short-term intraday volatility — only block lower-TF channels.
            if chan_name not in ("360_SPOT", "360_GEM"):
                log.debug(
                    "Skipping {} {} – volatile/unsuitable market state",
                    symbol,
                    chan_name,
                )
                self._suppression_counters[f"volatile_unsuitable:{chan_name}"] += 1
                return True
        if chan_name in self.paused_channels:
            self._suppression_counters[f"paused_channel:{chan_name}"] += 1
            return True
        if self._is_in_cooldown(symbol, chan_name):
            log.debug("Cooldown active: skipping {} {}", symbol, chan_name)
            self._suppression_counters[f"cooldown:{chan_name}"] += 1
            return True
        # Per-symbol circuit breaker: suppress the symbol across all channels
        # when it has accumulated too many consecutive SL hits.
        if self.circuit_breaker is not None and self.circuit_breaker.is_symbol_tripped(symbol):
            log.debug(
                "Per-symbol circuit breaker active: skipping {} {}", symbol, chan_name
            )
            self._suppression_counters[f"circuit_breaker:{chan_name}"] += 1
            return True
        if any(
            s.symbol == symbol and s.channel == chan_name
            for s in self.router.active_signals.values()
        ):
            log.debug("Skipping {} {} – active signal already exists", symbol, chan_name)
            self._suppression_counters[f"active_signal:{chan_name}"] += 1
            return True
        if (
            chan_name == "360_SCALP"
            and ctx.is_ranging
            and ctx.adx_val < _RANGING_ADX_SUPPRESS_THRESHOLD
        ):
            # Soft-gate: apply confidence penalty instead of hard block.
            ctx.confidence_adjustments["RANGING_LOW_ADX"] = -_RANGING_LOW_ADX_CONF_PENALTY
            log.debug(
                "RANGING low-ADX soft penalty for {} {} (ADX={:.1f}, penalty={:.1f}pts)",
                symbol, chan_name, ctx.adx_val, _RANGING_LOW_ADX_CONF_PENALTY,
            )
            self._suppression_counters[f"ranging_low_adx_soft:{chan_name}"] += 1
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
            self._suppression_counters[f"regime:{current_regime}:{chan_name}"] += 1
            self.suppression_tracker.record(SuppressionEvent(
                symbol=symbol,
                channel=chan_name,
                reason=REASON_QUIET_REGIME,
                regime=current_regime,
            ))
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
        chan_name: str = "",
    ) -> RiskAssessment:
        return build_risk_plan(
            signal=sig,
            indicators=ctx.indicators,
            candles=ctx.candles,
            smc_data=ctx.smc_data,
            setup=setup.setup_class,
            spread_pct=ctx.spread_pct,
            channel=chan_name or sig.channel,
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

    @staticmethod
    def _get_primary_timeframe(chan_name: str) -> str:
        """Return the primary timeframe interval string for a given channel name."""
        if chan_name == "360_SWING":
            return "1h"
        if chan_name == "360_SPOT":
            return "4h"
        return "5m"

    @staticmethod
    def _resolve_candles(candles: Dict[str, dict], primary_tf: str) -> dict:
        """Return the best available candle dict for *primary_tf*, falling back to 5m/1m."""
        return candles.get(primary_tf) or candles.get("5m") or candles.get("1m") or {}

    @staticmethod
    def _classify_macro_trend(closes: list) -> tuple[str, float]:
        """Classify the macro trend from a close price series.

        Returns ``(trend_label, pct_change)`` where *trend_label* is one of
        ``"DUMPING"``, ``"BEARISH"``, ``"BULLISH"``, or ``"NEUTRAL"``.
        """
        first = float(closes[0])
        last = float(closes[-1])
        pct = (last - first) / first if first != 0 else 0.0
        if pct < -0.02:
            trend = "DUMPING"
        elif pct < -0.005:
            trend = "BEARISH"
        elif pct > 0.005:
            trend = "BULLISH"
        else:
            trend = "NEUTRAL"
        return trend, pct

    def _compute_base_confidence(
        self,
        symbol: str,
        volume_24h: float,
        sig: Any,
        ctx: ScanContext,
        cross_verified: Optional[bool],
        chan_name: str = "",
        funding_rate: Optional[float] = None,
        sentiment_score: float = 0.0,
    ) -> Optional[float]:
        # Gate: if OI is rising against the sweep direction, block the signal
        if ctx.smc_data.get("oi_invalidated", False):
            log.debug(
                "{}: signal blocked – OI rising against {} sweep direction",
                symbol, sig.direction.value,
            )
            return None

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

        # Order flow score: OI trend + liquidations + CVD divergence
        of_score = 0.0
        if self.order_flow_store is not None:
            oi_trend = self.order_flow_store.get_oi_trend(symbol)
            liq_vol = self.order_flow_store.get_recent_liq_volume_usd(symbol)
            cvd_div = ctx.smc_data.get("cvd_divergence")
            of_score = score_order_flow(
                oi_trend=oi_trend.value,
                liq_vol_usd=liq_vol,
                cvd_divergence=cvd_div,
                signal_direction=sig.direction.value,
                funding_rate=funding_rate,
            )

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
                macd_histogram=ctx.ind_for_predict.get("macd_histogram_last"),
                macd_histogram_prev=ctx.ind_for_predict.get("macd_histogram_prev"),
                signal_direction=sig.direction.value,
            ),
            liquidity_score=score_liquidity(volume_24h, channel=chan_name),
            spread_score=score_spread(ctx.spread_pct),
            data_sufficiency=score_data_sufficiency(ctx.candle_total),
            multi_exchange=score_multi_exchange(verified=cross_verified),
            onchain_score=score_onchain(ctx.onchain_data),
            order_flow_score=of_score,
            sentiment_score=sentiment_score,
            has_enough_history=self.pair_mgr.has_enough_history(symbol),
            opposing_position_open=any(
                s.symbol == symbol and s.direction.value != sig.direction.value
                for s in self.router.active_signals.values()
            ),
        )
        result = compute_confidence(cinp, channel=chan_name)
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
        if chan_name == "360_SCALP" and ctx.is_ranging and sig.setup_class == "RANGE_FADE":
            sig.confidence += _RANGING_RANGE_FADE_CONF_BOOST
            log.debug(
                "SCALP RANGE_FADE confidence boosted for {} (RANGING): {:.1f}",
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

    @staticmethod
    def _clamp_confidence(value: float) -> float:
        return max(0.0, min(100.0, round(value, 2)))

    def _populate_signal_context(self, sig: Any, volume_24h: float, ctx: ScanContext) -> None:
        sig.market_phase = ctx.market_state.value
        if ctx.regime_context is not None:
            rc = ctx.regime_context
            try:
                sig.market_phase = (
                    f"{rc.label} | ATR%ile={float(rc.atr_percentile):.0f} | "
                    f"Vol={rc.volume_profile}"
                )
                sig.regime_context = (
                    f"ADXslope={float(rc.adx_slope):.2f} strengthen={rc.is_regime_strengthening}"
                )
            except (TypeError, ValueError):
                pass  # Keep market_state.value when context is not a real RegimeContext
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
        # Attach the cached order book depth snapshot so downstream filters
        # (RiskManager OBI check) can evaluate it without an extra fetch.
        sig.order_book = self._get_order_book_depth(sig.symbol)
        # How long (minutes) the setup remains actionable — sourced from config.
        sig.valid_for_minutes = SIGNAL_VALID_FOR_MINUTES.get(sig.channel, 15)

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
        t0_signal = time.monotonic()
        soft_penalty: float = 0.0  # Accumulated confidence deduction from soft gates
        _fired_gates: list = []
        chan_name = chan.config.name
        try:
            sig = chan.evaluate(
                symbol=symbol,
                candles=ctx.candles,
                indicators=ctx.indicators,
                smc_data=ctx.smc_data,
                spread_pct=ctx.spread_pct,
                volume_24h_usd=volume_24h,
                regime=ctx.regime_result.regime.value,
            )
        except Exception as exc:
            log.debug("Channel {} eval error for {}: {}", chan_name, symbol, exc)
            return None, None
        if sig is None:
            return None, None

        # Record wall-clock time of signal detection for latency tracking.
        sig.detected_at = time.time()

        setup = self._evaluate_setup(chan_name, sig, ctx)
        if not setup.channel_compatible or not setup.regime_compatible:
            log.debug("Rejected {} {} setup: {}", symbol, chan_name, setup.reason)
            return None, None

        execution = self._evaluate_execution(sig, ctx, setup)
        if not execution.passed:
            log.debug("Rejected {} {} execution: {}", symbol, chan_name, execution.reason)
            return None, None

        # ── Filter 1: MTF Confluence Gate ──────────────────────────────────
        # Resolve the regime key early so regime-specific MTF config can
        # adjust the min_score and per-TF weight multipliers below.
        # (The same key is reused later for the regime penalty multiplier.)
        _regime_name = getattr(ctx.regime_result, "regime", None)
        if _regime_name is None:
            _regime_key = ""
        elif hasattr(_regime_name, "value"):
            _regime_key = _regime_name.value
        else:
            _regime_key = str(_regime_name)

        # Look up this channel's gate profile and penalty weights.
        # Unknown channels default to an empty profile (all gates on via .get(key, True))
        # and empty weights (gate-specific defaults apply via .get(key, default)).
        _gate_profile = _CHANNEL_GATE_PROFILE.get(chan_name, {})
        _penalty_weights = _CHANNEL_PENALTY_WEIGHTS.get(chan_name, {})
        if _gate_profile.get("mtf", True):
            # Base MTF min_score: relaxed for SCALP (range-fade setups need less confluence)
            _base_mtf_min_score = 0.4 if chan_name == "360_SCALP" else 0.5
            # Override with regime-specific min_score when configured
            _mtf_cfg = _MTF_REGIME_CONFIG.get(_regime_key, {})
            _mtf_min_score = _mtf_cfg.get("min_score", _base_mtf_min_score)
            # Build TF weight overrides from the regime config
            _higher_tfs = {"4h", "1d"}
            _lower_tfs = {"1m", "5m", "15m"}
            _tf_weight_overrides: Dict[str, float] = {}
            if _mtf_cfg:
                _hw = _mtf_cfg.get("higher_tf_weight", 1.0)
                _lw = _mtf_cfg.get("lower_tf_weight", 1.0)
                for _tf in _higher_tfs:
                    _tf_weight_overrides[_tf] = _MTF_TF_WEIGHTS.get(_tf, 1.0) * _hw
                for _tf in _lower_tfs:
                    _tf_weight_overrides[_tf] = _MTF_TF_WEIGHTS.get(_tf, 1.0) * _lw
            mtf_data: Dict[str, Dict[str, float]] = {}
            for tf_label, ind in ctx.indicators.items():
                ema_fast = ind.get("ema9_last")
                ema_slow = ind.get("ema21_last")
                cd = ctx.candles.get(tf_label, {})
                closes = cd.get("close", [])
                if ema_fast is not None and ema_slow is not None and len(closes) > 0:
                    mtf_data[tf_label] = {
                        "ema_fast": float(ema_fast),
                        "ema_slow": float(ema_slow),
                        "close": float(closes[-1]),
                    }
            mtf_allowed, mtf_reason = check_mtf_gate(
                sig.direction.value,
                mtf_data,
                min_score=_mtf_min_score,
                tf_weight_overrides=_tf_weight_overrides or None,
            )
            if not mtf_allowed:
                log.debug("MTF gate blocked {} {}: {}", symbol, chan_name, mtf_reason)
                return None, None

        # Resolve regime penalty multiplier for all soft gates below.
        # Scalp channels in QUIET regime use a higher multiplier to ensure
        # only top-tier mean-reversion setups pass the quality bar.
        if _regime_key == "QUIET" and chan_name.startswith("360_SCALP"):
            regime_mult = _SCALP_QUIET_REGIME_PENALTY
        else:
            regime_mult = _REGIME_PENALTY_MULTIPLIER.get(_regime_key, 1.0)

        # ── Filter 2: VWAP Extension Rejection ─────────────────────────────
        if _gate_profile.get("vwap", True):
            try:
                _primary_tf = self._get_primary_timeframe(chan_name)
                _cd = self._resolve_candles(ctx.candles, _primary_tf)
                _vwap_result = compute_vwap(
                    _cd.get("high", []),
                    _cd.get("low", []),
                    _cd.get("close", []),
                    _cd.get("volume", []),
                )
                vwap_allowed, vwap_reason = check_vwap_extension(
                    sig.direction.value, sig.entry, _vwap_result
                )
                if not vwap_allowed:
                    _base = _penalty_weights.get("vwap", 12.0)
                    _scaled = round(_base * regime_mult, 1)
                    soft_penalty += _scaled
                    _fired_gates.append("VWAP")
                    log.debug(
                        "SOFT_PENALTY {} {} {:+.1f} (base={:.0f} × regime={:.1f}) total={:.1f}: {}",
                        symbol, chan_name, _scaled, _base, regime_mult, soft_penalty, vwap_reason,
                    )
            except Exception as _vwap_exc:
                log.debug("VWAP gate error for {} {} (fail open): {}", symbol, chan_name, _vwap_exc)

        # ── Filter 3: Kill Zone / Session Filter ────────────────────────────
        if _gate_profile.get("kill_zone", True):
            # Relaxed minimum multiplier for scalp signals (scalps can trade lower-liquidity windows)
            _kz_min_mult = 0.40 if chan_name == "360_SCALP" else 0.50
            kz_allowed, kz_reason = check_kill_zone_gate(minimum_multiplier=_kz_min_mult)
            if not kz_allowed:
                _base = _penalty_weights.get("kill_zone", 10.0)
                _scaled = round(_base * regime_mult, 1)
                soft_penalty += _scaled
                _fired_gates.append("KZ")
                log.debug(
                    "SOFT_PENALTY {} {} {:+.1f} (base={:.0f} × regime={:.1f}) total={:.1f}: {}",
                    symbol, chan_name, _scaled, _base, regime_mult, soft_penalty, kz_reason,
                )

        # ── Filter 4: OI + Funding Rate Gate ────────────────────────────────
        _funding_rate: Optional[float] = None
        if _gate_profile.get("oi", True) and self.order_flow_store is not None:
            try:
                _oi_tf = self._get_primary_timeframe(chan_name)
                _oi_cd = self._resolve_candles(ctx.candles, _oi_tf)
                _prices = _oi_cd.get("close", [])
                _oi_snaps = list(getattr(self.order_flow_store, "_oi", {}).get(symbol, []))
                _oi_values = [s.open_interest for s in _oi_snaps]
                if _prices and _oi_values:
                    oi_analysis = analyse_oi(_prices, _oi_values)
                    _fr = oi_analysis.latest_funding_rate
                    if isinstance(_fr, (int, float)):
                        _funding_rate = float(_fr)
                    oi_allowed, oi_reason = check_oi_gate(sig.direction.value, oi_analysis)
                    if not oi_allowed:
                        _base = _penalty_weights.get("oi", 15.0)
                        _scaled = round(_base * regime_mult, 1)
                        soft_penalty += _scaled
                        _fired_gates.append("OI")
                        log.debug(
                            "SOFT_PENALTY {} {} {:+.1f} (base={:.0f} × regime={:.1f}) total={:.1f}: {}",
                            symbol, chan_name, _scaled, _base, regime_mult, soft_penalty, oi_reason,
                        )
            except Exception as _oi_exc:
                log.debug("OI gate error for {} {} (fail open): {}", symbol, chan_name, _oi_exc)

        # ── Filter 5: Cross-Asset Correlation ───────────────────────────────
        if _gate_profile.get("cross_asset", True) and symbol not in ("BTCUSDT", "ETHUSDT"):
            try:
                _asset_states: List[AssetState] = []
                for _major in ("BTCUSDT", "ETHUSDT"):
                    _major_cd = self.data_store.get_candles(_major, "5m") or {}
                    _major_closes = _major_cd.get("close", [])
                    if len(_major_closes) >= 2:
                        _trend, _pct = self._classify_macro_trend(_major_closes)
                        _asset_states.append(
                            AssetState(symbol=_major, trend=_trend, price_change_pct=_pct)
                        )
                if _asset_states:
                    ca_allowed, ca_reason = check_cross_asset_gate(
                        sig.direction.value, symbol, _asset_states
                    )
                    if not ca_allowed:
                        log.debug(
                            "Cross-asset gate blocked {} {}: {}", symbol, chan_name, ca_reason
                        )
                        return None, None
            except Exception as _ca_exc:
                log.debug(
                    "Cross-asset gate error for {} {} (fail open): {}", symbol, chan_name, _ca_exc
                )

        # ── Filter 6: Spoofing / Layering Detection ───────────────────────
        if _gate_profile.get("spoof", True):
            try:
                _ob = self._get_order_book_depth(symbol)
                spoof_allowed, spoof_reason = check_spoof_gate(
                    sig.direction.value, _ob, sig.entry
                )
                if not spoof_allowed:
                    _base = _penalty_weights.get("spoof", 10.0)
                    _scaled = round(_base * regime_mult, 1)
                    soft_penalty += _scaled
                    _fired_gates.append("SPOOF")
                    log.debug(
                        "SOFT_PENALTY {} {} {:+.1f} (base={:.0f} × regime={:.1f}) total={:.1f}: {}",
                        symbol, chan_name, _scaled, _base, regime_mult, soft_penalty, spoof_reason,
                    )
            except Exception as _spoof_exc:
                log.debug(
                    "Spoof gate error for {} {} (fail open): {}", symbol, chan_name, _spoof_exc
                )

        # ── Filter 7: Cross-Timeframe Volume Divergence ───────────────────
        if _gate_profile.get("volume_div", True):
            try:
                _vol_primary_tf = self._get_primary_timeframe(chan_name)
                # Relaxed spike threshold for scalp signals (volume spikes ARE valid for scalps)
                _vol_spike_thresh = 2.5 if chan_name == "360_SCALP" else 2.0
                vol_div_allowed, vol_div_reason = check_volume_divergence_gate(
                    sig.direction.value, ctx.candles, _vol_primary_tf,
                    spike_threshold=_vol_spike_thresh,
                    regime=_regime_key if _regime_key else None,
                )
                if not vol_div_allowed:
                    _base = _penalty_weights.get("volume_div", 10.0)
                    _scaled = round(_base * regime_mult, 1)
                    soft_penalty += _scaled
                    _fired_gates.append("VOL_DIV")
                    log.debug(
                        "SOFT_PENALTY {} {} {:+.1f} (base={:.0f} × regime={:.1f}) total={:.1f}: {}",
                        symbol, chan_name, _scaled, _base, regime_mult, soft_penalty, vol_div_reason,
                    )
            except Exception as _vol_div_exc:
                log.debug(
                    "Volume divergence gate error for {} {} (fail open): {}",
                    symbol, chan_name, _vol_div_exc,
                )

        # ── Filter 8: Signal Clustering Suppression ───────────────────────
        if _gate_profile.get("cluster", True):
            cluster_allowed, cluster_reason = self.cluster_suppressor.check_cluster_gate(
                symbol, sig.direction.value
            )
            if not cluster_allowed:
                _base = _penalty_weights.get("cluster", 8.0)
                _scaled = round(_base * regime_mult, 1)
                soft_penalty += _scaled
                _fired_gates.append("CLUSTER")
                log.debug(
                    "SOFT_PENALTY {} {} {:+.1f} (base={:.0f} × regime={:.1f}) total={:.1f}: {}",
                    symbol, chan_name, _scaled, _base, regime_mult, soft_penalty, cluster_reason,
                )

        risk = self._evaluate_risk(sig, ctx, setup, chan_name=chan_name)
        if not risk.passed:
            log.debug("Rejected {} {} risk: {}", symbol, chan_name, risk.reason)
            return None, None
        self._apply_risk_plan_to_signal(sig, risk)

        # ── Correlated position exposure cap ───────────────────────────────
        # Block new scalp signals when too many same-direction active signals
        # exist already, to limit correlated BTC-driven stop-out risk.
        if chan_name in _SCALP_CHANNELS:
            same_dir_count = sum(
                1
                for s in self.router.active_signals.values()
                if s.direction == sig.direction and s.channel in _SCALP_CHANNELS
            )
            if same_dir_count >= MAX_CORRELATED_SCALP_SIGNALS:
                log.info(
                    "Correlated exposure cap reached for {} {} (direction={}, active={}): "
                    "blocking signal",
                    symbol, chan_name, sig.direction.value, same_dir_count,
                )
                return None, None

        cross_verified = await self._verify_cross_exchange(
            symbol, sig.direction.value, sig.entry
        ) if chan_name not in _SCALP_CHANNELS else None

        # Fetch AI sentiment only for SPOT/GEM channels (4h/1d timeframes where
        # 10 s of network latency is irrelevant).  SCALP/SWING receive 0.0
        # (neutral) so the hot path has zero extra latency.
        sentiment_score = 0.0
        if chan_name in ("360_SPOT", "360_GEM"):
            try:
                ai_result = await get_ai_insight(symbol)
                sentiment_score = ai_result.score
                sig.ai_sentiment_label = ai_result.label
                sig.ai_sentiment_summary = ai_result.summary
            except Exception as _exc:
                log.debug("Sentiment fetch failed for {}: {}", symbol, _exc)

        legacy_confidence = self._compute_base_confidence(
            symbol,
            volume_24h,
            sig,
            ctx,
            cross_verified,
            chan_name=chan_name,
            funding_rate=_funding_rate,
            sentiment_score=sentiment_score,
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
        # Apply ML feedback adjustment based on historical outcomes for this
        # channel / setup combination.
        fb_adj = self.feedback_loop.get_confidence_adjustment(
            setup_score.components, chan_name, setup.setup_class.value
        )
        if fb_adj != 0.0:
            sig.confidence += fb_adj
            log.debug(
                "Feedback adjustment for {} {} {}: {:+.1f} → {:.1f}",
                symbol, chan_name, setup.setup_class.value, fb_adj, sig.confidence,
            )

        # Chart pattern bonus: detect confirming patterns from primary-TF candles
        primary_tf = self._get_primary_timeframe(chan_name)
        primary_candles = self._resolve_candles(ctx.candles, primary_tf)
        if primary_candles:
            try:
                patterns = detect_patterns(primary_candles)
                pat_bonus = pattern_confidence_bonus(patterns, sig.direction.value)
                if pat_bonus != 0.0:
                    sig.confidence += pat_bonus
                    log.debug(
                        "Chart pattern bonus {} {}: {:+.2f} (patterns={})",
                        symbol, chan_name, pat_bonus,
                        [p["pattern"] for p in patterns],
                    )
                # Record confirming pattern names for downstream consumers.
                confirming_names = []
                for _p in patterns:
                    _pname = _p.get("pattern", "")
                    if sig.direction.value == "LONG":
                        if _pname in _CHART_BULLISH_PATTERNS:
                            confirming_names.append(_pname)
                        elif _pname == "BB_SQUEEZE" and _p.get("expansion_direction") == "UP":
                            confirming_names.append(_pname)
                    else:
                        if _pname in _CHART_BEARISH_PATTERNS:
                            confirming_names.append(_pname)
                        elif _pname == "BB_SQUEEZE" and _p.get("expansion_direction") == "DOWN":
                            confirming_names.append(_pname)
                if confirming_names:
                    sig.chart_pattern_names = ",".join(confirming_names)
            except Exception as _exc:
                log.debug("Chart pattern detection error for {}: {}", symbol, _exc)

            # PR_05: candlestick pattern engine — confidence modifier (not hard gate)
            try:
                _open_arr = primary_candles.get("open", [])
                _high_arr = primary_candles.get("high", [])
                _low_arr = primary_candles.get("low", [])
                _close_arr = primary_candles.get("close", [])
                _vol_arr = primary_candles.get("volume", [])
                if len(_close_arr) >= 3 and len(_open_arr) >= 3:
                    _cp_results = detect_all_patterns(
                        np.asarray(_open_arr),
                        np.asarray(_high_arr),
                        np.asarray(_low_arr),
                        np.asarray(_close_arr),
                        np.asarray(_vol_arr) if _vol_arr else None,
                    )
                    # Store in smc_data for downstream consumers
                    ctx.smc_data["chart_patterns"] = _cp_results
                    # Filter to direction-aligned patterns only
                    _aligned = [
                        p for p in _cp_results
                        if p.direction == sig.direction.value or p.direction == "NEUTRAL"
                    ]
                    if _aligned:
                        _cp_bonus = sum(p.confidence_bonus for p in _aligned)
                        sig.confidence = max(0.0, min(100.0, sig.confidence + _cp_bonus))
                        _cp_names = ", ".join(p.name for p in _aligned)
                        # Append to chart_pattern_names (may already have legacy pattern names)
                        if sig.chart_pattern_names:
                            sig.chart_pattern_names = sig.chart_pattern_names + ", " + _cp_names
                        else:
                            sig.chart_pattern_names = _cp_names
                        log.debug(
                            "Candlestick pattern bonus {} {}: {:+.2f} ({})",
                            symbol, chan_name, _cp_bonus, _cp_names,
                        )
            except Exception as _exc:
                log.debug("Candlestick pattern detection error for {}: {}", symbol, _exc)

        # This augments the hard MTF gate above with a continuous confidence signal.
        try:
            _mtf_conf_data: Dict[str, Dict[str, float]] = {}
            for _tf in ("5m", "15m", "1h", "4h"):
                _ind = ctx.indicators.get(_tf, {})
                _ema_fast = _ind.get("ema9_last")
                _ema_slow = _ind.get("ema21_last")
                _cd = ctx.candles.get(_tf, {})
                _closes = _cd.get("close", [])
                if _ema_fast is not None and _ema_slow is not None and _closes:
                    _mtf_conf_data[_tf] = {
                        "ema_fast": float(_ema_fast),
                        "ema_slow": float(_ema_slow),
                        "close": float(_closes[-1]),
                    }
            if _mtf_conf_data:
                _mtf_result = compute_mtf_confluence(sig.direction.value, _mtf_conf_data)
                sig.mtf_score = _mtf_result.score
                if _mtf_result.is_strong:
                    sig.confidence += 3.0
                    log.debug(
                        "MTF strong-confluence boost {} {}: +3.0 (score={:.2f})",
                        symbol, chan_name, _mtf_result.score,
                    )
                elif not _mtf_result.is_aligned:
                    if MTF_HARD_BLOCK:
                        log.info(
                            "Signal blocked by MTF hard gate: {} {} (score={:.2f})",
                            symbol, chan_name, _mtf_result.score,
                        )
                        sig = None
                    else:
                        sig.confidence -= 5.0
                        log.debug(
                            "MTF misalignment penalty {} {}: -5.0 (score={:.2f})",
                            symbol, chan_name, _mtf_result.score,
                        )
        except Exception as _mtf_exc:
            log.debug("MTF confidence modifier error for {} {} (fail open): {}", symbol, chan_name, _mtf_exc)

        # Hard MTF block: signal was vetoed — return immediately.
        if sig is None:
            return None, cross_verified

        # Apply adaptive confidence decay based on signal freshness.
        # apply_confidence_decay clamps the final value to [0, 100].
        sig.confidence = apply_confidence_decay(
            confidence=sig.confidence,
            signal_generated_at=t0_signal,
            current_time=time.monotonic(),
            channel=chan_name,
        )
        sig.confidence = self._clamp_confidence(sig.confidence)
        sig.post_ai_confidence = sig.confidence
        # Apply accumulated soft-gate confidence penalty (regime-scaled).
        # A second clamp follows to prevent negative confidence after large penalty sums.
        if soft_penalty > 0.0:
            sig.confidence -= soft_penalty
            log.debug(
                "Soft-gate penalty applied {} {}: -{:.1f} → {:.1f}",
                symbol, chan_name, soft_penalty, sig.confidence,
            )
        sig.soft_penalty_total = soft_penalty
        sig.regime_penalty_multiplier = regime_mult
        sig.soft_gate_flags = ",".join(_fired_gates)
        # Second clamp: ensures confidence stays in [0, 100] after soft-penalty deduction.
        sig.confidence = self._clamp_confidence(sig.confidence)
        # Classify signal into quality tier based on final confidence.
        sig.signal_tier = classify_signal_tier(sig.confidence)
        # ── PR_09: Composite Signal Scoring Engine ────────────────────────
        # Overwrites sig.confidence and sig.signal_tier with the structured
        # 0-100 composite score.  Merges new dimension breakdown into the
        # existing component_scores so that downstream format checks still
        # see the "market"/"execution"/"risk" keys set earlier.
        try:
            _primary_tf = self._get_primary_timeframe(chan_name)
            _primary_ind = ctx.indicators.get(_primary_tf, {})
            _primary_cd = self._resolve_candles(ctx.candles, _primary_tf)
            _closes_arr = _primary_cd.get("close", [])
            _vol_arr = _primary_cd.get("volume", [])
            if _closes_arr and _vol_arr:
                _usd_vols = [c * v for c, v in zip(_closes_arr[-20:], _vol_arr[-20:])]
                _volume_last_usd = float(_usd_vols[-1]) if _usd_vols else 0.0
                _volume_avg_usd = float(np.mean(_usd_vols)) if _usd_vols else 0.0
            else:
                _volume_last_usd = 0.0
                _volume_avg_usd = 0.0
            _atr_pct = 50.0
            if ctx.regime_context is not None:
                try:
                    _atr_pct = float(ctx.regime_context.atr_percentile)
                except (TypeError, ValueError):
                    pass
            _scoring_inp = ScoringInput(
                sweeps=ctx.smc_result.sweeps,
                mss=ctx.smc_result.mss,
                fvg_zones=ctx.smc_result.fvg,
                regime=_regime_key,
                setup_class=sig.setup_class,
                atr_percentile=_atr_pct,
                volume_last_usd=_volume_last_usd,
                volume_avg_usd=_volume_avg_usd,
                macd_histogram_last=_primary_ind.get("macd_histogram_last"),
                macd_histogram_prev=_primary_ind.get("macd_histogram_prev"),
                rsi_last=_primary_ind.get("rsi_last"),
                ema_fast=_primary_ind.get("ema9_last"),
                ema_slow=_primary_ind.get("ema21_last"),
                adx_last=_primary_ind.get("adx_last"),
                direction=sig.direction.value,
                chart_patterns=ctx.smc_data.get("chart_patterns", []),
                mtf_score=getattr(sig, "mtf_score", 0.0),
            )
            _score_result = _scoring_engine.score(_scoring_inp)
            # Merge new dimension scores into component_scores (preserves existing keys)
            sig.component_scores.update(_score_result)
            sig.confidence = _score_result["total"]
            if _score_result["total"] >= 80:
                sig.signal_tier = "A+"
            elif _score_result["total"] >= 65:
                sig.signal_tier = "B"
            elif _score_result["total"] >= 50:
                sig.signal_tier = "WATCHLIST"
            else:
                return None, cross_verified
            log.debug(
                "PR09 score {} {} → {:.1f} (tier={}) smc={} regime={} vol={} ind={} pat={} mtf={}",
                symbol, chan_name, _score_result["total"], sig.signal_tier,
                _score_result["smc"], _score_result["regime"], _score_result["volume"],
                _score_result["indicators"], _score_result["patterns"], _score_result["mtf"],
            )
        except Exception as _score_exc:
            log.debug("PR09 scoring engine error for {} {} (fail open): {}", symbol, chan_name, _score_exc)

        # ── PR_12: Statistical False-Positive Filter ──────────────────────
        # Apply rolling win-rate gate after scoring. Fail-open when no history.
        try:
            _sf_allow, _sf_conf, _sf_reason = _stat_filter.check(
                channel=chan_name,
                pair=symbol,
                regime=_regime_key,
                current_confidence=sig.confidence,
            )
            if not _sf_allow:
                log.debug(
                    "stat_filter suppressed {}/{}: {}",
                    symbol, chan_name, _sf_reason,
                )
                self.suppression_tracker.record(SuppressionEvent(
                    symbol=symbol,
                    channel=chan_name,
                    reason=REASON_STAT_FILTER,
                    regime=_regime_key,
                    would_be_confidence=sig.confidence,
                ))
                return None, cross_verified
            sig.confidence = _sf_conf
            if "penalty" in _sf_reason:
                _existing_flags = sig.soft_gate_flags or ""
                sig.soft_gate_flags = (_existing_flags + f",{_sf_reason}").lstrip(",")
        except Exception as _sf_exc:
            log.debug("stat_filter error for {} {} (fail open): {}", symbol, chan_name, _sf_exc)

        min_conf = self.confidence_overrides.get(chan_name, chan.config.min_confidence)
        # Apply regime soft-gate confidence adjustments
        for _reason, _delta in ctx.confidence_adjustments.items():
            sig.confidence += _delta
            _existing = sig.soft_gate_flags or ""
            sig.soft_gate_flags = (_existing + f",REGIME_ADJUSTED:{_reason}").lstrip(",")
            log.debug("regime_penalty_applied {} {} reason={} delta={:.1f}", symbol, chan_name, _reason, _delta)
        # QUIET regime safety net for scalp channels: only the highest-quality
        # mean-reversion setups are allowed through when the market is compressed.
        if _regime_key == "QUIET" and chan_name.startswith("360_SCALP"):
            if sig.confidence < QUIET_SCALP_MIN_CONFIDENCE:
                log.info(
                    "QUIET_SCALP_BLOCK {} {} conf={:.1f} < min={:.1f}",
                    symbol, chan_name, sig.confidence, QUIET_SCALP_MIN_CONFIDENCE,
                )
                self.suppression_tracker.record(SuppressionEvent(
                    symbol=symbol,
                    channel=chan_name,
                    reason=REASON_CONFIDENCE,
                    regime=_regime_key,
                    would_be_confidence=sig.confidence,
                ))
                return None, cross_verified
        # WATCHLIST tier: signals with confidence 50-64 are kept as WATCHLIST
        # instead of being discarded.  Only the SCALP channel family generates
        # watchlist alerts; SWING and SPOT require higher confidence.
        _watchlist_confidence = 50.0
        if (
            sig.signal_tier == "WATCHLIST"
            and chan_name in _SCALP_CHANNELS
            and sig.confidence >= _watchlist_confidence
        ):
            # Keep as WATCHLIST — clear entry/SL/TP for zone-alert-only format
            self._populate_signal_context(sig, volume_24h, ctx)
            return sig, cross_verified
        if (
            sig.confidence < min_conf
            or sig.component_scores.get("market", 0.0) < 12.0
            or sig.component_scores.get("execution", 0.0) < 10.0
            or sig.component_scores.get("risk", 0.0) < 10.0
        ):
            self.suppression_tracker.record(SuppressionEvent(
                symbol=symbol,
                channel=chan_name,
                reason=REASON_CONFIDENCE,
                regime=_regime_key,
                would_be_confidence=sig.confidence,
            ))
            return None, cross_verified
        self._populate_signal_context(sig, volume_24h, ctx)
        return sig, cross_verified

    async def _scan_symbol(self, symbol: str, volume_24h: float) -> None:
        """Run all channel evaluations for one symbol."""
        ctx = await self._build_scan_context(symbol, volume_24h)
        if ctx is None:
            return
        ticks = self.data_store.ticks.get(symbol, [])
        for chan in self.channels:
            chan_name = chan.config.name
            if self._should_skip_channel(symbol, chan_name, ctx):
                continue
            # Re-detect SMC with channel-specific timeframe preference when available.
            # This ensures scalp channels see low-TF sweeps first while swing/spot
            # channels only act on high-TF institutional sweeps.
            _ch_tfs = _CHANNEL_SMC_TIMEFRAMES.get(chan_name)
            if _ch_tfs is not None:
                try:
                    _smc_r = self.smc_detector.detect(
                        symbol, ctx.candles, ticks, self.order_flow_store,
                        lookback=SMC_SCALP_LOOKBACK,
                        tolerance_pct=SMC_SCALP_TOLERANCE_PCT,
                        smc_timeframes=_ch_tfs,
                    )
                    ctx_for_chan = _dc.replace(
                        ctx,
                        smc_result=_smc_r,
                        smc_data=_smc_r.as_dict(),
                    )
                except Exception as _exc:
                    log.debug("Per-channel SMC re-detect failed for {} {}: {}", symbol, chan_name, _exc)
                    ctx_for_chan = ctx
            else:
                ctx_for_chan = ctx
            sig, cross_verified = await self._prepare_signal(symbol, volume_24h, chan, ctx_for_chan)
            if sig is None:
                continue
            # Only start scan cooldown after the signal has been accepted by the
            # queue; rejected/dropped signals must not suppress later scans.
            if not await self._enqueue_signal(sig):
                continue
            self._set_cooldown(symbol, chan_name)
            self.cluster_suppressor.record_signal(symbol, sig.direction.value)

    async def _lightweight_tier3_scan(self) -> None:
        """Lightweight volume/momentum scan for Tier 3 pairs.

        Checks whether any Tier 3 pair has experienced a volume surge exceeding
        ``TIER3_VOLUME_SURGE_MULTIPLIER`` × its previous 24h volume.  Qualifying
        pairs are promoted to Tier 2 via :meth:`PairManager.check_promotions` so
        that they receive full SWING+SPOT channel evaluation on the next cycle.

        No order book fetches, kline lookups, or indicator computation are
        performed — this is intentionally minimal to avoid Binance weight
        exhaustion.
        """
        tier3_pairs = [
            (sym, info)
            for sym, info in self.pair_mgr.pairs.items()
            if info.tier == PairTier.TIER3
        ]
        if not tier3_pairs:
            return
        log.debug("Tier 3 lightweight scan: %d pairs", len(tier3_pairs))
        promoted = self.pair_mgr.check_promotions()
        if promoted:
            log.info(
                "Tier 3 auto-promoted %d pairs to Tier 2: %s",
                len(promoted), promoted[:10],
            )

    async def _scan_gem_symbol(self, symbol: str) -> None:
        """Evaluate a single symbol through the gem scanner and enqueue if qualifying.

        Loads daily (1d) and weekly (1w) candles from the data store, invokes
        :meth:`src.gem_scanner.GemScanner.scan`, and converts a qualifying
        :class:`src.gem_scanner.GemSignal` into a :class:`src.channels.base.Signal`
        for the ``360_GEM`` channel before enqueuing it.
        """
        if self.gem_scanner is None or not self.gem_scanner.enabled:
            return

        # Adjust gem scanner thresholds for the current market regime (feature 7)
        try:
            self.gem_scanner.adjust_for_regime(self._last_market_regime)
        except Exception as _rexc:
            log.debug("Regime adjustment for gem scanner failed: {}", _rexc)

        daily_raw = self.data_store.get_candles(symbol, "1d")
        if not daily_raw:
            return
        daily_candles = _normalize_candle_dict(daily_raw)

        weekly_raw = self.data_store.get_candles(symbol, "1w")
        weekly_candles = _normalize_candle_dict(weekly_raw) if weekly_raw else None

        # Fetch on-chain enrichment data (feature 2) — fail-open on errors
        onchain_data: Optional[dict] = None
        if self.onchain_client is not None:
            try:
                oc_result = await self.onchain_client.get_exchange_flow(symbol)
                # Interpret a high (>7.5) exchange-flow score as whale accumulation
                onchain_data = {
                    "whale_accumulation": oc_result.score >= 7.5,
                    "social_volume_ratio": 1.0,  # placeholder; extend with real social API
                    "unlock_days": None,          # placeholder; extend with token unlock API
                }
            except Exception as _oexc:
                log.debug("On-chain enrichment fetch failed for {}: {}", symbol, _oexc)

        gem_sig = self.gem_scanner.scan(symbol, daily_candles, weekly_candles, onchain_data)
        if gem_sig is None:
            return

        if gem_sig.confidence < CHANNEL_GEM.min_confidence:
            log.debug(
                "Gem signal for {} below min confidence ({:.1f} < {})",
                symbol, gem_sig.confidence, CHANNEL_GEM.min_confidence,
            )
            return

        close = gem_sig.current_price
        # CHANNEL_GEM.sl_pct_range stores values as fractions (e.g. 0.10 = 10%),
        # used directly as a multiplier — NOT divided by 100 — to produce the
        # SL distance for deeply discounted altcoins (typically 10–30% from entry).
        sl_dist = close * CHANNEL_GEM.sl_pct_range[0]
        sl = close - sl_dist
        tp1 = close + sl_dist * CHANNEL_GEM.tp_ratios[0]
        tp2 = close + sl_dist * CHANNEL_GEM.tp_ratios[1]
        tp3 = close + sl_dist * CHANNEL_GEM.tp_ratios[2]

        if sl >= close:
            return

        sig = _Signal(
            channel=CHANNEL_GEM.name,
            symbol=symbol,
            direction=Direction.LONG,
            entry=close,
            stop_loss=round(sl, 8),
            tp1=round(tp1, 8),
            tp2=round(tp2, 8),
            tp3=round(tp3, 8),
            trailing_active=True,
            trailing_desc=f"{CHANNEL_GEM.trailing_atr_mult}×ATR",
            confidence=gem_sig.confidence,
            ai_sentiment_label="",
            ai_sentiment_summary="",
            risk_label="Speculative",
            timestamp=utcnow(),
            signal_id=f"GEM-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
            original_sl_distance=sl_dist,
            valid_for_minutes=SIGNAL_VALID_FOR_MINUTES.get(CHANNEL_GEM.name, 1440),
        )

        if await self._enqueue_signal(sig):
            self.gem_scanner.record_published()
            log.info(
                "💎 GEM signal enqueued: {} @ {:.8g} (confidence={:.1f})",
                symbol, close, gem_sig.confidence,
            )
