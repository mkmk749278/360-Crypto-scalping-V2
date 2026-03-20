"""SMC Detection Orchestrator.

Provides :class:`SMCDetector` which bundles all Smart Money Concepts detection
logic into a single, reusable component.  The result is returned as an
:class:`SMCResult` dataclass so ``main.py._scan_symbol()`` stays thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ai_engine import WhaleAlert, detect_volume_delta_spike, detect_whale_trade
from src.order_flow import OrderFlowStore, is_oi_invalidated
from src.smc import FVGZone, LiquiditySweep, MSSSignal, detect_fvg, detect_liquidity_sweeps, detect_mss
from src.utils import get_logger

log = get_logger("detector")

# Lower-timeframe lookup used for MSS confirmation
_LTF_MAP: Dict[str, str] = {
    "4h": "1h",
    "1h": "15m",
    "15m": "5m",
    "5m": "1m",
}

# Ordered preference for SMC detection timeframes (most sensitive first)
_SMC_TIMEFRAMES = ("5m", "4h", "15m", "1m")

# Minimum number of candles required for CVD divergence detection.
# Must be >= the default lookback passed to detect_cvd_divergence (20).
_CVD_MIN_CANDLES: int = 21


@dataclass
class SMCResult:
    """Unified output of SMC detection for a single symbol."""

    sweeps: List[LiquiditySweep] = field(default_factory=list)
    mss: Optional[MSSSignal] = None
    fvg: List[FVGZone] = field(default_factory=list)
    whale_alert: Optional[WhaleAlert] = None
    volume_delta_spike: bool = False
    recent_ticks: List[Dict[str, Any]] = field(default_factory=list)
    oi_invalidated: bool = False
    cvd_divergence: Optional[str] = None  # "BULLISH", "BEARISH", or None

    def as_dict(self) -> dict:
        """Return a plain dict for backward-compat with channel evaluate() calls."""
        return {
            "sweeps": self.sweeps,
            "mss": self.mss,
            "fvg": self.fvg,
            "whale_alert": self.whale_alert,
            "volume_delta_spike": self.volume_delta_spike,
            "recent_ticks": self.recent_ticks,
            "oi_invalidated": self.oi_invalidated,
            "cvd_divergence": self.cvd_divergence,
        }


class SMCDetector:
    """Runs all SMC + whale/tape detection for a given symbol snapshot."""

    def detect(
        self,
        symbol: str,
        candles: Dict[str, Dict[str, Any]],
        ticks: List[Dict[str, Any]],
        order_flow_store: Optional[OrderFlowStore] = None,
    ) -> SMCResult:
        """Run full SMC detection and return an :class:`SMCResult`.

        Parameters
        ----------
        symbol:
            Trading symbol (used only for logging).
        candles:
            Dict of timeframe → OHLCV arrays, e.g. ``{"5m": {"high": ..., ...}}``.
        ticks:
            Recent trade ticks from the data store.
        order_flow_store:
            Optional :class:`src.order_flow.OrderFlowStore` for OI trend and
            CVD divergence checks.  When provided, detected sweeps are validated
            against the current OI trend (rising OI during a sweep in the
            opposing direction sets ``oi_invalidated = True``).  CVD divergence
            is also queried and attached to the result.
        """
        result = SMCResult()

        # ------------------------------------------------------------------
        # SMC detection (sweeps, MSS, FVG) across preferred timeframes
        # ------------------------------------------------------------------
        for tf_key in _SMC_TIMEFRAMES:
            cd = candles.get(tf_key)
            if cd is None or len(cd.get("close", [])) < 51:
                continue

            sweeps = detect_liquidity_sweeps(cd["high"], cd["low"], cd["close"])
            if not sweeps:
                continue

            result.sweeps = sweeps

            ltf_key = _LTF_MAP.get(tf_key, "1m")
            ltf_cd = candles.get(ltf_key)
            if ltf_cd and len(ltf_cd.get("close", [])) > 1:
                mss_sig = detect_mss(sweeps[0], ltf_cd["close"])
                result.mss = mss_sig

            result.fvg = detect_fvg(cd["high"], cd["low"], cd["close"])
            break  # use first timeframe that has a sweep

        # ------------------------------------------------------------------
        # Order flow validation (OI trend + CVD divergence)
        # ------------------------------------------------------------------
        if order_flow_store is not None and result.sweeps:
            primary_sweep = result.sweeps[0]
            oi_trend = order_flow_store.get_oi_trend(symbol)

            # Invalidate if OI is rising while we have a sweep signal.
            # Rising OI means new aggressive positions are entering against
            # the proposed reversal direction.
            if is_oi_invalidated(oi_trend, primary_sweep.direction.value):
                result.oi_invalidated = True
                log.debug(
                    "{}: OI RISING during {} sweep – signal invalidated",
                    symbol, primary_sweep.direction.value,
                )

            # CVD divergence: check if price/CVD diverge (confirms the sweep)
            tf_key_for_cvd = next(
                (tf for tf in _SMC_TIMEFRAMES if candles.get(tf) and
                 len(candles[tf].get("close", [])) >= _CVD_MIN_CANDLES),
                None,
            )
            if tf_key_for_cvd is not None:
                import numpy as np
                close_arr = np.asarray(
                    candles[tf_key_for_cvd]["close"], dtype=np.float64
                ).ravel()
                result.cvd_divergence = order_flow_store.get_cvd_divergence(
                    symbol, close_arr
                )
                if result.cvd_divergence:
                    log.debug(
                        "{}: CVD divergence detected – {}",
                        symbol, result.cvd_divergence,
                    )

        # ------------------------------------------------------------------
        # Whale / tape detection
        # ------------------------------------------------------------------
        if ticks:
            latest = ticks[-1]
            result.whale_alert = detect_whale_trade(
                latest.get("price", 0.0), latest.get("qty", 0.0)
            )
            recent = ticks[-100:]
            result.recent_ticks = recent

            buy_v = sum(
                t.get("qty", 0) * t.get("price", 0)
                for t in recent
                if not t.get("isBuyerMaker", True)
            )
            sell_v = sum(
                t.get("qty", 0) * t.get("price", 0)
                for t in recent
                if t.get("isBuyerMaker", True)
            )
            avg_delta = (buy_v + sell_v) / 2.0 if (buy_v + sell_v) > 0 else 0.0
            result.volume_delta_spike = detect_volume_delta_spike(
                buy_v - sell_v, avg_delta
            )

        return result
