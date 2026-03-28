"""Simulation Module (PR08) – historical replay & performance validation.

Replays 7–30 days of historical candle data through the new filter chain
(high-probability filter, dynamic SL/TP, and batch scanning logic) to
validate expected performance **before** live deployment.

Output metrics
--------------
* hit_rate          – Proportion of signals that reached TP1 (%).
* sl_hit_rate       – Proportion of signals stopped out (%).
* avg_latency_ms    – Mean signal detection-to-post latency.
* best_tp_rate      – Proportion of signals that reached TP2 or TP3.
* suppression_rate  – Proportion of raw setups filtered by the probability gate.

Usage::

    sim = Simulator(historical_candles, channels)
    results = sim.run(days=14, probability_threshold=70.0)
    sim.export_csv(results, "sim_output.csv")
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.scanner.filter_module import get_pair_probability, DEFAULT_PROBABILITY_THRESHOLD
from src.volatility_metrics import compute_atr_pct, calculate_dynamic_sl_tp

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SimCandle:
    """A single historical OHLCV candle used as replay input."""

    timestamp: float   # Unix epoch seconds (open time)
    open: float
    high: float
    low: float
    close: float
    volume: float      # Base-asset volume
    symbol: str = ""
    timeframe: str = "5m"


@dataclass
class SimSignal:
    """A signal generated during simulation."""

    symbol: str
    direction: str       # "LONG" or "SHORT"
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    probability_score: float
    regime: str = ""
    channel: str = ""
    candle_timestamp: float = 0.0
    # Outcome fields (populated by _evaluate_outcome)
    outcome: str = ""    # "TP1", "TP2", "TP3", "SL", "OPEN"
    pnl_pct: float = 0.0
    latency_ms: float = 0.0


@dataclass
class SimResult:
    """Aggregated performance metrics for one simulation run."""

    days: int
    total_setups: int
    total_signals: int        # After probability filter
    hit_rate_pct: float       # TP1+ hit rate
    sl_hit_rate_pct: float
    avg_latency_ms: float
    best_tp_rate_pct: float   # TP2+ hit rate
    suppression_rate_pct: float
    signals: List[SimSignal] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class Simulator:
    """Replays historical candle data through the signal filter chain.

    Parameters
    ----------
    historical_candles : dict
        Mapping of ``symbol → timeframe → list[SimCandle]``.
        Each inner list must be sorted by timestamp (ascending).
    channels : sequence
        Channel evaluator objects (must implement ``evaluate()``).
    probability_threshold : float
        Minimum probability score to allow a signal through the filter.
    """

    def __init__(
        self,
        historical_candles: Dict[str, Dict[str, List[SimCandle]]],
        channels: Sequence[Any],
        probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    ) -> None:
        self._candles = historical_candles
        self._channels = list(channels)
        self._threshold = probability_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        days: int = 14,
        regime: str = "RANGING",
    ) -> SimResult:
        """Run the simulation and return aggregated performance metrics.

        Parameters
        ----------
        days:
            Number of historical days to replay (7–30).  Candles older than
            this window are skipped.
        regime:
            Market regime label to pass to the filter and dynamic SL/TP.

        Returns
        -------
        SimResult
        """
        days = max(1, min(days, 30))
        cutoff_ts = time.time() - days * 86_400.0

        all_signals: List[SimSignal] = []
        total_setups = 0
        total_filtered = 0

        for symbol, tf_map in self._candles.items():
            candles_5m = tf_map.get("5m", [])
            recent = [c for c in candles_5m if c.timestamp >= cutoff_ts]
            if len(recent) < 50:
                continue

            for i in range(50, len(recent)):
                window = recent[i - 50: i]
                candle = recent[i]
                total_setups += 1

                # --- Compute ATR over the window ---
                atr_val = self._compute_atr(window)
                atr_pct = compute_atr_pct(atr_val, candle.close)

                # --- Probability filter (PR01) ---
                vol_usd = candle.volume * candle.close
                pair_data = {
                    "regime": regime,
                    "spread_pct": 0.03,      # Simulated fixed spread
                    "volume_24h_usd": vol_usd * 288,  # ~5m candles in 24h
                    "atr_pct": atr_pct,
                    "hit_rate": 0.55,         # Conservative default
                }
                score = get_pair_probability(pair_data)
                if score < self._threshold:
                    total_filtered += 1
                    continue

                # --- Dynamic SL/TP (PR02) ---
                sl_mult, tp_ratios = calculate_dynamic_sl_tp(
                    pair=symbol,
                    regime=regime,
                    atr_pct=atr_pct,
                )
                sl_dist = atr_val * sl_mult

                # Produce one LONG and one SHORT candidate; keep the one
                # that would be triggered given the next candle's price action
                for direction in ("LONG", "SHORT"):
                    sig = self._make_signal(
                        symbol, direction, candle, sl_dist, tp_ratios,
                        score, regime,
                    )
                    if sig is None:
                        continue
                    # Evaluate outcome against subsequent candles
                    future = recent[i: min(i + 20, len(recent))]
                    self._evaluate_outcome(sig, future)
                    all_signals.append(sig)

        return self._aggregate(days, total_setups, total_filtered, all_signals)

    def export_csv(self, result: SimResult, filepath: str) -> None:
        """Write simulation results to a CSV file.

        Parameters
        ----------
        result:
            Output of :meth:`run`.
        filepath:
            Destination CSV file path.
        """
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            if not result.signals:
                fh.write("No signals generated\n")
                return
            writer = csv.DictWriter(fh, fieldnames=list(asdict(result.signals[0]).keys()))
            writer.writeheader()
            for sig in result.signals:
                writer.writerow(asdict(sig))
        log.info("Simulation results exported to %s (%d signals)", filepath, len(result.signals))

    def export_json(self, result: SimResult, filepath: str) -> None:
        """Write simulation results to a JSON file.

        Parameters
        ----------
        result:
            Output of :meth:`run`.
        filepath:
            Destination JSON file path.
        """
        payload = {
            "days": result.days,
            "total_setups": result.total_setups,
            "total_signals": result.total_signals,
            "hit_rate_pct": result.hit_rate_pct,
            "sl_hit_rate_pct": result.sl_hit_rate_pct,
            "avg_latency_ms": result.avg_latency_ms,
            "best_tp_rate_pct": result.best_tp_rate_pct,
            "suppression_rate_pct": result.suppression_rate_pct,
            "signals": [asdict(s) for s in result.signals],
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        log.info("Simulation results exported to %s (%d signals)", filepath, len(result.signals))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr(candles: List[SimCandle], period: int = 14) -> float:
        """Compute a simple ATR from the last *period* candles."""
        if len(candles) < 2:
            return candles[-1].close * 0.005 if candles else 1.0
        trs = []
        for i in range(1, min(period + 1, len(candles))):
            c = candles[i]
            prev_close = candles[i - 1].close
            tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else candles[-1].close * 0.005

    @staticmethod
    def _make_signal(
        symbol: str,
        direction: str,
        candle: SimCandle,
        sl_dist: float,
        tp_ratios: List[float],
        score: float,
        regime: str,
    ) -> Optional[SimSignal]:
        """Construct a SimSignal from the entry candle and dynamic SL/TP."""
        price = candle.close
        if sl_dist <= 0:
            return None
        if direction == "LONG":
            sl = price - sl_dist
            tp1 = price + sl_dist * tp_ratios[0]
            tp2 = price + sl_dist * tp_ratios[1]
            tp3 = price + sl_dist * (tp_ratios[2] if len(tp_ratios) > 2 else 2.0)
        else:
            sl = price + sl_dist
            tp1 = price - sl_dist * tp_ratios[0]
            tp2 = price - sl_dist * tp_ratios[1]
            tp3 = price - sl_dist * (tp_ratios[2] if len(tp_ratios) > 2 else 2.0)

        return SimSignal(
            symbol=symbol,
            direction=direction,
            entry=price,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            probability_score=score,
            regime=regime,
            candle_timestamp=candle.timestamp,
            latency_ms=50.0,   # Simulated fixed latency
        )

    @staticmethod
    def _evaluate_outcome(sig: SimSignal, future_candles: List[SimCandle]) -> None:
        """Set sig.outcome and sig.pnl_pct based on future price action."""
        for candle in future_candles:
            if sig.direction == "LONG":
                if candle.low <= sig.stop_loss:
                    sig.outcome = "SL"
                    risk = abs(sig.entry - sig.stop_loss)
                    sig.pnl_pct = -(risk / sig.entry * 100.0) if sig.entry > 0 else 0.0
                    return
                if candle.high >= sig.tp3:
                    sig.outcome = "TP3"
                    sig.pnl_pct = (sig.tp3 - sig.entry) / sig.entry * 100.0
                    return
                if candle.high >= sig.tp2:
                    sig.outcome = "TP2"
                    sig.pnl_pct = (sig.tp2 - sig.entry) / sig.entry * 100.0
                    return
                if candle.high >= sig.tp1:
                    sig.outcome = "TP1"
                    sig.pnl_pct = (sig.tp1 - sig.entry) / sig.entry * 100.0
                    return
            else:  # SHORT
                if candle.high >= sig.stop_loss:
                    sig.outcome = "SL"
                    risk = abs(sig.entry - sig.stop_loss)
                    sig.pnl_pct = -(risk / sig.entry * 100.0) if sig.entry > 0 else 0.0
                    return
                if candle.low <= sig.tp3:
                    sig.outcome = "TP3"
                    sig.pnl_pct = (sig.entry - sig.tp3) / sig.entry * 100.0
                    return
                if candle.low <= sig.tp2:
                    sig.outcome = "TP2"
                    sig.pnl_pct = (sig.entry - sig.tp2) / sig.entry * 100.0
                    return
                if candle.low <= sig.tp1:
                    sig.outcome = "TP1"
                    sig.pnl_pct = (sig.entry - sig.tp1) / sig.entry * 100.0
                    return
        sig.outcome = "OPEN"   # Not resolved within the look-forward window

    @staticmethod
    def _aggregate(
        days: int,
        total_setups: int,
        total_filtered: int,
        signals: List[SimSignal],
    ) -> SimResult:
        """Compute summary statistics from a list of evaluated SimSignals."""
        n = len(signals)
        if n == 0:
            suppression_pct = (
                total_filtered / total_setups * 100.0 if total_setups > 0 else 0.0
            )
            return SimResult(
                days=days,
                total_setups=total_setups,
                total_signals=0,
                hit_rate_pct=0.0,
                sl_hit_rate_pct=0.0,
                avg_latency_ms=0.0,
                best_tp_rate_pct=0.0,
                suppression_rate_pct=suppression_pct,
                signals=[],
            )

        tp1_plus = sum(1 for s in signals if s.outcome in ("TP1", "TP2", "TP3"))
        tp2_plus = sum(1 for s in signals if s.outcome in ("TP2", "TP3"))
        sl_hits = sum(1 for s in signals if s.outcome == "SL")
        avg_latency = sum(s.latency_ms for s in signals) / n

        total_raw = total_setups
        suppression_pct = (
            (total_filtered / total_raw * 100.0) if total_raw > 0 else 0.0
        )

        result = SimResult(
            days=days,
            total_setups=total_setups,
            total_signals=n,
            hit_rate_pct=tp1_plus / n * 100.0,
            sl_hit_rate_pct=sl_hits / n * 100.0,
            avg_latency_ms=avg_latency,
            best_tp_rate_pct=tp2_plus / n * 100.0,
            suppression_rate_pct=suppression_pct,
            signals=signals,
        )
        log.info(
            "Simulation complete: days=%d setups=%d signals=%d "
            "hit_rate=%.1f%% sl_rate=%.1f%% suppressed=%.1f%%",
            days, total_setups, n,
            result.hit_rate_pct,
            result.sl_hit_rate_pct,
            result.suppression_rate_pct,
        )
        return result
