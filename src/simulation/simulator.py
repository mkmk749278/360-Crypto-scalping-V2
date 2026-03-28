"""Simulation & Backtesting Module — historical replay engine.

Replays historical market data through the filter, dynamic SL/TP,
and signal pipeline to validate parameter changes before live
deployment.

PR 08 Implementation.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_logger

log = get_logger("simulator")


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    replay_days: int = 7                   # Number of days to replay
    probability_threshold: float = 70.0    # Filter threshold
    channels: List[str] = field(default_factory=lambda: ["360_SCALP"])
    sl_multiplier: float = 1.0             # SL adjustment
    tp_multiplier: float = 1.0             # TP adjustment
    slippage_pct: float = 0.03             # Simulated slippage


@dataclass
class SimulatedSignal:
    """A signal generated during simulation."""

    symbol: str
    channel: str
    direction: str                         # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float = 0.0
    probability_score: float = 0.0
    confidence: float = 0.0
    regime: str = ""
    timestamp: float = 0.0
    # Outcome tracking
    outcome: str = "PENDING"               # TP1_HIT, TP2_HIT, TP3_HIT, SL_HIT, EXPIRED
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    hold_duration_s: float = 0.0
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0


@dataclass
class SimulationResult:
    """Aggregated results from a simulation run."""

    config: SimulationConfig
    total_signals: int = 0
    tp1_hits: int = 0
    tp2_hits: int = 0
    tp3_hits: int = 0
    sl_hits: int = 0
    expired: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_hold_duration_s: float = 0.0
    avg_latency_ms: float = 0.0
    signals: List[SimulatedSignal] = field(default_factory=list)
    suppressed_count: int = 0
    duration_s: float = 0.0

    def compute_stats(self) -> None:
        """Compute aggregate statistics from individual signals."""
        if not self.signals:
            return
        self.total_signals = len(self.signals)
        self.tp1_hits = sum(1 for s in self.signals if s.outcome == "TP1_HIT")
        self.tp2_hits = sum(1 for s in self.signals if s.outcome == "TP2_HIT")
        self.tp3_hits = sum(1 for s in self.signals if s.outcome == "TP3_HIT")
        self.sl_hits = sum(1 for s in self.signals if s.outcome == "SL_HIT")
        self.expired = sum(1 for s in self.signals if s.outcome == "EXPIRED")
        winners = self.tp1_hits + self.tp2_hits + self.tp3_hits
        decided = winners + self.sl_hits
        self.win_rate = (winners / decided * 100) if decided > 0 else 0.0
        pnls = [s.pnl_pct for s in self.signals if s.pnl_pct != 0.0]
        self.avg_pnl_pct = sum(pnls) / len(pnls) if pnls else 0.0
        durations = [s.hold_duration_s for s in self.signals if s.hold_duration_s > 0]
        self.avg_hold_duration_s = sum(durations) / len(durations) if durations else 0.0


class Simulator:
    """Historical data replay simulator.

    Feeds historical candle data through the signal pipeline (filters,
    probability scoring, dynamic SL/TP) and tracks simulated outcomes.
    """

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self._signals: List[SimulatedSignal] = []
        self._suppressed: int = 0

    def simulate_signal(
        self,
        symbol: str,
        channel: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: float = 0.0,
        probability_score: float = 0.0,
        confidence: float = 0.0,
        regime: str = "",
    ) -> Optional[SimulatedSignal]:
        """Create a simulated signal if it passes the probability threshold.

        Parameters
        ----------
        symbol:
            Trading pair.
        channel:
            Signal channel.
        direction:
            ``"LONG"`` or ``"SHORT"``.
        entry_price:
            Simulated entry price.
        stop_loss:
            Computed stop-loss level.
        tp1, tp2, tp3:
            Take-profit levels.
        probability_score:
            Filter probability score (0-100).
        confidence:
            Composite confidence score (0-100).
        regime:
            Market regime at signal time.

        Returns
        -------
        SimulatedSignal or None
            None if filtered by probability threshold.
        """
        if probability_score < self.config.probability_threshold:
            self._suppressed += 1
            log.debug(
                "Simulation: suppressed {} {} (prob={:.1f} < threshold={:.1f})",
                symbol, channel, probability_score, self.config.probability_threshold,
            )
            return None

        sig = SimulatedSignal(
            symbol=symbol,
            channel=channel,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            probability_score=probability_score,
            confidence=confidence,
            regime=regime,
            timestamp=time.time(),
        )
        self._signals.append(sig)
        return sig

    def evaluate_outcome(
        self,
        signal: SimulatedSignal,
        price_series: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """Evaluate the outcome of a simulated signal against price data.

        Walks forward through the price series to determine whether TP1,
        TP2, TP3, or SL was hit first.

        Parameters
        ----------
        signal:
            The simulated signal to evaluate.
        price_series:
            Forward price data after signal entry.
        timestamps:
            Optional timestamps corresponding to price_series entries.
        """
        if not price_series:
            signal.outcome = "EXPIRED"
            return

        is_long = signal.direction == "LONG"
        entry = signal.entry_price
        best_pnl = 0.0
        worst_pnl = 0.0

        for price in price_series:
            if is_long:
                pnl_pct = (price - entry) / entry * 100
            else:
                pnl_pct = (entry - price) / entry * 100

            best_pnl = max(best_pnl, pnl_pct)
            worst_pnl = min(worst_pnl, pnl_pct)

            # Check SL hit
            if is_long and price <= signal.stop_loss:
                signal.outcome = "SL_HIT"
                signal.exit_price = signal.stop_loss
                signal.pnl_pct = (signal.stop_loss - entry) / entry * 100
                break
            if not is_long and price >= signal.stop_loss:
                signal.outcome = "SL_HIT"
                signal.exit_price = signal.stop_loss
                signal.pnl_pct = (entry - signal.stop_loss) / entry * 100
                break

            # Check TP levels (highest first)
            if signal.tp3 > 0:
                if (is_long and price >= signal.tp3) or (not is_long and price <= signal.tp3):
                    signal.outcome = "TP3_HIT"
                    signal.exit_price = signal.tp3
                    if is_long:
                        signal.pnl_pct = (signal.tp3 - entry) / entry * 100
                    else:
                        signal.pnl_pct = (entry - signal.tp3) / entry * 100
                    break
            if (is_long and price >= signal.tp2) or (not is_long and price <= signal.tp2):
                signal.outcome = "TP2_HIT"
                signal.exit_price = signal.tp2
                if is_long:
                    signal.pnl_pct = (signal.tp2 - entry) / entry * 100
                else:
                    signal.pnl_pct = (entry - signal.tp2) / entry * 100
                break
            if (is_long and price >= signal.tp1) or (not is_long and price <= signal.tp1):
                signal.outcome = "TP1_HIT"
                signal.exit_price = signal.tp1
                if is_long:
                    signal.pnl_pct = (signal.tp1 - entry) / entry * 100
                else:
                    signal.pnl_pct = (entry - signal.tp1) / entry * 100
                break
        else:
            signal.outcome = "EXPIRED"
            signal.exit_price = price_series[-1]
            if is_long:
                signal.pnl_pct = (price_series[-1] - entry) / entry * 100
            else:
                signal.pnl_pct = (entry - price_series[-1]) / entry * 100

        signal.max_favorable_pct = best_pnl
        signal.max_adverse_pct = worst_pnl
        if timestamps and len(timestamps) > 0:
            signal.hold_duration_s = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    def get_result(self) -> SimulationResult:
        """Compile and return the simulation results."""
        result = SimulationResult(
            config=self.config,
            signals=list(self._signals),
            suppressed_count=self._suppressed,
        )
        result.compute_stats()
        return result

    def reset(self) -> None:
        """Reset the simulator state for a new run."""
        self._signals.clear()
        self._suppressed = 0

    @staticmethod
    def export_csv(result: SimulationResult) -> str:
        """Export simulation results to CSV format.

        Returns
        -------
        str
            CSV-formatted string of all simulated signals.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "symbol", "channel", "direction", "entry_price", "stop_loss",
            "tp1", "tp2", "tp3", "probability_score", "confidence",
            "regime", "outcome", "exit_price", "pnl_pct",
            "hold_duration_s", "max_favorable_pct", "max_adverse_pct",
        ])
        for sig in result.signals:
            writer.writerow([
                sig.symbol, sig.channel, sig.direction,
                f"{sig.entry_price:.8f}", f"{sig.stop_loss:.8f}",
                f"{sig.tp1:.8f}", f"{sig.tp2:.8f}", f"{sig.tp3:.8f}",
                f"{sig.probability_score:.2f}", f"{sig.confidence:.2f}",
                sig.regime, sig.outcome, f"{sig.exit_price:.8f}",
                f"{sig.pnl_pct:.4f}", f"{sig.hold_duration_s:.1f}",
                f"{sig.max_favorable_pct:.4f}", f"{sig.max_adverse_pct:.4f}",
            ])
        return output.getvalue()

    @staticmethod
    def export_json(result: SimulationResult) -> str:
        """Export simulation results to JSON format.

        Returns
        -------
        str
            JSON-formatted string with summary and signal details.
        """
        data = {
            "summary": {
                "total_signals": result.total_signals,
                "tp1_hits": result.tp1_hits,
                "tp2_hits": result.tp2_hits,
                "tp3_hits": result.tp3_hits,
                "sl_hits": result.sl_hits,
                "expired": result.expired,
                "win_rate": round(result.win_rate, 2),
                "avg_pnl_pct": round(result.avg_pnl_pct, 4),
                "avg_hold_duration_s": round(result.avg_hold_duration_s, 1),
                "suppressed_count": result.suppressed_count,
                "config": {
                    "replay_days": result.config.replay_days,
                    "probability_threshold": result.config.probability_threshold,
                    "channels": result.config.channels,
                    "sl_multiplier": result.config.sl_multiplier,
                    "tp_multiplier": result.config.tp_multiplier,
                },
            },
            "signals": [
                {
                    "symbol": s.symbol,
                    "channel": s.channel,
                    "direction": s.direction,
                    "entry_price": s.entry_price,
                    "stop_loss": s.stop_loss,
                    "tp1": s.tp1,
                    "tp2": s.tp2,
                    "tp3": s.tp3,
                    "probability_score": s.probability_score,
                    "confidence": s.confidence,
                    "regime": s.regime,
                    "outcome": s.outcome,
                    "pnl_pct": round(s.pnl_pct, 4),
                    "hold_duration_s": round(s.hold_duration_s, 1),
                }
                for s in result.signals
            ],
        }
        return json.dumps(data, indent=2)
