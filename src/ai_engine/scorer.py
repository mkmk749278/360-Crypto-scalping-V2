"""AI Confidence Scorer — per-pair dynamic confidence scoring.

Provides per-pair confidence scoring with dynamic thresholds that
adjust based on market conditions, pair volatility, and historical
performance.

This module complements the main ``src.confidence`` scorer by adding
AI-specific adjustments and dynamic threshold management.

Typical usage
-------------
.. code-block:: python

    from src.ai_engine.scorer import AIConfidenceScorer

    scorer = AIConfidenceScorer()
    result = scorer.score_signal(symbol, base_confidence, context)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.utils import get_logger

log = get_logger("ai_scorer")

# Default dynamic threshold bounds.
_DEFAULT_MIN_THRESHOLD = 55.0
_DEFAULT_MAX_THRESHOLD = 85.0

# Regime-based threshold adjustments.
_REGIME_ADJUSTMENTS: Dict[str, float] = {
    "TRENDING": -5.0,     # Easier to profit in trends → lower bar
    "RANGING": +5.0,      # Harder in ranges → higher bar
    "VOLATILE": +10.0,    # Very risky → much higher bar
    "QUIET": 0.0,         # Neutral
}


@dataclass
class AIScoreResult:
    """Result of AI confidence scoring for a single signal.

    Attributes
    ----------
    symbol:
        Trading pair.
    base_confidence:
        Original confidence before AI adjustment.
    ai_adjustment:
        AI-derived adjustment applied to confidence.
    final_confidence:
        Adjusted confidence after AI processing.
    dynamic_threshold:
        Current dynamic threshold for this pair/regime.
    is_high_confidence:
        Whether the signal exceeds the dynamic threshold.
    metadata:
        Additional scoring metadata.
    """

    symbol: str = ""
    base_confidence: float = 0.0
    ai_adjustment: float = 0.0
    final_confidence: float = 0.0
    dynamic_threshold: float = 65.0
    is_high_confidence: bool = False
    metadata: Dict[str, float] = field(default_factory=dict)


class AIConfidenceScorer:
    """Per-pair AI confidence scorer with dynamic thresholds.

    Maintains per-pair scoring history and dynamically adjusts
    confidence thresholds based on recent pair performance and
    market regime.

    Parameters
    ----------
    base_threshold:
        Starting confidence threshold for high-confidence classification.
    history_size:
        Maximum number of historical scores to retain per pair.
    """

    def __init__(
        self,
        base_threshold: float = 65.0,
        history_size: int = 100,
    ) -> None:
        self._base_threshold = base_threshold
        self._history_size = history_size
        # Per-pair scoring history: symbol → deque of (timestamp, confidence)
        self._pair_history: Dict[str, deque] = {}
        # Per-pair dynamic thresholds
        self._pair_thresholds: Dict[str, float] = {}

    def score_signal(
        self,
        symbol: str,
        base_confidence: float,
        regime: str = "",
        volatility_percentile: float = 0.5,
        pair_win_rate: float = 0.5,
    ) -> AIScoreResult:
        """Score a signal with AI-based confidence adjustments.

        Parameters
        ----------
        symbol:
            Trading pair.
        base_confidence:
            Original confidence from the main scorer.
        regime:
            Current market regime (TRENDING, RANGING, VOLATILE, QUIET).
        volatility_percentile:
            Current volatility relative to history (0–1).
        pair_win_rate:
            Historical win rate for this pair (0–1).

        Returns
        -------
        AIScoreResult
            Scored result with dynamic threshold and adjustment.
        """
        # Compute AI adjustment
        ai_adj = self._compute_adjustment(
            symbol, base_confidence, regime, volatility_percentile, pair_win_rate
        )
        final = max(0.0, min(100.0, base_confidence + ai_adj))

        # Compute dynamic threshold
        threshold = self._compute_dynamic_threshold(symbol, regime, volatility_percentile)
        self._pair_thresholds[symbol] = threshold

        # Record in history
        self._record_score(symbol, final)

        is_high = final >= threshold

        result = AIScoreResult(
            symbol=symbol,
            base_confidence=base_confidence,
            ai_adjustment=ai_adj,
            final_confidence=final,
            dynamic_threshold=threshold,
            is_high_confidence=is_high,
            metadata={
                "regime_adjustment": _REGIME_ADJUSTMENTS.get(regime, 0.0),
                "volatility_percentile": volatility_percentile,
                "pair_win_rate": pair_win_rate,
            },
        )

        log.debug(
            "AI score {}: base={:.1f} adj={:+.1f} final={:.1f} threshold={:.1f} high={}",
            symbol, base_confidence, ai_adj, final, threshold, is_high,
        )
        return result

    def get_pair_threshold(self, symbol: str) -> float:
        """Return the current dynamic threshold for a pair."""
        return self._pair_thresholds.get(symbol, self._base_threshold)

    def get_pair_avg_confidence(self, symbol: str) -> float:
        """Return the average recent confidence for a pair."""
        history = self._pair_history.get(symbol)
        if not history:
            return 0.0
        scores = [score for _, score in history]
        return sum(scores) / len(scores) if scores else 0.0

    def _compute_adjustment(
        self,
        symbol: str,
        base_confidence: float,
        regime: str,
        volatility_percentile: float,
        pair_win_rate: float,
    ) -> float:
        """Compute the AI confidence adjustment.

        Components:
        1. Win-rate adjustment: boost pairs with high historical win rate
        2. Volatility penalty: penalise signals during extreme volatility
        3. Consistency bonus: reward pairs with consistently high confidence
        """
        adj = 0.0

        # Win-rate adjustment (-3 to +3)
        if pair_win_rate > 0.65:
            adj += min((pair_win_rate - 0.65) * 20.0, 3.0)
        elif pair_win_rate < 0.35:
            adj -= min((0.35 - pair_win_rate) * 20.0, 3.0)

        # Volatility penalty: extreme volatility reduces confidence
        if volatility_percentile > 0.9:
            adj -= (volatility_percentile - 0.9) * 30.0  # up to -3.0 at 100th pctile

        # Consistency bonus: if recent scores are consistently high, small boost
        avg = self.get_pair_avg_confidence(symbol)
        if avg > 70.0:
            adj += 1.0

        return max(-5.0, min(5.0, adj))

    def _compute_dynamic_threshold(
        self,
        symbol: str,
        regime: str,
        volatility_percentile: float,
    ) -> float:
        """Compute the dynamic confidence threshold for a pair/regime.

        The threshold is adjusted based on:
        1. Market regime (trends lower threshold, ranges higher)
        2. Volatility (extreme volatility raises threshold)
        """
        threshold = self._base_threshold

        # Regime adjustment
        threshold += _REGIME_ADJUSTMENTS.get(regime, 0.0)

        # Volatility adjustment
        if volatility_percentile > 0.8:
            threshold += (volatility_percentile - 0.8) * 25.0  # up to +5

        return max(_DEFAULT_MIN_THRESHOLD, min(_DEFAULT_MAX_THRESHOLD, threshold))

    def _record_score(self, symbol: str, confidence: float) -> None:
        """Record a confidence score in the pair history."""
        if symbol not in self._pair_history:
            self._pair_history[symbol] = deque(maxlen=self._history_size)
        self._pair_history[symbol].append((time.monotonic(), confidence))
