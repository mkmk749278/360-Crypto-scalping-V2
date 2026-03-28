"""AI Signal Predictor — async real-time inference pipeline.

Provides a multi-factor feature extraction and signal prediction layer
that combines price, volume, order-book, and correlation data into a
probabilistic signal prediction.

Typical usage
-------------
.. code-block:: python

    from src.ai_engine.predictor import SignalPredictor

    predictor = SignalPredictor()
    prediction = await predictor.predict("BTCUSDT", features)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_logger

log = get_logger("ai_predictor")


@dataclass
class PredictionFeatures:
    """Multi-factor feature set for signal prediction.

    Attributes
    ----------
    price_features:
        Price-derived features (EMA alignment, momentum, etc.).
    volume_features:
        Volume-derived features (OBV, VWAP deviation, volume spikes).
    order_book_features:
        Order-book features (bid/ask imbalance, depth, spread).
    correlation_features:
        Cross-asset correlation features (BTC correlation, sector beta).
    """

    price_features: Dict[str, float] = field(default_factory=dict)
    volume_features: Dict[str, float] = field(default_factory=dict)
    order_book_features: Dict[str, float] = field(default_factory=dict)
    correlation_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class SignalPrediction:
    """Result of a signal prediction.

    Attributes
    ----------
    symbol:
        Trading pair (e.g. ``"BTCUSDT"``).
    direction:
        Predicted direction: ``"LONG"``, ``"SHORT"``, or ``"NEUTRAL"``.
    probability:
        Probability of a successful trade in the predicted direction (0–1).
    features_used:
        List of feature names that contributed to the prediction.
    timestamp:
        ``time.monotonic()`` when prediction was made.
    """

    symbol: str = ""
    direction: str = "NEUTRAL"
    probability: float = 0.5
    features_used: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


class SignalPredictor:
    """Async multi-factor signal predictor.

    Combines multiple data sources into a unified prediction score.
    The predictor uses a weighted feature combination approach that
    can be replaced with ML models when sufficient training data is
    available.

    Parameters
    ----------
    min_probability:
        Minimum probability threshold for a non-neutral prediction.
    feature_weights:
        Optional custom weights for each feature category.
    """

    def __init__(
        self,
        min_probability: float = 0.55,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._min_probability = min_probability
        self._weights = feature_weights or {
            "price": 0.35,
            "volume": 0.25,
            "order_book": 0.25,
            "correlation": 0.15,
        }
        self._prediction_count = 0
        # Optional allowlist: when non-empty, predict() returns NEUTRAL for
        # any symbol not in this set (PR3 — top-50 futures-only AI inference).
        self._allowed_pairs: Optional[set] = None

    def set_allowed_pairs(self, pairs: List[str]) -> None:
        """Restrict AI inference to the specified symbol list.

        When set, :meth:`predict` returns a neutral result for any symbol
        that is not in *pairs*, keeping inference costs proportional to the
        top-50 futures universe rather than the full pair list.

        Pass an empty list (or call with ``pairs=[]``) to clear the filter
        and restore inference for all symbols.

        Parameters
        ----------
        pairs:
            Symbols allowed for full inference (case-insensitive normalised
            to upper-case internally).  An empty list clears the filter.
        """
        if pairs:
            self._allowed_pairs = {s.upper() for s in pairs}
            log.debug("AI predictor: allowed pairs set to %d symbols", len(self._allowed_pairs))
        else:
            self._allowed_pairs = None
            log.debug("AI predictor: allowed pairs filter cleared")

    async def predict(
        self,
        symbol: str,
        features: PredictionFeatures,
    ) -> SignalPrediction:
        """Generate a signal prediction from multi-factor features.

        Parameters
        ----------
        symbol:
            Trading pair.
        features:
            Multi-factor feature set.

        Returns
        -------
        SignalPrediction
            Prediction with direction and probability.  Returns a neutral
            prediction when *symbol* is not in the configured allowed-pairs
            list (see :meth:`set_allowed_pairs`).
        """
        # Top-50 futures-only gate (PR3): skip non-allowed symbols to keep
        # AI inference focused on the active high-quality universe.
        if self._allowed_pairs is not None and symbol.upper() not in self._allowed_pairs:
            log.debug("AI predictor: skipping non-allowed symbol %s", symbol)
            return SignalPrediction(symbol=symbol, direction="NEUTRAL", probability=0.5)

        scores = self._extract_scores(features)
        direction, probability = self._combine_scores(scores)

        self._prediction_count += 1
        used = [k for k, v in scores.items() if v != 0.0]

        prediction = SignalPrediction(
            symbol=symbol,
            direction=direction,
            probability=probability,
            features_used=used,
        )

        log.debug(
            "Prediction for {}: {} (p={:.3f}, features={})",
            symbol, direction, probability, used,
        )
        return prediction

    async def predict_batch(
        self,
        symbols_features: Dict[str, PredictionFeatures],
    ) -> Dict[str, SignalPrediction]:
        """Run predictions for multiple symbols concurrently.

        Parameters
        ----------
        symbols_features:
            Mapping of symbol → features.

        Returns
        -------
        Dict mapping symbol → prediction.  Non-allowed symbols (when an
        allowed-pairs filter is active) receive a neutral prediction and are
        not passed to the inference pipeline.
        """
        # Filter to allowed pairs before spawning tasks (PR3).
        if self._allowed_pairs is not None:
            filtered = {
                sym: feat for sym, feat in symbols_features.items()
                if sym.upper() in self._allowed_pairs
            }
            skipped = len(symbols_features) - len(filtered)
            if skipped:
                log.debug(
                    "AI predictor batch: skipping %d non-allowed symbols", skipped
                )
            symbols_features = filtered

        tasks = {
            sym: self.predict(sym, feat)
            for sym, feat in symbols_features.items()
        }
        results: Dict[str, SignalPrediction] = {}
        for sym, coro in tasks.items():
            results[sym] = await coro
        return results

    @property
    def prediction_count(self) -> int:
        """Total number of predictions made."""
        return self._prediction_count

    def _extract_scores(self, features: PredictionFeatures) -> Dict[str, float]:
        """Extract directional scores from each feature category.

        Returns a dict with keys matching ``self._weights`` and values in
        ``[-1, +1]`` where positive = bullish and negative = bearish.
        """
        scores: Dict[str, float] = {}

        # Price features: EMA alignment, momentum
        pf = features.price_features
        ema_signal = pf.get("ema_alignment", 0.0)
        momentum = pf.get("momentum", 0.0)
        adx = pf.get("adx", 0.0)
        trend_strength = min(adx / 40.0, 1.0) if adx > 0 else 0.5
        scores["price"] = max(-1.0, min(1.0, (ema_signal + momentum) / 2.0 * trend_strength))

        # Volume features: OBV trend, volume spikes
        vf = features.volume_features
        obv_trend = vf.get("obv_trend", 0.0)
        vol_spike = vf.get("volume_spike", 0.0)
        scores["volume"] = max(-1.0, min(1.0, obv_trend * (1.0 + vol_spike * 0.5)))

        # Order book features: bid/ask imbalance
        obf = features.order_book_features
        imbalance = obf.get("bid_ask_imbalance", 0.0)
        depth_ratio = obf.get("depth_ratio", 1.0)
        scores["order_book"] = max(-1.0, min(1.0, imbalance * min(depth_ratio, 2.0)))

        # Correlation features: BTC correlation, sector direction
        cf = features.correlation_features
        btc_corr = cf.get("btc_correlation", 0.0)
        sector = cf.get("sector_direction", 0.0)
        scores["correlation"] = max(-1.0, min(1.0, (btc_corr + sector) / 2.0))

        return scores

    def _combine_scores(self, scores: Dict[str, float]) -> tuple[str, float]:
        """Combine weighted feature scores into a direction and probability.

        Returns
        -------
        (direction, probability)
            Direction is ``"LONG"``, ``"SHORT"``, or ``"NEUTRAL"``.
            Probability is in ``[0, 1]``.
        """
        weighted_sum = 0.0
        total_weight = 0.0
        for category, score in scores.items():
            w = self._weights.get(category, 0.0)
            weighted_sum += score * w
            total_weight += w

        if total_weight == 0:
            return "NEUTRAL", 0.5

        combined = weighted_sum / total_weight  # in [-1, +1]
        probability = (combined + 1.0) / 2.0  # map to [0, 1]

        if probability >= self._min_probability:
            return "LONG", probability
        elif probability <= (1.0 - self._min_probability):
            return "SHORT", 1.0 - probability
        else:
            return "NEUTRAL", 0.5
