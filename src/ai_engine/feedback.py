"""AI Feedback Adapter — thin integration layer with the feedback loop.

Provides a bridge between the AI engine's prediction pipeline and the
main :class:`~src.feedback_loop.FeedbackLoop` so that AI prediction
outcomes can be tracked and used to improve future predictions.

Typical usage
-------------
.. code-block:: python

    from src.ai_engine.feedback import AIFeedbackAdapter

    adapter = AIFeedbackAdapter(feedback_loop)
    adapter.record_prediction_outcome(symbol, predicted_dir, actual_dir, confidence)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils import get_logger

log = get_logger("ai_feedback")


@dataclass
class PredictionRecord:
    """Record of an AI prediction and its outcome.

    Attributes
    ----------
    symbol:
        Trading pair.
    predicted_direction:
        Direction predicted by the AI engine.
    actual_direction:
        Actual market direction observed.
    confidence:
        Confidence score at prediction time.
    was_correct:
        Whether the prediction was correct.
    timestamp:
        When the prediction was made.
    """

    symbol: str
    predicted_direction: str
    actual_direction: str
    confidence: float
    was_correct: bool
    timestamp: float = field(default_factory=time.monotonic)


class AIFeedbackAdapter:
    """Tracks AI prediction accuracy and feeds results back to the feedback loop.

    Parameters
    ----------
    feedback_loop:
        The main :class:`~src.feedback_loop.FeedbackLoop` instance.
        May be ``None`` when running without the feedback system.
    max_records:
        Maximum prediction records to retain.
    """

    def __init__(
        self,
        feedback_loop: Optional[object] = None,
        max_records: int = 1000,
    ) -> None:
        self._feedback_loop = feedback_loop
        self._records: deque[PredictionRecord] = deque(maxlen=max_records)
        self._correct_count = 0
        self._total_count = 0

    def record_prediction_outcome(
        self,
        symbol: str,
        predicted_direction: str,
        actual_direction: str,
        confidence: float,
    ) -> PredictionRecord:
        """Record the outcome of an AI prediction.

        Parameters
        ----------
        symbol:
            Trading pair.
        predicted_direction:
            What the AI predicted (LONG/SHORT/NEUTRAL).
        actual_direction:
            What actually happened.
        confidence:
            Confidence at prediction time.

        Returns
        -------
        PredictionRecord
            The recorded prediction.
        """
        was_correct = predicted_direction == actual_direction
        record = PredictionRecord(
            symbol=symbol,
            predicted_direction=predicted_direction,
            actual_direction=actual_direction,
            confidence=confidence,
            was_correct=was_correct,
        )
        self._records.append(record)
        self._total_count += 1
        if was_correct:
            self._correct_count += 1

        log.debug(
            "AI prediction {}: predicted={} actual={} correct={} (accuracy={:.1%})",
            symbol, predicted_direction, actual_direction, was_correct,
            self.accuracy,
        )
        return record

    @property
    def accuracy(self) -> float:
        """Overall prediction accuracy (0–1)."""
        if self._total_count == 0:
            return 0.0
        return self._correct_count / self._total_count

    @property
    def total_predictions(self) -> int:
        """Total number of predictions recorded."""
        return self._total_count

    def get_pair_accuracy(self, symbol: str) -> float:
        """Return prediction accuracy for a specific pair.

        Returns 0.5 (neutral) when there are fewer than 5 records.
        """
        pair_records = [r for r in self._records if r.symbol == symbol]
        if len(pair_records) < 5:
            return 0.5
        correct = sum(1 for r in pair_records if r.was_correct)
        return correct / len(pair_records)

    def get_accuracy_by_confidence_tier(self) -> Dict[str, float]:
        """Return accuracy broken down by confidence tier.

        Tiers:
        - low: confidence < 60
        - medium: 60 <= confidence < 75
        - high: confidence >= 75
        """
        tiers: Dict[str, List[bool]] = {"low": [], "medium": [], "high": []}
        for r in self._records:
            if r.confidence < 60:
                tiers["low"].append(r.was_correct)
            elif r.confidence < 75:
                tiers["medium"].append(r.was_correct)
            else:
                tiers["high"].append(r.was_correct)

        result = {}
        for tier, outcomes in tiers.items():
            if outcomes:
                result[tier] = sum(outcomes) / len(outcomes)
            else:
                result[tier] = 0.0
        return result

    def get_recent_records(self, n: int = 50) -> List[PredictionRecord]:
        """Return the most recent *n* prediction records."""
        records = list(self._records)
        return records[-n:] if len(records) > n else records
