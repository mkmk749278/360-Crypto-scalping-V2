"""Post-Trade ML Feedback Loop.

Tracks completed trade outcomes and derives adaptive confidence adjustments
so that historically underperforming setup/channel combinations are
penalised while consistently winning ones receive a small boost.

This module is **stateful**: a :class:`FeedbackLoop` instance is held on
the :class:`~src.scanner.Scanner` and updated externally (from the trade
monitor) via :meth:`FeedbackLoop.record_outcome`.

Design notes
------------
* Outcomes are stored in a bounded :class:`collections.deque` (default 500).
* Weight adjustments are recomputed after every new outcome recording.
* The public :meth:`FeedbackLoop.get_confidence_adjustment` method is the
  only entry point used by the scanner hot path.
* The module is intentionally dependency-free beyond the standard library so
  it can be imported without any external packages.

Typical usage
-------------
.. code-block:: python

    from src.feedback_loop import FeedbackLoop, TradeOutcome

    loop = FeedbackLoop()

    # … at trade close …
    loop.record_outcome(TradeOutcome(
        symbol="SOLUSDT",
        channel="360_SCALP",
        direction="LONG",
        setup_class="SWEEP_REVERSAL",
        market_state="TRENDING",
        component_scores={"market": 20.0, "setup": 18.0, "execution": 14.0,
                          "risk": 12.0, "context": 8.0},
        confidence=72.5,
        r_multiple=1.8,
        outcome="TP2",
        hold_duration_seconds=240.0,
        timestamp=time.monotonic(),
    ))

    # … at next signal …
    adj = loop.get_confidence_adjustment({"market": 22.0, ...}, "360_SCALP")
    final_confidence = base_confidence + adj  # clamped externally
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict

from src.utils import get_logger

log = get_logger("feedback_loop")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Win outcomes — trades that hit at least TP1.
_WIN_OUTCOMES: frozenset[str] = frozenset({"TP1", "TP2", "TP3"})

#: Win-rate below this → penalise the setup/channel combination.
_PENALTY_WIN_RATE: float = 0.40

#: Win-rate above this → reward the setup/channel combination.
_BOOST_WIN_RATE: float = 0.70

#: Confidence penalty applied when win rate is below :data:`_PENALTY_WIN_RATE`.
_SETUP_PENALTY: float = -5.0

#: Confidence boost applied when win rate exceeds :data:`_BOOST_WIN_RATE`.
_SETUP_BOOST: float = +3.0

#: Execution score below which a penalty is applied when historical lose rate > 60%.
_EXEC_PENALTY_THRESHOLD: float = 14.0
_EXEC_LOSE_RATE_THRESHOLD: float = 0.60
_EXEC_PENALTY: float = -3.0

#: Market score above which a boost is applied when historical win rate > 65%.
_MARKET_BOOST_THRESHOLD: float = 22.0
_MARKET_WIN_RATE_THRESHOLD: float = 0.65
_MARKET_BOOST: float = +2.0

#: Clamp range for the total confidence adjustment returned by
#: :meth:`FeedbackLoop.get_confidence_adjustment`.
_ADJ_MIN: float = -10.0
_ADJ_MAX: float = +10.0

#: Minimum number of outcomes in a group before we trust its statistics.
_MIN_SAMPLE_SIZE: int = 10


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TradeOutcome:
    """Record of a completed trade used for feedback analysis.

    Attributes
    ----------
    symbol:
        Trading pair (e.g. ``"SOLUSDT"``).
    channel:
        Scanner channel name (e.g. ``"360_SCALP"``).
    direction:
        ``"LONG"`` or ``"SHORT"``.
    setup_class:
        The :class:`~src.signal_quality.SetupClass` value string
        (e.g. ``"SWEEP_REVERSAL"``).
    market_state:
        Market phase at signal time (e.g. ``"TRENDING"``).
    component_scores:
        Dict mapping component name → score (market, setup, execution, risk, context).
    confidence:
        Final confidence value at signal dispatch.
    r_multiple:
        Realised R-multiple (negative for losses).
    outcome:
        One of ``"TP1"``, ``"TP2"``, ``"TP3"``, ``"SL"``, ``"EXPIRED"``,
        ``"INVALIDATED"``.
    hold_duration_seconds:
        How long the trade was held.
    timestamp:
        ``time.monotonic()`` value when the outcome was recorded.
    """

    symbol: str
    channel: str
    direction: str
    setup_class: str
    market_state: str
    component_scores: Dict[str, float]
    confidence: float
    r_multiple: float
    outcome: str
    hold_duration_seconds: float
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FeedbackLoop:
    """Adaptive feedback engine that tunes confidence based on past outcomes.

    Parameters
    ----------
    max_history:
        Maximum number of :class:`TradeOutcome` records to retain.  Older
        entries are evicted automatically once the deque is full.
    """

    def __init__(self, max_history: int = 500) -> None:
        self._outcomes: deque[TradeOutcome] = deque(maxlen=max_history)
        # (channel, setup_class) → confidence adjustment
        self._weight_adjustments: Dict[tuple[str, str], float] = {}
        # Aggregated component-level statistics — recomputed in _recompute_weights
        self._exec_penalty_channels: set[str] = set()
        self._market_boost_channels: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(self, outcome: TradeOutcome) -> None:
        """Record a completed trade and recompute weight adjustments.

        Parameters
        ----------
        outcome:
            Completed :class:`TradeOutcome` instance.
        """
        self._outcomes.append(outcome)
        self._recompute_weights()
        log.debug(
            "Feedback: recorded {} {} {} → {}  (R={:.2f}, total history={})",
            outcome.symbol, outcome.channel, outcome.setup_class,
            outcome.outcome, outcome.r_multiple, len(self._outcomes),
        )

    def get_confidence_adjustment(
        self,
        component_scores: Dict[str, float],
        channel: str,
        setup_class: str = "",
    ) -> float:
        """Return a confidence adjustment based on historical patterns.

        Parameters
        ----------
        component_scores:
            Current signal component scores (market, setup, execution, risk, context).
        channel:
            Channel name (e.g. ``"360_SCALP"``).
        setup_class:
            Setup class string.  When empty, only component-level adjustments
            are applied (setup-level lookup is skipped).

        Returns
        -------
        float
            Adjustment in the range ``[-10, +10]``.
        """
        adj = 0.0

        # Setup/channel-level adjustment
        if setup_class:
            adj += self._weight_adjustments.get((channel, setup_class), 0.0)

        # Component-level adjustments (execution quality signal)
        # Default of 999.0 is intentionally high: when execution score is absent,
        # we do not penalise (fail-open semantics for missing data).
        exec_score = component_scores.get("execution", 999.0)
        if exec_score < _EXEC_PENALTY_THRESHOLD and channel in self._exec_penalty_channels:
            adj += _EXEC_PENALTY
            log.debug(
                "Feedback exec penalty for channel {}: execution={:.1f} < {:.1f}",
                channel, exec_score, _EXEC_PENALTY_THRESHOLD,
            )

        # Market score boost
        market_score = component_scores.get("market", 0.0)
        if market_score > _MARKET_BOOST_THRESHOLD and channel in self._market_boost_channels:
            adj += _MARKET_BOOST
            log.debug(
                "Feedback market boost for channel {}: market={:.1f} > {:.1f}",
                channel, market_score, _MARKET_BOOST_THRESHOLD,
            )

        clamped = max(_ADJ_MIN, min(_ADJ_MAX, adj))
        log.debug(
            "Feedback adjustment for {} / {}: raw={:.1f} → clamped={:.1f}",
            channel, setup_class, adj, clamped,
        )
        return clamped

    def get_setup_win_rate(self, setup_class: str, channel: str) -> float:
        """Return the historical win rate for *setup_class* in *channel*.

        Returns ``0.5`` (neutral) when there is insufficient history.
        """
        group = [
            o for o in self._outcomes
            if o.setup_class == setup_class and o.channel == channel
        ]
        if len(group) < _MIN_SAMPLE_SIZE:
            return 0.5
        wins = sum(1 for o in group if o.outcome in _WIN_OUTCOMES)
        return wins / len(group)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recompute_weights(self) -> None:
        """Analyse recent outcomes and update the stored weight adjustments."""
        # Build groups keyed by (channel, setup_class)
        groups: Dict[tuple[str, str], list[TradeOutcome]] = {}
        for o in self._outcomes:
            key = (o.channel, o.setup_class)
            groups.setdefault(key, []).append(o)

        new_adjustments: Dict[tuple[str, str], float] = {}
        for (channel, setup_class), records in groups.items():
            if len(records) < _MIN_SAMPLE_SIZE:
                continue
            wins = sum(1 for r in records if r.outcome in _WIN_OUTCOMES)
            win_rate = wins / len(records)
            if win_rate < _PENALTY_WIN_RATE:
                new_adjustments[(channel, setup_class)] = _SETUP_PENALTY
            elif win_rate > _BOOST_WIN_RATE:
                new_adjustments[(channel, setup_class)] = _SETUP_BOOST

        self._weight_adjustments = new_adjustments

        # Recompute component-level penalty sets
        exec_penalty_channels: set[str] = set()
        market_boost_channels: set[str] = set()

        # Group by channel for component analysis
        channels: Dict[str, list[TradeOutcome]] = {}
        for o in self._outcomes:
            channels.setdefault(o.channel, []).append(o)

        for channel, records in channels.items():
            if len(records) < _MIN_SAMPLE_SIZE:
                continue
            # Execution penalty: low execution score historically loses > 60 %
            low_exec = [r for r in records if r.component_scores.get("execution", 999) < _EXEC_PENALTY_THRESHOLD]
            if low_exec:
                loses = sum(1 for r in low_exec if r.outcome not in _WIN_OUTCOMES)
                if loses / len(low_exec) > _EXEC_LOSE_RATE_THRESHOLD:
                    exec_penalty_channels.add(channel)

            # Market boost: high market score historically wins > 65 %
            high_market = [r for r in records if r.component_scores.get("market", 0) > _MARKET_BOOST_THRESHOLD]
            if high_market:
                wins = sum(1 for r in high_market if r.outcome in _WIN_OUTCOMES)
                if wins / len(high_market) > _MARKET_WIN_RATE_THRESHOLD:
                    market_boost_channels.add(channel)

        self._exec_penalty_channels = exec_penalty_channels
        self._market_boost_channels = market_boost_channels

        log.debug(
            "Feedback weights recomputed: {} group adjustments, "
            "exec-penalty channels={}, market-boost channels={}",
            len(self._weight_adjustments),
            exec_penalty_channels,
            market_boost_channels,
        )
