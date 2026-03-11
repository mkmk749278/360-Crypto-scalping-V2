"""Base channel strategy and signal model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from config import ChannelConfig
from src.smc import Direction
from src.utils import utcnow


@dataclass
class Signal:
    """Represents a single trade signal."""
    channel: str
    symbol: str
    direction: Direction
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float] = None
    trailing_active: bool = True
    trailing_desc: str = ""
    confidence: float = 0.0
    ai_sentiment_label: str = "Neutral"
    ai_sentiment_summary: str = ""
    risk_label: str = "Medium"
    timestamp: datetime = field(default_factory=utcnow)
    # State for monitoring
    signal_id: str = ""
    status: str = "ACTIVE"  # ACTIVE, TP1_HIT, TP2_HIT, TP3_HIT, SL_HIT, CANCELLED
    current_price: float = 0.0
    pnl_pct: float = 0.0

    @property
    def r_multiple(self) -> float:
        risk = abs(self.entry - self.stop_loss)
        if risk == 0:
            return 0.0
        return abs(self.tp1 - self.entry) / risk


class BaseChannel:
    """Abstract base for channel-specific strategy logic."""

    def __init__(self, config: ChannelConfig) -> None:
        self.config = config

    def evaluate(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        ai_insight: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        """Evaluate whether to emit a signal. Override in subclasses."""
        raise NotImplementedError
