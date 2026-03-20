"""Base channel strategy and signal model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    risk_label: str = ""
    market_phase: str = "N/A"
    liquidity_info: str = "Standard"
    setup_class: str = "UNCLASSIFIED"
    quality_tier: str = "B"
    entry_zone: str = ""
    invalidation_summary: str = ""
    analyst_reason: str = ""
    execution_note: str = ""
    component_scores: Dict[str, float] = field(default_factory=dict)
    pair_quality_score: float = 0.0
    pair_quality_label: str = "UNRATED"
    pre_ai_confidence: float = 0.0
    post_ai_confidence: float = 0.0
    timestamp: datetime = field(default_factory=utcnow)
    # State for monitoring
    signal_id: str = ""
    status: str = "ACTIVE"  # ACTIVE, TP1_HIT, TP2_HIT, SL_HIT, BREAKEVEN_EXIT, PROFIT_LOCKED, FULL_TP_HIT, CANCELLED
    current_price: float = 0.0
    pnl_pct: float = 0.0
    max_favorable_excursion_pct: float = 0.0
    max_adverse_excursion_pct: float = 0.0
    # Original SL distance at signal creation (used by trailing stop logic so that
    # the trailing buffer doesn't collapse to zero after TP2 moves SL to break-even)
    original_sl_distance: float = 0.0
    # Scanner-enriched market context (set before enqueuing)
    spread_pct: float = 0.0
    volume_24h_usd: float = 0.0
    # Level-2 order book snapshot attached by the scanner for OBI filtering.
    # Format: {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
    order_book: Optional[Dict[str, List[Any]]] = None
    # Best TP level reached during this signal's lifetime (0 = none, 1 = TP1, 2 = TP2)
    best_tp_hit: int = 0
    # PnL % frozen at the moment the highest TP was hit (used for signal quality stats)
    best_tp_pnl_pct: float = 0.0

    # ---- DCA (Double Entry) fields ----
    entry_2: Optional[float] = None           # 2nd entry price
    entry_2_filled: bool = False              # Whether 2nd entry was taken
    avg_entry: float = 0.0                    # Weighted average entry
    position_weight_1: float = 0.6            # Weight of Entry 1 (default 60%)
    position_weight_2: float = 0.4            # Weight of Entry 2 (default 40%)
    dca_zone_lower: float = 0.0               # Lower bound of DCA zone
    dca_zone_upper: float = 0.0               # Upper bound of DCA zone
    dca_timestamp: Optional[datetime] = None  # When DCA Entry 2 was filled

    # ---- Original TP/Entry values (before DCA recalc) ----
    original_entry: float = 0.0               # Entry 1 price before averaging
    original_tp1: float = 0.0
    original_tp2: float = 0.0
    original_tp3: Optional[float] = None

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
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        """Evaluate whether to emit a signal. Override in subclasses."""
        raise NotImplementedError
