"""Base channel strategy and signal model."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import DYNAMIC_SL_TP_ENABLED, ChannelConfig
from src.channels.signal_params import lookup_signal_params
from src.dca import compute_dca_zone
from src.filters import check_spread, check_volume
from src.smc import Direction
from src.utils import utcnow

# ---------------------------------------------------------------------------
# Volatility-adaptive TP ratio constants
# ---------------------------------------------------------------------------
# When BB width exceeds this threshold, the environment is considered high-vol
# and TP targets are stretched to capture larger moves.
_HIGH_VOL_BB_WIDTH: float = 5.0  # percent

# When BB width is below this threshold, the environment is low-vol and
# compressed TP targets prevent capital sitting idle in unreached positions.
_LOW_VOL_BB_WIDTH: float = 1.5  # percent

# Multipliers applied to each TP ratio when volatility regime is detected.
_VOL_STRETCH_FACTOR: float = 1.3   # High-vol: stretch TP targets
_VOL_COMPRESS_FACTOR: float = 0.7  # Low-vol: compress TP targets

# Fallback TP3 ratio used when adj_ratios has fewer than 3 elements.
_DEFAULT_TP3_RATIO: float = 2.0


def _default_trailing_desc(trailing_atr_mult: float) -> str:
    """Return a standardised trailing stop description string."""
    return (
        f"Stage 1: {trailing_atr_mult}×ATR | "
        f"Post-TP1: 1×ATR (BE) | Post-TP2: 0.5×ATR (tight)"
    )


@dataclass
class TrailingStopState:
    """Encapsulates the dynamic trailing stop configuration for a live signal.

    Used by the trade monitor to compute the current trailing stop level.
    """
    initial_atr: float                # ATR at signal creation
    current_atr: float = 0.0         # ATR from most recent lifecycle check
    stage: int = 0                   # 0=entry, 1=TP1 hit, 2=TP2 hit
    breakeven_set: bool = False      # Whether SL has been moved to breakeven
    tight_trail_active: bool = False # Whether the tight 0.5× ATR trail is active

    @property
    def effective_mult(self) -> float:
        """ATR multiple for the current stage."""
        if self.stage == 2:
            return 0.5     # Post-TP2: tight trail
        if self.stage == 1:
            return 1.0     # Post-TP1: intermediate trail
        return 2.0         # Entry: standard trail

    @property
    def trail_distance(self) -> float:
        """Absolute trailing distance using current ATR."""
        atr = self.current_atr if self.current_atr > 0 else self.initial_atr
        return atr * self.effective_mult


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
    trailing_atr_mult_effective: float = 0.0   # Current trailing ATR multiple (updates during trade)
    trailing_stage: int = 0                     # 0=initial, 1=TP1_hit (breakeven), 2=TP2_hit (tight trail)
    partial_close_pct: float = 0.0             # Fraction of position notionally closed
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

    # ---- Soft-penalty gate tracking ----
    soft_penalty_total: float = 0.0           # Accumulated soft-gate confidence deduction
    regime_penalty_multiplier: float = 1.0    # Regime multiplier applied to base penalties
    soft_gate_flags: str = ""                 # Comma-separated list of soft gates that fired

    # ---- Signal tier (set by scanner after confidence scoring) ----
    signal_tier: str = "B"  # "A+" (80-100), "B" (65-79), "WATCHLIST" (50-64), "FILTERED" (<50)

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

    # ---- Entry zone for limit-order execution ----
    # Users should place limit orders within this zone rather than chasing
    # the exact entry price.  Populated by each channel's evaluate() method.
    entry_zone_low: Optional[float] = None    # Lower bound of limit order zone
    entry_zone_high: Optional[float] = None   # Upper bound of limit order zone
    # How long (minutes) the setup remains actionable.  After this window
    # the signal should no longer be entered even if price is still in zone.
    valid_for_minutes: int = 15
    # Tells the user what order type to use (e.g. "LIMIT_ZONE", "MARKET")
    execution_type: str = "LIMIT_ZONE"

    # ---- Delivery retry tracking (router-internal, not shown to users) ----
    _delivery_retries: int = 0

    # ---- Consecutive momentum invalidation counter ----
    # Tracks how many consecutive poll cycles momentum has been below the
    # invalidation threshold.  Resets to 0 when momentum recovers above it.
    # Used by _check_invalidation() to require multiple consecutive readings
    # before declaring momentum exhaustion (reduces false kills from 1m pauses).
    momentum_invalidation_count: int = 0

    # ---- Signal Lifecycle Monitor state ----
    # Populated after the signal is posted to Telegram so the lifecycle
    # monitor has a baseline for regime/momentum comparisons.
    entry_regime: str = ""                        # market regime when signal was opened
    entry_momentum_slope: float = 0.0             # EMA slope at entry (% diff)
    last_lifecycle_check: Optional[datetime] = None  # UTC timestamp of last check
    lifecycle_alert_level: str = "GREEN"          # GREEN, YELLOW, RED

    # ---- MTF confluence score (0-1, populated by scanner) ----
    mtf_score: float = 0.0

    # ---- Chart pattern names that confirmed the signal direction ----
    chart_pattern_names: str = ""

    # ---- Serialised regime context (ATR%ile, volume profile, ADX slope) ----
    regime_context: str = ""   # Serialised regime context for logging

    # ---- Latency tracking ----
    # detected_at: time.time() when channel.evaluate() first returned a non-None signal.
    # posted_at: time.time() when the signal was successfully delivered to Telegram.
    # enrichment_latency_ms: difference (ms) between detection and posting.
    detected_at: Optional[float] = None
    posted_at: Optional[float] = None
    enrichment_latency_ms: Optional[float] = None

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

    def _pass_basic_filters(self, spread_pct: float, volume_24h_usd: float) -> bool:
        """Return True if basic spread/volume filters pass."""
        return (
            check_spread(spread_pct, self.config.spread_max)
            and check_volume(volume_24h_usd, self.config.min_volume)
        )

    def evaluate(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
        regime: str = "",  # MarketRegime value string, e.g. "TRENDING_UP"
    ) -> Optional[Signal]:
        """Evaluate whether to emit a signal. Override in subclasses."""
        raise NotImplementedError


def compute_dynamic_sl_tp_ratios(
    base_tp_ratios: list,
    base_sl_mult: float,
    atr_percentile: float,
    regime: str,
    pair_tier: str = "MIDCAP",
) -> tuple:
    """Return (sl_multiplier, tp_ratios) adjusted for volatility, regime, and pair tier.

    Parameters
    ----------
    base_tp_ratios:
        Default TP ratios from channel config (e.g. [0.5, 1.0, 1.5]).
    base_sl_mult:
        Default SL multiplier (1.0 = no scaling).
    atr_percentile:
        Rolling ATR percentile 0–100 (from RegimeContext).
    regime:
        Current market regime string.
    pair_tier:
        "MAJOR", "MIDCAP", or "ALTCOIN".

    Returns
    -------
    (sl_multiplier, tp_ratios)
        sl_multiplier: float to multiply the ATR-based SL distance.
        tp_ratios: list of adjusted TP ratios.
    """
    # --- Volatility-percentile SL adjustment ---
    if atr_percentile >= 80:
        vol_sl_adj = 1.3    # Widen SL in high-vol environment
        vol_tp_adj = 1.25   # Wider TP targets too
    elif atr_percentile <= 20:
        vol_sl_adj = 0.8    # Tighter SL in low-vol
        vol_tp_adj = 0.75
    else:
        vol_sl_adj = 1.0
        vol_tp_adj = 1.0

    # --- Regime SL/TP adjustments ---
    regime_upper = regime.upper() if regime else ""
    regime_sl = {
        "TRENDING_UP": 1.0, "TRENDING_DOWN": 1.0,
        "RANGING": 0.9,      "QUIET": 0.85,
        "VOLATILE": 1.4,
    }.get(regime_upper, 1.0)

    # TP scaling: in trending regimes, boost TP3 (the runner target) by 20%
    regime_tp = [1.0] * len(base_tp_ratios)
    if regime_upper in ("TRENDING_UP", "TRENDING_DOWN"):
        if len(regime_tp) >= 3:
            regime_tp[-1] = 1.2   # Boost only the runner TP
    elif regime_upper in ("RANGING", "QUIET"):
        regime_tp = [0.9] * len(base_tp_ratios)  # Compress all TPs
    elif regime_upper == "VOLATILE":
        regime_tp = [1.1] * len(base_tp_ratios)

    # --- Pair-tier SL widening ---
    tier_sl = {"MAJOR": 0.95, "MIDCAP": 1.0, "ALTCOIN": 1.20}.get(pair_tier, 1.0)

    # Combine all adjustments
    final_sl_mult = base_sl_mult * vol_sl_adj * regime_sl * tier_sl
    final_tp = [
        r * vol_tp_adj * regime_tp[i]
        for i, r in enumerate(base_tp_ratios)
    ]
    return final_sl_mult, final_tp


def build_channel_signal(
    config: ChannelConfig,
    symbol: str,
    direction: Direction,
    close: float,
    sl: float,
    tp1: float,
    tp2: float,
    tp3: float,
    sl_dist: float,
    id_prefix: str,
    atr_val: float = 0.0,
    vwap_price: float = 0.0,
    setup_class: str = "",
    bb_width_pct: Optional[float] = None,
    regime: str = "",
    atr_percentile: float = 50.0,
    pair_tier: str = "MIDCAP",
) -> Optional[Signal]:
    """Shared signal construction for all scalp-family channels.

    Centralises Signal instantiation, DCA zone calculation, and direction-
    biased entry zone logic so that bug fixes propagate automatically to every
    channel that calls this helper.

    Parameters
    ----------
    bb_width_pct:
        Optional Bollinger Band width as a percentage of mid price.  Used only
        when ``DYNAMIC_SL_TP_ENABLED`` is ``False`` (legacy path) to stretch or
        compress TP ratios based on volatility regime.
    setup_class:
        Setup class string (e.g. "RANGE_FADE", "WHALE_MOMENTUM").  Used to
        set ``sig.setup_class`` and, together with ``regime``, to look up
        regime-aware signal parameters.
    regime:
        Market regime string (e.g. "TRENDING_UP", "RANGING", "VOLATILE").
        When provided alongside ``setup_class``, enables per-context signal
        parameter overrides via :func:`lookup_signal_params`.
    atr_percentile:
        Rolling ATR percentile 0–100 (from RegimeContext).  Used by the
        dynamic SL/TP computation path when ``DYNAMIC_SL_TP_ENABLED`` is
        ``True``.
    pair_tier:
        Pair classification tier: "MAJOR", "MIDCAP", or "ALTCOIN".  ALTCOIN
        pairs receive a wider SL multiplier to account for manipulation wicks.
    """
    if direction == Direction.LONG and sl >= close:
        return None
    if direction == Direction.SHORT and sl <= close:
        return None

    # Look up regime-aware parameters.  When setup_class or regime are empty
    # the lookup returns _DEFAULT which replicates the previous behaviour.
    params = lookup_signal_params(config.name, setup_class, regime)

    # Determine base TP ratios: use per-context override when available,
    # otherwise fall back to channel config.
    base_ratios = list(params.tp_ratios) if params.tp_ratios is not None else list(config.tp_ratios)

    if DYNAMIC_SL_TP_ENABLED:
        # Dynamic path: ATR-percentile + regime + pair-tier drive SL/TP adjustment.
        final_sl_mult, adj_ratios = compute_dynamic_sl_tp_ratios(
            base_ratios, params.sl_multiplier, atr_percentile, regime, pair_tier
        )
        sl_dist = sl_dist * final_sl_mult
    else:
        # Legacy path: apply static multiplier and bb_width adjustment.
        sl_dist = sl_dist * params.sl_multiplier
        if bb_width_pct is not None:
            if bb_width_pct > _HIGH_VOL_BB_WIDTH:
                adj_ratios = [r * params.vol_stretch_factor for r in base_ratios]
            elif bb_width_pct < _LOW_VOL_BB_WIDTH:
                adj_ratios = [r * params.vol_compress_factor for r in base_ratios]
            else:
                adj_ratios = list(base_ratios)
        else:
            adj_ratios = list(base_ratios)

    if direction == Direction.LONG:
        sl = close - sl_dist
    else:
        sl = close + sl_dist

    # Re-validate SL after applying multiplier.
    if direction == Direction.LONG and sl >= close:
        return None
    if direction == Direction.SHORT and sl <= close:
        return None

    # Compute TP levels from adj_ratios.
    # tp1/tp2/tp3 arguments are deprecated; TP is always computed from sl_dist here.
    if direction == Direction.LONG:
        tp1 = close + sl_dist * adj_ratios[0]
        tp2 = close + sl_dist * adj_ratios[1]
        tp3 = close + sl_dist * adj_ratios[2] if len(adj_ratios) > 2 else close + sl_dist * _DEFAULT_TP3_RATIO
    else:
        tp1 = close - sl_dist * adj_ratios[0]
        tp2 = close - sl_dist * adj_ratios[1]
        tp3 = close - sl_dist * adj_ratios[2] if len(adj_ratios) > 2 else close - sl_dist * _DEFAULT_TP3_RATIO

    sig = Signal(
        channel=config.name,
        symbol=symbol,
        direction=direction,
        entry=close,
        stop_loss=round(sl, 8),
        tp1=round(tp1, 8),
        tp2=round(tp2, 8),
        tp3=round(tp3, 8),
        trailing_active=True,
        trailing_desc=_default_trailing_desc(config.trailing_atr_mult),
        confidence=0.0,
        ai_sentiment_label="",
        ai_sentiment_summary="",
        risk_label="Aggressive",
        timestamp=utcnow(),
        signal_id=f"{id_prefix}-{uuid.uuid4().hex[:8].upper()}",
        current_price=close,
        original_sl_distance=sl_dist,
    )

    if params.dca_enabled:
        dca_lower, dca_upper = compute_dca_zone(
            close, round(sl, 8), direction, config.dca_zone_range
        )
        sig.dca_zone_lower = dca_lower
        sig.dca_zone_upper = dca_upper
    sig.original_entry = close
    sig.original_tp1 = round(tp1, 8)
    sig.original_tp2 = round(tp2, 8)
    sig.original_tp3 = round(tp3, 8)
    if setup_class:
        sig.setup_class = setup_class

    # Override validity window when the params table specifies one.
    if params.validity_minutes is not None:
        sig.valid_for_minutes = params.validity_minutes

    # Direction-biased entry zone: LONGs bias below close (buy on dips),
    # SHORTs bias above close (sell on rallies).
    zone_width = (atr_val * 0.4) if atr_val > 0 else (sl_dist * 0.6)

    # Volume-weighted anchoring: blend zone centre toward VWAP when it is
    # close to the current price and available.
    if vwap_price > 0 and abs(vwap_price - close) < zone_width:
        zone_center = close * 0.6 + vwap_price * 0.4
    else:
        zone_center = close

    bias = params.entry_zone_bias
    if direction == Direction.LONG:
        sig.entry_zone_low = round(zone_center - zone_width * bias, 8)
        sig.entry_zone_high = round(zone_center + zone_width * (1.0 - bias), 8)
    else:
        sig.entry_zone_low = round(zone_center - zone_width * (1.0 - bias), 8)
        sig.entry_zone_high = round(zone_center + zone_width * bias, 8)

    return sig
