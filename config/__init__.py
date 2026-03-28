"""360-Crypto-Eye-Scalping – configuration module.

All tunables live here so every other module simply does
``from config.settings import cfg`` and reads what it needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Binance endpoints
# ---------------------------------------------------------------------------
BINANCE_REST_BASE: str = os.getenv("BINANCE_REST_BASE", "https://api.binance.com")
BINANCE_WS_BASE: str = os.getenv("BINANCE_WS_BASE", "wss://stream.binance.com:9443/ws")
BINANCE_FUTURES_REST_BASE: str = os.getenv("BINANCE_FUTURES_REST_BASE", "https://fapi.binance.com")
BINANCE_FUTURES_WS_BASE: str = os.getenv("BINANCE_FUTURES_WS_BASE", "wss://fstream.binance.com/ws")

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_SCALP_CHANNEL_ID: str = os.getenv("TELEGRAM_SCALP_CHANNEL_ID", "")
TELEGRAM_SWING_CHANNEL_ID: str = os.getenv("TELEGRAM_SWING_CHANNEL_ID", "")
# REQUIRED: These MUST be set in .env for SPOT and GEM signals to be published.
# If left empty, signals are silently dropped by the signal router.
TELEGRAM_SPOT_CHANNEL_ID: str = os.getenv("TELEGRAM_SPOT_CHANNEL_ID", "")
TELEGRAM_FREE_CHANNEL_ID: str = os.getenv("TELEGRAM_FREE_CHANNEL_ID", "")
TELEGRAM_ADMIN_CHAT_ID: str = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
# REQUIRED: These MUST be set in .env for SPOT and GEM signals to be published.
# If left empty, signals are silently dropped by the signal router.
TELEGRAM_GEM_CHANNEL_ID: str = os.getenv("TELEGRAM_GEM_CHANNEL_ID", "")

# --- Merged Telegram Channel (recommended for user-facing deployment) ---
# When set, this OVERRIDES the individual per-channel IDs above.
# "Active Trading" channel receives: SCALP + SWING + SPOT + GEM signals
TELEGRAM_ACTIVE_CHANNEL_ID: str = os.getenv("TELEGRAM_ACTIVE_CHANNEL_ID", "")

# ---------------------------------------------------------------------------
# AI / Sentiment keys (optional)
# ---------------------------------------------------------------------------
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
SOCIAL_SENTIMENT_API_KEY: str = os.getenv("SOCIAL_SENTIMENT_API_KEY", "")

# Fear & Greed Index (free, no key needed)
FEAR_GREED_API_URL: str = os.getenv(
    "FEAR_GREED_API_URL", "https://api.alternative.me/fng/?limit=1"
)

# OpenAI GPT-4 – repurposed exclusively for macro/news event evaluation
# (no longer used in the trade-signal hot path)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Kept for backward compatibility – no longer used by the scanner.
OPENAI_MIN_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("OPENAI_MIN_CONFIDENCE_THRESHOLD", "85.0")
)
# Kept for backward compatibility – no longer used by the scanner.
OPENAI_HOT_PATH_BYPASS_CHANNELS: List[str] = ["360_SCALP"]

# ---------------------------------------------------------------------------
# Gem Scanner — macro-reversal detection for deeply discounted altcoins
# ---------------------------------------------------------------------------
GEM_SCANNER_ENABLED: bool = os.getenv("GEM_SCANNER_ENABLED", "true").lower() in (
    "true", "1", "yes"
)
GEM_MIN_DRAWDOWN_PCT: float = float(os.getenv("GEM_MIN_DRAWDOWN_PCT", "70.0"))
GEM_MAX_RANGE_PCT: float = float(os.getenv("GEM_MAX_RANGE_PCT", "40.0"))
GEM_MIN_VOLUME_RATIO: float = float(os.getenv("GEM_MIN_VOLUME_RATIO", "1.5"))
GEM_SCAN_INTERVAL_HOURS: int = int(os.getenv("GEM_SCAN_INTERVAL_HOURS", "6"))
GEM_MAX_DAILY_SIGNALS: int = int(os.getenv("GEM_MAX_DAILY_SIGNALS", "3"))
# Separate, wider pair universe for the gem scanner (small-cap gems like LYN)
GEM_PAIRS_COUNT: int = int(os.getenv("GEM_PAIRS_COUNT", "200"))
GEM_MIN_VOLUME_USD: float = float(os.getenv("GEM_MIN_VOLUME_USD", "250000"))
# Chart image generation for gem signals (requires mplfinance)
GEM_CHART_ENABLED: bool = os.getenv("GEM_CHART_ENABLED", "true").lower() in (
    "true", "1", "yes"
)

# ---------------------------------------------------------------------------
# Macro Watchdog – async background task for global market-event alerts
# ---------------------------------------------------------------------------
MACRO_WATCHDOG_ENABLED: bool = os.getenv("MACRO_WATCHDOG_ENABLED", "true").lower() in (
    "true", "1", "yes"
)
MACRO_WATCHDOG_POLL_INTERVAL: float = float(
    os.getenv("MACRO_WATCHDOG_POLL_INTERVAL", "300")  # seconds (5 min default)
)
MACRO_WATCHDOG_FEAR_GREED_THRESHOLD_LOW: int = int(
    os.getenv("MACRO_WATCHDOG_FEAR_GREED_THRESHOLD_LOW", "20")
)
MACRO_WATCHDOG_FEAR_GREED_THRESHOLD_HIGH: int = int(
    os.getenv("MACRO_WATCHDOG_FEAR_GREED_THRESHOLD_HIGH", "80")
)

# ---------------------------------------------------------------------------
# Dynamic Tiering (Market Watchdog) — PR 2
# ---------------------------------------------------------------------------
# Enable/disable the background TierManager that periodically polls Binance
# global 24hr tickers and re-ranks the entire pair universe into Hot / Warm /
# Cold tiers based on volume + volatility.
DYNAMIC_TIER_ENABLED: bool = os.getenv("DYNAMIC_TIER_ENABLED", "true").lower() in (
    "true", "1", "yes"
)
# How often (seconds) the TierManager polls Binance aggregate ticker endpoints.
DYNAMIC_TIER_POLL_INTERVAL: float = float(
    os.getenv("DYNAMIC_TIER_POLL_INTERVAL", "300")  # 5 minutes default
)
# Number of pairs in Tier 1 (Hot) — highest volume + volatility rank.
DYNAMIC_TIER1_HOT_COUNT: int = int(os.getenv("DYNAMIC_TIER1_HOT_COUNT", "50"))
# Total pairs in Tier 1 + Tier 2 combined; Tier 2 = (DYNAMIC_TIER12_WARM_CUTOFF - DYNAMIC_TIER1_HOT_COUNT).
DYNAMIC_TIER12_WARM_CUTOFF: int = int(os.getenv("DYNAMIC_TIER12_WARM_CUTOFF", "200"))
# Weighting of 24h quote-volume in the composite ranking score (0–1).
DYNAMIC_TIER_VOLUME_WEIGHT: float = float(os.getenv("DYNAMIC_TIER_VOLUME_WEIGHT", "0.7"))
# Weighting of absolute 24h price-change-percent in the composite ranking score (0–1).
DYNAMIC_TIER_VOLATILITY_WEIGHT: float = float(os.getenv("DYNAMIC_TIER_VOLATILITY_WEIGHT", "0.3"))
# Redis key names for tier membership sets.
DYNAMIC_TIER1_REDIS_KEY: str = os.getenv("DYNAMIC_TIER1_REDIS_KEY", "tier_1_active")
DYNAMIC_TIER2_REDIS_KEY: str = os.getenv("DYNAMIC_TIER2_REDIS_KEY", "tier_2_active")
DYNAMIC_TIER3_REDIS_KEY: str = os.getenv("DYNAMIC_TIER3_REDIS_KEY", "tier_3_active")

# On-chain intelligence — Glassnode (optional)
ONCHAIN_API_KEY: str = os.getenv("ONCHAIN_API_KEY", "")

# Whale Alert (free tier) — https://whale-alert.io/
# Optional; without a key on-chain scores fall back to Glassnode-only neutral
WHALE_ALERT_API_KEY: str = os.getenv("WHALE_ALERT_API_KEY", "")

# Etherscan (free tier, 5 calls/sec) — https://etherscan.io/apis
ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY", "")

# Cornix auto-execution signal formatting
# When true, a Cornix-compatible block is appended to SPOT/GEM/SWING signals
CORNIX_FORMAT_ENABLED: bool = os.getenv("CORNIX_FORMAT_ENABLED", "false").lower() in (
    "true", "1", "yes"
)

# Dynamic SL/TP based on ATR percentile, market regime, and pair tier (PR_07).
# Set to "false" to revert to the static signal_params.py behaviour for safety.
DYNAMIC_SL_TP_ENABLED: bool = os.getenv("DYNAMIC_SL_TP_ENABLED", "true").lower() in (
    "true", "1", "yes"
)

# ---------------------------------------------------------------------------
# Pair management
# ---------------------------------------------------------------------------
PAIR_FETCH_INTERVAL_HOURS: int = int(os.getenv("PAIR_FETCH_INTERVAL_HOURS", "6"))
TOP_PAIRS_COUNT: int = int(os.getenv("TOP_PAIRS_COUNT", "150"))
BATCH_REQUEST_DELAY: float = 0.75  # seconds between Binance REST calls
NEW_PAIR_MIN_CONFIDENCE: float = 50.0  # lower cap until enough data
# Minimum 24h USD volume for a symbol to be included in expensive API scans.
# Symbols below this threshold are skipped by the pre-filter before any
# order-book or kline fetches, reducing unnecessary weight consumption.
SCAN_MIN_VOLUME_USD: float = float(os.getenv("SCAN_MIN_VOLUME_USD", "500000"))

# ---------------------------------------------------------------------------
# Top-50 futures-only mode (PR1–PR5)
# ---------------------------------------------------------------------------
# When enabled, the engine restricts scanning, WS streams, and AI inference
# exclusively to the top-50 USDT-M futures pairs by 24h volume.  Spot pairs
# and all lower-ranked futures pairs are excluded.  This reduces API weight
# consumption and scan latency significantly.
TOP50_FUTURES_ONLY: bool = os.getenv("TOP50_FUTURES_ONLY", "false").lower() in (
    "true", "1", "yes"
)
# Number of top futures pairs to maintain in top-50 mode.
TOP50_FUTURES_COUNT: int = int(os.getenv("TOP50_FUTURES_COUNT", "50"))
# Minimum seconds between consecutive top-50 refresh calls (rate-limiting
# guard to prevent excessive Binance REST weight consumption).
TOP50_UPDATE_INTERVAL_SECONDS: int = int(os.getenv("TOP50_UPDATE_INTERVAL_SECONDS", "90"))

# ---------------------------------------------------------------------------
# Tiered pair universe
# ---------------------------------------------------------------------------
# Tier 1 — Core: top pairs by 24h volume.  Full scan every cycle, all channels,
# WebSocket streams + order book depth.  Primary signal source.
TIER1_PAIR_COUNT: int = int(os.getenv("TIER1_PAIR_COUNT", "75"))
# Tier 2 — Discovery: next tier by volume.  Scanned every N cycles, SWING +
# SPOT channels only (no SCALP), REST klines only (no WS, no order book).
TIER2_PAIR_COUNT: int = int(os.getenv("TIER2_PAIR_COUNT", "200"))
TIER2_SCAN_EVERY_N_CYCLES: int = int(os.getenv("TIER2_SCAN_EVERY_N_CYCLES", "3"))
# Tier 3 — Full Universe: all remaining USDT pairs.  Lightweight volume /
# momentum scan every N minutes.  Auto-promoted to Tier 2 on volume surges.
# Also supports cycle-based scheduling: Tier 3 is included in the main scan
# loop every TIER3_SCAN_EVERY_N_CYCLES cycles (default 6).
TIER3_SCAN_INTERVAL_MINUTES: int = int(os.getenv("TIER3_SCAN_INTERVAL_MINUTES", "30"))
TIER3_SCAN_EVERY_N_CYCLES: int = int(os.getenv("TIER3_SCAN_EVERY_N_CYCLES", "6"))
TIER3_VOLUME_SURGE_MULTIPLIER: float = float(os.getenv("TIER3_VOLUME_SURGE_MULTIPLIER", "3.0"))

# ---------------------------------------------------------------------------
# Tiered scanning configuration
# ---------------------------------------------------------------------------
SCANNING_TIERS: Dict[str, Any] = {
    "TIER_1_CRITICAL": {
        "pairs": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
        "scan_interval_seconds": 5,
        "priority": "HIGH",
    },
    "TIER_2_FUTURES_TOP": {
        "count": 50,
        "scan_interval_seconds": 15,
        "priority": "MEDIUM",
    },
    "TIER_3_SPOT_BATCH": {
        "count": 200,
        "scan_interval_seconds": 60,
        "priority": "LOW",
        "batch_size": 25,
    },
}

#: Enable adaptive per-tier regime threshold adjustment.
ADAPTIVE_REGIME_ENABLED: bool = os.getenv("ADAPTIVE_REGIME_ENABLED", "true").lower() in ("true", "1", "yes")
# When enabled, pairs absent from the latest exchange response are pruned from
# the active universe (handles delistings and low-volume pair removal).
PAIR_PRUNE_ENABLED: bool = os.getenv("PAIR_PRUNE_ENABLED", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Sweep detection tuning
# ---------------------------------------------------------------------------
# Scalp-optimised parameters: shorter lookback catches recent S/R levels
# relevant to 1m/5m timeframes; wider tolerance catches real institutional
# sweeps that reclaim $100-200 past the level on high-priced assets.
SMC_SCALP_LOOKBACK: int = int(os.getenv("SMC_SCALP_LOOKBACK", "20"))
SMC_SCALP_TOLERANCE_PCT: float = float(os.getenv("SMC_SCALP_TOLERANCE_PCT", "0.15"))
# Default (swing/spot) parameters — preserved for backward compatibility.
SMC_DEFAULT_LOOKBACK: int = int(os.getenv("SMC_DEFAULT_LOOKBACK", "50"))
SMC_DEFAULT_TOLERANCE_PCT: float = float(os.getenv("SMC_DEFAULT_TOLERANCE_PCT", "0.05"))


# ---------------------------------------------------------------------------
# Historical-data seeding – minimum candles per timeframe
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeframeSeed:
    interval: str
    limit: int


SEED_TIMEFRAMES: List[TimeframeSeed] = [
    TimeframeSeed("1m", 750),
    TimeframeSeed("5m", 750),
    TimeframeSeed("15m", 500),
    TimeframeSeed("1h", 500),
    TimeframeSeed("4h", 500),
    TimeframeSeed("1d", 365),
]
SEED_TICK_LIMIT: int = 5000  # recent trades

# Candle counts for gem scanner daily/weekly seeding (~1 year lookback).
# These are read from env-vars so they can be tuned without code changes.
GEM_SEED_DAILY_CANDLES: int = int(os.getenv("GEM_SEED_DAILY_CANDLES", "365"))
GEM_SEED_WEEKLY_CANDLES: int = int(os.getenv("GEM_SEED_WEEKLY_CANDLES", "52"))

# Timeframes fetched specifically for the gem scanner — daily for 1-year
# lookback and weekly for macro ATH detection.  Kept separate from
# SEED_TIMEFRAMES so existing SCALP/SWING/SPOT seeding is unaffected.
GEM_SEED_TIMEFRAMES: List[TimeframeSeed] = [
    TimeframeSeed("1d", GEM_SEED_DAILY_CANDLES),
    TimeframeSeed("1w", GEM_SEED_WEEKLY_CANDLES),
]

# ---------------------------------------------------------------------------
# Channel-level risk profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelConfig:
    name: str
    emoji: str
    timeframes: List[str]
    sl_pct_range: tuple  # (min%, max%)
    tp_ratios: List[float]  # R-multiples
    trailing_atr_mult: float
    adx_min: float
    adx_max: float
    spread_max: float
    min_confidence: float
    min_volume: float = 1_000_000.0  # minimum 24h USD volume
    # DCA (Double Entry / Dollar-Cost Averaging) config
    dca_enabled: bool = False                  # Whether DCA is enabled for this channel
    dca_zone_range: tuple = (0.30, 0.70)       # DCA zone as fraction of SL distance
    dca_weight_1: float = 0.6                  # Position weight for Entry 1
    dca_weight_2: float = 0.4                  # Position weight for Entry 2
    dca_min_momentum: float = 0.2              # Minimum |momentum| for DCA validation
    min_signal_lifespan: int = 900             # Default minimum lifespan; overridden per-channel by callers


# ---------------------------------------------------------------------------
# Per-Pair Config Profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairProfile:
    """Per-pair threshold profile applied on top of global channel config."""
    tier: str                         # "MAJOR", "MIDCAP", "ALTCOIN"
    # Multipliers applied to global config values (1.0 = no change)
    atr_mult: float = 1.0             # Multiplier for ATR-based SL distance
    momentum_threshold_mult: float = 1.0   # Multiplier for momentum threshold
    spread_max_mult: float = 1.0      # Multiplier for max spread tolerance
    volume_min_mult: float = 1.0      # Multiplier for minimum volume
    rsi_ob_level: float = 70.0        # RSI overbought level
    rsi_os_level: float = 30.0        # RSI oversold level
    adx_min_mult: float = 1.0         # Multiplier for minimum ADX
    bb_touch_pct: float = 0.002       # BB-touch proximity (0.2% default)
    momentum_persist_candles: int = 2  # Required consecutive momentum candles
    kill_zone_hard_gate: bool = False  # Hard-reject signals outside kill zones


# Tier profiles
PAIR_PROFILES: Dict[str, PairProfile] = {
    "MAJOR": PairProfile(
        tier="MAJOR",
        atr_mult=1.0,
        momentum_threshold_mult=0.8,   # BTC/ETH: lower threshold (tighter moves)
        spread_max_mult=0.5,           # Tighter spread requirement
        volume_min_mult=5.0,           # Higher absolute volume floor
        rsi_ob_level=75.0,
        rsi_os_level=25.0,
        adx_min_mult=0.9,
        bb_touch_pct=0.003,            # Slightly wider tolerance for majors
        momentum_persist_candles=2,
        kill_zone_hard_gate=False,
    ),
    "MIDCAP": PairProfile(
        tier="MIDCAP",
        atr_mult=1.1,
        momentum_threshold_mult=1.0,
        spread_max_mult=1.0,
        volume_min_mult=1.0,
        rsi_ob_level=70.0,
        rsi_os_level=30.0,
        adx_min_mult=1.0,
        bb_touch_pct=0.002,
        momentum_persist_candles=2,
        kill_zone_hard_gate=False,
    ),
    "ALTCOIN": PairProfile(
        tier="ALTCOIN",
        atr_mult=1.3,
        momentum_threshold_mult=2.0,   # High-vol pairs need larger momentum moves
        spread_max_mult=2.0,           # Wider spreads acceptable
        volume_min_mult=0.3,           # Lower volume floor (smaller markets)
        rsi_ob_level=65.0,
        rsi_os_level=35.0,
        adx_min_mult=1.1,
        bb_touch_pct=0.001,            # Tighter touch requirement
        momentum_persist_candles=3,    # Extra confirmation candles
        kill_zone_hard_gate=True,      # Hard-gate: only trade in kill zones
    ),
}

# Static symbol → tier mapping (auto-classified for unlisted pairs)
PAIR_TIER_MAP: Dict[str, str] = {
    "BTCUSDT": "MAJOR",
    "ETHUSDT": "MAJOR",
    "BNBUSDT": "MIDCAP",
    "SOLUSDT": "MIDCAP",
    "LINKUSDT": "MIDCAP",
    "MATICUSDT": "MIDCAP",
    "AVAXUSDT": "MIDCAP",
    "DOTUSDT": "MIDCAP",
    "DOGEUSDT": "ALTCOIN",
    "SHIBUSDT": "ALTCOIN",
    "PEPEUSDT": "ALTCOIN",
}


CHANNEL_SCALP = ChannelConfig(
    name="360_SCALP",
    emoji="⚡",
    timeframes=["1m", "5m"],
    sl_pct_range=(0.20, 0.50),
    tp_ratios=[1.5, 2.5, 4.0],
    trailing_atr_mult=1.5,
    adx_min=15,
    adx_max=100,
    spread_max=0.02,
    min_confidence=68,
    min_volume=5_000_000.0,
    dca_enabled=True,
    min_signal_lifespan=int(os.getenv("SCALP_MIN_LIFESPAN", "900")),
)

CHANNEL_SWING = ChannelConfig(
    name="360_SWING",
    emoji="🏛️",
    timeframes=["1h", "4h"],
    sl_pct_range=(0.2, 0.5),
    tp_ratios=[1.5, 3.0, 5.0],
    trailing_atr_mult=2.5,
    adx_min=20,
    adx_max=40,
    spread_max=0.02,
    min_confidence=72,
    min_volume=10_000_000.0,
    dca_enabled=True,
    min_signal_lifespan=int(os.getenv("SWING_MIN_LIFESPAN", "7200")),
)

CHANNEL_GEM = ChannelConfig(
    name="360_GEM",
    emoji="💎",
    timeframes=["1d", "1w"],
    sl_pct_range=(0.10, 0.30),
    tp_ratios=[2.0, 5.0, 10.0],
    trailing_atr_mult=3.0,
    adx_min=0,
    adx_max=100,
    spread_max=0.03,
    min_confidence=55,
    min_volume=1_000_000.0,
    dca_enabled=False,
    min_signal_lifespan=int(os.getenv("GEM_MIN_LIFESPAN", "43200")),
)

CHANNEL_SPOT = ChannelConfig(
    name="360_SPOT",
    emoji="📈",
    timeframes=["4h", "1d"],
    sl_pct_range=(0.005, 0.02),
    tp_ratios=[2.0, 5.0, 10.0],
    trailing_atr_mult=3.0,
    adx_min=0,
    adx_max=100,
    spread_max=0.02,
    min_confidence=65,
    min_volume=1_000_000.0,
    dca_enabled=True,
    dca_zone_range=(0.30, 0.70),
    dca_weight_1=0.6,
    dca_weight_2=0.4,
    dca_min_momentum=0.2,
    min_signal_lifespan=int(os.getenv("SPOT_MIN_LIFESPAN", "21600")),
)

# ---------------------------------------------------------------------------
# New scalp trigger channel configs (Phase 3)
# ---------------------------------------------------------------------------

CHANNEL_SCALP_FVG = ChannelConfig(
    name="360_SCALP_FVG",
    emoji="⚡",
    timeframes=["5m", "15m"],
    sl_pct_range=(0.05, 0.15),
    tp_ratios=[1.5, 2.5, 3.0],
    trailing_atr_mult=1.5,
    adx_min=15,
    adx_max=100,
    spread_max=0.02,
    min_confidence=68,
    min_volume=5_000_000.0,
    dca_enabled=True,
    min_signal_lifespan=int(os.getenv("SCALP_MIN_LIFESPAN", "900")),
)

CHANNEL_SCALP_CVD = ChannelConfig(
    name="360_SCALP_CVD",
    emoji="⚡",
    timeframes=["5m"],
    sl_pct_range=(0.15, 0.30),
    tp_ratios=[1.5, 2.5, 3.5],
    trailing_atr_mult=1.5,
    adx_min=15,
    adx_max=100,
    spread_max=0.02,
    min_confidence=68,
    min_volume=5_000_000.0,
    dca_enabled=True,
    min_signal_lifespan=int(os.getenv("SCALP_MIN_LIFESPAN", "900")),
)

CHANNEL_SCALP_VWAP = ChannelConfig(
    name="360_SCALP_VWAP",
    emoji="⚡",
    timeframes=["5m", "15m"],
    sl_pct_range=(0.10, 0.20),
    tp_ratios=[1.5, 2.5, 3.5],
    trailing_atr_mult=1.5,
    adx_min=0,
    adx_max=25,
    spread_max=0.02,
    min_confidence=68,
    min_volume=5_000_000.0,
    dca_enabled=True,
    min_signal_lifespan=int(os.getenv("SCALP_MIN_LIFESPAN", "900")),
)

CHANNEL_SCALP_OBI = ChannelConfig(
    name="360_SCALP_OBI",
    emoji="⚡",
    timeframes=["5m"],
    sl_pct_range=(0.10, 0.20),
    tp_ratios=[1.5, 2.5, 3.0],
    trailing_atr_mult=1.5,
    adx_min=0,
    adx_max=100,
    spread_max=0.02,
    min_confidence=68,
    min_volume=5_000_000.0,
    dca_enabled=True,
    min_signal_lifespan=int(os.getenv("SCALP_MIN_LIFESPAN", "900")),
)

ALL_CHANNELS: List[ChannelConfig] = [
    CHANNEL_SCALP,
    CHANNEL_SCALP_FVG,
    CHANNEL_SCALP_CVD,
    CHANNEL_SCALP_VWAP,
    CHANNEL_SCALP_OBI,
    CHANNEL_SWING,
    CHANNEL_SPOT,
    CHANNEL_GEM,
]

CHANNEL_EMOJIS: Dict[str, str] = {
    "360_SCALP": "⚡",
    "360_SWING": "🏛️",
    "360_SPOT": "📈",
    "360_GEM": "💎",
}

def _build_channel_telegram_map() -> Dict[str, str]:
    """Build the channel → Telegram chat-ID mapping.

    If the merged ``TELEGRAM_ACTIVE_CHANNEL_ID`` env var is set it takes
    precedence over the individual per-channel IDs, routing all signals to a
    single "Active Trading" channel.  When the merged var is **not** set the
    mapping falls back to the individual channel IDs, preserving full backward
    compatibility.
    """
    active = TELEGRAM_ACTIVE_CHANNEL_ID
    return {
        "360_SCALP":      active or TELEGRAM_SCALP_CHANNEL_ID,
        "360_SCALP_FVG":  active or TELEGRAM_SCALP_CHANNEL_ID,
        "360_SCALP_CVD":  active or TELEGRAM_SCALP_CHANNEL_ID,
        "360_SCALP_VWAP": active or TELEGRAM_SCALP_CHANNEL_ID,
        "360_SCALP_OBI":  active or TELEGRAM_SCALP_CHANNEL_ID,
        "360_SWING":      active or TELEGRAM_SWING_CHANNEL_ID,
        "360_SPOT":       active or TELEGRAM_SPOT_CHANNEL_ID,
        "360_GEM":        active or TELEGRAM_GEM_CHANNEL_ID,
    }


CHANNEL_TELEGRAM_MAP: Dict[str, str] = _build_channel_telegram_map()

CHART_ENABLED_CHANNELS: set = {"360_SPOT", "360_GEM"}

# ---------------------------------------------------------------------------
# WebSocket settings
# ---------------------------------------------------------------------------
# Binance allows up to 1024 streams per connection; keep well below that.
# 200 streams/connection is a safe operational cap that still gives plenty of
# room before Binance's hard limit while allowing reasonable shard counts.
WS_MAX_STREAMS_PER_CONN: int = int(os.getenv("WS_MAX_STREAMS_PER_CONN", "200"))
# Ping/pong latency threshold: if the RTT of a manual ping exceeds this value
# (in milliseconds) or a pong is not received within this window, the shard is
# force-closed so _run_connection can reconnect with fresh TCP state.
WS_PING_TIMEOUT_MS: int = int(os.getenv("WS_PING_TIMEOUT_MS", "5000"))
WS_HEARTBEAT_INTERVAL: int = 30  # seconds (spot)
# Futures WS endpoint (fstream.binance.com) is higher-throughput and can delay
# PONG responses beyond 45 s during liquidation cascades (e.g. Extreme Fear
# events); 60 s gives Binance enough headroom before aiohttp auto-closes.
WS_HEARTBEAT_INTERVAL_FUTURES: int = int(os.getenv("WS_HEARTBEAT_INTERVAL_FUTURES", "60"))
WS_RECONNECT_BASE_DELAY: float = 1.0
WS_RECONNECT_MAX_DELAY: float = 60.0
# Staleness multiplier: a connection is considered stale when
# (now - last_pong) >= heartbeat_interval * multiplier.
# Spot uses 10 (30 × 10 = 300 s).  Futures uses 15 (60 × 15 = 900 s) to
# provide extra headroom during liquidation cascades (Extreme Fear events)
# where Binance can delay PONG frames beyond the normal window.  The higher
# futures value also breaks the exact 600 s = WS_ALERT_COOLDOWN coincidence
# that was causing the repeating 10-minute drop/alert cycle.
WS_STALENESS_MULTIPLIER: int = 10  # spot
WS_STALENESS_MULTIPLIER_FUTURES: int = int(os.getenv("WS_STALENESS_MULTIPLIER_FUTURES", "15"))
# Admin alert dedup window (seconds) — alerts are throttled to at most one per
# 10-minute window per manager to avoid Telegram spam during prolonged outages.
WS_ALERT_COOLDOWN: int = int(os.getenv("WS_ALERT_COOLDOWN", "600"))
# How many consecutive failed reconnection attempts before the aiohttp session
# is recycled (clears stale TCP connection pool and DNS cache).
WS_SESSION_RECYCLE_ATTEMPTS: int = int(os.getenv("WS_SESSION_RECYCLE_ATTEMPTS", "5"))
# REST fallback — number of historical candles fetched in the one-time bulk
# backfill that warms indicator pipelines when a WS outage begins.
WS_FALLBACK_BULK_LIMIT: int = int(os.getenv("WS_FALLBACK_BULK_LIMIT", "200"))
# Timeframes fetched in the bulk backfill (covers all channel strategies).
WS_FALLBACK_TIMEFRAMES: List[str] = ["1m", "5m", "15m", "1h", "4h"]
# Timeframes polled in the ongoing limit=1 REST loop (most frequently needed).
WS_FALLBACK_POLL_INTERVALS: List[str] = ["1m", "5m"]

# ---------------------------------------------------------------------------
# Trade monitoring
# ---------------------------------------------------------------------------
MONITOR_POLL_INTERVAL: float = 5.0  # seconds

# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
TELEMETRY_INTERVAL: float = 60.0  # seconds

# ---------------------------------------------------------------------------
# Anti-duplicate: per-channel cooldown after a signal completes (seconds)
# ---------------------------------------------------------------------------
CHANNEL_COOLDOWN_SECONDS: Dict[str, int] = {
    "360_SCALP": 60,
    "360_SWING": 300,
    "360_SPOT": 600,
    "360_GEM": 21600,  # 6 hours — macro timeframe
}

# ---------------------------------------------------------------------------
# Scanner-level signal cooldown: per-(symbol, channel) cooldown after a
# signal is *fired* (i.e. enqueued), to prevent re-evaluating the same setup
# within the cooldown window.
# ---------------------------------------------------------------------------
SIGNAL_SCAN_COOLDOWN_SECONDS: Dict[str, int] = {
    "360_SCALP": int(os.getenv("SCALP_SCAN_COOLDOWN", "60")),
    "360_SWING": int(os.getenv("SWING_SCAN_COOLDOWN", "60")),
    "360_SPOT": int(os.getenv("SPOT_SCAN_COOLDOWN", "600")),
    "360_GEM": int(os.getenv("GEM_SCAN_COOLDOWN", "21600")),  # 6 hours
}

# ---------------------------------------------------------------------------
# Circuit Breaker thresholds
# ---------------------------------------------------------------------------
CIRCUIT_BREAKER_MAX_CONSECUTIVE_SL: int = int(
    os.getenv("CIRCUIT_BREAKER_MAX_CONSECUTIVE_SL", "3")
)
CIRCUIT_BREAKER_MAX_HOURLY_SL: int = int(
    os.getenv("CIRCUIT_BREAKER_MAX_HOURLY_SL", "5")
)
CIRCUIT_BREAKER_MAX_DAILY_DRAWDOWN_PCT: float = float(
    os.getenv("CIRCUIT_BREAKER_MAX_DAILY_DRAWDOWN_PCT", "10.0")
)
CIRCUIT_BREAKER_COOLDOWN_SECONDS: int = int(
    os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "900")
)

# Per-symbol consecutive SL tracking: after this many consecutive SL hits on
# the same symbol, that symbol is suppressed across all channels.
CIRCUIT_BREAKER_PER_SYMBOL_MAX_SL: int = int(
    os.getenv("CIRCUIT_BREAKER_PER_SYMBOL_MAX_SL", "3")
)
CIRCUIT_BREAKER_PER_SYMBOL_COOLDOWN_SECONDS: int = int(
    os.getenv("CIRCUIT_BREAKER_PER_SYMBOL_COOLDOWN_SECONDS", "3600")
)

# ---------------------------------------------------------------------------
# Thesis-based cooldown: after an SL hit, suppress the same (symbol, channel,
# direction, setup_class) tuple for a much longer period.
# ---------------------------------------------------------------------------
THESIS_COOLDOWN_AFTER_SL_SECONDS: Dict[str, int] = {
    "360_SCALP": int(os.getenv("THESIS_COOLDOWN_SCALP", "3600")),       # 1 hour
    "360_SWING": int(os.getenv("THESIS_COOLDOWN_SWING", "14400")),      # 4 hours
    "360_SPOT": int(os.getenv("THESIS_COOLDOWN_SPOT", "3600")),         # 1 hour
    "360_GEM": int(os.getenv("THESIS_COOLDOWN_GEM", "604800")),         # 7 days
}

# ---------------------------------------------------------------------------
# Performance Tracker persistence path
# ---------------------------------------------------------------------------
PERFORMANCE_TRACKER_PATH: str = os.getenv(
    "PERFORMANCE_TRACKER_PATH", "data/signal_performance.json"
)

# ---------------------------------------------------------------------------
# Max concurrent signals per channel.
#
# SCALP/SWING: capped for capital protection (leveraged trades).
# SPOT/GEM:    effectively unlimited (999) — long hold durations (7–30 days).
#              Natural daily throttle comes from GEM_MAX_DAILY_SIGNALS in
#              the gem scanner, not from a concurrent-position cap.
# ---------------------------------------------------------------------------
MAX_CONCURRENT_SIGNALS_PER_CHANNEL: Dict[str, int] = {
    "360_SCALP":      int(os.getenv("MAX_SCALP_SIGNALS", "5")),
    "360_SCALP_FVG":  int(os.getenv("MAX_SCALP_FVG_SIGNALS", "3")),
    "360_SCALP_CVD":  int(os.getenv("MAX_SCALP_CVD_SIGNALS", "3")),
    "360_SCALP_VWAP": int(os.getenv("MAX_SCALP_VWAP_SIGNALS", "3")),
    "360_SCALP_OBI":  int(os.getenv("MAX_SCALP_OBI_SIGNALS", "3")),
    "360_SWING":      int(os.getenv("MAX_SWING_SIGNALS", "10")),
    "360_SPOT":       int(os.getenv("MAX_SPOT_SIGNALS", "999")),
    "360_GEM":        int(os.getenv("MAX_GEM_SIGNALS", "999")),
}

# ---------------------------------------------------------------------------
# Signal Lifecycle Monitor — background loop that actively monitors every
# open SPOT, GEM, and SWING signal and posts human-readable updates to
# the Portfolio / Active Trading channels.
# ---------------------------------------------------------------------------
LIFECYCLE_CHECK_INTERVAL: Dict[str, int] = {
    "360_SWING": int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SWING", "14400")),   # 4 hours
    "360_SPOT":  int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SPOT",  "21600")),   # 6 hours
    "360_GEM":   int(os.getenv("LIFECYCLE_CHECK_INTERVAL_GEM",  "43200")),    # 12 hours
}

# Confidence drop thresholds for lifecycle alert levels.
# YELLOW fires when confidence drops by >= YELLOW points from entry.
# RED fires when confidence drops by >= RED points from entry (overrides YELLOW).
LIFECYCLE_CONFIDENCE_DROP_YELLOW: float = float(
    os.getenv("LIFECYCLE_CONFIDENCE_DROP_YELLOW", "15.0")
)
LIFECYCLE_CONFIDENCE_DROP_RED: float = float(
    os.getenv("LIFECYCLE_CONFIDENCE_DROP_RED", "25.0")
)

# ---------------------------------------------------------------------------
# Anti-noise: minimum signal lifespan before SL/TP checks are applied (secs)
# ---------------------------------------------------------------------------
MIN_SIGNAL_LIFESPAN_SECONDS: Dict[str, int] = {
    "360_SCALP":      int(os.getenv("MIN_LIFESPAN_SCALP",      "180")),
    "360_SCALP_FVG":  int(os.getenv("MIN_LIFESPAN_SCALP_FVG",  "180")),
    "360_SCALP_CVD":  int(os.getenv("MIN_LIFESPAN_SCALP_CVD",  "180")),
    "360_SCALP_VWAP": int(os.getenv("MIN_LIFESPAN_SCALP_VWAP", "180")),
    "360_SCALP_OBI":  int(os.getenv("MIN_LIFESPAN_SCALP_OBI",  "180")),
    "360_SWING":      int(os.getenv("MIN_LIFESPAN_SWING",       "300")),
    "360_SPOT":       int(os.getenv("MIN_LIFESPAN_SPOT",        "600")),
    "360_GEM":        int(os.getenv("MIN_LIFESPAN_GEM",         "21600")),
}

# ---------------------------------------------------------------------------
# QUIET regime scalp signal quality gates
# ---------------------------------------------------------------------------

#: Minimum confidence score for scalp signals to pass in QUIET regime.
#: Acts as a hard floor — only top-tier signals proceed when the market is
#: compressed.  Configurable via the QUIET_SCALP_MIN_CONFIDENCE env var.
QUIET_SCALP_MIN_CONFIDENCE: float = float(
    os.getenv("QUIET_SCALP_MIN_CONFIDENCE", "68.0")
)

#: Volume multiplier required for scalp entries in QUIET regime.
#: Scalp signals in low-volatility markets are only accepted when current
#: volume is at least this multiple of the rolling average, ensuring signals
#: fire on genuine micro-breakouts rather than random noise.
#: This constant is exported for use by volume-aware gate logic in the
#: scalp channel evaluation pipeline.  The `_compute_base_confidence`
#: path reads this value when checking volume-spike conditions in QUIET.
QUIET_SCALP_VOLUME_MULTIPLIER: float = float(
    os.getenv("QUIET_SCALP_VOLUME_MULTIPLIER", "2.5")
)

#: Confidence penalty applied to SCALP signals in QUIET regime.
REGIME_QUIET_PENALTY: float = float(os.getenv("REGIME_QUIET_PENALTY", "8.0"))

#: Confidence penalty applied to SCALP signals in RANGING regime with ADX below threshold.
REGIME_RANGING_PENALTY: float = float(os.getenv("REGIME_RANGING_PENALTY", "5.0"))

#: ADX threshold below which SCALP signals receive a soft penalty in RANGING.
RANGING_ADX_SUPPRESS_THRESHOLD: float = float(
    os.getenv("RANGING_ADX_SUPPRESS_THRESHOLD", "12.0")
)

# ---------------------------------------------------------------------------
# Per-channel pair quality thresholds (overridable via env vars)
# ---------------------------------------------------------------------------
PAIR_QUALITY_THRESHOLD_SCALP: float = float(os.getenv("PAIR_QUALITY_THRESHOLD_SCALP", "58.0"))
PAIR_QUALITY_THRESHOLD_SWING: float = float(os.getenv("PAIR_QUALITY_THRESHOLD_SWING", "50.0"))
PAIR_QUALITY_THRESHOLD_SPOT:  float = float(os.getenv("PAIR_QUALITY_THRESHOLD_SPOT",  "45.0"))
PAIR_QUALITY_THRESHOLD_GEM:   float = float(os.getenv("PAIR_QUALITY_THRESHOLD_GEM",   "40.0"))

PAIR_QUALITY_VOLUME_FLOOR_SWING: float = float(os.getenv("PAIR_QUALITY_VOLUME_FLOOR_SWING", "500000.0"))
PAIR_QUALITY_VOLUME_FLOOR_SPOT:  float = float(os.getenv("PAIR_QUALITY_VOLUME_FLOOR_SPOT",  "250000.0"))
PAIR_QUALITY_VOLUME_FLOOR_GEM:   float = float(os.getenv("PAIR_QUALITY_VOLUME_FLOOR_GEM",   "100000.0"))

# ---------------------------------------------------------------------------
# How long a signal setup remains actionable (minutes).  After this window
# users should NOT enter the trade even if price is still in zone.
# ---------------------------------------------------------------------------
SIGNAL_VALID_FOR_MINUTES: Dict[str, int] = {
    "360_SCALP":      int(os.getenv("SIGNAL_VALID_SCALP",  "15")),
    "360_SCALP_FVG":  int(os.getenv("SIGNAL_VALID_SCALP",  "15")),
    "360_SCALP_CVD":  int(os.getenv("SIGNAL_VALID_SCALP",  "15")),
    "360_SCALP_VWAP": int(os.getenv("SIGNAL_VALID_SCALP",  "15")),
    "360_SCALP_OBI":  int(os.getenv("SIGNAL_VALID_SCALP",  "15")),
    "360_SWING":      int(os.getenv("SIGNAL_VALID_SWING",   "60")),
    "360_SPOT":       int(os.getenv("SIGNAL_VALID_SPOT",   "240")),
    "360_GEM":        int(os.getenv("SIGNAL_VALID_GEM",   "1440")),
}

# ---------------------------------------------------------------------------
# Maximum hold duration per channel (seconds).  Signals older than this
# are auto-closed at current market price to free up concurrent-signal slots.
# ---------------------------------------------------------------------------
MAX_SIGNAL_HOLD_SECONDS: Dict[str, int] = {
    "360_SCALP": int(os.getenv("MAX_SCALP_HOLD", "3600")),       # 1 hour
    "360_SWING": int(os.getenv("MAX_SWING_HOLD", "172800")),     # 48 hours
    "360_SPOT": int(os.getenv("MAX_SPOT_HOLD", "604800")),       # 7 days
    "360_GEM": int(os.getenv("MAX_GEM_HOLD", "2592000")),        # 30 days
}

# ---------------------------------------------------------------------------
# Concurrency cap – DEPRECATED: replaced by per-channel cap above.
# Kept for backwards-compatibility with any external tooling that imports it.
# ---------------------------------------------------------------------------
MAX_CONCURRENT_SIGNALS: int = 5

# ---------------------------------------------------------------------------
# Signal invalidation – minimum age before market-structure checks apply (secs)
# ---------------------------------------------------------------------------
INVALIDATION_MIN_AGE_SECONDS: Dict[str, int] = {
    "360_SCALP": 600,       # was 300 — scalps need more time to develop on 1m/5m candles
    "360_SWING": 300,
    "360_SPOT": 1800,
    "360_GEM": 604800,      # 7 days — macro positions need much longer before invalidation
}

# Momentum threshold below which a signal is considered to have lost its thesis.
# Per-channel to account for different timeframe noise levels.
# SCALP uses 1m/5m candles which have rapid momentum oscillation — use a lower threshold.
INVALIDATION_MOMENTUM_THRESHOLD: Dict[str, float] = {
    "360_SCALP": float(os.getenv("INVALIDATION_MOMENTUM_THRESHOLD_SCALP", "0.10")),
    "360_SWING": float(os.getenv("INVALIDATION_MOMENTUM_THRESHOLD_SWING", "0.20")),
    "360_SPOT": float(os.getenv("INVALIDATION_MOMENTUM_THRESHOLD_SPOT", "0.30")),
    "360_GEM": float(os.getenv("INVALIDATION_MOMENTUM_THRESHOLD_GEM", "0.50")),
}

# Number of *consecutive* below-threshold momentum readings required before a
# signal is invalidated for momentum loss.  A single weak reading is common on
# 1m/5m candles (price pauses before continuation) — requiring two consecutive
# readings reduces false kills while still catching genuine exhaustion.
INVALIDATION_CONSECUTIVE_THRESHOLD: Dict[str, int] = {
    "360_SCALP": int(os.getenv("INVALIDATION_CONSECUTIVE_THRESHOLD_SCALP", "2")),
    "360_SWING": int(os.getenv("INVALIDATION_CONSECUTIVE_THRESHOLD_SWING", "1")),
    "360_SPOT": int(os.getenv("INVALIDATION_CONSECUTIVE_THRESHOLD_SPOT", "1")),
    "360_GEM": int(os.getenv("INVALIDATION_CONSECUTIVE_THRESHOLD_GEM", "1")),
}

# ---------------------------------------------------------------------------
# Backtester – default slippage per trade (percent, e.g. 0.03 = 0.03 %)
# ---------------------------------------------------------------------------
BACKTEST_SLIPPAGE_PCT: float = float(os.getenv("BACKTEST_SLIPPAGE_PCT", "0.03"))

# ---------------------------------------------------------------------------
# Auto-Execution (V3 groundwork) – when enabled the OrderManager will attempt
# to place orders directly on the exchange instead of (or in addition to)
# publishing Telegram signals.  Disabled by default; flip to True once real
# exchange API keys and order logic are wired in.
# ---------------------------------------------------------------------------
AUTO_EXECUTION_ENABLED: bool = os.getenv("AUTO_EXECUTION_ENABLED", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Exchange / CCXT execution config (feature 3)
# ---------------------------------------------------------------------------
EXCHANGE_ID: str = os.getenv("EXCHANGE_ID", "binance")
EXCHANGE_API_KEY: str = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET: str = os.getenv("EXCHANGE_API_SECRET", "")
EXCHANGE_SANDBOX: bool = os.getenv("EXCHANGE_SANDBOX", "true").lower() == "true"
POSITION_SIZE_PCT: float = float(os.getenv("POSITION_SIZE_PCT", "2.0"))
MAX_POSITION_USD: float = float(os.getenv("MAX_POSITION_USD", "100.0"))

# ---------------------------------------------------------------------------
# Trailing stop – ATR multiplier for adaptive trailing distance
# ---------------------------------------------------------------------------
TRAILING_ATR_MULTIPLIER: float = float(os.getenv("TRAILING_ATR_MULTIPLIER", "1.5"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# AI Trade Observer – background module that captures full trade lifecycle data
# and generates periodic AI-powered digests for the admin Telegram channel.
# ---------------------------------------------------------------------------
OBSERVER_ENABLED: bool = os.getenv("OBSERVER_ENABLED", "true").lower() in ("true", "1", "yes")
OBSERVER_POLL_INTERVAL: float = float(os.getenv("OBSERVER_POLL_INTERVAL", "60"))
OBSERVER_DIGEST_INTERVAL_SECONDS: int = int(os.getenv("OBSERVER_DIGEST_INTERVAL", "21600"))  # 6 hours
OBSERVER_DATA_PATH: str = os.getenv("OBSERVER_DATA_PATH", "data/trade_observations.json")
OBSERVER_MAX_OBSERVATIONS_PER_TRADE: int = int(os.getenv("OBSERVER_MAX_OBSERVATIONS", "120"))
OBSERVER_DIGEST_LOOKBACK_HOURS: int = int(os.getenv("OBSERVER_DIGEST_LOOKBACK", "24"))

# ---------------------------------------------------------------------------
# MTF hard block – when True, MTF misalignment is a hard veto (signal blocked)
# instead of a soft -5.0 confidence penalty.
# ---------------------------------------------------------------------------
MTF_HARD_BLOCK: bool = os.getenv("MTF_HARD_BLOCK", "true").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Correlated position exposure cap
# Maximum number of same-direction active scalp signals allowed concurrently.
# When this threshold is reached, additional signals in the same direction are
# blocked to limit correlated exposure (e.g. all LONG scalps stopped out by BTC).
# ---------------------------------------------------------------------------
MAX_CORRELATED_SCALP_SIGNALS: int = int(os.getenv("MAX_CORRELATED_SCALP_SIGNALS", "8"))

# ---------------------------------------------------------------------------
# Confidence log (data-driven weight profiling infrastructure)
# When enabled, compute_confidence() appends a structured JSON record to
# CONFIDENCE_LOG_PATH for each scored signal.  The log can be used offline
# for logistic-regression analysis to derive optimal weight profiles.
# ---------------------------------------------------------------------------
CONFIDENCE_LOG_ENABLED: bool = os.getenv("CONFIDENCE_LOG_ENABLED", "false").lower() in ("true", "1", "yes")
CONFIDENCE_LOG_PATH: str = os.getenv("CONFIDENCE_LOG_PATH", "data/confidence_log.jsonl")

# ---------------------------------------------------------------------------
# Macro blackout window – block signals before/after major macro events.
# ---------------------------------------------------------------------------
MACRO_BLACKOUT_PRE_MINUTES: int = int(os.getenv("MACRO_BLACKOUT_PRE_MINUTES", "30"))
MACRO_BLACKOUT_POST_MINUTES: int = int(os.getenv("MACRO_BLACKOUT_POST_MINUTES", "60"))


# ---------------------------------------------------------------------------
# No-signal watchdog — alert admin when no new signals are generated for an
# extended period while WebSocket health is degraded.
# ---------------------------------------------------------------------------
# Seconds without a new signal before the watchdog fires (default: 1 hour).
NO_SIGNAL_ALERT_THRESHOLD_SECONDS: int = int(
    os.getenv("NO_SIGNAL_ALERT_THRESHOLD_SECONDS", "3600")
)
# Minimum seconds between repeated no-signal alerts (cooldown to avoid spam).
NO_SIGNAL_ALERT_COOLDOWN_SECONDS: int = int(
    os.getenv("NO_SIGNAL_ALERT_COOLDOWN_SECONDS", "3600")
)

# ---------------------------------------------------------------------------
# Scan-latency circuit breaker thresholds
# ---------------------------------------------------------------------------
# Warn in logs when a single scan cycle exceeds this duration (ms).
SCAN_LATENCY_WARN_MS: float = float(os.getenv("SCAN_LATENCY_WARN_MS", "15000"))
# Fire an admin alert when latency exceeds this threshold for N consecutive
# cycles (see SCAN_LATENCY_ALERT_CONSECUTIVE).
SCAN_LATENCY_ALERT_MS: float = float(os.getenv("SCAN_LATENCY_ALERT_MS", "30000"))
# Number of consecutive over-threshold cycles before the admin alert fires.
SCAN_LATENCY_ALERT_CONSECUTIVE: int = int(os.getenv("SCAN_LATENCY_ALERT_CONSECUTIVE", "3"))
# When latency exceeds this value, automatically reduce the scan set to
# Tier 1 only and lower _MAX_ORDER_BOOK_FETCHES_PER_CYCLE temporarily (ms).
SCAN_LATENCY_REDUCE_MS: float = float(os.getenv("SCAN_LATENCY_REDUCE_MS", "60000"))

# ---------------------------------------------------------------------------
# WS health-aware scan gating
# ---------------------------------------------------------------------------
# Number of consecutive scan cycles with both WS managers unhealthy before
# an admin alert is sent.
WS_DEGRADED_CYCLES_ALERT: int = int(os.getenv("WS_DEGRADED_CYCLES_ALERT", "10"))

# Health-ratio threshold below which a single WS manager is considered
# "partially degraded".  When either WS manager drops below this fraction
# of healthy connections the scanner applies reduced scan limits to avoid
# burning Binance API weight on REST depth fetches for all 800 pairs.
# 0.5 = fewer than half of connections are open/non-stale → degraded mode.
WS_PARTIAL_HEALTH_THRESHOLD: float = float(
    os.getenv("WS_PARTIAL_HEALTH_THRESHOLD", "0.5")
)

# Maximum number of symbols to scan per cycle when WS is partially degraded.
# Reduces REST API consumption while still providing signals for top pairs.
WS_DEGRADED_MAX_PAIRS: int = int(os.getenv("WS_DEGRADED_MAX_PAIRS", "50"))

# ---------------------------------------------------------------------------
# Depth endpoint circuit breaker
# ---------------------------------------------------------------------------
# Consecutive timeout count for /fapi/v1/depth or /api/v3/depth that triggers
# the open-circuit state.  Lowered from 10 to 5 so the engine stops hammering
# a degraded depth endpoint sooner, reducing cumulative timeout latency.
DEPTH_CIRCUIT_BREAKER_THRESHOLD: int = int(
    os.getenv("DEPTH_CIRCUIT_BREAKER_THRESHOLD", "5")
)
# How long (seconds) the circuit stays open (depth fetches return None immediately).
DEPTH_CIRCUIT_BREAKER_COOLDOWN: float = float(
    os.getenv("DEPTH_CIRCUIT_BREAKER_COOLDOWN", "60")
)
# Maximum retries for depth endpoint specifically (prevents 75 s cumulative wait).
DEPTH_MAX_RETRIES: int = int(os.getenv("DEPTH_MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# WS reconnection resilience — escalation alert threshold
# ---------------------------------------------------------------------------
# After this many consecutive reconnect failures on any one connection, fire
# a "manual intervention needed" admin alert.
WS_RECONNECT_FAIL_ALERT_THRESHOLD: int = int(
    os.getenv("WS_RECONNECT_FAIL_ALERT_THRESHOLD", "50")
)

#: Interval (seconds) between WebSocket connection health checks.
WS_HEALTH_CHECK_INTERVAL: int = int(os.getenv("WS_HEALTH_CHECK_INTERVAL", "30"))
#: Minimum message rate (messages/minute) below which a connection is flagged unhealthy.
WS_MIN_MESSAGE_RATE: float = float(os.getenv("WS_MIN_MESSAGE_RATE", "1.0"))
#: Pairs that get dedicated (non-multiplexed) WebSocket connections for lowest latency.
WS_PRIORITY_DEDICATED_PAIRS: List[str] = [
    p.strip() for p in os.getenv(
        "WS_PRIORITY_DEDICATED_PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT"
    ).split(",") if p.strip()
]

# ---------------------------------------------------------------------------
# Suppression telemetry
# ---------------------------------------------------------------------------
#: Enable suppression telemetry collection.
SUPPRESSION_TELEMETRY_ENABLED: bool = os.getenv("SUPPRESSION_TELEMETRY_ENABLED", "true").lower() in ("true", "1", "yes")
#: Maximum number of suppression events to keep in memory.
SUPPRESSION_TELEMETRY_MAX_EVENTS: int = int(os.getenv("SUPPRESSION_TELEMETRY_MAX_EVENTS", "10000"))
