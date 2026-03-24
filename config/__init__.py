"""360-Crypto-Eye-Scalping – configuration module.

All tunables live here so every other module simply does
``from config.settings import cfg`` and reads what it needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

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

# --- Merged Telegram Channels (recommended for user-facing deployment) ---
# When set, these OVERRIDE the individual per-channel IDs above.
# "Active Trading" channel receives: SCALP + SWING signals
# "Portfolio" channel receives: SPOT + GEM signals
TELEGRAM_ACTIVE_CHANNEL_ID: str = os.getenv("TELEGRAM_ACTIVE_CHANNEL_ID", "")
TELEGRAM_PORTFOLIO_CHANNEL_ID: str = os.getenv("TELEGRAM_PORTFOLIO_CHANNEL_ID", "")

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
TIER3_SCAN_INTERVAL_MINUTES: int = int(os.getenv("TIER3_SCAN_INTERVAL_MINUTES", "30"))
TIER3_VOLUME_SURGE_MULTIPLIER: float = float(os.getenv("TIER3_VOLUME_SURGE_MULTIPLIER", "3.0"))
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
)

CHANNEL_SCALP_CVD = ChannelConfig(
    name="360_SCALP_CVD",
    emoji="⚡",
    timeframes=["5m"],
    sl_pct_range=(0.15, 0.30),
    tp_ratios=[1.0, 2.0, 3.0],
    trailing_atr_mult=1.5,
    adx_min=15,
    adx_max=100,
    spread_max=0.02,
    min_confidence=68,
    min_volume=5_000_000.0,
    dca_enabled=True,
)

CHANNEL_SCALP_VWAP = ChannelConfig(
    name="360_SCALP_VWAP",
    emoji="⚡",
    timeframes=["5m", "15m"],
    sl_pct_range=(0.10, 0.20),
    tp_ratios=[1.0, 2.0, 3.0],
    trailing_atr_mult=1.5,
    adx_min=0,
    adx_max=25,
    spread_max=0.02,
    min_confidence=68,
    min_volume=5_000_000.0,
    dca_enabled=True,
)

CHANNEL_SCALP_OBI = ChannelConfig(
    name="360_SCALP_OBI",
    emoji="⚡",
    timeframes=["5m"],
    sl_pct_range=(0.10, 0.20),
    tp_ratios=[1.0, 1.5, 2.0],
    trailing_atr_mult=1.5,
    adx_min=0,
    adx_max=100,
    spread_max=0.02,
    min_confidence=68,
    min_volume=5_000_000.0,
    dca_enabled=True,
)

ALL_CHANNELS: List[ChannelConfig] = [
    CHANNEL_SCALP,
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

    If the merged ``TELEGRAM_ACTIVE_CHANNEL_ID`` / ``TELEGRAM_PORTFOLIO_CHANNEL_ID``
    env vars are set they take precedence over the individual per-channel IDs,
    routing all SCALP*/SWING signals to the "Active Trading" channel and
    SPOT/GEM signals to the "Portfolio" channel.  When the merged vars are
    **not** set the mapping falls back to the individual channel IDs, preserving
    full backward compatibility.
    """
    active = TELEGRAM_ACTIVE_CHANNEL_ID
    portfolio = TELEGRAM_PORTFOLIO_CHANNEL_ID
    return {
        "360_SCALP":      active or TELEGRAM_SCALP_CHANNEL_ID,
        "360_SCALP_FVG":  active or TELEGRAM_SCALP_CHANNEL_ID,
        "360_SCALP_CVD":  active or TELEGRAM_SCALP_CHANNEL_ID,
        "360_SCALP_VWAP": active or TELEGRAM_SCALP_CHANNEL_ID,
        "360_SCALP_OBI":  active or TELEGRAM_SCALP_CHANNEL_ID,
        "360_SWING":      active or TELEGRAM_SWING_CHANNEL_ID,
        "360_SPOT":       portfolio or TELEGRAM_SPOT_CHANNEL_ID,
        "360_GEM":        portfolio or TELEGRAM_GEM_CHANNEL_ID,
    }


CHANNEL_TELEGRAM_MAP: Dict[str, str] = _build_channel_telegram_map()

# ---------------------------------------------------------------------------
# Portfolio signal channels – channels that use the enhanced portfolio format
# (narrative, sector comparison, chart image) instead of the compact scalp format.
# ---------------------------------------------------------------------------
PORTFOLIO_CHANNELS: set = {"360_SPOT", "360_GEM"}
CHART_ENABLED_CHANNELS: set = {"360_SPOT", "360_GEM"}

# ---------------------------------------------------------------------------
# WebSocket settings
# ---------------------------------------------------------------------------
WS_MAX_STREAMS_PER_CONN: int = 50
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
# SPOT/GEM:    effectively unlimited (999) — these are portfolio
#              recommendations with long hold durations (7–30 days).
#              Capping them silences the Portfolio channel for weeks.
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
    "360_SCALP": 180,
    "360_SWING": 300,
    "360_SPOT": 600,
    "360_GEM": 86400,  # 1 day — macro positions
}

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
    "360_SCALP": 300,       # was 120 — too aggressive for 1m candle noise
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
