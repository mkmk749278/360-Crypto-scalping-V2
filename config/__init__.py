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
TELEGRAM_RANGE_CHANNEL_ID: str = os.getenv("TELEGRAM_RANGE_CHANNEL_ID", "")
TELEGRAM_TAPE_CHANNEL_ID: str = os.getenv("TELEGRAM_TAPE_CHANNEL_ID", "")
TELEGRAM_FREE_CHANNEL_ID: str = os.getenv("TELEGRAM_FREE_CHANNEL_ID", "")
TELEGRAM_ADMIN_CHAT_ID: str = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
TELEGRAM_GEM_CHANNEL_ID: str = os.getenv("TELEGRAM_GEM_CHANNEL_ID", "")

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
OPENAI_HOT_PATH_BYPASS_CHANNELS: List[str] = ["360_SCALP", "360_THE_TAPE"]

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

# ---------------------------------------------------------------------------
# Pair management
# ---------------------------------------------------------------------------
PAIR_FETCH_INTERVAL_HOURS: int = int(os.getenv("PAIR_FETCH_INTERVAL_HOURS", "6"))
TOP_PAIRS_COUNT: int = int(os.getenv("TOP_PAIRS_COUNT", "50"))
BATCH_REQUEST_DELAY: float = 0.75  # seconds between Binance REST calls
NEW_PAIR_MIN_CONFIDENCE: float = 50.0  # lower cap until enough data


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
]
SEED_TICK_LIMIT: int = 5000  # recent trades

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
    sl_pct_range=(0.05, 0.1),
    tp_ratios=[1.0, 1.5, 2.0],
    trailing_atr_mult=1.5,
    adx_min=20,
    adx_max=100,
    spread_max=0.02,
    min_confidence=70,
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
    min_confidence=75,
    min_volume=10_000_000.0,
    dca_enabled=True,
)

CHANNEL_RANGE = ChannelConfig(
    name="360_RANGE",
    emoji="⚖️",
    timeframes=["15m"],
    sl_pct_range=(0.1, 0.2),
    tp_ratios=[1.0, 1.5],
    trailing_atr_mult=1.0,
    adx_min=0,
    adx_max=25,
    spread_max=0.02,
    min_confidence=70,
    min_volume=1_000_000.0,
    dca_enabled=True,
    dca_zone_range=(0.25, 0.60),
    dca_weight_1=0.5,
    dca_weight_2=0.5,
    dca_min_momentum=0.10,
)

CHANNEL_TAPE = ChannelConfig(
    name="360_THE_TAPE",
    emoji="🐋",
    timeframes=["1m"],
    sl_pct_range=(0.1, 0.3),
    tp_ratios=[1.5, 3.0, 5.0],  # Was [1.0, 2.0, 3.0] — whale moves are bigger
    trailing_atr_mult=2.5,       # Was 2.0 — give whale trades more room
    adx_min=0,
    adx_max=100,
    spread_max=0.02,
    min_confidence=75,
    min_volume=10_000_000.0,
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

ALL_CHANNELS: List[ChannelConfig] = [
    CHANNEL_SCALP,
    CHANNEL_SWING,
    CHANNEL_RANGE,
    CHANNEL_TAPE,
    CHANNEL_GEM,
]

CHANNEL_TELEGRAM_MAP: Dict[str, str] = {
    "360_SCALP": TELEGRAM_SCALP_CHANNEL_ID,
    "360_SWING": TELEGRAM_SWING_CHANNEL_ID,
    "360_RANGE": TELEGRAM_RANGE_CHANNEL_ID,
    "360_THE_TAPE": TELEGRAM_TAPE_CHANNEL_ID,
    "360_GEM": TELEGRAM_GEM_CHANNEL_ID,
}

# ---------------------------------------------------------------------------
# WebSocket settings
# ---------------------------------------------------------------------------
WS_MAX_STREAMS_PER_CONN: int = 5
WS_HEARTBEAT_INTERVAL: int = 30  # seconds (spot)
# Futures WS endpoint (fstream.binance.com) is higher-throughput and can delay
# PONG responses beyond 45 s during liquidation cascades (e.g. Extreme Fear
# events); 60 s gives Binance enough headroom before aiohttp auto-closes.
WS_HEARTBEAT_INTERVAL_FUTURES: int = int(os.getenv("WS_HEARTBEAT_INTERVAL_FUTURES", "60"))
WS_RECONNECT_BASE_DELAY: float = 1.0
WS_RECONNECT_MAX_DELAY: float = 60.0
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
WS_FALLBACK_TIMEFRAMES: List[str] = ["1m", "5m", "15m", "1h"]
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
    "360_RANGE": 120,
    "360_THE_TAPE": 30,
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
    "360_RANGE": int(os.getenv("RANGE_SCAN_COOLDOWN", "60")),
    "360_THE_TAPE": int(os.getenv("TAPE_SCAN_COOLDOWN", "60")),
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
    "360_RANGE": int(os.getenv("THESIS_COOLDOWN_RANGE", "7200")),       # 2 hours
    "360_THE_TAPE": int(os.getenv("THESIS_COOLDOWN_TAPE", "1800")),     # 30 min
    "360_GEM": int(os.getenv("THESIS_COOLDOWN_GEM", "604800")),         # 7 days
}

# ---------------------------------------------------------------------------
# Performance Tracker persistence path
# ---------------------------------------------------------------------------
PERFORMANCE_TRACKER_PATH: str = os.getenv(
    "PERFORMANCE_TRACKER_PATH", "data/signal_performance.json"
)

# ---------------------------------------------------------------------------
# Max concurrent signals per channel (5 per channel, independently capped)
# ---------------------------------------------------------------------------
MAX_CONCURRENT_SIGNALS_PER_CHANNEL: Dict[str, int] = {
    "360_SCALP": int(os.getenv("MAX_SCALP_SIGNALS", "5")),
    "360_SWING": int(os.getenv("MAX_SWING_SIGNALS", "5")),
    "360_RANGE": int(os.getenv("MAX_RANGE_SIGNALS", "5")),
    "360_THE_TAPE": int(os.getenv("MAX_TAPE_SIGNALS", "5")),
    "360_GEM": int(os.getenv("MAX_GEM_SIGNALS", "3")),
}

# ---------------------------------------------------------------------------
# Anti-noise: minimum signal lifespan before SL/TP checks are applied (secs)
# ---------------------------------------------------------------------------
MIN_SIGNAL_LIFESPAN_SECONDS: Dict[str, int] = {
    "360_SCALP": 30,
    "360_SWING": 60,
    "360_RANGE": 30,
    "360_THE_TAPE": 20,
    "360_GEM": 86400,  # 1 day — macro positions
}

# ---------------------------------------------------------------------------
# Maximum hold duration per channel (seconds).  Signals older than this
# are auto-closed at current market price to free up concurrent-signal slots.
# ---------------------------------------------------------------------------
MAX_SIGNAL_HOLD_SECONDS: Dict[str, int] = {
    "360_SCALP": int(os.getenv("MAX_SCALP_HOLD", "3600")),       # 1 hour
    "360_SWING": int(os.getenv("MAX_SWING_HOLD", "172800")),     # 48 hours
    "360_RANGE": int(os.getenv("MAX_RANGE_HOLD", "7200")),       # 2 hours
    "360_THE_TAPE": int(os.getenv("MAX_TAPE_HOLD", "1800")),     # 30 min
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
    "360_RANGE": 180,
    "360_THE_TAPE": 300,    # increased from 180 — regime flips happen at 180s boundary
    "360_GEM": 604800,      # 7 days — macro positions need much longer before invalidation
}

# Momentum threshold below which a signal is considered to have lost its thesis.
# Per-channel to account for different timeframe noise levels.
# TAPE uses 1m candles which have rapid momentum oscillation — use a lower threshold.
INVALIDATION_MOMENTUM_THRESHOLD: Dict[str, float] = {
    "360_THE_TAPE": float(os.getenv("INVALIDATION_MOMENTUM_THRESHOLD_TAPE", "0.05")),
    "360_SCALP": float(os.getenv("INVALIDATION_MOMENTUM_THRESHOLD_SCALP", "0.10")),
    "360_RANGE": float(os.getenv("INVALIDATION_MOMENTUM_THRESHOLD_RANGE", "0.15")),
    "360_SWING": float(os.getenv("INVALIDATION_MOMENTUM_THRESHOLD_SWING", "0.20")),
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
# Trailing stop – ATR multiplier for adaptive trailing distance
# ---------------------------------------------------------------------------
TRAILING_ATR_MULTIPLIER: float = float(os.getenv("TRAILING_ATR_MULTIPLIER", "1.5"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
