"""360-Crypto-Eye-Scalping – configuration module.

All tunables live here so every other module simply does
``from config.settings import cfg`` and reads what it needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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

# ---------------------------------------------------------------------------
# AI / Sentiment keys (optional)
# ---------------------------------------------------------------------------
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
SOCIAL_SENTIMENT_API_KEY: str = os.getenv("SOCIAL_SENTIMENT_API_KEY", "")

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


CHANNEL_SCALP = ChannelConfig(
    name="360_SCALP",
    emoji="⚡",
    timeframes=["1m", "5m"],
    sl_pct_range=(0.05, 0.1),
    tp_ratios=[1.0, 1.5, 2.0],
    trailing_atr_mult=1.5,
    adx_min=25,
    adx_max=100,
    spread_max=0.02,
    min_confidence=70,
    min_volume=5_000_000.0,
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
)

CHANNEL_RANGE = ChannelConfig(
    name="360_RANGE",
    emoji="⚖️",
    timeframes=["15m"],
    sl_pct_range=(0.1, 0.2),
    tp_ratios=[1.0, 1.5],
    trailing_atr_mult=1.0,
    adx_min=0,
    adx_max=20,
    spread_max=0.02,
    min_confidence=70,
    min_volume=1_000_000.0,
)

CHANNEL_TAPE = ChannelConfig(
    name="360_THE_TAPE",
    emoji="🐋",
    timeframes=["1m"],
    sl_pct_range=(0.1, 0.3),
    tp_ratios=[1.0, 2.0, 3.0],
    trailing_atr_mult=2.0,
    adx_min=0,
    adx_max=100,
    spread_max=0.02,
    min_confidence=75,
    min_volume=10_000_000.0,
)

ALL_CHANNELS: List[ChannelConfig] = [
    CHANNEL_SCALP,
    CHANNEL_SWING,
    CHANNEL_RANGE,
    CHANNEL_TAPE,
]

CHANNEL_TELEGRAM_MAP: Dict[str, str] = {
    "360_SCALP": TELEGRAM_SCALP_CHANNEL_ID,
    "360_SWING": TELEGRAM_SWING_CHANNEL_ID,
    "360_RANGE": TELEGRAM_RANGE_CHANNEL_ID,
    "360_THE_TAPE": TELEGRAM_TAPE_CHANNEL_ID,
}

# ---------------------------------------------------------------------------
# WebSocket settings
# ---------------------------------------------------------------------------
WS_MAX_STREAMS_PER_CONN: int = 5
WS_HEARTBEAT_INTERVAL: int = 30  # seconds
WS_RECONNECT_BASE_DELAY: float = 1.0
WS_RECONNECT_MAX_DELAY: float = 60.0

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
}

# ---------------------------------------------------------------------------
# Anti-noise: minimum signal lifespan before SL/TP checks are applied (secs)
# ---------------------------------------------------------------------------
MIN_SIGNAL_LIFESPAN_SECONDS: Dict[str, int] = {
    "360_SCALP": 10,
    "360_SWING": 30,
    "360_RANGE": 20,
    "360_THE_TAPE": 10,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
