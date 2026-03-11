"""Shared utilities for the 360-Crypto-Eye-Scalping engine."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger as _loguru_logger

from config import LOG_LEVEL

# Configure loguru once
_loguru_logger.remove()  # remove default handler
_loguru_logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {extra[name]:<24} | {level:<7} | {message}",
    level=LOG_LEVEL.upper(),
)
_loguru_logger.add(
    "logs/engine_{time}.log",
    rotation="50 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {extra[name]:<24} | {level:<7} | {message}",
    level="DEBUG",
)
_configured = True


def get_logger(name: str) -> Any:
    """Return a loguru logger bound with *name* context."""
    return _loguru_logger.bind(name=name)


def utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def fmt_price(price: float) -> str:
    """Format a price with comma grouping and adaptive decimals."""
    if price >= 1_000:
        return f"{price:,.0f}"
    if price >= 1:
        return f"{price:,.2f}"
    return f"{price:.6f}"


def fmt_ts(dt: Optional[datetime] = None) -> str:
    """Produce ``YYYY-MM-DD HH:MM:SS`` string."""
    dt = dt or utcnow()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def pct_change(old: float, new: float) -> float:
    """Return percentage change from *old* to *new*."""
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100.0
