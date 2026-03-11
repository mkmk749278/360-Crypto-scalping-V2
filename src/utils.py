"""Shared utilities for the 360-Crypto-Eye-Scalping engine."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger


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
