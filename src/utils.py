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


class _LoguroBridge:
    """Thin wrapper that accepts ``%``- or ``{}``-style format strings.

    Loguru uses ``{}`` (``str.format``) style by default, while parts of this
    codebase still use stdlib ``%s`` / ``%d`` placeholders. This bridge
    pre-formats either style before handing the message to Loguru so touched
    modules can be standardized incrementally without breaking older callers.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: Any) -> None:
        self._logger = logger

    @staticmethod
    def _fmt(msg: str, args: tuple) -> str:
        if args:
            try:
                return msg % args
            except (TypeError, ValueError):
                try:
                    return msg.format(*args)
                except (IndexError, KeyError, ValueError, AttributeError) as exc:
                    _loguru_logger.warning(
                        "Log format error ({}) for message: {!r}", exc, msg
                    )
                    return str(msg)
        return msg

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(self._fmt(msg, args), **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(self._fmt(msg, args), **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(self._fmt(msg, args), **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(self._fmt(msg, args), **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.critical(self._fmt(msg, args), **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(self._fmt(msg, args), **kwargs)


def get_logger(name: str) -> _LoguroBridge:
    """Return a logger bound with *name* context, accepting ``%`` and ``{}`` formatting."""
    return _LoguroBridge(_loguru_logger.bind(name=name))


def utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def fmt_price(price: float) -> str:
    """Format a price with comma grouping and adaptive decimals.

    Precision tiers:
      ≥ $1,000 → 0 decimals  (e.g. BTC, ETH)
      ≥ $100   → 2 decimals  (e.g. SOL, BNB)
      ≥ $1     → 4 decimals  (e.g. DOT, UNI – avoids identical TP display)
      ≥ $0.001 → 6 decimals  (e.g. DOGE, SHIB)
      < $0.001 → 8 decimals  (e.g. BONK, SHIB micro-cap tokens)
    """
    if price >= 1_000:
        return f"{price:,.0f}"
    if price >= 100:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    if price >= 0.001:
        return f"{price:.6f}"
    return f"{price:.8f}"


def price_decimal_fmt(price: float) -> str:
    """Return a Python format spec string for adaptive decimal precision.

    Used for inline f-string formatting of raw prices where commas and
    the ``fmt_price`` full formatter are not desired (e.g. zone labels,
    invalidation messages). Consistent with the tiers in :func:`fmt_price`.

    Examples::

        f"{zone_low:{price_decimal_fmt(zone_low)}}"
    """
    if price >= 1.0:
        return ".4f"
    if price >= 0.001:
        return ".6f"
    return ".8f"


def fmt_ts(dt: Optional[datetime] = None) -> str:
    """Produce ``YYYY-MM-DD HH:MM:SS`` string."""
    dt = dt or utcnow()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def pct_change(old: float, new: float) -> float:
    """Return percentage change from *old* to *new*."""
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100.0
