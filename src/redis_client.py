"""Async Redis client wrapper with graceful fallback to in-memory."""

import os
from typing import Optional

import redis.asyncio as redis

from src.utils import get_logger

log = get_logger("redis_client")


class RedisClient:
    """Thin wrapper around redis.asyncio.Redis with auto-reconnect and fallback."""

    def __init__(self, url: Optional[str] = None) -> None:
        self._url = url or os.getenv("REDIS_URL", "")
        self._redis: Optional[redis.Redis] = None
        self._available = False

    async def connect(self) -> bool:
        """Attempt to connect to Redis. Returns True if successful."""
        if not self._url:
            log.info("REDIS_URL not set — running in memory-only mode.")
            return False
        try:
            self._redis = redis.from_url(self._url, decode_responses=True)
            await self._redis.ping()
            self._available = True
            log.info("Connected to Redis at %s", self._url)
            return True
        except Exception as exc:
            log.warning("Redis connection failed (%s) — falling back to in-memory mode.", exc)
            self._redis = None
            self._available = False
            return False

    @property
    def available(self) -> bool:
        return self._available and self._redis is not None

    @property
    def mode(self) -> str:
        return "redis" if self.available else "memory"

    @property
    def client(self) -> Optional[redis.Redis]:
        return self._redis if self._available else None

    def mark_unavailable(self, operation: str, exc: Optional[Exception] = None) -> None:
        """Disable Redis usage after an operation failure.

        The caller can continue in explicit in-memory mode until :meth:`connect`
        is called again.
        """
        if exc is not None:
            log.warning(
                "Redis %s failed (%s) — switching to in-memory mode.",
                operation,
                exc,
            )
        elif self._available:
            log.warning("Redis %s unavailable — switching to in-memory mode.", operation)
        self._available = False

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            self._available = False
