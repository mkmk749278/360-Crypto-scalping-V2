"""Redis-backed state cache with TTL support."""

import json
import time
from typing import Any, Optional

from src.redis_client import RedisClient
from src.utils import get_logger

log = get_logger("state_cache")

CACHE_PREFIX = "360crypto:cache:"


class StateCache:
    """Key-value cache backed by Redis, with optional TTL."""

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client
        self._local: dict[str, tuple[str, Optional[float]]] = {}

    def _set_local(self, key: str, value: str, ttl: int = 0) -> None:
        expiry = (time.monotonic() + ttl) if ttl > 0 else None
        self._local[key] = (value, expiry)

    def _get_local(self, key: str) -> Optional[str]:
        entry = self._local.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry is not None and time.monotonic() >= expiry:
            self._local.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        serialized = json.dumps(value) if not isinstance(value, str) else value
        if self._redis.available:
            try:
                if ttl > 0:
                    await self._redis.client.setex(f"{CACHE_PREFIX}{key}", ttl, serialized)  # type: ignore[union-attr]
                else:
                    await self._redis.client.set(f"{CACHE_PREFIX}{key}", serialized)  # type: ignore[union-attr]
                return
            except Exception as exc:
                self._redis.mark_unavailable("set", exc)
        self._set_local(key, serialized, ttl=ttl)

    async def get(self, key: str) -> Optional[str]:
        if self._redis.available:
            try:
                return await self._redis.client.get(f"{CACHE_PREFIX}{key}")  # type: ignore[union-attr]
            except Exception as exc:
                self._redis.mark_unavailable("get", exc)
        return self._get_local(key)

    async def delete(self, key: str) -> None:
        if self._redis.available:
            try:
                await self._redis.client.delete(f"{CACHE_PREFIX}{key}")  # type: ignore[union-attr]
            except Exception as exc:
                self._redis.mark_unavailable("delete", exc)
        self._local.pop(key, None)

    async def incr(self, key: str) -> int:
        if self._redis.available:
            try:
                return await self._redis.client.incr(f"{CACHE_PREFIX}{key}")  # type: ignore[union-attr]
            except Exception as exc:
                self._redis.mark_unavailable("incr", exc)
        try:
            current = self._get_local(key)
            val = int(current or 0) + 1
        except (ValueError, TypeError):
            val = 1
        _, expiry = self._local.get(key, ("", None))
        self._local[key] = (str(val), expiry)
        return val
