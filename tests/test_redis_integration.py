"""Tests for Redis integration — redis_client, signal_queue, state_cache.

All tests operate in fallback (no Redis) mode and do NOT require a running
Redis instance.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.redis_client import RedisClient
from src.signal_queue import SignalQueue, QUEUE_MAXSIZE
from src.state_cache import StateCache
from src.channels.base import Signal
from src.smc import Direction
from src.utils import utcnow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(channel="360_SCALP", symbol="BTCUSDT", direction=Direction.LONG, confidence=85):
    return Signal(
        channel=channel,
        symbol=symbol,
        direction=direction,
        entry=32000,
        stop_loss=31900,
        tp1=32100,
        tp2=32200,
        confidence=confidence,
        signal_id=f"TEST-{symbol}-001",
        timestamp=utcnow(),
    )


# ---------------------------------------------------------------------------
# RedisClient tests
# ---------------------------------------------------------------------------

class TestRedisClient:
    @pytest.mark.asyncio
    async def test_connect_no_url_returns_false(self):
        """When REDIS_URL is empty, connect() returns False gracefully."""
        client = RedisClient(url="")
        result = await client.connect()
        assert result is False
        assert client.available is False
        assert client.client is None

    @pytest.mark.asyncio
    async def test_connect_unreachable_returns_false(self):
        """When Redis is unreachable, connect() returns False gracefully."""
        client = RedisClient(url="redis://127.0.0.1:1")  # nothing listening
        result = await client.connect()
        assert result is False
        assert client.available is False

    @pytest.mark.asyncio
    async def test_close_when_not_connected_is_safe(self):
        """Calling close() before connecting should not raise."""
        client = RedisClient(url="")
        await client.close()  # must not raise
        assert client.available is False

    @pytest.mark.asyncio
    async def test_available_false_by_default(self):
        client = RedisClient()
        assert client.available is False

    @pytest.mark.asyncio
    async def test_client_property_none_when_unavailable(self):
        client = RedisClient(url="")
        assert client.client is None

    def test_mark_unavailable_switches_mode_to_memory(self):
        client = RedisClient(url="redis://example")
        client._available = True
        client.mark_unavailable("test")
        assert client.available is False
        assert client.mode == "memory"


# ---------------------------------------------------------------------------
# SignalQueue tests (fallback mode)
# ---------------------------------------------------------------------------

class TestSignalQueue:
    @pytest_asyncio.fixture
    async def queue(self):
        rc = RedisClient(url="")  # no Redis
        await rc.connect()
        return SignalQueue(rc)

    @pytest.mark.asyncio
    async def test_put_and_get_signal(self, queue):
        sig = _make_signal()
        await queue.put(sig)
        result = await queue.get(timeout=1.0)
        assert result is not None
        # Fallback returns Signal objects directly
        assert isinstance(result, Signal)
        assert result.signal_id == sig.signal_id

    @pytest.mark.asyncio
    async def test_get_returns_none_on_timeout(self, queue):
        result = await queue.get(timeout=0.05)
        assert result is None

    @pytest.mark.asyncio
    async def test_put_nowait_enqueues_signal(self, queue):
        sig = _make_signal()
        queue.put_nowait(sig)
        # Small wait to allow any scheduled task to execute
        await asyncio.sleep(0.01)
        result = await queue.get(timeout=0.5)
        assert result is not None
        assert isinstance(result, Signal)
        assert result.signal_id == sig.signal_id

    @pytest.mark.asyncio
    async def test_qsize_reflects_items(self, queue):
        assert await queue.qsize() == 0
        sig = _make_signal()
        await queue.put(sig)
        assert await queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_empty_when_no_items(self, queue):
        assert await queue.empty() is True

    @pytest.mark.asyncio
    async def test_not_empty_after_put(self, queue):
        await queue.put(_make_signal())
        assert await queue.empty() is False

    @pytest.mark.asyncio
    async def test_put_nowait_drops_when_full(self, queue):
        """Queue drops signals gracefully when at capacity."""
        # Fill the fallback queue to max
        for i in range(QUEUE_MAXSIZE):
            sig = _make_signal()
            sig.signal_id = f"TEST-{i}"
            queue._fallback.put_nowait(sig)

        # This should not raise — drops silently
        overflow = _make_signal()
        overflow.signal_id = "OVERFLOW"
        queue.put_nowait(overflow)  # must not raise

    @pytest.mark.asyncio
    async def test_multiple_put_get_ordering(self, queue):
        """Items retrieved in FIFO order from the fallback queue."""
        sig1 = _make_signal()
        sig1.signal_id = "ID-001"
        sig2 = _make_signal()
        sig2.signal_id = "ID-002"

        await queue.put(sig1)
        await queue.put(sig2)

        r1 = await queue.get(timeout=0.5)
        r2 = await queue.get(timeout=0.5)

        assert r1 is not None and r2 is not None
        assert r1.signal_id == "ID-001"
        assert r2.signal_id == "ID-002"

    @pytest.mark.asyncio
    async def test_put_failure_disables_redis_and_falls_back(self):
        redis_client = RedisClient(url="redis://example")
        redis_client._available = True
        redis_client._redis = MagicMock()
        redis_client._redis.rpush = AsyncMock(side_effect=RuntimeError("down"))
        queue = SignalQueue(redis_client)
        sig = _make_signal()

        ok = await queue.put(sig)
        result = await queue.get(timeout=0.5)

        assert ok is True
        assert redis_client.available is False
        assert isinstance(result, Signal)
        assert result.signal_id == sig.signal_id

    @pytest.mark.asyncio
    async def test_queue_stats_track_drops(self, queue):
        for i in range(QUEUE_MAXSIZE):
            sig = _make_signal()
            sig.signal_id = f"FULL-{i}"
            queue._fallback.put_nowait(sig)

        overflow = _make_signal()
        overflow.signal_id = "OVERFLOW-STATS"
        ok = queue.put_nowait(overflow)

        assert ok is False
        stats = queue.stats()
        assert stats["dropped_signals"] == 1
        assert stats["overflow_events"] == 1
        assert stats["last_dropped_signal_id"] == "OVERFLOW-STATS"


# ---------------------------------------------------------------------------
# StateCache tests (fallback mode)
# ---------------------------------------------------------------------------

class TestStateCache:
    @pytest_asyncio.fixture
    async def cache(self):
        rc = RedisClient(url="")  # no Redis
        await rc.connect()
        return StateCache(rc)

    @pytest.mark.asyncio
    async def test_set_and_get_string(self, cache):
        await cache.set("key1", "hello")
        val = await cache.get("key1")
        assert val == "hello"

    @pytest.mark.asyncio
    async def test_set_and_get_dict(self, cache):
        await cache.set("key2", {"a": 1, "b": 2})
        val = await cache.get("key2")
        assert json.loads(val) == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_get_missing_key_returns_none(self, cache):
        val = await cache.get("nonexistent")
        assert val is None

    @pytest.mark.asyncio
    async def test_delete_removes_key(self, cache):
        await cache.set("del_key", "value")
        await cache.delete("del_key")
        val = await cache.get("del_key")
        assert val is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_safe(self, cache):
        await cache.delete("ghost_key")  # must not raise

    @pytest.mark.asyncio
    async def test_incr_starts_at_one(self, cache):
        val = await cache.incr("counter1")
        assert val == 1

    @pytest.mark.asyncio
    async def test_incr_increments(self, cache):
        await cache.incr("counter2")
        await cache.incr("counter2")
        val = await cache.incr("counter2")
        assert val == 3

    @pytest.mark.asyncio
    async def test_set_with_ttl_uses_local_fallback(self, cache):
        """TTL should behave predictably in memory fallback too."""
        await cache.set("ttl_key", "data", ttl=60)
        val = await cache.get("ttl_key")
        assert val == "data"

    @pytest.mark.asyncio
    async def test_set_integer_value(self, cache):
        await cache.set("int_key", 42)
        val = await cache.get("int_key")
        assert json.loads(val) == 42

    @pytest.mark.asyncio
    async def test_local_ttl_expiry_is_enforced(self, cache):
        with patch("src.state_cache.time.monotonic", side_effect=[100.0, 100.5, 102.0]):
            await cache.set("ttl_key", "data", ttl=1)
            assert await cache.get("ttl_key") == "data"
            assert await cache.get("ttl_key") is None

    @pytest.mark.asyncio
    async def test_redis_get_failure_switches_to_local_mode(self):
        redis_client = RedisClient(url="redis://example")
        redis_client._available = True
        redis_client._redis = MagicMock()
        redis_client._redis.get = AsyncMock(side_effect=RuntimeError("down"))
        cache = StateCache(redis_client)
        cache._set_local("fallback", "value")

        value = await cache.get("fallback")

        assert value == "value"
        assert redis_client.available is False
