"""Tests for src.signal_router – queue-based signal routing."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

import src.signal_router as signal_router_module
from src.channels.base import Signal
from src.signal_router import (
    SignalRouter,
    _signal_from_dict,
    _signal_to_dict,
    _REDIS_KEY_SIGNALS,
    _REDIS_KEY_POSITION_LOCK,
    _REDIS_KEY_COOLDOWNS,
)
from src.smc import Direction
from src.utils import utcnow


@pytest.fixture
def sent_messages():
    """Collects (chat_id, text) tuples sent by the router."""
    return []


@pytest.fixture
def queue():
    return asyncio.Queue()


@pytest.fixture
def router(queue, sent_messages, monkeypatch):
    for channel in ("360_SCALP", "360_SPOT", "360_SWING", "360_GEM"):
        monkeypatch.setitem(signal_router_module.CHANNEL_TELEGRAM_MAP, channel, "premium")

    async def mock_send(chat_id: str, text: str):
        sent_messages.append((chat_id, text))
        return True

    def mock_format(sig: Signal) -> str:
        return f"Signal: {sig.channel} {sig.symbol} {sig.direction.value}"

    return SignalRouter(queue=queue, send_telegram=mock_send, format_signal=mock_format)


def _make_signal(channel="360_SCALP", symbol="BTCUSDT", direction=Direction.LONG, confidence=85):
    return Signal(
        channel=channel,
        symbol=symbol,
        direction=direction,
        entry=32000,
        stop_loss=31900,
        tp1=32100,  # sl_dist=100, tp_dist=100 → R:R=1.0 ≥ 1.0 floor
        tp2=32200,
        confidence=confidence,
        signal_id=f"TEST-{symbol}-001",
        timestamp=utcnow(),
    )


class TestSignalRouter:
    @pytest.mark.asyncio
    async def test_signal_processed_and_sent(self, queue, router, sent_messages):
        sig = _make_signal(confidence=90)
        await queue.put(sig)
        # Run router briefly
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert sig.signal_id in router.active_signals

    @pytest.mark.asyncio
    async def test_send_exception_cleans_up_and_router_continues(self, monkeypatch):
        for channel in ("360_SCALP", "360_SPOT", "360_SWING", "360_GEM"):
            monkeypatch.setitem(signal_router_module.CHANNEL_TELEGRAM_MAP, channel, "premium")

        queue = asyncio.Queue()

        # Patch _delivery_sleep (not asyncio.sleep) so re-queue delays don't slow
        # the test without affecting the test's own asyncio.sleep() calls.
        # BTC will be re-queued twice (retries 0→1, 1→2) then permanently lost.
        # Order of send calls: BTC attempt1 (RuntimeError), ETH attempt1 (True),
        # BTC attempt2/retry1 (RuntimeError), BTC attempt3/retry2 (RuntimeError → permanent loss).
        async def instant_sleep(_secs):
            pass

        monkeypatch.setattr(signal_router_module, "_delivery_sleep", instant_sleep)

        send_results = [RuntimeError("telegram down"), True, RuntimeError("down"), RuntimeError("down")]

        async def flaky_send(chat_id: str, text: str):
            result = send_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        router = SignalRouter(
            queue=queue,
            send_telegram=flaky_send,
            format_signal=lambda sig: f"Signal: {sig.channel} {sig.symbol} {sig.direction.value}",
        )

        failed = _make_signal(symbol="BTCUSDT", confidence=90)
        failed.signal_id = "TEST-BTC-FAIL"
        succeeded = _make_signal(symbol="ETHUSDT", confidence=90)
        succeeded.signal_id = "TEST-ETH-OK"
        await queue.put(failed)
        await queue.put(succeeded)

        task = asyncio.create_task(router.start())
        # Allow enough time for both queued signals and all BTC retries to complete.
        await asyncio.sleep(0.5)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "TEST-BTC-FAIL" not in router.active_signals
        assert failed.symbol not in router._position_lock
        assert "TEST-ETH-OK" in router.active_signals
        assert router._position_lock[succeeded.symbol] == succeeded.direction

    @pytest.mark.asyncio
    async def test_low_confidence_filtered(self, queue, router, sent_messages):
        sig = _make_signal(confidence=30)  # below min 70
        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert sig.signal_id not in router.active_signals

    @pytest.mark.asyncio
    async def test_correlation_lock(self, queue, router, sent_messages):
        sig1 = _make_signal(symbol="BTCUSDT", direction=Direction.LONG, confidence=90)
        sig1.signal_id = "TEST-BTC-001"
        sig2 = _make_signal(symbol="BTCUSDT", direction=Direction.SHORT, confidence=90)
        sig2.signal_id = "TEST-BTC-002"

        await queue.put(sig1)
        await queue.put(sig2)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.3)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Only the first should be active (second blocked by correlation lock)
        assert "TEST-BTC-001" in router.active_signals
        assert "TEST-BTC-002" not in router.active_signals

    @pytest.mark.asyncio
    async def test_remove_signal(self, router):
        sig = _make_signal()
        router._active_signals[sig.signal_id] = sig
        router._position_lock[sig.symbol] = sig.direction

        router.remove_signal(sig.signal_id)
        assert sig.signal_id not in router.active_signals
        assert sig.symbol not in router._position_lock

    @pytest.mark.asyncio
    async def test_correlation_lock_blocks_same_direction(self, queue, router, sent_messages):
        """A second LONG for the same symbol must be blocked while the first is active."""
        sig1 = _make_signal(symbol="ETHUSDT", direction=Direction.LONG, confidence=90)
        sig1.signal_id = "TEST-ETH-001"
        sig2 = _make_signal(symbol="ETHUSDT", direction=Direction.LONG, confidence=90)
        sig2.signal_id = "TEST-ETH-002"

        await queue.put(sig1)
        await queue.put(sig2)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.3)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "TEST-ETH-001" in router.active_signals
        assert "TEST-ETH-002" not in router.active_signals

    @pytest.mark.asyncio
    async def test_cooldown_prevents_reentry(self, queue, router, sent_messages):
        """After a signal is removed, a new signal for the same (symbol, channel)
        within the cooldown window must be blocked."""
        sig1 = _make_signal(symbol="SOLUSDT", channel="360_SCALP", confidence=90)
        sig1.signal_id = "TEST-SOL-001"

        # Process first signal
        await queue.put(sig1)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert "TEST-SOL-001" in router.active_signals

        # Remove the signal (simulates SL hit) – cooldown clock starts now
        router.remove_signal("TEST-SOL-001")
        assert "TEST-SOL-001" not in router.active_signals
        assert ("SOLUSDT", "360_SCALP") in router._cooldown_timestamps

        # Immediately try a second signal for same (symbol, channel)
        sig2 = _make_signal(symbol="SOLUSDT", channel="360_SCALP", confidence=90)
        sig2.signal_id = "TEST-SOL-002"

        queue2 = asyncio.Queue()
        await queue2.put(sig2)
        router2 = SignalRouter(
            queue=queue2,
            send_telegram=router._send_telegram,
            format_signal=router._format_signal,
        )
        # Copy the cooldown state over so router2 sees the active cooldown
        router2._cooldown_timestamps = dict(router._cooldown_timestamps)

        task2 = asyncio.create_task(router2.start())
        await asyncio.sleep(0.2)
        await router2.stop()
        task2.cancel()
        try:
            await task2
        except asyncio.CancelledError:
            pass

        # Second signal should be blocked by cooldown
        assert "TEST-SOL-002" not in router2.active_signals

    @pytest.mark.asyncio
    async def test_cooldown_allows_reentry_after_expiry(self, queue, router, sent_messages):
        """After the cooldown window expires, a new signal for (symbol, channel)
        must be accepted."""
        # Manually set an expired cooldown timestamp
        router._cooldown_timestamps[("ADAUSDT", "360_SCALP")] = (
            datetime.now(timezone.utc) - timedelta(seconds=120)  # 120s ago ensures 60s SCALP cooldown has expired
        )

        sig = _make_signal(symbol="ADAUSDT", channel="360_SCALP", confidence=90)
        sig.signal_id = "TEST-ADA-001"

        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "TEST-ADA-001" in router.active_signals

    @pytest.mark.asyncio
    async def test_tp_direction_rejected_long(self, queue, router, sent_messages):
        """LONG signal where TP1 <= entry must be rejected."""
        sig = Signal(
            channel="360_SCALP",
            symbol="DOTUSDT",
            direction=Direction.LONG,
            entry=1.5100,
            stop_loss=1.5000,
            tp1=1.5100,  # TP1 == entry → invalid
            tp2=1.5200,
            confidence=85,
            signal_id="TEST-DOT-TP-LONG",
            timestamp=utcnow(),
        )
        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "TEST-DOT-TP-LONG" not in router.active_signals

    @pytest.mark.asyncio
    async def test_tp_direction_rejected_short(self, queue, router, sent_messages):
        """SHORT signal where TP1 >= entry must be rejected."""
        sig = Signal(
            channel="360_SCALP",
            symbol="AVNTUSDT",
            direction=Direction.SHORT,
            entry=0.175700,
            stop_loss=0.176500,
            tp1=0.177899,  # TP1 > entry for SHORT → invalid
            tp2=0.177522,
            confidence=85,
            signal_id="TEST-AVNT-TP-SHORT",
            timestamp=utcnow(),
        )
        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "TEST-AVNT-TP-SHORT" not in router.active_signals

    @pytest.mark.asyncio
    async def test_sl_direction_rejected_long(self, queue, router, sent_messages):
        """LONG signal where SL >= entry must be rejected."""
        sig = Signal(
            channel="360_SCALP",
            symbol="XYZUSDT",
            direction=Direction.LONG,
            entry=1.0000,
            stop_loss=1.0050,  # SL > entry for LONG → invalid
            tp1=1.0200,
            tp2=1.0300,
            confidence=85,
            signal_id="TEST-XYZ-SL-LONG",
            timestamp=utcnow(),
        )
        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "TEST-XYZ-SL-LONG" not in router.active_signals

    @pytest.mark.asyncio
    async def test_sl_direction_rejected_short(self, queue, router, sent_messages):
        """SHORT signal where SL <= entry must be rejected."""
        sig = Signal(
            channel="360_SCALP",
            symbol="PIPUSDT",
            direction=Direction.SHORT,
            entry=0.355990,
            stop_loss=0.354000,  # SL < entry for SHORT → invalid
            tp1=0.353000,
            tp2=0.351000,
            confidence=85,
            signal_id="TEST-PIP-SL-SHORT",
            timestamp=utcnow(),
        )
        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "TEST-PIP-SL-SHORT" not in router.active_signals

    @pytest.mark.asyncio
    async def test_per_channel_cap_blocks_excess_within_same_channel(self, queue, router, sent_messages):
        """When a channel is full, additional signals for that channel are blocked."""
        from config import MAX_CONCURRENT_SIGNALS_PER_CHANNEL

        channel = "360_SCALP"
        cap = MAX_CONCURRENT_SIGNALS_PER_CHANNEL.get(channel, 5)

        # Pre-fill the channel to its cap using distinct symbols
        for i in range(cap):
            dummy = _make_signal(symbol=f"DUMMY{i}USDT", channel=channel, confidence=90)
            dummy.signal_id = f"DUMMY-{i}"
            router._active_signals[dummy.signal_id] = dummy
            router._position_lock[dummy.symbol] = dummy.direction

        # Now try to add one more signal for the same channel (brand-new symbol)
        sig = Signal(
            channel=channel,
            symbol="NEWUSDT",
            direction=Direction.LONG,
            entry=1.0000,
            stop_loss=0.9900,
            tp1=1.0200,
            tp2=1.0300,
            confidence=90,
            signal_id="TEST-NEW-CAP",
            timestamp=utcnow(),
        )
        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The new signal must be blocked; channel cap must not be exceeded
        assert "TEST-NEW-CAP" not in router.active_signals
        channel_count = sum(
            1 for s in router.active_signals.values() if s.channel == channel
        )
        assert channel_count == cap

    @pytest.mark.asyncio
    async def test_per_channel_cap_does_not_block_other_channels(self, queue, router, sent_messages):
        """When one channel is full, signals from other channels are still accepted."""
        from config import MAX_CONCURRENT_SIGNALS_PER_CHANNEL

        scalp_channel = "360_SCALP"
        scalp_cap = MAX_CONCURRENT_SIGNALS_PER_CHANNEL.get(scalp_channel, 5)

        # Pre-fill the SCALP channel to its cap
        for i in range(scalp_cap):
            dummy = _make_signal(symbol=f"SCALP{i}USDT", channel=scalp_channel, confidence=90)
            dummy.signal_id = f"SCALP-DUMMY-{i}"
            router._active_signals[dummy.signal_id] = dummy
            router._position_lock[dummy.symbol] = dummy.direction

        # Now try to add a signal for a DIFFERENT channel (SWING)
        sig = Signal(
            channel="360_SWING",
            symbol="SWINGUSDT",
            direction=Direction.LONG,
            entry=1.0000,
            stop_loss=0.9900,
            tp1=1.0200,
            tp2=1.0300,
            confidence=90,
            signal_id="TEST-SWING-CROSS",
            timestamp=utcnow(),
        )
        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The SWING signal must be accepted even though SCALP is full
        assert "TEST-SWING-CROSS" in router.active_signals

    @pytest.mark.asyncio
    async def test_spot_gem_signals_never_blocked_by_cap(self, queue, router, sent_messages):
        """SPOT and GEM signals are never blocked by per-channel caps (cap=999)."""
        from config import MAX_CONCURRENT_SIGNALS_PER_CHANNEL

        spot_cap = MAX_CONCURRENT_SIGNALS_PER_CHANNEL.get("360_SPOT", 999)
        gem_cap = MAX_CONCURRENT_SIGNALS_PER_CHANNEL.get("360_GEM", 999)

        # Both caps should be effectively unlimited (999 by default).
        assert spot_cap >= 999, (
            f"360_SPOT cap should be 999 (unlimited) but got {spot_cap}. "
            "SPOT signals have 7-day hold — a lower cap silences the Portfolio channel."
        )
        assert gem_cap >= 999, (
            f"360_GEM cap should be 999 (unlimited) but got {gem_cap}. "
            "GEM signals have 30-day hold — a lower cap silences the Portfolio channel."
        )

        # Pre-fill SPOT channel with 10 dummy signals (well above the old cap of 5)
        for i in range(10):
            dummy = _make_signal(symbol=f"SPOT{i}USDT", channel="360_SPOT", confidence=90)
            dummy.signal_id = f"SPOT-DUMMY-{i}"
            router._active_signals[dummy.signal_id] = dummy
            router._position_lock[dummy.symbol] = dummy.direction

        # A new SPOT signal for a fresh symbol must still be accepted.
        sig = Signal(
            channel="360_SPOT",
            symbol="NEWSPOTUSDT",
            direction=Direction.LONG,
            entry=1.0000,
            stop_loss=0.9900,
            tp1=1.0200,
            tp2=1.0300,
            confidence=90,
            signal_id="TEST-SPOT-UNLIMITED",
            timestamp=utcnow(),
        )
        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "TEST-SPOT-UNLIMITED" in router.active_signals, (
            "SPOT signal was blocked despite cap being 999 — per-channel cap logic is wrong."
        )

    @pytest.mark.asyncio
    async def test_failed_send_does_not_leave_active_signal_or_lock(self, queue, sent_messages, monkeypatch):
        monkeypatch.setitem(signal_router_module.CHANNEL_TELEGRAM_MAP, "360_SCALP", "premium")

        async def failed_send(_chat_id: str, _text: str):
            sent_messages.append(("failed", "attempt"))
            return False

        router = SignalRouter(
            queue=queue,
            send_telegram=failed_send,
            format_signal=lambda sig: f"Signal: {sig.signal_id}",
        )
        sig = _make_signal(confidence=90)
        sig.signal_id = "TEST-SEND-FAIL"

        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "TEST-SEND-FAIL" not in router.active_signals
        assert sig.symbol not in router._position_lock

    @pytest.mark.asyncio
    async def test_failed_delivery_requeues_signal(self, monkeypatch):
        """A failed delivery re-queues the signal (appears back in queue)."""
        for channel in ("360_SCALP", "360_SPOT", "360_SWING", "360_GEM"):
            monkeypatch.setitem(signal_router_module.CHANNEL_TELEGRAM_MAP, channel, "premium")

        # Patch _delivery_sleep to be instant
        async def instant_sleep(_secs):
            pass

        monkeypatch.setattr(signal_router_module, "_delivery_sleep", instant_sleep)

        queue = asyncio.Queue()
        send_call_count = [0]

        # Always fail to deliver; we stop the router after the first failure+requeue
        async def always_fail(_chat_id: str, _text: str):
            send_call_count[0] += 1
            return False

        router = SignalRouter(
            queue=queue,
            send_telegram=always_fail,
            format_signal=lambda sig: f"Signal: {sig.signal_id}",
        )

        sig = _make_signal(confidence=90)
        sig.signal_id = "TEST-REQUEUE"
        await queue.put(sig)

        task = asyncio.create_task(router.start())
        # Give enough time for first attempt + one re-queue cycle
        await asyncio.sleep(0.3)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Signal was attempted at least once and re-queued (retry counter incremented)
        assert send_call_count[0] >= 1
        assert sig._delivery_retries >= 1
        assert "TEST-REQUEUE" not in router.active_signals

    @pytest.mark.asyncio
    async def test_failed_delivery_permanent_loss_after_max_retries(self, monkeypatch):
        """Signal is permanently dropped (with log) after 3 failed delivery attempts."""
        for channel in ("360_SCALP", "360_SPOT", "360_SWING", "360_GEM"):
            monkeypatch.setitem(signal_router_module.CHANNEL_TELEGRAM_MAP, channel, "premium")

        async def instant_sleep(_secs):
            pass

        monkeypatch.setattr(signal_router_module, "_delivery_sleep", instant_sleep)

        queue = asyncio.Queue()
        send_call_count = [0]

        async def always_fail(_chat_id: str, _text: str):
            send_call_count[0] += 1
            return False

        router = SignalRouter(
            queue=queue,
            send_telegram=always_fail,
            format_signal=lambda sig: f"Signal: {sig.signal_id}",
        )

        sig = _make_signal(confidence=90)
        sig.signal_id = "TEST-PERMANENT-LOSS"
        await queue.put(sig)

        task = asyncio.create_task(router.start())
        # Allow sufficient time for all 3 attempts (2 sends + permanent loss on 3rd)
        await asyncio.sleep(0.5)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # All 3 send attempts completed (2 re-queues + final permanent loss)
        assert send_call_count[0] == 3
        assert sig._delivery_retries == 2
        assert "TEST-PERMANENT-LOSS" not in router.active_signals
        assert sig.symbol not in router._position_lock


        sig = _make_signal(confidence=95)
        router._daily_best = [sig]
        router.set_free_limit(0)
        assert router._daily_best == []

    @pytest.mark.asyncio
    async def test_publish_free_signals_respects_zero_limit(self, sent_messages):
        async def mock_send(chat_id: str, text: str):
            sent_messages.append((chat_id, text))
            return True

        router = SignalRouter(
            queue=asyncio.Queue(),
            send_telegram=mock_send,
            format_signal=lambda sig: f"Signal: {sig.signal_id}",
        )
        router._daily_best = [_make_signal(confidence=95)]
        router.set_free_limit(0)

        await router.publish_free_signals()

        assert sent_messages == []


def _make_mock_redis(stored: dict):
    """Build a fake RedisClient that stores/retrieves from `stored` dict."""
    mock_redis = MagicMock()
    mock_redis.available = True

    async def fake_get(key):
        return stored.get(key)

    async def fake_set(key, value):
        stored[key] = value

    mock_client = MagicMock()
    mock_client.get = fake_get
    mock_client.set = fake_set
    mock_redis.client = mock_client
    return mock_redis


class TestSignalToDict:
    """_signal_to_dict must produce a JSON-serializable, reversible dict."""

    def test_direction_is_string(self):
        sig = _make_signal()
        d = _signal_to_dict(sig)
        assert isinstance(d["direction"], str)
        assert d["direction"] == "LONG"

    def test_timestamp_is_isoformat(self):
        sig = _make_signal()
        d = _signal_to_dict(sig)
        assert isinstance(d["timestamp"], str)
        # Must be valid ISO format
        datetime.fromisoformat(d["timestamp"])

    def test_roundtrip_via_signal_from_dict(self):
        sig = _make_signal(symbol="ETHUSDT", confidence=88)
        d = _signal_to_dict(sig)
        restored = _signal_from_dict(d)
        assert restored is not None
        assert restored.symbol == sig.symbol
        assert restored.confidence == sig.confidence
        assert restored.direction == sig.direction

    def test_json_serializable(self):
        sig = _make_signal()
        d = _signal_to_dict(sig)
        # Must not raise
        json.dumps(d)

    def test_lifecycle_datetime_fields_serialized(self):
        """last_lifecycle_check and dca_timestamp must be ISO strings, not datetime objects."""
        sig = _make_signal()
        sig.last_lifecycle_check = utcnow()
        sig.dca_timestamp = utcnow()
        d = _signal_to_dict(sig)
        # Both datetime fields must be ISO strings
        assert isinstance(d["last_lifecycle_check"], str)
        assert isinstance(d["dca_timestamp"], str)
        datetime.fromisoformat(d["last_lifecycle_check"])
        datetime.fromisoformat(d["dca_timestamp"])
        # Must not raise
        json.dumps(d)

    def test_lifecycle_datetime_fields_roundtrip(self):
        """last_lifecycle_check and dca_timestamp must survive a serialize→deserialize round-trip."""
        now = utcnow()
        sig = _make_signal()
        sig.last_lifecycle_check = now
        sig.dca_timestamp = now
        d = _signal_to_dict(sig)
        restored = _signal_from_dict(d)
        assert restored is not None
        assert restored.last_lifecycle_check == now
        assert restored.dca_timestamp == now


class TestRedisPersistence:
    """SignalRouter must persist and restore state via RedisClient."""

    def _make_router(self, redis_store: dict):
        mock_redis = _make_mock_redis(redis_store)

        async def mock_send(chat_id: str, text: str):
            return True

        router = SignalRouter(
            queue=asyncio.Queue(),
            send_telegram=mock_send,
            format_signal=lambda sig: f"Signal: {sig.signal_id}",
            redis_client=mock_redis,
        )
        return router

    @pytest.mark.asyncio
    async def test_persist_state_saves_active_signals(self):
        """_persist_state must write active signals, position lock, and cooldowns to Redis."""
        store: dict = {}
        router = self._make_router(store)

        sig = _make_signal(symbol="BTCUSDT", confidence=90)
        router._active_signals[sig.signal_id] = sig
        router._position_lock[sig.symbol] = sig.direction

        await router._persist_state()

        assert _REDIS_KEY_SIGNALS in store
        saved = json.loads(store[_REDIS_KEY_SIGNALS])
        assert sig.signal_id in saved
        assert _REDIS_KEY_POSITION_LOCK in store
        lock = json.loads(store[_REDIS_KEY_POSITION_LOCK])
        assert lock.get("BTCUSDT") == "LONG"

    @pytest.mark.asyncio
    async def test_restore_reloads_active_signals(self):
        """restore() must load previously persisted signals back into memory."""
        sig = _make_signal(symbol="SOLUSDT", confidence=82)
        store: dict = {
            _REDIS_KEY_SIGNALS: json.dumps({sig.signal_id: _signal_to_dict(sig)}),
            _REDIS_KEY_POSITION_LOCK: json.dumps({"SOLUSDT": "LONG"}),
            _REDIS_KEY_COOLDOWNS: json.dumps({}),
        }
        router = self._make_router(store)
        await router.restore()

        assert sig.signal_id in router._active_signals
        assert router._position_lock.get("SOLUSDT") == Direction.LONG

    @pytest.mark.asyncio
    async def test_restore_reloads_cooldown_timestamps(self):
        """restore() must reload cooldown timestamps with proper tuple keys."""
        ts = datetime.now(timezone.utc)
        store: dict = {
            _REDIS_KEY_SIGNALS: json.dumps({}),
            _REDIS_KEY_POSITION_LOCK: json.dumps({}),
            _REDIS_KEY_COOLDOWNS: json.dumps({"ADAUSDT|360_SCALP": ts.isoformat()}),
        }
        router = self._make_router(store)
        await router.restore()

        assert ("ADAUSDT", "360_SCALP") in router._cooldown_timestamps

    @pytest.mark.asyncio
    async def test_persist_called_on_remove_signal(self):
        """remove_signal() must schedule a Redis persist."""
        store: dict = {}
        router = self._make_router(store)

        sig = _make_signal()
        router._active_signals[sig.signal_id] = sig
        router._position_lock[sig.symbol] = sig.direction

        router.remove_signal(sig.signal_id)
        # Flush pending tasks
        await asyncio.sleep(0)

        assert sig.signal_id not in router._active_signals
        # Persistence must have fired (signals key updated)
        assert _REDIS_KEY_SIGNALS in store

    @pytest.mark.asyncio
    async def test_persist_called_on_update_signal(self):
        """update_signal() must schedule a Redis persist."""
        store: dict = {}
        router = self._make_router(store)

        sig = _make_signal()
        router._active_signals[sig.signal_id] = sig

        router.update_signal(sig.signal_id, status="TP1_HIT")
        # Flush pending tasks
        await asyncio.sleep(0)

        assert router._active_signals[sig.signal_id].status == "TP1_HIT"
        assert _REDIS_KEY_SIGNALS in store

    @pytest.mark.asyncio
    async def test_no_redis_skips_persist(self):
        """When no redis_client is provided, _persist_state must be a no-op."""
        async def mock_send(chat_id: str, text: str):
            return True

        router = SignalRouter(
            queue=asyncio.Queue(),
            send_telegram=mock_send,
            format_signal=lambda sig: f"Signal: {sig.signal_id}",
            redis_client=None,
        )
        # Must not raise
        await router._persist_state()

    @pytest.mark.asyncio
    async def test_no_redis_skips_restore(self):
        """When no redis_client is provided, restore() must be a no-op."""
        async def mock_send(chat_id: str, text: str):
            return True

        router = SignalRouter(
            queue=asyncio.Queue(),
            send_telegram=mock_send,
            format_signal=lambda sig: f"Signal: {sig.signal_id}",
            redis_client=None,
        )
        # Must not raise and must leave state empty
        await router.restore()
        assert router._active_signals == {}


# ---------------------------------------------------------------------------
# Fix 7: Position lock cleanup via cleanup_expired()
# ---------------------------------------------------------------------------


class TestCleanupExpired:
    """cleanup_expired() must remove stale signals and their position locks."""

    def test_cleanup_removes_expired_signal(self, router):
        """A signal older than its channel max hold must be removed."""
        sig = _make_signal(channel="360_SCALP")
        # Age the signal far beyond its 1-hour hold
        sig.timestamp = datetime.now(timezone.utc) - timedelta(hours=3)
        router._active_signals[sig.signal_id] = sig
        router._position_lock[sig.symbol] = sig.direction

        removed = router.cleanup_expired()
        assert removed == 1
        assert sig.signal_id not in router._active_signals

    def test_cleanup_clears_position_lock(self, router):
        """After cleanup, the position lock for the expired symbol is released."""
        sig = _make_signal(channel="360_SCALP", symbol="ETHUSDT")
        sig.timestamp = datetime.now(timezone.utc) - timedelta(hours=3)
        router._active_signals[sig.signal_id] = sig
        router._position_lock["ETHUSDT"] = sig.direction

        router.cleanup_expired()
        assert "ETHUSDT" not in router._position_lock

    def test_cleanup_sets_cooldown_on_expiry(self, router):
        """Expired signals must record a cooldown timestamp for re-entry suppression."""
        sig = _make_signal(channel="360_SCALP", symbol="SOLUSDT")
        sig.timestamp = datetime.now(timezone.utc) - timedelta(hours=3)
        router._active_signals[sig.signal_id] = sig
        router._position_lock["SOLUSDT"] = sig.direction

        router.cleanup_expired()
        assert ("SOLUSDT", "360_SCALP") in router._cooldown_timestamps

    def test_cleanup_does_not_remove_active_signal(self, router):
        """A fresh signal must not be removed by cleanup_expired."""
        sig = _make_signal(channel="360_SCALP", symbol="BNBUSDT")
        sig.timestamp = datetime.now(timezone.utc)  # just created
        router._active_signals[sig.signal_id] = sig
        router._position_lock["BNBUSDT"] = sig.direction

        removed = router.cleanup_expired()
        assert removed == 0
        assert sig.signal_id in router._active_signals

    def test_cleanup_returns_zero_on_empty_router(self, router):
        """cleanup_expired with no active signals must return 0."""
        assert router.cleanup_expired() == 0


# ---------------------------------------------------------------------------
# BUG 4: cleanup_expired is wired into start() loop
# ---------------------------------------------------------------------------


class TestStartLoopCallsCleanup:
    """start() must call cleanup_expired() periodically via the timeout path."""

    @pytest.mark.asyncio
    async def test_cleanup_called_on_timeout_counter_overflow(self, queue):
        """Simulate 60 timeout ticks – cleanup_expired must be called exactly once."""
        cleanup_calls = []

        async def mock_send(chat_id: str, text: str):
            return True

        router = SignalRouter(
            queue=queue,
            send_telegram=mock_send,
            format_signal=lambda sig: "",
        )
        # Monkey-patch cleanup_expired to record calls
        original = router.cleanup_expired

        def tracking_cleanup():
            result = original()
            cleanup_calls.append(True)
            return result

        router.cleanup_expired = tracking_cleanup

        # Seed one expired signal so cleanup has something to do
        sig = _make_signal(channel="360_SCALP")
        sig.timestamp = datetime.now(timezone.utc) - timedelta(hours=5)
        router._active_signals[sig.signal_id] = sig
        router._position_lock[sig.symbol] = sig.direction

        # Drive the start() loop with a very short timeout so it fires quickly.
        # We put None in the queue to create a fast timeout-style loop.
        # Instead, run the loop just long enough for at least 60 iterations.
        # We do this by draining timeouts: with timeout=1.0 that would need 60s.
        # Instead, we test the counter logic directly by patching asyncio.wait_for
        # to always raise TimeoutError — simulating 60 rapid timeout ticks.

        timeout_count = 0

        async def fast_wait_for(coro, timeout):
            nonlocal timeout_count
            timeout_count += 1
            # On the 60th tick, stop the router so the test finishes
            if timeout_count >= 60:
                router._running = False
            coro.close()
            raise asyncio.TimeoutError

        router._queue_has_timeout = False  # force the asyncio.wait_for code path
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(asyncio, "wait_for", fast_wait_for)

        try:
            await router.start()
        finally:
            monkeypatch.undo()

        # After 60 simulated timeout ticks, cleanup must have been called
        assert len(cleanup_calls) >= 1, "cleanup_expired was never called from the start() loop"
        # The expired signal must have been removed
        assert sig.signal_id not in router._active_signals


class TestPublishHighlight:
    """Tests for SignalRouter.publish_highlight() – rate limit and min TP."""

    @pytest.fixture
    def router_with_free(self, queue, sent_messages, monkeypatch):
        """Router with TELEGRAM_FREE_CHANNEL_ID configured."""
        import src.signal_router as m
        monkeypatch.setattr(m, "TELEGRAM_FREE_CHANNEL_ID", "free_channel")
        for channel in ("360_SCALP", "360_SPOT", "360_SWING", "360_GEM"):
            monkeypatch.setitem(m.CHANNEL_TELEGRAM_MAP, channel, "premium")

        async def mock_send(chat_id: str, text: str):
            sent_messages.append((chat_id, text))
            return True

        def mock_format(sig):
            return f"Signal: {sig.symbol}"

        return SignalRouter(queue=queue, send_telegram=mock_send, format_signal=mock_format)

    def _make_sig(self):
        return _make_signal()

    @pytest.mark.asyncio
    async def test_highlight_posted_to_free_channel(self, router_with_free, sent_messages):
        sig = self._make_sig()
        await router_with_free.publish_highlight(sig, 2, 0.62)
        assert any(chat_id == "free_channel" for chat_id, _ in sent_messages)

    @pytest.mark.asyncio
    async def test_highlight_skipped_for_tp1(self, router_with_free, sent_messages):
        sig = self._make_sig()
        await router_with_free.publish_highlight(sig, 1, 0.31)
        assert sent_messages == []

    @pytest.mark.asyncio
    async def test_highlight_rate_limit_respected(self, router_with_free, sent_messages):
        sig = self._make_sig()
        # Post 4 highlights (max)
        for _ in range(4):
            await router_with_free.publish_highlight(sig, 2, 0.62)
        # 5th should be blocked
        await router_with_free.publish_highlight(sig, 2, 0.62)
        free_msgs = [m for m in sent_messages if m[0] == "free_channel"]
        assert len(free_msgs) == 4

    @pytest.mark.asyncio
    async def test_highlight_daily_reset(self, router_with_free, sent_messages):
        import datetime as dt
        sig = self._make_sig()
        # Simulate yesterday's limit
        router_with_free._highlight_count_today = 4
        yesterday = dt.date.today() - dt.timedelta(days=1)
        router_with_free._highlight_date = yesterday

        # First post on new day should succeed
        await router_with_free.publish_highlight(sig, 2, 0.62)
        free_msgs = [m for m in sent_messages if m[0] == "free_channel"]
        assert len(free_msgs) == 1
        assert router_with_free._highlight_count_today == 1

    @pytest.mark.asyncio
    async def test_highlight_tp3_posted(self, router_with_free, sent_messages):
        sig = self._make_sig()
        await router_with_free.publish_highlight(sig, 3, 1.25)
        free_msgs = [m for m in sent_messages if m[0] == "free_channel"]
        assert len(free_msgs) == 1

    @pytest.mark.asyncio
    async def test_highlight_not_posted_when_no_free_channel_id(
        self, queue, sent_messages, monkeypatch
    ):
        import src.signal_router as m
        monkeypatch.setattr(m, "TELEGRAM_FREE_CHANNEL_ID", "")

        async def mock_send(chat_id, text):
            sent_messages.append((chat_id, text))
            return True

        r = SignalRouter(queue=queue, send_telegram=mock_send, format_signal=lambda s: "")
        sig = self._make_sig()
        await r.publish_highlight(sig, 2, 0.62)
        assert sent_messages == []

    @pytest.mark.asyncio
    async def test_highlight_message_contains_tp_level(self, router_with_free, sent_messages):
        sig = self._make_sig()
        await router_with_free.publish_highlight(sig, 2, 0.62)
        _, text = sent_messages[-1]
        assert "TP2" in text


class TestPublishDailyRecap:
    """Tests for SignalRouter.publish_daily_recap()."""

    @pytest.fixture
    def router_with_free(self, queue, sent_messages, monkeypatch):
        import src.signal_router as m
        monkeypatch.setattr(m, "TELEGRAM_FREE_CHANNEL_ID", "free_channel")
        for channel in ("360_SCALP", "360_SPOT", "360_SWING", "360_GEM"):
            monkeypatch.setitem(m.CHANNEL_TELEGRAM_MAP, channel, "premium")

        async def mock_send(chat_id, text):
            sent_messages.append((chat_id, text))
            return True

        return SignalRouter(queue=queue, send_telegram=mock_send, format_signal=lambda s: "")

    @pytest.mark.asyncio
    async def test_recap_skipped_when_no_trades(self, router_with_free, sent_messages):
        mock_tracker = MagicMock()
        mock_tracker.get_daily_summary.return_value = {
            "total": 0, "wins": 0, "losses": 0, "breakeven": 0,
            "win_rate": 0.0, "avg_pnl": 0.0, "best_trade": None, "top_trades": [],
        }
        await router_with_free.publish_daily_recap(mock_tracker)
        assert sent_messages == []

    @pytest.mark.asyncio
    async def test_recap_posted_to_free_channel(self, router_with_free, sent_messages):
        mock_tracker = MagicMock()
        mock_tracker.get_daily_summary.return_value = {
            "total": 5, "wins": 4, "losses": 1, "breakeven": 0,
            "win_rate": 80.0, "avg_pnl": 1.2, "best_trade": None, "top_trades": [],
        }
        await router_with_free.publish_daily_recap(mock_tracker)
        free_msgs = [m for m in sent_messages if m[0] == "free_channel"]
        assert len(free_msgs) == 1

    @pytest.mark.asyncio
    async def test_recap_contains_stats(self, router_with_free, sent_messages):
        mock_tracker = MagicMock()
        mock_tracker.get_daily_summary.return_value = {
            "total": 10, "wins": 7, "losses": 2, "breakeven": 1,
            "win_rate": 77.8, "avg_pnl": 1.5, "best_trade": None, "top_trades": [],
        }
        await router_with_free.publish_daily_recap(mock_tracker)
        _, text = sent_messages[-1]
        assert "10" in text
        assert "RECAP" in text

    @pytest.mark.asyncio
    async def test_recap_not_posted_when_no_free_channel_id(
        self, queue, sent_messages, monkeypatch
    ):
        import src.signal_router as m
        monkeypatch.setattr(m, "TELEGRAM_FREE_CHANNEL_ID", "")

        async def mock_send(chat_id, text):
            sent_messages.append((chat_id, text))
            return True

        r = SignalRouter(queue=queue, send_telegram=mock_send, format_signal=lambda s: "")
        mock_tracker = MagicMock()
        mock_tracker.get_daily_summary.return_value = {
            "total": 5, "wins": 4, "losses": 1, "breakeven": 0,
            "win_rate": 80.0, "avg_pnl": 1.2, "best_trade": None, "top_trades": [],
        }
        await r.publish_daily_recap(mock_tracker)
        assert sent_messages == []
