"""Tests for src.signal_router – queue-based signal routing."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from src.channels.base import Signal
from src.signal_router import SignalRouter
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
def router(queue, sent_messages):
    async def mock_send(chat_id: str, text: str):
        sent_messages.append((chat_id, text))

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
        tp1=32100,
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
