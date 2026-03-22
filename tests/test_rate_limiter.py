"""Tests for src.rate_limiter — RateLimiter class.

Covers:
- Weight tracking and accumulation via acquire()
- Auto-reset after the 60-second window elapses
- acquire() blocks (suspends) when the budget is exhausted, then resumes
- update_from_header() syncs weight from server-reported values
- Pre-filter logic (_prefilter_pairs) reduces the symbol set
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from src.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_limiter(budget: int = 100, window_s: float = 60.0) -> RateLimiter:
    """Return a fresh RateLimiter with the given budget and window."""
    return RateLimiter(budget=budget, window_s=window_s)


# ---------------------------------------------------------------------------
# Basic weight tracking
# ---------------------------------------------------------------------------

class TestWeightTracking:
    """acquire() correctly accumulates weight."""

    def test_initial_state(self):
        rl = _make_limiter(budget=100)
        assert rl.used == 0
        assert rl.remaining == 100

    @pytest.mark.asyncio
    async def test_acquire_single(self):
        rl = _make_limiter(budget=100)
        await rl.acquire(10)
        assert rl.used == 10
        assert rl.remaining == 90

    @pytest.mark.asyncio
    async def test_acquire_multiple(self):
        rl = _make_limiter(budget=100)
        await rl.acquire(20)
        await rl.acquire(30)
        assert rl.used == 50
        assert rl.remaining == 50

    @pytest.mark.asyncio
    async def test_acquire_exact_budget(self):
        """Consuming exactly the budget should not block."""
        rl = _make_limiter(budget=50)
        await rl.acquire(25)
        await rl.acquire(25)
        assert rl.used == 50
        assert rl.remaining == 0

    @pytest.mark.asyncio
    async def test_default_weight_is_one(self):
        rl = _make_limiter(budget=100)
        await rl.acquire()
        assert rl.used == 1


# ---------------------------------------------------------------------------
# Auto-reset
# ---------------------------------------------------------------------------

class TestAutoReset:
    """Weight counter resets when the rolling window elapses."""

    @pytest.mark.asyncio
    async def test_reset_after_window(self):
        rl = _make_limiter(budget=100, window_s=0.05)  # very short window
        await rl.acquire(80)
        assert rl.used == 80
        # Wait for the window to expire
        await asyncio.sleep(0.1)
        # remaining triggers _maybe_reset()
        assert rl.remaining == 100

    @pytest.mark.asyncio
    async def test_acquire_after_reset(self):
        rl = _make_limiter(budget=50, window_s=0.05)
        await rl.acquire(40)
        await asyncio.sleep(0.1)
        # Should succeed after the window resets
        await rl.acquire(40)
        assert rl.used == 40

    def test_remaining_triggers_reset(self):
        rl = _make_limiter(budget=100, window_s=0.01)
        # Manually advance past the window by setting the start time far back
        rl._window_start = time.monotonic() - 1.0
        rl._used = 75
        # remaining should detect the stale window and reset
        assert rl.remaining == 100


# ---------------------------------------------------------------------------
# acquire() blocking behaviour
# ---------------------------------------------------------------------------

class TestAcquireBlocking:
    """acquire() suspends when the budget is exhausted and resumes after reset."""

    @pytest.mark.asyncio
    async def test_acquire_blocks_until_reset(self):
        """When budget is exhausted, acquire() waits for window reset."""
        rl = _make_limiter(budget=10, window_s=0.1)
        await rl.acquire(10)  # drain budget
        assert rl.remaining == 0

        t0 = time.monotonic()
        # This should block until the window resets (~0.1 s)
        await rl.acquire(5)
        elapsed = time.monotonic() - t0
        # Should have waited at least a short time for reset
        assert elapsed >= 0.05, f"Expected blocking, got elapsed={elapsed:.3f}s"
        assert rl.used == 5

    @pytest.mark.asyncio
    async def test_second_acquire_after_exhaustion(self):
        """After blocking and reset, subsequent acquires proceed immediately."""
        rl = _make_limiter(budget=10, window_s=0.1)
        await rl.acquire(10)
        await rl.acquire(3)  # blocks until reset
        # After reset, another small acquire should be instant
        t0 = time.monotonic()
        await rl.acquire(2)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, f"Expected fast acquire, got elapsed={elapsed:.3f}s"
        assert rl.used == 5


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

class TestUpdateFromHeader:
    """update_from_header() syncs weight from server responses."""

    def test_updates_used_weight_from_header(self):
        rl = _make_limiter(budget=1000)
        rl.update_from_header("42")
        assert rl.used == 42

    def test_server_value_wins_when_higher(self):
        """Server-reported weight overrides a lower local estimate."""
        rl = _make_limiter(budget=1000)
        rl._used = 10
        rl.update_from_header("50")
        assert rl.used == 50

    def test_local_value_kept_when_server_is_lower(self):
        """When local estimate is already higher, it is preserved."""
        rl = _make_limiter(budget=1000)
        rl._used = 80
        rl.update_from_header("20")
        assert rl.used == 80

    def test_none_header_is_noop(self):
        rl = _make_limiter(budget=1000)
        rl._used = 30
        rl.update_from_header(None)
        assert rl.used == 30

    def test_invalid_header_is_noop(self):
        rl = _make_limiter(budget=1000)
        rl._used = 30
        rl.update_from_header("not-a-number")
        assert rl.used == 30

    def test_zero_header_does_not_reset_local(self):
        """A header of '0' should not override a non-zero local estimate."""
        rl = _make_limiter(budget=1000)
        rl._used = 50
        rl.update_from_header("0")
        assert rl.used == 50

    def test_warning_logged_at_threshold(self):
        """update_from_header does not raise when usage is at warning threshold."""
        rl = _make_limiter(budget=100)
        # Should not raise; logging happens internally
        rl.update_from_header("85")
        assert rl.used == 85


# ---------------------------------------------------------------------------
# Pre-filter logic (Scanner._prefilter_pairs)
# ---------------------------------------------------------------------------

class TestPrefilterPairs:
    """_prefilter_pairs removes low-volume / all-active / all-cooldown symbols."""

    def _make_scanner(self, channel_names=None):
        """Build a minimal Scanner-like object with the _prefilter_pairs method."""
        # Import the real Scanner so we test the actual method
        from src.scanner import Scanner

        scanner = object.__new__(Scanner)

        # Wire stub channels
        if channel_names is None:
            channel_names = ["360_SCALP", "360_SWING"]

        fake_channels = []
        for name in channel_names:
            ch = MagicMock()
            ch.config.name = name
            fake_channels.append(ch)
        scanner.channels = fake_channels

        # Stub router with no active signals
        scanner.router = MagicMock()
        scanner.router.active_signals = {}

        # Stub cooldown dict
        scanner._cooldown_until = {}

        return scanner

    def _make_pair(self, symbol: str, volume: float):
        info = MagicMock()
        info.volume_24h_usd = volume
        return symbol, info

    def test_all_pass_above_volume_threshold(self):
        from config import SCAN_MIN_VOLUME_USD
        scanner = self._make_scanner()
        pairs = [
            self._make_pair("BTCUSDT", SCAN_MIN_VOLUME_USD + 1),
            self._make_pair("ETHUSDT", SCAN_MIN_VOLUME_USD + 1),
        ]
        result = scanner._prefilter_pairs(pairs)
        assert len(result) == 2

    def test_low_volume_symbol_removed(self):
        from config import SCAN_MIN_VOLUME_USD
        scanner = self._make_scanner()
        pairs = [
            self._make_pair("HIGHVOL", SCAN_MIN_VOLUME_USD + 1),
            self._make_pair("LOWVOL", SCAN_MIN_VOLUME_USD - 1),
        ]
        result = scanner._prefilter_pairs(pairs)
        assert len(result) == 1
        assert result[0][0] == "HIGHVOL"

    def test_all_low_volume_returns_empty(self):
        scanner = self._make_scanner()
        pairs = [
            self._make_pair("A", 100),
            self._make_pair("B", 500),
        ]
        result = scanner._prefilter_pairs(pairs)
        assert result == []

    def test_symbol_with_active_signals_on_all_channels_removed(self):
        from config import SCAN_MIN_VOLUME_USD
        scanner = self._make_scanner(channel_names=["360_SCALP", "360_SWING"])

        # Create active signal objects for BTCUSDT on both channels
        sig_scalp = MagicMock()
        sig_scalp.symbol = "BTCUSDT"
        sig_scalp.channel = "360_SCALP"
        sig_swing = MagicMock()
        sig_swing.symbol = "BTCUSDT"
        sig_swing.channel = "360_SWING"
        scanner.router.active_signals = {"s1": sig_scalp, "s2": sig_swing}

        pairs = [
            self._make_pair("BTCUSDT", SCAN_MIN_VOLUME_USD + 1),
            self._make_pair("ETHUSDT", SCAN_MIN_VOLUME_USD + 1),
        ]
        result = scanner._prefilter_pairs(pairs)
        # BTCUSDT should be filtered (both channels active), ETHUSDT passes
        symbols = [s for s, _ in result]
        assert "BTCUSDT" not in symbols
        assert "ETHUSDT" in symbols

    def test_symbol_with_active_signal_on_only_one_channel_kept(self):
        from config import SCAN_MIN_VOLUME_USD
        scanner = self._make_scanner(channel_names=["360_SCALP", "360_SWING"])

        sig_scalp = MagicMock()
        sig_scalp.symbol = "BTCUSDT"
        sig_scalp.channel = "360_SCALP"
        scanner.router.active_signals = {"s1": sig_scalp}

        pairs = [self._make_pair("BTCUSDT", SCAN_MIN_VOLUME_USD + 1)]
        result = scanner._prefilter_pairs(pairs)
        # Only one channel is active; the other could still fire → keep symbol
        assert len(result) == 1

    def test_symbol_fully_in_cooldown_removed(self):
        from config import SCAN_MIN_VOLUME_USD
        scanner = self._make_scanner(channel_names=["360_SCALP", "360_SWING"])
        # Put XRPUSDT in cooldown for both channels
        far_future = time.monotonic() + 9999
        scanner._cooldown_until = {
            ("XRPUSDT", "360_SCALP"): far_future,
            ("XRPUSDT", "360_SWING"): far_future,
        }
        pairs = [
            self._make_pair("XRPUSDT", SCAN_MIN_VOLUME_USD + 1),
            self._make_pair("ETHUSDT", SCAN_MIN_VOLUME_USD + 1),
        ]
        result = scanner._prefilter_pairs(pairs)
        symbols = [s for s, _ in result]
        assert "XRPUSDT" not in symbols
        assert "ETHUSDT" in symbols

    def test_symbol_in_cooldown_on_one_channel_kept(self):
        from config import SCAN_MIN_VOLUME_USD
        scanner = self._make_scanner(channel_names=["360_SCALP", "360_SWING"])
        far_future = time.monotonic() + 9999
        scanner._cooldown_until = {("BTCUSDT", "360_SCALP"): far_future}
        pairs = [self._make_pair("BTCUSDT", SCAN_MIN_VOLUME_USD + 1)]
        result = scanner._prefilter_pairs(pairs)
        assert len(result) == 1

    def test_prefilter_significantly_reduces_symbol_count(self):
        """Simulate 200 pairs where most are low-volume → large reduction."""
        from config import SCAN_MIN_VOLUME_USD
        scanner = self._make_scanner()
        # 180 low-volume, 20 high-volume
        pairs = (
            [self._make_pair(f"LOW{i}", 1000) for i in range(180)]
            + [self._make_pair(f"HIGH{i}", SCAN_MIN_VOLUME_USD + 1) for i in range(20)]
        )
        result = scanner._prefilter_pairs(pairs)
        assert len(result) == 20
        # Reduction > 80%
        assert len(result) / len(pairs) < 0.15  # expect ≤15% of pairs to pass through
