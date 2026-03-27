"""Tests for src.rate_limiter — RateLimiter class.

Covers:
- Weight tracking and accumulation via acquire()
- Auto-reset after the 60-second window elapses
- acquire() blocks (suspends) when the budget is exhausted, then resumes
- update_from_header() syncs weight from server-reported values
- set_budget() dynamically adjusts the budget
- Separate spot/futures limiter singletons are distinct instances
- Pre-filter logic (_prefilter_pairs) reduces the symbol set
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from src.rate_limiter import RateLimiter, spot_rate_limiter, futures_rate_limiter, rate_limiter


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
        await rl.acquire(3)  # blocks until reset (sleep happens outside lock)
        # After the blocking acquire's sleep completes (~0.1s), the window
        # resets again when acquire(2) calls _maybe_reset(), so used == 2.
        t0 = time.monotonic()
        await rl.acquire(2)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, f"Expected fast acquire, got elapsed={elapsed:.3f}s"
        assert rl.used == 2


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
# set_budget()
# ---------------------------------------------------------------------------

class TestSetBudget:
    """set_budget() dynamically adjusts the weight budget."""

    def test_set_budget_changes_budget_property(self):
        rl = _make_limiter(budget=1000)
        rl.set_budget(1100)
        assert rl.budget == 1100

    def test_set_budget_allows_higher_acquire(self):
        """After raising budget, previously exhausting acquire should succeed."""
        rl = _make_limiter(budget=100)
        rl._used = 90
        # With the old budget, acquiring 20 would block; after raising it won't.
        rl.set_budget(200)
        assert rl.remaining >= 110  # 200 - 90

    def test_set_budget_affects_remaining(self):
        rl = _make_limiter(budget=500)
        rl._used = 300
        rl.set_budget(1000)
        assert rl.remaining == 700

    def test_set_budget_lower_reduces_remaining(self):
        rl = _make_limiter(budget=1000)
        rl._used = 0
        rl.set_budget(500)
        assert rl.budget == 500
        assert rl.remaining == 500


# ---------------------------------------------------------------------------
# Separate spot / futures limiter singletons
# ---------------------------------------------------------------------------

class TestSeparateLimiters:
    """spot_rate_limiter and futures_rate_limiter are distinct instances."""

    def test_spot_and_futures_are_separate_objects(self):
        assert spot_rate_limiter is not futures_rate_limiter

    def test_rate_limiter_alias_points_to_spot(self):
        """Backward-compatible `rate_limiter` should be the spot limiter."""
        assert rate_limiter is spot_rate_limiter

    def test_spot_limiter_budget_change_does_not_affect_futures(self):
        original_spot_budget = spot_rate_limiter.budget
        original_futures_budget = futures_rate_limiter.budget
        try:
            spot_rate_limiter.set_budget(999)
            # Futures budget must be unchanged
            assert futures_rate_limiter.budget == original_futures_budget
        finally:
            # Restore original budgets so other tests aren't affected
            spot_rate_limiter.set_budget(original_spot_budget)

    def test_binance_client_spot_uses_spot_limiter(self):
        from src.binance import BinanceClient
        client = BinanceClient("spot")
        assert client._rate_limiter is spot_rate_limiter

    def test_binance_client_futures_uses_futures_limiter(self):
        from src.binance import BinanceClient
        client = BinanceClient("futures")
        assert client._rate_limiter is futures_rate_limiter


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


# ---------------------------------------------------------------------------
# Burst protection
# ---------------------------------------------------------------------------

class TestBurstProtection:
    """acquire() adds a micro-sleep when budget is nearly exhausted."""

    @pytest.mark.asyncio
    async def test_no_burst_sleep_on_fresh_budget(self):
        """No extra delay when the budget has plenty of headroom."""
        rl = _make_limiter(budget=1000, window_s=60.0)
        # Consume 50% — well above the 15% threshold
        rl._used = 500
        t0 = time.monotonic()
        await rl.acquire(1)
        elapsed = time.monotonic() - t0
        # Should complete quickly (< 50 ms) — no burst sleep triggered
        assert elapsed < 0.05, f"Unexpected delay: {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_burst_sleep_triggered_near_exhaustion(self):
        """A micro-sleep is injected when remaining budget < 15%."""
        rl = _make_limiter(budget=1000, window_s=60.0)
        # Fill to 90% (100 remaining = 10%, below the 15% threshold)
        rl._used = 900
        t0 = time.monotonic()
        await rl.acquire(1)
        elapsed = time.monotonic() - t0
        # Should have slept at least a small amount
        assert elapsed >= 0.05, f"Expected burst-protection sleep, got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_burst_sleep_increases_near_zero(self):
        """Sleep grows as remaining budget approaches zero."""
        rl_low = _make_limiter(budget=1000, window_s=60.0)
        rl_very_low = _make_limiter(budget=1000, window_s=60.0)

        rl_low._used = 860   # 14% remaining — just inside the threshold
        rl_very_low._used = 990  # 1% remaining — near zero

        t0 = time.monotonic()
        await rl_low.acquire(1)
        t_low = time.monotonic() - t0

        t0 = time.monotonic()
        await rl_very_low.acquire(1)
        t_very_low = time.monotonic() - t0

        # Deeper into burst territory means longer sleep
        assert t_very_low > t_low, (
            f"Expected t_very_low ({t_very_low:.3f}s) > t_low ({t_low:.3f}s)"
        )

    @pytest.mark.asyncio
    async def test_burst_protection_does_not_exceed_max_sleep(self):
        """The injected sleep never exceeds _BURST_PROTECTION_MAX_SLEEP_S."""
        from src.rate_limiter import _BURST_PROTECTION_MAX_SLEEP_S
        rl = _make_limiter(budget=1000, window_s=60.0)
        rl._used = 999  # 0.1% remaining

        t0 = time.monotonic()
        await rl.acquire(1)
        elapsed = time.monotonic() - t0

        # Allow a small margin for scheduling overhead
        assert elapsed < _BURST_PROTECTION_MAX_SLEEP_S + 0.1, (
            f"Sleep exceeded maximum: {elapsed:.3f}s > {_BURST_PROTECTION_MAX_SLEEP_S + 0.1:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_full_budget_acquire_no_burst_sleep(self):
        """Consuming exactly the full budget in one call must not trigger burst sleep."""
        rl = _make_limiter(budget=10, window_s=60.0)
        # _used=0 → remaining_before=10 (100%), well above 15% threshold
        t0 = time.monotonic()
        await rl.acquire(10)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, f"Unexpected burst sleep on full-budget acquire: {elapsed:.3f}s"
        assert rl.used == 10


# ---------------------------------------------------------------------------
# Tier-pause properties (PR 4 — Intelligent Preemptive Rate Limiting)
# ---------------------------------------------------------------------------

class TestTierPauseProperties:
    """is_tier3_paused and is_tier2_paused reflect preemptive throttle thresholds."""

    def test_not_paused_at_low_usage(self):
        """Below 70% — neither tier is paused."""
        rl = _make_limiter(budget=100)
        rl._used = 50  # 50%
        assert not rl.is_tier3_paused
        assert not rl.is_tier2_paused

    def test_tier3_paused_at_exactly_70_pct(self):
        """At exactly 70% usage, Tier 3 is paused but Tier 2 is not."""
        rl = _make_limiter(budget=100)
        rl._used = 70  # exactly 70%
        assert rl.is_tier3_paused
        assert not rl.is_tier2_paused

    def test_tier3_paused_above_70_pct(self):
        """Above 70% — Tier 3 is paused."""
        rl = _make_limiter(budget=100)
        rl._used = 75  # 75%
        assert rl.is_tier3_paused

    def test_tier2_paused_at_exactly_85_pct(self):
        """At exactly 85% usage, both Tier 2 and Tier 3 are paused."""
        rl = _make_limiter(budget=100)
        rl._used = 85  # exactly 85%
        assert rl.is_tier2_paused
        assert rl.is_tier3_paused  # 85% > 70%

    def test_both_paused_above_85_pct(self):
        """Above 85% — both Tier 2 and Tier 3 are paused."""
        rl = _make_limiter(budget=100)
        rl._used = 90  # 90%
        assert rl.is_tier3_paused
        assert rl.is_tier2_paused

    def test_not_paused_just_below_70_pct(self):
        """Just below 70% — neither tier is paused."""
        rl = _make_limiter(budget=100)
        rl._used = 69  # 69%
        assert not rl.is_tier3_paused
        assert not rl.is_tier2_paused

    def test_tier3_only_between_70_and_85_pct(self):
        """Between 70% and 85% — only Tier 3 is paused, Tier 2 is still active."""
        rl = _make_limiter(budget=100)
        rl._used = 80  # 80% — above Tier 3 threshold, below Tier 2 threshold
        assert rl.is_tier3_paused
        assert not rl.is_tier2_paused

    def test_pauses_reset_after_window_expires(self):
        """Pauses clear automatically when the rate-limit window resets."""
        rl = _make_limiter(budget=100, window_s=0.01)
        rl._used = 90  # 90% — both paused
        # Expire the window
        rl._window_start = time.monotonic() - 1.0
        # Properties call _maybe_reset() internally
        assert not rl.is_tier3_paused
        assert not rl.is_tier2_paused

    def test_pauses_work_with_large_budget(self):
        """Thresholds are fractional so they scale with any budget size."""
        rl = _make_limiter(budget=5000)
        rl._used = 3500  # 70% of 5000 = 3500
        assert rl.is_tier3_paused
        assert not rl.is_tier2_paused

        rl._used = 4250  # 85% of 5000 = 4250
        assert rl.is_tier2_paused
