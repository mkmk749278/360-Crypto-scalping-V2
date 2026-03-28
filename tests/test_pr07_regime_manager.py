"""Tests for PR07 – Regime-Adaptive Signal Scheduler."""

from __future__ import annotations

import pytest

from src.scanner.regime_manager import RegimeChannelScheduler


ALL_CHANNELS = [
    "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD",
    "360_SCALP_VWAP", "360_SCALP_OBI",
    "360_SWING", "360_SPOT", "360_GEM",
]


class TestIsChannelAllowed:
    def test_vwap_blocked_in_quiet(self):
        sched = RegimeChannelScheduler()
        assert sched.is_channel_allowed("360_SCALP_VWAP", "QUIET") is False

    def test_vwap_allowed_in_ranging(self):
        sched = RegimeChannelScheduler()
        assert sched.is_channel_allowed("360_SCALP_VWAP", "RANGING") is True

    def test_swing_blocked_in_volatile(self):
        sched = RegimeChannelScheduler()
        assert sched.is_channel_allowed("360_SWING", "VOLATILE") is False

    def test_swing_allowed_in_trending(self):
        sched = RegimeChannelScheduler()
        assert sched.is_channel_allowed("360_SWING", "TRENDING_UP") is True

    def test_spot_blocked_in_dirty_range(self):
        sched = RegimeChannelScheduler()
        assert sched.is_channel_allowed("360_SPOT", "DIRTY_RANGE") is False

    def test_unknown_channel_allowed(self):
        sched = RegimeChannelScheduler()
        assert sched.is_channel_allowed("360_UNKNOWN", "QUIET") is True

    def test_case_insensitive_regime(self):
        sched = RegimeChannelScheduler()
        assert sched.is_channel_allowed("360_SCALP_VWAP", "quiet") is False

    def test_empty_regime_allows_all(self):
        sched = RegimeChannelScheduler()
        for ch in ALL_CHANNELS:
            # Empty regime should not block anything (no empty-string in maps)
            assert sched.is_channel_allowed(ch, "") is True


class TestGetAllowedChannels:
    def test_volatile_blocks_swing(self):
        sched = RegimeChannelScheduler()
        allowed = sched.get_allowed_channels("VOLATILE", ALL_CHANNELS)
        assert "360_SWING" not in allowed

    def test_quiet_blocks_vwap(self):
        sched = RegimeChannelScheduler()
        allowed = sched.get_allowed_channels("QUIET", ALL_CHANNELS)
        assert "360_SCALP_VWAP" not in allowed

    def test_trending_up_allows_all_scalp(self):
        sched = RegimeChannelScheduler()
        scalp_channels = [c for c in ALL_CHANNELS if "SCALP" in c]
        allowed = sched.get_allowed_channels("TRENDING_UP", scalp_channels)
        assert set(allowed) == set(scalp_channels)

    def test_returns_list(self):
        sched = RegimeChannelScheduler()
        result = sched.get_allowed_channels("RANGING", ALL_CHANNELS)
        assert isinstance(result, list)

    def test_empty_channel_list(self):
        sched = RegimeChannelScheduler()
        assert sched.get_allowed_channels("QUIET", []) == []

    def test_preserves_input_order(self):
        sched = RegimeChannelScheduler()
        channels = ["360_SCALP", "360_SCALP_FVG", "360_SCALP_OBI"]
        allowed = sched.get_allowed_channels("TRENDING_UP", channels)
        # Order must be preserved
        assert allowed == [c for c in channels if c in allowed]


class TestGetPriorityChannels:
    def test_trending_up_has_priorities(self):
        sched = RegimeChannelScheduler()
        prio = sched.get_priority_channels("TRENDING_UP")
        assert len(prio) > 0
        assert "360_SCALP" in prio

    def test_ranging_prioritises_mean_reversion(self):
        sched = RegimeChannelScheduler()
        prio = sched.get_priority_channels("RANGING")
        assert "360_SCALP_VWAP" in prio or "360_SCALP" in prio

    def test_volatile_prioritises_order_flow(self):
        sched = RegimeChannelScheduler()
        prio = sched.get_priority_channels("VOLATILE")
        assert "360_SCALP_OBI" in prio or "360_SCALP_CVD" in prio

    def test_unknown_regime_returns_list(self):
        sched = RegimeChannelScheduler()
        result = sched.get_priority_channels("UNKNOWN_REGIME")
        assert isinstance(result, list)

    def test_returns_list(self):
        sched = RegimeChannelScheduler()
        for regime in ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "QUIET", "VOLATILE"):
            prio = sched.get_priority_channels(regime)
            assert isinstance(prio, list)


class TestExtraBlocked:
    def test_extra_blocked_applied(self):
        sched = RegimeChannelScheduler(extra_blocked={"360_SCALP": ["QUIET"]})
        assert sched.is_channel_allowed("360_SCALP", "QUIET") is False
        assert sched.is_channel_allowed("360_SCALP", "RANGING") is True

    def test_extra_blocked_merges_with_defaults(self):
        sched = RegimeChannelScheduler(extra_blocked={"360_SCALP": ["QUIET"]})
        # Default VWAP/QUIET block still applies
        assert sched.is_channel_allowed("360_SCALP_VWAP", "QUIET") is False
        # New block also applies
        assert sched.is_channel_allowed("360_SCALP", "QUIET") is False


class TestLogSkippedPairs:
    def test_does_not_raise_for_empty(self):
        sched = RegimeChannelScheduler()
        sched.log_skipped_pairs([], "QUIET", "360_SCALP")

    def test_does_not_raise_for_many(self):
        sched = RegimeChannelScheduler()
        pairs = [f"TOKEN{i}USDT" for i in range(20)]
        sched.log_skipped_pairs(pairs, "QUIET", "360_SCALP")
