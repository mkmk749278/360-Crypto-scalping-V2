"""Tests for Phase 3: get_channel_scoreboard() and _format_scoreboard()."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.performance_tracker import PerformanceTracker
from src.signal_router import SignalRouter


def _make_tracker(tmp_path):
    return PerformanceTracker(storage_path=str(tmp_path / "perf.json"))


def _record(tracker, channel, hit_tp=1, hit_sl=False, pnl_pct=1.5, days_ago=0):
    """Helper to record an outcome with a timestamp offset."""
    record_time = time.time() - (days_ago * 86400)
    tracker.record_outcome(
        signal_id=f"SIG-{channel}-{record_time}",
        channel=channel,
        symbol="BTCUSDT",
        direction="LONG",
        entry=50000.0,
        hit_tp=hit_tp,
        hit_sl=hit_sl,
        pnl_pct=pnl_pct,
    )
    # Manually fix timestamp since record_outcome always sets current time
    tracker._records[-1].timestamp = record_time


class TestGetChannelScoreboard:
    def test_empty_records_returns_empty_dict(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        scoreboard = tracker.get_channel_scoreboard(window_days=7)
        assert scoreboard == {}

    def test_single_win_counted(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        _record(tracker, "360_SCALP", hit_tp=1, hit_sl=False, pnl_pct=2.0)
        scoreboard = tracker.get_channel_scoreboard(window_days=7)
        assert "360_SCALP" in scoreboard
        assert scoreboard["360_SCALP"]["wins"] == 1
        assert scoreboard["360_SCALP"]["losses"] == 0

    def test_single_loss_counted(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        _record(tracker, "360_SCALP", hit_tp=0, hit_sl=True, pnl_pct=-1.0)
        scoreboard = tracker.get_channel_scoreboard(window_days=7)
        assert scoreboard["360_SCALP"]["losses"] == 1
        assert scoreboard["360_SCALP"]["wins"] == 0

    def test_win_rate_calculation_accuracy(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        for _ in range(8):
            _record(tracker, "360_SWING", hit_tp=1, hit_sl=False, pnl_pct=2.0)
        for _ in range(2):
            _record(tracker, "360_SWING", hit_tp=0, hit_sl=True, pnl_pct=-1.0)
        scoreboard = tracker.get_channel_scoreboard(window_days=7)
        assert scoreboard["360_SWING"]["wins"] == 8
        assert scoreboard["360_SWING"]["losses"] == 2
        assert scoreboard["360_SWING"]["win_rate"] == 80.0

    def test_avg_pnl_calculation(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        _record(tracker, "360_SPOT", hit_tp=1, hit_sl=False, pnl_pct=3.0)
        _record(tracker, "360_SPOT", hit_tp=1, hit_sl=False, pnl_pct=1.0)
        scoreboard = tracker.get_channel_scoreboard(window_days=7)
        assert scoreboard["360_SPOT"]["avg_pnl"] == 2.0

    def test_records_outside_window_excluded(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        # Record 10 days ago (outside 7-day window)
        _record(tracker, "360_SCALP", hit_tp=1, hit_sl=False, pnl_pct=2.0, days_ago=10)
        scoreboard = tracker.get_channel_scoreboard(window_days=7)
        assert "360_SCALP" not in scoreboard

    def test_records_inside_window_included(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        _record(tracker, "360_SCALP", hit_tp=1, hit_sl=False, pnl_pct=2.0, days_ago=3)
        scoreboard = tracker.get_channel_scoreboard(window_days=7)
        assert "360_SCALP" in scoreboard
        assert scoreboard["360_SCALP"]["wins"] == 1

    def test_multiple_channels_tracked(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        _record(tracker, "360_SCALP", hit_tp=1, hit_sl=False, pnl_pct=1.5)
        _record(tracker, "360_SWING", hit_tp=0, hit_sl=True, pnl_pct=-1.0)
        _record(tracker, "360_SPOT", hit_tp=2, hit_sl=False, pnl_pct=5.0)
        scoreboard = tracker.get_channel_scoreboard(window_days=7)
        assert "360_SCALP" in scoreboard
        assert "360_SWING" in scoreboard
        assert "360_SPOT" in scoreboard

    def test_breakeven_counted_separately(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        _record(tracker, "360_SCALP", hit_tp=0, hit_sl=False, pnl_pct=0.0)
        scoreboard = tracker.get_channel_scoreboard(window_days=7)
        assert scoreboard["360_SCALP"]["breakeven"] == 1
        assert scoreboard["360_SCALP"]["wins"] == 0
        assert scoreboard["360_SCALP"]["losses"] == 0

    def test_win_rate_zero_when_no_wins_or_losses(self, tmp_path):
        """Only breakevens → win_rate is 0.0 (no wins or losses to divide by)."""
        tracker = _make_tracker(tmp_path)
        _record(tracker, "360_GEM", hit_tp=0, hit_sl=False, pnl_pct=0.0)
        scoreboard = tracker.get_channel_scoreboard(window_days=7)
        assert scoreboard["360_GEM"]["win_rate"] == 0.0


class TestFormatScoreboard:
    def _make_scoreboard(self):
        return {
            "360_SCALP": {"wins": 23, "losses": 4, "breakeven": 0, "win_rate": 85.2, "avg_pnl": 1.3, "total_pnl": 0.0, "count": 27},
            "360_SWING": {"wins": 8, "losses": 2, "breakeven": 0, "win_rate": 80.0, "avg_pnl": 3.8, "total_pnl": 0.0, "count": 10},
            "360_SPOT": {"wins": 5, "losses": 1, "breakeven": 0, "win_rate": 83.3, "avg_pnl": 6.2, "total_pnl": 0.0, "count": 6},
            "360_GEM": {"wins": 2, "losses": 0, "breakeven": 0, "win_rate": 100.0, "avg_pnl": 18.7, "total_pnl": 0.0, "count": 2},
        }

    def test_header_present(self):
        text = SignalRouter._format_scoreboard(self._make_scoreboard())
        assert "360 Crypto" in text
        assert "Weekly Performance" in text

    def test_channel_emojis_present(self):
        text = SignalRouter._format_scoreboard(self._make_scoreboard())
        assert "⚡" in text  # SCALP
        assert "🏛️" in text  # SWING
        assert "📈" in text  # SPOT
        assert "💎" in text  # GEM

    def test_wins_and_losses_shown(self):
        text = SignalRouter._format_scoreboard(self._make_scoreboard())
        assert "23W" in text
        assert "4L" in text

    def test_total_shown(self):
        text = SignalRouter._format_scoreboard(self._make_scoreboard())
        assert "Total:" in text
        assert "38W" in text
        assert "7L" in text

    def test_empty_scoreboard(self):
        text = SignalRouter._format_scoreboard({})
        assert "Total:" in text
        # No channels → 0W / 0L
        assert "0W" in text
        assert "0L" in text

    def test_avg_pnl_shown(self):
        text = SignalRouter._format_scoreboard(self._make_scoreboard())
        assert "+1.3%" in text

    def test_footer_present(self):
        text = SignalRouter._format_scoreboard(self._make_scoreboard())
        assert "premium" in text.lower() or "Premium" in text
        assert "Sunday" in text


class TestPublishScoreboard:
    @pytest.mark.asyncio
    async def test_does_nothing_when_no_free_channel(self, monkeypatch):
        monkeypatch.setattr("src.signal_router.TELEGRAM_FREE_CHANNEL_ID", "")
        q = MagicMock()
        q.get = AsyncMock(side_effect=Exception("should not be called"))
        q.get.__self__ = q
        router = SignalRouter(
            queue=q,
            send_telegram=AsyncMock(),
            format_signal=MagicMock(),
        )
        tracker = MagicMock()
        tracker.get_channel_scoreboard.return_value = {"360_SCALP": {"wins": 1, "losses": 0}}
        # Should silently return without posting
        await router.publish_scoreboard(tracker)
        router._send_telegram.assert_not_called()

    @pytest.mark.asyncio
    async def test_posts_to_free_channel(self, monkeypatch):
        monkeypatch.setattr("src.signal_router.TELEGRAM_FREE_CHANNEL_ID", "free_id")
        q = MagicMock()
        router = SignalRouter(
            queue=q,
            send_telegram=AsyncMock(return_value=True),
            format_signal=MagicMock(),
        )
        tracker = MagicMock()
        tracker.get_channel_scoreboard.return_value = {
            "360_SCALP": {"wins": 5, "losses": 1, "breakeven": 0, "win_rate": 83.3, "avg_pnl": 1.2, "total_pnl": 6.0, "count": 6},
        }
        await router.publish_scoreboard(tracker)
        router._send_telegram.assert_called_once()
        call_args = router._send_telegram.call_args[0]
        assert call_args[0] == "free_id"
        assert "Scalp" in call_args[1] or "SCALP" in call_args[1]
