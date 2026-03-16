"""Tests for PerformanceTracker – recording and stats computation."""

from __future__ import annotations

import json
import time

import pytest

from src.performance_tracker import PerformanceTracker


class TestPerformanceTrackerRecording:
    def test_records_outcome(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome(
            signal_id="SIG001",
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction="LONG",
            entry=50000.0,
            hit_tp=1,
            hit_sl=False,
            pnl_pct=1.5,
        )
        assert len(pt._records) == 1
        assert pt._records[0].signal_id == "SIG001"

    def test_persists_to_file(self, tmp_path):
        path = tmp_path / "perf.json"
        pt = PerformanceTracker(storage_path=str(path))
        pt.record_outcome(
            "S1",
            "360_SCALP",
            "BTCUSDT",
            "LONG",
            50000,
            1,
            False,
            1.5,
            pre_ai_confidence=78.0,
            post_ai_confidence=82.0,
            setup_class="BREAKOUT_RETEST",
            market_phase="STRONG_TREND",
            quality_tier="A",
            spread_pct=0.008,
            volume_24h_usd=15_000_000.0,
            hold_duration_sec=3600.0,
        )
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["signal_id"] == "S1"
        assert data[0]["setup_class"] == "BREAKOUT_RETEST"
        assert data[0]["quality_tier"] == "A"

    def test_loads_from_file(self, tmp_path):
        path = tmp_path / "perf.json"
        pt1 = PerformanceTracker(storage_path=str(path))
        pt1.record_outcome("S1", "360_SCALP", "BTCUSDT", "LONG", 50000, 1, False, 1.5)

        # Load in new instance
        pt2 = PerformanceTracker(storage_path=str(path))
        assert len(pt2._records) == 1
        assert pt2._records[0].signal_id == "S1"

    def test_multiple_records(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        for i in range(5):
            pt.record_outcome(
                f"SIG{i}", "360_SCALP", "BTCUSDT", "LONG", 50000, 1, False, float(i)
            )
        assert len(pt._records) == 5


class TestPerformanceTrackerStats:
    def _make_tracker(self, tmp_path, records):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        for r in records:
            pt.record_outcome(*r)
        return pt

    def test_win_rate_all_wins(self, tmp_path):
        records = [
            ("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 2.0),
            ("S2", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 1.5),
        ]
        pt = self._make_tracker(tmp_path, records)
        stats = pt.get_stats(channel="360_SCALP")
        assert stats.win_rate == 100.0

    def test_win_rate_mixed(self, tmp_path):
        records = [
            ("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 2.0),
            ("S2", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -1.0),
            ("S3", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 1.5),
            ("S4", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -1.0),
        ]
        pt = self._make_tracker(tmp_path, records)
        stats = pt.get_stats(channel="360_SCALP")
        assert abs(stats.win_rate - 50.0) < 0.01

    def test_avg_pnl(self, tmp_path):
        records = [
            ("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 3.0),
            ("S2", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -1.0),
        ]
        pt = self._make_tracker(tmp_path, records)
        stats = pt.get_stats(channel="360_SCALP")
        assert abs(stats.avg_pnl_pct - 1.0) < 0.01

    def test_best_worst_trade(self, tmp_path):
        records = [
            ("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 5.0),
            ("S2", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -2.0),
            ("S3", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 1.0),
        ]
        pt = self._make_tracker(tmp_path, records)
        stats = pt.get_stats(channel="360_SCALP")
        assert stats.best_trade == 5.0
        assert stats.worst_trade == -2.0

    def test_max_drawdown_computed(self, tmp_path):
        records = [
            ("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 5.0),
            ("S2", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -3.0),
            ("S3", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -3.0),
        ]
        pt = self._make_tracker(tmp_path, records)
        stats = pt.get_stats(channel="360_SCALP")
        # Equity curve: 1.00 -> 1.05 -> 1.0185 -> 0.987945, so max drawdown
        # is (1.05 - 0.987945) / 1.05 = 5.91%.
        assert stats.max_drawdown == pytest.approx(5.91, abs=0.01)

    def test_break_even_exit_is_not_counted_as_loss(self, tmp_path):
        records = [
            ("S1", "360_SCALP", "BTC", "LONG", 100.0, 0, True, 0.0),
            ("S2", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 2.0),
        ]
        pt = self._make_tracker(tmp_path, records)
        stats = pt.get_stats(channel="360_SCALP")
        assert stats.win_count == 1
        assert stats.loss_count == 0
        assert stats.breakeven_count == 1

    def test_stats_keep_semantic_counts_consistent(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome("L1", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -1.0)
        pt.record_outcome("B1", "360_SCALP", "BTC", "LONG", 100.0, 0, True, 0.0)
        pt.record_outcome("P1", "360_SCALP", "BTC", "LONG", 100.0, 0, True, 0.6)
        pt.record_outcome("T1", "360_SCALP", "BTC", "LONG", 100.0, 3, False, 1.5)

        stats = pt.get_stats(channel="360_SCALP")
        assert stats.total_signals == 4
        assert stats.win_count == 2
        assert stats.loss_count == 1
        assert stats.breakeven_count == 1
        # 2 wins (P1, T1) / 3 non-breakeven trades (L1, P1, T1) = 66.67%
        # because breakeven exits are excluded from the win-rate denominator.
        assert stats.win_rate == pytest.approx(66.67, abs=0.01)

    def test_unrealistic_losses_are_clamped(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome("S1", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -150.0)
        stats = pt.get_stats(channel="360_SCALP")
        assert stats.worst_trade == -99.99
        assert stats.max_drawdown == pytest.approx(99.99, abs=0.01)

    def test_no_records_returns_zero_stats(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        stats = pt.get_stats(channel="360_SCALP")
        assert stats.total_signals == 0
        assert stats.win_rate == 0.0

    def test_channel_filter(self, tmp_path):
        records = [
            ("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 2.0),
            ("S2", "360_SWING", "ETH", "LONG", 200.0, 1, False, 3.0),
        ]
        pt = self._make_tracker(tmp_path, records)
        scalp_stats = pt.get_stats(channel="360_SCALP")
        swing_stats = pt.get_stats(channel="360_SWING")
        assert scalp_stats.total_signals == 1
        assert swing_stats.total_signals == 1

    def test_rolling_window_filter(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        # Add one recent record
        pt.record_outcome("new", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 2.0)
        # Inject one old record (31 days ago)
        from src.performance_tracker import SignalRecord
        old = SignalRecord(
            signal_id="old",
            channel="360_SCALP",
            symbol="BTC",
            direction="LONG",
            entry=100.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-1.0,
            confidence=50.0,
            timestamp=time.time() - 31 * 86400,
        )
        pt._records.insert(0, old)

        stats_7d = pt.get_stats(channel="360_SCALP", window_days=7)
        assert stats_7d.total_signals == 1  # only recent one


class TestPerformanceTrackerFormatting:
    def test_format_message_contains_key_fields(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 2.0)
        msg = pt.format_stats_message(channel="360_SCALP")
        assert "Win rate" in msg
        assert "Total signals" in msg
        assert "Avg PnL" in msg
        assert "Max drawdown" in msg
        assert "Breakeven" in msg

    def test_format_message_all_channels(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        msg = pt.format_stats_message()
        assert "All Channels" in msg


class TestPerformanceTrackerAnalyticsFields:
    def test_records_extended_analytics_fields(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome(
            signal_id="SIGA",
            channel="360_SELECT",
            symbol="BTCUSDT",
            direction="LONG",
            entry=100.0,
            hit_tp=3,
            hit_sl=False,
            pnl_pct=4.2,
            confidence=91.0,
            pre_ai_confidence=88.0,
            post_ai_confidence=91.0,
            setup_class="TREND_PULLBACK_CONTINUATION",
            market_phase="STRONG_TREND",
            quality_tier="A+",
            spread_pct=0.007,
            volume_24h_usd=22_000_000.0,
            hold_duration_sec=5400.0,
            max_favorable_excursion_pct=5.0,
            max_adverse_excursion_pct=-0.8,
        )
        record = pt._records[0]
        assert record.pre_ai_confidence == 88.0
        assert record.post_ai_confidence == 91.0
        assert record.outcome_label == "FULL_TP_HIT"
        assert record.setup_class == "TREND_PULLBACK_CONTINUATION"
        assert record.market_phase == "STRONG_TREND"
        assert record.quality_tier == "A+"
        assert record.hold_duration_sec == 5400.0
        assert record.max_favorable_excursion_pct == 5.0
        assert record.max_adverse_excursion_pct == -0.8

    def test_profit_lock_and_breakeven_outcomes_are_classified(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome("B1", "360_SCALP", "BTC", "LONG", 100.0, 0, True, 0.0)
        pt.record_outcome("P1", "360_SCALP", "BTC", "LONG", 100.0, 0, True, 0.4)
        pt.record_outcome("L1", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -0.4)

        assert [record.outcome_label for record in pt._records] == [
            "BREAKEVEN_EXIT",
            "PROFIT_LOCKED",
            "SL_HIT",
        ]
