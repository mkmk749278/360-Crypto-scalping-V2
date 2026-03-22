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
            channel="360_GEM",
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


class TestPerformanceTrackerReset:
    def test_reset_all(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 2.0)
        pt.record_outcome("S2", "360_SWING", "ETH", "LONG", 200.0, 1, False, 3.0)
        assert pt.get_stats().total_signals == 2
        cleared = pt.reset_stats()
        assert cleared == 2
        assert pt.get_stats().total_signals == 0

    def test_reset_specific_channel(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 2.0)
        pt.record_outcome("S2", "360_SWING", "ETH", "LONG", 200.0, 1, False, 3.0)
        cleared = pt.reset_stats(channel="360_SCALP")
        assert cleared == 1
        stats = pt.get_stats()
        assert stats.total_signals == 1  # only 360_SWING remains

    def test_reset_persists_to_disk(self, tmp_path):
        path = str(tmp_path / "perf.json")
        pt = PerformanceTracker(storage_path=path)
        pt.record_outcome("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 2.0)
        pt.reset_stats()
        # Reload from disk
        pt2 = PerformanceTracker(storage_path=path)
        assert pt2.get_stats().total_signals == 0

    def test_reset_empty_tracker(self, tmp_path):
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        cleared = pt.reset_stats()
        assert cleared == 0


class TestSignalQualityStats:
    """Tests for signal quality (TP-based) PnL tracking."""

    def test_signal_quality_fields_stored(self, tmp_path):
        """signal_quality_pnl_pct and signal_quality_hit_tp are stored correctly."""
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome(
            signal_id="S1",
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction="LONG",
            entry=100.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-0.5,
            signal_quality_pnl_pct=0.5,
            signal_quality_hit_tp=1,
        )
        record = pt._records[0]
        assert record.pnl_pct == pytest.approx(-0.5)
        assert record.signal_quality_pnl_pct == pytest.approx(0.5)
        assert record.signal_quality_hit_tp == 1

    def test_signal_quality_defaults_to_actual_pnl_when_not_provided(self, tmp_path):
        """When signal_quality_pnl_pct is not provided, it defaults to pnl_pct."""
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome(
            signal_id="S1",
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction="LONG",
            entry=100.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-1.5,
        )
        record = pt._records[0]
        assert record.pnl_pct == pytest.approx(-1.5)
        assert record.signal_quality_pnl_pct == pytest.approx(-1.5)
        assert record.signal_quality_hit_tp == 0

    def test_signal_quality_stats_uses_sq_pnl(self, tmp_path):
        """_compute_signal_quality_stats uses signal_quality_pnl_pct, not pnl_pct."""
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        # Trade 1: TP1 hit (+0.5%) then SL (-0.5%) → quality=win, actual=loss
        pt.record_outcome(
            "S1", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -0.5,
            signal_quality_pnl_pct=0.5, signal_quality_hit_tp=1,
        )
        # Trade 2: Straight SL → quality=loss, actual=loss
        pt.record_outcome(
            "S2", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -1.0,
        )
        # Trade 3: TP2 (+1%) then SL → quality=win, actual=loss
        pt.record_outcome(
            "S3", "360_SCALP", "BTC", "LONG", 100.0, 0, True, -0.5,
            signal_quality_pnl_pct=1.0, signal_quality_hit_tp=2,
        )

        records = pt._filter(channel="360_SCALP")
        sq_stats = PerformanceTracker._compute_signal_quality_stats("360_SCALP", records)
        actual_stats = PerformanceTracker._compute_stats("360_SCALP", records)

        # Signal quality: 2 wins (S1, S3) out of 3
        assert sq_stats.win_count == 2
        assert sq_stats.loss_count == 1
        assert sq_stats.win_rate == pytest.approx(66.67, abs=0.01)

        # Actual: 0 wins, 3 losses
        assert actual_stats.win_count == 0
        assert actual_stats.loss_count == 3
        assert actual_stats.win_rate == 0.0

    def test_format_signal_quality_stats_message_header(self, tmp_path):
        """format_signal_quality_stats_message must use 🎯 emoji and 'Signal Quality Stats' header."""
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 0.5)
        msg = pt.format_signal_quality_stats_message(channel="360_SCALP")
        assert "🎯" in msg
        assert "Signal Quality Stats" in msg
        assert "Win rate" in msg

    def test_format_stats_message_has_account_pnl_header(self, tmp_path):
        """format_stats_message must say 'Account PnL Stats'."""
        pt = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        pt.record_outcome("S1", "360_SCALP", "BTC", "LONG", 100.0, 1, False, 1.0)
        msg = pt.format_stats_message(channel="360_SCALP")
        assert "Account PnL Stats" in msg
        assert "📊" in msg

    def test_backward_compat_load_old_records(self, tmp_path):
        """Old records without signal_quality fields should load with defaults from pnl_pct."""
        import json
        path = tmp_path / "perf.json"
        # Write an old-format record (no signal_quality fields)
        old_record = {
            "signal_id": "OLD001",
            "channel": "360_SCALP",
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry": 50000.0,
            "hit_tp": 0,
            "hit_sl": True,
            "pnl_pct": -1.0,
            "confidence": 75.0,
            "outcome_label": "SL_HIT",
            "pre_ai_confidence": 0.0,
            "post_ai_confidence": 0.0,
            "setup_class": "",
            "market_phase": "",
            "quality_tier": "",
            "spread_pct": 0.0,
            "volume_24h_usd": 0.0,
            "hold_duration_sec": 0.0,
            "max_favorable_excursion_pct": 0.0,
            "max_adverse_excursion_pct": 0.0,
            "timestamp": 1700000000.0,
        }
        with open(path, "w") as f:
            json.dump([old_record], f)

        pt = PerformanceTracker(storage_path=str(path))
        assert len(pt._records) == 1
        record = pt._records[0]
        assert record.signal_quality_pnl_pct == pytest.approx(-1.0)
        assert record.signal_quality_hit_tp == 0

    def test_signal_quality_stats_empty_returns_zero(self, tmp_path):
        """_compute_signal_quality_stats on empty list returns zero stats."""
        stats = PerformanceTracker._compute_signal_quality_stats("360_SCALP", [])
        assert stats.total_signals == 0
        assert stats.win_rate == 0.0


# ---------------------------------------------------------------------------
# Fix 11: Partial TP tracking + /tp_stats
# ---------------------------------------------------------------------------


class TestTPStats:
    def _make_tracker(self, tmp_path):
        return PerformanceTracker(storage_path=str(tmp_path / "perf.json"))

    def _record(self, pt, signal_id, hit_tp, pnl_pct, hit_sl=False):
        pt.record_outcome(
            signal_id=signal_id,
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction="LONG",
            entry=50000.0,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
            pnl_pct=pnl_pct,
        )

    def test_tp_stats_empty(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        stats = pt.get_tp_stats()
        assert stats["total"] == 0
        assert stats["tp1_rate"] == 0.0

    def test_tp1_hit_rate(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        self._record(pt, "A", hit_tp=1, pnl_pct=1.0)
        self._record(pt, "B", hit_tp=1, pnl_pct=0.8)
        self._record(pt, "C", hit_tp=0, pnl_pct=-1.0, hit_sl=True)
        stats = pt.get_tp_stats()
        assert stats["total"] == 3
        assert stats["tp1_hits"] == 2
        assert stats["tp1_rate"] == pytest.approx(66.7, rel=1e-2)

    def test_tp2_and_tp3_hit_rates(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        self._record(pt, "A", hit_tp=3, pnl_pct=3.0)
        self._record(pt, "B", hit_tp=2, pnl_pct=2.0)
        self._record(pt, "C", hit_tp=1, pnl_pct=1.0)
        self._record(pt, "D", hit_tp=0, pnl_pct=-1.0, hit_sl=True)
        stats = pt.get_tp_stats()
        assert stats["tp1_hits"] == 3   # TP1 reached on TP2 and TP3 trades too
        assert stats["tp2_hits"] == 2
        assert stats["tp3_hits"] == 1

    def test_sl_hit_rate(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        self._record(pt, "A", hit_tp=0, pnl_pct=-1.0, hit_sl=True)
        self._record(pt, "B", hit_tp=0, pnl_pct=-0.8, hit_sl=True)
        self._record(pt, "C", hit_tp=1, pnl_pct=1.0)
        stats = pt.get_tp_stats()
        assert stats["sl_hits"] == 2
        assert stats["sl_rate"] == pytest.approx(66.7, rel=1e-2)

    def test_format_tp_stats_message_contains_tp_levels(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        self._record(pt, "A", hit_tp=1, pnl_pct=1.0)
        msg = pt.format_tp_stats_message()
        assert "TP1" in msg
        assert "TP2" in msg
        assert "TP3" in msg

    def test_format_tp_stats_message_no_data(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        msg = pt.format_tp_stats_message()
        assert "No data" in msg

    def test_tp_stats_channel_filter(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        pt.record_outcome("A", "360_SCALP", "BTCUSDT", "LONG", 50000, 1, False, 1.0)
        pt.record_outcome("B", "360_SWING", "ETHUSDT", "LONG", 3000, 2, False, 2.0)
        stats_scalp = pt.get_tp_stats(channel="360_SCALP")
        stats_swing = pt.get_tp_stats(channel="360_SWING")
        assert stats_scalp["total"] == 1
        assert stats_swing["total"] == 1
        assert stats_scalp["tp1_hits"] == 1
        assert stats_swing["tp2_hits"] == 1


class TestGetTopTradesAndDailySummary:
    """Tests for get_top_trades() and get_daily_summary()."""

    def _make_tracker(self, tmp_path):
        return PerformanceTracker(storage_path=str(tmp_path / "perf.json"))

    def _record(self, pt, signal_id, pnl_pct, hit_tp=1, hit_sl=False, channel="360_SCALP"):
        pt.record_outcome(
            signal_id=signal_id,
            channel=channel,
            symbol="BTCUSDT",
            direction="LONG",
            entry=50000.0,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
            pnl_pct=pnl_pct,
            signal_quality_pnl_pct=pnl_pct,
        )

    def test_get_top_trades_returns_top_n_winners(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        self._record(pt, "A", pnl_pct=3.0, hit_tp=2)
        self._record(pt, "B", pnl_pct=1.5, hit_tp=1)
        self._record(pt, "C", pnl_pct=5.0, hit_tp=3)
        self._record(pt, "D", pnl_pct=-1.0, hit_tp=0, hit_sl=True)

        top = pt.get_top_trades(n=2, window_days=1)
        assert len(top) == 2
        assert top[0].signal_id == "C"  # highest PnL
        assert top[1].signal_id == "A"

    def test_get_top_trades_excludes_losers(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        self._record(pt, "SL1", pnl_pct=-1.0, hit_sl=True)
        self._record(pt, "SL2", pnl_pct=-2.0, hit_sl=True)
        top = pt.get_top_trades(n=3, window_days=1)
        assert top == []

    def test_get_top_trades_returns_at_most_n(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        for i in range(5):
            self._record(pt, f"W{i}", pnl_pct=float(i + 1), hit_tp=1)
        top = pt.get_top_trades(n=3, window_days=1)
        assert len(top) == 3

    def test_get_daily_summary_correct_counts(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        self._record(pt, "WIN1", pnl_pct=2.0, hit_tp=2)
        self._record(pt, "WIN2", pnl_pct=1.0, hit_tp=1)
        self._record(pt, "LOSS1", pnl_pct=-1.0, hit_sl=True)
        self._record(pt, "BE1", pnl_pct=0.0, hit_tp=0)  # breakeven

        summary = pt.get_daily_summary(window_days=1)
        assert summary["total"] == 4
        assert summary["wins"] == 2
        assert summary["losses"] == 1
        assert summary["breakeven"] == 1

    def test_get_daily_summary_win_rate(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        self._record(pt, "W", pnl_pct=2.0, hit_tp=1)
        self._record(pt, "L", pnl_pct=-1.0, hit_sl=True)

        summary = pt.get_daily_summary(window_days=1)
        assert summary["win_rate"] == pytest.approx(50.0)

    def test_get_daily_summary_empty(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        summary = pt.get_daily_summary(window_days=1)
        assert summary["total"] == 0
        assert summary["wins"] == 0
        assert summary["top_trades"] == []
        assert summary["best_trade"] is None

    def test_get_daily_summary_top_trades_included(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        for i in range(4):
            self._record(pt, f"W{i}", pnl_pct=float(i + 1), hit_tp=1)
        summary = pt.get_daily_summary(window_days=1)
        # top_trades should be capped at 3
        assert len(summary["top_trades"]) == 3
        assert summary["best_trade"] is not None

    def test_get_daily_summary_avg_pnl(self, tmp_path):
        pt = self._make_tracker(tmp_path)
        self._record(pt, "A", pnl_pct=2.0, hit_tp=1)
        self._record(pt, "B", pnl_pct=-1.0, hit_sl=True)
        summary = pt.get_daily_summary(window_days=1)
        assert summary["avg_pnl"] == pytest.approx(0.5)
