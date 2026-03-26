"""Tests for the Backtester – backtest framework with synthetic candle data."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtester import (
    Backtester,
    BacktestConfig,
    BacktestResult,
    WalkForwardReport,
    _compute_indicators,
    _simulate_trade,
)
from src.channels.scalp import ScalpChannel
from src.channels.spot import SpotChannel


def _make_candles(
    n: int = 100,
    base: float = 100.0,
    trend: float = 0.1,
    noise: float = 0.2,
) -> dict:
    """Generate synthetic OHLCV data with a fixed RNG seed for reproducibility."""
    close = np.cumsum(np.ones(n) * trend) + base
    close += np.random.default_rng(42).normal(0, noise, n)  # seed=42 for stable tests
    high = close + 0.5
    low = close - 0.5
    volume = np.ones(n) * 1000.0
    return {"open": close - 0.05, "high": high, "low": low, "close": close, "volume": volume}


class TestBacktestResult:
    def test_summary_contains_channel(self):
        r = BacktestResult(channel="360_SCALP", total_signals=10, wins=7, losses=3)
        assert "360_SCALP" in r.summary()

    def test_summary_contains_win_rate(self):
        r = BacktestResult(channel="360_SCALP", win_rate=70.0)
        assert "70.0" in r.summary()

    def test_default_values(self):
        r = BacktestResult(channel="TEST")
        assert r.total_signals == 0
        assert r.wins == 0
        assert r.losses == 0
        assert r.signal_details == []


class TestComputeIndicators:
    def test_returns_dict_with_ema(self):
        n = 250
        c = np.cumsum(np.ones(n) * 0.1) + 100.0
        h = c + 0.5
        lo = c - 0.5
        candles = {"high": h, "low": lo, "close": c, "volume": np.ones(n)}
        ind = _compute_indicators(candles)
        assert "ema9_last" in ind
        assert "ema21_last" in ind

    def test_insufficient_data_skips_indicators(self):
        c = np.ones(5) * 100.0
        ind = _compute_indicators({"high": c + 1, "low": c - 1, "close": c})
        assert "ema9_last" not in ind  # need >= 21 candles


class TestSimulateTrade:
    def _fake_signal(self, direction="LONG", entry=100.0, sl=99.0, tp1=101.0, tp2=102.0):
        class _FakeDir:
            value = direction

        class _FakeSig:
            pass

        s = _FakeSig()
        s.direction = _FakeDir()
        s.entry = entry
        s.stop_loss = sl
        s.tp1 = tp1
        s.tp2 = tp2
        s.tp3 = None
        return s

    def test_long_tp1_hit(self):
        sig = self._fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([100.5, 101.5]),
            "low": np.array([99.5, 100.0]),
            "close": np.array([100.5, 101.5]),
        }
        won, pnl, tp_level = _simulate_trade(sig, future)
        assert won is True
        assert tp_level == 1
        assert pnl > 0

    def test_long_sl_hit(self):
        sig = self._fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([100.5, 100.0]),
            "low": np.array([99.5, 98.5]),  # low < SL
            "close": np.array([100.5, 99.0]),
        }
        won, pnl, tp_level = _simulate_trade(sig, future)
        assert won is False
        assert tp_level == 0

    def test_short_tp1_hit(self):
        sig = self._fake_signal("SHORT", entry=100.0, sl=101.0, tp1=99.0, tp2=98.0)
        future = {
            "high": np.array([100.5, 100.0]),
            "low": np.array([99.5, 98.8]),  # low <= TP1
            "close": np.array([99.5, 99.0]),
        }
        won, pnl, tp_level = _simulate_trade(sig, future)
        assert won is True
        assert tp_level == 1

    def test_no_future_candles(self):
        sig = self._fake_signal()
        future = {"high": np.array([]), "low": np.array([]), "close": np.array([])}
        won, pnl, tp_level = _simulate_trade(sig, future)
        assert won is False
        assert pnl == 0.0


class TestBacktester:
    def test_run_returns_results_for_each_channel(self):
        bt = Backtester(min_window=30, lookahead_candles=5)
        candles = _make_candles(n=200)
        candles_by_tf = {
            "5m": candles,
            "1m": candles,
            "15m": candles,
            "1h": candles,
            "4h": candles,
        }
        results = bt.run(candles_by_tf, symbol="BTCUSDT")
        assert isinstance(results, list)
        assert len(results) == 3  # one per channel (SCALP, SWING, SPOT)

    def test_run_single_channel(self):
        bt = Backtester(min_window=30, lookahead_candles=5)
        candles = _make_candles(n=200)
        candles_by_tf = {"5m": candles, "1m": candles}
        results = bt.run(candles_by_tf, channel_name="360_SCALP")
        assert len(results) == 1
        assert results[0].channel == "360_SCALP"

    def test_result_has_win_rate_in_range(self):
        bt = Backtester(min_window=30, lookahead_candles=10)
        candles = _make_candles(n=300, trend=0.5)
        candles_by_tf = {"5m": candles, "1m": candles}
        results = bt.run(candles_by_tf, channel_name="360_SCALP")
        if results[0].total_signals > 0:
            assert 0.0 <= results[0].win_rate <= 100.0

    def test_missing_timeframe_returns_empty_result(self):
        bt = Backtester(channels=[ScalpChannel()], min_window=30, lookahead_candles=5)
        # No 5m timeframe provided
        candles_by_tf = {"1h": _make_candles(200)}
        results = bt.run(candles_by_tf)
        assert results[0].total_signals == 0

    def test_custom_channel_list(self):
        bt = Backtester(
            channels=[SpotChannel()], min_window=30, lookahead_candles=5
        )
        candles = _make_candles(n=200)
        candles_by_tf = {"4h": candles, "5m": candles}
        results = bt.run(candles_by_tf)
        assert len(results) == 1
        assert results[0].channel == "360_SPOT"


class TestFeeDeduction:
    """Fee model in _simulate_trade must reduce PnL by fee_pct."""

    def _fake_signal(self, direction="LONG", entry=100.0, sl=99.0, tp1=101.0, tp2=102.0):
        class _FakeDir:
            value = direction

        class _FakeSig:
            pass

        s = _FakeSig()
        s.direction = _FakeDir()
        s.entry = entry
        s.stop_loss = sl
        s.tp1 = tp1
        s.tp2 = tp2
        s.tp3 = None
        return s

    def test_fee_deducted_from_winning_trade(self):
        sig = self._fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([101.5]),
            "low": np.array([99.5]),
            "close": np.array([101.5]),
        }
        won_no_fee, pnl_no_fee, _ = _simulate_trade(sig, future, fee_pct=0.0)
        won_fee, pnl_fee, _ = _simulate_trade(sig, future, fee_pct=0.08)

        assert won_no_fee is True
        assert won_fee is True
        assert pnl_fee == pytest.approx(pnl_no_fee - 0.08)

    def test_fee_deducted_from_losing_trade(self):
        sig = self._fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([100.5]),
            "low": np.array([98.5]),
            "close": np.array([99.0]),
        }
        won_no_fee, pnl_no_fee, _ = _simulate_trade(sig, future, fee_pct=0.0)
        won_fee, pnl_fee, _ = _simulate_trade(sig, future, fee_pct=0.08)

        assert won_no_fee is False
        assert won_fee is False
        assert pnl_fee == pytest.approx(pnl_no_fee - 0.08)

    def test_backtester_fee_pct_reduces_total_pnl(self):
        """A Backtester with fee_pct set should produce lower total_pnl than without fees."""
        candles = _make_candles(n=200, trend=0.5)
        candles_by_tf = {"5m": candles, "1m": candles}

        bt_no_fee = Backtester(channels=[ScalpChannel()], min_window=30, lookahead_candles=10, fee_pct=0.0)
        bt_with_fee = Backtester(channels=[ScalpChannel()], min_window=30, lookahead_candles=10, fee_pct=0.08)

        results_no_fee = bt_no_fee.run(candles_by_tf, channel_name="360_SCALP")
        results_with_fee = bt_with_fee.run(candles_by_tf, channel_name="360_SCALP")

        if results_no_fee[0].total_signals > 0:
            assert results_with_fee[0].total_pnl_pct < results_no_fee[0].total_pnl_pct

    def test_simulated_ai_score_passed_to_channel(self):
        """Backtester.run() should accept simulated_ai_score without error."""
        bt = Backtester(channels=[ScalpChannel()], min_window=30, lookahead_candles=5)
        candles = _make_candles(n=200)
        candles_by_tf = {"5m": candles, "1m": candles}
        results = bt.run(candles_by_tf, channel_name="360_SCALP", simulated_ai_score=-0.5)
        assert isinstance(results, list)
        assert len(results) == 1


class TestSlippageModel:
    """Slippage must be applied adversely to every SL/TP fill price."""

    def _fake_signal(self, direction="LONG", entry=100.0, sl=99.0, tp1=101.0, tp2=102.0):
        class _FakeDir:
            value = direction

        class _FakeSig:
            pass

        s = _FakeSig()
        s.direction = _FakeDir()
        s.entry = entry
        s.stop_loss = sl
        s.tp1 = tp1
        s.tp2 = tp2
        s.tp3 = None
        return s

    def test_long_sl_fill_below_stop_level(self):
        """LONG SL fill must be below the nominal stop level (adverse slippage)."""
        sig = self._fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([100.5]),
            "low": np.array([98.5]),  # below SL
            "close": np.array([99.0]),
        }
        # Without slippage
        _, pnl_no_slip, _ = _simulate_trade(sig, future, slippage_pct=0.0)
        # With slippage = 0.1 % → fill at 99.0 * (1 - 0.001) = 98.901
        _, pnl_with_slip, _ = _simulate_trade(sig, future, slippage_pct=0.1)
        # Slippage makes the loss WORSE (more negative PnL)
        assert pnl_with_slip < pnl_no_slip

    def test_short_sl_fill_above_stop_level(self):
        """SHORT SL fill must be above the nominal stop level (adverse slippage)."""
        sig = self._fake_signal("SHORT", entry=100.0, sl=101.0, tp1=99.0)
        future = {
            "high": np.array([101.5]),  # above SL
            "low": np.array([100.0]),
            "close": np.array([101.0]),
        }
        _, pnl_no_slip, _ = _simulate_trade(sig, future, slippage_pct=0.0)
        _, pnl_with_slip, _ = _simulate_trade(sig, future, slippage_pct=0.1)
        assert pnl_with_slip < pnl_no_slip

    def test_long_tp_fill_below_target(self):
        """LONG TP fill must be slightly below the target (adverse slippage)."""
        sig = self._fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([101.5]),
            "low": np.array([100.0]),
            "close": np.array([101.5]),
        }
        _, pnl_no_slip, _ = _simulate_trade(sig, future, slippage_pct=0.0)
        _, pnl_with_slip, _ = _simulate_trade(sig, future, slippage_pct=0.1)
        # TP PnL is slightly reduced by slippage
        assert pnl_with_slip < pnl_no_slip

    def test_short_tp_fill_above_target(self):
        """SHORT TP fill must be slightly above the target (adverse slippage)."""
        sig = self._fake_signal("SHORT", entry=100.0, sl=101.0, tp1=99.0, tp2=98.0)
        future = {
            "high": np.array([100.0]),
            "low": np.array([98.5]),  # below tp1
            "close": np.array([99.0]),
        }
        _, pnl_no_slip, _ = _simulate_trade(sig, future, slippage_pct=0.0)
        _, pnl_with_slip, _ = _simulate_trade(sig, future, slippage_pct=0.1)
        assert pnl_with_slip < pnl_no_slip

    def test_zero_slippage_matches_original_behavior(self):
        """With slippage_pct=0 the result must match the no-slippage path exactly."""
        sig = self._fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([101.5]),
            "low": np.array([99.5]),
            "close": np.array([101.5]),
        }
        won_a, pnl_a, lvl_a = _simulate_trade(sig, future)
        won_b, pnl_b, lvl_b = _simulate_trade(sig, future, slippage_pct=0.0)
        assert won_a == won_b
        assert pnl_a == pytest.approx(pnl_b)
        assert lvl_a == lvl_b

    def test_backtester_slippage_reduces_pnl(self):
        """Backtester with slippage_pct set must produce lower PnL than without."""
        candles = _make_candles(n=200, trend=0.5)
        candles_by_tf = {"5m": candles, "1m": candles}

        bt_no_slip = Backtester(
            channels=[ScalpChannel()], min_window=30, lookahead_candles=10, slippage_pct=0.0
        )
        bt_with_slip = Backtester(
            channels=[ScalpChannel()], min_window=30, lookahead_candles=10, slippage_pct=0.03
        )

        res_no = bt_no_slip.run(candles_by_tf, channel_name="360_SCALP")
        res_with = bt_with_slip.run(candles_by_tf, channel_name="360_SCALP")

        if res_no[0].total_signals > 0:
            assert res_with[0].total_pnl_pct <= res_no[0].total_pnl_pct

    def test_backtest_result_slippage_stored(self):
        """BacktestResult must expose the slippage_pct used."""
        r = BacktestResult(channel="TEST", slippage_pct=0.03)
        assert r.slippage_pct == pytest.approx(0.03)

    def test_backtest_result_summary_shows_slippage(self):
        """summary() must mention slippage when slippage_pct > 0."""
        r = BacktestResult(channel="TEST", slippage_pct=0.03)
        assert "Slippage" in r.summary()

    def test_backtest_result_summary_no_slippage_note_when_zero(self):
        """summary() must NOT mention slippage when slippage_pct == 0."""
        r = BacktestResult(channel="TEST", slippage_pct=0.0)
        assert "Slippage" not in r.summary()


# ---------------------------------------------------------------------------
# Fix 8: Realistic default fee and slippage
# ---------------------------------------------------------------------------


class TestRealisticDefaults:
    """Backtester must default to realistic fee and slippage out of the box."""

    def test_default_fee_is_008(self):
        bt = Backtester()
        assert bt._fee_pct == pytest.approx(0.08)

    def test_default_slippage_is_002(self):
        bt = Backtester()
        assert bt._slippage_pct == pytest.approx(0.02)

    def test_explicit_zero_overrides_default(self):
        """Callers can still opt out of fees and slippage explicitly."""
        bt = Backtester(fee_pct=0.0, slippage_pct=0.0)
        assert bt._fee_pct == 0.0
        assert bt._slippage_pct == 0.0


# ---------------------------------------------------------------------------
# PR_11: Per-pair sweep, regime tagging, walk-forward validation
# ---------------------------------------------------------------------------


class TestRegimeTagging:
    def test_regime_tagging_populates_regime_field(self):
        """Every signal_detail must contain a 'regime' key when tag_regimes=True."""
        bt = Backtester()
        n = 200
        hist = {
            "close": np.linspace(100, 110, n).tolist(),
            "open": np.linspace(99, 109, n).tolist(),
            "high": np.linspace(101, 111, n).tolist(),
            "low": np.linspace(98, 108, n).tolist(),
            "volume": [1e6] * n,
        }
        results = bt.run(hist, tag_regimes=True)
        assert isinstance(results, list)
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, BacktestResult)
        for detail in result.signal_details:
            assert "regime" in detail

    def test_regime_field_present_even_without_signals(self):
        """run() with tag_regimes must not raise even when no signals are emitted."""
        bt = Backtester()
        n = 60  # very short – likely no signals
        hist = {
            "close": [100.0] * n,
            "open": [99.9] * n,
            "high": [100.1] * n,
            "low": [99.8] * n,
            "volume": [1e6] * n,
        }
        results = bt.run(hist, tag_regimes=True)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], BacktestResult)


class TestWalkForwardValidation:
    def _make_hist(self, n: int = 500) -> dict:
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        return {
            "close": close.tolist(),
            "open": (close - 0.1).tolist(),
            "high": (close + 0.3).tolist(),
            "low": (close - 0.3).tolist(),
            "volume": [1e6] * n,
        }

    def test_walk_forward_returns_report(self):
        bt = Backtester()
        report = bt.walk_forward_validate(self._make_hist(), n_folds=3)
        assert isinstance(report, WalkForwardReport)
        assert report.n_folds == 3
        assert 0.0 <= report.avg_in_sample_winrate <= 1.0
        assert 0.0 <= report.avg_out_sample_winrate <= 1.0

    def test_walk_forward_fold_count_matches(self):
        bt = Backtester()
        report = bt.walk_forward_validate(self._make_hist(), n_folds=4)
        assert len(report.fold_results) == 4

    def test_walk_forward_summary_string(self):
        bt = Backtester()
        report = bt.walk_forward_validate(self._make_hist(), n_folds=2)
        s = report.summary()
        assert "Walk-Forward" in s
        assert "IS WR" in s
        assert "OOS WR" in s


class TestPerPairSweep:
    def _make_pair_data(self, n: int = 200) -> dict:
        rng = np.random.default_rng(1)
        close = 100 + np.cumsum(rng.normal(0, 0.3, n))
        return {
            "close": close.tolist(),
            "open": (close - 0.1).tolist(),
            "high": (close + 0.2).tolist(),
            "low": (close - 0.2).tolist(),
            "volume": [1e6] * n,
        }

    def test_per_pair_sweep_returns_results_per_pair(self):
        bt = Backtester()
        data_by_pair = {sym: self._make_pair_data() for sym in ["BTCUSDT", "ETHUSDT"]}
        configs = [BacktestConfig(atr_sl_mult=1.0), BacktestConfig(atr_sl_mult=1.5)]
        results = bt.run_per_pair_sweep(data_by_pair, configs)
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert len(results["BTCUSDT"]) == 2
        assert len(results["ETHUSDT"]) == 2

    def test_per_pair_sweep_channel_label_includes_pair(self):
        bt = Backtester()
        data_by_pair = {"SOLUSDT": self._make_pair_data()}
        configs = [BacktestConfig()]
        results = bt.run_per_pair_sweep(data_by_pair, configs)
        assert "SOLUSDT" in results["SOLUSDT"][0].channel
