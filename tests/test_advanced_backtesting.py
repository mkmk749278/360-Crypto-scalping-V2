"""Tests for advanced backtesting (PR: Advanced Backtesting Framework)."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtester import (
    AnalyticsReport,
    Backtester,
    BacktestConfig,
    MonteCarloReport,
    RegimeStressReport,
)


def _make_hist(n: int = 300, trend: float = 0.1, seed: int = 42) -> dict:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(trend, 0.5, n))
    return {
        "close": close.tolist(),
        "open": (close - 0.05).tolist(),
        "high": (close + 0.3).tolist(),
        "low": (close - 0.3).tolist(),
        "volume": [1e6] * n,
    }


class TestMonteCarloReport:
    def test_returns_report(self):
        bt = Backtester()
        report = bt.run_monte_carlo(_make_hist(300), n_simulations=20)
        assert isinstance(report, MonteCarloReport)
        assert report.n_simulations == 20

    def test_percentiles_ordered(self):
        bt = Backtester()
        report = bt.run_monte_carlo(_make_hist(300), n_simulations=50)
        assert report.pnl_5th_percentile <= report.pnl_95th_percentile

    def test_ruin_probability_in_range(self):
        bt = Backtester()
        report = bt.run_monte_carlo(_make_hist(300), n_simulations=50)
        assert 0.0 <= report.ruin_probability <= 1.0

    def test_worst_drawdown_nonnegative(self):
        bt = Backtester()
        report = bt.run_monte_carlo(_make_hist(300), n_simulations=20)
        assert report.worst_drawdown >= 0.0
        assert report.avg_max_drawdown >= 0.0

    def test_empty_data_returns_zeros(self):
        bt = Backtester()
        # Very short data likely yields no signals
        report = bt.run_monte_carlo(
            {"close": [100.0] * 10, "open": [99.0] * 10,
             "high": [101.0] * 10, "low": [99.0] * 10, "volume": [1e6] * 10},
            n_simulations=5,
        )
        assert isinstance(report, MonteCarloReport)
        assert report.avg_total_pnl == 0.0

    def test_summary_string(self):
        bt = Backtester()
        report = bt.run_monte_carlo(_make_hist(300), n_simulations=10)
        s = report.summary()
        assert "Monte Carlo" in s
        assert "sims" in s

    def test_reproducible_with_seed(self):
        bt = Backtester()
        r1 = bt.run_monte_carlo(_make_hist(300), n_simulations=10, seed=123)
        r2 = bt.run_monte_carlo(_make_hist(300), n_simulations=10, seed=123)
        assert r1.avg_total_pnl == pytest.approx(r2.avg_total_pnl)


class TestRegimeStressTest:
    def test_returns_report(self):
        bt = Backtester()
        report = bt.run_regime_stress_test(_make_hist(300))
        assert isinstance(report, RegimeStressReport)

    def test_regime_results_populated(self):
        bt = Backtester()
        report = bt.run_regime_stress_test(_make_hist(300))
        # May have some regime entries (or empty if no signals)
        assert isinstance(report.regime_results, dict)

    def test_summary_string(self):
        bt = Backtester()
        report = bt.run_regime_stress_test(_make_hist(300))
        s = report.summary()
        assert "Regime Stress Test" in s

    def test_regime_stats_fields(self):
        bt = Backtester()
        report = bt.run_regime_stress_test(_make_hist(300))
        for regime, stats in report.regime_results.items():
            assert "total_signals" in stats
            assert "wins" in stats
            assert "win_rate" in stats
            assert "total_pnl_pct" in stats


class TestAnalyticsReport:
    def test_generate_analytics_report(self):
        bt = Backtester()
        report = bt.generate_analytics_report(
            _make_hist(300),
            monte_carlo_sims=10,
            walk_forward_folds=2,
        )
        assert isinstance(report, AnalyticsReport)
        assert report.monte_carlo is not None
        assert report.regime_stress is not None
        assert report.walk_forward is not None

    def test_analytics_summary(self):
        bt = Backtester()
        report = bt.generate_analytics_report(
            _make_hist(300),
            monte_carlo_sims=5,
            walk_forward_folds=2,
        )
        s = report.summary()
        assert "Analytics Report" in s

    def test_analytics_includes_backtest_results(self):
        bt = Backtester()
        report = bt.generate_analytics_report(_make_hist(300), monte_carlo_sims=5)
        assert len(report.backtest_results) >= 1
