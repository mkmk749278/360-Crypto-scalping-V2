"""Tests for enhanced confidence scoring (PR: Signal Confidence Scoring)."""

from __future__ import annotations

import pytest

from src.confidence import (
    ConfidenceInput,
    ConfidenceResult,
    compute_adaptive_threshold,
    compute_per_signal_confidence,
)


class TestAdaptiveThreshold:
    def test_default_threshold(self):
        threshold = compute_adaptive_threshold()
        assert 50.0 <= threshold <= 90.0

    def test_trending_lowers_threshold(self):
        trending = compute_adaptive_threshold(base_threshold=65.0, regime="TRENDING")
        neutral = compute_adaptive_threshold(base_threshold=65.0, regime="QUIET")
        assert trending < neutral

    def test_volatile_raises_threshold(self):
        volatile = compute_adaptive_threshold(base_threshold=65.0, regime="VOLATILE")
        quiet = compute_adaptive_threshold(base_threshold=65.0, regime="QUIET")
        assert volatile > quiet

    def test_ranging_raises_threshold(self):
        ranging = compute_adaptive_threshold(base_threshold=65.0, regime="RANGING")
        quiet = compute_adaptive_threshold(base_threshold=65.0, regime="QUIET")
        assert ranging > quiet

    def test_extreme_volatility_adds_buffer(self):
        normal = compute_adaptive_threshold(base_threshold=65.0, volatility_percentile=0.5)
        extreme = compute_adaptive_threshold(base_threshold=65.0, volatility_percentile=0.95)
        assert extreme > normal

    def test_gem_channel_lower_threshold(self):
        gem = compute_adaptive_threshold(base_threshold=65.0, channel="360_GEM")
        scalp = compute_adaptive_threshold(base_threshold=65.0, channel="360_SCALP")
        assert gem < scalp

    def test_threshold_clamped_low(self):
        threshold = compute_adaptive_threshold(base_threshold=40.0, regime="TRENDING")
        assert threshold >= 50.0

    def test_threshold_clamped_high(self):
        threshold = compute_adaptive_threshold(base_threshold=95.0, regime="VOLATILE",
                                                volatility_percentile=0.99)
        assert threshold <= 90.0


class TestPerSignalConfidence:
    def _make_input(self, **kwargs):
        defaults = dict(
            smc_score=20.0,
            trend_score=15.0,
            liquidity_score=10.0,
            spread_score=5.0,
            data_sufficiency=8.0,
            multi_exchange=2.5,
            onchain_score=5.0,
            order_flow_score=10.0,
        )
        defaults.update(kwargs)
        return ConfidenceInput(**defaults)

    def test_returns_confidence_result(self):
        inp = self._make_input()
        result = compute_per_signal_confidence(inp, channel="360_SCALP")
        assert isinstance(result, ConfidenceResult)
        assert result.total >= 0
        assert result.total <= 100

    def test_includes_adaptive_threshold(self):
        inp = self._make_input()
        result = compute_per_signal_confidence(inp, regime="TRENDING")
        assert result.adaptive_threshold > 0

    def test_includes_regime(self):
        inp = self._make_input()
        result = compute_per_signal_confidence(inp, regime="VOLATILE")
        assert result.regime == "VOLATILE"

    def test_cluster_suppression_applied(self):
        inp = self._make_input()
        result = compute_per_signal_confidence(
            inp,
            cluster_suppressed=True,
            cluster_reason="Too many signals",
        )
        assert result.suppressed is True
        assert result.suppressed_reason == "Too many signals"

    def test_not_suppressed_by_default(self):
        inp = self._make_input()
        result = compute_per_signal_confidence(inp)
        assert result.suppressed is False
        assert result.suppressed_reason == ""


class TestConfidenceResultNewFields:
    def test_default_new_fields(self):
        result = ConfidenceResult(total=70.0)
        assert result.suppressed is False
        assert result.suppressed_reason == ""
        assert result.regime == ""
        assert result.adaptive_threshold == 65.0
