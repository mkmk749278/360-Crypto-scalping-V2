"""Tests for the AI Engine sub-modules (predictor, scorer, feedback)."""

from __future__ import annotations

import pytest

from src.ai_engine.predictor import (
    PredictionFeatures,
    SignalPrediction,
    SignalPredictor,
)
from src.ai_engine.scorer import (
    AIConfidenceScorer,
    AIScoreResult,
)
from src.ai_engine.feedback import (
    AIFeedbackAdapter,
    PredictionRecord,
)


# ---------------------------------------------------------------------------
# SignalPredictor tests
# ---------------------------------------------------------------------------


class TestSignalPredictor:
    async def test_predict_returns_prediction(self):
        predictor = SignalPredictor()
        features = PredictionFeatures(
            price_features={"ema_alignment": 0.8, "momentum": 0.5, "adx": 30.0},
            volume_features={"obv_trend": 0.6, "volume_spike": 0.3},
        )
        result = await predictor.predict("BTCUSDT", features)
        assert isinstance(result, SignalPrediction)
        assert result.symbol == "BTCUSDT"
        assert result.direction in ("LONG", "SHORT", "NEUTRAL")
        assert 0.0 <= result.probability <= 1.0

    async def test_predict_neutral_with_empty_features(self):
        predictor = SignalPredictor()
        features = PredictionFeatures()
        result = await predictor.predict("ETHUSDT", features)
        assert result.direction == "NEUTRAL"
        assert result.probability == pytest.approx(0.5, abs=0.1)

    async def test_predict_bullish_with_strong_signals(self):
        predictor = SignalPredictor(min_probability=0.55)
        features = PredictionFeatures(
            price_features={"ema_alignment": 1.0, "momentum": 1.0, "adx": 40.0},
            volume_features={"obv_trend": 1.0, "volume_spike": 1.0},
            order_book_features={"bid_ask_imbalance": 0.8, "depth_ratio": 1.5},
            correlation_features={"btc_correlation": 0.5, "sector_direction": 0.5},
        )
        result = await predictor.predict("SOLUSDT", features)
        assert result.direction == "LONG"
        assert result.probability > 0.55

    async def test_predict_batch(self):
        predictor = SignalPredictor()
        features_map = {
            "BTCUSDT": PredictionFeatures(
                price_features={"ema_alignment": 0.5, "momentum": 0.3, "adx": 25.0}
            ),
            "ETHUSDT": PredictionFeatures(
                price_features={"ema_alignment": -0.5, "momentum": -0.3, "adx": 20.0}
            ),
        }
        results = await predictor.predict_batch(features_map)
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert isinstance(results["BTCUSDT"], SignalPrediction)

    async def test_prediction_count_increments(self):
        predictor = SignalPredictor()
        assert predictor.prediction_count == 0
        features = PredictionFeatures()
        await predictor.predict("BTC", features)
        assert predictor.prediction_count == 1
        await predictor.predict("ETH", features)
        assert predictor.prediction_count == 2


# ---------------------------------------------------------------------------
# AIConfidenceScorer tests
# ---------------------------------------------------------------------------


class TestAIConfidenceScorer:
    def test_score_signal_returns_result(self):
        scorer = AIConfidenceScorer()
        result = scorer.score_signal("BTCUSDT", 70.0, regime="TRENDING")
        assert isinstance(result, AIScoreResult)
        assert result.symbol == "BTCUSDT"
        assert result.base_confidence == 70.0
        assert 0.0 <= result.final_confidence <= 100.0

    def test_high_confidence_above_threshold(self):
        scorer = AIConfidenceScorer(base_threshold=60.0)
        result = scorer.score_signal("BTCUSDT", 75.0, regime="TRENDING")
        assert result.is_high_confidence is True

    def test_low_confidence_below_threshold(self):
        scorer = AIConfidenceScorer(base_threshold=80.0)
        result = scorer.score_signal("BTCUSDT", 50.0, regime="VOLATILE")
        assert result.is_high_confidence is False

    def test_trending_regime_lowers_threshold(self):
        scorer = AIConfidenceScorer(base_threshold=65.0)
        trending = scorer.score_signal("BTCUSDT", 62.0, regime="TRENDING")
        ranging = scorer.score_signal("BTCUSDT", 62.0, regime="RANGING")
        assert trending.dynamic_threshold < ranging.dynamic_threshold

    def test_volatile_regime_raises_threshold(self):
        scorer = AIConfidenceScorer(base_threshold=65.0)
        volatile = scorer.score_signal("BTCUSDT", 70.0, regime="VOLATILE")
        quiet = scorer.score_signal("BTCUSDT", 70.0, regime="QUIET")
        assert volatile.dynamic_threshold > quiet.dynamic_threshold

    def test_high_win_rate_boosts_confidence(self):
        scorer = AIConfidenceScorer()
        result = scorer.score_signal("BTCUSDT", 70.0, pair_win_rate=0.80)
        assert result.ai_adjustment > 0

    def test_low_win_rate_penalises_confidence(self):
        scorer = AIConfidenceScorer()
        result = scorer.score_signal("BTCUSDT", 70.0, pair_win_rate=0.20)
        assert result.ai_adjustment < 0

    def test_pair_threshold_stored(self):
        scorer = AIConfidenceScorer(base_threshold=65.0)
        scorer.score_signal("BTCUSDT", 70.0, regime="TRENDING")
        assert scorer.get_pair_threshold("BTCUSDT") < 65.0

    def test_pair_avg_confidence(self):
        scorer = AIConfidenceScorer()
        scorer.score_signal("BTCUSDT", 70.0)
        scorer.score_signal("BTCUSDT", 80.0)
        avg = scorer.get_pair_avg_confidence("BTCUSDT")
        assert avg > 0


# ---------------------------------------------------------------------------
# AIFeedbackAdapter tests
# ---------------------------------------------------------------------------


class TestAIFeedbackAdapter:
    def test_record_prediction_outcome(self):
        adapter = AIFeedbackAdapter()
        record = adapter.record_prediction_outcome("BTCUSDT", "LONG", "LONG", 75.0)
        assert isinstance(record, PredictionRecord)
        assert record.was_correct is True

    def test_accuracy_tracking(self):
        adapter = AIFeedbackAdapter()
        adapter.record_prediction_outcome("BTC", "LONG", "LONG", 70.0)
        adapter.record_prediction_outcome("ETH", "SHORT", "LONG", 60.0)
        assert adapter.accuracy == pytest.approx(0.5)
        assert adapter.total_predictions == 2

    def test_pair_accuracy_insufficient_data(self):
        adapter = AIFeedbackAdapter()
        adapter.record_prediction_outcome("BTC", "LONG", "LONG", 70.0)
        assert adapter.get_pair_accuracy("BTC") == 0.5  # < 5 records

    def test_pair_accuracy_with_enough_data(self):
        adapter = AIFeedbackAdapter()
        for _ in range(8):
            adapter.record_prediction_outcome("BTC", "LONG", "LONG", 70.0)
        for _ in range(2):
            adapter.record_prediction_outcome("BTC", "LONG", "SHORT", 70.0)
        assert adapter.get_pair_accuracy("BTC") == pytest.approx(0.8)

    def test_accuracy_by_confidence_tier(self):
        adapter = AIFeedbackAdapter()
        adapter.record_prediction_outcome("A", "LONG", "LONG", 50.0)  # low, correct
        adapter.record_prediction_outcome("B", "LONG", "SHORT", 65.0)  # medium, wrong
        adapter.record_prediction_outcome("C", "LONG", "LONG", 80.0)  # high, correct
        tiers = adapter.get_accuracy_by_confidence_tier()
        assert "low" in tiers
        assert "medium" in tiers
        assert "high" in tiers
        assert tiers["low"] == 1.0
        assert tiers["medium"] == 0.0
        assert tiers["high"] == 1.0

    def test_get_recent_records(self):
        adapter = AIFeedbackAdapter()
        for i in range(10):
            adapter.record_prediction_outcome(f"SYM{i}", "LONG", "LONG", 70.0)
        recent = adapter.get_recent_records(5)
        assert len(recent) == 5

    def test_max_records_eviction(self):
        adapter = AIFeedbackAdapter(max_records=5)
        for i in range(10):
            adapter.record_prediction_outcome(f"SYM{i}", "LONG", "LONG", 70.0)
        assert len(adapter._records) == 5


# ---------------------------------------------------------------------------
# Backward compatibility — ensure original imports still work
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_import_from_ai_engine_package(self):
        from src.ai_engine import (
            SentimentResult,
            WhaleAlert,
            detect_whale_trade,
            detect_volume_delta_spike,
            get_ai_insight,
            close_shared_session,
        )
        assert SentimentResult is not None
        assert WhaleAlert is not None

    def test_import_module_style(self):
        import src.ai_engine as ai_engine
        assert hasattr(ai_engine, "SentimentResult")
        assert hasattr(ai_engine, "get_ai_insight")

    def test_new_exports_accessible(self):
        from src.ai_engine import SignalPredictor, AIConfidenceScorer, AIFeedbackAdapter
        assert SignalPredictor is not None
        assert AIConfidenceScorer is not None
        assert AIFeedbackAdapter is not None
