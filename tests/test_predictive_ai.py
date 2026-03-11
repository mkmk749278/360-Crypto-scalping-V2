"""Tests for src.predictive_ai – predictive engine, TP/SL adjustment, confidence updates."""

import pytest

from src.channels.base import Signal
from src.predictive_ai import PredictiveEngine, PredictionResult
from src.smc import Direction
from src.utils import utcnow


class TestPredictionResult:
    def test_defaults(self):
        r = PredictionResult()
        assert r.predicted_direction == "NEUTRAL"
        assert r.confidence_adjustment == 0.0
        assert r.suggested_tp_adjustment == 1.0
        assert r.suggested_sl_adjustment == 1.0
        assert r.model_name == "none"

    def test_custom_values(self):
        r = PredictionResult(
            predicted_price=32200.0,
            predicted_direction="UP",
            confidence_adjustment=5.0,
            suggested_tp_adjustment=1.05,
            suggested_sl_adjustment=0.95,
            model_name="test-model",
        )
        assert r.predicted_direction == "UP"
        assert r.confidence_adjustment == 5.0


class TestPredictiveEngine:
    @pytest.mark.asyncio
    async def test_predict_without_model(self):
        engine = PredictiveEngine()
        result = await engine.predict("BTCUSDT", {}, {})
        assert result.predicted_direction == "NEUTRAL"
        assert result.confidence_adjustment == 0.0

    @pytest.mark.asyncio
    async def test_predict_with_model_loaded(self):
        engine = PredictiveEngine()
        await engine.load_model()
        assert engine.model_loaded is True

        result = await engine.predict(
            "BTCUSDT",
            {"close": 32000.0},
            {"momentum": 0.5, "ema_fast": 32100.0, "ema_slow": 32000.0, "close": 32000.0},
        )
        assert result.predicted_direction in ("UP", "DOWN", "NEUTRAL")
        assert result.model_name == "placeholder-momentum-v0"

    @pytest.mark.asyncio
    async def test_predict_bullish_heuristic(self):
        engine = PredictiveEngine()
        await engine.load_model()

        result = await engine.predict(
            "BTCUSDT",
            {"close": 32000.0},
            {"momentum": 2.0, "ema_fast": 32200.0, "ema_slow": 32000.0, "close": 32000.0},
        )
        assert result.predicted_direction == "UP"
        assert result.confidence_adjustment > 0

    @pytest.mark.asyncio
    async def test_predict_bearish_heuristic(self):
        engine = PredictiveEngine()
        await engine.load_model()

        result = await engine.predict(
            "BTCUSDT",
            {"close": 32000.0},
            {"momentum": -2.0, "ema_fast": 31800.0, "ema_slow": 32000.0, "close": 32000.0},
        )
        assert result.predicted_direction == "DOWN"
        assert result.confidence_adjustment < 0


class TestAdjustTPSL:
    def _make_signal(self, **kwargs):
        defaults = dict(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32000.0,
            stop_loss=31900.0,
            tp1=32100.0,
            tp2=32200.0,
            tp3=32400.0,
            confidence=85.0,
            timestamp=utcnow(),
        )
        defaults.update(kwargs)
        return Signal(**defaults)

    def test_no_adjustment_at_one(self):
        engine = PredictiveEngine()
        sig = self._make_signal()
        pred = PredictionResult(suggested_tp_adjustment=1.0, suggested_sl_adjustment=1.0)
        engine.adjust_tp_sl(sig, pred)
        assert sig.tp1 == 32100.0
        assert sig.stop_loss == 31900.0

    def test_tp_adjustment_scales_targets(self):
        engine = PredictiveEngine()
        sig = self._make_signal()
        pred = PredictionResult(suggested_tp_adjustment=1.1, suggested_sl_adjustment=1.0)
        engine.adjust_tp_sl(sig, pred)
        assert sig.tp1 == pytest.approx(32100.0 * 1.1, rel=1e-6)
        assert sig.tp2 == pytest.approx(32200.0 * 1.1, rel=1e-6)
        assert sig.tp3 == pytest.approx(32400.0 * 1.1, rel=1e-6)

    def test_sl_adjustment(self):
        engine = PredictiveEngine()
        sig = self._make_signal()
        pred = PredictionResult(suggested_tp_adjustment=1.0, suggested_sl_adjustment=0.95)
        engine.adjust_tp_sl(sig, pred)
        assert sig.stop_loss == pytest.approx(31900.0 * 0.95, rel=1e-6)

    def test_tp3_none_not_adjusted(self):
        engine = PredictiveEngine()
        sig = self._make_signal(tp3=None)
        pred = PredictionResult(suggested_tp_adjustment=1.1, suggested_sl_adjustment=1.0)
        engine.adjust_tp_sl(sig, pred)
        assert sig.tp3 is None


class TestUpdateConfidence:
    def _make_signal(self, confidence=85.0):
        return Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32000.0,
            stop_loss=31900.0,
            tp1=32100.0,
            tp2=32200.0,
            confidence=confidence,
            timestamp=utcnow(),
        )

    def test_positive_adjustment(self):
        engine = PredictiveEngine()
        sig = self._make_signal(confidence=85.0)
        pred = PredictionResult(confidence_adjustment=5.0)
        engine.update_confidence(sig, pred)
        assert sig.confidence == 90.0

    def test_negative_adjustment(self):
        engine = PredictiveEngine()
        sig = self._make_signal(confidence=85.0)
        pred = PredictionResult(confidence_adjustment=-10.0)
        engine.update_confidence(sig, pred)
        assert sig.confidence == 75.0

    def test_clamp_upper(self):
        engine = PredictiveEngine()
        sig = self._make_signal(confidence=98.0)
        pred = PredictionResult(confidence_adjustment=10.0)
        engine.update_confidence(sig, pred)
        assert sig.confidence == 100.0

    def test_clamp_lower(self):
        engine = PredictiveEngine()
        sig = self._make_signal(confidence=3.0)
        pred = PredictionResult(confidence_adjustment=-10.0)
        engine.update_confidence(sig, pred)
        assert sig.confidence == 0.0

    def test_zero_adjustment(self):
        engine = PredictiveEngine()
        sig = self._make_signal(confidence=85.0)
        pred = PredictionResult(confidence_adjustment=0.0)
        engine.update_confidence(sig, pred)
        assert sig.confidence == 85.0
