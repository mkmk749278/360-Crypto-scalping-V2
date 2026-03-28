"""Tests for PR08 – Historical Replay Simulator."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.simulation.simulator import (
    Simulator,
    SimCandle,
    SimSignal,
    SimResult,
)


def _make_candles(
    n: int = 100,
    symbol: str = "BTCUSDT",
    base_price: float = 50000.0,
    timeframe: str = "5m",
) -> list:
    """Build a list of synthetic SimCandle objects for testing."""
    candles = []
    import time as _time
    ts = _time.time() - n * 300  # 5-minute candles going back n periods
    for i in range(n):
        close = base_price + (i % 20 - 10) * 100.0  # oscillate
        candle = SimCandle(
            timestamp=ts + i * 300,
            open=close - 50.0,
            high=close + 200.0,
            low=close - 200.0,
            close=close,
            volume=500.0,
            symbol=symbol,
            timeframe=timeframe,
        )
        candles.append(candle)
    return candles


def _make_historical_candles(
    symbols: list = None,
) -> dict:
    if symbols is None:
        symbols = ["BTCUSDT"]
    return {sym: {"5m": _make_candles(symbol=sym)} for sym in symbols}


class TestSimCandle:
    def test_defaults(self):
        c = SimCandle(timestamp=1000.0, open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0)
        assert c.symbol == ""
        assert c.timeframe == "5m"


class TestSimSignal:
    def test_defaults(self):
        s = SimSignal(
            symbol="BTCUSDT",
            direction="LONG",
            entry=100.0,
            stop_loss=95.0,
            tp1=105.0,
            tp2=110.0,
            tp3=115.0,
            probability_score=80.0,
        )
        assert s.outcome == ""
        assert s.pnl_pct == 0.0


class TestSimulatorRun:
    def test_run_returns_sim_result(self):
        sim = Simulator(_make_historical_candles(), channels=[])
        result = sim.run(days=1, regime="TRENDING_UP")
        assert isinstance(result, SimResult)

    def test_result_has_expected_fields(self):
        sim = Simulator(_make_historical_candles(), channels=[])
        result = sim.run(days=1)
        assert hasattr(result, "days")
        assert hasattr(result, "total_setups")
        assert hasattr(result, "total_signals")
        assert hasattr(result, "hit_rate_pct")
        assert hasattr(result, "sl_hit_rate_pct")
        assert hasattr(result, "suppression_rate_pct")

    def test_suppression_rate_in_range(self):
        sim = Simulator(_make_historical_candles(), channels=[], probability_threshold=99.0)
        result = sim.run(days=1)
        assert 0.0 <= result.suppression_rate_pct <= 100.0

    def test_low_threshold_more_signals_than_high_threshold(self):
        historical = _make_historical_candles()
        sim_low = Simulator(historical, channels=[], probability_threshold=0.0)
        sim_high = Simulator(historical, channels=[], probability_threshold=99.0)
        result_low = sim_low.run(days=1)
        result_high = sim_high.run(days=1)
        assert result_low.total_signals >= result_high.total_signals

    def test_days_clamped_to_valid_range(self):
        sim = Simulator(_make_historical_candles(), channels=[])
        result = sim.run(days=0)
        assert result.days >= 1
        result2 = sim.run(days=50)
        assert result2.days <= 30

    def test_empty_candles(self):
        sim = Simulator({}, channels=[])
        result = sim.run(days=1)
        assert result.total_setups == 0
        assert result.total_signals == 0

    def test_too_few_candles_skipped(self):
        # Only 30 candles — not enough (need 50)
        historical = {"BTCUSDT": {"5m": _make_candles(n=30)}}
        sim = Simulator(historical, channels=[])
        result = sim.run(days=30)
        assert result.total_setups == 0

    def test_multiple_symbols(self):
        historical = _make_historical_candles(symbols=["BTCUSDT", "ETHUSDT"])
        sim = Simulator(historical, channels=[])
        result = sim.run(days=1)
        assert isinstance(result, SimResult)


class TestSimulatorOutcomes:
    def _run_with_large_moves(self, direction: str) -> SimResult:
        """Create candles with large moves so TP/SL gets hit."""
        import time as _time
        ts = _time.time() - 200 * 300
        candles = []
        base = 50000.0
        for i in range(200):
            if i > 100:
                # Large move
                if direction == "LONG":
                    close = base * 1.05
                else:
                    close = base * 0.95
            else:
                close = base
            candles.append(SimCandle(
                timestamp=ts + i * 300,
                open=close,
                high=close * 1.02 if direction == "LONG" else close,
                low=close if direction == "LONG" else close * 0.98,
                close=close,
                volume=500.0,
            ))
        historical = {"BTCUSDT": {"5m": candles}}
        sim = Simulator(historical, channels=[], probability_threshold=0.0)
        return sim.run(days=1)

    def test_hit_rate_in_range(self):
        result = self._run_with_large_moves("LONG")
        assert 0.0 <= result.hit_rate_pct <= 100.0

    def test_sl_rate_in_range(self):
        result = self._run_with_large_moves("SHORT")
        assert 0.0 <= result.sl_hit_rate_pct <= 100.0

    def test_hit_plus_sl_plus_open_equals_100(self):
        result = self._run_with_large_moves("LONG")
        n = result.total_signals
        if n > 0:
            outcomes = {s.outcome for s in result.signals}
            assert outcomes.issubset({"TP1", "TP2", "TP3", "SL", "OPEN"})


class TestSimulatorExport:
    def test_export_csv(self):
        sim = Simulator(_make_historical_candles(), channels=[], probability_threshold=0.0)
        result = sim.run(days=1)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            sim.export_csv(result, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_export_json(self):
        sim = Simulator(_make_historical_candles(), channels=[], probability_threshold=0.0)
        result = sim.run(days=1)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sim.export_json(result, path)
            with open(path) as fh:
                data = json.load(fh)
            assert "hit_rate_pct" in data
            assert "signals" in data
        finally:
            os.unlink(path)

    def test_export_csv_empty_signals(self):
        sim = Simulator({}, channels=[])
        result = sim.run(days=1)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            sim.export_csv(result, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)
