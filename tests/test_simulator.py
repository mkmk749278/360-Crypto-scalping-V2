"""Tests for src.simulation.simulator — historical replay."""

import json

from src.simulation.simulator import (
    SimulatedSignal,
    SimulationConfig,
    SimulationResult,
    Simulator,
)


def test_simulator_basic_signal():
    sim = Simulator()
    sig = sim.simulate_signal(
        symbol="BTCUSDT", channel="360_SCALP", direction="LONG",
        entry_price=50000.0, stop_loss=49800.0, tp1=50200.0, tp2=50400.0,
        probability_score=80.0, confidence=75.0,
    )
    assert sig is not None
    assert sig.symbol == "BTCUSDT"


def test_simulator_suppressed_signal():
    sim = Simulator(SimulationConfig(probability_threshold=80.0))
    sig = sim.simulate_signal(
        symbol="BTCUSDT", channel="360_SCALP", direction="LONG",
        entry_price=50000.0, stop_loss=49800.0, tp1=50200.0, tp2=50400.0,
        probability_score=60.0,
    )
    assert sig is None
    result = sim.get_result()
    assert result.suppressed_count == 1


def test_evaluate_outcome_tp1():
    sim = Simulator()
    sig = SimulatedSignal(
        symbol="BTCUSDT", channel="360_SCALP", direction="LONG",
        entry_price=50000.0, stop_loss=49800.0,
        tp1=50200.0, tp2=50400.0, tp3=50600.0,
    )
    prices = [50050.0, 50100.0, 50150.0, 50210.0]
    sim.evaluate_outcome(sig, prices)
    assert sig.outcome == "TP1_HIT"
    assert sig.pnl_pct > 0


def test_evaluate_outcome_sl():
    sim = Simulator()
    sig = SimulatedSignal(
        symbol="BTCUSDT", channel="360_SCALP", direction="LONG",
        entry_price=50000.0, stop_loss=49800.0,
        tp1=50200.0, tp2=50400.0,
    )
    prices = [49900.0, 49750.0]
    sim.evaluate_outcome(sig, prices)
    assert sig.outcome == "SL_HIT"
    assert sig.pnl_pct < 0


def test_evaluate_outcome_short():
    sim = Simulator()
    sig = SimulatedSignal(
        symbol="ETHUSDT", channel="360_SCALP", direction="SHORT",
        entry_price=3000.0, stop_loss=3050.0,
        tp1=2950.0, tp2=2900.0,
    )
    prices = [2980.0, 2960.0, 2940.0]
    sim.evaluate_outcome(sig, prices)
    assert sig.outcome == "TP1_HIT"
    assert sig.pnl_pct > 0


def test_evaluate_outcome_expired():
    sim = Simulator()
    sig = SimulatedSignal(
        symbol="BTCUSDT", channel="360_SCALP", direction="LONG",
        entry_price=50000.0, stop_loss=49800.0,
        tp1=50200.0, tp2=50400.0,
    )
    prices = [50010.0, 50020.0, 50015.0]
    sim.evaluate_outcome(sig, prices)
    assert sig.outcome == "EXPIRED"


def test_simulation_result_stats():
    result = SimulationResult(config=SimulationConfig())
    result.signals = [
        SimulatedSignal(symbol="A", channel="X", direction="LONG",
                       entry_price=100, stop_loss=95, tp1=105, tp2=110,
                       outcome="TP1_HIT", pnl_pct=5.0, hold_duration_s=300),
        SimulatedSignal(symbol="B", channel="X", direction="LONG",
                       entry_price=100, stop_loss=95, tp1=105, tp2=110,
                       outcome="SL_HIT", pnl_pct=-5.0, hold_duration_s=60),
    ]
    result.compute_stats()
    assert result.total_signals == 2
    assert result.tp1_hits == 1
    assert result.sl_hits == 1
    assert result.win_rate == 50.0


def test_export_csv():
    result = SimulationResult(config=SimulationConfig())
    result.signals = [
        SimulatedSignal(symbol="BTCUSDT", channel="360_SCALP", direction="LONG",
                       entry_price=50000, stop_loss=49800, tp1=50200, tp2=50400,
                       outcome="TP1_HIT", pnl_pct=0.4),
    ]
    csv_str = Simulator.export_csv(result)
    assert "BTCUSDT" in csv_str
    assert "TP1_HIT" in csv_str


def test_export_json():
    result = SimulationResult(config=SimulationConfig())
    result.signals = [
        SimulatedSignal(symbol="BTCUSDT", channel="360_SCALP", direction="LONG",
                       entry_price=50000, stop_loss=49800, tp1=50200, tp2=50400,
                       outcome="TP1_HIT", pnl_pct=0.4),
    ]
    result.compute_stats()
    json_str = Simulator.export_json(result)
    data = json.loads(json_str)
    assert data["summary"]["total_signals"] == 1
    assert len(data["signals"]) == 1


def test_simulator_reset():
    sim = Simulator()
    sim.simulate_signal(
        symbol="BTCUSDT", channel="360_SCALP", direction="LONG",
        entry_price=50000.0, stop_loss=49800.0, tp1=50200.0, tp2=50400.0,
        probability_score=80.0,
    )
    sim.reset()
    result = sim.get_result()
    assert result.total_signals == 0


def test_evaluate_outcome_empty_prices():
    sim = Simulator()
    sig = SimulatedSignal(
        symbol="BTCUSDT", channel="360_SCALP", direction="LONG",
        entry_price=50000.0, stop_loss=49800.0,
        tp1=50200.0, tp2=50400.0,
    )
    sim.evaluate_outcome(sig, [])
    assert sig.outcome == "EXPIRED"
