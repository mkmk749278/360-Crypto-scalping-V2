"""Tests for src.scanner.common_gates — shared gate logic."""

from src.scanner.common_gates import (
    GateCheckResult,
    GateResult,
    apply_soft_gate_penalty,
    check_regime_compatibility,
    check_spread_gate,
    check_volume_gate,
    compute_regime_penalty_multiplier,
    run_common_gates,
)


def test_regime_compatibility_allowed():
    r = check_regime_compatibility("360_SCALP", "TRENDING_UP", {"360_SCALP_VWAP": ["QUIET"]})
    assert r.passed is True


def test_regime_compatibility_blocked():
    r = check_regime_compatibility("360_SCALP_VWAP", "QUIET", {"360_SCALP_VWAP": ["QUIET"]})
    assert r.passed is False


def test_spread_gate_pass():
    r = check_spread_gate(0.01, 0.02)
    assert r.passed is True


def test_spread_gate_fail():
    r = check_spread_gate(0.03, 0.02)
    assert r.passed is False


def test_spread_gate_quiet_relaxed():
    # Should pass in QUIET (spread tolerance relaxed by 50%)
    r = check_spread_gate(0.025, 0.02, regime="QUIET")
    assert r.passed is True


def test_volume_gate_pass():
    r = check_volume_gate(10_000_000.0, 5_000_000.0)
    assert r.passed is True


def test_volume_gate_fail():
    r = check_volume_gate(1_000_000.0, 5_000_000.0)
    assert r.passed is False


def test_regime_penalty_multiplier_scalp_quiet():
    m = compute_regime_penalty_multiplier("QUIET", "360_SCALP", {"QUIET": 0.8})
    assert m == 1.8  # Special SCALP QUIET penalty


def test_regime_penalty_multiplier_normal():
    m = compute_regime_penalty_multiplier("TRENDING_UP", "360_SWING", {"TRENDING_UP": 0.6})
    assert m == 0.6


def test_soft_gate_penalty_not_fired():
    r = apply_soft_gate_penalty("vwap", False, 12.0, 1.0)
    assert r.penalty == 0.0


def test_soft_gate_penalty_fired():
    r = apply_soft_gate_penalty("vwap", True, 12.0, 1.5)
    assert r.penalty == 18.0  # 12 * 1.5


def test_run_common_gates_all_pass():
    result = run_common_gates(
        channel="360_SCALP",
        regime="TRENDING_UP",
        spread_pct=0.005,
        volume_24h_usd=10_000_000.0,
        max_spread=0.02,
        min_volume=5_000_000.0,
    )
    assert result.passed is True
    assert result.total_penalty == 0.0


def test_run_common_gates_spread_fail():
    result = run_common_gates(
        channel="360_SCALP",
        regime="TRENDING_UP",
        spread_pct=0.05,
        volume_24h_usd=10_000_000.0,
        max_spread=0.02,
        min_volume=5_000_000.0,
    )
    assert result.passed is False
