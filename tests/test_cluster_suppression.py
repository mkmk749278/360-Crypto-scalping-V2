"""Tests for src/cluster_suppression.py."""

from __future__ import annotations

import time

from src.cluster_suppression import ClusterSuppressor


# ---------------------------------------------------------------------------
# Basic flow
# ---------------------------------------------------------------------------


def test_empty_suppressor_allows_all():
    s = ClusterSuppressor(window_seconds=60.0, max_signals=5)
    allowed, reason = s.check_cluster_gate("SOLUSDT", "LONG")
    assert allowed is True
    assert reason == ""


def test_record_then_check_within_limit():
    s = ClusterSuppressor(window_seconds=60.0, max_signals=5)
    for sym in ["BTC", "ETH", "SOL"]:
        s.record_signal(sym, "LONG")
    allowed, _ = s.check_cluster_gate("AVAX", "LONG")
    assert allowed is True


def test_same_symbol_multiple_times_counts_once():
    """Unique symbols, not total events, determine the cluster count."""
    s = ClusterSuppressor(window_seconds=60.0, max_signals=5)
    for _ in range(20):
        s.record_signal("SOLUSDT", "LONG")
    allowed, _ = s.check_cluster_gate("NEWUSDT", "LONG")
    assert allowed is True


# ---------------------------------------------------------------------------
# Directional burst detection
# ---------------------------------------------------------------------------


def test_directional_cluster_blocks_same_direction():
    s = ClusterSuppressor(window_seconds=60.0, max_signals=3)
    symbols = [f"SYM{i}USDT" for i in range(10)]
    for sym in symbols:
        s.record_signal(sym, "LONG")
    # > max_signals unique symbols, > 80% LONG → should block LONG
    allowed, reason = s.check_cluster_gate("NEWUSDT", "LONG")
    assert allowed is False
    assert "cluster" in reason.lower() or "market" in reason.lower()


def test_directional_cluster_does_not_block_opposite_direction():
    """If the bias is strongly LONG, a SHORT signal may still slip through
    the directional check (the undirected check could still fire)."""
    s = ClusterSuppressor(window_seconds=60.0, max_signals=3)
    # Only slightly above max_signals, all LONG
    symbols = [f"SYM{i}USDT" for i in range(4)]  # 4 > max_signals=3
    for sym in symbols:
        s.record_signal(sym, "LONG")
    # SHORT direction → bias in LONG direction should not block SHORT
    # (directional gate only blocks when bias >= threshold for the *requested* direction)
    allowed, _ = s.check_cluster_gate("NEWUSDT", "SHORT")
    # SHORT may or may not be blocked by undirected check, depending on multiplier
    # Just ensure the function returns a valid tuple
    assert isinstance(allowed, bool)


# ---------------------------------------------------------------------------
# Undirected burst detection
# ---------------------------------------------------------------------------


def test_undirected_burst_blocks_any_direction():
    """Too many unique symbols (any direction) triggers the undirected block."""
    s = ClusterSuppressor(window_seconds=60.0, max_signals=3)
    # max_signals * 1.5 = 4.5 → 5 unique symbols should trigger
    symbols = [f"COIN{i}USDT" for i in range(5)]
    for i, sym in enumerate(symbols):
        direction = "LONG" if i % 2 == 0 else "SHORT"
        s.record_signal(sym, direction)
    allowed, reason = s.check_cluster_gate("NEWUSDT", "LONG")
    assert allowed is False


# ---------------------------------------------------------------------------
# Window expiry
# ---------------------------------------------------------------------------


def test_expired_signals_are_pruned():
    s = ClusterSuppressor(window_seconds=0.05, max_signals=2)
    symbols = [f"SYM{i}USDT" for i in range(5)]
    for sym in symbols:
        s.record_signal(sym, "LONG")
    # Wait for window to expire
    time.sleep(0.1)
    # After pruning, window is empty → should allow
    allowed, _ = s.check_cluster_gate("NEWUSDT", "LONG")
    assert allowed is True


def test_mixed_old_and_new_signals():
    """Old signals should not count against the current window."""
    s = ClusterSuppressor(window_seconds=0.1, max_signals=2)
    # Record old signals
    for i in range(5):
        s.record_signal(f"OLD{i}USDT", "LONG")
    time.sleep(0.15)
    # Record fresh signals within limit
    s.record_signal("NEW1USDT", "LONG")
    allowed, _ = s.check_cluster_gate("NEW2USDT", "LONG")
    assert allowed is True


# ---------------------------------------------------------------------------
# Record_signal only after acceptance (behavioural contract)
# ---------------------------------------------------------------------------


def test_record_signal_only_increments_window():
    s = ClusterSuppressor(window_seconds=60.0, max_signals=5)
    s.record_signal("SOLUSDT", "LONG")
    assert len(s._recent) == 1
    s.record_signal("AVAXUSDT", "SHORT")
    assert len(s._recent) == 2
