"""Tests for src.scanner.ws_optimizer — WS latency optimization."""

from src.scanner.ws_optimizer import (
    LatencyTracker,
    ShardHealth,
    compute_reconnect_delay,
    score_shard_health,
    select_priority_pairs,
)


def test_score_shard_health_perfect():
    score = score_shard_health(
        last_pong_age_s=5.0,
        ping_latency_ms=20.0,
        message_rate=10.0,
        reconnect_attempts=0,
    )
    assert score >= 80.0


def test_score_shard_health_degraded():
    score = score_shard_health(
        last_pong_age_s=300.0,
        ping_latency_ms=600.0,
        message_rate=0.1,
        reconnect_attempts=10,
    )
    assert score < 30.0


def test_score_shard_health_range():
    score = score_shard_health(0, 0, 0, 0)
    assert 0.0 <= score <= 100.0


def test_latency_tracker_basic():
    lt = LatencyTracker()
    lt.record(5000.0)
    assert lt.last_ms == 5000.0
    assert lt.average_ms == 5000.0
    assert not lt.is_high_latency


def test_latency_tracker_high_latency():
    lt = LatencyTracker()
    for _ in range(5):
        lt.record(20000.0)
    assert lt.is_high_latency
    assert lt.should_skip_low_priority()


def test_latency_tracker_critical():
    lt = LatencyTracker()
    lt.record(35000.0)
    assert lt.is_critical_latency


def test_latency_tracker_recommended_limit():
    lt = LatencyTracker()
    lt.record(5000.0)
    assert lt.get_recommended_pair_limit(200) == 200

    lt.record(35000.0)
    assert lt.get_recommended_pair_limit(200) < 200


def test_compute_reconnect_delay():
    d0 = compute_reconnect_delay(0)
    d1 = compute_reconnect_delay(1)
    d5 = compute_reconnect_delay(5)
    assert d0 > 0
    assert d5 > d0  # Generally increases (ignoring jitter)
    assert compute_reconnect_delay(100) <= 60.0 + 60.0 * 0.25  # Capped at max + jitter


def test_select_priority_pairs_normal():
    lt = LatencyTracker()
    lt.record(5000.0)
    all_p = ["A", "B", "C", "D", "E"]
    tier1 = ["A", "B"]
    result = select_priority_pairs(all_p, tier1, lt)
    assert result == all_p


def test_select_priority_pairs_critical():
    lt = LatencyTracker()
    lt.record(35000.0)
    all_p = ["A", "B", "C", "D", "E"]
    tier1 = ["A", "B"]
    result = select_priority_pairs(all_p, tier1, lt)
    assert result == tier1
