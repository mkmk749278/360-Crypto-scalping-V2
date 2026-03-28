"""Tests for src.api_limits — API rate limit management."""

import time

from src.api_limits import APIWeightTracker, BatchScheduler, check_rate_limit


def test_api_weight_tracker_basic():
    t = APIWeightTracker()
    assert t.current_weight == 0
    assert t.can_make_request


def test_api_weight_tracker_record():
    t = APIWeightTracker()
    for _ in range(10):
        t.record(weight=5)
    assert t.current_weight == 50
    assert t.can_make_request


def test_api_weight_tracker_at_limit():
    t = APIWeightTracker()
    t._safety = 100
    for _ in range(20):
        t.record(weight=5)
    assert t.current_weight == 100
    assert not t.can_make_request


def test_api_weight_tracker_usage_pct():
    t = APIWeightTracker()
    t.record(weight=600)
    assert t.usage_pct == 50.0


def test_batch_scheduler_futures():
    bs = BatchScheduler()
    futures = [f"PAIR{i}" for i in range(200)]
    rt = bs.get_futures_realtime_pairs(futures)
    assert len(rt) == 100  # TOP_FUTURES_REALTIME_COUNT default


def test_batch_scheduler_spot_rotation():
    bs = BatchScheduler()
    bs._spot_batch_size = 5
    spot = [f"SPOT{i}" for i in range(20)]
    b1 = bs.get_spot_batch(spot)
    assert len(b1) == 5
    assert b1[0] == "SPOT0"
    b2 = bs.get_spot_batch(spot)
    assert b2[0] == "SPOT5"


def test_batch_scheduler_spot_wrap():
    bs = BatchScheduler()
    bs._spot_batch_size = 10
    spot = [f"SPOT{i}" for i in range(15)]
    bs.get_spot_batch(spot)  # 0-9
    b2 = bs.get_spot_batch(spot)  # 10-14
    assert len(b2) == 5
    b3 = bs.get_spot_batch(spot)  # wraps to 0-9
    assert b3[0] == "SPOT0"


def test_check_rate_limit_ok():
    t = APIWeightTracker()
    assert check_rate_limit(t) is True


def test_check_rate_limit_exceeded():
    t = APIWeightTracker()
    t._safety = 10
    t.record(weight=10)
    assert check_rate_limit(t) is False
