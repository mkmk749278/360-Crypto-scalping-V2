"""Tests for PR04 – API Rate-Limit Aware Batch Scheduling Helpers."""

from __future__ import annotations

import pytest

from src.scanner.api_limits import (
    BatchScheduler,
    DEFAULT_SPOT_WINDOW_MINUTES,
    TOP_FUTURES_REALTIME_COUNT,
    log_api_usage,
    should_scan_spot_pair,
)


class TestBatchSchedulerBasics:
    def test_get_batch_empty_list(self):
        sched = BatchScheduler()
        assert sched.get_batch([]) == []

    def test_assign_buckets_sets_n_buckets(self):
        sched = BatchScheduler()
        pairs = [f"TOKEN{i}USDT" for i in range(30)]
        sched.assign_buckets(pairs, n_buckets=3)
        assert sched._n_buckets == 3

    def test_get_batch_returns_subset(self):
        sched = BatchScheduler()
        pairs = [f"TOKEN{i}USDT" for i in range(30)]
        sched.assign_buckets(pairs, n_buckets=3)
        batch = sched.get_batch(pairs)
        assert 0 < len(batch) <= len(pairs)

    def test_rotation_covers_all_pairs(self):
        """All pairs should be covered after n_buckets cycles."""
        sched = BatchScheduler()
        pairs = [f"TOKEN{i}USDT" for i in range(30)]
        sched.assign_buckets(pairs, n_buckets=3)
        seen = set()
        for _ in range(3):
            batch = sched.get_batch(pairs)
            seen.update(batch)
        assert seen == set(pairs)

    def test_auto_bucket_assignment(self):
        sched = BatchScheduler()
        pairs = [f"TOKEN{i}USDT" for i in range(100)]
        sched.assign_buckets(pairs)  # no n_buckets → auto
        assert sched._n_buckets >= 1
        assert sched._n_buckets <= len(pairs)

    def test_stats(self):
        sched = BatchScheduler()
        sched.assign_buckets(["BTCUSDT", "ETHUSDT"], n_buckets=2)
        sched.get_batch(["BTCUSDT", "ETHUSDT"])
        stats = sched.stats
        assert stats["total_cycles"] == 1
        assert stats["skipped_cycles"] == 0

    def test_skip_cycle_increments_counter(self):
        sched = BatchScheduler()
        sched.get_batch([])  # cycle but empty
        sched.skip_cycle()
        assert sched.stats["skipped_cycles"] == 1

    def test_single_pair_universe(self):
        sched = BatchScheduler()
        sched.assign_buckets(["BTCUSDT"], n_buckets=1)
        batch = sched.get_batch(["BTCUSDT"])
        assert batch == ["BTCUSDT"]

    def test_n_buckets_one(self):
        sched = BatchScheduler()
        pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        sched.assign_buckets(pairs, n_buckets=1)
        batch = sched.get_batch(pairs)
        assert set(batch) == set(pairs)

    def test_assign_empty_list(self):
        sched = BatchScheduler()
        sched.assign_buckets([])  # Should not raise


class TestShouldScanSpotPair:
    def test_pair_in_batch(self):
        assert should_scan_spot_pair("BTCUSDT", ["BTCUSDT", "ETHUSDT"]) is True

    def test_pair_not_in_batch(self):
        assert should_scan_spot_pair("SOLUSDT", ["BTCUSDT", "ETHUSDT"]) is False

    def test_empty_batch(self):
        assert should_scan_spot_pair("BTCUSDT", []) is False


class TestLogApiUsage:
    def test_does_not_raise(self):
        # Just ensure the function runs without exception
        log_api_usage(
            cycle=1,
            spot_budget_used=600,
            futures_budget_used=400,
            spot_budget_total=1200,
            futures_budget_total=1200,
            pairs_scanned=50,
            elapsed_ms=1500.0,
        )

    def test_high_usage_does_not_raise(self):
        log_api_usage(
            cycle=5,
            spot_budget_used=1100,
            futures_budget_used=900,
            spot_budget_total=1200,
            futures_budget_total=1200,
            pairs_scanned=200,
            elapsed_ms=3000.0,
        )

    def test_zero_totals_does_not_raise(self):
        log_api_usage(
            cycle=1,
            spot_budget_used=0,
            futures_budget_used=0,
            spot_budget_total=0,
            futures_budget_total=0,
            pairs_scanned=0,
            elapsed_ms=0.0,
        )


class TestConstants:
    def test_default_window(self):
        assert DEFAULT_SPOT_WINDOW_MINUTES == 60.0

    def test_top_futures_count(self):
        assert TOP_FUTURES_REALTIME_COUNT == 100
