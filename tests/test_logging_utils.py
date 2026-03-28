"""Tests for src.logging_utils — telemetry & logging enhancements."""

from src.logging_utils import LatencyMonitor, SuppressionLogEntry, SuppressionLogger


def test_suppression_logger_basic():
    sl = SuppressionLogger()
    sl.log_suppressed_signal("BTCUSDT", "360_SCALP", "regime", 45.0)
    assert sl.total_suppressed == 1


def test_suppression_logger_stats_by_reason():
    sl = SuppressionLogger()
    sl.log_suppressed_signal("BTCUSDT", "360_SCALP", "regime", 45.0)
    sl.log_suppressed_signal("ETHUSDT", "360_SCALP", "volume", 30.0)
    sl.log_suppressed_signal("BNBUSDT", "360_SWING", "regime", 55.0)
    stats = sl.get_stats_by_reason()
    assert stats["regime"] == 2
    assert stats["volume"] == 1


def test_suppression_logger_top_pairs():
    sl = SuppressionLogger()
    for _ in range(5):
        sl.log_suppressed_signal("BTCUSDT", "360_SCALP", "regime")
    for _ in range(3):
        sl.log_suppressed_signal("ETHUSDT", "360_SCALP", "regime")
    top = sl.get_top_suppressed_pairs(limit=2)
    assert top[0] == ("BTCUSDT", 5)
    assert top[1] == ("ETHUSDT", 3)


def test_suppression_logger_max_entries():
    sl = SuppressionLogger(max_entries=10)
    for i in range(20):
        sl.log_suppressed_signal(f"PAIR{i}", "360_SCALP", "test")
    entries = sl.get_recent_entries(count=100)
    assert len(entries) == 10


def test_suppression_logger_telemetry_summary():
    sl = SuppressionLogger()
    sl.log_suppressed_signal("BTCUSDT", "360_SCALP", "regime", 45.0)
    summary = sl.format_telemetry_summary()
    assert "total=1" in summary
    assert "regime" in summary


def test_suppression_log_entry_format():
    entry = SuppressionLogEntry(
        pair="BTCUSDT", channel="360_SCALP", reason="regime",
        probability_score=45.0, regime="QUIET", threshold=70.0,
    )
    fmt = entry.format_log()
    assert "BTCUSDT" in fmt
    assert "regime" in fmt
    assert "45.0" in fmt


def test_latency_monitor_record():
    lm = LatencyMonitor()
    lm.record("scan_loop", 3000.0)
    assert lm.get_average("scan_loop") == 3000.0


def test_latency_monitor_p95():
    lm = LatencyMonitor()
    for i in range(100):
        lm.record("scan_loop", float(i * 100))
    p95 = lm.get_p95("scan_loop")
    assert p95 >= 9000.0  # ~95th percentile of 0..9900
