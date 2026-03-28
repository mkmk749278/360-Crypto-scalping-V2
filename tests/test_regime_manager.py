"""Tests for src.scanner.regime_manager — regime-adaptive scheduling."""

from src.scanner.regime_manager import RegimeManager, RegimeSchedule


class _FakeChannel:
    def __init__(self, name: str):
        self.config = type("Config", (), {"name": name})()


def test_regime_schedule_trending():
    rm = RegimeManager()
    sched = rm.get_schedule("TRENDING_UP")
    assert "360_SCALP" in sched.allowed_channels
    assert "360_SWING" in sched.allowed_channels


def test_regime_schedule_volatile():
    rm = RegimeManager()
    sched = rm.get_schedule("VOLATILE")
    assert "360_SWING" not in sched.allowed_channels  # Blocked in VOLATILE


def test_is_channel_allowed():
    rm = RegimeManager()
    assert rm.is_channel_allowed("360_SCALP", "TRENDING_UP")
    assert not rm.is_channel_allowed("360_SWING", "VOLATILE")


def test_is_channel_priority():
    rm = RegimeManager()
    assert rm.is_channel_priority("360_SCALP", "TRENDING_UP")
    assert not rm.is_channel_priority("360_SPOT", "TRENDING_UP")


def test_filter_channels_trending():
    rm = RegimeManager()
    channels = [_FakeChannel("360_SCALP"), _FakeChannel("360_SWING"), _FakeChannel("360_SPOT")]
    allowed, skipped = rm.filter_channels(channels, "TRENDING_UP")
    names = [c.config.name for c in allowed]
    assert "360_SCALP" in names
    assert "360_SWING" in names
    assert len(skipped) == 0


def test_filter_channels_volatile_skips_swing():
    rm = RegimeManager()
    channels = [_FakeChannel("360_SCALP"), _FakeChannel("360_SWING"), _FakeChannel("360_SPOT")]
    allowed, skipped = rm.filter_channels(channels, "VOLATILE")
    assert "360_SWING" in skipped


def test_filter_channels_priority_first():
    rm = RegimeManager()
    channels = [_FakeChannel("360_SPOT"), _FakeChannel("360_SCALP")]
    allowed, skipped = rm.filter_channels(channels, "TRENDING_UP")
    # Priority channels should come first
    assert allowed[0].config.name == "360_SCALP"


def test_skip_stats():
    rm = RegimeManager()
    channels = [_FakeChannel("360_SWING")]
    rm.filter_channels(channels, "VOLATILE")
    rm.filter_channels(channels, "VOLATILE")
    stats = rm.get_skip_stats()
    assert stats.get("360_SWING", 0) == 2


def test_unknown_regime_allows_all():
    rm = RegimeManager()
    sched = rm.get_schedule("UNKNOWN_REGIME")
    assert "360_SCALP" in sched.allowed_channels
