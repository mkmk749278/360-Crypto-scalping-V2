"""Tests for logging and general utility helpers."""

from src.utils import _LoguroBridge


class _Recorder:
    def __init__(self) -> None:
        self.messages = []

    def info(self, message: str, **_kwargs) -> None:
        self.messages.append(message)


class TestLoguroBridge:
    def test_accepts_percent_style_formatting(self):
        recorder = _Recorder()
        logger = _LoguroBridge(recorder)

        logger.info("Pair %s moved %.1f%%", "BTCUSDT", 2.5)

        assert recorder.messages == ["Pair BTCUSDT moved 2.5%"]

    def test_accepts_brace_style_formatting(self):
        recorder = _Recorder()
        logger = _LoguroBridge(recorder)

        logger.info("Pair {} moved {:.1f}%", "ETHUSDT", 1.5)

        assert recorder.messages == ["Pair ETHUSDT moved 1.5%"]
