"""Tests for src.order_book – Order Book Imbalance (OBI) execution filter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.order_book import (
    OrderBookSnapshot,
    calculate_order_book_imbalance,
    check_order_book_execution,
)
from src.risk import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_book(bid_qty: float, ask_qty: float, price: float = 100.0, levels: int = 10):
    """Create a simple symmetric order book with identical levels."""
    bids = [[str(price), str(bid_qty)] for _ in range(levels)]
    asks = [[str(price), str(ask_qty)] for _ in range(levels)]
    return {"bids": bids, "asks": asks}


def _make_risk_signal(
    entry: float = 100.0,
    stop_loss: float = 97.0,
    tp1: float = 106.0,
    direction: str = "LONG",
    symbol: str = "BTCUSDT",
    order_book: object = None,
) -> SimpleNamespace:
    dir_ns = SimpleNamespace(value=direction)
    return SimpleNamespace(
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        direction=dir_ns,
        symbol=symbol,
        spread_pct=0.0,
        confidence=70.0,
        order_book=order_book,
    )


# ---------------------------------------------------------------------------
# calculate_order_book_imbalance
# ---------------------------------------------------------------------------


class TestCalculateOrderBookImbalance:
    def test_balanced_book(self):
        """Equal bid and ask volume → bid_pct == ask_pct == 0.5."""
        snap = calculate_order_book_imbalance(
            [["100.0", "10"]] * 5,
            [["100.0", "10"]] * 5,
        )
        assert snap is not None
        assert snap.bid_pct == pytest.approx(0.5)
        assert snap.ask_pct == pytest.approx(0.5)
        assert snap.dominant_side == "balanced"
        assert snap.imbalance_ratio == pytest.approx(1.0)

    def test_bid_heavy_book(self):
        """3× more bids than asks → bids dominate."""
        snap = calculate_order_book_imbalance(
            [["100.0", "30"]] * 5,
            [["100.0", "10"]] * 5,
        )
        assert snap is not None
        assert snap.dominant_side == "bids"
        assert snap.bid_pct == pytest.approx(0.75)
        assert snap.ask_pct == pytest.approx(0.25)
        assert snap.imbalance_ratio == pytest.approx(3.0)

    def test_ask_heavy_book(self):
        """3× more asks than bids → asks dominate."""
        snap = calculate_order_book_imbalance(
            [["100.0", "10"]] * 5,
            [["100.0", "30"]] * 5,
        )
        assert snap is not None
        assert snap.dominant_side == "asks"
        assert snap.bid_pct == pytest.approx(0.25)
        assert snap.ask_pct == pytest.approx(0.75)

    def test_empty_bids_returns_none(self):
        assert calculate_order_book_imbalance([], [["100.0", "10"]]) is None

    def test_empty_asks_returns_none(self):
        assert calculate_order_book_imbalance([["100.0", "10"]], []) is None

    def test_both_empty_returns_none(self):
        assert calculate_order_book_imbalance([], []) is None

    def test_malformed_entry_returns_none(self):
        """Non-numeric entries must not crash and should return None."""
        snap = calculate_order_book_imbalance(
            [["bad", "data"]],
            [["100.0", "10"]],
        )
        assert snap is None

    def test_levels_parameter_limits_depth(self):
        """Only the first *levels* entries from each side should be used."""
        # 20 entries per side, but only 5 should be consumed
        bids = [["100.0", "10"]] * 20
        asks = [["100.0", "10"]] * 20
        snap_full = calculate_order_book_imbalance(bids, asks, levels=20)
        snap_partial = calculate_order_book_imbalance(bids, asks, levels=5)
        assert snap_full is not None
        assert snap_partial is not None
        # Ratios should be the same because all levels are symmetric,
        # but volumes should differ.
        assert snap_full.total_volume > snap_partial.total_volume
        assert snap_full.bid_pct == pytest.approx(snap_partial.bid_pct)

    def test_returns_frozen_dataclass(self):
        snap = calculate_order_book_imbalance(
            [["100.0", "10"]],
            [["100.0", "10"]],
        )
        assert isinstance(snap, OrderBookSnapshot)
        with pytest.raises((AttributeError, TypeError)):
            snap.bid_pct = 0.99  # type: ignore[misc]

    def test_usd_weighted_volume(self):
        """Volume = price × qty, so higher-priced levels contribute more."""
        bids = [["200.0", "5"]]   # USD vol = 1000
        asks = [["100.0", "5"]]   # USD vol = 500
        snap = calculate_order_book_imbalance(bids, asks)
        assert snap is not None
        assert snap.bid_volume == pytest.approx(1000.0)
        assert snap.ask_volume == pytest.approx(500.0)
        assert snap.dominant_side == "bids"


# ---------------------------------------------------------------------------
# check_order_book_execution
# ---------------------------------------------------------------------------


class TestCheckOrderBookExecution:
    # ------------------------------------------------------------------
    # Graceful fallback (fail open)
    # ------------------------------------------------------------------

    def test_none_order_book_allows_trade(self):
        """None order book → fail open, trade allowed."""
        allowed, reason = check_order_book_execution("LONG", None)
        assert allowed is True
        assert reason == ""

    def test_missing_bids_key_allows_trade(self):
        allowed, reason = check_order_book_execution("LONG", {"asks": [["100", "10"]]})
        assert allowed is True

    def test_missing_asks_key_allows_trade(self):
        allowed, reason = check_order_book_execution("SHORT", {"bids": [["100", "10"]]})
        assert allowed is True

    def test_empty_bids_and_asks_allows_trade(self):
        """Empty lists → snapshot is None → fail open."""
        allowed, reason = check_order_book_execution("LONG", {"bids": [], "asks": []})
        assert allowed is True
        assert reason == ""

    def test_malformed_data_allows_trade(self):
        """Malformed book data must not raise and must fail open."""
        allowed, reason = check_order_book_execution(
            "LONG", {"bids": [["bad", "data"]], "asks": [["100", "10"]]}
        )
        assert allowed is True

    # ------------------------------------------------------------------
    # LONG rejection (heavy ask wall)
    # ------------------------------------------------------------------

    def test_long_rejected_when_asks_exceed_threshold(self):
        """Asks at 70% of book → LONG blocked (above 65% default)."""
        book = _make_book(bid_qty=3.0, ask_qty=7.0)  # asks = 70%, bids = 30%
        allowed, reason = check_order_book_execution("LONG", book)
        assert allowed is False
        assert "LONG blocked" in reason
        assert "OBI" in reason

    def test_long_allowed_when_asks_below_threshold(self):
        """Asks at 60% → below the 65% threshold, LONG allowed."""
        book = _make_book(bid_qty=4.0, ask_qty=6.0)  # asks = 60%
        allowed, reason = check_order_book_execution("LONG", book)
        assert allowed is True
        assert reason == ""

    def test_long_allowed_when_bids_dominate(self):
        """Bids at 70% → very bullish, LONG always allowed."""
        book = _make_book(bid_qty=7.0, ask_qty=3.0)
        allowed, reason = check_order_book_execution("LONG", book)
        assert allowed is True

    # ------------------------------------------------------------------
    # SHORT rejection (heavy bid wall)
    # ------------------------------------------------------------------

    def test_short_rejected_when_bids_exceed_threshold(self):
        """Bids at 70% → SHORT blocked (above 65% default)."""
        book = _make_book(bid_qty=7.0, ask_qty=3.0)
        allowed, reason = check_order_book_execution("SHORT", book)
        assert allowed is False
        assert "SHORT blocked" in reason
        assert "OBI" in reason

    def test_short_allowed_when_bids_below_threshold(self):
        """Bids at 60% → below threshold, SHORT allowed."""
        book = _make_book(bid_qty=6.0, ask_qty=4.0)
        allowed, reason = check_order_book_execution("SHORT", book)
        assert allowed is True

    def test_short_allowed_when_asks_dominate(self):
        """Asks at 70% → bearish, SHORT always allowed."""
        book = _make_book(bid_qty=3.0, ask_qty=7.0)
        allowed, reason = check_order_book_execution("SHORT", book)
        assert allowed is True

    # ------------------------------------------------------------------
    # Balanced book
    # ------------------------------------------------------------------

    def test_balanced_book_long_allowed(self):
        """50-50 book → well below any rejection threshold, LONG allowed."""
        book = _make_book(bid_qty=5.0, ask_qty=5.0)
        assert check_order_book_execution("LONG", book)[0] is True

    def test_balanced_book_short_allowed(self):
        book = _make_book(bid_qty=5.0, ask_qty=5.0)
        assert check_order_book_execution("SHORT", book)[0] is True

    # ------------------------------------------------------------------
    # Custom threshold
    # ------------------------------------------------------------------

    def test_custom_rejection_threshold_tighter(self):
        """With a tighter 55% threshold, a 60% ask wall blocks LONG."""
        book = _make_book(bid_qty=4.0, ask_qty=6.0)  # asks = 60%
        allowed, _ = check_order_book_execution("LONG", book, rejection_threshold=0.55)
        assert allowed is False

    def test_custom_rejection_threshold_looser(self):
        """With a looser 80% threshold, a 70% ask wall allows LONG."""
        book = _make_book(bid_qty=3.0, ask_qty=7.0)  # asks = 70%
        allowed, _ = check_order_book_execution("LONG", book, rejection_threshold=0.80)
        assert allowed is True

    # ------------------------------------------------------------------
    # Reason string content
    # ------------------------------------------------------------------

    def test_reason_contains_percentages(self):
        book = _make_book(bid_qty=3.0, ask_qty=7.0)
        _, reason = check_order_book_execution("LONG", book)
        assert "70%" in reason
        assert "65%" in reason

    def test_allowed_reason_is_empty(self):
        book = _make_book(bid_qty=5.0, ask_qty=5.0)
        _, reason = check_order_book_execution("LONG", book)
        assert reason == ""


# ---------------------------------------------------------------------------
# Integration: RiskManager respects the OBI filter
# ---------------------------------------------------------------------------


class TestRiskManagerOBIIntegration:
    """Verify that RiskManager.calculate_risk applies the OBI filter from signal."""

    def setup_method(self):
        self.rm = RiskManager()

    def test_trade_allowed_when_no_order_book(self):
        """Signals without order book data are allowed (fail open)."""
        sig = _make_risk_signal(order_book=None)
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is True

    def test_trade_blocked_by_ask_wall_for_long(self):
        """Heavy ask wall blocks LONG via the OBI filter in RiskManager."""
        # 70% ask volume → above 65% threshold
        ob = _make_book(bid_qty=3.0, ask_qty=7.0)
        sig = _make_risk_signal(direction="LONG", order_book=ob)
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is False
        assert "OBI" in result.reason
        assert "LONG blocked" in result.reason

    def test_trade_blocked_by_bid_wall_for_short(self):
        """Heavy bid wall blocks SHORT via the OBI filter in RiskManager."""
        ob = _make_book(bid_qty=7.0, ask_qty=3.0)
        sig = _make_risk_signal(direction="SHORT", order_book=ob)
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is False
        assert "OBI" in result.reason
        assert "SHORT blocked" in result.reason

    def test_trade_allowed_when_book_supports_direction(self):
        """Bid-heavy book allows LONG."""
        ob = _make_book(bid_qty=7.0, ask_qty=3.0)
        sig = _make_risk_signal(direction="LONG", order_book=ob)
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is True

    def test_rr_rejection_takes_priority_over_obi(self):
        """Insufficient R:R must still reject even if OBI would allow."""
        # R:R = 1.0 < 1.2 floor, order book supports LONG
        ob = _make_book(bid_qty=7.0, ask_qty=3.0)
        sig = _make_risk_signal(
            entry=100.0, stop_loss=97.0, tp1=103.0,  # R:R = 1.0
            direction="LONG", order_book=ob,
        )
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is False
        assert "R:R" in result.reason  # R:R rejection reason, not OBI

    def test_obi_only_applied_when_rr_passes(self):
        """OBI check is only reached after R:R passes."""
        # Good R:R but heavy ask wall
        ob = _make_book(bid_qty=3.0, ask_qty=7.0)
        sig = _make_risk_signal(
            entry=100.0, stop_loss=97.0, tp1=106.0,  # R:R = 2.0
            direction="LONG", order_book=ob,
        )
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is False
        assert "OBI" in result.reason

    def test_signal_without_order_book_attribute_allowed(self):
        """Signals without an order_book attribute at all must be allowed (fail open)."""
        sig = SimpleNamespace(
            entry=100.0,
            stop_loss=97.0,
            tp1=106.0,
            direction=SimpleNamespace(value="LONG"),
            symbol="BTCUSDT",
            spread_pct=0.0,
            confidence=70.0,
            # no 'order_book' attribute
        )
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is True
