"""Tests for src/spoof_detect.py."""

from __future__ import annotations

from src.spoof_detect import (
    LAYER_CONCENTRATION_THRESHOLD,
    MIN_LEVELS_FOR_ANALYSIS,
    WALL_TOP_N,
    check_spoof_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ob(
    bid_qtys: list[float],
    ask_qtys: list[float],
    base_price: float = 100.0,
) -> dict:
    """Build a synthetic order book from lists of quantities."""
    bids = [[base_price - i * 0.01, q] for i, q in enumerate(bid_qtys)]
    asks = [[base_price + (i + 1) * 0.01, q] for i, q in enumerate(ask_qtys)]
    return {"bids": bids, "asks": asks}


# ---------------------------------------------------------------------------
# Fail-open cases
# ---------------------------------------------------------------------------


def test_none_order_book_returns_allowed():
    allowed, reason = check_spoof_gate("LONG", None, 100.0)
    assert allowed is True
    assert reason == ""


def test_empty_order_book_returns_allowed():
    ob = {"bids": [], "asks": []}
    allowed, reason = check_spoof_gate("LONG", ob, 100.0)
    assert allowed is True


def test_thin_order_book_below_min_levels_returns_allowed():
    """Fewer levels than MIN_LEVELS_FOR_ANALYSIS → no analysis → allowed."""
    ob = _make_ob(bid_qtys=[1.0, 1.0], ask_qtys=[1.0, 1.0])
    assert len(ob["bids"]) < MIN_LEVELS_FOR_ANALYSIS
    allowed, reason = check_spoof_gate("LONG", ob, 100.0)
    assert allowed is True


# ---------------------------------------------------------------------------
# Normal balanced book
# ---------------------------------------------------------------------------


def test_balanced_book_allowed_long():
    """Balanced bids and asks should pass for LONG."""
    ob = _make_ob(
        bid_qtys=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ask_qtys=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    )
    allowed, reason = check_spoof_gate("LONG", ob, 100.0)
    assert allowed is True
    assert reason == ""


def test_balanced_book_allowed_short():
    """Balanced bids and asks should pass for SHORT."""
    ob = _make_ob(
        bid_qtys=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ask_qtys=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    )
    allowed, reason = check_spoof_gate("SHORT", ob, 100.0)
    assert allowed is True


# ---------------------------------------------------------------------------
# Spoofed ask wall (LONG should be blocked)
# ---------------------------------------------------------------------------


def test_large_ask_wall_blocks_long():
    """A dominant ask-side wall should block LONG signals."""
    # Top-3 asks each hold 100 units; remaining hold 1 unit each → wall_ratio >> 5
    ob = _make_ob(
        bid_qtys=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ask_qtys=[100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0],
    )
    allowed, reason = check_spoof_gate("LONG", ob, 100.0)
    assert allowed is False
    assert "ask" in reason.lower()


def test_large_ask_wall_does_not_block_short():
    """A large ask wall is only relevant for LONG, not SHORT."""
    ob = _make_ob(
        bid_qtys=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ask_qtys=[100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0],
    )
    allowed, reason = check_spoof_gate("SHORT", ob, 100.0)
    # SHORT is only blocked by bid-side walls, not ask-side walls
    assert allowed is True


# ---------------------------------------------------------------------------
# Spoofed bid wall (SHORT should be blocked)
# ---------------------------------------------------------------------------


def test_large_bid_wall_blocks_short():
    """A dominant bid-side wall should block SHORT signals."""
    ob = _make_ob(
        bid_qtys=[100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0],
        ask_qtys=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    )
    allowed, reason = check_spoof_gate("SHORT", ob, 100.0)
    assert allowed is False
    assert "bid" in reason.lower()


def test_large_bid_wall_does_not_block_long():
    """A large bid wall should not block LONG signals."""
    ob = _make_ob(
        bid_qtys=[100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0],
        ask_qtys=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    )
    allowed, reason = check_spoof_gate("LONG", ob, 100.0)
    assert allowed is True


# ---------------------------------------------------------------------------
# Layering detection via concentration threshold
# ---------------------------------------------------------------------------


def test_ask_layering_blocks_long():
    """High concentration on the ask side (layering) should block LONG."""
    # Top-3 asks hold >> 70% of total ask liquidity
    ask_qtys = [50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 1.0]
    total = sum(ask_qtys)
    top3 = sum(ask_qtys[:WALL_TOP_N])
    assert top3 / total > LAYER_CONCENTRATION_THRESHOLD, "Precondition: concentration must exceed threshold"

    ob = _make_ob(bid_qtys=[5.0] * 6, ask_qtys=ask_qtys)
    allowed, reason = check_spoof_gate("LONG", ob, 100.0)
    assert allowed is False


def test_bid_layering_blocks_short():
    """High concentration on the bid side (layering) should block SHORT."""
    bid_qtys = [50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 1.0]
    total = sum(bid_qtys)
    top3 = sum(bid_qtys[:WALL_TOP_N])
    assert top3 / total > LAYER_CONCENTRATION_THRESHOLD

    ob = _make_ob(bid_qtys=bid_qtys, ask_qtys=[5.0] * 6)
    allowed, reason = check_spoof_gate("SHORT", ob, 100.0)
    assert allowed is False


# ---------------------------------------------------------------------------
# Edge: malformed / un-parseable entries
# ---------------------------------------------------------------------------


def test_malformed_entries_skipped():
    """Malformed entries should not crash the gate."""
    ob = {
        "bids": [["bad", "data"], [100.0, 5.0], [99.9, 5.0]],
        "asks": [[None, 5.0], [100.1, 5.0], [100.2, 5.0]],
    }
    # Should not raise
    allowed, reason = check_spoof_gate("LONG", ob, 100.0)
    assert isinstance(allowed, bool)
