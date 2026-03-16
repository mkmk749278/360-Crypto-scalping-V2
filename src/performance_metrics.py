"""Shared PnL and drawdown helpers for signal-performance tracking."""

from __future__ import annotations

from typing import Iterable, Tuple

_MIN_PNL_PCT = -99.99


def normalize_pnl_pct(pnl_pct: float) -> float:
    """Clamp realized PnL to a sane lower bound."""
    return max(float(pnl_pct), _MIN_PNL_PCT)


def calculate_trade_pnl_pct(entry_price: float, exit_price: float, direction: str) -> float:
    """Calculate realized PnL % for a long or short trade."""
    if entry_price <= 0 or exit_price <= 0:
        return 0.0

    direction_name = direction.upper()
    if direction_name == "SHORT":
        pnl_pct = (entry_price - exit_price) / entry_price * 100.0
    else:
        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    return normalize_pnl_pct(pnl_pct)


def calculate_drawdown_metrics(pnl_pcts: Iterable[float]) -> Tuple[float, float]:
    """Return current and maximum drawdown (%) from a compounded equity curve."""
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0

    for pnl_pct in pnl_pcts:
        equity *= max(0.0, 1.0 + normalize_pnl_pct(pnl_pct) / 100.0)
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    current_drawdown = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
    return current_drawdown, max_drawdown
