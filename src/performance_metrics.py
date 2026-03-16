"""Shared PnL, drawdown, and outcome classification helpers."""

from __future__ import annotations

from typing import Iterable, Tuple

_MIN_PNL_PCT = -99.99
_BREAKEVEN_PNL_THRESHOLD_PCT = 0.01


def normalize_pnl_pct(pnl_pct: float) -> float:
    """Clamp realized PnL to a sane lower bound."""
    return max(float(pnl_pct), _MIN_PNL_PCT)


def is_breakeven_pnl(pnl_pct: float) -> bool:
    """Return True when realized PnL is close enough to treat as breakeven."""
    return abs(normalize_pnl_pct(pnl_pct)) < _BREAKEVEN_PNL_THRESHOLD_PCT


def classify_trade_outcome(pnl_pct: float, hit_tp: int = 0, hit_sl: bool = False) -> str:
    """Classify the final realized trade outcome.

    The classification is semantic rather than purely mechanical:
    - stop exits with negative realized PnL remain ``SL_HIT``
    - stop exits around flat are ``BREAKEVEN_EXIT``
    - stop exits with positive realized PnL become ``PROFIT_LOCKED``
    - a final TP completion is ``FULL_TP_HIT``
    - exits that are neither stop nor TP completions fall back to ``CLOSED``
    """
    normalized_pnl = normalize_pnl_pct(pnl_pct)
    if hit_tp >= 3 and not hit_sl:
        return "FULL_TP_HIT"
    if hit_sl:
        if is_breakeven_pnl(normalized_pnl):
            return "BREAKEVEN_EXIT"
        if normalized_pnl < 0.0:
            return "SL_HIT"
        return "PROFIT_LOCKED"
    if hit_tp > 0:
        return f"TP{hit_tp}_HIT"
    return "CLOSED"


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
