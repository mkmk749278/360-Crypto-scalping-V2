"""Spoofing / Layering Detection Gate.

Detects order-book manipulation (large walls that appear/disappear) using
the cached order-book depth snapshot already available in :mod:`src.scanner`
via ``Scanner._get_order_book_depth(symbol)``.

Two manipulation signatures are caught:

1. **Large wall on the opposing side** — a single dominant level on the ask
   side (for LONG signals) or bid side (for SHORT signals) whose quantity is
   more than :data:`SPOOF_WALL_RATIO_THRESHOLD` times the average of the
   remaining levels. Such walls frequently indicate a large spoofed limit
   order that will be pulled if price approaches it.

2. **Layering** — the top 3 levels on one side concentrate more than
   :data:`LAYER_CONCENTRATION_THRESHOLD` of that side's total visible
   liquidity.  Multiple thin stacked orders create the illusion of depth
   while actually being trivial to pull.

All public constants are defined at module level for easy tuning.

Typical usage
-------------
.. code-block:: python

    from src.spoof_detect import check_spoof_gate

    ob = {"bids": [[101.0, 5.0], [100.9, 1.0]], "asks": [[101.1, 30.0], ...]}
    allowed, reason = check_spoof_gate("LONG", ob, entry=101.05)
    if not allowed:
        print(reason)
"""

from __future__ import annotations

from src.utils import get_logger

log = get_logger("spoof_detect")

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: Minimum number of levels on a side to perform analysis (avoid false positives
#: on thin books with only 1-2 visible levels).
MIN_LEVELS_FOR_ANALYSIS: int = 4

#: Number of "top" levels that constitute the "wall" bucket.
WALL_TOP_N: int = 3

#: Wall ratio threshold: wall-bucket total / average of remaining levels.
#: If the top-N levels are this many times larger than the rest → likely spoof.
SPOOF_WALL_RATIO_THRESHOLD: float = 5.0

#: Layering threshold: fraction of total side liquidity held by the top-N
#: levels.  Above this fraction the top is considered suspicious.
LAYER_CONCENTRATION_THRESHOLD: float = 0.70


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_levels(raw: list) -> list[tuple[float, float]]:
    """Convert raw ``[[price, qty], ...]`` to a list of ``(price, qty)`` floats.

    Silently skips entries that cannot be parsed.
    """
    parsed: list[tuple[float, float]] = []
    for entry in raw:
        try:
            price, qty = float(entry[0]), float(entry[1])
            parsed.append((price, qty))
        except (IndexError, TypeError, ValueError):
            continue
    return parsed


def _analyse_side(
    levels: list[tuple[float, float]],
) -> tuple[float, float, bool]:
    """Compute wall-ratio and concentration for one side of the order book.

    Parameters
    ----------
    levels:
        Parsed ``(price, qty)`` tuples ordered from best to worst price
        (i.e. asks ascending, bids descending).

    Returns
    -------
    wall_ratio : float
        ``sum(top-N qty) / mean(remaining qty)``.  0.0 when insufficient data.
    concentration : float
        Fraction of total qty held by the top-N levels.  0.0 when no data.
    is_suspicious : bool
        True when *either* wall_ratio or concentration exceeds its threshold.
    """
    if len(levels) < MIN_LEVELS_FOR_ANALYSIS:
        return 0.0, 0.0, False

    qtys = [qty for _, qty in levels]
    top_qty = sum(qtys[:WALL_TOP_N])
    rest_qty = qtys[WALL_TOP_N:]

    avg_rest = sum(rest_qty) / len(rest_qty) if rest_qty else 0.0
    wall_ratio = (top_qty / avg_rest) if avg_rest > 0 else 0.0

    total_qty = sum(qtys)
    concentration = (top_qty / total_qty) if total_qty > 0 else 0.0

    is_suspicious = (
        wall_ratio > SPOOF_WALL_RATIO_THRESHOLD
        or concentration > LAYER_CONCENTRATION_THRESHOLD
    )
    return wall_ratio, concentration, is_suspicious


# ---------------------------------------------------------------------------
# Public gate function
# ---------------------------------------------------------------------------


def check_spoof_gate(
    direction: str,
    order_book: dict | None,
    entry: float,  # noqa: ARG001 — reserved for future price-proximity check
) -> tuple[bool, str]:
    """Pipeline gate: detect order-book spoofing / layering.

    Parameters
    ----------
    direction:
        Signal direction – ``"LONG"`` or ``"SHORT"``.
    order_book:
        Dict with ``"bids"`` and ``"asks"`` keys, each a list of
        ``[price, qty]`` pairs.  When ``None`` the gate **fails open**
        (returns ``(True, "")``).
    entry:
        Intended entry price (reserved for future proximity filtering).

    Returns
    -------
    ``(allowed, reason)``
        *allowed* is ``False`` when a spoofing or layering pattern is
        detected that would likely block the signal direction.
    """
    if order_book is None:
        return True, ""

    raw_bids = order_book.get("bids") or []
    raw_asks = order_book.get("asks") or []

    if not raw_bids and not raw_asks:
        return True, ""

    # Bids: sort descending by price (best bid first)
    bids = sorted(_parse_levels(raw_bids), key=lambda x: x[0], reverse=True)
    # Asks: sort ascending by price (best ask first)
    asks = sorted(_parse_levels(raw_asks), key=lambda x: x[0])

    ask_ratio, ask_conc, ask_suspicious = _analyse_side(asks)
    bid_ratio, bid_conc, bid_suspicious = _analyse_side(bids)

    direction_upper = direction.upper()

    if direction_upper == "LONG":
        # Large resistance wall on the ask side → spoofed ceiling above entry
        if ask_suspicious:
            if ask_ratio > SPOOF_WALL_RATIO_THRESHOLD:
                return False, (
                    f"Spoof gate: ask-side wall ratio {ask_ratio:.1f}× "
                    f"(threshold {SPOOF_WALL_RATIO_THRESHOLD}×) – likely spoofed resistance"
                )
            return False, (
                f"Spoof gate: ask-side layering concentration {ask_conc:.0%} "
                f"(threshold {LAYER_CONCENTRATION_THRESHOLD:.0%}) – suspicious depth"
            )

    elif direction_upper == "SHORT":
        # Large support wall on the bid side → spoofed floor below entry
        if bid_suspicious:
            if bid_ratio > SPOOF_WALL_RATIO_THRESHOLD:
                return False, (
                    f"Spoof gate: bid-side wall ratio {bid_ratio:.1f}× "
                    f"(threshold {SPOOF_WALL_RATIO_THRESHOLD}×) – likely spoofed support"
                )
            return False, (
                f"Spoof gate: bid-side layering concentration {bid_conc:.0%} "
                f"(threshold {LAYER_CONCENTRATION_THRESHOLD:.0%}) – suspicious depth"
            )

    return True, ""
