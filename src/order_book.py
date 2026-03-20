"""Order Book Imbalance (OBI) execution filter.

Provides utilities to calculate bid/ask depth imbalance and act as the
final execution gate before a trade signal goes live.  If the order book
is heavily stacked against the signal direction the trade is rejected to
avoid buying into a wall of sellers (or selling into a wall of buyers).

Usage
-----
::

    from src.order_book import check_order_book_execution

    allowed, reason = check_order_book_execution(
        direction="LONG",
        order_book={"bids": [...], "asks": [...]},
    )
    if not allowed:
        log.warning("Trade blocked: {}", reason)

Graceful fallback
-----------------
If ``order_book`` is ``None`` or contains insufficient data the filter
*fails open* – the trade is allowed through without crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.utils import get_logger

log = get_logger("order_book")

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------

#: If the opposing side represents more than this fraction of total order book
#: volume, the signal is rejected.
#:
#: * For a LONG signal: if asks_pct > threshold → reject (too many sellers)
#: * For a SHORT signal: if bids_pct > threshold → reject (too many buyers)
OBI_REJECTION_THRESHOLD: float = 0.65  # 65%

#: Number of depth levels (from each side) to include in the calculation.
OBI_DEFAULT_LEVELS: int = 20


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Computed metrics from a single order book depth snapshot."""

    bid_volume: float       # USD-weighted total bid depth
    ask_volume: float       # USD-weighted total ask depth
    total_volume: float     # bid_volume + ask_volume
    bid_pct: float          # fraction of total that is bids  (0-1)
    ask_pct: float          # fraction of total that is asks  (0-1)
    imbalance_ratio: float  # dominant_vol / weak_vol  (≥ 1.0)
    dominant_side: str      # "bids" | "asks" | "balanced"


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------


def calculate_order_book_imbalance(
    bids: Sequence,
    asks: Sequence,
    levels: int = OBI_DEFAULT_LEVELS,
) -> Optional[OrderBookSnapshot]:
    """Compute bid/ask volume imbalance from raw order book depth data.

    Each entry in *bids* / *asks* must be a sequence ``[price, quantity]``
    where both values are numeric or string-encoded floats.

    Parameters
    ----------
    bids:
        Best bids list, ordered highest price first.  Each element is
        ``[price, qty]``.
    asks:
        Best asks list, ordered lowest price first.  Each element is
        ``[price, qty]``.
    levels:
        Maximum number of levels to consume from each side.

    Returns
    -------
    :class:`OrderBookSnapshot` or ``None`` when data is insufficient.
    """
    if not bids or not asks:
        return None

    try:
        bid_vol: float = sum(
            float(b[0]) * float(b[1]) for b in bids[:levels]
        )
        ask_vol: float = sum(
            float(a[0]) * float(a[1]) for a in asks[:levels]
        )
    except (IndexError, TypeError, ValueError) as exc:
        log.debug("OBI calculation error: {}", exc)
        return None

    total = bid_vol + ask_vol
    if total <= 0:
        return None

    bid_pct = bid_vol / total
    ask_pct = ask_vol / total

    weak = min(bid_vol, ask_vol)
    ratio = (max(bid_vol, ask_vol) / weak) if weak > 0 else 0.0

    if bid_vol > ask_vol:
        dominant = "bids"
    elif ask_vol > bid_vol:
        dominant = "asks"
    else:
        dominant = "balanced"

    return OrderBookSnapshot(
        bid_volume=bid_vol,
        ask_volume=ask_vol,
        total_volume=total,
        bid_pct=bid_pct,
        ask_pct=ask_pct,
        imbalance_ratio=round(ratio, 4),
        dominant_side=dominant,
    )


# ---------------------------------------------------------------------------
# Execution gate
# ---------------------------------------------------------------------------


def check_order_book_execution(
    direction: str,
    order_book: Optional[dict],
    rejection_threshold: float = OBI_REJECTION_THRESHOLD,
    levels: int = OBI_DEFAULT_LEVELS,
) -> Tuple[bool, str]:
    """Determine whether order book conditions support executing a signal.

    This is designed to be the **final execution filter** – called just
    before a signal is dispatched to the exchange.  It returns ``(True, "")``
    in all cases where data is missing so the system *fails open* rather
    than silently blocking trades due to data gaps.

    Parameters
    ----------
    direction:
        ``"LONG"`` or ``"SHORT"``.
    order_book:
        Dict with ``"bids"`` and ``"asks"`` keys containing depth lists.
        ``None`` or missing keys → fails open (returns ``True``).
    rejection_threshold:
        Fraction of total book volume.  If the opposing side exceeds this
        fraction the signal is rejected.  Default: ``0.65`` (65 %).
    levels:
        Number of depth levels to evaluate from each side.

    Returns
    -------
    ``(allowed, reason)`` where *allowed* is ``False`` only when the book
    clearly opposes the signal direction above the threshold.

    Examples
    --------
    >>> # Asks dominate 70 % of the book – LONG rejected
    >>> check_order_book_execution("LONG", {"bids": [...], "asks": [...]})
    (False, 'OBI: ask wall 70% > 65% threshold – LONG blocked')

    >>> # Data unavailable – trade allowed through
    >>> check_order_book_execution("LONG", None)
    (True, '')
    """
    if order_book is None:
        return True, ""

    bids: List = order_book.get("bids", [])
    asks: List = order_book.get("asks", [])

    snapshot = calculate_order_book_imbalance(bids, asks, levels)
    if snapshot is None:
        # Insufficient data – fail open
        return True, ""

    if direction == "LONG" and snapshot.ask_pct > rejection_threshold:
        return (
            False,
            (
                f"OBI: ask wall {snapshot.ask_pct:.0%} > "
                f"{rejection_threshold:.0%} threshold – LONG blocked"
            ),
        )

    if direction == "SHORT" and snapshot.bid_pct > rejection_threshold:
        return (
            False,
            (
                f"OBI: bid wall {snapshot.bid_pct:.0%} > "
                f"{rejection_threshold:.0%} threshold – SHORT blocked"
            ),
        )

    return True, ""
