"""Open Interest (OI) & Funding Rate Filter (Futures Context).

Provides utilities for analysing open interest and funding rate data to
differentiate between high-quality momentum moves and low-quality short
squeezes.

Design
------
* **High quality**: price rising AND OI rising → new money entering the market.
* **Low quality (squeeze)**: price rising but OI falling → shorts being liquidated,
  not genuine demand.  Signal is flagged as low quality and may be rejected.
* **Funding rate context**: extreme positive funding discourages new longs (over-
  crowded); extreme negative funding discourages new shorts.

This module is intentionally data-agnostic – pass arrays from any source
(live exchange feed, backtesting CSV, mock data).  The live feed wiring is
a separate concern.

Typical usage
-------------
.. code-block:: python

    import numpy as np
    from src.oi_filter import analyse_oi, check_oi_gate

    prices = np.array([100.0, 101.0, 102.0, 103.0])
    oi     = np.array([5000.0, 4800.0, 4600.0, 4400.0])  # falling OI

    result = analyse_oi(prices, oi)
    print(result.quality)   # "LOW"
    print(result.signal)    # "SQUEEZE"

    allowed, reason = check_oi_gate("LONG", result)
    # (False, 'OI: rising price + falling OI (squeeze) – LONG quality: LOW')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.utils import get_logger

log = get_logger("oi_filter")

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Minimum fractional change in OI to be considered "rising" or "falling".
#: Changes smaller than this are treated as flat / neutral.
OI_CHANGE_THRESHOLD: float = 0.005  # 0.5 %

#: Extreme funding rate (absolute) beyond which we flag crowded positioning.
#: Binance funding is typically ±0.01 % to ±0.1 %.  Beyond ±0.3 % is extreme.
FUNDING_EXTREME_THRESHOLD: float = 0.003  # 0.3 % (as a decimal)

#: Minimum OI change magnitude treated as a meaningful signal.  Changes
#: smaller than this threshold are classified as market noise and will not
#: trigger a hard rejection — only a debug log.  Corresponds to 1% OI change,
#: which is well-documented in quant literature as the minimum meaningful OI
#: shift on Binance perpetuals.
OI_NOISE_THRESHOLD: float = 0.01  # 1 %

#: Below this threshold (1%) OI changes are treated as pure market noise — no
#: rejection occurs (same as :data:`OI_NOISE_THRESHOLD`).
OI_SOFT_THRESHOLD: float = 0.01  # 1 %

#: Above this threshold (3%) OI changes paired with LOW quality trigger a hard
#: rejection.  Between :data:`OI_SOFT_THRESHOLD` and this value a soft warning
#: is issued but the signal is still allowed through.
OI_HARD_THRESHOLD: float = 0.03  # 3 %


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OIAnalysis:
    """Result of an Open Interest analysis pass."""

    price_direction: str    # "RISING" | "FALLING" | "FLAT"
    oi_direction: str       # "RISING" | "FALLING" | "FLAT"
    signal: str             # "MOMENTUM" | "SQUEEZE" | "DISTRIBUTION" | "NEUTRAL"
    quality: str            # "HIGH" | "MEDIUM" | "LOW"
    funding_bias: str       # "LONG_CROWDED" | "SHORT_CROWDED" | "NEUTRAL"
    price_change_pct: float
    oi_change_pct: float
    latest_funding_rate: Optional[float]
    reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pct_change(arr: np.ndarray) -> float:
    """Return percentage change from first to last element."""
    if len(arr) < 2 or arr[0] == 0:
        return 0.0
    return float((arr[-1] - arr[0]) / abs(arr[0]))


def _classify_direction(pct: float, threshold: float) -> str:
    if pct > threshold:
        return "RISING"
    if pct < -threshold:
        return "FALLING"
    return "FLAT"


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def analyse_oi(
    prices: Sequence[float],
    open_interest: Sequence[float],
    funding_rates: Optional[Sequence[float]] = None,
    oi_threshold: float = OI_CHANGE_THRESHOLD,
    funding_threshold: float = FUNDING_EXTREME_THRESHOLD,
) -> OIAnalysis:
    """Analyse price vs open interest to classify signal quality.

    Parameters
    ----------
    prices:
        Sequence of price values (e.g. close prices) over the lookback window.
        First element is the oldest, last is the most recent.
    open_interest:
        Sequence of open interest values aligned with *prices*.
    funding_rates:
        Optional sequence of funding rate values (as decimals, e.g. 0.0001).
        Only the most recent value is used for bias classification.
    oi_threshold:
        Minimum fractional OI change to be considered directional.
    funding_threshold:
        Absolute funding rate beyond which positioning is considered extreme.

    Returns
    -------
    :class:`OIAnalysis`
    """
    price_arr = np.asarray(prices, dtype=np.float64).ravel()
    oi_arr = np.asarray(open_interest, dtype=np.float64).ravel()

    if len(price_arr) == 0 or len(oi_arr) == 0:
        return OIAnalysis(
            price_direction="FLAT",
            oi_direction="FLAT",
            signal="NEUTRAL",
            quality="MEDIUM",
            funding_bias="NEUTRAL",
            price_change_pct=0.0,
            oi_change_pct=0.0,
            latest_funding_rate=None,
            reason="empty input arrays",
        )

    price_chg = _pct_change(price_arr)
    oi_chg = _pct_change(oi_arr)

    price_dir = _classify_direction(price_chg, oi_threshold)
    oi_dir = _classify_direction(oi_chg, oi_threshold)

    # Classify signal type
    if price_dir == "RISING" and oi_dir == "RISING":
        signal = "MOMENTUM"
        quality = "HIGH"
    elif price_dir == "RISING" and oi_dir == "FALLING":
        signal = "SQUEEZE"
        quality = "LOW"
    elif price_dir == "FALLING" and oi_dir == "RISING":
        signal = "DISTRIBUTION"
        quality = "LOW"
    elif price_dir == "FALLING" and oi_dir == "FALLING":
        signal = "MOMENTUM"  # shorts covering / long liquidations → bearish continuation
        quality = "HIGH"
    else:
        signal = "NEUTRAL"
        quality = "MEDIUM"

    # Funding rate bias
    latest_fr: Optional[float] = None
    funding_bias = "NEUTRAL"
    if funding_rates is not None and len(funding_rates) > 0:
        fr_arr = np.asarray(funding_rates, dtype=np.float64).ravel()
        latest_fr = float(fr_arr[-1])
        if latest_fr > funding_threshold:
            funding_bias = "LONG_CROWDED"
        elif latest_fr < -funding_threshold:
            funding_bias = "SHORT_CROWDED"

    reason = (
        f"price {price_dir} ({price_chg:+.2%}), "
        f"OI {oi_dir} ({oi_chg:+.2%}), "
        f"signal={signal}, funding_bias={funding_bias}"
    )

    return OIAnalysis(
        price_direction=price_dir,
        oi_direction=oi_dir,
        signal=signal,
        quality=quality,
        funding_bias=funding_bias,
        price_change_pct=round(price_chg, 6),
        oi_change_pct=round(oi_chg, 6),
        latest_funding_rate=latest_fr,
        reason=reason,
    )


def check_oi_gate(
    direction: str,
    oi_analysis: Optional[OIAnalysis],
    reject_low_quality: bool = True,
    min_oi_change_pct: float = OI_NOISE_THRESHOLD,
) -> tuple[bool, str]:
    """Pipeline hook: return ``(allowed, reason)`` for the OI/funding gate.

    Fails open when *oi_analysis* is ``None`` (data not yet available).

    Graduated response (per :data:`OI_SOFT_THRESHOLD` / :data:`OI_HARD_THRESHOLD`):

    * ``|Δ OI| < OI_SOFT_THRESHOLD`` (1 %) — treat as noise, allow through.
    * ``OI_SOFT_THRESHOLD ≤ |Δ OI| < OI_HARD_THRESHOLD`` (1–3 %) — soft
      warning, signal still allowed (``allowed=True`` with non-empty reason).
    * ``|Δ OI| ≥ OI_HARD_THRESHOLD`` (≥ 3 %) **and** quality is ``"LOW"`` —
      hard reject (``allowed=False``).

    Parameters
    ----------
    direction:
        ``"LONG"`` or ``"SHORT"``.
    oi_analysis:
        Output of :func:`analyse_oi`.  ``None`` → fails open.
    reject_low_quality:
        When ``True`` (default), hard-reject LOW quality signals above the
        hard threshold.  Set to ``False`` to always return a warning.
    min_oi_change_pct:
        Minimum absolute OI change (as a fraction, e.g. ``0.01`` = 1%) for a
        SQUEEZE or DISTRIBUTION pattern to trigger any action.  Changes
        below this threshold are treated as noise and the signal is allowed
        through with a debug log.  Defaults to :data:`OI_NOISE_THRESHOLD`.

    Returns
    -------
    ``(allowed, reason)``
    """
    if oi_analysis is None:
        return True, ""

    dir_upper = direction.upper()

    # Reject squeeze signals for LONG (price up, OI down)
    if dir_upper == "LONG" and oi_analysis.signal == "SQUEEZE":
        abs_chg = abs(oi_analysis.oi_change_pct)
        if abs_chg < min_oi_change_pct:
            log.debug(
                "OI squeeze below noise threshold ({:.2%}), allowing",
                oi_analysis.oi_change_pct,
            )
            return True, (
                f"OI: minor squeeze ({oi_analysis.oi_change_pct:+.2%})"
                " — below noise threshold"
            )
        if abs_chg < OI_HARD_THRESHOLD:
            # Soft warning — marginal OI move, allow with annotation
            soft_reason = (
                f"OI: moderate squeeze ({oi_analysis.oi_change_pct:+.2%})"
                " — soft warning only"
            )
            log.debug("OI squeeze soft warning (not blocking): {}", soft_reason)
            return True, soft_reason
        msg = f"OI: rising price + falling OI (squeeze) – LONG quality: {oi_analysis.quality}"
        if reject_low_quality:
            return False, msg
        log.debug("OI squeeze warning (not blocking): {}", msg)

    # Reject distribution signals for SHORT (price down, OI up – longs being trapped)
    if dir_upper == "SHORT" and oi_analysis.signal == "DISTRIBUTION":
        abs_chg = abs(oi_analysis.oi_change_pct)
        if abs_chg < min_oi_change_pct:
            log.debug(
                "OI distribution below noise threshold ({:.2%}), allowing",
                oi_analysis.oi_change_pct,
            )
            return True, (
                f"OI: minor distribution ({oi_analysis.oi_change_pct:+.2%})"
                " — below noise threshold"
            )
        if abs_chg < OI_HARD_THRESHOLD:
            # Soft warning — marginal OI move, allow with annotation
            soft_reason = (
                f"OI: moderate distribution ({oi_analysis.oi_change_pct:+.2%})"
                " — soft warning only"
            )
            log.debug("OI distribution soft warning (not blocking): {}", soft_reason)
            return True, soft_reason
        msg = f"OI: falling price + rising OI (distribution) – SHORT quality: {oi_analysis.quality}"
        if reject_low_quality:
            return False, msg
        log.debug("OI distribution warning (not blocking): {}", msg)

    # Warn about crowded funding for the signal direction
    if dir_upper == "LONG" and oi_analysis.funding_bias == "LONG_CROWDED":
        fr = oi_analysis.latest_funding_rate
        if fr is not None:
            log.debug("OI: funding rate {:.4%} – longs crowded, reduce position size", fr)
        else:
            log.debug("OI: longs crowded (funding rate unavailable) – reduce position size")

    if dir_upper == "SHORT" and oi_analysis.funding_bias == "SHORT_CROWDED":
        fr = oi_analysis.latest_funding_rate
        if fr is not None:
            log.debug("OI: funding rate {:.4%} – shorts crowded, reduce position size", fr)
        else:
            log.debug("OI: shorts crowded (funding rate unavailable) – reduce position size")

    return True, ""
