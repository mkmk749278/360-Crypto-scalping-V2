"""DCA (Dollar-Cost Averaging / Double Entry) logic for 360-Crypto-scalping.

When a trade dips after Entry 1 but still has good momentum/structure, a 2nd
entry is taken at the lower price.  The averaged entry is now lower, so when
price returns to Entry 1's level the trade is already at breakeven or small
profit.  After taking the 2nd entry, all TPs are recalculated from the new
averaged entry using the same R-multiples (tp_ratios) from the channel config.

Key constraints
---------------
* Max 2 entries — no 3rd entry (not martingale).
* SL stays fixed at the original structural invalidation level.
* TPs are recalculated from avg_entry, preserving R:R multiples.
* Position is split 60/40 (weight_1/weight_2) by default.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

from src.smc import Direction
from src.utils import get_logger, utcnow

if TYPE_CHECKING:
    from config import ChannelConfig
    from src.channels.base import Signal

log = get_logger("dca")


def compute_dca_zone(
    entry: float,
    stop_loss: float,
    direction: Direction,
    zone_range: Tuple[float, float] = (0.30, 0.70),
    regime: Optional[str] = None,
) -> Tuple[float, float]:
    """Compute the DCA entry zone bounds.

    The DCA zone is defined as a fraction of the SL distance away from the
    entry price, towards the stop-loss.  By default the zone spans 30–70 % of
    the SL distance so that:

    * Entries too close to the original entry (< 30 %) are ignored — not
      enough improvement in average entry price.
    * Entries too close to the stop-loss (> 70 %) are ignored — the trade is
      near structural invalidation and DCA would be risky.

    When *regime* is provided the zone is adjusted automatically:

    * ``VOLATILE`` or ``TRENDING``: zone pushed deeper ``(0.50, 0.85)`` because
      volatile wicks will often reach 50 % drawdown before reversing.  Entering
      too early in these regimes risks a false DCA that gets stopped out.
    * ``RANGING``: tighter zone ``(0.30, 0.60)`` — ranging markets reverse more
      predictably, so waiting too deep wastes opportunity.

    Parameters
    ----------
    entry:
        Original entry price (Entry 1).
    stop_loss:
        Fixed stop-loss price.
    direction:
        Trade direction (LONG or SHORT).
    zone_range:
        ``(lower_fraction, upper_fraction)`` of the SL distance that defines
        the DCA zone.  Defaults to ``(0.30, 0.70)``.  Overridden by *regime*
        when provided.
    regime:
        Optional market regime string (case-insensitive).  Accepted values:
        ``"VOLATILE"``, ``"TRENDING"``, ``"RANGING"``.  Any other value leaves
        *zone_range* unchanged.

    Returns
    -------
    tuple[float, float]
        ``(zone_lower, zone_upper)`` — absolute price bounds of the DCA zone.
        For a LONG trade, zone_lower < zone_upper < entry.
        For a SHORT trade, entry < zone_lower < zone_upper.
    """
    # Apply regime-specific zone overrides
    if regime is not None:
        regime_upper = regime.upper()
        if regime_upper in ("VOLATILE", "TRENDING"):
            zone_range = (0.50, 0.85)
        elif regime_upper == "RANGING":
            zone_range = (0.30, 0.60)

    sl_dist = abs(entry - stop_loss)
    lo_frac, hi_frac = zone_range

    if direction == Direction.LONG:
        zone_upper = entry - lo_frac * sl_dist
        zone_lower = entry - hi_frac * sl_dist
    else:
        zone_lower = entry + lo_frac * sl_dist
        zone_upper = entry + hi_frac * sl_dist

    return (zone_lower, zone_upper)


def recalculate_after_dca(
    sig: "Signal",
    entry_2_price: float,
    tp_ratios: List[float],
    weight_1: float = 0.6,
    weight_2: float = 0.4,
) -> None:
    """Update a signal in-place after a DCA (2nd) entry is taken.

    Steps performed
    ---------------
    1. Persist original values (entry, TP1, TP2, TP3) before modification.
    2. Compute the weighted average entry.
    3. Keep SL unchanged — it represents structural invalidation.
    4. Recalculate new SL distance from avg_entry to the fixed SL.
    5. Recalculate TPs from avg_entry using the same R-multiples (tp_ratios).
    6. Update ``sig.entry`` to avg_entry so that all downstream code (PnL,
       trailing stop, break-even move on TP2) works automatically.
    7. Store DCA metadata on the signal.

    Parameters
    ----------
    sig:
        Active :class:`~src.channels.base.Signal` to update in-place.
    entry_2_price:
        Executed price of the 2nd entry.
    tp_ratios:
        R-multiples for TP1, TP2, TP3 (from channel config).
    weight_1:
        Fraction of position at Entry 1 (default 0.6 = 60 %).
    weight_2:
        Fraction of position at Entry 2 (default 0.4 = 40 %).
    """
    old_entry = sig.entry
    old_sl = sig.stop_loss

    # Persist originals (only on the first DCA — guard against double-call)
    if sig.original_entry == 0.0:
        sig.original_entry = old_entry
        sig.original_tp1 = sig.tp1
        sig.original_tp2 = sig.tp2
        sig.original_tp3 = sig.tp3

    # Weighted average entry
    avg_entry = old_entry * weight_1 + entry_2_price * weight_2

    # New SL distance from avg_entry to the fixed (unchanged) SL
    new_sl_dist = abs(avg_entry - old_sl)

    # Recalculate TPs from avg_entry using same R-multiples
    is_long = sig.direction == Direction.LONG
    new_tps = []
    for ratio in tp_ratios:
        if is_long:
            new_tps.append(avg_entry + new_sl_dist * ratio)
        else:
            new_tps.append(avg_entry - new_sl_dist * ratio)

    # Apply updated values to the signal
    sig.entry = round(avg_entry, 8)
    sig.avg_entry = round(avg_entry, 8)
    sig.entry_2 = round(entry_2_price, 8)
    sig.entry_2_filled = True
    sig.dca_timestamp = utcnow()
    sig.position_weight_1 = weight_1
    sig.position_weight_2 = weight_2

    if len(new_tps) >= 1:
        sig.tp1 = round(new_tps[0], 8)
    if len(new_tps) >= 2:
        sig.tp2 = round(new_tps[1], 8)
    if len(new_tps) >= 3:
        sig.tp3 = round(new_tps[2], 8)

    # Update original_sl_distance so trailing stop uses the new buffer
    sig.original_sl_distance = new_sl_dist

    log.info(
        "DCA recalculated | %s %s | Entry1=%.6f Entry2=%.6f AvgEntry=%.6f "
        "| SL=%.6f (unchanged) | NewTP1=%.6f NewTP2=%.6f NewTP3=%s",
        sig.symbol,
        sig.direction.value,
        old_entry,
        entry_2_price,
        avg_entry,
        old_sl,
        sig.tp1,
        sig.tp2,
        f"{sig.tp3:.6f}" if sig.tp3 is not None else "N/A",
    )


def check_dca_entry(
    sig: "Signal",
    current_price: float,
    indicators: Optional[dict] = None,
    smc_data: Optional[dict] = None,
    channel_config: Optional["ChannelConfig"] = None,
) -> Optional[float]:
    """Determine whether conditions allow a DCA (2nd) entry at *current_price*.

    Returns *current_price* when all conditions are met, or ``None`` otherwise.

    Conditions checked
    ------------------
    1. ``sig.entry_2_filled`` must be ``False`` — only one DCA allowed.
    2. Current price must be inside the pre-computed DCA zone
       (``sig.dca_zone_lower`` ≤ price ≤ ``sig.dca_zone_upper``).
    3. *Momentum re-validation* (when indicators are provided): the absolute
       momentum value must be ≥ ``dca_min_momentum`` (default 0.2).
    4. *Structure re-validation* (when smc_data is provided): at least one MSS
       event must still be present (``smc_data["mss"]`` is not ``None``).
    5. *EMA200 bias* (Swing channel + indicators provided): price must still be
       on the correct side of EMA200 for the trade direction.

    Parameters
    ----------
    sig:
        Active signal being monitored.
    current_price:
        Latest market price.
    indicators:
        Per-timeframe indicator dict (optional; momentum/EMA checks skipped
        when ``None``).
    smc_data:
        SMC detection result dict (optional; structure check skipped when
        ``None``).
    channel_config:
        Channel configuration (optional; used to get ``dca_min_momentum`` and
        ``dca_zone_range``; default thresholds are used when ``None``).

    Returns
    -------
    float or None
        *current_price* when all checks pass, ``None`` otherwise.
    """
    # Guard: already DCA'd
    if sig.entry_2_filled:
        return None

    # Guard: DCA zone must be configured (non-zero bounds)
    if sig.dca_zone_lower == 0.0 and sig.dca_zone_upper == 0.0:
        return None

    # Zone check — price must be between the pre-computed bounds
    if not (sig.dca_zone_lower <= current_price <= sig.dca_zone_upper):
        return None

    # Momentum re-validation (optional — skip when indicators not available)
    if indicators is not None:
        min_momentum = (
            channel_config.dca_min_momentum
            if channel_config is not None
            else 0.2
        )
        # Check the most relevant timeframe — use first available with data
        momentum_ok = False
        for tf_ind in indicators.values():
            mom = tf_ind.get("momentum_last")
            if mom is not None:
                if abs(mom) >= min_momentum:
                    momentum_ok = True
                break
        if not momentum_ok:
            log.debug(
                "DCA rejected for %s %s — momentum faded",
                sig.symbol,
                sig.direction.value,
            )
            return None

    # Volume delta check — avoid DCAing into heavy, aggressive counter-pressure.
    # A strongly negative delta while we're LONG (or positive while SHORT) means
    # smart money is actively selling/buying against our position.
    # Only the first timeframe with volume_delta data is checked, consistent
    # with the momentum check above.
    if indicators is not None:
        for tf_ind in indicators.values():
            vd = tf_ind.get("volume_delta")
            if vd is not None:
                if sig.direction == Direction.LONG and vd < -0.7:
                    log.debug(
                        "DCA rejected for %s LONG — heavy selling pressure (volume_delta=%.2f)",
                        sig.symbol,
                        vd,
                    )
                    return None
                if sig.direction == Direction.SHORT and vd > 0.7:
                    log.debug(
                        "DCA rejected for %s SHORT — heavy buying pressure (volume_delta=%.2f)",
                        sig.symbol,
                        vd,
                    )
                    return None
                break  # only check the first timeframe with delta data

    # Structure re-validation (optional — skip when smc_data not available)
    if smc_data is not None:
        mss = smc_data.get("mss")
        if mss is None:
            log.debug(
                "DCA rejected for %s %s — MSS no longer present",
                sig.symbol,
                sig.direction.value,
            )
            return None

    # EMA200 bias check for Swing channel (optional)
    if (
        channel_config is not None
        and channel_config.name == "360_SWING"
        and indicators is not None
    ):
        h1_ind = indicators.get("1h", {})
        ema200 = h1_ind.get("ema200_last")
        if ema200 is not None:
            if sig.direction == Direction.LONG and current_price < ema200:
                log.debug(
                    "DCA rejected for %s LONG — price %.6f below EMA200 %.6f",
                    sig.symbol,
                    current_price,
                    ema200,
                )
                return None
            if sig.direction == Direction.SHORT and current_price > ema200:
                log.debug(
                    "DCA rejected for %s SHORT — price %.6f above EMA200 %.6f",
                    sig.symbol,
                    current_price,
                    ema200,
                )
                return None

    log.debug(
        "DCA entry valid for %s %s at %.6f (zone %.6f–%.6f)",
        sig.symbol,
        sig.direction.value,
        current_price,
        sig.dca_zone_lower,
        sig.dca_zone_upper,
    )
    return current_price
