"""Chart generation for 360_GEM signals.

Generates TradingView-style dark-theme candlestick charts using ``mplfinance``.
Chart generation is fully optional — if ``mplfinance`` is not installed the
module degrades gracefully and every public function returns ``None``.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional

from src.utils import get_logger

log = get_logger("chart_generator")

try:
    import matplotlib
    matplotlib.use("Agg")
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    import pandas as pd
    _MPF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPF_AVAILABLE = False
    log.info("mplfinance not installed – gem chart generation disabled")


def generate_gem_chart(
    symbol: str,
    daily_candles: Dict[str, list],
    ath: float,
    current_price: float,
    ema_20: List[float],
    ema_50: List[float],
) -> Optional[bytes]:
    """Generate a dark-theme daily candlestick chart for a gem signal.

    Parameters
    ----------
    symbol:
        Trading pair symbol (e.g. ``"LYNUSDT"``).
    daily_candles:
        Dict with keys ``"open"``, ``"high"``, ``"low"``, ``"close"``,
        ``"volume"`` — daily OHLCV lists.
    ath:
        All-time high price drawn as a horizontal dashed line.
    current_price:
        Current price (used to compute x-potential for the chart title).
    ema_20:
        EMA(20) values aligned to the *last* N candles.
    ema_50:
        EMA(50) values aligned to the *last* N candles.

    Returns
    -------
    Optional[bytes]
        PNG image bytes if generation succeeds, ``None`` otherwise.
    """
    if not _MPF_AVAILABLE:
        return None

    try:
        opens = [float(v) for v in daily_candles.get("open", [])]
        highs = [float(v) for v in daily_candles.get("high", [])]
        lows = [float(v) for v in daily_candles.get("low", [])]
        closes = [float(v) for v in daily_candles.get("close", [])]
        volumes = [float(v) for v in daily_candles.get("volume", [])]

        n = min(len(opens), len(highs), len(lows), len(closes), len(volumes))
        if n < 10:
            return None

        # Use last 90–120 candles for the chart
        window = min(120, n)
        opens = opens[-window:]
        highs = highs[-window:]
        lows = lows[-window:]
        closes = closes[-window:]
        volumes = volumes[-window:]

        # Build a DatetimeIndex (synthetic daily dates ending today)
        import datetime
        end_date = datetime.date.today()
        dates = [end_date - datetime.timedelta(days=window - 1 - i) for i in range(window)]
        idx = pd.DatetimeIndex(dates)

        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }, index=idx)

        # Align EMA arrays to the chart window
        ema20_window = list(ema_20[-window:]) if len(ema_20) >= window else list(ema_20)
        ema50_window = list(ema_50[-window:]) if len(ema_50) >= window else list(ema_50)

        # Pad shorter EMA arrays with NaN at the start
        def _pad(arr: list, target: int) -> list:
            if len(arr) < target:
                return [float("nan")] * (target - len(arr)) + arr
            return arr[-target:]

        ema20_padded = _pad(ema20_window, window)
        ema50_padded = _pad(ema50_window, window)

        x_potential = ath / current_price if current_price > 0 else 0.0
        x_label = f"x{x_potential:.0f}" if x_potential > 0 else "N/A"
        title = f"{symbol} — Daily | [GEM] {x_label}"

        # ATH horizontal line
        ath_line = [ath] * window

        addplots = [
            mpf.make_addplot(ema20_padded, color="#26a69a", width=1.2, label="EMA 20"),
            mpf.make_addplot(ema50_padded, color="#ef5350", width=1.2, label="EMA 50"),
            mpf.make_addplot(ath_line, color="#ffd700", width=1.0, linestyle="--", label="ATH"),
        ]

        # Dark TradingView-like style
        mc = mpf.make_marketcolors(
            up="#26a69a",
            down="#ef5350",
            edge="inherit",
            wick="inherit",
            volume={"up": "#26a69a", "down": "#ef5350"},
        )
        style = mpf.make_mpf_style(
            marketcolors=mc,
            facecolor="#131722",
            figcolor="#131722",
            gridcolor="#1e222d",
            gridstyle="--",
            gridaxis="both",
            y_on_right=True,
            rc={
                "axes.labelcolor": "#787b86",
                "xtick.color": "#787b86",
                "ytick.color": "#787b86",
                "text.color": "#d1d4dc",
            },
        )

        buf = io.BytesIO()
        mpf.plot(
            df,
            type="candle",
            style=style,
            title=title,
            volume=True,
            addplot=addplots,
            figsize=(14, 8),
            savefig=dict(fname=buf, format="png", dpi=150, bbox_inches="tight"),
        )
        plt.close("all")
        buf.seek(0)
        return buf.read()

    except Exception as exc:
        log.error("generate_gem_chart error for %s: %s", symbol, exc)
        return None

