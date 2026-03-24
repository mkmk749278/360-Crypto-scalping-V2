"""Public Performance Dashboard — HTML report generator.

Reads from :class:`~src.performance_tracker.PerformanceTracker` and produces
an HTML file at ``data/performance_report.html``.

Features
--------
* Overall win rate, total trades, cumulative PnL
* Per-channel breakdown (SCALP, SWING, SPOT, GEM) with tables
* 7-day and 30-day rolling stats
* Top 5 trades
* Monthly performance summary table

The report is intentionally self-contained (no external CSS/JS dependencies)
so it can be sent as a Telegram document or hosted anywhere.

Usage
-----
.. code-block:: python

    from src.performance_report import generate_html_report
    path = generate_html_report(tracker)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from src.utils import get_logger

if TYPE_CHECKING:
    from src.performance_tracker import PerformanceTracker, SignalRecord

log = get_logger("performance_report")

_DEFAULT_OUTPUT_PATH = "data/performance_report.html"


def generate_html_report(
    tracker: "PerformanceTracker",
    output_path: str = _DEFAULT_OUTPUT_PATH,
) -> str:
    """Generate an HTML performance report from *tracker* data.

    Parameters
    ----------
    tracker:
        :class:`~src.performance_tracker.PerformanceTracker` instance with
        completed signal records.
    output_path:
        Path to write the HTML file.  The directory is created if absent.

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Gather stats
    overall = tracker.get_stats()
    stats_7d = tracker.get_stats(window_days=7)
    stats_30d = tracker.get_stats(window_days=30)
    channel_stats = tracker.all_channel_stats()
    top5 = tracker.get_top_trades(n=5, window_days=365)

    # Monthly breakdown: group records by YYYY-MM
    monthly: dict = {}
    for record in tracker._records:  # pylint: disable=protected-access
        dt = datetime.fromtimestamp(record.timestamp, tz=timezone.utc)
        key = dt.strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0}
        entry = monthly[key]
        entry["count"] += 1
        entry["total_pnl"] += record.pnl_pct
        if record.hit_sl:
            entry["losses"] += 1
        elif record.hit_tp > 0:
            entry["wins"] += 1

    html = _render_html(
        now_str=now_str,
        overall=overall,
        stats_7d=stats_7d,
        stats_30d=stats_30d,
        channel_stats=channel_stats,
        top5=top5,
        monthly=monthly,
    )

    out.write_text(html, encoding="utf-8")
    log.info("Performance report written to %s", out.resolve())
    return str(out.resolve())


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _pnl_color(pnl: float) -> str:
    """Return a CSS color string based on PnL sign."""
    if pnl > 0:
        return "color:#00c853"
    if pnl < 0:
        return "color:#d50000"
    return "color:#9e9e9e"


def _render_html(
    now_str: str,
    overall: Any,
    stats_7d: Any,
    stats_30d: Any,
    channel_stats: Dict[str, Any],
    top5: List["SignalRecord"],
    monthly: Dict[str, Any],
) -> str:
    """Produce the full HTML string."""

    # Per-channel rows
    channel_rows = ""
    for ch, st in sorted(channel_stats.items()):
        win_rate_pct = f"{st.win_rate:.1f}%"
        avg_pnl = st.avg_pnl_pct
        channel_rows += (
            f"<tr>"
            f"<td>{ch}</td>"
            f"<td>{st.total_signals}</td>"
            f"<td>{st.win_count}</td>"
            f"<td>{st.loss_count}</td>"
            f"<td>{win_rate_pct}</td>"
            f"<td style='{_pnl_color(avg_pnl)}'>{avg_pnl:+.2f}%</td>"
            f"<td style='{_pnl_color(st.best_trade)}'>{st.best_trade:+.2f}%</td>"
            f"<td style='{_pnl_color(-abs(st.worst_trade))}'>{st.worst_trade:+.2f}%</td>"
            f"</tr>\n"
        )

    # Top 5 trade rows
    top5_rows = ""
    for r in top5:
        dt_str = datetime.fromtimestamp(r.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        top5_rows += (
            f"<tr>"
            f"<td>{r.symbol}</td>"
            f"<td>{r.channel}</td>"
            f"<td>{r.direction}</td>"
            f"<td style='{_pnl_color(r.pnl_pct)}'>{r.pnl_pct:+.2f}%</td>"
            f"<td>TP{r.hit_tp}</td>"
            f"<td>{dt_str}</td>"
            f"</tr>\n"
        )

    # Monthly rows
    monthly_rows = ""
    for month in sorted(monthly.keys(), reverse=True)[:12]:
        data = monthly[month]
        total = data["wins"] + data["losses"]
        wr = f"{data['wins'] / total * 100:.1f}%" if total > 0 else "—"
        total_pnl = data["total_pnl"]
        monthly_rows += (
            f"<tr>"
            f"<td>{month}</td>"
            f"<td>{data['count']}</td>"
            f"<td>{data['wins']}</td>"
            f"<td>{data['losses']}</td>"
            f"<td>{wr}</td>"
            f"<td style='{_pnl_color(total_pnl)}'>{total_pnl:+.2f}%</td>"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>360 Crypto Scalping — Performance Report</title>
<style>
  body {{font-family:Arial,sans-serif;background:#111;color:#eee;margin:0;padding:20px}}
  h1 {{color:#ffd700;border-bottom:2px solid #333;padding-bottom:10px}}
  h2 {{color:#90caf9;margin-top:30px}}
  .summary-grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0}}
  .stat-card {{background:#1e1e1e;border-radius:8px;padding:16px;text-align:center}}
  .stat-card .label {{font-size:12px;color:#aaa;text-transform:uppercase;letter-spacing:1px}}
  .stat-card .value {{font-size:24px;font-weight:bold;margin-top:4px}}
  table {{width:100%;border-collapse:collapse;margin-top:12px}}
  th {{background:#1e1e1e;padding:10px;text-align:left;color:#90caf9;font-size:13px}}
  td {{padding:8px 10px;border-bottom:1px solid #222;font-size:13px}}
  tr:hover td {{background:#1a1a2e}}
  .footer {{margin-top:40px;font-size:12px;color:#555;text-align:center}}
</style>
</head>
<body>
<h1>💹 360 Crypto Scalping — Performance Report</h1>
<p style="color:#aaa;font-size:13px">Generated: {now_str}</p>

<h2>📊 Overall Statistics</h2>
<div class="summary-grid">
  <div class="stat-card">
    <div class="label">Total Trades</div>
    <div class="value">{overall.total_signals}</div>
  </div>
  <div class="stat-card">
    <div class="label">Win Rate</div>
    <div class="value" style="color:#00c853">{overall.win_rate:.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="label">Avg PnL</div>
    <div class="value" style="{_pnl_color(overall.avg_pnl_pct)}">{overall.avg_pnl_pct:+.2f}%</div>
  </div>
  <div class="stat-card">
    <div class="label">Best Trade</div>
    <div class="value" style="color:#00c853">{overall.best_trade:+.2f}%</div>
  </div>
  <div class="stat-card">
    <div class="label">Max Drawdown</div>
    <div class="value" style="color:#d50000">{overall.max_drawdown:.2f}%</div>
  </div>
</div>

<h2>📅 Rolling Performance</h2>
<div class="summary-grid">
  <div class="stat-card">
    <div class="label">7-Day Trades</div>
    <div class="value">{stats_7d.total_signals}</div>
  </div>
  <div class="stat-card">
    <div class="label">7-Day Win Rate</div>
    <div class="value" style="{_pnl_color(stats_7d.win_rate - 50)}">{stats_7d.win_rate:.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="label">7-Day Avg PnL</div>
    <div class="value" style="{_pnl_color(stats_7d.avg_pnl_pct)}">{stats_7d.avg_pnl_pct:+.2f}%</div>
  </div>
  <div class="stat-card">
    <div class="label">30-Day Trades</div>
    <div class="value">{stats_30d.total_signals}</div>
  </div>
  <div class="stat-card">
    <div class="label">30-Day Win Rate</div>
    <div class="value" style="{_pnl_color(stats_30d.win_rate - 50)}">{stats_30d.win_rate:.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="label">30-Day Avg PnL</div>
    <div class="value" style="{_pnl_color(stats_30d.avg_pnl_pct)}">{stats_30d.avg_pnl_pct:+.2f}%</div>
  </div>
</div>

<h2>📈 Per-Channel Breakdown</h2>
<table>
<thead><tr>
  <th>Channel</th><th>Trades</th><th>Wins</th><th>Losses</th>
  <th>Win Rate</th><th>Avg PnL</th><th>Best</th><th>Worst</th>
</tr></thead>
<tbody>
{channel_rows}
</tbody>
</table>

<h2>🏆 Top 5 Trades (Last 12 Months)</h2>
<table>
<thead><tr>
  <th>Symbol</th><th>Channel</th><th>Direction</th><th>PnL</th><th>Exit</th><th>Date</th>
</tr></thead>
<tbody>
{top5_rows if top5_rows else "<tr><td colspan='6' style='color:#555;text-align:center'>No winning trades recorded yet</td></tr>"}
</tbody>
</table>

<h2>📆 Monthly Performance (Last 12 Months)</h2>
<table>
<thead><tr>
  <th>Month</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>Total PnL</th>
</tr></thead>
<tbody>
{monthly_rows if monthly_rows else "<tr><td colspan='6' style='color:#555;text-align:center'>No monthly data yet</td></tr>"}
</tbody>
</table>

<div class="footer">360 Crypto Scalping V2 · Automated Performance Dashboard · {now_str}</div>
</body>
</html>"""
