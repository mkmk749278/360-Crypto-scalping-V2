"""Paper Trading Portfolio Simulator.

Manages per-user, per-channel virtual portfolios that shadow real signals.
Portfolios are updated silently whenever a signal completes (TP or SL hit).
All output is returned as formatted strings — this module never sends any
Telegram messages directly.

Storage: ``data/paper_portfolios.json`` (created automatically on first write).
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.utils import get_logger

log = get_logger("paper_portfolio")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PaperTrade:
    """A single completed paper trade."""

    signal_id: str
    channel: str
    symbol: str
    direction: str          # "LONG" / "SHORT"
    entry_price: float
    exit_price: float
    leverage: int
    position_size_usdt: float
    fee_paid: float         # entry + exit fees
    pnl_usdt: float         # net PnL in USDT (after fees)
    pnl_pct: float          # percentage PnL from the signal
    status: str             # "WIN" / "LOSS" / "BREAKEVEN"
    timestamp: float


@dataclass
class ChannelPortfolio:
    """Per-channel portfolio for a single user."""

    channel: str
    initial_balance: float = 1000.0
    current_balance: float = 1000.0
    leverage: int = 1
    risk_per_trade_pct: float = 2.0  # Risk 2% of balance per trade
    trades: List[PaperTrade] = field(default_factory=list)
    total_fees: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0
    reset_count: int = 0
    peak_balance: float = 1000.0
    max_drawdown_pct: float = 0.0
    # Phase 2 fields
    liquidation_count: int = 0
    is_liquidated: bool = False
    current_streak: int = 0        # Positive = win streak, negative = loss streak
    best_win_streak: int = 0
    worst_loss_streak: int = 0     # Negative number (most negative = worst)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _portfolio_to_dict(portfolio: ChannelPortfolio) -> dict:
    """Serialize a ChannelPortfolio to a plain dict for JSON storage."""
    d = dataclasses.asdict(portfolio)
    return d


def _portfolio_from_dict(d: dict) -> ChannelPortfolio:
    """Deserialize a ChannelPortfolio from a plain dict.

    Backward-compatible: Phase-1 JSON files that don't have the Phase-2
    fields will use dataclass defaults via the ``pop(key, default)`` pattern.
    """
    trades_raw = d.pop("trades", [])
    trades = [PaperTrade(**t) for t in trades_raw]
    # Provide defaults for Phase-2 fields missing from older JSON files
    d.setdefault("liquidation_count", 0)
    d.setdefault("is_liquidated", False)
    d.setdefault("current_streak", 0)
    d.setdefault("best_win_streak", 0)
    d.setdefault("worst_loss_streak", 0)
    portfolio = ChannelPortfolio(**d)
    portfolio.trades = trades
    return portfolio


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class PaperPortfolioManager:
    """Manages virtual paper trading portfolios for all users.

    Thread-safety: this class is designed to be used from a single asyncio
    event loop.  ``_save`` is wrapped in a try/except so persistence errors
    never propagate to callers.
    """

    INITIAL_BALANCE: float = 1000.0
    DEFAULT_LEVERAGE: int = 1
    DEFAULT_RISK_PCT: float = 2.0
    FEE_RATE: float = 0.001  # 0.1% per side (Binance taker)

    CHANNELS = ["360_SCALP", "360_SWING", "360_SPOT"]

    def __init__(self, storage_path: str = "data/paper_portfolios.json") -> None:
        self._path = Path(storage_path)
        # chat_id (str) → channel (str) → ChannelPortfolio
        self._portfolios: Dict[str, Dict[str, ChannelPortfolio]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def record_trade(
        self,
        channel: str,
        signal_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        hit_tp: int,
        hit_sl: bool,
        pnl_pct: float,
        tp_prices: Optional[List[float]] = None,  # [tp1_price, tp2_price, tp3_price]
    ) -> None:
        """Record a completed signal outcome across ALL user portfolios.

        Called by ``TradeMonitor._record_outcome()`` for every completed signal.
        Updates every registered user's portfolio for the given channel silently.
        Only channels in :attr:`CHANNELS` are tracked (SELECT is excluded).
        """
        if channel not in self.CHANNELS:
            return

        _TP_SCALE_RATIOS = [0.30, 0.30, 0.40]

        for chat_id in list(self._portfolios.keys()):
            portfolio = self._portfolios[chat_id].get(channel)
            if portfolio is None:
                continue

            # Skip if the channel is already liquidated — user must reset first
            if portfolio.is_liquidated:
                log.debug(
                    "Skipping trade for %s/%s: channel is liquidated", chat_id, channel
                )
                continue

            # Position sizing
            risk_amount = portfolio.current_balance * (portfolio.risk_per_trade_pct / 100.0)
            position_size = risk_amount * portfolio.leverage

            # --- Liquidation check (only for leveraged losing positions) ---
            if portfolio.leverage > 1 and pnl_pct < 0:
                liquidation_threshold = (100.0 / portfolio.leverage) * 0.9
                if abs(pnl_pct) >= liquidation_threshold:
                    lost_amount = portfolio.current_balance
                    portfolio.is_liquidated = True
                    portfolio.liquidation_count += 1
                    portfolio.loss_count += 1
                    portfolio.total_pnl -= lost_amount
                    portfolio.current_balance = 0.0
                    portfolio.max_drawdown_pct = 100.0
                    # Update streak
                    if portfolio.current_streak < 0:
                        portfolio.current_streak -= 1
                    else:
                        portfolio.current_streak = -1
                    portfolio.worst_loss_streak = min(
                        portfolio.worst_loss_streak, portfolio.current_streak
                    )
                    trade = PaperTrade(
                        signal_id=signal_id,
                        channel=channel,
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        leverage=portfolio.leverage,
                        position_size_usdt=position_size,
                        fee_paid=0.0,
                        pnl_usdt=-lost_amount,
                        pnl_pct=pnl_pct,
                        status="LIQUIDATED",
                        timestamp=time.time(),
                    )
                    portfolio.trades.append(trade)
                    continue  # Skip normal PnL calculation for this user

            # --- Partial TP scaling or single-exit PnL ---
            if tp_prices and hit_tp >= 1:
                blended_pnl_raw = 0.0
                remaining_ratio = 1.0
                n_exits = min(hit_tp, len(tp_prices), len(_TP_SCALE_RATIOS))
                for i in range(n_exits):
                    ratio = _TP_SCALE_RATIOS[i]
                    tp_pnl_pct = self._calculate_directional_pnl_pct(
                        entry_price, tp_prices[i], direction
                    )
                    blended_pnl_raw += position_size * ratio * (tp_pnl_pct / 100.0)
                    remaining_ratio -= ratio
                if remaining_ratio > 0:
                    blended_pnl_raw += position_size * remaining_ratio * (pnl_pct / 100.0)
                trade_pnl_raw = blended_pnl_raw
            else:
                trade_pnl_raw = position_size * (pnl_pct / 100.0)

            # Fees: entry + exit (both sides)
            fee = position_size * self.FEE_RATE * 2

            # Net PnL after fees
            net_pnl = trade_pnl_raw - fee

            # Determine outcome status
            if abs(pnl_pct) < 0.05:
                status = "BREAKEVEN"
                portfolio.breakeven_count += 1
            elif pnl_pct > 0:
                status = "WIN"
                portfolio.win_count += 1
            else:
                status = "LOSS"
                portfolio.loss_count += 1

            # Update streaks
            if status == "WIN":
                if portfolio.current_streak > 0:
                    portfolio.current_streak += 1
                else:
                    portfolio.current_streak = 1
                portfolio.best_win_streak = max(
                    portfolio.best_win_streak, portfolio.current_streak
                )
            elif status == "LOSS":
                if portfolio.current_streak < 0:
                    portfolio.current_streak -= 1
                else:
                    portfolio.current_streak = -1
                portfolio.worst_loss_streak = min(
                    portfolio.worst_loss_streak, portfolio.current_streak
                )
            # BREAKEVEN doesn't break the streak

            # Update balance (floor at 0)
            portfolio.current_balance += net_pnl
            portfolio.current_balance = max(portfolio.current_balance, 0.0)
            portfolio.total_pnl += net_pnl
            portfolio.total_fees += fee

            # Track peak and max drawdown
            if portfolio.current_balance > portfolio.peak_balance:
                portfolio.peak_balance = portfolio.current_balance
            if portfolio.peak_balance > 0:
                dd = (
                    (portfolio.peak_balance - portfolio.current_balance)
                    / portfolio.peak_balance
                    * 100
                )
                portfolio.max_drawdown_pct = max(portfolio.max_drawdown_pct, dd)

            # Record the individual trade
            trade = PaperTrade(
                signal_id=signal_id,
                channel=channel,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                leverage=portfolio.leverage,
                position_size_usdt=position_size,
                fee_paid=fee,
                pnl_usdt=net_pnl,
                pnl_pct=pnl_pct,
                status=status,
                timestamp=time.time(),
            )
            portfolio.trades.append(trade)

        self._save()

    @staticmethod
    def _calculate_directional_pnl_pct(entry: float, exit_: float, direction: str) -> float:
        """Calculate PnL percentage based on direction."""
        if entry == 0:
            return 0.0
        if direction == "LONG":
            return (exit_ - entry) / entry * 100.0
        else:  # SHORT
            return (entry - exit_) / entry * 100.0

    def ensure_user(self, chat_id: str) -> None:
        """Initialize portfolios for a new user if they don't already exist."""
        if chat_id not in self._portfolios:
            self._portfolios[chat_id] = {}
            for channel in self.CHANNELS:
                self._portfolios[chat_id][channel] = ChannelPortfolio(
                    channel=channel,
                    initial_balance=self.INITIAL_BALANCE,
                    current_balance=self.INITIAL_BALANCE,
                    peak_balance=self.INITIAL_BALANCE,
                )
            self._save()

    # ------------------------------------------------------------------
    # Query / formatting
    # ------------------------------------------------------------------

    def get_portfolio_summary(self, chat_id: str) -> str:
        """Return a formatted Telegram message with all channel balances."""
        self.ensure_user(chat_id)
        portfolios = self._portfolios[chat_id]

        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SWING": "🏛️",
            "360_SPOT": "📈",
        }

        lines = ["💼 *Paper Trading Portfolio*\n"]
        total_balance = 0.0
        total_pnl = 0.0
        total_fees = 0.0
        total_resets = 0

        for channel in self.CHANNELS:
            p = portfolios.get(channel)
            if p is None:
                continue
            emoji = chan_emojis.get(channel, "📡")
            pnl_pct_display = (
                (p.current_balance - p.initial_balance) / p.initial_balance * 100
                if p.initial_balance > 0
                else 0.0
            )
            win_rate = (
                p.win_count / (p.win_count + p.loss_count) * 100
                if (p.win_count + p.loss_count) > 0
                else 0.0
            )

            lines.append(f"{emoji} *{channel.replace('360_', '')}*")
            lines.append(f"   Balance: ${p.current_balance:,.2f} ({pnl_pct_display:+.1f}%)")
            if p.is_liquidated:
                lines.append("   💀 LIQUIDATED")
            else:
                if p.current_streak > 0:
                    streak_tag = f" 🔥{p.current_streak}W"
                elif p.current_streak < 0:
                    streak_tag = f" ❄️{abs(p.current_streak)}L"
                else:
                    streak_tag = ""
                lines.append(
                    f"   Leverage: {p.leverage}×  |  Risk: {p.risk_per_trade_pct:.0f}%"
                    f"{streak_tag}"
                )
            lines.append(
                f"   W/L/BE: {p.win_count}/{p.loss_count}/{p.breakeven_count}"
                f"  |  WR: {win_rate:.0f}%"
            )
            lines.append(f"   Fees: ${p.total_fees:,.2f}  |  DD: {p.max_drawdown_pct:.1f}%")
            lines.append("")

            total_balance += p.current_balance
            total_pnl += p.total_pnl
            total_fees += p.total_fees
            total_resets += p.reset_count

        initial_total = self.INITIAL_BALANCE * len(self.CHANNELS)
        total_pnl_pct = (
            (total_balance - initial_total) / initial_total * 100
            if initial_total > 0
            else 0.0
        )

        lines.append("━━━━━━━━━━━━━━━━━━")
        lines.append(f"📊 Total: ${total_balance:,.2f} ({total_pnl_pct:+.1f}%)")
        lines.append(f"💰 Net PnL: ${total_pnl:+,.2f}")
        lines.append(f"💸 Total Fees: ${total_fees:,.2f}")
        lines.append(f"🔄 Resets: {total_resets}")

        return "\n".join(lines)

    def get_channel_detail(self, chat_id: str, channel: str) -> str:
        """Return detailed view of a single channel portfolio."""
        self.ensure_user(chat_id)
        portfolios = self._portfolios.get(chat_id, {})
        p = portfolios.get(channel)
        if p is None:
            return f"❌ Channel `{channel}` not found."

        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SWING": "🏛️",
            "360_SPOT": "📈",
        }
        emoji = chan_emojis.get(channel, "📡")
        pnl_pct = (
            (p.current_balance - p.initial_balance) / p.initial_balance * 100
            if p.initial_balance > 0
            else 0.0
        )
        total_trades = p.win_count + p.loss_count + p.breakeven_count
        win_rate = (
            p.win_count / (p.win_count + p.loss_count) * 100
            if (p.win_count + p.loss_count) > 0
            else 0.0
        )

        lines = [
            f"{emoji} *{channel} Portfolio Detail*\n",
            f"💰 Balance: ${p.current_balance:,.2f} ({pnl_pct:+.1f}%)",
            f"📊 Initial: ${p.initial_balance:,.2f}",
            f"⚙️ Leverage: {p.leverage}×  |  Risk: {p.risk_per_trade_pct:.0f}%",
            f"📈 Trades: {total_trades} (W:{p.win_count} L:{p.loss_count} BE:{p.breakeven_count})",
            f"🏆 Win Rate: {win_rate:.1f}%",
        ]

        if p.current_streak > 0:
            streak_str = f"🔥 {p.current_streak}W"
        elif p.current_streak < 0:
            streak_str = f"❄️ {abs(p.current_streak)}L"
        else:
            streak_str = "—"
        lines.append(
            f"📊 Streak: {streak_str} | Best: 🔥{p.best_win_streak}W"
            f" / ❄️{abs(p.worst_loss_streak)}L"
        )

        lines += [
            f"💸 Total Fees: ${p.total_fees:,.2f}",
            f"📉 Max Drawdown: {p.max_drawdown_pct:.1f}%",
            f"🏔️ Peak Balance: ${p.peak_balance:,.2f}",
            f"🔄 Resets: {p.reset_count}",
        ]

        if p.is_liquidated:
            lines.append(
                "💀 LIQUIDATED — Balance wiped. Use /reset_portfolio to start over."
            )

        if p.trades:
            lines.append("\n📋 *Last 5 Trades:*")
            for t in p.trades[-5:]:
                status_emoji = (
                    "✅" if t.status == "WIN" else ("❌" if t.status == "LOSS" else "➖")
                )
                lines.append(
                    f"  {status_emoji} {t.symbol} {t.direction} | "
                    f"PnL: ${t.pnl_usdt:+.2f} ({t.pnl_pct:+.2f}%) | Fee: ${t.fee_paid:.2f}"
                )

        return "\n".join(lines)

    def get_trade_history(
        self,
        chat_id: str,
        channel: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Return formatted trade history for a user."""
        self.ensure_user(chat_id)
        portfolios = self._portfolios[chat_id]

        all_trades: List[PaperTrade] = []
        if channel:
            p = portfolios.get(channel)
            if p is None:
                return f"❌ Channel `{channel}` not found."
            all_trades = list(p.trades)
        else:
            for ch in self.CHANNELS:
                p = portfolios.get(ch)
                if p:
                    all_trades.extend(p.trades)

        if not all_trades:
            return "📋 No trades yet. Trades are recorded when signals complete."

        # Most recent first
        all_trades.sort(key=lambda t: t.timestamp, reverse=True)
        recent = all_trades[:limit]

        label = channel or "All Channels"
        lines = [f"📋 *Trade History — {label}*\n"]
        for t in recent:
            status_emoji = (
                "✅" if t.status == "WIN"
                else ("❌" if t.status == "LOSS"
                      else ("💀" if t.status == "LIQUIDATED" else "➖"))
            )
            chan_short = t.channel.replace("360_", "")
            lines.append(
                f"{status_emoji} *{t.symbol}* {t.direction} ({chan_short})\n"
                f"   PnL: ${t.pnl_usdt:+.2f} ({t.pnl_pct:+.2f}%) | "
                f"Lev: {t.leverage}× | Fee: ${t.fee_paid:.2f}"
            )

        return "\n".join(lines)

    def get_leaderboard(self, sort_by: str = "pnl", limit: int = 10) -> str:
        """Return a formatted leaderboard of all users ranked by total PnL or ROI.

        Parameters
        ----------
        sort_by:
            "pnl" for absolute dollar PnL, "roi" for percentage return on initial.
        limit:
            Max number of users to show.
        """
        entries = []
        for chat_id, channels in self._portfolios.items():
            total_balance = sum(p.current_balance for p in channels.values())
            total_pnl = sum(p.total_pnl for p in channels.values())
            initial_total = self.INITIAL_BALANCE * len(self.CHANNELS)
            roi_pct = (
                (total_balance - initial_total) / initial_total * 100
            ) if initial_total > 0 else 0.0
            total_trades = sum(
                p.win_count + p.loss_count + p.breakeven_count for p in channels.values()
            )
            total_wins = sum(p.win_count for p in channels.values())
            total_losses = sum(p.loss_count for p in channels.values())
            win_rate = (
                total_wins / (total_wins + total_losses) * 100
            ) if (total_wins + total_losses) > 0 else 0.0
            total_resets = sum(p.reset_count for p in channels.values())

            entries.append({
                "chat_id": chat_id,
                "total_balance": total_balance,
                "total_pnl": total_pnl,
                "roi_pct": roi_pct,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "total_resets": total_resets,
            })

        if not entries:
            return "🏆 No users registered yet. Use /portfolio to get started!"

        if sort_by == "roi":
            entries.sort(key=lambda e: e["roi_pct"], reverse=True)
        else:
            entries.sort(key=lambda e: e["total_pnl"], reverse=True)

        top = entries[:limit]
        medal_emojis = ["🥇", "🥈", "🥉"]
        sort_label = "ROI" if sort_by == "roi" else "PnL"
        lines = [f"🏆 *Paper Trading Leaderboard* (by {sort_label})\n"]

        for i, e in enumerate(top):
            medal = medal_emojis[i] if i < 3 else f"#{i + 1}"
            cid = str(e["chat_id"])
            display_id = f"{cid[:4]}…{cid[-2:]}" if len(cid) > 6 else cid
            lines.append(
                f"{medal} `{display_id}` — ${e['total_balance']:,.2f} "
                f"({e['roi_pct']:+.1f}%)\n"
                f"   PnL: ${e['total_pnl']:+,.2f} | WR: {e['win_rate']:.0f}% | "
                f"Trades: {e['total_trades']} | Resets: {e['total_resets']}"
            )

        lines.append(f"\n📊 Total participants: {len(entries)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Mutation commands
    # ------------------------------------------------------------------

    def reset_portfolio(self, chat_id: str, channel: Optional[str] = None) -> str:
        """Reset portfolio to $1,000. Returns confirmation message."""
        self.ensure_user(chat_id)
        portfolios = self._portfolios[chat_id]

        if channel:
            p = portfolios.get(channel)
            if p is None:
                return f"❌ Channel `{channel}` not found."
            p.current_balance = self.INITIAL_BALANCE
            p.total_pnl = 0.0
            p.total_fees = 0.0
            p.win_count = 0
            p.loss_count = 0
            p.breakeven_count = 0
            p.peak_balance = self.INITIAL_BALANCE
            p.max_drawdown_pct = 0.0
            p.trades.clear()
            p.reset_count += 1
            p.is_liquidated = False
            p.current_streak = 0
            p.best_win_streak = 0
            p.worst_loss_streak = 0
            # liquidation_count is a lifetime stat — keep it
            self._save()
            return (
                f"🔄 Portfolio reset for *{channel}*. "
                f"Balance: $1,000.00 (Reset #{p.reset_count})"
            )
        else:
            for ch in self.CHANNELS:
                p = portfolios.get(ch)
                if p:
                    p.current_balance = self.INITIAL_BALANCE
                    p.total_pnl = 0.0
                    p.total_fees = 0.0
                    p.win_count = 0
                    p.loss_count = 0
                    p.breakeven_count = 0
                    p.peak_balance = self.INITIAL_BALANCE
                    p.max_drawdown_pct = 0.0
                    p.trades.clear()
                    p.reset_count += 1
                    p.is_liquidated = False
                    p.current_streak = 0
                    p.best_win_streak = 0
                    p.worst_loss_streak = 0
                    # liquidation_count is a lifetime stat — keep it
            self._save()
            return "🔄 All portfolios reset to $1,000.00 each."

    def set_leverage(self, chat_id: str, channel: str, leverage: int) -> str:
        """Set leverage for a channel. Returns confirmation message."""
        self.ensure_user(chat_id)
        if leverage < 1 or leverage > 20:
            return "❌ Leverage must be between 1 and 20."
        p = self._portfolios[chat_id].get(channel)
        if p is None:
            return f"❌ Channel `{channel}` not found."
        p.leverage = leverage
        self._save()
        return f"✅ Leverage for *{channel}* set to *{leverage}×*"

    def set_risk(self, chat_id: str, channel: str, risk_pct: float) -> str:
        """Set risk percentage per trade. Returns confirmation message."""
        self.ensure_user(chat_id)
        if risk_pct < 0.5 or risk_pct > 10.0:
            return "❌ Risk must be between 0.5% and 10%."
        p = self._portfolios[chat_id].get(channel)
        if p is None:
            return f"❌ Channel `{channel}` not found."
        p.risk_per_trade_pct = risk_pct
        self._save()
        return f"✅ Risk for *{channel}* set to *{risk_pct:.1f}%* per trade"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist all portfolios to disk (JSON). Errors are logged, not raised."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data: dict = {}
            for chat_id, channels in self._portfolios.items():
                data[chat_id] = {
                    ch: _portfolio_to_dict(p) for ch, p in channels.items()
                }
            self._path.write_text(json.dumps(data, indent=2))
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to save paper portfolios: %s", exc)

    def _load(self) -> None:
        """Load portfolios from disk. Missing file is treated as empty state."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for chat_id, channels in raw.items():
                self._portfolios[chat_id] = {
                    ch: _portfolio_from_dict(p_dict)
                    for ch, p_dict in channels.items()
                }
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to load paper portfolios: %s", exc)
