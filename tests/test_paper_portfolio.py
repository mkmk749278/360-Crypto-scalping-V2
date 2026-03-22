"""Tests for PaperPortfolioManager (src/paper_portfolio.py)."""

from __future__ import annotations

from pathlib import Path

from src.paper_portfolio import PaperPortfolioManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> PaperPortfolioManager:
    return PaperPortfolioManager(storage_path=str(tmp_path / "portfolios.json"))


def _record_win(mgr: PaperPortfolioManager, channel: str = "360_SCALP") -> None:
    mgr.record_trade(
        channel=channel,
        signal_id="sig-win",
        symbol="BTCUSDT",
        direction="LONG",
        entry_price=30000.0,
        exit_price=30450.0,
        hit_tp=3,
        hit_sl=False,
        pnl_pct=1.5,
    )


def _record_loss(mgr: PaperPortfolioManager, channel: str = "360_SCALP") -> None:
    mgr.record_trade(
        channel=channel,
        signal_id="sig-loss",
        symbol="BTCUSDT",
        direction="LONG",
        entry_price=30000.0,
        exit_price=29700.0,
        hit_tp=0,
        hit_sl=True,
        pnl_pct=-1.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPaperPortfolioManager:
    def test_ensure_user_creates_4_channels(self, tmp_path):
        """New user gets portfolios for all 4 channels with $1000 each."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        portfolios = mgr._portfolios["user1"]
        assert set(portfolios.keys()) == set(PaperPortfolioManager.CHANNELS)
        for ch, p in portfolios.items():
            assert p.current_balance == 1000.0
            assert p.initial_balance == 1000.0

    def test_ensure_user_idempotent(self, tmp_path):
        """Calling ensure_user twice doesn't reset portfolios."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        # Record a trade to mutate state
        _record_win(mgr)
        balance_before = mgr._portfolios["user1"]["360_SCALP"].current_balance
        mgr.ensure_user("user1")
        balance_after = mgr._portfolios["user1"]["360_SCALP"].current_balance
        assert balance_before == balance_after

    def test_record_winning_trade(self, tmp_path):
        """A winning trade increases balance and increments win_count."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        initial = mgr._portfolios["user1"]["360_SCALP"].current_balance

        _record_win(mgr)

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_balance > initial
        assert p.win_count == 1
        assert p.loss_count == 0

    def test_record_losing_trade(self, tmp_path):
        """A losing trade decreases balance and increments loss_count."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        initial = mgr._portfolios["user1"]["360_SCALP"].current_balance

        _record_loss(mgr)

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_balance < initial
        assert p.loss_count == 1
        assert p.win_count == 0

    def test_record_breakeven_trade(self, tmp_path):
        """A breakeven trade (~0 PnL) increments breakeven_count."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-be",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=30000.0,
            hit_tp=0,
            hit_sl=False,
            pnl_pct=0.0,  # exactly 0 → BREAKEVEN
        )

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.breakeven_count == 1
        assert p.win_count == 0
        assert p.loss_count == 0

    def test_fees_deducted_correctly(self, tmp_path):
        """Fees = position_size × 0.001 × 2 (entry + exit)."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-fee",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=30000.0,
            hit_tp=0,
            hit_sl=False,
            pnl_pct=0.0,
        )

        p = mgr._portfolios["user1"]["360_SCALP"]
        # risk_amount = 1000 * 2% = 20; position = 20 * 1 = 20; fee = 20 * 0.001 * 2 = 0.04
        expected_fee = 1000.0 * 0.02 * 1 * 0.001 * 2
        assert abs(p.total_fees - expected_fee) < 1e-9

    def test_leverage_amplifies_pnl(self, tmp_path):
        """Setting leverage to 5x should amplify PnL by 5x."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        # Default leverage (1x) — record a win
        _record_win(mgr)
        balance_1x = mgr._portfolios["user1"]["360_SCALP"].current_balance

        # Reset and set 5x leverage
        mgr.reset_portfolio("user1", "360_SCALP")
        mgr.set_leverage("user1", "360_SCALP", 5)
        _record_win(mgr)
        balance_5x = mgr._portfolios["user1"]["360_SCALP"].current_balance

        # With 5x leverage, net PnL change from initial should be ~5x bigger
        delta_1x = balance_1x - 1000.0
        delta_5x = balance_5x - 1000.0
        # Not exactly 5x due to fees scaling too, but should be close
        assert delta_5x > delta_1x * 4

    def test_balance_cannot_go_below_zero(self, tmp_path):
        """A huge loss should cap balance at 0, not go negative."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-huge-loss",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=0.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-10000.0,  # Catastrophic loss
        )

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_balance >= 0.0

    def test_reset_portfolio_single_channel(self, tmp_path):
        """Reset one channel back to $1000, others unchanged."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr, channel="360_SCALP")
        _record_win(mgr, channel="360_SWING")

        swing_before = mgr._portfolios["user1"]["360_SWING"].current_balance
        mgr.reset_portfolio("user1", "360_SCALP")

        assert mgr._portfolios["user1"]["360_SCALP"].current_balance == 1000.0
        assert mgr._portfolios["user1"]["360_SWING"].current_balance == swing_before

    def test_reset_portfolio_all_channels(self, tmp_path):
        """Reset all channels back to $1000."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        for ch in PaperPortfolioManager.CHANNELS:
            _record_win(mgr, channel=ch)

        mgr.reset_portfolio("user1")

        for ch in PaperPortfolioManager.CHANNELS:
            assert mgr._portfolios["user1"][ch].current_balance == 1000.0

    def test_reset_increments_reset_count(self, tmp_path):
        """Each reset increments the reset_count."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        mgr.reset_portfolio("user1", "360_SCALP")
        assert mgr._portfolios["user1"]["360_SCALP"].reset_count == 1
        mgr.reset_portfolio("user1", "360_SCALP")
        assert mgr._portfolios["user1"]["360_SCALP"].reset_count == 2

    def test_set_leverage_valid(self, tmp_path):
        """Setting leverage within 1-20 works."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.set_leverage("user1", "360_SCALP", 10)
        assert "✅" in result
        assert mgr._portfolios["user1"]["360_SCALP"].leverage == 10

    def test_set_leverage_invalid(self, tmp_path):
        """Leverage outside 1-20 is rejected."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result_low = mgr.set_leverage("user1", "360_SCALP", 0)
        result_high = mgr.set_leverage("user1", "360_SCALP", 21)
        assert "❌" in result_low
        assert "❌" in result_high
        assert mgr._portfolios["user1"]["360_SCALP"].leverage == 1  # unchanged

    def test_set_risk_valid(self, tmp_path):
        """Setting risk within 0.5-10% works."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.set_risk("user1", "360_SCALP", 5.0)
        assert "✅" in result
        assert mgr._portfolios["user1"]["360_SCALP"].risk_per_trade_pct == 5.0

    def test_set_risk_invalid(self, tmp_path):
        """Risk outside 0.5-10% is rejected."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result_low = mgr.set_risk("user1", "360_SCALP", 0.1)
        result_high = mgr.set_risk("user1", "360_SCALP", 11.0)
        assert "❌" in result_low
        assert "❌" in result_high
        assert mgr._portfolios["user1"]["360_SCALP"].risk_per_trade_pct == 2.0  # unchanged

    def test_max_drawdown_tracked(self, tmp_path):
        """Max drawdown is updated when balance drops from peak."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        # Win first to raise peak, then lose
        _record_win(mgr)
        _record_loss(mgr)

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.max_drawdown_pct > 0.0

    def test_peak_balance_tracked(self, tmp_path):
        """Peak balance updates when balance reaches new high."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        initial_peak = mgr._portfolios["user1"]["360_SCALP"].peak_balance

        _record_win(mgr)

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.peak_balance > initial_peak

    def test_persistence_save_and_load(self, tmp_path):
        """Portfolios persist to JSON and reload correctly."""
        path = str(tmp_path / "portfolios.json")
        mgr = PaperPortfolioManager(storage_path=path)
        mgr.ensure_user("user1")
        _record_win(mgr)
        saved_balance = mgr._portfolios["user1"]["360_SCALP"].current_balance

        # Reload from disk
        mgr2 = PaperPortfolioManager(storage_path=path)
        assert "user1" in mgr2._portfolios
        assert abs(mgr2._portfolios["user1"]["360_SCALP"].current_balance - saved_balance) < 1e-9
        assert mgr2._portfolios["user1"]["360_SCALP"].win_count == 1

    def test_get_portfolio_summary_format(self, tmp_path):
        """Summary message contains key fields: balance, PnL, fees, leverage."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)

        summary = mgr.get_portfolio_summary("user1")
        assert "Paper Trading Portfolio" in summary
        assert "Balance" in summary
        assert "Leverage" in summary
        assert "Total" in summary
        assert "Fees" in summary

    def test_get_channel_detail_format(self, tmp_path):
        """Channel detail contains trades, win rate, drawdown."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)

        detail = mgr.get_channel_detail("user1", "360_SCALP")
        assert "Portfolio Detail" in detail
        assert "Win Rate" in detail
        assert "Max Drawdown" in detail
        assert "Last 5 Trades" in detail

    def test_get_trade_history_format(self, tmp_path):
        """Trade history shows recent trades with PnL and fees."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)
        _record_loss(mgr)

        history = mgr.get_trade_history("user1")
        assert "Trade History" in history
        assert "BTCUSDT" in history
        assert "PnL" in history
        assert "Fee" in history

    def test_get_trade_history_empty(self, tmp_path):
        """Trade history returns helpful message when no trades exist."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        history = mgr.get_trade_history("user1")
        assert "No trades yet" in history

    def test_record_trade_updates_all_users(self, tmp_path):
        """When a signal completes, all users' portfolios are updated."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        mgr.ensure_user("user2")

        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-multi",
            symbol="ETHUSDT",
            direction="SHORT",
            entry_price=2000.0,
            exit_price=1960.0,
            hit_tp=3,
            hit_sl=False,
            pnl_pct=2.0,
        )

        assert mgr._portfolios["user1"]["360_SCALP"].win_count == 1
        assert mgr._portfolios["user2"]["360_SCALP"].win_count == 1

    def test_record_trade_only_updates_matching_channel(self, tmp_path):
        """A SCALP signal only affects the SCALP portfolio, not SWING."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        initial_swing = mgr._portfolios["user1"]["360_SWING"].current_balance

        _record_win(mgr, channel="360_SCALP")

        swing_balance = mgr._portfolios["user1"]["360_SWING"].current_balance
        assert swing_balance == initial_swing  # SWING should be untouched

    def test_custom_risk_affects_position_size(self, tmp_path):
        """Setting risk to 5% should use 5% of balance per trade."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        mgr.set_risk("user1", "360_SCALP", 5.0)

        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-risk",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=30000.0,
            hit_tp=0,
            hit_sl=False,
            pnl_pct=0.0,  # breakeven so we only see fee impact
        )

        p = mgr._portfolios["user1"]["360_SCALP"]
        # position_size = 1000 * 5% * 1 = 50; fee = 50 * 0.001 * 2 = 0.10
        expected_fee = 1000.0 * 0.05 * 1 * 0.001 * 2
        assert abs(p.total_fees - expected_fee) < 1e-9

    def test_select_channel_ignored(self, tmp_path):
        """Trades for the SELECT channel are silently ignored."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        mgr.record_trade(
            channel="360_GEM",
            signal_id="sig-select",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=30450.0,
            hit_tp=3,
            hit_sl=False,
            pnl_pct=1.5,
        )

        # No portfolio mutation should have happened since SELECT is not tracked
        for ch in PaperPortfolioManager.CHANNELS:
            assert mgr._portfolios["user1"][ch].win_count == 0

    def test_get_channel_detail_unknown_channel(self, tmp_path):
        """Requesting detail for an unknown channel returns an error string."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.get_channel_detail("user1", "360_UNKNOWN")
        assert "❌" in result

    def test_reset_portfolio_unknown_channel(self, tmp_path):
        """Resetting an unknown channel returns an error string."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.reset_portfolio("user1", "360_UNKNOWN")
        assert "❌" in result

    def test_set_leverage_unknown_channel(self, tmp_path):
        """Setting leverage on unknown channel returns error."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.set_leverage("user1", "360_UNKNOWN", 5)
        assert "❌" in result

    def test_set_risk_unknown_channel(self, tmp_path):
        """Setting risk on unknown channel returns error."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.set_risk("user1", "360_UNKNOWN", 3.0)
        assert "❌" in result

    def test_trade_history_single_channel(self, tmp_path):
        """Trade history filtered to a single channel only shows that channel."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr, channel="360_SCALP")
        _record_win(mgr, channel="360_SWING")

        history = mgr.get_trade_history("user1", channel="360_SCALP")
        assert "Trade History" in history
        assert "360_SCALP" in history

    # -----------------------------------------------------------------------
    # Phase 2: Leaderboard
    # -----------------------------------------------------------------------

    def test_leaderboard_empty(self, tmp_path):
        """Leaderboard with no users shows helpful message."""
        mgr = _make_manager(tmp_path)
        result = mgr.get_leaderboard()
        assert "No users registered yet" in result

    def test_leaderboard_ranked_by_pnl(self, tmp_path):
        """Users are ranked by total PnL (default sort)."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("uHigh")  # ≤6 chars → not anonymized
        mgr.ensure_user("uLow")
        # Give uHigh 5x leverage so same trade yields 5x more PnL
        mgr.set_leverage("uHigh", "360_SCALP", 5)
        # Record a win — updates both users, but uHigh has 5x leverage
        _record_win(mgr)
        result = mgr.get_leaderboard(sort_by="pnl")
        assert "Leaderboard" in result
        assert "PnL" in result
        # Verify ordering: uHigh's entry comes before uLow's entry
        assert result.index("uHigh") < result.index("uLow")

    def test_leaderboard_ranked_by_roi(self, tmp_path):
        """Users are ranked by ROI when sort_by='roi'."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)
        result = mgr.get_leaderboard(sort_by="roi")
        assert "Leaderboard" in result
        assert "ROI" in result

    def test_leaderboard_anonymizes_chat_ids(self, tmp_path):
        """Chat IDs longer than 6 chars are partially hidden in leaderboard."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user123456")
        result = mgr.get_leaderboard()
        # Full ID should not appear; truncated form should
        assert "user123456" not in result
        assert "user" in result  # first 4 chars

    def test_leaderboard_shows_medal_emojis(self, tmp_path):
        """Top 3 get 🥇🥈🥉 medals."""
        mgr = _make_manager(tmp_path)
        for uid in ["userA", "userB", "userC"]:
            mgr.ensure_user(uid)
            _record_win(mgr)
        result = mgr.get_leaderboard()
        assert "🥇" in result
        assert "🥈" in result
        assert "🥉" in result

    # -----------------------------------------------------------------------
    # Phase 2: Liquidation
    # -----------------------------------------------------------------------

    def test_liquidation_at_high_leverage(self, tmp_path):
        """At 10x leverage, a -9% loss should trigger liquidation."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        mgr.set_leverage("user1", "360_SCALP", 10)
        # threshold = (100/10)*0.9 = 9%
        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-liq",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=27300.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-9.0,
        )
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.is_liquidated is True
        assert p.liquidation_count == 1

    def test_no_liquidation_at_1x(self, tmp_path):
        """At 1x leverage, even a -90% loss should NOT liquidate."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        # leverage=1 → liquidation check skipped
        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-big-loss",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=3000.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-90.0,
        )
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.is_liquidated is False

    def test_liquidation_zeros_balance(self, tmp_path):
        """After liquidation, balance is exactly 0."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        mgr.set_leverage("user1", "360_SCALP", 10)
        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-liq",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=27300.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-9.0,
        )
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_balance == 0.0

    def test_liquidation_increments_count(self, tmp_path):
        """liquidation_count increases on each liquidation after reset."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        mgr.set_leverage("user1", "360_SCALP", 10)
        for _ in range(2):
            mgr.record_trade(
                channel="360_SCALP",
                signal_id="sig-liq",
                symbol="BTCUSDT",
                direction="LONG",
                entry_price=30000.0,
                exit_price=27300.0,
                hit_tp=0,
                hit_sl=True,
                pnl_pct=-9.0,
            )
            mgr.reset_portfolio("user1", "360_SCALP")
        # After 2 liquidations + 2 resets the lifetime count should be 2
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.liquidation_count == 2

    def test_liquidated_channel_skips_trades(self, tmp_path):
        """After liquidation, new trades are skipped until reset."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        mgr.set_leverage("user1", "360_SCALP", 10)
        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-liq",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=27300.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-9.0,
        )
        # Now try another trade — should be skipped
        _record_win(mgr)
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_balance == 0.0  # still 0

    def test_reset_clears_liquidation(self, tmp_path):
        """Resetting a liquidated channel sets is_liquidated=False."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        mgr.set_leverage("user1", "360_SCALP", 10)
        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-liq",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=27300.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-9.0,
        )
        mgr.reset_portfolio("user1", "360_SCALP")
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.is_liquidated is False
        assert p.current_balance == 1000.0

    def test_liquidation_preserves_count_on_reset(self, tmp_path):
        """Resetting does NOT zero the liquidation_count (lifetime stat)."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        mgr.set_leverage("user1", "360_SCALP", 10)
        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-liq",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=27300.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-9.0,
        )
        mgr.reset_portfolio("user1", "360_SCALP")
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.liquidation_count == 1  # preserved across reset

    # -----------------------------------------------------------------------
    # Phase 2: Partial TP Scaling
    # -----------------------------------------------------------------------

    def test_partial_tp_scaling_tp3(self, tmp_path):
        """With TP prices and hit_tp=3, PnL uses 30/30/40 blended scaling."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        entry = 30000.0
        tp1 = 30300.0  # +1%
        tp2 = 30600.0  # +2%
        tp3 = 31500.0  # +5%
        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-tp3",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=entry,
            exit_price=tp3,
            hit_tp=3,
            hit_sl=False,
            pnl_pct=5.0,
            tp_prices=[tp1, tp2, tp3],
        )
        p = mgr._portfolios["user1"]["360_SCALP"]
        # Blended = 30%*1% + 30%*2% + 40%*5% = 0.3+0.6+2.0 = 2.9% effective
        # Without scaling it would be 5% straight
        assert p.win_count == 1
        # The effective PnL should be less than pure 5% due to blending
        risk_amount = 1000.0 * 0.02
        position = risk_amount * 1
        straight_pnl = position * 0.05  # 5%
        assert p.total_pnl < straight_pnl  # blended is less than straight 5%

    def test_partial_tp_scaling_tp1_then_sl(self, tmp_path):
        """TP1 hit then effectively SL: 30% at TP1 price, 70% at final price."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        entry = 30000.0
        tp1 = 30300.0   # +1%
        sl_exit = 29700.0  # -1%
        # hit_tp=1 means TP1 was hit; final exit at sl_exit price
        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-tp1sl",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=entry,
            exit_price=sl_exit,
            hit_tp=1,
            hit_sl=True,
            pnl_pct=-1.0,  # final signal pnl
            tp_prices=[tp1],
        )
        p = mgr._portfolios["user1"]["360_SCALP"]
        # Blended: 30%*+1% + 70%*(-1%) = 0.3 - 0.7 = -0.4% effective → still a loss
        assert p.loss_count == 1

    def test_partial_tp_no_prices_fallback(self, tmp_path):
        """Without tp_prices, falls back to single-exit PnL calculation."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)  # uses no tp_prices

        p = mgr._portfolios["user1"]["360_SCALP"]
        # Standard 1.5% win: position=20, raw=0.30, fee=0.04, net≈0.26
        risk = 1000.0 * 0.02
        pos = risk * 1
        expected_net = pos * 0.015 - pos * 0.001 * 2
        assert abs(p.total_pnl - expected_net) < 1e-6

    # -----------------------------------------------------------------------
    # Phase 2: Streak Tracking
    # -----------------------------------------------------------------------

    def test_win_streak_increments(self, tmp_path):
        """Consecutive wins increase current_streak."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)
        _record_win(mgr)
        _record_win(mgr)
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_streak == 3

    def test_loss_streak_decrements(self, tmp_path):
        """Consecutive losses decrease current_streak (negative)."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_loss(mgr)
        _record_loss(mgr)
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_streak == -2

    def test_streak_resets_on_opposite(self, tmp_path):
        """A loss after wins resets streak to -1."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)
        _record_win(mgr)
        _record_loss(mgr)
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_streak == -1

    def test_best_win_streak_tracked(self, tmp_path):
        """best_win_streak captures the longest win run."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)
        _record_win(mgr)
        _record_win(mgr)
        _record_loss(mgr)
        _record_win(mgr)
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.best_win_streak == 3

    def test_worst_loss_streak_tracked(self, tmp_path):
        """worst_loss_streak captures the longest loss run (most negative)."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_loss(mgr)
        _record_loss(mgr)
        _record_win(mgr)
        _record_loss(mgr)
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.worst_loss_streak == -2

    def test_breakeven_does_not_break_streak(self, tmp_path):
        """BREAKEVEN trades don't reset the current streak."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)
        _record_win(mgr)
        # Record a breakeven
        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-be",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=30000.0,
            hit_tp=0,
            hit_sl=False,
            pnl_pct=0.0,
        )
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_streak == 2  # streak unchanged by breakeven

    def test_streak_reset_on_portfolio_reset(self, tmp_path):
        """Resetting portfolio zeros all streak counters."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)
        _record_win(mgr)
        mgr.reset_portfolio("user1", "360_SCALP")
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_streak == 0
        assert p.best_win_streak == 0
        assert p.worst_loss_streak == 0

    # -----------------------------------------------------------------------
    # Phase 2: Backward compatibility
    # -----------------------------------------------------------------------

    def test_old_json_loads_without_new_fields(self, tmp_path):
        """A JSON file from Phase 1 (without Phase-2 fields) loads without errors."""
        import json

        path = tmp_path / "old_portfolios.json"
        # Simulate a Phase-1 JSON (no Phase-2 fields)
        old_data = {
            "user1": {
                "360_SCALP": {
                    "channel": "360_SCALP",
                    "initial_balance": 1000.0,
                    "current_balance": 1050.0,
                    "leverage": 1,
                    "risk_per_trade_pct": 2.0,
                    "trades": [],
                    "total_fees": 0.04,
                    "total_pnl": 50.0,
                    "win_count": 3,
                    "loss_count": 1,
                    "breakeven_count": 0,
                    "reset_count": 0,
                    "peak_balance": 1050.0,
                    "max_drawdown_pct": 1.2,
                    # No: liquidation_count, is_liquidated, current_streak,
                    #     best_win_streak, worst_loss_streak
                }
            }
        }
        path.write_text(json.dumps(old_data))
        mgr = PaperPortfolioManager(storage_path=str(path))
        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_balance == 1050.0
        assert p.liquidation_count == 0
        assert p.is_liquidated is False
        assert p.current_streak == 0
        assert p.best_win_streak == 0
        assert p.worst_loss_streak == 0
