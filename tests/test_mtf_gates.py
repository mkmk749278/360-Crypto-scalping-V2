from src.mtf import (
    check_mtf_ema_alignment,
    check_mtf_rsi,
    check_mtf_adx,
    mtf_gate_scalp_standard,
    mtf_gate_scalp_range_fade,
    mtf_gate_swing,
)


def test_mtf_ema_aligned_long():
    """EMA fast > slow with EMA200 below → aligned for LONG."""
    ind = {"ema9_last": 105.0, "ema21_last": 100.0, "ema200_last": 95.0}
    ok, reason, adj = check_mtf_ema_alignment(ind, "LONG", strict=True)
    assert ok
    assert reason == "mtf_ema_aligned"
    assert adj == 0.0


def test_mtf_ema_opposed_long_strict():
    """EMA fast < slow → opposed LONG, strict mode rejects."""
    ind = {"ema9_last": 98.0, "ema21_last": 100.0}
    ok, reason, adj = check_mtf_ema_alignment(ind, "LONG", strict=True)
    assert not ok
    assert reason == "mtf_ema_opposed"


def test_mtf_ema_opposed_long_soft():
    """EMA fast < slow → opposed LONG, soft mode applies penalty."""
    ind = {"ema9_last": 98.0, "ema21_last": 100.0}
    ok, reason, adj = check_mtf_ema_alignment(ind, "LONG", strict=False)
    assert ok
    assert adj == -10.0


def test_mtf_ema_no_data():
    """Missing EMA data → fail open."""
    ok, reason, adj = check_mtf_ema_alignment({}, "LONG")
    assert ok
    assert reason == "mtf_ema_no_data"


def test_mtf_rsi_overbought_blocks_long():
    ind = {"rsi_last": 75.0}
    ok, reason, adj = check_mtf_rsi(ind, "LONG")
    assert not ok


def test_mtf_rsi_oversold_blocks_short():
    ind = {"rsi_last": 25.0}
    ok, reason, adj = check_mtf_rsi(ind, "SHORT")
    assert not ok


def test_mtf_rsi_normal_passes():
    ind = {"rsi_last": 50.0}
    ok, reason, adj = check_mtf_rsi(ind, "LONG")
    assert ok


def test_mtf_adx_within_range():
    ind = {"adx_last": 30.0}
    ok, reason, adj = check_mtf_adx(ind)
    assert ok


def test_mtf_adx_too_weak():
    ind = {"adx_last": 10.0}
    ok, reason, adj = check_mtf_adx(ind)
    assert not ok


def test_mtf_adx_too_extreme():
    ind = {"adx_last": 80.0}
    ok, reason, adj = check_mtf_adx(ind)
    assert not ok


def test_scalp_standard_gate_trending_hard_reject():
    """Both EMA and RSI fail in TRENDING regime → hard reject."""
    ind_1h = {"ema9_last": 95.0, "ema21_last": 100.0, "rsi_last": 75.0}
    ok, reason, adj = mtf_gate_scalp_standard(ind_1h, "LONG", regime="TRENDING_UP")
    assert not ok


def test_scalp_standard_gate_ranging_soft_fail():
    """Both fail in RANGING regime → soft penalty."""
    ind_1h = {"ema9_last": 95.0, "ema21_last": 100.0, "rsi_last": 75.0}
    ok, reason, adj = mtf_gate_scalp_standard(ind_1h, "LONG", regime="RANGING")
    assert ok
    assert adj == -10.0


def test_scalp_standard_gate_partial():
    """EMA aligned but RSI overbought → partial penalty."""
    ind_1h = {"ema9_last": 105.0, "ema21_last": 100.0, "rsi_last": 75.0}
    ok, reason, adj = mtf_gate_scalp_standard(ind_1h, "LONG", regime="TRENDING_UP")
    assert ok
    assert adj == -5.0


def test_range_fade_15m_rsi_oversold_passes_long():
    ind = {"rsi_last": 35.0}
    ok, reason, adj = mtf_gate_scalp_range_fade(ind, "LONG")
    assert ok


def test_range_fade_15m_rsi_not_oversold_blocks_long():
    ind = {"rsi_last": 55.0}
    ok, reason, adj = mtf_gate_scalp_range_fade(ind, "LONG")
    assert not ok


def test_swing_gate_all_ok():
    ind_4h = {"ema9_last": 110.0, "ema21_last": 100.0, "adx_last": 30.0}
    ok, reason, adj = mtf_gate_swing(ind_4h, "LONG")
    assert ok
    assert adj == 0.0


def test_swing_gate_ema_opposed_rejects():
    ind_4h = {"ema9_last": 95.0, "ema21_last": 100.0, "adx_last": 30.0}
    ok, reason, adj = mtf_gate_swing(ind_4h, "LONG")
    assert not ok


def test_swing_gate_adx_fail_soft_penalty():
    ind_4h = {"ema9_last": 110.0, "ema21_last": 100.0, "adx_last": 10.0}
    ok, reason, adj = mtf_gate_swing(ind_4h, "LONG")
    assert ok
    assert adj == -5.0
