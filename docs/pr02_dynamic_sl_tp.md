# PR 02: Dynamic SL/TP System

**Objective:** Replace static SL/TP with adaptive SL/TP per pair & regime.

**Description:**
- SL/TP adjusted based on pair volatility, spread, and historical success.
- Regime-aware: QUIET → wider SL, TRENDING → tighter SL.
- Improves scalp signal survival and reduces SL hits.

**Files to Update:**
- trade_monitor/sl_tp.py (update calculation)
- utils/volatility_metrics.py (new helper functions)

**Implementation Steps:**
1. Replace fixed SL/TP values with calculate_dynamic_sl_tp(pair, regime, volatility, hit_rate).
2. Update trade execution module to use dynamic SL/TP.
3. Add logging for dynamically assigned SL/TP.