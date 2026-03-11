# 360-Crypto-Eye-Scalping — Ultimate Institutional AI Signal Engine

An asynchronous Python crypto signal engine that detects **Smart Money Concepts (SMC)** via Binance WebSockets and REST APIs, integrates AI-driven insights (news sentiment, social sentiment, whale flows), calculates dynamic confidence scores (0–100), and routes high-confidence signals to **4 specialized Telegram channels**.

---

## Architecture

```
Binance REST ──► PairManager (top 50–100 pairs, 6–12 h refresh)
                       │
                       ▼
              HistoricalDataStore (OHLCV seed per timeframe)
                       │
Binance WS ──► WebSocketManager (multi-conn, heartbeat, auto-reconnect)
                       │
                       ▼
                  Scanner Loop
           ┌───────────┤────────────┐
     Indicators    SMC Algos    AI Engine
           └───────────┤────────────┘
                       ▼
                 Channel Strategies
         (SCALP · SWING · RANGE · THE_TAPE)
                       │
               ConfidenceScorer (0–100)
                       │
                 asyncio.Queue
                       │
                       ▼
                 SignalRouter ──► Telegram Channels
                       │             ├─ ⚡ 360_SCALP
                       │             ├─ 🏛️ 360_SWING
                       │             ├─ ⚖️ 360_RANGE
                       │             ├─ 🐋 360_THE_TAPE
                       │             └─ 🆓 Free Channel
                       ▼
                 TradeMonitor
           (TP/SL · Trailing · Updates)
```

## Features

| Feature | Description |
|---|---|
| **SMC Detection** | Liquidity Sweeps, Market Structure Shifts (MSS), Fair Value Gaps (FVG) |
| **4 Channels** | SCALP (M1/M5), SWING (H1/H4), RANGE (M15), THE_TAPE (Tick) |
| **AI Modules** | News sentiment, social sentiment, whale detection |
| **Confidence Scoring** | Multi-layer 0–100 with 7 sub-components |
| **Dynamic Pairs** | Auto-fetch top 50–100 Spot & Futures pairs |
| **WebSocket Resilience** | Multi-connection, heartbeat, exponential-backoff reconnect |
| **Trade Monitoring** | Real-time TP/SL tracking, trailing stops, PnL updates |
| **Free/Premium** | Top 1–2 daily signals to free channel |
| **Telemetry** | CPU, memory, WS health, scan latency, API usage |
| **Admin Commands** | `/view_dashboard`, `/update_pairs`, `/subscribe_alerts` |

## Channel Details

### ⚡ 360_SCALP — M1/M5 High-Frequency Scalping
- **Trigger**: M5 Liquidity Sweep + Momentum > 0.3% over 3 candles
- **Filters**: EMA alignment, ADX > 25, ATR-based volatility, spread < 0.02%
- **Risk**: SL 0.05–0.1%, TP1 1R, TP2 1.5R, TP3 2R, Trailing 1.5×ATR

### 🏛️ 360_SWING — H1/H4 Institutional Swing
- **Trigger**: H4 ERL Sweep + H1 MSS
- **Filters**: EMA200, Bollinger rejection, ADX 20–40, spread < 0.02%
- **Risk**: SL 0.2–0.5%, TP1 1.5R, TP2 3R, TP3 5R, Trailing 2.5×ATR

### ⚖️ 360_RANGE — M15 Mean-Reversion
- **Trigger**: ADX < 20 + Bollinger Band rejection
- **Filters**: SMA trend, RSI mean-reversion, ATR volatility
- **Risk**: SL 0.1–0.2%, TP1 1R, TP2 1.5R, Trailing 1×ATR

### 🐋 360_THE_TAPE — Tick/Data Whale Tracking
- **Trigger**: Trade > 1M USD or Volume Delta > 2×
- **Filters**: Order book imbalance, whale detection, AI sentiment
- **Risk**: SL 0.1–0.3% AI-adaptive, Trailing AI-adaptive

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Telegram bot token and channel IDs

# 3. Run the engine
python -m src.main

# 4. Run tests
python -m pytest tests/ -v
```

## Project Structure

```
config/
  __init__.py          # All settings, channel configs, constants
src/
  main.py              # Orchestrator & entry point
  indicators.py        # EMA, SMA, ADX, ATR, RSI, Bollinger, Momentum
  smc.py               # Liquidity Sweep, MSS, FVG detection
  confidence.py        # Multi-layer confidence scorer (0–100)
  ai_engine.py         # News/social sentiment, whale detection
  pair_manager.py      # Dynamic pair management (Binance REST)
  historical_data.py   # OHLCV & tick seeding
  websocket_manager.py # Multi-connection WS with resilience
  signal_router.py     # Queue-based signal dispatch
  trade_monitor.py     # TP/SL/trailing real-time monitoring
  telegram_bot.py      # Rich signal formatting & admin commands
  telemetry.py         # System health monitoring
  utils.py             # Logging, formatting helpers
  channels/
    base.py            # Signal model & base strategy
    scalp.py           # 360_SCALP strategy
    swing.py           # 360_SWING strategy
    range_channel.py   # 360_RANGE strategy
    tape.py            # 360_THE_TAPE strategy
tests/
  test_indicators.py
  test_smc.py
  test_confidence.py
  test_channels.py
  test_signal_router.py
  test_telegram_format.py
```

## Signal Format Example

```
⚡ 360_SCALP ALERT 💎
Pair: BTCUSDT
📈 LONG 🚀
🚀 Entry: 32,150
🛡️ SL: 32,120
🎯 TP1: 32,200 ✅
🎯 TP2: 32,300
🎯 TP3: 32,400
💹 Trailing Active (1.5×ATR)
🤖 Confidence: 87%
📰 AI Sentiment: Positive — Whale Activity
⚠️ Risk: Aggressive
⏰ Time: 2026-03-11 12:34:22
```