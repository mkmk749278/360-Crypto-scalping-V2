# 360-Crypto-Eye-Scalping — Ultimate Institutional AI Signal Engine

An asynchronous Python crypto signal engine that detects **Smart Money Concepts (SMC)** via Binance WebSockets and REST APIs, integrates AI-driven insights (news sentiment, social sentiment, whale flows), calculates dynamic confidence scores (0–100), and routes high-confidence signals to **4 specialized Telegram channels** (SCALP, SWING, SPOT, GEM).

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
         (SCALP · SWING · SPOT · GEM)
                       │
               ConfidenceScorer (0–100)
                       │
                 asyncio.Queue
                       │
                       ▼
                 SignalRouter ──► Telegram Channels
                       │             ├─ ⚡ 360_SCALP
                       │             ├─ 🏛️ 360_SWING
                       │             ├─ 📈 360_SPOT
                       │             ├─ 💎 360_GEM
                       │             └─ 🆓 Free Channel
                       ▼
                 TradeMonitor
           (TP/SL · Trailing · Updates)
                       │
           ┌───────────┼───────────────┐
     PerformanceTracker          CircuitBreaker
     (real stats)                (auto-pause)
                       │
                 SelectModeFilter
           (premium 360_SELECT channel)
```

## Features

| Feature | Module | Description |
|---|---|---|
| **SMC Detection** | `src/smc.py` | Liquidity Sweeps, Market Structure Shifts (MSS), Fair Value Gaps (FVG) |
| **4 Channels** | `src/channels/` | SCALP (M1/M5), SWING (H1/H4), SPOT (H4/D1), GEM (D1/W1) |
| **AI Modules** | `src/ai_engine.py` | CryptoPanic news sentiment, LunarCrush social sentiment, Alternative.me Fear & Greed, whale detection |
| **Confidence Scoring** | `src/confidence.py` | Multi-layer 0–100 with 7 sub-components |
| **Dynamic Pairs** | `src/pair_manager.py` | Auto-fetch top 50–100 Spot & Futures pairs |
| **WebSocket Resilience** | `src/websocket_manager.py` | Multi-connection, heartbeat, exponential-backoff reconnect |
| **Trade Monitoring** | `src/trade_monitor.py` | Real-time TP/SL tracking, trailing stops, PnL updates |
| **Free/Premium** | `src/signal_router.py` | Top 1–2 daily signals to free channel |
| **Telemetry** | `src/telemetry.py` | CPU, memory, WS health, scan latency, API usage |
| **Admin Commands** | `src/commands.py` | Full suite of admin and user Telegram commands |
| **Performance Tracker** | `src/performance_tracker.py` | Tracks real signal outcomes per channel — win rates, TP hit rates, signal quality scoring. Provides `/stats`, `/signal_stats`, `/tp_stats` commands. |
| **Backtester** | `src/backtester.py` | Full backtesting engine that runs channel strategies against historical candle data. Configurable fee, slippage, lookahead candles, min window. Supports single-symbol and multi-symbol aggregate backtests. |
| **Circuit Breaker** | `src/circuit_breaker.py` | Auto-pauses signal generation after consecutive losses (rolling window). Prevents compounding drawdowns. Admin can check status and manually reset. |
| **Gem Scanner (360_GEM)** | `src/gem_scanner.py` | Independent macro scanner that finds deeply discounted altcoins showing early reversal signals — potential x10 tokens. Configurable via admin commands. |
| **Signal Quality Scoring** | `src/signal_quality.py` | Advanced multi-factor signal quality analysis beyond basic confidence. |
| **Predictive AI** | `src/predictive_ai.py` | ML-based price direction prediction using historical patterns. |
| **OpenAI Evaluator** | `src/openai_evaluator.py` | GPT-powered signal evaluation for natural language trade rationale. |
| **On-Chain Analysis** | `src/onchain.py` | On-chain data integration (whale movements, exchange flows). |
| **Market Regime Detection** | `src/regime.py` | Classifies current market regime (trending, ranging, volatile) to adapt strategy behavior. |
| **Cross-Pair Correlation** | `src/correlation.py` | Analyzes correlation between pairs to avoid overexposure and improve diversification. |
| **DCA Engine** | `src/dca.py` | Dollar-cost averaging module for SPOT and SWING channel DCA entries. |
| **Performance Metrics** | `src/performance_metrics.py` | Sharpe ratio, Sortino ratio, and other quantitative metrics. |
| **Exchange Abstraction** | `src/exchange.py` | Unified exchange interface abstracting Binance-specific API calls. |
| **Redis Caching** | `src/redis_client.py` | Optional Redis-backed caching layer for signal state and AI results. |
| **State Cache** | `src/state_cache.py` | In-memory state caching for scanner loop efficiency. |
| **Risk Manager** | `src/risk.py` | Position sizing, stop-loss calculation, and risk-per-trade enforcement. |
| **Signal Filters** | `src/filters.py` | Pre-signal filters (spread, volume, volatility checks). |
| **Detector** | `src/detector.py` | Pattern detection utilities used by the scanner. |

## Channel Details

### ⚡ 360_SCALP — M1/M5 High-Frequency Scalping
- **Trigger**: M5 Liquidity Sweep + Momentum > 0.3% over 3 candles
- **Filters**: EMA alignment, ADX > 25, ATR-based volatility, spread < 0.02%
- **Risk**: SL 0.05–0.1%, TP1 1R, TP2 1.5R, TP3 2R, Trailing 1.5×ATR

### 🏛️ 360_SWING — H1/H4 Institutional Swing
- **Trigger**: H4 ERL Sweep + H1 MSS
- **Filters**: EMA200, Bollinger rejection, ADX 20–40, spread < 0.02%
- **Risk**: SL 0.2–0.5%, TP1 1.5R, TP2 3R, TP3 5R, Trailing 2.5×ATR

### 📈 360_SPOT — H4/D1 Spot DCA Accumulation
- **Trigger**: H4/D1 structure + DCA zone detection
- **Filters**: ADX 0–100, spread < 0.02%, volume > $1M
- **Risk**: SL 0.5–2%, TP1 2R, TP2 5R, TP3 10R, Trailing 3×ATR
- **DCA**: Enabled (60/40 weight split, zone 30–70% of SL distance, min momentum 0.2)

### 💎 360_GEM — D1/W1 Macro Reversal Scanner
- **Trigger**: Deeply discounted altcoins showing early reversal signals
- **Filters**: ATH drawdown analysis, accumulation base detection, volume surge
- **Risk**: SL 10–30%, TP1 2R, TP2 5R, TP3 10R, Trailing 3×ATR
- **Target**: Previous ATH region — potential x10+ returns

## Telegram Commands Reference

### Admin Commands

| Command | Description |
|---|---|
| `/view_dashboard` | Show telemetry dashboard |
| `/update_pairs [spot/futures] [n]` | Refresh trading pairs |
| `/subscribe_alerts` | Subscribe to admin alerts |
| `/view_pairs [spot/futures]` | View active pairs (top 10 by volume) |
| `/force_scan` | Trigger immediate scanner cycle |
| `/pause_channel <name>` | Pause a channel |
| `/resume_channel <name>` | Resume a paused channel |
| `/set_confidence_threshold <channel> <value>` | Override confidence threshold |
| `/engine_status` (alias: `/status`) | Show uptime, tasks, queue, WS health |
| `/memory_usage` | Show RSS, VMS, CPU usage |
| `/set_free_channel_limit <n>` | Set daily free signal limit |
| `/force_update_ai` | Refresh AI/sentiment cache |
| `/view_active_signals` | List all active signals with full detail |
| `/view_logs [lines]` | View recent log file (1–200 lines) |
| `/update_code` | Run `git pull` on the server |
| `/restart_engine` | Restart all engine tasks |
| `/rollback_code <commit>` | Checkout a specific commit |
| `/circuit_breaker_status` | Show circuit breaker state |
| `/reset_circuit_breaker` | Reset circuit breaker |
| `/stats [channel]` | Show performance stats |
| `/real_stats [channel]` | Show real signal stats |
| `/reset_stats [channel]` | Clear performance records |
| `/gem_mode [on/off/status]` | Toggle 360_GEM gem scanner channel |
| `/gem_config <key> <value>` | Update gem scanner configuration |
| `/backtest <symbol> [channel] [lookahead]` | Run backtest for a single symbol |
| `/backtest_all [channel] [lookahead]` | Run backtest across all tracked symbols |
| `/backtest_config [key] [value]` | View or update backtest parameters (fee, slippage, lookahead, min_window) |

### User Commands

| Command | Description |
|---|---|
| `/signals` | Show last 5 active signals |
| `/free_signals` | Show today's free channel signals |
| `/signal_info <id>` | Detailed info on a specific signal |
| `/last_update` | Show last scan latency, pair count, active signal count |
| `/subscribe` | Subscribe to premium signals |
| `/unsubscribe` | Unsubscribe from premium signals |
| `/signal_history` | Show last 10 completed signals |
| `/signal_stats [channel]` | Show signal quality stats per channel |
| `/tp_stats [channel]` | Show TP hit rates per channel |

## AI Sentiment APIs

The engine integrates three external APIs to power the `ai_sentiment_score` component (0–15 points) of the confidence scorer.  All APIs have free tiers and degrade gracefully if a key is absent.

| Service | Data | Key required | Cache TTL |
|---|---|---|---|
| **CryptoPanic** | News sentiment (bullish/bearish/neutral counts) | Yes — [get free key](https://cryptopanic.com/developers/api/) | 60 s |
| **LunarCrush** | Social sentiment & galaxy score (0–100, normalised to ±1) | Yes — [get free key](https://lunarcrush.com/developers) | 300 s |
| **Alternative.me** | Bitcoin Fear & Greed Index (0–100) | No — completely free | 3600 s |

### Configuration

Set the following environment variables in your `.env` file:

```env
# CryptoPanic (News Sentiment) - Get free key at https://cryptopanic.com/developers/api/
NEWS_API_KEY=your_cryptopanic_token

# LunarCrush (Social Sentiment) - Get free key at https://lunarcrush.com/developers
SOCIAL_SENTIMENT_API_KEY=your_lunarcrush_bearer_token

# Fear & Greed Index (free, no key needed)
FEAR_GREED_API_URL=https://api.alternative.me/fng/?limit=1
```

If no keys are set, all sentiment functions return a neutral score of 0.0 — the bot runs at up to 85/100 confidence maximum instead of 100/100.

## Quick VPS Deployment

One-line fresh VPS deployment (Ubuntu 20.04 / 22.04 / 24.04).  
Requires Docker — install it first if needed:

```bash
# Install Docker (if not already installed)
curl -fsSL https://get.docker.com | sh
```

Then deploy the engine:

```bash
# 1. Clone the repo
git clone https://github.com/mkmk749278/360-Crypto-scalping-V2.git
cd 360-Crypto-scalping-V2

# 2. Configure credentials
cp .env.example .env
nano .env  # Set your TELEGRAM_BOT_TOKEN and channel IDs

# 3. Deploy with Docker
chmod +x deploy.sh
bash deploy.sh
```

### Service Management

```bash
# Follow live logs
docker compose logs -f engine

# Restart the engine
docker compose restart engine

# Stop the engine
docker compose down

# Rebuild and restart after code changes
docker compose up -d --build

# Check service status
docker compose ps
```

---

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your Telegram bot token and channel IDs

# 2. Start with Docker
docker compose up -d

# 3. Follow logs
docker compose logs -f engine

# 4. Run tests (local)
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Project Structure

```
config/
  __init__.py          # All settings, channel configs, constants
src/
  main.py              # Orchestrator & entry point
  bootstrap.py         # Dependency wiring and initialisation
  indicators.py        # EMA, SMA, ADX, ATR, RSI, Bollinger, Momentum
  smc.py               # Liquidity Sweep, MSS, FVG detection
  confidence.py        # Multi-layer confidence scorer (0–100)
  ai_engine.py         # News/social sentiment, whale detection
  pair_manager.py      # Dynamic pair management (Binance REST)
  historical_data.py   # OHLCV & tick seeding
  websocket_manager.py # Multi-connection WS with resilience
  signal_router.py     # Queue-based signal dispatch
  signal_queue.py      # Async signal queue with Redis fallback
  trade_monitor.py     # TP/SL/trailing real-time monitoring
  telegram_bot.py      # Rich signal formatting & admin commands
  commands.py          # Full Telegram command handler routing
  telemetry.py         # System health monitoring
  scanner.py           # Core scanner loop
  detector.py          # Pattern detection utilities
  filters.py           # Pre-signal filters (spread, volume, volatility)
  risk.py              # Position sizing, SL calculation, risk enforcement
  exchange.py          # Unified exchange interface (Binance abstraction)
  binance.py           # Binance REST/WS API client
  logger.py            # Centralised logging configuration
  utils.py             # Logging, formatting helpers
  performance_tracker.py # Real signal outcome tracking per channel
  performance_metrics.py # Sharpe/Sortino ratio and quantitative metrics
  backtester.py        # Historical backtest engine
  circuit_breaker.py   # Auto-pause after consecutive losses
  gem_scanner.py       # 360_GEM macro-reversal gem scanner
  signal_quality.py    # Advanced multi-factor signal quality scoring
  predictive_ai.py     # ML-based price direction prediction
  openai_evaluator.py  # GPT-powered signal evaluation
  onchain.py           # On-chain data integration (whale/exchange flows)
  regime.py            # Market regime classification
  correlation.py       # Cross-pair correlation analysis
  dca.py               # DCA engine for SPOT and SWING channel entries
  redis_client.py      # Optional Redis caching layer
  state_cache.py       # In-memory state caching for scanner loop
  channels/
    base.py            # Signal model & base strategy
    scalp.py           # 360_SCALP strategy
    swing.py           # 360_SWING strategy
    spot.py            # 360_SPOT strategy
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