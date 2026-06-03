# 🚀 Vanguard V2.3: Elite Trading Engine (NSE)

> [!IMPORTANT]
> **🤖 AI Developer Continuity & Memory Layer Protocol**:
> If you are an AI coding agent (Claude, Gemini, etc.) pair-programming with the user on this repository, you **MUST** read and follow the **[[finalgo-memory-layer/finalgo/AGENTS.md|AGENTS.md operating protocol]]** as your very first action. This imports active development contexts, next steps, and records segregated, compacted updates to ensure continuity.

Welcome to the **Vanguard V2.3** high-precision trading system. This engine is designed for institutional-grade intraday scanning on the NSE (India) market, employing a highly modular hybrid architecture that merges deterministic XGBoost mathematical ranking with a Dual-Stage Gemini LLM Risk Veto.

---

## 🎯 Global Objective
**Project Goal:** Achieve highly constrained, high-win-rate intraday trading (targeting absolute technical confluences) by executing only the top machine-learning candidates that pass a rigorous news and structural AI audit.

---

## 🧠 Core Intelligence Architecture

### 1. The Predictive Layer (XGBoost)
- **Model**: Operates via `v8_upstox_3y` trained on 3 years of Upstox 1-Hour candle data.
- **Features**: Computes an **86-feature vector** per ticker (including microstructural data like IBS, relative volume, volatility bands, and market regimes).
- **Ranking Engine**: Evaluates 170+ liquid NSE stocks cross-sectionally every 15 minutes. Long and Short boosters output a net "Conviction Score".

### 2. Hierarchical Dual-Stage AI Veto
- **Stage 1 (Technical Triage)**: A fast Gemini 3.5-Flash agent structurally validates the trade (evaluating proximity to dynamic support/resistance, RSI, and Bollinger bands) to prevent technical traps.
- **Stage 2 (CRO News Grounding Audit)**: Gemini 2.5-Flash executes Google-grounded web searches for the company to catch block deals, earnings shocks, and SEBI regulatory actions, overriding technicals if fundamental reality conflicts.

### 3. Shadow Tracker & Dynamic Risk Execution
- **Dynamic Risk**: Drops static percentage targets in favor of programmatic brackets based on a 15-minute resampled Average True Range (ATR). Default TP is `3x ATR`, SL is `1.5x ATR`.
- **Active Trailing**: Engages breakeven locks and trailing stop-losses dynamically once profitability buffers are achieved.
- **EOD Flush**: Hard deadline to flush all open trades by **15:15 IST**.

---

## 🚀 Quick Start Operator Guide

### 1. Environment Setup
The system requires Python 3.10+ and a CUDA-capable GPU (optional but recommended for XGBoost inference).

```powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install strict dependencies
pip install -r requirements.txt
```

### 2. Configuration & API Tokens
You must configure your `.env` file in the root directory before booting the engine:
```env
# Upstox API (For Live execution and Data Fetching)
UPSTOX_API_KEY="your_api_key"
UPSTOX_API_SECRET="your_api_secret"

# Gemini LLM Keys (Comma-separated for auto-rotation)
GEMINI_API_KEYS="key1,key2,key3"
BACKUP_GEMINI_API_KEY="emergency_key"
```

### 3. System Activation

The codebase is highly modular under `scripts/vanguard/`. 

**Boot the Live Trading Engine:**
```powershell
.\run_vanguard_system.bat
# Or manually via python:
python scripts/vanguard_signal_engine.py
```

**Launch the Real-Time Dashboard:**
```powershell
python scripts/vanguard_dashboard.py
```
*Navigate to `http://127.0.0.1:5000` to monitor live shadow trades, portfolio P&L, and Veto diagnostics.*

---

## 🗄️ Core Directory Map
- **`scripts/vanguard/`**: The core modular trading logic (`orchestrator.py`, `model_inference.py`, `ai_veto.py`, `trade_state.py`).
- **`scripts/backtests/`**: Contains regime-aware and multi-timeframe strategy simulators (e.g., `strategy_50_regime_backtest.py`).
- **`scripts/training/`**: Model generation using strict walk-forward temporal cross-validation (`train_ranking.py`, `walk_forward_validation.py`).
- **`finalgo-memory-layer/`**: The Obsidian vault storing all architectural schematics, model registry documentation, and active context logs.

---

## 🛡️ Operational Safety Caveats
- **Timezone Sync**: All timestamps are natively handled in **IST (UTC+5:30)**. Ensure your host machine clock is synced.
- **Sandbox Mode**: `SANDBOX_MODE` in `scripts/vanguard/config.py` defaults to `True`. Keep this active to run paper-trades in the SQLite ledger before linking real Upstox funds.
- **API Limits**: The engine heavily caches Upstox and Yahoo Finance queries to `data/`. If you hit rate limits, check the `GeminiRateTracker` logs.
